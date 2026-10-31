#include "kvmem/source_archive.hpp"
#include "memory_test_support.hpp"

#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <future>
#include <mutex>
#include <new>
#include <numeric>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

namespace allocation {
thread_local size_t fail_at = SIZE_MAX, calls = 0;
void check() {
    if (fail_at != SIZE_MAX && calls++ == fail_at) { fail_at = SIZE_MAX; throw std::bad_alloc(); }
}
}
void * operator new(size_t n) {
    allocation::check();
    if (auto p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void * operator new[](size_t n) { return ::operator new(n); }
void operator delete(void * p) noexcept { std::free(p); }
void operator delete[](void * p) noexcept { std::free(p); }
void operator delete(void * p, size_t) noexcept { std::free(p); }
void operator delete[](void * p, size_t) noexcept { std::free(p); }

// Instrument the actual product translation unit's OS calls, without a public
// test hook or a fake persistence implementation. Error injection is one-shot.
namespace fault {
enum Mode { Off, Count, Error, Crash };
enum Kind { Write, Sync, Rename, Unlink };
std::atomic<int> mode{Off};
std::atomic<size_t> calls{0}, target{SIZE_MAX};
std::atomic<bool> fired{false};
std::atomic<bool> writes{true};
size_t before(Kind kind) {
    if (mode.load() == Off || (kind == Write && !writes.load()) || kind == Unlink) return SIZE_MAX;
    auto step = calls.fetch_add(1);
    if (step == target.load() && mode.load() == Error) { fired = true; errno = kind == Write ? ENOSPC : EIO; }
    return step;
}
bool fails(size_t step) { return step != SIZE_MAX && step == target.load() && mode.load() == Error; }
void after(size_t step, bool success) {
    if (success && step != SIZE_MAX && step == target.load() && mode.load() == Crash) ::_exit(71);
}
struct Scope {
    Scope(Mode selected, size_t step = SIZE_MAX, bool include_writes = true) {
        calls = 0; target = step; fired = false; writes = include_writes; mode = selected;
    }
    ~Scope() { mode = Off; }
};
}
extern "C" {
ssize_t __real_write(int, const void *, size_t);
int __real_fsync(int);
int __real_renameat(int, const char *, int, const char *);
ssize_t __wrap_write(int fd, const void * p, size_t n) {
    auto step = fault::before(fault::Write);
    if (fault::fails(step)) return -1;
    auto result = __real_write(fd, p, n); fault::after(step, result >= 0); return result;
}
int __wrap_fsync(int fd) {
    auto step = fault::before(fault::Sync);
    if (fault::fails(step)) return -1;
    auto result = __real_fsync(fd); fault::after(step, result == 0); return result;
}
int __wrap_renameat(int a, const char * b, int c, const char * d) {
    auto step = fault::before(fault::Rename);
    if (fault::fails(step)) return -1;
    auto result = __real_renameat(a, b, c, d); fault::after(step, result == 0); return result;
}
}

using namespace kvmem;
namespace {
SourceConfiguration config() { SourceConfiguration c{}; c[0] = 43; c[31] = 201; return c; }
SourceArchiveLimits limits() { return {1<<20, 1<<20, 4, 12, {1<<16, 8192, 256}}; }
struct Temp {
    std::filesystem::path root;
    Temp() {
        char name[] = "/tmp/kvmem-source-archive-XXXXXX";
        auto made = ::mkdtemp(name); CHECK(made); root = made;
    }
    ~Temp() { std::error_code ec; std::filesystem::remove_all(root, ec); }
    std::filesystem::path path(const char * name = "archive") const { return root/name; }
};
struct Fixture {
    SourceJournalStore store{{1<<20, 8192, 256, 8}};
    HostKvStorage staging{1<<20};
    std::vector<int32_t> rows;
    SourceRef ref;
    Fixture() : rows(513) {
        std::iota(rows.begin(), rows.end(), 0);
        const SourceEvent events[] = {{1,0,64,SourceEventKind::System,true}, {2,64,513,SourceEventKind::User,true}};
        ref = store.capture(config(), rows.data(), rows.size(), events, 2, 512);
    }
    void append() { const int32_t token = 514; ref = store.append(ref, &token, 1, nullptr, 0, 513); rows.push_back(token); }
};
void rows_match(const SourceImage & image, const std::vector<int32_t> & rows) {
    std::vector<int32_t> actual(rows.size()); image.read_tokens(0, actual.data(), actual.size()); CHECK(actual == rows);
}
template<class F> void corrupt(F action) {
    bool caught = false;
    try { action(); } catch (const SnapshotCorrupt &) { caught = true; }
    CHECK(caught);
}
template<class F> void io_failure(F action) {
    bool caught = false;
    try { action(); } catch (const std::system_error &) { caught = true; }
    CHECK(caught);
}
std::vector<uint8_t> contents(const std::filesystem::path & path) {
    std::ifstream input(path, std::ios::binary); CHECK(input.good());
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(input), {});
}
void replace(const std::filesystem::path & path, const std::vector<uint8_t> & bytes) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char *>(bytes.data()), bytes.size()); CHECK(output.good());
}
void copy_archive(const std::filesystem::path & source, const std::filesystem::path & target) {
    std::filesystem::copy(source, target, std::filesystem::copy_options::recursive);
}
struct Gate {
    std::promise<void> entered, released;
    std::shared_future<void> exit = released.get_future().share();
    StorageTicket occupy(StorageIoQueue & queue) {
        auto ticket = queue.submit(1, [this] { entered.set_value(); exit.wait(); });
        entered.get_future().wait(); return ticket;
    }
};

void roundtrip_reopen_and_lock() {
    Temp temp; Fixture f; SourceArchiveKey key;
    {
        StorageIoQueue queue(2, 1<<20);
        auto archive = SourceArchive::create(temp.path(), config(), limits());
        CHECK(archive.stats().healthy && archive.stats().sources == 0);
        rejects(MemoryErrorCode::Pending, [&] { (void)SourceArchive::open(temp.path(), config(), limits()); });
        io_failure([&] { (void)SourceArchive::create(temp.path(), config(), limits()); });
        key = archive.enroll(f.store, f.ref); CHECK(key.id == 1);
        CHECK(archive.enroll(f.store, f.ref) == key);
        rejects(MemoryErrorCode::Stale, [&] { (void)archive.read(key, f.staging); });
        auto write = archive.save(queue, f.store, f.ref, f.staging); write.wait();
        CHECK(f.staging.charged_bytes() == 0); archive.publish(write, f.store);
        archive.publish(write, f.store); // idempotent same receipt
        auto read = archive.read(key, f.staging); archive.validate(read); rows_match(read.image(), f.rows);
        CHECK(read.key() == key && read.image().origin().computed_frontier == 512);
        CHECK(archive.stats().object_bytes == 144+4*513+26*2);
        CHECK(archive.stats().charged <= archive.stats().capacity);
        HostKvStorage short_staging(1);
        rejects(MemoryErrorCode::BudgetExceeded, [&] { (void)archive.read(key, short_staging); });
        CHECK(short_staging.charged_bytes() == 0);
        auto wrong = config(); wrong[3] = 7;
        // With a live owner, lock rejection takes precedence over content access.
        rejects(MemoryErrorCode::Pending, [&] { (void)SourceArchive::open(temp.path(), wrong, limits()); });
    }
    CHECK(f.staging.charged_bytes() == 0);
    {
        auto archive = SourceArchive::open(temp.path(), config(), limits());
        auto read = archive.read(key, f.staging); archive.validate(read); rows_match(read.image(), f.rows);
        CHECK(archive.stats().sources == 1 && archive.stats().objects == 1);
        // SourceRefs do not automatically regain an owner binding after restart.
        StorageIoQueue queue(1, 1<<20);
        rejects(MemoryErrorCode::Stale, [&] { (void)archive.save(queue, f.store, f.ref, f.staging); });
    }
    auto wrong = config(); wrong[3] = 7;
    rejects(MemoryErrorCode::InvalidPlan, [&] { (void)SourceArchive::open(temp.path(), wrong, limits()); });
    io_failure([&] { (void)SourceArchive::open(temp.path("absent"), config(), limits()); });
}
void versions_and_reader_retention() {
    Temp temp; Fixture f; StorageIoQueue queue(4, 1<<20);
    auto archive = SourceArchive::create(temp.path(), config(), limits()); auto key = archive.enroll(f.store, f.ref);
    auto first = archive.save(queue, f.store, f.ref, f.staging); first.wait();
    f.append(); archive.publish(first, f.store); // immutable earlier prefix is allowed
    std::optional<SourceArchiveRead> old_read(archive.read(key, f.staging)); archive.validate(*old_read);
    CHECK(old_read->image().origin().accepted_frontier == 513);
    auto newer = archive.save(queue, f.store, f.ref, f.staging); newer.wait(); archive.publish(newer, f.store);
    rejects(MemoryErrorCode::Stale, [&] { archive.validate(*old_read); });
    rejects(MemoryErrorCode::Stale, [&] { archive.publish(first, f.store); });
    const auto old_bytes = old_read->image().image_bytes();
    first = {}; CHECK(archive.collect().quota_released == 0); // reader still owns disk
    old_read.reset(); const auto old_cleanup = archive.collect();
    CHECK(old_cleanup.quota_released == old_bytes && old_cleanup.file_bytes_removed == old_bytes);
    auto latest = archive.read(key, f.staging); rows_match(latest.image(), f.rows);
    auto same_revision = archive.save(queue, f.store, f.ref, f.staging); same_revision.wait();
    archive.publish(same_revision, f.store);
    rejects(MemoryErrorCode::Stale, [&] { archive.publish(newer, f.store); });
    rejects(MemoryErrorCode::Stale, [&] { archive.validate(latest); });
}
void forget_family_late_writer_and_reuse() {
    Temp temp; Fixture f; SourceArchiveKey root, child, grandchild, unrelated;
    {
        auto archive = SourceArchive::create(temp.path(), config(), limits());
        root = archive.enroll(f.store, f.ref);
        const auto child_ref = f.store.branch(f.ref, 300); child = archive.enroll(f.store, child_ref);
        const auto grand_ref = f.store.branch(child_ref, 100); grandchild = archive.enroll(f.store, grand_ref);
        const auto other = f.store.capture(config(), f.rows.data(), 20, nullptr, 0, 0);
        unrelated = archive.enroll(f.store, other);
        StorageIoQueue queue(4, 1<<20);
        auto existing = archive.save(queue, f.store, child_ref, f.staging); existing.wait(); archive.publish(existing, f.store); existing = {};
        std::optional<SourceArchiveRead> reader(archive.read(child, f.staging));
        Gate gate; auto blocker = gate.occupy(queue);
        auto late = archive.save(queue, f.store, grand_ref, f.staging);
        const auto held = f.staging.charged_bytes(); CHECK(held > reader->image().staging_bytes());
        const auto charged = archive.stats().object_bytes;
        CHECK(archive.forget(root) == 3 && archive.stats().sources == 1);
        rejects(MemoryErrorCode::Stale, [&] { archive.validate(*reader); });
        rejects(MemoryErrorCode::Stale, [&] { (void)archive.read(root, f.staging); });
        CHECK(archive.collect().quota_released == 0 && archive.stats().object_bytes == charged);
        gate.released.set_value(); blocker.wait(); late.wait();
        rejects(MemoryErrorCode::Stale, [&] { archive.publish(late, f.store); });
        CHECK(archive.collect().quota_released == 0); // completed writer receipt still pins its file
        late = {}; CHECK(archive.collect().quota_released > 0);
        reader.reset(); CHECK(archive.collect().quota_released > 0 && archive.stats().object_bytes == 0);
        CHECK(f.staging.charged_bytes() == 0);
        const auto replacement = archive.enroll(f.store, f.ref);
        CHECK(replacement.id > unrelated.id && replacement.space == root.space);
        CHECK(!(replacement == root));
    }
    auto reopened = SourceArchive::open(temp.path(), config(), limits());
    rejects(MemoryErrorCode::Stale, [&] { (void)reopened.read(child, f.staging); });
    rejects(MemoryErrorCode::Stale, [&] { (void)reopened.read(grandchild, f.staging); });
    CHECK(reopened.stats().sources == 2);
}
void cancel_queue_quota_and_local_forget() {
    Temp temp; Fixture f;
    auto archive = SourceArchive::create(temp.path(), config(), limits()); auto key = archive.enroll(f.store, f.ref);
    StorageIoQueue queue(2, 1<<20);
    Gate gate; auto blocker = gate.occupy(queue);
    auto cancel = archive.save(queue, f.store, f.ref, f.staging);
    const auto held = f.staging.charged_bytes(); CHECK(held > 0);
    CHECK(cancel.cancel());
    rejects(MemoryErrorCode::Pending, [&] { (void)archive.save(queue, f.store, f.ref, f.staging); });
    CHECK(f.staging.charged_bytes() == held && archive.stats().objects == 2);
    const auto no_file = archive.collect();
    CHECK(no_file.quota_released > 0 && no_file.file_bytes_removed == 0 && no_file.objects == 1);
    CHECK(archive.stats().objects == 1); // refused job never owned a file
    gate.released.set_value(); blocker.wait();
    bool cancelled = false; try { cancel.wait(); } catch (const MemoryError &) { cancelled = true; } CHECK(cancelled);
    CHECK(f.staging.charged_bytes() == 0); cancel = {};
    const auto cancelled_cleanup = archive.collect();
    CHECK(cancelled_cleanup.quota_released > 0 && cancelled_cleanup.file_bytes_removed == 0);
    auto late = archive.save(queue, f.store, f.ref, f.staging); late.wait();
    CHECK(f.store.forget(f.ref) == 1);
    rejects(MemoryErrorCode::Stale, [&] { archive.publish(late, f.store); });
    rejects(MemoryErrorCode::Stale, [&] { (void)archive.read(key, f.staging); });

    auto bounds = limits(); bounds.disk_bytes = archive.stats().control_reserved + (144+4*513+26*2)-1;
    auto small = SourceArchive::create(temp.path("small"), config(), bounds);
    Fixture other; small.enroll(other.store, other.ref);
    rejects(MemoryErrorCode::BudgetExceeded, [&] { (void)small.save(queue, other.store, other.ref, other.staging); });
    CHECK(other.staging.charged_bytes() == 0 && small.stats().objects == 0);
    bounds = limits(); bounds.metadata_bytes = 1;
    rejects(MemoryErrorCode::BudgetExceeded, [&] { (void)SourceArchive::create(temp.path("tiny-tables"), config(), bounds); });
    CHECK(!std::filesystem::exists(temp.path("tiny-tables")));
    bounds = limits(); bounds.max_objects = bounds.max_sources = 1;
    auto one = SourceArchive::create(temp.path("one"), config(), bounds); one.enroll(other.store, other.ref);
    auto retained = one.save(queue, other.store, other.ref, other.staging); retained.wait();
    rejects(MemoryErrorCode::BudgetExceeded, [&] { (void)one.save(queue, other.store, other.ref, other.staging); });
    const auto another = other.store.capture(config(), nullptr, 0, nullptr, 0, 0);
    rejects(MemoryErrorCode::BudgetExceeded, [&] { one.enroll(other.store, another); });
}
void receipts_owners_and_survivors() {
    Temp temp; Fixture f; StorageIoQueue queue(2, 1<<20);
    std::optional<SourceArchive> first(SourceArchive::create(temp.path(), config(), limits()));
    auto key = first->enroll(f.store, f.ref);
    auto second = SourceArchive::create(temp.path("second"), config(), limits()); second.enroll(f.store, f.ref);
    auto write = first->save(queue, f.store, f.ref, f.staging); write.wait();
    rejects(MemoryErrorCode::Stale, [&] { second.publish(write, f.store); });
    first->publish(write, f.store);
    std::optional<SourceArchiveRead> read(first->read(key, f.staging));
    rejects(MemoryErrorCode::Stale, [&] { second.validate(*read); });
    auto moved = std::move(write); CHECK(write.status() == StorageTaskStatus::Cancelled);
    rejects(MemoryErrorCode::Stale, [&] { first->publish(write, f.store); });
    first.reset();
    rejects(MemoryErrorCode::Pending, [&] { (void)SourceArchive::open(temp.path(), config(), limits()); });
    moved = {}; // reader still holds namespace lock and R bytes
    rows_match(read->image(), f.rows); CHECK(f.staging.charged_bytes() > 0);
    read.reset(); CHECK(f.staging.charged_bytes() == 0);
    auto reopened = SourceArchive::open(temp.path(), config(), limits());
    auto restored = reopened.read(key, f.staging); reopened.validate(restored); rows_match(restored.image(), f.rows);
}
void restart_orphans_and_corruption() {
    Temp temp; Fixture f; SourceArchiveKey key; uint64_t published = 0;
    {
        auto archive = SourceArchive::create(temp.path(), config(), limits()); key = archive.enroll(f.store, f.ref);
        StorageIoQueue queue(2, 1<<20);
        auto saved = archive.save(queue, f.store, f.ref, f.staging); saved.wait(); archive.publish(saved, f.store);
        published = archive.inspect(0)->object;
        auto orphan = archive.save(queue, f.store, f.ref, f.staging); orphan.wait(); // never publish
    }
    CHECK(std::filesystem::exists(temp.path()/"blob.2"));
    {
        auto archive = SourceArchive::open(temp.path(), config(), limits());
        CHECK(archive.stats().objects == 2); auto read = archive.read(key, f.staging); archive.validate(read);
        CHECK(archive.collect().file_bytes_removed == read.image().image_bytes()); CHECK(archive.stats().objects == 1);
    }
    const auto blob = temp.path()/ ("blob."+std::to_string(published));
    const auto original = contents(blob); auto broken = original; broken[150] ^= 1; replace(blob, broken);
    {
        auto archive = SourceArchive::open(temp.path(), config(), limits());
        corrupt([&] { (void)archive.read(key, f.staging); }); CHECK(f.staging.charged_bytes() == 0);
    }
    replace(blob, original);
    copy_archive(temp.path(), temp.path("missing-blob"));
    std::filesystem::remove(temp.path("missing-blob")/("blob."+std::to_string(published)));
    corrupt([&] { (void)SourceArchive::open(temp.path("missing-blob"), config(), limits()); });
    copy_archive(temp.path(), temp.path("unknown-file"));
    replace(temp.path("unknown-file")/"blob.999999", original);
    corrupt([&] { (void)SourceArchive::open(temp.path("unknown-file"), config(), limits()); });
    auto catalog = contents(temp.path()/"catalog"); catalog.back() ^= 1; replace(temp.path()/"catalog", catalog);
    corrupt([&] { (void)SourceArchive::open(temp.path(), config(), limits()); });
    CHECK(std::filesystem::exists(blob)); // no scan-based permission recovery
}
void rollback_and_missing_authority() {
    Temp temp; Fixture f; SourceArchiveKey key;
    std::vector<uint8_t> catalog, seal;
    {
        auto archive = SourceArchive::create(temp.path(), config(), limits()); key = archive.enroll(f.store, f.ref);
        StorageIoQueue queue(1, 1<<20); auto saved = archive.save(queue, f.store, f.ref, f.staging); saved.wait();
        archive.publish(saved, f.store); catalog = contents(temp.path()/"catalog"); seal = contents(temp.path()/"seal");
        CHECK(archive.forget(key) == 1);
    }
    const auto forgotten_catalog = contents(temp.path()/"catalog");
    const auto forgotten_seal = contents(temp.path()/"seal");
    replace(temp.path()/"catalog", catalog);
    corrupt([&] { (void)SourceArchive::open(temp.path(), config(), limits()); });
    replace(temp.path()/"catalog", forgotten_catalog); replace(temp.path()/"seal", seal);
    corrupt([&] { (void)SourceArchive::open(temp.path(), config(), limits()); });
    replace(temp.path()/"seal", forgotten_seal);
    {
        auto reopened = SourceArchive::open(temp.path(), config(), limits());
        rejects(MemoryErrorCode::Stale, [&] { (void)reopened.read(key, f.staging); });
        CHECK(reopened.stats().objects == 1); CHECK(reopened.collect().file_bytes_removed > 0);
    }
    std::filesystem::remove(temp.path()/"seal");
    io_failure([&] { (void)SourceArchive::open(temp.path(), config(), limits()); });
}

// A crash/error before acknowledgement may leave the preceding committed
// version or a denied ambiguous archive; it may not synthesize new authority.
bool reopened_has_source(const std::filesystem::path & path, SourceArchiveKey key, Fixture & f) {
    try {
        auto archive = SourceArchive::open(path, config(), limits());
        auto read = archive.read(key, f.staging); archive.validate(read); rows_match(read.image(), f.rows); return true;
    } catch (const SnapshotCorrupt &) { return false; }
      catch (const MemoryError & e) { CHECK(e.code() == MemoryErrorCode::Stale); return false; }
}
void error_and_process_crash_boundaries() {
    Temp temp; Fixture f; SourceArchiveKey key;
    {
        auto archive = SourceArchive::create(temp.path("seed"), config(), limits()); key = archive.enroll(f.store, f.ref);
        StorageIoQueue queue(1, 1<<20); auto saved = archive.save(queue, f.store, f.ref, f.staging); saved.wait(); archive.publish(saved, f.store);
    }
    size_t operations = 0, boundaries = 0;
    copy_archive(temp.path("seed"), temp.path("probe"));
    {
        auto archive = SourceArchive::open(temp.path("probe"), config(), limits());
        { fault::Scope count(fault::Count); archive.forget(key); operations = fault::calls; }
    }
    copy_archive(temp.path("seed"), temp.path("probe-sync"));
    {
        auto archive = SourceArchive::open(temp.path("probe-sync"), config(), limits());
        { fault::Scope count(fault::Count, SIZE_MAX, false); archive.forget(key); boundaries = fault::calls; }
    }
    CHECK(operations > boundaries && boundaries == 6);
    size_t old_allowed = 0, refused = 0;
    for (size_t i = 0; i < operations; ++i) {
        const auto path = temp.root/("error-"+std::to_string(i)); copy_archive(temp.path("seed"), path);
        {
            auto archive = SourceArchive::open(path, config(), limits());
            { fault::Scope error(fault::Error, i); io_failure([&] { archive.forget(key); }); CHECK(fault::fired); }
            CHECK(!archive.stats().healthy);
            rejects(MemoryErrorCode::ServiceUnavailable, [&] { (void)archive.read(key, f.staging); });
        }
        if (reopened_has_source(path, key, f)) ++old_allowed; else ++refused;
        CHECK(f.staging.charged_bytes() == 0);
    }
    for (size_t i = 0; i < boundaries; ++i) {
        const auto path = temp.root/("crash-"+std::to_string(i)); copy_archive(temp.path("seed"), path);
        const auto child = ::fork(); CHECK(child >= 0);
        if (!child) {
            auto archive = SourceArchive::open(path, config(), limits());
            fault::Scope crash(fault::Crash, i, false); archive.forget(key); ::_exit(72);
        }
        int result = 0; CHECK(::waitpid(child, &result, 0) == child); CHECK(WIFEXITED(result) && WEXITSTATUS(result) == 71);
        (void)reopened_has_source(path, key, f); CHECK(f.staging.charged_bytes() == 0);
    }
    std::printf("metadata I/O error points=%zu old-version-reopen=%zu denied-or-forgotten=%zu process-crash-boundaries=%zu\n",
        operations, old_allowed, refused, boundaries);
}
void actual_save_io_failures() {
    Temp temp; Fixture f;
    size_t points = 0;
    {
        auto archive = SourceArchive::create(temp.path("probe"), config(), limits()); archive.enroll(f.store, f.ref);
        StorageIoQueue queue(1, 1<<20);
        fault::Scope count(fault::Count);
        auto write = archive.save(queue, f.store, f.ref, f.staging); write.wait(); points = fault::calls;
    }
    size_t metadata_failures = 0, blob_failures = 0;
    for (size_t i = 0; i < points; ++i) {
        const auto path = temp.root/("save-error-"+std::to_string(i)); SourceArchiveKey saved_key;
        {
            auto archive = SourceArchive::create(path, config(), limits()); saved_key = archive.enroll(f.store, f.ref);
            StorageIoQueue queue(1, 1<<20); std::optional<SourceArchiveWrite> write;
            {
                fault::Scope error(fault::Error, i);
                io_failure([&] { write.emplace(archive.save(queue, f.store, f.ref, f.staging)); write->wait(); });
                CHECK(fault::fired);
            }
            CHECK(f.staging.charged_bytes() == 0 && archive.stats().charged <= archive.stats().capacity);
            if (archive.stats().healthy) {
                ++blob_failures; CHECK(write && write->status() == StorageTaskStatus::Failed);
                io_failure([&] { archive.publish(*write, f.store); });
                rejects(MemoryErrorCode::Stale, [&] { (void)archive.read(saved_key, f.staging); });
                CHECK(archive.collect().quota_released == 0); write.reset();
                const auto cleanup = archive.collect();
                CHECK(cleanup.quota_released > 0 && cleanup.file_bytes_removed <= cleanup.quota_released);
            } else {
                ++metadata_failures; CHECK(!write);
                rejects(MemoryErrorCode::ServiceUnavailable, [&] { (void)archive.read(saved_key, f.staging); });
            }
        }
        CHECK(!reopened_has_source(path, saved_key, f)); // even complete-looking uncommitted blobs grant nothing
    }
    CHECK(metadata_failures > 0 && blob_failures > 0);
    std::printf("save I/O error points=%zu metadata=%zu background-blob=%zu\n", points, metadata_failures, blob_failures);
}
void allocation_failures() {
    Temp temp; size_t save_points = 0, read_points = 0;
    for (size_t point = 0; point < 128; ++point) {
        Fixture f; StorageIoQueue queue(1, 1<<20);
        auto archive = SourceArchive::create(temp.root/("allocation-save-"+std::to_string(point)), config(), limits());
        archive.enroll(f.store, f.ref); const auto before = f.store.stats().charged;
        bool failed = false; std::optional<SourceArchiveWrite> write;
        allocation::calls = 0; allocation::fail_at = point;
        try { write.emplace(archive.save(queue, f.store, f.ref, f.staging)); }
        catch (const std::bad_alloc &) { failed = true; }
        allocation::fail_at = SIZE_MAX;
        CHECK(f.store.stats().charged == before && archive.stats().charged <= archive.stats().capacity);
        if (!failed) { write->wait(); CHECK(f.staging.charged_bytes() == 0); break; }
        ++save_points; CHECK(f.staging.charged_bytes() == 0);
        if (archive.stats().healthy) { archive.collect(); CHECK(archive.stats().object_bytes == 0); }
    }
    {
        Fixture f; StorageIoQueue queue(1, 1<<20);
        auto archive = SourceArchive::create(temp.path("allocation-read"), config(), limits()); auto key = archive.enroll(f.store, f.ref);
        auto write = archive.save(queue, f.store, f.ref, f.staging); write.wait(); archive.publish(write, f.store); write = {};
        for (size_t point = 0; point < 128; ++point) {
            bool failed = false; std::optional<SourceArchiveRead> read;
            allocation::calls = 0; allocation::fail_at = point;
            try { read.emplace(archive.read(key, f.staging)); }
            catch (const std::bad_alloc &) { failed = true; }
            allocation::fail_at = SIZE_MAX;
            if (!failed) { archive.validate(*read); rows_match(read->image(), f.rows); break; }
            ++read_points;
            CHECK(f.staging.charged_bytes() == 0 && archive.stats().leased_objects == 0 && archive.stats().healthy);
        }
    }
    CHECK(save_points > 0 && save_points < 128 && read_points > 0 && read_points < 128);
    std::printf("foreground allocation failure points: save=%zu read=%zu\n", save_points, read_points);
}
void shared_metadata_and_cleanup_retry() {
    Temp temp; Fixture f; uint64_t bytes = 0;
    { auto probe = SourceArchive::create(temp.path("probe"), config(), limits()); bytes = probe.stats().metadata_bytes; }
    auto budget = std::make_shared<HistoricalKvAuxBudget>(2*bytes-1);
    std::optional<SourceArchive> archive(SourceArchive::create(temp.path(), config(), limits(), budget));
    CHECK(budget->stats().charged == bytes);
    rejects(MemoryErrorCode::BudgetExceeded, [&] {
        (void)SourceArchive::create(temp.path("second"), config(), limits(), budget);
    });
    CHECK(!std::filesystem::exists(temp.path("second")) && budget->stats().charged == bytes);
    auto key = archive->enroll(f.store, f.ref); StorageIoQueue queue(1, 1<<20);
    auto write = archive->save(queue, f.store, f.ref, f.staging); write.wait(); archive->publish(write, f.store); write = {};
    std::optional<SourceArchiveRead> read(archive->read(key, f.staging)); archive->forget(key);
    const auto charged = archive->stats().object_bytes; CHECK(archive->collect().quota_released == 0);
    read.reset();
    {
        fault::Scope sync_failure(fault::Error, 0, false);
        io_failure([&] { archive->collect(); }); CHECK(fault::fired);
    }
    CHECK(archive->stats().healthy && archive->stats().object_bytes == charged);
    const auto retry = archive->collect();
    CHECK(retry.quota_released == charged && retry.file_bytes_removed == charged);
    CHECK(archive->collect().quota_released == 0);
    // The same source can be explicitly enrolled anew, with a new durable key.
    const auto replacement = archive->enroll(f.store, f.ref); CHECK(replacement.id > key.id);
    auto last_write = archive->save(queue, f.store, f.ref, f.staging); last_write.wait(); archive->publish(last_write, f.store);
    read.emplace(archive->read(replacement, f.staging)); archive.reset();
    CHECK(budget->stats().charged == bytes); last_write = {}; CHECK(budget->stats().charged == bytes);
    read.reset(); CHECK(budget->stats().charged == 0 && f.staging.charged_bytes() == 0);
    auto reopened = SourceArchive::open(temp.path(), config(), limits(), budget);
    auto new_key = reopened.enroll(f.store, f.ref); CHECK(new_key.id > replacement.id);
}
}
int main() {
    roundtrip_reopen_and_lock(); versions_and_reader_retention(); forget_family_late_writer_and_reuse();
    cancel_queue_quota_and_local_forget(); receipts_owners_and_survivors(); restart_orphans_and_corruption();
    rollback_and_missing_authority(); error_and_process_crash_boundaries(); actual_save_io_failures(); allocation_failures();
    shared_metadata_and_cleanup_retry();
    std::puts("source archive lifecycle, quotas, durable authority and failure boundaries PASS");
}
