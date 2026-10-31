#include "kvmem/source_archive.hpp"
#include "kvmem/source_recovery.hpp"
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
void import_receipt_version_and_family() {
    Temp temp; Fixture f; SourceArchiveKey root, child;
    for (int i = 0; i < 4; ++i) f.append();
    {
        auto archive = SourceArchive::create(temp.path(),config(),limits()); root = archive.enroll(f.store,f.ref);
        const auto branch = f.store.branch(f.ref,300); child = archive.enroll(f.store,branch);
        StorageIoQueue queue(2,1<<20);
        auto a = archive.save(queue,f.store,f.ref,f.staging), b = archive.save(queue,f.store,branch,f.staging);
        a.wait(); b.wait(); archive.publish(a,f.store); archive.publish(b,f.store);
    }
    auto archive = SourceArchive::open(temp.path(),config(),limits());
    auto read = archive.read(root,f.staging);
    SourceJournalStore small{{1<<16,32,2,2}}, restored{{1<<20,8192,256,4}}, another{{1<<20,8192,256,4}};
    const auto small_before = small.stats();
    rejects(MemoryErrorCode::BudgetExceeded,[&] { archive.import_source(read,small); });
    CHECK(small.stats().sources == 0 && small.stats().charged == small_before.charged);
    const auto imported = archive.import_source(read,restored);
    CHECK(imported.revision == 1 && imported.revision < read.image().origin().ref.revision);
    CHECK(!restored.state(imported).computed_frontier && restored.state(imported).parent == SourceRef{});
    const auto count = restored.stats().sources;
    rejects(MemoryErrorCode::Pending,[&] { archive.import_source(read,restored); }); CHECK(restored.stats().sources == count);
    const int32_t token = 777; const auto appended = restored.append(imported,&token,1,nullptr,0,0);
    StorageIoQueue queue(2,1<<20); auto next = archive.save(queue,restored,appended,f.staging); next.wait();
    CHECK(appended.revision < read.image().origin().ref.revision);
    archive.publish(next,restored); // durable sequence wins over the old process/store revision
    rejects(MemoryErrorCode::Stale,[&] { archive.import_source(read,another); }); CHECK(!another.stats().sources);
    auto child_read = archive.read(child,f.staging);
    const auto imported_child = archive.import_source(child_read,restored);
    CHECK(restored.state(imported_child).parent == SourceRef{} && archive.inspect(1)->parent == root);
    auto late = archive.save(queue,restored,imported_child,f.staging); late.wait();
    CHECK(archive.forget(root) == 2);
    rejects(MemoryErrorCode::Stale,[&] { archive.publish(late,restored); });
    rejects(MemoryErrorCode::Stale,[&] { archive.import_source(child_read,another); });
    CHECK(restored.state(imported_child).ref == imported_child); // archive-only forget, not Engine alias revocation
}
void import_allocation_rollback() {
    Temp temp; Fixture f; SourceArchiveKey key;
    {
        auto archive = SourceArchive::create(temp.path(),config(),limits()); key = archive.enroll(f.store,f.ref);
        StorageIoQueue queue(1,1<<20); auto write = archive.save(queue,f.store,f.ref,f.staging); write.wait(); archive.publish(write,f.store);
    }
    auto archive = SourceArchive::open(temp.path(),config(),limits()); auto read = archive.read(key,f.staging);
    size_t failures = 0; bool complete = false;
    for (size_t point = 0; point < 32; ++point) {
        SourceJournalStore destination{{1<<20,8192,256,4}}; const auto before = destination.stats().charged;
        bool failed = false; allocation::calls = 0; allocation::fail_at = point;
        try { archive.import_source(read,destination); } catch (const std::bad_alloc &) { failed = true; }
        allocation::fail_at = SIZE_MAX;
        if (!failed) { complete = true; break; }
        ++failures; CHECK(!destination.stats().sources && destination.stats().charged == before);
        archive.validate(read); CHECK(archive.stats().healthy);
    }
    CHECK(complete && failures > 0); std::printf("archive authorized import allocation failure points: %zu\n",failures);
}
void write_client_key(const std::filesystem::path & path, SourceArchiveKey key) {
    std::ofstream out(path.string()+".key",std::ios::binary);
    out.write(reinterpret_cast<const char *>(key.space.data()),key.space.size());
    out.write(reinterpret_cast<const char *>(&key.id),sizeof(key.id)); CHECK(out.good());
}
SourceArchiveKey client_key(const std::filesystem::path & path) {
    SourceArchiveKey key; std::ifstream in(path.string()+".key",std::ios::binary);
    in.read(reinterpret_cast<char *>(key.space.data()),key.space.size());
    in.read(reinterpret_cast<char *>(&key.id),sizeof(key.id)); CHECK(in.good()); return key;
}
void checkpoint_restart(const char *,const std::filesystem::path &);
void restart_child(const char * mode, const std::filesystem::path & path) {
    if (!std::strncmp(mode,"--checkpoint-",13)) {checkpoint_restart(mode,path);return;}
    if (!std::strcmp(mode,"--restart-write")) {
        Fixture f; for (int i = 0; i < 3; ++i) f.append();
        auto archive = SourceArchive::create(path,config(),limits()); const auto key = archive.enroll(f.store,f.ref);
        StorageIoQueue queue(1,1<<20); auto write = archive.save(queue,f.store,f.ref,f.staging); write.wait(); archive.publish(write,f.store);
        write_client_key(path,key); return;
    }
    SourceJournalStore store{{1<<20,8192,256,4}}; HostKvStorage staging(1<<20);
    auto archive = SourceArchive::open(path,config(),limits()); const auto key = client_key(path);
    if (!std::strcmp(mode,"--restart-deny")) {
        rejects(MemoryErrorCode::Stale,[&] { (void)archive.read(key,staging); });
        CHECK(archive.stats().sources == 0); archive.collect(); return;
    }
    auto read = archive.read(key,staging);
    const auto imported = archive.import_source(read,store); CHECK(!store.state(imported).computed_frontier);
    if (!std::strcmp(mode,"--restart-append")) {
        // Both executables start their RAM incarnation counter at1. Numeric
        // identity collision is legal; only a current archive receipt authorizes
        // this import, and durable version ordering must not use those counters.
        CHECK(imported.source == read.image().origin().ref.source);
        CHECK(imported.revision == 1 && read.image().origin().ref.revision == 4);
        const int32_t token = 777; const auto next = store.append(imported,&token,1,nullptr,0,0);
        StorageIoQueue queue(1,1<<20); auto write = archive.save(queue,store,next,staging); write.wait(); archive.publish(write,store);
        std::printf("exec restart: numeric-origin-collision=1 saved-revision=4 new-revision=1 durable-publication=PASS\n"); return;
    }
    CHECK(!std::strcmp(mode,"--restart-forget"));
    CHECK(read.image().origin().accepted_frontier == 517);
    int32_t last = 0; store.read_tokens(imported,516,&last,1); CHECK(last == 777);
    CHECK(archive.forget(key) == 1);
    rejects(MemoryErrorCode::Stale,[&] { archive.validate(read); });
}
void separate_exec_restart() {
    Temp temp;
    for (const auto mode : {"--restart-write","--restart-append","--restart-forget","--restart-deny"}) {
        const auto child = ::fork(); CHECK(child >= 0);
        if (!child) { ::execl("/proc/self/exe","source_archive_test",mode,temp.path().c_str(),static_cast<char *>(nullptr)); ::_exit(74); }
        int result = 0; CHECK(::waitpid(child,&result,0) == child);
        CHECK(WIFEXITED(result) && WEXITSTATUS(result) == 0);
    }
    std::puts("four separate executable processes: save/import-append/reimport-forget/reopen-deny PASS");
}
void owner_queries_and_structural_queue_limit() {
    Temp temp; Fixture f; auto archive = SourceArchive::create(temp.path(),config(),limits());
    const auto root = archive.enroll(f.store,f.ref); const auto branch = f.store.branch(f.ref,300);
    const auto child = archive.enroll(f.store,branch);
    CHECK(archive.slots() == limits().max_sources && archive.key_for(f.store,f.ref) == root);
    CHECK(archive.bound_source(child,f.store) == branch && archive.family_contains(root,child));
    CHECK(!archive.family_contains(child,root));
    const int32_t token = 17; f.ref = f.store.append(f.ref,&token,1,nullptr,0,0);
    CHECK(archive.bound_source(root,f.store) == f.ref);
    const auto before = archive.stats(); const auto r = f.staging.charged_bytes();
    StorageIoQueue impossible(1,1);
    CHECK(impossible.capacity_bytes() == 1);
    rejects(MemoryErrorCode::BudgetExceeded,[&] { (void)archive.save(impossible,f.store,f.ref,f.staging); });
    CHECK(archive.stats().objects == before.objects && archive.stats().object_bytes == before.object_bytes);
    CHECK(f.staging.charged_bytes() == r && archive.stats().healthy);
    StorageIoQueue queue(1,1<<20); auto write = archive.save(queue,f.store,branch,f.staging);
    CHECK(write.key() == child); write.wait(); archive.publish(write,f.store);
    CHECK(archive.forget(root) == 2);
    rejects(MemoryErrorCode::Stale,[&] { (void)archive.bound_source(child,f.store); });
}
void current_family_cooldown_rebind_and_old_receipts() {
    Temp temp;Fixture f;auto archive=SourceArchive::create(temp.path(),config(),limits());
    StorageIoQueue queue(2,1<<20);
    const auto save=[&](SourceRef ref) {
        const auto key=archive.enroll(f.store,ref);auto write=archive.save(queue,f.store,ref,f.staging);
        write.wait();archive.publish(write,f.store);return key;
    };
    const auto root=save(f.ref);const auto child=f.store.branch(f.ref,300);const auto child_key=save(child);
    const auto independent=f.store.capture(config(),f.rows.data(),17,nullptr,0,0);const auto other=save(independent);
    const auto grandchild=f.store.branch(child,129);
    const auto before=f.store.stats();
    rejects(MemoryErrorCode::Pending,[&] { archive.cool_sources(f.store,&f.ref,1); });
    CHECK(f.store.stats().charged==before.charged && archive.bound_source(root,f.store)==f.ref);
    const std::array<SourceRef,3> family{f.ref,grandchild,child};
    rejects(MemoryErrorCode::Pending,[&] { archive.cool_sources(f.store,family.data(),family.size()); });
    CHECK(f.store.stats().sources==4 && archive.bound_source(child_key,f.store)==child);
    const auto grand_key=save(grandchild);
    auto read=archive.read(root,f.staging);
    const SourceRecoveryContext current{{},config()};
    const SourceRecoveryRequest request{f.ref,current,{91,1},128,SourceRecoveryMode::RebuildAsNew,{128,64,192,0}};
    HostKvStorage prepared_ram(1024);auto preparation=prepare_source_recovery(
        plan_source_recovery(f.store,descriptor(2),current,request),f.store,current,prepared_ram);
    auto late=archive.save(queue,f.store,child,f.staging);late.wait();
    const auto catalog=contents(temp.path()/"catalog"),seal=contents(temp.path()/"seal");
    const auto disk=archive.stats().charged,r=f.staging.charged_bytes();
    allocation::calls=0;allocation::fail_at=0;
    const auto removed=archive.cool_sources(f.store,family.data(),family.size());
    allocation::fail_at=SIZE_MAX;
    CHECK(removed==3 && allocation::calls==0 && f.store.stats().sources==1);
    CHECK(archive.stats().sources==4 && archive.stats().charged==disk && f.staging.charged_bytes()==r);
    CHECK(contents(temp.path()/"catalog")==catalog && contents(temp.path()/"seal")==seal);
    CHECK(!archive.bound_source(root,f.store) && !archive.bound_source(child_key,f.store));
    CHECK(archive.bound_source(other,f.store)==independent && f.store.state(independent).accepted_frontier==17);
    rejects(MemoryErrorCode::Stale,[&] { preparation.validate(f.store,current); });
    CHECK(prepared_ram.charged_bytes()==512);
    rejects(MemoryErrorCode::Stale,[&] { archive.publish(late,f.store); });
    archive.validate(read);rows_match(read.image(),f.rows);
    const auto fresh=archive.import_source(read,f.store);
    auto child_read=archive.read(child_key,f.staging);const auto fresh_child=archive.import_source(child_read,f.store);
    auto grand_read=archive.read(grand_key,f.staging);const auto fresh_grand=archive.import_source(grand_read,f.store);
    CHECK(fresh.source.id>grandchild.source.id && fresh.revision==1 && !f.store.state(fresh).parent.source.id);
    CHECK(!f.store.state(fresh).computed_frontier && !f.store.state(fresh_child).parent.source.id);
    rejects(MemoryErrorCode::Stale,[&] { archive.publish(late,f.store); });
    // No resave required: current read-authorized imported data is already
    // fully persisted, despite old disk ref/revision/parent/computed provenance.
    const std::array<SourceRef,3> imported{fresh_child,fresh,fresh_grand};
    {fault::Scope count(fault::Count);CHECK(archive.cool_sources(f.store,imported.data(),imported.size())==3);CHECK(fault::calls==0);}
    CHECK(archive.cool_sources(f.store,nullptr,0)==0 && f.store.stats().sources==1);
    CHECK(contents(temp.path()/"catalog")==catalog && contents(temp.path()/"seal")==seal);
    CHECK(archive.forget(root)==3);
    rejects(MemoryErrorCode::Stale,[&] { archive.validate(read); });
    rejects(MemoryErrorCode::Stale,[&] { archive.import_source(read,f.store); });
    CHECK(archive.stats().sources==1 && archive.bound_source(other,f.store)==independent);
}
void cooldown_latest_copy_and_file_failures_are_atomic() {
    Temp temp;Fixture f;auto archive=SourceArchive::create(temp.path(),config(),limits());
    const auto key=archive.enroll(f.store,f.ref);StorageIoQueue queue(1,1<<20);
    const auto save=[&] {auto write=archive.save(queue,f.store,f.ref,f.staging);write.wait();archive.publish(write,f.store);};
    rejects(MemoryErrorCode::Pending,[&] { archive.cool_sources(f.store,&f.ref,1); });
    save();f.append();
    rejects(MemoryErrorCode::Pending,[&] { archive.cool_sources(f.store,&f.ref,1); });
    save();const auto old=f.ref;f.ref=f.store.append(f.ref,nullptr,0,nullptr,0,513);
    rejects(MemoryErrorCode::Pending,[&] { archive.cool_sources(f.store,&f.ref,1); });
    rejects(MemoryErrorCode::Stale,[&] { archive.cool_sources(f.store,&old,1); });
    save();const auto snapshot=*archive.inspect(0);
    const auto blob=temp.path()/("blob."+std::to_string(snapshot.object));const auto bytes=contents(blob);
    const auto charged=f.store.stats().charged;
    auto verify_unchanged=[&] {
        CHECK(f.store.stats().charged==charged && archive.bound_source(key,f.store)==f.ref && archive.stats().healthy);
        std::vector<int32_t> rows(f.rows.size());f.store.read_tokens(f.ref,0,rows.data(),rows.size());CHECK(rows==f.rows);
    };
    auto shorter=bytes;shorter.pop_back();replace(blob,shorter);
    corrupt([&] { archive.cool_sources(f.store,&f.ref,1); });verify_unchanged();replace(blob,bytes);
    auto damaged=bytes;damaged.at(200)^=1;replace(blob,damaged);
    corrupt([&] { archive.cool_sources(f.store,&f.ref,1); });verify_unchanged();replace(blob,bytes);
    const auto held=temp.root/"held-blob";std::filesystem::rename(blob,held);
    io_failure([&] { archive.cool_sources(f.store,&f.ref,1); });verify_unchanged();std::filesystem::rename(held,blob);
    const std::array<SourceRef,2> duplicate{f.ref,f.ref};
    rejects(MemoryErrorCode::InvalidPlan,[&] { archive.cool_sources(f.store,duplicate.data(),duplicate.size()); });verify_unchanged();
    rejects(MemoryErrorCode::InvalidPlan,[&] { archive.cool_sources(f.store,nullptr,1); });verify_unchanged();
    Fixture foreign;
    rejects(MemoryErrorCode::Stale,[&] { archive.cool_sources(f.store,&foreign.ref,1); });verify_unchanged();
    CHECK(archive.cool_sources(f.store,&f.ref,1)==1 && f.store.stats().buffer_bytes==0 && archive.stats().sources==1);
    auto current=archive.read(key,f.staging);const auto imported=archive.import_source(current,f.store);
    CHECK(imported.revision==1 && current.image().origin().ref.revision==3);
    CHECK(archive.cool_sources(f.store,&imported,1)==1 && f.store.stats().sources==0);
}
SourceArchiveLimits checkpoint_limits() {auto value=limits();value.max_checkpoint_bytes=65536;return value;}
SourceConfiguration checkpoint_schema() {auto value=config();value[1]=91;return value;}
constexpr size_t checkpoint_payload=20017;
SourceArchiveCheckpointInput checkpoint_input(HostKvStorage & ram) {
    auto record=ram.allocate(checkpoint_payload);
    for (size_t i=0;i<checkpoint_payload;++i) record.writable_data()[i]=uint8_t(i*31+7);
    record.seal();auto owner=std::make_shared<HostKvRecord>(std::move(record));
    SourceArchiveCheckpointInput input;
    input.schema=checkpoint_schema();input.payload_bytes=input.retained_bytes=checkpoint_payload;input.owner=owner;
    input.write=[owner](const SnapshotWriter::Sink & sink) {
        sink(owner->data(),7777);sink(owner->data()+7777,owner->size()-7777);
    };
    return input;
}
SourceArchiveKey checkpoint_source(SourceArchive & archive,Fixture & f,StorageIoQueue & queue) {
    const auto key=archive.enroll(f.store,f.ref);auto write=archive.save(queue,f.store,f.ref,f.staging);
    write.wait();archive.publish(write,f.store);return key;
}
SourceArchiveCheckpointRead checkpoint_read(SourceArchive & archive,const SourceArchiveRead & source) {
    std::vector<uint8_t> private_bytes;
    auto read=archive.read_checkpoint(source,checkpoint_schema(),[&](const void * p,size_t n) {
        CHECK(n<=8192);const auto bytes=static_cast<const uint8_t *>(p);private_bytes.insert(private_bytes.end(),bytes,bytes+n);
    });
    CHECK(private_bytes.size()==checkpoint_payload && read.payload_bytes()==checkpoint_payload && read.key()==source.key());
    for (size_t i=0;i<private_bytes.size();++i) CHECK(private_bytes[i]==uint8_t(i*31+7));
    archive.validate(read);return read;
}
void checkpoint_publication_and_owned_lifetimes() {
    Temp temp;Fixture f;HostKvStorage ram(65536);StorageIoQueue queue(2,1<<20);
    auto archive=SourceArchive::create(temp.path(),config(),checkpoint_limits());
    const auto key=checkpoint_source(archive,f,queue);const auto base=archive.stats().object_bytes;
    const auto catalog=contents(temp.path()/"catalog"),seal=contents(temp.path()/"seal");
    Gate gate;auto occupied=gate.occupy(queue);
    auto write=archive.save_checkpoint(queue,f.store,f.ref,checkpoint_input(ram));
    CHECK(ram.charged_bytes()==checkpoint_payload && queue.pinned_bytes()==checkpoint_payload+1);
    CHECK(archive.stats().object_bytes==base+checkpoint_payload+104 && archive.stats().objects==1);
    rejects(MemoryErrorCode::Pending,[&]{archive.publish_checkpoint(write,f.store);});
    rejects(MemoryErrorCode::Pending,[&]{archive.retire_checkpoint(key);});
    {
        auto source=archive.read(key,f.staging);
        rejects(MemoryErrorCode::Stale,[&]{(void)checkpoint_read(archive,source);});
        CHECK(!archive.collect().quota_released);
        gate.released.set_value();occupied.wait();write.wait();
        CHECK(!ram.charged_bytes());
        rejects(MemoryErrorCode::Stale,[&]{(void)checkpoint_read(archive,source);}); // I/O completion is not publication
        archive.publish_checkpoint(write,f.store);archive.publish_checkpoint(write,f.store);
        CHECK(contents(temp.path()/"catalog")==catalog && contents(temp.path()/"seal")==seal);
        const auto r=f.staging.charged_bytes();auto checkpoint=checkpoint_read(archive,source);
        CHECK(f.staging.charged_bytes()==r); // no full-payload R copy in the archive
        CHECK(archive.cool_sources(f.store,&f.ref,1)==1);archive.validate(checkpoint);
        rejects(MemoryErrorCode::Stale,[&]{archive.publish_checkpoint(write,f.store);});
        const auto imported=archive.import_source(source,f.store);CHECK(!f.store.state(imported).computed_frontier);
        archive.validate(checkpoint);CHECK(archive.forget(key)==1);
        rejects(MemoryErrorCode::Stale,[&]{archive.validate(checkpoint);});
        CHECK(!archive.collect().quota_released && archive.stats().object_bytes==base+checkpoint_payload+104);
        f.store.forget(imported);
    }
    CHECK(!archive.collect().quota_released);write={};
    const auto freed=archive.collect();CHECK(freed.quota_released==base+checkpoint_payload+104 && freed.file_bytes_removed==freed.quota_released);
    CHECK(!archive.stats().object_bytes && !archive.stats().objects);
}
void checkpoint_versions_retirement_and_corruption() {
    Temp temp;Fixture f;HostKvStorage ram(65536);StorageIoQueue queue(1,1<<20);
    auto archive=SourceArchive::create(temp.path(),config(),checkpoint_limits());
    const auto key=checkpoint_source(archive,f,queue);const auto source_bytes=archive.stats().object_bytes;
    auto write=archive.save_checkpoint(queue,f.store,f.ref,checkpoint_input(ram));write.wait();archive.publish_checkpoint(write,f.store);
    const auto file=temp.path()/"checkpoint.1";const auto bytes=contents(file);
    {
        auto source=archive.read(key,f.staging);auto held=checkpoint_read(archive,source);
        auto wrong=checkpoint_schema();wrong[1]^=1;size_t callbacks=0;
        rejects(MemoryErrorCode::InvalidPlan,[&]{(void)archive.read_checkpoint(source,wrong,[&](const void *,size_t){++callbacks;});});CHECK(!callbacks);
        auto damaged=bytes;damaged[123]^=1;replace(file,damaged);
        corrupt([&]{(void)checkpoint_read(archive,source);});archive.validate(source);replace(file,bytes);
        damaged.pop_back();replace(file,damaged);corrupt([&]{(void)checkpoint_read(archive,source);});replace(file,bytes);
        CHECK(archive.retire_checkpoint(key) && !archive.retire_checkpoint(key));
        rejects(MemoryErrorCode::Stale,[&]{archive.validate(held);});
        rejects(MemoryErrorCode::Stale,[&]{archive.publish_checkpoint(write,f.store);});
        rejects(MemoryErrorCode::Stale,[&]{(void)checkpoint_read(archive,source);});
        CHECK(!archive.collect().quota_released && archive.stats().sources==1);
    }
    write={};const auto retired=archive.collect();
    CHECK(retired.quota_released==bytes.size() && retired.file_bytes_removed==bytes.size());
    CHECK(archive.stats().object_bytes==source_bytes && std::filesystem::file_size(temp.path()/"checkpoint-retired.1")==0);
    rejects(MemoryErrorCode::Pending,[&]{(void)archive.save_checkpoint(queue,f.store,f.ref,checkpoint_input(ram));});
    CHECK(!ram.charged_bytes());
    // A new committed source blob grants a distinct attachment identity, even
    // when saving the same live RAM revision and identical source bytes.
    checkpoint_source(archive,f,queue);
    auto next=archive.save_checkpoint(queue,f.store,f.ref,checkpoint_input(ram));next.wait();archive.publish_checkpoint(next,f.store);
    {
        auto source=archive.read(key,f.staging);auto read=checkpoint_read(archive,source);
        f.append();
        rejects(MemoryErrorCode::Stale,[&]{archive.publish_checkpoint(next,f.store);});
        rejects(MemoryErrorCode::Pending,[&]{(void)archive.save_checkpoint(queue,f.store,f.ref,checkpoint_input(ram));});
        checkpoint_source(archive,f,queue);
        rejects(MemoryErrorCode::Stale,[&]{archive.validate(read);});
        rejects(MemoryErrorCode::Stale,[&]{(void)checkpoint_read(archive,source);});
    }
    next={};archive.collect();CHECK(!std::filesystem::exists(temp.path()/"checkpoint-retired.1"));
    CHECK(!std::filesystem::exists(temp.path()/"checkpoint.2"));
}
void checkpoint_bounds_producer_failures_and_late_writer() {
    Temp temp;Fixture f;HostKvStorage ram(65536);StorageIoQueue queue(2,1<<20);
    auto archive=SourceArchive::create(temp.path(),config(),checkpoint_limits());
    const auto key=checkpoint_source(archive,f,queue);const auto charged=archive.stats().object_bytes;
    StorageIoQueue small(1,checkpoint_payload-1);
    rejects(MemoryErrorCode::BudgetExceeded,[&]{(void)archive.save_checkpoint(small,f.store,f.ref,checkpoint_input(ram));});
    CHECK(archive.stats().object_bytes==charged && !ram.charged_bytes());
    auto invalid=checkpoint_input(ram);invalid.retained_bytes=1;
    rejects(MemoryErrorCode::InvalidPlan,[&]{(void)archive.save_checkpoint(queue,f.store,f.ref,std::move(invalid));});
    CHECK(!ram.charged_bytes());
    for (const bool too_long:{false,true}) {
        auto input=checkpoint_input(ram);input.write=[too_long](const SnapshotWriter::Sink & sink) {
            uint8_t byte=7;sink(&byte,too_long ? checkpoint_payload+1 : 1);
        };
        auto failed=archive.save_checkpoint(queue,f.store,f.ref,std::move(input));
        corrupt([&]{failed.wait();});corrupt([&]{archive.publish_checkpoint(failed,f.store);});
        CHECK(!ram.charged_bytes() && archive.stats().healthy);failed={};archive.collect();
        CHECK(archive.stats().object_bytes==charged);
    }
    auto late=archive.save_checkpoint(queue,f.store,f.ref,checkpoint_input(ram));late.wait();
    CHECK(archive.cool_sources(f.store,&f.ref,1)==1);
    {
        auto source=archive.read(key,f.staging);const auto fresh=archive.import_source(source,f.store);
        rejects(MemoryErrorCode::Stale,[&]{archive.publish_checkpoint(late,f.store);});
        rejects(MemoryErrorCode::Stale,[&]{(void)checkpoint_read(archive,source);});
        CHECK(archive.cool_sources(f.store,&fresh,1)==1);
    }
    CHECK(!archive.collect().quota_released);late={};
    CHECK(archive.collect().quota_released==checkpoint_payload+104 && archive.stats().object_bytes==charged);
    CHECK(archive.stats().healthy && archive.stats().sources==1);
}
void checkpoint_restart(const char * mode,const std::filesystem::path & path) {
    Fixture f;HostKvStorage ram(65536);StorageIoQueue queue(1,1<<20);
    if (!std::strcmp(mode,"--checkpoint-write")) {
        auto archive=SourceArchive::create(path,config(),checkpoint_limits());
        const auto key=checkpoint_source(archive,f,queue);write_client_key(path,key);
        auto write=archive.save_checkpoint(queue,f.store,f.ref,checkpoint_input(ram));write.wait();archive.publish_checkpoint(write,f.store);return;
    }
    auto archive=SourceArchive::open(path,config(),checkpoint_limits());const auto key=client_key(path);
    auto source=archive.read(key,f.staging);rows_match(source.image(),f.rows);
    if (!std::strncmp(mode,"--checkpoint-crash-",19)) {
        const std::string action=mode+19;
        if (action.rfind("publish",0)==0 || action=="write") {
            const auto imported=archive.import_source(source,f.store);
            if (action=="write") {
                fault::Scope crash(fault::Crash,0);
                auto write=archive.save_checkpoint(queue,f.store,imported,checkpoint_input(ram));write.wait();
            } else {
                auto write=archive.save_checkpoint(queue,f.store,imported,checkpoint_input(ram));write.wait();
                fault::Scope crash(fault::Crash,size_t(action.back()-'0'));archive.publish_checkpoint(write,f.store);
            }
        } else {
            CHECK(action.rfind("retire",0)==0);
            fault::Scope crash(fault::Crash,size_t(action.back()-'0'));archive.retire_checkpoint(key);
        }
        CHECK(false);
    }
    if (!std::strcmp(mode,"--checkpoint-read")) {(void)checkpoint_read(archive,source);return;}
    if (!std::strcmp(mode,"--checkpoint-retire")) {
        auto read=checkpoint_read(archive,source);CHECK(archive.retire_checkpoint(key));
        rejects(MemoryErrorCode::Stale,[&]{archive.validate(read);});return;
    }
    CHECK(!std::strcmp(mode,"--checkpoint-deny"));
    rejects(MemoryErrorCode::Stale,[&]{(void)checkpoint_read(archive,source);});
    const auto imported=archive.import_source(source,f.store);CHECK(!f.store.state(imported).computed_frontier);
}
void checkpoint_exec_and_source_only_reopen() {
    Temp temp;
    for (const auto mode:{"--checkpoint-write","--checkpoint-retire","--checkpoint-deny"}) {
        const auto child=::fork();CHECK(child>=0);
        if (!child){::execl("/proc/self/exe","source_archive_test",mode,temp.path().c_str(),static_cast<char *>(nullptr));::_exit(74);}
        int result=0;CHECK(::waitpid(child,&result,0)==child && WIFEXITED(result) && WEXITSTATUS(result)==0);
    }
    {auto archive=SourceArchive::open(temp.path(),config(),checkpoint_limits());archive.collect();}
    CHECK(std::filesystem::file_size(temp.path()/"checkpoint-retired.1")==0);
    {auto archive=SourceArchive::open(temp.path(),config(),checkpoint_limits());Fixture f;
     auto source=archive.read(client_key(temp.path()),f.staging);
     rejects(MemoryErrorCode::Stale,[&]{(void)checkpoint_read(archive,source);});}
    // The optional derived-state feature does not change the v1 source format.
    auto data_only=SourceArchive::open(temp.path(),config(),limits());Fixture f;
    auto source=data_only.read(client_key(temp.path()),f.staging);rows_match(source.image(),f.rows);
    rejects(MemoryErrorCode::Unsupported,[&]{(void)checkpoint_read(data_only,source);});
    CHECK(data_only.forget(source.key())==1);
    std::puts("checkpoint separate exec: write/read-retire/reopen-deny, source-data fallback PASS");
}
void checkpoint_actual_io_failure_and_crash_boundaries() {
    Temp temp;size_t points=0;
    {
        Fixture f;HostKvStorage ram(65536);StorageIoQueue queue(1,1<<20);
        auto archive=SourceArchive::create(temp.path("count"),config(),checkpoint_limits());checkpoint_source(archive,f,queue);
        {fault::Scope count(fault::Count);auto write=archive.save_checkpoint(queue,f.store,f.ref,checkpoint_input(ram));write.wait();points=fault::calls.load();}
    }
    CHECK(points>0 && points<64);
    for (size_t point=0;point<points;++point) {
        Fixture f;HostKvStorage ram(65536);StorageIoQueue queue(1,1<<20);
        auto archive=SourceArchive::create(temp.root/("checkpoint-io-"+std::to_string(point)),config(),checkpoint_limits());
        const auto key=checkpoint_source(archive,f,queue);const auto charged=archive.stats().object_bytes;
        {
            fault::Scope error(fault::Error,point);auto write=archive.save_checkpoint(queue,f.store,f.ref,checkpoint_input(ram));
            io_failure([&]{write.wait();});CHECK(fault::fired);
        }
        CHECK(!ram.charged_bytes() && archive.stats().healthy && f.store.state(f.ref).accepted_frontier==513);
        {auto source=archive.read(key,f.staging);rejects(MemoryErrorCode::Stale,[&]{(void)checkpoint_read(archive,source);});}
        archive.collect();CHECK(archive.stats().object_bytes==charged);
    }
    for (size_t point=0;point<2;++point) {
        const auto path=temp.root/("checkpoint-publish-"+std::to_string(point));SourceArchiveKey key;
        {
            Fixture f;HostKvStorage ram(65536);StorageIoQueue queue(1,1<<20);
            auto archive=SourceArchive::create(path,config(),checkpoint_limits());key=checkpoint_source(archive,f,queue);
            auto write=archive.save_checkpoint(queue,f.store,f.ref,checkpoint_input(ram));write.wait();
            {fault::Scope error(fault::Error,point);io_failure([&]{archive.publish_checkpoint(write,f.store);});CHECK(fault::fired);}
            CHECK(!archive.stats().healthy);
        }
        auto reopened=SourceArchive::open(path,config(),checkpoint_limits());Fixture f;auto source=reopened.read(key,f.staging);
        if(point==0) rejects(MemoryErrorCode::Stale,[&]{(void)checkpoint_read(reopened,source);});
        else (void)checkpoint_read(reopened,source); // rename committed but acknowledgement failed, never reported as success
    }
    for (const auto mode:{"--checkpoint-crash-write","--checkpoint-crash-publish0","--checkpoint-crash-publish1",
                           "--checkpoint-crash-retire0","--checkpoint-crash-retire1"}) {
        const auto path=temp.root/mode;bool retire=std::strstr(mode,"retire")!=nullptr;
        {
            Fixture f;HostKvStorage ram(65536);StorageIoQueue queue(1,1<<20);
            auto archive=SourceArchive::create(path,config(),checkpoint_limits());const auto key=checkpoint_source(archive,f,queue);write_client_key(path,key);
            if (retire) {auto write=archive.save_checkpoint(queue,f.store,f.ref,checkpoint_input(ram));write.wait();archive.publish_checkpoint(write,f.store);}
        }
        auto child=::fork();CHECK(child>=0);
        if(!child){::execl("/proc/self/exe","source_archive_test",mode,path.c_str(),static_cast<char *>(nullptr));::_exit(74);}
        int result=0;CHECK(::waitpid(child,&result,0)==child && WIFEXITED(result) && WEXITSTATUS(result)==71);
        child=::fork();CHECK(child>=0);
        const char * check=retire || std::strstr(mode,"write") ? "--checkpoint-deny" : "--checkpoint-read";
        if(!child){::execl("/proc/self/exe","source_archive_test",check,path.c_str(),static_cast<char *>(nullptr));::_exit(74);}
        CHECK(::waitpid(child,&result,0)==child && WIFEXITED(result) && WEXITSTATUS(result)==0);
    }
    std::printf("checkpoint actual worker I/O error points=%zu publication error points=2 process-crash boundaries=5 PASS\n",points);
}
void checkpoint_foreground_allocation_and_queue_rollback() {
    Temp temp;size_t failures=0;
    for(size_t point=0;point<32;++point) {
        Fixture f;HostKvStorage ram(65536);StorageIoQueue queue(1,1<<20);
        auto archive=SourceArchive::create(temp.root/("checkpoint-alloc-"+std::to_string(point)),config(),checkpoint_limits());
        checkpoint_source(archive,f,queue);const auto charged=archive.stats().object_bytes;
        auto input=checkpoint_input(ram);bool failed=false;std::optional<SourceArchiveCheckpointWrite> write;
        allocation::calls=0;allocation::fail_at=point;
        try {write.emplace(archive.save_checkpoint(queue,f.store,f.ref,std::move(input)));}
        catch(const std::bad_alloc &){failed=true;}
        allocation::fail_at=SIZE_MAX;
        if(!failed){write->wait();break;}
        ++failures;CHECK(archive.stats().healthy && !ram.charged_bytes());archive.collect();
        CHECK(archive.stats().object_bytes==charged && f.store.state(f.ref).accepted_frontier==513);
    }
    CHECK(failures>0 && failures<32);
    {
        Fixture f;HostKvStorage ram(65536);StorageIoQueue queue(1,1<<20);
        auto archive=SourceArchive::create(temp.path(),config(),checkpoint_limits());checkpoint_source(archive,f,queue);
        const auto charged=archive.stats().object_bytes;Gate gate;auto occupied=gate.occupy(queue);
        rejects(MemoryErrorCode::Pending,[&]{(void)archive.save_checkpoint(queue,f.store,f.ref,checkpoint_input(ram));});
        CHECK(!ram.charged_bytes() && archive.stats().healthy);
        archive.collect();CHECK(archive.stats().object_bytes==charged);
        gate.released.set_value();occupied.wait();
    }
    std::printf("checkpoint foreground allocation failure points=%zu queue refusal rollback PASS\n",failures);
}
void checkpoint_disk_bounds_and_midread_revocation() {
    Temp temp;uint64_t control=0;
    {auto probe=SourceArchive::create(temp.path("probe"),config(),checkpoint_limits());control=probe.stats().control_reserved;}
    for (const bool disk:{false,true}) {
        Fixture f;HostKvStorage ram(65536);StorageIoQueue queue(1,1<<20);auto bounds=checkpoint_limits();
        if(disk) bounds.disk_bytes=control+source_image_bytes(f.store.state(f.ref),bounds.image)+checkpoint_payload+103;
        else bounds.max_checkpoint_bytes=checkpoint_payload+103;
        auto archive=SourceArchive::create(temp.path(disk ? "disk" : "frame"),config(),bounds);
        const auto key=checkpoint_source(archive,f,queue);const auto before=archive.stats();
        rejects(MemoryErrorCode::BudgetExceeded,[&]{(void)archive.save_checkpoint(queue,f.store,f.ref,checkpoint_input(ram));});
        CHECK(!ram.charged_bytes() && archive.stats().charged==before.charged && archive.stats().objects==before.objects);
        auto source=archive.read(key,f.staging);rows_match(source.image(),f.rows);
    }
    for(const bool forget:{false,true}) {
        Fixture f;HostKvStorage ram(65536);StorageIoQueue queue(1,1<<20);
        auto archive=SourceArchive::create(temp.path(forget ? "mid-forget" : "mid-retire"),config(),checkpoint_limits());
        const auto key=checkpoint_source(archive,f,queue);
        auto write=archive.save_checkpoint(queue,f.store,f.ref,checkpoint_input(ram));write.wait();archive.publish_checkpoint(write,f.store);
        {
            auto source=archive.read(key,f.staging);bool revoked=false;size_t copied=0;
            rejects(MemoryErrorCode::Stale,[&]{
                (void)archive.read_checkpoint(source,checkpoint_schema(),[&](const void *,size_t n) {
                    copied+=n;
                    if(!revoked){revoked=true;if(forget) archive.forget(key);else archive.retire_checkpoint(key);}
                });
            });
            CHECK(revoked && copied==checkpoint_payload && !archive.collect().quota_released);
            if(!forget) archive.validate(source);
        }
        write={};CHECK(archive.collect().quota_released>=checkpoint_payload+104);
    }
    SourceArchiveKey key;uint64_t bytes=0;
    {
        Fixture f;HostKvStorage ram(65536);StorageIoQueue queue(1,1<<20);
        auto archive=SourceArchive::create(temp.path("data-only"),config(),checkpoint_limits());key=checkpoint_source(archive,f,queue);
        auto write=archive.save_checkpoint(queue,f.store,f.ref,checkpoint_input(ram));write.wait();archive.publish_checkpoint(write,f.store);
        bytes=archive.stats().object_bytes;
    }
    {
        auto bounds=checkpoint_limits();bounds.max_checkpoint_bytes=checkpoint_payload+103;
        auto archive=SourceArchive::open(temp.path("data-only"),config(),bounds);Fixture f;
        auto source=archive.read(key,f.staging);
        rejects(MemoryErrorCode::BudgetExceeded,[&]{(void)checkpoint_read(archive,source);});
        CHECK(archive.stats().healthy);rows_match(source.image(),f.rows);
    }
    {
        auto archive=SourceArchive::open(temp.path("data-only"),config(),limits());Fixture f;
        CHECK(archive.stats().object_bytes==bytes);auto source=archive.read(key,f.staging);rows_match(source.image(),f.rows);
        rejects(MemoryErrorCode::Unsupported,[&]{(void)checkpoint_read(archive,source);});
    }
}
}
int main(int argc, char ** argv) {
    if (argc == 3) { restart_child(argv[1],argv[2]); return 0; }
    roundtrip_reopen_and_lock(); versions_and_reader_retention(); forget_family_late_writer_and_reuse();
    cancel_queue_quota_and_local_forget(); receipts_owners_and_survivors(); restart_orphans_and_corruption();
    rollback_and_missing_authority(); error_and_process_crash_boundaries(); actual_save_io_failures(); allocation_failures();
    shared_metadata_and_cleanup_retry();
    import_receipt_version_and_family(); import_allocation_rollback(); separate_exec_restart();
    owner_queries_and_structural_queue_limit();
    current_family_cooldown_rebind_and_old_receipts();
    cooldown_latest_copy_and_file_failures_are_atomic();
    checkpoint_publication_and_owned_lifetimes();checkpoint_versions_retirement_and_corruption();
    checkpoint_bounds_producer_failures_and_late_writer();checkpoint_exec_and_source_only_reopen();
    checkpoint_actual_io_failure_and_crash_boundaries();checkpoint_foreground_allocation_and_queue_rollback();
    checkpoint_disk_bounds_and_midread_revocation();
    std::puts("source archive lifecycle, quotas, durable authority and failure boundaries PASS");
}
