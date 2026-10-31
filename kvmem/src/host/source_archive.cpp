#include "kvmem/source_archive.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <limits>
#include <sys/file.h>
#include <sys/random.h>
#include <sys/stat.h>
#include <system_error>
#include <unistd.h>

namespace kvmem {
namespace {
constexpr uint64_t catalog_magic = 0x3143524153564bULL; // KVSARC1
constexpr uint64_t seal_magic = 0x314c414553564bULL; // KVSEAL1
constexpr uint64_t version = 1, record_bytes = 120, header_bytes = 112, seal_bytes = 64;
constexpr size_t absent = SIZE_MAX;
[[noreturn]] void io_error(const char * what) { throw std::system_error(errno, std::generic_category(), what); }
struct File {
    int fd = -1;
    explicit File(int value = -1) : fd(value) {}
    ~File() { if (fd >= 0) ::close(fd); }
    File(const File &) = delete;
    File & operator=(const File &) = delete;
};
void write_all(int fd, const void * input, size_t n) {
    auto p = static_cast<const uint8_t *>(input);
    while (n) {
        auto done = ::write(fd, p, n);
        if (done < 0 && errno == EINTR) continue;
        if (done <= 0) { if (!done) errno = EIO; io_error("write source archive"); }
        p += done; n -= size_t(done);
    }
}
void read_all(int fd, void * output, size_t n) {
    auto p = static_cast<uint8_t *>(output);
    while (n) {
        auto done = ::read(fd, p, n);
        if (done < 0 && errno == EINTR) continue;
        if (done < 0) io_error("read source archive");
        if (!done) throw SnapshotCorrupt("truncated source archive file");
        p += done; n -= size_t(done);
    }
}
void sync_file(int fd) {
    while (::fsync(fd) != 0) { if (errno != EINTR) io_error("sync source archive"); }
}
void close_file(File & file) {
    const int fd = std::exchange(file.fd, -1);
    if (::close(fd) != 0) io_error("close source archive"); // Linux: do not retry close.
}
int open_file(int directory, const char * name, int flags, mode_t mode = 0600) {
    int fd = ::openat(directory, name, flags | O_CLOEXEC | O_NOFOLLOW, mode);
    if (fd < 0) io_error("open source archive file");
    return fd;
}
uint64_t file_size(int fd) {
    struct stat value{};
    if (::fstat(fd, &value) != 0) io_error("stat source archive file");
    if (!S_ISREG(value.st_mode) || value.st_size < 0) throw SnapshotCorrupt("invalid archive file kind");
    return uint64_t(value.st_size);
}
void put(SnapshotWriter & out, uint64_t value) {
    uint8_t bytes[8];
    for (size_t i = 0; i < 8; ++i) bytes[i] = uint8_t(value >> (8*i));
    out.write(bytes, 8);
}
uint64_t get(SnapshotReader & in) {
    uint8_t bytes[8]; in.read(bytes, 8); uint64_t value = 0;
    for (size_t i = 0; i < 8; ++i) value |= uint64_t(bytes[i]) << (8*i);
    return value;
}
void put_ref(SnapshotWriter & out, SourceRef value) {
    put(out, value.source.id); put(out, value.source.incarnation); put(out, value.revision);
}
SourceRef get_ref(SnapshotReader & in) { return {{get(in), get(in)}, get(in)}; }
bool same_origin(const SourceState & a, const SourceState & b) {
    return a.ref == b.ref && a.configuration == b.configuration && a.parent == b.parent &&
        a.parent_prefix == b.parent_prefix && a.accepted_frontier == b.accepted_frontier &&
        a.computed_frontier == b.computed_frontier && a.events == b.events;
}
std::string blob_name(uint64_t id) { return "blob." + std::to_string(id); }
uint64_t blob_id(const char * name) {
    if (std::strncmp(name, "blob.", 5) || !name[5] || name[5] == '0') return 0;
    uint64_t id = 0;
    for (auto p = name + 5; *p; ++p) {
        if (*p < '0' || *p > '9' || id > (UINT64_MAX - uint64_t(*p-'0'))/10) return 0;
        id = id*10 + uint64_t(*p-'0');
    }
    return id;
}
}

namespace detail {
struct SourceArchiveState {
    struct Entry {
        uint64_t id = 0, parent = 0, blob = 0, bytes = 0, hash = 0;
        SourceState origin;
        const SourceJournalStore * bound_store = nullptr;
        SessionIdentity bound_source;
    };
    struct Object {
        uint64_t id = 0, bytes = 0, source = 0, hash = 0;
        uint64_t removed_bytes = 0;
        bool unlinked = false;
        SourceState origin;
        StorageTicket ticket;
        std::atomic<size_t> handles{0};
    };
    SourceArchiveLimits limits;
    SourceConfiguration configuration;
    std::array<uint8_t, 16> space{};
    File directory, lock;
    std::unique_ptr<HistoricalKvAuxArray<Entry>> entries_storage, scratch_storage;
    std::unique_ptr<HistoricalKvAuxArray<Object>> objects_storage;
    Entry * entries = nullptr, * scratch = nullptr;
    Object * objects = nullptr;
    uint64_t sequence = 0, next_source = 1, next_blob = 1;
    uint64_t catalog_bytes = 0, control = 0, metadata = 0, object_bytes = 0, peak = 0;
    bool healthy = true;

    SourceArchiveState(SourceArchiveLimits bounds, SourceConfiguration config, std::shared_ptr<HistoricalKvAuxBudget> auxiliary)
        : limits(bounds), configuration(config) {
        if (!limits.max_sources || !limits.max_objects || limits.max_objects < limits.max_sources ||
            limits.max_sources > (UINT64_MAX - header_bytes)/record_bytes ||
            limits.max_sources > SIZE_MAX/(2*sizeof(Entry)) || limits.max_objects > SIZE_MAX/sizeof(Object) ||
            !std::any_of(config.begin(), config.end(), [](uint8_t x) { return x != 0; }) ||
            limits.image.max_bytes < 144 || !limits.image.max_tokens || !limits.image.max_events)
            throw MemoryError(MemoryErrorCode::InvalidPlan, "invalid source archive limits/configuration");
        const auto entry_cost = uint64_t(limits.max_sources) * 2 * sizeof(Entry);
        const auto object_cost = uint64_t(limits.max_objects) * sizeof(Object);
        if (object_cost > UINT64_MAX - entry_cost)
            throw MemoryError(MemoryErrorCode::InvalidPlan, "source archive metadata overflow");
        metadata = entry_cost + object_cost;
        catalog_bytes = header_bytes + record_bytes*limits.max_sources;
        if (catalog_bytes > (UINT64_MAX - 2*seal_bytes)/2)
            throw MemoryError(MemoryErrorCode::InvalidPlan, "source archive control size overflow");
        control = 2*catalog_bytes + 2*seal_bytes;
        if (metadata > limits.metadata_bytes || control > limits.disk_bytes)
            throw MemoryError(MemoryErrorCode::BudgetExceeded, "source archive fixed tables/control exceed quota");
        entries_storage = std::make_unique<HistoricalKvAuxArray<Entry>>(limits.max_sources, auxiliary, HistoricalKvAuxKind::Metadata);
        scratch_storage = std::make_unique<HistoricalKvAuxArray<Entry>>(limits.max_sources, auxiliary, HistoricalKvAuxKind::Metadata);
        objects_storage = std::make_unique<HistoricalKvAuxArray<Object>>(limits.max_objects, auxiliary, HistoricalKvAuxKind::Metadata);
        entries = &(*entries_storage)[0]; scratch = &(*scratch_storage)[0]; objects = &(*objects_storage)[0];
        peak = control;
    }
    void check() const {
        if (!healthy) throw MemoryError(MemoryErrorCode::ServiceUnavailable, "source archive requires a verified reopen");
    }
    SourceArchiveKey key(uint64_t id) const { return {space, id}; }
    size_t find(SourceArchiveKey value) const {
        check();
        if (!value.id || value.space != space)
            throw MemoryError(MemoryErrorCode::Stale, "foreign/empty archive identity");
        for (size_t i = 0; i < limits.max_sources; ++i) if (entries[i].id == value.id) return i;
        throw MemoryError(MemoryErrorCode::Stale, "archive source no longer authorized");
    }
    size_t binding(const SourceJournalStore & store, SessionIdentity source) const {
        for (size_t i = 0; i < limits.max_sources; ++i)
            if (entries[i].id && entries[i].bound_store == &store && entries[i].bound_source == source) return i;
        return absent;
    }
    SourceState current(const SourceJournalStore & store, SessionIdentity source) const {
        for (size_t i = 0; i < store.slots(); ++i) {
            auto entry = store.inspect(i);
            if (entry && entry->ref.source == source) return *entry;
        }
        throw MemoryError(MemoryErrorCode::Stale, "source disappeared before archive publication");
    }
    void copy_entries() { std::copy_n(entries, limits.max_sources, scratch); }
    bool referenced(uint64_t id) const {
        for (size_t i = 0; i < limits.max_sources; ++i) if (entries[i].blob == id) return true;
        return false;
    }
    size_t object_slot() const {
        for (size_t i = 0; i < limits.max_objects; ++i) if (!objects[i].id) return i;
        throw MemoryError(MemoryErrorCode::BudgetExceeded, "archive object/lease table is full");
    }
    uint64_t encode_catalog(int fd, uint64_t seq, uint64_t source_high, uint64_t blob_high) {
        SnapshotWriter out([&](const void * p, size_t n) { write_all(fd, p, n); });
        put(out, catalog_magic); put(out, version); put(out, catalog_bytes); out.write(space.data(), space.size());
        put(out, seq); put(out, source_high); put(out, blob_high); put(out, limits.max_sources);
        out.write(configuration.data(), configuration.size());
        for (size_t i = 0; i < limits.max_sources; ++i) {
            const auto & e = scratch[i];
            put(out, e.id); put(out, e.parent); put(out, e.blob); put(out, e.bytes); put(out, e.hash);
            put_ref(out, e.origin.ref); put_ref(out, e.origin.parent); put(out, e.origin.parent_prefix);
            put(out, e.origin.accepted_frontier); put(out, e.origin.computed_frontier); put(out, e.origin.events);
        }
        const auto hash = out.hash(); put(out, hash);
        if (out.bytes() != catalog_bytes) throw std::logic_error("archive catalog size mismatch");
        return hash;
    }
    void encode_seal(int fd, uint64_t seq, uint64_t hash) {
        SnapshotWriter out([&](const void * p, size_t n) { write_all(fd, p, n); });
        put(out, seal_magic); put(out, version); out.write(space.data(), space.size());
        put(out, seq); put(out, hash); put(out, catalog_bytes); put(out, out.hash());
        if (out.bytes() != seal_bytes) throw std::logic_error("archive seal size mismatch");
    }
    // An ambiguous metadata failure poisons this owner. Reopen requires BOTH
    // catalog and seal to agree; it never chooses an older valid-looking copy.
    void commit(uint64_t source_high, uint64_t blob_high) {
        check();
        if (sequence == UINT64_MAX) throw MemoryError(MemoryErrorCode::BudgetExceeded, "archive sequence exhausted");
        try {
            File catalog(open_file(directory.fd, "catalog.tmp", O_WRONLY | O_CREAT | O_TRUNC));
            const auto hash = encode_catalog(catalog.fd, sequence+1, source_high, blob_high);
            sync_file(catalog.fd); close_file(catalog);
            if (::renameat(directory.fd, "catalog.tmp", directory.fd, "catalog") != 0) io_error("publish archive catalog");
            sync_file(directory.fd);
            File seal(open_file(directory.fd, "seal.tmp", O_WRONLY | O_CREAT | O_TRUNC));
            encode_seal(seal.fd, sequence+1, hash); sync_file(seal.fd); close_file(seal);
            if (::renameat(directory.fd, "seal.tmp", directory.fd, "seal") != 0) io_error("publish archive seal");
            sync_file(directory.fd);
            entries_storage.swap(scratch_storage); std::swap(entries, scratch);
            ++sequence; next_source = source_high; next_blob = blob_high;
        } catch (...) { healthy = false; throw; }
    }
    void load() {
        File catalog(open_file(directory.fd, "catalog", O_RDONLY));
        if (file_size(catalog.fd) != catalog_bytes) throw SnapshotCorrupt("archive catalog geometry mismatch");
        SnapshotReader in([&](void * p, size_t n) { read_all(catalog.fd, p, n); }, catalog_bytes);
        if (get(in) != catalog_magic || get(in) != version || get(in) != catalog_bytes)
            throw SnapshotCorrupt("incompatible archive catalog");
        in.read(space.data(), space.size()); sequence = get(in); next_source = get(in); next_blob = get(in);
        if (!sequence || !next_source || !next_blob || get(in) != limits.max_sources ||
            !std::any_of(space.begin(), space.end(), [](uint8_t x) { return x != 0; }))
            throw SnapshotCorrupt("invalid archive identity/high-water mark");
        SourceConfiguration config{}; in.read(config.data(), config.size());
        if (config != configuration) throw MemoryError(MemoryErrorCode::InvalidPlan, "archive configuration mismatch");
        for (size_t i = 0; i < limits.max_sources; ++i) {
            auto & e = entries[i];
            e.id = get(in); e.parent = get(in); e.blob = get(in); e.bytes = get(in); e.hash = get(in);
            e.origin.ref = get_ref(in); e.origin.parent = get_ref(in); e.origin.parent_prefix = get(in);
            e.origin.accepted_frontier = get(in); e.origin.computed_frontier = get(in); e.origin.events = get(in);
            if (!e.id) {
                if (e.parent || e.blob || e.bytes || e.hash || !same_origin(e.origin, SourceState{}))
                    throw SnapshotCorrupt("nonempty unused archive slot");
            } else {
                if (e.id >= next_source || e.parent >= e.id || e.blob >= next_blob)
                    throw SnapshotCorrupt("archive ID exceeds committed high-water mark");
                if (e.blob) {
                    e.origin.configuration = configuration;
                    if (source_image_bytes(e.origin, limits.image) != e.bytes)
                        throw SnapshotCorrupt("archive image geometry mismatch");
                } else if (e.bytes || e.hash || !same_origin(e.origin, SourceState{}))
                    throw SnapshotCorrupt("unpublished source has snapshot metadata");
            }
        }
        const auto hash = in.hash();
        if (get(in) != hash || in.remaining()) throw SnapshotCorrupt("archive catalog checksum mismatch");
        File seal(open_file(directory.fd, "seal", O_RDONLY));
        if (file_size(seal.fd) != seal_bytes) throw SnapshotCorrupt("archive seal geometry mismatch");
        SnapshotReader seal_in([&](void * p, size_t n) { read_all(seal.fd, p, n); }, seal_bytes);
        if (get(seal_in) != seal_magic || get(seal_in) != version) throw SnapshotCorrupt("invalid archive seal");
        std::array<uint8_t,16> seal_space{}; seal_in.read(seal_space.data(), seal_space.size());
        if (seal_space != space || get(seal_in) != sequence || get(seal_in) != hash || get(seal_in) != catalog_bytes)
            throw SnapshotCorrupt("archive catalog/seal disagree: incomplete commit or rollback");
        const auto seal_hash = seal_in.hash();
        if (get(seal_in) != seal_hash || seal_in.remaining()) throw SnapshotCorrupt("archive seal checksum mismatch");
        for (size_t i = 0; i < limits.max_sources; ++i) if (entries[i].id) {
            bool parent_found = !entries[i].parent;
            for (size_t j = 0; j < limits.max_sources; ++j) {
                if (entries[j].id == entries[i].parent) parent_found = true;
                if (i != j && entries[i].id == entries[j].id) throw SnapshotCorrupt("duplicate archive identity");
                if (i != j && entries[i].blob && entries[i].blob == entries[j].blob)
                    throw SnapshotCorrupt("shared archive object identity");
            }
            if (!parent_found) throw SnapshotCorrupt("missing archive ancestor");
        }
    }
    void scan_objects() {
        // A fresh directory description avoids sharing readdir's offset with
        // the durable directory descriptor. Scan supplies bytes, NEVER rights.
        File scan_fd(open_file(directory.fd, ".", O_RDONLY | O_DIRECTORY));
        DIR * raw = ::fdopendir(scan_fd.fd);
        if (!raw) io_error("enumerate archive objects");
        scan_fd.fd = -1;
        std::unique_ptr<DIR, int(*)(DIR*)> scan(raw, &::closedir);
        errno = 0;
        while (auto item = ::readdir(scan.get())) {
            const auto name = item->d_name;
            if (!std::strcmp(name, ".") || !std::strcmp(name, "..") || !std::strcmp(name, "lock") ||
                !std::strcmp(name, "catalog") || !std::strcmp(name, "seal")) continue;
            File file(open_file(directory.fd, name, O_RDONLY));
            const auto bytes = file_size(file.fd);
            if (!std::strcmp(name, "catalog.tmp") || !std::strcmp(name, "seal.tmp")) {
                if (bytes > (!std::strcmp(name, "catalog.tmp") ? catalog_bytes : seal_bytes))
                    throw SnapshotCorrupt("oversized archive control temporary");
            } else {
                const auto id = blob_id(name);
                if (!id || id >= next_blob || bytes > limits.image.max_bytes)
                    throw SnapshotCorrupt("unknown or uncommitted archive file identity");
                if (bytes > limits.disk_bytes - control - object_bytes)
                    throw MemoryError(MemoryErrorCode::BudgetExceeded, "existing archive files exceed disk quota");
                auto & object = objects[object_slot()]; object.id = id; object.bytes = bytes; object_bytes += bytes;
            }
            errno = 0;
        }
        if (errno) io_error("enumerate archive directory");
        for (size_t i = 0; i < limits.max_sources; ++i) if (entries[i].blob) {
            bool found = false;
            for (size_t j = 0; j < limits.max_objects; ++j) if (objects[j].id == entries[i].blob) {
                if (objects[j].bytes != entries[i].bytes) throw SnapshotCorrupt("missing/truncated published archive object");
                objects[j].source = entries[i].id; objects[j].origin = entries[i].origin; objects[j].hash = entries[i].hash;
                found = true; break;
            }
            if (!found) throw SnapshotCorrupt("published archive object missing");
        }
        peak = control + object_bytes;
    }
};
} // namespace detail

namespace {
using State = detail::SourceArchiveState;
std::shared_ptr<State> make_state(const std::filesystem::path & path, SourceConfiguration config,
    SourceArchiveLimits limits, bool create, std::shared_ptr<HistoricalKvAuxBudget> auxiliary) {
    auto state = std::make_shared<State>(limits, config, std::move(auxiliary));
    if (create) {
        if (::mkdir(path.c_str(), 0700) != 0) io_error("create new source archive directory");
        auto parent = path.parent_path(); if (parent.empty()) parent = ".";
        File parent_fd(::open(parent.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC));
        if (parent_fd.fd < 0) io_error("open archive parent directory");
        sync_file(parent_fd.fd);
    }
    state->directory.fd = ::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
    if (state->directory.fd < 0) io_error("open source archive directory");
    state->lock.fd = open_file(state->directory.fd, "lock", O_RDWR | (create ? O_CREAT | O_EXCL : 0));
    if (::flock(state->lock.fd, LOCK_EX | LOCK_NB) != 0)
        throw MemoryError(MemoryErrorCode::Pending, "source archive already has an owner");
    if (file_size(state->lock.fd)) throw SnapshotCorrupt("invalid archive lock file");
    if (create) {
        size_t offset = 0;
        while (offset < state->space.size()) {
            auto n = ::getrandom(state->space.data()+offset, state->space.size()-offset, 0);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) { if (!n) errno = EIO; io_error("create archive namespace"); }
            offset += size_t(n);
        }
        if (!std::any_of(state->space.begin(), state->space.end(), [](uint8_t x) { return x != 0; }))
            throw MemoryError(MemoryErrorCode::ServiceUnavailable, "empty generated archive namespace");
        state->commit(1, 1);
    } else { state->load(); state->scan_objects(); }
    return state;
}
}

SourceArchive SourceArchive::create(const std::filesystem::path & path, const SourceConfiguration & config, SourceArchiveLimits limits,
    std::shared_ptr<HistoricalKvAuxBudget> auxiliary) {
    return SourceArchive(make_state(path, config, limits, true, std::move(auxiliary)));
}
SourceArchive SourceArchive::open(const std::filesystem::path & path, const SourceConfiguration & config, SourceArchiveLimits limits,
    std::shared_ptr<HistoricalKvAuxBudget> auxiliary) {
    return SourceArchive(make_state(path, config, limits, false, std::move(auxiliary)));
}
SourceArchiveKey SourceArchive::enroll(const SourceJournalStore & store, SourceRef ref) {
    auto & s = *state_; s.check(); const auto current = store.state(ref);
    if (current.configuration != s.configuration)
        throw MemoryError(MemoryErrorCode::InvalidPlan, "source/archive configuration mismatch");
    const auto existing = s.binding(store, ref.source);
    if (existing != absent) return s.key(s.entries[existing].id);
    uint64_t parent = 0;
    if (current.parent.source.id) {
        auto index = s.binding(store, current.parent.source);
        if (index == absent) throw MemoryError(MemoryErrorCode::InvalidPlan, "enroll archive ancestor first");
        parent = s.entries[index].id;
    }
    size_t slot = absent;
    for (size_t i = 0; i < s.limits.max_sources; ++i) if (!s.entries[i].id) { slot = i; break; }
    if (slot == absent || s.next_source == UINT64_MAX)
        throw MemoryError(MemoryErrorCode::BudgetExceeded, "archive source table/identity exhausted");
    s.copy_entries(); auto & e = s.scratch[slot];
    e.id = s.next_source; e.parent = parent; e.bound_store = &store; e.bound_source = ref.source;
    const auto id = e.id; s.commit(s.next_source+1, s.next_blob); return s.key(id);
}
SourceArchiveWrite SourceArchive::save(StorageIoQueue & queue, const SourceJournalStore & store, SourceRef ref, HostKvStorage & staging) {
    auto & s = *state_; s.check(); const auto source = store.state(ref);
    const auto entry = s.binding(store, ref.source);
    if (entry == absent || source.configuration != s.configuration)
        throw MemoryError(MemoryErrorCode::Stale, "source is not enrolled by this archive owner");
    const auto bytes = source_image_bytes(source, s.limits.image);
    if (bytes > s.limits.disk_bytes - s.control - s.object_bytes || s.next_blob == UINT64_MAX)
        throw MemoryError(MemoryErrorCode::BudgetExceeded, "archive disk quota/identity exhausted");
    const auto slot = s.object_slot();
    auto image = std::make_shared<SourceImage>(prepare_source_image(store, ref, s.limits.image, staging));
    // Consume the durable identity BEFORE any worker may create its file.
    s.copy_entries(); const auto id = s.next_blob; s.commit(s.next_source, id+1);
    auto & object = s.objects[slot]; object.id = id; object.bytes = bytes;
    object.source = s.entries[entry].id; object.origin = image->origin();
    s.object_bytes += bytes; s.peak = std::max(s.peak, s.control+s.object_bytes);
    // Queue refusal/allocation failure leaves an unpublished charged slot for
    // collect(); no file is granted publication rights and S remains intact.
    object.ticket = queue.submit(image->staging_bytes(), [owner = state_, image, slot] {
        auto & object = owner->objects[slot];
        const auto name = blob_name(object.id);
        File file(open_file(owner->directory.fd, name.c_str(), O_WRONLY | O_CREAT | O_EXCL));
        uint64_t hash = 14695981039346656037ULL;
        const auto written = write_source_image(*image, owner->limits.image, [&](const void * p, size_t n) {
            write_all(file.fd, p, n);
            auto bytes = static_cast<const uint8_t *>(p);
            for (size_t i = 0; i < n; ++i) hash = (hash ^ bytes[i])*1099511628211ULL;
        });
        if (written != object.bytes) throw std::logic_error("archive blob size mismatch");
        sync_file(file.fd); close_file(file); sync_file(owner->directory.fd); object.hash = hash;
    });
    return SourceArchiveWrite(state_, slot);
}
void SourceArchive::publish(const SourceArchiveWrite & write, const SourceJournalStore & store) {
    auto & s = *state_; s.check();
    if (write.state_.get() != state_.get()) throw MemoryError(MemoryErrorCode::Stale, "foreign archive write receipt");
    auto & object = s.objects[write.slot_];
    if (object.ticket.status() == StorageTaskStatus::Pending) throw MemoryError(MemoryErrorCode::Pending, "archive write still running");
    object.ticket.wait();
    const auto index = s.find(s.key(object.source)); auto & e = s.entries[index];
    if (e.bound_store != &store || !(e.bound_source == object.origin.ref.source))
        throw MemoryError(MemoryErrorCode::Stale, "archive source binding changed");
    const auto current = s.current(store, e.bound_source);
    if (current.configuration != s.configuration || current.ref.revision < object.origin.ref.revision ||
        current.accepted_frontier < object.origin.accepted_frontier)
        throw MemoryError(MemoryErrorCode::Stale, "archive source version invalidated");
    // IDs are committed monotonically before a CURRENT source snapshot is
    // queued. Use that durable order across restart/import: SourceRef counters
    // belong to their RAM store and may reset or collide in another process.
    if (e.blob == object.id) return;
    if (object.id < e.blob)
        throw MemoryError(MemoryErrorCode::Stale, "archive completion would regress saved version");
    s.copy_entries(); auto & next = s.scratch[index];
    next.blob = object.id; next.bytes = object.bytes; next.hash = object.hash; next.origin = object.origin;
    s.commit(s.next_source, s.next_blob);
}
SourceArchiveRead SourceArchive::read(SourceArchiveKey key, HostKvStorage & staging) {
    auto & s = *state_; const auto index = s.find(key); const auto & e = s.entries[index];
    if (!e.blob) throw MemoryError(MemoryErrorCode::Stale, "archive source has no committed image");
    size_t slot = absent;
    for (size_t i = 0; i < s.limits.max_objects; ++i) if (s.objects[i].id == e.blob) { slot = i; break; }
    if (slot == absent) throw SnapshotCorrupt("archive object ledger lost published file");
    auto name = blob_name(e.blob); File file(open_file(s.directory.fd, name.c_str(), O_RDONLY));
    if (file_size(file.fd) != e.bytes) throw SnapshotCorrupt("archive object size changed");
    uint64_t hash = 14695981039346656037ULL;
    auto image = read_source_image([&](void * p, size_t n) {
        read_all(file.fd, p, n); auto bytes = static_cast<const uint8_t *>(p);
        for (size_t i = 0; i < n; ++i) hash = (hash ^ bytes[i])*1099511628211ULL;
    }, e.bytes, s.configuration, s.limits.image, staging);
    if (hash != e.hash || !same_origin(image.origin(), e.origin))
        throw SnapshotCorrupt("archive image disagrees with authority");
    close_file(file); return SourceArchiveRead(state_, slot, std::move(image));
}
void SourceArchive::validate(const SourceArchiveRead & read) const {
    auto & s = *state_; s.check();
    if (read.state_.get() != state_.get()) throw MemoryError(MemoryErrorCode::Stale, "foreign archive read receipt");
    const auto & object = s.objects[read.slot_];
    const auto index = s.find(s.key(object.source));
    if (s.entries[index].blob != object.id) throw MemoryError(MemoryErrorCode::Stale, "archive read was replaced or forgotten");
}
SourceRef SourceArchive::import_source(const SourceArchiveRead & read, SourceJournalStore & store) {
    validate(read); auto & s = *state_;
    const auto index = s.find(read.key()); auto & entry = s.entries[index];
    if (entry.bound_store)
        throw MemoryError(MemoryErrorCode::Pending, "archive source already has a live owner binding");
    const auto imported = store.import_image(read.image());
    try { validate(read); }
    catch (...) { store.forget(imported); throw; }
    entry.bound_store = &store; entry.bound_source = imported.source;
    return imported;
}
std::optional<SourceArchiveSnapshot> SourceArchive::inspect(size_t slot) const {
    auto & s = *state_; s.check();
    if (slot >= s.limits.max_sources) throw MemoryError(MemoryErrorCode::InvalidPlan, "archive slot out of range");
    const auto & e = s.entries[slot];
    if (!e.id) return std::nullopt;
    return SourceArchiveSnapshot{s.key(e.id), e.parent ? s.key(e.parent) : SourceArchiveKey{}, e.blob, e.bytes, e.origin};
}
size_t SourceArchive::forget(SourceArchiveKey key) {
    auto & s = *state_; s.find(key); s.copy_entries(); size_t count = 0;
    for (size_t i = 0; i < s.limits.max_sources; ++i) if (s.entries[i].id) {
        auto id = s.entries[i].id;
        for (size_t depth = 0; id && depth <= s.limits.max_sources; ++depth) {
            if (id == key.id) { s.scratch[i] = {}; ++count; break; }
            const auto index = s.find(s.key(id)); id = s.entries[index].parent;
        }
    }
    s.commit(s.next_source, s.next_blob); return count;
}
SourceArchive::Collection SourceArchive::collect() {
    auto & s = *state_; s.check(); Collection result;
    for (size_t i = 0; i < s.limits.max_objects; ++i) {
        auto & object = s.objects[i];
        if (!object.id || s.referenced(object.id) || object.handles.load(std::memory_order_acquire) ||
            (object.ticket && !object.ticket.ready())) continue;
        auto name = blob_name(object.id);
        if (!object.unlinked) {
            struct stat value{};
            if (::fstatat(s.directory.fd, name.c_str(), &value, AT_SYMLINK_NOFOLLOW) != 0) {
                if (errno != ENOENT) io_error("stat orphan archive file");
            } else {
                if (!S_ISREG(value.st_mode) || value.st_size < 0 || uint64_t(value.st_size) > object.bytes) {
                    s.healthy = false; throw SnapshotCorrupt("orphan archive file exceeds its reservation");
                }
                if (::unlinkat(s.directory.fd, name.c_str(), 0) == 0) object.removed_bytes = uint64_t(value.st_size);
                else if (errno != ENOENT) io_error("remove orphan archive file");
            }
            object.unlinked = true;
        }
        sync_file(s.directory.fd);
        s.object_bytes -= object.bytes; result.quota_released += object.bytes;
        result.file_bytes_removed += object.removed_bytes; ++result.objects;
        object.ticket = {}; object.id = object.bytes = object.source = object.hash = 0; object.origin = {};
        object.removed_bytes = 0; object.unlinked = false;
    }
    return result;
}
SourceArchive::Stats SourceArchive::stats() const noexcept {
    const auto & s = *state_; Stats result;
    result.capacity = s.limits.disk_bytes; result.control_reserved = s.control;
    result.object_bytes = s.object_bytes; result.charged = s.control+s.object_bytes;
    result.peak = s.peak; result.metadata_bytes = s.metadata; result.healthy = s.healthy;
    for (size_t i = 0; i < s.limits.max_sources; ++i) if (s.entries[i].id) ++result.sources;
    for (size_t i = 0; i < s.limits.max_objects; ++i) if (s.objects[i].id) {
        ++result.objects;
        if (s.objects[i].handles.load(std::memory_order_relaxed)) ++result.leased_objects;
    }
    return result;
}

SourceArchiveWrite::SourceArchiveWrite(std::shared_ptr<State> state, size_t slot) : state_(std::move(state)), slot_(slot) {
    state_->objects[slot_].handles.fetch_add(1, std::memory_order_relaxed);
}
void SourceArchiveWrite::reset() noexcept {
    if (state_) state_->objects[slot_].handles.fetch_sub(1, std::memory_order_release);
    state_.reset();
}
SourceArchiveWrite::~SourceArchiveWrite() { reset(); }
SourceArchiveWrite::SourceArchiveWrite(SourceArchiveWrite && other) noexcept
    : state_(std::move(other.state_)), slot_(other.slot_) {}
SourceArchiveWrite & SourceArchiveWrite::operator=(SourceArchiveWrite && other) noexcept {
    if (this != &other) { reset(); state_ = std::move(other.state_); slot_ = other.slot_; } return *this;
}
StorageTaskStatus SourceArchiveWrite::status() const noexcept {
    return state_ ? state_->objects[slot_].ticket.status() : StorageTaskStatus::Cancelled;
}
void SourceArchiveWrite::wait() const {
    if (!state_) throw MemoryError(MemoryErrorCode::Stale, "empty archive write receipt");
    state_->objects[slot_].ticket.wait();
}
bool SourceArchiveWrite::cancel() const { return state_ && state_->objects[slot_].ticket.cancel(); }
SourceArchiveRead::SourceArchiveRead(std::shared_ptr<State> state, size_t slot, SourceImage image)
    : state_(std::move(state)), slot_(slot), image_(std::move(image)) {
    state_->objects[slot_].handles.fetch_add(1, std::memory_order_relaxed);
}
void SourceArchiveRead::reset() noexcept {
    if (state_) state_->objects[slot_].handles.fetch_sub(1, std::memory_order_release);
    state_.reset();
}
SourceArchiveRead::~SourceArchiveRead() { reset(); }
SourceArchiveRead::SourceArchiveRead(SourceArchiveRead && other) noexcept
    : state_(std::move(other.state_)), slot_(other.slot_), image_(std::move(other.image_)) {}
SourceArchiveRead & SourceArchiveRead::operator=(SourceArchiveRead && other) noexcept {
    if (this != &other) { reset(); image_ = std::move(other.image_); state_ = std::move(other.state_); slot_ = other.slot_; } return *this;
}
SourceArchiveKey SourceArchiveRead::key() const noexcept {
    return state_ ? state_->key(state_->objects[slot_].source) : SourceArchiveKey{};
}
} // namespace kvmem
