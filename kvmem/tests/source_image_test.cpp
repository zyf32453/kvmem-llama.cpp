#include "kvmem/source_image.hpp"
#include "kvmem/spill_io.hpp"
#include "memory_test_support.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <limits>
#include <new>
#include <numeric>

namespace {
thread_local size_t fail_at = SIZE_MAX, allocations = 0;
void allocation_fault() {
    if (fail_at != SIZE_MAX && allocations++ == fail_at) { fail_at = SIZE_MAX; throw std::bad_alloc(); }
}
}
void * operator new(size_t n) {
    allocation_fault();
    if (auto * p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void * operator new[](size_t n) { return ::operator new(n); }
void operator delete(void * p) noexcept { std::free(p); }
void operator delete[](void * p) noexcept { std::free(p); }
void operator delete(void * p, size_t) noexcept { std::free(p); }
void operator delete[](void * p, size_t) noexcept { std::free(p); }

using namespace kvmem;
namespace {
SourceConfiguration configuration() { SourceConfiguration c{}; c[0] = 7; c[31] = 211; return c; }
SourceImageLimits limits() { return {1 << 20, 8192, 256}; }
struct Fixture {
    SourceJournalStore store{{1 << 20, 8192, 256, 8}};
    std::vector<int32_t> rows;
    SourceRef ref;
    Fixture() : rows(513) {
        std::iota(rows.begin(), rows.end(), 0);
        rows[256] = 0x1020304; rows.back() = INT32_MAX;
        // Duplicate IDs and incomplete non-final events are legal source facts.
        const SourceEvent events[] = {{7, 0, 64, SourceEventKind::System, false},
            {7, 64, 257, SourceEventKind::ToolResult, true}, {9, 257, 513, SourceEventKind::Assistant, false}};
        ref = store.capture(configuration(), rows.data(), rows.size(), events, 3, 512);
    }
};
std::vector<uint8_t> encode(const SourceJournalStore & store, SourceRef ref, size_t * largest = nullptr) {
    std::vector<uint8_t> bytes;
    const auto count = write_source_image(store, ref, limits(), [&](const void * p, size_t n) {
        if (largest) *largest = std::max(*largest, n);
        auto data = static_cast<const uint8_t *>(p); bytes.insert(bytes.end(), data, data + n);
    });
    CHECK(count == bytes.size()); return bytes;
}
std::vector<uint8_t> encode(const SourceImage & image) {
    std::vector<uint8_t> bytes;
    const auto count = write_source_image(image, limits(), [&](const void * p, size_t n) {
        CHECK(n <= 1024);
        auto data = static_cast<const uint8_t *>(p); bytes.insert(bytes.end(), data, data + n);
    });
    CHECK(count == bytes.size()); return bytes;
}
SourceImage decode(const std::vector<uint8_t> & bytes, HostKvStorage & ram,
    SourceConfiguration expected = configuration(), SourceImageLimits bounds = limits()) {
    // Value-held mutable cursor tests the codec's single-reader callback lifetime.
    return read_source_image([&bytes, offset = size_t(0)](void * p, size_t n) mutable {
        if (n > bytes.size() - offset) throw std::runtime_error("transport truncated");
        std::memcpy(p, bytes.data() + offset, n); offset += n;
    }, bytes.size(), expected, bounds, ram);
}
template<class F> void corrupt(F action) {
    try { action(); } catch (const SnapshotCorrupt &) { return; }
    throw std::runtime_error("invalid image was accepted");
}
void little(std::vector<uint8_t> & bytes, size_t offset, uint64_t value, size_t count = 8) {
    for (size_t i = 0; i < count; ++i) bytes.at(offset+i) = uint8_t(value >> (8*i));
}
void rehash(std::vector<uint8_t> & bytes) {
    uint64_t hash = 14695981039346656037ULL;
    for (size_t i = 0; i < bytes.size()-8; ++i) hash = (hash ^ bytes[i])*1099511628211ULL;
    little(bytes, bytes.size()-8, hash);
}
void check_rows(const SourceImage & image, const std::vector<int32_t> & expected) {
    std::vector<int32_t> rows(expected.size()); image.read_tokens(0, rows.data(), rows.size());
    CHECK(rows == expected);
}
void roundtrip_and_file() {
    Fixture f; HostKvStorage ram(1 << 16); const auto before = f.store.stats();
    size_t largest = 0; const auto bytes = encode(f.store, f.ref, &largest);
    CHECK(largest <= 1024 && bytes.size() == 144 + 4*f.rows.size() + 26*3);
    const uint8_t prefix[] = {'K','V','S','I','M','G','1',0, 1,0,0,0,0,0,0,0};
    CHECK(std::equal(std::begin(prefix), std::end(prefix), bytes.begin()));
    CHECK(bytes[136+256*4] == 4 && bytes[137+256*4] == 3 &&
          bytes[138+256*4] == 2 && bytes[139+256*4] == 1);
    {
        auto image = decode(bytes, ram);
        CHECK(image.origin().ref == f.ref && image.origin().computed_frontier == 512 && image.origin().events == 3);
        CHECK(image.origin().configuration == configuration() && image.image_bytes() == bytes.size());
        CHECK(image.staging_bytes() == f.rows.size()*sizeof(int32_t) + 3*sizeof(SourceEvent));
        CHECK(ram.charged_bytes() == image.staging_bytes()); check_rows(image, f.rows);
        int32_t cut[4]{}; image.read_tokens(255, cut, 4);
        CHECK(std::equal(std::begin(cut), std::end(cut), f.rows.begin()+255));
        CHECK(!image.event(0).complete && image.event(0).id == image.event(1).id);
        CHECK(image.event(1).kind == SourceEventKind::ToolResult && image.event(2).end == 513);
        rejects(MemoryErrorCode::InvalidPlan, [&] { image.read_tokens(513, cut, 1); });
        rejects(MemoryErrorCode::InvalidPlan, [&] { image.read_tokens(0, nullptr, 1); });
        rejects(MemoryErrorCode::InvalidPlan, [&] { image.event(3); });
        auto moved = std::move(image);
        CHECK(image.image_bytes() == 0 && image.staging_bytes() == 0 && !image.origin().ref.source.id);
        rejects(MemoryErrorCode::InvalidPlan, [&] { image.read_tokens(0, cut, 1); });
        check_rows(moved, f.rows);
    }
    CHECK(ram.charged_bytes() == 0 && f.store.stats().charged == before.charged);
    std::unique_ptr<FILE, int(*)(FILE*)> file(std::tmpfile(), &std::fclose);
    CHECK(file);
    const auto count = write_source_image(f.store, f.ref, limits(), [&](const void * p, size_t n) {
        if (std::fwrite(p, 1, n, file.get()) != n) throw std::runtime_error("file write failed");
    });
    CHECK(std::fflush(file.get()) == 0); std::rewind(file.get());
    auto from_file = read_source_image([&](void * p, size_t n) {
        if (std::fread(p, 1, n, file.get()) != n) throw std::runtime_error("file read failed");
    }, count, configuration(), limits(), ram);
    check_rows(from_file, f.rows);
}
void branch_empty_and_shared_quota() {
    Fixture f; const auto branch = f.store.branch(f.ref, 258); const auto bytes = encode(f.store, branch);
    const auto required = 258*sizeof(int32_t) + 3*sizeof(SourceEvent);
    HostKvStorage ram(required), short_ram(required-1);
    rejects(MemoryErrorCode::BudgetExceeded, [&] { (void)decode(bytes, short_ram); });
    CHECK(short_ram.charged_bytes() == 0);
    {
        auto first = decode(bytes, ram);
        CHECK(first.origin().parent == f.ref && first.origin().parent_prefix == 258);
        CHECK(first.origin().computed_frontier == 0 && !first.event(2).complete && first.event(2).end == 258);
        rejects(MemoryErrorCode::BudgetExceeded, [&] { (void)decode(bytes, ram); });
        auto moved = std::move(first); CHECK(ram.charged_bytes() == required);
        CHECK(moved.staging_bytes() == required);
    }
    CHECK(ram.charged_bytes() == 0);
    const auto empty = f.store.capture(configuration(), nullptr, 0, nullptr, 0, 0);
    const auto empty_bytes = encode(f.store, empty);
    CHECK(empty_bytes.size() == 144);
    auto image = decode(empty_bytes, ram);
    CHECK(image.origin().accepted_frontier == 0 && image.staging_bytes() == 0 && !ram.charged_bytes());
    image.read_tokens(0, nullptr, 0);
    auto prepared = prepare_source_image(f.store, empty, limits(), ram);
    CHECK(!prepared.staging_bytes() && encode(prepared) == empty_bytes && !ram.charged_bytes());
}
void immutable_prepare_and_background_transport() {
    Fixture f; const auto bytes = encode(f.store, f.ref);
    const auto required = f.rows.size()*sizeof(int32_t) + 3*sizeof(SourceEvent);
    HostKvStorage short_ram(required-1), ram(required*2);
    rejects(MemoryErrorCode::BudgetExceeded, [&] {
        (void)prepare_source_image(f.store, f.ref, limits(), short_ram);
    });
    CHECK(!short_ram.charged_bytes());
    {
        auto prepared = prepare_source_image(f.store, f.ref, limits(), ram);
        auto decoded = decode(bytes, ram);
        CHECK(ram.charged_bytes() == 2*required && prepared.staging_bytes() == required);
        CHECK(encode(prepared) == bytes && encode(decoded) == bytes);
        rejects(MemoryErrorCode::BudgetExceeded, [&] {
            (void)prepare_source_image(f.store, f.ref, limits(), ram);
        });
        rejects(MemoryErrorCode::BudgetExceeded, [&] { (void)decode(bytes, ram); });
        const int32_t next = 7;
        const auto old_ref = f.ref;
        f.ref = f.store.append(f.ref, &next, 1, nullptr, 0, 512);
        rejects(MemoryErrorCode::Stale, [&] {
            (void)prepare_source_image(f.store, old_ref, limits(), ram);
        });
        CHECK(prepared.origin().ref == old_ref && encode(prepared) == bytes);
        CHECK(f.store.forget(f.ref) == 1);
        // An in-flight DATA owner can outlive local forgetting. A future durable
        // publisher must reject it through its own authoritative forget fence.
        CHECK(ram.charged_bytes() == 2*required);
        auto completed = std::async(std::launch::async, [image = std::move(prepared)]() mutable {
            auto owned = std::move(image); // release this actual R owner before returning
            return encode(owned); // never accesses the mutable source directory
        });
        CHECK(completed.get() == bytes && ram.charged_bytes() == required);
        rejects(MemoryErrorCode::InvalidPlan, [&] { (void)encode(prepared); });
        check_rows(decoded, f.rows);
    }
    CHECK(!ram.charged_bytes());
    // The allocation account also survives destruction of its caller's wrapper.
    auto detached = [&] {
        Fixture temporary; HostKvStorage temporary_ram(required);
        return prepare_source_image(temporary.store, temporary.ref, limits(), temporary_ram);
    }();
    check_rows(detached, f.rows);
}
void corruption_and_semantics() {
    Fixture f; const auto bytes = encode(f.store, f.ref); HostKvStorage ram(1 << 16);
    size_t rejected_bytes = 0;
    for (size_t i = 0; i < bytes.size(); ++i) {
        auto bad = bytes; bad[i] ^= 1;
        bool rejected = false;
        try { (void)decode(bad, ram); }
        catch (const SnapshotCorrupt &) { rejected = true; }
        catch (const MemoryError &) { rejected = true; }
        CHECK(rejected && ram.charged_bytes() == 0); ++rejected_bytes;
    }
    for (size_t n = 0; n < bytes.size(); ++n) {
        auto short_image = std::vector<uint8_t>(bytes.begin(), bytes.begin()+n);
        corrupt([&] { (void)decode(short_image, ram); }); CHECK(!ram.charged_bytes());
    }
    auto trailing = bytes; trailing.push_back(0);
    corrupt([&] { (void)decode(trailing, ram); });
    const auto event_start = 136 + 4*f.rows.size();
    for (int kind = 0; kind < 11; ++kind) {
        auto bad = bytes;
        if (kind == 0) little(bad, 136, UINT32_MAX, 4);
        if (kind == 1) little(bad, 24, 0); // origin id
        if (kind == 2) little(bad, 88, 514); // computed > accepted
        if (kind == 3) little(bad, 48, 1); // incomplete parent reference
        if (kind == 4) little(bad, event_start, 0);
        if (kind == 5) little(bad, event_start+16, 514);
        if (kind == 6) little(bad, event_start+24, 255, 1);
        if (kind == 7) little(bad, event_start+25, 2, 1);
        if (kind == 8) little(bad, event_start+26+8, 0); // overlapping event
        if (kind == 9) std::fill(bad.begin()+104, bad.begin()+136, uint8_t(0));
        if (kind == 10) little(bad, 80, UINT64_MAX); // declared token geometry
        rehash(bad); corrupt([&] { (void)decode(bad, ram); }); CHECK(!ram.charged_bytes());
    }
    std::printf("source image corruption bytes checked: %zu, truncations: %zu, semantic forgeries: 11\n", rejected_bytes, bytes.size());
}
void limits_transport_and_revision() {
    Fixture f; const auto bytes = encode(f.store, f.ref); HostKvStorage ram(1 << 16);
    auto wrong = configuration(); wrong[0] ^= 1;
    rejects(MemoryErrorCode::Unsupported, [&] { (void)decode(bytes, ram, wrong); });
    rejects(MemoryErrorCode::InvalidPlan, [&] { (void)decode(bytes, ram, {}); });
    HostKvStorage tiered(1 << 16, std::make_shared<SpillFile>(std::filesystem::temp_directory_path(), 1 << 16));
    rejects(MemoryErrorCode::Unsupported, [&] { (void)decode(bytes, tiered); });
    rejects(MemoryErrorCode::Unsupported, [&] { (void)prepare_source_image(f.store, f.ref, limits(), tiered); });
    CHECK(tiered.charged_bytes() == 0 && tiered.disk_bytes() == 0);
    auto small = limits(); small.max_tokens = 1;
    rejects(MemoryErrorCode::BudgetExceeded, [&] { (void)decode(bytes, ram, configuration(), small); });
    small = limits(); small.max_events = 2;
    rejects(MemoryErrorCode::BudgetExceeded, [&] { (void)decode(bytes, ram, configuration(), small); });
    small = limits(); small.max_bytes = bytes.size()-1;
    rejects(MemoryErrorCode::BudgetExceeded, [&] { (void)decode(bytes, ram, configuration(), small); });
    size_t writes = 0;
    rejects(MemoryErrorCode::BudgetExceeded, [&] {
        write_source_image(f.store, f.ref, small, [&](const void *, size_t) { ++writes; });
    });
    CHECK(writes == 0 && ram.charged_bytes() == 0);
    const auto charge = f.store.stats().charged;
    bool failed = false;
    try { write_source_image(f.store, f.ref, limits(), [&](const void *, size_t) { throw std::runtime_error("write fault"); }); }
    catch (const std::runtime_error&) { failed = true; }
    CHECK(failed && f.store.stats().charged == charge && f.store.state(f.ref).ref == f.ref);
    failed = false; size_t read = 0;
    try {
        (void)read_source_image([&](void * p, size_t n) {
            if (read >= 136) throw std::runtime_error("read fault");
            std::memcpy(p, bytes.data()+read, n); read += n;
        }, bytes.size(), configuration(), limits(), ram);
    } catch (const std::runtime_error&) { failed = true; }
    CHECK(failed && ram.charged_bytes() == 0);
    const auto old = f.ref;
    const int32_t next = 4;
    f.ref = f.store.append(f.ref, &next, 1, nullptr, 0, 512);
    rejects(MemoryErrorCode::Stale, [&] { (void)encode(f.store, old); });
}
void every_transport_failure() {
    Fixture f; const auto bytes = encode(f.store, f.ref); HostKvStorage ram(1 << 16);
    const auto source_charge = f.store.stats().charged;
    auto image = prepare_source_image(f.store, f.ref, limits(), ram);
    const auto image_charge = ram.charged_bytes();
    size_t calls = 0;
    write_source_image(image, limits(), [&](const void *, size_t) { ++calls; });
    for (size_t failure = 0; failure < calls; ++failure) {
        for (int mode = 0; mode < 2; ++mode) {
            size_t writes = 0; bool failed = false;
            SnapshotWriter::Sink sink = [&](const void *, size_t) {
                if (writes++ == failure) throw std::runtime_error("write fault");
            };
            try {
                if (mode == 0) write_source_image(image, limits(), sink);
                else write_source_image(f.store, f.ref, limits(), sink);
            } catch (const std::runtime_error &) { failed = true; }
            CHECK(failed && ram.charged_bytes() == image_charge && f.store.stats().charged == source_charge);
        }
        size_t reads = 0, offset = 0; bool failed = false;
        try {
            (void)read_source_image([&](void * p, size_t n) {
                if (reads++ == failure) throw std::runtime_error("read fault");
                CHECK(n <= 1024 && n <= bytes.size()-offset);
                std::memcpy(p, bytes.data()+offset, n); offset += n;
            }, bytes.size(), configuration(), limits(), ram);
        } catch (const std::runtime_error &) { failed = true; }
        CHECK(failed && ram.charged_bytes() == image_charge);
    }
    CHECK(encode(image) == bytes);
    std::printf("source image transport failures checked: %zu per writer/reader, two writers\n", calls);
}
void allocation_failures_and_move_assignment() {
    Fixture f; const auto bytes = encode(f.store, f.ref);
    const auto source_charge = f.store.stats().charged;
    for (int mode = 0; mode < 2; ++mode) {
        size_t faults = 0; bool done = false;
        for (size_t failure = 0; failure < 32; ++failure) {
            HostKvStorage ram(1 << 16);
            fail_at = failure; allocations = 0;
            bool failed = false;
            try {
                auto image = mode == 0 ? decode(bytes, ram) : prepare_source_image(f.store, f.ref, limits(), ram);
            } catch (const std::bad_alloc&) { failed = true; }
            fail_at = SIZE_MAX;
            CHECK(ram.charged_bytes() == 0 && f.store.stats().charged == source_charge);
            if (failed) ++faults;
            else { done = true; break; }
        }
        CHECK(done && faults >= 2);
        std::printf("source image allocation failures checked: %zu (%s)\n", faults, mode ? "prepare" : "decode");
    }
    HostKvStorage ram(1 << 16);
    {
        auto first = decode(bytes, ram), second = decode(bytes, ram);
        const auto one = first.staging_bytes(); CHECK(ram.charged_bytes() == 2*one);
        second = std::move(first);
        CHECK(!first.image_bytes() && ram.charged_bytes() == one); check_rows(second, f.rows);
    }
    CHECK(!ram.charged_bytes());
}
void import_new_root_and_atomic_failure() {
    Fixture f; HostKvStorage ram(1<<16);
    auto image = prepare_source_image(f.store, f.ref, limits(), ram);
    SourceJournalStore target{{1<<20,8192,256,4}};
    const auto seed = target.capture(configuration(), nullptr, 0, nullptr, 0, 0);
    const auto imported = target.import_image(image); const auto state = target.state(imported);
    CHECK(!(imported.source == f.ref.source) && imported.revision == 1 && imported.source.id == seed.source.id+1);
    CHECK(state.configuration == configuration() && state.accepted_frontier == 513 && !state.computed_frontier);
    CHECK(state.parent == SourceRef{} && state.parent_prefix == 0 && state.events == 3);
    std::vector<int32_t> actual(f.rows.size()); target.read_tokens(imported,0,actual.data(),actual.size()); CHECK(actual == f.rows);
    for (uint64_t i = 0; i < state.events; ++i) {
        const auto expected = image.event(i), event = target.event(imported,i);
        CHECK(event.id == expected.id && event.begin == expected.begin && event.end == expected.end &&
            event.kind == expected.kind && event.complete == expected.complete);
    }
    const auto child = f.store.branch(f.ref,258);
    auto branch_image = prepare_source_image(f.store,child,limits(),ram);
    const auto branch_import = target.import_image(branch_image);
    CHECK(target.state(branch_import).parent == SourceRef{} && target.state(branch_import).accepted_frontier == 258);
    CHECK(!target.event(branch_import,2).complete && target.event(branch_import,2).end == 258);
    CHECK(target.state(seed).accepted_frontier == 0);
    const int32_t next = 17;
    const auto appended = target.append(imported,&next,1,nullptr,0,513);
    CHECK(target.state(appended).accepted_frontier == 514);

    SourceJournalStore probe{{1<<20,8192,256,2}};
    const auto metadata = probe.stats().metadata_bytes;
    const auto payload = 513*sizeof(int32_t)+8*sizeof(SourceEvent);
    SourceJournalStore exact{{metadata+payload,8192,256,2}}, short_s{{metadata+payload-1,8192,256,2}};
    const auto old = short_s.capture(configuration(),nullptr,0,nullptr,0,0);
    const auto before = short_s.stats();
    rejects(MemoryErrorCode::BudgetExceeded,[&] { short_s.import_image(image); });
    CHECK(short_s.stats().charged == before.charged && short_s.stats().sources == 1 && short_s.stats().denied > before.denied);
    CHECK(short_s.state(old).ref == old);
    exact.import_image(image); CHECK(exact.stats().charged == exact.stats().capacity);
    SourceJournalStore token_limit{{1<<20,512,256,2}}, event_limit{{1<<20,8192,2,2}}, full{{1<<20,8192,256,1}};
    rejects(MemoryErrorCode::BudgetExceeded,[&] { token_limit.import_image(image); });
    rejects(MemoryErrorCode::BudgetExceeded,[&] { event_limit.import_image(image); });
    const auto occupied = full.capture(configuration(),nullptr,0,nullptr,0,0);
    rejects(MemoryErrorCode::BudgetExceeded,[&] { full.import_image(image); });
    CHECK(full.state(occupied).ref == occupied && full.stats().sources == 1);

    size_t failures = 0; bool complete = false; const auto held_r = ram.charged_bytes();
    for (size_t point = 0; point < 32; ++point) {
        SourceJournalStore destination{{1<<20,8192,256,2}};
        const auto existing = destination.capture(configuration(),nullptr,0,nullptr,0,0);
        const auto charge = destination.stats().charged; bool failed = false;
        fail_at = point; allocations = 0;
        try { destination.import_image(image); } catch (const std::bad_alloc &) { failed = true; }
        fail_at = SIZE_MAX;
        CHECK(ram.charged_bytes() == held_r && destination.state(existing).ref == existing);
        if (!failed) { complete = true; break; }
        ++failures; CHECK(destination.stats().charged == charge && destination.stats().sources == 1);
        const auto retry = destination.import_image(image); CHECK(retry.source.id == existing.source.id+1);
    }
    CHECK(complete && failures > 0);
    const auto empty_source = f.store.capture(configuration(),nullptr,0,nullptr,0,0);
    auto empty_image = prepare_source_image(f.store,empty_source,limits(),ram);
    const auto empty_import = probe.import_image(empty_image); CHECK(!probe.state(empty_import).accepted_frontier);
    auto moved = std::move(empty_image); const auto sources = probe.stats().sources;
    rejects(MemoryErrorCode::InvalidPlan,[&] { probe.import_image(empty_image); }); CHECK(probe.stats().sources == sources);
    std::printf("atomic source image import allocation failure points: %zu\n",failures);
}
} // namespace
int main() {
    roundtrip_and_file();
    branch_empty_and_shared_quota();
    immutable_prepare_and_background_transport();
    corruption_and_semantics();
    limits_transport_and_revision();
    every_transport_failure();
    allocation_failures_and_move_assignment();
    import_new_root_and_atomic_failure();
    std::puts("source image codec and atomic data import contracts passed; no native restore claim");
}
