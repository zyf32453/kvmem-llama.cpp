#include "kvmem/historical_kv_gc.hpp"
#include "kvmem/historical_kv_ram_reservation.hpp"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <new>
#include <stdexcept>
#include <thread>

// Fault injection is local to the calling thread. It checks every ordinary
// allocation boundary at admission, and forbids allocation during consumption.
static thread_local int64_t allocation_budget = -1;
static std::atomic<void *> delayed_free{nullptr};
static std::atomic<bool> free_entered{false}, allow_free{false}, free_timed_out{false};
void * operator new(size_t size) {
    if (allocation_budget == 0) throw std::bad_alloc();
    if (allocation_budget > 0) --allocation_budget;
    if (void * p = std::malloc(size ? size : 1)) return p;
    throw std::bad_alloc();
}
void * operator new[](size_t size) { return ::operator new(size); }
static void test_free(void * p) noexcept {
    if (p && delayed_free.load(std::memory_order_acquire) == p) {
        free_entered.store(true, std::memory_order_release);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!allow_free.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline)
            std::this_thread::yield();
        if (!allow_free.load(std::memory_order_acquire)) free_timed_out.store(true);
    }
    std::free(p);
}
void operator delete(void * p) noexcept { test_free(p); }
void operator delete[](void * p) noexcept { test_free(p); }
void operator delete(void * p, size_t) noexcept { test_free(p); }
void operator delete[](void * p, size_t) noexcept { test_free(p); }
struct AllocationBudget {
    explicit AllocationBudget(int64_t count) { allocation_budget = count; }
    ~AllocationBudget() { allocation_budget = -1; }
};

using namespace kvmem;
using Repo = HistoricalKvRepository;
using Storage = HistoricalKvRawStore;
using Ram = HistoricalKvRamReservation;

static void check(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}
template <typename F> static bool invalid(F fn) {
    try { fn(); } catch (const std::invalid_argument &) { return true; }
    return false;
}
static HistoricalKvRawStoreConfig config() { return {2, 4, 3, 2, 2, {}}; }
static HistoricalKvRamDemand demand(uint64_t tokens, uint64_t boundaries) {
    HistoricalKvRamDemand d;
    d.max_accepted_output_tokens = tokens;
    d.max_append_boundaries = boundaries;
    return d;
}
static Repo::ReleaseCallback release_to(std::shared_ptr<Repo> repo, std::shared_ptr<Storage> storage,
                                      std::atomic<unsigned> & releases) {
    return [weak = std::weak_ptr<Repo>(repo), storage, &releases](HistoricalKvHandle handle) {
        const auto owner = weak.lock();
        if (!owner) return false;
        const auto result = storage->release(*owner, handle);
        if (result.code == Storage::ReleaseCode::Released) ++releases;
        return result.code == Storage::ReleaseCode::Released ||
               result.code == Storage::ReleaseCode::AlreadyReleased ||
               result.code == Storage::ReleaseCode::Absent;
    };
}

static void slot_entitlements() {
    Repo repo(3), foreign(3);
    auto all = repo.try_reserve_slots(3);
    check(all && all->remaining() == 3 && repo.ready_count() == 0, "slots not deducted");
    const auto handle = *all->next_handle();
    check(repo.snapshot().empty() && repo.observe_range(0, 3).pending == 0 &&
          repo.observe_range(0, 3).reserved == 3 &&
          !repo.inspect(handle) && !repo.acquire_new(handle) && !repo.acquire_read(handle) &&
          !repo.acquire_transfer(handle) && repo.retire(handle) == Repo::RetireResult::Stale,
          "Reserved slot leaked into history or pending GC credits");
    check(!repo.try_reserve_slots(1) && !repo.begin_write(1, 0, 1, 1, 0, [](HistoricalKvHandle) { return true; }),
          "another request stole private slots");
    check(!foreign.begin_reserved_write(*all, 1, 0, 1, 1, 0, [](HistoricalKvHandle) { return true; }) &&
          all->remaining() == 3, "foreign repository consumed reservation");
    check(!repo.begin_reserved_write(*all, 1, UINT64_MAX, 1, 1, 0, [](HistoricalKvHandle) { return true; }) &&
          all->remaining() == 3, "invalid range consumed reservation");
    auto moved = std::move(*all);
    check(all->remaining() == 0 && moved.remaining() == 3, "moved reservation still owns credits");
    auto writer = repo.begin_reserved_write(moved, 1, 7, 1, 1, 0, [](HistoricalKvHandle) { return true; });
    check(writer && writer->handle().generation == handle.generation &&
          writer->handle().slot == handle.slot, "consumption changed reserved identity");
    moved.reset();
    check(repo.ready_count() == 2 && repo.inspect(handle)->life == HistoricalKvLife::Writing,
          "unused return discarded consumed writer");
    check(!repo.try_reserve_slots(3) && repo.ready_count() == 2,
          "failed all-or-none reserve retained some slots");
    repo.close();
    check(!repo.try_reserve_slots(1), "closed repository admitted reservation");
    check(repo.retire(handle) == Repo::RetireResult::Committed, "retire consumed slot");
    writer.reset();
    check(repo.try_reclaim(handle) == Repo::ReclaimResult::Complete && repo.ready_count() == 3,
          "consumed slot failed ordinary lifecycle");
    // A reservation can safely outlive the public repository object.
    std::optional<Repo::Reservation> orphan;
    { Repo temporary(1); orphan = temporary.try_reserve_slots(1); }
    orphan.reset();
}

static uint64_t max_partition_slots(uint32_t tokens, uint32_t calls, uint32_t block) {
    if (!tokens) return 0;
    if (!calls) return 0;
    uint64_t best = (tokens + block - 1) / block;
    if (calls > 1) for (uint32_t first = 1; first < tokens; ++first)
        best = std::max(best, (first + block - 1) / block + max_partition_slots(tokens - first, calls - 1, block));
    return best;
}

static void demand_bounds() {
    for (uint32_t block = 1; block <= 5; ++block)
        for (uint32_t tokens = 1; tokens <= 10; ++tokens)
            for (uint32_t calls = 1; calls <= 5; ++calls)
                check(Ram::required_slots(demand(tokens, calls), block) == max_partition_slots(tokens, calls, block),
                      "tail-fragmentation bound differs from exhaustive token partitions");
    auto d = demand(8, 8);
    d.extra_inflight_slots = 2;
    check(Ram::required_slots(d, 4) == 10, "inflight allocations omitted");
    d.streaming = true;
    check(invalid([&] { Ram::required_slots(d, 4); }), "streaming silently accepted");
    d.streaming = false;
    d.max_accepted_output_tokens.reset();
    check(invalid([&] { Ram::required_slots(d, 4); }), "unbounded output accepted");
    d = demand(UINT64_MAX, 1);
    d.uncached_input_tokens = 1;
    check(invalid([&] { Ram::required_slots(d, 4); }), "token demand overflow accepted");
    d = demand(1, 0);
    check(invalid([&] { Ram::required_slots(d, 4); }), "unknown append bound accepted");
}

static void admission_failure_rollback() {
    auto repo = std::make_shared<Repo>(3);
    auto cfg = config();
    cfg.payload_budget = std::make_shared<HistoricalKvPayloadBudget>(Storage::max_payload_bytes(cfg, 3));
    auto storage = std::make_shared<Storage>(cfg, 3);
    std::atomic<unsigned> releases{0};
    auto history = repo->begin_write(9, 90, 1, 1, 0, release_to(repo, storage, releases));
    check(history && storage->reserve(*history, history->handle(), 90, 1), "history setup failed");
    const auto existing = history->handle();
    const auto before = storage->allocated_payload_bytes();
    auto d = demand(2, 2);
    d.private_scratch_bytes = 4096;
    bool succeeded = false;
    unsigned failed_allocations = 0;
    for (int64_t budget = 0; budget < 128; ++budget) {
        std::optional<Ram> request;
        {
            AllocationBudget fail_after(budget);
            request = Ram::try_reserve(repo, storage, d);
        }
        if (request) {
            check(request->slots().remaining() == 2 && request->payload_count() == 2 &&
                  request->scratch_size() == 4096 && request->reserved_scratch_bytes() >= 4096 &&
                  storage->allocated_payload_bytes() == before + request->reserved_payload_bytes(),
                  "request only reserved logical credits");
            succeeded = true;
            request.reset();
        } else ++failed_allocations;
        check(repo->ready_count() == 2 && storage->allocated_payload_bytes() == before &&
              cfg.payload_budget->used_bytes() == before &&
              repo->inspect(existing)->life == HistoricalKvLife::Writing && releases == 0,
              "failed admission leaked credits or reclaimed existing history");
        if (succeeded) break;
    }
    check(succeeded && failed_allocations > 10, "allocation failure sweep did not cover payload/scratch stages");
    check(!Ram::try_reserve(repo, storage, demand(3, 3)), "capacity check assumed future reclaim");
    repo->retire(existing);
    history.reset();
    check(repo->try_reclaim(existing) == Repo::ReclaimResult::Complete, "history cleanup");
    check(cfg.payload_budget->used_bytes() == 0, "allocation failure sweep leaked shared budget");
}

static void shared_payload_budget_across_incarnations() {
    auto cfg = config();
    const auto unit = Storage::max_payload_bytes(cfg, 1);
    check(unit == 56, "shared budget fixture size");
    cfg.payload_budget = std::make_shared<HistoricalKvPayloadBudget>(2 * unit);
    auto old_repo = std::make_shared<Repo>(2), next_repo = std::make_shared<Repo>(2);
    auto old_store = std::make_shared<Storage>(cfg, 2), next_store = std::make_shared<Storage>(cfg, 2);
    std::atomic<unsigned> releases{0};
    auto old = Ram::try_reserve(old_repo, old_store, demand(1, 1));
    check(old && cfg.payload_budget->used_bytes() == unit, "old domain reservation");
    auto writer = old_repo->begin_reserved_write(old->slots(), 1, 0, 1, 1, unit,
                                                 release_to(old_repo, old_store, releases));
    check(bool(writer), "old writer creation");
    const auto handle = writer->handle();
    check(old_store->install_payload(*writer, handle, 0, 1, old->payload(0)), "old install");
    for (unsigned layer = 0; layer < 2; ++layer)
        check(old_store->write_layer(*writer, handle, layer, {1, 2, 3}, {4, 5}, {6, 7}, 1), "old write");
    check(old_store->publish_complete(*old_repo, *writer, handle), "old publish");
    auto reader = old_repo->acquire_read(handle);
    check(bool(reader), "old reader");
    old_repo->retire(handle);
    old_repo->close();
    writer.reset(); old.reset();
    check(old_repo->try_reclaim(handle) == Repo::ReclaimResult::Deferred, "old reader lost payload");
    auto next = Ram::try_reserve(next_repo, next_store, demand(1, 1));
    check(next && cfg.payload_budget->used_bytes() == 2 * unit, "replacement did not share old quota");
    check(!Ram::try_reserve(next_repo, next_store, demand(1, 1)) &&
          next_repo->ready_count() == 1 && cfg.payload_budget->used_bytes() == 2 * unit,
          "third payload exceeded common quota or lost rollback credit");
    next.reset();
    // First payload fits, second fails. Partial preparation must return every
    // byte and slot, without clearing the retained old conversation.
    check(!Ram::try_reserve(next_repo, next_store, demand(2, 2)) &&
          next_repo->ready_count() == 2 && next_store->allocated_payload_bytes() == 0 &&
          cfg.payload_budget->used_bytes() == unit && old_store->read_layer(*reader, handle, 0),
          "partial cross-domain admission damaged old history or leaked credit");
    reader.reset();
    check(!Ram::try_reserve(next_repo, next_store, demand(2, 2)), "future GC release counted as free bytes");
    check(old_repo->try_reclaim(handle) == Repo::ReclaimResult::Complete && releases == 1 &&
          cfg.payload_budget->used_bytes() == 0, "old physical release did not return shared quota");
    for (unsigned generation = 0; generation < 64; ++generation) {
        auto request = Ram::try_reserve(next_repo, next_store, demand(2, 2));
        check(request && cfg.payload_budget->used_bytes() == 2 * unit, "shared quota was not reusable");
    }
    check(cfg.payload_budget->used_bytes() == 0 && cfg.payload_budget->refusals() >= 3,
          "cross-domain retries lost accounting");
    Storage::PreparedPayload orphan;
    { Storage temporary(cfg, 1); orphan = temporary.prepare_payload(4); }
    check(cfg.payload_budget->used_bytes() == unit, "store teardown returned a still-owned payload's credit");
    orphan = {};
    check(cfg.payload_budget->used_bytes() == 0, "orphan payload lost its shared budget owner");
}

static void payload_budget_returns_after_physical_free() {
    auto budget = std::make_shared<HistoricalKvPayloadBudget>(64);
    HistoricalKvPayloadAllocator<uint8_t> allocator(budget);
    auto * data = allocator.allocate(64);
    delayed_free.store(data, std::memory_order_release);
    std::thread worker([&] { allocator.deallocate(data, 64); });
    struct Finish {
        std::thread & worker;
        ~Finish() {
            allow_free.store(true, std::memory_order_release);
            if (worker.joinable()) worker.join();
            delayed_free.store(nullptr, std::memory_order_release);
        }
    } finish{worker};
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!free_entered.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
    check(free_entered && budget->used_bytes() == 64, "slow physical deleter returned quota early");
    bool refused = false;
    try { auto * unexpected = allocator.allocate(1); allocator.deallocate(unexpected, 1); }
    catch (const std::bad_alloc &) { refused = true; }
    check(refused, "new domain reused memory while deallocation was still blocked");
    allow_free.store(true, std::memory_order_release);
    worker.join();
    check(!free_timed_out && budget->used_bytes() == 0, "physical free did not return exact credit");
}

static void concurrent_payload_budget() {
    auto budget = std::make_shared<HistoricalKvPayloadBudget>(128);
    std::atomic<bool> failed{false};
    std::atomic<unsigned> completed{0};
    auto actor = [&] {
        HistoricalKvPayloadAllocator<uint8_t> allocator(budget);
        for (unsigned i = 0; i < 80; ++i) {
            uint8_t * data = nullptr;
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            do {
                try { data = allocator.allocate(64); } catch (const std::bad_alloc &) {}
                if (!data) std::this_thread::yield();
            } while (!data && std::chrono::steady_clock::now() < deadline);
            if (!data) { failed = true; return; }
            if (budget->used_bytes() > budget->limit_bytes()) failed = true;
            std::fill_n(data, 64, 0x5a);
            allocator.deallocate(data, 64);
            ++completed;
        }
    };
    std::thread a(actor), b(actor), c(actor), d(actor);
    a.join(); b.join(); c.join(); d.join();
    check(!failed && completed == 320 && budget->used_bytes() == 0, "concurrent payload quota leaked or overspent");
}

static void paused_reclaimer_and_allocation_free_consumption() {
    constexpr uint32_t count = 8;
    auto repo = std::make_shared<Repo>(count);
    auto storage = std::make_shared<Storage>(config(), count);
    auto request = Ram::try_reserve(repo, storage, demand(count, count));
    check(request && repo->ready_count() == 0 && storage->allocated_payload_bytes() > 0, "reservation failed");
    HistoricalKvGc::Config gc_config;
    gc_config.mode = HistoricalKvGc::Mode::Delete;
    gc_config.ready_low = 1;
    gc_config.ready_high = 2;
    gc_config.scan_blocks = gc_config.batch_blocks = gc_config.reclaim_blocks = count;
    HistoricalKvGc gc(repo, std::make_unique<HistoricalKvLruPolicy>(count), gc_config);
    gc.manual_tick();
    check(gc.stats().committed == 0 && request->slots().remaining() == count, "GC consumed request entitlement");
    // No GC ticks until every promised token is written. Pre-build all caller
    // inputs and callbacks, so denial of new() detects hidden payload growth.
    std::atomic<unsigned> releases{0};
    std::vector<Repo::ReleaseCallback> callbacks;
    std::vector<Repo::Lease> writers;
    callbacks.reserve(count);
    writers.reserve(count);
    for (uint32_t i = 0; i < count; ++i) callbacks.push_back(release_to(repo, storage, releases));
    const std::vector<uint8_t> k{1, 2, 3}, v{4, 5};
    const std::vector<float> mean{6, 7};
    const uint64_t bytes = storage->allocated_payload_bytes();
    bool ok = true;
    {
        AllocationBudget no_allocations(0);
        for (uint32_t i = 0; i < count; ++i) {
            auto writer = repo->begin_reserved_write(request->slots(), i, i, 1, i + 1,
                request->payload(i).payload_bytes(), std::move(callbacks[i]));
            if (!writer) { ok = false; break; }
            const auto handle = writer->handle();
            ok = ok && storage->install_payload(*writer, handle, i, 1, request->payload(i));
            ok = ok && repo->retire(handle) == Repo::RetireResult::Committed;
            for (uint32_t il = 0; il < 2; ++il)
                ok = ok && storage->write_layer(*writer, handle, il, k, v, mean, 1);
            ok = ok && !storage->publish_complete(*repo, *writer, handle);
            writers.push_back(std::move(*writer));
        }
    }
    check(ok && writers.size() == count && request->slots().remaining() == 0 &&
          storage->allocated_payload_bytes() == bytes && releases == 0,
          "accepted RAM demand depended on foreground allocation or GC supply");
    request.reset();
    check(repo->ready_count() == 0 && storage->allocated_payload_bytes() == bytes,
          "request end returned in-flight resources early");
    for (const auto & writer : writers)
        check(repo->try_reclaim(writer.handle()) == Repo::ReclaimResult::Deferred, "writer lost physical ownership");
    writers.clear();
    std::thread worker([&] { for (unsigned i = 0; i < 3; ++i) gc.manual_tick(); });
    worker.join();
    check(releases == count && storage->allocated_payload_bytes() == 0 && repo->ready_count() == count,
          "background completion did not return real memory and slots");
}

static void prepared_payload_validation_and_moves() {
    auto repo = std::make_shared<Repo>(2);
    auto storage = std::make_shared<Storage>(config(), 2);
    Storage foreign(config(), 2);
    auto first = Ram::try_reserve(repo, storage, demand(1, 1));
    auto second = Ram::try_reserve(repo, storage, demand(1, 1));
    check(first && second, "move setup failed");
    const auto unit = first->reserved_payload_bytes();
    *first = std::move(*second);
    check(first->slots().remaining() == 1 && second->slots().remaining() == 0 &&
          repo->ready_count() == 1 && storage->allocated_payload_bytes() == unit, "bundle move leaked old credits");
    std::atomic<unsigned> releases{0};
    auto writer = repo->begin_reserved_write(first->slots(), 1, 713, 3, 1, unit, release_to(repo, storage, releases));
    check(writer.has_value(), "reserved writer activation");
    const auto handle = writer->handle();
    check(!foreign.install_payload(*writer, handle, 713, 3, first->payload(0)) && first->payload(0),
          "foreign store consumed prepared payload");
    check(!storage->install_payload(*writer, handle, 714, 3, first->payload(0)) && first->payload(0),
          "wrong original span consumed payload");
    check(storage->install_payload(*writer, handle, 713, 3, first->payload(0)) && !first->payload(0) &&
          storage->inspect(*repo, handle)->payload_bytes == unit, "short span discarded reserved capacity");
    first.reset(); second.reset();
    check(repo->ready_count() == 1 && storage->allocated_payload_bytes() == unit, "installed payload was returned unused");
    repo->retire(handle); writer.reset();
    check(repo->try_reclaim(handle) == Repo::ReclaimResult::Complete, "short installed payload cleanup");
    Storage::PreparedPayload orphan;
    { Storage temporary(config(), 1); orphan = temporary.prepare_payload(4); }
    orphan = {};
}

static void concurrent_reservations_and_observation() {
    auto repo = std::make_shared<Repo>(4);
    std::atomic<bool> running{true}, failed{false};
    std::atomic<unsigned> completed{0};
    std::atomic<unsigned> owners[4]{};
    std::thread observer([&] {
        while (running.load()) {
            const auto view = repo->observe_range(0, 4);
            for (const auto & meta : view.blocks)
                if (meta.life == HistoricalKvLife::Ready) failed.store(true);
        }
    });
    auto actor = [&](uint64_t stream) {
        try {
            for (unsigned i = 0; i < 200; ++i) {
                std::optional<Repo::Reservation> reserved;
                for (unsigned attempt = 0; attempt < 10000 && !reserved; ++attempt)
                    reserved = repo->try_reserve_slots(1);
                if (!reserved) { failed.store(true); break; }
                const auto handle = *reserved->next_handle();
                if (owners[handle.slot].fetch_add(1) != 0) failed.store(true);
                if (i % 2 == 0) {
                    owners[handle.slot].fetch_sub(1);
                    reserved.reset(); // Cancellation races metadata observations.
                } else {
                    auto writer = repo->begin_reserved_write(*reserved, stream, i, 1, i, 0,
                        [](HistoricalKvHandle) { return true; });
                    if (!writer) { failed.store(true); break; }
                    auto retired = Repo::RetireResult::Retry;
                    for (unsigned attempt = 0; attempt < 10000 && retired == Repo::RetireResult::Retry; ++attempt)
                        retired = repo->retire(handle);
                    if (retired != Repo::RetireResult::Committed) failed.store(true);
                    owners[handle.slot].fetch_sub(1);
                    writer.reset();
                    auto reclaimed = Repo::ReclaimResult::Deferred;
                    for (unsigned attempt = 0; attempt < 10000 && reclaimed != Repo::ReclaimResult::Complete; ++attempt) {
                        reclaimed = repo->try_reclaim(handle);
                        if (reclaimed != Repo::ReclaimResult::Complete) std::this_thread::yield();
                    }
                    if (reclaimed != Repo::ReclaimResult::Complete) failed.store(true);
                }
                ++completed;
            }
        } catch (...) { failed.store(true); }
    };
    std::thread a(actor, 1), b(actor, 2);
    a.join(); b.join();
    running.store(false);
    observer.join();
    check(!failed && completed == 400 && repo->ready_count() == 4 && repo->snapshot().empty(),
          "concurrent reserve/cancel/activate leaked or duplicated physical ownership");
}

int main() {
    try {
        slot_entitlements();
        demand_bounds();
        admission_failure_rollback();
        shared_payload_budget_across_incarnations();
        payload_budget_returns_after_physical_free();
        concurrent_payload_budget();
        paused_reclaimer_and_allocation_free_consumption();
        prepared_payload_validation_and_moves();
        concurrent_reservations_and_observation();
        std::puts("PASS historical RAM reservation: rollback, fragmentation, paused reclaim, no allocation on consume");
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL historical RAM reservation: %s\n", e.what());
        return 1;
    }
}
