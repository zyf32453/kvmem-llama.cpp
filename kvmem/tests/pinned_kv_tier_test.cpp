// PinnedKvTier host-logic unit test (CPU pinned KV tier, Task #43).
//
// Pure host logic — no GPU. Covers: slot sizing from byte budget, slot
// allocation/release (LIFO), block<->slot residency map, full-pool behavior
// (no eviction in v1 skeleton), and LRU victim ordering.

#include "kvmem/pinned_kv_tier.hpp"

#include <cstdio>
#include <atomic>
#include <cstdlib>
#include <new>
#include <thread>
#include <type_traits>
#include "kvmem/nvme_kv_allocation.hpp"

using namespace kvmem;

static thread_local bool deny_allocation = false;
void * operator new(size_t n) {
    if (deny_allocation) throw std::bad_alloc();
    if (void * p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void * operator new[](size_t n) { return ::operator new(n); }
void operator delete(void * p) noexcept { std::free(p); }
void operator delete[](void * p) noexcept { std::free(p); }
void operator delete(void * p, size_t) noexcept { std::free(p); }
void operator delete[](void * p, size_t) noexcept { std::free(p); }
static_assert(!std::is_same<PinnedAllocationHandle, NvmeAllocationHandle>::value,
              "a disk allocation must not authorize a pinned copy");

static int g_fail = 0;
#define CHECK(cond) do {                                                   \
    if (!(cond)) {                                                         \
        std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);        \
        ++g_fail;                                                          \
    }                                                                      \
} while (0)

static void test_sizing() {
    // 100 MB budget, 8 MB slots -> 12 whole slots (96 MB), remainder unused.
    PinnedKvTierConfig cfg;
    cfg.total_bytes = 100ull * 1024 * 1024;
    cfg.slot_bytes = 8ull * 1024 * 1024;
    PinnedKvTier t(cfg);
    CHECK(t.enabled());
    CHECK(t.slot_count() == 12);
    CHECK(t.free_slots() == 12);
    CHECK(t.used_slots() == 0);
    // slot byte offsets are slot_index * slot_bytes.
    CHECK(t.slot_offset(0) == 0);
    CHECK(t.slot_offset(3) == 3ull * cfg.slot_bytes);
}

static void test_disabled() {
    PinnedKvTierConfig cfg;  // total_bytes=0 -> disabled
    PinnedKvTier t(cfg);
    CHECK(!t.enabled());
    CHECK(t.slot_count() == 0);
    CHECK(t.place_block(0).slot == -1);  // nothing to hand out

    PinnedKvTierConfig small;
    small.total_bytes = 4ull * 1024 * 1024;   // smaller than one slot
    small.slot_bytes = 8ull * 1024 * 1024;
    PinnedKvTier t2(small);
    CHECK(!t2.enabled());
    CHECK(t2.slot_count() == 0);
}

static void test_place_release() {
    PinnedKvTierConfig cfg;
    cfg.total_bytes = 32ull * 1024 * 1024;
    cfg.slot_bytes = 8ull * 1024 * 1024;       // 4 slots
    PinnedKvTier t(cfg);
    CHECK(t.slot_count() == 4);

    // Place three blocks; LIFO hands out 0,1,2.
    auto p0 = t.place_block(10);
    auto p1 = t.place_block(11);
    auto p2 = t.place_block(12);
    CHECK(p0.slot == 0 && p0.evicted_block == -1);
    CHECK(p1.slot == 1);
    CHECK(p2.slot == 2);
    CHECK(t.used_slots() == 3 && t.free_slots() == 1);

    // Residency lookups.
    CHECK(t.block_slot(10) == 0);
    CHECK(t.block_slot(11) == 1);
    CHECK(t.block_slot(99) == -1);

    // Re-placing an already-resident block returns its slot, no new alloc.
    auto p0b = t.place_block(10);
    CHECK(p0b.slot == 0);
    CHECK(t.used_slots() == 3);

    // Release block 11 -> its slot returns to the pool, reused next.
    t.release_block(11);
    CHECK(t.block_slot(11) == -1);
    CHECK(t.free_slots() == 2);
    auto p3 = t.place_block(13);
    CHECK(p3.slot == 1);  // freed slot 1 reused (LIFO)

    // Releasing a non-resident block is a no-op.
    t.release_block(12345);
    CHECK(t.free_slots() == 1);
}

static void test_full_no_evict() {
    PinnedKvTierConfig cfg;
    cfg.total_bytes = 16ull * 1024 * 1024;
    cfg.slot_bytes = 8ull * 1024 * 1024;       // 2 slots
    PinnedKvTier t(cfg);
    CHECK(t.place_block(1).slot == 0);
    CHECK(t.place_block(2).slot == 1);
    // Full: v1 skeleton returns slot=-1, no eviction.
    auto full = t.place_block(3);
    CHECK(full.slot == -1);
    CHECK(full.evicted_block == -1);
    CHECK(t.block_slot(3) == -1);
    CHECK(t.used_slots() == 2);
}

static void test_full_with_explicit_evict() {
    PinnedKvTierConfig cfg;
    cfg.total_bytes = 16ull * 1024 * 1024;
    cfg.slot_bytes = 8ull * 1024 * 1024;       // 2 slots
    PinnedKvTier t(cfg);
    CHECK(t.place_block(1).slot == 0);
    CHECK(t.place_block(2).slot == 1);
    t.touch(1);  // block 2 is now LRU.

    auto p = t.place_block_evicting(3);
    CHECK(p.slot == 1);
    CHECK(p.evicted_block == 2);
    CHECK(t.block_slot(2) == -1);
    CHECK(t.block_slot(3) == 1);
    CHECK(t.used_slots() == 2);
    CHECK(t.lru_victim() == 1);
}

static void test_lru_order() {
    PinnedKvTierConfig cfg;
    cfg.total_bytes = 32ull * 1024 * 1024;
    cfg.slot_bytes = 8ull * 1024 * 1024;       // 4 slots
    PinnedKvTier t(cfg);
    t.place_block(1);
    t.place_block(2);
    t.place_block(3);
    // LRU victim is the oldest placed: block 1.
    CHECK(t.lru_victim() == 1);
    // Touch block 1 -> it becomes most recent, victim is now block 2.
    t.touch(1);
    CHECK(t.lru_victim() == 2);
    // Release block 2 -> victim is block 3.
    t.release_block(2);
    CHECK(t.lru_victim() == 3);

    t.clear();
    CHECK(t.lru_victim() == -1);
    CHECK(t.free_slots() == 4);
    CHECK(t.block_slot(1) == -1);
}

static void test_heat_aware_admission() {
    PinnedKvTierConfig cfg;
    cfg.total_bytes = 16ull * 1024 * 1024;
    cfg.slot_bytes = 8ull * 1024 * 1024;       // 2 slots
    cfg.cache_policy = PinnedKvCachePolicy::HeatAware;
    PinnedKvTier t(cfg);
    CHECK(t.place_block(1).slot == 0);
    CHECK(t.place_block(2).slot == 1);

    // Both residents have semantic reuse evidence. A one-pass streaming block
    // bypasses CPU instead of displacing either hot block.
    t.begin_selection_epoch();
    t.record_selected(1);
    t.record_selected(2);
    auto rejected = t.place_block_evicting(3);
    CHECK(rejected.slot == -1);
    CHECK(rejected.evicted_block == -1);
    CHECK(t.block_slot(1) == 0);
    CHECK(t.block_slot(2) == 1);
    CHECK(t.block_slot(3) == -1);
    CHECK(t.stats().admission_rejections == 1);

    // Repeated selection makes block 3 hotter than the old residents, so its
    // next stage-out is admitted and evicts the LRU resident.
    t.begin_selection_epoch();
    t.record_selected(3);
    t.begin_selection_epoch();
    t.record_selected(3);
    auto admitted = t.place_block_evicting(3);
    CHECK(admitted.slot >= 0);
    CHECK(admitted.evicted_block >= 0);
    CHECK(t.block_slot(3) == admitted.slot);
    CHECK(t.stats().evictions == 1);
    CHECK(t.stats().selection_epochs == 3);
}

static void test_heat_aware_replacement_before_logical_full() {
    PinnedKvTierConfig cfg;
    cfg.total_bytes = 24ull * 1024 * 1024;
    cfg.slot_bytes = 8ull * 1024 * 1024;       // 3 logical slots
    cfg.cache_policy = PinnedKvCachePolicy::HeatAware;
    PinnedKvTier t(cfg);
    CHECK(t.place_block(1).slot == 0);
    CHECK(t.place_block(2).slot == 1);
    CHECK(t.free_slots() == 1);

    // Immutable raw-K buffers can exhaust the shared CPU byte budget while the
    // logical tier still has a free slot. A cold candidate must bypass CPU.
    t.begin_selection_epoch();
    t.record_selected(1);
    t.record_selected(2);
    auto rejected = t.place_block_replacing(3);
    CHECK(rejected.slot == -1);
    CHECK(rejected.evicted_block == -1);
    CHECK(t.free_slots() == 1);

    // Repeated semantic reuse promotes the candidate. It reuses an allocated
    // resident slot and does not consume the unavailable logical free slot.
    t.begin_selection_epoch();
    t.record_selected(3);
    t.begin_selection_epoch();
    t.record_selected(3);
    auto admitted = t.place_block_replacing(3);
    CHECK(admitted.slot >= 0);
    CHECK(admitted.evicted_block >= 0);
    CHECK(t.block_slot(3) == admitted.slot);
    CHECK(t.free_slots() == 1);
    CHECK(t.used_slots() == 2);
}

static void test_exact_lifetime_and_generations() {
    PinnedKvTierConfig cfg{64, 64};
    cfg.handle_addressed = true;
    PinnedKvTier tier(cfg), foreign(cfg);
    PinnedAllocationHandle previous;
    uint8_t arena[64]{};
    for (unsigned generation = 1; generation <= 128; ++generation) {
        deny_allocation = true;
        auto p = tier.try_allocate();
        CHECK(p.slot == 0 && p.allocation.generation == generation);
        CHECK(!tier.acquire_io(previous) && !foreign.acquire_io(p.allocation));
        if (generation > 1) CHECK(tier.try_release_allocation(previous) == PinnedReleaseResult::Stale);
        auto io = tier.acquire_io(p.allocation);
        CHECK(io && tier.owns(io) && !foreign.owns(io));
        arena[0] = uint8_t(generation);
        CHECK(tier.try_release_allocation(p.allocation) == PinnedReleaseResult::Deferred);
        CHECK(!tier.acquire_io(p.allocation) && tier.free_slots() == 0 && tier.try_allocate().slot < 0);
        PinnedIoLease moved(std::move(io));
        CHECK(!io && moved && arena[0] == uint8_t(generation));
        moved.reset();
        CHECK(tier.try_release_allocation(p.allocation) == PinnedReleaseResult::Released);
        CHECK(tier.try_release_allocation(p.allocation) == PinnedReleaseResult::AlreadyReleased);
        CHECK(tier.free_slots() == 1);
        previous = p.allocation;
        deny_allocation = false;
    }
    bool refused = false;
    try { tier.place_block(0); } catch (const std::logic_error &) { refused = true; }
    CHECK(refused);
}

static void test_eviction_and_reset_with_old_io() {
    PinnedKvTier tier({128, 64});
    auto a = tier.place_block(1), b = tier.place_block(2);
    auto old = tier.acquire_io(a.allocation);
    auto replacement = tier.place_block_evicting(3);
    CHECK(replacement.slot == b.slot && replacement.evicted_block == 2);
    CHECK(tier.block_slot(1) == a.slot && !tier.acquire_io(b.allocation));
    auto other = tier.acquire_io(replacement.allocation);
    CHECK(tier.place_block_evicting(4).slot < 0);
    CHECK(tier.acquire_io(a.allocation)); // unsuccessful eviction did not close it
    tier.clear();
    CHECK(tier.block_slot(1) < 0 && tier.block_slot(3) < 0 && !tier.free_slots());
    CHECK(!tier.acquire_io(a.allocation) && !tier.acquire_io(replacement.allocation));
    CHECK(tier.place_block(1).slot < 0); // same block ID cannot resurrect the old slot
    old.reset(); tier.collect_released();
    CHECK(tier.free_slots() == 1);
    auto fresh = tier.place_block(1);
    CHECK(fresh.slot == a.slot && fresh.allocation.generation > a.allocation.generation);
    CHECK(tier.try_release_allocation(a.allocation) == PinnedReleaseResult::Stale);
    CHECK(tier.block_slot(1) == fresh.slot);
    other.reset(); tier.collect_released();
    tier.release_block(1);
    CHECK(tier.free_slots() == 2);

    PinnedKvTierConfig heat_cfg{128, 64}; heat_cfg.cache_policy = PinnedKvCachePolicy::HeatAware;
    PinnedKvTier heat(heat_cfg);
    heat.place_block(1); heat.place_block(2);
    heat.begin_selection_epoch(); heat.record_selected(1); heat.touch(2);
    CHECK(heat.lru_victim() == 1);
    // A candidate colder than the LRU is rejected even if a later resident is cold.
    // Adding lifetime protection must not change this established policy.
    CHECK(heat.place_block_evicting(3).slot < 0 && heat.block_slot(2) >= 0);
}

static void test_background_close_during_copy() {
    PinnedKvTierConfig cfg{64, 64}; cfg.handle_addressed = true;
    PinnedKvTier tier(cfg);
    uint8_t arena[64]{};
    for (unsigned round = 0; round < 64; ++round) {
        auto allocation = tier.try_allocate().allocation;
        std::fill(std::begin(arena), std::end(arena), uint8_t(round));
        auto copy = tier.acquire_io(allocation); // captured before dispatch
        std::atomic<bool> closed{false}, done{false};
        bool bytes_ok = false;
        std::thread reader([&, copy = std::move(copy)]() mutable {
            while (!closed.load(std::memory_order_acquire)) std::this_thread::yield();
            bytes_ok = std::all_of(std::begin(arena), std::end(arena), [&](uint8_t value) { return value == uint8_t(round); });
            copy.reset();
            done.store(true, std::memory_order_release);
        });
        CHECK(tier.try_release_allocation(allocation) == PinnedReleaseResult::Deferred);
        CHECK(!tier.acquire_io(allocation) && tier.try_allocate().slot < 0);
        closed.store(true, std::memory_order_release);
        while (!done.load(std::memory_order_acquire)) std::this_thread::yield();
        CHECK(tier.try_release_allocation(allocation) == PinnedReleaseResult::Released);
        reader.join();
        CHECK(bytes_ok && tier.free_slots() == 1);
    }
}

int main() {
    test_sizing();
    test_disabled();
    test_place_release();
    test_full_no_evict();
    test_full_with_explicit_evict();
    test_lru_order();
    test_heat_aware_admission();
    test_heat_aware_replacement_before_logical_full();
    test_exact_lifetime_and_generations();
    test_eviction_and_reset_with_old_io();
    test_background_close_during_copy();

    if (g_fail != 0) {
        std::printf("FAILED: %d check(s)\n", g_fail);
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
