#include "kvmem/kvmem_gc.hpp"
#include "kvmem/kvmem_store.hpp"

#include <atomic>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <thread>

using namespace kvmem;

static void check(bool ok, const char *why) {
    if (!ok) throw std::runtime_error(why);
}

template <typename F> static void eventually(F predicate, const char *why) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (!predicate()) {
        if (std::chrono::steady_clock::now() >= deadline) throw std::runtime_error(why);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
}

static KvMemGcConfig config() {
    KvMemGcConfig c;
    c.max_live_blocks = 3;
    c.scan_blocks = 2;
    c.batch_blocks = 1;
    c.poll_ms = 1;
    c.grace_steps = 1;
    c.candidate_steps = 1;
    return c;
}

static void test_policy() {
    KvMemGcConfig c;
    KvMemGcFeatures f;
    f.step = 100;
    check(kvmem_gc_score(f, c).level == 2, "unknown score must not mean low score");
    f.score_valid = true;
    f.low_streak = 3;
    check(kvmem_gc_score(f, c).level == 1, "sustained low score was ignored");
    f.duplicate = true;
    check(kvmem_gc_score(f, c).level == 0, "duplicate priority was lost");
    f.last_use = 99;
    check(kvmem_gc_score(f, c).level == 3, "recent block was not protected");
    f.anchor = true;
    check(kvmem_gc_score(f, c).level == 4, "anchor must dominate");
}

static void test_capacity_failure_is_atomic() {
    auto gc = std::make_shared<KvMemGc>(config(), 1, [](uint32_t) {
        return []() { return true; };
    });
    KvMemStoreConfig sc;
    sc.block_tokens = 4;
    KvMemStore store(sc);
    store.attach_gc(gc);
    bool rejected = false;
    try { store.register_append(8); }
    catch (const std::out_of_range &) { rejected = true; }
    check(rejected && store.block_count() == 0 && store.total_tokens() == 0,
          "capacity error left a partial block table");
    gc->stop();
}

static void test_store_holes_and_deferred_release() {
    std::atomic<unsigned> releases{0};
    KvMemGcConfig c = config();
    auto gc = std::make_shared<KvMemGc>(c, 8, [&](uint32_t id) {
        check(id == 2, "wrong victim");
        return [&]() { ++releases; return true; };
    });
    KvMemStoreConfig sc;
    sc.block_tokens = 4;
    sc.select_budget = 16; // All-fit paths must still respect dead holes.
    sc.prefill_budget = 16;
    sc.sink_blocks = 1;
    sc.recent_blocks = 1;
    KvMemStore store(sc);
    store.attach_gc(gc);
    store.register_append(16);
    for (uint32_t id = 0; id < 4; ++id) {
        store.set_block_tier(id, KvTier::SSD, -1, static_cast<int32_t>(id));
        store.set_block_ssd_backing(id, static_cast<int32_t>(id), true);
        gc->update_flags(id, KvMemGc::RawBacked, KvMemGc::RawBacked);
    }
    gc->update_flags(1, KvMemGc::Anchor, KvMemGc::Anchor);
    auto old_reader = gc->read_view();
    for (int i = 0; i < 20; ++i) gc->begin_step();
    eventually([&]() { return gc->stats().candidates > 0; }, "candidate was not observed");
    gc->begin_step();
    eventually([&]() { return !gc->is_live(2, gc->read_view()); },
               "cold historical block was not retired");
    store.gc_begin_step();
    check(store.pick_topk_blocks() == std::vector<uint32_t>({0, 1, 3}),
          "all-fit top-k resurrected a hole");
    check(store.pick_prefill_pressure_blocks() == std::vector<uint32_t>({0, 1, 3}),
          "all-fit prefill resurrected a hole");
    check(store.pick_semantic_groups({}, {}) == std::vector<uint32_t>({0, 1, 3}),
          "all-fit semantic selection resurrected a hole");
    check(store.gc_missing_required_prefix_block(4, 12, 8) == UINT32_MAX,
          "recomputed suffix invalidated a shorter checkpoint");
    check(store.gc_missing_required_prefix_block(4, 12, 12) == 2,
          "deleted checkpoint prefix went unnoticed");
    bool rejected = false;
    try { (void)store.set_selection({2}); }
    catch (const std::runtime_error &) { rejected = true; }
    check(rejected, "explicit selection of deleted block did not fail closed");
    check(releases == 0, "payload released under an old reader");
    store.gc_end_step();
    old_reader.reset();
    eventually([&]() { return releases == 1; }, "deferred release did not run");
    gc->stop();
}

static void test_late_reader_rescue() {
    std::atomic<unsigned> releases{0};
    auto c = config();
    c.max_live_blocks = 2;
    KvMemGc gc(c, 4, [&](uint32_t) {
        return [&]() { ++releases; return true; };
    });
    auto reader = gc.read_view();
    for (uint32_t id = 0; id < 3; ++id) {
        gc.register_block(id, true);
        gc.update_flags(id, KvMemGc::Gpu | KvMemGc::RawBacked | KvMemGc::ValueBacked,
                        KvMemGc::RawBacked | KvMemGc::ValueBacked);
    }
    gc.update_flags(0, KvMemGc::Anchor, KvMemGc::Anchor);
    gc.update_flags(1, KvMemGc::Anchor, KvMemGc::Anchor);
    for (int i = 0; i < 20; ++i) gc.begin_step();
    eventually([&]() { return gc.stats().candidates > 0; }, "candidate was not observed");
    gc.begin_step();
    eventually([&]() { return !gc.is_live(2, gc.read_view()); }, "victim was not retired");
    auto pin = gc.pin({2}); // Transfer an old directory read into a working lease.
    reader.reset();
    eventually([&]() { return gc.stats().cancelled > 0; }, "late use was not rescued");
    check(releases == 0, "rescued block was freed");
    pin.reset();
    gc.stop();
}

static void test_shadow_never_releases() {
    auto c = config();
    c.max_live_blocks = 1;
    c.shadow = true;
    KvMemGc gc(c, 8, [](uint32_t) -> std::function<bool()> {
        throw std::runtime_error("shadow invoked a physical release callback");
    });
    for (uint32_t id = 0; id < 4; ++id) {
        gc.register_block(id, true);
        gc.update_flags(id, KvMemGc::Gpu | KvMemGc::RawBacked | KvMemGc::ValueBacked,
                        KvMemGc::RawBacked | KvMemGc::ValueBacked);
    }
    gc.update_flags(0, KvMemGc::Anchor, KvMemGc::Anchor);
    gc.update_flags(1, KvMemGc::Io, KvMemGc::Io);
    gc.update_flags(2, KvMemGc::CpuCached, KvMemGc::CpuCached);
    for (int i = 0; i < 20; ++i) gc.begin_step();
    eventually([&]() { return gc.stats().candidates > 0; }, "shadow candidate missing");
    gc.begin_step();
    eventually([&]() { return gc.stats().shadow_candidates == 1; }, "shadow nomination missing");
    auto view = gc.read_view();
    for (uint32_t id = 0; id < 4; ++id) {
        check(gc.is_live(id, view), "shadow mode deleted historical data");
    }
    check(gc.stats().reclaimed == 0 && gc.stats().failures == 0,
          "shadow mode touched physical storage");
    gc.begin_step();
    gc.used(3);
    eventually([&]() { return gc.stats().shadow_reuses == 1; }, "shadow reuse was not recorded");
    gc.stop();
}

static void test_generation_avoids_stale_delete() {
    std::atomic<unsigned> releases{0};
    auto c = config();
    c.max_live_blocks = 2;
    KvMemGc gc(c, 4, [&](uint32_t) {
        return [&]() { ++releases; return true; };
    });
    auto old_reader = gc.read_view();
    for (uint32_t id = 0; id < 3; ++id) {
        gc.register_block(id, true);
        gc.update_flags(id, KvMemGc::Gpu | KvMemGc::RawBacked | KvMemGc::ValueBacked,
                        KvMemGc::RawBacked | KvMemGc::ValueBacked);
    }
    gc.update_flags(0, KvMemGc::Anchor, KvMemGc::Anchor);
    gc.update_flags(1, KvMemGc::Anchor, KvMemGc::Anchor);
    for (int i = 0; i < 20; ++i) gc.begin_step();
    eventually([&]() { return gc.stats().candidates > 0; }, "candidate missing");
    gc.begin_step();
    eventually([&]() { return !gc.is_live(2, gc.read_view()); }, "victim missing");
    auto stale_view = gc.read_view();
    old_reader.reset();
    eventually([&]() { return releases == 1; }, "first generation was not released");
    const auto first_generation = gc.generation(2);
    gc.forget_tail(2);
    gc.register_block(2, false);
    check(gc.generation(2) != first_generation && gc.is_live(2, stale_view),
          "stale tombstone affected a re-appended block");
    gc.stop();
}

int main() {
    try {
        test_policy();
        test_capacity_failure_is_atomic();
        test_store_holes_and_deferred_release();
        test_late_reader_rescue();
        test_shadow_never_releases();
        test_generation_avoids_stale_delete();
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
    std::cout << "KVMem historical GC tests passed\n";
}
