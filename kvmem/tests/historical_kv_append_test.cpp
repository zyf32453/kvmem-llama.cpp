#include "kvmem/historical_kv_raw_store.hpp"
#include "kvmem/kvmem_runtime.hpp"

#include <cstdio>
#include <stdexcept>
#include <thread>

using namespace kvmem;
using Repo = HistoricalKvRepository;
using Payload = HistoricalKvRawStore;

static void check(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

static KvMemRuntimeConfig config() {
    KvMemRuntimeConfig cfg;
    cfg.store.block_tokens = 4;
    cfg.store.select_budget = 8;
    cfg.store.sink_blocks = cfg.store.recent_blocks = 0;
    return cfg;
}

struct Fixture {
    std::shared_ptr<Repo> repo = std::make_shared<Repo>(8);
    std::shared_ptr<Payload> payload = std::make_shared<Payload>(
        HistoricalKvRawStoreConfig{2, 4, 1, 1, 1, {}}, 8);
    KvMemRuntime runtime{config()};

    Repo::Lease write(uint32_t id, uint32_t begin, uint32_t count) {
        const std::weak_ptr<Repo> weak = repo;
        auto writer = repo->begin_write(id, begin, count, id + 1, 0,
            [weak, storage = payload](HistoricalKvHandle handle) {
                auto owner = weak.lock();
                if (!owner) return false;
                const auto result = storage->release(*owner, handle);
                return result.code == Payload::ReleaseCode::Released ||
                       result.code == Payload::ReleaseCode::AlreadyReleased ||
                       result.code == Payload::ReleaseCode::Absent;
            });
        check(writer.has_value(), "no writer slot");
        check(payload->reserve(*writer, writer->handle(), begin, count), "payload reserve failed");
        return std::move(*writer);
    }

    Repo::ReclaimResult reclaim(HistoricalKvHandle handle) {
        auto result = Repo::ReclaimResult::Stale;
        std::thread worker([&] { result = repo->try_reclaim(handle); });
        worker.join();
        return result;
    }
};

static void append_transaction_and_tail_rollover() {
    Fixture f;
    std::vector<HistoricalKvShadowDirectory::Producer> producers;
    std::vector<Repo::Lease> writers;
    writers.push_back(f.write(0, 0, 3));
    writers.push_back(f.write(1, 4, 2)); // Gap: must reject the entire append.
    const auto bad = writers.back().handle();
    check(!f.runtime.register_historical_append(f.repo, f.payload, writers, producers), "accepted gap");
    check(writers[0] && writers[1] && f.runtime.store().total_tokens() == 0 &&
          f.runtime.historical_shadow_directory().size() == 0, "partial append leaked");
    check(f.repo->retire(bad) == Repo::RetireResult::Committed, "retire bad admission");
    writers.back().reset();
    check(f.reclaim(bad) == Repo::ReclaimResult::Complete, "failed admission cleanup");
    writers.back() = f.write(1, 3, 2);
    const auto old = writers.back().handle();
    // Already admitted work remains legal even if policy retires before its
    // metadata transaction. No new execution permission follows from append.
    check(f.repo->retire(old) == Repo::RetireResult::Committed, "retire Writing tail");
    check(f.runtime.register_historical_append(f.repo, f.payload, writers, producers), "append admitted spans");
    check(!writers[0] && !writers[1], "transaction did not consume writers");
    const auto & directory = f.runtime.historical_shadow_directory();
    auto late = directory.find(1);
    check(late && f.runtime.unbind_retired_historical_block(1), "tail unbind");
    check(f.reclaim(old) == Repo::ReclaimResult::Deferred, "late producer lost old allocation");

    producers.clear();
    writers.clear();
    writers.push_back(f.write(2, 5, 4));
    writers.push_back(f.write(3, 9, 1));
    check(f.runtime.register_historical_append(f.repo, f.payload, writers, producers), "non-aligned rollover");
    const auto & store = f.runtime.store();
    check(store.total_tokens() == 10 && store.block_count() == 4 &&
          store.blocks()[1].orig_pos_start == 3 && store.blocks()[1].n_tokens == 2 &&
          store.blocks()[2].orig_pos_start == 5 && store.blocks()[2].n_tokens == 4,
          "rollover reinterpreted old positions");
    check(store.append_tail_room() == 0 && store.additional_blocks_for_append(5) == 2,
          "fixed tail counted as free space");
    for (uint32_t pos = 0; pos < 10; ++pos) {
        const int32_t expected = pos < 3 ? 0 : pos < 5 ? 1 : pos < 9 ? 2 : 3;
        check(store.block_id_containing(pos) == expected, "non-aligned position lookup");
    }
    check(store.block_id_containing(10) == -1, "lookup past end");
    check(store.gc_missing_required_prefix_block(3, 5, 10) == 1 &&
          store.gc_missing_required_prefix_block(5, 9, 10) == UINT32_MAX,
          "retirement filter used positional block IDs");
    for (uint32_t il = 0; il < 2; ++il)
        check(f.payload->write_layer(late->writer_context->writer(), old, il,
            {1, 2}, {3, 4}, {7.0f}, 2), "late writer cannot finish original short range");
    check(!f.payload->publish_complete(*f.repo, late->writer_context->writer(), old),
          "late completion resurrected retired tail");
    late.reset();
    check(f.reclaim(old) == Repo::ReclaimResult::Complete &&
          f.runtime.confirm_reclaimed_historical_block(1), "old tail did not reclaim");
    check(!directory.acquire_read(1) && !directory.allows_new_selection(1),
          "rollover revived old history");
    const auto selected = store.pick_semantic_groups({{5, 9}}, {10.0});
    check(selected == std::vector<uint32_t>({2}), "semantic group ignored explicit spans");

    // A fixed 1-token tail is never extended, even when it is still Writing.
    f.runtime.register_append(1);
    check(store.block_count() == 5 && store.blocks()[3].n_tokens == 1 &&
          store.blocks()[4].orig_pos_start == 10 && store.blocks()[4].n_tokens == 1,
          "direct append mutated fixed tail");
    f.runtime.register_append(1); // Unmanaged tail retains legacy behavior.
    check(store.blocks()[4].n_tokens == 2 && store.append_tail_room() == 2,
          "legacy mutable tail no longer extends");
}

static void partial_binding_and_media_ranges() {
    Fixture f;
    f.runtime.register_append(3);
    HistoricalKvShadowDirectory::Producer producer;
    auto first = f.write(0, 0, 3);
    const auto handle = first.handle();
    check(!f.runtime.bind_historical_full_block(0, f.repo, handle, first, producer, f.payload),
          "full-only compatibility API accepted short block");
    check(!f.runtime.bind_historical_block(0, f.repo, handle, first, producer, {}),
          "short span accepted positional legacy storage");
    check(f.runtime.bind_historical_block(0, f.repo, handle, first, producer, f.payload), "partial bind");
    std::vector<HistoricalKvShadowDirectory::Producer> producers;
    std::vector<Repo::Lease> writers;
    writers.push_back(f.write(1, 3, 2));
    writers.push_back(f.write(2, 5, 4));
    writers.push_back(f.write(3, 9, 1));
    check(f.runtime.register_historical_append(f.repo, f.payload, writers, producers), "media spans append");
    auto & store = f.runtime.store();
    store.set_retrieval_scores({100.0, 0.0, 50.0, 1.0});
    store.set_media_ranges({{3, 6}});
    check(store.pick_topk_blocks({1}) == std::vector<uint32_t>({1, 2}),
          "media group split after non-aligned tail");
    bool rejected = false;
    try { f.runtime.truncate_to(2); } catch (const std::logic_error &) { rejected = true; }
    check(rejected && store.total_tokens() == 10, "truncate shrank admitted span");
}

static void fragmented_capacity_and_overflow() {
    Fixture f;
    KvMemGcConfig cfg;
    cfg.max_live_blocks = 2;
    auto gc = std::make_shared<KvMemGc>(cfg, 2, [](uint32_t) {
        return [] { return true; };
    });
    gc->stop(); // Capacity regression only; no legacy policy in this test.
    f.runtime.store().attach_gc(gc);
    for (uint32_t id = 0; id < 2; ++id) {
        f.runtime.register_append(1);
        HistoricalKvShadowDirectory::Producer producer;
        auto writer = f.write(id, id, 1);
        check(f.runtime.bind_historical_block(id, f.repo, writer.handle(), writer, producer, f.payload),
              "fragment binding failed");
    }
    bool rejected = false;
    try { f.runtime.register_append(1); } catch (const std::out_of_range &) { rejected = true; }
    check(rejected && f.runtime.store().total_tokens() == 2 && f.runtime.store().block_count() == 2,
          "fragmentation bypassed physical block capacity");
    rejected = false;
    try { f.runtime.register_append(UINT32_MAX); } catch (const std::overflow_error &) { rejected = true; }
    check(rejected && f.runtime.store().total_tokens() == 2, "position overflow mutated tail");
    auto zero = config();
    zero.store.block_tokens = 0;
    KvMemRuntime invalid(zero);
    rejected = false;
    try { invalid.register_append(1); } catch (const std::invalid_argument &) { rejected = true; }
    check(rejected && invalid.store().block_count() == 0, "zero block size accepted");
}

int main() {
    try {
        append_transaction_and_tail_rollover();
        partial_binding_and_media_ranges();
        fragmented_capacity_and_overflow();
        std::puts("PASS historical append: transaction, partial retirement, original spans, capacity");
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL historical append: %s\n", e.what());
        return 1;
    }
}
