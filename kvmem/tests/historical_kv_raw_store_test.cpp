#include "kvmem/historical_kv_raw_store.hpp"
#include "kvmem/historical_kv_repository.hpp"

#include <atomic>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace kvmem;

static void check(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

static bool callback_succeeded(HistoricalKvRawStore::ReleaseCode code) {
    return code == HistoricalKvRawStore::ReleaseCode::Released ||
           code == HistoricalKvRawStore::ReleaseCode::AlreadyReleased ||
           code == HistoricalKvRawStore::ReleaseCode::Absent;
}

static void test_exact_generation_ram_reclamation() {
    HistoricalKvRawStoreConfig config;
    config.n_layer = 2;
    config.max_tokens_per_slot = 8;
    config.packed_k_bytes_per_token = 3;
    config.packed_v_bytes_per_token = 2;
    config.mean_k_elements = 4;
    auto store = std::make_shared<HistoricalKvRawStore>(config, 1);
    HistoricalKvRepository repo(1, 83);
    constexpr uint64_t old_min_bytes = 2 * (3 * 3 + 3 * 2 + 4 * sizeof(float));
    std::atomic<unsigned> callbacks{0};
    std::thread::id release_thread;
    HistoricalKvRawStore::ReleaseResult release_result;
    auto release = [&](const HistoricalKvHandle & handle) {
        release_thread = std::this_thread::get_id();
        release_result = store->release(handle);
        callbacks.fetch_add(1, std::memory_order_relaxed);
        return callback_succeeded(release_result.code);
    };

    // Original position 713 is intentionally not aligned to the configured
    // maximum of eight tokens per physical slot.
    auto writer = repo.begin_write(901, 713, 3, 1, old_min_bytes, release);
    check(writer.has_value(), "could not reserve repository writer");
    const HistoricalKvHandle old = writer->handle();
    check(!store->reserve(*writer, old, UINT64_MAX, 3),
          "overflowing original token span was accepted");
    check(!store->reserve(*writer, old, 714, 3) &&
          !store->reserve(*writer, old, 713, 2),
          "RAM payload accepted a span different from its repository record");
    check(store->reserve(*writer, old, 713, 3), "could not reserve RAM payload");
    const uint64_t old_bytes = store->allocated_payload_bytes();
    check(old_bytes >= old_min_bytes,
          "multi-layer K/V/mean bytes were not allocated");
    check(store->write_layer(*writer, old, 0, {1, 2, 3, 4, 5, 6, 7, 8, 9},
                             {10, 11, 12, 13, 14, 15},
                             {1.0f, 2.0f, 3.0f, 4.0f}, 3),
          "first layer write failed");
    check(store->write_layer(*writer, old, 1, {21, 22, 23, 24, 25, 26, 27, 28, 29},
                             {30, 31, 32, 33, 34, 35},
                             {5.0f, 6.0f, 7.0f, 8.0f}, 3),
          "second layer write failed");
    check(!store->write_layer(*writer, old, 1, {21, 22, 23, 24, 25, 26, 27, 28, 29},
                              {30, 31, 32, 33, 34, 35},
                              {5.0f, 6.0f, 7.0f, 8.0f}, 3),
          "write-once layer was overwritten");
    auto info = store->inspect(old);
    check(info && info->original_begin == 713 && info->valid_tokens == 3 &&
          info->written_layers == 2 && info->payload_bytes == old_bytes,
          "explicit original span or byte accounting was lost");
    check(repo.publish_write(*writer), "complete RAM payload did not publish");

    auto reader = repo.acquire_read(old);
    check(reader.has_value(), "reader lease was not acquired");
    check(!store->write_layer(*reader, old, 0,
                              {1, 2, 3, 4, 5, 6, 7, 8, 9},
                              {10, 11, 12, 13, 14, 15},
                              {1.0f, 2.0f, 3.0f, 4.0f}, 3),
          "read lease gained write permission");
    auto layer0 = store->read_layer(*reader, old, 0);
    auto layer1 = store->read_layer(*reader, old, 1);
    check(layer0 && layer1 && layer0->packed_k.front() == 1 &&
          layer0->packed_v.back() == 15 && layer0->mean_k_sum[3] == 4.0f &&
          layer1->packed_k.front() == 21 && layer1->packed_v.back() == 35 &&
          layer1->mean_k_sum[3] == 8.0f && layer1->mean_tokens == 3,
          "multi-layer payload was not read back exactly");

    check(repo.retire(old) == HistoricalKvRepository::RetireResult::Committed,
          "old block did not retire");
    check(store->read_layer(*reader, old, 1).has_value(),
          "retirement interrupted a previously admitted reader");
    writer->reset();
    check(!repo.acquire_new(old), "retired block was newly admitted");
    check(repo.try_reclaim(old) == HistoricalKvRepository::ReclaimResult::Deferred,
          "RAM was released under a reader lease");
    check(store->allocated_payload_bytes() == old_bytes && callbacks.load() == 0,
          "foreground retirement freed RAM");
    reader->reset();
    check(!store->read_layer(*reader, old, 0),
          "reset reader lease still granted payload access");

    HistoricalKvRepository::ReclaimResult reclaim_result =
        HistoricalKvRepository::ReclaimResult::Retry;
    std::thread reclaimer([&] { reclaim_result = repo.try_reclaim(old); });
    reclaimer.join();
    check(reclaim_result == HistoricalKvRepository::ReclaimResult::Complete &&
          callbacks.load() == 1 && release_thread != std::this_thread::get_id(),
          "RAM release did not run once on the background reclaimer");
    check(release_result.code == HistoricalKvRawStore::ReleaseCode::Released &&
          release_result.physical_payload_bytes == old_bytes &&
          store->allocated_payload_bytes() == 0 && !store->inspect(old),
          "physical payload bytes were not returned after reclaim");

    constexpr uint64_t new_min_bytes = 2 * (2 * 3 + 2 * 2 + 4 * sizeof(float));
    auto new_writer = repo.begin_write(902, 1427, 2, 2, new_min_bytes, release);
    check(new_writer && new_writer->handle().slot == old.slot &&
          new_writer->handle().generation == old.generation + 1,
          "repository did not reuse the physical slot at a new generation");
    const HistoricalKvHandle current = new_writer->handle();
    check(store->reserve(*new_writer, current, 1427, 2), "new generation RAM reserve failed");
    const uint64_t new_bytes = store->allocated_payload_bytes();
    check(new_bytes >= new_min_bytes, "new generation allocation was incomplete");
    check(store->write_layer(*new_writer, current, 0, {1, 2, 3, 4, 5, 6},
                             {7, 8, 9, 10}, {1, 1, 1, 1}, 2) &&
          store->write_layer(*new_writer, current, 1, {11, 12, 13, 14, 15, 16},
                             {17, 18, 19, 20}, {2, 2, 2, 2}, 2) &&
          repo.publish_write(*new_writer), "new generation did not publish");

    HistoricalKvRawStore::ReleaseResult stale_release;
    std::thread stale_reclaimer([&] { stale_release = store->release(old); });
    stale_reclaimer.join();
    check(stale_release.code == HistoricalKvRawStore::ReleaseCode::Stale &&
          stale_release.physical_payload_bytes == 0 &&
          store->allocated_payload_bytes() == new_bytes &&
          !store->read_layer(*new_writer, old, 0) &&
          store->read_layer(*new_writer, current, 1).has_value(),
          "old handle modified the reused physical slot");
    check(repo.retire(old) == HistoricalKvRepository::RetireResult::Stale,
          "old repository handle retired the new generation");

    check(repo.retire(current) == HistoricalKvRepository::RetireResult::Committed,
          "new generation retirement failed");
    new_writer->reset();
    std::thread final_reclaimer([&] { reclaim_result = repo.try_reclaim(current); });
    final_reclaimer.join();
    check(reclaim_result == HistoricalKvRepository::ReclaimResult::Complete &&
          callbacks.load() == 2 &&
          release_result.physical_payload_bytes == new_bytes &&
          store->allocated_payload_bytes() == 0,
          "second generation did not return its own physical bytes");
}

int main() {
    test_exact_generation_ram_reclamation();
    std::cout << "historical KV RAM payload checks passed\n";
}
