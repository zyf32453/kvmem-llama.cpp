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
        release_result = store->release(repo, handle);
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
    check(!store->write_layer(*writer, old, 0,
              {1, 2, 3, 4, 5, 6, 7, 8, 9}, {10, 11, 12, 13, 14, 15},
              {1.0f, 2.0f, 3.0f, 4.0f}, 2),
          "incomplete whole-layer mean was accepted for publication");
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
    auto info = store->inspect(repo, old);
    check(info && info->original_begin == 713 && info->valid_tokens == 3 &&
          info->written_layers == 2 && info->payload_bytes == old_bytes,
          "explicit original span or byte accounting was lost");
    check(store->publish_complete(repo, *writer, old),
          "complete RAM payload did not publish");
    check(store->release(repo, old).code == HistoricalKvRawStore::ReleaseCode::Busy &&
          store->allocated_payload_bytes() == old_bytes,
          "public release freed a Live block before repository reclamation");

    HistoricalKvRepository foreign(1, 83);
    HistoricalKvRawStore::ReleaseResult foreign_release;
    auto foreign_writer = foreign.begin_write(901, 713, 3, 1, old_min_bytes,
        [&](const HistoricalKvHandle & handle) {
            foreign_release = store->release(foreign, handle);
            return foreign_release.code == HistoricalKvRawStore::ReleaseCode::Stale;
        });
    check(foreign_writer &&
          foreign_writer->handle().incarnation == old.incarnation &&
          foreign_writer->handle().slot == old.slot &&
          foreign_writer->handle().generation == old.generation,
          "foreign repository did not reproduce an equal external handle");
    check(!store->inspect(foreign, old) &&
          !store->read_layer(*foreign_writer, old, 0) &&
          !store->write_layer(*foreign_writer, old, 0,
              {1, 2, 3, 4, 5, 6, 7, 8, 9}, {10, 11, 12, 13, 14, 15},
              {1.0f, 2.0f, 3.0f, 4.0f}, 3),
          "foreign lease accessed the owner's RAM payload");
    check(foreign.retire(old) == HistoricalKvRepository::RetireResult::Committed,
          "foreign block retirement failed");
    foreign_writer->reset();
    check(foreign.try_reclaim(old) == HistoricalKvRepository::ReclaimResult::Complete &&
          foreign_release.code == HistoricalKvRawStore::ReleaseCode::Stale &&
          store->allocated_payload_bytes() == old_bytes,
          "foreign repository released the owner's RAM payload");

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
          store->allocated_payload_bytes() == 0 && !store->inspect(repo, old),
          "physical payload bytes were not returned after reclaim");

    // A different repository may use the same externally chosen incarnation
    // and first-generation handle after the old allocation has gone. The
    // previous repository must not be able to release the new owner's bytes.
    HistoricalKvRepository replacement(1, 83);
    HistoricalKvRawStore::ReleaseResult replacement_release;
    auto replacement_writer = replacement.begin_write(903, 1500, 2, 1, 8,
        [&](const HistoricalKvHandle & handle) {
            replacement_release = store->release(replacement, handle);
            return callback_succeeded(replacement_release.code);
        });
    check(replacement_writer &&
          replacement_writer->handle().incarnation == old.incarnation &&
          replacement_writer->handle().slot == old.slot &&
          replacement_writer->handle().generation == old.generation,
          "replacement did not reuse the same external handle");
    check(store->reserve(*replacement_writer, old, 1500, 2) &&
          store->write_layer(*replacement_writer, old, 0,
              {1, 2, 3, 4, 5, 6}, {7, 8, 9, 10}, {1, 1, 1, 1}, 2) &&
          store->write_layer(*replacement_writer, old, 1,
              {11, 12, 13, 14, 15, 16}, {17, 18, 19, 20}, {2, 2, 2, 2}, 2) &&
          store->publish_complete(replacement, *replacement_writer, old),
          "replacement repository could not own freed physical storage");
    const uint64_t replacement_bytes = store->allocated_payload_bytes();
    check(store->release(repo, old).code == HistoricalKvRawStore::ReleaseCode::Stale &&
          store->allocated_payload_bytes() == replacement_bytes,
          "old repository released replacement repository's data");
    check(replacement.retire(old) == HistoricalKvRepository::RetireResult::Committed,
          "replacement retirement failed");
    replacement_writer->reset();
    check(replacement.try_reclaim(old) == HistoricalKvRepository::ReclaimResult::Complete &&
          replacement_release.code == HistoricalKvRawStore::ReleaseCode::Released &&
          store->allocated_payload_bytes() == 0,
          "replacement's own callback did not release its data");

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
          store->publish_complete(repo, *new_writer, current),
          "new generation did not publish");

    HistoricalKvRawStore::ReleaseResult stale_release;
    std::thread stale_reclaimer([&] { stale_release = store->release(repo, old); });
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

static void test_incremental_unaligned_tail_retired_while_writing() {
    HistoricalKvRawStoreConfig config;
    config.n_layer = 2;
    config.max_tokens_per_slot = 4;
    config.packed_k_bytes_per_token = 2;
    config.packed_v_bytes_per_token = 2;
    config.mean_k_elements = 2;
    HistoricalKvRawStore store(config, 1);
    HistoricalKvRepository repo(1, 96);
    HistoricalKvRawStore::ReleaseResult released;
    std::thread::id release_thread;
    auto writer = repo.begin_write(17, 7, 3, 1, 40,
        [&](const HistoricalKvHandle & handle) {
            release_thread = std::this_thread::get_id();
            released = store.release(repo, handle);
            return callback_succeeded(released.code);
        });
    check(writer.has_value(), "unaligned tail writer could not be reserved");
    const HistoricalKvHandle handle = writer->handle();
    check(store.reserve(*writer, handle, 7, 3),
          "unaligned partial span could not reserve a physical slot");
    const uint64_t allocated = store.allocated_payload_bytes();

    check(store.write_slice(*writer, handle, 0, 0, 1,
              {1, 2}, {7, 8}, {1.0f, 2.0f}, 1) &&
          store.write_slice(*writer, handle, 1, 0, 1,
              {21, 22}, {27, 28}, {5.0f, 6.0f}, 1),
          "first incremental K/V/mean slice failed");
    check(!store.write_slice(*writer, handle, 0, 2, 1,
              {5, 6}, {11, 12}, {3.0f, 4.0f}, 1) &&
          !store.write_slice(*writer, handle, 0, 0, 1,
              {1, 2}, {7, 8}, {1.0f, 2.0f}, 1) &&
          !store.write_slice(*writer, handle, 0, 1, 2,
              {3, 4}, {9, 10, 11, 12}, {3.0f, 4.0f}, 2),
          "hole, duplicate or short packed slice was accepted");
    auto partial = store.inspect(repo, handle);
    check(partial && partial->original_begin == 7 &&
          partial->valid_tokens == 3 && partial->written_layers == 0 &&
          !store.read_layer(*writer, handle, 0),
          "partial slice was exposed as a complete layer");

    check(repo.retire(handle) == HistoricalKvRepository::RetireResult::Committed &&
          !repo.acquire_new(handle) &&
          repo.try_reclaim(handle) == HistoricalKvRepository::ReclaimResult::Deferred,
          "retired partial block admitted new work or freed an old writer");
    check(store.write_slice(*writer, handle, 0, 1, 2,
              {3, 4, 5, 6}, {9, 10, 11, 12}, {3.0f, 4.0f}, 2) &&
          store.write_slice(*writer, handle, 1, 1, 2,
              {23, 24, 25, 26}, {29, 30, 31, 32}, {7.0f, 8.0f}, 2),
          "previously admitted writer could not finish after retirement");
    auto layer = store.read_layer(*writer, handle, 0);
    auto complete = store.inspect(repo, handle);
    check(layer && complete && complete->written_layers == 2 &&
          layer->packed_k == std::vector<uint8_t>({1, 2, 3, 4, 5, 6}) &&
          layer->packed_v == std::vector<uint8_t>({7, 8, 9, 10, 11, 12}) &&
          layer->mean_k_sum == std::vector<float>({4.0f, 6.0f}) &&
          layer->mean_tokens == 3,
          "incremental slice content or mean accumulation was incorrect");
    check(!store.publish_complete(repo, *writer, handle) &&
          store.allocated_payload_bytes() == allocated,
          "retired writer republished or prematurely released its block");
    writer->reset();
    HistoricalKvRepository::ReclaimResult result =
        HistoricalKvRepository::ReclaimResult::Retry;
    std::thread reclaimer([&] { result = repo.try_reclaim(handle); });
    reclaimer.join();
    check(result == HistoricalKvRepository::ReclaimResult::Complete &&
          released.code == HistoricalKvRawStore::ReleaseCode::Released &&
          released.physical_payload_bytes == allocated &&
          release_thread != std::this_thread::get_id() &&
          store.allocated_payload_bytes() == 0,
          "retired unaligned tail was not physically reclaimed by the worker");
}

static void test_abandoned_partial_writer_reclaims_payload() {
    HistoricalKvRawStoreConfig config;
    config.n_layer = 1;
    config.max_tokens_per_slot = 4;
    config.packed_k_bytes_per_token = 1;
    config.packed_v_bytes_per_token = 1;
    HistoricalKvRawStore store(config, 1);
    HistoricalKvRepository repo(1, 97);
    HistoricalKvRawStore::ReleaseResult released;
    auto writer = repo.begin_write(19, 11, 3, 1, 6,
        [&](const HistoricalKvHandle & handle) {
            released = store.release(repo, handle);
            return callback_succeeded(released.code);
        });
    check(writer.has_value(), "abandoned writer could not reserve repository slot");
    const HistoricalKvHandle handle = writer->handle();
    check(store.reserve(*writer, handle, 11, 3) &&
          store.write_slice(*writer, handle, 0, 0, 1,
              {1}, {2}, {}, 0),
          "abandoned writer did not create partial payload");
    const uint64_t bytes = store.allocated_payload_bytes();
    check(bytes >= 6 && !store.read_layer(*writer, handle, 0),
          "partial abandoned payload was mistaken for a complete layer");
    check(!store.publish_complete(repo, *writer, handle),
          "partial abandoned writer published an incomplete layer");
    check(repo.retire(handle) == HistoricalKvRepository::RetireResult::Committed,
          "partial writer could not be retired");
    writer->reset();
    std::thread reclaimer([&] { repo.try_reclaim(handle); });
    reclaimer.join();
    check(released.code == HistoricalKvRawStore::ReleaseCode::Released &&
          released.physical_payload_bytes == bytes &&
          store.allocated_payload_bytes() == 0 &&
          repo.ready_count() == 1,
          "abandoned partial payload was not reclaimed");
}

static void test_published_writer_cannot_mutate_payload() {
    HistoricalKvRawStoreConfig config;
    config.n_layer = 1;
    config.max_tokens_per_slot = 3;
    config.packed_k_bytes_per_token = 1;
    config.packed_v_bytes_per_token = 1;
    HistoricalKvRawStore store(config, 1);
    HistoricalKvRepository repo(1, 98);
    auto writer = repo.begin_write(20, 0, 3, 1, 6,
        [&](const HistoricalKvHandle & handle) {
            return callback_succeeded(store.release(repo, handle).code);
        });
    check(writer.has_value(), "publication test writer missing");
    const HistoricalKvHandle handle = writer->handle();
    check(store.reserve(*writer, handle, 0, 3) &&
          store.write_slice(*writer, handle, 0, 0, 1,
              {1}, {2}, {}, 0) &&
          !store.publish_complete(repo, *writer, handle) &&
          repo.publish_write(*writer),
          "publication test could not establish a Live generation");
    check(!store.write_slice(*writer, handle, 0, 1, 2,
              {3, 4}, {5, 6}, {}, 0) &&
          !store.write_layer(*writer, handle, 0,
              {1, 3, 4}, {2, 5, 6}, {}, 0),
          "published writer mutated an immutable payload");
    check(repo.retire(handle) == HistoricalKvRepository::RetireResult::Committed,
          "publication test block did not retire");
    writer->reset();
    HistoricalKvRepository::ReclaimResult reclaimed =
        HistoricalKvRepository::ReclaimResult::Retry;
    std::thread reclaimer([&] { reclaimed = repo.try_reclaim(handle); });
    reclaimer.join();
    check(reclaimed == HistoricalKvRepository::ReclaimResult::Complete &&
          store.allocated_payload_bytes() == 0,
          "publication test payload did not reclaim");
}

int main() {
    test_exact_generation_ram_reclamation();
    test_incremental_unaligned_tail_retired_while_writing();
    test_abandoned_partial_writer_reclaims_payload();
    test_published_writer_cannot_mutate_payload();
    std::cout << "historical KV RAM payload checks passed\n";
}
