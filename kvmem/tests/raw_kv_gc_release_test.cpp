#include "kvmem/historical_kv_repository.hpp"
#include "kvmem/raw_kv_store.hpp"

#include <atomic>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace kvmem;

static void check(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

static bool rejects(const std::function<void()> & action) {
    try { action(); } catch (const std::runtime_error &) { return true; }
    return false;
}

static void write_full_block(RawKvStore & raw, uint32_t start) {
    const uint8_t k0[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    const uint8_t v0[] = {13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
    const uint8_t k1[] = {25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36};
    const uint8_t v1[] = {37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48};
    const float mean0[] = {3.0f, 5.0f};
    const float mean1[] = {7.0f, 11.0f};
    raw.write_layer_k_gpu(start, 4, 0, k0);
    raw.write_layer_v_gpu(start, 4, 0, v0);
    raw.write_layer_mean_sum(start, 4, 0, mean0);
    raw.write_layer_k_gpu(start, 4, 1, k1);
    raw.write_layer_v_gpu(start, 4, 1, v1);
    raw.write_layer_mean_sum(start, 4, 1, mean1);
}

static void test_real_raw_store_callback_releases_exact_generation() {
    RawKvStoreConfig config;
    config.n_layer = 2;
    config.n_embd_k = 2;
    config.n_embd_v = 2;
    config.block_tokens = 4;
    config.k_gpu_row_bytes = 3;
    config.v_gpu_row_bytes = 3;
    auto raw = std::make_shared<RawKvStore>(config);
    HistoricalKvRepository repo(1, 91);
    std::atomic<unsigned> callbacks{0};
    RawKvStore::HistoricalGcReleaseResult first_release;
    std::thread::id callback_thread;
    auto first = repo.begin_write(100, 0, 4, 1, 80,
        [&](const HistoricalKvHandle & handle) {
            callback_thread = std::this_thread::get_id();
            first_release = raw->release_for_historical_gc(repo, handle, 0);
            callbacks.fetch_add(1, std::memory_order_relaxed);
            return first_release.code == RawKvStore::HistoricalGcReleaseCode::Released ||
                   first_release.code == RawKvStore::HistoricalGcReleaseCode::AlreadyReleased;
        });
    check(first.has_value(), "first repository writer failed");
    const HistoricalKvHandle old = first->handle();

    const uint8_t partial_k[] = {1, 2, 3, 4, 5, 6};
    raw->write_layer_k_gpu(0, 2, 0, partial_k);
    check(!raw->seal_for_historical_gc(repo, *first, old, 0, 0, 4),
          "incomplete incremental write was sealed");
    const uint8_t full_k0[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    const uint8_t full_v0[] = {13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
    const float full_mean0[] = {3.0f, 5.0f};
    raw->write_layer_k_gpu(0, 4, 0, full_k0);
    raw->write_layer_v_gpu(0, 4, 0, full_v0);
    raw->write_layer_mean_sum(0, 4, 0, full_mean0);
    check(!raw->seal_for_historical_gc(repo, *first, old, 0, 0, 4),
          "block with a missing required layer was sealed");
    write_full_block(*raw, 0);
    check(!raw->seal_for_historical_gc(repo, *first, old, 0, 1, 4),
          "nonaligned original position was accepted by the legacy block bridge");
    check(raw->seal_for_historical_gc(repo, *first, old, 0, 0, 4),
          "full RAM block did not bind to its exact generation");
    const uint8_t late_k[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    check(rejects([&] { raw->write_layer_k_gpu(0, 4, 0, late_k); }),
          "sealed positional block accepted a late write");
    check(rejects([&] { raw->clear(); }),
          "clear bypassed the repository-bound block lifetime");
    check(repo.publish_write(*first), "first writer failed to publish");
    check(raw->release_for_historical_gc(repo, old, 0).code ==
              RawKvStore::HistoricalGcReleaseCode::Busy && raw->has_block(0),
          "public release freed a Live block before repository reclamation");
    auto old_reader = repo.acquire_read(old);
    check(old_reader.has_value(), "old reader was not admitted");
    const size_t before = raw->allocated_bytes();
    check(raw->has_block(0) && repo.retire(old) == HistoricalKvRepository::RetireResult::Committed,
          "first block did not retire");
    first->reset();
    check(repo.try_reclaim(old) == HistoricalKvRepository::ReclaimResult::Deferred &&
          raw->has_block(0) && raw->allocated_bytes() == before && callbacks.load() == 0,
          "foreground retirement freed the old reader's RAM");
    uint8_t copied[12]{};
    check(raw->copy_k_gpu(0, 0, copied, 4) && copied[0] == 1 && copied[11] == 12,
          "old reader could not finish its admitted read");
    old_reader->reset();

    HistoricalKvRepository::ReclaimResult reclaim = HistoricalKvRepository::ReclaimResult::Retry;
    std::thread worker([&] { reclaim = repo.try_reclaim(old); });
    worker.join();
    check(reclaim == HistoricalKvRepository::ReclaimResult::Complete &&
          callbacks.load() == 1 && callback_thread != std::this_thread::get_id() &&
          first_release.code == RawKvStore::HistoricalGcReleaseCode::Released &&
          first_release.physical_payload_bytes > 0,
          "physical release did not run on the background callback");
    check(!raw->has_block(0) &&
          before - raw->allocated_bytes() == first_release.physical_payload_bytes,
          "real RawKvStore payload capacities were not returned exactly");

    RawKvStore::HistoricalGcReleaseResult second_release;
    auto second = repo.begin_write(101, 4, 4, 2, 80,
        [&](const HistoricalKvHandle & handle) {
            second_release = raw->release_for_historical_gc(repo, handle, 1);
            return second_release.code == RawKvStore::HistoricalGcReleaseCode::Released ||
                   second_release.code == RawKvStore::HistoricalGcReleaseCode::AlreadyReleased;
        });
    check(second && second->handle().slot == old.slot &&
          second->handle().generation == old.generation + 1,
          "repository slot was not reused at a new generation");
    const HistoricalKvHandle current = second->handle();
    write_full_block(*raw, 4);
    check(raw->seal_for_historical_gc(repo, *second, current, 1, 4, 4) &&
          repo.publish_write(*second), "new generation was not sealed and published");
    RawKvStore::HistoricalGcReleaseResult stale;
    std::thread stale_worker([&] { stale = raw->release_for_historical_gc(repo, old, 1); });
    stale_worker.join();
    check(stale.code == RawKvStore::HistoricalGcReleaseCode::Stale &&
          stale.physical_payload_bytes == 0 && raw->has_block(1) &&
          repo.retire(old) == HistoricalKvRepository::RetireResult::Stale,
          "old handle damaged a different generation");
    check(repo.retire(current) == HistoricalKvRepository::RetireResult::Committed,
          "second block retirement failed");
    second->reset();
    std::thread final_worker([&] { reclaim = repo.try_reclaim(current); });
    final_worker.join();
    check(reclaim == HistoricalKvRepository::ReclaimResult::Complete &&
          second_release.code == RawKvStore::HistoricalGcReleaseCode::Released &&
          !raw->has_block(1), "second block was not reclaimed");
}

static void test_retired_incremental_writer_can_seal_then_release() {
    RawKvStoreConfig config;
    config.n_layer = 1;
    config.n_embd_k = 2;
    config.block_tokens = 4;
    config.k_gpu_row_bytes = 3;
    config.v_gpu_row_bytes = 3;
    auto raw = std::make_shared<RawKvStore>(config);
    HistoricalKvRepository repo(1, 92);
    RawKvStore::HistoricalGcReleaseResult released;
    std::thread::id callback_thread;
    auto writer = repo.begin_write(200, 0, 4, 1, 32,
        [&](const HistoricalKvHandle & handle) {
            callback_thread = std::this_thread::get_id();
            released = raw->release_for_historical_gc(repo, handle, 0);
            return released.code == RawKvStore::HistoricalGcReleaseCode::Released;
        });
    check(writer.has_value(), "incremental writer was not reserved");
    const HistoricalKvHandle handle = writer->handle();
    const uint8_t first_k[] = {1, 2, 3, 4, 5, 6};
    const uint8_t last_k[] = {7, 8, 9, 10, 11, 12};
    const uint8_t first_v[] = {13, 14, 15, 16, 17, 18};
    const uint8_t last_v[] = {19, 20, 21, 22, 23, 24};
    const float first_sum[] = {1.0f, 2.0f};
    const float last_sum[] = {3.0f, 4.0f};
    raw->write_layer_k_gpu(0, 2, 0, first_k);
    raw->write_layer_v_gpu(0, 2, 0, first_v);
    raw->write_layer_mean_sum(0, 2, 0, first_sum);
    check(repo.retire(handle) == HistoricalKvRepository::RetireResult::Committed &&
          repo.try_reclaim(handle) == HistoricalKvRepository::ReclaimResult::Deferred,
          "retired incremental writer was released before its old lease ended");
    raw->write_layer_k_gpu(2, 2, 0, last_k);
    raw->write_layer_v_gpu(2, 2, 0, last_v);
    raw->write_layer_mean_sum(2, 2, 0, last_sum);
    check(raw->seal_for_historical_gc(repo, *writer, handle, 0, 0, 4),
          "old writer could not seal its completed block after retirement");
    check(!repo.publish_write(*writer),
          "retired writer resurrected the historical block");
    const size_t before = raw->allocated_bytes();
    writer->reset();
    HistoricalKvRepository::ReclaimResult outcome =
        HistoricalKvRepository::ReclaimResult::Retry;
    std::thread worker([&] { outcome = repo.try_reclaim(handle); });
    worker.join();
    check(outcome == HistoricalKvRepository::ReclaimResult::Complete &&
          released.code == RawKvStore::HistoricalGcReleaseCode::Released &&
          callback_thread != std::this_thread::get_id() && !raw->has_block(0) &&
          before - raw->allocated_bytes() == released.physical_payload_bytes,
          "retired writer's RAM was not physically released by the worker");
}

static void test_seal_rejects_repository_span_mismatch() {
    RawKvStoreConfig config;
    config.n_layer = 1;
    config.n_embd_k = 2;
    config.block_tokens = 4;
    config.k_gpu_row_bytes = 3;
    config.v_gpu_row_bytes = 3;
    RawKvStore raw(config);
    HistoricalKvRepository repo(1, 93);
    auto writer = repo.begin_write(300, 4, 4, 1, 32,
        [](const HistoricalKvHandle &) { return true; });
    check(writer.has_value(), "mismatch test writer was not reserved");
    const uint8_t data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    raw.write_layer_k_gpu(0, 4, 0, data);
    raw.write_layer_v_gpu(0, 4, 0, data);
    check(!raw.seal_for_historical_gc(repo, *writer, writer->handle(), 0, 0, 4),
          "seal accepted a span different from repository metadata");
}

static void test_tail_only_write_cannot_claim_complete_block() {
    RawKvStoreConfig config;
    config.n_layer = 1;
    config.n_embd_k = 2;
    config.block_tokens = 4;
    config.k_gpu_row_bytes = 3;
    config.v_gpu_row_bytes = 3;
    RawKvStore raw(config);
    HistoricalKvRepository repo(1, 94);
    auto writer = repo.begin_write(400, 0, 4, 1, 32,
        [&](const HistoricalKvHandle & handle) {
            return raw.release_for_historical_gc(repo, handle, 0).code ==
                   RawKvStore::HistoricalGcReleaseCode::Released;
        });
    check(writer.has_value(), "hole test writer was not reserved");
    const HistoricalKvHandle handle = writer->handle();
    const uint8_t head_k[] = {1, 2, 3, 4, 5, 6};
    const uint8_t tail_k[] = {7, 8, 9, 10, 11, 12};
    const uint8_t head_v[] = {13, 14, 15, 16, 17, 18};
    const uint8_t tail_v[] = {19, 20, 21, 22, 23, 24};
    const float head_sum[] = {1.0f, 2.0f};
    const float tail_sum[] = {3.0f, 4.0f};
    const auto write = [&](uint32_t offset, const uint8_t * k,
                           const uint8_t * v, const float * sum) {
        raw.write_layer_k_gpu(offset, 2, 0, k);
        raw.write_layer_v_gpu(offset, 2, 0, v);
        raw.write_layer_mean_sum(offset, 2, 0, sum);
    };
    write(2, tail_k, tail_v, tail_sum);
    check(raw.n_tokens(0) == 4 &&
          !raw.seal_for_historical_gc(repo, *writer, handle, 0, 0, 4),
          "tail-only write was mistaken for complete K/V/mean");
    write(0, head_k, head_v, head_sum);
    check(!raw.seal_for_historical_gc(repo, *writer, handle, 0, 0, 4),
          "out-of-order head write falsely credited an earlier tail");
    write(2, tail_k, tail_v, tail_sum);
    check(raw.seal_for_historical_gc(repo, *writer, handle, 0, 0, 4) &&
          repo.publish_write(*writer),
          "contiguous second pass did not complete block coverage");
    check(repo.retire(handle) == HistoricalKvRepository::RetireResult::Committed,
          "hole test block did not retire");
    writer->reset();
    HistoricalKvRepository::ReclaimResult outcome =
        HistoricalKvRepository::ReclaimResult::Retry;
    std::thread worker([&] { outcome = repo.try_reclaim(handle); });
    worker.join();
    check(outcome == HistoricalKvRepository::ReclaimResult::Complete &&
          !raw.has_block(0), "hole test block was not reclaimed");
}

static void test_repository_identity_is_not_just_incarnation() {
    RawKvStoreConfig config;
    config.n_layer = 1;
    config.n_embd_k = 2;
    config.block_tokens = 4;
    config.k_gpu_row_bytes = 3;
    config.v_gpu_row_bytes = 3;
    RawKvStore raw(config);
    HistoricalKvRepository owner(1, 95);
    HistoricalKvRepository other(1, 95);
    RawKvStore::HistoricalGcReleaseResult foreign_release;
    auto writer = owner.begin_write(500, 0, 4, 1, 32,
        [&](const HistoricalKvHandle & handle) {
            return raw.release_for_historical_gc(owner, handle, 0).code ==
                   RawKvStore::HistoricalGcReleaseCode::Released;
        });
    auto foreign_writer = other.begin_write(501, 0, 4, 1, 32,
        [&](const HistoricalKvHandle & handle) {
            foreign_release = raw.release_for_historical_gc(other, handle, 0);
            return true; // This repository owns no RawKvStore allocation.
        });
    check(writer && foreign_writer &&
          writer->handle().incarnation == foreign_writer->handle().incarnation &&
          writer->handle().slot == foreign_writer->handle().slot &&
          writer->handle().generation == foreign_writer->handle().generation,
          "cross-repository identity test did not create matching handles");
    const HistoricalKvHandle handle = writer->handle();
    const uint8_t data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    raw.write_layer_k_gpu(0, 4, 0, data);
    raw.write_layer_v_gpu(0, 4, 0, data);
    check(!raw.seal_for_historical_gc(other, *writer, handle, 0, 0, 4),
          "foreign repository accepted another repository's writer lease");
    check(raw.seal_for_historical_gc(owner, *writer, handle, 0, 0, 4) &&
          owner.publish_write(*writer), "owner failed to seal its block");
    const size_t before = raw.allocated_bytes();
    check(other.retire(handle) == HistoricalKvRepository::RetireResult::Committed,
          "foreign repository retirement failed");
    foreign_writer->reset();
    HistoricalKvRepository::ReclaimResult outcome =
        HistoricalKvRepository::ReclaimResult::Retry;
    std::thread foreign_worker([&] { outcome = other.try_reclaim(handle); });
    foreign_worker.join();
    check(outcome == HistoricalKvRepository::ReclaimResult::Complete &&
          foreign_release.code == RawKvStore::HistoricalGcReleaseCode::Stale &&
          raw.has_block(0) && raw.allocated_bytes() == before,
          "foreign repository released the owner's RAM");
    check(owner.retire(handle) == HistoricalKvRepository::RetireResult::Committed,
          "owner retirement failed");
    writer->reset();
    std::thread owner_worker([&] { outcome = owner.try_reclaim(handle); });
    owner_worker.join();
    check(outcome == HistoricalKvRepository::ReclaimResult::Complete &&
          !raw.has_block(0), "owner did not reclaim its own RAM");
}

static void test_sparse_attention_layers_seal_and_release() {
    RawKvStoreConfig config;
    config.n_layer = 3;
    config.n_embd_k = 2;
    config.block_tokens = 4;
    config.k_gpu_row_bytes = 3;
    config.v_gpu_row_bytes = 3;
    config.historical_gc_attention_layers = {0, 2};
    config.historical_gc_require_mean_k = true;
    auto raw = std::make_shared<RawKvStore>(config);
    HistoricalKvRepository repo(1, 96);
    RawKvStore::HistoricalGcReleaseResult released;
    auto writer = repo.begin_write(600, 0, 4, 1, 80,
        [&](const HistoricalKvHandle & handle) {
            released = raw->release_for_historical_gc(repo, handle, 0);
            return released.code == RawKvStore::HistoricalGcReleaseCode::Released;
        });
    check(writer.has_value(), "sparse attention writer was not reserved");
    const HistoricalKvHandle handle = writer->handle();
    const uint8_t data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    const float mean[] = {2.0f, 4.0f};
    for (uint32_t il : {0u, 2u}) {
        raw->write_layer_k_gpu(0, 4, il, data);
        raw->write_layer_v_gpu(0, 4, il, data);
    }
    check(!raw->seal_for_historical_gc(repo, *writer, handle, 0, 0, 4),
          "retrieval block without Mean-K was sealed");
    raw->clear();
    const auto write_attention = [&] {
        for (uint32_t il : {0u, 2u}) {
            raw->write_layer_k_gpu(0, 4, il, data);
            raw->write_layer_v_gpu(0, 4, il, data);
            raw->write_layer_mean_sum(0, 4, il, mean);
        }
    };
    write_attention();
    raw->write_layer_k_gpu(0, 4, 1, data);
    check(!raw->seal_for_historical_gc(repo, *writer, handle, 0, 0, 4),
          "sparse seal accepted payload in a non-attention layer");
    raw->clear();
    write_attention();
    check(raw->seal_for_historical_gc(repo, *writer, handle, 0, 0, 4) &&
          repo.publish_write(*writer),
          "sparse attention block was not sealed and published");
    const size_t before = raw->allocated_bytes();
    check(repo.retire(handle) == HistoricalKvRepository::RetireResult::Committed,
          "sparse attention block did not retire");
    writer->reset();
    HistoricalKvRepository::ReclaimResult result =
        HistoricalKvRepository::ReclaimResult::Retry;
    std::thread worker([&] { result = repo.try_reclaim(handle); });
    worker.join();
    check(result == HistoricalKvRepository::ReclaimResult::Complete &&
          released.code == RawKvStore::HistoricalGcReleaseCode::Released &&
          released.physical_payload_bytes > 0 && !raw->has_block(0) &&
          before - raw->allocated_bytes() == released.physical_payload_bytes,
          "sparse attention block did not release its RAM payload");
}

static void test_attention_layer_list_validation() {
    RawKvStoreConfig config;
    config.n_layer = 3;
    for (const std::vector<uint32_t> invalid :
         {std::vector<uint32_t>{2, 0}, {0, 0}, {0, 3}}) {
        config.historical_gc_attention_layers = invalid;
        bool rejected = false;
        try { RawKvStore raw(config); }
        catch (const std::invalid_argument &) { rejected = true; }
        check(rejected, "invalid historical GC attention layer list was accepted");
    }
}

int main() {
    test_real_raw_store_callback_releases_exact_generation();
    test_retired_incremental_writer_can_seal_then_release();
    test_seal_rejects_repository_span_mismatch();
    test_tail_only_write_cannot_claim_complete_block();
    test_repository_identity_is_not_just_incarnation();
    test_sparse_attention_layers_seal_and_release();
    test_attention_layer_list_validation();
    std::cout << "raw historical GC release checks passed\n";
}
