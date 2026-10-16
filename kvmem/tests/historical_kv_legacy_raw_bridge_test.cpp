#include "kvmem/historical_kv_gc.hpp"
#include "kvmem/kvmem_runtime.hpp"
#include "kvmem/raw_kv_store.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace kvmem;

namespace {

void check(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

class DeleteOnce final : public HistoricalKvPolicy {
public:
    explicit DeleteOnce(HistoricalKvHandle victim) : victim_(victim) {}

    HistoricalKvStepResult step(const HistoricalKvSnapshot & snapshot,
                                const HistoricalKvPressure &,
                                HistoricalKvWorkBudget,
                                HistoricalKvDecisionBuffer & out) override {
        if (sent_) return {HistoricalKvStepStatus::NoCandidate, 0, 0};
        const auto found = std::find_if(snapshot.blocks.begin(), snapshot.blocks.end(),
            [&](const HistoricalKvBlockMeta & block) {
                return block.handle.incarnation == victim_.incarnation &&
                       block.handle.slot == victim_.slot &&
                       block.handle.generation == victim_.generation;
            });
        if (found == snapshot.blocks.end())
            return {HistoricalKvStepStatus::NoCandidate, snapshot.blocks.size(), 0};
        HistoricalKvDeleteIntent intent;
        intent.victim = victim_;
        intent.round_id = snapshot.round_id;
        intent.reason = HistoricalKvDeleteReason::Custom;
        check(out.push(intent), "GC decision buffer rejected one intent");
        sent_ = true;
        return {HistoricalKvStepStatus::Complete, snapshot.blocks.size(), 1};
    }

private:
    HistoricalKvHandle victim_;
    bool sent_ = false;
};

void test_gc_releases_real_raw_kv_after_old_active_finishes() {
    RawKvStoreConfig raw_config;
    raw_config.n_layer = 2;
    raw_config.n_embd_k = 2;
    raw_config.n_embd_v = 2;
    raw_config.block_tokens = 4;
    auto raw = std::make_shared<RawKvStore>(raw_config);
    auto repository = std::make_shared<HistoricalKvRepository>(1, 101);
    const std::weak_ptr<HistoricalKvRepository> weak_repository = repository;

    auto writer = repository->begin_write(77, 0, 4, 1, 80,
        [raw, weak_repository](const HistoricalKvHandle & handle) {
            const auto owner = weak_repository.lock();
            if (!owner) return false;
            const auto result = raw->release_for_historical_gc(*owner, handle, 0);
            return result.code == RawKvStore::HistoricalGcReleaseCode::Released ||
                   result.code == RawKvStore::HistoricalGcReleaseCode::AlreadyReleased;
        });
    check(writer.has_value(), "repository did not reserve the raw block");
    const HistoricalKvHandle handle = writer->handle();
    const std::vector<uint16_t> k(8, 0x3c00);
    const std::vector<uint16_t> v(8, 0x4000);
    for (uint32_t layer = 0; layer < raw_config.n_layer; ++layer)
        raw->write_layer_tokens_f16(0, 4, layer, k.data(), v.data());
    check(raw->has_block(0) && raw->bytes_k() > 0 && raw->bytes_v() > 0,
          "real RawKvStore was not populated");
    check(raw->seal_for_historical_gc(*repository, *writer, handle, 0, 0, 4),
          "full RAM block could not be sealed for historical GC");
    check(repository->publish_write(*writer), "repository did not publish raw block");
    const size_t before = raw->allocated_bytes();
    auto old_active = repository->acquire_new(handle);
    check(old_active.has_value(), "old active did not acquire execution permission");

    HistoricalKvGc::Config gc_config;
    gc_config.mode = HistoricalKvGc::Mode::Delete;
    gc_config.scan_blocks = 1;
    gc_config.batch_blocks = 1;
    gc_config.reclaim_blocks = 1;
    gc_config.poll_ms = 1;
    HistoricalKvGc gc(repository, std::make_unique<DeleteOnce>(handle), gc_config);
    gc.manual_tick();
    check(!repository->acquire_new(handle), "retired block entered a new active");
    check(raw->has_block(0) && raw->allocated_bytes() == before,
          "GC released real CPU data while an old active was still using it");
    writer->reset();
    gc.manual_tick();
    check(raw->has_block(0) && repository->ready_count() == 0,
          "GC released real CPU data before the old active completed");

    old_active->reset();
    gc.manual_tick();
    check(!raw->has_block(0) && raw->bytes_k() == 0 && raw->bytes_v() == 0 &&
          raw->allocated_bytes() < before,
          "GC did not return real RawKvStore K/V and mean payload bytes");
    check(repository->ready_count() == 1 && gc.stats().reclaimed == 1,
          "repository did not return its slot after real CPU release");
    check(raw->release_for_historical_gc(*repository, handle, 0).code ==
              RawKvStore::HistoricalGcReleaseCode::Busy,
          "public release bypassed the repository's Releasing phase");
}

void test_directory_reads_hold_real_payload_until_last_transfer_finishes() {
    RawKvStoreConfig raw_config;
    raw_config.n_layer = 2;
    raw_config.n_embd_k = raw_config.n_embd_v = 2;
    raw_config.block_tokens = 4;
    raw_config.k_gpu_row_bytes = raw_config.v_gpu_row_bytes = 4;
    raw_config.historical_gc_require_mean_k = true;
    auto raw = std::make_shared<RawKvStore>(raw_config);
    auto repository = std::make_shared<HistoricalKvRepository>(1, 102);
    const std::weak_ptr<HistoricalKvRepository> weak_repository = repository;
    KvMemRuntimeConfig runtime_config;
    runtime_config.store.block_tokens = 4;
    KvMemRuntime runtime(runtime_config);
    runtime.register_append(4);
    const auto & directory = runtime.historical_shadow_directory();

    auto writer = repository->begin_write(0, 0, 4, 1, 80,
        [raw, weak_repository](const HistoricalKvHandle & handle) {
            const auto owner = weak_repository.lock();
            if (!owner) return false;
            const auto result = raw->release_for_historical_gc(*owner, handle, 0);
            return result.code == RawKvStore::HistoricalGcReleaseCode::Released ||
                   result.code == RawKvStore::HistoricalGcReleaseCode::AlreadyReleased;
        });
    check(writer.has_value(), "did not reserve directory-backed raw block");
    const auto handle = writer->handle();
    const std::vector<uint8_t> k(16, 0x31), v(16, 0x72);
    const std::vector<float> mean(8, 1.0f);
    for (uint32_t layer = 0; layer < 2; ++layer) {
        raw->write_layer_mean_k(0, 4, layer, mean.data());
        raw->write_layer_k_gpu(0, 4, layer, k.data());
        raw->write_layer_v_gpu(0, 4, layer, v.data());
    }
    check(raw->seal_for_historical_gc(*repository, *writer, handle, 0, 0, 4),
          "could not seal packed raw block");
    check(repository->publish_write(*writer), "could not publish packed raw block");
    HistoricalKvShadowDirectory::Producer producer;
    check(runtime.bind_historical_full_block(0, repository, handle, *writer, producer),
          "could not bind published raw block");
    producer.reset(); // Sealed producer work is already complete.
    auto read = directory.acquire_read(0);
    auto transfer = directory.acquire_transfer(0);
    check(read && transfer && read.lease && transfer.lease,
          "directory did not admit the initial read and transfer");
    const size_t before = raw->allocated_bytes();

    HistoricalKvGc::Config gc_config;
    gc_config.mode = HistoricalKvGc::Mode::Delete;
    gc_config.scan_blocks = gc_config.batch_blocks = gc_config.reclaim_blocks = 1;
    HistoricalKvGc gc(repository, std::make_unique<DeleteOnce>(handle), gc_config);
    gc.manual_tick();
    check(!directory.acquire_read(0) && !directory.acquire_transfer(0),
          "retirement still admitted a new raw read before unbind");
    check(runtime.unbind_retired_historical_block(0), "could not unbind retired writer");
    gc.manual_tick();
    check(raw->allocated_bytes() == before && gc.stats().reclaimed == 0,
          "GC released payload with admitted CPU readers");
    std::vector<float> got_mean(2);
    raw->mean_k(0, 1, got_mean.data());
    check(got_mean == std::vector<float>(2, 1.0f),
          "admitted mean read could not finish after retirement");
    read.lease.reset();
    gc.manual_tick();
    check(raw->allocated_bytes() == before && gc.stats().reclaimed == 0,
          "GC released payload with an admitted transfer remaining");
    std::vector<uint8_t> got_k(16), got_v(16);
    check(raw->copy_k_gpu(0, 1, got_k.data(), 4) && got_k == k &&
          raw->copy_v_gpu(0, 1, got_v.data(), 4) && got_v == v,
          "admitted packed transfer could not finish after retirement");
    transfer.lease.reset();
    gc.manual_tick();
    check(!raw->has_block(0) && raw->bytes_k() == 0 && raw->bytes_v() == 0 &&
          raw->allocated_bytes() < before && gc.stats().reclaimed == 1,
          "GC did not release the actual payload after the last transfer");
    check(runtime.confirm_reclaimed_historical_block(0), "release was not acknowledged");
    check(!directory.acquire_read(0) && !directory.acquire_transfer(0),
          "reclaimed tombstone fell back to legacy raw reads");
}

} // namespace

int main() {
    test_gc_releases_real_raw_kv_after_old_active_finishes();
    test_directory_reads_hold_real_payload_until_last_transfer_finishes();
    std::cout << "historical KV real RawKvStore GC bridge checks passed\n";
}
