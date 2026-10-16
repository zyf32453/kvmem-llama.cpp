#include "kvmem/historical_kv_gc.hpp"
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

} // namespace

int main() {
    test_gc_releases_real_raw_kv_after_old_active_finishes();
    std::cout << "historical KV real RawKvStore GC bridge checks passed\n";
}
