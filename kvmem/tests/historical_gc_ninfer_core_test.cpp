#include "kvmem/historical_kv_gc.hpp"
#include "kvmem/historical_kv_page_gc.hpp"
#include "kvmem/historical_kv_redundancy.hpp"

#include <array>
#include <atomic>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

void require_at(bool condition, int line) {
    if (!condition) throw std::runtime_error("NInfer page GC contract failed at line " + std::to_string(line));
}
#define require(condition) require_at((condition), __LINE__)

class PageFeatures final : public kvmem::HistoricalKvRedundancySource {
public:
    size_t dimensions() const noexcept override { return 3; }
    bool begin_round(uint64_t) override { return true; }
    bool read(const kvmem::HistoricalKvBlockMeta & meta, float * descriptor, double & logit) override {
        const std::array<float, 3> key = meta.logical_id ?
            std::array<float, 3>{1.0f, 0.0f, 0.0f} :
            std::array<float, 3>{0.0f, 1.0f, 0.0f};
        for (size_t i = 0; i < key.size(); ++i) descriptor[i] = key[i];
        logit = 0.0;
        return true;
    }
};

} // namespace

static void native_page_handoff() {
    kvmem::HistoricalKvPageGc::Config config;
    config.slots = 3;
    config.logical_pages = 3;
    config.page_tokens = 64;
    config.sink_tokens = 64;
    config.recent_tokens = 64;
    config.low_ready = 1;
    config.high_ready = 2;
    config.batch_blocks = 3;
    kvmem::HistoricalKvPageGc bridge(config);
    for (uint32_t page = 0; page < 3; ++page)
        require(bridge.register_page(page, 1024, page + 1));
    // Sink and current checkpoint/view remain protected. Only the cold middle
    // page may be retired, even if the worker already sampled another round.
    bridge.protect_page(1, false);
    for (int i = 0; i < 8 && bridge.stats().proposed == 0; ++i) bridge.manual_tick();
    require(bridge.stats().proposed == 1);
    require(bridge.is_live(1));
    const auto& committed = bridge.commit_boundary();
    require(committed.size() == 1 && committed.front().page == 1);
    require(!bridge.is_live(1));
    bridge.manual_tick();
    require(bridge.stats().reclaimed == 0);
    kvmem::HostKvStorage native_host(1024);
    auto native_record = native_host.allocate(128);
    native_record.seal();
    auto reader = native_record.lease();
    bridge.acknowledge_host_release(committed.front(), std::move(native_record));
    for (int i = 0; i < 8; ++i) bridge.manual_tick();
    require(bridge.stats().reclaimed == 0 && native_host.charged_bytes() == 128);
    reader = {};
    for (int i = 0; i < 8 && bridge.stats().reclaimed == 0; ++i) bridge.manual_tick();
    require(bridge.stats().reclaimed == 1 && native_host.charged_bytes() == 0);

    kvmem::HistoricalKvPageGc delayed(config);
    for (uint32_t page = 0; page < 3; ++page)
        require(delayed.register_page(page, 1024, page + 1));
    delayed.protect_page(1, false);
    for (int i = 0; i < 8 && delayed.stats().proposed == 0; ++i) delayed.manual_tick();
    require(delayed.stats().proposed == 1);
    // A checkpoint can be captured after the worker's snapshot. The boundary
    // must recheck its current protection before publishing the tombstone.
    delayed.protect_page(1, true);
    require(delayed.commit_boundary().empty());
    require(delayed.is_live(1));
}

int main() {
    native_page_handoff();
    constexpr uint32_t page_tokens = 64;
    auto repository = std::make_shared<kvmem::HistoricalKvRepository>(3, 0, page_tokens, page_tokens);
    std::atomic<unsigned> releases{0};
    std::array<kvmem::HistoricalKvHandle, 3> handles{};
    for (uint32_t page = 0; page < handles.size(); ++page) {
        auto writer = repository->begin_write(page, page * page_tokens, page_tokens,
            page + 1, 1024, [&releases](const kvmem::HistoricalKvHandle &) {
                ++releases;
                return true;
            });
        require(bool(writer));
        handles[page] = writer->handle();
        require(repository->publish_write(*writer));
    }

    auto thinking = std::make_shared<kvmem::HistoricalKvThinkingRanges>();
    thinking->note(page_tokens, 3 * page_tokens);
    thinking->arm();
    auto source = std::make_shared<PageFeatures>();
    kvmem::HistoricalKvRedundancyPolicy::Config policy_config;
    policy_config.thinking = thinking;
    policy_config.pool_blocks = 3;
    policy_config.segment_blocks = 2;
    kvmem::HistoricalKvGc::Config gc_config;
    gc_config.mode = kvmem::HistoricalKvGc::Mode::Delete;
    gc_config.ready_low = 1;
    gc_config.ready_high = 2;
    gc_config.scan_blocks = 3;
    gc_config.batch_blocks = 3;
    gc_config.reclaim_blocks = 3;
    gc_config.retire_at_decode_boundary = true;
    struct BoundaryObserver {
        std::atomic<unsigned> calls{0};
        kvmem::HistoricalKvHandle retired{};
    } observer;
    gc_config.boundary_context = &observer;
    gc_config.on_boundary_retired = [](void * context, kvmem::HistoricalKvHandle handle) noexcept {
        auto& state = *static_cast<BoundaryObserver *>(context);
        state.retired = handle;
        state.calls.fetch_add(1, std::memory_order_relaxed);
    };
    kvmem::HistoricalKvGc gc(repository,
        std::make_unique<kvmem::HistoricalKvRedundancyPolicy>(3, source, policy_config), gc_config);

    for (int i = 0; i < 8 && gc.stats().proposed == 0; ++i) gc.manual_tick();
    require(gc.stats().proposed > 0);
    require(gc.stats().committed == 0);
    require(observer.calls.load(std::memory_order_relaxed) == 0);
    auto in_flight = repository->acquire_gpu(handles[1]);
    require(bool(in_flight));
    const auto boundary = gc.commit_at_decode_boundary();
    require(boundary.committed == 1);
    require(observer.calls.load(std::memory_order_relaxed) == 1);
    require(observer.retired.slot == handles[1].slot &&
            observer.retired.generation == handles[1].generation);
    require(repository->completion_state(handles[0]) == kvmem::HistoricalKvRepository::CompletionState::Live);
    require(repository->completion_state(handles[2]) == kvmem::HistoricalKvRepository::CompletionState::Live);
    require(!repository->acquire_new(handles[1]));
    gc.manual_tick();
    require(repository->ready_count() == 0);
    in_flight->reset();
    for (int i = 0; i < 8 && gc.stats().reclaimed == 0; ++i) gc.manual_tick();
    require(gc.stats().reclaimed == 1);
    require(releases == 1);
    require(repository->ready_count() == 1);
}
