#include "kvmem/historical_kv_gc.hpp"
#include "kvmem/historical_kv_page_gc.hpp"
#include "kvmem/historical_kv_page_features.hpp"
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

static void native_page_feature_snapshot() {
    kvmem::HistoricalKvPageFeatures features(3, 3);
    features.publish_page(0, {0.0f, 1.0f, 0.0f});
    features.publish_page(1, {1.0f, 2.0f, 3.0f});
    require(features.has_page(1));
    features.publish_question_scores({0.1, -0.2, 0.3});
    require(features.begin_round(1));
    kvmem::HistoricalKvBlockMeta meta;
    meta.logical_id = 1;
    std::array<float, 3> descriptor{};
    double importance = 0;
    require(features.read(meta, descriptor.data(), importance));
    require(descriptor == (std::array<float, 3>{1.0f, 2.0f, 3.0f}));
    require(importance == -0.2);
    const auto partial_epoch = features.question_epoch();
    features.publish_question_scores({0.1, 0.25, 0.3}, false);
    require(features.question_epoch() == partial_epoch);
    features.publish_output_attention(1, 0.5);
    require(features.read(meta, descriptor.data(), importance));
    require(importance == -0.2); // unchanged within the frozen round
    require(features.begin_round(1));
    require(features.read(meta, descriptor.data(), importance) && importance == 0.5);
    const auto epoch = features.question_epoch();
    features.clear_question();
    require(features.question_epoch() > epoch);
    require(features.begin_round(1));
    require(features.read(meta, descriptor.data(), importance) && importance == 0.0);
    // Restoring to 96 tokens keeps one complete 64-token page and a half
    // page. Its former full-page descriptor must not survive the rewind.
    features.truncate_to(96, 64);
    require(features.has_page(0));
    require(!features.has_page(1) && !features.read(meta, descriptor.data(), importance));
    features.publish_page(1, {3.0f, 2.0f, 1.0f});
    require(features.begin_round(2));
    require(features.read(meta, descriptor.data(), importance));
    require(descriptor == (std::array<float, 3>{3.0f, 2.0f, 1.0f}));
    features.truncate_to(128, 64); // A fully preserved page remains immutable.
    require(features.has_page(0) && features.has_page(1));
    features.truncate_to(64, 64);
    require(features.has_page(0) && !features.has_page(1));
}

static void truncated_thinking_reopens_candidates() {
    kvmem::HistoricalKvThinkingRanges thinking;
    thinking.note(64, 192);
    thinking.arm();
    thinking.truncate(128);
    require(thinking.armed() && thinking.covers(64, 64));
    thinking.truncate(64);
    require(!thinking.armed() && !thinking.contains(64));
}

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

static void native_page_prefill_release() {
    kvmem::HistoricalKvPageGc::Config config;
    config.slots = 3;
    config.logical_pages = 3;
    config.page_tokens = 64;
    config.sink_tokens = 64;
    config.recent_tokens = 64;
    config.low_ready = 1;
    config.high_ready = 2;
    kvmem::HistoricalKvPageGc bridge(config);
    for (uint32_t page = 0; page < 3; ++page)
        require(bridge.register_page(page, 128, page + 1));
    bridge.protect_page(1, false);
    for (int i = 0; i < 8 && !bridge.stats().proposed; ++i) bridge.manual_tick();
    const auto & retired = bridge.commit_boundary();
    require(retired.size() == 1 && retired.front().page == 1);
    kvmem::HostKvStorage native_host(256);
    auto record = native_host.allocate(128);
    record.seal();
    auto reader = record.lease();
    bridge.acknowledge_host_release(retired.front(), std::move(record), true);
    require(native_host.charged_bytes() == 128); // Reader still owns the payload.
    reader = {};
    require(native_host.charged_bytes() == 0); // Prefill can allocate immediately.
    for (int i = 0; i < 8 && !bridge.stats().reclaimed; ++i) bridge.manual_tick();
    require(bridge.stats().reclaimed == 1);
}

static void native_page_redundancy_handoff() {
    kvmem::HistoricalKvPageGc::Config config;
    config.slots = 5;
    config.logical_pages = 5;
    config.page_tokens = 64;
    config.sink_tokens = 64;
    config.recent_tokens = 64;
    config.low_ready = 1;
    config.high_ready = 2;
    config.batch_blocks = 5;
    auto thinking = std::make_shared<kvmem::HistoricalKvThinkingRanges>();
    thinking->note(64, 256);
    thinking->arm();
    kvmem::HistoricalKvRedundancyPolicy::Config policy_config;
    policy_config.thinking = thinking;
    policy_config.pool_blocks = 5;
    policy_config.segment_blocks = 3;
    policy_config.lru_fallback_unclassified = true;
    auto source = std::make_shared<PageFeatures>();
    kvmem::HistoricalKvPageGc bridge(config,
        std::make_unique<kvmem::HistoricalKvRedundancyPolicy>(5, source, policy_config));
    for (uint32_t page = 0; page < 5; ++page)
        require(bridge.register_page(page, 1024, page + 1));
    for (uint32_t page = 1; page < 4; ++page) bridge.protect_page(page, false);
    for (int i = 0; i < 20 && bridge.stats().proposed < 2; ++i) bridge.manual_tick();
    require(bridge.stats().proposed == 2);
    const auto metrics = bridge.policy_metrics();
    require(metrics.comparisons >= 3);
    require(metrics.lossy_intents == 0);
    const auto& retired = bridge.commit_boundary();
    require(retired.size() == 2);
    require(bridge.is_live(0) && bridge.is_live(4));
    kvmem::HostKvStorage native_host(2048);
    for (const auto& page : retired) {
        require(page.page > 0 && page.page < 4);
        auto record = native_host.allocate(128);
        record.seal();
        bridge.acknowledge_host_release(page, std::move(record));
    }
    for (int i = 0; i < 20 && bridge.stats().reclaimed < 2; ++i) bridge.manual_tick();
    require(bridge.stats().reclaimed == 2);
    require(native_host.charged_bytes() == 0);
}

static void native_thinking_lru_fallback() {
    auto thinking = std::make_shared<kvmem::HistoricalKvThinkingRanges>();
    thinking->note(64, 128);
    thinking->arm();
    kvmem::HistoricalKvRedundancyPolicy::Config config;
    config.thinking = thinking;
    config.lru_fallback_unclassified = true;
    auto source = std::make_shared<PageFeatures>();
    kvmem::HistoricalKvRedundancyPolicy policy(4, source, config);
    kvmem::HistoricalKvSnapshot snapshot;
    snapshot.incarnation = 7;
    snapshot.round_id = 1;
    for (uint32_t id = 0; id < 4; ++id) {
        kvmem::HistoricalKvBlockMeta meta;
        meta.handle = {7, id, 1};
        meta.logical_id = id;
        meta.original_begin = uint64_t(id) * 64;
        meta.valid_tokens = 64;
        meta.content_version = 1;
        meta.life = kvmem::HistoricalKvLife::Live;
        meta.sealed = true;
        meta.mandatory = id < 2; // Sink and sole thinking block cannot be victims.
        meta.last_use_seq = id == 3 ? 1 : 10;
        snapshot.blocks.push_back(meta);
    }
    kvmem::HistoricalKvPressure pressure;
    pressure.ready = 0;
    pressure.low = 1;
    pressure.high = 2;
    kvmem::HistoricalKvDecisionBuffer out(1);
    const auto result = policy.step(snapshot, pressure, {4, 1}, out);
    require(result.status == kvmem::HistoricalKvStepStatus::Complete && result.emitted == 1);
    require(out.values().front().reason == kvmem::HistoricalKvDeleteReason::Lru);
    require(out.values().front().victim.slot == 3);
    require(policy.permits(snapshot.blocks[3]) && !policy.permits(snapshot.blocks[1]));
    snapshot.round_id = 2;
    snapshot.blocks[3].handle.incarnation = 6; // stale previous owner
    kvmem::HistoricalKvDecisionBuffer second(1);
    require(policy.step(snapshot, pressure, {4, 1}, second).emitted == 1);
    require(second.values().front().victim.slot == 2);
}

static void native_thinking_partial_lru_fallback() {
    auto thinking = std::make_shared<kvmem::HistoricalKvThinkingRanges>();
    thinking->note(64, 192);
    thinking->arm();
    kvmem::HistoricalKvRedundancyPolicy::Config config;
    config.thinking = thinking;
    config.lru_fallback_unclassified = true;
    auto source = std::make_shared<PageFeatures>();
    kvmem::HistoricalKvRedundancyPolicy policy(6, source, config);
    kvmem::HistoricalKvSnapshot snapshot;
    snapshot.incarnation = 8;
    snapshot.round_id = 1;
    for (uint32_t id = 0; id < 6; ++id) {
        kvmem::HistoricalKvBlockMeta meta;
        meta.handle = {8, id, 1};
        meta.logical_id = id;
        meta.original_begin = uint64_t(id) * 64;
        meta.valid_tokens = 64;
        meta.content_version = 1;
        meta.life = kvmem::HistoricalKvLife::Live;
        meta.sealed = true;
        meta.mandatory = id == 0 || id == 5;
        meta.last_use_seq = id == 4 ? 1 : 10;
        snapshot.blocks.push_back(meta);
    }
    kvmem::HistoricalKvPressure pressure;
    pressure.ready = 0;
    pressure.low = 1;
    pressure.high = 4;
    kvmem::HistoricalKvDecisionBuffer out(4);
    kvmem::HistoricalKvStepResult result;
    for (int i = 0; i < 20; ++i) {
        result = policy.step(snapshot, pressure, {6, 4}, out);
        if (result.status != kvmem::HistoricalKvStepStatus::Yielded) break;
    }
    require(result.status == kvmem::HistoricalKvStepStatus::Complete);
    require(result.emitted == 3 && out.values().size() == 3);
    size_t preferred = 0, fallback = 0;
    for (const auto & intent : out.values()) {
        require(policy.permits(snapshot.blocks[intent.victim.slot]));
        if (intent.victim.slot == 1 || intent.victim.slot == 2) ++preferred;
        if (intent.victim.slot == 3 || intent.victim.slot == 4) {
            require(intent.reason == kvmem::HistoricalKvDeleteReason::Lru);
            ++fallback;
        }
        require(intent.victim.slot != 0 && intent.victim.slot != 5);
    }
    require(preferred == 1 && fallback == 2);
}

static void native_unsealed_pages_drive_pressure() {
    kvmem::HostKvStorage storage(10 * 64);
    kvmem::HistoricalKvPageGc::Config config;
    config.slots = config.logical_pages = 10;
    config.low_ready = 2;
    config.high_ready = 4;
    config.sink_tokens = 64;
    config.batch_blocks = 4;
    config.physical_storage = storage;
    config.physical_page_bytes = 64;
    kvmem::HistoricalKvPageGc bridge(config);
    std::vector<kvmem::HostKvRecord> records;
    records.reserve(9);
    for (uint32_t page = 0; page < 6; ++page) {
        records.push_back(storage.allocate(64));
        records.back().seal();
        require(bridge.register_page(page, 64, page + 1));
        if (page) bridge.protect_page(page, false);
    }
    for (int i = 0; i < 8; ++i) bridge.manual_tick();
    require(bridge.stats().proposed == 0); // Four real free pages meet the high water.
    // These native append buffers occupy RAM before they have captured KV.
    // Repository accounting still reports four free slots, but RAM has only one.
    for (int i = 0; i < 3; ++i) records.push_back(storage.allocate(64));
    auto reader = records[1].lease();
    for (int i = 0; i < 12 && !bridge.stats().proposed; ++i) bridge.manual_tick();
    require(bridge.stats().proposed == 3);
    const auto& retired = bridge.commit_boundary();
    require(retired.size() == 3 && bridge.is_live(0));
    for (const auto& page : retired) {
        require(page.page > 0 && page.page < 6);
        bridge.acknowledge_host_release(page, std::move(records[page.page]));
    }
    for (int i = 0; i < 12; ++i) bridge.manual_tick();
    require(bridge.stats().reclaimed == 2 && storage.charged_bytes() == 7 * 64);
    require(bridge.stats().proposed == 3); // Pending reader release already covers the deficit.
    reader = {};
    for (int i = 0; i < 12 && bridge.stats().reclaimed < 3; ++i) bridge.manual_tick();
    require(bridge.stats().reclaimed == 3 && storage.charged_bytes() == 6 * 64);
    for (int i = 0; i < 12; ++i) bridge.manual_tick();
    require(bridge.stats().proposed == 3); // Actual headroom reached the high water.
    require(records[6] && records[7] && records[8] && !bridge.contains(6));
}

int main() {
    native_unsealed_pages_drive_pressure();
    native_page_feature_snapshot();
    truncated_thinking_reopens_candidates();
    native_page_handoff();
    native_page_prefill_release();
    native_page_redundancy_handoff();
    native_thinking_lru_fallback();
    native_thinking_partial_lru_fallback();
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
