#include "kvmem/historical_kv_gc.hpp"
#include "kvmem/historical_kv_page_gc.hpp"
#include "kvmem/historical_kv_page_features.hpp"
#include "kvmem/historical_kv_redundancy.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
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

static void proactive_step_evidence_lifecycle() {
    using namespace kvmem;
    // Semantic eligibility alone may request a fenced native handoff. Actual
    // retirement additionally needs exact current source and result bindings.
    HistoricalKvStepEvidence::Config artifact_limits;
    artifact_limits.min_observations=1; artifact_limits.min_cold_observations=0;
    artifact_limits.max_importance=1;
    HistoricalKvStepEvidence unbound(32,64,artifact_limits);
    HistoricalKvStepEvidence::Update artifact{"file","tool_detail",65,320,384,512,1,1,false,true};
    unbound.publish({artifact},640);
    require(unbound.handoff_candidate(2) && !unbound.eligible(2));
    std::vector<uint8_t> flags(10,1);
    unbound.mark_result_pages(flags,128);
    for(uint32_t i=0;i<flags.size();++i) require(flags[i]==(i==6 || i==7 ? 129 : 1));
    std::vector<uint8_t> partial_flags(7,2);
    unbound.mark_result_pages(partial_flags,128);
    require(partial_flags[5]==2 && partial_flags[6]==130);
    unbound.bind(2,{1,2,1}); unbound.bind(6,{1,6,1});
    require(!unbound.eligible(2)); // Incomplete retained result cannot witness deletion.
    require(unbound.permits_selection(2));
    unbound.bind(7,{1,7,1});
    require(unbound.eligible(2));
    // Selection observes the completed mark before any worker tick/retirement.
    require(!unbound.permits_selection(2) && unbound.permits_selection(2,true));
    HistoricalKvBlockMeta old;
    old.logical_id=2; old.handle={1,2,1};
    require(unbound.revision(old)!=0);
    unbound.bind(2,{1,2,2});
    require(unbound.revision(old)==0); // Same page ID is not the same KV generation.
    unbound.retired(2,{1,2,2});
    require(!unbound.permits_selection(2,true)); // Mandatory cannot revive retired KV.
    auto evidence = std::make_shared<HistoricalKvStepEvidence>(32,64,HistoricalKvStepEvidence::Config{});
    HostKvStorage storage(32*64);
    HistoricalKvPageGc::Config config;
    config.slots = config.logical_pages = 32;
    config.low_ready = 2; config.high_ready = 4;
    config.sink_tokens = config.recent_tokens = 64;
    config.evidence = evidence; config.physical_storage = storage; config.physical_page_bytes = 64;
    HistoricalKvPageGc gc(config,std::make_unique<HistoricalKvStatisticalPolicy>(evidence));
    std::vector<HostKvRecord> records(10);
    for(uint32_t i=0;i<10;++i) {
        records[i]=storage.allocate(64); records[i].seal();
        require(gc.register_page(i,64,i+1)); gc.protect_page(i,i==0 || i==9);
    }
    auto reader=records[2].lease();
    for(int i=0;i<16;++i)gc.manual_tick();
    require(gc.stats().proposed==0); // No evidence never means cold.
    HistoricalKvStepEvidence::Update u{"step-1","thinking",65,320,384,512,0,.01,true,false};
    for(uint64_t i=0;i<3;++i) {u.observation_id=i;evidence->publish({u},640);}
    for(int i=0;i<16;++i)gc.manual_tick();
    require(gc.stats().proposed==0); // Cold observations without completion do not retire.
    u.observation_id=3;u.complete=true;evidence->publish({u},640);
    require(!evidence->eligible(1) && evidence->eligible(2) && evidence->result_page(6));
    for(int i=0;i<32 && !gc.stats().proposed;++i)gc.manual_tick();
    require(gc.stats().proposed==3); // 22 free > high=4: genuinely proactive.
    // A later eligible revision still invalidates the old queued decisions.
    u.observation_id=4;evidence->publish({u},640);
    require(gc.commit_boundary().empty());
    auto duplicate=u;duplicate.importance=.5;
    bool rejected=false;
    try {evidence->publish({duplicate},640);}catch(const std::invalid_argument&){rejected=true;}
    require(rejected);
    for(int i=0;i<64 && !gc.stats().committed;++i) {
        gc.manual_tick();
        for(const auto& page:gc.commit_boundary())gc.acknowledge_host_release(page,std::move(records[page.page]),true);
    }
    require(gc.stats().committed==3);
    for(int i=0;i<32;++i)gc.manual_tick();
    require(gc.stats().reclaimed==2); // A real reader delays exactly one payload.
    reader={};for(int i=0;i<32;++i)gc.manual_tick();
    require(gc.stats().reclaimed==3 && gc.is_live(6) && gc.is_live(7));
    require(gc.is_live(1) && gc.is_live(5)); // Mixed boundary and unclassified content survive.
    require(evidence->stats().tracked==0 && evidence->stats().completed==1);
    require(!evidence->result_page(6)); // Completed evidence no longer pins a witness.
    flags.assign(10,1); evidence->mark_result_pages(flags,128);
    for(const auto flag:flags) require(flag==1);
    rejected=false;
    auto bad=u;bad.observation_id=5;bad.source_end=448;
    try {evidence->publish({bad},640);}catch(const std::invalid_argument&){rejected=true;}
    require(rejected);
    evidence->publish({u},640); // Replayed completed metadata is a no-op.
    require(evidence->stats().tracked==0 && !evidence->eligible(2));
    evidence->truncate_to(400); // Removed/partial result invalidates all associated eligibility.
    require(!evidence->eligible(2) && !evidence->result_page(6));
}

static void proactive_long_step_chain() {
    using namespace kvmem;
    HistoricalKvStepEvidence::Config limits;
    limits.min_observations=1; limits.min_cold_observations=0; limits.max_importance=1;
    auto evidence=std::make_shared<HistoricalKvStepEvidence>(256,64,limits);
    HostKvStorage storage(8*64);
    HistoricalKvPageGc::Config config;
    config.slots=8; config.logical_pages=256; config.low_ready=1; config.high_ready=2;
    config.evidence=evidence; config.physical_storage=storage; config.physical_page_bytes=64;
    HistoricalKvPageGc gc(config,std::make_unique<HistoricalKvStatisticalPolicy>(evidence));
    std::vector<HostKvRecord> records(121);
    records[0]=storage.allocate(64);records[0].seal();
    require(gc.register_page(0,64,1));gc.protect_page(0,false);
    for(uint32_t step=0;step<120;++step) {
        const auto witness=step+1;
        records[witness]=storage.allocate(64);records[witness].seal();
        require(gc.register_page(witness,64,witness+1));gc.protect_page(witness,false);
        HistoricalKvStepEvidence::Update u{"step-"+std::to_string(step),"completed_step",
            step*64,(step+1)*64,witness*64,(witness+1)*64,1,1,false,true};
        evidence->publish({u},(witness+1)*64);
        require(evidence->selection_excluded(step));
        require(!evidence->selection_excluded(witness));
        for(int tick=0;tick<128 && gc.stats().reclaimed<step+1;++tick) {
            gc.manual_tick();
            for(const auto& page:gc.commit_boundary())
                gc.acknowledge_host_release(page,std::move(records[page.page]),true);
        }
        require(gc.stats().committed==step+1 && gc.stats().reclaimed==step+1);
        require(evidence->stats().tracked==0 && evidence->stats().completed==step+1);
        require(gc.is_live(witness) && !evidence->result_page(witness));
    }
    HistoricalKvStepEvidence::Update old{"step-0","completed_step",0,64,64,128,1,1,false,true};
    evidence->publish({old},121*64);
    require(evidence->stats().tracked==0 && evidence->selection_excluded(0));
    require(evidence->stats().ignored_retired_updates==1);
    // A genuine rewind starts a new suffix; old tombstones cannot poison it.
    evidence->truncate_to(0);
    require(!evidence->selection_excluded(0));
    evidence->bind(0,{2,0,1});
}

static void proactive_bounded_evidence_admission() {
    using namespace kvmem;
    HistoricalKvStepEvidence::Config limits;
    limits.min_observations=1;limits.min_cold_observations=0;limits.max_importance=1;limits.max_steps=1;
    auto evidence=std::make_shared<HistoricalKvStepEvidence>(16,64,limits);
    for(uint32_t id=0;id<8;++id)evidence->bind(id,{1,id,1});
    HistoricalKvStepEvidence::Update first{"first","thinking",0,64,64,128,1,1,false,true};
    evidence->publish({first},512);
    HistoricalKvStatisticalPolicy policy(evidence);
    HistoricalKvBlockMeta block;
    block.logical_id=0;block.handle={1,0,1};block.life=HistoricalKvLife::Live;block.sealed=true;
    HistoricalKvSnapshot snapshot; snapshot.incarnation=1;snapshot.round_id=1;snapshot.blocks.push_back(block);
    HistoricalKvDecisionBuffer decisions(8);
    policy.step(snapshot,{},HistoricalKvWorkBudget{},decisions);
    require(decisions.values().size()==1);
    HistoricalKvStepEvidence::Update second{"second","thinking",128,192,192,256,1,1,false,true};
    evidence->publish({second},512); // Full metadata is advisory; inference need not fail.
    require(evidence->stats().tracked==1 && evidence->stats().evicted_evidence==1);
    require(!policy.permits_intent(block,decisions.values()[0]));
    require(!evidence->selection_excluded(0) && evidence->selection_excluded(2));
    auto invalid=first;invalid.id="invalid";invalid.importance=2;
    bool rejected=false;
    try {evidence->publish({first,invalid},512);}catch(const std::invalid_argument&){rejected=true;}
    require(rejected && evidence->stats().evicted_evidence==1 && evidence->selection_excluded(2));
    limits.source_floor_tokens=64;
    HistoricalKvStepEvidence sink(16,64,limits);
    sink.publish({first},512);
    require(sink.stats().tracked==0 && !sink.handoff_candidate(0));
}

static void proactive_real_background_worker() {
    using namespace kvmem;
    // Unlike the deterministic lifecycle tests, the request owner never calls
    // manual_tick here. Only the real background worker scans and reclaims.
    HistoricalKvStepEvidence::Config limits;
    limits.min_observations=1; limits.min_cold_observations=0; limits.max_importance=1;
    auto evidence=std::make_shared<HistoricalKvStepEvidence>(128,64,limits);
    HostKvStorage storage(8*64);
    HistoricalKvPageGc::Config config;
    config.slots=8; config.logical_pages=128; config.low_ready=1; config.high_ready=2;
    config.poll_ms=2; config.evidence=evidence;
    config.physical_storage=storage; config.physical_page_bytes=64;
    HistoricalKvPageGc gc(config,std::make_unique<HistoricalKvStatisticalPolicy>(evidence));
    std::vector<HostKvRecord> records(49);
    records[0]=storage.allocate(64); records[0].seal();
    require(gc.register_page(0,64,1)); gc.protect_page(0,false);
    gc.start();
    const auto deadline_after=[] {return std::chrono::steady_clock::now()+std::chrono::seconds(3);};
    for(uint32_t step=0;step<48;++step) {
        const auto witness=step+1;
        records[witness]=storage.allocate(64); records[witness].seal();
        require(gc.register_page(witness,64,witness+1)); gc.protect_page(witness,false);
        auto reader=step%3==0 ? records[step].lease() : HostKvLease{};
        HistoricalKvStepEvidence::Update u{"async-"+std::to_string(step),"completed_step",
            step*64,(step+1)*64,witness*64,(witness+1)*64,1,1,false,true};
        evidence->publish({u},(witness+1)*64);
        require(!evidence->permits_selection(step));
        bool committed=false;
        const auto until_commit=deadline_after();
        while(!committed && std::chrono::steady_clock::now()<until_commit) {
            for(const auto& page:gc.commit_boundary()) {
                require(page.page==step && page.page!=witness);
                require(!evidence->permits_selection(step,true));
                // Move ownership only; payload destruction stays on worker.
                gc.acknowledge_host_release(page,std::move(records[page.page]));
                committed=true;
            }
            if(!committed) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        require(committed && gc.is_live(witness));
        require(evidence->stats().tracked==0 && evidence->stats().completed==step+1);
        // Logical retirement is already effective while a real Host reader
        // still retains the bytes. Releasing it enables asynchronous reuse.
        if(step%3==0) require(storage.charged_bytes()==128);
        reader={};
        const auto until_reclaim=deadline_after();
        while(gc.stats().reclaimed<step+1 && std::chrono::steady_clock::now()<until_reclaim)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        require(gc.stats().reclaimed==step+1 && storage.charged_bytes()==64);
    }
    const auto stats=gc.stats();
    require(stats.worker_failures==0 && stats.policy_failures==0 && stats.boundary_committed==48);
}

static void attainable_watermark_hysteresis() {
    using Gc = kvmem::HistoricalKvPageGc;
    // 64K Host, 8K selected + 16K reserve, 512-token append, 64 tokens/page.
    // Sink alone needs 16 pages: the former high=1022 was unreachable.
    const auto marks = Gc::derive_watermarks(1024, 384, 8);
    require(marks.low_ready == 42 && marks.high_ready == 64);
    require(marks.high_ready + 384 <= 1024);
    const auto large = Gc::derive_watermarks(4096, 1792, 8);
    require(large.low_ready == 128 && large.high_ready == 256);
    const auto tight = Gc::derive_watermarks(64, 50, 8);
    require(tight.low_ready < tight.high_ready && tight.high_ready + 50 <= 64);
    // Reject a layout that cannot retain its view plus an aligned append.
    bool rejected = false;
    try { (void)Gc::derive_watermarks(64, 55, 8); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected);
    rejected = false;
    try { (void)Gc::derive_watermarks(UINT32_MAX, 0, UINT32_MAX); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected);

    // Physical allocation, not just sealed repository slots, drives pressure.
    kvmem::HostKvStorage storage(1024 * 64);
    Gc::Config config;
    config.slots = config.logical_pages = 1024;
    config.sink_tokens = 1024;
    config.recent_tokens = 64;
    config.low_ready = marks.low_ready;
    config.high_ready = marks.high_ready;
    config.physical_storage = storage;
    config.physical_page_bytes = 64;
    Gc bridge(config);
    std::vector<kvmem::HostKvRecord> records(981);
    for (uint32_t id = 0; id < records.size(); ++id) {
        records[id] = storage.allocate(64);
        records[id].seal();
        require(bridge.register_page(id, 64, id + 1));
        bridge.protect_page(id, id < 384);
    }
    for (int i = 0; i < 40; ++i) bridge.manual_tick();
    require(bridge.stats().proposed == 0); // 43 free >= low: no cleanup.
    auto append = storage.allocate(5 * 64); // Unsealed lookahead leaves 38 free.
    for (int i = 0; i < 400; ++i) {
        bridge.manual_tick();
        for (const auto& page : bridge.commit_boundary()) {
            require(page.page >= 384);
            bridge.acknowledge_host_release(page, std::move(records[page.page]), true);
        }
        if (bridge.stats().reclaimed == 26) break;
    }
    require(bridge.stats().committed == 26 && bridge.stats().reclaimed == 26);
    require((storage.capacity_bytes() - storage.charged_bytes()) / 64 == 64);
    for (int i = 0; i < 100; ++i) {
        bridge.manual_tick();
        require(bridge.commit_boundary().empty());
    }
    require(bridge.stats().proposed == 26); // High reached: no more deletions.
    for (uint32_t id = 0; id < 384; ++id) require(bridge.is_live(id));
}

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
    proactive_step_evidence_lifecycle();
    proactive_long_step_chain();
    proactive_bounded_evidence_admission();
    proactive_real_background_worker();
    attainable_watermark_hysteresis();
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
