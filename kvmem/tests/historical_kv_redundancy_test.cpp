#include "kvmem/historical_kv_redundancy.hpp"
#include "kvmem/historical_kv_gc.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace kvmem;
static void check(bool ok, const char * message) { if (!ok) throw std::runtime_error(message); }

struct Features : HistoricalKvRedundancySource {
    bool available = true;
    size_t missing = SIZE_MAX;
    uint64_t epoch = 0;
    std::array<std::array<float, 3>, 6> keys{{{{0,1,0}}, {{1,0,0}}, {{1,0,0}}, {{1,0,0}}, {{1,0,0}}, {{0,0,1}}}};
    std::array<double, 6> logits{{3,0,0,0,0,-1}};
    size_t dimensions() const noexcept override { return 3; }
    bool begin_round(uint64_t) override { return available; }
    uint64_t question_epoch() const noexcept override { return epoch; }
    bool read(const HistoricalKvBlockMeta & m, float * out, double & logit) override {
        if (m.logical_id == missing) return false;
        std::copy(keys.at(m.logical_id).begin(), keys.at(m.logical_id).end(), out);
        logit = logits.at(m.logical_id); return true;
    }
};
struct CountedFeatures : HistoricalKvRedundancySource {
    std::vector<uint64_t> reads;
    bool distinct = false;
    size_t dimensions() const noexcept override { return 3; }
    bool begin_round(uint64_t) override { return true; }
    bool read(const HistoricalKvBlockMeta & m, float * out, double & logit) override {
        reads.push_back(m.logical_id);
        if (distinct) {
            const float angle = float(m.logical_id) * 0.5f;
            out[0] = std::cos(angle); out[1] = std::sin(angle); out[2] = 0;
        } else {
            out[0] = 1; out[1] = 0; out[2] = 0;
        }
        logit = 0;
        return true;
    }
};
struct MappedFeatures : HistoricalKvRedundancySource {
    std::vector<std::array<float, 3>> keys;
    std::vector<double> logits;
    std::vector<uint64_t> reads;
    size_t dimensions() const noexcept override { return 3; }
    bool begin_round(uint64_t) override { return true; }
    bool read(const HistoricalKvBlockMeta & m, float * out, double & logit) override {
        reads.push_back(m.logical_id);
        const auto & key = keys.at(m.logical_id);
        std::copy(key.begin(), key.end(), out);
        logit = logits.at(m.logical_id);
        return true;
    }
};
static HistoricalKvSnapshot fixture() {
    HistoricalKvSnapshot s; s.incarnation = 7; s.round_id = 1;
    for (unsigned i = 0; i < 6; ++i) {
        HistoricalKvBlockMeta m; m.handle = {7, i, 1}; m.logical_id = i;
        m.original_begin = i * 32; m.valid_tokens = 32; m.content_version = 9;
        m.last_use_seq = i; m.life = HistoricalKvLife::Live; m.sealed = true;
        s.blocks.push_back(m);
    }
    return s;
}
static void run(HistoricalKvPolicy & p, HistoricalKvSnapshot & s, HistoricalKvDecisionBuffer & out,
                HistoricalKvPressure pressure = {0, 1, 2, 0}, size_t scan = 1) {
    for (unsigned i = 0; i < 100; ++i)
        if (p.step(s, pressure, {scan, 6}, out).status != HistoricalKvStepStatus::Yielded) return;
    throw std::runtime_error("policy failed to finish bounded work");
}
static bool selected(const HistoricalKvDecisionBuffer & out, size_t slot) {
    for (const auto & i : out.values()) if (i.victim.slot == slot) return true;
    return false;
}

static void policy_tests() {
    auto f = std::make_shared<Features>(); auto s = fixture();
    HistoricalKvRedundancyPolicy p(6, f, {}); HistoricalKvDecisionBuffer out(6);
    run(p, s, out);
    check(out.values().size() == 2 && selected(out, 1) && selected(out, 2), "redundant blocks not selected");
    check(!selected(out, 0) && !selected(out, 4), "unique fact or latest representative deleted");
    for (const auto & i : out.values()) check(i.expected_content_version == 9 && i.reason == HistoricalKvDeleteReason::Redundancy,
        "missing generation/content condition");
    check(p.metrics().comparisons == 30 && p.metrics().representatives == 1, "comparison or representative accounting");
    HistoricalKvLruPolicy lru(6); HistoricalKvDecisionBuffer old(6); run(lru, s, old);
    check(selected(old, 0), "controlled fixture does not distinguish LRU");
    std::cout << "EFFECT controlled fixture: LRU deletes unique old fact; redundancy removes 2 duplicates and retains its latest representative\n";

    out.clear_after_ack(); s.round_id++;
    check(p.step(s, {0,1,2,0}, {1,6}, out).status == HistoricalKvStepStatus::Yielded, "did not yield feature work");
    run(p, s, out, {2,1,2,0}); check(out.values().empty(), "obsolete pressure still caused deletion");
    s.round_id++; run(p, s, out, {0,1,2,2}); check(out.values().empty(), "pending credits ignored");
    f->available = false; s.round_id++; run(p, s, out); check(out.values().empty() && p.metrics().no_query_rounds == 1, "missing Query caused deletion");
    f->available = true; f->missing = 1; s.round_id++; s.blocks[2].mandatory = true;
    run(p, s, out); check(!selected(out,1) && !selected(out,2), "missing/mandatory candidate deleted");
    check(p.metrics().missing_features > 0, "missing feature not visible");

    f->keys[3][0] = std::numeric_limits<float>::quiet_NaN();
    out.clear_after_ack(); s.round_id++; run(p, s, out);
    check(!selected(out,3), "nonfinite descriptor was used");
    f->keys[3] = {{0,0,0}};
    out.clear_after_ack(); s.round_id++; run(p, s, out); check(!selected(out,3), "zero descriptor was used");
    // Importance can distinguish the members of a redundant cluster.
    f = std::make_shared<Features>(); f->logits[1] = 20;
    HistoricalKvRedundancyPolicy important(6, f, {0.9, 0.9, 32}); s = fixture(); out.clear_after_ack();
    run(important, s, out); check(!selected(out,1), "importance has no effect on retention");
    // Force bounded recent references, not a full quadratic matrix.
    HistoricalKvRedundancyPolicy bounded(6, f, {0.1, 0.9, 2}); out.clear_after_ack();
    run(bounded, s, out); check(bounded.metrics().comparisons == 10, "reference limit not honored");
}

static void candidate_budget_tests() {
    // Age-banded references, not newest-only, find an old/old duplicate.
    auto s = fixture();
    auto f = std::make_shared<Features>();
    f->keys = {{{{1,0,0}}, {{1,0,0}}, {{0,1,0}}, {{0,0,1}}, {{0,-1,0}}, {{0,0,-1}}}};
    HistoricalKvRedundancyPolicy recent(6, f, {0.1, 0.9, 2});
    HistoricalKvDecisionBuffer recent_out(6);
    run(recent, s, recent_out);
    check(recent.metrics().representatives == 1 && selected(recent_out, 1) && !selected(recent_out, 0),
          "age-banded references missed the old duplicate");
    HistoricalKvRedundancyPolicy full(6, f, {0.1, 0.9, 2, 6});
    HistoricalKvDecisionBuffer full_out(6);
    run(full, s, full_out);
    check(full.metrics().comparisons == 30 && full.metrics().representatives == 1,
          "bounded full comparison missed old duplicate");
    check(selected(full_out, 1) && !selected(full_out, 0),
          "more important duplicate was retired instead of the substitute");

    // A fixed 64-block pool reads at most 64 feature rows and performs exactly
    // 64*63 comparisons, regardless of how many live blocks are in the CPU pool.
    HistoricalKvSnapshot large; large.incarnation = 7; large.round_id = 1;
    for (unsigned i = 0; i < 512; ++i) {
        HistoricalKvBlockMeta m;
        m.handle = {7, i, 1}; m.logical_id = i; m.original_begin = i * 32;
        m.valid_tokens = 32; m.content_version = 9;
        m.life = HistoricalKvLife::Live; m.sealed = true;
        large.blocks.push_back(m);
    }
    large.blocks[0].mandatory = true;
    auto counted = std::make_shared<CountedFeatures>();
    HistoricalKvRedundancyPolicy capped(512, counted, {0.1, 0.9, 32, 64});
    HistoricalKvDecisionBuffer out(64);
    run(capped, large, out, {0, 1, 2, 0}, 16);
    check(counted->reads.size() == 64 && capped.metrics().comparisons == 64 * 63,
          "candidate budget did not bound feature or pair work");
    check(std::find(counted->reads.begin(), counted->reads.end(), 0) != counted->reads.end(),
          "mandatory block missing from substitute clustering");
    const auto first = counted->reads;
    counted->reads.clear(); out.clear_after_ack(); ++large.round_id;
    run(capped, large, out, {0, 1, 2, 0}, 16);
    check(counted->reads.size() == 64 && counted->reads != first,
          "age bands did not rotate excluded candidates across rounds");
    check(capped.metrics().comparisons == 2 * 64 * 63,
          "second bounded round exceeded pair budget");

    // Protected references are part of N even when thinking capture is off.
    // A large mandatory prefix must not turn the opt-in cap into an unbounded
    // all-pairs comparison or leave no room for an evictable candidate.
    auto protected_prefix = large;
    for (unsigned i = 0; i < 128; ++i) protected_prefix.blocks[i].mandatory = true;
    auto protected_features = std::make_shared<CountedFeatures>();
    HistoricalKvRedundancyPolicy protected_capped(512, protected_features, {0.1, 0.9, 32, 64});
    HistoricalKvDecisionBuffer protected_out(64);
    run(protected_capped, protected_prefix, protected_out, {0, 1, 2, 0}, 16);
    check(protected_features->reads.size() == 64 &&
          protected_capped.metrics().max_round_features == 64 &&
          protected_capped.metrics().comparisons == 64 * 63,
          "mandatory references defeated the non-thinking candidate cap");
    check(!protected_out.values().empty(), "mandatory references crowded out all eviction candidates");
    for (const auto & intent : protected_out.values())
        check(intent.victim.slot >= 128, "bounded pool proposed a mandatory block for retirement");

    // Restricting a round must not prevent the manager from satisfying a
    // larger Ready-capacity demand over successive rounds.
    size_t ready = 0;
    for (unsigned round = 0; round < 100 && ready < 100; ++round) {
        out.clear_after_ack(); ++large.round_id;
        run(capped, large, out, {uint32_t(ready), 99, 100, 0}, 16);
        check(!out.values().empty(), "bounded candidate pool stalled capacity recovery");
        for (const auto & intent : out.values()) {
            check(intent.victim.slot != 0, "mandatory block selected in bounded recovery");
            large.blocks[intent.victim.slot].life = HistoricalKvLife::Retired;
            ++ready;
        }
    }
    check(ready == 100, "bounded candidate pool failed to meet Ready target");
}

static void source_and_gc_test() {
    HistoricalKvRawStoreConfig c; c.n_layer = 2; c.max_tokens_per_slot = 1;
    c.packed_k_bytes_per_token = c.packed_v_bytes_per_token = 1;
    c.mean_k_elements = 2; c.attention_layers = {0,1};
    auto repo = std::make_shared<HistoricalKvRepository>(4);
    auto store = std::make_shared<HistoricalKvRawStore>(c, 4);
    auto source = std::make_shared<HistoricalKvMeanKeySource>(repo, store, 2, 1, 2);
    std::vector<HistoricalKvHandle> handles;
    for (unsigned i = 0; i < 4; ++i) {
        auto writer = repo->begin_write(i, i, 1, 1, 20, [&, store](const HistoricalKvHandle & h) {
            const auto r = store->release(*repo, h);
            return r.code == HistoricalKvRawStore::ReleaseCode::Released;
        });
        check(bool(writer), "writer missing"); auto h = writer->handle(); handles.push_back(h);
        check(store->reserve(*writer,h,i,1), "reserve failed");
        const std::vector<float> k = i == 0 ? std::vector<float>{0,1} : std::vector<float>{1,0};
        for (unsigned l = 0; l < 2; ++l) check(store->write_layer(*writer,h,l,{1},{2},k,1), "write failed");
        check(store->publish_complete(*repo,*writer,h), "publish failed");
    }
    check(!source->begin_round(repo->incarnation()), "missing Query treated as valid");
    float q[] = {0,3,0,4}; source->publish_query(0,10,q,1);
    check(!source->begin_round(repo->incarnation()), "partial layer Query published");
    source->publish_query(1,10,q,1);
    check(source->finish_question(), "complete input question rejected");
    check(source->begin_round(repo->incarnation()), "complete Query missing");
    std::vector<float> descriptor(source->dimensions()); double logit = 0;
    check(source->read(*repo->inspect(handles[0]),descriptor.data(),logit) &&
          std::abs(logit - 4/std::sqrt(2.0)) < 1e-6, "GQA max/Mean-K importance incorrect");
    double norm = 0; for (auto x : descriptor) norm += x*x;
    check(std::abs(norm-1) < 1e-6, "descriptor not normalized");
    // A producer publishes opposite per-layer Queries. A coherent frame has
    // exactly zero logit; mixing two different frame numbers does not.
    float zero[] = {0,0,0,0};
    source->begin_question();
    source->publish_query(0,20,zero,1); source->publish_query(1,20,zero,1);
    check(source->finish_question(), "zero question rejected");
    std::atomic<bool> finished{false};
    std::thread producer([&] {
        for (unsigned i=1;i<=1000;++i) {
            float a[] = {0,float(i),0,float(i)}, b[] = {0,-float(i),0,-float(i)};
            source->publish_observation(0,20+i,a,1); source->publish_observation(1,20+i,b,1);
        }
        finished = true;
    });
    bool coherent = true;
    while (!finished.load()) if (source->begin_round(repo->incarnation())) {
        coherent &= source->read(*repo->inspect(handles[0]),descriptor.data(),logit) && std::abs(logit) < 1e-8;
    }
    producer.join(); check(coherent, "mixed Query layers across concurrent publication");
    source->begin_question();
    source->publish_query(0,2000,q,1); source->publish_query(1,2000,q,1);
    check(source->finish_question(), "replacement question rejected");
    HistoricalKvGc::Config cfg; cfg.mode = HistoricalKvGc::Mode::Delete;
    cfg.ready_low = 1; cfg.ready_high = 2; cfg.scan_blocks = cfg.reclaim_blocks = 4; cfg.batch_blocks = 2;
    HistoricalKvGc gc(repo, std::make_unique<HistoricalKvRedundancyPolicy>(4,source,HistoricalKvRedundancyPolicy::Config{}), cfg);
    const auto bytes = store->allocated_payload_bytes();
    for (int i = 0; i < 10; ++i) gc.manual_tick();
    check(gc.stats().committed == 2 && gc.stats().reclaimed == 2 && store->allocated_payload_bytes() < bytes,
          "redundancy intents did not release real CPU payload");
    check(repo->completion_state(handles[0]) == HistoricalKvRepository::CompletionState::Live &&
          repo->completion_state(handles[3]) == HistoricalKvRepository::CompletionState::Live,
          "lost unique block or representative in real repository");
    // Reusing a slot must not reuse the old cached Mean-K.
    auto writer = repo->begin_write(8,8,1,2,20,[&,store](const HistoricalKvHandle & h){
        return store->release(*repo,h).code == HistoricalKvRawStore::ReleaseCode::Released;
    });
    check(bool(writer), "reclaimed slot not reusable"); const auto h = writer->handle();
    check(store->reserve(*writer,h,8,1), "reuse reserve failed");
    for(unsigned l=0;l<2;++l) check(store->write_layer(*writer,h,l,{3},{4},{0,-1},1), "reuse write failed");
    check(store->publish_complete(*repo,*writer,h), "reuse publish failed"); writer->reset();
    check(source->read(*repo->inspect(h),descriptor.data(),logit) && logit < 0, "old generation feature reused");
    auto background = repo->acquire_background_mean_k(h);
    check(background && !repo->outstanding_operations(), "GC read counted as unfinished request work");
    const auto retained_bytes = store->allocated_payload_bytes();
    repo->retire(h); repo->try_reclaim(h);
    check(store->allocated_payload_bytes() == retained_bytes, "background reader lost its payload");
    check(!repo->acquire_background_mean_k(h), "retired block accepted a new background reader");
    background.reset(); repo->try_reclaim(h);
    check(store->allocated_payload_bytes() < retained_bytes, "background read prevented eventual release");
    repo->close(); for (auto m : repo->snapshot()) { repo->retire(m.handle); repo->try_reclaim(m.handle); }
    check(store->allocated_payload_bytes() == 0, "fixture did not drain");
}

static void manager_ack_preserves_cross_round_witness() {
    auto repo = std::make_shared<HistoricalKvRepository>(3);
    std::array<HistoricalKvHandle, 3> handles;
    for (unsigned i = 0; i < 3; ++i) {
        auto writer = repo->begin_write(i, i * 32, 32, 1, 1024,
            [](const HistoricalKvHandle &) { return true; });
        check(writer && repo->publish_write(*writer), "manager chain publication failed");
        handles[i] = writer->handle();
    }
    auto source = std::make_shared<MappedFeatures>();
    constexpr double pi = 3.14159265358979323846;
    for (unsigned i = 0; i < 3; ++i) {
        const double radians = double(i) * 20.0 * pi / 180.0;
        source->keys.push_back({float(std::cos(radians)), float(std::sin(radians)), 0});
        source->logits.push_back(i == 1 ? 10.0 : (i == 0 ? -10.0 : 0.0));
    }
    auto thinking = std::make_shared<HistoricalKvThinkingRanges>();
    thinking->arm();
    thinking->note(0, 96);
    HistoricalKvRedundancyPolicy::Config policy_cfg;
    policy_cfg.thinking = thinking;
    policy_cfg.pool_blocks = 3;
    policy_cfg.segment_blocks = 1;
    HistoricalKvGc::Config gc_cfg;
    gc_cfg.mode = HistoricalKvGc::Mode::Delete;
    gc_cfg.ready_low = 1;
    gc_cfg.ready_high = 2;
    gc_cfg.scan_blocks = gc_cfg.reclaim_blocks = 3;
    gc_cfg.batch_blocks = 1;
    gc_cfg.retire_at_decode_boundary = true;
    HistoricalKvGc gc(repo,
        std::make_unique<HistoricalKvRedundancyPolicy>(3, source, policy_cfg), gc_cfg);
    HistoricalKvGc::BoundaryResult first;
    for (unsigned tick = 0; tick < 20 && !first.committed; ++tick) {
        gc.manual_tick();
        first = gc.commit_at_decode_boundary();
    }
    check(first.committed == 1 &&
          repo->completion_state(handles[0]) == HistoricalKvRepository::CompletionState::Retired,
          "manager did not commit A at a decode boundary");
    source->logits[2] = 20.0;
    HistoricalKvGc::BoundaryResult second;
    for (unsigned tick = 0; tick < 20 && !second.committed; ++tick) {
        gc.manual_tick();
        second = gc.commit_at_decode_boundary();
    }
    check(second.committed == 1 && gc.policy_metrics().lossy_intents == 1 &&
          repo->completion_state(handles[1]) == HistoricalKvRepository::CompletionState::Live &&
          repo->completion_state(handles[2]) == HistoricalKvRepository::CompletionState::Retired,
          "manager fallback removed B despite its cross-round substitute obligation");
    gc.manual_tick();
    check(gc.stats().committed == 2, "manager failed to acknowledge both boundary retirements");
}
static void thinking_pool_tests() {
    auto ranges = std::make_shared<HistoricalKvThinkingRanges>();
    ranges->arm();
    ranges->note(1, 4);
    ranges->clear_spans();
    check(ranges->armed() && !ranges->contains(1), "reset dropped the armed flag or kept a span");
    auto s = fixture();
    auto quiet = std::make_shared<CountedFeatures>();
    HistoricalKvRedundancyPolicy::Config cfg;
    cfg.thinking = ranges;
    cfg.segment_blocks = 2;
    cfg.pool_blocks = 4;
    HistoricalKvRedundancyPolicy armed_empty(6, quiet, cfg);
    HistoricalKvDecisionBuffer none(6);
    run(armed_empty, s, none, {0, 1, 2, 1});
    check(none.values().empty() && quiet->reads.empty(), "armed policy deleted before a thinking span existed");

    ranges->note(32, 96);
    auto recorded = std::make_shared<CountedFeatures>();
    HistoricalKvRedundancyPolicy thinking(6, recorded, cfg);
    HistoricalKvDecisionBuffer out(6);
    run(thinking, s, out, {0, 1, 2, 1});
    check(recorded->reads == std::vector<uint64_t>({1, 2}), "prompt block entered the thinking matrix");
    check(out.values().size() == 1 && selected(out, 1) && !selected(out, 2),
          "older thinking duplicate was not the only retirement");
    check(!selected(out, 0) && !selected(out, 3), "non-thinking block was retired");

    auto crossed = fixture();
    crossed.round_id = 3;
    crossed.blocks[1].original_begin = 16;
    crossed.blocks[2].original_begin = 80;
    auto crossed_reads = std::make_shared<CountedFeatures>();
    HistoricalKvRedundancyPolicy crossed_pool(6, crossed_reads, cfg);
    HistoricalKvDecisionBuffer crossed_out(6);
    run(crossed_pool, crossed, crossed_out, {0, 1, 2, 1});
    check(crossed_reads->reads.empty() && crossed_out.values().empty(),
          "a block that only overlaps a thinking span was scored");

    s.blocks[2].mandatory = true;
    auto protected_reads = std::make_shared<CountedFeatures>();
    HistoricalKvRedundancyPolicy protected_pool(6, protected_reads, cfg);
    HistoricalKvDecisionBuffer kept(6);
    run(protected_pool, s, kept, {0, 1, 2, 1});
    check(protected_reads->reads == std::vector<uint64_t>({2, 1}),
          "mandatory thinking block missing from substitute clustering");
    check(!selected(kept, 2), "mandatory thinking block was retired");

    HistoricalKvSnapshot large;
    large.incarnation = 7;
    large.round_id = 1;
    for (unsigned i = 0; i < 11; ++i) {
        HistoricalKvBlockMeta m;
        m.handle = {7, i, 1};
        m.logical_id = i;
        m.original_begin = i * 32;
        m.valid_tokens = 32;
        m.content_version = 9;
        m.life = HistoricalKvLife::Live;
        m.sealed = true;
        large.blocks.push_back(m);
    }
    auto sliding = std::make_shared<HistoricalKvThinkingRanges>();
    sliding->arm();
    sliding->note(0, 320);
    auto reads = std::make_shared<CountedFeatures>();
    reads->distinct = true;
    HistoricalKvRedundancyPolicy::Config slide;
    slide.thinking = sliding;
    slide.segment_blocks = 2;
    slide.pool_blocks = 6;
    HistoricalKvRedundancyPolicy window(11, reads, slide);
    HistoricalKvDecisionBuffer pooled(16);
    const auto expect = [](const std::vector<uint64_t> & got, std::initializer_list<uint64_t> ids, const char * message) {
        check(got == std::vector<uint64_t>(ids), message);
    };
    run(window, large, pooled);
    expect(reads->reads, {0, 1, 2, 3, 8, 9}, "first thinking window missed the newest segment");
    check(std::find(reads->reads.begin(), reads->reads.end(), 10) == reads->reads.end(),
          "non-thinking block entered the sliding pool");
    reads->reads.clear();
    pooled.clear_after_ack();
    ++large.round_id;
    run(window, large, pooled);
    expect(reads->reads, {2, 3, 4, 5, 8, 9}, "thinking window did not slide");
    reads->reads.clear();
    pooled.clear_after_ack();
    ++large.round_id;
    run(window, large, pooled);
    expect(reads->reads, {4, 5, 6, 7, 8, 9}, "thinking window did not reach the last older cluster");

    auto tail = large;
    for (unsigned i = 11; i < 13; ++i) {
        auto meta = tail.blocks.back();
        meta.handle.slot = i;
        meta.logical_id = i;
        meta.original_begin = i * 32;
        tail.blocks.push_back(meta);
    }
    auto tail_ranges = std::make_shared<HistoricalKvThinkingRanges>();
    tail_ranges->arm();
    tail_ranges->note(0, 13 * 32);
    auto tail_cfg = slide;
    tail_cfg.thinking = tail_ranges;
    auto tail_reads = std::make_shared<CountedFeatures>();
    tail_reads->distinct = true;
    HistoricalKvRedundancyPolicy tail_window(13, tail_reads, tail_cfg);
    HistoricalKvDecisionBuffer tail_out(16);
    bool visited_tail = false;
    for (unsigned round = 0; round < 5; ++round) {
        tail.round_id = 10 + round;
        tail_reads->reads.clear();
        tail_out.clear_after_ack();
        run(tail_window, tail, tail_out);
        visited_tail |= std::find(tail_reads->reads.begin(), tail_reads->reads.end(), 10) != tail_reads->reads.end();
    }
    check(visited_tail, "sliding window never visited the last older thinking block");

    // Mandatory thinking blocks must count against the core N budget: they
    // participate in similarity comparisons even though they cannot retire.
    large.blocks[0].mandatory = true;
    large.blocks[1].mandatory = true;
    auto bounded_reads = std::make_shared<CountedFeatures>();
    bounded_reads->distinct = true;
    HistoricalKvRedundancyPolicy bounded(11, bounded_reads, slide);
    HistoricalKvDecisionBuffer bounded_out(16);
    run(bounded, large, bounded_out);
    expect(bounded_reads->reads, {0, 1, 2, 3, 8, 9},
           "mandatory references exceeded the thinking comparison budget");
    check(bounded.metrics().max_round_features <= slide.pool_blocks &&
          bounded.metrics().comparisons <= 6 * 5,
          "mandatory references exceeded pair budget");

    for (unsigned i = 2; i < 7; ++i) large.blocks[i].mandatory = true;
    ++large.round_id;
    auto many_reads = std::make_shared<CountedFeatures>();
    many_reads->distinct = true;
    HistoricalKvRedundancyPolicy many_protected(11, many_reads, slide);
    HistoricalKvDecisionBuffer many_out(16);
    run(many_protected, large, many_out);
    check(many_reads->reads.size() <= slide.pool_blocks &&
          many_protected.metrics().max_round_features <= slide.pool_blocks &&
          many_protected.metrics().comparisons <= 6 * 5,
          "many mandatory blocks defeated the thinking comparison budget");
    for (const auto & intent : many_out.values())
        check(intent.victim.slot >= 7, "mandatory thinking block was proposed for retirement");
}

static bool all_reason(const HistoricalKvDecisionBuffer & out, HistoricalKvDeleteReason reason) {
    if (out.values().empty()) return false;
    for (const auto & intent : out.values()) if (intent.reason != reason) return false;
    return true;
}

static void substitute_policy_tests() {
    // A newer replica must not outrank an older, more important substitute.
    auto s = fixture();
    auto important = std::make_shared<Features>();
    important->keys = {{{{1,0,0}}, {{1,0,0}}, {{0,1,0}}, {{0,0,1}}, {{0,-1,0}}, {{0,0,-1}}}};
    important->logits = {{10, 0, 0, 0, 0, 0}};
    HistoricalKvRedundancyPolicy keep_old(6, important, {});
    HistoricalKvDecisionBuffer out(6);
    run(keep_old, s, out, {0, 0, 1, 0, 1});
    check(out.values().size() == 1 && selected(out, 1) && !selected(out, 0),
          "newer low-importance replica was kept");
    check(all_reason(out, HistoricalKvDeleteReason::Redundancy),
          "substitute retirement was not marked redundancy");

    // A mandatory sink already represents the cluster, so every extra copy can go.
    auto sink = std::make_shared<Features>();
    sink->keys = {{{{1,0,0}}, {{1,0,0}}, {{1,0,0}}, {{1,0,0}}, {{0,1,0}}, {{0,0,1}}}};
    s = fixture();
    s.blocks[0].mandatory = true;
    HistoricalKvRedundancyPolicy keep_sink(6, sink, {});
    out.clear_after_ack();
    run(keep_sink, s, out, {0, 1, 3, 0});
    check(out.values().size() == 3 && selected(out, 1) && selected(out, 2) && selected(out, 3),
          "mandatory substitute did not release every extra replica");
    check(!selected(out, 0) && !selected(out, 4) && !selected(out, 5),
          "unique or mandatory block retired while copies remained");
    check(all_reason(out, HistoricalKvDeleteReason::Redundancy) && keep_sink.metrics().lossy_intents == 0,
          "cluster retirement used the lossy path");

    // Similarity is not transitive. The mandatory A covers B and B covers C,
    // but A does not cover C. C must remain when B is retired.
    auto chain = std::make_shared<MappedFeatures>();
    constexpr double pi = 3.14159265358979323846;
    for (unsigned i = 0; i < 3; ++i) {
        const double radians = double(i) * 20.0 * pi / 180.0;
        chain->keys.push_back({float(std::cos(radians)), float(std::sin(radians)), 0});
        chain->logits.push_back(i == 0 ? 10.0 : 0.0);
    }
    s = fixture();
    s.blocks.resize(3);
    s.blocks[0].mandatory = true;
    auto thinking = std::make_shared<HistoricalKvThinkingRanges>();
    thinking->arm();
    thinking->note(0, 96);
    HistoricalKvRedundancyPolicy::Config chain_cfg;
    chain_cfg.thinking = thinking;
    chain_cfg.pool_blocks = 3;
    chain_cfg.segment_blocks = 1;
    HistoricalKvRedundancyPolicy direct(3, chain, chain_cfg);
    HistoricalKvDecisionBuffer direct_out(3);
    run(direct, s, direct_out, {0, 1, 2, 0, 2});
    check(direct_out.values().size() == 1 && selected(direct_out, 1) && !selected(direct_out, 2),
          "non-transitive chain retired a block without a retained direct substitute");
    check(all_reason(direct_out, HistoricalKvDeleteReason::Redundancy) && direct.metrics().lossy_intents == 0,
          "direct substitute case used a lossy retirement");

    // A later Query can prefer C to B. Once A is retired using B, B->C
    // alone does not prove C can replace the unavailable A.
    auto across = std::make_shared<MappedFeatures>();
    for (unsigned i = 0; i < 3; ++i) {
        const double radians = double(i) * 20.0 * pi / 180.0;
        across->keys.push_back({float(std::cos(radians)), float(std::sin(radians)), 0});
        across->logits.push_back(i == 1 ? 10.0 : (i == 0 ? -10.0 : 0.0));
    }
    auto later = fixture();
    later.blocks.resize(3);
    HistoricalKvRedundancyPolicy successive(3, across, chain_cfg);
    HistoricalKvDecisionBuffer successive_out(3);
    run(successive, later, successive_out, {0, 1, 2, 1});
    check(successive_out.values().size() == 1 && selected(successive_out, 0) &&
          successive_out.values()[0].reason == HistoricalKvDeleteReason::Redundancy,
          "cross-round fixture did not retire A using B");
    successive.on_result({successive_out.values()[0], HistoricalKvCommitCode::Committed});
    later.blocks[0].life = HistoricalKvLife::Retired;
    across->logits[2] = 20.0;
    ++later.round_id;
    successive_out.clear_after_ack();
    run(successive, later, successive_out, {0, 1, 2, 1});
    for (const auto & intent : successive_out.values())
        check(intent.victim.slot != 1 || intent.reason != HistoricalKvDeleteReason::Redundancy,
              "cross-round substitute B was falsely called redundant using C");
    check(successive_out.values().size() == 1 && selected(successive_out, 2) &&
          !selected(successive_out, 1) &&
          successive_out.values()[0].reason == HistoricalKvDeleteReason::Lru,
          "lossy fallback discarded a prior witness while an independent block remained");
    later.blocks[2].mandatory = true;
    ++later.round_id;
    successive_out.clear_after_ack();
    run(successive, later, successive_out, {0, 1, 2, 1});
    check(successive_out.values().size() == 1 && selected(successive_out, 1) &&
          successive_out.values()[0].reason == HistoricalKvDeleteReason::Lru,
          "forced last-resort witness loss was not explicitly marked lossy");
    later.blocks[2].mandatory = false;
    ++later.round_id;
    successive_out.clear_after_ack();
    run(successive, later, successive_out, {0, 1, 2, 1, 1});
    check(successive_out.values().empty(),
          "admission demand silently deleted a prior direct substitute");

    // Medium similarity to many references is not a substitute.
    auto mixed = std::make_shared<Features>();
    mixed->keys = {{{{1,0,0}}, {{1,0,0}}, {{0.6f,0.8f,0}}, {{0,1,0}}, {{0,0,1}}, {{0,-1,0}}}};
    HistoricalKvRedundancyPolicy clones(6, mixed, {});
    s = fixture();
    out.clear_after_ack();
    run(clones, s, out, {0, 0, 1, 0, 1});
    check(selected(out, 1) && !selected(out, 2),
          "medium similarity to the reference set was treated as a substitute");
    check(all_reason(out, HistoricalKvDeleteReason::Redundancy),
          "near-clone retirement was not marked redundancy");

    // No pair above the eligibility threshold: admit a lossy ranking.
    auto unique = std::make_shared<Features>();
    unique->keys = {{{{1,0,0}}, {{0,1,0}}, {{0,0,1}}, {{0,-1,0}}, {{0,0,-1}}, {{-1,0,0}}}};
    HistoricalKvRedundancyPolicy lossy(6, unique, {});
    s = fixture();
    out.clear_after_ack();
    run(lossy, s, out);
    check(out.values().size() == 2 && all_reason(out, HistoricalKvDeleteReason::Lru),
          "unique blocks were retired as redundancy");
    check(lossy.metrics().representatives == 0 && lossy.metrics().lossy_intents == 2,
          "lossy fallback accounting");
    check(selected(out, 5) && selected(out, 1), "lossy ranking did not drop the lowest importance");

    // A unique block that was important earlier stays important after the
    // current frozen Query moves away, until the next question epoch.
    auto peaked = std::make_shared<Features>();
    peaked->keys = {{{{1,0,0}}, {{0,1,0}}, {{0,0,1}}, {{0,-1,0}}, {{0,0,-1}}, {{-1,0,0}}}};
    peaked->logits = {{10, 0, 1, 2, 3, 4}};
    HistoricalKvRedundancyPolicy keep_peak(6, peaked, {});
    s = fixture();
    out.clear_after_ack();
    run(keep_peak, s, out, {0, 0, 1, 0, 1});
    check(out.values().empty() && keep_peak.metrics().lossy_intents == 0,
          "admission wait retired a unique block without a substitute");
    ++s.round_id;
    out.clear_after_ack();
    run(keep_peak, s, out);
    check(selected(out, 1) && !selected(out, 0), "first peak round deleted the important unique block");
    peaked->logits[0] = -50;
    ++s.round_id;
    out.clear_after_ack();
    run(keep_peak, s, out);
    check(!selected(out, 0), "a later low Query score overrode the recent importance peak");
    peaked->epoch = 1;
    ++s.round_id;
    out.clear_after_ack();
    run(keep_peak, s, out);
    check(selected(out, 0), "a new question epoch kept the previous request peak");
}

static void remembered_neighbor_tests() {
    auto ranges = std::make_shared<HistoricalKvThinkingRanges>();
    ranges->arm();
    ranges->note(0, 320);
    HistoricalKvSnapshot large;
    large.incarnation = 7;
    large.round_id = 1;
    for (unsigned i = 0; i < 11; ++i) {
        HistoricalKvBlockMeta m;
        m.handle = {7, i, 1};
        m.logical_id = i;
        m.original_begin = i * 32;
        m.valid_tokens = 32;
        m.content_version = 9;
        m.life = HistoricalKvLife::Live;
        m.sealed = true;
        large.blocks.push_back(m);
    }
    auto features = std::make_shared<MappedFeatures>();
    features->keys.resize(11);
    features->logits.assign(11, 0);
    for (unsigned i = 0; i < 11; ++i) {
        const float angle = float(i) * 0.5f;
        features->keys[i] = {std::cos(angle), std::sin(angle), 0};
    }
    features->keys[0] = {1, 0, 0};
    features->keys[1] = {1, 0, 0};
    features->keys[2] = {1, 0, 0};
    HistoricalKvRedundancyPolicy::Config cfg;
    cfg.thinking = ranges;
    cfg.segment_blocks = 2;
    cfg.pool_blocks = 6;
    HistoricalKvRedundancyPolicy pinned(11, features, cfg);
    HistoricalKvDecisionBuffer out(8);
    run(pinned, large, out, {0, 0, 2, 0, 1});
    check(selected(out, 0) && !selected(out, 2), "first thinking window missed the duplicate cluster");
    features->reads.clear();
    out.clear_after_ack();
    ++large.round_id;
    run(pinned, large, out, {0, 0, 2, 0, 1});
    check(std::find(features->reads.begin(), features->reads.end(), 1) != features->reads.end(),
          "remembered substitute was dropped after the thinking window slid");
}

static std::shared_ptr<HistoricalKvRepository> write_query_blocks(
        std::shared_ptr<HistoricalKvRawStore> store, std::vector<HistoricalKvHandle> & handles) {
    auto repo = std::make_shared<HistoricalKvRepository>(2);
    for (unsigned i = 0; i < 2; ++i) {
        auto writer = repo->begin_write(i, i, 1, 1, 20, [repo, store](const HistoricalKvHandle & h) {
            return store->release(*repo, h).code == HistoricalKvRawStore::ReleaseCode::Released;
        });
        check(bool(writer), "query fixture writer missing");
        auto h = writer->handle();
        handles.push_back(h);
        check(store->reserve(*writer, h, i, 1), "query fixture reserve failed");
        const std::vector<float> k = i == 0 ? std::vector<float>{0, 1} : std::vector<float>{1, 0};
        for (unsigned l = 0; l < 2; ++l) check(store->write_layer(*writer, h, l, {1}, {2}, k, 1), "query fixture write failed");
        check(store->publish_complete(*repo, *writer, h), "query fixture publish failed");
    }
    return repo;
}

static void query_window_tests() {
    HistoricalKvRawStoreConfig c;
    c.n_layer = 2;
    c.max_tokens_per_slot = 1;
    c.packed_k_bytes_per_token = c.packed_v_bytes_per_token = 1;
    c.mean_k_elements = 2;
    c.attention_layers = {0, 1};
    auto store = std::make_shared<HistoricalKvRawStore>(c, 2);
    std::vector<HistoricalKvHandle> handles;
    auto repo = write_query_blocks(store, handles);
    auto source = std::make_shared<HistoricalKvMeanKeySource>(repo, store, 2, 1, 2);
    source->set_query_windows(2, 2);
    float classic[] = {0, 3, 0, 4};
    source->publish_query(0, 10, classic, 1);
    source->publish_query(1, 10, classic, 1);
    check(source->finish_question(), "short question rejected");
    check(source->begin_round(repo->incarnation()), "one-token question was not published");
    std::vector<float> descriptor(source->dimensions());
    double logit = 0;
    check(source->read(*repo->inspect(handles[0]), descriptor.data(), logit) &&
          std::abs(logit - 4 / std::sqrt(2.0)) < 1e-6, "one-token question logit changed");

    source->set_query_windows(2, 2);
    float observation[] = {1, 0, 1, 0};
    source->publish_observation(0, 1, observation, 1);
    source->publish_observation(1, 1, observation, 1);
    check(source->observation_samples() == 0 && !source->begin_round(repo->incarnation()),
          "a partial observation window became the query");
    source->publish_observation(0, 2, observation, 1);
    source->publish_observation(1, 2, observation, 1);
    check(source->observation_samples() == 1, "a full observation window did not publish");
    float question[] = {0, 1, 0, 1};
    source->publish_query(0, 30, question, 1);
    source->publish_query(1, 30, question, 1);
    check(source->finish_question(), "question plus observation rejected");
    check(source->begin_round(repo->incarnation()), "question plus observation query missing");
    const double expected = 1 / std::sqrt(2.0);
    check(source->read(*repo->inspect(handles[0]), descriptor.data(), logit) &&
          std::abs(logit - expected) < 1e-6, "importance did not keep the larger question logit");
    check(source->read(*repo->inspect(handles[1]), descriptor.data(), logit) &&
          std::abs(logit - expected) < 1e-6, "importance did not keep the larger observation logit");

    source->begin_question();
    check(!source->begin_round(repo->incarnation()),
          "GC used an old observation while the new question was incomplete");
    float zero[] = {0, 0, 0, 0};
    source->publish_query(0, 50, zero, 1);
    source->publish_query(1, 50, zero, 1);
    check(source->finish_question(), "new question did not finish");
    check(source->begin_round(repo->incarnation()) &&
          source->read(*repo->inspect(handles[1]), descriptor.data(), logit) &&
          std::abs(logit) < 1e-8, "new question did not retire the previous observation");

    const auto before = source->observation_samples();
    source->set_query_windows(2, 2);
    source->publish_observation(0, 8, classic, 1);
    source->publish_observation(1, 8, classic, 1);
    check(source->observation_samples() == before, "short observation published before flush");
    source->flush_observation();
    check(source->observation_samples() == before + 1, "flush did not publish the observation tail");
    source->flush_observation();
    check(source->observation_samples() == before + 1, "flush published the same tail twice");
    check(source->begin_round(repo->incarnation()) &&
          source->read(*repo->inspect(handles[0]), descriptor.data(), logit) &&
          std::abs(logit - 4 / std::sqrt(2.0)) < 1e-6, "flushed observation logit incorrect");

    source->set_query_windows(8, 8, 2);
    const auto slow0 = source->observation_samples();
    const auto fast0 = source->fast_samples();
    float fast_obs[] = {1, 0, 1, 0};
    source->publish_observation(0, 1, fast_obs, 1);
    source->publish_observation(1, 1, fast_obs, 1);
    check(source->observation_samples() == slow0 && source->fast_samples() == fast0 &&
          !source->begin_round(repo->incarnation()),
          "a partial fast window became the query");
    source->publish_observation(0, 2, fast_obs, 1);
    source->publish_observation(1, 2, fast_obs, 1);
    check(source->observation_samples() == slow0 && source->fast_samples() == fast0 + 1,
          "fast observation waited for the slow block window");
    check(source->begin_round(repo->incarnation()) &&
          source->read(*repo->inspect(handles[1]), descriptor.data(), logit) &&
          std::abs(logit - 1 / std::sqrt(2.0)) < 1e-6, "fast observation logit incorrect");

    source->set_query_windows(2, 2, 8);
    const auto slow1 = source->observation_samples();
    float first[] = {1, 0, 1, 0};
    float second[] = {0, 1, 0, 1};
    source->publish_observation(0, 1, first, 1);
    source->publish_observation(1, 1, first, 1);
    source->publish_observation(0, 2, first, 1);
    source->publish_observation(1, 2, first, 1);
    check(source->observation_samples() == slow1 + 1, "first slow observation window missing");
    source->publish_observation(0, 3, second, 1);
    source->publish_observation(1, 3, second, 1);
    source->publish_observation(0, 4, second, 1);
    source->publish_observation(1, 4, second, 1);
    check(source->observation_samples() == slow1 + 2, "second slow observation window missing");
    check(source->begin_round(repo->incarnation()) &&
          source->read(*repo->inspect(handles[1]), descriptor.data(), logit) &&
          std::abs(logit - 1 / std::sqrt(2.0)) < 1e-6,
          "a later observation window erased the earlier peak");

    // Skipping one selected layer under try_to_lock must not desynchronize
    // every later generated Query window.
    source->set_query_windows(2, 2, 2);
    const auto recovered_slow = source->observation_samples();
    const auto recovered_fast = source->fast_samples();
    source->publish_observation(0, 1, first, 1);
    source->publish_observation(1, 1, first, 1);
    source->publish_observation(0, 2, first, 1); // layer 1 was skipped
    source->publish_observation(0, 3, first, 1);
    source->publish_observation(1, 3, first, 1);
    source->publish_observation(0, 4, first, 1);
    source->publish_observation(1, 4, first, 1);
    check(source->observation_samples() == recovered_slow + 1 &&
          source->fast_samples() == recovered_fast + 1,
          "one skipped layer permanently desynchronized generated Query windows");

    source->set_query_windows(2, 2, 2);
    source->publish_query(0, 1, first, 1);
    source->publish_query(1, 1, first, 1);
    source->publish_query(0, 2, first, 1); // layer 1 was skipped
    source->publish_query(0, 3, first, 1);
    source->publish_query(1, 3, first, 1);
    check(!source->finish_question() && !source->begin_round(repo->incarnation()),
          "an incomplete question silently fell back to its surviving suffix");
    repo->close();
    for (auto m : repo->snapshot()) { repo->retire(m.handle); repo->try_reclaim(m.handle); }
}

int main() {
    try {
        policy_tests();
        candidate_budget_tests();
        source_and_gc_test();
        manager_ack_preserves_cross_round_witness();
        thinking_pool_tests();
        substitute_policy_tests();
        remembered_neighbor_tests();
        query_window_tests();
        std::cout << "historical redundancy: passed\n";
    }
    catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
