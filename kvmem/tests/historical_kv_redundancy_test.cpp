#include "kvmem/historical_kv_redundancy.hpp"
#include "kvmem/historical_kv_gc.hpp"
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>

using namespace kvmem;
static void check(bool ok, const char * message) { if (!ok) throw std::runtime_error(message); }

struct Features : HistoricalKvRedundancySource {
    bool available = true;
    size_t missing = SIZE_MAX;
    std::array<std::array<float, 3>, 6> keys{{{{0,1,0}}, {{1,0,0}}, {{1,0,0}}, {{1,0,0}}, {{1,0,0}}, {{0,0,1}}}};
    std::array<double, 6> logits{{3,0,0,0,0,-1}};
    size_t dimensions() const noexcept override { return 3; }
    bool begin_round(uint64_t) override { return available; }
    bool read(const HistoricalKvBlockMeta & m, float * out, double & logit) override {
        if (m.logical_id == missing) return false;
        std::copy(keys.at(m.logical_id).begin(), keys.at(m.logical_id).end(), out);
        logit = logits.at(m.logical_id); return true;
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
    check(source->begin_round(repo->incarnation()), "complete Query missing");
    std::vector<float> descriptor(source->dimensions()); double logit = 0;
    check(source->read(*repo->inspect(handles[0]),descriptor.data(),logit) &&
          std::abs(logit - 4/std::sqrt(2.0)) < 1e-6, "GQA max/Mean-K importance incorrect");
    double norm = 0; for (auto x : descriptor) norm += x*x;
    check(std::abs(norm-1) < 1e-6, "descriptor not normalized");
    // A producer publishes opposite per-layer Queries. A coherent frame has
    // exactly zero logit; mixing two different frame numbers does not.
    float zero[] = {0,0,0,0};
    source->publish_query(0,20,zero,1); source->publish_query(1,20,zero,1);
    std::atomic<bool> finished{false};
    std::thread producer([&] {
        for (unsigned i=1;i<=1000;++i) {
            float a[] = {0,float(i),0,float(i)}, b[] = {0,-float(i),0,-float(i)};
            source->publish_query(0,20+i,a,1); source->publish_query(1,20+i,b,1);
        }
        finished = true;
    });
    bool coherent = true;
    while (!finished.load()) if (source->begin_round(repo->incarnation())) {
        coherent &= source->read(*repo->inspect(handles[0]),descriptor.data(),logit) && std::abs(logit) < 1e-8;
    }
    producer.join(); check(coherent, "mixed Query layers across concurrent publication");
    source->publish_query(0,2000,q,1); source->publish_query(1,2000,q,1);
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
int main() {
    try { policy_tests(); source_and_gc_test(); std::cout << "historical redundancy: passed\n"; }
    catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
