// Diagnostic executable, not a throughput acceptance test. Uses the real
// manager/policies with immutable synthetic features and immediately releasable
// host payloads to isolate scheduling from Query availability and GPU leases.
// Compile separately against libkvmem; it does not change production settings.
#include "kvmem/historical_kv_gc.hpp"
#include "kvmem/historical_kv_redundancy.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace kvmem;
struct Features final : HistoricalKvRedundancySource {
    size_t dimensions() const noexcept override { return 8; }
    bool begin_round(uint64_t) override { return true; }
    bool read(const HistoricalKvBlockMeta &, float * key, double & logit) override {
        std::fill(key, key + dimensions(), 0.f); key[0] = 1.f; logit = 0;
        return true;
    }
};

static void probe(bool redundancy, unsigned scan, unsigned batch) {
    constexpr unsigned capacity = 8192, live = 2048, ticks = 2000;
    auto repo = std::make_shared<HistoricalKvRepository>(capacity, 0, 1, 0);
    auto payload = std::make_shared<std::vector<std::unique_ptr<unsigned char[]>>>(live);
    auto released = std::make_shared<unsigned>(0);
    for (unsigned i = 0; i < live; ++i) {
        (*payload)[i] = std::make_unique<unsigned char[]>(32);
        auto writer = repo->begin_write(i, i, 1, i + 1, 32,
            [payload, i, released](const HistoricalKvHandle &) {
                (*payload)[i].reset(); ++*released; return true;
            });
        if (!writer || !repo->publish_write(*writer)) throw std::runtime_error("fixture publication failed");
    }
    HistoricalKvGc::Config cfg;
    cfg.mode = HistoricalKvGc::Mode::Delete;
    cfg.ready_low = 7000; cfg.ready_high = 7168;
    cfg.scan_blocks = scan; cfg.batch_blocks = batch;
    cfg.reclaim_blocks = 64; cfg.poll_ms = 5;
    std::unique_ptr<HistoricalKvPolicy> policy;
    if (redundancy) policy = std::make_unique<HistoricalKvRedundancyPolicy>(capacity,
        std::make_shared<Features>(), HistoricalKvRedundancyPolicy::Config{});
    else policy = std::make_unique<HistoricalKvLruPolicy>(capacity);
    HistoricalKvGc gc(repo, std::move(policy), cfg);
    unsigned first_commit_tick = 0, first_release_tick = 0;
    auto start = std::chrono::steady_clock::now();
    for (unsigned tick = 1; tick <= ticks; ++tick) {
        gc.manual_tick();
        if (!first_commit_tick && gc.stats().committed) first_commit_tick = tick;
        if (!first_release_tick && *released) first_release_tick = tick;
    }
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    const auto stats = gc.stats(); const auto metrics = gc.policy_metrics();
    if (stats.policy_failures || stats.worker_failures || metrics.missing_features || metrics.no_query_rounds ||
        stats.proposed != stats.committed || !first_release_tick || *released != stats.reclaimed)
        throw std::runtime_error("probe did not isolate uninterrupted retirement/release");
    std::cout << "{\"policy\":\"" << (redundancy ? "redundancy" : "lru")
        << "\",\"capacity\":" << capacity << ",\"initial_live\":" << live
        << ",\"scan\":" << scan << ",\"batch\":" << batch << ",\"ticks\":" << ticks
        << ",\"first_commit_tick\":" << first_commit_tick << ",\"first_release_tick\":" << first_release_tick
        << ",\"first_commit_poll_floor_ms\":" << (first_commit_tick - 1) * cfg.poll_ms
        << ",\"committed\":" << stats.committed << ",\"released\":" << *released
        << ",\"scored_rounds\":" << metrics.scored_rounds << ",\"feature_reads\":" << metrics.feature_blocks
        << ",\"pairs\":" << metrics.comparisons << ",\"representatives\":" << metrics.representatives
        << ",\"missing\":" << metrics.missing_features << ",\"no_query\":" << metrics.no_query_rounds
        << ",\"compute_seconds_without_poll_sleep\":" << elapsed << "}\n";
    // Explicitly drain the remaining fixture payloads after reporting.
    repo->close();
    for (const auto & meta : repo->snapshot()) { repo->retire(meta.handle); repo->try_reclaim(meta.handle); }
}

int main() {
    try {
        probe(false, 64, 16); // Adapter's existing LRU setting.
        probe(true, 16, 16);  // Adapter's existing redundancy setting.
        probe(true, 64, 16);  // Diagnostic counterfactuals only, not a product change.
        probe(true, 16, 64);
    } catch (const std::exception & e) { std::cerr << e.what() << '\n'; return 1; }
}
