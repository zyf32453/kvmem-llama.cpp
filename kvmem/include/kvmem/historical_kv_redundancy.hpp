#pragma once

#include "kvmem/historical_kv_policy.hpp"
#include "kvmem/historical_kv_raw_store.hpp"

#include <atomic>
#include <memory>
#include <mutex>

namespace kvmem {

// A read-only feature boundary. The strategy never owns a release callback.
// begin_round freezes the query; read returns a unit descriptor and a query/key
// logit for the exact generation/content version in the manager's snapshot.
class HistoricalKvRedundancySource {
public:
    virtual ~HistoricalKvRedundancySource() = default;
    virtual size_t dimensions() const noexcept = 0;
    virtual bool begin_round(uint64_t incarnation) = 0;
    virtual bool read(const HistoricalKvBlockMeta &, float * descriptor, double & logit) = 0;
};

// Block approximation: pre-RoPE Mean-K on up to four evenly spaced attention
// layers, and the most recent complete input Query sample (up to eight rows).
// This is not token-level attention and does not capture a new Query each decode.
// The producer never waits for GC: a contended sample is skipped. Complete
// previous samples remain usable; partial layer updates are never published.
class HistoricalKvMeanKeySource final : public HistoricalKvRedundancySource {
public:
    HistoricalKvMeanKeySource(std::shared_ptr<HistoricalKvRepository> repository,
        std::shared_ptr<HistoricalKvRawStore> storage, uint32_t head_dimension,
        uint32_t kv_heads, uint32_t query_heads);
    bool uses_layer(uint32_t layer) const noexcept;
    void publish_query(uint32_t layer, uint64_t end, const float * rows, uint32_t count);
    size_t dimensions() const noexcept override { return dimensions_; }
    bool begin_round(uint64_t incarnation) override;
    bool read(const HistoricalKvBlockMeta &, float *, double &) override;
    uint64_t query_samples() const noexcept { return samples_.load(); }
    uint64_t skipped_query_samples() const noexcept { return skipped_.load(); }
private:
    struct Cached { HistoricalKvHandle handle; uint64_t version = 0; bool valid = false; };
    std::shared_ptr<HistoricalKvRepository> repository_;
    std::shared_ptr<HistoricalKvRawStore> storage_;
    uint32_t head_dimension_, kv_heads_, query_heads_;
    size_t dimensions_, query_width_;
    std::vector<uint32_t> layers_;
    std::vector<Cached> cached_;
    std::vector<float> keys_, staged_query_, published_query_, round_query_;
    std::vector<uint64_t> query_ends_;
    std::mutex query_mutex_;
    bool query_valid_ = false;
    std::atomic<uint64_t> samples_{0}, skipped_{0};
};

class HistoricalKvRedundancyPolicy final : public HistoricalKvPolicy {
public:
    struct Config {
        double importance_weight = 0.1;
        double representative_similarity = 0.9;
        size_t reference_blocks = 32;
    };
    HistoricalKvRedundancyPolicy(size_t capacity,
        std::shared_ptr<HistoricalKvRedundancySource> source, Config config);
    bool permits(const HistoricalKvBlockMeta &) const noexcept override;
    HistoricalKvStepResult step(const HistoricalKvSnapshot &, const HistoricalKvPressure &,
        HistoricalKvWorkBudget, HistoricalKvDecisionBuffer &) override;
    Metrics metrics() const noexcept override;
private:
    struct Row {
        const HistoricalKvBlockMeta * meta;
        double importance = 0, redundancy = 0, score = 0;
        size_t representative = SIZE_MAX, compared = 0;
        bool retained = false;
    };
    size_t capacity_, dimensions_;
    std::shared_ptr<HistoricalKvRedundancySource> source_;
    Config config_;
    uint64_t round_ = 0, incarnation_ = 0;
    size_t cursor_ = 0, pair_row_ = 0, pair_reference_ = 0;
    bool initialized_ = false, pressure_mode_ = false, done_ = false, collected_ = false;
    std::vector<Row> rows_;
    std::vector<float> descriptors_;
    std::vector<size_t> references_, ranked_;
    std::atomic<uint64_t> feature_blocks_{0}, missing_{0}, comparisons_{0};
    std::atomic<uint64_t> scored_{0}, representatives_{0}, no_query_{0};
};

} // namespace kvmem
