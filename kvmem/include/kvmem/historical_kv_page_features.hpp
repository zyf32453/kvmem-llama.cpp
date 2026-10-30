#pragma once

#include "kvmem/historical_kv_redundancy.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

namespace kvmem {

// An immutable-per-sealed-page feature boundary for native backends. The
// foreground publishes compact Mean-K descriptors and already computed Query
// scores; the policy worker never calls into mutable backend or CUDA state.
class HistoricalKvPageFeatures final : public HistoricalKvRedundancySource {
public:
    HistoricalKvPageFeatures(uint32_t logical_pages, size_t dimensions);
    size_t dimensions() const noexcept override { return dimensions_; }
    bool begin_round(uint64_t incarnation) override;
    bool read(const HistoricalKvBlockMeta & meta, float * descriptor, double & importance) override;
    bool refresh_before_rank() const noexcept override { return true; }
    uint64_t question_epoch() const noexcept override {
        return question_epoch_.load(std::memory_order_acquire);
    }
    std::optional<HistoricalKvEpochGuard> commit_epoch_guard(uint64_t expected) const noexcept override {
        return HistoricalKvEpochGuard{&question_epoch_, expected};
    }
    uint64_t score_epoch() const noexcept override { return score_epoch_.load(std::memory_order_acquire); }
    std::optional<HistoricalKvEpochGuard> commit_score_guard(uint64_t expected) const noexcept override {
        return HistoricalKvEpochGuard{&score_epoch_, expected};
    }

    bool has_page(uint32_t page) const;
    // A sealed page is published once. Rewriting an ID requires truncate_to()
    // after the old GC owner has stopped and every speculative suffix is gone.
    void publish_page(uint32_t page, const std::vector<float> & descriptor);
    // Partial accepted Query updates keep the same question epoch so a long
    // prefill cannot repeatedly abort an in-progress GC scoring round.
    void publish_question_scores(const std::vector<double> & scores, bool new_question = true);
    void publish_output_attention(uint32_t page, double mass);
    void clear_question();
    // Only complete pages before this token frontier remain immutable.
    // A partial tail must be recomputed when new tokens seal it again.
    void truncate_to(uint64_t token_frontier, uint32_t page_tokens);

private:
    void advance_question_epoch() noexcept;
    const size_t dimensions_;
    mutable std::mutex mu_;
    std::vector<std::vector<float>> pages_;
    std::vector<double> question_scores_;
    std::vector<double> output_peaks_;
    // Only the policy worker calls begin_round/read. Foreground updates never
    // mutate this frozen view, even during a multi-tick comparison round.
    std::vector<double> frozen_importance_;
    std::atomic<uint64_t> question_epoch_{1};
    std::atomic<uint64_t> score_epoch_{0};
};

} // namespace kvmem
