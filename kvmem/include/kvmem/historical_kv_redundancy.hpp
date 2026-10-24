#pragma once

#include "kvmem/historical_kv_policy.hpp"
#include "kvmem/historical_kv_raw_store.hpp"

#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>

namespace kvmem {

// Half-open token spans of thinking regions. The policy reads them while the
// server extends the current region; a span never shrinks. arm() switches the
// redundancy policy onto thinking blocks. With no spans yet, an armed policy
// deletes nothing, so prompt blocks are not a stand-in for thinking.
class HistoricalKvThinkingRanges {
public:
    void arm() noexcept;
    void disarm() noexcept;
    bool armed() const noexcept;
    void note(uint64_t begin, uint64_t end);
    void clear_spans() noexcept;
    bool contains(uint64_t position) const noexcept;
    bool covers(uint64_t begin, uint32_t tokens) const noexcept;
private:
    struct Span { uint64_t begin = 0; uint64_t end = 0; };
    mutable std::mutex mu_;
    std::vector<Span> spans_;
    bool armed_ = false;
};

// A read-only feature boundary. The strategy never owns a release callback.
// begin_round freezes the question, recent observation peaks, and the fast
// observation; read returns a unit descriptor and the largest query/key logit
// among those frozen channels for the exact generation in the snapshot.
class HistoricalKvRedundancySource {
public:
    enum class ReadResult { Ready, Missing, Yielded };
    virtual ~HistoricalKvRedundancySource() = default;
    virtual size_t dimensions() const noexcept = 0;
    virtual bool begin_round(uint64_t incarnation) = 0;
    virtual bool read(const HistoricalKvBlockMeta &, float * descriptor, double & logit) = 0;
    // Incremental sources may need several slices before a score covers the
    // whole question. Yielded is not a missing feature or permission to rank.
    virtual ReadResult read_bounded(const HistoricalKvBlockMeta & meta, float * descriptor, double & logit) {
        return read(meta, descriptor, logit) ? ReadResult::Ready : ReadResult::Missing;
    }
    // Live generated observations can change during a long scoring round.
    // Opt in to a second snapshot and bounded importance pass before ranking.
    // Sealed block descriptors must remain unchanged for the same version.
    virtual bool refresh_before_rank() const noexcept { return false; }
    // Increments when the next user question replaces the previous one so the
    // policy can drop request-local importance peaks without forgetting
    // content-based substitutes.
    virtual uint64_t question_epoch() const noexcept { return 0; }
};

// Block approximation of R-KV. Each sealed block keeps a pre-RoPE Mean-K.
// Every input Query token contributes to one contiguous segment mean. Keep
// all segments for the request, score their maximum on the background thread,
// and cache it by question/slot generation/content version. This is full-span
// coverage, not token-exact attention: within-segment averaging and Mean-K
// remain approximations. Generated observations still use nonblocking rings.
class HistoricalKvMeanKeySource final : public HistoricalKvRedundancySource {
public:
    static constexpr size_t kObservationKeep = 4;
    static constexpr size_t kQuestionSegmentsPerRead = 8;
    static constexpr size_t kQuestionMaxBytes = 64 * 1024 * 1024;

    HistoricalKvMeanKeySource(std::shared_ptr<HistoricalKvRepository> repository,
        std::shared_ptr<HistoricalKvRawStore> storage, uint32_t head_dimension,
        uint32_t kv_heads, uint32_t query_heads);
    bool uses_layer(uint32_t layer) const noexcept;
    void set_query_windows(uint32_t question_segment_tokens, uint32_t observation_tokens,
        uint32_t fast_tokens = 8);
    // Call before capture. Overflow invalidates the whole question instead of
    // silently using a suffix. The limit includes segment metadata and floats.
    void set_question_byte_budget(size_t bytes);
    // Prefill only: collect every selected Query position. Gaps start a new
    // segment; microbatch boundaries do not. No GC contention drops input Q.
    void publish_query(uint32_t layer, uint64_t end, const float * rows, uint32_t count);
    // Publish one immutable bank after all prefill producers have drained.
    // False means missing/inconsistent layers, nonfinite data, or overflow.
    bool finish_question(uint64_t expected_tokens = UINT64_MAX);
    // Only after the caller verifies an exact cached prefix and identical
    // full-question bounds. Reuse the immutable bank, not a generated suffix.
    bool reuse_question(uint64_t expected_tokens);
    // Generated observation, including final answers and no-thinking turns.
    // The slow ring publishes a block-aligned mean; the
    // fast ring publishes a sliding full window so a mid-block focus shift is
    // visible before the next slow mean. Both rings receive the same tokens.
    void publish_observation(uint32_t layer, uint64_t end, const float * rows, uint32_t count);
    void flush_observation();
    // Next request: drop the previous question. Previous observation peaks
    // stay until the new question sample is complete.
    void begin_question();
    size_t dimensions() const noexcept override { return dimensions_; }
    bool begin_round(uint64_t incarnation) override;
    bool read(const HistoricalKvBlockMeta &, float *, double &) override;
    ReadResult read_bounded(const HistoricalKvBlockMeta &, float *, double &) override;
    bool refresh_before_rank() const noexcept override { return true; }
    uint64_t question_epoch() const noexcept override { return question_epoch_.load(); }
    uint64_t query_samples() const noexcept { return samples_.load(); }
    uint64_t observation_samples() const noexcept { return observation_samples_.load(); }
    uint64_t fast_samples() const noexcept { return fast_samples_.load(); }
    uint64_t skipped_query_samples() const noexcept { return skipped_.load(); }
    uint64_t question_tokens() const noexcept { return question_tokens_.load(); }
    uint64_t question_segments() const noexcept { return question_segments_.load(); }
    uint64_t question_bytes() const noexcept { return question_bytes_.load(); }
    uint64_t question_score_evaluations() const noexcept { return question_score_evaluations_.load(); }
    uint64_t question_score_yields() const noexcept { return question_score_yields_.load(); }
    uint64_t invalid_questions() const noexcept { return invalid_questions_.load(); }
private:
    struct QuestionSegment {
        uint64_t begin = 0, end = 0;
        uint32_t count = 0;
        std::vector<float> mean;
    };
    struct Question {
        uint64_t epoch = 0;
        size_t bytes = 0;
        std::vector<std::vector<QuestionSegment>> layers;
    };
    struct QuestionScore {
        uint64_t epoch = UINT64_MAX;
        size_t next = 0;
        double peak = -std::numeric_limits<double>::infinity();
    };
    void reset_question_locked();
    double score_question_segment(const float * key, size_t segment) const;
    double score_query_layer(const float * key, const float * query) const;
    struct Ring {
        std::vector<float> samples, mean;
        std::vector<uint32_t> count, cursor;
        std::vector<uint64_t> end, accepted;
        uint32_t window = 32;
        bool valid = false;
        uint64_t batch_end = 0;
        uint64_t published_accepted = 0;
    };
    bool append_ring(Ring & ring, uint32_t layer, uint64_t end, const float * rows, uint32_t count);
    bool publish_ring(Ring & ring, uint64_t end, bool block_aligned, bool require_full = false);
    void reset_ring(Ring & ring);
    void reset_observations();
    void remember_observation();
    void ring_mean(const Ring & ring, size_t layer, float * dst) const;
    double score_query(const float * key, const float * query) const;
    std::shared_ptr<HistoricalKvRepository> repository_;
    std::shared_ptr<HistoricalKvRawStore> storage_;
    uint32_t head_dimension_, kv_heads_, query_heads_;
    size_t dimensions_, query_width_;
    std::vector<uint32_t> layers_;
    struct Cached { HistoricalKvHandle handle; uint64_t version = 0; bool valid = false; };
    std::vector<Cached> cached_;
    std::vector<float> keys_;
    Ring observation_, observation_fast_;
    uint32_t question_segment_tokens_ = 32;
    size_t question_byte_budget_ = kQuestionMaxBytes;
    bool question_collecting_ = false, question_invalid_ = false;
    std::shared_ptr<Question> question_building_;
    std::shared_ptr<const Question> question_, question_round_;
    std::vector<QuestionScore> question_scores_;
    std::vector<float> observation_keep_;
    size_t observation_keep_count_ = 0, observation_keep_head_ = 0;
    std::vector<float> observation_keep_round_, fast_round_;
    size_t observation_keep_round_count_ = 0;
    bool fast_round_valid_ = false;
    bool retire_observation_ = false;
    std::mutex query_mutex_;
    std::atomic<uint64_t> samples_{0}, observation_samples_{0}, fast_samples_{0}, skipped_{0};
    std::atomic<uint64_t> question_epoch_{0};
    std::atomic<uint64_t> question_tokens_{0}, question_segments_{0}, question_bytes_{0};
    std::atomic<uint64_t> question_score_evaluations_{0}, question_score_yields_{0}, invalid_questions_{0};
};

class HistoricalKvRedundancyPolicy final : public HistoricalKvPolicy {
public:
    struct Config {
        double importance_weight = 0.1;
        // Eligibility and clustering threshold: a block is redundant only when
        // some compared block reaches this cosine. It is not a deletion score.
        double representative_similarity = 0.9;
        // Newest half of this many blocks plus an age-banded older half. It is
        // not a newest-only window. Remembered substitutes are added on top.
        size_t reference_blocks = 32;
        // Experimental, used only while thinking capture is not armed: 0 keeps
        // the all-block / recent-reference policy. A positive limit scores a
        // bounded age-stratified pool and compares every selected candidate
        // with every other selected one.
        size_t max_candidate_blocks = 0;
        // Armed thinking rounds compare a capped pool pairwise. The newest
        // segment_blocks are always included; older thinking blocks slide
        // through the remaining slots. 0 means every current thinking block.
        size_t segment_blocks = 4;
        size_t pool_blocks = 0;
        std::shared_ptr<HistoricalKvThinkingRanges> thinking;
    };
    HistoricalKvRedundancyPolicy(size_t capacity,
        std::shared_ptr<HistoricalKvRedundancySource> source, Config config);
    bool permits(const HistoricalKvBlockMeta &) const noexcept override;
    HistoricalKvStepResult step(const HistoricalKvSnapshot &, const HistoricalKvPressure &,
        HistoricalKvWorkBudget, HistoricalKvDecisionBuffer &) override;
    void on_result(const HistoricalKvCommitResult &) noexcept override;
    Metrics metrics() const noexcept override;
private:
    struct Row {
        const HistoricalKvBlockMeta * meta;
        double importance = 0, redundancy = 0, score = 0, max_sim = -1;
        size_t compared = 0, mate = SIZE_MAX;
        bool retained = false;
    };
    struct Seen {
        uint64_t generation = 0, content_version = 0;
        double peak = -std::numeric_limits<double>::infinity();
        uint32_t mate_slot = UINT32_MAX;
        uint64_t mate_generation = 0, mate_version = 0;
        double mate_cosine = -1;
        // A committed redundancy retirement named this live generation as
        // its direct substitute. Later B->C similarity alone cannot retire B
        // as redundancy for the already unavailable A.
        bool carries_retired = false;
    };
    bool thinking_block(const HistoricalKvBlockMeta & meta) const noexcept;
    bool eligible_candidate(const HistoricalKvBlockMeta & meta) const noexcept;
    bool carries_retired_substitute(const HistoricalKvBlockMeta & meta) const noexcept;
    void select_thinking_pool();
    void limit_older_candidates(size_t limit);
    void select_references();
    void add_remembered_references();
    void pin_remembered_neighbors(const HistoricalKvSnapshot & snapshot);
    void remember_scored();
    void refresh_question_peaks();
    double peaked_logit(const HistoricalKvBlockMeta & meta, double logit) const noexcept;
    size_t row_for(uint32_t slot, uint64_t generation, uint64_t version) const noexcept;
    size_t snapshot_index(const HistoricalKvSnapshot & snapshot, uint32_t slot,
        uint64_t generation, uint64_t version) const noexcept;
    bool better_representative(size_t a, size_t b) const noexcept;
    size_t emit_ranked(std::vector<size_t> & group, size_t take, HistoricalKvDeleteReason reason,
        const HistoricalKvSnapshot & snapshot, HistoricalKvDecisionBuffer & out);
    size_t capacity_, dimensions_;
    std::shared_ptr<HistoricalKvRedundancySource> source_;
    Config config_;
    uint64_t round_ = 0, incarnation_ = 0, question_epoch_ = 0;
    size_t cursor_ = 0, pair_row_ = 0, pair_reference_ = 0, older_origin_ = 0, refresh_cursor_ = 0;
    bool initialized_ = false, pressure_mode_ = false, done_ = false, collected_ = false, source_ready_ = false;
    bool pool_built_ = false;
    bool refresh_started_ = false;
    bool thinking_mode_ = false;
    std::vector<Row> rows_;
    std::vector<float> descriptors_;
    struct DirectEdge { size_t index; double similarity; };
    // Sparse direct-similarity edges from the comparisons already budgeted
    // for this round; unlike connected components, each edge is a proof of
    // an actual retained substitute when a victim is proposed.
    std::vector<std::vector<DirectEdge>> neighbors_;
    std::vector<size_t> references_, ranked_, candidate_indices_, candidate_scratch_;
    std::vector<size_t> protected_indices_;
    std::vector<Seen> seen_;
    std::atomic<uint64_t> feature_blocks_{0}, missing_{0}, comparisons_{0}, max_round_features_{0};
    std::atomic<uint64_t> scored_{0}, representatives_{0}, no_query_{0}, lossy_{0};
};

} // namespace kvmem
