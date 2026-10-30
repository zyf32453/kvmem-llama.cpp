#pragma once

#include "kvmem/historical_kv_policy.hpp"
#include "kvmem/historical_kv_auxiliary.hpp"

#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>

namespace kvmem {

// Half-open token spans of thinking regions. The policy reads them while the
// server extends the current region; a span never shrinks. arm() switches the
// redundancy policy onto thinking blocks. With no spans yet, an armed policy
// deletes nothing by default, so prompt blocks are not a stand-in for thinking.
// The optional reply tier extends eligibility using proven generated spans.
class HistoricalKvThinkingRanges {
public:
    explicit HistoricalKvThinkingRanges(std::shared_ptr<HistoricalKvAuxBudget> auxiliary = {})
        : spans_(HistoricalKvAuxAllocator<Span>(auxiliary, HistoricalKvAuxKind::Ranges)),
          completed_replies_(spans_.get_allocator()) {}
    void arm() noexcept;
    void disarm() noexcept;
    bool armed() const noexcept;
    void note(uint64_t begin, uint64_t end);
    void clear_spans() noexcept;
    void truncate(uint64_t end) noexcept;
    bool contains(uint64_t position) const noexcept;
    bool covers(uint64_t begin, uint32_t tokens) const noexcept;
    // Optional second tier: only server-generated, successfully completed
    // replies older than the latest reply. The latest answer body is protected;
    // its thinking remains tier 0. Never infer roles from raw text.
    void enable_completed_replies(bool enabled) noexcept;
    bool tiered() const noexcept;
    void note_completed_reply(uint64_t begin, uint64_t end);
    // 0 = thinking, 1 = older completed reply, 2 = protected/unclassified.
    unsigned priority(uint64_t begin, uint32_t tokens) const noexcept;
    // Worker-only classification certificate. Monotone appends cannot change
    // classifications entirely before the captured thinking frontier. Rewinds,
    // overlapping edits and reply-tier changes revoke such certificates.
    bool prepare_commit(uint64_t begin, uint32_t tokens, unsigned allowed_tiers,
                        HistoricalKvCommitConditions &) const noexcept;
private:
    friend struct HistoricalKvThinkingRangesTestAccess;
    unsigned priority_locked(uint64_t begin, uint64_t end) const noexcept;
    static void advance(std::atomic<uint64_t> &) noexcept;
    struct Span { uint64_t begin = 0; uint64_t end = 0; };
    mutable std::mutex mu_;
    HistoricalKvAuxVector<Span> spans_;
    bool armed_ = false;
    bool tiered_ = false;
    HistoricalKvAuxVector<Span> completed_replies_;
    uint64_t thinking_frontier_ = 0;
    std::atomic<uint64_t> classification_revision_{1};
    std::atomic<uint64_t> growth_revision_{1};
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
    // Dynamic sources must expose a stable atomic owned for their lifetime.
    // The default is deliberately unsupported, even if question_epoch() is 0.
    virtual std::optional<HistoricalKvEpochGuard> commit_epoch_guard(uint64_t) const noexcept {
        return std::nullopt;
    }
    // Schema is immutable for the source/manager lifetime. Score epochs name
    // frozen importance banks, not every live attention update, so decoding
    // does not perpetually revoke pending decisions.
    virtual uint64_t feature_schema_id() const noexcept { return 1; }
    virtual uint64_t score_epoch() const noexcept { return 0; }
    virtual std::optional<HistoricalKvEpochGuard> commit_score_guard(uint64_t) const noexcept {
        return std::nullopt;
    }
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
        // When the preferred thinking/reply tier cannot satisfy the whole
        // capacity request, fill the remainder with cold unclassified blocks.
        // Disabled by default to preserve existing backend policy semantics.
        bool lru_fallback_unclassified = false;
        // Share across domains/components; null preserves unbudgeted legacy
        // callers and is not a claim that their total memory is bounded.
        std::shared_ptr<HistoricalKvAuxBudget> auxiliary;
    };
    HistoricalKvRedundancyPolicy(size_t capacity,
        std::shared_ptr<HistoricalKvRedundancySource> source, Config config);
    bool permits(const HistoricalKvBlockMeta &) const noexcept override;
    HistoricalKvPrepareCode prepare_commit(const HistoricalKvBlockMeta &,
                        HistoricalKvCommitConditions &) const noexcept override;
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
    size_t emit_ranked(HistoricalKvAuxVector<size_t> & group, size_t take, HistoricalKvDeleteReason reason,
        const HistoricalKvSnapshot & snapshot, HistoricalKvDecisionBuffer & out);
    size_t emit_unclassified_lru(HistoricalKvAuxVector<size_t> & indices, size_t take,
        const HistoricalKvSnapshot & snapshot, HistoricalKvDecisionBuffer & out);
    HistoricalKvScoreEvidence evidence(uint64_t round, bool scored) const noexcept;
    size_t capacity_, dimensions_;
    std::shared_ptr<HistoricalKvRedundancySource> source_;
    Config config_;
    uint64_t round_ = 0, incarnation_ = 0, question_epoch_ = 0;
    uint64_t score_epoch_ = 0;
    size_t cursor_ = 0, pair_row_ = 0, pair_reference_ = 0, older_origin_ = 0, refresh_cursor_ = 0;
    bool initialized_ = false, pressure_mode_ = false, done_ = false, collected_ = false, source_ready_ = false;
    bool pool_built_ = false;
    bool refresh_started_ = false;
    bool thinking_mode_ = false;
    bool lru_fallback_mode_ = false;
    bool fallback_commit_enabled_ = false;
    unsigned content_tier_ = 0;
    bool audit_ = false;
    HistoricalKvAuxVector<Row> rows_;
    HistoricalKvAuxVector<float> descriptors_;
    struct DirectEdge { size_t index; double similarity; };
    // Sparse direct-similarity edges from the comparisons already budgeted
    // for this round; unlike connected components, each edge is a proof of
    // an actual retained substitute when a victim is proposed.
    HistoricalKvAuxVector<HistoricalKvAuxVector<DirectEdge>> neighbors_;
    HistoricalKvAuxVector<size_t> references_, ranked_, candidate_indices_, candidate_scratch_;
    HistoricalKvAuxVector<size_t> unclassified_indices_;
    HistoricalKvAuxVector<size_t> protected_indices_;
    HistoricalKvAuxVector<Seen> seen_;
    std::atomic<uint64_t> feature_blocks_{0}, missing_{0}, comparisons_{0}, max_round_features_{0};
    std::atomic<uint64_t> scored_{0}, representatives_{0}, no_query_{0}, lossy_{0};
};

} // namespace kvmem
