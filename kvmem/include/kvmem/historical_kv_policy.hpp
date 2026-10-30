#pragma once

// Policy-only API for historical KV deletion. Policies propose exact-generation
// intents; a separate manager validates and commits them, then reclaims storage.
// No policy may free storage or alter a live repository directly.

#include <cstddef>
#include <cstdint>
#include <array>
#include <atomic>
#include <optional>
#include <memory>
#include <vector>

namespace kvmem {
class HistoricalKvThinkingRanges;

struct HistoricalKvHandle {
    uint64_t incarnation = 0;
    uint32_t slot = 0;
    uint64_t generation = 0;
};

enum class HistoricalKvLife { Ready, Writing, Live, Retired, Releasing };

struct HistoricalKvBlockMeta {
    HistoricalKvHandle handle;
    uint64_t logical_id = 0;
    uint64_t original_begin = 0;
    uint64_t birth_seq = 0;
    uint64_t last_use_seq = 0;
    uint64_t content_version = 0;
    uint64_t payload_bytes = 0;
    uint32_t valid_tokens = 0;
    HistoricalKvLife life = HistoricalKvLife::Ready;
    uint32_t references = 0;
    bool active = false;
    bool sealed = false;
    // A strategy input, not a mechanism-level retirement veto.
    bool mandatory = false;
    // Observations only, never permission to read or release. Partial Host
    // release sets payload_bytes to zero but does not imply a Ready slot.
    uint32_t host_references = 0;
    bool host_released = false;
    uint32_t packed_kv_references = 0;
    uint32_t mean_k_references = 0;
    bool packed_kv_released = false;
    bool mean_k_released = false;
    uint64_t nvme_bytes = 0;
    bool nvme_released = false;
    uint64_t pinned_bytes = 0;
    bool pinned_released = false;
};

// Bounded, allocation-free commit conditions compiled by the policy worker.
// Referenced atomics must be owned by that policy/backend for the GC manager's
// entire lifetime. Producers and commits share the native serialized execution
// owner; these checks do not make concurrent arbitrary source mutation atomic
// with repository retirement. No policy function runs in matches().
struct HistoricalKvEpochGuard {
    const std::atomic<uint64_t> * value = nullptr;
    uint64_t expected = 0;
};

struct HistoricalKvWitnessRequirement {
    HistoricalKvHandle handle;
    uint64_t content_version = 0;
};

struct HistoricalKvCommitConditions {
    static constexpr size_t kEpochGuards = 4;
    static_assert(std::atomic<uint64_t>::is_always_lock_free &&
                  std::atomic<uint8_t>::is_always_lock_free,
                  "historical commit conditions require lock-free counters");
    bool live_only = false;
    bool sealed_only = false;
    bool reject_mandatory = false;
    const std::atomic<uint8_t> * protected_page = nullptr;
    std::array<HistoricalKvEpochGuard, kEpochGuards> epochs{};
    size_t epoch_count = 0;
    bool requires_score_evidence = false;
    uint64_t evidence_schema = 0;
    uint64_t evidence_query_epoch = 0;
    uint64_t evidence_score_epoch = 0;
    uint64_t evidence_round = 0;
    bool add_epoch(HistoricalKvEpochGuard guard) noexcept;
    bool matches(const HistoricalKvBlockMeta &) const noexcept;
};

enum class HistoricalKvPrepareCode { Ready, Reevaluate, Unsupported };

// The manager owns this immutable-for-one-round, safely sampled observation.
// A snapshot need not represent a globally simultaneous instant.
struct HistoricalKvSnapshot {
    uint64_t round_id = 0;
    uint64_t incarnation = 0;
    std::vector<HistoricalKvBlockMeta> blocks;
};

struct HistoricalKvPressure {
    uint32_t ready = 0;
    uint32_t low = 0;
    uint32_t high = 0;
    // Planning credit for committed retirements expected to return full slots.
    // This is not available capacity and must not be given to the foreground.
    uint32_t pending_slot_credit = 0;
    // An idle request can ask for more Ready capacity than the low watermark.
    // Pending retirements remain planning credit, never admission capacity.
    uint32_t requested_ready = 0;
};

struct HistoricalKvWorkBudget {
    size_t scan = 256;
    size_t intents = 64;
};

enum class HistoricalKvDeleteReason { Lru, Custom, Redundancy };

// Provenance of this bounded policy decision, not a proof of semantic/KV
// equivalence. A zero score epoch is allowed only for unscored fallback.
struct HistoricalKvScoreEvidence {
    uint64_t feature_schema = 0;
    uint64_t query_epoch = 0;
    uint64_t score_epoch = 0;
    uint64_t scoring_round = 0;
    bool feature_score_used = false;
    double importance = 0;
    double retention_score = 0;
    double similarity = 0;
    double threshold = 0;
    std::optional<HistoricalKvWitnessRequirement> witness;
};

struct HistoricalKvDeleteIntent {
    HistoricalKvHandle victim;
    uint64_t round_id = 0;
    HistoricalKvDeleteReason reason = HistoricalKvDeleteReason::Custom;
    // Only policies which explicitly need an optimistic content condition set it.
    std::optional<uint64_t> expected_content_version;
    std::optional<HistoricalKvScoreEvidence> evidence;
};

enum class HistoricalKvCommitCode {
    Committed, AlreadyRetired, Stale, Reevaluate, Retry
};

struct HistoricalKvCommitResult {
    HistoricalKvDeleteIntent intent;
    HistoricalKvCommitCode code = HistoricalKvCommitCode::Retry;
};

enum class HistoricalKvStepStatus { Yielded, Complete, NoCandidate };

struct HistoricalKvStepResult {
    HistoricalKvStepStatus status = HistoricalKvStepStatus::Yielded;
    size_t examined = 0;
    size_t emitted = 0;
};

class HistoricalKvDecisionBuffer {
public:
    explicit HistoricalKvDecisionBuffer(size_t limit);

    size_t remaining() const { return limit_ - values_.size(); }
    bool push(const HistoricalKvDeleteIntent & intent);
    const std::vector<HistoricalKvDeleteIntent> & values() const { return values_; }
    // Call only after the manager has retained or acknowledged every intent.
    void clear_after_ack() { values_.clear(); }

private:
    std::vector<HistoricalKvDeleteIntent> values_;
    size_t limit_;
};

class HistoricalKvPolicy {
public:
    struct Metrics {
        uint64_t feature_blocks = 0, missing_features = 0, comparisons = 0;
        uint64_t scored_rounds = 0, representatives = 0, no_query_rounds = 0;
        uint64_t lossy_intents = 0;
        uint64_t max_round_features = 0;
    };
    virtual ~HistoricalKvPolicy() = default;
    // Revalidate delayed decisions against the same generation's current
    // metadata. Custom strategies remain free to retire Writing/active blocks.
    virtual bool permits(const HistoricalKvBlockMeta &) const noexcept { return true; }
    // Worker-only preparation for boundary mode. Unsupported custom conditions
    // fail closed; do not fall back to calling permits() on the foreground.
    virtual HistoricalKvPrepareCode prepare_commit(const HistoricalKvBlockMeta &,
        HistoricalKvCommitConditions &) const noexcept { return HistoricalKvPrepareCode::Unsupported; }
    virtual HistoricalKvStepResult step(const HistoricalKvSnapshot & snapshot,
                                        const HistoricalKvPressure & pressure,
                                        HistoricalKvWorkBudget budget,
                                        HistoricalKvDecisionBuffer & out) = 0;
    virtual void on_result(const HistoricalKvCommitResult &) noexcept {}
    virtual Metrics metrics() const noexcept { return {}; }
};

// Deliberately independent of age, hotness, GPU residency, active status,
// sealing, and reference count. These are policy inputs, not mechanism vetoes.
bool historical_kv_can_request_delete(const HistoricalKvBlockMeta & block,
                                      uint64_t incarnation);

// Capacity-triggered LRU over a fixed per-round snapshot. One instance per
// repository, called only by its single policy/manager owner.
class HistoricalKvLruPolicy final : public HistoricalKvPolicy {
public:
    explicit HistoricalKvLruPolicy(size_t capacity, bool keep_mandatory = true,
        std::shared_ptr<HistoricalKvThinkingRanges> content_ranges = {});
    bool permits(const HistoricalKvBlockMeta & block) const noexcept override;
    HistoricalKvPrepareCode prepare_commit(const HistoricalKvBlockMeta &,
                        HistoricalKvCommitConditions &) const noexcept override;
    HistoricalKvStepResult step(const HistoricalKvSnapshot & snapshot,
                                const HistoricalKvPressure & pressure,
                                HistoricalKvWorkBudget budget,
                                HistoricalKvDecisionBuffer & out) override;

private:
    size_t capacity_;
    bool keep_mandatory_;
    std::shared_ptr<HistoricalKvThinkingRanges> content_ranges_;
    size_t cursor_ = 0;
    uint64_t round_ = 0;
    uint64_t incarnation_ = 0;
    bool initialized_ = false;
    bool pressure_mode_ = false;
    bool done_ = false;
    uint32_t wanted_ = 0;
    HistoricalKvStepStatus terminal_ = HistoricalKvStepStatus::Complete;
    std::vector<const HistoricalKvBlockMeta *> candidates_;
};

// Optional base for future strategies, including redundancy. It owns no
// repository, payload, GPU pointer, or resource-release callback.
class HistoricalKvPredicatePolicyTemplate : public HistoricalKvPolicy {
public:
    HistoricalKvStepResult step(const HistoricalKvSnapshot & snapshot,
                                const HistoricalKvPressure & pressure,
                                HistoricalKvWorkBudget budget,
                                HistoricalKvDecisionBuffer & out) final;

protected:
    virtual void begin_round(const HistoricalKvSnapshot &) {}
    virtual std::optional<HistoricalKvDeleteIntent> decide(
        const HistoricalKvBlockMeta & block,
        const HistoricalKvSnapshot & snapshot,
        const HistoricalKvPressure & pressure) = 0;

private:
    uint64_t round_ = 0;
    uint64_t incarnation_ = 0;
    size_t cursor_ = 0;
    bool initialized_ = false;
    bool done_ = false;
    bool emitted_any_ = false;
};

} // namespace kvmem
