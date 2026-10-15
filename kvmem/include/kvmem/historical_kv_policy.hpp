#pragma once

// Policy-only API for historical KV deletion. Policies propose exact-generation
// intents; a separate manager validates and commits them, then reclaims storage.
// No policy may free storage or alter a live repository directly.

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace kvmem {

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
};

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
};

struct HistoricalKvWorkBudget {
    size_t scan = 256;
    size_t intents = 64;
};

enum class HistoricalKvDeleteReason { Lru, Custom, Redundancy };

struct HistoricalKvDeleteIntent {
    HistoricalKvHandle victim;
    uint64_t round_id = 0;
    HistoricalKvDeleteReason reason = HistoricalKvDeleteReason::Custom;
    // Only policies which explicitly need an optimistic content condition set it.
    std::optional<uint64_t> expected_content_version;
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
    virtual ~HistoricalKvPolicy() = default;
    virtual HistoricalKvStepResult step(const HistoricalKvSnapshot & snapshot,
                                        const HistoricalKvPressure & pressure,
                                        HistoricalKvWorkBudget budget,
                                        HistoricalKvDecisionBuffer & out) = 0;
    virtual void on_result(const HistoricalKvCommitResult &) noexcept {}
};

// Deliberately independent of age, hotness, GPU residency, active status,
// sealing, and reference count. These are policy inputs, not mechanism vetoes.
bool historical_kv_can_request_delete(const HistoricalKvBlockMeta & block,
                                      uint64_t incarnation);

// Capacity-triggered LRU over a fixed per-round snapshot. One instance per
// repository, called only by its single policy/manager owner.
class HistoricalKvLruPolicy final : public HistoricalKvPolicy {
public:
    explicit HistoricalKvLruPolicy(size_t capacity);
    HistoricalKvStepResult step(const HistoricalKvSnapshot & snapshot,
                                const HistoricalKvPressure & pressure,
                                HistoricalKvWorkBudget budget,
                                HistoricalKvDecisionBuffer & out) override;

private:
    size_t capacity_;
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
