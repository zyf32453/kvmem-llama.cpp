#include "kvmem/historical_kv_policy.hpp"

#include <algorithm>
#include <stdexcept>
#include <tuple>

namespace kvmem {

HistoricalKvDecisionBuffer::HistoricalKvDecisionBuffer(size_t limit) : limit_(limit) {
    values_.reserve(limit);
}

bool HistoricalKvDecisionBuffer::push(const HistoricalKvDeleteIntent & intent) {
    if (remaining() == 0) return false;
    values_.push_back(intent);
    return true;
}

bool historical_kv_can_request_delete(const HistoricalKvBlockMeta & block,
                                      uint64_t incarnation) {
    return block.handle.incarnation == incarnation &&
           block.handle.generation != 0 &&
           (block.life == HistoricalKvLife::Writing ||
            block.life == HistoricalKvLife::Live);
}

HistoricalKvLruPolicy::HistoricalKvLruPolicy(size_t capacity, bool keep_mandatory)
    : capacity_(capacity), keep_mandatory_(keep_mandatory) {
    if (capacity == 0) throw std::invalid_argument("historical KV policy capacity");
    candidates_.reserve(capacity);
}

bool HistoricalKvLruPolicy::permits(const HistoricalKvBlockMeta & block) const noexcept {
    return block.life == HistoricalKvLife::Live && block.sealed &&
           (!keep_mandatory_ || !block.mandatory);
}

HistoricalKvStepResult HistoricalKvLruPolicy::step(
    const HistoricalKvSnapshot & snapshot, const HistoricalKvPressure & pressure,
    HistoricalKvWorkBudget budget, HistoricalKvDecisionBuffer & out) {
    if (snapshot.blocks.size() > capacity_)
        throw std::length_error("historical KV snapshot exceeds policy capacity");
    if (!(pressure.low < pressure.high) || pressure.high > capacity_)
        throw std::invalid_argument("historical KV LRU watermarks");
    if (budget.scan == 0 || budget.intents == 0)
        throw std::invalid_argument("historical KV policy work budget");

    if (!initialized_ || round_ != snapshot.round_id ||
        incarnation_ != snapshot.incarnation) {
        if (!initialized_ || incarnation_ != snapshot.incarnation)
            pressure_mode_ = false;
        initialized_ = true;
        round_ = snapshot.round_id;
        incarnation_ = snapshot.incarnation;
        cursor_ = 0;
        done_ = false;
        candidates_.clear();
        terminal_ = HistoricalKvStepStatus::Complete;
    }
    if (done_) return {terminal_, 0, 0};
    if (pressure.ready < pressure.low) pressure_mode_ = true;
    if (pressure.ready >= pressure.high) pressure_mode_ = false;
    const uint32_t target = std::max(pressure_mode_ ? pressure.high : 0u,
                                     pressure.requested_ready);
    const uint32_t deficit = target > pressure.ready ? target - pressure.ready : 0;
    wanted_ = deficit > pressure.pending_slot_credit ? deficit - pressure.pending_slot_credit : 0;
    if (!wanted_) {
        done_ = true;
        return {HistoricalKvStepStatus::Complete, 0, 0};
    }
    if (out.remaining() == 0) return {HistoricalKvStepStatus::Yielded, 0, 0};

    size_t examined = 0;
    while (cursor_ < snapshot.blocks.size() && examined < budget.scan) {
        const auto & block = snapshot.blocks[cursor_++];
        ++examined;
        if (historical_kv_can_request_delete(block, snapshot.incarnation) && permits(block))
            candidates_.push_back(&block);
    }
    if (cursor_ != snapshot.blocks.size())
        return {HistoricalKvStepStatus::Yielded, examined, 0};
    if (candidates_.empty()) {
        done_ = true;
        terminal_ = HistoricalKvStepStatus::NoCandidate;
        return {terminal_, examined, 0};
    }

    const size_t n = std::min({size_t(wanted_), budget.intents,
                               out.remaining(), candidates_.size()});
    if (n == 0) return {HistoricalKvStepStatus::Yielded, examined, 0};
    const auto key = [](const HistoricalKvBlockMeta * block) {
        return std::make_tuple(block->last_use_seq, block->birth_seq,
                               block->logical_id, block->handle.slot,
                               block->handle.generation);
    };
    std::partial_sort(candidates_.begin(), candidates_.begin() + n,
                      candidates_.end(), [&](const auto * left, const auto * right) {
                          return key(left) < key(right);
                      });
    for (size_t i = 0; i < n; ++i) {
        HistoricalKvDeleteIntent intent;
        intent.victim = candidates_[i]->handle;
        intent.round_id = snapshot.round_id;
        intent.reason = HistoricalKvDeleteReason::Lru;
        if (!out.push(intent))
            throw std::logic_error("historical KV policy output capacity changed");
    }
    done_ = true;
    return {HistoricalKvStepStatus::Complete, examined, n};
}

HistoricalKvStepResult HistoricalKvPredicatePolicyTemplate::step(
    const HistoricalKvSnapshot & snapshot, const HistoricalKvPressure & pressure,
    HistoricalKvWorkBudget budget, HistoricalKvDecisionBuffer & out) {
    if (budget.scan == 0 || budget.intents == 0)
        throw std::invalid_argument("historical KV policy work budget");
    if (!initialized_ || round_ != snapshot.round_id ||
        incarnation_ != snapshot.incarnation) {
        initialized_ = true;
        round_ = snapshot.round_id;
        incarnation_ = snapshot.incarnation;
        cursor_ = 0;
        done_ = false;
        emitted_any_ = false;
        begin_round(snapshot);
    }
    if (done_)
        return {emitted_any_ ? HistoricalKvStepStatus::Complete
                             : HistoricalKvStepStatus::NoCandidate, 0, 0};

    size_t examined = 0;
    size_t emitted = 0;
    while (cursor_ < snapshot.blocks.size() && examined < budget.scan &&
           emitted < budget.intents && out.remaining() != 0) {
        const auto & block = snapshot.blocks[cursor_++];
        ++examined;
        if (!historical_kv_can_request_delete(block, snapshot.incarnation)) continue;
        auto intent = decide(block, snapshot, pressure);
        if (!intent) continue;
        intent->victim = block.handle;
        intent->round_id = snapshot.round_id;
        if (!out.push(*intent))
            throw std::logic_error("historical KV policy output capacity changed");
        ++emitted;
        emitted_any_ = true;
    }
    done_ = cursor_ == snapshot.blocks.size();
    return {done_ ? (emitted_any_ ? HistoricalKvStepStatus::Complete
                                 : HistoricalKvStepStatus::NoCandidate)
                  : HistoricalKvStepStatus::Yielded,
            examined, emitted};
}

} // namespace kvmem
