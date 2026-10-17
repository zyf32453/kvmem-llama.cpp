#include "kvmem/historical_kv_gc.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <utility>

namespace kvmem {

HistoricalKvGc::HistoricalKvGc(std::shared_ptr<HistoricalKvRepository> repository,
                               std::unique_ptr<HistoricalKvPolicy> policy,
                               Config config)
    : repository_(std::move(repository)), policy_(std::move(policy)), config_(config),
      decisions_(repository_ ? std::min(config.batch_blocks, repository_->capacity()) : 0) {
    if (!repository_ || !policy_ || !repository_->capacity() ||
        ((config_.ready_low || config_.ready_high) &&
         (!config_.ready_low || !(config_.ready_low < config_.ready_high) ||
          config_.ready_high > repository_->capacity())) ||
        !config_.scan_blocks || !config_.batch_blocks ||
        config_.batch_blocks > repository_->capacity() ||
        !config_.reclaim_blocks || !config_.poll_ms) {
        throw std::invalid_argument("invalid historical KV GC configuration");
    }
    retry_.reserve(repository_->capacity());
    round_.blocks.reserve(repository_->capacity());
    observation_.blocks.reserve(repository_->capacity());
    seen_generation_.resize(repository_->capacity());
    shadow_generation_.resize(repository_->capacity());
    if (!repository_->claim_gc_owner())
        throw std::logic_error("historical KV repository already has a GC owner");
}

HistoricalKvGc::~HistoricalKvGc() {
    stop();
    repository_->release_gc_owner();
}

void HistoricalKvGc::start() {
    std::lock_guard<std::mutex> guard(tick_mu_);
    if (worker_.joinable()) throw std::logic_error("historical KV GC already started");
    stopping_.store(false, std::memory_order_release);
    worker_finished_.store(false, std::memory_order_release);
    try { worker_ = std::thread([this] { run(); }); }
    catch (...) { worker_finished_.store(true, std::memory_order_release); throw; }
}

void HistoricalKvGc::stop() noexcept {
    request_stop();
    if (worker_.joinable()) worker_.join();
}

void HistoricalKvGc::run() noexcept {
    while (!stopping_.load(std::memory_order_acquire)) {
        try { manual_tick(); }
        catch (const std::exception & e) {
            ++worker_failures_;
            std::fprintf(stderr, "[historical-kv-gc] worker stopped: %s\n", e.what());
            stopping_.store(true, std::memory_order_release);
            break;
        } catch (...) {
            ++worker_failures_;
            std::fprintf(stderr, "[historical-kv-gc] worker stopped: unknown error\n");
            stopping_.store(true, std::memory_order_release);
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(config_.poll_ms));
    }
    worker_finished_.store(true, std::memory_order_release);
}

HistoricalKvCommitCode HistoricalKvGc::commit(const HistoricalKvDeleteIntent & intent) {
    // The repository does not yet have an atomic content-version/retire
    // transaction. Never silently ignore a condition from redundancy policy.
    if (intent.expected_content_version) return HistoricalKvCommitCode::Reevaluate;
    switch (repository_->retire(intent.victim)) {
    case HistoricalKvRepository::RetireResult::Committed:
        return HistoricalKvCommitCode::Committed;
    case HistoricalKvRepository::RetireResult::AlreadyRetired:
        return HistoricalKvCommitCode::AlreadyRetired;
    case HistoricalKvRepository::RetireResult::Stale:
        return HistoricalKvCommitCode::Stale;
    case HistoricalKvRepository::RetireResult::Retry:
        return HistoricalKvCommitCode::Retry;
    }
    return HistoricalKvCommitCode::Retry;
}

void HistoricalKvGc::acknowledge(const HistoricalKvDeleteIntent & intent,
                                 HistoricalKvCommitCode code) {
    policy_->on_result({intent, code});
    switch (code) {
    case HistoricalKvCommitCode::Committed: ++committed_; break;
    case HistoricalKvCommitCode::AlreadyRetired: break;
    case HistoricalKvCommitCode::Stale: ++stale_; break;
    case HistoricalKvCommitCode::Reevaluate: ++reevaluate_; break;
    case HistoricalKvCommitCode::Retry: ++retry_count_; break;
    }
}

void HistoricalKvGc::manual_tick() {
    std::lock_guard<std::mutex> guard(tick_mu_);
    ++ticks_;
    // Progress previous retirements even when policy has no candidate, the
    // output buffer is full, or new policy decisions are switched off.
    const uint32_t capacity = repository_->capacity();
    const uint32_t reclaim_scan = std::min(config_.reclaim_blocks,
                                           capacity - reclaim_cursor_);
    repository_->observe_range_into(reclaim_cursor_, reclaim_scan, observation_);
    for (const auto & block : observation_.blocks) {
            if (block.life != HistoricalKvLife::Retired &&
                block.life != HistoricalKvLife::Releasing) continue;
            switch (repository_->try_reclaim(block.handle)) {
            case HistoricalKvRepository::ReclaimResult::Complete: ++reclaimed_; break;
            case HistoricalKvRepository::ReclaimResult::Deferred: ++deferred_; break;
            case HistoricalKvRepository::ReclaimResult::Stale: break;
            case HistoricalKvRepository::ReclaimResult::Retry: ++retry_count_; break;
            }
    }
    reclaim_cursor_ = reclaim_scan == capacity - reclaim_cursor_ ? 0 :
                      reclaim_cursor_ + reclaim_scan;
    if (config_.mode == Mode::Off) return;

    // Retry retained commands before taking a fresh snapshot. A previous
    // failure must not cause the next round to over-delete new candidates.
    if (!retry_.empty()) {
        size_t write = 0;
        for (size_t i = 0; i < retry_.size(); ++i) {
            const auto code = commit(retry_[i]);
            acknowledge(retry_[i], code);
            if (code == HistoricalKvCommitCode::Retry) retry_[write++] = retry_[i];
        }
        retry_.resize(write);
        if (!retry_.empty()) return;
    }
    if (policy_failed_) return;

    if (!round_active_) {
        round_active_ = true;
        round_.round_id = next_round_id_++;
        round_.incarnation = repository_->incarnation();
        round_.blocks.clear();
        std::fill(seen_generation_.begin(), seen_generation_.end(), 0);
        collect_cursor_ = 0;
        pressure_.ready = 0;
        pressure_.low = config_.ready_low;
        pressure_.high = config_.ready_high;
        pressure_.pending_slot_credit = 0;
    }
    // Form one fixed observation over bounded slices; the policy never sees
    // a half-built round. It retains the resulting snapshot across Yielded.
    if (collect_cursor_ < capacity) {
        const uint32_t scan = std::min(config_.scan_blocks, capacity - collect_cursor_);
        repository_->observe_range_into(collect_cursor_, scan, observation_);
        pressure_.ready += observation_.ready;
        pressure_.pending_slot_credit += observation_.pending;
        if (config_.mode == Mode::Shadow) {
            for (auto & block : observation_.blocks) {
                if (block.handle.slot >= shadow_generation_.size() ||
                    shadow_generation_[block.handle.slot] != block.handle.generation ||
                    !historical_kv_can_request_delete(block, round_.incarnation)) continue;
                // A shadow candidate remains physically Live. Model the
                // effect of its hypothetical retirement for the policy only.
                block.life = HistoricalKvLife::Retired;
                block.sealed = false;
                ++pressure_.pending_slot_credit;
            }
        }
        round_.blocks.insert(round_.blocks.end(), observation_.blocks.begin(),
                             observation_.blocks.end());
        collect_cursor_ += scan;
        if (collect_cursor_ < capacity) return;
    }
    decisions_.clear_after_ack();
    HistoricalKvStepResult step;
    try {
        step = policy_->step(round_, pressure_,
                             {config_.scan_blocks, config_.batch_blocks}, decisions_);
    } catch (const std::exception & e) {
        ++policy_failures_;
        policy_failed_ = true;
        decisions_.clear_after_ack();
        round_.blocks.clear();
        round_active_ = false;
        collect_cursor_ = 0;
        std::fprintf(stderr, "[historical-kv-gc] policy disabled: %s\n", e.what());
        return;
    } catch (...) {
        ++policy_failures_;
        policy_failed_ = true;
        decisions_.clear_after_ack();
        round_.blocks.clear();
        round_active_ = false;
        collect_cursor_ = 0;
        std::fprintf(stderr, "[historical-kv-gc] policy disabled: unknown error\n");
        return;
    }
    proposed_.fetch_add(decisions_.values().size(), std::memory_order_relaxed);
    for (const auto & intent : decisions_.values()) {
        // A policy cannot delete an arbitrary live block outside the fixed
        // observation it was given for this round.
        // observe_range_into() returns blocks in slot order, and each slice is
        // appended in that order. An intent therefore needs one logarithmic
        // lookup instead of rescanning the full directory for every victim.
        const auto observed_slot = std::lower_bound(
            round_.blocks.begin(), round_.blocks.end(), intent.victim.slot,
            [](const HistoricalKvBlockMeta & block, uint32_t slot) {
                return block.handle.slot < slot;
            });
        const bool observed = intent.round_id == round_.round_id &&
            observed_slot != round_.blocks.end() &&
            observed_slot->handle.slot == intent.victim.slot &&
            observed_slot->handle.incarnation == intent.victim.incarnation &&
            observed_slot->handle.generation == intent.victim.generation &&
            historical_kv_can_request_delete(*observed_slot, round_.incarnation);
        if (!observed) {
            acknowledge(intent, HistoricalKvCommitCode::Stale);
            continue;
        }
        if (seen_generation_[intent.victim.slot] == intent.victim.generation) {
            if (config_.mode != Mode::Shadow)
                acknowledge(intent, HistoricalKvCommitCode::AlreadyRetired);
            continue;
        }
        seen_generation_[intent.victim.slot] = intent.victim.generation;
        if (config_.mode == Mode::Shadow) {
            shadow_generation_[intent.victim.slot] = intent.victim.generation;
            ++shadow_;
            continue;
        }
        const auto code = commit(intent);
        acknowledge(intent, code);
        if (code == HistoricalKvCommitCode::Retry) {
            const bool duplicate = std::any_of(retry_.begin(), retry_.end(),
                [&](const auto & pending) {
                    return pending.victim.incarnation == intent.victim.incarnation &&
                           pending.victim.slot == intent.victim.slot &&
                           pending.victim.generation == intent.victim.generation;
                });
            if (!duplicate) retry_.push_back(intent);
        }
    }
    decisions_.clear_after_ack();
    if (step.status != HistoricalKvStepStatus::Yielded) {
        round_.blocks.clear();
        round_active_ = false;
        collect_cursor_ = 0;
    }
}

HistoricalKvGc::Stats HistoricalKvGc::stats() const {
    return {ticks_.load(), proposed_.load(), committed_.load(), shadow_.load(),
            reclaimed_.load(), deferred_.load(), stale_.load(),
            reevaluate_.load(), retry_count_.load(), policy_failures_.load(),
            worker_failures_.load()};
}

} // namespace kvmem
