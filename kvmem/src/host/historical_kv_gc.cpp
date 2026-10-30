#include "kvmem/historical_kv_gc.hpp"
#include "kvmem/historical_kv_audit.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <utility>

namespace kvmem {
namespace {
HistoricalKvGc::Config effective_config(const std::shared_ptr<HistoricalKvRepository> & repository,
                                       HistoricalKvGc::Config config) {
    const auto account = repository ? repository->auxiliary_budget() : nullptr;
    if (account && config.auxiliary && account != config.auxiliary)
        throw std::invalid_argument("historical GC repository budget mismatch");
    if (!config.auxiliary) config.auxiliary = account;
    return config;
}
bool same(HistoricalKvHandle a, HistoricalKvHandle b) noexcept {
    return a.incarnation == b.incarnation && a.slot == b.slot && a.generation == b.generation;
}

bool valid_evidence(const HistoricalKvDeleteIntent & intent,
                    const HistoricalKvCommitConditions * conditions = nullptr) noexcept {
    if (!intent.evidence)
        return intent.reason != HistoricalKvDeleteReason::Redundancy &&
               (!conditions || !conditions->requires_score_evidence);
    const auto & proof = *intent.evidence;
    if (!intent.expected_content_version || !*intent.expected_content_version ||
        !proof.feature_schema || !proof.scoring_round || proof.scoring_round != intent.round_id ||
        (proof.feature_score_used && !proof.score_epoch) ||
        !std::isfinite(proof.importance) || !std::isfinite(proof.retention_score)) return false;
    if (intent.reason == HistoricalKvDeleteReason::Redundancy && !proof.witness) return false;
    if (proof.witness) {
        const auto & witness = *proof.witness;
        if (!proof.feature_score_used || !witness.content_version || !witness.handle.generation ||
            witness.handle.incarnation != intent.victim.incarnation ||
            witness.handle.slot == intent.victim.slot ||
            !std::isfinite(proof.similarity) || !std::isfinite(proof.threshold) ||
            proof.threshold < 0 || proof.threshold > 1 ||
            proof.similarity < proof.threshold || proof.similarity > 1) return false;
    }
    return !conditions || (conditions->requires_score_evidence &&
        conditions->evidence_schema == proof.feature_schema &&
        conditions->evidence_query_epoch == proof.query_epoch &&
        conditions->evidence_score_epoch == proof.score_epoch &&
        conditions->evidence_round == proof.scoring_round);
}
} // namespace

HistoricalKvGc::HistoricalKvGc(std::shared_ptr<HistoricalKvRepository> repository,
                               std::unique_ptr<HistoricalKvPolicy> policy,
                               Config config)
    : HistoricalKvGc(std::move(repository), historical_kv_aux_adopt(std::move(policy)), config) {}

HistoricalKvGc::HistoricalKvGc(std::shared_ptr<HistoricalKvRepository> repository,
                               HistoricalKvAuxOwner<HistoricalKvPolicy> policy,
                               Config config)
    : repository_(std::move(repository)), policy_(std::move(policy)), config_(effective_config(repository_, config)),
      prepared_(HistoricalKvAuxAllocator<HistoricalKvDeleteIntent>(config_.auxiliary, HistoricalKvAuxKind::Decisions)),
      prepared_conditions_(HistoricalKvAuxAllocator<HistoricalKvCommitConditions>(config_.auxiliary, HistoricalKvAuxKind::Decisions)),
      pending_intents_(prepared_.get_allocator()), pending_conditions_(prepared_conditions_.get_allocator()),
      pending_codes_(HistoricalKvAuxAllocator<HistoricalKvCommitCode>(config_.auxiliary, HistoricalKvAuxKind::Decisions)),
      acknowledged_intents_(prepared_.get_allocator()), acknowledged_conditions_(prepared_conditions_.get_allocator()),
      acknowledged_codes_(pending_codes_.get_allocator()), audit_committed_(prepared_.get_allocator()),
      decisions_(repository_ ? std::min(config.batch_blocks, repository_->capacity()) : 0, config_.auxiliary),
      retry_(prepared_.get_allocator()), retry_conditions_(prepared_conditions_.get_allocator()),
      seen_generation_(HistoricalKvAuxAllocator<uint64_t>(config_.auxiliary, HistoricalKvAuxKind::Metadata)),
      shadow_generation_(seen_generation_.get_allocator()) {
    if (!repository_ || !policy_ || !repository_->capacity() ||
        ((config_.ready_low || config_.ready_high) &&
         (!config_.ready_low || !(config_.ready_low < config_.ready_high) ||
          config_.ready_high > repository_->capacity())) ||
        !config_.scan_blocks || !config_.batch_blocks ||
        config_.batch_blocks > repository_->capacity() ||
        !config_.reclaim_blocks || !config_.poll_ms) {
        throw std::invalid_argument("invalid historical KV GC configuration");
    }
    if (config_.auxiliary && (policy_->auxiliary_budget() != config_.auxiliary ||
        policy_.get_deleter().budget != config_.auxiliary))
        throw std::invalid_argument("historical GC policy budget mismatch");
    // A retry blocks new proposals; at most one configured batch is retained.
    retry_.reserve(config_.batch_blocks);
    retry_conditions_.reserve(config_.batch_blocks);
    prepared_.reserve(config_.batch_blocks);
    prepared_conditions_.reserve(config_.batch_blocks);
    pending_intents_.reserve(config_.batch_blocks);
    pending_conditions_.reserve(config_.batch_blocks);
    pending_codes_.reserve(config_.batch_blocks);
    acknowledged_intents_.reserve(config_.batch_blocks);
    acknowledged_conditions_.reserve(config_.batch_blocks);
    acknowledged_codes_.reserve(config_.batch_blocks);
    audit_ = historical_kv_audit_enabled();
    if (audit_) audit_committed_.resize(repository_->capacity());
    round_.blocks = HistoricalKvAuxVector<HistoricalKvBlockMeta>(
        HistoricalKvAuxAllocator<HistoricalKvBlockMeta>(config_.auxiliary, HistoricalKvAuxKind::Metadata));
    observation_.blocks = HistoricalKvAuxVector<HistoricalKvBlockMeta>(round_.blocks.get_allocator());
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

HistoricalKvCommitCode HistoricalKvGc::commit(const HistoricalKvDeleteIntent & intent,
                                             const HistoricalKvCommitConditions * conditions) {
    HistoricalKvCommitConditions direct_conditions;
    if (!conditions && intent.evidence) {
        // Worker-only mode also binds scored decisions to their exact source
        // context; old unscored custom predicates keep their worker path.
        const auto meta = repository_->inspect(intent.victim);
        if (!meta) return HistoricalKvCommitCode::Stale;
        if (policy_->prepare_commit(*meta, direct_conditions) != HistoricalKvPrepareCode::Ready)
            return HistoricalKvCommitCode::Reevaluate;
        conditions = &direct_conditions;
    }
    if (!valid_evidence(intent, conditions)) {
        ++invalid_evidence_;
        return HistoricalKvCommitCode::Reevaluate;
    }
    HistoricalKvRepository::RetirePredicate predicate;
    const void * context;
    if (conditions) {
        predicate = [](const HistoricalKvBlockMeta & meta, const void * prepared) noexcept {
            return static_cast<const HistoricalKvCommitConditions *>(prepared)->matches(meta);
        };
        context = conditions;
    } else {
        predicate = [](const HistoricalKvBlockMeta & meta, const void * policy) noexcept {
            return static_cast<const HistoricalKvPolicy *>(policy)->permits(meta);
        };
        context = policy_.get();
    }
    const auto * witness = intent.evidence && intent.evidence->witness ? &*intent.evidence->witness : nullptr;
    const auto code = witness
        ? repository_->retire_with_witness(intent.victim, intent.expected_content_version, *witness, predicate, context)
        : repository_->retire(intent.victim, intent.expected_content_version, predicate, context);
    switch (code) {
    case HistoricalKvRepository::RetireResult::Committed:
        return HistoricalKvCommitCode::Committed;
    case HistoricalKvRepository::RetireResult::AlreadyRetired:
        return HistoricalKvCommitCode::AlreadyRetired;
    case HistoricalKvRepository::RetireResult::Stale:
        return HistoricalKvCommitCode::Stale;
    case HistoricalKvRepository::RetireResult::VersionChanged:
    case HistoricalKvRepository::RetireResult::NotPermitted:
        return HistoricalKvCommitCode::Reevaluate;
    case HistoricalKvRepository::RetireResult::WitnessChanged:
        ++witness_rejected_;
        return HistoricalKvCommitCode::Reevaluate;
    case HistoricalKvRepository::RetireResult::Retry:
        return HistoricalKvCommitCode::Retry;
    }
    return HistoricalKvCommitCode::Retry;
}

void HistoricalKvGc::request_capacity(uint32_t ready) {
    if (ready > repository_->capacity())
        throw std::invalid_argument("historical GC capacity request exceeds repository");
    requested_ready_.store(ready, std::memory_order_release);
}

void HistoricalKvGc::acknowledge(const HistoricalKvDeleteIntent & intent,
                                 HistoricalKvCommitCode code) {
    policy_->on_result({intent, code});
    if (audit_) {
        if (code == HistoricalKvCommitCode::Committed && intent.victim.slot < audit_committed_.size())
            audit_committed_[intent.victim.slot] = intent;
        audit_acknowledgement(intent, historical_kv_audit_outcome(code));
    }
    switch (code) {
    case HistoricalKvCommitCode::Committed: ++committed_; break;
    case HistoricalKvCommitCode::AlreadyRetired: break;
    case HistoricalKvCommitCode::Stale: ++stale_; break;
    case HistoricalKvCommitCode::Reevaluate: ++reevaluate_; break;
    case HistoricalKvCommitCode::Retry: ++retry_count_; break;
    }
}

void HistoricalKvGc::audit_acknowledgement(const HistoricalKvDeleteIntent & intent,
                                          const char * outcome) const noexcept {
    if (!audit_) return;
    // Use the intent's original round. A terminal policy step can already have
    // cleared round_.blocks before a decode boundary commits this batch.
    const HistoricalKvScoreEvidence empty{};
    const auto & proof = intent.evidence ? *intent.evidence : empty;
    const HistoricalKvWitnessRequirement absent{};
    const auto & witness = proof.witness ? *proof.witness : absent;
    std::fprintf(stderr, "KVMEM_GC_AUDIT event=ack incarnation=%llu slot=%u generation=%llu content_version=%llu has_content_version=%d round=%llu intent_reason=%s outcome=%s mode=%s evidence=%d feature_schema=%llu query_epoch=%llu score_epoch=%llu scoring_round=%llu feature_score_used=%d importance=%.17g retention_score=%.17g witness=%d witness_incarnation=%llu witness_slot=%u witness_generation=%llu witness_content_version=%llu similarity=%.17g threshold=%.17g\n",
        (unsigned long long)intent.victim.incarnation, intent.victim.slot,
        (unsigned long long)intent.victim.generation,
        (unsigned long long)intent.expected_content_version.value_or(0),
        int(intent.expected_content_version.has_value()), (unsigned long long)intent.round_id,
        historical_kv_audit_reason(intent.reason), outcome,
        config_.mode == Mode::Delete ? "delete" : config_.mode == Mode::Shadow ? "shadow" : "off",
        int(intent.evidence.has_value()), (unsigned long long)proof.feature_schema,
        (unsigned long long)proof.query_epoch, (unsigned long long)proof.score_epoch,
        (unsigned long long)proof.scoring_round, int(proof.feature_score_used),
        proof.importance, proof.retention_score, int(proof.witness.has_value()),
        (unsigned long long)witness.handle.incarnation, witness.handle.slot,
        (unsigned long long)witness.handle.generation, (unsigned long long)witness.content_version,
        proof.similarity, proof.threshold);
}

void HistoricalKvGc::audit_reclaim(const HistoricalKvBlockMeta & block,
                                  HistoricalKvRepository::ReclaimResult result) const noexcept {
    if (!audit_) return;
    using Result = HistoricalKvRepository::ReclaimResult;
    // Waiting/retrying a retained GPU/Host reference is normal. Log physical
    // progress, not the same non-progressing record on every worker poll.
    if (result == Result::Deferred || result == Result::Stale || result == Result::Retry) return;
    const auto & intent = audit_committed_[block.handle.slot];
    const bool known = intent.victim.incarnation == block.handle.incarnation &&
        intent.victim.generation == block.handle.generation && intent.victim.slot == block.handle.slot;
    const char * outcome = result == Result::Complete ? "ready" :
        result == Result::HostReleased ? "host_released" :
        result == Result::ResourceReleased ? "resource_released" : "retry";
    const bool host_released = block.host_released || result == Result::Complete || result == Result::HostReleased;
    std::fprintf(stderr, "KVMEM_GC_AUDIT event=reclaim incarnation=%llu slot=%u generation=%llu content_version=%llu round=%llu intent_known=%d original_begin=%llu valid_tokens=%u outcome=%s host_released=%d ready=%d\n",
        (unsigned long long)block.handle.incarnation, block.handle.slot,
        (unsigned long long)block.handle.generation, (unsigned long long)block.content_version,
        (unsigned long long)(known ? intent.round_id : 0), int(known),
        (unsigned long long)block.original_begin, block.valid_tokens, outcome,
        int(host_released), int(result == Result::Complete));
}

HistoricalKvGc::BoundaryResult HistoricalKvGc::commit_at_decode_boundary() noexcept {
    BoundaryResult result;
    if (!config_.retire_at_decode_boundary || config_.mode != Mode::Delete) return result;
    auto prepared = PendingState::Prepared;
    // One attempt: skip an unpublished, in-use, or unacknowledged batch. The
    // acquire pairs with the worker's Prepared publication; no spinning or
    // waiting for the worker, and no vector mutation on this path.
    if (!pending_state_.compare_exchange_strong(prepared, PendingState::Committing,
            std::memory_order_acquire, std::memory_order_relaxed)) return result;
    const auto ns = [] {
        return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
    };
    result.started_ns = ns();
    result.prepared = static_cast<uint32_t>(pending_intents_.size());
    for (size_t i = 0; i < pending_intents_.size(); ++i) {
        const auto code = commit(pending_intents_[i], &pending_conditions_[i]);
        pending_codes_[i] = code;
        if (code == HistoricalKvCommitCode::Committed) {
            ++result.committed;
            if (config_.on_boundary_retired)
                config_.on_boundary_retired(config_.boundary_context, pending_intents_[i].victim);
        }
    }
    result.finished_ns = ns();
    ++boundary_batches_;
    boundary_committed_.fetch_add(result.committed, std::memory_order_relaxed);
    // Publish result codes only after every adapter tombstone notification.
    // The worker cannot reuse these buffers while the boundary is in flight.
    pending_state_.store(PendingState::Committed, std::memory_order_release);
    return result;
}

void HistoricalKvGc::manual_tick() {
    std::lock_guard<std::mutex> guard(tick_mu_);
    ++ticks_;
    if (config_.retire_at_decode_boundary) {
        if (pending_state_.load(std::memory_order_acquire) == PendingState::Committed) {
            acknowledged_intents_.swap(pending_intents_);
            acknowledged_conditions_.swap(pending_conditions_);
            acknowledged_codes_.swap(pending_codes_);
            pending_state_.store(PendingState::Idle, std::memory_order_release);
        }
        // The worker owns these buffers even after the mailbox becomes Idle.
        // Slow policy callbacks/audit output cannot hold a foreground lock.
        for (size_t i = 0; i < acknowledged_intents_.size(); ++i) {
            acknowledge(acknowledged_intents_[i], acknowledged_codes_[i]);
            if (acknowledged_codes_[i] == HistoricalKvCommitCode::Retry) {
                retry_.push_back(acknowledged_intents_[i]);
                retry_conditions_.push_back(acknowledged_conditions_[i]);
            }
        }
        acknowledged_intents_.clear();
        acknowledged_conditions_.clear();
        acknowledged_codes_.clear();
    }
    // Progress previous retirements even when policy has no candidate, the
    // output buffer is full, or new policy decisions are switched off.
    const uint32_t capacity = repository_->capacity();
    const uint32_t reclaim_scan = std::min(config_.reclaim_blocks,
                                           capacity - reclaim_cursor_);
    repository_->observe_range_into(reclaim_cursor_, reclaim_scan, observation_);
    for (const auto & block : observation_.blocks) {
            if (block.life != HistoricalKvLife::Retired &&
                block.life != HistoricalKvLife::Releasing) continue;
            const auto result = repository_->try_reclaim(block.handle);
            audit_reclaim(block, result);
            switch (result) {
            case HistoricalKvRepository::ReclaimResult::Complete: ++reclaimed_; break;
            case HistoricalKvRepository::ReclaimResult::HostReleased: ++host_released_early_; break;
            case HistoricalKvRepository::ReclaimResult::ResourceReleased: ++partial_resource_releases_; break;
            case HistoricalKvRepository::ReclaimResult::Deferred: ++deferred_; break;
            case HistoricalKvRepository::ReclaimResult::Stale: break;
            case HistoricalKvRepository::ReclaimResult::Retry: ++retry_count_; break;
            }
    }
    reclaim_cursor_ = reclaim_scan == capacity - reclaim_cursor_ ? 0 :
                      reclaim_cursor_ + reclaim_scan;
    if (config_.mode == Mode::Off) return;
    if (config_.retire_at_decode_boundary) {
        if (pending_state_.load(std::memory_order_acquire) != PendingState::Idle) return;
    }

    // Retry retained commands before taking a fresh snapshot. A previous
    // failure must not cause the next round to over-delete new candidates.
    if (!retry_.empty()) {
        if (config_.retire_at_decode_boundary) {
            pending_intents_.swap(retry_);
            pending_conditions_.swap(retry_conditions_);
            pending_codes_.resize(pending_intents_.size());
            pending_state_.store(PendingState::Prepared, std::memory_order_release);
            return;
        }
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
        pressure_.requested_ready = requested_ready_.load(std::memory_order_acquire);
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
    // Release/admission may have changed capacity during a sliced snapshot.
    // Refresh pressure without allocating or reading mutable payload objects.
    const auto capacity_now = repository_->observe_capacity();
    pressure_.ready = capacity_now.ready;
    pressure_.pending_slot_credit = capacity_now.pending;
    if (config_.mode == Mode::Shadow) {
        for (const auto & block : round_.blocks)
            if (shadow_generation_[block.handle.slot] == block.handle.generation &&
                repository_->completion_state(block.handle) == HistoricalKvRepository::CompletionState::Live)
                ++pressure_.pending_slot_credit;
    }
    pressure_.requested_ready = requested_ready_.load(std::memory_order_acquire);
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
        bool evidence_ok = valid_evidence(intent);
        if (evidence_ok && intent.evidence && intent.evidence->witness) {
            const auto & witness = *intent.evidence->witness;
            const auto found = std::lower_bound(round_.blocks.begin(), round_.blocks.end(), witness.handle.slot,
                [](const HistoricalKvBlockMeta & meta, uint32_t slot) { return meta.handle.slot < slot; });
            evidence_ok = found != round_.blocks.end() && same(found->handle, witness.handle) &&
                found->life == HistoricalKvLife::Live && found->sealed &&
                found->content_version == witness.content_version &&
                std::none_of(decisions_.values().begin(), decisions_.values().end(),
                    [&](const auto & other) { return same(other.victim, witness.handle); });
        }
        if (!evidence_ok) {
            ++invalid_evidence_;
            acknowledge(intent, HistoricalKvCommitCode::Reevaluate);
            continue;
        }
        if (seen_generation_[intent.victim.slot] == intent.victim.generation) {
            if (config_.mode != Mode::Shadow)
                acknowledge(intent, HistoricalKvCommitCode::AlreadyRetired);
            else audit_acknowledgement(intent, "shadow_duplicate");
            continue;
        }
        seen_generation_[intent.victim.slot] = intent.victim.generation;
        if (config_.mode == Mode::Shadow) {
            const auto current = repository_->inspect(intent.victim);
            HistoricalKvCommitConditions simulated;
            const bool score_valid = !intent.evidence || (current &&
                policy_->prepare_commit(*current, simulated) == HistoricalKvPrepareCode::Ready &&
                valid_evidence(intent, &simulated) && simulated.matches(*current));
            if (!current || !policy_->permits(*current) || !score_valid) {
                audit_acknowledgement(intent, "shadow_skipped");
                continue;
            }
            shadow_generation_[intent.victim.slot] = intent.victim.generation;
            ++shadow_;
            audit_acknowledgement(intent, "shadow");
            continue;
        }
        if (config_.retire_at_decode_boundary) {
            HistoricalKvCommitConditions conditions;
            const auto prepared = policy_->prepare_commit(*observed_slot, conditions);
            if (prepared != HistoricalKvPrepareCode::Ready) {
                if (prepared == HistoricalKvPrepareCode::Unsupported) {
                    ++unsupported_conditions_;
                    if (!policy_failed_) {
                        ++policy_failures_;
                        policy_failed_ = true;
                        std::fprintf(stderr, "[historical-kv-gc] boundary conditions unsupported; policy disabled\n");
                    }
                }
                acknowledge(intent, HistoricalKvCommitCode::Reevaluate);
                continue;
            }
            if (!valid_evidence(intent, &conditions)) {
                ++invalid_evidence_;
                acknowledge(intent, HistoricalKvCommitCode::Reevaluate);
                continue;
            }
            prepared_.push_back(intent);
            prepared_conditions_.push_back(conditions);
        } else {
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
    }
    if (!prepared_.empty()) {
        pending_intents_.swap(prepared_);
        pending_conditions_.swap(prepared_conditions_);
        pending_codes_.resize(pending_intents_.size());
        pending_state_.store(PendingState::Prepared, std::memory_order_release);
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
            worker_failures_.load(), host_released_early_.load(), partial_resource_releases_.load(),
            boundary_batches_.load(), boundary_committed_.load(), unsupported_conditions_.load(),
            invalid_evidence_.load(), witness_rejected_.load()};
}

} // namespace kvmem
