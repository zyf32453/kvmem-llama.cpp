#include "kvmem/historical_kv_gc.hpp"
#include "kvmem/historical_kv_page_features.hpp"
#include "kvmem/historical_kv_page_gc.hpp"
#include <atomic>
#include <iostream>
#include <stdexcept>
#include <type_traits>

using namespace kvmem;
static void check(bool value, const char * message) { if (!value) throw std::runtime_error(message); }

namespace kvmem {
struct HistoricalKvGcAuditTestAccess {
    // Inject a failed CAS result without making a retirement, to exercise the
    // reliable Retry transport deterministically rather than depend on timing.
    static void failed_attempt(HistoricalKvGc & gc) {
        check(gc.pending_state_.load() == HistoricalKvGc::PendingState::Prepared, "no prepared retry fixture");
        for (auto & result : gc.pending_codes_) result = HistoricalKvCommitCode::Retry;
        gc.pending_state_.store(HistoricalKvGc::PendingState::Committed, std::memory_order_release);
    }
};
struct HistoricalKvThinkingRangesTestAccess {
    static void saturate(HistoricalKvThinkingRanges & ranges) {
        std::lock_guard<std::mutex> lock(ranges.mu_);
        ranges.classification_revision_.store(UINT64_MAX, std::memory_order_release);
    }
};
}

static HistoricalKvGc::Config config(unsigned capacity) {
    HistoricalKvGc::Config cfg;
    cfg.mode = HistoricalKvGc::Mode::Delete;
    cfg.retire_at_decode_boundary = true;
    cfg.scan_blocks = cfg.reclaim_blocks = capacity;
    cfg.batch_blocks = 1;
    return cfg;
}

static HistoricalKvHandle append(const std::shared_ptr<HistoricalKvRepository> & repository, uint64_t id) {
    auto writer = repository->begin_write(id, id * 64, 64, id + 1, 16,
        [](const HistoricalKvHandle &) { return true; });
    check(bool(writer) && repository->publish_write(*writer), "fixture writer");
    return writer->handle();
}

class PreparedPolicy final : public HistoricalKvPolicy {
public:
    mutable unsigned prepared = 0, permits_called = 0;
    std::atomic<uint64_t> epoch{1};
    bool permits(const HistoricalKvBlockMeta &) const noexcept override {
        ++permits_called; // Poison: boundary-mode commits must not call this.
        return false;
    }
    HistoricalKvPrepareCode prepare_commit(const HistoricalKvBlockMeta &,
        HistoricalKvCommitConditions & out) const noexcept override {
        ++prepared;
        out = {};
        out.live_only = out.sealed_only = true;
        return out.add_epoch({&epoch, epoch.load()}) ? HistoricalKvPrepareCode::Ready
                                                   : HistoricalKvPrepareCode::Reevaluate;
    }
    HistoricalKvStepResult step(const HistoricalKvSnapshot & snapshot, const HistoricalKvPressure &,
        HistoricalKvWorkBudget, HistoricalKvDecisionBuffer & out) override {
        for (const auto & block : snapshot.blocks)
            if (block.life == HistoricalKvLife::Live &&
                out.push({block.handle, snapshot.round_id, HistoricalKvDeleteReason::Custom, block.content_version}))
                return {HistoricalKvStepStatus::Complete, 1, 1};
        return {HistoricalKvStepStatus::NoCandidate, 0, 0};
    }
};

class UnknownPolicy final : public HistoricalKvPolicy {
public:
    mutable unsigned calls = 0;
    bool permits(const HistoricalKvBlockMeta &) const noexcept override { ++calls; return true; }
    HistoricalKvStepResult step(const HistoricalKvSnapshot & snapshot, const HistoricalKvPressure &,
        HistoricalKvWorkBudget, HistoricalKvDecisionBuffer & out) override {
        for (const auto & block : snapshot.blocks)
            if (block.life == HistoricalKvLife::Live &&
                out.push({block.handle, snapshot.round_id, HistoricalKvDeleteReason::Custom, {}}))
                return {HistoricalKvStepStatus::Complete, 1, 1};
        return {HistoricalKvStepStatus::NoCandidate, 0, 0};
    }
};

static void no_policy_callback_and_retry_keeps_original_conditions() {
    auto repository = std::make_shared<HistoricalKvRepository>(1);
    const auto handle = append(repository, 0);
    auto policy = std::make_unique<PreparedPolicy>();
    auto & observed = *policy;
    HistoricalKvGc gc(repository, std::move(policy), config(1));
    gc.manual_tick();
    check(observed.prepared == 1 && observed.permits_called == 0, "preparation called arbitrary predicate");
    HistoricalKvGcAuditTestAccess::failed_attempt(gc);
    observed.epoch.fetch_add(1);
    gc.manual_tick();
    check(observed.prepared == 1, "retry was reauthorized against a new epoch");
    check(gc.commit_at_decode_boundary().committed == 0, "stale retry committed");
    gc.manual_tick();
    check(gc.stats().retry == 1 && gc.stats().reevaluate == 1 &&
          repository->completion_state(handle) == HistoricalKvRepository::CompletionState::Live,
          "retry conditions/result lost");
    // A newly scored round can use the new epoch and still commit correctly.
    check(gc.commit_at_decode_boundary().committed == 1 && observed.permits_called == 0,
          "compiled commit did not replace policy callback");
    gc.manual_tick();
    check(gc.stats().committed == 1 && gc.stats().reclaimed == 1, "compiled commit not released");
}

static void unsupported_policy_fails_closed_and_direct_mode_still_works() {
    for (bool boundary : {true, false}) {
        auto repository = std::make_shared<HistoricalKvRepository>(1);
        const auto handle = append(repository, 0);
        auto policy = std::make_unique<UnknownPolicy>();
        auto & observed = *policy;
        auto cfg = config(1); cfg.retire_at_decode_boundary = boundary;
        HistoricalKvGc gc(repository, std::move(policy), cfg);
        gc.manual_tick();
        if (boundary) {
            check(gc.commit_at_decode_boundary().prepared == 0 && observed.calls == 0,
                  "unsupported policy ran on foreground");
            check(gc.stats().unsupported_conditions == 1 && gc.stats().policy_failures == 1 &&
                  repository->completion_state(handle) == HistoricalKvRepository::CompletionState::Live,
                  "unsupported policy not explicit/fail-closed");
            gc.manual_tick();
            check(gc.stats().proposed == 1, "disabled policy kept producing");
        } else {
            check(observed.calls > 0 && gc.stats().committed == 1,
                  "worker-only custom policy compatibility changed");
            gc.manual_tick();
            check(gc.stats().reclaimed == 1, "worker-only release failed");
        }
    }
}

static HistoricalKvBlockMeta meta(uint64_t begin, uint32_t tokens = 64) {
    HistoricalKvBlockMeta block;
    block.life = HistoricalKvLife::Live; block.sealed = true;
    block.original_begin = begin; block.valid_tokens = tokens;
    return block;
}

static void range_certificates_revoke_edits_without_append_starvation() {
    HistoricalKvThinkingRanges ranges;
    ranges.note(0, 128); ranges.arm();
    HistoricalKvCommitConditions prefix;
    check(ranges.prepare_commit(0, 64, 1u, prefix), "thinking prefix certificate");
    for (unsigned token = 129; token < 1024; ++token) {
        ranges.note(0, token); ranges.arm();
        check(prefix.matches(meta(0)), "normal append starved prefix retirement");
    }
    ranges.truncate(32);
    check(!prefix.matches(meta(0)), "truncated certificate stayed valid");
    ranges.clear_spans(); ranges.note(0, 64); ranges.arm();
    HistoricalKvCommitConditions ahead;
    check(ranges.prepare_commit(128, 64, 1u << 2, ahead), "future unclassified certificate");
    ranges.note(128, 192);
    check(!ahead.matches(meta(128)), "future span reclassification ignored");
    HistoricalKvCommitConditions gap;
    check(ranges.prepare_commit(64, 64, 1u << 2, gap), "old gap certificate");
    ranges.note(64, 128);
    check(!gap.matches(meta(64)), "historical overlap reclassification ignored");
    HistoricalKvCommitConditions cleared;
    check(ranges.prepare_commit(0, 64, 1u, cleared), "clear fixture");
    ranges.clear_spans();
    check(!cleared.matches(meta(0)), "clear did not revoke certificate");
    ranges.note(0, 64);
    HistoricalKvThinkingRangesTestAccess::saturate(ranges);
    HistoricalKvCommitConditions exhausted;
    check(!ranges.prepare_commit(0, 64, 1u, exhausted), "saturated revision authorized commit");

    HistoricalKvThinkingRanges replies;
    replies.enable_completed_replies(true);
    replies.note_completed_reply(0, 64);
    replies.note_completed_reply(64, 128);
    HistoricalKvCommitConditions older;
    check(replies.prepare_commit(0, 64, 1u << 1, older), "older reply certificate");
    replies.enable_completed_replies(false);
    check(!older.matches(meta(0)), "reply protection change ignored");
}

static void native_protection_and_question_epoch_are_rechecked() {
    HistoricalKvPageGc::Config cfg;
    cfg.slots = cfg.logical_pages = 3; cfg.page_tokens = 64;
    cfg.sink_tokens = cfg.recent_tokens = 64;
    cfg.low_ready = 1; cfg.high_ready = 2; cfg.batch_blocks = 1;
    {
        HistoricalKvPageGc bridge(cfg);
        for (unsigned page = 0; page < 3; ++page) check(bridge.register_page(page, 16, page + 1), "register LRU page");
        bridge.protect_page(1, false);
        for (int i = 0; i < 20 && !bridge.stats().proposed; ++i) bridge.manual_tick();
        check(bridge.stats().proposed == 1, "no native LRU proposal");
        bridge.protect_page(1, true);
        check(bridge.commit_boundary().empty() && bridge.is_live(1), "late native protection ignored");
        bridge.manual_tick();
        check(bridge.stats().reevaluate == 1, "late protection result lost");
        bridge.protect_page(1, false);
        for (int i = 0; i < 20 && bridge.stats().proposed < 2; ++i) bridge.manual_tick();
        check(bridge.commit_boundary().size() == 1, "protection did not yield after use");
    }
    auto source = std::make_shared<HistoricalKvPageFeatures>(3, 3);
    for (unsigned page = 0; page < 3; ++page) source->publish_page(page, {1, 0, 0});
    source->publish_question_scores({0, 0, 0});
    HistoricalKvRedundancyPolicy::Config policy;
    HistoricalKvPageGc bridge(cfg, std::make_unique<HistoricalKvRedundancyPolicy>(3, source, policy));
    for (unsigned page = 0; page < 3; ++page) check(bridge.register_page(page, 16, page + 1), "register scored page");
    bridge.protect_page(1, false);
    for (int i = 0; i < 30 && !bridge.stats().proposed; ++i) bridge.manual_tick();
    check(bridge.stats().proposed == 1, "no native redundancy proposal");
    source->clear_question();
    check(bridge.commit_boundary().empty() && bridge.is_live(1), "new Query accepted old decision");
    bridge.manual_tick();
    check(bridge.stats().reevaluate == 1, "stale Query result lost");
    // Partial scores belong to the same question and must not invalidate a
    // newly prepared decision; this is required by long-prefill admission.
    for (int i = 0; i < 30 && bridge.stats().proposed < 2; ++i) bridge.manual_tick();
    source->publish_question_scores({0, .2, 0}, false);
    check(bridge.commit_boundary().size() == 1, "partial Query update starved same-epoch decision");
}

int main() {
    try {
        static_assert(std::is_trivially_copyable<HistoricalKvCommitConditions>::value, "no foreground ownership destructor");
        no_policy_callback_and_retry_keeps_original_conditions();
        unsupported_policy_fails_closed_and_direct_mode_still_works();
        range_certificates_revoke_edits_without_append_starvation();
        native_protection_and_question_epoch_are_rechecked();
        std::cout << "PASS typed conditions: no policy callback, retry identity, fail-closed, range growth/edit, native protection and Query epoch\n";
        std::cout << "conditions_bytes=" << sizeof(HistoricalKvCommitConditions) << " max_epoch_guards="
                  << HistoricalKvCommitConditions::kEpochGuards << '\n';
    } catch (const std::exception & error) {
        std::cerr << "FAIL typed commit conditions: " << error.what() << '\n';
        return 1;
    }
}
