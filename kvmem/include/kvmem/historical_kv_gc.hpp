#pragma once

#include "kvmem/historical_kv_policy.hpp"
#include "kvmem/historical_kv_repository.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace kvmem {

// The sole owner of delete decisions for one repository. The policy proposes
// exact-generation intents; this manager commits retirement and asks the
// repository to release Host resources after their dependencies have drained.
// Full slot reuse additionally waits for GPU-only and metadata references.
// This module is host-only. The adapter currently owns it only for explicitly
// enabled finite recency/RAM requests in a single-lane capture domain.
class HistoricalKvGc {
public:
    enum class Mode { Off, Shadow, Delete };
    struct Config {
        Mode mode = Mode::Off;
        // 0/0 is valid for a strategy that does not use capacity pressure.
        // The default LRU itself requires 0 < low < high <= capacity.
        uint32_t ready_low = 0;
        uint32_t ready_high = 0;
        uint32_t scan_blocks = 256;
        uint32_t batch_blocks = 64;
        uint32_t reclaim_blocks = 64;
        uint32_t poll_ms = 25;
        // Prepare policy decisions on the worker; retire exact generations in
        // the next pre/post-decode gap. Resource release stays on the worker.
        bool retire_at_decode_boundary = false;
        // Optional allocation-free notification on the serialized decode
        // boundary. The adapter may publish a tombstone for this exact handle;
        // it must not release Host resources or block in this callback.
        void (*on_boundary_retired)(void *, HistoricalKvHandle) noexcept = nullptr;
        void * boundary_context = nullptr;
    };
    struct Stats {
        uint64_t ticks = 0;
        uint64_t proposed = 0;
        uint64_t committed = 0;
        uint64_t shadow = 0;
        uint64_t reclaimed = 0;
        uint64_t deferred = 0;
        uint64_t stale = 0;
        uint64_t reevaluate = 0;
        uint64_t retry = 0;
        uint64_t policy_failures = 0;
        uint64_t worker_failures = 0;
        uint64_t host_released_early = 0; // Host bundle gone, identity/GPU still held.
        uint64_t partial_resource_releases = 0; // Progress before the complete Host bundle is gone.
        uint64_t boundary_batches = 0;
        uint64_t boundary_committed = 0;
    };
    struct BoundaryResult {
        uint64_t started_ns = 0;
        uint64_t finished_ns = 0;
        uint32_t prepared = 0;
        uint32_t committed = 0;
    };

    HistoricalKvGc(std::shared_ptr<HistoricalKvRepository> repository,
                   std::unique_ptr<HistoricalKvPolicy> policy, Config config);
    ~HistoricalKvGc();
    HistoricalKvGc(const HistoricalKvGc &) = delete;
    HistoricalKvGc & operator=(const HistoricalKvGc &) = delete;

    // start() is explicit so tests can deterministically drive manual_tick().
    void start();
    void request_stop() noexcept { stopping_.store(true, std::memory_order_release); }
    bool worker_finished() const noexcept { return worker_finished_.load(std::memory_order_acquire); }
    void stop() noexcept;
    void manual_tick();
    // Called after the public decode's host work returns, or immediately
    // before a new active selection. The GPU may still be in flight; existing
    // execution leases remain valid. Never calls a release callback.
    BoundaryResult commit_at_decode_boundary() noexcept;
    // Set by the serialized request owner only while no request is executing.
    // Zero clears the demand. This never reserves or waits for pending bytes.
    void request_capacity(uint32_t ready);
    Stats stats() const;
    HistoricalKvPolicy::Metrics policy_metrics() const noexcept { return policy_->metrics(); }

private:
    friend struct HistoricalKvGcAuditTestAccess;
    HistoricalKvCommitCode commit(const HistoricalKvDeleteIntent & intent);
    void acknowledge(const HistoricalKvDeleteIntent & intent,
                     HistoricalKvCommitCode code);
    void audit_acknowledgement(const HistoricalKvDeleteIntent & intent, const char * outcome) const noexcept;
    void audit_reclaim(const HistoricalKvBlockMeta & block,
                       HistoricalKvRepository::ReclaimResult result) const noexcept;
    void run() noexcept;

    std::shared_ptr<HistoricalKvRepository> repository_;
    std::unique_ptr<HistoricalKvPolicy> policy_;
    Config config_;
    mutable std::mutex tick_mu_;
    mutable std::mutex pending_mu_;
    enum class PendingState { Idle, Prepared, Committed };
    PendingState pending_state_ = PendingState::Idle;
    std::atomic<bool> pending_ready_{false};
    std::vector<HistoricalKvDeleteIntent> prepared_;
    std::vector<HistoricalKvDeleteIntent> pending_intents_;
    std::vector<HistoricalKvCommitCode> pending_codes_;
    // Worker-owned handoff buffers: acknowledgement (including diagnostic IO)
    // runs after releasing pending_mu_, which the decode boundary also needs.
    std::vector<HistoricalKvDeleteIntent> acknowledged_intents_;
    std::vector<HistoricalKvCommitCode> acknowledged_codes_;
    bool audit_ = false;
    // Allocated only for opt-in auditing. Exact generations prevent associating
    // a reused slot's release with an earlier proposal/round.
    std::vector<HistoricalKvDeleteIntent> audit_committed_;
    HistoricalKvSnapshot round_;
    bool round_active_ = false;
    HistoricalKvRepository::RangeObservation observation_;
    HistoricalKvDecisionBuffer decisions_;
    HistoricalKvPressure pressure_{};
    std::vector<HistoricalKvDeleteIntent> retry_;
    // One generation per physical slot is enough to suppress duplicate
    // commands across all yielded batches of the same policy round.
    std::vector<uint64_t> seen_generation_;
    // Shadow mode simulates committed retirements as pending Ready credit so
    // the same generation is not proposed on every polling tick.
    std::vector<uint64_t> shadow_generation_;
    uint64_t next_round_id_ = 1;
    uint32_t reclaim_cursor_ = 0;
    uint32_t collect_cursor_ = 0;
    std::atomic<bool> stopping_{false};
    std::atomic<bool> worker_finished_{true};
    std::thread worker_;
    std::atomic<uint64_t> ticks_{0}, proposed_{0}, committed_{0}, shadow_{0};
    std::atomic<uint64_t> reclaimed_{0}, deferred_{0}, stale_{0};
    std::atomic<uint64_t> reevaluate_{0}, retry_count_{0};
    std::atomic<uint64_t> policy_failures_{0};
    std::atomic<uint64_t> worker_failures_{0};
    std::atomic<uint64_t> host_released_early_{0};
    std::atomic<uint64_t> partial_resource_releases_{0};
    std::atomic<uint64_t> boundary_batches_{0}, boundary_committed_{0};
    std::atomic<uint32_t> requested_ready_{0};
    bool policy_failed_ = false;
};

} // namespace kvmem
