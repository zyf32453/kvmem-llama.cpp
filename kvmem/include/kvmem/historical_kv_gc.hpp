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
// repository to release resources only after its old leases have gone away.
// This module is host-only and is not wired into the production adapter yet.
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
    };

    HistoricalKvGc(std::shared_ptr<HistoricalKvRepository> repository,
                   std::unique_ptr<HistoricalKvPolicy> policy, Config config);
    ~HistoricalKvGc();
    HistoricalKvGc(const HistoricalKvGc &) = delete;
    HistoricalKvGc & operator=(const HistoricalKvGc &) = delete;

    // start() is explicit so tests can deterministically drive manual_tick().
    void start();
    void stop() noexcept;
    void manual_tick();
    Stats stats() const;

private:
    HistoricalKvCommitCode commit(const HistoricalKvDeleteIntent & intent);
    void acknowledge(const HistoricalKvDeleteIntent & intent,
                     HistoricalKvCommitCode code);
    void run() noexcept;

    std::shared_ptr<HistoricalKvRepository> repository_;
    std::unique_ptr<HistoricalKvPolicy> policy_;
    Config config_;
    mutable std::mutex tick_mu_;
    std::unique_ptr<HistoricalKvSnapshot> round_;
    HistoricalKvPressure pressure_{};
    std::vector<HistoricalKvDeleteIntent> retry_;
    // One generation per physical slot is enough to suppress duplicate
    // commands across all yielded batches of the same policy round.
    std::vector<uint64_t> seen_generation_;
    uint64_t next_round_id_ = 1;
    uint32_t reclaim_cursor_ = 0;
    uint32_t collect_cursor_ = 0;
    std::atomic<bool> stopping_{false};
    std::thread worker_;
    std::atomic<uint64_t> ticks_{0}, proposed_{0}, committed_{0}, shadow_{0};
    std::atomic<uint64_t> reclaimed_{0}, deferred_{0}, stale_{0};
    std::atomic<uint64_t> reevaluate_{0}, retry_count_{0};
    std::atomic<uint64_t> policy_failures_{0};
    std::atomic<uint64_t> worker_failures_{0};
    bool policy_failed_ = false;
};

} // namespace kvmem
