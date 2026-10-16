#pragma once

#include "kvmem/historical_kv_policy.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

namespace kvmem {

class HistoricalKvGc;

// A fixed-size, host-only lifetime directory. It deliberately does not own a
// RawKvStore allocation yet. The release callback must identify and release
// the exact allocation named by its handle and must tolerate a partial retry.
class HistoricalKvRepository {
    struct Shared;

public:
    using ReleaseCallback = std::function<bool(const HistoricalKvHandle &)>;

    struct RangeObservation {
        // At most `count` entries, in ascending physical slot order.
        std::vector<HistoricalKvBlockMeta> blocks;
        uint32_t ready = 0;
        uint32_t pending = 0; // Retired or Releasing full-slot credits.
    };

    enum class RetireResult { Committed, AlreadyRetired, Stale, Retry };
    enum class ReclaimResult { Complete, Deferred, Stale, Retry };
    enum class LeaseKind { Writer, Execution, Read, Transfer };

    class Lease {
    public:
        Lease() = default;
        ~Lease();
        Lease(Lease &&) noexcept;
        Lease & operator=(Lease &&) noexcept;
        Lease(const Lease &) = delete;
        Lease & operator=(const Lease &) = delete;

        explicit operator bool() const { return static_cast<bool>(shared_); }
        HistoricalKvHandle handle() const { return handle_; }
        LeaseKind kind() const { return kind_; }
        bool is_writer() const { return kind_ == LeaseKind::Writer; }
        uint64_t repository_id() const noexcept;
        std::optional<HistoricalKvBlockMeta> metadata() const;
        void reset() noexcept;

    private:
        friend class HistoricalKvRepository;
        Lease(std::shared_ptr<Shared>, HistoricalKvHandle, LeaseKind);
        void disarm() noexcept;
        std::shared_ptr<Shared> shared_;
        HistoricalKvHandle handle_{};
        LeaseKind kind_ = LeaseKind::Read;
        bool counted_exec_ = false;
    };

    // Zero requests an automatically unique incarnation for this process.
    // Explicit nonzero incarnations are for a caller-controlled session domain.
    explicit HistoricalKvRepository(uint32_t capacity, uint64_t incarnation = 0);
    ~HistoricalKvRepository();
    HistoricalKvRepository(const HistoricalKvRepository &) = delete;
    HistoricalKvRepository & operator=(const HistoricalKvRepository &) = delete;

    // The caller receives the sole writer lease. An internal Reserving phase
    // protects metadata publication before Writing becomes visible.
    std::optional<Lease> begin_write(uint64_t logical_id, uint64_t original_begin,
                                     uint32_t valid_tokens, uint64_t birth_seq,
                                     uint64_t payload_bytes, ReleaseCallback release);
    bool publish_write(const Lease & writer);
    // A retired block never grants a new lease, even if old leases remain.
    // acquire_new is the execution permission used at each active boundary.
    std::optional<Lease> acquire_new(HistoricalKvHandle handle);
    std::optional<Lease> acquire_read(HistoricalKvHandle handle);
    std::optional<Lease> acquire_transfer(HistoricalKvHandle handle);
    RetireResult retire(HistoricalKvHandle handle);
    // Call from the CPU reclaimer, not from the foreground. False/throwing
    // callbacks leave the slot Releasing and are retried with the same handle.
    ReclaimResult try_reclaim(HistoricalKvHandle handle);

    bool mark_executed(HistoricalKvHandle handle, uint64_t sequence);
    // Records an execution already admitted before retire_commit. Such an
    // operation may finish after retirement without reopening new admission.
    bool mark_executed(const Lease & admitted, uint64_t sequence);
    // A callback must not accept a writer from another repository even if a
    // caller deliberately reused the same incarnation, slot and generation.
    bool owns_writer(const Lease & writer) const noexcept;
    // Process-unique repository identity, independent of caller-controlled
    // incarnation values. Storage bindings use this to reject a foreign GC.
    uint64_t repository_id() const noexcept;
    // Only Releasing has closed all lease admission and drained old refs.
    // Storage callbacks may release physical payload only in this phase.
    bool is_releasing(HistoricalKvHandle handle) const noexcept;
    // True only after the exact generation's release callback completed and
    // its slot crossed to Ready. A later generation also proves completion.
    // The repository ID prevents equal-looking handles from another owner
    // from being used as a reclaim acknowledgement.
    bool reclaim_completed(HistoricalKvHandle handle,
                           uint64_t expected_repository_id) const noexcept;
    std::optional<HistoricalKvBlockMeta> inspect(HistoricalKvHandle handle) const;
    // Bounded, individually guarded observations; not a global atomic instant.
    // `active` counts repository ExecLease objects, not device work; the
    // adapter must hold those leases until its actual completion boundary.
    // observe_range obtains blocks and pressure counts in one bounded pass.
    RangeObservation observe_range(uint32_t first, uint32_t count) const;
    std::vector<HistoricalKvBlockMeta> snapshot() const;
    std::vector<HistoricalKvBlockMeta> snapshot_range(uint32_t first, uint32_t count) const;
    uint32_t ready_count() const;
    uint32_t ready_count_range(uint32_t first, uint32_t count) const;
    uint32_t capacity() const;
    uint64_t incarnation() const;
    // Stops calls which begin after close; an already admitted writer may
    // finish under its existing lease.
    void close() noexcept;
    bool closed() const;

private:
    friend class HistoricalKvGc;
    bool claim_gc_owner() noexcept;
    void release_gc_owner() noexcept;
    std::optional<Lease> acquire_kind(HistoricalKvHandle, LeaseKind);
    std::shared_ptr<Shared> shared_;
};

} // namespace kvmem
