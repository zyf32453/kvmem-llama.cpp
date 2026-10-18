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
        uint32_t reserved = 0; // Request-private; never pending reclaim credits.
        uint32_t pending = 0; // Retired or Releasing full-slot credits.
    };

    enum class RetireResult { Committed, AlreadyRetired, Stale, VersionChanged, Retry };
    enum class ReclaimResult { Complete, Deferred, Stale, Retry };
    enum class LeaseKind { Writer, Execution, Read, Transfer };
    enum class CompletionState { Unfinished, Live, Retired };

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
        void count_operation() noexcept;
        std::shared_ptr<Shared> shared_;
        HistoricalKvHandle handle_{};
        LeaseKind kind_ = LeaseKind::Read;
        bool counted_exec_ = false;
        bool counted_operation_ = false;
    };

    // Request-private, real physical slots. Reserved slots are neither Ready
    // credits nor policy candidates. Unused slots return on reset/destruction;
    // consumed slots follow the ordinary writer/retire/reclaim lifecycle.
    // Operations on one reservation are serialized by its request owner.
    class Reservation {
    public:
        Reservation() = default;
        ~Reservation();
        Reservation(Reservation &&) noexcept;
        Reservation & operator=(Reservation &&) noexcept;
        Reservation(const Reservation &) = delete;
        Reservation & operator=(const Reservation &) = delete;
        uint32_t remaining() const noexcept;
        std::optional<HistoricalKvHandle> next_handle() const noexcept;
        uint64_t repository_id() const noexcept;
        void reset() noexcept;

    private:
        friend class HistoricalKvRepository;
        std::shared_ptr<Shared> shared_;
        std::vector<HistoricalKvHandle> handles_;
        uint32_t consumed_ = 0;
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
    // All-or-none acquisition in a bounded pass; no waiting for GC. This
    // reserves slots only. A request owner must also reserve real payload,
    // scratch and metadata before promising whole-request admission.
    std::optional<Reservation> try_reserve_slots(uint32_t count);
    // Consumes one already reserved slot without allocating or scanning.
    // Failure preserves the reservation; callbacks should be prepared before
    // request admission, then moved here to avoid allocating in the foreground.
    std::optional<Lease> begin_reserved_write(
        Reservation & reservation, uint64_t logical_id, uint64_t original_begin,
        uint32_t valid_tokens, uint64_t birth_seq,
        uint64_t payload_bytes, ReleaseCallback release);
    // One atomic phase-bit update under the original writer: only Writing
    // becomes Live. Retirement is sticky and refcount churn cannot postpone
    // the last producer's completion publication.
    bool publish_write(const Lease & writer);
    // A retired block never grants a new lease, even if old leases remain.
    // acquire_new is the execution permission used at each active boundary.
    std::optional<Lease> acquire_new(HistoricalKvHandle handle);
    // A specialized execution permission for a request's older GPU-complete
    // span while its asynchronous Host capture is still Writing. The adapter
    // must prove the whole span's GPU producer has completed and it does not
    // overlap the current write range. This never grants a CPU read/transfer.
    std::optional<Lease> acquire_gpu_ready_writing(HistoricalKvHandle handle);
    std::optional<Lease> acquire_read(HistoricalKvHandle handle);
    std::optional<Lease> acquire_transfer(HistoricalKvHandle handle);
    // When supplied, the content version is checked under the same slot guard
    // as the phase transition. A mismatch leaves the generation Live/Writing.
    RetireResult retire(HistoricalKvHandle handle,
                        std::optional<uint64_t> expected_content_version = std::nullopt);
    // Call from the CPU reclaimer, not from the foreground. False/throwing
    // callbacks leave the slot Releasing and are retried with the same handle.
    ReclaimResult try_reclaim(HistoricalKvHandle handle);

    // Record a successfully submitted model computation using its already
    // admitted execution lease. Submission can follow retire_commit if that
    // lease was acquired first; a bare handle cannot mint permission here.
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
    // Request-completion observations only; neither grants access nor permits
    // physical release. Call after the request has quiesced new task admission.
    // Public writer/read/transfer/execution leases count, internal metadata
    // guards do not. Reclamation still uses the full control reference count.
    uint64_t outstanding_operations() const noexcept;
    // After close and after the GC/metadata producers stop: no lease or
    // reservation still owns the shared state. Unlike the operation count,
    // this also covers a lease's final shared_ptr release. The background
    // domain cleanup must retain its repository/storage until this is true.
    bool cleanup_ready() const noexcept;
    // For a handle returned by begin_write/begin_reserved_write, Live means
    // its producer published; Retired includes ongoing/completed release or
    // later slot reuse. Uses atomic scalars, never a worker lock or a guard.
    CompletionState completion_state(HistoricalKvHandle handle) const noexcept;
    std::optional<HistoricalKvBlockMeta> inspect(HistoricalKvHandle handle) const;
    // Bounded, individually guarded observations; not a global atomic instant.
    // `active` counts repository ExecLease objects, not device work; the
    // adapter must hold those leases until its actual completion boundary.
    // observe_range obtains blocks and pressure counts in one bounded pass.
    RangeObservation observe_range(uint32_t first, uint32_t count) const;
    // Reuse caller-owned storage for repeated worker scans. The caller must
    // reserve at least min(count, capacity - first) entries before calling;
    // insufficient capacity is rejected before any observation is changed.
    void observe_range_into(uint32_t first, uint32_t count,
                            RangeObservation & out) const;
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
    std::optional<Lease> acquire_kind(HistoricalKvHandle, LeaseKind, uint32_t allowed_phases);
    std::shared_ptr<Shared> shared_;
};

} // namespace kvmem
