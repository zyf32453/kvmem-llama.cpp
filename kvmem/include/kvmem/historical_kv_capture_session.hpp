#pragma once

#include "kvmem/historical_kv_ram_reservation.hpp"
#include "kvmem/historical_kv_cleanup.hpp"
#include "kvmem/kvmem_runtime.hpp"

namespace kvmem {

class HistoricalKvCaptureDomain;

// A finite, single-request capture owner. The caller may attach one GC
// manager to its repository after admission. It connects actual appends to
// the RAM reservation and producer handoff. This is NOT
// full production request admission: cell/active/backend allocations and
// additional GPU routes remain fallible; failed sessions drain their
// tasks. A domain may admit a new finite request after finish/cancel; reset,
// branching and streaming are not supported by this owner.
class HistoricalKvCaptureSession {
public:
    // Prepare private RAM and append metadata without changing the runtime.
    // Destruction before admit() rolls back synchronously, independent of GC
    // or the cleanup worker. The caller prepares its other resources first.
    static std::unique_ptr<HistoricalKvCaptureSession> try_prepare(
        KvMemRuntime & runtime, HistoricalKvRawStoreConfig config,
        uint32_t capacity, const HistoricalKvRamDemand & demand,
        HistoricalKvCleanup * cleanup = nullptr);
    // Reserve one bounded request against a persistent domain. The first
    // request can be prepared while its domain is private; admit() installs
    // that domain after the caller prepares its other resources. Only one
    // request may be prepared/active in a domain at a time.
    static std::unique_ptr<HistoricalKvCaptureSession> try_prepare_next(
        std::shared_ptr<HistoricalKvCaptureDomain> domain,
        const HistoricalKvRamDemand & demand,
        HistoricalKvCleanup * cleanup = nullptr);
    bool admit() noexcept;
    bool admitted() const noexcept { return admitted_; }
    // Convenience for host-only callers with no external preparation phase.
    static std::unique_ptr<HistoricalKvCaptureSession> try_create(
        KvMemRuntime & runtime, HistoricalKvRawStoreConfig config,
        uint32_t capacity, const HistoricalKvRamDemand & demand,
        HistoricalKvCleanup * cleanup = nullptr);
    ~HistoricalKvCaptureSession();
    HistoricalKvCaptureSession(const HistoricalKvCaptureSession &) = delete;
    HistoricalKvCaptureSession & operator=(const HistoricalKvCaptureSession &) = delete;

    // Append immutable spans using reserved RAM, directory nodes, producers
    // and dense block descriptors; the successful path does not allocate.
    // Quota/state rejection does not change the runtime. Internal installation
    // failure closes the session, retaining installed data until domain teardown.
    bool append(uint32_t tokens);
    // Call only after EVERY ubatch for the append has captured its producer.
    void producers_captured() noexcept { producers_.clear(); }
    bool end_input() noexcept;
    // Caller first drains graph, D2H and accumulators. Remaining Live blocks
    // must be published; blocks retired during the request may already be
    // releasing or reclaimed. No task references may remain; background
    // metadata guards/release callbacks do not delay this request boundary.
    // Unused private allocations are submitted to the pre-reserved cleanup
    // cell. payload_bytes() may still include them until its worker runs.
    bool finish();
    // Healthy cancellation after submitted work has completed. Partial input
    // is allowed; installed history stays owned by the domain, unused RAM is
    // returned by the cleanup worker. This cannot recover a failed session.
    bool cancel();
    void fail() noexcept;
    bool failed() const noexcept { return failed_; }
    bool finished() const noexcept { return finished_; }
    uint32_t created_blocks() const noexcept { return static_cast<uint32_t>(handles_.size()); }
    uint32_t live_blocks_at_finish() const noexcept { return live_blocks_at_finish_; }
    uint32_t reserved_blocks() const noexcept { return reserved_blocks_; }
    uint32_t metadata_remaining() const noexcept;
    uint64_t payload_bytes() const noexcept { return storage_->allocated_payload_bytes(); }
    // The request owner may attach one GC manager to this exact repository.
    // Keeping these owners alive also keeps late release callbacks' target alive.
    std::shared_ptr<HistoricalKvRepository> repository() const noexcept { return repository_; }
    std::shared_ptr<HistoricalKvRawStore> storage() const noexcept { return storage_; }
    std::shared_ptr<HistoricalKvCaptureDomain> domain() const noexcept { return domain_; }
    uint64_t next_use_sequence() noexcept;
    uint64_t input_remaining() const noexcept { return input_remaining_; }
    uint64_t output_remaining() const noexcept { return output_remaining_; }
    uint64_t appends_remaining() const noexcept { return appends_remaining_; }

private:
    friend struct HistoricalKvCaptureSessionTestAccess;
    explicit HistoricalKvCaptureSession(KvMemRuntime & runtime);
    KvMemRuntime & runtime_;
    std::shared_ptr<HistoricalKvCaptureDomain> domain_;
    std::shared_ptr<HistoricalKvRepository> repository_;
    std::shared_ptr<HistoricalKvRawStore> storage_;
    struct ReservationCleanup;
    HistoricalKvCleanup::Ticket cleanup_ticket_;
    std::unique_ptr<ReservationCleanup> reservation_;
    void release_unused() noexcept;
    bool complete(bool cancelled);
    std::vector<HistoricalKvRepository::ReleaseCallback> callbacks_;
    std::vector<HistoricalKvRepository::Lease> writers_;
    std::vector<HistoricalKvShadowDirectory::Producer> producers_;
    std::vector<HistoricalKvHandle> handles_;
    uint64_t input_remaining_ = 0, output_remaining_ = 0, appends_remaining_ = 0;
    uint64_t last_use_sequence_ = 0;
    uint64_t birth_start_ = 0;
    uint64_t expected_revision_ = 0;
    uint32_t expected_rows_ = 0, expected_blocks_ = 0, reserved_blocks_ = 0;
    uint32_t live_blocks_at_finish_ = 0;
    bool input_ = true, failed_ = false, finished_ = false, admitted_ = false;
    bool owns_domain_claim_ = false;
};

// Runtime-scoped ownership of the physical pool and bounded metadata table.
// A request returns only its private unused RAM; installed history and its
// repository remain available to later requests until the domain is closed.
// The runtime and cleanup service must outlive this object. All foreground
// methods are serialized by their caller; the GC may reclaim concurrently.
class HistoricalKvCaptureDomain {
public:
    // Prepare the repository and full physical table privately. Compose this
    // with the adapter's other resources, then commit with admit().
    static std::shared_ptr<HistoricalKvCaptureDomain> try_prepare(
        KvMemRuntime & runtime, HistoricalKvRawStoreConfig config,
        uint32_t capacity, HistoricalKvCleanup * cleanup = nullptr);
    bool admit() noexcept;
    // Host-only convenience for callers with no other admission resources.
    static std::shared_ptr<HistoricalKvCaptureDomain> try_create(
        KvMemRuntime & runtime, HistoricalKvRawStoreConfig config,
        uint32_t capacity, HistoricalKvCleanup * cleanup = nullptr);
    ~HistoricalKvCaptureDomain();
    HistoricalKvCaptureDomain(const HistoricalKvCaptureDomain &) = delete;
    HistoricalKvCaptureDomain & operator=(const HistoricalKvCaptureDomain &) = delete;
    std::shared_ptr<HistoricalKvRepository> repository() const noexcept { return repository_; }
    std::shared_ptr<HistoricalKvRawStore> storage() const noexcept { return storage_; }
    uint32_t metadata_remaining() const noexcept;
    void close() noexcept;

private:
    friend class HistoricalKvCaptureSession;
    explicit HistoricalKvCaptureDomain(KvMemRuntime & runtime);
    struct MetadataCleanup;
    KvMemRuntime & runtime_;
    std::shared_ptr<HistoricalKvRepository> repository_;
    std::shared_ptr<HistoricalKvRawStore> storage_;
    HistoricalKvCleanup::Ticket cleanup_ticket_;
    std::unique_ptr<MetadataCleanup> cleanup_task_;
    uint64_t next_logical_id_ = 0;
    uint64_t last_use_sequence_ = 0;
    bool request_claimed_ = false;
    bool installed_ = false;
};

} // namespace kvmem
