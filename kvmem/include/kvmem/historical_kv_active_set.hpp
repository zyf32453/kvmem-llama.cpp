#pragma once

#include "kvmem/historical_kv_repository.hpp"

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace kvmem {

struct HistoricalKvActiveRequest {
    HistoricalKvHandle handle;
    bool mandatory = false;
    bool gpu_ready_writing = false;
};

// One caller-defined active operation. Every invocation of acquire() obtains
// fresh execution permission, including for blocks still held by an older
// active operation. Retired optional blocks are omitted; a retired mandatory
// block aborts the entire acquisition. The owner keeps this object alive until
// its actual execution and device work have completed.
class HistoricalKvActiveSet {
public:
    struct AcquireResult {
        std::unique_ptr<HistoricalKvActiveSet> active;
        std::optional<HistoricalKvHandle> missing_mandatory;

        explicit operator bool() const { return static_cast<bool>(active); }
    };

    static AcquireResult acquire(
        std::shared_ptr<HistoricalKvRepository> repository,
        const std::vector<HistoricalKvActiveRequest> & requested);

    // Allocate the object and both arrays before request admission. One
    // prepared set supports one active operation at a time; each successful
    // try_acquire obtains fresh leases. Before calling release(), the owner
    // must confirm that its real execution/device work has finished; release
    // itself does not synchronize a device. Calls are owner-serialized.
    static std::unique_ptr<HistoricalKvActiveSet> prepare(
        std::shared_ptr<HistoricalKvRepository> repository, size_t max_requested);
    enum class AcquireStatus { Acquired, Busy, CapacityExceeded, Closed, MissingMandatory };
    struct TryAcquireResult {
        AcquireStatus status;
        std::optional<HistoricalKvHandle> missing_mandatory;
        explicit operator bool() const { return status == AcquireStatus::Acquired; }
    };
    // Does not allocate or grow prepared arrays. Busy/overflow preserves the
    // previous operation; a failed mandatory admission returns all new leases.
    TryAcquireResult try_acquire(const std::vector<HistoricalKvActiveRequest> & requested);
    void release() noexcept;
    bool engaged() const noexcept { return engaged_; }
    size_t capacity() const noexcept { return max_requested_; }
    size_t capacity_bytes() const noexcept {
        return leases_.capacity() * sizeof(HistoricalKvRepository::Lease) +
               handles_.capacity() * sizeof(HistoricalKvHandle);
    }

    HistoricalKvActiveSet(const HistoricalKvActiveSet &) = delete;
    HistoricalKvActiveSet & operator=(const HistoricalKvActiveSet &) = delete;

    const std::vector<HistoricalKvHandle> & handles() const { return handles_; }
    bool empty() const { return leases_.empty(); }

    // Call immediately after the admitted model computation is successfully
    // submitted. An acquired but abandoned plan does not update last_use.
    // A later decode submission under the same active permission may call it
    // again with that submission's sequence, even after retirement.
    bool mark_execution_submitted(uint64_t sequence);

private:
    explicit HistoricalKvActiveSet(std::shared_ptr<HistoricalKvRepository> repository);

    std::shared_ptr<HistoricalKvRepository> repository_;
    std::vector<HistoricalKvRepository::Lease> leases_;
    std::vector<HistoricalKvHandle> handles_;
    size_t max_requested_ = 0;
    bool engaged_ = false;
};

} // namespace kvmem
