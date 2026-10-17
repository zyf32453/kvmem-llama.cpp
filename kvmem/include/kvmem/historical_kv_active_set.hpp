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
};

} // namespace kvmem
