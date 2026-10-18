#pragma once

#include "kvmem/historical_kv_raw_store.hpp"

namespace kvmem {

// The RAM part of request admission. This is deliberately not a promise that
// an adapter has also reserved GPU pages or directory nodes. If the store
// enables independent NVMe copies, their real extents are reserved here too.
// The adapter must compose those resources before marking a request active,
// and enforce these token/append limits during execution.
struct HistoricalKvRamDemand {
    uint64_t uncached_input_tokens = 0;
    std::optional<uint64_t> max_accepted_output_tokens;
    uint64_t max_append_boundaries = 0;
    // Additional simultaneous future allocations only. Existing records and
    // their outstanding readers already consume pool slots, not this margin.
    uint64_t extra_inflight_slots = 0;
    uint64_t private_scratch_bytes = 0;
    bool streaming = false;
};

class HistoricalKvRamReservation {
public:
    HistoricalKvRamReservation(HistoricalKvRamReservation &&) noexcept;
    HistoricalKvRamReservation & operator=(HistoricalKvRamReservation &&) noexcept;
    HistoricalKvRamReservation(const HistoricalKvRamReservation &) = delete;
    HistoricalKvRamReservation & operator=(const HistoricalKvRamReservation &) = delete;

    static uint64_t required_slots(const HistoricalKvRamDemand & demand, uint32_t block_tokens);
    // All-or-none physical slots + full-size layer payloads + real scratch.
    // Invalid/unbounded demand throws. Insufficient capacity or bad_alloc
    // returns nullopt, returning all private credits without releasing history.
    static std::optional<HistoricalKvRamReservation> try_reserve(
        std::shared_ptr<HistoricalKvRepository> repository,
        std::shared_ptr<HistoricalKvRawStore> storage,
        const HistoricalKvRamDemand & demand);

    HistoricalKvRepository::Reservation & slots() noexcept { return slots_; }
    HistoricalKvRawStore::PreparedPayload & payload(uint32_t index) { return payloads_.at(index); }
    uint32_t payload_count() const noexcept { return static_cast<uint32_t>(payloads_.size()); }
    uint8_t * scratch_data() noexcept { return scratch_.data(); }
    size_t scratch_size() const noexcept { return scratch_.size(); }
    // Actual vector capacity acquired at admission, not remaining usage after
    // installing/releasing some of the payloads.
    uint64_t reserved_payload_bytes() const noexcept { return reserved_payload_bytes_; }
    uint64_t reserved_nvme_bytes() const noexcept { return reserved_nvme_bytes_; }
    uint64_t reserved_scratch_bytes() const noexcept { return scratch_.capacity(); }

private:
    HistoricalKvRamReservation() = default;
    // Keep owners alive through reservation cancellation and move. Payloads
    // are returned before the unused repository slots become Ready again.
    std::shared_ptr<HistoricalKvRepository> repository_;
    std::shared_ptr<HistoricalKvRawStore> storage_;
    HistoricalKvRepository::Reservation slots_;
    std::vector<HistoricalKvRawStore::PreparedPayload> payloads_;
    std::vector<uint8_t> scratch_;
    uint64_t reserved_payload_bytes_ = 0;
    uint64_t reserved_nvme_bytes_ = 0;
};

} // namespace kvmem
