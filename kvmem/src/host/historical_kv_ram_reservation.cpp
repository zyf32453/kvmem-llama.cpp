#include "kvmem/historical_kv_ram_reservation.hpp"

#include <algorithm>
#include <limits>
#include <new>
#include <stdexcept>

namespace kvmem {
namespace {
uint64_t add_checked(uint64_t a, uint64_t b) {
    if (b > std::numeric_limits<uint64_t>::max() - a)
        throw std::invalid_argument("historical KV request demand overflow");
    return a + b;
}
} // namespace

HistoricalKvRamReservation::HistoricalKvRamReservation(HistoricalKvRamReservation && other) noexcept
    : repository_(std::move(other.repository_)), storage_(std::move(other.storage_)),
      slots_(std::move(other.slots_)), payloads_(std::move(other.payloads_)),
      scratch_(std::move(other.scratch_)), reserved_payload_bytes_(other.reserved_payload_bytes_) {
    other.reserved_payload_bytes_ = 0;
}

HistoricalKvRamReservation & HistoricalKvRamReservation::operator=(HistoricalKvRamReservation && other) noexcept {
    if (this != &other) {
        // Swap complete bundles, then destroy the old one in field destruction
        // order (payloads before slot credits). Memberwise move assignment
        // would return old slots before discarding their prepared allocations.
        HistoricalKvRamReservation replacement(std::move(other));
        using std::swap;
        swap(repository_, replacement.repository_);
        swap(storage_, replacement.storage_);
        swap(slots_, replacement.slots_);
        swap(payloads_, replacement.payloads_);
        swap(scratch_, replacement.scratch_);
        swap(reserved_payload_bytes_, replacement.reserved_payload_bytes_);
    }
    return *this;
}

uint64_t HistoricalKvRamReservation::required_slots(
        const HistoricalKvRamDemand & demand, uint32_t block_tokens) {
    if (demand.streaming)
        throw std::invalid_argument("historical KV streaming reservation is not implemented");
    if (!block_tokens || !demand.max_accepted_output_tokens)
        throw std::invalid_argument("historical KV requires a finite output bound and block size");
    const uint64_t tokens = add_checked(demand.uncached_input_tokens, *demand.max_accepted_output_tokens);
    if (tokens && !demand.max_append_boundaries)
        throw std::invalid_argument("historical KV requires an append boundary bound");
    const uint64_t boundaries = std::min(tokens, demand.max_append_boundaries);
    // For k nonempty appends with total T, sum ceil(t_i/B) is at most
    // k + floor((T-k)/B). This bound is attainable (put all extra tokens in
    // one append). k=min(T,max_boundaries) covers deleting every short tail;
    // no old tail room or future reclaim credit is assumed.
    const uint64_t spans = boundaries + (tokens - boundaries) / block_tokens;
    return add_checked(spans, demand.extra_inflight_slots);
}

std::optional<HistoricalKvRamReservation> HistoricalKvRamReservation::try_reserve(
        std::shared_ptr<HistoricalKvRepository> repository,
        std::shared_ptr<HistoricalKvRawStore> storage,
        const HistoricalKvRamDemand & demand) {
    if (!repository || !storage || repository->capacity() != storage->capacity())
        throw std::invalid_argument("historical KV RAM reservation requires matching pool capacities");
    const uint64_t count = required_slots(demand, storage->config().max_tokens_per_slot);
    if (count > repository->capacity()) return std::nullopt;
    if (demand.private_scratch_bytes > std::numeric_limits<size_t>::max())
        throw std::invalid_argument("historical KV scratch demand exceeds address space");
    try {
        HistoricalKvRamReservation result;
        result.repository_ = std::move(repository);
        result.storage_ = std::move(storage);
        auto slots = result.repository_->try_reserve_slots(static_cast<uint32_t>(count));
        if (!slots) return std::nullopt;
        result.slots_ = std::move(*slots);
        result.payloads_.reserve(static_cast<size_t>(count));
        for (uint32_t i = 0; i < count; ++i) {
            auto payload = result.storage_->prepare_payload(result.storage_->config().max_tokens_per_slot);
            result.reserved_payload_bytes_ = add_checked(result.reserved_payload_bytes_, payload.payload_bytes());
            result.payloads_.push_back(std::move(payload));
        }
        result.scratch_.resize(static_cast<size_t>(demand.private_scratch_bytes));
        if (result.repository_->closed()) return std::nullopt;
        return result;
    } catch (const std::bad_alloc &) {
        return std::nullopt;
    } catch (const std::length_error &) {
        return std::nullopt;
    }
}

} // namespace kvmem
