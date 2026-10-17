#include "kvmem/historical_kv_capture_session.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace kvmem {

std::unique_ptr<HistoricalKvCaptureSession> HistoricalKvCaptureSession::try_create(
        KvMemRuntime & runtime, HistoricalKvRawStoreConfig config,
        uint32_t capacity, const HistoricalKvRamDemand & demand) {
    if (!capacity || runtime.store().total_tokens() || runtime.store().block_count() ||
        runtime.pending() || runtime.historical_shadow_directory().size() ||
        runtime.config().cpu_bytes || runtime.config().nvme_bytes ||
        config.max_tokens_per_slot != runtime.store().config().block_tokens ||
        demand.extra_inflight_slots)
        throw std::invalid_argument("capture baseline requires an empty RAM-only runtime without extra in-flight allocations");
    const auto required = HistoricalKvRamReservation::required_slots(demand, config.max_tokens_per_slot);
    // Production positions and dense IDs remain 32-bit in this baseline.
    if (demand.uncached_input_tokens > uint64_t(INT32_MAX) ||
        *demand.max_accepted_output_tokens > uint64_t(INT32_MAX) - demand.uncached_input_tokens)
        throw std::invalid_argument("capture request exceeds logical position range");
    if (required > capacity) return {};
    try {
        auto result = std::unique_ptr<HistoricalKvCaptureSession>(new HistoricalKvCaptureSession(runtime));
        result->repository_ = std::make_shared<HistoricalKvRepository>(capacity);
        result->storage_ = std::make_shared<HistoricalKvRawStore>(std::move(config), capacity);
        auto ram = HistoricalKvRamReservation::try_reserve(result->repository_, result->storage_, demand);
        if (!ram) return {};
        result->reservation_ = std::move(ram);
        result->reserved_blocks_ = static_cast<uint32_t>(required);
        // Birth sequences are the admitted slot ordinal + 1. Every actual
        // model submission must sort after those stable initial LRU values.
        result->last_use_sequence_ = required;
        result->input_remaining_ = demand.uncached_input_tokens;
        result->output_remaining_ = *demand.max_accepted_output_tokens;
        result->appends_remaining_ = demand.max_append_boundaries;
        result->handles_.reserve(static_cast<size_t>(required));
        result->writers_.reserve(static_cast<size_t>(required));
        result->callbacks_.reserve(static_cast<size_t>(required));
        // Prepare callable allocations before publishing the session. Weak
        // repository capture avoids a repository -> callback -> owner cycle.
        for (uint64_t i = 0; i < required; ++i) {
            result->callbacks_.emplace_back([
                weak = std::weak_ptr<HistoricalKvRepository>(result->repository_),
                storage = result->storage_](HistoricalKvHandle handle) {
                const auto owner = weak.lock();
                if (!owner) return false;
                const auto released = storage->release(*owner, handle).code;
                return released == HistoricalKvRawStore::ReleaseCode::Released ||
                       released == HistoricalKvRawStore::ReleaseCode::AlreadyReleased ||
                       released == HistoricalKvRawStore::ReleaseCode::Absent;
            });
        }
        return result;
    } catch (const std::bad_alloc &) {
        return {};
    } catch (const std::length_error &) {
        return {};
    }
}

HistoricalKvCaptureSession::~HistoricalKvCaptureSession() {
    // Tasks retain their own repository/storage/context. Closing admission
    // does not revoke their existing writer or free their installed payload.
    if (repository_) repository_->close();
}

void HistoricalKvCaptureSession::fail() noexcept {
    failed_ = true;
    repository_->close();
    producers_.clear();
    writers_.clear();
    reservation_.reset(); // Only unused private allocations are returned here.
}

bool HistoricalKvCaptureSession::append(uint32_t tokens) {
    auto & remaining = input_ ? input_remaining_ : output_remaining_;
    if (failed_ || finished_ || !reservation_ || !producers_.empty() ||
        tokens > remaining || (tokens && !appends_remaining_) ||
        runtime_.store().total_tokens() != expected_rows_ ||
        runtime_.store().block_count() != expected_blocks_) return false;
    if (!tokens) return true;
    const uint32_t block_tokens = storage_->config().max_tokens_per_slot;
    const uint64_t count = (uint64_t(tokens) + block_tokens - 1) / block_tokens;
    if (count > reservation_->slots().remaining()) return false;
    try {
        writers_.clear();
        uint32_t position = expected_rows_;
        uint32_t left = tokens;
        for (uint64_t i = 0; i < count; ++i) {
            const auto index = static_cast<uint32_t>(handles_.size());
            const auto n = std::min(left, block_tokens);
            auto & payload = reservation_->payload(index);
            auto writer = repository_->begin_reserved_write(reservation_->slots(),
                expected_blocks_ + i, position, n, uint64_t(index) + 1,
                payload.payload_bytes(), std::move(callbacks_[index]));
            if (!writer) throw std::runtime_error("capture reserved writer admission failed");
            handles_.push_back(writer->handle());
            writers_.push_back(std::move(*writer));
            if (!storage_->install_payload(writers_.back(), handles_.back(), position, n, payload))
                throw std::runtime_error("capture reserved payload installation failed");
            position += n;
            left -= n;
        }
        if (!runtime_.register_historical_append(repository_, storage_, writers_, producers_))
            throw std::runtime_error("capture span registration failed");
        writers_.clear();
        expected_rows_ = position;
        expected_blocks_ += static_cast<uint32_t>(count);
        remaining -= tokens;
        --appends_remaining_;
        return true;
    } catch (...) {
        fail();
        throw;
    }
}

bool HistoricalKvCaptureSession::end_input() noexcept {
    if (failed_ || finished_ || input_remaining_ || !producers_.empty()) return false;
    input_ = false;
    return true;
}

bool HistoricalKvCaptureSession::finish() {
    if (failed_ || finished_ || input_ || !producers_.empty() ||
        repository_->outstanding_operations()) return false;
    uint32_t live = 0;
    for (const auto handle : handles_) {
        switch (repository_->completion_state(handle)) {
        case HistoricalKvRepository::CompletionState::Unfinished: return false;
        case HistoricalKvRepository::CompletionState::Live: ++live; break;
        case HistoricalKvRepository::CompletionState::Retired: break;
        }
    }
    reservation_.reset();
    live_blocks_at_finish_ = live;
    finished_ = true;
    return true;
}

} // namespace kvmem
