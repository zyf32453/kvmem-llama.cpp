#include "kvmem/historical_kv_capture_session.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace kvmem {

struct HistoricalKvCaptureSession::ReservationCleanup final : HistoricalKvCleanup::Task {
    std::optional<HistoricalKvRamReservation> ram;
    std::unique_ptr<HistoricalKvShadowDirectory::AppendReservation> metadata;
};

struct HistoricalKvCaptureDomain::MetadataCleanup final : HistoricalKvCleanup::Task {
    std::unique_ptr<HistoricalKvShadowDirectory::AppendReservation> metadata;
};

HistoricalKvCaptureDomain::HistoricalKvCaptureDomain(KvMemRuntime & runtime) : runtime_(runtime) {}

std::shared_ptr<HistoricalKvCaptureDomain> HistoricalKvCaptureDomain::try_prepare(
        KvMemRuntime & runtime, HistoricalKvRawStoreConfig config, uint32_t capacity,
        HistoricalKvCleanup * cleanup) {
    if (!capacity || runtime.store().total_tokens() || runtime.store().block_count() ||
        runtime.pending() || runtime.historical_shadow_directory().size() ||
        runtime.config().cpu_bytes || runtime.config().nvme_bytes ||
        config.max_tokens_per_slot != runtime.store().config().block_tokens)
        throw std::invalid_argument("capture domain requires an empty RAM-only runtime");
    try {
        auto ticket = (cleanup ? *cleanup : HistoricalKvCleanup::process()).try_reserve();
        if (!ticket) return {};
        auto result = std::shared_ptr<HistoricalKvCaptureDomain>(new HistoricalKvCaptureDomain(runtime));
        result->cleanup_ticket_ = std::move(*ticket);
        result->cleanup_task_ = std::make_unique<MetadataCleanup>();
        result->repository_ = std::make_shared<HistoricalKvRepository>(capacity);
        result->storage_ = std::make_shared<HistoricalKvRawStore>(std::move(config), capacity);
        result->cleanup_task_->metadata = runtime.prepare_historical_metadata(
            result->repository_, result->storage_, capacity);
        if (!result->cleanup_task_->metadata) return {};
        return result;
    } catch (const std::bad_alloc &) {
        return {};
    } catch (const std::length_error &) {
        return {};
    }
}

bool HistoricalKvCaptureDomain::admit() noexcept {
    if (installed_ || !cleanup_task_ || !cleanup_task_->metadata ||
        !runtime_.install_historical_metadata(*cleanup_task_->metadata)) return false;
    installed_ = true;
    return true;
}

std::shared_ptr<HistoricalKvCaptureDomain> HistoricalKvCaptureDomain::try_create(
        KvMemRuntime & runtime, HistoricalKvRawStoreConfig config, uint32_t capacity,
        HistoricalKvCleanup * cleanup) {
    auto result = try_prepare(runtime, std::move(config), capacity, cleanup);
    if (!result || !result->admit()) return {};
    return result;
}

HistoricalKvCaptureDomain::~HistoricalKvCaptureDomain() {
    close();
    if (installed_) cleanup_ticket_.submit(std::move(cleanup_task_));
}

uint32_t HistoricalKvCaptureDomain::metadata_remaining() const noexcept {
    return cleanup_task_ && cleanup_task_->metadata ? cleanup_task_->metadata->remaining() : 0;
}

void HistoricalKvCaptureDomain::close() noexcept {
    if (repository_) repository_->close();
}

HistoricalKvCaptureSession::HistoricalKvCaptureSession(KvMemRuntime & runtime) : runtime_(runtime) {}

std::unique_ptr<HistoricalKvCaptureSession> HistoricalKvCaptureSession::try_prepare_next(
        std::shared_ptr<HistoricalKvCaptureDomain> domain,
        const HistoricalKvRamDemand & demand, HistoricalKvCleanup * cleanup) {
    if (!domain || domain->request_claimed_ ||
        domain->repository_->closed() || domain->runtime_.pending() || demand.extra_inflight_slots)
        return {};
    auto & runtime = domain->runtime_;
    if (!domain->installed_ && (runtime.store().total_tokens() ||
        runtime.store().block_count() || runtime.historical_shadow_directory().size())) return {};
    const auto required = HistoricalKvRamReservation::required_slots(
        demand, domain->storage_->config().max_tokens_per_slot);
    if (required > domain->repository_->capacity() ||
        required > UINT64_MAX - domain->next_logical_id_ ||
        required > UINT64_MAX - domain->last_use_sequence_ ||
        runtime.store().total_tokens() > uint64_t(INT32_MAX) ||
        demand.uncached_input_tokens > uint64_t(INT32_MAX) - runtime.store().total_tokens() ||
        *demand.max_accepted_output_tokens > uint64_t(INT32_MAX) -
            runtime.store().total_tokens() - demand.uncached_input_tokens) return {};
    try {
        auto ticket = (cleanup ? *cleanup : HistoricalKvCleanup::process()).try_reserve();
        if (!ticket) return {};
        auto result = std::unique_ptr<HistoricalKvCaptureSession>(
            new HistoricalKvCaptureSession(runtime));
        result->domain_ = std::move(domain);
        result->cleanup_ticket_ = std::move(*ticket);
        result->reservation_ = std::make_unique<ReservationCleanup>();
        result->repository_ = result->domain_->repository_;
        result->storage_ = result->domain_->storage_;
        auto ram = HistoricalKvRamReservation::try_reserve(
            result->repository_, result->storage_, demand);
        if (!ram) return {};
        result->reservation_->ram = std::move(ram);
        // Only after private RAM and cleanup capacity are real can admission
        // maintenance detach old, exactly reclaimed position mappings. GC
        // never mutates the runtime or supplies speculative slot credits.
        auto & metadata = *result->domain_->cleanup_task_->metadata;
        for (uint32_t id = 0; result->domain_->installed_ &&
                              id < runtime.store().block_count() &&
                              metadata.remaining() < required; ++id) {
            if (runtime.store().blocks()[id].n_tokens &&
                !runtime.detach_reclaimed_historical_block(id)) continue;
            if (!runtime.store().blocks()[id].n_tokens)
                runtime.prepare_recycled_historical_slot(metadata, id);
        }
        if (metadata.remaining() < required) return {};
        result->reserved_blocks_ = static_cast<uint32_t>(required);
        result->expected_rows_ = runtime.store().total_tokens();
        result->expected_blocks_ = runtime.store().block_count();
        result->expected_revision_ = runtime.store().table_revision();
        result->birth_start_ = result->domain_->last_use_sequence_ + 1;
        result->input_remaining_ = demand.uncached_input_tokens;
        result->output_remaining_ = *demand.max_accepted_output_tokens;
        result->appends_remaining_ = demand.max_append_boundaries;
        result->handles_.reserve(static_cast<size_t>(required));
        result->writers_.reserve(static_cast<size_t>(required));
        result->producers_.reserve(static_cast<size_t>(required));
        result->callbacks_.reserve(static_cast<size_t>(required));
        for (uint64_t i = 0; i < required; ++i) {
            result->callbacks_.push_back({[
                weak = std::weak_ptr<HistoricalKvRepository>(result->repository_),
                storage = result->storage_](HistoricalKvHandle handle, HistoricalKvRepository::HostResource resource) {
                const auto owner = weak.lock();
                if (!owner) return false;
                const auto released = storage->release_resource(*owner, handle, resource).code;
                return released == HistoricalKvRawStore::ReleaseCode::Released ||
                       released == HistoricalKvRawStore::ReleaseCode::AlreadyReleased ||
                       released == HistoricalKvRawStore::ReleaseCode::Absent;
            }, result->reservation_->ram->payload(i).packed_kv_bytes(), result->reservation_->ram->payload(i).mean_k_bytes(),
               result->reservation_->ram->payload(i).nvme_bytes(), result->reservation_->ram->payload(i).writer_resource(),
               result->reservation_->ram->payload(i).pinned_bytes()});
        }
        result->domain_->last_use_sequence_ += required;
        result->domain_->request_claimed_ = true;
        result->owns_domain_claim_ = true;
        return result;
    } catch (const std::bad_alloc &) {
        return {};
    } catch (const std::length_error &) {
        return {};
    }
}

std::unique_ptr<HistoricalKvCaptureSession> HistoricalKvCaptureSession::try_create(
        KvMemRuntime & runtime, HistoricalKvRawStoreConfig config,
        uint32_t capacity, const HistoricalKvRamDemand & demand, HistoricalKvCleanup * cleanup) {
    auto result = try_prepare(runtime, std::move(config), capacity, demand, cleanup);
    if (!result || !result->admit()) return {};
    return result;
}

std::unique_ptr<HistoricalKvCaptureSession> HistoricalKvCaptureSession::try_prepare(
        KvMemRuntime & runtime, HistoricalKvRawStoreConfig config,
        uint32_t capacity, const HistoricalKvRamDemand & demand, HistoricalKvCleanup * cleanup) {
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
        auto ticket = (cleanup ? *cleanup : HistoricalKvCleanup::process()).try_reserve();
        if (!ticket) return {};
        auto result = std::unique_ptr<HistoricalKvCaptureSession>(new HistoricalKvCaptureSession(runtime));
        result->cleanup_ticket_ = std::move(*ticket);
        result->reservation_ = std::make_unique<ReservationCleanup>();
        result->repository_ = std::make_shared<HistoricalKvRepository>(capacity);
        result->storage_ = std::make_shared<HistoricalKvRawStore>(std::move(config), capacity);
        auto ram = HistoricalKvRamReservation::try_reserve(result->repository_, result->storage_, demand);
        if (!ram) return {};
        result->reservation_->ram = std::move(ram);
        result->reservation_->metadata = runtime.prepare_historical_metadata(
            result->repository_, result->storage_, static_cast<uint32_t>(required));
        if (!result->reservation_->metadata) return {};
        result->reserved_blocks_ = static_cast<uint32_t>(required);
        // Birth sequences are the admitted slot ordinal + 1. Every actual
        // model submission must sort after those stable initial LRU values.
        result->last_use_sequence_ = required;
        result->birth_start_ = 1;
        result->input_remaining_ = demand.uncached_input_tokens;
        result->output_remaining_ = *demand.max_accepted_output_tokens;
        result->appends_remaining_ = demand.max_append_boundaries;
        result->handles_.reserve(static_cast<size_t>(required));
        result->writers_.reserve(static_cast<size_t>(required));
        result->producers_.reserve(static_cast<size_t>(required));
        result->callbacks_.reserve(static_cast<size_t>(required));
        // Prepare callable allocations before publishing the session. Weak
        // repository capture avoids a repository -> callback -> owner cycle.
        for (uint64_t i = 0; i < required; ++i) {
            result->callbacks_.push_back({[
                weak = std::weak_ptr<HistoricalKvRepository>(result->repository_),
                storage = result->storage_](HistoricalKvHandle handle, HistoricalKvRepository::HostResource resource) {
                const auto owner = weak.lock();
                if (!owner) return false;
                const auto released = storage->release_resource(*owner, handle, resource).code;
                return released == HistoricalKvRawStore::ReleaseCode::Released ||
                       released == HistoricalKvRawStore::ReleaseCode::AlreadyReleased ||
                       released == HistoricalKvRawStore::ReleaseCode::Absent;
            }, result->reservation_->ram->payload(i).packed_kv_bytes(), result->reservation_->ram->payload(i).mean_k_bytes(),
               result->reservation_->ram->payload(i).nvme_bytes(), result->reservation_->ram->payload(i).writer_resource(),
               result->reservation_->ram->payload(i).pinned_bytes()});
        }
        return result;
    } catch (const std::bad_alloc &) {
        return {};
    } catch (const std::length_error &) {
        return {};
    }
}

bool HistoricalKvCaptureSession::prepare_gpu_window() {
    if (admitted_ || failed_ || finished_ || gpu_window_enabled_ || gpu_window_closed_) return false;
    gpu_window_.resize(repository_->capacity());
    gpu_window_enabled_ = true;
    return true;
}

bool HistoricalKvCaptureSession::has_gpu_permission(HistoricalKvHandle handle) const noexcept {
    if (!gpu_window_open() || handle.slot >= gpu_window_.size()) return false;
    const auto & lease = gpu_window_[handle.slot];
    if (!lease) return false;
    const auto held = lease.handle();
    return held.incarnation == handle.incarnation && held.generation == handle.generation &&
        held.slot == handle.slot;
}

bool HistoricalKvCaptureSession::mark_gpu_execution(HistoricalKvHandle handle, uint64_t sequence) {
    return has_gpu_permission(handle) && repository_->mark_executed(gpu_window_[handle.slot], sequence);
}

void HistoricalKvCaptureSession::close_gpu_window() noexcept {
    // The caller proves completion; this method neither synchronizes CUDA nor
    // frees Host payload. Lease release only makes background cleanup eligible.
    gpu_window_.clear();
    gpu_window_closed_ = true;
}

bool HistoricalKvCaptureSession::admit() noexcept {
    if (admitted_ || failed_ || finished_ || !reservation_ || runtime_.pending()) return false;
    if (domain_) {
        if (!owns_domain_claim_ || !domain_->request_claimed_ || domain_->repository_->closed() ||
            runtime_.store().total_tokens() != expected_rows_ ||
            runtime_.store().block_count() != expected_blocks_ ||
            runtime_.store().table_revision() != expected_revision_ ||
            domain_->metadata_remaining() < reserved_blocks_) return false;
        if (!domain_->installed_ && !domain_->admit()) return false;
    } else if (!reservation_->metadata ||
               !runtime_.install_historical_metadata(*reservation_->metadata)) return false;
    admitted_ = true;
    return true;
}

uint32_t HistoricalKvCaptureSession::metadata_remaining() const noexcept {
    if (domain_) return domain_->metadata_remaining();
    return reservation_ && reservation_->metadata ? reservation_->metadata->remaining() : 0;
}

HistoricalKvCaptureSession::~HistoricalKvCaptureSession() {
    // Tasks retain their own repository/storage/context. Closing admission
    // does not revoke their existing writer or free their installed payload.
    if (repository_ && !domain_) repository_->close();
    if (domain_ && owns_domain_claim_) {
        if (admitted_ && !finished_) domain_->close();
        domain_->request_claimed_ = false;
    }
    if (admitted_) release_unused();
}

void HistoricalKvCaptureSession::release_unused() noexcept {
    if (!reservation_) return;
    if (admitted_) cleanup_ticket_.submit(std::move(reservation_));
    else {
        reservation_.reset();
        cleanup_ticket_ = {};
    }
}

void HistoricalKvCaptureSession::fail() noexcept {
    failed_ = true;
    repository_->close();
    if (domain_ && owns_domain_claim_) {
        domain_->request_claimed_ = false;
        owns_domain_claim_ = false;
    }
    producers_.clear();
    writers_.clear();
    release_unused(); // Only unused private allocations, on the CPU worker.
}

bool HistoricalKvCaptureSession::append(uint32_t tokens) {
    auto & remaining = input_ ? input_remaining_ : output_remaining_;
    if (!admitted_ || failed_ || finished_ || (gpu_window_enabled_ && gpu_window_closed_) ||
        !reservation_ || !producers_.empty() ||
        tokens > remaining || (tokens && !appends_remaining_) ||
        runtime_.store().total_tokens() != expected_rows_ ||
        runtime_.store().block_count() != expected_blocks_) return false;
    if (!tokens) return true;
    const uint32_t block_tokens = storage_->config().max_tokens_per_slot;
    const uint64_t count = (uint64_t(tokens) + block_tokens - 1) / block_tokens;
    if (count > reservation_->ram->slots().remaining()) return false;
    try {
        writers_.clear();
        uint32_t position = expected_rows_;
        uint32_t left = tokens;
        for (uint64_t i = 0; i < count; ++i) {
            const auto index = static_cast<uint32_t>(handles_.size());
            const auto n = std::min(left, block_tokens);
            auto & payload = reservation_->ram->payload(index);
            HistoricalKvRepository::Lease gpu_execution;
            auto writer = gpu_window_enabled_ ?
                repository_->begin_reserved_write_with_gpu(reservation_->ram->slots(),
                    (domain_ ? domain_->next_logical_id_ : uint64_t(expected_blocks_)) + i,
                    position, n, birth_start_ + index, payload.payload_bytes(),
                    std::move(callbacks_[index]), gpu_execution) :
                repository_->begin_reserved_write(reservation_->ram->slots(),
                (domain_ ? domain_->next_logical_id_ : uint64_t(expected_blocks_)) + i,
                position, n, birth_start_ + index,
                payload.payload_bytes(), std::move(callbacks_[index]));
            if (!writer) throw std::runtime_error("capture reserved writer admission failed");
            if (gpu_execution) {
                auto & held = gpu_window_[writer->handle().slot];
                if (held) throw std::logic_error("capture GPU window slot already occupied");
                held = std::move(gpu_execution);
            }
            handles_.push_back(writer->handle());
            writers_.push_back(std::move(*writer));
            if (!storage_->install_payload(writers_.back(), handles_.back(), position, n, payload))
                throw std::runtime_error("capture reserved payload installation failed");
            position += n;
            left -= n;
        }
        if (!runtime_.register_historical_append(repository_, storage_, writers_, producers_,
                domain_ ? domain_->cleanup_task_->metadata.get() : reservation_->metadata.get()))
            throw std::runtime_error("capture span registration failed");
        writers_.clear();
        expected_rows_ = position;
        expected_blocks_ = runtime_.store().block_count();
        if (domain_) domain_->next_logical_id_ += count;
        remaining -= tokens;
        --appends_remaining_;
        return true;
    } catch (...) {
        fail();
        throw;
    }
}

bool HistoricalKvCaptureSession::end_input() noexcept {
    if (!admitted_ || failed_ || finished_ || input_remaining_ || !producers_.empty()) return false;
    input_ = false;
    return true;
}

bool HistoricalKvCaptureSession::finish() {
    return complete(false);
}

bool HistoricalKvCaptureSession::cancel() {
    return complete(true);
}

bool HistoricalKvCaptureSession::complete(bool cancelled) {
    if (!admitted_ || failed_ || finished_ || gpu_window_open() ||
        (!cancelled && input_) || !producers_.empty() ||
        repository_->outstanding_operations()) return false;
    uint32_t live = 0;
    for (const auto handle : handles_) {
        switch (repository_->completion_state(handle)) {
        case HistoricalKvRepository::CompletionState::Unfinished: return false;
        case HistoricalKvRepository::CompletionState::Live: ++live; break;
        case HistoricalKvRepository::CompletionState::Retired: break;
        }
    }
    release_unused();
    if (domain_ && owns_domain_claim_) {
        domain_->request_claimed_ = false;
        owns_domain_claim_ = false;
    }
    live_blocks_at_finish_ = live;
    finished_ = true;
    return true;
}

uint64_t HistoricalKvCaptureSession::next_use_sequence() noexcept {
    return domain_ ? ++domain_->last_use_sequence_ : ++last_use_sequence_;
}

} // namespace kvmem
