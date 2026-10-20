#include "kvmem/historical_kv_active_set.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace kvmem {
namespace {

bool same_handle(HistoricalKvHandle left, HistoricalKvHandle right) {
    return left.incarnation == right.incarnation && left.slot == right.slot &&
           left.generation == right.generation;
}

} // namespace

HistoricalKvActiveSet::HistoricalKvActiveSet(
    std::shared_ptr<HistoricalKvRepository> repository)
    : repository_(std::move(repository)) {}

HistoricalKvActiveSet::AcquireResult HistoricalKvActiveSet::acquire(
    std::shared_ptr<HistoricalKvRepository> repository,
    const std::vector<HistoricalKvActiveRequest> & requested) {
    AcquireResult result;
    auto active = prepare(std::move(repository), requested.size());
    auto acquired = active->try_acquire(requested);
    result.missing_mandatory = acquired.missing_mandatory;
    if (acquired) result.active = std::move(active);
    return result;
}

std::unique_ptr<HistoricalKvActiveSet> HistoricalKvActiveSet::prepare(
        std::shared_ptr<HistoricalKvRepository> repository, size_t max_requested) {
    if (!repository) throw std::invalid_argument("historical KV active repository is required");
    auto active = std::unique_ptr<HistoricalKvActiveSet>(new HistoricalKvActiveSet(std::move(repository)));
    active->leases_.reserve(max_requested);
    active->handles_.reserve(max_requested);
    active->max_requested_ = max_requested;
    return active;
}

HistoricalKvActiveSet::TryAcquireResult HistoricalKvActiveSet::try_acquire(
        const std::vector<HistoricalKvActiveRequest> & requested) {
    if (engaged_) return {AcquireStatus::Busy, {}};
    if (requested.size() > max_requested_) return {AcquireStatus::CapacityExceeded, {}};
    if (repository_->closed()) return {AcquireStatus::Closed, {}};

    // Clear partially acquired permissions even if a repository operation
    // throws. Busy/capacity rejection above must preserve an older operation.
    struct Rollback {
        HistoricalKvActiveSet & active;
        ~Rollback() { if (!active.engaged()) active.release(); }
    } rollback{*this};

    for (const auto & item : requested) {
        const bool admitted = std::any_of(handles_.begin(), handles_.end(),
            [&](HistoricalKvHandle handle) { return same_handle(handle, item.handle); });
        if (admitted) continue;

        // Duplicate requests must retain the strongest resource dependency,
        // even if the GPU-only occurrence was listed first.
        bool host = false, mandatory = false, gpu_ready_writing = false;
        for (const auto & need : requested) {
            if (!same_handle(need.handle, item.handle)) continue;
            host |= !need.gpu_only && !need.gpu_ready_writing;
            mandatory |= need.mandatory;
            gpu_ready_writing |= need.gpu_ready_writing;
        }
        auto lease = host ? repository_->acquire_new(item.handle) :
            gpu_ready_writing ? repository_->acquire_gpu_ready_writing(item.handle) :
            repository_->acquire_gpu(item.handle);
        if (!lease) {
            if (!mandatory) continue;
            return {AcquireStatus::MissingMandatory, item.handle};
        }
        handles_.push_back(item.handle);
        leases_.push_back(std::move(*lease));
    }

    engaged_ = true; // Even an empty selection is an admitted active operation.
    return {AcquireStatus::Acquired, {}};
}

void HistoricalKvActiveSet::release() noexcept {
    leases_.clear();
    handles_.clear();
    engaged_ = false;
}

bool HistoricalKvActiveSet::contains(HistoricalKvHandle handle) const noexcept {
    return engaged_ && std::any_of(handles_.begin(), handles_.end(),
        [&](HistoricalKvHandle held) { return same_handle(handle, held); });
}

bool HistoricalKvActiveSet::mark_execution_submitted(HistoricalKvHandle handle, uint64_t sequence) {
    if (!engaged_) return false;
    for (const auto & lease : leases_)
        if (same_handle(lease.handle(), handle)) return repository_->mark_executed(lease, sequence);
    return false;
}

bool HistoricalKvActiveSet::mark_execution_submitted(uint64_t sequence) {
    if (!engaged_) return false;
    bool all_recorded = true;
    for (const auto & lease : leases_)
        all_recorded = repository_->mark_executed(lease, sequence) && all_recorded;
    return all_recorded;
}

} // namespace kvmem
