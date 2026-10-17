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
    if (!repository) throw std::invalid_argument("historical KV active repository is required");

    AcquireResult result;
    auto active = std::unique_ptr<HistoricalKvActiveSet>(
        new HistoricalKvActiveSet(std::move(repository)));
    active->leases_.reserve(requested.size());
    active->handles_.reserve(requested.size());

    for (const auto & item : requested) {
        const bool admitted = std::any_of(active->handles_.begin(), active->handles_.end(),
            [&](HistoricalKvHandle handle) { return same_handle(handle, item.handle); });
        if (admitted) continue;

        auto lease = item.gpu_ready_writing ?
            active->repository_->acquire_gpu_ready_writing(item.handle) :
            active->repository_->acquire_new(item.handle);
        if (!lease) {
            if (!item.mandatory) continue;
            // Destroying active drops every lease obtained earlier in this
            // attempt. No partial active window escapes to the caller.
            result.missing_mandatory = item.handle;
            return result;
        }
        active->handles_.push_back(item.handle);
        active->leases_.push_back(std::move(*lease));
    }

    result.active = std::move(active);
    return result;
}

bool HistoricalKvActiveSet::mark_execution_submitted(uint64_t sequence) {
    bool all_recorded = true;
    for (const auto & lease : leases_)
        all_recorded = repository_->mark_executed(lease, sequence) && all_recorded;
    return all_recorded;
}

} // namespace kvmem
