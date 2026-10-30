#include "kvmem/historical_kv_auxiliary.hpp"
#include "kvmem/memory_contract.hpp"

#include <stdexcept>

namespace kvmem {
HistoricalKvAuxBudget::HistoricalKvAuxBudget(uint64_t capacity, std::pmr::memory_resource * upstream)
    : capacity_(capacity), upstream_(upstream) {
    if (!upstream_) throw std::invalid_argument("null historical auxiliary upstream");
}

void * HistoricalKvAuxBudget::allocate(size_t bytes, size_t alignment, HistoricalKvAuxKind kind) {
    const auto index = static_cast<size_t>(kind);
    if (index >= kKinds || !alignment || (alignment & (alignment - 1)))
        throw std::invalid_argument("invalid historical auxiliary allocation");
    auto observed = charged_.load(std::memory_order_relaxed);
    for (;;) {
        if (bytes > capacity_ || observed > capacity_ - bytes) {
            auto denied = denied_.load(std::memory_order_relaxed);
            while (denied != UINT64_MAX && !denied_.compare_exchange_weak(denied, denied + 1, std::memory_order_relaxed)) {}
            throw MemoryError(MemoryErrorCode::BudgetExceeded, "historical KV auxiliary budget exceeded");
        }
        if (charged_.compare_exchange_weak(observed, observed + bytes, std::memory_order_acq_rel,
                                          std::memory_order_relaxed)) break;
    }
    by_kind_[index].fetch_add(bytes, std::memory_order_relaxed);
    auto peak = peak_.load(std::memory_order_relaxed);
    while (peak < observed + bytes && !peak_.compare_exchange_weak(peak, observed + bytes, std::memory_order_relaxed)) {}
    try { return upstream_->allocate(bytes, alignment); }
    catch (...) { release(bytes, kind); throw; }
}

void HistoricalKvAuxBudget::release(size_t bytes, HistoricalKvAuxKind kind) noexcept {
    by_kind_[static_cast<size_t>(kind)].fetch_sub(bytes, std::memory_order_relaxed);
    charged_.fetch_sub(bytes, std::memory_order_release);
}

void HistoricalKvAuxBudget::deallocate(void * pointer, size_t bytes, size_t alignment, HistoricalKvAuxKind kind) noexcept {
    upstream_->deallocate(pointer, bytes, alignment);
    release(bytes, kind);
}

HistoricalKvAuxBudget::Stats HistoricalKvAuxBudget::stats() const noexcept {
    Stats result;
    result.capacity = capacity_;
    result.charged = charged_.load(std::memory_order_acquire);
    result.peak = peak_.load(std::memory_order_acquire);
    result.denied = denied_.load(std::memory_order_relaxed);
    for (size_t i = 0; i < kKinds; ++i) result.by_kind[i] = by_kind_[i].load(std::memory_order_relaxed);
    // Category counters are diagnostic samples, not one concurrent transaction.
    return result;
}
} // namespace kvmem
