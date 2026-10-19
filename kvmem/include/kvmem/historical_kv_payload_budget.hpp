#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

namespace kvmem {

// Shared across old/new historical stores. Counts allocator-requested K/V
// and Mean-K capacity, including private reservations and partial failures.
// It does not count allocator overhead, metadata, staging or model buffers.
// Acquiring credit never waits for the GC. The caller rejects admission if
// the bounded attempt fails; already admitted payloads need no new credit.
class HistoricalKvPayloadBudget {
public:
    explicit HistoricalKvPayloadBudget(uint64_t limit) noexcept : limit_(limit) {}
    uint64_t limit_bytes() const noexcept { return limit_; }
    uint64_t used_bytes() const noexcept { return used_.load(std::memory_order_acquire); }
    uint64_t refusals() const noexcept { return refusals_.load(std::memory_order_relaxed); }

private:
    template <class T> friend class HistoricalKvPayloadAllocator;
    bool try_acquire(uint64_t bytes) noexcept {
        auto used = used_.load(std::memory_order_relaxed);
        for (unsigned attempt = 0; attempt < 32; ++attempt) {
            if (bytes > limit_ - used) break;
            if (used_.compare_exchange_strong(used, used + bytes,
                    std::memory_order_acq_rel, std::memory_order_relaxed)) return true;
        }
        refusals_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    void release(uint64_t bytes) noexcept {
        if (used_.fetch_sub(bytes, std::memory_order_release) < bytes) std::terminate();
    }
    static_assert(std::atomic<uint64_t>::is_always_lock_free,
                  "historical payload budget requires lock-free accounting");
    const uint64_t limit_;
    std::atomic<uint64_t> used_{0}, refusals_{0};
};

// Charge the actual capacity requested by vector before allocation, not an
// estimate measured after allocating. Keep the shared owner through moves,
// GC's swap-to-local disposal and private reservation rollback. Copies into
// public LayerSnapshot buffers use the caller's ordinary allocator instead.
template <class T> class HistoricalKvPayloadAllocator {
public:
    using value_type = T;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;
    using is_always_equal = std::false_type;

    HistoricalKvPayloadAllocator() noexcept = default;
    explicit HistoricalKvPayloadAllocator(std::shared_ptr<HistoricalKvPayloadBudget> budget) noexcept
        : budget_(std::move(budget)) {}
    template <class U>
    HistoricalKvPayloadAllocator(const HistoricalKvPayloadAllocator<U> & other) noexcept
        : budget_(other.budget()) {}

    T * allocate(size_t count) {
        if (count > std::numeric_limits<size_t>::max() / sizeof(T)) throw std::bad_alloc();
        const auto bytes = count * sizeof(T);
        if (budget_ && !budget_->try_acquire(bytes)) throw std::bad_alloc();
        try { return std::allocator<T>{}.allocate(count); }
        catch (...) {
            if (budget_) budget_->release(bytes);
            throw;
        }
    }
    void deallocate(T * ptr, size_t count) noexcept {
        std::allocator<T>{}.deallocate(ptr, count);
        // Actual deallocation must finish before another domain can spend it.
        if (budget_) budget_->release(count * sizeof(T));
    }
    const std::shared_ptr<HistoricalKvPayloadBudget> & budget() const noexcept { return budget_; }
    template <class U> bool operator==(const HistoricalKvPayloadAllocator<U> & other) const noexcept {
        return budget_ == other.budget();
    }
    template <class U> bool operator!=(const HistoricalKvPayloadAllocator<U> & other) const noexcept {
        return !(*this == other);
    }

private:
    std::shared_ptr<HistoricalKvPayloadBudget> budget_;
};

} // namespace kvmem
