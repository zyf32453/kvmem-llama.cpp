#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <memory_resource>
#include <type_traits>
#include <vector>

namespace kvmem {

// Requested allocator bytes, separate from KV payload/SSD accounts and RSS.
// Owners from old domains keep this shared account until actual deallocation.
enum class HistoricalKvAuxKind : size_t { Metadata, Decisions, Policy, Features, Ranges, NativeBridge, Count };

class HistoricalKvAuxBudget final {
public:
    static constexpr size_t kKinds = static_cast<size_t>(HistoricalKvAuxKind::Count);
    struct Stats {
        uint64_t capacity = 0, charged = 0, peak = 0, denied = 0;
        std::array<uint64_t, kKinds> by_kind{};
    };
    // The optional upstream must outlive this account and all its allocations.
    explicit HistoricalKvAuxBudget(uint64_t capacity,
        std::pmr::memory_resource * upstream = std::pmr::new_delete_resource());
    Stats stats() const noexcept;
    void * allocate(size_t bytes, size_t alignment, HistoricalKvAuxKind kind);
    void deallocate(void * pointer, size_t bytes, size_t alignment, HistoricalKvAuxKind kind) noexcept;
private:
    void release(size_t bytes, HistoricalKvAuxKind kind) noexcept;
    const uint64_t capacity_;
    std::pmr::memory_resource * const upstream_;
    std::atomic<uint64_t> charged_{0}, peak_{0}, denied_{0};
    std::array<std::atomic<uint64_t>, kKinds> by_kind_{};
};

// Standard container allocator. Allocation/reallocation charges before calling
// upstream; failures roll back, and refunds follow actual deallocation. These
// paths are for setup/producer/worker code, never decode-boundary commit.
template<class T> class HistoricalKvAuxAllocator {
public:
    using value_type = T;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;
    using propagate_on_container_copy_assignment = std::false_type;
    using is_always_equal = std::false_type;
    HistoricalKvAuxAllocator() noexcept = default;
    HistoricalKvAuxAllocator(std::shared_ptr<HistoricalKvAuxBudget> budget, HistoricalKvAuxKind kind) noexcept
        : budget_(std::move(budget)), kind_(kind) {}
    template<class U> HistoricalKvAuxAllocator(const HistoricalKvAuxAllocator<U> & other) noexcept
        : budget_(other.budget_), kind_(other.kind_) {}
    T * allocate(size_t count) {
        if (count > std::numeric_limits<size_t>::max() / sizeof(T)) throw std::bad_array_new_length();
        if (!count) return nullptr;
        if (!budget_) return std::allocator<T>{}.allocate(count);
        return static_cast<T *>(budget_->allocate(count * sizeof(T), alignof(T), kind_));
    }
    void deallocate(T * pointer, size_t count) noexcept {
        if (!pointer) return;
        if (budget_) budget_->deallocate(pointer, count * sizeof(T), alignof(T), kind_);
        else std::allocator<T>{}.deallocate(pointer, count);
    }
    template<class U> bool operator==(const HistoricalKvAuxAllocator<U> & other) const noexcept {
        return budget_ == other.budget_ && kind_ == other.kind_;
    }
    template<class U> bool operator!=(const HistoricalKvAuxAllocator<U> & other) const noexcept { return !(*this == other); }
private:
    template<class U> friend class HistoricalKvAuxAllocator;
    std::shared_ptr<HistoricalKvAuxBudget> budget_;
    HistoricalKvAuxKind kind_ = HistoricalKvAuxKind::Metadata;
};
template<class T> using HistoricalKvAuxVector = std::vector<T, HistoricalKvAuxAllocator<T>>;

} // namespace kvmem
