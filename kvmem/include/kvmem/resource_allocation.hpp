#pragma once

#include <atomic>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

namespace kvmem {

template<class Tag> struct ResourceAllocationHandle {
    uint64_t arena = 0;
    uint64_t generation = 0;
    int32_t slot = -1;
};

enum class ResourceReleaseResult { Released, AlreadyReleased, Deferred, Retry, Stale, Unsupported };

// Physical allocation rights. Registry mutations are serialized by the tier's
// metadata owner; IO admission/drop use the same atomic phase/count word and
// never take that mutex. A lease must be captured before an async job is queued.
// This protects extent lifetime, not read/write ordering or content validity;
// those remain the submitting owner's responsibility. The tier must outlive IO.
template<class Tag> class ResourceAllocationRegistry {
    static constexpr uint64_t kFree = 0, kBuilding = 1, kLive = 2, kClosed = 3;
    static constexpr uint64_t kPhaseMask = 7, kRef = 8;
    struct Slot {
        std::atomic<uint64_t> control{kFree};
        std::atomic<uint64_t> generation{0};
    };
    struct Shared {
        explicit Shared(uint32_t n) : count(n), slots(new Slot[n]), id(next_id()) {}
        static uint64_t next_id() {
            static std::atomic<uint64_t> counter{1};
            auto n = counter.load();
            for (;;) {
                if (n == std::numeric_limits<uint64_t>::max())
                    throw std::overflow_error("resource arena identity exhausted");
                if (counter.compare_exchange_weak(n, n + 1)) return n;
            }
        }
        uint32_t count;
        std::unique_ptr<Slot[]> slots;
        uint64_t id;
    };
public:
    class IoLease {
    public:
        IoLease() = default;
        IoLease(const IoLease &) = delete;
        IoLease & operator=(const IoLease &) = delete;
        IoLease(IoLease && other) noexcept
            : shared_(std::move(other.shared_)), handle_(other.handle_) {}
        IoLease & operator=(IoLease && other) noexcept {
            if (this != &other) { reset(); shared_ = std::move(other.shared_); handle_ = other.handle_; }
            return *this;
        }
        ~IoLease() { reset(); }
        explicit operator bool() const noexcept { return bool(shared_); }
        ResourceAllocationHandle<Tag> handle() const noexcept { return shared_ ? handle_ : ResourceAllocationHandle<Tag>{}; }
        void reset() noexcept {
            if (shared_) {
                shared_->slots[handle_.slot].control.fetch_sub(kRef, std::memory_order_acq_rel);
                shared_.reset();
            }
        }
    private:
        friend class ResourceAllocationRegistry;
        IoLease(std::shared_ptr<Shared> shared, ResourceAllocationHandle<Tag> handle)
            : shared_(std::move(shared)), handle_(handle) {}
        std::shared_ptr<Shared> shared_;
        ResourceAllocationHandle<Tag> handle_;
    };

    explicit ResourceAllocationRegistry(uint32_t count) : shared_(std::make_shared<Shared>(count)) {}
    ResourceAllocationRegistry(const ResourceAllocationRegistry &) = delete;
    ResourceAllocationRegistry & operator=(const ResourceAllocationRegistry &) = delete;

    // Called only while holding the tier metadata lock, after reserving map
    // storage. No free slot or generation is fabricated on failure.
    ResourceAllocationHandle<Tag> allocate(int32_t index) noexcept {
        if (!valid_index(index)) return {};
        auto & slot = shared_->slots[index];
        const auto generation = slot.generation.load(std::memory_order_acquire);
        if (generation == std::numeric_limits<uint64_t>::max()) return {};
        uint64_t expected = kFree;
        if (!slot.control.compare_exchange_strong(expected, kBuilding, std::memory_order_acq_rel)) return {};
        slot.generation.store(generation + 1, std::memory_order_relaxed);
        slot.control.store(kLive, std::memory_order_release);
        return {shared_->id, generation + 1, index};
    }

    ResourceAllocationHandle<Tag> handle(int32_t index) const noexcept {
        return valid_index(index) ? ResourceAllocationHandle<Tag>{shared_->id,
            shared_->slots[index].generation.load(std::memory_order_acquire), index} : ResourceAllocationHandle<Tag>{};
    }
    bool live(int32_t index) const noexcept {
        return valid_index(index) && (shared_->slots[index].control.load(std::memory_order_acquire) & kPhaseMask) == kLive;
    }
    bool closed(int32_t index) const noexcept {
        return valid_index(index) && (shared_->slots[index].control.load(std::memory_order_acquire) & kPhaseMask) == kClosed;
    }
    IoLease acquire(ResourceAllocationHandle<Tag> h) const noexcept {
        if (h.arena != shared_->id || !h.generation || !valid_index(h.slot)) return {};
        auto & slot = shared_->slots[h.slot];
        auto control = slot.control.load(std::memory_order_acquire);
        for (unsigned attempt = 0; attempt < 16; ++attempt) {
            if ((control & kPhaseMask) != kLive || control > std::numeric_limits<uint64_t>::max() - kRef) return {};
            if (slot.control.compare_exchange_weak(control, control + kRef, std::memory_order_acq_rel)) {
                if (slot.generation.load(std::memory_order_acquire) == h.generation) return IoLease(shared_, h);
                slot.control.fetch_sub(kRef, std::memory_order_acq_rel);
                return {};
            }
        }
        return {};
    }
    bool owns(const IoLease & lease) const noexcept { return lease.shared_.get() == shared_.get(); }

    // Called under the tier metadata lock. Ordinary retirement closes new IO
    // even while old IO is active. Legacy LRU eviction uses idle_only instead:
    // inability to evict must leave the existing allocation selectable.
    ResourceReleaseResult close(ResourceAllocationHandle<Tag> h, bool idle_only = false) noexcept {
        if (!matches(h)) return ResourceReleaseResult::Stale;
        auto & control = shared_->slots[h.slot].control;
        auto old = control.load(std::memory_order_acquire);
        for (unsigned attempt = 0; attempt < 16; ++attempt) {
            const auto phase = old & kPhaseMask;
            if (phase == kFree) return ResourceReleaseResult::AlreadyReleased;
            if (phase == kClosed) return old == kClosed ? ResourceReleaseResult::Released : ResourceReleaseResult::Deferred;
            if (phase != kLive) return ResourceReleaseResult::Retry;
            if (idle_only && old != kLive) return ResourceReleaseResult::Deferred;
            if (control.compare_exchange_weak(old, (old & ~kPhaseMask) | kClosed, std::memory_order_acq_rel))
                return old == kLive ? ResourceReleaseResult::Released : ResourceReleaseResult::Deferred;
        }
        return ResourceReleaseResult::Retry;
    }
    bool finish(ResourceAllocationHandle<Tag> h) noexcept {
        if (!matches(h)) return false;
        uint64_t expected = kClosed;
        return shared_->slots[h.slot].control.compare_exchange_strong(expected, kFree, std::memory_order_acq_rel);
    }
private:
    bool valid_index(int32_t index) const noexcept { return index >= 0 && uint32_t(index) < shared_->count; }
    bool matches(ResourceAllocationHandle<Tag> h) const noexcept {
        return h.arena == shared_->id && h.generation && valid_index(h.slot) &&
            shared_->slots[h.slot].generation.load(std::memory_order_acquire) == h.generation;
    }
    std::shared_ptr<Shared> shared_;
};



} // namespace kvmem
