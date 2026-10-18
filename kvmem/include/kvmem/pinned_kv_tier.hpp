#pragma once

// Slot bookkeeping for a CPU cache. The caller owns the actual host arena and
// must keep it alive through every copy. Allocation leases protect slot reuse,
// not content ordering: publish/cache version checks belong to that owner.
#include "kvmem/resource_allocation.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace kvmem {

struct PinnedAllocationTag;
using PinnedAllocationHandle = ResourceAllocationHandle<PinnedAllocationTag>;
using PinnedAllocationRegistry = ResourceAllocationRegistry<PinnedAllocationTag>;
using PinnedIoLease = PinnedAllocationRegistry::IoLease;
using PinnedReleaseResult = ResourceReleaseResult;

enum class PinnedKvCachePolicy : uint8_t { LegacyLru = 0, HeatAware = 1 };
struct PinnedKvTierConfig {
    uint64_t total_bytes = 0;
    uint64_t slot_bytes = 0;
    PinnedKvCachePolicy cache_policy = PinnedKvCachePolicy::LegacyLru;
    uint32_t heat_half_life_epochs = 8;
    // Historical owners use exact allocations, without a block-ID cache/LRU.
    bool handle_addressed = false;
};
struct PinnedSlotPlacement {
    int32_t slot = -1;
    int32_t evicted_block = -1;
    PinnedAllocationHandle allocation;
};
struct PinnedKvCacheStats {
    uint64_t selection_epochs = 0, admissions = 0, admission_rejections = 0, evictions = 0;
};

class PinnedKvTier {
public:
    explicit PinnedKvTier(PinnedKvTierConfig cfg)
        : cfg_(cfg), slot_count_(count_slots(cfg)), allocations_(slot_count_), owners_(slot_count_, -1) {
        free_slots_.reserve(slot_count_);
        lru_.reserve(slot_count_);
        block_to_slot_.reserve(cfg_.handle_addressed ? 0 : slot_count_);
        for (uint32_t i = slot_count_; i > 0; --i) free_slots_.push_back(int32_t(i - 1));
    }
    const PinnedKvTierConfig & config() const { return cfg_; }
    bool enabled() const { return slot_count_ > 0; }
    uint32_t slot_count() const { return slot_count_; }
    uint32_t free_slots() const { std::lock_guard<std::mutex> lock(mu_); return uint32_t(free_slots_.size()); }
    uint32_t used_slots() const { return slot_count_ - free_slots(); }
    PinnedKvCachePolicy cache_policy() const { return cfg_.cache_policy; }
    PinnedKvCacheStats stats() const { std::lock_guard<std::mutex> lock(mu_); return stats_; }
    uint64_t slot_offset(int32_t slot) const {
        if (slot < 0 || uint32_t(slot) >= slot_count_) throw std::out_of_range("invalid pinned cache slot");
        return uint64_t(slot) * cfg_.slot_bytes;
    }
    int32_t block_slot(uint32_t id) const {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = block_to_slot_.find(id);
        return it != block_to_slot_.end() && allocations_.live(it->second) ? it->second : -1;
    }
    PinnedAllocationHandle allocation_for_block(uint32_t id) const {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = block_to_slot_.find(id);
        return it != block_to_slot_.end() && allocations_.live(it->second) ? allocations_.handle(it->second) : PinnedAllocationHandle{};
    }
    PinnedIoLease acquire_io(PinnedAllocationHandle h) const noexcept { return allocations_.acquire(h); }
    bool owns(const PinnedIoLease & lease) const noexcept { return allocations_.owns(lease); }

    // Nonblocking exact allocation/release entry points for historical owners.
    // Closing prevents new IO immediately; Deferred never advertises free space.
    PinnedSlotPlacement try_allocate() {
        std::unique_lock<std::mutex> lock(mu_, std::try_to_lock);
        if (!lock.owns_lock() || !cfg_.handle_addressed || free_slots_.empty()) return {};
        const auto slot = free_slots_.back();
        const auto h = allocations_.allocate(slot);
        if (h.slot < 0) return {};
        free_slots_.pop_back();
        return {slot, -1, h};
    }
    PinnedReleaseResult try_release_allocation(PinnedAllocationHandle h) {
        std::unique_lock<std::mutex> lock(mu_, std::try_to_lock);
        if (!lock.owns_lock()) return PinnedReleaseResult::Retry;
        const auto result = allocations_.close(h);
        return result == PinnedReleaseResult::Released ? finish_locked(h) : result;
    }
    // Request rollback/normal owner boundary; GC uses the try-lock entry above.
    PinnedReleaseResult release_owned_allocation(PinnedAllocationHandle h) {
        std::lock_guard<std::mutex> lock(mu_);
        const auto result = allocations_.close(h);
        return result == PinnedReleaseResult::Released ? finish_locked(h) : result;
    }
    PinnedSlotPlacement place_block(uint32_t id) {
        std::lock_guard<std::mutex> lock(mu_);
        require_legacy(id);
        return place_locked(id);
    }
    PinnedSlotPlacement place_block_evicting(uint32_t id) {
        std::lock_guard<std::mutex> lock(mu_);
        require_legacy(id);
        if (block_to_slot_.count(id) || !free_slots_.empty()) return place_locked(id, true);
        return replace_locked(id);
    }
    PinnedSlotPlacement place_block_replacing(uint32_t id) {
        std::lock_guard<std::mutex> lock(mu_);
        require_legacy(id);
        return block_to_slot_.count(id) ? place_locked(id, true) : replace_locked(id);
    }
    void release_block(uint32_t id) {
        std::lock_guard<std::mutex> lock(mu_);
        require_legacy(id);
        auto it = block_to_slot_.find(id);
        if (it == block_to_slot_.end()) return;
        const auto h = allocations_.handle(it->second);
        if (allocations_.close(h) == PinnedReleaseResult::Released) finish_locked(h);
    }
    // The owner periodically acknowledges closed allocations after its old
    // copies have drained. This touches metadata only; the arena stays allocated.
    void collect_released() {
        std::lock_guard<std::mutex> lock(mu_);
        for (uint32_t i = 0; i < slot_count_; ++i) {
            if (allocations_.closed(int32_t(i))) {
                const auto h = allocations_.handle(int32_t(i));
                if (allocations_.close(h) == PinnedReleaseResult::Released) finish_locked(h);
            }
        }
    }
    void begin_selection_epoch() {
        std::lock_guard<std::mutex> lock(mu_);
        require_legacy(0);
        ++selection_epoch_; ++stats_.selection_epochs;
    }
    void record_selected(uint32_t id) {
        std::lock_guard<std::mutex> lock(mu_);
        require_legacy(id);
        Heat & h = heat_[id];
        decay(h);
        h.consecutive = h.last_epoch + 1 == selection_epoch_ ? h.consecutive + 1 : 1;
        h.frequency += 1.0;
        h.last_epoch = selection_epoch_;
        auto it = block_to_slot_.find(id);
        if (it != block_to_slot_.end() && allocations_.live(it->second)) touch_locked(id);
    }
    void touch(uint32_t id) {
        std::lock_guard<std::mutex> lock(mu_);
        require_legacy(id);
        auto it = block_to_slot_.find(id);
        if (it != block_to_slot_.end() && allocations_.live(it->second)) touch_locked(id);
    }
    int32_t lru_victim() const {
        std::lock_guard<std::mutex> lock(mu_);
        for (auto id : lru_) if (allocations_.live(block_to_slot_.at(id))) return int32_t(id);
        return -1;
    }
    // Reset closes all generations, including in-flight copies. Old IO can
    // finish, but its slot is unavailable until explicitly collected/released.
    void clear() {
        std::lock_guard<std::mutex> lock(mu_);
        for (uint32_t i = 0; i < slot_count_; ++i) {
            const auto h = allocations_.handle(int32_t(i));
            if (allocations_.close(h) == PinnedReleaseResult::Released) finish_locked(h);
        }
        std::sort(free_slots_.begin(), free_slots_.end(), std::greater<int32_t>());
        heat_.clear(); selection_epoch_ = 0; stats_ = {};
    }
private:
    static uint32_t count_slots(const PinnedKvTierConfig & cfg) {
        const auto count = cfg.slot_bytes ? cfg.total_bytes / cfg.slot_bytes : 0;
        if (count > uint64_t(INT32_MAX)) throw std::invalid_argument("too many pinned cache slots");
        return uint32_t(count);
    }
    void require_legacy(uint32_t id) const {
        if (cfg_.handle_addressed) throw std::logic_error("exact pinned pool has no block-ID cache operations");
        if (id > uint32_t(INT32_MAX)) throw std::invalid_argument("pinned cache block ID exceeds signed range");
    }
    PinnedSlotPlacement place_locked(uint32_t id, bool touch_existing = false) {
        auto it = block_to_slot_.find(id);
        if (it != block_to_slot_.end()) {
            if (!allocations_.live(it->second)) return {};
            if (touch_existing) touch_locked(id);
            return {it->second, -1, allocations_.handle(it->second)};
        }
        if (free_slots_.empty()) return {};
        const int32_t slot = free_slots_.back();
        // Allocate the map node before consuming physical capacity.
        const auto entry = block_to_slot_.emplace(id, slot);
        const auto h = allocations_.allocate(slot);
        if (h.slot < 0) { block_to_slot_.erase(entry.first); return {}; }
        owners_[slot] = id;
        free_slots_.pop_back();
        touch_locked(id);
        ++stats_.admissions;
        return {slot, -1, h};
    }
    PinnedSlotPlacement replace_locked(uint32_t id) {
        // Prepare the only fallible mutation before closing any victim.
        const auto candidate = block_to_slot_.emplace(id, -1);
        for (auto victim : lru_) {
            auto it = block_to_slot_.find(victim);
            const int32_t slot = it->second;
            if (!allocations_.live(slot)) continue;
            // Preserve the existing heat admission decision for the first
            // eligible LRU candidate; IO safety must not silently retune it.
            if (cfg_.cache_policy == PinnedKvCachePolicy::HeatAware && effective_heat(id) < effective_heat(victim)) break;
            const auto old = allocations_.handle(slot);
            if (old.generation == UINT64_MAX || allocations_.close(old, true) != PinnedReleaseResult::Released) continue;
            if (!allocations_.finish(old)) std::terminate();
            const auto fresh = allocations_.allocate(slot);
            if (fresh.slot < 0) std::terminate();
            block_to_slot_.erase(it);
            candidate.first->second = slot;
            owners_[slot] = id;
            erase_lru(victim); touch_locked(id);
            ++stats_.admissions; ++stats_.evictions;
            return {slot, int32_t(victim), fresh};
        }
        block_to_slot_.erase(candidate.first);
        ++stats_.admission_rejections;
        return {};
    }
    PinnedReleaseResult finish_locked(PinnedAllocationHandle h) {
        if (!allocations_.finish(h)) return PinnedReleaseResult::Retry;
        if (owners_[h.slot] >= 0) {
            const auto id = uint32_t(owners_[h.slot]);
            block_to_slot_.erase(id); erase_lru(id); owners_[h.slot] = -1;
        }
        free_slots_.push_back(h.slot);
        return PinnedReleaseResult::Released;
    }
    struct Heat { double frequency = 0.0; uint64_t last_epoch = 0; uint32_t consecutive = 0; };
    void decay(Heat & h) const {
        if (h.frequency <= 0 || selection_epoch_ <= h.last_epoch || !cfg_.heat_half_life_epochs) return;
        h.frequency *= std::exp2(-double(selection_epoch_ - h.last_epoch) / cfg_.heat_half_life_epochs);
        if (h.frequency < 0.01) h.frequency = 0;
    }
    double effective_heat(uint32_t id) const {
        auto it = heat_.find(id);
        if (it == heat_.end()) return 0;
        auto h = it->second; decay(h);
        return h.frequency + 0.25 * h.consecutive;
    }
    void erase_lru(uint32_t id) { auto it = std::find(lru_.begin(), lru_.end(), id); if (it != lru_.end()) lru_.erase(it); }
    void touch_locked(uint32_t id) { erase_lru(id); lru_.push_back(id); }
    const PinnedKvTierConfig cfg_;
    const uint32_t slot_count_;
    mutable std::mutex mu_;
    PinnedAllocationRegistry allocations_;
    std::vector<int64_t> owners_;
    std::vector<int32_t> free_slots_;
    std::unordered_map<uint32_t, int32_t> block_to_slot_;
    std::vector<uint32_t> lru_;
    std::unordered_map<uint32_t, Heat> heat_;
    uint64_t selection_epoch_ = 0;
    PinnedKvCacheStats stats_;
};
} // namespace kvmem
