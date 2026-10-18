#pragma once

#include <cstddef>
#include <limits>
#include <memory>
#include <new>
#include <type_traits>
#include <vector>

namespace kvmem {

// Fixed, request-owned storage. All bytes are acquired before admission;
// exhaustion is a hard failure rather than an allocation from the process.
// Individual allocations are not recycled. The owner must outlive its users.
class FixedByteArena {
public:
    explicit FixedByteArena(size_t bytes) {
        constexpr size_t unit = sizeof(std::max_align_t);
        if (bytes > std::numeric_limits<size_t>::max() - (unit - 1)) throw std::bad_alloc();
        storage_.resize((bytes + unit - 1) / unit);
    }
    FixedByteArena(const FixedByteArena &) = delete;
    FixedByteArena & operator=(const FixedByteArena &) = delete;

    void * allocate(size_t bytes, size_t alignment) {
        if (!alignment || alignment > alignof(std::max_align_t) ||
            (alignment & (alignment - 1))) throw std::bad_alloc();
        if (!bytes) return nullptr;
        const size_t offset = (used_ + alignment - 1) & ~(alignment - 1);
        if (offset < used_ || offset > capacity_bytes() ||
            bytes > capacity_bytes() - offset) throw std::bad_alloc();
        used_ = offset + bytes;
        return static_cast<unsigned char *>(static_cast<void *>(storage_.data())) + offset;
    }
    size_t capacity_bytes() const noexcept { return storage_.size() * sizeof(std::max_align_t); }
    size_t used_bytes() const noexcept { return used_; }
    // The serialized owner may reuse the arena only after all owning users
    // (including empty containers with retained buffers) have been destroyed.
    // Every such user must hold a shared owner, as CaptureBatch does.
    static bool reset_exclusive(const std::shared_ptr<FixedByteArena> & owner) noexcept {
        if (!owner || owner.use_count() != 1) return false;
        owner->used_ = 0;
        return true;
    }

private:
    std::vector<std::max_align_t> storage_;
    size_t used_ = 0;
};

template <class T>
class FixedArenaAllocator {
public:
    using value_type = T;
    using propagate_on_container_copy_assignment = std::true_type;
    using propagate_on_container_move_assignment = std::true_type;
    using propagate_on_container_swap = std::true_type;
    using is_always_equal = std::false_type;

    FixedArenaAllocator() noexcept = default;
    explicit FixedArenaAllocator(FixedByteArena * arena) noexcept : arena_(arena) {}
    template <class U>
    FixedArenaAllocator(const FixedArenaAllocator<U> & other) noexcept : arena_(other.arena()) {}
    T * allocate(size_t count) {
        if (count > std::numeric_limits<size_t>::max() / sizeof(T)) throw std::bad_alloc();
        if (!arena_) return std::allocator<T>{}.allocate(count);
        return static_cast<T *>(arena_->allocate(count * sizeof(T), alignof(T)));
    }
    void deallocate(T * ptr, size_t count) noexcept {
        if (!arena_) std::allocator<T>{}.deallocate(ptr, count);
    }
    FixedByteArena * arena() const noexcept { return arena_; }

    template <class U>
    bool operator==(const FixedArenaAllocator<U> & other) const noexcept { return arena_ == other.arena(); }
    template <class U>
    bool operator!=(const FixedArenaAllocator<U> & other) const noexcept { return !(*this == other); }

private:
    template <class> friend class FixedArenaAllocator;
    FixedByteArena * arena_ = nullptr;
};

} // namespace kvmem
