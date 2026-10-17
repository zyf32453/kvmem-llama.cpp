#pragma once

#include <cstddef>
#include <new>
#include <set>
#include <utility>
#include <vector>

namespace kvmem {

// Ordered index with an opt-in, real node reservation. Ordinary instances
// retain std::set behavior. Reserved instances never allocate/free nodes on
// insert/erase/clear; exhausted capacity throws instead of growing. Copying
// an entire index allocates and is not an admitted-operation API.
template <class Key>
class ReservedNodeSet {
    using Set = std::set<Key>;
public:
    using const_iterator = typename Set::const_iterator;
    struct Pool {
        Pool() = default;
        Pool(Pool &&) noexcept = default;
        Pool & operator=(Pool &&) noexcept = default;
        Pool(const Pool &) = delete;
        Pool & operator=(const Pool &) = delete;
        size_t capacity() const noexcept { return limit; }
    private:
        friend class ReservedNodeSet;
        std::vector<typename Set::node_type> nodes;
        size_t limit = 0;
    };

    ReservedNodeSet() = default;
    ReservedNodeSet(const ReservedNodeSet & other) : live_(other.live_) {
        if (other.reserved_) {
            auto pool = prepare_pool(other.limit_, other.limit_ - live_.size());
            spare_.swap(pool.nodes);
            limit_ = other.limit_;
            reserved_ = true;
        }
    }
    ReservedNodeSet(ReservedNodeSet && other) { swap(other); }
    ReservedNodeSet & operator=(const ReservedNodeSet & other) {
        if (this != &other) { ReservedNodeSet copy(other); swap(copy); }
        return *this;
    }
    ReservedNodeSet & operator=(ReservedNodeSet && other) noexcept {
        if (this != &other) swap(other);
        return *this;
    }
    void swap(ReservedNodeSet & other) noexcept {
        live_.swap(other.live_); spare_.swap(other.spare_);
        std::swap(limit_, other.limit_); std::swap(reserved_, other.reserved_);
    }

    static Pool prepare_pool(size_t capacity) { return prepare_pool(capacity, capacity); }
    bool can_install() const noexcept { return !reserved_ && live_.empty(); }
    bool install(Pool & pool) noexcept {
        if (!can_install() || pool.nodes.size() != pool.limit) return false;
        spare_.swap(pool.nodes);
        limit_ = pool.limit; pool.limit = 0;
        reserved_ = true;
        return true;
    }
    // Preparation rollback only: no admitted cells may remain. The output
    // pool owns the detached allocations; its destruction returns them.
    bool uninstall(Pool & pool) noexcept {
        if (!reserved_ || !live_.empty() || !pool.nodes.empty() ||
            spare_.size() != limit_) return false;
        spare_.swap(pool.nodes);
        pool.limit = limit_; limit_ = 0;
        reserved_ = false;
        return true;
    }
    bool reserved() const noexcept { return reserved_; }
    size_t capacity() const noexcept { return limit_; }
    size_t available() const noexcept { return spare_.size(); }
    size_t size() const noexcept { return live_.size(); }
    bool empty() const noexcept { return live_.empty(); }
    auto begin() const noexcept { return live_.begin(); }
    auto end() const noexcept { return live_.end(); }
    auto rbegin() const noexcept { return live_.rbegin(); }
    auto upper_bound(const Key & key) const { return live_.upper_bound(key); }

    std::pair<const_iterator, bool> insert(const Key & key) {
        if (!reserved_) return live_.insert(key);
        const auto found = live_.find(key);
        if (found != live_.end()) return {found, false};
        if (spare_.empty()) throw std::bad_alloc();
        auto node = std::move(spare_.back());
        spare_.pop_back();
        try {
            node.value() = key;
            auto result = live_.insert(std::move(node));
            if (!result.inserted) spare_.push_back(std::move(result.node));
            return {result.position, result.inserted};
        } catch (...) {
            if (node) spare_.push_back(std::move(node));
            throw;
        }
    }
    size_t erase(const Key & key) {
        if (!reserved_) return live_.erase(key);
        const auto found = live_.find(key);
        if (found == live_.end()) return 0;
        spare_.push_back(live_.extract(found));
        return 1;
    }
    void clear() noexcept {
        if (!reserved_) { live_.clear(); return; }
        while (!live_.empty()) spare_.push_back(live_.extract(live_.begin()));
    }

private:
    static Pool prepare_pool(size_t capacity, size_t unused) {
        Pool pool;
        pool.nodes.reserve(capacity);
        Set source;
        for (size_t i = 0; i < unused; ++i) {
            const auto node = source.insert(Key{}).first;
            pool.nodes.push_back(source.extract(node));
        }
        pool.limit = capacity;
        return pool;
    }
    Set live_;
    std::vector<typename Set::node_type> spare_;
    size_t limit_ = 0;
    bool reserved_ = false;
};

} // namespace kvmem
