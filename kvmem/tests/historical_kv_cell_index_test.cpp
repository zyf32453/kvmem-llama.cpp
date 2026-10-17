#include "kvmem/reserved_node_set.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <stdexcept>
#include <set>

static thread_local int64_t allocation_budget = -1;
static thread_local size_t frees = 0;
void * operator new(size_t n) {
    if (!allocation_budget) throw std::bad_alloc();
    if (allocation_budget > 0) --allocation_budget;
    if (void * p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void * operator new[](size_t n) { return ::operator new(n); }
void operator delete(void * p) noexcept { if (p) ++frees; std::free(p); }
void operator delete[](void * p) noexcept { ::operator delete(p); }
void operator delete(void * p, size_t) noexcept { ::operator delete(p); }
void operator delete[](void * p, size_t) noexcept { ::operator delete(p); }
static void check(bool ok, const char * message) { if (!ok) throw std::runtime_error(message); }

template <class Key>
static void compare(const kvmem::ReservedNodeSet<Key> & actual, const std::set<Key> & expected) {
    check(actual.size() == expected.size() && std::equal(actual.begin(), actual.end(), expected.begin()),
          "reserved index differs from std::set");
    check(actual.available() + actual.size() == actual.capacity(), "node credit conservation failed");
}

static void fault_and_reuse() {
    using Index = kvmem::ReservedNodeSet<uint32_t>;
    Index index;
    bool succeeded = false;
    unsigned failures = 0;
    for (int64_t budget = 0; budget < 32; ++budget) {
        allocation_budget = budget;
        try { auto pool = Index::prepare_pool(8); succeeded = pool.capacity() == 8; }
        catch (const std::bad_alloc &) { ++failures; }
        allocation_budget = -1;
        check(index.empty() && !index.reserved(), "private preparation changed the live index");
        if (succeeded) break;
    }
    check(succeeded && failures, "node allocation fault sweep missed success or failure");
    auto pool = Index::prepare_pool(8);
    allocation_budget = 0;
    const bool installed = index.install(pool);
    allocation_budget = -1;
    check(installed && index.capacity() == 8, "node reservation install failed");
    std::set<uint32_t> expected;
    uint32_t random = 137;
    for (unsigned i = 0; i < 512; ++i) {
        random = random * 1664525u + 1013904223u;
        const auto key = (random >> 8) % 12;
        const auto operation = (random >> 24) % 5;
        const bool full = expected.size() == 8 && !expected.count(key);
        const auto before = frees;
        bool exhausted = false;
        allocation_budget = 0;
        try {
            if (operation < 3) index.insert(key);
            else if (operation == 3) index.erase(key);
            else index.clear();
        } catch (const std::bad_alloc &) { exhausted = true; }
        allocation_budget = -1;
        check(frees == before && exhausted == (operation < 3 && full), "reserved mutation allocated/freed or misreported overflow");
        if (operation < 3 && !full) expected.insert(key);
        else if (operation == 3) expected.erase(key);
        else if (operation == 4) expected.clear();
        compare(index, expected);
        const auto a = index.upper_bound(key), b = expected.upper_bound(key);
        check((a == index.end()) == (b == expected.end()) &&
              (a == index.end() || *a == *b), "ordered successor changed");
    }
    index.clear();
    index.insert(1); index.insert(2);
    auto copied = index; // Snapshot creation remains an explicit allocation.
    const auto before = frees;
    allocation_budget = 0;
    copied.clear();
    for (uint32_t i = 0; i < 8; ++i) copied.insert(i);
    const bool refused = !copied.uninstall(pool);
    copied.clear();
    const bool returned = copied.uninstall(pool);
    allocation_budget = -1;
    check(refused && returned && frees == before && pool.capacity() == 8 &&
          !copied.reserved() && index.size() == 2, "copy lost spare capacity or rollback released live nodes");
    auto moved = std::move(index);
    check(moved.reserved() && moved.size() == 2 && index.empty() && !index.reserved(), "move broke index ownership");
    index.insert(99); // Moved-from ordinary behavior is valid.
}

static void repeated_positions_and_zero_capacity() {
    using Index = kvmem::ReservedNodeSet<std::pair<int32_t, uint32_t>>;
    Index index, zero;
    auto pool = Index::prepare_pool(4), empty = Index::prepare_pool(0);
    check(index.install(pool) && zero.install(empty), "pair/zero reservation failed");
    const auto before = frees;
    allocation_budget = 0;
    for (uint32_t i = 0; i < 4; ++i) index.insert({17, i});
    const auto end = index.upper_bound({17, UINT32_MAX});
    bool rejected = false;
    try { zero.insert({0, 0}); } catch (const std::bad_alloc &) { rejected = true; }
    index.clear();
    allocation_budget = -1;
    check(end == index.end() && index.available() == 4 && rejected && frees == before,
          "repeated positions collided or zero reservation grew");
}

int main() try {
    fault_and_reuse(); repeated_positions_and_zero_capacity();
    std::puts("PASS reserved cell indexes: real node faults, 512 operations without allocation/free, ordered lookup, rollback, copy capacity");
} catch (const std::exception & e) {
    allocation_budget = -1;
    std::fprintf(stderr, "FAIL reserved cell indexes: %s\n", e.what());
    return 1;
}
