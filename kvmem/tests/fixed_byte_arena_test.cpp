#include "kvmem/fixed_byte_arena.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <vector>

static thread_local bool deny_new = false;
static thread_local size_t frees = 0;
void * operator new(size_t n) {
    if (deny_new) throw std::bad_alloc();
    if (void * p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void * operator new[](size_t n) { return ::operator new(n); }
void operator delete(void * p) noexcept { if (p) ++frees; std::free(p); }
void operator delete[](void * p) noexcept { ::operator delete(p); }
void operator delete(void * p, size_t) noexcept { ::operator delete(p); }
void operator delete[](void * p, size_t) noexcept { ::operator delete(p); }
static void check(bool ok, const char * message) { if (!ok) throw std::runtime_error(message); }

int main() try {
    kvmem::FixedByteArena arena(4096);
    using Vector = std::vector<uint64_t, kvmem::FixedArenaAllocator<uint64_t>>;
    Vector positions{kvmem::FixedArenaAllocator<uint64_t>(&arena)};
    Vector copy, moved;
    const auto before = frees;
    deny_new = true;
    for (uint64_t i = 0; i < 64; ++i) positions.push_back(i * 3);
    copy = positions;
    moved = std::move(copy);
    positions.clear();
    bool refused = false;
    try { (void) arena.allocate(4096, alignof(uint64_t)); }
    catch (const std::bad_alloc &) { refused = true; }
    deny_new = false;
    check(refused && moved.size() == 64 && moved[63] == 189 &&
          moved.get_allocator().arena() == &arena && frees == before &&
          arena.used_bytes() <= arena.capacity_bytes(),
          "fixed arena copy/move grew the heap or lost bounded storage");
    std::puts("PASS fixed byte arena: real storage, no C++ allocation or free during vector growth/copy/move, bounded exhaustion");
} catch (const std::exception & e) {
    deny_new = false;
    std::fprintf(stderr, "FAIL fixed byte arena: %s\n", e.what());
    return 1;
}
