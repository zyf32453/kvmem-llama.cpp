#include "kvmem/historical_kv_cleanup.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <new>
#include <stdexcept>
#include <vector>

static thread_local bool forbid_allocation = false;
static thread_local uint64_t frees = 0;
void * operator new(size_t n) {
    if (forbid_allocation) throw std::bad_alloc();
    if (auto * p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void * operator new[](size_t n) { return ::operator new(n); }
void operator delete(void * p) noexcept { if (p) ++frees; std::free(p); }
void operator delete[](void * p) noexcept { ::operator delete(p); }
void operator delete(void * p, size_t) noexcept { ::operator delete(p); }
void operator delete[](void * p, size_t) noexcept { ::operator delete(p); }

using kvmem::HistoricalKvCleanup;
static void check(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}
template<class Predicate> static void until(Predicate predicate) {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!predicate() && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
    check(predicate(), "cleanup worker did not progress");
}
struct Probe final : HistoricalKvCleanup::Task {
    Probe(std::atomic<unsigned> & count, std::atomic<bool> & gate)
        : count(count), gate(gate) {}
    std::atomic<unsigned> & count;
    std::atomic<bool> & gate;
    bool ready() const noexcept override { return gate.load(std::memory_order_acquire); }
    ~Probe() override { count.fetch_add(1, std::memory_order_release); }
};

static void bounds_publication_and_fairness() {
    std::atomic<unsigned> first{0}, second{0};
    std::atomic<bool> closed{false}, open{true};
    HistoricalKvCleanup service(2);
    // Always release a queued gate before service shutdown on test failure.
    struct Open { std::atomic<bool> & gate; ~Open() { gate = true; } } unblock{closed};
    auto a = service.try_reserve(), b = service.try_reserve();
    check(a && b && !service.try_reserve() && service.occupied() == 2, "capacity was not reserved");
    auto x = std::make_unique<Probe>(first, closed);
    auto y = std::make_unique<Probe>(second, open);
    const auto before = frees;
    forbid_allocation = true;
    a->submit(std::move(x));
    b->submit(std::move(y));
    forbid_allocation = false;
    check(frees == before && !*a && !*b, "publication freed resources on caller or retained ticket");
    until([&] { return second.load(std::memory_order_acquire) == 1 && service.occupied() == 1; });
    check(first == 0, "not-ready task destroyed");
    auto reused = service.try_reserve();
    check(reused && !service.try_reserve(), "not-ready cell was reused or ready task starved");
    // Cancelling an unused cell performs no work and returns exact credit.
    reused.reset();
    check(service.occupied() == 1, "cancel lost reserved credit");
    closed.store(true, std::memory_order_release);
    until([&] { return service.occupied() == 0; });
    check(first == 1 && second == 1, "task destroyed more than once");
}

static void concurrent_reuse() {
    std::atomic<unsigned> destroyed{0};
    std::atomic<bool> ready{true};
    HistoricalKvCleanup service(4);
    std::vector<std::thread> producers;
    for (unsigned thread = 0; thread < 4; ++thread) producers.emplace_back([&] {
        for (unsigned i = 0; i < 80; ++i) {
            auto task = std::make_unique<Probe>(destroyed, ready);
            std::optional<HistoricalKvCleanup::Ticket> ticket;
            while (!(ticket = service.try_reserve())) std::this_thread::yield();
            forbid_allocation = true;
            ticket->submit(std::move(task));
            forbid_allocation = false;
        }
    });
    for (auto & producer : producers) producer.join();
    until([&] { return service.occupied() == 0; });
    check(destroyed == 320, "concurrent producers lost or repeated a cleanup");
}

int main() {
    try {
        bounds_publication_and_fairness();
        concurrent_reuse();
        std::puts("PASS historical cleanup: bounded admission, no foreground allocation/free, readiness isolation, 320 concurrent handoffs");
    } catch (const std::exception & e) {
        forbid_allocation = false;
        std::fprintf(stderr, "FAIL historical cleanup: %s\n", e.what());
        return 1;
    }
}
