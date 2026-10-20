// Standalone toolchain probe, intentionally independent of project headers.
// The normal build is correctly synchronized. The positive control contains
// an intentional race and must only be run by scripts/check_tsan_runtime.sh.
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <thread>

#ifdef KVMEM_TSAN_PROBE_RACE
static volatile int raced = 0;
#endif

int main() {
#ifdef KVMEM_TSAN_PROBE_RACE
    std::atomic<unsigned> ready{0};
    const auto worker = [&] {
        ready.fetch_add(1, std::memory_order_relaxed);
        while (ready.load(std::memory_order_relaxed) != 2) std::this_thread::yield();
        for (int i = 0; i != 10000; ++i) ++raced;
    };
    std::thread first(worker), second(worker);
    first.join();
    second.join();
    std::puts("intentional race was not detected");
    return 0;
#else
    for (int round = 0; round != 20; ++round) {
        std::mutex mutex;
        std::condition_variable changed;
        bool done = false;
        std::unique_lock<std::mutex> lock(mutex);
        std::thread worker([&] {
            std::lock_guard<std::mutex> held(mutex);
            done = true;
            changed.notify_one();
        });
        while (!done) changed.wait_for(lock, std::chrono::milliseconds(50));
        lock.unlock();
        worker.join();
    }
    std::puts("condition-variable handoff: passed");
    return 0;
#endif
}
