// Compile unchanged against B1 and B2. For the B1 negative control only,
// -fno-access-control permits taking the old private content mutex; the old
// production headers/library and their locking behavior remain unchanged.
#include "kvmem/historical_kv_gc.hpp"
#include "kvmem/historical_kv_redundancy.hpp"
#include <chrono>
#include <condition_variable>
#include <future>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace {
struct Pause {
    std::mutex mutex;
    std::condition_variable cv;
    bool entered = false, opened = false;
    void hold() {
        std::unique_lock<std::mutex> lock(mutex);
        entered = true;
        cv.notify_all();
        cv.wait(lock, [&] { return opened; });
    }
    bool await() {
        std::unique_lock<std::mutex> lock(mutex);
        return cv.wait_for(lock, std::chrono::seconds(5), [&] { return entered; });
    }
    void open() {
        std::lock_guard<std::mutex> lock(mutex);
        opened = true;
        cv.notify_all();
    }
};
}
namespace kvmem {
struct HistoricalKvThinkingRangesTestAccess {
    static void pause(HistoricalKvThinkingRanges & ranges, Pause & pause) {
        std::lock_guard<std::mutex> lock(ranges.mu_);
        pause.hold();
    }
};
}

int main() {
    try {
        auto ranges = std::make_shared<kvmem::HistoricalKvThinkingRanges>();
        ranges->note(0, 128);
        ranges->arm();
        auto repository = std::make_shared<kvmem::HistoricalKvRepository>(2);
        for (unsigned i = 0; i < 2; ++i) {
            auto writer = repository->begin_write(i, i * 64, 64, i + 1, 16,
                [](const kvmem::HistoricalKvHandle &) { return true; });
            if (!writer || !repository->publish_write(*writer)) throw std::runtime_error("fixture admission");
        }
        kvmem::HistoricalKvGc::Config config;
        config.mode = kvmem::HistoricalKvGc::Mode::Delete;
        config.ready_low = 1; config.ready_high = 2;
        config.scan_blocks = config.reclaim_blocks = 2;
        config.batch_blocks = 1;
        config.retire_at_decode_boundary = true;
        kvmem::HistoricalKvGc gc(repository,
            std::make_unique<kvmem::HistoricalKvLruPolicy>(2, true, ranges), config);
        gc.manual_tick();
        if (gc.stats().proposed != 1) throw std::runtime_error("missing LRU proposal");
        Pause pause;
        std::thread reader([&] { kvmem::HistoricalKvThinkingRangesTestAccess::pause(*ranges, pause); });
        if (!pause.await()) {
            pause.open(); reader.join();
            throw std::runtime_error("content lock was not acquired");
        }
        auto boundary = std::async(std::launch::async, [&] { return gc.commit_at_decode_boundary(); });
        const bool completed = boundary.wait_for(std::chrono::seconds(2)) == std::future_status::ready;
        pause.open(); reader.join();
        const auto result = boundary.get();
        if (!completed) throw std::runtime_error("boundary waited for content range mutex");
        if (result.committed != 1) throw std::runtime_error("qualified LRU victim was not committed");
        gc.manual_tick();
        if (gc.stats().committed != 1 || gc.stats().reclaimed != 1)
            throw std::runtime_error("LRU result was lost");
        std::cout << "PASS real LRU commits with content mutex held; retirement and release preserved\n";
    } catch (const std::exception & error) {
        std::cerr << "FAIL range lock: " << error.what() << '\n';
        return 1;
    }
}
