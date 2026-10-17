#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <thread>

namespace kvmem {

// Bounded CPU-only disposal. Reserve a cell AND allocate its Task before
// admitting work. Publishing to that exclusive cell never allocates, waits
// for a worker, or destroys the task on the publishing thread. Saturation
// rejects new admission; there is no synchronous/unbounded fallback.
class HistoricalKvCleanup {
    struct Slot;
public:
    struct Task {
        virtual ~Task() = default;
        // A closed domain may still have a policy worker finishing a callback.
        // Polling readiness lets other domains clean up while that one waits.
        // ready() must be nonblocking and noexcept. Both methods are CPU-only.
        virtual bool ready() const noexcept { return true; }
    };

    class Ticket {
    public:
        Ticket() = default;
        ~Ticket();
        Ticket(Ticket && other) noexcept;
        Ticket & operator=(Ticket && other) noexcept;
        Ticket(const Ticket &) = delete;
        Ticket & operator=(const Ticket &) = delete;
        explicit operator bool() const noexcept { return slot_ != nullptr; }
        // Single-use. Both a valid ticket and a non-null task are required.
        void submit(std::unique_ptr<Task> task) noexcept;
    private:
        friend class HistoricalKvCleanup;
        explicit Ticket(Slot * slot) : slot_(slot) {}
        void cancel() noexcept;
        Slot * slot_ = nullptr;
    };

    explicit HistoricalKvCleanup(uint32_t capacity);
    // Service lifetime must exceed every Ticket. Shutdown drains submitted
    // tasks and joins the worker; use only at process/test-service shutdown.
    ~HistoricalKvCleanup();
    HistoricalKvCleanup(const HistoricalKvCleanup &) = delete;
    HistoricalKvCleanup & operator=(const HistoricalKvCleanup &) = delete;
    std::optional<Ticket> try_reserve() noexcept;
    uint32_t occupied() const noexcept; // diagnostic observation only
    uint32_t capacity() const noexcept { return capacity_; }
    static HistoricalKvCleanup & process();

private:
    void run() noexcept;
    const uint32_t capacity_;
    std::unique_ptr<Slot[]> slots_;
    std::atomic<bool> stopping_{false};
    std::thread worker_;
};

} // namespace kvmem
