#include "kvmem/historical_kv_cleanup.hpp"

#include <chrono>
#include <exception>
#include <stdexcept>
#include <utility>

namespace kvmem {
namespace {
enum : uint32_t { Free, Reserved, Queued };
static_assert(std::atomic<uint32_t>::is_always_lock_free,
              "historical cleanup requires lock-free admission/publication");
}

struct HistoricalKvCleanup::Slot {
    std::atomic<uint32_t> state{Free};
    std::unique_ptr<Task> task;
};

HistoricalKvCleanup::Ticket::~Ticket() { cancel(); }
HistoricalKvCleanup::Ticket::Ticket(Ticket && other) noexcept
    : slot_(std::exchange(other.slot_, nullptr)) {}
HistoricalKvCleanup::Ticket & HistoricalKvCleanup::Ticket::operator=(Ticket && other) noexcept {
    if (this != &other) {
        cancel();
        slot_ = std::exchange(other.slot_, nullptr);
    }
    return *this;
}
void HistoricalKvCleanup::Ticket::cancel() noexcept {
    if (slot_) {
        // Reserved cells have never received a task: no payload destruction.
        slot_->state.store(Free, std::memory_order_release);
        slot_ = nullptr;
    }
}
void HistoricalKvCleanup::Ticket::submit(std::unique_ptr<Task> task) noexcept {
    if (!slot_ || !task) std::terminate();
    auto * slot = std::exchange(slot_, nullptr);
    slot->task = std::move(task);
    slot->state.store(Queued, std::memory_order_release);
}

HistoricalKvCleanup::HistoricalKvCleanup(uint32_t capacity) : capacity_(capacity) {
    if (!capacity) throw std::invalid_argument("historical cleanup capacity must be positive");
    slots_ = std::make_unique<Slot[]>(capacity);
    worker_ = std::thread([this] { run(); });
}
HistoricalKvCleanup::~HistoricalKvCleanup() {
    stopping_.store(true, std::memory_order_release);
    worker_.join();
}
HistoricalKvCleanup & HistoricalKvCleanup::process() {
    // A production adapter domain holds cells for metadata, request-private
    // RAM and final teardown, plus a transient cell during request handoff.
    // All admissions share this fixed bound; reuse follows CPU destruction.
    static HistoricalKvCleanup service(64);
    return service;
}
std::optional<HistoricalKvCleanup::Ticket> HistoricalKvCleanup::try_reserve() noexcept {
    for (uint32_t i = 0; i < capacity_; ++i) {
        uint32_t expected = Free;
        if (slots_[i].state.compare_exchange_strong(expected, Reserved,
                std::memory_order_acquire, std::memory_order_relaxed))
            return Ticket(&slots_[i]);
    }
    return std::nullopt;
}
uint32_t HistoricalKvCleanup::occupied() const noexcept {
    uint32_t count = 0;
    for (uint32_t i = 0; i < capacity_; ++i)
        count += slots_[i].state.load(std::memory_order_acquire) != Free;
    return count;
}
void HistoricalKvCleanup::run() noexcept {
    for (;;) {
        bool pending = false;
        for (uint32_t i = 0; i < capacity_; ++i) {
            auto & slot = slots_[i];
            const auto state = slot.state.load(std::memory_order_acquire);
            pending |= state != Free;
            if (state == Queued && slot.task->ready()) {
                slot.task.reset();
                // Payload and metadata are gone before capacity is reusable.
                slot.state.store(Free, std::memory_order_release);
            }
        }
        if (stopping_.load(std::memory_order_acquire) && !pending) return;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

} // namespace kvmem
