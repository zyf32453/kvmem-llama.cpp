#include "kvmem/historical_kv_gc.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

namespace {
using namespace std::chrono_literals;
using Repo = kvmem::HistoricalKvRepository;
using Gc = kvmem::HistoricalKvGc;

void require(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}

struct Gate {
    std::mutex mutex;
    std::condition_variable changed;
    bool entered = false;
    bool opened = false;
    void pause() {
        std::unique_lock<std::mutex> guard(mutex);
        entered = true;
        changed.notify_all();
        changed.wait(guard, [&] { return opened; });
    }
    bool await_entry() {
        std::unique_lock<std::mutex> guard(mutex);
        return changed.wait_for(guard, 5s, [&] { return entered; });
    }
    void open() {
        std::lock_guard<std::mutex> guard(mutex);
        opened = true;
        changed.notify_all();
    }
};

class OnePerRound final : public kvmem::HistoricalKvPolicy {
public:
    explicit OnePerRound(Gate * acknowledgement = nullptr) : acknowledgement_(acknowledgement) {}
    kvmem::HistoricalKvPrepareCode prepare_commit(const kvmem::HistoricalKvBlockMeta &,
        kvmem::HistoricalKvCommitConditions & conditions) const noexcept override {
        conditions = {};
        return kvmem::HistoricalKvPrepareCode::Ready;
    }
    kvmem::HistoricalKvStepResult step(const kvmem::HistoricalKvSnapshot & snapshot,
        const kvmem::HistoricalKvPressure &, kvmem::HistoricalKvWorkBudget,
        kvmem::HistoricalKvDecisionBuffer & out) override {
        for (const auto & block : snapshot.blocks) {
            if (block.life != kvmem::HistoricalKvLife::Live) continue;
            if (out.push({block.handle, snapshot.round_id, kvmem::HistoricalKvDeleteReason::Custom, {}}))
                return {kvmem::HistoricalKvStepStatus::Complete, 1, 1};
        }
        return {kvmem::HistoricalKvStepStatus::NoCandidate, snapshot.blocks.size(), 0};
    }
    void on_result(const kvmem::HistoricalKvCommitResult & result) noexcept override {
        if (acknowledgement_ && result.code == kvmem::HistoricalKvCommitCode::Committed)
            acknowledgement_->pause();
    }
private:
    Gate * acknowledgement_;
};

Gc::Config config(unsigned capacity = 1) {
    Gc::Config result;
    result.mode = Gc::Mode::Delete;
    result.retire_at_decode_boundary = true;
    result.scan_blocks = result.reclaim_blocks = capacity;
    result.batch_blocks = 1;
    result.poll_ms = 1;
    return result;
}

kvmem::HistoricalKvHandle append(const std::shared_ptr<Repo> & repository, uint64_t id,
                                std::atomic<unsigned> & released) {
    auto writer = repository->begin_write(id, id * 3 + 1, 2, id + 1, 16,
        [&](const kvmem::HistoricalKvHandle &) { ++released; return true; });
    require(bool(writer), "writer admission failed");
    const auto handle = writer->handle();
    require(repository->publish_write(*writer), "writer publication failed");
    return handle;
}
} // namespace

namespace kvmem {
// Pause with *all* worker mutexes held. The SFINAE legacy branch intentionally
// lets this exact probe compile against stage A: its boundary must then fail
// the non-waiting test. No production scheduling hook or new callback is added.
struct HistoricalKvGcAuditTestAccess {
    template<class Manager>
    static auto pause_handoff(Manager & gc, Gate & gate, int)
        -> decltype((void)gc.pending_mu_, void()) {
        std::lock_guard<std::mutex> guard(gc.pending_mu_);
        gate.pause();
    }
    template<class Manager>
    static void pause_handoff(Manager &, Gate & gate, long) { gate.pause(); }
    static void pause_worker(Gc & gc, Gate & gate) {
        std::lock_guard<std::mutex> guard(gc.tick_mu_);
        pause_handoff(gc, gate, 0);
    }
};
} // namespace kvmem

namespace {
void boundary_does_not_wait_for_worker_mutexes() {
    std::atomic<unsigned> released{0};
    auto repository = std::make_shared<Repo>(1);
    append(repository, 0, released);
    Gc gc(repository, std::make_unique<OnePerRound>(), config());
    gc.manual_tick();
    require(gc.stats().proposed == 1 && gc.stats().committed == 0, "missing prepared batch");
    Gate gate;
    std::thread paused([&] { kvmem::HistoricalKvGcAuditTestAccess::pause_worker(gc, gate); });
    if (!gate.await_entry()) {
        gate.open();
        paused.join();
        throw std::runtime_error("worker pause did not start");
    }
    auto boundary = std::async(std::launch::async, [&] { return gc.commit_at_decode_boundary(); });
    const bool completed_while_paused = boundary.wait_for(2s) == std::future_status::ready;
    gate.open();
    paused.join();
    const auto result = boundary.get();
    require(completed_while_paused, "boundary waited for worker mutexes");
    require(result.prepared == 1 && result.committed == 1, "batch was lost or duplicated");
    gc.manual_tick();
    require(gc.stats().committed == 1 && released == 1, "acknowledgement/release failed");
}

void pending_batch_does_not_stop_prior_reclamation() {
    std::atomic<unsigned> released{0};
    auto repository = std::make_shared<Repo>(3);
    const auto prepared = append(repository, 0, released);
    const auto earlier = append(repository, 1, released);
    Gc gc(repository, std::make_unique<OnePerRound>(), config(3));
    gc.manual_tick();
    require(gc.stats().proposed == 1, "no pending batch");
    require(repository->retire(earlier) == Repo::RetireResult::Committed, "explicit retirement failed");
    gc.manual_tick();
    require(released == 1 && gc.stats().reclaimed == 1 && gc.stats().proposed == 1,
            "outstanding batch stopped reclamation or allowed extra decisions");
    require(bool(repository->acquire_read(prepared)), "batch retired before boundary");
    require(gc.commit_at_decode_boundary().committed == 1, "pending command lost");
    gc.manual_tick();
    require(released == 2, "pending command did not release");
}

void slow_acknowledgement_does_not_hold_boundary() {
    std::atomic<unsigned> released{0};
    auto repository = std::make_shared<Repo>(1);
    append(repository, 0, released);
    Gate gate;
    Gc gc(repository, std::make_unique<OnePerRound>(&gate), config());
    gc.manual_tick();
    require(gc.commit_at_decode_boundary().committed == 1, "initial commit failed");
    std::thread worker([&] { gc.manual_tick(); });
    if (!gate.await_entry()) {
        gate.open();
        worker.join();
        throw std::runtime_error("acknowledgement pause did not start");
    }
    auto boundary = std::async(std::launch::async, [&] { return gc.commit_at_decode_boundary(); });
    const bool completed = boundary.wait_for(2s) == std::future_status::ready;
    gate.open();
    worker.join();
    require(completed && boundary.get().prepared == 0, "boundary waited for policy acknowledgement");
    require(gc.stats().committed == 1 && released == 1, "slow acknowledgement lost result");
}

void in_progress_batch_cannot_be_reentered() {
    std::atomic<unsigned> released{0};
    auto repository = std::make_shared<Repo>(1);
    append(repository, 0, released);
    struct Context { Gc * gc = nullptr; unsigned calls = 0, nested_prepared = 999; } context;
    auto cfg = config();
    cfg.boundary_context = &context;
    cfg.on_boundary_retired = [](void * opaque, kvmem::HistoricalKvHandle) noexcept {
        auto & value = *static_cast<Context *>(opaque);
        ++value.calls;
        value.nested_prepared = value.gc->commit_at_decode_boundary().prepared;
    };
    Gc gc(repository, std::make_unique<OnePerRound>(), cfg);
    context.gc = &gc;
    gc.manual_tick();
    require(gc.commit_at_decode_boundary().committed == 1, "outer batch failed");
    require(context.calls == 1 && context.nested_prepared == 0, "in-flight batch consumed twice");
    require(gc.commit_at_decode_boundary().prepared == 0, "unacknowledged batch repeated");
    gc.manual_tick();
    require(gc.stats().committed == 1 && released == 1, "reentrant result accounting");
}

void background_exchange_and_generation_reuse() {
    std::atomic<unsigned> released{0};
    auto repository = std::make_shared<Repo>(1);
    Gc gc(repository, std::make_unique<OnePerRound>(), config());
    gc.start();
    // The production worker sleeps between ticks. A 1-ms requested sleep can
    // use a coarser OS timer; this is a 500-exchange safety test, not a 15-s
    // throughput gate. Preserve progress counters if it really fails to drain.
    const auto deadline = std::chrono::steady_clock::now() + 45s;
    unsigned current_round = 0;
    const auto before_deadline = [&] {
        if (std::chrono::steady_clock::now() >= deadline) {
            const auto stats = gc.stats();
            throw std::runtime_error("background exchange stalled: round=" + std::to_string(current_round) +
                " ticks=" + std::to_string(stats.ticks) + " proposed=" + std::to_string(stats.proposed) +
                " committed=" + std::to_string(stats.committed) + " boundary=" + std::to_string(stats.boundary_committed) +
                " released=" + std::to_string(released.load()) + " ready=" + std::to_string(repository->ready_count()));
        }
        std::this_thread::yield();
    };
    constexpr unsigned rounds = 500;
    kvmem::HistoricalKvHandle previous{};
    for (unsigned round = 0; round < rounds; ++round) {
        current_round = round;
        while (repository->ready_count() != 1) {
            gc.commit_at_decode_boundary();
            before_deadline();
        }
        const auto handle = append(repository, round, released);
        if (round) {
            require(handle.generation != previous.generation, "slot generation reused");
            require(!repository->acquire_read(previous), "old generation granted read");
        }
        auto reader = repository->acquire_read(handle);
        require(bool(reader), "read before retirement failed");
        while (gc.stats().boundary_committed != round + 1) {
            gc.commit_at_decode_boundary();
            before_deadline();
        }
        require(!repository->acquire_new(handle), "retired generation admitted new execution");
        require(released == round, "old reader released too early");
        reader->reset();
        previous = handle;
    }
    while (released != rounds || gc.stats().committed != rounds) before_deadline();
    gc.stop();
    const auto stats = gc.stats();
    require(stats.boundary_batches == rounds && stats.boundary_committed == rounds &&
            stats.committed == rounds && stats.reclaimed == rounds && released == rounds,
            "exchange lost/duplicated commands or acknowledgements");
    require(!stats.worker_failures && !stats.policy_failures, "background worker failure");
}
} // namespace

int main(int argc, char ** argv) {
    try {
        boundary_does_not_wait_for_worker_mutexes();
        std::cout << "PASS boundary completes while all worker mutexes are held\n";
        if (argc == 2 && std::string(argv[1]) == "--worker-locks") return 0;
        pending_batch_does_not_stop_prior_reclamation();
        slow_acknowledgement_does_not_hold_boundary();
        in_progress_batch_cannot_be_reentered();
        background_exchange_and_generation_reuse();
        std::cout << "PASS pending/acknowledgement/reentrant ownership and 500-generation background exchange\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "FAIL historical boundary mailbox: " << error.what() << '\n';
        return 1;
    }
}
