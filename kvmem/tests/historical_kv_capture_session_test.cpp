#include "kvmem/historical_kv_capture_session.hpp"
#include "kvmem/historical_kv_gc.hpp"

#include <cstdio>
#include <cstdlib>
#include <atomic>
#include <new>
#include <stdexcept>
#include <thread>
#include <future>
#include <chrono>

namespace kvmem {
struct HistoricalKvCaptureSessionTestAccess {
    static void before_release(HistoricalKvCaptureSession & session, uint32_t index,
                               std::function<void()> before) {
        auto original = std::move(session.callbacks_.at(index));
        session.callbacks_[index] = [original = std::move(original), before = std::move(before)](
                HistoricalKvHandle handle) {
            before();
            return original(handle);
        };
    }
};
}

static thread_local int64_t allocation_budget = -1;
void * operator new(size_t size) {
    if (!allocation_budget) throw std::bad_alloc();
    if (allocation_budget > 0) --allocation_budget;
    if (void * p = std::malloc(size ? size : 1)) return p;
    throw std::bad_alloc();
}
void * operator new[](size_t size) { return ::operator new(size); }
void operator delete(void * p) noexcept { std::free(p); }
void operator delete[](void * p) noexcept { std::free(p); }
void operator delete(void * p, size_t) noexcept { std::free(p); }
void operator delete[](void * p, size_t) noexcept { std::free(p); }

using namespace kvmem;
static void check(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}
static KvMemRuntimeConfig runtime_config() {
    KvMemRuntimeConfig cfg;
    cfg.store.block_tokens = 4;
    return cfg;
}
static HistoricalKvRawStoreConfig storage_config() { return {1, 4, 2, 3, 0, {}}; }
static HistoricalKvRamDemand demand(uint64_t input, uint64_t output, uint64_t appends) {
    HistoricalKvRamDemand d;
    d.uncached_input_tokens = input;
    d.max_accepted_output_tokens = output;
    d.max_append_boundaries = appends;
    return d;
}
static void complete(const HistoricalKvShadowDirectory::Producer & task, uint32_t n) {
    const auto & s = task->storage();
    check(s->write_layer(task->writer(), task->handle(), 0,
                        std::vector<uint8_t>(2 * n, 17), std::vector<uint8_t>(3 * n, 23), {}, 0),
          "task failed to write captured payload");
    check(s->publish_complete(task->repository(), task->writer(), task->handle()), "complete task did not publish");
}

static void quotas_and_handoff() {
    KvMemRuntime runtime(runtime_config());
    auto session = HistoricalKvCaptureSession::try_create(runtime, storage_config(), 6, demand(6, 3, 5));
    check(session && session->reserved_blocks() == 6 && session->payload_bytes() == 120, "wrong real reservation");
    check(!session->end_input() && !session->finish(), "unfinished input accepted");
    check(!session->append(7) && !runtime.store().block_count(), "input quota exceeded");
    for (unsigned id = 0; id < 2; ++id) {
        check(session->append(3), "input append failed");
        auto task = runtime.historical_shadow_directory().find(id)->writer_context;
        check(!session->append(1), "producer handoff bypassed");
        session->producers_captured();
        complete(task, 3);
    }
    check(session->end_input() && !session->append(4), "output quota not enforced");
    check(session->append(1), "output append failed");
    auto task = runtime.historical_shadow_directory().find(2)->writer_context;
    session->producers_captured();
    check(!session->finish(), "Writing span accepted as complete");
    complete(task, 1);
    check(!session->finish(), "outstanding task accepted as complete");
    task.reset();
    check(session->finish() && session->payload_bytes() == 60 && session->output_remaining() == 2,
          "early EOS did not return unused private payloads");
    check(!session->append(1) && !session->finish() && !session->end_input(), "finished session admitted work");
    check(runtime.store().blocks()[1].orig_pos_start == 3 &&
          runtime.store().blocks()[2].orig_pos_start == 6, "short spans were repacked");
    auto access = runtime.historical_shadow_directory().acquire_read(2);
    check(bool(access), "published identity unreadable after producer exit");
}

static void retired_blocks_do_not_invalidate_completed_request() {
    KvMemRuntime runtime(runtime_config());
    auto session = HistoricalKvCaptureSession::try_create(runtime, storage_config(), 3, demand(8, 0, 1));
    check(session && session->append(8), "retired completion fixture append failed");
    auto first = runtime.historical_shadow_directory().find(0)->writer_context;
    auto second = runtime.historical_shadow_directory().find(1)->writer_context;
    const auto first_handle = first->handle();
    const auto second_handle = second->handle();
    session->producers_captured();
    complete(first, 4);
    complete(second, 4);
    first.reset();
    second.reset();
    check(session->end_input(), "retired completion fixture input remained open");
    const auto repository = session->repository();
    check(repository->retire(first_handle) == HistoricalKvRepository::RetireResult::Committed &&
          repository->retire(second_handle) == HistoricalKvRepository::RetireResult::Committed,
          "completed historical block could not retire");
    check(repository->try_reclaim(first_handle) == HistoricalKvRepository::ReclaimResult::Complete,
          "first retired block did not reclaim");
    check(session->finish() && session->payload_bytes() == 20,
          "completed request rejected reclaimed or still-retired historical blocks");
    check(repository->try_reclaim(second_handle) == HistoricalKvRepository::ReclaimResult::Complete &&
          session->payload_bytes() == 0, "retired payload did not finish releasing after request");
}

static void request_finishes_while_background_release_is_blocked() {
    KvMemRuntime runtime(runtime_config());
    auto session = HistoricalKvCaptureSession::try_create(runtime, storage_config(), 3, demand(8, 0, 1));
    check(bool(session), "blocked release fixture admission failed");
    std::promise<void> entered, allow_release;
    auto entered_future = entered.get_future();
    auto release_future = allow_release.get_future().share();
    std::thread::id release_thread;
    HistoricalKvCaptureSessionTestAccess::before_release(*session, 0, [&] {
        release_thread = std::this_thread::get_id();
        entered.set_value();
        release_future.wait();
    });
    check(session->append(8), "blocked release fixture append failed");
    auto first = runtime.historical_shadow_directory().find(0)->writer_context;
    auto second = runtime.historical_shadow_directory().find(1)->writer_context;
    const auto handle = first->handle();
    session->producers_captured();
    complete(first, 4);
    complete(second, 4);
    first.reset();
    second.reset();
    check(session->end_input(), "blocked release fixture input remained open");
    const auto repository = session->repository();
    check(repository->retire(handle) == HistoricalKvRepository::RetireResult::Committed,
          "blocked release fixture did not retire");
    HistoricalKvGc::Config config;
    config.mode = HistoricalKvGc::Mode::Off;
    config.scan_blocks = config.reclaim_blocks = 3;
    config.batch_blocks = 1;
    config.poll_ms = 1;
    HistoricalKvGc gc(repository, std::make_unique<HistoricalKvLruPolicy>(3), config);
    // Declared after the manager so unwinding cannot join a closed barrier.
    struct Unblock {
        std::promise<void> & promise;
        bool opened = false;
        void open() { if (!opened) { promise.set_value(); opened = true; } }
        ~Unblock() { open(); }
    } unblock{allow_release};
    gc.start();
    check(entered_future.wait_for(std::chrono::seconds(5)) == std::future_status::ready &&
          repository->is_releasing(handle) && repository->outstanding_operations() == 0,
          "worker did not reach the controlled release barrier");
    auto finish = std::async(std::launch::async, [&] { return session->finish(); });
    const bool returned_while_blocked = finish.wait_for(std::chrono::seconds(5)) == std::future_status::ready;
    const auto bytes_while_blocked = session->payload_bytes();
    // Always unblock before retrieving the future, including a timeout.
    unblock.open();
    const bool finished = finish.get();
    gc.stop();
    check(returned_while_blocked && finished && bytes_while_blocked == 40 &&
          session->live_blocks_at_finish() == 1,
          "request completion depended on background physical release");
    check(release_thread != std::this_thread::get_id() &&
          repository->reclaim_completed(handle, repository->repository_id()) &&
          session->payload_bytes() == 20 && gc.stats().reclaimed == 1,
          "background release did not finish after request completion");
}

static void boundary_limit() {
    KvMemRuntime runtime(runtime_config());
    auto session = HistoricalKvCaptureSession::try_create(runtime, storage_config(), 3, demand(9, 0, 1));
    check(session && session->append(1), "bounded first append failed");
    session->producers_captured();
    check(!session->append(8) && runtime.store().total_tokens() == 1 && session->input_remaining() == 8,
          "append budget bypassed by unused token/slot credits");
    session->fail();
    check(session->failed() && !session->append(0), "failed session reopened");
}

static void cancelled_owner_keeps_tasks() {
    std::weak_ptr<HistoricalKvRawStore> observed;
    HistoricalKvShadowDirectory::Producer task;
    std::atomic<bool> finish_task{false}, succeeded{false};
    std::thread worker;
    {
        KvMemRuntime runtime(runtime_config());
        auto session = HistoricalKvCaptureSession::try_create(runtime, storage_config(), 2, demand(3, 1, 2));
        check(session && session->append(3), "cancel fixture append failed");
        task = runtime.historical_shadow_directory().find(0)->writer_context;
        observed = task->storage();
        session->producers_captured();
        session->fail();
        check(task->storage()->allocated_payload_bytes() == 20 && task->repository().closed(),
              "cancel freed installed payload or kept admission open");
        worker = std::thread([captured = task, &finish_task, &succeeded] {
            while (!finish_task.load(std::memory_order_acquire)) std::this_thread::yield();
            try { complete(captured, 3); succeeded = true; } catch (...) {}
        });
        task.reset();
    }
    const bool survived = !observed.expired();
    finish_task.store(true, std::memory_order_release);
    worker.join();
    check(survived && succeeded, "background task lost owner at runtime destruction");
    check(observed.expired(), "repository release callback retained ownership cycle");
}

static void failures_are_atomic() {
    KvMemRuntime runtime(runtime_config());
    check(!HistoricalKvCaptureSession::try_create(runtime, storage_config(), 1, demand(3, 1, 2)),
          "insufficient capacity accepted");
    bool reached_success = false;
    for (int64_t budget = 0; budget < 500; ++budget) {
        allocation_budget = budget;
        auto session = HistoricalKvCaptureSession::try_create(runtime, storage_config(), 3, demand(9, 0, 1));
        allocation_budget = -1;
        check(!runtime.store().block_count(), "admission changed runtime");
        if (session) { reached_success = true; break; }
    }
    check(reached_success, "admission fault sweep never succeeded");
    reached_success = false;
    unsigned failures = 0;
    for (int64_t budget = 0; budget < 100; ++budget) {
        KvMemRuntime rt(runtime_config());
        auto session = HistoricalKvCaptureSession::try_create(rt, storage_config(), 3, demand(9, 0, 1));
        bool appended = false;
        allocation_budget = budget;
        try { appended = session->append(9); }
        catch (const std::bad_alloc &) { ++failures; }
        allocation_budget = -1;
        if (appended) { reached_success = true; break; }
        check(session->failed() && !session->append(9) && !rt.store().block_count(),
              "failed registration changed runtime or admitted a retry");
    }
    check(reached_success && failures, "metadata fault sweep did not cover failure and success");
}

int main() {
    try {
        quotas_and_handoff();
        retired_blocks_do_not_invalidate_completed_request();
        request_finishes_while_background_release_is_blocked();
        boundary_limit();
        cancelled_owner_keeps_tasks();
        failures_are_atomic();
        std::puts("PASS historical capture session: bounded quotas, producer handoff, publication, terminal cancellation, allocation faults");
    } catch (const std::exception & e) {
        allocation_budget = -1;
        std::fprintf(stderr, "FAIL historical capture session: %s\n", e.what());
        return 1;
    }
}
