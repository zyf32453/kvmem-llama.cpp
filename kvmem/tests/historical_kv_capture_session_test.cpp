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
        auto original = std::move(session.callbacks_.at(index).callback);
        session.callbacks_[index].callback = [original = std::move(original), before = std::move(before)](
                HistoricalKvHandle handle, HistoricalKvRepository::HostResource resource) {
            if (resource == HistoricalKvRepository::HostResource::PackedKv) before();
            return original(handle, resource);
        };
    }
};
}

static thread_local int64_t allocation_budget = -1;
static thread_local uint64_t foreground_frees = 0;
void * operator new(size_t size) {
    if (!allocation_budget) throw std::bad_alloc();
    if (allocation_budget > 0) --allocation_budget;
    if (void * p = std::malloc(size ? size : 1)) return p;
    throw std::bad_alloc();
}
void * operator new[](size_t size) { return ::operator new(size); }
void operator delete(void * p) noexcept { if (p) ++foreground_frees; std::free(p); }
void operator delete[](void * p) noexcept { ::operator delete(p); }
void operator delete(void * p, size_t) noexcept { ::operator delete(p); }
void operator delete[](void * p, size_t) noexcept { ::operator delete(p); }

using namespace kvmem;
static void check(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}
static void wait_cleanup(HistoricalKvCleanup & cleanup = HistoricalKvCleanup::process()) {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (cleanup.occupied() && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
    check(!cleanup.occupied(), "background private reservation cleanup did not finish");
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
    check(session->finish(), "early EOS did not finish");
    wait_cleanup();
    check(session->payload_bytes() == 60 && session->output_remaining() == 2,
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
    check(session->failed() && !session->append(0) && !session->cancel(), "failed session reopened");
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
        wait_cleanup();
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
    wait_cleanup();
    check(survived && succeeded, "background task lost owner at runtime destruction");
    check(observed.expired(), "repository release callback retained ownership cycle");
}

static void failures_are_atomic() {
    KvMemRuntime runtime(runtime_config());
    const auto original_bytes = runtime.store().allocated_bytes();
    check(!HistoricalKvCaptureSession::try_create(runtime, storage_config(), 1, demand(3, 1, 2)),
          "insufficient capacity accepted");
    bool reached_success = false;
    for (int64_t budget = 0; budget < 500; ++budget) {
        allocation_budget = budget;
        auto session = HistoricalKvCaptureSession::try_prepare(runtime, storage_config(), 3, demand(9, 0, 1));
        allocation_budget = -1;
        check(!runtime.store().block_count() && runtime.store().allocated_bytes() == original_bytes &&
              !runtime.historical_shadow_directory().size(), "private preparation changed runtime");
        if (session) { reached_success = true; break; }
        check(!HistoricalKvCleanup::process().occupied(), "failed preparation queued private cleanup");
    }
    check(reached_success, "admission fault sweep never succeeded");
    auto session = HistoricalKvCaptureSession::try_prepare(runtime, storage_config(), 3, demand(9, 0, 1));
    check(session && !session->admitted() && session->metadata_remaining() == 3 &&
          !session->append(9) && !session->end_input() && !session->finish(),
          "private preparation admitted model work");
    allocation_budget = 0;
    const bool admitted = session->admit();
    const auto table_bytes = runtime.store().allocated_bytes();
    const auto frees_before = foreground_frees;
    const bool appended = session->append(9);
    allocation_budget = -1;
    check(admitted && appended && foreground_frees == frees_before &&
          session->metadata_remaining() == 0 && !session->admit() &&
          runtime.store().block_count() == 3 && runtime.store().total_tokens() == 9 &&
          runtime.store().allocated_bytes() == table_bytes,
          "reserved metadata append allocated, freed, or changed capacity");
    for (uint32_t id = 0; id < 3; ++id) {
        auto task = runtime.historical_shadow_directory().find(id)->writer_context;
        complete(task, id == 2 ? 1 : 4);
    }
    session->producers_captured();
    check(session->end_input() && session->finish(), "reserved metadata request did not finish");
    wait_cleanup();

    KvMemRuntime empty(runtime_config());
    auto zero = HistoricalKvCaptureSession::try_prepare(empty, storage_config(), 1, demand(0, 0, 0));
    check(zero && zero->admit() && zero->append(0) && zero->end_input() && zero->finish(),
          "zero-token request failed metadata admission");
    wait_cleanup();

    KvMemRuntime changed(runtime_config());
    auto stale = HistoricalKvCaptureSession::try_prepare(changed, storage_config(), 3, demand(9, 0, 1));
    changed.register_append(1);
    check(stale && !stale->admit() && changed.store().total_tokens() == 1,
          "stale preparation replaced newer runtime data");
    auto storage = stale->storage();
    stale.reset();
    check(!storage->allocated_payload_bytes() && !HistoricalKvCleanup::process().occupied(),
          "rejected admission did not synchronously return private RAM");
}

static void reserved_metadata_batch_is_atomic() {
    KvMemRuntime runtime(runtime_config()), other(runtime_config());
    auto repo = std::make_shared<HistoricalKvRepository>(4);
    auto storage = std::make_shared<HistoricalKvRawStore>(storage_config(), 4);
    auto reservation = runtime.prepare_historical_metadata(repo, storage, 2);
    check(reservation && !other.install_historical_metadata(*reservation) &&
          runtime.install_historical_metadata(*reservation), "metadata ownership validation failed");
    const auto table_bytes = runtime.store().allocated_bytes();
    auto write = [&](uint32_t id, uint32_t begin, uint32_t count,
                     uint64_t logical_id = UINT64_MAX) {
        auto writer = repo->begin_write(logical_id == UINT64_MAX ? 1000 + id : logical_id,
            begin, count, id + 1, 0,
            [weak = std::weak_ptr<HistoricalKvRepository>(repo), storage](HistoricalKvHandle handle) {
                auto owner = weak.lock();
                return owner && storage->release(*owner, handle).code == HistoricalKvRawStore::ReleaseCode::Released;
            });
        check(writer && storage->reserve(*writer, writer->handle(), begin, count), "writer fixture failed");
        return std::move(*writer);
    };
    std::vector<HistoricalKvRepository::Lease> writers;
    std::vector<HistoricalKvShadowDirectory::Producer> producers;
    writers.push_back(write(0, 0, 3));
    writers.push_back(write(1, 4, 2)); // Reject the late gap without consuming the first writer.
    producers.reserve(2);
    allocation_budget = 0;
    const bool gap = runtime.register_historical_append(repo, storage, writers, producers, reservation.get());
    allocation_budget = -1;
    check(!gap && writers[0] && writers[1] && producers.empty() && reservation->remaining() == 2 &&
          !runtime.store().block_count() && !runtime.historical_shadow_directory().size(),
          "malformed batch partially committed reserved metadata");
    const auto bad = writers[1].handle();
    check(repo->retire(bad) == HistoricalKvRepository::RetireResult::Committed, "bad writer retirement failed");
    writers[1].reset();
    check(repo->try_reclaim(bad) == HistoricalKvRepository::ReclaimResult::Complete, "bad writer cleanup failed");
    writers[1] = write(1, 3, 2, 999);
    allocation_budget = 0;
    const bool out_of_order = runtime.register_historical_append(
        repo, storage, writers, producers, reservation.get());
    allocation_budget = -1;
    check(!out_of_order && writers[0] && writers[1] && producers.empty() &&
          reservation->remaining() == 2 && !runtime.store().block_count(),
          "reserved append accepted a nonmonotonic logical identity");
    const auto old_identity = writers[1].handle();
    check(repo->retire(old_identity) == HistoricalKvRepository::RetireResult::Committed,
          "nonmonotonic writer retirement failed");
    writers[1].reset();
    check(repo->try_reclaim(old_identity) == HistoricalKvRepository::ReclaimResult::Complete,
          "nonmonotonic writer cleanup failed");
    writers[1] = write(1, 3, 2);
    std::vector<HistoricalKvShadowDirectory::Producer> no_room;
    check(!runtime.register_historical_append(repo, storage, writers, no_room, reservation.get()) &&
          !other.register_historical_append(repo, storage, writers, producers, reservation.get()) &&
          writers[0] && writers[1], "reserved append ignored output capacity or owner");
    const auto retired = writers[1].handle();
    check(repo->retire(retired) == HistoricalKvRepository::RetireResult::Committed, "Writing retirement failed");
    allocation_budget = 0;
    const bool appended = runtime.register_historical_append(repo, storage, writers, producers, reservation.get());
    allocation_budget = -1;
    check(appended && !writers[0] && !writers[1] && !reservation->remaining() &&
          runtime.store().total_tokens() == 5 && runtime.store().allocated_bytes() == table_bytes,
          "reserved retry failed or grew metadata");
    check(runtime.store().blocks()[0].block_id == 0 &&
          runtime.store().blocks()[0].logical_id == 1000 &&
          runtime.store().blocks()[1].block_id == 1 &&
          runtime.store().blocks()[1].logical_id == 1001 &&
          runtime.historical_shadow_directory().observe(1).logical_id == 1001,
          "reserved metadata still conflated logical identity with table index");
    auto late = producers[1];
    complete(producers[0], 3);
    producers.clear();
    writers.clear();
    writers.push_back(write(2, 5, 1));
    allocation_budget = 0;
    const bool overflow = runtime.register_historical_append(repo, storage, writers, producers, reservation.get());
    const bool bypass = runtime.register_historical_append(repo, storage, writers, producers);
    allocation_budget = -1;
    check(!overflow && !bypass && writers[0] && runtime.store().total_tokens() == 5,
          "metadata quota bypassed by unreserved append");
    bool legacy_rejected = false;
    try { runtime.register_append(1); } catch (const std::logic_error &) { legacy_rejected = true; }
    check(legacy_rejected, "legacy append bypassed the metadata reservation");
    reservation.reset();
    check(storage->write_layer(late->writer(), retired, 0, {1, 2, 3, 4}, {5, 6, 7, 8, 9, 10}, {}, 0) &&
          !storage->publish_complete(*repo, late->writer(), retired),
          "reservation cleanup lost or resurrected an admitted retired writer");
    check(repo->try_reclaim(retired) == HistoricalKvRepository::ReclaimResult::Deferred,
          "retired task lost its writer protection");
    late.reset();
    check(repo->try_reclaim(retired) == HistoricalKvRepository::ReclaimResult::Complete,
          "metadata reservation retained a permanent producer reference");
}

static void reserved_metadata_reuses_reclaimed_table_slots() {
    auto config = runtime_config();
    config.store.select_budget = 8;
    config.store.prefill_budget = 8;
    config.store.sink_blocks = 1;
    config.store.recent_blocks = 1;
    KvMemRuntime runtime(config);
    auto repo = std::make_shared<HistoricalKvRepository>(2);
    auto storage = std::make_shared<HistoricalKvRawStore>(storage_config(), 2);
    auto reservation = runtime.prepare_historical_metadata(repo, storage, 2);
    check(reservation && runtime.install_historical_metadata(*reservation),
          "cyclic metadata reservation failed");
    const auto table_bytes = runtime.store().allocated_bytes();
    std::vector<HistoricalKvRepository::Lease> writers;
    std::vector<HistoricalKvShadowDirectory::Producer> producers;
    writers.reserve(1); producers.reserve(1);
    HistoricalKvShadowDirectory::Producer previous_task;
    HistoricalKvHandle previous_handle{};
    uint32_t previous_id = UINT32_MAX;
    std::optional<HistoricalKvHandle> reclaimed_handle;
    uint32_t reclaimed_begin = 0;
    for (uint32_t i = 0; i < 20; ++i) {
        const uint32_t begin = 2 * i;
        auto writer = repo->begin_write(1000 + i, begin, 2, i + 1, 0,
            [weak = std::weak_ptr<HistoricalKvRepository>(repo), storage](HistoricalKvHandle handle) {
                auto owner = weak.lock();
                return owner && storage->release(*owner, handle).code ==
                    HistoricalKvRawStore::ReleaseCode::Released;
            });
        check(writer && storage->reserve(*writer, writer->handle(), begin, 2),
              "cyclic writer/payload reservation failed");
        writers.push_back(std::move(*writer));
        allocation_budget = 0;
        const bool appended = runtime.register_historical_append(
            repo, storage, writers, producers, reservation.get());
        allocation_budget = -1;
        check(appended && producers.size() == 1 && runtime.store().block_count() <= 2 &&
              runtime.store().allocated_bytes() == table_bytes &&
              runtime.historical_shadow_directory().size() <= 2,
              "recycled metadata append allocated, grew, or failed");
        const uint32_t id = runtime.store().chronological_ids().back();
        check(runtime.store().blocks()[id].logical_id == 1000 + i &&
              runtime.store().block_id_containing(begin) == static_cast<int32_t>(id) &&
              (i < 2 || id != previous_id),
              "cyclic table index or original-position mapping was stale");
        if (reclaimed_handle) {
            const auto now = runtime.historical_shadow_directory().observe(id).handle;
            check(!repo->acquire_read(*reclaimed_handle) &&
                  runtime.store().block_id_containing(reclaimed_begin) == -1 &&
                  (now.slot != reclaimed_handle->slot ||
                   now.generation != reclaimed_handle->generation),
                  "old generation or position reopened after table-slot reuse");
            if (i == 2) {
                bool rejected = false;
                try { runtime.truncate_to(begin); }
                catch (const std::logic_error &) { rejected = true; }
                check(rejected && runtime.store().block_id_containing(begin) == static_cast<int32_t>(id),
                      "unsafe recycled table truncation changed the live generation");
            }
        }
        auto current_task = std::move(producers[0]);
        complete(current_task, 2);
        if (i >= 2)
            check(runtime.store().pick_topk_blocks() == std::vector<uint32_t>({previous_id, id}),
                  "recycled table slot changed chronological selector order");
        producers.clear(); writers.clear();
        if (previous_task) {
            check(repo->retire(previous_handle) == HistoricalKvRepository::RetireResult::Committed,
                  "cyclic prior generation did not retire");
            previous_task.reset();
            check(repo->try_reclaim(previous_handle) == HistoricalKvRepository::ReclaimResult::Complete &&
                  !repo->acquire_read(previous_handle),
                  "cyclic prior generation did not release exactly once");
            if (i == 1) {
                const auto plan = runtime.prepare_selection({});
                check(runtime.plan_is_current(plan) &&
                      !runtime.detach_reclaimed_historical_block(previous_id),
                      "reclaimed span detached during an unapplied selection plan");
                runtime.discard_pending();
            }
            check(runtime.detach_reclaimed_historical_block(previous_id) &&
                  runtime.store().block_id_containing(begin - 2) == -1,
                  "cyclic reclaimed span was not detached");
            if (i == 1) {
                runtime.prepare_selection({});
                check(!runtime.prepare_recycled_historical_slot(*reservation, previous_id),
                      "recycled slot was prepared during an unapplied selection plan");
                runtime.discard_pending();
            }
            check(runtime.prepare_recycled_historical_slot(*reservation, previous_id),
                  "cyclic reclaimed table slot was not prepared for a new generation");
            reclaimed_handle = previous_handle;
            reclaimed_begin = begin - 2;
        }
        previous_task = std::move(current_task);
        previous_handle = previous_task->handle();
        previous_id = id;
    }
    check(repo->retire(previous_handle) == HistoricalKvRepository::RetireResult::Committed,
          "final cyclic generation did not retire");
    previous_task.reset();
    check(repo->try_reclaim(previous_handle) == HistoricalKvRepository::ReclaimResult::Complete &&
          runtime.detach_reclaimed_historical_block(previous_id) &&
          repo->ready_count() == 2 && !storage->allocated_payload_bytes() &&
          runtime.store().block_count() == 2 && runtime.store().chronological_ids().empty(),
          "cyclic metadata/payload slots did not return to bounded Ready state");
}

static void persistent_domain_reuses_capacity_across_requests() {
    KvMemRuntime runtime(runtime_config());
    HistoricalKvCleanup cleanup(4);
    const auto initial_bytes = runtime.store().allocated_bytes();
    {
        auto cancelled = HistoricalKvCaptureDomain::try_prepare(
            runtime, storage_config(), 2, &cleanup);
        check(cancelled && runtime.store().allocated_bytes() == initial_bytes,
              "cancelled domain changed the runtime during preparation");
    }
    check(!cleanup.occupied() && !runtime.historical_shadow_directory().size(),
          "unadmitted domain leaked its cleanup credit or table");
    auto domain = HistoricalKvCaptureDomain::try_prepare(runtime, storage_config(), 2, &cleanup);
    check(domain && runtime.store().allocated_bytes() == initial_bytes &&
          !runtime.historical_shadow_directory().size(),
          "domain private preparation changed runtime before admission");
    allocation_budget = 0;
    const bool installed = domain->admit();
    allocation_budget = -1;
    check(installed && !domain->admit(), "domain did not commit its fixed table exactly once");
    check(domain && domain->metadata_remaining() == 2,
          "persistent domain did not reserve its bounded table");
    auto repository = domain->repository();
    auto storage = domain->storage();
    const auto table_bytes = runtime.store().allocated_bytes();
    HistoricalKvHandle previous{};
    uint64_t previous_use = 0;
    std::unique_ptr<HistoricalKvCaptureSession> finished_request;
    for (uint32_t request = 0; request < 24; ++request) {
        auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (cleanup.occupied() > 1 && std::chrono::steady_clock::now() < deadline)
            std::this_thread::yield();
        check(cleanup.occupied() == 1, "prior request private cleanup did not drain");
        auto session = HistoricalKvCaptureSession::try_prepare_next(
            domain, demand(4, 0, 1), &cleanup);
        check(session && session->reserved_blocks() == 1 && session->admit(),
              "reclaimed Ready slot was not admitted into the next request");
        finished_request.reset(); // Must not cancel the new request's domain claim.
        check(!HistoricalKvCaptureSession::try_prepare_next(domain, demand(4, 0, 1), &cleanup),
              "a second request bypassed the domain claim");
        check(session->next_use_sequence() > previous_use,
              "use sequence reset across request boundaries");
        previous_use = session->next_use_sequence();
        const auto frees_before = foreground_frees;
        allocation_budget = 0;
        const bool appended = session->append(4);
        allocation_budget = -1;
        check(appended && foreground_frees == frees_before,
              "persistent request append allocated or freed on the foreground");
        const auto id = runtime.store().chronological_ids().back();
        auto task = runtime.historical_shadow_directory().find(id)->writer_context;
        const auto handle = task->handle();
        session->producers_captured();
        complete(task, 4);
        task.reset();
        check(session->end_input() && session->finish(),
              "persistent request did not complete its own producers");
        finished_request = std::move(session);
        check(!repository->closed() &&
              runtime.store().block_count() <= 2 &&
              runtime.historical_shadow_directory().size() <= 2 &&
              runtime.store().allocated_bytes() == table_bytes &&
              runtime.store().block_id_containing(4 * request) == static_cast<int32_t>(id),
              "request teardown closed the domain or grew its physical table");
        if (request)
            check(!repository->acquire_read(previous),
                  "recycled request reopened an old handle generation");
        check(repository->retire(handle) == HistoricalKvRepository::RetireResult::Committed &&
              repository->try_reclaim(handle) == HistoricalKvRepository::ReclaimResult::Complete &&
              !storage->allocated_payload_bytes(),
              "request block was not physically released before reuse");
        previous = handle;
    }
    check(runtime.store().total_tokens() == 96 && repository->ready_count() == 2 &&
          runtime.store().block_count() == 2,
          "24 requests did not exceed physical capacity by twelvefold");
    finished_request.reset();
    domain.reset();
    wait_cleanup(cleanup);
    check(repository->closed(), "persistent domain teardown left writer admission open");
}

static void persistent_domain_rejects_pending_and_live_capacity() {
    KvMemRuntime runtime(runtime_config());
    HistoricalKvCleanup cleanup(3);
    auto domain = HistoricalKvCaptureDomain::try_prepare(runtime, storage_config(), 1, &cleanup);
    check(bool(domain), "single-slot domain setup failed");
    auto prepared = HistoricalKvCaptureSession::try_prepare_next(domain, demand(4, 0, 1), &cleanup);
    check(bool(prepared), "initial private request reservation failed");
    runtime.prepare_selection({});
    check(!prepared->admit(), "request admitted across a pending selection plan");
    runtime.discard_pending();
    prepared.reset();
    check(domain->repository()->ready_count() == 1 && domain->metadata_remaining() == 1,
          "cancelled request leaked its private RAM or metadata");
    check(!runtime.historical_shadow_directory().uses_reserved_table(),
          "cancelled initial request installed the domain table");
    auto session = HistoricalKvCaptureSession::try_prepare_next(domain, demand(4, 0, 1), &cleanup);
    check(session && session->admit() && session->append(4),
          "cancelled request prevented a later admission");
    auto task = runtime.historical_shadow_directory().find(0)->writer_context;
    const auto handle = task->handle();
    session->producers_captured();
    complete(task, 4);
    task.reset();
    check(session->end_input() && session->finish(), "single-slot request did not finish");
    session.reset();
    check(!HistoricalKvCaptureSession::try_prepare_next(domain, demand(4, 0, 1), &cleanup) &&
          domain->repository()->ready_count() == 0,
          "a Live slot was counted as a future reclaim credit");
    auto execution = domain->repository()->acquire_new(handle);
    check(bool(execution) &&
          domain->repository()->retire(handle) == HistoricalKvRepository::RetireResult::Committed &&
          domain->repository()->try_reclaim(handle) == HistoricalKvRepository::ReclaimResult::Deferred &&
          !HistoricalKvCaptureSession::try_prepare_next(domain, demand(4, 0, 1), &cleanup),
          "retired slot with an execution lease was reused");
    execution.reset();
    check(domain->repository()->try_reclaim(handle) == HistoricalKvRepository::ReclaimResult::Complete,
          "released execution lease did not permit exact reclaim");
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (cleanup.occupied() > 1 && std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
    auto next = HistoricalKvCaptureSession::try_prepare_next(domain, demand(4, 0, 1), &cleanup);
    check(next && next->admit() && next->append(4),
          "fully reclaimed single-slot generation was not reused");
    auto new_task = runtime.historical_shadow_directory().find(0)->writer_context;
    const auto new_handle = new_task->handle();
    check(!domain->repository()->acquire_read(handle) &&
          (handle.slot != new_handle.slot || handle.generation != new_handle.generation),
          "new request revived the old generation");
    next->producers_captured();
    complete(new_task, 4);
    new_task.reset();
    check(next->end_input() && next->finish(), "reused request did not finish");
    next.reset();
    domain.reset();
    wait_cleanup(cleanup);
}

static void reserved_metadata_mixes_fresh_and_recycled_slots_atomically() {
    KvMemRuntime runtime(runtime_config());
    auto repo = std::make_shared<HistoricalKvRepository>(3);
    auto storage = std::make_shared<HistoricalKvRawStore>(storage_config(), 3);
    auto reservation = runtime.prepare_historical_metadata(repo, storage, 3);
    check(reservation && runtime.install_historical_metadata(*reservation),
          "mixed table reservation failed");
    const auto table_bytes = runtime.store().allocated_bytes();
    auto write = [&](uint64_t logical_id, uint32_t begin) {
        auto writer = repo->begin_write(logical_id, begin, 2, logical_id, 0,
            [weak = std::weak_ptr<HistoricalKvRepository>(repo), storage](HistoricalKvHandle handle) {
                auto owner = weak.lock();
                return owner && storage->release(*owner, handle).code ==
                    HistoricalKvRawStore::ReleaseCode::Released;
            });
        check(writer && storage->reserve(*writer, writer->handle(), begin, 2),
              "mixed writer reservation failed");
        return std::move(*writer);
    };
    std::vector<HistoricalKvRepository::Lease> writers;
    std::vector<HistoricalKvShadowDirectory::Producer> producers;
    std::vector<HistoricalKvShadowDirectory::Producer> baseline;
    writers.reserve(2); producers.reserve(2);
    baseline.reserve(2);
    for (uint32_t i = 0; i < 2; ++i) {
        writers.push_back(write(1000 + i, 2 * i));
        check(runtime.register_historical_append(repo, storage, writers, producers, reservation.get()),
              "mixed baseline append failed");
        complete(producers.back(), 2);
        baseline.push_back(std::move(producers.back()));
        producers.clear();
        writers.clear();
    }
    const auto old_handle = baseline[0]->handle();
    check(repo->retire(old_handle) == HistoricalKvRepository::RetireResult::Committed,
          "mixed old generation did not retire");
    baseline.erase(baseline.begin());
    check(repo->try_reclaim(old_handle) == HistoricalKvRepository::ReclaimResult::Complete &&
          runtime.detach_reclaimed_historical_block(0) &&
          runtime.prepare_recycled_historical_slot(*reservation, 0) && reservation->remaining() == 2,
          "mixed reclaimed table slot was not prepared");
    writers.push_back(write(1002, 4));
    writers.push_back(write(1003, 7)); // Invalid second span must leave the fresh slot unused.
    const auto bad_handle = writers.back().handle();
    check(!runtime.register_historical_append(repo, storage, writers, producers, reservation.get()) &&
          writers[0] && writers[1] && producers.empty() &&
          runtime.store().block_count() == 2 && reservation->remaining() == 2,
          "mixed invalid batch partially committed");
    check(repo->retire(bad_handle) == HistoricalKvRepository::RetireResult::Committed,
          "mixed invalid writer did not retire");
    writers.back().reset();
    check(repo->try_reclaim(bad_handle) == HistoricalKvRepository::ReclaimResult::Complete,
          "mixed invalid writer did not release");
    writers.back() = write(1003, 6);
    allocation_budget = 0;
    const bool appended = runtime.register_historical_append(repo, storage, writers, producers, reservation.get());
    allocation_budget = -1;
    check(appended && producers.size() == 2 && runtime.store().block_count() == 3 &&
          runtime.store().chronological_ids() == std::vector<uint32_t>({1, 2, 0}) &&
          runtime.store().block_id_containing(4) == 2 &&
          runtime.store().block_id_containing(6) == 0 &&
          !repo->acquire_read(old_handle) && !reservation->remaining(),
          "mixed fresh/recycled append lost atomic order or revived old generation");
    bool blocked = false;
    try { runtime.truncate_to(4); }
    catch (const std::logic_error &) { blocked = true; }
    check(blocked && runtime.store().total_tokens() == 8 &&
          runtime.store().chronological_ids() == std::vector<uint32_t>({1, 2, 0}),
          "rollback removed a live recycled generation");
    const auto first_new = producers[0]->handle();
    const auto second_new = producers[1]->handle();
    complete(producers[0], 2);
    complete(producers[1], 2);
    check(repo->retire(first_new) == HistoricalKvRepository::RetireResult::Committed &&
          repo->retire(second_new) == HistoricalKvRepository::RetireResult::Committed,
          "mixed suffix retirement failed");
    producers.clear(); writers.clear();
    check(repo->try_reclaim(first_new) == HistoricalKvRepository::ReclaimResult::Complete &&
          repo->try_reclaim(second_new) == HistoricalKvRepository::ReclaimResult::Complete,
          "mixed suffix payloads did not release");
    blocked = false;
    try { runtime.truncate_to(5); }
    catch (const std::logic_error &) { blocked = true; }
    check(blocked && runtime.store().total_tokens() == 8,
          "rollback cut through an immutable historical generation");
    runtime.prepare_selection({});
    blocked = false;
    try { runtime.truncate_to(4); }
    catch (const std::logic_error &) { blocked = true; }
    check(blocked && runtime.store().total_tokens() == 8,
          "rollback bypassed an unapplied selection plan");
    runtime.discard_pending();
    runtime.store().set_block_gpu_slot(0, 7);
    blocked = false;
    try { runtime.truncate_to(4); }
    catch (const std::logic_error &) { blocked = true; }
    check(blocked && runtime.store().chronological_ids() == std::vector<uint32_t>({1, 2, 0}),
          "rollback released a suffix with an outstanding GPU slot");
    runtime.store().set_block_gpu_slot(0, -1);
    const auto before_rollback = runtime.store().table_revision();
    const auto dropped = runtime.truncate_to(4);
    check(dropped.size() == 2 && dropped[0].block_id == 2 && dropped[1].block_id == 0 &&
          runtime.store().block_count() == 3 && runtime.store().allocated_bytes() == table_bytes &&
          runtime.store().total_tokens() == 4 &&
          runtime.store().table_revision() > before_rollback &&
          runtime.store().chronological_ids() == std::vector<uint32_t>({1}) &&
          runtime.store().block_id_containing(4) == -1 &&
          runtime.historical_shadow_directory().size() == 3 &&
          runtime.prepare_recycled_historical_slot(*reservation, 0) &&
          runtime.prepare_recycled_historical_slot(*reservation, 2),
          "rollback by original position lost stable recycled table slots");
    writers.push_back(write(1004, 4));
    allocation_budget = 0;
    const bool resumed = runtime.register_historical_append(repo, storage, writers, producers, reservation.get());
    allocation_budget = -1;
    check(resumed && runtime.store().block_count() == 3 &&
          runtime.store().chronological_ids() == std::vector<uint32_t>({1, 0}) &&
          runtime.store().block_id_containing(4) == 0 &&
          !repo->acquire_read(first_new) && !repo->acquire_read(second_new),
          "rollback did not preserve bounded new-generation append");
}

static void unadmitted_rollback_with_stalled_cleanup() {
    std::promise<void> entered, resume;
    auto entered_signal = entered.get_future();
    auto resume_signal = resume.get_future().share();
    HistoricalKvCleanup cleanup(2);
    struct Stall final : HistoricalKvCleanup::Task {
        std::promise<void> & entered;
        std::shared_future<void> resume;
        Stall(std::promise<void> & e, std::shared_future<void> r) : entered(e), resume(std::move(r)) {}
        ~Stall() override { entered.set_value(); resume.wait(); }
    };
    struct Unblock {
        std::promise<void> & resume;
        ~Unblock() { resume.set_value(); }
    } unblock{resume};
    auto ticket = cleanup.try_reserve();
    ticket->submit(std::make_unique<Stall>(entered, resume_signal));
    check(entered_signal.wait_for(std::chrono::seconds(5)) == std::future_status::ready,
          "private rollback stall was not reached");
    KvMemRuntime runtime(runtime_config());
    const auto bytes = runtime.store().allocated_bytes();
    auto d = demand(9, 0, 1);
    d.private_scratch_bytes = 4096;
    for (unsigned i = 0; i < 3; ++i) {
        auto session = HistoricalKvCaptureSession::try_prepare(runtime, storage_config(), 3, d, &cleanup);
        check(session && cleanup.occupied() == 2, "private rollback did not return admission credit");
        auto storage = session->storage();
        std::weak_ptr<HistoricalKvRepository> repository = session->repository();
        allocation_budget = 0;
        if (i == 1) session->fail(); // Explicit cancellation uses the same immediate rollback.
        session.reset();
        allocation_budget = -1;
        check(!storage->allocated_payload_bytes() && repository.expired() && cleanup.occupied() == 1 &&
              !runtime.store().block_count() && runtime.store().allocated_bytes() == bytes,
              "unadmitted rollback waited for cleanup or retained private allocations");
    }
}

static void unused_reservation_handoff_with_stalled_cleanup(bool cancel = false) {
    std::promise<void> entered, resume;
    auto entered_signal = entered.get_future();
    auto resume_signal = resume.get_future().share();
    HistoricalKvCleanup cleanup(2);
    // Simulate a slow CPU allocator on the worker. Production tasks use
    // nonblocking ready() to postpone domains with unfinished GC callbacks.
    struct Stall final : HistoricalKvCleanup::Task {
        std::promise<void> & entered;
        std::shared_future<void> resume;
        Stall(std::promise<void> & e, std::shared_future<void> r) : entered(e), resume(std::move(r)) {}
        ~Stall() override { entered.set_value(); resume.wait(); }
    };
    struct Unblock {
        std::promise<void> & resume;
        bool opened = false;
        void open() { if (!opened) { resume.set_value(); opened = true; } }
        ~Unblock() { open(); }
    } unblock{resume};
    auto ticket = cleanup.try_reserve();
    ticket->submit(std::make_unique<Stall>(entered, resume_signal));
    check(entered_signal.wait_for(std::chrono::seconds(5)) == std::future_status::ready,
          "cleanup stall barrier was not reached");
    KvMemRuntime runtime(runtime_config());
    auto d = cancel ? demand(8, 0, 2) : demand(4, 4, 2);
    d.private_scratch_bytes = 4096;
    auto session = HistoricalKvCaptureSession::try_create(runtime, storage_config(), 3, d, &cleanup);
    check(session && session->append(4), "stalled cleanup fixture was not admitted");
    auto producer = runtime.historical_shadow_directory().find(0)->writer_context;
    session->producers_captured();
    if (cancel) check(!session->cancel(), "cancellation ignored an unfinished producer");
    complete(producer, 4);
    if (cancel) check(!session->cancel(), "cancellation ignored a live task lease");
    producer.reset();
    check(session->end_input() != cancel, "partial input was mistaken for completed input");
    const auto frees_before = foreground_frees;
    allocation_budget = 0;
    const bool finished = cancel ? session->cancel() : session->finish();
    allocation_budget = -1;
    check(finished && foreground_frees == frees_before &&
          session->payload_bytes() == 60 && cleanup.occupied() == 2,
          "finish waited, freed unused RAM, or returned credit before worker cleanup");
    KvMemRuntime another(runtime_config());
    check(!HistoricalKvCaptureSession::try_create(another, storage_config(), 3, d, &cleanup) &&
          !another.store().total_tokens(), "full cleanup queue admitted another request");
    unblock.open();
    wait_cleanup(cleanup);
    check(session->payload_bytes() == 20 && session->repository()->ready_count() == 2,
          "worker did not return unused payload and slot credits");
    auto access = runtime.historical_shadow_directory().acquire_read(0);
    check(bool(access), "private cleanup released installed history");
    check(session->finished() && !session->failed() && !session->repository()->closed() &&
          !session->append(1) && !session->cancel(), "completed/cancelled session reopened");
}

int main() {
    try {
        quotas_and_handoff();
        retired_blocks_do_not_invalidate_completed_request();
        request_finishes_while_background_release_is_blocked();
        boundary_limit();
        wait_cleanup();
        cancelled_owner_keeps_tasks();
        failures_are_atomic();
        wait_cleanup();
        reserved_metadata_batch_is_atomic();
        reserved_metadata_reuses_reclaimed_table_slots();
        persistent_domain_reuses_capacity_across_requests();
        persistent_domain_rejects_pending_and_live_capacity();
        reserved_metadata_mixes_fresh_and_recycled_slots_atomically();
        unadmitted_rollback_with_stalled_cleanup();
        unused_reservation_handoff_with_stalled_cleanup();
        unused_reservation_handoff_with_stalled_cleanup(true);
        std::puts("PASS historical capture session: private prepare/admit, immediate rollback, reserved metadata, bounded quotas, publication, cancellation, allocation faults");
    } catch (const std::exception & e) {
        allocation_budget = -1;
        std::fprintf(stderr, "FAIL historical capture session: %s\n", e.what());
        return 1;
    }
}
