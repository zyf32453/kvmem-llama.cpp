#include "kvmem/historical_kv_capture_session.hpp"
#include "kvmem/nvme_kv_tier.hpp"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <new>
#include <stdexcept>
#include <thread>
#include <sys/resource.h>
#include <unistd.h>

using namespace kvmem;
using Repo = HistoricalKvRepository;
using Store = HistoricalKvRawStore;
using Resource = Repo::HostResource;
static thread_local bool deny_allocations = false;
void * operator new(size_t n) {
    if (deny_allocations) throw std::bad_alloc();
    if (void * p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void * operator new[](size_t n) { return ::operator new(n); }
void operator delete(void * p) noexcept { std::free(p); }
void operator delete[](void * p) noexcept { std::free(p); }
void operator delete(void * p, size_t) noexcept { std::free(p); }
void operator delete[](void * p, size_t) noexcept { std::free(p); }
struct NoAllocations {
    NoAllocations() { deny_allocations = true; }
    ~NoAllocations() { deny_allocations = false; }
};
static void check(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}
static bool succeeded(Store::ReleaseCode code) {
    return code == Store::ReleaseCode::Released || code == Store::ReleaseCode::AlreadyReleased ||
        code == Store::ReleaseCode::Absent;
}
static HistoricalKvRawStoreConfig config(const std::string & dir, unsigned extents = 1) {
    // Hybrid layout: only physical layers 0 and 2 have attention payloads.
    HistoricalKvRawStoreConfig out{3, 4, 2, 3, 1, {0, 2}};
    out.nvme_dir = dir;
    out.nvme_bytes = uint64_t(extents) * 3 * 4 * 5;
    return out;
}
static Repo::ResourceRelease binding(Repo & repo, std::shared_ptr<Store> store,
                                     const Store::PreparedPayload & payload) {
    return {[&repo, store](HistoricalKvHandle handle, Resource resource) {
        return succeeded(store->release_resource(repo, handle, resource).code);
    }, payload.packed_kv_bytes(), payload.mean_k_bytes(), payload.nvme_bytes(), payload.writer_resource()};
}
static Repo::Lease install(Repo & repo, std::shared_ptr<Store> store, uint64_t id, uint32_t tokens = 3) {
    auto payload = store->prepare_payload(4);
    auto writer = repo.begin_write(id, 713 + 5 * id, tokens, id + 1, payload.payload_bytes(), binding(repo, store, payload));
    check(bool(writer), "writer admission failed");
    check(store->install_payload(*writer, writer->handle(), 713 + 5 * id, tokens, payload), "payload install failed");
    return std::move(*writer);
}
static void fill(Store & store, Repo & repo, Repo::Lease & writer, uint8_t value = 19) {
    const auto handle = writer.handle();
    // Whole-layer and incremental paths write the same short/non-aligned data.
    check(store.write_layer(writer, handle, 0, std::vector<uint8_t>(6, value),
        std::vector<uint8_t>(9, value + 1), {6.f}, 3), "whole-layer write failed");
    check(store.write_slice(writer, handle, 2, 0, 1, std::vector<uint8_t>(2, value),
        std::vector<uint8_t>(3, value + 1), {2.f}, 1), "first disk slice failed");
    check(store.write_packed_k_slice(writer, handle, 2, 1, 2, std::vector<uint8_t>(4, value)), "K disk slice failed");
    check(store.write_packed_v_slice(writer, handle, 2, 1, 2, std::vector<uint8_t>(6, value + 1)), "V disk slice failed");
    check(store.write_mean_k_slice(writer, handle, 2, 1, 2, {4.f}), "mean slice failed");
    check(store.publish_complete(repo, writer, handle), "disk payload did not publish");
}
static void read(Store & store, const Store::NvmeRead & reader, uint8_t value = 19) {
    std::vector<uint8_t> k(6), v(9);
    for (unsigned layer : {0u, 2u}) {
        check(store.read_nvme_layer(reader, layer, k.data(), k.size(), v.data(), v.size()), "disk read rejected");
        check(k == std::vector<uint8_t>(6, value) && v == std::vector<uint8_t>(9, value + 1), "disk bytes changed");
    }
    check(!store.read_nvme_layer(reader, 1, k.data(), k.size(), v.data(), v.size()), "non-attention layer accepted");
    check(!store.read_nvme_layer(reader, 0, k.data(), k.size() - 1, v.data(), v.size()), "wrong read length accepted");
}
static void retire(Repo & repo, HistoricalKvHandle handle) {
    check(repo.retire(handle) == Repo::RetireResult::Committed, "retire failed");
    check(!repo.acquire_external_read(handle), "retirement admitted a new external reader");
}
static void reclaim(Repo & repo, HistoricalKvHandle handle) {
    check(repo.try_reclaim(handle) == Repo::ReclaimResult::Complete, "final reclaim did not complete");
}

static void independent_resources(const std::string & dir) {
    auto store = std::make_shared<Store>(config(dir), 2);
    Repo repo(2);
    auto writer = install(repo, store, 1);
    const auto handle = writer.handle();
    fill(*store, repo, writer);
    writer.reset();
    auto disk_reader = store->acquire_nvme_read(repo, handle);
    check(bool(disk_reader), "live disk reader denied");
    const auto old_allocation = disk_reader->allocation_handle();
    retire(repo, handle);
    check(repo.try_reclaim(handle) == Repo::ReclaimResult::HostReleased, "CPU did not release independently of disk IO");
    auto meta = repo.inspect(handle);
    check(meta && meta->host_released && !meta->nvme_released && meta->nvme_bytes == 60 &&
        !store->allocated_payload_bytes() && store->allocated_nvme_bytes() == 60, "independent byte accounting wrong");
    check(!store->acquire_nvme_read(repo, handle), "retired disk read admitted");
    bool full = false;
    try { auto private_payload = store->prepare_payload(4); }
    catch (const std::bad_alloc &) { full = true; }
    check(full, "pending old disk IO was counted as free capacity");

    // Actual read syscalls continue while the background reclaimer repeatedly
    // closes/retries the same allocation. The CPU vectors are already gone.
    std::atomic<bool> stop{false};
    std::atomic<unsigned> scans{0};
    std::thread gc([&] {
        NoAllocations guard;
        while (!stop.load(std::memory_order_acquire)) {
            repo.try_reclaim(handle); scans.fetch_add(1, std::memory_order_release); std::this_thread::yield();
        }
    });
    while (!scans.load(std::memory_order_acquire)) std::this_thread::yield();
    try { for (unsigned i = 0; i < 64; ++i) read(*store, *disk_reader); }
    catch (...) { stop.store(true, std::memory_order_release); gc.join(); throw; }
    stop.store(true, std::memory_order_release); gc.join();
    disk_reader.reset();
    reclaim(repo, handle);
    check(!store->allocated_nvme_bytes(), "drained disk allocation was not returned");

    // A Host reader may retain CPU while a completed disk copy is independently
    // returned and reused by another historical slot.
    writer = install(repo, store, 2);
    const auto host_handle = writer.handle();
    fill(*store, repo, writer, 42); writer.reset();
    auto host_reader = repo.acquire_read(host_handle);
    auto transient = store->acquire_nvme_read(repo, host_handle);
    check(transient && transient->allocation_handle().generation > old_allocation.generation, "extent generation did not advance");
    transient.reset();
    retire(repo, host_handle);
    check(repo.try_reclaim(host_handle) == Repo::ReclaimResult::ResourceReleased, "disk did not release ahead of CPU");
    check(!store->allocated_nvme_bytes() && store->allocated_payload_bytes() > 0, "Host reader pinned disk unnecessarily");
    auto next = install(repo, store, 3);
    const auto next_handle = next.handle();
    fill(*store, repo, next, 75); next.reset();
    auto next_read = store->acquire_nvme_read(repo, next_handle);
    read(*store, *next_read, 75);
    auto cpu = store->read_layer(*host_reader, host_handle, 0);
    check(cpu && cpu->packed_k == std::vector<uint8_t>(6, 42), "extent reuse affected retained Host copy");
    host_reader.reset(); reclaim(repo, host_handle);
    read(*store, *next_read, 75); // old metadata cleanup must not free the new extent
    next_read.reset(); retire(repo, next_handle); reclaim(repo, next_handle);
}

static void producer_cancellation_and_install_race(const std::string & dir) {
    auto store = std::make_shared<Store>(config(dir), 1);
    Repo repo(1);
    auto payload = store->prepare_payload(4);
    auto writer = repo.begin_write(1, 27, 3, 1, payload.payload_bytes(), binding(repo, store, payload));
    check(bool(writer), "pre-install writer failed");
    const auto handle = writer->handle();
    retire(repo, handle);
    repo.try_reclaim(handle);
    check(!repo.inspect(handle)->nvme_released && store->allocated_nvme_bytes() == 60,
          "retired, not-yet-installed disk allocation was acknowledged absent");
    check(store->install_payload(*writer, handle, 27, 3, payload), "admitted old writer could not install");
    check(store->write_packed_k_slice(*writer, handle, 0, 0, 1, {31, 32}), "old producer lost disk authorization");
    repo.try_reclaim(handle);
    check(store->write_packed_v_slice(*writer, handle, 0, 0, 1, {41, 42, 43}), "retirement prevented admitted disk write");
    check(!store->publish_complete(repo, *writer, handle), "partial retired writer published");
    writer.reset(); // Unfinished producer drops its last IO right, without publication.
    reclaim(repo, handle);
    check(!store->allocated_nvme_bytes() && !store->allocated_payload_bytes(), "cancel left permanent disk claim");

    // A completed packed writer relinquishes disk IO before Mean-K completes.
    auto late = install(repo, store, 2);
    const auto late_handle = late.handle();
    retire(repo, late_handle);
    for (unsigned layer : {0u, 2u}) {
        check(store->write_packed_k_slice(late, late_handle, layer, 0, 3, std::vector<uint8_t>(6, 7)), "late K failed");
        check(store->write_packed_v_slice(late, late_handle, layer, 0, 3, std::vector<uint8_t>(9, 8)), "late V failed");
    }
    check(!store->publish_complete(repo, late, late_handle), "missing Mean-K published");
    repo.try_reclaim(late_handle);
    check(!store->allocated_nvme_bytes() && repo.inspect(late_handle)->nvme_released,
          "completed packed producer held disk through unfinished Mean-K");
    for (unsigned layer : {0u, 2u})
        check(store->write_mean_k_slice(late, late_handle, layer, 0, 3, {9.f}), "old Mean-K could not finish after disk release");
    check(!store->publish_complete(repo, late, late_handle), "retirement resurrected after final mean");
    late.reset(); reclaim(repo, late_handle);
}

static void reservations_and_reuse(const std::string & dir) {
    auto store = std::make_shared<Store>(config(dir), 2); // 2 RAM slots, only 1 disk extent
    auto repo = std::make_shared<Repo>(2);
    HistoricalKvRamDemand demand;
    demand.uncached_input_tokens = 8; demand.max_accepted_output_tokens = 0; demand.max_append_boundaries = 1;
    check(!HistoricalKvRamReservation::try_reserve(repo, store, demand), "disk shortfall admitted request");
    check(!store->allocated_payload_bytes() && !store->allocated_nvme_bytes() && repo->observe_range(0, 2).ready == 2,
          "failed multi-extent preparation did not roll back all credits");
    demand.uncached_input_tokens = 3;
    auto reservation = HistoricalKvRamReservation::try_reserve(repo, store, demand);
    check(reservation && reservation->reserved_nvme_bytes() == 60, "request did not reserve worst-case disk extent");
    reservation.reset();
    uint64_t generation = 0;
    for (unsigned i = 0; i < 32; ++i) { // 32x disk capacity, 16x repository capacity
        auto writer = install(*repo, store, 100 + i);
        const auto handle = writer.handle();
        fill(*store, *repo, writer, uint8_t(i + 1)); writer.reset();
        auto reader = store->acquire_nvme_read(*repo, handle);
        check(reader && reader->allocation_handle().generation > generation, "reuse retained stale allocation identity");
        generation = reader->allocation_handle().generation;
        read(*store, *reader, uint8_t(i + 1)); reader.reset();
        retire(*repo, handle); reclaim(*repo, handle);
        check(!store->allocated_payload_bytes() && !store->allocated_nvme_bytes() && repo->observe_range(0, 2).ready == 2,
              "long reuse lost payload, disk or Ready credit");
    }
    // Private reservation owns its arena even after its store is destroyed.
    auto detached = store->prepare_payload(4);
    repo.reset(); store.reset(); detached = {};
}

static void reader_move_and_ack_retry(const std::string & dir) {
    auto store = std::make_shared<Store>(config(dir, 2), 2);
    Repo repo(2);
    auto first = install(repo, store, 1); const auto a = first.handle();
    fill(*store, repo, first); first.reset();
    auto second = install(repo, store, 2); const auto b = second.handle();
    fill(*store, repo, second, 51); second.reset();
    auto ar = store->acquire_nvme_read(repo, a), br = store->acquire_nvme_read(repo, b);
    retire(repo, a); retire(repo, b);
    repo.try_reclaim(a); repo.try_reclaim(b);
    *ar = std::move(*br); // drops old IO before old allocation ownership
    br.reset(); reclaim(repo, a);
    read(*store, *ar, 51); ar.reset(); reclaim(repo, b);

    // Withhold first disk acknowledgement after physical return, reuse that
    // extent, then retry old confirmation. It cannot close the new generation.
    auto payload = store->prepare_payload(4);
    auto release = binding(repo, store, payload);
    auto callback = std::move(release.callback);
    bool lost = false;
    release.callback = [callback, &lost](HistoricalKvHandle h, Resource r) {
        const bool done = callback(h, r);
        if (r == Resource::Nvme && done && !lost) { lost = true; return false; }
        return done;
    };
    auto writer = repo.begin_write(3, 17, 3, 3, payload.payload_bytes(), std::move(release));
    const auto old = writer->handle();
    check(store->install_payload(*writer, old, 17, 3, payload), "retry fixture install failed");
    fill(*store, repo, *writer); writer.reset(); retire(repo, old); repo.try_reclaim(old);
    check(lost && !store->allocated_nvme_bytes(), "lost disk acknowledgement did not return physical capacity");
    auto newer = install(repo, store, 4); const auto next = newer.handle();
    fill(*store, repo, newer, 85); newer.reset();
    reclaim(repo, old);
    auto reader = store->acquire_nvme_read(repo, next); read(*store, *reader, 85);
    reader.reset(); retire(repo, next); reclaim(repo, next);
}

static void capture_handoff(const std::string & dir) {
    KvMemRuntimeConfig rc; rc.store.block_tokens = 4;
    KvMemRuntime runtime(rc);
    auto cfg = config(dir, 2);
    auto domain = HistoricalKvCaptureDomain::try_create(runtime, cfg, 2);
    HistoricalKvRamDemand demand;
    demand.uncached_input_tokens = 3; demand.max_accepted_output_tokens = 0; demand.max_append_boundaries = 1;
    for (unsigned i = 0; i < 24; ++i) {
        auto session = HistoricalKvCaptureSession::try_prepare_next(domain, demand);
        check(session && session->admit(), "persistent disk session admission failed");
        const auto start = runtime.store().total_tokens();
        {
            NoAllocations guard;
            check(session->append(3), "disk append allocated or failed");
        }
        HistoricalKvShadowDirectory::Producer task;
        uint32_t block_id = 0;
        for (uint32_t id = 0; id < runtime.store().block_count(); ++id) {
            const auto entry = runtime.historical_shadow_directory().find(id);
            if (entry && entry->writer_context && entry->original_begin == start) {
                task = entry->writer_context;
                block_id = id;
            }
        }
        check(bool(task), "disk session producer missing");
        session->producers_captured();
        const auto handle = task->handle();
        auto storage = task->storage();
        const std::vector<uint8_t> k(6, 29), v(9, 30);
        const std::vector<float> mean{6.f};
        {
            NoAllocations guard;
            for (unsigned layer : {0u, 2u})
                check(storage->write_layer(task->writer(), handle, layer, k, v, mean, 3), "captured disk write failed");
            check(storage->publish_complete(task->repository(), task->writer(), handle), "captured disk publish failed");
        }
        auto reader = storage->acquire_nvme_read(task->repository(), handle);
        check(bool(reader), "captured disk read denied"); read(*storage, *reader, 29); reader.reset();
        task.reset();
        check(session->end_input(), "disk request input did not finish");
        auto stage_in = runtime.historical_shadow_directory().acquire_stage_in(block_id);
        check(stage_in && stage_in.from_nvme() && !stage_in.host.lease, "stage-in pinned an unrelated Host copy");
        retire(*session->repository(), handle);
        check(session->repository()->try_reclaim(handle) == Repo::ReclaimResult::HostReleased,
              "stage-in source retained CPU payload");
        std::vector<uint8_t> actual_k(6), actual_v(9);
        {
            NoAllocations guard;
            check(stage_in.copy_layer(2, actual_k.data(), actual_k.size(), actual_v.data(), actual_v.size()) &&
                  actual_k == k && actual_v == v, "old stage-in disk source changed after Host release");
            check(!runtime.historical_shadow_directory().acquire_stage_in(block_id), "new stage-in resurrected retired source");
        }
        stage_in = {}; reclaim(*session->repository(), handle);
        check(session->finish(), "disk request did not finish after old transfer");
        check(!storage->allocated_nvme_bytes(), "disk request did not recycle extent");
    }
}

static size_t fd_count() {
    size_t n = 0;
    for (const auto & entry : std::filesystem::directory_iterator("/proc/self/fd")) { (void) entry; ++n; }
    return n;
}
static void preparation_failure_cleanup(const std::string & dir) {
#if defined(__linux__)
    NvmeKvTierConfig cfg; cfg.dir = dir; cfg.total_bytes = 4096; cfg.slot_bytes = 4096;
    cfg.preallocate = true; cfg.handle_addressed = true;
    const auto before = fd_count();
    struct rlimit original{}; check(!getrlimit(RLIMIT_FSIZE, &original), "getrlimit failed");
    auto limited = original; limited.rlim_cur = 512;
    const auto previous_signal = std::signal(SIGXFSZ, SIG_IGN);
    check(!setrlimit(RLIMIT_FSIZE, &limited), "setrlimit failed");
    bool refused = false;
    try { NvmeKvTier tier(cfg); } catch (const std::runtime_error &) { refused = true; }
    check(!setrlimit(RLIMIT_FSIZE, &original), "restore rlimit failed");
    std::signal(SIGXFSZ, previous_signal);
    check(refused && fd_count() == before && std::filesystem::is_empty(dir),
          "failed preallocation leaked file or descriptor");
#endif
}

static void failed_write_keeps_unpublished_prefix(const std::string & dir) {
#if defined(__linux__)
    auto store = std::make_shared<Store>(config(dir), 1);
    Repo repo(1);
    auto writer = install(repo, store, 1);
    const auto handle = writer.handle();
    struct rlimit original{}; check(!getrlimit(RLIMIT_FSIZE, &original), "get write limit failed");
    auto limited = original; limited.rlim_cur = 4;
    const auto previous_signal = std::signal(SIGXFSZ, SIG_IGN);
    check(!setrlimit(RLIMIT_FSIZE, &limited), "set write limit failed");
    bool failed = false;
    try { store->write_packed_k_slice(writer, handle, 0, 0, 3, std::vector<uint8_t>(6, 19)); }
    catch (const std::runtime_error &) { failed = true; }
    check(!setrlimit(RLIMIT_FSIZE, &original), "restore write limit failed");
    std::signal(SIGXFSZ, previous_signal);
    const auto progress = store->write_progress(writer, handle, 0);
    check(failed && progress && !progress->packed_k_tokens && !store->acquire_nvme_read(repo, handle) &&
          !store->publish_complete(repo, writer, handle), "failed partial syscall published a completed prefix");
    // Retry overwrites the private incomplete bytes before marking progress.
    fill(*store, repo, writer); writer.reset();
    auto reader = store->acquire_nvme_read(repo, handle); read(*store, *reader);
    reader.reset(); retire(repo, handle); reclaim(repo, handle);
    check(!store->allocated_nvme_bytes(), "failed/retried disk write lost capacity");
#endif
}

int main() {
    try {
        const auto root = std::filesystem::temp_directory_path() /
            ("historical-nvme-" + std::to_string(::getpid()));
        std::filesystem::create_directories(root);
        const auto dir = root.string();
        preparation_failure_cleanup(dir);
        failed_write_keeps_unpublished_prefix(dir);
        independent_resources(dir);
        producer_cancellation_and_install_race(dir);
        reservations_and_reuse(dir);
        reader_move_and_ack_retry(dir);
        capture_handoff(dir);
        check(std::filesystem::is_empty(root), "ephemeral arena left directory entries");
        std::filesystem::remove(root);
        std::cout << "historical NVMe: independent resources, old IO, cancellation, rollback, 32 generations, 24 requests passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "historical NVMe test failed: " << error.what() << '\n'; return 1;
    }
}
