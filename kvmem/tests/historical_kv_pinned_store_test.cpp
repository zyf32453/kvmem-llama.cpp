#include "kvmem/historical_kv_raw_store.hpp"
#include "kvmem/historical_kv_capture_session.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <new>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace kvmem;
using Repo = HistoricalKvRepository;
using Store = HistoricalKvRawStore;

static void check(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

static bool released(Store::ReleaseCode code) {
    return code == Store::ReleaseCode::Released || code == Store::ReleaseCode::AlreadyReleased ||
           code == Store::ReleaseCode::Absent;
}

static Repo::ResourceRelease binding(Repo & repo, const std::shared_ptr<Store> & store,
                                     const Store::PreparedPayload & payload) {
    return {[&repo, store](HistoricalKvHandle handle, Repo::HostResource resource) {
        return released(store->release_resource(repo, handle, resource).code);
    }, payload.packed_kv_bytes(), payload.mean_k_bytes(), payload.nvme_bytes(),
       payload.writer_resource(), payload.pinned_bytes()};
}

static void fill(Store & store, Repo & repo, Repo::Lease & writer, uint8_t seed) {
    for (uint32_t layer = 0; layer < 2; ++layer) {
        const auto k = std::vector<uint8_t>(8, seed + layer);
        const auto v = std::vector<uint8_t>(12, seed + layer + 10);
        check(store.write_layer(writer, writer.handle(), layer, k, v, {}, 0), "pinned source write failed");
    }
    check(store.publish_complete(repo, writer, writer.handle()), "pinned source did not publish");
}

static void captured_stage_in() {
    KvMemRuntimeConfig runtime_config;
    runtime_config.store.block_tokens = 4;
    KvMemRuntime runtime(runtime_config);
    HistoricalKvRawStoreConfig config{2, 4, 2, 3, 0, {}};
    config.pinned_bytes = 40;
    config.allocate_pinned = [](size_t bytes) -> std::shared_ptr<uint8_t> {
        return {new uint8_t[bytes], std::default_delete<uint8_t[]>()};
    };
    auto domain = HistoricalKvCaptureDomain::try_create(runtime, config, 1);
    HistoricalKvRamDemand demand;
    demand.uncached_input_tokens = 3;
    demand.max_accepted_output_tokens = 0;
    demand.max_append_boundaries = 1;
    auto session = HistoricalKvCaptureSession::try_prepare_next(domain, demand);
    check(session && session->admit() && session->append(3), "pinned capture admission failed");
    auto binding = runtime.historical_shadow_directory().find(0);
    check(binding && binding->writer_context, "pinned capture producer missing");
    auto producer = binding->writer_context;
    session->producers_captured();
    const auto handle = producer->handle();
    for (uint32_t layer = 0; layer < 2; ++layer)
        check(producer->storage()->write_layer(producer->writer(), handle, layer,
              std::vector<uint8_t>(6, 31 + layer), std::vector<uint8_t>(9, 41 + layer), {}, 0),
              "pinned capture write failed");
    check(producer->storage()->publish_complete(producer->repository(), producer->writer(), handle),
          "pinned capture publish failed");
    producer.reset();
    binding.reset();
    check(session->end_input(), "pinned capture input did not finish");
    auto source = runtime.historical_shadow_directory().acquire_stage_in(0);
    check(source && source.from_pinned() && !source.from_nvme() && !source.host.lease,
          "stage-in retained RAM or chose the wrong source");
    const auto borrowed = source.pinned_layer(1);
    check(borrowed && borrowed->packed_k[0] == 32 && borrowed->packed_v[0] == 42,
          "stage-in did not borrow the pinned source directly");
    check(session->repository()->retire(handle) == Repo::RetireResult::Committed &&
          session->repository()->try_reclaim(handle) == Repo::ReclaimResult::HostReleased,
          "pinned stage-in kept RAM alive after retirement");
    std::vector<uint8_t> k(6), v(9);
    check(source.copy_layer(1, k.data(), k.size(), v.data(), v.size()) &&
          k == std::vector<uint8_t>(6, 32) && v == std::vector<uint8_t>(9, 42) &&
          !runtime.historical_shadow_directory().acquire_stage_in(0),
          "old pinned stage-in lost data or new stage-in crossed retirement");
    source = {};
    check(session->repository()->try_reclaim(handle) == Repo::ReclaimResult::Complete &&
          !session->storage()->allocated_pinned_bytes() && session->finish(),
          "pinned captured extent did not finish reclamation");
}

static void deallocation_precedes_reuse(bool installed) {
    std::atomic<bool> entered{false}, proceed{false}, timed_out{false};
    const auto await = [](const std::atomic<bool> & flag, int seconds) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
        while (!flag.load(std::memory_order_acquire)) {
            if (std::chrono::steady_clock::now() >= deadline) return false;
            std::this_thread::yield();
        }
        return true;
    };
    std::atomic<bool> block_first_free{true};
    HistoricalKvRawStoreConfig config{2, 4, 2, 3, 0, {}};
    config.pinned_bytes = 40;
    config.allocate_pinned = [&](size_t bytes) -> std::shared_ptr<uint8_t> {
        return {new uint8_t[bytes], [&](uint8_t * p) {
            if (block_first_free.exchange(false)) {
                entered.store(true, std::memory_order_release);
                timed_out.store(!await(proceed, 10), std::memory_order_release);
            }
            delete[] p;
        }};
    };
    auto store = std::make_shared<Store>(config, 2); // RAM has room; pinned does not.
    Repo repo(2);
    auto payload = store->prepare_payload(4);
    HistoricalKvHandle handle;
    if (installed) {
        auto writer = repo.begin_write(1, 0, 4, 1, payload.payload_bytes(), binding(repo, store, payload));
        check(bool(writer), "deallocation fixture admission failed");
        handle = writer->handle();
        check(store->install_payload(*writer, handle, 0, 4, payload), "deallocation fixture install failed");
        fill(*store, repo, *writer, 12);
        writer.reset();
        check(repo.retire(handle) == Repo::RetireResult::Committed, "deallocation fixture retirement failed");
    }
    auto result = Repo::ReclaimResult::Retry;
    std::thread reclaimer([&, payload = std::move(payload)]() mutable {
        if (installed) result = repo.try_reclaim(handle);
        else payload = {}; // unused request reservation rollback
    });
    const bool reached = await(entered, 5);
    bool refused = false;
    Store::PreparedPayload replacement;
    try { replacement = store->prepare_payload(4); }
    catch (const std::bad_alloc &) { refused = true; }
    const auto charged = store->allocated_pinned_bytes();
    proceed.store(true, std::memory_order_release);
    reclaimer.join();
    check(reached && !timed_out && refused && charged == 40,
          "pinned slot or byte credit returned before physical deallocation completed");
    check(!store->allocated_pinned_bytes() && (!installed || result == Repo::ReclaimResult::Complete),
          "completed pinned deallocation did not return capacity");
    replacement = store->prepare_payload(4);
    check(replacement.pinned_bytes() == 40, "capacity was not reusable after physical free");
}

static void retired_producers_and_reuse() {
    HistoricalKvRawStoreConfig config{2, 4, 2, 3, 1, {}};
    config.pinned_bytes = 40;
    config.allocate_pinned = [](size_t bytes) -> std::shared_ptr<uint8_t> {
        return {new uint8_t[bytes], std::default_delete<uint8_t[]>()};
    };
    auto store = std::make_shared<Store>(config, 1);
    Repo repo(1);
    for (uint64_t round = 1; round <= 32; ++round) {
        auto payload = store->prepare_payload(4);
        auto writer = repo.begin_write(round, round * 3, 3, round, payload.payload_bytes(), binding(repo, store, payload));
        check(bool(writer), "retired pinned producer admission failed");
        const auto handle = writer->handle();
        check(repo.retire(handle) == Repo::RetireResult::Committed, "Writing pin retirement failed");
        repo.try_reclaim(handle);
        check(!repo.inspect(handle)->pinned_released && store->allocated_pinned_bytes() == 40,
              "not-yet-installed pinned producer was acknowledged absent");
        check(store->install_payload(*writer, handle, round * 3, 3, payload), "retired pin producer could not install");
        repo.try_reclaim(handle); // closes new IO while the original writer is pending
        check(!repo.inspect(handle)->pinned_released && store->allocated_pinned_bytes() == 40,
              "GC discarded a promised pinned producer copy");
        if (round % 2) {
            for (uint32_t layer = 0; layer < 2; ++layer) {
                check(store->write_packed_k_slice(*writer, handle, layer, 0, 3, std::vector<uint8_t>(6, 17)), "retired K failed");
                check(store->write_packed_v_slice(*writer, handle, layer, 0, 3, std::vector<uint8_t>(9, 18)), "retired V failed");
            }
            check(!store->publish_complete(repo, *writer, handle), "missing Mean-K or retired data published");
            repo.try_reclaim(handle);
            check(!store->allocated_pinned_bytes() && repo.inspect(handle)->pinned_released,
                  "completed packed copy retained pinned storage through unfinished Mean-K");
            for (uint32_t layer = 0; layer < 2; ++layer)
                check(store->write_mean_k_slice(*writer, handle, layer, 0, 3, {21.f}), "late Mean-K failed");
            check(!store->publish_complete(repo, *writer, handle), "retired producer resurrected historical view");
        } else {
            check(store->write_packed_k_slice(*writer, handle, 0, 0, 1, {7, 8}), "cancel fixture write failed");
        }
        writer.reset(); // includes half-written cancellation
        check(repo.try_reclaim(handle) == Repo::ReclaimResult::Complete &&
              !store->allocated_pinned_bytes() && !store->allocated_payload_bytes() &&
              repo.observe_range(0, 1).ready == 1,
              "old pinned producer leaked bytes or Ready credit across reuse");
    }
}

static void request_reservation_rollback() {
    for (bool fail_allocator : {false, true}) {
        unsigned allocations = 0, frees = 0;
        HistoricalKvRawStoreConfig config{2, 4, 2, 3, 0, {}};
        config.pinned_bytes = fail_allocator ? 80 : 40;
        config.allocate_pinned = [&](size_t bytes) -> std::shared_ptr<uint8_t> {
            if (++allocations == 2 && fail_allocator) return nullptr;
            return {new uint8_t[bytes], [&](uint8_t * p) { delete[] p; ++frees; }};
        };
        auto store = std::make_shared<Store>(config, 2);
        auto repo = std::make_shared<Repo>(2);
        HistoricalKvRamDemand demand;
        demand.uncached_input_tokens = 8;
        demand.max_accepted_output_tokens = 0;
        demand.max_append_boundaries = 1;
        check(!HistoricalKvRamReservation::try_reserve(repo, store, demand),
              "pinned capacity/allocator failure admitted a partial request");
        check(frees == 1 && !store->allocated_payload_bytes() && !store->allocated_pinned_bytes() &&
              repo->observe_range(0, 2).ready == 2,
              "pinned admission failure leaked physical capacity or Ready credit");
        demand.uncached_input_tokens = 4;
        auto retry = HistoricalKvRamReservation::try_reserve(repo, store, demand);
        check(retry && retry->reserved_pinned_bytes() == 40, "pinned admission rollback prevented a smaller request");
        retry.reset();
        check(frees == 2 && !store->allocated_pinned_bytes() && repo->observe_range(0, 2).ready == 2,
              "unused pinned reservation did not return its capacity");
    }
}

int main() {
    try {
        std::atomic<uint32_t> frees{0};
        HistoricalKvRawStoreConfig config{2, 4, 2, 3, 0, {}};
        config.pinned_bytes = 40; // Exactly one full two-layer extent.
        config.allocate_pinned = [&frees](size_t bytes) -> std::shared_ptr<uint8_t> {
            check(bytes == 40, "pinned reservation was not a full physical extent");
            return {new uint8_t[bytes], [&frees](uint8_t * p) {
                delete[] p;
                frees.fetch_add(1, std::memory_order_relaxed);
            }};
        };
        auto unsupported = config;
        unsupported.nvme_bytes = 40;
        bool refused = false;
        try { Store both(unsupported, 2); } catch (const std::invalid_argument &) { refused = true; }
        check(refused, "simultaneous pinned and NVMe backends were silently admitted");
        auto store = std::make_shared<Store>(config, 2);
        Repo repo(2);
        auto prepared = store->prepare_payload(4);
        check(prepared.pinned_bytes() == 40 && store->allocated_pinned_bytes() == 40,
              "private reservation did not charge pinned memory");
        bool full = false;
        try { auto second = store->prepare_payload(4); } catch (const std::bad_alloc &) { full = true; }
        check(full && store->allocated_pinned_bytes() == 40 && frees == 0,
              "pinned capacity was overcommitted or the first extent was lost");

        auto writer = repo.begin_write(7, 713, 3, 1, prepared.payload_bytes(), binding(repo, store, prepared));
        check(bool(writer), "repository admission failed");
        const auto old = writer->handle();
        check(store->install_payload(*writer, old, 713, 3, prepared), "pinned installation failed");
        // A short tail still owns a full extent but reads only its valid rows.
        for (uint32_t layer = 0; layer < 2; ++layer) {
            check(store->write_layer(*writer, old, layer, std::vector<uint8_t>(6, 19 + layer),
                std::vector<uint8_t>(9, 29 + layer), {}, 0), "short pinned write failed");
        }
        check(store->publish_complete(repo, *writer, old), "short pinned publish failed");
        writer.reset();
        auto reader = store->acquire_pinned_read(repo, old);
        check(bool(reader), "live pinned read was denied");
        const auto old_allocation = reader->allocation_handle();
        const auto direct = store->pinned_layer(*reader, 1);
        check(direct && direct->k_bytes == 6 && direct->v_bytes == 9 &&
              direct->packed_k[0] == 20 && direct->packed_v[0] == 30,
              "pinned transfer did not expose the admitted page-locked address");
        std::vector<uint8_t> k(6), v(9);
        check(store->read_pinned_layer(*reader, 1, k.data(), k.size(), v.data(), v.size()) &&
              k == std::vector<uint8_t>(6, 20) && v == std::vector<uint8_t>(9, 30),
              "pinned copy changed short-tail bytes");
        check(repo.retire(old) == Repo::RetireResult::Committed &&
              !store->acquire_pinned_read(repo, old), "retirement admitted a new pinned reader");
        check(repo.try_reclaim(old) == Repo::ReclaimResult::HostReleased,
              "old pinned IO prevented independent RAM reclamation");
        auto meta = repo.inspect(old);
        check(meta && meta->host_released && !meta->pinned_released && meta->pinned_bytes == 40 &&
              !store->allocated_payload_bytes() && store->allocated_pinned_bytes() == 40 && frees == 0,
              "old pinned IO was freed or credited too early");
        check(store->read_pinned_layer(*reader, 0, k.data(), k.size(), v.data(), v.size()) &&
              k == std::vector<uint8_t>(6, 19) && v == std::vector<uint8_t>(9, 29),
              "admitted old pinned read failed after retirement");
        full = false;
        try { auto second = store->prepare_payload(4); } catch (const std::bad_alloc &) { full = true; }
        check(full, "old pinned reader allowed premature slot reuse");
        reader.reset();
        check(repo.try_reclaim(old) == Repo::ReclaimResult::Complete &&
              !store->allocated_pinned_bytes() && frees == 1,
              "pinned memory did not return before Ready");

        auto next = store->prepare_payload(4);
        auto next_writer = repo.begin_write(8, 721, 4, 2, next.payload_bytes(), binding(repo, store, next));
        check(next_writer && store->install_payload(*next_writer, next_writer->handle(), 721, 4, next),
              "released pinned capacity could not be reused");
        const auto fresh = next_writer->handle();
        fill(*store, repo, *next_writer, 40);
        next_writer.reset();
        auto next_reader = store->acquire_pinned_read(repo, fresh);
        // The next allocation is a different generation even when the same
        // physical pinned slot is selected.
        check(next_reader && next_reader->allocation_handle().slot == old_allocation.slot &&
              next_reader->allocation_handle().generation != old_allocation.generation,
              "pinned slot reuse did not advance its allocation generation");
        next_reader.reset();
        check(repo.retire(fresh) == Repo::RetireResult::Committed &&
              repo.try_reclaim(fresh) == Repo::ReclaimResult::Complete && frees == 2,
              "reused pinned extent did not reclaim");
        captured_stage_in();
        deallocation_precedes_reuse(false);
        deallocation_precedes_reuse(true);
        retired_producers_and_reuse();
        request_reservation_rollback();
        std::cout << "historical pinned store tests passed\n";
    } catch (const std::exception & e) {
        std::cerr << "historical pinned store test failed: " << e.what() << '\n';
        return 1;
    }
}
