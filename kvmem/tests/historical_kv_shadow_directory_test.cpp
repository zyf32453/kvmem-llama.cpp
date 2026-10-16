#include "kvmem/kvmem_runtime.hpp"
#include "kvmem/historical_kv_raw_store.hpp"
#include "kvmem/historical_kv_gc.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <memory>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

using namespace kvmem;

static int failures = 0;
#define CHECK(condition) do {                                                \
    if (!(condition)) {                                                       \
        std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition);      \
        ++failures;                                                            \
    }                                                                         \
} while (0)

static KvMemRuntimeConfig config() {
    KvMemRuntimeConfig cfg;
    cfg.store.block_tokens = 4;
    cfg.store.select_budget = 16;
    cfg.store.sink_blocks = 1;
    cfg.store.recent_blocks = 1;
    return cfg;
}

template <typename Fn>
static bool throws_logic(Fn && fn) {
    try { fn(); }
    catch (const std::logic_error &) { return true; }
    return false;
}

template <typename Fn>
static bool throws_runtime(Fn && fn) {
    try { fn(); }
    catch (const std::runtime_error &) { return true; }
    return false;
}

static bool contains(const std::vector<uint32_t> & ids, uint32_t target) {
    return std::find(ids.begin(), ids.end(), target) != ids.end();
}

static void discard_unbound_writer(
        HistoricalKvRepository & repository,
        std::optional<HistoricalKvRepository::Lease> & writer) {
    if (!writer || !static_cast<bool>(*writer)) return;
    const auto handle = writer->handle();
    CHECK(repository.retire(handle) ==
          HistoricalKvRepository::RetireResult::Committed);
    writer->reset();
    CHECK(repository.try_reclaim(handle) ==
          HistoricalKvRepository::ReclaimResult::Complete);
}

static void test_full_block_binding_and_runtime_owner_move() {
    std::map<uint32_t, HistoricalKvShadowDirectory::Producer> producers;
    auto repo = std::make_shared<HistoricalKvRepository>(2, 81);
    auto runtime = std::make_unique<KvMemRuntime>(config());
    runtime->register_append(4);
    auto writer = repo->begin_write(0, 0, 4, 1, 8,
        [](const HistoricalKvHandle &) { return true; });
    CHECK(writer.has_value());
    if (!writer) return;
    const auto handle = writer->handle();
    CHECK(runtime->bind_historical_full_block(0, repo, handle, *writer, producers[0]));
    CHECK(!static_cast<bool>(*writer));

    auto binding = runtime->historical_shadow_directory().find(0);
    CHECK(binding.has_value());
    if (!binding) return;
    CHECK(binding->block_id == 0 && binding->original_begin == 0 &&
          binding->valid_tokens == 4);
    CHECK(binding->handle.incarnation == handle.incarnation &&
          binding->handle.slot == handle.slot &&
          binding->handle.generation == handle.generation);
    CHECK(binding->writer_context->writer().is_writer() &&
          binding->writer_context->repository().owns_writer(
              binding->writer_context->writer()));

    // The adapter moves a unique_ptr<KvMemRuntime> between conversation
    // owners. A captured async context must keep naming the original writer.
    auto moved = std::move(runtime);
    CHECK(!runtime && moved->historical_shadow_directory().size() == 1);
    auto after_move = moved->historical_shadow_directory().find(0);
    CHECK(after_move && after_move->writer_context == binding->writer_context);
    moved->register_append(4); // A new block does not extend the bound one.
    CHECK(moved->store().block_count() == 2);

    CHECK(!moved->unbind_retired_historical_block(0));
    producers[0].reset();
    CHECK(throws_logic([&] { moved->truncate_to(0); }));
    CHECK(moved->store().total_tokens() == 8);

    CHECK(repo->retire(handle) == HistoricalKvRepository::RetireResult::Committed);
    CHECK(moved->unbind_retired_historical_block(0));
    producers[0].reset();
    CHECK(!moved->historical_shadow_directory().find(0));
    const auto retired = moved->historical_shadow_directory().observe(0);
    CHECK(retired.state == HistoricalKvShadowDirectory::State::Retired &&
          retired.repository_id == repo->repository_id() &&
          retired.handle.generation == handle.generation &&
          retired.original_begin == 0 && retired.valid_tokens == 4);
    CHECK(!moved->confirm_reclaimed_historical_block(0));
    CHECK(throws_logic([&] { moved->truncate_to(0); }));
    CHECK(moved->historical_shadow_directory().size() == 1);
    // External harvest may still hold the old writer after the directory
    // unbinds. No release is attempted until all copies go away.
    CHECK(repo->try_reclaim(handle) == HistoricalKvRepository::ReclaimResult::Deferred);
    after_move.reset();
    binding.reset();
    CHECK(repo->try_reclaim(handle) == HistoricalKvRepository::ReclaimResult::Complete);
    CHECK(moved->confirm_reclaimed_historical_block(0));
    CHECK(moved->historical_shadow_directory().observe(0).state ==
          HistoricalKvShadowDirectory::State::Reclaimed);
    CHECK(!moved->historical_shadow_directory().find(0) &&
          moved->store().block_count() == 2);
    moved->truncate_to(0);
    CHECK(moved->historical_shadow_directory().size() == 0 &&
          moved->store().total_tokens() == 0 &&
          moved->historical_shadow_directory().observe(0).state ==
              HistoricalKvShadowDirectory::State::Unmanaged);
}

static void test_truncate_then_reuse_dense_id() {
    std::map<uint32_t, HistoricalKvShadowDirectory::Producer> producers;
    auto repo = std::make_shared<HistoricalKvRepository>(2, 82);
    KvMemRuntime runtime(config());
    runtime.register_append(8);
    auto first = repo->begin_write(0, 0, 4, 1, 8,
        [](const HistoricalKvHandle &) { return true; });
    auto second = repo->begin_write(1, 4, 4, 2, 8,
        [](const HistoricalKvHandle &) { return true; });
    CHECK(first && second);
    if (!first || !second) return;
    CHECK(runtime.bind_historical_full_block(0, repo, first->handle(), *first, producers[0]));
    CHECK(runtime.bind_historical_full_block(1, repo, second->handle(), *second, producers[1]));
    const auto old = runtime.historical_shadow_directory().find(1)->handle;

    CHECK(throws_logic([&] { runtime.truncate_to(4); }));
    CHECK(runtime.store().block_count() == 2 && runtime.store().total_tokens() == 8);
    CHECK(!runtime.unbind_retired_historical_block(1));
    producers[1].reset();
    CHECK(repo->retire(old) == HistoricalKvRepository::RetireResult::Committed);
    CHECK(runtime.unbind_retired_historical_block(1));
    producers[1].reset();
    CHECK(!runtime.historical_shadow_directory().find(1));
    CHECK(runtime.historical_shadow_directory().observe(1).state ==
          HistoricalKvShadowDirectory::State::Retired);
    CHECK(!runtime.confirm_reclaimed_historical_block(1));
    CHECK(throws_logic([&] { runtime.truncate_to(4); }));
    CHECK(repo->try_reclaim(old) == HistoricalKvRepository::ReclaimResult::Complete);
    CHECK(runtime.confirm_reclaimed_historical_block(1));
    CHECK(runtime.historical_shadow_directory().observe(1).state ==
          HistoricalKvShadowDirectory::State::Reclaimed);
    CHECK(throws_logic([&] { runtime.truncate_to(6); }));
    const auto dropped = runtime.truncate_to(4);
    CHECK(dropped.size() == 1 && dropped[0].block_id == 1 &&
          runtime.historical_shadow_directory().observe(1).state ==
              HistoricalKvShadowDirectory::State::Unmanaged);
    runtime.register_append(4);
    CHECK(runtime.store().blocks()[1].block_id == 1);

    auto replacement = repo->begin_write(1, 4, 4, 3, 8,
        [](const HistoricalKvHandle &) { return true; });
    CHECK(replacement.has_value());
    if (!replacement) return;
    CHECK(runtime.bind_historical_full_block(1, repo,
                                            replacement->handle(), *replacement, producers[1]));
    auto current = runtime.historical_shadow_directory().find(1);
    CHECK(current && current->handle.slot == old.slot &&
          current->handle.generation > old.generation);
    CHECK(throws_logic([&] { runtime.truncate_to(0); }));
    CHECK(runtime.store().total_tokens() == 8);

    const auto first_handle = runtime.historical_shadow_directory().find(0)->handle;
    CHECK(repo->retire(first_handle) ==
          HistoricalKvRepository::RetireResult::Committed);
    CHECK(repo->retire(current->handle) ==
          HistoricalKvRepository::RetireResult::Committed);
    CHECK(runtime.unbind_retired_historical_block(0));
    producers[0].reset();
    CHECK(runtime.unbind_retired_historical_block(1));
    producers[1].reset();
    const auto replacement_handle = current->handle;
    current.reset();
    CHECK(repo->try_reclaim(first_handle) ==
          HistoricalKvRepository::ReclaimResult::Complete);
    CHECK(repo->try_reclaim(replacement_handle) ==
          HistoricalKvRepository::ReclaimResult::Complete);
    CHECK(runtime.confirm_reclaimed_historical_block(0));
    CHECK(runtime.confirm_reclaimed_historical_block(1));
    runtime.truncate_to(0);
}

static void test_reject_partial_foreign_and_mismatched_writer() {
    std::map<uint32_t, HistoricalKvShadowDirectory::Producer> producers;
    auto repo = std::make_shared<HistoricalKvRepository>(5, 83);
    auto foreign = std::make_shared<HistoricalKvRepository>(3, 83);
    KvMemRuntime runtime(config());
    runtime.register_append(3);
    auto partial = repo->begin_write(0, 0, 3, 1, 6,
        [](const HistoricalKvHandle &) { return true; });
    CHECK(partial && !runtime.bind_historical_full_block(
        0, repo, partial->handle(), *partial, producers[0]));
    CHECK(partial && static_cast<bool>(*partial));
    runtime.register_append(1);

    auto wrong_span = repo->begin_write(0, 7, 4, 2, 8,
        [](const HistoricalKvHandle &) { return true; });
    CHECK(wrong_span && !runtime.bind_historical_full_block(
        0, repo, wrong_span->handle(), *wrong_span, producers[0]));
    CHECK(wrong_span && static_cast<bool>(*wrong_span));
    auto wrong_id = repo->begin_write(99, 0, 4, 3, 8,
        [](const HistoricalKvHandle &) { return true; });
    CHECK(wrong_id && !runtime.bind_historical_full_block(
        0, repo, wrong_id->handle(), *wrong_id, producers[0]));
    CHECK(wrong_id && static_cast<bool>(*wrong_id));

    auto foreign_writer = foreign->begin_write(0, 0, 4, 1, 8,
        [](const HistoricalKvHandle &) { return true; });
    CHECK(foreign_writer &&
          !runtime.bind_historical_full_block(
              0, repo, foreign_writer->handle(), *foreign_writer, producers[0]));
    CHECK(foreign_writer && static_cast<bool>(*foreign_writer));

    // A fresh runtime accepts one repository only, even when the other has
    // the same caller-controlled incarnation and an equal-looking handle.
    CHECK(foreign_writer &&
          runtime.bind_historical_full_block(
              0, foreign, foreign_writer->handle(), *foreign_writer, producers[0]));
    runtime.register_append(4);
    auto valid = repo->begin_write(1, 4, 4, 4, 8,
        [](const HistoricalKvHandle &) { return true; });
    CHECK(valid && !runtime.bind_historical_full_block(
        1, repo, valid->handle(), *valid, producers[1]));
    CHECK(valid && static_cast<bool>(*valid));

    discard_unbound_writer(*repo, partial);
    discard_unbound_writer(*repo, wrong_span);
    discard_unbound_writer(*repo, wrong_id);
    discard_unbound_writer(*repo, valid);
    auto bound = runtime.historical_shadow_directory().find(0);
    CHECK(bound.has_value());
    if (!bound) return;
    CHECK(foreign->retire(bound->handle) ==
          HistoricalKvRepository::RetireResult::Committed);
    CHECK(runtime.unbind_retired_historical_block(0));
    producers[0].reset();
    CHECK(!runtime.historical_shadow_directory().find(0) &&
          runtime.historical_shadow_directory().observe(0).state ==
              HistoricalKvShadowDirectory::State::Retired);
    CHECK(!runtime.confirm_reclaimed_historical_block(0));
    CHECK(foreign->try_reclaim(bound->handle) ==
          HistoricalKvRepository::ReclaimResult::Deferred);
    const auto bound_handle = bound->handle;
    bound.reset();
    CHECK(foreign->try_reclaim(bound_handle) ==
          HistoricalKvRepository::ReclaimResult::Complete);
    CHECK(runtime.confirm_reclaimed_historical_block(0));
    runtime.truncate_to(0);
}

static void test_closed_session_keeps_writer_with_caller() {
    std::map<uint32_t, HistoricalKvShadowDirectory::Producer> producers;
    auto repo = std::make_shared<HistoricalKvRepository>(1, 84);
    KvMemRuntime runtime(config());
    runtime.register_append(4);
    auto writer = repo->begin_write(0, 0, 4, 1, 8,
        [](const HistoricalKvHandle &) { return true; });
    CHECK(writer.has_value());
    if (!writer) return;
    repo->close();
    CHECK(!runtime.bind_historical_full_block(0, repo, writer->handle(), *writer, producers[0]));
    CHECK(static_cast<bool>(*writer));
    CHECK(runtime.historical_shadow_directory().size() == 0);
    discard_unbound_writer(*repo, writer);
}

static void test_wrong_expected_generation_keeps_writer() {
    std::map<uint32_t, HistoricalKvShadowDirectory::Producer> producers;
    auto repo = std::make_shared<HistoricalKvRepository>(1, 85);
    KvMemRuntime runtime(config());
    runtime.register_append(4);
    auto writer = repo->begin_write(0, 0, 4, 1, 8,
        [](const HistoricalKvHandle &) { return true; });
    CHECK(writer.has_value());
    if (!writer) return;
    const auto exact = writer->handle();
    auto stale = exact;
    ++stale.generation;
    CHECK(!runtime.bind_historical_full_block(0, repo, stale, *writer, producers[0]));
    CHECK(static_cast<bool>(*writer) &&
          runtime.historical_shadow_directory().size() == 0);
    CHECK(runtime.bind_historical_full_block(0, repo, exact, *writer, producers[0]));
    CHECK(repo->retire(exact) == HistoricalKvRepository::RetireResult::Committed);
    CHECK(runtime.unbind_retired_historical_block(0));
    producers[0].reset();
    CHECK(repo->try_reclaim(exact) == HistoricalKvRepository::ReclaimResult::Complete);
    CHECK(runtime.confirm_reclaimed_historical_block(0));
}

static void test_tombstone_survives_deferred_and_releasing_retry() {
    std::map<uint32_t, HistoricalKvShadowDirectory::Producer> producers;
    auto repo = std::make_shared<HistoricalKvRepository>(1, 86);
    KvMemRuntime runtime(config());
    runtime.register_append(4);
    unsigned attempts = 0;
    auto writer = repo->begin_write(0, 0, 4, 1, 8,
        [&attempts](const HistoricalKvHandle &) { return ++attempts > 1; });
    CHECK(writer.has_value());
    if (!writer) return;
    const auto handle = writer->handle();
    CHECK(runtime.bind_historical_full_block(0, repo, handle, *writer, producers[0]));
    auto in_flight = runtime.historical_shadow_directory().find(0);
    CHECK(in_flight.has_value());
    CHECK(repo->retire(handle) == HistoricalKvRepository::RetireResult::Committed);
    CHECK(runtime.unbind_retired_historical_block(0));
    producers[0].reset();
    CHECK(!runtime.historical_shadow_directory().find(0));
    CHECK(runtime.historical_shadow_directory().observe(0).state ==
          HistoricalKvShadowDirectory::State::Retired);

    HistoricalKvRepository::ReclaimResult result =
        HistoricalKvRepository::ReclaimResult::Stale;
    std::thread deferred_worker([&] { result = repo->try_reclaim(handle); });
    deferred_worker.join();
    CHECK(result == HistoricalKvRepository::ReclaimResult::Deferred);
    CHECK(!runtime.confirm_reclaimed_historical_block(0));
    in_flight.reset();

    std::thread retry_worker([&] { result = repo->try_reclaim(handle); });
    retry_worker.join();
    CHECK(result == HistoricalKvRepository::ReclaimResult::Retry);
    CHECK(!repo->inspect(handle)); // Releasing is intentionally unreadable.
    CHECK(!repo->reclaim_completed(handle, repo->repository_id()));
    CHECK(!runtime.confirm_reclaimed_historical_block(0));
    CHECK(throws_logic([&] { runtime.truncate_to(0); }));

    std::thread complete_worker([&] { result = repo->try_reclaim(handle); });
    complete_worker.join();
    CHECK(result == HistoricalKvRepository::ReclaimResult::Complete && attempts == 2);
    CHECK(repo->reclaim_completed(handle, repo->repository_id()));
    CHECK(runtime.confirm_reclaimed_historical_block(0));
    CHECK(!runtime.historical_shadow_directory().find(0) &&
          runtime.historical_shadow_directory().observe(0).state ==
              HistoricalKvShadowDirectory::State::Reclaimed);
    CHECK(throws_logic([&] { runtime.truncate_to(2); }));
    runtime.truncate_to(0);
    CHECK(runtime.historical_shadow_directory().observe(0).state ==
          HistoricalKvShadowDirectory::State::Unmanaged);
}

static void test_retirement_closes_new_host_selection_before_unbind() {
    std::map<uint32_t, HistoricalKvShadowDirectory::Producer> producers;
    auto repo = std::make_shared<HistoricalKvRepository>(1, 87);
    KvMemRuntime runtime(config());
    runtime.register_append(12);
    auto writer = repo->begin_write(1, 4, 4, 1, 8,
        [](const HistoricalKvHandle &) { return true; });
    CHECK(writer.has_value());
    if (!writer) return;
    const auto handle = writer->handle();
    CHECK(runtime.bind_historical_full_block(1, repo, handle, *writer, producers[1]));
    auto bound = runtime.historical_shadow_directory().find(1);
    CHECK(bound.has_value());
    if (!bound) return;
    CHECK(repo->publish_write(bound->writer_context->writer()));
    CHECK(runtime.store().is_live(1) &&
          contains(runtime.store().pick_topk_blocks(), 1));

    CHECK(repo->retire(handle) == HistoricalKvRepository::RetireResult::Committed);
    // Producer work still owns its writer here; selection must observe
    // retirement before the foreground updates its directory observation.
    CHECK(!runtime.store().is_live(1));
    CHECK(!contains(runtime.store().pick_topk_blocks(), 1));
    CHECK(!contains(runtime.store().pick_prefill_pressure_blocks(), 1));
    CHECK(runtime.store().gc_missing_required_prefix_block(0, 12, 12) == 1);
    CHECK(throws_runtime([&] { runtime.store().set_selection({0, 1}); }));
    CHECK(throws_runtime([&] { runtime.store().commit_resident_selection({0, 1}); }));
    CHECK(throws_runtime([&] { runtime.store().pick_topk_blocks({1}); }));

    CHECK(runtime.unbind_retired_historical_block(1));
    producers[1].reset();
    CHECK(!runtime.store().is_live(1));
    CHECK(repo->try_reclaim(handle) == HistoricalKvRepository::ReclaimResult::Deferred);
    bound.reset();
    CHECK(repo->try_reclaim(handle) == HistoricalKvRepository::ReclaimResult::Complete);
    CHECK(runtime.confirm_reclaimed_historical_block(1));
    CHECK(!runtime.store().is_live(1) &&
          !contains(runtime.store().pick_topk_blocks(), 1));
    runtime.truncate_to(4);
    CHECK(runtime.historical_shadow_directory().observe(1).state ==
          HistoricalKvShadowDirectory::State::Unmanaged);
}

static void test_direct_store_mutations_cannot_bypass_identity_guards() {
    std::map<uint32_t, HistoricalKvShadowDirectory::Producer> producers;
    auto repo = std::make_shared<HistoricalKvRepository>(1, 88);
    KvMemRuntime runtime(config());
    runtime.register_append(12);
    auto writer = repo->begin_write(1, 4, 4, 1, 8,
        [](const HistoricalKvHandle &) { return true; });
    CHECK(writer.has_value());
    if (!writer) return;
    const auto handle = writer->handle();
    CHECK(runtime.bind_historical_full_block(1, repo, handle, *writer, producers[1]));

    auto & store = runtime.store();
    store.register_append(4);
    CHECK(store.total_tokens() == 16 && store.block_count() == 4);
    CHECK(throws_logic([&] { store.truncate_to(6); }));
    CHECK(throws_logic([&] { store.truncate_to(4); }));
    CHECK(store.total_tokens() == 16 && store.block_count() == 4 &&
          runtime.historical_shadow_directory().observe(1).state ==
              HistoricalKvShadowDirectory::State::Bound);

    CHECK(repo->retire(handle) == HistoricalKvRepository::RetireResult::Committed);
    CHECK(runtime.unbind_retired_historical_block(1));
    producers[1].reset();
    CHECK(throws_logic([&] { store.truncate_to(0); }));
    CHECK(store.total_tokens() == 16 && store.block_count() == 4);
    CHECK(repo->try_reclaim(handle) == HistoricalKvRepository::ReclaimResult::Complete);
    CHECK(runtime.confirm_reclaimed_historical_block(1));
    CHECK(throws_logic([&] { store.truncate_to(6); }));
    CHECK(store.total_tokens() == 16 && store.block_count() == 4);

    const auto dropped = store.truncate_to(4);
    CHECK(dropped.size() == 3 && dropped[0].block_id == 3 &&
          dropped[2].block_id == 1 && store.total_tokens() == 4 &&
          runtime.historical_shadow_directory().size() == 0);
    store.register_append(4);
    auto replacement = repo->begin_write(1, 4, 4, 2, 8,
        [](const HistoricalKvHandle &) { return true; });
    CHECK(replacement.has_value());
    if (!replacement) return;
    CHECK(replacement->handle().generation > handle.generation);
    const auto next = replacement->handle();
    CHECK(runtime.bind_historical_full_block(1, repo, next, *replacement, producers[1]));
    CHECK(repo->retire(next) == HistoricalKvRepository::RetireResult::Committed);
    CHECK(runtime.unbind_retired_historical_block(1));
    producers[1].reset();
    CHECK(repo->try_reclaim(next) == HistoricalKvRepository::ReclaimResult::Complete);
    CHECK(runtime.confirm_reclaimed_historical_block(1));
    runtime.truncate_to(0);
}

static void test_read_admission_requires_live_exact_generation() {
    std::map<uint32_t, HistoricalKvShadowDirectory::Producer> producers;
    using Status = HistoricalKvShadowDirectory::Access::Status;
    auto repo = std::make_shared<HistoricalKvRepository>(1, 89);
    KvMemRuntime runtime(config());
    runtime.register_append(8);
    const auto & directory = runtime.historical_shadow_directory();
    auto legacy = directory.acquire_read(1);
    CHECK(legacy && legacy.status == Status::Unmanaged && !legacy.lease);
    auto legacy_transfer = directory.acquire_transfer(1);
    CHECK(legacy_transfer && legacy_transfer.status == Status::Unmanaged &&
          !legacy_transfer.lease);

    auto writer = repo->begin_write(0, 0, 4, 1, 8,
        [](const HistoricalKvHandle &) { return true; });
    CHECK(writer.has_value());
    if (!writer) return;
    const auto handle = writer->handle();
    CHECK(runtime.bind_historical_full_block(0, repo, handle, *writer, producers[0]));
    CHECK(!directory.acquire_read(0) && !directory.acquire_transfer(0));
    CHECK(repo->publish_write(directory.find(0)->writer_context->writer()));
    auto read = directory.acquire_read(0);
    auto transfer = directory.acquire_transfer(0);
    CHECK(read && read.status == Status::Acquired && read.lease &&
          read.lease.kind() == HistoricalKvRepository::LeaseKind::Read &&
          read.lease.repository_id() == repo->repository_id() &&
          read.lease.handle().generation == handle.generation);
    CHECK(transfer && transfer.lease &&
          transfer.lease.kind() == HistoricalKvRepository::LeaseKind::Transfer);
    auto moved_read = std::move(read);
    CHECK(!read && moved_read);
    repo->close();
    CHECK(!directory.acquire_read(0) && !directory.acquire_transfer(0));
    CHECK(repo->retire(handle) == HistoricalKvRepository::RetireResult::Committed);
    CHECK(!directory.acquire_read(0) && !directory.acquire_transfer(0));
    CHECK(runtime.unbind_retired_historical_block(0));
    producers[0].reset();
    CHECK(repo->try_reclaim(handle) == HistoricalKvRepository::ReclaimResult::Deferred);
    moved_read.lease.reset();
    CHECK(!moved_read);
    CHECK(repo->try_reclaim(handle) == HistoricalKvRepository::ReclaimResult::Deferred);
    transfer.lease.reset();
    CHECK(repo->try_reclaim(handle) == HistoricalKvRepository::ReclaimResult::Complete);
    CHECK(runtime.confirm_reclaimed_historical_block(0));
    CHECK(!directory.acquire_read(0) && !directory.acquire_transfer(0));
    runtime.truncate_to(0);
    runtime.register_append(4);
    CHECK(directory.acquire_read(0).status == Status::Unmanaged);
}

static void test_binding_captures_exact_payload_owner() {
    std::map<uint32_t, HistoricalKvShadowDirectory::Producer> producers;
    auto repo = std::make_shared<HistoricalKvRepository>(1, 90);
    auto foreign_repo = std::make_shared<HistoricalKvRepository>(1, 90);
    HistoricalKvRawStoreConfig cfg;
    cfg.n_layer = 1;
    cfg.max_tokens_per_slot = 4;
    cfg.packed_k_bytes_per_token = cfg.packed_v_bytes_per_token = 1;
    auto storage = std::make_shared<HistoricalKvRawStore>(cfg, 1);
    auto foreign_storage = std::make_shared<HistoricalKvRawStore>(cfg, 1);
    auto writer = repo->begin_write(0, 0, 4, 1, 8,
        [storage, owner = std::weak_ptr<HistoricalKvRepository>(repo)](const HistoricalKvHandle & h) {
            const auto locked = owner.lock();
            return locked && storage->release(*locked, h).code == HistoricalKvRawStore::ReleaseCode::Released;
        });
    auto foreign_writer = foreign_repo->begin_write(0, 0, 4, 1, 8,
        [](const HistoricalKvHandle &) { return true; });
    CHECK(writer && foreign_writer);
    if (!writer || !foreign_writer) return;
    const auto handle = writer->handle();
    CHECK(foreign_storage->reserve(*foreign_writer, foreign_writer->handle(), 0, 4));
    KvMemRuntime runtime(config());
    runtime.register_append(4);
    CHECK(!runtime.bind_historical_full_block(0, repo, handle, *writer, producers[0], storage));
    CHECK(!runtime.bind_historical_full_block(0, repo, handle, *writer, producers[0], foreign_storage));
    CHECK(static_cast<bool>(*writer));
    CHECK(storage->reserve(*writer, handle, 0, 4));
    CHECK(runtime.bind_historical_full_block(0, repo, handle, *writer, producers[0], storage));
    auto captured = runtime.historical_shadow_directory().find(0);
    CHECK(captured && captured->writer_context->storage() == storage);
    CHECK(storage->write_layer(captured->writer_context->writer(), handle, 0,
                               {1, 2, 3, 4}, {5, 6, 7, 8}, {}, 0));
    CHECK(storage->publish_complete(*repo, captured->writer_context->writer(), handle));
    auto read = runtime.historical_shadow_directory().acquire_read(0);
    CHECK(read && read.storage == storage &&
          read.storage->read_layer(read.lease, handle, 0)->packed_k == std::vector<uint8_t>({1, 2, 3, 4}));
    CHECK(repo->retire(handle) == HistoricalKvRepository::RetireResult::Committed);
    CHECK(runtime.unbind_retired_historical_block(0));
    producers[0].reset();
    captured.reset();
    CHECK(repo->try_reclaim(handle) == HistoricalKvRepository::ReclaimResult::Deferred);
    read.lease.reset();
    CHECK(repo->try_reclaim(handle) == HistoricalKvRepository::ReclaimResult::Complete);
    CHECK(storage->allocated_payload_bytes() == 0);
    discard_unbound_writer(*foreign_repo, foreign_writer);
}

class DeleteObserved final : public HistoricalKvPredicatePolicyTemplate {
    std::optional<HistoricalKvDeleteIntent> decide(
            const HistoricalKvBlockMeta & block, const HistoricalKvSnapshot & snapshot,
            const HistoricalKvPressure &) override {
        HistoricalKvDeleteIntent intent;
        intent.victim = block.handle;
        intent.round_id = snapshot.round_id;
        return intent;
    }
};

template <typename Predicate>
static bool eventually(Predicate predicate) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    do {
        if (predicate()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    } while (std::chrono::steady_clock::now() < deadline);
    return false;
}

// There is no foreground unbind, manual GC tick, or next active in this test.
// Both real component producers and reads/transfers finish on other threads.
static void test_task_completion_allows_autonomous_reclaim() {
    using Repo = HistoricalKvRepository;
    using Payload = HistoricalKvRawStore;
    for (bool published : {false, true}) {
        auto repo = std::make_shared<Repo>(1);
        auto storage = std::make_shared<Payload>(
            HistoricalKvRawStoreConfig{1, 4, 1, 1, 1, {}}, 1);
        KvMemRuntime runtime(config());
        runtime.register_append(4);
        const auto & directory = runtime.historical_shadow_directory();
        const auto foreground = std::this_thread::get_id();
        std::atomic<bool> foreground_release{false}, task_ok{true};
        auto writer = repo->begin_write(0, 0, 4, 1, 0,
            [storage, weak = std::weak_ptr<Repo>(repo), &foreground_release, foreground](auto h) {
                if (std::this_thread::get_id() == foreground) foreground_release = true;
                auto owner = weak.lock();
                return owner && storage->release(*owner, h).code == Payload::ReleaseCode::Released;
            });
        CHECK(writer.has_value());
        if (!writer) return;
        const auto handle = writer->handle();
        CHECK(storage->reserve(*writer, handle, 0, 4));
        HistoricalKvShadowDirectory::Producer request;
        CHECK(runtime.bind_historical_block(0, repo, handle, *writer, request, storage));
        auto packed_task = request;
        auto mean_task = request;
        HistoricalKvShadowDirectory::Access read, transfer;
        if (published) {
            CHECK(storage->write_layer(request->writer(), handle, 0,
                {1, 2, 3, 4}, {5, 6, 7, 8}, {10.0f}, 4));
            CHECK(storage->publish_complete(*repo, request->writer(), handle));
            packed_task.reset();
            mean_task.reset();
            request.reset();
            CHECK(directory.find(0) && !directory.find(0)->writer_context);
            // Reading uses the independent storage owner, not a live writer.
            read = directory.acquire_read(0);
            transfer = directory.acquire_transfer(0);
            CHECK(read && transfer && read.storage == storage && transfer.storage == storage);
        }
        request.reset(); // Only the actual tasks retain generation leases now.

        HistoricalKvGc::Config cfg;
        cfg.mode = HistoricalKvGc::Mode::Delete;
        cfg.scan_blocks = cfg.batch_blocks = cfg.reclaim_blocks = 1;
        cfg.poll_ms = 1;
        HistoricalKvGc gc(repo, std::make_unique<DeleteObserved>(), cfg);
        gc.start();
        const bool retired = eventually([&] {
            const auto meta = repo->inspect(handle);
            return meta && meta->life == HistoricalKvLife::Retired;
        });
        CHECK(retired);
        CHECK(!directory.allows_new_selection(0) && !directory.acquire_read(0) &&
              !directory.acquire_transfer(0));
        CHECK(storage->allocated_payload_bytes() != 0 && gc.stats().reclaimed == 0);
        // A directory state update is not part of either task completion.
        std::thread packed([&, task = std::move(packed_task), access = std::move(transfer)]() mutable {
            if (task) {
                if (!storage->write_packed_k_slice(task->writer(), handle, 0, 0, 4, {1, 2, 3, 4}) ||
                    !storage->write_packed_v_slice(task->writer(), handle, 0, 0, 4, {5, 6, 7, 8}))
                    task_ok = false;
            } else if (!storage->read_layer(access.lease, handle, 0)) task_ok = false;
            task.reset();
            access.lease.reset();
        });
        std::thread mean([&, task = std::move(mean_task), access = std::move(read)]() mutable {
            if (task) {
                if (!storage->write_mean_k_slice(task->writer(), handle, 0, 0, 4, {10.0f}) ||
                    storage->publish_complete(*repo, task->writer(), handle)) task_ok = false;
            } else if (!storage->read_layer(access.lease, handle, 0)) task_ok = false;
            task.reset();
            access.lease.reset();
        });
        packed.join();
        mean.join();
        CHECK(eventually([&] { return repo->reclaim_completed(handle, repo->repository_id()); }));
        gc.stop();
        CHECK(task_ok && !foreground_release && storage->allocated_payload_bytes() == 0 &&
              gc.stats().reclaimed == 1);
        CHECK(directory.observe(0).state == HistoricalKvShadowDirectory::State::Bound);
        CHECK(!directory.allows_new_selection(0) && !directory.acquire_read(0));
        CHECK(throws_logic([&] { runtime.truncate_to(0); }));
        // Direct exact-generation acknowledgement, without unbind, is enough
        // for the later metadata cleanup. A reused slot cannot reopen it.
        auto replacement = repo->begin_write(9, 40, 4, 2, 0, [](auto) { return true; });
        CHECK(replacement && replacement->handle().generation > handle.generation);
        CHECK(runtime.confirm_reclaimed_historical_block(0));
        CHECK(!directory.acquire_transfer(0));
        runtime.truncate_to(0);
        CHECK(directory.size() == 0);
        discard_unbound_writer(*repo, replacement);
    }
}

int main() {
    test_full_block_binding_and_runtime_owner_move();
    test_truncate_then_reuse_dense_id();
    test_reject_partial_foreign_and_mismatched_writer();
    test_closed_session_keeps_writer_with_caller();
    test_wrong_expected_generation_keeps_writer();
    test_tombstone_survives_deferred_and_releasing_retry();
    test_retirement_closes_new_host_selection_before_unbind();
    test_direct_store_mutations_cannot_bypass_identity_guards();
    test_read_admission_requires_live_exact_generation();
    test_binding_captures_exact_payload_owner();
    test_task_completion_allows_autonomous_reclaim();
    if (failures != 0) {
        std::printf("FAILED: %d check(s)\n", failures);
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
