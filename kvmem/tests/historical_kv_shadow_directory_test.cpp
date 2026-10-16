#include "kvmem/kvmem_runtime.hpp"

#include <algorithm>
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
    auto repo = std::make_shared<HistoricalKvRepository>(2, 81);
    auto runtime = std::make_unique<KvMemRuntime>(config());
    runtime->register_append(4);
    auto writer = repo->begin_write(0, 0, 4, 1, 8,
        [](const HistoricalKvHandle &) { return true; });
    CHECK(writer.has_value());
    if (!writer) return;
    const auto handle = writer->handle();
    CHECK(runtime->bind_historical_full_block(0, repo, handle, *writer));
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
    CHECK(throws_logic([&] { moved->truncate_to(0); }));
    CHECK(moved->store().total_tokens() == 8);

    CHECK(repo->retire(handle) == HistoricalKvRepository::RetireResult::Committed);
    CHECK(moved->unbind_retired_historical_block(0));
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
    auto repo = std::make_shared<HistoricalKvRepository>(2, 82);
    KvMemRuntime runtime(config());
    runtime.register_append(8);
    auto first = repo->begin_write(0, 0, 4, 1, 8,
        [](const HistoricalKvHandle &) { return true; });
    auto second = repo->begin_write(1, 4, 4, 2, 8,
        [](const HistoricalKvHandle &) { return true; });
    CHECK(first && second);
    if (!first || !second) return;
    CHECK(runtime.bind_historical_full_block(0, repo, first->handle(), *first));
    CHECK(runtime.bind_historical_full_block(1, repo, second->handle(), *second));
    const auto old = runtime.historical_shadow_directory().find(1)->handle;

    CHECK(throws_logic([&] { runtime.truncate_to(4); }));
    CHECK(runtime.store().block_count() == 2 && runtime.store().total_tokens() == 8);
    CHECK(!runtime.unbind_retired_historical_block(1));
    CHECK(repo->retire(old) == HistoricalKvRepository::RetireResult::Committed);
    CHECK(runtime.unbind_retired_historical_block(1));
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
                                            replacement->handle(), *replacement));
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
    CHECK(runtime.unbind_retired_historical_block(1));
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
    auto repo = std::make_shared<HistoricalKvRepository>(5, 83);
    auto foreign = std::make_shared<HistoricalKvRepository>(3, 83);
    KvMemRuntime runtime(config());
    runtime.register_append(3);
    auto partial = repo->begin_write(0, 0, 3, 1, 6,
        [](const HistoricalKvHandle &) { return true; });
    CHECK(partial && !runtime.bind_historical_full_block(
        0, repo, partial->handle(), *partial));
    CHECK(partial && static_cast<bool>(*partial));
    runtime.register_append(1);

    auto wrong_span = repo->begin_write(0, 7, 4, 2, 8,
        [](const HistoricalKvHandle &) { return true; });
    CHECK(wrong_span && !runtime.bind_historical_full_block(
        0, repo, wrong_span->handle(), *wrong_span));
    CHECK(wrong_span && static_cast<bool>(*wrong_span));
    auto wrong_id = repo->begin_write(99, 0, 4, 3, 8,
        [](const HistoricalKvHandle &) { return true; });
    CHECK(wrong_id && !runtime.bind_historical_full_block(
        0, repo, wrong_id->handle(), *wrong_id));
    CHECK(wrong_id && static_cast<bool>(*wrong_id));

    auto foreign_writer = foreign->begin_write(0, 0, 4, 1, 8,
        [](const HistoricalKvHandle &) { return true; });
    CHECK(foreign_writer &&
          !runtime.bind_historical_full_block(
              0, repo, foreign_writer->handle(), *foreign_writer));
    CHECK(foreign_writer && static_cast<bool>(*foreign_writer));

    // A fresh runtime accepts one repository only, even when the other has
    // the same caller-controlled incarnation and an equal-looking handle.
    CHECK(foreign_writer &&
          runtime.bind_historical_full_block(
              0, foreign, foreign_writer->handle(), *foreign_writer));
    runtime.register_append(4);
    auto valid = repo->begin_write(1, 4, 4, 4, 8,
        [](const HistoricalKvHandle &) { return true; });
    CHECK(valid && !runtime.bind_historical_full_block(
        1, repo, valid->handle(), *valid));
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
    auto repo = std::make_shared<HistoricalKvRepository>(1, 84);
    KvMemRuntime runtime(config());
    runtime.register_append(4);
    auto writer = repo->begin_write(0, 0, 4, 1, 8,
        [](const HistoricalKvHandle &) { return true; });
    CHECK(writer.has_value());
    if (!writer) return;
    repo->close();
    CHECK(!runtime.bind_historical_full_block(0, repo, writer->handle(), *writer));
    CHECK(static_cast<bool>(*writer));
    CHECK(runtime.historical_shadow_directory().size() == 0);
    discard_unbound_writer(*repo, writer);
}

static void test_wrong_expected_generation_keeps_writer() {
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
    CHECK(!runtime.bind_historical_full_block(0, repo, stale, *writer));
    CHECK(static_cast<bool>(*writer) &&
          runtime.historical_shadow_directory().size() == 0);
    CHECK(runtime.bind_historical_full_block(0, repo, exact, *writer));
    CHECK(repo->retire(exact) == HistoricalKvRepository::RetireResult::Committed);
    CHECK(runtime.unbind_retired_historical_block(0));
    CHECK(repo->try_reclaim(exact) == HistoricalKvRepository::ReclaimResult::Complete);
    CHECK(runtime.confirm_reclaimed_historical_block(0));
}

static void test_tombstone_survives_deferred_and_releasing_retry() {
    auto repo = std::make_shared<HistoricalKvRepository>(1, 86);
    KvMemRuntime runtime(config());
    runtime.register_append(4);
    unsigned attempts = 0;
    auto writer = repo->begin_write(0, 0, 4, 1, 8,
        [&attempts](const HistoricalKvHandle &) { return ++attempts > 1; });
    CHECK(writer.has_value());
    if (!writer) return;
    const auto handle = writer->handle();
    CHECK(runtime.bind_historical_full_block(0, repo, handle, *writer));
    auto in_flight = runtime.historical_shadow_directory().find(0);
    CHECK(in_flight.has_value());
    CHECK(repo->retire(handle) == HistoricalKvRepository::RetireResult::Committed);
    CHECK(runtime.unbind_retired_historical_block(0));
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
    auto repo = std::make_shared<HistoricalKvRepository>(1, 87);
    KvMemRuntime runtime(config());
    runtime.register_append(12);
    auto writer = repo->begin_write(1, 4, 4, 1, 8,
        [](const HistoricalKvHandle &) { return true; });
    CHECK(writer.has_value());
    if (!writer) return;
    const auto handle = writer->handle();
    CHECK(runtime.bind_historical_full_block(1, repo, handle, *writer));
    auto bound = runtime.historical_shadow_directory().find(1);
    CHECK(bound.has_value());
    if (!bound) return;
    CHECK(repo->publish_write(bound->writer_context->writer()));
    CHECK(runtime.store().is_live(1) &&
          contains(runtime.store().pick_topk_blocks(), 1));

    CHECK(repo->retire(handle) == HistoricalKvRepository::RetireResult::Committed);
    // The directory still owns its writer here; the selector must observe
    // repository retirement before the foreground has unbound the block.
    CHECK(!runtime.store().is_live(1));
    CHECK(!contains(runtime.store().pick_topk_blocks(), 1));
    CHECK(!contains(runtime.store().pick_prefill_pressure_blocks(), 1));
    CHECK(runtime.store().gc_missing_required_prefix_block(0, 12, 12) == 1);
    CHECK(throws_runtime([&] { runtime.store().set_selection({0, 1}); }));
    CHECK(throws_runtime([&] { runtime.store().commit_resident_selection({0, 1}); }));
    CHECK(throws_runtime([&] { runtime.store().pick_topk_blocks({1}); }));

    CHECK(runtime.unbind_retired_historical_block(1));
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

int main() {
    test_full_block_binding_and_runtime_owner_move();
    test_truncate_then_reuse_dense_id();
    test_reject_partial_foreign_and_mismatched_writer();
    test_closed_session_keeps_writer_with_caller();
    test_wrong_expected_generation_keeps_writer();
    test_tombstone_survives_deferred_and_releasing_retry();
    test_retirement_closes_new_host_selection_before_unbind();
    if (failures != 0) {
        std::printf("FAILED: %d check(s)\n", failures);
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
