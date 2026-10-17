#include "kvmem/historical_kv_active_set.hpp"

#include <atomic>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace kvmem;

static void check(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}

static HistoricalKvHandle live_block(
    const std::shared_ptr<HistoricalKvRepository> & repository,
    uint64_t logical_id, std::atomic<unsigned> & released) {
    auto writer = repository->begin_write(logical_id, logical_id * 8, 8, 1, 64,
        [&released](const HistoricalKvHandle &) {
            released.fetch_add(1, std::memory_order_relaxed);
            return true;
        });
    check(writer.has_value(), "block writer was not admitted");
    const auto handle = writer->handle();
    check(repository->publish_write(*writer), "block was not published");
    writer->reset();
    return handle;
}

static void test_preview_then_retire_then_new_active() {
    auto repository = std::make_shared<HistoricalKvRepository>(1);
    std::atomic<unsigned> released{0};
    const auto handle = live_block(repository, 10, released);
    const std::vector<HistoricalKvActiveRequest> preview = {{handle, false}};

    check(repository->retire(handle) == HistoricalKvRepository::RetireResult::Committed,
          "preview victim did not retire");
    auto optional = HistoricalKvActiveSet::acquire(repository, preview);
    check(optional && optional.active->empty(),
          "retired optional block entered a new active operation");
    check(repository->inspect(handle)->last_use_seq == 1,
          "mere preview or admission updated execution recency");

    auto mandatory = HistoricalKvActiveSet::acquire(repository, {{handle, true}});
    check(!mandatory && mandatory.missing_mandatory.has_value(),
          "retired mandatory block was not reported as unavailable");
    check(repository->try_reclaim(handle) == HistoricalKvRepository::ReclaimResult::Complete &&
          released.load() == 1,
          "unadmitted retired block was not reclaimed");
}

static void test_old_active_finishes_after_retirement() {
    auto repository = std::make_shared<HistoricalKvRepository>(1);
    std::atomic<unsigned> released{0};
    const auto handle = live_block(repository, 20, released);

    auto old_active = HistoricalKvActiveSet::acquire(repository, {{handle, true}});
    check(old_active && old_active.active->handles().size() == 1,
          "old active operation was not admitted");
    check(repository->inspect(handle)->last_use_seq == 1,
          "admission alone incorrectly marked execution submitted");
    check(repository->retire(handle) == HistoricalKvRepository::RetireResult::Committed,
          "active block could not retire");

    auto new_active = HistoricalKvActiveSet::acquire(repository, {{handle, false}});
    check(new_active && new_active.active->empty(),
          "new active reused an older execution lease");
    check(repository->try_reclaim(handle) == HistoricalKvRepository::ReclaimResult::Deferred,
          "payload was released during the old active operation");
    check(old_active.active->mark_execution_submitted(7),
          "admitted old operation could not report submission after retirement");
    check(repository->inspect(handle)->last_use_seq == 7,
          "submission did not update LRU recency");
    check(old_active.active->mark_execution_submitted(9) &&
          repository->inspect(handle)->last_use_seq == 9,
          "later decode submission under the same permission was not recorded");
    old_active.active.reset();
    check(repository->try_reclaim(handle) == HistoricalKvRepository::ReclaimResult::Complete &&
          released.load() == 1,
          "payload was not released after the old active operation ended");
}

static void test_mandatory_failure_rolls_back_prior_admission() {
    auto repository = std::make_shared<HistoricalKvRepository>(2);
    std::atomic<unsigned> released{0};
    const auto first = live_block(repository, 30, released);
    const auto second = live_block(repository, 31, released);
    check(repository->retire(second) == HistoricalKvRepository::RetireResult::Committed,
          "second block could not retire");

    auto result = HistoricalKvActiveSet::acquire(repository,
                                                {{first, true}, {second, true}});
    check(!result && result.missing_mandatory.has_value(),
          "mandatory failure returned a partial active window");
    const auto meta = repository->inspect(first);
    check(meta.has_value() && meta->references == 0 && meta->last_use_seq == 1,
          "failed active admission leaked its earlier lease or marked use");
    check(repository->try_reclaim(second) == HistoricalKvRepository::ReclaimResult::Complete,
          "failed admission retained the retired mandatory block");

    auto abandoned = HistoricalKvActiveSet::acquire(repository,
                                                    {{first, true}, {first, true}});
    check(abandoned && abandoned.active->handles().size() == 1,
          "duplicate selected handle was admitted twice");
    abandoned.active.reset();
    check(repository->inspect(first)->last_use_seq == 1,
          "abandoned active operation recorded a use");
}

static void test_gpu_ready_writing_requires_explicit_request() {
    auto repository = std::make_shared<HistoricalKvRepository>(1);
    std::atomic<unsigned> released{0};
    auto writer = repository->begin_write(40, 0, 8, 1, 64,
        [&released](const HistoricalKvHandle &) { ++released; return true; });
    check(writer.has_value(), "GPU-ready active fixture had no writer");
    const auto handle = writer->handle();
    auto ordinary = HistoricalKvActiveSet::acquire(repository, {{handle, false}});
    check(ordinary && ordinary.active->empty(),
          "ordinary active request selected unpublished Host payload");
    auto admitted = HistoricalKvActiveSet::acquire(repository, {{handle, true, true}});
    check(admitted && admitted.active->handles().size() == 1 &&
          admitted.active->mark_execution_submitted(5),
          "GPU-ready Writing active request did not submit its real use");
    check(repository->retire(handle) == HistoricalKvRepository::RetireResult::Committed,
          "GPU-ready active block did not retire");
    auto later = HistoricalKvActiveSet::acquire(repository, {{handle, false, true}});
    check(later && later.active->empty(), "retired GPU-ready block entered a later active");
    writer->reset();
    check(repository->try_reclaim(handle) == HistoricalKvRepository::ReclaimResult::Deferred,
          "old GPU-ready active did not protect the retiring payload");
    admitted.active.reset();
    check(repository->try_reclaim(handle) == HistoricalKvRepository::ReclaimResult::Complete &&
          released.load() == 1, "GPU-ready active payload did not finish releasing");
}

int main() {
    test_preview_then_retire_then_new_active();
    test_old_active_finishes_after_retirement();
    test_mandatory_failure_rolls_back_prior_admission();
    test_gpu_ready_writing_requires_explicit_request();
    std::cout << "historical KV active-set checks passed\n";
}
