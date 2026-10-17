#include "kvmem/historical_kv_active_set.hpp"

#include <atomic>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <new>
#include <stdexcept>
#include <vector>

using namespace kvmem;

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

static void test_prepared_active_reuses_storage_but_not_permission() {
    auto repository = std::make_shared<HistoricalKvRepository>(1);
    std::atomic<unsigned> released{0};
    bool prepared = false;
    unsigned failures = 0;
    for (int64_t budget = 0; budget < 10; ++budget) {
        allocation_budget = budget;
        try { auto attempt = HistoricalKvActiveSet::prepare(repository, 2); prepared = bool(attempt); }
        catch (const std::bad_alloc &) { ++failures; }
        allocation_budget = -1;
        check(!repository->outstanding_operations(), "failed preparation leaked permissions");
        if (prepared) break;
    }
    check(prepared && failures, "prepared active allocation sweep missed failure or success");
    auto first = HistoricalKvActiveSet::prepare(repository, 2);
    auto later = HistoricalKvActiveSet::prepare(repository, 1);
    const auto bytes = first->capacity_bytes();
    const auto * data = first->handles().data();
    std::vector<HistoricalKvActiveRequest> requested(2), optional(1), too_many(3);
    using Status = HistoricalKvActiveSet::AcquireStatus;
    for (unsigned i = 0; i < 64; ++i) {
        const auto handle = live_block(repository, i, released);
        requested[0] = requested[1] = {handle, true};
        optional[0] = {handle, false};
        const auto before = foreground_frees;
        allocation_budget = 0;
        const auto oversized = first->try_acquire(too_many);
        const auto acquired = first->try_acquire(requested);
        const auto overlapping = first->try_acquire(requested);
        const bool submitted = first->mark_execution_submitted(i + 2);
        const auto retirement = repository->retire(handle);
        const auto denied = later->try_acquire(optional);
        const auto deferred = repository->try_reclaim(handle);
        allocation_budget = -1;
        check(oversized.status == Status::CapacityExceeded && acquired && submitted &&
              overlapping.status == Status::Busy && first->engaged() && first->handles().size() == 1 &&
              denied && later->engaged() && later->empty() && foreground_frees == before &&
              retirement == HistoricalKvRepository::RetireResult::Committed &&
              deferred == HistoricalKvRepository::ReclaimResult::Deferred &&
              repository->inspect(handle)->last_use_seq == i + 2,
              "prepared active grew, freed, overlapped, reused retired permission, or lost its old lease");
        allocation_budget = 0;
        first->release(); later->release();
        const bool idle_submission = first->mark_execution_submitted(i + 3);
        allocation_budget = -1;
        check(!idle_submission && !first->engaged() && first->empty() &&
              first->handles().data() == data && first->capacity_bytes() == bytes &&
              foreground_frees == before &&
              repository->try_reclaim(handle) == HistoricalKvRepository::ReclaimResult::Complete,
              "active release retained a lease, freed prepared storage, or recorded an idle use");
    }
    check(released.load() == 64, "generation reuse lost physical release");
    repository->close();
    allocation_budget = 0;
    const auto closed = first->try_acquire({});
    allocation_budget = -1;
    check(closed.status == Status::Closed && !first->engaged(), "closed repository admitted an empty active");
}

static void test_prepared_mandatory_failure_returns_all_leases() {
    auto repository = std::make_shared<HistoricalKvRepository>(2);
    std::atomic<unsigned> released{0};
    const auto a = live_block(repository, 0, released);
    const auto b = live_block(repository, 1, released);
    auto active = HistoricalKvActiveSet::prepare(repository, 2);
    const std::vector<HistoricalKvActiveRequest> requested{{a, true}, {b, true}};
    repository->retire(b);
    allocation_budget = 0;
    const auto failed = active->try_acquire(requested);
    allocation_budget = -1;
    check(failed.status == HistoricalKvActiveSet::AcquireStatus::MissingMandatory &&
          failed.missing_mandatory && !active->engaged() && active->empty() &&
          !repository->outstanding_operations() && repository->inspect(a)->last_use_seq == 1,
          "mandatory failure leaked a partial reserved active");
    const std::vector<HistoricalKvActiveRequest> retry{{a, true}};
    allocation_budget = 0;
    const auto acquired = active->try_acquire(retry);
    active->release();
    allocation_budget = -1;
    check(bool(acquired), "failed prepared active could not be reused");
}

int main() try {
    test_preview_then_retire_then_new_active();
    test_old_active_finishes_after_retirement();
    test_mandatory_failure_rolls_back_prior_admission();
    test_gpu_ready_writing_requires_explicit_request();
    test_prepared_active_reuses_storage_but_not_permission();
    test_prepared_mandatory_failure_returns_all_leases();
    std::cout << "historical KV active-set checks passed: prepared storage, 64 generation reuses, fresh permission, bounded admission\n";
} catch (const std::exception & e) {
    allocation_budget = -1;
    std::cerr << "FAIL historical active set: " << e.what() << '\n';
    return 1;
}
