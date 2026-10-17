#include "kvmem/historical_kv_repository.hpp"

#include <atomic>
#include <iostream>
#include <stdexcept>
#include <thread>

using namespace kvmem;

static void check(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

static HistoricalKvRepository::ReleaseCallback successful_release(
    std::atomic<unsigned> & count) {
    return [&count](const HistoricalKvHandle &) {
        count.fetch_add(1, std::memory_order_relaxed);
        return true;
    };
}

static void test_retire_closes_new_entry_but_keeps_old_reader() {
    HistoricalKvRepository repo(1, 19);
    std::atomic<unsigned> releases{0};
    auto writer = repo.begin_write(42, 512, 31, 1, 4096, successful_release(releases));
    check(writer.has_value(), "initial slot reservation failed");
    const auto handle = writer->handle();
    check(repo.publish_write(*writer), "writer failed to publish Live");
    auto old_execution = repo.acquire_new(handle);
    check(old_execution.has_value(), "old execution did not get a lease");
    check(repo.retire(handle, 2) == HistoricalKvRepository::RetireResult::VersionChanged &&
          repo.acquire_new(handle).has_value(),
          "mismatched content version retired a live generation");
    check(repo.retire(handle, 1) == HistoricalKvRepository::RetireResult::Committed,
          "retire with live references failed");
    check(!repo.acquire_new(handle), "retired block granted new execution");
    check(repo.mark_executed(*old_execution, 5),
          "previously admitted execution could not report its use after retirement");
    check(repo.retire(handle) == HistoricalKvRepository::RetireResult::AlreadyRetired,
          "duplicate retire changed an already retired block");
    writer->reset();
    check(repo.try_reclaim(handle) == HistoricalKvRepository::ReclaimResult::Deferred,
          "resource freed while an old execution still held its lease");
    check(!repo.reclaim_completed(handle, repo.repository_id()),
          "deferred release was reported as complete");
    check(repo.ready_count() == 0 && releases.load() == 0,
          "retired slot became ready before physical release");
    check(!repo.begin_write(43, 543, 5, 2, 64, successful_release(releases)),
          "full repository accepted another block");
    old_execution->reset();
    const auto foreground_thread = std::this_thread::get_id();
    std::thread::id callback_thread;
    // The release callback runs only when the worker calls try_reclaim.
    std::thread worker([&] {
        callback_thread = std::this_thread::get_id();
        check(repo.try_reclaim(handle) == HistoricalKvRepository::ReclaimResult::Complete,
              "worker failed to reclaim quiescent block");
    });
    worker.join();
    check(callback_thread != foreground_thread && releases.load() == 1,
          "release callback did not run exactly once on the reclaimer");
    check(repo.ready_count() == 1, "reclaimed slot was not returned to Ready");
    check(repo.reclaim_completed(handle, repo.repository_id()),
          "completed generation had no exact release acknowledgement");

    auto next = repo.begin_write(43, 543, 5, 2, 64, successful_release(releases));
    check(next && next->handle().slot == handle.slot &&
          next->handle().generation == handle.generation + 1,
          "Ready slot did not get a new generation");
    check(repo.reclaim_completed(handle, repo.repository_id()) &&
          !repo.reclaim_completed(next->handle(), repo.repository_id()),
          "new generation confused the old release acknowledgement");
    check(repo.retire(handle) == HistoricalKvRepository::RetireResult::Stale,
          "stale handle retired a reused slot");
    check(!repo.acquire_new(handle), "stale handle acquired a reused slot");
    check(repo.publish_write(*next), "new generation was damaged by stale handle");
    next->reset();
}

static void test_gpu_ready_writing_execution_is_not_cpu_publication() {
    HistoricalKvRepository repo(1);
    std::atomic<unsigned> releases{0};
    auto writer = repo.begin_write(50, 0, 16, 1, 128, successful_release(releases));
    check(writer.has_value(), "GPU-ready Writing fixture had no slot");
    const auto handle = writer->handle();
    check(!repo.acquire_new(handle) && !repo.acquire_read(handle) &&
          !repo.acquire_transfer(handle),
          "Writing block became generally visible before Host publication");
    auto gpu_execution = repo.acquire_gpu_ready_writing(handle);
    check(gpu_execution && repo.mark_executed(*gpu_execution, 5) &&
          repo.inspect(handle)->last_use_seq == 5,
          "GPU-complete Writing span did not receive execution permission/use");
    check(repo.retire(handle) == HistoricalKvRepository::RetireResult::Committed &&
          !repo.acquire_gpu_ready_writing(handle) && !repo.acquire_new(handle),
          "retired Writing span granted a new GPU execution lease");
    check(repo.mark_executed(*gpu_execution, 6) &&
          !repo.publish_write(*writer),
          "old admitted GPU execution lost use or retired writer republished");
    writer->reset();
    check(repo.try_reclaim(handle) == HistoricalKvRepository::ReclaimResult::Deferred &&
          releases == 0, "retired Writing span was released before old GPU execution finished");
    gpu_execution->reset();
    check(repo.try_reclaim(handle) == HistoricalKvRepository::ReclaimResult::Complete &&
          releases == 1, "GPU-ready Writing span failed eventual physical release");
}

static void test_operation_accounting_and_completion_observation() {
    HistoricalKvRepository repo(2);
    std::atomic<unsigned> releases{0};
    auto first = repo.begin_write(1, 0, 4, 1, 64, successful_release(releases));
    auto reserved = repo.try_reserve_slots(1);
    check(first && reserved && repo.outstanding_operations() == 1,
          "writer or reservation counted as the wrong number of operations");
    auto second = repo.begin_reserved_write(*reserved, 2, 4, 4, 2, 64,
                                            successful_release(releases));
    check(second && repo.outstanding_operations() == 2, "reserved writer was not counted");
    const auto old = first->handle();
    const auto unfinished = second->handle();
    check(repo.completion_state(old) == HistoricalKvRepository::CompletionState::Unfinished &&
          repo.publish_write(*first) &&
          repo.completion_state(old) == HistoricalKvRepository::CompletionState::Live,
          "completion observation lost unpublished/published state");
    auto read = repo.acquire_read(old);
    auto transfer = repo.acquire_transfer(old);
    auto execution = repo.acquire_new(old);
    check(read && transfer && execution && repo.outstanding_operations() == 5,
          "public operation lease was not counted");
    (void) repo.snapshot();
    (void) repo.inspect(old);
    check(repo.retire(old) == HistoricalKvRepository::RetireResult::Committed &&
          !repo.acquire_read(old) && repo.outstanding_operations() == 5 &&
          repo.completion_state(old) == HistoricalKvRepository::CompletionState::Retired,
          "control observations or rejected admission changed operation count");
    auto moved = std::move(*read);
    read.reset();
    check(repo.outstanding_operations() == 5, "move construction lost operation ownership");
    moved = std::move(*transfer);
    transfer.reset();
    check(repo.outstanding_operations() == 4, "move assignment duplicated operation ownership");
    execution.reset();
    moved.reset();
    first.reset();
    second.reset();
    check(repo.outstanding_operations() == 0 &&
          repo.completion_state(unfinished) == HistoricalKvRepository::CompletionState::Unfinished,
          "zero operations incorrectly proved producer publication");
    check(repo.try_reclaim(old) == HistoricalKvRepository::ReclaimResult::Complete &&
          repo.completion_state(old) == HistoricalKvRepository::CompletionState::Retired,
          "released generation no longer observed as retired");
    auto reused = repo.begin_write(3, 8, 4, 3, 64, successful_release(releases));
    check(reused && reused->handle().slot == old.slot &&
          repo.completion_state(old) == HistoricalKvRepository::CompletionState::Retired &&
          repo.completion_state(reused->handle()) == HistoricalKvRepository::CompletionState::Unfinished &&
          repo.outstanding_operations() == 1,
          "completion observation confused old and reused generation");
    const auto next = reused->handle();
    reused.reset();
    check(repo.retire(unfinished) == HistoricalKvRepository::RetireResult::Committed &&
          repo.retire(next) == HistoricalKvRepository::RetireResult::Committed &&
          repo.try_reclaim(unfinished) == HistoricalKvRepository::ReclaimResult::Complete &&
          repo.try_reclaim(next) == HistoricalKvRepository::ReclaimResult::Complete &&
          repo.outstanding_operations() == 0 && releases == 3,
          "operation accounting prevented eventual physical release");
}

static void test_writing_retire_cannot_resurrect() {
    HistoricalKvRepository repo(2, 7);
    std::atomic<unsigned> releases{0};
    auto writer = repo.begin_write(1, 0, 1, 1, 128, successful_release(releases));
    check(writer.has_value(), "writer reservation failed");
    const auto handle = writer->handle();
    check(repo.retire(handle) == HistoricalKvRepository::RetireResult::Committed,
          "Writing block could not retire");
    check(!repo.publish_write(*writer), "late writer resurrected retired history");
    writer->reset();
    check(repo.try_reclaim(handle) == HistoricalKvRepository::ReclaimResult::Complete,
          "retired writer was not released");

    auto second = repo.begin_write(2, 1, 1, 2, 128, successful_release(releases));
    check(second && repo.publish_write(*second), "publish-before-retire interleaving failed");
    const auto second_handle = second->handle();
    check(repo.retire(second_handle) == HistoricalKvRepository::RetireResult::Committed,
          "Live block could not retire");
    second->reset();
    check(repo.try_reclaim(second_handle) == HistoricalKvRepository::ReclaimResult::Complete,
          "published block was not released");
    check(releases.load() == 2, "wrong number of release callbacks");
}

static void test_failed_release_stays_closed_and_retries() {
    HistoricalKvRepository repo(1);
    std::atomic<unsigned> attempts{0};
    auto writer = repo.begin_write(9, 300, 8, 4, 256,
        [&attempts](const HistoricalKvHandle &) {
            const unsigned n = attempts.fetch_add(1) + 1;
            if (n == 2) throw std::runtime_error("transient allocator failure");
            return n >= 3;
        });
    check(writer && repo.publish_write(*writer), "writer setup failed");
    const auto handle = writer->handle();
    check(repo.retire(handle) == HistoricalKvRepository::RetireResult::Committed,
          "retirement failed");
    writer->reset();
    check(repo.try_reclaim(handle) == HistoricalKvRepository::ReclaimResult::Retry,
          "failed callback was treated as complete");
    check(repo.try_reclaim(handle) == HistoricalKvRepository::ReclaimResult::Retry,
          "throwing callback was treated as complete");
    check(repo.ready_count() == 0 && !repo.inspect(handle) && !repo.acquire_new(handle),
          "partially released block became visible");
    check(!repo.reclaim_completed(handle, repo.repository_id()),
          "unreadable Releasing generation was reported as complete");
    const auto releasing = repo.snapshot();
    check(releasing.size() == 1 && releasing[0].life == HistoricalKvLife::Releasing &&
          releasing[0].handle.generation == handle.generation,
          "snapshot lost a failed release ticket");
    check(repo.retire(handle) == HistoricalKvRepository::RetireResult::AlreadyRetired,
          "matching Releasing generation was not recognized as retired");
    check(!repo.begin_write(10, 0, 1, 5, 16,
          [](const HistoricalKvHandle &) { return true; }),
          "Releasing slot was reused before callback completed");
    check(repo.try_reclaim(handle) == HistoricalKvRepository::ReclaimResult::Complete,
          "third release attempt failed");
    check(attempts.load() == 3 && repo.ready_count() == 1,
          "callback retry/Ready accounting incorrect");
    check(repo.reclaim_completed(handle, repo.repository_id()),
          "successful callback was not acknowledged");
    check(repo.try_reclaim(handle) == HistoricalKvRepository::ReclaimResult::Stale,
          "completed release was repeated");
}

static void test_snapshot_and_last_use() {
    HistoricalKvRepository repo(1, 55);
    auto writer = repo.begin_write(44, 713, 3, 10, 120,
        [](const HistoricalKvHandle &) { return true; });
    check(writer && repo.publish_write(*writer), "snapshot setup failed");
    const auto handle = writer->handle();
    auto scorer = repo.acquire_read(handle);
    auto prefetch = repo.acquire_transfer(handle);
    check(scorer && prefetch && !repo.mark_executed(*scorer, 99) &&
          !repo.mark_executed(*prefetch, 99),
          "score/prefetch leases incorrectly counted as execution");
    check(!repo.inspect(handle)->active && repo.inspect(handle)->last_use_seq == 10,
          "read/transfer changed active or LRU recency");
    auto execution = repo.acquire_new(handle);
    check(execution && repo.inspect(handle)->active,
          "execution lease was not visible as active");
    check(repo.mark_executed(*execution, 20) && repo.mark_executed(*execution, 12),
          "execution mark failed");
    auto observed = repo.inspect(handle);
    check(observed && observed->logical_id == 44 && observed->original_begin == 713 &&
          observed->valid_tokens == 3 && observed->birth_seq == 10 &&
          observed->last_use_seq == 20 &&
          observed->payload_bytes == 120 && observed->life == HistoricalKvLife::Live,
          "guarded metadata was not copied correctly");
    const auto scan = repo.snapshot();
    check(scan.size() == 1 && scan[0].handle.generation == handle.generation,
          "bounded snapshot did not include exact generation");
    check(repo.snapshot_range(0, 1).size() == 1 &&
          repo.snapshot_range(1, 1).empty() && repo.ready_count_range(0, 1) == 0,
          "bounded range scan did not honor slot limits");
    execution->reset();
    check(!repo.inspect(handle)->active && repo.inspect(handle)->last_use_seq == 20,
          "dropping execution did not clear active while read leases remained");
    scorer->reset();
    prefetch->reset();
    check(repo.retire(handle) == HistoricalKvRepository::RetireResult::Committed,
          "snapshot block retirement failed");
    check(!repo.acquire_new(handle), "retired block accepted a new execution lease");
    writer->reset();
    check(repo.try_reclaim(handle) == HistoricalKvRepository::ReclaimResult::Complete,
          "snapshot block release failed");
}

static void test_default_incarnation_is_unique() {
    HistoricalKvRepository first(1);
    HistoricalKvRepository second(1);
    check(first.incarnation() != second.incarnation(),
          "default repositories share an incarnation");
    auto one = first.begin_write(1, 0, 1, 1, 16,
        [](const HistoricalKvHandle &) { return true; });
    auto two = second.begin_write(2, 0, 1, 1, 16,
        [](const HistoricalKvHandle &) { return true; });
    check(one && two && first.publish_write(*one) && second.publish_write(*two),
          "cross-repository setup failed");
    check(!second.acquire_new(one->handle()) &&
          second.retire(one->handle()) == HistoricalKvRepository::RetireResult::Stale &&
          second.try_reclaim(one->handle()) == HistoricalKvRepository::ReclaimResult::Stale,
          "handle from another repository modified this incarnation");
    check(second.acquire_new(two->handle()).has_value(),
          "cross-repository stale handle damaged the valid block");
}

static void test_bounded_observation_counts_each_slot_once() {
    HistoricalKvRepository repo(4, 91);
    const auto release_now = [](const HistoricalKvHandle &) { return true; };
    auto live = repo.begin_write(1, 0, 2, 1, 16, release_now);
    auto retired = repo.begin_write(2, 2, 2, 2, 16, release_now);
    unsigned attempts = 0;
    auto releasing = repo.begin_write(3, 4, 2, 3, 16,
        [&attempts](const HistoricalKvHandle &) { return ++attempts > 1; });
    check(live && retired && releasing && repo.publish_write(*live) &&
          repo.publish_write(*retired) && repo.publish_write(*releasing),
          "range observation setup failed");
    const auto retired_handle = retired->handle();
    const auto releasing_handle = releasing->handle();
    live->reset();
    releasing->reset();
    check(repo.retire(retired_handle) == HistoricalKvRepository::RetireResult::Committed &&
          repo.retire(releasing_handle) == HistoricalKvRepository::RetireResult::Committed &&
          repo.try_reclaim(releasing_handle) == HistoricalKvRepository::ReclaimResult::Retry,
          "mixed state setup failed");

    const auto whole = repo.observe_range(0, 4);
    check(whole.ready == 1 && whole.pending == 2 && whole.blocks.size() == 3 &&
          whole.blocks[0].handle.slot == 0 && whole.blocks[0].life == HistoricalKvLife::Live &&
          whole.blocks[1].handle.slot == 1 && whole.blocks[1].life == HistoricalKvLife::Retired &&
          whole.blocks[2].handle.slot == 2 && whole.blocks[2].life == HistoricalKvLife::Releasing,
          "one-pass range observation lost order or double-counted a slot");
    const auto middle = repo.observe_range(1, 2);
    check(middle.ready == 0 && middle.pending == 2 && middle.blocks.size() == 2,
          "bounded middle range counted an outside slot");
    const auto tail = repo.observe_range(3, 100);
    check(tail.ready == 1 && tail.pending == 0 && tail.blocks.empty(),
          "bounded tail range failed to clamp at capacity");
    HistoricalKvRepository::RangeObservation reused;
    reused.blocks.reserve(4);
    const auto * scratch = reused.blocks.data();
    repo.observe_range_into(0, 4, reused);
    check(reused.blocks.data() == scratch && reused.ready == whole.ready &&
          reused.pending == whole.pending && reused.blocks.size() == whole.blocks.size(),
          "reusable observation changed storage or pressure counts");
    repo.observe_range_into(3, 100, reused);
    check(reused.blocks.data() == scratch && reused.ready == 1 &&
          !reused.pending && reused.blocks.empty(),
          "reusable observation retained a prior round's entries");
    HistoricalKvRepository::RangeObservation undersized;
    undersized.ready = 9;
    bool rejected = false;
    try { repo.observe_range_into(0, 4, undersized); }
    catch (const std::length_error &) { rejected = true; }
    check(rejected && undersized.ready == 9 && undersized.blocks.empty(),
          "undersized observation mutated output before rejection");
    check(repo.snapshot_range(1, 2).size() == 2 && repo.ready_count_range(3, 1) == 1,
          "legacy range APIs changed behavior");

    retired->reset();
    check(repo.try_reclaim(retired_handle) == HistoricalKvRepository::ReclaimResult::Complete &&
          repo.try_reclaim(releasing_handle) == HistoricalKvRepository::ReclaimResult::Complete,
          "range observation cleanup failed");
    const auto after = repo.observe_range(0, 4);
    check(after.ready == 3 && after.pending == 0 && after.blocks.size() == 1,
          "range pressure counts did not follow completed release");
}

static void test_close_blocks_new_work_but_old_lease_can_finish() {
    HistoricalKvRepository repo(1);
    auto writer = repo.begin_write(1, 0, 1, 1, 8,
        [](const HistoricalKvHandle &) { return true; });
    check(writer && repo.publish_write(*writer), "close setup failed");
    const auto handle = writer->handle();
    auto old = repo.acquire_new(handle);
    check(old.has_value(), "old lease missing before close");
    repo.close();
    check(!repo.acquire_new(handle), "closed repository granted a new lease");
    check(!repo.begin_write(2, 1, 1, 2, 8,
          [](const HistoricalKvHandle &) { return true; }),
          "closed repository accepted a new writer");
    check(repo.retire(handle) == HistoricalKvRepository::RetireResult::Committed,
          "close prevented retirement of existing work");
    writer->reset();
    check(repo.try_reclaim(handle) == HistoricalKvRepository::ReclaimResult::Deferred,
          "close freed old lease before it completed");
    old->reset();
    check(repo.try_reclaim(handle) == HistoricalKvRepository::ReclaimResult::Complete,
          "close prevented draining existing work");
}

static void test_repository_identity_and_releasing_authority() {
    HistoricalKvRepository first(1, 70);
    HistoricalKvRepository second(1, 70);
    bool callback_saw_releasing = false;
    auto writer = first.begin_write(1, 0, 1, 1, 8,
        [&](const HistoricalKvHandle & handle) {
            callback_saw_releasing = first.is_releasing(handle);
            return callback_saw_releasing;
        });
    auto foreign = second.begin_write(1, 0, 1, 1, 8,
        [](const HistoricalKvHandle &) { return true; });
    check(writer && foreign &&
          writer->handle().incarnation == foreign->handle().incarnation &&
          writer->handle().slot == foreign->handle().slot &&
          writer->handle().generation == foreign->handle().generation,
          "explicit equal incarnations did not reproduce a handle collision");
    check(first.repository_id() != second.repository_id() &&
          first.owns_writer(*writer) && !first.owns_writer(*foreign) &&
          second.owns_writer(*foreign) && !second.owns_writer(*writer),
          "repository identity accepted a foreign writer with the same handle");
    const auto handle = writer->handle();
    check(!first.reclaim_completed(handle, second.repository_id()) &&
          !second.reclaim_completed(handle, first.repository_id()),
          "equal-looking foreign handle was accepted as a release acknowledgement");
    check(!first.is_releasing(handle), "Writing was mistaken for Releasing");
    check(first.retire(handle) == HistoricalKvRepository::RetireResult::Committed &&
          !first.is_releasing(handle),
          "Retired with an old writer was mistaken for Releasing");
    writer->reset();
    check(first.try_reclaim(handle) == HistoricalKvRepository::ReclaimResult::Complete &&
          callback_saw_releasing && !first.is_releasing(handle),
          "physical release did not occur only in the Releasing phase");
    check(first.reclaim_completed(handle, first.repository_id()) &&
          !first.reclaim_completed(handle, second.repository_id()) &&
          !second.reclaim_completed(handle, second.repository_id()),
          "one repository's release completed the other's equal-looking handle");
    check(second.retire(foreign->handle()) == HistoricalKvRepository::RetireResult::Committed,
          "foreign repository cleanup could not retire its writer");
    foreign->reset();
    check(second.try_reclaim(handle) == HistoricalKvRepository::ReclaimResult::Complete,
          "foreign repository cleanup could not release its slot");
    check(second.reclaim_completed(handle, second.repository_id()),
          "second repository did not acknowledge its own release");
}

static void test_final_publication_under_ref_churn_and_retirement() {
    using Repo = HistoricalKvRepository;
    Repo repo(1);
    for (uint32_t round = 0; round < 200; ++round) {
        auto writer = repo.begin_write(round, round, 1, round + 1, 8, [](auto) { return true; });
        check(writer.has_value(), "publication race could not reuse slot");
        const auto handle = writer->handle();
        std::atomic<bool> start{false}, done{false};
        std::thread observer([&] {
            while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
            while (!done.load(std::memory_order_acquire)) (void) repo.inspect(handle);
        });
        start.store(true, std::memory_order_release);
        // Metadata pins constantly change refs. The last producer must still
        // publish in one operation; it cannot wait for another future write.
        check(repo.publish_write(*writer), "reference churn lost final publication");
        check(!repo.publish_write(*writer), "duplicate publication changed a Live phase");
        done.store(true, std::memory_order_release);
        observer.join();
        check(repo.inspect(handle)->references == 1, "publication changed reference bits");
        check(repo.retire(handle) == Repo::RetireResult::Committed, "publication cleanup retirement failed");
        check(!repo.publish_write(*writer) && repo.inspect(handle)->life == HistoricalKvLife::Retired,
              "completion resurrected a retired generation");
        writer->reset();
        check(!repo.publish_write(*writer) && repo.try_reclaim(handle) == Repo::ReclaimResult::Complete,
              "completed writer could modify/release the wrong phase");

        writer = repo.begin_write(round, round, 1, round + 1, 8, [](auto) { return true; });
        check(writer.has_value(), "retirement race could not reuse slot");
        const auto next = writer->handle();
        std::atomic<bool> race{false};
        Repo::RetireResult retired = Repo::RetireResult::Retry;
        std::thread retire([&] {
            while (!race.load(std::memory_order_acquire)) std::this_thread::yield();
            // Retire retains its bounded-CAS retry contract. Only this test
            // actor retries; publication itself has no retry loop.
            for (unsigned attempt = 0; attempt < 100 && retired == Repo::RetireResult::Retry; ++attempt)
                retired = repo.retire(next);
        });
        race.store(true, std::memory_order_release);
        (void) repo.publish_write(*writer);
        retire.join();
        check(retired == Repo::RetireResult::Committed &&
              repo.inspect(next)->life == HistoricalKvLife::Retired &&
              repo.inspect(next)->references == 1 && !repo.publish_write(*writer),
              "publication and retirement did not preserve sticky retirement/refs");
        writer->reset();
        check(repo.try_reclaim(next) == Repo::ReclaimResult::Complete,
              "publication race lost physical release");
    }
}

int main() {
    test_retire_closes_new_entry_but_keeps_old_reader();
    test_gpu_ready_writing_execution_is_not_cpu_publication();
    test_operation_accounting_and_completion_observation();
    test_writing_retire_cannot_resurrect();
    test_failed_release_stays_closed_and_retries();
    test_snapshot_and_last_use();
    test_default_incarnation_is_unique();
    test_bounded_observation_counts_each_slot_once();
    test_close_blocks_new_work_but_old_lease_can_finish();
    test_repository_identity_and_releasing_authority();
    test_final_publication_under_ref_churn_and_retirement();
    std::cout << "historical KV repository checks passed\n";
}
