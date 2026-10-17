#include "kvmem/historical_kv_gc.hpp"
#include "kvmem/historical_kv_raw_store.hpp"

#include <atomic>
#include <chrono>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>

using namespace kvmem;

namespace {

void check(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

// A policy can request deletion of an active block, regardless of its age or
// reference count. The manager must apply the same safe protocol as for LRU.
class DeleteOnce final : public HistoricalKvPolicy {
public:
    explicit DeleteOnce(HistoricalKvHandle victim, bool conditional = false)
        : victim_(victim), conditional_(conditional) {}
    HistoricalKvStepResult step(const HistoricalKvSnapshot & snapshot,
                                const HistoricalKvPressure &,
                                HistoricalKvWorkBudget,
                                HistoricalKvDecisionBuffer & out) override {
        if (sent_) return {HistoricalKvStepStatus::NoCandidate, 0, 0};
        HistoricalKvDeleteIntent intent;
        intent.victim = victim_;
        intent.round_id = snapshot.round_id;
        intent.reason = HistoricalKvDeleteReason::Custom;
        if (conditional_) intent.expected_content_version = 1;
        check(out.push(intent), "policy output buffer unexpectedly full");
        sent_ = true;
        return {HistoricalKvStepStatus::Complete, 1, 1};
    }
private:
    HistoricalKvHandle victim_;
    bool conditional_ = false;
    bool sent_ = false;
};

class ThrowingPolicy final : public HistoricalKvPolicy {
public:
    HistoricalKvStepResult step(const HistoricalKvSnapshot &,
                                const HistoricalKvPressure &,
                                HistoricalKvWorkBudget,
                                HistoricalKvDecisionBuffer &) override {
        throw std::runtime_error("test policy failure");
    }
};

class DuplicatePolicy final : public HistoricalKvPolicy {
public:
    explicit DuplicatePolicy(HistoricalKvHandle victim) : victim_(victim) {}
    HistoricalKvStepResult step(const HistoricalKvSnapshot & snapshot,
                                const HistoricalKvPressure &,
                                HistoricalKvWorkBudget,
                                HistoricalKvDecisionBuffer & out) override {
        HistoricalKvDeleteIntent intent;
        intent.victim = victim_;
        intent.round_id = snapshot.round_id;
        check(out.push(intent) && out.push(intent),
              "duplicate policy output buffer unexpectedly full");
        return {HistoricalKvStepStatus::Complete, 1, 2};
    }
private:
    HistoricalKvHandle victim_;
};

HistoricalKvGc::Config config(HistoricalKvGc::Mode mode) {
    HistoricalKvGc::Config cfg;
    cfg.mode = mode;
    cfg.ready_low = 1;
    cfg.ready_high = 2;
    cfg.scan_blocks = 2;
    cfg.batch_blocks = 1;
    cfg.reclaim_blocks = 2;
    cfg.poll_ms = 1;
    return cfg;
}

void test_retire_active_then_background_release() {
    auto repo = std::make_shared<HistoricalKvRepository>(2);
    std::atomic<int> releases{0};
    std::atomic<bool> callback_on_main{false};
    const auto main_thread = std::this_thread::get_id();
    auto writer = repo->begin_write(7, 128, 32, 1, 1024,
        [&](const HistoricalKvHandle &) {
            callback_on_main.store(std::this_thread::get_id() == main_thread);
            ++releases;
            return true;
        });
    check(writer.has_value(), "write slot missing");
    const auto handle = writer->handle();
    check(repo->publish_write(*writer), "write publish failed");
    writer->reset();
    auto current = repo->acquire_new(handle);
    check(current.has_value(), "execution lease missing");

    HistoricalKvGc gc(repo, std::make_unique<DeleteOnce>(handle),
                      config(HistoricalKvGc::Mode::Delete));
    gc.manual_tick();
    check(gc.stats().committed == 1, "active block was vetoed by GC manager");
    check(!repo->acquire_new(handle), "retired block granted a new active lease");
    gc.manual_tick();
    check(releases == 0 && gc.stats().deferred != 0,
          "physical bytes freed while an old execution lease was alive");
    current->reset();
    gc.start();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (releases == 0 && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    gc.stop();
    check(releases == 1 && !callback_on_main,
          "resource release did not run exactly once on the background worker");
    check(repo->ready_count() == 2, "reclaimed slot was not reusable");
}

void test_shadow_and_conditional_intent() {
    auto repo = std::make_shared<HistoricalKvRepository>(2);
    std::atomic<int> releases{0};
    auto writer = repo->begin_write(9, 0, 16, 1, 128,
        [&](const HistoricalKvHandle &) { ++releases; return true; });
    check(writer.has_value(), "write slot missing");
    const auto handle = writer->handle();
    check(repo->publish_write(*writer), "write publish failed");
    writer->reset();
    {
        HistoricalKvGc gc(repo, std::make_unique<DeleteOnce>(handle),
                          config(HistoricalKvGc::Mode::Shadow));
        gc.manual_tick();
        check(gc.stats().shadow == 1 && gc.stats().committed == 0,
              "shadow mode changed the repository");
        check(repo->acquire_new(handle).has_value() && releases == 0,
              "shadow mode made the block unavailable");
    }
    {
        HistoricalKvGc gc(repo, std::make_unique<DeleteOnce>(handle, true),
                          config(HistoricalKvGc::Mode::Delete));
        gc.manual_tick();
        check(gc.stats().reevaluate == 1 && gc.stats().committed == 0,
              "unsupported content-version condition was ignored");
        check(repo->acquire_new(handle).has_value() && releases == 0,
              "conditional intent deleted data without atomic version check");
    }
}

void test_releasing_retry_and_off_drain() {
    auto repo = std::make_shared<HistoricalKvRepository>(2);
    int attempts = 0;
    auto writer = repo->begin_write(11, 64, 16, 1, 256,
        [&](const HistoricalKvHandle &) { return ++attempts >= 2; });
    check(writer.has_value(), "write slot missing");
    const auto handle = writer->handle();
    check(repo->publish_write(*writer), "write publish failed");
    writer->reset();
    {
        HistoricalKvGc gc(repo, std::make_unique<DeleteOnce>(handle),
                          config(HistoricalKvGc::Mode::Delete));
        gc.manual_tick();
        check(gc.stats().committed == 1, "retirement was not committed");
    }
    // Turning off new decisions must not abandon resources already retired.
    HistoricalKvGc gc(repo, std::make_unique<DeleteOnce>(handle),
                      config(HistoricalKvGc::Mode::Off));
    gc.manual_tick();
    check(attempts == 1 && repo->ready_count() == 1,
          "temporary release failure must keep the slot unavailable");
    gc.manual_tick();
    check(attempts == 2 && repo->ready_count() == 2,
          "releasing allocation was not retried after GC was switched off");
}

void test_lru_policy_through_manager() {
    auto repo = std::make_shared<HistoricalKvRepository>(4);
    std::atomic<int> reclaimed_oldest{0};
    HistoricalKvHandle handles[3];
    for (uint32_t i = 0; i < 3; ++i) {
        auto writer = repo->begin_write(i, i * 16, 16, i + 1, 64,
            [&, i](const HistoricalKvHandle &) {
                if (i == 2) ++reclaimed_oldest;
                return true;
            });
        check(writer && repo->publish_write(*writer), "LRU setup failed");
        handles[i] = writer->handle();
        writer->reset();
    }
    auto recent_a = repo->acquire_new(handles[0]);
    auto recent_b = repo->acquire_new(handles[1]);
    check(recent_a && recent_b && repo->mark_executed(*recent_a, 20) &&
          repo->mark_executed(*recent_b, 10), "LRU use tracking failed");
    recent_a->reset();
    recent_b->reset();
    auto active_oldest = repo->acquire_new(handles[2]);
    check(active_oldest && repo->mark_executed(*active_oldest, 4),
          "active oldest block setup failed");
    auto cfg = config(HistoricalKvGc::Mode::Delete);
    cfg.ready_low = 2;
    cfg.ready_high = 3;
    HistoricalKvGc gc(repo, std::make_unique<HistoricalKvLruPolicy>(4), cfg);
    for (int i = 0; i < 12 && gc.stats().committed == 0; ++i) gc.manual_tick();
    check(gc.stats().committed == 1 && !repo->acquire_new(handles[2]),
          "LRU/manager failed to retire the oldest active block");
    check(reclaimed_oldest == 0, "LRU victim released under active execution");
    active_oldest->reset();
    for (int i = 0; i < 6 && reclaimed_oldest == 0; ++i) gc.manual_tick();
    check(reclaimed_oldest == 1, "LRU victim was not physically reclaimed");
}

void test_policy_failure_does_not_stop_reclaimer() {
    auto repo = std::make_shared<HistoricalKvRepository>(2);
    int releases = 0;
    auto writer = repo->begin_write(13, 0, 1, 1, 8,
        [&](const HistoricalKvHandle &) { ++releases; return true; });
    check(writer && repo->publish_write(*writer), "failure setup failed");
    const auto handle = writer->handle();
    HistoricalKvGc gc(repo, std::make_unique<ThrowingPolicy>(),
                      config(HistoricalKvGc::Mode::Delete));
    gc.manual_tick();
    check(gc.stats().policy_failures == 1, "policy error was not reported");
    check(repo->retire(handle) == HistoricalKvRepository::RetireResult::Committed,
          "retire after policy error failed");
    writer->reset();
    gc.manual_tick();
    check(releases == 1 && repo->ready_count() == 2,
          "policy error stopped already retired resource cleanup");
}

void test_one_manager_owns_each_repository() {
    auto repo = std::make_shared<HistoricalKvRepository>(2);
    {
        HistoricalKvGc owner(repo, std::make_unique<DeleteOnce>(HistoricalKvHandle{}),
                             config(HistoricalKvGc::Mode::Off));
        bool rejected = false;
        try {
            HistoricalKvGc second(repo, std::make_unique<DeleteOnce>(HistoricalKvHandle{}),
                                  config(HistoricalKvGc::Mode::Off));
        } catch (const std::logic_error &) { rejected = true; }
        check(rejected, "two GC managers claimed one repository");
    }
    HistoricalKvGc replacement(repo,
        std::make_unique<DeleteOnce>(HistoricalKvHandle{}),
        config(HistoricalKvGc::Mode::Off));
    replacement.manual_tick();
}

void test_duplicate_intents_count_once_in_shadow() {
    auto repo = std::make_shared<HistoricalKvRepository>(2);
    auto writer = repo->begin_write(17, 0, 1, 1, 8,
        [](const HistoricalKvHandle &) { return true; });
    check(writer && repo->publish_write(*writer), "duplicate setup failed");
    const auto handle = writer->handle();
    writer->reset();
    auto cfg = config(HistoricalKvGc::Mode::Shadow);
    cfg.batch_blocks = 2;
    HistoricalKvGc gc(repo, std::make_unique<DuplicatePolicy>(handle), cfg);
    gc.manual_tick();
    check(gc.stats().proposed == 2 && gc.stats().shadow == 1 &&
          repo->acquire_new(handle).has_value(),
          "duplicate victim was counted twice in a shadow round");
}

void test_lru_shadow_does_not_repropose_same_generations() {
    auto repo = std::make_shared<HistoricalKvRepository>(4);
    HistoricalKvHandle handles[2];
    for (uint32_t i = 0; i < 2; ++i) {
        auto writer = repo->begin_write(30 + i, i, 1, i + 1, 8,
            [](const HistoricalKvHandle &) { return true; });
        check(writer && repo->publish_write(*writer), "shadow LRU setup failed");
        handles[i] = writer->handle();
        writer->reset();
    }
    auto cfg = config(HistoricalKvGc::Mode::Shadow);
    cfg.ready_low = 3;
    cfg.ready_high = 4;
    cfg.scan_blocks = cfg.reclaim_blocks = 4;
    cfg.batch_blocks = 2;
    HistoricalKvGc gc(repo, std::make_unique<HistoricalKvLruPolicy>(repo->capacity()), cfg);
    for (int tick = 0; tick < 100; ++tick) gc.manual_tick();
    const auto stats = gc.stats();
    check(stats.proposed == 2 && stats.shadow == 2 && !stats.committed &&
          repo->ready_count() == 2 && repo->acquire_new(handles[0]).has_value() &&
          repo->acquire_new(handles[1]).has_value(),
          "LRU shadow repeatedly proposed or physically retired the same generations");
}

void test_custom_policy_needs_no_lru_watermarks() {
    auto repo = std::make_shared<HistoricalKvRepository>(2);
    auto writer = repo->begin_write(18, 0, 1, 1, 8,
        [](const HistoricalKvHandle &) { return true; });
    check(writer && repo->publish_write(*writer), "custom policy setup failed");
    const auto handle = writer->handle();
    writer->reset();
    auto cfg = config(HistoricalKvGc::Mode::Delete);
    cfg.ready_low = 0;
    cfg.ready_high = 0;
    HistoricalKvGc gc(repo, std::make_unique<DeleteOnce>(handle), cfg);
    gc.manual_tick();
    check(gc.stats().committed == 1 && !repo->acquire_new(handle),
          "manager imposed LRU pressure on a custom deletion strategy");
}

void test_manager_reclaims_real_ram_payload() {
    auto repo = std::make_shared<HistoricalKvRepository>(2);
    HistoricalKvRawStoreConfig raw_cfg;
    raw_cfg.n_layer = 1;
    raw_cfg.max_tokens_per_slot = 4;
    raw_cfg.packed_k_bytes_per_token = 1;
    raw_cfg.packed_v_bytes_per_token = 1;
    auto raw = std::make_shared<HistoricalKvRawStore>(raw_cfg, 2);
    std::weak_ptr<HistoricalKvRepository> weak_repo = repo;
    auto writer = repo->begin_write(21, 713, 2, 1, 4,
        [raw, weak_repo](const HistoricalKvHandle & handle) {
            auto owner = weak_repo.lock();
            if (!owner) return false;
            const auto result = raw->release(*owner, handle);
            return result.code == HistoricalKvRawStore::ReleaseCode::Released ||
                   result.code == HistoricalKvRawStore::ReleaseCode::AlreadyReleased ||
                   result.code == HistoricalKvRawStore::ReleaseCode::Absent;
        });
    check(writer.has_value(), "RAM manager writer missing");
    const auto handle = writer->handle();
    check(raw->reserve(*writer, handle, 713, 2) &&
          raw->write_layer(*writer, handle, 0, {1, 2}, {3, 4}, {}, 0) &&
          raw->publish_complete(*repo, *writer, handle),
          "RAM manager payload setup failed");
    writer->reset();
    auto reader = repo->acquire_read(handle);
    check(reader.has_value(), "RAM manager reader missing");

    HistoricalKvGc gc(repo, std::make_unique<DeleteOnce>(handle),
                      config(HistoricalKvGc::Mode::Delete));
    gc.manual_tick();
    check(gc.stats().committed == 1 && !repo->acquire_new(handle),
          "manager did not close new RAM reads");
    gc.manual_tick();
    check(raw->allocated_payload_bytes() >= 4 &&
          raw->read_layer(*reader, handle, 0).has_value(),
          "old admitted RAM reader lost its payload before completion");
    reader->reset();
    gc.manual_tick();
    check(raw->allocated_payload_bytes() == 0 && repo->ready_count() == 2 &&
          !raw->inspect(*repo, handle), "manager did not release real RAM bytes");
}

} // namespace

int main() {
    try {
        test_retire_active_then_background_release();
        test_shadow_and_conditional_intent();
        test_releasing_retry_and_off_drain();
        test_lru_policy_through_manager();
        test_policy_failure_does_not_stop_reclaimer();
        test_one_manager_owns_each_repository();
        test_duplicate_intents_count_once_in_shadow();
        test_lru_shadow_does_not_repropose_same_generations();
        test_custom_policy_needs_no_lru_watermarks();
        test_manager_reclaims_real_ram_payload();
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
    std::cout << "Historical KV manager tests passed\n";
}
