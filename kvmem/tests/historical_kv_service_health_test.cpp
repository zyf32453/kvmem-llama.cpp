#include "kvmem/historical_kv_gc.hpp"
#include "kvmem/memory_contract.hpp"
#include <iostream>
#include <stdexcept>

using namespace kvmem;
using Repo = HistoricalKvRepository;
static void check(bool value, const char * message) { if (!value) throw std::runtime_error(message); }
template<class F> static void unavailable(F operation) {
    try { operation(); }
    catch (const MemoryError & error) {
        check(error.code() == MemoryErrorCode::ServiceUnavailable, "unavailable reported as capacity exhaustion");
        return;
    }
    throw std::runtime_error("unhealthy GC accepted an operation");
}
class BrokenPolicy final : public HistoricalKvPolicy {
public:
    explicit BrokenPolicy(bool throws) : throws_(throws) {}
    HistoricalKvStepResult step(const HistoricalKvSnapshot & snapshot, const HistoricalKvPressure &,
        HistoricalKvWorkBudget, HistoricalKvDecisionBuffer & out) override {
        ++calls;
        for (const auto & meta : snapshot.blocks) if (meta.life == HistoricalKvLife::Live) {
            out.push({meta.handle, snapshot.round_id, HistoricalKvDeleteReason::Custom, {}});
            if (throws_) throw std::runtime_error("injected failure after partial proposal");
            return {HistoricalKvStepStatus::Complete, 1, 1};
        }
        return {HistoricalKvStepStatus::NoCandidate, 0, 0};
    }
    unsigned calls = 0;
private:
    bool throws_;
};
static HistoricalKvGc::Config config() {
    HistoricalKvGc::Config cfg;
    cfg.mode = HistoricalKvGc::Mode::Delete; cfg.retire_at_decode_boundary = true;
    cfg.ready_low = 1; cfg.ready_high = 2; cfg.batch_blocks = 1;
    cfg.scan_blocks = cfg.reclaim_blocks = 4;
    return cfg;
}
static void policy_failure_is_visible_and_old_retirement_progresses(bool throws) {
    auto repository = std::make_shared<Repo>(4);
    bool allow_release = false;
    unsigned releases = 0;
    HistoricalKvHandle old;
    for (unsigned i = 0; i < 4; ++i) {
        auto writer = repository->begin_write(i, i * 64, 64, i + 1, 16,
            [&](const HistoricalKvHandle &) { if (!allow_release) return false; ++releases; return true; });
        check(bool(writer) && repository->publish_write(*writer), "health fixture publish");
        old = writer->handle();
    }
    check(repository->retire(old) == Repo::RetireResult::Committed, "health fixture retirement");
    auto policy = std::make_unique<BrokenPolicy>(throws);
    auto * observed = policy.get();
    HistoricalKvGc gc(repository, std::move(policy), config());
    gc.manual_tick();
    check(gc.health() == HistoricalKvGc::Health::PolicyFailed && gc.stats().policy_failures == 1 &&
          gc.stats().unsupported_conditions == (throws ? 0u : 1u), "policy failure not published");
    check(gc.commit_at_decode_boundary().committed == 0, "failed partial proposal committed");
    unavailable([&] { gc.ensure_healthy(); });
    unavailable([&] { gc.request_capacity(1); });
    unavailable([&] { gc.start(); });
    allow_release = true;
    gc.manual_tick();
    check(releases == 1 && gc.stats().reclaimed == 1 && gc.stats().committed == 0 && observed->calls == 1,
          "failed strategy retried or blocked old release");
    gc.request_capacity(0);
    gc.stop();
    check(gc.health() == HistoricalKvGc::Health::PolicyFailed, "stop erased failure diagnosis");
}
static void stopped_prepared_batch_cannot_change_visibility() {
    auto repository = std::make_shared<Repo>(4);
    HistoricalKvHandle victim;
    for (unsigned i = 0; i < 4; ++i) {
        auto writer = repository->begin_write(i, i * 64, 64, i + 1, 16,
            [](const HistoricalKvHandle &) { return true; });
        check(bool(writer) && repository->publish_write(*writer), "stop fixture publish");
        if (i == 0) victim = writer->handle();
    }
    HistoricalKvGc gc(repository, std::make_unique<HistoricalKvLruPolicy>(4), config());
    gc.manual_tick();
    check(gc.stats().proposed == 1, "no prepared stop fixture");
    gc.request_stop();
    check(gc.health() == HistoricalKvGc::Health::Stopped && gc.commit_at_decode_boundary().prepared == 0 &&
          repository->completion_state(victim) == Repo::CompletionState::Live, "stopped owner retired an orphan batch");
    unavailable([&] { gc.request_capacity(1); });
    unavailable([&] { gc.start(); });
    gc.manual_tick();
    check(gc.stats().proposed == 1 && !gc.stats().committed, "stopped owner generated fresh decisions");
}
int main() {
    try {
        policy_failure_is_visible_and_old_retirement_progresses(true);
        policy_failure_is_visible_and_old_retirement_progresses(false);
        stopped_prepared_batch_cannot_change_visibility();
        std::cout << "PASS service health: partial policy failure/unsupported conditions visible, old release progresses, stopped prepared batch closed and no silent restart\n";
    } catch (const std::exception & error) { std::cerr << "FAIL service health: " << error.what() << '\n'; return 1; }
}
