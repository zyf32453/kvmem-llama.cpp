#include "kvmem/historical_kv_gc.hpp"
#include "kvmem/historical_kv_page_features.hpp"
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <iostream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace kvmem;
using Repo = HistoricalKvRepository;
static void check(bool value, const char * message) { if (!value) throw std::runtime_error(message); }
static bool same(HistoricalKvHandle a, HistoricalKvHandle b) {
    return a.incarnation == b.incarnation && a.slot == b.slot && a.generation == b.generation;
}

enum class Fault { None, Missing, NoWitness, Self, Generation, Version, Incarnation,
    Similarity, Nonfinite, Query, Score, Schema, Round, VictimVersion, BatchDelete, Cycle, Alternate };

class RecordingPolicy final : public HistoricalKvPolicy {
public:
    RecordingPolicy(size_t capacity, std::shared_ptr<HistoricalKvPageFeatures> source, Fault fault,
                    bool one_shot = true)
        : inner_(capacity, std::move(source), policy_config(capacity)), temporary_(capacity),
          fault_(fault), one_shot_(one_shot) {}
    HistoricalKvPrepareCode prepare_commit(const HistoricalKvBlockMeta & meta,
        HistoricalKvCommitConditions & conditions) const noexcept override { return inner_.prepare_commit(meta, conditions); }
    bool permits(const HistoricalKvBlockMeta & meta) const noexcept override { return inner_.permits(meta); }
    void on_result(const HistoricalKvCommitResult & result) noexcept override { inner_.on_result(result); }
    HistoricalKvStepResult step(const HistoricalKvSnapshot & snapshot, const HistoricalKvPressure & pressure,
        HistoricalKvWorkBudget budget, HistoricalKvDecisionBuffer & out) override {
        if (one_shot_ && !observed.empty()) return {HistoricalKvStepStatus::NoCandidate, 0, 0};
        temporary_.clear_after_ack();
        auto result = inner_.step(snapshot, pressure, budget, temporary_);
        for (auto intent : temporary_.values()) {
            check(intent.evidence && intent.evidence->witness, "fixture must produce direct evidence");
            const bool first = observed.empty();
            if (first) mutate(intent, snapshot);
            observed.push_back(intent);
            check(out.push(intent), "fixture batch capacity");
            if (first && (fault_ == Fault::BatchDelete || fault_ == Fault::Cycle)) {
                auto other = intent;
                const auto witness = *intent.evidence->witness;
                other.victim = witness.handle;
                other.expected_content_version = witness.content_version;
                if (fault_ == Fault::BatchDelete) {
                    other.reason = HistoricalKvDeleteReason::Lru;
                    other.evidence->witness.reset();
                } else {
                    other.evidence->witness = HistoricalKvWitnessRequirement{intent.victim, *intent.expected_content_version};
                }
                observed.push_back(other);
                check(out.push(other), "two-command fixture capacity");
            }
        }
        result.emitted = temporary_.values().empty() ? 0 : out.values().size();
        return result;
    }
    std::vector<HistoricalKvDeleteIntent> observed;
private:
    static HistoricalKvRedundancyPolicy::Config policy_config(size_t capacity) {
        HistoricalKvRedundancyPolicy::Config cfg;
        cfg.max_candidate_blocks = capacity;
        return cfg;
    }
    void mutate(HistoricalKvDeleteIntent & intent, const HistoricalKvSnapshot & snapshot) {
        auto & evidence = *intent.evidence;
        switch (fault_) {
        case Fault::Missing: intent.evidence.reset(); break;
        case Fault::NoWitness: evidence.witness.reset(); break;
        case Fault::Self: evidence.witness->handle = intent.victim; break;
        case Fault::Generation: ++evidence.witness->handle.generation; break;
        case Fault::Version: ++evidence.witness->content_version; break;
        case Fault::Incarnation: ++evidence.witness->handle.incarnation; break;
        case Fault::Similarity: evidence.similarity = evidence.threshold - .1; break;
        case Fault::Nonfinite: evidence.retention_score = std::numeric_limits<double>::quiet_NaN(); break;
        case Fault::Query: ++evidence.query_epoch; break;
        case Fault::Score: ++evidence.score_epoch; break;
        case Fault::Schema: ++evidence.feature_schema; break;
        case Fault::Round: ++evidence.scoring_round; break;
        case Fault::VictimVersion: intent.expected_content_version.reset(); break;
        case Fault::Alternate:
            for (const auto & meta : snapshot.blocks)
                if (meta.logical_id == 1) evidence.witness = HistoricalKvWitnessRequirement{meta.handle, meta.content_version};
            break;
        default: break;
        }
    }
    HistoricalKvRedundancyPolicy inner_;
    HistoricalKvDecisionBuffer temporary_;
    Fault fault_;
    bool one_shot_;
};

struct Fixture {
    std::atomic<unsigned> releases{0};
    std::shared_ptr<Repo> repository;
    std::shared_ptr<HistoricalKvPageFeatures> features;
    RecordingPolicy * policy;
    std::unique_ptr<HistoricalKvGc> gc;
    std::vector<HistoricalKvHandle> handles;
    Fixture(Fault fault = Fault::None, bool boundary = true, unsigned capacity = 2, bool one_shot = true)
        : repository(std::make_shared<Repo>(capacity)),
          features(std::make_shared<HistoricalKvPageFeatures>(capacity, 3)) {
        for (unsigned id = 0; id < capacity; ++id) {
            handles.push_back(append(id));
            features->publish_page(id, {1, 0, 0});
        }
        features->publish_question_scores(std::vector<double>(capacity, 0));
        auto implementation = std::make_unique<RecordingPolicy>(capacity, features, fault, one_shot);
        policy = implementation.get();
        HistoricalKvGc::Config cfg;
        cfg.mode = HistoricalKvGc::Mode::Delete;
        cfg.retire_at_decode_boundary = boundary;
        cfg.ready_low = 1; cfg.ready_high = 2;
        cfg.scan_blocks = cfg.reclaim_blocks = capacity;
        cfg.batch_blocks = fault == Fault::BatchDelete || fault == Fault::Cycle ? 2 : 1;
        gc = std::make_unique<HistoricalKvGc>(repository, std::move(implementation), cfg);
    }
    HistoricalKvHandle append(uint64_t logical) {
        auto writer = repository->begin_write(logical, logical * 64, 64, logical + 1, 16,
            [this](const HistoricalKvHandle &) { ++releases; return true; });
        check(bool(writer) && repository->publish_write(*writer), "fixture write failed");
        return writer->handle();
    }
    void prepare() {
        for (unsigned i = 0; i < 40 && policy->observed.empty(); ++i) gc->manual_tick();
        check(!policy->observed.empty(), "no scored fixture proposal");
    }
};

static void malformed_evidence_and_batch_conflicts() {
    const Fault faults[] = {Fault::Missing, Fault::NoWitness, Fault::Self, Fault::Generation, Fault::Version,
        Fault::Incarnation, Fault::Similarity, Fault::Nonfinite, Fault::Query, Fault::Score,
        Fault::Schema, Fault::Round, Fault::VictimVersion};
    for (bool boundary : {false, true}) for (Fault fault : faults) {
        Fixture fixture(fault, boundary);
        fixture.prepare();
        check(fixture.gc->commit_at_decode_boundary().committed == 0, "invalid proof committed");
        check(fixture.gc->stats().committed == 0 && fixture.gc->stats().invalid_evidence > 0,
              "invalid proof not rejected explicitly");
        for (auto handle : fixture.handles)
            check(fixture.repository->completion_state(handle) == Repo::CompletionState::Live,
                  "invalid evidence changed visibility");
    }
    for (bool boundary : {false, true}) for (Fault fault : {Fault::BatchDelete, Fault::Cycle}) {
        Fixture fixture(fault, boundary);
        fixture.prepare();
        fixture.gc->commit_at_decode_boundary();
        fixture.gc->manual_tick();
        check(fixture.repository->completion_state(fixture.handles[0]) == Repo::CompletionState::Live,
              "same-batch victim/witness conflict removed both");
        check(fixture.gc->stats().committed == (fault == Fault::Cycle ? 0u : 1u), "wrong conflict outcome");
    }
}

static void witness_retirement_reuse_and_score_replacement() {
    for (unsigned mode = 0; mode < 6; ++mode) {
        Fixture fixture;
        fixture.prepare();
        const auto intent = fixture.policy->observed.front();
        const auto witness = *intent.evidence->witness;
        check(same(intent.victim, fixture.handles[0]) && same(witness.handle, fixture.handles[1]),
              "unexpected fixture preference");
        std::optional<Repo::Lease> reader;
        if (mode <= 2) {
            if (mode == 0) reader = fixture.repository->acquire_read(witness.handle);
            check(fixture.repository->retire(witness.handle) == Repo::RetireResult::Committed, "retire witness");
            if (mode != 0) {
                check(fixture.repository->try_reclaim(witness.handle) == Repo::ReclaimResult::Complete, "release witness");
                if (mode == 2) {
                    const auto fresh = fixture.append(99);
                    check(fresh.slot == witness.handle.slot && fresh.generation != witness.handle.generation,
                          "fixture did not reuse witness slot");
                }
            }
        } else if (mode == 3) {
            const auto question = fixture.features->question_epoch();
            fixture.features->publish_output_attention(1, .7);
            check(fixture.features->begin_round(fixture.repository->incarnation()), "replace score snapshot");
            check(fixture.features->question_epoch() == question, "score test changed Query");
        } else if (mode == 4) {
            fixture.features->publish_output_attention(1, .7); // Raw update, frozen bank remains valid.
        }
        const auto result = fixture.gc->commit_at_decode_boundary();
        check(result.committed == (mode >= 4 ? 1u : 0u), "stale/valid evidence outcome");
        fixture.gc->manual_tick();
        if (mode <= 2) check(fixture.gc->stats().witness_rejected == 1, "lost witness reason");
        if (mode == 3) check(fixture.gc->stats().reevaluate == 1, "lost score epoch rejection");
    }
}

static void acknowledgement_uses_committed_witness_and_does_not_pin_forever() {
    Fixture fixture(Fault::Alternate, true, 3, false);
    fixture.prepare();
    const auto first = fixture.policy->observed.front();
    check(same(first.victim, fixture.handles[0]) && same(first.evidence->witness->handle, fixture.handles[1]),
          "alternate witness fixture");
    check(fixture.gc->commit_at_decode_boundary().committed == 1, "alternate direct witness commit");
    fixture.gc->request_capacity(2); // Allows redundancy only, not lossy fallback.
    for (unsigned i = 0; i < 24; ++i) fixture.gc->manual_tick();
    check(fixture.gc->stats().proposed == 1, "acknowledgement reconstructed a different cached witness");
    check(fixture.repository->retire(fixture.handles[1]) == Repo::RetireResult::Committed,
          "witness was permanently pinned by an old deletion");
}

static void cross_retirement_is_atomic_and_nonwaiting() {
    // Racing conditional removals may remove one block, never both.
    for (unsigned iteration = 0; iteration < 100; ++iteration) {
        auto repository = std::make_shared<Repo>(2);
        std::array<HistoricalKvHandle, 2> handles{};
        for (unsigned id = 0; id < 2; ++id) {
            auto writer = repository->begin_write(id, id * 64, 64, id + 1, 16,
                [](const HistoricalKvHandle &) { return true; });
            check(bool(writer) && repository->publish_write(*writer), "cross fixture");
            handles[id] = writer->handle();
        }
        std::atomic<unsigned> ready{0}; std::atomic<bool> go{false};
        std::array<Repo::RetireResult, 2> results{};
        const auto retire = [&](unsigned id) {
            ++ready;
            while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
            results[id] = repository->retire_with_witness(handles[id], 1, {handles[1 - id], 1});
        };
        std::thread a(retire, 0), b(retire, 1);
        while (ready != 2) std::this_thread::yield();
        go.store(true, std::memory_order_release); a.join(); b.join();
        const auto committed = unsigned(results[0] == Repo::RetireResult::Committed) +
                               unsigned(results[1] == Repo::RetireResult::Committed);
        check(committed == 1, "cross-witness race removed both or made no progress");
        for (unsigned id = 0; id < 2; ++id)
            if (results[id] == Repo::RetireResult::Retry)
                check(repository->retire_with_witness(handles[id], 1, {handles[1 - id], 1}) ==
                      Repo::RetireResult::WitnessChanged, "retry used a retired witness");
    }
    // A paused transaction cannot make a second caller wait on a mutex/spin.
    Fixture fixture;
    std::mutex mutex; std::condition_variable cv;
    bool entered = false, opened = false;
    struct Context { std::mutex & mutex; std::condition_variable & cv; bool & entered; bool & opened; } context{mutex, cv, entered, opened};
    std::thread paused([&] {
        fixture.repository->retire(fixture.handles[1], {},
            [](const HistoricalKvBlockMeta &, const void * opaque) noexcept {
                auto & state = *static_cast<Context *>(const_cast<void *>(opaque));
                std::unique_lock<std::mutex> lock(state.mutex);
                state.entered = true; state.cv.notify_all();
                state.cv.wait(lock, [&] { return state.opened; });
                return false; // No retirement, just a controlled scheduling pause.
            }, &context);
    });
    bool started;
    { std::unique_lock<std::mutex> lock(mutex); started = cv.wait_for(lock, std::chrono::seconds(5), [&] { return entered; }); }
    auto attempt = std::async(std::launch::async, [&] {
        return fixture.repository->retire_with_witness(fixture.handles[0], 1, {fixture.handles[1], 1});
    });
    const bool returned = attempt.wait_for(std::chrono::seconds(2)) == std::future_status::ready;
    { std::lock_guard<std::mutex> lock(mutex); opened = true; cv.notify_all(); }
    paused.join();
    const auto code = attempt.get();
    check(started && returned && code == Repo::RetireResult::Retry, "contended retirement waited");
    check(fixture.repository->retire_with_witness(fixture.handles[0], 1, {fixture.handles[1], 1}) ==
          Repo::RetireResult::Committed, "released retirement transaction failed");
}

int main() {
    try {
        malformed_evidence_and_batch_conflicts();
        witness_retirement_reuse_and_score_replacement();
        acknowledgement_uses_committed_witness_and_does_not_pin_forever();
        cross_retirement_is_atomic_and_nonwaiting();
        std::cout << "PASS score evidence: malformed/context, batch conflicts, witness lifecycle, frozen scores, exact ACK, 100 cross-retirement races and nonwaiting Retry\n";
        std::cout << "intent_bytes=" << sizeof(HistoricalKvDeleteIntent) << " conditions_bytes=" << sizeof(HistoricalKvCommitConditions)
                  << " max_epoch_guards=" << HistoricalKvCommitConditions::kEpochGuards << '\n';
    } catch (const std::exception & error) {
        std::cerr << "FAIL score evidence: " << error.what() << '\n';
        return 1;
    }
}
