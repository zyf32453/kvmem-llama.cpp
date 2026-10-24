#include "kvmem/historical_kv_redundancy.hpp"
#include "kvmem/thinking_span_tracker.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace kvmem;
static void check(bool value, const char * message) {
    if (!value) throw std::runtime_error(message);
}

struct Fixture {
    std::shared_ptr<HistoricalKvRepository> repo = std::make_shared<HistoricalKvRepository>(2);
    std::shared_ptr<HistoricalKvRawStore> store;
    std::shared_ptr<HistoricalKvMeanKeySource> source;
    std::vector<HistoricalKvHandle> handles;

    Fixture() {
        HistoricalKvRawStoreConfig config;
        config.n_layer = 2;
        config.max_tokens_per_slot = 1;
        config.packed_k_bytes_per_token = config.packed_v_bytes_per_token = 1;
        config.mean_k_elements = 2;
        config.attention_layers = {0, 1};
        store = std::make_shared<HistoricalKvRawStore>(config, 2);
        for (unsigned i = 0; i < 2; ++i) {
            auto writer = repo->begin_write(i, i, 1, 1, 20, [this](const HistoricalKvHandle & h) {
                return store->release(*repo, h).code == HistoricalKvRawStore::ReleaseCode::Released;
            });
            check(bool(writer), "fixture writer missing");
            const auto h = writer->handle();
            handles.push_back(h);
            check(store->reserve(*writer, h, i, 1), "fixture reserve failed");
            for (unsigned l = 0; l < 2; ++l)
                check(store->write_layer(*writer, h, l, {1}, {2},
                    i ? std::vector<float>{1, 0} : std::vector<float>{0, 1}, 1), "fixture write failed");
            check(store->publish_complete(*repo, *writer, h), "fixture publish failed");
        }
        source = std::make_shared<HistoricalKvMeanKeySource>(repo, store, 2, 1, 2);
    }
    ~Fixture() {
        repo->close();
        for (const auto & block : repo->snapshot()) {
            repo->retire(block.handle);
            repo->try_reclaim(block.handle);
        }
    }
    void publish(const std::vector<float> & q, uint32_t batch, uint64_t origin = 0) {
        const uint32_t count = uint32_t(q.size() / 4);
        for (uint32_t start = 0; start < count; start += batch) {
            const uint32_t n = std::min(batch, count - start);
            for (unsigned l = 0; l < 2; ++l)
                source->publish_query(l, origin + start + n, q.data() + start * 4, n);
        }
    }
    double score(size_t slot, size_t * yields = nullptr) {
        std::vector<float> descriptor(source->dimensions());
        double logit = 0;
        for (size_t calls = 0; calls < 10000; ++calls) {
            const auto before = source->question_score_evaluations();
            const auto result = source->read_bounded(*repo->inspect(handles[slot]), descriptor.data(), logit);
            check(source->question_score_evaluations() - before <= HistoricalKvMeanKeySource::kQuestionSegmentsPerRead,
                  "a feature read exceeded its question work budget");
            if (result == HistoricalKvRedundancySource::ReadResult::Ready) return logit;
            check(result == HistoricalKvRedundancySource::ReadResult::Yielded, "question feature unavailable");
            if (yields) ++*yields;
        }
        throw std::runtime_error("question score never completed");
    }
};

static void full_span_and_cache() {
    Fixture f;
    const double expected = 1 / std::sqrt(2.0);
    // A single salient token has the same score at the beginning, middle,
    // and end. None of these scores may decay with the length of the suffix.
    for (uint32_t hot : {0u, 2048u, 4095u}) {
        for (uint32_t batch : {1u, 17u, 512u, 4096u}) {
            std::vector<float> q(4096 * 4, 0.f);
            q[hot * 4 + 1] = q[hot * 4 + 3] = 32.f;
            f.source->begin_question();
            f.publish(q, batch);
            check(!f.source->begin_round(f.repo->incarnation()), "partial prefill was exposed to GC");
            check(f.source->finish_question(4096), "full question rejected");
            check(f.source->question_tokens() == 4096 && f.source->question_segments() == 128,
                  "question lost a prefix or changed with microbatch partition");
            check(f.source->begin_round(f.repo->incarnation()), "finished question unavailable");
            size_t yields = 0;
            check(std::abs(f.score(0, &yields) - expected) < 1e-6, "early/middle/tail constraint lost");
            check(yields == 15, "long question did not yield at its eight-segment work bound");
            const auto evaluations = f.source->question_score_evaluations();
            check(f.source->begin_round(f.repo->incarnation()), "next round unavailable");
            check(std::abs(f.score(0) - expected) < 1e-6 &&
                  f.source->question_score_evaluations() == evaluations, "unchanged question was rescored");
        }
    }
    f.source->begin_question();
    f.publish(std::vector<float>(33 * 4, 0.f), 7);
    check(f.source->finish_question(33), "partial final segment rejected");
    check(f.source->question_segments() == 2 && f.source->begin_round(f.repo->incarnation()), "partial tail dropped");
    check(f.score(0) == 0, "new question reused the previous question's cached importance");

    // Opposite constraints in different segments must not cancel in a single
    // whole-question mean. The maximum is across coherent complete segments.
    std::vector<float> opposed(64 * 4, 0.f);
    for (unsigned i = 0; i < 64; ++i) opposed[i * 4 + 1] = opposed[i * 4 + 3] = i < 32 ? 1.f : -1.f;
    f.source->begin_question();
    f.publish(opposed, 11);
    check(f.source->finish_question(64) && f.source->begin_round(f.repo->incarnation()), "opposed question rejected");
    check(std::abs(f.score(0) - expected) < 1e-6, "different query segments cancelled each other");

    const auto cached_evaluations = f.source->question_score_evaluations();
    f.source->begin_question();
    check(!f.source->reuse_question(65), "different-length question accepted as a cached question");
    check(f.source->reuse_question(64) && f.source->finish_question(64) &&
          f.source->begin_round(f.repo->incarnation()), "exact cached question could not reuse full coverage");
    check(std::abs(f.score(0) - expected) < 1e-6 &&
          f.source->question_score_evaluations() == cached_evaluations, "cached question recomputed its segment scores");

    // Generated Query can change while the full original question remains.
    const float generated[] = {2, 0, 2, 0};
    for (unsigned t = 0; t < 8; ++t)
        for (unsigned l = 0; l < 2; ++l) f.source->publish_observation(l, 65 + t, generated, 1);
    check(f.source->begin_round(f.repo->incarnation()), "generated observation missing");
    check(std::abs(f.score(0) - expected) < 1e-6 && std::abs(f.score(1) - 2 * expected) < 1e-6,
          "generated focus erased input coverage or failed to update");
}

static void single_token_question_and_long_decode() {
    Fixture f;
    const double expected = 1 / std::sqrt(2.0);
    // The adapter uses its explicit input/decode boundary, not batch size.
    // A one-token input at a nonzero position must remain a complete question.
    f.source->begin_question();
    f.publish({0, 1, 0, 1}, 1, 99);
    check(f.source->finish_question(1) && f.source->begin_round(f.repo->incarnation()),
          "single-token question was treated as generated observation");
    const auto question_bytes = f.source->question_bytes();
    const auto epoch = f.source->question_epoch();
    check(f.source->question_tokens() == 1 && f.source->question_segments() == 1 &&
          f.source->observation_samples() == 0 && f.source->fast_samples() == 0,
          "single-token input contaminated generated windows");
    check(std::abs(f.score(0) - expected) < 1e-6 && f.score(1) == 0,
          "single-token question importance is incorrect");
    const auto cached_evaluations = f.source->question_score_evaluations();

    // No thinking-range object or candidate worker is needed by this source.
    // This checks its phase/window contract; the model benchmark separately
    // verifies that the adapter actually captures unmarked answer/no-thinking Q.
    uint64_t position = 100;
    const auto decode = [&](const float * row, unsigned count) {
        for (unsigned t = 0; t < count; ++t) {
            ++position;
            for (unsigned l = 0; l < 2; ++l) f.source->publish_observation(l, position, row, 1);
        }
    };
    const float x[] = {4, 0, 4, 0};
    const float zero[] = {0, 0, 0, 0};
    decode(x, 7);
    check(f.source->fast_samples() == 0 && f.source->observation_samples() == 0,
          "incomplete generated windows were published");
    decode(x, 1);
    check(f.source->fast_samples() > 0 && f.source->observation_samples() == 0 &&
          f.source->begin_round(f.repo->incarnation()) &&
          std::abs(f.score(1) - 4 * expected) < 1e-6,
          "generated focus was delayed until the slow window");
    decode(x, 24);
    check(f.source->observation_samples() > 0, "slow generated window did not publish");
    f.source->flush_observation(); // a thinking-close notification is not an observation stop
    const auto slow = f.source->observation_samples();
    const auto fast = f.source->fast_samples();
    decode(zero, 512);
    check(f.source->observation_samples() > slow && f.source->fast_samples() > fast &&
          f.source->begin_round(f.repo->incarnation()) && f.score(1) == 0,
          "later answer observations stopped or old generated peaks grew without bound");
    check(std::abs(f.score(0) - expected) < 1e-6 && f.source->question_epoch() == epoch &&
          f.source->question_tokens() == 1 && f.source->question_segments() == 1 &&
          f.source->question_bytes() == question_bytes &&
          f.source->question_score_evaluations() == cached_evaluations,
          "long decoding changed the full question bank or repeated its dot products");

    decode(x, 8);
    check(f.source->begin_round(f.repo->incarnation()) && std::abs(f.score(1) - 4 * expected) < 1e-6,
          "generated Query did not update after a closed thinking span");
    f.source->begin_question();
    check(f.source->question_epoch() != epoch && !f.source->begin_round(f.repo->incarnation()),
          "next prefill exposed the previous request's generated Query");
    f.publish({0, 0, 0, 0}, 1, position);
    check(f.source->finish_question(1) && f.source->begin_round(f.repo->incarnation()) &&
          f.score(0) == 0 && f.score(1) == 0,
          "previous question or generated Query leaked into the next question");
}

static void closed_thinking_event_preserves_slow_cadence() {
    Fixture f;
    f.source->begin_question();
    f.publish({0, 0, 0, 0}, 1);
    check(f.source->finish_question(1), "closed-event question setup failed");
    ThinkingSpanTracker tracker({11, 12}, {{13, 14}}, true);
    tracker.observe_prompt({11, 12, 7});
    const auto publish = [&] {
        ThinkingSpanTracker::SpanUpdate update;
        if (tracker.take_update(update) && update.closed) f.source->flush_observation();
    };
    const float generated[] = {1, 0, 1, 0};
    uint64_t position = 3;
    const auto decode = [&](int32_t token) {
        tracker.observe_generated(token, int(position));
        // Match the server ordering: publish markers before decoding this
        // token. Its actual GPU Q is harvested afterwards, at this position.
        publish();
        for (unsigned layer = 0; layer < 2; ++layer)
            f.source->publish_observation(layer, position + 1, generated, 1);
        ++position;
    };
    publish();
    decode(13);
    decode(14);
    const auto slow_at_close = f.source->observation_samples();
    check(slow_at_close == 1, "thinking close did not flush its partial observation once");
    for (unsigned i = 0; i < 256; ++i) decode(7);
    check(f.source->observation_samples() == slow_at_close + 8,
          "repeated closed notification turned slow observations into per-token snapshots");
    check(f.source->fast_samples() > 0 && f.source->question_tokens() == 1,
          "answer observations stopped or changed question coverage");
}

static void fail_closed_and_gaps() {
    Fixture f;
    std::vector<float> q(35 * 4, 1.f);
    f.source->begin_question();
    f.publish(q, 35);
    f.publish(q, 7, 100); // deliberate separate Query spans retain both parts
    check(f.source->finish_question(70) && f.source->question_segments() == 4,
          "a disjoint span replaced earlier question segments");

    f.source->begin_question();
    f.source->publish_query(0, 35, q.data(), 35);
    f.source->publish_query(1, 35, q.data() + 4, 34);
    check(!f.source->finish_question(35) && !f.source->begin_round(f.repo->incarnation()),
          "inconsistent layer coverage was published");
    f.source->begin_question();
    f.publish(q, 35, 100);
    check(!f.source->finish_question(36), "a missing token in all layers escaped the expected-length check");

    f.source->set_question_byte_budget(1);
    f.source->begin_question();
    f.publish(q, 35);
    check(!f.source->finish_question(35) && !f.source->begin_round(f.repo->incarnation()),
          "overflow silently became a truncated question");
    f.source->set_question_byte_budget(HistoricalKvMeanKeySource::kQuestionMaxBytes);
    f.source->begin_question();
    q[0] = std::numeric_limits<float>::quiet_NaN();
    f.publish(q, 35);
    check(!f.source->finish_question(35), "nonfinite question published");
    q[0] = 1.f;
    f.source->begin_question();
    f.publish(q, 3);
    check(f.source->finish_question(35) && f.source->begin_round(f.repo->incarnation()),
          "new valid question did not recover from a rejected question");
    check(f.source->invalid_questions() == 4, "invalid question accounting lost a failure");
    const auto epoch = f.source->question_epoch();
    check(!f.source->finish_question(36) && !f.source->begin_round(f.repo->incarnation()) &&
          f.source->question_epoch() != epoch, "repeated finalization bypassed the coverage contract");
}

static void policy_yields_and_reset() {
    Fixture f;
    std::vector<float> q(4096 * 4, 0.f);
    q[1] = q[3] = 32.f; // the unique old fact is needed at the start
    for (unsigned i = 4064; i < 4096; ++i) q[i * 4] = q[i * 4 + 2] = 0.5f;
    f.source->begin_question();
    f.publish(q, 512);
    check(f.source->finish_question(4096), "policy question rejected");
    HistoricalKvRedundancyPolicy policy(2, f.source, {0.1, 0.9, 2, 2});
    HistoricalKvSnapshot snapshot{1, f.repo->incarnation(), f.repo->snapshot()};
    HistoricalKvDecisionBuffer out(2);
    bool completed = false;
    for (unsigned tick = 0; tick < 100; ++tick) {
        const auto before = f.source->question_score_evaluations();
        const auto result = policy.step(snapshot, {0, 1, 2, 0}, {1, 1}, out);
        check(f.source->question_score_evaluations() - before <= 8, "policy bypassed bounded scoring");
        if (result.status != HistoricalKvStepStatus::Yielded) { completed = true; break; }
        check(out.values().empty(), "policy emitted a deletion from incomplete question scores");
    }
    check(completed && out.values().size() == 1 && out.values()[0].victim.slot == f.handles[1].slot,
          "policy lost the fact required only by the question prefix");
    check(policy.metrics().missing_features == 0, "pending question work counted as a missing feature");

    // A new request invalidates both an old-round score and delayed decisions.
    f.source->begin_question();
    check(!policy.permits(*f.repo->inspect(f.handles[1])), "old-query retirement survived a question change");
    f.publish(q, 512);
    check(f.source->finish_question(4096), "replacement question rejected");
    out.clear_after_ack();
    ++snapshot.round_id;
    check(policy.step(snapshot, {0, 1, 2, 0}, {1, 1}, out).status == HistoricalKvStepStatus::Yielded,
          "long question was not sliced");
    f.source->begin_question();
    check(policy.step(snapshot, {0, 1, 2, 0}, {1, 1}, out).status == HistoricalKvStepStatus::NoCandidate &&
          out.values().empty(), "in-flight scoring ignored a question change");
}

static void immutable_question_snapshot() {
    Fixture f;
    f.source->begin_question();
    f.publish(std::vector<float>(32 * 4, 1.f), 32);
    check(f.source->finish_question(32) && f.source->begin_round(f.repo->incarnation()), "snapshot setup failed");
    const double old = f.score(0);
    std::atomic<bool> started{false};
    std::thread producer([&] {
        f.source->begin_question();
        started = true;
        f.publish(std::vector<float>(4096 * 4, 0.f), 7);
        f.source->finish_question(4096);
    });
    while (!started.load()) std::this_thread::yield();
    bool stable = true;
    for (unsigned i = 0; i < 100; ++i) stable &= f.score(0) == old;
    producer.join();
    check(stable, "prefill mutated the worker's immutable question snapshot");
    check(f.source->begin_round(f.repo->incarnation()) && f.score(0) == 0, "replacement snapshot was not published");
}

static void latest_generated_query_before_ranking() {
    Fixture f;
    std::vector<float> q(4096 * 4, 0.f);
    q[1] = q[3] = 32.f; // initial question favors the Y block
    f.source->begin_question();
    f.publish(q, 512);
    check(f.source->finish_question(4096), "refresh setup failed");
    HistoricalKvRedundancyPolicy policy(2, f.source, {0.1, 0.9, 2, 2});
    HistoricalKvSnapshot snapshot{1, f.repo->incarnation(), f.repo->snapshot()};
    HistoricalKvDecisionBuffer out(2);
    const auto before = f.source->question_score_evaluations();
    bool completed = false;
    for (unsigned tick = 0; tick < 100; ++tick) {
        if (tick == 5) {
            // Focus changes while full-question scoring is still yielding.
            // The eventual retirement must observe it in this same round.
            const float x[] = {5, 0, 5, 0};
            for (unsigned t = 0; t < 8; ++t)
                for (unsigned l = 0; l < 2; ++l) f.source->publish_observation(l, 4097 + t, x, 1);
        }
        if (policy.step(snapshot, {0, 1, 2, 0}, {1, 1}, out).status != HistoricalKvStepStatus::Yielded) {
            completed = true;
            break;
        }
        check(out.values().empty(), "ranking used an incomplete importance refresh");
    }
    check(completed && out.values().size() == 1 && out.values()[0].victim.slot == f.handles[0].slot,
          "long scoring round ignored the newly generated Query before retirement");
    check(f.source->question_score_evaluations() - before == 2 * 128,
          "observation refresh recomputed full-question dot products");
}

// Optional deterministic CPU cost probe. This does not load a language model
// and its timings must never be reported as model decode p95/throughput.
static void performance_probe() {
    HistoricalKvRawStoreConfig config;
    config.n_layer = 4;
    config.max_tokens_per_slot = 1;
    config.packed_k_bytes_per_token = config.packed_v_bytes_per_token = 1;
    config.mean_k_elements = 128 * 8;
    config.attention_layers = {0, 1, 2, 3};
    auto repo = std::make_shared<HistoricalKvRepository>(1);
    auto store = std::make_shared<HistoricalKvRawStore>(config, 1);
    auto writer = repo->begin_write(0, 0, 1, 1, 20, [&](const HistoricalKvHandle & h) {
        return store->release(*repo, h).code == HistoricalKvRawStore::ReleaseCode::Released;
    });
    check(bool(writer), "benchmark writer missing");
    const auto h = writer->handle();
    check(store->reserve(*writer, h, 0, 1), "benchmark reserve failed");
    for (unsigned l = 0; l < 4; ++l)
        check(store->write_layer(*writer, h, l, {1}, {2}, std::vector<float>(1024, 0.125f), 1), "benchmark write failed");
    check(store->publish_complete(*repo, *writer, h), "benchmark publish failed");
    writer->reset();
    HistoricalKvMeanKeySource source(repo, store, 128, 8, 32);
    const auto ms = [](auto start) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    };
    std::vector<float> batch(256 * 4096, 0.25f), descriptor(source.dimensions());
    for (unsigned tokens : {512u, 4096u, 16384u}) {
        auto start = std::chrono::steady_clock::now();
        source.begin_question();
        for (unsigned pos = 0; pos < tokens; pos += 256)
            for (unsigned l = 0; l < 4; ++l) source.publish_query(l, pos + 256, batch.data(), 256);
        check(source.finish_question(tokens) && source.begin_round(repo->incarnation()), "benchmark question rejected");
        const double capture_ms = ms(start);
        size_t slices = 0;
        double max_slice_ms = 0, logit = 0;
        start = std::chrono::steady_clock::now();
        for (;;) {
            const auto slice_start = std::chrono::steady_clock::now();
            const auto before = source.question_score_evaluations();
            const auto result = source.read_bounded(*repo->inspect(h), descriptor.data(), logit);
            max_slice_ms = std::max(max_slice_ms, ms(slice_start));
            ++slices;
            check(source.question_score_evaluations() - before <= source.kQuestionSegmentsPerRead,
                  "benchmark exceeded per-slice work bound");
            if (result == HistoricalKvRedundancySource::ReadResult::Ready) break;
            check(result == HistoricalKvRedundancySource::ReadResult::Yielded, "benchmark feature missing");
        }
        const double first_score_ms = ms(start);
        const auto evaluations = source.question_score_evaluations();
        start = std::chrono::steady_clock::now();
        for (unsigned repeat = 0; repeat < 100; ++repeat)
            check(source.read_bounded(*repo->inspect(h), descriptor.data(), logit) == HistoricalKvRedundancySource::ReadResult::Ready,
                  "cached score yielded");
        const double cached_score_ms = ms(start) / 100;
        check(source.question_score_evaluations() == evaluations, "cache repeated question dot products");
        std::cout << "{\"tokens\":" << tokens << ",\"layers\":4,\"head_dim\":128,\"kv_heads\":8,\"query_heads\":32"
                  << ",\"segments\":" << source.question_segments() << ",\"question_bytes\":" << source.question_bytes()
                  << ",\"capture_ms\":" << capture_ms << ",\"first_block_score_ms\":" << first_score_ms
                  << ",\"slices\":" << slices << ",\"max_slice_ms\":" << max_slice_ms
                  << ",\"cached_block_score_ms\":" << cached_score_ms << "}\n";
    }
    repo->close(); repo->retire(h); repo->try_reclaim(h);
}

int main(int argc, char ** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--bench") { performance_probe(); return 0; }
        const auto start = std::chrono::steady_clock::now();
        full_span_and_cache();
        single_token_question_and_long_decode();
        closed_thinking_event_preserves_slow_cadence();
        fail_closed_and_gaps();
        policy_yields_and_reset();
        immutable_question_snapshot();
        latest_generated_query_before_ranking();
        std::cout << "full-question coverage, bounded scoring and cache: passed ("
                  << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count()
                  << " ms; synthetic CPU test, not model decode performance)\n";
    } catch (const std::exception & e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
