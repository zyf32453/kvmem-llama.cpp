#include "kvmem/historical_kv_auxiliary.hpp"
#include "kvmem/historical_kv_gc.hpp"
#include "kvmem/historical_kv_page_features.hpp"
#include "kvmem/memory_contract.hpp"

#include <algorithm>
#include <atomic>
#include <iostream>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>

using namespace kvmem;
using Kind = HistoricalKvAuxKind;
using Budget = HistoricalKvAuxBudget;
template<class T> using Vector = HistoricalKvAuxVector<T>;
template<class T> using Alloc = HistoricalKvAuxAllocator<T>;
static void check(bool condition, const char * message) { if (!condition) throw std::runtime_error(message); }
template<class F> static void denied(F operation) {
    try { operation(); }
    catch (const MemoryError & error) { check(error.code() == MemoryErrorCode::BudgetExceeded, "wrong rejection code"); return; }
    throw std::runtime_error("budget exhaustion silently accepted");
}
template<class F> static void allocation_failed(F operation) {
    try { operation(); }
    catch (const std::bad_alloc &) { return; }
    throw std::runtime_error("injected upstream failure not observed");
}

// Observes actual upstream requests independently of the budget. Observer
// bookkeeping is test infrastructure, not part of the component under test.
class RecordingResource final : public std::pmr::memory_resource {
public:
    bool fail_next = false;
    size_t fail_request = SIZE_MAX;
    size_t charged = 0, peak = 0, requests = 0;
    std::map<void *, std::pair<size_t, size_t>> live;
private:
    void * do_allocate(size_t bytes, size_t alignment) override {
        ++requests;
        if (fail_next || requests == fail_request) { fail_next = false; throw std::bad_alloc(); }
        auto * pointer = std::pmr::new_delete_resource()->allocate(bytes, alignment);
        try { live.emplace(pointer, std::make_pair(bytes, alignment)); }
        catch (...) { std::pmr::new_delete_resource()->deallocate(pointer, bytes, alignment); throw; }
        charged += bytes; peak = std::max(peak, charged);
        return pointer;
    }
    void do_deallocate(void * pointer, size_t bytes, size_t alignment) override {
        const auto found = live.find(pointer);
        if (found == live.end() || found->second != std::make_pair(bytes, alignment)) std::terminate();
        live.erase(found); charged -= bytes;
        std::pmr::new_delete_resource()->deallocate(pointer, bytes, alignment);
    }
    bool do_is_equal(const memory_resource & other) const noexcept override { return this == &other; }
};

static void reallocation_and_owner_lifetime() {
    RecordingResource upstream;
    auto budget = std::make_shared<Budget>(64, &upstream);
    {
        Vector<uint8_t> data(32, 7, Alloc<uint8_t>(budget, Kind::Features));
        const auto before = upstream.requests;
        denied([&] { data.reserve(64); }); // Old 32 + new 64 must coexist.
        check(upstream.requests == before && data.size() == 32 && data.capacity() == 32 && data[31] == 7,
              "failed reallocation changed data or reached upstream");
        check(budget->stats().charged == 32 && budget->stats().denied == 1, "failed reallocation refunded old storage");
        data.clear();
        check(budget->stats().charged == 32, "clear pretended capacity was released");
        Vector<uint8_t> replacement(32, 9, Alloc<uint8_t>(budget, Kind::Policy));
        check(budget->stats().charged == 64 && upstream.charged == 64, "cross-component account escaped");
        denied([&] { replacement.push_back(1); });
    }
    check(budget->stats().charged == 0 && upstream.live.empty() && budget->stats().peak == 64, "destructor refund");
    {
        Vector<uint8_t> full(64, 1, Alloc<uint8_t>(budget, Kind::Features));
        check(upstream.charged == 64, "released credit not reusable");
    }
    std::weak_ptr<Budget> weak = budget;
    auto retained = std::make_unique<Vector<uint8_t>>(16, 1, Alloc<uint8_t>(budget, Kind::Ranges));
    budget.reset();
    check(!weak.expired() && weak.lock()->stats().charged == 16, "old owner refunded or dropped account early");
    retained.reset();
    check(weak.expired() && upstream.live.empty(), "allocator account lifetime leak");
}

static void upstream_failure_copy_swap_alignment_overflow() {
    RecordingResource upstream;
    auto a = std::make_shared<Budget>(4096, &upstream), b = std::make_shared<Budget>(4096, &upstream);
    {
        Vector<uint64_t> first(2, 3, Alloc<uint64_t>(a, Kind::Features));
        const auto old_charge = a->stats().charged;
        upstream.fail_next = true;
        allocation_failed([&] { first.reserve(16); });
        check(a->stats().charged == old_charge && first.size() == 2 && first[1] == 3, "upstream failure lost old allocation");
        Vector<uint64_t> second(first); // A real copy charges twice.
        check(a->stats().charged == 2 * old_charge, "copy bypassed accounting");
        Vector<uint64_t> third(4, 8, Alloc<uint64_t>(b, Kind::Policy));
        const auto b_charge = b->stats().charged;
        second.swap(third);
        check(second[0] == 8 && second.get_allocator() == Alloc<uint64_t>(b, Kind::Policy), "swap lost allocator owner");
        third = std::move(second);
        check(a->stats().charged == old_charge && b->stats().charged == b_charge, "move refunded wrong account");
        first = third; // Copy assignment retains the target's account.
        check(first.get_allocator() == Alloc<uint64_t>(a, Kind::Features) && first[0] == 8,
              "copy assignment moved account");
    }
    check(a->stats().charged == 0 && b->stats().charged == 0 && upstream.charged == 0, "copy/move/swap leak");
    {
        struct alignas(128) Aligned { uint8_t data[128]; };
        Vector<Aligned> aligned(1, Alloc<Aligned>(a, Kind::Metadata));
        check(reinterpret_cast<uintptr_t>(aligned.data()) % alignof(Aligned) == 0 && a->stats().charged == sizeof(Aligned),
              "over-aligned allocation mischarged");
    }
    const auto requests = upstream.requests;
    try { Alloc<uint64_t>(a, Kind::Policy).allocate(SIZE_MAX); throw std::runtime_error("overflow accepted"); }
    catch (const std::bad_array_new_length &) {}
    check(upstream.requests == requests && a->stats().charged == 0, "overflow reached allocation");
}

static void features_ranges_and_failed_publication() {
    RecordingResource upstream;
    auto budget = std::make_shared<Budget>(1 << 20, &upstream);
    {
        HistoricalKvPageFeatures features(4, 3, budget);
        const auto initial = budget->stats().charged;
        upstream.fail_next = true;
        allocation_failed([&] { features.publish_page(0, {1, 0, 0}); });
        check(!features.has_page(0) && budget->stats().charged == initial, "failed nested page publication leaked");
        features.publish_page(0, {1, 0, 0});
        check(budget->stats().charged == initial + 3 * sizeof(float), "nested descriptor bypassed account");
        const auto query = features.question_epoch();
        const auto charge = budget->stats().charged;
        upstream.fail_next = true;
        allocation_failed([&] { features.publish_question_scores({1, 2}); });
        check(features.question_epoch() == query && budget->stats().charged == charge, "failed Query publication changed epoch");
        features.publish_question_scores({1, 2});
        const auto before_frozen = budget->stats().charged;
        upstream.fail_next = true;
        allocation_failed([&] { features.begin_round(1); });
        check(features.score_epoch() == 0 && budget->stats().charged == before_frozen, "failed frozen bank changed epoch");
        check(features.begin_round(1) && features.score_epoch() == 1, "frozen retry failed");
        try { features.publish_question_scores(std::vector<double>(5, 0)); throw std::runtime_error("oversized Query accepted"); }
        catch (const std::invalid_argument &) {}
        const auto retained = budget->stats().charged;
        features.truncate_to(0, 64);
        features.clear_question();
        check(budget->stats().charged == retained, "truncate counted reserved capacity as free");
        HistoricalKvThinkingRanges ranges(budget);
        ranges.note(0, 64);
        HistoricalKvCommitConditions old;
        check(ranges.prepare_commit(0, 32, 1, old), "range fixture");
        HistoricalKvCommitConditions future;
        check(ranges.prepare_commit(128, 32, 1u << 2, future), "future range fixture");
        upstream.fail_next = true;
        allocation_failed([&] { ranges.note(128, 192); });
        HistoricalKvBlockMeta meta;
        check(ranges.covers(0, 32) && !ranges.contains(128) && old.matches(meta) && future.matches(meta),
              "failed range publication changed state or revoked the growth certificate");
        ranges.note(128, 192);
        check(budget->stats().by_kind[static_cast<size_t>(Kind::Ranges)] > 0, "range accounting missing");
        check(budget->stats().charged == upstream.charged, "component allocation escaped observer");
    }
    check(budget->stats().charged == 0 && upstream.live.empty(), "feature/range lifetime leak");
    // Partial constructors must return every prior allocation when exhausted.
    const auto requests = upstream.requests;
    const auto outer_bytes = 8 * sizeof(Vector<float>);
    auto small = std::make_shared<Budget>(outer_bytes, &upstream);
    denied([&] { HistoricalKvPageFeatures oversized(8, 4, small); });
    check(small->stats().charged == 0 && small->stats().peak == outer_bytes &&
          upstream.requests > requests && upstream.live.empty(), "partial constructor leak");
}

static std::shared_ptr<HistoricalKvPageFeatures> make_features(const std::shared_ptr<Budget> & budget, uint32_t capacity) {
    auto source = std::make_shared<HistoricalKvPageFeatures>(capacity, 3, budget);
    for (uint32_t i = 0; i < capacity; ++i) source->publish_page(i, {1, 0, 0});
    source->publish_question_scores(std::vector<double>(capacity, 0));
    check(source->begin_round(1), "freeze fixture");
    return source;
}
static std::unique_ptr<HistoricalKvRedundancyPolicy> make_policy(const std::shared_ptr<Budget> & budget,
    const std::shared_ptr<HistoricalKvPageFeatures> & source, uint32_t capacity) {
    HistoricalKvRedundancyPolicy::Config config;
    config.auxiliary = budget;
    config.max_candidate_blocks = capacity;
    return std::make_unique<HistoricalKvRedundancyPolicy>(capacity, source, config);
}

static void real_policy_exhaustion_keeps_reclaimer_alive() {
    constexpr uint32_t capacity = 8;
    uint64_t setup_bytes;
    {
        auto probe = std::make_shared<Budget>(1 << 20);
        auto source = make_features(probe, capacity);
        auto policy = make_policy(probe, source, capacity);
        setup_bytes = probe->stats().charged;
    }
    RecordingResource upstream;
    auto budget = std::make_shared<Budget>(setup_bytes, &upstream);
    bool allow_release = false;
    unsigned released = 0;
    {
        auto source = make_features(budget, capacity);
        auto policy = make_policy(budget, source, capacity);
        auto repository = std::make_shared<HistoricalKvRepository>(capacity);
        for (uint32_t id = 0; id < capacity; ++id) {
            auto writer = repository->begin_write(id, id * 64, 64, id + 1, 16,
                [&](const HistoricalKvHandle &) { if (!allow_release) return false; ++released; return true; });
            check(bool(writer) && repository->publish_write(*writer), "repository fixture");
            if (id == capacity - 1)
                check(repository->retire(writer->handle()) == HistoricalKvRepository::RetireResult::Committed, "retire fixture");
        }
        HistoricalKvGc::Config config;
        config.mode = HistoricalKvGc::Mode::Delete;
        config.retire_at_decode_boundary = true;
        config.ready_low = 1; config.ready_high = 2;
        config.scan_blocks = config.reclaim_blocks = capacity; config.batch_blocks = 1;
        HistoricalKvGc gc(repository, std::move(policy), config);
        for (unsigned tick = 0; tick < 30 && !gc.stats().policy_failures; ++tick) gc.manual_tick();
        check(gc.stats().policy_failures == 1 && budget->stats().denied == 1,
              "dense-policy allocation did not fail closed");
        check(gc.stats().committed == 0 && gc.commit_at_decode_boundary().committed == 0,
              "partial failed round published a deletion");
        check(upstream.charged == budget->stats().charged && budget->stats().peak <= setup_bytes,
              "policy neighbors escaped cap");
        allow_release = true;
        gc.manual_tick();
        check(released == 1 && gc.stats().reclaimed == 1 && !gc.stats().worker_failures,
              "policy exhaustion stopped old resource reclamation");
    }
    check(budget->stats().charged == 0 && upstream.live.empty(), "failed policy leaked account");
    std::cout << "policy_setup_bytes=" << setup_bytes << '\n';
}

static void concurrent_shared_cap() {
    auto budget = std::make_shared<Budget>(64);
    std::atomic<unsigned> rejections{0}, failures{0};
    {
        Vector<uint8_t> held(64, 1, Alloc<uint8_t>(budget, Kind::Features));
        std::vector<std::thread> threads;
        for (unsigned t = 0; t < 4; ++t) threads.emplace_back([&] {
            for (unsigned n = 0; n < 100; ++n) {
                try { Vector<uint8_t> data(8, 0, Alloc<uint8_t>(budget, Kind::Policy)); ++failures; }
                catch (const MemoryError & error) { if (error.code() == MemoryErrorCode::BudgetExceeded) ++rejections; else ++failures; }
            }
        });
        for (auto & thread : threads) thread.join();
        check(rejections == 400 && failures == 0 && budget->stats().charged == 64, "shared cap bypassed");
    }
    std::vector<std::thread> threads;
    for (unsigned t = 0; t < 4; ++t) threads.emplace_back([&] {
        for (unsigned n = 0; n < 500; ++n) {
            Vector<uint8_t> data(8, 0, Alloc<uint8_t>(budget, Kind::Policy));
            if (budget->stats().charged > 64) ++failures;
        }
    });
    for (auto & thread : threads) thread.join();
    check(failures == 0 && budget->stats().charged == 0 && budget->stats().peak == 64,
          "concurrent charge/refund accounting failed");
}

static void allocation_failure_matrix() {
    size_t normal_requests = 0;
    const auto run = [&](size_t fail_request) {
        RecordingResource upstream;
        upstream.fail_request = fail_request;
        auto budget = std::make_shared<Budget>(1 << 20, &upstream);
        bool producer_failed = false;
        uint64_t committed = 0, policy_failed = 0;
        try {
            auto source = make_features(budget, 8);
            auto policy = make_policy(budget, source, 8);
            auto repository = std::make_shared<HistoricalKvRepository>(8);
            for (uint32_t id = 0; id < 8; ++id) {
                auto writer = repository->begin_write(id, id * 64, 64, id + 1, 16,
                    [](const HistoricalKvHandle &) { return true; });
                check(bool(writer) && repository->publish_write(*writer), "matrix repository fixture");
            }
            HistoricalKvGc::Config config;
            config.mode = HistoricalKvGc::Mode::Delete;
            config.retire_at_decode_boundary = true;
            config.ready_low = 1; config.ready_high = 2;
            config.scan_blocks = config.reclaim_blocks = 8; config.batch_blocks = 2;
            HistoricalKvGc gc(repository, std::move(policy), config);
            for (unsigned tick = 0; tick < 100 && !gc.stats().proposed && !gc.stats().policy_failures; ++tick)
                gc.manual_tick();
            committed = gc.commit_at_decode_boundary().committed;
            policy_failed = gc.stats().policy_failures;
            check(!gc.stats().worker_failures, "allocation failure escaped policy boundary");
            check(upstream.charged == budget->stats().charged, "matrix live memory miscount");
        } catch (const std::bad_alloc &) { producer_failed = true; }
        check(budget->stats().charged == 0 && upstream.live.empty(), "matrix failure leaked memory");
        if (fail_request == SIZE_MAX) {
            check(!producer_failed && !policy_failed && committed == 2, "budgeted policy changed valid behavior");
            check(budget->stats().peak == upstream.peak, "successful requests omitted from peak");
            normal_requests = upstream.requests;
        } else {
            check(upstream.requests == fail_request && !committed && (producer_failed || policy_failed == 1),
                  "partial allocation failure was not atomic");
        }
    };
    run(SIZE_MAX);
    check(normal_requests > 30, "matrix did not reach dynamic neighbors and rank temporaries");
    for (size_t request = 1; request <= normal_requests; ++request) run(request);
    std::cout << "allocation_failure_points=" << normal_requests << '\n';
}

int main() {
    try {
        reallocation_and_owner_lifetime();
        upstream_failure_copy_swap_alignment_overflow();
        features_ranges_and_failed_publication();
        real_policy_exhaustion_keeps_reclaimer_alive();
        concurrent_shared_cap();
        allocation_failure_matrix();
        std::cout << "PASS auxiliary budget: actual requests, transient peak, old owners, copy/move/swap, alignment/overflow, injected failures, real policy exhaustion and 2400 concurrent attempts\n";
    } catch (const std::exception & error) {
        std::cerr << "FAIL auxiliary budget: " << error.what() << '\n';
        return 1;
    }
}
