#include "kvmem/source_recovery.hpp"
#include "kvmem/historical_kv_page_gc.hpp"
#include "memory_test_support.hpp"

#include <array>
#include <cstring>
#include <limits>
#include <new>
#include <numeric>
#include <optional>

namespace {
thread_local size_t fail_at = SIZE_MAX, allocations = 0;
void fail_allocation() {
    if (fail_at != SIZE_MAX && allocations++ == fail_at) {
        fail_at = SIZE_MAX;
        throw std::bad_alloc();
    }
}
}
void * operator new(size_t bytes) {
    fail_allocation();
    if (auto * pointer = std::malloc(bytes ? bytes : 1)) return pointer;
    throw std::bad_alloc();
}
void * operator new[](size_t bytes) { return ::operator new(bytes); }
void operator delete(void * pointer) noexcept { std::free(pointer); }
void operator delete(void * pointer, size_t) noexcept { std::free(pointer); }
void operator delete[](void * pointer) noexcept { std::free(pointer); }
void operator delete[](void * pointer, size_t) noexcept { std::free(pointer); }

namespace {
struct Fixture {
    SourceJournalStore sources{{1 << 20, 4096, 64, 8}};
    SourceRecoveryContext current{{true, {42, 3}, 7, 11, 192, 5}, {}};
    MemoryDescriptor model = descriptor(2);
    std::array<int32_t, 193> accepted{};
    SourceRef source;
    Fixture() {
        current.configuration[0] = 99;
        std::iota(accepted.begin(), accepted.end(), 1);
        source = sources.capture(current.configuration, accepted.data(), accepted.size(), nullptr, 0, 192);
    }
    SourceRecoveryRequest request() const {
        return {source, current, {43, 3}, accepted.size(), SourceRecoveryMode::RebuildAsNew,
            {256, 65, 384, 0}};
    }
    SourceRecoveryPlan plan() const { return plan_source_recovery(sources, model, current, request()); }
    void unchanged() const {
        std::array<int32_t, 193> actual{};
        sources.read_tokens(source, 0, actual.data(), actual.size());
        CHECK(actual == accepted);
        CHECK(sources.state(source).computed_frontier == 192);
    }
};

void explicit_semantics_and_capacity() {
    Fixture f;
    const auto request = f.request();
    const auto plan = f.plan();
    // 193 source rows -> 256 slots; 65 output tokens -> another 128 slots.
    CHECK(plan.device_tokens_required() == 384);
    CHECK(plan.staging_bytes_required() == 193 * sizeof(int32_t));
    CHECK(plan.request().destination == (SessionIdentity{43, 3}));
    CHECK(plan.request().source == f.source);
    auto changed = request;
    changed.mode = SourceRecoveryMode::ExactState;
    rejects(MemoryErrorCode::Unsupported, [&] { plan_source_recovery(f.sources, f.model, f.current, changed); });
    changed.mode = SourceRecoveryMode(19);
    rejects(MemoryErrorCode::InvalidPlan, [&] { plan_source_recovery(f.sources, f.model, f.current, changed); });
    changed = request; changed.destination = changed.expected.stamp.session;
    rejects(MemoryErrorCode::InvalidPlan, [&] { plan_source_recovery(f.sources, f.model, f.current, changed); });
    changed = request; changed.destination.incarnation = 0;
    rejects(MemoryErrorCode::InvalidPlan, [&] { plan_source_recovery(f.sources, f.model, f.current, changed); });
    changed = request; changed.expected.stamp.history_revision = 0;
    rejects(MemoryErrorCode::InvalidPlan, [&] { plan_source_recovery(f.sources, f.model, f.current, changed); });
    changed = request; changed.prefix_tokens = 194;
    rejects(MemoryErrorCode::InvalidPlan, [&] { plan_source_recovery(f.sources, f.model, f.current, changed); });
    // Small prefill chunks cannot make this full prefix fit a 256-slot window.
    changed = request; changed.budget.device_tokens = 256;
    rejects(MemoryErrorCode::BudgetExceeded, [&] { plan_source_recovery(f.sources, f.model, f.current, changed); });
    changed = request; changed.budget.selected_tokens = 192;
    rejects(MemoryErrorCode::BudgetExceeded, [&] { plan_source_recovery(f.sources, f.model, f.current, changed); });
    changed = request; changed.budget.device_tokens = 383;
    rejects(MemoryErrorCode::InvalidPlan, [&] { plan_source_recovery(f.sources, f.model, f.current, changed); });
    auto limited = f.model; limited.capabilities.max_position = 257;
    rejects(MemoryErrorCode::BudgetExceeded, [&] { plan_source_recovery(f.sources, limited, f.current, request); });
    changed = request; changed.budget.reserve_tokens = UINT64_MAX;
    rejects(MemoryErrorCode::InvalidPlan, [&] { plan_source_recovery(f.sources, f.model, f.current, changed); });
    // An invalidated old computation can be replaced, with an explicit new ID.
    f.current.stamp.valid = false;
    CHECK(f.plan().request().expected.stamp.valid == false);
    f.unchanged();
}

void partial_prefix_and_empty_source() {
    Fixture f;
    auto request = f.request(); request.prefix_tokens = 96;
    request.budget = {128, 1, 192, 0};
    const auto plan = plan_source_recovery(f.sources, f.model, f.current, request);
    CHECK(plan.device_tokens_required() == 192);
    HostKvStorage staging(4096);
    auto prepared = prepare_source_recovery(plan, f.sources, f.current, staging);
    std::array<int32_t, 96> output{};
    prepared.read_tokens(f.sources, f.current, 0, output.data(), output.size());
    CHECK(std::equal(output.begin(), output.end(), f.accepted.begin()));
    CHECK(staging.charged_bytes() == 384);
    rejects(MemoryErrorCode::InvalidPlan, [&] { prepared.read_tokens(f.sources, f.current, 96, output.data(), 1); });
    rejects(MemoryErrorCode::InvalidPlan, [&] { prepared.read_tokens(f.sources, f.current, 0, nullptr, 1); });
    request.source = f.sources.capture(f.current.configuration, nullptr, 0, nullptr, 0, 0);
    request.prefix_tokens = 0; request.budget = {0, 1, 64, 0};
    const auto empty = plan_source_recovery(f.sources, f.model, f.current, request);
    CHECK(empty.staging_bytes_required() == 0 && empty.device_tokens_required() == 64);
    HostKvStorage no_bytes(0);
    auto zero = prepare_source_recovery(empty, f.sources, f.current, no_bytes);
    zero.read_tokens(f.sources, f.current, 0, nullptr, 0);
    CHECK(no_bytes.charged_bytes() == 0);
    // Cross both bounded-copy chunk boundaries, and read across one of them.
    std::array<int32_t, 600> long_input{};
    std::iota(long_input.begin(), long_input.end(), 1000);
    request.source = f.sources.capture(f.current.configuration, long_input.data(), long_input.size(), nullptr, 0, 0);
    request.prefix_tokens = long_input.size(); request.budget = {640, 1, 704, 0};
    const auto long_plan = plan_source_recovery(f.sources, f.model, f.current, request);
    auto long_prepared = prepare_source_recovery(long_plan, f.sources, f.current, staging);
    std::array<int32_t, 600> long_output{};
    long_prepared.read_tokens(f.sources, f.current, 0, long_output.data(), long_output.size());
    CHECK(long_output == long_input);
    std::array<int32_t, 20> crossed{};
    long_prepared.read_tokens(f.sources, f.current, 250, crossed.data(), crossed.size());
    CHECK(std::equal(crossed.begin(), crossed.end(), long_input.begin() + 250));
    f.unchanged();
}

void source_configuration_and_domain_fences() {
    Fixture f;
    const auto plan = f.plan();
    HostKvStorage staging(4096);
    auto prepared = prepare_source_recovery(plan, f.sources, f.current, staging);
    for (int field = 0; field < 7; ++field) {
        auto changed = f.current;
        if (field == 0) ++changed.stamp.session.incarnation;
        if (field == 1) ++changed.stamp.session.id;
        if (field == 2) ++changed.stamp.history_revision; // retirement / suffix edit by owner
        if (field == 3) ++changed.stamp.view_version;
        if (field == 4) ++changed.stamp.execution_history;
        if (field == 5) ++changed.stamp.frontier;
        if (field == 6) changed.stamp.valid = false;
        rejects(MemoryErrorCode::Stale, [&] { prepare_source_recovery(plan, f.sources, changed, staging); });
        rejects(MemoryErrorCode::Stale, [&] { prepared.validate(f.sources, changed); });
    }
    auto other_model = f.current; ++other_model.configuration[0];
    rejects(MemoryErrorCode::Unsupported, [&] { prepared.validate(f.sources, other_model); });
    auto request = f.request(); request.expected = other_model;
    rejects(MemoryErrorCode::Unsupported, [&] { plan_source_recovery(f.sources, f.model, other_model, request); });
    // Even computed-frontier-only edits stale a prepared plan, conservatively.
    const auto original = f.source;
    f.source = f.sources.append(f.source, nullptr, 0, nullptr, 0, 192);
    rejects(MemoryErrorCode::Stale, [&] { prepared.read_tokens(f.sources, f.current, 0, nullptr, 0); });
    rejects(MemoryErrorCode::Stale, [&] { prepare_source_recovery(plan, f.sources, f.current, staging); });
    CHECK(!(f.source == original));
    f.unchanged();
}

void forget_revokes_prepared_and_old_plans() {
    Fixture f;
    const auto child = f.sources.branch(f.source, 96);
    auto request = f.request(); request.source = child; request.prefix_tokens = 96;
    const auto plan = plan_source_recovery(f.sources, f.model, f.current, request);
    HostKvStorage staging(4096);
    std::optional<PreparedSourceRecovery> prepared;
    prepared.emplace(prepare_source_recovery(plan, f.sources, f.current, staging));
    CHECK(staging.charged_bytes() == 384);
    CHECK(f.sources.forget(f.source) == 2);
    rejects(MemoryErrorCode::Stale, [&] { prepared->validate(f.sources, f.current); });
    std::array<int32_t, 96> output{}; output.fill(-1);
    rejects(MemoryErrorCode::Stale, [&] { prepared->read_tokens(f.sources, f.current, 0, output.data(), output.size()); });
    CHECK(output.front() == -1 && output.back() == -1);
    rejects(MemoryErrorCode::Stale, [&] { prepare_source_recovery(plan, f.sources, f.current, staging); });
    const auto replacement = f.sources.capture(f.current.configuration, f.accepted.data(), 193, nullptr, 0, 192);
    CHECK(!(replacement.source == f.source.source));
    rejects(MemoryErrorCode::Stale, [&] { prepared->validate(f.sources, f.current); });
    SourceJournalStore different{{1 << 20, 4096, 64, 8}};
    rejects(MemoryErrorCode::Stale, [&] { prepared->validate(different, f.current); });
    // Revoked data stays charged until its real staging owner is destroyed.
    CHECK(staging.charged_bytes() == 384);
    prepared.reset(); CHECK(staging.charged_bytes() == 0);
}

void real_staging_quota_and_failure_atomicity() {
    Fixture f;
    const auto plan = f.plan();
    HostKvStorage short_by_one(plan.staging_bytes_required() - 1);
    const auto source_bytes = f.sources.stats().charged;
    rejects(MemoryErrorCode::BudgetExceeded, [&] { prepare_source_recovery(plan, f.sources, f.current, short_by_one); });
    CHECK(short_by_one.charged_bytes() == 0 && f.sources.stats().charged == source_bytes);
    HostKvStorage shared(plan.staging_bytes_required());
    {
        auto first = prepare_source_recovery(plan, f.sources, f.current, shared);
        rejects(MemoryErrorCode::BudgetExceeded, [&] { prepare_source_recovery(plan, f.sources, f.current, shared); });
        CHECK(shared.charged_bytes() == plan.staging_bytes_required());
        auto moved = std::move(first);
        rejects(MemoryErrorCode::InvalidSession, [&] { first.validate(f.sources, f.current); });
        moved.validate(f.sources, f.current);
    }
    CHECK(shared.charged_bytes() == 0);
    size_t faults = 0; bool completed = false;
    for (size_t failure = 0; failure < 32; ++failure) {
        fail_at = failure; allocations = 0;
        try {
            auto prepared = prepare_source_recovery(plan, f.sources, f.current, shared);
            fail_at = SIZE_MAX;
            CHECK(shared.charged_bytes() == plan.staging_bytes_required());
            prepared.validate(f.sources, f.current);
            completed = true;
        } catch (const std::bad_alloc &) { fail_at = SIZE_MAX; ++faults; }
        CHECK(shared.charged_bytes() == 0 && f.sources.stats().charged == source_bytes);
        f.unchanged();
        if (completed) break;
    }
    CHECK(completed && faults >= 2);
    std::printf("source recovery allocation failures checked: %zu\n", faults);
}

void plan_after_actual_cpu_release_does_not_revive_retired_page() {
    Fixture f;
    HistoricalKvPageGc::Config config;
    config.slots = config.logical_pages = 3; config.page_tokens = 64;
    config.sink_tokens = config.recent_tokens = 64;
    config.low_ready = 1; config.high_ready = 2;
    HistoricalKvPageGc gc(config);
    for (uint32_t id = 0; id < 3; ++id) CHECK(gc.register_page(id, 128, id + 1));
    gc.protect_page(1, false);
    for (int i = 0; i < 8 && !gc.stats().proposed; ++i) gc.manual_tick();
    const auto & retired = gc.commit_boundary();
    CHECK(retired.size() == 1 && retired.front().page == 1);
    ++f.current.stamp.history_revision; // derived owner fences plans at retirement
    HostKvStorage old_payload(128);
    auto record = old_payload.allocate(128); record.seal();
    auto reader = record.lease();
    gc.acknowledge_host_release(retired.front(), std::move(record));
    gc.manual_tick(); CHECK(old_payload.charged_bytes() == 128);
    reader = {};
    for (int i = 0; i < 8 && !gc.stats().reclaimed; ++i) gc.manual_tick();
    CHECK(old_payload.charged_bytes() == 0 && !gc.is_live(1));
    const auto plan = f.plan();
    HostKvStorage recovery(4096);
    auto prepared = prepare_source_recovery(plan, f.sources, f.current, recovery);
    std::array<int32_t, 193> actual{};
    prepared.read_tokens(f.sources, f.current, 0, actual.data(), actual.size());
    CHECK(actual == f.accepted && plan.request().destination == (SessionIdentity{43, 3}));
    CHECK(!gc.is_live(1) && old_payload.charged_bytes() == 0 && gc.stats().reclaimed == 1);
    f.unchanged();
}
} // namespace

int main() {
    explicit_semantics_and_capacity();
    partial_prefix_and_empty_source();
    source_configuration_and_domain_fences();
    forget_revokes_prepared_and_old_plans();
    real_staging_quota_and_failure_atomicity();
    plan_after_actual_cpu_release_does_not_revive_retired_page();
    std::puts("source recovery planning/staging contracts passed; native rebuild/publication is not implemented here");
}
