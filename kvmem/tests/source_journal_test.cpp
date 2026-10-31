#include "kvmem/source_journal.hpp"
#include "kvmem/historical_kv_page_gc.hpp"
#include "memory_test_support.hpp"

#include <array>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <numeric>

namespace {
thread_local size_t allocation_failure = SIZE_MAX, allocation_count = 0;
void fail_allocation() {
    if (allocation_failure != SIZE_MAX && allocation_count++ == allocation_failure) {
        allocation_failure = SIZE_MAX;
        throw std::bad_alloc();
    }
}
}
void * operator new(size_t bytes) {
    fail_allocation();
    if (auto * p = std::malloc(bytes ? bytes : 1)) return p;
    throw std::bad_alloc();
}
void * operator new[](size_t bytes) { return ::operator new(bytes); }
void operator delete(void * p) noexcept { std::free(p); }
void operator delete(void * p, size_t) noexcept { std::free(p); }
void operator delete[](void * p) noexcept { std::free(p); }
void operator delete[](void * p, size_t) noexcept { std::free(p); }

namespace {
SourceConfiguration configuration(uint8_t value = 1) {
    SourceConfiguration key{}; key[0] = value; return key;
}
SourceLimits limits(size_t records = 8) { return {1 << 20, 4096, 64, records}; }
std::vector<int32_t> read(const SourceJournalStore & store, SourceRef ref) {
    std::vector<int32_t> result(size_t(store.state(ref).accepted_frontier));
    store.read_tokens(ref, 0, result.data(), result.size());
    return result;
}
void accepted_rows_and_events() {
    SourceJournalStore store(limits());
    const int32_t prompt[]{11, 12, 13, 14};
    const SourceEvent events[]{{1, 0, 2, SourceEventKind::User, true},
                               {2, 2, 4, SourceEventKind::ToolResult, true}};
    auto original = store.capture(configuration(), prompt, 4, events, 2, 3);
    auto state = store.state(original);
    CHECK(state.accepted_frontier == 4 && state.computed_frontier == 3 && state.events == 2);
    CHECK(state.configuration == configuration());
    CHECK(read(store, original) == (std::vector<int32_t>{11, 12, 13, 14}));
    CHECK(store.event(original, 1).kind == SourceEventKind::ToolResult);
    const int32_t accepted[]{21, 22}; // e.g. two accepted tokens from a wider verify round
    const SourceEvent output{3, 4, 6, SourceEventKind::Assistant, false};
    auto appended = store.append(original, accepted, 2, &output, 1, 5);
    CHECK(appended.source == original.source && appended.revision > original.revision);
    rejects(MemoryErrorCode::Stale, [&] { store.state(original); });
    CHECK(read(store, appended) == (std::vector<int32_t>{11, 12, 13, 14, 21, 22}));
    const int32_t last = 23;
    const SourceEvent extended{3, 4, 7, SourceEventKind::Assistant, true};
    auto completed = store.append(appended, &last, 1, &extended, 1, 6);
    CHECK(store.state(completed).events == 3 && store.event(completed, 2).complete);
    CHECK(store.state(completed).computed_frontier + 1 == store.state(completed).accepted_frontier);
    // Re-evaluation/rewind affects the derived frontier, not the source rows.
    auto rewound = store.append(completed, nullptr, 0, nullptr, 0, 2);
    CHECK(read(store, rewound).back() == 23 && store.state(rewound).accepted_frontier == 7);
    rejects(MemoryErrorCode::InvalidPlan, [&] { store.read_tokens(rewound, 7, nullptr, 1); });
    CHECK(store.stats().charged <= store.stats().peak && store.stats().peak <= store.stats().capacity);
}
void reject_rewrites_without_partial_publication() {
    SourceJournalStore store(limits());
    const int32_t tokens[]{1, 2, 3, 4};
    const SourceEvent closed{1, 0, 4, SourceEventKind::ToolResult, true};
    const auto ref = store.capture(configuration(), tokens, 4, &closed, 1, 3);
    const auto before = store.stats();
    const int32_t next = 5;
    const SourceEvent rewrite{1, 0, 5, SourceEventKind::ToolResult, true};
    rejects(MemoryErrorCode::InvalidPlan, [&] { store.append(ref, &next, 1, &rewrite, 1, 4); });
    const SourceEvent overlap{2, 3, 5, SourceEventKind::User, true};
    rejects(MemoryErrorCode::InvalidPlan, [&] { store.append(ref, &next, 1, &overlap, 1, 4); });
    rejects(MemoryErrorCode::InvalidPlan, [&] { store.append(ref, &next, 1, nullptr, 0, 6); });
    const int32_t invalid = -1;
    rejects(MemoryErrorCode::InvalidPlan, [&] { store.append(ref, &invalid, 1, nullptr, 0, 4); });
    rejects(MemoryErrorCode::InvalidPlan, [&] { store.capture({}, tokens, 4, nullptr, 0, 0); });
    CHECK(store.state(ref).accepted_frontier == 4 && read(store, ref) == (std::vector<int32_t>{1, 2, 3, 4}));
    CHECK(store.stats().charged == before.charged && store.stats().sources == 1);
}
void branch_and_forget_fence() {
    SourceJournalStore store(limits());
    const int32_t input[]{1, 2, 3, 4, 5};
    const SourceEvent event{10, 0, 5, SourceEventKind::Assistant, true};
    auto parent = store.capture(configuration(), input, 5, &event, 1, 4);
    auto branch = store.branch(parent, 3);
    CHECK(!(branch.source == parent.source));
    CHECK(store.state(branch).parent == parent && store.state(branch).parent_prefix == 3);
    CHECK(store.state(branch).computed_frontier == 0);
    CHECK(!store.event(branch, 0).complete && store.event(branch, 0).end == 3);
    const int32_t changed = 99;
    const SourceEvent replacement{10, 0, 4, SourceEventKind::Assistant, true};
    branch = store.append(branch, &changed, 1, &replacement, 1, 0);
    CHECK(read(store, branch) == (std::vector<int32_t>{1, 2, 3, 99}));
    CHECK(read(store, parent) == (std::vector<int32_t>{1, 2, 3, 4, 5}));
    const auto grandchild = store.branch(branch, 2);
    const auto empty = store.branch(parent, 0);
    CHECK(store.state(empty).accepted_frontier == 0 && store.state(empty).events == 0);
    const auto unrelated = store.capture(configuration(2), input, 5, nullptr, 0, 0);
    CHECK(store.forget(parent) == 4);
    for (auto ref : {parent, branch, grandchild, empty})
        rejects(MemoryErrorCode::Stale, [&] { store.state(ref); });
    CHECK(read(store, unrelated) == (std::vector<int32_t>{1, 2, 3, 4, 5}));
    const auto reused_slot = store.capture(configuration(), input, 5, nullptr, 0, 0);
    CHECK(!(reused_slot.source == parent.source));
    SourceJournalStore other(limits());
    const auto other_ref = other.capture(configuration(), input, 5, nullptr, 0, 0);
    rejects(MemoryErrorCode::Stale, [&] { store.state(other_ref); });
    rejects(MemoryErrorCode::Stale, [&] { other.state(reused_slot); });
}
void real_capacity_and_independent_limits() {
    SourceJournalStore probe(limits(1));
    const auto metadata = probe.stats().metadata_bytes;
    SourceJournalStore store({metadata + 700, 1024, 64, 1});
    std::array<int32_t, 64> tokens{};
    const auto old = store.capture(configuration(), tokens.data(), tokens.size(), nullptr, 0, 64);
    const auto before = store.stats();
    const int32_t next = 8;
    // The eventual 512-byte replacement fits, but old + new (768) does not.
    rejects(MemoryErrorCode::BudgetExceeded, [&] { store.append(old, &next, 1, nullptr, 0, 64); });
    CHECK(store.stats().charged == before.charged && store.state(old).accepted_frontier == 64);
    CHECK(store.stats().denied == 1 && store.stats().peak <= store.stats().capacity);
    rejects(MemoryErrorCode::BudgetExceeded, [&] { store.capture(configuration(), &next, 1, nullptr, 0, 0); });
    CHECK(store.forget(old) == 1 && store.stats().charged == metadata);
    const auto fresh = store.capture(configuration(), &next, 1, nullptr, 0, 0);
    CHECK(!(old.source == fresh.source));
    SourceJournalStore small({1 << 16, 2, 1, 2});
    const SourceEvent one{1, 0, 1, SourceEventKind::User, true};
    const auto ref = small.capture(configuration(), &next, 1, &one, 1, 0);
    const SourceEvent two{2, 1, 2, SourceEventKind::ToolResult, true};
    rejects(MemoryErrorCode::BudgetExceeded, [&] { small.append(ref, &next, 1, &two, 1, 0); });
    rejects(MemoryErrorCode::BudgetExceeded, [&] { small.append(ref, tokens.data(), 2, nullptr, 0, 0); });
    CHECK(small.state(ref).accepted_frontier == 1 && small.state(ref).events == 1);
}
void owner_family_preflight() {
    SourceJournalStore store(limits());
    const int32_t token = 17;
    auto root = store.capture(configuration(), &token, 1, nullptr, 0, 0);
    auto child = store.branch(root, 1);
    auto grandchild = store.branch(child, 1);
    auto other = store.capture(configuration(), &token, 1, nullptr, 0, 0);
    const auto old_child = child;
    child = store.append(child, &token, 1, nullptr, 0, 1);
    const auto before = store.stats();
    // Native windows may still carry an older revision of the same source.
    // Membership and a valid ancestor must require no heap preparation.
    allocation_failure = 0; allocation_count = 0;
    const bool self = store.family_contains(root, root.source);
    const bool direct = store.family_contains(root, old_child.source);
    const bool deep = store.family_contains(root, grandchild.source);
    const bool unrelated = store.family_contains(root, other.source);
    const bool foreign = store.family_contains(root, {root.source.id, root.source.incarnation + 1});
    allocation_failure = SIZE_MAX;
    CHECK(self && direct && deep && !unrelated && !foreign);
    CHECK(store.stats().charged == before.charged && store.stats().sources == before.sources);
    rejects(MemoryErrorCode::Stale, [&] { store.family_contains(old_child, grandchild.source); });
    CHECK(store.forget(child) == 2);
    CHECK(!store.family_contains(root, old_child.source));
    CHECK(store.family_contains(root, root.source) && !store.family_contains(root, other.source));
    const auto reused = store.capture(configuration(), &token, 1, nullptr, 0, 0);
    CHECK(!store.family_contains(root, reused.source));
    CHECK(store.forget(root) == 1 && store.state(other).ref == other && store.state(reused).ref == reused);
}
void allocation_failure_atomicity() {
    size_t total_failures = 0;
    for (int operation = 0; operation < 3; ++operation) {
        bool reached_success = false;
        for (size_t fail = 0; fail < 32; ++fail) {
            SourceJournalStore store(limits());
            std::array<int32_t, 64> input{}; std::iota(input.begin(), input.end(), 1);
            const SourceEvent event{1, 0, 64, SourceEventKind::Assistant, false};
            const auto ref = store.capture(configuration(), input.data(), input.size(), &event, 1, 63);
            std::array<int32_t, 129> output{}; output.fill(99);
            std::array<SourceEvent, 10> events{};
            events[0] = {1, 0, 65, SourceEventKind::Assistant, false};
            for (size_t i = 1; i < events.size(); ++i)
                events[i] = {i + 1, 64 + i, 65 + i, SourceEventKind::User, true};
            const auto before = store.stats();
            allocation_failure = fail; allocation_count = 0;
            SourceRef result;
            bool failed = false;
            try {
                if (operation == 0) result = store.append(ref, output.data(), 65, events.data(), events.size(), 128);
                if (operation == 1) result = store.branch(ref, 63);
                if (operation == 2) result = store.capture(configuration(), output.data(), output.size(), events.data(), events.size(), 0);
            } catch (const std::bad_alloc &) { failed = true; }
            allocation_failure = SIZE_MAX;
            if (failed) {
                ++total_failures;
                CHECK(store.stats().charged == before.charged && store.stats().sources == before.sources);
                CHECK(store.state(ref).accepted_frontier == 64 && store.event(ref, 0).end == 64);
                CHECK(read(store, ref) == std::vector<int32_t>(input.begin(), input.end()));
            } else {
                CHECK(result.source.id && store.state(result).accepted_frontier != 0);
                reached_success = true; break;
            }
            CHECK(store.stats().peak <= store.stats().capacity);
        }
        CHECK(reached_success);
    }
    CHECK(total_failures >= 6);
    std::printf("source allocation failures checked: %zu\n", total_failures);
}
void source_survives_actual_page_retirement() {
    SourceJournalStore store(limits());
    std::array<int32_t, 193> input{}; std::iota(input.begin(), input.end(), 1);
    const auto source = store.capture(configuration(), input.data(), input.size(), nullptr, 0, 192);
    const auto source_charge = store.stats().charged;
    HistoricalKvPageGc::Config config;
    config.slots = config.logical_pages = 3; config.page_tokens = 64;
    config.sink_tokens = config.recent_tokens = 64;
    config.low_ready = 1; config.high_ready = 2;
    HistoricalKvPageGc gc(config);
    for (uint32_t i = 0; i < 3; ++i) CHECK(gc.register_page(i, 128, i + 1));
    gc.protect_page(1, false);
    for (int i = 0; i < 8 && !gc.stats().proposed; ++i) gc.manual_tick();
    const auto & retired = gc.commit_boundary();
    CHECK(retired.size() == 1 && retired.front().page == 1);
    HostKvStorage host(128);
    auto record = host.allocate(128); record.seal();
    auto reader = record.lease();
    gc.acknowledge_host_release(retired.front(), std::move(record));
    gc.manual_tick(); CHECK(host.charged_bytes() == 128);
    reader = {};
    for (int i = 0; i < 8 && !gc.stats().reclaimed; ++i) gc.manual_tick();
    CHECK(host.charged_bytes() == 0 && gc.stats().reclaimed == 1 && !gc.is_live(1));
    CHECK(store.stats().charged == source_charge);
    CHECK(read(store, source) == std::vector<int32_t>(input.begin(), input.end()));
    // Reading source cannot revive the retired physical KV handle.
    CHECK(!gc.is_live(1) && store.state(source).computed_frontier == 192);
}
} // namespace

int main() {
    accepted_rows_and_events();
    reject_rewrites_without_partial_publication();
    branch_and_forget_fence();
    real_capacity_and_independent_limits();
    owner_family_preflight();
    allocation_failure_atomicity();
    source_survives_actual_page_retirement();
    std::puts("source journal contracts passed; no native recovery integration claimed");
}
