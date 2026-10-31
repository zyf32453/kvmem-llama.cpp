#include "kvmem/source_recovery.hpp"

#include <algorithm>
#include <limits>

namespace kvmem {
namespace {
[[noreturn]] void fail(MemoryErrorCode code, const char * message) {
    throw MemoryError(code, message);
}
uint64_t add(uint64_t a, uint64_t b) {
    if (b > UINT64_MAX - a) fail(MemoryErrorCode::InvalidPlan, "source recovery size overflow");
    return a + b;
}
uint64_t rounded(uint64_t tokens, uint64_t page) {
    const auto remainder = tokens % page;
    return remainder ? add(tokens, page - remainder) : tokens;
}
void validate_fences(const SourceJournalStore & store, const SourceRecoveryContext & current,
    const SourceRecoveryRequest & request) {
    if (!(request.expected.stamp == current.stamp))
        fail(MemoryErrorCode::Stale, "source recovery derived domain changed");
    if (current.configuration != request.expected.configuration)
        fail(MemoryErrorCode::Unsupported, "source recovery execution configuration changed");
    const auto source = store.state(request.source); // strict revision + store identity + local forget
    if (source.configuration != request.expected.configuration)
        fail(MemoryErrorCode::Unsupported, "source recovery configuration differs; migration is not supported");
    if (request.prefix_tokens > source.accepted_frontier)
        fail(MemoryErrorCode::InvalidPlan, "source recovery exceeds accepted rows");
}
} // namespace

SourceRecoveryPlan plan_source_recovery(const SourceJournalStore & store,
    const MemoryDescriptor & descriptor, const SourceRecoveryContext & current,
    const SourceRecoveryRequest & request) {
    descriptor.validate();
    const auto & expected = request.expected.stamp;
    if ((!(expected == MemoryStamp{}) && (!expected.session.id || !expected.session.incarnation ||
         !expected.history_revision || !expected.execution_history)) ||
        !request.destination.id || !request.destination.incarnation ||
        request.destination == request.expected.stamp.session)
        fail(MemoryErrorCode::InvalidPlan, "source recovery needs an owner stamp and a fresh destination identity");
    if (request.mode == SourceRecoveryMode::ExactState)
        fail(MemoryErrorCode::Unsupported, "accepted source rows cannot restore exact model state");
    if (request.mode != SourceRecoveryMode::RebuildAsNew)
        fail(MemoryErrorCode::InvalidPlan, "unknown source recovery mode");
    validate_fences(store, current, request);
    const auto page = descriptor.capabilities.page_tokens;
    const auto & budget = request.budget;
    if (!budget.device_tokens || budget.device_tokens % page || budget.selected_tokens % page)
        fail(MemoryErrorCode::InvalidPlan, "source recovery budget must respect native page granularity");
    if (add(request.prefix_tokens, budget.reserve_tokens) > descriptor.capabilities.max_position)
        fail(MemoryErrorCode::BudgetExceeded, "source recovery prefix and output exceed logical context");
    const auto prefix = rounded(request.prefix_tokens, page);
    const auto device = add(prefix, rounded(budget.reserve_tokens, page));
    if (prefix > budget.selected_tokens || device > budget.device_tokens)
        fail(MemoryErrorCode::BudgetExceeded, "complete canonical source recovery does not fit the device window");
    if (request.prefix_tokens > std::numeric_limits<size_t>::max() / sizeof(int32_t))
        fail(MemoryErrorCode::InvalidPlan, "source recovery staging size overflow");
    return SourceRecoveryPlan(request, device, request.prefix_tokens * sizeof(int32_t));
}

PreparedSourceRecovery prepare_source_recovery(const SourceRecoveryPlan & plan,
    const SourceJournalStore & store, const SourceRecoveryContext & current, HostKvStorage & staging) {
    validate_fences(store, current, plan.request());
    // No model mutation, source borrow or asynchronous work precedes reservation.
    auto reserved = staging.reserve(plan.staging_bytes_required());
    HostKvRecord rows;
    if (plan.staging_bytes_required()) {
        rows = reserved.allocate(size_t(plan.staging_bytes_required()));
        // Copy in bounded stack chunks: no second prefix-sized, uncharged vector.
        int32_t chunk[256];
        for (uint64_t begin = 0; begin < plan.request().prefix_tokens;) {
            const auto count = size_t(std::min<uint64_t>(256, plan.request().prefix_tokens - begin));
            store.read_tokens(plan.request().source, begin, chunk, count);
            rows.write(size_t(begin * sizeof(int32_t)), chunk, count * sizeof(int32_t));
            begin += count;
        }
        rows.seal();
    }
    return PreparedSourceRecovery(plan, std::move(rows));
}

void PreparedSourceRecovery::validate(const SourceJournalStore & store, const SourceRecoveryContext & current) const {
    validate_fences(store, current, plan_.request());
    if (rows_.size() != plan_.staging_bytes_required())
        fail(MemoryErrorCode::InvalidSession, "source recovery staging is detached");
}
void PreparedSourceRecovery::read_tokens(const SourceJournalStore & store, const SourceRecoveryContext & current,
    uint64_t begin, int32_t * out, size_t count) const {
    validate(store, current);
    const auto prefix = plan_.request().prefix_tokens;
    if ((count && !out) || begin > prefix || count > prefix - begin)
        fail(MemoryErrorCode::InvalidPlan, "source recovery read exceeds planned prefix");
    if (count) rows_.read(size_t(begin * sizeof(int32_t)), out, count * sizeof(int32_t));
}
} // namespace kvmem
