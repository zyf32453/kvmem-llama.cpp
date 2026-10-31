#include "kvmem/source_journal.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace kvmem {
struct SourceJournalStore::Entry {
    SourceState state;
    HostKvRecord tokens, events;
    uint64_t token_capacity = 0, event_capacity = 0;
    bool forgetting = false;
};

namespace {
std::atomic<uint64_t> next_incarnation{1};
uint64_t incarnation() {
    auto current = next_incarnation.load(std::memory_order_relaxed);
    for (;;) {
        if (!current || current == UINT64_MAX)
            throw MemoryError(MemoryErrorCode::InvalidSession, "source directory identity exhausted");
        if (next_incarnation.compare_exchange_weak(current, current + 1, std::memory_order_relaxed))
            return current;
    }
}
uint64_t metadata(SourceLimits limits, size_t entry_bytes) {
    if (!limits.max_sources || !limits.max_tokens || !limits.max_events ||
        limits.max_sources > std::numeric_limits<size_t>::max() / entry_bytes ||
        limits.max_tokens > std::numeric_limits<size_t>::max() / sizeof(int32_t) ||
        limits.max_events > std::numeric_limits<size_t>::max() / sizeof(SourceEvent))
        throw MemoryError(MemoryErrorCode::InvalidPlan, "invalid source directory geometry");
    const auto bytes = uint64_t(limits.max_sources) * entry_bytes;
    if (bytes >= limits.bytes)
        throw MemoryError(MemoryErrorCode::BudgetExceeded, "source quota cannot hold its directory");
    return bytes;
}
uint64_t growth(uint64_t need, uint64_t old, uint64_t limit, uint64_t initial) {
    if (need <= old) return old;
    const auto doubled = old > limit / 2 ? limit : old * 2;
    return std::max(need, std::min(limit, std::max(initial, doubled)));
}
void valid_event(const SourceEvent & e, uint64_t end) {
    if (!e.id || e.begin >= e.end || e.end > end ||
        static_cast<unsigned>(e.kind) > static_cast<unsigned>(SourceEventKind::RawInput))
        throw MemoryError(MemoryErrorCode::InvalidPlan, "invalid accepted source event");
}
void valid_tokens(const int32_t * tokens, size_t count) {
    if (count && !tokens)
        throw MemoryError(MemoryErrorCode::InvalidPlan, "null accepted source rows");
    for (size_t i = 0; i < count; ++i)
        if (tokens[i] < 0)
            throw MemoryError(MemoryErrorCode::InvalidPlan, "negative accepted source token");
}
} // namespace

SourceJournalStore::SourceJournalStore(SourceLimits limits)
    : limits_(limits), metadata_bytes_(metadata(limits, sizeof(Entry))), incarnation_(incarnation()),
      buffers_(limits.bytes - metadata_bytes_), entries_(new Entry[limits.max_sources]), peak_(metadata_bytes_) {}
SourceJournalStore::~SourceJournalStore() = default;

[[noreturn]] void SourceJournalStore::exhausted(const char * why) {
    if (denied_ != UINT64_MAX) ++denied_;
    throw MemoryError(MemoryErrorCode::BudgetExceeded, why);
}
const SourceJournalStore::Entry & SourceJournalStore::find(SourceRef ref) const {
    if (ref.source.incarnation == incarnation_ && ref.source.id && ref.revision)
        for (size_t i = 0; i < limits_.max_sources; ++i)
            if (entries_[i].state.ref == ref) return entries_[i];
    throw MemoryError(MemoryErrorCode::Stale, "source record is missing or its revision changed");
}
SourceJournalStore::Entry & SourceJournalStore::find(SourceRef ref) {
    return const_cast<Entry &>(static_cast<const SourceJournalStore &>(*this).find(ref));
}

void SourceJournalStore::update(Entry & entry, const int32_t * tokens, size_t count,
    const SourceEvent * events, size_t event_count, uint64_t computed) {
    const auto old = entry.state.accepted_frontier;
    if (count > limits_.max_tokens - old) exhausted("source token limit exceeded");
    const auto end = old + count;
    if (computed > end || (event_count && !events))
        throw MemoryError(MemoryErrorCode::InvalidPlan, "source progress exceeds accepted rows");
    valid_tokens(tokens, count);
    bool extend = false;
    SourceEvent previous{};
    if (entry.state.events) previous = event(entry.state.ref, entry.state.events - 1);
    if (event_count && entry.state.events && events[0].id == previous.id &&
        events[0].kind == previous.kind && events[0].begin == previous.begin) {
        if (events[0].end < previous.end || (previous.complete &&
            (events[0].end != previous.end || !events[0].complete)))
            throw MemoryError(MemoryErrorCode::InvalidPlan, "completed source event cannot be rewritten");
        extend = true;
    }
    uint64_t last_end = extend ? previous.begin : old;
    for (size_t i = 0; i < event_count; ++i) {
        valid_event(events[i], end);
        if (events[i].begin < last_end)
            throw MemoryError(MemoryErrorCode::InvalidPlan, "source events overlap or rewrite old rows");
        last_end = events[i].end;
    }
    const auto added = event_count - size_t(extend);
    if (added > limits_.max_events - entry.state.events) exhausted("source event limit exceeded");
    const auto total_events = entry.state.events + added;
    const auto token_capacity = growth(end, entry.token_capacity, limits_.max_tokens, 64);
    const auto event_capacity = growth(total_events, entry.event_capacity, limits_.max_events, 8);
    const auto token_bytes = token_capacity != entry.token_capacity ? token_capacity * sizeof(int32_t) : 0;
    const auto event_bytes = event_capacity != entry.event_capacity ? event_capacity * sizeof(SourceEvent) : 0;
    if (token_bytes > UINT64_MAX - event_bytes) exhausted("source growth size overflow");
    // HostKvStorage reserves both buffers together. Existing capacities remain
    // charged throughout growth; a failure leaves the entire old entry intact.
    auto reserved = [&] {
        try { return buffers_.reserve(token_bytes + event_bytes); }
        catch (const MemoryError & e) {
            if (e.code() != MemoryErrorCode::BudgetExceeded) throw;
            exhausted("historical source budget exceeded");
        }
    }();
    peak_ = std::max(peak_, metadata_bytes_ + buffers_.charged_bytes());
    HostKvRecord new_tokens, new_events;
    if (token_bytes) new_tokens = reserved.allocate(size_t(token_bytes));
    if (event_bytes) new_events = reserved.allocate(size_t(event_bytes));
    const auto * old_tokens = old ? entry.tokens.data() : nullptr;
    const auto * old_events = entry.state.events ? entry.events.data() : nullptr;
    auto * token_destination = end ? (new_tokens ? new_tokens.writable_data() : entry.tokens.writable_data()) : nullptr;
    auto * event_destination = total_events ? (new_events ? new_events.writable_data() : entry.events.writable_data()) : nullptr;
    // All fallible work precedes the first write. The stores are private RAM
    // records without readers, so the remaining copies and publication cannot throw.
    if (new_tokens && old) std::memcpy(token_destination, old_tokens, size_t(old) * sizeof(int32_t));
    if (count) std::memcpy(token_destination + old * sizeof(int32_t), tokens, count * sizeof(int32_t));
    if (new_events && entry.state.events)
        std::memcpy(event_destination, old_events, size_t(entry.state.events) * sizeof(SourceEvent));
    if (event_count)
        std::memcpy(event_destination + (entry.state.events - size_t(extend)) * sizeof(SourceEvent),
                    events, event_count * sizeof(SourceEvent));
    if (new_tokens) entry.tokens = std::move(new_tokens);
    if (new_events) entry.events = std::move(new_events);
    entry.token_capacity = token_capacity; entry.event_capacity = event_capacity;
    entry.state.accepted_frontier = end; entry.state.computed_frontier = computed;
    entry.state.events = total_events;
}

SourceRef SourceJournalStore::capture_impl(const SourceConfiguration & configuration,
    const int32_t * tokens, size_t count, const SourceEvent * events, size_t event_count,
    uint64_t computed, SourceRef parent, uint64_t prefix) {
    if (std::all_of(configuration.begin(), configuration.end(), [](uint8_t x) { return x == 0; }))
        throw MemoryError(MemoryErrorCode::InvalidPlan, "source computation identity is missing");
    Entry * destination = nullptr;
    for (size_t i = 0; i < limits_.max_sources; ++i)
        if (!entries_[i].state.ref.source.id) { destination = &entries_[i]; break; }
    if (!destination) exhausted("source directory entry limit exceeded");
    if (next_id_ == UINT64_MAX)
        throw MemoryError(MemoryErrorCode::InvalidSession, "source identity space exhausted");
    Entry prepared;
    prepared.state.configuration = configuration;
    update(prepared, tokens, count, events, event_count, computed);
    prepared.state.ref = {{next_id_++, incarnation_}, 1};
    prepared.state.parent = parent; prepared.state.parent_prefix = prefix;
    *destination = std::move(prepared);
    return destination->state.ref;
}
SourceRef SourceJournalStore::capture(const SourceConfiguration & configuration,
    const int32_t * tokens, size_t count, const SourceEvent * events, size_t event_count, uint64_t computed) {
    return capture_impl(configuration, tokens, count, events, event_count, computed);
}
SourceRef SourceJournalStore::append(SourceRef expected, const int32_t * tokens, size_t count,
    const SourceEvent * events, size_t event_count, uint64_t computed) {
    auto & entry = find(expected);
    if (entry.state.ref.revision == UINT64_MAX)
        throw MemoryError(MemoryErrorCode::Stale, "source revision space exhausted");
    update(entry, tokens, count, events, event_count, computed);
    ++entry.state.ref.revision;
    return entry.state.ref;
}
SourceState SourceJournalStore::state(SourceRef ref) const { return find(ref).state; }
std::optional<SourceState> SourceJournalStore::inspect(size_t slot) const {
    if (slot >= limits_.max_sources)
        throw MemoryError(MemoryErrorCode::InvalidPlan, "source directory slot out of range");
    if (!entries_[slot].state.ref.source.id) return std::nullopt;
    return entries_[slot].state;
}
void SourceJournalStore::read_tokens(SourceRef ref, uint64_t begin, int32_t * out, size_t count) const {
    const auto & entry = find(ref);
    if ((count && !out) || begin > entry.state.accepted_frontier || count > entry.state.accepted_frontier - begin)
        throw MemoryError(MemoryErrorCode::InvalidPlan, "source read exceeds accepted rows");
    if (count) std::memcpy(out, entry.tokens.data() + begin * sizeof(int32_t), count * sizeof(int32_t));
}
SourceEvent SourceJournalStore::event(SourceRef ref, uint64_t index) const {
    const auto & entry = find(ref);
    if (index >= entry.state.events)
        throw MemoryError(MemoryErrorCode::InvalidPlan, "source event index out of range");
    SourceEvent result;
    std::memcpy(&result, entry.events.data() + index * sizeof(SourceEvent), sizeof(result));
    return result;
}
SourceRef SourceJournalStore::branch(SourceRef expected, uint64_t prefix) {
    const auto & entry = find(expected);
    if (prefix > entry.state.accepted_frontier)
        throw MemoryError(MemoryErrorCode::InvalidPlan, "source branch exceeds accepted rows");
    Entry * destination = nullptr;
    for (size_t i = 0; i < limits_.max_sources; ++i)
        if (!entries_[i].state.ref.source.id) { destination = &entries_[i]; break; }
    if (!destination) exhausted("source directory entry limit exceeded");
    if (next_id_ == UINT64_MAX)
        throw MemoryError(MemoryErrorCode::InvalidSession, "source identity space exhausted");
    // Copy directly into separately charged RAM preparation. In particular a
    // partial last event is clipped BEFORE the new entry becomes visible.
    Entry prepared;
    uint64_t count = 0;
    while (count < entry.state.events && event(expected, count).begin < prefix) ++count;
    prepared.token_capacity = growth(prefix, 0, limits_.max_tokens, 64);
    prepared.event_capacity = growth(count, 0, limits_.max_events, 8);
    const auto token_bytes = prepared.token_capacity * sizeof(int32_t);
    const auto event_bytes = prepared.event_capacity * sizeof(SourceEvent);
    if (token_bytes > UINT64_MAX - event_bytes) exhausted("source branch size overflow");
    auto reserved = [&] {
        try { return buffers_.reserve(token_bytes + event_bytes); }
        catch (const MemoryError & e) {
            if (e.code() != MemoryErrorCode::BudgetExceeded) throw;
            exhausted("historical source branch budget exceeded");
        }
    }();
    peak_ = std::max(peak_, metadata_bytes_ + buffers_.charged_bytes());
    if (token_bytes) {
        prepared.tokens = reserved.allocate(size_t(token_bytes));
        std::memcpy(prepared.tokens.writable_data(), entry.tokens.data(), size_t(prefix) * sizeof(int32_t));
    }
    if (event_bytes) {
        prepared.events = reserved.allocate(size_t(event_bytes));
        std::memcpy(prepared.events.writable_data(), entry.events.data(), size_t(count) * sizeof(SourceEvent));
    }
    if (count) {
        auto last = event(expected, count - 1);
        if (last.end > prefix) {
            last.end = prefix; last.complete = false;
            std::memcpy(prepared.events.writable_data() + (count - 1) * sizeof(SourceEvent), &last, sizeof(last));
        }
    }
    prepared.state = {{{next_id_++, incarnation_}, 1}, entry.state.configuration, expected, prefix, prefix, 0, count};
    *destination = std::move(prepared);
    return destination->state.ref;
}
bool SourceJournalStore::family_contains(SourceRef ancestor, SessionIdentity member) const {
    find(ancestor);
    for (size_t depth = 0; depth < limits_.max_sources && member.id; ++depth) {
        if (member == ancestor.source) return true;
        const Entry * entry = nullptr;
        for (size_t i = 0; i < limits_.max_sources; ++i)
            if (entries_[i].state.ref.source == member) { entry = &entries_[i]; break; }
        if (!entry) return false;
        member = entry->state.parent.source;
    }
    return false;
}
size_t SourceJournalStore::forget(SourceRef expected) {
    find(expected).forgetting = true;
    bool changed = true;
    while (changed) {
        changed = false;
        for (size_t i = 0; i < limits_.max_sources; ++i) {
            auto & e = entries_[i];
            if (!e.state.ref.source.id || e.forgetting || !e.state.parent.source.id) continue;
            for (size_t j = 0; j < limits_.max_sources; ++j)
                if (entries_[j].forgetting && e.state.parent.source == entries_[j].state.ref.source) {
                    e.forgetting = true; changed = true; break;
                }
        }
    }
    size_t count = 0;
    for (size_t i = 0; i < limits_.max_sources; ++i)
        if (entries_[i].forgetting) { entries_[i] = Entry{}; ++count; }
    return count;
}
SourceJournalStore::Stats SourceJournalStore::stats() const noexcept {
    Stats s;
    s.capacity = limits_.bytes; s.metadata_bytes = metadata_bytes_;
    s.buffer_bytes = buffers_.charged_bytes(); s.charged = metadata_bytes_ + s.buffer_bytes;
    s.peak = peak_; s.denied = denied_;
    for (size_t i = 0; i < limits_.max_sources; ++i) s.sources += bool(entries_[i].state.ref.source.id);
    return s;
}
} // namespace kvmem
