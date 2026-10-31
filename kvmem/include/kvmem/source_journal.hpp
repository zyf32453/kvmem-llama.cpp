#pragma once

#include "host_kv_storage.hpp"
#include "memory_contract.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

namespace kvmem {
class SourceImage;

// Backend-produced digest of model, tokenizer/template, position/row encoding,
// KV/state schema and execution configuration. A match authorizes interpreting
// source rows only; it NEVER proves that any derived KV/checkpoint is reusable.
using SourceConfiguration = std::array<uint8_t, 32>;

struct SourceRef {
    SessionIdentity source; // independent of physical KV slots and their lifetime
    uint64_t revision = 0;
    bool operator==(const SourceRef & other) const noexcept {
        return source == other.source && revision == other.revision;
    }
};

enum class SourceEventKind : uint8_t { System, Developer, User, Assistant, ToolResult, Control, RawInput };
struct SourceEvent {
    uint64_t id = 0; // caller's structured message/tool event, not inferred semantics
    uint64_t begin = 0, end = 0;
    SourceEventKind kind = SourceEventKind::User;
    bool complete = true;
};

struct SourceState {
    SourceRef ref;
    SourceConfiguration configuration{};
    SourceRef parent;
    uint64_t parent_prefix = 0;
    uint64_t accepted_frontier = 0;
    // Logical evaluated frontier, not resident bytes or proof of complete KV.
    // It can trail accepted_frontier (including the last sampled output token).
    uint64_t computed_frontier = 0;
    uint64_t events = 0;
};

struct SourceLimits {
    uint64_t bytes = 0;
    uint64_t max_tokens = 0;
    uint64_t max_events = 0;
    size_t max_sources = 0;
};

// Engine-owner serialized RAM source directory. Source records survive derived
// KV eviction; capacity pressure rejects capture/append and never forgets data.
// One fixed entry array + real token/event buffer capacities count against S.
// HostKvStorage's bounded allocation/control objects, the store/account object,
// libc overhead and caller read buffers are fixed/runtime overhead outside S.
// No mutable borrow, async reader or unaccounted snapshot copy is exposed.
class SourceJournalStore {
public:
    struct Stats {
        uint64_t capacity = 0, metadata_bytes = 0, buffer_bytes = 0;
        uint64_t charged = 0, peak = 0, denied = 0, sources = 0;
    };

    explicit SourceJournalStore(SourceLimits);
    ~SourceJournalStore();
    SourceJournalStore(const SourceJournalStore &) = delete;
    SourceJournalStore & operator=(const SourceJournalStore &) = delete;

    SourceRef capture(const SourceConfiguration &, const int32_t * tokens, size_t count,
        const SourceEvent * events, size_t event_count, uint64_t computed_frontier);
    // Copy verified source DATA into a new canonical RAM root under S. Both
    // buffers are reserved before allocation; failure publishes nothing. Old
    // origin/parent refs are not current identities and computed restarts at0.
    // This grants no archive/derived-KV authority; an adapter must validate its
    // own read receipt and current configuration before calling this primitive.
    SourceRef import_image(const SourceImage &);
    // Only accepted rows, never speculative/provisional candidates. All inputs
    // are validated and both growth buffers reserved before publishing anything.
    // A final event may be extended using its same id/kind/begin, then new events
    // start at or beyond the old accepted frontier. Empty append can update KV progress.
    SourceRef append(SourceRef expected, const int32_t * tokens, size_t count,
        const SourceEvent * events, size_t event_count, uint64_t computed_frontier);
    // Explicit new source identity. The original remains immutable; a clipped
    // event becomes incomplete and the new derived domain has no computed KV.
    SourceRef branch(SourceRef expected, uint64_t prefix);
    SourceState state(SourceRef) const;
    size_t slots() const noexcept { return limits_.max_sources; }
    // Owner-serialized bounded enumeration; empty physical directory slots have
    // no record. The returned SourceRef remains subject to revision/forget fences.
    std::optional<SourceState> inspect(size_t slot) const;
    void read_tokens(SourceRef, uint64_t begin, int32_t * out, size_t count) const;
    SourceEvent event(SourceRef, uint64_t index) const;
    // Bounded, allocation-free preflight for an owner's alias revocation.
    // Validates the ancestor's current revision; member identity deliberately
    // ignores its revision so older derived bindings cannot evade the fence.
    bool family_contains(SourceRef ancestor, SessionIdentity member) const;
    // Explicitly forget this record and branches in THIS store. Backend KV,
    // persisted archives and external aliases must be fenced by their owner;
    // this primitive alone is not the system-wide ForgetSource operation.
    size_t forget(SourceRef);
    Stats stats() const noexcept;

private:
    struct Entry;
    Entry & find(SourceRef);
    const Entry & find(SourceRef) const;
    SourceRef capture_impl(const SourceConfiguration &, const int32_t *, size_t,
        const SourceEvent *, size_t, uint64_t, SourceRef parent = {}, uint64_t prefix = 0);
    void update(Entry &, const int32_t *, size_t, const SourceEvent *, size_t, uint64_t);
    [[noreturn]] void exhausted(const char *);
    const SourceLimits limits_;
    const uint64_t metadata_bytes_;
    const uint64_t incarnation_;
    HostKvStorage buffers_;
    std::unique_ptr<Entry[]> entries_;
    uint64_t next_id_ = 1, peak_ = 0, denied_ = 0;
};
} // namespace kvmem
