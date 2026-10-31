#pragma once

#include "source_journal.hpp"
#include <utility>

namespace kvmem {

// Source tokens can rebuild a NEW computation, not restore the original sparse,
// recurrent or speculative state. ExactState deliberately has no source fallback.
enum class SourceRecoveryMode { ExactState, RebuildAsNew };

// Snapshot of two existing authorities, not a second mutable catalog. The
// adapter supplies its CURRENT execution configuration on every validation.
struct SourceRecoveryContext {
    // Empty for source-only creation with no derived domain to replace. A
    // nonempty stamp is required/current only when replacing an existing domain.
    MemoryStamp stamp;
    SourceConfiguration configuration{};
};

struct SourceRecoveryRequest {
    SourceRef source;
    SourceRecoveryContext expected; // derived-domain owner; may already be invalid
    SessionIdentity destination; // fresh identity issued by that same owner
    uint64_t prefix_tokens = 0; // accepted source prefix, not computed_frontier
    SourceRecoveryMode mode = SourceRecoveryMode::ExactState;
    MemoryBudget budget;
};

// A fixed-size plan, not a resource reservation or a KV/state reuse permit.
// SourceJournalStore is the existing logical source directory; no parallel
// history catalog, checkpoint registry or identity allocator is created here.
class SourceRecoveryPlan {
public:
    const SourceRecoveryRequest & request() const noexcept { return request_; }
    uint64_t device_tokens_required() const noexcept { return device_tokens_; }
    uint64_t staging_bytes_required() const noexcept { return staging_bytes_; }
private:
    friend SourceRecoveryPlan plan_source_recovery(const SourceJournalStore &,
        const MemoryDescriptor &, const SourceRecoveryContext &, const SourceRecoveryRequest &);
    SourceRecoveryPlan(SourceRecoveryRequest request, uint64_t device, uint64_t staging)
        : request_(request), device_tokens_(device), staging_bytes_(staging) {}
    SourceRecoveryRequest request_;
    uint64_t device_tokens_ = 0, staging_bytes_ = 0;
};

// Conservative canonical text prefill: round the ENTIRE prefix and generation
// reserve independently, as in WorkingSetPlan. Chunk size is not a capacity
// substitute. The backend still has to reserve actual main/draft/state/scratch
// resources before executing; these arithmetic checks do not do that for it.
SourceRecoveryPlan plan_source_recovery(const SourceJournalStore &,
    const MemoryDescriptor &, const SourceRecoveryContext & current, const SourceRecoveryRequest &);

// Owner-serialized source staging only. Its real immutable copy counts against
// caller-owned R HostKvStorage for its full lifetime, separately from S/H/M.
// Neither raw pointers nor leases escape. Every read rechecks source revision /
// local forget and the current derived-domain stamp. Backend execution and
// publication must revalidate at their own safe boundary without interleaving
// owner mutation. This class does not publish model state or reserve GPU bytes.
class PreparedSourceRecovery {
public:
    PreparedSourceRecovery(PreparedSourceRecovery &&) noexcept = default;
    PreparedSourceRecovery & operator=(PreparedSourceRecovery &&) noexcept = default;
    PreparedSourceRecovery(const PreparedSourceRecovery &) = delete;
    PreparedSourceRecovery & operator=(const PreparedSourceRecovery &) = delete;
    const SourceRecoveryPlan & plan() const noexcept { return plan_; }
    void validate(const SourceJournalStore &, const SourceRecoveryContext & current) const;
    void read_tokens(const SourceJournalStore &, const SourceRecoveryContext & current,
        uint64_t begin, int32_t * out, size_t count) const;
private:
    friend PreparedSourceRecovery prepare_source_recovery(const SourceRecoveryPlan &,
        const SourceJournalStore &, const SourceRecoveryContext &, HostKvStorage &);
    PreparedSourceRecovery(SourceRecoveryPlan plan, HostKvRecord rows)
        : plan_(plan), rows_(std::move(rows)) {}
    SourceRecoveryPlan plan_;
    HostKvRecord rows_;
};

// Reserves and allocates all source-copy bytes before copying, then seals. On
// failure no source or derived-domain mutation occurs and R is fully refunded.
PreparedSourceRecovery prepare_source_recovery(const SourceRecoveryPlan &,
    const SourceJournalStore &, const SourceRecoveryContext & current, HostKvStorage & staging);

} // namespace kvmem
