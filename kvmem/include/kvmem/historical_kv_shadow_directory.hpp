#pragma once

#include "kvmem/historical_kv_repository.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>

namespace kvmem {

class KvMemStore;

// An opt-in, per-runtime identity bridge for the gradual migration away from
// dense block IDs. Selection filters and read/transfer admission consult the
// exact repository generation. This does not admit GPU execution, retire a
// block, or release storage. A lookup is never a substitute for a lease.
// The owner must keep a release callback's storage target alive separately.
// Directory operations are serialized by the foreground owner. Only the
// repository may be retired/reclaimed concurrently by its background worker.
class HistoricalKvShadowDirectory {
public:
    enum class State { Unmanaged, Bound, Retired, Reclaimed };

    class WriterContext {
    public:
        WriterContext(const WriterContext &) = delete;
        WriterContext & operator=(const WriterContext &) = delete;

        const HistoricalKvRepository & repository() const { return *repository_; }
        HistoricalKvRepository & repository() { return *repository_; }
        const HistoricalKvRepository::Lease & writer() const { return writer_; }
        HistoricalKvHandle handle() const { return writer_.handle(); }

    private:
        friend class HistoricalKvShadowDirectory;
        explicit WriterContext(std::shared_ptr<HistoricalKvRepository> repository)
            : repository_(std::move(repository)) {}

        std::shared_ptr<HistoricalKvRepository> repository_;
        HistoricalKvRepository::Lease writer_;
    };

    struct Binding {
        uint32_t block_id = 0;
        HistoricalKvHandle handle;
        uint64_t original_begin = 0;
        uint32_t valid_tokens = 0;
        // Async harvest may retain this after the runtime changes owner. It
        // must write through the captured handle, never a fresh block-ID lookup.
        std::shared_ptr<WriterContext> writer_context;
    };

    struct Observation {
        State state = State::Unmanaged;
        HistoricalKvHandle handle{};
        uint64_t repository_id = 0;
        uint64_t original_begin = 0;
        uint32_t valid_tokens = 0;
    };

    struct Access {
        enum class Status { Unmanaged, Acquired, Denied };
        Status status = Status::Denied;
        HistoricalKvRepository::Lease lease;

        explicit operator bool() const {
            return status == Status::Unmanaged ||
                   (status == Status::Acquired && static_cast<bool>(lease));
        }
    };

    // On success moves writer into a shared context. On failure leaves writer
    // with the caller, including on allocation failure. One repository owns
    // this directory for the lifetime of its runtime, even when it is empty.
    bool bind_full_block(const KvMemStore & store, uint32_t block_id,
                         std::shared_ptr<HistoricalKvRepository> repository,
                         HistoricalKvHandle handle,
                         HistoricalKvRepository::Lease & writer);
    std::optional<Binding> find(uint32_t block_id) const;
    // Unlike find(), this preserves a retired generation's identity even
    // after physical release. Retired and Reclaimed are both closed to new
    // reads; neither is an unmanaged legacy block.
    Observation observe(uint32_t block_id) const;
    // Host selection prefilter only. It reads the exact repository generation
    // so a retirement is visible even before the owner unbinds its writer.
    // It does not grant an execution or transfer lease.
    bool allows_new_selection(uint32_t block_id) const;
    // Unmanaged blocks keep their legacy path. Managed blocks require an
    // atomic lease admission; Writing, retired, reclaimed and closed owners
    // are denied. Retain Access until all CPU reads / transfers finish.
    // A failed managed admission must never fall back to a legacy read.
    Access acquire_read(uint32_t block_id) const;
    Access acquire_transfer(uint32_t block_id) const;

    // This does NOT call retire. A separately coordinated owner must first
    // retire the exact generation and ensure all production reads are gated.
    // Drops this directory's writer lease but retains a retired tombstone.
    // Existing async owners can keep WriterContext until their work finishes.
    bool unbind_retired(uint32_t block_id);
    // Marks the tombstone Reclaimed only after this repository confirms the
    // exact generation's release callback completed. It remains fail-closed
    // until the runtime actually drops the dense block on truncate.
    bool confirm_reclaimed(uint32_t block_id);

    // Called before store mutations. A bound or retired block may not be
    // extended, shrunk, dropped, or reused before confirmed reclamation.
    void validate_append(const KvMemStore & store, uint32_t n_tokens) const;
    void validate_truncate(const KvMemStore & store, uint32_t token_pos) const;
    // Called by KvMemStore after a successful truncate. Only tombstones
    // whose old dense blocks no longer exist may be removed.
    void prune_dropped(const KvMemStore & store);

    size_t size() const noexcept { return entries_.size(); }
    size_t allocated_bytes() const noexcept;

private:
    Access acquire(uint32_t block_id, HistoricalKvRepository::LeaseKind kind) const;
    bool tail_span_matches(const KvMemStore & store) const;

    struct Entry {
        Binding binding;
        std::shared_ptr<HistoricalKvRepository> repository;
        uint64_t repository_id = 0;
        State state = State::Bound;
    };

    // KvMemStore appends monotonically, so the greatest bound ID also has
    // the greatest original span. Mutation guards inspect only that tail.
    std::map<uint32_t, Entry> entries_;
    uint64_t repository_id_ = 0;
};

} // namespace kvmem
