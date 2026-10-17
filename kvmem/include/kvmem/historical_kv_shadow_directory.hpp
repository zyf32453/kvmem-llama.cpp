#pragma once

#include "kvmem/historical_kv_repository.hpp"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>

namespace kvmem {

class KvMemStore;
class HistoricalKvRawStore;

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
        const std::shared_ptr<HistoricalKvRawStore> & storage() const { return storage_; }

    private:
        friend class HistoricalKvShadowDirectory;
        WriterContext(std::shared_ptr<HistoricalKvRepository> repository,
                      std::shared_ptr<HistoricalKvRawStore> storage)
            : repository_(std::move(repository)), storage_(std::move(storage)) {}

        std::shared_ptr<HistoricalKvRepository> repository_;
        std::shared_ptr<HistoricalKvRawStore> storage_;
        HistoricalKvRepository::Lease writer_;
    };

    // The admitting request owns this until every producer task has captured
    // its context, then returns it. Only those actual tasks keep the writer
    // alive thereafter. The directory never owns a producer reference.
    using Producer = std::shared_ptr<WriterContext>;

    // Finite request metadata: real map nodes, producer control blocks and
    // dense-table capacity are allocated before admission. The runtime must
    // outlive this token. Unused records may be destroyed by CPU cleanup;
    // installed nodes and captured producers have independent ownership.
    class AppendReservation {
    public:
        ~AppendReservation();
        AppendReservation(const AppendReservation &) = delete;
        AppendReservation & operator=(const AppendReservation &) = delete;
        uint32_t capacity() const noexcept;
        uint32_t remaining() const noexcept;
    private:
        friend class HistoricalKvShadowDirectory;
        struct Impl;
        explicit AppendReservation(std::unique_ptr<Impl> impl);
        std::unique_ptr<Impl> impl_;
    };
    std::unique_ptr<AppendReservation> prepare_append_reservation(
        KvMemStore & store, std::shared_ptr<HistoricalKvRepository> repository,
        std::shared_ptr<HistoricalKvRawStore> storage, uint32_t capacity);
    // Only swaps privately prepared storage into an unchanged, empty runtime.
    // No allocation or publication of block identities occurs here.
    bool install_append_reservation(KvMemStore & store, AppendReservation & reservation) noexcept;
    bool append_reserved_blocks(KvMemStore & store, AppendReservation & reservation,
        std::shared_ptr<HistoricalKvRepository> repository,
        std::shared_ptr<HistoricalKvRawStore> storage,
        std::vector<HistoricalKvRepository::Lease> & writers,
        std::vector<Producer> & producers);

    struct Binding {
        uint32_t block_id = 0;
        uint64_t logical_id = 0;
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
        uint64_t logical_id = 0;
        uint64_t original_begin = 0;
        uint32_t valid_tokens = 0;
    };

    struct Access {
        enum class Status { Unmanaged, Acquired, Denied };
        Status status = Status::Denied;
        HistoricalKvRepository::Lease lease;
        // Non-null for the new handle-addressed payload store. Captured
        // accesses never resolve storage again through a current block ID.
        std::shared_ptr<HistoricalKvRawStore> storage;

        explicit operator bool() const {
            return status == Status::Unmanaged ||
                   (status == Status::Acquired && static_cast<bool>(lease));
        }
    };

    // On success moves writer into the empty producer output. On failure
    // leaves both outputs with the caller, including on allocation failure.
    // Producer must be kept until all admitted tasks have captured it, then
    // reset (also on cancellation). One repository owns
    // this directory for the lifetime of its runtime, even when it is empty.
    bool bind_full_block(const KvMemStore & store, uint32_t block_id,
                         std::shared_ptr<HistoricalKvRepository> repository,
                         HistoricalKvHandle handle,
                         HistoricalKvRepository::Lease & writer,
                         Producer & producer,
                         std::shared_ptr<HistoricalKvRawStore> storage = {});
    // Bind an immutable admitted span, including a short/non-aligned tail.
    // Such spans require handle-addressed storage: positional legacy Raw
    // cannot represent them. Later appends start a new dense block, regardless
    // of whether this generation is Writing, Live or already retired.
    bool bind_block(const KvMemStore & store, uint32_t block_id,
                    std::shared_ptr<HistoricalKvRepository> repository,
                    HistoricalKvHandle handle,
                    HistoricalKvRepository::Lease & writer,
                    Producer & producer,
                    std::shared_ptr<HistoricalKvRawStore> storage);
    bool fixes_span(uint32_t block_id) const noexcept;
    // Atomically append already admitted, payload-backed spans and install
    // their identities before an adapter can capture work for those positions.
    // On failure all writers remain with the caller and the table is unchanged.
    // The caller still owns request-wide resource reservation and GPU admission.
    bool append_blocks(KvMemStore & store,
                       std::shared_ptr<HistoricalKvRepository> repository,
                       std::shared_ptr<HistoricalKvRawStore> storage,
                       std::vector<HistoricalKvRepository::Lease> & writers,
                       std::vector<Producer> & producers);
    // A bound entry survives task completion. Its writer_context is null
    // once the request and all producer tasks have returned their references.
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
    // Optional foreground observation only; never drops a task's writer.
    // Reclamation does not depend on calling this, or on another active.
    bool unbind_retired(uint32_t block_id);
    // Marks the tombstone Reclaimed only after this repository confirms the
    // exact generation's release callback completed. The old identity stays
    // fail-closed through position detachment and until table-slot reuse.
    bool confirm_reclaimed(uint32_t block_id);
    // At a foreground quiescent boundary, stop indexing an exactly reclaimed
    // span once every physical tier and producer has let go. The stable table
    // slot remains a tombstone until a later generation can reuse it.
    bool detach_reclaimed_span(KvMemStore & store, uint32_t block_id) noexcept;

    // Called before store mutations. Fixed spans may not be extended/shrunk;
    // append must roll over. Drop/reuse requires confirmed reclamation.
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
        // binding.writer_context is always empty in the stored descriptor.
        Binding binding;
        std::weak_ptr<WriterContext> producer;
        std::shared_ptr<HistoricalKvRepository> repository;
        std::shared_ptr<HistoricalKvRawStore> storage;
        uint64_t repository_id = 0;
        State state = State::Bound;
    };

    // KvMemStore appends monotonically, so the greatest bound ID also has
    // the greatest original span. Mutation guards inspect only that tail.
    std::map<uint32_t, Entry> entries_;
    uint64_t repository_id_ = 0;
    uint32_t reserved_append_capacity_ = 0;
    bool reserved_append_mode_ = false;
};

} // namespace kvmem
