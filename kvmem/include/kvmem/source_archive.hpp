#pragma once

#include "source_image.hpp"
#include "spill_io.hpp"
#include "historical_kv_auxiliary.hpp"

namespace kvmem {
namespace detail { struct SourceArchiveState; }

// Stable archive identity; never a SourceRef, physical KV ID or restore permit.
struct SourceArchiveKey {
    std::array<uint8_t, 16> space{};
    uint64_t id = 0;
    bool operator==(const SourceArchiveKey & other) const noexcept {
        return space == other.space && id == other.id;
    }
};
struct SourceArchiveLimits {
    uint64_t disk_bytes = 0, metadata_bytes = 0;
    size_t max_sources = 0, max_objects = 0;
    SourceImageLimits image;
};
struct SourceArchiveSnapshot {
    SourceArchiveKey key, parent;
    uint64_t object = 0, bytes = 0;
    SourceState origin; // saved frontier, not the current journal frontier
};

// Move-only owner-issued receipt. Keeping it alive retains its bounded object
// slot and disk reservation, even after cancellation/forget. Completion means
// only that background I/O finished; publish() must still revalidate authority.
class SourceArchiveWrite {
public:
    SourceArchiveWrite() = default;
    ~SourceArchiveWrite();
    SourceArchiveWrite(SourceArchiveWrite &&) noexcept;
    SourceArchiveWrite & operator=(SourceArchiveWrite &&) noexcept;
    SourceArchiveWrite(const SourceArchiveWrite &) = delete;
    SourceArchiveWrite & operator=(const SourceArchiveWrite &) = delete;
    StorageTaskStatus status() const noexcept;
    void wait() const;
    bool cancel() const;
    SourceArchiveKey key() const noexcept;
private:
    friend class SourceArchive;
    SourceArchiveWrite(std::shared_ptr<detail::SourceArchiveState>, size_t);
    void reset() noexcept;
    std::shared_ptr<detail::SourceArchiveState> state_;
    size_t slot_ = 0;
};

class SourceArchiveRead {
public:
    ~SourceArchiveRead();
    SourceArchiveRead(SourceArchiveRead &&) noexcept;
    SourceArchiveRead & operator=(SourceArchiveRead &&) noexcept;
    SourceArchiveRead(const SourceArchiveRead &) = delete;
    SourceArchiveRead & operator=(const SourceArchiveRead &) = delete;
    const SourceImage & image() const noexcept { return image_; }
    SourceArchiveKey key() const noexcept;
private:
    friend class SourceArchive;
    SourceArchiveRead(std::shared_ptr<detail::SourceArchiveState>, size_t, SourceImage);
    void reset() noexcept;
    std::shared_ptr<detail::SourceArchiveState> state_;
    size_t slot_ = 0;
    SourceImage image_;
};

// Linux local-filesystem archive, one serialized owner. Only immutable blob I/O
// runs in StorageIoQueue; methods below may perform blocking metadata/file I/O
// and belong outside a decode boundary. No Engine/model publication is done here.
// Existing directories MUST be opened, never implicitly recreated or scanned to
// reconstruct permissions. create() requires a new directory. Configuration and
// max_sources are fixed for a namespace. All source mutations/forget and calls
// on this owner must be serialized by the adapter.
class SourceArchive {
public:
    struct Stats {
        uint64_t capacity = 0, control_reserved = 0, object_bytes = 0;
        uint64_t charged = 0, peak = 0, metadata_bytes = 0;
        size_t sources = 0, objects = 0, leased_objects = 0;
        bool healthy = false;
    };
    struct Collection {
        uint64_t quota_released = 0, file_bytes_removed = 0;
        size_t objects = 0;
    };
    // Tables use the existing auxiliary allocator. An adapter can supply its
    // shared M account; outstanding receipts retain charges after owner exit.
    static SourceArchive create(const std::filesystem::path &, const SourceConfiguration &, SourceArchiveLimits,
        std::shared_ptr<HistoricalKvAuxBudget> auxiliary = {});
    static SourceArchive open(const std::filesystem::path &, const SourceConfiguration &, SourceArchiveLimits,
        std::shared_ptr<HistoricalKvAuxBudget> auxiliary = {});
    SourceArchive(SourceArchive &&) noexcept = default;
    SourceArchive & operator=(SourceArchive &&) noexcept = default;
    SourceArchive(const SourceArchive &) = delete;
    SourceArchive & operator=(const SourceArchive &) = delete;

    SourceArchiveKey enroll(const SourceJournalStore &, SourceRef);
    SourceArchiveWrite save(StorageIoQueue &, const SourceJournalStore &, SourceRef, HostKvStorage & staging);
    void publish(const SourceArchiveWrite &, const SourceJournalStore &);
    SourceArchiveRead read(SourceArchiveKey, HostKvStorage & staging);
    // Required again immediately before an adapter imports/rebuilds from read().
    // A decoded image by itself is data and cannot replace this receipt check.
    void validate(const SourceArchiveRead &) const;
    // Explicit restart import: validate CURRENT receipt, atomically capture a
    // fresh S root with computed0, then bind this durable key to it. A live
    // binding returns Pending; no implicit replacement, alias or decoded-origin
    // matching. Durable lineage remains in the archive, not old RAM parent refs.
    SourceRef import_source(const SourceArchiveRead &, SourceJournalStore &);
    // Explicit RAM eviction, not durable forgetting. Every selected current
    // source must have a complete current published copy, and the selection
    // must include all its RAM descendants. Verify all files/fences first,
    // then nonallocating unbind + erase RAM leaves before parents. The adapter
    // must preflight its native aliases and release them in this same serialized
    // control operation, before any new execution. Archive keys/read authority
    // remain valid; old RAM preparations and old-binding writers do not.
    size_t cool_sources(SourceJournalStore &, const SourceRef *, size_t count);
    size_t slots() const noexcept;
    std::optional<SourceArchiveKey> key_for(const SourceJournalStore &, SourceRef) const;
    std::optional<SourceRef> bound_source(SourceArchiveKey, const SourceJournalStore &) const;
    bool family_contains(SourceArchiveKey ancestor, SourceArchiveKey member) const;
    std::optional<SourceArchiveSnapshot> inspect(size_t slot) const;
    // Durable archive-family revocation ONLY. An Engine adapter must then revoke
    // its online aliases/source family; this is not system-wide ForgetSource.
    size_t forget(SourceArchiveKey);
    // Unpublished/replaced/revoked files only, after jobs AND handles finish.
    // Refunds and actual file bytes are distinct: refused/partial jobs may have
    // reserved more than they wrote. Both are reported only after directory sync.
    Collection collect();
    Stats stats() const noexcept;
private:
    explicit SourceArchive(std::shared_ptr<detail::SourceArchiveState> state) : state_(std::move(state)) {}
    std::shared_ptr<detail::SourceArchiveState> state_;
};
} // namespace kvmem
