#pragma once

#include "source_journal.hpp"
#include "snapshot.hpp"

namespace kvmem {

// Transport and decoded-payload bounds. Metadata/account/control and the
// caller's transport buffers remain bounded runtime overhead outside staging.
struct SourceImageLimits {
    uint64_t max_bytes = 0, max_tokens = 0, max_events = 0;
};

uint64_t source_image_bytes(const SourceState &, const SourceImageLimits &);
// Read-only, owner-serialized export. A failed transport may contain partial
// bytes; its owner must keep that object unpublished. No file/manifest commit
// or derived-state restore is performed by this codec.
uint64_t write_source_image(const SourceJournalStore &, SourceRef,
    const SourceImageLimits &, SnapshotWriter::Sink);

// Verified source DATA, never a restore capability. Origin SourceRefs describe
// the old snapshot, not a current in-process source. A durable authority must
// check retirement/forget/model fences before any future import/publication.
class SourceImage {
public:
    SourceImage(SourceImage && other) noexcept
        : origin_(std::exchange(other.origin_, {})), tokens_(std::move(other.tokens_)),
          events_(std::move(other.events_)), image_bytes_(std::exchange(other.image_bytes_, 0)) {}
    SourceImage & operator=(SourceImage && other) noexcept {
        if (this != &other) {
            tokens_ = std::move(other.tokens_); events_ = std::move(other.events_);
            origin_ = std::exchange(other.origin_, {}); image_bytes_ = std::exchange(other.image_bytes_, 0);
        }
        return *this;
    }
    SourceImage(const SourceImage &) = delete;
    SourceImage & operator=(const SourceImage &) = delete;
    const SourceState & origin() const noexcept { return origin_; }
    uint64_t image_bytes() const noexcept { return image_bytes_; }
    uint64_t staging_bytes() const noexcept { return tokens_.size() + events_.size(); }
    void read_tokens(uint64_t begin, int32_t * out, size_t count) const;
    SourceEvent event(uint64_t index) const;
private:
    friend SourceImage prepare_source_image(const SourceJournalStore &, SourceRef,
        const SourceImageLimits &, HostKvStorage &);
    friend SourceImage read_source_image(SnapshotReader::Source, uint64_t,
        const SourceConfiguration &, const SourceImageLimits &, HostKvStorage &);
    SourceImage(SourceState origin, HostKvRecord tokens, HostKvRecord events, uint64_t bytes)
        : origin_(origin), tokens_(std::move(tokens)), events_(std::move(events)), image_bytes_(bytes) {}
    SourceState origin_;
    HostKvRecord tokens_, events_;
    uint64_t image_bytes_ = 0;
};

// Owner-serialized RAM copy, with real shared staging quota before either
// allocation. A later writer can hold this immutable payload without reading
// a mutable journal or holding its Engine lock during file I/O.
SourceImage prepare_source_image(const SourceJournalStore &, SourceRef,
    const SourceImageLimits &, HostKvStorage & staging);
uint64_t write_source_image(const SourceImage &, const SourceImageLimits &, SnapshotWriter::Sink);

// Reserves both real RAM buffers before decoding. No mutable/raw borrow escapes;
// only a checksum-verified, structurally valid object can be returned. Failure
// releases its reservation/buffers and does not alter any source directory.
SourceImage read_source_image(SnapshotReader::Source, uint64_t bytes,
    const SourceConfiguration & expected_configuration, const SourceImageLimits &, HostKvStorage & staging);

} // namespace kvmem
