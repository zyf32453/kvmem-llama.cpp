#include "kvmem/historical_kv_repository.hpp"

#include <algorithm>
#include <atomic>
#include <limits>
#include <stdexcept>
#include <utility>

namespace kvmem {
namespace {

// One atomic word: phase (3 bits), then four 15-bit reference counts for
// all-Host, packed-KV-only, Mean-K-only, and identity/GPU-only. Admission is
// bounded and rejects a saturated counter. It registers future dependencies in the
// SAME CAS as execution admission, so retirement cannot release payload in
// a gap between obtaining permission and publishing a separate Host count.
// Reserving is internal: it keeps an unpublished slot invisible while its
// generation, immutable metadata, and release callback are installed.
// Publication sets bit 0. A retained writer can observe only Writing, Live
// or Retired; that bit is already set in both terminal cases. This lets its
// final producer publish once without losing the transition to refcount CAS
// contention, and without ever undoing concurrent retirement.
enum class Phase : uint64_t {
    Ready = 0, Reserving = 1, Writing = 2, Live = 3,
    Releasing = 4, Reserved = 6, Retired = 7
};
constexpr uint64_t kPhaseMask = 7;
constexpr uint64_t kPublishBit = 1;
static_assert((uint64_t(Phase::Writing) | kPublishBit) == uint64_t(Phase::Live) &&
              (uint64_t(Phase::Live) | kPublishBit) == uint64_t(Phase::Live) &&
              (uint64_t(Phase::Retired) | kPublishBit) == uint64_t(Phase::Retired),
              "publication must preserve every non-Writing writer phase");
constexpr uint64_t kHostOne = 8;
constexpr uint64_t kKvOne = uint64_t(1) << 18;
constexpr uint64_t kMeanOne = uint64_t(1) << 33;
constexpr uint64_t kOtherOne = uint64_t(1) << 48;
constexpr uint8_t kWriterKv = 1, kWriterMean = 2, kWriterAll = kWriterKv | kWriterMean;
constexpr uint64_t kMaxRefs = (uint64_t(1) << 15) - 1;
constexpr unsigned kCasAttempts = 16;
enum class HostRelease : uint8_t { Bound, Releasing, Released };
using HostAccess = HistoricalKvRepository::HostAccess;
using HostResource = HistoricalKvRepository::HostResource;

Phase phase_of(uint64_t control) { return static_cast<Phase>(control & kPhaseMask); }
unsigned ref_shift(HostAccess access) {
    switch (access) {
        case HostAccess::All: return 3;
        case HostAccess::PackedKv: return 18;
        case HostAccess::MeanK: return 33;
        case HostAccess::None: return 48;
    }
    return 48;
}
uint64_t ref_one(HostAccess access) { return uint64_t(1) << ref_shift(access); }
uint64_t access_refs_of(uint64_t control, HostAccess access) { return (control >> ref_shift(access)) & kMaxRefs; }
uint64_t host_refs_of(uint64_t control) {
    return access_refs_of(control, HostAccess::All) + access_refs_of(control, HostAccess::PackedKv) +
           access_refs_of(control, HostAccess::MeanK);
}
uint64_t resource_refs_of(uint64_t control, HostResource resource) {
    return access_refs_of(control, HostAccess::All) +
        access_refs_of(control, resource == HostResource::PackedKv ? HostAccess::PackedKv : HostAccess::MeanK);
}
uint64_t other_refs_of(uint64_t control) { return access_refs_of(control, HostAccess::None); }
uint64_t refs_of(uint64_t control) { return host_refs_of(control) + other_refs_of(control); }
uint64_t code(Phase phase, uint64_t host = 0, uint64_t other = 0) {
    return host * kHostOne | other * kOtherOne | static_cast<uint64_t>(phase);
}
uint32_t phase_bit(Phase phase) { return 1u << static_cast<unsigned>(phase); }
constexpr uint32_t kReadable = (1u << static_cast<unsigned>(Phase::Writing)) |
                               (1u << static_cast<unsigned>(Phase::Live)) |
                               (1u << static_cast<unsigned>(Phase::Retired));

std::atomic<uint64_t> next_incarnation{1};
std::atomic<uint64_t> next_repository_id{1};

uint64_t allocate_incarnation() {
    uint64_t current = next_incarnation.load(std::memory_order_relaxed);
    for (;;) {
        if (current == std::numeric_limits<uint64_t>::max())
            throw std::overflow_error("historical KV incarnation exhausted");
        if (next_incarnation.compare_exchange_weak(current, current + 1,
                std::memory_order_acq_rel, std::memory_order_relaxed)) return current;
    }
}

uint64_t allocate_repository_id() {
    uint64_t current = next_repository_id.load(std::memory_order_relaxed);
    for (;;) {
        if (current == std::numeric_limits<uint64_t>::max())
            throw std::overflow_error("historical KV repository IDs exhausted");
        if (next_repository_id.compare_exchange_weak(current, current + 1,
                std::memory_order_acq_rel, std::memory_order_relaxed)) return current;
    }
}

HistoricalKvLife public_life(Phase phase) {
    switch (phase) {
        case Phase::Writing: return HistoricalKvLife::Writing;
        case Phase::Live: return HistoricalKvLife::Live;
        case Phase::Retired: return HistoricalKvLife::Retired;
        case Phase::Releasing: return HistoricalKvLife::Releasing;
        case Phase::Ready:
        case Phase::Reserved: // Private entitlement; never emitted as a block.
        case Phase::Reserving: return HistoricalKvLife::Ready;
    }
    return HistoricalKvLife::Ready;
}

} // namespace

struct HistoricalKvRepository::Shared {
    struct Slot {
        std::atomic<uint64_t> control{code(Phase::Ready)};
        std::atomic<uint64_t> generation{0};
        std::atomic<uint64_t> last_use_seq{0};
        std::atomic<uint64_t> exec_refs{0};
        std::atomic<bool> release_busy{false};
        std::atomic<HostRelease> host_release{HostRelease::Bound};
        std::atomic<HostRelease> kv_release{HostRelease::Bound};
        std::atomic<HostRelease> mean_release{HostRelease::Bound};
        std::atomic<HostRelease> nvme_release{HostRelease::Bound};
        std::atomic<HostRelease> pinned_release{HostRelease::Bound};
        // A resource writer owns one identity ref plus one claim per open
        // component. Closing a claim never invalidates its immutable handle.
        std::atomic<uint8_t> writer_resources{0};
        uint64_t logical_id = 0;
        uint64_t original_begin = 0;
        uint64_t birth_seq = 0;
        uint64_t content_version = 1;
        uint64_t payload_bytes = 0;
        uint32_t valid_tokens = 0;
        ReleaseCallback release;
        ResourceRelease resources;
    };

    Shared(uint32_t n, uint64_t session, uint64_t sink, uint64_t recent)
        : capacity(n), incarnation(session), repository_id(allocate_repository_id()),
          slots(new Slot[n]), sink_tokens(sink), recent_tokens(recent) {}

    // Publication is append-only within an incarnation. A stale observation
    // protects extra recent history rather than reopening an expired window.
    void observe_end(uint64_t begin, uint32_t count) noexcept {
        const uint64_t end = begin > UINT64_MAX - count ? UINT64_MAX : begin + count;
        auto old = history_end.load(std::memory_order_relaxed);
        for (unsigned attempt = 0; old < end && attempt < kCasAttempts; ++attempt)
            if (history_end.compare_exchange_weak(old, end, std::memory_order_release,
                                                  std::memory_order_relaxed)) break;
    }

    bool try_pin(uint32_t index, uint32_t phases, HostAccess access = HostAccess::None) const {
        auto & control = slots[index].control;
        uint64_t observed = control.load(std::memory_order_acquire);
        for (unsigned attempt = 0; attempt < kCasAttempts; ++attempt) {
            if (!(phases & phase_bit(phase_of(observed))) ||
                access_refs_of(observed, access) == kMaxRefs)
                return false;
            if (control.compare_exchange_weak(observed, observed + ref_one(access),
                                              std::memory_order_acq_rel,
                                              std::memory_order_acquire)) return true;
        }
        return false;
    }

    void unpin(uint32_t index, HostAccess access) const noexcept {
        slots[index].control.fetch_sub(ref_one(access), std::memory_order_acq_rel);
    }

    bool matches(uint32_t index, HistoricalKvHandle handle) const {
        return handle.incarnation == incarnation && handle.slot == index &&
               handle.generation != 0 &&
               slots[index].generation.load(std::memory_order_acquire) == handle.generation;
    }

    HistoricalKvBlockMeta copy_meta(uint32_t index, uint64_t control) const {
        const auto & slot = slots[index];
        HistoricalKvBlockMeta out;
        out.handle = {incarnation, index, slot.generation.load(std::memory_order_acquire)};
        out.logical_id = slot.logical_id;
        out.original_begin = slot.original_begin;
        out.valid_tokens = slot.valid_tokens;
        out.birth_seq = slot.birth_seq;
        out.last_use_seq = slot.last_use_seq.load(std::memory_order_acquire);
        out.content_version = slot.content_version;
        out.payload_bytes = slot.payload_bytes;
        out.life = public_life(phase_of(control));
        // This observation includes our short-lived metadata guard.
        const auto writer_resources = slot.writer_resources.load(std::memory_order_acquire);
        const uint64_t extra_writer_claims = bool(writer_resources & kWriterKv) + bool(writer_resources & kWriterMean);
        const uint64_t raw_refs = refs_of(control);
        const uint64_t refs = raw_refs > extra_writer_claims ? raw_refs - extra_writer_claims : 0;
        out.references = static_cast<uint32_t>(refs > 0 ?
            std::min<uint64_t>(refs - 1, std::numeric_limits<uint32_t>::max()) : 0);
        out.active = slot.exec_refs.load(std::memory_order_acquire) != 0;
        out.sealed = out.life == HistoricalKvLife::Live;
        const auto end = history_end.load(std::memory_order_acquire);
        const auto recent_begin = end > recent_tokens ? end - recent_tokens : 0;
        const auto block_end = slot.original_begin > UINT64_MAX - slot.valid_tokens
            ? UINT64_MAX : slot.original_begin + slot.valid_tokens;
        out.mandatory = slot.original_begin < sink_tokens ||
            (recent_tokens && block_end > recent_begin);
        out.host_references = static_cast<uint32_t>(host_refs_of(control));
        out.host_released = slot.host_release.load(std::memory_order_acquire) == HostRelease::Released;
        out.packed_kv_references = static_cast<uint32_t>(resource_refs_of(control, HostResource::PackedKv));
        out.mean_k_references = static_cast<uint32_t>(resource_refs_of(control, HostResource::MeanK));
        out.packed_kv_released = slot.kv_release.load(std::memory_order_acquire) == HostRelease::Released;
        out.mean_k_released = slot.mean_release.load(std::memory_order_acquire) == HostRelease::Released;
        out.nvme_released = slot.nvme_release.load(std::memory_order_acquire) == HostRelease::Released;
        out.nvme_bytes = out.nvme_released ? 0 : slot.resources.nvme_bytes;
        out.pinned_released = slot.pinned_release.load(std::memory_order_acquire) == HostRelease::Released;
        out.pinned_bytes = out.pinned_released ? 0 : slot.resources.pinned_bytes;
        if (out.packed_kv_released) out.payload_bytes -= slot.resources.packed_kv_bytes;
        if (out.mean_k_released) out.payload_bytes -= slot.resources.mean_k_bytes;
        if (out.host_released) out.payload_bytes = 0;
        return out;
    }

    const uint32_t capacity;
    const uint64_t incarnation;
    const uint64_t repository_id;
    std::unique_ptr<Slot[]> slots;
    const uint64_t sink_tokens, recent_tokens;
    std::atomic<uint64_t> history_end{0};
    std::atomic<uint32_t> cursor{0};
    std::atomic<bool> closing{false};
    std::atomic<bool> gc_owner_claimed{false};
    std::atomic<uint64_t> operations{0};
};

HistoricalKvRepository::Lease::Lease(std::shared_ptr<Shared> shared,
                                     HistoricalKvHandle handle, LeaseKind kind, HostAccess access, bool split_writer)
    : shared_(std::move(shared)), handle_(handle), kind_(kind), access_(access), split_writer_(split_writer) {}

HistoricalKvRepository::Lease::~Lease() { reset(); }

HistoricalKvRepository::Lease::Lease(Lease && other) noexcept
    : shared_(std::move(other.shared_)), handle_(other.handle_), kind_(other.kind_),
      counted_exec_(other.counted_exec_), counted_operation_(other.counted_operation_),
      access_(other.access_), split_writer_(other.split_writer_), writer_resource_(std::move(other.writer_resource_)) {
    other.kind_ = LeaseKind::Read;
    other.counted_exec_ = false;
    other.counted_operation_ = false;
    other.access_ = HostAccess::None;
    other.split_writer_ = false;
}

HistoricalKvRepository::Lease & HistoricalKvRepository::Lease::operator=(Lease && other) noexcept {
    if (this != &other) {
        reset();
        shared_ = std::move(other.shared_);
        handle_ = other.handle_;
        kind_ = other.kind_;
        counted_exec_ = other.counted_exec_;
        counted_operation_ = other.counted_operation_;
        access_ = other.access_;
        split_writer_ = other.split_writer_;
        writer_resource_ = std::move(other.writer_resource_);
        other.kind_ = LeaseKind::Read;
        other.counted_exec_ = false;
        other.counted_operation_ = false;
        other.access_ = HostAccess::None;
        other.split_writer_ = false;
    }
    return *this;
}

std::optional<HistoricalKvBlockMeta> HistoricalKvRepository::Lease::metadata() const {
    if (!shared_ || handle_.slot >= shared_->capacity ||
        !shared_->matches(handle_.slot, handle_)) return std::nullopt;
    const auto control = shared_->slots[handle_.slot].control.load(std::memory_order_acquire);
    if (!(kReadable & phase_bit(phase_of(control)))) return std::nullopt;
    return shared_->copy_meta(handle_.slot, control);
}

uint64_t HistoricalKvRepository::Lease::repository_id() const noexcept {
    return shared_ ? shared_->repository_id : 0;
}

bool HistoricalKvRepository::Lease::has_host_access() const noexcept {
    if (!shared_) return false;
    return split_writer_ ? shared_->slots[handle_.slot].writer_resources.load(std::memory_order_acquire) == kWriterAll
                         : access_ == HostAccess::All;
}

bool HistoricalKvRepository::Lease::has_packed_kv_access() const noexcept {
    if (!shared_) return false;
    return split_writer_ ? (shared_->slots[handle_.slot].writer_resources.load(std::memory_order_acquire) & kWriterKv) != 0
                         : access_ == HostAccess::All || access_ == HostAccess::PackedKv;
}

bool HistoricalKvRepository::Lease::has_mean_k_access() const noexcept {
    if (!shared_) return false;
    return split_writer_ ? (shared_->slots[handle_.slot].writer_resources.load(std::memory_order_acquire) & kWriterMean) != 0
                         : access_ == HostAccess::All || access_ == HostAccess::MeanK;
}

void HistoricalKvRepository::Lease::reset() noexcept {
    // Drop external producer rights before repository identity. Installed
    // allocations remain owned by their storage binding; an uninstalled
    // private allocation may instead roll back here.
    writer_resource_.reset();
    if (shared_) {
        if (counted_exec_)
            shared_->slots[handle_.slot].exec_refs.fetch_sub(1, std::memory_order_acq_rel);
        if (split_writer_) {
            auto & slot = shared_->slots[handle_.slot];
            const auto resources = slot.writer_resources.exchange(0, std::memory_order_acq_rel);
            slot.control.fetch_sub(kOtherOne + ((resources & kWriterKv) ? kKvOne : 0) +
                                   ((resources & kWriterMean) ? kMeanOne : 0), std::memory_order_acq_rel);
        } else {
            shared_->unpin(handle_.slot, access_);
        }
        if (counted_operation_) shared_->operations.fetch_sub(1, std::memory_order_acq_rel);
        shared_.reset();
    }
    kind_ = LeaseKind::Read;
    counted_exec_ = false;
    counted_operation_ = false;
    access_ = HostAccess::None;
    split_writer_ = false;
}

void HistoricalKvRepository::Lease::count_operation() noexcept {
    shared_->operations.fetch_add(1, std::memory_order_acq_rel);
    counted_operation_ = true;
    if (kind_ == LeaseKind::Execution || kind_ == LeaseKind::GpuExecution) {
        shared_->slots[handle_.slot].exec_refs.fetch_add(1, std::memory_order_acq_rel);
        counted_exec_ = true;
    }
}

void HistoricalKvRepository::Lease::disarm() noexcept {
    shared_.reset();
    kind_ = LeaseKind::Read;
    counted_exec_ = false;
    counted_operation_ = false;
    access_ = HostAccess::None;
    split_writer_ = false;
}

HistoricalKvRepository::HistoricalKvRepository(uint32_t capacity, uint64_t incarnation,
                                               uint64_t sink_tokens, uint64_t recent_tokens)
    : shared_(std::make_shared<Shared>(capacity,
          incarnation ? incarnation : allocate_incarnation(), sink_tokens, recent_tokens)) {
    if (!capacity) throw std::invalid_argument("historical KV capacity must be nonzero");
    if (!shared_->slots[0].control.is_lock_free())
        throw std::runtime_error("historical KV control atomic is not lock-free on this host");
}

HistoricalKvRepository::~HistoricalKvRepository() { close(); }

HistoricalKvRepository::Reservation::~Reservation() { reset(); }

HistoricalKvRepository::Reservation::Reservation(Reservation && other) noexcept
    : shared_(std::move(other.shared_)), handles_(std::move(other.handles_)),
      consumed_(other.consumed_) {
    other.consumed_ = 0;
}

HistoricalKvRepository::Reservation &
HistoricalKvRepository::Reservation::operator=(Reservation && other) noexcept {
    if (this != &other) {
        reset();
        shared_ = std::move(other.shared_);
        handles_ = std::move(other.handles_);
        consumed_ = other.consumed_;
        other.consumed_ = 0;
    }
    return *this;
}

uint32_t HistoricalKvRepository::Reservation::remaining() const noexcept {
    return shared_ ? static_cast<uint32_t>(handles_.size()) - consumed_ : 0;
}

std::optional<HistoricalKvHandle>
HistoricalKvRepository::Reservation::next_handle() const noexcept {
    return remaining() ? std::optional<HistoricalKvHandle>(handles_[consumed_]) : std::nullopt;
}

uint64_t HistoricalKvRepository::Reservation::repository_id() const noexcept {
    return shared_ ? shared_->repository_id : 0;
}

void HistoricalKvRepository::Reservation::reset() noexcept {
    if (shared_) {
        for (size_t i = consumed_; i < handles_.size(); ++i) {
            // No other owner can consume or retire a Reserved slot. Its
            // generation was already advanced, so stale handles stay stale.
            shared_->slots[handles_[i].slot].control.store(code(Phase::Ready), std::memory_order_release);
        }
    }
    shared_.reset();
    handles_.clear();
    consumed_ = 0;
}

std::optional<HistoricalKvRepository::Reservation>
HistoricalKvRepository::try_reserve_slots(uint32_t count) {
    const auto shared = shared_;
    if (count > shared->capacity || shared->closing.load(std::memory_order_acquire)) return std::nullopt;
    Reservation reservation;
    reservation.handles_.reserve(count); // All allocation precedes slot mutation.
    reservation.shared_ = shared;
    if (!count) return reservation;
    const uint32_t start = shared->cursor.fetch_add(1, std::memory_order_relaxed);
    for (uint32_t offset = 0; offset < shared->capacity; ++offset) {
        const uint32_t index = static_cast<uint32_t>((uint64_t(start) + offset) % shared->capacity);
        auto & slot = shared->slots[index];
        uint64_t ready = code(Phase::Ready);
        if (!slot.control.compare_exchange_strong(ready, code(Phase::Reserving),
                std::memory_order_acq_rel, std::memory_order_acquire)) continue;
        const uint64_t previous = slot.generation.load(std::memory_order_relaxed);
        if (previous == std::numeric_limits<uint64_t>::max()) {
            slot.control.store(code(Phase::Ready), std::memory_order_release);
            continue;
        }
        const HistoricalKvHandle handle{shared->incarnation, index, previous + 1};
        slot.generation.store(handle.generation, std::memory_order_release);
        reservation.handles_.push_back(handle);
        slot.control.store(code(Phase::Reserved), std::memory_order_release);
        if (reservation.handles_.size() == count) {
            // close() stops admission, including a reservation acquired while
            // shutdown began. The local destructor returns all private slots.
            if (shared->closing.load(std::memory_order_acquire)) return std::nullopt;
            return reservation;
        }
    }
    return std::nullopt;
}

std::optional<HistoricalKvRepository::Lease> HistoricalKvRepository::begin_reserved_write(
        Reservation & reservation, uint64_t logical_id, uint64_t original_begin,
        uint32_t valid_tokens, uint64_t birth_seq, uint64_t payload_bytes,
        ReleaseCallback release) {
    return begin_reserved_write_impl(reservation, logical_id, original_begin, valid_tokens,
        birth_seq, payload_bytes, std::move(release), {});
}

std::optional<HistoricalKvRepository::Lease> HistoricalKvRepository::begin_reserved_write(
        Reservation & reservation, uint64_t logical_id, uint64_t original_begin,
        uint32_t valid_tokens, uint64_t birth_seq, uint64_t payload_bytes, ResourceRelease release) {
    if (!release.callback || release.packed_kv_bytes > payload_bytes ||
        release.mean_k_bytes != payload_bytes - release.packed_kv_bytes)
        throw std::invalid_argument("historical KV resource release bytes/callback mismatch");
    return begin_reserved_write_impl(reservation, logical_id, original_begin, valid_tokens,
        birth_seq, payload_bytes, {}, std::move(release));
}

std::optional<HistoricalKvRepository::Lease> HistoricalKvRepository::begin_reserved_write_with_gpu(
        Reservation & reservation, uint64_t logical_id, uint64_t original_begin,
        uint32_t valid_tokens, uint64_t birth_seq, uint64_t payload_bytes,
        ResourceRelease release, Lease & gpu_execution) {
    if (gpu_execution) return std::nullopt;
    if (!release.callback || release.packed_kv_bytes > payload_bytes ||
        release.mean_k_bytes != payload_bytes - release.packed_kv_bytes)
        throw std::invalid_argument("historical KV resource release bytes/callback mismatch");
    return begin_reserved_write_impl(reservation, logical_id, original_begin, valid_tokens,
        birth_seq, payload_bytes, {}, std::move(release), &gpu_execution);
}

std::optional<HistoricalKvRepository::Lease> HistoricalKvRepository::begin_reserved_write_impl(
        Reservation & reservation, uint64_t logical_id, uint64_t original_begin,
        uint32_t valid_tokens, uint64_t birth_seq, uint64_t payload_bytes,
        ReleaseCallback release, ResourceRelease resources, Lease * gpu_execution) {
    if (!release && !resources.callback) throw std::invalid_argument("historical KV release callback is required");
    const auto shared = shared_;
    if (reservation.shared_.get() != shared.get() || !reservation.remaining() ||
        !valid_tokens || original_begin > std::numeric_limits<uint64_t>::max() - valid_tokens ||
        shared->closing.load(std::memory_order_acquire)) return std::nullopt;
    const auto handle = reservation.handles_[reservation.consumed_];
    auto & slot = shared->slots[handle.slot];
    // Exclusive reservation ownership makes the metadata write race-free.
    // Reserved is hidden from metadata readers, new leases and the policy.
    slot.logical_id = logical_id;
    slot.original_begin = original_begin;
    slot.valid_tokens = valid_tokens;
    slot.birth_seq = birth_seq;
    slot.content_version = 1;
    slot.payload_bytes = payload_bytes;
    slot.last_use_seq.store(birth_seq, std::memory_order_relaxed);
    slot.exec_refs.store(0, std::memory_order_relaxed);
    slot.host_release.store(HostRelease::Bound, std::memory_order_relaxed);
    slot.kv_release.store(HostRelease::Bound, std::memory_order_relaxed);
    slot.mean_release.store(HostRelease::Bound, std::memory_order_relaxed);
    slot.nvme_release.store(resources.nvme_bytes ? HostRelease::Bound : HostRelease::Released,
                            std::memory_order_relaxed);
    slot.pinned_release.store(resources.pinned_bytes ? HostRelease::Bound : HostRelease::Released,
                              std::memory_order_relaxed);
    slot.resources = std::move(resources);
    const bool split_writer = bool(slot.resources.callback);
    slot.writer_resources.store(split_writer ? kWriterAll : 0, std::memory_order_relaxed);
    slot.release = std::move(release);
    ++reservation.consumed_;
    Lease writer(shared, handle, LeaseKind::Writer, split_writer ? HostAccess::None : HostAccess::All, split_writer);
    writer.writer_resource_ = std::move(slot.resources.writer_resource);
    writer.count_operation();
    if (gpu_execution) {
        *gpu_execution = Lease(shared, handle, LeaseKind::GpuExecution, HostAccess::None);
        gpu_execution->count_operation();
    }
    // Publish BOTH permissions in the same phase transition. In particular,
    // GC cannot retire a visible Writing slot between writer admission and
    // the producer window's execution admission.
    const uint64_t writer_refs = split_writer ?
        code(Phase::Writing, 0, 1) + kKvOne + kMeanOne : code(Phase::Writing, 1);
    shared->observe_end(original_begin, valid_tokens);
    slot.control.store(writer_refs + (gpu_execution ? kOtherOne : 0), std::memory_order_release);
    return writer;
}

std::optional<HistoricalKvRepository::Lease> HistoricalKvRepository::begin_write(
    uint64_t logical_id, uint64_t original_begin, uint32_t valid_tokens,
    uint64_t birth_seq, uint64_t payload_bytes, ReleaseCallback release) {
    return begin_write_impl(logical_id, original_begin, valid_tokens, birth_seq,
        payload_bytes, std::move(release), {});
}

std::optional<HistoricalKvRepository::Lease> HistoricalKvRepository::begin_write(
    uint64_t logical_id, uint64_t original_begin, uint32_t valid_tokens,
    uint64_t birth_seq, uint64_t payload_bytes, ResourceRelease release) {
    if (!release.callback || release.packed_kv_bytes > payload_bytes ||
        release.mean_k_bytes != payload_bytes - release.packed_kv_bytes)
        throw std::invalid_argument("historical KV resource release bytes/callback mismatch");
    return begin_write_impl(logical_id, original_begin, valid_tokens, birth_seq,
        payload_bytes, {}, std::move(release));
}

std::optional<HistoricalKvRepository::Lease> HistoricalKvRepository::begin_write_impl(
    uint64_t logical_id, uint64_t original_begin, uint32_t valid_tokens,
    uint64_t birth_seq, uint64_t payload_bytes, ReleaseCallback release, ResourceRelease resources) {
    if (!release && !resources.callback) throw std::invalid_argument("historical KV release callback is required");
    const auto shared = shared_;
    if (shared->closing.load(std::memory_order_acquire)) return std::nullopt;
    const uint32_t start = shared->cursor.fetch_add(1, std::memory_order_relaxed);
    for (uint32_t offset = 0; offset < shared->capacity; ++offset) {
        const uint32_t index = static_cast<uint32_t>(
            (static_cast<uint64_t>(start) + offset) % shared->capacity);
        auto & slot = shared->slots[index];
        uint64_t ready = code(Phase::Ready);
        if (!slot.control.compare_exchange_strong(ready, code(Phase::Reserving),
                                                  std::memory_order_acq_rel,
                                                  std::memory_order_acquire)) continue;
        const uint64_t previous = slot.generation.load(std::memory_order_relaxed);
        if (previous == std::numeric_limits<uint64_t>::max()) {
            slot.control.store(code(Phase::Ready), std::memory_order_release);
            continue;
        }
        const HistoricalKvHandle handle{shared->incarnation, index, previous + 1};
        slot.logical_id = logical_id;
        slot.original_begin = original_begin;
        slot.valid_tokens = valid_tokens;
        slot.birth_seq = birth_seq;
        slot.content_version = 1;
        slot.payload_bytes = payload_bytes;
        slot.last_use_seq.store(birth_seq, std::memory_order_relaxed);
        slot.exec_refs.store(0, std::memory_order_relaxed);
        slot.host_release.store(HostRelease::Bound, std::memory_order_relaxed);
        slot.kv_release.store(HostRelease::Bound, std::memory_order_relaxed);
        slot.mean_release.store(HostRelease::Bound, std::memory_order_relaxed);
        slot.nvme_release.store(resources.nvme_bytes ? HostRelease::Bound : HostRelease::Released,
                                std::memory_order_relaxed);
        slot.pinned_release.store(resources.pinned_bytes ? HostRelease::Bound : HostRelease::Released,
                                  std::memory_order_relaxed);
        slot.resources = std::move(resources);
        const bool split_writer = bool(slot.resources.callback);
        slot.writer_resources.store(split_writer ? kWriterAll : 0, std::memory_order_relaxed);
        slot.release = std::move(release);
        slot.generation.store(handle.generation, std::memory_order_release);
        shared->observe_end(original_begin, valid_tokens);
        slot.control.store(split_writer ? code(Phase::Writing, 0, 1) + kKvOne + kMeanOne : code(Phase::Writing, 1),
                           std::memory_order_release);
        Lease writer(shared, handle, LeaseKind::Writer, split_writer ? HostAccess::None : HostAccess::All, split_writer);
        writer.writer_resource_ = std::move(slot.resources.writer_resource);
        writer.count_operation();
        // A write admitted before close may complete with its writer lease.
        // Returning nullopt here would orphan a Writing slot if a bounded
        // retire CAS happened to return Retry during the close race.
        return writer;
    }
    return std::nullopt;
}

bool HistoricalKvRepository::publish_write(const Lease & writer) {
    if (!writer.shared_ || writer.shared_.get() != shared_.get() || !writer.is_writer()) return false;
    const HistoricalKvHandle handle = writer.handle_;
    if (handle.slot >= shared_->capacity || !shared_->matches(handle.slot, handle)) return false;
    auto & control = shared_->slots[handle.slot].control;
    // The writer reference excludes Releasing/Ready and generation reuse.
    // fetch_or changes only Writing->Live; concurrent retire installs Retired
    // with this bit already set. Reference bits are unaffected. Repeated
    // publication remains false, as does publication after retirement.
    return phase_of(control.fetch_or(kPublishBit, std::memory_order_acq_rel)) == Phase::Writing;
}

bool HistoricalKvRepository::release_writer_resource(const Lease & writer, HostResource resource) noexcept {
    if (!owns_writer(writer) || !writer.split_writer_ ||
        (resource != HostResource::PackedKv && resource != HostResource::MeanK)) return false;
    auto & slot = shared_->slots[writer.handle_.slot];
    const uint8_t bit = resource == HostResource::PackedKv ? kWriterKv : kWriterMean;
    // Close the producer's future access first, then drop the backing resource
    // claim. The surviving identity ref keeps metadata/publication usable. A
    // duplicate completion cannot subtract the same claim twice.
    const auto previous = slot.writer_resources.fetch_and(static_cast<uint8_t>(~bit), std::memory_order_acq_rel);
    if (previous & bit) slot.control.fetch_sub(resource == HostResource::PackedKv ? kKvOne : kMeanOne,
                                               std::memory_order_acq_rel);
    return true;
}

std::optional<HistoricalKvRepository::Lease> HistoricalKvRepository::acquire_new(
    HistoricalKvHandle handle) {
    return acquire_kind(handle, LeaseKind::Execution, phase_bit(Phase::Live));
}

std::optional<HistoricalKvRepository::Lease> HistoricalKvRepository::acquire_gpu_ready_writing(
    HistoricalKvHandle handle) {
    return acquire_kind(handle, LeaseKind::GpuExecution,
        phase_bit(Phase::Writing) | phase_bit(Phase::Live));
}

std::optional<HistoricalKvRepository::Lease> HistoricalKvRepository::acquire_gpu(
    HistoricalKvHandle handle) {
    return acquire_kind(handle, LeaseKind::GpuExecution, phase_bit(Phase::Live));
}

std::optional<HistoricalKvRepository::Lease> HistoricalKvRepository::acquire_read(
    HistoricalKvHandle handle) {
    return acquire_kind(handle, LeaseKind::Read, phase_bit(Phase::Live));
}

std::optional<HistoricalKvRepository::Lease> HistoricalKvRepository::acquire_transfer(
    HistoricalKvHandle handle) {
    return acquire_kind(handle, LeaseKind::Transfer, phase_bit(Phase::Live));
}

std::optional<HistoricalKvRepository::Lease> HistoricalKvRepository::acquire_packed_kv(HistoricalKvHandle handle) {
    return acquire_kind(handle, LeaseKind::PackedKvRead, phase_bit(Phase::Live));
}
std::optional<HistoricalKvRepository::Lease> HistoricalKvRepository::acquire_packed_kv_transfer(HistoricalKvHandle handle) {
    return acquire_kind(handle, LeaseKind::PackedKvTransfer, phase_bit(Phase::Live));
}
std::optional<HistoricalKvRepository::Lease> HistoricalKvRepository::acquire_mean_k(HistoricalKvHandle handle) {
    return acquire_kind(handle, LeaseKind::MeanKRead, phase_bit(Phase::Live));
}
std::optional<HistoricalKvRepository::Lease> HistoricalKvRepository::acquire_background_mean_k(HistoricalKvHandle handle) {
    return acquire_kind(handle, LeaseKind::MeanKRead, phase_bit(Phase::Live), false);
}
std::optional<HistoricalKvRepository::Lease> HistoricalKvRepository::acquire_external_read(HistoricalKvHandle handle) {
    return acquire_kind(handle, LeaseKind::ExternalRead, phase_bit(Phase::Live));
}

std::optional<HistoricalKvRepository::Lease> HistoricalKvRepository::acquire_kind(
    HistoricalKvHandle handle, LeaseKind kind, uint32_t allowed_phases, bool request_operation) {
    const auto shared = shared_;
    if (handle.incarnation != shared->incarnation || handle.slot >= shared->capacity ||
        shared->closing.load(std::memory_order_acquire)) return std::nullopt;
    const HostAccess access = (kind == LeaseKind::GpuExecution || kind == LeaseKind::ExternalRead) ? HostAccess::None :
        kind == LeaseKind::MeanKRead ? HostAccess::MeanK :
        (kind == LeaseKind::PackedKvRead || kind == LeaseKind::PackedKvTransfer) ? HostAccess::PackedKv :
        HostAccess::All;
    if (!shared->try_pin(handle.slot, allowed_phases, access)) return std::nullopt;
    Lease lease(shared, handle, kind, access);
    if (!shared->matches(handle.slot, handle) || shared->closing.load(std::memory_order_acquire))
        return std::nullopt;
    if (request_operation) lease.count_operation();
    return lease;
}

HistoricalKvRepository::RetireResult HistoricalKvRepository::retire(
        HistoricalKvHandle handle, std::optional<uint64_t> expected_content_version,
        RetirePredicate permits, const void * policy) {
    const auto shared = shared_;
    if (handle.incarnation != shared->incarnation || handle.slot >= shared->capacity ||
        handle.generation == 0) return RetireResult::Stale;
    if (!shared->try_pin(handle.slot, kReadable)) {
        const Phase current = phase_of(shared->slots[handle.slot].control.load(std::memory_order_acquire));
        if (current == Phase::Releasing) {
            auto & slot = shared->slots[handle.slot];
            bool available = false;
            if (!slot.release_busy.compare_exchange_strong(available, true,
                    std::memory_order_acq_rel, std::memory_order_acquire)) return RetireResult::Retry;
            // The releasing owner cannot publish Ready while this short guard
            // is held. Do not read its generation without that protection.
            const bool matching = phase_of(slot.control.load(std::memory_order_acquire)) == Phase::Releasing &&
                                  shared->matches(handle.slot, handle);
            slot.release_busy.store(false, std::memory_order_release);
            return matching ? RetireResult::AlreadyRetired : RetireResult::Stale;
        }
        return current == Phase::Ready || current == Phase::Reserved
            ? RetireResult::Stale : RetireResult::Retry;
    }
    Lease guard(shared, handle, LeaseKind::Read);
    if (!shared->matches(handle.slot, handle)) return RetireResult::Stale;
    auto & slot = shared->slots[handle.slot];
    auto & control = slot.control;
    uint64_t observed = control.load(std::memory_order_acquire);
    for (unsigned attempt = 0; attempt < kCasAttempts; ++attempt) {
        if (phase_of(observed) == Phase::Retired) return RetireResult::AlreadyRetired;
        if (phase_of(observed) != Phase::Writing && phase_of(observed) != Phase::Live)
            return RetireResult::Retry;
        // This field is published before Writing and remains immutable for the
        // generation. The guard prevents the slot from being reused here.
        if (expected_content_version && slot.content_version != *expected_content_version)
            return RetireResult::VersionChanged;
        if (permits && !permits(shared->copy_meta(handle.slot, observed), policy))
            return RetireResult::NotPermitted;
        const uint64_t desired = (observed & ~kPhaseMask) | static_cast<uint64_t>(Phase::Retired);
        if (control.compare_exchange_weak(observed, desired, std::memory_order_acq_rel,
                                          std::memory_order_acquire)) return RetireResult::Committed;
    }
    return RetireResult::Retry;
}

HistoricalKvRepository::ReclaimResult HistoricalKvRepository::try_reclaim(
    HistoricalKvHandle handle) {
    const auto shared = shared_;
    if (handle.incarnation != shared->incarnation || handle.slot >= shared->capacity ||
        handle.generation == 0) return ReclaimResult::Stale;
    auto & slot = shared->slots[handle.slot];
    bool expected = false;
    if (!slot.release_busy.compare_exchange_strong(expected, true,
                                                   std::memory_order_acq_rel,
                                                   std::memory_order_acquire)) return ReclaimResult::Retry;
    struct ClearBusy {
        Shared::Slot & slot;
        ~ClearBusy() { slot.release_busy.store(false, std::memory_order_release); }
    } clear{slot};

    const Phase phase = phase_of(slot.control.load(std::memory_order_acquire));
    Lease guard;
    bool full_release = phase == Phase::Releasing;
    if (phase == Phase::Retired) {
        if (!shared->try_pin(handle.slot, phase_bit(Phase::Retired))) return ReclaimResult::Deferred;
        guard = Lease(shared, handle, LeaseKind::Read);
        if (!shared->matches(handle.slot, handle)) return ReclaimResult::Stale;
        if (!slot.resources.callback && host_refs_of(slot.control.load(std::memory_order_acquire)))
            return ReclaimResult::Deferred;
        // New resource access closed at retire_commit; every admitted future
        // dependency was registered in the same control word. Keep identity
        // pinned during partial release, or consume our sole guard to enter
        // full Releasing when all resource/GPU/metadata references are gone.
        uint64_t only_guard = code(Phase::Retired, 0, 1);
        full_release = slot.control.compare_exchange_strong(only_guard, code(Phase::Releasing),
            std::memory_order_acq_rel, std::memory_order_acquire);
        if (full_release) guard.disarm();
    } else if (phase == Phase::Releasing) {
        if (!shared->matches(handle.slot, handle)) return ReclaimResult::Stale;
    } else if (phase == Phase::Writing || phase == Phase::Live) {
        if (!shared->try_pin(handle.slot, phase_bit(phase))) return ReclaimResult::Deferred;
        guard = Lease(shared, handle, LeaseKind::Read);
        return shared->matches(handle.slot, handle) ? ReclaimResult::Deferred : ReclaimResult::Stale;
    } else {
        return phase == Phase::Reserving ? ReclaimResult::Deferred : ReclaimResult::Stale;
    }

    bool external_progress = false, external_retry = false;
    if (slot.resources.pinned_bytes &&
        slot.pinned_release.load(std::memory_order_acquire) != HostRelease::Released) {
        slot.pinned_release.store(HostRelease::Releasing, std::memory_order_release);
        bool complete = false;
        try { complete = slot.resources.callback(handle, HostResource::Pinned); }
        catch (...) { complete = false; }
        if (complete) {
            slot.pinned_release.store(HostRelease::Released, std::memory_order_release);
            external_progress = true;
        } else external_retry = true;
    }
    if (slot.resources.nvme_bytes &&
        slot.nvme_release.load(std::memory_order_acquire) != HostRelease::Released) {
        // The allocation closes admission and counts its own actual IO. Its
        // callback must tolerate an admitted old producer and return false
        // while that producer or any admitted disk operation still owns it.
        slot.nvme_release.store(HostRelease::Releasing, std::memory_order_release);
        bool complete = false;
        try { complete = slot.resources.callback(handle, HostResource::Nvme); }
        catch (...) { complete = false; }
        if (complete) {
            slot.nvme_release.store(HostRelease::Released, std::memory_order_release);
            external_progress = true;
        } else external_retry = true;
    }
    const bool newly_released = slot.host_release.load(std::memory_order_acquire) != HostRelease::Released;
    if (slot.resources.callback && newly_released) {
        bool progress = external_progress, retry = external_retry;
        for (auto resource : {HostResource::PackedKv, HostResource::MeanK}) {
            auto & state = resource == HostResource::PackedKv ? slot.kv_release : slot.mean_release;
            if (state.load(std::memory_order_acquire) == HostRelease::Released ||
                resource_refs_of(slot.control.load(std::memory_order_acquire), resource)) continue;
            // Retired closes admission irreversibly; these resource counts can
            // only decrease. The guard retains immutable identity, not payload.
            state.store(HostRelease::Releasing, std::memory_order_release);
            bool complete = false;
            try { complete = slot.resources.callback(handle, resource); }
            catch (...) { complete = false; }
            if (!complete) { retry = true; continue; }
            state.store(HostRelease::Released, std::memory_order_release);
            progress = true;
        }
        if (slot.kv_release.load(std::memory_order_acquire) != HostRelease::Released ||
            slot.mean_release.load(std::memory_order_acquire) != HostRelease::Released)
            return progress ? ReclaimResult::ResourceReleased : retry ? ReclaimResult::Retry : ReclaimResult::Deferred;
        slot.host_release.store(HostRelease::Released, std::memory_order_release);
    } else if (newly_released) {
        slot.host_release.store(HostRelease::Releasing, std::memory_order_release);
        bool complete = false;
        try { complete = slot.release && slot.release(handle); }
        catch (...) { complete = false; }
        if (!complete) return ReclaimResult::Retry;
        slot.release = {};
        slot.kv_release.store(HostRelease::Released, std::memory_order_release);
        slot.mean_release.store(HostRelease::Released, std::memory_order_release);
        slot.host_release.store(HostRelease::Released, std::memory_order_release);
    }
    if (slot.nvme_release.load(std::memory_order_acquire) != HostRelease::Released ||
        slot.pinned_release.load(std::memory_order_acquire) != HostRelease::Released)
        return newly_released ? ReclaimResult::HostReleased : ReclaimResult::Retry;
    if (!full_release) {
        uint64_t only_guard = code(Phase::Retired, 0, 1);
        full_release = slot.control.compare_exchange_strong(only_guard, code(Phase::Releasing),
            std::memory_order_acq_rel, std::memory_order_acquire);
        if (!full_release) return newly_released ? ReclaimResult::HostReleased :
            external_progress ? ReclaimResult::ResourceReleased : ReclaimResult::Deferred;
        guard.disarm();
    }
    if (slot.resources.callback) {
        bool complete = false;
        try { complete = slot.resources.callback(handle, HostResource::Metadata); }
        catch (...) { complete = false; }
        if (!complete) return ReclaimResult::Retry;
        slot.resources.callback = {};
    }
    // No live metadata/execution owner remains and the Host callback completed.
    // A partial Host release never grants a physical slot to request admission.
    slot.control.store(code(Phase::Ready), std::memory_order_release);
    return ReclaimResult::Complete;
}

bool HistoricalKvRepository::mark_executed(const Lease & admitted, uint64_t sequence) {
    const auto shared = shared_;
    if (!admitted.shared_ || admitted.shared_.get() != shared.get() ||
        (admitted.kind_ != LeaseKind::Execution && admitted.kind_ != LeaseKind::GpuExecution)) return false;
    const HistoricalKvHandle handle = admitted.handle_;
    if (handle.slot >= shared->capacity || !shared->matches(handle.slot, handle)) return false;
    const Phase phase = phase_of(shared->slots[handle.slot].control.load(std::memory_order_acquire));
    if (phase != Phase::Writing && phase != Phase::Live && phase != Phase::Retired) return false;
    auto & use = shared->slots[handle.slot].last_use_seq;
    uint64_t previous = use.load(std::memory_order_relaxed);
    for (unsigned attempt = 0; attempt < kCasAttempts; ++attempt) {
        if (previous >= sequence) return true;
        if (use.compare_exchange_weak(previous, sequence, std::memory_order_release,
                                      std::memory_order_relaxed)) return true;
    }
    return false;
}

bool HistoricalKvRepository::owns_writer(const Lease & writer) const noexcept {
    return writer.shared_.get() == shared_.get() && writer.kind_ == LeaseKind::Writer;
}

uint64_t HistoricalKvRepository::repository_id() const noexcept {
    return shared_->repository_id;
}

bool HistoricalKvRepository::is_releasing(HistoricalKvHandle handle) const noexcept {
    const auto shared = shared_;
    if (handle.incarnation != shared->incarnation ||
        handle.slot >= shared->capacity || handle.generation == 0) return false;
    const auto & slot = shared->slots[handle.slot];
    // The release owner enters this phase only after every old lease has
    // drained. Once it publishes Ready, the bound payload is already gone.
    return phase_of(slot.control.load(std::memory_order_acquire)) == Phase::Releasing &&
           slot.generation.load(std::memory_order_acquire) == handle.generation;
}

bool HistoricalKvRepository::is_host_releasing(HistoricalKvHandle handle) const noexcept {
    const auto shared = shared_;
    if (handle.incarnation != shared->incarnation || handle.slot >= shared->capacity || !handle.generation)
        return false;
    const auto & slot = shared->slots[handle.slot];
    const auto control = slot.control.load(std::memory_order_acquire);
    return (phase_of(control) == Phase::Retired || phase_of(control) == Phase::Releasing) &&
        !host_refs_of(control) && slot.host_release.load(std::memory_order_acquire) == HostRelease::Releasing &&
        slot.generation.load(std::memory_order_acquire) == handle.generation;
}

bool HistoricalKvRepository::is_resource_releasing(HistoricalKvHandle handle, HostResource resource) const noexcept {
    const auto shared = shared_;
    if (handle.incarnation != shared->incarnation || handle.slot >= shared->capacity || !handle.generation)
        return false;
    const auto & slot = shared->slots[handle.slot];
    const auto control = slot.control.load(std::memory_order_acquire);
    if (slot.generation.load(std::memory_order_acquire) != handle.generation) return false;
    if (resource == HostResource::Metadata)
        return phase_of(control) == Phase::Releasing &&
            slot.kv_release.load(std::memory_order_acquire) == HostRelease::Released &&
            slot.mean_release.load(std::memory_order_acquire) == HostRelease::Released &&
            slot.nvme_release.load(std::memory_order_acquire) == HostRelease::Released &&
            slot.pinned_release.load(std::memory_order_acquire) == HostRelease::Released;
    if (resource == HostResource::Nvme)
        return (phase_of(control) == Phase::Retired || phase_of(control) == Phase::Releasing) &&
            slot.nvme_release.load(std::memory_order_acquire) == HostRelease::Releasing;
    if (resource == HostResource::Pinned)
        return (phase_of(control) == Phase::Retired || phase_of(control) == Phase::Releasing) &&
            slot.pinned_release.load(std::memory_order_acquire) == HostRelease::Releasing;
    const auto & state = resource == HostResource::PackedKv ? slot.kv_release : slot.mean_release;
    return (phase_of(control) == Phase::Retired || phase_of(control) == Phase::Releasing) &&
        !resource_refs_of(control, resource) && state.load(std::memory_order_acquire) == HostRelease::Releasing;
}

bool HistoricalKvRepository::reclaim_completed(
        HistoricalKvHandle handle, uint64_t expected_repository_id) const noexcept {
    const auto shared = shared_;
    if (expected_repository_id != shared->repository_id ||
        handle.incarnation != shared->incarnation ||
        handle.slot >= shared->capacity || handle.generation == 0) return false;
    const auto & slot = shared->slots[handle.slot];
    // A newer generation can only have been reserved after the old callback
    // returned true and Ready was published. Generation never wraps.
    const auto generation = slot.generation.load(std::memory_order_acquire);
    if (generation > handle.generation) return true;
    if (generation != handle.generation) return false;
    // Ready is published with release ordering after the callback completes.
    // Reserving is reachable only by claiming that Ready state. Reading the
    // phase alone, rather than inspect()==null, avoids mistaking Releasing or
    // a bounded pin failure for a completed physical release.
    const Phase phase = phase_of(slot.control.load(std::memory_order_acquire));
    return phase == Phase::Ready || phase == Phase::Reserving;
}

std::optional<HistoricalKvBlockMeta> HistoricalKvRepository::inspect(
    HistoricalKvHandle handle) const {
    const auto shared = shared_;
    if (handle.incarnation != shared->incarnation || handle.slot >= shared->capacity ||
        !shared->try_pin(handle.slot, kReadable)) return std::nullopt;
    Lease guard(shared, handle, LeaseKind::Read);
    if (!shared->matches(handle.slot, handle)) return std::nullopt;
    return shared->copy_meta(handle.slot,
        shared->slots[handle.slot].control.load(std::memory_order_acquire));
}

uint64_t HistoricalKvRepository::outstanding_operations() const noexcept {
    return shared_->operations.load(std::memory_order_acquire);
}

bool HistoricalKvRepository::cleanup_ready() const noexcept {
    return shared_->closing.load(std::memory_order_acquire) &&
        !shared_->operations.load(std::memory_order_acquire) && shared_.use_count() == 1;
}

HistoricalKvRepository::CompletionState HistoricalKvRepository::completion_state(
        HistoricalKvHandle handle) const noexcept {
    const auto & shared = shared_;
    if (handle.incarnation != shared->incarnation || handle.slot >= shared->capacity ||
            !handle.generation) return CompletionState::Unfinished;
    const auto & slot = shared->slots[handle.slot];
    const auto before = slot.generation.load(std::memory_order_acquire);
    const auto phase = phase_of(slot.control.load(std::memory_order_acquire));
    const auto after = slot.generation.load(std::memory_order_acquire);
    // A newer generation can exist only after this one's release completed.
    // Double sampling avoids mistaking the next writer for this request's
    // unfinished writer when Ready is claimed between these atomic reads.
    if (before > handle.generation || after > handle.generation) return CompletionState::Retired;
    if (before != handle.generation || after != handle.generation) return CompletionState::Unfinished;
    switch (phase) {
    case Phase::Live: return CompletionState::Live;
    case Phase::Retired:
    case Phase::Releasing:
    case Phase::Ready:
    case Phase::Reserving: return CompletionState::Retired;
    case Phase::Writing:
    case Phase::Reserved: return CompletionState::Unfinished;
    }
    return CompletionState::Unfinished;
}

std::vector<HistoricalKvBlockMeta> HistoricalKvRepository::snapshot() const {
    return snapshot_range(0, shared_->capacity);
}

HistoricalKvRepository::RangeObservation HistoricalKvRepository::observe_range(
    uint32_t first, uint32_t count) const {
    RangeObservation out;
    if (first < shared_->capacity && count)
        out.blocks.reserve(std::min(count, shared_->capacity - first));
    observe_range_into(first, count, out);
    return out;
}

void HistoricalKvRepository::observe_range_into(
    uint32_t first, uint32_t count, RangeObservation & out) const {
    const auto shared = shared_;
    const uint32_t scan = first < shared->capacity
        ? std::min(count, shared->capacity - first) : 0;
    if (out.blocks.capacity() < scan)
        throw std::length_error("historical KV observation scratch capacity exceeded");
    out.blocks.clear();
    out.ready = out.reserved = out.pending = 0;
    const uint32_t end = first + scan;
    for (uint32_t index = first; index < end; ++index) {
        if (!shared->try_pin(index, kReadable)) {
            // Ready needs no payload metadata; if it races a reservation,
            // this remains a best-effort pressure observation. The nonblocking
            // claim keeps a Releasing generation stable while observing it.
            auto & slot = shared->slots[index];
            const Phase observed = phase_of(slot.control.load(std::memory_order_acquire));
            if (observed == Phase::Reserved) {
                ++out.reserved;
                continue;
            }
            if (observed == Phase::Ready) {
                const auto generation = slot.generation.load(std::memory_order_acquire);
                if (generation != std::numeric_limits<uint64_t>::max() &&
                    phase_of(slot.control.load(std::memory_order_acquire)) == Phase::Ready)
                    ++out.ready;
                continue;
            }
            if (observed != Phase::Releasing && observed != Phase::Retired) continue;
            bool available = false;
            if (!slot.release_busy.compare_exchange_strong(available, true,
                    std::memory_order_acq_rel, std::memory_order_acquire)) continue;
            struct ClearBusy {
                Shared::Slot & slot;
                ~ClearBusy() { slot.release_busy.store(false, std::memory_order_release); }
            } clear{slot};
            const Phase phase = phase_of(slot.control.load(std::memory_order_acquire));
            if (phase == Phase::Releasing || phase == Phase::Retired) {
                HistoricalKvBlockMeta meta;
                meta.handle = {shared->incarnation, index,
                               slot.generation.load(std::memory_order_acquire)};
                meta.life = phase == Phase::Releasing ? HistoricalKvLife::Releasing :
                            HistoricalKvLife::Retired;
                out.blocks.push_back(meta);
                ++out.pending;
            } else if (phase == Phase::Ready &&
                       slot.generation.load(std::memory_order_acquire) !=
                           std::numeric_limits<uint64_t>::max() &&
                       phase_of(slot.control.load(std::memory_order_acquire)) == Phase::Ready) {
                ++out.ready;
            }
            continue;
        }
        const HistoricalKvHandle handle{shared->incarnation, index,
            shared->slots[index].generation.load(std::memory_order_acquire)};
        Lease guard(shared, handle, LeaseKind::Read);
        if (!shared->matches(index, handle)) continue;
        auto meta = shared->copy_meta(index,
            shared->slots[index].control.load(std::memory_order_acquire));
        if (meta.life == HistoricalKvLife::Retired) ++out.pending;
        out.blocks.push_back(meta);
    }
}

std::vector<HistoricalKvBlockMeta> HistoricalKvRepository::snapshot_range(
    uint32_t first, uint32_t count) const {
    return observe_range(first, count).blocks;
}

uint32_t HistoricalKvRepository::ready_count() const {
    return ready_count_range(0, shared_->capacity);
}

HistoricalKvRepository::CapacityObservation HistoricalKvRepository::observe_capacity() const noexcept {
    CapacityObservation out;
    for (uint32_t index = 0; index < shared_->capacity; ++index) {
        const auto & slot = shared_->slots[index];
        const auto phase = phase_of(slot.control.load(std::memory_order_acquire));
        if (phase == Phase::Ready && slot.generation.load(std::memory_order_acquire) != UINT64_MAX)
            ++out.ready;
        else if (phase == Phase::Retired || phase == Phase::Releasing) ++out.pending;
    }
    return out;
}

uint32_t HistoricalKvRepository::ready_count_range(uint32_t first, uint32_t count) const {
    if (first >= shared_->capacity || count == 0) return 0;
    const uint32_t end = first + std::min(count, shared_->capacity - first);
    uint32_t ready = 0;
    for (uint32_t index = first; index < end; ++index) {
        const auto & slot = shared_->slots[index];
        ready += phase_of(slot.control.load(std::memory_order_acquire)) == Phase::Ready &&
                 slot.generation.load(std::memory_order_acquire) !=
                     std::numeric_limits<uint64_t>::max();
    }
    return ready;
}

uint32_t HistoricalKvRepository::capacity() const { return shared_->capacity; }
uint64_t HistoricalKvRepository::incarnation() const { return shared_->incarnation; }
bool HistoricalKvRepository::claim_gc_owner() noexcept {
    bool expected = false;
    return shared_->gc_owner_claimed.compare_exchange_strong(expected, true,
        std::memory_order_acq_rel, std::memory_order_acquire);
}
void HistoricalKvRepository::release_gc_owner() noexcept {
    shared_->gc_owner_claimed.store(false, std::memory_order_release);
}
void HistoricalKvRepository::close() noexcept { shared_->closing.store(true, std::memory_order_release); }
bool HistoricalKvRepository::closed() const { return shared_->closing.load(std::memory_order_acquire); }

} // namespace kvmem
