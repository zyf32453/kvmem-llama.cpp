#include "kvmem/historical_kv_repository.hpp"

#include <algorithm>
#include <atomic>
#include <limits>
#include <stdexcept>
#include <utility>

namespace kvmem {
namespace {

// The low three bits are the phase. The remaining bits count references.
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
constexpr uint64_t kRefOne = 8;
constexpr uint64_t kMaxRefs = std::numeric_limits<uint64_t>::max() >> 3;
constexpr unsigned kCasAttempts = 16;

Phase phase_of(uint64_t control) { return static_cast<Phase>(control & kPhaseMask); }
uint64_t refs_of(uint64_t control) { return control >> 3; }
uint64_t code(Phase phase, uint64_t refs = 0) {
    return (refs << 3) | static_cast<uint64_t>(phase);
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
        uint64_t logical_id = 0;
        uint64_t original_begin = 0;
        uint64_t birth_seq = 0;
        uint64_t content_version = 1;
        uint64_t payload_bytes = 0;
        uint32_t valid_tokens = 0;
        ReleaseCallback release;
    };

    Shared(uint32_t n, uint64_t session)
        : capacity(n), incarnation(session), repository_id(allocate_repository_id()),
          slots(new Slot[n]) {}

    bool try_pin(uint32_t index, uint32_t phases) const {
        auto & control = slots[index].control;
        uint64_t observed = control.load(std::memory_order_acquire);
        for (unsigned attempt = 0; attempt < kCasAttempts; ++attempt) {
            if (!(phases & phase_bit(phase_of(observed))) || refs_of(observed) == kMaxRefs) return false;
            if (control.compare_exchange_weak(observed, observed + kRefOne,
                                              std::memory_order_acq_rel,
                                              std::memory_order_acquire)) return true;
        }
        return false;
    }

    void unpin(uint32_t index) const noexcept {
        slots[index].control.fetch_sub(kRefOne, std::memory_order_acq_rel);
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
        const uint64_t refs = refs_of(control);
        out.references = static_cast<uint32_t>(refs > 0 ?
            std::min<uint64_t>(refs - 1, std::numeric_limits<uint32_t>::max()) : 0);
        out.active = slot.exec_refs.load(std::memory_order_acquire) != 0;
        out.sealed = out.life == HistoricalKvLife::Live;
        return out;
    }

    const uint32_t capacity;
    const uint64_t incarnation;
    const uint64_t repository_id;
    std::unique_ptr<Slot[]> slots;
    std::atomic<uint32_t> cursor{0};
    std::atomic<bool> closing{false};
    std::atomic<bool> gc_owner_claimed{false};
    std::atomic<uint64_t> operations{0};
};

HistoricalKvRepository::Lease::Lease(std::shared_ptr<Shared> shared,
                                     HistoricalKvHandle handle, LeaseKind kind)
    : shared_(std::move(shared)), handle_(handle), kind_(kind) {}

HistoricalKvRepository::Lease::~Lease() { reset(); }

HistoricalKvRepository::Lease::Lease(Lease && other) noexcept
    : shared_(std::move(other.shared_)), handle_(other.handle_), kind_(other.kind_),
      counted_exec_(other.counted_exec_), counted_operation_(other.counted_operation_) {
    other.kind_ = LeaseKind::Read;
    other.counted_exec_ = false;
    other.counted_operation_ = false;
}

HistoricalKvRepository::Lease & HistoricalKvRepository::Lease::operator=(Lease && other) noexcept {
    if (this != &other) {
        reset();
        shared_ = std::move(other.shared_);
        handle_ = other.handle_;
        kind_ = other.kind_;
        counted_exec_ = other.counted_exec_;
        counted_operation_ = other.counted_operation_;
        other.kind_ = LeaseKind::Read;
        other.counted_exec_ = false;
        other.counted_operation_ = false;
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

void HistoricalKvRepository::Lease::reset() noexcept {
    if (shared_) {
        if (counted_exec_)
            shared_->slots[handle_.slot].exec_refs.fetch_sub(1, std::memory_order_acq_rel);
        shared_->unpin(handle_.slot);
        if (counted_operation_) shared_->operations.fetch_sub(1, std::memory_order_acq_rel);
        shared_.reset();
    }
    kind_ = LeaseKind::Read;
    counted_exec_ = false;
    counted_operation_ = false;
}

void HistoricalKvRepository::Lease::count_operation() noexcept {
    shared_->operations.fetch_add(1, std::memory_order_acq_rel);
    counted_operation_ = true;
    if (kind_ == LeaseKind::Execution) {
        shared_->slots[handle_.slot].exec_refs.fetch_add(1, std::memory_order_acq_rel);
        counted_exec_ = true;
    }
}

void HistoricalKvRepository::Lease::disarm() noexcept {
    shared_.reset();
    kind_ = LeaseKind::Read;
    counted_exec_ = false;
    counted_operation_ = false;
}

HistoricalKvRepository::HistoricalKvRepository(uint32_t capacity, uint64_t incarnation)
    : shared_(std::make_shared<Shared>(capacity,
          incarnation ? incarnation : allocate_incarnation())) {
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
    if (!release) throw std::invalid_argument("historical KV release callback is required");
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
    slot.release = std::move(release);
    ++reservation.consumed_;
    slot.control.store(code(Phase::Writing, 1), std::memory_order_release);
    Lease writer(shared, handle, LeaseKind::Writer);
    writer.count_operation();
    return writer;
}

std::optional<HistoricalKvRepository::Lease> HistoricalKvRepository::begin_write(
    uint64_t logical_id, uint64_t original_begin, uint32_t valid_tokens,
    uint64_t birth_seq, uint64_t payload_bytes, ReleaseCallback release) {
    if (!release) throw std::invalid_argument("historical KV release callback is required");
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
        slot.release = std::move(release);
        slot.generation.store(handle.generation, std::memory_order_release);
        slot.control.store(code(Phase::Writing, 1), std::memory_order_release);
        Lease writer(shared, handle, LeaseKind::Writer);
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

std::optional<HistoricalKvRepository::Lease> HistoricalKvRepository::acquire_new(
    HistoricalKvHandle handle) {
    return acquire_kind(handle, LeaseKind::Execution, phase_bit(Phase::Live));
}

std::optional<HistoricalKvRepository::Lease> HistoricalKvRepository::acquire_gpu_ready_writing(
    HistoricalKvHandle handle) {
    return acquire_kind(handle, LeaseKind::Execution,
        phase_bit(Phase::Writing) | phase_bit(Phase::Live));
}

std::optional<HistoricalKvRepository::Lease> HistoricalKvRepository::acquire_read(
    HistoricalKvHandle handle) {
    return acquire_kind(handle, LeaseKind::Read, phase_bit(Phase::Live));
}

std::optional<HistoricalKvRepository::Lease> HistoricalKvRepository::acquire_transfer(
    HistoricalKvHandle handle) {
    return acquire_kind(handle, LeaseKind::Transfer, phase_bit(Phase::Live));
}

std::optional<HistoricalKvRepository::Lease> HistoricalKvRepository::acquire_kind(
    HistoricalKvHandle handle, LeaseKind kind, uint32_t allowed_phases) {
    const auto shared = shared_;
    if (handle.incarnation != shared->incarnation || handle.slot >= shared->capacity ||
        shared->closing.load(std::memory_order_acquire)) return std::nullopt;
    if (!shared->try_pin(handle.slot, allowed_phases)) return std::nullopt;
    Lease lease(shared, handle, kind);
    if (!shared->matches(handle.slot, handle) || shared->closing.load(std::memory_order_acquire))
        return std::nullopt;
    lease.count_operation();
    return lease;
}

HistoricalKvRepository::RetireResult HistoricalKvRepository::retire(HistoricalKvHandle handle) {
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
    auto & control = shared->slots[handle.slot].control;
    uint64_t observed = control.load(std::memory_order_acquire);
    for (unsigned attempt = 0; attempt < kCasAttempts; ++attempt) {
        if (phase_of(observed) == Phase::Retired) return RetireResult::AlreadyRetired;
        if (phase_of(observed) != Phase::Writing && phase_of(observed) != Phase::Live)
            return RetireResult::Retry;
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
    if (phase == Phase::Retired) {
        if (!shared->try_pin(handle.slot, phase_bit(Phase::Retired))) return ReclaimResult::Deferred;
        Lease guard(shared, handle, LeaseKind::Read);
        if (!shared->matches(handle.slot, handle)) return ReclaimResult::Stale;
        uint64_t only_guard = code(Phase::Retired, 1);
        if (!slot.control.compare_exchange_strong(only_guard, code(Phase::Releasing),
                                                  std::memory_order_acq_rel,
                                                  std::memory_order_acquire)) return ReclaimResult::Deferred;
        guard.disarm(); // The CAS consumed the sole reference and closed entry.
    } else if (phase == Phase::Releasing) {
        // Only a release owner can publish Ready, and release_busy excludes
        // another owner. The generation is stable for this entire branch.
        if (!shared->matches(handle.slot, handle)) return ReclaimResult::Stale;
    } else if (phase == Phase::Writing || phase == Phase::Live) {
        if (!shared->try_pin(handle.slot, phase_bit(phase))) return ReclaimResult::Deferred;
        Lease guard(shared, handle, LeaseKind::Read);
        return shared->matches(handle.slot, handle) ? ReclaimResult::Deferred : ReclaimResult::Stale;
    } else {
        return phase == Phase::Reserving ? ReclaimResult::Deferred : ReclaimResult::Stale;
    }

    bool complete = false;
    try { complete = slot.release && slot.release(handle); }
    catch (...) { complete = false; }
    if (!complete) return ReclaimResult::Retry;
    slot.release = {};
    slot.control.store(code(Phase::Ready), std::memory_order_release);
    return ReclaimResult::Complete;
}

bool HistoricalKvRepository::mark_executed(const Lease & admitted, uint64_t sequence) {
    const auto shared = shared_;
    if (!admitted.shared_ || admitted.shared_.get() != shared.get() ||
        admitted.kind_ != LeaseKind::Execution) return false;
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
    const auto shared = shared_;
    RangeObservation out;
    if (first >= shared->capacity || count == 0) return out;
    const uint32_t end = first + std::min(count, shared->capacity - first);
    out.blocks.reserve(end - first);
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
    return out;
}

std::vector<HistoricalKvBlockMeta> HistoricalKvRepository::snapshot_range(
    uint32_t first, uint32_t count) const {
    return observe_range(first, count).blocks;
}

uint32_t HistoricalKvRepository::ready_count() const {
    return ready_count_range(0, shared_->capacity);
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
