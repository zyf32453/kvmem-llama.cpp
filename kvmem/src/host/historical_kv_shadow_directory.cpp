#include "kvmem/historical_kv_shadow_directory.hpp"

#include "kvmem/kvmem_store.hpp"
#include "kvmem/historical_kv_raw_store.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace kvmem {
namespace {

bool same_handle(HistoricalKvHandle a, HistoricalKvHandle b) {
    return a.incarnation == b.incarnation && a.slot == b.slot &&
           a.generation == b.generation;
}

bool same_span(const HistoricalKvShadowDirectory::Binding & entry,
               const KvMemBlock & block) {
    return entry.block_id == block.block_id &&
           entry.original_begin == block.orig_pos_start &&
           entry.valid_tokens == block.n_tokens;
}

} // namespace

struct HistoricalKvShadowDirectory::AppendReservation::Impl {
    HistoricalKvShadowDirectory * owner;
    KvMemStore * store;
    std::shared_ptr<HistoricalKvRepository> repository;
    std::shared_ptr<HistoricalKvRawStore> storage;
    uint32_t limit = 0, used = 0, block_tokens = 0;
    bool installed = false;
    std::map<uint32_t, Entry> nodes;
    std::vector<Producer> contexts;
    std::vector<KvMemBlock> table, pending;
};

HistoricalKvShadowDirectory::AppendReservation::AppendReservation(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
HistoricalKvShadowDirectory::AppendReservation::~AppendReservation() = default;
uint32_t HistoricalKvShadowDirectory::AppendReservation::capacity() const noexcept { return impl_->limit; }
uint32_t HistoricalKvShadowDirectory::AppendReservation::remaining() const noexcept { return impl_->limit - impl_->used; }

std::unique_ptr<HistoricalKvShadowDirectory::AppendReservation>
HistoricalKvShadowDirectory::prepare_append_reservation(
        KvMemStore & store, std::shared_ptr<HistoricalKvRepository> repository,
        std::shared_ptr<HistoricalKvRawStore> storage, uint32_t capacity) {
    if (capacity > INT32_MAX || !repository || repository->closed() ||
        !storage || capacity > repository->capacity() || capacity > storage->capacity() ||
        store.block_count() || store.total_tokens() ||
        store.gc_ || store.historical_shadow_directory_ != this || repository_id_ ||
        reserved_append_mode_ || !entries_.empty() ||
        storage->config().max_tokens_per_slot != store.config().block_tokens) return {};
    auto r = std::make_unique<AppendReservation::Impl>();
    r->owner = this; r->store = &store;
    r->repository = repository; r->storage = storage;
    r->limit = capacity; r->block_tokens = store.config().block_tokens;
    r->table.reserve(capacity);
    r->pending.reserve(capacity);
    r->contexts.reserve(capacity);
    for (uint32_t id = 0; id < capacity; ++id) {
        auto context = std::shared_ptr<WriterContext>(new WriterContext(repository, storage));
        Entry entry;
        entry.repository = repository;
        entry.repository_id = repository->repository_id();
        entry.storage = storage;
        entry.producer = context;
        r->nodes.emplace(id, std::move(entry));
        r->contexts.push_back(std::move(context));
    }
    return std::unique_ptr<AppendReservation>(new AppendReservation(std::move(r)));
}

bool HistoricalKvShadowDirectory::install_append_reservation(
        KvMemStore & store, AppendReservation & reservation) noexcept {
    auto & r = *reservation.impl_;
    if (r.installed || r.owner != this || r.store != &store || r.repository->closed() ||
        store.block_count() || store.total_tokens() || store.gc_ ||
        store.historical_shadow_directory_ != this || repository_id_ || !entries_.empty() ||
        reserved_append_mode_ || store.config().block_tokens != r.block_tokens) return false;
    store.blocks_.swap(r.table);
    repository_id_ = r.repository->repository_id();
    reserved_append_capacity_ = r.limit;
    reserved_append_mode_ = true;
    r.installed = true;
    return true;
}

bool HistoricalKvShadowDirectory::append_reserved_blocks(
        KvMemStore & store, AppendReservation & reservation,
        std::shared_ptr<HistoricalKvRepository> repository,
        std::shared_ptr<HistoricalKvRawStore> storage,
        std::vector<HistoricalKvRepository::Lease> & writers, std::vector<Producer> & producers) {
    static_assert(std::is_trivially_copyable<KvMemBlock>::value, "nonthrowing metadata commit");
    auto & r = *reservation.impl_;
    if (!r.installed || r.owner != this || r.store != &store || r.repository != repository ||
        r.storage != storage || !repository || repository->closed() || !producers.empty() ||
        repository_id_ != repository->repository_id() || !reserved_append_mode_ || reserved_append_capacity_ != r.limit ||
        store.block_count() != r.used || store.gc_ || store.historical_shadow_directory_ != this ||
        store.config().block_tokens != r.block_tokens || !tail_span_matches(store) ||
        store.append_tail_room() || writers.size() > r.limit - r.used ||
        writers.size() > producers.capacity() || store.blocks_.capacity() < r.limit) return false;
    r.pending.clear();
    uint64_t position = store.total_tokens();
    for (size_t i = 0; i < writers.size(); ++i) {
        const uint32_t id = r.used + uint32_t(i);
        const auto & writer = writers[i];
        const auto meta = writer.metadata();
        const auto node = r.nodes.find(id);
        const auto & context = r.contexts[id];
        if (node == r.nodes.end() || entries_.count(id) || !context || context->writer_ ||
            !repository->owns_writer(writer) || !meta ||
            (meta->life != HistoricalKvLife::Writing && meta->life != HistoricalKvLife::Retired) ||
            meta->logical_id != id || meta->original_begin != position || !meta->valid_tokens ||
            meta->valid_tokens > r.block_tokens || position + meta->valid_tokens > UINT32_MAX) return false;
        const auto payload = storage->inspect(*repository, writer.handle());
        if (!payload || payload->original_begin != position || payload->valid_tokens != meta->valid_tokens)
            return false;
        node->second.binding = {id, writer.handle(), position, meta->valid_tokens, {}};
        KvMemBlock block;
        block.block_id = id; block.orig_pos_start = uint32_t(position);
        block.n_tokens = meta->valid_tokens; block.baked_pos = int64_t(position);
        r.pending.push_back(block);
        position += meta->valid_tokens;
    }
    // Validation above changes only private records. Below, integer-key map
    // node transfers use the same allocator; all vector capacities are held.
    for (size_t i = 0; i < writers.size(); ++i) {
        const uint32_t id = r.used + uint32_t(i);
        auto & context = r.contexts[id];
        context->writer_ = std::move(writers[i]);
        const auto inserted = entries_.insert(r.nodes.extract(id));
        if (!inserted.inserted) std::terminate(); // impossible after owner-serialized validation
        producers.push_back(std::move(context));
    }
    store.blocks_.insert(store.blocks_.end(), r.pending.begin(), r.pending.end());
    store.total_tokens_ = uint32_t(position);
    r.used += uint32_t(writers.size());
    return true;
}

bool HistoricalKvShadowDirectory::bind_full_block(
        const KvMemStore & store, uint32_t block_id,
        std::shared_ptr<HistoricalKvRepository> repository,
        HistoricalKvHandle handle,
        HistoricalKvRepository::Lease & writer,
        Producer & producer,
        std::shared_ptr<HistoricalKvRawStore> storage) {
    if (block_id >= store.block_count() ||
        store.blocks()[block_id].n_tokens != store.config().block_tokens) return false;
    return bind_block(store, block_id, std::move(repository), handle, writer, producer, std::move(storage));
}

bool HistoricalKvShadowDirectory::bind_block(
        const KvMemStore & store, uint32_t block_id,
        std::shared_ptr<HistoricalKvRepository> repository,
        HistoricalKvHandle handle,
        HistoricalKvRepository::Lease & writer,
        Producer & producer,
        std::shared_ptr<HistoricalKvRawStore> storage) {
    if (reserved_append_mode_ || producer || !repository || repository->closed() || !writer ||
        !repository->owns_writer(writer) ||
        !same_handle(handle, writer.handle()) ||
        writer.repository_id() != repository->repository_id() ||
        (repository_id_ != 0 && repository_id_ != repository->repository_id()) ||
        block_id >= store.block_count() || entries_.count(block_id) != 0) return false;

    const auto & block = store.blocks()[block_id];
    const auto bt = store.config().block_tokens;
    if (!bt || !block.n_tokens || block.n_tokens > bt) return false;
    if (!storage && (block.n_tokens != bt ||
                    block.orig_pos_start != uint64_t(block_id) * bt)) return false;

    const auto meta = writer.metadata();
    if (!meta || !same_handle(meta->handle, handle) ||
        meta->logical_id != block_id ||
        meta->original_begin != block.orig_pos_start ||
        meta->valid_tokens != block.n_tokens ||
        (meta->life != HistoricalKvLife::Writing &&
         meta->life != HistoricalKvLife::Live)) return false;
    if (storage) {
        const auto payload = storage->inspect(*repository, handle);
        if (!payload || payload->original_begin != block.orig_pos_start ||
            payload->valid_tokens != block.n_tokens) return false;
    }

    // Allocate every potentially throwing object before moving the sole
    // writer. Failed binding must not silently abandon a Writing generation.
    auto context = std::shared_ptr<WriterContext>(new WriterContext(repository, storage));
    Entry entry;
    entry.binding = {block_id, handle, block.orig_pos_start,
                     block.n_tokens, {}};
    entry.producer = context;
    entry.storage = std::move(storage);
    entry.repository = std::move(repository);
    entry.repository_id = entry.repository->repository_id();
    const auto inserted = entries_.emplace(block_id, std::move(entry));
    if (!inserted.second) return false;
    context->writer_ = std::move(writer);
    producer = std::move(context);
    repository_id_ = inserted.first->second.repository_id;
    return true;
}

bool HistoricalKvShadowDirectory::fixes_span(uint32_t block_id) const noexcept {
    return entries_.count(block_id) != 0;
}

bool HistoricalKvShadowDirectory::append_blocks(
        KvMemStore & store, std::shared_ptr<HistoricalKvRepository> repository,
        std::shared_ptr<HistoricalKvRawStore> storage,
        std::vector<HistoricalKvRepository::Lease> & writers,
        std::vector<Producer> & producers) {
    static_assert(std::is_trivially_copyable<KvMemBlock>::value,
                  "historical append commit requires nonthrowing block copies");
    if (reserved_append_mode_ || !producers.empty() || !repository || repository->closed() || !storage ||
        store.historical_shadow_directory_ != this || store.gc_ ||
        (repository_id_ && repository_id_ != repository->repository_id()) ||
        !tail_span_matches(store) || store.append_tail_room() != 0 ||
        !store.config().block_tokens ||
        writers.size() > size_t(INT32_MAX) - store.block_count()) return false;
    if (writers.empty()) return true;

    // Prepare all fallible metadata allocations before moving any writer or
    // publishing any position. No rollback frees a historical payload here.
    std::map<uint32_t, Entry> pending;
    std::vector<Producer> contexts;
    contexts.reserve(writers.size());
    std::vector<KvMemBlock> blocks;
    blocks.reserve(writers.size());
    uint64_t position = store.total_tokens();
    for (size_t i = 0; i < writers.size(); ++i) {
        const auto & writer = writers[i];
        const auto meta = writer.metadata();
        const uint32_t id = store.block_count() + static_cast<uint32_t>(i);
        if (!repository->owns_writer(writer) || !meta ||
            (meta->life != HistoricalKvLife::Writing && meta->life != HistoricalKvLife::Retired) ||
            meta->logical_id != id || meta->original_begin != position ||
            !meta->valid_tokens || meta->valid_tokens > store.config().block_tokens ||
            position + meta->valid_tokens > std::numeric_limits<uint32_t>::max() ||
            entries_.count(id)) return false;
        const auto payload = storage->inspect(*repository, writer.handle());
        if (!payload || payload->original_begin != position ||
            payload->valid_tokens != meta->valid_tokens) return false;
        auto context = std::shared_ptr<WriterContext>(new WriterContext(repository, storage));
        Entry entry;
        entry.binding = {id, writer.handle(), position, meta->valid_tokens, {}};
        entry.producer = context;
        entry.storage = storage;
        contexts.push_back(std::move(context));
        entry.repository = repository;
        entry.repository_id = repository->repository_id();
        pending.emplace(id, std::move(entry));
        KvMemBlock block;
        block.block_id = id;
        block.orig_pos_start = static_cast<uint32_t>(position);
        block.n_tokens = meta->valid_tokens;
        block.baked_pos = static_cast<int64_t>(position);
        blocks.push_back(block);
        position += meta->valid_tokens;
    }
    const auto count = store.blocks_.size() + blocks.size();
    if (count > store.blocks_.capacity())
        store.blocks_.reserve(std::max(count, std::min<size_t>(INT32_MAX,
            std::max<size_t>(1, store.blocks_.capacity()) * 2)));
    size_t i = 0;
    for (auto & context : contexts) context->writer_ = std::move(writers[i++]);
    // map node transfer uses the same allocator and a nonthrowing integer
    // comparator. The reserved vector only copies plain block descriptors.
    entries_.merge(pending);
    store.blocks_.insert(store.blocks_.end(), blocks.begin(), blocks.end());
    store.total_tokens_ = static_cast<uint32_t>(position);
    repository_id_ = repository->repository_id();
    producers = std::move(contexts);
    return true;
}

std::optional<HistoricalKvShadowDirectory::Binding>
HistoricalKvShadowDirectory::find(uint32_t block_id) const {
    const auto it = entries_.find(block_id);
    if (it == entries_.end() || it->second.state != State::Bound) return std::nullopt;
    auto binding = it->second.binding;
    binding.writer_context = it->second.producer.lock();
    return binding;
}

HistoricalKvShadowDirectory::Observation
HistoricalKvShadowDirectory::observe(uint32_t block_id) const {
    const auto it = entries_.find(block_id);
    if (it == entries_.end()) return {};
    const auto & entry = it->second;
    return {entry.state, entry.binding.handle, entry.repository_id,
            entry.binding.original_begin, entry.binding.valid_tokens};
}

bool HistoricalKvShadowDirectory::allows_new_selection(uint32_t block_id) const {
    const auto it = entries_.find(block_id);
    if (it == entries_.end()) return true;
    const auto & entry = it->second;
    if (entry.state != State::Bound || !entry.repository ||
        entry.repository->repository_id() != entry.repository_id ||
        entry.repository->closed()) return false;
    const auto meta = entry.repository->inspect(entry.binding.handle);
    return meta && same_handle(meta->handle, entry.binding.handle) &&
        (meta->life == HistoricalKvLife::Writing ||
         meta->life == HistoricalKvLife::Live);
}

HistoricalKvShadowDirectory::Access
HistoricalKvShadowDirectory::acquire_read(uint32_t block_id) const {
    return acquire(block_id, HistoricalKvRepository::LeaseKind::Read);
}

HistoricalKvShadowDirectory::Access
HistoricalKvShadowDirectory::acquire_transfer(uint32_t block_id) const {
    return acquire(block_id, HistoricalKvRepository::LeaseKind::Transfer);
}

HistoricalKvShadowDirectory::Access HistoricalKvShadowDirectory::acquire(
        uint32_t block_id, HistoricalKvRepository::LeaseKind kind) const {
    const auto it = entries_.find(block_id);
    if (it == entries_.end()) return {Access::Status::Unmanaged, {}, {}};
    const auto & entry = it->second;
    if (entry.state != State::Bound || !entry.repository ||
        entry.repository->repository_id() != entry.repository_id) return {};
    // Do not inspect then read: retirement can commit between those actions.
    // The repository admits the lease atomically against the exact handle.
    auto lease = kind == HistoricalKvRepository::LeaseKind::Read
        ? entry.repository->acquire_read(entry.binding.handle)
        : entry.repository->acquire_transfer(entry.binding.handle);
    if (!lease) return {};
    return {Access::Status::Acquired, std::move(*lease),
            entry.storage};
}

bool HistoricalKvShadowDirectory::unbind_retired(uint32_t block_id) {
    const auto it = entries_.find(block_id);
    if (it == entries_.end() || it->second.state != State::Bound) return false;
    auto & entry = it->second;
    const auto meta = entry.repository->inspect(entry.binding.handle);
    // Releasing is unreadable. Leave it alone until exact-generation release
    // acknowledgement is available; there is no writer to detach here.
    if (!entry.repository->reclaim_completed(entry.binding.handle, entry.repository_id) &&
        (!meta || !same_handle(meta->handle, entry.binding.handle) ||
         meta->life != HistoricalKvLife::Retired)) return false;
    entry.state = State::Retired;
    return true;
}

bool HistoricalKvShadowDirectory::confirm_reclaimed(uint32_t block_id) {
    const auto it = entries_.find(block_id);
    if (it == entries_.end() || it->second.state == State::Reclaimed) return false;
    auto & entry = it->second;
    if (!entry.repository || entry.repository->repository_id() != entry.repository_id ||
        !entry.repository->reclaim_completed(entry.binding.handle,
                                              entry.repository_id)) return false;
    entry.state = State::Reclaimed;
    return true;
}

bool HistoricalKvShadowDirectory::tail_span_matches(const KvMemStore & store) const {
    if (entries_.empty()) return true;
    const auto & record = entries_.rbegin()->second;
    const auto & entry = record.binding;
    return entry.block_id < store.block_count() &&
           same_span(entry, store.blocks()[entry.block_id]);
}

void HistoricalKvShadowDirectory::validate_append(
        const KvMemStore & store, uint32_t n_tokens) const {
    if (reserved_append_mode_ && n_tokens)
        throw std::logic_error("admitted historical appends require their reserved metadata");
    if (!tail_span_matches(store))
        throw std::logic_error("historical shadow directory has a stale block span");
}

void HistoricalKvShadowDirectory::validate_truncate(
        const KvMemStore & store, uint32_t token_pos) const {
    if (!tail_span_matches(store))
        throw std::logic_error("historical shadow directory has a stale block span");
    if (token_pos >= store.total_tokens()) return;
    for (auto it = entries_.rbegin(); it != entries_.rend(); ++it) {
        const auto & entry = it->second;
        const auto & span = entry.binding;
        if (token_pos >= span.original_begin + span.valid_tokens) break;
        // A confirmed release permits dropping the entire dense block, not
        // keeping a partial prefix whose Raw payload has already gone.
        if (entry.state != State::Reclaimed || token_pos > span.original_begin)
            throw std::logic_error("cannot truncate a historical generation before safe reclamation");
    }
}

void HistoricalKvShadowDirectory::prune_dropped(const KvMemStore & store) {
    const auto first_dropped = entries_.lower_bound(store.block_count());
    for (auto it = first_dropped; it != entries_.end(); ++it) {
        if (it->second.state != State::Reclaimed)
            throw std::logic_error("historical shadow directory dropped an unreclaimed block");
    }
    entries_.erase(first_dropped, entries_.end());
}

size_t HistoricalKvShadowDirectory::allocated_bytes() const noexcept {
    return entries_.size() * (sizeof(decltype(entries_)::value_type) +
                              sizeof(WriterContext) + 3 * sizeof(void *));
}

} // namespace kvmem
