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

bool HistoricalKvShadowDirectory::bind_full_block(
        const KvMemStore & store, uint32_t block_id,
        std::shared_ptr<HistoricalKvRepository> repository,
        HistoricalKvHandle handle,
        HistoricalKvRepository::Lease & writer,
        std::shared_ptr<HistoricalKvRawStore> storage) {
    if (block_id >= store.block_count() ||
        store.blocks()[block_id].n_tokens != store.config().block_tokens) return false;
    return bind_block(store, block_id, std::move(repository), handle, writer, std::move(storage));
}

bool HistoricalKvShadowDirectory::bind_block(
        const KvMemStore & store, uint32_t block_id,
        std::shared_ptr<HistoricalKvRepository> repository,
        HistoricalKvHandle handle,
        HistoricalKvRepository::Lease & writer,
        std::shared_ptr<HistoricalKvRawStore> storage) {
    if (!repository || repository->closed() || !writer ||
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
    auto context = std::shared_ptr<WriterContext>(new WriterContext(repository, std::move(storage)));
    Entry entry;
    entry.binding = {block_id, handle, block.orig_pos_start,
                     block.n_tokens, std::move(context)};
    entry.repository = std::move(repository);
    entry.repository_id = entry.repository->repository_id();
    const auto inserted = entries_.emplace(block_id, std::move(entry));
    if (!inserted.second) return false;
    inserted.first->second.binding.writer_context->writer_ = std::move(writer);
    repository_id_ = inserted.first->second.repository_id;
    return true;
}

bool HistoricalKvShadowDirectory::fixes_span(uint32_t block_id) const noexcept {
    return entries_.count(block_id) != 0;
}

bool HistoricalKvShadowDirectory::append_blocks(
        KvMemStore & store, std::shared_ptr<HistoricalKvRepository> repository,
        std::shared_ptr<HistoricalKvRawStore> storage,
        std::vector<HistoricalKvRepository::Lease> & writers) {
    static_assert(std::is_trivially_copyable<KvMemBlock>::value,
                  "historical append commit requires nonthrowing block copies");
    if (!repository || repository->closed() || !storage ||
        store.historical_shadow_directory_ != this || store.gc_ ||
        (repository_id_ && repository_id_ != repository->repository_id()) ||
        !tail_span_matches(store) || store.append_tail_room() != 0 ||
        !store.config().block_tokens ||
        writers.size() > size_t(INT32_MAX) - store.block_count()) return false;
    if (writers.empty()) return true;

    // Prepare all fallible metadata allocations before moving any writer or
    // publishing any position. No rollback frees a historical payload here.
    std::map<uint32_t, Entry> pending;
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
        entry.binding = {id, writer.handle(), position, meta->valid_tokens, std::move(context)};
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
    for (auto & pair : pending) pair.second.binding.writer_context->writer_ = std::move(writers[i++]);
    // map node transfer uses the same allocator and a nonthrowing integer
    // comparator. The reserved vector only copies plain block descriptors.
    entries_.merge(pending);
    store.blocks_.insert(store.blocks_.end(), blocks.begin(), blocks.end());
    store.total_tokens_ = static_cast<uint32_t>(position);
    repository_id_ = repository->repository_id();
    return true;
}

std::optional<HistoricalKvShadowDirectory::Binding>
HistoricalKvShadowDirectory::find(uint32_t block_id) const {
    const auto it = entries_.find(block_id);
    if (it == entries_.end() || it->second.state != State::Bound) return std::nullopt;
    return it->second.binding;
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
            entry.binding.writer_context->storage()};
}

bool HistoricalKvShadowDirectory::unbind_retired(uint32_t block_id) {
    const auto it = entries_.find(block_id);
    if (it == entries_.end() || it->second.state != State::Bound) return false;
    auto & entry = it->second;
    const auto & context = *entry.binding.writer_context;
    const auto meta = context.repository().inspect(entry.binding.handle);
    // The retained writer prevents Releasing while this entry exists. A
    // future directory that drops its writer earlier must handle that phase
    // explicitly; never erase a non-Retired entry on an ambiguous observation.
    if (!meta || !same_handle(meta->handle, entry.binding.handle) ||
        meta->life != HistoricalKvLife::Retired) return false;
    entry.state = State::Retired;
    entry.binding.writer_context.reset();
    return true;
}

bool HistoricalKvShadowDirectory::confirm_reclaimed(uint32_t block_id) {
    const auto it = entries_.find(block_id);
    if (it == entries_.end() || it->second.state != State::Retired) return false;
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
           same_span(entry, store.blocks()[entry.block_id]) &&
           (record.state == State::Retired || record.state == State::Reclaimed ||
            (entry.writer_context && entry.writer_context->writer() &&
             same_handle(entry.handle, entry.writer_context->handle())));
}

void HistoricalKvShadowDirectory::validate_append(
        const KvMemStore & store, uint32_t n_tokens) const {
    (void) n_tokens;
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
