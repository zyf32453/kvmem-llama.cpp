#include "kvmem/historical_kv_shadow_directory.hpp"

#include "kvmem/kvmem_store.hpp"

#include <stdexcept>
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
        HistoricalKvRepository::Lease & writer) {
    if (!repository || repository->closed() || !writer ||
        !repository->owns_writer(writer) ||
        !same_handle(handle, writer.handle()) ||
        writer.repository_id() != repository->repository_id() ||
        (repository_id_ != 0 && repository_id_ != repository->repository_id()) ||
        block_id >= store.block_count() || entries_.count(block_id) != 0) return false;

    const auto & block = store.blocks()[block_id];
    if (store.config().block_tokens == 0 ||
        block.n_tokens != store.config().block_tokens) return false;

    const auto meta = writer.metadata();
    if (!meta || !same_handle(meta->handle, handle) ||
        meta->logical_id != block_id ||
        meta->original_begin != block.orig_pos_start ||
        meta->valid_tokens != block.n_tokens ||
        (meta->life != HistoricalKvLife::Writing &&
         meta->life != HistoricalKvLife::Live)) return false;

    // Allocate every potentially throwing object before moving the sole
    // writer. Failed binding must not silently abandon a Writing generation.
    auto context = std::shared_ptr<WriterContext>(new WriterContext(repository));
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
    if (!tail_span_matches(store))
        throw std::logic_error("historical shadow directory has a stale block span");
    if (n_tokens == 0 || entries_.empty()) return;
    const auto & tail = entries_.rbegin()->second.binding;
    if (tail.block_id == store.block_count() - 1 &&
        tail.valid_tokens < store.config().block_tokens)
        throw std::logic_error("cannot extend a bound historical generation");
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
