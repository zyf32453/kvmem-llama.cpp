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
    auto context = std::shared_ptr<WriterContext>(new WriterContext(std::move(repository)));
    Binding entry{block_id, handle, block.orig_pos_start,
                  block.n_tokens, std::move(context)};
    const auto inserted = entries_.emplace(block_id, std::move(entry));
    if (!inserted.second) return false;
    inserted.first->second.writer_context->writer_ = std::move(writer);
    repository_id_ = inserted.first->second.writer_context->repository().repository_id();
    return true;
}

std::optional<HistoricalKvShadowDirectory::Binding>
HistoricalKvShadowDirectory::find(uint32_t block_id) const {
    const auto it = entries_.find(block_id);
    if (it == entries_.end()) return std::nullopt;
    return it->second;
}

bool HistoricalKvShadowDirectory::unbind_retired(uint32_t block_id) {
    const auto it = entries_.find(block_id);
    if (it == entries_.end()) return false;
    const auto & entry = it->second;
    const auto & context = *entry.writer_context;
    const auto meta = context.repository().inspect(entry.handle);
    // The retained writer prevents Releasing while this entry exists. A
    // future directory that drops its writer earlier must handle that phase
    // explicitly; never erase a non-Retired entry on an ambiguous observation.
    if (!meta || !same_handle(meta->handle, entry.handle) ||
        meta->life != HistoricalKvLife::Retired) return false;
    entries_.erase(it);
    return true;
}

bool HistoricalKvShadowDirectory::tail_span_matches(const KvMemStore & store) const {
    if (entries_.empty()) return true;
    const auto & entry = entries_.rbegin()->second;
    return entry.block_id < store.block_count() &&
           same_span(entry, store.blocks()[entry.block_id]) &&
           entry.writer_context && entry.writer_context->writer() &&
           same_handle(entry.handle, entry.writer_context->handle());
}

void HistoricalKvShadowDirectory::validate_append(
        const KvMemStore & store, uint32_t n_tokens) const {
    if (!tail_span_matches(store))
        throw std::logic_error("historical shadow directory has a stale block span");
    if (n_tokens == 0 || entries_.empty()) return;
    const auto & tail = entries_.rbegin()->second;
    if (tail.block_id == store.block_count() - 1 &&
        tail.valid_tokens < store.config().block_tokens)
        throw std::logic_error("cannot extend a bound historical generation");
}

void HistoricalKvShadowDirectory::validate_truncate(
        const KvMemStore & store, uint32_t token_pos) const {
    if (!tail_span_matches(store))
        throw std::logic_error("historical shadow directory has a stale block span");
    if (token_pos >= store.total_tokens()) return;
    if (!entries_.empty()) {
        const auto & tail = entries_.rbegin()->second;
        if (token_pos < tail.original_begin + tail.valid_tokens)
            throw std::logic_error("cannot truncate a bound historical generation");
    }
}

size_t HistoricalKvShadowDirectory::allocated_bytes() const noexcept {
    return entries_.size() * (sizeof(decltype(entries_)::value_type) +
                              sizeof(WriterContext) + 3 * sizeof(void *));
}

} // namespace kvmem
