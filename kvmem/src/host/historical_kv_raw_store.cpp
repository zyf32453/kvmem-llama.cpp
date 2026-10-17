#include "kvmem/historical_kv_raw_store.hpp"

#include <algorithm>
#include <limits>
#include <new>
#include <stdexcept>
#include <utility>

namespace kvmem {
namespace {

size_t multiply(size_t a, size_t b) {
    if (a && b > std::numeric_limits<size_t>::max() / a)
        throw std::invalid_argument("historical KV payload size overflow");
    return a * b;
}

size_t add(size_t a, size_t b) {
    if (b > std::numeric_limits<size_t>::max() - a)
        throw std::invalid_argument("historical KV payload size overflow");
    return a + b;
}

size_t payload_bytes(const std::vector<uint8_t> & k,
                     const std::vector<uint8_t> & v,
                     const std::vector<float> & mean) {
    return add(add(k.capacity(), v.capacity()),
               multiply(mean.capacity(), sizeof(float)));
}

} // namespace

struct HistoricalKvRawStore::Accounting {
    std::atomic<uint64_t> bytes{0};
    std::atomic<uint32_t> allocations{0};
};

HistoricalKvRawStore::Block::~Block() {
    // Destroy payload before returning its actual capacity credit. The shared
    // account also makes an unused reservation safe after its store is gone.
    layers.clear();
    if (accounting) {
        if (payload_accounted) accounting->bytes.fetch_sub(payload_bytes, std::memory_order_relaxed);
        accounting->allocations.fetch_sub(1, std::memory_order_release);
    }
}

HistoricalKvRawStore::PreparedPayload::PreparedPayload() = default;
HistoricalKvRawStore::PreparedPayload::~PreparedPayload() = default;
HistoricalKvRawStore::PreparedPayload::PreparedPayload(PreparedPayload &&) noexcept = default;
HistoricalKvRawStore::PreparedPayload &
HistoricalKvRawStore::PreparedPayload::operator=(PreparedPayload &&) noexcept = default;
HistoricalKvRawStore::PreparedPayload::operator bool() const noexcept { return bool(block_); }
uint64_t HistoricalKvRawStore::PreparedPayload::payload_bytes() const noexcept {
    return block_ ? block_->payload_bytes : 0;
}
uint32_t HistoricalKvRawStore::PreparedPayload::token_capacity() const noexcept {
    return block_ ? block_->valid_tokens : 0;
}

bool HistoricalKvRawStore::same(HistoricalKvHandle a,
                                HistoricalKvHandle b) noexcept {
    return a.incarnation == b.incarnation && a.slot == b.slot &&
           a.generation == b.generation;
}

bool HistoricalKvRawStore::required_layer(uint32_t layer) const noexcept {
    return layer < config_.n_layer && (config_.attention_layers.empty() ||
        std::binary_search(config_.attention_layers.begin(), config_.attention_layers.end(), layer));
}

void HistoricalKvRawStore::mark_layer_complete(Block & block,
                                                Layer & layer) const noexcept {
    if (layer.written || layer.k_written_tokens != block.valid_tokens ||
        layer.v_written_tokens != block.valid_tokens ||
        (config_.mean_k_elements &&
         layer.mean_written_tokens != block.valid_tokens)) return;
    layer.written = true;
    ++block.written_layers;
}

HistoricalKvRawStore::HistoricalKvRawStore(HistoricalKvRawStoreConfig config,
                                           uint32_t capacity)
    : config_(config), accounting_(std::make_shared<Accounting>()) {
    if (!capacity || !config_.n_layer || !config_.max_tokens_per_slot ||
        !config_.packed_k_bytes_per_token || !config_.packed_v_bytes_per_token)
        throw std::invalid_argument("invalid historical KV raw store configuration");
    if (!std::is_sorted(config_.attention_layers.begin(), config_.attention_layers.end()) ||
        std::adjacent_find(config_.attention_layers.begin(), config_.attention_layers.end()) !=
            config_.attention_layers.end() ||
        (!config_.attention_layers.empty() && config_.attention_layers.back() >= config_.n_layer))
        throw std::invalid_argument("invalid historical KV attention layer IDs");

    // Bound every successful reservation by a finite, representable maximum.
    const size_t k_max = multiply(config_.max_tokens_per_slot,
                                  config_.packed_k_bytes_per_token);
    const size_t v_max = multiply(config_.max_tokens_per_slot,
                                  config_.packed_v_bytes_per_token);
    const size_t mean_max = multiply(config_.mean_k_elements, sizeof(float));
    const size_t layer_max = add(add(k_max, v_max), mean_max);
    const size_t block_max = multiply(layer_max, config_.n_layer);
    (void) multiply(block_max, capacity);

    slots_.reserve(capacity);
    for (uint32_t i = 0; i < capacity; ++i)
        slots_.push_back(std::make_unique<Slot>());
}

HistoricalKvRawStore::~HistoricalKvRawStore() = default;

bool HistoricalKvRawStore::reserve(
    const HistoricalKvRepository::Lease & writer,
    HistoricalKvHandle handle, uint64_t original_begin, uint32_t valid_tokens) {
    const auto meta = writer.metadata();
    if (!meta || !writer.is_writer() || !same(writer.handle(), handle) ||
        handle.slot >= slots_.size() || !valid_tokens || valid_tokens > config_.max_tokens_per_slot ||
        meta->original_begin != original_begin || meta->valid_tokens != valid_tokens ||
        (meta->life != HistoricalKvLife::Writing && meta->life != HistoricalKvLife::Retired)) return false;
    {
        const auto & slot = *slots_[handle.slot];
        std::lock_guard<std::mutex> lock(slot.mu);
        if (slot.block || slot.releasing) return false;
    }
    auto prepared = prepare_payload(valid_tokens);
    return install_payload(writer, handle, original_begin, valid_tokens, prepared);
}

HistoricalKvRawStore::PreparedPayload HistoricalKvRawStore::prepare_payload(uint32_t token_capacity) {
    if (!token_capacity || token_capacity > config_.max_tokens_per_slot)
        throw std::invalid_argument("invalid historical KV payload reservation size");
    PreparedPayload prepared;
    auto block = std::make_unique<Block>();
    uint32_t allocations = accounting_->allocations.load(std::memory_order_acquire);
    for (;;) {
        if (allocations >= slots_.size()) throw std::bad_alloc();
        if (accounting_->allocations.compare_exchange_weak(allocations, allocations + 1,
                std::memory_order_acq_rel, std::memory_order_acquire)) break;
    }
    block->accounting = accounting_;
    block->valid_tokens = token_capacity;
    block->layers.resize(config_.n_layer);
    const size_t k_bytes = multiply(token_capacity, config_.packed_k_bytes_per_token);
    const size_t v_bytes = multiply(token_capacity, config_.packed_v_bytes_per_token);
    for (uint32_t il = 0; il < config_.n_layer; ++il) {
        if (!required_layer(il)) continue;
        auto & layer = block->layers[il];
        layer.packed_k.resize(k_bytes);
        layer.packed_v.resize(v_bytes);
        layer.mean_k_sum.resize(config_.mean_k_elements);
        block->payload_bytes = add(static_cast<size_t>(block->payload_bytes),
                                   payload_bytes(layer.packed_k, layer.packed_v, layer.mean_k_sum));
    }
    uint64_t previous = accounting_->bytes.load(std::memory_order_relaxed);
    for (;;) {
        if (block->payload_bytes > std::numeric_limits<uint64_t>::max() - previous)
            throw std::overflow_error("historical KV aggregate payload size overflow");
        if (accounting_->bytes.compare_exchange_weak(previous, previous + block->payload_bytes,
                std::memory_order_relaxed, std::memory_order_relaxed)) break;
    }
    block->payload_accounted = true;
    prepared.block_ = std::move(block);
    return prepared;
}

bool HistoricalKvRawStore::install_payload(
    const HistoricalKvRepository::Lease & writer,
    HistoricalKvHandle handle, uint64_t original_begin, uint32_t valid_tokens,
    PreparedPayload & prepared) {
    if (!writer || !writer.is_writer() || !writer.repository_id() ||
        !same(writer.handle(), handle) ||
        !handle.incarnation || !handle.generation || handle.slot >= slots_.size() ||
        !prepared.block_ || prepared.block_->accounting != accounting_ ||
        !valid_tokens || valid_tokens > prepared.token_capacity() ||
        original_begin > std::numeric_limits<uint64_t>::max() - valid_tokens)
        return false;
    const auto meta = writer.metadata();
    if (!meta || meta->original_begin != original_begin ||
        meta->valid_tokens != valid_tokens ||
        (meta->life != HistoricalKvLife::Writing && meta->life != HistoricalKvLife::Retired)) return false;

    Slot & slot = *slots_[handle.slot];
    std::lock_guard<std::mutex> lock(slot.mu);
    if (slot.block || slot.releasing) return false;
    if (slot.last_released_repository_id == writer.repository_id() &&
        slot.last_released.incarnation == handle.incarnation &&
        handle.generation <= slot.last_released.generation)
        return false;

    auto & block = prepared.block_;
    block->handle = handle;
    block->repository_id = writer.repository_id();
    block->original_begin = original_begin;
    block->valid_tokens = valid_tokens;
    const size_t k_bytes = multiply(valid_tokens, config_.packed_k_bytes_per_token);
    const size_t v_bytes = multiply(valid_tokens, config_.packed_v_bytes_per_token);
    for (uint32_t il = 0; il < config_.n_layer; ++il) {
        if (!required_layer(il)) continue;
        auto & layer = block->layers[il];
        layer.packed_k.resize(k_bytes);
        layer.packed_v.resize(v_bytes);
        // resize only shrinks the logical range; capacity stays owned. This
        // is how one worst-case reservation backs a short/non-aligned tail.
    }
    slot.block = std::move(block);
    return true;
}

bool HistoricalKvRawStore::write_layer(
    const HistoricalKvRepository::Lease & writer,
    HistoricalKvHandle handle, uint32_t layer,
    const std::vector<uint8_t> & packed_k,
    const std::vector<uint8_t> & packed_v,
    const std::vector<float> & mean_k_sum,
    uint32_t mean_tokens) {
    if (!writer || !writer.is_writer() || !writer.repository_id() ||
        !same(writer.handle(), handle) ||
        !handle.incarnation || !handle.generation || handle.slot >= slots_.size() ||
        !required_layer(layer)) return false;
    Slot & slot = *slots_[handle.slot];
    std::lock_guard<std::mutex> lock(slot.mu);
    // The same slot lock protects publish_complete. A published layer is
    // immutable; a retired old writer may still finish admitted writes.
    const auto meta = writer.metadata();
    if (!meta || (meta->life != HistoricalKvLife::Writing &&
                  meta->life != HistoricalKvLife::Retired)) return false;
    Block * block = slot.block.get();
    if (!block || block->repository_id != writer.repository_id() ||
        !same(block->handle, handle) ||
        (config_.mean_k_elements ? mean_tokens != block->valid_tokens
                                 : mean_tokens != 0)) return false;
    Layer & target = block->layers[layer];
    if (target.written || target.k_written_tokens != 0 ||
        target.v_written_tokens != 0 || target.mean_written_tokens != 0 ||
        packed_k.size() != target.packed_k.size() ||
        packed_v.size() != target.packed_v.size() ||
        mean_k_sum.size() != target.mean_k_sum.size() ||
        (config_.mean_k_elements == 0 && mean_tokens != 0)) return false;
    std::copy(packed_k.begin(), packed_k.end(), target.packed_k.begin());
    std::copy(packed_v.begin(), packed_v.end(), target.packed_v.begin());
    std::copy(mean_k_sum.begin(), mean_k_sum.end(), target.mean_k_sum.begin());
    target.mean_tokens = mean_tokens;
    target.k_written_tokens = block->valid_tokens;
    target.v_written_tokens = block->valid_tokens;
    target.mean_written_tokens = config_.mean_k_elements ? block->valid_tokens : 0;
    mark_layer_complete(*block, target);
    return true;
}

bool HistoricalKvRawStore::write_slice(
    const HistoricalKvRepository::Lease & writer,
    HistoricalKvHandle handle, uint32_t layer,
    uint32_t token_offset, uint32_t token_count,
    const std::vector<uint8_t> & packed_k,
    const std::vector<uint8_t> & packed_v,
    const std::vector<float> & mean_k_sum,
    uint32_t mean_tokens) {
    if (!writer || !writer.is_writer() || !writer.repository_id() ||
        !same(writer.handle(), handle) || !handle.incarnation ||
        !handle.generation || handle.slot >= slots_.size() ||
        !required_layer(layer) || !token_count) return false;
    Slot & slot = *slots_[handle.slot];
    std::lock_guard<std::mutex> lock(slot.mu);
    const auto meta = writer.metadata();
    if (!meta || (meta->life != HistoricalKvLife::Writing &&
                  meta->life != HistoricalKvLife::Retired)) return false;
    Block * block = slot.block.get();
    if (!block || block->repository_id != writer.repository_id() ||
        !same(block->handle, handle) ||
        token_offset >= block->valid_tokens ||
        token_count > block->valid_tokens - token_offset) return false;

    Layer & target = block->layers[layer];
    if (target.written || token_offset != target.k_written_tokens ||
        token_offset != target.v_written_tokens ||
        (config_.mean_k_elements &&
         token_offset != target.mean_written_tokens) ||
        packed_k.size() != static_cast<size_t>(token_count) *
                           config_.packed_k_bytes_per_token ||
        packed_v.size() != static_cast<size_t>(token_count) *
                           config_.packed_v_bytes_per_token ||
        mean_k_sum.size() != target.mean_k_sum.size() ||
        (config_.mean_k_elements ? mean_tokens != token_count
                                 : mean_tokens != 0)) return false;

    const size_t k_offset = static_cast<size_t>(token_offset) *
                            config_.packed_k_bytes_per_token;
    const size_t v_offset = static_cast<size_t>(token_offset) *
                            config_.packed_v_bytes_per_token;
    std::copy(packed_k.begin(), packed_k.end(), target.packed_k.begin() + k_offset);
    std::copy(packed_v.begin(), packed_v.end(), target.packed_v.begin() + v_offset);
    for (size_t i = 0; i < mean_k_sum.size(); ++i)
        target.mean_k_sum[i] += mean_k_sum[i];
    target.mean_tokens += mean_tokens;
    target.k_written_tokens += token_count;
    target.v_written_tokens += token_count;
    if (config_.mean_k_elements) target.mean_written_tokens += token_count;
    mark_layer_complete(*block, target);
    return true;
}

bool HistoricalKvRawStore::write_packed_k_slice(
    const HistoricalKvRepository::Lease & writer,
    HistoricalKvHandle handle, uint32_t layer, uint32_t token_offset,
    uint32_t token_count, const std::vector<uint8_t> & packed_k) {
    return write_packed_k_slice(writer, handle, layer, token_offset, token_count,
                                packed_k.data(), packed_k.size());
}

bool HistoricalKvRawStore::write_packed_k_slice(
    const HistoricalKvRepository::Lease & writer,
    HistoricalKvHandle handle, uint32_t layer, uint32_t token_offset,
    uint32_t token_count, const uint8_t * packed_k, size_t bytes) {
    if (!writer || !writer.is_writer() || !writer.repository_id() ||
        !same(writer.handle(), handle) || !handle.incarnation ||
        !handle.generation || handle.slot >= slots_.size() ||
        !required_layer(layer) || !token_count || !packed_k) return false;
    Slot & slot = *slots_[handle.slot];
    std::lock_guard<std::mutex> lock(slot.mu);
    const auto meta = writer.metadata();
    if (!meta || (meta->life != HistoricalKvLife::Writing &&
                  meta->life != HistoricalKvLife::Retired)) return false;
    Block * block = slot.block.get();
    if (!block || block->repository_id != writer.repository_id() ||
        !same(block->handle, handle) || token_offset >= block->valid_tokens ||
        token_count > block->valid_tokens - token_offset) return false;
    Layer & target = block->layers[layer];
    if (target.written || token_offset != target.k_written_tokens ||
        bytes != static_cast<size_t>(token_count) *
                           config_.packed_k_bytes_per_token) return false;
    const size_t offset = static_cast<size_t>(token_offset) *
                          config_.packed_k_bytes_per_token;
    std::copy_n(packed_k, bytes, target.packed_k.begin() + offset);
    target.k_written_tokens += token_count;
    mark_layer_complete(*block, target);
    return true;
}

bool HistoricalKvRawStore::write_packed_v_slice(
    const HistoricalKvRepository::Lease & writer,
    HistoricalKvHandle handle, uint32_t layer, uint32_t token_offset,
    uint32_t token_count, const std::vector<uint8_t> & packed_v) {
    return write_packed_v_slice(writer, handle, layer, token_offset, token_count,
                                packed_v.data(), packed_v.size());
}

bool HistoricalKvRawStore::write_packed_v_slice(
    const HistoricalKvRepository::Lease & writer,
    HistoricalKvHandle handle, uint32_t layer, uint32_t token_offset,
    uint32_t token_count, const uint8_t * packed_v, size_t bytes) {
    if (!writer || !writer.is_writer() || !writer.repository_id() ||
        !same(writer.handle(), handle) || !handle.incarnation ||
        !handle.generation || handle.slot >= slots_.size() ||
        !required_layer(layer) || !token_count || !packed_v) return false;
    Slot & slot = *slots_[handle.slot];
    std::lock_guard<std::mutex> lock(slot.mu);
    const auto meta = writer.metadata();
    if (!meta || (meta->life != HistoricalKvLife::Writing &&
                  meta->life != HistoricalKvLife::Retired)) return false;
    Block * block = slot.block.get();
    if (!block || block->repository_id != writer.repository_id() ||
        !same(block->handle, handle) || token_offset >= block->valid_tokens ||
        token_count > block->valid_tokens - token_offset) return false;
    Layer & target = block->layers[layer];
    if (target.written || token_offset != target.v_written_tokens ||
        bytes != static_cast<size_t>(token_count) *
                           config_.packed_v_bytes_per_token) return false;
    const size_t offset = static_cast<size_t>(token_offset) *
                          config_.packed_v_bytes_per_token;
    std::copy_n(packed_v, bytes, target.packed_v.begin() + offset);
    target.v_written_tokens += token_count;
    mark_layer_complete(*block, target);
    return true;
}

bool HistoricalKvRawStore::write_mean_k_slice(
    const HistoricalKvRepository::Lease & writer,
    HistoricalKvHandle handle, uint32_t layer, uint32_t token_offset,
    uint32_t token_count, const std::vector<float> & mean_k_sum) {
    if (!writer || !writer.is_writer() || !writer.repository_id() ||
        !same(writer.handle(), handle) || !handle.incarnation ||
        !handle.generation || handle.slot >= slots_.size() ||
        !required_layer(layer) || !config_.mean_k_elements ||
        !token_count) return false;
    Slot & slot = *slots_[handle.slot];
    std::lock_guard<std::mutex> lock(slot.mu);
    const auto meta = writer.metadata();
    if (!meta || (meta->life != HistoricalKvLife::Writing &&
                  meta->life != HistoricalKvLife::Retired)) return false;
    Block * block = slot.block.get();
    if (!block || block->repository_id != writer.repository_id() ||
        !same(block->handle, handle) || token_offset >= block->valid_tokens ||
        token_count > block->valid_tokens - token_offset) return false;
    Layer & target = block->layers[layer];
    if (target.written || token_offset != target.mean_written_tokens ||
        mean_k_sum.size() != config_.mean_k_elements) return false;
    for (size_t i = 0; i < mean_k_sum.size(); ++i)
        target.mean_k_sum[i] += mean_k_sum[i];
    target.mean_tokens += token_count;
    target.mean_written_tokens += token_count;
    mark_layer_complete(*block, target);
    return true;
}

std::optional<HistoricalKvRawStore::WriteProgress>
HistoricalKvRawStore::write_progress(
    const HistoricalKvRepository::Lease & writer,
    HistoricalKvHandle handle, uint32_t layer) const {
    if (!writer || !writer.is_writer() || !writer.repository_id() ||
        !same(writer.handle(), handle) || handle.slot >= slots_.size() ||
        !required_layer(layer)) return std::nullopt;
    const Slot & slot = *slots_[handle.slot];
    std::lock_guard<std::mutex> lock(slot.mu);
    const Block * block = slot.block.get();
    if (!block || block->repository_id != writer.repository_id() ||
        !same(block->handle, handle)) return std::nullopt;
    const auto & source = block->layers[layer];
    return WriteProgress{source.k_written_tokens, source.v_written_tokens,
                         source.mean_written_tokens};
}

bool HistoricalKvRawStore::publish_complete(
    HistoricalKvRepository & repository,
    const HistoricalKvRepository::Lease & writer,
    HistoricalKvHandle handle) {
    if (!repository.owns_writer(writer) || !same(writer.handle(), handle) ||
        !handle.incarnation || !handle.generation ||
        handle.slot >= slots_.size()) return false;
    Slot & slot = *slots_[handle.slot];
    std::lock_guard<std::mutex> lock(slot.mu);
    const Block * block = slot.block.get();
    if (!block || block->repository_id != repository.repository_id() ||
        !same(block->handle, handle) ||
        block->written_layers != (config_.attention_layers.empty()
            ? config_.n_layer : config_.attention_layers.size())) return false;
    return repository.publish_write(writer);
}

std::optional<HistoricalKvRawStore::LayerSnapshot>
HistoricalKvRawStore::read_layer(
    const HistoricalKvRepository::Lease & reader,
    HistoricalKvHandle handle, uint32_t layer) const {
    if (!reader || !reader.repository_id() ||
        !same(reader.handle(), handle) ||
        !handle.incarnation || !handle.generation || handle.slot >= slots_.size() ||
        !required_layer(layer)) return std::nullopt;
    const Slot & slot = *slots_[handle.slot];
    std::lock_guard<std::mutex> lock(slot.mu);
    const Block * block = slot.block.get();
    if (!block || block->repository_id != reader.repository_id() ||
        !same(block->handle, handle)) return std::nullopt;
    const Layer & source = block->layers[layer];
    if (!source.written) return std::nullopt;
    return LayerSnapshot{source.packed_k, source.packed_v,
                         source.mean_k_sum, source.mean_tokens};
}

std::optional<HistoricalKvRawStore::BlockInfo>
HistoricalKvRawStore::inspect(const HistoricalKvRepository & repository,
                              HistoricalKvHandle handle) const {
    if (!handle.incarnation || !handle.generation || handle.slot >= slots_.size())
        return std::nullopt;
    const Slot & slot = *slots_[handle.slot];
    std::lock_guard<std::mutex> lock(slot.mu);
    const Block * block = slot.block.get();
    if (!block || block->repository_id != repository.repository_id() ||
        !same(block->handle, handle)) return std::nullopt;
    return BlockInfo{block->handle, block->original_begin,
                     block->valid_tokens, block->written_layers,
                     block->payload_bytes};
}

bool HistoricalKvRawStore::copy_mean_k(
    const HistoricalKvRepository::Lease & reader,
    HistoricalKvHandle handle, uint32_t layer, float * output, size_t elements) const {
    if (!reader || !reader.repository_id() || !same(reader.handle(), handle) ||
        handle.slot >= slots_.size() || !required_layer(layer) || !output ||
        !config_.mean_k_elements || elements != config_.mean_k_elements) return false;
    const Slot & slot = *slots_[handle.slot];
    std::lock_guard<std::mutex> lock(slot.mu);
    const Block * block = slot.block.get();
    if (!block || block->repository_id != reader.repository_id() ||
        !same(block->handle, handle)) return false;
    const auto & source = block->layers[layer];
    if (!source.written || !source.mean_tokens) return false;
    for (size_t i = 0; i < elements; ++i)
        output[i] = source.mean_k_sum[i] / static_cast<float>(source.mean_tokens);
    return true;
}

HistoricalKvRawStore::ReleaseResult
HistoricalKvRawStore::release(const HistoricalKvRepository & repository,
                              HistoricalKvHandle handle) noexcept {
    if (!handle.incarnation || !handle.generation || handle.slot >= slots_.size())
        return {ReleaseCode::Stale, 0};

    Slot & slot = *slots_[handle.slot];
    std::unique_ptr<Block> retired;
    uint64_t bytes = 0;
    {
        std::lock_guard<std::mutex> lock(slot.mu);
        if (slot.releasing) return {ReleaseCode::Busy, 0};
        if (!slot.block) {
            if (slot.last_released_repository_id == repository.repository_id() &&
                same(slot.last_released, handle))
                return {ReleaseCode::AlreadyReleased, 0};
            if (slot.last_released_repository_id == repository.repository_id() &&
                slot.last_released.incarnation == handle.incarnation &&
                handle.generation <= slot.last_released.generation)
                return {ReleaseCode::Stale, 0};
            return repository.is_releasing(handle)
                ? ReleaseResult{ReleaseCode::Absent, 0}
                : ReleaseResult{ReleaseCode::Busy, 0};
        }
        if (slot.block->repository_id != repository.repository_id() ||
            !same(slot.block->handle, handle))
            return {ReleaseCode::Stale, 0};
        if (!repository.is_releasing(handle))
            return {ReleaseCode::Busy, 0};
        retired = std::move(slot.block);
        bytes = retired->payload_bytes;
        slot.releasing = true;
    }

    // The large K/V and mean allocations are destroyed by the caller's
    // reclaimer thread, outside all store locks. The slot remains unavailable
    // until that physical destruction is complete.
    retired.reset();
    {
        std::lock_guard<std::mutex> lock(slot.mu);
        slot.last_released = handle;
        slot.last_released_repository_id = repository.repository_id();
        slot.releasing = false;
    }
    return {ReleaseCode::Released, bytes};
}

uint64_t HistoricalKvRawStore::allocated_payload_bytes() const noexcept {
    return accounting_->bytes.load(std::memory_order_relaxed);
}

uint32_t HistoricalKvRawStore::capacity() const noexcept {
    return static_cast<uint32_t>(slots_.size());
}

const HistoricalKvRawStoreConfig & HistoricalKvRawStore::config() const noexcept {
    return config_;
}

} // namespace kvmem
