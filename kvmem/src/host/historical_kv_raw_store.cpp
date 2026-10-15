#include "kvmem/historical_kv_raw_store.hpp"

#include <algorithm>
#include <limits>
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

bool HistoricalKvRawStore::same(HistoricalKvHandle a,
                                HistoricalKvHandle b) noexcept {
    return a.incarnation == b.incarnation && a.slot == b.slot &&
           a.generation == b.generation;
}

HistoricalKvRawStore::HistoricalKvRawStore(HistoricalKvRawStoreConfig config,
                                           uint32_t capacity)
    : config_(config) {
    if (!capacity || !config_.n_layer || !config_.max_tokens_per_slot ||
        !config_.packed_k_bytes_per_token || !config_.packed_v_bytes_per_token)
        throw std::invalid_argument("invalid historical KV raw store configuration");

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
    if (!writer || !writer.is_writer() || !same(writer.handle(), handle) ||
        !handle.incarnation || !handle.generation || handle.slot >= slots_.size() ||
        !valid_tokens || valid_tokens > config_.max_tokens_per_slot ||
        original_begin > std::numeric_limits<uint64_t>::max() - valid_tokens)
        return false;
    const auto meta = writer.metadata();
    if (!meta || meta->original_begin != original_begin ||
        meta->valid_tokens != valid_tokens) return false;

    Slot & slot = *slots_[handle.slot];
    std::lock_guard<std::mutex> lock(slot.mu);
    if (slot.block || slot.releasing) return false;
    if (slot.last_released.incarnation == handle.incarnation &&
        handle.generation <= slot.last_released.generation)
        return false;

    // Only install after allocation succeeds. If allocation fails, the
    // repository's release callback can safely treat Absent as a no-op.
    auto block = std::make_unique<Block>();
    block->handle = handle;
    block->original_begin = original_begin;
    block->valid_tokens = valid_tokens;
    block->layers.resize(config_.n_layer);
    const size_t k_bytes = multiply(valid_tokens, config_.packed_k_bytes_per_token);
    const size_t v_bytes = multiply(valid_tokens, config_.packed_v_bytes_per_token);
    for (auto & layer : block->layers) {
        layer.packed_k.resize(k_bytes);
        layer.packed_v.resize(v_bytes);
        layer.mean_k_sum.resize(config_.mean_k_elements);
        block->payload_bytes = add(static_cast<size_t>(block->payload_bytes),
                                   payload_bytes(layer.packed_k, layer.packed_v,
                                                 layer.mean_k_sum));
    }
    uint64_t previous = allocated_payload_bytes_.load(std::memory_order_relaxed);
    for (;;) {
        if (block->payload_bytes > std::numeric_limits<uint64_t>::max() - previous)
            throw std::overflow_error("historical KV aggregate payload size overflow");
        if (allocated_payload_bytes_.compare_exchange_weak(
                previous, previous + block->payload_bytes,
                std::memory_order_relaxed, std::memory_order_relaxed)) break;
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
    if (!writer || !writer.is_writer() || !same(writer.handle(), handle) ||
        !handle.incarnation || !handle.generation || handle.slot >= slots_.size() ||
        layer >= config_.n_layer) return false;
    Slot & slot = *slots_[handle.slot];
    std::lock_guard<std::mutex> lock(slot.mu);
    Block * block = slot.block.get();
    if (!block || !same(block->handle, handle) ||
        mean_tokens > block->valid_tokens) return false;
    Layer & target = block->layers[layer];
    if (target.written || packed_k.size() != target.packed_k.size() ||
        packed_v.size() != target.packed_v.size() ||
        mean_k_sum.size() != target.mean_k_sum.size() ||
        (config_.mean_k_elements == 0 && mean_tokens != 0)) return false;
    std::copy(packed_k.begin(), packed_k.end(), target.packed_k.begin());
    std::copy(packed_v.begin(), packed_v.end(), target.packed_v.begin());
    std::copy(mean_k_sum.begin(), mean_k_sum.end(), target.mean_k_sum.begin());
    target.mean_tokens = mean_tokens;
    target.written = true;
    ++block->written_layers;
    return true;
}

std::optional<HistoricalKvRawStore::LayerSnapshot>
HistoricalKvRawStore::read_layer(
    const HistoricalKvRepository::Lease & reader,
    HistoricalKvHandle handle, uint32_t layer) const {
    if (!reader || !same(reader.handle(), handle) ||
        !handle.incarnation || !handle.generation || handle.slot >= slots_.size() ||
        layer >= config_.n_layer) return std::nullopt;
    const Slot & slot = *slots_[handle.slot];
    std::lock_guard<std::mutex> lock(slot.mu);
    const Block * block = slot.block.get();
    if (!block || !same(block->handle, handle)) return std::nullopt;
    const Layer & source = block->layers[layer];
    if (!source.written) return std::nullopt;
    return LayerSnapshot{source.packed_k, source.packed_v,
                         source.mean_k_sum, source.mean_tokens};
}

std::optional<HistoricalKvRawStore::BlockInfo>
HistoricalKvRawStore::inspect(HistoricalKvHandle handle) const {
    if (!handle.incarnation || !handle.generation || handle.slot >= slots_.size())
        return std::nullopt;
    const Slot & slot = *slots_[handle.slot];
    std::lock_guard<std::mutex> lock(slot.mu);
    const Block * block = slot.block.get();
    if (!block || !same(block->handle, handle)) return std::nullopt;
    return BlockInfo{block->handle, block->original_begin,
                     block->valid_tokens, block->written_layers,
                     block->payload_bytes};
}

HistoricalKvRawStore::ReleaseResult
HistoricalKvRawStore::release(HistoricalKvHandle handle) noexcept {
    if (!handle.incarnation || !handle.generation || handle.slot >= slots_.size())
        return {ReleaseCode::Stale, 0};

    Slot & slot = *slots_[handle.slot];
    std::unique_ptr<Block> retired;
    uint64_t bytes = 0;
    {
        std::lock_guard<std::mutex> lock(slot.mu);
        if (slot.releasing) return {ReleaseCode::Busy, 0};
        if (!slot.block) {
            if (same(slot.last_released, handle))
                return {ReleaseCode::AlreadyReleased, 0};
            if (slot.last_released.incarnation == handle.incarnation &&
                handle.generation <= slot.last_released.generation)
                return {ReleaseCode::Stale, 0};
            return {ReleaseCode::Absent, 0};
        }
        if (!same(slot.block->handle, handle))
            return {ReleaseCode::Stale, 0};
        retired = std::move(slot.block);
        bytes = retired->payload_bytes;
        slot.releasing = true;
    }

    // The large K/V and mean allocations are destroyed by the caller's
    // reclaimer thread, outside all store locks. The slot remains unavailable
    // until that physical destruction is complete.
    retired.reset();
    allocated_payload_bytes_.fetch_sub(bytes, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> lock(slot.mu);
        slot.last_released = handle;
        slot.releasing = false;
    }
    return {ReleaseCode::Released, bytes};
}

uint64_t HistoricalKvRawStore::allocated_payload_bytes() const noexcept {
    return allocated_payload_bytes_.load(std::memory_order_relaxed);
}

uint32_t HistoricalKvRawStore::capacity() const noexcept {
    return static_cast<uint32_t>(slots_.size());
}

const HistoricalKvRawStoreConfig & HistoricalKvRawStore::config() const noexcept {
    return config_;
}

} // namespace kvmem
