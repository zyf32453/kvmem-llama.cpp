#include "kvmem/historical_kv_raw_store.hpp"
#if defined(_WIN32)
#include "kvmem/nvme_kv_tier_win.hpp"
#else
#include "kvmem/nvme_kv_tier.hpp"
#endif

#include <algorithm>
#include <cstring>
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
    std::atomic<uint64_t> nvme_bytes{0};
    std::atomic<uint64_t> pinned_bytes{0};
    std::atomic<uint32_t> allocations{0};
};

struct HistoricalKvRawStore::DiskAllocation {
    std::shared_ptr<NvmeKvTier> tier;
    std::shared_ptr<Accounting> accounting;
    NvmeAllocationHandle handle;
    uint64_t bytes = 0;
    std::atomic<bool> returned{false};
    void return_credit() noexcept {
        if (!returned.exchange(true, std::memory_order_acq_rel) && accounting)
            accounting->nvme_bytes.fetch_sub(bytes, std::memory_order_relaxed);
    }
    ~DiskAllocation() {
        // Every IO owner also owns this object and drops its IO lease first.
        // This is private reservation rollback or already-acknowledged final
        // metadata cleanup; live GC uses the nonblocking exact callback.
        if (tier && handle.generation) {
            const auto result = tier->release_owned_allocation(handle);
            if (result != NvmeReleaseResult::Released && result != NvmeReleaseResult::AlreadyReleased &&
                result != NvmeReleaseResult::Stale) std::terminate();
            return_credit();
        }
    }
};
struct HistoricalKvRawStore::DiskProducer {
    std::shared_ptr<DiskAllocation> allocation;
    NvmeIoLease io;
};
struct HistoricalKvRawStore::PinnedAllocation {
    std::shared_ptr<PinnedKvTier> tier;
    std::shared_ptr<Accounting> accounting;
    std::shared_ptr<uint8_t> data;
    PinnedAllocationHandle handle;
    uint64_t bytes = 0;
    std::atomic<bool> memory_returned{false};
    std::atomic<bool> slot_returned{false};
    void return_memory() noexcept {
        if (memory_returned.exchange(true, std::memory_order_acq_rel)) return;
        data.reset();
        if (accounting) accounting->pinned_bytes.fetch_sub(bytes, std::memory_order_relaxed);
    }
    ~PinnedAllocation() {
        // An old read holds this allocation and its IO lease. Destruction is
        // therefore either private rollback or follows completed GC release.
        if (tier && handle.generation && !slot_returned.load(std::memory_order_acquire)) {
            auto result = tier->close_owned_allocation(handle);
            if (result != PinnedReleaseResult::Released && result != PinnedReleaseResult::AlreadyReleased &&
                result != PinnedReleaseResult::Stale) std::terminate();
            // Rollback obeys the same physical-capacity ordering as GC.
            // Keep the extent unavailable while its deleter is still running.
            return_memory();
            result = tier->release_owned_allocation(handle);
            if (result != PinnedReleaseResult::Released && result != PinnedReleaseResult::AlreadyReleased &&
                result != PinnedReleaseResult::Stale) std::terminate();
        }
        return_memory();
    }
};
struct HistoricalKvRawStore::PinnedProducer {
    std::shared_ptr<PinnedAllocation> allocation;
    PinnedIoLease io;
};

HistoricalKvRawStore::NvmeRead & HistoricalKvRawStore::NvmeRead::operator=(NvmeRead && other) noexcept {
    if (this != &other) {
        // Old IO must drain before its final allocation owner can disappear.
        io_.reset();
        allocation_.reset();
        identity_ = std::move(other.identity_);
        accounting_ = std::move(other.accounting_);
        allocation_ = std::move(other.allocation_);
        io_ = std::move(other.io_);
        valid_tokens_ = other.valid_tokens_;
    }
    return *this;
}

HistoricalKvRawStore::PinnedRead & HistoricalKvRawStore::PinnedRead::operator=(PinnedRead && other) noexcept {
    if (this != &other) {
        io_.reset();
        allocation_.reset();
        identity_ = std::move(other.identity_);
        accounting_ = std::move(other.accounting_);
        allocation_ = std::move(other.allocation_);
        io_ = std::move(other.io_);
        valid_tokens_ = other.valid_tokens_;
    }
    return *this;
}

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
uint64_t HistoricalKvRawStore::PreparedPayload::packed_kv_bytes() const noexcept {
    return block_ ? block_->packed_kv_bytes : 0;
}
uint64_t HistoricalKvRawStore::PreparedPayload::mean_k_bytes() const noexcept {
    return block_ ? block_->mean_k_bytes : 0;
}
uint64_t HistoricalKvRawStore::PreparedPayload::nvme_bytes() const noexcept {
    return block_ && block_->disk ? block_->disk->bytes : 0;
}
uint64_t HistoricalKvRawStore::PreparedPayload::pinned_bytes() const noexcept {
    return block_ && block_->pinned ? block_->pinned->bytes : 0;
}
std::shared_ptr<void> HistoricalKvRawStore::PreparedPayload::writer_resource() const noexcept {
    if (!block_) return nullptr;
    if (block_->private_pinned_producer) return block_->private_pinned_producer;
    return block_->private_disk_producer;
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
    if (!layer.packed_complete && layer.k_written_tokens == block.valid_tokens &&
        layer.v_written_tokens == block.valid_tokens) {
        layer.packed_complete = true;
        ++block.packed_complete_layers;
    }
    if (!layer.mean_complete && (!config_.mean_k_elements || layer.mean_written_tokens == block.valid_tokens)) {
        layer.mean_complete = true;
        ++block.mean_complete_layers;
    }
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
    if (config_.nvme_bytes && config_.pinned_bytes)
        throw std::invalid_argument("simultaneous historical NVMe and pinned copies are not supported");

    // Bound every successful reservation by a finite, representable maximum.
    const size_t k_max = multiply(config_.max_tokens_per_slot,
                                  config_.packed_k_bytes_per_token);
    const size_t v_max = multiply(config_.max_tokens_per_slot,
                                  config_.packed_v_bytes_per_token);
    const size_t mean_max = multiply(config_.mean_k_elements, sizeof(float));
    const size_t layer_max = add(add(k_max, v_max), mean_max);
    const size_t block_max = multiply(layer_max, config_.n_layer);
    (void) multiply(block_max, capacity);

    if (config_.nvme_bytes) {
        if (config_.nvme_dir.empty()) throw std::invalid_argument("historical NVMe requires a private directory");
        disk_k_stride_ = k_max;
        disk_layer_stride_ = add(k_max, v_max);
        NvmeKvTierConfig disk;
        disk.dir = config_.nvme_dir;
        disk.file_name = "historical-kv.bin";
        disk.total_bytes = config_.nvme_bytes;
        disk.slot_bytes = multiply(disk_layer_stride_, config_.n_layer);
        disk.preallocate = true;
        disk.drop_page_cache = true;
        disk.handle_addressed = true;
        nvme_ = std::make_shared<NvmeKvTier>(disk);
        if (!nvme_->enabled()) throw std::invalid_argument("historical NVMe budget cannot hold one full extent");
    }
    if (config_.pinned_bytes) {
        if (!config_.allocate_pinned)
            throw std::invalid_argument("historical pinned budget requires a page-locked allocator");
        pinned_k_stride_ = k_max;
        pinned_layer_stride_ = add(k_max, v_max);
        PinnedKvTierConfig pin;
        pin.total_bytes = config_.pinned_bytes;
        pin.slot_bytes = multiply(pinned_layer_stride_, config_.n_layer);
        pin.handle_addressed = true;
        pinned_ = std::make_shared<PinnedKvTier>(pin);
        if (!pinned_->enabled()) throw std::invalid_argument("historical pinned budget cannot hold one full extent");
    }

    slots_.reserve(capacity);
    for (uint32_t i = 0; i < capacity; ++i)
        slots_.push_back(std::make_unique<Slot>());
}

HistoricalKvRawStore::~HistoricalKvRawStore() = default;

bool HistoricalKvRawStore::reserve(
    const HistoricalKvRepository::Lease & writer,
    HistoricalKvHandle handle, uint64_t original_begin, uint32_t valid_tokens) {
    const auto meta = writer.metadata();
    // Disk writers must be prepared before begin_write_resources so their
    // original IO ownership can move into the admitted producer.
    if (nvme_ || pinned_ || !meta || !writer.is_writer() || !same(writer.handle(), handle) ||
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
    if (nvme_) {
        // Prepare ownership objects before changing allocator capacity. They
        // roll back exactly if any later RAM preparation fails.
        block->disk = std::make_shared<DiskAllocation>();
        block->disk->tier = nvme_;
        block->disk->bytes = nvme_->slot_bytes();
        block->private_disk_producer = std::make_shared<DiskProducer>();
        block->private_disk_producer->allocation = block->disk;
        const auto placement = nvme_->try_allocate();
        if (placement.slot < 0) throw std::bad_alloc();
        block->disk->handle = placement.allocation;
        block->disk->accounting = accounting_;
        accounting_->nvme_bytes.fetch_add(block->disk->bytes, std::memory_order_relaxed);
        block->private_disk_producer->io = nvme_->acquire_io(placement.allocation);
        if (!block->private_disk_producer->io) throw std::bad_alloc();
    }
    if (pinned_) {
        auto allocation = std::make_shared<PinnedAllocation>();
        auto producer = std::make_shared<PinnedProducer>();
        allocation->tier = pinned_;
        allocation->bytes = pinned_->config().slot_bytes;
        const auto placement = pinned_->try_allocate();
        if (placement.slot < 0) throw std::bad_alloc();
        allocation->handle = placement.allocation;
        allocation->data = config_.allocate_pinned(static_cast<size_t>(allocation->bytes));
        if (!allocation->data) throw std::bad_alloc();
        if (allocation->data.use_count() != 1)
            throw std::invalid_argument("historical pinned allocator must return sole ownership");
        allocation->accounting = accounting_;
        accounting_->pinned_bytes.fetch_add(allocation->bytes, std::memory_order_relaxed);
        producer->allocation = allocation;
        producer->io = pinned_->acquire_io(placement.allocation);
        if (!producer->io) throw std::bad_alloc();
        block->private_pinned_producer = std::move(producer);
        block->pinned = std::move(allocation);
    }
    const size_t k_bytes = multiply(token_capacity, config_.packed_k_bytes_per_token);
    const size_t v_bytes = multiply(token_capacity, config_.packed_v_bytes_per_token);
    for (uint32_t il = 0; il < config_.n_layer; ++il) {
        if (!required_layer(il)) continue;
        auto & layer = block->layers[il];
        layer.packed_k.resize(k_bytes);
        layer.packed_v.resize(v_bytes);
        layer.mean_k_sum.resize(config_.mean_k_elements);
        block->packed_kv_bytes = add(block->packed_kv_bytes, add(layer.packed_k.capacity(), layer.packed_v.capacity()));
        block->mean_k_bytes = add(block->mean_k_bytes, multiply(layer.mean_k_sum.capacity(), sizeof(float)));
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
    if (prepared.block_->disk && writer.writer_resource_identity() != prepared.block_->private_disk_producer.get())
        return false;
    if (prepared.block_->pinned && writer.writer_resource_identity() != prepared.block_->private_pinned_producer.get())
        return false;
    const auto meta = writer.metadata();
    if (!meta || meta->original_begin != original_begin ||
        meta->valid_tokens != valid_tokens || meta->nvme_bytes != prepared.nvme_bytes() ||
        meta->pinned_bytes != prepared.pinned_bytes() ||
        (meta->life != HistoricalKvLife::Writing && meta->life != HistoricalKvLife::Retired)) return false;

    Slot & slot = *slots_[handle.slot];
    std::lock_guard<std::mutex> lock(slot.mu);
    if (slot.block || slot.releasing) return false;
    if (slot.last_released_repository_id == writer.repository_id() &&
        slot.last_released.incarnation == handle.incarnation &&
        handle.generation <= slot.last_released.generation)
        return false;

    auto & block = prepared.block_;
    block->disk_producer = block->private_disk_producer.get();
    block->private_disk_producer.reset(); // the admitted writer now owns it
    block->pinned_producer = block->private_pinned_producer.get();
    block->private_pinned_producer.reset();
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

void HistoricalKvRawStore::write_disk(const HistoricalKvRepository::Lease & writer, Block & block,
        uint32_t layer, bool value, uint32_t token_offset, const uint8_t * bytes, size_t count) {
    if (!block.disk) return;
    if (writer.writer_resource_identity() != block.disk_producer || !block.disk_producer->io)
        throw std::logic_error("historical NVMe producer has no original allocation right");
    const uint64_t offset = uint64_t(layer) * disk_layer_stride_ + (value ? disk_k_stride_ : 0) +
        uint64_t(token_offset) * (value ? config_.packed_v_bytes_per_token : config_.packed_k_bytes_per_token);
    block.disk->tier->write_allocation(block.disk_producer->io, offset, bytes, count);
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
    if (!writer.has_packed_kv_access() || (config_.mean_k_elements && !writer.has_mean_k_access())) return false;
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
    write_disk(writer, *block, layer, false, 0, packed_k.data(), packed_k.size());
    write_disk(writer, *block, layer, true, 0, packed_v.data(), packed_v.size());
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
    if (!writer.has_packed_kv_access() || (config_.mean_k_elements && !writer.has_mean_k_access())) return false;
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
    write_disk(writer, *block, layer, false, token_offset, packed_k.data(), packed_k.size());
    write_disk(writer, *block, layer, true, token_offset, packed_v.data(), packed_v.size());
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
    if (!writer.has_packed_kv_access()) return false;
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
    write_disk(writer, *block, layer, false, token_offset, packed_k, bytes);
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
    if (!writer.has_packed_kv_access()) return false;
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
    write_disk(writer, *block, layer, true, token_offset, packed_v, bytes);
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
    if (!writer.has_mean_k_access()) return false;
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
    Block * block = slot.block.get();
    if (!block || block->repository_id != repository.repository_id() || !same(block->handle, handle)) return false;
    const auto required = config_.attention_layers.empty() ? config_.n_layer : config_.attention_layers.size();
    if (block->pinned && block->packed_complete_layers == required && !block->pinned_filled) {
        // The slot lock serializes this copy with the GC close callback. Copy
        // before closing the RAM producer claim: a retired generation may
        // release RAM independently while Mean-K is still in flight.
        // The producer reserved this right before admission. Retirement may
        // close new IO while this already-promised copy is still pending.
        if (writer.writer_resource_identity() != block->pinned_producer || !block->pinned_producer->io)
            return false;
        auto * dst = block->pinned->data.get();
        for (uint32_t il = 0; il < config_.n_layer; ++il) {
            if (!required_layer(il)) continue;
            const auto & source = block->layers[il];
            if (source.packed_k.size() != size_t(block->valid_tokens) * config_.packed_k_bytes_per_token ||
                source.packed_v.size() != size_t(block->valid_tokens) * config_.packed_v_bytes_per_token)
                return false;
            const size_t offset = size_t(il) * pinned_layer_stride_;
            std::memcpy(dst + offset, source.packed_k.data(), source.packed_k.size());
            std::memcpy(dst + offset + pinned_k_stride_, source.packed_v.data(), source.packed_v.size());
        }
        block->pinned_filled = true;
        block->pinned_producer->io.reset();
    }
    if (block->disk && block->packed_complete_layers == required) {
        if (writer.writer_resource_identity() != block->disk_producer) return false;
        block->disk_producer->io.reset();
    }
    if (writer.has_resource_writer()) {
        if (block->packed_complete_layers == required)
            repository.release_writer_resource(writer, HistoricalKvRepository::HostResource::PackedKv);
        if (!config_.mean_k_elements || block->mean_complete_layers == required)
            repository.release_writer_resource(writer, HistoricalKvRepository::HostResource::MeanK);
    }
    if (block->written_layers != required) return false;
    return repository.publish_write(writer);
}

std::optional<HistoricalKvRawStore::LayerSnapshot>
HistoricalKvRawStore::read_layer(
    const HistoricalKvRepository::Lease & reader,
    HistoricalKvHandle handle, uint32_t layer) const {
    if (!reader || !reader.has_host_access() || !reader.repository_id() ||
        !same(reader.handle(), handle) ||
        !handle.incarnation || !handle.generation || handle.slot >= slots_.size() ||
        !required_layer(layer)) return std::nullopt;
    const Slot & slot = *slots_[handle.slot];
    std::lock_guard<std::mutex> lock(slot.mu);
    // A producer's resource rights can close between the initial permission
    // check and this content lock. Ordinary reader leases remain immutable.
    if (!reader.has_host_access()) return std::nullopt;
    const Block * block = slot.block.get();
    if (!block || block->repository_id != reader.repository_id() ||
        !same(block->handle, handle) || block->packed_kv_released || block->mean_k_released) return std::nullopt;
    const Layer & source = block->layers[layer];
    if (!source.written) return std::nullopt;
    return LayerSnapshot{source.packed_k, source.packed_v,
                         source.mean_k_sum, source.mean_tokens};
}

std::optional<HistoricalKvRawStore::LayerSnapshot>
HistoricalKvRawStore::read_packed_kv(const HistoricalKvRepository::Lease & reader,
                                   HistoricalKvHandle handle, uint32_t layer) const {
    if (!reader.has_packed_kv_access() || !reader.repository_id() || !same(reader.handle(), handle) ||
        handle.slot >= slots_.size() || !required_layer(layer)) return std::nullopt;
    const auto & slot = *slots_[handle.slot];
    std::lock_guard<std::mutex> lock(slot.mu);
    if (!reader.has_packed_kv_access()) return std::nullopt;
    const auto * block = slot.block.get();
    if (!block || block->repository_id != reader.repository_id() ||
        !same(block->handle, handle) || block->packed_kv_released) return std::nullopt;
    const auto & source = block->layers[layer];
    if (!source.written) return std::nullopt;
    return LayerSnapshot{source.packed_k, source.packed_v, {}, 0};
}

bool HistoricalKvRawStore::copy_packed_kv(const HistoricalKvRepository::Lease & reader,
        HistoricalKvHandle handle, uint32_t layer, uint8_t * k, size_t k_bytes,
        uint8_t * v, size_t v_bytes) const {
    if (!reader.has_packed_kv_access() || !reader.repository_id() || !same(reader.handle(), handle) ||
        handle.slot >= slots_.size() || !required_layer(layer) || !k || !v) return false;
    const auto & slot = *slots_[handle.slot];
    std::lock_guard<std::mutex> lock(slot.mu);
    if (!reader.has_packed_kv_access()) return false;
    const auto * block = slot.block.get();
    if (!block || block->repository_id != reader.repository_id() || !same(block->handle, handle) ||
        block->packed_kv_released) return false;
    const auto & source = block->layers[layer];
    if (!source.written || source.packed_k.size() != k_bytes || source.packed_v.size() != v_bytes) return false;
    std::copy(source.packed_k.begin(), source.packed_k.end(), k);
    std::copy(source.packed_v.begin(), source.packed_v.end(), v);
    return true;
}

std::optional<HistoricalKvRawStore::NvmeRead> HistoricalKvRawStore::acquire_nvme_read(
        HistoricalKvRepository & repository, HistoricalKvHandle handle) const {
    if (!nvme_ || handle.slot >= slots_.size()) return std::nullopt;
    auto identity = repository.acquire_external_read(handle);
    if (!identity) return std::nullopt;
    const auto & slot = *slots_[handle.slot];
    std::lock_guard<std::mutex> lock(slot.mu);
    const auto * block = slot.block.get();
    const auto required = config_.attention_layers.empty() ? config_.n_layer : config_.attention_layers.size();
    if (!block || block->repository_id != repository.repository_id() || !same(block->handle, handle) ||
        !block->disk || block->nvme_released || block->written_layers != required) return std::nullopt;
    auto io = block->disk->tier->acquire_io(block->disk->handle);
    if (!io) return std::nullopt;
    // Retirement may have won after the identity pin but before the disk pin.
    // Refuse that new operation; an already returned read keeps its old rights.
    const auto meta = identity->metadata();
    if (!meta || meta->life != HistoricalKvLife::Live) return std::nullopt;
    NvmeRead result;
    result.identity_ = std::move(*identity);
    result.accounting_ = accounting_;
    result.allocation_ = block->disk;
    result.io_ = std::move(io);
    result.valid_tokens_ = block->valid_tokens;
    return result;
}

bool HistoricalKvRawStore::read_nvme_layer(const NvmeRead & reader, uint32_t layer,
        uint8_t * packed_k, size_t k_bytes, uint8_t * packed_v, size_t v_bytes) const {
    if (!reader || reader.accounting_ != accounting_ || !required_layer(layer) ||
        !packed_k || !packed_v || k_bytes != size_t(reader.valid_tokens_) * config_.packed_k_bytes_per_token ||
        v_bytes != size_t(reader.valid_tokens_) * config_.packed_v_bytes_per_token) return false;
    const auto & allocation = *reader.allocation_;
    const uint64_t offset = uint64_t(layer) * disk_layer_stride_;
    allocation.tier->read_allocation(reader.io_, offset, packed_k, k_bytes);
    allocation.tier->read_allocation(reader.io_, offset + disk_k_stride_, packed_v, v_bytes);
    return true;
}

std::optional<HistoricalKvRawStore::PinnedRead> HistoricalKvRawStore::acquire_pinned_read(
        HistoricalKvRepository & repository, HistoricalKvHandle handle) const {
    if (!pinned_ || handle.slot >= slots_.size()) return std::nullopt;
    auto identity = repository.acquire_external_read(handle);
    if (!identity) return std::nullopt;
    const auto & slot = *slots_[handle.slot];
    std::lock_guard<std::mutex> lock(slot.mu);
    const auto * block = slot.block.get();
    const auto required = config_.attention_layers.empty() ? config_.n_layer : config_.attention_layers.size();
    if (!block || block->repository_id != repository.repository_id() || !same(block->handle, handle) ||
        !block->pinned || block->pinned_released || !block->pinned_filled ||
        block->written_layers != required) return std::nullopt;
    auto io = block->pinned->tier->acquire_io(block->pinned->handle);
    if (!io) return std::nullopt;
    const auto meta = identity->metadata();
    if (!meta || meta->life != HistoricalKvLife::Live) return std::nullopt;
    PinnedRead result;
    result.identity_ = std::move(*identity);
    result.accounting_ = accounting_;
    result.allocation_ = block->pinned;
    result.io_ = std::move(io);
    result.valid_tokens_ = block->valid_tokens;
    return result;
}

bool HistoricalKvRawStore::read_pinned_layer(const PinnedRead & reader, uint32_t layer,
        uint8_t * packed_k, size_t k_bytes, uint8_t * packed_v, size_t v_bytes) const {
    const auto source = pinned_layer(reader, layer);
    if (!source || !packed_k || !packed_v || k_bytes != source->k_bytes || v_bytes != source->v_bytes)
        return false;
    std::memcpy(packed_k, source->packed_k, k_bytes);
    std::memcpy(packed_v, source->packed_v, v_bytes);
    return true;
}

std::optional<HistoricalKvRawStore::PinnedLayerView> HistoricalKvRawStore::pinned_layer(
        const PinnedRead & reader, uint32_t layer) const {
    if (!reader || reader.accounting_ != accounting_ || !required_layer(layer) ||
        !reader.allocation_->tier->owns(reader.io_) || !reader.allocation_->data) return std::nullopt;
    const auto * source = reader.allocation_->data.get() + size_t(layer) * pinned_layer_stride_;
    return PinnedLayerView{source, source + pinned_k_stride_,
        size_t(reader.valid_tokens_) * config_.packed_k_bytes_per_token,
        size_t(reader.valid_tokens_) * config_.packed_v_bytes_per_token};
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
        !same(block->handle, handle) || (block->packed_kv_released && block->mean_k_released)) return std::nullopt;
    return BlockInfo{block->handle, block->original_begin,
                     block->valid_tokens, block->written_layers,
                     block->payload_bytes};
}

bool HistoricalKvRawStore::copy_mean_k(
    const HistoricalKvRepository::Lease & reader,
    HistoricalKvHandle handle, uint32_t layer, float * output, size_t elements) const {
    if (!reader || !reader.has_mean_k_access() || !reader.repository_id() || !same(reader.handle(), handle) ||
        handle.slot >= slots_.size() || !required_layer(layer) || !output ||
        !config_.mean_k_elements || elements != config_.mean_k_elements) return false;
    const Slot & slot = *slots_[handle.slot];
    std::lock_guard<std::mutex> lock(slot.mu);
    if (!reader.has_mean_k_access()) return false;
    const Block * block = slot.block.get();
    if (!block || block->repository_id != reader.repository_id() ||
        !same(block->handle, handle) || block->mean_k_released) return false;
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
            return repository.is_host_releasing(handle)
                ? ReleaseResult{ReleaseCode::Absent, 0}
                : ReleaseResult{ReleaseCode::Busy, 0};
        }
        if (slot.block->repository_id != repository.repository_id() ||
            !same(slot.block->handle, handle))
            return {ReleaseCode::Stale, 0};
        if (!repository.is_host_releasing(handle))
            return {ReleaseCode::Busy, 0};
        if (slot.block->disk || slot.block->pinned) return {ReleaseCode::Busy, 0}; // independent callbacks
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

HistoricalKvRawStore::ReleaseResult HistoricalKvRawStore::release_resource(
    const HistoricalKvRepository & repository, HistoricalKvHandle handle,
    HistoricalKvRepository::HostResource resource) noexcept {
    using Resource = HistoricalKvRepository::HostResource;
    if (!handle.incarnation || !handle.generation || handle.slot >= slots_.size())
        return {ReleaseCode::Stale, 0};
    auto & slot = *slots_[handle.slot];
    if (resource == Resource::Pinned) {
        std::shared_ptr<PinnedAllocation> allocation;
        {
            std::unique_lock<std::mutex> lock(slot.mu, std::try_to_lock);
            if (!lock.owns_lock() || slot.releasing) return {ReleaseCode::Busy, 0};
            if (!slot.block) {
                if (slot.last_released_repository_id == repository.repository_id() && same(slot.last_released, handle))
                    return {ReleaseCode::AlreadyReleased, 0};
                return repository.is_releasing(handle) && repository.is_resource_releasing(handle, resource) ?
                    ReleaseResult{ReleaseCode::Absent, 0} : ReleaseResult{ReleaseCode::Busy, 0};
            }
            const auto & block = *slot.block;
            if (block.repository_id != repository.repository_id() || !same(block.handle, handle))
                return {ReleaseCode::Stale, 0};
            if (!repository.is_resource_releasing(handle, resource)) return {ReleaseCode::Busy, 0};
            if (!block.pinned) return block.pinned_released ? ReleaseResult{ReleaseCode::AlreadyReleased, 0} :
                ReleaseResult{ReleaseCode::Absent, 0};
            allocation = block.pinned;
            if (!allocation->slot_returned.load(std::memory_order_acquire)) {
                const auto close = allocation->tier->try_close_allocation(allocation->handle);
                if (close != PinnedReleaseResult::Released) return {
                    close == PinnedReleaseResult::Stale ? ReleaseCode::Stale : ReleaseCode::Busy, 0};
            }
        }
        // No admitted IO survives close. Free page-locked memory outside the
        // content lock while the tier slot remains unavailable to admission.
        const bool physically_owned = !allocation->memory_returned.load(std::memory_order_acquire);
        allocation->return_memory();
        if (!allocation->slot_returned.load(std::memory_order_acquire)) {
            const auto finish = allocation->tier->try_finish_closed_allocation(allocation->handle);
            if (finish != PinnedReleaseResult::Released) return {ReleaseCode::Busy, 0};
            allocation->slot_returned.store(true, std::memory_order_release);
        }
        std::shared_ptr<PinnedAllocation> retired;
        {
            std::unique_lock<std::mutex> lock(slot.mu, std::try_to_lock);
            if (!lock.owns_lock()) return {ReleaseCode::Busy, 0};
            if (!slot.block || slot.block->repository_id != repository.repository_id() ||
                !same(slot.block->handle, handle)) return {ReleaseCode::Stale, 0};
            retired = std::move(slot.block->pinned);
            slot.block->pinned_released = true;
        }
        allocation.reset();
        retired.reset();
        return {ReleaseCode::Released, physically_owned ? pinned_->config().slot_bytes : 0};
    }
    if (resource == Resource::Nvme) {
        std::shared_ptr<DiskAllocation> allocation;
        {
            std::unique_lock<std::mutex> lock(slot.mu, std::try_to_lock);
            if (!lock.owns_lock() || slot.releasing) return {ReleaseCode::Busy, 0};
            if (!slot.block) {
                if (slot.last_released_repository_id == repository.repository_id() && same(slot.last_released, handle))
                    return {ReleaseCode::AlreadyReleased, 0};
                if (slot.last_released_repository_id == repository.repository_id() &&
                    slot.last_released.incarnation == handle.incarnation && handle.generation <= slot.last_released.generation)
                    return {ReleaseCode::Stale, 0};
                // A retired writer may still install its prepared payload.
                // Only the final no-reference phase can acknowledge absence.
                return repository.is_releasing(handle) && repository.is_resource_releasing(handle, resource) ?
                    ReleaseResult{ReleaseCode::Absent, 0} : ReleaseResult{ReleaseCode::Busy, 0};
            }
            const auto & block = *slot.block;
            if (block.repository_id != repository.repository_id() || !same(block.handle, handle))
                return {ReleaseCode::Stale, 0};
            if (!repository.is_resource_releasing(handle, resource)) return {ReleaseCode::Busy, 0};
            if (!block.disk) return {ReleaseCode::Absent, 0};
            if (block.nvme_released) return {ReleaseCode::AlreadyReleased, 0};
            allocation = block.disk;
        }
        const auto result = allocation->returned.load(std::memory_order_acquire) ? NvmeReleaseResult::AlreadyReleased :
            allocation->tier->try_release_allocation(allocation->handle);
        if (result != NvmeReleaseResult::Released && result != NvmeReleaseResult::AlreadyReleased)
            return {result == NvmeReleaseResult::Stale ? ReleaseCode::Stale : ReleaseCode::Busy, 0};
        allocation->return_credit();
        // The repository's release owner retains this exact metadata record.
        // If its content lock is busy, retry acknowledgement on the next scan.
        std::unique_lock<std::mutex> lock(slot.mu, std::try_to_lock);
        if (!lock.owns_lock()) return {ReleaseCode::Busy, 0};
        slot.block->nvme_released = true;
        return {ReleaseCode::Released, result == NvmeReleaseResult::Released ? allocation->bytes : 0};
    }
    std::unique_ptr<Block> metadata;
    Block * block = nullptr;
    {
        std::lock_guard<std::mutex> lock(slot.mu);
        if (slot.releasing) return {ReleaseCode::Busy, 0};
        if (!slot.block) {
            if (slot.last_released_repository_id == repository.repository_id() && same(slot.last_released, handle))
                return {ReleaseCode::AlreadyReleased, 0};
            if (slot.last_released_repository_id == repository.repository_id() &&
                slot.last_released.incarnation == handle.incarnation && handle.generation <= slot.last_released.generation)
                return {ReleaseCode::Stale, 0};
            return repository.is_resource_releasing(handle, resource) ?
                ReleaseResult{ReleaseCode::Absent, 0} : ReleaseResult{ReleaseCode::Busy, 0};
        }
        block = slot.block.get();
        if (block->repository_id != repository.repository_id() || !same(block->handle, handle))
            return {ReleaseCode::Stale, 0};
        if (!repository.is_resource_releasing(handle, resource)) return {ReleaseCode::Busy, 0};
        if (resource == Resource::Metadata) {
            if (!block->packed_kv_released || !block->mean_k_released || block->payload_bytes ||
                (block->disk && !block->nvme_released) ||
                (block->pinned && !block->pinned_released))
                return {ReleaseCode::Busy, 0};
            metadata = std::move(slot.block);
        } else if (resource == Resource::PackedKv ? block->packed_kv_released : block->mean_k_released) {
            return {ReleaseCode::AlreadyReleased, 0};
        }
        slot.releasing = true;
    }
    if (metadata) {
        // The allocation/header credit stays charged until every reference
        // drains. Payload bytes alone never authorize a new physical slot.
        metadata.reset();
        std::lock_guard<std::mutex> lock(slot.mu);
        slot.last_released = handle;
        slot.last_released_repository_id = repository.repository_id();
        slot.releasing = false;
        return {ReleaseCode::Released, 0};
    }
    uint64_t total = 0;
    for (size_t layer = 0; layer < config_.n_layer; ++layer) {
        uint64_t bytes = 0;
        {
            std::vector<uint8_t> k, v;
            std::vector<float> mean;
            {
                std::lock_guard<std::mutex> lock(slot.mu);
                auto & source = block->layers[layer];
                if (resource == Resource::PackedKv) {
                    k.swap(source.packed_k); v.swap(source.packed_v);
                } else {
                    mean.swap(source.mean_k_sum);
                }
            }
            bytes = k.capacity() + v.capacity() + mean.capacity() * sizeof(float);
        } // Physical vector destruction outside the slot lock, no allocations.
        block->accounting->bytes.fetch_sub(bytes, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lock(slot.mu);
            block->payload_bytes -= bytes;
            (resource == Resource::PackedKv ? block->packed_kv_bytes : block->mean_k_bytes) -= bytes;
        }
        total += bytes;
    }
    {
        std::lock_guard<std::mutex> lock(slot.mu);
        (resource == Resource::PackedKv ? block->packed_kv_released : block->mean_k_released) = true;
        slot.releasing = false;
    }
    return {ReleaseCode::Released, total};
}

uint64_t HistoricalKvRawStore::allocated_payload_bytes() const noexcept {
    return accounting_->bytes.load(std::memory_order_relaxed);
}

uint64_t HistoricalKvRawStore::allocated_nvme_bytes() const noexcept {
    return accounting_->nvme_bytes.load(std::memory_order_relaxed);
}

uint64_t HistoricalKvRawStore::allocated_pinned_bytes() const noexcept {
    return accounting_->pinned_bytes.load(std::memory_order_relaxed);
}

uint32_t HistoricalKvRawStore::capacity() const noexcept {
    return static_cast<uint32_t>(slots_.size());
}

const HistoricalKvRawStoreConfig & HistoricalKvRawStore::config() const noexcept {
    return config_;
}

} // namespace kvmem
