#pragma once

#include "kvmem/historical_kv_repository.hpp"
#include "kvmem/historical_kv_payload_budget.hpp"
#include "kvmem/nvme_kv_allocation.hpp"
#include "kvmem/pinned_kv_tier.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace kvmem {
class NvmeKvTier;

// Payload store for the historical KV repository, with optional disk copies. Its slot is a
// physical allocation address, never original_begin / a block size. The
// repository owns admission and lifetime: callers must hold its writer/read/
// transfer lease until their use of a payload is complete. In particular,
// release()/release_resource() belong in repository background callbacks only.
// That callback must keep this store alive until the repository has drained.
struct HistoricalKvRawStoreConfig {
    uint32_t n_layer = 0;
    uint32_t max_tokens_per_slot = 0;
    size_t packed_k_bytes_per_token = 0;
    size_t packed_v_bytes_per_token = 0;
    size_t mean_k_elements = 0; // F32 sum length per layer; 0 disables it.
    // Physical attention layer IDs. Empty preserves the all-layers default.
    // Non-attention layers have no payload and do not block publication.
    std::vector<uint32_t> attention_layers;
    // Optional independent disk copy. Full extents are reserved alongside RAM
    // before request admission; this does not yet eliminate the RAM copy.
    // Requires the Linux NVMe tier with working physical preallocation.
    uint64_t nvme_bytes = 0;
    std::string nvme_dir;
    // Optional independent pinned copy. The allocator must return actual
    // page-locked memory of at least the requested size with sole ownership
    // in the returned shared_ptr. No pageable fallback is permitted.
    // Full physical extents are allocated before request admission.
    uint64_t pinned_bytes = 0;
    std::function<std::shared_ptr<uint8_t>(size_t)> allocate_pinned;
    // Optional common quota across successive incarnations. Shared ownership
    // is copied by reset; a retained old payload still spends this same quota.
    std::shared_ptr<HistoricalKvPayloadBudget> payload_budget;
};

class HistoricalKvRawStore {
    struct Block;
    struct Accounting;
    struct DiskAllocation;
    struct DiskProducer;
    struct PinnedAllocation;
    struct PinnedProducer;
public:
    // Actual K/V/Mean-K allocations prepared before request admission. A
    // prepared payload is private, not historical data and not selectable.
    // Moving it into a reserved writer needs no payload allocation. Unused
    // objects may be returned at request cancellation/end without running GC.
    class PreparedPayload {
    public:
        PreparedPayload();
        ~PreparedPayload();
        PreparedPayload(PreparedPayload &&) noexcept;
        PreparedPayload & operator=(PreparedPayload &&) noexcept;
        PreparedPayload(const PreparedPayload &) = delete;
        PreparedPayload & operator=(const PreparedPayload &) = delete;
        explicit operator bool() const noexcept;
        uint64_t payload_bytes() const noexcept;
        uint64_t packed_kv_bytes() const noexcept;
        uint64_t mean_k_bytes() const noexcept;
        uint64_t nvme_bytes() const noexcept;
        uint64_t pinned_bytes() const noexcept;
        std::shared_ptr<void> writer_resource() const noexcept;
        uint32_t token_capacity() const noexcept;
    private:
        friend class HistoricalKvRawStore;
        std::unique_ptr<Block> block_;
    };
    // Move-only permission for one exact disk allocation and historical
    // identity. It may finish after retirement, and pins no Host payload.
    class NvmeRead {
    public:
        NvmeRead() = default;
        NvmeRead(NvmeRead &&) noexcept = default;
        NvmeRead & operator=(NvmeRead &&) noexcept;
        NvmeRead(const NvmeRead &) = delete;
        NvmeRead & operator=(const NvmeRead &) = delete;
        explicit operator bool() const noexcept { return bool(io_); }
        NvmeAllocationHandle allocation_handle() const noexcept { return io_.handle(); }
    private:
        friend class HistoricalKvRawStore;
        HistoricalKvRepository::Lease identity_;
        std::shared_ptr<Accounting> accounting_;
        std::shared_ptr<DiskAllocation> allocation_;
        NvmeIoLease io_;
        uint32_t valid_tokens_ = 0;
    };
    class PinnedRead {
    public:
        PinnedRead() = default;
        PinnedRead(PinnedRead &&) noexcept = default;
        PinnedRead & operator=(PinnedRead &&) noexcept;
        PinnedRead(const PinnedRead &) = delete;
        PinnedRead & operator=(const PinnedRead &) = delete;
        explicit operator bool() const noexcept { return bool(io_); }
        PinnedAllocationHandle allocation_handle() const noexcept { return io_.handle(); }
    private:
        friend class HistoricalKvRawStore;
        HistoricalKvRepository::Lease identity_;
        std::shared_ptr<Accounting> accounting_;
        std::shared_ptr<PinnedAllocation> allocation_;
        PinnedIoLease io_;
        uint32_t valid_tokens_ = 0;
    };
    struct BlockInfo {
        HistoricalKvHandle handle;
        uint64_t original_begin = 0;
        uint32_t valid_tokens = 0;
        uint32_t written_layers = 0;
        // Capacity owned by K/V/mean payload vectors, excluding persistent
        // slot bookkeeping and allocator overhead. This is what release frees.
        uint64_t payload_bytes = 0;
    };

    struct LayerSnapshot {
        std::vector<uint8_t> packed_k;
        std::vector<uint8_t> packed_v;
        std::vector<float> mean_k_sum;
        uint32_t mean_tokens = 0;
    };

    struct PinnedLayerView {
        const uint8_t * packed_k = nullptr;
        const uint8_t * packed_v = nullptr;
        size_t k_bytes = 0;
        size_t v_bytes = 0;
    };

    struct WriteProgress {
        uint32_t packed_k_tokens = 0;
        uint32_t packed_v_tokens = 0;
        uint32_t mean_k_tokens = 0;
    };

    enum class ReleaseCode { Released, AlreadyReleased, Absent, Busy, Stale };
    struct ReleaseResult {
        ReleaseCode code = ReleaseCode::Stale;
        uint64_t physical_payload_bytes = 0;
    };

    HistoricalKvRawStore(HistoricalKvRawStoreConfig config, uint32_t capacity);
    ~HistoricalKvRawStore();
    HistoricalKvRawStore(const HistoricalKvRawStore &) = delete;
    HistoricalKvRawStore & operator=(const HistoricalKvRawStore &) = delete;

    // Reserve the exact repository generation and its original token span.
    // Returns false for invalid/occupied slots; allocation failure propagates.
    // A failed allocation has no installed payload, so release() reports Absent.
    bool reserve(const HistoricalKvRepository::Lease & writer,
                 HistoricalKvHandle handle, uint64_t original_begin,
                 uint32_t valid_tokens);
    PreparedPayload prepare_payload(uint32_t token_capacity);
    // Same exact writer/span checks as reserve(). Failure preserves prepared.
    // valid_tokens may be shorter than the reserved capacity; the full owned
    // capacity remains charged until this generation is physically released.
    bool install_payload(const HistoricalKvRepository::Lease & writer,
                         HistoricalKvHandle handle, uint64_t original_begin,
                         uint32_t valid_tokens, PreparedPayload & prepared);

    // Whole-layer, write-once publication. The packed rows are opaque bytes;
    // configured mean-K must cover the complete reserved span.
    // The caller must publish the repository writer only after all layers are
    // written. Reader snapshots copy under a per-slot lock and require a
    // repository lease for the whole operation.
    bool write_layer(const HistoricalKvRepository::Lease & writer,
                     HistoricalKvHandle handle, uint32_t layer,
                     const std::vector<uint8_t> & packed_k,
                     const std::vector<uint8_t> & packed_v,
                     const std::vector<float> & mean_k_sum,
                     uint32_t mean_tokens);
    // Incremental, contiguous write for one layer of an explicitly reserved
    // span. offset must equal the layer's current written prefix; holes and
    // duplicate writes are rejected. The old writer may finish after retire,
    // but cannot publish that retired generation back to Live. When mean-K is
    // enabled, each slice supplies the sum for all its tokens.
    bool write_slice(const HistoricalKvRepository::Lease & writer,
                     HistoricalKvHandle handle, uint32_t layer,
                     uint32_t token_offset, uint32_t token_count,
                     const std::vector<uint8_t> & packed_k,
                     const std::vector<uint8_t> & packed_v,
                     const std::vector<float> & mean_k_sum,
                     uint32_t mean_tokens);
    // Harvest may deliver K, V and the mean-K sum independently and in a
    // different order. Each component has its own contiguous token prefix.
    // A layer is complete only after every configured component covers the
    // reserved span; no partial layer can be published or read as complete.
    bool write_packed_k_slice(const HistoricalKvRepository::Lease & writer,
                              HistoricalKvHandle handle, uint32_t layer,
                              uint32_t token_offset, uint32_t token_count,
                              const std::vector<uint8_t> & packed_k);
    bool write_packed_v_slice(const HistoricalKvRepository::Lease & writer,
                              HistoricalKvHandle handle, uint32_t layer,
                              uint32_t token_offset, uint32_t token_count,
                              const std::vector<uint8_t> & packed_v);
    // Synchronous copy into the already reserved payload. The source need
    // only remain valid through this call; no temporary payload is allocated.
    bool write_packed_k_slice(const HistoricalKvRepository::Lease & writer,
                              HistoricalKvHandle handle, uint32_t layer,
                              uint32_t token_offset, uint32_t token_count,
                              const uint8_t * packed_k, size_t bytes);
    bool write_packed_v_slice(const HistoricalKvRepository::Lease & writer,
                              HistoricalKvHandle handle, uint32_t layer,
                              uint32_t token_offset, uint32_t token_count,
                              const uint8_t * packed_v, size_t bytes);
    bool write_mean_k_slice(const HistoricalKvRepository::Lease & writer,
                            HistoricalKvHandle handle, uint32_t layer,
                            uint32_t token_offset, uint32_t token_count,
                            const std::vector<float> & mean_k_sum);
    // Captured producers use these independent prefixes to submit only the
    // missing suffix. Observing progress never grants a new writer lease.
    std::optional<WriteProgress> write_progress(
        const HistoricalKvRepository::Lease & writer,
        HistoricalKvHandle handle, uint32_t layer) const;
    // Publish only when every layer has its full reserved span. The slot lock
    // serializes this transition with write_slice/write_layer in this store.
    // A strategy may still retire the writer; then publication fails.
    // For resource writers, also closes each fully completed component before
    // the other component finishes. Progress metadata remains under the writer
    // identity so admitted retries can observe completion without payload access.
    bool publish_complete(HistoricalKvRepository & repository,
                          const HistoricalKvRepository::Lease & writer,
                          HistoricalKvHandle handle);
    std::optional<LayerSnapshot> read_layer(
        const HistoricalKvRepository::Lease & reader,
        HistoricalKvHandle handle, uint32_t layer) const;
    // Copies K/V only; requires no Mean-K dependency and never returns it.
    std::optional<LayerSnapshot> read_packed_kv(
        const HistoricalKvRepository::Lease & reader,
        HistoricalKvHandle handle, uint32_t layer) const;
    // Copies into caller-owned, already prepared stage-in buffers.
    bool copy_packed_kv(const HistoricalKvRepository::Lease & reader,
                       HistoricalKvHandle handle, uint32_t layer,
                       uint8_t * k, size_t k_bytes, uint8_t * v, size_t v_bytes) const;
    // Scoring copies only the normalized mean, never the large packed K/V
    // vectors. The admitted reader must cover the entire call.
    bool copy_mean_k(const HistoricalKvRepository::Lease & reader,
                     HistoricalKvHandle handle, uint32_t layer,
                     float * output, size_t elements) const;
    std::optional<NvmeRead> acquire_nvme_read(HistoricalKvRepository & repository,
                                             HistoricalKvHandle handle) const;
    bool read_nvme_layer(const NvmeRead & reader, uint32_t layer,
                         uint8_t * packed_k, size_t k_bytes,
                         uint8_t * packed_v, size_t v_bytes) const;
    std::optional<PinnedRead> acquire_pinned_read(HistoricalKvRepository & repository,
                                                 HistoricalKvHandle handle) const;
    bool read_pinned_layer(const PinnedRead & reader, uint32_t layer,
                          uint8_t * packed_k, size_t k_bytes,
                          uint8_t * packed_v, size_t v_bytes) const;
    // Borrowed page-locked addresses. Keep PinnedRead alive until the GPU
    // transfer has *completed*, not merely until it has been submitted.
    std::optional<PinnedLayerView> pinned_layer(const PinnedRead & reader, uint32_t layer) const;
    // Repository identity is required even for metadata observations: an
    // external incarnation can be deliberately reused by another owner.
    std::optional<BlockInfo> inspect(const HistoricalKvRepository & repository,
                                     HistoricalKvHandle handle) const;

    // Exact-repository, exact-generation Host-bundle release. The repository
    // must hold Host release ownership after actual Host dependencies drain.
    // GPU-only/metadata leases may still retain the identity and prevent Ready.
    // No large allocation is freed
    // while holding the slot lock. On Released, the returned bytes are the
    // payload vector capacity actually destroyed by this invocation. A stale
    // handle cannot free a later generation. Absent is a safe no-op (including
    // a failed reserve); Busy means another release is still finishing.
    ReleaseResult release(const HistoricalKvRepository & repository,
                          HistoricalKvHandle handle) noexcept;
    // Independent resource disposal with per-component idempotence. Physical
    // bytes return after destruction outside slot.mu; the allocation/header
    // credit returns only in Metadata after all references have drained.
    ReleaseResult release_resource(const HistoricalKvRepository & repository,
        HistoricalKvHandle handle, HistoricalKvRepository::HostResource resource) noexcept;
    // Includes private PreparedPayload allocations as well as installed data.
    uint64_t allocated_payload_bytes() const noexcept;
    uint64_t allocated_nvme_bytes() const noexcept;
    uint64_t allocated_pinned_bytes() const noexcept;
    static uint64_t max_payload_bytes(const HistoricalKvRawStoreConfig & config, uint32_t capacity);
    uint32_t capacity() const noexcept;
    const HistoricalKvRawStoreConfig & config() const noexcept;

private:
    template <class T> using PayloadVector = std::vector<T, HistoricalKvPayloadAllocator<T>>;
    struct Layer {
        explicit Layer(const std::shared_ptr<HistoricalKvPayloadBudget> & budget = {})
            : packed_k(HistoricalKvPayloadAllocator<uint8_t>(budget)),
              packed_v(HistoricalKvPayloadAllocator<uint8_t>(budget)),
              mean_k_sum(HistoricalKvPayloadAllocator<float>(budget)) {}
        PayloadVector<uint8_t> packed_k;
        PayloadVector<uint8_t> packed_v;
        PayloadVector<float> mean_k_sum;
        uint32_t mean_tokens = 0;
        uint32_t k_written_tokens = 0;
        uint32_t v_written_tokens = 0;
        uint32_t mean_written_tokens = 0;
        bool written = false;
        bool packed_complete = false;
        bool mean_complete = false;
    };
    struct Block {
        ~Block();
        HistoricalKvHandle handle;
        uint64_t repository_id = 0;
        uint64_t original_begin = 0;
        uint32_t valid_tokens = 0;
        uint32_t written_layers = 0;
        uint32_t packed_complete_layers = 0;
        uint32_t mean_complete_layers = 0;
        uint64_t payload_bytes = 0;
        uint64_t packed_kv_bytes = 0;
        uint64_t mean_k_bytes = 0;
        bool packed_kv_released = false;
        bool mean_k_released = false;
        std::shared_ptr<Accounting> accounting;
        bool payload_accounted = false;
        std::vector<Layer> layers;
        std::shared_ptr<DiskAllocation> disk;
        std::shared_ptr<DiskProducer> private_disk_producer;
        DiskProducer * disk_producer = nullptr; // owned by the admitted writer
        bool nvme_released = false;
        std::shared_ptr<PinnedAllocation> pinned;
        std::shared_ptr<PinnedProducer> private_pinned_producer;
        PinnedProducer * pinned_producer = nullptr; // owned by the admitted writer
        bool pinned_released = false;
        bool pinned_filled = false;
    };
    struct Slot {
        mutable std::mutex mu;
        std::unique_ptr<Block> block;
        HistoricalKvHandle last_released{};
        uint64_t last_released_repository_id = 0;
        bool releasing = false;
    };

    static bool same(HistoricalKvHandle a, HistoricalKvHandle b) noexcept;
    bool required_layer(uint32_t layer) const noexcept;
    void mark_layer_complete(Block & block, Layer & layer) const noexcept;
    void write_disk(const HistoricalKvRepository::Lease & writer, Block & block,
                    uint32_t layer, bool value, uint32_t token_offset,
                    const uint8_t * bytes, size_t count);
    HistoricalKvRawStoreConfig config_;
    std::shared_ptr<Accounting> accounting_;
    std::vector<std::unique_ptr<Slot>> slots_;
    std::shared_ptr<NvmeKvTier> nvme_;
    size_t disk_k_stride_ = 0, disk_layer_stride_ = 0;
    std::shared_ptr<PinnedKvTier> pinned_;
    size_t pinned_k_stride_ = 0, pinned_layer_stride_ = 0;
};

} // namespace kvmem
