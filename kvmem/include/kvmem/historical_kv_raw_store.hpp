#pragma once

#include "kvmem/historical_kv_repository.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

namespace kvmem {

// RAM-only payload store for the new historical KV repository. Its slot is a
// physical allocation address, never original_begin / a block size. The
// repository owns admission and lifetime: callers must hold its writer/read/
// transfer lease until their use of a payload is complete. In particular,
// release() belongs in the repository's background release callback only.
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
};

class HistoricalKvRawStore {
    struct Block;
    struct Accounting;
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
        uint32_t token_capacity() const noexcept;
    private:
        friend class HistoricalKvRawStore;
        std::unique_ptr<Block> block_;
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
    bool publish_complete(HistoricalKvRepository & repository,
                          const HistoricalKvRepository::Lease & writer,
                          HistoricalKvHandle handle);
    std::optional<LayerSnapshot> read_layer(
        const HistoricalKvRepository::Lease & reader,
        HistoricalKvHandle handle, uint32_t layer) const;
    // Scoring copies only the normalized mean, never the large packed K/V
    // vectors. The admitted reader must cover the entire call.
    bool copy_mean_k(const HistoricalKvRepository::Lease & reader,
                     HistoricalKvHandle handle, uint32_t layer,
                     float * output, size_t elements) const;
    // Repository identity is required even for metadata observations: an
    // external incarnation can be deliberately reused by another owner.
    std::optional<BlockInfo> inspect(const HistoricalKvRepository & repository,
                                     HistoricalKvHandle handle) const;

    // Exact-repository, exact-generation, whole-bundle release. The caller's
    // repository must be in Releasing, after all old leases have drained.
    // No large allocation is freed
    // while holding the slot lock. On Released, the returned bytes are the
    // payload vector capacity actually destroyed by this invocation. A stale
    // handle cannot free a later generation. Absent is a safe no-op (including
    // a failed reserve); Busy means another release is still finishing.
    ReleaseResult release(const HistoricalKvRepository & repository,
                          HistoricalKvHandle handle) noexcept;
    // Includes private PreparedPayload allocations as well as installed data.
    uint64_t allocated_payload_bytes() const noexcept;
    uint32_t capacity() const noexcept;
    const HistoricalKvRawStoreConfig & config() const noexcept;

private:
    struct Layer {
        std::vector<uint8_t> packed_k;
        std::vector<uint8_t> packed_v;
        std::vector<float> mean_k_sum;
        uint32_t mean_tokens = 0;
        uint32_t k_written_tokens = 0;
        uint32_t v_written_tokens = 0;
        uint32_t mean_written_tokens = 0;
        bool written = false;
    };
    struct Block {
        ~Block();
        HistoricalKvHandle handle;
        uint64_t repository_id = 0;
        uint64_t original_begin = 0;
        uint32_t valid_tokens = 0;
        uint32_t written_layers = 0;
        uint64_t payload_bytes = 0;
        std::shared_ptr<Accounting> accounting;
        bool payload_accounted = false;
        std::vector<Layer> layers;
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
    HistoricalKvRawStoreConfig config_;
    std::shared_ptr<Accounting> accounting_;
    std::vector<std::unique_ptr<Slot>> slots_;
};

} // namespace kvmem
