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
};

class HistoricalKvRawStore {
public:
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

    // Whole-layer, write-once publication. The packed rows are opaque bytes.
    // The caller must publish the repository writer only after all layers are
    // written. Reader snapshots copy under a per-slot lock and require a
    // repository lease for the whole operation.
    bool write_layer(const HistoricalKvRepository::Lease & writer,
                     HistoricalKvHandle handle, uint32_t layer,
                     const std::vector<uint8_t> & packed_k,
                     const std::vector<uint8_t> & packed_v,
                     const std::vector<float> & mean_k_sum,
                     uint32_t mean_tokens);
    std::optional<LayerSnapshot> read_layer(
        const HistoricalKvRepository::Lease & reader,
        HistoricalKvHandle handle, uint32_t layer) const;
    std::optional<BlockInfo> inspect(HistoricalKvHandle handle) const;

    // Exact-generation, whole-bundle release. No large allocation is freed
    // while holding the slot lock. On Released, the returned bytes are the
    // payload vector capacity actually destroyed by this invocation. A stale
    // handle cannot free a later generation. Absent is a safe no-op (including
    // a failed reserve); Busy means another release is still finishing.
    ReleaseResult release(HistoricalKvHandle handle) noexcept;
    uint64_t allocated_payload_bytes() const noexcept;
    uint32_t capacity() const noexcept;
    const HistoricalKvRawStoreConfig & config() const noexcept;

private:
    struct Layer {
        std::vector<uint8_t> packed_k;
        std::vector<uint8_t> packed_v;
        std::vector<float> mean_k_sum;
        uint32_t mean_tokens = 0;
        bool written = false;
    };
    struct Block {
        HistoricalKvHandle handle;
        uint64_t original_begin = 0;
        uint32_t valid_tokens = 0;
        uint32_t written_layers = 0;
        uint64_t payload_bytes = 0;
        std::vector<Layer> layers;
    };
    struct Slot {
        mutable std::mutex mu;
        std::unique_ptr<Block> block;
        HistoricalKvHandle last_released{};
        bool releasing = false;
    };

    static bool same(HistoricalKvHandle a, HistoricalKvHandle b) noexcept;
    HistoricalKvRawStoreConfig config_;
    std::vector<std::unique_ptr<Slot>> slots_;
    std::atomic<uint64_t> allocated_payload_bytes_{0};
};

} // namespace kvmem
