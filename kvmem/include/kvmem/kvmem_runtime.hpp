#pragma once

// Host-side KVMem runtime: block table + CPU/NVMe tiers + reselect cadence.
// finish_reselect always runs stage_out before stage_in. GPU copies go through
// KvMemBackend (no-op in P0).

#include "kvmem/kvmem_backend.hpp"
#include "kvmem/historical_kv_shadow_directory.hpp"
#include "kvmem/kvmem_store.hpp"
#if defined(_WIN32)
// The POSIX implementation cannot be built on Windows; see the header.
#include "kvmem/nvme_kv_tier_win.hpp"
#else
#include "kvmem/nvme_kv_tier.hpp"
#endif
#include "kvmem/pinned_kv_tier.hpp"

#include <cstdint>
#include <future>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace kvmem {

struct KvMemRuntimeConfig {
    KvMemStoreConfig store;
    uint64_t cpu_bytes = 0;
    uint64_t nvme_bytes = 0;
    std::string nvme_dir;
};

class KvMemRuntime {
public:
    explicit KvMemRuntime(KvMemRuntimeConfig cfg,
                          KvMemBackend *backend = nullptr);

    const KvMemRuntimeConfig &config() const { return cfg_; }
    KvMemStore &store() { return store_; }
    const KvMemStore &store() const { return store_; }
    const KvMemPlan &last_plan() const { return last_plan_; }
    // Opt-in identity bridge for selection and individual CPU reads/transfers.
    // Callers must separately coordinate retirement and GPU execution leases.
    const HistoricalKvShadowDirectory &historical_shadow_directory() const {
        return historical_shadow_directory_;
    }
    bool bind_historical_full_block(
        uint32_t block_id, std::shared_ptr<HistoricalKvRepository> repository,
        HistoricalKvHandle handle,
        HistoricalKvRepository::Lease & writer,
        HistoricalKvShadowDirectory::Producer & producer,
        std::shared_ptr<HistoricalKvRawStore> storage = {}) {
        return historical_shadow_directory_.bind_full_block(
            store_, block_id, std::move(repository), handle, writer, producer, std::move(storage));
    }
    bool unbind_retired_historical_block(uint32_t block_id) {
        return historical_shadow_directory_.unbind_retired(block_id);
    }
    bool bind_historical_block(
        uint32_t block_id, std::shared_ptr<HistoricalKvRepository> repository,
        HistoricalKvHandle handle, HistoricalKvRepository::Lease & writer,
        HistoricalKvShadowDirectory::Producer & producer,
        std::shared_ptr<HistoricalKvRawStore> storage) {
        return historical_shadow_directory_.bind_block(
            store_, block_id, std::move(repository), handle, writer, producer, std::move(storage));
    }
    bool register_historical_append(
        std::shared_ptr<HistoricalKvRepository> repository,
        std::shared_ptr<HistoricalKvRawStore> storage,
        std::vector<HistoricalKvRepository::Lease> & writers,
        std::vector<HistoricalKvShadowDirectory::Producer> & producers,
        HistoricalKvShadowDirectory::AppendReservation * reservation = nullptr) {
        if (reservation && pending_) return false;
        if (reservation) return historical_shadow_directory_.append_reserved_blocks(
            store_, *reservation, std::move(repository), std::move(storage), writers, producers);
        return historical_shadow_directory_.append_blocks(
            store_, std::move(repository), std::move(storage), writers, producers);
    }
    std::unique_ptr<HistoricalKvShadowDirectory::AppendReservation> prepare_historical_metadata(
        std::shared_ptr<HistoricalKvRepository> repository,
        std::shared_ptr<HistoricalKvRawStore> storage, uint32_t capacity) {
        return historical_shadow_directory_.prepare_append_reservation(
            store_, std::move(repository), std::move(storage), capacity);
    }
    bool install_historical_metadata(HistoricalKvShadowDirectory::AppendReservation & reservation) noexcept {
        return historical_shadow_directory_.install_append_reservation(store_, reservation);
    }
    bool prepare_recycled_historical_slot(
            HistoricalKvShadowDirectory::AppendReservation & reservation, uint32_t block_id) {
        if (pending_ || !pending_gpu_frees_.empty() || !prefetch_futs_.empty()) return false;
        return historical_shadow_directory_.prepare_recycled_slot(store_, reservation, block_id);
    }
    bool confirm_reclaimed_historical_block(uint32_t block_id) {
        return historical_shadow_directory_.confirm_reclaimed(block_id);
    }
    bool detach_reclaimed_historical_block(uint32_t block_id) noexcept {
        if (pending_ || !pending_gpu_frees_.empty() || !prefetch_futs_.empty()) return false;
        return historical_shadow_directory_.detach_reclaimed_span(store_, block_id);
    }
    // True while a prepared plan has not been applied yet (prepare_selection /
    // prepare_prefill_pressure set it, admit_incoming clears it).
    bool pending() const { return pending_; }
    bool plan_is_current(const KvMemPlan & plan) const noexcept {
        return pending_ && plan.plan_id != 0 && plan.plan_id == last_plan_.plan_id &&
               plan.table_revision == last_plan_.table_revision &&
               plan.table_revision == store_.table_revision();
    }
    // A parked runtime has no GPU slots or unapplied transfers.
    void rebind_backend(KvMemBackend * backend);
    size_t allocated_bytes() const {
        size_t bytes = sizeof(*this) + store_.allocated_bytes() +
            historical_shadow_directory_.allocated_bytes() +
            cpu_arena_.capacity() + scratch_.capacity();
        bytes += last_plan_.stage_in.capacity()*sizeof(uint32_t) +
            last_plan_.stage_out.capacity()*sizeof(uint32_t) + last_plan_.remaps.capacity()*sizeof(KvMemRemap);
        bytes += pending_gpu_frees_.capacity()*sizeof(int32_t) + prefetch_futs_.capacity()*sizeof(std::future<void>);
        for (const auto & entry : prefetch_buf_) if (entry.second) bytes += entry.second->capacity();
        return bytes;
    }
    // Abandon a prepared plan instead of applying it. The staging half may
    // already have run, so the caller owns putting its own view of residency
    // back; this drops the pending marker and the GPU slots the plan had
    // queued for admit_incoming() to free through the backend, which a caller
    // that rebuilds its whole free-slot list does not need handed back.
    void discard_pending() {
        pending_ = false;
        pending_gpu_frees_.clear();
    }

    PinnedKvTier *cpu_tier() { return cpu_tier_.get(); }
    NvmeKvTier *nvme_tier() { return nvme_tier_.get(); }

    void register_append(uint32_t n_tokens);
    std::vector<KvMemDroppedBlock> truncate_to(uint32_t token_pos);

    // Recency / retrieval selection. `mandatory` consumes budget slots.
    KvMemPlan prepare_reselect(const std::vector<uint32_t> &mandatory = {},
                               bool force_raw_refresh = false);
    std::vector<uint32_t> preview_reselect(const std::vector<uint32_t> & mandatory = {}) const;
    KvMemPlan prepare_selection(const std::vector<uint32_t> & selected, bool force_raw_refresh = false);
    bool commit_resident_selection(const std::vector<uint32_t> & selected);
    KvMemPlan prepare_prefill_pressure(
        const std::vector<uint32_t> &mandatory = {});

    // If the next prefill chunk would overflow the semantic budget, the hard
    // GPU pool, or the high watermark, build a sink+tail pressure plan.
    // Does not apply it; caller runs finish_reselect (evict-before-stage-in).
    bool maybe_offload_during_prefill(
        uint32_t incoming_tokens,
        uint32_t resident_tokens,
        uint32_t pool_tokens,
        const std::vector<uint32_t> &mandatory = {});

    // Copy outgoing GPU blocks to CPU/NVMe. Does not free GPU slots yet so
    // the adapter can seq_rm the cells afterwards.
    void spill_outgoing();
    // Free spilled GPU slots then stage-in (host/NVMe -> GPU).
    void admit_incoming();
    // spill_outgoing + admit_incoming (host tests; no llama seq_rm).
    void finish_reselect();

    void reselect() {
        prepare_reselect();
        finish_reselect();
    }

private:
    void stage_out(uint32_t block_id);
    void stage_in(uint32_t block_id);
    void start_prefetch();
    void wait_prefetch();
    uint8_t *cpu_ptr(int32_t slot);
    const uint8_t *cpu_ptr(int32_t slot) const;
    bool spill_bytes_to_nvme(uint32_t block_id, const void *data);
    void trace_tier(const char *tag, uint32_t block_id, int32_t slot) const;

    KvMemRuntimeConfig cfg_;
    KvMemStore store_;
    HistoricalKvShadowDirectory historical_shadow_directory_;
    KvMemBackend *backend_;
    KvMemBackend null_backend_;
    std::unique_ptr<PinnedKvTier> cpu_tier_;
    std::unique_ptr<NvmeKvTier> nvme_tier_;
    KvMemPlan last_plan_;
    uint64_t plan_serial_ = 0;
    bool pending_ = false;
    bool trace_ = false;
    uint64_t slot_bytes_ = 0;
    std::vector<uint8_t> cpu_arena_;
    std::vector<uint8_t> scratch_;
    std::unordered_map<uint32_t, std::shared_ptr<std::vector<uint8_t>>> prefetch_buf_;
    std::vector<std::future<void>> prefetch_futs_;
    std::vector<int32_t> pending_gpu_frees_;
};

} // namespace kvmem
