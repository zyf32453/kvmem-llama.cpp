#pragma once

#include "llama-kv-cache.h"
#include "llama-memory.h"
#include "llama-kvmem-hooks.h"

#include "kvmem/kvmem_runtime.hpp"
#include "kvmem/historical_kv_cleanup.hpp"
#include "kvmem/historical_kv_active_set.hpp"
#include "kvmem/fixed_byte_arena.hpp"
#include "kvmem/raw_kv_store.hpp"
#include "kvmem/rope.hpp"

#include <condition_variable>
#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <exception>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

struct llama_model;
struct llama_cparams;
struct llama_memory_params;
struct ggml_tensor;
class llama_memory_recurrent;
class llama_memory_kvmem_mtp;
namespace kvmem { class HistoricalKvCaptureSession; class HistoricalKvCaptureDomain; class HistoricalKvActiveSet; class HistoricalKvGc; }

// Bounded block-slot pool over a llama_kv_cache.
//
// GPU attention cache size = min(n_ctx, budget + gen_reserve). Each logical
// KVMem block occupies one slot of `block_tokens` cells. Reselect never packs
// cells into [0, W); resident blocks keep their slot. init_batch returns a
// llama_kv_cache_context so llama-graph.cpp can static_cast as usual.
class llama_memory_kvmem : public llama_memory_i {
public:
    llama_memory_kvmem(
            const llama_model & model,
            const llama_memory_params & params,
            const llama_cparams & cparams,
            llama_kv_cache * ext_kv = nullptr);

    ~llama_memory_kvmem() override;

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    bool get_can_shift() const override { return false; }

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id) override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    llama_kv_cache * get_kv() { return kv_; }
    kvmem::KvMemRuntime & runtime() { return *runtime_; }
    const kvmem::KvMemRuntime & runtime() const { return *runtime_; }

    uint32_t kv_size() const { return kv_size_; }
    uint32_t block_tokens() const { return block_tokens_; }
    uint32_t n_slots() const { return n_slots_; }
    bool multi_gpu() const { return multi_gpu_; }

    // Slot-pool prepare used by both the dense KVMem memory and the hybrid
    // wrapper (attn half). Fills per-ubatch slot_info and the capture pos queue.
    bool prepare_ubatches(
            const std::vector<llama_ubatch> & ubatches,
            uint32_t n_new_tokens,
            llama_kv_cache::slot_info_vec_t & sinfos);
    void reset_policy();

    int32_t alloc_slot();
    void free_slot(int32_t slot);
    int32_t peek_free_slot() const;
    // Map original pos to (gpu_slot, offset in block). For a token the target
    // has not appended yet (MTP draft), predict the slot the next alloc would
    // take without popping the free list. Returns false if no mapping exists.
    bool slot_for_orig_pos(llama_pos pos, int32_t * slot, uint32_t * off) const;
    const kvmem::KvMemStore & store() const { return runtime_->store(); }

    void note_ubatch_pos(const std::vector<llama_pos> & pos);
    void reset_query_acc();
    void register_capture(struct ggml_tensor * t, int il, char which);
    void capture_on_new_graph();
    bool capture_can_reuse(uint32_t n_tokens, uint32_t n_pos, const llama_pos * pos) const;
    uint64_t capture_stamp() const;
    void harvest_pending(struct ggml_backend_sched * sched);
    bool begin_active_ubatch(uint32_t n_tokens, const llama_pos * logical_positions);
    void mark_active_submission();
    void complete_active_ubatch() noexcept;
    uint64_t historical_active_skips() const noexcept { return historical_active_skips_; }
    void harvest_flush();
    void harvest_perf_print_sum();
    void harvest_capture(struct ggml_tensor * t, int il, char which);
    void apply_retrieval();
    void set_turn_spans(const llama_kvmem_turn_spans & spans);
    bool query_contains(llama_pos row) const;
    bool query_overlaps(uint32_t n, const llama_pos * rows) const;
    llama_kvmem_attention_view attention_view(bool canonical = true) const;
    bool can_append(uint32_t end, uint32_t generation_rows, bool all_history, std::string & reason) const;
    llama_kvmem_selection preview_retrieval();
    bool selection_fits(const llama_kvmem_selection & selection, uint32_t end, uint32_t generation_rows) const;
    bool commit_unchanged(const llama_kvmem_attention_view & view, const llama_kvmem_selection & selection);
    void apply_selection(const llama_kvmem_selection & selection);
    bool commit_resident(bool canonical = true);
    bool get_query(llama_kvmem_query_state & state);
    bool set_query(const llama_kvmem_query_state & state);
    void freeze_query(bool frozen) { query_frozen_ = frozen; }
    // The follower also invalidates a pending proof when old draft KV changes.
    void note_attention_change() { ++attention_epoch_; }
    bool query_replay_fits(uint32_t query_begin, uint32_t prompt_end) const;
    void dump_kv_compare(int32_t block_id, bool writeback_test = false);
    void trace_working_set(const char * tag) const;
    void set_replay(bool replay);
    bool replay() const { return replay_; }
    bool want_prefill_capture() const {
        return prefill_capture_ && !retrieval_pinned_ && !replay_;
    }
    // Recapture Q for retrieval even while replaying a cached query span.
    bool want_q_capture() const {
        return method_ == 1 && !retrieval_pinned_ && !query_frozen_ &&
            (explicit_spans_ ? !turn_spans_.query.empty() : query_begin_ >= 0);
    }
    bool want_decode_mean() const {
        return retrieval_pinned_ && method_ == 1 && !replay_;
    }
    void end_prefill_capture();
    // Finite sequential requests on one recency RAM sequence. Other combinations
    // remain unsupported until their execution boundaries are lease-gated.
    bool begin_historical_capture(uint32_t capacity, uint64_t input_tokens,
                                  uint64_t max_output_tokens, uint64_t max_appends);
    bool start_historical_gc(int mode, uint32_t ready_low, uint32_t ready_high);
    bool finish_historical_capture();
    void fail_historical_capture() noexcept;
    // Next request continues this sequence: flush decode mean, unpin
    // retrieval, harvest mean-K for the suffix. Does not wipe prefix KV.
    void begin_cached_turn(bool reset_query = true);
    // Same-query skip: keep the retrieved GPU window. Prefill still harvests
    // the new tail; recency pressure must not evict selected history.
    void keep_selected_window() { keep_selected_ = true; }
    // After skip/reselect prefill: pin so decode mean-K uses gen_reserve.
    void pin_working_set() {
        retrieval_pinned_ = true;
        keep_selected_ = true;
    }
    size_t free_slot_count() const { return free_slots_.size(); }
    void truncate_cached(uint32_t n_past);
    void occupy_in(llama_kv_cache * cache, uint32_t block_id);
    llama_pos model_pos(uint32_t logical_pos) const;
    bool remove_logical(llama_context * ctx, llama_pos begin, llama_pos end);
    uint32_t store_n_tokens() const {
        return runtime_ ? runtime_->store().total_tokens() : 0;
    }
    llama_pos recr_pos_max() const;
    void decode_mean_commit(uint32_t n_keep);
    void decode_mean_discard();
    void decode_mean_flush();
    void set_recurrent(llama_memory_recurrent * recr);
    bool has_recurrent() const { return recr_ != nullptr; }
    bool gdn_replay_enabled() const;
    bool gdn_replay_begin(llama_pos start, uint32_t width);
    bool gdn_replay_commit(llama_context * ctx, uint32_t n_keep);
    void set_query_span(int32_t begin, int32_t end) {
        harvest_flush();
        explicit_spans_ = false;
        query_begin_ = begin;
        query_end_ = end;
    }
    void set_force_pos(int32_t pos) { force_pos_ = pos; }
    void set_mtp_follower(llama_memory_kvmem_mtp * mtp);
    llama_memory_kvmem_mtp * mtp_follower() { return mtp_; }

    const kvmem::RopeConfig & rope() const { return rope_; }
    ggml_type type_k() const { return type_k_; }
    ggml_type type_v() const { return type_v_; }
    bool v_trans() const { return v_trans_; }
    uint32_t n_embd_k() const { return n_embd_k_; }
    uint32_t n_embd_v() const { return n_embd_v_; }

    kvmem::RawKvStore & raw() { return *raw_; }

    // One conversation's host KV, moved out of this object and back in so N
    // conversations keep their store in host RAM while the single GPU working
    // set is time-multiplexed between them. Declared here and defined in the
    // .cpp like GdnReplay above, so the hook header stays free of kvmem types.
    struct ConvStore;
    // False when a swap can never be safe in this configuration: without flash
    // attention V is not mirrored to host, and raw-K NVMe sizes and names one
    // arena per process.
    bool conv_swap_supported(std::string & reason) const;
    bool conv_compatible(const llama_memory_kvmem & other) const;
    std::unique_ptr<ConvStore> make_conv();
    // Exchange the per-conversation host state. Call only between requests,
    // under the server's request lock. `conv` must come from make_conv() or an
    // earlier swap_conv() on a compatible memory object; on return it holds the outgoing
    // conversation, drained to host. False means the incoming store holds no
    // rows live on the GPU: it was empty, or its working set could not be
    // rebuilt from host RAM and it was reset to empty. Either way the server
    // reads it as an ordinary cache miss, the same claim store_n_tokens() > 0
    // makes. Ownership is all-or-nothing, and there is exactly one window: this
    // can throw only before the handover -- out of the drain, or out of the
    // clear a store that cannot be drained safely gets -- and then this object
    // still owns the outgoing store while `conv` still holds the incoming one.
    // Past the handover nothing throws: a failed attach empties the incoming
    // store and returns false. Contents are not all-or-nothing -- a store that
    // cannot be drained safely, or whose drain throws, is emptied rather than
    // kept, which the caller sees as its row count dropping to zero.
    bool swap_conv(std::unique_ptr<ConvStore> & conv);
    uint32_t conv_n_tokens(const ConvStore & conv) const;
    uint64_t conv_host_bytes(const ConvStore & conv) const;
    uint64_t host_bytes() const;
    uint64_t host_capacity(uint32_t tokens) const;

private:
    bool continue_historical_capture(uint32_t capacity, uint64_t input_tokens,
                                     uint64_t max_output_tokens, uint64_t max_appends);
    size_t row_metadata_index(uint32_t logical) const;
    friend struct kvmem_transfer_test_access;
    struct SlotBackend : public kvmem::KvMemBackend {
        llama_memory_kvmem * owner = nullptr;
        int32_t alloc_gpu_slot() override { return owner->alloc_slot(); }
        void free_gpu_slot(int32_t slot) override { owner->free_slot(slot); }
        void copy_block_to_host(uint32_t block_id, int32_t gpu_slot,
                                void * host, uint64_t bytes) override {
            owner->copy_gpu_block_to_host(block_id, gpu_slot, host, bytes);
        }
        void copy_block_from_host(uint32_t block_id, int32_t gpu_slot,
                                  const void * host, uint64_t bytes) override {
            owner->copy_gpu_block_from_host(block_id, gpu_slot, host, bytes);
        }
    };

    uint32_t resident_tokens() const;
    bool prepare_working_set(uint32_t n_new_tokens);
    void apply_plan_to_kv(const kvmem::KvMemPlan & plan);
    // Place GPU-resident blocks into slots 0..N-1 in orig_pos order.
    // Resident KV is copied slot-to-slot; cold blocks memcpy packed GPU K/V.
    bool layout_gpu_slots_by_orig_pos();
    struct LayoutItem {
        uint32_t id, orig;
        int32_t slot;
        uint32_t n;
        bool resident;
    };
    struct LayoutWorkspace {
        std::vector<LayoutItem> items;
        std::vector<uint8_t> mtp_res, seen;
        std::vector<size_t> res_ix, gbytes, sbytes;
        std::vector<const void *> gsrc, ssrc;
        std::vector<void *> gdst, sdst;
        void reserve(size_t slots) {
            items.reserve(slots); mtp_res.reserve(slots); seen.reserve(slots);
            res_ix.reserve(slots);
            gsrc.reserve(2 * slots); gdst.reserve(2 * slots); gbytes.reserve(2 * slots);
            ssrc.reserve(2 * slots); sdst.reserve(2 * slots); sbytes.reserve(2 * slots);
        }
        size_t capacity_bytes() const noexcept {
            return items.capacity() * sizeof(LayoutItem) + mtp_res.capacity() + seen.capacity() +
                (res_ix.capacity() + gbytes.capacity() + sbytes.capacity()) * sizeof(size_t) +
                (gsrc.capacity() + ssrc.capacity() + gdst.capacity() + sdst.capacity()) * sizeof(void *);
        }
    } layout_workspace_;
    struct HistoricalActiveWorkspace {
        std::vector<kvmem::HistoricalKvActiveRequest> requested;
        std::vector<uint32_t> requested_ids, remove_ids;
        std::vector<uint8_t> seen;
        void reserve(size_t slots) {
            requested.reserve(slots); requested_ids.reserve(slots); remove_ids.reserve(slots);
            seen.resize(slots);
        }
        size_t capacity_bytes() const noexcept {
            return requested.capacity() * sizeof(kvmem::HistoricalKvActiveRequest) +
                (requested_ids.capacity() + remove_ids.capacity()) * sizeof(uint32_t) + seen.capacity();
        }
    } historical_active_workspace_;
    // Scratch for rebuilding the cells of one GPU block. Historical requests
    // also reserve upstream index nodes; full graph ubatches remain separate.
    struct CellWorkspace {
        llama_kv_cache::slot_info slots;
        std::vector<llama_pos> positions, logical;
        std::vector<int32_t> n_seq_id;
        std::vector<llama_seq_id> seq_data;
        std::vector<llama_seq_id *> seq_id;
        std::vector<int8_t> output;
        std::vector<llama_token> tokens;
        uint32_t max_tokens = 0;
        void reserve(uint32_t count) {
            if (size_t(count) > SIZE_MAX / 4) throw std::length_error("cell workspace size overflow");
            slots.s0 = slots.s1 = 0;
            slots.resize(1); slots.strm[0] = 0; slots.idxs[0].resize(count);
            positions.resize(size_t(count) * 4); logical.resize(count);
            n_seq_id.resize(count, 1); seq_data.resize(count, 0);
            seq_id.resize(count); output.resize(count, 0); tokens.resize(count, 0);
            max_tokens = count;
        }
        size_t capacity_bytes() const noexcept {
            return slots.strm.capacity() * sizeof(llama_seq_id) +
                slots.idxs.capacity() * sizeof(llama_kv_cache::slot_info::idx_vec_t) +
                (slots.idxs.empty() ? 0 : slots.idxs[0].capacity() * sizeof(uint32_t)) +
                (positions.capacity() + logical.capacity()) * sizeof(llama_pos) +
                n_seq_id.capacity() * sizeof(int32_t) + seq_data.capacity() * sizeof(llama_seq_id) +
                seq_id.capacity() * sizeof(llama_seq_id *) + output.capacity() +
                tokens.capacity() * sizeof(llama_token);
        }
    } cell_workspace_;
    // The actual slot_info vectors travel into one llama_kv_cache_context and
    // return from its destructor. Only one decode context may own them at a
    // time in the admitted single-request path.
    struct HistoricalSlotWorkspace {
        llama_kv_cache::slot_info_vec_t slots;
        uint32_t n_batch = 0, n_ubatch = 0;
        uint32_t max_tokens = 0;
        void prepare(uint32_t batch, uint32_t ubatch, uint64_t request_tokens) {
            if (!batch || !ubatch || !request_tokens) throw std::length_error("invalid historical slot workspace bound");
            n_batch = batch; n_ubatch = ubatch;
            max_tokens = static_cast<uint32_t>(std::min<uint64_t>(batch, request_tokens));
            slots.resize((size_t(max_tokens) + ubatch - 1) / ubatch);
            for (size_t i = 0; i < slots.size(); ++i) {
                auto & slot = slots[i];
                slot.s0 = slot.s1 = 0;
                slot.resize(1); slot.strm[0] = 0;
                slot.idxs[0].reserve(std::min<size_t>(ubatch, size_t(max_tokens) - i * ubatch));
            }
        }
        size_t capacity_bytes() const noexcept {
            size_t result = slots.capacity() * sizeof(llama_kv_cache::slot_info);
            for (const auto & slot : slots) {
                result += slot.strm.capacity() * sizeof(llama_seq_id) +
                    slot.idxs.capacity() * sizeof(llama_kv_cache::slot_info::idx_vec_t);
                for (const auto & idx : slot.idxs) result += idx.capacity() * sizeof(uint32_t);
            }
            return result;
        }
    } historical_slot_workspace_;
    struct HistoricalUbatchWorkspace {
        // Pinned upstream has eight width-specific decode graph results and
        // two ordinary graph results. Each may retain one older data_t.
        static constexpr size_t graph_retained_max = 10;
        std::vector<std::shared_ptr<llama_ubatch::data_t>> data;
        std::vector<llama_ubatch> ubatches;
        std::vector<int32_t> indices;
        uint32_t max_tokens = 0, n_ubatch = 0;
        size_t max_ubatches = 0;
        void prepare(uint32_t batch, uint32_t ubatch, uint64_t request_tokens) {
            if (!batch || !ubatch || !request_tokens)
                throw std::length_error("invalid historical microbatch bound");
            max_tokens = static_cast<uint32_t>(std::min<uint64_t>(batch, request_tokens));
            n_ubatch = ubatch;
            max_ubatches = (size_t(max_tokens) + ubatch - 1) / ubatch;
            if (max_ubatches > SIZE_MAX - graph_retained_max)
                throw std::length_error("historical microbatch pool count overflow");
            data.reserve(max_ubatches + graph_retained_max);
            ubatches.reserve(max_ubatches);
            indices.reserve(std::min<uint32_t>(max_tokens, ubatch));
            for (size_t i = 0; i < max_ubatches + graph_retained_max; ++i) {
                const size_t n = std::min<uint32_t>(max_tokens, ubatch);
                auto item = std::make_shared<llama_ubatch::data_t>();
                item->token.reserve(n);
                item->logical_pos.reserve(n);
                item->pos.reserve(n);
                item->n_seq_id.reserve(n);
                item->seq_id.reserve(n);
                item->seq_id_unq.reserve(1);
                item->seq_idx.resize(LLAMA_MAX_SEQ, -1);
                item->output.reserve(n);
                item->seq_id_data.reserve(n);
                data.push_back(std::move(item));
            }
        }
        size_t vector_bytes() const noexcept {
            size_t bytes = data.capacity() * sizeof(data[0]) +
                ubatches.capacity() * sizeof(llama_ubatch) + indices.capacity() * sizeof(int32_t);
            for (const auto & item : data) {
                bytes += item->token.capacity() * sizeof(llama_token) +
                    (item->logical_pos.capacity() + item->pos.capacity()) * sizeof(llama_pos) +
                    item->n_seq_id.capacity() * sizeof(int32_t) +
                    item->seq_id.capacity() * sizeof(llama_seq_id *) +
                    item->seq_id_unq.capacity() * sizeof(llama_seq_id) +
                    item->seq_idx.capacity() * sizeof(int32_t) + item->output.capacity() +
                    item->seq_id_data.capacity() * sizeof(llama_seq_id);
            }
            return bytes;
        }
    } historical_ubatch_workspace_;
    class HistoricalBatchContext;
    uint32_t context_n_batch_ = 0, context_n_ubatch_ = 0;
    bool gpu_kv_already_resident(uint32_t block_id) const;
    bool gpu_kv_complete(uint32_t block_id, const llama_kv_cache * cache) const;
    std::vector<uint32_t> retrieval_mandatory() const;
    void occupy_block_cells(uint32_t block_id);
    void reset_slots();
    // The non-destructive half of reset_policy(): turn/request flags only.
    void reset_turn_policy();
    bool conv_can_drain(std::string & reason) const;
    std::unique_ptr<ConvStore> detach_conv();
    bool attach_conv(std::unique_ptr<ConvStore> conv);
    // Best-effort "attached, holding nothing", for swap_conv's two repair
    // paths. Never throws: the drain path rethrows the drain's own exception
    // after it, and the attach path runs after both bundles have changed
    // hands, where an escape would corrupt store identity. `what` names the
    // path in the log and nothing else.
    void conv_reset_to_empty(const char * what) noexcept;
    void trace_plan(const char * tag, const kvmem::KvMemPlan & plan) const;
    void write_block_to_gpu(uint32_t block_id);
    void harvest_gpu_v(uint32_t block_id);
    void harvest_gpu_v_commit();
    void harvest_gpu_v_flush_slab();
    void harvest_write_batch();
    // After a prefill graph, enqueue packed K/V D2H for GPU-resident
    // full blocks. Does not wait; apply_plan / retrieval commit.
    void harvest_full_blocks_async();
    void decode_mean_ingest(struct ggml_backend_sched * sched);
    void decode_mean_reset();
    void decode_mean_shutdown();
    struct MeanWriteRange;
    void decode_mean_add_range(const MeanWriteRange & range, uint32_t n_add);
    void decode_mean_zero_acc();
    bool decode_mean_add_host_layer(struct ggml_tensor * t, int il, uint32_t tok0, uint32_t n_add);
    void decode_mean_print_sum();
    struct HarvestVJob {
        uint32_t pos0 = 0;
        uint32_t n = 0;
        uint32_t il = 0;
        bool is_k = false;
        const uint8_t * gpu_src = nullptr;
        size_t nbytes = 0;
        size_t pin_off = 0;
        ggml_tensor * vt = nullptr;
        size_t tensor_off = 0;
        uint32_t token_offset = 0;
        // Present only for the finite managed capture owner. The physical
        // slot cannot be reused until this queued source has completed.
        int32_t historical_gpu_slot = -1;
        bool committed = false;
        // Captured before D2H submission. Commit uses this exact writer and
        // allocation even after the foreground retires/unbinds the block.
        std::shared_ptr<kvmem::HistoricalKvShadowDirectory::WriterContext> writer;
    };
    void commit_harvest_packed(const HarvestVJob & job, const uint8_t * bytes);
    void enqueue_harvest_packed(HarvestVJob job);
    uint8_t * harvest_packed_fallback(size_t bytes, std::vector<uint8_t> & temporary);
    void complete_historical_gpu_source(int32_t slot);
    struct HarvestVBatch {
        int slot = -1;
        std::vector<HarvestVJob> jobs;
    };
    std::vector<HarvestVJob> harvest_v_jobs_;
    HarvestVBatch harvest_v_pending_;
    size_t harvest_v_job_limit_ = 0;
    std::vector<uint8_t> harvest_v_fallback_;
    // 1 while a block's packed D2H is queued or in flight (until commit).
    std::vector<uint8_t> harvest_gpu_queued_;
    void copy_gpu_block_to_host(uint32_t block_id, int32_t gpu_slot,
                                void * host, uint64_t bytes);
    void copy_gpu_block_from_host(uint32_t block_id, int32_t gpu_slot,
                                  const void * host, uint64_t bytes);
    void score_retrieval();
    bool read_gpu_block(uint32_t block_id, uint32_t il, bool is_k, std::vector<float> & out) const;
    static void kv_stats(const char * tag, const float * a, const float * b, size_t n);
    static void tensor_to_f32_token_major(const struct ggml_tensor * t, std::vector<float> & out, int64_t * read_us = nullptr);
    static void bytes_to_f32_token_major(const uint8_t * data, ggml_type type,
                                         int64_t d, int64_t h, int64_t n,
                                         size_t nb0, size_t nb1, size_t nb2,
                                         std::vector<float> & out);
    static void bytes_to_f16_token_major(const uint8_t * data, ggml_type type,
                                         int64_t d, int64_t h, int64_t n,
                                         size_t nb0, size_t nb1, size_t nb2,
                                         std::vector<uint16_t> & out);
    struct CaptureBatch;
    CaptureBatch capture_batch(const llama_pos * positions, size_t count);
    // Uses the producer and GPU source captured before execution. Only the
    // accepted prefix of this executed batch may enter the host payload.
    void harvest_captured_packed(const CaptureBatch & batch, uint32_t n_keep);
    void commit_mean_k(const CaptureBatch & batch, uint32_t il, const std::vector<float> & flat);
    void harvest_from_host(const CaptureBatch & batch, int il, char which, const uint8_t * host,
                           ggml_type type, int64_t d, int64_t h, int64_t n,
                           size_t nb0, size_t nb1, size_t nb2);
    bool d2h_init();
    void d2h_free();
    bool multi_d2h_submit();
    void multi_d2h_free();
    void d2h_commit(int slot);
    bool d2h_submit(struct ggml_backend * be);
    bool harvest_perf_on() const { return perf_.enabled; }
    void harvest_perf_emit_graph_line();
    void retr_perf_print();
    void harvest_worker_start();
    void harvest_worker_stop(bool cancel_queued = false);
    void harvest_loop();
    void harvest_wait_slot(int slot);
    bool harvest_worker_on() const;
    void harvest_check_error();
    void harvest_merge_stats();
    void harvest_shutdown();
    void capture_fail() noexcept;
    enum class CaptureFaultPoint { DeviceSelect, EventWait, HostCommit, SnapshotCopy, HostCopy, CompletionEvent, Drain,
                                   DecodeHostCommit };
    using CaptureFault = bool (*)(void *, CaptureFaultPoint, int, size_t) noexcept;
    bool capture_fault(CaptureFaultPoint point, int slot = -1, size_t index = 0) const noexcept;
    CaptureFault capture_fault_ = nullptr;
    void * capture_fault_context_ = nullptr;
    bool capture_failed_ = false; // foreground-owned; worker publishes error instead

    struct HarvestPerf {
        bool enabled = false;
        bool sum_printed = false;
        bool graph_line_printed = false;
        uint32_t n_ubatch = 0;
        uint32_t n_tok = 0;
        int64_t harvest_entry_us = 0;
        int64_t sync_us = 0;
        int64_t d2d_us = 0;
        int64_t snap_wait_us = 0;
        int64_t d2h_submit_us = 0;
        int64_t commit_us = 0;
        int64_t d2h_wait_us = 0;
        int64_t pack_us = 0;
        int64_t nvme_us = 0;
        uint64_t nvme_bytes = 0;
        uint64_t nvme_syscalls = 0;
        int64_t last_sync_us = 0;
        int64_t last_d2d_us = 0;
        int64_t last_snap_wait_us = 0;
        int64_t last_d2h_submit_us = 0;
        int64_t last_commit_us = 0;
        int64_t last_d2h_wait_us = 0;
        int64_t last_pack_us = 0;
        int64_t last_nvme_us = 0;
        uint64_t last_nvme_bytes = 0;
        uint64_t last_nvme_syscalls = 0;
        uint32_t n_pressure = 0;
        uint32_t n_pressure_out = 0;
        int64_t multi_read_us = 0;
        int64_t multi_host_us = 0;
    };

    struct RetrPerf {
        bool enabled = false;
        int64_t total_us = 0;
        int64_t flush_us = 0;
        int64_t score_us = 0;
        int64_t plan_us = 0;
        int64_t stage_out_us = 0;
        int64_t stage_out_gpu_us = 0;
        int64_t stage_out_host_us = 0;
        int64_t admit_us = 0;
        int64_t seq_rm_us = 0;
        int64_t occupy_us = 0;
        int64_t layout_d2h_us = 0;
        int64_t layout_h2d_us = 0;
        int64_t copy_us = 0;
        int64_t rope_us = 0;
        int64_t hadamard_us = 0;
        int64_t set_us = 0;
        int64_t mtp_us = 0;
        int64_t dump_us = 0;
        uint32_t n_move = 0;
        uint32_t n_raw = 0;
        uint32_t n_skip = 0;
        uint32_t n_stage_in = 0;
        int laid_out = 0;
    };

    const llama_model & model_;
    uint32_t block_tokens_ = 128;
    uint32_t kv_size_ = 0;
    uint32_t n_slots_ = 0;
    bool multi_gpu_ = false;
    bool trace_ = false;

    std::unique_ptr<llama_kv_cache> kv_owned_;
    llama_kv_cache * kv_ = nullptr;
    llama_memory_recurrent * recr_ = nullptr;
    struct GdnReplay;
    std::unique_ptr<GdnReplay> gdn_replay_;
    llama_memory_kvmem_mtp * mtp_ = nullptr;
    SlotBackend backend_;
    // Copies of the configs the constructor computes, so a sibling store for
    // another conversation is configured identically.
    kvmem::KvMemRuntimeConfig rt_cfg_{};
    kvmem::RawKvStoreConfig raw_cfg_{};
    std::unique_ptr<kvmem::KvMemRuntime> runtime_;
    std::unique_ptr<kvmem::RawKvStore> raw_;
    struct HistoricalCpuCleanup;
    kvmem::HistoricalKvCleanup::Ticket historical_cleanup_ticket_;
    std::unique_ptr<HistoricalCpuCleanup> historical_cleanup_;
    std::shared_ptr<kvmem::HistoricalKvCaptureDomain> historical_domain_;
    std::unique_ptr<kvmem::HistoricalKvCaptureSession> historical_capture_;
    // One decode context at a time. Prepared before capture admission and
    // reused by successful historical init_batch calls; failure is terminal.
    void * historical_batch_context_storage_ = nullptr;
    size_t historical_batch_context_storage_bytes_ = 0;
    bool historical_batch_context_in_use_ = false;
    std::unique_ptr<kvmem::HistoricalKvGc> historical_gc_;
    int historical_gc_mode_ = 0;
    std::unique_ptr<kvmem::HistoricalKvActiveSet> historical_active_;
    // Deterministic test seam immediately after graph submission and before
    // harvest synchronization. Empty in production requests.
    std::function<void()> historical_submission_test_hook_;
    uint64_t historical_active_skips_ = 0;
    // Foreground-owned, fixed-size accounting; the CPU GC never touches GPU
    // slots. Retired cells disappear immediately, source slots return later.
    std::vector<uint32_t> historical_gpu_readers_;
    std::vector<uint8_t> historical_gpu_retired_;
    uint64_t historical_gpu_deferred_ = 0;
    uint64_t historical_gpu_returned_ = 0;
    std::vector<int32_t> free_slots_;
    struct RowPosition {
        std::array<llama_pos, 4> pos{};
        llama_token token = LLAMA_TOKEN_NULL;
        bool spatial = false;
    };
    std::vector<RowPosition> row_positions_;

    kvmem::RopeConfig rope_{};
    uint32_t n_layer_ = 0;
    uint32_t n_embd_k_ = 0;
    uint32_t n_embd_v_ = 0;
    uint32_t n_head_ = 0;
    uint32_t n_head_kv_ = 0;
    uint32_t n_embd_head_ = 0;
    ggml_type type_k_ = GGML_TYPE_F16;
    ggml_type type_v_ = GGML_TYPE_F16;
    bool v_trans_ = false;
    bool replay_ = false;
    // Set when a full seq_rm zeroes the block table while leaving the host
    // mirror behind, so raw_ still holds blocks the table no longer owns and
    // truncate_cached's own guard can no longer reach them. Nothing on the
    // default path reads this; a store swap refuses to carry such a store and
    // clears it instead of restaging another turn's packed K. reset_policy()
    // puts the two back in lockstep at zero and clears it.
    bool host_mirror_stale_ = false;
    bool retrieval_pinned_ = false;
    bool keep_selected_ = false;
    bool prefill_capture_ = true;
    int32_t method_ = 0;
    int32_t query_begin_ = -1;
    int32_t query_end_ = -1;
    int32_t force_pos_ = -1;
    llama_kvmem_turn_spans turn_spans_;
    bool explicit_spans_ = false;
    bool query_frozen_ = false;
    uint64_t attention_epoch_ = 0;

    struct MeanWriteRange {
        enum class Route { Legacy, Managed, Closed };
        Route route = Route::Closed;
        uint32_t source_token = 0;
        uint32_t token_count = 0;
        uint32_t original_begin = 0;
        uint32_t block_id = 0;
        uint64_t original_end = 0; // admitted span end, not (block_id + 1) * B
        uint32_t token_offset = 0;
        int32_t gpu_slot = -1;
        std::shared_ptr<kvmem::HistoricalKvShadowDirectory::WriterContext> writer;
    };
    struct CaptureBatch {
        // Keep the arena alive through asynchronous copies. Destroy vectors
        // first so their allocators never refer to an already freed arena.
        std::shared_ptr<kvmem::FixedByteArena> arena;
        std::vector<llama_pos, kvmem::FixedArenaAllocator<llama_pos>> positions;
        std::vector<MeanWriteRange, kvmem::FixedArenaAllocator<MeanWriteRange>> ranges;
        CaptureBatch() = default;
        explicit CaptureBatch(std::shared_ptr<kvmem::FixedByteArena> prepared)
            : arena(std::move(prepared)), positions(kvmem::FixedArenaAllocator<llama_pos>(arena.get())),
              ranges(kvmem::FixedArenaAllocator<MeanWriteRange>(arena.get())) {}
        CaptureBatch(const CaptureBatch &) = default;
        CaptureBatch(CaptureBatch &&) noexcept = default;
        CaptureBatch & operator=(CaptureBatch other) noexcept(false) { swap(other); return *this; }
        void swap(CaptureBatch & other) noexcept {
            positions.swap(other.positions);
            ranges.swap(other.ranges);
            arena.swap(other.arena);
        }
    };
    // Positions and writer identity are captured together before model work.
    // Async workers consume their own batch, never a current directory lookup.
    std::vector<CaptureBatch> pos_queue_;
    std::shared_ptr<kvmem::FixedByteArena> capture_arena_;
    size_t capture_max_tokens_ = 0;
    size_t capture_token_capacity_ = 0;
    size_t capture_tokens_used_ = 0;
    CaptureBatch cur_capture_; // synchronous fallback only
    struct CaptureNode {
        ggml_tensor * t = nullptr;
        int il = 0;
        char which = 0;
    };
    std::vector<CaptureNode> pending_capture_;
    std::vector<CaptureNode> decode_mean_pending_k_;
    CaptureBatch decode_mean_pending_batch_;
    MeanWriteRange decode_mean_target_;
    std::vector<uint8_t> decode_mean_committed_;
    uint32_t decode_mean_n_ = 0;
    uint32_t decode_mean_block_ = ~0u;
    uint32_t decode_mean_pos0_ = 0;
    std::vector<std::vector<float>> decode_mean_host_;
    std::vector<uint8_t> decode_mean_src_; // 0 none, 1 gpu, 2 host
    struct DecodeMeanStats {
        uint32_t n_tok = 0;
        uint32_t n_flush = 0;
        uint32_t n_gpu = 0;
        uint32_t n_host = 0;
        uint32_t n_miss = 0;
        uint32_t last_block = ~0u;
        uint32_t last_n = 0;
        uint32_t last_layers = 0;
        float last_rms = 0.0f;
        bool printed = false;
    };
    DecodeMeanStats decode_mean_stats_;
    bool graph_has_q_ = false;
    bool graph_has_k_ = false;
    bool graph_has_record_ = false;
    struct CaptureD2hPipe;
    std::unique_ptr<CaptureD2hPipe> d2h_;
    struct MultiD2hPipe;
    std::unique_ptr<MultiD2hPipe> multi_d2h_;
    struct HarvestWorker {
        std::mutex mu;
        std::condition_variable cv;
        std::array<int, 2> q{};
        size_t first = 0, queued = 0;
        std::exception_ptr error;
        struct CommitStats {
            int64_t wait = 0, pack = 0, nvme = 0, commit = 0;
            uint64_t bytes = 0, syscalls = 0;
        } total, last;
        bool have_stats = false;
        std::thread th;
        bool stop = false;
        bool cancel_queued = false;
        bool ready = false;
    };
    std::unique_ptr<HarvestWorker> harvest_w_;
    HarvestPerf perf_;
    RetrPerf retr_;
    std::vector<std::vector<float>> q_sum_;
    std::vector<uint32_t> q_count_;
};

// GPU attn-cache cell count for a KVMem slot pool (budget + gen_reserve,
// clamped to n_ctx on identity). Hybrid uses this as llama_memory_hybrid's
// attn kv_size so the two halves agree.
uint32_t llama_kvmem_pool_cells(
        const llama_model & model,
        const llama_memory_params & params,
        const llama_cparams & cparams);
