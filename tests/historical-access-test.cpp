// Run manually with a local GGUF and CUDA. This binds one completed block in
// a controlled fixture; it does not enable background GC in CLI/server runs.
#include "llama-memory-kvmem-hybrid.h"
#include "llama-batch.h"
#include "llama-context.h"
#include "llama-kvmem-batch.h"
#include "llama-kvmem-hooks.h"
#include "llama-kvmem-stagein.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "kvmem/historical_kv_capture_session.hpp"
#include "kvmem/historical_kv_raw_store.hpp"
#include "kvmem/historical_kv_gc.hpp"
#include "kvmem/historical_kv_active_set.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <new>
#include <stdexcept>
#include <thread>
#include <vector>
#include <future>
#ifdef KVMEM_TEST_CUDA
#include <cuda_runtime_api.h>
#endif

// Historical graph rebuilds must not duplicate the large hparams/cparams
// value objects inside every per-microbatch input holder.
static_assert(sizeof(llm_graph_input_out_ids) < 1024, "output IDs input copied model parameters");
static_assert(sizeof(llm_graph_input_attn_kv) < 1024, "KV attention input copied model parameters");

// Only the foreground interval explicitly guarded by a test rejects C++
// allocation. Model loading, CUDA and background capture are unaffected.
static thread_local bool deny_cpp_allocation = false;
void * operator new(size_t size) {
    if (deny_cpp_allocation) throw std::bad_alloc();
    if (void * p = std::malloc(size ? size : 1)) return p;
    throw std::bad_alloc();
}
void * operator new[](size_t size) { return ::operator new(size); }
void operator delete(void * p) noexcept { std::free(p); }
void operator delete[](void * p) noexcept { ::operator delete(p); }
void operator delete(void * p, size_t) noexcept { ::operator delete(p); }
void operator delete[](void * p, size_t) noexcept { ::operator delete(p); }
struct DenyCppAllocation {
    DenyCppAllocation() { deny_cpp_allocation = true; }
    ~DenyCppAllocation() { deny_cpp_allocation = false; }
};

struct kvmem_transfer_test_access {
    struct SlotWorkspaceStatus {
        size_t bytes = 0;
        std::vector<const void *> buffers;
        std::vector<size_t> capacities;
    };
    static SlotWorkspaceStatus slot_workspace(const llama_memory_kvmem & mem) {
        const auto & ws = mem.historical_slot_workspace_;
        SlotWorkspaceStatus status;
        status.bytes = ws.capacity_bytes();
        status.buffers.push_back(ws.slots.data());
        status.capacities.push_back(ws.slots.capacity());
        for (const auto & slot : ws.slots) {
            status.buffers.push_back(slot.strm.data());
            status.buffers.push_back(slot.idxs.data());
            status.buffers.push_back(slot.idxs.empty() ? nullptr : slot.idxs[0].data());
            status.capacities.push_back(slot.strm.capacity());
            status.capacities.push_back(slot.idxs.capacity());
            status.capacities.push_back(slot.idxs.empty() ? 0 : slot.idxs[0].capacity());
        }
        return status;
    }
    static bool slot_fill_without_cpp_allocation(llama_memory_kvmem & mem) {
        auto & slots = mem.historical_slot_workspace_.slots;
        if (slots.empty() || slots[0].idxs.empty()) return false;
        const auto count = slots[0].idxs[0].capacity();
        std::vector<llama_pos> positions(count, 0);
        llama_ubatch ubatch{};
        ubatch.n_tokens = static_cast<uint32_t>(count);
        ubatch.pos = positions.data();
        DenyCppAllocation guard;
        return kvmem_fill_slot_info(mem.runtime().store(), mem.block_tokens(),
                mem.get_kv()->get_size(), ubatch, slots[0]);
    }
    struct MicrobatchStatus {
        size_t bytes = 0;
        size_t objects = 0, graph_held = 0, max_ubatches = 0;
        std::vector<const void *> buffers;
        std::vector<size_t> capacities;
    };
    static MicrobatchStatus microbatch_workspace(const llama_memory_kvmem & mem) {
        const auto & ws = mem.historical_ubatch_workspace_;
        MicrobatchStatus status;
        status.bytes = ws.vector_bytes();
        status.objects = ws.data.size();
        status.max_ubatches = ws.max_ubatches;
        status.buffers = {ws.data.data(), ws.ubatches.data(), ws.indices.data()};
        status.capacities = {ws.data.capacity(), ws.ubatches.capacity(), ws.indices.capacity()};
        for (const auto & item : ws.data) {
            const bool returned = std::any_of(ws.ubatches.begin(), ws.ubatches.end(),
                [&](const llama_ubatch & ubatch) { return ubatch.data.get() == item.get(); });
            if (item.use_count() > 1 + size_t(returned)) ++status.graph_held;
            status.buffers.insert(status.buffers.end(), {
                item.get(), item->token.data(), item->logical_pos.data(), item->pos.data(),
                item->n_seq_id.data(), item->seq_id.data(), item->seq_id_unq.data(),
                item->seq_idx.data(), item->output.data(), item->seq_id_data.data()});
            status.capacities.insert(status.capacities.end(), {
                item->token.capacity(), item->logical_pos.capacity(), item->pos.capacity(),
                item->n_seq_id.capacity(), item->seq_id.capacity(), item->seq_id_unq.capacity(),
                item->seq_idx.capacity(), item->output.capacity(), item->seq_id_data.capacity()});
        }
        return status;
    }
    static bool microbatch_split_without_cpp_allocation(llama_memory_kvmem & mem,
            const llama_vocab & vocab, llama_token * tokens) {
        auto & ws = mem.historical_ubatch_workspace_;
        llama_batch_allocr probe(1);
        auto batch = llama_batch_get_one(tokens, 16);
        if (!probe.init(batch, vocab, nullptr, 0, 1, false)) return false;
        probe.get_out_ids().reserve(16);
        if (!probe.use_prepared_simple_pool(&ws.data, &ws.indices)) return false;
        bool valid = true;
        for (int round = 0; round < 3; ++round) {
            llama_ubatch ubatch{};
            {
                DenyCppAllocation guard;
                probe.split_reset();
                ubatch = probe.split_simple(16);
            }
            valid = valid && ubatch.n_tokens == 16 && ubatch.data.get() == ws.data[0].get() &&
                    ubatch.seq_id[0] == ubatch.data->seq_id_data.data() &&
                    ubatch.data->seq_id_data.size() == 16;
        }
        probe.clear_prepared_simple_pool();
        return valid;
    }
    static bool input_batch_without_cpp_allocation(const llama_vocab & vocab, llama_token * tokens) {
        llama_batch_allocr probe(1);
        if (!probe.prepare_historical_simple_input(64)) return false;
        auto batch = llama_batch_get_one(tokens, 64);
        for (int round = 0; round < 3; ++round) {
            bool valid = false;
            {
                DenyCppAllocation guard;
                if (!probe.init(batch, vocab, nullptr, 0, 1, false)) return false;
                probe.get_out_ids().push_back(63);
                valid = probe.seq_pos_min(0) == 0 && probe.seq_pos_max(0) == 63 &&
                        probe.get_n_tokens() == 64 && probe.get_n_outputs() == 1 &&
                        probe.get_out_ids().size() == 1;
            }
            if (!valid) return false;
        }
        probe.clear_historical_simple_input();
        return !probe.historical_simple_input_prepared();
    }
    static bool microbatch_graph_bound_without_cpp_allocation(llama_memory_kvmem & mem,
            const llama_vocab & vocab, llama_token * tokens) {
        auto & ws = mem.historical_ubatch_workspace_;
        if (ws.data.size() != ws.max_ubatches + ws.graph_retained_max) return false;
        std::vector<std::shared_ptr<llama_ubatch::data_t>> held;
        held.reserve(ws.graph_retained_max);
        for (size_t i = 0; i < ws.graph_retained_max; ++i) held.push_back(ws.data[i]);
        llama_batch_allocr probe(1);
        auto batch = llama_batch_get_one(tokens, 64);
        if (!probe.init(batch, vocab, nullptr, 0, 1, false)) return false;
        probe.get_out_ids().reserve(64);
        if (!probe.use_prepared_simple_pool(&ws.data, &ws.indices)) return false;
        std::vector<llama_ubatch> ubatches;
        ubatches.reserve(ws.max_ubatches);
        {
            DenyCppAllocation guard;
            probe.split_reset();
            for (size_t i = 0; i < ws.max_ubatches; ++i)
                ubatches.push_back(probe.split_simple(16));
        }
        probe.clear_prepared_simple_pool();
        for (size_t i = 0; i < ws.max_ubatches; ++i)
            if (ubatches[i].n_tokens != 16 ||
                ubatches[i].data.get() != ws.data[ws.graph_retained_max + i].get()) return false;
        return true;
    }
    static std::array<const void *, 16> execution_buffers(const llama_memory_kvmem & mem) {
        const auto & a = mem.historical_active_workspace_;
        const auto & c = mem.cell_workspace_;
        return {mem.historical_active_.get(), mem.historical_active_->handles().data(),
                a.requested.data(), a.requested_ids.data(), a.remove_ids.data(), a.seen.data(),
                c.slots.strm.data(), c.slots.idxs.data(), c.slots.idxs[0].data(),
                c.positions.data(), c.logical.data(), c.n_seq_id.data(), c.seq_data.data(),
                c.seq_id.data(), c.output.data(), c.tokens.data()};
    }
    static size_t execution_bytes(const llama_memory_kvmem & mem) {
        return mem.historical_active_workspace_.capacity_bytes() + mem.cell_workspace_.capacity_bytes() +
            (mem.historical_active_ ? mem.historical_active_->capacity_bytes() : 0);
    }
    static bool execution_storage_empty(const llama_memory_kvmem & mem) {
        return !mem.historical_active_ && !execution_bytes(mem);
    }
    struct BatchContextStatus {
        const void * storage = nullptr;
        size_t bytes = 0;
        bool in_use = false;
    };
    static BatchContextStatus batch_context(const llama_memory_kvmem & mem) {
        return {mem.historical_batch_context_storage_,
                mem.historical_batch_context_storage_bytes_,
                mem.historical_batch_context_in_use_};
    }
    static size_t active_count(const llama_memory_kvmem & mem) {
        return mem.historical_active_->handles().size();
    }
    static bool layout(llama_memory_kvmem & mem) { return mem.layout_gpu_slots_by_orig_pos(); }
    static std::array<const void *, 10> layout_buffers(const llama_memory_kvmem & mem) {
        const auto & w = mem.layout_workspace_;
        return {w.items.data(), w.mtp_res.data(), w.seen.data(), w.res_ix.data(), w.gbytes.data(),
                w.sbytes.data(), w.gsrc.data(), w.ssrc.data(), w.gdst.data(), w.sdst.data()};
    }
    static void swap_layout_slots(llama_memory_kvmem & mem, uint32_t a, uint32_t b) {
        auto & store = mem.runtime().store();
        const auto first = store.blocks()[a], second = store.blocks()[b];
        for (auto il : mem.raw().config().historical_gc_attention_layers) {
            for (auto * tensor : {mem.get_kv()->get_k_storage(il), mem.get_kv()->get_v_storage(il)}) {
                const size_t bytes = size_t(mem.block_tokens()) * tensor->nb[1];
                std::vector<uint8_t> x(bytes), y(bytes);
                const size_t off_a = size_t(first.gpu_slot) * bytes, off_b = size_t(second.gpu_slot) * bytes;
                ggml_backend_tensor_get(tensor, x.data(), off_a, bytes);
                ggml_backend_tensor_get(tensor, y.data(), off_b, bytes);
                ggml_backend_tensor_set(tensor, y.data(), off_a, bytes);
                ggml_backend_tensor_set(tensor, x.data(), off_b, bytes);
            }
        }
        mem.get_kv()->seq_rm_logical(0, first.orig_pos_start, first.orig_pos_end());
        mem.get_kv()->seq_rm_logical(0, second.orig_pos_start, second.orig_pos_end());
        store.set_block_gpu_slot(a, second.gpu_slot);
        store.set_block_gpu_slot(b, first.gpu_slot);
        {
            DenyCppAllocation guard;
            mem.occupy_block_cells(a); mem.occupy_block_cells(b);
        }
#ifdef KVMEM_TEST_CUDA
        if (cudaDeviceSynchronize() != cudaSuccess) throw std::runtime_error("layout fixture source completion");
#endif
    }
    struct PackedWorkspace {
        const void * a;
        const void * b;
        const void * fallback;
        size_t capacity, fallback_bytes;
    };
    static PackedWorkspace packed_workspace(const llama_memory_kvmem & mem) {
        return {mem.harvest_v_jobs_.data(), mem.harvest_v_pending_.jobs.data(), mem.harvest_v_fallback_.data(),
            mem.harvest_v_jobs_.capacity() + mem.harvest_v_pending_.jobs.capacity(), mem.harvest_v_fallback_.size()};
    }
    static void small_packed_queue(llama_memory_kvmem & mem) { mem.harvest_v_job_limit_ = 8; }
    static bool packed_workspace_unchanged(const llama_memory_kvmem & mem, PackedWorkspace p) {
        const auto q = packed_workspace(mem);
        return q.capacity == p.capacity && q.fallback_bytes == p.fallback_bytes && q.fallback == p.fallback &&
            ((q.a == p.a && q.b == p.b) || (q.a == p.b && q.b == p.a));
    }
    using CaptureFaultPoint = llama_memory_kvmem::CaptureFaultPoint;
    using CaptureFault = llama_memory_kvmem::CaptureFault;
    static void capture_hook(llama_memory_kvmem & mem, CaptureFault hook, void * context) {
        mem.harvest_worker_stop();
        mem.capture_fault_ = hook;
        mem.capture_fault_context_ = context;
    }
    static void capture_init(llama_memory_kvmem & mem) {
        if (!mem.d2h_init()) throw std::runtime_error("capture initialization failed");
        mem.harvest_worker_stop();
    }
    static void capture_start(llama_memory_kvmem & mem) { mem.harvest_worker_start(); }
    static bool capture_submit(llama_memory_kvmem & mem) { return mem.d2h_submit(nullptr); }
    static void capture_shutdown(llama_memory_kvmem & mem) { mem.harvest_shutdown(); }
    static void capture_wait_first(llama_memory_kvmem & mem) { mem.harvest_wait_slot(0); }
    static bool capture_domain_failed(const llama_memory_kvmem & mem) { return mem.capture_failed_; }
    static size_t capture_queued(llama_memory_kvmem & mem) {
        std::lock_guard<std::mutex> lock(mem.harvest_w_->mu);
        return mem.harvest_w_->queued;
    }
    static size_t capture_positions(const llama_memory_kvmem & mem) { return mem.pos_queue_.size(); }
    static void cancel_capture_inputs(llama_memory_kvmem & mem) {
        mem.pos_queue_.clear();
        mem.pending_capture_.clear();
    }
    static uint32_t query_dim(const llama_memory_kvmem & mem) { return mem.n_head_ * mem.n_embd_head_; }
    static void query_nodes(llama_memory_kvmem & mem, const std::vector<ggml_tensor *> & tensors) {
        const auto & layers = mem.raw().config().historical_gc_attention_layers;
        mem.query_begin_ = 0;
        mem.query_end_ = 32;
        for (size_t i = 0; i < layers.size(); ++i) mem.register_capture(tensors[i], layers[i], 'q');
    }
    static bool query_matches(llama_memory_kvmem & mem, uint32_t count, const std::vector<float> & sum) {
        for (uint32_t il : mem.raw().config().historical_gc_attention_layers)
            if (mem.q_count_[il] != count || mem.q_sum_[il] != sum) return false;
        return true;
    }
    static bool capture_failed(const llama_memory_kvmem & mem) { return mem.historical_capture_->failed(); }
    static bool append_after_failure(llama_memory_kvmem & mem) { return mem.prepare_working_set(1); }
    static uint64_t source_readers(const llama_memory_kvmem & mem) {
        uint64_t total = 0;
        for (auto count : mem.historical_gpu_readers_) total += count;
        return total;
    }
    static kvmem::HistoricalKvHandle pending_writer(const llama_memory_kvmem & mem) {
        for (const auto * jobs : {&mem.harvest_v_jobs_, &mem.harvest_v_pending_.jobs})
            for (const auto & job : *jobs)
                if (job.writer && !job.committed) return job.writer->handle();
        throw std::runtime_error("failed batch retained no captured writer");
    }
    static bool pending_resources_held(const llama_memory_kvmem & mem) {
        bool observed = false;
        for (const auto * jobs : {&mem.harvest_v_jobs_, &mem.harvest_v_pending_.jobs}) {
            for (const auto & job : *jobs) {
                if (!job.writer || job.committed) continue;
                observed = true;
                if (job.historical_gpu_slot < 0 || !mem.historical_gpu_readers_[job.historical_gpu_slot] ||
                        std::find(mem.free_slots_.begin(), mem.free_slots_.end(), job.historical_gpu_slot) != mem.free_slots_.end() ||
                        !job.writer->storage()->write_progress(job.writer->writer(), job.writer->handle(), job.il))
                    return false;
            }
        }
        return observed;
    }
    static void install_gc_policy(llama_memory_kvmem & mem,
                                  std::unique_ptr<kvmem::HistoricalKvPolicy> policy) {
        const auto repository = mem.historical_capture_->repository();
        kvmem::HistoricalKvGc::Config cfg;
        cfg.mode = kvmem::HistoricalKvGc::Mode::Delete;
        cfg.scan_blocks = cfg.reclaim_blocks = repository->capacity();
        cfg.batch_blocks = 1;
        cfg.poll_ms = 1;
        mem.historical_gc_ = std::make_unique<kvmem::HistoricalKvGc>(repository, std::move(policy), cfg);
        mem.historical_gc_mode_ = 2;
        mem.historical_gc_->start();
    }
    static void stop_gc(llama_memory_kvmem & mem) { mem.historical_gc_->stop(); }
    static void on_submission(llama_memory_kvmem & mem, std::function<void()> hook) {
        mem.historical_submission_test_hook_ = std::move(hook);
    }
    static bool capture_copy_without_cpp_allocation(llama_memory_kvmem & mem) {
        if (!mem.capture_arena_ || mem.pos_queue_.empty() ||
                mem.pos_queue_.front().positions.empty()) return false;
        const auto before = mem.capture_arena_->used_bytes();
        bool copied = false;
        {
            DenyCppAllocation guard;
            const auto clone = mem.pos_queue_.front();
            copied = clone.positions.size() == mem.pos_queue_.front().positions.size() &&
                    clone.ranges.size() == mem.pos_queue_.front().ranges.size() &&
                    clone.arena == mem.capture_arena_;
        }
        return copied && mem.capture_arena_->used_bytes() > before &&
                mem.capture_arena_->used_bytes() <= mem.capture_arena_->capacity_bytes();
    }
    static bool capture_arena_bounded(const llama_memory_kvmem & mem, size_t tokens) {
        return mem.capture_arena_ && mem.capture_max_tokens_ == tokens &&
               mem.capture_tokens_used_ <= tokens &&
               mem.capture_arena_->used_bytes() <= mem.capture_arena_->capacity_bytes() &&
               mem.pos_queue_.capacity() >= tokens;
    }
    static bool capture_arena_unpublished(const llama_memory_kvmem & mem) {
        return !mem.capture_arena_ && mem.pos_queue_.empty();
    }
    static bool capture_tokens_complete(const llama_memory_kvmem & mem, size_t tokens) {
        return mem.capture_tokens_used_ == tokens;
    }
    static std::vector<kvmem::HistoricalKvHandle> active_handles(const llama_memory_kvmem & mem) {
        return mem.historical_active_ ? mem.historical_active_->handles() :
                                        std::vector<kvmem::HistoricalKvHandle>{};
    }
    static uint64_t deferred_slots(const llama_memory_kvmem & mem) {
        return mem.historical_gpu_deferred_;
    }
    static uint64_t returned_slots(const llama_memory_kvmem & mem) {
        return mem.historical_gpu_returned_;
    }
    static bool retired_source_held(const llama_memory_kvmem & mem, int32_t slot) {
        return slot >= 0 && size_t(slot) < mem.historical_gpu_readers_.size() &&
            mem.historical_gpu_readers_[slot] && mem.historical_gpu_retired_[slot] &&
            std::find(mem.free_slots_.begin(), mem.free_slots_.end(), slot) == mem.free_slots_.end();
    }
    static uint32_t submitted_end(const llama_memory_kvmem & mem) {
        return mem.pos_queue_.empty() || mem.pos_queue_.front().positions.empty() ? 0 :
            uint32_t(mem.pos_queue_.front().positions.back()) + 1;
    }
    static void restore(llama_memory_kvmem & mem, uint32_t id) {
        mem.write_block_to_gpu(id);
    }
    static void capture(llama_memory_kvmem & mem, uint32_t id) { mem.harvest_gpu_v(id); }
    static void capture_current(llama_memory_kvmem & mem) {
        if (mem.pos_queue_.empty()) throw std::runtime_error("fixture has no admitted capture batch");
        mem.harvest_captured_packed(mem.pos_queue_.front(), mem.pos_queue_.front().positions.size());
    }
    static void place(llama_memory_kvmem & mem, uint32_t id) {
        const auto slot = mem.alloc_slot();
        if (slot < 0) throw std::runtime_error("fixture has no GPU slot");
        mem.runtime().store().set_block_gpu_slot(id, slot);
    }
    static void commit(llama_memory_kvmem & mem) { mem.harvest_gpu_v_commit(); }
    static size_t pending(const llama_memory_kvmem & mem) {
        return mem.harvest_v_jobs_.size() + mem.harvest_v_pending_.jobs.size();
    }
    static std::shared_ptr<kvmem::HistoricalKvRepository> capture_repository(const llama_memory_kvmem & mem) {
        return mem.historical_capture_ ? mem.historical_capture_->repository() : nullptr;
    }
    static std::shared_ptr<kvmem::HistoricalKvRawStore> capture_storage(const llama_memory_kvmem & mem) {
        return mem.historical_capture_ ? mem.historical_capture_->storage() : nullptr;
    }
    static void mean_inputs(llama_memory_kvmem & mem, const std::vector<ggml_tensor *> & tensors,
                            uint32_t begin, uint32_t count) {
        mem.pos_queue_.clear();
        std::vector<llama_pos> positions(count);
        for (uint32_t i = 0; i < count; ++i) positions[i] = begin + i;
        mem.note_ubatch_pos(positions);
        mem.pending_capture_.clear();
        const auto & layers = mem.raw().config().historical_gc_attention_layers;
        for (size_t i = 0; i < layers.size(); ++i) mem.register_capture(tensors[i], layers[i], 'k');
    }
    static void mean_positions(llama_memory_kvmem & mem, const std::vector<llama_pos> & positions) {
        mem.pos_queue_.clear();
        mem.note_ubatch_pos(positions);
    }
    static void mean_nodes(llama_memory_kvmem & mem, const std::vector<ggml_tensor *> & tensors) {
        mem.pending_capture_.clear();
        const auto & layers = mem.raw().config().historical_gc_attention_layers;
        for (size_t i = 0; i < layers.size(); ++i) mem.register_capture(tensors[i], layers[i], 'k');
    }
    static bool mean_submit(llama_memory_kvmem & mem) {
        // Deterministic barrier: queue the real DMA but do not let its worker
        // commit before the test has retired/unbound the original generation.
        if (!mem.d2h_init()) return false;
        mem.harvest_worker_stop();
        return mem.d2h_submit(nullptr);
    }
    static void mean_sync(llama_memory_kvmem & mem) {
        mem.cur_capture_ = std::move(mem.pos_queue_.front());
        mem.pos_queue_.erase(mem.pos_queue_.begin());
        for (const auto & node : mem.pending_capture_) mem.harvest_capture(node.t, node.il, node.which);
        mem.cur_capture_ = {};
    }
    static void mean_decode(llama_memory_kvmem & mem, uint32_t accepted) {
        mem.decode_mean_ingest(nullptr);
        mem.decode_mean_commit(accepted);
    }
    static void clear_mean_nodes(llama_memory_kvmem & mem) { mem.pending_capture_.clear(); }
    static uint32_t decode_tokens(const llama_memory_kvmem & mem) { return mem.decode_mean_n_; }
    static void decode_ingest(llama_memory_kvmem & mem) { mem.decode_mean_ingest(nullptr); }
    static size_t decode_pending(const llama_memory_kvmem & mem) { return mem.decode_mean_pending_batch_.positions.size(); }
    static llama_pos decode_pending_begin(const llama_memory_kvmem & mem) {
        return mem.decode_mean_pending_batch_.positions.empty() ? -1 : mem.decode_mean_pending_batch_.positions.front();
    }
    static void decode_shutdown(llama_memory_kvmem & mem) {
        mem.harvest_shutdown();
        mem.decode_mean_shutdown();
    }
    static void score(llama_memory_kvmem & mem) {
        mem.reset_query_acc();
        for (uint32_t il : mem.raw().config().historical_gc_attention_layers) {
            std::fill(mem.q_sum_[il].begin(), mem.q_sum_[il].end(), 1.0f);
            mem.q_count_[il] = 1;
        }
        mem.score_retrieval();
    }
};

static void require(bool ok, const char * message) {
    if (!ok) throw std::runtime_error(message);
}

// Retirement in the controlled fixture is explicit; the real GC worker must
// still drain retired slots without any further foreground directory action.
class ObserveOnlyPolicy final : public kvmem::HistoricalKvPolicy {
    kvmem::HistoricalKvStepResult step(const kvmem::HistoricalKvSnapshot &,
            const kvmem::HistoricalKvPressure &, kvmem::HistoricalKvWorkBudget,
            kvmem::HistoricalKvDecisionBuffer &) override {
        return {kvmem::HistoricalKvStepStatus::NoCandidate, 0, 0};
    }
};

static std::vector<uint8_t> pattern(size_t size, unsigned seed) {
    std::vector<uint8_t> bytes(size);
    for (auto & byte : bytes) {
        seed = seed * 1664525u + 1013904223u;
        byte = uint8_t(seed >> 24);
    }
    return bytes;
}

// Known pre-RoPE tensors in the actual model's layer/head dimensions. These
// exercise the production capture/conversion paths, not model-generated K.
struct MeanTensors {
    std::unique_ptr<ggml_context, decltype(&ggml_free)> context{nullptr, ggml_free};
    std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer{
        nullptr, ggml_backend_buffer_free};
    std::vector<ggml_tensor *> layers;

    MeanTensors(llama_memory_kvmem & mem, uint32_t begin, uint32_t count, bool host, ggml_type type,
                uint32_t width = 0) {
        const auto & cfg = mem.raw().config();
        if (!width) width = cfg.n_embd_k;
        context.reset(ggml_init({cfg.historical_gc_attention_layers.size() * ggml_tensor_overhead() + 1024,
                                 nullptr, true}));
        require(bool(context), "Mean-K tensor context allocation failed");
        for (uint32_t il : cfg.historical_gc_attention_layers) {
            (void) il;
            layers.push_back(ggml_new_tensor_3d(context.get(), type, width, 1, count));
        }
        auto buft = host ? ggml_backend_cpu_buffer_type() : ggml_backend_buffer_get_type(
            mem.get_kv()->get_k_storage(cfg.historical_gc_attention_layers.front())->buffer);
        buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(context.get(), buft));
        require(bool(buffer), "Mean-K tensor buffer allocation failed");
        for (size_t i = 0; i < layers.size(); ++i) {
            std::vector<float> values(static_cast<size_t>(count) * width);
            for (uint32_t t = 0; t < count; ++t)
                for (uint32_t d = 0; d < width; ++d)
                    values[t * width + d] = float(begin + t + 1) + float(d % 8) / 8;
            if (type == GGML_TYPE_F16) {
                std::vector<ggml_fp16_t> half(values.size());
                ggml_fp32_to_fp16_row(values.data(), half.data(), values.size());
                ggml_backend_tensor_set(layers[i], half.data(), 0, half.size() * sizeof(half[0]));
            } else {
                ggml_backend_tensor_set(layers[i], values.data(), 0, values.size() * sizeof(float));
            }
        }
    }
};

static std::vector<float> mean_sum(uint32_t begin, uint32_t count, size_t dim) {
    std::vector<float> result(dim);
    for (size_t d = 0; d < dim; ++d)
        result[d] = float(count) * (float(begin) + float(count + 1) / 2 + float(d % 8) / 8);
    return result;
}

using CaptureFaultPoint = kvmem_transfer_test_access::CaptureFaultPoint;

struct CaptureFailure {
    CaptureFaultPoint point;
    size_t index = 0;
    bool gate = false;
    bool fail_drain = false;
    bool ordered_fallback = false;
    std::atomic<bool> entered{false}, open{false};
    std::atomic<bool> fallback_entered{false};
    static bool hook(void * context, CaptureFaultPoint point, int slot, size_t index) noexcept {
        auto & self = *static_cast<CaptureFailure *>(context);
        if (point == CaptureFaultPoint::Drain) return self.fail_drain;
        if (self.ordered_fallback && point == CaptureFaultPoint::CompletionEvent && slot == 1) {
            self.fallback_entered = true;
            return true;
        }
        if (point != self.point || index != self.index || self.entered.exchange(true)) return false;
        while (self.gate && !self.open.load()) std::this_thread::yield();
        return !self.ordered_fallback;
    }
    void wait_entered() const {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!entered.load() && std::chrono::steady_clock::now() < deadline)
            std::this_thread::yield();
        require(entered.load(), "capture worker did not reach deterministic failure gate");
    }
};

// Install/remove hooks only while the worker is joined. On test exceptions,
// release the gate and settle real device work before tensor buffers unwind.
struct CaptureFailureGuard {
    llama_memory_kvmem & mem;
    CaptureFailure & fault;
    CaptureFailureGuard(llama_memory_kvmem & mem, CaptureFailure & fault) : mem(mem), fault(fault) {
        kvmem_transfer_test_access::capture_init(mem);
        kvmem_transfer_test_access::capture_hook(mem, &CaptureFailure::hook, &fault);
    }
    ~CaptureFailureGuard() {
        fault.open = true;
        kvmem_transfer_test_access::capture_hook(mem, nullptr, nullptr);
        kvmem_transfer_test_access::capture_shutdown(mem);
        kvmem_transfer_test_access::cancel_capture_inputs(mem);
    }
};

static void expect_capture_error(llama_memory_kvmem & mem, bool wait_slot = false) {
    // Only this task accesses foreground-owned state until get(). The deadline
    // is a watchdog, not the mechanism used to place concurrent operations.
    auto result = std::async(std::launch::async, [&] {
        try {
            if (wait_slot) kvmem_transfer_test_access::capture_wait_first(mem);
            else mem.harvest_flush();
        } catch (const std::exception &) { return true; }
        return false;
    });
    if (result.wait_for(std::chrono::seconds(5)) != std::future_status::ready) {
        std::fprintf(stderr, "FAIL capture error did not wake foreground within watchdog\n");
        std::abort(); // Do not let std::future's destructor mask a deadlock.
    }
    require(result.get(), "capture failure was reported as successful completion");
    require(kvmem_transfer_test_access::capture_domain_failed(mem), "failed capture admission remained open");
    require(!kvmem_transfer_test_access::append_after_failure(mem), "failed capture accepted another append");
    llama_pos pos = 0;
    require(!mem.begin_active_ubatch(1, &pos), "failed capture admitted another active");
}

static void check_capture_start_failure(llama_memory_kvmem & mem) {
    CaptureFailure fault{CaptureFaultPoint::DeviceSelect};
    CaptureFailureGuard guard(mem, fault);
    kvmem_transfer_test_access::capture_start(mem);
    expect_capture_error(mem);
    require(fault.entered, "empty-queue flush skipped worker initialization failure");
    std::puts("PASS capture worker startup: empty-queue flush propagates device selection failure");
}

struct DecodeFailure {
    KvmemMeanKPoint point;
    uint32_t layer = 0;
    bool fired = false, deny_drain = false;
    static bool inject(void * context, KvmemMeanKPoint point, uint32_t layer) noexcept {
        auto & f = *static_cast<DecodeFailure *>(context);
        if (point == KvmemMeanKPoint::Drain) return f.deny_drain;
        if (!f.fired && point == f.point && layer == f.layer) { f.fired = true; return true; }
        return false;
    }
};
struct DecodeFailureGuard {
    llama_memory_kvmem & mem;
    ~DecodeFailureGuard() {
        kvmem_meank_test_fault(nullptr, nullptr);
        kvmem_transfer_test_access::capture_hook(mem, nullptr, nullptr);
        kvmem_transfer_test_access::decode_shutdown(mem);
        kvmem_transfer_test_access::commit(mem);
        kvmem_transfer_test_access::cancel_capture_inputs(mem);
    }
};

static void check_decode_failure(llama_memory_kvmem & mem, unsigned scenario,
        const std::shared_ptr<kvmem::HistoricalKvRepository> & repository,
        const std::shared_ptr<kvmem::HistoricalKvRawStore> & payload,
        kvmem::HistoricalKvShadowDirectory::Producer & producer, kvmem::HistoricalKvHandle handle,
        const MeanTensors & first, const MeanTensors & second) {
    using Repository = kvmem::HistoricalKvRepository;
    const auto & cfg = payload->config();
    std::weak_ptr<kvmem::HistoricalKvShadowDirectory::WriterContext> weak = producer;
    DecodeFailure fault{scenario == 21 ? KvmemMeanKPoint::AddResult :
                        scenario == 23 ? KvmemMeanKPoint::ReadCopy : KvmemMeanKPoint::Wait};
    fault.layer = (scenario == 21 || scenario == 23) ? cfg.attention_layers[1] : 0;
    fault.deny_drain = scenario == 22 || scenario == 24;
    CaptureFailure host_fault{CaptureFaultPoint::DecodeHostCommit};
    host_fault.index = cfg.attention_layers[1];
    DecodeFailureGuard guard{mem};
    kvmem_transfer_test_access::mean_inputs(mem, first.layers, 0, 8);
    kvmem_transfer_test_access::decode_ingest(mem);
    const auto cell = mem.runtime().store().blocks()[0].gpu_slot * mem.block_tokens();
    for (uint32_t il : cfg.attention_layers) {
        const auto k = pattern(mem.block_tokens() * cfg.packed_k_bytes_per_token, il + 900);
        const auto v = pattern(mem.block_tokens() * cfg.packed_v_bytes_per_token, il + 1000);
        ggml_backend_tensor_set(mem.get_kv()->get_k_storage(il), k.data(), cell * cfg.packed_k_bytes_per_token, k.size());
        ggml_backend_tensor_set(mem.get_kv()->get_v_storage(il), v.data(), cell * cfg.packed_v_bytes_per_token, v.size());
    }
    if (scenario == 28) {
        mem.decode_mean_commit(0);
        require(kvmem_transfer_test_access::decode_pending(mem) == 0 &&
                kvmem_transfer_test_access::decode_tokens(mem) == 0 &&
                kvmem_transfer_test_access::pending(mem) == 0, "rejected draft created historical output");
        kvmem_transfer_test_access::mean_inputs(mem, first.layers, 0, 8);
        kvmem_transfer_test_access::mean_decode(mem, 4);
        kvmem_transfer_test_access::commit(mem);
        require(kvmem_transfer_test_access::decode_tokens(mem) == 4, "accepted draft did not retain exact prefix");
        producer.reset();
        require(repository->retire(handle) == Repository::RetireResult::Committed &&
                repository->try_reclaim(handle) == Repository::ReclaimResult::Deferred,
                "accepted decode prefix lost its producer before flush");
        mem.decode_mean_flush();
        require(weak.expired() && repository->inspect(handle)->life == kvmem::HistoricalKvLife::Retired &&
                repository->try_reclaim(handle) == Repository::ReclaimResult::Complete,
                "old accepted decode prefix revived retired history or prevented release");
        std::puts("PASS decode accept/reject: rejected draft wrote nothing; accepted four-token prefix flushed after retirement without resurrection");
        return;
    }
    bool threw = false;
    try {
        if (scenario == 26) {
            kvmem_transfer_test_access::mean_inputs(mem, second.layers, 8, 8);
            kvmem_transfer_test_access::decode_ingest(mem);
        } else {
            if (scenario == 27 || scenario == 29) {
                // Re-ingest a malformed layer list only after explicitly
                // rejecting the prior draft, so this tests layer validation.
                mem.decode_mean_discard();
                auto incomplete = first.layers;
                incomplete.pop_back();
                kvmem_transfer_test_access::mean_inputs(mem, first.layers, 0, 8);
                kvmem_transfer_test_access::clear_mean_nodes(mem);
                for (size_t i = 0; scenario != 29 && i < incomplete.size(); ++i)
                    mem.register_capture(incomplete[i], cfg.attention_layers[i], 'k');
                kvmem_transfer_test_access::decode_ingest(mem);
            }
            if (scenario == 21 || scenario == 22) kvmem_meank_test_fault(DecodeFailure::inject, &fault);
            mem.decode_mean_commit(4);
            if (scenario == 23 || scenario == 24) kvmem_meank_test_fault(DecodeFailure::inject, &fault);
            if (scenario == 25) kvmem_transfer_test_access::capture_hook(mem, CaptureFailure::hook, &host_fault);
            mem.decode_mean_flush();
        }
    } catch (const std::exception &) { threw = true; }
    require(threw && kvmem_transfer_test_access::capture_domain_failed(mem), "decode error left request admission open");
    if (scenario <= 24) require(fault.fired, "decode CUDA fault not reached");
    if (scenario == 25) require(host_fault.entered, "decode host write fault not reached");
    if (scenario == 26) require(kvmem_transfer_test_access::decode_pending_begin(mem) == 0,
                              "rejected second draft overwrote the original captured identity");
    require(!kvmem_transfer_test_access::append_after_failure(mem), "failed decode accepted another append");
    for (unsigned action = 0; action < 3; ++action) {
        bool rejected = false;
        try {
            if (action == 0) mem.decode_mean_commit(4);
            if (action == 1) mem.decode_mean_discard();
            if (action == 2) mem.decode_mean_flush();
        } catch (const std::exception &) { rejected = true; }
        require(rejected, "failed decode replayed or discarded an unconfirmed task");
    }
    producer.reset();
    require(repository->retire(handle) == Repository::RetireResult::Committed, "failed decode writer retirement failed");
    kvmem_transfer_test_access::commit(mem); // settle packed tasks; Mean-K alone must retain the writer
    require(repository->try_reclaim(handle) == Repository::ReclaimResult::Deferred,
            "decode failure lost its CPU owner before cancellation");
    {
        auto held = weak.lock();
        require(bool(held), "failed decode did not retain its original producer");
        for (size_t i = 0; i < cfg.attention_layers.size(); ++i) {
            const auto p = payload->write_progress(held->writer(), handle, cfg.attention_layers[i]);
            const uint32_t expected = (scenario == 23 || scenario == 25) && i == 0 ? 4 : 0;
            require(p && p->mean_k_tokens == expected, "failed decode repeated or published an incomplete sum");
        }
    }
    if (fault.deny_drain) {
        bool rejected = false;
        try { kvmem_transfer_test_access::decode_shutdown(mem); }
        catch (const std::exception &) { rejected = true; }
        require(rejected && !weak.expired() && repository->try_reclaim(handle) == Repository::ReclaimResult::Deferred,
                "unconfirmed decode cancellation released its writer");
    }
    kvmem_meank_test_fault(nullptr, nullptr);
    kvmem_transfer_test_access::capture_hook(mem, nullptr, nullptr);
    kvmem_transfer_test_access::decode_shutdown(mem);
    kvmem_transfer_test_access::cancel_capture_inputs(mem);
    require(weak.expired() && repository->inspect(handle)->life == kvmem::HistoricalKvLife::Retired &&
            repository->try_reclaim(handle) == Repository::ReclaimResult::Complete && payload->allocated_payload_bytes() == 0,
            "confirmed decode cancellation revived history or leaked CPU payload");
    std::printf("PASS decode failure scenario=%u: closed admission, exact partial writes, retained producer, checked cancellation, CPU release\n", scenario);
}

static void check_capture_failure(llama_memory_kvmem & mem, unsigned scenario) {
    using Repository = kvmem::HistoricalKvRepository;
    using Payload = kvmem::HistoricalKvRawStore;
    const bool worker_error = scenario == 14 || scenario == 15;
    const bool host_error = scenario == 15;
    const bool unconfirmed = scenario == 18;
    const bool ordered_fallback = scenario == 20;
    const auto & legacy_cfg = mem.raw().config();
    const uint32_t block = mem.block_tokens();
    std::vector<llama_pos> positions(block);
    for (uint32_t i = 0; i < block; ++i) positions[i] = i;
    llama_ubatch ub{};
    ub.n_tokens = block;
    ub.n_pos = 1;
    ub.pos = ub.logical_pos = positions.data();
    llama_kv_cache::slot_info_vec_t slots;
    require(mem.prepare_ubatches({ub}, block, slots), "capture failure fixture slot preparation failed");
    mem.occupy_in(mem.get_kv(), 0);
    kvmem::HistoricalKvRawStoreConfig cfg;
    cfg.n_layer = legacy_cfg.n_layer;
    cfg.max_tokens_per_slot = block;
    cfg.packed_k_bytes_per_token = legacy_cfg.k_gpu_row_bytes;
    cfg.packed_v_bytes_per_token = legacy_cfg.v_gpu_row_bytes;
    cfg.mean_k_elements = legacy_cfg.n_embd_k;
    cfg.attention_layers = legacy_cfg.historical_gc_attention_layers;
    require(cfg.attention_layers.size() >= 2, "capture fault fixture requires two attention layers");
    auto payload = std::make_shared<Payload>(cfg, 1);
    auto repository = std::make_shared<Repository>(1);
    std::weak_ptr<Repository> weak_repository = repository;
    auto writer = repository->begin_write(0, 0, block, 1, 0,
        [payload, weak_repository](const kvmem::HistoricalKvHandle & h) {
            const auto repo = weak_repository.lock();
            return repo && payload->release(*repo, h).code == Payload::ReleaseCode::Released;
        });
    require(writer.has_value(), "capture failure fixture writer failed");
    const auto handle = writer->handle();
    require(payload->reserve(*writer, handle, 0, block), "capture failure fixture reserve failed");
    kvmem::HistoricalKvShadowDirectory::Producer producer;
    require(mem.runtime().bind_historical_full_block(0, repository, handle, *writer, producer, payload),
            "capture failure fixture binding failed");
    std::weak_ptr<kvmem::HistoricalKvShadowDirectory::WriterContext> weak_producer = producer;
    const uint32_t qdim = kvmem_transfer_test_access::query_dim(mem);
    MeanTensors first(mem, 0, 8, false, GGML_TYPE_F32);
    MeanTensors second(mem, 8, 8, false, GGML_TYPE_F32);
    if (scenario >= 21) {
        check_decode_failure(mem, scenario, repository, payload, producer, handle, first, second);
        return;
    }
    MeanTensors first_q(mem, 0, 8, false, GGML_TYPE_F32, qdim);
    MeanTensors second_q(mem, 8, 8, false, GGML_TYPE_F32, qdim);
    CaptureFailure fault{scenario == 14 || ordered_fallback ? CaptureFaultPoint::EventWait :
                        host_error ? CaptureFaultPoint::HostCommit :
                        scenario == 16 ? CaptureFaultPoint::SnapshotCopy :
                        scenario == 19 ? CaptureFaultPoint::HostCopy : CaptureFaultPoint::CompletionEvent};
    fault.index = (host_error || scenario == 16) ? 1 : 0;
    fault.gate = worker_error || ordered_fallback;
    fault.fail_drain = unconfirmed;
    fault.ordered_fallback = ordered_fallback;
    CaptureFailureGuard guard(mem, fault);
    kvmem_transfer_test_access::mean_inputs(mem, first.layers, 0, 8);
    kvmem_transfer_test_access::query_nodes(mem, first_q.layers);
    if (ordered_fallback) {
        // The second batch enters the full production harvest entry, which
        // also captures packed KV. Supply its first executed prefix too.
        const auto cell = mem.runtime().store().blocks()[0].gpu_slot * block;
        for (uint32_t il : cfg.attention_layers) {
            const auto k = pattern(block * cfg.packed_k_bytes_per_token, il + 700);
            const auto v = pattern(block * cfg.packed_v_bytes_per_token, il + 800);
            ggml_backend_tensor_set(mem.get_kv()->get_k_storage(il), k.data(),
                                    cell * cfg.packed_k_bytes_per_token, k.size());
            ggml_backend_tensor_set(mem.get_kv()->get_v_storage(il), v.data(),
                                    cell * cfg.packed_v_bytes_per_token, v.size());
        }
        kvmem_transfer_test_access::capture_current(mem);
        kvmem_transfer_test_access::commit(mem);
        kvmem_transfer_test_access::capture_start(mem);
        require(kvmem_transfer_test_access::capture_submit(mem), "ordered first batch not submitted");
        fault.wait_entered();
        kvmem_transfer_test_access::mean_inputs(mem, second.layers, 8, 8);
        kvmem_transfer_test_access::query_nodes(mem, second_q.layers);
        auto fallback = std::async(std::launch::async, [&] { mem.harvest_pending(nullptr); });
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!fault.fallback_entered && std::chrono::steady_clock::now() < deadline)
            std::this_thread::yield();
        const bool reached = fault.fallback_entered;
        const bool waited = fallback.wait_for(std::chrono::milliseconds(100)) == std::future_status::timeout;
        fault.open = true; // Always release before any throwing assertion/future destructor.
        fallback.get();
        require(reached && waited, "synchronous fallback overtook the unfinished previous capture");
        mem.harvest_flush();
        kvmem_transfer_test_access::commit(mem);
    } else if (worker_error) {
        kvmem_transfer_test_access::capture_start(mem);
        require(kvmem_transfer_test_access::capture_submit(mem), "first worker fault batch not submitted");
        fault.wait_entered();
        kvmem_transfer_test_access::mean_inputs(mem, second.layers, 8, 8);
        kvmem_transfer_test_access::query_nodes(mem, second_q.layers);
        require(kvmem_transfer_test_access::capture_submit(mem), "second worker fault batch not submitted");
        require(kvmem_transfer_test_access::capture_queued(mem) == 1, "second task missing behind blocked worker");
        producer.reset();
        fault.open = true;
        expect_capture_error(mem, scenario == 14);
        expect_capture_error(mem); // A second completion call must not replay partial Q/K commits.
        require(kvmem_transfer_test_access::capture_queued(mem) == 1, "failed worker consumed queued successor");
    } else if (unconfirmed) {
        bool threw = false;
        try { kvmem_transfer_test_access::capture_submit(mem); }
        catch (const std::exception &) { threw = true; }
        require(threw && kvmem_transfer_test_access::capture_domain_failed(mem),
                "unconfirmed partial DMA permitted synchronous fallback");
        expect_capture_error(mem);
        bool shutdown_rejected = false;
        try { kvmem_transfer_test_access::capture_shutdown(mem); }
        catch (const std::exception &) { shutdown_rejected = true; }
        require(shutdown_rejected, "unconfirmed shutdown released device task ownership");
        require(kvmem_transfer_test_access::capture_positions(mem) == 1, "failed submission lost original positions");
        producer.reset();
    } else {
        require(!kvmem_transfer_test_access::capture_submit(mem), "fault did not select checked sync fallback");
        require(fault.entered && !kvmem_transfer_test_access::capture_domain_failed(mem) &&
                kvmem_transfer_test_access::capture_positions(mem) == 1,
                "drained submission failed to preserve fallback input");
        kvmem_transfer_test_access::mean_sync(mem);
        require(kvmem_transfer_test_access::query_matches(mem, 8, mean_sum(0, 8, qdim)),
                "synchronous fallback lost or duplicated query rows");
    }
    if (worker_error || unconfirmed) {
        require(repository->retire(handle) == Repository::RetireResult::Committed, "failed task retirement failed");
        require(repository->try_reclaim(handle) == Repository::ReclaimResult::Deferred,
                "failed capture dropped its CPU writer before device settlement/cancellation");
        require(!repository->acquire_read(handle), "retired incomplete generation admitted new read");
    }
    {
        auto held = weak_producer.lock();
        require(bool(held), "failed capture lost same-generation producer");
        for (size_t i = 0; i < cfg.attention_layers.size(); ++i) {
            const auto progress = payload->write_progress(held->writer(), handle, cfg.attention_layers[i]);
            const uint32_t expected = ordered_fallback ? 16 :
                worker_error ? (host_error && i == 0 ? 8 : 0) : (unconfirmed ? 0 : 8);
            require(progress && progress->mean_k_tokens == expected,
                    "failed capture wrote unexpected Mean-K prefix or lost its payload");
        }
    }
    const uint32_t expected_q = ordered_fallback ? 16 : (host_error || (!worker_error && !unconfirmed)) ? 8 : 0;
    require(kvmem_transfer_test_access::query_matches(mem, expected_q, mean_sum(0, expected_q, qdim)),
            "worker failure duplicated or partially replayed Q accumulation");
    // Remove the hook after joining; only a successful real stream drain may
    // discard the incomplete CPU tasks and their original position owners.
    kvmem_transfer_test_access::capture_hook(mem, nullptr, nullptr);
    kvmem_transfer_test_access::capture_shutdown(mem);
    kvmem_transfer_test_access::cancel_capture_inputs(mem);
    require(kvmem_transfer_test_access::query_matches(mem, expected_q, mean_sum(0, expected_q, qdim)),
            "cancellation retried already committed Q rows");
    if (!worker_error && !unconfirmed) {
        producer.reset();
        require(repository->retire(handle) == Repository::RetireResult::Committed, "fallback retirement failed");
    }
    require(weak_producer.expired(), "cancelled capture retained writer after confirmed drain");
    require(repository->inspect(handle)->life == kvmem::HistoricalKvLife::Retired,
            "failed/cancelled capture resurrected a retired generation");
    require(repository->try_reclaim(handle) == Repository::ReclaimResult::Complete && payload->allocated_payload_bytes() == 0,
            "cancelled capture payload was not reclaimable after writer release");
    std::printf("PASS capture failure scenario=%u: error/fallback, exact Q/K prefixes, retained writer, checked cancellation, CPU release\n", scenario);
}

static void check_mean_ranges(llama_memory_kvmem & mem) {
    const auto & cfg = mem.raw().config();
    const auto block = mem.block_tokens();
    MeanTensors means(mem, 0, 4, false, GGML_TYPE_F32);
    kvmem_transfer_test_access::mean_inputs(mem, means.layers, 0, 4);
    kvmem_transfer_test_access::mean_positions(mem, {0, 1, llama_pos(block), llama_pos(block + 1)});
    kvmem_transfer_test_access::mean_sync(mem);
    for (uint32_t il : cfg.historical_gc_attention_layers) {
        std::vector<float> first(cfg.n_embd_k), second(cfg.n_embd_k);
        mem.raw().mean_k(0, il, first.data());
        mem.raw().mean_k(1, il, second.data());
        for (size_t d = 0; d < first.size(); ++d) {
            require(first[d] == 1.5f + float(d % 8) / 8 &&
                    second[d] == 3.5f + float(d % 8) / 8,
                    "Mean-K lost original positions across a gap/block boundary");
        }
    }
    kvmem_transfer_test_access::clear_mean_nodes(mem);
    mem.clear(true);
    std::printf("PASS Mean-K original positions: discontiguous, cross-block\n");
}

static void check_access(llama_memory_kvmem & mem) {
    using Repository = kvmem::HistoricalKvRepository;
    auto & runtime = mem.runtime();
    auto & store = runtime.store();
    auto & raw = mem.raw();
    const auto & cfg = raw.config();
    auto * kv = mem.get_kv();
    const uint32_t block = mem.block_tokens();
    require(!cfg.historical_gc_attention_layers.empty() &&
            cfg.k_gpu_row_bytes && cfg.v_gpu_row_bytes && !mem.v_trans(),
            "test needs attention layers and packed K/V");

    std::vector<llama_pos> positions(2 * block);
    for (size_t i = 0; i < positions.size(); ++i) positions[i] = i;
    llama_ubatch ub{};
    ub.n_tokens = static_cast<uint32_t>(positions.size());
    ub.n_pos = 1;
    ub.pos = ub.logical_pos = positions.data();
    llama_kv_cache::slot_info_vec_t slots;
    require(mem.prepare_ubatches({ub}, ub.n_tokens, slots), "slot preparation failed");

    struct TensorCheck {
        ggml_tensor * tensor;
        std::vector<uint8_t> expected;
    };
    std::vector<TensorCheck> checks;
    const uint32_t cell0 = store.blocks()[0].gpu_slot * block;
    const std::vector<float> mean(block * cfg.n_embd_k, 1.0f);
    for (uint32_t il : cfg.historical_gc_attention_layers) {
        const auto k = pattern(block * cfg.k_gpu_row_bytes, il + 100);
        const auto v = pattern(block * cfg.v_gpu_row_bytes, il + 200);
        for (uint32_t id = 0; id < 2; ++id) {
            raw.write_layer_mean_k(id * block, block, il, mean.data());
            raw.write_layer_k_gpu(id * block, block, il, k.data());
            raw.write_layer_v_gpu(id * block, block, il, v.data());
        }
        for (auto * tensor : {kv->get_k_storage(il), kv->get_v_storage(il)}) {
            require(tensor && !ggml_backend_buffer_is_host(tensor->buffer),
                    "test requires GPU KV tensors");
            const auto & packed = tensor == kv->get_k_storage(il) ? k : v;
            const size_t row = ggml_row_size(tensor->type, tensor->ne[0]);
            require(packed.size() == block * row, "packed row format mismatch");
            TensorCheck check{tensor, std::vector<uint8_t>(ggml_nbytes(tensor), 0x5a)};
            ggml_backend_tensor_set(tensor, check.expected.data(), 0, check.expected.size());
            std::copy(packed.begin(), packed.end(), check.expected.begin() + cell0 * row);
            checks.push_back(std::move(check));
        }
    }

    auto repository = std::make_shared<Repository>(1);
    const std::weak_ptr<Repository> weak_repository = repository;
    // No async producer or worker outlives this fixture. All release attempts
    // below are joined before the adapter's RawKvStore can be destroyed.
    auto writer = repository->begin_write(0, 0, block, 1, raw.allocated_bytes(),
        [raw_ptr = &raw, weak_repository](const kvmem::HistoricalKvHandle & handle) {
            const auto owner = weak_repository.lock();
            if (!owner) return false;
            const auto result = raw_ptr->release_for_historical_gc(*owner, handle, 0);
            return result.code == kvmem::RawKvStore::HistoricalGcReleaseCode::Released ||
                   result.code == kvmem::RawKvStore::HistoricalGcReleaseCode::AlreadyReleased;
        });
    require(writer.has_value(), "could not reserve historical block");
    const auto handle = writer->handle();
    require(raw.seal_for_historical_gc(*repository, *writer, handle, 0, 0, block),
            "could not seal all attention layers");
    require(repository->publish_write(*writer), "could not publish completed block");
    kvmem::HistoricalKvShadowDirectory::Producer producer;
    require(runtime.bind_historical_full_block(0, repository, handle, *writer, producer),
            "could not bind adapter block identity");
    producer.reset(); // The completed producer is not a permanent directory pin.

    kvmem_transfer_test_access::score(mem);
    require(store.blocks()[0].retrieval_score > 0.9 &&
            store.blocks()[1].retrieval_score > 0.9, "live/legacy scoring failed");
    kvmem_transfer_test_access::restore(mem, 0);
    // Intentionally no stage-in flush: managed restores must complete H2D
    // before returning and releasing their transfer lease.
    auto compare_gpu = [&] {
        for (const auto & check : checks) {
            std::vector<uint8_t> actual(check.expected.size());
            ggml_backend_tensor_get(check.tensor, actual.data(), 0, actual.size());
            require(actual == check.expected, "managed GPU restore changed wrong bytes");
        }
    };
    compare_gpu();
    require(repository->inspect(handle)->references == 0,
            "adapter leaked a read or transfer lease after completion");

    auto old_read = runtime.historical_shadow_directory().acquire_read(0);
    require(bool(old_read), "old reader was not admitted");
    const auto bytes_before = raw.allocated_bytes();
    require(repository->retire(handle) == Repository::RetireResult::Committed,
            "could not retire historical block");
    auto reject_restore = [&] {
        bool rejected = false;
        try { kvmem_transfer_test_access::restore(mem, 0); }
        catch (const std::runtime_error &) { rejected = true; }
        require(rejected, "retired block was accepted for GPU restore");
        compare_gpu();
    };
    reject_restore();
    kvmem_transfer_test_access::score(mem);
    require(store.blocks()[0].retrieval_score == 0.0 &&
            store.blocks()[1].retrieval_score > 0.9 && raw.has_block(0),
            "retired raw payload was scored or legacy scoring changed");
    require(runtime.unbind_retired_historical_block(0), "retired writer was not detached");
    auto reclaim = [&] {
        auto result = Repository::ReclaimResult::Stale;
        std::thread worker([&] { result = repository->try_reclaim(handle); });
        worker.join();
        return result;
    };
    require(reclaim() == Repository::ReclaimResult::Deferred &&
            raw.allocated_bytes() == bytes_before, "old reader did not defer CPU release");
    std::vector<float> old_mean(cfg.n_embd_k);
    raw.mean_k(0, cfg.historical_gc_attention_layers.front(), old_mean.data());
    require(std::all_of(old_mean.begin(), old_mean.end(), [](float x) { return x == 1.0f; }),
            "old admitted reader could not finish");
    old_read.lease.reset();
    require(reclaim() == Repository::ReclaimResult::Complete && !raw.has_block(0) &&
            raw.has_block(1) && raw.allocated_bytes() < bytes_before,
            "CPU release did not remove only the retired historical payload");
    require(runtime.confirm_reclaimed_historical_block(0), "reclaim confirmation failed");
    reject_restore();
    kvmem_transfer_test_access::score(mem);
    require(store.blocks()[0].retrieval_score == 0.0 &&
            store.blocks()[1].retrieval_score > 0.9, "reclaimed tombstone was bypassed");
    require(store.blocks()[0].gpu_slot * block == cell0,
            "CPU reclamation changed the GPU slot");
    std::printf("PASS historical adapter access: layers=%zu payload_bytes_released=%zu\n",
                cfg.historical_gc_attention_layers.size(), bytes_before - raw.allocated_bytes());
    runtime.truncate_to(0);
}

static void check_managed_async_harvest(llama_memory_kvmem & mem) {
    using Repository = kvmem::HistoricalKvRepository;
    using Payload = kvmem::HistoricalKvRawStore;
    const auto & legacy_cfg = mem.raw().config();
    const uint32_t block = mem.block_tokens();
    kvmem::HistoricalKvRawStoreConfig cfg;
    cfg.n_layer = legacy_cfg.n_layer;
    cfg.max_tokens_per_slot = block;
    cfg.packed_k_bytes_per_token = legacy_cfg.k_gpu_row_bytes;
    cfg.packed_v_bytes_per_token = legacy_cfg.v_gpu_row_bytes;
    cfg.mean_k_elements = legacy_cfg.n_embd_k;
    cfg.attention_layers = legacy_cfg.historical_gc_attention_layers;
    auto payload = std::make_shared<Payload>(cfg, 1);
    auto repository = std::make_shared<Repository>(1);
    const std::weak_ptr<Repository> weak_repository = repository;
    kvmem::HistoricalKvHandle previous{};
    for (uint32_t round = 0; round < 5; ++round) {
        mem.clear(true);
        std::vector<llama_pos> positions(block);
        for (uint32_t i = 0; i < block; ++i) positions[i] = i;
        llama_ubatch ub{};
        ub.n_tokens = block;
        ub.n_pos = 1;
        ub.pos = ub.logical_pos = positions.data();
        llama_kv_cache::slot_info_vec_t slots;
        require(mem.prepare_ubatches({ub}, block, slots), "managed harvest slot preparation failed");
        mem.occupy_in(mem.get_kv(), 0);
        auto & runtime = mem.runtime();
        const auto & directory = runtime.historical_shadow_directory();
        auto writer = repository->begin_write(0, 0, block, round + 1, 0,
            [payload, weak_repository](const kvmem::HistoricalKvHandle & h) {
                const auto owner = weak_repository.lock();
                if (!owner) return false;
                const auto result = payload->release(*owner, h);
                return result.code == Payload::ReleaseCode::Released ||
                       result.code == Payload::ReleaseCode::AlreadyReleased;
            });
        require(writer.has_value(), "managed harvest slot was not reusable");
        const auto handle = writer->handle();
        require(!round || (handle.slot == previous.slot && handle.generation > previous.generation),
                "reused payload slot has no new generation");
        require(payload->reserve(*writer, handle, 0, block), "managed payload reserve failed");
        struct LayerCheck {
            uint32_t il;
            std::vector<uint8_t> k, v;
        };
        std::vector<LayerCheck> checks;
        const uint32_t cell0 = runtime.store().blocks()[0].gpu_slot * block;
        for (uint32_t il : cfg.attention_layers) {
            LayerCheck check{il, pattern(block * cfg.packed_k_bytes_per_token, il + 300 + round),
                                pattern(block * cfg.packed_v_bytes_per_token, il + 400 + round)};
            // Distinct K/V prefixes prove capture submits only each missing
            // component suffix; blindly rewriting from row zero is rejected.
            if (round == 0) {
                require(payload->write_mean_k_slice(*writer, handle, il, 0, 4,
                                                    mean_sum(0, 4, cfg.mean_k_elements)),
                        "managed fixture Mean-K prefix write failed");
                require(payload->write_packed_k_slice(*writer, handle, il, 0, 4,
                            std::vector<uint8_t>(check.k.begin(), check.k.begin() + 4 * cfg.packed_k_bytes_per_token)) &&
                        payload->write_packed_v_slice(*writer, handle, il, 0, 8,
                            std::vector<uint8_t>(check.v.begin(), check.v.begin() + 8 * cfg.packed_v_bytes_per_token)),
                        "managed fixture component prefix write failed");
            }
            ggml_backend_tensor_set(mem.get_kv()->get_k_storage(il), check.k.data(),
                                     cell0 * cfg.packed_k_bytes_per_token, check.k.size());
            ggml_backend_tensor_set(mem.get_kv()->get_v_storage(il), check.v.data(),
                                     cell0 * cfg.packed_v_bytes_per_token, check.v.size());
            checks.push_back(std::move(check));
        }
        kvmem::HistoricalKvShadowDirectory::Producer producer;
        require(runtime.bind_historical_full_block(0, repository, handle, *writer, producer, payload),
                "managed payload directory binding failed");
        MeanTensors means(mem, 0, block, round == 3, round < 2 ? GGML_TYPE_F16 : GGML_TYPE_F32);
        if (round == 0) {
            // A prepared block can span multiple executed microbatches. Even
            // though all GPU rows exist in this fixture, only the executed
            // prefix is eligible for capture. A second submission before DMA
            // commit must account for queued prefixes, not write them twice.
            kvmem_transfer_test_access::mean_inputs(mem, means.layers, 0, 16);
            kvmem_transfer_test_access::capture_current(mem);
            const auto queued = kvmem_transfer_test_access::pending(mem);
            kvmem_transfer_test_access::capture_current(mem);
            require(queued == kvmem_transfer_test_access::pending(mem),
                    "packed retry duplicated still-pending component slices");
            kvmem_transfer_test_access::commit(mem);
            for (uint32_t il : cfg.attention_layers) {
                const auto progress = payload->write_progress(producer->writer(), handle, il);
                require(progress && progress->packed_k_tokens == 16 && progress->packed_v_tokens == 16 &&
                        progress->mean_k_tokens == 4, "packed capture copied unexecuted future rows");
            }
        }
        kvmem_transfer_test_access::mean_inputs(mem, means.layers, 0, block);
        if (round == 1) {
            // Preparation already owns the producer. Later retirement closes
            // new work, but cannot cancel this admitted batch's packed task.
            producer.reset();
            require(repository->retire(handle) == Repository::RetireResult::Committed,
                    "could not retire the prepared capture before packed submission");
        }
        if (round < 2) kvmem_transfer_test_access::capture_current(mem);
        if (round < 2) {
            require(kvmem_transfer_test_access::mean_submit(mem), "asynchronous Mean-K submit failed");
        } else if (round == 2) {
            // Exercise the production graph-completion entry, including the
            // packed task handoff before Mean-K consumes each microbatch.
            // Both tasks are admitted before dropping the request producer;
            // the first packed DMA is still pending when the second arrives.
            auto first = ub, second = ub;
            first.n_tokens = 16;
            second.n_tokens = block - 16;
            second.pos = second.logical_pos = positions.data() + 16;
            require(mem.prepare_ubatches({first, second}, 0, slots), "split producer admission failed");
            MeanTensors prefix(mem, 0, 16, false, GGML_TYPE_F32);
            MeanTensors suffix(mem, 16, block - 16, false, GGML_TYPE_F32);
            producer.reset();
            kvmem_transfer_test_access::mean_nodes(mem, prefix.layers);
            mem.harvest_pending(nullptr);
            kvmem_transfer_test_access::mean_nodes(mem, suffix.layers);
            mem.harvest_pending(nullptr);
            mem.harvest_flush();
        } else {
            // Only 8 accepted tokens from a 32-token draft enter the sum.
            kvmem_transfer_test_access::mean_decode(mem, 8);
            require(payload->write_progress(directory.find(0)->writer_context->writer(), handle,
                                            cfg.attention_layers.front())->mean_k_tokens == 0,
                    "decode accumulator flushed speculative tokens prematurely");
        }
        require(kvmem_transfer_test_access::pending(mem) > 0, "test did not enqueue asynchronous packed writes");
        const bool retire_in_flight = round == 1 || round == 4;
        std::unique_ptr<kvmem::HistoricalKvGc> background_gc;
        auto reclaim = [&] {
            auto result = Repository::ReclaimResult::Stale;
            std::thread worker([&] { result = repository->try_reclaim(handle); });
            worker.join();
            return result;
        };
        if (retire_in_flight) {
            producer.reset(); // Only the queued DMA/Mean-K tasks now hold the writer.
            require(repository->retire(handle) == (round == 1
                    ? Repository::RetireResult::AlreadyRetired : Repository::RetireResult::Committed),
                    "in-flight writer retirement failed");
            if (round != 1) require(runtime.unbind_retired_historical_block(0),
                                   "retired writer observation failed");
            require(reclaim() == Repository::ReclaimResult::Deferred,
                    "queued DMA writer failed to retain its allocation");
        }
        kvmem_transfer_test_access::commit(mem);
        if (retire_in_flight) require(reclaim() == Repository::ReclaimResult::Deferred,
                "Mean-K task/accumulator lost its writer after packed completion");
        if (round < 2) {
            if (round == 1) {
                kvmem::HistoricalKvGc::Config gc_cfg;
                gc_cfg.mode = kvmem::HistoricalKvGc::Mode::Delete;
                gc_cfg.scan_blocks = gc_cfg.batch_blocks = gc_cfg.reclaim_blocks = 1;
                gc_cfg.poll_ms = 1;
                background_gc = std::make_unique<kvmem::HistoricalKvGc>(
                    repository, std::make_unique<ObserveOnlyPolicy>(), gc_cfg);
                background_gc->start();
                std::exception_ptr failure;
                // Drain the real CUDA event and captured Mean-K on another
                // thread. This is task completion, not another active/query.
                std::thread completion([&] {
                    try { mem.harvest_flush(); } catch (...) { failure = std::current_exception(); }
                });
                completion.join();
                if (failure) std::rethrow_exception(failure);
                const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
                while (!repository->reclaim_completed(handle, repository->repository_id()) &&
                        std::chrono::steady_clock::now() < deadline)
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                background_gc->stop();
                require(background_gc->stats().reclaimed == 1 &&
                        directory.observe(0).state == kvmem::HistoricalKvShadowDirectory::State::Bound,
                        "CUDA producer completion required a foreground unbind to reclaim");
                std::puts("PASS managed task lifetime: late CUDA Mean-K completed, background CPU release, no unbind/next active");
            } else mem.harvest_flush();
            if (round == 0) {
                // Replaying an already published capture must not count its
                // tokens again or discover new writer work.
                kvmem_transfer_test_access::mean_inputs(mem, means.layers, 0, block);
                kvmem_transfer_test_access::mean_sync(mem);
            }
        } else if (round == 3) {
            // Host accumulator followed by GPU inputs must preserve both sums.
            MeanTensors suffix(mem, 8, block - 8, false, GGML_TYPE_F32);
            kvmem_transfer_test_access::mean_inputs(mem, suffix.layers, 8, block - 8);
            kvmem_transfer_test_access::mean_decode(mem, block - 8);
            kvmem_transfer_test_access::commit(mem);
        } else if (round == 4) {
            kvmem_transfer_test_access::mean_inputs(mem, means.layers, 8, block - 8);
            kvmem_transfer_test_access::mean_decode(mem, block - 8);
            require(kvmem_transfer_test_access::decode_tokens(mem) == 8,
                    "retired accumulator accepted a newly submitted range");
            mem.decode_mean_flush();
        }
        kvmem_transfer_test_access::clear_mean_nodes(mem);
        require(kvmem_transfer_test_access::pending(mem) == 0 && !mem.raw().has_block(0),
                "managed completion leaked jobs or wrote the positional legacy store");
        if (!retire_in_flight) {
            require(repository->inspect(handle)->life == kvmem::HistoricalKvLife::Live,
                    "last managed component did not automatically publish");
            producer.reset();
            require(!directory.find(0)->writer_context && repository->inspect(handle)->references == 0,
                    "completed capture retained a producer after publication");
            {
                auto read = directory.acquire_read(0);
                require(read && read.storage == payload, "managed read resolved wrong allocation");
                for (const auto & check : checks) {
                    const auto layer = payload->read_layer(read.lease, handle, check.il);
                    require(layer && layer->packed_k == check.k && layer->packed_v == check.v,
                            "async component suffixes corrupted captured bytes");
                    require(layer->mean_tokens == block &&
                            layer->mean_k_sum == mean_sum(0, block, cfg.mean_k_elements),
                            "Mean-K capture duplicated, omitted or included speculative tokens");
                }
                if (round) require(!payload->read_layer(read.lease, previous, cfg.attention_layers.front()),
                                   "old generation read the reused slot");
            }
            kvmem_transfer_test_access::score(mem);
            require(runtime.store().blocks()[0].retrieval_score > 0.9,
                    "scorer did not read managed Mean-K");
            for (const auto & check : checks) {
                for (auto * tensor : {mem.get_kv()->get_k_storage(check.il), mem.get_kv()->get_v_storage(check.il)}) {
                    std::vector<uint8_t> blank(ggml_nbytes(tensor), 0x5a);
                    ggml_backend_tensor_set(tensor, blank.data(), 0, blank.size());
                }
            }
            kvmem_transfer_test_access::restore(mem, 0);
            for (const auto & check : checks) {
                for (auto * tensor : {mem.get_kv()->get_k_storage(check.il), mem.get_kv()->get_v_storage(check.il)}) {
                    const auto & bytes = tensor == mem.get_kv()->get_k_storage(check.il) ? check.k : check.v;
                    const size_t row = ggml_row_size(tensor->type, tensor->ne[0]);
                    std::vector<uint8_t> expected(ggml_nbytes(tensor), 0x5a), actual(expected.size());
                    std::copy(bytes.begin(), bytes.end(), expected.begin() + cell0 * row);
                    ggml_backend_tensor_get(tensor, actual.data(), 0, actual.size());
                    require(actual == expected, "managed storage stage-in restored incorrect GPU bytes");
                }
            }
            require(repository->retire(handle) == Repository::RetireResult::Committed &&
                    runtime.unbind_retired_historical_block(0), "completed capture retirement failed");
        }
        const bool reclaimed = background_gc
            ? repository->reclaim_completed(handle, repository->repository_id())
            : reclaim() == Repository::ReclaimResult::Complete;
        require(reclaimed && payload->allocated_payload_bytes() == 0 &&
                runtime.confirm_reclaimed_historical_block(0), "captured generation did not fully reclaim");
        const auto queued_before = kvmem_transfer_test_access::pending(mem);
        kvmem_transfer_test_access::capture(mem, 0);
        kvmem_transfer_test_access::mean_inputs(mem, means.layers, 0, block);
        kvmem_transfer_test_access::mean_sync(mem);
        kvmem_transfer_test_access::clear_mean_nodes(mem);
        require(kvmem_transfer_test_access::pending(mem) == queued_before && !mem.raw().has_block(0),
                "reclaimed GPU block was recaptured into legacy storage");
        previous = handle;
        std::printf("PASS managed harvest: round=%u generation=%llu retired_in_flight=%d\n",
                    round, (unsigned long long) handle.generation, retire_in_flight);
    }
    mem.clear(true);
}

static void check_partial_rollover(llama_memory_kvmem & mem) {
    using Repository = kvmem::HistoricalKvRepository;
    using Payload = kvmem::HistoricalKvRawStore;
    const auto & legacy = mem.raw().config();
    const uint32_t block = mem.block_tokens();
    kvmem::HistoricalKvRawStoreConfig cfg{legacy.n_layer, block,
        legacy.k_gpu_row_bytes, legacy.v_gpu_row_bytes, legacy.n_embd_k,
        legacy.historical_gc_attention_layers};
    auto repository = std::make_shared<Repository>(2);
    auto payload = std::make_shared<Payload>(cfg, 2);
    const std::weak_ptr<Repository> weak = repository;
    auto & runtime = mem.runtime();
    const auto & directory = runtime.historical_shadow_directory();
    auto make_writer = [&](uint32_t id, uint32_t begin, uint32_t count) {
        auto writer = repository->begin_write(id, begin, count, id + 1, 0,
            [payload, weak](kvmem::HistoricalKvHandle h) {
                auto owner = weak.lock();
                if (!owner) return false;
                const auto result = payload->release(*owner, h);
                return result.code == Payload::ReleaseCode::Released ||
                       result.code == Payload::ReleaseCode::AlreadyReleased;
            });
        require(writer && payload->reserve(*writer, writer->handle(), begin, count), "tail reservation failed");
        return std::move(*writer);
    };
    auto reclaim = [&](kvmem::HistoricalKvHandle handle) {
        auto result = Repository::ReclaimResult::Stale;
        std::thread worker([&] { result = repository->try_reclaim(handle); });
        worker.join();
        return result;
    };
    struct Check {
        uint32_t id, il, cell0;
        std::vector<uint8_t> k, v;
    };
    std::vector<Check> checks;
    for (uint32_t il : cfg.attention_layers) {
        for (auto * tensor : {mem.get_kv()->get_k_storage(il), mem.get_kv()->get_v_storage(il)}) {
            std::vector<uint8_t> blank(ggml_nbytes(tensor), 0x5a);
            ggml_backend_tensor_set(tensor, blank.data(), 0, blank.size());
        }
    }
    auto place = [&](uint32_t id) {
        kvmem_transfer_test_access::place(mem, id);
        const auto & span = runtime.store().blocks()[id];
        const uint32_t cell0 = span.gpu_slot * block;
        for (uint32_t il : cfg.attention_layers) {
            Check c{id, il, cell0,
                pattern(span.n_tokens * cfg.packed_k_bytes_per_token, 500 + 7 * id + il),
                pattern(span.n_tokens * cfg.packed_v_bytes_per_token, 600 + 7 * id + il)};
            ggml_backend_tensor_set(mem.get_kv()->get_k_storage(il), c.k.data(),
                cell0 * cfg.packed_k_bytes_per_token, c.k.size());
            ggml_backend_tensor_set(mem.get_kv()->get_v_storage(il), c.v.data(),
                cell0 * cfg.packed_v_bytes_per_token, c.v.size());
            checks.push_back(std::move(c));
        }
    };
    auto prepare = [&](uint32_t begin, uint32_t count) {
        std::vector<llama_pos> positions(count);
        for (uint32_t i = 0; i < count; ++i) positions[i] = begin + i;
        llama_ubatch ub{};
        ub.n_tokens = count;
        ub.n_pos = 1;
        ub.pos = ub.logical_pos = positions.data();
        llama_kv_cache::slot_info_vec_t slots;
        require(mem.prepare_ubatches({ub}, 0, slots), "non-aligned batch prepare failed");
        for (uint32_t i = 0; i < count; ++i) {
            const auto id = runtime.store().block_id_containing(begin + i);
            const auto & span = runtime.store().blocks()[id];
            require(slots[0].idxs[0][i] == span.gpu_slot * block + begin + i - span.orig_pos_start,
                    "slot mapping used aligned original positions");
        }
    };
    std::vector<kvmem::HistoricalKvShadowDirectory::Producer> producers;
    std::vector<Repository::Lease> writers;
    writers.push_back(make_writer(0, 0, 3));
    writers.push_back(make_writer(1, 3, 5));
    const auto old = writers[1].handle();
    require(runtime.register_historical_append(repository, payload, writers, producers), "short spans append failed");
    place(0);
    place(1);
    prepare(0, 8);
    mem.occupy_in(mem.get_kv(), 0);
    mem.occupy_in(mem.get_kv(), 1);
    MeanTensors first(mem, 0, 8, false, GGML_TYPE_F16);
    kvmem_transfer_test_access::mean_inputs(mem, first.layers, 0, 8);
    kvmem_transfer_test_access::capture_current(mem);
    require(kvmem_transfer_test_access::mean_submit(mem), "short Mean-K DMA failed");
    require(kvmem_transfer_test_access::pending(mem) > 0, "short packed spans were skipped");
    auto late = directory.find(1);
    auto first_producer = producers[0];
    producers.clear();
    require(repository->retire(old) == Repository::RetireResult::Committed &&
            runtime.unbind_retired_historical_block(1), "short Writing tail retirement failed");
    require(reclaim(old) == Repository::ReclaimResult::Deferred, "short writer not retained");
    kvmem_transfer_test_access::commit(mem);
    mem.harvest_flush();
    kvmem_transfer_test_access::clear_mean_nodes(mem);
    for (uint32_t il : cfg.attention_layers) {
        auto progress = payload->write_progress(late->writer_context->writer(), old, il);
        require(progress && progress->packed_k_tokens == 5 && progress->packed_v_tokens == 5 &&
                progress->mean_k_tokens == 5, "late short capture lost part of its admitted range");
    }
    require(!payload->publish_complete(*repository, late->writer_context->writer(), old),
            "short tail was resurrected");
    late.reset();
    require(reclaim(old) == Repository::ReclaimResult::Complete &&
            runtime.confirm_reclaimed_historical_block(1), "short tail reclaim failed");

    writers.clear();
    // Cross the old position/B boundary (32) within a single admitted span.
    writers.push_back(make_writer(2, 8, 27));
    const auto fresh = writers[0].handle();
    require(fresh.slot == old.slot && fresh.generation > old.generation, "tail slot generation not advanced");
    require(runtime.register_historical_append(repository, payload, writers, producers), "rollover append failed");
    place(2);
    prepare(8, 27);
    mem.occupy_in(mem.get_kv(), 2);
    MeanTensors next(mem, 8, 27, true, GGML_TYPE_F32);
    kvmem_transfer_test_access::mean_inputs(mem, next.layers, 8, 27);
    kvmem_transfer_test_access::mean_decode(mem, 27);
    require(kvmem_transfer_test_access::decode_tokens(mem) == 0,
            "non-aligned decode mean did not flush at admitted end");
    kvmem_transfer_test_access::commit(mem);
    kvmem_transfer_test_access::clear_mean_nodes(mem);
    for (uint32_t id : {0u, 2u}) {
        auto binding = directory.find(id);
        const auto & span = runtime.store().blocks()[id];
        require(repository->inspect(binding->handle)->life == kvmem::HistoricalKvLife::Live,
                "last short-span component did not automatically publish");
        auto read = directory.acquire_read(id);
        for (const auto & c : checks) if (c.id == id) {
            auto layer = payload->read_layer(read.lease, binding->handle, c.il);
            require(layer && layer->packed_k == c.k && layer->packed_v == c.v &&
                    layer->mean_tokens == span.n_tokens &&
                    layer->mean_k_sum == mean_sum(span.orig_pos_start, span.n_tokens, cfg.mean_k_elements),
                    "non-aligned producer wrote the wrong generation/mean/token span");
        }
        kvmem_transfer_test_access::restore(mem, id);
    }
    // CPU tail release and later stage-in must leave all original GPU bytes,
    // including the retired span and unused cells, untouched.
    for (uint32_t il : cfg.attention_layers) {
        for (bool is_k : {true, false}) {
            auto * tensor = is_k ? mem.get_kv()->get_k_storage(il) : mem.get_kv()->get_v_storage(il);
            const auto row = is_k ? cfg.packed_k_bytes_per_token : cfg.packed_v_bytes_per_token;
            std::vector<uint8_t> expected(ggml_nbytes(tensor), 0x5a), actual(expected.size());
            for (const auto & c : checks) if (c.il == il) {
                const auto & bytes = is_k ? c.k : c.v;
                std::copy(bytes.begin(), bytes.end(), expected.begin() + c.cell0 * row);
            }
            ggml_backend_tensor_get(tensor, actual.data(), 0, actual.size());
            require(actual == expected, "CPU partial rollover changed unrelated GPU cells");
        }
    }
    require(!mem.raw().has_block(0) && !mem.raw().has_block(1) && !mem.raw().has_block(2),
            "short managed spans leaked into positional Raw storage");
    std::string reason;
    require(!mem.can_append(35, 1, false, reason) && reason == "historical_reservation_required",
            "legacy fast path promised managed capacity without request reservation");
    require(!mem.query_replay_fits(3, 35), "legacy replay fit accepted fragmented managed history");
    runtime.register_append(1); // Deliberately leave the new non-aligned span unbound.
    auto rejects_legacy = [](auto operation) {
        try { operation(); } catch (const std::logic_error &) { return true; }
        return false;
    };
    require(rejects_legacy([&] { kvmem_transfer_test_access::score(mem); }) &&
            rejects_legacy([&] { kvmem_transfer_test_access::restore(mem, 3); }) &&
            rejects_legacy([&] { kvmem_transfer_test_access::mean_positions(mem, {35}); }),
            "unbound non-aligned span fell back to positional legacy storage");
    first_producer.reset();
    producers.clear();
    for (uint32_t id : {0u, 2u}) {
        const auto handle = directory.find(id)->handle;
        require(repository->retire(handle) == Repository::RetireResult::Committed &&
                runtime.unbind_retired_historical_block(id) &&
                reclaim(handle) == Repository::ReclaimResult::Complete &&
                runtime.confirm_reclaimed_historical_block(id), "short fixture cleanup failed");
    }
    require(payload->allocated_payload_bytes() == 0, "short fixture retained payload");
    mem.clear(true);
    std::puts("PASS managed partial rollover: spans=[0,3),[3,8),[8,35), CPU slot reused, GPU unchanged");
}

// Real graph outputs, not fixture-written GPU tensors or manual bindings.
struct HistoricalPolicyGate {
    std::promise<void> entered, resume;
    std::future<void> entered_signal = entered.get_future();
    std::shared_future<void> resume_signal = resume.get_future().share();
    std::atomic<bool> announced{false}, resumed{false};
    void wait() {
        if (!announced.exchange(true)) entered.set_value();
        resume_signal.wait();
    }
    void open() { if (!resumed.exchange(true)) resume.set_value(); }
};

class PausedHistoricalPolicy final : public kvmem::HistoricalKvPolicy {
public:
    explicit PausedHistoricalPolicy(std::shared_ptr<HistoricalPolicyGate> gate) : gate_(std::move(gate)) {}
    kvmem::HistoricalKvStepResult step(const kvmem::HistoricalKvSnapshot &,
            const kvmem::HistoricalKvPressure &, kvmem::HistoricalKvWorkBudget,
            kvmem::HistoricalKvDecisionBuffer &) override {
        gate_->wait();
        return {kvmem::HistoricalKvStepStatus::Complete, 0, 0};
    }
private:
    std::shared_ptr<HistoricalPolicyGate> gate_;
};

enum class RetirementPoint {
    None, BetweenCalls, AfterSubmission, WritingAfterSubmission, PausedWorker,
    StageoutRecord, StageoutWait, StageoutUnconfirmed
};
struct ModelStageoutFault {
    KvmemStageoutPoint point = KvmemStageoutPoint::EventRecord;
    bool fired = false, unconfirmed = false;
    unsigned drains = 0;
    static bool inject(void * context, KvmemStageoutPoint point, size_t) noexcept {
        auto & fault = *static_cast<ModelStageoutFault *>(context);
        if (point == KvmemStageoutPoint::Drain) { ++fault.drains; return fault.unconfirmed && fault.fired; }
        if (point == fault.point && !fault.fired) { fault.fired = true; return true; }
        return false;
    }
    void clear() { kvmem_stageout_test_fault(nullptr, nullptr); }
    ~ModelStageoutFault() { clear(); }
};
static void check_request_capture(llama_context * ctx, llama_model * model,
                                  llama_memory_kvmem & mem,
                                  RetirementPoint retirement = RetirementPoint::None,
                                  bool saturated_queue = false, unsigned layout_case = 0) {
    std::string text;
    for (unsigned i = 0; i < 30; ++i) text += "The river passes a small village. The bridge is made of stone. ";
    const auto * vocab = llama_model_get_vocab(model);
    const int count = -llama_tokenize(vocab, text.c_str(), text.size(), nullptr, 0, true, true);
    require(count >= 160, "real capture fixture needs enough prompt tokens");
    std::vector<llama_token> tokens(count);
    require(llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), tokens.size(), true, true) == count,
            "real capture prompt tokenization failed");
    require(!llama_kvmem_historical_capture_begin(ctx, 1, 160, 3, 7) && mem.store_n_tokens() == 0 &&
            !ctx->historical_simple_input_prepared(),
            "insufficient capture reservation changed runtime");
    if (saturated_queue) {
        auto & cleanup = kvmem::HistoricalKvCleanup::process();
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (cleanup.occupied() && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
        require(!cleanup.occupied(), "earlier context cleanup did not finish before rollback fixture");
        std::promise<void> entered, resume;
        auto entered_signal = entered.get_future();
        struct Stall final : kvmem::HistoricalKvCleanup::Task {
            std::promise<void> & entered;
            std::shared_future<void> resume;
            Stall(std::promise<void> & e, std::shared_future<void> r) : entered(e), resume(std::move(r)) {}
            ~Stall() override { entered.set_value(); resume.wait(); }
        };
        struct Unblock {
            std::promise<void> & resume;
            ~Unblock() { resume.set_value(); }
        } unblock{resume};
        auto ticket = cleanup.try_reserve();
        require(bool(ticket), "rollback fixture could not reserve cleanup barrier");
        ticket->submit(std::make_unique<Stall>(entered, resume.get_future().share()));
        require(entered_signal.wait_for(std::chrono::seconds(5)) == std::future_status::ready,
                "rollback fixture did not pause the CPU cleanup worker");
        const auto table_bytes = mem.runtime().store().allocated_bytes();
        const auto inject = [](void *, KvmemCaptureReservePoint point) noexcept {
            return point == KvmemCaptureReservePoint::Pin1;
        };
        kvmem_capture_reserve_test_fault(inject, nullptr);
        const bool admitted = llama_kvmem_historical_capture_begin(ctx, 12, 160, 3, 7);
        kvmem_capture_reserve_test_fault(nullptr, nullptr);
        const auto failed = kvmem_capture_workspace_status();
        require(!admitted && !ctx->historical_simple_input_prepared() &&
                mem.store_n_tokens() == 0 && !failed.reserved && !failed.device_bytes &&
                !failed.pinned_bytes && !failed.events && !failed.host_bytes && cleanup.occupied() == 1 &&
                mem.runtime().store().allocated_bytes() == table_bytes &&
                !mem.runtime().historical_shadow_directory().size() &&
                !mem.get_kv()->get_cells(0).index_status().reserved &&
                kvmem_transfer_test_access::capture_arena_unpublished(mem) &&
                !kvmem_transfer_test_access::slot_workspace(mem).bytes &&
                !kvmem_transfer_test_access::microbatch_workspace(mem).bytes &&
                !kvmem_transfer_test_access::batch_context(mem).storage &&
                kvmem_transfer_test_access::execution_storage_empty(mem),
                "real request preparation failure published an owner or retained workspace");
        std::puts("PASS request prepare rollback: CPU cleanup paused, GPU workspace returned, no metadata or cleanup credit retained");
    }
    require(llama_kvmem_historical_capture_begin(ctx, 12, 160, 3, 7), "real request capture admission failed");
    const auto backend_workspace = ctx->get_historical_backend_status();
    require(backend_workspace.scheduler && backend_workspace.output_buffer &&
            !backend_workspace.needs_scheduler_reserve &&
            backend_workspace.output_bytes >=
                size_t(ctx->n_batch()) * llama_vocab_n_tokens(vocab) * sizeof(float) &&
            backend_workspace.graph_meta_bytes &&
            std::all_of(backend_workspace.graph_results.begin(), backend_workspace.graph_results.end(),
                        [](const void * ptr) { return ptr != nullptr; }) &&
            std::all_of(backend_workspace.graph_meta.begin(), backend_workspace.graph_meta.end(),
                        [](const void * ptr) { return ptr != nullptr; }),
            "real request did not prepare output, scheduler and all ten graph results");
    require(ctx->historical_simple_input_prepared() &&
            kvmem_transfer_test_access::input_batch_without_cpp_allocation(*vocab, tokens.data()),
            "historical input batch normalization allocated after admission");
    const auto table_bytes = mem.runtime().store().allocated_bytes();
    const auto execution_buffers = kvmem_transfer_test_access::execution_buffers(mem);
    const auto execution_bytes = kvmem_transfer_test_access::execution_bytes(mem);
    const auto slot_workspace = kvmem_transfer_test_access::slot_workspace(mem);
    const auto microbatch_workspace = kvmem_transfer_test_access::microbatch_workspace(mem);
    const auto batch_context = kvmem_transfer_test_access::batch_context(mem);
    require(batch_context.storage && batch_context.bytes && !batch_context.in_use,
            "real request did not reserve one decode context before admission");
    require(microbatch_workspace.objects == 14 && microbatch_workspace.max_ubatches == 4 &&
            microbatch_workspace.capacities.size() == 3 + 9 * microbatch_workspace.objects &&
            microbatch_workspace.capacities[0] >= 14 &&
            microbatch_workspace.capacities[1] >= 4 &&
            microbatch_workspace.capacities[2] >= 16 && microbatch_workspace.bytes,
            "real request did not acquire four microbatch data objects and scratch");
    require(kvmem_transfer_test_access::microbatch_split_without_cpp_allocation(mem, *vocab, tokens.data()),
            "prepared upstream simple split allocated C++ storage");
    require(kvmem_transfer_test_access::microbatch_graph_bound_without_cpp_allocation(mem, *vocab, tokens.data()),
            "prepared microbatch pool could not serve a full batch with ten graph-held owners");
    require(slot_workspace.capacities.size() == 13 && slot_workspace.capacities[0] >= 4 &&
            slot_workspace.capacities[3] >= 16 && slot_workspace.bytes,
            "real request did not acquire four microbatch slot buffers before admission");
    const auto cell_nodes = mem.get_kv()->get_cells(0).index_status().nodes;
    require(mem.get_kv()->get_cells(0).index_status().reserved &&
            cell_nodes == size_t(mem.get_kv()->get_cells(0).size()) * 2,
            "real request did not reserve both cell indexes");
    require(kvmem_transfer_test_access::capture_arena_bounded(mem, 163),
            "real request did not reserve capture positions and ranges");
    const auto packed_workspace = kvmem_transfer_test_access::packed_workspace(mem);
    require(packed_workspace.capacity >= 2 * KVMEM_CAPTURE_MAX_JOBS && packed_workspace.fallback_bytes,
            "real request did not prepare both queues and fallback");
    if (saturated_queue) kvmem_transfer_test_access::small_packed_queue(mem);
    require(!llama_kvmem_historical_capture_begin(ctx, 12, 160, 3, 7), "duplicate capture owner accepted");
    struct Saved { uint32_t id, il; uint64_t k, v; };
    std::vector<Saved> saved;
    uint32_t retired_during_request = UINT32_MAX;
    const auto hash = [](const std::vector<uint8_t> & bytes) {
        uint64_t value = 14695981039346656037ull;
        for (auto byte : bytes) value = (value ^ byte) * 1099511628211ull;
        return value;
    };
    const auto execute = [&](llama_token * input, uint32_t n) {
        const auto first = mem.store().block_count();
        auto batch = llama_batch_get_one(input, n);
        require(llama_decode(ctx, batch) == 0, "real managed graph failed");
        llama_synchronize(ctx);
        mem.harvest_flush();
        kvmem_transfer_test_access::commit(mem);
        require(kvmem_transfer_test_access::packed_workspace_unchanged(mem, packed_workspace),
                "real capture reallocated a prepared queue or fallback buffer");
        require(mem.runtime().store().allocated_bytes() == table_bytes,
                "real capture grew the prepared dense block table");
        require(kvmem_transfer_test_access::capture_arena_bounded(mem, 163),
                "real capture exceeded its admitted position/range arena");
        require(kvmem_transfer_test_access::execution_buffers(mem) == execution_buffers &&
                kvmem_transfer_test_access::execution_bytes(mem) == execution_bytes,
                "real capture changed prepared active or cell scratch storage");
        const auto returned_slots = kvmem_transfer_test_access::slot_workspace(mem);
        require(returned_slots.bytes == slot_workspace.bytes &&
                returned_slots.buffers == slot_workspace.buffers &&
                returned_slots.capacities == slot_workspace.capacities,
                "real decode did not return the same prepared slot buffers");
        const auto returned_microbatches = kvmem_transfer_test_access::microbatch_workspace(mem);
        require(returned_microbatches.bytes == microbatch_workspace.bytes &&
                returned_microbatches.buffers == microbatch_workspace.buffers &&
                returned_microbatches.capacities == microbatch_workspace.capacities &&
                returned_microbatches.graph_held <= 10,
                "real decode did not return the same prepared microbatch buffers");
        const auto returned_context = kvmem_transfer_test_access::batch_context(mem);
        require(returned_context.storage == batch_context.storage &&
                returned_context.bytes == batch_context.bytes && !returned_context.in_use,
                "real decode did not return the same prepared context storage");
        const auto backend_after = ctx->get_historical_backend_status();
        require(backend_after.scheduler == backend_workspace.scheduler &&
                backend_after.output_buffer == backend_workspace.output_buffer &&
                backend_after.output_bytes == backend_workspace.output_bytes &&
                backend_after.graph_results == backend_workspace.graph_results &&
                backend_after.graph_meta == backend_workspace.graph_meta &&
                backend_after.graph_meta_bytes == backend_workspace.graph_meta_bytes &&
                !backend_after.needs_scheduler_reserve,
                "real decode changed prepared output, scheduler or graph metadata storage");
        const auto & cells = mem.get_kv()->get_cells(0);
        const auto index = cells.index_status();
        require(index.reserved && index.nodes == cell_nodes &&
                index.available + size_t(cells.get_used()) * 2 == cell_nodes,
                "real capture lost cell node credits or grew indexes");
        for (uint32_t id = first; id < mem.store().block_count(); ++id) {
            if (id == retired_during_request) continue;
            const auto & block = mem.store().blocks()[id];
            const auto & dir = mem.runtime().historical_shadow_directory();
            const auto binding = dir.find(id);
            require(binding && !binding->writer_context && block.gpu_slot >= 0,
                    "actual graph retained creation writer or lost incoming GPU slot");
            auto access = dir.acquire_read(id);
            require(bool(access), "actual graph did not automatically publish packed KV");
            for (auto il : mem.raw().config().historical_gc_attention_layers) {
                const auto layer = access.storage->read_layer(access.lease, binding->handle, il);
                require(bool(layer), "actual graph CPU layer unavailable");
                for (bool is_k : {true, false}) {
                    auto * tensor = is_k ? mem.get_kv()->get_k_storage(il) : mem.get_kv()->get_v_storage(il);
                    const auto & expected = is_k ? layer->packed_k : layer->packed_v;
                    std::vector<uint8_t> gpu(expected.size());
                    const size_t cell = size_t(block.gpu_slot) * mem.block_tokens();
                    ggml_backend_tensor_get(tensor, gpu.data(), cell * tensor->nb[1], gpu.size());
                    require(gpu == expected, "actual graph CPU payload differs from GPU output");
                }
                saved.push_back({id, il, hash(layer->packed_k), hash(layer->packed_v)});
            }
        }
    };
    bool copied_capture = false;
    const bool check_capture_copy = retirement == RetirementPoint::None && !saturated_queue && !layout_case;
    if (check_capture_copy) kvmem_transfer_test_access::on_submission(mem, [&] {
        copied_capture = kvmem_transfer_test_access::capture_copy_without_cpp_allocation(mem);
    });
    uint32_t position = 0;
    execute(tokens.data(), 48);
    require(kvmem_transfer_test_access::slot_fill_without_cpp_allocation(mem),
            "filling a prepared real-model slot list allocated C++ storage");
    if (check_capture_copy) {
        kvmem_transfer_test_access::on_submission(mem, {});
        require(copied_capture, "actual capture batch copy allocated outside its prepared arena");
    }
    position += 48;
    if (layout_case == 1) {
        const auto repository = kvmem_transfer_test_access::capture_repository(mem);
        const auto first = mem.runtime().historical_shadow_directory().observe(0).handle;
        const auto last_use = repository->inspect(first)->last_use_seq;
        const llama_pos next_position = position;
        bool admitted = true;
        size_t selected = 0;
        for (unsigned i = 0; i < 32; ++i) {
            DenyCppAllocation guard;
            admitted = mem.begin_active_ubatch(1, &next_position) && admitted;
            selected = kvmem_transfer_test_access::active_count(mem);
            // An abandoned plan releases permission without recording use.
            mem.complete_active_ubatch();
        }
        require(admitted && selected && repository->inspect(first)->last_use_seq == last_use &&
                !repository->outstanding_operations() &&
                kvmem_transfer_test_access::execution_buffers(mem) == execution_buffers,
                "prepared model active allocated, retained a lease, or recorded abandoned execution");
        std::puts("PASS prepared model active: 32 fresh admissions without C++ allocation, no abandoned LRU use, fixed storage");
    }
    if (layout_case) {
        struct LayoutFault {
            KvmemLayoutPoint point;
            unsigned target, calls = 0;
            bool undrained = false, fired = false;
            static bool inject(void * context, KvmemLayoutPoint point) noexcept {
                auto & f = *static_cast<LayoutFault *>(context);
                if (point == KvmemLayoutPoint::Drain) return f.undrained;
                if (point == f.point && ++f.calls == f.target) { f.fired = true; return true; }
                return false;
            }
            ~LayoutFault() { kvmem_layout_test_fault(nullptr, nullptr); }
        } fault{layout_case == 2 ? KvmemLayoutPoint::KernelResult : KvmemLayoutPoint::Wait,
                layout_case == 2 ? 1u : 2u};
        fault.undrained = layout_case == 4;
        const auto buffers = kvmem_transfer_test_access::layout_buffers(mem);
        const auto prepared = kvmem_capture_workspace_status();
        kvmem_transfer_test_access::swap_layout_slots(mem, 0, 1);
        const auto slot0 = mem.store().blocks()[0].gpu_slot, slot1 = mem.store().blocks()[1].gpu_slot;
        if (layout_case > 1) kvmem_layout_test_fault(LayoutFault::inject, &fault);
        bool threw = false;
        try { require(kvmem_transfer_test_access::layout(mem), "scrambled layout was not moved"); }
        catch (const std::exception &) { threw = true; }
        require(kvmem_transfer_test_access::layout_buffers(mem) == buffers &&
                kvmem_capture_workspace_status().device_bytes == prepared.device_bytes &&
                kvmem_transfer_test_access::execution_buffers(mem) == execution_buffers &&
                kvmem_transfer_test_access::execution_bytes(mem) == execution_bytes,
                "real layout reallocated admitted scratch or descriptors");
        if (layout_case > 1) {
            require(threw && fault.fired && kvmem_transfer_test_access::capture_domain_failed(mem) &&
                    kvmem_transfer_test_access::capture_failed(mem) &&
                    !kvmem_transfer_test_access::append_after_failure(mem) &&
                    mem.store().blocks()[0].gpu_slot == slot0 && mem.store().blocks()[1].gpu_slot == slot1,
                    "failed layout published new slot metadata or left request open");
            if (fault.undrained) {
                bool held = false;
                try { kvmem_stagein_gpu_free(); } catch (const std::exception &) { held = true; }
                require(held && kvmem_capture_workspace_status().device_bytes == prepared.device_bytes,
                        "unconfirmed real layout released its scratch");
            }
            kvmem_layout_test_fault(nullptr, nullptr);
            kvmem_layout_cancel();
            std::printf("PASS real model layout failure: case=%u, closed request, unchanged slot metadata, checked cancel\n", layout_case);
            return;
        }
        require(!threw && mem.store().blocks()[0].gpu_slot == 0 && mem.store().blocks()[1].gpu_slot == 1,
                "real layout did not restore original ordering");
        for (uint32_t id : {0u, 1u}) {
            auto access = mem.runtime().historical_shadow_directory().acquire_read(id);
            require(bool(access), "layout fixture lost CPU history");
            const auto binding = mem.runtime().historical_shadow_directory().find(id);
            for (auto il : mem.raw().config().historical_gc_attention_layers) {
                const auto layer = access.storage->read_layer(access.lease, binding->handle, il);
                require(bool(layer), "layout lost captured payload");
                for (bool is_k : {true, false}) {
                    const auto & expected = is_k ? layer->packed_k : layer->packed_v;
                    auto * tensor = is_k ? mem.get_kv()->get_k_storage(il) : mem.get_kv()->get_v_storage(il);
                    std::vector<uint8_t> actual(expected.size());
                    ggml_backend_tensor_get(tensor, actual.data(), size_t(id) * mem.block_tokens() * tensor->nb[1], actual.size());
                    require(actual == expected, "layout corrupted real model K/V bytes");
                }
            }
        }
        std::puts("PASS real model layout: reordered two resident blocks, rebuilt real cells without C++ allocation, prepared scratch and fixed indexes");
    }
    const bool stageout_fault = retirement == RetirementPoint::StageoutRecord ||
        retirement == RetirementPoint::StageoutWait || retirement == RetirementPoint::StageoutUnconfirmed;
    ModelStageoutFault fault;
    if (stageout_fault) {
        fault.point = retirement == RetirementPoint::StageoutWait ?
            KvmemStageoutPoint::EventWait : KvmemStageoutPoint::EventRecord;
        fault.unconfirmed = retirement == RetirementPoint::StageoutUnconfirmed;
        kvmem_stageout_test_fault(ModelStageoutFault::inject, &fault);
    }
    if (fault.unconfirmed) {
        bool rejected = false;
        try { execute(tokens.data() + position, 32); } catch (const std::exception &) { rejected = true; }
        auto repository = kvmem_transfer_test_access::capture_repository(mem);
        require(rejected && fault.fired && fault.drains && repository->closed() &&
                kvmem_transfer_test_access::capture_failed(mem) &&
                kvmem_transfer_test_access::source_readers(mem) &&
                kvmem_transfer_test_access::pending(mem) && kvmem_stageout_used(),
                "unconfirmed real-model DMA did not close admission and retain source/writer ownership");
        const auto handle = kvmem_transfer_test_access::pending_writer(mem);
        require(repository->retire(handle) == kvmem::HistoricalKvRepository::RetireResult::Committed &&
                repository->try_reclaim(handle) == kvmem::HistoricalKvRepository::ReclaimResult::Deferred &&
                kvmem_transfer_test_access::pending_resources_held(mem),
                "CPU GC freed a failed DMA's captured writer generation");
        const auto rows = mem.store_n_tokens();
        require(!kvmem_transfer_test_access::append_after_failure(mem) && mem.store_n_tokens() == rows,
                "failed model request admitted a new append");
        fault.clear();
        kvmem_transfer_test_access::commit(mem);
        require(!kvmem_transfer_test_access::source_readers(mem) &&
                repository->completion_state(handle) == kvmem::HistoricalKvRepository::CompletionState::Retired &&
                !llama_kvmem_historical_capture_finish(),
                "settling old DMA reopened or falsely completed a failed request");
        std::puts("PASS real model unconfirmed DMA: request closed, writer/source retained, retirement deferred, checked cleanup did not revive history");
        return;
    }
    std::shared_ptr<HistoricalPolicyGate> paused;
    // Open the barrier before the context's GC destructor can join, including
    // on any failed model assertion below.
    struct UnpauseOnExit {
        std::shared_ptr<HistoricalPolicyGate> & gate;
        ~UnpauseOnExit() { if (gate) gate->open(); }
    } unpause{paused};
    if (retirement == RetirementPoint::PausedWorker) {
        paused = std::make_shared<HistoricalPolicyGate>();
        kvmem_transfer_test_access::install_gc_policy(mem, std::make_unique<PausedHistoricalPolicy>(paused));
        require(paused->entered_signal.wait_for(std::chrono::seconds(5)) == std::future_status::ready,
                "GC worker did not reach controlled pause before remaining model work");
    }
    if (retirement == RetirementPoint::BetweenCalls) {
        auto repository = kvmem_transfer_test_access::capture_repository(mem);
        auto storage = kvmem_transfer_test_access::capture_storage(mem);
        require(repository && storage, "running GC fixture has no capture owner");
        const auto before_retire = repository->snapshot();
        require(std::any_of(before_retire.begin(), before_retire.end(),
                    [](const kvmem::HistoricalKvBlockMeta & block) {
                        return block.life == kvmem::HistoricalKvLife::Live &&
                               block.last_use_seq > block.birth_seq;
                    }), "real submitted model computation did not update historical LRU use");
        const auto & target = mem.store().blocks().back();
        retired_during_request = target.block_id;
        require(target.gpu_slot >= 0, "running GC fixture target is not GPU-resident");
        const auto & directory = mem.runtime().historical_shadow_directory();
        const auto handle = directory.observe(target.block_id).handle;
        auto old_reader = directory.acquire_read(target.block_id);
        require(bool(old_reader), "running GC fixture has no old reader");
        const auto il = mem.raw().config().historical_gc_attention_layers.front();
        const auto old_layer = old_reader.storage->read_layer(old_reader.lease, handle, il);
        require(bool(old_layer), "running GC fixture has no original Host payload");
        std::vector<uint8_t> gpu_before(old_layer->packed_k.size());
        std::vector<uint8_t> gpu_after(gpu_before.size());
        auto * gpu_tensor = mem.get_kv()->get_k_storage(il);
        const auto gpu_offset = size_t(target.gpu_slot) * mem.block_tokens() * gpu_tensor->nb[1];
        ggml_backend_tensor_get(gpu_tensor, gpu_before.data(), gpu_offset, gpu_before.size());
        const auto bytes_before = storage->allocated_payload_bytes();
        require(repository->retire(handle) == kvmem::HistoricalKvRepository::RetireResult::Committed &&
                !directory.acquire_read(target.block_id),
                "running GC did not close the next CPU read after retirement");
        ggml_backend_tensor_get(gpu_tensor, gpu_after.data(), gpu_offset, gpu_after.size());
        require(gpu_after == gpu_before,
                "logical retirement changed an already submitted GPU payload");
        require(repository->try_reclaim(handle) == kvmem::HistoricalKvRepository::ReclaimResult::Deferred,
                "running GC released a block while the old reader was active");
        kvmem::HistoricalKvGc::Config cfg;
        cfg.mode = kvmem::HistoricalKvGc::Mode::Off; // External retire; worker only reclaims.
        cfg.ready_low = cfg.ready_high = 0; // Only progress this explicit retirement.
        cfg.scan_blocks = cfg.reclaim_blocks = repository->capacity();
        cfg.batch_blocks = 1;
        cfg.poll_ms = 1;
        kvmem::HistoricalKvGc gc(repository,
            std::make_unique<kvmem::HistoricalKvLruPolicy>(repository->capacity()), cfg);
        gc.start();
        const auto skips_before = mem.historical_active_skips();
        execute(tokens.data() + position, 32);
        position += 32;
        require(mem.historical_active_skips() > skips_before,
                "next real model active did not exclude the retired resident GPU block");
        old_reader = {};
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!repository->reclaim_completed(handle, repository->repository_id()) &&
                std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        gc.stop();
        require(repository->reclaim_completed(handle, repository->repository_id()) &&
                storage->allocated_payload_bytes() < bytes_before,
                "running GC did not release retired real-model CPU payload");
    } else if (retirement == RetirementPoint::AfterSubmission ||
               retirement == RetirementPoint::WritingAfterSubmission) {
        const bool writing = retirement == RetirementPoint::WritingAfterSubmission;
        auto repository = kvmem_transfer_test_access::capture_repository(mem);
        auto storage = kvmem_transfer_test_access::capture_storage(mem);
        require(repository && storage, "submitted retirement fixture has no capture owner");
        const auto bytes_before = storage->allocated_payload_bytes();
        kvmem::HistoricalKvGc::Config cfg;
        cfg.mode = kvmem::HistoricalKvGc::Mode::Off; // External retire; worker only reclaims.
        cfg.scan_blocks = cfg.reclaim_blocks = repository->capacity();
        cfg.batch_blocks = 1;
        cfg.poll_ms = 1;
        kvmem::HistoricalKvGc gc(repository,
            std::make_unique<kvmem::HistoricalKvLruPolicy>(repository->capacity()), cfg);
        gc.start();
        bool retired_after_submit = false, protected_old_execution = false;
        bool observed_deferred_source = false;
        int32_t retired_gpu_slot = -1;
        const auto skips_before = mem.historical_active_skips();
        const auto deferred_before = kvmem_transfer_test_access::deferred_slots(mem);
        const auto returned_before = kvmem_transfer_test_access::returned_slots(mem);
        kvmem_transfer_test_access::on_submission(mem, [&] {
            if (retired_after_submit) {
                if (writing && !observed_deferred_source) {
                    require(mem.store().blocks()[retired_during_request].gpu_slot < 0 &&
                            kvmem_transfer_test_access::retired_source_held(mem, retired_gpu_slot),
                            "next active did not exclude retired cells while retaining the exact D2H source slot");
                    observed_deferred_source = true;
                }
                return;
            }
            auto handles = kvmem_transfer_test_access::active_handles(mem);
            const auto & directory = mem.runtime().historical_shadow_directory();
            if (writing) {
                // Retire a current writer after its last GPU-producing graph
                // was submitted, before harvest queues that graph's D2H.
                // The following ubatch must run with this exact source held.
                handles.clear();
                const auto end = kvmem_transfer_test_access::submitted_end(mem);
                for (const auto & block : mem.store().blocks()) {
                    if (block.orig_pos_end() != end || block.gpu_slot < 0) continue;
                    const auto handle = directory.observe(block.block_id).handle;
                    const auto meta = repository->inspect(handle);
                    if (meta && meta->life == kvmem::HistoricalKvLife::Writing)
                        handles.push_back(handle);
                }
            }
            if (handles.empty()) return;
            const auto handle = handles.front();
            for (const auto & block : mem.store().blocks()) {
                const auto observed = directory.observe(block.block_id).handle;
                if (observed.incarnation == handle.incarnation &&
                    observed.slot == handle.slot && observed.generation == handle.generation) {
                    retired_during_request = block.block_id;
                    retired_gpu_slot = block.gpu_slot;
                    break;
                }
            }
            if (retired_during_request == UINT32_MAX) return;
            const auto before = repository->inspect(handle);
            if (!before || (writing ? !before->references : !before->active) ||
                repository->retire(handle) != kvmem::HistoricalKvRepository::RetireResult::Committed)
                return;
            const auto after = repository->inspect(handle);
            protected_old_execution = after && (writing ? after->references > 0 : after->active) &&
                repository->try_reclaim(handle) == kvmem::HistoricalKvRepository::ReclaimResult::Deferred;
            retired_after_submit = true;
        });
        const uint32_t chunk = writing ? 64 : 32;
        execute(tokens.data() + position, chunk);
        position += chunk;
        kvmem_transfer_test_access::on_submission(mem, {});
        require(retired_after_submit && protected_old_execution &&
                mem.historical_active_skips() > skips_before,
                "submitted GPU operation lost its lease or next microbatch reused retired history");
        if (writing) require(observed_deferred_source &&
                kvmem_transfer_test_access::deferred_slots(mem) == deferred_before + 1 &&
                kvmem_transfer_test_access::returned_slots(mem) == returned_before + 1 &&
                !kvmem_transfer_test_access::retired_source_held(mem, retired_gpu_slot),
                "exact retired GPU source was not deferred and returned once after ordinary D2H completion");
        else require(kvmem_transfer_test_access::deferred_slots(mem) == deferred_before,
                     "retirement waited on an unrelated block's D2H sources");
        const auto handle = mem.runtime().historical_shadow_directory().observe(retired_during_request).handle;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!repository->reclaim_completed(handle, repository->repository_id()) &&
                std::chrono::steady_clock::now() < deadline)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        gc.stop();
        require(repository->reclaim_completed(handle, repository->repository_id()) &&
                storage->allocated_payload_bytes() < bytes_before,
                "submitted-operation retirement never released its CPU payload");
    } else {
        execute(tokens.data() + position, 32);
        position += 32;
    }
    while (position < 160) {
        const uint32_t chunk = std::min(48u, 160 - position);
        execute(tokens.data() + position, chunk);
        position += chunk;
    }
    llama_kvmem_end_prefill_capture();
    for (unsigned i = 0; i < 3; ++i) {
        const auto * logits = llama_get_logits_ith(ctx, -1);
        require(logits != nullptr, "real capture graph has no logits");
        llama_token next = std::max_element(logits, logits + llama_vocab_n_tokens(vocab)) - logits;
        execute(&next, 1);
    }
    if (paused) {
        std::promise<void> returned;
        auto finished_signal = returned.get_future();
        // A watchdog makes a regression fail instead of hanging the test.
        // In the passing case ONLY request finish opens the barrier.
        auto watchdog = std::async(std::launch::async, [&, signal = std::move(finished_signal)]() mutable {
            const bool request_returned = signal.wait_for(std::chrono::seconds(5)) == std::future_status::ready;
            paused->open();
            return request_returned;
        });
        const bool finished = llama_kvmem_historical_capture_finish();
        returned.set_value();
        const bool worker_was_still_paused = watchdog.get();
        kvmem_transfer_test_access::stop_gc(mem); // Test cleanup after the proven request boundary.
        require(finished && worker_was_still_paused,
                "admitted model request or its finish waited for paused GC worker");
    } else {
        require(llama_kvmem_historical_capture_finish(), "real managed request failed completion verification");
    }
    bool cold = false;
    for (const auto & expected : saved) {
        if (expected.id == retired_during_request) continue;
        const auto & directory = mem.runtime().historical_shadow_directory();
        const auto binding = directory.find(expected.id);
        auto read = directory.acquire_read(expected.id);
        const auto layer = read.storage->read_layer(read.lease, binding->handle, expected.il);
        require(layer && hash(layer->packed_k) == expected.k && hash(layer->packed_v) == expected.v,
                "GPU eviction/reuse changed captured CPU payload");
        cold |= mem.store().blocks()[expected.id].gpu_slot < 0;
    }
    require(cold && mem.store_n_tokens() == 163, "real capture did not exercise GPU eviction");
    require(kvmem_transfer_test_access::capture_arena_bounded(mem, 163) &&
            kvmem_transfer_test_access::capture_tokens_complete(mem, 163),
            "real capture did not account for every executed position");
    if (stageout_fault) {
        require(fault.fired && fault.drains, "model did not execute the selected stageout failure path");
        std::printf("PASS real model stageout recovery: point=%s, %zu layer spans byte-equal, full request completed\n",
                    retirement == RetirementPoint::StageoutRecord ? "event-record" : "event-wait", saved.size());
        return;
    }
    if (paused) {
        std::printf("PASS real model with paused GC worker: 160 input + full 3-token output bound, request finish returned before worker resumed\n");
        return;
    }
    if (retirement != RetirementPoint::None) {
        std::printf("PASS real model active retirement: type=%s, point=%s, skipped GPU-resident historical block=%u, request completed\n",
                ggml_type_name(mem.type_k()),
                retirement == RetirementPoint::BetweenCalls ? "between-calls" :
                retirement == RetirementPoint::AfterSubmission ? "after-submission" :
                                                               "writing-after-submission/deferred-source",
                retired_during_request);
        return;
    }
    // Let LRU reclaim real graph-produced allocations. With all model
    // work drained, the background manager must return Host bytes and Ready
    // slots while leaving the adapter's existing GPU cells untouched.
    auto repository = kvmem_transfer_test_access::capture_repository(mem);
    auto storage = kvmem_transfer_test_access::capture_storage(mem);
    require(repository && storage && repository->capacity() == 12,
            "real capture owner did not retain the managed RAM repository");
    // Request finish only hands unused RAM to the cleanup worker. Wait here
    // solely to isolate the following GC-release byte measurement.
    const auto cleanup_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (repository->ready_count() != 3 && std::chrono::steady_clock::now() < cleanup_deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    const auto bytes_before_gc = storage->allocated_payload_bytes();
    const auto ready_before_gc = repository->ready_count();
    require(bytes_before_gc > 0 && ready_before_gc == 3,
            "finished request did not return its unused private RAM slots");
    const auto & resident = mem.store().blocks().back();
    require(resident.gpu_slot >= 0, "real capture has no resident GPU comparison block");
    const auto il = mem.raw().config().historical_gc_attention_layers.front();
    const auto resident_binding = mem.runtime().historical_shadow_directory().find(resident.block_id);
    auto resident_read = mem.runtime().historical_shadow_directory().acquire_read(resident.block_id);
    require(resident_binding && resident_read, "resident block was unavailable before LRU deletion");
    const auto resident_layer = resident_read.storage->read_layer(
        resident_read.lease, resident_binding->handle, il);
    require(bool(resident_layer), "resident CPU payload was unavailable before LRU deletion");
    std::vector<uint8_t> gpu_k_before(resident_layer->packed_k.size());
    auto * gpu_k = mem.get_kv()->get_k_storage(il);
    const size_t resident_cell = size_t(resident.gpu_slot) * mem.block_tokens();
    ggml_backend_tensor_get(gpu_k, gpu_k_before.data(), resident_cell * gpu_k->nb[1], gpu_k_before.size());
    resident_read = {};
    kvmem::HistoricalKvGc::Config gc_config;
    gc_config.mode = kvmem::HistoricalKvGc::Mode::Delete;
    gc_config.ready_low = 4;
    gc_config.ready_high = 5;
    gc_config.scan_blocks = 12;
    gc_config.batch_blocks = 2;
    gc_config.reclaim_blocks = 12;
    gc_config.poll_ms = 1;
    kvmem::HistoricalKvGc gc(repository,
        std::make_unique<kvmem::HistoricalKvLruPolicy>(repository->capacity()), gc_config);
    gc.start();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (repository->ready_count() < gc_config.ready_high &&
            std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    gc.stop();
    require(repository->ready_count() >= gc_config.ready_high &&
            gc.stats().committed >= 2 && gc.stats().reclaimed >= 2 &&
            storage->allocated_payload_bytes() < bytes_before_gc,
            "LRU did not physically release real model CPU payloads");
    uint32_t reclaimed_model_blocks = 0;
    for (uint32_t id = 0; id < mem.store().block_count(); ++id) {
        const auto handle = mem.runtime().historical_shadow_directory().observe(id).handle;
        if (!repository->reclaim_completed(handle, repository->repository_id())) continue;
        ++reclaimed_model_blocks;
        require(!mem.runtime().historical_shadow_directory().acquire_read(id),
                "reclaimed real-model block remained readable");
    }
    require(reclaimed_model_blocks >= 2, "LRU released no exact real-model generations");
    std::vector<uint8_t> gpu_k_after(gpu_k_before.size());
    ggml_backend_tensor_get(gpu_k, gpu_k_after.data(), resident_cell * gpu_k->nb[1], gpu_k_after.size());
    require(gpu_k_after == gpu_k_before,
            "background Host GC changed the resident GPU working set");
    auto extra = llama_batch_get_one(tokens.data(), 1);
    require(llama_decode(ctx, extra) != 0 && mem.store_n_tokens() == 163,
            "finished capture session accepted another append");
    const auto failed_context = kvmem_transfer_test_access::batch_context(mem);
    require(failed_context.storage == batch_context.storage &&
            failed_context.bytes == batch_context.bytes && !failed_context.in_use,
            "failed decode consumed or leaked the prepared context slot");
    std::printf("PASS real model historical capture and post-request LRU release: type=%s, 160 input + 3 generated tokens, 16-token microbatches, %zu layer spans byte-equal, reclaimed=%u, released_bytes=%llu, GPU unchanged\n",
                 ggml_type_name(mem.type_k()), saved.size(), reclaimed_model_blocks,
                 (unsigned long long) (bytes_before_gc - storage->allocated_payload_bytes()));
}

static void check_context_close_with_paused_worker(
        std::unique_ptr<llama_context, decltype(&llama_free)> & ctx, llama_memory_kvmem & mem) {
    auto gate = std::make_shared<HistoricalPolicyGate>();
    struct Unpause { std::shared_ptr<HistoricalPolicyGate> gate; ~Unpause() { gate->open(); } } unpause{gate};
    std::weak_ptr<kvmem::HistoricalKvRepository> repository =
        kvmem_transfer_test_access::capture_repository(mem);
    std::weak_ptr<kvmem::HistoricalKvRawStore> storage =
        kvmem_transfer_test_access::capture_storage(mem);
    kvmem_transfer_test_access::install_gc_policy(mem, std::make_unique<PausedHistoricalPolicy>(gate));
    require(gate->entered_signal.wait_for(std::chrono::seconds(5)) == std::future_status::ready,
            "close fixture worker did not reach policy barrier");
    std::promise<void> returned;
    auto signal = returned.get_future();
    auto watchdog = std::async(std::launch::async, [gate, signal = std::move(signal)]() mutable {
        const bool did_return = signal.wait_for(std::chrono::seconds(5)) == std::future_status::ready;
        gate->open();
        return did_return;
    });
    // Exercise actual llama/context/adapter destruction on the GPU owner.
    ctx.reset();
    const bool retained_while_paused = !repository.expired() && !storage.expired();
    returned.set_value();
    const bool closed_without_wait = watchdog.get();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while ((!repository.expired() || !storage.expired()) && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    require(closed_without_wait && retained_while_paused && repository.expired() && storage.expired(),
            "context close joined paused GC, lost CPU ownership early, or leaked the closed domain");
    std::puts("PASS real context teardown: returned before paused GC resumed, CPU owners retained then released on cleanup worker");
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "Usage: %s model.gguf (requires CUDA)\n", argv[0]);
        return argc == 1 ? 77 : 1;
    }
    try {
        llama_backend_init();
        ggml_backend_load_all();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 99;
        mp.load_mtp = false;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(argv[1], mp), llama_model_free);
        require(bool(model), "model load failed");
        llama_kvmem_params kp{};
        kp.enabled = true;
        kp.method = 1;
        kp.block_tokens = 32;
        kp.budget = kp.gen_reserve = 128;
        kp.query_begin = kp.query_end = kp.force_pos = -1;
        llama_kvmem_set_params(&kp);
        auto cp = llama_context_default_params();
        cp.n_ctx = 512;
        cp.n_batch = cp.n_ubatch = 64;
        cp.n_seq_max = 1;
        cp.type_k = cp.type_v = GGML_TYPE_F16;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        for (unsigned scenario = 0; scenario < 35; ++scenario) {
            if ((scenario >= 3 && scenario <= 12) || scenario >= 30) {
                kp.method = 0;
                kp.budget = 96;
                llama_kvmem_set_params(&kp);
                cp.n_ubatch = 16;
                cp.type_k = cp.type_v = scenario == 4 ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;
            } else if (scenario >= 13) {
                kp.method = 1;
                kp.budget = 128;
                llama_kvmem_set_params(&kp);
                cp.n_ubatch = 64;
                cp.type_k = cp.type_v = GGML_TYPE_F16;
            }
            std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
                llama_init_from_model(model.get(), cp), llama_free);
            require(bool(ctx), "context init failed");
            auto * mem = dynamic_cast<llama_memory_kvmem *>(llama_get_memory(ctx.get()));
            if (auto * hybrid = dynamic_cast<llama_memory_kvmem_hybrid *>(llama_get_memory(ctx.get())))
                mem = hybrid->attn_kvmem();
            require(mem != nullptr, "missing KVMem adapter");
            if (scenario >= 31) check_request_capture(ctx.get(), model.get(), *mem, RetirementPoint::None, false, scenario - 30);
            else if (scenario == 30) check_request_capture(ctx.get(), model.get(), *mem, RetirementPoint::None, true);
            else if (scenario == 13) check_capture_start_failure(*mem);
            else if (scenario >= 14) check_capture_failure(*mem, scenario);
            else if (scenario >= 3) check_request_capture(ctx.get(), model.get(), *mem,
                scenario == 5 ? RetirementPoint::BetweenCalls :
                scenario == 6 ? RetirementPoint::AfterSubmission :
                scenario == 7 ? RetirementPoint::WritingAfterSubmission :
                scenario == 8 ? RetirementPoint::PausedWorker :
                scenario == 10 ? RetirementPoint::StageoutRecord :
                scenario == 11 ? RetirementPoint::StageoutWait :
                scenario == 12 ? RetirementPoint::StageoutUnconfirmed : RetirementPoint::None);
            else if (scenario == 2) check_partial_rollover(*mem);
            else if (scenario == 1) check_managed_async_harvest(*mem);
            else {
                check_mean_ranges(*mem);
                check_access(*mem);
            }
            if (scenario == 9) check_context_close_with_paused_worker(ctx, *mem);
            if (scenario == 12 || scenario >= 32) {
                std::weak_ptr<kvmem::HistoricalKvRawStore> storage = kvmem_transfer_test_access::capture_storage(*mem);
                ctx.reset();
                const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
                while (!storage.expired() && std::chrono::steady_clock::now() < deadline)
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                require(storage.expired(), "failed real-model domain leaked captured history after checked teardown");
                std::puts("PASS failed real-model teardown: captured CPU owner released after device cleanup");
            }
            std::printf("PASS historical model scenario %u/35\n", scenario + 1);
        }
        kp.method = 0;
        kp.budget = 96;
        llama_kvmem_set_params(&kp);
        cp.n_ubatch = 16;
        cp.type_k = cp.type_v = GGML_TYPE_F16;
        std::unique_ptr<llama_context, decltype(&llama_free)> backend_guard_ctx(
            llama_init_from_model(model.get(), cp), llama_free);
        require(bool(backend_guard_ctx), "backend guard context init failed");
        auto * backend_guard_mem = dynamic_cast<llama_memory_kvmem *>(llama_get_memory(backend_guard_ctx.get()));
        require(backend_guard_mem &&
                llama_kvmem_historical_capture_begin(backend_guard_ctx.get(), 12, 16, 0, 1),
                "backend guard capture admission failed");
        const auto backend_before = backend_guard_ctx->get_historical_backend_status();
        llama_set_causal_attn(backend_guard_ctx.get(), false);
        bool rejected_backend_growth = false;
        try {
            backend_guard_ctx->sched_reserve();
        } catch (const std::runtime_error &) {
            rejected_backend_growth = true;
        }
        const auto backend_after = backend_guard_ctx->get_historical_backend_status();
        require(rejected_backend_growth &&
                kvmem_transfer_test_access::capture_failed(*backend_guard_mem) &&
                backend_after.scheduler == backend_before.scheduler &&
                backend_after.output_buffer == backend_before.output_buffer &&
                backend_after.output_bytes == backend_before.output_bytes &&
                backend_after.graph_results == backend_before.graph_results &&
                backend_after.graph_meta == backend_before.graph_meta &&
                backend_after.needs_scheduler_reserve,
                "admitted backend change allocated or left capture open");
        std::puts("PASS historical backend guard: scheduler growth denied and capture closed");
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL historical adapter access: %s\n", e.what());
        return 1;
    }
    llama_backend_free();
    return 0;
}
