// Run manually with a local GGUF and CUDA. This binds one completed block in
// a controlled fixture; it does not enable background GC in CLI/server runs.
#include "llama-memory-kvmem-hybrid.h"
#include "llama-batch.h"
#include "llama-kvmem-hooks.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "kvmem/historical_kv_raw_store.hpp"

#include <algorithm>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

struct kvmem_transfer_test_access {
    static void restore(llama_memory_kvmem & mem, uint32_t id) {
        mem.write_block_to_gpu(id);
    }
    static void capture(llama_memory_kvmem & mem, uint32_t id) { mem.harvest_gpu_v(id); }
    static void commit(llama_memory_kvmem & mem) { mem.harvest_gpu_v_commit(); }
    static size_t pending(const llama_memory_kvmem & mem) {
        return mem.harvest_v_jobs_.size() + mem.harvest_v_pending_.jobs.size();
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

    MeanTensors(llama_memory_kvmem & mem, uint32_t begin, uint32_t count, bool host, ggml_type type) {
        const auto & cfg = mem.raw().config();
        context.reset(ggml_init({cfg.historical_gc_attention_layers.size() * ggml_tensor_overhead() + 1024,
                                 nullptr, true}));
        require(bool(context), "Mean-K tensor context allocation failed");
        for (uint32_t il : cfg.historical_gc_attention_layers) {
            (void) il;
            layers.push_back(ggml_new_tensor_3d(context.get(), type, cfg.n_embd_k, 1, count));
        }
        auto buft = host ? ggml_backend_cpu_buffer_type() : ggml_backend_buffer_get_type(
            mem.get_kv()->get_k_storage(cfg.historical_gc_attention_layers.front())->buffer);
        buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(context.get(), buft));
        require(bool(buffer), "Mean-K tensor buffer allocation failed");
        for (size_t i = 0; i < layers.size(); ++i) {
            std::vector<float> values(static_cast<size_t>(count) * cfg.n_embd_k);
            for (uint32_t t = 0; t < count; ++t)
                for (uint32_t d = 0; d < cfg.n_embd_k; ++d)
                    values[t * cfg.n_embd_k + d] = float(begin + t + 1) + float(d % 8) / 8;
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
    require(runtime.bind_historical_full_block(0, repository, handle, *writer),
            "could not bind adapter block identity");

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
    require(repository->inspect(handle)->references == 1,
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
        require(runtime.bind_historical_full_block(0, repository, handle, *writer, payload),
                "managed payload directory binding failed");
        MeanTensors means(mem, 0, block, round == 3, round < 2 ? GGML_TYPE_F16 : GGML_TYPE_F32);
        kvmem_transfer_test_access::mean_inputs(mem, means.layers, 0, block);
        if (round < 2) {
            require(kvmem_transfer_test_access::mean_submit(mem), "asynchronous Mean-K submit failed");
        } else if (round == 2) {
            kvmem_transfer_test_access::mean_sync(mem);
        } else {
            // Only 8 accepted tokens from a 32-token draft enter the sum.
            kvmem_transfer_test_access::mean_decode(mem, 8);
            require(payload->write_progress(directory.find(0)->writer_context->writer(), handle,
                                            cfg.attention_layers.front())->mean_k_tokens == 0,
                    "decode accumulator flushed speculative tokens prematurely");
        }
        kvmem_transfer_test_access::capture(mem, 0);
        require(kvmem_transfer_test_access::pending(mem) > 0, "test did not enqueue asynchronous packed writes");
        const bool retire_in_flight = round == 1 || round == 4;
        auto reclaim = [&] {
            auto result = Repository::ReclaimResult::Stale;
            std::thread worker([&] { result = repository->try_reclaim(handle); });
            worker.join();
            return result;
        };
        if (retire_in_flight) {
            require(repository->retire(handle) == Repository::RetireResult::Committed &&
                    runtime.unbind_retired_historical_block(0), "in-flight writer retirement failed");
            require(reclaim() == Repository::ReclaimResult::Deferred,
                    "queued DMA writer failed to retain its allocation");
        }
        kvmem_transfer_test_access::commit(mem);
        if (retire_in_flight) require(reclaim() == Repository::ReclaimResult::Deferred,
                "Mean-K task/accumulator lost its writer after packed completion");
        if (round < 2) {
            mem.harvest_flush();
            if (round == 0) {
                // Replaying an already committed capture while still Writing
                // must not count its tokens again.
                kvmem_transfer_test_access::mean_inputs(mem, means.layers, 0, block);
                kvmem_transfer_test_access::mean_sync(mem);
            }
        } else if (round == 3) {
            // Host accumulator followed by GPU inputs must preserve both sums.
            MeanTensors suffix(mem, 8, block - 8, false, GGML_TYPE_F32);
            kvmem_transfer_test_access::mean_inputs(mem, suffix.layers, 8, block - 8);
            kvmem_transfer_test_access::mean_decode(mem, block - 8);
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
            require(payload->publish_complete(*repository, directory.find(0)->writer_context->writer(), handle),
                    "completed managed capture could not publish");
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
        require(reclaim() == Repository::ReclaimResult::Complete && payload->allocated_payload_bytes() == 0 &&
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
        for (bool managed : {false, true}) {
            std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
                llama_init_from_model(model.get(), cp), llama_free);
            require(bool(ctx), "context init failed");
            auto * mem = dynamic_cast<llama_memory_kvmem *>(llama_get_memory(ctx.get()));
            if (auto * hybrid = dynamic_cast<llama_memory_kvmem_hybrid *>(llama_get_memory(ctx.get())))
                mem = hybrid->attn_kvmem();
            require(mem != nullptr, "missing KVMem adapter");
            if (managed) check_managed_async_harvest(*mem);
            else {
                check_mean_ranges(*mem);
                check_access(*mem);
            }
        }
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL historical adapter access: %s\n", e.what());
        return 1;
    }
    llama_backend_free();
    return 0;
}
