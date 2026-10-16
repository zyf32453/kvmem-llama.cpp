// Run manually with a local GGUF and CUDA. This binds one completed block in
// a controlled fixture; it does not enable background GC in CLI/server runs.
#include "llama-memory-kvmem-hybrid.h"
#include "llama-batch.h"
#include "llama-kvmem-hooks.h"
#include "ggml-backend.h"

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
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(
            llama_init_from_model(model.get(), cp), llama_free);
        require(bool(ctx), "context init failed");
        auto * mem = dynamic_cast<llama_memory_kvmem *>(llama_get_memory(ctx.get()));
        if (auto * hybrid = dynamic_cast<llama_memory_kvmem_hybrid *>(llama_get_memory(ctx.get())))
            mem = hybrid->attn_kvmem();
        require(mem != nullptr, "missing KVMem adapter");
        check_access(*mem);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "FAIL historical adapter access: %s\n", e.what());
        return 1;
    }
    llama_backend_free();
    return 0;
}
