#include "llama-kvmem-diag.h"
#include "llama-memory-kvmem.h"
#include "kvmem/historical_kv_raw_store.hpp"
#include "kvmem/historical_kv_capture_session.hpp"
#include "kvmem/historical_kv_active_set.hpp"
#include "kvmem/historical_kv_gc.hpp"
#include "llama-memory-kvmem-hybrid.h"
#include "llama-memory-kvmem-mtp.h"

#include "llama-kvmem-batch.h"
#include "llama-kvmem-capture.h"
#include "llama-kvmem-execution.h"
#include "llama-kvmem-factory.h"
#include "llama-kvmem-hooks.h"
#include "llama-kvmem-gpu.h"
#include "llama-kvmem-quant.h"
#include "llama-kvmem-stagein.h"
#include "llama-kvmem-transfer.h"

#include "llama-arch.h"
#include "llama-cparams.h"
#include "llama-context.h"
#include "llama-impl.h"
#include "llama-memory-recurrent.h"
#include "llama-memory-hybrid.h"
#include "llama-model.h"

#include "llama.h"
#include "kvmem/snapshot.hpp"
#include "kvmem/snapshot_buffer.hpp"

#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-cuda.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <map>
#include <mutex>
#include <new>
#include <set>
#include <stdexcept>
#include <thread>


struct llama_memory_kvmem::GdnReplay {
    struct Group {
        ggml_backend_buffer_ptr descriptors;
        cudaStream_t stream = nullptr;
        int layers = 0;
        int device = -1;
        ~Group() {
            if (stream) {
                cudaSetDevice(device);
                cudaStreamSynchronize(stream);
                cudaStreamDestroy(stream);
            }
        }
    };
    std::vector<std::unique_ptr<Group>> groups;
    uint64_t folds = 0;
    int64_t fold_us = 0;
};

// One conversation's host KV. Moved out of the memory object at a store swap
// and moved back in at the next one, so the object is never observable without
// a store. KvMemRuntime owns NVMe prefetch futures and RawKvStore owns a mutex,
// a condition variable and an io thread, so both travel as unique_ptr, which is
// how they are already held.
struct llama_memory_kvmem::ConvStore {
    const llama_model * model = nullptr;
    ggml_type type_k = GGML_TYPE_F16, type_v = GGML_TYPE_F16;
    uint32_t slots = 0;
    bool has_mtp = false;
    bool cold = false;
    std::unique_ptr<kvmem::KvMemRuntime> runtime;
    std::unique_ptr<kvmem::RawKvStore>   raw;
    std::vector<RowPosition>             row_positions;
    std::vector<std::vector<float>>      q_sum;
    std::vector<uint32_t>                q_count;
    // Follower mirror, swapped in lockstep: it is keyed by the trunk's block
    // ids and is meaningless beside another conversation's store.
    std::unique_ptr<kvmem::RawKvStore>   mtp_raw;
    // Block ids that were GPU-resident at detach, ascending.
    std::vector<uint32_t>                resident;
};

struct llama_kvmem_store_bundle {
    std::unique_ptr<llama_memory_kvmem::ConvStore> store;
};

// Detached stores, keyed by an opaque handle. Handle 0 is the store the
// constructor built, which is all a default server ever uses: the pool stays
// empty until llama_kvmem_store_create() asks for a second one.
struct kvmem_conv_entry {
    int32_t id = 0;
    // Null for the active handle: that store's members live on the memory
    // object itself.
    std::unique_ptr<llama_memory_kvmem::ConvStore> store;
};

struct kvmem_conv_pool {
    llama_memory_kvmem * owner = nullptr;
    std::vector<kvmem_conv_entry> entries;
    int32_t active = 0;
    int32_t next_id = 1;
};

static thread_local llama_kvmem_execution_state * active_execution = nullptr;

llama_kvmem_execution_state::llama_kvmem_execution_state()
    : conversations(std::make_unique<kvmem_conv_pool>()), gpu(kvmem_gpu_state_create()) {}

llama_kvmem_execution_state::~llama_kvmem_execution_state() {
    auto * previous = llama_kvmem_execution_exchange(this);
    GGML_ASSERT(!memory && !mtp && "free target/draft contexts before their execution state");
    kvmem_gpu_state_free(gpu);
    llama_kvmem_execution_exchange(previous);
}

llama_kvmem_execution_state & kvmem_current_execution() {
    static llama_kvmem_execution_state legacy;
    return active_execution ? *active_execution : legacy;
}

llama_kvmem_execution_state * llama_kvmem_execution_create() {
    return new llama_kvmem_execution_state;
}

llama_kvmem_execution_state * llama_kvmem_execution_exchange(llama_kvmem_execution_state * state) {
    auto * previous = active_execution;
    active_execution = state;
    return previous;
}

void llama_kvmem_execution_free(llama_kvmem_execution_state * state) {
    if (!state) return;
    GGML_ASSERT(state != active_execution && "deactivate before freeing execution state");
    delete state;
}

// ~KvMemRuntime does not drain start_prefetch's futures; only std::async's
// blocking future destructor does. truncate_to(0) waits for them first. Every
// path that drops a detached bundle goes through here: a bundle's runtime also
// holds a KvMemBackend pointing at its owner's SlotBackend, so dropping one in
// place after the owner changed would free against a stale backend.
static void kvmem_conv_release(std::unique_ptr<llama_memory_kvmem::ConvStore> & store) {
    if (!store) {
        return;
    }
    if (store->runtime) {
        store->runtime->truncate_to(0);
    }
    store.reset();
}

static void kvmem_conv_pool_unbind(llama_memory_kvmem * mem) {
    if ((*kvmem_current_execution().conversations).owner != mem) {
        return;
    }
    for (auto & e : (*kvmem_current_execution().conversations).entries) {
        kvmem_conv_release(e.store);
    }
    (*kvmem_current_execution().conversations) = kvmem_conv_pool{};
}

static kvmem_conv_pool * kvmem_conv_pool_bind(llama_memory_kvmem * mem) {
    if (!mem) {
        return nullptr;
    }
    if ((*kvmem_current_execution().conversations).owner != mem) {
        // Reached only if a second memory object arms while an earlier one
        // still holds detached bundles. Release them the way destroy and
        // unbind do rather than dropping them where they lie; unbind() leaves
        // the pool freshly default-constructed.
        kvmem_conv_pool_unbind((*kvmem_current_execution().conversations).owner);
        (*kvmem_current_execution().conversations).owner = mem;
        (*kvmem_current_execution().conversations).entries.emplace_back();
    }
    return &(*kvmem_current_execution().conversations);
}

static kvmem_conv_entry * kvmem_conv_find(int32_t id) {
    for (auto & e : (*kvmem_current_execution().conversations).entries) {
        if (e.id == id) {
            return &e;
        }
    }
    return nullptr;
}

void llama_memory_kvmem::set_recurrent(llama_memory_recurrent * recr) {
    recr_ = recr;
    if (!recr) return;
    size_t recurrent_bytes = 0, conv_bytes = 0;
    for (size_t il = 0; il < recr->r_l.size(); ++il) {
        if (recr->r_l[il]) conv_bytes += ggml_nbytes(recr->r_l[il]);
        if (recr->s_l[il]) recurrent_bytes += ggml_nbytes(recr->s_l[il]);
    }
    const size_t planes = recr->n_rs_seq + 1;
    kvmem_diag("KVMEM_GDN_ALLOCATION mode=%s recurrent_bytes=%zu conv_bytes=%zu rollback_bytes=%zu\n",
            recr->replay_capacity ? "replay" : "snapshots", recurrent_bytes / planes, conv_bytes / planes,
            (recurrent_bytes + conv_bytes) / planes * (planes - 1));
    if (!recr->replay_capacity) return;
    auto replay = std::make_unique<GdnReplay>();
    struct GroupBuild {
        std::vector<ggml_cuda_gdn_replay_layer> layers;
        ggml_backend_buffer_type_t buft = nullptr;
        size_t states = 0;
        size_t records = 0;
    };
    std::map<int, GroupBuild> grouped;
    for (size_t il = 0; il < recr->r_l.size(); ++il) {
        if (!recr->r_l[il]) continue;
        const auto & r = recr->replay_l[il];
        ggml_cuda_gdn_replay_layer layer = {static_cast<float *>(recr->s_l[il]->data), static_cast<float *>(recr->r_l[il]->data),
                static_cast<float *>(r[0]->data), static_cast<float *>(r[1]->data), static_cast<float *>(r[2]->data),
                static_cast<float *>(r[3]->data), static_cast<float *>(r[4]->data)};
        const void * ptrs[] = {layer.state, layer.conv, layer.key, layer.value, layer.gate, layer.beta, layer.conv_input};
        int device = -1;
        for (const void * ptr : ptrs) {
            cudaPointerAttributes attrs{};
            if (!ptr || cudaPointerGetAttributes(&attrs, ptr) != cudaSuccess || attrs.type != cudaMemoryTypeDevice ||
                    (device >= 0 && device != attrs.device)) {
                throw std::runtime_error("GDN replay layer has missing or cross-device state/record tensors");
            }
            device = attrs.device;
        }
        auto & group = grouped[device];
        group.layers.push_back(layer);
        for (auto * t : r) group.records += ggml_nbytes(t);
        group.states += ggml_nbytes(recr->r_l[il]) + ggml_nbytes(recr->s_l[il]);
        if (!group.buft) group.buft = ggml_backend_buffer_get_type(recr->s_l[il]->buffer);
    }
    if (grouped.empty()) throw std::runtime_error("GDN replay has no recurrent layers");
    for (auto & [device, build] : grouped) {
        auto group = std::make_unique<GdnReplay::Group>();
        group->device = device;
        group->layers = static_cast<int>(build.layers.size());
        const size_t descriptor_bytes = build.layers.size() * sizeof(build.layers[0]);
        if (cudaSetDevice(device) != cudaSuccess) throw std::runtime_error("cannot select GDN replay device");
        group->descriptors.reset(ggml_backend_buft_alloc_buffer(build.buft, descriptor_bytes));
        cudaPointerAttributes attrs{};
        const void * dst = group->descriptors ? ggml_backend_buffer_get_base(group->descriptors.get()) : nullptr;
        if (!dst || cudaPointerGetAttributes(&attrs, dst) != cudaSuccess || attrs.device != device ||
                cudaStreamCreateWithFlags(&group->stream, cudaStreamNonBlocking) != cudaSuccess ||
                cudaMemcpy(const_cast<void *>(dst), build.layers.data(), descriptor_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            throw std::runtime_error("cannot allocate GDN replay descriptors on their layer GPU");
        }
        kvmem_diag("KVMEM_GDN_MEMORY mode=replay device=%d layers=%d capacity=%u state_bytes=%zu record_bytes=%zu descriptor_bytes=%zu\n",
                device, group->layers, recr->replay_capacity, build.states, build.records, descriptor_bytes);
        replay->groups.push_back(std::move(group));
    }
    gdn_replay_ = std::move(replay);
}

bool llama_memory_kvmem::gdn_replay_enabled() const {
    return gdn_replay_ != nullptr;
}

bool llama_memory_kvmem::gdn_replay_begin(llama_pos start, uint32_t width) {
    return gdn_replay_ && recr_->replay_begin(start, width);
}

bool llama_memory_kvmem::gdn_replay_commit(llama_context * ctx, uint32_t n_keep) {
    if (!gdn_replay_ || !recr_->replay_recording || n_keep > recr_->replay_width) return false;
    llama_synchronize(ctx);
    auto & replay = *gdn_replay_;
    const int64_t started = ggml_time_us();
    std::vector<GdnReplay::Group *> launched;
    bool ok = true;
    for (const auto & group : replay.groups) {
        const auto * layers = static_cast<const ggml_cuda_gdn_replay_layer *>(ggml_backend_buffer_get_base(group->descriptors.get()));
        if (cudaSetDevice(group->device) != cudaSuccess ||
                !ggml_backend_cuda_gdn_fold(layers, group->layers, n_keep, recr_->replay_capacity, group->stream)) {
            ok = false;
            break;
        }
        launched.push_back(group.get());
    }
    for (auto * group : launched) {
        if (cudaSetDevice(group->device) != cudaSuccess || cudaStreamSynchronize(group->stream) != cudaSuccess) ok = false;
    }
    if (!ok) {
        recr_->replay_poisoned = true;
        recr_->replay_finish(0);
        return false;
    }
    recr_->replay_finish(n_keep);
    replay.fold_us += ggml_time_us() - started;
    replay.folds += n_keep != 0;
    return true;
}
static std::atomic<uint64_t> transfer_bytes[3]{};
static std::atomic<uint64_t> transfer_calls[3]{};

static bool transfer_stats_enabled() {
    static const bool enabled = [] {
        const char * e = std::getenv("KVMEM_PERF");
        return e && e[0] && e[0] != '0';
    }();
    return enabled;
}

void kvmem_record_transfer(int kind, uint64_t bytes) {
    if (!transfer_stats_enabled() || kind < 1 || kind > 3 || !bytes) return;
    transfer_bytes[kind - 1].fetch_add(bytes, std::memory_order_relaxed);
    transfer_calls[kind - 1].fetch_add(1, std::memory_order_relaxed);
}

llama_kvmem_transfer_stats llama_kvmem_get_transfer_stats() {
    llama_kvmem_transfer_stats stats;
    stats.enabled = transfer_stats_enabled();
    for (int i = 0; i < 3; ++i) {
        stats.bytes[i] = transfer_bytes[i].load(std::memory_order_relaxed);
        stats.calls[i] = transfer_calls[i].load(std::memory_order_relaxed);
    }
    return stats;
}

struct llama_memory_kvmem::CaptureD2hPipe {
    struct Item {
        int il = 0;
        char which = 0;
        size_t offset = 0;
        size_t nbytes = 0;
        int64_t d = 0;
        int64_t h = 0;
        int64_t n = 0;
        size_t nb0 = 0;
        size_t nb1 = 0;
        size_t nb2 = 0;
        ggml_type type = GGML_TYPE_F16;
        bool from_gpu = false;
        bool committed = false;
    };
    struct Slot {
        uint8_t * gpu = nullptr;
        uint8_t * pin = nullptr;
        size_t cap = 0;
        cudaEvent_t done = nullptr;
        std::atomic<bool> inflight{false};
        // Owned by the foreground until publication, then by the worker.
        // The foreground may touch it again only after completion or join.
        bool device_pending = false;
        std::vector<Item> items;
        CaptureBatch capture;
    };
    cudaStream_t stream = nullptr;
    cudaEvent_t snap = nullptr;
    ggml_backend_event_t compute_done = nullptr;
    ggml_backend_event_t snap_be = nullptr;
    Slot slots[2];
    int next = 0;
    int device = 0;
    bool ok = false;
};

struct llama_memory_kvmem::MultiD2hPipe {
    struct Device {
        int id = -1;
        cudaStream_t stream = nullptr;
        uint8_t * pin = nullptr;
        size_t cap = 0;
        size_t used = 0;
    };
    struct Item {
        CaptureD2hPipe::Item capture;
        size_t device = 0;
        const uint8_t * src = nullptr;
    };
    std::vector<Device> devices;
    std::vector<Item> items;
};

static bool kvmem_cuda_ok(cudaError_t e, const char * what) {
    if (e == cudaSuccess) {
        return true;
    }
    fprintf(stderr, "KVMEM D2H %s: %s\n", what, cudaGetErrorString(e));
    return false;
}

static uint8_t * kvmem_cuda_tensor_ptr(ggml_tensor * t) {
    if (!t || !t->data) {
        return nullptr;
    }
    ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
    if (!buf || ggml_backend_buffer_is_host(buf) || ggml_backend_buffer_is_meta(buf)) {
        return nullptr;
    }
    return static_cast<uint8_t *>(t->data);
}

static void kvmem_stagein_flush_sync(int64_t * copy_us, int64_t * rope_us,
                                     int64_t * hadamard_us, int64_t * set_us) {
    kvmem_stagein_flush(copy_us, rope_us, hadamard_us, set_us);
    const int64_t t0 = ggml_time_us();
    kvmem_stagein_sync();
    if (set_us) {
        *set_us += ggml_time_us() - t0;
    }
}

static bool kvmem_harvest_sync_old() {
    const char * e = getenv("KVMEM_HARVEST_SYNC");
    return e && e[0] != '\0' && e[0] != '0';
}

static cudaEvent_t kvmem_ggml_cuda_event(ggml_backend_event_t ev) {
    return ev ? static_cast<cudaEvent_t>(ev->context) : nullptr;
}

static bool kvmem_env_perf() {
    static int v = -1;
    if (v < 0) {
        const char * e = getenv("KVMEM_PERF");
        v = (e && e[0] != '\0' && e[0] != '0') ? 1 : 0;
    }
    return v == 1;
}

void llama_kvmem_set_params(const struct llama_kvmem_params * params) {
    if (params) {
        kvmem_current_execution().params = *params;
    } else {
        kvmem_current_execution().params = {};
    }
}

const struct llama_kvmem_params * llama_kvmem_get_params(void) {
    return &kvmem_current_execution().params;
}

static uint32_t kvmem_align_tokens(uint32_t tokens, uint32_t block_tokens) {
    if (block_tokens == 0) {
        return 0;
    }
    if (tokens < block_tokens) {
        return block_tokens;
    }
    return (tokens / block_tokens) * block_tokens;
}

static double kvmem_default_ratio(float v, double fallback) {
    return v > 0.0f ? static_cast<double>(v) : fallback;
}

static uint64_t kvmem_first_gpu_total_bytes() {
    const size_t n = ggml_backend_dev_count();
    for (size_t i = 0; i < n; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            size_t free_m = 0;
            size_t total_m = 0;
            ggml_backend_dev_memory(dev, &free_m, &total_m);
            return static_cast<uint64_t>(total_m);
        }
    }
    return 0;
}

static kvmem::KvMemRuntimeConfig make_runtime_cfg(
        uint32_t block_tokens,
        uint32_t budget,
        uint32_t sink_tokens,
        uint32_t recent_tokens,
        uint64_t block_bytes) {
    kvmem::KvMemRuntimeConfig cfg;
    cfg.store.block_tokens = block_tokens;
    cfg.store.select_budget = budget;
    cfg.store.prefill_budget = budget;
    cfg.store.select_method = kvmem_current_execution().params.method == 1
        ? kvmem::KvMemMethod::Retrieval
        : kvmem::KvMemMethod::Recency;
    cfg.store.sink_blocks = sink_tokens == 0
        ? 1u
        : std::max(1u, sink_tokens / block_tokens);
    cfg.store.recent_blocks = recent_tokens / block_tokens;
    cfg.store.gen_budget = 0;
    cfg.store.gpu_memory_ratio = kvmem_default_ratio(kvmem_current_execution().params.gpu_memory_ratio, 0.50);
    cfg.store.gpu_high_watermark = kvmem_default_ratio(kvmem_current_execution().params.gpu_high_watermark, 0.95);
    cfg.store.gpu_low_watermark = kvmem_default_ratio(kvmem_current_execution().params.gpu_low_watermark, 0.85);
    cfg.store.estimated_block_bytes = block_bytes;
    cfg.store.gpu_resident_block_bytes = block_bytes;
    cfg.cpu_bytes = kvmem_current_execution().params.cpu_bytes;
    // GPU-block NVMe spill is separate from raw-K NVMe. When raw-K owns the
    // SSD budget, skip the 2 MiB GPU-format copies (V already lives in raw).
    if (!kvmem_current_execution().params.raw_k_nvme) {
        cfg.nvme_bytes = kvmem_current_execution().params.nvme_bytes;
        if (kvmem_current_execution().params.nvme_dir && kvmem_current_execution().params.nvme_dir[0]) {
            cfg.nvme_dir = kvmem_current_execution().params.nvme_dir;
        } else if (cfg.nvme_bytes > 0) {
            cfg.nvme_dir = "/tmp/kvmem_nvme";
        }
    }
    return cfg;
}

static bool kvmem_cache_has_layer(const llama_kv_cache * kv, int32_t il) {
    if (!kv) {
        return false;
    }
    for (uint32_t id : kv->get_layer_ids()) {
        if (static_cast<int32_t>(id) == il) {
            return true;
        }
    }
    return false;
}

static uint32_t kvmem_first_attn_layer(const llama_model & model) {
    const uint32_t n = model.hparams.n_layer();
    for (uint32_t il = 0; il < n; ++il) {
        if (!model.hparams.is_recr(il)) {
            return il;
        }
    }
    return 0;
}

static uint32_t kvmem_n_attn_layers(const llama_model & model) {
    const uint32_t n = model.hparams.n_layer();
    uint32_t c = 0;
    for (uint32_t il = 0; il < n; ++il) {
        if (!model.hparams.is_recr(il)) {
            ++c;
        }
    }
    return c == 0 ? n : c;
}

struct kvmem_pool_plan {
    uint32_t block_tokens = 32;
    uint32_t budget = 0;
    uint32_t gen_reserve = 0;
    uint32_t kv_size = 0;
    uint32_t n_slots = 0;
    uint64_t block_bytes = 0;
    uint32_t cap_blocks = 0;
    uint64_t gpu_total = 0;
};

static bool kvmem_tensor_multi_gpu(const llama_model & model) {
    return model.split_mode() == LLAMA_SPLIT_MODE_TENSOR && model.get_split_state_ud.n_devices > 1;
}

static std::vector<ggml_backend_dev_t> kvmem_tensor_devices(const llama_model & model) {
    if (model.devices.size() != 1) throw std::runtime_error("tensor model has no unique Meta device");
    auto * meta = model.devices[0].dev;
    if (ggml_backend_dev_type(meta) != GGML_BACKEND_DEVICE_TYPE_META) {
        throw std::runtime_error("tensor model device is not Meta");
    }
    const size_t count = ggml_backend_meta_device_count(meta);
    if (count != model.get_split_state_ud.n_devices) {
        throw std::runtime_error("tensor Meta physical GPU count does not match split metadata");
    }
    std::vector<ggml_backend_dev_t> result;
    for (size_t i = 0; i < count; ++i) {
        auto * dev = ggml_backend_meta_device_get(meta, i);
        if (!dev || ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU ||
                std::string(ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev))) != "CUDA" ||
                std::find(result.begin(), result.end(), dev) != result.end()) {
            throw std::runtime_error("tensor Meta physical GPU list is invalid");
        }
        result.push_back(dev);
    }
    return result;
}

static std::vector<uint64_t> kvmem_tensor_kv_row_bytes(
        const llama_model & model, ggml_type type_k, ggml_type type_v, bool include_mtp) {
    const size_t n_devices = model.get_split_state_ud.n_devices;
    std::vector<uint64_t> bytes(n_devices, 0);
    const uint32_t n_layers = include_mtp ? model.hparams.n_layer_all : model.hparams.n_layer();
    ggml_init_params ip = {
        /*.mem_size   =*/ 2 * n_layers * ggml_tensor_overhead() + 1024,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx{ggml_init(ip)};
    if (!ctx) throw std::runtime_error("cannot allocate tensor KV split metadata");
    for (uint32_t il = 0; il < n_layers; ++il) {
        if (model.hparams.is_recr(il) || !model.hparams.has_kv(il)) continue;
        for (bool is_k : {true, false}) {
            // The draft context can choose a wider cache type than the target.
            const ggml_type type = il >= model.hparams.n_layer() ? GGML_TYPE_F32 : is_k ? type_k : type_v;
            const uint32_t dim = is_k ? model.hparams.n_embd_k_gqa(il) : model.hparams.n_embd_v_gqa(il);
            ggml_tensor * t = ggml_new_tensor_2d(ctx.get(), type, dim, 1);
            ggml_format_name(t, "cache_%c_l%u", is_k ? 'k' : 'v', il);
            auto * ud = const_cast<llama_meta_device_get_split_state_userdata *>(&model.get_split_state_ud);
            const auto split = llama_meta_device_get_split_state(t, ud);
            if (split.axis != GGML_BACKEND_SPLIT_AXIS_0 || split.n_segments != 1 || split.nr[0] != 1) {
                throw std::runtime_error("unsupported tensor KV split layout for KVMem pool planning");
            }
            for (size_t j = 0; j < n_devices; ++j) {
                const int64_t shard_dim = split.ne[j];
                if (shard_dim < 0 || shard_dim % ggml_blck_size(type) != 0) {
                    throw std::runtime_error("tensor KV split is incompatible with the selected cache type");
                }
                if (shard_dim > 0) bytes[j] += ggml_row_size(type, shard_dim);
            }
        }
    }
    return bytes;
}

static kvmem_pool_plan kvmem_compute_pool(
        const llama_model & model,
        const llama_memory_params & params,
        const llama_cparams & cparams,
        uint32_t borrowed_kv_size = 0) {
    kvmem_pool_plan p;
    p.block_tokens = kvmem_current_execution().params.block_tokens ? kvmem_current_execution().params.block_tokens : 32u;
    uint32_t budget = kvmem_current_execution().params.budget;
    if (budget == 0) {
        budget = kvmem_align_tokens(cparams.n_ctx_seq, p.block_tokens);
        if (budget == 0) {
            budget = p.block_tokens;
        }
    } else {
        budget = kvmem_align_tokens(budget, p.block_tokens);
    }
    uint32_t gen_reserve = kvmem_current_execution().params.gen_reserve ? kvmem_current_execution().params.gen_reserve : 256u;
    gen_reserve = kvmem_align_tokens(gen_reserve, p.block_tokens);

    const uint32_t il0 = kvmem_first_attn_layer(model);
    const uint32_t n_attn = kvmem_n_attn_layers(model);
    const uint32_t n_embd_k = model.hparams.n_embd_k_gqa(il0);
    const uint32_t n_embd_v = model.hparams.n_embd_v_gqa(il0);
    const uint64_t k_row = ggml_row_size(params.type_k, n_embd_k);
    const uint64_t v_row = ggml_row_size(params.type_v, n_embd_v);
    p.block_bytes = static_cast<uint64_t>(n_attn) * (k_row + v_row) * p.block_tokens;

    const double ratio = kvmem_default_ratio(kvmem_current_execution().params.gpu_memory_ratio, 0.50);
    p.gpu_total = kvmem_first_gpu_total_bytes();
    if (p.gpu_total > 0 && p.block_bytes > 0 && ratio > 0.0) {
        p.cap_blocks = static_cast<uint32_t>(
                (p.gpu_total * ratio) / std::max(p.block_bytes, uint64_t{1}));
    }

    // A layer split has one common token window, but each GPU stores only the
    // KV of its own layers. The smallest per-device capacity bounds that window.
    std::map<ggml_backend_dev_t, uint64_t> bytes_per_token;
    std::map<ggml_backend_dev_t, uint64_t> mtp_state_bytes;
    for (uint32_t il = 0; il < model.hparams.n_layer(); ++il) {
        auto * dev = model.dev_layer(static_cast<int>(il));
        if (!dev || ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) continue;
        if (model.hparams.is_recr(il)) {
            if (cparams.n_rs_seq > 0) {
                const uint64_t state = static_cast<uint64_t>(model.hparams.n_embd_r() + model.hparams.n_embd_s()) * sizeof(float);
                // Match the five FP32 ReplaySSM record widths in llama_memory_recurrent.
                const uint64_t records = static_cast<uint64_t>(2048 + 6144 + 48 + 48 + 10240) *
                        sizeof(float) * (1 + cparams.n_rs_seq);
                mtp_state_bytes[dev] += kvmem_current_execution().params.mtp_state == 2 ? state + records : state * (1 + cparams.n_rs_seq);
            }
            continue;
        }
        bytes_per_token[dev] += ggml_row_size(params.type_k, model.hparams.n_embd_k_gqa(il))
                              + ggml_row_size(params.type_v, model.hparams.n_embd_v_gqa(il));
    }
    ggml_backend_dev_t mtp_owner = nullptr;
    uint64_t mtp_row_bytes = 0;
    if (cparams.n_rs_seq > 0 && model.hparams.n_layer_nextn > 0) {
        const uint32_t il = model.hparams.n_layer();
        mtp_owner = model.dev_layer(static_cast<int>(il));
        mtp_row_bytes = static_cast<uint64_t>(model.hparams.n_layer_nextn) *
                (ggml_row_size(GGML_TYPE_F32, model.hparams.n_embd_k_gqa(il)) +
                 ggml_row_size(GGML_TYPE_F32, model.hparams.n_embd_v_gqa(il)));
        if (mtp_owner && ggml_backend_dev_type(mtp_owner) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            bytes_per_token.try_emplace(mtp_owner, 0);
        }
    }
    if (kvmem_tensor_multi_gpu(model) && borrowed_kv_size == 0) {
        const auto rows = kvmem_tensor_kv_row_bytes(model, params.type_k, params.type_v, cparams.n_rs_seq > 0);
        const auto devices = kvmem_tensor_devices(model);
        uint64_t state_reserve = 0;
        if (cparams.n_rs_seq > 0) {
            for (uint32_t il = 0; il < model.hparams.n_layer(); ++il) {
                if (model.hparams.is_recr(il)) {
                    state_reserve += static_cast<uint64_t>(model.hparams.n_embd_r() + model.hparams.n_embd_s()) *
                                     sizeof(float) * (1 + cparams.n_rs_seq);
                }
            }
        }
        p.gpu_total = 0;
        uint64_t cap = UINT64_MAX;
        for (size_t j = 0; j < rows.size(); ++j) {
            auto * dev = devices[j];
            size_t free_bytes = 0, total_bytes = 0;
            ggml_backend_dev_memory(dev, &free_bytes, &total_bytes);
            // Reserve the whole snapshot state on each card as a safe upper bound
            // until per-shard recurrent memory accounting is exposed by Meta.
            const uint64_t reserve = std::max<uint64_t>(256ull << 20, free_bytes / 10) + state_reserve;
            const uint64_t free_for_kv = free_bytes > reserve ? free_bytes - reserve : 0;
            const uint64_t kv_budget = std::min<uint64_t>(free_for_kv, total_bytes * ratio);
            const uint64_t device_cap = rows[j] ? kv_budget / (rows[j] * p.block_tokens) : UINT64_MAX;
            LLAMA_LOG_INFO("%s: tensor KV device=%s free=%zu total=%zu row_bytes=%llu state_reserve=%llu cap_blocks=%llu\n",
                    __func__, ggml_backend_dev_name(dev), free_bytes, total_bytes,
                    (unsigned long long) rows[j], (unsigned long long) state_reserve,
                    (unsigned long long) device_cap);
            cap = std::min(cap, device_cap);
        }
        if (cap < 2) throw std::runtime_error("tensor KV pool cannot fit a working block plus generation reserve");
        p.cap_blocks = static_cast<uint32_t>(std::min<uint64_t>(cap, UINT32_MAX));
        if (kvmem_current_execution().params.budget &&
                static_cast<uint64_t>(budget) + gen_reserve > cap * p.block_tokens) {
            throw std::runtime_error("requested KVMem budget plus generation reserve exceeds a tensor GPU KV capacity");
        }
    } else if (model.n_devices() > 1 && !bytes_per_token.empty() && borrowed_kv_size == 0) {
        p.gpu_total = 0; // the per-device limits below replace a first-GPU total
        uint64_t cap = UINT64_MAX;
        for (const auto & [dev, row_bytes] : bytes_per_token) {
            size_t free_bytes = 0, total_bytes = 0;
            ggml_backend_dev_memory(dev, &free_bytes, &total_bytes);
            // free_bytes is queried after model weights load. Leave headroom
            // for the scheduler graph and staging buffers on every device.
            const uint64_t mtp_fixed = mtp_state_bytes[dev] + (cparams.n_rs_seq > 0 ? 128ull << 20 : 0);
            const uint64_t reserve = std::max<uint64_t>(256ull << 20, free_bytes / 10) + mtp_fixed;
            const uint64_t free_for_kv = free_bytes > reserve ? free_bytes - reserve : 0;
            const uint64_t ratio_budget = static_cast<uint64_t>(total_bytes * ratio);
            const uint64_t kv_budget = std::min(free_for_kv, ratio_budget);
            const uint64_t planned_row = row_bytes + (dev == mtp_owner ? mtp_row_bytes : 0);
            const uint64_t device_cap = planned_row ? kv_budget / (planned_row * p.block_tokens) : UINT64_MAX;
            LLAMA_LOG_INFO("%s: layer KV device=%s free=%zu total=%zu row_bytes=%llu mtp_row_bytes=%llu mtp_state_reserve=%llu cap_blocks=%llu\n",
                    __func__, ggml_backend_dev_name(dev), free_bytes, total_bytes,
                    (unsigned long long) row_bytes, (unsigned long long) (dev == mtp_owner ? mtp_row_bytes : 0),
                    (unsigned long long) mtp_fixed, (unsigned long long) device_cap);
            cap = std::min(cap, device_cap);
        }
        if (cap < 2) throw std::runtime_error("multi-GPU layer KV pool cannot fit a working block plus generation reserve on every owning GPU");
        p.cap_blocks = static_cast<uint32_t>(std::min<uint64_t>(cap, UINT32_MAX));
        if (kvmem_current_execution().params.budget &&
                static_cast<uint64_t>(budget) + gen_reserve > cap * p.block_tokens) {
            throw std::runtime_error("requested KVMem budget plus generation reserve exceeds a layer GPU KV capacity");
        }
    } else if (borrowed_kv_size > 0) {
        // Hybrid models allocate the attention cache before constructing this
        // adapter. Reuse its committed capacity instead of subtracting that
        // allocation from free VRAM and planning a smaller cache a second time.
        p.gpu_total = 0;
        p.cap_blocks = borrowed_kv_size / p.block_tokens;
    }

    uint32_t pool = budget + gen_reserve;
    if (pool > cparams.n_ctx_seq && kvmem_current_execution().params.budget == 0) {
        pool = cparams.n_ctx_seq;
    }
    if (p.cap_blocks > 0) {
        const uint32_t cap_tokens = p.cap_blocks * p.block_tokens;
        if (pool > cap_tokens) {
            pool = cap_tokens;
            if (budget >= pool) {
                gen_reserve = std::min(gen_reserve, p.block_tokens);
                budget = pool > gen_reserve ? pool - gen_reserve : pool;
            } else if (budget + gen_reserve > pool) {
                gen_reserve = pool - budget;
            }
            budget = kvmem_align_tokens(std::max(budget, p.block_tokens), p.block_tokens);
            gen_reserve = kvmem_align_tokens(std::max(gen_reserve, p.block_tokens), p.block_tokens);
            pool = budget + gen_reserve;
            if (pool > cap_tokens) {
                pool = cap_tokens;
            }
        }
    }
    p.budget = budget;
    p.gen_reserve = gen_reserve;
    p.kv_size = borrowed_kv_size ? borrowed_kv_size : std::max(pool, 1u);
    p.n_slots = (p.kv_size + p.block_tokens - 1) / p.block_tokens;
    if (p.n_slots * p.block_tokens < p.kv_size) {
        p.n_slots += 1;
    }
    return p;
}

uint32_t llama_kvmem_pool_cells(
        const llama_model & model,
        const llama_memory_params & params,
        const llama_cparams & cparams) {
    return kvmem_compute_pool(model, params, cparams).kv_size;
}

llama_memory_i * llama_memory_kvmem_maybe_create(
        const llama_model & model,
        const llama_memory_params & params,
        const llama_cparams & cparams) {
    if (!kvmem_current_execution().params.enabled) {
        return nullptr;
    }
    if (params.ctx_type == LLAMA_CONTEXT_TYPE_MTP) {
        llama_memory_kvmem * tgt = nullptr;
        if (cparams.ctx_other) {
            llama_memory_t mem = llama_get_memory(cparams.ctx_other);
            if (auto * hyb = dynamic_cast<llama_memory_kvmem_hybrid *>(mem)) {
                tgt = hyb->attn_kvmem();
            } else {
                tgt = dynamic_cast<llama_memory_kvmem *>(mem);
            }
        }
        if (!tgt) {
            return nullptr;
        }
        return new llama_memory_kvmem_mtp(model, params, cparams, tgt);
    }
    if (llm_arch_is_recurrent(model.arch)) {
        LLAMA_LOG_WARN("%s: KVMem skips purely recurrent arch %s\n",
                __func__, llm_arch_name(model.arch));
        return nullptr;
    }
    if (cparams.n_seq_max > 1) {
        LLAMA_LOG_WARN("%s: KVMem requires n_seq_max=1 (got %u)\n",
                __func__, cparams.n_seq_max);
        return nullptr;
    }
    if (model.hparams.swa_type != LLAMA_SWA_TYPE_NONE) {
        LLAMA_LOG_WARN("%s: KVMem skips SWA models\n", __func__);
        return nullptr;
    }
    if (llm_arch_is_hybrid(model.arch)) {
        return new llama_memory_kvmem_hybrid(model, params, cparams);
    }
    return new llama_memory_kvmem(model, params, cparams);
}

llama_memory_kvmem::llama_memory_kvmem(
        const llama_model & model,
        const llama_memory_params & params,
        const llama_cparams & cparams,
        llama_kv_cache * ext_kv) :
    model_(model) {
    backend_.owner = this;
    trace_ = kvmem_diag_enabled();
    perf_.enabled = kvmem_env_perf();
    retr_.enabled = perf_.enabled;
    harvest_perf_emit_graph_line();

    const kvmem_pool_plan pool = kvmem_compute_pool(
            model, params, cparams,
            ext_kv && (model.n_devices() > 1 || kvmem_tensor_multi_gpu(model)) ? ext_kv->get_size() : 0);
    block_tokens_ = pool.block_tokens;
    kv_size_ = pool.kv_size;
    n_slots_ = pool.n_slots;
    context_n_batch_ = cparams.n_batch;
    context_n_ubatch_ = cparams.n_ubatch;

    // Both store configs are kept as members so a sibling store for another
    // conversation is configured identically.
    rt_cfg_ = make_runtime_cfg(
            block_tokens_, pool.budget, kvmem_current_execution().params.sink_tokens, kvmem_current_execution().params.recent_tokens,
            pool.block_bytes);
    rt_cfg_.store.estimated_gpu_block_capacity = pool.cap_blocks;
    runtime_ = std::make_unique<kvmem::KvMemRuntime>(rt_cfg_, &backend_);

    if (ext_kv) {
        kv_ = ext_kv;
        if (kv_->get_size() != kv_size_) {
            LLAMA_LOG_WARN("%s: borrowed attn cache size %u != planned pool %u\n",
                    __func__, kv_->get_size(), kv_size_);
            kv_size_ = kv_->get_size();
            n_slots_ = (kv_size_ + block_tokens_ - 1) / block_tokens_;
            if (n_slots_ * block_tokens_ < kv_size_) {
                n_slots_ += 1;
            }
        }
    } else {
        kv_owned_ = std::make_unique<llama_kv_cache>(
                model,
                model.hparams,
                params.type_k,
                params.type_v,
                !cparams.flash_attn,
                cparams.offload_kqv,
                /* unified */ true,
                kv_size_,
                /* n_seq_max */ 1,
                /* n_pad */ 1,
                model.hparams.n_swa,
                model.hparams.swa_type,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                kvmem_tensor_multi_gpu(model) ? "" : "kvmem");
        kv_ = kv_owned_.get();
    }

    std::set<ggml_backend_dev_t> kv_owners;
    for (uint32_t il : kv_->get_layer_ids()) {
        auto * expected = model.dev_layer(static_cast<int>(il));
        if (kvmem_tensor_multi_gpu(model)) {
            for (ggml_tensor * t : {kv_->get_k_storage(static_cast<int32_t>(il)),
                                    kv_->get_v_storage(static_cast<int32_t>(il))}) {
                if (!t) continue;
                auto * buffer = t->buffer ? t->buffer : t->view_src ? t->view_src->buffer : nullptr;
                if (!ggml_backend_buffer_is_meta(buffer)) {
                    throw std::runtime_error("tensor split requires Meta attention KV buffers");
                }
            }
        } else if (model.n_devices() > 1) {
            for (ggml_tensor * t : {kv_->get_k_storage(static_cast<int32_t>(il)),
                                    kv_->get_v_storage(static_cast<int32_t>(il))}) {
                if (!t) continue;
                auto * buffer = t->buffer ? t->buffer : t->view_src ? t->view_src->buffer : nullptr;
                auto * actual = buffer ? ggml_backend_buft_get_device(ggml_backend_buffer_get_type(buffer)) : nullptr;
                if (actual != expected) {
                    throw std::runtime_error("multi-GPU layer requires every attention KV tensor on its model layer's GPU");
                }
            }
        }
        if (expected && ggml_backend_dev_type(expected) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            kv_owners.insert(expected);
        }
    }
    multi_gpu_ = model.n_devices() > 1 || kvmem_tensor_multi_gpu(model);
    if (multi_gpu_) {
        if (kvmem_tensor_multi_gpu(model)) {
            LLAMA_LOG_INFO("%s: synchronous tensor KV path across %zu physical GPUs\n",
                           __func__, model.get_split_state_ud.n_devices);
        } else {
            LLAMA_LOG_INFO("%s: synchronous layer KV path across %zu selected GPUs (%zu attention owners)\n",
                           __func__, model.n_devices(), kv_owners.size());
        }
    }

    reset_slots();

    const uint32_t il0 = kvmem_first_attn_layer(model);
    n_layer_ = model.hparams.n_layer();
    n_embd_head_ = model.hparams.n_embd_head_k(il0);
    n_head_kv_ = model.hparams.n_head_kv(il0);
    n_head_ = model.hparams.n_head(il0);
    n_embd_k_ = model.hparams.n_embd_k_gqa(il0);
    n_embd_v_ = model.hparams.n_embd_v_gqa(il0);
    rope_.n_rot = model.hparams.n_rot(il0);
    rope_.n_embd_head = n_embd_head_;
    rope_.n_head_kv = n_head_kv_;
    rope_.freq_base = cparams.rope_freq_base;
    rope_.freq_scale = cparams.rope_freq_scale;
    type_k_ = params.type_k;
    type_v_ = params.type_v;
    v_trans_ = !cparams.flash_attn;
    method_ = kvmem_current_execution().params.method;
    query_begin_ = kvmem_current_execution().params.query_begin;
    query_end_ = kvmem_current_execution().params.query_end;
    force_pos_ = kvmem_current_execution().params.force_pos;
    kvmem::RawKvStoreConfig & rcfg = raw_cfg_;
    rcfg.n_layer = n_layer_;
    rcfg.n_embd_k = n_embd_k_;
    rcfg.n_embd_v = n_embd_v_;
    rcfg.block_tokens = block_tokens_;
    rcfg.historical_gc_require_mean_k = method_ == 1;
    rcfg.historical_gc_attention_layers.clear();
    for (uint32_t il = 0; il < n_layer_; ++il) {
        if (kvmem_cache_has_layer(kv_, static_cast<int32_t>(il))) {
            rcfg.historical_gc_attention_layers.push_back(il);
        }
    }
    if (ggml_is_quantized(type_k_)) {
        rcfg.k_row_bytes = ggml_row_size(type_k_, n_embd_k_);
    }
    rcfg.k_gpu_row_bytes = ggml_row_size(type_k_, n_embd_k_);
    if (!v_trans_) {
        rcfg.v_gpu_row_bytes = ggml_row_size(type_v_, n_embd_v_);
    }
    if (kvmem_current_execution().params.raw_k_nvme) {
        rcfg.nvme_bytes = kvmem_current_execution().params.nvme_bytes
                ? kvmem_current_execution().params.nvme_bytes
                : (32ull * 1024ull * 1024ull * 1024ull);
        rcfg.nvme_dir = (kvmem_current_execution().params.nvme_dir && kvmem_current_execution().params.nvme_dir[0])
                ? kvmem_current_execution().params.nvme_dir
                : "/tmp/kvmem_nvme";
        rcfg.nvme_file = "kvmem_raw_k.bin";
    }
    raw_ = std::make_unique<kvmem::RawKvStore>(rcfg);
    q_sum_.assign(n_layer_, std::vector<float>(n_head_ * n_embd_head_, 0.0f));
    q_count_.assign(n_layer_, 0);
    kvmem_capture_bind(this);

    size_t kv_bytes = 0;
    if (kv_) {
        for (const auto & kv : kv_->memory_breakdown()) {
            kv_bytes += kv.second;
        }
    }
    LLAMA_LOG_INFO(
            "%s: KVMem slot-pool cells=%u slots=%u block_tokens=%u budget=%u gen_reserve=%u sink_blocks=%u method=%s harvest_v=%d type_k=%s type_v=%s n_embd_k=%u attn_layers=%u%s\n",
            __func__, kv_size_, n_slots_, block_tokens_, pool.budget, pool.gen_reserve,
            rt_cfg_.store.sink_blocks,
            method_ == 1 ? "retrieval" : "recency",
            (int) kvmem_current_execution().params.harvest_v,
            ggml_type_name(type_k_), ggml_type_name(type_v_),
            n_embd_k_, kvmem_n_attn_layers(model),
            ext_kv ? " hybrid_attn" : "");
    kvmem_diag("KVMEM_KV_BYTES bytes=%zu cells=%u slots=%u budget=%u pool=%u "
            "ratio=%.2f high=%.2f low=%.2f cap_blocks=%u gpu_total=%llu block_bytes=%llu\n",
            kv_bytes, kv_size_, n_slots_, pool.budget, kv_size_,
            rt_cfg_.store.gpu_memory_ratio,
            rt_cfg_.store.gpu_high_watermark,
            rt_cfg_.store.gpu_low_watermark,
            pool.cap_blocks,
            (unsigned long long) pool.gpu_total,
            (unsigned long long) pool.block_bytes);
}

// Only the admitted RAM-only domain can use this task. Its runtime has no
// prefetch/NVMe workers, and destruction does not invoke SlotBackend. GPU
// buffers, streams, graph completion and source-slot accounting stay with
// the adapter. Do not extend this handoff to other modes without that audit.
struct llama_memory_kvmem::HistoricalCpuCleanup final : kvmem::HistoricalKvCleanup::Task {
    std::unique_ptr<kvmem::KvMemRuntime> runtime;
    std::unique_ptr<kvmem::RawKvStore> raw;
    std::unique_ptr<kvmem::HistoricalKvCaptureSession> capture;
    std::unique_ptr<kvmem::HistoricalKvActiveSet> active;
    HistoricalActiveWorkspace active_workspace;
    CellWorkspace cells;
    std::vector<RowPosition> positions;
    std::vector<CaptureBatch> queued_captures;
    CaptureBatch current_capture, pending_mean;
    MeanWriteRange mean_target;
    std::vector<HarvestVJob> packed_jobs;
    HarvestVBatch pending_packed;
    std::function<void()> submission_hook;
    // Destroy the manager before directory/session/storage ownership.
    std::unique_ptr<kvmem::HistoricalKvGc> gc;
    bool ready() const noexcept override { return !gc || gc->worker_finished(); }
};

// A decode context owns the prepared slot vectors while upstream is using
// them. The base context has a virtual destructor, so return the same storage
// to the request before the next decode instead of allocating another set.
class llama_memory_kvmem::HistoricalBatchContext final : public llama_kv_cache_context {
public:
    HistoricalBatchContext(llama_memory_kvmem & owner, llama_kv_cache * kv,
            llama_kv_cache::slot_info_vec_t sinfos, std::vector<llama_ubatch> ubatches) :
        llama_kv_cache_context(kv, std::move(sinfos), std::move(ubatches)), owner_(owner) {}

    ~HistoricalBatchContext() override {
        // The request has only one live decode context. An unexpected second
        // owner means reusing these buffers would invalidate an active decode.
        if (!owner_.historical_slot_workspace_.slots.empty() ||
            !owner_.historical_ubatch_workspace_.ubatches.empty()) std::terminate();
        owner_.historical_ubatch_workspace_.ubatches = take_ubatches();
        owner_.historical_slot_workspace_.slots = take_slot_infos();
        owner_.historical_batch_context_in_use_ = false;
    }

    // The unique_ptr returned to upstream still uses a normal virtual delete.
    // Its destructor runs, but the request owns this fixed storage until
    // llama_memory_kvmem teardown.
    static void operator delete(void *) noexcept {}

private:
    llama_memory_kvmem & owner_;
};

llama_memory_kvmem::~llama_memory_kvmem() {
    if (historical_batch_context_in_use_) std::terminate();
    ::operator delete(historical_batch_context_storage_);
    historical_batch_context_storage_ = nullptr;
    if (historical_capture_) historical_capture_->repository()->close();
    if (historical_gc_) historical_gc_->request_stop();
    try {
        // A partial failed sum must never be retried or published by teardown.
        // Keep its writers until the normal device owner proves completion.
        if (!capture_failed_) {
            try { decode_mean_flush(); }
            catch (const std::exception & e) {
                capture_fail();
                LLAMA_LOG_ERROR("KVMem cancelling decode Mean-K at teardown: %s\n", e.what());
            }
        }
        harvest_shutdown();
        decode_mean_shutdown();
        decode_mean_print_sum();
        if (historical_capture_) harvest_gpu_v_commit();
        harvest_worker_stop();
        harvest_perf_print_sum();
        d2h_free();
        multi_d2h_free();
        if (!multi_gpu_) kvmem_stagein_gpu_free();
    } catch (const std::exception & e) {
        // A persistent device failure cannot grant reuse/free permission.
        // Abort before member unwinding can destroy still-used GPU sources
        // or publish their CPU owners to the background cleanup service.
        GGML_ABORT("KVMem teardown could not safely finish device work: %s", e.what());
    } catch (...) {
        GGML_ABORT("KVMem teardown could not safely finish device work");
    }
    if (mtp_) {
        mtp_->detach_target();
        mtp_ = nullptr;
    }
    kvmem_conv_pool_unbind(this);
    kvmem_capture_unbind(this);
    if (historical_cleanup_) {
        // All adapter-owned GPU work has completed above. Move the entire
        // CPU ownership graph, so no foreground directory/shared_ptr teardown
        // can become the last owner of historical payloads.
        auto & cleanup = *historical_cleanup_;
        cleanup.runtime = std::move(runtime_);
        cleanup.raw = std::move(raw_);
        cleanup.capture = std::move(historical_capture_);
        cleanup.active = std::move(historical_active_);
        cleanup.active_workspace = std::move(historical_active_workspace_);
        cleanup.cells = std::move(cell_workspace_);
        cleanup.gc = std::move(historical_gc_);
        cleanup.positions = std::move(row_positions_);
        cleanup.queued_captures = std::move(pos_queue_);
        cleanup.current_capture = std::move(cur_capture_);
        cleanup.pending_mean = std::move(decode_mean_pending_batch_);
        cleanup.mean_target = std::move(decode_mean_target_);
        cleanup.packed_jobs = std::move(harvest_v_jobs_);
        cleanup.pending_packed = std::move(harvest_v_pending_);
        cleanup.submission_hook = std::move(historical_submission_test_hook_);
        historical_cleanup_ticket_.submit(std::move(historical_cleanup_));
    }
}

bool llama_memory_kvmem::begin_historical_capture(uint32_t capacity, uint64_t input_tokens,
        uint64_t max_output_tokens, uint64_t max_appends) {
    if (historical_capture_ || historical_batch_context_storage_ || method_ != 0 || multi_gpu_ || mtp_ || recr_ || v_trans_ ||
        replay_ || !kv_ || raw_cfg_.nvme_bytes || rt_cfg_.cpu_bytes || rt_cfg_.nvme_bytes ||
        raw_cfg_.historical_gc_attention_layers.empty() ||
        !harvest_v_jobs_.empty() || !harvest_v_pending_.jobs.empty() ||
        !pos_queue_.empty() || !cur_capture_.positions.empty() ||
        !decode_mean_pending_batch_.positions.empty()) {
        LLAMA_LOG_ERROR("historical capture: requires a fresh single-GPU recency RAM context without MTP/hybrid/spill tiers\n");
        return false;
    }
    try {
        kvmem::HistoricalKvRawStoreConfig cfg;
        cfg.n_layer = n_layer_;
        cfg.max_tokens_per_slot = block_tokens_;
        cfg.packed_k_bytes_per_token = ggml_row_size(type_k_, n_embd_k_);
        cfg.packed_v_bytes_per_token = ggml_row_size(type_v_, n_embd_v_);
        cfg.attention_layers = raw_cfg_.historical_gc_attention_layers;
        // Recency needs the packed KV only. Mean-K/retrieval joins this owner
        // after replay and execution admission are integrated.
        int storage_device = -1;
        for (uint32_t il : cfg.attention_layers) {
            auto * k = kv_->get_k_storage(il);
            auto * v = kv_->get_v_storage(il);
            if (!k || !v || k->nb[1] != cfg.packed_k_bytes_per_token ||
                v->nb[1] != cfg.packed_v_bytes_per_token)
                throw std::runtime_error("historical capture requires uniform packed attention rows");
            for (auto * tensor : {k, v}) {
                const auto * ptr = kvmem_cuda_tensor_ptr(tensor);
                cudaPointerAttributes attrs{};
                if (!ptr || cudaPointerGetAttributes(&attrs, ptr) != cudaSuccess ||
                    attrs.type != cudaMemoryTypeDevice ||
                    (storage_device >= 0 && attrs.device != storage_device))
                    throw std::runtime_error("historical capture requires all attention KV on one CUDA device");
                storage_device = attrs.device;
            }
        }
        kvmem::HistoricalKvRamDemand demand;
        demand.uncached_input_tokens = input_tokens;
        demand.max_accepted_output_tokens = max_output_tokens;
        demand.max_append_boundaries = max_appends;
        auto & cleanup_service = kvmem::HistoricalKvCleanup::process();
        auto cleanup_ticket = cleanup_service.try_reserve();
        if (!cleanup_ticket) {
            LLAMA_LOG_ERROR("historical capture: CPU cleanup capacity exhausted before admission\n");
            return false;
        }
        auto cleanup = std::make_unique<HistoricalCpuCleanup>();
        auto session = kvmem::HistoricalKvCaptureSession::try_prepare(*runtime_, cfg, capacity, demand);
        if (!session) {
            LLAMA_LOG_ERROR("historical capture: insufficient reserved RAM or CPU cleanup capacity\n");
            return false;
        }
        auto cell_indices = kv_->prepare_cell_index_nodes(0);
        if (!cell_indices) {
            LLAMA_LOG_ERROR("historical capture: cell indexes are not fresh before admission\n");
            return false;
        }
        auto release_context_storage = [](void * ptr) noexcept { ::operator delete(ptr); };
        std::unique_ptr<void, decltype(release_context_storage)> context_storage(
            ::operator new(sizeof(HistoricalBatchContext)), release_context_storage);
        // Prepare both sides of the pipeline before publishing this owner.
        // Full ubatch contexts and backend allocations still need admission.
        auto active = kvmem::HistoricalKvActiveSet::prepare(session->repository(), n_slots_);
        HistoricalActiveWorkspace active_workspace;
        active_workspace.reserve(n_slots_);
        CellWorkspace cells;
        cells.reserve(block_tokens_);
        if (input_tokens > SIZE_MAX - max_output_tokens)
            throw std::length_error("historical capture token count overflow");
        const size_t max_capture_tokens = static_cast<size_t>(input_tokens + max_output_tokens);
        HistoricalSlotWorkspace slot_workspace;
        slot_workspace.prepare(context_n_batch_, context_n_ubatch_, max_capture_tokens);
        HistoricalUbatchWorkspace ubatch_workspace;
        ubatch_workspace.prepare(context_n_batch_, context_n_ubatch_, max_capture_tokens);
        // Each token has at most one original position and one capture range.
        // One extra copy can be held by a D2H slot. Four allocations per
        // one-token batch cover alignment padding in the fixed arena.
        const size_t arena_per_token = 2 * (sizeof(llama_pos) + sizeof(MeanWriteRange) +
                2 * alignof(std::max_align_t));
        if (max_capture_tokens > SIZE_MAX / arena_per_token)
            throw std::length_error("historical capture arena size overflow");
        auto capture_arena = std::make_shared<kvmem::FixedByteArena>(max_capture_tokens * arena_per_token);
        std::vector<CaptureBatch> queued_captures;
        queued_captures.reserve(max_capture_tokens);
        std::vector<RowPosition> positions;
        positions.reserve(max_capture_tokens);
        std::vector<uint32_t> readers(n_slots_, 0);
        std::vector<uint8_t> retired(n_slots_, 0);
        std::vector<HarvestVJob> jobs, pending;
        jobs.reserve(KVMEM_CAPTURE_MAX_JOBS);
        pending.reserve(KVMEM_CAPTURE_MAX_JOBS);
        const size_t row = std::max(cfg.packed_k_bytes_per_token, cfg.packed_v_bytes_per_token);
        const size_t width = std::max(n_embd_k_, n_embd_v_);
        if (!block_tokens_ || row > SIZE_MAX / block_tokens_ || width > SIZE_MAX / block_tokens_)
            throw std::runtime_error("historical capture workspace size overflow");
        std::vector<uint8_t> fallback(row * block_tokens_);
        if (!n_slots_ || cfg.packed_k_bytes_per_token > SIZE_MAX - cfg.packed_v_bytes_per_token ||
            cfg.packed_k_bytes_per_token + cfg.packed_v_bytes_per_token > SIZE_MAX / block_tokens_)
            throw std::runtime_error("historical layout stride overflow");
        const size_t layout_stride = (cfg.packed_k_bytes_per_token + cfg.packed_v_bytes_per_token) * block_tokens_;
        if (layout_stride > SIZE_MAX / n_slots_ || size_t(n_slots_) > SIZE_MAX / 2)
            throw std::runtime_error("historical layout workspace overflow");
        LayoutWorkspace layout;
        layout.reserve(n_slots_);
        if (!kvmem_capture_workspace_reserve(width * block_tokens_, fallback.size(),
                KVMEM_CAPTURE_MAX_JOBS, layout_stride * n_slots_)) {
            LLAMA_LOG_ERROR("historical capture: packed workspace unavailable before admission\n");
            return false;
        }
        // All request resources prepared so far are still private. If the
        // runtime changed before commit, release GPU resources here; the
        // unadmitted session then returns RAM/metadata without a GC worker.
        // Installation is reversible until the RAM/metadata commit succeeds.
        // This owner performs no cell writes in that preparation interval.
        if (!cell_indices->install() || !session->admit()) {
            kvmem_stagein_gpu_free();
            LLAMA_LOG_ERROR("historical capture: runtime changed before admission commit\n");
            return false;
        }
        cell_indices->accept();
        // Only nonthrowing ownership transfers follow successful admission.
        row_positions_.swap(positions);
        historical_gpu_readers_.swap(readers);
        historical_gpu_retired_.swap(retired);
        harvest_v_jobs_.swap(jobs);
        harvest_v_pending_.jobs.swap(pending);
        harvest_v_fallback_.swap(fallback);
        layout_workspace_ = std::move(layout);
        historical_active_ = std::move(active);
        historical_active_workspace_ = std::move(active_workspace);
        cell_workspace_ = std::move(cells);
        historical_slot_workspace_ = std::move(slot_workspace);
        historical_ubatch_workspace_ = std::move(ubatch_workspace);
        historical_batch_context_storage_ = context_storage.release();
        historical_batch_context_storage_bytes_ = sizeof(HistoricalBatchContext);
        pos_queue_.swap(queued_captures);
        capture_arena_.swap(capture_arena);
        capture_max_tokens_ = max_capture_tokens;
        capture_tokens_used_ = 0;
        harvest_v_job_limit_ = KVMEM_CAPTURE_MAX_JOBS;
        historical_cleanup_ticket_ = std::move(*cleanup_ticket);
        historical_cleanup_ = std::move(cleanup);
        historical_capture_ = std::move(session);
        const auto workspace = kvmem_capture_workspace_status();
        const auto cell_nodes = kv_->get_cells(0).index_status();
        LLAMA_LOG_INFO("KVMEM_CELL_INDEX nodes=%zu available=%zu\n", cell_nodes.nodes, cell_nodes.available);
        LLAMA_LOG_INFO("KVMEM_SLOT_INFO slots=%zu bytes=%zu max_batch=%u ubatch=%u\n",
            historical_slot_workspace_.slots.size(), historical_slot_workspace_.capacity_bytes(),
            historical_slot_workspace_.max_tokens, historical_slot_workspace_.n_ubatch);
        LLAMA_LOG_INFO("KVMEM_UBATCH_POOL blocks=%zu vector_bytes=%zu max_batch=%u ubatch=%u\n",
            historical_ubatch_workspace_.data.size(), historical_ubatch_workspace_.vector_bytes(),
            historical_ubatch_workspace_.max_tokens, historical_ubatch_workspace_.n_ubatch);
        LLAMA_LOG_INFO("KVMEM_BATCH_CONTEXT reserved_bytes=%zu\n", historical_batch_context_storage_bytes_);
        LLAMA_LOG_INFO("KVMEM_CAPTURE_ARENA bytes=%zu queue_capacity=%zu max_tokens=%zu\n",
            capture_arena_->capacity_bytes(), pos_queue_.capacity(), max_capture_tokens);
        LLAMA_LOG_INFO("KVMEM_CAPTURE_WORKSPACE jobs_per_batch=%zu device_bytes=%zu pinned_bytes=%zu host_queue_bytes=%zu fallback_bytes=%zu layout_bytes=%zu layout_slots=%u layout_host_bytes=%zu\n",
            harvest_v_job_limit_, workspace.device_bytes, workspace.pinned_bytes,
            workspace.host_bytes + (harvest_v_jobs_.capacity() + harvest_v_pending_.jobs.capacity()) * sizeof(HarvestVJob),
            harvest_v_fallback_.size(), workspace.layout_bytes, n_slots_, layout_workspace_.capacity_bytes());
        LLAMA_LOG_INFO("KVMEM_HISTORICAL_CAPTURE begin capacity=%u reserved=%u payload_bytes=%llu input=%llu output_max=%llu appends_max=%llu metadata_slots=%u active_slots=%zu active_bytes=%zu cell_scratch_bytes=%zu\n",
            capacity, historical_capture_->reserved_blocks(),
            (unsigned long long) historical_capture_->payload_bytes(),
            (unsigned long long) input_tokens, (unsigned long long) max_output_tokens,
            (unsigned long long) max_appends, historical_capture_->metadata_remaining(),
            historical_active_->capacity(), historical_active_->capacity_bytes() + historical_active_workspace_.capacity_bytes(),
            cell_workspace_.capacity_bytes());
        return true;
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("historical capture begin: %s\n", e.what());
        return false;
    }
}

bool llama_memory_kvmem::start_historical_gc(int mode, uint32_t ready_low,
                                             uint32_t ready_high) {
    if (!historical_capture_ || historical_capture_->failed() ||
        historical_capture_->finished() || historical_gc_ ||
        (mode != 1 && mode != 2) || !ready_low ||
        ready_low >= ready_high || ready_high > historical_capture_->repository()->capacity()) {
        LLAMA_LOG_ERROR("historical GC: invalid request owner, mode or Ready thresholds\n");
        return false;
    }
    try {
        const auto repository = historical_capture_->repository();
        kvmem::HistoricalKvGc::Config cfg;
        cfg.mode = mode == 1 ? kvmem::HistoricalKvGc::Mode::Shadow :
                               kvmem::HistoricalKvGc::Mode::Delete;
        cfg.ready_low = ready_low;
        cfg.ready_high = ready_high;
        cfg.scan_blocks = std::min<uint32_t>(repository->capacity(), 64);
        cfg.batch_blocks = std::min<uint32_t>(repository->capacity(), 16);
        cfg.reclaim_blocks = std::min<uint32_t>(repository->capacity(), 64);
        cfg.poll_ms = 5;
        auto manager = std::make_unique<kvmem::HistoricalKvGc>(repository,
            std::make_unique<kvmem::HistoricalKvLruPolicy>(repository->capacity()), cfg);
        manager->start();
        historical_gc_ = std::move(manager);
        historical_gc_mode_ = mode;
        LLAMA_LOG_INFO("KVMEM_HISTORICAL_GC begin mode=%s low=%u high=%u capacity=%u\n",
            mode == 1 ? "shadow" : "delete", ready_low, ready_high, repository->capacity());
        return true;
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("historical GC start: %s\n", e.what());
        return false;
    }
}

void llama_memory_kvmem::fail_historical_capture() noexcept {
    if (historical_capture_) capture_fail();
}

void llama_memory_kvmem::end_prefill_capture() {
    if (historical_capture_ && !historical_capture_->end_input()) {
        historical_capture_->fail();
        LLAMA_LOG_ERROR("historical capture: input quota or producer handoff incomplete; context closed\n");
    }
    prefill_capture_ = false;
}

void llama_memory_kvmem::set_mtp_follower(llama_memory_kvmem_mtp * mtp) {
    if (mtp && historical_capture_) throw std::runtime_error("historical capture baseline does not support MTP");
    mtp_ = mtp;
}

bool llama_memory_kvmem::finish_historical_capture() {
    if (!historical_capture_) return false;
    try {
        harvest_flush();
        harvest_gpu_v_commit();
        if (std::any_of(historical_gpu_readers_.begin(), historical_gpu_readers_.end(),
                        [](uint32_t readers) { return readers != 0; }) ||
            std::any_of(historical_gpu_retired_.begin(), historical_gpu_retired_.end(),
                        [](uint8_t retired) { return retired != 0; }))
            throw std::logic_error("historical request still owns pending GPU source slots");
        // Request completion observes task leases only. The GC may keep
        // observing metadata or releasing payloads after this boundary;
        // joining it belongs to context shutdown, never to request finish.
        if (!historical_capture_->finish()) {
            LLAMA_LOG_ERROR("historical capture: incomplete publication or outstanding producers\n");
            return false;
        }
        if (historical_gc_) {
            const auto stats = historical_gc_->stats();
            LLAMA_LOG_INFO("KVMEM_HISTORICAL_GC finish mode=%s proposed=%llu committed=%llu shadow=%llu reclaimed=%llu deferred=%llu retry=%llu policy_failures=%llu worker_failures=%llu ready=%u payload_bytes=%llu\n",
                historical_gc_mode_ == 1 ? "shadow" : "delete",
                (unsigned long long) stats.proposed, (unsigned long long) stats.committed,
                (unsigned long long) stats.shadow, (unsigned long long) stats.reclaimed,
                (unsigned long long) stats.deferred, (unsigned long long) stats.retry,
                (unsigned long long) stats.policy_failures,
                (unsigned long long) stats.worker_failures,
                historical_capture_->repository()->ready_count(),
                (unsigned long long) historical_capture_->payload_bytes());
            if (stats.policy_failures || stats.worker_failures) return false;
        }
        LLAMA_LOG_INFO("KVMEM_HISTORICAL_CAPTURE finish blocks=%u live_observed=%u task_references=0 rows=%u payload_bytes=%llu gpu_deferred=%llu gpu_returned=%llu\n",
            historical_capture_->created_blocks(), historical_capture_->live_blocks_at_finish(), store_n_tokens(),
            (unsigned long long) historical_capture_->payload_bytes(),
            (unsigned long long) historical_gpu_deferred_,
            (unsigned long long) historical_gpu_returned_);
        LLAMA_LOG_INFO("KVMEM_CAPTURE_ARENA finish used_bytes=%zu capacity_bytes=%zu captured_tokens=%zu max_tokens=%zu\n",
            capture_arena_->used_bytes(), capture_arena_->capacity_bytes(),
            capture_tokens_used_, capture_max_tokens_);
        return true;
    } catch (const std::exception & e) {
        historical_capture_->fail();
        LLAMA_LOG_ERROR("historical capture finish: %s\n", e.what());
        return false;
    }
}

void llama_memory_kvmem::reset_slots() {
    if (std::any_of(historical_gpu_readers_.begin(), historical_gpu_readers_.end(),
                   [](uint32_t readers) { return readers != 0; }) ||
        std::any_of(historical_gpu_retired_.begin(), historical_gpu_retired_.end(),
                   [](uint8_t retired) { return retired != 0; }))
        throw std::logic_error("GPU slot reset before historical sources completed");
    free_slots_.clear();
    free_slots_.reserve(n_slots_);
    for (int32_t i = static_cast<int32_t>(n_slots_) - 1; i >= 0; --i) {
        free_slots_.push_back(i);
    }
}

void llama_memory_kvmem::reset_turn_policy() {
    explicit_spans_ = false;
    query_frozen_ = false;
    turn_spans_ = {};
    retrieval_pinned_ = false;
    keep_selected_ = false;
    prefill_capture_ = true;
}

void llama_memory_kvmem::reset_policy() {
    ++attention_epoch_;
    reset_turn_policy();
    row_positions_.clear();
    decode_mean_reset();
    reset_query_acc();
    if (raw_) {
        raw_->clear();
    }
    if (runtime_) {
        runtime_->truncate_to(0);
    }
    host_mirror_stale_ = false;
    reset_slots();
}

void llama_memory_kvmem::begin_cached_turn(bool reset_query) {
    if (historical_capture_) throw std::runtime_error("historical capture baseline does not support continuation");
    harvest_flush();
    query_frozen_ = false;
    decode_mean_flush();
    decode_mean_discard();
    if (reset_query) {
        reset_query_acc();
    }
    retrieval_pinned_ = false;
    keep_selected_ = false;
    prefill_capture_ = true;
}

void llama_memory_kvmem::truncate_cached(uint32_t n_past) {
    // Keyed off the block table alone, so it cannot reach a host mirror the
    // full-seq_rm branch left behind: see host_mirror_stale_ there.
    if (runtime_ && n_past >= runtime_->store().total_tokens()) return;
    if (historical_capture_) throw std::runtime_error("historical capture baseline does not support truncation");
    ++attention_epoch_;
    harvest_flush();
    harvest_gpu_v_commit();
    if (row_positions_.size() > n_past) row_positions_.resize(n_past);
    if (mtp_) mtp_->truncate_cached(n_past);
    if (raw_) {
        raw_->truncate_to(n_past);
    }
    if (runtime_ && n_past < runtime_->store().total_tokens()) {
        runtime_->truncate_to(n_past);
    }
}

void llama_memory_kvmem::set_replay(bool replay) {
    if (replay && historical_capture_) throw std::runtime_error("historical capture baseline does not support replay");
    if (replay && !replay_) {
        harvest_flush();
        harvest_gpu_v_commit();
        ++attention_epoch_;
        const uint32_t begin = std::max(0, explicit_spans_ ? turn_spans_.replay_begin : query_begin_);
        if (raw_) raw_->invalidate_packed_from(begin);
        if (mtp_) mtp_->invalidate_packed_from(begin);
    }
    replay_ = replay;
}

bool llama_memory_kvmem::conv_swap_supported(std::string & reason) const {
    if (historical_capture_) { reason = "historical_capture_single_request"; return false; }
    if (!runtime_ || !raw_ || !kv_) {
        reason = "no_store";
        return false;
    }
    // Without flash attention V is never mirrored to host: harvest_gpu_v
    // returns immediately and RawKvStoreConfig::v_gpu_row_bytes stays 0, so a
    // drained conversation could not be restaged. A swap drains everything on
    // every switch, so refuse to arm instead of restaging stale V.
    if (v_trans_) {
        reason = "v_trans";
        return false;
    }
    // Each RawKvStore opens its own arena, sized for one conversation and named
    // by a fixed nvme_file.
    if (kvmem_current_execution().params.raw_k_nvme) {
        reason = "raw_k_nvme";
        return false;
    }
    return true;
}

bool llama_memory_kvmem::conv_can_drain(std::string & reason) const {
    // A replay hole is described by the GPU snapshot planes and the fold
    // window, neither of which the server's recurrent byte dump covers.
    if (replay_) {
        reason = "replay_in_flight";
        return false;
    }
    if (recr_ && recr_->replay_recording) {
        reason = "gdn_replay_recording";
        return false;
    }
    // A prepared-but-unapplied plan would strand pending_gpu_frees_. Requests
    // are serialized and every prepare is applied in the same request, so this
    // should never fire.
    if (runtime_ && runtime_->pending()) {
        reason = "plan_pending";
        return false;
    }
    // The host mirror no longer describes the block table (see seq_rm), so its
    // packed K/V cannot be trusted to restage this conversation.
    if (host_mirror_stale_) {
        reason = "host_mirror_stale";
        return false;
    }
    return true;
}

bool llama_memory_kvmem::conv_compatible(const llama_memory_kvmem & other) const {
    return &model_ == &other.model_ && kv_->type_k() == other.kv_->type_k() &&
        kv_->type_v() == other.kv_->type_v() && n_slots_ == other.n_slots_ &&
        (mtp_ != nullptr) == (other.mtp_ != nullptr) &&
        rt_cfg_.store.block_tokens == other.rt_cfg_.store.block_tokens &&
        rt_cfg_.store.select_budget == other.rt_cfg_.store.select_budget &&
        rt_cfg_.cpu_bytes == other.rt_cfg_.cpu_bytes;
}

std::unique_ptr<llama_memory_kvmem::ConvStore> llama_memory_kvmem::make_conv() {
    auto conv = std::make_unique<ConvStore>();
    conv->model = &model_;
    conv->type_k = kv_->type_k();
    conv->type_v = kv_->type_v();
    conv->slots = n_slots_;
    conv->has_mtp = mtp_ != nullptr;
    conv->runtime = std::make_unique<kvmem::KvMemRuntime>(rt_cfg_);
    conv->raw = std::make_unique<kvmem::RawKvStore>(raw_cfg_);
    conv->q_sum.assign(n_layer_, std::vector<float>(n_head_ * n_embd_head_, 0.0f));
    conv->q_count.assign(n_layer_, 0);
    if (mtp_) {
        conv->mtp_raw = mtp_->make_raw();
    }
    return conv;
}

uint32_t llama_memory_kvmem::conv_n_tokens(const ConvStore & conv) const {
    return conv.runtime ? conv.runtime->store().total_tokens() : 0;
}

uint64_t llama_memory_kvmem::conv_host_bytes(const ConvStore & conv) const {
    if (!conv.raw) {
        return 0;
    }
    uint64_t bytes = conv.raw->allocated_bytes() + (conv.mtp_raw ? conv.mtp_raw->allocated_bytes() : 0);
    bytes += conv.runtime->allocated_bytes() + conv.row_positions.capacity()*sizeof(RowPosition);
    bytes += conv.q_count.capacity()*sizeof(uint32_t) + conv.resident.capacity()*sizeof(uint32_t);
    bytes += conv.q_sum.capacity()*sizeof(std::vector<float>);
    for (const auto & q : conv.q_sum) bytes += q.capacity()*sizeof(float);
    return bytes;
}

uint64_t llama_memory_kvmem::host_bytes() const {
    if (!raw_) {
        return 0;
    }
    uint64_t bytes = raw_->allocated_bytes() + (mtp_ ? mtp_->host_bytes() : 0);
    bytes += runtime_->allocated_bytes() + row_positions_.capacity()*sizeof(RowPosition);
    bytes += q_count_.capacity()*sizeof(uint32_t);
    bytes += q_sum_.capacity()*sizeof(std::vector<float>);
    for (const auto & q : q_sum_) bytes += q.capacity()*sizeof(float);
    return bytes;
}

uint64_t llama_memory_kvmem::host_capacity(uint32_t tokens) const {
    // Bound retained checkpoints as well as raw KV. A single sequence writes
    // one logical recurrent row, irrespective of GPU rollback plane count.
    uint64_t checkpoint = 65536;
    if (recr_) {
        for (const auto * list : { &recr_->r_l, &recr_->s_l, &recr_->p_l })
            for (const auto * t : *list) if (t) checkpoint += ggml_row_size(t->type, t->ne[0]);
    } else {
        // Dense memory does not implement PARTIAL_ONLY: its server checkpoint
        // contains the attention working set and cell metadata.
        for (const auto & buffer : kv_->memory_breakdown()) checkpoint += buffer.second;
        checkpoint += uint64_t(kv_size_)*64;
    }
    checkpoint += uint64_t(n_layer_)*(n_embd_k_ + uint64_t(n_head_)*n_embd_head_)*sizeof(float);
    // MTP carry, token/prompt indexes, query state and bounded allocator slack.
    return 8*checkpoint + 8*1024*1024 + uint64_t(tokens)*64 +
        raw_->capacity_bytes(tokens, uint32_t(kv_->get_layer_ids().size())) + (mtp_ ? mtp_->capacity_bytes(tokens) : 0) +
        uint64_t(tokens)*sizeof(RowPosition)*2 +
        ((uint64_t(tokens)/block_tokens_) + 2)*sizeof(kvmem::KvMemBlock)*2;
}

std::unique_ptr<llama_memory_kvmem::ConvStore> llama_memory_kvmem::detach_conv() {
    // Quiesce every writer of raw_, q_sum_ and the GPU cells before either
    // moves. The order matters and each step closes a different queue:
    //
    // 1. The stage-in slab is a process global holding packed K/V aimed at
    //    cells this store still owns. Land it here, or the incoming store's
    //    first flush would write these bytes into its own cells.
    // 2. decode_mean_reset() writes the partial-block running mean into raw_
    //    and sets decode_mean_block_ = ~0u, which makes the incoming
    //    conversation take decode_mean_add_range()'s block-change branch and
    //    zero the process-global mean-K accumulator. decode_mean_print_sum()
    //    is deliberately not called: it latches once per process.
    // 3. harvest_flush() waits for the harvest worker queue and both
    //    CaptureD2hPipe slots under harvest_w_->mu. That mutex is also the
    //    release/acquire edge for the worker's last write to raw_ and q_sum_
    //    through this, so the wait must happen even when the pipe looks idle:
    //    skipping it is a data race on vectors that are about to be moved.
    //    The pipe itself is kept, as in the destructor's ordering; d2h_free()
    //    would drop engine-sized staging buffers a swap does not change.
    // 4. harvest_gpu_v_commit() drains the stage-out slab into raw_ and clears
    //    harvest_gpu_queued_, but it does not wait for the store's own writes.
    //    The three existing call sites get away with that only because each
    //    ends in a raw_ mutation (truncate_to / invalidate_packed_from /
    //    clear) that waits first. A detach mutates nothing, so it waits here.
    // The harvest worker is drained, never stopped: after the wait it parks in
    // harvest_loop() holding no reference to raw_ or q_sum_.
    kvmem_stagein_flush_sync(nullptr, nullptr, nullptr, nullptr);
    decode_mean_reset();
    harvest_flush();
    harvest_gpu_v_commit();
    if (raw_) {
        raw_->wait_writes();
    }
    if (mtp_) {
        mtp_->harvest_flush();
    }
    // The one condition that would corrupt the store being moved: a slot still
    // in flight means the worker can still call harvest_from_host() and write
    // raw_ and q_sum_ through this. harvest_flush() rules it out, both by
    // waiting on the worker and by committing synchronously without one, so
    // this waits rather than only warning: a warning that walks into the
    // hazard it names reads as covered and is not.
    if (d2h_) {
        for (int slot = 0; slot < 2; ++slot) {
            if (!d2h_->slots[slot].inflight) {
                continue;
            }
            LLAMA_LOG_WARN("%s: KVMem capture pipe slot %d still in flight at a store detach\n",
                    __func__, slot);
            harvest_wait_slot(slot);
        }
    }
    // Rows of the outgoing conversation. An unconsumed ubatch note would make
    // the incoming conversation's first harvest write to this conversation's
    // rows. Each batch also owns the original historical writer contexts.
    // prepare_ubatches() refills pos_queue_ on every init_batch, so dropping
    // it here costs the incoming conversation nothing.
    //
    // pending_capture_ and graph_has_* are deliberately left alone: they
    // describe the llama_context's cached graph, which a swap does not
    // rebuild. capture_on_new_graph() owns them, harvest_pending() needs
    // pending_capture_ to survive a reused graph, and capture_can_reuse()
    // compares graph_has_q_/graph_has_k_ against that same graph.
    if (!pos_queue_.empty()) {
        kvmem_diag("KVMEM_STORE_DETACH_PENDING posq=%zu capture=%zu\n",
                pos_queue_.size(), pending_capture_.size());
    }
    pos_queue_.clear();
    cur_capture_ = {};

    auto conv = std::make_unique<ConvStore>();
    {
        auto & store = runtime_->store();
        // Per-block residency lives inside the bundle (KvMemBlock::gpu_slot)
        // while the slot pool is engine-side, and every conversation numbers
        // its rows from 0, so the working set must be fully drained to host.
        // tier is the residency truth, not gpu_slot: the slot is one field of a
        // pair KvMemStore::set_block_tier writes together
        // (kvmem/src/host/kvmem_store.cpp:249-276), so this filters on the half
        // that names the tier. This is the same set set_selection stages out.
        for (uint32_t id = 0; id < store.block_count(); ++id) {
            const kvmem::KvMemBlock & b = store.blocks()[id];
            if (b.tier == kvmem::KvTier::GPU && b.gpu_slot >= 0 && b.n_tokens > 0) {
                conv->resident.push_back(id);
            }
        }
        // An empty selection puts every GPU block in stage_out, and
        // apply_plan_to_kv is the drain half of the sequence relayout already
        // uses: harvest_gpu_v + mtp_->on_stage_out per block, commit,
        // spill_outgoing, seq_rm_logical per block, admit_incoming.
        const kvmem::KvMemPlan plan = runtime_->prepare_selection({});
        trace_plan("conv_detach", plan);
        apply_plan_to_kv(plan);
        // Belt and braces: a detached store carrying a stale gpu_slot would
        // make gpu_kv_already_resident() and attention_view() read cells that
        // belong to another conversation. KvMemStore::set_block_tier already
        // clears the slot on the way off GPU, so this loop finds nothing
        // unless a block never reached a lower tier. Demote rather than only
        // clearing the slot: tier GPU with gpu_slot -1 is read three
        // incompatible ways downstream (stage_in wants a fresh slot,
        // resident_tokens counts it as absent, set_selection will not stage it
        // out again), so clearing alone would trade one inconsistent state for
        // another on a block conv->resident will ask to be restaged.
        for (uint32_t id = 0; id < store.block_count(); ++id) {
            const kvmem::KvMemBlock & b = store.blocks()[id];
            if (b.gpu_slot < 0) {
                continue;
            }
            LLAMA_LOG_WARN("%s: KVMem block %u still held GPU slot %d at tier %d after a full drain\n",
                    __func__, id, (int) b.gpu_slot, (int) b.tier);
            store.set_block_tier(id, kvmem::KvTier::CPU, b.cpu_slot, b.nvme_slot);
        }
    }
    // Issued on the borrowed attention cache, not through the hybrid override,
    // so the recurrent half is deliberately left alone: the server owns that as
    // a byte snapshot. This also drops an untrimmed speculative tail, which
    // attention_view() counts as live attention.
    if (kv_) {
        (void) kv_->seq_rm(0, -1, -1);
    }
    if (mtp_) {
        mtp_->drop_gpu();
    }
    reset_slots();
    // Server-held attention views and selections must not validate across a
    // swap, so the epoch stays engine-side and only ever increases.
    ++attention_epoch_;
    // Post-detach invariant: nothing of this conversation is left on the GPU,
    // and the slot pool is whole again for the incoming one.
    kvmem_diag("KVMEM_STORE_DETACH rows=%u resident=%zu cells_used=%u free_slots=%zu/%u\n",
            runtime_ ? runtime_->store().total_tokens() : 0, conv->resident.size(),
            (unsigned) (kv_ ? kv_->get_cells(0).get_used() : 0), free_slots_.size(), n_slots_);

    // The drain above wrote the packed K/V of every resident block through
    // harvest_gpu_v_commit(), which does not wait for the store's own writes.
    if (raw_) {
        raw_->wait_writes();
    }
    if (mtp_) {
        mtp_->harvest_flush();
    }

    // Unbind before moving ownership; this validates that the drain finished.
    runtime_->rebind_backend(nullptr);
    // Nothing from here to the return may throw. Past these two moves this
    // object owns no store at all and swap_conv's drain catch cannot put them
    // back: the move-assignments are noexcept, swap_raw() only waits on the
    // follower's io thread and fills a vector it sized at construction, and
    // reset_turn_policy() assigns scalars.
    conv->model = &model_;
    conv->type_k = kv_->type_k();
    conv->type_v = kv_->type_v();
    conv->slots = n_slots_;
    conv->has_mtp = mtp_ != nullptr;
    conv->runtime = std::move(runtime_);
    conv->raw = std::move(raw_);
    // One statement after the trunk's own mirror, the way truncate_cached()
    // drives mtp_->truncate_cached(): the follower's packed draft K/V is keyed
    // by the trunk's block ids and means nothing beside another store.
    if (mtp_) {
        conv->mtp_raw = mtp_->swap_raw(nullptr);
    }
    conv->row_positions = std::move(row_positions_);
    conv->q_sum = std::move(q_sum_);
    conv->q_count = std::move(q_count_);
    row_positions_.clear();
    q_sum_.clear();
    q_count_.clear();
    reset_turn_policy();
    return conv;
}

bool llama_memory_kvmem::attach_conv(std::unique_ptr<ConvStore> conv) {
    // The mirror image of detach_conv(). Every member this object owns per
    // conversation is installed first, in container move-assignments that
    // cannot throw, so this object owns the incoming conversation before
    // anything that can throw runs: that is what makes swap_conv's catch able
    // to reset it to empty instead of losing it. The size repair that follows
    // the moves does allocate and can throw; reset_query_acc() resizes as well
    // as zeroes, so the reset_policy() in that catch leaves an accumulator
    // score_retrieval() and get_query() can index over 0..n_layer_-1, which
    // they do with no size guard.
    //
    // The follower's mirror follows, because acquiring it can throw. It is
    // keyed by this store's block ids, so a bundle made before the follower
    // existed has none and gets a fresh one to rebuild at the next stage-out.
    runtime_ = std::move(conv->runtime);
    raw_ = std::move(conv->raw);
    row_positions_ = std::move(conv->row_positions);
    q_sum_ = std::move(conv->q_sum);
    q_count_ = std::move(conv->q_count);
    runtime_->rebind_backend(&backend_);
    if (q_sum_.size() != n_layer_ || q_count_.size() != n_layer_) {
        q_sum_.assign(n_layer_, std::vector<float>(n_head_ * n_embd_head_, 0.0f));
        q_count_.assign(n_layer_, 0);
    }
    if (mtp_) {
        std::unique_ptr<kvmem::RawKvStore> mirror = std::move(conv->mtp_raw);
        if (!mirror) {
            // The follower mirror is an accept-rate input and never a
            // correctness one, so a failed allocation must not fail the
            // attach. It does cost this conversation its draft mirror for the
            // whole of this residency: with the follower's mirror null,
            // harvest_k() and harvest_v() return at their first guard, so no
            // stage-out mirrors anything. The allocation is retried at the
            // next attach of this bundle, because the detach parks a null
            // mirror (see the class comment in llama-memory-kvmem-mtp.h).
            try {
                mirror = mtp_->make_raw();
            } catch (const std::exception & e) {
                LLAMA_LOG_WARN("%s: KVMem MTP mirror allocation failed (%s); the follower keeps "
                        "no packed draft K/V for this conversation\n", __func__, e.what());
            }
        }
        mtp_->swap_raw(std::move(mirror));
    }
    ++attention_epoch_;
    reset_slots();
    // Not reset_policy(): its destructive half would clear the store just
    // attached. The server calls llama_kvmem_begin_cached_turn() and
    // llama_kvmem_set_request_span() after the attach anyway.
    reset_turn_policy();
    // runtime_ and raw_ are dereferenced unguarded from here on, and so is
    // runtime_ in detach_conv(): make_conv() allocates both or throws, and
    // every other bundle came from a detach that moved two non-null members
    // out of this object. There is deliberately no null check reporting a
    // miss: false means "attached, holding no rows the caller may decode
    // against", and llama_kvmem_store_switch acts on it by naming this store
    // as the attached one.
    //
    // An empty store always arrives beside an empty follower mirror: every
    // path that empties one (reset_policy here, the refusal in swap_conv)
    // empties the other. Report false, which is what the return value means
    // everywhere else: the attached store holds no rows the caller may decode
    // against. A freshly created store lands here beside an empty payload, so
    // the server's !restaged branch is a no-op for it; a store the refusal
    // path wiped lands here beside a payload that still claims rows, and that
    // is exactly the mismatch the branch exists to catch.
    if (runtime_->store().total_tokens() == 0) {
        return false;
    }

    // write_block_to_gpu() silently skips a layer with no packed bytes, which
    // would leave live cells holding another conversation's K. This is the same
    // predicate harvest_full_blocks_async() uses; it must not be dropped.
    const char * miss = nullptr;
    {
        const kvmem::KvMemStore & store = runtime_->store();
        for (uint32_t id : conv->resident) {
            if (id >= store.block_count()) {
                miss = "block_gone";
                break;
            }
            const kvmem::KvMemBlock & b = store.blocks()[id];
            if (b.n_tokens == 0) {
                continue;
            }
            for (uint32_t il = 0; il < n_layer_ && !miss; ++il) {
                if (!kvmem_cache_has_layer(kv_, static_cast<int32_t>(il))) {
                    continue;
                }
                if (!raw_->has_k_gpu(id, il, b.n_tokens)) {
                    miss = "packed_k_missing";
                } else if (!v_trans_ && !raw_->has_v_gpu(id, il, b.n_tokens)) {
                    miss = "packed_v_missing";
                }
            }
            if (miss) {
                break;
            }
        }
    }
    if (miss) {
        kvmem_diag("KVMEM_STORE_ATTACH_MISS reason=%s rows=%u resident=%zu\n",
                miss, runtime_->store().total_tokens(), conv->resident.size());
        reset_policy();
        // reset_policy() clears the trunk's mirror only. Clear the follower's
        // in lockstep so an empty store never sits beside a mirror still
        // holding draft K/V for block ids the next prefill reuses.
        if (mtp_) {
            mtp_->clear(true);
        }
        return false;
    }

    // Nothing sits at tier GPU, so every requested block lands in stage_in and
    // admit_incoming allocates one slot each through the backend. Slots come
    // out ascending because alloc_slot pops the back of a list reset_slots
    // filled n_slots_-1..0, which is the layout attention_view() calls
    // canonical. Positions need no fixup: the pool keeps original positions and
    // the packed K in raw_ was harvested already RoPE'd at that position.
    const kvmem::KvMemPlan plan = runtime_->prepare_selection(conv->resident);
    trace_plan("conv_attach", plan);
    apply_plan_to_kv(plan);
    auto & store = runtime_->store();
    uint32_t n_raw = 0;
    uint32_t n_skip = 0;
    for (uint32_t id : plan.stage_in) {
        if (id >= store.block_count() || store.blocks()[id].gpu_slot < 0) {
            continue;
        }
        if (gpu_kv_already_resident(id)) {
            n_skip++;
            continue;
        }
        write_block_to_gpu(id);
        n_raw++;
    }
    kvmem_stagein_flush_sync(retr_.enabled ? &retr_.copy_us : nullptr,
                             retr_.enabled ? &retr_.rope_us : nullptr,
                             retr_.enabled ? &retr_.hadamard_us : nullptr,
                             retr_.enabled ? &retr_.set_us : nullptr);
    if (mtp_) {
        // A follower coverage miss leaves its cells empty rather than
        // labelling them with this block's positions, because the bytes in
        // them are the previous conversation's draft K. That holds for the
        // rest of the request and not only for this call: the strictness is a
        // per-slot mark swap_raw() set above, which follow_retrieval() reads
        // on the retrieval path too, and which only a full packed write
        // clears.
        mtp_->follow_retrieval();
    }
    if (trace_) {
        kvmem_diag("KVMEM_STORE_ATTACH rows=%u blocks=%zu raw=%u skip=%u free_slots=%zu\n",
                store.total_tokens(), plan.stage_in.size(), n_raw, n_skip, free_slots_.size());
    }
    return true;
}

// Best-effort return to "attached, holding nothing". Both of swap_conv's
// repair paths call this and neither may throw out of it: the drain path
// rethrows the drain's own exception afterwards, and the attach path runs
// after the two bundles have changed hands, where an escape would make
// llama_kvmem_store_switch's catch skip a handover this object has already
// made -- parking each conversation's payload beside the other's KV, which is
// silent wrong-content service rather than a cache miss. So every step runs
// under its own catch and the whole function is noexcept.
void llama_memory_kvmem::conv_reset_to_empty(const char * what) noexcept {
    try {
        set_replay(false);
        if (runtime_) {
            // A prepared-but-unapplied plan would strand its queued slot frees
            // and refuse the next swap with plan_pending. reset_policy() below
            // rebuilds the free-slot list those frees would have fed.
            runtime_->discard_pending();
        }
        if (kv_) {
            // reset_policy() clears the host mirror and the block table but
            // not the attention cells, so empty them the way detach_conv()
            // does: apply_plan_to_kv() may already have admitted cells for a
            // store that is about to be emptied.
            (void) kv_->seq_rm(0, -1, -1);
        }
        // Set before reset_policy(), which clears it again on its way out. A
        // store whose cells are gone while its block table still reads full is
        // the one state store_n_tokens() cannot expose, so if the reset below
        // does not complete, this mark is what stops the half-repaired store
        // from being carried into another conversation: conv_can_drain()
        // refuses it and the next swap clears it instead.
        host_mirror_stale_ = true;
        reset_slots();
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("%s: KVMem %s repair could not quiesce the active store (%s)\n",
                __func__, what, e.what());
    } catch (...) {
        LLAMA_LOG_ERROR("%s: KVMem %s repair could not quiesce the active store\n",
                __func__, what);
    }
    // reset_policy() -> runtime_->truncate_to(0) -> wait_prefetch() rethrows a
    // stored NVMe prefetch error exactly once: wait_prefetch() clears the
    // futures before it rethrows, so the second attempt gets past it and does
    // empty the block table. Without the retry the store would keep reading
    // full with a working set that no longer describes it.
    bool emptied = false;
    for (int attempt = 1; attempt <= 2 && !emptied; ++attempt) {
        try {
            reset_policy();
            emptied = true;
        } catch (const std::exception & e) {
            LLAMA_LOG_ERROR("%s: KVMem %s repair could not empty the active store "
                    "(%s, attempt %d)\n", __func__, what, e.what(), attempt);
        } catch (...) {
            LLAMA_LOG_ERROR("%s: KVMem %s repair could not empty the active store "
                    "(attempt %d)\n", __func__, what, attempt);
        }
    }
    // The follower mirror is keyed by the trunk's block ids, so an emptied
    // store must never sit beside a mirror still holding draft K/V for ids the
    // next prefill reuses.
    try {
        if (mtp_) {
            mtp_->clear(true);
        }
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("%s: KVMem %s repair could not clear the follower mirror (%s)\n",
                __func__, what, e.what());
    } catch (...) {
        LLAMA_LOG_ERROR("%s: KVMem %s repair could not clear the follower mirror\n",
                __func__, what);
    }
    if (!emptied) {
        LLAMA_LOG_ERROR("%s: KVMem %s repair left the active store unreset; it is marked "
                "host_mirror_stale, so the next switch clears it rather than carrying it\n",
                __func__, what);
    }
}

bool llama_memory_kvmem::swap_conv(std::unique_ptr<ConvStore> & conv) {
    if (!conv) {
        return false;
    }
    if (conv->model != &model_ || conv->type_k != kv_->type_k() || conv->type_v != kv_->type_v() ||
            conv->slots != n_slots_ || conv->has_mtp != (mtp_ != nullptr) ||
            conv->runtime->config().store.block_tokens != rt_cfg_.store.block_tokens ||
            conv->runtime->config().store.select_budget != rt_cfg_.store.select_budget ||
            conv->runtime->config().cpu_bytes != rt_cfg_.cpu_bytes) {
        throw std::invalid_argument("incompatible conversation bundle");
    }
    std::string reason;
    if (!conv_can_drain(reason)) {
        // The outgoing conversation cannot be drained safely. Discard it rather
        // than leave the GPU holding cells two stores both claim; that is the
        // same destruction a single-store server does today when a different
        // conversation arrives. The return value describes the incoming store
        // only, so the wipe is reported the one way a caller can act on: the
        // parked handle's llama_kvmem_store_rows() drops to zero, which the
        // server cross-checks after every switch.
        LLAMA_LOG_WARN("%s: KVMem store swap cannot drain the active store (%s); clearing it\n",
                __func__, reason.c_str());
        set_replay(false);
        clear(true);
        // The follower mirror is keyed by the trunk's block ids, and the server
        // is switching rather than clearing, so it will not clear the draft
        // context for us the way memory_clear_all() does. Leaving it would let
        // follow_retrieval() write this conversation's draft K into a block id
        // the next one re-prefills.
        if (mtp_) {
            mtp_->clear(true);
        }
    }
    // All-or-nothing in the handles. detach_conv() moves both bundle members
    // out in a run of noexcept move-assignments with nothing allocating past
    // them, so a throw out of detach_conv() moves neither bundle: the caller's
    // handle still holds the incoming conversation and this object still owns
    // the outgoing one. The outgoing store's contents are a separate question.
    // The refusal above may already have cleared them, and the drain can throw
    // with a plan half applied (spill_outgoing() rethrows a stored prefetch
    // exception and an NVMe write can fail), which leaves a working set that no
    // longer describes the store while total_tokens() still reads full -- the
    // one thing the server's post-switch cross-check cannot see. So empty this
    // store on that path and let it be read as the cache miss it now is.
    //
    // Past the two moves below the outgoing conversation is the caller's, and
    // nothing from there on throws: the attach's repair is noexcept and its
    // failure is reported as false. The outgoing bundle used to stay in a local
    // until the last statement, so a throw inside the attach destroyed a whole
    // conversation's host KV and left the caller's handle moved-from and null
    // while the pool still named it.
    std::unique_ptr<ConvStore> outgoing;
    try {
        outgoing = detach_conv();
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("%s: KVMem store drain failed (%s); the active store is reset to empty\n",
                __func__, e.what());
        conv_reset_to_empty("drain");
        throw;
    } catch (...) {
        LLAMA_LOG_ERROR("%s: KVMem store drain failed; the active store is reset to empty\n",
                __func__);
        conv_reset_to_empty("drain");
        throw;
    }
    std::unique_ptr<ConvStore> incoming = std::move(conv);
    conv = std::move(outgoing);
    try {
        return attach_conv(std::move(incoming));
    } catch (const std::exception & e) {
        // occupy_in throws on missing cache row position metadata, and the
        // staging paths throw on a CUDA failure. attach_conv() takes the
        // incoming store's members before any of that, so this object owns it:
        // reset it to empty and keep it active, which is the state the
        // coverage-miss path above leaves and which the server reads as an
        // ordinary cache miss from the false return. The reset is noexcept,
        // and catch (...) is here for the same reason it is: both bundles have
        // already changed hands, so an escape from this handler would make
        // llama_kvmem_store_switch's catch skip a handover that already
        // happened and leave each conversation's payload beside the other's KV.
        LLAMA_LOG_ERROR("%s: KVMem store attach failed (%s); the incoming store is reset to empty\n",
                __func__, e.what());
        conv_reset_to_empty("attach");
        return false;
    } catch (...) {
        LLAMA_LOG_ERROR("%s: KVMem store attach failed; the incoming store is reset to empty\n",
                __func__);
        conv_reset_to_empty("attach");
        return false;
    }
}

llama_pos llama_memory_kvmem::recr_pos_max() const {
    return recr_ ? recr_->seq_pos_max(0) : (llama_pos) -1;
}

int32_t llama_memory_kvmem::alloc_slot() {
    if (free_slots_.empty()) {
        return -1;
    }
    const int32_t slot = free_slots_.back();
    free_slots_.pop_back();
    return slot;
}

void llama_memory_kvmem::free_slot(int32_t slot) {
    if (slot < 0) {
        return;
    }
    if (size_t(slot) < historical_gpu_readers_.size() &&
            (historical_gpu_readers_[slot] || historical_gpu_retired_[slot]))
        throw std::logic_error("GPU slot returned before historical sources completed");
    free_slots_.push_back(slot);
}

void llama_memory_kvmem::complete_historical_gpu_source(int32_t slot) {
    if (slot < 0) return;
    if (size_t(slot) >= historical_gpu_readers_.size() || !historical_gpu_readers_[slot])
        throw std::logic_error("historical GPU source completion without ownership");
    if (--historical_gpu_readers_[slot] == 0 && historical_gpu_retired_[slot]) {
        historical_gpu_retired_[slot] = 0;
        free_slot(slot);
        ++historical_gpu_returned_;
        if (trace_) kvmem_diag("KVMEM_TRACE historical_gpu_return slot=%d\n", slot);
    }
}

int32_t llama_memory_kvmem::peek_free_slot() const {
    if (free_slots_.empty()) {
        return -1;
    }
    return free_slots_.back();
}

bool llama_memory_kvmem::slot_for_orig_pos(llama_pos pos, int32_t * slot, uint32_t * off) const {
    if (!slot || !off || pos < 0 || !runtime_) {
        return false;
    }
    const auto & st = runtime_->store();
    const uint32_t bt = block_tokens_;
    if (bt == 0) {
        return false;
    }
    const int32_t bid = st.block_id_containing(static_cast<uint32_t>(pos));
    if (bid >= 0) {
        const kvmem::KvMemBlock & b = st.blocks()[static_cast<uint32_t>(bid)];
        if (b.gpu_slot < 0) {
            return false;
        }
        *slot = b.gpu_slot;
        *off = static_cast<uint32_t>(pos) - b.orig_pos_start;
        return *off < bt;
    }
    // Draft placement beyond a fixed historical span needs its own reserved
    // handle/slot. Do not guess a slot from positional legacy arithmetic.
    if (runtime_->historical_shadow_directory().size() != 0) return false;
    // Token not yet on the target store (MTP draft before verify). Stay in
    // the last GPU block if it still has room; otherwise peek the next free
    // slot without popping so target's next alloc_slot() takes the same one.
    if (!st.blocks().empty()) {
        const kvmem::KvMemBlock & last = st.blocks().back();
        if (last.gpu_slot >= 0 &&
            static_cast<uint32_t>(pos) >= last.orig_pos_start &&
            static_cast<uint32_t>(pos) < last.orig_pos_start + bt) {
            *slot = last.gpu_slot;
            *off = static_cast<uint32_t>(pos) - last.orig_pos_start;
            return true;
        }
    }
    const uint32_t orig = (static_cast<uint32_t>(pos) / bt) * bt;
    *off = static_cast<uint32_t>(pos) - orig;
    *slot = peek_free_slot();
    return *slot >= 0 && *off < bt;
}

uint32_t llama_memory_kvmem::resident_tokens() const {
    uint32_t n = 0;
    for (const auto & b : runtime_->store().blocks()) {
        if (b.gpu_slot >= 0) {
            n += b.n_tokens;
        }
    }
    return n;
}

void llama_memory_kvmem::trace_plan(const char * tag, const kvmem::KvMemPlan & plan) const {
    if (!trace_) {
        return;
    }
    uint32_t skip = 0;
    for (const auto & r : plan.remaps) {
        if (r.skip) {
            skip++;
        }
    }
    kvmem_diag("KVMEM_TRACE %s stage_in=%zu stage_out=%zu skip=%u gpu_reused=%u window=%u free_slots=%zu\n",
            tag,
            plan.stage_in.size(),
            plan.stage_out.size(),
            skip,
            plan.gpu_reused_blocks,
            plan.total_window_tokens,
            free_slots_.size());
}

void llama_memory_kvmem::apply_plan_to_kv(const kvmem::KvMemPlan & plan) {
    if (!plan.stage_in.empty() || !plan.stage_out.empty()) ++attention_epoch_;
    auto & store = runtime_->store();
    for (uint32_t id : plan.stage_out) {
        if (id == decode_mean_block_) {
            decode_mean_flush();
            break;
        }
    }
    {
        const int64_t t0 = ggml_time_us();
        for (uint32_t id : plan.stage_out) {
            harvest_gpu_v(id);
            if (mtp_) {
                mtp_->on_stage_out(id);
            }
        }
        harvest_gpu_v_commit();
        runtime_->spill_outgoing();
        if (retr_.enabled) {
            retr_.stage_out_us += ggml_time_us() - t0;
        }
    }
    {
        const int64_t t0 = ggml_time_us();
        for (uint32_t id : plan.stage_out) {
            if (id >= store.block_count()) {
                continue;
            }
            const kvmem::KvMemBlock & b = store.blocks()[id];
            if (b.n_tokens > 0) {
                kv_->seq_rm_logical(0, static_cast<llama_pos>(b.orig_pos_start),
                            static_cast<llama_pos>(b.orig_pos_end()));
            }
        }
        if (retr_.enabled) {
            retr_.seq_rm_us += ggml_time_us() - t0;
        }
    }
    {
        const int64_t t0 = ggml_time_us();
        runtime_->admit_incoming();
        if (retr_.enabled) {
            retr_.admit_us += ggml_time_us() - t0;
        }
    }
}

bool llama_memory_kvmem::gpu_kv_already_resident(uint32_t block_id) const {
    if (!kv_ || block_id >= runtime_->store().block_count()) {
        return false;
    }
    const kvmem::KvMemBlock & b = runtime_->store().blocks()[block_id];
    if (b.gpu_slot < 0 || b.n_tokens == 0) {
        return false;
    }
    const llama_kv_cells & cells = kv_->get_cells(0);
    const uint32_t idx = static_cast<uint32_t>(b.gpu_slot) * block_tokens_;
    if (idx >= cells.size() || cells.is_empty(idx)) {
        return false;
    }
    return cells.ext_get(idx).logical_pos == static_cast<llama_pos>(b.orig_pos_start);
}

void llama_memory_kvmem::occupy_in(llama_kv_cache * cache, uint32_t block_id) {
    auto & store = runtime_->store();
    if (block_id >= store.block_count()) {
        return;
    }
    const kvmem::KvMemBlock & blk = store.blocks()[block_id];
    if (blk.gpu_slot < 0 || blk.n_tokens == 0) {
        return;
    }
    const uint32_t nt = blk.n_tokens;
    CellWorkspace temporary;
    auto & workspace = historical_capture_ ? cell_workspace_ : temporary;
    if (!historical_capture_) workspace.reserve(nt);
    if (nt > workspace.max_tokens || uint32_t(blk.gpu_slot) >= n_slots_)
        throw std::runtime_error("cell occupancy exceeds prepared workspace");
    auto & sinfo = workspace.slots;
    sinfo.idxs[0].resize(nt);
    auto & pos = workspace.positions;
    auto & logical = workspace.logical;
    auto & n_seq_id = workspace.n_seq_id;
    auto & seq_data = workspace.seq_data;
    auto & seq_id = workspace.seq_id;
    auto & output = workspace.output;
    auto & tok = workspace.tokens;
    llama_seq_id seq0 = 0;
    for (uint32_t t = 0; t < nt; ++t) {
        sinfo.idxs[0][t] = static_cast<uint32_t>(blk.gpu_slot) * block_tokens_ + t;
        logical[t] = blk.orig_pos_start + t;
        if ((size_t) logical[t] >= row_positions_.size()) {
            throw std::runtime_error("missing cache row position metadata");
        }
        const auto & row = row_positions_[logical[t]];
        for (uint32_t j = 0; j < 4; ++j) pos[j*nt+t] = row.pos[j];
        tok[t] = row.token;
        seq_id[t] = &seq_data[t];
    }
    llama_ubatch ub{};
    ub.n_tokens = nt;
    ub.n_seq_tokens = nt;
    ub.n_seqs = 1;
    ub.n_seqs_unq = 1;
    ub.n_pos = 4;
    ub.logical_pos = logical.data();
    ub.pos = pos.data();
    ub.n_seq_id = n_seq_id.data();
    ub.seq_id = seq_id.data();
    ub.seq_id_unq = &seq0;
    ub.output = output.data();
    ub.token = tok.data();
    {
        const int64_t t0 = ggml_time_us();
        cache->apply_ubatch(sinfo, ub);
        if (retr_.enabled) {
            retr_.occupy_us += ggml_time_us() - t0;
        }
    }
}

void llama_memory_kvmem::occupy_block_cells(uint32_t block_id) {
    occupy_in(kv_, block_id);
}

llama_pos llama_memory_kvmem::model_pos(uint32_t logical) const {
    if (logical < row_positions_.size()) return row_positions_[logical].pos[0];
    if (row_positions_.empty()) return (llama_pos) logical;
    const auto & last = row_positions_.back();
    // Appending text after a media chunk uses the helper's explicit cursor.
    return last.pos[0] + 1 + (llama_pos) (logical - row_positions_.size());
}

bool llama_memory_kvmem::remove_logical(llama_context * ctx, llama_pos begin, llama_pos end) {
    if (historical_capture_) return false;
    ++attention_epoch_;
    if (mtp_ && llama_get_memory(ctx) == mtp_) return mtp_->remove_logical(begin, end);
    // Recurrent rollback is only valid across consecutive text positions.
    if (recr_ && end < 0 && begin > 0 && (size_t) begin < row_positions_.size()) {
        const auto p = model_pos(begin);
        const auto prev = model_pos(begin - 1);
        const auto rmax = recr_->seq_pos_max(0);
        if (p == prev + 1 && rmax >= p && rmax - prev <= (llama_pos) recr_->n_rs_seq) {
            if (!recr_->seq_rm(0, p, -1)) return false;
        }
    }
    return kv_->seq_rm_logical(0, begin, end);
}

bool llama_memory_kvmem::layout_gpu_slots_by_orig_pos() try {
    if (capture_failed_) throw std::runtime_error("layout on a failed KVMem context");
    auto & store = runtime_->store();
    LayoutWorkspace temporary;
    auto & work = historical_capture_ ? layout_workspace_ : temporary;
    if (!historical_capture_) work.reserve(n_slots_);
    auto & items = work.items;
    items.clear();
    work.seen.assign(n_slots_, 0);
    for (const auto & b : store.blocks()) {
        if (b.gpu_slot < 0 || b.n_tokens == 0) {
            continue;
        }
        if (uint32_t(b.gpu_slot) >= n_slots_ || b.n_tokens > block_tokens_ || work.seen[b.gpu_slot])
            throw std::logic_error("layout has an invalid or duplicate physical slot");
        work.seen[b.gpu_slot] = 1;
        items.push_back({b.block_id, b.orig_pos_start, b.gpu_slot, b.n_tokens,
                         gpu_kv_already_resident(b.block_id)});
    }
    if (items.empty()) {
        return false;
    }
    std::sort(items.begin(), items.end(),
              [](const LayoutItem & a, const LayoutItem & b) { return a.orig < b.orig; });
    bool sorted = true;
    for (size_t i = 0; i < items.size(); ++i) {
        if (items[i].slot != static_cast<int32_t>(i)) {
            sorted = false;
            break;
        }
    }
    if (sorted) {
        return false;
    }

    // Layout mutates the sources of packed capture. Settle their real readers
    // before moving any slot, and before reusing shared copy descriptors.
    harvest_flush();
    harvest_gpu_v_commit();

    ++attention_epoch_;
    auto & mtp_res = work.mtp_res;
    mtp_res.assign(items.size(), 0);
    bool mtp_d2d_ok = false;
    if (mtp_) {
        for (size_t i = 0; i < items.size(); ++i) {
            if (items[i].resident &&
                mtp_->slot_holds(items[i].slot, items[i].orig)) {
                mtp_res[i] = 1;
            }
        }
    }

    const size_t krow = ggml_row_size(type_k_, n_embd_k_);
    const size_t vrow = ggml_row_size(type_v_, n_embd_v_);
    const uint64_t kspan = static_cast<uint64_t>(block_tokens_) * krow;
    const uint64_t vspan = static_cast<uint64_t>(block_tokens_) * vrow;
    const uint64_t scratch_stride = kspan + vspan;

    auto & res_ix = work.res_ix;
    res_ix.assign(items.size(), static_cast<size_t>(-1));
    size_t n_res = 0;
    for (size_t i = 0; i < items.size(); ++i) {
        if (items[i].resident) {
            res_ix[i] = n_res++;
        }
    }

    // The single scratch allocation and batched D2D kernel can only touch
    // tensors on its own GPU. Use the backend's per-tensor host path for a
    // layer split until device-local layout scratch is introduced.
    bool d2d_ok = n_res > 0 && !multi_gpu_ && !v_trans_;
    uint8_t * scratch = nullptr;
    if (d2d_ok) {
        if (!scratch_stride || scratch_stride > SIZE_MAX / n_res)
            throw std::runtime_error("layout scratch size overflow");
        scratch = kvmem_layout_workspace(n_res * scratch_stride);
        if (!scratch) {
            if (historical_capture_) throw std::logic_error("layout exceeded admitted workspace");
            d2d_ok = false;
            LLAMA_LOG_WARN("%s: layout scratch unavailable, host fallback before submission\n", __func__);
        }
        kvmem_stagein_gpu_ready((size_t) block_tokens_ * std::max(n_embd_k_, n_embd_v_),
                                std::max(krow, vrow) * (size_t) block_tokens_);
    }

    if (d2d_ok) {
        for (uint32_t il = 0; il < n_layer_; ++il) {
            if (!kvmem_cache_has_layer(kv_, static_cast<int32_t>(il))) {
                continue;
            }
            ggml_tensor * kt = kv_->get_k_storage(static_cast<int32_t>(il));
            ggml_tensor * vt = kv_->get_v_storage(static_cast<int32_t>(il));
            uint8_t * kbase = kt ? kvmem_cuda_tensor_ptr(kt) : nullptr;
            uint8_t * vbase = (vt && !v_trans_) ? kvmem_cuda_tensor_ptr(vt) : nullptr;
            if (!kbase) {
                throw std::runtime_error("layout lost its K tensor source");
            }
            if (!vbase) {
                throw std::runtime_error("layout lost its V tensor source");
            }
            auto & gsrc = work.gsrc; auto & gdst = work.gdst; auto & gbytes = work.gbytes;
            auto & ssrc = work.ssrc; auto & sdst = work.sdst; auto & sbytes = work.sbytes;
            gsrc.clear(); gdst.clear(); gbytes.clear();
            ssrc.clear(); sdst.clear(); sbytes.clear();
            for (size_t i = 0; i < items.size(); ++i) {
                if (res_ix[i] == static_cast<size_t>(-1)) {
                    continue;
                }
                const uint32_t nt = items[i].n;
                const size_t src0 = size_t(items[i].slot) * block_tokens_;
                const size_t dst0 = i * block_tokens_;
                for (auto * tensor : {kt, vt}) {
                    const size_t rows = ggml_nbytes(tensor) / tensor->nb[1];
                    if (src0 > rows || nt > rows - src0 || dst0 > rows || nt > rows - dst0)
                        throw std::logic_error("layout copy exceeds actual GPU tensor storage");
                }
                uint8_t * slot_sc = scratch + res_ix[i] * scratch_stride;
                if (kbase && krow && nt) {
                    const size_t nb = static_cast<size_t>(nt) * krow;
                    gsrc.push_back(kbase + static_cast<size_t>(src0) * krow);
                    gdst.push_back(slot_sc);
                    gbytes.push_back(nb);
                    ssrc.push_back(slot_sc);
                    sdst.push_back(kbase + static_cast<size_t>(dst0) * krow);
                    sbytes.push_back(nb);
                }
                if (vbase && vrow && nt) {
                    const size_t nb = static_cast<size_t>(nt) * vrow;
                    gsrc.push_back(vbase + static_cast<size_t>(src0) * vrow);
                    gdst.push_back(slot_sc + kspan);
                    gbytes.push_back(nb);
                    ssrc.push_back(slot_sc + kspan);
                    sdst.push_back(vbase + static_cast<size_t>(dst0) * vrow);
                    sbytes.push_back(nb);
                }
            }
            const int64_t t_g = ggml_time_us();
            kvmem_layout_copy(gsrc.data(), gdst.data(), gbytes.data(), gsrc.size());
            if (retr_.enabled) {
                retr_.layout_d2h_us += ggml_time_us() - t_g;
            }
            const int64_t t_s = ggml_time_us();
            kvmem_layout_copy(ssrc.data(), sdst.data(), sbytes.data(), ssrc.size());
            if (retr_.enabled) {
                retr_.layout_h2d_us += ggml_time_us() - t_s;
            }
        }
        if (d2d_ok && mtp_) {
            std::vector<llama_memory_kvmem_mtp::LayoutMove> mm;
            for (size_t i = 0; i < items.size(); ++i) {
                if (!mtp_res[i]) {
                    continue;
                }
                mm.push_back({items[i].slot, static_cast<int32_t>(i), items[i].n});
            }
            mtp_d2d_ok = mm.empty() || mtp_->layout_d2d(mm.data(), mm.size());
            if (!mtp_d2d_ok) {
                LLAMA_LOG_WARN("%s: MTP layout D2D failed; restoring those blocks from raw\n",
                               __func__);
            }
        }
    } else if (n_res > 0) {
        const uint64_t payload_bytes =
                static_cast<uint64_t>(n_layer_) * (krow + vrow) * block_tokens_;
        std::vector<std::vector<uint8_t>> payloads(items.size());
        const int64_t t_d2h = ggml_time_us();
        for (size_t i = 0; i < items.size(); ++i) {
            if (res_ix[i] == static_cast<size_t>(-1) || payload_bytes == 0) {
                continue;
            }
            payloads[i].assign(static_cast<size_t>(payload_bytes), 0);
            copy_gpu_block_to_host(items[i].id, items[i].slot,
                                   payloads[i].data(), payload_bytes);
        }
        if (retr_.enabled) {
            retr_.layout_d2h_us += ggml_time_us() - t_d2h;
        }
        {
            const int64_t t0 = ggml_time_us();
            for (const auto & it : items) {
                kv_->seq_rm_logical(0, static_cast<llama_pos>(it.orig),
                            static_cast<llama_pos>(it.orig + it.n));
                if (mtp_) {
                    mtp_->remove_logical(static_cast<llama_pos>(it.orig),
                                 static_cast<llama_pos>(it.orig + it.n));
                }
                store.set_block_gpu_slot(it.id, -1);
            }
            if (retr_.enabled) {
                retr_.seq_rm_us += ggml_time_us() - t0;
            }
        }
        reset_slots();
        if (trace_) {
            kvmem_diag("KVMEM_TRACE layout_orig_pos");
            for (size_t i = 0; i < items.size(); ++i) {
                fprintf(stderr, " %u", items[i].id);
            }
            fprintf(stderr, "\n");
        }
        uint32_t n_move = 0;
        uint32_t n_raw = 0;
        for (size_t i = 0; i < items.size(); ++i) {
            const int32_t slot = alloc_slot();
            if (slot < 0) {
                LLAMA_LOG_ERROR("%s: no GPU slot while laying out block %u\n",
                                __func__, items[i].id);
                return false;
            }
            store.set_block_gpu_slot(items[i].id, slot);
            store.set_block_tier(items[i].id, kvmem::KvTier::GPU, -1,
                                 store.blocks()[items[i].id].nvme_slot);
            if (items[i].resident && !payloads[i].empty()) {
                occupy_block_cells(items[i].id);
                const int64_t t_h2d = ggml_time_us();
                copy_gpu_block_from_host(items[i].id, slot,
                                         payloads[i].data(), payload_bytes);
                if (retr_.enabled) {
                    retr_.layout_h2d_us += ggml_time_us() - t_h2d;
                }
                n_move++;
            } else {
                write_block_to_gpu(items[i].id);
                n_raw++;
            }
        }
        if (retr_.enabled) {
            retr_.n_move += n_move;
            retr_.n_raw += n_raw;
            retr_.laid_out = 1;
        }
        kvmem_stagein_flush_sync(retr_.enabled ? &retr_.copy_us : nullptr,
                                 retr_.enabled ? &retr_.rope_us : nullptr,
                                 retr_.enabled ? &retr_.hadamard_us : nullptr,
                                 retr_.enabled ? &retr_.set_us : nullptr);
        if (trace_) {
            kvmem_diag("KVMEM_TRACE layout_writeback move=%u raw=%u\n", n_move, n_raw);
        }
        return true;
    }

    {
        const int64_t t0 = ggml_time_us();
        for (const auto & it : items) {
            kv_->seq_rm_logical(0, static_cast<llama_pos>(it.orig),
                        static_cast<llama_pos>(it.orig + it.n));
            if (mtp_) {
                mtp_->remove_logical(static_cast<llama_pos>(it.orig),
                             static_cast<llama_pos>(it.orig + it.n));
            }
            store.set_block_gpu_slot(it.id, -1);
        }
        if (retr_.enabled) {
            retr_.seq_rm_us += ggml_time_us() - t0;
        }
    }
    reset_slots();
    if (trace_) {
        kvmem_diag("KVMEM_TRACE layout_orig_pos");
        for (size_t i = 0; i < items.size(); ++i) {
            fprintf(stderr, " %u", items[i].id);
        }
        fprintf(stderr, "\n");
    }
    uint32_t n_move = 0;
    uint32_t n_raw = 0;
    for (size_t i = 0; i < items.size(); ++i) {
        const int32_t slot = alloc_slot();
        if (slot < 0) {
            LLAMA_LOG_ERROR("%s: no GPU slot while laying out block %u\n",
                            __func__, items[i].id);
            return false;
        }
        store.set_block_gpu_slot(items[i].id, slot);
        store.set_block_tier(items[i].id, kvmem::KvTier::GPU, -1,
                             store.blocks()[items[i].id].nvme_slot);
        if (items[i].resident && d2d_ok) {
            occupy_block_cells(items[i].id);
            if (mtp_ && mtp_res[i] && mtp_d2d_ok) {
                mtp_->occupy_block(items[i].id);
            }
            n_move++;
        } else {
            write_block_to_gpu(items[i].id);
            n_raw++;
        }
    }
    if (retr_.enabled) {
        retr_.n_move += n_move;
        retr_.n_raw += n_raw;
        retr_.laid_out = 1;
    }
    kvmem_stagein_flush_sync(retr_.enabled ? &retr_.copy_us : nullptr,
                             retr_.enabled ? &retr_.rope_us : nullptr,
                             retr_.enabled ? &retr_.hadamard_us : nullptr,
                             retr_.enabled ? &retr_.set_us : nullptr);
    if (trace_) {
        kvmem_diag("KVMEM_TRACE layout_writeback move=%u raw=%u d2d=%d\n",
                n_move, n_raw, (int) d2d_ok);
    }
    return true;
} catch (...) {
    capture_fail();
    throw;
}

bool llama_memory_kvmem::prepare_working_set(uint32_t n_new_tokens) {
    if (capture_failed_) return false;
    harvest_check_error();
    if (historical_capture_ && (historical_capture_->failed() || historical_capture_->finished() || replay_))
        return false;
    if (n_new_tokens == 0) {
        return true;
    }
    if (replay_) {
        return true;
    }
    auto & store = runtime_->store();
    const uint32_t t0 = store.total_tokens();
    if (historical_capture_) {
        if (!historical_capture_->append(n_new_tokens)) return false;
    } else if (!replay_) {
        runtime_->register_append(n_new_tokens);
    }
    const auto cancel_append = [&] {
        // A managed span cannot be truncated while its actual tasks own it.
        // This baseline rejects the context; teardown drains tasks first.
        if (historical_capture_) historical_capture_->fail();
        else runtime_->truncate_to(t0);
    };
    const uint32_t t1 = store.total_tokens();

    std::vector<uint32_t> incoming;
    for (const auto & b : store.blocks()) {
        if (b.orig_pos_end() > t0 && b.orig_pos_start < t1) {
            incoming.push_back(b.block_id);
        }
    }

    const uint32_t budget_blocks = store.prefill_budget_blocks();
    bool need_offload = false;
    // After retrieval the host store still holds every historical block, so
    // block_count() > budget is true and a recency pressure reselect would
    // drop the resurrected needle on the first generated token. Pin the
    // working set and place decode tokens into gen_reserve slots.
    if (!retrieval_pinned_ && !keep_selected_) {
        try {
            const auto slots_needed = std::count_if(incoming.begin(), incoming.end(),
                [&](uint32_t id) { return store.blocks()[id].gpu_slot < 0; });
            if (runtime_->historical_shadow_directory().size() &&
                    static_cast<size_t>(slots_needed) > free_slots_.size()) {
                // Fixed short spans occupy full physical slots. Token counts
                // alone can report room even when every slot is occupied.
                runtime_->prepare_prefill_pressure(incoming);
                need_offload = true;
            } else {
                need_offload = runtime_->maybe_offload_during_prefill(
                        n_new_tokens, resident_tokens(), kv_size_, incoming);
            }
        } catch (const std::exception & e) {
            LLAMA_LOG_ERROR("%s: KVMem reselect failed: %s\n", __func__, e.what());
            cancel_append();
            return false;
        }
    }

    if (trace_) {
        kvmem_diag("KVMEM_TRACE append n=%u total=%u resident=%u incoming_blocks=%zu "
                "over_budget=%d need_offload=%d free_slots=%zu\n",
                n_new_tokens, t1, resident_tokens(), incoming.size(),
                (int) (store.block_count() > budget_blocks), (int) need_offload,
                free_slots_.size());
    }

    if (need_offload) {
        try {
            const kvmem::KvMemPlan & plan = runtime_->last_plan();
            trace_plan("prefill_pressure", plan);
            apply_plan_to_kv(plan);
            if (perf_.enabled) {
                perf_.n_pressure += 1;
                perf_.n_pressure_out += static_cast<uint32_t>(plan.stage_out.size());
            }
        } catch (const std::exception & e) {
            LLAMA_LOG_ERROR("%s: KVMem reselect failed: %s\n", __func__, e.what());
            cancel_append();
            return false;
        }
    } else {
        for (uint32_t id : incoming) {
            if (store.blocks()[id].gpu_slot >= 0) {
                continue;
            }
            const int32_t slot = alloc_slot();
            if (slot < 0) {
                // Pinned retrieval will not evict the working set. gen_reserve
                // exhaustion is a known v1 limit (see docs/architecture.md).
                LLAMA_LOG_ERROR("%s: no free GPU slot for block %u\n", __func__, id);
                cancel_append();
                return false;
            }
            store.set_block_gpu_slot(id, slot);
            store.set_block_tier(id, kvmem::KvTier::GPU, -1, store.blocks()[id].nvme_slot);
        }
    }

    for (uint32_t id : incoming) {
        if (id < store.block_count() && store.blocks()[id].gpu_slot < 0) {
            LLAMA_LOG_ERROR("%s: incoming block %u was not placed on GPU\n", __func__, id);
            cancel_append();
            return false;
        }
    }
    return true;
}

bool llama_memory_kvmem::prepare_ubatches(
        const std::vector<llama_ubatch> & ubatches,
        uint32_t n_new_tokens,
        llama_kv_cache::slot_info_vec_t & sinfos) try {
    const uint32_t old_rows = store_n_tokens();
    if (historical_capture_) {
        uint64_t row = old_rows;
        for (const auto & ub : ubatches) {
            if (!ub.token || ub.n_pos != 1 || ub.n_seqs_unq != 1 || !ub.seq_id_unq || ub.seq_id_unq[0] != 0)
                throw std::runtime_error("historical capture requires single-sequence text appends");
            for (uint32_t i = 0; i < ub.n_tokens; ++i, ++row) {
                const auto logical = ub.logical_pos ? ub.logical_pos[i] : ub.pos[i];
                if (logical < 0 || uint64_t(logical) != row || ub.pos[i] != logical)
                    throw std::runtime_error("historical capture requires consecutive new positions");
            }
        }
        if (row - old_rows != n_new_tokens)
            throw std::runtime_error("historical capture append token count mismatch");
    }
    for (const auto & ub : ubatches) {
        for (uint32_t i = 0; i < ub.n_tokens; ++i) {
            const llama_pos row = ub.logical_pos ? ub.logical_pos[i] : ub.pos[i];
            if (row >= 0 && (uint32_t) row < old_rows) { ++attention_epoch_; break; }
        }
    }
    if (!prepare_working_set(n_new_tokens)) {
        return false;
    }
    for (const auto & ub : ubatches) {
        for (uint32_t i = 0; i < ub.n_tokens; ++i) {
            const llama_pos logical = ub.logical_pos ? ub.logical_pos[i] : ub.pos[i];
            if (logical < 0 || (uint32_t) logical >= runtime_->store().total_tokens()) return false;
            if ((size_t) logical >= row_positions_.size()) row_positions_.resize(logical + 1);
            auto & row = row_positions_[logical];
            row.spatial = ub.is_pos_2d();
            for (uint32_t j = 0; j < 4; ++j) row.pos[j] = ub.pos[(j < ub.n_pos ? j*ub.n_tokens : 0) + i];
            row.token = ub.token ? ub.token[i] : LLAMA_TOKEN_NULL;
        }
    }
    if (historical_capture_) {
        if (ubatches.size() > sinfos.size())
            throw std::length_error("historical slot list exceeded request reservation");
        for (size_t i = 0; i < ubatches.size(); ++i) {
            auto & slot = sinfos[i];
            if (slot.strm.size() != 1 || slot.idxs.size() != 1 ||
                slot.idxs[0].capacity() < ubatches[i].n_tokens)
                throw std::length_error("historical slot indexes exceeded request reservation");
            if (!kvmem_fill_slot_info(runtime_->store(), block_tokens_, kv_size_, ubatches[i], slot))
                return false;
        }
    } else {
        sinfos.clear();
        sinfos.reserve(ubatches.size());
        for (const auto & ubatch : ubatches) {
            llama_kv_cache::slot_info sinfo;
            if (!kvmem_fill_slot_info(runtime_->store(), block_tokens_, kv_size_, ubatch, sinfo)) {
                return false;
            }
            sinfos.push_back(std::move(sinfo));
        }
    }
    pos_queue_.clear();
    for (const auto & ubatch : ubatches) {
        pos_queue_.push_back(capture_batch(
                ubatch.logical_pos ? ubatch.logical_pos : ubatch.pos, ubatch.n_tokens));
    }
    if (historical_capture_) historical_capture_->producers_captured();
    return true;
} catch (const std::exception & e) {
    if (!historical_capture_) throw;
    historical_capture_->fail();
    LLAMA_LOG_ERROR("historical capture prepare: %s\n", e.what());
    return false;
}

bool llama_memory_kvmem::begin_active_ubatch(
        uint32_t n_tokens, const llama_pos * logical_positions) {
    if (capture_failed_) return false;
    harvest_check_error();
    if (!historical_capture_) return true;
    const auto reject = [&](const char * reason, uint32_t id = UINT32_MAX) {
        LLAMA_LOG_ERROR("historical active admission: %s block=%u pos=%d tokens=%u\n",
                reason, id, logical_positions ? logical_positions[0] : -1, n_tokens);
        return false;
    };
    if (!historical_active_ || historical_active_->engaged() || !n_tokens ||
        !logical_positions || logical_positions[0] < 0 ||
        uint64_t(logical_positions[0]) + n_tokens > uint64_t(INT32_MAX))
        return reject("invalid or overlapping active");
    const uint64_t current_begin = uint64_t(logical_positions[0]);
    const uint64_t current_end = current_begin + n_tokens;
    for (uint32_t i = 1; i < n_tokens; ++i)
        if (logical_positions[i] < 0 ||
            uint64_t(logical_positions[i]) != current_begin + i)
            return reject("noncontiguous logical positions");

    auto repository = historical_capture_->repository();
    const auto & directory = runtime_->historical_shadow_directory();
    auto & store = runtime_->store();
    auto & workspace = historical_active_workspace_;
    auto & requested = workspace.requested;
    auto & requested_ids = workspace.requested_ids;
    auto & remove_ids = workspace.remove_ids;
    if (workspace.seen.size() != n_slots_ || requested.capacity() < n_slots_ ||
        requested_ids.capacity() < n_slots_ || remove_ids.capacity() < n_slots_)
        return reject("missing active workspace");
    requested.clear(); requested_ids.clear(); remove_ids.clear();
    std::fill(workspace.seen.begin(), workspace.seen.end(), 0);
    for (const auto & block : store.blocks()) {
        if (block.gpu_slot < 0) continue;
        if (uint32_t(block.gpu_slot) >= n_slots_) return reject("invalid resident slot", block.block_id);
        if (!gpu_kv_already_resident(block.block_id)) continue;
        if (workspace.seen[block.gpu_slot]++) return reject("duplicate resident slot", block.block_id);
        // The request's producer already owns the current write range. These
        // newly applied cells are not an old historical view being reselected.
        if (uint64_t(block.orig_pos_end()) > current_begin &&
            uint64_t(block.orig_pos_start) < current_end) continue;
        if (uint64_t(block.orig_pos_end()) > current_begin)
            return reject("future resident span is not GPU-complete history", block.block_id);
        const auto observation = directory.observe(block.block_id);
        if (observation.state == kvmem::HistoricalKvShadowDirectory::State::Unmanaged)
            return reject("unmanaged resident history", block.block_id);
        if (observation.state == kvmem::HistoricalKvShadowDirectory::State::Bound) {
            const auto meta = repository->inspect(observation.handle);
            if (meta && meta->life == kvmem::HistoricalKvLife::Live) {
                requested.push_back({observation.handle, false});
                requested_ids.push_back(block.block_id);
                continue;
            }
            if (meta && meta->life == kvmem::HistoricalKvLife::Writing) {
                // This older span's GPU graph finished in the prior harvest,
                // but its packed Host copy may still be asynchronous.
                requested.push_back({observation.handle, false, true});
                requested_ids.push_back(block.block_id);
                continue;
            }
        }
        remove_ids.push_back(block.block_id);
    }
    auto admitted = historical_active_->try_acquire(requested);
    if (!admitted) return reject("active set acquire failed");
    struct ReleaseOnFailure {
        kvmem::HistoricalKvActiveSet & active;
        bool keep = false;
        ~ReleaseOnFailure() { if (!keep) active.release(); }
    } release_on_failure{*historical_active_};
    const auto & handles = historical_active_->handles();
    for (size_t i = 0; i < requested.size(); ++i) {
        const auto target = requested[i].handle;
        const bool permitted = std::any_of(handles.begin(), handles.end(),
            [&](kvmem::HistoricalKvHandle handle) {
                return handle.incarnation == target.incarnation && handle.slot == target.slot &&
                       handle.generation == target.generation;
            });
        if (!permitted) remove_ids.push_back(requested_ids[i]);
    }
    for (uint32_t id : remove_ids) {
        const auto block = store.blocks()[id];
        if (!store.forget_retired_gpu_residency(id))
            return reject("retired GPU residency changed", id);
        kv_->seq_rm_logical(0, static_cast<llama_pos>(block.orig_pos_start),
                            static_cast<llama_pos>(block.orig_pos_end()));
        // Removing cells changes only the next attention view. A queued D2H
        // may still read this physical slot, so retain its allocation without
        // waiting here; ordinary transfer completion returns it exactly once.
        if (historical_gpu_readers_[block.gpu_slot]) {
            historical_gpu_retired_[block.gpu_slot] = 1;
            ++historical_gpu_deferred_;
            if (trace_) kvmem_diag("KVMEM_TRACE historical_gpu_defer slot=%d readers=%u\n",
                block.gpu_slot, historical_gpu_readers_[block.gpu_slot]);
        } else {
            free_slot(block.gpu_slot);
        }
        ++attention_epoch_;
        ++historical_active_skips_;
        if (trace_) kvmem_diag("KVMEM_TRACE historical_active_skip block=%u slot=%d\n",
                               id, block.gpu_slot);
    }
    release_on_failure.keep = true;
    return true;
}

void llama_memory_kvmem::mark_active_submission() {
    if (historical_active_ && historical_active_->engaged() &&
        !historical_active_->mark_execution_submitted(historical_capture_->next_use_sequence()))
        throw std::runtime_error("historical active submission could not record LRU use");
    if (historical_submission_test_hook_) historical_submission_test_hook_();
}

void llama_memory_kvmem::complete_active_ubatch() noexcept {
    if (historical_active_) historical_active_->release();
}

llama_memory_context_ptr llama_memory_kvmem::init_batch(
        llama_batch_allocr & balloc,
        uint32_t n_ubatch,
        bool embd_all) {
    GGML_UNUSED(embd_all);

    if (historical_capture_) {
        auto & slots = historical_slot_workspace_;
        auto & micro = historical_ubatch_workspace_;
        if (!historical_batch_context_storage_ || historical_batch_context_in_use_) {
            historical_capture_->fail();
            LLAMA_LOG_ERROR("historical capture: prepared decode context is unavailable\n");
            return {};
        }
        if (slots.slots.empty() || micro.ubatches.capacity() < micro.max_ubatches ||
            n_ubatch != slots.n_ubatch || n_ubatch != micro.n_ubatch ||
            balloc.get_n_tokens() > slots.max_tokens || balloc.get_n_tokens() > micro.max_tokens) {
            historical_capture_->fail();
            LLAMA_LOG_ERROR("historical capture: decode exceeds prepared microbatch bound\n");
            return {};
        }
        auto ubatches = std::move(micro.ubatches);
        ubatches.clear(); // release the previous context's shared data owners
        llama_kv_cache::slot_info_vec_t sinfos;
        try {
            if (!balloc.use_prepared_simple_pool(&micro.data, &micro.indices))
                throw std::runtime_error("historical capture requires one text sequence and one position axis");
            struct ClearPreparedPool {
                llama_batch_allocr & balloc;
                ~ClearPreparedPool() { balloc.clear_prepared_simple_pool(); }
            } clear_pool{balloc};
            balloc.split_reset();
            while (true) {
                auto ubatch = balloc.split_simple(n_ubatch);
                if (!ubatch.n_tokens) break;
                if (ubatches.size() >= micro.max_ubatches)
                    throw std::length_error("historical microbatch count exceeded reservation");
                ubatches.push_back(std::move(ubatch));
            }
            if (ubatches.empty() || balloc.get_n_used() != balloc.get_n_tokens() ||
                ubatches.size() > slots.slots.size())
                throw std::length_error("historical microbatch split exceeded reservation");
            sinfos = std::move(slots.slots);
            if (!prepare_ubatches(ubatches, balloc.get_n_tokens(), sinfos))
                throw std::runtime_error("historical microbatch preparation failed");
            auto * context = ::new (historical_batch_context_storage_)
                HistoricalBatchContext(*this, kv_, std::move(sinfos), std::move(ubatches));
            historical_batch_context_in_use_ = true;
            return llama_memory_context_ptr(context);
        } catch (const std::exception & e) {
            historical_capture_->fail();
            if (slots.slots.empty()) slots.slots = std::move(sinfos);
            if (micro.ubatches.empty()) micro.ubatches = std::move(ubatches);
            LLAMA_LOG_ERROR("historical capture microbatch: %s\n", e.what());
            return {};
        }
    }

    do {
        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            auto ubatch = balloc.split_simple(n_ubatch);
            if (ubatch.n_tokens == 0) {
                break;
            }
            ubatches.push_back(std::move(ubatch));
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            break;
        }

        llama_kv_cache::slot_info_vec_t sinfos;
        if (!prepare_ubatches(ubatches, balloc.get_n_tokens(), sinfos)) {
            break;
        }

        return std::make_unique<llama_kv_cache_context>(
                kv_, std::move(sinfos), std::move(ubatches));
    } while (false);

    return std::make_unique<llama_kv_cache_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_kvmem::init_full() {
    return kv_->init_full();
}

llama_memory_context_ptr llama_memory_kvmem::init_update(llama_context * lctx, bool optimize) {
    return kv_->init_update(lctx, optimize);
}

void llama_memory_kvmem::clear(bool data) {
    if (historical_capture_) throw std::runtime_error("historical capture baseline requires context teardown for reset");
    harvest_flush();
    harvest_gpu_v_commit();
    if (kv_) {
        kv_->clear(data);
    }
    reset_policy();
}

bool llama_memory_kvmem::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    if (historical_capture_) return false;
    ++attention_epoch_;
    const bool ok = kv_->seq_rm(seq_id, p0, p1);
    if (!ok) {
        return false;
    }
    // Query replay seq_rm's the query span, which can be the whole prompt on
    // short inputs. That must not wipe KvMemStore metadata / GPU slots.
    if (!replay_ &&
        seq_id <= 0 && p0 <= 0 &&
        (p1 < 0 || p1 >= static_cast<llama_pos>(runtime_->store().total_tokens()))) {
        // Known divergence, deliberately not repaired here so the default
        // path stays byte-identical: the block table is zeroed while the host
        // mirror keeps its blocks, and truncate_cached's guard
        // (n_past >= total_tokens) can no longer reach them. The re-prefilled
        // block 0 then keeps the previous content's packed K, because both
        // harvest_gpu_v and harvest_full_blocks_async skip a layer whose
        // packed K is already present. Record it so a store swap declines to
        // carry this store rather than restaging those bytes into another
        // conversation's cells; reset_policy() clears both and the flag.
        host_mirror_stale_ = true;
        runtime_->truncate_to(0);
        reset_slots();
        retrieval_pinned_ = false;
    }
    return true;
}

void llama_memory_kvmem::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    if (historical_capture_) throw std::runtime_error("historical capture baseline does not support sequence copies");
    ++attention_epoch_;
    kv_->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_memory_kvmem::seq_keep(llama_seq_id seq_id) {
    if (historical_capture_) throw std::runtime_error("historical capture baseline does not support sequence mutation");
    ++attention_epoch_;
    kv_->seq_keep(seq_id);
}

void llama_memory_kvmem::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    if (historical_capture_) throw std::runtime_error("historical capture baseline does not support position shifts");
    ++attention_epoch_;
    kv_->seq_add(seq_id, p0, p1, shift);
}

void llama_memory_kvmem::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    if (historical_capture_) throw std::runtime_error("historical capture baseline does not support position shifts");
    ++attention_epoch_;
    kv_->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_memory_kvmem::seq_pos_min(llama_seq_id seq_id) const {
    return kv_->seq_pos_min(seq_id);
}

llama_pos llama_memory_kvmem::seq_pos_max(llama_seq_id seq_id) const {
    return kv_->seq_pos_max(seq_id);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_kvmem::memory_breakdown() const {
    return kv_->memory_breakdown();
}

void llama_memory_kvmem::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    if (historical_capture_) throw std::runtime_error("historical capture baseline does not support snapshots");
    kv_->state_write(io, seq_id, flags);
}

void llama_memory_kvmem::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    if (historical_capture_) throw std::runtime_error("historical capture baseline does not support restore");
    ++attention_epoch_;
    kv_->state_read(io, seq_id, flags);
}

void llama_memory_kvmem::note_ubatch_pos(const std::vector<llama_pos> & pos) {
    pos_queue_.push_back(capture_batch(pos.data(), pos.size()));
}

llama_memory_kvmem::CaptureBatch llama_memory_kvmem::capture_batch(
        const llama_pos * positions, size_t count) {
    if (capture_arena_ && count > capture_max_tokens_ - capture_tokens_used_)
        throw std::length_error("historical capture exceeded prepared position capacity");
    CaptureBatch batch(capture_arena_);
    if (!count) return batch;
    batch.positions.assign(positions, positions + count);
    batch.ranges.reserve(count);
    const auto & directory = runtime_->historical_shadow_directory();
    for (uint32_t i = 0; i < count;) {
        if (positions[i] < 0) throw std::invalid_argument("negative historical capture position");
        MeanWriteRange range;
        range.source_token = i;
        range.original_begin = static_cast<uint32_t>(positions[i]);
        range.token_count = 1;
        const auto & store = runtime_->store();
        const auto found = store.block_id_containing(range.original_begin);
        const uint32_t bid = found >= 0 ? static_cast<uint32_t>(found)
                                       : range.original_begin / block_tokens_;
        if (found < 0 && directory.size())
            throw std::logic_error("historical capture has no admitted original span");
        const auto observation = directory.observe(bid);
        range.block_id = bid;
        range.original_end = observation.state == kvmem::HistoricalKvShadowDirectory::State::Unmanaged
            ? (uint64_t(bid) + 1) * block_tokens_
            : observation.original_begin + observation.valid_tokens;
        if (observation.state == kvmem::HistoricalKvShadowDirectory::State::Unmanaged && found >= 0 &&
                store.blocks()[bid].orig_pos_start != uint64_t(bid) * block_tokens_)
            throw std::logic_error("non-aligned capture requires a historical payload identity");
        while (i + range.token_count < count) {
            const auto next = positions[i + range.token_count];
            if (next < 0 || static_cast<uint64_t>(next) !=
                    uint64_t(range.original_begin) + range.token_count ||
                    static_cast<uint64_t>(next) >= range.original_end) break;
            ++range.token_count;
        }
        if (observation.state == kvmem::HistoricalKvShadowDirectory::State::Unmanaged) {
            range.route = MeanWriteRange::Route::Legacy;
        } else if (const auto binding = directory.find(bid)) {
            const auto & context = binding->writer_context;
            const auto meta = context ? context->writer().metadata() : std::nullopt;
            if (context && context->storage() &&
                    meta && meta->life == kvmem::HistoricalKvLife::Writing &&
                    !context->repository().closed()) {
                if (range.original_begin < binding->original_begin ||
                        uint64_t(range.original_begin) + range.token_count >
                            binding->original_begin + binding->valid_tokens) {
                    throw std::logic_error("historical capture exceeds its original span");
                }
                range.route = MeanWriteRange::Route::Managed;
                range.token_offset = static_cast<uint32_t>(range.original_begin - binding->original_begin);
                range.gpu_slot = store.blocks()[bid].gpu_slot;
                range.writer = context;
            }
        }
        batch.ranges.push_back(std::move(range));
        i += batch.ranges.back().token_count;
    }
    if (capture_arena_) capture_tokens_used_ += count;
    return batch;
}

void llama_memory_kvmem::register_capture(ggml_tensor * t, int il, char which) {
    if (!t) {
        return;
    }
    pending_capture_.push_back({t, il, which});
    if (which == 'q') {
        graph_has_q_ = true;
    }
    if (which == 'k') {
        graph_has_k_ = true;
    }
}

void llama_memory_kvmem::capture_on_new_graph() {
    pending_capture_.clear();
    graph_has_q_ = false;
    graph_has_k_ = false;
    graph_has_record_ = recr_ && recr_->replay_recording;
}

bool llama_memory_kvmem::capture_can_reuse(uint32_t n_tokens, uint32_t n_pos,
                                           const llama_pos * pos) const {
    if (graph_has_record_ != bool(recr_ && recr_->replay_recording)) return false;
    // After pin, n=1 decode must not reuse a graph built without K capture
    // (last prefill ubatch of 1 token, or query replay with capture off).
    if (want_decode_mean() && !graph_has_k_) {
        return false;
    }
    // Post-pin query replay must not keep rebuilding just because the ubatch
    // overlaps the query span — Q nodes are off after pin. T5 recapture
    // (want_q_capture, including replay) still requires a Q graph.
    const bool need_q = want_q_capture() &&
            llama_kvmem_ubatch_needs_q_capture(n_tokens, n_pos, pos);
    return need_q == graph_has_q_;
}

uint64_t llama_memory_kvmem::capture_stamp() const {
    uint64_t stamp = 1;
    if (retrieval_pinned_) {
        stamp |= 2;
    }
    if (want_decode_mean()) {
        stamp |= 4;
    }
    if (want_q_capture()) {
        stamp |= 8;
    }
    if (replay_) {
        stamp |= 16;
    }
    if (graph_has_k_) {
        stamp |= 32;
    }
    if (graph_has_q_) {
        stamp |= 64;
    }
    if (graph_has_record_) {
        stamp |= 128;
    }
    return stamp;
}

bool llama_memory_kvmem::d2h_init() {
    harvest_check_error();
    if (d2h_ && d2h_->ok) {
        return true;
    }
    if (!d2h_) {
        d2h_ = std::make_unique<CaptureD2hPipe>();
    }
    if (d2h_->stream == nullptr) {
        if (!kvmem_cuda_ok(cudaStreamCreateWithFlags(&d2h_->stream, cudaStreamNonBlocking),
                           "stream")) {
            return false;
        }
    }
    if (d2h_->snap == nullptr) {
        if (!kvmem_cuda_ok(cudaEventCreateWithFlags(&d2h_->snap, cudaEventDisableTiming),
                           "snap")) {
            return false;
        }
    }
    for (int i = 0; i < 2; ++i) {
        if (d2h_->slots[i].done == nullptr) {
            if (!kvmem_cuda_ok(cudaEventCreateWithFlags(&d2h_->slots[i].done, cudaEventDisableTiming),
                               "event")) {
                return false;
            }
        }
    }
    if (!kvmem_cuda_ok(cudaGetDevice(&d2h_->device), "capture current device")) return false;
    d2h_->ok = true;
    harvest_worker_start();
    return true;
}

void llama_memory_kvmem::d2h_free() {
    if (!d2h_) {
        return;
    }
    if (harvest_worker_on()) throw std::logic_error("capture worker must stop before buffer destruction");
    if (d2h_->stream && (capture_fault(CaptureFaultPoint::Drain) ||
            !kvmem_cuda_ok(cudaStreamSynchronize(d2h_->stream), "capture free drain")))
        throw std::runtime_error("capture buffer destruction requires confirmed device completion");
    for (int i = 0; i < 2; ++i) {
        auto & s = d2h_->slots[i];
        if (s.gpu) {
            cudaFree(s.gpu);
            s.gpu = nullptr;
        }
        if (s.pin) {
            cudaFreeHost(s.pin);
            s.pin = nullptr;
        }
        if (s.done) {
            cudaEventDestroy(s.done);
            s.done = nullptr;
        }
        s.cap = 0;
        s.inflight = false;
        s.items.clear();
        s.capture = {};
    }
    if (d2h_->snap) {
        cudaEventDestroy(d2h_->snap);
        d2h_->snap = nullptr;
    }
    if (d2h_->compute_done) {
        ggml_backend_event_free(d2h_->compute_done);
        d2h_->compute_done = nullptr;
    }
    if (d2h_->snap_be) {
        ggml_backend_event_free(d2h_->snap_be);
        d2h_->snap_be = nullptr;
    }
    if (d2h_->stream) {
        cudaStreamDestroy(d2h_->stream);
        d2h_->stream = nullptr;
    }
    d2h_->ok = false;
}

void llama_memory_kvmem::multi_d2h_free() {
    if (!multi_d2h_) return;
    int original = -1;
    (void) cudaGetDevice(&original);
    for (auto & dev : multi_d2h_->devices) {
        if (dev.id >= 0) (void) cudaSetDevice(dev.id);
        if (dev.stream) {
            (void) cudaStreamSynchronize(dev.stream);
            (void) cudaStreamDestroy(dev.stream);
            dev.stream = nullptr;
        }
        if (dev.pin) {
            (void) cudaFreeHost(dev.pin);
            dev.pin = nullptr;
        }
    }
    if (original >= 0) (void) cudaSetDevice(original);
    multi_d2h_.reset();
}

bool llama_memory_kvmem::multi_d2h_submit() {
    if (pending_capture_.empty() || pos_queue_.empty()) return false;
    if (!multi_d2h_) multi_d2h_ = std::make_unique<MultiD2hPipe>();
    auto & pipe = *multi_d2h_;
    pipe.items.clear();
    for (auto & dev : pipe.devices) dev.used = 0;

    int original = -1;
    if (cudaGetDevice(&original) != cudaSuccess) return false;
    struct DeviceRestore {
        int id;
        ~DeviceRestore() { (void) cudaSetDevice(id); }
    } restore{original};

    // A layer split has ordinary CUDA tensors owned by one physical GPU.
    for (const CaptureNode & n : pending_capture_) {
        if (!n.t || n.which == 'v') continue;
        const uint8_t * src = kvmem_cuda_tensor_ptr(n.t);
        if (!src) return false;
        cudaPointerAttributes attr{};
        if (cudaPointerGetAttributes(&attr, src) != cudaSuccess || attr.type != cudaMemoryTypeDevice) {
            (void) cudaGetLastError();
            return false;
        }
        size_t index = 0;
        for (; index < pipe.devices.size(); ++index) {
            if (pipe.devices[index].id == attr.device) break;
        }
        if (index == pipe.devices.size()) {
            pipe.devices.emplace_back();
            pipe.devices.back().id = attr.device;
        }
        auto & dev = pipe.devices[index];
        if (!dev.stream && (!kvmem_cuda_ok(cudaSetDevice(dev.id), "multi set device") ||
                !kvmem_cuda_ok(cudaStreamCreateWithFlags(&dev.stream, cudaStreamNonBlocking), "multi stream"))) {
            return false;
        }
        MultiD2hPipe::Item item;
        item.device = index;
        item.src = src;
        item.capture.il = n.il;
        item.capture.which = n.which;
        item.capture.offset = dev.used;
        item.capture.nbytes = ggml_nbytes(n.t);
        item.capture.d = n.t->ne[0];
        item.capture.h = n.t->ne[1];
        item.capture.n = n.t->ne[2];
        item.capture.nb0 = n.t->nb[0];
        item.capture.nb1 = n.t->nb[1];
        item.capture.nb2 = n.t->nb[2];
        item.capture.type = n.t->type;
        dev.used += item.capture.nbytes;
        pipe.items.push_back(item);
    }
    for (auto & dev : pipe.devices) {
        if (dev.used <= dev.cap) continue;
        if (dev.pin) {
            (void) cudaFreeHost(dev.pin);
            dev.pin = nullptr;
            dev.cap = 0;
        }
        if (!kvmem_cuda_ok(cudaSetDevice(dev.id), "multi set device") ||
            !kvmem_cuda_ok(cudaMallocHost(reinterpret_cast<void **>(&dev.pin), dev.used), "multi pinned host")) {
            (void) cudaGetLastError();
            return false;
        }
        dev.cap = dev.used;
    }

    const int64_t t_read = ggml_time_us();
    bool submitted = true;
    for (size_t index = 0; index < pipe.devices.size(); ++index) {
        auto & dev = pipe.devices[index];
        if (!dev.used) continue;
        if (!kvmem_cuda_ok(cudaSetDevice(dev.id), "multi set device")) {
            submitted = false;
            break;
        }
        for (const auto & item : pipe.items) {
            if (item.device != index) continue;
            if (!kvmem_cuda_ok(kvmem_copy_async(dev.pin + item.capture.offset, item.src,
                    item.capture.nbytes, cudaMemcpyDeviceToHost, dev.stream), "multi D2H")) {
                submitted = false;
                break;
            }
        }
        if (!submitted) break;
    }
    for (auto & dev : pipe.devices) {
        if (!dev.stream || !dev.used) continue;
        (void) cudaSetDevice(dev.id);
        if (!kvmem_cuda_ok(cudaStreamSynchronize(dev.stream), "multi D2H wait")) submitted = false;
    }
    if (!submitted) {
        (void) cudaGetLastError();
        return false;
    }
    const int64_t read_us = ggml_time_us() - t_read;
    const auto batch = std::move(pos_queue_.front());
    pos_queue_.erase(pos_queue_.begin());
    const int64_t t_host = ggml_time_us();
    for (const auto & item : pipe.items) {
        const auto & c = item.capture;
        harvest_from_host(batch, c.il, c.which, pipe.devices[item.device].pin + c.offset,
                          c.type, c.d, c.h, c.n, c.nb0, c.nb1, c.nb2);
    }
    const int64_t host_us = ggml_time_us() - t_host;
    if (perf_.enabled) {
        perf_.multi_read_us += read_us;
        perf_.multi_host_us += host_us;
        size_t bytes = 0;
        for (const auto & dev : pipe.devices) bytes += dev.used;
        fprintf(stderr, "KVMEM_MULTI_D2H mode=batch devices=%zu copies=%zu bytes=%zu read_ms=%.3f host_ms=%.3f\n",
                pipe.devices.size(), pipe.items.size(), bytes, read_us / 1000.0, host_us / 1000.0);
    }
    return true;
}

bool llama_memory_kvmem::harvest_worker_on() const {
    return harvest_w_ && harvest_w_->th.joinable();
}

bool llama_memory_kvmem::capture_fault(CaptureFaultPoint point, int slot, size_t index) const noexcept {
    return capture_fault_ && capture_fault_(capture_fault_context_, point, slot, index);
}

void llama_memory_kvmem::capture_fail() noexcept {
    capture_failed_ = true;
    if (historical_capture_) historical_capture_->fail();
}

void llama_memory_kvmem::harvest_check_error() {
    std::exception_ptr error;
    if (harvest_w_) {
        std::lock_guard<std::mutex> guard(harvest_w_->mu);
        error = harvest_w_->error;
    }
    if (error) {
        capture_fail();
        std::rethrow_exception(error);
    }
    if (capture_failed_) throw std::runtime_error("KVMem capture domain is closed after failure");
}

void llama_memory_kvmem::harvest_merge_stats() {
    if (!harvest_w_) return;
    HarvestWorker::CommitStats total, last;
    {
        std::lock_guard<std::mutex> guard(harvest_w_->mu);
        if (!harvest_w_->have_stats) return;
        total = harvest_w_->total;
        last = harvest_w_->last;
        harvest_w_->total = {};
        harvest_w_->have_stats = false;
    }
    perf_.last_d2h_wait_us = last.wait;
    perf_.last_pack_us = last.pack;
    perf_.last_nvme_us = last.nvme;
    perf_.last_commit_us = last.commit;
    perf_.last_nvme_bytes = last.bytes;
    perf_.last_nvme_syscalls = last.syscalls;
    perf_.d2h_wait_us += total.wait;
    perf_.pack_us += total.pack;
    perf_.nvme_us += total.nvme;
    perf_.commit_us += total.commit;
    perf_.nvme_bytes += total.bytes;
    perf_.nvme_syscalls += total.syscalls;
}

void llama_memory_kvmem::harvest_shutdown() {
    // Context disposal cancels uncommitted CPU work. The worker must have
    // returned before touching its slots; device work must also truly finish.
    harvest_worker_stop(true);
    // A failure while establishing the compute -> capture dependency can
    // leave source computation outside the capture stream. A failed request
    // must also confirm that device work before destroying source ownership.
    if (capture_failed_ && (capture_fault(CaptureFaultPoint::Drain) ||
            !kvmem_cuda_ok(cudaDeviceSynchronize(), "failed capture source completion")))
        throw std::runtime_error("capture shutdown cannot confirm source computation completion");
    if (d2h_ && d2h_->stream && (capture_fault(CaptureFaultPoint::Drain) ||
            !kvmem_cuda_ok(cudaStreamSynchronize(d2h_->stream), "capture shutdown drain")))
        throw std::runtime_error("capture shutdown cannot confirm device completion");
    if (d2h_) for (auto & slot : d2h_->slots) {
        slot.device_pending = false;
        slot.items.clear();
        slot.capture = {};
        slot.inflight.store(false, std::memory_order_release);
    }
    if (harvest_w_) {
        std::lock_guard<std::mutex> guard(harvest_w_->mu);
        harvest_w_->first = harvest_w_->queued = 0;
        // Preserve error/closed admission; shutdown never resurrects work.
    }
    harvest_merge_stats();
    if (raw_) raw_->wait_writes();
    if (mtp_) mtp_->harvest_flush();
}

void llama_memory_kvmem::harvest_worker_start() {
    harvest_check_error();
    if (kvmem_harvest_sync_old() || harvest_worker_on()) {
        return;
    }
    if (!harvest_w_) {
        harvest_w_ = std::make_unique<HarvestWorker>();
    }
    harvest_w_->stop = false;
    harvest_w_->cancel_queued = false;
    harvest_w_->ready = false;
    harvest_w_->th = std::thread([this] { harvest_loop(); });
}

void llama_memory_kvmem::harvest_worker_stop(bool cancel_queued) {
    if (!harvest_w_) {
        return;
    }
    {
        std::lock_guard<std::mutex> lk(harvest_w_->mu);
        harvest_w_->stop = true;
        harvest_w_->cancel_queued = cancel_queued;
    }
    harvest_w_->cv.notify_all();
    if (harvest_w_->th.joinable()) {
        harvest_w_->th.join();
    }
}

void llama_memory_kvmem::harvest_wait_slot(int slot) try {
    if (!harvest_w_ || !d2h_ || slot < 0 || slot > 1) {
        return;
    }
    std::unique_lock<std::mutex> lk(harvest_w_->mu);
    harvest_w_->cv.wait(lk, [&] {
        return harvest_w_->error || !d2h_->slots[slot].inflight.load(std::memory_order_acquire);
    });
    const auto error = harvest_w_->error;
    lk.unlock();
    if (error) std::rethrow_exception(error);
    harvest_merge_stats();
} catch (...) {
    capture_fail();
    throw;
}

void llama_memory_kvmem::harvest_loop() try {
    if (d2h_) {
        if (capture_fault(CaptureFaultPoint::DeviceSelect) ||
                !kvmem_cuda_ok(cudaSetDevice(d2h_->device), "harvest worker set device"))
            throw std::runtime_error("capture worker could not select its device");
    }
    {
        std::lock_guard<std::mutex> guard(harvest_w_->mu);
        harvest_w_->ready = true;
    }
    harvest_w_->cv.notify_all();
    while (true) {
        int slot = -1;
        {
            std::unique_lock<std::mutex> lk(harvest_w_->mu);
            harvest_w_->cv.wait(lk, [&] {
                return harvest_w_->stop || harvest_w_->queued;
            });
            if (harvest_w_->stop && (harvest_w_->cancel_queued || !harvest_w_->queued)) {
                break;
            }
            slot = harvest_w_->q[harvest_w_->first];
            harvest_w_->first = (harvest_w_->first + 1) % harvest_w_->q.size();
            --harvest_w_->queued;
        }
        d2h_commit(slot);
    }
} catch (...) {
    // Do not clear inflight, payload or task ownership on error. The GPU
    // owner closes admission and explicitly settles/cancels during teardown.
    {
        std::lock_guard<std::mutex> guard(harvest_w_->mu);
        harvest_w_->error = std::current_exception();
    }
    harvest_w_->cv.notify_all();
}

void llama_memory_kvmem::d2h_commit(int slot) {
    if (!d2h_ || slot < 0 || slot > 1) {
        return;
    }
    auto & s = d2h_->slots[slot];
    if (!s.inflight) {
        return;
    }
    const int64_t t0 = ggml_time_us();
    int64_t wait_us = 0;
    if (s.done) {
        const int64_t tw = ggml_time_us();
        if (capture_fault(CaptureFaultPoint::EventWait, slot) ||
                !kvmem_cuda_ok(cudaEventSynchronize(s.done), "capture commit wait")) {
            throw std::runtime_error("historical capture D2H did not complete");
        }
        wait_us = ggml_time_us() - tw;
    }
    s.device_pending = false;
    const uint64_t nv0 = raw_ ? raw_->nvme_wait_ns() : 0;
    const uint64_t sc0 = raw_ ? raw_->nvme_syscalls() : 0;
    const uint64_t b0 = raw_ ? raw_->nvme_bytes_written() : 0;
    const int64_t tp = ggml_time_us();
    for (auto & it : s.items) {
        if (it.which == 'q' && !it.committed) {
            if (capture_fault(CaptureFaultPoint::HostCommit, slot, size_t(&it - s.items.data())))
                throw std::runtime_error("injected capture host commit failure");
            harvest_from_host(s.capture, it.il, it.which, s.pin + it.offset, it.type,
                              it.d, it.h, it.n, it.nb0, it.nb1, it.nb2);
            it.committed = true;
        }
    }
    for (auto & it : s.items) {
        if (it.which != 'q' && !it.committed) {
            if (capture_fault(CaptureFaultPoint::HostCommit, slot, size_t(&it - s.items.data())))
                throw std::runtime_error("injected capture host commit failure");
            harvest_from_host(s.capture, it.il, it.which, s.pin + it.offset, it.type,
                              it.d, it.h, it.n, it.nb0, it.nb1, it.nb2);
            it.committed = true;
        }
    }
    const int64_t pack_wall_us = ggml_time_us() - tp;
    const uint64_t nvme_us = raw_ ? (raw_->nvme_wait_ns() - nv0) / 1000 : 0;
    HarvestWorker::CommitStats stats;
    stats.wait = wait_us;
    stats.nvme = static_cast<int64_t>(nvme_us);
    stats.pack = pack_wall_us > static_cast<int64_t>(nvme_us)
            ? pack_wall_us - static_cast<int64_t>(nvme_us) : 0;
    stats.bytes = raw_ ? (raw_->nvme_bytes_written() - b0) : 0;
    stats.syscalls = raw_ ? (raw_->nvme_syscalls() - sc0) : 0;
    stats.commit = ggml_time_us() - t0;
    s.items.clear();
    s.capture = {};
    if (harvest_w_) {
        std::lock_guard<std::mutex> lk(harvest_w_->mu);
        // Only the foreground merges into perf_: it also resets/prints those
        // fields while the worker commits a previous microbatch.
        auto & total = harvest_w_->total;
        total.wait += stats.wait;
        total.pack += stats.pack;
        total.nvme += stats.nvme;
        total.commit += stats.commit;
        total.bytes += stats.bytes;
        total.syscalls += stats.syscalls;
        harvest_w_->last = stats;
        harvest_w_->have_stats = true;
        s.inflight.store(false, std::memory_order_release);
        harvest_w_->cv.notify_all();
    } else {
        // Legacy synchronous mode never creates a worker.
        perf_.last_d2h_wait_us = stats.wait;
        perf_.last_pack_us = stats.pack;
        perf_.last_nvme_us = stats.nvme;
        perf_.last_commit_us = stats.commit;
        perf_.last_nvme_bytes = stats.bytes;
        perf_.last_nvme_syscalls = stats.syscalls;
        perf_.d2h_wait_us += stats.wait;
        perf_.pack_us += stats.pack;
        perf_.nvme_us += stats.nvme;
        perf_.commit_us += stats.commit;
        perf_.nvme_bytes += stats.bytes;
        perf_.nvme_syscalls += stats.syscalls;
        s.inflight.store(false, std::memory_order_release);
    }
}

void llama_memory_kvmem::harvest_flush() try {
    harvest_check_error();
    if (harvest_worker_on() && d2h_) {
        std::unique_lock<std::mutex> lk(harvest_w_->mu);
        harvest_w_->cv.wait(lk, [&] {
            return harvest_w_->error || (harvest_w_->ready && harvest_w_->queued == 0 &&
                    !d2h_->slots[0].inflight &&
                    !d2h_->slots[1].inflight);
        });
        const auto error = harvest_w_->error;
        lk.unlock();
        if (error) std::rethrow_exception(error);
    } else if (d2h_) {
        d2h_commit(1 - d2h_->next);
        d2h_commit(d2h_->next);
    }
    harvest_merge_stats();
    if (raw_) {
        raw_->wait_writes();
    }
    if (mtp_) {
        mtp_->harvest_flush();
    }
    harvest_perf_print_sum();
} catch (...) {
    capture_fail();
    throw;
}

void llama_memory_kvmem::harvest_perf_emit_graph_line() {
    if (!perf_.enabled || perf_.graph_line_printed) {
        return;
    }
    perf_.graph_line_printed = true;
    // Capture Y/N needs GGML_LOG_LEVEL=DEBUG; this line is the adapter signal.
    fprintf(stderr, "KVMEM_CUDA_GRAPH compiled=1 captured=-1 resets=-1\n");
}

void llama_memory_kvmem::harvest_perf_print_sum() {
    if (!perf_.enabled || perf_.sum_printed || perf_.n_ubatch == 0) {
        return;
    }
    perf_.sum_printed = true;
    int64_t mtp_sync_us = 0;
    uint64_t mtp_nvme_bytes = 0;
    uint64_t mtp_nvme_syscalls = 0;
    uint32_t mtp_n = 0;
    if (mtp_) {
        mtp_n = mtp_->harvest_perf_n_ubatch();
        mtp_sync_us = mtp_->harvest_perf_sync_us();
        mtp_nvme_bytes = mtp_->harvest_perf_nvme_bytes();
        mtp_nvme_syscalls = mtp_->harvest_perf_nvme_syscalls();
    }
    fprintf(stderr, "KVMEM_HARVEST_SUM n_ubatch=%u n_tok=%u "
            "sync_ms=%.3f d2d_ms=%.3f d2h_wait_ms=%.3f pack_ms=%.3f nvme_ms=%.3f "
            "nvme_bytes=%llu nvme_syscalls=%llu "
            "n_pressure=%u n_pressure_out=%u "
            "mtp_n_ubatch=%u mtp_sync_ms=%.3f mtp_nvme_bytes=%llu mtp_nvme_syscalls=%llu\n",
            perf_.n_ubatch, perf_.n_tok,
            perf_.sync_us / 1000.0, perf_.d2d_us / 1000.0,
            perf_.d2h_wait_us / 1000.0, perf_.pack_us / 1000.0,
            perf_.nvme_us / 1000.0,
            (unsigned long long) perf_.nvme_bytes,
            (unsigned long long) perf_.nvme_syscalls,
            perf_.n_pressure, perf_.n_pressure_out,
            mtp_n, mtp_sync_us / 1000.0,
            (unsigned long long) mtp_nvme_bytes,
            (unsigned long long) mtp_nvme_syscalls);
}

bool llama_memory_kvmem::d2h_submit(ggml_backend_t be) try {
    if (!d2h_init() || pending_capture_.empty() || pos_queue_.empty()) {
        return false;
    }
    size_t bytes = 0;
    for (const CaptureNode & n : pending_capture_) {
        if (n.t && n.which != 'v') {
            bytes += ggml_nbytes(n.t);
        }
    }
    if (bytes == 0) {
        return false;
    }
    auto & s = d2h_->slots[d2h_->next];
    if (s.inflight) {
        if (harvest_worker_on()) {
            const int64_t tw = ggml_time_us();
            harvest_wait_slot(d2h_->next);
            perf_.last_commit_us = ggml_time_us() - tw;
            perf_.commit_us += perf_.last_commit_us;
        } else {
            d2h_commit(d2h_->next);
        }
    }
    if (s.device_pending) throw std::logic_error("capture slot reused before confirmed device completion");
    // Finish all potentially allocating slot setup before issuing any DMA.
    s.items.clear();
    s.items.reserve(pending_capture_.size());
    perf_.last_sync_us = 0;
    perf_.last_d2d_us = 0;
    perf_.last_snap_wait_us = 0;
    perf_.last_d2h_submit_us = 0;
    const int64_t t_submit0 = ggml_time_us();
    if (s.cap < bytes) {
        if (s.gpu) {
            cudaFree(s.gpu);
            s.gpu = nullptr;
        }
        if (s.pin) {
            cudaFreeHost(s.pin);
            s.pin = nullptr;
        }
        if (!kvmem_cuda_ok(cudaMalloc(reinterpret_cast<void **>(&s.gpu), bytes), "gpu staging") ||
            !kvmem_cuda_ok(cudaMallocHost(reinterpret_cast<void **>(&s.pin), bytes), "pinned host")) {
            if (s.gpu) {
                cudaFree(s.gpu);
                s.gpu = nullptr;
            }
            if (s.pin) {
                cudaFreeHost(s.pin);
                s.pin = nullptr;
            }
            s.cap = 0;
            return false;
        }
        s.cap = bytes;
        kvmem_diag("KVMEM_CAPTURE_MEMORY mode=raw slot0_bytes=%zu slot1_bytes=%zu last_bytes=%zu pinned_bytes=%zu\n",
                d2h_->slots[0].cap, d2h_->slots[1].cap, bytes, d2h_->slots[0].cap + d2h_->slots[1].cap);
    }
    if (be && !d2h_->compute_done) {
        ggml_backend_dev_t dev = ggml_backend_get_device(be);
        if (dev) {
            d2h_->compute_done = ggml_backend_event_new(dev);
            d2h_->snap_be = ggml_backend_event_new(dev);
        }
    }
    const bool old_sync = kvmem_harvest_sync_old() || !be || !d2h_->compute_done ||
            !kvmem_ggml_cuda_event(d2h_->compute_done);
    s.capture = pos_queue_.front();
    {
        const int64_t t_sync = ggml_time_us();
        if (old_sync) {
            if (be) {
                ggml_backend_synchronize(be);
            } else {
                if (!kvmem_cuda_ok(cudaDeviceSynchronize(), "capture compute completion"))
                    throw std::runtime_error("capture source computation did not complete");
            }
        } else {
            ggml_backend_event_record(d2h_->compute_done, be);
            cudaEvent_t cde = kvmem_ggml_cuda_event(d2h_->compute_done);
            if (cde) {
                if (!kvmem_cuda_ok(cudaStreamWaitEvent(d2h_->stream, cde, 0), "compute wait")) {
                    throw std::runtime_error("capture could not order source computation");
                }
            } else {
                ggml_backend_synchronize(be);
            }
        }
        perf_.last_sync_us = ggml_time_us() - t_sync;
        perf_.sync_us += perf_.last_sync_us;
    }
    // The position queue is the synchronous fallback's input. Keep its front
    // intact until every CUDA operation and worker-queue insertion succeeds.
    const auto abort_submit = [&]() -> bool {
        // A failed copy/event can leave earlier copies queued against s.gpu or
        // s.pin. Drain the stream before a retry can reuse/free either buffer.
        if (capture_fault(CaptureFaultPoint::Drain) ||
                !kvmem_cuda_ok(cudaStreamSynchronize(d2h_->stream), "failed D2H drain")) {
            throw std::runtime_error("KVMem D2H stream could not be drained after submission failure");
        }
        s.device_pending = false;
        s.items.clear();
        s.capture = {};
        return false;
    };
    size_t off = 0;
    uint32_t n_q = 0, n_k = 0, n_v = 0, n_host = 0, n_dev = 0;
    const int64_t t_d2d = ggml_time_us();
    for (const CaptureNode & n : pending_capture_) {
        if (!n.t || n.which == 'v') {
            continue;
        }
        CaptureD2hPipe::Item it;
        it.il = n.il;
        it.which = n.which;
        it.offset = off;
        it.nbytes = ggml_nbytes(n.t);
        it.d = n.t->ne[0];
        it.h = n.t->ne[1];
        it.n = n.t->ne[2];
        it.nb0 = n.t->nb[0];
        it.nb1 = n.t->nb[1];
        it.nb2 = n.t->nb[2];
        it.type = n.t->type;
        ggml_backend_buffer_t buf = n.t->view_src ? n.t->view_src->buffer : n.t->buffer;
        const uint8_t * src = static_cast<const uint8_t *>(n.t->data);
        if (buf && ggml_backend_buffer_is_host(buf)) {
            memcpy(s.pin + off, src, it.nbytes);
            n_host++;
        } else {
            s.device_pending = true;
            if (capture_fault(CaptureFaultPoint::SnapshotCopy, d2h_->next, s.items.size()) ||
                    !kvmem_cuda_ok(kvmem_copy_async(s.gpu + off, src, it.nbytes,
                                               cudaMemcpyDeviceToDevice, d2h_->stream),
                               "D2D")) {
                return abort_submit();
            }
            it.from_gpu = true;
            n_dev++;
        }
        if (n.which == 'q') {
            n_q++;
        } else if (n.which == 'k') {
            n_k++;
        } else if (n.which == 'v') {
            n_v++;
        }
        s.items.push_back(it);
        off += it.nbytes;
    }
    perf_.last_d2d_us = ggml_time_us() - t_d2d;
    perf_.d2d_us += perf_.last_d2d_us;
    if (n_dev > 0) {
        if (!kvmem_cuda_ok(cudaEventRecord(d2h_->snap, d2h_->stream), "snap record")) {
            return abort_submit();
        }
        const int64_t t_snap = ggml_time_us();
        cudaEvent_t snap_cuda = kvmem_ggml_cuda_event(d2h_->snap_be);
        if (!old_sync && be && snap_cuda) {
            // Record the ggml event on the harvest stream (never
            // ggml_backend_event_record(snap_be, be) — that is the compute stream).
            if (!kvmem_cuda_ok(cudaEventRecord(snap_cuda, d2h_->stream), "snap_be record")) {
                return abort_submit();
            }
            ggml_backend_event_wait(be, d2h_->snap_be);
        } else if (!kvmem_cuda_ok(cudaEventSynchronize(d2h_->snap), "snap wait")) {
            return abort_submit();
        }
        perf_.last_snap_wait_us = ggml_time_us() - t_snap;
        perf_.snap_wait_us += perf_.last_snap_wait_us;
        if (n_host == 0) {
            if (capture_fault(CaptureFaultPoint::HostCopy, d2h_->next) ||
                    !kvmem_cuda_ok(kvmem_copy_async(s.pin, s.gpu, off, cudaMemcpyDeviceToHost, d2h_->stream),
                               "D2H")) {
                return abort_submit();
            }
        } else {
            for (const auto & it : s.items) {
                if (!it.from_gpu) {
                    continue;
                }
                if (capture_fault(CaptureFaultPoint::HostCopy, d2h_->next, size_t(&it - s.items.data())) ||
                        !kvmem_cuda_ok(kvmem_copy_async(s.pin + it.offset, s.gpu + it.offset, it.nbytes,
                                                   cudaMemcpyDeviceToHost, d2h_->stream),
                                   "D2H item")) {
                    return abort_submit();
                }
            }
        }
    }
    s.device_pending = true;
    if (capture_fault(CaptureFaultPoint::CompletionEvent, d2h_->next) ||
            !kvmem_cuda_ok(cudaEventRecord(s.done, d2h_->stream), "record")) {
        return abort_submit();
    }
    const int submitted = d2h_->next;
    // Once queued, a spurious worker wake may clear s.items/s.capture before the
    // foreground reaches its trace. Capture these while this thread owns it.
    const size_t submitted_items = s.items.size();
    const size_t submitted_positions = s.capture.positions.size();
    const bool worker_active = harvest_worker_on();
    if (worker_active) {
        std::exception_ptr error;
        bool full = false;
        {
            std::lock_guard<std::mutex> lk(harvest_w_->mu);
            error = harvest_w_->error;
            full = harvest_w_->queued == harvest_w_->q.size();
            if (!error && !full) {
                harvest_w_->q[(harvest_w_->first + harvest_w_->queued) % harvest_w_->q.size()] = submitted;
                ++harvest_w_->queued;
                s.inflight.store(true, std::memory_order_release);
            }
        }
        if (error || full) {
            abort_submit();
            if (error) std::rethrow_exception(error);
            throw std::logic_error("capture queue exceeded its two physical slots");
        }
    } else {
        s.inflight = true;
    }
    pos_queue_.erase(pos_queue_.begin());
    if (worker_active) harvest_w_->cv.notify_one();
    if (!old_sync && d2h_->compute_done) {
        ggml_backend_event_synchronize(d2h_->compute_done);
    }
    perf_.last_d2h_submit_us = ggml_time_us() - t_submit0;
    perf_.d2h_submit_us += perf_.last_d2h_submit_us;
    if (trace_) {
        kvmem_diag("KVMEM_TRACE harvest n=%zu q=%u k=%u v=%u host=%u gpu=%u bytes=%zu n_pos=%zu async=1 slot=%d\n",
                submitted_items, n_q, n_k, n_v, n_host, n_dev, off, submitted_positions, submitted);
    }
    d2h_->next = 1 - d2h_->next;
    return true;
} catch (...) {
    capture_fail();
    throw;
}

void llama_memory_kvmem::harvest_pending(ggml_backend_sched_t sched) try {
    harvest_check_error();
    if (want_decode_mean()) {
        decode_mean_ingest(sched);
        return;
    }
    const int64_t t_entry = ggml_time_us();
    const bool managed_packed = !pos_queue_.empty() && std::any_of(pos_queue_.front().ranges.begin(),
            pos_queue_.front().ranges.end(), [](const MeanWriteRange & range) {
                return range.route == MeanWriteRange::Route::Managed;
            });
    if (managed_packed) {
        // Packed KV is produced by the graph, unlike the pre-RoPE Mean-K
        // tensors. Finish that graph before gathering its accepted KV rows.
        // Keep the original batch alive until both producer routes own it.
        if (sched) ggml_backend_sched_synchronize(sched);
        else if (!kvmem_cuda_ok(cudaDeviceSynchronize(), "managed packed compute wait"))
            throw std::runtime_error("historical packed source graph did not complete");
        harvest_captured_packed(pos_queue_.front(), pos_queue_.front().positions.size());
    }
    perf_.last_commit_us = 0;
    perf_.last_d2h_wait_us = 0;
    perf_.last_pack_us = 0;
    perf_.last_nvme_us = 0;
    perf_.last_nvme_bytes = 0;
    perf_.last_nvme_syscalls = 0;
    perf_.last_sync_us = 0;
    perf_.last_d2d_us = 0;
    perf_.last_snap_wait_us = 0;
    perf_.last_d2h_submit_us = 0;
    if (d2h_ && d2h_->ok && !harvest_worker_on()) {
        d2h_commit(d2h_->next == 0 ? 1 : 0);
    }
    if (pending_capture_.empty() || pos_queue_.empty()) {
        // Recency has no Mean-K graph nodes. Consume the executed batch after
        // its packed jobs captured ownership, including between microbatches.
        if (managed_packed) pos_queue_.erase(pos_queue_.begin());
        harvest_flush();
        if (sched) {
            ggml_backend_sched_synchronize(sched);
        }
        harvest_full_blocks_async();
        return;
    }
    if (multi_gpu_) {
        // Finish the graph before capture tensors can be recycled by the next ubatch.
        // The CUDA layer path uses one D2H stream per physical GPU.
        const int64_t t_sync = ggml_time_us();
        if (sched) ggml_backend_sched_synchronize(sched);
        const int64_t sync_us = ggml_time_us() - t_sync;
        const int64_t read_before = perf_.multi_read_us;
        const int64_t host_before = perf_.multi_host_us;
        const char * batch_env = getenv("KVMEM_MULTI_D2H_BATCH");
        auto * reg = !kvmem_tensor_multi_gpu(model_) && !model_.devices.empty() ?
                ggml_backend_dev_backend_reg(model_.devices[0].dev) : nullptr;
        const bool is_cuda = reg && std::strcmp(ggml_backend_reg_name(reg), "CUDA") == 0;
        const bool batch_on = is_cuda && (!batch_env || batch_env[0] != '0');
        const size_t n_pos = pos_queue_.front().positions.size();
        const bool batched = batch_on && multi_d2h_submit();
        if (!batched) {
            cur_capture_ = std::move(pos_queue_.front());
            pos_queue_.erase(pos_queue_.begin());
            for (const CaptureNode & n : pending_capture_) {
                harvest_capture(n.t, n.il, n.which);
            }
            cur_capture_ = {};
        }
        const int64_t t_block = ggml_time_us();
        harvest_full_blocks_async();
        const int64_t block_us = ggml_time_us() - t_block;
        if (perf_.enabled) {
            fprintf(stderr, "KVMEM_MULTI_HARVEST mode=%s n=%zu sync_ms=%.3f read_ms=%.3f host_ms=%.3f block_ms=%.3f\n",
                    batched ? "batch" : "legacy", n_pos, sync_us / 1000.0,
                    (perf_.multi_read_us - read_before) / 1000.0,
                    (perf_.multi_host_us - host_before) / 1000.0, block_us / 1000.0);
        }
        return;
    }
    ggml_backend_t be = nullptr;
    if (sched) {
        for (const CaptureNode & n : pending_capture_) {
            if (n.t) {
                be = ggml_backend_sched_get_tensor_backend(sched, n.t);
                if (be) {
                    break;
                }
            }
        }
    }
    const uint32_t n_pos = pos_queue_.empty() ? 0u : (uint32_t) pos_queue_.front().positions.size();
    bool submitted = d2h_submit(be);
    if (!submitted) {
        // The previous microbatch can still be committing Q/Mean-K on the
        // worker. Preserve prefix order and never write the same accumulator
        // concurrently from synchronous fallback.
        harvest_flush();
        // The failed async path leaves its input at the queue front. Finish
        // the graph before the synchronous tensor reads below.
        if (sched) {
            ggml_backend_sched_synchronize(sched);
        }
        if (pos_queue_.empty()) {
            throw std::runtime_error("KVMem D2H fallback lost its position batch");
        }
        cur_capture_ = std::move(pos_queue_.front());
        pos_queue_.erase(pos_queue_.begin());
        for (const CaptureNode & n : pending_capture_) {
            harvest_capture(n.t, n.il, n.which);
        }
        cur_capture_ = {};
    }
    harvest_full_blocks_async();
    const int64_t entry_us = ggml_time_us() - t_entry;
    perf_.harvest_entry_us += entry_us;
    perf_.n_ubatch += 1;
    perf_.n_tok += n_pos;
    if (perf_.enabled) {
        fprintf(stderr, "KVMEM_HARVEST ubatch=%u n=%u is_mtp=0 "
                "harvest_entry_us=%lld sync_us=%lld d2d_us=%lld snap_wait_us=%lld "
                "d2h_submit_us=%lld commit_us=%lld pack_us=%lld "
                "nvme_us=%lld nvme_bytes=%llu nvme_syscalls=%llu\n",
                perf_.n_ubatch, n_pos,
                (long long) entry_us,
                (long long) perf_.last_sync_us,
                (long long) perf_.last_d2d_us,
                (long long) perf_.last_snap_wait_us,
                (long long) perf_.last_d2h_submit_us,
                (long long) perf_.last_commit_us,
                (long long) perf_.last_pack_us,
                (long long) perf_.last_nvme_us,
                (unsigned long long) perf_.last_nvme_bytes,
                (unsigned long long) perf_.last_nvme_syscalls);
    }
} catch (...) {
    capture_fail();
    throw;
}

void llama_memory_kvmem::reset_query_acc() {
    // Repair the size, not only the contents. attach_conv() installs the
    // incoming conversation's accumulators and then resizes them, and that
    // resize can throw between the two; reset_policy() calls this to make the
    // object usable again, and score_retrieval() and get_query() index
    // q_count_ over 0..n_layer_-1 with no size guard.
    if (q_sum_.size() != n_layer_ || q_count_.size() != n_layer_) {
        q_sum_.assign(n_layer_, std::vector<float>(n_head_ * n_embd_head_, 0.0f));
        q_count_.assign(n_layer_, 0);
        return;
    }
    for (auto & s : q_sum_) {
        std::fill(s.begin(), s.end(), 0.0f);
    }
    std::fill(q_count_.begin(), q_count_.end(), 0);
}

void llama_memory_kvmem::bytes_to_f32_token_major(const uint8_t * data, ggml_type type,
                                                  int64_t d, int64_t h, int64_t n,
                                                  size_t nb0, size_t nb1, size_t nb2,
                                                  std::vector<float> & out) {
    out.assign(static_cast<size_t>(n * h * d), 0.0f);
    if (!data) {
        return;
    }
    for (int64_t tok = 0; tok < n; ++tok) {
        for (int64_t head = 0; head < h; ++head) {
            for (int64_t dim = 0; dim < d; ++dim) {
                const size_t off = static_cast<size_t>(tok * nb2 + head * nb1 + dim * nb0);
                float val = 0.0f;
                if (type == GGML_TYPE_F32) {
                    val = *reinterpret_cast<const float *>(data + off);
                } else if (type == GGML_TYPE_F16) {
                    val = ggml_fp16_to_fp32(*reinterpret_cast<const ggml_fp16_t *>(data + off));
                } else if (type == GGML_TYPE_BF16) {
                    val = ggml_bf16_to_fp32(*reinterpret_cast<const ggml_bf16_t *>(data + off));
                }
                out[static_cast<size_t>(tok * h * d + head * d + dim)] = val;
            }
        }
    }
}

void llama_memory_kvmem::bytes_to_f16_token_major(const uint8_t * data, ggml_type type,
                                                  int64_t d, int64_t h, int64_t n,
                                                  size_t nb0, size_t nb1, size_t nb2,
                                                  std::vector<uint16_t> & out) {
    out.assign(static_cast<size_t>(n * h * d), 0);
    if (!data || type != GGML_TYPE_F16) {
        return;
    }
    const bool packed = nb0 == sizeof(uint16_t) &&
            nb1 == static_cast<size_t>(d) * sizeof(uint16_t) &&
            nb2 == static_cast<size_t>(h) * static_cast<size_t>(d) * sizeof(uint16_t);
    if (packed) {
        std::memcpy(out.data(), data, out.size() * sizeof(uint16_t));
        return;
    }
    for (int64_t tok = 0; tok < n; ++tok) {
        for (int64_t head = 0; head < h; ++head) {
            for (int64_t dim = 0; dim < d; ++dim) {
                const size_t off = static_cast<size_t>(tok * nb2 + head * nb1 + dim * nb0);
                out[static_cast<size_t>(tok * h * d + head * d + dim)] =
                        *reinterpret_cast<const uint16_t *>(data + off);
            }
        }
    }
}

void llama_memory_kvmem::tensor_to_f32_token_major(const ggml_tensor * t, std::vector<float> & out, int64_t * read_us) {
    std::vector<uint8_t> tmp;
    const uint8_t * data = nullptr;
    if (t->buffer && ggml_backend_buffer_is_host(t->buffer)) {
        data = static_cast<const uint8_t *>(t->data);
    } else {
        tmp.resize(ggml_nbytes(t));
        const int64_t t_read = read_us ? ggml_time_us() : 0;
        kvmem_tensor_get(t, tmp.data(), 0, tmp.size());
        if (read_us) *read_us += ggml_time_us() - t_read;
        data = tmp.data();
    }
    bytes_to_f32_token_major(data, t->type, t->ne[0], t->ne[1], t->ne[2],
                             t->nb[0], t->nb[1], t->nb[2], out);
}

void llama_memory_kvmem::commit_mean_k(
        const CaptureBatch & batch, uint32_t il, const std::vector<float> & flat) {
    if (flat.size() < batch.positions.size() * n_embd_k_) {
        throw std::runtime_error("historical Mean-K capture has too few tensor elements");
    }
    for (const auto & range : batch.ranges) {
        if (range.route == MeanWriteRange::Route::Closed) continue;
        if (range.route == MeanWriteRange::Route::Legacy) {
            raw_->write_layer_mean_k(range.original_begin, range.token_count, il,
                flat.data() + static_cast<size_t>(range.source_token) * n_embd_k_);
            continue;
        }
        const auto & context = range.writer;
        const auto & storage = context->storage();
        if (!storage->config().mean_k_elements) continue;
        const auto progress = storage->write_progress(context->writer(), context->handle(), il);
        if (!progress || storage->config().mean_k_elements != n_embd_k_ ||
                progress->mean_k_tokens < range.token_offset) {
            throw std::logic_error("historical Mean-K capture has a missing prefix or invalid writer");
        }
        // A replay/commit retry cannot add the same token to the sum twice.
        // Only the original generation's still-missing suffix is written.
        const uint32_t skip = std::min(range.token_count, progress->mean_k_tokens - range.token_offset);
        if (skip == range.token_count) continue;
        std::vector<float> sum(n_embd_k_, 0.0f);
        for (uint32_t i = skip; i < range.token_count; ++i) {
            const auto * src = flat.data() + static_cast<size_t>(range.source_token + i) * n_embd_k_;
            for (uint32_t d = 0; d < n_embd_k_; ++d) sum[d] += src[d];
        }
        if (!storage->write_mean_k_slice(context->writer(), context->handle(), il,
                range.token_offset + skip, range.token_count - skip, sum)) {
            throw std::logic_error("historical Mean-K capture rejected its original writer/range");
        }
        storage->publish_complete(context->repository(), context->writer(), context->handle());
    }
}

void llama_memory_kvmem::harvest_from_host(const CaptureBatch & batch, int il, char which, const uint8_t * host,
                                           ggml_type type, int64_t d, int64_t h, int64_t ntok,
                                           size_t nb0, size_t nb1, size_t nb2) {
    if (which == 'v') {
        return;
    }
    if (!host || il < 0 || static_cast<uint32_t>(il) >= n_layer_ || batch.positions.empty()) {
        return;
    }
    // The worker calls this through `this`, and a store swap moves raw_ and
    // q_sum_ out and back in. detach_conv() drains the pipe first, so the
    // window is not reachable today; guard it anyway, because the alternative
    // is a null dereference and an out-of-range write on an empty vector.
    if (!raw_ || q_sum_.size() != n_layer_ || q_count_.size() != n_layer_) {
        return;
    }
    const uint32_t n = static_cast<uint32_t>(batch.positions.size());
    if (which == 'k') {
        if (d * h != n_embd_k_) throw std::logic_error("historical Mean-K tensor dimension mismatch");
        std::vector<float> flat;
        if (type == GGML_TYPE_F16) {
            std::vector<uint16_t> flat16;
            bytes_to_f16_token_major(host, type, d, h, ntok, nb0, nb1, nb2, flat16);
            flat.resize(flat16.size());
            ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t *>(flat16.data()),
                                  flat.data(), static_cast<int64_t>(flat16.size()));
        } else {
            bytes_to_f32_token_major(host, type, d, h, ntok, nb0, nb1, nb2, flat);
        }
        commit_mean_k(batch, static_cast<uint32_t>(il), flat);
    } else if (which == 'q') {
        std::vector<float> flat;
        bytes_to_f32_token_major(host, type, d, h, ntok, nb0, nb1, nb2, flat);
        const uint32_t qdim = n_head_ * n_embd_head_;
        if (flat.size() < static_cast<size_t>(n) * qdim) {
            return;
        }
        for (uint32_t i = 0; i < n; ++i) {
            if (!query_contains(batch.positions[i])) continue;
            float * dst = q_sum_[static_cast<uint32_t>(il)].data();
            const float * src = flat.data() + i * qdim;
            for (uint32_t d0 = 0; d0 < qdim; ++d0) {
                dst[d0] += src[d0];
            }
            q_count_[static_cast<uint32_t>(il)]++;
        }
    }
}

void llama_memory_kvmem::harvest_capture(ggml_tensor * t, int il, char which) {
    if (which == 'v') {
        return;
    }
    if (!t || il < 0 || static_cast<uint32_t>(il) >= n_layer_) {
        return;
    }
    if (cur_capture_.positions.empty()) {
        return;
    }
    std::vector<float> flat;
    const bool timed = perf_.enabled && multi_gpu_;
    const int64_t t_host = timed ? ggml_time_us() : 0;
    int64_t read_us = 0;
    tensor_to_f32_token_major(t, flat, timed ? &read_us : nullptr);
    if (timed) perf_.multi_read_us += read_us;
    const uint32_t n = static_cast<uint32_t>(cur_capture_.positions.size());
    if (n == 0) {
        return;
    }
    if (which == 'k') {
        if (t->ne[0] * t->ne[1] != n_embd_k_)
            throw std::logic_error("historical Mean-K tensor dimension mismatch");
        commit_mean_k(cur_capture_, static_cast<uint32_t>(il), flat);
        static bool dumped = false;
        if (!dumped && getenv("KVMEM_DUMP_CAPTURE") && il == 0) {
            dumped = true;
            double acc = 0;
            for (float x : flat) {
                acc += static_cast<double>(x) * x;
            }
            kvmem_diag("KVMEM_DUMP k_prerope layer0 n=%u rms=%.6f shape_elems=%zu\n",
                    n, std::sqrt(acc / std::max<size_t>(flat.size(), 1)), flat.size());
        }
    } else if (which == 'q') {
        const uint32_t qdim = n_head_ * n_embd_head_;
        if (flat.size() < static_cast<size_t>(n) * qdim) {
            return;
        }
        for (uint32_t i = 0; i < n; ++i) {
            if (!query_contains(cur_capture_.positions[i])) continue;
            float * dst = q_sum_[static_cast<uint32_t>(il)].data();
            const float * src = flat.data() + i * qdim;
            for (uint32_t d = 0; d < qdim; ++d) {
                dst[d] += src[d];
            }
            q_count_[static_cast<uint32_t>(il)]++;
        }
    }
    if (timed) perf_.multi_host_us += ggml_time_us() - t_host - read_us;
}

void llama_memory_kvmem::commit_harvest_packed(const HarvestVJob & job, const uint8_t * bytes) {
    if (job.writer) {
        const auto & storage = job.writer->storage();
        if (!storage) throw std::logic_error("historical harvest has no captured storage");
        const bool written = job.is_k
            ? storage->write_packed_k_slice(job.writer->writer(), job.writer->handle(),
                                           job.il, job.token_offset, job.n, bytes, job.nbytes)
            : storage->write_packed_v_slice(job.writer->writer(), job.writer->handle(),
                                           job.il, job.token_offset, job.n, bytes, job.nbytes);
        if (!written) throw std::runtime_error("historical harvest rejected captured generation or range");
        storage->publish_complete(job.writer->repository(), job.writer->writer(), job.writer->handle());
        return;
    }
    if (job.is_k) raw_->write_layer_k_gpu(job.pos0, job.n, job.il, bytes);
    else raw_->write_layer_v_gpu(job.pos0, job.n, job.il, bytes);
}

uint8_t * llama_memory_kvmem::harvest_packed_fallback(size_t bytes, std::vector<uint8_t> & temporary) {
    if (harvest_v_job_limit_) {
        if (bytes > harvest_v_fallback_.size())
            throw std::logic_error("historical packed fallback exceeds admitted workspace");
        return harvest_v_fallback_.data();
    }
    temporary.resize(bytes);
    return temporary.data();
}

void llama_memory_kvmem::enqueue_harvest_packed(HarvestVJob job) try {
    if (job.historical_gpu_slot >= 0 &&
            (size_t(job.historical_gpu_slot) >= historical_gpu_readers_.size() ||
             historical_gpu_retired_[job.historical_gpu_slot] ||
             historical_gpu_readers_[job.historical_gpu_slot] == UINT32_MAX))
        throw std::logic_error("historical GPU source admission has no live slot");
    if (multi_gpu_) {
        std::vector<uint8_t> packed(job.nbytes);
        kvmem_tensor_get(job.vt, packed.data(), job.tensor_off, job.nbytes);
        commit_harvest_packed(job, packed.data());
        return;
    }
    auto * base = kvmem_cuda_tensor_ptr(job.vt);
    job.gpu_src = base && job.nbytes ? base + job.tensor_off : nullptr;
    const auto enqueue = [&] {
        if (harvest_v_job_limit_ && harvest_v_jobs_.size() >= harvest_v_job_limit_) return false;
        // No allocation may fail between registering the DMA source and
        // retaining its writer. The job move itself is nonthrowing.
        if (harvest_v_jobs_.size() == harvest_v_jobs_.capacity()) {
            if (harvest_v_job_limit_)
                throw std::logic_error("historical packed queue lost its admitted capacity");
            harvest_v_jobs_.reserve(std::max<size_t>(8, harvest_v_jobs_.capacity() * 2));
        }
        job.pin_off = kvmem_stageout_used();
        if (!kvmem_stageout_enqueue(job.gpu_src, job.nbytes)) return false;
        if (job.historical_gpu_slot >= 0) ++historical_gpu_readers_[job.historical_gpu_slot];
        harvest_v_jobs_.push_back(std::move(job));
        return true;
    };
    if (job.gpu_src && enqueue()) return;
    if (job.gpu_src && (!harvest_v_jobs_.empty() || harvest_v_pending_.slot >= 0)) {
        harvest_gpu_v_flush_slab();
        if (enqueue()) return;
    }
    std::vector<uint8_t> temporary;
    auto * packed = harvest_packed_fallback(job.nbytes, temporary);
    kvmem_tensor_get(job.vt, packed, job.tensor_off, job.nbytes);
    commit_harvest_packed(job, packed);
} catch (...) {
    if (historical_capture_) historical_capture_->fail();
    throw;
}

void llama_memory_kvmem::harvest_captured_packed(const CaptureBatch & batch, uint32_t n_keep) {
    n_keep = std::min(n_keep, static_cast<uint32_t>(batch.positions.size()));
    bool initialized = false;
    for (const auto & range : batch.ranges) {
        if (range.source_token >= n_keep) break;
        if (range.route != MeanWriteRange::Route::Managed) continue;
        const auto & writer = range.writer;
        if (!kv_ || v_trans_ || !writer || !writer->storage())
            throw std::logic_error("historical packed capture has no admitted producer");
        const auto metadata = writer->writer().metadata();
        if (!metadata) throw std::logic_error("historical packed capture lost its generation");
        const auto meta = *metadata;
        const uint32_t take = std::min(range.token_count, n_keep - range.source_token);
        if (range.gpu_slot < 0 || uint32_t(range.gpu_slot) >= n_slots_ ||
                range.token_offset > meta.valid_tokens || take > meta.valid_tokens - range.token_offset ||
                meta.original_begin + range.token_offset != range.original_begin ||
                meta.original_begin + meta.valid_tokens != range.original_end)
            throw std::logic_error("historical packed capture has no valid admitted source/range");
        const size_t krow = ggml_row_size(type_k_, n_embd_k_);
        const size_t vrow = ggml_row_size(type_v_, n_embd_v_);
        const auto & cfg = writer->storage()->config();
        if (cfg.packed_k_bytes_per_token != krow || cfg.packed_v_bytes_per_token != vrow)
            throw std::logic_error("historical packed capture has incompatible row formats");
        if (!initialized && !multi_gpu_) {
            const bool ready = kvmem_stagein_gpu_ready(size_t(block_tokens_) * std::max(n_embd_k_, n_embd_v_),
                std::max(krow, vrow) * size_t(block_tokens_));
            if (historical_capture_ && !ready)
                throw std::runtime_error("historical packed capture exceeded reserved staging capacity");
            initialized = true;
        }
        const uint32_t end = range.token_offset + take;
        const uint32_t cell0 = uint32_t(range.gpu_slot) * block_tokens_;
        for (uint32_t il = 0; il < n_layer_; ++il) {
            if (!kvmem_cache_has_layer(kv_, il)) continue;
            for (bool is_k : {true, false}) {
                // Enqueueing K may flush/commit an older slab, including V.
                // Refresh component progress after each such transition.
                const auto progress = writer->storage()->write_progress(writer->writer(), writer->handle(), il);
                if (!progress) throw std::logic_error("historical packed capture is missing a layer allocation");
                uint32_t prefix = is_k ? progress->packed_k_tokens : progress->packed_v_tokens;
                const auto include_queued = [&](const std::vector<HarvestVJob> & jobs) {
                    for (const auto & job : jobs) {
                        if (job.writer != writer || job.il != il || job.is_k != is_k || job.committed) continue;
                        if (job.token_offset > prefix)
                            throw std::logic_error("historical packed queue contains a hole");
                        prefix = std::max(prefix, job.token_offset + job.n);
                    }
                };
                include_queued(harvest_v_pending_.jobs);
                include_queued(harvest_v_jobs_);
                if (prefix < range.token_offset)
                    throw std::logic_error("historical packed capture is missing an earlier executed prefix");
                if (prefix >= end) continue; // Retries/replays never rewrite a captured prefix.
                HarvestVJob job;
                const size_t row = is_k ? krow : vrow;
                job.writer = writer;
                if (historical_capture_) job.historical_gpu_slot = range.gpu_slot;
                job.il = il;
                job.is_k = is_k;
                job.token_offset = prefix;
                job.pos0 = static_cast<uint32_t>(meta.original_begin) + prefix;
                job.n = end - prefix;
                job.nbytes = size_t(job.n) * row;
                job.tensor_off = size_t(cell0 + prefix) * row;
                job.vt = is_k ? kv_->get_k_storage(il) : kv_->get_v_storage(il);
                if (!job.vt || job.tensor_off > ggml_nbytes(job.vt) ||
                        job.nbytes > ggml_nbytes(job.vt) - job.tensor_off)
                    throw std::logic_error("historical packed capture exceeds its GPU storage");
                enqueue_harvest_packed(std::move(job));
            }
        }
    }
    if (initialized && !harvest_v_jobs_.empty()) harvest_gpu_v_flush_slab();
}

void llama_memory_kvmem::harvest_write_batch() {
    HarvestVBatch & b = harvest_v_pending_;
    if (b.jobs.empty()) {
        b.slot = -1;
        return;
    }
    auto fallback = [&](const HarvestVJob & j) {
        if (j.nbytes == 0) {
            return;
        }
        std::vector<uint8_t> temporary;
        auto * packed = harvest_packed_fallback(j.nbytes, temporary);
        auto commit = [&](const uint8_t * src) {
            commit_harvest_packed(j, src);
        };
        if (j.gpu_src &&
            kvmem_copy(packed, j.gpu_src, j.nbytes,
                       cudaMemcpyDeviceToHost) == cudaSuccess) {
            commit(packed);
            return;
        }
        if (j.vt) {
            kvmem_tensor_get(j.vt, packed, j.tensor_off, j.nbytes);
            commit(packed);
        } else if (j.writer) {
            throw std::runtime_error("historical harvest fallback lost its captured source");
        }
    };
    const uint8_t * base = (b.slot >= 0) ? kvmem_stageout_slot_base(b.slot) : nullptr;
    const int64_t t0 = ggml_time_us();
    for (HarvestVJob & j : b.jobs) {
        if (j.committed) continue;
        const uint8_t * p = (base && j.nbytes) ? base + j.pin_off : nullptr;
        if (p) {
            commit_harvest_packed(j, p);
        } else {
            fallback(j);
        }
        // A later job may throw (for example allocation failure). Retrying
        // this batch must not submit an already committed write-once slice.
        j.committed = true;
        complete_historical_gpu_source(j.historical_gpu_slot);
    }
    if (retr_.enabled) {
        retr_.stage_out_host_us += ggml_time_us() - t0;
    }
    b.jobs.clear();
    b.slot = -1;
}

void llama_memory_kvmem::harvest_gpu_v_flush_slab() try {
    if (harvest_v_jobs_.empty()) {
        return;
    }
    int64_t * gpu_us = retr_.enabled ? &retr_.stage_out_gpu_us : nullptr;
    if (harvest_v_pending_.slot >= 0) {
        if (!kvmem_stageout_wait(harvest_v_pending_.slot, gpu_us)) {
            harvest_v_pending_.slot = -1;
        }
        // Complete the old host commit before handing off the next slab. If
        // a component commit throws, both batches retain their writer/source
        // ownership; no newly submitted DMA loses its jobs during unwinding.
        harvest_write_batch();
    } else if (!harvest_v_pending_.jobs.empty()) {
        harvest_write_batch();
    }
    const int slot = kvmem_stageout_submit(gpu_us);
    harvest_v_pending_.slot = slot;
    harvest_v_pending_.jobs.swap(harvest_v_jobs_);
    if (harvest_v_pending_.slot < 0) {
        harvest_write_batch();
    }
} catch (...) {
    if (historical_capture_) historical_capture_->fail();
    throw;
}

void llama_memory_kvmem::harvest_gpu_v_commit() try {
    harvest_gpu_v_flush_slab();
    int64_t * gpu_us = retr_.enabled ? &retr_.stage_out_gpu_us : nullptr;
    if (harvest_v_pending_.slot >= 0) {
        if (!kvmem_stageout_wait(harvest_v_pending_.slot, gpu_us)) {
            harvest_v_pending_.slot = -1;
        }
        harvest_write_batch();
    } else if (!harvest_v_pending_.jobs.empty()) {
        harvest_write_batch();
    }
    kvmem_stageout_clear();
    harvest_gpu_queued_.clear();
} catch (...) {
    if (historical_capture_) historical_capture_->fail();
    throw;
}

void llama_memory_kvmem::harvest_gpu_v(uint32_t block_id) {
    if (v_trans_ || !raw_ || !kv_) {
        return;
    }
    if (block_id < harvest_gpu_queued_.size() && harvest_gpu_queued_[block_id]) {
        return;
    }
    auto & store = runtime_->store();
    if (block_id >= store.block_count()) {
        return;
    }
    const kvmem::KvMemBlock & blk = store.blocks()[block_id];
    if (blk.gpu_slot < 0 || blk.n_tokens == 0) {
        return;
    }
    const auto & directory = runtime_->historical_shadow_directory();
    const auto identity = directory.observe(block_id);
    if (identity.state == kvmem::HistoricalKvShadowDirectory::State::Unmanaged &&
            blk.orig_pos_start != uint64_t(block_id) * block_tokens_)
        throw std::logic_error("non-aligned packed harvest requires a historical payload identity");
    // Managed production is admitted with the original execution batch.
    // A later stage-out / directory scan cannot discover new producer work.
    if (identity.state != kvmem::HistoricalKvShadowDirectory::State::Unmanaged) return;
    const llama_kv_cells & cells = kv_->get_cells(0);
    const uint32_t idx = static_cast<uint32_t>(blk.gpu_slot) * block_tokens_;
    if (idx >= cells.size() || cells.is_empty(idx)) {
        return;
    }
    const size_t krow = ggml_row_size(type_k_, n_embd_k_);
    const size_t vrow = ggml_row_size(type_v_, n_embd_v_);
    const uint32_t nt = blk.n_tokens;
    const uint32_t cell0 = static_cast<uint32_t>(blk.gpu_slot) * block_tokens_;
    if (!multi_gpu_) {
        kvmem_stagein_gpu_ready((size_t) block_tokens_ * std::max(n_embd_k_, n_embd_v_),
                                std::max(krow, vrow) * (size_t) block_tokens_);
    }
    auto enqueue_or_get = [&](ggml_tensor * t, uint32_t il, uint32_t nget,
                              uint32_t token_offset, bool is_k) {
        const size_t row = is_k ? krow : vrow;
        const size_t nbytes = static_cast<size_t>(nget - token_offset) * row;
        const size_t toff = static_cast<size_t>(cell0 + token_offset) * row;
        HarvestVJob job;
        job.pos0 = blk.orig_pos_start + token_offset;
        job.n = nget - token_offset;
        job.il = il;
        job.is_k = is_k;
        job.nbytes = nbytes;
        job.vt = t;
        job.tensor_off = toff;
        job.token_offset = token_offset;
        enqueue_harvest_packed(std::move(job));
    };
    uint32_t n_ok = 0;
    uint32_t n_skip = 0;
    for (uint32_t il = 0; il < n_layer_; ++il) {
        if (!kvmem_cache_has_layer(kv_, static_cast<int32_t>(il))) {
            continue;
        }
        const uint32_t nget = (cell0 < kv_size_)
                ? std::min(nt, kv_size_ - cell0) : 0;
        if (nget == 0) {
            continue;
        }
        ggml_tensor * kt = kv_->get_k_storage(static_cast<int32_t>(il));
        if (kt && krow && !raw_->has_k_gpu(block_id, il, nget)) {
            enqueue_or_get(kt, il, nget, 0, true);
            n_ok++;
        } else {
            n_skip++;
        }
        ggml_tensor * vt = kv_->get_v_storage(static_cast<int32_t>(il));
        if (vt && !v_trans_ && vrow && !raw_->has_v_gpu(block_id, il, nget)) {
            enqueue_or_get(vt, il, nget, 0, false);
            n_ok++;
        } else {
            n_skip++;
        }
    }
    if (n_ok > 0) {
        if (harvest_gpu_queued_.size() <= block_id) {
            harvest_gpu_queued_.resize(block_id + 1);
        }
        harvest_gpu_queued_[block_id] = 1;
    }
    if (trace_) {
        kvmem_diag("KVMEM_TRACE harvest_kv block=%u slot=%d n=%u layers=%u skip=%u\n",
                block_id, blk.gpu_slot, blk.n_tokens, n_ok, n_skip);
    }
}

void llama_memory_kvmem::harvest_full_blocks_async() {
    if (retrieval_pinned_ || replay_ || v_trans_ || !raw_ || !kv_ || !runtime_) {
        return;
    }
    auto & store = runtime_->store();
    uint32_t n_enq = 0;
    for (const auto & b : store.blocks()) {
        const bool managed = runtime_->historical_shadow_directory().observe(b.block_id).state !=
                kvmem::HistoricalKvShadowDirectory::State::Unmanaged;
        if (managed || b.gpu_slot < 0 || b.n_tokens != block_tokens_) {
            continue;
        }
        if (b.block_id < harvest_gpu_queued_.size() && harvest_gpu_queued_[b.block_id]) {
            continue;
        }
        bool need = false;
        for (uint32_t il = 0; il < n_layer_; ++il) {
            if (!kvmem_cache_has_layer(kv_, static_cast<int32_t>(il))) {
                continue;
            }
            if (!raw_->has_k_gpu(b.block_id, il, b.n_tokens) ||
                (!v_trans_ && !raw_->has_v_gpu(b.block_id, il, b.n_tokens))) {
                need = true;
                break;
            }
        }
        if (!need) {
            continue;
        }
        harvest_gpu_v(b.block_id);
        n_enq++;
    }
    if (!harvest_v_jobs_.empty()) {
        harvest_gpu_v_flush_slab();
    }
    if (trace_ && n_enq > 0) {
        kvmem_diag("KVMEM_TRACE harvest_full_async n=%u pending_slot=%d\n",
                n_enq, harvest_v_pending_.slot);
    }
}

void llama_memory_kvmem::decode_mean_reset() {
    decode_mean_flush();
    decode_mean_discard();
    decode_mean_n_ = 0;
    decode_mean_block_ = ~0u;
    decode_mean_pos0_ = 0;
    decode_mean_target_ = {};
    decode_mean_committed_.clear();
}

void llama_memory_kvmem::decode_mean_discard() try {
    harvest_check_error();
    if (!multi_gpu_) kvmem_meank_sync();
    decode_mean_pending_k_.clear();
    decode_mean_pending_batch_ = {};
} catch (...) {
    capture_fail();
    throw;
}

void llama_memory_kvmem::decode_mean_shutdown() {
    // Caller has settled source computation before invoking this. A failed
    // accumulator permits checked cancellation, never another flush/replay.
    if (!multi_gpu_) kvmem_meank_cancel();
    decode_mean_pending_k_.clear();
    decode_mean_pending_batch_ = {};
    decode_mean_target_ = {};
    decode_mean_n_ = 0;
    decode_mean_block_ = ~0u;
}

void llama_memory_kvmem::decode_mean_zero_acc() {
    if (!multi_gpu_ && kvmem_meank_ready(n_layer_, n_embd_k_)) {
        for (uint32_t il = 0; il < n_layer_; ++il) {
            kvmem_meank_zero(il);
        }
    }
    decode_mean_host_.assign(n_layer_, std::vector<float>(n_embd_k_, 0.0f));
    decode_mean_src_.assign(n_layer_, 0);
    decode_mean_committed_.assign(n_layer_, 0);
}

void llama_memory_kvmem::decode_mean_print_sum() {
    if (decode_mean_stats_.printed) {
        return;
    }
    decode_mean_stats_.printed = true;
    kvmem_diag("KVMEM_DECODE_MEAN n_tok=%u n_flush=%u n_gpu=%u n_host=%u n_miss=%u "
            "last_block=%d last_n=%u last_layers=%u last_rms=%.6f\n",
            decode_mean_stats_.n_tok, decode_mean_stats_.n_flush,
            decode_mean_stats_.n_gpu, decode_mean_stats_.n_host,
            decode_mean_stats_.n_miss,
            decode_mean_stats_.last_block == ~0u
                    ? -1 : static_cast<int>(decode_mean_stats_.last_block),
            decode_mean_stats_.last_n, decode_mean_stats_.last_layers,
            decode_mean_stats_.last_rms);
}

bool llama_memory_kvmem::decode_mean_add_host_layer(ggml_tensor * t, int il,
                                                   uint32_t tok0, uint32_t n_add) {
    if (!t || il < 0 || static_cast<uint32_t>(il) >= n_layer_ || n_add == 0) {
        return false;
    }
    if (decode_mean_host_.size() != n_layer_) {
        decode_mean_host_.assign(n_layer_, std::vector<float>(n_embd_k_, 0.0f));
        decode_mean_src_.assign(n_layer_, 0);
    }
    const uint32_t uil = static_cast<uint32_t>(il);
    if (decode_mean_src_[uil] == 1) {
        if (decode_mean_host_[uil].size() != n_embd_k_) {
            decode_mean_host_[uil].assign(n_embd_k_, 0.0f);
        }
        if (!kvmem_meank_d2h(uil, decode_mean_host_[uil].data(), n_embd_k_)) {
            return false;
        }
        kvmem_meank_zero(uil);
        decode_mean_src_[uil] = 2;
    }
    std::vector<float> flat;
    tensor_to_f32_token_major(t, flat);
    const size_t need = (static_cast<size_t>(tok0) + n_add) * n_embd_k_;
    if (flat.size() < need) {
        return false;
    }
    if (decode_mean_host_[uil].size() != n_embd_k_) {
        decode_mean_host_[uil].assign(n_embd_k_, 0.0f);
    }
    const float * src = flat.data() + static_cast<size_t>(tok0) * n_embd_k_;
    for (uint32_t t_i = 0; t_i < n_add; ++t_i) {
        for (uint32_t d = 0; d < n_embd_k_; ++d) {
            decode_mean_host_[uil][d] += src[static_cast<size_t>(t_i) * n_embd_k_ + d];
        }
    }
    decode_mean_src_[uil] = 2;
    decode_mean_stats_.n_host += n_add;
    return true;
}

void llama_memory_kvmem::decode_mean_ingest(ggml_backend_sched_t sched) try {
    harvest_check_error();
    if (!decode_mean_pending_batch_.positions.empty())
        throw std::logic_error("decode Mean-K must accept or discard the previous draft before another ingest");
    const auto requires_managed_mean = [](const CaptureBatch & batch) {
        return std::any_of(batch.ranges.begin(), batch.ranges.end(), [](const MeanWriteRange & range) {
            return range.route == MeanWriteRange::Route::Managed && range.writer &&
                   range.writer->storage()->config().mean_k_elements != 0;
        });
    };
    if (pending_capture_.empty() || pos_queue_.empty()) {
        if (!pos_queue_.empty() && requires_managed_mean(pos_queue_.front()))
            throw std::logic_error("historical decode Mean-K has no capture nodes for its admitted range");
        decode_mean_stats_.n_miss++;
        if (trace_ || decode_mean_stats_.n_miss <= 3) {
            kvmem_diag("KVMEM_DECODE_MEAN ingest_skip capture=%zu posq=%zu "
                    "pinned=%d replay=%d graph_k=%d\n",
                    pending_capture_.size(), pos_queue_.size(),
                    (int) retrieval_pinned_, (int) replay_, (int) graph_has_k_);
        }
        return;
    }
    if (sched) {
        ggml_backend_sched_synchronize(sched);
    } else if (!kvmem_cuda_ok(cudaDeviceSynchronize(), "decode Mean-K source computation"))
        throw std::runtime_error("decode Mean-K source computation did not complete");
    decode_mean_pending_k_.clear();
    decode_mean_pending_k_.reserve(pending_capture_.size());
    for (const CaptureNode & n : pending_capture_) {
        if (n.t && n.which == 'k') {
            decode_mean_pending_k_.push_back(n);
        }
    }
    decode_mean_pending_batch_ = std::move(pos_queue_.front());
    pos_queue_.erase(pos_queue_.begin());
    if (decode_mean_pending_k_.empty()) {
        if (requires_managed_mean(decode_mean_pending_batch_))
            throw std::logic_error("historical decode Mean-K has no K nodes for its admitted range");
        decode_mean_stats_.n_miss++;
        if (trace_ || decode_mean_stats_.n_miss <= 3) {
            kvmem_diag("KVMEM_DECODE_MEAN ingest_skip no_k_nodes capture=%zu\n",
                    pending_capture_.size());
        }
        decode_mean_pending_batch_ = {};
        return;
    }
    if (trace_ && decode_mean_stats_.n_tok == 0 && !decode_mean_pending_k_.empty()) {
        const CaptureNode & n0 = decode_mean_pending_k_.front();
        kvmem_diag("KVMEM_TRACE decode_mean_ingest n_pos=%zu n_k=%zu type=%s gpu=%d\n",
                decode_mean_pending_batch_.positions.size(), decode_mean_pending_k_.size(),
                n0.t ? ggml_type_name(n0.t->type) : "?",
                n0.t && kvmem_cuda_tensor_ptr(n0.t) ? 1 : 0);
    }
    if (decode_mean_pending_batch_.positions.size() == 1) {
        decode_mean_commit(1);
    }
} catch (...) {
    capture_fail();
    throw;
}

void llama_memory_kvmem::decode_mean_add_range(const MeanWriteRange & range, uint32_t n_add) {
    if (n_add == 0 || range.route == MeanWriteRange::Route::Closed || !raw_) {
        return;
    }
    const uint32_t tok0 = range.source_token;
    const uint32_t pos0 = range.original_begin;
    const uint32_t bid = range.block_id;
    if (range.route == MeanWriteRange::Route::Managed) {
        // Validate the whole admitted set before launching any accumulation.
        // A missing layer must not silently produce a partial-token sum that
        // later claims to cover the full accepted range.
        std::vector<uint8_t> seen(n_layer_, 0);
        for (const auto & node : decode_mean_pending_k_) {
            if (!node.t || node.il < 0 || static_cast<uint32_t>(node.il) >= n_layer_ ||
                    seen[node.il] || node.t->ne[0] * node.t->ne[1] != n_embd_k_ ||
                    node.t->ne[2] < int64_t(tok0) + n_add) {
                throw std::logic_error("historical decode Mean-K has invalid/duplicate capture layers");
            }
            seen[node.il] = 1;
        }
        for (uint32_t il : raw_cfg_.historical_gc_attention_layers) {
            if (!seen[il]) throw std::logic_error("historical decode Mean-K is missing an attention layer");
        }
    }
    if (decode_mean_block_ != bid || decode_mean_target_.route != range.route ||
            decode_mean_target_.writer != range.writer ||
            uint64_t(decode_mean_pos0_) + decode_mean_n_ != pos0) {
        if (decode_mean_n_ > 0) {
            decode_mean_flush();
        }
        decode_mean_zero_acc();
        decode_mean_block_ = bid;
        decode_mean_pos0_ = pos0;
        decode_mean_n_ = 0;
        decode_mean_target_ = range;
    }
    const bool gpu_ready = !multi_gpu_ && kvmem_meank_ready(n_layer_, n_embd_k_);
    bool any = false;
    for (const CaptureNode & n : decode_mean_pending_k_) {
        if (!n.t || static_cast<uint32_t>(n.il) >= n_layer_) {
            continue;
        }
        bool ok = false;
        // Once a layer has fallen back to a host sum, keep accumulating there;
        // replacing it with a later GPU sum would drop the earlier tokens.
        if (gpu_ready && (decode_mean_src_.size() <= static_cast<uint32_t>(n.il) ||
                          decode_mean_src_[n.il] != 2)) {
            const uint8_t * gpu = kvmem_cuda_tensor_ptr(n.t);
            if (gpu &&
                kvmem_meank_add(static_cast<uint32_t>(n.il), n.t->type, gpu,
                                tok0, n_add, n_embd_k_, n.t->ne[0],
                                n.t->nb[0], n.t->nb[1], n.t->nb[2])) {
                if (decode_mean_src_.size() != n_layer_) {
                    decode_mean_src_.assign(n_layer_, 0);
                }
                decode_mean_src_[static_cast<uint32_t>(n.il)] = 1;
                decode_mean_stats_.n_gpu += n_add;
                ok = true;
            }
        }
        if (!ok) {
            ok = decode_mean_add_host_layer(n.t, n.il, tok0, n_add);
        }
        if (!ok) {
            throw std::runtime_error("historical decode Mean-K failed to capture an admitted layer");
        }
        any = any || ok;
    }
    if (!any) {
        decode_mean_stats_.n_miss++;
        if (trace_ || decode_mean_stats_.n_miss <= 3) {
            kvmem_diag("KVMEM_DECODE_MEAN add_fail n_add=%u n_k=%zu gpu_ready=%d\n",
                    n_add, decode_mean_pending_k_.size(), (int) gpu_ready);
        }
        return;
    }
    decode_mean_n_ += n_add;
    decode_mean_stats_.n_tok += n_add;
}

void llama_memory_kvmem::decode_mean_commit(uint32_t n_keep) try {
    harvest_check_error();
    if (n_keep == 0 || decode_mean_pending_batch_.positions.empty()) {
        decode_mean_discard();
        return;
    }
    n_keep = std::min(n_keep, static_cast<uint32_t>(decode_mean_pending_batch_.positions.size()));
    harvest_captured_packed(decode_mean_pending_batch_, n_keep);
    for (const auto & range : decode_mean_pending_batch_.ranges) {
        if (range.source_token >= n_keep) break;
        const uint32_t take = std::min(range.token_count, n_keep - range.source_token);
        const uint32_t bid = range.block_id;
        decode_mean_add_range(range, take);
        // Accepted tokens only. Store n_tokens can include MTP drafts that
        // seq_rm will drop; do not flush on that watermark.
        if (decode_mean_block_ == bid && decode_mean_n_ > 0 &&
            uint64_t(decode_mean_pos0_) + decode_mean_n_ >= range.original_end) {
            decode_mean_flush();
        }
    }
    // The accumulation kernels read graph tensors on the transfer stream.
    // Complete them before releasing the original batch or allowing the next
    // graph to reuse those tensors, including an accepted partial draft.
    decode_mean_discard();
} catch (...) {
    capture_fail();
    throw;
}

void llama_memory_kvmem::decode_mean_flush() try {
    harvest_check_error();
    if (decode_mean_n_ == 0 || !raw_ || decode_mean_block_ == ~0u) {
        return;
    }
    if (!multi_gpu_) kvmem_meank_sync();
    std::vector<float> sum(n_embd_k_, 0.0f);
    uint32_t n_ok = 0;
    uint32_t n_gpu = 0;
    uint32_t n_host = 0;
    float rms = 0.0f;
    for (uint32_t il = 0; il < n_layer_; ++il) {
        if (!kvmem_cache_has_layer(kv_, static_cast<int32_t>(il))) {
            continue;
        }
        if (il < decode_mean_committed_.size() && decode_mean_committed_[il]) continue;
        const uint8_t src = (il < decode_mean_src_.size()) ? decode_mean_src_[il] : 0;
        bool got = false;
        if (src == 2 && il < decode_mean_host_.size() &&
            decode_mean_host_[il].size() == n_embd_k_) {
            std::memcpy(sum.data(), decode_mean_host_[il].data(),
                        n_embd_k_ * sizeof(float));
            got = true;
            n_host++;
        } else if (src == 1 && kvmem_meank_d2h(il, sum.data(), n_embd_k_)) {
            got = true;
            n_gpu++;
        }
        if (!got) {
            if (decode_mean_target_.route == MeanWriteRange::Route::Managed) {
                throw std::runtime_error("historical decode Mean-K could not read an accumulated layer");
            }
            continue;
        }
        if (capture_fault(CaptureFaultPoint::DecodeHostCommit, -1, il))
            throw std::runtime_error("injected decode Mean-K host commit failure");
        if (decode_mean_target_.route == MeanWriteRange::Route::Managed) {
            const auto & context = decode_mean_target_.writer;
            if (!context->storage()->write_mean_k_slice(context->writer(), context->handle(), il,
                    decode_mean_target_.token_offset, decode_mean_n_, sum)) {
                throw std::logic_error("historical decode Mean-K rejected its original writer/range");
            }
            context->storage()->publish_complete(context->repository(), context->writer(), context->handle());
        } else {
            raw_->write_layer_mean_sum(decode_mean_pos0_, decode_mean_n_, il, sum.data());
        }
        decode_mean_committed_[il] = 1;
        if (!multi_gpu_) kvmem_meank_zero(il);
        if (n_ok == 0 && n_embd_k_ > 0) {
            double acc = 0;
            for (uint32_t d = 0; d < n_embd_k_; ++d) {
                acc += static_cast<double>(sum[d]) * sum[d];
            }
            rms = std::sqrt(acc / static_cast<double>(n_embd_k_)) /
                    static_cast<float>(std::max(1u, decode_mean_n_));
        }
        n_ok++;
    }
    if (!multi_gpu_) kvmem_meank_sync();
    kvmem_diag("KVMEM_DECODE_MEAN flush block=%u n=%u pos0=%u layers=%u gpu=%u host=%u rms=%.6f\n",
            decode_mean_block_, decode_mean_n_, decode_mean_pos0_, n_ok, n_gpu, n_host, rms);
    if (trace_) {
        kvmem_diag("KVMEM_TRACE decode_mean_flush block=%u n=%u pos0=%u layers=%u\n",
                decode_mean_block_, decode_mean_n_, decode_mean_pos0_, n_ok);
    }
    decode_mean_stats_.n_flush++;
    decode_mean_stats_.last_block = decode_mean_block_;
    decode_mean_stats_.last_n = decode_mean_n_;
    decode_mean_stats_.last_layers = n_ok;
    decode_mean_stats_.last_rms = rms;
    decode_mean_n_ = 0;
    decode_mean_block_ = ~0u;
    decode_mean_target_ = {};
    decode_mean_src_.assign(n_layer_, 0);
    for (auto & h : decode_mean_host_) {
        std::fill(h.begin(), h.end(), 0.0f);
    }
} catch (...) {
    capture_fail();
    throw;
}

void llama_memory_kvmem::write_block_to_gpu(uint32_t block_id) {
    auto & store = runtime_->store();
    if (block_id >= store.block_count()) {
        return;
    }
    auto transfer = runtime_->historical_shadow_directory().acquire_transfer(block_id);
    if (!transfer) {
        throw std::runtime_error("KVMem historical block is unavailable for stage-in");
    }
    const kvmem::KvMemBlock & blk = store.blocks()[block_id];
    if (!transfer.storage && blk.orig_pos_start != uint64_t(block_id) * block_tokens_)
        throw std::logic_error("non-aligned stage-in requires a historical payload identity");
    if (blk.gpu_slot < 0 || (!transfer.storage && !raw_->has_block(block_id))) {
        return;
    }
    ++attention_epoch_;
    occupy_block_cells(block_id);
    const uint32_t nt = blk.n_tokens;
    const size_t krow = ggml_row_size(type_k_, n_embd_k_);
    const size_t vrow = ggml_row_size(type_v_, n_embd_v_);
    std::vector<uint8_t> kpack(static_cast<size_t>(nt) * krow);
    std::vector<uint8_t> vpack(static_cast<size_t>(nt) * vrow);
    const uint32_t cell0 = static_cast<uint32_t>(blk.gpu_slot) * block_tokens_;
    if (!multi_gpu_) {
        kvmem_stagein_gpu_ready((size_t) block_tokens_ * std::max(n_embd_k_, n_embd_v_),
                                std::max(krow, vrow) * (size_t) block_tokens_);
    }
    int64_t * sacc = retr_.enabled ? &retr_.set_us : nullptr;
    auto write_packed = [&](ggml_tensor * t, uint8_t * base, const uint8_t * host,
                            size_t row, size_t nbytes) {
        if (!t || !host || nbytes == 0 || cell0 >= kv_size_) {
            return;
        }
        if (multi_gpu_) {
            kvmem_tensor_set(t, host, static_cast<size_t>(cell0) * row, nbytes);
            return;
        }
        uint8_t * dst = base ? base + static_cast<size_t>(cell0) * row : nullptr;
        // Until batched stage-in owns its leases, managed blocks use the
        // synchronous H2D path. The transfer lease covers every layer through
        // device copy completion, including exceptions and fallback paths.
        if (!transfer.lease && dst && kvmem_stagein_enqueue_v(host, nbytes, dst, sacc)) {
            return;
        }
        const int64_t t_set = ggml_time_us();
        if (!(dst && kvmem_stagein_h2d_bytes(dst, host, nbytes))) {
            kvmem_tensor_set(t, host, static_cast<size_t>(cell0) * row, nbytes);
        }
        if (retr_.enabled) {
            retr_.set_us += ggml_time_us() - t_set;
        }
    };
    for (uint32_t il = 0; il < n_layer_; ++il) {
        if (!kvmem_cache_has_layer(kv_, static_cast<int32_t>(il))) {
            continue;
        }
        const int64_t t_copy = ggml_time_us();
        bool have_k = false, have_v = false;
        if (transfer.storage) {
            auto layer = transfer.storage->read_layer(transfer.lease, transfer.lease.handle(), il);
            if (!layer || layer->packed_k.size() != kpack.size() ||
                layer->packed_v.size() != vpack.size())
                throw std::runtime_error("historical stage-in payload does not match the admitted block");
            kpack = std::move(layer->packed_k);
            vpack = std::move(layer->packed_v);
            have_k = true;
            have_v = !v_trans_;
        } else {
            have_k = raw_->copy_k_gpu(block_id, il, kpack.data(), nt);
            have_v = !v_trans_ && raw_->copy_v_gpu(block_id, il, vpack.data(), nt);
        }
        if (retr_.enabled) {
            retr_.copy_us += ggml_time_us() - t_copy;
        }
        if (!have_k && !have_v) {
            continue;
        }
        ggml_tensor * kt = kv_->get_k_storage(static_cast<int32_t>(il));
        ggml_tensor * vt = kv_->get_v_storage(static_cast<int32_t>(il));
        uint8_t * kbase = multi_gpu_ ? nullptr : kvmem_cuda_tensor_ptr(kt);
        uint8_t * vbase = multi_gpu_ ? nullptr : (vt && !v_trans_) ? kvmem_cuda_tensor_ptr(vt) : nullptr;
        if (have_k) {
            write_packed(kt, kbase, kpack.data(), krow, krow * (size_t) nt);
        }
        if (have_v) {
            write_packed(vt, vbase, vpack.data(), vrow, vrow * (size_t) nt);
        }
    }
    if (trace_) {
        kvmem_diag("KVMEM_TRACE stage_in_packed block=%u slot=%d n=%u orig=%u\n",
                block_id, blk.gpu_slot, nt, blk.orig_pos_start);
    }
}

void llama_memory_kvmem::copy_gpu_block_to_host(uint32_t block_id, int32_t gpu_slot,
                                                void * host, uint64_t bytes) {
    if (!host || !kv_ || gpu_slot < 0 || bytes == 0) {
        return;
    }
    (void) block_id;
    auto * dst = static_cast<uint8_t *>(host);
    uint64_t off = 0;
    const size_t krow = ggml_row_size(type_k_, n_embd_k_);
    const size_t vrow = ggml_row_size(type_v_, n_embd_v_);
    const uint32_t cell0 = static_cast<uint32_t>(gpu_slot) * block_tokens_;
    const uint32_t nget = (cell0 < kv_size_)
            ? std::min(block_tokens_, kv_size_ - cell0) : 0;
    const uint64_t kspan = static_cast<uint64_t>(block_tokens_) * krow;
    const uint64_t vspan = static_cast<uint64_t>(block_tokens_) * vrow;
    for (uint32_t il = 0; il < n_layer_; ++il) {
        if (!kvmem_cache_has_layer(kv_, static_cast<int32_t>(il))) {
            continue;
        }
        ggml_tensor * kt = kv_->get_k_storage(static_cast<int32_t>(il));
        ggml_tensor * vt = kv_->get_v_storage(static_cast<int32_t>(il));
        if (off + kspan > bytes) {
            return;
        }
        if (kt && nget > 0) {
            kvmem_tensor_get(kt, dst + off, cell0 * krow, nget * krow);
        }
        off += kspan;
        if (off + vspan > bytes) {
            return;
        }
        if (vt && !v_trans_ && nget > 0) {
            kvmem_tensor_get(vt, dst + off, cell0 * vrow, nget * vrow);
        }
        off += vspan;
    }
}

void llama_memory_kvmem::copy_gpu_block_from_host(uint32_t block_id, int32_t gpu_slot,
                                                  const void * host, uint64_t bytes) {
    if (!host || !kv_ || gpu_slot < 0 || bytes == 0) {
        return;
    }
    (void) block_id;
    const auto * src = static_cast<const uint8_t *>(host);
    uint64_t off = 0;
    const size_t krow = ggml_row_size(type_k_, n_embd_k_);
    const size_t vrow = ggml_row_size(type_v_, n_embd_v_);
    const uint32_t cell0 = static_cast<uint32_t>(gpu_slot) * block_tokens_;
    const uint32_t nset = (cell0 < kv_size_)
            ? std::min(block_tokens_, kv_size_ - cell0) : 0;
    const uint64_t kspan = static_cast<uint64_t>(block_tokens_) * krow;
    const uint64_t vspan = static_cast<uint64_t>(block_tokens_) * vrow;
    for (uint32_t il = 0; il < n_layer_; ++il) {
        if (!kvmem_cache_has_layer(kv_, static_cast<int32_t>(il))) {
            continue;
        }
        ggml_tensor * kt = kv_->get_k_storage(static_cast<int32_t>(il));
        ggml_tensor * vt = kv_->get_v_storage(static_cast<int32_t>(il));
        if (off + kspan > bytes) {
            return;
        }
        if (kt && nset > 0) {
            kvmem_tensor_set(kt, src + off, cell0 * krow, nset * krow);
        }
        off += kspan;
        if (off + vspan > bytes) {
            return;
        }
        if (vt && !v_trans_ && nset > 0) {
            kvmem_tensor_set(vt, src + off, cell0 * vrow, nset * vrow);
        }
        off += vspan;
    }
}

void llama_memory_kvmem::score_retrieval() {
    auto & store = runtime_->store();
    const uint32_t nblk = store.block_count();
    std::vector<double> scores(nblk, 0.0);
    std::vector<float> mk(n_embd_k_, 0.0f);
    const uint32_t g = n_head_kv_ == 0 ? 1 : n_head_ / n_head_kv_;
    for (uint32_t b = 0; b < nblk; ++b) {
        auto read = runtime_->historical_shadow_directory().acquire_read(b);
        if (!read) {
            continue;
        }
        if (!read.storage && store.blocks()[b].orig_pos_start != uint64_t(b) * block_tokens_)
            throw std::logic_error("non-aligned scoring requires a historical payload identity");
        if (!read.storage && !raw_->has_block(b)) {
            continue;
        }
        double acc = 0;
        uint32_t nlay = 0;
        for (uint32_t il = 0; il < n_layer_; ++il) {
            if (q_count_[il] == 0) {
                continue;
            }
            if (read.storage) {
                if (!read.storage->copy_mean_k(read.lease, read.lease.handle(), il, mk.data(), mk.size()))
                    throw std::runtime_error("historical scoring payload has no complete Mean-K");
            } else {
                raw_->mean_k(b, il, mk.data());
            }
            double layer = 0;
            const float invq = 1.0f / static_cast<float>(q_count_[il] * std::max(1u, g));
            for (uint32_t h = 0; h < n_head_kv_; ++h) {
                std::vector<float> qh(n_embd_head_, 0.0f);
                for (uint32_t gi = 0; gi < g; ++gi) {
                    const uint32_t qh_i = h * g + gi;
                    if (qh_i >= n_head_) {
                        break;
                    }
                    const float * q = q_sum_[il].data() + qh_i * n_embd_head_;
                    for (uint32_t d = 0; d < n_embd_head_; ++d) {
                        qh[d] += q[d];
                    }
                }
                const float * kh = mk.data() + h * n_embd_head_;
                float dot = 0, nq = 0, nk = 0;
                for (uint32_t d = 0; d < n_embd_head_; ++d) {
                    const float qv = qh[d] * invq;
                    dot += qv * kh[d];
                    nq += qv * qv;
                    nk += kh[d] * kh[d];
                }
                layer += dot / (std::sqrt(nq) * std::sqrt(nk) + 1e-6f);
            }
            acc += layer / static_cast<double>(std::max(1u, n_head_kv_));
            nlay++;
        }
        scores[b] = nlay ? acc / nlay : 0;
    }
    store.set_retrieval_scores(scores);
}

void llama_memory_kvmem::retr_perf_print() {
    if (!retr_.enabled) {
        return;
    }
    fprintf(stderr, "KVMEM_RETR_SUM total_ms=%.3f flush_ms=%.3f score_ms=%.3f plan_ms=%.3f "
            "stage_out_ms=%.3f stage_out_gpu_ms=%.3f stage_out_host_ms=%.3f "
            "admit_ms=%.3f seq_rm_ms=%.3f occupy_ms=%.3f "
            "layout_d2h_ms=%.3f layout_h2d_ms=%.3f "
            "copy_ms=%.3f rope_ms=%.3f hadamard_ms=%.3f set_ms=%.3f mtp_ms=%.3f dump_ms=%.3f "
            "n_move=%u n_raw=%u n_skip=%u n_stage_in=%u laid_out=%d\n",
            retr_.total_us / 1000.0, retr_.flush_us / 1000.0, retr_.score_us / 1000.0,
            retr_.plan_us / 1000.0, retr_.stage_out_us / 1000.0,
            retr_.stage_out_gpu_us / 1000.0, retr_.stage_out_host_us / 1000.0,
            retr_.admit_us / 1000.0, retr_.seq_rm_us / 1000.0, retr_.occupy_us / 1000.0,
            retr_.layout_d2h_us / 1000.0, retr_.layout_h2d_us / 1000.0,
            retr_.copy_us / 1000.0, retr_.rope_us / 1000.0, retr_.hadamard_us / 1000.0,
            retr_.set_us / 1000.0, retr_.mtp_us / 1000.0, retr_.dump_us / 1000.0,
            retr_.n_move, retr_.n_raw, retr_.n_skip, retr_.n_stage_in, retr_.laid_out);
}

void llama_memory_kvmem::set_turn_spans(const llama_kvmem_turn_spans & spans) {
    for (const auto & ranges : {spans.query, spans.mandatory}) {
        for (const auto & r : ranges) {
            if (r.begin < 0 || r.end < r.begin) throw std::invalid_argument("invalid KVMem row range");
        }
    }
    if (spans.replay_begin < 0) throw std::invalid_argument("invalid KVMem replay start");
    harvest_flush();
    turn_spans_ = spans;
    explicit_spans_ = true;
}

bool llama_memory_kvmem::query_contains(llama_pos row) const {
    if (query_frozen_) return false;
    if (!explicit_spans_) return query_begin_ >= 0 && row >= query_begin_ && (query_end_ <= 0 || row < query_end_);
    for (const auto & r : turn_spans_.query) if (row >= r.begin && row < r.end) return true;
    return false;
}

bool llama_memory_kvmem::query_overlaps(uint32_t n, const llama_pos * rows) const {
    if (!want_q_capture() || !n) return false;
    if (!rows) return true;
    for (uint32_t i = 0; i < n; ++i) if (query_contains(rows[i])) return true;
    return false;
}

bool llama_memory_kvmem::gpu_kv_complete(uint32_t id, const llama_kv_cache * cache) const {
    if (!cache || id >= store().block_count()) return false;
    const auto & b = store().blocks()[id];
    if (b.gpu_slot < 0 || b.tier != kvmem::KvTier::GPU || b.in_flight || !b.n_tokens) return false;
    const auto & cells = cache->get_cells(0);
    const uint64_t first = (uint64_t) b.gpu_slot * block_tokens_;
    if (first + b.n_tokens > cells.size() || b.orig_pos_end() > row_positions_.size()) return false;
    for (uint32_t i = 0; i < b.n_tokens; ++i) {
        const uint32_t cell = first + i, row = b.orig_pos_start + i;
        if (cells.is_empty(cell) || !cells.seq_has(cell, 0)) return false;
        const auto & ext = cells.ext_get(cell);
        const auto & pos = row_positions_[row].pos;
        if (ext.logical_pos != (llama_pos) row || cells.pos_get(cell) != pos[0] ||
                (row_positions_[row].spatial && (ext.y != pos[1] || ext.x != pos[2]))) return false;
    }
    return true;
}

llama_kvmem_attention_view llama_memory_kvmem::attention_view(bool canonical) const {
    llama_kvmem_attention_view view;
    view.epoch = attention_epoch_;
    view.rows = store_n_tokens();
    view.valid = kv_ && runtime_ && store().config().optimize_stage_in;
    uint32_t resident_rows = 0;
    for (const auto & b : store().blocks()) {
        if (b.gpu_slot < 0 || !b.n_tokens) continue;
        resident_rows += b.n_tokens;
        if ((canonical && b.gpu_slot != (int32_t) view.blocks.size()) || !gpu_kv_complete(b.block_id, kv_) ||
                (mtp_ && !gpu_kv_complete(b.block_id, mtp_->get_kv()))) view.valid = false;
        view.blocks.push_back(b.block_id);
    }
    // Extra live cells (for example an untrimmed speculative tail) are also
    // part of attention. A proof must account for them, not just known blocks.
    if (kv_ && kv_->get_cells(0).get_used() != resident_rows) view.valid = false;
    if (mtp_ && mtp_->get_kv()->get_cells(0).get_used() != resident_rows) view.valid = false;
    return view;
}

bool llama_memory_kvmem::can_append(uint32_t end, uint32_t generation_rows, bool all_history, std::string & reason) const {
    auto fail = [&](const char * why) { reason = why; return false; };
    if (!runtime_ || !kv_ || end < store_n_tokens()) return fail("invalid_boundary");
    // These fast-path proofs currently reserve aligned legacy slots only.
    // A managed tail can roll over at every future append boundary; until the
    // request reservation owns that bound, do not grant a false fit proof.
    if (runtime_->historical_shadow_directory().size()) return fail("historical_reservation_required");
    const auto view = attention_view(all_history);
    if (!view.valid) return fail("coverage_or_layout");
    const auto & s = store();
    const uint64_t end_blocks = ((uint64_t) end + block_tokens_ - 1) / block_tokens_;
    if (view.rows % block_tokens_ && !gpu_kv_complete(view.rows / block_tokens_, kv_)) return fail("partial_tail_missing");
    if (all_history) {
        if (view.blocks.size() != s.block_count()) return fail("history_not_resident");
        if (s.budget_blocks() && end_blocks > s.budget_blocks()) return fail("selection_budget");
        if (s.prefill_needs_offload(resident_tokens(), end - view.rows, kv_size_)) return fail("prefill_pressure");
    }
    const uint64_t final_blocks = ((uint64_t) end + generation_rows + block_tokens_ - 1) / block_tokens_;
    if (final_blocks > n_slots_ + (uint64_t) s.block_count()) return fail("insufficient_slots");
    uint64_t needed = 0;
    for (uint64_t id = view.rows / block_tokens_; id < final_blocks; ++id) {
        if (id >= s.block_count() || s.blocks()[id].gpu_slot < 0) ++needed;
    }
    if (needed > free_slots_.size()) return fail("insufficient_slots");
    reason = all_history ? "all_resident" : "same_query";
    return true;
}

std::vector<uint32_t> llama_memory_kvmem::retrieval_mandatory() const {
    const auto & s = store();
    std::vector<llama_kvmem_row_range> ranges;
    if (explicit_spans_) ranges = turn_spans_.mandatory;
    else if (query_begin_ >= 0) ranges.push_back({query_begin_, query_end_ > 0 ? query_end_ : (int32_t) s.total_tokens()});
    std::vector<uint32_t> mandatory;
    for (const auto & b : s.blocks()) {
        for (const auto & r : ranges) {
            if ((int64_t) b.orig_pos_end() > r.begin && (int64_t) b.orig_pos_start < r.end) {
                mandatory.push_back(b.block_id);
                break;
            }
        }
    }
    if (force_pos_ >= 0) {
        const int32_t id = s.block_id_containing(force_pos_);
        if (id >= 0) mandatory.push_back(id);
    }
    return mandatory;
}

llama_kvmem_selection llama_memory_kvmem::preview_retrieval() {
    const int64_t t0 = ggml_time_us();
    const bool enabled = retr_.enabled;
    retr_ = RetrPerf{};
    retr_.enabled = enabled;
    harvest_flush();
    const int64_t t1 = ggml_time_us();
    if (method_ == 1) score_retrieval();
    const int64_t t2 = ggml_time_us();
    llama_kvmem_selection selection;
    selection.epoch = attention_epoch_;
    selection.rows = store_n_tokens();
    selection.blocks = runtime_->preview_reselect(retrieval_mandatory());
    if (enabled) {
        retr_.flush_us = t1 - t0;
        retr_.score_us = t2 - t1;
        retr_.plan_us = ggml_time_us() - t2;
        retr_.total_us = ggml_time_us() - t0;
    }
    return selection;
}

bool llama_memory_kvmem::selection_fits(const llama_kvmem_selection & selection, uint32_t end, uint32_t generation_rows) const {
    if (runtime_->historical_shadow_directory().size()) return false;
    if (selection.epoch != attention_epoch_ || selection.rows != store_n_tokens() || end < selection.rows) return false;
    // The selector can trim mandatory ranges to its budget. Such a plan is
    // unsuitable for continuation even if it fits the physical slot pool.
    for (uint32_t id : retrieval_mandatory()) {
        if (!std::binary_search(selection.blocks.begin(), selection.blocks.end(), id)) return false;
    }
    const uint64_t final = ((uint64_t) end + generation_rows + block_tokens_ - 1) / block_tokens_;
    const uint64_t first = selection.rows / block_tokens_;
    if (final < first || final - first > n_slots_) return false;
    uint64_t count = selection.blocks.size();
    for (uint64_t id = first; id < final; ++id) {
        if (!std::binary_search(selection.blocks.begin(), selection.blocks.end(), (uint32_t) id)) ++count;
    }
    return count <= n_slots_;
}

bool llama_memory_kvmem::commit_unchanged(const llama_kvmem_attention_view & before,
                                        const llama_kvmem_selection & selection) {
    if (!before.valid || before.epoch != attention_epoch_ || before.rows > store_n_tokens() ||
            selection.epoch != attention_epoch_ || selection.rows != store_n_tokens()) return false;
    const auto after = attention_view();
    if (!after.valid || after.blocks != selection.blocks) return false;
    for (uint32_t id : before.blocks) {
        if (!std::binary_search(after.blocks.begin(), after.blocks.end(), id)) return false;
    }
    return runtime_->commit_resident_selection(selection.blocks);
}

bool llama_memory_kvmem::commit_resident(bool canonical) {
    harvest_flush();
    const auto view = attention_view(canonical);
    return view.valid && runtime_->commit_resident_selection(view.blocks);
}

bool llama_memory_kvmem::get_query(llama_kvmem_query_state & state) {
    harvest_flush();
    bool any = false;
    uint32_t rows = 0;
    for (uint32_t il = 0; il < n_layer_; ++il) {
        if (!kvmem_cache_has_layer(kv_, il)) continue;
        if (!q_count_[il] || (rows && rows != q_count_[il])) return false;
        rows = q_count_[il];
        any = true;
    }
    if (!any) return false;
    state.sum = q_sum_;
    state.count = q_count_;
    return true;
}

bool llama_memory_kvmem::set_query(const llama_kvmem_query_state & state) {
    if (state.sum.size() != n_layer_ || state.count.size() != n_layer_) return false;
    uint32_t rows = 0;
    for (uint32_t il = 0; il < n_layer_; ++il) {
        const auto & v = state.sum[il];
        if (v.size() != n_head_ * n_embd_head_ ||
                !std::all_of(v.begin(), v.end(), [](float x) { return std::isfinite(x); })) return false;
        if (!kvmem_cache_has_layer(kv_, il)) continue;
        if (!state.count[il] || (rows && rows != state.count[il])) return false;
        rows = state.count[il];
    }
    if (!rows) return false;
    harvest_flush();
    q_sum_ = state.sum;
    q_count_ = state.count;
    query_frozen_ = true;
    return true;
}

void llama_memory_kvmem::apply_retrieval() {
    if (method_ != 1) { harvest_flush(); return; }
    apply_selection(preview_retrieval());
}

void llama_memory_kvmem::apply_selection(const llama_kvmem_selection & selection) {
    if (selection.epoch != attention_epoch_ || selection.rows != store_n_tokens()) {
        throw std::runtime_error("stale KVMem selection");
    }
    const int64_t t_all = ggml_time_us();
    kvmem::KvMemPlan plan;
    {
        const int64_t t0 = ggml_time_us();
        plan = runtime_->prepare_selection(selection.blocks);
        if (retr_.enabled) {
            retr_.plan_us += ggml_time_us() - t0;
            retr_.n_stage_in = static_cast<uint32_t>(plan.stage_in.size());
            for (const auto & r : plan.remaps) {
                if (r.skip) {
                    retr_.n_skip++;
                }
            }
        }
    }
    trace_plan("retrieval", plan);
    if (trace_) {
        kvmem_diag("KVMEM_TRACE selected");
        for (const auto & r : plan.remaps) {
            fprintf(stderr, " %u", r.block_id);
        }
        fprintf(stderr, "\n");
    }
    apply_plan_to_kv(plan);
    auto & store = runtime_->store();
    retrieval_pinned_ = true;
    if (mtp_) {
        const int64_t t0 = ggml_time_us();
        mtp_->harvest_resident_v();
        if (retr_.enabled) {
            retr_.mtp_us += ggml_time_us() - t0;
        }
    }
    // Native GPU KV of a reused block at orig_pos != 0 is the ground truth for
    // host RoPE + capture layout. Prefer a block that is NOT in stage_in so the
    // slot still holds prefill bytes. Must run before layout rewrite.
    if (trace_) {
        const int64_t t_dump = ggml_time_us();
        std::vector<bool> is_stage_in(store.block_count(), false);
        for (uint32_t id : plan.stage_in) {
            if (id < is_stage_in.size()) {
                is_stage_in[id] = true;
            }
        }
        for (const auto & r : plan.remaps) {
            if (r.block_id >= store.block_count() || is_stage_in[r.block_id]) {
                continue;
            }
            const kvmem::KvMemBlock & b = store.blocks()[r.block_id];
            if (b.gpu_slot < 0 || b.orig_pos_start == 0 || !raw_->has_block(r.block_id)) {
                continue;
            }
            kvmem_diag("KVMEM_KV --- native reused block %u orig=%u skip=%d (GPU vs packed) ---\n",
                    r.block_id, b.orig_pos_start, (int) r.skip);
            dump_kv_compare(static_cast<int32_t>(r.block_id), false);
            break;
        }
        if (retr_.enabled) {
            retr_.dump_us += ggml_time_us() - t_dump;
        }
    }
    const bool laid_out = layout_gpu_slots_by_orig_pos();
    uint32_t n_raw = 0;
    uint32_t n_skip = 0;
    if (!laid_out) {
        for (uint32_t id : plan.stage_in) {
            if (id >= store.block_count() || store.blocks()[id].gpu_slot < 0) {
                continue;
            }
            if (gpu_kv_already_resident(id)) {
                n_skip++;
                continue;
            }
            write_block_to_gpu(id);
            n_raw++;
        }
        kvmem_stagein_flush_sync(retr_.enabled ? &retr_.copy_us : nullptr,
                                 retr_.enabled ? &retr_.rope_us : nullptr,
                                 retr_.enabled ? &retr_.hadamard_us : nullptr,
                                 retr_.enabled ? &retr_.set_us : nullptr);
        if (retr_.enabled) {
            retr_.n_raw += n_raw;
            retr_.n_skip += n_skip;
            retr_.laid_out = 0;
        }
    }
    if (trace_) {
        kvmem_diag("KVMEM_TRACE writeback laid_out=%d raw=%u skip_resident=%u stage_in=%zu\n",
                (int) laid_out, n_raw, n_skip, plan.stage_in.size());
    }
    if (trace_) {
        const int64_t t_dump = ggml_time_us();
        for (uint32_t id : plan.stage_in) {
            if (id < store.block_count() && store.blocks()[id].gpu_slot >= 0) {
                kvmem_diag("KVMEM_KV --- after stage_in_packed block %u ---\n", id);
                dump_kv_compare(static_cast<int32_t>(id), false);
                break;
            }
        }
        trace_working_set("after_retrieval");
        if (retr_.enabled) {
            retr_.dump_us += ggml_time_us() - t_dump;
        }
    }
    if (mtp_) {
        const int64_t t0 = ggml_time_us();
        mtp_->follow_retrieval();
        if (retr_.enabled) {
            retr_.mtp_us += ggml_time_us() - t0;
        }
    }
    if (retr_.enabled) {
        retr_.total_us += ggml_time_us() - t_all;
        retr_perf_print();
    }
}

void llama_memory_kvmem::trace_working_set(const char * tag) const {
    if (!kv_ || !runtime_) {
        return;
    }
    const llama_kv_cells & cells = kv_->get_cells(0);
    kvmem_diag("KVMEM_TRACE %s cells used=%u used_max_p1=%u size=%u seq_pos=[%d,%d]\n",
            tag,
            cells.get_used(), cells.used_max_p1(), cells.size(),
            kv_->seq_pos_min(0), kv_->seq_pos_max(0));
    if (recr_) {
        kvmem_diag("KVMEM_TRACE %s recr_seq_pos=[%d,%d]\n",
                tag, recr_->seq_pos_min(0), recr_->seq_pos_max(0));
    }
    const auto & store = runtime_->store();
    for (const auto & b : store.blocks()) {
        if (b.gpu_slot < 0 || b.n_tokens == 0) {
            continue;
        }
        const uint32_t idx = static_cast<uint32_t>(b.gpu_slot) * block_tokens_;
        const bool empty = idx >= cells.size() || cells.is_empty(idx);
        llama_pos p_last = -1;
        bool last_empty = true;
        if (!empty && b.n_tokens > 0) {
            const uint32_t last = idx + b.n_tokens - 1;
            last_empty = last >= cells.size() || cells.is_empty(last);
            p_last = last_empty ? -1 : cells.pos_get(last);
        }
        kvmem_diag("KVMEM_TRACE occupy tag=%s block=%u slot=%d cell=%u empty=%d pos=%d last_empty=%d last_pos=%d n=%u orig=%u\n",
                tag, b.block_id, b.gpu_slot, idx, (int) empty,
                empty ? -1 : (int) cells.pos_get(idx),
                (int) last_empty, (int) p_last, b.n_tokens, b.orig_pos_start);
    }
}

void llama_memory_kvmem::kv_stats(const char * tag, const float * a, const float * b, size_t n) {
    if (!a || !b || n == 0) {
        kvmem_diag("KVMEM_KV %s missing n=%zu\n", tag, n);
        return;
    }
    double dot = 0, na = 0, nb = 0, se = 0, maxabs = 0;
    for (size_t i = 0; i < n; ++i) {
        const double da = a[i];
        const double db = b[i];
        dot += da * db;
        na += da * da;
        nb += db * db;
        const double e = da - db;
        se += e * e;
        maxabs = std::max(maxabs, std::abs(e));
    }
    const double cos = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
    const double rmse = std::sqrt(se / static_cast<double>(n));
    const double rms_a = std::sqrt(na / static_cast<double>(n));
    kvmem_diag("KVMEM_KV %s n=%zu cos=%.6f rmse=%.6f maxabs=%.6f rms_gpu=%.6f\n",
            tag, n, cos, rmse, maxabs, rms_a);
}

bool llama_memory_kvmem::read_gpu_block(uint32_t block_id, uint32_t il, bool is_k,
                                        std::vector<float> & out) const {
    const auto & store = runtime_->store();
    if (block_id >= store.block_count() || il >= n_layer_ ||
        !kvmem_cache_has_layer(kv_, static_cast<int32_t>(il))) {
        return false;
    }
    const kvmem::KvMemBlock & blk = store.blocks()[block_id];
    if (blk.gpu_slot < 0) {
        return false;
    }
    ggml_tensor * t = is_k ? kv_->get_k_storage(static_cast<int32_t>(il))
                           : kv_->get_v_storage(static_cast<int32_t>(il));
    if (!t) {
        return false;
    }
    const uint32_t dim = is_k ? n_embd_k_ : n_embd_v_;
    const ggml_type ty = is_k ? type_k_ : type_v_;
    const size_t row = ggml_row_size(ty, dim);
    const uint32_t nt = blk.n_tokens;
    const uint32_t cell0 = static_cast<uint32_t>(blk.gpu_slot) * block_tokens_;
    out.assign(static_cast<size_t>(nt) * dim, 0.0f);
    if (nt == 0 || cell0 >= kv_size_) {
        return false;
    }
    const uint32_t nget = std::min(nt, kv_size_ - cell0);
    std::vector<uint8_t> raw(static_cast<size_t>(nget) * row);
    kvmem_tensor_get(t, raw.data(), static_cast<size_t>(cell0) * row,
                            static_cast<size_t>(nget) * row);
    if (!kvmem_cache_unpack_rows(ty, raw.data(), out.data(),
                                 static_cast<int64_t>(nget),
                                 static_cast<int64_t>(dim))) {
        return false;
    }
    if (kvmem_attn_rot_on(ty, static_cast<int>(n_embd_head_))) {
        const int nrot = is_k
                ? kvmem_hadamard_nrot_k(static_cast<int>(n_embd_head_))
                : kvmem_hadamard_nrot_v(static_cast<int>(n_embd_head_));
        kvmem_hadamard_rows(out.data(), static_cast<int64_t>(nget),
                            static_cast<int>(n_head_kv_), static_cast<int>(n_embd_head_),
                            nrot);
    }
    return true;
}

void llama_memory_kvmem::dump_kv_compare(int32_t block_id, bool writeback_test) {
    auto & store = runtime_->store();
    if (block_id < 0 && force_pos_ >= 0) {
        block_id = store.block_id_containing(static_cast<uint32_t>(force_pos_));
    }
    if (block_id < 0) {
        block_id = 0;
    }
    const uint32_t bid = static_cast<uint32_t>(block_id);
    if (bid >= store.block_count()) {
        kvmem_diag("KVMEM_KV dump: no block %d\n", block_id);
        return;
    }
    const kvmem::KvMemBlock & blk = store.blocks()[bid];
    bool any_pk = false;
    bool any_pv = false;
    for (uint32_t il = 0; il < n_layer_; ++il) {
        any_pk = any_pk || raw_->has_k_gpu(bid, il);
        any_pv = any_pv || raw_->has_v_gpu(bid, il);
    }
    kvmem_diag("KVMEM_KV dump block=%u slot=%d n=%u orig=%u packed_k=%d packed_v=%d v_trans=%d type_k=%s pos0=%d\n",
            bid, blk.gpu_slot, blk.n_tokens, blk.orig_pos_start,
            (int) any_pk, (int) any_pv,
            (int) v_trans_, ggml_type_name(type_k_),
            static_cast<int>(blk.orig_pos_start));

    const uint32_t nt = blk.n_tokens;
    const size_t nk = static_cast<size_t>(nt) * n_embd_k_;
    const size_t nv = static_cast<size_t>(nt) * n_embd_v_;
    const size_t krow = ggml_row_size(type_k_, n_embd_k_);
    const size_t vrow = ggml_row_size(type_v_, n_embd_v_);
    const uint32_t cell0 = (blk.gpu_slot >= 0)
            ? static_cast<uint32_t>(blk.gpu_slot) * block_tokens_ : 0;
    std::vector<float> gpu_k, gpu_v, packed_f;

    auto unpack_packed = [&](ggml_type ty, const uint8_t * packed, uint32_t ntok,
                             uint32_t dim, bool is_k, std::vector<float> & out) -> bool {
        out.assign(static_cast<size_t>(ntok) * dim, 0.0f);
        if (!kvmem_cache_unpack_rows(ty, packed, out.data(),
                                     static_cast<int64_t>(ntok),
                                     static_cast<int64_t>(dim))) {
            return false;
        }
        if (kvmem_attn_rot_on(ty, static_cast<int>(n_embd_head_))) {
            const int nrot = is_k
                    ? kvmem_hadamard_nrot_k(static_cast<int>(n_embd_head_))
                    : kvmem_hadamard_nrot_v(static_cast<int>(n_embd_head_));
            kvmem_hadamard_rows(out.data(), static_cast<int64_t>(ntok),
                                static_cast<int>(n_head_kv_),
                                static_cast<int>(n_embd_head_), nrot);
        }
        return true;
    };

    const auto & attn_layers = kv_->get_layer_ids();
    if (attn_layers.empty()) return;
    const uint32_t layers_show[] = {attn_layers.front(), attn_layers[attn_layers.size() / 2], attn_layers.back()};
    for (uint32_t li = 0; li < 3; ++li) {
        const uint32_t il = layers_show[li];
        if (il >= n_layer_ || !kvmem_cache_has_layer(kv_, static_cast<int32_t>(il))) {
            continue;
        }
        if (!read_gpu_block(bid, il, true, gpu_k)) {
            kvmem_diag("KVMEM_KV L%u K gpu read fail\n", il);
            continue;
        }
        char tag[64];
        std::vector<uint8_t> kpack(static_cast<size_t>(nt) * krow);
        if (raw_->copy_k_gpu(bid, il, kpack.data(), nt)) {
            ggml_tensor * kt = kv_->get_k_storage(static_cast<int32_t>(il));
            size_t nmis = 0;
            if (kt && cell0 < kv_size_) {
                std::vector<uint8_t> kgpu(kpack.size());
                kvmem_tensor_get(kt, kgpu.data(),
                                        static_cast<size_t>(cell0) * krow,
                                        kpack.size());
                for (size_t i = 0; i < kpack.size(); ++i) {
                    nmis += (kpack[i] != kgpu[i]);
                }
            }
            kvmem_diag("KVMEM_KV L%u K packed_bytes mismatch=%zu / %zu\n",
                    il, nmis, kpack.size());
            if (unpack_packed(type_k_, kpack.data(), nt, n_embd_k_, true, packed_f)) {
                snprintf(tag, sizeof(tag), "L%u K packed_vs_gpu", il);
                kv_stats(tag, gpu_k.data(), packed_f.data(), nk);
            }
        } else {
            kvmem_diag("KVMEM_KV L%u no packed K\n", il);
        }
        if (v_trans_) {
            continue;
        }
        std::vector<uint8_t> vpack(static_cast<size_t>(nt) * vrow);
        if (raw_->copy_v_gpu(bid, il, vpack.data(), nt) &&
            read_gpu_block(bid, il, false, gpu_v) &&
            unpack_packed(type_v_, vpack.data(), nt, n_embd_v_, false, packed_f)) {
            snprintf(tag, sizeof(tag), "L%u V packed_vs_gpu", il);
            kv_stats(tag, gpu_v.data(), packed_f.data(), nv);
        }
    }

    uint32_t il_wb = 0;
    while (il_wb < n_layer_ &&
           (!kvmem_cache_has_layer(kv_, static_cast<int32_t>(il_wb)) ||
            !raw_->has_k_gpu(bid, il_wb))) {
        il_wb++;
    }
    if (writeback_test && il_wb < n_layer_ && read_gpu_block(bid, il_wb, true, gpu_k)) {
        std::vector<float> before = gpu_k;
        write_block_to_gpu(bid);
        kvmem_stagein_flush_sync(nullptr, nullptr, nullptr, nullptr);
        std::vector<float> after;
        if (read_gpu_block(bid, il_wb, true, after)) {
            kv_stats("K gpu_after_writeback_vs_gpu_before", before.data(), after.data(),
                     before.size());
        }
    }
}

void llama_kvmem_dump_kv_compare(struct llama_context * /*ctx*/, int32_t block_id) {
    if (llama_memory_kvmem * mem = kvmem_capture_active()) {
        mem->dump_kv_compare(block_id, false);
    }
}

void llama_kvmem_dump_kv_writeback(struct llama_context * /*ctx*/, int32_t block_id) {
    if (llama_memory_kvmem * mem = kvmem_capture_active()) {
        mem->dump_kv_compare(block_id, true);
    }
}

void llama_kvmem_apply_retrieval(struct llama_context * /*ctx*/) {
    if (llama_memory_kvmem * mem = kvmem_capture_active()) {
        mem->apply_retrieval();
    }
}

bool llama_memory_kvmem::query_replay_fits(uint32_t query_begin, uint32_t prompt_end) const {
    if (!runtime_ || query_begin >= prompt_end) {
        return true;
    }
    if (runtime_->historical_shadow_directory().size()) return false;
    const kvmem::KvMemStore & store = runtime_->store();
    const uint32_t bt = std::max(1u, store.config().block_tokens);
    const uint32_t budget = store.budget_blocks();
    if (budget == 0) {
        return true;
    }
    const uint64_t total_blocks =
            (static_cast<uint64_t>(prompt_end) + bt - 1) / bt;
    const uint64_t boundary_block = query_begin / bt;
    if (boundary_block >= total_blocks) {
        return true;
    }
    const uint64_t sink =
            std::min<uint64_t>(store.config().sink_blocks, total_blocks);
    const uint64_t suffix = total_blocks - boundary_block;
    const uint64_t required =
            boundary_block < sink ? total_blocks : sink + suffix;
    return required <= static_cast<uint64_t>(budget);
}

bool llama_kvmem_query_replay_fits(uint32_t query_begin, uint32_t prompt_end) {
    if (llama_memory_kvmem * mem = kvmem_capture_active()) {
        return mem->query_replay_fits(query_begin, prompt_end);
    }
    return true;
}

void llama_kvmem_set_replay(bool replay) {
    if (llama_memory_kvmem * mem = kvmem_capture_active()) {
        mem->set_replay(replay);
    }
}

void llama_kvmem_trace_cells(struct llama_context * /*ctx*/, const char * tag) {
    if (llama_memory_kvmem * mem = kvmem_capture_active()) {
        mem->trace_working_set(tag ? tag : "cells");
    }
}

bool llama_kvmem_has_recurrent(void) {
    llama_memory_kvmem * mem = kvmem_capture_active();
    return mem && mem->has_recurrent();
}

bool llama_kvmem_gdn_replay_enabled(void) {
    auto * mem = kvmem_capture_active();
    return mem && mem->gdn_replay_enabled();
}

bool llama_kvmem_gdn_replay_begin(llama_pos start, uint32_t width) {
    auto * mem = kvmem_capture_active();
    return mem && mem->gdn_replay_begin(start, width);
}

bool llama_kvmem_gdn_replay_commit(llama_context * ctx, uint32_t n_keep) {
    auto * mem = kvmem_capture_active();
    return mem && mem->gdn_replay_commit(ctx, n_keep);
}

bool llama_kvmem_want_decode_mean(void) {
    llama_memory_kvmem * mem = kvmem_capture_active();
    return mem && mem->want_decode_mean();
}

bool llama_kvmem_historical_capture_begin(llama_context * ctx, uint32_t capacity, uint64_t input_tokens,
        uint64_t max_output_tokens, uint64_t max_appends) {
    auto * mem = kvmem_capture_active();
    if (!mem || !ctx || ctx->get_memory() != mem) return false;
    const bool previously_prepared = ctx->historical_simple_input_prepared();
    try {
        if (!ctx->prepare_historical_simple_input()) return false;
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("historical capture: input reservation failed: %s\n", e.what());
        return false;
    }
    if (mem->begin_historical_capture(capacity, input_tokens, max_output_tokens, max_appends)) return true;
    if (!previously_prepared) ctx->clear_historical_simple_input();
    return false;
}

bool llama_kvmem_historical_gc_start(int mode, uint32_t ready_low, uint32_t ready_high) {
    auto * mem = kvmem_capture_active();
    return mem && mem->start_historical_gc(mode, ready_low, ready_high);
}

bool llama_kvmem_historical_capture_finish(void) {
    auto * mem = kvmem_capture_active();
    return mem && mem->finish_historical_capture();
}

void llama_kvmem_historical_capture_fail(llama_context * ctx) {
    auto * mem = kvmem_capture_active();
    if (mem && ctx && ctx->get_memory() == mem) mem->fail_historical_capture();
}

void llama_kvmem_decode_mean_commit(uint32_t n_keep) {
    if (llama_memory_kvmem * mem = kvmem_capture_active()) {
        mem->decode_mean_commit(n_keep);
    }
}

void llama_kvmem_decode_mean_discard(void) {
    if (llama_memory_kvmem * mem = kvmem_capture_active()) {
        mem->decode_mean_discard();
    }
}

void llama_kvmem_decode_mean_flush(void) {
    if (llama_memory_kvmem * mem = kvmem_capture_active()) {
        mem->decode_mean_flush();
    }
}

bool llama_kvmem_want_prefill_capture(void) {
    const llama_kvmem_params * kp = llama_kvmem_get_params();
    if (!kp || !kp->enabled) {
        return false;
    }
    if (kp->method != 1 && !kp->harvest_v && getenv("KVMEM_DUMP_CAPTURE") == nullptr) {
        return false;
    }
    llama_memory_kvmem * mem = kvmem_capture_active();
    if (!mem) {
        return false;
    }
    return mem->want_prefill_capture();
}

bool llama_kvmem_want_q_capture(uint32_t n_tokens, uint32_t n_pos, const llama_pos * pos) {
    llama_memory_kvmem * mem = kvmem_capture_active();
    if (!mem || !mem->want_q_capture()) {
        return false;
    }
    return llama_kvmem_ubatch_needs_q_capture(n_tokens, n_pos, pos);
}

void llama_kvmem_reset_query(void) {
    if (llama_memory_kvmem * mem = kvmem_capture_active()) {
        mem->reset_query_acc();
    }
}

void llama_kvmem_end_prefill_capture(void) {
    if (llama_memory_kvmem * mem = kvmem_capture_active()) {
        mem->end_prefill_capture();
    }
}

void llama_kvmem_begin_cached_turn(void) {
    if (llama_memory_kvmem * mem = kvmem_capture_active()) {
        mem->begin_cached_turn(true);
    }
}

void llama_kvmem_begin_cached_turn_keep_query(void) {
    if (llama_memory_kvmem * mem = kvmem_capture_active()) {
        mem->begin_cached_turn(false);
    }
}

void llama_kvmem_keep_selected(void) {
    if (llama_memory_kvmem * mem = kvmem_capture_active()) {
        mem->keep_selected_window();
    }
}

void llama_kvmem_pin_working_set(void) {
    if (llama_memory_kvmem * mem = kvmem_capture_active()) {
        mem->pin_working_set();
    }
}

uint32_t llama_kvmem_free_slots(void) {
    if (llama_memory_kvmem * mem = kvmem_capture_active()) {
        return (uint32_t) mem->free_slot_count();
    }
    return 0;
}

uint32_t llama_kvmem_store_n_tokens(void) {
    if (llama_memory_kvmem * mem = kvmem_capture_active()) {
        return mem->store_n_tokens();
    }
    return 0;
}

void llama_kvmem_truncate_cached(uint32_t n_past) {
    if (llama_memory_kvmem * mem = kvmem_capture_active()) {
        mem->truncate_cached(n_past);
    }
}

llama_pos llama_kvmem_recr_pos_max(void) {
    if (llama_memory_kvmem * mem = kvmem_capture_active()) {
        return mem->recr_pos_max();
    }
    return -1;
}

void llama_kvmem_set_request_span(int32_t query_begin, int32_t query_end, int32_t force_pos) {
    kvmem_current_execution().params.query_begin = query_begin;
    kvmem_current_execution().params.query_end = query_end;
    kvmem_current_execution().params.force_pos = force_pos;
    if (llama_memory_kvmem * mem = kvmem_capture_active()) {
        mem->set_query_span(query_begin, query_end);
        mem->set_force_pos(force_pos);
    }
}

void llama_kvmem_set_turn_spans(const llama_kvmem_turn_spans & spans) {
    if (auto * mem = kvmem_capture_active()) mem->set_turn_spans(spans);
}

llama_kvmem_attention_view llama_kvmem_get_attention_view() {
    if (auto * mem = kvmem_capture_active()) return mem->attention_view();
    return {};
}

bool llama_kvmem_can_append(uint32_t end, uint32_t generation_rows, bool all_history, std::string & reason) {
    if (auto * mem = kvmem_capture_active()) return mem->can_append(end, generation_rows, all_history, reason);
    reason = "no_kvmem";
    return false;
}

llama_kvmem_selection llama_kvmem_preview_retrieval() {
    if (auto * mem = kvmem_capture_active()) return mem->preview_retrieval();
    return {};
}

bool llama_kvmem_selection_fits(const llama_kvmem_selection & selection, uint32_t end, uint32_t generation_rows) {
    if (auto * mem = kvmem_capture_active()) return mem->selection_fits(selection, end, generation_rows);
    return false;
}

bool llama_kvmem_commit_unchanged(const llama_kvmem_attention_view & view, const llama_kvmem_selection & selection) {
    if (auto * mem = kvmem_capture_active()) return mem->commit_unchanged(view, selection);
    return false;
}

void llama_kvmem_apply_selection(const llama_kvmem_selection & selection) {
    if (auto * mem = kvmem_capture_active()) mem->apply_selection(selection);
}

bool llama_kvmem_commit_resident(bool canonical) {
    if (auto * mem = kvmem_capture_active()) return mem->commit_resident(canonical);
    return false;
}

bool llama_kvmem_get_query(llama_kvmem_query_state & state) {
    if (auto * mem = kvmem_capture_active()) return mem->get_query(state);
    return false;
}

bool llama_kvmem_set_query(const llama_kvmem_query_state & state) {
    if (auto * mem = kvmem_capture_active()) return mem->set_query(state);
    return false;
}

void llama_kvmem_freeze_query(bool frozen) {
    if (auto * mem = kvmem_capture_active()) mem->freeze_query(frozen);
}

llama_pos llama_kvmem_model_pos(uint32_t logical) {
    auto * mem = kvmem_capture_active();
    return mem ? mem->model_pos(logical) : (llama_pos) logical;
}

bool llama_kvmem_remove_logical(llama_context * ctx, llama_pos begin, llama_pos end) {
    auto * mem = kvmem_capture_active();
    if (mem) return mem->remove_logical(ctx, begin, end);
    auto * native = llama_get_memory(ctx);
    auto * hybrid = dynamic_cast<llama_memory_hybrid *>(native);
    auto * kv = hybrid ? hybrid->get_mem_attn() : dynamic_cast<llama_kv_cache *>(native);
    if (!kv) return llama_memory_seq_rm(native, 0, begin, end);
    if (hybrid && end < 0) {
        const auto & cells = kv->get_cells(0);
        llama_pos first = INT32_MAX;
        llama_pos model_begin = -1;
        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (cells.is_empty(i)) continue;
            const auto row = cells.ext_get(i).logical_pos;
            if (row >= begin && row < first) { first = row; model_begin = cells.pos_get(i); }
        }
        auto * recurrent = hybrid->get_mem_recr();
        if (model_begin >= 0 && recurrent->seq_pos_max(0) >= model_begin
                && !recurrent->seq_rm(0, model_begin, -1)) return false;
    }
    return kv->seq_rm_logical(0, begin, end);
}

void llama_kvmem_set_media_ranges(const uint32_t * starts, const uint32_t * ends, size_t count) {
    if (auto * mem = kvmem_capture_active()) {
        std::vector<std::pair<uint32_t, uint32_t>> ranges;
        for (size_t i = 0; i < count; ++i) ranges.emplace_back(starts[i], ends[i]);
        mem->runtime().store().set_media_ranges(std::move(ranges));
    }
}

void llama_kvmem_get_tail_mean(uint32_t row, std::vector<float> & state) {
    if (auto * mem = kvmem_capture_active()) {
        mem->harvest_flush();
        mem->decode_mean_flush();
        state = mem->raw().mean_checkpoint(row);
    }
}

void llama_kvmem_set_tail_mean(uint32_t row, const std::vector<float> & state) {
    if (auto * mem = kvmem_capture_active()) mem->raw().restore_mean_checkpoint(row, state);
}

bool llama_kvmem_store_swap_supported(void) {
    llama_memory_kvmem * mem = kvmem_capture_active();
    if (!mem) {
        return false;
    }
    std::string reason;
    return mem->conv_swap_supported(reason);
}

bool llama_kvmem_store_compatible_with_execution(const llama_kvmem_execution_state * other) {
    auto * mem = kvmem_capture_active();
    return mem && other && other->memory && mem->conv_compatible(*other->memory);
}

llama_kvmem_store_bundle * llama_kvmem_store_bundle_create() {
    auto * mem = kvmem_capture_active();
    std::string reason;
    if (!mem || !mem->conv_swap_supported(reason)) {
        throw std::runtime_error("portable conversation bundles require flash attention and host KV");
    }
    auto result = std::make_unique<llama_kvmem_store_bundle>();
    result->store = mem->make_conv();
    return result.release();
}

void llama_kvmem_store_bundle_free(llama_kvmem_store_bundle * bundle) {
    delete bundle;
}

bool llama_kvmem_store_bundle_swap(llama_kvmem_store_bundle * bundle) {
    auto * mem = kvmem_capture_active();
    if (!mem || !bundle || !bundle->store) {
        throw std::invalid_argument("missing portable conversation bundle");
    }
    return mem->swap_conv(bundle->store);
}

uint32_t llama_kvmem_store_bundle_rows(const llama_kvmem_store_bundle * bundle) {
    return bundle && bundle->store && bundle->store->runtime
        ? bundle->store->runtime->store().total_tokens() : 0;
}

void llama_kvmem_store_bundle_reset(llama_kvmem_store_bundle * bundle) {
    if (!bundle || !bundle->store) throw std::invalid_argument("missing portable conversation bundle");
    auto & store = *bundle->store;
    store.runtime->truncate_to(0);
    store.raw->truncate_to(0);
    if (store.mtp_raw) store.mtp_raw->truncate_to(0);
    store.row_positions.clear();
    store.resident.clear();
    for (auto & sum : store.q_sum) std::fill(sum.begin(), sum.end(), 0.0f);
    std::fill(store.q_count.begin(), store.q_count.end(), 0);
}

uint64_t llama_kvmem_store_active_bytes() {
    auto * mem = kvmem_capture_active();
    return mem ? mem->host_bytes() : 0;
}

uint64_t llama_kvmem_store_bundle_bytes(const llama_kvmem_store_bundle * bundle) {
    if (!bundle || !bundle->store) return 0;
    const auto & store = *bundle->store;
    uint64_t bytes = store.runtime ? store.runtime->allocated_bytes() : 0;
    bytes += store.raw ? store.raw->allocated_bytes() : 0;
    bytes += store.mtp_raw ? store.mtp_raw->allocated_bytes() : 0;
    bytes += store.row_positions.capacity()*sizeof(store.row_positions.front());
    bytes += store.resident.capacity()*sizeof(uint32_t) + store.q_count.capacity()*sizeof(uint32_t);
    for (const auto & sum : store.q_sum) bytes += sizeof(sum) + sum.capacity()*sizeof(float);
    return bytes;
}

int32_t llama_kvmem_store_create(void) {
    llama_memory_kvmem * mem = kvmem_capture_active();
    if (!mem) {
        return -1;
    }
    std::string reason;
    if (!mem->conv_swap_supported(reason)) {
        LLAMA_LOG_WARN("%s: KVMem host-store swap unavailable (%s)\n", __func__, reason.c_str());
        return -1;
    }
    kvmem_conv_pool * pool = kvmem_conv_pool_bind(mem);
    if (!pool) {
        return -1;
    }
    // Nothing may cross the LLAMA_API boundary: a bundle is a runtime, two
    // host mirrors and a pinned arena, so allocation failure is a -1 the
    // server already handles and not an exception httplib would swallow.
    std::unique_ptr<llama_memory_kvmem::ConvStore> store;
    try {
        store = mem->make_conv();
    } catch (const std::exception & e) {
        LLAMA_LOG_WARN("%s: KVMem host store allocation failed (%s)\n", __func__, e.what());
        return -1;
    }
    if (!store) {
        return -1;
    }
    kvmem_conv_entry entry;
    entry.id = pool->next_id++;
    entry.store = std::move(store);
    const int32_t id = entry.id;
    pool->entries.push_back(std::move(entry));
    kvmem_diag("KVMEM_STORE_CREATE id=%d n_stores=%zu\n", id, pool->entries.size());
    return id;
}

bool llama_kvmem_store_switch(int32_t store_id) {
    llama_memory_kvmem * mem = kvmem_capture_active();
    if (!mem || (*kvmem_current_execution().conversations).owner != mem) {
        return false;
    }
    if (store_id == (*kvmem_current_execution().conversations).active) {
        return mem->store_n_tokens() > 0;
    }
    kvmem_conv_entry * in = kvmem_conv_find(store_id);
    kvmem_conv_entry * out = kvmem_conv_find((*kvmem_current_execution().conversations).active);
    if (!in || !in->store || in->store->cold || (out && out->store) || (!out && (*kvmem_current_execution().conversations).active != -1)) {
        LLAMA_LOG_ERROR("%s: KVMem store %d is not a detached host store\n", __func__, store_id);
        return false;
    }
    const int64_t t0 = ggml_time_us();
    const uint32_t out_rows = mem->store_n_tokens();
    bool restaged = false;
    try {
        restaged = mem->swap_conv(in->store);
    } catch (const std::exception & e) {
        // swap_conv throws only before it hands either bundle over -- out of
        // the drain, or out of the clear it runs when the active store cannot
        // be drained safely -- so both handles and (*kvmem_current_execution().conversations).active still
        // name what they named. Past the handover nothing throws: the attach's
        // repair path is noexcept and reports its failure as false, so this
        // catch can never run with the two bundles already exchanged, which is
        // what makes skipping the two assignments below the right repair.
        // What the handles name is not necessarily unchanged: swap_conv empties
        // this store both when it cannot be drained safely and when the drain
        // throws, so store_n_tokens() may now read zero, and that is the one
        // signal the server acts on. Report the refusal rather than letting the
        // exception reach httplib, which would swallow it and leave the server
        // serving against a pool whose active entry no longer describes the
        // store. catch (...) is here for that last reason: this is an
        // LLAMA_API boundary, so nothing may cross it, not only what derives
        // from std::exception.
        LLAMA_LOG_ERROR("%s: KVMem store swap to %d failed (%s); store %d stays active\n",
                __func__, store_id, e.what(), (int) (*kvmem_current_execution().conversations).active);
        return false;
    } catch (...) {
        LLAMA_LOG_ERROR("%s: KVMem store swap to %d failed; store %d stays active\n",
                __func__, store_id, (int) (*kvmem_current_execution().conversations).active);
        return false;
    }
    // swap_conv leaves the outgoing conversation in the handle it was given.
    const int32_t out_id = out ? out->id : -1;
    if (out) out->store = std::move(in->store);
    else kvmem_conv_release(in->store); // temporary empty execution store
    (*kvmem_current_execution().conversations).active = store_id;
    kvmem_diag("KVMEM_STORE_SWAP out=%d in=%d n_stores=%zu out_rows=%u in_rows=%u restaged=%d ms=%.2f\n",
            out_id, store_id, (*kvmem_current_execution().conversations).entries.size(), out_rows, mem->store_n_tokens(),
            (int) restaged, (ggml_time_us() - t0) / 1000.0);
    return restaged;
}

int32_t llama_kvmem_store_current(void) {
    llama_memory_kvmem * mem = kvmem_capture_active();
    if (!mem) {
        return -1;
    }
    return (*kvmem_current_execution().conversations).owner == mem ? (*kvmem_current_execution().conversations).active : 0;
}

bool llama_kvmem_store_destroy(int32_t store_id) {
    llama_memory_kvmem * mem = kvmem_capture_active();
    if (!mem || (*kvmem_current_execution().conversations).owner != mem || store_id == (*kvmem_current_execution().conversations).active) {
        return false;
    }
    for (auto it = (*kvmem_current_execution().conversations).entries.begin(); it != (*kvmem_current_execution().conversations).entries.end(); ++it) {
        if (it->id != store_id) {
            continue;
        }
        kvmem_conv_release(it->store);
        (*kvmem_current_execution().conversations).entries.erase(it);
        kvmem_diag("KVMEM_STORE_DESTROY id=%d n_stores=%zu\n", store_id, (*kvmem_current_execution().conversations).entries.size());
        return true;
    }
    return false;
}

uint32_t llama_kvmem_store_rows(int32_t store_id) {
    llama_memory_kvmem * mem = kvmem_capture_active();
    if (!mem) {
        return 0;
    }
    if ((*kvmem_current_execution().conversations).owner != mem) {
        return store_id == 0 ? mem->store_n_tokens() : 0;
    }
    if (store_id == (*kvmem_current_execution().conversations).active) {
        return mem->store_n_tokens();
    }
    const kvmem_conv_entry * e = kvmem_conv_find(store_id);
    return (e && e->store) ? mem->conv_n_tokens(*e->store) : 0;
}

uint64_t llama_kvmem_store_bytes(int32_t store_id) {
    llama_memory_kvmem * mem = kvmem_capture_active();
    if (!mem) {
        return 0;
    }
    const bool pooled = (*kvmem_current_execution().conversations).owner == mem;
    if (!pooled) {
        return store_id == 0 ? mem->host_bytes() : 0;
    }
    if (store_id == (*kvmem_current_execution().conversations).active) {
        return mem->host_bytes();
    }
    const kvmem_conv_entry * e = kvmem_conv_find(store_id);
    return (e && e->store) ? mem->conv_host_bytes(*e->store) : 0;
}

bool llama_kvmem_store_park() {
    auto * mem = kvmem_capture_active();
    if (!mem || !kvmem_conv_pool_bind(mem)) return false;
    if ((*kvmem_current_execution().conversations).active == -1) return true;
    auto * out = kvmem_conv_find((*kvmem_current_execution().conversations).active);
    if (!out || out->store) return false;
    try {
        auto empty = mem->make_conv();
        mem->swap_conv(empty);
        out->store = std::move(empty);
        (*kvmem_current_execution().conversations).active = -1;
        return true;
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("KVMEM session park failed: %s\n", e.what());
        return false;
    }
}

static llama_memory_kvmem::ConvStore & kvmem_detached(int32_t id) {
    if ((*kvmem_current_execution().conversations).owner != kvmem_capture_active() || id == (*kvmem_current_execution().conversations).active)
        throw std::runtime_error("session snapshot requires a detached store");
    auto * e = kvmem_conv_find(id);
    if (!e || !e->store) throw std::runtime_error("unknown detached session store");
    return *e->store;
}

void llama_kvmem_store_freeze(int32_t id, std::vector<kvmem::SnapshotBuffer> & buffers) {
    auto & s = kvmem_detached(id);
    if (s.cold) throw std::runtime_error("session is already frozen");
    s.raw->snapshot_buffers(buffers);
    if (s.mtp_raw) s.mtp_raw->snapshot_buffers(buffers);
    s.cold = true; // partial stores must never be attached to inference
}

void llama_kvmem_store_thaw(int32_t id) {
    kvmem_detached(id).cold = false;
}

void llama_kvmem_store_snapshot_write(int32_t id, kvmem::SnapshotWriter & out) {
    auto & s = kvmem_detached(id);
    if (s.cold) throw std::runtime_error("session is already on disk");
    out.scalar(id);
    out.scalar(s.runtime->store().total_tokens());
    s.raw->snapshot_write(out);
    out.scalar(uint8_t(s.mtp_raw != nullptr));
    if (s.mtp_raw) s.mtp_raw->snapshot_write(out);
}

void llama_kvmem_store_release_payload(int32_t id) {
    auto & s = kvmem_detached(id);
    auto raw = std::make_unique<kvmem::RawKvStore>(s.raw->config());
    auto mtp = s.mtp_raw ? std::make_unique<kvmem::RawKvStore>(s.mtp_raw->config()) : nullptr;
    s.raw = std::move(raw); s.mtp_raw = std::move(mtp); s.cold = true;
}

void llama_kvmem_store_snapshot_read(int32_t id, kvmem::SnapshotReader & in) {
    auto & s = kvmem_detached(id);
    if (!s.cold) throw std::runtime_error("session restore requires an empty host payload");
    in.expect(id); in.expect(s.runtime->store().total_tokens());
    auto raw = std::make_unique<kvmem::RawKvStore>(s.raw->config());
    raw->snapshot_read(in, s.runtime->store().block_count());
    in.expect(uint8_t(s.mtp_raw != nullptr));
    auto mtp = s.mtp_raw ? std::make_unique<kvmem::RawKvStore>(s.mtp_raw->config()) : nullptr;
    if (mtp) mtp->snapshot_read(in, s.runtime->store().block_count());
    s.raw = std::move(raw); s.mtp_raw = std::move(mtp); s.cold = false;
}

uint64_t llama_kvmem_store_capacity(uint32_t tokens) {
    auto * mem = kvmem_capture_active();
    return mem ? mem->host_capacity(tokens) : 0;
}
