#pragma once

// GPU stage-in for quantized raw-K:
//   H2D packed q8/q4 → dequant F32 → NeoX RoPE → Walsh-Hadamard → quant into cache.
// F32-H2D path remains for fallback / unrotated V. Host Hadamard/RoPE is fallback only.

#include "ggml.h"

#include <cstddef>
#include <cstdint>

struct kvmem_gpu_state;
kvmem_gpu_state * kvmem_gpu_state_create();
// The owning execution state must be active while destroying GPU scratch.
void kvmem_gpu_state_free(kvmem_gpu_state * state);

bool kvmem_stagein_gpu_ready(size_t n_f32, size_t n_packed = 0);
void kvmem_stagein_gpu_free();

// Request-owned packed-capture workspace, prepared on a fresh staging domain
// before publishing the request. Queue saturation returns false to the caller
// for flush/retry; gpu_ready cannot grow this workspace after admission.
constexpr size_t KVMEM_CAPTURE_MAX_JOBS = 8192;
bool kvmem_capture_workspace_reserve(size_t n_f32, size_t max_packed_slice,
                                      size_t max_jobs = KVMEM_CAPTURE_MAX_JOBS);
struct KvmemCaptureWorkspace {
    bool reserved;
    size_t job_limit, device_bytes, pinned_bytes, events, host_bytes;
};
KvmemCaptureWorkspace kvmem_capture_workspace_status();
enum class KvmemCaptureReservePoint { F32, Packed, Pin0, Pin1, Event0, Event1,
                                      H2DEvent, DeviceOps, HostItems, HostOuts, HostOps };
using KvmemCaptureReserveFault = bool (*)(void *, KvmemCaptureReservePoint) noexcept;
// Deterministic preparation failure seam. Empty in production; owner thread only.
void kvmem_capture_reserve_test_fault(KvmemCaptureReserveFault fault, void * context);

bool kvmem_stagein_fwht_ok(int nrot);
bool kvmem_stagein_quant_ok(ggml_type ty);

// Host buffer is free to reuse after return (H2D is fenced). Later kernels stay async.
bool kvmem_stagein_h2d_f32(const float * host, int64_t n);
bool kvmem_stagein_h2d_packed(const void * host, size_t n);
bool kvmem_stagein_dequant(ggml_type ty, int64_t n_rows, int64_t n_embd);
bool kvmem_stagein_rope_neox(int64_t n_tokens, int n_head, int n_embd_head, int n_rot,
                             int32_t pos0, const float * theta, int n_theta);
bool kvmem_stagein_fwht(int64_t n_rows, int64_t n_embd, int nrot);
bool kvmem_stagein_quantize(ggml_type ty, void * gpu_dst, int64_t n_rows, int64_t n_embd);
bool kvmem_stagein_h2d_bytes(void * gpu_dst, const void * host, size_t n);
void kvmem_stagein_sync();

// 32 MiB packed host+GPU slab. Enqueue (block,layer) rows; flush does one H2D
// then dequant/RoPE/Hadamard/quant (K) or D2D (packed V). Extra VRAM = 32 MiB.
bool kvmem_stagein_enqueue_k(
        ggml_type ty, const void * packed, size_t nbytes, uint8_t * dst,
        int64_t nt, int64_t n_embd, int nrot,
        int n_head, int n_embd_head, int n_rot_rope, int32_t pos0,
        const float * theta, int n_theta,
        int64_t * copy_us, int64_t * rope_us, int64_t * hadamard_us, int64_t * set_us);
bool kvmem_stagein_enqueue_v(const void * packed, size_t nbytes, uint8_t * dst,
                             int64_t * set_us);
bool kvmem_stagein_flush(int64_t * copy_us, int64_t * rope_us,
                         int64_t * hadamard_us, int64_t * set_us);

// Reverse slab: packed GPU V → pin. Gather kernel packs scattered rows into
// the 32 MiB GPU slab, then one D2H. Two host pins; no extra VRAM slab.
bool kvmem_stageout_enqueue(const void * gpu_src, size_t nbytes);
size_t kvmem_stageout_used();
// -1 permits a synchronous fallback: no submitted operation from that batch
// remains and its raw-pointer queue has been discarded. An unconfirmed drain
// throws, retaining queue/buffer ownership; the request must be closed.
int kvmem_stageout_submit(int64_t * copy_us);
// false also guarantees the failed batch's device work has drained. It does
// not discard the NEXT, as-yet-unsubmitted queue. An unconfirmed drain throws.
bool kvmem_stageout_wait(int slot, int64_t * copy_us);
const uint8_t * kvmem_stageout_slot_base(int slot);
// May throw if in-flight work cannot be proven complete. Never clears an
// unresolved batch or grants permission to reuse its buffers/source slots.
void kvmem_stageout_clear();

// Internal deterministic fault seam, scoped to the current execution domain.
// Empty in production. A true result skips the named CUDA call. The hook
// must not throw or change domains; install/remove it on the GPU owner thread.
enum class KvmemStageoutPoint { OpsCopy, GatherLaunch, PackedCopy, ItemCopy, EventRecord, EventWait, Drain };
using KvmemStageoutFault = bool (*)(void *, KvmemStageoutPoint, size_t) noexcept;
void kvmem_stageout_test_fault(KvmemStageoutFault fault, void * context);

// Batched device copies (layout gather/scatter). One kernel; false → caller D2D.
bool kvmem_d2d_batched(const void * const * src, void * const * dst,
                       const size_t * nbytes, int n);

// Decode mean-K running sum on GPU. Extra VRAM = n_layer * n_embd * 4;
// one owned pinned layer is used for reads. CUDA errors close this accumulator
// until explicit free. false means unsupported input/allocation before work,
// never permission to replay an uncertain submitted addition on the CPU.
bool kvmem_meank_ready(uint32_t n_layer, uint32_t n_embd);
void kvmem_meank_free();
void kvmem_meank_sync();
// Prove outstanding device work finished before discarding its owners.
// Does not reopen a failed accumulator or publish any accumulated contents.
void kvmem_meank_cancel();
void kvmem_meank_zero(uint32_t il);
bool kvmem_meank_add(uint32_t il, ggml_type ty, const void * gpu_k,
                     uint32_t tok0, uint32_t n_keep, uint32_t n_embd,
                     int64_t ne0, size_t nb0, size_t nb1, size_t nb2);
// Copies into caller memory only after a checked completion; an exception
// leaves the caller buffer untouched and retains the internal DMA destination.
bool kvmem_meank_d2h(uint32_t il, float * host, uint32_t n_embd);
enum class KvmemMeanKPoint { Zero, AddResult, ReadCopy, Wait, Drain };
using KvmemMeanKFault = bool (*)(void *, KvmemMeanKPoint, uint32_t) noexcept;
// Per execution domain. AddResult observes an actually submitted kernel;
// other true results skip the named call. Production defaults to no hook.
void kvmem_meank_test_fault(KvmemMeanKFault fault, void * context);
