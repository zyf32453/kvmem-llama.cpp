# Historical KV GC on NInfer: integration and acceptance record

This branch ports the portable historical KV manager to the pinned NInfer backend. The native adapter is **opt in** with `--kvmem-gc-lru` and currently implements **LRU Host-page retirement only**. The portable redundancy policy is present in this repository, but NInfer does not yet publish the Mean-K, Query and thinking-range features needed to use that policy honestly. NInfer results must be reported separately from llama.cpp results.

## What changed

- NInfer's 64-token logical Host pages are registered with the portable GC manager only after their native KV version is complete. In RAM-only mode, `HostKvRecord::mutable_resident()` remains true even for complete pages, so the accepted frontier and `host_versions` are the seal signal.
- Sink, recent, active-window, image, query-selected and checkpoint-resident pages are protected. An in-flight append protects only the pages it touches. A committed victim becomes a tombstone and cannot be selected or recalled; its native Host record is detached at the foreground boundary. The worker releases the record after outstanding readers finish.
- Sparse-archive Query, Pressure and checkpoint paths skip retired pages. Per-request Host admission is bounded by physical Host capacity. The first adapter permits one active request, since independent sequence directories cannot yet coordinate one shared Host budget.
- GC pressure starts early enough for the worker and native Host allocator to finish a prefill burst: low ready is at least five prefill bursts plus two pages, and high ready is at least six bursts plus two, bounded by the actual Host slot count. The earlier two-to-five-slot threshold reached the next allocation before the release callback finished.
- The patch is the seventeenth ordered NInfer patch over pinned commit `63428ac249da`. `backends/versions.json` records the patch hash and exact prepared tree.

## Verification on 2026-10-29

- `python scripts/prepare-backends.py --backend ninfer --check`: passed, pinned submodule plus 17 patches ready.
- Portable Windows Host suite: 25/25 tests passed, including the page-GC bridge, Host storage, selection and memory-contract tests.
- A complete patched `ninfer-serve` CUDA sm_86 binary built in an isolated Ubuntu 24.04 container on `yifan@100.118.87.22`. The container supplied the required CMake and FFmpeg development packages; the CUDA toolkit came from the host. The copied source lacks `.git`, so the binary reports build ID unknown; provenance is the local patch manifest, not that runtime string.
- The patched service loaded Qwen3.8-27B GSQ-RCO IQ3_S NInfer v3 on the 16 GB RTX 3080 Ti Laptop GPU and answered a short KVMem/GC-enabled request with HTTP 200.
- A 7,651-token prompt, 48-token output limit and 96 MiB Host capacity exercised actual pressure. Configuration: `--kvmem-budget 2048 --kvmem-gen-reserve 512 --kvmem-host-mib 96 --kvmem-gc-lru --prefill-chunk 512 --max-context 8192 --kv-dtype rk4v4 --max-concurrency 1`, with MeanQ cosine working-set selection and device profiling disabled for this smoke test. The request returned HTTP 200. Native logs show LRU retirements beginning at frontier 4,096 and a total of 80 pages proposed, committed and reclaimed. The measured prompt rate was 668 tokens/s and the short 48-token generation averaged 23.3 tokens/s; these are diagnostic observations, not a controlled throughput comparison.
- A second request with a different prefix returned HTTP 200 and final text `HEALTHY`, confirming the service continued after pressure and Host-page release.
- After a fresh model start, a 7,657-token variant with a 160-token output limit also triggered LRU retirement and returned HTTP 200 with `finish_reason=stop` and the correct final answer `599`. This checks final output rather than merely surviving a request that ends at its output limit.
- Earlier runs failed with `native KV RAM capacity exhausted`. The first failure exposed two integration bugs: RAM-only pages were never registered because `mutable_resident()` was used as a seal test, and every pending append protected the entire archive. A later failure exposed a late pressure trigger: 11 pages were marked for retirement at frontier 5,120, but the next Host allocation arrived before their physical release. The fixes above were applied before the successful run.

## Current limits and next acceptance

1. Native NInfer uses 64 tokens per block and at least one recent block. This acceptance is separate from llama.cpp's earlier 32/0 experiment.
2. Native **redundancy GC is not implemented**. It needs sealed feature publication and a distinct policy flag; choosing LRU victims under a redundancy label would be misleading.
3. Only the single-request CUDA path, long prefill pressure, decode continuation and a subsequent different-prefix request have been exercised. Prefix reuse, query replay after retirement, MTP accept/rollback, SSD mode, parallel requests, long-running stability and official Terminal-Bench quality remain to be tested before production use.
4. The 96 MiB Host pool is deliberately small to force GC on an 8K prompt. It is not the proposed 256K historical budget for the full benchmark. The observed speed does not establish that NInfer is faster than llama.cpp on matched hardware and workload.
5. Debug state logs require `--kvmem-verify-transfers`; the default service does not emit a line per token.

The remote test source and binary are isolated under `/tmp/kvmem-ninfer-gc-20261028`; no model weights were copied into that directory.