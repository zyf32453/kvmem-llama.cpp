# Historical KV GC on the NInfer backend: integration record

This branch is an isolated port against the multi-backend `upstream/master` tree.
The NInfer source now contains an opt-in **LRU** Host-page retirement path,
reproduced by `backends/patches/ninfer-historical-gc-lru.patch` after the prior
16 NInfer patches. The portable LRU and redundancy policies both build, but
the NInfer native path selects LRU only. **No NInfer model run has yet validated
the new GC path.** Do not report the model checks below as a GC acceptance result.

## Integration work in progress on 2026-10-28

- The common working-set contract now keeps a retired page as a tombstone in
  the contiguous logical catalog. A retired page cannot be selected or copied
  back to Host when its old GPU view is replaced. A host contract test covers
  retirement while the page remains resident on the device.
- The NInfer Query and Pressure candidate lists exclude retired logical pages.
  A saved checkpoint checks that its own resident pages are available; it no
  longer requires every earlier logical page to retain Host payload. Host
  accounting after restore/retention sums actual records rather than assuming
  a dense archive.
- The affected host tests (`historical_gc_ninfer_core_test`,
  `kvmem_store_test`, `memory_contract_test`) pass 3/3 with MinGW. The NInfer
  CUDA program itself has not been linked or run with these changes. The
  serving option parser and its new test compile in host-only syntax checks.
- A host-tested page handoff now connects the portable LRU manager to an
  exact 64-token logical-page directory. It lets the worker propose a victim,
  publishes the committed handle at a decode boundary without allocating, and
  keeps the physical slot unavailable until the native Host owner confirms
  release. An additional test checks this handoff and protection of the other
  pages. The NInfer Program now calls this handoff through the opt-in
  `--kvmem-gc-lru` path: it registers complete sealed pages, protects current
  and checkpoint-selected rows, commits retirement before the next working-set
  choice, detaches native Host records, and acknowledges release. This native
  call path is source-complete for LRU. In an isolated copy on the remote host,
  `kvmem_window.cpp`, `planning/request_plan.cpp`, `model_instance.cpp`, and
  the portable page bridge pass low-priority C++ syntax checks with CUDA 13.2
  headers and the sm_86 build macro in an isolated remote source directory.
  A complete NInfer
  build is still unavailable there: host CMake is 3.22 versus the project's
  3.28 minimum, and the host lacks pkg-config/required FFmpeg development
  packages. No model/GPU run has been performed for this GC path.
- A long GC-enabled request now reserves at most the physical Host-page budget
  at admission; automatic output remains bounded by logical context. The
  per-step append still prepares every new logical page, so the Host store must
  actually recover capacity as old pages retire. The first adapter accepts
  `max_concurrency=1` only because a separate GC directory per sequence cannot
  yet coordinate a shared Host budget across multiple requests.
- Native Host release uses a weak allocation ticket. The decode boundary only
  detaches the old record; the GC worker destroys it. The repository cannot
  reuse a retired page's slot until all in-flight Host readers have released
  that allocation. A host test exercises a reader that outlives the archive
  owner.
- The patched NInfer source is recorded as the seventeenth ordered patch;
  `python scripts/prepare-backends.py --backend ninfer --check` reproduces the
  exact prepared tree. All 25 portable host tests pass on Windows.
- Native CUDA validation, multi-request capacity coordination, redundancy
  feature publication and redundancy policy selection are still pending. The
  ongoing 89-task llama.cpp experiment owns the available remote GPU; do not
  use its results as NInfer GC evidence.

## Verified on 2026-10-27

- Prepared the pinned NInfer submodule (`63428ac249da`) with the repository's
  16 integration patches.
- Built the portable `kvmem` library and passed
  `historical_gc_ninfer_core_test`, `kvmem_store_test`, and
  `memory_contract_test` (3/3). The new GC test checks a protected sink page,
  a recent page, a redundant thinking page, a decode-boundary retirement, and
  deferred physical-slot reuse while a GPU lease remains.
- `python scripts/prepare-backends.py --backend ninfer --check` reports the
  pinned submodule plus all 16 patches ready. Host tests use the MinGW runtime
  on `PATH` when invoking CTest; without its DLLs, Windows exits with
  `0xc0000135` before any assertion runs.
- Downloaded the published IQ3_S NInfer v3 Qwen3.8-27B artifact to
  `/home/yifan/models/WaveCut/Qwen3.8-27B-GSQ-RCO-IQ3_S-NInfer-v3/` on
  `yifan@100.118.87.22`. Both the model and conversion metadata matched the
  publisher's SHA256SUMS.
- The unmodified `ghcr.io/iamwavecut/ninfer-all:cuda12.9` image loaded that
  artifact on the RTX 3080 Ti Laptop 16 GB and answered a short chat request
  with `READY`. That smoke request used native NInfer with 8192 context tokens,
  RK4V4 KV, no KVMem and no GC; its reported generation rate was 32.47 tok/s
  over only 16 output tokens. It is not a throughput comparison.
- The same upstream image rejects `--kvmem` as an unknown option. Runtime GC
  acceptance requires building the **patched** NInfer submodule, rather than
  using this published upstream image.

## Native invariants that the adapter must preserve

1. NInfer uses fixed 64-token physical pages; its KVMem parser requires at
   least one 64-token recent page. The first NInfer acceptance therefore uses
   native 64/64 and reports separately from llama.cpp 32/0.
2. `KvmemWindowState::archive` is indexed by every logical page. Both
   `restore_memory_prefix()` and `memory_restorable_history()` currently
   require *every* page of a checkpointed prefix to have a Host record.
   Releasing a page without changing these checks breaks query replay and
   later prefix reuse. A safe sparse archive needs tombstones, selectors that
   skip tombstones, and checkpoint protection for every resident page a saved
   checkpoint can restore.
3. Query replay currently scores all logical page IDs, and Pressure uses the
   resident page list. Neither excludes GC-retired pages. Both paths must use
   the same committed retirement state before the next active-window choice.
4. `planning/request_plan.cpp` used to reserve Host payload for the entire
   expected prompt plus output before execution. The opt-in LRU path now caps
   this reservation at the physical Host budget. The runtime can still fail
   under sudden pressure if the background worker has not freed enough eligible
   pages by the next append. Such a case must be measured in native model tests;
   a bounded reservation alone does not guarantee every long request succeeds.
5. `window.payload_bytes`, active request reservations, retained histories,
   asynchronous copies and `HostKvRecord` leases all contribute to the
   allocation lifetime. Marking a page unavailable can happen at a decode
   boundary; releasing or overwriting its native record must wait until every
   in-flight transfer, GPU use and checkpoint dependency is safe.

## Remaining integration and acceptance

- Publish sealed Mean-K and query/observation features to the redundancy
  strategy, then add a separate NInfer policy flag. Keep the working-set
  selection strategy independent of the Host GC deletion policy.
- Test the implemented decode-boundary retirement, sparse Query/Pressure
  filtering, Host release ticket and bounded admission on native CUDA. Cover
  prefill, decode, query replay, prefix reuse, MTP rollback, capacity pressure
  and in-flight transfer races with 64-token pages.
- Build the patched NInfer worker on the remote CUDA toolchain and run the
  IQ3_S model with the agreed native 64/64 layout. Only then compare no-GC,
  LRU and redundancy with identical NInfer settings and official task rewards.

The portable strategy tests alone do not establish native NInfer compatibility.
