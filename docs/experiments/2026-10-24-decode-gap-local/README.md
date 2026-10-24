# Decode-gap GC and token-prefix continuity: local checkpoint (2026-10-24)

This follows `36041c2`. The initial local
checkpoint below predates remote validation; current remote IQ3 findings are
recorded in [REMOTE-VALIDATION.md](REMOTE-VALIDATION.md). The updated r2
library subsequently passed an eight-turn GC game with 29/29 browser checks,
strict prefix reuse and decode-gap ordering. Five paired efficiency rounds
also passed the stated experimental p95 and throughput thresholds. This is
not a production acceptance guarantee. The older
r5 long GC game was a separate failed 7/8-turn run; its 27/29 checks are not
task completion.

## Changes under test

- The GC worker still scans, computes policy scores, and validates a bounded
  batch. In delete mode it now publishes prepared intents without retiring
  them. The foreground commits exact-generation `Retired` transitions only
  before and after the host decode method runs through the public
  `llama_decode` wrapper, or before a new request's historical admission.
  Patch `0018` moves these hooks out of the graph harvest, which may run while
  a decode is still in progress. A decision prepared after the entry hook
  waits for the exit hook; no retirement is done inside active selection.
  A separate short mutex means
  foreground commit does not wait for the worker's scoring mutex.
- The worker acknowledges committed results and performs physical Host release
  later. New active admission rejects `Retired`; existing execution leases
  remain valid. Slots are still reusable only at `Ready`, after references and
  callbacks drain. No direct overwrite of `Retired` was introduced.
- `--kvmem-trace` now records monotonic timestamps for host decode entry and
  return, logical commit, and new active-window selection. Use
  `scripts/check_gc_decode_gap.py LOG --require-commit --require-selection`
  against a complete new server log. The host decode return does not imply
  asynchronous GPU work has finished; execution leases protect that work.
  At this initial checkpoint the check had not yet run on the model; the
  subsequent IQ3 run and its passing result are in `REMOTE-VALIDATION.md`.
- The redundancy policy now requires a directly compared, retained witness for
  each redundancy-ranked victim. Its A~B, B~C, A!~C regression passes locally.
  An acknowledged committed victim also pins that witness generation against
  being called redundant in a later round merely because B~C. The two-round
  regression first retires A using B, changes Query to favor C, and verifies
  that B cannot be retired with a `Redundancy` reason. Capacity fallback may
  still choose B with the distinct `Lru` reason; this is explicitly lossy and
  is not claimed to preserve A's information. A manager-level test now drives
  the deferred boundary commit and acknowledgement across both rounds, and
  confirms the second B deletion is counted as one lossy intent.
- The server retains the actual generated token IDs when a later rendered chat
  prompt is an exact byte extension of the **full cached token sequence's**
  detokenized surface but canonical re-tokenization differs. It validates both
  suffix and whole joined-token text round-trips plus native query-span
  location before stitching. Otherwise it retains the existing reset path and
  logs the byte common-prefix length and failed validation stage. The old r5
  probe found a one-token canonical re-tokenization discrepancy, but did not
  archive original sampled IDs or raw pre-parser bytes; only a new model run
  can establish that this guard fixes that case.

## Local verification

- MinGW host build of `historical_kv_gc_test` and
  `historical_kv_redundancy_test`: passed.
- Relevant historical CTest subset: **18/18 passed**. The new manager test
  proves no retirement before the boundary, no new active lease after it,
  old-lease survival, no overwrite before `Ready`, and asynchronous callback
  release. A blocked synthetic policy scorer does not block the boundary call.
- Decode-gap trace checker tests: **7/7 passed**, including negative ordering
  cases. The checker correctly rejects the old r5 log because it lacks the new
  event timestamps.
- `tools/llama-kvmem-server.cpp` single-file MinGW C++17 syntax check:
  passed. The patched `llama-context.cpp` and capture hook also passed
  single-file C++17 syntax checks, and the `0018` patch passed reverse-apply
  validation against the current llama.cpp checkout. A CPU-only integration
  build stops on the pre-existing CUDA dependency (`cuda_runtime.h` missing),
  so it does not establish a full adapter build or model run.

## Required remote validation

1. Build the updated isolated CUDA tree and run the native manager,
   redundancy, Query coverage, and active-selection tests.
2. Run a traced small delete-mode model probe with actual CPU release. Verify
   commit ordering and bounded foreground commit duration with the checker;
   record whether the next active window sees retired blocks.
3. Repeat the original eight-turn no-repair game with GC off and on. Every
   continuation must reuse cached token IDs; all eight turns and 29 browser
   checks must pass after real CPU reclamation.
4. Only after quality passes, run paired repeated off/shadow/delete decode
   timing, reporting p95, throughput, time to first release, and memory.

Source files and hashes are captured in `source-manifest.json`. They should be
compared with the remote isolated tree after any authorized transfer.
