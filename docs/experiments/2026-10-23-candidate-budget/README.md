# Historical redundancy GC: bounded-candidate trial (2026-10-23)

This is an opt-in experiment, not a change to the default GC policy. It tests
whether bounding the *scored CPU block pool* and comparing all blocks within
that pool is useful here. It is not R-KV's token-level implementation: the
paper bounds retained KV plus a generation buffer and still does an all-pairs
comparison within each compression round. This trial leaves the GPU budget,
CPU historical capacity, and existing reclamation manager unchanged.

## Trial implementation

Set `KVMEM_HISTORICAL_GC_MAX_CANDIDATES=64` before starting redundancy GC.
Unset it to keep the existing all-live-block / newest-32-reference behavior.
The cap applies only to live, sealed, non-mandatory historical blocks. When
eligible blocks exceed the cap, the policy keeps up to `min(32, N/4)` newest
blocks as anchors, samples the older age range in bands, and rotates the
sample position on later GC rounds. It then reads at most N feature rows and
compares every selected row with every other selected row. It does not build
an N-by-N matrix. Retirement still requires exact handle and content-version
validation by the manager.

This is a coverage/computation tradeoff, not an optimality claim. A duplicate
outside the chosen bands can be missed in a particular round. Sampling all
eligible metadata and sorting it still costs O(C log C) for CPU capacity C;
the cap reduces expensive feature reads and similarity work, not snapshot
creation or the fixed-capacity feature cache. A thinking-only filter was not
used because historical block metadata has no trustworthy thinking label,
and a 32-token block may straddle a thinking boundary.

## Verification

Local MinGW historical test subset: 16/16 passed. The new controlled tests
verify old/old duplicate detection inside the bounded pool, capped feature
reads/pairs, mandatory-block exclusion, sampling rotation, and multi-round
progress to a Ready target larger than the candidate cap. The same redundancy
test passed on the remote Linux/CUDA build.

The remote IQ3 model was
`Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf`. The native benchmark retained the
accepted 32-token block, GPU budget 36,864, generation reserve 16,384,
sink 1,024, recent 0, q8_0 KV, CPU capacity 1,024, and Ready low/high
352/384. Each run used one warmup and one measured round per off/shadow/delete
mode, with 6,144 input tokens and 256 measured decode steps.

- Default newest-32-reference policy: four GC scoring rounds read 672 block
  features and performed 21,376 similarity comparisons. The measured off
  and delete p95 were 65.66 and 70.97 ms; throughput was 16.64 and
  16.81 tokens/s.
- N=64 full-comparison pool: four scoring rounds read 256 features and
  performed 16,128 comparisons, reductions of 61.9% and 24.5% in this
  short workload. Off and delete p95 were 55.62 and 53.19 ms; throughput
  was 20.13 and 19.93 tokens/s.

Both benchmarks completed with identical output across their own modes,
stable GPU residency, no policy/worker failures, and 64 CPU blocks committed
and host-released during delete-mode decoding. **The absolute timing values
between the two separate runs are not a valid speedup comparison:** even
their off baselines differ substantially. One measured round per setting
cannot establish a stable p95 result. The measured pool had about 168 valid
blocks per GC round, so the full-capacity 1,024-block calculation would also
overstate the savings for this particular workload.

An eight-turn browser-FPS task with N=64, no automatic repairs, and the same
IQ3/GC parameters finished 8/8 turns and passed 29/29 browser checks.
It committed 54 CPU retirements, with 38 early host payload releases and
16 complete slot reclaims; policy and worker failures were zero. Each turn's
output SHA-256 equals the earlier default-policy run. GC first acted in the
last turn, so this does **not** establish sustained quality after reclamation
or the value of N=64 for a longer history.

## Evidence and next decision

Machine-readable native results are `default-runs.json`,
`default-summary.json`, `n64-runs.json`, and `n64-summary.json` in this
directory. The browser summary, 29 checks, and output hashes are in
`quality-n64-*.json`. Full raw logs and model outputs remain in the isolated
remote directory `/tmp/kvmem-candidate-budget-20261023/` and the local
`tmp/candidate-n64-fps-r1/` directory.

The next discriminating experiment should use a history that exceeds the GPU
working set and continues after multiple real CPU retirements. Compare a
small sweep of N values against the default under matched task inputs, with
multiple paired timing rounds. Keep N=64 opt-in until that quality and p95
evidence exists.
