# Remote CUDA and IQ3 validation checkpoint (2026-10-24)

This is an uncommitted follow-up to `36041c2`, not a production acceptance
report. The source snapshot and its hashes remain frozen in `source-manifest.json`
and `README.md`. The previous r5 long GC game failed after 7/8 turns; its
27/29 checks are not task completion.

With explicit user authorization, the 18 changed source/test/document files
were transferred to `/tmp/kvmem-full-query-20261023`, and all SHA-256 hashes
matched the snapshot manifest. The test driver and fixed browser workload were
synced separately afterward because the isolated directory contained an older
driver and lacked the fixture; their transfer hashes also matched. Patch
`0018` applied to the isolated llama.cpp source. The Release/CUDA 86 server
build and four relevant CTests (`historical_kv_gc_test`,
`historical_kv_redundancy_test`, `historical_kv_query_coverage_test`, and
`thinking_span_tracker_test`) passed.

The first three-turn IQ3 probe completed and committed four intents but had no
subsequent active-window selection. The strict checker correctly reported that
evidence missing. A fourth request was appended to the fixed field-extraction
probe without changing the first three turns. In the second probe, all four
requests returned HTTP 200/stop, all three continuations reused the cached
prefix, the full Query span was observed, and no GC failure was reported.
The strict decode-gap checker passed: 481 decode boundaries, 5 committed
intents in 5 prepared batches, 4 active-window selections, and one selection
after a committed retirement. The longest observed foreground commit was
4.532 microseconds; none exceeded 10 milliseconds. By the final request,
the runtime reported 5 early host payload releases and 4 reclaimed slots.
Raw logs, request/reply data, and the checker result are under
`tmp/full-query-20261023/decode-gap-probe-r2/`.

This small test establishes that logical retirement and later physical reclaim
run on the IQ3 model. It does not establish long-run prefix continuity, 29
browser checks, a memory upper bound, or paired p95/throughput. The eight-turn
delete-mode run started as `gc-long-r6c`; earlier r6 and r6b starts stopped
before model loading because of the stale driver and missing workload file.
The original browser checker was independently rerun against the old r5 off
output using local Edge and still passed 29/29, so the browser check environment
is available for the new generated game.

The independent `scripts/check_strict_prefix_continuity.py` requires each
continuation's cache hit and server admission count to equal the previous
turn's actual total token count; any reset or refusal fails. It passes the
four-turn IQ3 smoke, rejects the old r5 GC refusal (7/8 completed), and
rejects the old r5 off eighth-turn reset even though that game's browser
checks passed. The ongoing r6c run must pass this checker across all eight
turns; a merely positive cache-hit count is insufficient.

After r6c started, local policy review found a further capacity fallback
ordering issue: a block carrying an already-retired block could be chosen by
LRU while another independent live candidate remained. The local policy now
tries independent retained candidates before previous witnesses, and the
cross-round manager/policy regression passes under MinGW. A second regression
shows that, when every independent candidate is protected, a prior witness
may still be chosen only as an explicitly lossy last resort. The relevant
host CTest subset passed 17/17. The three changed files are fingerprinted in
`source-manifest-witness-fallback-r2.json`. This later edit is
**not in the running r6c binary** or the frozen 18-file source snapshot. The
r6c outcome remains evidence for its recorded revision; the revised fallback
requires a separate CUDA build and model validation before acceptance.
The three-file r2 overlay archive is staged in the same remote isolated root
as `gc-witness-fallback-r2.tar` (SHA-256
`9fe75512a37ea058fc64bf943cbfe6898a0e4f7adcbfc22a36a184c8e424c3ef`),
but has **not** been extracted over the source or built while r6c is running.

During r6c, turn 3 completed with a full 13,881-token prefix hit, 311
committed retirements and 311 early Host payload releases. Turn 4 was admitted
with `reset=0`, cached 24,198 tokens, and its first active-window selection
recorded a timestamp later than the previous GC commit. A checkpoint trace
cut only at a completed decode boundary passed the strict ordering checker:
24,416 decode boundaries, 633 committed intents, 1 selection after commit,
4.26 microsecond foreground-commit p95, 9.64 microsecond maximum, and zero
commits above 10 milliseconds. See `gc-long-r6c-decode-gap-checkpoint.json`.
This is an in-progress checkpoint on the old revision, not eight-turn or
updated-fallback acceptance. Turn 3's `reclaimed=0` means payload release
alone had not yet made those slots Ready; later turns must verify that step.
The running r6c server binary has SHA-256
`0701f82a3f4846216e640197e4dd6a13e43787ff776b7c060cac82c329a3104b`;
the r2 policy overlay remains unextracted. During turn 4, position-detach
trace events began appearing, but the final `reclaimed` statistic was not yet
available when this checkpoint was written.

Turn 4 later completed with HTTP 200/stop, 9,107 generated tokens and a full
24,198-token cache hit. Its end-of-turn GC statistics advanced to
`committed=633`, `host_released_early=633`, `reclaimed=311`, `ready=62`, with
zero worker/policy failures. This is the first confirmed progression from
asynchronous payload release to 311 reusable slots in the long model run.
Turn 5 was then admitted with `reset=0`, the full 33,416-token preceding
history, and an active-window selection later than the previous commit.
The fifth turn and remaining browser task were still running at this point.

Turn 5 subsequently completed with HTTP 200/stop and 10,386 generated tokens.
It reused the entire prior 33,416-token history, bringing the conversation to
43,920 tokens. The sixth request was admitted with `reset=0` and
`cached=43920`. The latest turn-5 GC finish line reported `committed=922`,
`host_released_early=922`, `reclaimed=633`, `ready=95`, and zero worker or policy
failures. The full Query trace located the next request's 121-token span. This
is a five-turn checkpoint; the remaining turns and browser checks are still
required for acceptance.
At the start of turn 6, the server memory sampler had 3,386 one-second samples:
process RSS was 5,081 MiB at the latest sample and its observed high-water mark
was 11,936 MiB. These are in-progress process measurements, not a completed
memory-bound acceptance result.

Turn 6 completed with HTTP 200/stop, 5,421 generated tokens, and a full
43,920-token cache hit; turn 7 was admitted with `reset=0`, `cached=49473`.
The finish statistics advanced to `committed=1252`, `host_released_early=1252`,
`reclaimed=922`, `ready=54`, and zero worker/policy failures. With the first
six generated files, the isolated Edge browser check created a working game
and passed 27/29 checks. Its two failures are expected at this incomplete
checkpoint: turns 7 and 8 are missing, and the final `state.js` revision that
adds the round field is scheduled for turn 8. The raw provisional checker
result is in `tmp/full-query-20261023/gc-long-r6c-through6-browser/`.

With turn 7 still generating, a second timestamp audit cut the live log at
its last complete decode return and passed: 53,113 decode boundaries, 1,425
committed intents, 7 active-window selections, 4 selections after a commit,
4.26 microsecond commit p95, 24.688 microsecond maximum, and no commit above
10 milliseconds. See `gc-long-r6c-decode-gap-turn7-in-progress.json`. This is
still a checkpoint; the final full-log audit must be rerun after the process
exits.

## First seven-turn long run: terminal result

`gc-long-r6c` stopped at turn 7. Turns 1–6 finished normally and reused the
complete prior token count. Turn 7 entered with `reset=0`,
`cached=49473` and returned HTTP 200 with `finish_reason=length` at exactly
16,384 generated tokens. All 57,297 output characters were reasoning text;
there was no `weapons.js` content. Thus the eighth turn was never attempted.
The strict continuity report correctly fails 8-turn acceptance on the missing
turn and non-stop seventh response, despite all six preceding cache hits and
the seventh cache hit being exact. There were no historical request refusals.
The final GC statistics were `committed=1425`, `host_released_early=1425`,
`reclaimed=1252`, `ready=211`, with zero policy/worker failures. The policy
reported 815 explicitly lossy LRU fallbacks; this is a material quality risk
and must be considered when interpreting the truncated seventh turn.
The older off-mode r5 run with the same model, native template, fixed seed and
16K cap did complete its seventh turn at 13,913 generated tokens, but its own
response history differed, so that alone does not isolate GC as the cause.

The final full-log decode-gap audit **passed**: 64,795 decode boundaries,
1,425 committed intents, 7 active selections, 4 selections following a GC
commit, 4.26 microsecond foreground-commit p95, 24.688 microsecond maximum,
zero commits over 10 milliseconds. See `gc-long-r6c-decode-gap-final.json` and
`gc-long-r6c-prefix-final.json`. This establishes timing safety for the
recorded run, not task quality or the later r2 policy revision.

To distinguish a sampling/model difficulty from quality loss after GC, an
off-mode diagnostic has been launched using exactly the seventh request body
from this failed run as its sole input. The replay file is
`redundancy-gc/off-same-prompt-turn7-r1-request.json` on the remote host, with
SHA-256 `3939e078e9cc85843599f96541bae22cfaf94637e5ae97f0fe3abef2037857de`.
This control still starts from a newly prefetched context, so it is a diagnostic
comparison, not a complete paired eight-turn workload.
The running off-mode driver's submitted body was independently compared with
the failed GC run's seventh request and found byte-for-byte equivalent after
JSON parsing (canonical body SHA-256
`80afae865233225b87de324824880fef82ddb3bfae6a813e3f37715f3cbd53ab`).
The diagnostic also ended with `finish_reason=length` at exactly 16,384
generated tokens: 53,118 reasoning characters and 3,032 content characters.
Its code fence was incomplete and `weapons.js` was cut midway through
`G.reload`. The identical prompt thus fails under default `xhigh` with GC
off, too. The GC arm produced no code at all, but one pair of runs does not
prove GC is the sole cause of the output difference.

The r6c server trace also recorded one real
`historical_prefix_stitch canonical=49576 cached=49473 suffix=103 stable=49576`
event before the seventh request. Its preceding response had committed 49,473
sampled token IDs, and the seventh request reported a 49,473-token exact
cache hit instead of resetting and recomputing the history. This exercises
the token-ID-preserving guard on the native xhigh template. The later r2
eight-turn low-effort runs had no stitch event: all continuations already had
a canonical token prefix. Thus the low-effort game proves sustained prefix
continuity but should not be described as a second direct exercise of the
stitch branch. The r6c seventh output truncation under xhigh is separate from
prefix reuse, as the same request also truncated with GC off.

After r6c exited, the r2 witness-fallback overlay was extracted into the
remote isolated source tree. All three source/test/checker hashes match
`source-manifest-witness-fallback-r2.json`. The off-mode diagnostic still uses
the unchanged server binary with SHA-256
`0701f82a3f4846216e640197e4dd6a13e43787ff776b7c060cac82c329a3104b`;
the newly extracted r2 source had not yet been compiled at that point.
An isolated `build-cuda-r2/` CUDA build was then configured from this source
and started at lower CPU priority; it writes no files into the old binary's
build directory. The running off-mode diagnostic therefore continues on the
unchanged old binary while r2 compilation proceeds independently.
That separate build was stopped after confirming it was recompiling the full
CUDA template set, which added unnecessary CPU load during the diagnostic.
Its isolated process group exited; the inference driver and server remained
live. Once the diagnostic ended, the existing `build-cuda/` tree was rebuilt
incrementally in 14 seconds. It recompiled `historical_kv_redundancy.cpp`
and relinked `libkvmem.so`; the server executable hash stayed the same while
the new runtime library SHA-256 became
`37ef7e0ad3fe53f7aa56078570e3d4e66915a56c867017b27bbc7d144a7f9081`.
The four targeted remote CTests for GC, redundancy, Query coverage and
thinking ranges passed 4/4. A first broad regex also selected an unbuilt
demo executable and marked it `Not Run`; the exact four-test selection is
the relevant passing result. The r2 runtime has not yet passed a model GC
quality run.

The IQ3 GGUF's native Jinja template explicitly defaults thinking effort to
`xhigh` and inserts an instruction to validate assumptions and alternatives.
It supports `low`, which asks for brief, focused thinking while preserving
the native thinking path. The experiment driver now accepts and records
`--reasoning-effort {low,medium,xhigh}` and passes it to the server only when
thinking is enabled. A dry run verified the exact `--enable-thinking
--reasoning-effort low` command and the no-thinking rejection; the updated
driver SHA-256 is
`e1ca1adf19c32a7a0e0396d7d69a4b60d59bde1595a6bd7762064d5088ade6c2`
locally and remotely. Future low-effort runs retain the teacher's 16K token
limit, model, block size, KV budget, sink and recent parameters. They are a
**separate sampling/template configuration**, not a retroactive pass for the
default-xhigh run.
An off-mode, one-turn low-effort smoke checked that the native model template
accepts the setting before the full eight-turn off/on runs.
The smoke completed HTTP 200/stop in 696 generated tokens, with 1,068
reasoning characters and 678 code characters in `state.js`; it verified that
low effort still uses the native thinking path. The original default-xhigh
first turn of the same task took 2,868 tokens. This one-turn probe is not a
game-quality result. The eight-turn off-mode baseline was then started with
`--reasoning-effort low` and strict prefix reuse, preserving all teacher
model, token-budget, block, sink and recent settings.

The explicit-low off-mode baseline subsequently completed all eight turns
with HTTP 200 and `finish_reason=stop`, no repair attempts, and 13,095 total
generated tokens. Independent prefix inspection passed all eight requests:
each continuation reused the exact preceding token prefix, with no cache
reset or refusal. The isolated browser checks passed **29/29**. Local copies
are `off-low-full-r1-summary.json`, `off-low-full-r1-prefix-final.json`,
`off-low-full-r1-browser-checks.json`, and `off-low-full-r1-config.json` in
this directory. This establishes a complete game-quality baseline for native
low thinking, not the effect of GC.

The first low-effort GC attempt configured only 384 CPU slots and received
HTTP 503 before the first decode: historical capture could not reserve enough
RAM or CPU cleanup capacity. Admission reserves for the full possible 16,384
output tokens, independent of the observed low-effort output length. With the
443-token first prompt and 32-token blocks, the minimum reservation alone is
`ceil((443 + 16384) / 32) = 526` slots, so 384 slots was an invalid experiment
configuration rather than a model-quality failure. That request-private
reservation also decreases the Ready space visible to GC. A replacement
eight-turn GC run, `redundancy-gc/gc-low-full-r2`, uses the prior viable
1,024-slot CPU pool and low/high watermarks 352/384, the r2 witness-fallback
library, and the same native low-effort game workload. Acceptance still
requires eight completed turns, 29/29 browser checks, exact prefix reuse,
committed logical GC, released host payload, physical reclamation, and a
post-commit active selection.

That r2 GC run **passed the eight-turn quality case**: eight HTTP 200/stop
responses, no repair requests, all seven continuations reusing the entire
previously cached token prefix, and **29/29 isolated Edge browser checks**.
Its native low-effort thinking path was active; the maximum observed scoring
round used 55 features, within the 56-feature cap. Independent full-log
checkers passed exact prefix continuity and decode-gap ordering. The latter
covered 12,741 decode boundaries, 75 prepared batches, 236 committed intents,
eight active windows (four after a commit), with a 3.831 microsecond p95 and
5.933 microsecond maximum foreground commit duration; no commit exceeded 10
milliseconds. Final GC counters were 236 committed, 236 early CPU payload
releases, 222 reclaimed slots, zero policy or worker failures. Release and
reclamation began before the final turn, and later turns continued with full
prefix reuse. The controlled off-mode baseline also passed 29/29, so this
quality case did not reveal a functional regression under the two tested
settings. It does not establish semantic equivalence of every generated line
or a general guarantee for other workloads.
The run's 783 process-memory samples showed a peak resident set of about
11,936 MiB (including model load) and a final sampled resident set of about
6,289 MiB; 782 GPU samples peaked at 13,910 MiB on the 16,384 MiB device.
These are observed bounds for this finite run, not a proof of a long-term
memory upper bound.

Local evidence copies are `gc-low-full-r2-summary.json`,
`gc-low-full-r2-config.json`, `gc-low-full-r2-prefix-final.json`,
`gc-low-full-r2-decode-gap-final.json`, and
`gc-low-full-r2-browser-checks.json`. The full 18 MB raw run, including
requests, responses, server trace, and generated game files, is retained in
`tmp/full-query-20261023/gc-low-full-r2/`; the original remains on the
remote host. The off quality control had 8,192 CPU slots because off mode
retains all history, whereas the GC arm had 1,024 with 352/384 watermarks.
This quality contrast therefore establishes completion under each viable
configuration, not a pure one-variable latency comparison.

After quality passed, the native full-Query paired benchmark completed on the
same r2 runtime library and IQ3 model: one warmup plus **five measured rounds
per mode** (`off`, `shadow`, `delete`), 18 complete requests in all. Its strict
parser passed full-Query coverage, bounded 56-feature thinking scoring,
generated-Query observations, identical output tokens, stable GPU windows,
zero worker/policy failures, and actual decode-period CPU payload release.
Each delete round retired and released 48 historical payloads during the 256
timed decode steps. It did not return any complete slots to `Ready` inside
those short timed windows; the longer browser-game run independently observed
222 reclaimed slots.

The **paired per-round** delete-versus-off median change was **+0.0533%** in
decode-step p95 (range -4.398% to +0.799%) and **+0.3891%** in tokens/second
(range -0.193% to +1.801%). Thus all five paired rounds and their median meet
the previously proposed p95 increase <=5% and throughput drop <=3% thresholds
on this fixture. For reference, the unpaired median mode summaries were off
65.729 ms p95 / 16.033 tokens/s and delete 65.134 ms / 16.168 tokens/s;
those ratios are not the paired statistic and should not replace it. The
shadow arm is reported in the saved summary as an extra decomposition of
policy scoring without deletion.

The benchmark uses a fixed 6,144-token synthetic input, 256 greedy decode
steps, a 65,536-token context, and the same 36,864-token GPU budget,
16,384-token generation reservation, 1,024 sink, zero recent, 32-token
blocks, and 1,024-slot CPU pool. Prompt positions are synthetically labeled
thinking; generated positions are not, while Query observation continues.
All modes collect Query. It measures greedy sampling, `llama_decode`, and
synchronization, excluding HTTP token streaming and GC-enabled prefill. It
does not prove latency under arbitrary long-running agent workloads or a
statistical performance guarantee; GPU power/clocks were not pinned.
Reproducible source data and parser output are in
`full-query-bench-r2/native.log.gz` (gzip of the unchanged raw log),
`full-query-bench-r2/runs.json`, and `full-query-bench-r2/summary.json` here.
Decompress the log before passing it to
`scripts/summarize_historical_gc_bench.py --require-full-query`. The raw
uncompressed log SHA-256 is
`35eaf971eb78a58892f1ffebe2c9313229f270856d0710fc4acb99cde3a487ec`.
