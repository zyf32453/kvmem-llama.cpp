#!/usr/bin/env python3
"""Matched raw-token latency qualification; each case resets the model and Host KV."""

import argparse
import hashlib
import itertools
import json
import math
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


def run(*args):
    return subprocess.run(args, capture_output=True, text=True, check=True).stdout.strip()


def quantile(values, fraction):
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)] if values else None


def gpu_state(name):
    value = subprocess.run(["docker", "exec", name, "nvidia-smi",
        "--query-gpu=temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu",
        "--format=csv,noheader"], capture_output=True, text=True)
    return value.stdout.strip() if value.returncode == 0 else None


def warmup(port, tokens, directory, iteration=1):
    if not tokens:
        return
    # ReadWrite timing cases follow a non-retained, disjoint raw prefix. This
    # heats the same model without keeping warmup Host KV or prefix checkpoints.
    payload = dict(prompt="Continue this integer sequence, one number per line:\n1\n2\n3\n",
                   n_predict=tokens, temperature=0, seed=42, ignore_eos=True,
                   cache_prompt=False, stream=False)
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as response:
        (directory / f"warmup-{iteration:02d}-response.json").write_bytes(response.read())


def request(port, directory, records, output_tokens, timeout=600):
    context = "".join(
        f"Record {i:04d}: The riverstone ledger stores neutral archival context; "
        "this line has no bearing on the final calculation.\n" for i in range(records)
    )
    prompt = (
        "<|im_start|>user\n" + context +
        "Write the integers from 1 through 2000, one number per line. No explanation."
        "<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n\n1\n"
    )
    payload = dict(prompt=prompt, n_predict=output_tokens, temperature=0, seed=42,
                   ignore_eos=True, cache_prompt=True, stream=True, timings_per_token=True)
    encoded = json.dumps(payload).encode()
    (directory / "request.json").write_bytes(encoded)
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=encoded,
                                 headers={"Content-Type": "application/json"})
    started = time.monotonic()
    first = None
    observations, text, errors = [], [], []
    final = None
    with (directory / "stream.jsonl").open("w") as log:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            status = response.status
            for raw in response:
                if not raw.startswith(b"data: ") or raw[6:].strip() == b"[DONE]":
                    continue
                item = json.loads(raw[6:])
                arrived = time.monotonic() - started
                log.write(json.dumps(dict(arrived_seconds=arrived, event=item)) + "\n")
                if item.get("error"):
                    errors.append(item["error"])
                content = item.get("content", "")
                if content:
                    text.append(content)
                    if first is None:
                        first = arrived
                timing = item.get("timings") or {}
                if "predicted_n" in timing and "predicted_ms" in timing:
                    value = (int(timing["predicted_n"]), float(timing["predicted_ms"]))
                    if value[0] and (not observations or value[0] > observations[-1][0]):
                        observations.append(value)
                    final = timing
    intervals = [(n - old_n, ms - old_ms)
                 for (old_n, old_ms), (n, ms) in zip(observations, observations[1:])]
    single = [ms for n, ms in intervals if n == 1]
    rounds = [ms for _, ms in intervals]
    full = bool(observations) and observations[0][0] == 1 and observations[-1][0] == output_tokens
    full = full and len(single) == output_tokens - 1 and all(ms >= 0 for ms in single)
    tokens = sum(n for n, _ in intervals)
    elapsed_ms = sum(ms for _, ms in intervals)
    answer = "".join(text)
    return dict(http_status=status, stream_errors=errors, elapsed_seconds=time.monotonic() - started,
                ttft_seconds=first, request_sha256=hashlib.sha256(encoded).hexdigest(),
                observed_tokens=observations[-1][0] if observations else 0,
                timing_observations=observations, token_intervals=intervals,
                complete_single_token_coverage=full,
                decode_round_p95_ms=quantile(rounds, .95),
                single_token_p95_ms=quantile(single, .95),
                decode_tokens_per_second=1000 * tokens / elapsed_ms if elapsed_ms > 0 else None,
                final_timings=final, content=answer,
                expected_prefix=re.findall(r"\b\d+\b", answer)[:20] == [str(i) for i in range(2, 22)])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--spec", choices=("none", "mtp"), default="none")
    p.add_argument("--profile", choices=("micro", "teacher"), default="micro")
    p.add_argument("--modes", nargs="+", choices=("off", "lru", "redundancy"),
                   default=["off", "lru", "redundancy"])
    p.add_argument("--repetitions", type=int, default=6)
    p.add_argument("--records", type=int)
    p.add_argument("--output-tokens", type=int, default=2048)
    p.add_argument("--warmup-tokens", type=int, default=512)
    p.add_argument("--warmup-target-c", type=int, default=80)
    p.add_argument("--port", type=int, default=18111)
    args = p.parse_args()
    if args.records is None:
        args.records = 6660 if args.profile == "teacher" else 450
    if min(args.repetitions, args.records, args.output_tokens) < 1:
        p.error("counts must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    active, reserve, host, context = ((98304, 16384, 4760, 262144)
                                    if args.profile == "teacher" else (2048, 512, 320, 16384))
    plan = dict(profile=args.profile, host_mib=host, active_tokens=active,
                reserve_tokens=reserve, context_tokens=context,
                spec=args.spec, modes=args.modes, repetitions=args.repetitions, records=args.records,
                output_tokens=args.output_tokens, request_timeout_seconds=1800 if args.profile == "teacher" else 600,
                audit=False, verify_transfers=False,
                warmup_tokens=args.warmup_tokens, warmup_cache_prompt=False,
                warmup_target_c=args.warmup_target_c, warmup_max_repetitions=6,
                fresh_model_and_host_per_case=True,
                acceptance="Pooled decode p95 increase <=5%; aggregate decode throughput decrease <=3%; "
                           "disclose per-run worst change. Ordinary decode requires complete token timing coverage; "
                           "MTP measurements are decode-round latency.")
    (args.out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    orders = list(itertools.permutations(args.modes))
    results = []
    for repeat in range(args.repetitions):
        for mode in orders[repeat % len(orders)]:
            directory = args.out / f"{repeat + 1:02d}-{mode}"
            directory.mkdir(exist_ok=False)
            name = f"ninfer-latency-{args.spec}-{repeat + 1}-{mode}-{int(time.time())}"
            cmd = ["docker", "run", "-d", "--name", name, "--gpus", "all",
                   "-p", f"127.0.0.1:{args.port}:{args.port}",
                   "-v", f"{args.root}:/work", "-v", f"{args.model_dir}:/models:ro",
                   "-v", "/usr/local/cuda-13.2:/usr/local/cuda-13.2:ro", "-e", "KVMEM_GC_AUDIT=0",
                   "kvmem-ninfer-gc-builder:20261029", "/work/build-native/apps/ninfer-serve",
                   "/models/Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer",
                   "--host", "0.0.0.0", "--port", str(args.port),
                   "--kvmem", "--kvmem-budget", str(active), "--kvmem-gen-reserve", str(reserve),
                   "--kvmem-host-mib", str(host), "--max-context", str(context),
                   "--kvmem-query-scorer", "meanq-cosine", "--kvmem-score-agg", "max",
                   "--no-kvmem-lrfu", "--kvmem-graph-alpha", "0",
                   "--kvmem-query-max-tokens", "8192", "--kvmem-query-replay-max-tokens", "8192",
                   "--kv-dtype", "rk4v4", "--max-concurrency", "1", "--prefill-chunk", "512",
                   "--default-max-tokens", str(args.output_tokens), "--default-thinking-budget", "128",
                   "--spec", args.spec, "--device-profile", "off"]
            if args.profile == "teacher":
                cmd += ["--kvmem-sink-tokens", "1024", "--kvmem-recent-tokens", "64"]
            if args.spec == "mtp":
                cmd += ["--draft-tokens", "4", "--lm-head-draft"]
            if mode != "off":
                cmd += [f"--kvmem-gc-{mode}"]
            result = dict(repeat=repeat + 1, mode=mode, spec=args.spec, container=name, command=cmd)
            try:
                run(*cmd)
                deadline = time.monotonic() + 90
                while True:
                    try:
                        urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health", timeout=2).close()
                        break
                    except (OSError, TimeoutError):
                        if run("docker", "inspect", "--format", "{{.State.Running}}", name) != "true":
                            raise RuntimeError("model exited during startup; see server.log")
                        if time.monotonic() > deadline:
                            raise TimeoutError("model readiness")
                        time.sleep(.5)
                result["gpu_before_warmup"] = gpu_state(name)
                result["warmup_states"] = []
                for iteration in range(1, 7):
                    warmup(args.port, args.warmup_tokens, directory, iteration)
                    state = gpu_state(name)
                    result["warmup_states"].append(state)
                    if state and int(state.split(",")[0].strip()) >= args.warmup_target_c:
                        break
                else:
                    raise RuntimeError("GPU did not reach the declared warmup temperature")
                result["gpu_after_warmup"] = state
                result.update(request(args.port, directory, args.records, args.output_tokens,
                                      timeout=plan["request_timeout_seconds"]))
                result["gpu_after_measurement"] = gpu_state(name)
            except Exception as error:
                result["exception"] = repr(error)
            finally:
                time.sleep(.2)
                logs = subprocess.run(["docker", "logs", name], text=True, capture_output=True)
                log = logs.stdout + logs.stderr
                (directory / "server.log").write_text(log)
                summaries = re.findall(r"KVMEM_NATIVE_GC_SUMMARY ([^\n]+)", log)
                result["gc_summaries"] = [dict(re.findall(r"(\w+)=([^ ]+)", row)) for row in summaries]
                subprocess.run(["docker", "rm", "-f", name], capture_output=True)
                (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
                results.append(result)
                (args.out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
                print(json.dumps({key: result.get(key) for key in (
                    "repeat", "mode", "spec", "http_status", "exception", "observed_tokens",
                    "complete_single_token_coverage", "single_token_p95_ms", "decode_round_p95_ms",
                    "decode_tokens_per_second", "gc_summaries")}), flush=True)


if __name__ == "__main__":
    main()
