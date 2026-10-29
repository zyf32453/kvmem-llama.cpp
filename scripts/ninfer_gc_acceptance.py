#!/usr/bin/env python3
"""Run matched, isolated NInfer GC smoke requests and retain timing evidence."""

import argparse
import json
import math
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


def command(*args):
    return subprocess.run(args, text=True, capture_output=True, check=True).stdout.strip()


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def request_prompt(kind, stress_decode=False):
    records = []
    for index in range(280):
        if kind == "needle" and index == 127:
            records.append(f"Record {index:04d}: The only hidden code is AMBER-4729. Remember this exact code.\n")
        else:
            records.append(
                f"Record {index:04d}: The riverstone ledger stores neutral archival context; "
                "this line has no bearing on the final calculation.\n"
            )
    ending = {
        "math": "After reading every record, compute 17 * 35 + 4. Reply with only the integer.",
        "needle": "What is the only hidden code in the records? Reply with only that code.",
        "sequence": ("Write the integers from 1 through 2000, one number per line. No explanation."
                     if stress_decode else
                     "Write the integers from 1 through 80, one number per line. No explanation."),
    }[kind]
    return "".join(records) + ending


def run_request(port, kind, stress_decode=False, stress_tokens=4096):
    payload = {
        "model": "qwen3.8-27b",
        "messages": [{"role": "user", "content": request_prompt(kind, stress_decode)}],
        "max_tokens": stress_tokens if stress_decode else (384 if kind == "sequence" else 160),
        "thinking_budget": 128,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "timings_per_token": True,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    chunks = []
    reasoning = []
    answer = []
    observations = []
    usage = None
    finish_reason = None
    stream_error = None
    with urllib.request.urlopen(req, timeout=300) as response:
        status = response.status
        for raw in response:
            if not raw.startswith(b"data: "):
                continue
            data = raw[6:].strip()
            if data == b"[DONE]":
                break
            item = json.loads(data)
            chunks.append(item)
            if item.get("error"):
                stream_error = item["error"]
            if item.get("usage"):
                usage = item["usage"]
            timing = item.get("timings") or {}
            if "predicted_n" in timing and "predicted_ms" in timing:
                observation = (int(timing["predicted_n"]), float(timing["predicted_ms"]))
                if not observations or observation[0] > observations[-1][0]:
                    observations.append(observation)
            for choice in item.get("choices", []):
                delta = choice.get("delta") or {}
                reasoning.append(delta.get("reasoning_content") or "")
                answer.append(delta.get("content") or "")
                finish_reason = choice.get("finish_reason") or finish_reason
    intervals = [
        (n - old_n, ms - old_ms)
        for (old_n, old_ms), (n, ms) in zip(observations, observations[1:])
        if n > old_n and ms >= old_ms
    ]
    unit_ms = [elapsed for tokens, elapsed in intervals if tokens == 1]
    round_ms = [elapsed for _, elapsed in intervals]
    output = "".join(answer).strip()
    numbers = [int(x) for x in re.findall(r"\b\d+\b", output)] if kind == "sequence" else []
    expected = {
        "math": output == "599",
        "needle": output == "AMBER-4729",
        "sequence": (numbers[:100] == list(range(1, 101)) and len(numbers) >= 1000)
                    if stress_decode else numbers == list(range(1, 81)),
    }[kind]
    expected = expected and stream_error is None and finish_reason is not None
    return {
        "http_status": status,
        "stream_error": stream_error,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "usage": usage,
        "finish_reason": finish_reason,
        "answer": output,
        "reasoning_characters": len("".join(reasoning)),
        "exact_task_check": expected,
        "check_scope": ("first 100 numbers in order and at least 1000 numbers emitted"
                        if stress_decode else "full expected answer"),
        "timing_observations": observations,
        "timing_intervals": len(intervals),
        "unit_token_intervals": len(unit_ms),
        "multi_token_intervals": sum(tokens > 1 for tokens, _ in intervals),
        "unit_token_p95_ms": round(percentile(unit_ms, 0.95), 3) if unit_ms else None,
        "p95_is_complete": bool(intervals) and len(unit_ms) == len(intervals),
        "decode_round_p95_ms": round(percentile(round_ms, 0.95), 3) if round_ms else None,
        "sequence_numbers": len(numbers),
        "stream_chunk_count": len(chunks),
        "final_timings": chunks[-1].get("timings") if chunks else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--image", default="kvmem-ninfer-gc-builder:20261029")
    parser.add_argument("--cuda-dir", type=Path, default=Path("/usr/local/cuda-13.2"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--modes", nargs="+", choices=("off", "lru", "redundancy"),
                        default=["off", "lru", "redundancy"])
    parser.add_argument("--tasks", nargs="+", choices=("math", "needle", "sequence"),
                        default=["math", "needle", "sequence"])
    parser.add_argument("--stress-decode", action="store_true",
                        help="request long output with a 16K context to stress GC during decode")
    parser.add_argument("--stress-output-tokens", type=int, default=4096)
    parser.add_argument("--stress-host-mib", type=int, default=160)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    binary = args.root / "build-native/apps/ninfer-serve"
    model_file = args.model_dir / "Qwen3.8-27B-GSQ-RCO-IQ3_S-ninfer-v3.ninfer"
    if not binary.is_file() or not model_file.is_file():
        parser.error("NInfer binary or model file is missing")
    for mode in args.modes:
        for kind in args.tasks:
            port = 18110
            name = f"ninfer-gc-accept-{mode}-{kind}-{int(time.time())}"
            host_mib = 256 if mode == "off" else (args.stress_host_mib if args.stress_decode else 96)
            common = [
                "docker", "run", "-d", "--name", name, "--gpus", "all",
                "-p", f"127.0.0.1:{port}:{port}",
                "-v", f"{args.root}:/work",
                "-v", f"{args.model_dir}:/models:ro",
                "-v", f"{args.cuda_dir}:{args.cuda_dir}:ro",
                "-e", "KVMEM_GC_AUDIT=1", args.image,
                "/work/build-native/apps/ninfer-serve", f"/models/{model_file.name}",
                "--host", "0.0.0.0", "--port", str(port),
                "--kvmem-budget", "2048", "--kvmem-gen-reserve",
                str(args.stress_output_tokens) if args.stress_decode else "512",
                "--kvmem-host-mib", str(host_mib),
                "--kvmem-query-scorer", "meanq-cosine", "--kvmem-score-agg", "max",
                "--no-kvmem-lrfu", "--kvmem-graph-alpha", "0",
                "--kvmem-query-max-tokens", "8192", "--kvmem-query-replay-max-tokens", "8192",
                "--max-context", "16384" if args.stress_decode else "8192",
                "--kv-dtype", "rk4v4", "--max-concurrency", "1",
                "--prefill-chunk", "512", "--default-max-tokens",
                str(args.stress_output_tokens) if args.stress_decode else "384",
                "--default-thinking-budget", "128", "--spec", "mtp", "--draft-tokens", "4",
                "--lm-head-draft", "--device-profile", "off", "--kvmem-verify-transfers",
            ]
            if mode != "off":
                common.append(f"--kvmem-gc-{mode}")
            record = {"mode": mode, "task": kind, "host_mib": host_mib,
                      "stress_decode": args.stress_decode,
                      "stress_output_tokens": args.stress_output_tokens if args.stress_decode else None,
                      "container": name, "started_at_unix": time.time()}
            try:
                command(*common)
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline:
                    try:
                        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=1):
                            break
                    except (OSError, TimeoutError):
                        time.sleep(0.5)
                else:
                    raise TimeoutError("NInfer service did not become ready")
                record.update(run_request(port, kind, args.stress_decode,
                                          args.stress_output_tokens))
            except Exception as error:
                record["error"] = f"{type(error).__name__}: {error}"
            finally:
                logs = subprocess.run(["docker", "logs", name], capture_output=True, text=True)
                (args.out / f"{mode}-{kind}.server.log").write_text(
                    logs.stdout + logs.stderr, encoding="utf-8")
                match = re.findall(r"KVMEM_NATIVE_GC_STATE[^\n]*proposed=(\d+) committed=(\d+) reclaimed=(\d+)",
                                   logs.stdout + logs.stderr)
                record["gc_counters"] = dict(zip(("proposed", "committed", "reclaimed"),
                                                  map(int, match[-1]))) if match else None
                proposals = re.findall(r"KVMEM_GC_AUDIT event=proposed[^\n]*?reason=(\w+)",
                                       logs.stdout + logs.stderr)
                record["proposal_reasons"] = {reason: proposals.count(reason)
                                              for reason in sorted(set(proposals))}
                subprocess.run(["docker", "rm", "-f", name], capture_output=True, text=True)
                (args.out / f"{mode}-{kind}.json").write_text(
                    json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                print(json.dumps({key: record.get(key) for key in
                                  ("mode", "task", "http_status", "stream_error", "exact_task_check", "error",
                                   "elapsed_seconds", "gc_counters", "decode_round_p95_ms",
                                   "unit_token_p95_ms",
                                   "p95_is_complete")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
