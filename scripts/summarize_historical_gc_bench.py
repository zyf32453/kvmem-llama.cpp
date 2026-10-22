"""Summarize --gc-bench native timings; reject incomplete or unequal-work runs."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize(path):
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    configs = [json.loads(line.split(" ", 1)[1]) for line in lines
               if line.startswith("KVMEM_GC_BENCH_CONFIG ")]
    rows = [json.loads(line.split(" ", 1)[1]) for line in lines if line.startswith("KVMEM_GC_BENCH ")]
    if len(configs) != 1:
        raise ValueError("expected exactly one benchmark configuration")
    config = configs[0]
    expected = {(round_id, mode) for round_id in range(config["rounds"] + 1)
                for mode in ("off", "shadow", "delete")}
    if len(rows) != len(expected) or {(row["round"], row["mode"]) for row in rows} != expected:
        raise ValueError("incomplete or duplicated benchmark rounds")
    if not any(line.startswith("PASS GC decode benchmark:") for line in lines):
        raise ValueError("native benchmark did not reach its successful completion")
    reference = rows[0]["tokens"]
    for row in rows:
        if (not row["stable_window"] or not row["same_output"] or row["tokens"] != reference or
                row["policy_failures"] or row["worker_failures"] or
                row["decode_steps"] != config["decode_steps"] or
                len(row["step_ms"]) != config["decode_steps"] or
                row["warmup"] != (row["round"] == 0) or
                min(row["step_ms"]) <= 0 or row["generation_ms"] <= 0):
            raise ValueError("failed equal-work, timing, or worker checks")
        if row["mode"] == "delete" and (row["active_release_steps"] == 0 or
                row["committed_after_decode"] <= row["committed_before_decode"]):
            raise ValueError("delete did not actually retire and release during decoding")
        if row["mode"] == "shadow" and (row["shadow_after_decode"] == 0 or row["committed_after_decode"]):
            raise ValueError("shadow strategy did not run correctly")
        row["tokens_per_second"] = 1000 * row["decode_steps"] / row["generation_ms"]
        for label, fraction in (("p50_step_ms", .5), ("p95_step_ms", .95), ("p99_step_ms", .99)):
            row[label] = percentile(row["step_ms"], fraction)
    measured = [row for row in rows if not row["warmup"]]
    metrics = ("tokens_per_second", "generation_ms", "p50_step_ms", "p95_step_ms", "p99_step_ms", "prefill_ms")
    result = {
        "configuration": config,
        "log_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "output_tokens_sha256": hashlib.sha256(json.dumps(reference).encode()).hexdigest(),
        "all_outputs_identical": True,
        "all_gpu_windows_stable": True,
        "measured_requests": len(measured),
        "modes": {},
        "paired_changes_percent": {},
        "method": "One shared model/context; one warmup per mode; rotating order. GC starts only after prefill. Mode summaries are medians of per-run statistics; paired changes compare the same round. Native steps include greedy sampling, llama_decode and synchronize. No HTTP timings.",
        "limitations": [
            "Synthetic stable-window fixture; not a long-context, request-admission, HTTP or quality benchmark.",
            "GC during prefill is excluded. Two earlier 256-step fixtures lacked worst-case short-span GPU slots; their output diverged and they were rejected. The final fixture checks full token residency after every step, not just the request window ID.",
            "off retains historical capture/ownership: this isolates incremental GC overhead, not total historical KV overhead.",
            "Finite retirement target, not sustained maximal release pressure; GPU power/clocks are not pinned.",
            "Small sample; any performance target must be stated separately. Tiny differences can be noise.",
            "The test binary retains its ordinary operator-new counting hook in all modes; product libraries are unchanged.",
        ],
    }
    for mode in ("off", "shadow", "delete"):
        subset = [row for row in measured if row["mode"] == mode]
        result["modes"][mode] = {
            "samples": len(subset),
            "median": {metric: statistics.median(row[metric] for row in subset) for metric in metrics},
            "range": {metric: [min(row[metric] for row in subset), max(row[metric] for row in subset)]
                      for metric in metrics},
            "decode_retirements": [row["committed_after_decode"] - row["committed_before_decode"] for row in subset],
            "decode_cpu_early_releases": [row["host_released_after_decode"] - row["host_released_before_decode"] for row in subset],
            "decode_slot_reclaims": [row["slots_reclaimed_after_decode"] - row["slots_reclaimed_before_decode"] for row in subset],
        }
    for mode in ("shadow", "delete"):
        changes = {metric: [] for metric in metrics}
        for round_id in range(1, config["rounds"] + 1):
            baseline = next(row for row in measured if row["round"] == round_id and row["mode"] == "off")
            target = next(row for row in measured if row["round"] == round_id and row["mode"] == mode)
            for metric in metrics:
                changes[metric].append(100 * (target[metric] / baseline[metric] - 1))
        result["paired_changes_percent"][mode] = {
            metric: {"median": statistics.median(values), "range": [min(values), max(values)], "rounds": values}
            for metric, values in changes.items()
        }
    return result, rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary, rows = summarize(args.log)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output / "runs.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps({"modes": {key: value["median"] for key, value in summary["modes"].items()},
                      "paired_changes_percent": summary["paired_changes_percent"]}, indent=2))
