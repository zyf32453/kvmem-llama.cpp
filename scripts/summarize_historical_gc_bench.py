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


FULL_QUERY_FIXTURE = "mentor-full-query-synthetic-thinking"


def validate_full_query(config, rows, lines, require_full_query):
    """Join independently emitted capture evidence to every timed request."""
    full_configs = [json.loads(line.split(" ", 1)[1]) for line in lines
                    if line.startswith("KVMEM_GC_FULL_QUERY_CONFIG ")]
    evidence = [json.loads(line.split(" ", 1)[1]) for line in lines
                if line.startswith("KVMEM_GC_FULL_QUERY ")]
    is_full_query = config.get("fixture") == FULL_QUERY_FIXTURE
    if not is_full_query:
        if require_full_query:
            raise ValueError("full Query benchmark required; legacy fixture is not evidence for this path")
        if full_configs or evidence:
            raise ValueError("full Query metadata does not match the benchmark fixture")
        return {"status": "not_exercised", "reason": "legacy fixture has no full Query/thinking-path gate"}
    if len(full_configs) != 1:
        raise ValueError("expected exactly one full Query configuration")
    full_config = full_configs[0]
    input_tokens = config.get("input_tokens")
    steps = config["decode_steps"]
    if (type(input_tokens) is not int or input_tokens <= 32 or config.get("block_tokens") != 32 or
            config.get("policy") != "redundancy" or full_config.get("fixture") != FULL_QUERY_FIXTURE or
            full_config.get("full_query_span") != [0, input_tokens] or
            full_config.get("synthetic_thinking_span") != [0, input_tokens] or
            full_config.get("thinking_pool_limit") != 56 or
            full_config.get("thinking_span_is_synthetic") is not True or
            full_config.get("generated_query_capture") is not True or
            full_config.get("generated_tokens_marked_thinking") is not False or
            full_config.get("off_thinking_policy_armed") is not False or
            full_config.get("off_includes_query_collection") is not True or
            full_config.get("worker_starts_after_prefill") is not True or
            full_config.get("task_quality_evaluation") is not False):
        raise ValueError("invalid or incomplete full Query fixture configuration")
    keys = {(row["round"], row["mode"]) for row in rows}
    if len(evidence) != len(keys) or {(item.get("round"), item.get("mode")) for item in evidence} != keys:
        raise ValueError("missing or duplicated per-request full Query evidence")
    by_key = {(item["round"], item["mode"]): item for item in evidence}
    counters = (
        "question_tokens", "question_segments", "question_bytes", "query_samples", "invalid_questions",
        "observation_samples_before_decode", "observation_samples_after_decode",
        "fast_samples_before_decode", "fast_samples_after_decode", "question_score_evals", "question_score_yields",
        "feature_blocks", "pair_comparisons", "scored_rounds", "max_round_features",
    )
    for row in rows:
        item = by_key[row["round"], row["mode"]]
        if (any(type(item.get(key)) is not int or item[key] < 0 for key in counters) or
                item.get("fixture") != FULL_QUERY_FIXTURE or
                item.get("thinking_span_is_synthetic") is not True or
                item.get("generated_tokens_marked_thinking") is not False or
                item.get("thinking_policy_armed") is not (row["mode"] != "off") or
                item.get("full_question_coverage") is not True or
                item.get("warmup") is not row["warmup"] or item.get("thinking_pool_limit") != 56):
            raise ValueError("invalid or incomplete per-request full Query evidence")
        if (item["question_tokens"] != input_tokens or item["question_segments"] != (input_tokens + 31) // 32 or
                not item["question_bytes"] or not item["query_samples"] or item["invalid_questions"] or
                item["observation_samples_after_decode"] <= item["observation_samples_before_decode"] or
                item["fast_samples_after_decode"] <= item["fast_samples_before_decode"]):
            raise ValueError("full Query input coverage or generated observation gate failed")
        worker_keys = ("question_score_evals", "question_score_yields", "feature_blocks",
                       "pair_comparisons", "scored_rounds", "max_round_features")
        if row["mode"] == "off":
            if any(item[key] for key in worker_keys) or any(row[key] for key in (
                    "committed_after_decode", "shadow_after_decode", "host_released_after_decode",
                    "slots_reclaimed_after_decode", "active_release_steps")):
                raise ValueError("full Query off baseline ran the worker or released CPU history")
        elif any(item[key] == 0 for key in worker_keys) or item["max_round_features"] > 56:
            raise ValueError("full Query worker did not exercise bounded scoring and the thinking pool")
        if row["mode"] == "delete":
            if (not 0 < row["active_release_steps"] <= steps or not 0 < row["first_release_step"] <= steps or
                    (row["host_released_after_decode"] <= row["host_released_before_decode"] and
                     row["slots_reclaimed_after_decode"] <= row["slots_reclaimed_before_decode"])):
                raise ValueError("full Query delete has no physical CPU release during timed decoding")
        elif row["mode"] == "shadow" and any(row[key] for key in (
                "host_released_after_decode", "slots_reclaimed_after_decode", "active_release_steps")):
            raise ValueError("full Query shadow unexpectedly released CPU history")
        row["full_query_evidence"] = item
    return {
        "status": "passed", "configuration": full_config, "verified_requests_including_warmups": len(evidence),
        "question_tokens_per_request": input_tokens,
        "max_round_features": max(item["max_round_features"] for item in evidence),
        "minimum_generated_observation_samples": min(item["observation_samples_after_decode"] -
                                                     item["observation_samples_before_decode"] for item in evidence),
        "minimum_generated_fast_samples": min(item["fast_samples_after_decode"] -
                                               item["fast_samples_before_decode"] for item in evidence),
    }


def summarize(path, require_full_query=False):
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
    full_query_validation = validate_full_query(config, rows, lines, require_full_query)
    measured = [row for row in rows if not row["warmup"]]
    metrics = ("tokens_per_second", "generation_ms", "p50_step_ms", "p95_step_ms", "p99_step_ms", "prefill_ms")
    result = {
        "configuration": config,
        "log_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "output_tokens_sha256": hashlib.sha256(json.dumps(reference).encode()).hexdigest(),
        "all_outputs_identical": True,
        "all_gpu_windows_stable": True,
        "full_query_validation": full_query_validation,
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
    if full_query_validation["status"] == "passed":
        result["limitations"].append(
            "Only prompt positions are synthetically labeled as thinking. Timed generated positions have no "
            "thinking annotation, verifying that Query observation continues outside thinking. This verifies "
            "bounded thinking eligibility and full/generated Query capture, not actual model reasoning semantics. "
            "off includes Query collection, so paired timings isolate incremental background policy/GC cost.")
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
    parser.add_argument("--require-full-query", action="store_true",
                        help="reject legacy fixtures and require complete per-request full Query/thinking-path evidence")
    args = parser.parse_args()
    summary, rows = summarize(args.log, require_full_query=args.require_full_query)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output / "runs.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps({"full_query_validation": summary["full_query_validation"],
                      "modes": {key: value["median"] for key, value in summary["modes"].items()},
                      "paired_changes_percent": summary["paired_changes_percent"]}, indent=2))
