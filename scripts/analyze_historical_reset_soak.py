#!/usr/bin/env python3
"""Audit the frozen RAM-only historical reset/cancel hour-long trajectory.

Input: model.log, resources.jsonl, and (for final acceptance) summary.json
and driver.exit from the sampling driver. Writes analysis.json beside them.
This checks lifecycle accounting and sampled resources, not full process
memory bounds, coding quality, or the outstanding intermittent CUDA failure.
Use --partial for a live snapshot; it can never declare acceptance.
"""
import argparse
import csv
import json
import re
import statistics
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("directory", type=Path)
parser.add_argument("--partial", action="store_true")
args = parser.parse_args()
if not __debug__:
    parser.error("Do not use Python -O: this evidence audit requires assertions")
root = args.directory
lines = (root / "model.log").read_text(encoding="utf-8", errors="replace").splitlines()
records = []
for index, line in enumerate(lines):
    if not line.startswith("KVMEM_RESET_SOAK "):
        continue
    if args.partial and index == len(lines) - 1 and not line.endswith("}"):
        continue
    records.append(json.loads(line.removeprefix("KVMEM_RESET_SOAK ")))
assert records, "No lifecycle samples"
assert not any(line.startswith("FAIL historical") or any(marker in line for marker in (
    "GGML_CUDA_ERROR", "CUDA error:", "an illegal instruction was encountered",
    "an illegal memory access was encountered", "GGML_ASSERT(",
)) for line in lines), "Model log contains a failure"
initial_resources = {}
for prefix in ("KVMEM_BACKEND_RESERVE", "KVMEM_CELL_INDEX", "KVMEM_SLOT_INFO",
               "KVMEM_UBATCH_POOL", "KVMEM_BATCH_CONTEXT", "KVMEM_STAGEIN_WORKSPACE",
               "KVMEM_CAPTURE_ARENA", "KVMEM_CAPTURE_WORKSPACE",
               "KVMEM_HISTORICAL_CAPTURE begin", "KVMEM_HISTORICAL_PAYLOAD_BUDGET begin"):
    line = next(line for line in lines if line.startswith(prefix + " "))
    initial_resources[prefix] = {key: int(value) for key, value in
                                 re.findall(r"(\w+)=(\d+)(?:\s|$)", line)}
capture = initial_resources["KVMEM_HISTORICAL_CAPTURE begin"]
payload = initial_resources["KVMEM_HISTORICAL_PAYLOAD_BUDGET begin"]
workspace = initial_resources["KVMEM_CAPTURE_WORKSPACE"]
backend = initial_resources["KVMEM_BACKEND_RESERVE"]
assert capture["reserved"] <= capture["capacity"]
assert capture["payload_bytes"] == payload["used"] <= payload["limit"]
assert payload["limit"] % (2 * capture["capacity"]) == 0
slot_payload_bytes = payload["limit"] // (2 * capture["capacity"])
assert capture["payload_bytes"] == capture["reserved"] * slot_payload_bytes
# device_bytes includes layout_bytes, not a separate additional allocation.
assert 0 <= workspace["layout_bytes"] <= workspace["device_bytes"]
# Frozen RAM configuration still owns two real 32 MiB D2H pinned slabs.
assert capture["pinned_bytes"] == 0 and workspace["pinned_bytes"] == 64 * 1024 * 1024
assert workspace["jobs_per_batch"] == 8192
assert records[0]["shared_payload_limit"] == payload["limit"]
assert records[0]["arena_bytes"] == initial_resources["KVMEM_CAPTURE_ARENA"]["bytes"]
assert records[0]["backend_output_bytes"] == backend["output_bytes"]
assert records[0]["graph_meta_bytes"] == backend["graph_meta_bytes"]
fixed_fields = ("shared_payload_limit", "table_bytes", "row_capacity", "arena_bytes",
                "backend_output_bytes", "graph_meta_bytes", "cleanup_capacity")
for field in fixed_fields:
    assert len({record[field] for record in records}) == 1, f"{field} changed"
for record in records:
    count = record["requests"]
    assert record["finished"] == (count + 2) // 3
    assert record["prefill_cancelled"] == (count + 1) // 3
    assert record["generation_cancelled"] == count // 3
    assert record["tokens"] == (67 * record["finished"] + 32 * record["prefill_cancelled"]
                               + 65 * record["generation_cancelled"])
    assert record["generated"] == 3 * record["finished"] + record["generation_cancelled"]
    assert record["old_domains_drained"] + record["old_domains_pending"] + 1 == count
    assert 0 <= record["shared_payload_bytes"] <= record["shared_payload_limit"]
    assert record["shared_payload_bytes"] <= record["sampled_peak_payload"] <= record["shared_payload_limit"]
    assert 0 <= record["cleanup_occupied"] <= record["cleanup_capacity"]
    assert 0 <= record["directory_entries"] <= capture["capacity"]
for previous, current in zip(records, records[1:]):
    for field in ("seconds", "requests", "old_domains_drained", "admission_refusals", "sampled_peak_payload",
                  "decode_cpp_allocations", "decode_cpp_allocated_bytes"):
        assert current[field] >= previous[field], f"{field} moved backwards"
finals = [record for record in records if record["final"]]
if not args.partial:
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    assert (root / "driver.exit").read_text().strip() == "0"
    assert summary["exit_code"] == 0 and not summary["timed_out"] and summary["passed_lifecycle_gate"]
    assert summary["samples"] == len(records), "Summary and raw sample count disagree"
    assert len(finals) == 1 and finals[0] == records[-1] == summary["final"]
    assert records[-1]["seconds"] >= 3600 and records[-1]["old_domains_pending"] == 0
    assert any(line.startswith("PASS reset soak:") for line in lines)
    assert any(line.startswith("PASS historical backend guard:") for line in lines)

def range_of(rows, field):
    values = [row[field] for row in rows]
    return {"min": min(values), "max": max(values), "first": values[0], "last": values[-1]}


# These samples begin after model load. Loading peak remains separately visible
# in VmHWM; a stable post-load RSS is not a proof covering every allocation.
steady = [record for record in records if record["seconds"] >= 60]
rss = {"unit": "KiB", "post_load": range_of(records, "rss_kib"),
       "lifetime_high_water": max(record["peak_rss_kib"] for record in records)}
if len(steady) >= 2:
    mx = statistics.mean(record["seconds"] for record in steady)
    my = statistics.mean(record["rss_kib"] for record in steady)
    denominator = sum((record["seconds"] - mx) ** 2 for record in steady)
    if denominator:
        rss["descriptive_slope_MiB_per_hour_after_60s"] = round(
            sum((record["seconds"] - mx) * (record["rss_kib"] - my)
                for record in steady) / denominator * 3600 / 1024, 3)
    if records[-1]["seconds"] >= 1260:
        early = [record["rss_kib"] for record in steady if record["seconds"] <= 660]
        late = [record["rss_kib"] for record in steady
                if record["seconds"] >= records[-1]["seconds"] - 600]
        rss["median_first_10min_after_60s_KiB"] = statistics.median(early)
        rss["median_final_10min_KiB"] = statistics.median(late)

resource_text = (root / "resources.jsonl").read_text(encoding="utf-8")
resource_lines = resource_text.splitlines()
if args.partial and resource_lines and not resource_text.endswith("\n"):
    resource_lines.pop()  # A live snapshot may end in one incomplete write.
resources = [json.loads(line) for line in resource_lines]
gpu = []
gpu_failures = 0
for sample in resources:
    observation = sample["gpu_device_wide"]
    if observation.get("exit_code") != 0:
        gpu_failures += 1
        continue
    row = next(csv.reader([observation["stdout"]]))
    if len(row) != 5 or not row[1].strip().isdigit():
        gpu_failures += 1
        continue
    gpu.append(int(row[1].strip()))
assert resources, "No external process/device samples"
result = {
    "status": "partial_observation_not_acceptance" if args.partial else "reset_lifecycle_gate_passed",
    "sample_count": len(records), "final_or_latest": records[-1],
    "initial_resource_inventory": initial_resources,
    "slot_payload_bytes": slot_payload_bytes,
    "fixed_fields": {field: records[0][field] for field in fixed_fields},
    "payload_bytes": range_of(records, "shared_payload_bytes"),
    "sampled_payload_high_water": max(record["sampled_peak_payload"] for record in records),
    "old_domains_pending": range_of(records, "old_domains_pending"),
    "cleanup_occupied": range_of(records, "cleanup_occupied"),
    "rss": rss,
    "external_samples": len(resources), "gpu_sampling_failures": gpu_failures,
    "device_wide_gpu_used_MiB_including_loading": {"min": min(gpu), "max": max(gpu)} if gpu else None,
    "limitations": [
        "Results cover only this reset/cancel lifecycle and sampled resources, not full production readiness.",
        "RSS trends are descriptive; no accepted regression threshold or complete memory upper bound is inferred.",
        "Payload quota excludes metadata, allocator overhead, staging and model buffers.",
        "Device-wide GPU samples include other processes; ordinary allocation counters are cumulative.",
        "Current domain remains held by the context at the final in-test sample.",
    ],
}
(root / "analysis.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
