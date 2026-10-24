#!/usr/bin/env python3
"""Audit every continuation against the preceding turn's actual token IDs."""

import argparse
import json
import re
from pathlib import Path


ADMITTED = re.compile(r"KVMEM_HISTORICAL_HTTP admitted reset=(\d+) cached=(\d+)")


def analyze(rows, log_lines, expected_turns):
    errors = []
    admissions = []
    refusals = 0
    for line in log_lines:
        match = ADMITTED.search(line)
        if match:
            admissions.append((int(match[1]), int(match[2])))
        if "KVMEM_HISTORICAL_HTTP refused" in line:
            refusals += 1
    if len(rows) != expected_turns:
        errors.append(f"recorded {len(rows)} of {expected_turns} required turns")
    if len(admissions) != expected_turns:
        errors.append(f"observed {len(admissions)} of {expected_turns} admissions")
    if refusals:
        errors.append(f"observed {refusals} historical admission refusals")
    checked = []
    previous_total = 0
    for index, row in enumerate(rows):
        turn = index + 1
        usage = row.get("usage") or {}
        hit = usage.get("prompt_cache_hit_tokens")
        total = usage.get("total_tokens")
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        if row.get("turn") != turn or not row.get("done") or row.get("http_status") != 200 or row.get("finish_reason") != "stop":
            errors.append(f"turn {turn}: incomplete or non-stop response")
        if not all(type(value) is int and value >= 0 for value in (hit, total, prompt, completion)):
            errors.append(f"turn {turn}: incomplete usage")
            continue
        if total != prompt + completion:
            errors.append(f"turn {turn}: total usage does not equal prompt plus completion")
        if hit != previous_total:
            errors.append(f"turn {turn}: cache hit {hit} != preceding token count {previous_total}")
        if index < len(admissions):
            reset, cached = admissions[index]
            if reset or cached != previous_total:
                errors.append(f"turn {turn}: admission reset={reset}, cached={cached}, expected={previous_total}")
        checked.append({"turn": turn, "expected_cached_tokens": previous_total,
                        "observed_cache_hit_tokens": hit, "total_tokens": total})
        previous_total = total
    completed = sum(row.get("done") and row.get("http_status") == 200 and
                    row.get("finish_reason") == "stop" for row in rows)
    return {"passed": not errors, "errors": errors, "expected_turns": expected_turns,
            "result_rows": len(rows), "completed_turns": completed,
            "admissions": len(admissions), "refusals": refusals,
            "turns": checked}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("server_log", type=Path)
    parser.add_argument("--expected-turns", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.expected_turns < 1:
        parser.error("--expected-turns must be positive")
    rows = json.loads(args.results.read_text(encoding="utf-8"))
    with args.server_log.open(encoding="utf-8", errors="replace") as source:
        result = analyze(rows, source, args.expected_turns)
    report = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(report, encoding="utf-8")
    else:
        print(report, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
