#!/usr/bin/env python3
"""Check logical GC retirement ordering in a --kvmem-trace server log."""

import argparse
import json
import re
from pathlib import Path


NUMBER = re.compile(r"([a-z_]+)=(-?\d+)")
PHASE = re.compile(r"phase=([a-z_]+)")


def analyze(lines, require_commit=False, require_selection=False):
    boundaries = []
    selections = []
    errors = []
    committed_ends = set()
    decode_count = 0
    last_decode_return_ns = 0
    last_decode_enter_ns = 0
    in_decode = False
    for line_no, line in enumerate(lines, 1):
        if "KVMEM_TRACE gc_gap phase=" in line:
            phase_match = PHASE.search(line)
            if not phase_match:
                errors.append(f"line {line_no}: missing phase")
                continue
            phase = phase_match.group(1)
            values = {name: int(value) for name, value in NUMBER.findall(line)}
            if phase == "decode_begin":
                enter = values.get("decode_enter_ns", 0)
                if not enter:
                    errors.append(f"line {line_no}: missing decode entry timestamp")
                if in_decode:
                    errors.append(f"line {line_no}: decode began before the previous one ended")
                if enter < last_decode_return_ns:
                    errors.append(f"line {line_no}: decode entry preceded previous return")
                last_decode_enter_ns = enter
                in_decode = True
                done = 0
                start = values.get("commit_start_ns", 0)
                end = values.get("commit_end_ns", 0)
            elif phase == "decode_end":
                decode_count += 1
                done = values.get("decode_return_ns", 0)
                start = values.get("commit_start_ns", 0)
                end = values.get("commit_end_ns", 0)
                if not in_decode:
                    errors.append(f"line {line_no}: decode return has no entry")
                if not done:
                    errors.append(f"line {line_no}: missing decode return timestamp")
                if done < last_decode_enter_ns:
                    errors.append(f"line {line_no}: decode returned before it began")
                if done < last_decode_return_ns:
                    errors.append(f"line {line_no}: decode timestamps moved backward")
                last_decode_return_ns = done
                in_decode = False
            elif phase == "request_begin":
                if in_decode:
                    errors.append(f"line {line_no}: request admission committed during decode")
                done = 0
                start = values.get("start_ns", 0)
                end = values.get("end_ns", 0)
            else:
                errors.append(f"line {line_no}: unknown phase {phase}")
                continue
            prepared = values.get("prepared", 0)
            committed = values.get("committed", 0)
            if committed > prepared:
                errors.append(f"line {line_no}: more commits than prepared intents")
            if prepared:
                if not start or end < start:
                    errors.append(f"line {line_no}: invalid commit timestamp")
                if done and start < done:
                    errors.append(f"line {line_no}: retirement preceded decode completion")
                if phase != "decode_end" and last_decode_return_ns and start < last_decode_return_ns:
                    errors.append(f"line {line_no}: preselection retirement preceded last decode completion")
                if phase == "decode_begin" and end > last_decode_enter_ns:
                    errors.append(f"line {line_no}: retirement overlapped the next decode")
                if committed:
                    committed_ends.add(end)
                boundaries.append({"phase": phase, "prepared": prepared,
                                   "committed": committed, "duration_us": (end - start) / 1000})
        elif "KVMEM_TRACE historical_active_window_begin " in line:
            values = {name: int(value) for name, value in NUMBER.findall(line)}
            selected = values.get("select_ns", 0)
            last = values.get("last_gc_commit_ns", 0)
            if not selected:
                errors.append(f"line {line_no}: active selection has no timestamp")
            if last and (last not in committed_ends or selected < last):
                errors.append(f"line {line_no}: active selection did not follow its committed retirement")
            selections.append({"select_ns": selected, "last_gc_commit_ns": last,
                               "skips": values.get("skips", 0)})
    if in_decode:
        errors.append("last decode has no return boundary")
    durations = sorted(item["duration_us"] for item in boundaries)
    p95 = durations[max(0, (95 * len(durations) + 99) // 100 - 1)] if durations else None
    if require_commit and not committed_ends:
        errors.append("no committed decode-boundary retirement observed")
    if require_selection and not any(s["last_gc_commit_ns"] for s in selections):
        errors.append("no active selection after a committed retirement observed")
    return {
        "passed": not errors,
        "errors": errors,
        "decode_boundaries": decode_count,
        "prepared_batches": len(boundaries),
        "committed_intents": sum(item["committed"] for item in boundaries),
        "active_windows": len(selections),
        "windows_after_commit": sum(bool(s["last_gc_commit_ns"]) for s in selections),
        "commit_duration_p95_us": p95,
        "commit_duration_max_us": max(durations) if durations else None,
        "commit_durations_over_10ms": sum(value > 10000 for value in durations),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--require-commit", action="store_true")
    parser.add_argument("--require-selection", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with args.log.open(encoding="utf-8", errors="replace") as source:
        result = analyze(source, args.require_commit, args.require_selection)
    report = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(report + "\n", encoding="utf-8")
    else:
        print(report)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
