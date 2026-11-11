#!/usr/bin/env python3
"""Audit a supplied trajectory bundle and profile one task without prefix double-counting.

Bundle results are historical measurements, never results from the current
backend. No executable or instruction from the supplied archive is executed.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reconcile_trace(trace, kinds):
    previous = Counter(trace["initial"]["counts"])
    require(sum(previous.values()) == trace["initial"]["tokens"], "Initial count mismatch")
    output, incoming, rounds = Counter(), Counter(), []
    for ordinal, row in enumerate(trace["rounds"], 1):
        require(row["turn"] == ordinal, "Turn sequence is not contiguous")
        require(Counter(row["counts_before"]) == previous, "History counted twice or lost")
        require(sum(previous.values()) == row["prompt_tokens"], "Prompt frontier mismatch")
        cursor = row["prompt_tokens"]
        generated, feedback = Counter(), Counter()
        for event in row["events"]:
            require(event["kind"] in kinds, "Unknown token label")
            require(event["begin"] == cursor and event["end"] > cursor, "Overlapping or missing token span")
            require(event["tokens"] == event["end"] - cursor, "Token span size mismatch")
            require(event["origin"] in ("model_output", "following_input"), "Unknown origin")
            (generated if event["origin"] == "model_output" else feedback)[event["kind"]] += event["tokens"]
            cursor = event["end"]
        require(sum(generated.values()) == row["output_tokens"], "Accepted output count mismatch")
        require(sum(feedback.values()) == row["following_input_tokens"], "Following input count mismatch")
        added = generated + feedback
        require(added == Counter(row["counts_added"]), "Added token labels mismatch")
        require(previous + added == Counter(row["counts_after"]), "Cumulative count mismatch")
        require(cursor == row["context_after"] == sum(row["counts_after"].values()), "Final frontier mismatch")
        output.update(generated)
        incoming.update(feedback)
        previous = Counter(row["counts_after"])
        rounds.append(dict(turn=ordinal, outcome=row["outcome"], prompt_tokens=row["prompt_tokens"],
            output_tokens=row["output_tokens"], following_input_tokens=row["following_input_tokens"],
            context_tokens=row["context_after"], output_counts=dict(generated),
            input_counts=dict(feedback), context_counts=dict(previous)))
    return dict(initial_counts=dict(trace["initial"]["counts"]), output_counts=dict(output),
        input_counts=dict(incoming), final_context_counts=dict(previous), rounds=rounds)


def fractions(counts):
    total = sum(counts.values())
    return [dict(kind=k, tokens=v, fraction=v / total) for k, v in sorted(counts.items(), key=lambda x: -x[1])]


def compare_profiles(teacher, current):
    """Compare disjoint generated counts and one final full-context snapshot.

    Per-round prompt lengths are retained as separate observations. They are
    deliberately never added into a cumulative input denominator.
    """
    groups = dict(teacher['groups'])
    groups['structure'] = list(groups['structure']) + ['unexposed_control_tokens']
    groups['unknown'] = ['protocol_or_unclassified']
    if not current.get('latest_successful_input', {}).get('unmapped_fields'):
        groups['structure'].append('protocol_or_unclassified')
        groups['unknown'] = []
    group_for = {kind: group for group, kinds in groups.items() for kind in kinds}
    output = current['generated_detailed_counts']
    context = current.get('latest_logical_context', {}).get('counts')
    denominators = [('model_output', teacher['output_counts'], output)]
    if context:
        denominators.append(('final_logical_context', teacher['final_context_counts'], context))
    details, grouped = [], []
    for denominator, old, new in denominators:
        old_total, new_total = sum(old.values()), sum(new.values())
        require(old_total > 0 and new_total > 0, 'Empty comparison denominator')
        old_groups, new_groups = Counter(), Counter()
        for kind in set(old) | set(new):
            group = group_for.get(kind, 'unknown')
            old_groups[group] += old.get(kind, 0)
            new_groups[group] += new.get(kind, 0)
            details.append(dict(denominator=denominator, kind=kind, group=group,
                teacher_tokens=old.get(kind, 0), current_tokens=new.get(kind, 0),
                teacher_fraction=old.get(kind, 0)/old_total, current_fraction=new.get(kind, 0)/new_total,
                difference_percentage_points=100*(new.get(kind, 0)/new_total-old.get(kind, 0)/old_total)))
        for group in groups:
            if not old_groups[group] and not new_groups[group]:
                continue
            grouped.append(dict(denominator=denominator, group=group,
                teacher_tokens=old_groups[group], current_tokens=new_groups[group],
                teacher_fraction=old_groups[group]/old_total, current_fraction=new_groups[group]/new_total,
                difference_percentage_points=100*(new_groups[group]/new_total-old_groups[group]/old_total)))
    rounds = current['rounds']
    return dict(task=teacher['task'], current_project_run=True, detailed_rows=sorted(details, key=lambda r:(r['denominator'],r['kind'])),
        grouped_rows=grouped, teacher_rounds=len(teacher['rounds']), current_requests=current['requests'],
        teacher_generated_tokens=sum(teacher['output_counts'].values()), current_generated_tokens=sum(output.values()),
        teacher_context_tokens=sum(teacher['final_context_counts'].values()),
        current_context_tokens=sum(context.values()) if context else None,
        teacher_format_retries=sum(r['outcome']=='format_retry' for r in teacher['rounds']),
        current_format_retries=sum(not r['terminus_parser']['valid'] for r in rounds if 'terminus_parser' in r),
        detailed_coverage=dict(detailed=current['detailed_requests'], native_usage=current['native_usage_requests'],
            total_requests=current['requests']),
        context_alignment=current.get('latest_successful_input'), official_reward=current.get('official_reward'),
        agent_exception=current.get('agent_exception'), service_errors=current['service_errors'],
        limitations=['Single same-task observational comparison, not benchmark accuracy or a GC-policy causal experiment.',
            'Native reasoning-counter conventions can differ at control boundaries; small boundary classes are explicit.',
            'Final context is one complete latest successful input plus its accepted output; repeated prompts are not summed.',
            'No semantic summary class. Token proportions do not establish redundancy or safe deletion.'])


def save_comparison(teacher, current, out):
    result = compare_profiles(teacher, current)
    (out / 'comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    for name, rows in [('phase-shares', result['detailed_rows']), ('phase-groups', result['grouped_rows'])]:
        with (out / (name+'.csv')).open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    with (out / 'current-rounds.csv').open('w', encoding='utf-8-sig', newline='') as f:
        fields = ['turn','http_status','prompt_tokens','completion_tokens','thinking_tokens','parser_valid',
            'parsed_command_count','elapsed_seconds','logical_after_reply_tokens','input_unmapped_fields']
        writer=csv.DictWriter(f, fieldnames=fields);writer.writeheader()
        for turn,row in enumerate(current['rounds'], 1):
            usage = (row.get('prompt_tokens'),row.get('completion_tokens'))
            writer.writerow(dict(turn=turn,http_status=row.get('http_status'),prompt_tokens=usage[0],completion_tokens=usage[1],
                thinking_tokens=row.get('counts',{}).get('thinking'),parser_valid=row.get('terminus_parser',{}).get('valid'),
                parsed_command_count=row.get('terminus_parser',{}).get('parsed_command_count'),elapsed_seconds=row.get('elapsed_seconds'),
                logical_after_reply_tokens=sum(usage) if all(isinstance(v,int) for v in usage) else None,
                input_unmapped_fields=len(row['input_profile']['unmapped_fields']) if 'input_profile' in row else None))
    return result


def audit_bundle(path, task):
    import zipfile
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        roots = [n[:-len("MANIFEST.json")] for n in names if n.endswith("/MANIFEST.json")]
        require(len(roots) == 1, "Ambiguous archive root")
        root = roots[0]
        require(len(names) == len(set(names)), "Duplicate ZIP entries")
        for name in names:
            relative = PurePosixPath(name)
            require(not relative.is_absolute() and ".." not in relative.parts and "\\" not in name,
                    "Unsafe archive path")
        read = lambda rel: z.read(root + rel)
        load = lambda rel: json.loads(read(rel))
        manifest_bytes = read("MANIFEST.json")
        sums = read("SHA256SUMS").decode("utf-8").splitlines()
        manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
        require(any(line.split() == [manifest_hash, "MANIFEST.json"] for line in sums), "Manifest digest mismatch")
        manifest = json.loads(manifest_bytes)
        for entry in manifest["files"]:
            data = read(entry["path"])
            require(len(data) == entry["bytes"] and hashlib.sha256(data).hexdigest() == entry["sha256"],
                    "Bundle digest mismatch: " + entry["path"])
        index = next(v for v in load("index.json") if v["family"] == "terminalbench" and v["task_id"] == task)
        schema = load("docs/label-schema.json")
        data = load("analysis/token-growth/token-growth.json")
        selected = [v for v in data["traces"] if v["family"] == "terminalbench" and v["task_id"] == task]
        require(len(selected) == 1, "Task has no detailed accepted-token analysis in this bundle")
        report = reconcile_trace(selected[0], schema["exclusive_token_labels"])
        generation_ms, ttft_ms = [], []
        for row in report["rounds"]:
            native = load(f"examples/terminalbench/{task}/bridge/{row['turn']:05d}/native.json")
            require(len(native["token_ids"]) == row["output_tokens"] == native["usage"]["completion_tokens"],
                    "Native accepted-token ledger mismatch")
            require(native["usage"]["prompt_tokens"] == row["prompt_tokens"], "Native prompt count mismatch")
            generation_ms.append(native["metrics"]["generation_time_ms"])
            ttft_ms.append(native["metrics"]["time_to_first_token_ms"])
            row["native_generation_ms"] = generation_ms[-1]
            row["native_ttft_ms"] = ttft_ms[-1]
        messages = [json.loads(s) for s in read(index["trace_path"]).decode("utf-8").splitlines()]
        executions = [v["original"] for v in messages if v["record_type"] == "execution_event"]
        observed = [v["observed_seconds"] for v in executions if isinstance(v.get("observed_seconds"), (int, float))]
        profile = load("metadata/terminalbench-profile.json")
        report.update(schema_version=1, task=task, source="teacher_bundle_historical",
            current_project_run=False, bundle_path=str(path), bundle_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            integrity=dict(verified_files=len(manifest["files"]), manifest_sha256=manifest_hash),
            task_instruction_sha256=hashlib.sha256(read(index["task_path"])).hexdigest(),
            labels=schema["exclusive_token_labels"], groups=schema["token_groups"],
            fractions={"model_output": fractions(report["output_counts"]), "new_feedback": fractions(report["input_counts"]),
                       "final_logical_context": fractions(report["final_context_counts"])},
            teacher_profile=dict(model=profile["model"], model_format=profile["model_format"],
                active_tokens=profile["kvmem"]["select_budget"], gen_reserve_tokens=profile["kvmem"]["gen_budget"],
                host_tokens=profile["kvmem"]["repository_blocks"] * profile["kvmem"]["block_tokens"],
                block_tokens=profile["kvmem"]["block_tokens"], draft_tokens=profile["draft_tokens"],
                gc=profile["kvmem"]["historical_gc"], concurrency=profile["max_num_seqs"]),
            native_timing=dict(generation_seconds=sum(generation_ms) / 1000,
                sum_ttft_seconds=sum(ttft_ms) / 1000, execution_events=len(executions),
                observed_execution_seconds=sum(observed), observed_execution_events=len(observed),
                warning="Server timing and sparse session-level executor events do not form a complete wall-time partition; no per-phase time percentages inferred."),
            official_reward=None, official_reward_status="not_provided_by_bundle",
            limitations=["Historical data, not a new run of the current project.",
                "Accepted ID counts and saved exclusive labels verified; original tokenizer/protocol alignment not rerun.",
                "No semantic summary label; analysis and plan do not prove summary coverage.",
                "Logical history is not GPU-resident KV. Token share is not latency share.",
                "Only this task has been profiled here; this is not benchmark accuracy."])
        return report


def save_report(report, out):
    out.mkdir(parents=True, exist_ok=True)
    (out / "teacher-baseline.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (out / "teacher-phase-shares.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["denominator", "kind", "label", "tokens", "fraction"])
        writer.writeheader()
        for denominator, values in report["fractions"].items():
            for value in values:
                writer.writerow(dict(denominator=denominator, label=report["labels"][value["kind"]], **value))
    with (out / "teacher-rounds.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["turn", "outcome", "prompt_tokens", "output_tokens", "following_input_tokens", "context_tokens",
                  "native_generation_ms", "native_ttft_ms"]
        writer = csv.DictWriter(f, fieldnames=fields + list(report["labels"]))
        writer.writeheader()
        for row in report["rounds"]:
            writer.writerow({**{k: row[k] for k in fields}, **{k: row["output_counts"].get(k, 0) for k in report["labels"]}})


def align_terminus_captures(directory):
    """Use the installed official parser, never code from the supplied ZIP.

    Preserve original live profiles and write separate offline aligned files.
    This operation does not submit model requests or execute any commands.
    """
    import inspect
    import importlib.metadata
    from harbor.agents.terminus_2.terminus_json_plain_parser import TerminusJSONPlainParser
    from agent_phase_capture import object_spans, profile_response
    parser = TerminusJSONPlainParser()
    module = Path(inspect.getfile(TerminusJSONPlainParser))
    provenance = dict(harbor_version=importlib.metadata.version('harbor'), parser_path=str(module),
        parser_sha256=hashlib.sha256(module.read_bytes()).hexdigest(), source='installed_official_harbor')
    validity, audits = {}, []
    for folder in sorted(p.parent for p in directory.glob('*/response.json')):
        response = json.loads((folder / 'response.json').read_text(encoding='utf-8'))
        raw_profile = json.loads((folder / 'profile.json').read_text(encoding='utf-8'))
        if raw_profile.get('http_status') != 200 or response.get('error'):
            continue
        content = ((response.get('choices') or [{}])[0].get('message') or {}).get('content') or ''
        parsed = parser.parse_response(content)
        digest = hashlib.sha256(content.encode()).hexdigest()
        validity[digest] = not bool(parsed.error)
        verdict = dict(content_sha256=digest, valid=not bool(parsed.error), error=parsed.error,
            warning=parsed.warning, parsed_command_count=len(parsed.commands), is_task_complete=parsed.is_task_complete,
            action_text_unmapped=bool(parsed.commands) and not any(kind=='command_request' for _,_,kind in object_spans(content)))
        stats = dict(raw_profile, **profile_response(response, classify_fields=not bool(parsed.error)),
            terminus_parser=verdict, classification_provenance=provenance)
        (folder / 'profile.aligned.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        audits.append(dict(folder=folder.name, **verdict))
    # Chat histories can normalize a reply's surrounding whitespace. Parse the
    # actual stored assistant fields too, rather than silently treating an
    # unmatched response digest as a known parser rejection.
    history_verdicts = []
    for file in sorted(directory.glob('*/request.original.json')):
        request = json.loads(file.read_text(encoding='utf-8'))
        for message in request.get('messages', []):
            content = message.get('content')
            if message.get('role') != 'assistant' or not isinstance(content, str):
                continue
            digest = hashlib.sha256(content.encode()).hexdigest()
            if digest in validity:
                continue
            parsed = parser.parse_response(content)
            validity[digest] = not bool(parsed.error)
            history_verdicts.append(dict(content_sha256=digest,valid=not bool(parsed.error),
                error=parsed.error,warning=parsed.warning,parsed_command_count=len(parsed.commands)))
    audit = dict(provenance=provenance, content_validity=validity, replies=audits,
        additional_history_verdicts=history_verdicts)
    (directory / 'terminus-parser-audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return audit


def summarize_captures(directory, base_url=None, instruction="", trial_result=None, profile_all_inputs=False):
    from agent_phase_capture import profile_native_prompt
    from ninfer_gc_step import post_json
    records = []
    for request in directory.glob("*/request.original.json"):
        folder = request.parent
        file = folder / "profile.aligned.json"
        if not file.exists():
            file = folder / "profile.json"
        stats = json.loads(file.read_text(encoding="utf-8")) if file.exists() else dict(status="capture_incomplete")
        records.append(dict(folder=str(folder), **stats))
    require(records, "No model request captures")
    records.sort(key=lambda r: (r.get("started_unix_ns", 0), r["folder"]))
    valid = [r for r in records if r.get("http_status") == 200 and "coarse_counts" in r]
    fine = [r for r in valid if "counts" in r]
    coarse, details = Counter(), Counter()
    for row in valid:
        coarse.update(row["coarse_counts"])
    for row in fine:
        details.update(row["counts"])
    result = dict(schema_version=1, source="captured_model_api", requests=len(records),
        native_usage_requests=len(valid), detailed_requests=len(fine),
        generated_coarse_counts=dict(coarse), generated_detailed_counts=dict(details),
        coarse_fractions=fractions(coarse), detailed_fractions=fractions(details),
        detailed_coverage_warning="Detailed fractions cover classified requests only; inspect detailed_requests/native_usage_requests.",
        service_errors=[r for r in records if r.get("status") == "service_error" or r.get("http_status", 200) != 200],
        incomplete_captures=[r for r in records if r.get("status") in ("capture_incomplete", "usage_unavailable")],
        rounds=records, official_reward=None,
        limitations=["Observation/logprob collection adds overhead; not a decoding p95 benchmark.",
            "No summary semantics inferred. Logical input token shares are not physical KV residency."])
    parser_file = directory / 'terminus-parser-audit.json'
    if parser_file.exists():
        result['parser_audit'] = json.loads(parser_file.read_text(encoding='utf-8'))
    validity = result.get('parser_audit', {}).get('content_validity')
    if valid and base_url:
        last = valid[-1]
        folder = Path(last["folder"])
        request = json.loads((folder / "request.original.json").read_text(encoding="utf-8"))
        try:
            prompt = profile_native_prompt(request, lambda p, v: post_json(base_url, p, v), instruction,
                                           expected_tokens=last.get("prompt_tokens"), content_validity=validity)
            result["latest_successful_input"] = prompt
            if "counts" in last:
                counts = Counter(prompt["counts"]) + Counter(last["counts"])
                result["latest_logical_context"] = dict(tokens=sum(counts.values()), counts=dict(counts),
                    fractions=fractions(counts), boundary="Latest successful request's complete input plus its accepted output, not a sum of all prompts")
        except (OSError, ValueError, KeyError) as error:
            result["input_profile_error"] = repr(error)
        if profile_all_inputs:
            for row in valid:
                folder = Path(row['folder'])
                try:
                    prompt = profile_native_prompt(json.loads((folder / 'request.original.json').read_text(encoding='utf-8')),
                        lambda p, v: post_json(base_url, p, v), instruction,
                        expected_tokens=row.get('prompt_tokens'), content_validity=validity)
                    row['input_profile'] = prompt
                    (folder / 'input-profile.json').write_text(json.dumps(prompt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
                except (OSError, ValueError, KeyError) as error:
                    row['input_profile_error'] = repr(error)
    if trial_result:
        trial = json.loads(trial_result.read_text(encoding="utf-8"))
        result["official_reward"] = (trial.get("verifier_result") or {}).get("rewards")
        result["agent_exception"] = trial.get("exception_info")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--bundle", type=Path)
    source.add_argument("--captures", type=Path)
    parser.add_argument("--task", default="dna-assembly")
    parser.add_argument("--base-url", help="Owned native service kept online for whole-input token alignment after the task")
    parser.add_argument("--instruction", type=Path)
    parser.add_argument("--trial-result", type=Path)
    parser.add_argument("--align-terminus", action='store_true', help='Offline classification using the installed official Harbor parser')
    parser.add_argument("--profile-all-inputs", action='store_true', help='Also align every complete round input, without summing their repeated prefixes')
    parser.add_argument("--teacher-baseline", type=Path, help='Verified baseline JSON produced by --bundle')
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.captures:
        if args.align_terminus:
            align_terminus_captures(args.captures)
        result = summarize_captures(args.captures, args.base_url,
            args.instruction.read_text(encoding="utf-8") if args.instruction else "", args.trial_result, args.profile_all_inputs)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "current-api-profile.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if args.teacher_baseline:
            save_comparison(json.loads(args.teacher_baseline.read_text(encoding='utf-8')), result, args.out)
        print(json.dumps(dict(requests=result["requests"], detailed_requests=result["detailed_requests"],
                             native_usage_requests=result["native_usage_requests"], official_reward=result["official_reward"])))
        return
    report = audit_bundle(args.bundle, args.task)
    save_report(report, args.out)
    print(json.dumps(dict(task=report["task"], verified_files=report["integrity"]["verified_files"],
        rounds=len(report["rounds"]), generated_tokens=sum(report["output_counts"].values()),
        new_feedback_tokens=sum(report["input_counts"].values()), logical_tokens=sum(report["final_context_counts"].values()),
        model_output_shares=report["fractions"]["model_output"], current_project_run=False), ensure_ascii=False))


if __name__ == "__main__":
    main()
