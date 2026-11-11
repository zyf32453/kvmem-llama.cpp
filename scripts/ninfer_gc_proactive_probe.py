#!/usr/bin/env python3
"""Run a real tool-using model probe; keep GC evidence separate from task success.

This is an owned, small artifact task, not an official Terminal-Bench score.
It neither launches a server nor changes its configuration. Run matched off and
statistical servers separately and retain their actual commands and logs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

from ninfer_gc_agent_bridge import AutomaticArtifactGc, NativeClient


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def owned_logs(container):
    labels = json.loads(subprocess.check_output(['docker','inspect','--format','{{json .Config.Labels}}',container]))
    if labels.get('kvmem.proactive.acceptance') != 'true':
        raise ValueError('Server log target is not an owned proactive acceptance container')
    return subprocess.check_output(['docker','logs',container], stderr=subprocess.STDOUT)


def audit_gc(log, proofs, task_passed, policy):
    """A proposal alone, pressure deletion, or a stale prior run cannot pass."""
    summaries, proposals, acknowledgements, releases = [], {}, [], set()
    def identity(row):
        return tuple(row.get(k) for k in ('incarnation','slot','generation','round'))
    for line in log.decode(errors='replace').splitlines():
        for marker in ('KVMEM_NATIVE_GC_SUMMARY ', 'KVMEM_GC_AUDIT '):
            if marker not in line:
                continue
            row = dict(item.split('=',1) for item in line.split(marker,1)[1].split() if '=' in item)
            if marker.startswith('KVMEM_NATIVE'):
                summaries.append(row)
            elif row.get('event')=='propose' and row.get('policy')=='statistical':
                proposals[identity(row)] = row
            elif row.get('event')=='ack' and row.get('outcome')=='committed' and row.get('intent_reason')=='semantic':
                acknowledgements.append(row)
            elif row.get('event')=='reclaim' and row.get('ready')=='1' and row.get('intent_known')=='1':
                releases.add(identity(row))
    matched, nonpressure, unverified = 0, 0, 0
    distinct_acks = {identity(ack): ack for ack in acknowledgements}
    for ack in distinct_acks.values():
        proposal = proposals.get(identity(ack))
        try:
            begin = int(proposal['original_begin'])
            end = begin + int(proposal['valid_tokens'])
            covered = any(p['source_begin']<=begin<end<=p['source_end'] and
                not (begin<p['result_end'] and p['result_begin']<end) for p in proofs)
            if not covered:
                unverified += 1
                continue
            matched += 1
            nonpressure += int(proposal['ready']) > int(proposal['high'])
        except (TypeError,KeyError,ValueError):
            unverified += 1
    # Counters can repeat while an owner lives, or reset when its window moves.
    # Count unique actual events instead, including incarnation across owners.
    committed = len(distinct_acks)
    reclaimed = len(releases.intersection(distinct_acks))
    reported_committed = max((int(r.get('committed',0)) for r in summaries), default=0)
    reported_reclaimed = max((int(r.get('reclaimed',0)) for r in summaries), default=0)
    single_history = len({a.get('incarnation') for a in distinct_acks.values()}) <= 1
    return dict(native_gc_summaries=summaries, native_committed=committed, native_reclaimed=reclaimed,
        committed_with_source_proof=matched, nonpressure_commits=nonpressure,
        unverified_commits=unverified,
        audit_scope='unique_native_gc_events', single_history=single_history,
        native_reported_committed_peak=reported_committed, native_reported_reclaimed_peak=reported_reclaimed,
        task_gate=task_passed,
        proactive_model_gate=None if policy=='off' else (task_passed and
            committed>0 and reclaimed==committed and reported_committed>0 and reported_reclaimed>0 and matched==committed and
            nonpressure==matched and unverified==0))


def execute(call, workspace):
    """The owned executor exposes two file tools, never arbitrary shell code."""
    function = call['function']
    args = json.loads(function['arguments'])
    path = Path(args['file_path']).resolve()
    if path.parent != workspace or path.name != 'artifact.txt':
        raise ValueError('Tool path is not the assigned artifact')
    if function['name'] == 'write':
        if set(args) != {'file_path', 'content'} or not isinstance(args['content'], str):
            raise ValueError('Invalid write arguments')
        data = args['content'].encode('utf-8')
        if len(data) > 1048576:
            raise ValueError('Artifact exceeds the file-equality profile')
        existed = path.exists()
        path.write_bytes(data)
        verb = 'Updated' if existed else 'Created'
        return f'<path>{path}</path>\n<type>file</type>\n<content>\n{verb} file\n</content>'
    if function['name'] == 'read':
        if set(args) != {'file_path'}:
            raise ValueError('Invalid read arguments')
        data = path.read_bytes()
        if len(data) > 1048576:
            raise ValueError('Artifact exceeds the read budget')
        return json.dumps(dict(file_path=str(path), content=data.decode('utf-8'),
            line_count=len(data.splitlines()), sha256=hashlib.sha256(data).hexdigest()), ensure_ascii=False)
    raise ValueError('Unexpected tool name')


def run(args):
    args.workspace.mkdir(parents=True, exist_ok=False)
    args.workspace = args.workspace.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    client = NativeClient(args.base_url)
    props = client.json('/props?' + urllib.parse.urlencode(dict(model=args.model)))
    profile = props.get('kvmem_gc', {})
    if profile.get('policy') != args.policy:
        raise ValueError('Actual native policy differs from the declared probe')
    session = 'proactive-artifact-' + uuid.uuid4().hex
    adapter = AutomaticArtifactGc(args.workspace, args.base_url, session_key=session)
    path = args.workspace / 'artifact.txt'
    body = ''.join(f'ROW {i:04d} : preserved-artifact-value {i*17+5}\n' for i in range(args.lines))
    expected_hash = hashlib.sha256(body.encode()).hexdigest()
    tools = [dict(type='function', function=dict(name=name, description=description, parameters=dict(
        type='object', properties=properties, required=list(properties), additionalProperties=False)))
        for name, description, properties in (
            ('write', 'Write the entire UTF-8 file and return its completion result.',
             dict(file_path=dict(type='string'), content=dict(type='string'))),
            ('read', 'Read the actual file; returns full content, line_count and SHA256.',
             dict(file_path=dict(type='string'))))]
    messages = [dict(role='system', content='Use the supplied file tools. Preserve exact file bytes. '
        'After writing, inspect the actual file with read before reporting results. Do not rewrite a correct file.'),
        dict(role='user', content=f'Create {path} with exactly the following UTF-8 content, including its final newline. '
            'Then read the saved file. Return only a JSON object with line_count and sha256 from the actual read result.\n'
            '<required-file-content>\n' + body + '</required-file-content>')]
    plan = dict(policy=args.policy, model=args.model, props=props, session_key=session,
        expected_lines=args.lines, expected_sha256=expected_hash, max_rounds=args.max_rounds,
        max_output=args.max_output, thinking_budget=args.thinking_budget, temperature=args.temperature,
        seed=args.seed, top_p=.95, top_k=20, min_p=0,
        scope='real model + owned file tools; not an official benchmark score or a decode-latency comparison')
    save(args.out/'plan.json', plan)
    before_log = owned_logs(args.server_container) if args.server_container else None
    proofs = []
    result = dict(status='running', policy=args.policy, rounds=[], calls=[], evidence_attached=0,
                  artifact_exact=False, read_executed=False, final_summary_correct=False)
    save(args.out/'result.json', result)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        for turn in range(1, args.max_rounds+1):
            original = dict(model=args.model, messages=messages, tools=tools, tool_choice='auto',
                max_tokens=args.max_output, thinking_budget=args.thinking_budget,
                temperature=args.temperature, seed=args.seed, top_p=.95, top_k=20, min_p=0)
            request, audit = adapter(original)
            record = args.out/f'{turn:03d}'
            record.mkdir()
            save(record/'request.original.json', original)
            save(record/'request.forwarded.json', request)
            save(record/'gc-prepare.json', audit)
            result['evidence_attached'] += audit.get('attached', 0)
            proofs.extend(request.get('kvmem_gc',{}).get('steps',[]))
            started = time.monotonic()
            req = urllib.request.Request(args.base_url.rstrip('/')+'/v1/chat/completions',
                data=json.dumps(request, ensure_ascii=False).encode(), headers={'Content-Type':'application/json'})
            try:
                with opener.open(req, timeout=7200) as response:
                    status = response.status
                    data = json.load(response)
            except urllib.error.HTTPError as error:
                result.update(status='service_error', http_status=error.code,
                              service_error_body=error.read().decode('utf-8', errors='replace'))
                break
            save(record/'response.json', data)
            result['rounds'].append(dict(turn=turn, http_status=status,
                elapsed_seconds=time.monotonic()-started, usage=data.get('usage'), timings=data.get('timings'),
                finish_reason=data['choices'][0].get('finish_reason'), gc_audit=audit))
            assistant = dict(data['choices'][0]['message'])
            messages.append(assistant)
            calls = assistant.get('tool_calls') or []
            for call in calls:
                tool_result = execute(call, args.workspace)
                messages.append(dict(role='tool', tool_call_id=call['id'], content=tool_result))
                name = call['function']['name']
                result['calls'].append(dict(turn=turn, id=call['id'], name=name))
                result['read_executed'] |= name == 'read'
            if path.exists():
                result['artifact_exact'] = path.read_bytes() == body.encode()
            if not calls:
                text = (assistant.get('content') or '').strip()
                try:
                    summary = json.loads(text)
                    result['final_summary_correct'] = summary.get('line_count') == args.lines and summary.get('sha256') == expected_hash
                except (ValueError, AttributeError):
                    result['final_summary_correct'] = False
                if result['artifact_exact'] and result['read_executed'] and result['final_summary_correct']:
                    result['status'] = 'task_passed'
                    break
                result['status'] = 'task_failed'
                break
            save(args.out/'result.json', result)
        else:
            result['status'] = 'agent_round_limit'
    except Exception as error:
        result.update(status='probe_error', error=repr(error))
    finally:
        save(args.out/'messages.final.json', messages)
        result['evidence_is_not_retirement_proof'] = True
        if args.server_container:
            try:
                log = owned_logs(args.server_container)
                (args.out/'server.log').write_bytes(log)
                if not log.startswith(before_log):
                    raise ValueError('Native log rotated; current probe isolation is unproven')
                result.update(audit_gc(log[len(before_log):], proofs, result['status']=='task_passed', args.policy))
            except Exception as error:
                result.update(proactive_model_gate=False, log_error=repr(error))
        save(args.out/'result.json', result)
    print(json.dumps(result, ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url',default='http://127.0.0.1:18103')
    parser.add_argument('--model',default='qwen38-proactive')
    parser.add_argument('--workspace',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--policy',choices=('off','statistical'),required=True)
    parser.add_argument('--server-container')
    parser.add_argument('--lines',type=int,default=64)
    parser.add_argument('--max-rounds',type=int,default=8)
    parser.add_argument('--max-output',type=int,default=16384)
    parser.add_argument('--thinking-budget',type=int,default=4096)
    parser.add_argument('--temperature',type=float,default=.6)
    parser.add_argument('--seed',type=int,default=1000)
    args=parser.parse_args()
    url=urllib.parse.urlsplit(args.base_url)
    if url.scheme!='http' or url.hostname not in ('127.0.0.1','localhost','::1'):
        parser.error('Use the owned local native service')
    if not 16<=args.lines<=256 or not 1<=args.max_rounds<=32:
        parser.error('Invalid bounded probe size')
    run(args)


if __name__=='__main__':
    main()
