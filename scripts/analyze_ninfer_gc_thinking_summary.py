#!/usr/bin/env python3
"""Audit a real owned Harbor step-summary run; final-only GC cannot pass."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

from ninfer_gc_proactive_probe import audit_gc
from ninfer_gc_step import pieces,token_digest
from ninfer_gc_thinking_summary import digest


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def analyze(root):
    lifecycle=root/'step-controller/lifecycle'
    records=[read(path) for path in sorted((lifecycle/'steps').glob('*.json'))]
    allowed={};unit_rows=[]
    for record in records:
        review=record.get('review',{});coverage=review.get('coverage',{})
        for run in coverage.get('covered_runs',[]):
            identifier=record['step_id']+'-'+digest(str(run['begin'])+':'+str(run['end']))[:12]
            allowed[identifier]=(record,run)
        for row in coverage.get('decisions',[]):
            unit_rows.append(dict(step=record['ordinal'],step_id=record['step_id'],unit=row['id'],
                begin=row['begin'],end=row['end'],accepted=row['accepted'],reason=row['reason'],
                source=record['thinking'][row['begin']:row['end']],summary=review['summary'],
                supporting_quotes=json.dumps(row.get('summary_quotes',[]),ensure_ascii=False)))
    proofs=[];invalid=[];requests=[]
    for folder in sorted((root/'step-controller/main').glob('*')):
        if not (folder/'profile.json').exists():continue
        profile=read(folder/'profile.json');request=read(folder/'request.forwarded.json')
        if request.get('kvmem_control_purpose'):continue
        requests.append(profile)
        evidence=request.get('kvmem_gc')
        if not evidence:continue
        audit=read(folder/'gc-prepare.json')
        try:
            prepared=read(lifecycle/audit['prepared_prefix'])['prepared'];rows=prepared['tokens'];chunks=pieces(rows)
            assert b''.join(chunks)==prepared['prompt'].encode('utf-8')
            assert evidence['prefix_tokens']==len(rows)
            assert evidence['prefix_sha256']==token_digest([r['id'] for r in rows])
            for proof in evidence['steps']:
                record,run=allowed[proof['id']]
                assert record['review']['coverage_status']=='reviewed' and proof['kind']=='thinking'
                message=request['messages'][record['source_message']]
                assert message['role']=='assistant' and message['reasoning_content']==record['thinking']
                assert run['text']==record['thinking'][run['begin']:run['end']]
                sb,se,rb,re_=(proof[k] for k in ('source_begin','source_end','result_begin','result_end'))
                assert all(x%64==0 for x in (sb,se,rb,re_)) and 0<=sb<se<=rb<re_<=len(rows)
                assert b''.join(chunks[sb:se]) in run['text'].encode('utf-8')
                assert record['review']['summary'].encode('utf-8') in b''.join(chunks[rb:re_])
                proofs.append(proof)
        except (AssertionError,OSError,ValueError,TypeError,KeyError) as error:
            invalid.append(dict(capture=str(folder),error=repr(error)))
    trials=[path for path in (root/'jobs-agent/job').glob('*/result.json')]
    trial=read(trials[0]) if len(trials)==1 else None
    reward=((trial or {}).get('verifier_result') or {}).get('rewards',{}).get('reward')
    exception=(trial or {}).get('exception_info')
    agent_metadata=((trial or {}).get('agent_result') or {}).get('metadata') or {}
    compactions=agent_metadata.get('summarization_count')
    snapshot=lifecycle/'native-snapshots/agent-returned-before-final-maintenance.log'
    before=snapshot.read_bytes() if snapshot.exists() else b''
    task_passed=reward==1 and exception is None
    live=audit_gc(before,proofs,task_passed,'statistical')
    final_path=root/'native-after-agent.log'
    full=final_path.read_bytes() if final_path.exists() else b''
    with_final=audit_gc(full,proofs,task_passed,'statistical')
    auxiliary=[read(path) for path in sorted((root/'step-controller').glob('auxiliary-*.json'))]
    aux_kinds=Counter('coverage_review' if 'thinking_units' in json.loads(row['input']['messages'][1]['content'])
        else 'summary' for row in auxiliary)
    result=dict(task='cobol-modernization',official_reward=reward,official_exception=exception,
        agent_summarization_count=compactions,agent_metadata=agent_metadata,
        main_requests=len(requests),main_http_statuses=dict(Counter(str(row['http_status']) for row in requests)),
        main_http_seconds=sum(row.get('elapsed_seconds',0) for row in requests),
        step_callbacks=len(records),step_statuses=dict(Counter(row['status'] for row in records)),
        thinking_units=len(unit_rows),accepted_units=sum(row['accepted'] for row in unit_rows),
        coverage_reasons=dict(Counter(row['reason'] for row in unit_rows)),
        auxiliary_calls=len(auxiliary),auxiliary_kinds=dict(aux_kinds),
        auxiliary_seconds=sum(row['elapsed_seconds'] for row in auxiliary),
        auxiliary_output_tokens=sum(row['response'].get('tokens_predicted',0) for row in auxiliary),
        source_verified_publications=len(proofs),invalid_publications=invalid,
        during_task_before_final_maintenance=live,including_final_maintenance=with_final,
        final_maintenance_only_does_not_pass=True,
        acceptance_passed=bool(task_passed and compactions==0 and requests and
            all(row['http_status']==200 for row in requests) and not invalid and
            live['proactive_model_gate'] and records),
        limits=['Model semantic coverage is heuristic, not a formal no-information-loss proof.',
            'One official task is a functional acceptance; not a full benchmark accuracy or decode p95 result.',
            'Main HTTP plus auxiliary time excludes tool/setup time; original Job timestamps define full task time.'])
    return result,unit_rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(exist_ok=False)
    result,units=analyze(a.run)
    (a.out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if units:
        with (a.out/'thinking-coverage.csv').open('w',encoding='utf-8-sig',newline='') as output:
            writer=csv.DictWriter(output,fieldnames=list(units[0]));writer.writeheader();writer.writerows(units)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
