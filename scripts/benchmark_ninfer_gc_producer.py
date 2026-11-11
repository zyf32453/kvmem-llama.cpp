"""Compare provenance preparation on two recorded, consecutive Agent requests.

Uses only the native template endpoint. No model outputs or verifier scores are
produced. The before module must be the frozen pre-change producer, not an older
unbatched implementation. Every timed result must equal its independent baseline.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import time

from ninfer_gc_agent_bridge import NativeClient
from ninfer_gc_inline_summary import summary_span
import ninfer_gc_step as current


def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def owned_scopes(messages,records):
    completed={r['source_message']:r for r in records if r['status']!='incomplete_tools'}
    for index in range(len(messages)-3,-1,-1):
        source,feedback,result=messages[index:index+3]
        if [r['role'] for r in (source,feedback,result)]!=['assistant','user','assistant']:continue
        record=completed.get(index);thinking=source.get('reasoning_content') or ''
        if (record is None or len(thinking.strip())<512 or record['content']!=source.get('content') or
            record['thinking']!=thinking or not record['feedback'] or record['feedback'] not in feedback.get('content','')):continue
        try:text,begin,end=summary_span(result['content'])
        except (ValueError,KeyError,IndexError):continue
        if not text.strip():continue
        start=len(thinking)-len(thinking.lstrip());source_text=thinking.strip()
        yield dict(id=f'recorded-step-{index}',kind='thinking',source_text=source_text,
            result_text=result['content'][begin:end],complete=True,observation_id=1,importance=1.0,cold=False,
            _source_message_span=dict(message=index,field='reasoning_content',begin=start,end=start+len(source_text)),
            _result_message_span=dict(message=index+2,field='content',begin=begin,end=end,kind='inline_step_summary'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--endpoint',required=True);p.add_argument('--request-dir',type=Path,required=True)
    p.add_argument('--records',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);args=p.parse_args();args.out.mkdir(exist_ok=False)
    spec=importlib.util.spec_from_file_location('frozen_producer',args.baseline)
    before=importlib.util.module_from_spec(spec);spec.loader.exec_module(before)
    if hasattr(before,'NativeSpanCache'):raise ValueError('Baseline already contains this optimization')
    paths=sorted(args.request_dir.glob('*/request.forwarded.json'))[-2:]
    if len(paths)!=2:raise ValueError('Two recorded requests are required')
    bodies=[json.loads(path.read_text()) for path in paths]
    for body in bodies:
        for key in ('kvmem_gc','stream','timings_per_token','stream_options'):body.pop(key,None)
    records=json.loads(args.records.read_text())['records']
    native=NativeClient(args.endpoint,timeout=120);cache=current.NativeSpanCache()
    def measure(module,body,batch,use_cache=False):
        calls=0
        def post(path,value):
            nonlocal calls
            if path!='/apply-template':raise AssertionError('Unexpected inference call')
            calls+=1;return native.json(path,value)
        start=time.perf_counter_ns()
        result=module.attach_evidence('',body,batch,post=post,**({'span_cache':cache} if use_cache else {}))
        proof=result.pop('kvmem_gc')
        if result!=body:raise AssertionError('Producer altered model input')
        return dict(template_calls=calls,elapsed_ms=(time.perf_counter_ns()-start)/1e6),proof
    batch=[];skipped=[]
    for scope in owned_scopes(bodies[0]['messages'],records):
        try:measure(before,bodies[0],[scope])
        except ValueError as error:skipped.append(dict(id=scope['id'],reason=str(error)));continue
        batch.append(scope)
        if len(batch)==4:break
    if len(batch)!=4:raise ValueError('Need four independently located real scopes')
    result=dict(scope='CPU producer, native tokenizer, recorded Agent history; no inference',
        baseline_sha256=hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        candidate_sha256=hashlib.sha256(Path(current.__file__).read_bytes()).hexdigest(),
        input_sha256=[hashlib.sha256(path.read_bytes()).hexdigest() for path in paths],skipped=skipped,pairs=[])
    _,expected=measure(before,bodies[0],batch)
    result['cold'],actual=measure(current,bodies[0],batch,True)
    if actual!=expected:raise AssertionError('Cold proof differs')
    _,expected=measure(before,bodies[1],batch)
    result['append'],actual=measure(current,bodies[1],batch,True)
    if actual!=expected:raise AssertionError('Appended-history proof differs')
    result['proven_prefix_tokens']=expected['prefix_tokens']
    for i in range(8):
        row={};order=('before','after') if i%2==0 else ('after','before')
        for label in order:
            row[label],actual=measure(before if label=='before' else current,bodies[1],batch,label=='after')
            if actual!=expected:raise AssertionError('Timed proof differs')
        result['pairs'].append(dict(order=order,**row));save(args.out/'result.json',result)
    changed=copy.deepcopy(bodies[1]);changed['messages'][0]['content']='Changed prefix for invalidation check.\n'+changed['messages'][0]['content']
    _,expected=measure(before,changed,batch)
    result['invalidated'],actual=measure(current,changed,batch,True)
    if actual!=expected or result['invalidated']['template_calls']!=2:raise AssertionError('Changed prefix reused an old location')
    result.update(status='complete',all_proofs_equal=True,cached_spans=cache.hits,
        before_median_ms=statistics.median(row['before']['elapsed_ms'] for row in result['pairs']),
        after_median_ms=statistics.median(row['after']['elapsed_ms'] for row in result['pairs']))
    save(args.out/'result.json',result)


if __name__=='__main__':main()
