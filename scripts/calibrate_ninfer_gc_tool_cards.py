"""Measure a conservative marginal decode cost using matched MTP work.

The output is grammar-constrained to one nonce in both contexts. Whole-request
matching requires identical MTP work counts. The separate full-acceptance-rounds
experiment matches exact output-token spans with four accepted drafts and one
bonus token, excluding unequal-work rounds. Neither experiment is a task quality
result or a guarantee of future gain; rejected earlier measurements stay intact.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import re
import statistics
import time

from ninfer_gc_agent_bridge import NativeClient
from ninfer_gc_tool_cards import CostProfile


def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def matched_full_rounds(short,long,context_delta):
    """Compare equal generated-token positions, page offsets and MTP4 widths.

    Token-byte segmentation, not just decoded text, must match. The first four
    rounds are omitted to avoid request-start transient work. Other mismatched
    intervals are reported/excluded, never imputed as equal-cost work. This
    estimates a context slope within fixed-width rounds, not a whole-task slope.
    """
    if context_delta<=0 or context_delta%64:
        return dict(eligible=False,reason='page_offsets_differ')
    if not short.get('output_piece_sha256') or short['output_piece_sha256']!=long.get('output_piece_sha256'):
        return dict(eligible=False,reason='generated_token_pieces_differ')
    indexed=[]
    for sample in (short,long):
        spec=sample['speculative'];rows=sample.get('decode_rounds',[])
        if (spec.get('backend')!='mtp' or spec['fallback_rounds'] or
            spec['drafted_tokens']!=4*spec['generation_rounds'] or
            len(rows)!=spec['generation_rounds'] or
            sum(r['tokens'] for r in rows)!=spec['output_tokens']):
            return dict(eligible=False,reason='unverified_round_work')
        position=0;spans={}
        for i,row in enumerate(rows):
            stop=position+row['tokens']
            if i>=4 and row['tokens']==5:spans[position,stop]=row['elapsed_ns']
            position=stop
        indexed.append(spans)
    left,right=indexed;common=sorted(set(left)&set(right))
    coverage=5*len(common)/max(short['speculative']['output_tokens'],long['speculative']['output_tokens'])
    if len(common)<64 or coverage<.5:
        return dict(eligible=False,reason='insufficient_matched_work',matched_rounds=len(common),matched_fraction=coverage)
    deltas=[(right[span]-left[span])/(5*context_delta) for span in common]
    return dict(eligible=True,matched_rounds=len(common),matched_fraction=coverage,
                matched_output_spans=common,marginal_ns_per_context_token=statistics.mean(deltas),
                median_marginal_ns_per_context_token=statistics.median(deltas),paired_slopes=deltas,
                scope='same output-token spans; full MTP4 acceptance; request-start rounds excluded')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--endpoint',required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--trace-path',type=Path,help='Owned server trace, with no concurrent inference clients')
    p.add_argument('--matching',choices=('whole-request','full-acceptance-rounds'),default='whole-request')
    p.add_argument('--long-records',type=int,default=1750)
    args=p.parse_args();args.out.mkdir(exist_ok=False)
    if args.matching=='full-acceptance-rounds' and not args.trace_path:p.error('Round matching requires the owned native trace')
    native=NativeClient(args.endpoint,timeout=600)
    props=native.json('/props');save(args.out/'props.json',props)
    if props['kvmem']['selected_tokens']!=98304 or props['kvmem']['capabilities']['mtp_drafts']!=4:
        raise ValueError('Keep the agreed 96K/MTP4 configuration for calibration')
    nonce=''.join(hashlib.sha256(f'fixed-card-calibration-{i}'.encode()).hexdigest() for i in range(12))
    schema=dict(type='json_schema',json_schema=dict(name='fixed_nonce',strict=True,
        schema=dict(type='object',properties=dict(nonce=dict(type='string',enum=[nonce])),
                    required=['nonce'],additionalProperties=False)))
    requests={};lengths={}
    for label,records in (('short',220),('long',args.long_records)):
        body='\n'.join(f'Archived log row {i:04d}: checksum {(i*97)%10000:04d}; unrelated old record.' for i in range(records))
        request=dict(model='qwen38-inline-step',prompt_cache_key='card-calibration-'+label,
            messages=[dict(role='user',content='The following is unrelated old tool data.\n'+body+
                           '\nReturn the exact nonce object required by the response schema.')],
            response_format=schema,enable_thinking=False,temperature=0,seed=1000,max_tokens=1536,logprobs=True)
        prepared=native.json('/apply-template',dict(request,return_tokens=True))
        if label=='long':
            # Different page offsets add archive/seal boundary effects to an
            # attention-length comparison. Align the real token lengths, not
            # character counts, while leaving the 96K cap and MTP4 unchanged.
            for _ in range(4):
                padding=(-(len(prepared['tokens'])-lengths['short']))%64
                if not padding:break
                request['messages'][0]['content']+=' x'*padding
                prepared=native.json('/apply-template',dict(request,return_tokens=True))
            if (len(prepared['tokens'])-lengths['short'])%64:
                raise ValueError('Cannot align the calibration page boundaries')
        lengths[label]=len(prepared['tokens']);requests[label]=request
        if lengths[label]+1536>98304:raise ValueError('Calibration would cross the active limit while generating')
        save(args.out/(label+'-request.json'),request)
    results=dict(scope='matched constrained output; same binary, 96K budget, MTP4',
                 context_tokens=lengths,pairs=[],warmup={},future_horizon_assumption=4096,matching=args.matching)
    def run(label,identity):
        save(args.out/'status.json',dict(state='running',sample=identity))
        offset=args.trace_path.stat().st_size if args.trace_path else None
        start=time.monotonic();response=native.json('/v1/chat/completions',requests[label])
        save(args.out/(identity+'-response.json'),response)
        text=response['choices'][0]['message']['content']
        if response['choices'][0]['finish_reason']!='stop' or json.loads(text)!={'nonce':nonce}:
            raise ValueError('Calibration output differs from the declared nonce')
        result=dict(wall_seconds=time.monotonic()-start,timings=response['timings'],
                    speculative=response['kvmem']['speculative'])
        pieces=[r['bytes'] for r in response['choices'][0]['logprobs']['content']]
        if len(pieces)!=result['speculative']['output_tokens'] or any(piece is None for piece in pieces):
            raise ValueError('Missing exact generated-token boundaries')
        result['output_piece_sha256']=hashlib.sha256(json.dumps(pieces,separators=(',',':')).encode()).hexdigest()
        if args.trace_path:
            with args.trace_path.open('rb') as stream:
                stream.seek(offset);trace=stream.read().decode('utf-8',errors='replace')
            rows=[dict(request=int(m[0]),tokens=int(m[1]),elapsed_ns=int(m[2])) for m in
                  re.findall(r'KVMEM_DECODE_ROUND request=(\d+) tokens=(\d+) elapsed_ns=(\d+)',trace)]
            if len({r['request'] for r in rows})!=1 or sum(r['tokens'] for r in rows)!=result['speculative']['output_tokens']:
                raise ValueError('Incomplete or concurrent decode trace; do not infer a cost profile')
            result['decode_rounds']=rows
        return result
    for label in ('short','long'):results['warmup'][label]=run(label,'warmup-'+label)
    delta=lengths['long']-lengths['short']
    if delta<=0:raise ValueError('Invalid calibration context ordering')
    slopes=[];work_matches=True
    for i in range(4):
        row={}
        order=('short','long') if i%2==0 else ('long','short')
        for label in order:row[label]=run(label,f'pair-{i}-{label}')
        keys=('accepted_per_position','drafted_tokens','accepted_tokens','output_tokens','generation_rounds','fallback_rounds')
        same=all(row['short']['speculative'][k]==row['long']['speculative'][k] for k in keys)
        same &= row['short']['output_piece_sha256']==row['long']['output_piece_sha256']
        per_token={label:row[label]['timings']['predicted_ms']/row[label]['speculative']['output_tokens']
                   for label in ('short','long')}
        slope=(per_token['long']-per_token['short'])*1e6/delta
        matched=matched_full_rounds(row['short'],row['long'],delta) if args.trace_path else None
        eligible=same if args.matching=='whole-request' else matched['eligible']
        chosen=slope if args.matching=='whole-request' else (matched.get('marginal_ns_per_context_token',0))
        work_matches &= eligible;slopes.append(chosen)
        results['pairs'].append(dict(order=order,same_mtp_work=same,marginal_ns_per_context_token=slope,
                                    matched_full_acceptance=matched,**row))
        save(args.out/'result.json',results)
    prefill=min(r['timings']['prompt_per_second'] for r in results['warmup'].values())
    admitted=work_matches and min(slopes)>0
    results.update(profile_published=admitted,minimum_measured_slope=min(slopes),
                   all_selected_work_matched=work_matches,
                   all_mtp_work_matched=all(p['same_mtp_work'] for p in results['pairs']),prefill_tokens_per_second=prefill,
                   reason='matched_positive_pairs' if admitted else 'context_effect_not_identifiable',
                   profile_rate_scope=args.matching,profile_haircut=.8)
    if admitted:
        profile=CostProfile(min(slopes)*.8,prefill,horizon_generated_tokens=4096,
                            evidence=str(args.out/'result.json'))
        save(args.out/'cost-profile.json',asdict(profile))
    save(args.out/'result.json',results)
    save(args.out/'status.json',dict(state='complete',profile_published=admitted))


if __name__=='__main__':main()
