"""Real-model synthetic lifecycle smoke; not an Agent score or semantic proof."""
import argparse,json,re,time
from pathlib import Path
from ninfer_gc_agent_bridge import NativeClient
from ninfer_gc_step import attach_evidence
from ninfer_gc_inline_summary import response_format,summary_span

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base-url',required=True)
    p.add_argument('--model',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(exist_ok=False);client=NativeClient(a.base_url,timeout=600)
    def save(name,value):(a.out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    fact='The inspection is complete and the stable output remains unchanged. '
    def body(analysis):
        return json.dumps(dict(analysis=analysis,plan='Report the recorded state.',commands=[],task_complete=False))
    thinking=fact*300
    witness=body(fact*32+' The verified identifier is WITNESS_TAG.')
    _,begin,end=summary_span(witness)
    messages=[
        dict(role='user',content='Read-only synthetic memory fixture. '+'Neutral archival record. '*750),
        dict(role='assistant',reasoning_content=thinking,content=body('The fixture will be inspected.')),
        dict(role='user',content='New Terminal Output:\nInspection completed; stable output unchanged.'),
        dict(role='assistant',reasoning_content='The returned observation confirms the recorded state.',content=witness),
        dict(role='user',content='Report WITNESS_TAG from the retained analysis. Give a detailed explanation of the stable facts. '
             'Return normal Terminus JSON and no commands; this is a read-only fixture.')]
    request=dict(model=a.model,prompt_cache_key='synthetic-boundary-smoke',messages=messages,
        max_tokens=768,thinking_budget=256,temperature=.6,seed=1000,top_p=.95,top_k=20,min_p=0,
        response_format=response_format())
    step=dict(id='synthetic-stable-thinking',kind='thinking',source_text=thinking.strip(),
        result_text=witness[begin:end],observation_id=1,importance=1,cold=False,complete=True,
        _source_message_span=dict(message=1,field='reasoning_content',begin=0,end=len(thinking.strip())),
        _result_message_span=dict(message=3,field='content',begin=begin,end=end,kind='inline_step_summary'))
    prepared=attach_evidence('',request,[step],post=client.json);save('request.json',prepared)
    started=time.monotonic();first=client.json('/v1/chat/completions',prepared);save('response.json',first)
    message=first['choices'][0]['message'];props=client.json('/props?model='+a.model)
    next_request=dict(request,messages=messages+[dict(role='assistant',content=message.get('content') or '',
        reasoning_content=message.get('reasoning_content') or ''),dict(role='user',content='Acknowledge WITNESS_TAG briefly; no commands.')],
        max_tokens=256,thinking_budget=64)
    save('continuation-request.json',next_request)
    second=client.json('/v1/chat/completions',next_request);save('continuation-response.json',second)
    gc=second.get('kvmem',{}).get('historical_gc',{})
    first_gc=first.get('kvmem',{}).get('historical_gc',{})
    # Scoring may retain candidates. Such a run is not a lifecycle pass.
    frontier=first['usage']['prompt_tokens']+first['usage']['completion_tokens']
    cached=second.get('timings',{}).get('cache_n',0)
    reclaimed=max(gc.get('reclaimed_pages',0),first_gc.get('reclaimed_pages',0))
    committed=max(gc.get('committed_pages',0),first_gc.get('committed_pages',0))
    passed=committed>0 and reclaimed>0 and cached>=frontier-64
    result=dict(status='passed' if passed else 'no_physical_gc_or_failure',native_gc=gc,
        first_gc=first_gc,elapsed_seconds=time.monotonic()-started,expected_cache_floor=frontier-64,
        first_output_tokens=first.get('usage',{}).get('completion_tokens'),
        continuation_cached_tokens=second.get('timings',{}).get('cache_n'),
        synthetic_fixture=True,official_terminal_bench_score=False,auxiliary_summary_calls=0)
    save('result.json',result);print(json.dumps(result,indent=2))
    if not passed:raise SystemExit(2)

if __name__=='__main__':main()
