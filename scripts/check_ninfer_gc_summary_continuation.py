"""Real normal-response summary / prefix continuation preflight; zero aux calls."""
import argparse,json
from pathlib import Path
from ninfer_gc_agent_bridge import NativeClient
from ninfer_gc_inline_summary import response_format,summary_span

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url',required=True);p.add_argument('--model',required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(exist_ok=False);client=NativeClient(a.base_url,timeout=600)
    def save(name,value):(a.out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    request=dict(model=a.model,prompt_cache_key='inline-summary-continuation',messages=[
        dict(role='user',content='Plan a single read-only directory inspection. Do not execute it. '
             'Use a Terminus JSON response with analysis, plan, commands=[], task_complete=false.')],
        max_tokens=1024,thinking_budget=256,temperature=.6,seed=1000,top_p=.95,top_k=20,min_p=0,
        response_format=response_format())
    rows=[]
    for i in range(3):
        save(str(i+1)+'-request.json',request)
        response=client.json('/v1/chat/completions',request);save(str(i+1)+'-response.json',response)
        message=response['choices'][0]['message']
        summary,_,_=summary_span(message['content'])
        usage=response['usage'];cached=response.get('timings',{}).get('cache_n',0)
        rows.append(dict(input_tokens=usage['prompt_tokens'],output_tokens=usage['completion_tokens'],
                         cached_tokens=cached,summary_chars=len(summary)))
        request['messages'] += [dict(role='assistant',content=message['content'],
            reasoning_content=message.get('reasoning_content') or ''),
            dict(role='user',content='New Terminal Output:\nThe read-only inspection returned two files, a.txt and b.txt. '
                 'Acknowledge the observation with no commands.')]
    passed=all(rows[i]['cached_tokens']>=rows[i-1]['input_tokens']+rows[i-1]['output_tokens']-64 for i in (1,2))
    result=dict(status='passed' if passed else 'failed',rows=rows,auxiliary_model_calls=0,
                terminal_bench_acceptance=False,scope='normal response JSON and native prefix reuse, not GC yield')
    save('result.json',result);print(json.dumps(result,indent=2))
    if not passed:raise SystemExit(1)

if __name__=='__main__':main()
