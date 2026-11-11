"""Audit card opportunities in recorded tasks without changing or scoring them.

Samples four points of each recorded trajectory. Only model-visible feedback
can be reconstructed from these lifecycle files; this is NOT a raw archive or
new Agent acceptance run. No cards are published and no inference is requested.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

from ninfer_gc_agent_bridge import NativeClient
from ninfer_gc_tool_cards import OriginalArchive,OwnedToolHistory,ToolCardPlanner


def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--endpoint',required=True);args=p.parse_args();args.out.mkdir(exist_ok=False)
    native=NativeClient(args.endpoint,timeout=120)
    result=dict(scope='recorded opportunities; no model generation, no edits, no new verifier scores',tasks=[])
    for folder in sorted(args.root.glob('*-after')):
        records=json.loads((folder/'controller/lifecycle/completion.json').read_text())['records']
        paths=sorted((folder/'controller/main').glob('*/request.forwarded.json'))
        if not paths:raise ValueError('No captured request in '+str(folder))
        task=dict(name=folder.name,request_count=len(paths),cases=[])
        indexes=sorted({min(len(paths)-1,max(0,int(len(paths)*fraction)-1)) for fraction in (.25,.5,.75,1)})
        for index in indexes:
            started=time.monotonic();path=paths[index];request=json.loads(path.read_text())
            for key in ('kvmem_gc','stream','timings_per_token','stream_options'):request.pop(key,None)
            messages=request['messages'];history=OwnedToolHistory()
            for record in sorted(records,key=lambda r:r['source_message']):
                i=record['source_message'];feedback=record['feedback']
                if (record['status']=='incomplete_tools' or i+1>=len(messages) or
                    messages[i].get('role')!='assistant' or messages[i+1].get('role')!='user' or
                    messages[i].get('content')!=record['content'] or
                    messages[i].get('reasoning_content','')!=record['thinking']):continue
                history.register(feedback=feedback,raw_feedback=feedback,commands=record.get('commands',[]),timed_out=False)
                history.bind(messages[:i+1],messages[i+1]['content'])
            candidates=history.candidates(messages)
            planner=ToolCardPlanner(OriginalArchive(args.out/'unpublished-archive',folder.name),native)
            proposal,audit=planner.prepare(request,candidates)
            if proposal is not None:raise AssertionError('Uncalibrated offline audit cannot publish')
            row=dict(request_index=index+1,input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                trusted_feedbacks=len(history.records),feedback_bytes=[len(r['feedback'].encode()) for r in history.records],
                messages=len(messages),first_saved_source_message=min((r['source_message'] for r in records),default=None),
                provenance_scope='only retained completed-step records; missing earlier records are not negative evidence',
                elapsed_seconds=time.monotonic()-started,**audit)
            task['cases'].append(row)
            save(args.out/'status.json',dict(state='running',task=folder.name,request=index+1))
        result['tasks'].append(task);save(args.out/'result.json',result)
    if len(result['tasks'])!=16:raise ValueError('The expected fixed 16 tasks are not all present')
    result['status']='complete';save(args.out/'result.json',result)
    save(args.out/'status.json',dict(state='complete',tasks=len(result['tasks'])))


if __name__=='__main__':main()
