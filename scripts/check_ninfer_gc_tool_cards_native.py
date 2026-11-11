"""Real-model card-edit/cache probe. Explicit diagnostic, not policy acceptance.

Requires an already isolated NInfer server. No benchmark task or verifier is
changed. The edit is forced only to test the native history-rewrite contract;
no synthetic rate is installed in the production cost gate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

from ninfer_gc_agent_bridge import NativeClient
from ninfer_gc_tool_cards import CostProfile, OriginalArchive, OwnedToolHistory, ToolCardPlanner, build_card, common_prefix, sha256


def save(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def replay_repeat_arm(native, warm, request, out, label, key):
    save(out/'status.json', dict(state='warming', case=label))
    warmed = native.json('/v1/chat/completions', dict(warm, prompt_cache_key=key))
    save(out/(label+'-warm.json'), warmed)
    if warmed.get('error') or warmed['timings']['cache_n'] != 0:
        raise RuntimeError('Replay must warm an independent session')
    save(out/'status.json', dict(state='prefilling', case=label))
    started = time.monotonic()
    body = native.json('/v1/chat/completions', dict(request, prompt_cache_key=key))
    elapsed = time.monotonic()-started
    save(out/(label+'-response.json'), body)
    if body.get('error'):
        raise RuntimeError(body['error'])
    return dict(timings=body['timings'], wall_seconds=elapsed, usage=body['usage'],
                output=body['choices'][0]['message'])


def recorded_repeat(native, directory, out, profile):
    """Compare warmed prefill for a recorded request containing real repeats."""
    records = json.loads((directory/'controller/lifecycle/completion.json').read_text())['records']
    paths = sorted((directory/'controller/main').glob('*/request.forwarded.json'))
    archive = OriginalArchive(out/'originals', 'recorded-repeat-diagnostic')
    planner = ToolCardPlanner(archive, native, profile)
    best = None
    attempts = []
    for index, path in enumerate(paths[1:], start=1):
        request = json.loads(path.read_text())
        messages = request['messages']
        history = OwnedToolHistory()
        for record in sorted(records, key=lambda r: r['source_message']):
            i = record['source_message']
            if (record['status'] == 'incomplete_tools' or i+1 >= len(messages)
                    or messages[i].get('role') != 'assistant' or messages[i+1].get('role') != 'user'
                    or messages[i].get('content') != record['content']
                    or messages[i].get('reasoning_content', '') != record['thinking']):
                continue
            feedback = record['feedback']
            history.register(feedback=feedback, raw_feedback=feedback, commands=record.get('commands', []),
                             timed_out=False)
            history.bind(messages[:i+1], messages[i+1]['content'])
        repeat = history.repeated_tail(messages)
        if repeat is None:
            continue
        previous = json.loads(paths[index-1].with_name('response.json').read_text())
        frontier = (previous.get('usage') or {}).get('total_tokens')
        for key in ('kvmem_gc', 'stream_options', 'timings_per_token'):
            request.pop(key, None)
        request.update(stream=False, max_tokens=1, temperature=0, logprobs=True, top_logprobs=0)
        plan, audit = planner.measure_repeated_tail(request, repeat, frontier)
        attempts.append(dict(request=index+1, **audit))
        if plan is not None and (best is None or plan['saved_tokens'] > best[0]['saved_tokens']):
            best = plan, request, repeat, audit, index+1
    save(out/'repeat-candidates.json', attempts)
    if best is None:
        raise ValueError('No verified first-presentation repeat in this recording')
    plan, request, repeat, audit, index = best
    started = time.monotonic()
    archive.save(plan['archive'])
    archive_seconds = time.monotonic()-started
    changed = dict(request, messages=[*request['messages'][:-1],
                                      dict(request['messages'][-1], content=plan['replacement'])])
    record = repeat['record']
    original_tail = record['original_message']
    # Both arms build their own identical cache from the shared prefix just
    # before the changed bytes. No earlier arm can populate a later cache key.
    prefix = original_tail[:record['begin']+repeat['begin']]
    warm = dict(request, messages=[*request['messages'][:-1], dict(role='user', content=prefix)])
    result = dict(scope='frozen real task input; forced one-token native replay; no new verifier',
                  recorded_task=directory.name, recorded_request=index, admission=audit,
                  archive_sha256=plan['archive']['sha256'], archive_write_seconds=archive_seconds,
                  archive_scope='recorded model-visible bytes, not pre-truncation tool output', pairs=[])
    save(out/'repeat-warm-request.json', warm)
    save(out/'repeat-original-request.json', request)
    save(out/'repeat-card-request.json', changed)
    for pair in range(2):
        rows = {}
        order = ('original', 'card') if pair == 0 else ('card', 'original')
        for arm in order:
            key = f'cards-recorded-{index}-{pair}-{arm}'
            label = f'pair-{pair+1}-{arm}'
            rows[arm] = replay_repeat_arm(native, warm, changed if arm == 'card' else request,
                                          out, label, key)
        original, card = rows['original']['timings'], rows['card']['timings']
        row = dict(order=list(order), cases=rows,
                   cached_tokens_equal=original['cache_n'] == card['cache_n'] > 0,
                   removed_prefill_tokens=original['prompt_n']-card['prompt_n'],
                   prefill_ms_saved=original['prompt_ms']-card['prompt_ms'],
                   same_generated_tokens=original['predicted_n'] == card['predicted_n'] == 1,
                   same_output=rows['original']['output'] == rows['card']['output'])
        result['pairs'].append(row)
        save(out/'result.json', result)
    result['token_work_verified'] = all(row['cached_tokens_equal'] and row['same_generated_tokens']
        and row['removed_prefill_tokens'] == plan['saved_tokens'] for row in result['pairs'])
    save(out/'result.json', result)
    if not result['token_work_verified']:
        raise RuntimeError('Replay cache and evaluated token counts did not isolate the expected saving')
    save(out/'status.json', dict(state='complete'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint',required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--prime-prefix',action='store_true',help='Create the retained prefix in a genuine earlier request')
    parser.add_argument('--recorded-repeat-dir', type=Path)
    parser.add_argument('--cost-profile', type=Path)
    parser.add_argument('--repeat-replay-plan', type=Path)
    parser.add_argument('--repeat-replay-arm', choices=('original', 'card'))
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    native=NativeClient(args.endpoint,timeout=600)
    props=native.json('/props')
    save(args.out/'props.json',props)
    if props['kvmem']['selected_tokens']!=98304 or props['kvmem']['capabilities']['mtp_drafts']!=4:
        raise ValueError('The diagnostic requires the agreed 96K/MTP4 configuration')
    if args.repeat_replay_plan:
        arm = args.repeat_replay_arm
        if arm is None:
            parser.error('--repeat-replay-arm is required for a frozen repeat replay')
        root = args.repeat_replay_plan
        warm = json.loads((root/'repeat-warm-request.json').read_text())
        request = json.loads((root/f'repeat-{arm}-request.json').read_text())
        result = replay_repeat_arm(native, warm, request, args.out, arm, 'cards-fresh-model-repeat')
        save(args.out/'result.json', dict(arm=arm, source_plan=str(root), **result))
        save(args.out/'status.json', dict(state='complete', case=arm))
        return
    if args.recorded_repeat_dir:
        profile = CostProfile(**json.loads(args.cost_profile.read_text())) if args.cost_profile else None
        recorded_repeat(native, args.recorded_repeat_dir, args.out, profile)
        return
    prefix='\n'.join(f'Permanent fact {i}: project marker KEY-{(i*37)%997:03d}.' for i in range(256))
    feedback='\n'.join(f'Tool log row {i:04d}: checksum {(i*97)%10000:04d}; old temporary record.' for i in range(1200))
    archive=OriginalArchive(args.out/'originals','native-card-diagnostic')
    record=dict(ordinal=1,commands=[dict(keystrokes='synthetic log fixture; not executed')],feedback=feedback,
                raw_feedback=feedback,feedback_sha256=sha256(feedback))
    description=archive.describe(record);archive.save(description)
    card=build_card(record,description)
    messages=[dict(role='user',content='Keep these permanent facts for the diagnostic.\n'+prefix),
              dict(role='assistant',content='The facts are retained.'),
              dict(role='user',content='Old tool observation:\n'+feedback),
              dict(role='assistant',content='The tool observation is complete.'),
              dict(role='user',content='Reply only with the marker for permanent fact 7. Do not discuss the tool log.')]
    request=dict(model='qwen38-inline-step',prompt_cache_key='card-native-diagnostic',messages=messages,
                 enable_thinking=False,temperature=0,seed=1000,max_tokens=32,logprobs=True,top_logprobs=5)
    changed=dict(request,messages=[dict(m) for m in messages])
    if args.prime_prefix:
        seed=dict(request,messages=[dict(messages[0],content=messages[0]['content']+'\nReply only READY.')])
        request['messages'][0]=seed['messages'][0]
        response=native.json('/v1/chat/completions',seed)
        save(args.out/'seed-request.json',seed);save(args.out/'seed-response.json',response)
        request['messages'][1]=response['choices'][0]['message']
        changed=dict(request,messages=[dict(m) for m in request['messages']])
    changed['messages'][2]['content']='Old tool observation:\n'+card
    prepared=native.json('/apply-template',dict(request,return_tokens=True))
    shortened=native.json('/apply-template',dict(changed,return_tokens=True))
    original_ids=[r['id'] for r in prepared['tokens']]
    short_ids=[r['id'] for r in shortened['tokens']]
    common=common_prefix(original_ids,short_ids)
    result=dict(scope='controlled synthetic input; real model; forced history-edit diagnostic',
                model_and_host_initially_reset=True,production_cost_profile_installed=False,
                original_tokens=len(original_ids),shortened_tokens=len(short_ids),
                unchanged_prefix_tokens=common,archive_sha256=description['sha256'],cases={})
    for label,value in (('original',request),('original_cached',request),('edited',changed),('edited_cached',changed),
                        ('fresh_shortened',dict(changed,prompt_cache_key='card-native-fresh'))):
        save(args.out/'status.json',dict(state='request_running',case=label))
        save(args.out/(label+'-request.json'),value)
        started=time.monotonic()
        response=native.json('/v1/chat/completions',value)
        elapsed=time.monotonic()-started
        save(args.out/(label+'-response.json'),response)
        save(args.out/(label+'-slots.json'),native.json('/slots'))
        if response.get('error'):raise RuntimeError(response['error'])
        result['cases'][label]=dict(wall_seconds=elapsed,timings=response.get('timings'),
            speculative=response.get('kvmem',{}).get('speculative'),
            content=response['choices'][0]['message']['content'])
        save(args.out/'result.json',result)
    expected='KEY-259'
    result['edited_matches_fresh_output']=(result['cases']['edited']['content']==result['cases']['fresh_shortened']['content'])
    result['retained_information_correct']=all(expected in case['content'] for case in result['cases'].values())
    result['cached_prefix_tokens_after_edit']=result['cases']['edited']['timings']['cache_n']
    # Text parity is a functional check, not a proof all internal KV are equal.
    result['kv_bitwise_equality_proven']=False
    save(args.out/'result.json',result)
    if not result['edited_matches_fresh_output'] or not result['retained_information_correct']:
        raise RuntimeError('Edited history did not match a fresh shortened request')
    save(args.out/'status.json',dict(state='complete'))


if __name__=='__main__':main()
