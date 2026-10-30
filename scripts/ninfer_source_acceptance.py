#!/usr/bin/env python3
"""Isolated accepted-source functional gates; no recovery/performance claim."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import struct
import time
import urllib.error
import urllib.request

from ninfer_lifecycle_service_acceptance import Client, Stream, await_phase, bounds, check_error_stream, digest, docker, require
import ninfer_gc_acceptance as prompts


def post(client, path, data):
    request = urllib.request.Request(f'http://127.0.0.1:{client.port}'+path,
        data=json.dumps(data).encode(), headers={'Content-Type': 'application/json'})
    try:
        with client.opener.open(request, timeout=300) as response:
            return dict(status=response.status, body=json.load(response))
    except urllib.error.HTTPError as error:
        return dict(status=error.code, body=json.load(error))


def raw(client, tokens, count=1):
    return post(client, '/completion', dict(prompt=tokens, n_predict=count, return_tokens=True,
        cache_prompt=False, temperature=0, seed=42, stream=False))


def fnv(tokens):
    value = 14695981039346656037
    for token in tokens:
        for byte in struct.pack('<i', token):
            value = ((value ^ byte)*1099511628211) & ((1 << 64)-1)
    return value


def source_bounds(props):
    usage = bounds(props)
    require(0 <= usage['source_bytes'] <= usage['source_peak_bytes'] <= usage['source_budget_bytes'], 'source quota exceeded')
    require(usage['source_records'] <= 256, 'source directory exceeded its explicit entry bound')
    require(usage['source_last_computed_tokens'] <= usage['source_last_accepted_tokens'], 'source computed rows exceed accepted rows')
    return usage


def raw_checked(client, tokens, count):
    response = raw(client, tokens, count)
    after = client.settled()
    usage = source_bounds(after)
    if response['status'] == 200:
        output = response['body']['tokens']
        require(output and len(output) <= count, 'completion did not return actual accepted token IDs')
        require(usage['source_last_accepted_tokens'] == len(tokens)+len(output), 'source accepted uncommitted rows or lost output')
        require(usage['source_last_computed_tokens']+1 == usage['source_last_accepted_tokens'], 'last sampled row was falsely marked computed')
        require(usage['source_last_tokens_fnv1a64'] == fnv(tokens+output), 'actual journal bytes differ from public input and accepted output IDs')
    else:
        error = response['body'].get('error', {})
        require(response['status'] == 503 and error.get('code') == 'service_unavailable' and
                'source' in error.get('message', '').lower(), 'failure was not typed source quota rejection: '+str(response))
    return dict(response=response, after=after)


def capture(client, result, args):
    text = prompts.request_prompt('math', False, args.records)
    text = '<|im_start|>user\n'+text+'<|im_end|>\n<|im_start|>assistant\n'
    tokenized = post(client, '/tokenize', dict(content=text, add_special=False, parse_special=True))
    require(tokenized['status'] == 200, 'tokenization failed')
    tokens = tokenized['body']['tokens']
    before = source_bounds(result['before'])
    result['capture'] = raw_checked(client, tokens, 64)
    require(result['capture']['response']['status'] == 200, 'capture request rejected')
    after = source_bounds(result['capture']['after'])
    require(after['source_records'] == before['source_records']+1, 'request did not create one independent source record')
    result['input_tokens'] = len(tokens)
    result['input_sha256'] = hashlib.sha256(b''.join(struct.pack('<i', x) for x in tokens)).hexdigest()
    # Another request must not discard the prior source when the window changes.
    result['followup'] = raw_checked(client, tokens[-16:], 1)
    require(result['followup']['response']['status'] == 200 and
        source_bounds(result['followup']['after'])['source_records'] == after['source_records']+1, 'later request lost an older source')


def quota(client, result, args):
    tokenized = post(client, '/tokenize', dict(content='A', add_special=False, parse_special=True))
    require(tokenized['status'] == 200 and tokenized['body']['tokens'], 'no filler token')
    token = tokenized['body']['tokens'][0]
    # Measure the real small-record/event footprint instead of assuming its C++ ABI.
    before = source_bounds(client.settled())
    result['small'] = raw_checked(client, [token], 1)
    require(result['small']['response']['status'] == 200, 'initial small source rejected')
    current = source_bounds(result['small']['after'])
    event_bytes = current['source_bytes']-before['source_bytes']-64*4
    require(event_bytes > 0, 'source fixture did not observe a real event buffer')
    result['fill'] = []
    for iteration in range(32):
        item = raw_checked(client, [token]*8192, 1)
        result['fill'].append(item)
        print(json.dumps(dict(event='source-fill', iteration=iteration, status=item['response']['status'])), flush=True)
        if item['response']['status'] == 503:
            break
    else:
        raise RuntimeError('finite fixture never reached source pressure')
    usage = source_bounds(result['fill'][-1]['after'])
    require(usage['source_denied'] > 0, 'no real source denial observed')
    free = usage['source_budget_bytes']-usage['source_bytes']
    # Leave 8 KiB for a short stream. The filler input fits; doubling it for its
    # first output cannot fit. Rejection preserves that accepted input record.
    count = (free-8192-event_bytes)//4
    require(count > 64, 'unexpected fixture headroom; refusing a different failure scenario')
    result['reserve_tail'] = raw_checked(client, [token]*count, 1)
    require(result['reserve_tail']['response']['status'] == 503, 'filler did not refuse its uncommitted output')
    tail = source_bounds(result['reserve_tail']['after'])
    require(tail['source_last_accepted_tokens'] == count and tail['source_last_tokens_fnv1a64'] == fnv([token]*count),
        'failed output partially entered source')
    stream = Stream(client.port, prompts.request_prompt('sequence', True, 8)).start()
    deadline = time.monotonic()+60
    while time.monotonic() < deadline:
        if any(c.get('delta', {}).get('content') or c.get('delta', {}).get('reasoning_content')
               for f in stream.result['frames'] for c in f.get('choices', [])):
            break
        require(not stream.result['finished'], 'source stream failed before any output')
        time.sleep(.02)
    else:
        raise TimeoutError('stream never started')
    queued = Stream(client.port, 'Compute 17 * 35 + 4. Reply with only the integer.').start()
    deadline = time.monotonic()+30
    while time.monotonic() < deadline:
        metrics = client.metrics()
        if metrics['llamacpp:requests_deferred'] >= 1:
            result['queued_before_failure'] = metrics;break
        require(not queued.result['finished'], 'followup never queued')
        time.sleep(.02)
    else:
        raise TimeoutError('followup never entered FIFO')
    result['stream'] = stream.join(300)
    check_error_stream(result['stream'])
    require('source' in result['stream']['frames'][-1]['error']['message'].lower(), 'stream failed outside source quota')
    result['queued'] = queued.join(60)
    require(result['queued']['status'] == 200 and result['queued']['done'] and
        not result['queued'].get('transport_error') and
        not any(f.get('error') for f in result['queued']['frames']), 'pending request did not recover')
    text = ''.join(c.get('delta', {}).get('content') or '' for f in result['queued']['frames'] for c in f.get('choices', []))
    require(text.strip() == '599', 'pending request returned wrong answer')
    result['after'] = client.settled();source_bounds(result['after'])


def cancel(client, result, phase):
    prompt = prompts.request_prompt('sequence', True, 512 if phase=='prefill' else 128)
    template = post(client, '/apply-template', dict(messages=[dict(role='user',content=prompt)],
        max_tokens=4096, thinking_budget=128, temperature=0))
    require(template['status']==200, 'could not obtain exact prepared input serialization')
    tokenized = post(client, '/tokenize', dict(content=template['body']['prompt'],parse_special=True))
    require(tokenized['status']==200, 'cancel input tokenization failed')
    tokens = tokenized['body']['tokens']
    before = client.metrics()
    stream = Stream(client.port,prompt).start()
    result['at_cancel'] = await_phase(client,stream,phase,before)
    connection = stream.connection.sock
    require(connection is not None and not stream.result['finished'], 'stream ended before disconnect')
    connection.shutdown(socket.SHUT_RDWR)
    stream.connection.close()
    result['disconnected_stream'] = stream.join(30)
    result['after_cancel'] = client.settled()
    usage = source_bounds(result['after_cancel'])
    require(usage['source_records']==result['before']['kvmem']['usage']['source_records']+1,
        'cancellation discarded an accepted source record')
    if phase=='prefill':
        require(usage['source_last_accepted_tokens']==len(tokens) and
            usage['source_last_tokens_fnv1a64']==fnv(tokens) and
            usage['source_last_computed_tokens']<len(tokens),
            'cancelled prefill confused admitted input, computed KV or provisional output')
    else:
        require(usage['source_last_accepted_tokens']>len(tokens) and
            usage['source_last_computed_tokens']+1==usage['source_last_accepted_tokens'],
            'cancelled generation lost its accepted frontier')
    result['input_tokens']=len(tokens)
    result['followup']=raw_checked(client,tokens[-16:],1)
    require(result['followup']['response']['status']==200,'request after cancellation failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--model-sha256', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--configs', nargs='+', choices=['lru-none','redundancy-none','lru-mtp4','redundancy-mtp4'], default=['lru-none','lru-mtp4'])
    parser.add_argument('--cases', nargs='+', choices=['capture','quota','cancel-prefill','cancel-decode'], default=['capture'])
    parser.add_argument('--records', type=int, default=256)
    args = parser.parse_args()
    args.root=args.root.resolve();args.model=args.model.resolve();args.out.mkdir(parents=True,exist_ok=False)
    logs=args.root/'native-logs';require(json.loads((logs/'phase.json').read_text())['phase']=='completed', 'native checks incomplete')
    binary=digest(args.root/'build-native/apps/ninfer-serve')
    require(binary==json.loads((logs/'binaries.json').read_text())['build-native/apps/ninfer-serve'], 'binary identity changed')
    require(digest(args.model)==args.model_sha256, 'model identity changed')
    meta=dict(binary_sha256=binary,model_sha256=args.model_sha256,script_sha256=digest(Path(__file__)),
        image=json.loads(docker('image','inspect',args.image).stdout)[0]['Id'],cases=[],
        scope='accepted text source recording only; no rebuild, persistence or performance qualification')
    for config in args.configs:
        for case in args.cases:
            source_mib = 1 if case=='quota' else 16
            with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
            name='kvmem-source-'+config+'-'+case+'-'+str(time.time_ns())
            policy=config.rsplit('-',1)[0];mtp=config.endswith('mtp4')
            command=['run','-d','--name',name,'--gpus','all','-p',f'127.0.0.1:{port}:{port}',
                '-v',f'{args.root}:/work:ro','-v',f'{args.model.parent}:/models:ro',
                '-v','/usr/local/cuda-13.2:/usr/local/cuda-13.2:ro','-v','/usr/lib/wsl/lib:/usr/lib/wsl/lib:ro',
                '-e','LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/lib:/usr/local/cuda-13.2/lib64',
                '-e','KVMEM_GC_AUDIT=1',args.image,'/work/build-native/apps/ninfer-serve','/models/'+args.model.name,
                '--host','0.0.0.0','--port',str(port),'--max-context','262144','--prefill-chunk','512',
                '--kv-dtype','rk4v4','--max-concurrency','1','--device-profile','off',
                '--kvmem-budget','2048','--kvmem-gen-reserve','512','--kvmem-host-mib','96',
                '--kvmem-query-scorer','meanq-cosine','--kvmem-score-agg','max','--no-kvmem-lrfu',
                '--kvmem-graph-alpha','0','--kvmem-query-max-tokens','8192','--kvmem-query-replay-max-tokens','8192',
                '--kvmem-verify-transfers','--kvmem-gc-'+policy,'--kvmem-source-mib',str(source_mib),
                '--default-thinking-budget','128','--spec','mtp' if mtp else 'none']
            if mtp:command+=['--draft-tokens','4','--lm-head-draft']
            result=dict(config=config,case=case,source_mib=source_mib,command=['docker',*command],passed=False)
            created=False
            try:
                result['container_id']=docker(*command).stdout.strip();created=True
                client=Client(port);deadline=time.monotonic()+240
                while True:
                    state=json.loads(docker('inspect',name).stdout)[0]['State']
                    require(state['Running'],'service failed at startup: '+str(state))
                    try:result['before']=client.get('/props');break
                    except (OSError,TimeoutError):
                        if time.monotonic()>=deadline:raise TimeoutError('service readiness')
                        time.sleep(.5)
                source_bounds(result['before'])
                if case=='capture':capture(client,result,args)
                elif case=='quota':quota(client,result,args)
                else:cancel(client,result,case.split('-',1)[1])
                result['passed']=True
            except Exception as error:result['error']=repr(error)
            finally:
                if created:
                    docker('stop','--time','20',name,check=False,timeout=35)
                    log=docker('logs',name,check=False)
                    (args.out/(config+'-'+case+'.server.log')).write_text(log.stdout+log.stderr)
                    result['final_state']=json.loads(docker('inspect',name).stdout)[0]['State']
                    docker('rm','-f',name,check=False)
                (args.out/(config+'-'+case+'.json')).write_text(json.dumps(result,indent=2))
                meta['cases'].append(dict(config=config,case=case,passed=result['passed'],error=result.get('error')))
                (args.out/'run.json').write_text(json.dumps(meta,indent=2))
                print(json.dumps(meta['cases'][-1]),flush=True)
            if not result['passed']:return 1
    return 0


if __name__=='__main__':raise SystemExit(main())
