#!/usr/bin/env python3
"""Isolated native GC service gates; functional evidence, not a performance benchmark.

Runs only the explicitly supplied built workspace/model, owns its Docker containers,
and retains failures. The workspace must contain terminal native build evidence.
"""
import argparse
import hashlib
import http.client
import json
from pathlib import Path
import re
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request

import ninfer_gc_acceptance as requests


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(path):
    value = hashlib.sha256()
    before = path.stat()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(16 << 20), b''):
            value.update(block)
    require(path.stat().st_mtime_ns == before.st_mtime_ns, 'input changed during hashing: '+str(path))
    return value.hexdigest()


def docker(*arguments, check=True, timeout=60):
    return subprocess.run(['docker', *arguments], capture_output=True, text=True,
                          check=check, timeout=timeout)


class Client:
    def __init__(self, port):
        self.port = port
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def get(self, path):
        with self.opener.open(f'http://127.0.0.1:{self.port}'+path, timeout=10) as response:
            return json.load(response)

    def metrics(self):
        with self.opener.open(f'http://127.0.0.1:{self.port}/metrics', timeout=10) as response:
            text = response.read().decode()
        return {name: float(value) for name, value in re.findall(
            r'^([^#\s{]+)\s+([0-9.eE+\-]+)\s*$', text, re.M)}

    def settled(self):
        deadline = time.monotonic()+45
        while time.monotonic() < deadline:
            props = self.get('/props')
            usage = props['kvmem']['usage']
            if usage['active_sessions'] == 0:
                require(usage['active_payload_reservation_bytes'] == 0, 'idle request retained payload reservation')
                return props
            time.sleep(.05)
        raise TimeoutError('request owner failed to settle')

    def complete(self, prompt, max_tokens=160):
        body = dict(messages=[dict(role='user', content=prompt)], max_tokens=max_tokens,
                    thinking_budget=128, temperature=0, stream=False)
        request = urllib.request.Request(f'http://127.0.0.1:{self.port}/v1/chat/completions',
            data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        try:
            with self.opener.open(request, timeout=300) as response:
                return dict(status=response.status, body=json.load(response))
        except urllib.error.HTTPError as error:
            return dict(status=error.code, body=json.load(error))


class Stream:
    def __init__(self, port, prompt):
        self.result = dict(status=None, frames=[], done=False, finished=False)
        self.connection = http.client.HTTPConnection('127.0.0.1', port, timeout=300)
        self.prompt = prompt
        self.thread = threading.Thread(target=self.read, daemon=True)

    def read(self):
        try:
            body = dict(messages=[dict(role='user', content=self.prompt)], max_tokens=4096,
                        thinking_budget=128, temperature=0, stream=True,
                        stream_options={'include_usage': True})
            self.connection.request('POST', '/v1/chat/completions', json.dumps(body),
                                    {'Content-Type': 'application/json'})
            response = self.connection.getresponse()
            self.result['status'] = response.status
            if response.status != 200:
                self.result['body'] = json.load(response)
            else:
                for line in response:
                    if not line.startswith(b'data: '):
                        continue
                    data = line[6:].strip()
                    if data == b'[DONE]':
                        self.result['done'] = True
                        break
                    self.result['frames'].append(json.loads(data))
                    require(len(self.result['frames']) <= 20000, 'unbounded SSE response')
        except Exception as error:
            self.result['transport_error'] = repr(error)
        finally:
            self.result['finished'] = True
            self.connection.close()

    def start(self):
        self.thread.start()
        return self

    def join(self, seconds):
        self.thread.join(seconds)
        require(not self.thread.is_alive(), 'stream reader did not finish')
        return self.result


def bounds(props):
    usage = props['kvmem']['usage']
    require(0 <= usage['gc_aux_bytes'] <= usage['gc_aux_budget_bytes'], 'auxiliary charge outside bound')
    require(usage['gc_aux_peak_bytes'] <= usage['gc_aux_budget_bytes'], 'auxiliary peak outside bound')
    require(usage['host_payload_bytes'] <= usage['host_payload_budget_bytes'], 'Host payload outside bound')
    require(usage['retained_sessions'] <= usage['retained_sessions_limit'], 'retained history count outside bound')
    return usage


def resources(name):
    code = ("from pathlib import Path; import json; "
            "print(json.dumps({'process':Path('/proc/1/status').read_text(),"
            "'cgroup':{n:(Path('/sys/fs/cgroup')/n).read_text() for n in "
            "['memory.peak','memory.current','memory.events'] if (Path('/sys/fs/cgroup')/n).exists()}}))")
    return json.loads(docker('exec', name, '/usr/local/bin/python3.11', '-c', code).stdout)


def await_phase(client, stream, phase, before):
    deadline = time.monotonic()+300
    while time.monotonic() < deadline:
        current = client.metrics()
        prompt = current['llamacpp:prompt_tokens_total']-before['llamacpp:prompt_tokens_total']
        decoded = current['llamacpp:tokens_predicted_total']-before['llamacpp:tokens_predicted_total']
        active = client.get('/props')['kvmem']['usage']['active_sessions']
        if active and ((phase == 'prefill' and prompt >= 512 and decoded == 0) or
                       (phase == 'decode' and decoded >= 16)):
            return dict(prompt_delta=prompt, decode_delta=decoded, active_sessions=active, metrics=current)
        require(not stream.result['finished'], 'stream finished before requested stop phase')
        if phase == 'prefill':
            require(decoded == 0, 'missed prefill; refusing to mislabel a decode stop')
        time.sleep(.02)
    raise TimeoutError('execution phase not observed')


def check_error_stream(response):
    require(not response.get('transport_error') and response['finished'], 'error response truncated HTTP transport')
    if response['status'] == 200:
        errors = [frame['error'] for frame in response['frames'] if frame.get('error')]
        # Native error streams end with an error event and a completed HTTP
        # chunked body. The success-only [DONE] sentinel is not required here.
        require(len(errors) == 1 and errors[0]['code'] == 'service_unavailable' and
                response['frames'][-1].get('error') == errors[0], 'stream lacked a typed terminal error')
    else:
        require(response['status'] == 503 and response['body']['error']['code'] == 'service_unavailable',
                'unexpected unavailable response')


def shutdown_case(client, name, phase, result):
    before = client.metrics()
    prompt = requests.request_prompt('sequence', True, 512 if phase == 'prefill' else 128)
    stream = Stream(client.port, prompt).start()
    result['at_stop'] = await_phase(client, stream, phase, before)
    queued = Stream(client.port, 'Reply with READY.').start()
    deadline = time.monotonic()+30
    while time.monotonic() < deadline:
        current = client.metrics()
        if current['llamacpp:requests_deferred'] >= 1:
            require(current['llamacpp:requests_processing'] == 1, 'queue test lost its running request')
            if phase == 'prefill':
                require(current['llamacpp:tokens_predicted_total'] == before['llamacpp:tokens_predicted_total'],
                        'prefill finished before the queued stop')
            result['at_stop_with_queue'] = current
            break
        require(not queued.result['finished'], 'second request did not enter the admission queue')
        time.sleep(.02)
    else:
        raise TimeoutError('second request never entered the admission queue')
    result['resources_before_stop'] = resources(name)
    started = time.monotonic()
    docker('stop', '--time', '20', name, timeout=35)
    result['stop_seconds'] = time.monotonic()-started
    state = json.loads(docker('inspect', name).stdout)[0]['State']
    result['terminal_state'] = state
    result['stream'] = stream.join(10)
    result['queued_stream'] = queued.join(10)
    require(not state['Running'] and state['ExitCode'] == 0 and not state['OOMKilled'],
            'server did not exit gracefully: '+str(state))
    check_error_stream(result['stream'])
    check_error_stream(result['queued_stream'])


def churn_case(client, name, result, args):
    result['requests'] = []
    for iteration in range(args.iterations):
        prompt = f'Fresh lifecycle root {iteration:04d}.\n'+requests.request_prompt('math', False, args.records)
        started = time.monotonic()
        response = client.complete(prompt)
        item = dict(iteration=iteration, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                    response=response, elapsed_seconds=time.monotonic()-started)
        # Persist each failure, even when it prevents a subsequent observation.
        result['requests'].append(item)
        item['after'] = client.settled()
        item['resources'] = resources(name)
        bounds(item['after'])
        require(response['status'] == 200, 'churn request rejected: '+json.dumps(response))
        require(response['body']['choices'][0]['message']['content'].strip() == '599', 'churn answer incorrect')
        require(item['after']['kvmem']['usage']['gc_aux_denied'] == 0, 'churn used an auxiliary rejection')
        print(json.dumps(dict(event='churn', iteration=iteration,
              auxiliary_bytes=item['after']['kvmem']['usage']['gc_aux_bytes'],
              auxiliary_peak=item['after']['kvmem']['usage']['gc_aux_peak_bytes'],
              evictions=item['after']['kvmem']['usage']['history_evictions'])), flush=True)
    result['after'] = client.settled()
    usage = bounds(result['after'])
    require(usage['history_evictions'] > result['before']['kvmem']['usage']['history_evictions'],
            'churn did not exercise retained-domain eviction')


def stream_quota_case(client, result):
    stream = Stream(client.port, requests.request_prompt('sequence', True, 8)).start()
    deadline = time.monotonic()+300
    while time.monotonic() < deadline:
        if any(choice.get('delta', {}).get('content') or choice.get('delta', {}).get('reasoning_content')
               for frame in stream.result['frames'] for choice in frame.get('choices', [])):
            break
        require(not stream.result['finished'], 'quota stream finished before output began')
        time.sleep(.02)
    else:
        raise TimeoutError('quota stream never began output')
    queued = Stream(client.port, 'Compute 17 * 35 + 4. Reply with only the integer.').start()
    deadline = time.monotonic()+30
    while time.monotonic() < deadline:
        current = client.metrics()
        if current['llamacpp:requests_deferred'] >= 1:
            result['queued_before_failure'] = current
            break
        require(not queued.result['finished'], 'follow-up did not enter the admission queue')
        time.sleep(.02)
    else:
        raise TimeoutError('follow-up never entered the admission queue')
    result['stream'] = stream.join(300)
    result['queued_stream'] = queued.join(30)
    frames = result['stream']['frames']
    content_frames = [frame for frame in frames if any(
        choice.get('delta', {}).get('content') or choice.get('delta', {}).get('reasoning_content')
        for choice in frame.get('choices', []))]
    errors = [frame['error'] for frame in frames if frame.get('error')]
    require(result['stream']['status'] == 200 and content_frames, 'quota failure did not occur after output started')
    check_error_stream(result['stream'])
    require(len(errors) == 1 and errors[0]['code'] == 'service_unavailable', 'quota stream missing typed error')
    require('auxiliary' in errors[0]['message'].lower(), 'stream failed for a different reason')
    queued_response = result['queued_stream']
    require(queued_response['status'] == 200 and queued_response['done'] and
            not queued_response.get('transport_error') and
            not any(frame.get('error') for frame in queued_response['frames']), 'pending FIFO request failed')
    queued_text = ''.join(choice.get('delta', {}).get('content') or ''
                          for frame in queued_response['frames'] for choice in frame.get('choices', []))
    require(queued_text.strip() == '599', 'pending FIFO answer incorrect')
    result['after'] = client.settled()
    require(bounds(result['after'])['gc_aux_denied'] > 0, 'no quota denial observed')
    result['followup'] = client.complete('Compute 17 * 35 + 4. Reply with only the integer.')
    require(result['followup']['status'] == 200 and
            result['followup']['body']['choices'][0]['message']['content'].strip() == '599', 'post-failure follow-up failed')
    result['after_followup'] = client.settled()
    bounds(result['after_followup'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--model-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--configs', nargs='+', choices=('lru-none', 'redundancy-none', 'lru-mtp4', 'redundancy-mtp4'),
                        default=['lru-none', 'redundancy-none', 'lru-mtp4', 'redundancy-mtp4'])
    parser.add_argument('--cases', nargs='+', choices=('shutdown-prefill', 'shutdown-decode', 'churn', 'stream-quota'), required=True)
    parser.add_argument('--iterations', type=int, default=12)
    parser.add_argument('--records', type=int, default=256)
    parser.add_argument('--aux-mib', type=int, default=1024)
    args = parser.parse_args()
    require(args.iterations > 4 and args.records > 0 and args.aux_mib > 0, 'invalid finite workload/budget')
    args.root = args.root.resolve();args.model = args.model.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    native = args.root/'native-logs'
    require(json.loads((native/'phase.json').read_text())['phase'] == 'completed', 'native build not completed')
    binary = args.root/'build-native/apps/ninfer-serve'
    binary_hash = digest(binary)
    require(binary_hash == json.loads((native/'binaries.json').read_text())['build-native/apps/ninfer-serve'], 'native binary identity changed')
    require(digest(args.model) == args.model_sha256, 'model identity mismatch')
    metadata = dict(binary_sha256=binary_hash, model_sha256=args.model_sha256, model=str(args.model),
                    script_sha256=digest(Path(__file__)), request_helper_sha256=digest(Path(requests.__file__)),
                    image=json.loads(docker('image', 'inspect', args.image).stdout)[0]['Id'], cases=[],
                    scope='single GPU/RAM/one active request; functional gates, no RSS bound or performance qualification')
    (args.out/'run.json').write_text(json.dumps(metadata, indent=2))
    for config in args.configs:
        for case in args.cases:
            policy = config.rsplit('-', 1)[0];spec = 'mtp' if config.endswith('mtp4') else 'none'
            if case == 'stream-quota':
                require(policy == 'redundancy', 'stream quota fixture requires redundancy feature growth')
            auxiliary = 8 if case == 'stream-quota' else args.aux_mib
            with socket.socket() as sock:
                sock.bind(('127.0.0.1', 0));port = sock.getsockname()[1]
            name = 'kvmem-life-'+config+'-'+case+'-'+str(time.time_ns())
            label = config+'-'+case
            command = ['run', '-d', '--name', name, '--gpus', 'all', '-p', f'127.0.0.1:{port}:{port}',
                '-v', f'{args.root}:/work:ro', '-v', f'{args.model.parent}:/models:ro',
                '-v', '/usr/local/cuda-13.2:/usr/local/cuda-13.2:ro',
                '-v', '/usr/lib/wsl/lib:/usr/lib/wsl/lib:ro', '-e', 'KVMEM_GC_AUDIT=1',
                '-e', 'LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/lib:/usr/local/cuda-13.2/lib64', args.image,
                '/work/build-native/apps/ninfer-serve', '/models/'+args.model.name,
                '--host', '0.0.0.0', '--port', str(port), '--max-context', '262144', '--prefill-chunk', '512',
                '--kv-dtype', 'rk4v4', '--max-concurrency', '1', '--device-profile', 'off',
                '--kvmem-budget', '2048', '--kvmem-gen-reserve', '512', '--kvmem-host-mib', '96',
                '--kvmem-query-scorer', 'meanq-cosine', '--kvmem-score-agg', 'max', '--no-kvmem-lrfu',
                '--kvmem-graph-alpha', '0', '--kvmem-query-max-tokens', '8192', '--kvmem-query-replay-max-tokens', '8192',
                '--kvmem-verify-transfers', '--kvmem-gc-'+policy, '--kvmem-gc-aux-mib', str(auxiliary),
                '--default-thinking-budget', '128', '--spec', spec]
            if spec == 'mtp':command += ['--draft-tokens', '4', '--lm-head-draft']
            if case == 'churn':command += ['--derive-session-keys']
            result = dict(config=config, case=case, command=['docker', *command], aux_mib=auxiliary,
                          host_mib=96, records=args.records, iterations=args.iterations, passed=False)
            created = False
            try:
                result['container_id'] = docker(*command).stdout.strip();created = True
                client = Client(port);deadline = time.monotonic()+240
                while True:
                    state = json.loads(docker('inspect', name).stdout)[0]['State']
                    require(state['Running'], 'owned service exited during startup: '+str(state))
                    try:
                        result['before'] = client.get('/props');break
                    except (OSError, TimeoutError):
                        if time.monotonic() >= deadline:raise TimeoutError('service readiness')
                        time.sleep(.5)
                bounds(result['before'])
                if case.startswith('shutdown-'):shutdown_case(client, name, case.split('-', 1)[1], result)
                elif case == 'churn':churn_case(client, name, result, args)
                else:stream_quota_case(client, result)
                result['passed'] = True
            except Exception as error:
                result['error'] = repr(error)
            finally:
                if created:
                    docker('stop', '--time', '20', name, check=False, timeout=35)
                    logs = docker('logs', name, check=False)
                    (args.out/(label+'.server.log')).write_text(logs.stdout+logs.stderr)
                    result['final_state'] = json.loads(docker('inspect', name).stdout)[0]['State']
                    docker('rm', '-f', name, check=False)
                (args.out/(label+'.json')).write_text(json.dumps(result, indent=2))
                metadata['cases'].append(dict(config=config, case=case, passed=result['passed'], error=result.get('error')))
                (args.out/'run.json').write_text(json.dumps(metadata, indent=2))
                print(json.dumps(metadata['cases'][-1]), flush=True)
            if not result['passed']:return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
