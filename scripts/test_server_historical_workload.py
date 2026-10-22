"""Coding conversations and exact request replay for historical GC comparison.

This is request-level acceptance, not an agent or a SWE-bench evaluator.
Record a reference conversation, then replay identical HTTP bodies for each
mode with --replay PATH/requests.json. Replies are retained for review; no
generated code is executed. Exit zero means HTTP completion without truncation
and observed GC activity when enabled, not correct generated code or a
performance/quality acceptance threshold. --workload selects a fixed multi-turn
task; each arm continues its own responses unless --replay is supplied. The
native model template is the default. --stable-qwen-template is a separate,
text-only experimental fixture, not native-template or agent acceptance.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request


TURNS = [
    "Fix this Python function: def take_last(values, n): return values[-n:]. "
    "It must return a new list containing the last n values, return [] for n=0, "
    "preserve the input list, and raise ValueError for negative n. Return only the complete function.",
    "Add precise type hints for a list of integers. For n greater than the list length, "
    "return a copy of the whole list. Return only the complete function.",
    "Write five small pytest tests for that function, covering zero, negative, oversized, "
    "normal n and preservation of the input list. Return only the test code.",
    "The input can now be a one-pass iterable of integers, too large to turn into a list. "
    "Update take_last to use O(n) auxiliary memory. Return only the complete implementation with imports.",
    "Add two pytest tests for generators: a finite generator with n=3, and an empty generator. "
    "Also check n=0. Return only the test code.",
    "Review the implementation for n=0 and negative n. For n=0 it must not consume "
    "the iterable. Supply the final implementation with imports, keeping the other requirements."
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--server', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=['off', 'shadow', 'delete'], required=True)
    p.add_argument('--policy', choices=['lru', 'redundancy'], default='lru')
    p.add_argument('--workload', type=Path, help='JSON with system and nonempty turns strings')
    p.add_argument('--context-size', type=int, default=4096)
    p.add_argument('--block-tokens', type=int, default=32, help='recorded experimental block size; baseline is 32')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-turns', type=int, help='bounded pilot; explicitly reported as a partial workload')
    p.add_argument('--require-prefix-reuse', action='store_true',
                   help='fail if any continuation recomputes its history instead of extending it')
    p.add_argument('--trace', action='store_true')
    p.add_argument('--stable-qwen-template', action='store_true',
                   help='experimental text-only Qwen ChatML fixture preserving the empty thinking prefix on older replies')
    p.add_argument('--capacity', type=int, required=True)
    p.add_argument('--low', type=int, default=384)
    p.add_argument('--high', type=int, default=512)
    p.add_argument('--budget', type=int, default=1024)
    p.add_argument('--max-tokens', type=int, default=512,
                   help='both request output limit and GPU generation reserve')
    p.add_argument('--replay', type=Path)
    args = p.parse_args()
    if not 1 <= args.max_tokens <= 2048 or not 1 <= args.budget <= 4096:
        p.error('max-tokens must be 1..2048 and budget must be 1..4096')
    if not 1 <= args.capacity <= 65536:
        p.error('capacity must be 1..65536')
    if not 1024 <= args.context_size <= 32768:
        p.error('context-size must be 1024..32768')
    if not 1 <= args.block_tokens <= 256:
        p.error('block-tokens must be 1..256')
    workload = {'system': 'You are a Python coding assistant. Follow the requested output format and keep code concise.',
                'turns': TURNS}
    if args.workload:
        workload = json.loads(args.workload.read_text(encoding='utf-8'))
    if (not isinstance(workload, dict) or not isinstance(workload.get('system'), str) or
            not isinstance(workload.get('turns'), list) or not workload['turns'] or
            any(not isinstance(turn, str) or not turn.strip() for turn in workload['turns'])):
        p.error('workload requires a system string and nonempty turns strings')
    turns = workload['turns']
    if args.max_turns is not None:
        if not 1 <= args.max_turns <= len(turns) or args.replay:
            p.error('max-turns must fit the workload and cannot be combined with replay')
        turns = turns[:args.max_turns]
    if args.mode != 'off' and not 0 < args.low < args.high <= args.capacity:
        p.error('watermarks must satisfy 0 < low < high <= capacity')
    bodies = json.loads(args.replay.read_text(encoding='utf-8')) if args.replay else []
    if args.replay and (not isinstance(bodies, list) or not bodies or
            any(not isinstance(body, dict) or body.get('stream') is not True or
                body.get('max_tokens') != args.max_tokens for body in bodies)):
        p.error('replay must contain streaming requests with the selected max-tokens')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / 'workload.json').write_text(json.dumps(workload, indent=2, ensure_ascii=False), encoding='utf-8')
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    base = f'http://127.0.0.1:{port}'
    env = {k: v for k, v in os.environ.items() if not k.startswith('LLAMA_ARG_') and k != 'LLAMA_API_KEY'}
    command = [str(args.server.resolve(strict=True)), '-m', str(args.model.resolve(strict=True)),
        '--host', '127.0.0.1', '--port', str(port), '-ngl', '99', '-c', str(args.context_size), '-b', '64',
        '-ub', '16', '-n', str(args.max_tokens), '-t', '4', '--no-ui', '--kv-dtype', 'f16',
        '--kvmem-method', 'recency', '--kvmem-block-tokens', str(args.block_tokens), '--kvmem-budget', str(args.budget),
        '--kvmem-gen-reserve', str(args.max_tokens), '--kvmem-historical-capture', str(args.capacity),
        '--kvmem-historical-gc-mode', args.mode, '--kvmem-historical-gc-policy', args.policy,
        '--reasoning-effort', 'none', '-lv', '4']
    if args.trace:
        command += ['--kvmem-trace']
    if args.stable_qwen_template:
        # Qwen's native template removes the empty thinking prefix when a reply
        # becomes historical. This fixture keeps the exact initial formatting;
        # it is explicitly not native-template or agent acceptance.
        template = out / 'stable-qwen.jinja'
        template.write_text(
            "{%- for message in messages %}"
            "{{- '<|im_start|>' + message.role + '\\n' }}"
            "{%- if message.role == 'assistant' %}{{- '<think>\\n\\n</think>\\n\\n' }}{%- endif %}"
            "{{- message.content + '<|im_end|>\\n' }}"
            "{%- endfor %}"
            "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' }}{%- endif %}",
            encoding='utf-8')
        command += ['--chat-template-file', str(template)]
    if args.mode != 'off':
        command += ['--kvmem-historical-gc-low', str(args.low), '--kvmem-historical-gc-high', str(args.high)]
    (out / 'command.json').write_text(json.dumps(command, indent=2), encoding='utf-8')
    results = []
    messages = [{'role': 'system', 'content': workload['system']}]
    process = None
    gpu_monitor = None
    sampler = None
    stopped = threading.Event()

    def sample_process(pid):
        with (out / 'process-memory.jsonl').open('w') as log:
            while not stopped.is_set():
                try:
                    status = Path(f'/proc/{pid}/status').read_text()
                    row = {'monotonic_seconds': time.monotonic()}
                    for line in status.splitlines():
                        key, _, value = line.partition(':')
                        if key in ('VmRSS', 'VmHWM', 'Threads'):
                            row[key] = int(value.split()[0])
                    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
                    row['cpu_seconds'] = (int(fields[11]) + int(fields[12])) / os.sysconf('SC_CLK_TCK')
                    log.write(json.dumps(row) + '\n')
                    log.flush()
                except FileNotFoundError:
                    break
                stopped.wait(1)

    def request(path, body=None, timeout=300):
        data = None if body is None else json.dumps(body, ensure_ascii=False).encode()
        req = urllib.request.Request(base + path, data=data, headers={'Content-Type': 'application/json'})
        return urllib.request.urlopen(req, timeout=timeout)

    def save():
        (out / 'requests.json').write_text(json.dumps(bodies, indent=2, ensure_ascii=False), encoding='utf-8')
        (out / 'results.json').write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')

    try:
        with (out / 'server.log').open('wb') as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            sampler = threading.Thread(target=sample_process, args=(process.pid,))
            sampler.start()
            smi = Path('/usr/lib/wsl/lib/nvidia-smi')
            if smi.exists():
                with (out / 'gpu-memory.csv').open('wb') as gpu_log:
                    gpu_monitor = subprocess.Popen([str(smi),
                        '--query-gpu=timestamp,memory.used,memory.total,utilization.gpu',
                        '--format=csv', '-l', '1'], stdout=gpu_log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 240
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(f'server exited {process.returncode}')
                try:
                    with request('/health', timeout=2) as response:
                        if response.status == 200:
                            break
                except (OSError, urllib.error.URLError):
                    pass
                time.sleep(0.5)
            else:
                raise TimeoutError('server startup timeout')
            with request('/props') as response:
                props_raw = response.read()
                (out / 'props.json').write_bytes(props_raw)
                props = json.loads(props_raw)
                config = props['kvmem']
                if config['generation_limit'] != args.max_tokens:
                    raise RuntimeError('published generation limit differs from requested configuration')
                if config['historical']['capacity'] != args.capacity or config['historical']['mode'] != args.mode:
                    raise RuntimeError('published historical configuration differs from requested configuration')
                if config['historical']['policy'] != args.policy:
                    raise RuntimeError('published historical policy differs from requested configuration')
            for i in range(len(bodies) if args.replay else len(turns)):
                if not args.replay:
                    messages.append({'role': 'user', 'content': turns[i]})
                    body = {'messages': list(messages), 'max_tokens': args.max_tokens, 'temperature': 0,
                            'presence_penalty': 0, 'frequency_penalty': 0, 'repeat_penalty': 1,
                            'seed': args.seed, 'stream': True}
                    bodies.append(body)
                else:
                    body = bodies[i]
                # Save input before submitting: failures must retain their exact body.
                save()
                started = time.monotonic()
                row = {'turn': i + 1, 'started_monotonic_seconds': started, 'body_sha256': hashlib.sha256(
                    json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
                    'content': '', 'reasoning_content': '', 'events': [], 'done': False}
                results.append(row)
                try:
                    with request('/v1/chat/completions', body) as response:
                        row['http_status'] = response.status
                        for raw in response:
                            if not raw.startswith(b'data: '):
                                continue
                            when = time.monotonic() - started
                            payload = raw[6:].strip()
                            if payload == b'[DONE]':
                                row['done'] = True
                                break
                            event = json.loads(payload)
                            row['events'].append({'seconds': when, 'event': event})
                            if 'error' in event:
                                row['error'] = event['error']
                            for choice in event.get('choices', []):
                                delta = choice.get('delta', {})
                                for key in ('content', 'reasoning_content'):
                                    part = delta.get(key) or ''
                                    if part:
                                        row.setdefault('first_text_seconds', when)
                                        row[key] += part
                                if choice.get('finish_reason'):
                                    row['finish_reason'] = choice['finish_reason']
                            for key in ('usage', 'timings'):
                                if key in event:
                                    row[key] = event[key]
                except urllib.error.HTTPError as error:
                    row['http_status'] = error.code
                    row['error_body'] = error.read().decode(errors='replace')
                finally:
                    row['elapsed_seconds'] = time.monotonic() - started
                    save()
                print(json.dumps({k: v for k, v in row.items() if k not in ('events', 'content', 'reasoning_content')}, ensure_ascii=False), flush=True)
                if not row['done'] or row.get('error') or (args.require_prefix_reuse and row.get('finish_reason') != 'stop'):
                    # No hidden retries or reset: a refusal is an observed outcome.
                    break
                if args.require_prefix_reuse and i > 0 and row.get('usage', {}).get('prompt_cache_hit_tokens', 0) <= 0:
                    # A reset would silently restore evicted text and invalidate this experiment.
                    break
                if not args.replay:
                    assistant = {'role': 'assistant', 'content': row['content']}
                    if row['reasoning_content']:
                        assistant['reasoning_content'] = row['reasoning_content']
                    messages.append(assistant)
    finally:
        stopped.set()
        if sampler:
            sampler.join(timeout=3)
        if gpu_monitor and gpu_monitor.poll() is None:
            gpu_monitor.terminate()
            try:
                gpu_monitor.wait(timeout=10)
            except subprocess.TimeoutExpired:
                gpu_monitor.kill()
                gpu_monitor.wait(timeout=10)
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        save()
        expected = len(bodies) if args.replay else len(turns)
        complete = sum(row['done'] and row.get('http_status') == 200 and
                       not row.get('error') and row.get('finish_reason') == 'stop' for row in results)
        gc_samples = []
        if (out / 'server.log').exists():
            for line in (out / 'server.log').read_text(encoding='utf-8', errors='replace').splitlines():
                if 'KVMEM_HISTORICAL_GC finish ' in line:
                    gc_samples.append({key: int(value) for key, value in re.findall(r'(\w+)=(\d+)', line)})
        # These are finish-boundary observations. Do not sum them: repeated
        # requests in the same domain may report overlapping cumulative stats.
        # GPU permission can keep identity alive after CPU payload release.
        # A retirement proposal/commit alone is not physical release evidence.
        activity = args.mode == 'off' or any(
            row.get('shadow', 0) > 0 if args.mode == 'shadow' else
            row.get('committed', 0) > 0 and
            (row.get('reclaimed', 0) > 0 or row.get('host_released_early', 0) > 0)
            for row in gc_samples)
        gc_failed = any(row.get('policy_failures', 0) or row.get('worker_failures', 0) for row in gc_samples)
        prefix_hits = [row.get('usage', {}).get('prompt_cache_hit_tokens', 0) for row in results]
        reused = len(results) == expected and expected > 1 and all(n > 0 for n in prefix_hits[1:])
        summary = {
            'mode': args.mode, 'policy': args.policy, 'capacity': args.capacity, 'budget': args.budget,
            'context_size': args.context_size, 'block_tokens': args.block_tokens,
            'seed': args.seed if not args.replay else None,  # Replay bodies own their sampling settings.
            'chat_template': 'stable_qwen_text_fixture' if args.stable_qwen_template else 'native',
            'conversation': 'fixed_reference_replay' if args.replay else 'own_response_continuation',
            'workload_turns': len(workload['turns']), 'partial_workload': len(turns) < len(workload['turns']),
            'prompt_cache_hit_tokens': prefix_hits, 'all_continuations_reused_history': reused,
            'ready_low': args.low if args.mode != 'off' else 0,
            'ready_high': args.high if args.mode != 'off' else 0,
            'max_tokens': args.max_tokens, 'expected_requests': expected, 'attempted_requests': len(results),
            'completed_without_truncation': complete,
            'truncated_requests': sum(row.get('finish_reason') == 'length' for row in results),
            'refused_requests': sum(row.get('http_status') == 503 for row in results),
            'all_requests_completed_without_truncation': complete == expected,
            'gc_finish_samples': gc_samples, 'required_gc_activity_observed': activity,
            'gc_failure_observed': gc_failed,
            'semantic_quality_evaluated': False,
        }
        (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    if (not summary['all_requests_completed_without_truncation'] or not activity or gc_failed or
            (args.require_prefix_reuse and not reused)):
        raise RuntimeError('requests incomplete, GC pressure/failures, or required prefix reuse failed; see summary.json')


if __name__ == '__main__':
    main()
