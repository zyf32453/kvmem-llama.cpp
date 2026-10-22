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


MENTOR_27B = {
    'budget': 36 * 1024, 'gen_reserve': 16 * 1024, 'max_tokens': 16 * 1024,
    'block_tokens': 32, 'sink_tokens': 1024, 'recent_tokens': 0, 'kv_dtype': 'q8_0',
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--server', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=['off', 'shadow', 'delete'], required=True)
    p.add_argument('--policy', choices=['lru', 'redundancy'], default='lru')
    p.add_argument('--workload', type=Path, help='JSON with system and nonempty turns strings')
    p.add_argument('--preset', choices=['mentor-27b'],
                   help='Qwen 3.8 27B experiment: 32-token blocks, 36K history, 16K generation, 1K sink, recent 0')
    p.add_argument('--dry-run', action='store_true',
                   help='save the requested configuration and command without loading a model or running requests')
    p.add_argument('--context-size', type=int, help='default 4096; mentor-27b uses 65536')
    p.add_argument('--block-tokens', type=int, help='default 32; mentor-27b requires 32')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max-turns', type=int, help='bounded pilot; explicitly reported as a partial workload')
    p.add_argument('--require-prefix-reuse', action='store_true',
                   help='fail if any continuation recomputes its history instead of extending it')
    p.add_argument('--trace', action='store_true')
    p.add_argument('--gpu-layers', type=int, default=99)
    p.add_argument('--kv-dtype', choices=['f16', 'q8_0'],
                   help='default f16; mentor-27b uses the documented IQ3 recipe KV type q8_0')
    p.add_argument('--sink-tokens', type=int, help='default 0 retains one block; mentor-27b requires 1024')
    p.add_argument('--recent-tokens', type=int)
    p.add_argument('--browser-feedback', action='store_true',
                   help='after the fixed task, wait for numbered local-browser feedback files in output/feedback')
    p.add_argument('--max-repairs', type=int, default=3)
    p.add_argument('--feedback-timeout', type=int, default=300)
    p.add_argument('--stable-qwen-template', action='store_true',
                   help='experimental text-only Qwen ChatML fixture preserving the empty thinking prefix on older replies')
    p.add_argument('--capacity', type=int, required=True)
    p.add_argument('--low', type=int, default=384)
    p.add_argument('--high', type=int, default=512)
    p.add_argument('--budget', type=int)
    p.add_argument('--gen-reserve', type=int, help='GPU generation reserve; defaults to max-tokens')
    p.add_argument('--max-tokens', type=int, help='per-request output limit; default 512, mentor-27b 16384')
    p.add_argument('--replay', type=Path)
    args = p.parse_args()
    defaults = {'budget': 1024, 'max_tokens': 512, 'block_tokens': 32,
                'sink_tokens': 0, 'recent_tokens': 0, 'kv_dtype': 'f16'}
    if args.preset == 'mentor-27b':
        for field, value in MENTOR_27B.items():
            if getattr(args, field) not in (None, value):
                p.error(f'mentor-27b requires --{field.replace("_", "-")} {value}')
        defaults.update(MENTOR_27B)
        if args.stable_qwen_template:
            p.error('mentor-27b requires the native model template; the previous Qwen3 fixture is not validated for it')
    for field, value in defaults.items():
        if getattr(args, field) is None:
            setattr(args, field, value)
    if args.context_size is None:
        args.context_size = 65536 if args.preset else 4096
    if args.gen_reserve is None:
        args.gen_reserve = args.max_tokens
    if not 1 <= args.max_tokens <= args.gen_reserve <= 262144 or not 1 <= args.budget <= 262144:
        p.error('require 1 <= max-tokens <= gen-reserve <= 262144 and budget in 1..262144')
    if not 1 <= args.capacity <= 65536:
        p.error('capacity must be 1..65536')
    if not 1024 <= args.context_size <= 262144:
        p.error('context-size must be 1024..262144')
    if args.budget + args.gen_reserve > args.context_size:
        p.error('context-size must cover budget + gen-reserve; budgets are never silently reduced')
    if not 1 <= args.block_tokens <= 256:
        p.error('block-tokens must be 1..256')
    if not 0 <= args.sink_tokens <= args.budget or not 0 <= args.recent_tokens <= args.budget:
        p.error('sink/recent must be between zero and budget')
    if not 0 <= args.max_repairs <= 4 or not 1 <= args.feedback_timeout <= 1800:
        p.error('max-repairs must be 0..4 and feedback-timeout 1..1800')
    if args.browser_feedback and (args.replay or args.max_turns):
        p.error('browser-feedback requires a full own-response workload')
    workload = {'system': 'You are a Python coding assistant. Follow the requested output format and keep code concise.',
                'turns': TURNS}
    if args.workload:
        workload = json.loads(args.workload.read_text(encoding='utf-8'))
    if (not isinstance(workload, dict) or not isinstance(workload.get('system'), str) or
            not isinstance(workload.get('turns'), list) or not workload['turns'] or
            any(not isinstance(turn, str) or not turn.strip() for turn in workload['turns'])):
        p.error('workload requires a system string and nonempty turns strings')
    turns = list(workload['turns'])
    files = list(workload.get('files', []))
    if args.browser_feedback and (workload.get('name') != 'browser-fps-v2' or len(files) != len(turns)):
        p.error('browser-feedback requires the browser-fps-v2 file contract')
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
    if args.browser_feedback:
        (out / 'feedback').mkdir()
    (out / 'workload.json').write_text(json.dumps(workload, indent=2, ensure_ascii=False), encoding='utf-8')
    experiment_config = {field: getattr(args, field) for field in
        ('preset', 'budget', 'gen_reserve', 'max_tokens', 'block_tokens', 'sink_tokens',
         'recent_tokens', 'kv_dtype', 'context_size', 'capacity', 'low', 'high', 'mode', 'policy')}
    experiment_config.update(model_path=str(args.model),
        requested_model='Qwen3.8-27B' if args.preset else None,
        chat_template='stable_qwen_text_fixture' if args.stable_qwen_template else 'native',
        dry_run=args.dry_run, units='tokens except capacity/low/high, which are historical slots')
    (out / 'experiment-config.json').write_text(json.dumps(experiment_config, indent=2), encoding='utf-8')
    port = 0
    if not args.dry_run:
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))
            port = probe.getsockname()[1]
    base = f'http://127.0.0.1:{port}'
    env = {k: v for k, v in os.environ.items() if not k.startswith('LLAMA_ARG_') and k != 'LLAMA_API_KEY'}
    command = [str(args.server.resolve(strict=not args.dry_run)), '-m', str(args.model.resolve(strict=not args.dry_run)),
        '--host', '127.0.0.1', '--port', str(port), '-ngl', str(args.gpu_layers), '-c', str(args.context_size), '-b', '64',
        '-ub', '16', '-n', str(args.max_tokens), '-t', '4', '--no-ui', '--kv-dtype', args.kv_dtype,
        '--kvmem-method', 'recency', '--kvmem-block-tokens', str(args.block_tokens), '--kvmem-budget', str(args.budget),
        '--kvmem-sink-tokens', str(args.sink_tokens), '--kvmem-recent-tokens', str(args.recent_tokens),
        '--kvmem-gen-reserve', str(args.gen_reserve), '--kvmem-historical-capture', str(args.capacity),
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
    if args.dry_run:
        print(json.dumps({'dry_run': True, 'configuration': experiment_config, 'command': command}), flush=True)
        return
    results = []
    messages = [{'role': 'system', 'content': workload['system']}]
    process = None
    gpu_monitor = None
    sampler = None
    stopped = threading.Event()
    browser_passed = None
    feedback_rounds = 0
    feedback_waits = []

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
        # Browser validation needs the final files and status, not thousands
        # of per-token SSE events. Keep the full trace on the remote host.
        (out / 'browser-results.json').write_text(json.dumps(
            [{key: value for key, value in row.items() if key != 'events'} for row in results],
            indent=2, ensure_ascii=False), encoding='utf-8')

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
                if config['generation_limit'] != args.gen_reserve:
                    raise RuntimeError('published generation limit differs from requested configuration')
                if config['historical']['capacity'] != args.capacity or config['historical']['mode'] != args.mode:
                    raise RuntimeError('published historical configuration differs from requested configuration')
                if config['historical']['policy'] != args.policy:
                    raise RuntimeError('published historical policy differs from requested configuration')
                if args.preset == 'mentor-27b':
                    startup_log = (out / 'server.log').read_text(encoding='utf-8', errors='replace')
                    name = re.search(r'general\.name\s+str\s*=\s*([^\r\n]+)', startup_log)
                    if not name or not re.search(r'qwen[ _.-]*3[._]8[ _.-]*27b', name[1], re.I):
                        raise RuntimeError('mentor-27b model metadata does not identify Qwen3.8-27B; refusing a substitute')
                    pool = re.search(r'KVMem slot-pool cells=(\d+) slots=(\d+) block_tokens=(\d+) '
                                     r'budget=(\d+) gen_reserve=(\d+) sink_blocks=(\d+)', startup_log)
                    expected_pool = (53248, 1664, 32, 36864, 16384, 32)
                    if not pool or tuple(map(int, pool.groups())) != expected_pool:
                        raise RuntimeError('effective GPU pool differs from mentor-27b; refusing a reduced configuration')
                    pool_line = next(line for line in startup_log.splitlines() if 'KVMem slot-pool ' in line)
                    if 'type_k=q8_0 type_v=q8_0' not in pool_line:
                        raise RuntimeError('mentor-27b requires q8_0 main K/V as in the IQ3 recipe')
            i = 0
            while i < (len(bodies) if args.replay else len(turns)):
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
                if i < len(files):
                    row['output_file'] = files[i]
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
                i += 1
                if args.browser_feedback and i == len(turns):
                    # The server stays alive: feedback extends this exact KV
                    # history. Browser execution happens on the controller host.
                    checkpoint = {'completed_requests': i, 'repair_index': feedback_rounds,
                                  'max_repairs': args.max_repairs}
                    save()
                    ready = out / 'feedback' / f'checkpoint-{feedback_rounds}.json'
                    ready.write_text(json.dumps(checkpoint), encoding='utf-8')
                    print(json.dumps({'awaiting_browser_feedback': str(ready)}), flush=True)
                    reply = out / 'feedback' / f'feedback-{feedback_rounds}.json'
                    wait_started = time.monotonic()
                    deadline = wait_started + args.feedback_timeout
                    while not reply.exists():
                        if process.poll() is not None:
                            raise RuntimeError('server exited while waiting for browser feedback')
                        if time.monotonic() >= deadline:
                            raise TimeoutError('browser feedback timeout')
                        time.sleep(.25)
                    feedback = json.loads(reply.read_text(encoding='utf-8'))
                    feedback_waits.append(time.monotonic() - wait_started)
                    if feedback.get('completed_requests') != i or feedback.get('action') not in ('pass', 'repair', 'fail'):
                        raise ValueError('invalid or stale browser feedback')
                    browser_passed = feedback['action'] == 'pass'
                    if feedback['action'] != 'repair':
                        break
                    if feedback_rounds >= args.max_repairs:
                        raise ValueError('browser feedback exceeded repair budget')
                    if feedback.get('file') not in workload['files'] or not isinstance(feedback.get('prompt'), str):
                        raise ValueError('repair must target one allowlisted file')
                    files.append(feedback['file'])
                    turns.append(feedback['prompt'])
                    feedback_rounds += 1
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
            'preset': args.preset, 'gen_reserve': args.gen_reserve,
            'kv_dtype': args.kv_dtype,
            'mode': args.mode, 'policy': args.policy, 'capacity': args.capacity, 'budget': args.budget,
            'context_size': args.context_size, 'block_tokens': args.block_tokens,
            'gpu_layers': args.gpu_layers, 'sink_tokens_requested': args.sink_tokens,
            'sink_tokens_effective': max(1, args.sink_tokens // args.block_tokens) * args.block_tokens,
            'recent_tokens_effective': args.recent_tokens // args.block_tokens * args.block_tokens,
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
            'semantic_quality_evaluated': args.browser_feedback and browser_passed is not None,
            'semantic_quality_source': 'external_browser_checks' if args.browser_feedback else None,
            'browser_feedback_enabled': args.browser_feedback, 'repair_requests': feedback_rounds,
            'browser_all_passed': browser_passed,
            'browser_feedback_wait_seconds': feedback_waits,
        }
        (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    if (not summary['all_requests_completed_without_truncation'] or not activity or gc_failed or
            (args.require_prefix_reuse and not reused) or (args.browser_feedback and browser_passed is not True)):
        raise RuntimeError('requests incomplete, GC pressure/failures, or required prefix reuse failed; see summary.json')


if __name__ == '__main__':
    main()
