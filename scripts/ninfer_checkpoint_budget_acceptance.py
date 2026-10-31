#!/usr/bin/env python3
"""Qualify checkpoint-owner quota, real pinned aliases and source regressions."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import time
import uuid

from ninfer_source_recovery_acceptance import command, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--image', default='kvmem-lifecycle-builder:20261031-d311d1b')
    args = parser.parse_args()
    args.root = args.root.resolve(); args.model = args.model.resolve(); args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    assert json.loads((args.root/'native-logs/phase.json').read_text())['phase'] == 'completed'
    if args.model.stat().st_size != 15017456896 or digest(args.model) != '29e54467ca0ddae7e9aac04439f72ad20dbf6dc8fb5e84833b41ec7de9c46750':
        raise RuntimeError('model differs from frozen scope')
    hashes = json.loads((args.root/'native-logs/binaries.json').read_text())
    report = dict(completed=False, root=str(args.root), model=str(args.model),
        native_inputs=json.loads((args.root/'inputs.json').read_text()), script_sha256=digest(Path(__file__)),
        helper_sha256=digest(Path(__file__).with_name('ninfer_source_recovery_acceptance.py')),
        image_id=command(['docker', 'image', 'inspect', '--format', '{{.Id}}', args.image]).strip(), cases=[],
        scope='bounded checkpoint C, quota cleanup and source regressions; no RSS/performance/Agent qualification')
    gpu = '/usr/lib/wsl/lib/nvidia-smi'

    def save():
        (args.out/'status.json').write_text(json.dumps(report, indent=2)+'\n')

    def run(config, binary, env, parameters=()):
        if digest(args.root/binary) != hashes[binary]: raise RuntimeError('binary changed: '+binary)
        if command([gpu, '--query-compute-apps=pid,process_name', '--format=csv,noheader']).strip():
            raise RuntimeError('GPU has another compute owner')
        name = 'kvmem-checkpoint-budget-'+uuid.uuid4().hex[:12]
        argv = ['docker', 'run', '-d', '--name', name, '--gpus', 'all',
            '-v', str(args.root)+':/work:ro', '-v', str(args.model.parent)+':/models:ro',
            '-v', '/usr/local/cuda-13.2:/usr/local/cuda-13.2:ro', '-v', '/usr/lib/wsl/lib:/usr/lib/wsl/lib:ro',
            '-e', 'LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/lib:/usr/local/cuda-13.2/lib64']
        for key, value in env.items(): argv += ['-e', key+'='+value]
        argv += [args.image, '/work/'+binary, *parameters]
        case = dict(config=config, argv=argv, binary_sha256=hashes[binary], passed=False, terminal=False)
        report['cases'].append(case); save(); created = False
        try:
            case['id'] = command(argv).strip(); created = True
            case['gpu'] = command([gpu, '--query-gpu=uuid,name,driver_version,memory.total', '--format=csv,noheader']).strip()
            deadline = time.monotonic()+1800
            while True:
                state = json.loads(command(['docker', 'inspect', '--format', '{{json .State}}', name]))
                if not state['Running']: break
                if time.monotonic() >= deadline: raise TimeoutError(config+' exceeded fixed 30-minute timeout')
                time.sleep(1)
            case['terminal_state'] = state; case['terminal'] = True
            logs = subprocess.run(['docker', 'logs', name], capture_output=True, text=True)
            if state['ExitCode'] != 0 or state['OOMKilled']:
                raise RuntimeError(config+' failed: '+str(state['ExitCode'])+' '+logs.stderr[-2400:])
            if config != 'pinned-pool-contract':
                records = [json.loads(line) for line in logs.stdout.splitlines() if line.startswith('{')]
                result = next(r for r in reversed(records) if r.get('status') == 'passed')
                case['result'] = result
                if config.startswith('quota-'):
                    case['checkpoint_evictions'] = len(re.findall('KVMEM_CHECKPOINT_EVICT ', logs.stderr))
                    if not case['checkpoint_evictions']: raise RuntimeError('no actual C-triggered eviction')
                    if result['peak_bytes'] > result['budget_bytes'] or not result['mandatory_quota_refused'] or not result['followup_after_refusal']:
                        raise RuntimeError('checkpoint budget/cleanup qualification failed')
                elif config.startswith('recovery-'):
                    before = logs.stderr.split('KVMEM_SOURCE_REBUILD_BEGIN', 1)[0]
                    spans = re.findall(r'KVMEM_GC_AUDIT event=reclaim [^\n]*original_begin=(\d+) valid_tokens=(\d+) [^\n]*host_released=1', before)
                    if not any(int(b)>0 and int(a)+int(b)<=1024 for a,b in spans):
                        raise RuntimeError('recovery regression lacks physically freed prefix page')
                elif config.startswith('forget-') and (result['forgotten_sources'] != 3 or result['discarded_native_domains'] != 3):
                    raise RuntimeError('source family regression was weakened')
            case['passed'] = True
        except Exception as error:
            case['error'] = repr(error); raise
        finally:
            if created:
                subprocess.run(['docker', 'stop', '--time', '20', name], capture_output=True, text=True)
                state = subprocess.run(['docker', 'inspect', '--format', '{{json .State}}', name], capture_output=True, text=True)
                if state.returncode == 0: case['final_container_state'] = json.loads(state.stdout)
                logs = subprocess.run(['docker', 'logs', name], capture_output=True, text=True)
                (args.out/(config+'.stdout.log')).write_text(logs.stdout)
                (args.out/(config+'.stderr.log')).write_text(logs.stderr)
                subprocess.run(['docker', 'rm', name], capture_output=True, text=True, check=True)
            save()
        print(json.dumps(dict(config=config, passed=case['passed'])), flush=True)

    save()
    run('pinned-pool-contract', 'build-native/tests/ninfer_query_policy_tests', {}, ['ninfer_kvmem_checkpoint_pool_test'])
    for config in ['lru-none', 'lru-mtp4', 'redundancy-none', 'redundancy-mtp4']:
        env = {'NINFER_CHECKPOINT_BUDGET_ARTIFACT': '/models/'+args.model.name}
        if config.endswith('mtp4'): env['NINFER_CHECKPOINT_BUDGET_MTP'] = '1'
        if config.startswith('redundancy'): env['NINFER_CHECKPOINT_BUDGET_REDUNDANCY'] = '1'
        run('quota-'+config, 'build-native/tests/ninfer_engine_checkpoint_budget_real_test', env)
    run('recovery-lru-mtp4', 'build-native/tests/ninfer_engine_source_recovery_real_test',
        {'NINFER_SOURCE_RECOVERY_ARTIFACT': '/models/'+args.model.name, 'NINFER_SOURCE_RECOVERY_MTP': '1', 'KVMEM_GC_AUDIT': '1'})
    run('forget-lru-mtp4', 'build-native/tests/ninfer_engine_source_forget_real_test',
        {'NINFER_SOURCE_FORGET_ARTIFACT': '/models/'+args.model.name, 'NINFER_SOURCE_FORGET_MTP': '1'})
    report['completed'] = True; save()


if __name__ == '__main__':
    main()
