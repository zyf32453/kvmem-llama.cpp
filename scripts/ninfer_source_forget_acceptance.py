#!/usr/bin/env python3
"""Qualify Engine-local source-family forgetting and native alias revocation."""
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
    relative = 'build-native/tests/ninfer_engine_source_forget_real_test'
    expected = json.loads((args.root/'native-logs/binaries.json').read_text())[relative]
    if digest(args.root/relative) != expected: raise RuntimeError('qualification binary changed')
    if args.model.stat().st_size != 15017456896 or digest(args.model) != '29e54467ca0ddae7e9aac04439f72ad20dbf6dc8fb5e84833b41ec7de9c46750':
        raise RuntimeError('model differs from frozen support scope')
    gpu = '/usr/lib/wsl/lib/nvidia-smi'
    report = dict(completed=False, root=str(args.root), model=str(args.model),
        native_inputs=json.loads((args.root/'inputs.json').read_text()), binary_sha256=expected,
        script_sha256=digest(Path(__file__)), helper_sha256=digest(Path(__file__).with_name('ninfer_source_recovery_acceptance.py')),
        image_id=command(['docker', 'image', 'inspect', '--format', '{{.Id}}', args.image]).strip(),
        scope='process-local explicit source families, idle RAM Engine; no persisted/global external erasure or performance claim', cases=[])

    def save():
        (args.out/'status.json').write_text(json.dumps(report, indent=2)+'\n')
    save()
    for config in ['lru-none', 'lru-mtp4', 'redundancy-none', 'redundancy-mtp4']:
        processes = command([gpu, '--query-compute-apps=pid,process_name', '--format=csv,noheader']).strip()
        if processes: raise RuntimeError('GPU compute work is already present: '+processes)
        name = 'kvmem-source-forget-'+uuid.uuid4().hex[:12]
        argv = ['docker', 'run', '-d', '--name', name, '--gpus', 'all',
            '-v', str(args.root)+':/work:ro', '-v', str(args.model.parent)+':/models:ro',
            '-v', '/usr/local/cuda-13.2:/usr/local/cuda-13.2:ro', '-v', '/usr/lib/wsl/lib:/usr/lib/wsl/lib:ro',
            '-e', 'LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/lib:/usr/local/cuda-13.2/lib64',
            '-e', 'NINFER_SOURCE_FORGET_ARTIFACT=/models/'+args.model.name]
        if config.endswith('mtp4'): argv += ['-e', 'NINFER_SOURCE_FORGET_MTP=1']
        if config.startswith('redundancy'): argv += ['-e', 'NINFER_SOURCE_FORGET_REDUNDANCY=1']
        argv += [args.image, '/work/'+relative]
        case = dict(config=config, argv=argv, passed=False, terminal=False)
        report['cases'].append(case); save(); created = False
        try:
            case['id'] = command(argv).strip(); created = True
            case['gpu'] = command([gpu, '--query-gpu=uuid,name,driver_version,memory.total', '--format=csv,noheader']).strip()
            deadline = time.monotonic()+900
            while True:
                state = json.loads(command(['docker', 'inspect', '--format', '{{json .State}}', name]))
                if not state['Running']: break
                if time.monotonic() >= deadline: raise TimeoutError(config+' exceeded fixed 15-minute timeout')
                time.sleep(1)
            case['terminal_state'] = state; case['terminal'] = True
            logs = subprocess.run(['docker', 'logs', name], capture_output=True, text=True)
            (args.out/(config+'.stdout.log')).write_text(logs.stdout)
            (args.out/(config+'.stderr.log')).write_text(logs.stderr)
            if state['ExitCode'] != 0 or state['OOMKilled']:
                raise RuntimeError(config+' native test failed: '+str(state['ExitCode'])+' '+logs.stderr[-2000:])
            records = [json.loads(line) for line in logs.stdout.splitlines() if line.startswith('{')]
            case['result'] = next(r for r in reversed(records) if r.get('status') == 'passed')
            case['forget_receipts'] = re.findall(r'KVMEM_SOURCE_FORGET root=(\d+) records=(\d+) native_domains=(\d+)', logs.stderr)
            if len(case['forget_receipts']) != 2 or case['forget_receipts'][0][1:] != ('3', '3'):
                raise RuntimeError('native family alias revocation receipt is missing')
            if len(re.findall('KVMEM_SOURCE_REBUILD_BEGIN ', logs.stderr)) != 2:
                raise RuntimeError('revoked preparation entered native rebuild')
            case['passed'] = True
        except Exception as error:
            case['error'] = repr(error)
            raise
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
    report['completed'] = True; save()


if __name__ == '__main__':
    main()
