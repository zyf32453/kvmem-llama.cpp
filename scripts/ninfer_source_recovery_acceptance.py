#!/usr/bin/env python3
"""Run the public Engine source-rebuild qualification on an isolated built tree."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
import uuid


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            value.update(block)
    return value.hexdigest()


def command(argv, **kwargs):
    return subprocess.run(argv, capture_output=True, text=True, check=True, **kwargs).stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--image', default='kvmem-lifecycle-builder:20261031-d311d1b')
    parser.add_argument('--configs', nargs='+', choices=['lru-none', 'lru-mtp4', 'redundancy-none', 'redundancy-mtp4'],
                        default=['lru-none', 'lru-mtp4', 'redundancy-none', 'redundancy-mtp4'])
    args = parser.parse_args()
    args.root = args.root.resolve(); args.model = args.model.resolve(); args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    phase = json.loads((args.root/'native-logs/phase.json').read_text())
    if phase['phase'] != 'completed': raise RuntimeError('native build/checks are not complete')
    relative = 'build-native/tests/ninfer_engine_source_recovery_real_test'
    expected = json.loads((args.root/'native-logs/binaries.json').read_text())[relative]
    if digest(args.root/relative) != expected: raise RuntimeError('qualification binary changed')
    if args.model.stat().st_size != 15017456896 or digest(args.model) != '29e54467ca0ddae7e9aac04439f72ad20dbf6dc8fb5e84833b41ec7de9c46750':
        raise RuntimeError('model differs from frozen support scope')
    gpu = '/usr/lib/wsl/lib/nvidia-smi'
    report = dict(completed=False, root=str(args.root), model=str(args.model),
        binary_sha256=expected, script_sha256=digest(Path(__file__)),
        native_inputs=json.loads((args.root/'inputs.json').read_text()),
        image_id=command(['docker', 'image', 'inspect', '--format', '{{.Id}}', args.image]).strip(),
        scope='canonical raw-token new-domain rebuild; no persisted/sparse restore, performance or Agent claim', cases=[])

    def save():
        (args.out/'status.json').write_text(json.dumps(report, indent=2)+'\n')
    save()
    for config in args.configs:
        # Never stop or evict another user's compute work to make this gate pass.
        processes = command([gpu, '--query-compute-apps=pid,process_name', '--format=csv,noheader']).strip()
        if processes: raise RuntimeError('GPU compute work is already present: '+processes)
        name = 'kvmem-source-rebuild-'+uuid.uuid4().hex[:12]
        argv = ['docker', 'run', '-d', '--name', name, '--gpus', 'all',
            '-v', str(args.root)+':/work:ro', '-v', str(args.model.parent)+':/models:ro',
            '-v', '/usr/local/cuda-13.2:/usr/local/cuda-13.2:ro',
            '-v', '/usr/lib/wsl/lib:/usr/lib/wsl/lib:ro',
            '-e', 'LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/lib:/usr/local/cuda-13.2/lib64',
            '-e', 'NINFER_SOURCE_RECOVERY_ARTIFACT=/models/'+args.model.name,
            '-e', 'KVMEM_GC_AUDIT=1']
        if config.endswith('mtp4'): argv += ['-e', 'NINFER_SOURCE_RECOVERY_MTP=1']
        if config.startswith('redundancy'): argv += ['-e', 'NINFER_SOURCE_RECOVERY_REDUNDANCY=1']
        argv += [args.image, '/work/'+relative]
        result = dict(config=config, argv=argv, passed=False, terminal=False)
        report['cases'].append(result); save(); created = False
        try:
            result['id'] = command(argv).strip(); created = True
            result['gpu'] = command([gpu, '--query-gpu=uuid,name,driver_version,memory.total', '--format=csv,noheader']).strip()
            save(); deadline = time.monotonic()+1800
            while True:
                state = json.loads(command(['docker', 'inspect', '--format', '{{json .State}}', name]))
                if not state['Running']: break
                if time.monotonic() >= deadline: raise TimeoutError(config+' exceeded fixed 30-minute gate timeout')
                time.sleep(1)
            result['terminal_state'] = state; result['terminal'] = True
            logs = subprocess.run(['docker', 'logs', name], capture_output=True, text=True)
            (args.out/(config+'.stdout.log')).write_text(logs.stdout)
            (args.out/(config+'.stderr.log')).write_text(logs.stderr)
            if state['ExitCode'] != 0 or state['OOMKilled']:
                raise RuntimeError(config+' native test failed: '+str(state['ExitCode'])+' '+logs.stderr[-2000:])
            records = [json.loads(line) for line in logs.stdout.splitlines() if line.startswith('{')]
            result['result'] = next(record for record in reversed(records) if record.get('status') == 'passed')
            gc = re.findall(r'KVMEM_NATIVE_GC_SUMMARY .*?committed=(\d+) reclaimed=(\d+)', logs.stderr)
            result['gc_summaries'] = [dict(committed=int(a), reclaimed=int(b)) for a, b in gc]
            if not any(int(b) > 0 for _, b in gc): raise RuntimeError('no physical native CPU reclamation proved')
            # At least one actually freed old page must be INSIDE the prefix
            # subsequently rebuilt. Unrelated system-wide GC is insufficient.
            before_rebuild = logs.stderr.split('KVMEM_SOURCE_REBUILD_BEGIN', 1)[0]
            released = re.findall(r'KVMEM_GC_AUDIT event=reclaim [^\n]*original_begin=(\d+) valid_tokens=(\d+) [^\n]*host_released=1', before_rebuild)
            result['reclaimed_prefix_spans'] = [dict(begin=int(a), count=int(b)) for a, b in released
                if int(b) > 0 and int(a)+int(b) <= result['result']['rebuild_prefix']]
            if not result['reclaimed_prefix_spans']:
                raise RuntimeError('no physically freed old page inside the rebuilt prefix was proved')
            begins = re.findall(r'KVMEM_SOURCE_REBUILD_BEGIN parent=(\d+) source=(\d+) domain=(\d+) prefix=(\d+)', logs.stderr)
            completes = re.findall(r'KVMEM_SOURCE_REBUILD_COMPLETE source=(\d+) domain=(\d+) accepted=(\d+) computed=(\d+)', logs.stderr)
            result['rebuild_begins'] = begins; result['rebuild_completes'] = completes
            if len(begins) != 4 or len(completes) != 3 or len({b[2] for b in begins}) != 4:
                raise RuntimeError('unexpected partial/successful rebuild publication or reused identity')
            result['passed'] = True
        except Exception as error:
            result['error'] = repr(error)
            raise
        finally:
            if created:
                # Only this explicitly owned test container is stopped/removed.
                subprocess.run(['docker', 'stop', '--time', '20', name], capture_output=True, text=True)
                final = subprocess.run(['docker', 'inspect', '--format', '{{json .State}}', name], capture_output=True, text=True)
                if final.returncode == 0: result['final_container_state'] = json.loads(final.stdout)
                logs = subprocess.run(['docker', 'logs', name], capture_output=True, text=True)
                (args.out/(config+'.stdout.log')).write_text(logs.stdout)
                (args.out/(config+'.stderr.log')).write_text(logs.stderr)
                subprocess.run(['docker', 'rm', name], capture_output=True, text=True, check=True)
            save()
        print(json.dumps(dict(config=config, passed=result['passed'])), flush=True)
    report['completed'] = True; save()


if __name__ == '__main__':
    main()
