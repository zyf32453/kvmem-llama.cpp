import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from ninfer_gc_agent_bridge import AutomaticArtifactGc
from ninfer_gc_step import attach_evidence
from ninfer_gc_terminal import DockerReader, WorkspaceReader, candidates, literal_write, verified_step
from test_ninfer_gc_agent_bridge import Client


def terminal_pair(body, path='/app/result.txt', alternate=False, ascii_json=False):
    header = f"cat <<'FILE' > {path}" if alternate else f"cat > {path} <<'FILE'"
    command = header + '\n' + body + 'FILE\n'
    return [dict(role='assistant', content=json.dumps(dict(analysis='Do not delete this analysis',
        commands=[dict(keystrokes=command, duration=1)]), ensure_ascii=ascii_json)),
        dict(role='user', content='New Terminal Output:\n\nroot@owned:/app# ' + header + '\nroot@owned:/app# ')]


class TerminalEvidenceTest(unittest.TestCase):
    def test_literal_body_and_actual_native_token_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            body = 'exact "quoted" \\ content 中文 😀\n' * 30
            (root / 'result.txt').write_bytes(body.encode())
            for alternate in (False, True):
                for ascii_json in (False, True):
                    messages = terminal_pair(body, alternate=alternate, ascii_json=ascii_json)
                    found = candidates(messages)
                    self.assertEqual(len(found), 1)
                    identity, candidate = next(iter(found.items()))
                    step = verified_step(messages, identity, candidate, WorkspaceReader(root))
                    self.assertNotIn('Do not delete this analysis', step['source_text'])
                    original = copy.deepcopy(messages)
                    request = dict(prompt_cache_key='terminal', messages=messages)
                    value = attach_evidence('', request, [step], post=Client('').json)
                    proof = value['kvmem_gc']['steps'][0]
                    self.assertGreater(proof['source_end'], proof['source_begin'])
                    self.assertLessEqual(proof['source_end'], proof['result_begin'])
                    self.assertEqual(messages, original)

    def test_failed_missing_changed_or_ambiguous_writes_do_not_authorize(self):
        for bad in ("cat > /app/a <<EOF\nbody\nEOF\n", "cat >> /app/a <<'EOF'\nbody\nEOF\n",
                    "cat > /app/a <<'EOF'\nbody\nEOF\nrm /app/a\n",
                    "cat > $TARGET <<'EOF'\nbody\nEOF\n"):
            with self.assertRaises(ValueError):
                literal_write(bad)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            body = 'unique retained data\n' * 40
            messages = terminal_pair(body)
            identity, candidate = next(iter(candidates(messages).items()))
            reader = WorkspaceReader(root)
            with self.assertRaises(OSError):
                verified_step(messages, identity, candidate, reader)
            (root / 'result.txt').write_text('different', encoding='utf-8')
            with self.assertRaises(ValueError):
                verified_step(messages, identity, candidate, reader)
            for name in ('/outside/file', '../outside'):
                with self.assertRaises(ValueError):
                    reader(name)
            relative = dict(candidate, path='result.txt')
            with self.assertRaises(ValueError):
                verified_step(messages, identity, relative, reader)
            messages[-1]['content'] = 'Ordinary human message is not a tool-result witness'
            self.assertEqual(candidates(messages), {})

    def test_repeated_body_uses_proven_message_not_global_text_search(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            body = 'repeated encoded file body\n' * 40
            (root / 'result.txt').write_bytes(body.encode())
            messages = terminal_pair(body)
            encoded_body = json.dumps(body)[1:-1]
            messages.insert(0, dict(role='user', content='Quoted example: ' + encoded_body))
            identity, candidate = next(iter(candidates(messages).items()))
            step = verified_step(messages, identity, candidate, WorkspaceReader(root))
            value = attach_evidence('', dict(prompt_cache_key='same', messages=messages), [step], post=Client('').json)
            self.assertGreater(value['kvmem_gc']['steps'][0]['source_begin'], len(encoded_body))

    def test_automatic_adapter_and_bounded_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            body = 'automatic terminal verification\n' * 40
            (root / 'result.txt').write_bytes(body.encode())
            messages = terminal_pair(body)
            request = dict(prompt_cache_key='terminal', messages=messages, temperature=.6, max_tokens=16384)
            reader = WorkspaceReader(root)
            adapter = AutomaticArtifactGc(root, '', client_factory=Client, terminal_reader=reader)
            value, audit = adapter(request)
            self.assertEqual(audit['attached'], 1)
            self.assertEqual({k: v for k, v in value.items() if k != 'kvmem_gc'}, request)
            self.assertFalse(value['kvmem_gc']['steps'][0]['cold'])
            self.assertEqual(adapter(request)[0]['kvmem_gc'], value['kvmem_gc'])
            (root / 'result.txt').write_bytes(b'changed')
            unchanged, audit = adapter(request)
            self.assertNotIn('kvmem_gc', unchanged)
            self.assertEqual(audit['attached'], 0)

    def test_container_reader_uses_fixed_owned_target_and_literal_argv(self):
        reader = DockerReader('trial-owned-123')
        name = '/app/a; echo SHOULD_NOT_EXECUTE'
        with patch('ninfer_gc_terminal.subprocess.run', return_value=subprocess.CompletedProcess([], 0, b'{"body":"b2s="}')) as run:
            self.assertEqual(reader(name), b'ok')
        argv = run.call_args.args[0]
        self.assertEqual(argv[:5], ['docker', 'exec', 'trial-owned-123', 'python3', '-I'])
        self.assertEqual(argv[-1], name)
        self.assertNotIn('shell', run.call_args.kwargs)
        with self.assertRaises(ValueError):
            DockerReader('container;evil')


if __name__ == '__main__':
    unittest.main()
