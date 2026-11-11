import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

from agent_phase_capture import handler
from ninfer_gc_agent_bridge import AutomaticArtifactGc
from ninfer_gc_terminal import WorkspaceReader


PROFILE = dict(policy="statistical", min_observations=1, min_cold_observations=0,
               max_importance=1, block_tokens=64, ram_only=True)


def write_pair(root, call_id="write-1", name="result.txt", content=None):
    content = content or ("verified artifact data " * 32)
    file = root / name
    file.write_bytes(content.encode())
    result = f"<path>{file}</path>\n<type>file</type>\n<content>\nCreated file\n</content>"
    return [dict(role="assistant", tool_calls=[dict(id=call_id, function=dict(name="write",
        arguments=json.dumps(dict(file_path=str(file), content=content))))]),
        dict(role="tool", tool_call_id=call_id, content=result)]


def render(request):
    text = "initial frame " * 100
    for message in request["messages"]:
        for call in message.get("tool_calls", []):
            text += json.loads(call["function"]["arguments"])["content"]
        text += "padding " * 16 + message.get("content", "") + " frame " * 16
    return text + "next Query " * 16


class Client:
    profile = PROFILE
    def __init__(self, base, authorization=None):
        self.authorization = authorization

    def json(self, path, value=None):
        if path.startswith("/props"):
            return dict(kvmem_gc=self.profile)
        if path == "/apply-template":
            text=render(value)
            result=dict(prompt=text)
            if value.get('return_tokens'):
                result['tokens']=[dict(id=ord(c),piece=c) for c in text]
            return result
        if path == "/tokenize":
            return dict(tokens=[dict(id=ord(c), piece=c) for c in value["content"]])
        raise AssertionError(path)


class AutomaticGcTest(unittest.TestCase):
    def test_automatic_discovery_retries_and_profile_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = dict(model="fixture", prompt_cache_key="agent", messages=write_pair(root),
                           temperature=.6, seed=1000, max_tokens=16384)
            original = copy.deepcopy(request)
            adapter = AutomaticArtifactGc(root, "http://test", client_factory=Client)
            value, audit = adapter(request, {})
            self.assertEqual(audit["status"], "attached")
            self.assertEqual(audit["attached"], 1)
            self.assertEqual(request, original)
            self.assertEqual({k: v for k, v in value.items() if k != "kvmem_gc"}, original)
            self.assertEqual(value["kvmem_gc"]["steps"][0]["cold"], False)
            # Identical retries have identical coordinates and one observation,
            # not another invented cold sample.
            retried, _ = adapter(request, {})
            self.assertEqual(retried["kvmem_gc"], value["kvmem_gc"])
            class DefaultStatistical(Client):
                profile = dict(PROFILE, min_observations=3, min_cold_observations=3, max_importance=.1)
            guarded = AutomaticArtifactGc(root, "http://test", client_factory=DefaultStatistical)
            unchanged, audit = guarded(request, {})
            self.assertEqual(unchanged, original)
            self.assertEqual(audit["status"], "incompatible_native_profile")
            (root / "result.txt").write_bytes(b"different")
            unchanged, audit = adapter(request, {})
            self.assertEqual(unchanged, original)
            self.assertEqual(audit["attached"], 0)

    def test_independent_candidates_and_bounded_sessions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            messages = write_pair(root, "a", "a.txt", "first content " * 60) + write_pair(root, "b", "b.txt", "second content " * 60)
            messages[1]["content"] = "write failed"
            adapter = AutomaticArtifactGc(root, "http://test", sessions=1, pending_per_session=2, client_factory=Client)
            request = dict(prompt_cache_key="one", messages=messages)
            value, audit = adapter(request, {})
            self.assertEqual(audit["attached"], 1)
            self.assertEqual(len(value["kvmem_gc"]["steps"]), 1)
            _, audit = adapter(dict(request, prompt_cache_key="two"), {})
            self.assertEqual(audit["evicted_sessions"], 1)
            self.assertEqual(len(adapter.states), 1)
            manual = dict(request, kvmem_gc=dict(existing=True))
            self.assertEqual(adapter(manual, {})[0], manual)

    def test_owner_binds_missing_agent_session_without_overriding_client(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = dict(messages=write_pair(root), temperature=.6, max_tokens=16384)
            original = copy.deepcopy(request)
            adapter = AutomaticArtifactGc(root, '', client_factory=Client, session_key='owned-harbor-trial')
            value, audit = adapter(request)
            self.assertEqual(request, original)
            self.assertEqual(value['prompt_cache_key'], 'owned-harbor-trial')
            self.assertEqual(audit['session_binding'], 'owned_trial')
            self.assertEqual(audit['attached'], 1)
            retry, _ = adapter(request)
            self.assertEqual(retry['kvmem_gc'], value['kvmem_gc'])
            own = dict(request, prompt_cache_key='client-session')
            self.assertEqual(adapter(own)[0]['prompt_cache_key'], 'client-session')
            none = dict(messages=[dict(role='user', content='No completed artifacts yet')])
            bound, audit = adapter(none)
            self.assertEqual(bound['prompt_cache_key'], 'owned-harbor-trial')
            self.assertNotIn('kvmem_gc', bound)
            manual = dict(request, kvmem_gc={'manual':True})
            self.assertEqual(adapter(manual)[0], manual)

    def test_real_http_path_without_logprob_or_sampling_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observed = []
            get_paths = []
            class Native(BaseHTTPRequestHandler):
                def log_message(self, *args):
                    pass
                def do_GET(self):
                    get_paths.append(self.path)
                    self.respond(dict(kvmem_gc=PROFILE))
                def respond(self, data, status=200):
                    self.send_response(status); self.send_header("Content-Type", "application/json"); self.end_headers()
                    self.wfile.write(json.dumps(data).encode())
                def do_POST(self):
                    raw = self.rfile.read(int(self.headers['Content-Length']))
                    try:
                        value = json.loads(raw)
                        if not isinstance(value, dict):
                            raise ValueError('Expected object')
                    except ValueError:
                        self.respond(dict(error='native request validation'), 400)
                        return
                    if self.path == "/apply-template":
                        self.respond(Client('').json(self.path, value))
                    elif self.path == "/tokenize":
                        self.respond(dict(tokens=[dict(id=ord(c), piece=c) for c in value["content"]]))
                    else:
                        observed.append(value)
                        self.respond(dict(choices=[dict(message=dict(content="OK"))], usage=dict(
                            completion_tokens=1, completion_tokens_details=dict(reasoning_tokens=0))))
            model = ThreadingHTTPServer(("127.0.0.1", 0), Native)
            base = f"http://127.0.0.1:{model.server_port}"
            adapter = AutomaticArtifactGc(root, base, terminal_reader=WorkspaceReader(root))
            def prepare(request, headers):
                if request.get('model') == 'prepare-fails':
                    raise OSError('File verifier unavailable')
                return adapter(request, headers)
            bridge = ThreadingHTTPServer(("127.0.0.1", 0), handler(base, root / "audit", prepare, False))
            for server in (model, bridge):
                threading.Thread(target=server.serve_forever, daemon=True).start()
            try:
                request = dict(model="fixture", prompt_cache_key="task", messages=write_pair(root),
                               temperature=.6, max_tokens=16384, seed=1000)
                req = urllib.request.Request(f"http://127.0.0.1:{bridge.server_port}/v1/chat/completions",
                    data=json.dumps(request).encode(), headers={"Content-Type": "application/json"})
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                with opener.open(req, timeout=5) as response:
                    self.assertEqual(json.load(response)["choices"][0]["message"]["content"], "OK")
                self.assertEqual(len(observed), 1)
                self.assertIn("kvmem_gc", observed[0])
                self.assertNotIn("logprobs", observed[0])
                self.assertEqual({k: v for k, v in observed[0].items() if k != "kvmem_gc"}, request)
                audit = next((root / "audit").glob("*/gc-prepare.json"))
                self.assertEqual(json.loads(audit.read_text())["status"], "attached")
                body = 'terminal file body\n' * 50
                (root / 'terminal.txt').write_bytes(body.encode())
                terminal = dict(request, prompt_cache_key='terminal', messages=[
                    dict(role='assistant', content=json.dumps(dict(analysis='preserve analysis', commands=[
                        dict(keystrokes="cat > /app/terminal.txt <<'EOF'\n" + body + 'EOF\n', duration=1)]))),
                    dict(role='user', content='New Terminal Output:\n\nroot@owned:/app# cat > /app/terminal.txt\nroot@owned:/app#')])
                for value in (terminal, dict(request, model='prepare-fails')):
                    req = urllib.request.Request(f"http://127.0.0.1:{bridge.server_port}/v1/chat/completions",
                        data=json.dumps(value).encode(), headers={"Content-Type": "application/json"})
                    with opener.open(req, timeout=5) as response:
                        self.assertEqual(response.status, 200)
                self.assertIn('kvmem_gc', observed[1])
                self.assertEqual(observed[2], dict(request, model='prepare-fails'))
                audits = [json.loads(p.read_text()) for p in (root / 'audit').glob('*/gc-prepare.json')]
                self.assertEqual(sum(a['status'] == 'attached' for a in audits), 2)
                self.assertEqual(sum(a['status'] == 'preparation_error' for a in audits), 1)
                with opener.open(f'http://127.0.0.1:{bridge.server_port}/props?model=fixture', timeout=5) as response:
                    self.assertEqual(response.status, 200)
                self.assertEqual(get_paths[-1], '/props?model=fixture')
                for invalid in (b'{not-json', b'[]'):
                    req = urllib.request.Request(f"http://127.0.0.1:{bridge.server_port}/v1/chat/completions",
                        data=invalid, headers={'Content-Type':'application/json'})
                    with self.assertRaises(urllib.error.HTTPError) as failed:
                        opener.open(req, timeout=5)
                    self.assertEqual(failed.exception.code, 400)
                bad = [p for p in (root/'audit').glob('*/request.forwarded.json') if p.read_bytes() in (b'{not-json', b'[]')]
                self.assertEqual(len(bad), 2)
            finally:
                for server in (bridge, model):
                    server.shutdown(); server.server_close()


if __name__ == "__main__":
    unittest.main()
