#!/usr/bin/env python3
"""Automatically attach verified-artifact GC evidence on real Agent requests.

The Agent keeps its existing OpenAI request protocol; point api_base at this
owned loopback bridge. Successful DSH write/result pairs are discovered from
history and verified against the actual shared workspace on each observation.
Generic thinking/summary coverage is not inferred or given fake cold scores.
"""
import argparse
from collections import OrderedDict
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import threading
import urllib.parse
import urllib.request
import uuid

from agent_phase_capture import handler
from ninfer_gc_step import attach_evidence, verified_write_step
from ninfer_gc_terminal import DockerReader, WorkspaceReader, candidates as terminal_candidates, verified_step


class NativeClient:
    def __init__(self, base, authorization=None, timeout=120):
        self.base = base.rstrip("/")
        self.authorization = authorization
        self.timeout = timeout
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def json(self, path, value=None):
        headers = {"Content-Type": "application/json"}
        if self.authorization:
            headers["Authorization"] = self.authorization
        req = urllib.request.Request(self.base + path, headers=headers,
            data=json.dumps(value, ensure_ascii=False).encode("utf-8") if value is not None else None)
        with self.opener.open(req, timeout=self.timeout) as response:
            return json.load(response)


class AutomaticArtifactGc:
    def __init__(self, workspace, upstream, candidates_per_request=4, sessions=16, pending_per_session=64,
                 client_factory=NativeClient, terminal_reader=None, session_key=None):
        if not 1 <= candidates_per_request <= 64 or not 1 <= sessions <= 16 or not 1 <= pending_per_session <= 64:
            raise ValueError("Invalid bounded Agent GC budget")
        self.workspace = Path(workspace).resolve(strict=True) if workspace is not None else None
        self.terminal_reader = terminal_reader
        if session_key is not None and (not isinstance(session_key, str) or not session_key or len(session_key.encode())>256):
            raise ValueError('Invalid owned trial session key')
        self.session_key = session_key
        self.upstream = upstream
        self.budget = candidates_per_request
        self.sessions = sessions
        self.pending = pending_per_session
        self.client_factory = client_factory
        self.states = OrderedDict()
        self.serial = 0
        self.lock = threading.Lock()

    def __call__(self, request, headers=None):
        # A model request owns its preparation. Scoring/IO never runs on a native
        # GC worker or inside the next decode boundary.
        with self.lock:
            return self.prepare(request, headers or {})

    def prepare(self, request, headers):
        audit = dict(status="no_candidates", candidates=0, attached=0, skipped=[])
        if "kvmem_gc" in request:
            audit["status"] = "client_evidence_preserved"
            return request, audit
        # Some Agent clients do not send this native continuation field. The
        # controller may bind this bridge to ONE owned trial. Do not infer a
        # shared session from message text or silently replace a client's key.
        if self.session_key is not None and 'prompt_cache_key' not in request:
            request = dict(request, prompt_cache_key=self.session_key)
            audit['session_binding'] = 'owned_trial'
        session = request.get("prompt_cache_key")
        if not isinstance(session, str) or not session or len(session.encode()) > 256:
            audit["status"] = "missing_named_session"
            return request, audit
        messages = request.get("messages")
        if not isinstance(messages, list):
            audit["status"] = "invalid_messages"
            return request, audit
        if any(not isinstance(m, dict) for m in messages):
            audit["status"] = "invalid_messages"
            return request, audit
        calls = [c.get("id") for m in messages if m.get("role") == "assistant"
                 for c in (m.get("tool_calls") or []) if isinstance(c, dict) and isinstance(c.get('function'), dict)
                 and c['function'].get("name") == "write"
                 and isinstance(c.get("id"), str) and 0 < len(c["id"]) <= 256]
        results = {m.get("tool_call_id") for m in messages if m.get("role") == "tool" and isinstance(m.get('tool_call_id'), str)}
        descriptors = {c: None for c in calls if c in results and self.workspace is not None}
        if self.terminal_reader is not None:
            descriptors.update(terminal_candidates(messages))
        all_available = list(descriptors)
        available = all_available[-self.pending:]
        audit['older_candidates_deferred'] = max(0, len(all_available) - len(available))
        if not available:
            return request, audit
        authorization = next((v for k, v in headers.items() if k.lower() == "authorization"), None)
        client = self.client_factory(self.upstream, authorization)
        try:
            suffix = "?" + urllib.parse.urlencode(dict(model=request["model"])) if request.get("model") else ""
            profile = client.json("/props" + suffix).get("kvmem_gc", {})
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            audit.update(status="profile_unavailable", reason=type(error).__name__)
            return request, audit
        expected = dict(policy="statistical", min_observations=1, min_cold_observations=0,
                        max_importance=1, block_tokens=64, ram_only=True)
        if any(profile.get(k) != v for k, v in expected.items()):
            audit.update(status="incompatible_native_profile", expected=expected, observed=profile)
            return request, audit
        state = self.states.pop(session, OrderedDict())
        previous = set(state)
        for key in list(state):
            if key not in available:
                del state[key]
        for key in available:
            state.setdefault(key, 0)
        self.states[session] = state
        evicted = 0
        while len(self.states) > self.sessions:
            self.states.popitem(last=False); evicted += 1
        # New completions go first; remaining budget retries older candidates
        # after rollback or a previously unsealed witness, in bounded rotation.
        order = sorted(available, key=lambda k: (k in previous, state[k], -available.index(k)))[:self.budget]
        self.serial += 1
        steps = []
        for call in order:
            state[call] = self.serial
            try:
                candidate = descriptors[call]
                steps.append(verified_write_step(messages, call, self.workspace) if candidate is None else
                             verified_step(messages, call, candidate, self.terminal_reader))
            except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as error:
                audit["skipped"].append(dict(call_id=call, reason=type(error).__name__))
        audit.update(candidates=len(order), pending=len(state), evicted_sessions=evicted)
        if not steps:
            return request, audit
        # Share the one full rendering/tokenization across individual candidate
        # checks. One malformed or unsealed witness must not suppress other
        # independent successful completions in this request.
        cache = {}
        def cached_post(path, value):
            key = (path, json.dumps(value, sort_keys=True, ensure_ascii=False))
            if key not in cache:
                cache[key] = client.json(path, value)
            return cache[key]
        batch = None
        for step in steps:
            try:
                proof = attach_evidence(self.upstream, request, [step], post=cached_post)["kvmem_gc"]
                if batch is None:
                    batch = dict(prefix_tokens=proof["prefix_tokens"], prefix_sha256=proof["prefix_sha256"], steps=[])
                batch["steps"].extend(proof["steps"])
            except (OSError, ValueError, TypeError, KeyError) as error:
                audit["skipped"].append(dict(step_id=step["id"], reason=str(error)))
        if batch is None:
            audit["status"] = "evidence_unavailable"
            return request, audit
        result = dict(request, kvmem_gc=batch)
        audit.update(status="attached", attached=len(batch["steps"]), step_ids=[s["id"] for s in batch["steps"]],
                     observation="file_bytes_equal_successful_write", cold_observations_invented=False)
        return result, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    location = parser.add_mutually_exclusive_group(required=True)
    location.add_argument("--workspace", type=Path, help="Actual workspace shared with Agent write tools")
    location.add_argument("--container", help="Fixed owned Terminus trial container ID/name; read-only file verification")
    parser.add_argument('--terminal-root', default='/app', help='Allowed absolute artifact root inside the owned trial')
    parser.add_argument('--terminus', action='store_true', help='Enable Terminus-2 literal file-write evidence for the shared workspace')
    parser.add_argument('--session-key', help='One owned trial session for clients omitting prompt_cache_key; defaults to a fresh key per bridge process')
    parser.add_argument("--upstream", default="http://127.0.0.1:18081")
    parser.add_argument("--port", type=int, default=18083)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--candidates-per-request", type=int, default=4)
    args = parser.parse_args()
    url = urllib.parse.urlsplit(args.upstream)
    if url.scheme != "http" or url.hostname not in ("127.0.0.1", "localhost", "::1"):
        parser.error("Use the owned loopback native model service")
    reader = DockerReader(args.container, args.terminal_root) if args.container else (
        WorkspaceReader(args.workspace, args.terminal_root) if args.terminus else None)
    trial_key = args.session_key or 'proactive-trial-' + uuid.uuid4().hex
    gc = AutomaticArtifactGc(args.workspace, args.upstream, args.candidates_per_request,
                             terminal_reader=reader, session_key=trial_key)
    args.out.mkdir(parents=True, exist_ok=True)
    print("Agent GC bridge ready; verified artifact profile 1/0/1 is checked on each candidate request", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), handler(args.upstream, args.out,
        before_request=gc, collect_logprobs=False)).serve_forever()


if __name__ == "__main__":
    main()
