#!/usr/bin/env python3
"""Loopback recording proxy for a single Terminal-Bench phase-profile run.

Capture accepted content-token bytes via native logprobs, plus API reasoning
counts. Does not change prompts, sampling, output limits or GC selection.
This is an instrumentation profile, not a decode-performance benchmark.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid


def object_spans(text):
    """Locate top-level JSON values without independently retokenizing fields."""
    decoder = json.JSONDecoder()
    start = text.find("{")
    if start < 0:
        return []
    try:
        obj, end = decoder.raw_decode(text, start)
        if not isinstance(obj, dict):
            return []
        spans = [(start, end, "protocol")]
        cursor = start + 1
        seen = set()
        while cursor < end:
            while text[cursor].isspace() or text[cursor] == ",":
                cursor += 1
            if text[cursor] == "}":
                break
            key, cursor = decoder.raw_decode(text, cursor)
            if not isinstance(key, str) or key in seen:
                return []
            seen.add(key)
            while text[cursor].isspace():
                cursor += 1
            if text[cursor] != ":":
                return []
            cursor += 1
            while text[cursor].isspace():
                cursor += 1
            begin = cursor
            value, cursor = decoder.raw_decode(text, cursor)
            kind = key if key in ("analysis", "plan") else (
                "command_request" if key == "commands" and isinstance(value, list) and value else
                "completion_declaration" if key == "task_complete" and value is True else None)
            if kind:
                spans.append((begin, cursor, kind))
        offsets = [0]
        for char in text:
            offsets.append(offsets[-1] + len(char.encode("utf-8")))
        return [(offsets[a], offsets[b], k) for a, b, k in spans]
    except (ValueError, IndexError, TypeError):
        return []


def profile_response(response, classify_fields=True):
    usage = response.get("usage") or {}
    total = usage.get("completion_tokens")
    thinking = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    if not isinstance(total, int) or not isinstance(thinking, int) or not 0 <= thinking <= total:
        return dict(status="usage_unavailable", warning="No complete native output/reasoning counts")
    coarse = dict(thinking=thinking, non_thinking=total - thinking)
    choice = (response.get("choices") or [{}])[0]
    content = (choice.get("message") or {}).get("content")
    probabilities = (choice.get("logprobs") or {}).get("content")
    result = dict(status="coarse_only", completion_tokens=total, prompt_tokens=usage.get("prompt_tokens"),
                  coarse_counts=coarse, timings=response.get("timings"),
                  caveat="Native API reasoning-token definition; do not infer semantic summary or phase latency.")
    if not isinstance(content, str) or not isinstance(probabilities, list):
        return result
    chunks = []
    for row in probabilities:
        value = row.get("bytes")
        if not isinstance(value, list):
            return result
        try:
            chunks.append(bytes(value))
        except (TypeError, ValueError):
            return result
    raw = content.encode("utf-8")
    if b"".join(chunks) != raw:
        result["warning"] = "Content logprob bytes do not exactly reproduce exposed content"
        return result
    classes = ["assistant_text", "protocol", "analysis", "plan", "command_request", "completion_declaration"]
    paint = bytearray(len(raw))
    # Invalid Terminus replies are text, not executed command requests. The
    # live recorder preserves raw measurements; offline analysis supplies the
    # installed harness parser's verdict without changing the running Agent.
    for begin, end, kind in object_spans(content) if classify_fields else []:
        paint[begin:end] = bytes([classes.index(kind)]) * (end - begin)
    counts, cursor = Counter(), 0
    for chunk in chunks:
        labels = set(paint[cursor:cursor + len(chunk)])
        kind = "protocol" if not chunk else classes[next(iter(labels))] if len(labels) == 1 else "boundary"
        counts[kind] += 1
        cursor += len(chunk)
    residual = total - thinking - len(chunks)
    if residual < 0:
        result["warning"] = "Content and reasoning counters do not form a disjoint output partition"
        return result
    counts["thinking"] += thinking
    if residual:
        counts["unexposed_control_tokens"] += residual
    result.update(status="accepted_content_bytes_and_native_reasoning", counts=dict(counts),
                  fractions={k: v / total for k, v in counts.items()} if total else {})
    return result


def aggregate_stream(events):
    content, reasoning, logprobs = [], [], []
    result = dict(choices=[dict(message={}, logprobs={})])
    for event in events:
        for key in ("usage", "timings", "kvmem"):
            if event.get(key) is not None:
                result[key] = event[key]
        if event.get("error"):
            result["error"] = event["error"]
        for choice in event.get("choices", []):
            delta = choice.get("delta") or {}
            content.append(delta.get("content") or "")
            reasoning.append(delta.get("reasoning_content") or "")
            logprobs.extend((choice.get("logprobs") or {}).get("content") or [])
    result["choices"][0]["message"] = dict(content="".join(content), reasoning_content="".join(reasoning))
    result["choices"][0]["logprobs"] = dict(content=logprobs)
    return result


def feedback_spans(text):
    """Frozen Terminus feedback framing, matching the supplied label method.

    Apply only to later user-role feedback of the owned Terminus harness.
    Inner ranges override their enclosing observation, without double counts.
    No keyword-based semantic classification is attempted.
    """
    confirmation = ('Are you sure you want to mark the task as complete? This will trigger '
        "your solution to be graded and you won't be able to make any further "
        'corrections. If so, include "task_complete": true in your JSON response again.')
    spans = []
    if text.startswith('Previous response had parsing errors:\n'):
        spans = [(0, len(text), 'parser_feedback')]
    else:
        start = 0
        markers = ('New Terminal Output:', 'Current terminal state:', 'Current Terminal Screen:')
        if text.startswith('Previous response had warnings:\n'):
            start = min((p for marker in markers if (p := text.find(marker)) >= 0), default=len(text))
            spans.append((0, start, 'parser_warning'))
        if text[start:].startswith(markers):
            end = len(text) - (len(confirmation) if text.endswith(confirmation) else 0)
            spans.append((start, end, 'terminal_observation'))
            status = text.rfind('\n[Executor status]\n', start, end)
            if status >= 0:
                spans.append((status + 1, end, 'execution_status'))
            if end < len(text):
                spans.append((end, len(text), 'completion_confirmation'))
    if not spans:
        spans = [(0, len(text), 'user_text')]
    offsets = [0]
    for char in text:
        offsets.append(offsets[-1] + len(char.encode('utf-8')))
    return [(offsets[a], offsets[b], kind) for a, b, kind in spans]


def profile_native_prompt(request, post, instruction="", expected_tokens=None, content_validity=None):
    """Classify one complete native rendered input, never sum repeated prefixes.

    Contrast rendering proves where a field is present in the current native
    template. Whole-input token bytes preserve BPE boundaries. The caller is
    the fixed autonomous Terminus-2 harness: later user messages are its
    feedback, not new human instructions. Unmapped text stays explicitly unknown.
    """
    import copy
    from ninfer_gc_step import pieces
    prepared = post("/apply-template", dict(request, return_tokens=True))
    rendered = prepared["prompt"]
    rows = prepared.get("tokens")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Native template endpoint must return actual prepared input tokens")
    chunks = pieces(rows)
    raw = rendered.encode("utf-8")
    if b"".join(chunks) != raw or expected_tokens is not None and len(rows) != expected_tokens:
        raise ValueError("Native rendered-input bytes/count do not match captured prompt usage")
    import hashlib
    spans, omitted, unmapped, transformed, parser_unmapped = [], [], [], [], []
    first_user = next((i for i, m in enumerate(request["messages"]) if m["role"] == "user"), -1)
    for index, message in enumerate(request["messages"]):
        for field in ("content", "reasoning_content"):
            original = message.get(field)
            if not isinstance(original, str) or not original:
                continue
            variant = copy.deepcopy(request)
            marker = "KVMEM_PHASE_" + uuid.uuid4().hex
            variant["messages"][index][field] = marker
            changed = post("/apply-template", variant)["prompt"]
            if marker not in changed and changed == rendered:
                omitted.append(dict(message=index, field=field))
                continue
            if changed.count(marker) != 1:
                unmapped.append(dict(message=index, field=field))
                continue
            before, after = changed.split(marker)
            # Templates may trim field-edge whitespace. Accept a transformation
            # only when it exactly reconstructs the whole native prompt.
            visible = next((s for s in dict.fromkeys((original, original.strip(), original.lstrip(), original.rstrip()))
                            if before + s + after == rendered), None)
            if visible is None:
                unmapped.append(dict(message=index, field=field))
                continue
            if visible != original:
                transformed.append(dict(message=index, field=field, transformation='template_edge_whitespace'))
            begin = len(before.encode("utf-8")); end = begin + len(visible.encode("utf-8"))
            role = message["role"]
            kind = "thinking" if field == "reasoning_content" else (
                "framework_rules" if role in ("system", "developer") else
                "assistant_text" if role == "assistant" else
                "tool_result" if role == "tool" else
                "initial_context" if index == first_user else "user_text")
            spans.append((begin, end, kind))
            if field == "content" and role == "assistant":
                digest = hashlib.sha256(original.encode()).hexdigest()
                if content_validity is not None and digest not in content_validity:
                    parser_unmapped.append(dict(message=index, field=field))
                valid = content_validity is None or content_validity.get(digest, False)
                if valid:
                    spans.extend((begin + a, begin + b, kind) for a, b, kind in object_spans(visible))
            if field == "content" and role == "user" and index != first_user:
                spans.extend((begin + a, begin + b, kind) for a, b, kind in feedback_spans(visible))
            if index == first_user and field == "content" and instruction.strip():
                task = instruction.rstrip()
                if visible.count(task) == 1:
                    a = begin + len(visible[:visible.index(task)].encode("utf-8"))
                    spans.append((begin, a, "framework_rules"))
                    spans.append((a, a + len(task.encode("utf-8")), "task_instruction"))
    classes = ["protocol_or_unclassified"] + sorted({kind for _, _, kind in spans})
    paint = bytearray(len(raw))
    for begin, end, kind in spans:
        paint[begin:end] = bytes([classes.index(kind)]) * (end - begin)
    cursor, counts = 0, Counter()
    for chunk in chunks:
        labels = set(paint[cursor:cursor + len(chunk)])
        kind = "protocol_or_unclassified" if not chunk else classes[next(iter(labels))] if len(labels) == 1 else "boundary"
        counts[kind] += 1; cursor += len(chunk)
    return dict(status="native_prepared_input_tokens", tokens=len(rows), counts=dict(counts),
                input_token_count_matches_api=expected_tokens is not None, omitted_fields=omitted,
                unmapped_fields=unmapped, transformed_fields=transformed,
                parser_unmapped_fields=parser_unmapped,
                labels_source="role/field provenance + native contrast rendering + Terminus feedback frames; no semantic summary classifier")


def handler(upstream, out, before_request=None, collect_logprobs=True, after_response=None):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def relay(self):
            target = urllib.parse.urlsplit(self.path)
            path = target.path
            upstream_path = urllib.parse.urlunsplit(('', '', path, target.query, ''))
            record = None
            start = time.monotonic()
            received_ns = time.time_ns()
            value = None
            raw = b""
            if self.command == "POST":
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 <= size <= 32 * 1024 * 1024:
                    self.send_error(413)
                    return
                raw = self.rfile.read(size)
            if path == "/v1/chat/completions" and raw:
                record = out / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + "-" + uuid.uuid4().hex)
                record.mkdir(parents=True)
                (record / "request.original.json").write_bytes(raw)
                try:
                    value = json.loads(raw)
                except (ValueError, UnicodeError):
                    value = None
                if not isinstance(value, dict):
                    # The native endpoint owns request validation. Preserve an
                    # invalid JSON/envelope byte-for-byte so optional evidence
                    # cannot replace its normal HTTP 400 with a disconnect.
                    if before_request:
                        (record / 'gc-prepare.json').write_text(json.dumps(dict(
                            status='invalid_request_passthrough')), encoding='utf-8')
                elif before_request:
                    # Evidence is optional. A verifier failure must not turn a
                    # valid Agent request into a disconnected model request.
                    original = value
                    try:
                        value, audit = before_request(value, dict(self.headers))
                    except Exception as error:
                        value, audit = original, dict(status='preparation_error', reason=type(error).__name__)
                    (record / "gc-prepare.json").write_text(json.dumps(audit, ensure_ascii=False), encoding="utf-8")
                if collect_logprobs and isinstance(value, dict):
                    value["logprobs"] = True
                    value["top_logprobs"] = 0
                if collect_logprobs and isinstance(value, dict) and value.get("stream"):
                    value["stream_options"] = {**(value.get("stream_options") or {}), "include_usage": True}
                if isinstance(value, dict):
                    raw = json.dumps(value, ensure_ascii=False).encode("utf-8")
                (record / "request.forwarded.json").write_bytes(raw)
            headers = {k: v for k, v in self.headers.items() if k.lower() in ("authorization", "content-type", "accept")}
            request = urllib.request.Request(upstream.rstrip("/") + upstream_path, data=raw if self.command == "POST" else None,
                                             headers=headers, method=self.command)
            status = 502
            events = []
            response_started = False
            try:
                try:
                    response = opener.open(request, timeout=7200)
                except urllib.error.HTTPError as error:
                    response = error
                with response:
                    status = response.status
                    content_type = response.headers.get("Content-Type", "application/json")
                    self.send_response(status)
                    self.send_header("Content-Type", content_type)
                    self.end_headers()
                    response_started = True
                    if "text/event-stream" in content_type:
                        for line in response:
                            self.wfile.write(line)
                            self.wfile.flush()
                            if line.startswith(b"data:"):
                                payload = line[5:].strip()
                                if payload != b"[DONE]":
                                    events.append(json.loads(payload))
                        body = aggregate_stream(events)
                    else:
                        payload = response.read()
                        self.wfile.write(payload)
                        if record:
                            (record / "response.raw.json").write_bytes(payload)
                        body = json.loads(payload) if record else {}
                    if record:
                        (record / "response.json").write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
                        if events:
                            (record / "stream.jsonl").write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in events) + "\n", encoding="utf-8")
                        stats = profile_response(body) if status == 200 and not body.get("error") else dict(status="service_error")
                        stats.update(http_status=status, elapsed_seconds=time.monotonic() - start,
                                     started_unix_ns=received_ns, raw_request_saved=True)
                        (record / "profile.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
                        if after_response is not None and isinstance(value,dict):
                            try:
                                after_response(value,body,record,status)
                            except Exception as error:
                                # An observation hook cannot retract a response
                                # already sent to the Agent. Failed receipts
                                # cannot authorize future semantic deletion.
                                (record/'receipt-error.json').write_text(json.dumps(dict(
                                    error=repr(error))),encoding='utf-8')
            except Exception as error:
                if record:
                    (record / "capture-error.json").write_text(json.dumps(dict(error=repr(error), http_status=status)), encoding="utf-8")
                if not response_started:
                    self.send_error(502, "Owned model service unavailable; see capture log")
                self.close_connection = True

        do_POST = relay
        do_GET = relay
    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", default="http://127.0.0.1:18081")
    parser.add_argument("--port", type=int, default=18082)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    parsed = urllib.parse.urlsplit(args.upstream)
    if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
        parser.error("Use the owned loopback model service")
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Phase capture proxy at 127.0.0.1:{args.port}; no generation parameters overridden", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), handler(args.upstream, args.out)).serve_forever()


if __name__ == "__main__":
    main()
