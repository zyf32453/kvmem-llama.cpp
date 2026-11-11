import copy
import hashlib
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import urllib.error
import urllib.request
import unittest
from agent_phase_capture import aggregate_stream, feedback_spans, handler, object_spans, profile_native_prompt, profile_response
from agent_phase_profile import compare_profiles, reconcile_trace


class PhaseProfileTest(unittest.TestCase):
    def test_no_duplicate_prefix_and_no_missing_token_span(self):
        trace = dict(initial=dict(tokens=2, counts=dict(prefix=2)), rounds=[dict(turn=1, prompt_tokens=2,
            output_tokens=2, following_input_tokens=1, context_after=5, outcome="command_then_observation",
            counts_before=dict(prefix=2), counts_added=dict(thinking=2, feedback=1),
            counts_after=dict(prefix=2, thinking=2, feedback=1), events=[
                dict(kind="thinking", origin="model_output", begin=2, end=4, tokens=2),
                dict(kind="feedback", origin="following_input", begin=4, end=5, tokens=1)])])
        result = reconcile_trace(trace, {"prefix", "thinking", "feedback"})
        self.assertEqual(sum(result["output_counts"].values()), 2)
        self.assertEqual(sum(result["final_context_counts"].values()), 5)
        bad = copy.deepcopy(trace)
        bad["rounds"][0]["events"][1]["begin"] = 3
        with self.assertRaises(ValueError):
            reconcile_trace(bad, {"prefix", "thinking", "feedback"})

    def test_accepted_content_bytes_not_field_retokenization(self):
        text = json.dumps(dict(analysis="思考后结论", plan="next", commands=[dict(keystrokes="echo ok")]), ensure_ascii=False)
        rows = [dict(bytes=list(c.encode())) for c in text]
        native = dict(usage=dict(prompt_tokens=100, completion_tokens=len(rows) + 12,
            completion_tokens_details=dict(reasoning_tokens=10)), choices=[dict(message=dict(content=text), logprobs=dict(content=rows))])
        result = profile_response(native)
        self.assertEqual(sum(result["counts"].values()), len(rows) + 12)
        self.assertEqual(result["counts"]["unexposed_control_tokens"], 2)
        self.assertGreater(result["counts"]["command_request"], 0)
        self.assertEqual(result["counts"]["thinking"], 10)
        invalid = profile_response(native, classify_fields=False)
        self.assertNotIn('command_request', invalid['counts'])
        self.assertEqual(invalid['counts']['assistant_text'], len(rows))
        damaged = copy.deepcopy(native); damaged["choices"][0]["logprobs"]["content"][0]["bytes"] = [1]
        self.assertEqual(profile_response(damaged)["status"], "coarse_only")
        native["choices"][0]["logprobs"]["content"] = [dict(bytes=list(text.encode()))]
        native["usage"]["completion_tokens"] = 11
        self.assertEqual(profile_response(native)["counts"], dict(boundary=1, thinking=10))

    def test_stream_and_missing_usage_do_not_become_zero(self):
        body = aggregate_stream([dict(choices=[dict(delta=dict(content="ok"), logprobs=dict(content=[dict(bytes=[111, 107])]))]),
            dict(choices=[], usage=dict(completion_tokens=1, completion_tokens_details=dict(reasoning_tokens=0)))])
        self.assertEqual(profile_response(body)["counts"], dict(assistant_text=1, thinking=0))
        self.assertEqual(profile_response({})["status"], "usage_unavailable")
        self.assertEqual(object_spans('{"analysis":"broken'), [])

    def test_whole_prompt_alignment_and_repeated_field_provenance(self):
        request = dict(messages=[dict(role="user", content="Rules TASK"),
            dict(role="assistant", reasoning_content="THINK", content='{"analysis":"YES","commands":[]}'),
            dict(role="user", content="Rules TASK")])
        def render(value):
            return "".join("[" + m["role"] + "]" + m.get("reasoning_content", "") + m["content"] + "[/]" for m in value["messages"])
        def native(path, value):
            if path == "/apply-template":
                text=render(value)
                result=dict(prompt=text)
                if value.get('return_tokens'):
                    result['tokens']=[dict(id=1000+ord(c),piece=c) for c in text]
                return result
            raise AssertionError('Whole-input profiling must use actual prepared tokens')
        stats = profile_native_prompt(request, native, "TASK", expected_tokens=len(render(request)))
        self.assertEqual(sum(stats["counts"].values()), len(render(request)))
        self.assertEqual(stats["counts"]["thinking"], 5)
        self.assertEqual(stats["counts"]["task_instruction"], 4)
        self.assertEqual(stats["counts"]["user_text"], 10)
        self.assertEqual(stats["unmapped_fields"], [])
        with self.assertRaises(ValueError):
            profile_native_prompt(request, native, expected_tokens=1)

    def test_feedback_nested_spans_and_invalid_reply_not_actions(self):
        text = 'Previous response had warnings:\nwarning\nNew Terminal Output:\n中文\n[Executor status]\nOK'
        spans = feedback_spans(text)
        self.assertEqual([k for _, _, k in spans], ['parser_warning', 'terminal_observation', 'execution_status'])
        paint = ['user_text'] * len(text.encode())
        for a, b, kind in spans:
            paint[a:b] = [kind] * (b-a)
        self.assertEqual(len(paint), len(text.encode()))
        self.assertEqual(paint[-2:], ['execution_status'] * 2)
        self.assertEqual(feedback_spans('Previous response had parsing errors:\nOops'), [(0, 42, 'parser_feedback')])

    def test_native_template_trimming_is_proven_and_unmapped_is_visible(self):
        request = dict(messages=[dict(role='user', content='Rules TASK\n'),
            dict(role='assistant', reasoning_content=' THINK\n', content=' {"commands":[1]}\n')])
        def render(v):
            return ''.join('['+m['role']+']'+m.get('reasoning_content','').strip()+m['content'].strip()+'[/]' for m in v['messages'])
        def native(path, v):
            s=render(v)
            return dict(prompt=s, **(dict(tokens=[dict(id=1000+ord(c),piece=c) for c in s]) if v.get('return_tokens') else {}))
        verdicts={hashlib.sha256(request['messages'][1]['content'].encode()).hexdigest():False}
        stats=profile_native_prompt(request, native, 'TASK', content_validity=verdicts)
        self.assertEqual(stats['unmapped_fields'], [])
        self.assertEqual(len(stats['transformed_fields']), 3)
        self.assertEqual(stats['counts']['thinking'], 5)
        self.assertEqual(stats['counts']['assistant_text'], len('{"commands":[1]}'))
        self.assertNotIn('command_request', stats['counts'])
        self.assertEqual(stats['parser_unmapped_fields'], [])
        unknown=profile_native_prompt(request,native,'TASK',content_validity={})
        self.assertEqual(unknown['parser_unmapped_fields'],[dict(message=1,field='content')])

    def test_comparison_uses_weighted_output_and_single_context(self):
        teacher=dict(task='test', groups=dict(thinking=['thinking'],structure=['protocol','boundary']),
            output_counts=dict(thinking=3,protocol=1),final_context_counts=dict(thinking=3,protocol=5),
            rounds=[dict(outcome='format_retry')])
        current=dict(requests=2,detailed_requests=2,native_usage_requests=2,
            generated_detailed_counts=dict(thinking=1,protocol=3),
            latest_logical_context=dict(counts=dict(thinking=1,protocol=9)),
            latest_successful_input=dict(unmapped_fields=[]),service_errors=[],
            rounds=[dict(prompt_tokens=100),dict(prompt_tokens=300)])
        result=compare_profiles(teacher,current)
        self.assertEqual(result['current_context_tokens'],10)
        thinking=next(r for r in result['grouped_rows'] if r['denominator']=='model_output' and r['group']=='thinking')
        self.assertEqual(thinking['current_fraction'],.25)
        self.assertEqual(thinking['difference_percentage_points'],-50)

    def test_http_capture_preserves_generation_and_separates_503(self):
        observed = []
        class Model(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                value = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                observed.append(value)
                status = 503 if len(observed) == 2 else 200
                body = dict(error=dict(code="capacity")) if status == 503 else dict(
                    choices=[dict(message=dict(content="ok"), logprobs=dict(content=[dict(bytes=[111, 107])]))],
                    usage=dict(completion_tokens=1, completion_tokens_details=dict(reasoning_tokens=0)))
                self.send_response(status); self.send_header("Content-Type", "application/json"); self.end_headers()
                self.wfile.write(json.dumps(body).encode())
        with tempfile.TemporaryDirectory() as directory:
            upstream = ThreadingHTTPServer(("127.0.0.1", 0), Model)
            capture = ThreadingHTTPServer(("127.0.0.1", 0), handler(f"http://127.0.0.1:{upstream.server_port}", Path(directory)))
            threads = [threading.Thread(target=s.serve_forever, daemon=True) for s in (upstream, capture)]
            for thread in threads:
                thread.start()
            try:
                original = dict(model="fixture", messages=[dict(role="user", content="hi")], temperature=.6, max_tokens=16, seed=1000)
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                for expected in (200, 503):
                    req = urllib.request.Request(f"http://127.0.0.1:{capture.server_port}/v1/chat/completions",
                        data=json.dumps(original).encode(), headers={"Content-Type": "application/json"})
                    try:
                        response = opener.open(req, timeout=5)
                    except urllib.error.HTTPError as error:
                        response = error
                    with response:
                        self.assertEqual(response.status, expected)
                        json.load(response)
                for request in observed:
                    self.assertEqual({k: v for k, v in request.items() if k not in ("logprobs", "top_logprobs")}, original)
                profiles = [json.loads(p.read_text()) for p in Path(directory).glob("*/profile.json")]
                self.assertEqual(sorted(p["http_status"] for p in profiles), [200, 503])
                self.assertEqual(next(p for p in profiles if p["http_status"] == 503)["status"], "service_error")
            finally:
                for server in (capture, upstream):
                    server.shutdown(); server.server_close()


if __name__ == "__main__":
    unittest.main()
