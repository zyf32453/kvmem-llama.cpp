import copy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
from ninfer_gc_step import attach_evidence, build_evidence, token_range, verified_write_step


class StepEvidenceTest(unittest.TestCase):
    def test_witness_keeps_partial_edge_tokens_across_page_boundaries(self):
        parts=['s']*64+['p']*63+['XBEGIN']+['M']*64+['ENDY']+['z']*64
        rows=[dict(id=i,piece=piece) for i,piece in enumerate(parts)]
        result='BEGIN'+'M'*64+'END'
        step=dict(id='edge-witness',kind='thinking',source_text='s'*64,result_text=result,
            observation_id=1,importance=1,cold=False,complete=True)
        proof=build_evidence(rows,''.join(parts),[step])['steps'][0]
        witness=''.join(parts[proof['result_begin']:proof['result_end']])
        self.assertIn(result,witness)  # Both partial edge tokens must survive.
        self.assertEqual(''.join(parts[proof['source_begin']:proof['source_end']]),'s'*64)

    def test_token_coordinates_and_digest(self):
        # Multi-byte tokens and byte-array pieces exercise actual UTF-8 boundaries.
        raw = "A" * 70 + "想" * 150 + "B" * 80 + "结果" * 80 + "Z" * 100
        rows = [{"id": i, "piece": c if c.isascii() else list(c.encode())} for i, c in enumerate(raw)]
        step = dict(id="s", kind="thinking", source_text="想" * 150,
            result_text="结果" * 80, observation_id=1, importance=.01, cold=True, complete=True)
        value = build_evidence(rows, raw, [step])
        self.assertEqual((value["steps"][0]["source_begin"], value["steps"][0]["source_end"]), (128, 192))
        self.assertEqual((value["steps"][0]["result_begin"], value["steps"][0]["result_end"]), (256, 512))
        self.assertEqual(value["prefix_sha256"], hashlib.sha256(struct.pack("<" + "I" * len(rows), *range(len(rows)))).hexdigest())
        bad = copy.deepcopy(rows); bad[0]["piece"] = "wrong"
        with self.assertRaises(ValueError):
            build_evidence(bad, raw, [step])
        with self.assertRaises(ValueError):
            token_range(rows, raw, "A")  # Ambiguous literal is never evidence.
        with self.assertRaises(ValueError):
            build_evidence(rows[:400], raw[:400], [step])

    def test_verified_write_rejects_changed_or_failed_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            file = root / "result.txt"
            content = "body" * 100
            file.write_bytes(content.encode())
            witness = f"<path>{file}</path>\n<type>file</type>\n<content>\nCreated file\n</content>"
            messages = [{"role": "assistant", "tool_calls": [{"id": "call1", "function": {
                "name": "write", "arguments": json.dumps(dict(file_path=str(file), content=content))}}]},
                {"role": "tool", "tool_call_id": "call1", "content": witness}]
            step = verified_write_step(messages, "call1", root)
            self.assertEqual((step["importance"], step["cold"], step["complete"]), (1, False, True))
            # Actual prepared IDs deliberately differ from text re-tokenization.
            def native(base, path, request):
                if path == "/apply-template":
                    arg = json.loads(request["messages"][0]["tool_calls"][0]["function"]["arguments"])
                    text = "A" * 70 + arg["content"] + "B" * 90 + request['messages'][1]['content'] + "Z" * 100
                    result = {"prompt": text}
                    if request.get('return_tokens'):
                        result['tokens']=[dict(id=1000+ord(c),piece=c) for c in text]
                    return result
                raise AssertionError('Must use prepared IDs, not re-tokenize flattened text')
            request = dict(messages=messages, prompt_cache_key="case")
            with patch("ninfer_gc_step.post_json", side_effect=native):
                evidence = attach_evidence("http://test", request, [step])["kvmem_gc"]
                self.assertEqual(evidence["steps"][0]["source_begin"], 128)
                self.assertNotIn("_write_call_id", evidence["steps"][0])
                prepared = native('', '/apply-template', dict(request, return_tokens=True))
                ids=[r['id'] for r in prepared['tokens']]
                self.assertEqual(evidence['prefix_sha256'],hashlib.sha256(struct.pack('<'+'I'*len(ids),*ids)).hexdigest())
            # If the template silently drops/quotes the tool body, a copy
            # elsewhere in the input must not acquire tool provenance.
            with patch("ninfer_gc_step.post_json", return_value={"prompt": content,
                    "tokens": [dict(id=ord(c),piece=c) for c in content]}):
                with self.assertRaises(ValueError):
                    attach_evidence("http://test", request, [step])
            with patch("ninfer_gc_step.post_json", return_value={"prompt": content}):
                with self.assertRaisesRegex(ValueError, 'actual prepared tokens'):
                    attach_evidence("http://test", request, [step])
            file.write_bytes(b"changed")
            with self.assertRaises(ValueError):
                verified_write_step(messages, "call1", root)
            file.write_bytes(content.encode())
            messages[-1]["content"] = "failed"
            with self.assertRaises(ValueError):
                verified_write_step(messages, "call1", root)
            messages[-1]["tool_call_id"] = "unrelated"
            with self.assertRaises(ValueError):
                verified_write_step(messages, "call1", root)


if __name__ == "__main__":
    unittest.main()
