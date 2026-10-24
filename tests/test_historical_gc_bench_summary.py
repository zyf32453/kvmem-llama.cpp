"""Keep native decode timings separate from proof that the full Query path ran."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location(
    "gc_bench_summary", Path(__file__).resolve().parents[1] / "scripts/summarize_historical_gc_bench.py")
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class HistoricalGcBenchSummaryTest(unittest.TestCase):
    def setUp(self):
        self.config = {
            "fixture": bench.FULL_QUERY_FIXTURE, "policy": "redundancy", "rounds": 1,
            "input_tokens": 6144, "decode_steps": 32, "block_tokens": 32,
        }
        self.full_config = {
            "fixture": bench.FULL_QUERY_FIXTURE, "full_query_span": [0, 6144],
            "synthetic_thinking_span": [0, 6144], "thinking_span_is_synthetic": True,
            "generated_tokens_marked_thinking": False,
            "off_thinking_policy_armed": False,
            "generated_query_capture": True, "thinking_pool_limit": 56,
            "off_includes_query_collection": True, "worker_starts_after_prefill": True,
            "task_quality_evaluation": False,
        }
        self.rows = []
        self.evidence = []
        for round_id in range(2):
            for mode in ("off", "shadow", "delete"):
                active = mode != "off"
                delete = mode == "delete"
                self.rows.append({
                    "round": round_id, "warmup": round_id == 0, "mode": mode,
                    "stable_window": True, "same_output": True, "tokens": list(range(32)),
                    "policy_failures": 0, "worker_failures": 0, "decode_steps": 32,
                    "step_ms": [10.0] * 32, "generation_ms": 320.0, "prefill_ms": 100.0,
                    "active_release_steps": int(delete), "first_release_step": int(delete),
                    "committed_before_decode": 0, "committed_after_decode": 5 if delete else 0,
                    "host_released_before_decode": 0, "host_released_after_decode": 5 if delete else 0,
                    "slots_reclaimed_before_decode": 0, "slots_reclaimed_after_decode": 0,
                    "shadow_after_decode": 5 if mode == "shadow" else 0,
                })
                self.evidence.append({
                    "round": round_id, "warmup": round_id == 0, "mode": mode,
                    "fixture": bench.FULL_QUERY_FIXTURE, "thinking_span_is_synthetic": True,
                    "generated_tokens_marked_thinking": False,
                    "thinking_policy_armed": active,
                    "full_question_coverage": True, "question_tokens": 6144, "question_segments": 192,
                    "question_bytes": 12582912, "query_samples": 1, "invalid_questions": 0,
                    "observation_samples_before_decode": 0, "observation_samples_after_decode": 1,
                    "fast_samples_before_decode": 0, "fast_samples_after_decode": 25,
                    "question_score_evals": 52 * 192 if active else 0,
                    "question_score_yields": 52 * 23 if active else 0,
                    "feature_blocks": 52 if active else 0, "pair_comparisons": 52 * 51 if active else 0,
                    "scored_rounds": 1 if active else 0, "max_round_features": 52 if active else 0,
                    "thinking_pool_limit": 56,
                })

    def summarize(self, require_full_query=False, full_configs=None, evidence=None, pass_line=True):
        lines = ["KVMEM_GC_BENCH_CONFIG " + json.dumps(self.config)]
        for config in [self.full_config] if full_configs is None else full_configs:
            lines.append("KVMEM_GC_FULL_QUERY_CONFIG " + json.dumps(config))
        lines.extend("KVMEM_GC_BENCH " + json.dumps(row) for row in self.rows)
        lines.extend("KVMEM_GC_FULL_QUERY " + json.dumps(row)
                     for row in (self.evidence if evidence is None else evidence))
        if pass_line:
            lines.append("PASS GC decode benchmark: identical output, stable GPU window, release during decoding")
        class Log:
            def read_text(self, **_):
                return "\n".join(lines)

            def read_bytes(self):
                return self.read_text().encode("utf-8")
        return bench.summarize(Log(), require_full_query=require_full_query)

    def test_complete_new_fixture_preserves_pairing_and_evidence(self):
        summary, rows = self.summarize(require_full_query=True, evidence=list(reversed(self.evidence)))
        gate = summary["full_query_validation"]
        self.assertEqual(gate["status"], "passed")
        self.assertEqual(gate["verified_requests_including_warmups"], 6)
        self.assertEqual(gate["max_round_features"], 52)
        self.assertEqual(gate["minimum_generated_observation_samples"], 1)
        self.assertEqual(summary["modes"]["delete"]["median"]["p95_step_ms"], 10.0)
        self.assertEqual(summary["paired_changes_percent"]["delete"]["tokens_per_second"]["median"], 0.0)
        for row in rows:
            self.assertEqual(row["full_query_evidence"]["mode"], row["mode"])
            self.assertEqual(row["full_query_evidence"]["round"], row["round"])

    def test_legacy_still_summarizes_but_cannot_pass_full_query(self):
        self.config["fixture"] = "mentor-27b"
        summary, rows = self.summarize(full_configs=[], evidence=[])
        self.assertEqual(summary["full_query_validation"]["status"], "not_exercised")
        self.assertNotIn("full_query_evidence", rows[0])
        with self.assertRaisesRegex(ValueError, "legacy fixture"):
            self.summarize(require_full_query=True, full_configs=[], evidence=[])
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.summarize()

    def test_thinking_only_generated_sampling_is_not_all_decode_evidence(self):
        self.full_config["synthetic_thinking_span"] = [0, 6176]
        self.full_config["generated_tokens_marked_thinking"] = True
        with self.assertRaisesRegex(ValueError, "configuration"):
            self.summarize(require_full_query=True)

    def test_missing_and_duplicate_full_config(self):
        for configs in ([], [self.full_config, self.full_config]):
            with self.subTest(configs=len(configs)), self.assertRaises(ValueError):
                self.summarize(full_configs=configs)
        for key in self.full_config:
            config = dict(self.full_config)
            del config[key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                self.summarize(full_configs=[config])

    def test_wrong_scope_or_non_synthetic_metadata(self):
        for key, value in (("full_query_span", [6112, 6144]),
                           ("synthetic_thinking_span", [6144, 6176]),
                           ("thinking_span_is_synthetic", False), ("task_quality_evaluation", True),
                           ("thinking_pool_limit", 57), ("off_includes_query_collection", False)):
            config = dict(self.full_config, **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.summarize(full_configs=[config])

    def test_each_warmup_and_measured_request_requires_unique_evidence(self):
        for i in range(len(self.evidence)):
            with self.subTest(missing=i), self.assertRaises(ValueError):
                self.summarize(evidence=self.evidence[:i] + self.evidence[i + 1:])
            duplicate = copy.deepcopy(self.evidence)
            duplicate[i] = duplicate[(i + 1) % len(duplicate)]
            with self.subTest(duplicate=i), self.assertRaises(ValueError):
                self.summarize(evidence=duplicate)

    def test_every_capture_field_is_required(self):
        for key in self.evidence[0]:
            evidence = copy.deepcopy(self.evidence)
            del evidence[0][key]
            with self.subTest(missing=key), self.assertRaises(ValueError):
                self.summarize(evidence=evidence)

    def test_zero_or_partial_capture_cannot_pass_even_in_off_mode(self):
        for index in range(len(self.evidence)):
            for key, value in (("question_tokens", 32), ("question_segments", 1), ("question_bytes", 0),
                               ("query_samples", 0), ("invalid_questions", 1), ("full_question_coverage", False),
                               ("observation_samples_after_decode", 0), ("fast_samples_after_decode", 0)):
                evidence = copy.deepcopy(self.evidence)
                evidence[index][key] = value
                with self.subTest(index=index, key=key), self.assertRaises(ValueError):
                    self.summarize(evidence=evidence)

    def test_worker_must_run_in_active_modes_and_stay_off_in_baseline(self):
        for index in (1, 2, 4, 5):
            for key in ("question_score_evals", "question_score_yields", "feature_blocks", "pair_comparisons",
                        "scored_rounds", "max_round_features"):
                evidence = copy.deepcopy(self.evidence)
                evidence[index][key] = 0
                with self.subTest(index=index, key=key), self.assertRaises(ValueError):
                    self.summarize(evidence=evidence)
        for index in (0, 3):
            evidence = copy.deepcopy(self.evidence)
            evidence[index]["question_score_evals"] = 1
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.summarize(evidence=evidence)
        self.evidence[1]["max_round_features"] = 57
        with self.assertRaisesRegex(ValueError, "bounded scoring"):
            self.summarize()

    def test_physical_release_counters_must_confirm_timed_release(self):
        self.rows[2]["host_released_after_decode"] = 0
        with self.assertRaisesRegex(ValueError, "physical CPU release"):
            self.summarize()
        self.rows[2]["slots_reclaimed_after_decode"] = 1
        self.assertEqual(self.summarize()[0]["full_query_validation"]["status"], "passed")
        self.rows[1]["host_released_after_decode"] = 1
        with self.assertRaisesRegex(ValueError, "shadow unexpectedly"):
            self.summarize()

    def test_success_footer_and_equal_work_are_still_required(self):
        with self.assertRaisesRegex(ValueError, "successful completion"):
            self.summarize(pass_line=False)
        self.rows[2]["tokens"][-1] = 500
        with self.assertRaisesRegex(ValueError, "equal-work"):
            self.summarize()


if __name__ == "__main__":
    unittest.main()
