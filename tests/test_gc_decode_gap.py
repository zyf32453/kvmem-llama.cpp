import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from check_gc_decode_gap import analyze


class DecodeGapTraceTests(unittest.TestCase):
    def test_commit_follows_decode_and_precedes_active_selection(self):
        result = analyze([
            "KVMEM_TRACE gc_gap phase=decode_begin decode_enter_ns=10 commit_start_ns=0 commit_end_ns=0 prepared=0 committed=0\n",
            "KVMEM_TRACE gc_gap phase=decode_end decode_return_ns=100 commit_start_ns=110 commit_end_ns=210 prepared=2 committed=1\n",
            "KVMEM_TRACE historical_active_window_begin repository=1 window=2 rows=64 select_ns=240 last_gc_commit_ns=210 skips=1\n",
        ], require_commit=True, require_selection=True)
        self.assertTrue(result["passed"])
        self.assertEqual(result["committed_intents"], 1)
        self.assertEqual(result["windows_after_commit"], 1)

    def test_rejects_commit_during_decode(self):
        result = analyze([
            "KVMEM_TRACE gc_gap phase=decode_begin decode_enter_ns=100 commit_start_ns=0 commit_end_ns=0 prepared=0 committed=0\n",
            "KVMEM_TRACE gc_gap phase=decode_end decode_return_ns=200 commit_start_ns=190 commit_end_ns=210 prepared=1 committed=1\n",
        ])
        self.assertFalse(result["passed"])
        self.assertIn("retirement preceded decode completion", result["errors"][0])

    def test_rejects_selection_before_commit(self):
        result = analyze([
            "KVMEM_TRACE gc_gap phase=request_begin start_ns=100 end_ns=200 prepared=1 committed=1\n",
            "KVMEM_TRACE historical_active_window_begin repository=1 window=3 rows=64 select_ns=190 last_gc_commit_ns=200 skips=0\n",
        ], require_selection=True)
        self.assertFalse(result["passed"])

    def test_rejects_preselection_commit_before_last_decode(self):
        result = analyze([
            "KVMEM_TRACE gc_gap phase=decode_begin decode_enter_ns=100 commit_start_ns=0 commit_end_ns=0 prepared=0 committed=0\n",
            "KVMEM_TRACE gc_gap phase=decode_end decode_return_ns=200 commit_start_ns=0 commit_end_ns=0 prepared=0 committed=0\n",
            "KVMEM_TRACE gc_gap phase=request_begin start_ns=190 end_ns=210 prepared=1 committed=1\n",
        ])
        self.assertFalse(result["passed"])

    def test_commit_precedes_next_decode_entry(self):
        result = analyze([
            "KVMEM_TRACE gc_gap phase=decode_begin decode_enter_ns=100 commit_start_ns=0 commit_end_ns=0 prepared=0 committed=0\n",
            "KVMEM_TRACE gc_gap phase=decode_end decode_return_ns=200 commit_start_ns=0 commit_end_ns=0 prepared=0 committed=0\n",
            "KVMEM_TRACE gc_gap phase=decode_begin decode_enter_ns=320 commit_start_ns=230 commit_end_ns=300 prepared=1 committed=1\n",
            "KVMEM_TRACE historical_active_window_begin repository=1 window=2 rows=64 select_ns=350 last_gc_commit_ns=300 skips=1\n",
            "KVMEM_TRACE gc_gap phase=decode_end decode_return_ns=400 commit_start_ns=0 commit_end_ns=0 prepared=0 committed=0\n",
        ], require_commit=True, require_selection=True)
        self.assertTrue(result["passed"])

    def test_rejects_commit_overlapping_next_decode_entry(self):
        result = analyze([
            "KVMEM_TRACE gc_gap phase=decode_begin decode_enter_ns=100 commit_start_ns=90 commit_end_ns=110 prepared=1 committed=1\n",
            "KVMEM_TRACE gc_gap phase=decode_end decode_return_ns=200 commit_start_ns=0 commit_end_ns=0 prepared=0 committed=0\n",
        ])
        self.assertFalse(result["passed"])
        self.assertTrue(any("overlapped the next decode" in e for e in result["errors"]))

    def test_rejects_retirement_inside_active_selection(self):
        result = analyze([
            "KVMEM_TRACE gc_gap phase=decode_begin decode_enter_ns=100 commit_start_ns=0 commit_end_ns=0 prepared=0 committed=0\n",
            "KVMEM_TRACE gc_gap phase=before_select start_ns=120 end_ns=130 prepared=1 committed=1\n",
            "KVMEM_TRACE gc_gap phase=decode_end decode_return_ns=200 commit_start_ns=0 commit_end_ns=0 prepared=0 committed=0\n",
        ])
        self.assertFalse(result["passed"])
        self.assertTrue(any("unknown phase before_select" in e for e in result["errors"]))


if __name__ == "__main__":
    unittest.main()
