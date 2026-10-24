"""Reject incomplete full-question evidence before remote quality acceptance."""
import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location(
    'historical_workload', Path(__file__).resolve().parents[1] / 'scripts/test_server_historical_workload.py')
workload = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workload)


class QueryCoverageLogTest(unittest.TestCase):
    span = 'KVMEM_TRACE gc_query_full begin=10 end=4106 tokens=4096 located=1'
    finish = ('KVMEM_HISTORICAL_REDUNDANCY features=52 question_tokens=4096 '
              'question_segments=128 invalid_questions=0')

    def accepted(self, lines, expected=1):
        return workload.full_question_report(lines, expected)['full_question_coverage_observed']

    def test_complete_and_duplicate_finish(self):
        self.assertTrue(self.accepted([self.span, self.finish, self.finish]))
        self.assertTrue(self.accepted([self.span, self.finish, self.span, self.finish], 2))

    def test_nonzero_suffix_is_not_full_question(self):
        self.assertFalse(self.accepted([self.span, self.finish.replace('question_tokens=4096', 'question_tokens=32')]))
        self.assertFalse(self.accepted([self.span, 'KVMEM_HISTORICAL_REDUNDANCY features=52']))

    def test_duplicate_finish_does_not_cover_missing_request(self):
        self.assertFalse(self.accepted([self.span, self.finish, self.finish], 2))
        self.assertFalse(self.accepted([self.span, self.finish, self.span], 2))

    def test_invalid_and_unlocated_spans(self):
        self.assertFalse(self.accepted([self.span.replace('located=1', 'located=0'), self.finish]))
        self.assertFalse(self.accepted([self.span.replace('end=4106', 'end=4105'), self.finish]))
        self.assertFalse(self.accepted([self.span, self.finish, 'full_question_invalid=1']))
        self.assertFalse(self.accepted([self.span, self.finish.replace('invalid_questions=0', 'invalid_questions=1')]))

    def test_each_request_needs_its_own_matching_count(self):
        second_span = 'KVMEM_TRACE gc_query_full begin=9000 end=9033 tokens=33 located=1'
        second_finish = ('KVMEM_HISTORICAL_REDUNDANCY features=52 question_tokens=33 '
                         'question_segments=2 invalid_questions=0')
        self.assertTrue(self.accepted([self.span, self.finish, second_span, second_finish], 2))
        self.assertFalse(self.accepted([self.span, second_finish, second_span, self.finish], 2))


if __name__ == '__main__':
    unittest.main()
