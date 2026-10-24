"""Model-free checks for the real-model Query probe's acceptance boundaries."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import check_full_query_probe as probe


class FullQueryProbeTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.out = Path(self.directory.name)
        self.fixture = json.loads(probe.DEFAULT_FIXTURE.read_text(encoding='utf-8'))
        results = []
        requests = []
        messages = [{'role': 'system', 'content': self.fixture['system']}]
        lines = []
        for index, prompt in enumerate(self.fixture['turns']):
            content = json.dumps(self.fixture['probe']['expected'][index])
            messages.append({'role': 'user', 'content': prompt})
            requests.append({'messages': list(messages)})
            results.append({'turn': index + 1, 'done': True, 'http_status': 200,
                            'finish_reason': 'stop', 'content': content,
                            'usage': {'prompt_cache_hit_tokens': 0 if not index else 2200 * index}})
            messages.append({'role': 'assistant', 'content': content})
            begin = 10 + index * 2300
            lines.extend([
                f'KVMEM_TRACE gc_query_full begin={begin} end={begin+2200} tokens=2200 located=1',
                'KVMEM_HISTORICAL_REDUNDANCY features=0 question_tokens=2200 '
                'question_segments=69 invalid_questions=0'])
        self.write('workload.json', self.fixture)
        self.write('results.json', results)
        self.write('requests.json', requests)
        config = {'chat_template': 'native', 'policy': 'redundancy', 'mode': 'shadow', 'block_tokens': 32}
        self.write('experiment-config.json', config)
        self.write('summary.json', dict(config, gc_failure_observed=False, gc_finish_samples=[],
                                        required_gc_activity_observed=False))
        (self.out / 'server.log').write_text('\n'.join(lines), encoding='utf-8')

    def write(self, name, value):
        (self.out / name).write_text(json.dumps(value), encoding='utf-8')

    def change(self, name, edit):
        value = json.loads((self.out / name).read_text(encoding='utf-8'))
        edit(value)
        self.write(name, value)

    def check(self):
        return probe.check_probe(self.out)

    def test_complete_probe_does_not_claim_gc_or_performance(self):
        report = self.check()
        self.assertTrue(report['passed'], report)
        self.assertFalse(report['gc_activity_observed'])
        self.assertFalse(report['gc_quality_accepted'])
        self.assertFalse(report['performance_accepted'])

    def test_fixture_places_unique_constraints_across_entire_query(self):
        for prompt, expected in zip(self.fixture['turns'], self.fixture['probe']['expected']):
            positions = [prompt.index(value) / len(prompt) for value in expected.values()]
            self.assertLess(positions[0], .05)
            self.assertTrue(.45 < positions[1] < .55)
            self.assertGreater(positions[2], .95)
            self.assertTrue(all(prompt.count(value) == 1 for value in expected.values()))
            self.assertGreater(len(prompt.split()), 1500)

    def test_wrong_field_and_extra_text_fail(self):
        self.change('results.json', lambda rows: rows[-1].update(content='Explanation: ' + rows[-1]['content']))
        self.assertFalse(self.check()['passed'])

    def test_old_round_value_fails_even_with_valid_json(self):
        self.change('results.json', lambda rows: rows[-1].update(content=rows[0]['content']))
        self.assertFalse(self.check()['passed'])

    def test_duplicate_json_keys_fail(self):
        self.change('results.json', lambda rows: rows[-1].update(content=rows[-1]['content'].replace(
            '{', '{"package_tag":"ignored",', 1)))
        self.assertFalse(self.check()['passed'])

    def test_non_native_template_or_missing_reuse_fails(self):
        self.change('experiment-config.json', lambda value: value.update(chat_template='stable_qwen_text_fixture'))
        self.assertFalse(self.check()['passed'])
        self.change('experiment-config.json', lambda value: value.update(chat_template='native'))
        self.change('results.json', lambda rows: rows[-1].update(usage={'prompt_cache_hit_tokens': 0}))
        self.assertFalse(self.check()['passed'])

    def test_matching_token_count_but_suffix_only_segments_fails(self):
        log = self.out / 'server.log'
        log.write_text(log.read_text().replace('question_segments=69', 'question_segments=1'), encoding='utf-8')
        self.assertFalse(self.check()['passed'])

    def test_missing_turn_scope_and_worker_failure_fail(self):
        log = self.out / 'server.log'
        original = log.read_text()
        log.write_text('\n'.join(original.splitlines()[:-2]), encoding='utf-8')
        self.assertFalse(self.check()['passed'])
        log.write_text(original, encoding='utf-8')
        self.change('summary.json', lambda value: value.update(gc_finish_samples=[{'worker_failures': 1}]))
        self.assertFalse(self.check()['passed'])

    def test_previous_question_scope_cannot_be_reused_as_current(self):
        log = self.out / 'server.log'
        original = log.read_text()
        log.write_text(original.replace('begin=2310 end=4510', 'begin=10 end=2210'), encoding='utf-8')
        self.assertFalse(self.check()['passed'])

    def test_changed_conversation_and_truncation_fail(self):
        self.change('requests.json', lambda rows: rows[-1]['messages'][-1].update(content='only the tail'))
        self.assertFalse(self.check()['passed'])
        self.change('results.json', lambda rows: rows[-1].update(finish_reason='length'))
        self.assertFalse(self.check()['passed'])

    def test_missing_evidence_fails_with_readable_result(self):
        (self.out / 'results.json').unlink()
        report = self.check()
        self.assertFalse(report['passed'])
        self.assertIn('missing or invalid', report['issues'][0])


if __name__ == '__main__':
    unittest.main()
