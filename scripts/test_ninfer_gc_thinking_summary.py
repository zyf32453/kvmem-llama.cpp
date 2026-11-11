import copy
import json
from pathlib import Path
import tempfile
import threading
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import urllib.request
import urllib.error
import unittest

from ninfer_gc_thinking_summary import (NativeAuxiliaryJson,ThinkingSummaryReview,coverage_decision,
    thinking_units,retained_observation,digest)
from ninfer_gc_step_controller import StepSummaryController,controller_handler
from ninfer_gc_agent_bridge import NativeClient
from ninfer_gc_step import attach_evidence


class SummaryCoverageTest(unittest.TestCase):
    def test_all_source_characters_including_tail_are_accounted_for(self):
        text='  Initial fact.\n\n'+'A complete paragraph has important context. '*90+'\n\n末尾独有的信息，必须保留。\n'
        units=thinking_units(text,max_chars=180)
        painted=set()
        for unit in units:
            self.assertEqual(text[unit['begin']:unit['end']],unit['text'])
            painted.update(range(unit['begin'],unit['end']))
        self.assertTrue(all(i in painted or char.isspace() for i,char in enumerate(text)))
        self.assertIn('末尾独有的信息',units[-1]['text'])

    def test_semantic_paraphrase_can_qualify_without_verbatim_copy(self):
        text='The completed build produced the requested executable successfully.'
        summary='The requested executable is available because the build succeeded.'
        units=thinking_units(text)
        review=dict(units=[dict(id='u1',fully_covered=True,summary_quotes=[summary],reason='same outcome')])
        result=coverage_decision(text,summary,units,review)
        self.assertNotIn(text,summary)
        self.assertEqual(result['accepted_units'],1)

    def test_similar_topics_changed_values_and_polarity_do_not_authorize(self):
        for text,summary,reason in [
            ('The total is 42 and the output is /app/result.json.','The total is 24 and the output is /app/result.json.','unpreserved_literal'),
            ('The test did not pass.','The test passed and the output is ready.','negation_requires_exact_support'),
            ('The program might still be running.','The program finished successfully.','unresolved_or_conditional')]:
            with self.subTest(text=text):
                units=thinking_units(text)
                review=dict(units=[dict(id='u1',fully_covered=True,summary_quotes=[summary],reason='incorrect model verdict')])
                result=coverage_decision(text,summary,units,review)
                self.assertEqual(result['accepted_units'],0)
                self.assertEqual(result['decisions'][0]['reason'],reason)

    def test_partial_foreign_duplicate_or_missing_support_is_retained(self):
        text='First fact.\n\nSecond fact.';units=thinking_units(text);summary='First fact is explicitly retained.'
        rows=[dict(id='u1',fully_covered=True,summary_quotes=['not in the summary'],reason='bad quote'),
            dict(id='u2',fully_covered=False,summary_quotes=[],reason='partial')]
        self.assertEqual(coverage_decision(text,summary,units,dict(units=rows))['covered_runs'],[])
        for modified in [rows[:1],[rows[0],rows[0]],[dict(rows[0],id='other'),rows[1]]]:
            with self.assertRaises(ValueError):coverage_decision(text,summary,units,dict(units=modified))

    def test_values_must_belong_to_the_quoted_support_not_an_unrelated_fact(self):
        text='Task A returned 42.';summary='Task A returned 24. Task B returned 42.'
        review=dict(units=[dict(id='u1',fully_covered=True,summary_quotes=['Task A returned 24.'],reason='misassociated value')])
        result=coverage_decision(text,summary,thinking_units(text),review)
        self.assertEqual(result['decisions'][0]['reason'],'unpreserved_literal')

    def test_list_ordinals_are_not_facts_and_numeric_substrings_do_not_match(self):
        source='3. The final balance is 42.'
        for summary,accepted in [('The final balance is 42.',1),('The final balance is 142.',0),
                ('The final balance is 42.5.',0),('The final balance is 420.',0)]:
            with self.subTest(summary=summary):
                review=dict(units=[dict(id='u1',fully_covered=True,summary_quotes=[summary],reason='same balance')])
                result=coverage_decision(source,summary,thinking_units(source),review)
                self.assertEqual(result['accepted_units'],accepted)

    def test_whole_thinking_is_summarized_and_every_unit_is_reviewed(self):
        text='\n\n'.join('Distinct fact '+str(i)+' remains.' for i in range(21))+'\n'
        seen=[]
        def generate(system,payload,schema,max_tokens):
            seen.append(copy.deepcopy(payload))
            if 'thinking' in payload:
                self.assertEqual(payload['thinking'],text)
                return dict(summary=text)
            return dict(units=[dict(id=u['id'],fully_covered=True,summary_quotes=[u['text']],reason='exact')
                for u in payload['thinking_units']])
        result=ThinkingSummaryReview(generate,review_batch=8).complete(text,'{}','x'*20000)
        self.assertEqual(len(seen),4)
        self.assertEqual(result['coverage']['accepted_units'],21)
        self.assertTrue(result['tool_feedback_excerpted'])
        self.assertEqual([u['id'] for call in seen[1:] for u in call['thinking_units']],['u'+str(i+1) for i in range(21)])

    def test_failed_review_keeps_summary_but_retires_no_source(self):
        def generate(system,payload,schema,max_tokens):
            if 'thinking' in payload:return dict(summary='The output is retained in the completed result.')
            return dict(units=[])
        result=ThinkingSummaryReview(generate).complete('The source contains a distinct claim.','{}','observed')
        self.assertEqual(result['coverage_status'],'review_failed')
        self.assertEqual(result['coverage']['covered_runs'],[])
        self.assertTrue(result['summary'])

    def test_auxiliary_uses_actual_native_ids_and_disables_cache(self):
        seen=[]
        def post(path,value):
            seen.append((path,copy.deepcopy(value)))
            if path=='/apply-template':return dict(tokens=[dict(id=123,piece='literal')])
            return dict(content='{"summary":"retained"}',stop_type='eos')
        result=NativeAuxiliaryJson(post,'owned-model')('system',dict(data='x'),{},20)
        self.assertEqual(result,dict(summary='retained'))
        self.assertEqual(seen[-1][1]['prompt'],[123])
        self.assertIs(seen[-1][1]['cache_prompt'],False)
        self.assertNotIn('prompt_cache_key',seen[-1][1])
        self.assertFalse(seen[0][1]['chat_template_kwargs']['enable_thinking'])
        def oversized(path,value):return dict(tokens=[dict(id=1)]*3)
        with self.assertRaises(ValueError):NativeAuxiliaryJson(oversized,'owned',max_input_tokens=2)('s',{}, {},10)

    def test_summary_observation_keeps_original_feedback_bytes(self):
        original='New Terminal Output:\n\n完整的工具反馈\n'
        result=retained_observation('step','Retained result summary.',original)
        self.assertIn(original,result['content'])
        self.assertEqual(result['content'][result['summary_begin']:result['summary_end']],'Retained result summary.')



if __name__ == '__main__': unittest.main()
