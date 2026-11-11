import copy
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from ninfer_gc_inline_summary import response_format, summary_span
from ninfer_gc_step import attach_evidence, duplicate_feedback_page_sets
from ninfer_gc_step_controller import StepSummaryController


class NativeFixture:
    """A byte-exact template fixture, not a tokenizer or real model oracle."""
    def __init__(self, compact=True):
        self.inferences = 0
        self.compact = compact
    def json(self, path, value=None):
        if path.startswith('/props'):
            return dict(kvmem_gc=dict(policy='step_redundancy', block_tokens=64))
        if path != '/apply-template':
            self.inferences += 1
            raise AssertionError('Production GC requested model inference')
        text = ''.join('['+m['role']+']'+m.get('reasoning_content','')+m.get('content','')+'[/]'
                       for m in value['messages']) + '[assistant generation framing]'
        result = dict(prompt=text)
        if value.get('return_tokens'):
            result['tokens'] = ([ord(c) for c in text] if self.compact and value.get('token_ids_only')
                                else [dict(id=ord(c),piece=c) for c in text])
        return result


def content(summary=''):
    return json.dumps(dict(analysis=summary or 'inspect', plan='act', commands=[], task_complete=False), ensure_ascii=False)


class InlineSummaryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.native = NativeFixture()
        self.c = StepSummaryController(self.native, 'owned', Path(self.temp.name))
        self.messages = [dict(role='user',content='Task')]

    def tearDown(self):
        self.c.finalize(); self.c.worker.join(timeout=2); self.temp.cleanup()

    def reply(self, index, summary='', thinking=None, feedback=None):
        body = content(summary)
        thinking = thinking or ('Full original thinking with facts. '*60)
        request = dict(model='fixture',prompt_cache_key='owned',messages=copy.deepcopy(self.messages))
        response = dict(id='reply-'+str(index),choices=[dict(message=dict(content=body,reasoning_content=thinking))])
        self.c.note_response(request,response,'fixture',200)
        event = dict(session='owned',phase='tool_feedback_complete',content=body,thinking=thinking,
                     feedback=feedback or 'actual feedback '+str(index),tool_timeout=False)
        self.c.complete_step(event)
        self.messages += [dict(role='assistant',content=body,reasoning_content=thinking),
                          dict(role='user',content='New Terminal Output:\n'+event['feedback'])]
        return body,thinking

    def build_ready(self):
        self.reply(0)
        self.reply(1,'Outcome preserved; next action remains pending.')
        request = dict(model='fixture',messages=copy.deepcopy(self.messages))
        self.c.prepare(request)
        with self.c.condition:
            self.assertTrue(self.c.condition.wait_for(lambda:self.c.ready is not None,timeout=3))
        return request

    def test_first_presentation_preserves_the_step_and_rejects_a_later_request(self):
        request, _ = self.c.prepare(dict(model='fixture', messages=self.messages))
        body, thinking = content(), 'Tool reasoning.'
        raw = 'Complete repeated tool feedback.'
        self.c.note_response(request, dict(id='owned-first',
            choices=[dict(message=dict(content=body, reasoning_content=thinking))],
            usage=dict(prompt_tokens=100, completion_tokens=20, total_tokens=120),
            timings=dict(cache_n=80, prompt_n=20, predicted_n=20)), 'fixture', 200)
        receipt = self.c.complete_step(dict(session='owned', phase='tool_feedback_complete',
            content=body, thinking=thinking, feedback=raw, tool_timeout=False))
        self.assertEqual(receipt['preceding_frontier'], 120)
        row = self.c.records[receipt['step_id']]
        revised = '[Archived repeat.]'
        event = dict(session='owned', step_id=receipt['step_id'], message=receipt['feedback_message'],
                     preceding_frontier=receipt['preceding_frontier'], original_sha256=row['feedback_sha256'],
                     feedback=revised)
        self.assertEqual(self.c.first_presentation(dict(event, preceding_frontier=119))['status'],
                         'first_presentation_stale')
        self.assertEqual(self.c.first_presentation(event)['status'], 'first_presentation_registered')
        self.assertEqual(self.c.history_epoch, 0)
        for presentation in (raw, revised):
            messages = self.messages + [dict(role='assistant', content=body, reasoning_content=thinking),
                                        dict(role='user', content=presentation)]
            self.assertTrue(self.c._matches(row, messages))
        self.c.prepare(dict(model='fixture', messages=messages))
        self.assertEqual(self.c.first_presentation(event)['status'], 'first_presentation_stale')

    def test_json_value_span_handles_unicode_escapes_and_rejects_ambiguity(self):
        value='文件 "x"\nnot complete'
        raw=content(value)
        summary,begin,end=summary_span(raw)
        self.assertEqual(summary,value)
        self.assertEqual(json.loads(raw[begin:end]),value)
        for bad in ('{"analysis":"a","analysis":"b"}',content('a'*4097),
                    '{"nested":{"analysis":"spoof"}}'):
            with self.assertRaises(ValueError):summary_span(bad)
        schema=response_format()['json_schema']['schema']
        self.assertNotIn('step_summary',schema['properties'])
        self.assertNotIn('maxLength',schema['properties']['analysis'])

    def test_whole_pages_sealed_anchor_and_native_ack_stop_republication(self):
        request=self.build_ready()
        forwarded,audit=self.c.prepare(request)
        proof=forwarded['kvmem_gc'];scope=proof['steps'][0]
        self.assertEqual(audit['attached'],1)
        for name in ('source_begin','source_end','result_begin','result_end'):
            self.assertEqual(scope[name]%64,0)
        self.assertGreater(scope['source_end']-scope['source_begin'],scope['result_end']-scope['result_begin'])
        self.assertEqual(proof['prefix_tokens'],scope['result_end'])
        self.assertLess(proof['prefix_tokens'],len(self.native.json('/apply-template',dict(request,return_tokens=True))['tokens']))
        response=dict(id='ack',choices=[dict(message=dict(content=content(),reasoning_content='r'))],
            kvmem=dict(historical_gc=dict(available=True,scopes=[dict(id=scope['id'],state='no_net_gain',selected_pages=0,retired_pages=0)])))
        self.c.note_response(forwarded,response,'fixture',200)
        again,_=self.c.prepare(request)
        self.assertNotIn('kvmem_gc',again)
        self.assertEqual(self.c.records[scope['id']]['status'],'native_no_net_gain')
        self.assertEqual(self.c.counts['native_acknowledged'],1)
        self.assertEqual(self.native.inferences,0)

    def test_worker_cannot_block_prepare_or_post_tool_callback(self):
        entered,release=threading.Event(),threading.Event()
        original=self.native.json
        def blocked(path,value=None):
            entered.set(); release.wait(3); return original(path,value)
        self.native.json=blocked
        self.reply(0);self.reply(1,'A compact retained result.')
        request=dict(model='fixture',messages=copy.deepcopy(self.messages))
        try:
            self.c.prepare(request);self.assertTrue(entered.wait(1))
            start=time.monotonic();self.c.prepare(request)
            self.assertLess(time.monotonic()-start,.1)
            start=time.monotonic();self.reply(2,'Another ordinary result.')
            self.assertLess(time.monotonic()-start,.1)
        finally:release.set()

    def test_edited_history_is_never_bound_to_old_token_proof(self):
        request=self.build_ready()
        request['messages'][1]['reasoning_content']='rewritten source'
        result,_=self.c.prepare(request)
        self.assertNotIn('kvmem_gc',result)
        self.assertEqual(self.c.counts['stale_ready_batches'],1)

    def test_explicit_thinking_budget_invalidates_ready_template_proof(self):
        request=self.build_ready()
        request['thinking_budget'] = 4096
        result,_=self.c.prepare(request)
        self.assertNotIn('kvmem_gc',result)
        self.assertEqual(self.c.counts['stale_ready_batches'],1)

    def test_late_response_cannot_republish_pre_edit_receipt(self):
        self.reply(0)
        request,_=self.c.prepare(dict(model='fixture',messages=copy.deepcopy(self.messages)))
        count=self.c.counts['main_responses']
        self.c.history_edit(dict(session='owned',message=1,edit_id='b'*64))
        response=dict(id='old-inflight',choices=[dict(message=dict(content=content(),reasoning_content='Old thinking'))])
        self.c.note_response(request,response,'late',200)
        self.assertEqual(self.c.counts['main_responses'],count)
        self.assertFalse(self.c.receipts)

    def test_card_history_edit_cancels_inflight_position_proof(self):
        entered,release=threading.Event(),threading.Event()
        original=self.c._build
        def blocked(*args):
            entered.set();release.wait(2)
            return original(*args)
        self.c._build=blocked
        self.reply(0);self.reply(1,'A retained outcome.')
        request=dict(model='fixture',messages=copy.deepcopy(self.messages))
        try:
            self.c.prepare(request)
            self.assertTrue(entered.wait(1))
            event=dict(session='owned',message=2,edit_id='a'*64)
            first=self.c.history_edit(event)
            self.assertEqual(self.c.history_edit(event),first)
            self.assertEqual(self.c.history_epoch,1)
            self.assertTrue(all(r['status']=='source_rewritten' for r in self.c.records.values()))
        finally:release.set()
        deadline=time.monotonic()+3
        while self.c.counts['stale_ready_batches']==0 and time.monotonic()<deadline:time.sleep(.01)
        self.assertEqual(self.c.counts['stale_ready_batches'],1)
        self.assertIsNone(self.c.ready)

    def test_native_field_edge_trimming_does_not_discard_whole_thinking(self):
        original=self.native.json
        def trimmed(path,value=None):
            if path=='/apply-template':
                value=copy.deepcopy(value)
                for m in value['messages']:
                    if 'reasoning_content' in m:m['reasoning_content']=m['reasoning_content'].strip()
            return original(path,value)
        self.native.json=trimmed
        self.reply(0,thinking='\n  '+('Whole source reasoning. '*50)+'\n\n')
        self.reply(1,'A concise retained outcome.')
        request=dict(model='fixture',messages=copy.deepcopy(self.messages))
        self.c.prepare(request)
        with self.c.condition:
            self.assertTrue(self.c.condition.wait_for(lambda:self.c.ready is not None,timeout=3))
        self.assertGreater(self.c.ready['proof']['steps'][0]['source_end'],64)

    def test_batch_render_matches_separate_marker_proofs(self):
        request = dict(prompt_cache_key='owned', messages=[])
        steps = []
        for index, letter in enumerate(('甲', '乙')):
            source, result = letter * 256, letter + ' evidence verified'
            request['messages'] += [dict(role='assistant', reasoning_content=source),
                                    dict(role='user', content=result)]
            steps.append(dict(id=str(index), source_text=source, result_text=result,
                _source_message_span=dict(message=2*index, field='reasoning_content', begin=0, end=len(source)),
                _result_message_span=dict(message=2*index+1, field='content', begin=0, end=len(result))))
        request['messages'].append(dict(role='user', content='padding to seal the result witness ' * 10))
        calls = []
        def post(path, value):
            calls.append(bool(value.get('return_tokens')))
            return self.native.json(path, value)
        batch = attach_evidence('', request, steps, post=post)
        self.assertEqual(calls, [True, False])
        calls.clear()
        with patch('ninfer_gc_step.inline_marker_spans', side_effect=ValueError('separate marker control')):
            control = attach_evidence('', request, steps, post=post)
        self.assertEqual(calls, [True, False, False, False, False])
        self.assertEqual(batch, control)

    def test_exact_feedback_proof_keeps_message_text_and_context(self):
        self.c.duplicate_feedback = True
        repeated = ''.join(f'read record {i:03d} from the same file\n' for i in range(30))
        self.reply(0, feedback='first screen\n'+repeated+'first-only output\n'*12)
        self.reply(1, feedback='second screen text\n'+repeated+'new-only output\n'*12)
        request = dict(model='fixture', messages=copy.deepcopy(self.messages))
        self.c.prepare(request)
        with self.c.condition:
            self.assertTrue(self.c.condition.wait_for(lambda:self.c.ready is not None,timeout=3))
        forwarded, _ = self.c.prepare(request)
        self.assertEqual(forwarded['messages'], request['messages'])
        rows = self.native.json('/apply-template', dict(request, return_tokens=True))['tokens']
        scopes = [step for step in forwarded['kvmem_gc']['steps'] if step['kind']=='duplicate_tool_feedback']
        self.assertTrue(scopes)
        for step in scopes:
            self.assertEqual(rows[step['source_begin']:step['source_end']],
                             rows[step['result_begin']:step['result_end']])
            source, witness = duplicate_feedback_page_sets(step)
            self.assertGreaterEqual(witness.start, source.stop)
            self.assertTrue(set(source).issubset(self.c.feedback_source_pages))
            self.assertTrue(set(witness).issubset(self.c.feedback_witness_pages))
        self.assertEqual(self.native.inferences,0)

    def test_observe_uses_same_summary_protocol_without_deletion(self):
        request=self.build_ready();self.c.mode='observe'
        result,_=self.c.prepare(request)
        self.assertNotIn('kvmem_gc',result)
        self.assertNotIn('step_summary',result['response_format']['json_schema']['schema']['required'])

    def test_native_observe_forwards_proof_to_the_observing_server(self):
        self.c.mode = 'native-observe'
        original = self.native.json
        def observing(path, value=None):
            reply = original(path, value)
            if path.startswith('/props'):
                reply['kvmem_gc']['observe_only'] = True
            return reply
        self.native.json = observing
        request = self.build_ready()
        forwarded, audit = self.c.prepare(request)
        self.assertTrue(forwarded['kvmem_gc']['steps'])
        self.assertGreater(audit['attached'], 0)

    def test_off_baseline_has_no_producer_or_template_work(self):
        controller=StepSummaryController(self.native,'disabled',Path(self.temp.name)/'off',mode='off')
        request=dict(model='fixture',messages=copy.deepcopy(self.messages))
        def unexpected(*args,**kwargs): raise AssertionError('Disabled GC touched Native template/props')
        self.native.json=unexpected
        result,audit=controller.prepare(request)
        self.assertIsNone(controller.worker)
        self.assertNotIn('kvmem_gc',result)
        self.assertEqual(audit['status'],'gc_disabled')
        self.assertEqual(result['response_format'],response_format())
        controller.note_response(result,dict(choices=[]),'off',200)
        self.assertEqual(controller.counts['main_responses'],1)
        self.assertFalse(controller.records)
        self.assertEqual(controller.finalize()['counters']['prepared_scopes'],0)

    def test_timeout_has_no_candidate_and_finalization_has_no_inference(self):
        body,thinking=self.reply(0)
        record=next(iter(self.c.records.values()));record['status']='incomplete_tools'
        self.reply(1,'Do not retire an incomplete step.')
        self.assertEqual(record['status'],'incomplete_tools')
        self.assertEqual(self.c.finalize()['status'],'closed_without_final_inference')
        self.assertEqual(self.native.inferences,0)

    def test_callback_receipt_race_never_waits_or_trusts_marker_text(self):
        event=dict(session='owned',phase='tool_feedback_complete',content=content(),thinking='source',feedback='x')
        self.assertEqual(self.c.complete_step(event)['status'],'receipt_pending')
        self.assertFalse(self.c.records)
        self.c.note_response(dict(prompt_cache_key='owned',messages=self.messages),
            dict(id='owned-race',choices=[dict(message=dict(content=content(),reasoning_content='source'))]),'fixture',200)
        self.assertEqual(len(self.c.records),1)
        with self.assertRaises(ValueError):self.c.complete_step(dict(event,session='foreign'))


if __name__=='__main__':unittest.main()
