import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from ninfer_gc_tool_cards import (CostProfile, OriginalArchive, OwnedToolHistory, ToolCardPlanner,
                                 archive_transfer_command, INLINE_ARCHIVE_MAX_BYTES, sha256)
from ninfer_gc_step import NativeSpanCache, TokenByteIndex, terminal_feedback_ranges
from test_ninfer_gc_inline_summary import NativeFixture
from calibrate_ninfer_gc_tool_cards import matched_full_rounds


class CardsTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.archive=OriginalArchive(self.temp.name,'fixture-session')
        self.native=NativeFixture()
        # Deliberately explicit fixture rates, not a real model speed claim.
        self.profile=CostProfile(1000,1000000,evidence='synthetic test fixture',horizon_generated_tokens=4096)

    def tearDown(self):self.temp.cleanup()

    def history(self,count=5):
        h=OwnedToolHistory()
        messages=[dict(role='user',content='Keep the system instruction and task unchanged.')]
        for i in range(count):
            messages.append(dict(role='assistant',content='inspect output',reasoning_content='reason'))
            body=''.join(f'line {j:04d}: checksum {(j*97)%10000:04d}\n' for j in range(500))
            body+='error: genuine failure preserved\n'
            prompt='New Terminal Output:\n'+body+'\nContinue the original task.'
            h.register(feedback=body,raw_feedback='untruncated header\n'+body,commands=[dict(keystrokes='cat log')],timed_out=False)
            h.bind(messages,prompt)
            messages.append(dict(role='user',content=prompt))
        return h,messages

    def planner(self,**kwargs):
        return ToolCardPlanner(self.archive,self.native,self.profile,min_context_tokens=0,**kwargs)

    def test_retention_counts_raw_utf8_and_wrapper_bytes_after_eviction(self):
        history = OwnedToolHistory(max_records=2, max_bytes=64)
        messages = []

        def bind(body, raw):
            history.register(feedback=body, raw_feedback=raw, commands=[], timed_out=False)
            prompt = 'Observation\n' + body
            history.bind(messages, prompt)
            messages.append(dict(role='user', content=prompt))
            return [r['ordinal'] for r in history.records]

        self.assertEqual(bind('one', '汉'*5), [1])
        self.assertEqual(bind('two', '汉'*5), [1, 2])  # 30 + 30 bytes
        self.assertEqual(bind('three', '短'), [2, 3])  # record limit
        self.assertEqual(bind('四', '源'*12), [4])  # 30 + 20 + 51 bytes
        self.assertEqual(bind('x', '源'*22), [4])  # oversized unbound output
        self.assertEqual(history.records[0]['raw_feedback'], '源'*12)

    def repeat_history(self, diagnostic=''):
        history = OwnedToolHistory()
        messages = [dict(role='user', content='Retain the user task.')]
        repeat = ''.join(f'Row {i:02d} confirmed file bytes: sample payload number {i}\n' for i in range(48))
        for body in ('Old unique beginning\n' + repeat + 'Old tail\n',
                     'New unique beginning\n' + repeat + 'New tail\n' + diagnostic):
            messages.append(dict(role='assistant', content='Check the files.'))
            prompt = 'Terminal output:\n' + body + '\nOuter reminder.'
            history.register(feedback=body, raw_feedback='UNTRUNCATED\n' + body, commands=[], timed_out=False)
            history.bind(messages, prompt)
            messages.append(dict(role='user', content=prompt))
        return history, messages, repeat

    def test_first_presentation_changes_only_exact_repeat_and_proves_cache_boundary(self):
        history, messages, repeat = self.repeat_history()
        request = dict(model='fixture', messages=messages)
        original = copy.deepcopy(messages)
        previous = self.native.json('/apply-template', dict(request, messages=messages[:-1], return_tokens=True))
        repeated = history.repeated_tail(messages)
        proposal, audit = self.planner().measure_repeated_tail(request, repeated, len(previous['tokens']))
        self.assertEqual(audit['status'], 'measured_repeated_tail')
        self.assertEqual(audit['prefill_tokens_avoided'], audit['saved_tokens'])
        self.assertGreater(audit['saved_tokens'], 1000)
        self.assertEqual(proposal['replacement_feedback'].splitlines()[0], 'New unique beginning')
        self.assertEqual(proposal['replacement_feedback'].splitlines()[-1], 'New tail')
        self.assertIn('Outer reminder.', proposal['replacement'])
        self.assertIn(repeat, proposal['archive']['body'])
        self.assertIn('UNTRUNCATED', proposal['archive']['body'])
        self.archive.save(proposal['archive'])
        first, last = audit['archive_lines']
        archive_lines = Path(proposal['archive']['local_path']).read_bytes().split(b'\n')
        self.assertEqual(b'\n'.join(archive_lines[first-1:last])+b'\n', repeat.encode())
        self.assertIn(f'Read lines {first}-{last}', proposal['replacement_feedback'])
        self.assertEqual(messages, original)
        self.assertEqual(self.native.inferences, 0)
        legacy = ToolCardPlanner(self.archive,NativeFixture(compact=False),self.profile,min_context_tokens=0)
        legacy_plan, legacy_audit = legacy.measure_repeated_tail(request,repeated,len(previous['tokens']))
        self.assertEqual(legacy_plan['replacement'],proposal['replacement'])
        self.assertEqual(legacy_audit['saved_tokens'],audit['saved_tokens'])
        self.assertEqual(legacy_audit['unchanged_prefix_tokens'],audit['unchanged_prefix_tokens'])
        # A cut crossing an already published native token frontier must fail.
        denied, status = self.planner().measure_repeated_tail(
            request, repeated, audit['unchanged_prefix_tokens'] + 1)
        self.assertIsNone(denied)
        self.assertEqual(status['status'], 'would_touch_cached_tokens')

    def test_repeat_requires_visible_owned_copy_and_protects_new_failures(self):
        history, messages, _ = self.repeat_history()
        self.assertIsNotNone(history.repeated_tail(messages))
        messages[2]['content'] = 'The old observation was edited.'
        self.assertIsNone(history.repeated_tail(messages))
        history, messages, _ = self.repeat_history('[exit code: 17]\n')
        self.assertIsNone(history.repeated_tail(messages))

    def test_repeat_can_use_a_visible_witness_beyond_the_last_three_feedbacks(self):
        history = OwnedToolHistory()
        messages = []
        repeat = ''.join(f'Row {i:02d} stable bytes for the runtime payload\n' for i in range(36))

        def bind(body):
            history.register(feedback=body, raw_feedback=body, commands=[], timed_out=False)
            prompt = 'Terminal output:\n' + body
            history.bind(messages, prompt)
            messages.append(dict(role='user', content=prompt))

        bind('Earlier header\n' + repeat)
        for index in range(10):
            bind(f'Unrelated tool observation {index}\n')
        bind('Current header\n' + repeat + 'New footer\n')
        result = history.repeated_tail(messages)
        self.assertEqual(result['witness']['ordinal'], 1)
        current = result['record']['feedback']
        self.assertEqual(current[result['begin']:result['end']], repeat)

    def command_history(self, *, executed=None, diagnostic='', prefix='> '):
        body = ''.join(f"result_{i:03d} = 'original unicode 信息 {i:03d} must remain retrievable'\n" for i in range(48))
        command = "cat > sample.py << 'PY'\n" + body + 'PY\n'
        content = json.dumps(dict(analysis='Write the file.', commands=[dict(keystrokes=command)]), ensure_ascii=False)
        messages = [dict(role='user', content='Preserve my task.'),
                    dict(role='assistant', content=content, reasoning_content='Retain this reasoning.')]
        feedback = 'New tool status header\n' + ''.join(prefix+line for line in body.splitlines(keepends=True))
        feedback += 'Unique new result: bytes written 3911\n' + diagnostic
        history = OwnedToolHistory()
        history.register(feedback=feedback, raw_feedback='RAW HEADER\n'+feedback,
            commands=[dict(keystrokes=command if executed is None else executed)], timed_out=False)
        prompt = 'Terminal observation:\n' + feedback + 'Wrapper remains.'
        history.bind(messages, prompt)
        messages.append(dict(role='user', content=prompt))
        return history, messages, body

    def test_executed_command_witness_preserves_unique_feedback_and_raw_recovery(self):
        history, messages, body = self.command_history()
        original = copy.deepcopy(messages)
        repeated = history.repeated_tail(messages)
        self.assertEqual(repeated['witness']['kind'], 'executed_command')
        self.assertEqual(repeated['witness']['message'], 1)
        self.assertEqual(repeated['witness']['command'], 0)
        request = dict(model='fixture', messages=messages)
        frontier = len(self.native.json('/apply-template', dict(request,messages=messages[:-1],return_tokens=True))['tokens'])
        proposal, audit = self.planner().measure_repeated_tail(request,repeated,frontier)
        self.assertEqual(audit['status'], 'measured_repeated_tail')
        self.assertEqual(audit['witness_kind'], 'executed_command')
        self.assertGreater(audit['saved_tokens'], 1000)
        self.assertIn('Unique new result: bytes written 3911',proposal['replacement'])
        self.assertIn('Wrapper remains.',proposal['replacement'])
        self.assertIn('command success is not inferred',proposal['replacement'])
        first,last = audit['archive_lines']
        self.archive.save(proposal['archive'])
        archive_lines = Path(proposal['archive']['local_path']).read_bytes().split(b'\n')
        span = b'\n'.join(archive_lines[first-1:last])+b'\n'
        self.assertEqual(span,b''.join(b'> '+line for line in body.encode().splitlines(keepends=True)))
        self.assertEqual(messages,original)
        self.assertEqual(self.native.inferences,0)
        denied,status = self.planner().measure_repeated_tail(request,repeated,audit['unchanged_prefix_tokens']+1)
        self.assertIsNone(denied)
        self.assertEqual(status['status'],'would_touch_cached_tokens')

    def test_command_witness_requires_executed_command_and_unchanged_assistant(self):
        history,messages,_ = self.command_history(executed='printf unrelated\n')
        self.assertIsNone(history.repeated_tail(messages))
        history,messages,_ = self.command_history()
        repeated = history.repeated_tail(messages)
        messages[1]['content'] += ' '
        self.assertIsNone(history.repeated_tail(messages))
        proposal,audit = self.planner().measure_repeated_tail(dict(messages=messages),repeated,1)
        self.assertIsNone(proposal)
        self.assertEqual(audit['status'],'unproved_append_boundary')
        history,messages,_ = self.command_history()
        messages[1]['role'] = 'user'
        self.assertIsNone(history.repeated_tail(messages))

    def test_command_witness_retains_diagnostics_and_does_not_normalize_arbitrary_prompts(self):
        for marker in ('error: current tool failed\n','[exit code: 7]\n'):
            history,messages,_ = self.command_history(diagnostic=marker)
            self.assertIsNone(history.repeated_tail(messages))
        history,messages,_ = self.command_history(prefix='PS2? ')
        self.assertIsNone(history.repeated_tail(messages))
        history,messages,_ = self.command_history(prefix='')
        self.assertEqual(history.repeated_tail(messages)['witness']['kind'],'executed_command')

    def test_repeat_span_cannot_be_extended_to_hide_new_status(self):
        history,messages,_ = self.command_history()
        repeated = history.repeated_tail(messages)
        repeated['end'] += len('Unique new result: bytes written 3911\n')
        repeated['lines'] += 1
        proposal,audit = self.planner().measure_repeated_tail(dict(messages=messages),repeated,1)
        self.assertIsNone(proposal)
        self.assertEqual(audit['status'],'unproved_repeat_span')

    def test_original_exact_searchable_and_quota_refuses_without_deleting(self):
        h,_=self.history()
        row=self.archive.describe(h.records[0])
        self.archive.save(row)
        raw=Path(row['local_path']).read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(),row['sha256'])
        self.assertIn(b'checksum 0679',raw)  # raw information not required to be in the card
        self.archive.max_bytes=self.archive.used_bytes
        changed=dict(h.records[0],raw_feedback='different original')
        with self.assertRaisesRegex(ValueError,'full'):self.archive.save(self.archive.describe(changed))
        self.assertEqual(Path(row['local_path']).read_bytes(),raw)
        Path(row['local_path']).write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError,'checksum'):self.archive.save(row)

    def test_inline_transfer_is_bounded_and_requires_the_exact_saved_original(self):
        history, _ = self.history()
        record = history.records[0]
        small = self.archive.describe(dict(record, raw_feedback='small raw source'))
        self.archive.save(small)
        self.assertIsNotNone(archive_transfer_command(small))
        Path(small['local_path']).write_bytes(b'corrupted locally')
        with self.assertRaisesRegex(ValueError, 'changed before transfer'):
            archive_transfer_command(small)
        large = self.archive.describe(dict(record, raw_feedback='x' * INLINE_ARCHIVE_MAX_BYTES))
        self.archive.save(large)
        self.assertIsNone(archive_transfer_command(large))

    def test_only_trusted_completed_old_feedback_and_wrappers_preserved(self):
        h,messages=self.history()
        records=h.candidates(messages,keep_recent=3)
        self.assertEqual([r['ordinal'] for r in records],[2,1])
        original=copy.deepcopy(messages)
        request=dict(model='fixture',messages=messages)
        plan,audit=self.planner().prepare(request,records)
        self.assertIsNotNone(plan)
        i=plan['record']['message']
        self.assertTrue(plan['replacement'].startswith('New Terminal Output:\n'))
        self.assertTrue(plan['replacement'].endswith('\nContinue the original task.'))
        self.assertIn('error: genuine failure preserved',plan['replacement'])
        self.assertIn('not a new instruction',plan['replacement'])
        self.assertEqual(messages,original)  # planning is not publication
        self.assertEqual(self.native.inferences,0)
        self.assertGreater(audit['attempts'][0]['saved_tokens'],1000)
        h.committed(plan['record'])
        self.assertFalse(h.candidates(messages,keep_recent=3))  # never cut left of published card

    def test_user_text_and_timeout_cannot_manufacture_provenance(self):
        h=OwnedToolHistory()
        messages=[dict(role='user',content='This is user text, not a tool observation.')]
        h.bind(messages,'anything')
        self.assertFalse(h.records)
        h.register(feedback='body',raw_feedback='body',commands=[],timed_out=True)
        h.bind(messages,'body')
        self.assertFalse(h.records)
        h.register(feedback='body',raw_feedback='body',commands=[],timed_out=False)
        h.bind(messages,'body body')
        self.assertFalse(h.records)

    def test_full_gpu_budget_no_attention_saving_is_not_fabricated(self):
        h,messages=self.history()
        plan,audit=self.planner(active_tokens=64).prepare(dict(messages=messages),h.candidates(messages))
        self.assertIsNone(plan)
        self.assertTrue(audit['attempts'])
        self.assertTrue(all(a['resident_tokens_removed']==0 and not a['admitted'] for a in audit['attempts']))
        self.assertFalse(list(Path(self.temp.name).glob('*.txt')))

    def test_longer_suffix_and_changed_history_revalidate_ready_result(self):
        h,messages=self.history()
        planner=self.planner()
        plan,_=planner.prepare(dict(messages=messages),h.candidates(messages))
        self.assertTrue(planner.recheck(plan,messages,.001)['admitted'])
        longer=messages+[dict(role='assistant',content='x'*100000)]
        self.assertFalse(planner.recheck(plan,longer,.001)['admitted'])
        changed=copy.deepcopy(messages);changed[0]['content']='changed prefix'
        self.assertEqual(planner.recheck(plan,changed,.001)['reason'],'history_changed')

    def test_missing_calibration_and_high_replay_cost_do_not_publish(self):
        h,messages=self.history()
        planner=ToolCardPlanner(self.archive,self.native,None,min_context_tokens=0)
        plan,audit=planner.prepare(dict(messages=messages),h.candidates(messages))
        self.assertIsNone(plan)
        self.assertEqual(audit['status'],'observed_without_cost_profile')
        self.assertEqual(audit['attempts'][0]['reason'],'uncalibrated_cost_profile')
        slow=CostProfile(1,10,evidence='slow fixture')
        planner.profile=slow
        plan,audit=planner.prepare(dict(messages=messages),h.candidates(messages))
        self.assertIsNone(plan)
        self.assertLess(audit['attempts'][0]['estimated_net_seconds'],0)

    def test_short_feedback_does_not_render_the_entire_history(self):
        h=OwnedToolHistory();messages=[dict(role='user',content='Original task')]
        for _ in range(5):
            h.register(feedback='Short complete tool result.',raw_feedback='Short complete tool result.',commands=[],timed_out=False)
            prompt='New Terminal Output:\nShort complete tool result.'
            h.bind(messages,prompt);messages.append(dict(role='user',content=prompt))
        class NoNativeWork:
            def json(self,*args,**kwargs):raise AssertionError('Small feedback caused a full-history template request')
        planner=ToolCardPlanner(self.archive,NoNativeWork(),self.profile,min_context_tokens=0)
        original=copy.deepcopy(messages)
        plan,audit=planner.prepare(dict(messages=messages),h.candidates(messages))
        self.assertIsNone(plan)
        self.assertEqual(audit['status'],'no_shorter_card')
        self.assertEqual(messages,original)
        self.assertFalse(list(Path(self.temp.name).glob('*.txt')))


@unittest.skipUnless(os.name == 'posix' and shutil.which('sh'), 'Requires the actual Linux task shell')
class ArchiveTransportTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.archive = OriginalArchive(self.temp.name, self.temp.name)
        self.remote_paths = []

    def tearDown(self):
        for path in self.remote_paths:
            path.unlink(missing_ok=True)
        self.temp.cleanup()

    def description(self):
        marker = str(Path(self.temp.name) / 'must-not-be-created')
        text = 'Unicode 原文 😀\r\n' + f'$(touch {marker})\n`touch {marker}`\n' + "'\";$HOME\n"
        row = self.archive.describe(dict(ordinal=1, commands=[], feedback=text,
            feedback_sha256=sha256(text), raw_feedback='RAW\x00\n' + text))
        self.archive.save(row)
        self.remote_paths.append(Path(row['remote_path']))
        return row, Path(marker)

    def execute(self, row):
        return subprocess.run(['sh', '-c', archive_transfer_command(row)], capture_output=True, text=True)

    def test_one_call_publishes_exact_bytes_without_executing_tool_text(self):
        row, marker = self.description()
        result = self.execute(row)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.split()[0], row['sha256'])
        self.assertEqual(Path(row['remote_path']).read_bytes(), Path(row['local_path']).read_bytes())
        self.assertFalse(marker.exists())
        self.assertEqual(Path(row['remote_path']).stat().st_mode & 0o077, 0)
        self.assertFalse(list(Path('/tmp').glob(Path(row['remote_path']).name + '.*')))
        self.assertEqual(self.execute(row).stdout, result.stdout)  # immutable retry

    def test_existing_corruption_is_not_overwritten_or_claimed_valid(self):
        row, _ = self.description()
        target = Path(row['remote_path'])
        target.write_bytes(b'wrong bytes already present')
        result = self.execute(row)
        self.assertNotEqual(result.stdout.split()[0], row['sha256'])
        self.assertEqual(target.read_bytes(), b'wrong bytes already present')

    def test_symlink_destination_is_rejected_without_writing_its_target(self):
        row, _ = self.description()
        sentinel = Path(self.temp.name) / 'unrelated'
        sentinel.write_bytes(b'untouched')
        Path(row['remote_path']).symlink_to(sentinel)
        result = self.execute(row)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(sentinel.read_bytes(), b'untouched')


class ProvenanceTest(unittest.TestCase):
    def test_reuse_exact_prefix_but_reprove_after_a_history_edit(self):
        native=NativeFixture();cache=NativeSpanCache();calls=[]
        def post(path,value):calls.append(value);return native.json(path,value)
        request=dict(messages=[dict(role='user',content='重复 old'),dict(role='user',content='重复 old')])
        prepared=post('/apply-template',dict(request,return_tokens=True))
        sites={(0,'content',0,6):'重复 old',(1,'content',0,6):'重复 old'}
        first=cache.locate(request,sites,prepared['prompt'],post)
        self.assertNotEqual(first[(0,'content',0,6)],first[(1,'content',0,6)])
        request['messages'].append(dict(role='assistant',content='new'))
        prepared=post('/apply-template',dict(request,return_tokens=True));count=len(calls)
        self.assertEqual(cache.locate(request,sites,prepared['prompt'],post),first)
        self.assertEqual(len(calls),count)
        request['messages'][0]['content']='prefix changed'
        prepared=post('/apply-template',dict(request,return_tokens=True));count=len(calls)
        second=cache.locate(request,{(1,'content',0,6):'重复 old'},prepared['prompt'],post)
        self.assertEqual(len(calls),count+1)
        begin,end=second[(1,'content',0,6)]
        self.assertEqual(prepared['prompt'].encode()[begin:end],'重复 old'.encode())

    def test_token_ranges_match_independent_scan_oracle(self):
        parts=[b'',b'a',b'bc',b'',b'def',b'g',b'']
        rows=[dict(id=i,piece=list(p)) for i,p in enumerate(parts)]
        raw=b''.join(parts);index=TokenByteIndex(rows,raw.decode())
        for begin in range(len(raw)):
            for end in range(begin+1,len(raw)+1):
                for intersect in (False,True):
                    expected=[];offset=0
                    for i,part in enumerate(parts):
                        stop=offset+len(part)
                        if part and ((begin<stop and offset<end) if intersect else begin<=offset<stop<=end):expected.append(i)
                        offset=stop
                    contiguous=bool(expected) and expected==list(range(expected[0],expected[-1]+1))
                    if contiguous:
                        self.assertEqual(index.range(raw[begin:end].decode(),(begin,end),intersect),
                                         (expected[0],expected[-1]+1))
                    else:
                        with self.assertRaises(ValueError):index.range(raw[begin:end].decode(),(begin,end),intersect)

    def test_cached_feedback_preserves_chronological_order(self):
        native=NativeFixture();cache=NativeSpanCache()
        request=dict(messages=[dict(role='user',content='first feedback'),dict(role='user',content='second feedback')])
        prepared=native.json('/apply-template',dict(request,return_tokens=True))
        terminal_feedback_ranges(request,[(1,'second feedback')],prepared,native.json,cache)
        plain=terminal_feedback_ranges(request,[(0,'first feedback'),(1,'second feedback')],prepared,native.json)
        cached=terminal_feedback_ranges(request,[(0,'first feedback'),(1,'second feedback')],prepared,native.json,cache)
        self.assertEqual(plain,cached)


class CalibrationTest(unittest.TestCase):
    @staticmethod
    def sample(elapsed_ns=1000):
        return dict(output_piece_sha256='same-pieces',decode_rounds=[dict(tokens=5,elapsed_ns=elapsed_ns) for _ in range(130)],
                    speculative=dict(backend='mtp',fallback_rounds=0,generation_rounds=130,drafted_tokens=520,output_tokens=650))

    def test_match_work_and_independent_slope(self):
        result=matched_full_rounds(self.sample(1000),self.sample(1640),64)
        self.assertTrue(result['eligible'])
        self.assertEqual(result['matched_rounds'],126)
        self.assertEqual(result['marginal_ns_per_context_token'],2)
        self.assertEqual(result['matched_output_spans'][0],(20,25))

    def test_text_identity_alone_or_unaligned_pages_cannot_match(self):
        left,right=self.sample(),self.sample()
        right['output_piece_sha256']='same-text-different-token-boundaries'
        self.assertEqual(matched_full_rounds(left,right,64)['reason'],'generated_token_pieces_differ')
        self.assertEqual(matched_full_rounds(left,left,65)['reason'],'page_offsets_differ')
        right=self.sample();right['speculative']['fallback_rounds']=1
        self.assertEqual(matched_full_rounds(left,right,64)['reason'],'unverified_round_work')

    def test_mismatched_generation_positions_are_excluded(self):
        left,right=self.sample(),self.sample()
        right['decode_rounds']=[dict(tokens=4,elapsed_ns=1000)]+right['decode_rounds'][:-1]+[dict(tokens=1,elapsed_ns=1000)]
        right['speculative'].update(generation_rounds=131,drafted_tokens=524)
        result=matched_full_rounds(left,right,64)
        self.assertFalse(result['eligible'])
        self.assertEqual(result['reason'],'insufficient_matched_work')


if __name__=='__main__':unittest.main()
