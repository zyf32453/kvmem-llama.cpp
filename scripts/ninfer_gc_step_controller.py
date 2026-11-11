"""Nonblocking completed-step GC producer. Normal responses carry summaries.

A CPU worker maps whole thinking pages to retained inline-summary pages.
These are candidate scopes, never deletion approvals. Native Mean-K, observed
importance and version/safe-boundary checks own the choice. No auxiliary LLM,
tool-feedback rewriting or final maintenance inference.
"""
import argparse
from collections import OrderedDict
import copy
import hashlib
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import threading
import time
import urllib.parse

from agent_phase_capture import handler
from ninfer_gc_agent_bridge import NativeClient
from ninfer_gc_step import (
    attach_evidence, duplicate_feedback_page_sets, duplicate_feedback_steps, terminal_feedback_ranges, token_digest,
    NativeSpanCache,
)
from ninfer_gc_inline_summary import PROMPT_TEMPLATE_OPTIONS, response_format, summary_span


def digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def response_identity(content, thinking):
    return digest(json.dumps([content, thinking], ensure_ascii=False, separators=(',', ':')))


class StepSummaryController:
    def __init__(self, native, session, out, mode='reclaim', max_records=64, max_spans=4,
                 native_audit_log=None, duplicate_feedback=False):
        if mode not in ('off', 'observe', 'native-observe', 'reclaim') or not session or not 1 <= max_records <= 64 or not 1 <= max_spans <= 4:
            raise ValueError('Invalid bounded step controller configuration')
        self.native, self.session, self.out = native, session, Path(out)
        self.mode, self.max_records, self.max_spans = mode, max_records, max_spans
        self.native_audit_log = Path(native_audit_log) if native_audit_log else None
        self.out.mkdir(parents=True, exist_ok=True)
        self.condition = threading.Condition(threading.RLock())
        self.receipts, self.records, self.pending_events = OrderedDict(), OrderedDict(), OrderedDict()
        self.serial = 0
        self.history_epoch = 0
        self.history_edits = OrderedDict()
        self.request_epochs=OrderedDict()
        self.request_serial = 0
        self.ready = None
        self.work = None
        self.finalized = False
        self.profile = None
        self.span_cache=NativeSpanCache()
        self.duplicate_feedback = duplicate_feedback
        self.feedback_source_pages, self.feedback_witness_pages = set(), set()
        self.counts = dict(main_responses=0, auxiliary_model_calls=0, prepared_scopes=0,
                           submitted_scopes=0, native_acknowledged=0, stale_ready_batches=0,
                           worker_failures=0, prepare_max_seconds=0.0)
        self.worker = None
        if mode != 'off':
            self.worker = threading.Thread(target=self._worker, name='step-gc-producer', daemon=True)
            self.worker.start()

    def save(self, name, value):
        path = self.out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        temporary.replace(path)

    def _matches(self, record, messages):
        index = record['source_message']
        if index + 1 >= len(messages): return False
        source, feedback = messages[index:index + 2]
        return (source.get('role') == 'assistant' and source.get('content') == record['content']
                and source.get('reasoning_content') == record['thinking']
                and feedback.get('role') == 'user' and isinstance(feedback.get('content'), str)
                and (record['feedback'] in feedback['content']
                     or (record.get('presented_feedback') and record['presented_feedback'] in feedback['content'])))

    def _register_locked(self, event, receipt):
        identity = 'agent-step-' + digest(receipt['id'])[:20]
        if identity in self.records:
            if self.records[identity]['feedback_sha256'] != digest(event['feedback']):
                raise ValueError('Completed response cannot acquire different tool feedback')
            return identity
        self.serial += 1
        record = dict(step_id=identity, ordinal=self.serial, source_message=receipt['source_message'],
            source_capture=receipt['capture'], content=receipt['content'], thinking=receipt['thinking'],
            feedback=event['feedback'], feedback_sha256=digest(event['feedback']),
            tool_feedback_unix_ns=event.get('tool_feedback_unix_ns'),
            tool_timed_out=bool(event.get('tool_timeout')), request_serial=receipt.get('request_serial'),
            preceding_frontier=receipt.get('preceding_frontier'),
            status='incomplete_tools' if event.get('tool_timeout') else 'awaiting_inline_summary')
        self.records[identity] = record
        while len(self.records) > self.max_records: self.records.popitem(last=False)
        return identity

    def complete_step(self, event):
        if event.get('session') != self.session or event.get('phase') != 'tool_feedback_complete':
            raise ValueError('Trusted tool-feedback completion event required')
        if self.mode == 'off': return dict(status='gc_disabled')
        if not all(isinstance(event.get(k), str) for k in ('content', 'thinking', 'feedback')):
            raise ValueError('Invalid source/feedback')
        key = response_identity(event['content'], event['thinking'])
        with self.condition:
            if key not in self.receipts:
                # Body delivery can race after_response; register without waiting.
                self.pending_events[key] = event
                while len(self.pending_events) > 8: self.pending_events.popitem(last=False)
                return dict(status='receipt_pending')
            identity = self._register_locked(event, self.receipts[key])
            record = self.records[identity]
            result = dict(status='registered', step_id=identity)
            if not record['tool_timed_out'] and record['request_serial'] == self.request_serial:
                result.update(preceding_frontier=record['preceding_frontier'],
                              feedback_message=record['source_message']+1)
            return result

    def first_presentation(self, event):
        """Register the presentation of a feedback message before it is sent."""
        text = event.get('feedback')
        frontier = event.get('preceding_frontier')
        if (event.get('session') != self.session or not isinstance(text, str)
                or not 0 < len(text.encode('utf-8')) <= 16384
                or type(frontier) is not int or frontier <= 0):
            raise ValueError('Owned first presentation required')
        with self.condition:
            record = self.records.get(event['step_id'])
            if (record is None or record['status'] != 'awaiting_inline_summary'
                    or record['request_serial'] != self.request_serial
                    or record['source_message']+1 != event.get('message')
                    or record['preceding_frontier'] != frontier
                    or record['feedback_sha256'] != event.get('original_sha256')):
                return dict(status='first_presentation_stale')
            existing = record.get('presented_feedback')
            if existing is not None and existing != text:
                return dict(status='first_presentation_stale')
            # A lost acknowledgement can still make the Agent send the
            # original. _matches accepts either authenticated presentation.
            record['presented_feedback'] = text
            return dict(status='first_presentation_registered')

    def history_edit(self,event):
        if (event.get('session')!=self.session or not isinstance(event.get('message'),int) or
                event['message']<1 or not isinstance(event.get('edit_id'),str) or len(event['edit_id'])!=64):
            raise ValueError('Owned history edit required')
        with self.condition:
            if event['edit_id'] in self.history_edits:return self.history_edits[event['edit_id']]
            self.history_epoch+=1
            self.ready=self.work=None
            self.receipts.clear();self.pending_events.clear()
            for record in self.records.values():
                if max(record['source_message']+1,record.get('witness_message',0))>=event['message']:
                    record.update(status='source_rewritten',reason='archived_tool_history_edit')
            self.feedback_source_pages.clear();self.feedback_witness_pages.clear()
            value=dict(status='history_proofs_invalidated',history_epoch=self.history_epoch)
            self.history_edits[event['edit_id']]=value
            while len(self.history_edits)>64:self.history_edits.popitem(last=False)
            self.condition.notify_all()
            return value

    def note_response(self, request, response, folder, status):
        with self.condition:
            epoch, serial = self.request_epochs.pop(id(request), (self.history_epoch, None))
            if epoch!=self.history_epoch:return
        if status != 200 or response.get('error') or request.get('prompt_cache_key') != self.session:
            return
        if self.mode == 'off':
            with self.condition: self.counts['main_responses'] += 1
            return
        message = (response.get('choices') or [{}])[0].get('message') or {}
        content, thinking = message.get('content') or '', message.get('reasoning_content') or ''
        if not isinstance(content, str) or not isinstance(thinking, str): return
        key = response_identity(content, thinking)
        try: summary, begin, end = summary_span(content)
        except (ValueError, TypeError, IndexError): summary = ''
        with self.condition:
            if epoch!=self.history_epoch:return
            self.counts['main_responses'] += 1
            native = response.get('kvmem', {}).get('historical_gc', {})
            for row in native.get('scopes', []) if native.get('available') else []:
                record = self.records.get(row.get('id'))
                if record is not None:
                    if record['status'] == 'submitted': self.counts['native_acknowledged'] += 1
                    record['status'] = 'native_' + row['state']
                    record['native'] = row
            messages = request['messages']
            last_assistant = max((i for i, m in enumerate(messages) if m.get('role') == 'assistant'), default=-1)
            previous = next((r for r in reversed(list(self.records.values()))
                if r['source_message'] == last_assistant and self._matches(r, messages)), None)
            if summary.strip() and previous and previous['status'] == 'awaiting_inline_summary':
                previous.update(status='candidate', summary=summary, witness_content=content,
                    witness_message=len(messages), witness_begin=begin, witness_end=end)
            usage = response.get('usage') or {}
            timing = response.get('timings') or {}
            prompt, generated, total = (usage.get(k) for k in ('prompt_tokens', 'completion_tokens', 'total_tokens'))
            cached, evaluated, predicted = (timing.get(k) for k in ('cache_n', 'prompt_n', 'predicted_n'))
            # The native retained ledger ends no later than the logical
            # prompt plus accepted generation. Require both native counters
            # to agree before pricing a first-presentation edit as replay-free.
            frontier = (total if all(type(n) is int and n >= 0 for n in
                                     (prompt, generated, total, cached, evaluated, predicted))
                        and total == prompt+generated and prompt == cached+evaluated
                        and generated == predicted else None)
            receipt = dict(id=response.get('id') or str(folder), content=content, thinking=thinking,
                source_message=len(messages), capture=str(folder), request_serial=serial, preceding_frontier=frontier)
            self.receipts[key] = receipt
            while len(self.receipts) > 8: self.receipts.popitem(last=False)
            if key in self.pending_events: self._register_locked(self.pending_events.pop(key), receipt)
            self.condition.notify_all()

    def prepare(self, request, headers=None):
        started = time.monotonic()
        if request.get('prompt_cache_key', self.session) != self.session:
            return request, dict(status='foreign_session_preserved')
        if 'kvmem_gc' in request: return request, dict(status='client_evidence_preserved')
        bound = dict(request, prompt_cache_key=self.session, response_format=response_format())
        with self.condition:
            self.request_serial += 1
            self.request_epochs[id(bound)] = (self.history_epoch, self.request_serial)
            while len(self.request_epochs)>64:self.request_epochs.popitem(last=False)
        if self.mode == 'off':
            return bound, dict(status='gc_disabled', attached=0, auxiliary_model_calls=0)
        with self.condition:
            audit = dict(status='worker_pending', attached=0, auxiliary_model_calls=0)
            ready = self.ready
            if ready:
                n = len(ready['messages'])
                same = bound['messages'][:n] == ready['messages'] and self._template_key(bound) == ready['template_key']
                if not same:
                    self.counts['stale_ready_batches'] += 1
                    for identity in ready['ids']:
                        if identity in self.records: self.records[identity]['status'] = 'source_rewritten'
                elif self.mode in ('reclaim', 'native-observe'):
                    bound['kvmem_gc'] = ready['proof']
                    for step in ready['proof']['steps']:
                        if step['kind'] == 'duplicate_tool_feedback':
                            source, witness = duplicate_feedback_page_sets(step)
                            self.feedback_source_pages.update(source)
                            self.feedback_witness_pages.update(witness)
                    for identity in ready['ids']:
                        if identity in self.records: self.records[identity]['status'] = 'submitted'
                    self.counts['submitted_scopes'] += len(ready['ids'])
                    audit.update(status='ready_scopes_attached', attached=len(ready['ids']))
                else:
                    for identity in ready['ids']:
                        if identity in self.records: self.records[identity]['status'] = 'observed_no_delete'
                    audit.update(status='observe_only', attached=0)
                self.ready = None
            self.work = bound  # latest snapshot only; bounded backpressure
            self.condition.notify_all()
            elapsed = time.monotonic() - started
            self.counts['prepare_max_seconds'] = max(self.counts['prepare_max_seconds'], elapsed)
        return bound, audit

    @staticmethod
    def _template_key(request):
        return {k: request.get(k) for k in ('model', 'response_format', *PROMPT_TEMPLATE_OPTIONS)}

    def _build(self, request, records, feedback=(), source_pages=(), witness_pages=()):
        if self.profile is None:
            props = self.native.json('/props?' + urllib.parse.urlencode(dict(model=request.get('model', ''))))
            self.profile = props.get('kvmem_gc', {})
            self.sink_tokens = props.get('kvmem', {}).get('sink_tokens', 64)
        allowed = ('step_redundancy', 'off') if self.mode == 'observe' else ('step_redundancy',)
        if self.profile.get('policy') not in allowed or self.profile.get('block_tokens') != 64:
            raise ValueError('Native step_redundancy / 64-token profile required')
        if self.mode in ('reclaim', 'native-observe') and bool(self.profile.get('observe_only')) != (self.mode == 'native-observe'):
            raise ValueError('Controller mode and native step observation must agree')
        cache, prepared = {}, None
        def post(path, value):
            nonlocal prepared
            if path != '/apply-template': raise AssertionError('GC producer cannot request inference')
            key = json.dumps(value, ensure_ascii=False, sort_keys=True)
            if key not in cache: cache[key] = self.native.json(path, value)
            if value.get('return_tokens'): prepared = cache[key]
            return cache[key]
        batch, ids, skipped = None, [], []
        messages = request['messages']
        candidates = []
        for record in records:
            identity = record['step_id']
            if record['source_message']+1 >= len(messages) or record['witness_message'] >= len(messages):
                continue  # Snapshot raced the response; await the next actual request.
            if not self._matches(record, messages):
                skipped.append(dict(id=identity,reason='owned_source_rewritten')); continue
            witness = messages[record['witness_message']]
            if witness.get('role') != 'assistant' or witness.get('content') != record['witness_content']:
                skipped.append(dict(id=identity,reason='owned_witness_rewritten')); continue
            # Native chat templates trim field-edge whitespace. Prove the
            # complete non-whitespace interior, never just a sampled tail.
            original = record['thinking']
            source = original.strip()
            begin = len(original) - len(original.lstrip())
            candidates.append(dict(id=identity, kind='thinking', source_text=source,
                result_text=record['witness_content'][record['witness_begin']:record['witness_end']],
                observation_id=1, importance=1.0, cold=False, complete=True,
                _source_message_span=dict(message=record['source_message'], field='reasoning_content',
                                          begin=begin, end=begin+len(source)),
                _result_message_span=dict(message=record['witness_message'], field='content',
                    begin=record['witness_begin'], end=record['witness_end'], kind='inline_step_summary')))
        mapped = {}
        if len(candidates) > 1:
            try:
                proof = attach_evidence('', request, candidates, post=post,span_cache=self.span_cache)['kvmem_gc']
                mapped = {row['id']: row for row in proof['steps']}
            except (OSError, ValueError, TypeError, KeyError):
                # Isolate an incomplete witness or malformed scope from the other candidates.
                pass
        for step in candidates:
            identity = step['id']
            try:
                row = mapped.get(identity)
                if row is None:
                    row = attach_evidence('', request, [step], post=post,span_cache=self.span_cache)['kvmem_gc']['steps'][0]
                if row['result_end'] > len(prepared['tokens'])-64:
                    skipped.append(dict(id=identity,reason='awaiting_sealed_witness_tail',retryable=True)); continue
                thinking_pages = (row['source_end'] - row['source_begin']) // 64
                analysis_pages = (row['result_end'] - row['result_begin']) // 64
                # Analysis already exists in the unmodified Agent protocol;
                # there are zero newly generated summary pages to amortize.
                if not 1 <= thinking_pages <= 128 or analysis_pages > 8:
                    skipped.append(dict(id=identity, reason='no_bounded_net_gain')); continue
                if batch is None: batch = dict(steps=[])
                batch['steps'].append(row); ids.append(identity)
            except (OSError, ValueError, TypeError, KeyError) as error:
                skipped.append(dict(id=identity, reason=str(error),
                    retryable='Result witness is not a complete accepted block yet' in str(error)))
        owned_feedback = [(r['source_message'] + 1, r['feedback']) for r in feedback if self._matches(r, messages)]
        if len(owned_feedback) > 1:
            try:
                if prepared is None:
                    prepared = post('/apply-template', dict(request, return_tokens=True))
                ranges = terminal_feedback_ranges(request, owned_feedback, prepared, post,self.span_cache)
                steps = duplicate_feedback_steps([r['id'] for r in prepared['tokens']], ranges,
                    sink_tokens=self.sink_tokens, source_pages=source_pages, witness_pages=witness_pages)
                if steps:
                    if batch is None: batch = dict(steps=[])
                    batch['steps'].extend(steps)
                    ids.extend(step['id'] for step in steps)
            except (OSError, ValueError, TypeError, KeyError) as error:
                skipped.append(dict(id='duplicate_tool_feedback', reason=str(error)))
        if batch:
            # Seal an interior prefix, excluding synthetic generation framing.
            anchor = max(max(s['source_end'], s['result_end']) for s in batch['steps'])
            anchor = (anchor + 63) // 64 * 64
            batch.update(prefix_tokens=anchor, prefix_sha256=token_digest([r['id'] for r in prepared['tokens'][:anchor]]))
            # The outward witness page can include adjacent framing/feedback.
            # Guard the entire actual input snapshot, not only the summary's message.
            prefix_messages = copy.deepcopy(messages)
            ready = dict(proof=batch, ids=ids, messages=prefix_messages, template_key=copy.deepcopy(self._template_key(request)))
        else: ready = None
        return ready, skipped

    def _worker(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.finalized or self.work is not None)
                if self.finalized: return
                request, self.work = self.work, None
                epoch=self.history_epoch
                if self.ready is not None: continue
                records = [dict(r) for r in self.records.values() if r['status'] == 'candidate'][:self.max_spans]
                feedback = [dict(r) for r in self.records.values() if not r['tool_timed_out']] if self.duplicate_feedback else []
                source_pages, witness_pages = self.feedback_source_pages.copy(), self.feedback_witness_pages.copy()
            if not records and len(feedback) < 2: continue
            started = time.monotonic()
            try:
                ready, skipped = self._build(request, records, feedback, source_pages, witness_pages)
                with self.condition:
                    if self.finalized: return
                    if epoch!=self.history_epoch:
                        self.counts['stale_ready_batches']+=1
                        continue
                    if ready:
                        self.ready = ready
                        self.counts['prepared_scopes'] += len(ready['ids'])
                        for identity in ready['ids']:
                            if identity in self.records: self.records[identity]['status'] = 'ready'
                    for row in skipped:
                        if row['id'] in self.records:
                            self.records[row['id']].update(status='candidate' if row.get('retryable') else 'retained', reason=row['reason'])
                    self.condition.notify_all()
                self.save('worker-%05d.json' % time.time_ns(), dict(elapsed_seconds=time.monotonic()-started,
                    scope_ids=ready['ids'] if ready else [], proof=ready['proof'] if ready else None, skipped=skipped))
            except (OSError, ValueError, TypeError, KeyError) as error:
                with self.condition: self.counts['worker_failures'] += 1
                self.save('worker-error-%05d.json' % time.time_ns(), dict(error=repr(error)))

    def finalize(self):
        with self.condition:
            self.finalized = True
            self.work = None
            value = dict(status='closed_without_final_inference', counters=dict(self.counts),
                records=[dict(r) for r in self.records.values()], final_step_retained=True,
                note='Last step has no later ordinary summary; no maintenance answer was generated')
            self.condition.notify_all()
        self.save('completion.json', value)
        if self.native_audit_log and self.native_audit_log.exists():
            (self.out/'native-during-task.log').write_bytes(self.native_audit_log.read_bytes())
        return dict(status=value['status'], counters=value['counters'])


def controller_handler(controller, upstream, out, token):
    base = handler(upstream, out, before_request=controller.prepare, collect_logprobs=False,
                   after_response=controller.note_response)
    class Handler(base):
        def do_POST(self):
            if self.path not in ('/kvmem/step-end', '/kvmem/finalize', '/kvmem/history-edit',
                                 '/kvmem/first-presentation'):
                return super().do_POST()
            if not secrets.compare_digest(self.headers.get('X-KVMem-Step-Key', ''), token):
                self.send_error(403, 'Owned harness callback required'); return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 1024*1024: raise ValueError('Invalid callback size')
                value = json.loads(self.rfile.read(size))
                callbacks = {'/kvmem/step-end': controller.complete_step,
                             '/kvmem/history-edit': controller.history_edit,
                             '/kvmem/first-presentation': controller.first_presentation}
                result = controller.finalize() if self.path == '/kvmem/finalize' else callbacks[self.path](value)
                raw = json.dumps(result, ensure_ascii=False).encode()
                self.send_response(200); self.send_header('Content-Type', 'application/json'); self.end_headers(); self.wfile.write(raw)
            except (OSError, ValueError, TypeError, KeyError) as error:
                raw = json.dumps(dict(error=repr(error))).encode()
                self.send_response(422); self.end_headers(); self.wfile.write(raw)
    return Handler


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream', required=True)
    p.add_argument('--native-endpoint', required=True, help='CPU template/props only')
    p.add_argument('--session-key', required=True)
    p.add_argument('--control-token-file', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--port', type=int, default=18093)
    p.add_argument('--mode', choices=('off', 'observe', 'native-observe', 'reclaim'), default='reclaim')
    p.add_argument('--duplicate-feedback', action='store_true', help='Retire exact repeats of owned terminal feedback')
    p.add_argument('--native-audit-log', type=Path)
    args = p.parse_args()
    for url in (args.upstream, args.native_endpoint):
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme != 'http' or parsed.hostname not in ('localhost', '127.0.0.1', '::1'):
            p.error('Use owned loopback endpoints')
    token = args.control_token_file.read_text().strip()
    if len(token) < 32: p.error('Use a fresh callback credential')
    controller = StepSummaryController(NativeClient(args.native_endpoint, timeout=60), args.session_key,
        args.out/'lifecycle', args.mode, native_audit_log=args.native_audit_log, duplicate_feedback=args.duplicate_feedback)
    print('Nonblocking inline-summary controller ready; mode='+args.mode, flush=True)
    ThreadingHTTPServer(('127.0.0.1', args.port),
        controller_handler(controller, args.upstream, args.out/'main', token)).serve_forever()


if __name__ == '__main__': main()
