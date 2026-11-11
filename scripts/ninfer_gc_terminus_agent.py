"""Harbor Terminus-2: normal response summary, trusted post-tool registration.

This subclass uses existing methods of the pinned Harbor build;
it does not patch Harbor's installed source, synthesize commands or inspect the
verifier. Base harness context compression remains disabled.
"""
import asyncio
import importlib.metadata
import inspect
import json
from pathlib import Path
import shlex
import time
from typing import Literal
import urllib.request
import urllib.parse

from pydantic import Field
from harbor.agents.terminus_2.terminus_2 import Terminus2,Terminus2Options
from ninfer_gc_agent_bridge import NativeClient
from ninfer_gc_inline_summary import PROMPT_TEMPLATE_OPTIONS, response_format
from ninfer_gc_tool_cards import (CostProfile, OriginalArchive, OwnedToolHistory, ToolCardPlanner,
                                 archive_transfer_command, INLINE_ARCHIVE_MAX_BYTES)


CARD_FOREGROUND_SECONDS = 0.35


class ThinkingSummaryTerminus2Options(Terminus2Options):
    gc_controller_url:str=Field(description='Owned loopback post-tool lifecycle controller')
    gc_session_key:str=Field(min_length=1,max_length=256,description='One isolated Agent attempt session')
    gc_control_token_file:str=Field(description='Owned local callback credential file')
    gc_tool_cards:Literal['off','observe','apply']='off'
    gc_cards_cost_profile:str|None=None
    gc_cards_active_tokens:int=Field(default=98304,ge=64)
    gc_cards_min_context_tokens:int=Field(default=20000,ge=0)
    gc_cards_min_saved_tokens:int=Field(default=1024,ge=1)
    gc_cards_keep_recent:int=Field(default=3,ge=1)


class ThinkingSummaryTerminus2(Terminus2):
    options_model=ThinkingSummaryTerminus2Options
    options:ThinkingSummaryTerminus2Options

    def __init__(self,*args,gc_controller_url,gc_session_key,gc_control_token_file,**kwargs):
        self._gc_current=None;self._gc_events=[]
        if importlib.metadata.version('harbor')!='0.25.0':
            raise ValueError('Revalidate the step adapter against this Harbor version first')
        expected={'_handle_llm_interaction':['self','chat','prompt','original_instruction','session'],
            '_execute_commands':['self','commands','session'],
            '_limit_output_length':['self','output','max_bytes']}
        for method,names in expected.items():
            if list(inspect.signature(getattr(Terminus2,method)).parameters)!=names:
                raise ValueError('Harbor step method contract changed: '+method)
        if kwargs.get('enable_summarize',False):
            raise ValueError('Harness compaction must stay disabled for this experiment')
        kwargs['enable_summarize']=False
        if not kwargs.get('interleaved_thinking',False):
            raise ValueError('Interleaved original thinking must remain in actual model history')
        # Harbor validates options before construction as well as inside the
        # BaseAgent constructor. Declare/pass the lifecycle fields in both.
        super().__init__(*args,gc_controller_url=gc_controller_url,gc_session_key=gc_session_key,
            gc_control_token_file=str(gc_control_token_file),**kwargs)
        self._gc_url=gc_controller_url.rstrip('/');self._gc_session=gc_session_key
        parsed=urllib.parse.urlsplit(self._gc_url)
        if parsed.scheme!='http' or parsed.hostname not in ('127.0.0.1','localhost','::1'):
            raise ValueError('Use an owned loopback lifecycle controller')
        self._gc_token=Path(gc_control_token_file).read_text().strip()
        self._gc_card_job=None;self._gc_repeat_job=None;self._gc_cards_tried=set();self._gc_raw_output=[]
        self._gc_capturing_output=False;self._gc_raw_bytes=0;self._gc_raw_complete=True
        self._gc_owned_tools=OwnedToolHistory()
        self._gc_card_planner=None
        if self.options.gc_tool_cards!='off':
            profile=(CostProfile(**json.loads(Path(self.options.gc_cards_cost_profile).read_text()))
                     if self.options.gc_cards_cost_profile else None)
            if self.options.gc_tool_cards=='apply' and profile is None:
                raise ValueError('Applying tool cards requires a measured cost profile')
            self._gc_card_planner=ToolCardPlanner(
                OriginalArchive(self.logs_dir/'tool-originals',gc_session_key),NativeClient(self._gc_url,timeout=15),profile,
                active_tokens=self.options.gc_cards_active_tokens,
                min_context_tokens=self.options.gc_cards_min_context_tokens,
                min_saved_tokens=self.options.gc_cards_min_saved_tokens)

    def _gc_card_audit(self,value):
        try:
            self.logs_dir.mkdir(parents=True,exist_ok=True)
            with (self.logs_dir/'tool-cards.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(dict(unix_ns=time.time_ns(),**value),ensure_ascii=False)+'\n')
            return True
        except OSError as error:
            self.logger.warning('Tool-card audit unavailable; retain uncommitted originals: %s',error)
            return False

    async def _gc_stage_card(self,request,candidates,session):
        started=time.monotonic()
        try:
            proposal,audit=await asyncio.to_thread(self._gc_card_planner.prepare,request,candidates)
            # A small context is a temporary gate, not a rejected original.
            # Also do not consume candidates beyond the first accepted edit:
            # prepare() has not evaluated those yet.
            considered={row['message'] for row in audit.get('attempts',[])}
            self._gc_cards_tried.update(r['ordinal'] for r in candidates if r['message'] in considered)
            if not self._gc_card_audit(audit):return None
            if proposal is None or self.options.gc_tool_cards=='observe':
                return None
            await self._gc_upload_archive(session,proposal['archive'])
            proposal['preparation_seconds']=time.monotonic()-started
            return proposal
        except Exception as error:
            # Environment upload/exec providers use several exception types.
            # Failure of optional compaction must not abort the actual Agent.
            self._gc_card_audit(dict(status='retained_after_preparation_failure',error=repr(error)))
            return None

    async def _gc_upload_archive(self, session, archive):
        started = time.monotonic()
        environment = session.environment
        command = await asyncio.to_thread(archive_transfer_command, archive)
        if command is not None:
            checked = await environment.exec(command, timeout_sec=5)
            if checked.return_code != 127:
                self._gc_check_archive_result(checked, archive)
                self._gc_card_audit(dict(status='archive_transferred', transport='inline_verified_exec',
                    archive_sha256=archive['sha256'], bytes=archive['bytes'],
                    elapsed_seconds=time.monotonic()-started))
                return
        await asyncio.wait_for(environment.upload_file(archive['local_path'], archive['remote_path']), timeout=15)
        await self._gc_verify_archive(session, archive)
        self._gc_card_audit(dict(status='archive_transferred', transport='upload_then_verify',
            archive_sha256=archive['sha256'], bytes=archive['bytes'],
            elapsed_seconds=time.monotonic()-started))

    async def _gc_verify_archive(self,session,archive):
        if session is None:raise ValueError('No task environment for original readback')
        checked=await session.environment.exec('sha256sum '+shlex.quote(archive['remote_path']),timeout_sec=5)
        self._gc_check_archive_result(checked, archive)

    @staticmethod
    def _gc_check_archive_result(checked, archive):
        fields=checked.stdout.split()
        if checked.return_code!=0 or not fields or fields[0]!=archive['sha256']:
            raise ValueError('Task cannot read the exact archived original')

    async def _gc_stage_repeat(self, request, repeat, receipt, session, started, deadline):
        archive_job = None
        staged_archive = None

        async def archive_original(archive):
            await asyncio.to_thread(self._gc_card_planner.archive.save, archive)
            if time.monotonic() >= deadline:
                raise TimeoutError('Original archive write reached the presentation deadline')
            await self._gc_upload_archive(session, archive)

        try:
            # A complete owned repeat can be copied while native proves the
            # proposed token edit. Copying does not authorize an edit: every
            # native, registration and cost check below is still mandatory.
            # Limit speculation to small originals in an already-large context.
            planner = self._gc_card_planner
            if (self.options.gc_tool_cards == 'apply'
                    and planner.min_context_tokens <= receipt['preceding_frontier'] <= planner.active_tokens):
                description = planner.archive.describe(repeat['record'])
                if description['bytes'] <= INLINE_ARCHIVE_MAX_BYTES:
                    staged_archive = description
                    archive_job = asyncio.create_task(archive_original(description))
            plan, audit = await asyncio.to_thread(self._gc_card_planner.measure_repeated_tail,
                                                 request, repeat, receipt['preceding_frontier'])
            if not self._gc_card_audit(dict(audit, route='first_presentation')):
                return None
            if plan is None or self.options.gc_tool_cards == 'observe':
                return None
            plan['created'] = started
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self._gc_card_planner.price_repeated_tail(plan)['admitted']:
                return None

            if staged_archive is not None and staged_archive != plan['archive']:
                raise ValueError('Original changed while the native proof was prepared')
            if archive_job is None:
                archive_job = asyncio.create_task(archive_original(plan['archive']))

            record = repeat['record']
            event = dict(session=self._gc_session, step_id=receipt['step_id'], message=record['message'],
                         preceding_frontier=receipt['preceding_frontier'], original_sha256=record['feedback_sha256'],
                         feedback=plan['replacement_feedback'])
            # Registration authorizes an alternate spelling of this observation.
            # The controller accepts the original if publication does not happen.
            registration, archive_result = await asyncio.gather(
                asyncio.to_thread(self._gc_post, '/kvmem/first-presentation', event, remaining),
                archive_job, return_exceptions=True)
            for outcome in (registration, archive_result):
                if isinstance(outcome, BaseException):
                    raise outcome
            if registration.get('status') != 'first_presentation_registered':
                raise ValueError('Controller did not register the planned tool feedback')
            return plan
        except Exception as error:
            self._gc_card_audit(dict(status='retained_after_preparation_failure',
                                     route='first_presentation', error=repr(error)))
            return None
        finally:
            # A rejected proof may leave a bounded copy in progress. The outer
            # foreground deadline still returns original text; drain this task
            # before accepting another preparation, including its exceptions.
            if archive_job is not None:
                await asyncio.gather(archive_job, return_exceptions=True)

    async def _gc_present_repeat(self, request, session, deadline):
        started = time.monotonic()
        repeat = self._gc_owned_tools.repeated_tail(request['messages'])
        if repeat is None:
            return None
        record = repeat['record']
        receipt = record.get('receipt', {})
        if (receipt.get('feedback_message') != record['message']
                or not isinstance(receipt.get('preceding_frontier'), int)):
            self._gc_card_audit(dict(status='first_presentation_missing_receipt', message=record['message'],
                                     witness_message=repeat['witness']['message']))
            return None
        self._gc_repeat_job = asyncio.create_task(
            self._gc_stage_repeat(request, repeat, receipt, session, started, deadline))
        if self.options.gc_tool_cards == 'observe':
            return None
        done, _ = await asyncio.wait({self._gc_repeat_job}, timeout=max(0, deadline-time.monotonic()))
        if not done:
            self._gc_card_audit(dict(status='first_presentation_deadline', message=record['message'],
                                     waited_seconds=time.monotonic()-started))
            return None
        plan = self._gc_repeat_job.result()
        self._gc_repeat_job = None
        if plan is None:
            return None
        decision = self._gc_card_planner.price_repeated_tail(plan)
        if time.monotonic() >= deadline or not decision['admitted']:
            self._gc_card_audit(dict(status='first_presentation_retained', message=record['message'], **decision))
            return None
        if not self._gc_card_audit(dict(status='first_presentation_committed', message=record['message'],
                                        archive_sha256=plan['archive']['sha256'], **decision)):
            return None
        self._gc_owned_tools.committed(record)
        return plan['replacement']

    async def _gc_commit_ready_card(self, chat, prompt, proposal, session, deadline):
        messages = [*chat.messages, dict(role='user', content=prompt)]
        planner = self._gc_card_planner
        prepared_seconds = proposal['preparation_seconds']
        record = proposal['record']
        checksum = proposal['archive']['sha256']
        started = time.monotonic()
        decision = planner.recheck(proposal, messages, prepared_seconds)
        if decision['admitted']:
            # Spend at most the unused gain margin on foreground publication.
            budget = min(deadline-started,
                         decision['estimated_net_seconds']-planner.profile.minimum_gain_seconds)
            failure = {}
            try:
                remaining = budget - (time.monotonic()-started)
                if remaining <= 0:
                    raise TimeoutError('No remaining card publication budget')
                async with asyncio.timeout(remaining):
                    # A tool may have removed the staged file after upload.
                    await self._gc_verify_archive(session, proposal['archive'])
                    remaining = budget - (time.monotonic()-started)
                    if remaining <= 0:
                        raise TimeoutError('Archive readback used the publication budget')
                    invalidated = await asyncio.to_thread(self._gc_post, '/kvmem/history-edit',
                        dict(session=self._gc_session, message=record['message'], edit_id=checksum), remaining)
                    if invalidated.get('status') != 'history_proofs_invalidated':
                        raise ValueError('Controller did not acknowledge proof invalidation')
            except TimeoutError as error:
                failure = dict(admitted=False, reason='commit_deadline', error=repr(error))
            except Exception as error:
                failure = dict(admitted=False, reason='commit_prerequisite_failed', error=repr(error))
            elapsed = time.monotonic()-started
            decision = planner.recheck(proposal, messages, prepared_seconds+elapsed)
            if not failure and elapsed >= budget:
                failure = dict(admitted=False, reason='commit_deadline')
            decision.update(failure, publication_seconds=elapsed)
        recorded = self._gc_card_audit(dict(status='committed' if decision['admitted'] else 'ready_card_retained',
            message=record['message'], archive_sha256=checksum, **decision))
        if not decision['admitted'] or not recorded:
            return False
        chat.messages[record['message']]['content'] = proposal['replacement']
        chat.reset_response_chain()
        self._gc_owned_tools.committed(record)
        return True

    async def _gc_before_chat(self,chat,prompt,session):
        if self._gc_card_planner is None:return prompt
        deadline = time.monotonic()+CARD_FOREGROUND_SECONDS
        history_edited = False
        if self._gc_repeat_job is not None and self._gc_repeat_job.done():
            await self._gc_repeat_job
            self._gc_repeat_job = None
        if self._gc_card_job is not None and self._gc_card_job.done():
            proposal=await self._gc_card_job
            self._gc_card_job=None
            if proposal:
                history_edited = await self._gc_commit_ready_card(chat, prompt, proposal, session, deadline)
        self._gc_owned_tools.bind(chat.messages,prompt)
        if self._gc_card_job is not None or self._gc_repeat_job is not None or session is None:return prompt
        snapshot=[dict(m) for m in chat.messages]+[dict(role='user',content=prompt)]
        request=dict(messages=snapshot,model=self._model_name.removeprefix('openai/'),
                     response_format=response_format())
        # LiteLLM puts extra_body fields at the top level of the native request.
        # In particular, thinking_budget changes the structured-output template.
        options = self._llm_call_kwargs | (self._llm_call_kwargs.get('extra_body') or {})
        request.update({key: options[key] for key in PROMPT_TEMPLATE_OPTIONS if key in options})
        if not history_edited and time.monotonic() < deadline:
            replacement = await self._gc_present_repeat(request, session, deadline)
            if replacement is not None:
                return replacement
        if self._gc_repeat_job is None:
            self._gc_cards_tried.intersection_update(r['ordinal'] for r in self._gc_owned_tools.records)
            candidates=[r for r in self._gc_owned_tools.candidates(snapshot,self.options.gc_cards_keep_recent)
                        if r['ordinal'] not in self._gc_cards_tried]
            if candidates:
                self._gc_card_job=asyncio.create_task(self._gc_stage_card(request,candidates,session))
        return prompt

    def _gc_post(self,path,value,timeout=3):
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        request=urllib.request.Request(self._gc_url+path,data=json.dumps(value,ensure_ascii=False).encode(),
            headers={'Content-Type':'application/json','X-KVMem-Step-Key':self._gc_token})
        with opener.open(request,timeout=timeout) as response:return json.load(response)

    async def _handle_llm_interaction(self,chat,prompt,original_instruction='',session=None):
        prompt=await self._gc_before_chat(chat,prompt,session)
        result=await super()._handle_llm_interaction(chat,prompt,original_instruction,session)
        self._gc_current=None
        if not (result[2] and 'ERROR:' in result[2]):
            response=result[-1]
            self._gc_current=dict(content=response.content or '',thinking=response.reasoning_content or '',
                instruction=original_instruction)
        return result

    def _limit_output_length(self,output,max_bytes=10000):
        if self._gc_capturing_output and self._gc_raw_complete:
            size=len(output.encode('utf-8'))
            if self._gc_raw_bytes+size>self._gc_owned_tools.max_bytes:
                self._gc_raw_output=[];self._gc_raw_complete=False
            else:
                self._gc_raw_output.append(output);self._gc_raw_bytes+=size
        return super()._limit_output_length(output,max_bytes)

    async def _execute_commands(self,commands,session):
        self._gc_raw_output=[];self._gc_raw_bytes=0;self._gc_raw_complete=True
        self._gc_capturing_output=self._gc_card_planner is not None
        try:timeout,feedback=await super()._execute_commands(commands,session)
        finally:self._gc_capturing_output=False
        if self._gc_card_planner is not None and self._gc_current is not None and self._gc_raw_complete:
            self._gc_owned_tools.register(feedback=feedback,raw_feedback='\n'.join(self._gc_raw_output) or feedback,
                commands=[dict(keystrokes=c.keystrokes,duration=c.duration_sec) for c in commands],timed_out=timeout)
        if self._gc_current is None:return timeout,feedback
        event=dict(self._gc_current,phase='tool_feedback_complete',session=self._gc_session,
            feedback=feedback,tool_timeout=timeout,tool_feedback_unix_ns=time.time_ns(),
            commands=[dict(keystrokes=c.keystrokes,duration=c.duration_sec) for c in commands])
        try:
            completed=await asyncio.to_thread(self._gc_post,'/kvmem/step-end',event)
            if self._gc_owned_tools.pending is not None:
                self._gc_owned_tools.pending['receipt'] = completed
            self._gc_events.append(dict(step_id=completed.get('step_id'),status=completed.get('status')))
            # This callback only registers the real tool boundary. It never
            # generates or appends text; the next ordinary model response will
            # carry the previous step's bounded summary.
            return timeout,feedback
        except (OSError,ValueError,TypeError,KeyError) as error:
            self._gc_events.append(dict(status='callback_failed',error=repr(error)))
            self.logger.warning('Step registration failed; original source retained: %s',error)
            return timeout,feedback

    async def run(self,*args,**kwargs):
        try:result=await super().run(*args,**kwargs)
        finally:
            if self._gc_card_job is not None:
                # Preparation owns local archive I/O: let it close its files, but
                # never publish a final card or generate a maintenance response.
                await self._gc_card_job
                self._gc_card_job=None
            if self._gc_repeat_job is not None:
                await self._gc_repeat_job
                self._gc_repeat_job=None
        try:
            final=await asyncio.to_thread(self._gc_post,'/kvmem/finalize',dict(session=self._gc_session))
        except (OSError,ValueError,TypeError,KeyError) as error:
            final=dict(status='final_callback_failed',error=repr(error))
        (self.logs_dir/'thinking-summary-lifecycle.json').write_text(json.dumps(dict(
            steps=self._gc_events,final=final,harness_compaction=False),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        return result
