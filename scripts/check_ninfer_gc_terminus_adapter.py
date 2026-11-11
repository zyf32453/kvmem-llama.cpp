#!/usr/bin/env python3
"""Exercise the pinned real Harbor parser/tool-return hooks without a GPU.

The LLM and tool session are explicit fixtures. This is an adapter contract
check, never real-model or official Terminal-Bench acceptance evidence.
"""
import asyncio
import hashlib
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace

from ninfer_gc_terminus_agent import ThinkingSummaryTerminus2
from ninfer_gc_step_controller import StepSummaryController,controller_handler
from ninfer_gc_agent_bridge import NativeClient
from test_ninfer_gc_inline_summary import NativeFixture
from ninfer_gc_tool_cards import CostProfile
from dataclasses import asdict


async def check(root):
    native=NativeFixture()
    controller=StepSummaryController(native,'contract-check',root/'lifecycle')
    token='contract-check-only-fixture-token-0001'
    token_file=root/'control.token';token_file.write_text(token)
    options=ThinkingSummaryTerminus2.parse_options(dict(gc_controller_url='http://127.0.0.1:1',
        gc_session_key='contract-check',gc_control_token_file=str(token_file),enable_summarize=False,interleaved_thinking=True))
    assert options.gc_session_key=='contract-check'
    server=ThreadingHTTPServer(('127.0.0.1',0),
        controller_handler(controller,'http://127.0.0.1:1',root/'capture',token))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        agent=ThinkingSummaryTerminus2(logs_dir=root/'agent',model_name='openai/contract-check',
            api_base='http://127.0.0.1:1/v1',enable_summarize=False,proactive_summarization_threshold=0,
            interleaved_thinking=True,model_info=dict(max_input_tokens=245760,max_output_tokens=16384,
                input_cost_per_token=0,output_cost_per_token=0),
            gc_controller_url='http://127.0.0.1:'+str(server.server_port),
            gc_session_key='contract-check',gc_control_token_file=token_file)
        body=dict(analysis='Inspect the fixture output.',plan='Read the fixture.',
            commands=[dict(keystrokes='printf fixture\\n\n',duration=.1)],task_complete=False)
        content=json.dumps(body);thinking='The fixture produces the requested observable output. '*10
        class Chat:
            async def chat(self,prompt,**kwargs):
                return SimpleNamespace(content=content,reasoning_content=thinking)
        returned=[]
        class Session:
            async def send_keys(self,keys,**kwargs):returned.append(('command',keys));await asyncio.sleep(0)
            async def get_incremental_output(self):returned.append(('feedback','fixture'));return 'fixture\n'
        parsed=await agent._handle_llm_interaction(Chat(),'fixture prompt','fixture instruction',Session())
        assert not parsed[2] and len(parsed[0])==1
        controller.note_response(dict(prompt_cache_key='contract-check',messages=[dict(role='user',content='Task')]),
            dict(id='fixture-reply',choices=[dict(message=dict(content=content,reasoning_content=thinking))]),'fixture',200)
        timed_out,observation=await agent._execute_commands(parsed[0],Session())
        assert not timed_out and [r[0] for r in returned]==['command','feedback']
        assert 'fixture\n' in observation and 'Retained summary' not in observation
        assert len(controller.records)==1 and not agent.options.enable_summarize
        assert native.inferences==0
        # Parse errors cannot manufacture a completed step.
        class InvalidChat:
            async def chat(self,prompt,**kwargs):return SimpleNamespace(content='unparseable',reasoning_content=thinking)
        invalid=await agent._handle_llm_interaction(InvalidChat(),'p','i',Session())
        assert 'ERROR:' in invalid[2] and agent._gc_current is None
        return dict(status='passed',harbor_version='0.25.0',cli_preflight_options=True,real_parent_parser=True,
            tool_feedback_before_callback=True,harness_compaction=False,invalid_reply_retained=True,
            fixture_only=True,real_model_acceptance=False)
    finally:controller.finalize();server.shutdown();server.server_close()


async def check_transport(root,cards=False,repeats=False,inline_archive=False,pipeline_archive=False,command_repeat=False):
    """Actual Harbor Chat + LiteLLM transport, with an explicit model fixture."""
    from harbor.llms.chat import Chat
    fixture=NativeFixture();observed=[];template_requests=[]
    repeated_lines = ''.join(f'row {i:04d} stable payload for file read\n' for i in range(90))
    issued_command = "cat << 'FIXTURE_END'\n" + repeated_lines + 'FIXTURE_END\n' if command_repeat else 'printf fixture\n'
    content=json.dumps(dict(analysis='Observe the fixture.',plan='Inspect its result.',
        commands=[dict(keystrokes=issued_command,duration=.1)],task_complete=False))
    thinking='The fixture returns observable output for inspection. '*20
    class Model(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):self.respond(fixture.json(self.path))
        def do_POST(self):
            value=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            if self.path=='/apply-template':
                template_requests.append(value)
                if require_overlap and pipeline_archive:
                    assert archive_verifying.wait(.2), 'Native proof serialized before archive transfer'
                return self.respond(fixture.json(self.path,value))
            observed.append(value)
            prompt_tokens = len(fixture.json('/apply-template', value)['prompt']) if repeats else 100
            completion_tokens = len(content)+len(thinking) if repeats else 300
            native_counters = dict(timings=dict(cache_n=0, prompt_n=prompt_tokens,
                predicted_n=completion_tokens)) if repeats else {}
            self.respond(dict(id='fixture-'+str(len(observed)),object='chat.completion',created=1,model='contract-check',
                choices=[dict(index=0,finish_reason='stop',message=dict(role='assistant',content=content,reasoning_content=thinking))],
                usage=dict(prompt_tokens=prompt_tokens,completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens+completion_tokens,completion_tokens_details=dict(reasoning_tokens=200)),
                **native_counters))
        def respond(self,body):
            raw=json.dumps(body).encode();self.send_response(200);self.send_header('Content-Type','application/json')
            self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
    model=ThreadingHTTPServer(('127.0.0.1',0),Model)
    native_url='http://127.0.0.1:'+str(model.server_port)
    controller=StepSummaryController(NativeClient(native_url),'transport',root/'lifecycle')
    token='transport-fixture-control-token-000000';token_file=root/'control.token';token_file.write_text(token)
    server=ThreadingHTTPServer(('127.0.0.1',0),controller_handler(controller,native_url,root/'capture',token))
    for surface in (model,server):threading.Thread(target=surface.serve_forever,daemon=True).start()
    archive_files={}
    created_remote_archives=set()
    corrupted = False
    slow_archive = None
    require_overlap = False
    archive_verifying = threading.Event()
    registration_started = threading.Event()
    class Environment:
        async def upload_file(self,source,target):archive_files[target]=Path(source).read_bytes()
        async def exec(self,command,**kwargs):
            import shlex
            inline = command.startswith('# KVMem verified archive transfer v1\n')
            if inline and not inline_archive:
                # Exercise the documented fallback for task images without
                # the inline transfer tools, independently of the fast route.
                return SimpleNamespace(return_code=127,stdout='')
            if require_overlap:
                archive_verifying.set()
                assert await asyncio.to_thread(registration_started.wait, .2)
                assert len(observed) == 2
            if inline:
                target=Path(shlex.split(next(line[7:] for line in command.splitlines()
                                           if line.startswith('target=')))[0])
                if not target.exists():created_remote_archives.add(target)
                process=await asyncio.create_subprocess_exec('sh','-c',command,
                    stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.PIPE)
                stdout,stderr=await process.communicate()
                assert process.returncode==0,stderr.decode()
                archive_files[str(target)]=target.read_bytes()
                text=stdout.decode()
                return SimpleNamespace(return_code=0,stdout=('0'*64+text[64:] if corrupted else text))
            args=shlex.split(command)
            assert args[0]=='sha256sum'
            if args[1] == slow_archive:
                await asyncio.sleep(2)
            checksum = hashlib.sha256(archive_files[args[1]]).hexdigest()
            return SimpleNamespace(return_code=0,stdout=('0'*64 if corrupted else checksum)+'  '+args[1])
    class Session:
        environment=Environment()
        async def send_keys(self,*args,**kwargs):pass
        async def get_incremental_output(self):
            if repeats:
                return (f'Unique beginning for invocation {len(observed)}\n'
                         + repeated_lines
                        + f'Unique ending for invocation {len(observed)}\n')
            return ('Beginning of raw log\n'+''.join(f'row {i:04d} archive-only detail\n' for i in range(1000))
                    if cards else 'The actual fixture feedback was returned. '*30)
    try:
        extra={}
        if cards or repeats:
            profile=root/'fixture-cost.json'
            profile.write_text(json.dumps(asdict(CostProfile(10000,1000000,evidence='adapter fixture, not model data'))))
            extra=dict(gc_tool_cards='apply',gc_cards_cost_profile=str(profile),
                       gc_cards_min_context_tokens=0 if repeats else 40000,
                       gc_cards_keep_recent=3 if repeats else 1)
        if repeats:
            extra['llm_call_kwargs'] = {'extra_body': {'thinking_budget': 4096}}
        agent=ThinkingSummaryTerminus2(logs_dir=root/'agent',model_name='openai/contract-check',
            api_base='http://127.0.0.1:'+str(server.server_port)+'/v1',enable_summarize=False,
            proactive_summarization_threshold=0,interleaved_thinking=True,
            model_info=dict(max_input_tokens=245760,max_output_tokens=16384,input_cost_per_token=0,output_cost_per_token=0),
            gc_controller_url='http://127.0.0.1:'+str(server.server_port),gc_session_key='transport',gc_control_token_file=token_file,**extra)
        chat=Chat(model=agent._llm,interleaved_thinking=True)
        parsed=await agent._handle_llm_interaction(chat,'Fixture task','Fixture instruction',Session())
        assert not parsed[2] and parsed[-1].reasoning_content==thinking
        timeout,observation=await agent._execute_commands(parsed[0],Session())
        assert not timeout and 'Retained summary' not in observation
        await agent._handle_llm_interaction(chat,observation,'Fixture instruction',Session())
        assert len(observed)==2 and observed[-1]['messages'][1]['reasoning_content']==thinking
        assert 'kvmem_gc' not in observed[-1]  # Summary is carried by response 2; scope maps later.
        assert 'step_summary' not in observed[-1]['response_format']['json_schema']['schema']['required']
        assert controller.counts['auxiliary_model_calls']==0
        if repeats:
            async def wait_for_receipt():
                # The proxy may finish logging after delivering the body.
                # Let the fixture provide the normal already-finished receipt.
                for _ in range(200):
                    if any(r['id'] == 'fixture-'+str(len(observed)) for r in list(controller.receipts.values())):
                        return
                    await asyncio.sleep(0.005)
                raise AssertionError('Latest native receipt unavailable')
            await wait_for_receipt()
            prefix = [dict(m) for m in chat.messages]
            _, original = await agent._execute_commands(parsed[0], Session())
            post = agent._gc_post

            def expect_concurrent_verification(path, value, timeout=3):
                if require_overlap and path == '/kvmem/first-presentation':
                    registration_started.set()
                    assert archive_verifying.wait(.2)
                return post(path, value, timeout)

            agent._gc_post = expect_concurrent_verification
            require_overlap = True
            await agent._handle_llm_interaction(chat, original, 'Fixture instruction', Session())
            require_overlap = False
            assert archive_verifying.is_set() and registration_started.is_set()
            assert '[Repeated tool output,' in observed[-1]['messages'][-1]['content']
            assert 'Unique ending for invocation 2' in observed[-1]['messages'][-1]['content']
            forwarded = observed[-1]
            planned = [r for r in template_requests if r.get('token_ids_only') and
                       r['messages'][:-1] == forwarded['messages'][:-1]]
            assert {r['messages'][-1]['content'] for r in planned} == {
                original, forwarded['messages'][-1]['content']}
            assert forwarded['thinking_budget'] == 4096
            assert all(r.get('thinking_budget') == forwarded['thinking_budget'] and
                       r['response_format'] == forwarded['response_format'] for r in planned)
            assert chat.messages[:len(prefix)] == prefix
            assert controller.history_epoch == 0
            assert any(original.encode() in body for body in archive_files.values())
            assert any(r.get('presented_feedback') for r in list(controller.records.values()))
            await wait_for_receipt()
            corrupted = True
            _, retained = await agent._execute_commands(parsed[0], Session())
            await agent._handle_llm_interaction(chat, retained, 'Fixture instruction', Session())
            assert observed[-1]['messages'][-1]['content'] == retained
            audits = [json.loads(line) for line in (root/'agent/tool-cards.jsonl').read_text().splitlines()]
            assert any(r['status'] == 'first_presentation_committed' for r in audits), audits
            assert any(r['status'] == 'retained_after_preparation_failure' for r in audits), audits
            if command_repeat:
                assert any(r.get('witness_kind') == 'executed_command' for r in audits), audits
                # It is possible to shorten the very first tool feedback:
                # the issued command is visible without any old tool witness.
                assert '[Repeated tool output,' in observed[1]['messages'][-1]['content']
                assert 'issued command #1' in observed[1]['messages'][-1]['content']
            if pipeline_archive:
                await wait_for_receipt()
                corrupted = False
                # A successfully copied original never substitutes for native
                # proof. Reject that proof after staging has been scheduled.
                agent._gc_card_planner.measure_repeated_tail = lambda *_: (
                    None, dict(status='unproved_template_edit'))
                _, retained = await agent._execute_commands(parsed[0], Session())
                await agent._handle_llm_interaction(chat, retained, 'Fixture instruction', Session())
                if agent._gc_repeat_job is not None:
                    await agent._gc_repeat_job
                assert observed[-1]['messages'][-1]['content'] == retained
                audits = [json.loads(line) for line in (root/'agent/tool-cards.jsonl').read_text().splitlines()]
                assert any(r['status'] == 'unproved_template_edit' for r in audits), audits
            return dict(status='passed', actual_harbor_chat=True, actual_litellm_transport=True,
                        first_presentation_shortened=True, prior_history_byte_equal=True,
                        template_matches_forwarded_thinking_budget=True,
                        archive_verification_overlaps_registration=True,
                        archive_overlaps_native_proof=pipeline_archive,
                        rejected_native_proof_preserves_feedback=pipeline_archive,
                        archive_failure_preserves_new_feedback=True, fixture_only=True)
        if cards:
            removed_archive=None
            slow_checksum=None
            for _ in range(8):
                _,observation=await agent._execute_commands(parsed[0],Session())
                parsed=await agent._handle_llm_interaction(chat,observation,'Fixture instruction',Session())
                if agent._gc_card_job is not None:
                    ready=await agent._gc_card_job
                    recent=json.loads((root/'agent/tool-cards.jsonl').read_text().splitlines()[-1])
                    if recent['status']=='below_context_gate':
                        assert not agent._gc_cards_tried, 'Below-threshold feedback must remain retryable'
                    if ready and removed_archive is None:
                        removed_archive=ready['archive']['sha256']
                        archive_files.pop(ready['archive']['remote_path'])
                    elif ready and slow_archive is None:
                        slow_checksum=ready['archive']['sha256']
                        slow_archive=ready['archive']['remote_path']
            audits=[json.loads(line) for line in (root/'agent/tool-cards.jsonl').read_text().splitlines()]
            assert any(r['status']=='below_context_gate' for r in audits),audits
            assert any(r['status']=='committed' for r in audits),audits
            assert any(r['status']=='ready_card_retained' and r.get('reason')=='commit_prerequisite_failed'
                       and r['archive_sha256']==removed_archive for r in audits),audits
            assert not any(r['status']=='committed' and r['archive_sha256']==removed_archive for r in audits)
            assert any(r['status']=='ready_card_retained' and r.get('reason')=='commit_deadline'
                       and r['archive_sha256']==slow_checksum and r['publication_seconds']<1 for r in audits),audits
            assert not any(r['status']=='committed' and r['archive_sha256']==slow_checksum for r in audits)
            assert controller.history_epoch>0
            assert any('[KVMem archived tool feedback;' in str(m.get('content')) for m in chat.messages)
            assert archive_files and any(b'row 0500 archive-only detail' in raw for raw in archive_files.values())
            # Raw middle data was excluded by Harbor's 10K limiter but remains
            # retrievable; the actual subsequent transport carries the card.
            assert any('[KVMem archived tool feedback;' in str(m.get('content')) for m in observed[-1]['messages'])
            # Capture limits cannot turn an incomplete original into an archive.
            agent._gc_owned_tools.max_bytes=128
            await agent._execute_commands(parsed[0],Session())
            assert not agent._gc_raw_complete and not agent._gc_raw_output
            assert agent._gc_owned_tools.pending is None
            return dict(status='passed',actual_harbor_chat=True,actual_litellm_transport=True,
                archived_before_history_edit=True,pre_truncation_original_preserved=True,
                below_threshold_retried_after_growth=True,
                missing_archive_at_commit_preserves_original=True,
                slow_archive_at_commit_preserves_original_with_bounded_wait=True,
                oversized_raw_kept_out_of_card_candidates=True,
                controller_epoch_invalidated=True,ordinary_request_contains_short_card=True,
                fixture_only=True,real_model_acceptance=False)
        return dict(status='passed',actual_harbor_chat=True,actual_litellm_transport=True,
            native_receipt_matches_client_thinking=True,next_history_keeps_original_thinking=True,
            summary_carried_by_normal_response=True,auxiliary_model_calls=0,fixture_only=True,real_model_acceptance=False)
    finally:
        controller.finalize()
        for surface in (server,model):surface.shutdown();surface.server_close()
        for path in created_remote_archives:path.unlink(missing_ok=True)


def main():
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);transport=root/'transport';transport.mkdir()
        result=dict(adapter=asyncio.run(check(root)),transport=asyncio.run(check_transport(transport)))
        cards=root/'cards';cards.mkdir()
        result['tool_cards']=asyncio.run(check_transport(cards,cards=True))
        repeats=root/'repeats';repeats.mkdir()
        result['repeated_feedback']=asyncio.run(check_transport(repeats,repeats=True))
        inline=root/'inline';inline.mkdir()
        result['inline_archive']=asyncio.run(check_transport(inline,repeats=True,inline_archive=True))
        pipeline=root/'pipeline';pipeline.mkdir()
        result['pipeline_archive']=asyncio.run(check_transport(pipeline,repeats=True,inline_archive=True,pipeline_archive=True))
        command=root/'command';command.mkdir()
        result['executed_command']=asyncio.run(check_transport(command,repeats=True,inline_archive=True,command_repeat=True))
        print(json.dumps(result,indent=2))


if __name__=='__main__':main()
