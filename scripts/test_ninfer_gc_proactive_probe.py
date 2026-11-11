import unittest
import json
from pathlib import Path
import tempfile
from ninfer_gc_proactive_probe import audit_gc, execute


class ProactiveAuditTest(unittest.TestCase):
    def test_only_actual_nonpressure_commits_with_source_proof_pass(self):
        proof=dict(source_begin=64,source_end=192,result_begin=256,result_end=320)
        proposal='KVMEM_GC_AUDIT event=propose policy=statistical incarnation=1 slot=2 generation=3 round=4 original_begin=64 valid_tokens=64 ready=500 low=20 high=40\n'
        ack='KVMEM_GC_AUDIT event=ack incarnation=1 slot=2 generation=3 round=4 intent_reason=semantic outcome=committed\n'
        reclaim='KVMEM_GC_AUDIT event=reclaim incarnation=1 slot=2 generation=3 round=4 intent_known=1 outcome=ready ready=1\n'
        summary='KVMEM_NATIVE_GC_SUMMARY policy=statistical committed=1 reclaimed=1\n'
        complete=proposal+ack+reclaim+summary
        passed=audit_gc(complete.encode(),[proof],True,'statistical')
        self.assertTrue(passed['proactive_model_gate'])
        self.assertEqual(passed['nonpressure_commits'],1)
        for invalid in (proposal+summary, ack+summary, proposal+ack+summary,
                        complete.replace('ready=500','ready=30'),
                        complete.replace('original_begin=64','original_begin=256'),
                        complete.replace('reclaimed=1','reclaimed=0'),
                        complete.replace('outcome=committed','outcome=reevaluate')):
            self.assertFalse(audit_gc(invalid.encode(),[proof],True,'statistical')['proactive_model_gate'])
        self.assertFalse(audit_gc(complete.encode(),[proof],False,'statistical')['proactive_model_gate'])

    def test_followup_cumulative_summaries_do_not_count_retirement_twice(self):
        proof=dict(source_begin=64,source_end=192,result_begin=256,result_end=320)
        proposal='KVMEM_GC_AUDIT event=propose policy=statistical incarnation=1 slot=2 generation=3 round=4 original_begin=64 valid_tokens=64 ready=500 low=20 high=40\n'
        ack='KVMEM_GC_AUDIT event=ack incarnation=1 slot=2 generation=3 round=4 intent_reason=semantic outcome=committed\n'
        reclaim='KVMEM_GC_AUDIT event=reclaim incarnation=1 slot=2 generation=3 round=4 intent_known=1 outcome=ready ready=1\n'
        summary='KVMEM_NATIVE_GC_SUMMARY policy=statistical committed=1 reclaimed=1\n'
        complete=proposal+ack+reclaim+summary
        result=audit_gc((complete+ack+reclaim+summary).encode(),[proof],True,'statistical')
        self.assertTrue(result['proactive_model_gate'])
        self.assertEqual(result['native_committed'],1)
        self.assertEqual(result['native_reclaimed'],1)
        self.assertEqual(result['committed_with_source_proof'],1)
        second=complete.replace('incarnation=1','incarnation=2')
        moved=audit_gc((complete+second).encode(),[proof],True,'statistical')
        self.assertTrue(moved['proactive_model_gate'])
        self.assertEqual(moved['native_committed'],2)
        self.assertEqual(moved['native_reclaimed'],2)

    def test_off_task_pass_is_distinct_from_nonzero_gc(self):
        result=audit_gc(b'',[],True,'off')
        self.assertIsNone(result['proactive_model_gate'])
        self.assertTrue(result['task_gate'])
        self.assertEqual(result['native_committed'],0)
        self.assertFalse(audit_gc(b'',[],True,'statistical')['proactive_model_gate'])

    def test_owned_executor_writes_exact_bytes_and_rejects_other_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve()
            target=root/'artifact.txt'
            def call(name, **args):
                return dict(function=dict(name=name,arguments=json.dumps(args)))
            body='actual artifact\n中文\n'
            result=execute(call('write',file_path=str(target),content=body),root)
            self.assertEqual(target.read_bytes(),body.encode())
            self.assertIn('Created file',result)
            read=json.loads(execute(call('read',file_path=str(target)),root))
            self.assertEqual(read['content'],body)
            self.assertEqual(read['line_count'],2)
            for other in (root/'other.txt',root.parent/'outside.txt'):
                with self.assertRaises(ValueError):
                    execute(call('write',file_path=str(other),content='unexpected'),root)
                self.assertFalse(other.exists())
            with self.assertRaises(ValueError):
                execute(call('remove',file_path=str(target)),root)
            self.assertTrue(target.exists())


if __name__=='__main__':
    unittest.main()
