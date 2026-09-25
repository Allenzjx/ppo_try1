"""Safe while Isaac runs: imports stdlib and the guarded candidate only."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('finite_aux_candidate',HERE/'finite_rr_mean_aux_candidate.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


class CandidateContracts(unittest.TestCase):
    def event(self,steps=2):
        return dict(schema='wlr50_clean.finite_rr_mean_row_aux_event.v1',event_id='a'*64,
                    actual_optimizer_steps=steps,accepted_steps=1,PPO_credit=0,extra={'kept':[3]})

    def test_ledger_zero_and_deepcopy(self):
        self.assertEqual(helper.validate_local_auxiliary_events(None,{'auxiliary_updates':0}),[])
        event=self.event()
        result=helper.validate_local_auxiliary_events([event],{'auxiliary_updates':2})
        self.assertEqual(result,[event])
        result[0]['extra']['kept'].append(4)
        self.assertEqual(event['extra']['kept'],[3])

    def test_ledger_rejects_inconsistent_credit_and_duplicates(self):
        for events,counts in ((None,1),([self.event()],0),([self.event(),self.event()],4)):
            with self.assertRaises(ValueError):
                helper.validate_local_auxiliary_events(events,{'auxiliary_updates':counts})
        for key,value in [('PPO_credit',1),('PPO_credit',False),('actual_optimizer_steps',65),
                          ('actual_optimizer_steps',0),('accepted_steps',3)]:
            event=self.event();event[key]=value
            with self.assertRaises(ValueError):
                helper.validate_local_auxiliary_events([event],{'auxiliary_updates':2})

    def test_recipe_finite_bounds(self):
        for steps in (1,8,64):
            self.assertEqual(helper.recipe(steps=steps)['steps'],steps)
        for steps in (0,65,True):
            with self.assertRaises(ValueError):helper.recipe(steps=steps)
        for lr in (0,-1,float('nan'),.1,True):
            with self.assertRaises(ValueError):helper.recipe(learning_rate=lr)

    def test_live_run_and_tensor_guard_reject_without_torch(self):
        before=set(sys.modules)
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError,'unsealed'):
                helper.prepare_success_package(folder,expected_runtime={},parent_checkpoint='',
                    checkpoint_sha256='',manifest_sha256='')
        with self.assertRaisesRegex(ValueError,'Isaac exit'):
            helper.fit_rr_mean_rows(None,{},helper.recipe())
        self.assertNotIn('torch',set(sys.modules)-before)
        self.assertNotIn('pxr',set(sys.modules)-before)

    def test_failed_nonfinite_evidence_is_explicit_serializable(self):
        value=helper._failure_json_safe({'rows':[float('nan'),float('inf')], 'actual_optimizer_steps':1})
        self.assertEqual(value['actual_optimizer_steps'],1)
        self.assertIn('nonfinite_numeric_evidence',value['rows'][0])
        json.dumps(value,allow_nan=False)


if __name__=='__main__':unittest.main()
