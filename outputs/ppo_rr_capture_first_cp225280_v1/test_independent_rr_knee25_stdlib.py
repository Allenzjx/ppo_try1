"""Stdlib-only tests: no production import, Torch, Isaac or actual robot."""
import ast
import copy
import importlib.util
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

PATH = Path(__file__).with_name('independent_rr_knee25_diagnostic.py')
spec = importlib.util.spec_from_file_location('knee25_probe',PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fake_core():
    baseline = [0.]*12
    baseline[6:8] = [-8.15,-39.05]
    cap = [36.]*12
    cap[6] = 24.
    final = [0.]*12
    final[6:8] = [7.485143838259594,-58.]
    frame = SimpleNamespace(physics_tick=100,state_id='P09',sim_time_s=1.,
        info={'drive_target_full12':final,'atomic_ack':{
            'policy_headroom_evidence':{'baseline_native_plus_controller_full12':baseline}}})
    return SimpleNamespace(frame=frame,backend=SimpleNamespace(execution_profile={
        'residual':{'phase_caps_full12':{'P09':cap}}}))


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.driver = {'script_sha256':'a'*64,'checkpoint_sha256':'b'*64}
        self.probe = m.Knee25Probe(self.driver)
        self.core = fake_core()
        self.selected = [i/100 for i in range(12)]
        self.entry = list(self.core.frame.info['drive_target_full12'])

    def test_stdlib_import_and_default_execute_guard(self):
        self.assertNotIn('torch',sys.modules)
        self.assertNotIn('pxr',sys.modules)
        self.assertFalse(any(x.startswith('wlr50_clean') for x in sys.modules))
        with self.assertRaisesRegex(ValueError,'explicit'):
            m.execute(SimpleNamespace(run_independent_diagnostic=False))
        self.assertNotIn('torch',sys.modules)

    def test_entry_anchor_not_accumulated_each_call(self):
        _,first = self.probe(self.core,self.selected,self.entry,0.)
        self.assertEqual(first['candidate_absolute_targets'],{6:self.entry[6],7:-58.})
        _,a = self.probe(self.core,self.selected,self.entry,1.5)
        _,b = self.probe(self.core,self.selected,self.entry,10.)
        self.assertEqual(a['candidate_absolute_targets'],b['candidate_absolute_targets'])
        self.assertAlmostEqual(a['candidate_absolute_targets'][6],-17.514856161740404)
        self.assertEqual(a['candidate_absolute_targets'][7],-33.)

    def test_ramp_half_and_other10_unchanged(self):
        issued,audit = self.probe(self.core,self.selected,self.entry,.75)
        self.assertEqual(audit['candidate_absolute_targets'][7],-45.5)
        for ch in set(range(12))-{6,7}:
            self.assertEqual(issued[ch],self.selected[ch])
        self.assertEqual(audit['PPO_credit'],0)
        self.assertFalse(audit['is_on_policy_action'])
        self.assertFalse(audit['automatic_teacher_label'])

    def test_positive25_candidate_expressible_and_not_saturated(self):
        issued,audit = self.probe(self.core,self.selected,self.entry,1.5)
        self.assertAlmostEqual(issued[7],.16966503092874927)
        reconstructed = -39.05+36.*math.tanh(issued[7])
        self.assertAlmostEqual(reconstructed,-33.)
        self.assertEqual(audit['inverse_tanh_limited_channels'],[])

    def test_no_first_TOP_latch_from_air_or_historical_placed(self):
        snap = {'active':True,'metrics':{'current_top_contact':False,'placed':True}}
        self.probe.observe_native(self.core.frame,snap)
        self.assertIsNone(self.probe.first_top)

    def test_first_actual_TOP_FINAL_latched_and_not_overwritten(self):
        self.core.frame.info['drive_target_full12'][6:8] = [-17.4,-34.2]
        snap = {'active':True,'metrics':{'current_top_contact':True,'tick':100,
            'current_top_bearing':False,'current_attempt_capture_eligible':True}}
        self.probe.observe_native(self.core.frame,snap)
        self.core.frame.info['drive_target_full12'][6:8] = [-10.,-55.]
        self.probe.observe_native(self.core.frame,snap)
        _,audit = self.probe(self.core,self.selected,self.entry,6.,{6:-10.,7:-55.})
        self.assertEqual(audit['candidate_absolute_targets'],{6:-17.4,7:-34.2})
        self.assertFalse(audit['first_native_TOP_latch']['current_bearing'])

    def test_limited_request_explicitly_records_projection_not_hidden(self):
        self.core.frame.info['atomic_ack']['policy_headroom_evidence'][
            'baseline_native_plus_controller_full12'][7] = -200.
        issued,audit = self.probe(self.core,self.selected,self.entry,1.5)
        self.assertIn(7,audit['inverse_tanh_limited_channels'])
        self.assertAlmostEqual(math.tanh(issued[7]),.985)
        self.assertTrue(audit['actual_final_targets_require_native_dispatch_evidence'])

    def test_declared_checkpoint_counts_not_latest_pointer_guess(self):
        m.validate_selected_counts(dict(local_policy_decisions=2560,local_ppo_updates=5,
            local_optimizer_steps=100,auxiliary_updates=0),expected_local_decisions=2560,
            expected_ppo_updates=5,expected_optimizer_steps=100)
        m.validate_selected_counts(dict(local_policy_decisions=3072,local_ppo_updates=6,
            local_optimizer_steps=120,auxiliary_updates=0),expected_local_decisions=3072,
            expected_ppo_updates=6,expected_optimizer_steps=120)
        expected = dict(expected_local_decisions=3072,expected_ppo_updates=6,
            expected_optimizer_steps=120)
        for key in ('local_policy_decisions','local_ppo_updates','local_optimizer_steps'):
            counts = dict(local_policy_decisions=3072,local_ppo_updates=6,
                local_optimizer_steps=120,auxiliary_updates=0)
            counts[key] -= 1
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'explicit expected'):
                m.validate_selected_counts(counts,**expected)

    def test_expected_counts_require_nonnegative_integers(self):
        counts = dict(local_policy_decisions=3072,local_ppo_updates=6,
            local_optimizer_steps=120,auxiliary_updates=0)
        for invalid in (-1,True,3072.0):
            with self.subTest(invalid=invalid),self.assertRaisesRegex(ValueError,'integers'):
                m.validate_selected_counts(counts,expected_local_decisions=invalid,
                    expected_ppo_updates=6,expected_optimizer_steps=120)

    def test_observed_step_one_forward_one_step_zero_credit(self):
        events,write_calls = [],[]
        snapshot = {'active':False,'metrics':{'phase_id':'P09','current_top_contact':False}}
        inner = fake_core()
        inner.observation = [0.]*448
        inner.task = SimpleNamespace(snapshot=lambda:copy.deepcopy(snapshot))
        calls = []
        def step(issued):
            calls.append(list(issued))
            inner.frame.physics_tick += 8
            return SimpleNamespace(observation=[1.]*448,terminated=False,info={
                'semantic_task':{'physical_evaluator':{'RR':{'top_contact':False}}},
                'actual_drive_target_full12':[2.]*12,
                'actuator_target_effect_audit':{'verified':True}})
        inner.step = step
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run/'source').mkdir()
            wrapped = m.RecordingCore(inner,self.probe,run,self.driver,
                lambda p,v:write_calls.append((p,v)),lambda stream,row:events.append(row))
            wrapped.step(self.selected)
            wrapped.close()
        self.assertEqual(len(calls),1)
        self.assertEqual(len(events),1)
        self.assertEqual(events[0]['observation_full448'],[0.]*448)
        self.assertEqual(events[0]['next_observation_full448'],[1.]*448)
        self.assertEqual(events[0]['selected_policy_raw_full12'],self.selected)
        self.assertEqual(events[0]['actually_issued_raw_full12'],self.selected)
        self.assertEqual(events[0]['PPO_credit'],0)

    def test_diagnostic_only_evaluate_and_no_save_or_optimizer_call(self):
        tree = ast.parse(PATH.read_text())
        execute = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='execute')
        text = ast.unparse(execute)
        self.assertIn('diagnostic=True',text)
        self.assertIn('route.diagnostic_request = old_callback',text)
        self.assertNotIn('route.save(',text)
        self.assertNotIn('.alg.update(',text)
        self.assertNotIn('route.train(',text)


if __name__=='__main__':
    unittest.main(verbosity=2)
