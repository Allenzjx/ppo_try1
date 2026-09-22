"""CPU-only checks; no main(), model forward, torch, Isaac, or optimizer."""
import ast
import copy
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import unittest

import direction_probe as original
import direction_probe_preaction_candidate as shared
import direction_probe_FL_minus6_candidate as previous
import direction_probe_FL_hip_minus1_knee_plus1_candidate as candidate
from test_direction_probe_preaction_candidate import fixture

CASE = 'FL_hip_minus1_knee_plus1'


def event(placed=False):
    return dict(valid=True, termination_reason=None, current_legs={'FL': dict(air=True, within_top_xy=True)},
        history={'placed': dict(FL=placed, FR=True, RR=False, RL=False), 'front_edge_crossed': dict(FL=True)})


def function_ast(module, name):
    return next(x for x in ast.parse(Path(module.__file__).read_text()).body
        if isinstance(x, ast.FunctionDef) and x.name == name)


class CoupledCandidateTests(unittest.TestCase):
    def test_CPU_environment_and_no_GPU_or_simulator_import(self):
        self.assertEqual(os.environ.get('CUDA_VISIBLE_DEVICES'), '-1')
        self.assertNotIn('torch', sys.modules)
        self.assertFalse(any(x == 'isaaclab' or x.startswith('isaaclab.') for x in sys.modules))

    def test_old_scripts_unchanged(self):
        expected = {original: '47d49535071af1b0f76cd13a249675a2c8cb93575b1b9751e3c702cdd16c1167',
            shared: 'c7c1eef885df4cccda650e81175fb73a30b6a327e5a3abbd76adaa682597690d',
            previous: 'ea6c9ac7591c50c7d4d4770206076672e31a5a8ff222d4a3567329eccaeb4577'}
        for module, expected_sha in expected.items():
            self.assertEqual(hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(), expected_sha)

    def test_core_state_machine_and_global_cases_unchanged(self):
        self.assertIs(candidate.FiniteFLHipKnee.apply, original.FiniteDirection.apply)
        self.assertIs(candidate.FiniteFLHipKnee.start, original.FiniteDirection.start)
        self.assertIs(candidate.FiniteFLHipKnee.complete, original.FiniteDirection.complete)
        saved = copy.deepcopy(original.CASES)
        probe = candidate.FiniteFLHipKnee(CASE)
        self.assertEqual(probe.offsets, {0: -1., 1: 1.})
        self.assertEqual(original.CASES, saved)
        self.assertNotIn(CASE, original.CASES)
        with self.assertRaises(ValueError): candidate.FiniteFLHipKnee('FL_minus6')
        self.assertEqual(candidate.CHANNELS[:2], ('front_left_hip', 'front_left_knee'))

    def test_anchor_is_actual_ACK_REQUEST_and_live_HISTORY_not_final_drive(self):
        history = fixture()['history']
        ack = dict(independent_policy_residual_requested_full12=list(history['previous_residual_full12']),
            drive_target_full12=[100.]*12, requested_full12=[200.]*12)
        saved = copy.deepcopy((history, ack))
        self.assertEqual(candidate.verified_ack_anchor(history, ack), history['previous_residual_full12'])
        self.assertEqual((history, ack), saved)
        for bad in (None, {}, {'independent_policy_residual_requested_full12': [float('nan')]*12},
                    {'independent_policy_residual_requested_full12': [0.]*12}):
            with self.assertRaises(ValueError): candidate.verified_ack_anchor(history, bad)

    def test_one_sealed_real_minus6_entry_ACK_equals_live_REQUEST_history(self):
        entry_path = candidate.ROOT/'runs/ppo_task_conditioned_hip_wheel_v1/diagnostics/FL_minus6_CP189952_20260921_01/probe_entry.json'
        entry = json.loads(entry_path.read_text())
        receipt = entry['pre_action_receipt']
        self.assertEqual(receipt['intervention']['case'], 'FL_minus6')
        self.assertEqual(receipt['start_tick'], 2976)
        anchor = candidate.verified_ack_anchor(receipt['actual_live_history'], receipt['previous_actual_ACK'])
        self.assertEqual(anchor, tuple(receipt['actual_live_history']['previous_residual_full12']))
        self.assertEqual(anchor, tuple(receipt['previous_actual_ACK']['independent_policy_residual_requested_full12']))
        self.assertEqual(anchor[:2], (2.4404078965400906, -8.14630390733378))
        self.assertNotEqual(anchor, tuple(receipt['previous_actual_ACK']['drive_target_full12']))

    def test_trigger_requires_actual_endpoint_and_physical_state(self):
        self.assertTrue(candidate.trigger_hip_knee(CASE, 'P05', event(), True))
        for phase, endpoint, ev in [('P02', True, event()), ('P05', False, event()), ('P05', True, event(True))]:
            self.assertFalse(candidate.trigger_hip_knee(CASE, phase, ev, endpoint))
        for mutate in ('air', 'within_top_xy', 'cross', 'valid', 'terminal'):
            ev = event()
            if mutate in ('air', 'within_top_xy'): ev['current_legs']['FL'][mutate] = False
            elif mutate == 'cross': ev['history']['front_edge_crossed']['FL'] = False
            elif mutate == 'valid': ev['valid'] = False
            else: ev['termination_reason'] = 'TASK_FAILURE_BODY_COLLISION'
            self.assertFalse(candidate.trigger_hip_knee(CASE, 'P05', ev, True))
        self.assertFalse(candidate.trigger_hip_knee('control', 'P05', event(), True))

    def test_anchor_offsets_not_recursive_other10_live_and_exact_timing(self):
        probe = candidate.FiniteFLHipKnee(CASE)
        anchor = [-2.18, -8.18] + [.2]*10
        saved = tuple(anchor); probe.start(anchor); anchor[0] = 100.
        records = []
        for i in range(118):
            baseline = tuple(.001*(i+j) for j in range(12)); caps = (18., 24.)+(18.,)*6+(1.,)*4
            raw, receipt = probe.apply(baseline, caps, 'P05', event())
            self.assertEqual(raw[2:], baseline[2:])
            self.assertEqual(receipt['case'], CASE)
            if 'index' in receipt:
                self.assertEqual(receipt['anchor_full12'], saved)
                self.assertEqual(receipt['delta_deg'], {0: -1., 1: 1.})
                self.assertTrue(receipt['not_added_recursively_to_HISTORY'])
            records.append((raw, receipt))
        for i in range(12):
            self.assertAlmostEqual(18*math.tanh(records[i][0][0]), saved[0]-original.quintic((i+1)/12))
            self.assertAlmostEqual(24*math.tanh(records[i][0][1]), saved[1]+original.quintic((i+1)/12))
        for i in range(11, 75):
            self.assertAlmostEqual(18*math.tanh(records[i][0][0]), saved[0]-1.)
            self.assertAlmostEqual(24*math.tanh(records[i][0][1]), saved[1]+1.)
        self.assertEqual(probe.release_age, 75)
        self.assertTrue(probe.complete)
        self.assertEqual(records[117][0], baseline)

    def test_capture24_release12_follow30_preserved(self):
        probe = candidate.FiniteFLHipKnee(CASE); probe.start((0.,)*12)
        for i in range(81):
            probe.apply((.02,)*12, (18.,)*12, 'P05', event(i >= 15))
            if i < 38: self.assertIsNone(probe.release_age)
            if i < 80: self.assertFalse(probe.complete)
        self.assertEqual(probe.capture_age, 15)
        self.assertEqual(probe.release_age, 39)
        self.assertTrue(probe.complete)

    def test_phase_cap_change_does_not_reset_physical_anchor(self):
        probe = candidate.FiniteFLHipKnee(CASE); probe.start((-2.18, -8.18)+(0.,)*10)
        for _ in range(12): probe.apply((.01,)*12, (18.,)*12, 'P05', event())
        saved = probe.anchor
        raw, receipt = probe.apply((.02,)*12, (32.,)*12, 'P06', event(True))
        self.assertEqual(probe.anchor, saved)
        self.assertEqual(receipt['index'], 12)
        self.assertAlmostEqual(32*math.tanh(raw[0]), -3.18)
        self.assertAlmostEqual(32*math.tanh(raw[1]), -7.18)
        self.assertEqual(raw[2:], (.02,)*10)

    def test_both_selected_channels_reject_capacity_and_nonfinite(self):
        for index, bad in ((0,-18.), (1,24.), (0,float('nan')), (1,float('inf'))):
            anchor = [0.]*12; anchor[index] = bad
            probe = candidate.FiniteFLHipKnee(CASE); probe.start(anchor)
            with self.assertRaisesRegex(ValueError, 'existing residual capacity'):
                probe.apply((0.,)*12, (18.,24.)+(18.,)*10, 'P05', event())

    def test_control_exact_pass_through(self):
        probe = candidate.FiniteFLHipKnee('control'); baseline = tuple(.01*i for i in range(12))
        self.assertEqual(probe.apply(baseline, (18.,)*12, 'P05', event()), (baseline, None))
        self.assertEqual(candidate.pre_action_receipt(**fixture())['manual_override_selector_full12'], (0,)*12)

    def receipt_arguments(self):
        args = fixture(); probe = candidate.FiniteFLHipKnee(CASE)
        probe.start(args['history']['previous_residual_full12'])
        for _ in range(12): raw, receipt = probe.apply(args['baseline'], args['caps'], 'P05', event())
        args.update(injected=raw, intervention=receipt)
        return args

    def test_receipt_true2channel_hash_and_history_no_credit(self):
        args = self.receipt_arguments(); saved = copy.deepcopy(args)
        r = candidate.pre_action_receipt(**args)
        self.assertEqual(args, saved)
        self.assertEqual(r['intervention']['case'], CASE)
        self.assertEqual(r['manual_override_selector_full12'], (1,1)+(0,)*10)
        self.assertEqual(r['manual_raw_delta_full12'][2:], (0.,)*10)
        self.assertEqual(r['actual_live_history'], args['history'])
        self.assertEqual(r['actor_input_float32_372'], tuple(candidate.f32(x) for x in args['observation']))
        self.assertAlmostEqual(r['candidate_physical_after_intervention_before_permission_and_slew_full12'][0], -.99)
        self.assertAlmostEqual(r['candidate_physical_after_intervention_before_permission_and_slew_full12'][1], 1.02)
        for name in ('new_PPO_decisions','new_PPO_updates','new_optimizer_steps'): self.assertEqual(r[name], 0)
        digest = r.pop('pre_action_receipt_sha256')
        self.assertEqual(digest, hashlib.sha256(json.dumps(r, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest())

    def test_wrong_case_foreign_channel_missing_channel_and_forged_request_rejected(self):
        for mutate, message in [('case','wrong finite intervention case'), ('foreign','unselected channel'),
            ('missing','both FL hip/knee'), ('offset','wrong finite hip/knee offsets'), ('raw','physical request differs')]:
            args = self.receipt_arguments()
            if mutate == 'case': args['intervention']['case'] = 'FL_minus6'
            elif mutate == 'missing': args['intervention']['requested_selected_deg'].pop(1)
            elif mutate == 'offset': args['intervention']['delta_deg'][1] = 6.
            else:
                raw = list(args['injected']); raw[2 if mutate == 'foreign' else 1] += .01; args['injected'] = raw
            with self.assertRaisesRegex(ValueError, message): candidate.pre_action_receipt(**args)

    def test_masks_affect_residual_only_and_bad_history_rejected(self):
        args = self.receipt_arguments(); args['runtime_mask'] = (0,)*12
        r = candidate.pre_action_receipt(**args)
        self.assertEqual(r['combined_residual_permission_mask_full12'], (0.,)*12)
        self.assertEqual(r['pre_source_nominal_full12'], args['nominal'])
        self.assertEqual(r['actual_live_history'], args['history'])
        args['observation'][195] += .1
        with self.assertRaisesRegex(ValueError, 'live history'): candidate.pre_action_receipt(**args)

    def test_receipt_AST_changes_are_only_explicit2channel_guards(self):
        source = inspect.getsource(previous.pre_action_receipt)
        source = source.replace("delta[1:]", "delta[2:]")
        before = """    selector = (1,) + (0,) * 11 if intervention and intervention.get('requested_selected_deg') else (0,) * 12
    require(not any(delta) or selector[0] == 1, 'changed action lacks intervention receipt')
    require(not selector[0] or intervention.get('case') == 'FL_minus6', 'wrong finite intervention case')"""
        after = """    selected = intervention.get('requested_selected_deg', {}) if intervention else {}
    require(not intervention or intervention.get('case') == 'FL_hip_minus1_knee_plus1',
            'wrong finite intervention case')
    require(not selected or set(selected) == {0, 1}, 'expected both FL hip/knee selected')
    require(not selected or intervention.get('delta_deg') == {0: -1.0, 1: 1.0},
            'wrong finite hip/knee offsets')
    selector = (1, 1) + (0,) * 10 if selected else (0,) * 12
    require(not any(delta) or bool(selected), 'changed action lacks intervention receipt')
    require(not selected or all(math.isclose(candidate_after[i], selected[i], rel_tol=1e-12, abs_tol=1e-10)
                               for i in (0, 1)), 'selected physical request differs from actual injected raw')"""
        self.assertIn(before, source)
        expected = ast.parse(source.replace(before, after)).body[0]
        self.assertEqual(ast.dump(expected), ast.dump(function_ast(candidate, 'pre_action_receipt')))

    def test_main_AST_only_case_names_and_explicit_manifest_fields_changed(self):
        before, after = function_ast(previous, 'main'), function_ast(candidate, 'main')
        strings = {'FL_minus6': CASE,
            'wlr50_clean.task_direction_probe_FL_minus6_candidate.v1': 'wlr50_clean.task_direction_probe_FL_hip_minus1_knee_plus1_candidate.v1',
            'FL_minus6: anchor-6deg; ramp12; release at entry-age75 or capture-age+24; release12; follow30':
                CASE+': ACK-anchor hip-1deg/knee+1deg; ramp12; release at entry-age75 or capture-age+24; release12; follow30'}
        for node in ast.walk(before):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in strings: node.value = strings[node.value]
            if isinstance(node, ast.Name) and node.id in ('FiniteFLMinus6','trigger_minus6'):
                node.id = {'FiniteFLMinus6':'FiniteFLHipKnee','trigger_minus6':'trigger_hip_knee'}[node.id]
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'probe' and node.func.attr == 'start':
                self.assertEqual(ast.dump(node.args[0]), ast.dump(ast.parse("pre_history['previous_residual_full12']", mode='eval').body))
                node.args[0] = ast.parse('verified_ack_anchor(pre_history, jsonable(backend._adapter.last_ack))', mode='eval').body
            if isinstance(node, ast.keyword) and node.arg == 'finite_intervention':
                node.value = ast.IfExp(test=ast.parse("args.case != 'control'", mode='eval').body,
                    body=node.value, orelse=ast.Constant(value='control: unmodified live deterministic policy; no intervention'))
        removed = {}
        for node in ast.walk(after):
            if isinstance(node, ast.Call):
                for kw in list(node.keywords):
                    if kw.arg in ('manual_channels','manual_offsets_deg','same_tick_live_channels','anchor_source'):
                        self.assertIsInstance(kw.value, ast.IfExp)
                        self.assertEqual(ast.dump(kw.value.test), ast.dump(ast.parse("args.case != 'control'", mode='eval').body))
                        removed[kw.arg] = (ast.literal_eval(kw.value.body), ast.literal_eval(kw.value.orelse)); node.keywords.remove(kw)
        self.assertEqual(removed, dict(manual_channels=(('front_left_hip','front_left_knee'),()),
            manual_offsets_deg=({'front_left_hip':-1.,'front_left_knee':1.},{}), same_tick_live_channels=(10,12),
            anchor_source=('previous_actual_ACK.independent_policy_residual_requested_full12_verified_equal_live_HISTORY',None)))
        self.assertEqual(ast.dump(before), ast.dump(after))

    def test_real_CP194560_metadata_uses_official_unchanged_validator(self):
        cp = candidate.OUT/'checkpoints/history/checkpoint_step_000194560.pt'
        metadata = json.loads(cp.with_name(cp.stem+'_manifest.json').read_text())
        candidate.validate_checkpoint_metadata(metadata, cp, metadata['runtime_contract'])
        self.assertIs(candidate.validate_checkpoint_metadata, shared.validate_checkpoint_metadata)
        self.assertIs(candidate.seal_step_exception, shared.seal_step_exception)


if __name__ == '__main__':
    unittest.main()
