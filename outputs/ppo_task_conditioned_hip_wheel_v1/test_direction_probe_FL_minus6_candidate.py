"""CPU-only candidate checks; never imports Isaac, launches physics, or trains."""
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import unittest

import direction_probe as old
import direction_probe_preaction_candidate as prior
import direction_probe_FL_minus6_candidate as new
from test_direction_probe_preaction_candidate import fixture


def function_ast(module, name):
    return next(x for x in ast.parse(Path(module.__file__).read_text()).body
                if isinstance(x, ast.FunctionDef) and x.name == name)


def event(placed=False):
    return dict(valid=True, termination_reason=None, current_legs={'FL':dict(air=True, within_top_xy=True)},
                history={'placed':dict(FL=placed, FR=True, RR=False, RL=False),
                         'front_edge_crossed':dict(FL=True)})


class IsolatedCandidateTests(unittest.TestCase):
    def test_old_source_bytes_are_unchanged(self):
        expected = {old:'47d49535071af1b0f76cd13a249675a2c8cb93575b1b9751e3c702cdd16c1167',
                    prior:'c7c1eef885df4cccda650e81175fb73a30b6a327e5a3abbd76adaa682597690d'}
        for module, digest in expected.items():
            self.assertEqual(hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(), digest)

    def test_copied_receipt_ast_differs_only_in_explicit_case(self):
        before = function_ast(prior, 'pre_action_receipt')
        matches = [n for n in ast.walk(before) if isinstance(n, ast.Constant) and n.value == 'FL_minus3']
        self.assertEqual(len(matches), 1)
        matches[0].value = 'FL_minus6'
        self.assertEqual(ast.dump(before), ast.dump(function_ast(new, 'pre_action_receipt')))

    def test_main_ast_only_declared_name_and_manifest_changes(self):
        before, after = function_ast(prior, 'main'), function_ast(new, 'main')
        strings = {'FL_minus3':'FL_minus6',
            'wlr50_clean.task_direction_probe_preaction_candidate.v1':'wlr50_clean.task_direction_probe_FL_minus6_candidate.v1',
            'unchanged FL_minus3: -3deg, 12-decision quintic ramp, old hold/release/tail':
                'FL_minus6: anchor-6deg; ramp12; release at entry-age75 or capture-age+24; release12; follow30'}
        for n in ast.walk(before):
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value in strings:
                n.value = strings[n.value]
            if isinstance(n, ast.Name) and n.id in ('FiniteDirection','trigger'):
                n.id = {'FiniteDirection':'FiniteFLMinus6','trigger':'trigger_minus6'}[n.id]
        removed = {}
        for n in ast.walk(after):
            if isinstance(n, ast.Call):
                for k in list(n.keywords):
                    if k.arg in ('auxiliary_loss','positive_auxiliary_labels_emitted','shared_preaction_helpers_sha256'):
                        removed[k.arg] = k.value
                        n.keywords.remove(k)
        self.assertEqual(removed['auxiliary_loss'].value, 0.)
        self.assertEqual(removed['positive_auxiliary_labels_emitted'].value, 0)
        self.assertEqual(set(removed), {'auxiliary_loss','positive_auxiliary_labels_emitted','shared_preaction_helpers_sha256'})
        self.assertEqual(ast.dump(before), ast.dump(after))

    def test_numerical_state_machine_and_global_cases_not_changed(self):
        self.assertIs(new.FiniteFLMinus6.apply, old.FiniteDirection.apply)
        self.assertIs(new.FiniteFLMinus6.start, old.FiniteDirection.start)
        self.assertIs(new.FiniteFLMinus6.complete, old.FiniteDirection.complete)
        cases = copy.deepcopy(old.CASES)
        p = new.FiniteFLMinus6('FL_minus6')
        self.assertEqual(p.offsets, {0:-6.})
        self.assertEqual(old.CASES, cases)
        self.assertNotIn('FL_minus6', old.CASES)
        with self.assertRaises(ValueError): new.FiniteFLMinus6('FL_minus3')

    def test_trigger_uses_same_actual_endpoint_and_physical_predicates(self):
        good = event()
        self.assertTrue(new.trigger_minus6('FL_minus6','P05',good,True))
        for phase, ep, ev in [('P02',True,good),('P05',False,good),('P05',True,event(True))]:
            self.assertFalse(new.trigger_minus6('FL_minus6',phase,ev,ep))
        for key,value in [('air',False),('within_top_xy',False)]:
            ev = copy.deepcopy(good);ev['current_legs']['FL'][key] = value
            self.assertFalse(new.trigger_minus6('FL_minus6','P05',ev,True))
        ev = copy.deepcopy(good);ev['termination_reason']='BODY_COLLISION'
        self.assertFalse(new.trigger_minus6('FL_minus6','P05',ev,True))
        self.assertFalse(new.trigger_minus6('control','P05',good,True))

    def test_fixed_anchor_not_recursively_subtracted_and_other11_are_live(self):
        p = new.FiniteFLMinus6('FL_minus6')
        anchor = [2.4404078965400906]+[.2]*11
        saved = tuple(anchor);p.start(anchor);anchor[0] = 100.
        records=[]
        for i in range(118):
            baseline=tuple(.001*(i+j) for j in range(12));caps=(18.,)*8+(.6,)*4
            raw, info = p.apply(baseline,caps,'P05',event())
            self.assertEqual(raw[1:],baseline[1:])
            self.assertEqual(info['case'],'FL_minus6')
            if 'index' in info:
                self.assertEqual(info['anchor_full12'],saved)
                self.assertEqual(info['delta_deg'],{0:-6.})
                self.assertTrue(info['not_added_recursively_to_HISTORY'])
            records.append((raw,info))
        for i in range(11,75):
            self.assertAlmostEqual(18*math.tanh(records[i][0][0]),saved[0]-6.,places=12)
        self.assertEqual(p.release_age,75)
        self.assertTrue(p.complete)
        self.assertEqual(records[117][0],baseline)
        self.assertEqual(records[117][1]['state'],'intervention_finished_selected_baseline_continues')

    def test_capture24_release12_and_tail30_are_preserved(self):
        p = new.FiniteFLMinus6('FL_minus6');p.start((0.,)*12)
        for i in range(39+12+30):
            p.apply((.02,)*12,(18.,)*12,'P05',event(i>=15))
            if i < 38: self.assertIsNone(p.release_age)
            if i < 80: self.assertFalse(p.complete)
        self.assertEqual(p.capture_age,15)
        self.assertEqual(p.release_age,39)
        self.assertTrue(p.complete)

    def test_control_is_exact_pass_through_without_anchor(self):
        p = new.FiniteFLMinus6('control');raw=tuple(.01*i for i in range(12))
        self.assertEqual(p.apply(raw,(18.,)*12,'P05',event()),(raw,None))

    def test_phase_cap_change_does_not_reset_or_rescale_physical_anchor(self):
        p=new.FiniteFLMinus6('FL_minus6');p.start((2.44,)+(0.,)*11)
        for _ in range(12):p.apply((.01,)*12,(18.,)*12,'P05',event())
        original_anchor=p.anchor
        raw,receipt=p.apply((.02,)*12,(32.,)*12,'P06',event(True))
        self.assertEqual(p.anchor,original_anchor)
        self.assertEqual(receipt['index'],12)
        self.assertAlmostEqual(32*math.tanh(raw[0]),2.44-6.,places=12)
        self.assertEqual(raw[1:],(.02,)*11)

    def test_capacity_and_nonfinite_rejection_are_preserved(self):
        for first in (-18.,float('nan')):
            p=new.FiniteFLMinus6('FL_minus6');p.start((first,)+(0.,)*11)
            with self.assertRaisesRegex(ValueError,'existing residual capacity'):
                p.apply((0.,)*12,(18.,)*12,'P05',event())

    def receipt_arguments(self):
        args=fixture();p=new.FiniteFLMinus6('FL_minus6')
        p.start(args['history']['previous_residual_full12'])
        for _ in range(12):
            raw, intervention=p.apply(args['baseline'],args['caps'],'P05',event())
        args.update(injected=raw,intervention=intervention)
        return args

    def test_receipt_label_hash_history_and_no_training_credit(self):
        args=self.receipt_arguments();snapshot=copy.deepcopy(args)
        r=new.pre_action_receipt(**args)
        self.assertEqual(args,snapshot)
        self.assertEqual(r['intervention']['case'],'FL_minus6')
        self.assertEqual(r['actual_live_history'],args['history'])
        self.assertEqual(r['manual_raw_delta_full12'][1:],(0.,)*11)
        self.assertEqual(r['manual_override_selector_full12'],(1,)+(0,)*11)
        self.assertAlmostEqual(r['candidate_physical_after_intervention_before_permission_and_slew_full12'][0],.01-6.)
        for key in ('new_PPO_decisions','new_PPO_updates','new_optimizer_steps'):self.assertEqual(r[key],0)
        digest=r.pop('pre_action_receipt_sha256')
        payload=json.dumps(r,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        self.assertEqual(digest,hashlib.sha256(payload).hexdigest())

    def test_wrong_case_or_knee_override_rejected_not_relabelled(self):
        args=self.receipt_arguments();args['intervention']['case']='FL_minus3'
        with self.assertRaisesRegex(ValueError,'wrong finite intervention case'):new.pre_action_receipt(**args)
        args=self.receipt_arguments();raw=list(args['injected']);raw[1]+=.01;args['injected']=raw
        with self.assertRaisesRegex(ValueError,'unselected channel'):new.pre_action_receipt(**args)

    def test_masks_still_apply_only_to_residual_and_no_history_reset(self):
        args=self.receipt_arguments();args['runtime_mask']=(0,)*12
        r=new.pre_action_receipt(**args)
        self.assertEqual(r['combined_residual_permission_mask_full12'],(0.,)*12)
        self.assertEqual(r['pre_source_nominal_full12'],args['nominal'])
        self.assertEqual(r['actual_live_history'],args['history'])
        self.assertNotEqual(r['actual_live_history']['previous_residual_full12'],(0.,)*12)

    def test_real_CP189952_binding_uses_existing_loader_validation(self):
        cp=new.OUT/'checkpoints/history/checkpoint_step_000189952.pt'
        metadata=json.loads(cp.with_name(cp.stem+'_manifest.json').read_text())
        new.validate_checkpoint_metadata(metadata,cp,metadata['runtime_contract'])
        self.assertIs(new.validate_checkpoint_metadata,prior.validate_checkpoint_metadata)
        self.assertIs(new.seal_step_exception,prior.seal_step_exception)


if __name__ == '__main__':
    unittest.main()
