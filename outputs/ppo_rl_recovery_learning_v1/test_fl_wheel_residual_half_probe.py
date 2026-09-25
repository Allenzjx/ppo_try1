"""Stdlib-only wiring tests. No production import, model or physics execution."""
import ast
from copy import deepcopy
import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

PATH=Path(__file__).with_name('fl_wheel_residual_half_probe.py')
sys.path.insert(0,str(PATH.parent))
SPEC=importlib.util.spec_from_file_location('fl_wheel_residual_half_probe',PATH)
M=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def raw(value=-1.):
    values=[i/20. for i in range(12)];values[8]=value
    return tuple(values)


def source_event(*,stage='P12',channels=None,wheel_values=(0.,)*4,tick=99):
    group=NS(channels=M.WHEELS if channels is None else channels,time_s=2.5)
    sample=NS(atomic_groups=(group,),full12=(0.,)*8+wheel_values,tick_index=301)
    provider=NS(_continuous_layers=[dict(stage=stage,advanced_this_tick=True,sample=sample)])
    frame=NS(physics_tick=tick,nominal_action_full12=(0.,)*8+wheel_values)
    return provider,frame


def dispatch(stop,*,before=99,command=500,nominal=(0.,)*12,ack_change=None,audit_change=None):
    ack=dict(articulation_writes_this_call=1,physics_tick=command)
    audit=dict(verified=True,setter_dispatch_targets_equal=True,
        actual_mapping_matches_dispatch=True,same_tick_counterfactual=True,physics_tick=command)
    ack.update(ack_change or {});audit.update(audit_change or {})
    return stop.observe_dispatch(before_tick=before,after_tick=before+1,
        source_nominal=nominal,ack=ack,audit=audit)


def issued_stop():
    stop=M.SourceStopReceipt();provider,frame=source_event()
    stop.pending=M.scheduled_wheel_event(provider,frame)
    dispatch(stop)
    return stop


def task():
    leg=dict(air=False,ground_contact=False,support=True,bearing_verified=True,
        top_contact=True,top_surface_contact=True,obstacle_pair_active=True,
        contact_surface='TOP',within_top_xy=True,within_lateral_span=True,
        bearing_force_n=8.,clearance_m=.003)
    return dict(stage_id='P12',termination_reason=None,physical_evaluator=dict(
        valid=True,physics_tick=100,termination_reason=None,
        current_legs={name:deepcopy(leg) for name in ('RR','FL')},
        history=dict(placed=dict(RR=True,FL=True))))


def evidence(value=None,**kwargs):
    args=dict(phase='P12',tick=100,nominal=(0.,)*12,stop=issued_stop(),
        force_floor=.2,top_gap=(-.015,.012))
    args.update(kwargs)
    return M.eligibility(task() if value is None else value,**args)


class RawTransform(unittest.TestCase):
    def test_only_negative_fl_tanh_is_halved(self):
        for value in (-1e-9,-.3,-2.,-1000.):
            with self.subTest(value=value):
                original=raw(value);applied=M.half_negative_fl_request(original)
                self.assertAlmostEqual(math.tanh(applied[8]),.5*math.tanh(value),places=14)
                self.assertLess(applied[8],0.)
                self.assertEqual([i for i in range(12) if original[i]!=applied[i]],[8])
                self.assertEqual(original,raw(value))

    def test_positive_and_zero_all12_unchanged(self):
        for value in (0.,.001,4.):
            self.assertEqual(M.half_negative_fl_request(raw(value)),raw(value))

    def test_fresh_request_not_last_output_is_used(self):
        probe=M.OneShot();original=raw();ev=evidence()
        first,_=probe.choose(original,100,ev);second,_=probe.choose(original,108,ev)
        self.assertEqual(first,second)
        self.assertNotEqual(M.half_negative_fl_request(first),second)

    def test_bad_inputs_fail_even_while_waiting(self):
        for bad in ((0.,)*11,(0.,)*13,raw(float('nan')),raw(float('inf')),raw(True)):
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                M.OneShot().choose(bad,100,dict(eligible=False,blockers=['none']))


class StopReceipt(unittest.TestCase):
    def test_scheduled_stop_is_not_issued_until_next_verified_dispatch(self):
        provider,frame=source_event();before=deepcopy(provider)
        stop=M.SourceStopReceipt();stop.pending=M.scheduled_wheel_event(provider,frame)
        self.assertEqual(provider,before)
        self.assertFalse(evidence(stop=stop)['eligible'])
        dispatch(stop)
        self.assertTrue(evidence(stop=stop)['eligible'])
        self.assertEqual(stop.receipt['issued_episode_tick'],100)
        self.assertEqual(stop.receipt['source_frame_tick'],99)
        self.assertEqual(stop.receipt['command_physics_tick'],500)
        self.assertIsNone(stop.pending)

    def test_zero_nominal_without_explicit_event_is_insufficient(self):
        stop=M.SourceStopReceipt();dispatch(stop)
        self.assertFalse(evidence(stop=stop)['eligible'])

    def test_no_live_event_no_false_stop(self):
        provider,frame=source_event();provider._continuous_layers[0]['advanced_this_tick']=False
        self.assertIsNone(M.scheduled_wheel_event(provider,frame))

    def test_other_phase_partial_or_conflicting_wheel_owner_fails_closed(self):
        for variation in ({'stage':'P09'},{'channels':M.WHEELS[:1]},
                          {'wheel_values':(-.3,)*4}):
            provider,frame=source_event(**variation)
            event=M.scheduled_wheel_event(provider,frame)
            self.assertFalse(event['explicit_P12_fourwheel_stop'])
        provider,frame=source_event()
        provider._continuous_layers.append(deepcopy(provider._continuous_layers[0]))
        self.assertFalse(M.scheduled_wheel_event(provider,frame)['explicit_P12_fourwheel_stop'])

    def test_stale_source_or_unverified_dispatch_rejected(self):
        stop=M.SourceStopReceipt();provider,frame=source_event(tick=98)
        stop.pending=M.scheduled_wheel_event(provider,frame)
        with self.assertRaises(ValueError):dispatch(stop)
        for key in ('verified','setter_dispatch_targets_equal','actual_mapping_matches_dispatch','same_tick_counterfactual'):
            with self.subTest(key=key),self.assertRaises(ValueError):
                dispatch(M.SourceStopReceipt(),audit_change={key:False})
        with self.assertRaises(ValueError):dispatch(M.SourceStopReceipt(),ack_change={'articulation_writes_this_call':2})
        with self.assertRaises(ValueError):dispatch(M.SourceStopReceipt(),audit_change={'physics_tick':499})

    def test_duplicate_command_tick_rejected(self):
        stop=issued_stop()
        with self.assertRaises(ValueError):dispatch(stop,before=100,command=500)

    def test_stop_retires_on_later_nonzero_nominal_or_other_wheel_event(self):
        stop=issued_stop();dispatch(stop,before=100,command=501,nominal=(0.,)*8+(.2,0.,0.,0.))
        self.assertFalse(stop.active)
        stop=issued_stop();provider,frame=source_event(stage='P11',tick=100)
        stop.pending=M.scheduled_wheel_event(provider,frame)
        dispatch(stop,before=100,command=501)
        self.assertFalse(stop.active)

    def test_snapshot_cannot_mutate_receipt(self):
        stop=issued_stop();snapshot=stop.snapshot();snapshot['receipt']['nominal_wheels'][0]=9.
        self.assertEqual(stop.receipt['nominal_wheels'],[0.]*4)


class CurrentPhysicalEligibility(unittest.TestCase):
    def test_current_top_bearing_allowed_with_no_input_mutation(self):
        value=task();stop=issued_stop();before=deepcopy(value);receipt=stop.snapshot()
        result=evidence(value,stop=stop)
        self.assertTrue(result['eligible']);self.assertEqual(value,before)
        self.assertEqual(stop.snapshot(),receipt)
        result['current']['RR']['support']=False
        self.assertTrue(value['physical_evaluator']['current_legs']['RR']['support'])

    def test_each_current_contact_or_geometry_condition_is_required(self):
        changes=dict(air=True,ground_contact=True,support=False,bearing_verified=False,
            top_contact=False,top_surface_contact=False,obstacle_pair_active=False,
            contact_surface='EDGE',within_top_xy=False,within_lateral_span=False,
            bearing_force_n=.199,clearance_m=.013)
        for leg in ('RR','FL'):
            for key,value in changes.items():
                current=task();current['physical_evaluator']['current_legs'][leg][key]=value
                with self.subTest(leg=leg,key=key):self.assertFalse(evidence(current)['eligible'])
        for force in (None,True,float('nan')):
            current=task();current['physical_evaluator']['current_legs']['RR']['bearing_force_n']=force
            self.assertFalse(evidence(current)['eligible'])

    def test_history_placed_air_is_not_support(self):
        current=task();current['physical_evaluator']['current_legs']['RR'].update(
            air=True,support=False,top_contact=False,bearing_force_n=0.)
        self.assertTrue(current['physical_evaluator']['history']['placed']['RR'])
        self.assertFalse(evidence(current)['eligible'])

    def test_stage_freshness_safety_and_current_nominal_required(self):
        for kwargs in ({'phase':'P11'},{'tick':101},{'nominal':(0.,)*8+(-.3,)*4}):
            self.assertFalse(evidence(**kwargs)['eligible'])
        for container,key,value in (('task','termination_reason','HARD_JOINT_LIMIT'),
                ('task','stage_id','P11'),('ev','valid',False),
                ('ev','termination_reason','BODY_COLLISION')):
            current=task();target=current if container=='task' else current['physical_evaluator']
            target[key]=value;self.assertFalse(evidence(current)['eligible'])


class FiniteSingleIntervention(unittest.TestCase):
    def test_positive_wait_does_not_consume_one_shot(self):
        probe=M.OneShot();ev=evidence()
        self.assertEqual(probe.choose(raw(.5),100,ev)[1],[])
        self.assertEqual(probe.state['status'],'WAIT')
        self.assertEqual(probe.choose(raw(-.5),108,ev)[1],[8])
        self.assertEqual(probe.state['status'],'ACTIVE')

    def test_first_physics_loss_latched_no_next_override_never_rearms(self):
        probe=M.OneShot();ev=evidence();probe.choose(raw(),100,ev)
        probe.observe(103,dict(eligible=False,blockers=['RR_left_TOP']))
        self.assertEqual(probe.state['status'],'RELEASE_PENDING')
        self.assertEqual(probe.choose(raw(),108,ev),(raw(),[]))
        self.assertEqual(probe.state['release_requested_tick'],103)
        self.assertEqual(probe.state['release_effective_tick'],108)
        self.assertEqual(probe.choose(raw(),116,ev),(raw(),[]))

    def test_three_seconds_is_at_most_45_full_decisions(self):
        probe=M.OneShot();ev=evidence()
        for tick in range(100,460,8):self.assertEqual(probe.choose(raw(),tick,ev)[1],[8])
        self.assertEqual(probe.state['issued_override_decisions'],45)
        self.assertEqual(probe.choose(raw(),460,ev),(raw(),[]))
        self.assertEqual(probe.state['status'],'RELEASED')
        self.assertEqual(probe.state['release_effective_tick']-probe.state['start_tick'],360)

    def test_fractional_last_decision_not_issued(self):
        probe=M.OneShot(.1);ev=evidence()
        self.assertEqual(probe.choose(raw(),100,ev)[1],[8])
        self.assertEqual(probe.choose(raw(),108,ev)[1],[])

    def test_duration_validation_and_terminal_finish(self):
        for value in (0.,.06,3.01,float('nan'),float('inf')):
            with self.assertRaises(ValueError):M.OneShot(value)
        probe=M.OneShot();probe.choose(raw(),100,evidence());probe.finish(103,'native_safety')
        self.assertEqual(probe.state['status'],'RELEASED')
        self.assertEqual(probe.state['release_effective_tick'],103)
        self.assertEqual(probe.choose(raw(),104,evidence()),(raw(),[]))


class RuntimeBoundary(unittest.TestCase):
    def test_default_invocation_refuses_before_paths_imports_or_launch(self):
        before=set(sys.modules)
        with self.assertRaisesRegex(ValueError,'NOT EXECUTED'):
            M.main(['--checkpoint','does-not-exist.pt','--run-dir','does-not-exist',
                '--checkpoint-sha256','0'*64,'--checkpoint-manifest-sha256','0'*64,
                '--expected-head','0'*40])
        added=set(sys.modules)-before
        self.assertFalse(any(name.startswith(('torch','pxr','isaac','wlr50_clean')) for name in added))

    def test_all_training_and_teacher_credit_zero(self):
        for key in ('new_PPO_samples','new_PPO_updates','new_optimizer_steps',
                    'new_AUX_accepted_updates','new_AUX_attempted_steps',
                    'teacher_prefix_sample_credit','teacher_sample_credit'):
            self.assertEqual(M.ZERO_CREDIT[key],0)
        self.assertFalse(M.ZERO_CREDIT['policy_credit'])
        self.assertFalse(M.ZERO_CREDIT['formal_deterministic_policy_result'])
        self.assertFalse(M.ZERO_CREDIT['diagnostic_rows_entered_in_on_policy_storage'])

    def test_suffix_audit_uses_actual_applied_raw(self):
        applied=M.half_negative_fl_request(raw())
        info=dict(raw_policy_action_full12=applied,physics_ticks=8,
            actuator_target_effect_audit=dict(raw_policy_action_full12=applied),
            actuator_target_effect_audit_summary=dict(all_ticks_verified=True,physics_ticks=8),
            no_in_episode_state_writes_verified=True)
        M.verified_suffix_step(info,applied)
        for key,value in (('raw_policy_action_full12',raw()),
                          ('no_in_episode_state_writes_verified',False)):
            altered=deepcopy(info);altered[key]=value
            with self.assertRaises(ValueError):M.verified_suffix_step(altered,applied)
        altered=deepcopy(info);altered['actuator_target_effect_audit']['raw_policy_action_full12']=raw()
        with self.assertRaises(ValueError):M.verified_suffix_step(altered,applied)

    def test_static_lifecycle_reuses_one_core_without_production_control_mutation(self):
        tree=ast.parse(PATH.read_text(encoding='utf-8'))
        calls=[node for node in ast.walk(tree) if isinstance(node,ast.Call)]
        names=[ast.unparse(node.func) for node in calls]
        self.assertEqual(names.count('build_video_core'),1)
        self.assertEqual(names.count('prefix.reset'),1)
        self.assertEqual(names.count('prefix.install'),1)
        self.assertEqual(names.count('core.step'),1)
        self.assertEqual(names.count('core.reset'),0)
        self.assertNotIn('core._history.update',names)
        self.assertNotIn('runner.alg.update',names)
        self.assertNotIn('optimizer.step',names)
        self.assertNotIn('torch.load',names)
        self.assertIn('_CheckpointPolicyCreditCore',names)
        self.assertIn('checkpoint_loader',names)


if __name__=='__main__':unittest.main(verbosity=2)
