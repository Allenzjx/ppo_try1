"""Focused source/reference wiring; synthetic states are not physics success."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from wlr50_clean.ppo import semantic_post_rr_front_prep_source as M
from wlr50_clean.ppo.semantic_headroom import HEADROOM_MODE
from wlr50_clean.infrastructure.command_batch import WHEEL_FORWARD_SIGN
from wlr50_clean.fsm.motion_executor import MotionExecutor
from wlr50_clean.reference.motion_contract import load_motion_contract

ZERO = (0.,) * 12
CONTRACT = load_motion_contract(ROOT/'configs/recording_motion_contract.json')


def state(**changes):
    final = [0.] * 12; final[1] = -58.; final[3] = -20.; final[8] = -2.; final[9] = .2
    request = [0.] * 12; request[1] = -32.; request[3] = -51.; request[8] = -1.
    result = dict(post_rr_active=True, post_rr_preparing=True, post_rr_prepared=False,
        post_rr_prep_exhausted=False, post_rr_touch_tick=100, observation_tick=100,
        post_rr_entry_final_full12=final, post_rr_entry_residual_full12=request,
        active=True, rr_touch_seen=True, rr_support_continuation_valid=True,
        rr_geometry_ready_for_RL_prep=True)
    result.update(changes)
    return result


def project(s=None, *, request=None, native=None, previous=None):
    s = state() if s is None else s
    previous = tuple(s['post_rr_entry_final_full12']) if previous is None else tuple(previous)
    request = tuple(s['post_rr_entry_residual_full12']) if request is None else tuple(request)
    ack = dict(physics_tick=279, write_count=101, drive_target_full12=previous,
               independent_policy_residual_requested_full12=s['post_rr_entry_residual_full12'])
    return M.final_reference_candidate((90.,)*12, native_full12=ZERO if native is None else native,
        controller_full12=ZERO, residual_full12=request,
        context=M.make_context(s,dispatch_physics_tick=280), previous_ack=ack,
        previous_tick=279,write_count=101,policy_headroom_mode=HEADROOM_MODE)


class ReferenceTests(unittest.TestCase):
    def test_both_knees_and_wheels_start_from_final_in_same_tick(self):
        r=project(); out=r['candidate_after_full12']
        self.assertEqual(out[1],-57.75); self.assertEqual(out[3],-19.75)
        self.assertAlmostEqual(out[8],-1.985); self.assertAlmostEqual(out[9],.215)
        self.assertEqual(out[6:8],[90.,90.])
        self.assertEqual(r['rebased_requested_full12'][1],0.)
        self.assertFalse(r['HISTORY_reset'])

    def test_old_large_negative_residual_is_rebased_not_reclipped(self):
        s=state(observation_tick=460)
        final=list(s['post_rr_entry_final_full12']);final[1]=final[3]=30.;final[8]=final[9]=1.
        r=project(s,previous=final)
        self.assertEqual([r['candidate_after_full12'][i] for i in M.PREP_INDICES],[30.,30.,1.,1.])
        self.assertEqual(r['requested_full12'][1],-32.)

    def test_current_policy_change_still_controls_one_axis_without_broadcast(self):
        s=state(); changed=list(s['post_rr_entry_residual_full12']);changed[1]+=2.
        base=project(s)['candidate_after_full12']; modified=project(s,request=changed)['candidate_after_full12']
        self.assertEqual(modified[1]-base[1],2.)
        self.assertEqual([i for i in range(12) if modified[i]!=base[i]],[1])

    def test_original_headroom_still_blocks_outward_but_allows_inward(self):
        s=state();q=list(s['post_rr_entry_residual_full12']);q[1]-=200.
        self.assertEqual(project(s,request=q)['candidate_after_full12'][1],-58.)
        q[0]=-200.;self.assertEqual(project(s,request=q)['candidate_after_full12'][0],-133.)

    def test_prepared_exits_plus1_but_keeps_reference_and_original_source(self):
        s=state(post_rr_preparing=False,post_rr_prepared=True,observation_tick=460)
        previous=[0.]*12;previous[8]=previous[9]=1.
        native=[0.]*12;native[1]=-31.4;native[3]=31.1
        r=project(s,native=native,previous=previous)
        self.assertEqual(r['candidate_after_full12'][1:4:2],[-31.4,31.1])
        self.assertEqual(r['candidate_after_full12'][8:10],[.985,.985])
        self.assertEqual(r['paused_strong_indices'],[])

    def test_exhausted_exits_wheel_preparation_not_false_ready(self):
        s=state(post_rr_prep_exhausted=True)
        old=list(s['post_rr_entry_final_full12']);old[1]=old[3]=20.;old[8]=old[9]=1.
        r=project(s,previous=old)
        self.assertEqual(r['paused_strong_indices'],[0,4])
        self.assertFalse(r['front_pair_nominal_candidate'])
        self.assertEqual([r['candidate_after_full12'][i] for i in (1,3,8,9)],[20.,20.,.985,.985])

    def test_configured_candidate_and_rates_reach_context(self):
        s=state(post_rr_candidate_knee_deg=32.,post_rr_candidate_wheel_rad_s=.8,
                post_rr_knee_rate_deg_s=24.,post_rr_wheel_rate_rad_s2=1.2)
        r=project(s)
        self.assertAlmostEqual(r['candidate_after_full12'][1],-57.8)
        self.assertAlmostEqual(r['candidate_after_full12'][8],-1.99)
        self.assertEqual(r['context']['candidate_knee_deg'],32.)
        self.assertEqual(r['context']['candidate_wheel_rad_s'],.8)

    def test_rebased_FL_knee_keeps_current_support_pause_after_preparation(self):
        s=state(post_rr_preparing=False,post_rr_prepared=True)
        c=M.make_context(s,dispatch_physics_tick=280);c['post_rr_source_pause_indices']=[0,1,4]
        ack=dict(physics_tick=279,write_count=101,drive_target_full12=[20.]*12,
                 independent_policy_residual_requested_full12=s['post_rr_entry_residual_full12'])
        r=M.final_reference_candidate(ZERO,native_full12=ZERO,controller_full12=ZERO,
            residual_full12=s['post_rr_entry_residual_full12'],context=c,previous_ack=ack,
            previous_tick=279,write_count=101,policy_headroom_mode=HEADROOM_MODE)
        self.assertEqual(r['candidate_after_full12'][1],20.)
        self.assertEqual(r['rebased_axes_with_current_support_pause'],[1])

    def test_canonical_native_mapping_not_abs_speed(self):
        self.assertEqual(WHEEL_FORWARD_SIGN['front_left_ankle'],-1.)
        self.assertEqual(WHEEL_FORWARD_SIGN['front_right_ankle'],1.)
        s=state(post_rr_preparing=False,post_rr_prepared=True)
        previous=[0.]*12;previous[8]=-.9
        native=[0.]*12;native[8]=-1.
        self.assertLess(project(s,native=native,previous=previous)['candidate_after_full12'][8],-.9)

    def test_invalid_or_nonadjacent_anchor_rejected(self):
        s=state();s['post_rr_touch_tick']=101
        with self.assertRaises(ValueError):project(s)
        s=state();s['post_rr_entry_residual_full12'][1]=float('nan')
        with self.assertRaises(ValueError):project(s)


def physical_task():
    top=dict(ground_contact=False,air=False,support=True,bearing_verified=True,
        bearing_force_n=10.,top_contact=True,top_surface_contact=True,contact_surface='TOP',
        obstacle_pair_active=True,within_top_xy=True,within_lateral_span=True)
    ground=dict(top,top_contact=False,top_surface_contact=False,contact_surface='GROUND',
        ground_contact=True,obstacle_pair_active=False)
    return dict(stage_id='P12',termination_reason=None,physical_evaluator=dict(valid=True,
        termination_reason=None,physics_tick=100,simulation_time_s=100/120.,
        current_legs=dict(RR=top,FL=top,FR=top,RL=ground),history={'placed':{'RR':True}}))


class DummyBase:
    def __init__(self,**kwargs):
        self._read_local_state=kwargs['read_local_state'];self.calls=[]
        self._p09_late_source=(5.333333333,5.400000000000851)
        self._rear_policy_timing_mode='rr_live_swing_evidence_v3'
        self.spec={'support':{'force_noise_floor_n':.2}}
        self.nominal_full12=CONTRACT.phase('P09').start_full12
    def _sequence_permission(self,layer,task,obs):
        self.calls.append(layer['stage']);return True
    def continuation_pause_inputs(self,task):return {'owned_indices':[0,1,4]}


def source_provider(s):
    return M.provider_type(DummyBase)(read_post_rr_state=lambda:s,read_local_state=lambda:s)


class SourceTests(unittest.TestCase):
    def test_before_touch_all_stages_delegate_without_mutation(self):
        s=state(post_rr_active=False,post_rr_preparing=False)
        p=source_provider(s)
        for stage in ('P01','P05','P09','P10','P11','P12'):
            l={'stage':stage};before=copy.deepcopy(l)
            self.assertTrue(p._sequence_permission(l,{},None));self.assertEqual(l,before)
        self.assertEqual(len(p.calls),6)

    def test_pending_late_clock_does_not_consume_then_prepared_releases(self):
        s=state();p=source_provider(s);phase=CONTRACT.phase('P09')
        motion=MotionExecutor(initial_full12=phase.start_full12);motion.start_phase(phase)
        for _ in range(648):sample=motion.tick()
        p.nominal_full12=sample.full12
        l=dict(stage='P09',motion=motion,ticks=648,sample=sample)
        for _ in range(20):
            self.assertTrue(p._sequence_permission(l,physical_task(),None))
            l['sample']=l['motion'].tick();l['ticks']+=1
        self.assertEqual(l['motion'].original._tick_index,648)
        self.assertFalse(l['motion'].strong_started)
        s.update(post_rr_preparing=False,post_rr_prepared=True)
        self.assertTrue(p._sequence_permission(l,physical_task(),None))
        self.assertEqual(p.calls,['P09'])

    def test_P12_started_pulse_stop_clock_continues_without_RL_joint_launch(self):
        s=state();p=source_provider(s);phase=CONTRACT.phase('P12')
        motion=MotionExecutor();motion.start_phase(phase)
        l=dict(stage='P12',motion=motion,ticks=56,rl_ticks=32,rl_sample=None)
        self.assertFalse(p._sequence_permission(l,physical_task(),None))
        self.assertTrue(l['rl_dependency_wait'])
        l['ticks']=57
        self.assertTrue(p._sequence_permission(l,physical_task(),None))
        self.assertTrue(l['rl_dependency_wait'])

    def test_P09_already_started_pulse_keeps_exact_stop_while_strong_paused(self):
        s=state();p=source_provider(s);phase=CONTRACT.phase('P09')
        motion=MotionExecutor(initial_full12=phase.start_full12);motion.start_phase(phase)
        for _ in range(648):sample=motion.tick()
        p.nominal_full12=sample.full12
        carrier=M.ContinuousLateCarrier(motion,previous_sample=sample,pending_event_tick=648,
                                       read_nominal=lambda:p.nominal_full12)
        carrier.permissions(preparation=True,strong=True);sample=carrier.tick()
        self.assertEqual(sample.full12[8],-1.07)
        l=dict(stage='P09',motion=carrier,ticks=649,sample=sample)
        for _ in range(216):
            self.assertTrue(p._sequence_permission(l,physical_task(),None))
            l['sample']=carrier.tick();l['ticks']+=1
        self.assertEqual(carrier.source_stop_ticks,[864])
        self.assertEqual(l['sample'].full12[8:],(0.,)*4)

    def test_actual_factory_wraps_existing_source_without_reset(self):
        s=state(post_rr_active=False,post_rr_preparing=False)
        factory=M.controller_factory(task_spec_path=ROOT/'configs/ppo_rr_capture_first_cp225280_v1/stage_task_spec.yaml',
            read_local_state=lambda:s,read_post_rr_state=lambda:s)
        controller=factory(ROOT/'configs/fsm_states.yaml',ROOT/'configs/recording_motion_contract.json')
        self.assertFalse(controller.nominal_provider.post_rr_preparation_inputs()['post_rr_active'])
        self.assertEqual(controller.supervisor.read_continuation_source_state(),{})

    def test_new_reference_removes_only_FL_knee_from_old_pause(self):
        s=state();p=source_provider(s)
        self.assertEqual(p.continuation_pause_inputs({})['owned_indices'],[0,4])


if __name__=='__main__':unittest.main(verbosity=2)
