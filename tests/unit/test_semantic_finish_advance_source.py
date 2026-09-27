"""Bounded A-only ownership; synthetic wiring is not physical success."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from wlr50_clean.ppo import semantic_finish_advance_source as F
from wlr50_clean.ppo import semantic_post_rr_front_prep_source as P
from wlr50_clean.ppo.semantic_headroom import HEADROOM_MODE

ZERO=(0.,)*12
HOME=(.5,-.7,3.7,.4,-2.6,-3.9,-.5,-6.)


def finish(**changes):
    result=dict(schema=F.CONTEXT_SCHEMA,active=True,mode=F.ADVANCE,
        activation_tick=100,reference_tick=100,observation_tick=100,
        reference_final_full12=[-18.,-31.,13.,34.,-22.,-14.,-11.,-23.,-.003,-.03,.19,-.08],
        reference_request_full12=[.5,-33.,10.,-50.,-12.,5.,-3.,6.,-1.,-.03,.19,-.08],
        reference_capacities_full12=[32.]*8+[1.2]*4,home_entry_tick=None,
        home_duration_s=0.,home_target_servo_deg=HOME,
        advance_wheel_rad_s=.3,wheel_rate_rad_s2=1.8)
    result.update(changes)
    return result


def state(context=None):
    result=dict(post_rr_active=True,post_rr_preparing=False,post_rr_prepared=True,
        post_rr_prep_exhausted=False,post_rr_touch_tick=80,observation_tick=100,
        post_rr_entry_final_full12=[1.]*12,post_rr_entry_residual_full12=[2.]*12)
    if context is not None:
        result['finish_context']=context
        result['observation_tick']=context.get('observation_tick',100)
    return result


def project(context, *, request=None,previous=None):
    previous=context['reference_final_full12'] if previous is None else previous
    request=context['reference_request_full12'] if request is None else request
    outer=P.make_context(state(context),dispatch_physics_tick=280)
    ack=dict(physics_tick=279,write_count=101,drive_target_full12=previous,
        independent_policy_residual_requested_full12=context['reference_request_full12'])
    return P.final_reference_candidate([90.]*12,native_full12=ZERO,controller_full12=ZERO,
        residual_full12=request,context=outer,previous_ack=ack,previous_tick=279,
        write_count=101,policy_headroom_mode=HEADROOM_MODE)


class DummyBase:
    def __init__(self,read_state):
        self.read_state=read_state;self.calls=0;self.physics_hz=120.
        self.elapsed_s=17.7;self.layer_ticks=2125;self.endpoint_issued=True
        self.nominal_full12=tuple(range(12))
    def post_rr_preparation_inputs(self):return self.read_state()
    def evaluate(self,stage,observation=None):
        self.calls+=1;self.layer_ticks+=1
        return ('old_path',stage,observation)
    @property
    def nominal_suggestion_diagnostics(self):return {'old_field':self.calls}


class FinishSourceTests(unittest.TestCase):
    def test_inactive_context_keeps_old_path_and_serialized_context_exact(self):
        old=state();inactive=state(dict(active=False))
        self.assertEqual(P.make_context(old,dispatch_physics_tick=280),
                         P.make_context(inactive,dispatch_physics_tick=280))
        p=F.provider_type(DummyBase)(read_state=lambda:inactive)
        for phase in ('P01','P05','P09','P12'):
            self.assertEqual(p.evaluate(phase,object())[:2],('old_path',phase))
        self.assertEqual(p.nominal_suggestion_diagnostics,{'old_field':4})

    def test_advance_starts_final_reference_all12_then_four_wheels_ramp(self):
        context=finish();r=project(context)
        self.assertEqual(r['candidate_after_full12'],context['reference_final_full12'])
        self.assertEqual(r['owner_indices'],list(range(12)))
        previous=r['candidate_after_full12']
        for tick in range(101,130):
            context['observation_tick']=tick
            r=project(context,previous=previous)
            self.assertEqual(r['candidate_after_full12'][:8],context['reference_final_full12'][:8])
            for i in range(8,12):
                self.assertLessEqual(abs(r['candidate_after_full12'][i]-previous[i]),.01500000001)
                self.assertGreaterEqual(r['candidate_after_full12'][i],previous[i])
            previous=r['candidate_after_full12']
        self.assertEqual(previous[8:],[.3]*4)
        self.assertEqual(r['requested_full12'],context['reference_request_full12'])
        self.assertFalse(r['HISTORY_reset']);self.assertFalse(r['actual_state_written'])

    def test_single_relative_axis_no_new_local_projection_or_policy_shutdown(self):
        context=finish();baseline=project(context)['candidate_after_full12']
        request=list(context['reference_request_full12']);request[4]+=2.
        r=project(context,request=request)
        self.assertEqual([i for i,(a,b) in enumerate(zip(baseline,r['candidate_after_full12'])) if a!=b],[4])
        self.assertEqual(r['candidate_after_full12'][4]-baseline[4],2.)
        request[4]+=100.
        r=project(context,request=request)
        self.assertEqual(r['rebased_requested_full12'][4],102.)
        self.assertEqual(r['candidate_after_full12'][4]-baseline[4],102.)
        self.assertFalse(r['additional_local_residual_projection'])
        self.assertEqual(r['requested_full12'][4],request[4])

    def test_home_relatched_final_starts_without_jump_and_duration_tracks_travel(self):
        entry=[-18.,-31.,13.,34.,-22.,-14.,-11.,-23.]+[.3]*4
        duration=F.home_duration_s(entry,HOME)
        self.assertGreaterEqual(duration,2.)
        context=finish(mode=F.HOME,reference_tick=300,observation_tick=300,
            home_entry_tick=300,reference_final_full12=entry,home_duration_s=duration)
        self.assertEqual(project(context)['candidate_after_full12'],entry)
        previous=entry
        for tick in range(301,301+int(duration*120)+2):
            context['observation_tick']=tick
            current,_,_=F.reference_baseline(context)
            self.assertLessEqual(max(abs(a-b) for a,b in zip(current[:8],previous[:8]))*120,20.0001)
            previous=current
        self.assertEqual(previous[:8],HOME)
        self.assertEqual(previous[8:],(0.,)*4)

    def test_old_source_stop_home_endpoint_clocks_not_consumed_during_new_advance(self):
        context=finish();s=state(context)
        p=F.provider_type(DummyBase)(read_state=lambda:s)
        old_ticks=p.layer_ticks
        for tick in range(100,130):
            context['observation_tick']=tick
            p.evaluate({'stage_id':'P13'})
        self.assertEqual(p.calls,0);self.assertEqual(p.layer_ticks,old_ticks)
        self.assertEqual(p.elapsed_s,17.7);self.assertFalse(p.endpoint_issued)
        self.assertEqual(p.nominal_full12[:8],tuple(context['reference_final_full12'][:8]))
        self.assertTrue(p.nominal_suggestion_diagnostics['requested_finish']['old_source_clocks_paused'])
        context.update(mode=F.HOME,reference_tick=130,observation_tick=130,home_entry_tick=130,
            reference_final_full12=list(p.nominal_full12),home_duration_s=4.)
        self.assertEqual(p.evaluate('P13'),tuple(context['reference_final_full12']))
        self.assertEqual(p.calls,0)

    def test_invalid_reference_and_existing_physical_headroom_not_relaxed(self):
        context=finish(reference_tick=101)
        with self.assertRaises(ValueError):project(context)
        context=finish();context['reference_final_full12'][0]=132.9
        request=list(context['reference_request_full12']);request[0]+=10.
        self.assertEqual(project(context,request=request)['candidate_after_full12'][0],133.)
        context['wheel_rate_rad_s2']=2.
        with self.assertRaises(ValueError):project(context)


if __name__=='__main__':unittest.main(verbosity=2)
