"""CPU-only real mapper/float32 buffers and independent preparation replay."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'tests/unit'))
from test_actuator_target_effect import _adapter
from wlr50_clean.infrastructure.command_batch import SERVO_ORDER
from wlr50_clean.ppo.semantic_post_rr_front_prep_source import make_context
from wlr50_clean.ppo.semantic_residual_adapter import apply_semantic_residual
from wlr50_clean.ppo.semantic_headroom import HEADROOM_MODE
from wlr50_clean.ppo.isaac_fsm_backend import build_residual_actuation_plan
from wlr50_clean.ppo.actuator_target_effect import (
    build_actuator_target_effect_audit, ActuatorTargetEffectError)

ZERO=(0.,)*12
NOMINAL=(0.,-41.4,0.,31.1,15.,20.,-6.9,-29.3,0.,0.,0.,0.)
REQUEST=(.3,-32.,-.2,-51.,-.4,.25,-3.5,-1.5,-1.,.2,.05,-.15)


def dispatch(adapter,request,tick,state=None):
    context=make_context(state,dispatch_physics_tick=tick) if state else None
    adapter._semantic_post_rr_prep_pre_dispatch=(dict(context=deepcopy(context),
        previous_ack=deepcopy(adapter.last_ack),previous_tick=adapter._last_physics_tick,
        write_count=adapter.write_count) if context else None)
    previous=tuple(adapter._final_drive_servo_deg[name] for name in SERVO_ORDER)
    actuation=build_residual_actuation_plan(tuple(n+r for n,r in zip(NOMINAL,request)),
        frozen_nominal_full12=NOMINAL,drive_feedback_bias_full12=ZERO,normal_drive_bias_full12=ZERO)
    ack=apply_semantic_residual(adapter,NOMINAL,physics_tick=tick,tracking_servo_names=(),
        controller_bias_full12=ZERO,projected_residual_full12=actuation.projected_residual_full12,
        policy_headroom_mode=HEADROOM_MODE,post_rr_preparation_context=context)
    inputs=dict(adapter=adapter,actuation=actuation,raw_ack=ack,previous_final_drive_servo_deg=previous,
        source_phase_id='P12',policy_request=None,policy_headroom_mode=HEADROOM_MODE)
    return ack,inputs


class NativePreparationTests(unittest.TestCase):
    def test_real_dispatch_ramp_zero_branch_and_tamper_receipts(self):
        adapter=_adapter()
        for tick in range(1,81):dispatch(adapter,REQUEST,tick)
        self.assertEqual(adapter.last_ack['drive_target_full12'][1],-58.)
        before=deepcopy(adapter)
        s=dict(post_rr_active=True,post_rr_preparing=True,post_rr_prepared=False,
            post_rr_prep_exhausted=False,post_rr_touch_tick=80,observation_tick=80,
            post_rr_entry_final_full12=list(adapter.last_ack['drive_target_full12']),
            post_rr_entry_residual_full12=list(adapter.last_ack['independent_policy_residual_requested_full12']))
        ack,inputs=dispatch(adapter,REQUEST,81,s)
        self.assertEqual(ack['drive_target_full12'][1],-57.75)
        self.assertAlmostEqual(ack['drive_target_full12'][3],s['post_rr_entry_final_full12'][3]+.25)
        self.assertAlmostEqual(ack['drive_target_full12'][8],s['post_rr_entry_final_full12'][8]+.015)
        self.assertAlmostEqual(ack['drive_target_full12'][9],s['post_rr_entry_final_full12'][9]+.015)
        count=(adapter.write_count,adapter.servo_target_mapper.feedback_tick,len(adapter.robot.events))
        audit=build_actuator_target_effect_audit(**inputs)
        self.assertEqual(count,(adapter.write_count,adapter.servo_target_mapper.feedback_tick,len(adapter.robot.events)))
        self.assertTrue(audit['post_rr_preparation_native_targets_independently_reconstructed'])
        zero_ack,zero_inputs=dispatch(deepcopy(before),ZERO,81,s)
        zero_audit=build_actuator_target_effect_audit(**zero_inputs)
        self.assertEqual(audit['counterfactual_native_targets'],zero_audit['actual_native_targets'])
        plain_ack,_=dispatch(deepcopy(before),REQUEST,81)
        for i in (2,5,6,7,10,11):
            self.assertEqual(ack['drive_target_full12'][i],plain_ack['drive_target_full12'][i])
        for key in ('candidate_after_full12','reference_requested_full12','rebased_requested_full12'):
            altered=deepcopy(ack);altered['post_rr_front_preparation_evidence'][key][1]+=.125
            with self.assertRaises(ActuatorTargetEffectError):
                build_actuator_target_effect_audit(**dict(inputs,raw_ack=altered))
        self.assertEqual(ack['independent_policy_residual_requested_full12'],
                         list(inputs['actuation'].projected_residual_full12))
        self.assertEqual(ack['nominal_command_servo_deg'],plain_ack['nominal_command_servo_deg'])
        self.assertEqual(adapter.robot.events.count('dispatch'),81)


if __name__=='__main__':unittest.main(verbosity=2)
