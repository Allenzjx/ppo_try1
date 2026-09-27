"""CPU mapper/atomic dispatch replay for new all12 finish owner, no Isaac."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'tests/unit'))
from test_actuator_target_effect import _adapter
from test_semantic_post_rr_front_prep_native import dispatch, REQUEST, ZERO
from test_semantic_finish_advance_source import finish
from wlr50_clean.ppo.actuator_target_effect import build_actuator_target_effect_audit, ActuatorTargetEffectError


class FinishNativeTests(unittest.TestCase):
    def test_all_four_wheels_dispatch_same_reference_counterfactual_and_tamper(self):
        adapter=_adapter()
        for tick in range(1,81):dispatch(adapter,REQUEST,tick)
        accepted=deepcopy(adapter)
        entry=list(adapter.last_ack['drive_target_full12'])
        request=list(adapter.last_ack['independent_policy_residual_requested_full12'])
        f=finish(activation_tick=80,reference_tick=80,observation_tick=80,
            reference_final_full12=entry,reference_request_full12=request)
        s=dict(post_rr_active=True,post_rr_preparing=False,post_rr_prepared=True,
            post_rr_prep_exhausted=False,post_rr_touch_tick=60,observation_tick=80,
            post_rr_entry_final_full12=entry,post_rr_entry_residual_full12=request,
            finish_context=f)
        ack,inputs=dispatch(adapter,REQUEST,81,s)
        self.assertEqual(ack['drive_target_full12'],entry)
        audit=build_actuator_target_effect_audit(**inputs)
        self.assertTrue(audit['post_rr_preparation_native_targets_independently_reconstructed'])
        self.assertEqual(ack['post_rr_front_preparation_evidence']['owner_indices'],list(range(12)))
        zero_ack,zero_inputs=dispatch(deepcopy(accepted),ZERO,81,s)
        zero_audit=build_actuator_target_effect_audit(**zero_inputs)
        self.assertEqual(audit['counterfactual_native_targets'],zero_audit['actual_native_targets'])
        f['observation_tick']=81;s['observation_tick']=81
        second,inputs2=dispatch(adapter,REQUEST,82,s)
        for i in range(8,12):
            self.assertAlmostEqual(second['drive_target_full12'][i],entry[i]+.015)
        self.assertEqual(second['drive_target_full12'][:8],entry[:8])
        count=(adapter.write_count,adapter.servo_target_mapper.feedback_tick,len(adapter.robot.events))
        build_actuator_target_effect_audit(**inputs2)
        self.assertEqual(count,(adapter.write_count,adapter.servo_target_mapper.feedback_tick,len(adapter.robot.events)))
        self.assertEqual(second['independent_policy_residual_requested_full12'],
                         list(inputs2['actuation'].projected_residual_full12))
        for key,index in (('candidate_after_full12',10),('reference_requested_full12',11),
                          ('rebased_requested_full12',4)):
            altered=deepcopy(second)
            altered['post_rr_front_preparation_evidence'][key][index]+=.125
            with self.assertRaises(ActuatorTargetEffectError):
                build_actuator_target_effect_audit(**dict(inputs2,raw_ack=altered))
        self.assertEqual(adapter.robot.events.count('dispatch'),82)


if __name__=='__main__':unittest.main(verbosity=2)
