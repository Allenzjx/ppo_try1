"""Pure target arithmetic only. NOTEXECUTED refers to physical diagnostic."""
from copy import deepcopy
import math
import unittest

from target_plan import OneShotRRDirection, qualified_carry


def context(tick=100, age=0.):
    zero = [0.] * 12
    final = zero.copy(); final[6], final[7] = 5., -20.
    nominal = zero.copy(); nominal[7] = -35.
    previous = zero.copy(); previous[6], previous[7] = 5., 15.
    ev = {"valid": True, "termination_reason": None, "physical_evidence_status": "VERIFIED", "current_legs": {
        leg: dict(support=True, bearing_verified=True, air=False, ground_contact=True,
                  top_surface_contact=False, bearing_force_n=1.) for leg in ("FL", "FR", "RL")}}
    ev["current_legs"]["RR"] = dict(current_lift_valid=True, motion_continuation_allowed=True,
        body_control_evidence=True, air=True, contact_mode="AIR", ground_contact=False,
        obstacle_pair_active=False, within_lateral_span=True, within_top_xy=True,
        front_distance_m=.02, clearance_m=.035)
    return dict(dispatch_tick=tick, same_tick_mapped_tick=tick, previous_ack_tick=tick-1,
        physics_dt_s=1/120, sim_time_s=50.+age, phase="P09", evaluation=ev,
        minimum_other_supports=2, force_noise_floor_n=.2, previous_final_full12=final,
        actual_full12=final.copy(), same_tick_mapped_nominal_full12=nominal,
        controller_bias_full12=zero.copy(), policy_projected_residual_full12=[.1*i for i in range(12)],
        previous_effective_residual_full12=previous, residual_caps_full12=[100.]*8+[1.]*4,
        policy_raw_full12=[.02*i for i in range(12)], policy_log_probability=-4.,
        servo_hard_limits_deg=[(-135.,135.)]*8,
        headroom_residual_intervals_servo_deg=[(-100.,100.)]*8,
        residual_rate_deg_s=60., final_slew_deg_per_tick=.5, rr_owner_ids=("P09carryHip","P09carryKnee"))


class TargetPlanTests(unittest.TestCase):
    def test_hold_is_previous_actual_final_not_raw_zero(self):
        p, c = OneShotRRDirection(), context()
        result, log = p.plan(c)
        self.assertEqual(log["desired_final_servo_deg"][7], -20.)
        self.assertEqual(result[7], 15.)  # Not zero: -35 mapped +15 correction=-20 final.
        self.assertEqual(result[6], 5.)
        self.assertEqual(log["original_policy_raw_full12"], tuple(c["policy_raw_full12"]))
        self.assertIsNone(log["diagnostic_injected_raw"])
        for i in range(12):
            if i not in (6, 7):
                self.assertEqual(result[i], c["policy_projected_residual_full12"][i])

    def test_same_tick_nominal_and_controller_are_both_compensated(self):
        p, c = OneShotRRDirection(), context()
        p.plan(c)
        c["dispatch_tick"] += 1; c["same_tick_mapped_tick"] += 1; c["previous_ack_tick"] += 1
        c["sim_time_s"] += 1/120
        c["same_tick_mapped_nominal_full12"][7] += .2
        c["controller_bias_full12"][7] += .1
        result, log = p.plan(c)
        self.assertAlmostEqual(result[7], 14.7)
        self.assertAlmostEqual(-34.8+.1+result[7], -20.)

    def test_previous_dispatch_nominal_is_rejected_not_approximated(self):
        c = context(); c["same_tick_mapped_tick"] -= 1
        with self.assertRaises(ValueError):
            OneShotRRDirection().plan(c)

    def test_duplicate_tick_is_idempotent(self):
        p, c = OneShotRRDirection(), context()
        first = p.plan(c)
        self.assertEqual(p.plan(deepcopy(c)), first)
        self.assertEqual(p.anchor["previous_ack_tick"], 99)

    def test_current_qualified_air_not_old_history_is_required(self):
        c = context()
        c["evaluation"]["history"] = {"active_lift": {"RR": True}, "placed": {"RR": True}}
        c["evaluation"]["current_legs"]["RR"]["current_lift_valid"] = False
        self.assertFalse(qualified_carry(c["evaluation"], "P09", 2, .2))
        self.assertIsNone(OneShotRRDirection().plan(c)[0])

    def test_hip_is_one_shot_absolute_trajectory_not_accumulated_delta(self):
        p, c = OneShotRRDirection(), context()
        for k in range(362):
            c["dispatch_tick"] = 100+k; c["same_tick_mapped_tick"] = 100+k
            c["previous_ack_tick"] = 99+k; c["sim_time_s"] = 50.+k/120
            result, log = p.plan(c)
            self.assertIsNotNone(result, log)
            c["previous_final_full12"][6] = log["desired_final_servo_deg"][6]
            c["actual_full12"][6] = c["previous_final_full12"][6]
            c["previous_effective_residual_full12"] = list(result)
        self.assertAlmostEqual(log["desired_final_servo_deg"][6], -20.)
        self.assertEqual(log["desired_final_servo_deg"][7], -20.)

    def test_contact_owner_or_physical_abort_releases_and_never_retriggers(self):
        for fault in ("contact", "owner", "safety"):
            p, c = OneShotRRDirection(), context()
            p.plan(c)
            c["dispatch_tick"] += 1; c["same_tick_mapped_tick"] += 1; c["previous_ack_tick"] += 1
            c["sim_time_s"] += 1/120
            if fault == "contact":
                c["evaluation"]["current_legs"]["RR"]["air"] = False
            elif fault == "owner":
                c["rr_owner_ids"] = ("P10hip", "P10knee")
            else:
                c["physical_abort"] = True
            result, log = p.plan(c)
            self.assertIsNone(result)
            self.assertEqual(log["state"], "RELEASED")
            c = context(102, 2/120)
            self.assertIsNone(p.plan(c)[0])
            self.assertTrue(p.plan(c)[1]["continue_same_episode_with_original_policy"])

    def test_unreachable_hold_releases_instead_of_clipping_or_claiming_hold(self):
        for fault in ("cap", "rate", "headroom", "slew"):
            p, c = OneShotRRDirection(), context()
            p.plan(c)
            c["dispatch_tick"] += 1; c["same_tick_mapped_tick"] += 1; c["previous_ack_tick"] += 1
            c["sim_time_s"] += 1/120
            if fault == "cap":
                c["residual_caps_full12"][7] = 10.
            elif fault == "rate":
                c["previous_effective_residual_full12"][7] = 0.
            elif fault == "headroom":
                c["headroom_residual_intervals_servo_deg"][7] = (-10., 10.)
            else:
                c["final_slew_deg_per_tick"] = 1e-12
            result, log = p.plan(c)
            self.assertIsNone(result)
            self.assertIn("saturates", log["release_reason"])

    def test_safe_drop_geometry_waits_without_consuming_then_allows_air_not_contact(self):
        for key, bad in (("within_top_xy", False), ("front_distance_m", -.001),
                         ("clearance_m", -.001), ("front_distance_m", math.nan)):
            p, c = OneShotRRDirection(), context()
            c["evaluation"]["current_legs"]["RR"][key] = bad
            self.assertIsNone(p.plan(c)[0])
            self.assertEqual(p.state, "WAIT")
            self.assertIsNone(p.anchor)
            c = context(101, 1/120)
            result, log = p.plan(c)
            self.assertIsNotNone(result)
            self.assertEqual(log["anchor"]["previous_ack_tick"], 100)
            self.assertTrue(c["evaluation"]["current_legs"]["RR"]["air"])
            self.assertFalse(c["evaluation"]["current_legs"]["RR"]["obstacle_pair_active"])

    def test_unexpressible_full_candidate_or_knee_waits_and_anchors_later_actual_final(self):
        for fault in ("hip_endpoint", "hip_headroom", "knee_cap", "initial_rate"):
            p, c = OneShotRRDirection(), context()
            c["residual_caps_full12"][6:8] = [24., 36.]
            if fault == "hip_endpoint":
                c["same_tick_mapped_nominal_full12"][6] = 65.
                c["previous_final_full12"][6] = 65.
                c["previous_effective_residual_full12"][6] = 0.
            elif fault == "hip_headroom":
                c["headroom_residual_intervals_servo_deg"][6] = (-10., 24.)
            elif fault == "knee_cap":
                c["same_tick_mapped_nominal_full12"][7] = -58.
            else:
                c["previous_effective_residual_full12"][7] = 0.
            result, log = p.plan(c)
            self.assertIsNone(result)
            self.assertEqual(log["state"], "WAIT")
            self.assertFalse(log["one_shot_consumed"])
            self.assertIsNone(p.anchor)
            c = context(101, 1/120)
            c["residual_caps_full12"][6:8] = [24., 36.]
            c["previous_final_full12"][7] = -19.
            c["previous_effective_residual_full12"][7] = 16.
            result, log = p.plan(c)
            self.assertIsNotNone(result)
            self.assertEqual(log["anchor"]["knee_hold_deg"], -19.)
            self.assertEqual(log["anchor"]["previous_ack_tick"], 100)

    def test_safe_drop_uses_current_measured_support_and_active_geometry_loss_releases(self):
        c = context()
        for leg in ("FR", "FL"):
            c["evaluation"]["current_legs"][leg]["bearing_force_n"] = .1
        self.assertFalse(qualified_carry(c["evaluation"], "P09", 2, .2))
        p, c = OneShotRRDirection(), context()
        self.assertIsNotNone(p.plan(c)[0])
        c = context(101, 1/120)
        c["evaluation"]["current_legs"]["RR"]["within_top_xy"] = False
        result, log = p.plan(c)
        self.assertIsNone(result)
        self.assertEqual(log["state"], "RELEASED")
        self.assertIsNone(p.plan(context(102, 2/120))[0])

    def test_no_measured_response_releases_with_zero_ppo_credit(self):
        p, c = OneShotRRDirection(), context()
        for k in range(140):
            c["dispatch_tick"] = 100+k; c["same_tick_mapped_tick"] = 100+k
            c["previous_ack_tick"] = 99+k; c["sim_time_s"] = 50.+k/120
            result, log = p.plan(c)
            if result is None:
                break
            c["previous_final_full12"][6] = log["desired_final_servo_deg"][6]
            c["previous_effective_residual_full12"] = list(result)
        self.assertIn("without_measured", log["release_reason"])
        self.assertEqual(log["new_PPO_updates"], 0)

    def test_maximum_four_seconds_releases_without_ending_episode(self):
        p, c = OneShotRRDirection(), context()
        for k in range(481):
            c["dispatch_tick"] = 100+k; c["same_tick_mapped_tick"] = 100+k
            c["previous_ack_tick"] = 99+k; c["sim_time_s"] = 50.+k/120
            result, log = p.plan(c)
            if result is not None:
                c["previous_final_full12"][6] = log["desired_final_servo_deg"][6]
                c["actual_full12"][6] = c["previous_final_full12"][6]
                c["previous_effective_residual_full12"] = list(result)
        self.assertIsNone(result)
        self.assertEqual(log["release_reason"], "finite_4s_intervention_complete")
        self.assertTrue(log["continue_same_episode_with_original_policy"])


if __name__ == "__main__":
    unittest.main()
