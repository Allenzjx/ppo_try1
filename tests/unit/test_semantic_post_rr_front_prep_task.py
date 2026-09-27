"""Synthetic wiring/task tests, not claims of Isaac physical success."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"src"))
from wlr50_clean.ppo.semantic_post_rr_front_prep_task import (
    POST_RR_FIELDS, POST_RR_OBSERVATION_DIM, PostRRFrontPrepTask, STAGES,
)
from wlr50_clean.ppo.semantic_rr_capture_continuation_task import RRCaptureContinuationTask

CAPS = [32., 36., 24., 112., 24., 36., 24., 36., 1.2, 1.2, 1., .6]
ENTRY = [0., -58., 0., -20., 0., 0., -10., -31., -1., .1, 0., 0.]
REQUEST = [0., -32., 0., -50., 0., 0., -3., -1., -1., .1, 0., 0.]


def frame(tick, *, top=False, gap=.03, bearing=False, actual=None, final=None,
          rl_lift=False, rl_top=False, full=False, terminal=None, com=(0., 0., .2),
          fr=(.2, -.1, .1), phase="P12", rr_ground=False):
    rr = dict(current_lift_valid=not top and not rr_ground, lift_established=not rr_ground,
        motion_continuation_allowed=True, air=not top and not rr_ground,
        ground_contact=rr_ground, obstacle_pair_active=top, within_top_xy=True,
        within_lateral_span=True, clearance_m=gap, top_xy_outside_distance_m=0.,
        top_contact=top, top_surface_contact=top, contact_surface="TOP" if top else "NONE",
        support=bearing, bearing_verified=True, bearing_force_n=12. if bearing else 0.,
        consecutive_top_samples=2 if top else 0, load_fraction=.2 if bearing else 0.,
        load_fraction_valid=True)
    rl = dict(current_lift_valid=rl_lift, lift_established=rl_lift or rl_top,
        motion_continuation_allowed=True, ground_contact=not(rl_lift or rl_top),
        air=rl_lift and not rl_top, obstacle_pair_active=rl_top, within_top_xy=rl_top,
        within_lateral_span=True, top_contact=rl_top, top_surface_contact=rl_top,
        contact_surface="TOP" if rl_top else "GROUND", support=not rl_lift,
        bearing_verified=True, bearing_force_n=0. if rl_lift else 8.,
        consecutive_top_samples=2 if rl_top else 0, clearance_m=0. if rl_top else .02)
    role = dict(valid=True, diagonal_receiving_side="FR", workspace_progress=.5,
        preparation_progress=.4, transfer_progress=.2,
        receiver_workspace_state=dict(wheel_relative_body_m=(.15, -.1, -.1),
                                      radial_contraction_m=.002))
    ev = dict(valid=True, physics_tick=tick, simulation_time_s=tick/120.,
        termination_reason=terminal, current_legs=dict(RR=rr, RL=rl),
        history=dict(front_edge_crossed=dict(RR=True, RL=rl_top),
                     placed=dict(RR=top, RL=rl_top)), transfer_roles=dict(RL=role),
        success=full, final_region_valid=full, final_controlled=full, final_support_available=full)
    final = list(ENTRY if final is None else final)
    actual = list(ENTRY if actual is None else actual)
    info = dict(semantic_task=dict(stage_id=phase, physical_evaluator=ev,
                                  termination_reason=terminal),
        drive_target_full12=final,
        atomic_ack=dict(drive_target_full12=list(final), physics_tick=tick+179,
                        independent_policy_residual_requested_full12=list(REQUEST)),
        actuator_target_effect_audit=dict(physics_tick=tick+179,
                                         policy_request_phase="P11", source_phase_id=phase),
        raw_observation=dict(actual_full12=actual,
            center_of_mass=dict(valid=True, position_w_m=com),
            wheels=dict(front_right_ankle=dict(center_w_m=fr))))
    return SimpleNamespace(info=info, physics_tick=tick, sim_time_s=tick/120.)


def ready_values(knee=30.):
    value = list(ENTRY)
    value[1] = value[3] = knee
    value[8] = value[9] = 1.
    return value


class PostRRTaskTests(unittest.TestCase):
    def setUp(self):
        self.task = PostRRFrontPrepTask(phase_caps_full12={"P11":CAPS, "P12":CAPS})

    def activate(self):
        self.task.observe(frame(0))
        return self.task.observe(frame(1, top=True, gap=0., bearing=True))

    def prepare(self):
        self.activate()
        q = ready_values()
        return self.task.observe(frame(2, top=True, gap=0., bearing=True, actual=q, final=q))

    def test_pre_touch_exact_accepted_observation_no_new_state_or_writes(self):
        accepted = RRCaptureContinuationTask()
        for tick in range(3):
            f = frame(tick)
            original = deepcopy(f.info)
            accepted.observe(f)
            out = self.task.observe(f)
            self.assertEqual(self.task.obs16(), accepted.obs16())
            self.assertEqual(out["post_rr_obs"], (0.,)*25)
            self.assertFalse(out["post_rr_active"])
            self.assertEqual(out["physical_state_or_target_writes"], 0)
            self.assertEqual(f.info, original)

    def test_touch_latches_once_from_real_ack_with_no_prefix_credit(self):
        self.task.observe(frame(0))
        before = self.task.snapshot()
        out = self.task.observe(frame(1, top=True, gap=0., bearing=True))
        self.assertEqual(out["post_rr_stage"], STAGES[0])
        self.assertEqual(out["post_rr_touch_tick"], 1)
        self.assertEqual(out["post_rr_entry_final_full12"], tuple(ENTRY))
        self.assertEqual(out["post_rr_entry_residual_full12"], tuple(REQUEST))
        self.assertEqual(out["post_rr_entry_provenance"]["phase"], "P11")
        reward = self.task.reward(before)
        self.assertEqual(reward["reward"], 0.)
        self.assertTrue(reward["prefix_excluded"])
        self.assertFalse(reward["terminated"])
        self.assertEqual(POST_RR_OBSERVATION_DIM,25)
        self.assertEqual(POST_RR_FIELDS[14], "entry_request_ratio_FL_knee")
        self.assertAlmostEqual(out["post_rr_obs"][14], -32./36.)

    def test_finish_recovery_is_visible_without_changing_reward_or_contact(self):
        self.prepare()
        other = PostRRFrontPrepTask(phase_caps_full12={"P11":CAPS, "P12":CAPS})
        other.observe(frame(0))
        other.observe(frame(1, top=True, gap=0., bearing=True))
        other.observe(frame(2, top=True, gap=0., bearing=True,
                            actual=ready_values(), final=ready_values()))
        f = frame(3, top=True, gap=0., bearing=True, rl_top=True,
                  actual=ready_values(), final=ready_values(), phase="P13")
        before = self.task.snapshot()
        a = self.task.observe(f)
        f.info["semantic_task"]["physical_evaluator"].update(
            finish_settle_pending=True, finish_endpoint_missed=True,
            finish_clean_elapsed_norm=.375)
        b = other.observe(f)
        self.assertEqual(a["post_rr_obs"][:22], b["post_rr_obs"][:22])
        self.assertEqual(a["post_rr_obs"][22:], (0., 0., 0.))
        self.assertEqual(b["post_rr_obs"][22:], (1., 1., .375))
        self.assertEqual(a["metrics"], b["metrics"])
        self.assertEqual(a["post_rr_potential"], b["post_rr_potential"])
        self.assertEqual(self.task.reward(before), other.reward(before))
        self.assertFalse(other.reward(before)["terminated"])
        other.reset()
        self.assertEqual(other.post_rr_observation(), (0.,)*25)

    def test_near_air_or_historical_touch_does_not_activate(self):
        self.task.observe(frame(0,gap=.007))
        self.assertFalse(self.task.post_rr_active)
        f=frame(1,gap=.007)
        f.info["semantic_task"]["physical_evaluator"]["history"]["placed"]["RR"]=True
        self.task.observe(f)
        self.assertFalse(self.task.post_rr_active)

    def test_short_air_grace_zero_force_and_no_reset_large_loss_not_bearing(self):
        before=self.activate()
        out=self.task.observe(frame(2,gap=.0085))
        self.assertTrue(out["rr_transient_grace"])
        self.assertTrue(out["post_rr_active"])
        self.assertFalse(out["rr_bearing_now"])
        self.assertEqual(out["metrics"]["bearing_force_n"],0.)
        self.assertFalse(self.task.reward(before)["terminated"])
        out=self.task.observe(frame(3,gap=.07))
        self.assertFalse(out["rr_support_continuation_valid"])
        self.assertEqual(out["post_rr_entry_final_full12"],tuple(ENTRY))
        self.assertEqual(out["post_rr_touch_tick"],1)

    def test_final_candidate_without_actual_following_is_not_preparation(self):
        self.activate()
        out=self.task.observe(frame(2,top=True,gap=0.,bearing=True,final=ready_values()))
        self.assertFalse(out["post_rr_prepared"])
        self.assertEqual(out["post_rr_prep_stall_reason"],"front_knee_actual_travel_not_yet_effective")

    def test_functional_candidate_does_not_require_exact_30_or_static_body(self):
        self.activate()
        q=ready_values(25.)
        out=self.task.observe(frame(2,top=True,gap=0.,bearing=True,actual=q,final=q))
        self.assertTrue(out["post_rr_prepared"])
        self.assertEqual(out["post_rr_stage"],STAGES[1])
        self.assertIsNotNone(out["post_rr_fixed_fr_reference"])
        self.assertFalse(out["local_success"])

    def test_bad_support_prevents_preparation_even_at_exact_angles(self):
        self.activate()
        q=ready_values()
        out=self.task.observe(frame(2,gap=.07,actual=q,final=q))
        self.assertFalse(out["post_rr_prepared"])
        self.assertIn("RR_outside",out["post_rr_prep_stall_reason"])

    def test_eight_second_stall_is_diagnostic_not_success_or_done(self):
        self.activate()
        before=self.task.snapshot()
        out=self.task.observe(frame(1000,gap=.07))
        self.assertTrue(out["post_rr_prep_exhausted"])
        self.assertFalse(out["post_rr_prepared"])
        result=self.task.reward(before)
        self.assertFalse(result["terminated"])
        self.assertTrue(result["terminal_bootstrap_allowed"])

    def test_fixed_fr_com_direction_not_receiver_motion_or_forward_only(self):
        before=self.prepare()
        reference=deepcopy(before["post_rr_fixed_fr_reference"])
        q=ready_values()
        out=self.task.observe(frame(3,top=True,gap=0.,bearing=True,actual=q,final=q,
                                   fr=(.4,.2,.1)))
        self.assertEqual(out["post_rr_fixed_fr_reference"],reference)
        self.assertEqual(out["post_rr_fixed_fr_com_progress_m"],0.)
        out=self.task.observe(frame(4,top=True,gap=0.,bearing=True,actual=q,final=q,
                                   com=(0.,-.02,.2),fr=(.4,.2,.1)))
        self.assertGreater(out["post_rr_fixed_fr_com_progress_m"],0.)

    def test_internal_phases_never_done_events_once_and_full_evaluator_success(self):
        self.prepare()
        before=self.task.snapshot()
        out=self.task.observe(frame(3,top=True,gap=0.,bearing=True,rl_lift=True))
        self.assertEqual(out["post_rr_stage"],STAGES[2])
        reward=self.task.reward(before)
        self.assertFalse(reward["terminated"])
        self.assertEqual(reward["milestone_events"]["rl_lift_seen"],10.)
        before=out
        out=self.task.observe(frame(4,top=True,gap=0.,bearing=True,rl_top=True))
        self.assertEqual(out["post_rr_stage"],STAGES[3])
        reward=self.task.reward(before)
        self.assertFalse(reward["terminated"])
        self.assertEqual(reward["milestone_events"]["rl_touch_seen"],20.)
        before=out
        self.task.observe(frame(5,top=True,gap=0.,bearing=True,rl_top=True,full=True))
        self.assertTrue(self.task.reward(before)["full_task_success"])

    def test_safety_and_global_deadline_remain_authoritative(self):
        self.activate()
        before=self.task.snapshot()
        self.task.observe(frame(2,terminal="BODY_COLLISION"))
        reward=self.task.reward(before)
        self.assertTrue(reward["terminated"])
        self.assertEqual(reward["terminal_event"],-40.)
        self.task.reset()
        self.activate()
        before=self.task.snapshot()
        self.task.observe(frame(24000,gap=.07))
        self.assertEqual(self.task.reward(before)["termination_reason"],"GLOBAL_TASK_DEADLINE_200S")

    def test_entry_ack_mismatch_saturation_and_same_tick_conflicts_rejected(self):
        self.task.observe(frame(0))
        f=frame(1,top=True,gap=0.,bearing=True)
        f.info["atomic_ack"]["independent_policy_residual_requested_full12"][1]=36.
        with self.assertRaisesRegex(ValueError,"saturated"):
            self.task.observe(f)
        self.task.reset()
        self.task.observe(frame(0))
        f=frame(1,top=True,gap=0.,bearing=True)
        f.info["actuator_target_effect_audit"]["physics_tick"]+=1
        with self.assertRaisesRegex(ValueError,"clocks disagree"):
            self.task.observe(f)
        self.task.reset()
        f=frame(0)
        self.task.observe(f)
        self.assertEqual(self.task.observe(f),self.task.snapshot())
        f.info["raw_observation"]["actual_full12"][1]+=1.
        with self.assertRaisesRegex(ValueError,"conflicting"):
            self.task.observe(f)


if __name__=="__main__":
    unittest.main()
