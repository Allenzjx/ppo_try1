"""Stdlib-only task arithmetic tests; no simulation or learned-policy evidence."""
from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace
import ast
import sys
import unittest


MODULE = "rr_capture_local_task_stdlib_test_target"
SPEC = spec_from_file_location(MODULE, Path(__file__).resolve().parents[2]
    / "src/wlr50_clean/ppo/semantic_rr_capture_local_task.py")
local = module_from_spec(SPEC)
sys.modules[MODULE] = local
SPEC.loader.exec_module(local)


def observed(tick, *, crossed=True, qualified=True, within=True, gap=.055,
             top=False, bearing=False, placed=False, samples=0, terminal=None,
             ground=False, outside=0., final=(8., -58.), phase="P09",
             established=None, continuation=True):
    drive = [0.] * 12
    drive[6:8] = final
    # Synthetic evaluator facts, never inferred from historical placed/crossed.
    # Default GROUND revokes establishment; legal loaded TOP with current AIR
    # validity false must explicitly provide established=True.
    if established is None:
        established = qualified and not ground
    rr = dict(current_lift_valid=qualified, lift_established=established,
        active_attempt=established, motion_continuation_allowed=continuation,
        air=not top and not ground, ground_contact=ground, obstacle_pair_active=top,
        within_top_xy=within, within_lateral_span=True, clearance_m=gap,
        top_xy_outside_distance_m=outside, top_contact=top, top_surface_contact=top,
        contact_surface="TOP" if top else "NONE", support=bearing,
        bearing_verified=True, bearing_force_n=3. if bearing else 0.,
        consecutive_top_samples=samples, load_fraction=.2 if bearing else 0.,
        load_fraction_valid=True)
    ev = dict(valid=True, physics_tick=tick, simulation_time_s=tick/120.,
        termination_reason=terminal, current_legs={"RR": rr},
        history=dict(front_edge_crossed={"RR": crossed}, placed={"RR": placed}))
    info = dict(semantic_task=dict(physical_evaluator=ev, stage_id=phase,
                                  termination_reason=terminal),
        drive_target_full12=drive, atomic_ack=dict(drive_target_full12=list(drive)),
        nominal_action_full12=[77.] * 12,
        raw_observation=dict(actual_full12=[99.] * 12))
    return SimpleNamespace(info=info, physics_tick=tick, sim_time_s=tick/120.)


class RRCaptureLocalTaskTests(unittest.TestCase):
    def test_real_gate_and_committed_final_anchor_are_immutable(self):
        task = local.RRCaptureLocalTask()
        self.assertEqual(task.obs9(), (0.,)*9)
        for tick, changes in enumerate((dict(crossed=False), dict(qualified=False),
                                       dict(within=False), dict(top=True, bearing=True, placed=True))):
            self.assertFalse(task.observe(observed(tick, **changes))["active"])
        frame = observed(4)
        original = deepcopy(frame.info)
        result = task.observe(SimpleNamespace(frame=frame))
        self.assertEqual(frame.info, original)
        self.assertTrue(result["active"])
        self.assertEqual(result["obs9"][2:5], (8./180., -58./180., .055/.1))
        task.observe(observed(5, final=(-20., -20.), qualified=False))
        self.assertTrue(task.active)
        self.assertEqual(task.entry_rr_hip_deg, 8.)
        self.assertEqual(task.entry_rr_knee_deg, -58.)

    def test_clock_and_ack_validation(self):
        task = local.RRCaptureLocalTask()
        frame = observed(10)
        task.observe(frame)
        self.assertEqual(task.observe(frame), task.snapshot())
        bad = observed(10, gap=.01)
        with self.assertRaisesRegex(ValueError, "conflicting"):
            task.observe(bad)
        with self.assertRaisesRegex(ValueError, "reset"):
            task.observe(observed(0))
        bad = observed(11)
        bad.info["atomic_ack"]["drive_target_full12"][6] = 9.
        with self.assertRaisesRegex(ValueError, "ACK"):
            task.observe(bad)
        bad = observed(11)
        bad.physics_tick = 12
        with self.assertRaisesRegex(ValueError, "clocks"):
            task.observe(bad)

    def test_no_historical_placed_gate_for_first_contact_reward(self):
        task = local.RRCaptureLocalTask()
        task.observe(observed(0))
        before = task.snapshot()
        task.observe(observed(1, top=True, bearing=True, placed=False, samples=1, gap=0.))
        reward = task.reward(before)
        self.assertGreater(reward["reward"], 0.)
        self.assertEqual(task.obs9()[5:7], (1., 1.))
        self.assertFalse(reward["rr_subtask_success"])
        self.assertTrue(reward["terminal_bootstrap_allowed"])

    def test_continuous_capture_hold_success_and_terminal_potential(self):
        task = local.RRCaptureLocalTask()
        task.observe(observed(0))
        for tick in range(1, 61):
            task.observe(observed(tick, top=True, bearing=True, placed=tick >= 2,
                                  samples=tick, gap=0.))
        self.assertFalse(task.local_success)
        before = task.snapshot()
        task.observe(observed(61, top=True, bearing=True, placed=True, samples=61, gap=0.))
        reward = task.reward(before)
        self.assertTrue(reward["rr_subtask_success"])
        self.assertEqual(reward["termination_reason"], local.SUCCESS)
        self.assertEqual(reward["potential_after"], 0.)
        self.assertFalse(reward["terminal_bootstrap_allowed"])
        self.assertFalse(reward["full_task_success"])
        self.assertAlmostEqual(reward["potential_shaping"], -5.*before["potential"])
        self.assertGreater(reward["reward"], 30.)

    def test_drop_load_and_missing_ticks_reset_hold_not_policy_gate(self):
        task = local.RRCaptureLocalTask()
        task.observe(observed(0))
        for tick in range(1, 25):
            task.observe(observed(tick, top=True, bearing=True, placed=True, samples=tick))
        self.assertGreater(task.hold_elapsed_s, 0.)
        task.observe(observed(25, top=True, bearing=False, placed=True, samples=25))
        self.assertEqual(task.hold_elapsed_s, 0.)
        self.assertTrue(task.active)
        task.observe(observed(26, top=True, bearing=True, placed=True, samples=26))
        task.observe(observed(90, top=True, bearing=True, placed=True, samples=90))
        self.assertEqual(task.hold_elapsed_s, 0.)
        self.assertFalse(task.local_success)
        task.observe(observed(91, ground=True, placed=True))
        self.assertEqual(task.snapshot()["potential"], 0.)
        self.assertTrue(task.active)
        task.reset()
        self.assertFalse(task.active)
        self.assertEqual(task.obs9(), (0.,)*9)
        self.assertIsNone(task.entry_rr_hip_deg)
        task.observe(observed(0, crossed=False))

    def test_prefix_activation_decision_is_not_new_sample(self):
        task = local.RRCaptureLocalTask()
        task.observe(observed(0, crossed=False))
        before = task.snapshot()
        task.observe(observed(8))
        result = task.reward(before)
        self.assertFalse(result["on_policy_rr_sample"])
        self.assertEqual(result["reward"], 0.)
        self.assertTrue(result["prefix_excluded"])

    def test_one_decision_discount_and_budget_bootstrap(self):
        task = local.RRCaptureLocalTask()
        task.observe(observed(0))
        before = task.snapshot()
        for tick in range(1, 9):
            task.observe(observed(tick, gap=.030, phase="P10"))
        reward = task.reward(before)
        self.assertAlmostEqual(reward["potential_shaping"],
            5.*(.9985*task.snapshot()["potential"]-before["potential"]))
        self.assertFalse(reward["terminated"])
        self.assertFalse(reward["truncated"])
        self.assertTrue(reward["terminal_bootstrap_allowed"])

    def test_true_failure_zeroes_potential_and_does_not_bootstrap(self):
        task = local.RRCaptureLocalTask()
        task.observe(observed(0))
        before = task.snapshot()
        task.observe(observed(1, terminal="BODY_COLLISION"))
        reward = task.reward(before)
        self.assertLess(reward["reward"], -40.)
        self.assertEqual(reward["potential_after"], 0.)
        self.assertFalse(reward["terminal_bootstrap_allowed"])
        self.assertFalse(reward["rr_subtask_success"])

    def test_numerical_terminal_keeps_invalid_measurements_unavailable(self):
        task = local.RRCaptureLocalTask()
        task.observe(observed(0))
        before = task.snapshot()
        frame = observed(1, terminal="NAN_INF", gap=float("nan"))
        frame.info["raw_observation"]["actual_full12"][6] = float("nan")
        task.observe(frame)
        self.assertIsNone(task.metrics["actual_rr_hip_knee_deg"])
        self.assertIsNone(task.metrics["gap_m"])
        result = task.reward(before)
        self.assertEqual(result["termination_reason"], "NAN_INF")
        self.assertFalse(result["terminal_bootstrap_allowed"])

    def test_geometry_rewards_descent_not_air_height_or_penetration(self):
        task = local.RRCaptureLocalTask()
        task.observe(observed(0))
        phi = task.snapshot()["potential"]
        task.observe(observed(1, gap=.1))
        self.assertLess(task.snapshot()["potential"], phi)
        task.observe(observed(2, gap=0.))
        at_surface = task.snapshot()["potential"]
        task.observe(observed(3, gap=-.01))
        self.assertEqual(task.snapshot()["potential"], at_surface)
        task.observe(observed(4, gap=-.1))
        self.assertEqual(task.snapshot()["potential_components"]["gap_closure"], 0.)
        task.observe(observed(5, within=False, outside=.03, gap=.001))
        self.assertEqual(task.snapshot()["potential_components"]["gap_closure"], 0.)

    def test_rl_and_angle_changes_do_not_enter_rr_potential(self):
        task = local.RRCaptureLocalTask()
        task.observe(observed(0))
        expected = task.snapshot()["potential"]
        frame = observed(1, final=(-30., 40.))
        frame.info["semantic_task"]["physical_evaluator"]["current_legs"]["RL"] = {
            "clearance_m": 100., "bearing_force_n": 500., "current_lift_valid": True}
        frame.info["semantic_task"]["task_progress_potential"] = 100.
        task.observe(frame)
        self.assertEqual(task.snapshot()["potential"], expected)
        self.assertEqual(task.entry_rr_hip_deg, 8.)

    def test_bearing_requires_actual_top_force_and_verification(self):
        task = local.RRCaptureLocalTask()
        task.observe(observed(0))
        frame = observed(1, top=True, bearing=True, placed=True, samples=200)
        frame.info["semantic_task"]["physical_evaluator"]["current_legs"]["RR"]["bearing_force_n"] = .1
        task.observe(frame)
        self.assertEqual(task.obs9()[5:7], (1., 0.))
        self.assertFalse(task.local_success)
        frame = observed(2, top=True, bearing=True, placed=True, samples=201)
        frame.info["semantic_task"]["physical_evaluator"]["current_legs"]["RR"]["bearing_verified"] = False
        task.observe(frame)
        self.assertFalse(task.metrics["current_top_bearing"])

    def test_stdlib_only_and_fixed_gamma(self):
        tree = ast.parse(Path(local.__file__).read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(item.name.split(".")[0] for item in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module.split(".")[0])
        self.assertTrue(imported <= {"__future__", "collections", "copy", "dataclasses", "math"})
        with self.assertRaises(ValueError):
            local.RRCaptureLocalTaskConfig(gamma=.99)
        with self.assertRaises(ValueError):
            local.RRCaptureLocalTaskConfig(physics_dt_s=1./60.)


if __name__ == "__main__":
    unittest.main()
