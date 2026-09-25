"""Synthetic controller ownership tests; never physical success evidence."""
from __future__ import annotations

from types import SimpleNamespace
import unittest

from wlr50_clean.infrastructure.command_batch import SERVO_ORDER
from wlr50_clean.ppo.semantic_capture_assist import (
    CAPTURE_ASSIST_FEATURE_NAMES, CAPTURE_ASSIST_RETIREMENT_REVISION,
    HipOnlyCaptureAssist, apply_capture_assist_snapshot, capture_assist_context,
    capture_assist_features,
)

DT = 1. / 120.
FINAL = (4.619735086982164, -31.985675796234965) + (0.,) * 10
CANDIDATE = (36.67992073724772, -43.147661960853746,
             7.640675997788197, 16.150073388126557,
             -8.923488274283551, 1.00695342784834,
             17.9552198028117, -22.797107794893613,
             -.7082219117196078, .31034106232945036,
             -.2492865231928772, .26630805785695927)


def exhausted():
    helper = HipOnlyCaptureAssist()
    helper.state.update(mode=3., initialized=1., knee_hold_deg=FINAL[1],
        hip_entry_deg=FINAL[0]+20., hip_target_deg=FINAL[0],
        best_gap_m=-.0012852651660299405, window_start_gap_m=-.0005110078167539084,
        window_elapsed_s=0., hold_elapsed_s=2.0833333333333286,
        release_fraction=0., blocked_reason=5., contact_seen=1.)
    return helper


def context(tick=1, **changes):
    return {
        "dispatch_physics_tick": tick, "stage_id": "P06", "physical_valid": True,
        "within_top_xy": True, "other_support_count": 3, "top_surface_contact": False,
        "obstacle_pair_active": False, "placed_FL": True, "current_FL_air": True,
        "current_FL_support": False, "gap_m": .027858857977039922,
        "hip_actual_deg": FINAL[0], "qualified_FL": True, "crossed_FL": True,
        "source_unfold_dispatched": True, **changes}


def advance(helper, tick=1, previous=FINAL, **changes):
    return helper.advance(context=context(tick, **changes),
        previous_final_full12=previous, physics_dt_s=DT)


class P06CaptureRetirementTests(unittest.TestCase):
    def test_exact_exhausted_placed_air_releases_actual_final_once(self):
        helper = exhausted()
        before = dict(helper.state)
        # Deliberately differ from internal target: final clamp/slew may have acted.
        actual_final = (FINAL[0]+.2, FINAL[1]-.3) + FINAL[2:]
        receipt = advance(helper, previous=actual_final)
        self.assertEqual(receipt["state_before"]["mode_name"], "BLOCKED")
        self.assertEqual(helper.snapshot()["mode_name"], "RELEASE")
        self.assertEqual(helper.state["hip_target_deg"], actual_final[0])
        self.assertEqual(helper.state["knee_hold_deg"], actual_final[1])
        self.assertAlmostEqual(helper.state["release_fraction"], DT/.75)
        for key in ("hip_entry_deg", "best_gap_m", "window_start_gap_m",
                    "window_elapsed_s", "hold_elapsed_s", "contact_seen"):
            self.assertEqual(helper.state[key], before[key])
        fraction = helper.state["release_fraction"]
        applied = apply_capture_assist_snapshot(CANDIDATE, helper.snapshot())
        self.assertAlmostEqual(applied[0], actual_final[0]+fraction*(CANDIDATE[0]-actual_final[0]))
        self.assertAlmostEqual(applied[1], actual_final[1]+fraction*(CANDIDATE[1]-actual_final[1]))
        self.assertEqual(applied[2:], CANDIDATE[2:])
        advance(helper, 2, previous=CANDIDATE)
        self.assertEqual(helper.state["hip_target_deg"], actual_final[0])
        self.assertEqual(helper.state["knee_hold_deg"], actual_final[1])

    def test_guard_does_not_release_pending_initial_or_contact_support_cases(self):
        cases = (
            {"stage_id": "P05"}, {"placed_FL": False},
            {"current_FL_air": False}, {"current_FL_support": True},
            {"current_FL_air": None}, {"current_FL_support": None},
            {"physical_valid": False}, {"top_surface_contact": True},
            {"obstacle_pair_active": True},
        )
        for changes in cases:
            with self.subTest(changes=changes):
                helper = exhausted()
                advance(helper, **changes)
                self.assertNotIn(helper.state["mode"], (4., 5.))

    def test_no_new_release_for_other_block_reason_or_hold(self):
        for mode, reason in ((3., 4.), (3., 6.), (2., 0.), (1., 0.)):
            with self.subTest(mode=mode, reason=reason):
                helper = exhausted()
                helper.state.update(mode=mode, blocked_reason=reason)
                advance(helper)
                self.assertNotIn(helper.state["mode"], (4., 5.))

    def test_top_hold_never_enters_release_on_this_guard(self):
        helper = exhausted()
        advance(helper, top_surface_contact=True, obstacle_pair_active=True,
                current_FL_air=False, current_FL_support=True)
        self.assertEqual(helper.snapshot()["mode_name"], "HOLD")
        self.assertEqual(apply_capture_assist_snapshot(CANDIDATE, helper.snapshot())[:2], FINAL[:2])

    def test_initial_p05_and_unplaced_p06_capture_still_take_ownership(self):
        for stage in ("P05", "P06"):
            helper = HipOnlyCaptureAssist()
            advance(helper, stage_id=stage, placed_FL=False)
            self.assertEqual(helper.snapshot()["mode_name"], "DESCEND")
            self.assertEqual(helper.state["hip_entry_deg"], FINAL[0])
            self.assertEqual(helper.state["hip_target_deg"], FINAL[0])
            advance(helper, 2, stage_id=stage, placed_FL=False)
            self.assertLess(helper.state["hip_target_deg"], FINAL[0])
            self.assertEqual(helper.state["knee_hold_deg"], FINAL[1])

    def test_exhausted_unplaced_p06_remains_pending_not_released(self):
        helper = exhausted()
        for tick in range(1, 110):
            advance(helper, tick, placed_FL=False)
            self.assertEqual(helper.snapshot()["mode_name"], "BLOCKED")
            self.assertEqual(helper.state["blocked_reason"], 5.)
            self.assertEqual(helper.state["hip_target_deg"], FINAL[0])

    def test_p07_existing_release_and_released_no_rearm(self):
        helper = exhausted()
        # Existing P07 path is independent of placement/AIR and keeps its anchors.
        advance(helper, previous=CANDIDATE, stage_id="P07", placed_FL=False,
                current_FL_air=False, current_FL_support=True)
        self.assertEqual(helper.state["hip_target_deg"], FINAL[0])
        self.assertEqual(helper.state["knee_hold_deg"], FINAL[1])
        for tick in range(2, 94):
            advance(helper, tick, stage_id="P07")
        self.assertEqual(helper.snapshot()["mode_name"], "RELEASED")
        advance(helper, 94, stage_id="P06", placed_FL=False)
        self.assertEqual(helper.snapshot()["mode_name"], "RELEASED")
        self.assertEqual(apply_capture_assist_snapshot(CANDIDATE, helper.snapshot()), CANDIDATE)

    def test_new_release_reaches_released_without_resetting_history(self):
        helper = exhausted()
        for tick in range(1, 94):
            advance(helper, tick)
        self.assertEqual(helper.snapshot()["mode_name"], "RELEASED")
        self.assertEqual(helper.state["contact_seen"], 1.)
        advance(helper, 94, placed_FL=False, top_surface_contact=True,
                obstacle_pair_active=True, current_FL_air=False, current_FL_support=True)
        self.assertEqual(helper.snapshot()["mode_name"], "RELEASED")

    def test_exact_replay_uses_existing_twelve_numeric_state_fields(self):
        helper = exhausted()
        before = helper.snapshot()
        receipt = advance(helper)
        replay = HipOnlyCaptureAssist()
        replay.state.update({name: before[name] for name in CAPTURE_ASSIST_FEATURE_NAMES})
        repeated = advance(replay)
        self.assertEqual(receipt, repeated)
        self.assertEqual(len(capture_assist_features(helper.snapshot())), 12)
        self.assertEqual(set(helper.state), set(CAPTURE_ASSIST_FEATURE_NAMES))
        self.assertEqual(helper.snapshot()["retirement_revision"], CAPTURE_ASSIST_RETIREMENT_REVISION)
        self.assertNotIn("retirement_revision", CAPTURE_ASSIST_FEATURE_NAMES)

    def test_invalid_actual_final_cannot_start_release(self):
        for previous in ((1.,) * 11, (float("nan"),) + FINAL[1:]):
            with self.subTest(previous=previous):
                helper = exhausted()
                with self.assertRaises(ValueError):
                    advance(helper, previous=previous)
                self.assertEqual(helper.state["mode"], 3.)

    def test_context_reads_current_air_support_separately_from_placed(self):
        frame = SimpleNamespace(state_id="P06", endpoint_issued=False, full12=FINAL)
        nominal = SimpleNamespace(_continuous_layers=())
        observation = {"joints": {SERVO_ORDER[0]: {"position_deg": FINAL[0]}}}
        for fl in ({"air": True, "support": False}, {"air": False, "support": True}, {}):
            task = {"physical_evaluator": {"valid": True, "physics_tick": 8791,
                    "current_legs": {"FL": fl}, "history": {"placed": {"FL": True}}}}
            result = capture_assist_context(task=task, observation=observation,
                source_frame=frame, previous_ack={}, nominal_provider=nominal, physics_tick=8971)
            self.assertIs(result["placed_FL"], True)
            self.assertIs(result["current_FL_air"], fl.get("air"))
            self.assertIs(result["current_FL_support"], fl.get("support"))
            self.assertEqual(result["source_observation_tick"], 8791)
            self.assertEqual(result["dispatch_physics_tick"], 8971)


if __name__ == "__main__":
    unittest.main()
