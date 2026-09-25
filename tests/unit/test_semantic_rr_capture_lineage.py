"""Stdlib-only local predicate tests; no physics or model claims."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def load(name, path):
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


v2 = load("production_capture_lineage_v2", ROOT / "src/wlr50_clean/ppo/semantic_rr_capture_local_task.py")
fixture = load("production_local_fixture", HERE / "test_semantic_rr_capture_local_task.py")


def observed(tick, *, established=True, continuation=True, **kwargs):
    frame = fixture.observed(tick, **kwargs)
    row = frame.info["semantic_task"]["physical_evaluator"]["current_legs"]["RR"]
    row.update(lift_established=established, active_attempt=established,
               motion_continuation_allowed=continuation)
    return frame


def hold(task, start, *, established=True, qualified=True, continuation=True):
    for index in range(61):
        task.observe(observed(start + index, established=established, qualified=qualified,
            continuation=continuation, top=True, bearing=True, placed=True, gap=0., samples=index+1))
    return task.snapshot()


class CaptureLineageTests(unittest.TestCase):
    def test_air_then_normal_top_hold_succeeds(self):
        task = v2.RRCaptureLocalTask()
        task.observe(observed(0))
        result = hold(task, 1)
        self.assertTrue(result["local_success"])
        self.assertEqual(len(result["obs9"]), 9)
        self.assertEqual(result["obs9"][-1], 1.)
        self.assertAlmostEqual(result["hold_elapsed_s"], .5)

    def test_recorded_success_top_need_not_have_current_air_validity(self):
        # Actual first success: established=true, current_lift_valid=false,
        # continuation=true, verified TOP/bearing=true, 61 samples/0.5s.
        task = v2.RRCaptureLocalTask()
        task.observe(observed(0))
        result = hold(task, 1, established=True, qualified=False)
        self.assertTrue(result["local_success"])
        self.assertFalse(result["metrics"]["current_lift_valid"])
        self.assertTrue(result["metrics"]["current_attempt_capture_eligible"])

    def test_old_top_ground_then_unqualified_top_cannot_succeed(self):
        task = v2.RRCaptureLocalTask()
        task.observe(observed(0))
        for tick in range(1, 5):
            task.observe(observed(tick, top=True, bearing=True, placed=True, samples=tick))
        task.observe(observed(5, established=False, qualified=False, ground=True, placed=True, gap=-.05))
        self.assertTrue(task.active)
        self.assertEqual(task.obs9()[-1], 0.)
        result = hold(task, 6, established=False, qualified=False)
        self.assertFalse(result["local_success"])
        self.assertEqual(result["hold_elapsed_s"], 0.)
        self.assertTrue(result["active"])
        self.assertTrue(result["metrics"]["placed"])
        self.assertTrue(result["metrics"]["crossed"])

    def test_new_qualified_air_after_ground_allows_new_top_success(self):
        task = v2.RRCaptureLocalTask()
        task.observe(observed(0))
        task.observe(observed(1, top=True, bearing=True, placed=True, samples=1))
        task.observe(observed(2, established=False, qualified=False, ground=True, placed=True))
        for tick in range(3, 7):
            task.observe(observed(tick, established=False, qualified=False, placed=True))
            self.assertTrue(task.active)
            self.assertEqual(task.obs9()[-1], 0.)
        # Synthetic input represents the existing evaluator's freshly earned
        # AIR establishment, not an external setter into that evaluator.
        task.observe(observed(7, established=True, qualified=True, placed=True))
        self.assertTrue(task.active)
        result = hold(task, 8, established=True, qualified=False)
        self.assertTrue(result["local_success"])
        self.assertEqual(task.activation_tick, 0)

    def test_no_qualification_is_unfinished_not_automatic_wheel_failure(self):
        task = v2.RRCaptureLocalTask()
        task.observe(observed(0))
        task.observe(observed(1, established=False, qualified=False, ground=True, placed=True))
        before = task.snapshot()
        task.observe(observed(2, established=False, qualified=False, top=True,
                              bearing=True, placed=True, samples=1))
        reward = task.reward(before)
        self.assertFalse(reward["terminated"])
        self.assertTrue(reward["terminal_bootstrap_allowed"])
        self.assertEqual(reward["terminal_event"], 0.)
        self.assertIsNone(reward["termination_reason"])

    def test_missing_establishment_not_silently_filled_from_history(self):
        frame = fixture.observed(0)
        del frame.info["semantic_task"]["physical_evaluator"]["current_legs"]["RR"]["lift_established"]
        with self.assertRaisesRegex(ValueError, "lift_established"):
            v2.RRCaptureLocalTask().observe(frame)

    def test_contact_potential_qualification_does_not_relabel_contact_or_change_valid_reward(self):
        task = v2.RRCaptureLocalTask()
        task.observe(observed(0))
        task.observe(observed(1, established=False, qualified=False, ground=True, placed=True))
        before = task.snapshot()
        task.observe(observed(2, established=False, qualified=False, top=True,
                              bearing=True, placed=True, samples=1, gap=0.))
        result = task.snapshot()
        self.assertEqual(result["potential_components"]["real_contact"], 0.)
        self.assertEqual(result["obs9"][5:7], (1., 1.))
        self.assertTrue(result["metrics"]["current_top_contact"])
        self.assertTrue(result["metrics"]["current_top_bearing"])
        self.assertFalse(task.reward(before)["terminated"])

        # Frozen v1 positive-path arithmetic, not a second copy of the current
        # implementation. The new eligibility guard changes no valid reward.
        candidate = v2.RRCaptureLocalTask()
        prior_potential = None
        for tick in range(62):
            frame = observed(tick, qualified=tick == 0, established=True,
                             top=tick > 0, bearing=tick > 0, placed=tick > 1,
                             samples=tick, gap=0. if tick else .055)
            before = candidate.snapshot()
            candidate.observe(frame)
            hold_progress = min(1., max(0., tick-1)/60.)
            expected_parts = dict(legal_xy=.2,
                gap_closure=.4 if tick else .4/(1.+.055/.025),
                real_contact=.2 if tick else 0., bearing_hold=.2*hold_progress)
            for key, expected in expected_parts.items():
                self.assertAlmostEqual(candidate.potential_components()[key], expected)
            expected_potential = sum(expected_parts.values())
            if tick:
                reward = candidate.reward(before)
                success = tick == 61
                shaping = 5.*(.9985*(0. if success else expected_potential)-prior_potential)
                event = 40. if success else 0.
                time_cost = .02*(tick/120.-(tick-1)/120.)
                self.assertAlmostEqual(reward["potential_shaping"], shaping)
                self.assertAlmostEqual(reward["terminal_event"], event)
                self.assertAlmostEqual(reward["time_cost"], time_cost)
                self.assertAlmostEqual(reward["reward"], shaping+event-time_cost)
                self.assertEqual(reward["terminated"], success)
                self.assertEqual(reward["rr_subtask_success"], success)
            prior_potential = expected_potential
        self.assertTrue(candidate.local_success)
        self.assertEqual(candidate.potential_components()["real_contact"], .2)

    def test_continuation_false_stops_hold_without_clearing_gate(self):
        task = v2.RRCaptureLocalTask()
        task.observe(observed(0))
        result = hold(task, 1, continuation=False)
        self.assertFalse(result["local_success"])
        self.assertEqual(result["hold_elapsed_s"], 0.)
        self.assertTrue(task.active)
        self.assertEqual(task.obs9()[-1], 0.)

    def test_original_eight_local_columns_are_not_reinterpreted(self):
        candidate = v2.RRCaptureLocalTask()
        original_fields = (
            "active", "activation_age_norm", "entry_rr_hip_norm", "entry_rr_knee_norm",
            "entry_gap_norm", "current_top_contact", "current_top_bearing",
            "capture_hold_progress",
        )
        for tick in range(63):
            frame = observed(tick, top=tick > 0, bearing=tick > 0,
                             placed=tick > 1, samples=tick, gap=0. if tick else .055)
            candidate.observe(frame)
            # v1 field order/scales remain fixed; ninth column is appended.
            expected = (1., (tick/120.)/200., 8./180., -58./180., .055/.1,
                        float(tick > 0), float(tick > 0),
                        min(1., max(0., tick-1)/60.))
            for actual, reference in zip(candidate.obs9()[:8], expected):
                self.assertAlmostEqual(actual, reference)
        self.assertEqual(v2.OBSERVATION_FIELDS[:8], original_fields)
        self.assertEqual(v2.OBSERVATION_FIELDS[-1], "current_attempt_capture_eligible")



if __name__ == "__main__":
    unittest.main(verbosity=2)
