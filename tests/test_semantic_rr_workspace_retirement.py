"""Installed-runtime tests for opt-in RR receiver-workspace retirement.

Synthetic evaluator dictionaries exercise the actual production potential;
they are formula/contract tests, not Isaac trajectories or task success.
No candidate subclass or duplicated retirement predicate is used.
"""
from copy import deepcopy
from pathlib import Path
import math
import tempfile
import unittest

import yaml

from wlr50_clean.ppo.semantic_supervisor import (
    TaskStageSupervisor, load_task_spec,
    RR_WORKSPACE_RETIREMENT_MODE as MODE,
    RR_WORKSPACE_RETIREMENT_MODE_V2 as MODE_V2,
    _rr_workspace_retirement_enabled as validate_candidate_mode,
    _current_rr_receiver_preparation_retired as current_rr_receiver_preparation_retired,
)

KEY = "rr_postcross_workspace_semantics"
ROOT = Path(__file__).resolve().parents[1]


class RetirementTests(unittest.TestCase):
    def setUp(self):
        spec = load_task_spec(ROOT / "configs/ppo_p05_hip_only_continuation_v1/stage_task_spec.yaml")
        self.assertEqual(spec.get(KEY), MODE_V2, "adopted config must explicitly enable v2")
        self.assertTrue(validate_candidate_mode(spec))
        spec[KEY] = MODE  # This suite remains the exact v1-vs-disabled regression.
        self.old = TaskStageSupervisor.__new__(TaskStageSupervisor)
        self.old.spec = deepcopy(spec)
        self.old.spec.pop(KEY, None)  # Config under test is NEW; old control must opt out explicitly.
        self.new = TaskStageSupervisor.__new__(TaskStageSupervisor)
        self.new.spec = deepcopy(spec)
        self.ev = {"valid": True, "termination_reason": None,
            "history": {"active_lift": {}, "front_edge_crossed": {}, "placed": {}},
            "current_legs": {}, "transfer_roles": {}}
        for leg in ("FL", "FR", "RL", "RR"):
            for key in self.ev["history"]:
                self.ev["history"][key][leg] = leg in ("FL", "FR")
            self.ev["current_legs"][leg] = dict(front_distance_m=.04, clearance_m=.06,
                top_xy_outside_distance_m=0., within_lateral_span=True,
                within_top_xy=True, ground_contact=False, air=True,
                initial_clearance=False, current_lift_valid=False,
                consecutive_top_samples=0)
            self.ev["transfer_roles"][leg] = dict(workspace_progress=.3, transfer_progress=1.)
        self.ev["history"]["active_lift"]["RR"] = True
        self.ev["history"]["front_edge_crossed"]["RR"] = True
        self.ev["current_legs"]["RR"].update(current_lift_valid=True, initial_clearance=True)

    @staticmethod
    def load_variant(spec):
        # Exercise the actual production YAML loader, not only its helper.
        with tempfile.TemporaryDirectory(prefix="rr_workspace_spec_test_") as directory:
            path = Path(directory) / "stage_task_spec.yaml"
            path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
            return load_task_spec(path)

    def assert_old_workspace(self, ev):
        self.assertEqual(self.new._workspace_potential_progress("RR", ev),
                         self.old._workspace_potential_progress("RR", ev))

    def test_missing_and_null_mode_are_exact_old(self):
        for mode in (None,):
            self.new.spec[KEY] = mode
            self.new.spec = self.load_variant(self.new.spec)
            self.assertFalse(validate_candidate_mode(self.new.spec))
            self.assertEqual(self.new.physical_potential(self.ev), self.old.physical_potential(self.ev))
        self.new.spec.pop(KEY)
        self.new.spec = self.load_variant(self.new.spec)
        self.assertNotIn(KEY, self.new.spec)
        self.assertFalse(validate_candidate_mode(self.new.spec))
        self.assertEqual(self.new.physical_potential(self.ev), self.old.physical_potential(self.ev))

    def test_unknown_or_incompatible_mode_rejected(self):
        for value in (True, "bad", 1, []):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_candidate_mode({**self.new.spec, KEY: value})
            with self.subTest(loader_value=value), self.assertRaises(ValueError):
                self.load_variant({**self.new.spec, KEY: value})
        for key, value in (("p09_lift_semantics", None), ("physical_acceptance_version", None),
                           ("transfer_roles", None), ("potential_definition", None),
                           ("workspace_potential_semantics", None)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_candidate_mode({**self.new.spec, key: value})
            with self.subTest(loader_key=key), self.assertRaises(ValueError):
                self.load_variant({**self.new.spec, key: value})

    def test_only_existing_RR_receiver_half_is_retired(self):
        delta = self.new.physical_potential(self.ev) - self.old.physical_potential(self.ev)
        self.assertAlmostEqual(delta, .85/4*.1*.5*(1.-.3))
        for leg in ("FL", "FR", "RL"):
            self.assertEqual(self.new._workspace_potential_progress(leg, self.ev),
                             self.old._workspace_potential_progress(leg, self.ev))

    def test_successful_cross_does_not_drop_preparation_potential(self):
        for role in (0., .2, .75, 1.):
            before = deepcopy(self.ev)
            before["history"]["front_edge_crossed"]["RR"] = False
            before["transfer_roles"]["RR"]["workspace_progress"] = role
            after = deepcopy(before)
            after["history"]["front_edge_crossed"]["RR"] = True
            self.assertGreaterEqual(self.new._workspace_potential_progress("RR", after),
                                    self.new._workspace_potential_progress("RR", before))

    def test_no_demand_for_continued_FL_contraction_after_qualified_cross(self):
        values = []
        for receiver in (0., .2, .7, 1.):
            self.ev["transfer_roles"]["RR"]["workspace_progress"] = receiver
            values.append(self.new.physical_potential(self.ev))
        self.assertTrue(all(value == values[0] for value in values))

    def test_pre_cross_exploration_is_exact_old(self):
        self.ev["history"]["front_edge_crossed"]["RR"] = False
        for receiver in (0., .5, 1.):
            self.ev["transfer_roles"]["RR"]["workspace_progress"] = receiver
            self.assert_old_workspace(self.ev)

    def test_unqualified_early_XY_and_old_ground_history_are_insufficient(self):
        for history_changes, current_changes in (
            ({"active_lift": False}, {}),
            ({"front_edge_crossed": False}, {}),
            ({}, {"current_lift_valid": False}),
            ({}, {"ground_contact": True}),
            ({}, {"within_top_xy": False}),
            ({}, {"within_lateral_span": False}),
            ({}, {"front_distance_m": -.001}),
            ({}, {"clearance_m": -.04}),
            ({}, {"air": False, "top_surface_contact": False}),
        ):
            ev = deepcopy(self.ev)
            for key, value in history_changes.items():
                ev["history"][key]["RR"] = value
            ev["current_legs"]["RR"].update(current_changes)
            self.assert_old_workspace(ev)

    def test_support_loss_revokes_current_retirement_not_history(self):
        # Production evaluator sets current_lift_valid false when its verified
        # other-support corroboration is lost; this reward does not invent it.
        before = deepcopy(self.ev["history"])
        self.ev["current_legs"]["RR"]["current_lift_valid"] = False
        self.assert_old_workspace(self.ev)
        self.assertEqual(self.ev["history"], before)

    def test_current_edge_distance_still_matters(self):
        near = self.new._workspace_potential_progress("RR", self.ev)
        self.ev["current_legs"]["RR"]["front_distance_m"] = .12
        far = self.new._workspace_potential_progress("RR", self.ev)
        self.assertLess(far, near)

    def test_descent_and_actual_TOP_share_remain_exact(self):
        before = deepcopy(self.ev)
        for changes in ({"clearance_m": .03}, {"consecutive_top_samples": 1, "air": False,
                                                "top_surface_contact": True, "top_contact": True}):
            after = deepcopy(before)
            after["current_legs"]["RR"].update(changes)
            old_delta = self.old.physical_potential(after) - self.old.physical_potential(before)
            new_delta = self.new.physical_potential(after) - self.new.physical_potential(before)
            self.assertGreater(new_delta, 0.)
            self.assertAlmostEqual(new_delta, old_delta)

    def test_real_TOP_with_signed_negative_gap_can_retain_credit(self):
        self.ev["current_legs"]["RR"].update(air=False, clearance_m=-.001,
            top_contact=True, top_surface_contact=True, consecutive_top_samples=1)
        self.assertTrue(current_rr_receiver_preparation_retired(self.new.spec, "RR", self.ev))
        self.assertFalse(self.ev["history"]["placed"]["RR"])

    def test_AIR_descent_across_zero_to_existing_capture_min_keeps_receiver_credit(self):
        previous = None
        minimum = self.new.spec["geometry"]["top_gap_min_m"]
        self.assertEqual(minimum, -.015)
        for gap in (.001, 0., -.001, minimum):
            ev = deepcopy(self.ev)
            ev["current_legs"]["RR"].update(clearance_m=gap, air=True,
                top_contact=False, top_surface_contact=False, consecutive_top_samples=0)
            snapshot = deepcopy(ev)
            self.assertTrue(current_rr_receiver_preparation_retired(self.new.spec, "RR", ev))
            self.assertEqual(self.new._workspace_potential_progress("RR", ev), 1.)
            self.assertEqual(self.new._current_capture_progress("RR", ev, 0.),
                             self.old._current_capture_progress("RR", ev, 0.))
            if previous is not None:
                self.assertAlmostEqual(
                    self.new.physical_potential(ev)-self.new.physical_potential(previous),
                    self.old.physical_potential(ev)-self.old.physical_potential(previous))
            self.assertEqual(ev, snapshot)
            self.assertFalse(ev["history"]["placed"]["RR"])
            self.assertFalse(ev["current_legs"]["RR"]["top_contact"])
            previous = ev

    def test_AIR_below_existing_capture_min_and_low_requalified_history_are_negative(self):
        minimum = self.new.spec["geometry"]["top_gap_min_m"]
        for gap in (math.nextafter(minimum, -math.inf), -.04):
            ev = deepcopy(self.ev)
            ev["current_legs"]["RR"]["clearance_m"] = gap
            self.assertFalse(current_rr_receiver_preparation_retired(self.new.spec, "RR", ev))
            self.assert_old_workspace(ev)
        # Historical crossing must not allow actual ground, even if an
        # adversarial synthetic current qualification field remains true.
        ev = deepcopy(self.ev)
        ev["current_legs"]["RR"].update(clearance_m=-.001, ground_contact=True)
        self.assertFalse(current_rr_receiver_preparation_retired(self.new.spec, "RR", ev))

    def test_AIR_tolerance_reads_existing_spec_not_new_hardcoded_plane(self):
        self.new.spec["geometry"]["top_gap_min_m"] = -.007
        for gap, eligible in ((-.007, True), (-.008, False)):
            ev = deepcopy(self.ev)
            ev["current_legs"]["RR"]["clearance_m"] = gap
            self.assertEqual(current_rr_receiver_preparation_retired(self.new.spec, "RR", ev), eligible)

    def test_real_placement_bypasses_retirement_and_retention_is_exact_old(self):
        self.ev["history"]["placed"]["RR"] = True
        for outside in (0., .05, .3):
            self.ev["current_legs"]["RR"]["top_xy_outside_distance_m"] = outside
            self.assertEqual(self.new.physical_potential(self.ev), self.old.physical_potential(self.ev))

    def test_no_fake_event_TOP_success_or_state_mutation(self):
        before = deepcopy(self.ev)
        for _ in range(10):
            self.new.physical_potential(self.ev)
        self.assertEqual(self.ev, before)
        self.assertFalse(self.ev["history"]["placed"]["RR"])
        self.assertEqual(self.ev["current_legs"]["RR"]["consecutive_top_samples"], 0)

    def test_phase_labels_cannot_award_progress(self):
        values = []
        for stage in ("P01", "P08", "P09", "P10", "P13"):
            self.new.stage_id = stage
            values.append(self.new.physical_potential(self.ev))
        self.assertTrue(all(value == values[0] for value in values))

    def test_repeated_eligibility_has_no_event_bonus_and_chatter_does_not_accumulate(self):
        before = deepcopy(self.ev)
        before["current_legs"]["RR"]["front_distance_m"] = -.001
        low, high = self.new.physical_potential(before), self.new.physical_potential(self.ev)
        gamma = .9985
        r1, r2 = 5*(gamma*high-low), 5*(gamma*low-high)
        self.assertAlmostEqual(r1+gamma*r2, 5*(gamma*gamma-1)*low)
        self.assertLessEqual(r1+gamma*r2, 0.)
        self.assertLess(5*(gamma-1)*high, 0.)

    def test_invalid_and_terminal_cannot_enable_retirement(self):
        for fields in ({"valid": False}, {"termination_reason": "BODY_COLLISION"}):
            ev = {**self.ev, **fields}
            self.assertFalse(current_rr_receiver_preparation_retired(self.new.spec, "RR", ev))
        self.assertEqual(self.new.physical_potential({**self.ev, "valid": False}), 0.)


if __name__ == "__main__":
    unittest.main(verbosity=2)
