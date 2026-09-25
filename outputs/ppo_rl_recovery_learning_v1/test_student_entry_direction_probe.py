"""Pure tests for the dormant student-entry direction diagnostic."""

import importlib.util
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
import math
from pathlib import Path
import sys
import unittest


PATH = Path(__file__).with_name("student_entry_direction_probe.py")
SPEC = importlib.util.spec_from_file_location("student_entry_direction_probe", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def task(*, phase="P09", qualified=True, xy=True, air=True, front=.01, gap=.02,
         fl_support=True, rr_contact=False):
    legs = {}
    for leg in ("FR", "FL", "RL"):
        support = fl_support if leg == "FL" else True
        legs[leg] = {"support": support, "bearing_verified": support,
                     "bearing_force_n": 2.0 if support else 0.0}
    legs["RR"] = {
        "current_lift_valid": qualified,
        "within_top_xy": xy,
        "within_lateral_span": True,
        "air": air,
        "top_contact": rr_contact,
        "ground_contact": False,
        "front_distance_m": front,
        "clearance_m": gap,
    }
    return {
        "phase_id": phase,
        "termination_reason": None,
        "physical_evaluator": {
            "valid": True,
            "current_legs": legs,
            "history": {"active_lift": {"RR": True}},
        },
    }


class CandidateTest(unittest.TestCase):
    def test_exact_three_raw_replacements_and_rr_knee_preserved(self):
        original = tuple(float(i) / 10.0 for i in range(12))
        applied = MODULE.apply_raw_candidate(
            original, fl_knee_raw=.7, fr_knee_raw=.8, rr_hip_raw=-.9)
        self.assertEqual(applied[1], .7)
        self.assertEqual(applied[3], .8)
        self.assertEqual(applied[6], -.9)
        self.assertEqual(applied[7], original[7])
        self.assertEqual(
            [applied[i] for i in range(12) if i not in (1, 3, 6)],
            [original[i] for i in range(12) if i not in (1, 3, 6)],
        )

    def test_candidate_is_absolute_not_accumulated(self):
        first = MODULE.apply_raw_candidate(
            (0.1,) * 12, fl_knee_raw=.7, fr_knee_raw=.8, rr_hip_raw=-.9)
        second = MODULE.apply_raw_candidate(
            first, fl_knee_raw=.7, fr_knee_raw=.8, rr_hip_raw=-.9)
        self.assertEqual(first, second)

    def test_wrong_sign_and_nonfinite_rejected(self):
        for values in ((-.1, .2, -.3), (.1, -.2, -.3), (.1, .2, .3),
                       (.1, .2, float("nan")), (3.1, .2, -.3)):
            with self.assertRaises(ValueError):
                MODULE.apply_raw_candidate(
                    (0.0,) * 12, fl_knee_raw=values[0],
                    fr_knee_raw=values[1], rr_hip_raw=values[2])


class EligibilityTest(unittest.TestCase):
    def check(self, value):
        return MODULE.rr_air_entry_evidence(
            value, minimum_other_supports=2, force_noise_floor_n=.2)

    def test_exact_qualified_air_xy_with_fl_and_other_support(self):
        result = self.check(task())
        self.assertTrue(result["eligible"])
        self.assertIn("FL", result["supporting_other_legs"])

    def test_phase_qualification_xy_contact_and_fl_support_fail_closed(self):
        cases = (
            task(phase="P08"), task(qualified=False), task(xy=False),
            task(air=False, rr_contact=True), task(front=-.001), task(gap=0.0),
            task(fl_support=False),
        )
        for value in cases:
            with self.subTest(value=value):
                result = self.check(value)
                self.assertFalse(result["eligible"])
                self.assertTrue(result["blockers"])

    def test_historical_lift_alone_is_insufficient(self):
        value = task(qualified=False)
        self.assertTrue(value["physical_evaluator"]["history"]["active_lift"]["RR"])
        self.assertFalse(self.check(value)["eligible"])

    def test_active_release_on_current_qualification_or_support_loss(self):
        for value, expected_fragment in (
            (task(qualified=False), "RR_not_currently_qualified"),
            (task(fl_support=False), "FL_real_support_absent"),
        ):
            evidence = self.check(value)
            reason = MODULE.active_release_reason(
                value, evidence, elapsed_s=.5, maximum_probe_seconds=5.0)
            self.assertTrue(reason.startswith("entry_eligibility_lost:"))
            self.assertIn(expected_fragment, reason)

    def test_contact_release_reason_precedes_generic_eligibility_loss(self):
        value = task(air=False, rr_contact=True)
        evidence = self.check(value)
        self.assertFalse(evidence["eligible"])
        self.assertEqual(
            MODULE.active_release_reason(
                value, evidence, elapsed_s=.5, maximum_probe_seconds=5.0),
            "RR_real_contact_observed",
        )


class SerializationTest(unittest.TestCase):
    def test_slots_physical_observation_and_nested_step_dataclass(self):
        class Contact(Enum):
            AIR = "AIR"

        @dataclass(frozen=True, slots=True)
        class SemanticPhysicalObservation:
            physics_tick: int
            contacts: dict
            all_finite: bool = True

        @dataclass(frozen=True, slots=True)
        class NestedAudit:
            observation: SemanticPhysicalObservation
            status: Contact
            raw_value: float

        def physical_json(value):
            if is_dataclass(value):
                return physical_json(asdict(value))
            if isinstance(value, Enum):
                return value.value
            if isinstance(value, dict):
                return {str(key): physical_json(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [physical_json(item) for item in value]
            if isinstance(value, float) and not math.isfinite(value):
                return {"nonfinite_raw_float": str(value)}
            if value is None or isinstance(value, (str, bool, float, int)):
                return value
            raise TypeError(type(value).__name__)

        raw = SemanticPhysicalObservation(
            physics_tick=17,
            contacts={"RR": {"mode": Contact.AIR}},
        )
        payload = MODULE.diagnostic_json_payload(
            {"raw_observation": raw,
             "step_info": {"nested": NestedAudit(raw, Contact.AIR, float("nan"))}},
            physical_json=physical_json,
            jsonable=lambda value: value,
        )
        self.assertEqual(payload["raw_observation"]["physics_tick"], 17)
        self.assertEqual(payload["raw_observation"]["contacts"]["RR"]["mode"], "AIR")
        self.assertEqual(payload["step_info"]["nested"]["status"], "AIR")
        self.assertEqual(payload["step_info"]["nested"]["raw_value"],
                         {"nonfinite_raw_float": "nan"})


if __name__ == "__main__":
    unittest.main()
