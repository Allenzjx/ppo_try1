"""CPU-only candidate harness; execution requires explicit root permission.

Uses the real frozen mapper, original final slew, and current pure headroom.
The plant is deliberately NOT Isaac: free q follows the just-commanded final
target exactly; blocked q remains fixed. No torch, CUDA, optimizer, or scene.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from active_reference_candidate import active_tracking_reference
from wlr50_clean.infrastructure.command_batch import SERVO_COMMAND_SIGN, SERVO_ORDER, servo_limits_deg
from wlr50_clean.infrastructure.robot_adapter import bounded_drive_feedback_step
from wlr50_clean.infrastructure.servo_target_mapper import ServoTargetMapper
from wlr50_clean.ppo.semantic_headroom import project_semantic_servo_headroom

if any(name.split(".")[0] in {"torch", "isaacsim", "isaaclab", "omni"} for name in sys.modules):
    raise RuntimeError("report candidate must remain pure CPU Python without simulator imports")

ZERO8 = (0.0,) * 8
STANDING = dict(zip(SERVO_ORDER, (12., 20., -8., 19., 7., -16., -9., 13.), strict=True))


def vector(index, value):
    result = list(ZERO8)
    result[index] = value
    return tuple(result)


def measured(canonical):
    return tuple(math.radians(STANDING[name] + SERVO_COMMAND_SIGN[name] * q)
                 for name, q in zip(SERVO_ORDER, canonical, strict=True))


class ToyPlant:
    """One stateful mapper and one final history; zero extra mapper advances."""
    def __init__(self, *, candidate, index=0, nominal=0., blocked=False):
        self.mapper = ServoTargetMapper(STANDING)
        self.candidate, self.index, self.blocked = candidate, index, blocked
        self.nominal = vector(index, nominal)
        self.q = ZERO8
        self.final = ZERO8
        self.previous_effective = ZERO8
        self.rows = []
        # Settle nominal through the actual original command-space slew, no
        # synthetic state copying. These initialization ticks are not scored.
        for _ in range(240):
            self.step(0., tracking=())
        self.rows.clear()

    def step(self, request, *, tracking=None, nominal=None):
        if nominal is not None:
            self.nominal = vector(self.index, nominal)
        tracking = (SERVO_ORDER[self.index],) if tracking is None else tuple(tracking)
        actual_q = measured(self.q)
        reference = active_tracking_reference(measured_physical_rad=actual_q,
            previous_effective_residual_deg=self.previous_effective,
            tracking_servo_names=tracking)
        before_clock = self.mapper.feedback_tick
        mapped = self.mapper.advance(self.nominal,
            reference.mapper_computational_reference_rad if self.candidate else actual_q,
            tracking_servo_names=tracking)
        if self.mapper.feedback_tick != before_clock + 1:
            raise AssertionError("mapper must advance exactly once")
        headroom = project_semantic_servo_headroom(
            mapped.applied_drive_command_deg + (0.,)*4, (0.,)*12,
            vector(self.index, request) + (0.,)*4)
        effective = tuple(headroom["effective_policy_residual_full12"][:8])
        final = tuple(bounded_drive_feedback_step(previous_deg=previous, native_deg=native,
            bias_deg=r, maximum_delta_deg=self.mapper.maximum_delta_deg,
            lower_deg=servo_limits_deg(name)[0], upper_deg=servo_limits_deg(name)[1])
            for name, previous, native, r in zip(SERVO_ORDER, self.final,
                mapped.applied_drive_command_deg, effective, strict=True))
        old_q = self.q
        # During initialization use the free plant; blocking begins only when
        # tracking is scheduled, after nominal was naturally slewed into place.
        self.q = old_q if self.blocked and tracking else final
        i = self.index
        row = {"tick": before_clock, "sampled": mapped.feedback_sampled,
            "nominal": self.nominal[i], "actual_q_before": old_q[i],
            "actual_q_after": self.q[i], "request": request,
            "previous_effective": self.previous_effective[i],
            "active_reference_deg": reference.active_reference_deg[i] if self.candidate else 0.,
            "effective": effective[i], "native": mapped.applied_drive_command_deg[i],
            "compensation": mapped.tracking_compensation_deg[i], "final": final[i],
            "final_delta": final[i] - self.final[i],
            "safe_band": headroom["servo_safety_limits_deg"][i],
            "baseline_outside_reserved_band": i in headroom["baseline_outside_reserved_servo_indices"]}
        self.final, self.previous_effective = final, effective
        self.rows.append(row)
        return row


def held_trial(*, candidate, index, held, nominal=0., blocked=False, ticks=960):
    plant = ToyPlant(candidate=candidate, index=index, nominal=nominal, blocked=blocked)
    for tick in range(ticks):
        # Existing projector 60 deg/s at 120 Hz: requested reference ramps at
        # 0.5 deg/tick rather than jumping immediately to a 20 degree request.
        request = math.copysign(min(abs(held), .5*(tick+1)), held)
        plant.step(request)
    return plant


def summarize(plant):
    tail = plant.rows[-240:]
    offsets = [r["actual_q_after"] - r["nominal"] for r in tail]
    return {"steady_window_ticks": len(tail), "steady_offset_mean_deg": statistics.fmean(offsets),
        "steady_offset_min_deg": min(offsets), "steady_offset_max_deg": max(offsets),
        "steady_offset_peak_to_peak_deg": max(offsets)-min(offsets),
        "steady_offset_population_std_deg": statistics.pstdev(offsets),
        "steady_compensation_mean_deg": statistics.fmean(r["compensation"] for r in tail),
        "maximum_final_delta_deg": max(abs(r["final_delta"]) for r in plant.rows),
        "reserved_band_baseline_ticks": sum(r["baseline_outside_reserved_band"] for r in plant.rows),
        "first_reserved_band_example": next((r for r in plant.rows if r["baseline_outside_reserved_band"]), None),
        "last_12_ticks": tail[-12:]}


class CandidateTests(unittest.TestCase):
    def test_zero_reference_bitwise_and_no_mutation(self):
        q = [-0., .1, -.2, .3, -.4, .5, -.6, .7]
        old = list(q)
        result = active_tracking_reference(measured_physical_rad=q,
            previous_effective_residual_deg=ZERO8, tracking_servo_names=SERVO_ORDER)
        self.assertEqual(struct.pack("8d", *result.mapper_computational_reference_rad), struct.pack("8d", *q))
        self.assertEqual(q, old)

    def test_zero_full_mapper_history_equivalence(self):
        a, b = ToyPlant(candidate=False), ToyPlant(candidate=True)
        for tick in range(180):
            nominal = 6. if tick < 70 else (-4. if tick < 130 else 0.)
            tracking = () if 45 <= tick < 90 else (SERVO_ORDER[0],)
            self.assertEqual(a.step(0., tracking=tracking, nominal=nominal),
                             b.step(0., tracking=tracking, nominal=nominal))
            self.assertEqual(a.mapper.__dict__, b.mapper.__dict__)

    def test_active_and_ended_retirement_are_disjoint(self):
        a, b = ServoTargetMapper(STANDING), ServoTargetMapper(STANDING)
        for _ in range(9):
            for mapper in (a, b):
                mapper.advance(ZERO8, measured(vector(0, -5.)), tracking_servo_names=(SERVO_ORDER[0], SERVO_ORDER[6]))
        self.assertGreater(a._compensation[SERVO_ORDER[0]], 2.)
        q = measured(ZERO8)
        reference = active_tracking_reference(measured_physical_rad=q,
            previous_effective_residual_deg=(20.,0.,0.,0.,0.,0.,20.,0.),
            tracking_servo_names=(SERVO_ORDER[6],))
        self.assertEqual(reference.mapper_computational_reference_rad[0], q[0])
        left = a.advance(ZERO8, q, tracking_servo_names=(SERVO_ORDER[6],))
        right = b.advance(ZERO8, reference.mapper_computational_reference_rad, tracking_servo_names=(SERVO_ORDER[6],))
        self.assertEqual(left.tracking_compensation_deg[0], right.tracking_compensation_deg[0])
        self.assertTrue(a._retiring_stale_bias[SERVO_ORDER[0]])
        self.assertEqual(a._retiring_stale_bias[SERVO_ORDER[0]], b._retiring_stale_bias[SERVO_ORDER[0]])
        # At the next sample only the still-active rear channel may differ.
        for _ in range(3):
            left = a.advance(ZERO8, q, tracking_servo_names=(SERVO_ORDER[6],))
            right = b.advance(ZERO8, reference.mapper_computational_reference_rad, tracking_servo_names=(SERVO_ORDER[6],))
        self.assertNotEqual(left.tracking_compensation_deg[6], right.tracking_compensation_deg[6])
        self.assertEqual(left.tracking_compensation_deg[0], right.tracking_compensation_deg[0])

    def test_negative_rear_sign_and_no_standing_double_subtraction(self):
        q = measured(ZERO8)
        result = active_tracking_reference(measured_physical_rad=q,
            previous_effective_residual_deg=(3.7,0.,0.,0.,0.,0.,20.,0.),
            tracking_servo_names=(SERVO_ORDER[0], SERVO_ORDER[6]))
        self.assertAlmostEqual(result.mapper_computational_reference_rad[0], q[0]-math.radians(3.7))
        self.assertAlmostEqual(result.mapper_computational_reference_rad[6], q[6]+math.radians(20.))

    def test_clipped_request_zero_is_not_reference(self):
        plant = ToyPlant(candidate=True, nominal=133.)
        for _ in range(24):
            row = plant.step(20.)
            self.assertAlmostEqual(row["effective"], 0., places=9)
            self.assertAlmostEqual(row["active_reference_deg"], 0., places=9)
            self.assertAlmostEqual(row["native"], 133., places=9)

    def test_nominal_change_restart_and_feedback_clock(self):
        plant = held_trial(candidate=True, index=0, held=3.7, ticks=24)
        row = plant.step(3.7, nominal=80.)
        self.assertEqual(row["compensation"], 0.)
        self.assertFalse(plant.mapper._nominal_reached[SERVO_ORDER[0]])
        for _ in range(80):
            plant.step(0., tracking=())
        clock = plant.mapper.feedback_tick
        rows = [plant.step(0.) for _ in range(8)]
        self.assertEqual(plant.mapper.feedback_tick, clock+8)
        self.assertEqual([r["sampled"] for r in rows], [r["tick"] % 4 == 0 for r in rows])

    def test_current_zero_previous_nonzero_reference_exits_next_tick(self):
        plant = held_trial(candidate=True, index=0, held=3.7, ticks=24)
        first, second = plant.step(0.), plant.step(0.)
        self.assertNotEqual(first["active_reference_deg"], 0.)
        self.assertEqual(first["effective"], 0.)
        self.assertEqual(second["active_reference_deg"], 0.)
        # No assertion that mapper compensation or physical motion instantly vanishes.

    def test_held_offsets_hard_and_slew_bounds_not_improvement_claim(self):
        for index, held in ((0,20.), (0,-20.), (0,3.7), (7,20.), (7,-20.)):
            for candidate in (False, True):
                with self.subTest(index=index, held=held, candidate=candidate):
                    plant = held_trial(candidate=candidate, index=index, held=held)
                    lower, upper = servo_limits_deg(SERVO_ORDER[index])
                    self.assertTrue(all(lower <= r["final"] <= upper for r in plant.rows))
                    self.assertTrue(all(abs(r["final_delta"]) <= 1.25+1e-12 for r in plant.rows))
                    self.assertTrue(all(abs(r["compensation"]) <= 10.+1e-12 for r in plant.rows))

    def test_blocked_contact_can_push_policy_aware_baseline_into_reserve(self):
        original = held_trial(candidate=False, index=0, nominal=130., held=3.7, blocked=True, ticks=128)
        candidate = held_trial(candidate=True, index=0, nominal=130., held=3.7, blocked=True, ticks=128)
        self.assertTrue(all(abs(r["actual_q_after"]-130.) < 1e-9 for r in candidate.rows))
        self.assertFalse(any(r["native"] > 133.+1e-9 for r in original.rows))
        self.assertTrue(any(r["native"] > 133.+1e-9 for r in candidate.rows))
        self.assertTrue(all(r["final"] <= 135. for r in candidate.rows))

    def test_same_history_target_delta_is_not_held_reference(self):
        actual = bounded_drive_feedback_step(previous_deg=20., native_deg=0., bias_deg=20.,
            maximum_delta_deg=1.25, lower_deg=-135., upper_deg=135.)
        zero = bounded_drive_feedback_step(previous_deg=20., native_deg=0., bias_deg=0.,
            maximum_delta_deg=1.25, lower_deg=-135., upper_deg=135.)
        self.assertEqual(actual-zero, 1.25)
        self.assertNotEqual(actual-zero, 20.)

    def test_unsafe_inputs_rejected(self):
        for bad in ((0.,)*7, (0.,)*9, (False,)+(0.,)*7, ("1",)+(0.,)*7,
                    (math.nan,)+(0.,)*7, (math.inf,)+(0.,)*7, None, "abcdefgh"):
            for field in ("measured_physical_rad", "previous_effective_residual_deg"):
                kwargs = dict(measured_physical_rad=ZERO8, previous_effective_residual_deg=ZERO8,
                              tracking_servo_names=())
                kwargs[field] = bad
                with self.subTest(field=field, bad=repr(bad)), self.assertRaises(ValueError):
                    active_tracking_reference(**kwargs)
        for names in (("bad",), (SERVO_ORDER[0],)*2, SERVO_ORDER[0], (False,), None):
            with self.subTest(names=names), self.assertRaises(ValueError):
                active_tracking_reference(measured_physical_rad=ZERO8,
                    previous_effective_residual_deg=ZERO8, tracking_servo_names=names)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional new JSON path; refuses overwrite")
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("output already exists; preserve earlier candidate evidence")
    result = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(CandidateTests))
    pairs = []
    for index, held in ((0,20.), (0,-20.), (0,3.7), (7,20.), (7,-20.)):
        pairs.append({"servo": SERVO_ORDER[index], "held_requested_deg": held,
            "original": summarize(held_trial(candidate=False,index=index,held=held)),
            "candidate": summarize(held_trial(candidate=True,index=index,held=held))})
    paths = ["src/wlr50_clean/infrastructure/servo_target_mapper.py",
        "src/wlr50_clean/infrastructure/command_batch.py",
        "src/wlr50_clean/infrastructure/robot_adapter.py", "src/wlr50_clean/ppo/semantic_headroom.py"]
    report = {"candidate_only": True, "production_wiring": False, "isaac_executed": False,
        "plant": "free: q_next=final_target; blocked: q_next=q_before; float64, no dynamics/contact solver",
        "not_proven": ["physical stability or task success", "ACK/tick/reset provenance",
            "float32 native buffers", "mapper-policy-reserve invariant", "observation Markov sufficiency"],
        "tests": {"run":result.testsRun, "failures":len(result.failures), "errors":len(result.errors),
                  "passed":result.wasSuccessful()},
        "source_sha256": {p:hashlib.sha256((PROJECT_ROOT/p).read_bytes()).hexdigest() for p in paths},
        "held_offset_comparisons": pairs,
        "blocking_contact_reserve_counterexample": {
            "original":summarize(held_trial(candidate=False,index=0,nominal=130.,held=3.7,blocked=True)),
            "candidate":summarize(held_trial(candidate=True,index=0,nominal=130.,held=3.7,blocked=True))}}
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(encoded+"\n")
    else:
        print(encoded)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
