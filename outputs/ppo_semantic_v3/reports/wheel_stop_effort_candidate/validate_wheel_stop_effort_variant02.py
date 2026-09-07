"""Standard-library-only candidate 02 tests; no robot/policy/evaluator creation."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
from validate_wheel_stop_effort_candidate import (VALUES, ORIGINAL_BOUND, ORIGINAL_SQUARE,
    PLACED,ZERO12,FAMILIES,assumed_event_ledger)
from wheel_stop_effort_candidate import (physical_wheel_stop_effort,ORIGINAL_WHEEL_HARD_LIMIT,FAMILY_WEIGHT)
from wheel_stop_effort_variant02 import physical_wheel_stop_effort_v2,remaining_capacity_cost


def sample(wheels, *, history=None, previous=None, older=None, servos=(0.,)*8):
    actual=tuple(servos)+tuple(wheels)
    return physical_wheel_stop_effort_v2(actual_drive_full12=actual,
        previous_actual_drive_full12=actual if previous is None else previous,
        previous_previous_actual_drive_full12=actual if older is None else older,
        dt_s=1./120.,placed_history=PLACED if history is None else history)


class RemainingCapacityTests(unittest.TestCase):
    def test_all_bounds_and_preserved_derivative_grid(self):
        for s in (0.,.001,.1,.25,.5,.75,.999,1.):
            for e in (0.,.001,.1,.25,.5,.75,.999,1.):
                with self.subTest(s=s,e=e):
                    c=remaining_capacity_cost(old_smoothness=s,wheel_effort=e,all_placed_history=True)
                    self.assertGreaterEqual(c,s)
                    self.assertGreaterEqual(c,.5*e-1e-15)
                    self.assertLessEqual(c,1.)
                    self.assertEqual(remaining_capacity_cost(old_smoothness=s,wheel_effort=e,all_placed_history=False),s)

    def test_derivative_monotonic_and_effort_nondecreasing(self):
        for e in (0.,.3,.8,1.):
            costs=[remaining_capacity_cost(old_smoothness=s,wheel_effort=e,all_placed_history=True)
                   for s in (0.,.1,.5,.9,1.)]
            self.assertTrue(all(a<b for a,b in zip(costs,costs[1:])))
        for s in (0.,.3,.8,1.):
            costs=[remaining_capacity_cost(old_smoothness=s,wheel_effort=e,all_placed_history=True)
                   for e in (0.,.1,.5,.9,1.)]
            self.assertTrue(all(a<=b for a,b in zip(costs,costs[1:])))
        self.assertEqual(remaining_capacity_cost(old_smoothness=1.,wheel_effort=0.,all_placed_history=True),
                         remaining_capacity_cost(old_smoothness=1.,wheel_effort=1.,all_placed_history=True))

    def test_servo_derivative_cost_no_longer_halved(self):
        kwargs=dict(actual_drive_full12=(.5,)*8+(0.,)*4,previous_actual_drive_full12=ZERO12,
            previous_previous_actual_drive_full12=ZERO12,dt_s=1./120.,placed_history=PLACED)
        old_blend=physical_wheel_stop_effort(**kwargs)
        v2=physical_wheel_stop_effort_v2(**kwargs)
        self.assertGreater(old_blend["reward_delta"],0.)
        self.assertEqual(v2["candidate_control_smoothness_cost"],v2["old_applied_derivative_cost"])
        self.assertEqual(v2["reward_delta"],0.)

    def test_pending_any_placement_exact_original_applied_derivatives(self):
        actual=(.2,)*8+(.1,)*4
        rates=(60.,)*8+(1.8,)*4
        dt=1./120.
        original=(ORIGINAL_SQUARE(tuple(x/dt for x in actual),rates)+
            ORIGINAL_SQUARE(tuple(x/dt**2 for x in actual),tuple(x*120. for x in rates)))/2.
        for leg in PLACED:
            row=sample((.1,)*4,servos=(.2,)*8,previous=ZERO12,older=ZERO12,history=dict(PLACED,**{leg:False}))
            self.assertEqual(row["candidate_control_smoothness_cost"],original)
            self.assertEqual(row["reward_delta"],0.)

    def test_actual_not_residual_opposing_wheels_with_stationary_body_assumption(self):
        self.assertGreater(sample((.4,-.4,.4,-.4))["candidate_control_smoothness_cost"],0.)
        # Body velocity is neither inferred from wheel sums nor passed to the
        # helper: cost stays positive if an external state has body velocity 0.
        nominal,residual=(.3,)*4,(-.3,)*4
        self.assertEqual(sample(tuple(n+r for n,r in zip(nominal,residual,strict=True)))["candidate_control_smoothness_cost"],0.)

    def test_phase_and_region_absent_persistent_history_gate(self):
        self.assertTrue(set(inspect.signature(physical_wheel_stop_effort_v2).parameters).isdisjoint(
            {"phase_id","current_region","nominal","residual","task_success"}))
        for phase,region in (("P06",True),("P13",True),("P13",False)):
            with self.subTest(phase=phase,region=region):
                self.assertEqual(sample((.2,)*4),sample((.2,)*4,history=dict(PLACED)))

    def test_all_four_mean_symmetry_not_max(self):
        h=ORIGINAL_WHEEL_HARD_LIMIT
        self.assertEqual(sample((h,)*4)["candidate_control_smoothness_cost"],.5)
        for i in range(4):
            wheel=[0.]*4
            wheel[i]=-h
            self.assertEqual(sample(wheel)["candidate_control_smoothness_cost"],.125)

    def test_actual_magnitude_monotone_and_family_max_weight(self):
        costs=[sample((x,)*4)["candidate_control_smoothness_cost"] for x in (0.,.01,.1,.4,ORIGINAL_WHEEL_HARD_LIMIT)]
        self.assertTrue(all(a<b for a,b in zip(costs,costs[1:])))
        row=sample((ORIGINAL_WHEEL_HARD_LIMIT,)*4,servos=(.5,)*8,previous=ZERO12,older=ZERO12)
        self.assertEqual(row["candidate_control_smoothness_cost"],1.)
        self.assertEqual(row["candidate_control_smoothness_reward"],-.1/120.)
        self.assertLessEqual(row["reward_delta"],0.)

    def test_original_failure_bound_and_dt_integration(self):
        bound=ORIGINAL_BOUND(SimpleNamespace(values=VALUES,gamma=VALUES["gamma"]))
        candidate=(sum(VALUES["family_weights"][k] for k in FAMILIES[1:])+VALUES["time_cost_per_s"])/15./(1.-VALUES["gamma"])+VALUES["potential_weight"]
        self.assertEqual(bound,candidate)
        self.assertAlmostEqual(bound,14.6)
        self.assertGreater(40.,bound)
        self.assertAlmostEqual(120.*sample((.2,)*4)["candidate_control_smoothness_reward"],-.05*.2/ORIGINAL_WHEEL_HARD_LIMIT)

    def test_delay_gate_and_long_discount_counterexamples_remain(self):
        self.assertLess(sample((.2,)*4)["candidate_control_smoothness_reward"],
                        sample((.2,)*4,history=dict(PLACED,RL=False))["candidate_control_smoothness_reward"])
        # S=0 means both candidates have exactly the same effort cost, so the
        # original explicitly assumed event ledgers still apply, not real success.
        slow=assumed_event_ledger(duration_s=180.,terminal_event=40.,commanded_segments=((60.,160.,.02),))
        idle=assumed_event_ledger(duration_s=200.,terminal_event=-40.,commanded_segments=())
        self.assertLess(slow["discounted_ledger_phi0_zero"],idle["discounted_ledger_phi0_zero"])

    def test_invalid_costs_history_and_actual_command(self):
        for s,e,gate in ((-.1,0.,True),(1.1,0.,True),(0.,-.1,True),(0.,1.1,True),
                         (float("nan"),0.,True),(0.,float("inf"),True),(False,0.,True),(0.,0.,1)):
            with self.subTest(s=s,e=e,gate=gate),self.assertRaises(ValueError):
                remaining_capacity_cost(old_smoothness=s,wheel_effort=e,all_placed_history=gate)
        for wheel,history in (((3.,)*4,PLACED),((float("nan"),)*4,PLACED),((0.,)*4,dict(PLACED,FR=1))):
            with self.subTest(wheel=wheel,history=history),self.assertRaises(ValueError):
                sample(wheel,history=history)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): parser.error("preserve previous receipts: output exists")
    if any(n.split(".")[0] in {"torch","isaaclab","isaacsim","omni","yaml"} for n in sys.modules):
        raise RuntimeError("candidate must remain standard-library-only")
    result=unittest.TextTestRunner(stream=sys.stderr,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(RemainingCapacityTests))
    matrix=[{"old_S":s,"wheel_E":e,"variant01_C":.5*s+.5*e,
        "variant02_C":remaining_capacity_cost(old_smoothness=s,wheel_effort=e,all_placed_history=True)}
        for s in (0.,.25,.5,.75,1.) for e in (0.,.25,.5,.75,1.)]
    report={"candidate_only":True,"production_wiring":False,"simulated_robot":False,
        "formula":"C=S+0.5*(1-S)*E after all four hard placed history; otherwise S",
        "tests":{"run":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),"passed":result.wasSuccessful()},
        "unchanged_failure_avoidance_bound":ORIGINAL_BOUND(SimpleNamespace(values=VALUES,gamma=VALUES["gamma"])),
        "cost_matrix":matrix,"constant_command_cases":{str(x):sample((x,)*4) for x in (0.,.02,.1,.4,ORIGINAL_WHEEL_HARD_LIMIT)},
        "assumed_zero_S_event_ledgers":{
            "reasonable_completion":assumed_event_ledger(duration_s=80.,terminal_event=40.,commanded_segments=((60.,70.,.2),)),
            "ultraslow_completion":assumed_event_ledger(duration_s=180.,terminal_event=40.,commanded_segments=((60.,160.,.02),)),
            "idle_deadline":assumed_event_ledger(duration_s=200.,terminal_event=-40.,commanded_segments=()),
            "early_failure":assumed_event_ledger(duration_s=20.,terminal_event=-40.,commanded_segments=())},
        "source_sha256":{name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in
            ("wheel_stop_effort_candidate.py","validate_wheel_stop_effort_candidate.py",
             "wheel_stop_effort_variant02.py","validate_wheel_stop_effort_variant02.py")}}
    with args.output.open("x",encoding="utf-8") as handle:
        json.dump(report,handle,ensure_ascii=False,indent=2,allow_nan=False)
        handle.write("\n")
    return 0 if result.wasSuccessful() else 1


if __name__=="__main__":
    raise SystemExit(main())
