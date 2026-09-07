"""Small standard-library-only report harness; no physical success is generated."""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from typing import Sequence
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
from wheel_stop_effort_candidate import (physical_wheel_stop_effort, ORIGINAL_WHEEL_HARD_LIMIT,
    BLEND_WEIGHT,FAMILY_WEIGHT)

ROOT=Path(__file__).resolve().parents[4]
REWARD_PATH=ROOT/"src/wlr50_clean/ppo/semantic_reward.py"
CONFIG_PATH=ROOT/"configs/ppo_semantic_v3/reward_config.yaml"
FAMILIES=("task_progress","body_stability","contact_motion_quality","control_smoothness","control_regularization")
PLACED={leg:True for leg in ("FR","FL","RR","RL")}
ZERO12=(0.,)*12


def original_pure_functions():
    # Compile ONLY two pure numeric AST function bodies. Do not import the
    # reward module (which imports observation/PyYAML) or any runtime package.
    tree=ast.parse(REWARD_PATH.read_text(encoding="utf-8"))
    square=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="_square_cost"))
    config=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=="SemanticRewardConfig")
    bound=copy.deepcopy(next(n for n in config.body if isinstance(n,ast.FunctionDef) and n.name=="failure_avoidance_bound"))
    bound.decorator_list=[]
    ns={"Sequence":Sequence,"FAMILIES":FAMILIES}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[square,bound],type_ignores=[])),
                 str(REWARD_PATH),"exec"),ns)
    return ns["_square_cost"],ns["failure_avoidance_bound"]


def configuration_values():
    text=CONFIG_PATH.read_text(encoding="utf-8")
    names=("gamma","decision_hz","potential_weight","success_reward","failure_cost","time_cost_per_s")
    result={name:float(re.search(r"^"+name+r":\s*([0-9.]+)\s*$",text,re.MULTILINE).group(1)) for name in names}
    result["family_weights"]={name:float(re.search(r"^  "+name+r":\s*([0-9.]+)\s*$",text,re.MULTILINE).group(1)) for name in FAMILIES}
    if not re.search(r"^smoothness_components: applied_only$",text,re.MULTILINE):
        raise ValueError("candidate premise differs from current applied-only configuration")
    return result


ORIGINAL_SQUARE,ORIGINAL_BOUND=original_pure_functions()
VALUES=configuration_values()


def stable(wheels,history=None):
    actual=(0.,)*8+tuple(wheels)
    return physical_wheel_stop_effort(actual_drive_full12=actual,
        previous_actual_drive_full12=actual,previous_previous_actual_drive_full12=actual,
        dt_s=1./120.,placed_history=PLACED if history is None else history)


def assumed_event_ledger(*, duration_s, terminal_event, commanded_segments):
    """Algebraic reward budget, NOT an evaluated state trajectory or policy.

    Segment tuple=(start second,end second,absolute mean wheel command).
    All placement-history flags are ASSUMED established before these segments.
    Body/contact and derivative costs are ASSUMED zero to isolate this term.
    Endpoint ramp costs are not modeled. No hard evaluator/validity flag exists.
    """
    hz,gamma=VALUES["decision_hz"],VALUES["gamma"]
    count=int(round(duration_s*hz))
    if count != duration_s*hz or count < 1:
        raise ValueError("duration must be a positive whole decision interval")
    wheel_integral=0.
    discounted_wheel=0.
    for start,end,command in commanded_segments:
        a,b=int(round(start*hz)),int(round(end*hz))
        if not (0<=a<b<=count) or not 0<=command<=ORIGINAL_WHEEL_HARD_LIMIT:
            raise ValueError("invalid assumed command segment")
        rate=FAMILY_WEIGHT*BLEND_WEIGHT*command/ORIGINAL_WHEEL_HARD_LIMIT
        wheel_integral+=rate*(end-start)
        discounted_wheel+=rate/hz*gamma**a*(1.-gamma**(b-a))/(1.-gamma)
    time_undiscounted=VALUES["time_cost_per_s"]*duration_s
    time_discounted=VALUES["time_cost_per_s"]/hz*(1.-gamma**count)/(1.-gamma)
    # Initial phi=0 and terminal phi=0 imply a telescoping discounted PBRS
    # contribution of zero; intermediate phis are unspecified, not fabricated.
    return {"assumptions_only":True,"duration_s":duration_s,"decisions":count,
        "assumed_terminal_event":terminal_event,"command_segments":commanded_segments,
        "undiscounted_time_cost":time_undiscounted,"undiscounted_added_wheel_cost":wheel_integral,
        "undiscounted_non_PBRS_ledger":terminal_event-time_undiscounted-wheel_integral,
        "discounted_added_wheel_cost":discounted_wheel,
        "discounted_ledger_phi0_zero":gamma**(count-1)*terminal_event-time_discounted-discounted_wheel}


class EffortTests(unittest.TestCase):
    def test_opposite_commands_zero_arithmetic_body_mean_still_cost(self):
        wheels=(.4,-.4,.4,-.4)
        self.assertEqual(sum(wheels),0.)
        self.assertGreater(stable(wheels)["candidate_control_smoothness_cost"],0.)

    def test_nominal_cancelled_by_residual_actual_zero_cost(self):
        nominal=(.3,)*4
        residual=(-.3,)*4
        actual=tuple(n+r for n,r in zip(nominal,residual,strict=True))
        self.assertEqual(stable(actual)["candidate_control_smoothness_cost"],0.)

    def test_phase_or_current_region_not_an_input(self):
        for phase,region in (("P09",True),("P13",True),("P13",False),("P06",False)):
            with self.subTest(phase=phase,current_region=region):
                self.assertEqual(stable((.2,)*4),stable((.2,)*4,dict(PLACED)))
        import inspect
        args=inspect.signature(physical_wheel_stop_effort).parameters
        self.assertTrue(set(args).isdisjoint({"phase_id","current_region","nominal","residual","task_success"}))

    def test_any_history_missing_exact_old_applied_derivative(self):
        actual=(.2,)*8+(.1,)*4
        dt=1./120.
        rates=(60.,)*8+(1.8,)*4
        old=(ORIGINAL_SQUARE(tuple(x/dt for x in actual),rates)+
            ORIGINAL_SQUARE(tuple(x/dt**2 for x in actual),tuple(x*120. for x in rates)))/2.
        for leg in PLACED:
            result=physical_wheel_stop_effort(actual_drive_full12=actual,
                previous_actual_drive_full12=ZERO12,previous_previous_actual_drive_full12=ZERO12,
                dt_s=dt,placed_history=dict(PLACED,**{leg:False}))
            self.assertEqual(result["candidate_control_smoothness_cost"],old)
            self.assertEqual(result["reward_delta"],0.)

    def test_mean_four_not_max_and_permutation_symmetry(self):
        hard=ORIGINAL_WHEEL_HARD_LIMIT
        one=stable((hard,0.,0.,0.))["candidate_control_smoothness_cost"]
        four=stable((hard,)*4)["candidate_control_smoothness_cost"]
        self.assertEqual(one,.125)
        self.assertEqual(four,4*one)
        for index in range(4):
            wheels=[0.]*4
            wheels[index]=-hard
            self.assertEqual(stable(wheels)["candidate_control_smoothness_cost"],one)

    def test_magnitude_monotonic(self):
        values=[stable((x,-x,x,-x))["candidate_control_smoothness_cost"] for x in (0.,.02,.1,.5,ORIGINAL_WHEEL_HARD_LIMIT)]
        self.assertTrue(all(a<b for a,b in zip(values,values[1:])))

    def test_family_bound_weight_and_derivative_tradeoff(self):
        actual=(.5,)*8+(ORIGINAL_WHEEL_HARD_LIMIT,)*4
        result=physical_wheel_stop_effort(actual_drive_full12=actual,previous_actual_drive_full12=ZERO12,
            previous_previous_actual_drive_full12=ZERO12,dt_s=1./120.,placed_history=PLACED)
        self.assertEqual(result["old_applied_derivative_cost"],1.)
        self.assertEqual(result["candidate_control_smoothness_cost"],1.)
        self.assertAlmostEqual(result["candidate_control_smoothness_reward"],-.1/120.)
        # Blending is NOT always an extra penalty: derivative quality loses
        # half its weight after all placed when actual wheel effort is zero.
        changed=physical_wheel_stop_effort(actual_drive_full12=(.5,)*8+(0.,)*4,
            previous_actual_drive_full12=ZERO12,previous_previous_actual_drive_full12=ZERO12,
            dt_s=1./120.,placed_history=PLACED)
        self.assertGreater(changed["reward_delta"],0.)

    def test_actual_dt_integration_and_equal_commanded_angular_distance(self):
        row=stable((.2,)*4)
        self.assertAlmostEqual(120*row["candidate_control_smoothness_reward"],-.05*.2/ORIGINAL_WHEEL_HARD_LIMIT)
        self.assertAlmostEqual(.2*10.,.02*100.)
        fast=assumed_event_ledger(duration_s=80.,terminal_event=40.,commanded_segments=((60.,70.,.2),))
        slow=assumed_event_ledger(duration_s=180.,terminal_event=40.,commanded_segments=((60.,160.,.02),))
        self.assertAlmostEqual(fast["undiscounted_added_wheel_cost"],slow["undiscounted_added_wheel_cost"])

    def test_existing_failure_avoidance_bound_unchanged(self):
        config=SimpleNamespace(values=VALUES,gamma=VALUES["gamma"])
        original=ORIGINAL_BOUND(config)
        candidate=(sum(VALUES["family_weights"][name] for name in FAMILIES[1:])+
            VALUES["time_cost_per_s"]*VALUES["family_weights"]["task_progress"])/VALUES["decision_hz"]/(1.-VALUES["gamma"])+VALUES["potential_weight"]
        self.assertEqual(original,candidate)
        self.assertAlmostEqual(candidate,14.6)
        self.assertGreater(VALUES["failure_cost"],candidate)
        self.assertEqual(VALUES["family_weights"]["control_smoothness"],FAMILY_WEIGHT)

    def test_original_hard_limit_literal_matches_source(self):
        tree=ast.parse((ROOT/"src/wlr50_clean/infrastructure/command_batch.py").read_text(encoding="utf-8"))
        node=next(n for n in tree.body if isinstance(n,ast.Assign) and
                  any(isinstance(target,ast.Name) and target.id=="WHEEL_VELOCITY_LIMIT_RAD_S" for target in n.targets))
        self.assertEqual(ast.literal_eval(node.value),ORIGINAL_WHEEL_HARD_LIMIT)

    def test_assumed_event_comparison_not_evaluator_success(self):
        reasonable=assumed_event_ledger(duration_s=80.,terminal_event=40.,commanded_segments=((60.,70.,.2),))
        slow=assumed_event_ledger(duration_s=180.,terminal_event=40.,commanded_segments=((60.,160.,.02),))
        idle=assumed_event_ledger(duration_s=200.,terminal_event=-40.,commanded_segments=())
        failure=assumed_event_ledger(duration_s=20.,terminal_event=-40.,commanded_segments=())
        for key in ("undiscounted_non_PBRS_ledger","discounted_ledger_phi0_zero"):
            self.assertGreater(reasonable[key],slow[key])
            self.assertGreater(reasonable[key],idle[key])
            self.assertGreater(reasonable[key],failure[key])
        for row in (reasonable,slow,idle,failure):
            self.assertNotIn("task_success",row)
            self.assertNotIn("physical_valid",row)

    def test_failure_bound_does_not_guarantee_all_long_success_rankings(self):
        slow=assumed_event_ledger(duration_s=180.,terminal_event=40.,commanded_segments=((60.,160.,.02),))
        idle=assumed_event_ledger(duration_s=200.,terminal_event=-40.,commanded_segments=())
        # Counterexample under these explicitly assumed costs/times; not an
        # observed policy exploit or a fabricated successful physical trace.
        self.assertLess(slow["discounted_ledger_phi0_zero"],idle["discounted_ledger_phi0_zero"])

    def test_delaying_fourth_placement_can_avoid_local_running_cost(self):
        complete_history=stable((.2,)*4,PLACED)
        pending_history=stable((.2,)*4,dict(PLACED,RL=False))
        self.assertLess(complete_history["candidate_control_smoothness_reward"],
                        pending_history["candidate_control_smoothness_reward"])
        # If terminal time/event and other costs were equal, discounted PBRS
        # telescopes; those assumptions can leave an incentive to delay this
        # history transition. This is not claimed to be physically feasible.

    def test_invalid_inputs(self):
        kwargs=dict(actual_drive_full12=ZERO12,previous_actual_drive_full12=ZERO12,
            previous_previous_actual_drive_full12=ZERO12,dt_s=1./120.,placed_history=PLACED)
        for key,value in (("actual_drive_full12",(0.,)*11),("actual_drive_full12",(float("nan"),)+(0.,)*11),
            ("actual_drive_full12",(0.,)*8+(3.,)*4),("dt_s",0.),("dt_s",.1),("dt_s",True),
            ("placed_history",dict(PLACED,FR=1)),("placed_history",{"FR":True})):
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                physical_wheel_stop_effort(**dict(kwargs,**{key:value}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): parser.error("preserve earlier evidence: output exists")
    if any(n.split(".")[0] in {"torch","isaaclab","isaacsim","omni","yaml"} for n in sys.modules):
        raise RuntimeError("candidate must remain standard-library-only")
    result=unittest.TextTestRunner(stream=sys.stderr,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(EffortTests))
    cases={"assumed_reasonable_completion":assumed_event_ledger(duration_s=80.,terminal_event=40.,commanded_segments=((60.,70.,.2),)),
        "assumed_ultraslow_completion":assumed_event_ledger(duration_s=180.,terminal_event=40.,commanded_segments=((60.,160.,.02),)),
        "assumed_idle_deadline":assumed_event_ledger(duration_s=200.,terminal_event=-40.,commanded_segments=()),
        "assumed_early_failure":assumed_event_ledger(duration_s=20.,terminal_event=-40.,commanded_segments=())}
    report={"candidate_only":True,"production_wiring":False,"simulated_robot":False,
        "tests":{"run":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),"passed":result.wasSuccessful()},
        "existing_config_values":VALUES,"unchanged_failure_avoidance_bound":ORIGINAL_BOUND(SimpleNamespace(values=VALUES,gamma=VALUES["gamma"])),
        "constant_command_cases":{str(x):stable((x,)*4) for x in (0.,.02,.1,.4,ORIGINAL_WHEEL_HARD_LIMIT)},
        "assumed_event_ledgers":cases,"source_sha256":{str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (REWARD_PATH,CONFIG_PATH,Path(__file__),Path(__file__).with_name("wheel_stop_effort_candidate.py"))}}
    with args.output.open("x",encoding="utf-8") as handle:
        json.dump(report,handle,ensure_ascii=False,indent=2,allow_nan=False)
        handle.write("\n")
    return 0 if result.wasSuccessful() else 1


if __name__=="__main__":
    raise SystemExit(main())
