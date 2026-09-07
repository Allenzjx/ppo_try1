"""Report-only variant 02; pure CPU toy plant, no production integration."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
from validate_tracking_reference_candidate import (PROJECT_ROOT, ZERO8, STANDING,
    ToyPlant, measured, vector, held_trial, summarize, SERVO_ORDER, servo_limits_deg,
    ServoTargetMapper, project_semantic_servo_headroom, bounded_drive_feedback_step)
from active_reference_candidate import active_tracking_reference
from bounded_active_reference_candidate import bounded_active_tracking_reference


def build(mapper, q, previous, names, nominal=ZERO8):
    return bounded_active_tracking_reference(measured_physical_rad=q,
        previous_effective_residual_deg=previous,tracking_servo_names=names,
        requested_command_deg=nominal,standing_pose_deg=mapper.standing_pose_deg,
        tracking_gain=mapper.tracking_gain,tracking_limit_deg=mapper.tracking_limit_deg)


class BoundedToyPlant(ToyPlant):
    def step(self, request, *, tracking=None, nominal=None):
        if nominal is not None:
            self.nominal=vector(self.index,nominal)
        names=(SERVO_ORDER[self.index],) if tracking is None else tuple(tracking)
        reference=build(self.mapper,measured(self.q),self.previous_effective,names,self.nominal)
        clock=self.mapper.feedback_tick
        mapped=self.mapper.advance(self.nominal,reference.mapper_computational_reference_rad,
                                   tracking_servo_names=names)
        if self.mapper.feedback_tick != clock+1:
            raise AssertionError("not exactly one mapper advance")
        headroom=project_semantic_servo_headroom(mapped.applied_drive_command_deg+(0.,)*4,
            (0.,)*12,vector(self.index,request)+(0.,)*4)
        effective=tuple(headroom["effective_policy_residual_full12"][:8])
        final=tuple(bounded_drive_feedback_step(previous_deg=previous,native_deg=native,
            bias_deg=r,maximum_delta_deg=self.mapper.maximum_delta_deg,
            lower_deg=servo_limits_deg(name)[0],upper_deg=servo_limits_deg(name)[1])
            for name,previous,native,r in zip(SERVO_ORDER,self.final,
                mapped.applied_drive_command_deg,effective,strict=True))
        old_q=self.q
        self.q=old_q if self.blocked and names else final
        i=self.index
        proof=reference.channels[i]
        row={"tick":clock,"sampled":mapped.feedback_sampled,"nominal":self.nominal[i],
            "actual_q_before":old_q[i],"actual_q_after":self.q[i],"request":request,
            "previous_effective":self.previous_effective[i],"effective":effective[i],
            "native":mapped.applied_drive_command_deg[i],"compensation":mapped.tracking_compensation_deg[i],
            "final":final[i],"final_delta":final[i]-self.final[i],
            "safe_band":headroom["servo_safety_limits_deg"][i],
            "baseline_outside_reserved_band":i in headroom["baseline_outside_reserved_servo_indices"],
            "reference_proof":proof}
        self.final,self.previous_effective=final,effective
        self.rows.append(row)
        return row


def trial(index=0,held=3.7,nominal=0.,blocked=False,ticks=960):
    plant=BoundedToyPlant(candidate=True,index=index,nominal=nominal,blocked=blocked)
    for tick in range(ticks):
        plant.step(math.copysign(min(abs(held),.5*(tick+1)),held))
    return plant


def summary(plant):
    result=summarize(plant)
    result["desired_clip_ticks"]=sum(r["reference_proof"]["clipped"] for r in plant.rows)
    result["max_positive_reserved_band_excess_deg"]=max(max(0.,r["native"]-r["safe_band"][1],
        r["safe_band"][0]-r["native"]) for r in plant.rows)
    result["max_inverse_error_deg"]=max((abs(r["reference_proof"]["reconstruction_error_deg"])
        for r in plant.rows if r["reference_proof"]["clipped"]),default=0.)
    return result


class BoundedTests(unittest.TestCase):
    def test_zero_gain_inactive_and_no_history_exact_original(self):
        for gain,previous,names in ((0.,vector(0,20.),SERVO_ORDER),(8.,ZERO8,SERVO_ORDER),
                                    (8.,vector(0,20.),())):
            mapper=ServoTargetMapper(STANDING,tracking_gain=gain)
            q=measured(vector(0,-2.))
            result=build(mapper,q,previous,names)
            self.assertEqual(result.mapper_computational_reference_rad,q)

    def test_no_clip_preserves_variant01_reference_bitwise(self):
        mapper=ServoTargetMapper(STANDING)
        for index,r in ((0,20.),(0,-20.),(0,3.7),(7,20.),(7,-20.)):
            q=measured(vector(index,r))
            names=(SERVO_ORDER[index],)
            old=active_tracking_reference(measured_physical_rad=q,
                previous_effective_residual_deg=vector(index,r),tracking_servo_names=names)
            new=build(mapper,q,vector(index,r),names)
            self.assertFalse(new.channels[index]["clipped"])
            self.assertEqual(new.mapper_computational_reference_rad,old.mapper_computational_reference_rad)

    def test_upper_and_lower_blocking_no_new_reserve_expansion(self):
        for nominal,held in ((130.,3.7),(-130.,-3.7)):
            plant=trial(nominal=nominal,held=held,blocked=True)
            self.assertTrue(any(r["reference_proof"]["clipped"] for r in plant.rows))
            self.assertLessEqual(summary(plant)["max_positive_reserved_band_excess_deg"],1e-12)
            self.assertLessEqual(summary(plant)["max_inverse_error_deg"],1e-12)
            self.assertTrue(all(-135. <= r["final"] <= 135. for r in plant.rows))

    def test_original_load_outside_reserved_band_is_allowed(self):
        for nominal,qactual,r in ((130.,110.,3.7),(-130.,-110.,-3.7)):
            a,b=ServoTargetMapper(STANDING),ServoTargetMapper(STANDING)
            n,q=vector(0,nominal),measured(vector(0,qactual))
            # Reach nominal via actual mapper calls, never copy private state.
            for _ in range(120):
                a.advance(n,measured(n),tracking_servo_names=())
                b.advance(n,measured(n),tracking_servo_names=())
            for _ in range(40):
                result=build(b,q,vector(0,r),(SERVO_ORDER[0],),n)
                old=a.advance(n,q,tracking_servo_names=(SERVO_ORDER[0],))
                new=b.advance(n,result.mapper_computational_reference_rad,tracking_servo_names=(SERVO_ORDER[0],))
                self.assertEqual(old,new)
            self.assertGreater(abs(new.applied_drive_command_deg[0]),133.)

    def test_ended_tracking_still_sees_actual_measurement(self):
        a,b=ServoTargetMapper(STANDING),ServoTargetMapper(STANDING)
        for _ in range(9):
            for mapper in (a,b):
                mapper.advance(ZERO8,measured(vector(0,-5.)),tracking_servo_names=(SERVO_ORDER[0],SERVO_ORDER[6]))
        q=measured(ZERO8)
        reference=build(b,q,(20.,0.,0.,0.,0.,0.,20.,0.),(SERVO_ORDER[6],))
        self.assertEqual(reference.mapper_computational_reference_rad[0],q[0])
        a.advance(ZERO8,q,tracking_servo_names=(SERVO_ORDER[6],))
        b.advance(ZERO8,reference.mapper_computational_reference_rad,tracking_servo_names=(SERVO_ORDER[6],))
        self.assertEqual(a._compensation[SERVO_ORDER[0]],b._compensation[SERVO_ORDER[0]])
        self.assertTrue(b._retiring_stale_bias[SERVO_ORDER[0]])

    def test_knee_rejected_request_previous_zero_unchanged(self):
        plant=BoundedToyPlant(candidate=True,index=7,nominal=-58.)
        for _ in range(24):
            row=plant.step(-20.)
            self.assertAlmostEqual(row["effective"],0.,places=10)
            self.assertEqual(row["reference_proof"]["status"],"original_no_reference")
            self.assertAlmostEqual(row["native"],-58.,places=10)

    def test_previous_compensation_outside_not_instantly_clipped(self):
        mapper=ServoTargetMapper(STANDING)
        n=vector(0,130.)
        for _ in range(120):
            mapper.advance(n,measured(n),tracking_servo_names=())
        for _ in range(20):
            mapper.advance(n,measured(vector(0,110.)),tracking_servo_names=(SERVO_ORDER[0],))
        before=mapper._compensation[SERVO_ORDER[0]]
        self.assertGreater(before,3.)
        reference=build(mapper,measured(n),vector(0,3.7),(SERVO_ORDER[0],),n)
        result=mapper.advance(n,reference.mapper_computational_reference_rad,tracking_servo_names=(SERVO_ORDER[0],))
        self.assertGreater(result.tracking_compensation_deg[0],3.)
        self.assertLessEqual(abs(result.tracking_compensation_deg[0]-before),1.25)

    def test_nominal_restart_clock_and_reference_withdrawal(self):
        plant=trial(ticks=24)
        clock=plant.mapper.feedback_tick
        first=plant.step(0.)
        second=plant.step(0.)
        self.assertNotEqual(first["reference_proof"]["status"],"original_no_reference")
        self.assertEqual(second["reference_proof"]["status"],"original_no_reference")
        changed=plant.step(0.,nominal=80.)
        self.assertEqual(changed["compensation"],0.)
        self.assertEqual(plant.mapper.feedback_tick,clock+3)

    def test_desired_band_all_servo_signs_and_clamped_nominal(self):
        mapper=ServoTargetMapper(STANDING)
        for i,name in enumerate(SERVO_ORDER):
            low,high=servo_limits_deg(name)
            for nominal,r in ((high-5.,3.7),(low+5.,-3.7),(high+10.,3.7),(low-10.,-3.7)):
                q=measured(vector(i,max(low,min(high,nominal))))
                record=build(mapper,q,vector(i,r),(name,),vector(i,nominal)).channels[i]
                lower,upper=record["allowed_desired_interval_deg"]
                self.assertLessEqual(lower,record["bounded_desired_c1_deg"])
                self.assertLessEqual(record["bounded_desired_c1_deg"],upper)
                self.assertLessEqual(abs(record["reconstructed_desired_deg"]-record["bounded_desired_c1_deg"]),1e-12)

    def test_invalid_configuration_rejected(self):
        kwargs=dict(measured_physical_rad=ZERO8,previous_effective_residual_deg=ZERO8,
            tracking_servo_names=(),requested_command_deg=ZERO8,standing_pose_deg=STANDING,
            tracking_gain=8.,tracking_limit_deg=10.)
        for key,value in (("tracking_gain",-1.),("tracking_gain",float("nan")),("tracking_gain",False),
            ("tracking_limit_deg",-1.),("tracking_limit_deg",float("inf")),("standing_pose_deg",{}),
            ("requested_command_deg",(0.,)*7)):
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                bounded_active_tracking_reference(**dict(kwargs,**{key:value}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        parser.error("preserve earlier evidence: output exists")
    result=unittest.TextTestRunner(stream=sys.stderr,verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(BoundedTests))
    comparisons=[]
    for index,held in ((0,20.),(0,-20.),(0,3.7),(7,20.),(7,-20.)):
        comparisons.append({"servo":SERVO_ORDER[index],"held_requested_deg":held,
            "original":summarize(held_trial(candidate=False,index=index,held=held)),
            "variant01":summarize(held_trial(candidate=True,index=index,held=held)),
            "variant02":summary(trial(index=index,held=held))})
    files=("active_reference_candidate.py","bounded_active_reference_candidate.py",
           "validate_tracking_reference_candidate.py","validate_bounded_reference_candidate.py")
    report={"candidate_only":True,"production_wiring":False,"isaac_executed":False,
        "tests":{"run":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),"passed":result.wasSuccessful()},
        "scope":"same-current-measurement desired c0, shared prior mapper state; not independent B trajectory",
        "not_proven":["real physical stability or task improvement","instant reserve correction of old compensation",
            "all-controller/geometry baseline reserve","ACK/reset/audit production provenance"],
        "source_sha256":{f:hashlib.sha256((Path(__file__).parent/f).read_bytes()).hexdigest() for f in files},
        "held_comparisons":comparisons,"upper_blocked":summary(trial(nominal=130.,held=3.7,blocked=True)),
        "lower_blocked":summary(trial(nominal=-130.,held=-3.7,blocked=True))}
    with args.output.open("x",encoding="utf-8") as handle:
        json.dump(report,handle,ensure_ascii=False,indent=2,allow_nan=False)
        handle.write("\n")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
