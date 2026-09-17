"""CPU replay of recorded composition and directed static-N counterexamples."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))


def main():
    from wlr50_clean.ppo.semantic_nominal_geometry import _bounded_rr_correction, BOUNDED_RR_MODE
    from wlr50_clean.infrastructure.robot_adapter import bounded_drive_feedback_step
    from wlr50_clean.ppo.semantic_nominal_geometry import SERVO_ORDER
    from wlr50_clean.ppo.semantic_training import write_json
    source=ROOT/"runs/ppo_task_first_recovery_v1/video_eval/validation/20260916T0428416958414Z_gb0438f66ec63_292464ef72094ece8a8e494452b8aa23/source/native_tick_audit.jsonl"
    real=[]
    with source.open() as stream:
        for line in stream:
            row=json.loads(line)
            if row["episode_physics_tick"]<6200:continue
            a=row["native_audit"]
            g=a.get("nominal_geometry_evidence")
            if not g:continue
            previous=a["previous_final_drive_servo_deg"]
            geom=a["geometry_adjusted_native_full12"]
            residual=a["policy_headroom_evidence"]["effective_policy_residual_full12"]
            final=[]
            for i in (6,7):
                final.append(bounded_drive_feedback_step(previous_deg=previous[i],native_deg=geom[i],
                    bias_deg=residual[i],maximum_delta_deg=1.25,lower_deg=-135. if i==6 else -60.,
                    upper_deg=135. if i==6 else 210.))
            real.append({"tick":row["episode_physics_tick"],"status":g["status"],
                "source_N_RR_deg":row["nominal_full12"][6:8],"mapped_N_RR_deg":a["native_drive_target_full12"][6:8],
                "previous_final_RR_deg":previous[6:8],"geometry_RR_deg":geom[6:8],
                "effective_policy_RR_deg":residual[6:8],"replayed_final_RR_deg":final,
                "recovery_flags":g.get("slew_recovery_toward_source_band"),
                "source_bands":g.get("source_relative_operating_bands_deg"),
                "context":g["context"],"standing":a["tracking_reference_evidence"]["standing_pose_deg"]})
    last=real[-1]
    # The same production helper, with a fixed source/actual state and known
    # out-of-band final target. This isolates composition, not robot dynamics.
    context=copy.deepcopy(last["context"])
    context.update(mode=BOUNDED_RR_MODE,physical_q_rad=[0.,0.],clearance_m=0.,clearance_margin_m=0.)
    nominal=[0.]*12
    def directed(initial,residual,*,subtract_previous_policy=False):
        actual_previous=[0.]*8
        actual_previous[6:8]=initial
        previous_policy=[0.]*12
        previous_policy[6:8]=residual
        seq=[]
        for step in range(8):
            reference=list(actual_previous)
            if subtract_previous_policy:
                reference=[x-y for x,y in zip(reference,previous_policy[:8])]
            adapter=SimpleNamespace(servo_target_mapper=SimpleNamespace(maximum_delta_deg=1.25),
                standing_pose_deg=dict(zip(SERVO_ORDER,[0.]*8)),
                _final_drive_servo_deg=dict(zip(SERVO_ORDER,reference)))
            adjusted,evidence=_bounded_rr_correction(adapter=adapter,native=tuple(nominal),bias=(0.,)*12,context=context)
            final=list(actual_previous)
            for i,r in zip((6,7),residual):
                final[i]=bounded_drive_feedback_step(previous_deg=actual_previous[i],native_deg=adjusted[i],bias_deg=r,
                    maximum_delta_deg=1.25,lower_deg=-135. if i==6 else -60.,upper_deg=135. if i==6 else 210.)
            seq.append({"step":step+1,"previous":actual_previous[6:8],"geometry":list(adjusted[6:8]),
                "residual":residual,"final":final[6:8],"status":evidence["status"]})
            actual_previous=final
        return seq
    cases={
        "zero_residual_recovery":directed([20.,-20.],[0.,0.]),
        "constant_positive_RRhip_residual":directed([20.,0.],[2.56,0.]),
        "constant_negative_RRknee_residual":directed([0.,-20.],[0.,-1.786]),
        "candidate_previous_policy_exclusion_positive":directed([20.,0.],[2.56,0.],subtract_previous_policy=True),
        "candidate_previous_policy_exclusion_negative":directed([0.,-20.],[0.,-1.786],subtract_previous_policy=True),
        "candidate_zero_identity":directed([20.,-20.],[0.,0.],subtract_previous_policy=True),
    }
    assert cases["zero_residual_recovery"]==cases["candidate_zero_identity"]
    assert cases["constant_positive_RRhip_residual"][-1]["final"][0]>20.
    assert cases["constant_negative_RRknee_residual"][-1]["final"][1]<-20.
    assert cases["candidate_previous_policy_exclusion_positive"][-1]["final"][0]<20.
    assert cases["candidate_previous_policy_exclusion_negative"][-1]["final"][1]>-20.
    first=next((r for r in real if any(r["recovery_flags"])),None)
    compact=lambda r:None if r is None else {k:v for k,v in r.items() if k not in ("context","standing")}
    output={"schema":"wlr50_clean.rr_geometry_residual_recurrence_repro.v1","source":str(source),
        "first_recorded_slew_recovery_in_window":compact(first),"last_recorded":compact(last),
        "directed_cases":cases,"tests_passed":6,
        "confirmed_composition_path":"previous combined final -> nominal geometry hold/recovery -> add current policy offset -> final slew -> persist combined final",
        "candidate_fix_is_only_offline_proxy_not_implemented":"Exclude independently recorded previous post-projection policy offset from nominal geometry history; keep actual final history for final safety slew. Need tests for effective-vs-requested offset, headroom clipping, phase handoff and residual removal before production.",
        "not_complete_collision_causal_proof":True,"no_physics_or_optimizer_steps":True}
    path=ROOT/"outputs/ppo_task_first_recovery_v1/c176768_rr_composition_recurrence.json"
    write_json(path,output)
    print(json.dumps({"path":str(path),"first":compact(first),"last":compact(last),
        "endpoints":{k:v[-1]["final"] for k,v in cases.items()}},indent=2))


if __name__=="__main__":main()
