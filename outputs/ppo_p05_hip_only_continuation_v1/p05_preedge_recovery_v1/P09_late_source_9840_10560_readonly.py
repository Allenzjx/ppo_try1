"""Fixed completed current-run source/residual/physics comparison; no forward/sim."""
from collections import Counter
import itertools
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT / "src"))
from wlr50_clean.ppo.semantic_observation import _rpy, _quaternion
from wlr50_clean.infrastructure.command_batch import FULL12_ORDER, WHEEL_ORDER

SOURCE=ROOT/"runs/ppo_p05_hip_only_continuation_v1/validation/20260922T1641475169869Z_g6ac7b553d792_13764630c7a0468285bacb6aa3607ee8"
FIRST,LAST=9840,10560
def window(name,start,stop,key):
    with (SOURCE/name).open("rb") as stream:
        rows=[json.loads(line) for line in itertools.islice(stream,start,stop) if line.endswith(b"\n")]
    return {row[key]:row for row in rows}
native=window("native_tick_audit.jsonl",FIRST-2,LAST,"episode_physics_tick")
physics=window("physical_observations.jsonl",FIRST-1,LAST+1,"physics_tick")
decisions=window("residual_and_projection_audit.jsonl",FIRST//8-1,LAST//8,"physics_tick")
assert sorted(native)==sorted(physics)==list(range(FIRST-1,LAST+1))
assert sorted(decisions)==list(range(FIRST,LAST+1,8))
def geometry(row):
    obstacle=row["obstacle"];rr=row["wheels"]["rear_right_ankle"]
    return dict(body_z_m=row["base"]["position_w_m"][2],
        body_pitch_deg=math.degrees(_rpy(_quaternion(row["base"]["orientation_wxyz"]))[1]),
        RR_gap_m=rr["bottom_w_m"][2]-obstacle["top_z_m"],
        RR_front_m=rr["center_w_m"][0]-obstacle["front_x_m"],
        RR_pair_active={key:row["contacts"]["rear_right_wheel"][key]["active"] for key in ("ground","obstacle")})
changes=[]
for tick in range(FIRST,LAST+1):
    before=native[tick-1]["nominal_full12"];after=native[tick]["nominal_full12"]
    changed=[i for i,(a,b) in enumerate(zip(before,after,strict=True)) if a!=b]
    if changed:
        changes.append(dict(episode_dispatch_tick=tick,channels=[FULL12_ORDER[i] for i in changed],
            before=before,after=after))
clipped=Counter();mask_ok=True;assist_states=Counter();geometry_corrections=Counter()
geometry_active_ticks=[]
for tick in range(FIRST,LAST+1):
    audit=native[tick]["native_audit"];head=audit["policy_headroom_evidence"]
    clipped.update(FULL12_ORDER[i] for i in head["clipped_servo_indices"])
    mask_ok &= audit["phase_mask_full12"]==[1]*12
    assist_states.update([audit["capture_assist_evidence"]["state_after"]["mode_name"]])
    assert audit["verified"] and audit["actual_mapping_matches_dispatch"] and audit["setter_dispatch_targets_equal"]
    for i,(a,b) in enumerate(zip(audit["native_drive_target_full12"],head["geometry_corrected_native_full12"],strict=True)):
        if a!=b:geometry_corrections.update([FULL12_ORDER[i]])
    if any(audit["native_drive_target_full12"][i]!=head["geometry_corrected_native_full12"][i] for i in (6,7)):
        geometry_active_ticks.append(tick)

snapshots=[]
for tick in (9840,9872,9960,9965,9966,9968,10000,10448,10560):
    row=native[tick];audit=row["native_audit"];head=audit["policy_headroom_evidence"]
    measured=physics[tick];final=measured["commanded_full12"]
    per_leg={}
    for i,leg in enumerate(("FL","FR","RL","RR")):
        a,b=2*i,2*i+2
        per_leg[leg]=dict(nominal_hip_knee_deg=row["nominal_full12"][a:b],
            mapped_nominal_hip_knee_deg=audit["native_drive_target_full12"][a:b],
            geometry_corrected_baseline_hip_knee_deg=head["geometry_corrected_native_full12"][a:b],
            requested_residual_hip_knee_deg=head["requested_policy_residual_full12"][a:b],
            effective_residual_hip_knee_deg=head["effective_policy_residual_full12"][a:b],
            effective_combined_postmapper_hip_knee_deg=head["effective_combined_post_mapper_bias_full12"][a:b],
            final_hip_knee_deg=final[a:b],actual_hip_knee_deg=measured["actual_full12"][a:b],
            actual_minus_final_deg=[x-y for x,y in zip(measured["actual_full12"][a:b],final[a:b],strict=True)],
            final_reserved_margin_deg=[min(final[j]-head["servo_safety_limits_deg"][j][0],
                head["servo_safety_limits_deg"][j][1]-final[j]) for j in (a,a+1)],
            headroom_clipped_channels=[FULL12_ORDER[j] for j in head["clipped_servo_indices"] if j in (a,a+1)],
            wheel_nominal_rad_s=row["nominal_full12"][8+i],
            wheel_mapped_nominal_rad_s=audit["native_drive_target_full12"][8+i],
            wheel_residual_rad_s=head["effective_policy_residual_full12"][8+i],
            wheel_final_rad_s=final[8+i],wheel_actual_rad_s=measured["actual_full12"][8+i])
    item=dict(tick=tick,time_s=tick/120.,phase=row["source_phase_id"],legs=per_leg,**geometry(measured),
        assist=audit["capture_assist_evidence"]["state_after"]["mode_name"])
    if tick in decisions:
        task=decisions[tick]["semantic_task"];diag=task["nominal_provider_diagnostics"];rr=task["physical_evaluator"]["current_legs"]["RR"]
        assert abs(item["RR_gap_m"]-rr["clearance_m"])<1e-12
        item.update(RR_current_Q=rr["current_lift_valid"],RR_history=task["physical_evaluator"]["history"]["event_ticks"],
            RR_current_contact={k:rr.get(k) for k in ("air","top_surface_contact","ground_contact","support")},
            current_leg_contacts={leg:{k:task["physical_evaluator"]["current_legs"][leg].get(k) for k in
                ("air","top_surface_contact","ground_contact","support","bearing_verified")} for leg in ("FL","FR","RL","RR")},
            source_partial_order=diag["source_partial_order"],height_recovery=diag["height_recovery"],
            rr_carry_continuation=diag["rr_carry_continuation"])
    snapshots.append(item)

last_task=decisions[LAST]["semantic_task"]
source_diag=[d["semantic_task"]["nominal_provider_diagnostics"]["source_partial_order"] for d in decisions.values()]
late_starts=sorted({layer["late_group_start_tick"] for d in source_diag for layer in d["layers"] if layer["stage"]=="P09" and "late_group_start_tick" in layer})
rr_cross=last_task["physical_evaluator"]["history"]["event_ticks"]["front_edge_crossed"]["RR"]
measured_geometry={tick:geometry(row) for tick,row in physics.items() if tick>=FIRST}
extrema={key:{"min":min(r[key] for r in measured_geometry.values()),"max":max(r[key] for r in measured_geometry.values())}
    for key in ("body_z_m","body_pitch_deg","RR_gap_m","RR_front_m")}
height=[d["semantic_task"]["nominal_provider_diagnostics"]["height_recovery"] for d in decisions.values()]
report=dict(schema="wlr50_clean.current_P09_late_source_readonly.v1",source=str(SOURCE),ticks=[FIRST,LAST],
    physics_tick_count=LAST-FIRST+1,RR_cross_tick=rr_cross,late_source_reconfiguration_observation_ticks=late_starts,
    nominal_changes=changes,snapshots=snapshots,extrema=extrema,
    all12_masks_one=mask_ok,headroom_clipped_tick_counts=dict(clipped),assist_mode_tick_counts=dict(assist_states),
    geometry_baseline_changed_tick_counts=dict(geometry_corrections),
    RR_geometry_active_tick_span=[geometry_active_ticks[0],geometry_active_ticks[-1]],
    first_RR_geometry_correction_absent_after_present_tick=geometry_active_ticks[-1]+1,
    RR_geometry_exit_source_observation=decisions[9960]["semantic_task"]["physical_evaluator"]["current_legs"]["RR"],
    RR_current_Q_true_decision_endpoints=sum(d["semantic_task"]["physical_evaluator"]["current_legs"]["RR"]["current_lift_valid"] for d in decisions.values()),
    decision_endpoint_count=len(decisions),
    height_recovery=dict(permitted_endpoint_count=sum(x["current_RR_carry_recovery_permitted"] for x in height),
        all_post_lift_offsets_zero=all(all(v==0. for v in x["post_lift_recovery_offsets_deg"].values()) for x in height),
        configured_candidate=height[-1]["candidate_id"],owners=height[-1]["owners"]),
    remaining_task_at_last=dict(phase=last_task["stage_id"],RR_placed=last_task["placed_history"]["RR"],
        RR_current_contact=last_task["physical_evaluator"]["current_legs"]["RR"]["contact_surface"],
        task_termination=last_task["termination_reason"],success=last_task["success"]),
    limits=["Fixed same-run actual window only; no causal counterfactual of removing policy/source, no success claim.",
        "Nominal is actual pre-dispatch source input; mapper baseline, effective residual, headroom projection and physical commanded/actual targets are separated.",
        "Body pitch uses the same base quaternion to RPY function as semantic_metrics, not inferred from leg angles.",
        "Frozen eval: no optimizer/AUX update or new data labelled as PPO learning."])
print(json.dumps(report,indent=2,allow_nan=False))
