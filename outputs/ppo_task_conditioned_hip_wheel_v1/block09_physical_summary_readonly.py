"""Only sealed block09; compact body/contact/wheel/linkage and front-quality audit."""
import json
import math
from collections import Counter
from pathlib import Path
from rr_probe_readonly import lines, stats, servo_limits_deg
from training_p06_readonly import variation

OUT=Path(__file__).resolve().parent
receipt=json.loads((OUT/"block09_P01_quantity_512_receipt.json").read_text())
RUN=Path(receipt["run"])
rows=list(lines(RUN/"residual_and_projection_audit.jsonl"))
assert len(rows)==512 and not any(r["terminal"] for r in rows)
samples=[r["applied_audit"] for r in rows]
last=samples[-1]; ev=last["semantic_task"]["physical_evaluator"]
assert last["physics_tick"]==4096 and last["end_phase_id"]=="P06"
hist=last["semantic_task"]["history"]
transitions=[t for s in samples for t in s["stage_transition_evidence"]]
start=next(t["physics_tick"] for t in transitions if t["to_stage"]=="P06")
window=[s for s in samples if start<=s["physics_tick"]<=4096]
assert [s["physics_tick"] for s in window]==list(range(start,4097,8))
physical=[s["semantic_task"]["physical_evaluator"] for s in window]
FL=[p["current_legs"]["FL"] for p in physical]
air_first=next((s["physics_tick"] for s,p in zip(window,FL) if p["air"]),None)
top=lambda a:a["top_surface_contact"] and a["bearing_verified"] and a["support"]
geometry=[g for s in window[1:] for g in s["reward_breakdown"]["task_space_quality_sample_audit"]]
assert len(geometry)==4096-start and all(g["eligible"] and g["valid"] for g in geometry)
positions={}
for p in physical:
    for role in p["transfer_roles"].values():
        for name,values in role["receiver_workspace_state"]["joint_range_margin_deg"].items():
            lower,upper=servo_limits_deg(name)
            actual=lower+values["negative_deg"]
            assert abs(upper-actual-values["positive_deg"])<1e-8
            positions.setdefault(name,[]).append(actual)
assert all(len(a)==len(window) for a in positions.values())
wheel_order=["FL","FR","RL","RR"]
wheels={}
for i,leg in enumerate(wheel_order):
    wheels[leg]=dict(nominal_rad_s=stats(s["nominal_action_full12"][8+i] for s in window[1:]),
        final_canonical_target_rad_s=stats(s["actual_drive_target_full12"][8+i] for s in window[1:]),
        measured_canonical_velocity_rad_s=stats(p["measured_wheel_velocity_rad_s"][i] for p in physical[1:]),
        raw_native_measured_qd=None,
        near_zero_measured_count_0_01_rad_s=sum(abs(p["measured_wheel_velocity_rad_s"][i])<.01 for p in physical[1:]))
front=[a for s in samples for a in s["reward_breakdown"]["front_quality_sample_audit"]]
dt=sum(a["dt_s"] for a in front)
assert dt>0 and all(a["phase"] in ("P01","P02") for a in front)
front_ticks={round(a["sim_time_s"]*120) for a in front}
front_geometry=[g for s in samples for g in s["reward_breakdown"]["task_space_quality_sample_audit"] if g["valid"] and round(g["sim_time_s"]*120) in front_ticks]
assert len(front_geometry)==len(front)
selected={start-8,start,start+8,((start+4096)//16)*8,4096}
if air_first is not None: selected.update((air_first-8,air_first))
timeline=[]
for s in samples:
    if s["physics_tick"] not in selected: continue
    p=s["semantic_task"]["physical_evaluator"]; fl=p["current_legs"]["FL"]
    timeline.append(dict(tick=s["physics_tick"],phase=s["phase_id"],end_phase=s["end_phase_id"],
        body_forward_m=p["goal_features"]["body_forward_m"],FL_air=fl["air"],FL_TOP_bearing=top(fl),
        FL_gap_m=fl["clearance_m"],nominal_full12=s["nominal_action_full12"],
        final_wheel_targets=s["actual_drive_target_full12"][8:],measured_canonical_wheel_velocity=p["measured_wheel_velocity_rad_s"],
        FL_placed_history=s["semantic_task"]["history"]["placed"]["FL"]))
result=dict(schema="block09.physical_summary_readonly.v1",source=str(RUN),
    classification="512 decisions of updating stochastic training; not fixed checkpoint evaluation",
    phase_coverage=receipt["actual_request_phase_coverage"],event_ticks=hist["event_ticks"],
    endpoint=dict(tick=4096,sim_time_s=4096/120,phase=last["end_phase_id"],terminal=False,
        physical_valid=ev["valid"],physical_termination=ev["termination_reason"],success=ev["success"],
        time_outs=last["time_outs"],bootstrap=last["terminal_bootstrap_allowed"],
        RR_current_valid=ev["current_legs"]["RR"]["current_lift_valid"],FL=ev["current_legs"]["FL"]),
    P06=dict(ticks=[start,4096],duration_s=(4096-start)/120,evaluator_endpoints=len(window),
        body_net_advance_m=physical[-1]["goal_features"]["body_forward_m"]-physical[0]["goal_features"]["body_forward_m"],
        FL_TOP_verified_bearing_count=sum(top(a) for a in FL),FL_air_count=sum(a["air"] for a in FL),
        FL_first_saved_AIR_tick=air_first,FL_gap_m=stats(a["clearance_m"] for a in FL),
        FL_contact_mode_changes=[dict(tick=window[i]["physics_tick"],air=FL[i]["air"],TOP_bearing=top(FL[i])) for i in range(1,len(FL)) if (FL[i]["air"],top(FL[i]))!=(FL[i-1]["air"],top(FL[i-1]))],
        other_current_support_counts={leg:sum(p["current_legs"][leg]["support"] for p in physical) for leg in ("FR","RR","RL")},
        body_collider_min_z_m=stats(g["body_collider_minimum_w_m"][2] for g in geometry),
        AABB_separation_lower_bound_m=stats(g["separation_lower_bound_m"] for g in geometry),
        wheel_channels=wheels, measured_joint_position_variation_deg={name:variation(a) for name,a in positions.items()},
        all_native_dispatch_verified=all(t["verified"] for s in window[1:] for t in s["actuator_target_effect_audit_ticks"]),
        nominal_servo_unique_vectors=[list(v) for v in sorted(set(tuple(s["nominal_action_full12"][:8]) for s in window[1:]))]),
    front_quality_window=dict(ticks=[min(front_ticks),max(front_ticks)],sample_count=len(front),integrated_duration_s=dt,
        roll_RMS_rad=math.sqrt(sum(a["roll_pitch_rad"][0]**2*a["dt_s"] for a in front)/dt),
        pitch_RMS_rad=math.sqrt(sum(a["roll_pitch_rad"][1]**2*a["dt_s"] for a in front)/dt),
        roll_pitch_rate_RMS_rad_s=math.sqrt(sum(sum(v*v for v in a["roll_pitch_rate_rad_s"])/2*a["dt_s"] for a in front)/dt),
        peak_tilt_rad=max(math.hypot(*a["roll_pitch_rad"]) for a in front),
        body_collider_min_z_m=min(g["body_collider_minimum_w_m"][2] for g in front_geometry),
        actual_front_cost=sum(a["weighted_quality_cost"] for a in front),substate_samples=dict(Counter(a["substate"] for a in front)),
        scope="actual P01/P02 weighted samples, not a full physical capture interval including P03"),
    actual_quality_contributions=receipt["actual_quality_contributions"],timeline=timeline,
    limitations="FL fractions are ~15Hz saved endpoints, not continuous support durations. Measured canonical wheel speed is not raw native qd or traction/work proof. Joint actual recovered from dual validated physical margins, not targets. First AIR is first saved endpoint, not exact loss tick. No hip mount invented. Budget tail is not terminal or success; no RR curriculum coverage.")
assert abs(result["front_quality_window"]["actual_front_cost"]-receipt["actual_quality_contributions"]["front_cost"])<1e-10
with (OUT/"block09_physical_summary.json").open("x",encoding="utf-8") as stream:
    json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps({k:v for k,v in result.items() if k not in ("P06","endpoint")},indent=2))
print("P06",json.dumps({k:v for k,v in result["P06"].items() if k not in ("wheel_channels","measured_joint_position_variation_deg")},indent=2))
print("WHEEL",json.dumps(wheels))
print("JOINT",json.dumps(result["P06"]["measured_joint_position_variation_deg"]))
print("END_FL",json.dumps({k:result["endpoint"]["FL"][k] for k in ("air","support","clearance_m","bearing_force_n")}))
