"""Sealed P05 chain; no source/mapper replay, model inference or optimization."""
import bisect
import json
from pathlib import Path
from rr_probe_readonly import ROOT, lines, stats, compact_physical

OUT=Path(__file__).resolve().parent
SOURCE=ROOT/"runs/ppo_task_conditioned_hip_wheel_v1/video_eval/validation/20260921T1006004011698Z_g97ecd305afb5_78f9ffd75d504f0e8447f44e61d134ce/source"
OLD=json.loads((OUT/"CP189952_P05_readonly.json").read_text(encoding="utf-8"))
manifest=json.loads((SOURCE/"semantic_video_source_manifest.json").read_text())
assert manifest["episode_physics_ticks"]==6220 and manifest["optimizer_updates"]==0
ds=list(lines(SOURCE/"video_policy_decisions.jsonl")); ends=[d["end_tick"] for d in ds]
ns=[r for r in lines(SOURCE/"native_tick_audit.jsonl") if r["source_phase_id"]=="P05"]
terminal_source=ns[-1]["nominal_full12"][:2]
assert terminal_source==[22.8,-13.4]
endpoint=next(n["episode_physics_tick"] for n in ns if n["nominal_full12"][:2]==terminal_source)
first_source=ns[0]["episode_physics_tick"]
assert endpoint-first_source==OLD["finite_trigger"]["endpoint_source_index"]
entry=next(t for t in ends if t>=endpoint)
raw={}
for r in lines(SOURCE/"physical_observations.jsonl"):
    if r["physics_tick"]<endpoint-1:continue
    c=compact_physical(r)
    raw[r["physics_tick"]]=dict(t=r["simulation_time_s"],gap=c["gap"]["FL"],body_z=c["body_min_z"],separation=c["separation"],
        FL_actual=[r["joints"][n]["position_deg"] for n in ("front_left_hip","front_left_knee")],
        FL_final=[r["joints"][n]["command_deg"] for n in ("front_left_hip","front_left_knee")],
        FL_velocity=[r["joints"][n]["velocity_deg_s"] for n in ("front_left_hip","front_left_knee")],
        FL_ground=r["contacts"]["FL"]["ground"]["active"] if "FL" in r["contacts"] else c["contacts"]["FL"]["ground"]["active"],
        FL_obstacle=c["contacts"]["FL"]["obstacle"]["active"])
assert list(raw)==list(range(endpoint-1,6221))
best=min(range(entry,6221),key=lambda t:raw[t]["gap"])
selected={endpoint-1,endpoint,entry,best,6160,6220}
samples=[]
for n in ns:
    tick=n["episode_physics_tick"]
    if tick not in selected:continue
    d=ds[bisect.bisect_left(ends,tick)]; assert d["start_tick"]<tick<=d["end_tick"]
    a=n["native_audit"];h=a["policy_headroom_evidence"];r=raw[tick];q=d.get("policy_request") or {}
    assert a["setter_dispatch_targets_equal"] and a["actual_mapping_matches_dispatch"]
    samples.append(dict(tick=tick,simulation_time_s=r["t"],native_command_clock_tick=a["physics_tick"],
        issuing_decision=dict(index=d["decision"],start_tick=d["start_tick"],end_tick=d["end_tick"],physics_ticks=d["physics_ticks"]),
        source_N_FL_hip_knee_deg=n["nominal_full12"][:2],mapped_baseline_FL_hip_knee_deg=h["baseline_native_plus_controller_full12"][:2],
        REQUEST_FL_hip_knee_deg=n["projected_residual_full12"][:2],effective_FL_hip_knee_deg=h["effective_policy_residual_full12"][:2],
        final_FL_hip_knee_deg=r["FL_final"],actual_FL_hip_knee_deg=r["FL_actual"],actual_FL_velocity_deg_s=r["FL_velocity"],
        network_base_mean_FL_hip_knee=(q.get("base_mean_full12") or [None]*12)[:2],
        actor_previous_raw_H_FL_hip_knee=(q.get("previous_raw_from_current_observation_full12") or [None]*12)[:2],
        actor_decoded_previous_filtered_REQUEST_FL_hip_knee=(q.get("previous_filtered_request_full12") or [None]*12)[:2],
        conditional_mean_FL_hip_knee=(q.get("conditional_mean_full12") or [None]*12)[:2],
        history_center_FL_hip_knee=(q.get("history_center_full12") or [None]*12)[:2],rho=q.get("rho"),
        selected_raw_FL_hip_knee=(q.get("selected_raw_full12") or [None]*12)[:2],
        original_full_observation_372=None,unlogged_history_groups=None,
        FL_gap_m=r["gap"],FL_ground_contact=r["FL_ground"],FL_obstacle_contact=r["FL_obstacle"],
        body_collider_min_z_m=r["body_z"],AABB_separation_lower_bound_m=r["separation"],
        residual_permission_mask=a["phase_mask_full12"],hip_knee_clipped=any(i in h["clipped_servo_indices"] for i in (0,1)),
        source_N_wheels=n["nominal_full12"][8:],issuing_decision_end_final_canonical_wheel_targets=d["step_info"]["actual_drive_target_full12"][8:]))
post=[n for n in ns if n["episode_physics_tick"]>=entry]
postraw=[raw[t] for t in range(entry,6221)]
end=ds[-1]["step_info"];ev=end["semantic_task"]["physical_evaluator"]
result=dict(schema="CP192512.sealed_P05_same_dispatch_readonly.v1",source=str(SOURCE),
    outcome=dict(tick=6220,duration_s=6220/120,semantic_termination=end["termination_reason"],
        termination_source=end["semantic_task"]["termination_source"],physical_valid=ev["valid"],
        physical_termination=ev["termination_reason"],physical_success=ev["success"],
        FL_events={k:v.get("FL") for k,v in ev["history"]["event_ticks"].items()},
        FL_current={k:ev["current_legs"]["FL"][k] for k in ("air","within_top_xy","clearance_m","bearing_force_n","support","contact_surface")}),
    source_endpoint=dict(first_P05_source_dispatch_tick=first_source,first_actual_final_source_FL_value_tick=endpoint,
        direct_endpoint_flag_logged=None,first_full_decision_endpoint_after_actual_final_source_dispatch=entry,
        inferred_preaction_frame_for_endpoint=endpoint-1,
        semantics="Actual same-dispatch N first reaches held terminal FL22.8/-13.4 at endpoint tick. Source interval matches prior verified1168; preaction frame is inferred, not a measured probe trigger. No Recording search or MotionExecutor/mapper replay.",
        old_comparison_aligned_entry_tick=OLD["entry_to_terminal"]["ticks"][0]),
    same_dispatch_samples=samples,
    aligned_endpoint_to_terminal=dict(ticks=[entry,6220],
        REQUEST_FL_hip_deg=stats(n["projected_residual_full12"][0] for n in post),
        REQUEST_FL_knee_deg=stats(n["projected_residual_full12"][1] for n in post),
        actual_FL_hip_deg=stats(r["FL_actual"][0] for r in postraw),actual_FL_knee_deg=stats(r["FL_actual"][1] for r in postraw),
        FL_gap_m=stats(r["gap"] for r in postraw),minimum_FL_gap_tick=best,
        body_collider_min_z_m=stats(r["body_z"] for r in postraw),
        negative_hip_REQUEST_ticks=sum(n["projected_residual_full12"][0]<0 for n in post),
        hip_knee_headroom_clip_ticks=sum(any(i in n["native_audit"]["policy_headroom_evidence"]["clipped_servo_indices"] for i in (0,1)) for n in post),
        max_REQUEST_effective_abs_difference=max(abs(n["projected_residual_full12"][i]-n["native_audit"]["policy_headroom_evidence"]["effective_policy_residual_full12"][i]) for n in post for i in (0,1)),
        FL_ground_contact_samples=sum(r["FL_ground"] for r in postraw),FL_obstacle_contact_samples=sum(r["FL_obstacle"] for r in postraw)),
    previous_CP189952_reused=dict(source_artifact=str(OUT/"CP189952_P05_readonly.json"),outcome=OLD["outcome"],
        aligned_entry_to_terminal=OLD["entry_to_terminal"],same_dispatch_samples=OLD["same_dispatch_samples"]),
    limitations="Two fixed deterministic checkpoint trajectories, not isolated causality. N->mapped feedback is separate from same-tick PPO REQUEST/effective; final slew may still differ. No original obs372 in this video stream: only logged network/raw and actor-decoded H fields preserved. No new probe, model forward, training or active stochastic log access.")
with (OUT/"CP192512_P05_readonly.json").open("x",encoding="utf-8") as stream:
    json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps({k:v for k,v in result.items() if k!="previous_CP189952_reused"},indent=2))
