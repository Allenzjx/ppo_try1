"""Sealed P05 same-dispatch chain, prior extracted JSON only for old comparison."""
import bisect
import hashlib
import json
from pathlib import Path
from rr_probe_readonly import ROOT, lines, stats, compact_physical

OUT = Path(__file__).resolve().parent
SOURCE = ROOT / "runs/ppo_task_conditioned_hip_wheel_v1/video_eval/validation/20260921T1117298582122Z_g97ecd305afb5_26ae0caab77f4c1bab164b45635744d4/source"
OLD_PATH = OUT / "CP192512_P05_readonly.json"
OLD = json.loads(OLD_PATH.read_text(encoding="utf-8"))
manifest = json.loads((SOURCE / "semantic_video_source_manifest.json").read_text())
END = 6276
assert manifest["episode_physics_ticks"] == END and manifest["optimizer_updates"] == 0
ds = []
for row in lines(SOURCE / "video_policy_decisions.jsonl"):
    ds.append({k: row[k] for k in ("decision", "start_tick", "end_tick", "physics_ticks", "policy_request")})
    last_step = row["step_info"]
ends = [d["end_tick"] for d in ds]
assert ends[-1] == END
ns = []
for n in lines(SOURCE / "native_tick_audit.jsonl"):
    if n["source_phase_id"] != "P05": continue
    a = n["native_audit"]; h = a["policy_headroom_evidence"]
    ns.append(dict(tick=n["episode_physics_tick"], native_tick=a["physics_tick"], N=n["nominal_full12"],
        mapped=h["baseline_native_plus_controller_full12"], request=n["projected_residual_full12"],
        effective=h["effective_policy_residual_full12"], mask=a["phase_mask_full12"],
        clipped=h["clipped_servo_indices"], verified=a["setter_dispatch_targets_equal"] and a["actual_mapping_matches_dispatch"]))
assert ns[-1]["N"][:2] == [22.8, -13.4]
endpoint = next(n["tick"] for n in ns if n["N"][:2] == [22.8, -13.4])
first_source = ns[0]["tick"]
assert endpoint - first_source == OLD["source_endpoint"]["first_actual_final_source_FL_value_tick"] - OLD["source_endpoint"]["first_P05_source_dispatch_tick"]
entry = next(t for t in ends if t >= endpoint)
joint_names = ("front_left_hip", "front_left_knee", "front_right_hip", "front_right_knee", "rear_left_hip", "rear_left_knee", "rear_right_hip", "rear_right_knee")
raw = {}
for row in lines(SOURCE / "physical_observations.jsonl"):
    tick = row["physics_tick"]
    if tick < endpoint - 1: continue
    assert tick <= END
    c = compact_physical(row)
    raw[tick] = dict(t=row["simulation_time_s"], gap=c["gap"]["FL"], body_z=c["body_min_z"],
        separation=c["separation"], rpy=c["rpy"], base=c["base"], all_leg_gap_m=c["gap"],
        contacts={leg:{surface: c["contacts"][leg][surface]["active"] for surface in ("ground","obstacle")} for leg in ("FL","FR","RL","RR")},
        actual=[row["joints"][name]["position_deg"] for name in joint_names],
        final=[row["joints"][name]["command_deg"] for name in joint_names],
        velocity=[row["joints"][name]["velocity_deg_s"] for name in joint_names])
assert list(raw) == list(range(endpoint - 1, END + 1))
best = min(range(entry, END + 1), key=lambda t: raw[t]["gap"])
selected = {endpoint - 1, endpoint, entry, best, END - 60, END}
samples = []
for n in ns:
    tick = n["tick"]
    if tick not in selected: continue
    d = ds[bisect.bisect_left(ends, tick)]; assert d["start_tick"] < tick <= d["end_tick"]
    r = raw[tick]; q = d.get("policy_request") or {}; assert n["verified"]
    samples.append(dict(tick=tick, simulation_time_s=r["t"], native_command_clock_tick=n["native_tick"],
        issuing_decision={k:d[k] for k in ("decision","start_tick","end_tick","physics_ticks")},
        source_N_FL_hip_knee_deg=n["N"][:2], mapped_baseline_FL_hip_knee_deg=n["mapped"][:2],
        REQUEST_FL_hip_knee_deg=n["request"][:2], effective_FL_hip_knee_deg=n["effective"][:2],
        final_FL_hip_knee_deg=r["final"][:2], actual_FL_hip_knee_deg=r["actual"][:2], actual_FL_velocity_deg_s=r["velocity"][:2],
        network_base_mean_FL_hip_knee=(q.get("base_mean_full12") or [None]*12)[:2],
        actor_previous_raw_H_FL_hip_knee=(q.get("previous_raw_from_current_observation_full12") or [None]*12)[:2],
        actor_decoded_previous_filtered_REQUEST_FL_hip_knee=(q.get("previous_filtered_request_full12") or [None]*12)[:2],
        conditional_mean_FL_hip_knee=(q.get("conditional_mean_full12") or [None]*12)[:2],
        history_center_FL_hip_knee=(q.get("history_center_full12") or [None]*12)[:2], rho=q.get("rho"),
        selected_raw_FL_hip_knee=(q.get("selected_raw_full12") or [None]*12)[:2],
        original_full_observation_372=None, unlogged_history_groups=None,
        FL_gap_m=r["gap"], FL_ground_contact=r["contacts"]["FL"]["ground"], FL_obstacle_contact=r["contacts"]["FL"]["obstacle"],
        body_collider_min_z_m=r["body_z"], AABB_separation_lower_bound_m=r["separation"],
        residual_permission_mask=n["mask"], hip_knee_clipped=any(i in n["clipped"] for i in (0,1)),
        all_joint_order=joint_names, all_joint_N_deg=n["N"][:8], all_joint_mapped_deg=n["mapped"][:8],
        all_joint_REQUEST_deg=n["request"][:8], all_joint_effective_deg=n["effective"][:8],
        all_joint_final_deg=r["final"], all_joint_actual_deg=r["actual"],
        body_rpy_rad=r["rpy"], base=r["base"], all_leg_gap_m=r["all_leg_gap_m"], contacts=r["contacts"]))
post = [n for n in ns if n["tick"] >= entry]; postraw = [raw[t] for t in range(entry, END+1)]
ev = last_step["semantic_task"]["physical_evaluator"]
result = dict(schema="CP194560.sealed_P05_same_dispatch_readonly.v1", source=str(SOURCE), training_label="PPO+LIMITEDAUX",
    outcome=dict(tick=END, duration_s=END/120, semantic_termination=last_step["termination_reason"],
        termination_source=last_step["semantic_task"]["termination_source"], physical_valid=ev["valid"],
        physical_termination=ev["termination_reason"], physical_success=ev["success"],
        FL_events={k:v.get("FL") for k,v in ev["history"]["event_ticks"].items()},
        FL_current={k:ev["current_legs"]["FL"][k] for k in ("air","within_top_xy","clearance_m","bearing_force_n","support","contact_surface")},
        all_current_legs=ev["current_legs"]),
    source_endpoint=dict(first_P05_source_dispatch_tick=first_source, first_actual_final_source_FL_value_tick=endpoint,
        direct_endpoint_flag_logged=None, first_full_decision_endpoint_after_actual_final_source_dispatch=entry,
        inferred_preaction_frame_for_endpoint=endpoint-1, source_interval_ticks=endpoint-first_source,
        semantics="actual N reaches terminal FL22.8/-13.4; inferred preaction tick is not a direct probe trigger"),
    same_dispatch_samples=samples,
    aligned_endpoint_to_terminal=dict(ticks=[entry,END],
        REQUEST_FL_hip_deg=stats(n["request"][0] for n in post), REQUEST_FL_knee_deg=stats(n["request"][1] for n in post),
        actual_FL_hip_deg=stats(r["actual"][0] for r in postraw),actual_FL_knee_deg=stats(r["actual"][1] for r in postraw),
        FL_gap_m=stats(r["gap"] for r in postraw), minimum_FL_gap_tick=best,
        body_collider_min_z_m=stats(r["body_z"] for r in postraw),
        negative_hip_REQUEST_ticks=sum(n["request"][0]<0 for n in post),
        hip_knee_headroom_clip_ticks=sum(any(i in n["clipped"] for i in (0,1)) for n in post),
        max_REQUEST_effective_abs_difference=max(abs(n["request"][i]-n["effective"][i]) for n in post for i in (0,1)),
        FL_ground_contact_samples=sum(r["contacts"]["FL"]["ground"] for r in postraw),
        FL_obstacle_contact_samples=sum(r["contacts"]["FL"]["obstacle"] for r in postraw),
        all_joint_actual_deg={name:stats(r["actual"][i] for r in postraw) for i,name in enumerate(joint_names)}),
    previous_CP192512_reused=dict(source_artifact=str(OLD_PATH), sha256=hashlib.sha256(OLD_PATH.read_bytes()).hexdigest(),
        outcome=OLD["outcome"], source_endpoint=OLD["source_endpoint"], aligned_endpoint_to_terminal=OLD["aligned_endpoint_to_terminal"],
        same_dispatch_samples=OLD["same_dispatch_samples"], other_hip_P05_actual_comparison=None,
        absence_reason="Prior extracted JSON did not include other P05 hips; old raw deliberately not rescanned"),
    limitations="Two fixed deterministic complete trajectories, no isolated joint/aux causality. Real N/mapped/REQUEST/effective/final/actual remain distinct. No full original372obs or new inference. No physical run, production edit or broader audit.")
with (OUT / "CP194560_P05_readonly.json").open("x", encoding="utf-8") as stream:
    json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
print(json.dumps({k:result[k] for k in ("source_endpoint", "aligned_endpoint_to_terminal")}, indent=2))
print("FINAL", json.dumps(samples[-1], indent=2))
print("OUTCOME", json.dumps({k:v for k,v in result["outcome"].items() if k!="all_current_legs"}))
print("OLD_FINAL", json.dumps(OLD["same_dispatch_samples"][-1]))
