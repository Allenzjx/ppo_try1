"""Bounded sealed finite FL diagnostic vs its fixed-checkpoint natural baseline."""
import json
from pathlib import Path
from rr_probe_readonly import ROOT, lines, stats, compact_physical

OUT = Path(__file__).resolve().parent
PROBE = ROOT / "runs/ppo_task_conditioned_hip_wheel_v1/diagnostics/FL_minus3_CP189952_20260921_01"
BASE = ROOT / "runs/ppo_task_conditioned_hip_wheel_v1/video_eval/validation/20260921T0801290800396Z_gee5a9651591d_a6d92def5d74491580cccd936a1edf18/source"
m = json.loads((PROBE / "run_manifest.json").read_text())
assert m["lifecycle"] == "DIAGNOSTIC_SEALED" and m["endpoint_tick"] == 4800
entry = json.loads((PROBE / "probe_entry.json").read_text())
pres, posts = {}, {}
for r in lines(PROBE / "probe_decisions.jsonl"):
    (pres if r["record_kind"] == "pre_action" else posts)[r["decision"]] = r
assert len(pres) == len(posts) == 600
assert all(pres[i]["pre_action_receipt_sha256"] == posts[i]["pre_action_receipt_sha256"] for i in pres)
assert all(p["combined_residual_permission_mask_full12"] == [1]*12 for p in pres.values())
assert max(abs(v) for p in pres.values() for v in p["manual_raw_delta_full12"][1:]) == 0.
bm = json.loads((BASE / "semantic_video_source_manifest.json").read_text())
assert bm["runtime_contract"] == m["runtime_contract"]
assert bm["checkpoint_load_provenance"]["parameter_hashes"] == m["checkpoint_load_provenance"]["parameter_hashes"]
start = entry["pre_action_receipt"]["start_tick"]
active = [p for p in pres.values() if (p.get("intervention") or {}).get("index") is not None]
assert [p["intervention"]["index"] for p in active] == list(range(117))
release = next(p["start_tick"] for p in active if p["intervention"]["release_age"] is not None)
ranges = dict(pre=(start-24,start), ramp=(start+1,start+96), hold=(start+97,release),
              release=(release+1,release+96), follow=(release+97,release+336),
              after_probe=(release+337,m["endpoint_tick"]))

def raw_rows(path):
    result = {}
    for r in lines(path / "physical_observations.jsonl"):
        if r["physics_tick"] > 4800:
            break
        c = compact_physical(r)
        c["FL_joint_actual_deg"] = [r["joints"][n]["position_deg"] for n in ("front_left_hip","front_left_knee")]
        c["FL_joint_target_deg"] = [r["joints"][n]["command_deg"] for n in ("front_left_hip","front_left_knee")]
        result[c["tick"]] = c
    assert list(result) == list(range(4801))
    return result

raw = {"probe":raw_rows(PROBE), "baseline":raw_rows(BASE)}
heights = {label:{r["physics_tick"]:r for r in lines(path / "height_diagnostics.jsonl") if r["physics_tick"] <= 4800}
           for label,path in (("probe",PROBE),("baseline",BASE))}
hips = {r["end_tick"]:r["four_hip_geometry"] for r in posts.values()}
assert all(h["clock_unchanged"] and h["frame_clock_aligned"] and h["physics_tick"] == t for t,h in hips.items())
first = {}
for field in ("full12", "base", "FL_joint_actual_deg", "FL_joint_target_deg"):
    first[field] = next((t for t in range(4801) if raw["probe"][t][field] != raw["baseline"][t][field]),None)

def window(label, lo, hi):
    rs = [raw[label][t] for t in range(lo,hi+1)]
    hs = [h for t,h in heights[label].items() if lo <= t <= hi]
    gs = [h for t,h in hips.items() if lo <= t <= hi] if label == "probe" else []
    return dict(ticks=[lo,hi], physical_samples=len(rs), FL_gap_m=stats(r["gap"]["FL"] for r in rs),
        FL_obstacle_contact_samples=sum(r["contacts"]["FL"]["obstacle"]["active"] for r in rs),
        FL_ground_contact_samples=sum(r["contacts"]["FL"]["ground"]["active"] for r in rs),
        FL_hip_actual_deg=stats(r["FL_joint_actual_deg"][0] for r in rs),
        FL_knee_actual_deg=stats(r["FL_joint_actual_deg"][1] for r in rs),
        body_collider_minimum_world_z_m=stats(r["body_min_z"] for r in rs),
        conservative_body_obstacle_AABB_separation_m=stats(r["separation"] for r in rs),
        RR_mount_world_z_m_sampled=stats(h["rr_hip_mount_w_m"]["value"][2] for h in hs if h["rr_hip_mount_w_m"]["value"] is not None),
        other_hip_mount_world_z_m_sampled=({leg:stats(h["hip_mount_world_m"][leg]["value"][2] for h in gs)
            for leg in ("FL","FR","RL")} if gs else None))

best = min((r for t,r in raw["probe"].items() if t >= start),key=lambda r:r["gap"]["FL"])
selected = {start,start+1,start+96,release,release+96,release+336,4800,best["tick"]}
examples = {"probe":[],"baseline":[]}
checks = []
for label,path in (("probe",PROBE),("baseline",BASE)):
    for r in lines(path / "native_tick_audit.jsonl"):
        tick = r["episode_physics_tick"]
        if tick > 4800:
            break
        a = r["native_audit"]; h = a["policy_headroom_evidence"]
        if label == "probe" and start < tick <= release:
            checks.append((r["projected_residual_full12"][0],h["effective_policy_residual_full12"][0],
                0 in h["clipped_servo_indices"],a["setter_dispatch_targets_equal"],a["actual_mapping_matches_dispatch"]))
        if tick not in selected:
            continue
        c = raw[label][tick]
        geometry = hips.get(tick) if label == "probe" else None
        examples[label].append(dict(tick=tick, simulation_time_s=c["t"],
            source_N_FL_hip_knee_deg=r["nominal_full12"][:2],
            mapped_baseline_FL_hip_knee_deg=h["baseline_native_plus_controller_full12"][:2],
            REQUEST_FL_hip_knee_deg=r["projected_residual_full12"][:2],
            effective_FL_hip_knee_deg=h["effective_policy_residual_full12"][:2],
            final_FL_hip_knee_deg=c["FL_joint_target_deg"],actual_FL_hip_knee_deg=c["FL_joint_actual_deg"],
            FL_gap_m=c["gap"]["FL"],FL_obstacle_contact=c["contacts"]["FL"]["obstacle"]["active"],
            FL_ground_contact=c["contacts"]["FL"]["ground"]["active"],
            body_collider_minimum_world_z_m=c["body_min_z"],
            four_hip_world_z_m=None if geometry is None else {leg:v["value"][2] for leg,v in geometry["hip_mount_world_m"].items()}))
hold = [p for p in active if 11 <= p["intervention"]["index"] <= 74]
anchor = entry["pre_action_receipt"]["intervention"]["anchor_full12"][0]
assert all(p["intervention"]["requested_selected_deg"]["0"] == anchor-3 for p in hold)
assert all(not clipped and dispatch and mapping for _,_,clipped,dispatch,mapping in checks)
assert max(abs(x-y) for x,y,_,_,_ in checks) == 0
ev = m["physical_summary"]["physical_task_evaluation"]
result = dict(schema="CP189952.FL_minus3.sealed_readonly.v1", source=str(PROBE), baseline=str(BASE),
    sealed={k:m[k] for k in ("lifecycle","endpoint_tick","final_phase","original_task_reason","probe_complete","external_budget_stop","learned_state_unchanged","actual_diagnostic_decisions_completed","new_PPO_decisions","new_PPO_updates")},
    outcome=dict(physical_success=ev["success"],physical_termination=ev["termination_reason"],FL_placed=ev["history"]["placed"]["FL"],first_unfinished_task="FL_capture"),
    direct_entry=dict(tick=start, simulation_time_s=entry["pre_action_receipt"]["start_sim_time_s"],
        precomputed_source_N_FL_hip_knee_deg=entry["pre_action_receipt"]["pre_source_nominal_full12"][:2],
        anchor_REQUEST_FL_hip_deg=anchor,hold_REQUEST_FL_hip_deg=anchor-3,release_start_tick=release,
        release_end_tick=release+96,follow_end_tick=release+336, previous_inferred_entry_tick=2984,
        correction="Direct probe_entry supersedes old post-step-clock inference. At frame2976 provider has already computed endpoint command for application2977; previous native command at2976 was still pre-endpoint."),
    first_difference_selected_raw_fields=first, same_pre_post_receipt_pairs=600,
    checks=dict(same_runtime_and_checkpoint_parameter_hashes=True, other11_manual_raw_delta_max=0.,
        all_permission_masks_full12=True, ramp_hold_all_native_verified=True, ramp_hold_hip_headroom_clip_ticks=0,
        ramp_hold_max_REQUEST_minus_effective_abs_deg=0., four_hip_geometry_clock_aligned=True),
    windows={name:{label:window(label,lo,hi) for label in raw} for name,(lo,hi) in ranges.items()},
    minimum_FL_gap=dict(tick=best["tick"],gap_m=best["gap"]["FL"],same_tick_baseline_gap_m=raw["baseline"][best["tick"]]["gap"]["FL"]),
    same_dispatch_examples=examples,
    limitations="Single finite external diagnostic, not PPO/teacher training. Only FL raw channel manually overridden; future all-channel closed-loop policy and mapper histories respond. Do not attribute the entire gap difference to isolated hip kinematics. Baseline non-RR hip mounts absent/null; no inferred hip height or traction.")
with (OUT / "CP189952_FL_minus3_evidence.json").open("x",encoding="utf-8") as stream:
    json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps({k:v for k,v in result.items() if k not in ("same_dispatch_examples","windows")},ensure_ascii=False,indent=2))
for name,labels in result["windows"].items():
    print(name, {label:{k:v for k,v in data.items() if k in ("FL_gap_m","FL_hip_actual_deg","body_collider_minimum_world_z_m","RR_mount_world_z_m_sampled","FL_obstacle_contact_samples")} for label,data in labels.items()})
print("EXAMPLES",json.dumps(examples,ensure_ascii=False))
