"""Sealed finite -6 diagnostic: saved physical data only; no policy or Isaac."""
import json
from pathlib import Path
from rr_probe_readonly import ROOT, lines, stats, compact_physical

OUT = Path(__file__).resolve().parent
PROBE = ROOT / "runs/ppo_task_conditioned_hip_wheel_v1/diagnostics/FL_minus6_CP189952_20260921_01"
OLD = ROOT / "runs/ppo_task_conditioned_hip_wheel_v1/diagnostics/FL_minus3_CP189952_20260921_01"
BASE = ROOT / "runs/ppo_task_conditioned_hip_wheel_v1/video_eval/validation/20260921T0801290800396Z_gee5a9651591d_a6d92def5d74491580cccd936a1edf18/source"
m = json.loads((PROBE / "run_manifest.json").read_text())
assert m["lifecycle"] == "DIAGNOSTIC_SEALED" and m["endpoint_tick"] == 4800
entry = json.loads((PROBE / "probe_entry.json").read_text())["pre_action_receipt"]
old = json.loads((OUT / "CP189952_FL_minus3_evidence.json").read_text())
start = entry["start_tick"]
anchor = entry["intervention"]["anchor_full12"][0]
assert start == old["direct_entry"]["tick"] == 2976
assert anchor == old["direct_entry"]["anchor_REQUEST_FL_hip_deg"]
pres, posts = {}, {}
for r in lines(PROBE / "probe_decisions.jsonl"):
    (pres if r["record_kind"] == "pre_action" else posts)[r["decision"]] = r
assert len(pres) == len(posts) == 600
assert all(pres[i]["pre_action_receipt_sha256"] == posts[i]["pre_action_receipt_sha256"] for i in pres)
assert all(p["combined_residual_permission_mask_full12"] == [1]*12 for p in pres.values())
assert all(all(v == 0 for v in p["manual_raw_delta_full12"][1:]) for p in pres.values())
active = [p for p in pres.values() if (p.get("intervention") or {}).get("index") is not None]
assert [p["intervention"]["index"] for p in active] == list(range(117))
release = next(p["start_tick"] for p in active if p["intervention"]["release_age"] is not None)
assert release == start+75*8
assert all(p["intervention"]["requested_selected_deg"]["0"] == anchor-6 for p in active if 11 <= p["intervention"]["index"] <= 74)
ranges = dict(ramp=(start+1,start+96), hold=(start+97,release),
    release=(release+1,release+96), follow=(release+97,release+336), after_probe=(release+337,4800))
for path, manifest in ((OLD,"run_manifest.json"),(BASE,"semantic_video_source_manifest.json")):
    other = json.loads((path/manifest).read_text())
    assert other["runtime_contract"] == m["runtime_contract"]
    assert other["checkpoint_load_provenance"]["parameter_hashes"] == m["checkpoint_load_provenance"]["parameter_hashes"]


def raw_rows(path):
    result = {}
    for r in lines(path/"physical_observations.jsonl"):
        if r["physics_tick"] > 4800:
            break
        c = compact_physical(r)
        c["FL_actual"] = [r["joints"][n]["position_deg"] for n in ("front_left_hip","front_left_knee")]
        c["FL_target"] = [r["joints"][n]["command_deg"] for n in ("front_left_hip","front_left_knee")]
        assert all(a["pair_verified"] for leg in c["contacts"].values() for a in leg.values() if isinstance(a,dict) and "pair_verified" in a)
        result[c["tick"]] = c
    assert list(result) == list(range(4801))
    return result


raw = {label:raw_rows(path) for label,path in (("minus6",PROBE),("minus3",OLD),("formal_det",BASE))}
hips = {r["end_tick"]:r["four_hip_geometry"] for r in posts.values()}
assert all(h["clock_unchanged"] and h["frame_clock_aligned"] and h["physics_tick"] == t for t,h in hips.items())


def window(label,lo,hi):
    rs = [raw[label][t] for t in range(lo,hi+1)]
    hs = [h for t,h in hips.items() if lo <= t <= hi] if label == "minus6" else []
    return dict(ticks=[lo,hi],n=len(rs),FL_gap_m=stats(r["gap"]["FL"] for r in rs),
        FL_hip_actual_deg=stats(r["FL_actual"][0] for r in rs),
        FL_knee_actual_deg=stats(r["FL_actual"][1] for r in rs),
        body_collider_min_z_m=stats(r["body_min_z"] for r in rs),
        AABB_separation_lower_bound_m=stats(r["separation"] for r in rs),
        actual_contact_samples={leg:{kind:sum(r["contacts"][leg][kind]["active"] for r in rs) for kind in ("ground","obstacle")} for leg in ("FL","FR","RR","RL")},
        hip_mount_z_m_sampled={leg:stats(h["hip_mount_world_m"][leg]["value"][2] for h in hs) for leg in ("FL","FR","RR","RL")} if hs else None)


best = {label:min((r for t,r in data.items() if t>=start),key=lambda r:r["gap"]["FL"]) for label,data in raw.items()}
selected = {start,start+1,start+96,release,release+96,release+336,4800,best["minus6"]["tick"]}
examples = {label:[] for label in raw}
checks = []
for label,path in (("minus6",PROBE),("minus3",OLD),("formal_det",BASE)):
    for r in lines(path/"native_tick_audit.jsonl"):
        t = r["episode_physics_tick"]
        if t > 4800:
            break
        a=r["native_audit"]; h=a["policy_headroom_evidence"]
        if label=="minus6" and start<t<=release:
            checks.append((r["projected_residual_full12"][0],h["effective_policy_residual_full12"][0],0 in h["clipped_servo_indices"],a["setter_dispatch_targets_equal"],a["actual_mapping_matches_dispatch"]))
        if t in selected:
            c=raw[label][t]
            examples[label].append(dict(tick=t,N_FL=r["nominal_full12"][:2],
                mapped_N_FL=h["baseline_native_plus_controller_full12"][:2],
                REQUEST_FL=r["projected_residual_full12"][:2],effective_FL=h["effective_policy_residual_full12"][:2],
                final_FL=c["FL_target"],actual_FL=c["FL_actual"],FL_gap_m=c["gap"]["FL"],
                body_collider_min_z_m=c["body_min_z"]))
assert len(checks)==600 and all(not clip and dispatch and mapping and req==eff for req,eff,clip,dispatch,mapping in checks)
ev=m["physical_summary"]["physical_task_evaluation"]
result=dict(schema="CP189952.FL_minus6.sealed_readonly.v1",source=str(PROBE),
    sealed={k:m[k] for k in ("lifecycle","endpoint_tick","final_phase","original_task_reason","probe_complete","external_budget_stop","learned_state_unchanged","actual_diagnostic_decisions_completed","new_PPO_decisions","new_PPO_updates")},
    outcome=dict(success=ev["success"],termination=ev["termination_reason"],FL_placed=ev["history"]["placed"]["FL"],FL_current=ev["current_legs"]["FL"]),
    direct_entry=dict(tick=start,anchor_REQUEST_deg=anchor,hold_REQUEST_deg=anchor-6,release_start_tick=release,release_end_tick=release+96,follow_end_tick=release+336),
    checks=dict(pre_post_receipt_pairs=600,all_masks_one=True,other11_manual_delta_max=0,
        ramp_hold_hip_clipped_ticks=0,ramp_hold_REQUEST_effective_max_difference=0,
        four_hip_clock_aligned=True,same_CP_and_runtime=True),
    first_physical_difference_vs_formal_det={k:next((t for t in range(4801) if raw["minus6"][t][k]!=raw["formal_det"][t][k]),None) for k in ("full12","base","FL_actual","FL_target")},
    minimum_gap={label:dict(tick=r["tick"],gap_m=r["gap"]["FL"],same_tick_all_gaps_m={other:data[r["tick"]]["gap"]["FL"] for other,data in raw.items()}) for label,r in best.items()},
    windows={name:{label:window(label,lo,hi) for label in raw} for name,(lo,hi) in ranges.items()},
    same_dispatch_examples=examples,
    diagnostic_credit=0, limitations="Only one external finite FL hip override. Other 11 manual deltas zero but all live policy outputs/mapper/physics can respond. No successful contact label or causal isolation from correlated whole-body feedback. Missing baseline non-RR hip mounts remain null. No PPO/auxiliary learning credit.")
with (OUT/"CP189952_FL_minus6_evidence.json").open("x",encoding="utf-8") as stream:
    json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps({k:v for k,v in result.items() if k not in ("windows","same_dispatch_examples","outcome")},ensure_ascii=False,indent=2))
for name,labels in result["windows"].items():
    print(name,json.dumps({label:{k:v for k,v in data.items() if k in ("FL_gap_m","FL_hip_actual_deg","body_collider_min_z_m","actual_contact_samples")} for label,data in labels.items()}))
