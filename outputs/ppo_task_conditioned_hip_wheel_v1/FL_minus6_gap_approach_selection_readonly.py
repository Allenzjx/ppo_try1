"""Index real sealed pre/post samples, never fabricate observations or train."""
import hashlib
import json
import math
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUN = ROOT / "runs/ppo_task_conditioned_hip_wheel_v1/diagnostics/FL_minus6_CP189952_20260921_01"
BASE = ROOT / "runs/ppo_task_conditioned_hip_wheel_v1/video_eval/validation/20260921T0801290800396Z_gee5a9651591d_a6d92def5d74491580cccd936a1edf18/source"


def records(path):
    with path.open("rb") as stream:
        line = 0
        while True:
            offset = stream.tell(); data = stream.readline()
            if not data: break
            assert data.endswith(b"\n")
            line += 1
            yield json.loads(data), dict(line_1based=line,byte_offset=offset,byte_length=len(data))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def f32(value):
    return struct.unpack("<f",struct.pack("<f",value))[0]


manifest = json.loads((RUN/"run_manifest.json").read_text())
assert manifest["lifecycle"] == "DIAGNOSTIC_SEALED" and manifest["learned_state_unchanged"]
schema_path = ROOT/"configs/ppo_task_conditioned_hip_wheel_v1/observation_schema.json"
assert sha(schema_path) == manifest["observation_schema_sha256"]
clip = json.loads(schema_path.read_text())["clip"]
pres,posts,locations = {},{},{}
for r,loc in records(RUN/"probe_decisions.jsonl"):
    (pres if r["record_kind"]=="pre_action" else posts)[r["decision"]] = r
    locations[r["record_kind"],r["decision"]] = loc
assert len(pres)==len(posts)==600
selected = [i for i,p in pres.items() if (p.get("intervention") or {}).get("index") is not None and p["intervention"]["index"]<75]
assert selected==list(range(372,447))
entry = json.loads((RUN/"probe_entry.json").read_text())
start = entry["pre_action_receipt"]["start_tick"]
end = posts[selected[-1]]["end_tick"]
assert (start,end)==(2976,3576)
holdout = sorted(set(list(range(16)) + [ids[round((len(ids)-1)*fraction)] for phase in ("P02","P03","P04") for ids in [[i for i,p in pres.items() if p["phase"]==phase and p["start_tick"]<start]] if ids for fraction in (0.,1/3,2/3,1.)]))


def verify_pair(i):
    p,q=pres[i],posts[i]; s=q["step_info"]
    expected=p["pre_action_receipt_sha256"]
    clean={k:v for k,v in p.items() if k!="pre_action_receipt_sha256"}
    assert hashlib.sha256(json.dumps(clean,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()==expected==q["pre_action_receipt_sha256"]
    obs=p["observation_encoded_372"]; actor=p["actor_input_float32_372"]
    assert len(obs)==len(actor)==372 and all(math.isfinite(x) for x in obs+actor)
    assert struct.pack("<372f",*obs)==struct.pack("<372f",*actor)
    assert hashlib.sha256(struct.pack("<372f",*actor)).hexdigest()==p["actor_input_float32_le_sha256"]
    for name,desc in p["observation_history_groups"].items():
        scale=desc["scale"]; scales=[scale]*12 if isinstance(scale,(int,float)) else scale
        encoded=[f32(max(-clip,min(clip,x/a))) for x,a in zip(p["actual_live_history"][name],scales)]
        assert encoded==actor[desc["start"]:desc["stop_exclusive"]]
    h=p["actual_live_history"]; ack=p["previous_actual_ACK"]
    if "independent_policy_residual_requested_full12" in ack:
        assert h["previous_residual_full12"]==ack["independent_policy_residual_requested_full12"]
    else:
        assert i==0 and p["start_tick"]==0 and h["previous_residual_full12"]==[0.]*12
    assert h["previous_applied_full12"]==ack["drive_target_full12"]
    assert p["manually_selected_raw_full12"]==s["raw_policy_action_full12"]
    assert p["policy_baseline_raw_full12"][1:]==p["manually_selected_raw_full12"][1:]
    assert all(x==0 for x in p["manual_raw_delta_full12"][1:])
    assert q["start_tick"]==p["start_tick"] and q["end_tick"]==p["start_tick"]+8
    assert q["environment_step_returned"] and s["no_in_episode_state_writes_verified"]
    return dict(decision=i,start_tick=p["start_tick"],end_tick=q["end_tick"],
        pre=locations["pre_action",i],post=locations["step_result",i],initial_reset_observation=(i==0),
        pre_action_receipt_sha256=expected,actor_input_float32_le_sha256=p["actor_input_float32_le_sha256"])


raw = {}; baseline_gap = {}; native = {}
for r,_ in records(RUN/"physical_observations.jsonl"):
    if start<=r["physics_tick"]<=4800: raw[r["physics_tick"]]=r
for r,_ in records(BASE/"physical_observations.jsonl"):
    if r["physics_tick"]>4800: break
    if r["physics_tick"]>=start: baseline_gap[r["physics_tick"]]=r["wheels"]["front_left_ankle"]["bottom_w_m"][2]-r["obstacle"]["top_z_m"]
for r,_ in records(RUN/"native_tick_audit.jsonl"):
    if r["episode_physics_tick"]>end: break
    if r["episode_physics_tick"]>start: native[r["episode_physics_tick"]]=r
assert len(raw)==4800-start+1 and len(native)==600


def gap(t):
    r=raw[t]
    return r["wheels"]["front_left_ankle"]["bottom_w_m"][2]-r["obstacle"]["top_z_m"]


def bodyz(t):
    return raw[t]["body_bounds_w_m"]["base_link"]["minimum_m"][2]


rows=[]
for i in selected:
    row=verify_pair(i); p,q=pres[i],posts[i]; lo,hi=row["start_tick"],row["end_tick"]
    pre_ev=posts[i-1]["step_info"]["semantic_task"]["physical_evaluator"]
    post_ev=q["step_info"]["semantic_task"]["physical_evaluator"]
    assert pre_ev["physics_tick"]==lo and post_ev["physics_tick"]==hi
    fl=[e["current_legs"]["FL"] for e in (pre_ev,post_ev)]
    physical=[raw[t] for t in range(lo+1,hi+1)]
    other={leg:sum(r["contacts"][name+"_wheel"][kind]["active"] and r["contacts"][name+"_wheel"][kind]["pair_verified"] for r in physical) for leg,name,kind in (("FR","front_right","obstacle"),("RR","rear_right","ground"),("RL","rear_left","ground"))}
    audits=[native[t]["native_audit"] for t in range(lo+1,hi+1)]
    headroom_clips=sum(bool(a["policy_headroom_evidence"]["clipped_servo_indices"]) for a in audits)
    dispatch_ok=all(a["verified"] and a["setter_dispatch_targets_equal"] and a["actual_mapping_matches_dispatch"] for a in audits)
    all_air=all(not r["contacts"]["front_left_wheel"][kind]["active"] for r in physical for kind in ("ground","obstacle"))
    strict_xy=all(r["obstacle"]["front_x_m"]<=r["wheels"]["front_left_ankle"]["center_w_m"][0]<=r["obstacle"]["back_x_m"] and r["obstacle"]["right_y_m"]<=r["wheels"]["front_left_ankle"]["center_w_m"][1]<=r["obstacle"]["left_y_m"] for r in physical)
    no_collision=all(not r["body_collision"]["detected"] and not r["body_collision"]["real_pair_active"] for r in physical)
    flags=dict(P05=p["phase"]==q["step_info"]["phase_id"]==q["step_info"]["end_phase_id"]=="P05",
        valid_pre_post=all(e["valid"] and e["termination_reason"] is None for e in (pre_ev,post_ev)),
        FL_crossed_pre_post=all(e["history"]["front_edge_crossed"]["FL"] for e in (pre_ev,post_ev)),
        FL_air_pre_post=all(a["air"] for a in fl),FL_within_top_xy_pre_post=all(a["within_top_xy"] for a in fl),
        FL_air_all8=all_air,FL_center_inside_obstacle_xy_all8=strict_xy,
        no_body_collision_all8=no_collision,no_headroom_clip_all8=headroom_clips==0,
        actual_dispatch_verified_all8=dispatch_ok,other_contacts_valid_all8=all(n==8 for n in other.values()),
        gap_better_than_entry=gap(hi)<gap(start),gap_better_than_same_tick_formal_det=gap(hi)<baseline_gap[hi])
    row.update(intervention_index=p["intervention"]["index"],segment="ramp" if p["intervention"]["index"]<12 else "hold",
        eligible_for_gap_approach=all(flags.values()),reason=[k for k,v in flags.items() if not v],checks=flags,
        FL_gap_pre_post_mm=[1000*gap(lo),1000*gap(hi)],same_tick_formal_det_gap_mm=1000*baseline_gap[hi],
        gap_local_change_mm=1000*(gap(hi)-gap(lo)),body_min_z_pre_post_mm=[1000*bodyz(lo),1000*bodyz(hi)],
        body_min_z_interval_mm=1000*min(bodyz(t) for t in range(lo+1,hi+1)),other_contact_valid_ticks=other,
        selected_hip_raw_issued=p["manually_selected_raw_full12"][0],hip_REQUEST_deg=q["step_info"]["projected_residual_full12"][0],
        capture_label=False)
    rows.append(row)
holdouts=[]
for i in holdout:
    row=verify_pair(i); p=pres[i]
    assert p["phase"]!="P05" and not any(p["manual_raw_delta_full12"])
    row.update(phase=p["phase"],auxiliary_label=False,purpose="mean_sigma_change_holdout_only")
    holdouts.append(row)
summary=json.loads((OUT/"CP189952_FL_minus6_evidence.json").read_text())
source_files={name:dict(path=str(RUN/name),sha256=sha(RUN/name)) for name in ("run_manifest.json","probe_decisions.jsonl","physical_observations.jsonl","native_tick_audit.jsonl","probe_entry.json")}
result=dict(schema="wlr50_clean.FL_minus6_real_AIR_gap_approach_selection.v1",run_dir=str(RUN),
    source_files=source_files,formal_det_physical_reference=dict(path=str(BASE/"physical_observations.jsonl"),sha256=sha(BASE/"physical_observations.jsonl")),
    label_scope="FL_AIR_gap_approach_not_capture",training_executed=False,diagnostic_credit=0,
    selected_action_channel=0,other_action_channels_supervised=False,
    field_locations=dict(observation="pre.observation_encoded_372",actor_input="pre.actor_input_float32_372",history="pre.actual_live_history",raw_issued="pre.manually_selected_raw_full12 == post.step_info.raw_policy_action_full12",nominal="post.step_info.nominal_action_full12",filtered_REQUEST="post.step_info.projected_residual_full12",effective="post.step_info.actuator_target_effect_audit.policy_headroom_evidence.effective_policy_residual_full12",final="post.step_info.actual_drive_target_full12"),
    decision_indices=selected,rows=rows,holdout_non_P05=holdouts,
    window_summary=dict(ticks=[start,end],duration_s=(end-start)/120,decisions=75,ramp_decisions=12,hold_decisions=63,
        eligible_count=sum(r["eligible_for_gap_approach"] for r in rows),
        locally_gap_decreasing_decisions=sum(r["gap_local_change_mm"]<0 for r in rows),
        locally_gap_increasing_decisions=sum(r["gap_local_change_mm"]>0 for r in rows),
        gap_entry_end_min_mm=[1000*gap(start),1000*gap(end),1000*min(gap(t) for t in range(start,end+1))],
        body_min_z_mm=1000*min(bodyz(t) for t in range(start,end+1)),
        release_follow_retained=summary["windows"]["release"],
        follow_retained=summary["windows"]["follow"],after_probe_retained=summary["windows"]["after_probe"],
        sealed_outcome=summary["outcome"],
        release_follow_after_indices=list(range(447,600)),no_capture_positive_labels=True),
    limitations="Eligibility denotes evidence for this whole finite AIR-approach window, not every-step gap reduction, causal isolated hip effect, capture, learned retention or authorization to train. No negative local-progress row dropped. H/obs are exact stored values; no reconstruction or reset. Release and follow remain negative retention evidence and are not auxiliary targets.")
with (OUT/"FL_minus6_gap_approach_selection.json").open("x",encoding="utf-8") as stream:
    json.dump(result,stream,ensure_ascii=False,separators=(",",":"),allow_nan=False)
print(json.dumps(dict(selected=selected,eligible=sum(r["eligible_for_gap_approach"] for r in rows),
    failures=[dict(decision=r["decision"],reason=r["reason"]) for r in rows if not r["eligible_for_gap_approach"]],
    local_progress={s:dict(decreasing=sum(r["gap_local_change_mm"]<0 for r in rows if r["segment"]==s),increasing=sum(r["gap_local_change_mm"]>0 for r in rows if r["segment"]==s)) for s in ("ramp","hold")},
    holdout_indices=holdout),indent=2))
