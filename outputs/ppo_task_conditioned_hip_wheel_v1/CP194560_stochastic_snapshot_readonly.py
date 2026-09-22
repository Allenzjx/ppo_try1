"""CPU-only fixed-byte-prefix decision evidence; no simulator/torch/GPU imports."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RUN = ROOT / "runs/ppo_task_conditioned_hip_wheel_v1/video_eval/validation/20260921T1136245327456Z_g97ecd305afb5_1e424676283e49bdbee680a08111d59e"
PATH = RUN / "source/video_policy_decisions.jsonl"
limit = PATH.stat().st_size
digest = hashlib.sha256(); used = 0; rows = []; transitions = []
with PATH.open("rb") as stream:
    while used < limit:
        line = stream.readline(limit - used)
        if not line.endswith(b"\n"): break
        used += len(line); digest.update(line)
        d = json.loads(line); s = d["step_info"]; task = s["semantic_task"]
        ev = task["physical_evaluator"]
        rows.append(dict(tick=d["end_tick"],start=d["start_tick"],phase=s["phase_id"],end_phase=s["end_phase_id"],
            ev=ev,history=task["history"],semantic_termination=s.get("termination_reason"),
            task_geometry_samples=(s.get("reward") or {}).get("task_space_quality_sample_audit",[])))
        transitions.extend(s.get("stage_transition_evidence", []))
        if not s.get("stage_transition_evidence") and task.get("transition_evidence"):
            value=task["transition_evidence"]
            transitions.extend(value if isinstance(value,list) else [value])
assert rows
last = rows[-1]
def leg_compact(c):
    return {k:c.get(k) for k in ("air","ground_contact","contact_mode","top_surface_contact","bearing_verified", "support",
        "bearing_force_n","within_top_xy","front_distance_m","clearance_m","current_lift_valid","active_attempt","placed_on_top")}
def top(c): return bool(c.get("top_surface_contact") and c.get("bearing_verified") and c.get("support"))
def summary(selected):
    if not selected: return None
    first,end=selected[0],selected[-1]
    cs=[r["ev"]["current_legs"]["FL"] for r in selected]
    geom=[g for r in selected for g in r["task_geometry_samples"] if g.get("eligible") and g.get("valid")
        and g.get("body_collider_minimum_w_m") is not None]
    return dict(endpoint_ticks=[first["tick"],end["tick"]],n=len(selected),
        body_net_forward_m=end["ev"]["goal_features"]["body_forward_m"]-first["ev"]["goal_features"]["body_forward_m"],
        FL_TOP_verified_bearing_endpoints=sum(top(c) for c in cs),FL_AIR_endpoints=sum(c["air"] for c in cs),
        FL_first=leg_compact(cs[0]),FL_last=leg_compact(cs[-1]),FL_gap_max_m=max(c["clearance_m"] for c in cs),
        logged_reward_geometry_body_min_z_m=min((g["body_collider_minimum_w_m"][2] for g in geom),default=None),
        logged_reward_geometry_samples=len(geom),measurement="15Hz decision endpoints; geometry comes separately from logged reward samples; not full raw physical window")
p06=[r for r in rows if r["phase"]=="P06"]
rr=[r for r in rows if r["phase"] in ("P07","P08","P09")]
changes=[]; previous=None
for r in rr:
    c=r["ev"]["current_legs"]["RR"]
    key=(c.get("current_lift_valid"),c.get("contact_mode"),c.get("active_attempt"))
    if key != previous:
        changes.append(dict(tick=r["tick"],phase=r["phase"],RR=leg_compact(c),
            FL=leg_compact(r["ev"]["current_legs"]["FL"]),body_front_m=r["ev"]["goal_features"]["body_forward_m"]))
    previous=key
result=dict(schema="CP194560.stochastic.live_fixed_prefix.v1",run=str(RUN),role="PPO+LIMITEDAUX stochastic fixed-checkpoint evaluation",
    source_receipt=dict(path=str(PATH),snapshot_limit_bytes=limit,consumed_complete_prefix_bytes=used,
        sha256=digest.hexdigest(),complete_decisions=len(rows),last_complete_tick=last["tick"],not_whole_active_file_hash=True),
    episode_final_result=None,semantic_termination_at_last_complete=last["semantic_termination"],
    finality="PROVISIONAL only; wait for authoritative naturally sealed source manifest",
    phase=last["phase"],stage_transitions=transitions,history_event_ticks=last["history"]["event_ticks"],
    RR_lift_events=[e for e in last["history"]["lift_attempt_events"] if e["leg"]=="RR"],
    P06_endpoint_window=summary(p06),RR_preparation_endpoint_window=summary(rr),RR_contact_or_validity_changes=changes,
    latest_physical=dict(valid=last["ev"]["valid"],termination=last["ev"]["termination_reason"],success=last["ev"]["success"],
        legs={leg:leg_compact(c) for leg,c in last["ev"]["current_legs"].items()}),
    limitations="No raw physical stream was required while its buffered file was empty. No native actual wheel qd or traction inference. History qualified/placed not current support. No zero filling or fixed-checkpoint stability ranking from incomplete windows.")
with (OUT / "CP194560_stochastic_prefix_readonly.json").open("x",encoding="utf-8") as stream:
    json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps({k:result[k] for k in ("source_receipt","phase","history_event_ticks","RR_lift_events","P06_endpoint_window","latest_physical")},indent=2))
