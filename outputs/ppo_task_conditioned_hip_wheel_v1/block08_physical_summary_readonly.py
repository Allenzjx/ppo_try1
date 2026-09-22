"""Bounded saved-data summary; no simulator, model, or production imports."""
import json
from collections import Counter
from pathlib import Path

OUT = Path(__file__).resolve().parent
receipt = json.loads((OUT / "block08_P01_1536_receipt.json").read_text())
RUN = Path(receipt["run"])


def lines(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            assert line.endswith("\n"), "sealed file has incomplete record"
            yield json.loads(line)


groups = [[]]
for row in lines(RUN / "residual_and_projection_audit.jsonl"):
    s = row["applied_audit"]
    task = s["semantic_task"]
    reward = s["reward_breakdown"]
    groups[-1].append(dict(global_decision=row["global_policy_decision"],
        tick=s["physics_tick"], phase=s["phase_id"], end_phase=s["end_phase_id"],
        terminal=row["terminal"], reason=s["termination_reason"],
        bootstrap=s["terminal_bootstrap_allowed"], time_outs=s["time_outs"],
        ev=task["physical_evaluator"], history=task["history"],
        transitions=s["stage_transition_evidence"],
        geometry=reward["task_space_quality_sample_audit"],
        front=reward["front_quality_sample_audit"],
        costs=reward["cost_components"], reward=reward["total"],
        terminal_event=reward["terminal_event"]))
    if row["terminal"]:
        groups.append([])
assert len(groups) == 3 and [len(g) for g in groups] == [612, 644, 280]
assert groups[-1][-1]["global_decision"] == 192000

episodes = []
for index, group in enumerate(groups):
    last = group[-1]
    hist = last["history"]
    transitions = [t for r in group for t in r["transitions"]]
    starts = {t["to_stage"]: t["physics_tick"] for t in transitions}
    gs = [a for r in group for a in r["geometry"] if a["valid"]]
    p06 = None
    if "P06" in starts and "P07" in starts:
        lo, hi = starts["P06"], starts["P07"]
        window = [r for r in group if lo <= r["tick"] <= hi]
        assert window[0]["tick"] == lo and window[-1]["tick"] == hi
        fl = [r["ev"]["current_legs"]["FL"] for r in window]
        top = sum(a["top_surface_contact"] and a["bearing_verified"] and a["support"] for a in fl)
        geometry = [a for a in gs if lo < round(a["sim_time_s"] * 120) <= hi]
        p06 = dict(ticks=[lo, hi], duration_s=(hi-lo)/120,
            body_net_forward_m=window[-1]["ev"]["goal_features"]["body_forward_m"]-window[0]["ev"]["goal_features"]["body_forward_m"],
            sampled_endpoints=len(window), FL_top_bearing_support_endpoints=top,
            FL_top_bearing_support_fraction=top/len(window), FL_air_endpoints=sum(a["air"] for a in fl),
            FL_first_saved_air_tick=next((r["tick"] for r in window if r["ev"]["current_legs"]["FL"]["air"]), None),
            FL_max_gap_m=max(a["clearance_m"] for a in fl),
            body_collider_min_z_m=min(a["body_collider_minimum_w_m"][2] for a in geometry),
            AABB_separation_lower_bound_min_m=min(a["separation_lower_bound_m"] for a in geometry),
            geometry_samples=len(geometry))
    rr_window = [r for r in group if r["tick"] >= starts.get("P07", float("inf"))]
    body = [a for a in gs if round(a["sim_time_s"]*120) >= starts.get("P07", float("inf"))]
    quality = dict(front_sample_count=sum(len(r["front"]) for r in group),
        front_cost=sum(a["weighted_quality_cost"] for r in group for a in r["front"]),
        geometry_eligible_samples=sum(a["eligible"] for r in group for a in r["geometry"]),
        geometry_valid_eligible_samples=sum(a["eligible"] and a["valid"] for r in group for a in r["geometry"]),
        geometry_cost=sum(a["weighted_geometry_cost"] for a in gs),
        terminal_event_sum=sum(r["terminal_event"] for r in group))
    episodes.append(dict(index=index, decisions=len(group),
        global_range=[group[0]["global_decision"],last["global_decision"]],
        end_tick=last["tick"], duration_s=last["tick"]/120,
        terminal=last["terminal"], reason=last["reason"],
        physical_termination=last["ev"]["termination_reason"],
        physical_valid=last["ev"]["valid"], success=last["ev"]["success"],
        end_phase=last["end_phase"], bootstrap=last["bootstrap"], time_outs=last["time_outs"],
        event_ticks=hist["event_ticks"], P06=p06,
        RR_events=[e for e in hist["lift_attempt_events"] if e["leg"] == "RR"],
        RR_current_valid_saved_ticks=[r["tick"] for r in rr_window if r["ev"]["current_legs"]["RR"]["current_lift_valid"]],
        RR_end={k:last["ev"]["current_legs"]["RR"][k] for k in ("current_lift_valid","contact_mode","active_attempt","front_distance_m","clearance_m")},
        FL_end={k:last["ev"]["current_legs"]["FL"].get(k) for k in ("contact_surface","air","support","top_surface_contact","clearance_m")},
        first_body_z_below_tick={str(z):next((round(a["sim_time_s"]*120) for a in body if a["body_collider_minimum_w_m"][2]<z),None) for z in (.1,.08,.06,.05)},
        request_phase_counts=dict(Counter(r["phase"] for r in group)), quality=quality))

assert abs(sum(e["quality"]["front_cost"] for e in episodes)-receipt["actual_quality_contributions"]["front_cost"]) < 1e-10
assert abs(sum(e["quality"]["geometry_cost"] for e in episodes)-receipt["actual_quality_contributions"]["geometry_cost_known"]) < 1e-10
assert episodes[0]["P06"]["sampled_endpoints"] == 193
assert abs(episodes[0]["P06"]["body_net_forward_m"]-.2680232524871826) < 1e-12
result = dict(schema="block08.physical_summary_readonly.v1", source=str(RUN),
    data_scope="sealed 1536 decisions; on-policy actor updated within trajectories, not fixed checkpoint evaluation",
    episodes=episodes, phase_coverage=receipt["actual_request_phase_coverage"],
    actual_quality_contributions=receipt["actual_quality_contributions"],
    actual_geometry_quality_by_phase=receipt["actual_geometry_quality_by_phase"],
    limitation="Contact fractions are saved evaluator endpoints, not continuous support duration. Body displacement uses actual base minus fixed obstacle front, not wheel motion. AABB lower bound is not exact mesh clearance. No missing raw/hip/COM data invented.")
with (OUT / "block08_physical_summary.json").open("x",encoding="utf-8") as stream:
    json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps(episodes,ensure_ascii=False,indent=2))
