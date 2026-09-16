"""Small scalar rows for root's CSV authorer; completed evidence only, no CSV."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TIMING = HERE.parent / "ppo_timing_task_priority_v1"
FIELDS = "candidate event_family event_detail occurrence phase physics_tick simulation_time_s source_event_pre_step_tick source_event_post_step_tick sampled_last_command_pre_tick sampled_last_command_post_tick owner_observation_tick decision_end_tick body_collider_min_recorded_m body_collider_min_independent_m base_origin_z_m RR_mount_USD_z_m RR_mount_URDF_estimate_z_m RR_upper_link_origin_measured_z_m RR_mount_to_wheel_center_vertical_distance_m RR_front_distance_m RR_top_gap_m FL_source_N_deg FL_final_target_deg FL_actual_deg RL_source_N_deg RL_final_target_deg RL_actual_deg RR_hip_source_N_deg RR_hip_final_target_deg RR_hip_actual_deg RR_knee_source_N_deg RR_knee_mapper_N_deg RR_knee_geometry_adjustment_deg RR_knee_geometry_corrected_N_deg RR_knee_final_target_deg RR_knee_actual_deg RR_knee_actual_lower_margin_deg RR_policy_interval_lower_deg RR_policy_interval_upper_deg RR_initial_lift_current RR_qualified_lift_current RR_current_lift_valid RR_history_Q RR_history_C RR_history_P RR_ground_contact RR_support RR_bearing_force_N RR_load_fraction RR_load_fraction_valid FL_source_reduction_deg RL_source_reduction_deg next_native_FL_N_deg next_native_RL_N_deg body_collision task_success termination_reason height_sample_exact_tick height_sample_before_tick height_sample_after_tick geometry_mode_and_confound source_physical_path source_physical_line source_native_path source_native_line source_owner_path source_owner_line evidence_pointer unknown_and_clock_notes".split()
FIELDS += "evaluation_termination_reason controller_termination_reason source_acceptance_error source_terminal_decision_path source_terminal_decision_line".split()
FIELDS = ["RR_mount_to_wheel_link_origin_vertical_distance_m" if field ==
          "RR_mount_to_wheel_center_vertical_distance_m" else field for field in FIELDS]
FIELDS += "candidate_source_head candidate_runtime_sha256 RR_wheel_link_origin_z_m RR_wheel_collider_min_z_m RR_wheel_center_z_m".split()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def get(value, *keys):
    for key in keys:
        if isinstance(value, dict): value = value.get(key)
        elif isinstance(value, (list, tuple)) and isinstance(key, int) and 0 <= key < len(value): value = value[key]
        else: return None
    return value


def base_row(candidate, family, tick, detail, pointer):
    row = dict.fromkeys(FIELDS)
    row.update(candidate=candidate, event_family=family, event_detail=detail, occurrence=1,
        physics_tick=tick, simulation_time_s=tick/120., sampled_last_command_pre_tick=tick-1 if tick else None,
        sampled_last_command_post_tick=tick if tick else None, evidence_pointer=pointer)
    return row


def final_row(path):
    """Read only the last nonblank JSON record, preserving its real source line."""
    last, line_number = None, None
    with path.open(encoding="utf-8-sig") as stream:
        for index, line in enumerate(stream, 1):
            if line.strip(): last, line_number = line, index
    if last is None: raise ValueError(f"empty finalized evidence: {path}")
    return json.loads(last), line_number


def terminal_metadata(source, aggregate):
    manifest = load(source / "semantic_video_source_manifest.json")
    run = load(source.parent / "run_manifest.json")
    if not run.get("completed_at_utc") or run.get("lifecycle") in (None, "STARTED", "RUNNING"):
        raise ValueError("terminal supplement requires finalized run")
    tick = aggregate["terminal"]["tick"]
    if manifest.get("episode_physics_ticks") != tick or run.get("runtime_contract") != manifest.get("runtime_contract"):
        raise ValueError("terminal supplement source/runtime mismatch")
    decision_path = source / "video_policy_decisions.jsonl"
    decision, line = final_row(decision_path)
    physical, _ = final_row(source / "physical_observations.jsonl")
    if physical.get("physics_tick") != tick: raise ValueError("physical terminal tick mismatch")
    # Safety may end inside a decision. Do not relabel the preceding decision as terminal.
    controller = get(decision, "step_info", "termination_reason") if decision.get("end_tick") == tick else None
    evaluator = aggregate["terminal"].get("evaluation_termination_reason")
    return dict(termination_reason=evaluator or controller,
        evaluation_termination_reason=evaluator, controller_termination_reason=controller,
        source_acceptance_error=aggregate["terminal"].get("source_acceptance_error"),
        source_terminal_decision_path=str(decision_path) if decision.get("end_tick") == tick else None,
        source_terminal_decision_line=line if decision.get("end_tick") == tick else None,
        body_collision=get(physical, "body_collision", "detected"))


def reference_rows():
    path = HERE / "existing_A_B2_height_window.json"
    values = {(r["run_label"], r["physics_tick"]): r for r in load(path)["rows"]}
    full = {r["episode_physics_tick"]: r for r in load(TIMING / "B_AFTER_FULL.records.json")}
    selections = {"A_FROZEN": [("P07_SOURCE_START", 6648), ("P08_SOURCE_START", 6856),
        ("RR_FIRST_QUALIFICATION", 6918), ("RR_CROSS_EVENT", 7109), ("RR_PLACE_EVENT", 7579), ("P09_FINITE_SOURCE_TAIL", 7777)],
        "B2_ZERO": [("P07_SOURCE_START", 5160), ("P08_SOURCE_START", 5360),
        ("RR_FIRST_QUALIFICATION", 5425), ("P09_FINITE_SOURCE_TAIL", 6273), ("TERMINAL", 6374)]}
    result = []
    for candidate, events in selections.items():
        for family, tick in events:
            raw = values[candidate, tick]
            r = base_row(candidate, family, tick, family, f"{path}#/rows where run_label={candidate} and physics_tick={tick}")
            r.update(body_collider_min_recorded_m=raw["body_collision_min_z_m"], base_origin_z_m=raw["base_origin_z_m"],
                RR_mount_URDF_estimate_z_m=get(raw, "legs", "RR", "hip_mount_URDF_transformed_w_m", 2),
                RR_upper_link_origin_measured_z_m=get(raw, "legs", "RR", "hip_upper_link_origin_w_m", 2),
                RR_front_distance_m=get(raw, "legs", "RR", "front_distance_m"), RR_top_gap_m=get(raw, "legs", "RR", "top_gap_m"),
                source_physical_path=raw["source_physical_path"], source_physical_line=raw["source_physical_line"],
                source_native_path=raw["source_command_path"], source_native_line=raw["source_command_line"], body_collision=raw["body_collision_detected"])
            for leg in ("FL", "RL"):
                r.update({f"{leg}_{suffix}": raw["legs"][leg][field] for suffix, field in
                    (("source_N_deg", "hip_N_deg"), ("final_target_deg", "hip_final_deg"), ("actual_deg", "hip_actual_deg"))})
            for joint in ("hip", "knee"):
                r.update({f"RR_{joint}_{suffix}": raw["legs"]["RR"][joint + "_" + field] for suffix, field in
                    (("source_N_deg", "N_deg"), ("final_target_deg", "final_deg"), ("actual_deg", "actual_deg"))})
            r["RR_knee_actual_lower_margin_deg"] = raw["legs"]["RR"]["knee_lower_margin_deg"]
            if "SOURCE_START" in family:
                r.update(source_event_pre_step_tick=tick, source_event_post_step_tick=tick+1)
            if candidate == "A_FROZEN":
                r["geometry_mode_and_confound"] = "Frozen A cached wheel extents; old ACTIVE_LIFT semantics, source entry/VERIFY behavior differs; not a same-runtime controlled comparison"
                r["unknown_and_clock_notes"] = "A body collider min and direct USD mount were not retained. 7777 is RR source tail, NOT full-run terminal. Old qualification is not current semantic Q."
                if family == "RR_FIRST_QUALIFICATION": r["event_detail"] = "A legacy ACTIVE_LIFT latch; not equivalent to revised Q"
                if family in ("RR_FIRST_QUALIFICATION", "RR_CROSS_EVENT", "RR_PLACE_EVENT", "P09_FINITE_SOURCE_TAIL"): r["RR_history_Q"] = True
                if family in ("RR_CROSS_EVENT", "RR_PLACE_EVENT", "P09_FINITE_SOURCE_TAIL"): r["RR_history_C"] = True
                if family in ("RR_PLACE_EVENT", "P09_FINITE_SOURCE_TAIL"): r["RR_history_P"] = True
            else:
                r["geometry_mode_and_confound"] = "B2 original link-origin nominal geometry projection; no height candidate; differs from new collider-point/trust-region geometry as well as FL/RL candidate"
                r["unknown_and_clock_notes"] = "Direct USD mount/independent height stream absent; URDF estimate is separate. Missing exact sparse headroom stays null."
                sparse = full.get(tick, {})
                for field in ("mapper_N_deg", "nominal_geometry_adjustment_deg", "geometry_corrected_N_deg"):
                    source = {"mapper_N_deg": "mapped_N_deg"}.get(field, field)
                    destination = field.replace("nominal_geometry", "geometry")
                    r["RR_knee_" + destination] = sparse.get("RR_knee_" + source)
                if family == "TERMINAL":
                    r.update(task_success=False, termination_reason="SAFETY_ABORT / HARD_JOINT_LIMIT",
                             RR_history_Q=True, RR_history_C=False, RR_history_P=False)
            result.append(r)
    return result


def candidate_rows(path):
    data = load(path)
    source = Path(data["source"])
    terminal = terminal_metadata(source, data)
    events = data["events"]
    last_revocation = max((e["physics_tick"] for e in events if e["event"] == "RR_current_lift_revoked_ground"), default=None)
    selected = [(i, e) for i, e in enumerate(events) if e["event"] in ("P07_SOURCE_START", "P08_SOURCE_START", "RR_Q", "RR_C", "RR_P", "P09_LATE_GROUP_RELEASE", "TERMINAL")
        or e["event"] == "RR_current_lift_revoked_ground" and e["physics_tick"] == last_revocation]
    reductions = data["height_reductions_at_actual_observation_clocks"]
    candidate_id = get(reductions, 0, "candidate_id") or data["label"]
    result = []
    for index, e in selected:
        tick, p = e["physics_tick"], e["physical"]
        family = ("RR_FIRST_QUALIFICATION" if e["event"] == "RR_Q" and e["occurrence"] == 1 else
                  "RR_RETRY_QUALIFICATION" if e["event"] == "RR_Q" else
                  "RR_CROSS_EVENT" if e["event"] == "RR_C" else "RR_PLACE_EVENT" if e["event"] == "RR_P" else
                  "PRE_LATE_LAST_GROUND_REVOCATION" if e["event"] == "RR_current_lift_revoked_ground" else e["event"])
        r = base_row(candidate_id, family, tick, e["event"], f"{path.resolve()}#/events/{index}")
        exact, rr = e["height"]["exact"], p.get("exact_semantic_RR") or {}
        if e["event"] == "TERMINAL": rr = data["terminal"].get("RR") or {}
        r.update(phase=p["source_phase_id"], occurrence=e["occurrence"], body_collider_min_recorded_m=p["recorded_body_collider_min_z_m"],
            base_origin_z_m=p["base_origin_height_world_m"], RR_upper_link_origin_measured_z_m=p["measured_RR_upper_link_origin_z_m"],
            body_collider_min_independent_m=get(exact, "body_collision_minimum_z_w_m"), RR_mount_USD_z_m=get(exact, "rr_hip_mount_w_m", "value", 2),
            height_sample_exact_tick=get(exact, "physics_tick"), height_sample_before_tick=get(e, "height", "before", "physics_tick"),
            height_sample_after_tick=get(e, "height", "after", "physics_tick"), RR_front_distance_m=p["RR_front_distance_derived_m"],
            RR_top_gap_m=p["RR_top_clearance_derived_m"], RR_knee_mapper_N_deg=p["RR_knee_mapped_N_deg"],
            RR_knee_geometry_adjustment_deg=p["RR_knee_geometry_adjustment_deg"], RR_knee_geometry_corrected_N_deg=p["RR_knee_geometry_corrected_N_deg"],
            RR_knee_actual_lower_margin_deg=p["RR_knee_lower_margin_deg"], RR_policy_interval_lower_deg=get(p, "RR_knee_policy_residual_interval_deg", 0),
            RR_policy_interval_upper_deg=get(p, "RR_knee_policy_residual_interval_deg", 1),
            owner_observation_tick=get(e, "owner", "observation_tick"), decision_end_tick=get(e, "owner", "decision_end_tick"),
            source_physical_path=str(source / "physical_observations.jsonl"), source_physical_line=p["physical_source_line"],
            source_native_path=str(source / "native_tick_audit.jsonl"), source_native_line=p["native_source_line"],
            source_owner_path=str(source / "video_policy_decisions.jsonl") if e.get("owner") else None,
            source_owner_line=get(e, "owner", "decision_line"),
            geometry_mode_and_confound="New height/collider-point geometry revision; BOTH nominal geometry and candidate differ versus B2. Same initial arrays do not remove this confound.",
            unknown_and_clock_notes="Physical/q exact120Hz. Independent height only exact grid samples; before/after are NOT interpolated. Current Q/support/load null when no exact semantic endpoint.")
        link_origin_z = get(exact, "RR_fresh_collider", "value", "link_origin_w_m", 2)
        r.update(candidate_source_head=data["source_head"], candidate_runtime_sha256=data["source_runtime_sha256"],
            RR_wheel_link_origin_z_m=link_origin_z,
            RR_wheel_collider_min_z_m=get(exact, "RR_fresh_collider", "value", "minimum_m", 2))
        # No independently identified wheel-center point in these records. A link
        # origin is a different named measurement; leave wheel-center null.
        if r["RR_mount_USD_z_m"] is not None and link_origin_z is not None:
            r["RR_mount_to_wheel_link_origin_vertical_distance_m"] = r["RR_mount_USD_z_m"] - link_origin_z
        for prefix in ("FL", "RL", "RR_hip", "RR_knee"):
            source_prefix = prefix + "_hip" if prefix in ("FL", "RL") else prefix
            for destination, original in (("source_N_deg", "nominal_request_deg"), ("final_target_deg", "final_target_deg"), ("actual_deg", "actual_deg")):
                r[prefix + "_" + destination] = p[source_prefix + "_" + original]
        for destination, original in (("initial_lift_current", "initial_lift_observed"), ("qualified_lift_current", "lift_established"),
            ("current_lift_valid", "current_lift_valid"), ("ground_contact", "ground_contact"), ("support", "support"),
            ("bearing_force_N", "bearing_force_n"), ("load_fraction", "load_fraction"), ("load_fraction_valid", "load_fraction_valid")):
            r["RR_" + destination] = rr.get(original)
        if e["event"] in ("P07_SOURCE_START", "P08_SOURCE_START", "P09_LATE_GROUP_RELEASE"):
            r.update(source_event_pre_step_tick=tick, source_event_post_step_tick=tick+1)
        for leg, channel in (("FL", "front_left_hip"), ("RL", "rear_left_hip")):
            annotation = next((a for a in reductions if a["channel"] == channel and a["source_observation_tick"] == tick), None)
            if annotation:
                owner = next(o for o in annotation["owners"] if o["channel"] == channel)
                r[leg + "_source_reduction_deg"] = owner["reduction_deg"]
            r["next_native_" + leg + "_N_deg"] = get(e, "next_post_step", leg + "_hip_nominal_request_deg")
        if e["event"] == "TERMINAL":
            r.update(task_success=data["terminal"]["task_success"], **terminal)
            for column, key in (("RR_history_Q", "active_lift"), ("RR_history_C", "front_edge_crossed"), ("RR_history_P", "placed")):
                r[column] = data["terminal"]["RR_history"].get(key)
        result.append(r)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise ValueError("refusing overwrite")
    rows = reference_rows()
    for path in args.aggregate: rows.extend(candidate_rows(path))
    assert all(set(row) == set(FIELDS) for row in rows)
    result = {"schema": "wlr50_clean.height_event_scalar_rows.v2", "rows": rows,
        "row_count": len(rows), "columns": FIELDS, "source_aggregates": [str(p.resolve()) for p in args.aggregate],
        "units_and_clocks": "All *_deg canonical relative-to-standing; metres are world heights/distances, NOT pixels. q/target is commandpre=t-1 -> physicalpost=t. source-start eventpre=t -> nextdispatchpost=t+1. True USD mount / URDF estimate / measured upper origin are separate.",
        "unknown_contract": "null means unavailable, never zero. Wheel link origin is NOT an independently identified wheel center; center stays unknown. Reference A terminal outside selected P09 window is not fabricated. Current Q differs from historical Q; old A qualification semantics differ from new. Source acceptance errors are NOT physical/controller termination reasons; their columns remain separate.",
        "comparison_contract": "Event-family descriptive comparison, not synchronized trajectories or FL-only causal test. Original B2 geometry differs from revised candidate geometry. Candidate-prelate revocation is its own event, not matched to A/B2 source tail."}
    with args.output.open("x", encoding="utf-8") as stream: json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({"output": str(args.output.resolve()), "rows": len(rows), "columns": len(FIELDS), "CSV_written": False}))


if __name__ == "__main__": main()
