"""Completed zero-run event JSON, exact clocks; no CSV, media, or simulation.

python aggregate_zero_height_run.py --run RUN --label B_HEIGHT_1 --output NEW.json
python aggregate_zero_height_run.py --self-test
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("height_after_extract", ROOT / "outputs/ppo_timing_task_priority_v1/after_run_extract.py")
after = importlib.util.module_from_spec(spec)
spec.loader.exec_module(after)
old, get, require = after.old, after.get, after.require
load = after.load
JOINTS = ("FL_hip", "RL_hip", "RR_hip", "RR_knee")


def source_durations(contract):
    import yaml
    payloads = {}
    for name in ("recording_motion_contract.json", "fsm_states.yaml"):
        path = ROOT / "configs" / name
        data = path.read_bytes()
        require(hashlib.sha256(data).hexdigest() == contract["files"]["configs/" + name], "source cadence hash mismatch")
        payloads[name] = json.loads(data) if name.endswith("json") else yaml.safe_load(data)
    states = {s["state_id"]: s for s in payloads["fsm_states.yaml"]["states"]}
    return {p["state_id"]: round(round(p["active_duration_s"] * 120) * states[p["state_id"]]["normal_time_scale"])
            for p in payloads["recording_motion_contract.json"]["phases"] if p["state_id"] in ("P07", "P08")}


def owner_events(owners, durations):
    result = []
    for stage in ("P07", "P08"):
        rows = [r for r in owners if r["stage"] == stage and type(r.get("observation_tick")) is int]
        started = next((r for r in rows if type(r.get("actual_start_tick")) is int), None)
        if started:
            result.append({"event": stage + "_SOURCE_START", "physics_tick": started["actual_start_tick"],
                "clock_semantics": "actual source start pre-step observation; next native post-step is tick+1", "owner": started})
        for index, row in enumerate(rows):
            if isinstance(row.get("source_ticks"), int) and row["source_ticks"] > durations[stage]:
                result.append({"event": stage + "_ENDPOINT_FIRST_OBSERVED", "physics_tick": row["observation_tick"],
                    "clock_semantics": "first recorded owner sample beyond endpoint; NOT exact endpoint-onset claim",
                    "source_endpoint_local_tick": durations[stage], "owner": row,
                    "previous_owner_observation_tick": rows[index-1]["observation_tick"] if index else None})
                break
        if rows:
            result.append({"event": stage + "_LAST_OWNER_OBSERVATION", "physics_tick": rows[-1]["observation_tick"],
                           "clock_semantics": "last actual owner observation, not necessarily physical terminal", "owner": rows[-1]})
    return result


def rr_events(events, evaluation):
    result = [{"event": "RR_I" if e["event"] == "whole_body_initial_clearance" else "RR_Q" if
        e["event"] == "qualified_measured_upward_lift" else "RR_" + e["event"], "physics_tick": e["physics_tick"],
        "clock_semantics": "actual physical event tick", "event_evidence": e} for e in events]
    for key, label in (("front_edge_crossed", "RR_C"), ("placed", "RR_P")):
        tick = get(evaluation, "history", "event_ticks", key, "RR")
        if type(tick) is int:
            result.append({"event": label, "physics_tick": tick, "clock_semantics": "physical evaluator event tick"})
    return result


def number_occurrences(events):
    counts = {}
    for event in events:
        counts[event["event"]] = counts.get(event["event"], 0) + 1
        event["occurrence"] = counts[event["event"]]
    return events


def late_group_events(endpoints):
    """Keep the actual source-release clock, not the later decision endpoint."""
    held, releases = None, {}
    for endpoint in endpoints.values():
        layers = get(endpoint, "task", "nominal_provider_diagnostics", "source_partial_order", "layers") or []
        for layer in layers:
            if layer.get("stage") != "P09":
                continue
            owner = {key: layer.get(key) for key in ("stage", "observation_tick", "source_ticks", "status", "wait_reason",
                "late_group_start_tick", "late_group_source_tick", "RR_late_front_distance_m", "RR_late_clearance_m", "RR_late_within_top_xy")}
            owner.update(decision_line=endpoint["line"], decision_end_tick=endpoint["observation_tick"])
            observed, released = owner["observation_tick"], owner["late_group_start_tick"]
            if held is None and owner["status"] == "holding" and owner["wait_reason"] == "current_RR_over_top_before_late_reconfiguration" and type(observed) is int:
                held = {"event": "P09_LATE_GROUP_HOLD_FIRST_OBSERVED", "physics_tick": observed, "owner": owner,
                    "clock_semantics": "first recorded holding-owner observation, NOT exact hold-onset claim"}
            if type(released) is int and released not in releases:
                releases[released] = {"event": "P09_LATE_GROUP_RELEASE", "physics_tick": released, "owner": owner,
                    "clock_semantics": "recorded source release pre-step; next native post-step tick+1 verifies actual command change; not capture causality"}
    return ([held] if held else []) + list(releases.values())


def height_samples(source, end):
    values = {}
    for line, row in old.lines(source / "height_diagnostics.jsonl"):
        tick = row.get("physics_tick")
        require(type(tick) is int and tick not in values and 0 <= tick <= end, "invalid height sample tick")
        require(abs(row["simulation_time_s"] - tick/120) < 1e-10, "height sample clock mismatch")
        require(row["schema"] == "wlr50_clean.height_diagnostic_sample.v1", "unknown height schema")
        values[tick] = {"source_line": line, "physics_tick": tick,
            "body_collision_minimum_z_w_m": row.get("body_collision_minimum_z_w_m"),
            "body_collision_minimum_reason": row.get("body_collision_minimum_reason"),
            "rr_hip_mount_w_m": row.get("rr_hip_mount_w_m"),
            "RR_fresh_collider": get(row, "fresh_collider_bounds", "rear_right_wheel"),
            "clock_unchanged": row.get("clock_unchanged"), "terminal_sample": row.get("terminal_sample")}
    require(list(values) == [*range(0, end+1, 8), *([] if end % 8 == 0 else [end])], "height stream not complete tick0/every8/terminal")
    return values


def choose_height(values, tick):
    exact = values.get(tick)
    before = max((t for t in values if t < tick), default=None)
    following = min((t for t in values if t > tick), default=None)
    return {"exact": exact, "before": None if exact else values.get(before),
            "after": None if exact else values.get(following),
            "reason": None if exact else "event has no encoded diagnostic sample; neighbor clocks retained, no interpolation"}


def compact(flat, physical, native, endpoint):
    row = {key: flat.get(key) for key in ("episode_physics_tick", "simulation_time_s", "source_phase_id",
        "base_origin_height_world_m", "calibrated_body_rpy_rad", "com_position_world_m",
        "RR_front_distance_derived_m", "RR_top_clearance_derived_m", "physical_source_line", "native_source_line")}
    for prefix in JOINTS:
        row.update({prefix + "_" + field: flat.get(prefix + "_" + field) for field in
            ("nominal_request_deg", "mapped_N_deg", "final_target_deg", "actual_deg", "velocity_deg_s")})
    audit = get(native, "native_audit") or {}
    row["post_step_nominal_full12"] = get(native, "nominal_full12")
    row["post_step_actual_commanded_full12"] = physical.get("commanded_full12")
    head = audit.get("policy_headroom_evidence") or {}
    row.update(RR_knee_lower_margin_deg=old.difference(row["RR_knee_actual_deg"], -60),
        RR_knee_geometry_adjustment_deg=get(audit, "nominal_geometry_adjustment_full12", 7),
        RR_knee_geometry_corrected_N_deg=get(head, "geometry_corrected_native_full12", 7),
        RR_knee_effective_policy_residual_deg=get(head, "effective_policy_residual_full12", 7),
        RR_knee_policy_residual_interval_deg=get(head, "policy_residual_intervals_servo_deg", 7),
        baseline_outside_reserved_servo_indices=head.get("baseline_outside_reserved_servo_indices"),
        servo_reserve_deg=head.get("servo_reserve_deg"),
        RR_knee_direct_current_policy_delta_deg=flat.get("RR_knee_direct_same_tick_policy_delta_deg"),
        geometry_status=get(audit, "nominal_geometry_evidence", "status"),
        geometry_context_tick=get(audit, "nominal_geometry_evidence", "context", "source_control_tick"),
        recorded_body_collider_min_z_m=get(physical, "body_bounds_w_m", "base_link", "minimum_m", 2),
        measured_RR_upper_link_origin_z_m=get(physical, "bodies", "rear_right_upper", "position_w_m", 2),
        exact_semantic_RR=None if endpoint is None else {key: get(endpoint, "task", "physical_evaluator", "current_legs", "RR", key)
            for key in ("initial_lift_observed", "lift_established", "current_lift_valid", "ground_contact", "support", "bearing_force_n", "load_fraction", "load_fraction_valid")})
    return row


def compact_baseline(row):
    if row is None:
        return None
    result = {key: row[key] for key in ("physics_tick", "simulation_time_s", "source_physical_path", "source_physical_line",
        "source_command_path", "source_command_line", "body_collision_min_z_m", "base_origin_z_m", "body_collision_detected")}
    result["RR_URDF_mount_z_m"] = get(row, "legs", "RR", "hip_mount_URDF_transformed_w_m", 2)
    result["RR_measured_upper_link_origin_z_m"] = get(row, "legs", "RR", "hip_upper_link_origin_w_m", 2)
    result["legs"] = {leg: {key: row["legs"][leg].get(key) for key in ("hip_N_deg", "hip_final_deg", "hip_actual_deg", "contact_class",
        "knee_N_deg", "knee_final_deg", "knee_actual_deg", "knee_lower_margin_deg", "front_distance_m", "top_gap_m")}
        for leg in ("FL", "RL", "RR")}
    return result


def aggregate(run, label):
    source, manifest = old.completed_source(run)
    final = load(source / "semantic_video_source_manifest.json")
    end = final.get("episode_physics_ticks")
    require(final.get("role") == "B" and final.get("optimizer_updates") == 0 and type(end) is int and 0 < end <= 24000, "not a completed zero evaluation")
    require(get(final, "height_diagnostics", "stream_closed") is True, "new diagnostic stream not finalized")
    endpoints, owners, activations, skipped, phases, issued, returned = after.decisions(source)
    changes, attempts = after.transitions(source)
    evaluation = get(final, "physical_episode", "physical_task_evaluation") or {}
    durations = source_durations(final["runtime_contract"])
    events = [{"event": "INITIAL", "physics_tick": 0}, *owner_events(owners, durations), *late_group_events(endpoints),
              *rr_events(attempts, evaluation), {"event": "TERMINAL", "physics_tick": end}]
    interventions = []
    for endpoint in endpoints.values():
        diag = get(endpoint, "task", "nominal_provider_diagnostics", "height_recovery") or {}
        if diag:
            interventions.append({"decision_line": endpoint["line"], "decision_end_tick": endpoint["observation_tick"],
                **{key: diag.get(key) for key in ("candidate_id", "source_observation_tick", "current_RR_carry_recovery_permitted", "post_lift_recovery_offsets_deg", "owners")}})
    selected_interventions = []
    for channel in ("front_left_hip", "rear_left_hip"):
        changed = [r for r in interventions if any(o["channel"] == channel and o["reduction_deg"] > 0 for o in r.get("owners") or [])]
        if changed:
            peak = max(changed, key=lambda r: max(o["reduction_deg"] for o in r["owners"] if o["channel"] == channel))
            for kind, record in (("FIRST", changed[0]), ("MAX", peak), ("LAST", changed[-1])):
                selected_interventions.append({"channel": channel, "selection": kind, **record})
                events.append({"event": channel + "_REDUCTION_" + kind, "physics_tick": record["source_observation_tick"],
                               "clock_semantics": "nominal annotation pre-step; actual native dispatch checked separately at tick+1"})
    number_occurrences(events)
    wanted = {e["physics_tick"] for e in events}
    require(all(type(t) is int and 0 <= t <= end for t in wanted), "event outside episode")
    wanted |= {t+1 for t in wanted if t < end}
    heights = height_samples(source, end)
    calibration, proof = old.load_calibration(manifest)
    stream = iter(old.lines(source / "native_tick_audit.jsonl"))
    current = next(stream, None)
    require(current is not None, "missing native stream")
    initial, records, maxima, tick = None, {}, {}, -1
    for line, physical in old.lines(source / "physical_observations.jsonl"):
        old.physical_schema(physical)
        require(physical["physics_tick"] == tick+1 <= end, "physical stream gap or overflow")
        tick += 1
        require(tick == 0 or current is not None, "native stream ended before physical stream")
        native_line, native = (None, None) if tick == 0 else current
        if tick == 0:
            initial = old.initial_state(physical, current[1])
        else:
            old.native_schema(native)
            require(native["episode_physics_tick"] == tick and get(native, "native_audit", "verified") is True, "native clock/unverified audit")
            for kind, values in (("raw", get(native, "native_audit", "raw_policy_action_full12")), ("projected", native["projected_residual_full12"])):
                require(old.sequence(values, 12) and not any(values), "zero evaluation emitted residual")
                after.maxima_update(maxima, kind, values, tick)
            direct = [old.difference(get(native, "native_audit", "actual_native_targets", field, i),
                get(native, "native_audit", "counterfactual_native_targets", field, i))
                for field, n in (("servo_position_rad", 8), ("wheel_velocity_rad_s", 4)) for i in range(n)]
            require(old.sequence(direct, 12) and not any(direct), "zero policy has unexplained direct target effect")
            after.maxima_update(maxima, "direct_native_servo_rad_wheel_rad_s", direct, tick)
            current = next(stream, None)
        if tick in wanted:
            records[tick] = compact(old.flatten(physical, native, label, (line, native_line), calibration=calibration), physical, native, endpoints.get(tick))
    require(tick == end and current is None, "physical/native endpoint mismatch")
    for intervention in selected_interventions:
        channel, observed = intervention["channel"], intervention["source_observation_tick"]
        prefix = "FL_hip" if channel == "front_left_hip" else "RL_hip"
        owner = next(o for o in intervention["owners"] if o["channel"] == channel)
        native_nominal = get(records, observed+1, prefix + "_nominal_request_deg")
        intervention.update(next_native_post_step_tick=observed+1 if observed < end else None,
            annotated_candidate_source_deg=owner["candidate_source_target_deg"], next_native_nominal_deg=native_nominal,
            annotated_candidate_equals_next_dispatched_nominal=(None if native_nominal is None else abs(native_nominal-owner["candidate_source_target_deg"]) < 1e-8),
            authority_limit="An older source-owner candidate can be replaced by a later source owner; mismatch is reported, not called an actual applied reduction.")
    for event in events:
        t = event["physics_tick"]
        event.update(physical=records[t], next_post_step=records.get(t+1), height=choose_height(heights, t))
        if event["event"] == "P09_LATE_GROUP_RELEASE":
            before, following = records[t].get("post_step_nominal_full12"), get(records, t+1, "post_step_nominal_full12")
            event["actual_dispatch_audit"] = {"source_pre_step_tick": t, "native_post_step_tick": t+1 if t < end else None,
                "source_nominal_changed_indices": [i for i in range(12) if before[i] != following[i]] if old.sequence(before, 12) and old.sequence(following, 12) else None,
                "native_source_line": get(records, t+1, "native_source_line"),
                "meaning": "actual same-stream source command difference across release; not a proof this group caused earlier RR C/P"}
    baseline_summary = load(ROOT / "outputs/ppo_timing_task_priority_v1/B_AFTER_FULL.summary.json")
    baseline_rows = load(Path(__file__).with_name("existing_A_B2_height_window.json"))["rows"]
    baseline_by_tick = {r["physics_tick"]: r for r in baseline_rows if r["run_label"] == "B2_ZERO"}
    b_events = owner_events(baseline_summary["source_owner_observations"], durations)
    b_events += rr_events(baseline_summary["RR_events"], {}) + [{"event": "TERMINAL", "physics_tick": baseline_summary["end_tick"]}]
    number_occurrences(b_events)
    comparison = []
    for event in events:
        counterpart = next((r for r in b_events if (r["event"], r["occurrence"]) == (event["event"], event["occurrence"])), None)
        baseline = baseline_by_tick.get(counterpart["physics_tick"]) if counterpart else None
        comparison.append({"event": event["event"], "occurrence": event["occurrence"], "new_tick": event["physics_tick"], "B2_tick": None if counterpart is None else counterpart["physics_tick"],
            "B2_exact_existing_window_row": compact_baseline(baseline), "reason": None if baseline else "no corresponding retained B2 event sample; not time-shifted or fabricated",
            "new_minus_B2_body_collider_min_m": None if baseline is None else old.difference(event["physical"]["recorded_body_collider_min_z_m"], baseline["body_collision_min_z_m"]),
            "new_minus_B2_FL_actual_deg": None if baseline is None else old.difference(event["physical"]["FL_hip_actual_deg"], baseline["legs"]["FL"]["hip_actual_deg"]),
            "new_minus_B2_RL_actual_deg": None if baseline is None else old.difference(event["physical"]["RL_hip_actual_deg"], baseline["legs"]["RL"]["hip_actual_deg"])})
    return {"schema": "wlr50_clean.zero_height_event_aggregate.v1", "label": label, "source": str(source),
        "source_head": final["runtime_contract"]["source_git_commit"], "source_runtime_sha256": final["runtime_contract"]["runtime_content_sha256"],
        "terminal": {"tick": end, "duration_s": end/120, "phase": records[end]["source_phase_id"],
            "task_success": final.get("physical_task_success"), "evaluation_termination_reason": evaluation.get("termination_reason"),
            "source_acceptance_error": final.get("source_acceptance_error"), "RR": evaluation.get("current_legs", {}).get("RR"),
            "RR_history": {key: get(evaluation, "history", key, "RR") for key in ("active_lift", "front_edge_crossed", "placed")}},
        "events": events, "height_reductions_at_actual_observation_clocks": selected_interventions,
        "B2_event_comparison": comparison, "initial_pair": old.compare_initial(baseline_summary["initial_state"], initial),
        "source_durations_local_ticks": durations, "source_activations": activations, "stage_transitions": changes,
        "zero_residual_full_stream_maxima": maxima, "exact_native_physical_join_ticks": end,
        "height_startup_path": str(source / "height_diagnostics_startup.json"), "diagnostic_receipt": final["height_diagnostics"],
        "limits": ["No simulation or success relabeling; actual completed run only.", "No same-wall-clock/event outcome equivalence assumed.",
            "Exact event physical/q rows use full120Hz native/physical join. Height is exact only where sampled; neighbors retain clocks.",
            "FL/RL source reduction annotation is not an actual-q reduction. Event-matched actual-q delta vs B2 is not causal attribution.",
            "Owner endpoint is first observed completed local clock, not exact onset. Owner observations and next native post-step remain separate.",
            "B2 RR upper-link origin/URDF mount estimate is not a directly logged USD mount; no cross-type height equality claimed.",
            "Missing semantic endpoint at partial safety termination stays unknown; independent final evaluator retained."]}


def self_test():
    owners = [{"stage": "P07", "observation_tick": 8, "source_ticks": 1, "actual_start_tick": 8},
              {"stage": "P07", "observation_tick": 24, "source_ticks": 201, "actual_start_tick": 8}]
    events = owner_events(owners, {"P07": 200, "P08": 48})
    assert [e["event"] for e in events] == ["P07_SOURCE_START", "P07_ENDPOINT_FIRST_OBSERVED", "P07_LAST_OWNER_OBSERVATION"]
    assert events[1]["previous_owner_observation_tick"] == 8
    selected = choose_height({0: {"physics_tick": 0}, 8: {"physics_tick": 8}}, 3)
    assert selected["exact"] is None and selected["before"]["physics_tick"] == 0 and selected["after"]["physics_tick"] == 8
    assert choose_height({8: {"physics_tick": 8}}, 8)["before"] is None
    assert rr_events([], {"history": {"event_ticks": {"placed": {"RR": 29}}}})[0]["physics_tick"] == 29
    assert [e["occurrence"] for e in number_occurrences([{"event": "RR_I"}, {"event": "RR_Q"}, {"event": "RR_I"}])] == [1, 1, 2]
    late = late_group_events({16: {"line": 2, "observation_tick": 16, "task": {"nominal_provider_diagnostics": {
        "source_partial_order": {"layers": [{"stage": "P09", "observation_tick": 15, "late_group_start_tick": 11}]}}}}})
    assert late[0]["physics_tick"] == 11 and late[0]["owner"]["decision_end_tick"] == 16
    print(json.dumps({"self_test": "PASS", "synthetic_only": True, "runs_read": 0, "production_modified": False}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--label")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        require(args.run and args.label and args.output and not args.output.exists(), "run,label,new output required")
        result = aggregate(args.run, args.label)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, allow_nan=False, indent=2)
        print(json.dumps({"output": str(args.output.resolve()), "terminal": {k:v for k,v in result["terminal"].items() if k != "RR"}}))
