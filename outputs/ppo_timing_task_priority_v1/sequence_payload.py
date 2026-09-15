"""Compact evidence rows for a separate spreadsheet author; JSON only.

Read completed summaries and a few existing source streams. No simulation,
inferred onset, waypoint expansion, forward-filled load, or source mutation.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("after_evidence", HERE / "after_run_extract.py")
extractor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extractor)
old, get = extractor.old, extractor.get
ORDER = ("FL_hip", "FR_knee", "RL_hip", "RR_hip", "RR_knee")
INDICES = (0, 3, 4, 6, 7)


def row(label, action, source, line=None, *, pre=None, post=None, event=None):
    return {"run_label": label, "action_identity": action, "source_event": event,
        "command_pre_step_tick": pre, "command_post_step_tick": post,
        "command_time_s": pre / 120. if pre is not None else None,
        "command_post_step_time_s": post / 120. if post is not None else None,
        "nominal_selected_joint_deg": None, "nominal_wheel_rad_s": None,
        "measured_response_tick": None, "measured_response_status": "exact physical response onset not established",
        "available_response_sample_tick": None, "available_response_sample": None,
        "entry_observation_tick": None, "entry_physical_state": None,
        "successor_start_tick": None, "continuation_condition_or_wait_reason": None,
        "source_path": str(source), "source_line": line, "evidence_pointer": None}


def before_state(value):
    if value is None:
        return None
    return {"physics_tick": value["tick"], "physical_source_line": value["line"],
        "body_origin_world_m": value["body_origin_w_m"], "com_world_m": value["CoM_w_m"],
        "com_valid": value["CoM_valid"],
        "actual_selected_joint_deg": [value["legs"][name[:2]]["actual_" + name[3:] + "_deg"] for name in ORDER],
        "legs_contact": [value["legs"][leg]["contact"] for leg in old.LEGS],
        "legs_ground_force_n": [value["legs"][leg]["ground_normal_force_n"] for leg in old.LEGS],
        "legs_obstacle_force_n": [value["legs"][leg]["obstacle_normal_force_n"] for leg in old.LEGS],
        "legs_ground_pair_verified": [value["legs"][leg]["ground_pair_verified"] for leg in old.LEGS],
        "legs_obstacle_pair_verified": [value["legs"][leg]["obstacle_pair_verified"] for leg in old.LEGS],
        "RR_wheel_center_world_m": value["legs"]["RR"]["wheel_center_w_m"],
        "RR_clearance_m": value["legs"]["RR"]["wheel_bottom_clearance_above_top_m"]}


def flat_state(value):
    if value is None:
        return None
    return {key: value.get(key) for key in ("episode_physics_tick", "source_lines", "semantic_endpoint_observation_tick",
        "calibrated_body_rpy_rad", "com_position_world_m", "com_valid", "RR_front_distance_derived_m",
        "RR_top_clearance_derived_m", "legs_contact_surface", "legs_top_contact", "legs_support",
        "legs_bearing_force_n", "legs_bearing_verified", "legs_load_fraction", "legs_load_fraction_valid",
        "RR_I_initial_lift_observed", "RR_Q_lift_established", "RR_current_lift_valid", "RR_C_front_edge_crossed", "RR_P_placed",
        *[name + "_actual_deg" for name in ORDER])}


def attach_samples(target, samples, tick_key, convert, physical_path):
    pre, post = target["command_pre_step_tick"], target["command_post_step_tick"]
    before = next((r for r in reversed(samples) if pre is not None and r[tick_key] <= pre), None)
    after = next((r for r in samples if post is not None and r[tick_key] >= post), None)
    target.update(entry_observation_tick=before[tick_key] if before else None, entry_physical_state=convert(before),
        available_response_sample_tick=after[tick_key] if after else None, available_response_sample=convert(after),
        physical_source_path=str(physical_path),
        sample_limit="Explicit nearest retained pre/after samples; not onset or same-tick fill. Body motion is not leg loading.")


def top_continuation_row(source, label, first, end_tick):
    """One real wheel restart inside the first recorded TOP-eligible decision."""
    line, decision, diagnostic = first
    restart, previous = None, None
    for native_line, native in old.lines(source / "native_tick_audit.jsonl"):
        tick, wheels = native["episode_physics_tick"], native["nominal_full12"][8:]
        if (decision["start_tick"] < tick <= decision["end_tick"] and previous == [0.]*4
                and all(v > 0 for v in wheels)):
            restart = native_line, native
            break
        previous = wheels
        if tick > decision["end_tick"]:
            break
    post = restart[1]["episode_physics_tick"] if restart else None
    r = row(label, "first_recorded_TOP_continuation", source / "native_tick_audit.jsonl" if restart else source / "video_policy_decisions.jsonl",
        restart[0] if restart else line, pre=post-1 if post else None, post=post,
        event="Native wheel restart within first decision reporting current_TOP_corner_continuation")
    r.update(TOP_diagnostic_decision_line=line, TOP_diagnostic_reported_at_decision_end_tick=decision["end_tick"],
        TOP_diagnostic=diagnostic, TOP_classification_at_command_tick="not separately recorded; do not move endpoint classification backward",
        continuation_condition_or_wait_reason="Measured TOP continuation after finite owner; advance is not C/P success")
    if restart:
        r.update(nominal_wheel_rad_s=restart[1]["nominal_full12"][8:],
            actual_native_wheel_target_rad_s=get(restart[1], "native_audit", "actual_native_targets", "wheel_velocity_rad_s"))
        physical = {}
        for physical_line, value in old.lines(source / "physical_observations.jsonl"):
            if value["physics_tick"] not in (post-1, post, end_tick):
                continue
            rr = value["wheels"][old.WHEELS[3]]
            physical[value["physics_tick"]] = {"physical_source_line": physical_line,
                "measured_wheel_velocity_rad_s": [value["wheels"][name]["velocity_rad_s"] for name in old.WHEELS],
                "RR_front_distance_m": rr["center_w_m"][0]-value["obstacle"]["front_x_m"],
                "RR_clearance_m": rr["bottom_w_m"][2]-value["obstacle"]["top_z_m"]}
        r.update(entry_observation_tick=post-1, entry_physical_state=physical[post-1],
            available_response_sample_tick=post, available_response_sample=physical[post],
            terminal_physical_sample_tick=end_tick, terminal_physical_sample=physical[end_tick],
            RR_front_advance_to_terminal_m=physical[end_tick]["RR_front_distance_m"]-physical[post-1]["RR_front_distance_m"],
            physical_source_path=str(source / "physical_observations.jsonl"))
    return r


def before_rows(evidence_path):
    evidence = extractor.load(evidence_path)
    result, seen = [], set()
    run = Path(evidence["A_run"])
    for index, event in enumerate(evidence["source_event_sequence"]):
        part = event["part"]
        if part in seen or part == "FL_followthrough_hold":
            continue
        seen.add(part)
        r = row("A_FSM", part, run / "full12_commands_120hz.jsonl", event["line"],
            pre=event["tick"], post=event["tick"]+1, event=event.get("source_event"))
        r.update(nominal_selected_joint_deg=[event["N"][i] for i in INDICES], nominal_wheel_rad_s=event["N"][8:],
            evidence_pointer=f"{evidence_path}#/source_event_sequence/{index}")
        attach_samples(r, evidence["A_physical_before_key_dispatches"], "tick", before_state, run / "observation_120hz.jsonl")
        if part == "explicit_all_four_wheel_stop":
            r["continuation_condition_or_wait_reason"] = "Authored four-wheel stop, not an ordinary phase-transition reset"
        result.append(r)
    for index, life in enumerate(evidence["A_lifecycle"]):
        endpoint_line = successor_line = None
        for line, transition in old.lines(run / "state_transitions.jsonl"):
            if transition.get("state_id") != life["phase"]:
                continue
            if abs(transition["sim_time_s"]-life["endpoint_s"]) < 1e-8 and transition["to_lifecycle"] == "VERIFY_RESULT":
                endpoint_line = line
            if abs(transition["sim_time_s"]-life["done_s"]) < 1e-8 and transition["to_lifecycle"] == "DONE":
                successor_line = line
        r = row("A_FSM", life["phase"] + "_legacy_endpoint_wait", run / "state_transitions.jsonl", endpoint_line)
        r.update(evidence_pointer=f"{evidence_path}#/A_lifecycle/{index}",
            endpoint_time_s=life["endpoint_s"], successor_time_s=life["done_s"],
            successor_source_line=successor_line,
            continuation_condition_or_wait_reason=life["guard"] + "; historical gate evidence only, MUST NOT restore as PPO gate")
        result.append(r)
    air_sample = next((r for r in evidence["A_physical_before_key_dispatches"] if r["legs"]["RR"]["contact"] == "AIR"), None)
    if air_sample:
        r = row("A_FSM", "RR_initial_AIR_before_own_swing_sample", run / "observation_120hz.jsonl", air_sample["line"])
        r.update(available_response_sample_tick=air_sample["tick"], available_response_sample=before_state(air_sample),
            measured_response_status="I onset unknown; earliest AIR among already selected A samples, before RR own command effect")
        result.append(r)
    for line, event in old.lines(run / "leg_crossing_events.jsonl"):
        if event.get("leg") == "RR":
            r = row("A_FSM", "RR_" + event["event"], run / "leg_crossing_events.jsonl", line, event=event["event"])
            r.update(measured_response_tick=event["physics_tick"], measured_response_status="Original A latch event, not identical to new I/Q/current-valid semantics",
                measured_event_time_s=event["physics_tick"]/120., continuation_condition_or_wait_reason=get(event, "evidence", "source"))
            result.append(r)
    names = {5161: "FL_open_space", 5169: "RL_transfer", 5177: "RR_swing_start",
             5201: "FR_knee_and_wheel_atomic", 5321: "authored_four_wheel_stop_not_fully_owned"}
    for index, event in enumerate(evidence["B0_selected_nominal_events"]):
        if event["tick"] not in names:
            continue
        source = Path(evidence["B0_source"])
        r = row("B0_BEFORE", names[event["tick"]], source / "native_tick_audit.jsonl", event["line"],
            pre=event["tick"]-1, post=event["tick"], event="Recorded nominal dispatch; source order from audited before evidence")
        r.update(nominal_selected_joint_deg=[event["N"][i] for i in INDICES], nominal_wheel_rad_s=event["N"][8:],
            evidence_pointer=f"{evidence_path}#/B0_selected_nominal_events/{index}")
        attach_samples(r, evidence["B0_physical_selected"], "tick", before_state, source / "physical_observations.jsonl")
        if event["tick"] == 5321:
            r["continuation_condition_or_wait_reason"] = "FR wheel zero, other wheels nonzero: source explicit four-wheel stop lost ownership"
        result.append(r)
    return result


def completed_rows(summary_path, *, label_override=None, actions=True):
    summary = extractor.load(summary_path)
    source, manifest = old.completed_source(summary["source"])
    extractor.require(summary["exact_full_stream_join_complete"] is True and
        summary["source_git_commit"] == manifest["runtime_contract"]["source_git_commit"] and
        summary["runtime_content_sha256"] == manifest["runtime_contract"]["runtime_content_sha256"], "Summary/source binding differs")
    label = label_override or summary["run_label"]
    records_path = Path(str(summary_path).replace(".summary.json", ".records.json"))
    samples = extractor.load(records_path)
    extractor.require(samples[-1]["episode_physics_tick"] == summary["end_tick"], "Sample terminal mismatch")
    result, found = [], {}
    wanted = {value["native_post_step_tick"]: key for key,value in summary["first_rear_sequence_nominal_dispatch_changes"].items() if value}
    if actions and "P07" in summary["native_source_phase_tick_counts"]:
        previous = None
        scan_end = max([*wanted, *[e["actual_source_tick"] for e in summary["source_activation_events"]]], default=summary["end_tick"]) + 256
        for line, native in old.lines(source / "native_tick_audit.jsonl"):
            tick, nominal = native["episode_physics_tick"], native["nominal_full12"]
            action = wanted.get(tick)
            if ("FL_P07" not in found and native["source_phase_id"] == "P07" and previous is not None
                    and abs(nominal[0] - previous[0]) > 1e-9):
                action = "FL_P07"
            if "FR_P07" in found and "four_wheel_stop_after_FR" not in found and nominal[8:] == [0.]*4 and previous[8:] != [0.]*4:
                action = "four_wheel_stop_after_FR"
            if action:
                r = row(label, action, source / "native_tick_audit.jsonl", line, pre=tick-1, post=tick,
                        event="Actual native nominal dispatch; no action onset inferred from semantic stage alone")
                r.update(nominal_selected_joint_deg=[nominal[i] for i in INDICES], nominal_wheel_rad_s=nominal[8:])
                attach_samples(r, samples, "episode_physics_tick", flat_state, source / "physical_observations.jsonl")
                found[action] = r; result.append(r)
            previous = nominal
            if len(found) == 5 or tick > scan_end:
                break
    for event in summary["source_activation_events"]:
        r = row(label, event["stage"] + "_" + event["event"], source / "video_policy_decisions.jsonl", event["decision_line"], event=event["event"])
        r.update(successor_start_tick=event["actual_source_tick"], owner_observation_tick=event["owner_observation_tick"],
                 decision_end_tick=event["first_observed_in_decision_end_tick"])
        result.append(r)
    waits = {}
    for owner in summary["source_owner_observations"]:
        if not owner["wait_reason"]:
            continue
        key = (owner["stage"], owner["status"], owner["wait_reason"])
        if key not in waits:
            r = row(label, owner["stage"] + "_" + owner["status"], source / "video_policy_decisions.jsonl", owner["decision_line"])
            r.update(owner_observation_tick=owner["observation_tick"], owner_source_ticks=owner["source_ticks"],
                continuation_condition_or_wait_reason=owner["wait_reason"], last_observed_wait_tick=owner["observation_tick"],
                observed_decision_snapshots=0)
            waits[key] = r; result.append(r)
        waits[key]["last_observed_wait_tick"] = owner["observation_tick"]
        waits[key]["observed_decision_snapshots"] += 1
    seen = set()
    for line, transition in old.lines(source / "stage_transition_evidence.jsonl"):
        history = transition.get("physical_history") or {}
        events = [e for e in history.get("lift_attempt_events", []) if e.get("leg") == "RR"]
        events += [{"event": name, "physics_tick": tick} for name, values in history.get("event_ticks", {}).items()
                   if name in ("front_edge_crossed", "placed") and (tick := values.get("RR")) is not None]
        for event in events:
            key = event["event"], event["physics_tick"]
            if key in seen:
                continue
            seen.add(key)
            r = row(label, "RR_" + key[0], source / "stage_transition_evidence.jsonl", line, event=key[0])
            r.update(measured_response_tick=key[1], measured_event_time_s=key[1]/120.,
                     measured_response_status="Recorded evaluator event at actual physical tick")
            result.append(r)
    last_line, last, first_top = None, None, None
    for last_line, last in old.lines(source / "video_policy_decisions.jsonl"):
        diag = get(last, "step_info", "semantic_task", "nominal_provider_diagnostics", "rr_carry_continuation") or {}
        if first_top is None and diag.get("added_rolling_suggestion") is True and diag.get("current_TOP_continuation_eligible") is True:
            first_top = last_line, last, diag
    if first_top:
        result.append(top_continuation_row(source, label, first_top, summary["end_tick"]))
    task = get(last, "step_info", "semantic_task") or {}
    final_ev = get(extractor.load(source / "semantic_video_source_manifest.json"), "physical_episode", "physical_task_evaluation") or {}
    extractor.require(final_ev.get("physics_tick") == summary["end_tick"], "Final evaluator tick mismatch")
    r = row(label, "RR_sequence_terminal_or_not_reached", source / "video_policy_decisions.jsonl", last_line)
    r.update(terminal_tick=summary["end_tick"], terminal_phase=get(last, "step_info", "end_phase_id") or last["request_phase"],
        available_response_sample_tick=summary["end_tick"], available_response_sample=flat_state(samples[-1]),
        continuation_condition_or_wait_reason=get(last, "step_info", "termination_reason") or summary["physical_evaluator_termination_reason"],
        completion_values=task.get("completion_values"), nominal_rr_carry=get(task, "nominal_provider_diagnostics", "rr_carry_continuation"),
        physical_source_path=str(source / "physical_observations.jsonl"), summary_source=str(summary_path),
        physical_task_success=summary["physical_task_success"],
        independent_final_evaluator={"physics_tick": final_ev["physics_tick"],
            "source": str(source / "semantic_video_source_manifest.json") + "#/physical_episode/physical_task_evaluation",
            "RR_current_lift_valid": get(final_ev, "current_legs", "RR", "current_lift_valid"),
            "RR_C": get(final_ev, "history", "front_edge_crossed", "RR"),
            "RR_P": get(final_ev, "history", "placed", "RR"),
            "termination_reason": final_ev.get("termination_reason"), "termination_source": final_ev.get("termination_source")})
    result.append(r)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-evidence", type=Path, default=HERE / "rear_sequence_before_source_evidence.json")
    parser.add_argument("--before-summary", type=Path, default=HERE / "old_b0_extractor_fixture_v2.summary.json")
    parser.add_argument("--run-summary", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    extractor.require(not args.output.exists(), "Sequence output exists; no overwrite")
    rows = before_rows(args.before_evidence.resolve())
    rows += completed_rows(args.before_summary.resolve(), label_override="B0_BEFORE", actions=False)
    for path in args.run_summary:
        rows += completed_rows(path.resolve())
    payload = {"schema": "wlr50_clean.compact_rear_sequence_rows.v1", "rows": rows,
        "selected_joint_order": list(ORDER), "leg_and_wheel_order": list(old.LEGS),
        "command_time_definition": "command_time_s=pre-step tick/120; native B/C rows are post-step, A commands are pre-step",
        "limits": ["Event-aligned table; not retiming videos", "Sparse response samples are not exact onset or causal attribution",
            "Missing measurement remains null; contact_surface TOP is not necessarily top_contact/placement",
            "A historical latch/guard semantics remain labeled; no historical posture gate imported",
            "No full trajectory copying or new simulation; JSON rows for separate spreadsheet author"]}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
    print(json.dumps({"output": str(args.output.resolve()), "rows": len(rows), "runs": sorted({r["run_label"] for r in rows})}))


if __name__ == "__main__":
    main()
