"""Read-only, standard-library analysis of one explicitly supplied decision log.

No simulator/model/project imports. --live takes a fixed byte-size snapshot and
ignores only an unfinished final line. A malformed *complete* line always fails.
Diagnostic intent is not an executed target, policy sample, or PPO training.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE = ROOT / (
    "runs/ppo_rr_rl_timing_policy_learning_v1/video_eval/validation/"
    "20260924T0214316632402Z_g49eb23163a6e_530cc3c96d834dc3ba84b0b19d8db4f1/"
    "source/video_policy_decisions.jsonl")
EVENTS = {"FRplaced": ("placed", "FR"), "FLplaced": ("placed", "FL"),
          "RRqualified": ("active_lift", "RR"),
          "RRcross": ("front_edge_crossed", "RR"), "RRplaced": ("placed", "RR")}
EVENT_ORDER = ("FRplaced", "FLplaced", "RRqualified", "RRcross", "RRfirstTOP", "RRplaced")
AFTER_SECONDS = (0., .5, 1., 1.5, 2., 3., 5., 10., 20., 30., 60.)


def mapping(value):
    return value if isinstance(value, dict) else {}


def vector(value, count=12):
    return (isinstance(value, list) and len(value) == count
            and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                    and math.isfinite(x) for x in value))


def pair(value):
    return value[6:8] if vector(value) else None


def delta(a, b):
    return max(abs(x-y) for x, y in zip(a, b)) if vector(a) and vector(b) else None


def decisions(path, live, status):
    """Read only bytes present at entry, never race newly appended partial JSON."""
    with path.open("rb") as stream:
        # Windows directory metadata may lag a live writer. Query this open
        # handle's EOF, then keep that byte boundary fixed for this read.
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(0)
        status.update(path=str(path.resolve()), snapshot_bytes=size,
                      snapshot_size_source="open_handle_seek_END_tell",
                      complete_rows=0, ignored_incomplete_tail_bytes=0)
        remaining = size
        while remaining:
            line = stream.readline(remaining)
            remaining -= len(line)
            if not line.endswith(b"\n"):
                if live:
                    status["ignored_incomplete_tail_bytes"] = len(line)
                    break
                raise ValueError(f"{path}: unfinished final line; use --live")
            try:
                row = json.loads(line)
            except (ValueError, UnicodeError) as exc:
                raise ValueError(f"{path}: malformed complete line {status['complete_rows']+1}") from exc
            if not isinstance(row, dict) or not isinstance(row.get("step_info"), dict):
                raise ValueError(f"{path}: not a video policy decision row")
            status["complete_rows"] += 1
            yield row


def sample(row):
    info = row["step_info"]
    task = mapping(info.get("semantic_task"))
    ev = mapping(task.get("physical_evaluator"))
    legs = mapping(ev.get("current_legs"))
    rr = mapping(legs.get("RR"))
    audit = mapping(info.get("actuator_target_effect_audit"))
    headroom = mapping(audit.get("policy_headroom_evidence"))
    local = mapping(info.get("rr_capture_local"))
    metrics = mapping(local.get("metrics"))
    policy = mapping(row.get("policy_request"))
    diagnostic = mapping(policy.get("independent_diagnostic"))
    intent = mapping(diagnostic.get("candidate_absolute_targets"))
    final = info.get("actual_drive_target_full12")
    roles = mapping(ev.get("transfer_roles"))
    direction = mapping(mapping(roles.get("RR")).get("transfer_direction_context"))
    goal = mapping(ev.get("goal_features"))
    wheel_order = ev.get("stop_progress_wheel_order", [])
    wheel_measured = ev.get("measured_wheel_velocity_rad_s", [])
    fl_actual = None
    if "front_left_ankle" in wheel_order and vector(wheel_measured, len(wheel_order)):
        fl_actual = wheel_measured[wheel_order.index("front_left_ankle")]
    leg_fields = ("support", "ground_contact", "top_contact", "top_surface_contact",
                  "contact_surface", "bearing_verified", "bearing_force_n",
                  "load_fraction", "load_fraction_valid")
    return dict(
        tick=ev.get("physics_tick", row.get("end_tick")), time_s=ev.get("simulation_time_s"),
        decision=row.get("decision"), start_tick=row.get("start_tick"), phase=info.get("end_phase_id"),
        request_active=policy.get("capture_active"), end_active=local.get("active"),
        entry_final_rr_deg=[local.get("entry_rr_hip_deg"), local.get("entry_rr_knee_deg")],
        activation_tick=local.get("activation_tick"), activation_time_s=local.get("activation_time_s"),
        rr=dict(
            diagnostic_candidate_rr_deg=[intent.get("6"), intent.get("7")] if intent else None,
            diagnostic_holds_first_contact_final=diagnostic.get("contact_holds_actual_FINAL_no_further_angle_ramp"),
            diagnostic_previous_ack_baseline_rr_deg=pair(diagnostic.get("baseline_from_previous_committed_ack")),
            prior_raw_mean_rr=pair(policy.get("prior_raw_mean_full12")),
            prior_raw_mean_full12=policy.get("prior_raw_mean_full12"),
            local_raw_mean_delta_rr=pair(policy.get("local_raw_mean_delta_full12")),
            applied_local_raw_delta_full12=policy.get("applied_local_raw_mean_delta_full12"),
            prior_history_conditional_rr=pair(policy.get("prior_same_observation_conditional_mean_full12")),
            selected_policy_raw_rr=pair(policy.get("selected_raw_full12")),
            actually_issued_raw_rr=pair(info.get("raw_policy_action_full12")),
            actually_issued_raw_full12=info.get("raw_policy_action_full12"),
            mapped_N_rr_deg=pair(headroom.get("geometry_corrected_native_full12")),
            bounded_controller_rr_deg=pair(headroom.get("bounded_controller_bias_full12")),
            same_tick_N_plus_controller_rr_deg=pair(headroom.get("baseline_native_plus_controller_full12")),
            request_residual_rr_deg=pair(headroom.get("requested_policy_residual_full12")),
            effective_residual_rr_deg=pair(headroom.get("effective_policy_residual_full12")),
            candidate_before_final_slew_rr_deg=pair(headroom.get("candidate_native_target_before_final_slew_full12")),
            final_rr_deg=pair(final), actual_rr_deg=metrics.get("actual_rr_hip_knee_deg"),
            gap_m=rr.get("clearance_m"), within_top_xy=rr.get("within_top_xy"),
            within_lateral_span=rr.get("within_lateral_span"),
            xy_outside_distance_m=rr.get("top_xy_outside_distance_m"),
            front_distance_m=rr.get("front_distance_m"), force_n=rr.get("bearing_force_n"),
            current_lift_valid=rr.get("current_lift_valid"), air=rr.get("air"),
            top_contact=rr.get("top_contact"), top_surface_contact=rr.get("top_surface_contact"),
            consecutive_top_samples=rr.get("consecutive_top_samples"),
            current_top_bearing=metrics.get("current_top_bearing"),
            hold_elapsed_s=local.get("hold_elapsed_s"), local_success=local.get("local_success")),
        clock=dict(observation_tick=ev.get("physics_tick"),
                   rr_actual_observation_tick=metrics.get("tick"),
                   native_dispatch_tick=audit.get("physics_tick"),
                   dispatch_source_observation_tick=mapping(mapping(audit.get("capture_assist_evidence")).get("context")).get("source_observation_tick"),
                   same_tick_counterfactual=audit.get("same_tick_counterfactual"), audit_verified=audit.get("verified")),
        support={leg: {key: mapping(legs.get(leg)).get(key) for key in leg_fields}
                 for leg in ("FL", "FR", "RL", "RR")},
        fl_wheel=dict(canonical_final_command_rad_s=final[8] if vector(final) else None,
                      measured_canonical_velocity_rad_s=fl_actual),
        body_com=dict(body_forward_m=goal.get("body_forward_m"),
                      body_linear_speed_m_s=goal.get("body_linear_speed_m_s"),
                      body_angular_speed_rad_s=goal.get("body_angular_speed_rad_s"),
                      body_angular_velocity_w_rad_s=direction.get("body_angular_velocity_w_rad_s"),
                      displacement_reference_tick=direction.get("reference_tick"),
                      body_world_displacement_m=direction.get("body_world_displacement_m"),
                      com_world_displacement_m=direction.get("com_world_displacement_m"),
                      com_position_w_m=direction.get("mass_weighted_com_position_w_m"),
                      com_velocity_w_m_s=direction.get("mass_weighted_com_velocity_w_m_s"),
                      body_bounds=mapping(ev.get("body_traversal_geometry"))),
        termination_reason=info.get("termination_reason"),
        full_task_success=info.get("full_task_success"),
        local_task_success=info.get("local_task_success"))


def analyze(path, *, live=False, reference=None):
    status, events, event_samples, after_samples, index = {}, {}, {}, {}, {}
    prefix = dict(checked_inactive_requests=0, all_local_exact_zero=True,
                  all_local_mean_exact_zero=True, all_combined_mean_equals_prior=True,
                  all_history_kernel_once=True,
                  all_local_forward_count_zero=True, all_issued_equal_prior_history_mean=True,
                  no_diagnostic_override=True, all_prefix_excluded_from_PPO=True,
                  unknown_request_gate_rows=0, bad_decisions=[], reference_compared_rows=0,
                  reference_missing_rows=0, reference_raw_max_abs_delta=0., reference_final_max_abs_delta=0.)
    entry = diagnostic_entry = last = previous = minimum_gap = None
    first_top_previous_tick = None
    diagnostic_count = 0
    for row in decisions(path, live, status):
        current = sample(row)
        tick = current["tick"]
        if not isinstance(tick, int) or not isinstance(current["time_s"], (float, int)):
            raise ValueError("decision lacks physical evaluator tick/time")
        if last is not None and tick <= last["tick"]:
            raise ValueError("non-increasing decision observation ticks")
        info, policy = row["step_info"], mapping(row.get("policy_request"))
        history = mapping(mapping(info.get("semantic_task")).get("physical_evaluator")).get("history", {})
        event_ticks = mapping(mapping(history).get("event_ticks"))
        # Event tick clocks are native recorded evaluator ticks, not stage dates.
        clock_origin = current["time_s"] - tick/120.
        for label, (kind, leg) in EVENTS.items():
            exact_tick = mapping(event_ticks.get(kind)).get(leg)
            if label not in events and isinstance(exact_tick, int):
                events[label] = dict(tick=exact_tick, time_s=clock_origin+exact_tick/120.,
                                     timing="recorded_history_event_tick_at_120Hz", observed_in_row_tick=tick)
                event_samples[label] = current
        if "RRfirstTOP" not in events and current["rr"]["top_contact"] is True and current["rr"]["top_surface_contact"] is True:
            events["RRfirstTOP"] = dict(tick=tick, time_s=current["time_s"],
                timing="first_TOP_visible_in_decision_rows_not_guaranteed_first_native_touch",
                previous_sample_tick=first_top_previous_tick,
                current_consecutive_top_samples=current["rr"]["consecutive_top_samples"])
            event_samples["RRfirstTOP"] = current
        first_top_previous_tick = tick
        raw, final = info.get("raw_policy_action_full12"), info.get("actual_drive_target_full12")
        index[(row.get("start_tick"), tick)] = (raw, final)
        active = policy.get("capture_active")
        if active is False:
            prefix["checked_inactive_requests"] += 1
            zero = policy.get("applied_local_raw_mean_delta_full12")
            mean_zero = policy.get("local_raw_mean_delta_full12")
            exact_zero = vector(zero) and all(x == 0. for x in zero)
            checks = dict(all_local_exact_zero=exact_zero,
                all_local_mean_exact_zero=vector(mean_zero) and all(x == 0. for x in mean_zero),
                all_combined_mean_equals_prior=delta(policy.get("combined_raw_mean_full12"), policy.get("prior_raw_mean_full12")) == 0.,
                all_history_kernel_once=policy.get("history_kernel_applications") == 1,
                all_local_forward_count_zero=policy.get("local_network_forwards") == 0,
                all_issued_equal_prior_history_mean=delta(raw, policy.get("prior_same_observation_conditional_mean_full12")) == 0.,
                no_diagnostic_override="independent_diagnostic" not in policy,
                all_prefix_excluded_from_PPO=policy.get("prefix_excluded_from_new_PPO_credit") is True)
            for name, passed in checks.items():
                prefix[name] = prefix[name] and passed
            if not all(checks.values()) and len(prefix["bad_decisions"]) < 10:
                prefix["bad_decisions"].append(dict(decision=row.get("decision"), checks=checks))
            if reference is not None:
                other = reference.get((row.get("start_tick"), tick))
                dr, df = (delta(raw, other[0]), delta(final, other[1])) if other else (None, None)
                if dr is None or df is None:
                    prefix["reference_missing_rows"] += 1
                else:
                    prefix["reference_compared_rows"] += 1
                    prefix["reference_raw_max_abs_delta"] = max(prefix["reference_raw_max_abs_delta"], dr)
                    prefix["reference_final_max_abs_delta"] = max(prefix["reference_final_max_abs_delta"], df)
        elif active is not True:
            prefix["unknown_request_gate_rows"] += 1
        if current["end_active"] is True:
            if entry is None:
                entry = current
            actual_entry = entry["rr"]["actual_rr_deg"]
            actual_now = current["rr"]["actual_rr_deg"]
            current["rr"]["actual_delta_from_first_active_decision_deg"] = (
                [a-b for a, b in zip(actual_now, actual_entry)]
                if vector(actual_now, 2) and vector(actual_entry, 2) else None)
            current["rr"]["final_delta_from_gate_entry_final_deg"] = (
                [a-b for a, b in zip(current["rr"]["final_rr_deg"], current["entry_final_rr_deg"])]
                if vector(current["rr"]["final_rr_deg"], 2) and vector(current["entry_final_rr_deg"], 2) else None)
            gap = current["rr"]["gap_m"]
            if isinstance(gap, (int, float)) and math.isfinite(gap) and (
                    minimum_gap is None or gap < minimum_gap["rr"]["gap_m"]):
                minimum_gap = current
            age = current["time_s"] - current["activation_time_s"]
            for offset in AFTER_SECONDS:
                label = f"activation_plus_{offset:g}s"
                if age >= offset and label not in after_samples:
                    after_samples[label] = current
        if "independent_diagnostic" in policy:
            diagnostic_count += 1
            if diagnostic_entry is None:
                diagnostic_entry = dict(first_request_start_tick=row.get("start_tick"),
                    actual_previous_committed_final_rr_deg=previous["rr"]["final_rr_deg"] if previous and previous["tick"] == row.get("start_tick") else None,
                    first_candidate_rr_deg=current["rr"]["diagnostic_candidate_rr_deg"], first_request_result=current)
        previous = last = current
    prefix["strict_zero_verified"] = bool(prefix["checked_inactive_requests"]) and prefix["unknown_request_gate_rows"] == 0 and all(
        prefix[key] for key in ("all_local_exact_zero", "all_local_mean_exact_zero",
            "all_combined_mean_equals_prior", "all_history_kernel_once", "all_local_forward_count_zero",
            "all_issued_equal_prior_history_mean", "no_diagnostic_override", "all_prefix_excluded_from_PPO"))
    return dict(source=status, events=events, event_samples=event_samples, prefix=prefix,
                gate_entry=entry, diagnostic_entry=diagnostic_entry, diagnostic_request_rows=diagnostic_count,
                after_active_samples=after_samples, after_active_minimum_sampled_gap=minimum_gap,
                latest=last), index


def report(source, reference, live):
    baseline, baseline_index = analyze(reference)
    current, _ = analyze(source, live=live, reference=baseline_index)
    alignment = {}
    for label in EVENT_ORDER:
        before, after = baseline["events"].get(label), current["events"].get(label)
        alignment[label] = dict(reference=before, new=after,
            delta_s=after["time_s"]-before["time_s"] if before and after else None)
    return dict(schema="wlr50_clean.rr_capture_actualmetrics_report.v1",
        diagnostic_is_training=False, PPO_credit_of_direction_diagnostic=0,
        semantics=["Only the two explicitly named decision files are read; no models or simulator imports.",
                   "REQUEST/effective/mapped baseline/FINAL are from the actual final-dispatch audit of each decision.",
                   "Diagnostic candidates use a previous committed baseline; they are not same-tick counterfactual FINAL.",
                   "Measurements are post-step evaluator observations; native dispatch clock is separately labelled.",
                   "RR actual angles use logged local metrics; missing values remain null, not target substitutions.",
                   "FL measured velocity is canonicalized, not a directly stored native-sign sensor quantity.",
                   "Sparse decision snapshots cannot prove uninterrupted hold or exclude unseen short contacts.",
                   "Minimum gap means the minimum after-active decision-row sample, not a claimed native-tick minimum.",
                   "Actual joint deltas use the first decision observation with active=True; FINAL deltas use the logged gate-entry FINAL.",
                   "Missing events in a live snapshot mean not yet observed, not final failure."],
        event_alignment=alignment, current=current,
        reference=dict(source=baseline["source"], events=baseline["events"],
                       event_samples=baseline["event_samples"], latest=baseline["latest"]))


def brief(result):
    current = result["current"]
    last = current["latest"]
    lines = ["Read-only actual metrics; this analysis adds no training or physical-success credit.",
             f"Complete rows={current['source']['complete_rows']}; partial tail ignored={current['source']['ignored_incomplete_tail_bytes']} bytes"]
    if last:
        lines.append(f"Latest tick={last['tick']} t={last['time_s']:.6f}s phase={last['phase']} active={last['end_active']}")
    prefix = current["prefix"]
    lines.append(f"Inactive strict zero={prefix['strict_zero_verified']} ({prefix['checked_inactive_requests']} requests); prior-reference raw/final max deltas={prefix['reference_raw_max_abs_delta']:.9g}/{prefix['reference_final_max_abs_delta']:.9g}; compared={prefix['reference_compared_rows']} missing={prefix['reference_missing_rows']}")
    for name, event in result["event_alignment"].items():
        stamp = lambda x: f"{x['time_s']:.6f}s/tick{x['tick']}" if x else "not observed"
        lines.append(f"{name}: original {stamp(event['reference'])}; new {stamp(event['new'])}; delta_s={event['delta_s']}")
    if last:
        rr = last["rr"]
        lines.append(f"RR entry={last['entry_final_rr_deg']} candidate={rr['diagnostic_candidate_rr_deg']} FINAL={rr['final_rr_deg']} actual={rr['actual_rr_deg']}")
        lines.append(f"RR gap={rr['gap_m']}m XY={rr['within_top_xy']} force={rr['force_n']}N TOP={rr['top_contact']} hold={rr['hold_elapsed_s']}s; FL wheel={last['fl_wheel']}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Exact new-route video_policy_decisions.jsonl path")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--live", action="store_true", help="Ignore only unfinished final line from initial byte snapshot")
    parser.add_argument("--json", action="store_true", help="Print full selected-sample report instead of brief summary")
    parser.add_argument("--output", type=Path, help="Optionally write full JSON report to a new file (never overwrite)")
    args = parser.parse_args()
    result = report(args.source, args.reference, args.live)
    serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(serialized + "\n")
    print(serialized if args.json else brief(result))


if __name__ == "__main__":
    main()
