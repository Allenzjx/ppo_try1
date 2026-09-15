"""Completed single-run JSON evidence, full exact native/physical stream join.

Reuses the prior coordinate/calibration/initial-state extractor. Records retain
tick0, every8th actual tick and terminal; semantic loads/flags are filled ONLY
at their exact recorded decision endpoint. No forward fill and no CSV output.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "outputs/ppo_rr_video_diagnosis_v1/rr_tracking_extract.py"
spec = importlib.util.spec_from_file_location("previous_tracking_extractor", OLD)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
get = old.get
BEFORE_HEAD = "4b2c038887c4109c639eea720f9e60de2c6d8d93"
LEG_FIELDS = ("contact_surface", "ground_contact", "top_contact", "support",
              "bearing_force_n", "bearing_verified", "load_fraction", "load_fraction_valid")
OWNER_FIELDS = ("stage", "source_ticks", "observation_tick", "status", "wait_reason", "wait_ticks",
                "actual_start_tick", "fr_group_start_tick", "support_count", "FL_space_measured",
                "RR_current_continuation", "FL_body_radial_contraction_m")


def require(test, message):
    if not test:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def decisions(source):
    endpoints, owner_rows, activations, skipped = {}, [], {}, []
    phase_counts, returned, issued = Counter(), 0, 0
    for line, row in old.lines(source / "video_policy_decisions.jsonl"):
        issued += 1
        phase_counts[row.get("request_phase")] += 1
        returned += row.get("environment_step_returned") is True
        info, task = row.get("step_info") or {}, get(row, "step_info", "semantic_task")
        ev = get(task, "physical_evaluator")
        if not isinstance(ev, dict):
            continue  # Native safety unwind may never return a semantic endpoint.
        end, observed = row.get("end_tick"), ev.get("physics_tick")
        if type(end) is int and type(observed) is int and end == observed:
            require(end not in endpoints, "duplicate semantic decision endpoint")
            endpoints[end] = {"line": line, "decision": row.get("decision"), "task": task,
                              "end_phase": info.get("end_phase_id"), "observation_tick": observed}
        else:
            skipped.append({"line": line, "decision_end_tick": end, "evaluator_observation_tick": observed})
        sequence = get(task, "nominal_provider_diagnostics", "source_partial_order") or {}
        dispatched_owners = []
        for layer in sequence.get("layers", []):
            compact = {key: layer.get(key) for key in OWNER_FIELDS}
            compact.update(decision=row.get("decision"), decision_end_tick=end, decision_line=line,
                           source_semantics=sequence.get("mode"))
            owner_rows.append(compact)
            dispatched_owners.append(compact)
            for field in ("actual_start_tick", "fr_group_start_tick"):
                actual = layer.get(field)
                if type(actual) is int:
                    key = (layer.get("stage"), field, actual)
                    activations.setdefault(key, {"stage": key[0], "event": field, "actual_source_tick": actual,
                        "first_observed_in_decision_end_tick": end, "owner_observation_tick": layer.get("observation_tick"),
                        "decision_line": line})
        if end in endpoints:
            endpoints[end]["dispatched_owners"] = dispatched_owners
    return endpoints, owner_rows, list(activations.values()), skipped, dict(phase_counts), issued, returned


def transitions(source):
    changes, rr_events = [], {}
    for line, row in old.lines(source / "stage_transition_evidence.jsonl"):
        tick = row.get("physics_tick")
        require(type(tick) is int, "transition is missing its actual physical tick")
        if row.get("from_stage") != row.get("to_stage"):
            changes.append({"physics_tick": tick, "from_stage": row.get("from_stage"),
                            "to_stage": row.get("to_stage"), "source_line": line})
        for event in get(row, "physical_history", "lift_attempt_events") or []:
            if event.get("leg") != "RR":
                continue
            key = (event.get("event"), event.get("physics_tick"))
            rr_events.setdefault(key, {key: event.get(key) for key in
                ("leg", "event", "physics_tick", "simulation_time_s", "reason", "upward_excursion_m", "clearance_gain_m")})
    return changes, sorted(rr_events.values(), key=lambda item: (item["physics_tick"], item["event"]))


def compact_record(flat, endpoint, owner_by_tick, native=None):
    tick = flat["episode_physics_tick"]
    task = endpoint["task"] if endpoint else None
    ev = get(task, "physical_evaluator")
    rr = get(ev, "current_legs", "RR")
    row = {key: flat[key] for key in ("run_label", "episode_physics_tick", "simulation_time_s",
        "source_phase_id", "native_internal_tick", "native_verified", "raw_residual_full12",
        "projected_residual_full12", "phase_mask_full12", "base_origin_height_world_m",
        "calibrated_body_rpy_rad", "body_angular_velocity_world_rad_s", "com_position_world_m",
        "com_velocity_world_m_s", "com_valid", "FR_front_distance_derived_m", "FR_top_clearance_derived_m",
        "RR_front_distance_derived_m", "RR_top_clearance_derived_m")}
    row.update(source_lines=[flat["physical_source_line"], flat["native_source_line"], endpoint["line"] if endpoint else None],
        semantic_endpoint_observation_tick=endpoint["observation_tick"] if endpoint else None,
        semantic_end_phase=endpoint["end_phase"] if endpoint else None, substage=get(task, "substage"))
    for leg in old.LEGS:
        for joint in ("hip", "knee"):
            for field in ("nominal_request_deg", "mapped_N_deg", "final_target_deg", "actual_deg",
                          "velocity_deg_s", "e_tracking_actual_minus_final_deg", "e_residual_final_minus_mapped_N_deg"):
                key = f"{leg}_{joint}_{field}"
                row[key] = flat[key]
    audit = get(native, "native_audit") or {}
    headroom = audit.get("policy_headroom_evidence") or {}
    for suffix, field in (("nominal_canonical_rad_s", "wheel_nominal_rad_s"),
                          ("final_target_canonical_rad_s", "wheel_final_rad_s"),
                          ("actual_canonical_rad_s", "wheel_actual_rad_s")):
        row["wheels_" + suffix] = [flat[f"{leg}_{field}"] for leg in old.LEGS]
    row["wheels_mapped_N_canonical_rad_s"] = [get(audit, "native_drive_target_full12", i) for i in range(8, 12)]
    row["wheels_actual_native_target_rad_s"] = get(audit, "actual_native_targets", "wheel_velocity_rad_s")
    row["wheels_zero_current_policy_native_target_rad_s"] = get(audit, "counterfactual_native_targets", "wheel_velocity_rad_s")
    row["wheels_direct_current_policy_delta_native_rad_s"] = get(audit, "native_target_delta", "wheel_velocity_rad_s")
    for index, joint in ((6, "hip"), (7, "knee")):
        prefix = f"RR_{joint}_"
        row.update({prefix + "mapper_N_before_geometry_deg": get(audit, "native_drive_target_full12", index),
            prefix + "nominal_geometry_adjustment_deg": get(audit, "nominal_geometry_adjustment_full12", index),
            prefix + "geometry_corrected_N_deg": get(headroom, "geometry_corrected_native_full12", index),
            prefix + "controller_feedback_plus_normal_bias_deg": get(audit, "controller_drive_bias_full12", index),
            prefix + "normal_bias_separately_recorded_deg": None,
            prefix + "bounded_controller_bias_deg": get(headroom, "bounded_controller_bias_full12", index),
            prefix + "effective_current_policy_residual_deg": get(headroom, "effective_policy_residual_full12", index),
            prefix + "candidate_before_final_slew_deg": get(headroom, "candidate_native_target_before_final_slew_full12", index),
            prefix + "previous_final_servo_deg": get(audit, "previous_final_drive_servo_deg", index),
            prefix + "direct_current_policy_delta_deg": flat[prefix + "direct_same_tick_policy_delta_deg"],
            prefix + "geometry_only_native_delta_rad": get(audit, "nominal_geometry_native_target_delta", "servo_position_rad", index)})
    row.update(nominal_geometry_status=get(audit, "nominal_geometry_evidence", "status"),
        nominal_geometry_context_source_control_tick=get(audit, "nominal_geometry_evidence", "context", "source_control_tick"),
        current_policy_counterfactual_scope=audit.get("counterfactual_scope"),
        nominal_geometry_counterfactual_scope=audit.get("nominal_geometry_counterfactual_scope"))
    for field in LEG_FIELDS:
        row[f"legs_{field}"] = [get(ev, "current_legs", leg, field) for leg in old.LEGS] if endpoint else None
    row.update(RR_I_initial_lift_observed=get(rr, "initial_lift_observed"),
        RR_Q_lift_established=get(rr, "lift_established"), RR_current_lift_valid=get(rr, "current_lift_valid"),
        RR_Q_history_active_lift=get(ev, "history", "active_lift", "RR"),
        RR_C_front_edge_crossed=get(ev, "history", "front_edge_crossed", "RR"),
        RR_P_placed=get(ev, "history", "placed", "RR"),
        source_owners_at_exact_observation_tick=owner_by_tick.get(tick),
        last_dispatched_owner_observations=endpoint.get("dispatched_owners") if endpoint else None)
    return row


def maxima_update(summary, name, values, tick):
    record = summary.setdefault(name, {"max_abs_full12": [None]*12, "peak_post_tick_full12": [None]*12,
                                       "measured_channel_samples_full12": [0]*12})
    for index in range(12):
        value = old.number(get(values, index))
        if value is None:
            continue
        record["measured_channel_samples_full12"][index] += 1
        peak = record["max_abs_full12"][index]
        if peak is None or abs(value) > peak:
            record["max_abs_full12"][index] = abs(value)
            record["peak_post_tick_full12"][index] = tick


def native_diagnostics(native, previous, maxima, first_changes):
    tick, audit = native["episode_physics_tick"], native["native_audit"]
    maxima_update(maxima, "raw_residual", audit.get("raw_policy_action_full12"), tick)
    maxima_update(maxima, "projected_residual", native.get("projected_residual_full12"), tick)
    direct = [old.difference(get(audit, "actual_native_targets", field, index),
              get(audit, "counterfactual_native_targets", field, index))
              for field, count in (("servo_position_rad", 8), ("wheel_velocity_rad_s", 4)) for index in range(count)]
    maxima_update(maxima, "direct_same_prestate_policy_native_delta", direct, tick)
    if previous is None:
        return
    for owner, minimum_phase, indices in (("FR_P07", "P07", (3,)), ("RL_P08", "P08", (4,)),
                                         ("RR_P09", "P09", (6, 7))):
        if owner in first_changes or native["source_phase_id"] < minimum_phase:
            continue
        changed = [i for i in indices if (delta := old.difference(get(native, "nominal_full12", i),
                    get(previous, "nominal_full12", i))) is not None and abs(delta) > 1e-9]
        if changed:
            first_changes[owner] = {"native_post_step_tick": tick,
                "native_internal_tick": audit.get("physics_tick"), "source_phase_id": native["source_phase_id"],
                "changed_joints": [old.JOINTS[i] for i in changed],
                "previous_nominal_deg": [get(previous, "nominal_full12", i) for i in changed],
                "dispatched_nominal_deg": [get(native, "nominal_full12", i) for i in changed]}


def extract(run, label, fixture_old=False):
    source, manifest = old.completed_source(run)  # Rejects running or nonfinal runs.
    final = load(source / "semantic_video_source_manifest.json")
    end = final.get("episode_physics_ticks")
    require(type(end) is int and 0 < end <= 24000 and final.get("optimizer_updates") == 0,
            "not a finalized <=200s single evaluation")
    if not fixture_old:
        require(get(manifest, "runtime_contract", "source_git_commit") != BEFORE_HEAD,
                "Old before run requires --fixture-old and cannot be labeled AFTER")
    label = "FIXTURE_OLD_NOT_AFTER_" + label if fixture_old else label
    calibration, calibration_proof = old.load_calibration(manifest)
    endpoints, owners, activations, skipped, decision_phases, issued, returned = decisions(source)
    changes, rr_events = transitions(source)
    owners_by_tick = {}
    for owner in owners:
        tick = owner["observation_tick"]
        if type(tick) is int:
            owners_by_tick.setdefault(tick, []).append(owner)
    native_stream = iter(old.lines(source / "native_tick_audit.jsonl"))
    current = next(native_stream, None)
    require(current is not None and current[1].get("episode_physics_tick") == 1, "first native tick is not1")
    native_phases, records, peaks, maxima, first_changes = Counter(), [], {}, {}, {}
    previous_native = None
    last_tick, initial, native_count, native_verified = -1, None, 0, 0
    for physical_line, physical in old.lines(source / "physical_observations.jsonl"):
        old.physical_schema(physical)
        tick = physical["physics_tick"]
        require(tick == last_tick + 1 and tick <= end, "physical stream gap, duplicate or after-terminal row")
        last_tick = tick
        if tick == 0:
            initial = old.initial_state(physical, current[1])
            native_line, native = None, None
        else:
            require(current is not None, "native stream ended before physical stream")
            native_line, native = current
            old.native_schema(native)
            require(native["episode_physics_tick"] == tick, "native/physical same-tick join failed")
            native_count += 1
            native_verified += get(native, "native_audit", "verified") is True
            native_phases[native["source_phase_id"]] += 1
            native_diagnostics(native, previous_native, maxima, first_changes)
            previous_native = native
            current = next(native_stream, None)
        for joint in ("hip", "knee"):
            data = physical["joints"][f"rear_right_{joint}"]
            error = old.difference(data.get("position_deg"), data.get("command_deg"))
            if error is not None and (joint not in peaks or abs(error) > abs(peaks[joint]["signed_error_deg"])):
                peaks[joint] = {"physics_tick": tick, "signed_error_deg": error}
        if tick == 0 or tick % 8 == 0 or tick == end:
            flat = old.flatten(physical, native, label, (physical_line, native_line), calibration=calibration)
            records.append(compact_record(flat, endpoints.get(tick), owners_by_tick, native=native))
    require(current is None and last_tick == end and native_count == end and initial is not None,
            "final physical/native counts do not match the completed source endpoint")
    require(set(endpoints) <= set(range(end + 1)), "semantic endpoint outside physical run")
    evaluation = get(final, "physical_episode", "physical_task_evaluation") or {}
    summary = {"run_label": label, "fixture_old_not_after": fixture_old, "source": str(source),
        "lifecycle": manifest["lifecycle"], "completed_at_utc": manifest["completed_at_utc"],
        "source_git_commit": get(manifest, "runtime_contract", "source_git_commit"),
        "runtime_content_sha256": get(manifest, "runtime_contract", "runtime_content_sha256"),
        "checkpoint_decisions": get(final, "checkpoint_load_provenance", "saved_global_policy_decisions"),
        "optimizer_updates": 0, "issued_decisions": issued, "returned_decisions": returned,
        "physical_task_success": final.get("physical_task_success"), "physical_evaluator_termination_reason": evaluation.get("termination_reason"),
        "physical_evaluator_termination_source": evaluation.get("termination_source"),
        "native_ticks": native_count, "native_verified_ticks": native_verified, "physical_rows": end + 1,
        "end_tick": end, "duration_s": end / 120., "sparse_record_count": len(records), "record_fields": len(records[0]),
        "native_source_phase_tick_counts": dict(native_phases), "decision_request_phase_counts": decision_phases,
        "exact_full_stream_join_complete": True, "initial_state": initial, "attitude_calibration": calibration_proof,
        "RR_tracking_peaks_over_full_run": peaks, "stage_transitions": changes, "RR_events": rr_events,
        "full_stream_residual_maxima": maxima,
        "residual_maxima_units": {"raw_residual": "policy normalized Full12",
            "projected_residual": "servo deg then wheel rad/s",
            "direct_same_prestate_policy_native_delta": "servo rad then wheel rad/s; same-prestate counterfactual"},
        "first_rear_sequence_nominal_dispatch_changes": {key: first_changes.get(key) for key in ("FR_P07", "RL_P08", "RR_P09")},
        "source_owner_observations": owners, "source_activation_events": activations,
        "semantic_clock_mismatches_not_joined": skipped,
        "array_leg_order": list(old.LEGS), "source_line_order": ["physical", "native", "semantic_decision"],
        "record_schema": "wlr50_clean.all_servo_wheel_decomposition_records.v2",
        "wheel_array_contract": {"order": list(old.LEGS), "physical_joint_order": list(old.WHEELS),
            "canonical_keys": ["wheels_nominal_canonical_rad_s", "wheels_mapped_N_canonical_rad_s",
                               "wheels_final_target_canonical_rad_s", "wheels_actual_canonical_rad_s"],
            "native_keys": ["wheels_actual_native_target_rad_s", "wheels_zero_current_policy_native_target_rad_s",
                            "wheels_direct_current_policy_delta_native_rad_s"],
            "native_equals_canonical_times_sign": list(old.WHEEL_SIGNS)},
        "evidence_limits": ["No15Hz forward fill; semantic contact/load/lift fields only at exact endpoint",
            "Owner clocks use their own observation_tick, separate from decision_end_tick; no invented alignment",
            "last_dispatched_owner_observations is from this decision, usually observation_tick=end_tick-1; not current-tick physical evidence",
            "First nominal changes detect FR knee since P07, RL hip since P08, RR hip/knee since P09 in native post-step stream; no exclusive-owner causality inferred",
            "Full tick stream joined; sparse records are tick0/every8/true terminal, not all rows",
            "Measured getter positions and derived errors are not motor force or causal attribution",
            "Missing fields are null, never zero; Q history is separate from current qualified validity",
            "mapped_N is the actual unique mapper output BEFORE geometry, using this run's shared history; not a separate fresh FSM trajectory",
            "e_residual_final_minus_mapped_N_deg is a legacy-named TOTAL difference including geometry/controller/slew/history; NOT policy or historical residual attribution",
            "RR direct_current_policy_delta uses the recorded same-prestate zero-current-policy counterfactual; geometry_only_native_delta has its separate geometry-only scope",
            "Native audit records controller feedback+normal sum only; normal_bias_separately_recorded_deg is null, not inferred or zero",
            "Geometry corrected N and bounded controller/current-policy terms come from recorded headroom, candidate precedes final slew; no missing decomposition inferred"]}
    return summary, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-prefix", required=True, type=Path)
    parser.add_argument("--fixture-old", action="store_true")
    parser.add_argument("--compare-with-summary", type=Path)
    args = parser.parse_args()
    prefix = args.output_prefix.resolve()
    targets = [Path(str(prefix) + suffix) for suffix in (".summary.json", ".records.json")]
    require(prefix.parent.is_dir() and not any(path.exists() for path in targets), "output parent missing or output exists")
    summary, records = extract(args.run, args.label, args.fixture_old)
    if args.compare_with_summary:
        earlier = load(args.compare_with_summary)
        summary["initial_pair_comparison"] = old.compare_initial(earlier["initial_state"], summary["initial_state"])
        summary["initial_pair_source_summary"] = str(args.compare_with_summary.resolve())
    summary.update(schema="wlr50_clean.after_run_sparse_evidence.v1", created_at_utc=datetime.now(timezone.utc).isoformat())
    for path, payload in zip(targets, (summary, records)):
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, allow_nan=False, indent=2 if path == targets[0] else None)
    print(json.dumps({"summary": str(targets[0]), "records": str(targets[1]), "label": summary["run_label"],
        "native_ticks": summary["native_ticks"], "records_count": len(records), "fields": summary["record_fields"],
        "same_tick_stream_join_complete": True, "source_activation_events": summary["source_activation_events"]}))


if __name__ == "__main__":
    main()
