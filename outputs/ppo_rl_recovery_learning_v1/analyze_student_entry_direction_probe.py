"""Compact one-pass analysis of one sealed student-entry direction probe.

No model, Torch, Isaac, interpolation, or counterfactual success attribution is
used.  The script selects actual 120 Hz rows immediately before/during/after
the finite intervention and the real episode endpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


CHANNELS = ("FL_hip", "FL_knee", "FR_hip", "FR_knee", "RL_hip", "RL_knee",
            "RR_hip", "RR_knee", "FL_wheel", "FR_wheel", "RL_wheel", "RR_wheel")
SELECTED = (1, 3, 6, 7)


def finite(value: Any) -> float | None:
    return (float(value) if isinstance(value, (int, float)) and
            not isinstance(value, bool) and math.isfinite(float(value)) else None)


def vector(value: Any, count: int) -> list[float | None]:
    return ([finite(item) for item in value] if isinstance(value, list) and
            len(value) == count else [None] * count)


def position(mapping: Any) -> list[float | None]:
    return vector((mapping or {}).get("position_w_m"), 3)


def current_row_entry_evidence(row: dict[str, Any]) -> dict[str, Any]:
    """Re-evaluate the entry predicate on this exact after-step physics row.

    ``decision.eligibility`` belongs to the decision-start clock and is kept
    separately.  This helper deliberately uses the row's current evaluator so
    a delayed post-release gap excursion is not called legal merely because an
    earlier decision was legal.
    """
    evaluator = row.get("physical_evaluator") or {}
    current = evaluator.get("current_legs") or {}
    history = evaluator.get("history") or {}
    rr = current.get("RR") or {}
    decision_eligibility = (row.get("decision") or {}).get("eligibility") or {}
    minimum = decision_eligibility.get("minimum_other_supports")
    floor = finite(decision_eligibility.get("force_noise_floor_n"))
    blockers: list[str] = []
    unknown: list[str] = []
    phase = row.get("resulting_phase")
    if phase is None:
        unknown.append("phase_missing")
    elif phase != "P09":
        blockers.append("phase_not_P09")
    if evaluator.get("valid") is None:
        unknown.append("physical_evaluator_valid_missing")
    elif evaluator.get("valid") is not True:
        blockers.append("physical_evaluator_not_valid")
    if rr.get("current_lift_valid") is None:
        unknown.append("RR_current_lift_valid_missing")
    elif rr.get("current_lift_valid") is not True:
        blockers.append("RR_not_currently_qualified")
    active_lift = (history.get("active_lift") or {}).get("RR")
    if active_lift is None:
        unknown.append("RR_active_lift_history_missing")
    elif active_lift is not True:
        blockers.append("RR_active_lift_history_absent")
    if rr.get("within_top_xy") is None:
        unknown.append("RR_within_top_xy_missing")
    elif rr.get("within_top_xy") is not True:
        blockers.append("RR_not_within_top_xy")
    if rr.get("within_lateral_span") is None:
        unknown.append("RR_within_lateral_span_missing")
    elif rr.get("within_lateral_span") is not True:
        blockers.append("RR_not_within_lateral_span")
    if rr.get("air") is None:
        unknown.append("RR_air_missing")
    elif rr.get("air") is not True:
        blockers.append("RR_not_AIR")
    top_contact = rr.get("top_contact", rr.get("top_surface_contact"))
    if top_contact is None:
        unknown.append("RR_top_contact_missing")
    elif top_contact is not False:
        blockers.append("RR_top_contact_true")
    ground_contact = rr.get("ground_contact")
    if ground_contact is None:
        unknown.append("RR_ground_contact_missing")
    elif ground_contact is not False:
        blockers.append("RR_ground_contact_true")
    front = finite(rr.get("front_distance_m"))
    gap = finite(rr.get("clearance_m"))
    if front is None or front < 0.0:
        blockers.append("RR_front_distance_not_nonnegative")
    if gap is None or gap <= 0.0:
        blockers.append("RR_positive_top_gap_absent")
    if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 1:
        blockers.append("minimum_other_supports_missing")
    if floor is None:
        blockers.append("force_noise_floor_missing")
    supporting: list[str] = []
    if floor is not None:
        for leg in ("FR", "FL", "RL"):
            item = current.get(leg) or {}
            force = finite(item.get("bearing_force_n"))
            if (item.get("support") is True and item.get("bearing_verified") is True
                    and force is not None and force >= floor):
                supporting.append(leg)
    if "FL" not in supporting:
        if (current.get("FL") or {}).get("support") is None:
            unknown.append("FL_real_support_missing")
        else:
            blockers.append("FL_real_support_absent")
    if isinstance(minimum, int) and not isinstance(minimum, bool) \
            and len(supporting) < minimum:
        blockers.append("insufficient_other_real_supports")
    return {
        "eligible": (False if blockers else (None if unknown else True)),
        "blockers": blockers,
        "unknown_fields": unknown,
        "clock": "same after-step episode_physics_tick as current evaluator/actual readback",
        "phase": phase,
        "RR_gap_m": gap,
        "RR_front_distance_m": front,
        "supporting_other_legs": supporting,
        "minimum_other_supports": minimum,
        "force_noise_floor_n": floor,
    }


def compact_decision(row: dict[str, Any]) -> dict[str, Any]:
    eligibility = row.get("eligibility") or {}
    rr = eligibility.get("RR") or {}
    original = vector(row.get("original_student_raw_full12"), 12)
    applied = vector(row.get("applied_raw_full12"), 12)
    step_info = row.get("step_info") or {}
    audit = step_info.get("actuator_target_effect_audit") or {}
    raw = step_info.get("raw_observation") or {}
    evaluator = (step_info.get("semantic_task") or {}).get("physical_evaluator") or {}
    legs = evaluator.get("current_legs") or {}
    fl = legs.get("FL") or {}
    executed_final = vector(step_info.get("actual_drive_target_full12"), 12)
    mapper_nominal = vector(audit.get("native_drive_target_full12"), 12)
    actual = vector(raw.get("actual_full12"), 12)
    return {
        "resolution": "15Hz decision-start eligibility; no interpolation",
        "decision_index": row.get("decision_index"),
        "decision_start_tick": row.get("decision_start_tick"),
        "decision_end_tick": row.get("decision_end_tick"),
        "decision_start_time_s": finite(row.get("decision_start_time_s")),
        "decision_physics_ticks": row.get("decision_physics_ticks"),
        "phase": row.get("phase"),
        "probe_active": row.get("probe_active"),
        "decision_start_eligibility": eligibility,
        "RR_start_gap_m": finite(rr.get("clearance_m")),
        "RR_start_front_distance_m": finite(rr.get("front_distance_m")),
        "student_source_raw_full12": dict(zip(CHANNELS, original)),
        "applied_raw_full12": dict(zip(CHANNELS, applied)),
        "request_audit": row.get("original_student_request_audit"),
        "same_decision_end_response": {
            "clock_note": (
                "step_info actual_drive is executed FINAL at decision end; native_drive "
                "is mapper nominal N; neither is relabelled as decision-start state"),
            "episode_physics_tick": row.get("decision_end_tick"),
            "executed_FINAL_target_canonical": dict(zip(CHANNELS, executed_final)),
            "mapper_nominal_native_target": dict(zip(CHANNELS, mapper_nominal)),
            "actual_q_or_qd": dict(zip(CHANNELS, actual)),
            "FL_support": {key: fl.get(key) for key in (
                "support", "bearing_verified", "bearing_force_n", "air",
                "top_surface_contact", "ground_contact", "clearance_m",
                "front_distance_m")},
            "COM_position_w_m": position(raw.get("center_of_mass")),
        },
    }


def compact(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("raw_observation") or {}
    evaluator = row.get("physical_evaluator") or {}
    legs = evaluator.get("current_legs") or {}
    rr = legs.get("RR") or {}
    audit = row.get("native_readback_audit") or {}
    ack = row.get("atomic_ACK") or {}
    decision = row.get("decision") or {}
    if (not isinstance(ack.get("drive_target_full12"), list) or
            len(ack["drive_target_full12"]) != 12 or
            ack.get("articulation_writes_this_call") != 1 or
            audit.get("verified") is not True or
            ack.get("physics_tick") != audit.get("physics_tick") or
            audit.get("setter_dispatch_targets_equal") is not True or
            audit.get("actual_mapping_matches_dispatch") is not True):
        raise RuntimeError("unverified atomic ACK / actuator readback row")
    executed_final = vector(ack.get("drive_target_full12"), 12)
    mapper_nominal = vector(audit.get("native_drive_target_full12"), 12)
    actual = vector(raw.get("actual_full12"), 12)
    original = vector(decision.get("original_student_raw_full12"), 12)
    applied = vector(decision.get("applied_raw_full12"), 12)
    selected = {}
    for index in SELECTED:
        selected[CHANNELS[index]] = {
            "student_raw": original[index], "applied_raw": applied[index],
            "executed_FINAL_target_canonical": executed_final[index],
            "mapper_nominal_native_target": mapper_nominal[index],
            "actual_q_or_qd": actual[index],
            "FINAL_minus_actual": (
                None if executed_final[index] is None or actual[index] is None
                else executed_final[index] - actual[index])}
    supporting = []
    for name in ("FL", "FR", "RL"):
        leg = legs.get(name) or {}
        if leg.get("support") is True and leg.get("bearing_verified") is True:
            supporting.append(name)
    fl = legs.get("FL") or {}
    return {
        "episode_physics_tick": row.get("episode_physics_tick"),
        "sim_time_s": finite(row.get("sim_time_s")),
        "source_phase": row.get("source_phase"),
        "resulting_phase": row.get("resulting_phase"),
        "probe_status": (row.get("probe_state") or {}).get("status"),
        "probe_active": decision.get("probe_active"),
        "decision_index": decision.get("decision_index"),
        "decision_start_tick": decision.get("decision_start_tick"),
        "decision_start_time_s": finite(decision.get("decision_start_time_s")),
        "decision_start_eligibility": decision.get("eligibility"),
        "exact_row_entry_evidence": current_row_entry_evidence(row),
        "selected_channels": selected,
        "student_source_raw_full12": dict(zip(CHANNELS, original)),
        "applied_raw_full12": dict(zip(CHANNELS, applied)),
        "all_executed_FINAL_target_canonical": dict(zip(CHANNELS, executed_final)),
        "all_mapper_nominal_native_target": dict(zip(CHANNELS, mapper_nominal)),
        "all_actual_q_or_qd": dict(zip(CHANNELS, actual)),
        "command_provenance": {
            "executed_FINAL_key": "atomic_ACK.drive_target_full12",
            "mapper_nominal_key": "native_readback_audit.native_drive_target_full12",
            "ACK_physics_tick": ack.get("physics_tick"),
            "audit_physics_tick": audit.get("physics_tick"),
            "articulation_writes_this_call": ack.get("articulation_writes_this_call"),
            "audit_verified": audit.get("verified"),
            "setter_dispatch_targets_equal": audit.get("setter_dispatch_targets_equal"),
            "actual_mapping_matches_dispatch": audit.get("actual_mapping_matches_dispatch"),
            "actual_native_targets_readback": audit.get("actual_native_targets"),
        },
        "RR": {key: rr.get(key) for key in (
            "current_lift_valid", "within_top_xy", "within_lateral_span", "air",
            "top_surface_contact", "ground_contact", "clearance_m", "front_distance_m",
            "bearing_verified", "bearing_force_n", "obstacle_normal_force_n")},
        "supporting_other_legs": supporting,
        "FL_support": {key: fl.get(key) for key in (
            "support", "bearing_verified", "bearing_force_n", "air",
            "top_surface_contact", "ground_contact", "clearance_m",
            "front_distance_m")},
        "base_position_w_m": position(raw.get("base")),
        "COM_position_w_m": position(raw.get("center_of_mass")),
        "RR_upper_mount_position_w_m": position((raw.get("bodies") or {}).get("rear_right_upper")),
        "RR_wheel_center_position_w_m": position((raw.get("bodies") or {}).get("rear_right_wheel")),
        "body_collision": raw.get("body_collision"),
    }


def actual_state_snapshot(row: dict[str, Any], note: str) -> dict[str, Any]:
    """Current measured state without leaking an adjacent decision's raw action."""
    result = compact(row)
    result.pop("student_source_raw_full12", None)
    result.pop("applied_raw_full12", None)
    for item in result.get("selected_channels", {}).values():
        item.pop("student_raw", None)
        item.pop("applied_raw", None)
    result["clock_note"] = note
    return result


def write_new(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def analyze(run_dir: Path, output: Path) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve(strict=True)
    manifest_path = run_dir / "run_manifest.json"
    physics_path = run_dir / "diagnostic_physics.jsonl"
    decisions_path = run_dir / "diagnostic_decisions.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("lifecycle") != "DIAGNOSTIC_SEALED" or
            manifest.get("frozen_learned_state_unchanged") is not True or
            manifest.get("new_PPO_updates") != 0 or
            manifest.get("new_AUX_accepted_updates") != 0):
        raise RuntimeError("probe must be naturally sealed, frozen, and zero-credit")
    decisions_digest = hashlib.sha256()
    probe = manifest.get("probe") or {}
    release_tick = probe.get("release_tick")
    active_decisions: set[int] = set()
    decision_line_count = 0
    minimum_legal_decision = minimum_positive_decision = None
    with decisions_path.open("rb") as stream:
        for raw_line in stream:
            decisions_digest.update(raw_line)
            row = json.loads(raw_line)
            decision_line_count += 1
            decision_index = row.get("decision_index")
            if row.get("probe_active") is True and isinstance(decision_index, int) \
                    and not isinstance(decision_index, bool):
                active_decisions.add(decision_index)
            start_tick = row.get("decision_start_tick")
            if not isinstance(release_tick, int) or not isinstance(start_tick, int) \
                    or start_tick < release_tick or row.get("probe_active") is True:
                continue
            eligibility = row.get("eligibility") or {}
            gap = finite((eligibility.get("RR") or {}).get("clearance_m"))
            selected = compact_decision(row)
            if gap is not None and gap > 0.0 and (
                    minimum_positive_decision is None or gap < minimum_positive_decision[0]):
                minimum_positive_decision = (gap, selected)
            if eligibility.get("eligible") is True and gap is not None and gap > 0.0 and (
                    minimum_legal_decision is None or gap < minimum_legal_decision[0]):
                minimum_legal_decision = (gap, selected)

    digest = hashlib.sha256()
    last_wait = first_active = last_active = first_released = terminal = None
    minimum_positive_post_release_120hz = None
    active_rows: list[dict[str, Any]] = []
    line_count = 0
    with physics_path.open("rb") as stream:
        for raw_line in stream:
            digest.update(raw_line)
            row = json.loads(raw_line)
            line_count += 1
            status = (row.get("probe_state") or {}).get("status")
            active = (row.get("decision") or {}).get("probe_active") is True
            current = compact(row)
            terminal = current
            if active:
                selected = current
                if first_active is None:
                    first_active = selected
                last_active = selected
                active_rows.append(selected)
            elif first_active is None:
                last_wait = current
            elif first_released is None and status == "RELEASED":
                first_released = current
            if first_active is not None and status == "RELEASED":
                selected = current
                exact = selected["exact_row_entry_evidence"]
                gap = exact.get("RR_gap_m")
                if gap is not None and gap > 0.0 and (
                        minimum_positive_post_release_120hz is None or
                        gap < minimum_positive_post_release_120hz[0]):
                    minimum_positive_post_release_120hz = (gap, selected)
            for candidate in (minimum_legal_decision, minimum_positive_decision):
                if candidate is None:
                    continue
                decision_row = candidate[1]
                start = decision_row.get("decision_start_tick")
                end = decision_row.get("decision_end_tick")
                tick = current.get("episode_physics_tick")
                if tick == start:
                    decision_row["decision_start_actual_state_120Hz"] = actual_state_snapshot(
                        row, "exact decision-start state; FINAL is last committed target")
                elif tick == start + 1 and "first_effect_actual_state_120Hz" not in decision_row:
                    decision_row["first_effect_actual_state_120Hz"] = actual_state_snapshot(
                        row, "first 120Hz effect tick after this decision start")
                if tick == end:
                    decision_row["decision_end_actual_state_120Hz"] = actual_state_snapshot(
                        row, "exact same-decision endpoint state and mapped FINAL target")
    if first_active is None:
        selected_rows = [item for item in (last_wait, terminal) if item is not None]
        selection = "NO_ACTIVATION_observed"
    else:
        indices = sorted(set((0, len(active_rows) // 2, len(active_rows) - 1)))
        selected_rows = ([last_wait] if last_wait is not None else []) + [
            active_rows[index] for index in indices]
        if first_released is not None:
            selected_rows.append(first_released)
        if terminal is not None and terminal.get("episode_physics_tick") != \
                selected_rows[-1].get("episode_physics_tick"):
            selected_rows.append(terminal)
        selection = "last_pre_then_first_mid_last_active_then_first_release_and_endpoint"
    start_time = finite(probe.get("start_time_s"))
    release_time = finite(probe.get("release_time_s"))
    start_tick = probe.get("start_tick")
    release_tick = probe.get("release_tick")
    active_duration = (None if start_time is None or release_time is None
                       else release_time - start_time)
    active_tick_span = (None if not isinstance(start_tick, int) or
                        not isinstance(release_tick, int)
                        else release_tick - start_tick)
    active_response = None
    if active_rows:
        first, last = active_rows[0], active_rows[-1]
        actual_delta = {}
        executed_final_start_end = {}
        mapper_nominal_start_end = {}
        for channel in ("FL_knee", "FR_knee", "RR_hip", "RR_knee"):
            before = finite(first["selected_channels"][channel].get("actual_q_or_qd"))
            after = finite(last["selected_channels"][channel].get("actual_q_or_qd"))
            actual_delta[channel] = (None if before is None or after is None
                                     else after - before)
            executed_final_start_end[channel] = [
                finite(first["selected_channels"][channel].get(
                    "executed_FINAL_target_canonical")),
                finite(last["selected_channels"][channel].get(
                    "executed_FINAL_target_canonical"))]
            mapper_nominal_start_end[channel] = [
                finite(first["selected_channels"][channel].get(
                    "mapper_nominal_native_target")),
                finite(last["selected_channels"][channel].get(
                    "mapper_nominal_native_target"))]
        gaps = [finite(item["RR"].get("clearance_m")) for item in active_rows]
        fronts = [finite(item["RR"].get("front_distance_m")) for item in active_rows]
        fl_forces = [finite(item["FL_support"].get("bearing_force_n"))
                     for item in active_rows]
        rr_forces = [finite(item["RR"].get("bearing_force_n")) for item in active_rows]
        active_response = {
            "first_tick": first.get("episode_physics_tick"),
            "last_tick": last.get("episode_physics_tick"),
            "actual_channel_delta_deg": actual_delta,
            "executed_FINAL_start_end": executed_final_start_end,
            "mapper_nominal_N_start_end": mapper_nominal_start_end,
            "RR_gap_start_end_m": [gaps[0], gaps[-1]],
            "RR_gap_min_max_m": [min(x for x in gaps if x is not None),
                                  max(x for x in gaps if x is not None)],
            "RR_front_start_end_m": [fronts[0], fronts[-1]],
            "FL_support_force_min_max_n": [min(x for x in fl_forces if x is not None),
                                            max(x for x in fl_forces if x is not None)],
            "RR_bearing_force_max_n": max(x for x in rr_forces if x is not None),
            "RR_top_contact_observed": any(
                item["RR"].get("top_surface_contact") is True for item in active_rows),
            "COM_displacement_m": [
                (None if first["COM_position_w_m"][index] is None or
                 last["COM_position_w_m"][index] is None else
                 last["COM_position_w_m"][index] - first["COM_position_w_m"][index])
                for index in range(3)],
        }
    result = {
        "schema": "outputs.student_entry_direction_probe_analysis.v1",
        "run_dir": str(run_dir), "run_manifest": str(manifest_path),
        "source": str(physics_path), "source_sha256": digest.hexdigest(),
        "source_rows": line_count, "selection": selection,
        "decision_source": str(decisions_path),
        "decision_source_sha256": decisions_digest.hexdigest(),
        "decision_source_rows": decision_line_count,
        "probe": probe,
        "task_modality": {key: manifest.get(key) for key in (
            "run_role", "experiment_id", "seed", "deterministic",
            "natural_P01_student_prefix", "successful_N_prefix_used",
            "state_injection_or_teleport", "candidate_semantics",
            "formal_deterministic_policy_result", "formal_policy_or_success_claim")},
        "actual_intervention_stats": {
            "active_decision_count": len(active_decisions),
            "active_physics_row_count": len(active_rows),
            "start_tick": start_tick,
            "release_tick": release_tick,
            "active_tick_span": active_tick_span,
            "start_time_s": start_time,
            "release_time_s": release_time,
            "active_duration_s": active_duration,
            "release_reason": probe.get("release_reason"),
            "count_semantics": "unique recorded decision_index values with probe_active=true",
            "measured_active_response": active_response,
        },
        "post_release_gap_evidence": {
            "minimum_legal_RR_gap_15Hz_decision_start": (
                None if minimum_legal_decision is None else minimum_legal_decision[1]),
            "minimum_positive_RR_gap_15Hz_regardless_of_legality": (
                None if minimum_positive_decision is None else minimum_positive_decision[1]),
            "minimum_positive_RR_gap_120Hz_geometry": (
                None if minimum_positive_post_release_120hz is None
                else minimum_positive_post_release_120hz[1]),
            "legal_definition": (
                "recorded complete 15Hz decision-start eligibility; the 120Hz evaluator "
                "may omit augmented fields and is never converted from missing to false"),
            "clock_alignment": (
                "decision-start eligibility and student/applied raw share the 15Hz start; "
                "matching 120Hz actual state is attached only at an exactly equal tick; "
                "mapped FINAL/readback is separately labelled as same-decision end response"),
            "interpretation": (
                "post-release minimum is a delayed measured response; it is not evidence "
                "that the three-decision intervention maintained capture"),
        },
        "actual_task_termination_reason": manifest.get("actual_task_termination_reason"),
        "final_phase": manifest.get("final_phase"),
        "physical_seconds": manifest.get("physical_seconds"),
        "zero_credit": {key: manifest.get(key) for key in (
            "new_PPO_samples", "new_PPO_updates", "new_optimizer_steps",
            "new_AUX_accepted_updates", "new_AUX_attempted_steps",
            "teacher_actions_deployed", "diagnostic_rows_entered_in_on_policy_storage")},
        "selected_actual_rows": selected_rows,
        "claims": {"candidate_is_policy_output": False,
            "candidate_is_PPO_or_AUX_data": False,
            "direction_probe_proves_success_or_causality": False,
            "post_release_rebound_is_maintained_capture": False,
            "missing_ticks_interpolated": False},
    }
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    write_new(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.run_dir, args.output)
    print(json.dumps({"output": str(Path(args.output).resolve()),
                      "probe": result["probe"],
                      "selected_rows": len(result["selected_actual_rows"])}, indent=2))


if __name__ == "__main__":
    main()
