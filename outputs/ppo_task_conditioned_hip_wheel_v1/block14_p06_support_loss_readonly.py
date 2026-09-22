"""Bounded, CPU-only audit of block14 FL support after first placement.

Only the first 437 complete audit records are read.  That endpoint (episode
tick 3496) is inside already completed rollout/update 1509.  The live audit
file may continue growing and is deliberately neither read nor hashed past
that fixed endpoint.  No torch/model/simulator import is used.
"""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import sys


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
RUN = ROOT / (
    "runs/ppo_task_conditioned_hip_wheel_v1/train/"
    "20260921T1355291518150Z_g649ccd906421_aac78369d59846409cb2d4039083e87a"
)
AUDIT = RUN / "residual_and_projection_audit.jsonl"
UPDATES = RUN / "optimizer_updates.jsonl"
B_CURRENT = OUT / "currentB_event_windows.json"
JSON_OUT = OUT / "block14_p06_support_loss_readonly.json"
MD_OUT = OUT / "block14_p06_support_loss_readonly.md"

FIXED_END_LINE = 437
FIXED_END_TICK = 3496
BASE_GLOBAL = 197120
CHANNEL_INDICES = (0, 1, 8, 9, 10, 11)
CHANNELS = ("FL_hip", "FL_knee", "FL_wheel", "FR_wheel", "RL_wheel", "RR_wheel")
UNITS = ("deg", "deg", "rad_s", "rad_s", "rad_s", "rad_s")


def read_complete_prefix(path: Path, count: int) -> tuple[list[dict], int]:
    rows: list[dict] = []
    consumed = 0
    with path.open("rb") as stream:
        for index in range(count):
            line = stream.readline()
            if not line or not line.endswith(b"\n"):
                raise RuntimeError(f"{path} lacks complete fixed line {index + 1}")
            consumed += len(line)
            rows.append(json.loads(line))
    return rows, consumed


def select(values: list | tuple) -> list:
    return [values[index] for index in CHANNEL_INDICES]


def is_top(leg: dict) -> bool:
    return bool(leg["top_surface_contact"] and leg["bearing_verified"] and leg["support"])


def mode(leg: dict) -> str:
    if is_top(leg):
        return "TOP"
    if leg["ground_contact"]:
        return "GROUND"
    if leg["air"]:
        return "AIR"
    return "AMBIGUOUS"


def contact(leg: dict) -> dict:
    return {key: leg[key] for key in (
        "air", "ground_contact", "top_surface_contact", "bearing_verified", "support",
        "bearing_force_n", "load_fraction", "clearance_m", "front_distance_m", "within_top_xy",
        "contact_surface", "consecutive_air_samples", "consecutive_top_samples"
    )} | {"mode": mode(leg), "current_TOP_support": is_top(leg)}


def actual_selected(applied: dict) -> list[float]:
    audit = applied["actuator_target_effect_audit"]
    tracking = audit["tracking_reference_evidence"]
    # Front servo axes use canonical = degrees(native q) - saved standing offset.
    servo = [
        math.degrees(tracking["actual_measured_physical_rad"][index])
        - tracking["standing_pose_deg"][index]
        for index in (0, 1)
    ]
    wheels = applied["semantic_task"]["physical_evaluator"]["measured_wheel_velocity_rad_s"]
    return servo + list(wheels)


def endpoint(row: dict, *, p06_ordinal: int | None) -> dict:
    applied = row["applied_audit"]
    task = applied["semantic_task"]
    evaluator = task["physical_evaluator"]
    actuator = applied["actuator_target_effect_audit"]
    headroom = actuator["policy_headroom_evidence"]
    tracking = actuator["tracking_reference_evidence"]
    request = row["policy_request"]
    ticks = applied["actuator_target_effect_audit_ticks"]
    return {
        "global_policy_decision": row["global_policy_decision"],
        "p06_request_ordinal": p06_ordinal,
        "request_phase": applied["phase_id"],
        "end_phase": applied["end_phase_id"],
        "episode_tick_range": [ticks[0]["episode_physics_tick"], ticks[-1]["episode_physics_tick"]],
        "endpoint_tick": applied["physics_tick"],
        "sim_time_s": applied["sim_time_s"],
        "contact": {
            "FL": contact(evaluator["current_legs"]["FL"]),
            "FR": contact(evaluator["current_legs"]["FR"]),
            "observed_support_contacts": task["transfer_role_context"]["observed_support_contacts"],
        },
        "historical": {
            "placed": task["placed_history"],
            "event_ticks": task["history"]["event_ticks"],
        },
        "progress": {
            "body_forward_m": evaluator["goal_features"]["body_forward_m"],
            "phase_progress": task["phase_progress"],
            "task_progress_potential": task["task_progress_potential"],
            "support_count": evaluator["goal_features"]["support_count"],
        },
        "control": {
            "channels": list(CHANNELS),
            "units": list(UNITS),
            "nominal": select(applied["nominal_action_full12"]),
            "mapped_baseline_native_plus_controller": select(headroom["baseline_native_plus_controller_full12"]),
            "requested_policy_residual": select(headroom["requested_policy_residual_full12"]),
            "effective_policy_residual": select(headroom["effective_policy_residual_full12"]),
            "final_drive_target": select(applied["actual_drive_target_full12"]),
            "measured_actual": actual_selected(applied),
            "actual_servo_measurement_semantics": (
                "saved dispatch tracking readback converted to canonical deg; front axes are "
                "degrees(actual_measured_physical_rad)-standing_pose_deg"
            ),
            "actual_wheel_measurement_semantics": "saved evaluator canonical joint velocity at decision endpoint",
            "dispatch_physics_tick": tracking["dispatch_physics_tick"],
            "previous_feedback_sample_tick": tracking["previous_feedback_sample_tick"],
            "all_native_ticks_verified": all(item["verified"] for item in ticks),
        },
        "history_and_cap": {
            "rho": request["rho"],
            "stage_index": request["stage_index"],
            "encoded_stage_age": request["encoded_stage_age"],
            "predecessor_completed": request["predecessor_completed"],
            "base_network_mean": select(request["base_mean_full12"]),
            "H_history_center": select(request["history_center_full12"]),
            "conditional_mean": select(request["conditional_mean_full12"]),
            "effective_sigma": select(request["effective_sigma_full12"]),
            "selected_raw": select(request["selected_raw_full12"]),
            "previous_filtered_request_physical": select(request["previous_filtered_request_full12"]),
            "predecessor_cap": select(request["predecessor_cap_full12"]),
            "current_cap": select(request["current_cap_full12"]),
            "cap_transition_gate": select(request["cap_transition_gate_full12"]),
            "receiving_sigma_multiplier": select(request["receiving_sigma_multiplier_full12"]),
        },
    }


def runs(rows: list[dict]) -> list[dict]:
    result: list[dict] = []
    for row in rows:
        applied = row["applied_audit"]
        leg = applied["semantic_task"]["physical_evaluator"]["current_legs"]["FL"]
        current = mode(leg)
        if not result or result[-1]["endpoint_mode"] != current:
            result.append({
                "endpoint_mode": current,
                "first_endpoint_tick": applied["physics_tick"],
                "last_endpoint_tick": applied["physics_tick"],
                "endpoint_count": 1,
            })
        else:
            result[-1]["last_endpoint_tick"] = applied["physics_tick"]
            result[-1]["endpoint_count"] += 1
    return result


def fmt(values: list[float], digits: int = 3) -> str:
    return " / ".join(f"{value:+.{digits}f}" for value in values)


def main(*, check_only: bool = False) -> None:
    source_size_before = AUDIT.stat().st_size
    rows, consumed_bytes = read_complete_prefix(AUDIT, FIXED_END_LINE)
    source_size_after = AUDIT.stat().st_size
    for one_based, row in enumerate(rows, 1):
        applied = row["applied_audit"]
        assert row["global_policy_decision"] == BASE_GLOBAL + one_based
        assert applied["physics_tick"] == 8 * one_based
        assert applied["physics_ticks"] == 8
    assert rows[-1]["applied_audit"]["physics_tick"] == FIXED_END_TICK

    update_rows, _ = read_complete_prefix(UPDATES, 4)
    completed_update = update_rows[-1]
    assert completed_update["ppo_update"] == 1509
    assert completed_update["global_policy_decisions"] == 197632
    fixed_artifacts = [
        RUN / "rollouts/rollout_001509.pt",
        RUN / "rollouts/update_001509_likelihood.json",
    ]
    assert all(path.is_file() for path in fixed_artifacts)

    by_tick = {row["applied_audit"]["physics_tick"]: row for row in rows}
    transition = by_tick[2960]
    transition_evidence = transition["applied_audit"]["stage_transition_evidence"]
    assert len(transition_evidence) == 1
    assert {key: transition_evidence[0][key] for key in ("from_stage", "to_stage", "physics_tick")} == {
        "from_stage": "P05", "to_stage": "P06", "physics_tick": 2960,
    }
    placement_tick = transition_evidence[0]["history"]["event_ticks"]["placed"]["FL"]
    assert placement_tick == 2957

    p06 = [row for row in rows if 2968 <= row["applied_audit"]["physics_tick"] <= FIXED_END_TICK]
    assert len(p06) == 67
    assert all(row["applied_audit"]["phase_id"] == row["applied_audit"]["end_phase_id"] == "P06" for row in p06)
    ordinal = {row["applied_audit"]["physics_tick"]: index + 1 for index, row in enumerate(p06)}
    first_p06, first64, latest = p06[0], p06[:64], p06[-1]
    assert first64[-1]["applied_audit"]["physics_tick"] == 3472

    first_loss_index = next(index for index, row in enumerate(p06)
                            if not is_top(row["applied_audit"]["semantic_task"]["physical_evaluator"]["current_legs"]["FL"]))
    loss = p06[first_loss_index]
    previous = p06[first_loss_index - 1]
    reacquired = next(row for row in p06[first_loss_index + 1:]
                      if is_top(row["applied_audit"]["semantic_task"]["physical_evaluator"]["current_legs"]["FL"]))
    loss_tick = loss["applied_audit"]["physics_tick"]
    previous_tick = previous["applied_audit"]["physics_tick"]
    reacquired_tick = reacquired["applied_audit"]["physics_tick"]
    loss_leg = loss["applied_audit"]["semantic_task"]["physical_evaluator"]["current_legs"]["FL"]
    reacquired_leg = reacquired["applied_audit"]["semantic_task"]["physical_evaluator"]["current_legs"]["FL"]
    first_air_tick = loss_tick - loss_leg["consecutive_air_samples"] + 1
    saved_reacquired_top_streak_tick = reacquired_tick - reacquired_leg["consecutive_top_samples"] + 1
    assert previous_tick == 3096 and loss_tick == 3104 and reacquired_tick == 3128
    assert first_air_tick == previous_tick + 1 == 3097
    assert saved_reacquired_top_streak_tick == 3126
    last_proven_air = by_tick[3120]["applied_audit"]["semantic_task"]["physical_evaluator"]["current_legs"]["FL"]
    assert last_proven_air["air"] and last_proven_air["consecutive_air_samples"] == 24

    selected_ticks = (2960, 2968, 3096, 3104, 3112, 3120, 3128, 3472, 3496)
    selected = [endpoint(by_tick[tick], p06_ordinal=ordinal.get(tick)) for tick in selected_ticks]

    # Same-row target and measurement semantics, plus the exact H/cap recurrence.
    max_final_error = 0.0
    max_effective_error = 0.0
    max_native_front_target_error = 0.0
    all_native_verified = True
    for row in [transition, *p06]:
        applied = row["applied_audit"]
        actuator = applied["actuator_target_effect_audit"]
        headroom = actuator["policy_headroom_evidence"]
        final = select(applied["actual_drive_target_full12"])
        baseline = select(headroom["baseline_native_plus_controller_full12"])
        effective = select(headroom["effective_policy_residual_full12"])
        requested = select(headroom["requested_policy_residual_full12"])
        max_final_error = max(max_final_error, *(abs(value - base - residual)
            for value, base, residual in zip(final, baseline, effective)))
        max_effective_error = max(max_effective_error, *(abs(value - requested_value)
            for value, requested_value in zip(effective, requested)))
        tracking = actuator["tracking_reference_evidence"]
        native = actuator["actual_native_targets"]["servo_position_rad"]
        for index in (0, 1):
            canonical = math.degrees(native[index]) - tracking["standing_pose_deg"][index]
            max_native_front_target_error = max(max_native_front_target_error,
                                                 abs(canonical - applied["actual_drive_target_full12"][index]))
        all_native_verified &= actuator["verified"] and all(
            item["verified"] for item in applied["actuator_target_effect_audit_ticks"])
    assert max_final_error < 1e-12 and max_effective_error < 1e-12
    assert max_native_front_target_error < 2e-5 and all_native_verified

    first_request = first_p06["policy_request"]
    transition_request = transition["applied_audit"]["actuator_target_effect_audit"]["policy_headroom_evidence"][
        "requested_policy_residual_full12"]
    prior_request_error = max(abs(a - b) for a, b in zip(
        first_request["previous_filtered_request_full12"], transition_request))
    assert prior_request_error < 5e-7
    gate_rows = [row["applied_audit"]["physics_tick"] for row in p06
                 if any(row["policy_request"]["cap_transition_gate_full12"])]
    assert gate_rows == [2968]
    assert [index for index, value in enumerate(first_request["cap_transition_gate_full12"]) if value] == list(range(10))
    loss_request = loss["policy_request"]
    previous_raw = previous["policy_request"]["selected_raw_full12"]
    loss_H_error = max(abs(a - b) for a, b in zip(loss_request["history_center_full12"], previous_raw))
    conditional_error = max(abs(mu - (.1 * base + .9 * history)) for mu, base, history in zip(
        loss_request["conditional_mean_full12"], loss_request["base_mean_full12"], loss_request["history_center_full12"]))
    loss_filtered_request = loss["applied_audit"]["actuator_target_effect_audit"]["policy_headroom_evidence"][
        "requested_policy_residual_full12"]
    loss_bounded_candidate = [math.tanh(raw) * cap for raw, cap in zip(
        loss_request["selected_raw_full12"], loss_request["current_cap_full12"])]
    bounded_to_filtered_delta = [filtered - bounded for filtered, bounded in zip(
        loss_filtered_request, loss_bounded_candidate)]
    rate_limited_indices = [index for index, delta in enumerate(bounded_to_filtered_delta) if abs(delta) > 1e-6]
    assert loss_H_error < 1e-12 and conditional_error < 5e-8
    assert rate_limited_indices == [8]

    def current_leg(row: dict, leg: str) -> dict:
        return row["applied_audit"]["semantic_task"]["physical_evaluator"]["current_legs"][leg]

    assert all(row["applied_audit"]["semantic_task"]["placed_history"]["FL"] for row in p06)
    assert all(is_top(current_leg(row, "FR")) for row in p06)
    p06_modes = Counter(mode(current_leg(row, "FL")) for row in p06)
    first64_modes = Counter(mode(current_leg(row, "FL")) for row in first64)
    assert p06_modes == Counter({"AIR": 41, "TOP": 26})
    assert first64_modes == Counter({"AIR": 38, "TOP": 26})

    latest_leg = current_leg(latest, "FL")
    latest_air_start = FIXED_END_TICK - latest_leg["consecutive_air_samples"] + 1
    assert latest_air_start == 3370 and latest_leg["consecutive_air_samples"] == 127

    # Optional descriptive anchor: consume only the existing sealed B summary,
    # never its raw recording.  It is not a time/state-matched counterfactual.
    b_current = json.loads(B_CURRENT.read_text(encoding="utf-8"))
    assert b_current["schema"] == "sealed_physical_event_windows.v1"
    assert b_current["role"] == "B" and b_current["sampling_mode"] == "nominal_without_learned_residual"
    assert b_current["history_event_ticks"]["placed"]["FL"] == 2677
    assert b_current["stage_entry_ticks"]["P06"] == 2680
    b_p06 = b_current["windows"]["P06_rear_approach"]
    b_physical_fl = b_p06["physical"]["actual_contact_ticks"]["FL"]
    b_endpoint_fl = b_p06["current_support_samples"]["FL"]
    assert b_p06["start_tick"] == 2680 and b_p06["observed_end_tick"] == 5160
    assert b_physical_fl == {"ground": 0, "obstacle": 2481, "air": 0}
    assert b_endpoint_fl["top_verified_bearing_samples"] == b_endpoint_fl["support_samples"] == 311
    assert b_endpoint_fl["air_samples"] == b_endpoint_fl["ground_samples"] == 0

    def progress(row: dict) -> dict:
        task = row["applied_audit"]["semantic_task"]
        evaluator = task["physical_evaluator"]
        return {
            "tick": row["applied_audit"]["physics_tick"],
            "body_forward_m": evaluator["goal_features"]["body_forward_m"],
            "phase_progress": task["phase_progress"],
            "task_progress_potential": task["task_progress_potential"],
        }

    p_transition, p_first, p_previous, p_loss, p_reacquired, p_latest = map(
        progress, (transition, first_p06, previous, loss, reacquired, latest))
    result = {
        "schema": "wlr50_clean.block14_p06_support_loss_readonly.v1",
        "status": "PASS",
        "scope": "physical/control window only; no reward/GAE/optimizer claim and no simulation",
        "source_binding": {
            "run": str(RUN),
            "audit_path": str(AUDIT),
            "fixed_complete_lines_read": FIXED_END_LINE,
            "consumed_prefix_bytes": consumed_bytes,
            "fixed_prefix_end": {"global_policy_decision": rows[-1]["global_policy_decision"], "episode_tick": FIXED_END_TICK},
            "analyzed_line_range_inclusive": [370, 437],
            "analyzed_episode_ticks_inclusive": [2960, FIXED_END_TICK],
            "active_file_observed_size_before": source_size_before,
            "active_file_observed_size_after": source_size_after,
            "whole_growing_file_hash": False,
            "bytes_after_fixed_endpoint_read": False,
            "completed_snapshot_boundary": {
                "ppo_update": completed_update["ppo_update"],
                "global_policy_decisions": completed_update["global_policy_decisions"],
                "files_present_not_loaded": [{"path": str(path), "size_bytes": path.stat().st_size} for path in fixed_artifacts],
            },
            "torch_loaded": False,
        },
        "events": {
            "FL_placed_history_event_tick": placement_tick,
            "P05_to_P06_transition_tick": 2960,
            "first_P06_request": {"global_policy_decision": first_p06["global_policy_decision"], "episode_tick_range": [2961, 2968]},
            "first_P06_current_TOP_support_loss": {
                "request_ordinal": ordinal[loss_tick],
                "global_policy_decision": loss["global_policy_decision"],
                "previous_TOP_endpoint_tick": previous_tick,
                "first_AIR_tick_from_per_physics_counter": first_air_tick,
                "AIR_endpoint_tick": loss_tick,
                "continuous_AIR_proven_ticks_inclusive": [first_air_tick, 3120],
                "continuous_AIR_proven_physics_ticks": 24,
                "continuous_AIR_proven_s": 24 / 120.0,
                "first_exact_recontact_tick": None,
                "first_exact_recontact_unavailable_reason": "ticks3121_to_3125_contact_modes_not_persisted",
                "first_loss_duration_bounds_physics_ticks": [24, 29],
                "first_loss_duration_bounds_s": [24 / 120.0, 29 / 120.0],
                "unpersisted_contact_ticks_inclusive": [3121, 3125],
                "saved_reacquired_TOP_streak_first_tick": saved_reacquired_top_streak_tick,
                "reacquired_TOP_endpoint_tick": reacquired_tick,
                "capture_to_first_AIR_ticks": first_air_tick - placement_tick,
                "capture_to_first_AIR_s": (first_air_tick - placement_tick) / 120.0,
                "P06_transition_to_first_AIR_ticks": first_air_tick - 2960,
                "P06_transition_to_first_AIR_s": (first_air_tick - 2960) / 120.0,
            },
            "latest_fixed_endpoint": {
                "tick": FIXED_END_TICK,
                "p06_request_ordinal": ordinal[FIXED_END_TICK],
                "FL_mode": mode(latest_leg),
                "FL_gap_m": latest_leg["clearance_m"],
                "FL_consecutive_AIR_samples": latest_leg["consecutive_air_samples"],
                "current_AIR_streak_first_tick": latest_air_start,
                "FL_historical_placed_tick": placement_tick,
                "FR_mode": mode(current_leg(latest, "FR")),
            },
        },
        "current_contact_vs_history": {
            "P06_endpoints_through_3496": len(p06),
            "FL_endpoint_modes": dict(p06_modes),
            "FR_TOP_endpoints": sum(is_top(current_leg(row, "FR")) for row in p06),
            "FL_historical_placed_true_endpoints": sum(row["applied_audit"]["semantic_task"]["placed_history"]["FL"] for row in p06),
            "first64_P06_end_tick": first64[-1]["applied_audit"]["physics_tick"],
            "first64_FL_endpoint_modes": dict(first64_modes),
            "FL_endpoint_mode_runs": runs(p06),
            "sampling_limit": "15 Hz endpoint modes; counters prove AIR onset/last continuous streaks but cannot reveal an earlier brief recontact in unsaved ticks3121..3125",
        },
        "B_nominal_descriptive_comparison": {
            "source": str(B_CURRENT),
            "role": b_current["role"],
            "sampling_mode": b_current["sampling_mode"],
            "FL_placed_tick": 2677,
            "P06_entry_tick": 2680,
            "P06_rear_approach_ticks_inclusive": [b_p06["start_tick"], b_p06["observed_end_tick"]],
            "physics_tick_contacts_FL": b_physical_fl,
            "saved_endpoint_contacts_FL": {
                "TOP_verified_bearing": b_endpoint_fl["top_verified_bearing_samples"],
                "support": b_endpoint_fl["support_samples"],
                "AIR": b_endpoint_fl["air_samples"],
                "GROUND": b_endpoint_fl["ground_samples"],
            },
            "FL_gap_max_m": b_p06["physical"]["wheel_bottom_gap_above_top_m"]["FL"]["maximum"],
            "bounded_inference": "current prolonged FL AIR is degraded support retention relative to the preserved successful nominal P06, not a normal short load change observed in that B window",
            "comparison_limit": "existing sealed aggregate only; not time-aligned, state-matched, raw-difference, or a causal policy/channel counterfactual",
        },
        "H_and_cap_transition": {
            "channel_order": list(CHANNELS),
            "units_for_physical_request_and_cap": list(UNITS),
            "rho": first_request["rho"],
            "only_P06_endpoint_with_any_cap_transition_gate": gate_rows,
            "first_P06_gate": select(first_request["cap_transition_gate_full12"]),
            "predecessor_cap": select(first_request["predecessor_cap_full12"]),
            "current_cap": select(first_request["current_cap_full12"]),
            "prior_P05_physical_request": select(transition_request),
            "first_P06_previous_filtered_request": select(first_request["previous_filtered_request_full12"]),
            "prior_request_carry_max_abs_error": prior_request_error,
            "first_P06_H": select(first_request["history_center_full12"]),
            "first_P06_conditional_mean": select(first_request["conditional_mean_full12"]),
            "first_P06_selected_raw": select(first_request["selected_raw_full12"]),
            "first_P06_physical_request": select(first_p06["applied_audit"]["actuator_target_effect_audit"]["policy_headroom_evidence"]["requested_policy_residual_full12"]),
            "loss_H_equals_previous_selected_raw_max_abs_error": loss_H_error,
            "loss_conditional_mean_identity_max_abs_error": conditional_error,
            "loss_bounded_candidate_from_tanh_raw_times_cap": select(loss_bounded_candidate),
            "loss_filtered_physical_request": select(loss_filtered_request),
            "loss_bounded_to_filtered_delta": select(bounded_to_filtered_delta),
            "loss_projection_rate_limited_full12_indices": rate_limited_indices,
            "receiving_x3_active_in_P06": any(row["policy_request"]["receiving_continuation_active"] for row in p06),
        },
        "same_tick_path_checks": {
            "selected_channel_order": list(CHANNELS),
            "units": list(UNITS),
            "final_equals_mapped_baseline_plus_effective_residual_max_abs_error": max_final_error,
            "requested_equals_effective_residual_max_abs_error": max_effective_error,
            "front_native_target_to_canonical_final_max_abs_error": max_native_front_target_error,
            "all_native_tick_dispatches_verified": all_native_verified,
            "note": "Servo final may differ from nominal+request because mapped baseline is slew-limited; wheels use the unchanged 0.3 baseline here.",
        },
        "body_progress": {
            "selected": [p_transition, p_first, p_previous, p_loss, p_reacquired, p_latest],
            "first_P06_to_first_AIR_body_forward_delta_m": p_loss["body_forward_m"] - p_first["body_forward_m"],
            "last_TOP_endpoint_to_first_AIR_endpoint_body_forward_delta_m": p_loss["body_forward_m"] - p_previous["body_forward_m"],
            "last_TOP_endpoint_to_first_AIR_endpoint_phase_progress_delta": p_loss["phase_progress"] - p_previous["phase_progress"],
            "last_TOP_endpoint_to_first_AIR_endpoint_potential_delta": p_loss["task_progress_potential"] - p_previous["task_progress_potential"],
            "first_P06_to_tick3496_body_forward_delta_m": p_latest["body_forward_m"] - p_first["body_forward_m"],
        },
        "selected_endpoints": selected,
        "limits": [
            "Historical placed is an irreversible event flag and is not current TOP support.",
            "Endpoint and consecutive-contact evidence do not prove force, slip, torque, or causal action contribution between saved samples.",
            "This active training file was read only to a fixed complete line inside completed update1509; no later line and no whole-file hash was read.",
            "No model/checkpoint/torch/Isaac load, optimizer operation, production edit, reward conclusion, or training gate was performed.",
        ],
    }

    loss_endpoint = next(item for item in selected if item["endpoint_tick"] == 3104)
    previous_endpoint = next(item for item in selected if item["endpoint_tick"] == 3096)
    reacquired_endpoint = next(item for item in selected if item["endpoint_tick"] == 3128)
    latest_endpoint = next(item for item in selected if item["endpoint_tick"] == 3496)
    report = f"""# Block14：P06 首次 FL 当前支撑丢失固定窗口（只读）

结论：FL 的历史 placed 事件是真实 tick **2957**，P05→P06 在 tick **2960**。P06 第18个学习请求（global **{loss['global_policy_decision']}**，作用 tick 3097–3104）发生第一次当前 TOP 支撑丢失：3096 端点仍为 TOP，3104 端点为 AIR 且 `consecutive_air_samples=8`，所以首个 AIR/失去当前 TOP 的精确物理 tick 是 **3097**。3120 的 `consecutive_air_samples=24` 严格证明连续 AIR 到3120；3128 的 `consecutive_top_samples=3` 只证明当时这段 TOP streak 从3126开始。3121–3125的逐tick接触未落盘，所以首次recontact不能伪造为3126，首次失载持续时间只界定为 **24–29 ticks / 0.2000–0.2417s**。这不是把历史 placed 撤销：窗口内 FL `placed_history=true` 67/67，而当前端点仅 TOP 26/67、AIR 41/67；FR 则 TOP 67/67。

## 固定读取边界

- run：`{RUN.relative_to(ROOT)}`；仅读取仍增长 audit 的前 **437** 条完整行，到 global {rows[-1]['global_policy_decision']} / episode tick 3496 立即停止；未读取其后内容，也没有全文件 hash。
- tick3496 位于已经完成并落盘的 rollout/update1509 边界内（完整边界 global197632）；只检查 rollout/likelihood 文件存在和字节数，未加载 `.pt`、torch、模型或 Isaac。
- 分析窗口是行370–437：P05→P06 过渡端点2960，加67个P06端点2968–3496。前64个P06请求止于tick3472。

## 当前接触与历史事件

|事件/端点|FL 当前状态|FL gap|FL bearing|FR 当前状态|body forward|phase progress|task potential|
|---|---|---:|---:|---|---:|---:|---:|
|placed/P06过渡端点 2960|TOP|{contact(current_leg(transition,'FL'))['clearance_m']*1000:+.3f} mm|{contact(current_leg(transition,'FL'))['bearing_force_n']:.3f} N|TOP|{p_transition['body_forward_m']*1000:+.3f} mm|{p_transition['phase_progress']:.6f}|{p_transition['task_progress_potential']:.6f}|
|首个P06端点 2968|TOP|{contact(current_leg(first_p06,'FL'))['clearance_m']*1000:+.3f} mm|{contact(current_leg(first_p06,'FL'))['bearing_force_n']:.3f} N|TOP|{p_first['body_forward_m']*1000:+.3f} mm|{p_first['phase_progress']:.6f}|{p_first['task_progress_potential']:.6f}|
|最后TOP端点 3096|TOP|{contact(current_leg(previous,'FL'))['clearance_m']*1000:+.3f} mm|{contact(current_leg(previous,'FL'))['bearing_force_n']:.3f} N|TOP|{p_previous['body_forward_m']*1000:+.3f} mm|{p_previous['phase_progress']:.6f}|{p_previous['task_progress_potential']:.6f}|
|首个AIR端点 3104|AIR|{contact(current_leg(loss,'FL'))['clearance_m']*1000:+.3f} mm|0 N|TOP ({contact(current_leg(loss,'FR'))['bearing_force_n']:.3f} N)|{p_loss['body_forward_m']*1000:+.3f} mm|{p_loss['phase_progress']:.6f}|{p_loss['task_progress_potential']:.6f}|
|重获TOP端点 3128|TOP|{contact(current_leg(reacquired,'FL'))['clearance_m']*1000:+.3f} mm|{contact(current_leg(reacquired,'FL'))['bearing_force_n']:.3f} N|TOP|{p_reacquired['body_forward_m']*1000:+.3f} mm|{p_reacquired['phase_progress']:.6f}|{p_reacquired['task_progress_potential']:.6f}|
|固定尾端 3496|AIR|{contact(current_leg(latest,'FL'))['clearance_m']*1000:+.3f} mm|0 N|TOP ({contact(current_leg(latest,'FR'))['bearing_force_n']:.3f} N)|{p_latest['body_forward_m']*1000:+.3f} mm|{p_latest['phase_progress']:.6f}|{p_latest['task_progress_potential']:.6f}|

3104 的 FL 是 `within_top_xy=true`、AIR、force=0，不是落到 ground；FR 仍承载。第一次失载当步，body 仍前送 **{(p_loss['body_forward_m']-p_previous['body_forward_m'])*1000:+.3f} mm**、phase progress **{p_loss['phase_progress']-p_previous['phase_progress']:+.6f}**，但 task potential **{p_loss['task_progress_potential']-p_previous['task_progress_potential']:+.6f}**。首个P06端点到首次AIR端点累计前送 **{(p_loss['body_forward_m']-p_first['body_forward_m'])*1000:+.3f} mm**。到tick3496虽累计前送 **{(p_latest['body_forward_m']-p_first['body_forward_m'])*1000:+.3f} mm**，FL 已连续 AIR 127 physics ticks（从3370到3496）且 gap **{latest_leg['clearance_m']*1000:.3f} mm**；因此“body前送”不能代替“FL当前承载”。

## 首次失载的同拍六通道控制链

顺序固定为 **FL hip / FL knee / FL wheel / FR wheel / RL wheel / RR wheel**；前两项单位deg，四轮单位rad/s。“REQUEST”是策略物理 residual；“actual”是保存的同次 dispatch tracking 前关节位置（转canonical）与决策端点 canonical wheel qd。final逐通道等于 mapped baseline + effective REQUEST（最大误差 {max_final_error:.1e}）。Hip/knee 的 mapped baseline 仍受 source mapper slew，因此不能拿 `nominal + REQUEST` 冒充 final。

|端点|当前FL|nominal|mapped baseline|REQUEST|final|actual|
|---|---|---|---|---|---|---|
|3096|TOP|{fmt(previous_endpoint['control']['nominal'])}|{fmt(previous_endpoint['control']['mapped_baseline_native_plus_controller'])}|{fmt(previous_endpoint['control']['requested_policy_residual'])}|{fmt(previous_endpoint['control']['final_drive_target'])}|{fmt(previous_endpoint['control']['measured_actual'])}|
|3104|AIR|{fmt(loss_endpoint['control']['nominal'])}|{fmt(loss_endpoint['control']['mapped_baseline_native_plus_controller'])}|{fmt(loss_endpoint['control']['requested_policy_residual'])}|{fmt(loss_endpoint['control']['final_drive_target'])}|{fmt(loss_endpoint['control']['measured_actual'])}|
|3128|TOP|{fmt(reacquired_endpoint['control']['nominal'])}|{fmt(reacquired_endpoint['control']['mapped_baseline_native_plus_controller'])}|{fmt(reacquired_endpoint['control']['requested_policy_residual'])}|{fmt(reacquired_endpoint['control']['final_drive_target'])}|{fmt(reacquired_endpoint['control']['measured_actual'])}|
|3496|AIR|{fmt(latest_endpoint['control']['nominal'])}|{fmt(latest_endpoint['control']['mapped_baseline_native_plus_controller'])}|{fmt(latest_endpoint['control']['requested_policy_residual'])}|{fmt(latest_endpoint['control']['final_drive_target'])}|{fmt(latest_endpoint['control']['measured_actual'])}|

在首次失载端点3104：FL hip/knee final 为 **+17.428° / −23.510°**，保存的实际位置为 **+17.210° / −23.420°**；四轮 final 为 **+.485 / +.395 / +.597 / +.182 rad/s**，实际为 **+.485 / +.343 / +.753 / +.094 rad/s**。与3096相比，FL实际hip/knee只变化约 **+0.006° / −0.027°**，接触却从TOP转AIR；这是同步事实，不单独证明哪个控制通道导致失载。

## H / cap transition

首个P06请求 tick2961–2968 的 cap（同一六通道顺序）从 **{fmt(select(first_request['predecessor_cap_full12']))}** 变成 **{fmt(select(first_request['current_cap_full12']))}**；gate 为 `{select(first_request['cap_transition_gate_full12'])}`，即 FL hip/knee、FL/FR wheel gate=true，未改cap的RL/RR wheel为false。上一P05物理REQUEST **{fmt(select(transition_request))}** 被原样记录为 `previous_filtered_request`（最大差 {prior_request_error:.2e}），转换后的 H 是 **{fmt(select(first_request['history_center_full12']))}**。该gate只出现在首个P06请求，之后到3496全部false。

首次失载请求3104没有cap切换：H **{fmt(select(loss_request['history_center_full12']))}** 与上一请求3096的 selected raw逐项相等（最大差 {loss_H_error:.1e}）；在rho=.9下 conditional mean为 **{fmt(select(loss_request['conditional_mean_full12']))}**，本次 selected raw为 **{fmt(select(loss_request['selected_raw_full12']))}**。`tanh(raw)×cap` 的 bounded candidate 是 **{fmt(select(loss_bounded_candidate))}**；随后既有projection把FL wheel按request rate limit从 **{loss_bounded_candidate[8]:+.3f}** 限到上表 **{loss_filtered_request[8]:+.3f} rad/s**，其余五个所列通道没有该差异。也就是说，3097掉载不是cap transition同拍突变；它发生在P06固定cap、常规H递推的第18个请求。P06没有触发只适用于P10–P12且RR历史placed的receiving-wheel ×3 gate。

## 与已保存 B_current 的简短比较

这里只读既有 `currentB_event_windows.json` 汇总，不重扫Recording。B 是 `nominal_without_learned_residual`：FL placed2677、P06 entry2680，同样相隔3ticks；其完整 P06 rear-approach 2680–5160 中，逐physics FL obstacle contact **2481/2481、AIR 0**，保存端点 FL TOP/support **311/311、AIR 0**，FL gap最大仅 **{b_p06['physical']['wheel_bottom_gap_above_top_m']['FL']['maximum']*1000:.4f} mm**。所以当前P06的41/67 AIR端点以及3370–3496连续AIR，不是成功N窗口中观察到的正常短暂载荷变化，可严格称为“相对B的support-retention degradation”。但两次轨迹没有时间对齐或状态匹配，这不是raw差分或因果counterfactual，不能据此单独归因某个policy通道或更新。

## 边界

逐physics consecutive counter允许精确定位首次 AIR tick3097、严格连续AIR到3120，以及3126开始的当前TOP streak；它不能恢复3121–3125未保存的中间接触，因此 `first_exact_recontact_tick=null`。其余表格仍是15 Hz端点。这里没有扭矩、slip、接触点连续轨迹或因果干预；也不覆盖reward/GAE/likelihood（由独立审计负责）。本审计没有加载checkpoint/torch、没有运行Isaac、没有修改生产或训练门禁。详细数值和全部选定端点在同名JSON；重现入口为本脚本。
"""

    if not check_only:
        with JSON_OUT.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        with MD_OUT.open("x", encoding="utf-8") as stream:
            stream.write(report)
    print(json.dumps({
        "status": result["status"],
        "fixed_end": result["source_binding"]["fixed_prefix_end"],
        "first_loss": result["events"]["first_P06_current_TOP_support_loss"],
        "latest": result["events"]["latest_fixed_endpoint"],
        "outputs": [str(JSON_OUT), str(MD_OUT)],
    }, indent=2))


if __name__ == "__main__":
    unknown = set(sys.argv[1:]) - {"--check"}
    if unknown:
        raise SystemExit(f"unsupported arguments: {sorted(unknown)}")
    main(check_only="--check" in sys.argv[1:])
