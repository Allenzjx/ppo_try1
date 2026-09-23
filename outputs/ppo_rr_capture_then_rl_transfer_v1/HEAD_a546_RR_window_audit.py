"""Bounded read-only audit of the sealed a546 RR-capture headless run.

The helper streams each large ledger once, retains only compact P09/selected
evidence, and writes a small JSON/Markdown report.  It does not import torch,
start Isaac, mutate the run, or infer causal credit from correlated motion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / "runs" / "ppo_rr_capture_then_rl_transfer_v1" / "validation" / (
    "20260923T0310001450430Z_ga54678ceef58_c5e1fa8c5a5f42089a96c3cf709bf324"
)
OUT_DIR = Path(__file__).resolve().parent
DEFAULT_JSON = OUT_DIR / "HEAD_a546_RR_window_audit.json"
DEFAULT_MD = OUT_DIR / "HEAD_a546_RR_window_audit.md"

WHEEL_KEYS = (
    "front_left_ankle", "front_right_ankle", "rear_left_ankle", "rear_right_ankle",
)
WHEEL_LABELS = ("FL", "FR", "RL", "RR")
CONTACT_KEYS = (
    "front_left_wheel", "front_right_wheel", "rear_left_wheel", "rear_right_wheel",
)
NATIVE_SIGNS = (-1.0, 1.0, -1.0, 1.0)
RR_HIP_INDEX = 6
RR_KNEE_INDEX = 7
WHEEL_START = 8


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    require(isinstance(value, dict), f"{path} must contain an object")
    return value


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def finite(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def vector(value: Any, size: int, name: str) -> list[float]:
    require(isinstance(value, list) and len(value) == size, f"invalid {name}")
    result = [finite(item) for item in value]
    require(all(item is not None for item in result), f"non-finite {name}")
    return [float(item) for item in result]


def rows(path: Path, *, skip_lines: int = 0) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("rb") as stream:
        for index, raw in enumerate(stream, 1):
            require(raw.endswith(b"\n"), f"partial JSONL row in {path.name}")
            if index <= skip_lines:
                continue
            value = json.loads(raw)
            require(isinstance(value, dict), f"non-object row in {path.name}")
            yield index, value


def nested_artifact_record(manifest: dict[str, Any], filename: str) -> dict[str, Any] | None:
    for key in ("artifacts", "files", "outputs"):
        group = manifest.get(key)
        if isinstance(group, dict):
            record = group.get(filename)
            if isinstance(record, dict):
                return record
    return None


def q_conjugate_rotate(q_wxyz: list[float], v: list[float]) -> list[float]:
    """Rotate a world vector into the body frame with a unit wxyz quaternion."""
    w, x, y, z = q_wxyz
    norm = math.sqrt(w*w + x*x + y*y + z*z)
    require(norm > 0.0, "zero base quaternion")
    w, x, y, z = (w/norm, x/norm, y/norm, z/norm)
    # R(q)^T v, written explicitly to avoid any numerical dependency.
    return [
        (1 - 2*y*y - 2*z*z)*v[0] + (2*x*y + 2*w*z)*v[1] + (2*x*z - 2*w*y)*v[2],
        (2*x*y - 2*w*z)*v[0] + (1 - 2*x*x - 2*z*z)*v[1] + (2*y*z + 2*w*x)*v[2],
        (2*x*z + 2*w*y)*v[0] + (2*y*z - 2*w*x)*v[1] + (1 - 2*x*x - 2*y*y)*v[2],
    ]


def physical_compact(row: dict[str, Any]) -> dict[str, Any]:
    tick = row.get("physics_tick")
    require(type(tick) is int, "physical row lacks tick")
    obstacle = row.get("obstacle") or {}
    base = row.get("base") or {}
    base_pos = vector(base.get("position_w_m"), 3, "base position")
    base_q = vector(base.get("orientation_wxyz"), 4, "base orientation")
    wheels = row.get("wheels") or {}
    joints = row.get("joints") or {}
    contacts = row.get("contacts") or {}
    bodies = row.get("bodies") or {}
    rr_wheel = wheels.get("rear_right_ankle") or {}
    center = vector(rr_wheel.get("center_w_m"), 3, "RR wheel center")
    bottom = vector(rr_wheel.get("bottom_w_m"), 3, "RR wheel bottom")
    rel_world = [center[i] - base_pos[i] for i in range(3)]
    center_body = q_conjugate_rotate(base_q, rel_world)

    wheel_actual = []
    contact_owner = []
    for wheel_key, contact_key, label in zip(WHEEL_KEYS, CONTACT_KEYS, WHEEL_LABELS):
        wheel = wheels.get(wheel_key) or {}
        contact = contacts.get(contact_key) or {}
        ground = contact.get("ground") or {}
        obstacle_contact = contact.get("obstacle") or {}
        wheel_actual.append(float(finite(wheel.get("velocity_rad_s")) or 0.0))
        contact_owner.append({
            "wheel": label,
            "contact_class": contact.get("contact_class"),
            "ground_active": ground.get("active"),
            "ground_normal_force_n": finite(ground.get("normal_force_n")),
            "obstacle_active": obstacle_contact.get("active"),
            "obstacle_normal_force_n": finite(obstacle_contact.get("normal_force_n")),
            "obstacle_other_body": obstacle_contact.get("other_body"),
        })
    rr_upper = bodies.get("rear_right_upper") or {}
    rr_upper_pos = vector(rr_upper.get("position_w_m"), 3, "RR upper position")
    actual = vector(row.get("actual_full12"), 12, "actual_full12")
    rr_contact = contacts.get("rear_right_wheel") or {}
    return {
        "tick": tick,
        "time_s": finite(row.get("simulation_time_s")),
        "base_world_z_m": base_pos[2],
        "rr_hip_link_world_z_m": rr_upper_pos[2],
        "rr_wheel_center_world_xz_m": [center[0], center[2]],
        "rr_wheel_center_body_xz_m": [center_body[0], center_body[2]],
        "rr_gap_mm": 1000.0 * (bottom[2] - float(obstacle.get("top_z_m"))),
        "rr_front_distance_mm": 1000.0 * (center[0] - float(obstacle.get("front_x_m"))),
        "rr_hip_actual_deg": finite((joints.get("rear_right_hip") or {}).get("position_deg")),
        "rr_knee_actual_deg": finite((joints.get("rear_right_knee") or {}).get("position_deg")),
        "wheel_actual_canonical_rad_s": actual[WHEEL_START:],
        "wheel_actual_named_rad_s": dict(zip(WHEEL_LABELS, wheel_actual)),
        "wheel_contact_owner": contact_owner,
        "rr_contact_class": rr_contact.get("contact_class"),
    }


def native_compact(row: dict[str, Any]) -> dict[str, Any]:
    tick = row.get("episode_physics_tick")
    require(type(tick) is int, "native row lacks episode tick")
    audit = row.get("native_audit") or {}
    require(audit.get("verified") is True, "native row is not verified")
    nominal = vector(row.get("nominal_full12"), 12, "nominal_full12")
    raw = vector(audit.get("raw_policy_action_full12"), 12, "raw policy action")
    effective = vector(audit.get("projected_residual_full12"), 12, "projected residual")
    mapper = vector(audit.get("native_drive_target_full12"), 12, "mapper target")
    bias = vector(audit.get("combined_post_mapper_bias_full12"), 12, "post-mapper bias")
    final = [a + b for a, b in zip(mapper, bias)]
    native_targets = vector((audit.get("actual_native_targets") or {}).get("wheel_velocity_rad_s"),
                            4, "native wheel target")
    evidence = audit.get("rr_capture_assist_evidence") or {}
    state = evidence.get("state_after") or {}
    context = evidence.get("context") or {}
    servo_final = vector(evidence.get("final_servo_target_deg"), 8, "final servo target")
    require(all(math.isclose(final[index], servo_final[index], abs_tol=1e-9, rel_tol=0.0)
                for index in range(8)), "native final servo reconstruction differs")
    expected_native = [NATIVE_SIGNS[i] * final[WHEEL_START+i] for i in range(4)]
    require(all(math.isclose(native_targets[i], expected_native[i], abs_tol=2e-6, rel_tol=0.0)
                for i in range(4)), "wheel canonical/native axis mapping differs")
    return {
        "tick": tick,
        "dispatch_physics_tick": audit.get("physics_tick"),
        "phase": audit.get("source_phase_id"),
        "source_observation_tick": context.get("source_observation_tick"),
        "pre_gap_mm": (None if finite(context.get("gap_m")) is None
                       else 1000.0 * float(context.get("gap_m"))),
        "pre_within_top_xy": context.get("within_top_xy"),
        "pre_qualified": context.get("qualified_RR"),
        "pre_crossed": context.get("crossed_RR"),
        "pre_contact": ("TOP" if context.get("top_surface_contact") else
                        "GROUND" if context.get("ground_contact") else
                        "AIR" if context.get("air") else
                        "OBSTACLE" if context.get("obstacle_pair_active") else "UNKNOWN"),
        "assist_mode": state.get("mode_name"),
        "assist_reason": state.get("reason"),
        "assist_active": bool(state.get("active")),
        "assist_initialized": bool(state.get("initialized")),
        "assist_window_start_gap_mm": 1000.0 * float(state.get("window_start_gap_m") or 0.0),
        "assist_hip_entry_deg": finite(state.get("hip_entry_deg")),
        "assist_hip_target_deg": finite(state.get("hip_target_deg")),
        "assist_travel_used_deg": finite(state.get("travel_used_deg")),
        "assist_knee_hold_deg": finite(state.get("knee_hold_deg")),
        "assist_blocked_code": finite(state.get("blocked_reason")),
        "rr_hip_final_deg": servo_final[RR_HIP_INDEX],
        "rr_knee_final_deg": servo_final[RR_KNEE_INDEX],
        "nominal_N_wheels_canonical_rad_s": nominal[WHEEL_START:],
        "raw_policy_wheels_unitless": raw[WHEEL_START:],
        "effective_residual_wheels_canonical_rad_s": effective[WHEEL_START:],
        "mapper_prebias_wheels_canonical_rad_s": mapper[WHEEL_START:],
        "final_wheels_canonical_rad_s": final[WHEEL_START:],
        "native_wheel_targets_physical_axis_rad_s": native_targets,
        "rr_owner_indices": evidence.get("owner_indices") or [],
    }


def decision_compact(row: dict[str, Any]) -> dict[str, Any]:
    audit = row.get("actuator_target_effect_audit") or {}
    evidence = audit.get("rr_capture_assist_evidence") or {}
    state = evidence.get("state_after") or {}
    context = evidence.get("context") or {}
    nominal_diag = (row.get("semantic_task") or {}).get("nominal_provider_diagnostics") or {}
    layers = ((nominal_diag.get("source_partial_order") or {}).get("layers") or [])
    p09_layer = next((value for value in layers if value.get("stage") == "P09"), None)
    return {
        "tick": row.get("physics_tick"),
        "time_s": finite(row.get("sim_time_s")),
        "phase": row.get("phase_id"),
        "context_source_tick": context.get("source_observation_tick"),
        "context_gap_mm": (None if finite(context.get("gap_m")) is None
                           else 1000.0 * float(context.get("gap_m"))),
        "context_within_top_xy": context.get("within_top_xy"),
        "context_crossed": context.get("crossed_RR"),
        "assist_mode": state.get("mode_name"),
        "assist_reason": state.get("reason"),
        "assist_window_start_gap_mm": 1000.0 * float(state.get("window_start_gap_m") or 0.0),
        "rr_hip_final_deg": (evidence.get("final_servo_target_deg") or [None] * 8)[6],
        "rr_knee_final_deg": (evidence.get("final_servo_target_deg") or [None] * 8)[7],
        "p09_late_group": p09_layer,
        "termination_reason": row.get("termination_reason"),
        "termination_source": (row.get("semantic_task") or {}).get("termination_source"),
    }


def mode_segments(native: list[dict[str, Any]]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for row in native:
        key = (row["assist_mode"], row["assist_reason"], row["assist_active"])
        if not segments or tuple(segments[-1]["key"]) != key:
            segments.append({
                "key": list(key), "start_tick": row["tick"], "end_tick": row["tick"],
                "start_source_observation_tick": row["source_observation_tick"],
                "end_source_observation_tick": row["source_observation_tick"],
            })
        else:
            segments[-1]["end_tick"] = row["tick"]
            segments[-1]["end_source_observation_tick"] = row["source_observation_tick"]
    return segments


def combine_snapshot(tick: int, physical: dict[int, dict[str, Any]],
                     native: dict[int, dict[str, Any]], label: str) -> dict[str, Any]:
    p = physical.get(tick)
    n = native.get(tick)
    require(p is not None and n is not None, f"missing joined snapshot tick {tick}")
    return {
        "label": label,
        "tick": tick,
        "time_s": p["time_s"],
        "phase": n["phase"],
        "geometry": {
            key: p[key] for key in (
                "rr_gap_mm", "rr_front_distance_mm", "base_world_z_m",
                "rr_hip_link_world_z_m", "rr_wheel_center_world_xz_m",
                "rr_wheel_center_body_xz_m", "rr_contact_class",
            )
        },
        "rr_servo": {
            "hip_final_deg": n["rr_hip_final_deg"],
            "hip_actual_deg": p["rr_hip_actual_deg"],
            "knee_final_deg": n["rr_knee_final_deg"],
            "knee_actual_deg": p["rr_knee_actual_deg"],
            "assist_mode": n["assist_mode"],
            "assist_reason": n["assist_reason"],
            "assist_hip_target_deg": n["assist_hip_target_deg"],
            "assist_knee_hold_deg": n["assist_knee_hold_deg"],
            "assist_travel_used_deg": n["assist_travel_used_deg"],
        },
        "four_wheels": {
            "order": list(WHEEL_LABELS),
            "N_nominal_canonical_rad_s": n["nominal_N_wheels_canonical_rad_s"],
            "raw_policy_unitless": n["raw_policy_wheels_unitless"],
            "effective_residual_canonical_rad_s": n["effective_residual_wheels_canonical_rad_s"],
            "mapper_prebias_canonical_rad_s": n["mapper_prebias_wheels_canonical_rad_s"],
            "final_canonical_target_rad_s": n["final_wheels_canonical_rad_s"],
            "native_physical_axis_target_rad_s": n["native_wheel_targets_physical_axis_rad_s"],
            "measured_actual_canonical_rad_s": p["wheel_actual_canonical_rad_s"],
            "contact_owner": p["wheel_contact_owner"],
        },
    }


def delta(after: dict[str, Any], before: dict[str, Any], key: str) -> float:
    return float(after[key]) - float(before[key])


def write_atomic_new(path: Path, text: str) -> None:
    require(not path.exists(), f"refusing to overwrite {path}")
    temporary = path.with_name(path.name + ".partial")
    require(not temporary.exists(), f"partial output already exists: {temporary}")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()
    run = args.run.resolve(strict=True)

    manifest_path = run / "run_manifest.json"
    evaluation_path = run / "evaluation_manifest.json"
    run_manifest = read_json(manifest_path)
    evaluation = read_json(evaluation_path)
    require(run_manifest.get("status") in ("DIAGNOSTIC_FAILURE", "FAILED", "COMPLETE") or
            run_manifest.get("lifecycle") == "DIAGNOSTIC_FAILURE",
            "run manifest is not a closed diagnostic run")

    stage_path = run / "stage_transition_evidence.jsonl"
    residual_path = run / "residual_and_projection_audit.jsonl"
    native_path = run / "native_tick_audit.jsonl"
    physical_path = run / "physical_observations.jsonl"
    for path in (stage_path, residual_path, native_path, physical_path):
        require(path.is_file() and path.stat().st_size > 0, f"missing {path.name}")

    stage_rows = [row for _, row in rows(stage_path)]
    require(stage_rows, "stage evidence is empty")
    final_history = stage_rows[-1].get("physical_history") or {}
    event_ticks = final_history.get("event_ticks") or {}
    fl_first_cross = (event_ticks.get("front_edge_crossed") or {}).get("FL")
    fl_first_placed = (event_ticks.get("placed") or {}).get("FL")
    rr_qualified = (event_ticks.get("active_lift") or {}).get("RR")
    rr_cross = (event_ticks.get("front_edge_crossed") or {}).get("RR")
    require((fl_first_cross, fl_first_placed, rr_qualified, rr_cross) == (3544, 6150, 8553, 9520),
            "sealed event ticks differ from the expected run")

    decisions: list[dict[str, Any]] = []
    terminal_decision: dict[str, Any] | None = None
    for _, row in rows(residual_path, skip_lines=1000):
        compact = decision_compact(row)
        terminal_decision = compact
        if row.get("phase_id") == "P09":
            decisions.append(compact)
    require(decisions and terminal_decision is not None, "P09 decision audit is missing")
    late = next((row for row in decisions if
                 isinstance(row.get("p09_late_group"), dict) and
                 row["p09_late_group"].get("late_group_start_tick") is not None), None)
    require(late is not None and late["p09_late_group"].get("late_group_start_tick") == 9520,
            "P09 late group did not start at sealed RR cross")

    native_list: list[dict[str, Any]] = []
    native_map: dict[int, dict[str, Any]] = {}
    for _, row in rows(native_path, skip_lines=8200):
        tick = row.get("episode_physics_tick")
        if type(tick) is not int or tick < 8288:
            continue
        compact = native_compact(row)
        native_list.append(compact)
        native_map[tick] = compact
    require(native_list and native_list[-1]["tick"] == 12484, "native audit endpoint differs")

    physical_map: dict[int, dict[str, Any]] = {}
    fl_positive_crossings: list[int] = []
    prior_fl_front: float | None = None
    for _, row in rows(physical_path, skip_lines=2200):
        tick = row.get("physics_tick")
        if type(tick) is not int:
            continue
        compact = physical_compact(row)
        physical_map[tick] = compact
        # Detect every measured negative/non-positive -> positive FL wheel-center crossing.
        fl_center = vector((row.get("wheels") or {}).get("front_left_ankle", {}).get("center_w_m"),
                           3, "FL center")
        front = 1000.0 * (fl_center[0] - float((row.get("obstacle") or {}).get("front_x_m")))
        if prior_fl_front is not None and prior_fl_front <= 0.0 < front:
            fl_positive_crossings.append(tick)
        prior_fl_front = front
    require(physical_map and max(physical_map) == 12484, "physical endpoint differs")
    require(fl_first_cross in fl_positive_crossings, "FL historical crossing lacks measured crossing")
    fl_recrosses = [tick for tick in fl_positive_crossings if tick > fl_first_cross]

    p09_entry = 8296
    first_within_native = next(row for row in native_list
                               if row["pre_within_top_xy"] is True and row["phase"] == "P09")
    first_initialized = next(row for row in native_list if row["assist_initialized"])
    require(first_initialized["tick"] == 9521 and
            math.isclose(first_initialized["assist_window_start_gap_mm"], 28.662031052807972,
                         abs_tol=1e-9, rel_tol=0.0), "assist anchor differs")
    segments = mode_segments(native_list)
    tracking_segment = next(segment for segment in segments
                            if segment["key"][0:2] == ["BLOCKED", "waiting_actual_tracking"])
    final_block_segment = next(segment for segment in segments
                               if segment["key"][0:2] == ["BLOCKED", "gap_not_improving"])

    anchor_tick = first_initialized["source_observation_tick"]
    rr_window_physical = [physical_map[tick] for tick in sorted(physical_map)
                          if anchor_tick <= tick <= 12484]
    peak = max(rr_window_physical, key=lambda row: row["rr_gap_mm"])
    after_peak = [row for row in rr_window_physical if row["tick"] > peak["tick"]]
    post_peak_min = min(after_peak, key=lambda row: row["rr_gap_mm"])
    endpoint = physical_map[12484]

    pre_cross_window = [physical_map[tick] for tick in sorted(physical_map)
                        if 8296 <= tick <= rr_cross]
    pre_cross_peak = max(pre_cross_window, key=lambda row: row["rr_gap_mm"])

    key_ticks: list[tuple[str, int]] = [
        ("P09_entry", p09_entry),
        ("RR_qualified", rr_qualified),
        ("RR_pre_cross_gap_peak", pre_cross_peak["tick"]),
        ("geometry_first_observed_within_top_xy", first_within_native["source_observation_tick"]),
        ("P09_late_group_and_RR_cross", rr_cross),
        ("assist_anchor", anchor_tick),
        ("first_tracking_block", tracking_segment["start_tick"]),
        ("descent_after_tracking", tracking_segment["end_tick"] + 1),
        ("post_anchor_gap_peak", peak["tick"]),
        ("first_gap_not_improving_block", final_block_segment["start_tick"]),
        ("post_peak_minimum", post_peak_min["tick"]),
        ("terminal", 12484),
    ]
    # Deduplicate without losing the semantic label list for coincident events.
    labels_by_tick: dict[int, list[str]] = {}
    for label, tick in key_ticks:
        labels_by_tick.setdefault(int(tick), []).append(label)
    snapshots = [combine_snapshot(tick, physical_map, native_map, "+".join(labels))
                 for tick, labels in sorted(labels_by_tick.items())]

    coarse_ticks = set(labels_by_tick)
    coarse_ticks.update(range(8400, 12485, 120))  # 1 s P09 trajectory.
    coarse_ticks.update(range(9480, 9841, 20))    # 1/6 s around geometry/assist takeover.
    coarse_curve = []
    for tick in sorted(coarse_ticks):
        if tick not in physical_map or tick not in native_map:
            continue
        p, n = physical_map[tick], native_map[tick]
        coarse_curve.append({
            "tick": tick,
            "time_s": p["time_s"],
            "rr_gap_mm": p["rr_gap_mm"],
            "rr_front_distance_mm": p["rr_front_distance_mm"],
            "base_world_z_m": p["base_world_z_m"],
            "rr_hip_link_world_z_m": p["rr_hip_link_world_z_m"],
            "rr_wheel_center_world_z_m": p["rr_wheel_center_world_xz_m"][1],
            "rr_wheel_center_body_z_m": p["rr_wheel_center_body_xz_m"][1],
            "rr_hip_final_deg": n["rr_hip_final_deg"],
            "rr_hip_actual_deg": p["rr_hip_actual_deg"],
            "rr_knee_final_deg": n["rr_knee_final_deg"],
            "rr_knee_actual_deg": p["rr_knee_actual_deg"],
            "assist_mode": n["assist_mode"],
            "assist_reason": n["assist_reason"],
            "rr_contact_class": p["rr_contact_class"],
        })

    peak_to_min = {
        "peak_tick": peak["tick"],
        "peak_time_s": peak["time_s"],
        "peak_gap_mm": peak["rr_gap_mm"],
        "post_peak_min_tick": post_peak_min["tick"],
        "post_peak_min_time_s": post_peak_min["time_s"],
        "post_peak_min_gap_mm": post_peak_min["rr_gap_mm"],
        "gap_change_mm": delta(post_peak_min, peak, "rr_gap_mm"),
        "base_world_z_change_mm": 1000.0 * delta(post_peak_min, peak, "base_world_z_m"),
        "rr_hip_link_world_z_change_mm": 1000.0 * delta(post_peak_min, peak, "rr_hip_link_world_z_m"),
        "rr_wheel_center_world_z_change_mm": 1000.0 * (
            post_peak_min["rr_wheel_center_world_xz_m"][1] -
            peak["rr_wheel_center_world_xz_m"][1]),
        "rr_wheel_center_body_z_change_mm": 1000.0 * (
            post_peak_min["rr_wheel_center_body_xz_m"][1] -
            peak["rr_wheel_center_body_xz_m"][1]),
        "hip_actual_change_deg": delta(post_peak_min, peak, "rr_hip_actual_deg"),
        "knee_actual_change_deg": delta(post_peak_min, peak, "rr_knee_actual_deg"),
    }
    entry_gap = first_initialized["assist_window_start_gap_mm"]
    conclusions = {
        "entry_baseline_can_miss_peak_relative_descent": peak_to_min["gap_change_mm"] < 0.0,
        "post_peak_min_better_than_entry_baseline": post_peak_min["rr_gap_mm"] < entry_gap,
        "capture_or_top_bearing_observed": any(
            row["rr_contact_class"] == "TOP" for row in rr_window_physical),
        "interpretation": (
            "A local post-peak RR-gap reduction is measured, so an entry-only baseline does not "
            "describe that local improvement. The gap never returns to the low entry baseline and "
            "no TOP capture occurs. The simultaneous body/link/wheel motion and pre-assist late-group "
            "knee change prevent attributing the outcome causally to negative hip motion alone."
        ),
    }

    file_records = {}
    for path in (stage_path, residual_path, native_path, physical_path):
        record = nested_artifact_record(evaluation, path.name) or nested_artifact_record(run_manifest, path.name)
        file_records[path.name] = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sealed_manifest_record": record,
            "note": "large ledger not rehashed by this bounded post-hoc audit",
        }

    report = {
        "schema": "wlr50_clean.head_a546_RR_window_readonly_audit.v1",
        "run": str(run),
        "run_manifest": str(manifest_path.resolve()),
        "run_manifest_sha256": sha256(manifest_path),
        "evaluation_manifest": str(evaluation_path.resolve()),
        "evaluation_manifest_sha256": sha256(evaluation_path),
        "sealed_files": file_records,
        "task_result": {
            "terminal_tick": terminal_decision["tick"],
            "terminal_time_s": terminal_decision["time_s"],
            "terminal_phase": terminal_decision["phase"],
            "termination_reason": terminal_decision["termination_reason"],
            "termination_source": terminal_decision["termination_source"],
            "physical_success": False,
        },
        "historical_events": {
            "FL_first_cross_tick": fl_first_cross,
            "FL_measured_positive_crossings": fl_positive_crossings,
            "FL_measured_recross_ticks_after_first": fl_recrosses,
            "FL_first_placed_tick": fl_first_placed,
            "RR_qualified_tick": rr_qualified,
            "RR_cross_tick": rr_cross,
            "P09_entry_tick": p09_entry,
            "P09_late_group_start_tick": late["p09_late_group"]["late_group_start_tick"],
            "geometry_first_observed_within_top_xy_source_tick":
                first_within_native["source_observation_tick"],
            "assist_anchor_source_tick": anchor_tick,
            "assist_window_start_gap_mm": entry_gap,
        },
        "assist_mode_segments": [segment for segment in segments
                                 if segment["end_tick"] >= 9500],
        "pre_cross_gap_peak": {
            "tick": pre_cross_peak["tick"], "time_s": pre_cross_peak["time_s"],
            "gap_mm": pre_cross_peak["rr_gap_mm"],
        },
        "post_anchor_peak_to_minimum": peak_to_min,
        "key_snapshots": snapshots,
        "coarse_curve": coarse_curve,
        "conclusions": conclusions,
        "scope_limits": {
            "same_run_video_exists": False,
            "video_synthesized": False,
            "causal_counterfactual_claimed": False,
            "negative_hip_motion_claimed_as_independent_cause": False,
            "data_read_mode": "sealed JSONL streaming; compact retained rows only",
        },
    }

    # A compact human-readable table; complete channel vectors remain in JSON.
    md = [
        "# HEAD a546 sealed RR-window audit",
        "",
        f"Run: `{run}`",
        "",
        "## Result",
        "",
        f"- Terminal: tick {terminal_decision['tick']} / {terminal_decision['time_s']:.6f}s / "
        f"{terminal_decision['phase']} / `{terminal_decision['termination_reason']}` "
        f"(`{terminal_decision['termination_source']}`).",
        f"- FL first cross `{fl_first_cross}`, measured recrosses after it `{fl_recrosses}`, "
        f"first placement `{fl_first_placed}`.",
        f"- RR qualified `{rr_qualified}`, crossed `{rr_cross}`, never captured TOP.",
        "- This headless run has no same-run video; none was synthesized.",
        "",
        "## Entry, peak, and decline",
        "",
        f"- P09 late group and RR cross coincide at tick `{rr_cross}`. The assist anchors on "
        f"source tick `{anchor_tick}` with gap `{entry_gap:.6f}` mm.",
        f"- Post-anchor gap peak: `{peak['rr_gap_mm']:.6f}` mm at tick `{peak['tick']}` "
        f"({peak['time_s']:.6f}s).",
        f"- Later minimum: `{post_peak_min['rr_gap_mm']:.6f}` mm at tick "
        f"`{post_peak_min['tick']}` ({post_peak_min['time_s']:.6f}s), a "
        f"`{peak_to_min['gap_change_mm']:.6f}` mm peak-relative change.",
        f"- Endpoint: `{endpoint['rr_gap_mm']:.6f}` mm at tick 12484. The later minimum is "
        f"`{post_peak_min['rr_gap_mm'] - entry_gap:+.6f}` mm relative to the entry baseline.",
        "- Therefore the low entry baseline misses a real *peak-relative* decline, but that "
        "decline never beats entry and is not evidence of capture or causal credit.",
        "",
        "## Takeover timing",
        "",
        f"- First current-XY observation source tick: "
        f"`{first_within_native['source_observation_tick']}`.",
        f"- Initial tracking block: ticks `{tracking_segment['start_tick']}–"
        f"{tracking_segment['end_tick']}` (`waiting_actual_tracking`).",
        f"- First terminal capture block: tick `{final_block_segment['start_tick']}` "
        "(`gap_not_improving`).",
        "- At tick 9512, knee final/actual are about −45.37/−45.20°. At tick 9520, "
        "before assist anchoring, they are −51.611/−46.111°. Thus the P09 late group changed "
        "the final knee request before the hip-only assist owned its channel.",
        "",
        "## Peak-to-minimum geometry deltas",
        "",
        "| quantity | delta |",
        "|---|---:|",
        f"| RR gap | {peak_to_min['gap_change_mm']:.6f} mm |",
        f"| base world z | {peak_to_min['base_world_z_change_mm']:.6f} mm |",
        f"| RR hip-link world z | {peak_to_min['rr_hip_link_world_z_change_mm']:.6f} mm |",
        f"| RR wheel-center world z | {peak_to_min['rr_wheel_center_world_z_change_mm']:.6f} mm |",
        f"| RR wheel-center body z | {peak_to_min['rr_wheel_center_body_z_change_mm']:.6f} mm |",
        f"| RR hip actual | {peak_to_min['hip_actual_change_deg']:.6f} deg |",
        f"| RR knee actual | {peak_to_min['knee_actual_change_deg']:.6f} deg |",
        "",
        "## Key snapshots",
        "",
        "| label | tick | t(s) | gap(mm) | front(mm) | hip final/actual | knee final/actual | assist | contact |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for snapshot in snapshots:
        g, s = snapshot["geometry"], snapshot["rr_servo"]
        md.append(
            f"| {snapshot['label']} | {snapshot['tick']} | {snapshot['time_s']:.3f} | "
            f"{g['rr_gap_mm']:.3f} | {g['rr_front_distance_mm']:.3f} | "
            f"{s['hip_final_deg']:.3f}/{s['hip_actual_deg']:.3f} | "
            f"{s['knee_final_deg']:.3f}/{s['knee_actual_deg']:.3f} | "
            f"{s['assist_mode']}:{s['assist_reason']} | {g['rr_contact_class']} |"
        )
    md.extend([
        "",
        "The JSON report contains, for every key snapshot, base/hip-link heights, RR wheel "
        "center world and body-relative x/z, and all four wheels' nominal N, raw policy sample, "
        "effective residual, mapper target, final canonical target, native-axis target, measured "
        "actual velocity, and contact owner/forces.",
        "",
        "## Interpretation boundary",
        "",
        conclusions["interpretation"],
    ])

    write_atomic_new(args.json.resolve(), json.dumps(report, indent=2, ensure_ascii=False,
                                                     allow_nan=False) + "\n")
    write_atomic_new(args.markdown.resolve(), "\n".join(md) + "\n")
    print(json.dumps({
        "json": str(args.json.resolve()),
        "markdown": str(args.markdown.resolve()),
        "anchor_gap_mm": entry_gap,
        "peak": {"tick": peak["tick"], "time_s": peak["time_s"],
                 "gap_mm": peak["rr_gap_mm"]},
        "post_peak_min": {"tick": post_peak_min["tick"], "time_s": post_peak_min["time_s"],
                          "gap_mm": post_peak_min["rr_gap_mm"]},
        "first_gap_not_improving_tick": final_block_segment["start_tick"],
        "FL_recross_ticks": fl_recrosses,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
