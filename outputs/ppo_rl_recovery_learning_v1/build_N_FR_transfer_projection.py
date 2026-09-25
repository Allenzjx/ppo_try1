"""Derive a fixed FR-receiver COM projection from the existing 21-row N table.

This reads only the already-produced compact JSON.  It does not rescan the
hundreds-of-MB sealed logs, interpolate missing ticks, or run physics/model code.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "successful_N_P09late_P12_event_response.json"
OUT_JSON = HERE / "N_FR_transfer_projection.json"
OUT_MD = HERE / "N_FR_transfer_projection.md"

KEEP = (
    "PRE_LATE_RR_LOADED_CONTEXT",
    "P09_LATE_ATOMIC_FL_RL_LAUNCH",
    "P09_LATE_EARLY_RESPONSE",
    "P10_RR_KNEE_POSITIVE_ENDPOINT",
    "P11_RESPONSE_FR_AIR_RR_LOADED",
    "P12_RL_KNEE_POSITIVE_AND_FR_HIP_RESPONSE",
    "P12_FOUR_WHEEL_REVERSE_BEGIN",
    "RL_QUALIFIED_EVENT",
    "REVERSE_AND_LOAD_RESPONSE",
    "RR_SHORT_AIR_SNAPSHOT",
    "FIRST_REVERSE_STOP",
    "SECOND_STOP_BEFORE_RL_CROSS",
    "RL_FRONT_EDGE_CROSSED_EVENT",
    "RL_PLACED_EVENT",
    "RL_CAPTURE_RESPONSE",
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _fr_support_point(row: dict[str, Any]) -> list[float] | None:
    support = row.get("support") or {}
    bodies = support.get("active_bodies")
    points = support.get("support_points_w_m")
    if not isinstance(bodies, list) or not isinstance(points, list):
        return None
    matches = [point for body, point in zip(bodies, points, strict=True)
               if body == "front_right_wheel"]
    if len(matches) != 1 or len(matches[0]) != 3:
        return None
    return [float(value) for value in matches[0]]


def build() -> dict[str, Any]:
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    rows = {row["event"]: row for row in source["rows"]}
    start = rows["PRE_LATE_RR_LOADED_CONTEXT"]
    com0 = [float(value) for value in start["center_of_mass_position_w_m"]]
    fr = _fr_support_point(start)
    if fr is None:
        direction = None
        norm = None
    else:
        delta = [fr[0] - com0[0], fr[1] - com0[1]]
        norm = math.hypot(*delta)
        direction = None if norm <= 0.0 else [delta[0] / norm, delta[1] / norm]
    if direction is None:
        raise RuntimeError("already-extracted start row lacks a unique FR wheel support xy")
    if (start["contacts"]["RR"]["total_normal_force_n"] is None or
            start["contacts"]["RR"]["total_normal_force_n"] <= 0.0):
        raise RuntimeError("selected transfer start is not actually RR-loaded")

    selected = []
    for name in KEEP:
        row = rows[name]
        com = [float(value) for value in row["center_of_mass_position_w_m"]]
        displacement = [com[index] - com0[index] for index in range(3)]
        velocity = row.get("center_of_mass_velocity_w_m_s") or [None, None, None]
        projected = displacement[0] * direction[0] + displacement[1] * direction[1]
        projected_velocity = (None if any(value is None for value in velocity[:2]) else
                              float(velocity[0]) * direction[0] +
                              float(velocity[1]) * direction[1])
        rl_event = ("QUALIFIED" if name == "RL_QUALIFIED_EVENT" else
                    "FRONT_EDGE_CROSSED" if name == "RL_FRONT_EDGE_CROSSED_EVENT" else
                    "PLACED" if name == "RL_PLACED_EVENT" else None)
        selected.append({
            "event": name,
            "episode_physics_tick": row["episode_physics_tick"],
            "time_s": row["time_s"],
            "source_phase": row["source_phase"],
            "COM_position_w_m": com,
            "COM_displacement_from_start_m": {
                "x": displacement[0], "y": displacement[1], "z": displacement[2],
                "fixed_FR_receiver_projection": projected,
            },
            "COM_velocity_w_m_s": velocity,
            "COM_velocity_fixed_FR_receiver_projection_m_s": projected_velocity,
            "yaw_displacement_rad": None,
            "yaw_rate_rad_s": row["base_angular_velocity_w_rad_s"][2],
            "yaw_note": "base orientation/yaw was not retained in the compact source table; yaw rate is logged, but sparse rows are not integrated",
            "measured_normal_load_n": {
                "FL": row["contacts"]["FL"]["total_normal_force_n"],
                "RR": row["contacts"]["RR"]["total_normal_force_n"],
            },
            "RL": {
                "event": rl_event,
                "contact_class": row["contacts"]["RL"]["contact_class"],
                "normal_force_n": row["contacts"]["RL"]["total_normal_force_n"],
                "front_distance_mm": row["geometry"]["RL"]["front_distance_mm"],
                "top_gap_mm": row["geometry"]["RL"]["top_gap_mm"],
            },
        })

    cp_rows = []
    for row in source["CP225280_comparison"]["selected_rows"]:
        rr = row["RR"]
        local = row["CoM_to_FR_local_window"]
        cp_rows.append({
            "event": row["event"], "physics_tick": row["physics_tick"],
            "time_s": row["time_s"], "RR_current_TOP": rr["current_TOP"],
            "RR_current_TOP_bearing": rr["current_TOP_bearing"],
            "RR_bearing_force_n": rr["bearing_force_n"],
            "RR_front_distance_mm": rr["front_distance_mm"],
            "RR_gap_mm": rr["gap_mm"],
            "local_0p5s_COM_toward_FR_mm": (
                None if not local.get("valid") else 1000.0 * local["com_toward_fr_m"]),
            "local_direction_recomputed_at_each_window": True,
        })

    transitions = source["stage_transition_event_evidence"][-1]["event_ticks"]
    return {
        "schema": "outputs.successful_N_fixed_FR_transfer_projection.v1",
        "source": {"path": str(SOURCE), "sha256": sha256(SOURCE),
                   "upstream_manifest_sha256": source["source"]["manifest_sha256"]},
        "scope": {
            "input": "existing 21 exact 120Hz rows only; no sealed-log rescan",
            "interpolation": False, "physics_or_model_execution": False,
            "reference": "successful historical N+0, not same-controller B",
        },
        "fixed_receiver_direction": {
            "reference_event": "PRE_LATE_RR_LOADED_CONTEXT",
            "reference_tick": start["episode_physics_tick"],
            "COM_start_xy_m": com0[:2],
            "FR_wheel_xy_m": fr[:2] if fr is not None else None,
            "FR_wheel_xy_semantics": (
                "already-extracted ContactSensor support point owned by front_right_wheel; "
                "the compact table did not retain the rigid-body wheel-center xy"),
            "requested_exact_rigid_body_wheel_center_available": False,
            "horizontal_distance_m": norm,
            "unit_xy": direction,
            "frozen_for_all_later_rows": True,
            "moving_FR_is_not_referenced_again": True,
            "RR_loaded_at_reference": True,
            "RR_load_n": start["contacts"]["RR"]["total_normal_force_n"],
        },
        "rear_event_ticks": {
            "RL_qualified": transitions["active_lift"]["RL"],
            "RL_front_edge_crossed": transitions["front_edge_crossed"]["RL"],
            "RL_placed": transitions["placed"]["RL"],
        },
        "selected_rows": selected,
        "CP225280_entrance_comparison": {
            "rows": cp_rows,
            "boundary": source["CP225280_comparison"]["interpretation_boundary"],
            "fixed_loaded_transfer_start_available": False,
            "reason": "CP225280 selected entrance never has current RR TOP/load; its local 0.5s directions are not the fixed successful-N axis",
        },
        "claims": {
            "joint_angles_define_FR_direction": False,
            "projection_proves_FL_or_any_single_channel_causal": False,
            "sparse_yaw_rate_integrated_into_yaw": False,
            "FR_support_point_equals_rigid_body_wheel_center": False,
            "CP225280_is_time_aligned_or_same_controller": False,
        },
    }


def fmt(value: Any, digits: int = 3) -> str:
    return "N/A" if value is None else f"{float(value):.{digits}f}"


def markdown(data: dict[str, Any]) -> str:
    fixed = data["fixed_receiver_direction"]
    lines = [
        "# Successful N: fixed FR-receiver COM projection",
        "",
        "This addendum uses only the existing 21 exact-tick successful-N table; it does not rescan "
        "the large sealed logs, interpolate ticks, or run physics. The fixed receiver axis is set once "
        "at the RR-loaded pre-late row and is never updated as FR moves.",
        "",
        "## Fixed direction and evidence boundary",
        "",
        f"- Start: tick {fixed['reference_tick']}; COM xy={fixed['COM_start_xy_m']}; logged FR-wheel "
        f"support-point xy={fixed['FR_wheel_xy_m']}; unit direction={fixed['unit_xy']}.",
        f"- RR load at start: {fmt(fixed['RR_load_n'])} N. The direction therefore begins in an "
        "actually RR-loaded context, not an AIR-RR assumption.",
        "- The compact table retained the ContactSensor support point owned by `front_right_wheel`, "
        "not the rigid-body wheel-center xy. That distinction is explicit in JSON; the support point "
        "is used as the only already-extracted actual FR receiver location.",
        "- Base orientation was not retained. Yaw displacement is therefore N/A; logged yaw rate is "
        "shown per row and is not sparsely integrated.",
        "",
        "## Actual COM response and loads",
        "",
        "|event|tick / t|dx / dy / fixed-FR projection (mm)|COM v·d (m/s)|yaw rate (rad/s)|FL / RR load (N)|RL context|",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in data["selected_rows"]:
        disp = row["COM_displacement_from_start_m"]
        load = row["measured_normal_load_n"]
        rl = row["RL"]
        rl_label = (f"{rl['event'] + '; ' if rl['event'] else ''}{rl['contact_class']} "
                    f"front/gap={fmt(rl['front_distance_mm'],1)}/{fmt(rl['top_gap_mm'],1)}mm")
        lines.append(
            f"|{row['event']}|{row['episode_physics_tick']} / {fmt(row['time_s'])}s|"
            f"{fmt(1000*disp['x'],1)} / {fmt(1000*disp['y'],1)} / "
            f"{fmt(1000*disp['fixed_FR_receiver_projection'],1)}|"
            f"{fmt(row['COM_velocity_fixed_FR_receiver_projection_m_s'])}|"
            f"{fmt(row['yaw_rate_rad_s'])}|{fmt(load['FL'])} / {fmt(load['RR'])}|{rl_label}|"
        )
    ticks = data["rear_event_ticks"]
    lines += [
        "",
        f"RL qualification, crossing, and placement are separately logged at ticks "
        f"{ticks['RL_qualified']}, {ticks['RL_front_edge_crossed']}, and {ticks['RL_placed']}; "
        "none is inferred from joint angles or COM direction.",
        "",
        "## CP225280 entrance contrast",
        "",
        "|event|tick / t|RR TOP / load|RR front / gap (mm)|local 0.5s COM→FR (mm)|",
        "|---|---:|---:|---:|---:|",
    ]
    for row in data["CP225280_entrance_comparison"]["rows"]:
        lines.append(
            f"|{row['event']}|{row['physics_tick']} / {fmt(row['time_s'])}s|"
            f"{row['RR_current_TOP']} / {fmt(row['RR_bearing_force_n'])}|"
            f"{fmt(row['RR_front_distance_mm'],1)} / {fmt(row['RR_gap_mm'],1)}|"
            f"{fmt(row['local_0p5s_COM_toward_FR_mm'],1)}|"
        )
    lines += [
        "",
        "CP225280 never supplies an equivalent loaded RR transfer start in these selected rows; its "
        "0.5 s projection axes were recomputed at each local window. Those numbers are descriptive, "
        "not time-aligned with—or projections onto—the successful-N fixed axis.",
        "",
        "The positive/negative fixed-axis projection is measured whole-body COM motion toward/away "
        "from the frozen FR receiver direction. It does not attribute that motion to FL, RR, a joint "
        "angle, or any single command channel.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    data = build()
    OUT_JSON.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    OUT_MD.write_text(markdown(data), encoding="utf-8")
    print(OUT_JSON)
    print(OUT_MD)


if __name__ == "__main__":
    main()
