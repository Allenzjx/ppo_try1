"""Compact, streaming RR-to-RL audit for one sealed cooperative video source.

This is outputs-only post-processing.  It never imports Torch/PXR, starts a
simulation, decodes video, interpolates telemetry, or treats geometry as
contact/bearing evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Iterator


HERE = Path(__file__).resolve().parent
EXPORTER_PATH = HERE / "export_policy_rear_no_assist_video.py"
SPEC = importlib.util.spec_from_file_location("_cooperative_window_exporter", EXPORTER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load the outputs-only exporter")
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)

HZ = 120.0
COARSE_TICKS = 240  # 2 s; event boundaries are retained independently.
MAX_SELECTED_ROWS = 128
WHEEL_LABELS = ("FL", "FR", "RL", "RR")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def checked_artifact(source: Path, manifest: dict[str, Any], name: str) -> tuple[Path, dict[str, Any]]:
    record = (manifest.get("artifacts") or {}).get(name)
    require(isinstance(record, dict), f"sealed manifest lacks {name}")
    path = Path(record.get("path", "")).resolve(strict=True)
    require(path.parent == source and path.name == name and
            path.stat().st_size == record.get("bytes"),
            f"{name} path/size differs from the sealed manifest")
    return path, record


def jsonl(path: Path, record: dict[str, Any]) -> Iterator[dict[str, Any]]:
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for raw in stream:
            require(raw.endswith(b"\n"), f"{path.name} has a partial final row")
            digest.update(raw)
            size += len(raw)
            value = json.loads(raw)
            require(isinstance(value, dict), f"{path.name} row is not an object")
            yield value
    require(size == record.get("bytes") and digest.hexdigest() == record.get("sha256"),
            f"{path.name} changed during its one-pass scan")


def phase_number(value: Any) -> int | None:
    if isinstance(value, str) and len(value) == 3 and value.startswith("P") and value[1:].isdigit():
        return int(value[1:])
    return None


def mm(value: Any) -> float | None:
    number = exporter._number(value)
    return None if number is None else 1000.0 * number


def update_range(ranges: dict[str, list[float]], name: str, value: Any) -> None:
    number = exporter._number(value)
    if number is None:
        return
    if name not in ranges:
        ranges[name] = [float(number), float(number)]
    else:
        ranges[name][0] = min(ranges[name][0], float(number))
        ranges[name][1] = max(ranges[name][1], float(number))


def event_flags(row: dict[str, Any], summary: dict[str, Any]) -> dict[str, bool]:
    phase = str(row.get("phase"))
    timing = summary["rear_policy_timing"]
    current = row.get("current_legs") or {}
    rr, rl = current.get("RR") or {}, current.get("RL") or {}
    history = row.get("placed_history") or {}

    def any_surface_bearing(leg: dict[str, Any]) -> bool:
        force = exporter._number(leg.get("bearing_force_n"))
        return bool(leg.get("bearing_verified") is True and
                    leg.get("support") is True and leg.get("air") is False and
                    force is not None and force > 0 and
                    (leg.get("ground_contact") is True or
                     leg.get("top_surface_contact") is True))

    def top_bearing(leg: dict[str, Any]) -> bool:
        return bool(any_surface_bearing(leg) and
                    leg.get("ground_contact") is False and
                    leg.get("top_contact") is True and
                    leg.get("top_surface_contact") is True and
                    leg.get("obstacle_pair_active") is True and
                    leg.get("within_top_xy") is True and
                    leg.get("contact_surface") == "TOP")

    return {
        f"{phase}_ENTRY": True,
        "RR_CARRY": timing["rr_carry_capture"],
        "RR_HANDOFF": timing["rr_support_handoff"],
        "RL_PREP": timing["rl_prep_transfer"],
        "RL_SWING": timing["rl_swing_capture"],
        "RR_QUALIFIED": bool(rr.get("current_lift_valid")),
        "RR_CROSSED": bool(rr.get("front_edge_crossed")),
        "RR_TOP": bool(rr.get("top_surface_contact")),
        "RR_ANY_SURFACE_BEARING": any_surface_bearing(rr),
        "RR_TOP_BEARING": top_bearing(rr),
        "RR_PLACED_HISTORY": history.get("RR") is True,
        "RL_QUALIFIED": bool(rl.get("current_lift_valid")),
        "RL_CROSSED": bool(rl.get("front_edge_crossed")),
        "RL_TOP": bool(rl.get("top_surface_contact")),
        "RL_ANY_SURFACE_BEARING": any_surface_bearing(rl),
        "RL_TOP_BEARING": top_bearing(rl),
        "RL_PLACED_HISTORY": history.get("RL") is True,
    }


def compact_row(row: dict[str, Any], native: dict[str, Any], height: dict[str, Any] | None,
                reasons: list[str]) -> dict[str, Any]:
    tick = row.get("episode_physics_tick")
    base = exporter.frame_summary(row, native)
    mount = (exporter.mount_frame_summary(height, tick) if height is not None else
             exporter.unavailable_mounts("no exact-tick height diagnostic"))
    dispatch = row.get("dispatch") or {}
    nominal = exporter._vector(row.get("nominal_full12"), 12, "nominal Full12")
    policy = exporter._vector(dispatch.get("independent_policy_residual_requested_full12"),
                              12, "policy Full12")
    final = exporter._vector(dispatch.get("drive_target_full12"), 12, "final Full12")
    actual = exporter._vector(row.get("actual_full12"), 12, "actual Full12")
    current = row.get("current_legs") or {}
    rr, rl = current.get("RR") or {}, current.get("RL") or {}
    body, com = row.get("body") or {}, row.get("center_of_mass") or {}
    return {
        "selection_reasons": sorted(set(reasons)),
        "physics_tick": tick, "time_s": row.get("sim_time_s"), "phase": row.get("phase"),
        "mount_geometry": mount,
        "front_knee_actual_deg": {
            "FL": exporter._joint(row, exporter.SERVO_ORDER[1]),
            "FR": exporter._joint(row, exporter.SERVO_ORDER[3]),
        },
        "wheel_canonical_rad_s": {
            kind: dict(zip(WHEEL_LABELS, values)) for kind, values in (
                ("nominal", nominal[8:]), ("policy_request", policy[8:]),
                ("final_target", final[8:]), ("actual_measured", actual[8:]),
                ("native_physical_target", base["wheel_native_physical"]),
            )
        },
        "RR": {
            "gap_mm": base["rr_gap_mm"], "front_distance_mm": base["rr_front_mm"],
            "current_TOP": bool(rr.get("top_surface_contact")),
            "bearing_verified_raw": bool(rr.get("bearing_verified")),
            "current_support": bool(rr.get("support")),
            "current_TOP_bearing": event_flags(row, base)["RR_TOP_BEARING"],
            "bearing_force_n": exporter._number(rr.get("bearing_force_n")),
            "current_qualified": bool(rr.get("current_lift_valid")),
            "front_edge_crossed": bool(rr.get("front_edge_crossed")),
            "placed_history": (row.get("placed_history") or {}).get("RR"),
            "contact_surface": rr.get("contact_surface"),
        },
        "RL": {
            "clearance_mm": mm(rl.get("clearance_m")),
            "front_distance_mm": mm(rl.get("front_distance_m")),
            "current_qualified": bool(rl.get("current_lift_valid")),
            "front_edge_crossed": bool(rl.get("front_edge_crossed")),
            "current_TOP": bool(rl.get("top_surface_contact")),
            "bearing_verified_raw": bool(rl.get("bearing_verified")),
            "current_support": bool(rl.get("support")),
            "current_TOP_bearing": event_flags(row, base)["RL_TOP_BEARING"],
            "placed_history": (row.get("placed_history") or {}).get("RL"),
        },
        "rear_policy_timing": base["rear_policy_timing"],
        "body_position_w_m": body.get("position_w_m"),
        "body_orientation_wxyz": body.get("orientation_wxyz"),
        "center_of_mass_position_w_m": com.get("position_w_m"),
        "com_to_FR_local_window": exporter.com_to_fr_window_summary(row),
        "geometry_is_contact_or_bearing_evidence": False,
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.source).resolve(strict=True)
    run_path = source.parent / "run_manifest.json"
    manifest_path = source / "semantic_video_source_manifest.json"
    require(exporter.sha256(run_path) == exporter.checked_sha(
        args.source_run_manifest_sha256, "--source-run-manifest-sha256") and
        exporter.sha256(manifest_path) == exporter.checked_sha(
        args.source_manifest_sha256, "--source-manifest-sha256"),
        "sealed run/source manifest hash differs")
    run, manifest = exporter.read_json(run_path), exporter.read_json(manifest_path)
    require(bool(run.get("completed_at_utc")) and run.get("lifecycle") != "RUNNING" and
            run.get("runtime_contract") == manifest.get("runtime_contract") and
            manifest.get("schema") == "wlr50_clean.semantic_video_source.v1" and
            manifest.get("experiment_id") == exporter.EXPERIMENT and
            manifest.get("role") == "C" and manifest.get("from_phase") == "P01" and
            manifest.get("fresh_process_single_episode") is True and
            manifest.get("episode_count") == 1 and
            manifest.get("optimizer_updates") == 0,
            "source is not one closed zero-update evaluation")
    identity = exporter.checkpoint_identity(manifest, args)
    require(identity.get("cooperative_preparation_revision") is not None,
            "checkpoint is not the cooperative-prep lineage")
    source_nominal_timing = (exporter.NOMINAL_TIMING
        if identity.get("p02_progress_revision") is not None else
        identity["nominal_timing"])
    require(manifest.get("control_method") == exporter.CONTROL_METHOD and
            manifest.get("capture_assist_enabled_in_training_and_evaluation") is True and
            manifest.get("capture_assist_is_policy_learning") is False and
            manifest.get("front_fl_capture_assist") == {
                "enabled": True, "mode": exporter.FL_ASSIST_MODE,
                "is_policy_learning": False} and
            manifest.get("rear_task_assist") == {
                "enabled": False, "rr_capture_assist_mode": None,
                "rr_capture_wheel_mode": "off",
                "nominal_geometry_advisory": None,
                "nominal_timing": source_nominal_timing,
                "policy_controls_rear_task_actions": True,
                "rear_action_request": "residual_policy"},
            "source does not prove exact FL-ON/rear-OFF evaluation semantics")
    error = manifest.get("source_acceptance_error")
    require(error is None or (isinstance(error, str) and
            "episode did not meet common physical task" in error and
            "VIDEO_OR_ARTIFACT_ERROR" not in error and
            "VideoArtifactError" not in error),
            "artifact/capture failure is outside this physical-window audit")

    paths = {}
    records = {}
    for name in ("capture_assist_ticks.jsonl", "native_tick_audit.jsonl",
                 "height_diagnostics.jsonl", "video_policy_decisions.jsonl"):
        paths[name], records[name] = checked_artifact(source, manifest, name)

    height_iter = iter(jsonl(paths["height_diagnostics.jsonl"],
                             records["height_diagnostics.jsonl"]))
    next_height = next(height_iter, None)
    # The sealed diagnostic intentionally includes the reset sample at tick 0.
    last_height_tick = -1
    capture_iter = iter(jsonl(paths["capture_assist_ticks.jsonl"],
                              records["capture_assist_ticks.jsonl"]))
    native_iter = iter(jsonl(paths["native_tick_audit.jsonl"],
                             records["native_tick_audit.jsonl"]))
    first_events: dict[str, dict[str, Any]] = {}
    rows_by_tick: dict[int, dict[str, Any]] = {}
    ranges: dict[str, list[float]] = {}
    first_window_tick = None
    final_row = final_native = final_height = None
    count = 0
    for count, row in enumerate(capture_iter, 1):
        native = next(native_iter, None)
        require(native is not None and row.get("episode_physics_tick") == count and
                native.get("episode_physics_tick") == count,
                "capture/native tick streams are not exact continuous peers")
        height = None
        while next_height is not None:
            height_tick = next_height.get("physics_tick")
            require(type(height_tick) is int and height_tick > last_height_tick,
                    "height diagnostics ticks are not strictly increasing")
            if height_tick > count:
                break
            last_height_tick = height_tick
            if height_tick == count:
                height = next_height
            next_height = next(height_iter, None)
        base = exporter.frame_summary(row, native)
        phase = phase_number(row.get("phase"))
        active = ((phase is not None and phase >= 7) or
                  any(base["rear_policy_timing"][key] for key in exporter.REAR_TIMING_FLAGS))
        if active and first_window_tick is None:
            first_window_tick = count
        flags = event_flags(row, base)
        new_events = [name for name, present in flags.items()
                      if present and name not in first_events]
        for name in new_events:
            first_events[name] = {"physics_tick": count,
                                  "time_s": row.get("sim_time_s"),
                                  "phase": row.get("phase")}
        reasons = list(new_events)
        if active and (count - first_window_tick) % COARSE_TICKS == 0:
            reasons.append("COARSE_2S")
        if reasons:
            rows_by_tick[count] = compact_row(row, native, height, reasons)
        if active:
            mount = (exporter.mount_frame_summary(height, count) if height is not None else {})
            if mount.get("valid") is True:
                for name in ("left_minus_right_mean_z_m", "rear_minus_front_mean_z_m",
                             "FR_world_z_m"):
                    update_range(ranges, name, mount.get(name))
            update_range(ranges, "RR_gap_mm", base["rr_gap_mm"])
            update_range(ranges, "RR_front_distance_mm", base["rr_front_mm"])
            update_range(ranges, "RL_clearance_mm", mm((row.get("current_legs") or {}).get(
                "RL", {}).get("clearance_m")))
            com_window = exporter.com_to_fr_window_summary(row)
            if com_window.get("valid") is True:
                update_range(ranges, "CoM_toward_FR_local_m",
                             com_window.get("com_toward_fr_m"))
        final_row, final_native, final_height = row, native, height
    require(next(native_iter, None) is None, "native audit has extra rows")
    while next_height is not None:
        height_tick = next_height.get("physics_tick")
        require(type(height_tick) is int and height_tick > last_height_tick,
                "height diagnostics tail is not strictly increasing")
        last_height_tick = height_tick
        next_height = next(height_iter, None)
    endpoint = manifest.get("episode_physics_ticks")
    require(count == endpoint and final_row is not None,
            "sealed capture tick stream does not cover the physical endpoint")
    if count not in rows_by_tick:
        rows_by_tick[count] = compact_row(final_row, final_native, final_height, ["ENDPOINT"])
    else:
        rows_by_tick[count]["selection_reasons"] = sorted(set(
            rows_by_tick[count]["selection_reasons"] + ["ENDPOINT"]))

    terminal = None
    decision_count = 0
    for decision_count, decision in enumerate(jsonl(
            paths["video_policy_decisions.jsonl"],
            records["video_policy_decisions.jsonl"]), 1):
        step = decision.get("step_info")
        if isinstance(step, dict):
            terminal = step
    task = ((terminal or {}).get("semantic_task") or {})
    require(len(rows_by_tick) <= MAX_SELECTED_ROWS,
            "bounded selector exceeded 128 rows; increase interval deliberately")
    selected = [rows_by_tick[tick] for tick in sorted(rows_by_tick)]
    result = {
        "schema": "wlr50_clean.cooperative_rr_rl_window_audit.v2",
        "source": str(source), "source_manifest": str(manifest_path),
        "source_manifest_sha256": args.source_manifest_sha256,
        "source_run_manifest_sha256": args.source_run_manifest_sha256,
        "checkpoint_identity": identity,
        "episode": {"physics_ticks": endpoint, "duration_s": endpoint/HZ,
            "policy_decision_rows": decision_count,
            "physical_task_success": manifest.get("physical_task_success"),
            "source_acceptance_error": manifest.get("source_acceptance_error"),
            "termination_reason": task.get("termination_reason"),
            "termination_source": task.get("termination_source")},
        "window": {"start_tick": first_window_tick,
            "start_time_s": None if first_window_tick is None else first_window_tick/HZ,
            "end_tick": endpoint, "end_time_s": endpoint/HZ,
            "scope": "actual same-episode P07-or-rear-timing onset through physical endpoint",
            "no_RR_or_RL_reached_is_reported_not_fabricated": True},
        "first_events": first_events,
        "whole_window_ranges": ranges,
        "selected_rows": selected,
        "selection": {"coarse_interval_ticks": COARSE_TICKS,
            "coarse_interval_s": COARSE_TICKS/HZ,
            "event_rows_plus_endpoint": True, "selected_row_count": len(selected),
            "maximum_rows": MAX_SELECTED_ROWS},
        "claims": {"geometry_is_bearing_or_contact": False,
            "raw_bearing_verified_without_TOP_and_current_support_is_TOP_bearing": False,
            "CoM_local_window_is_total_transfer": False,
            "CoM_toward_FR_implies_FR_bearing": False,
            "interpolation_used": False, "simulation_or_model_loaded": False,
            "video_decoded_or_reencoded": False},
        "artifact_sha256_verified_in_same_stream_pass": {
            name: records[name]["sha256"] for name in records},
    }
    output_json = Path(args.output_json).resolve()
    output_md = Path(args.output_md).resolve()
    require(output_json.parent == HERE and output_md.parent == HERE and
            not output_json.exists() and not output_md.exists(),
            "outputs must be new files in this isolated output directory")
    exporter.write_new_json(output_json, result)
    lines = [
        "# CP cooperative RR→RL window audit", "",
        f"- Source: `{source}`",
        f"- Physical window: {result['window']['start_time_s']}s → {result['window']['end_time_s']:.6f}s",
        f"- Result: success={manifest.get('physical_task_success')}; termination={task.get('termination_reason')}",
        f"- Selected: {len(selected)} event/coarse rows; no interpolation or video processing.",
        "- `*_ANY_SURFACE_BEARING` may be initial GROUND support; `*_TOP_BEARING` alone requires current TOP geometry/contact, support and positive force.",
        "", "## First observed events", "",
        "| Event | Tick | Time s | Phase |", "|---|---:|---:|---|",
    ]
    for name, value in first_events.items():
        lines.append(f"| {name} | {value['physics_tick']} | {value['time_s']:.6f} | {value['phase']} |")
    lines += ["", "## Whole-window ranges", "",
        "Values are descriptive extrema over the same sealed episode window. Hip-mount geometry is not contact/bearing; CoM→FR is the logged ≤0.5 s fixed-direction local projection, not total transfer or FR load.", "",
        "| Quantity | Min | Max |", "|---|---:|---:|"]
    for name, pair in sorted(ranges.items()):
        lines.append(f"| {name} | {pair[0]:.9g} | {pair[1]:.9g} |")
    lines += ["", f"Full selected rows: [{output_json.name}]({output_json.name})", ""]
    with output_md.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("\n".join(lines))
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument("--source", type=Path, required=True)
    result.add_argument("--source-manifest-sha256", required=True)
    result.add_argument("--source-run-manifest-sha256", required=True)
    result.add_argument("--expected-head", required=True)
    result.add_argument("--checkpoint-runtime-head")
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--checkpoint-sha256", required=True)
    result.add_argument("--checkpoint-manifest-sha256", required=True)
    result.add_argument("--expected-global-policy-decisions", type=int, required=True)
    result.add_argument("--expected-ppo-updates", type=int, required=True)
    result.add_argument("--expected-optimizer-steps", type=int, required=True)
    result.add_argument("--expected-new-auxiliary-updates", type=int, default=0)
    result.add_argument("--output-json", type=Path, required=True)
    result.add_argument("--output-md", type=Path, required=True)
    return result


if __name__ == "__main__":
    analyze(parser().parse_args())
