"""Small descriptive body-motion comparison from two sealed JSONL sources.

This outputs-only helper reads only each source manifest and its recorded
``height_diagnostics.jsonl``.  It does not decode video, import Torch/PXR,
start simulation, interpolate samples, or make a stability/causal ranking.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
PHYSICS_HZ = 120.0


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_sha(value: str, label: str) -> str:
    value = str(value).lower()
    require(len(value) == 64 and all(c in "0123456789abcdef" for c in value),
            f"{label} must be one SHA-256")
    return value


def number(value: Any, label: str) -> float:
    require(type(value) in (int, float) and math.isfinite(value),
            f"{label} is not finite")
    return float(value)


def vector(value: Any, length: int, label: str) -> list[float]:
    require(isinstance(value, list) and len(value) == length,
            f"{label} must contain {length} values")
    return [number(v, label) for v in value]


def phase_number(value: Any) -> int | None:
    if isinstance(value, str) and len(value) == 3 and value.startswith("P"):
        try:
            return int(value[1:])
        except ValueError:
            pass
    return None


def roll_pitch_deg(quat_wxyz: list[float]) -> tuple[float, float]:
    w, x, y, z = quat_wxyz
    norm = math.sqrt(sum(q * q for q in quat_wxyz))
    require(norm > 0, "zero base quaternion")
    w, x, y, z = (q / norm for q in (w, x, y, z))
    roll = math.atan2(2.0 * (w * x + y * z),
                      1.0 - 2.0 * (x * x + y * y))
    pitch_sine = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(pitch_sine)
    return math.degrees(roll), math.degrees(pitch)


def read_source(source: Path, manifest_sha: str, *, expected_experiment: str,
                expected_role: str) -> dict[str, Any]:
    source = source.resolve(strict=True)
    manifest_path = source / "semantic_video_source_manifest.json"
    require(manifest_path.is_file(), "source manifest is missing")
    actual_manifest_sha = sha256(manifest_path)
    require(actual_manifest_sha == checked_sha(manifest_sha, "manifest SHA"),
            "source manifest differs from the explicit SHA")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    endpoint = manifest.get("episode_physics_ticks")
    require(manifest.get("schema") == "wlr50_clean.semantic_video_source.v1" and
            manifest.get("experiment_id") == expected_experiment and
            manifest.get("role") == expected_role and
            manifest.get("episode_count") == 1 and
            type(endpoint) is int and endpoint > 0,
            "source identity/episode is not the requested sealed run")
    artifact = (manifest.get("artifacts") or {}).get("height_diagnostics.jsonl")
    require(isinstance(artifact, dict), "height diagnostic artifact is missing")
    path = Path(artifact.get("path", "")).resolve(strict=True)
    require(path == source / "height_diagnostics.jsonl" and
            path.stat().st_size == artifact.get("bytes"),
            "height diagnostic path/size differs from the manifest")

    digest, byte_count = hashlib.sha256(), 0
    samples: list[dict[str, Any]] = []
    last_tick = -1
    sensor_source = None
    with path.open("rb") as stream:
        for raw in stream:
            require(raw.endswith(b"\n"), "height diagnostics has a partial row")
            digest.update(raw)
            byte_count += len(raw)
            row = json.loads(raw)
            tick = row.get("physics_tick")
            require(type(tick) is int and last_tick < tick <= endpoint,
                    "height diagnostic ticks are not strictly increasing/in range")
            last_tick = tick
            base_record = row.get("base")
            require(isinstance(base_record, dict) and
                    isinstance(base_record.get("value"), dict),
                    "height diagnostic lacks measured base record")
            base = base_record["value"]
            source_label = base_record.get("source")
            if sensor_source is None:
                sensor_source = source_label
            require(source_label == sensor_source,
                    "base sensor source changes within episode")
            linear = vector(base.get("linear_velocity_w_m_s"), 3,
                            "base linear velocity")
            angular = vector(base.get("angular_velocity_w_rad_s"), 3,
                             "base angular velocity")
            quat = vector(base.get("orientation_wxyz"), 4,
                          "base orientation quaternion")
            roll, pitch = roll_pitch_deg(quat)
            samples.append({
                "tick": tick,
                "phase": row.get("phase"),
                "linear_speed_m_s": math.sqrt(sum(v * v for v in linear)),
                "angular_speed_rad_s": math.sqrt(sum(v * v for v in angular)),
                "roll_deg": roll,
                "pitch_deg": pitch,
            })
    require(byte_count == artifact.get("bytes") and
            digest.hexdigest() == artifact.get("sha256"),
            "height diagnostic changed or failed its sealed SHA")
    require(samples and samples[0]["tick"] == 0 and samples[-1]["tick"] == endpoint,
            "height diagnostics must include reset tick 0 and exact terminal tick")
    deltas = [b["tick"] - a["tick"] for a, b in zip(samples, samples[1:])]
    common_delta = Counter(deltas).most_common(1)[0][0]
    rear_index = next((i for i, row in enumerate(samples)
                       if (phase_number(row["phase"]) or 0) >= 7), None)
    return {
        "source": str(source),
        "manifest_path": str(manifest_path),
        "manifest_sha256": actual_manifest_sha,
        "height_diagnostics_path": str(path),
        "height_diagnostics_sha256": digest.hexdigest(),
        "height_diagnostics_bytes": byte_count,
        "height_sensor_source": sensor_source,
        "endpoint_tick": endpoint,
        "physical_duration_s": endpoint / PHYSICS_HZ,
        "physical_task_success": manifest.get("physical_task_success"),
        "source_acceptance_error": manifest.get("source_acceptance_error"),
        "sample_count": len(samples),
        "dominant_tick_stride": common_delta,
        "dominant_sample_rate_hz": PHYSICS_HZ / common_delta,
        "rear_index": rear_index,
        "samples": samples,
    }


def window_stats(run: dict[str, Any], start_index: int, scope: str) -> dict[str, Any]:
    rows = run["samples"][start_index:]
    start_tick = rows[0]["tick"]
    endpoint = run["endpoint_tick"]
    require(start_tick < endpoint and rows[-1]["tick"] == endpoint,
            f"{scope} window has no positive exact duration")
    fields = ("linear_speed_m_s", "angular_speed_rad_s", "roll_deg", "pitch_deg")
    weighted_squares = {name: 0.0 for name in fields}
    weighted_ticks = 0
    for current, following in zip(rows, rows[1:]):
        ticks = following["tick"] - current["tick"]
        require(ticks > 0, "non-positive diagnostic interval")
        weighted_ticks += ticks
        for name in fields:
            weighted_squares[name] += current[name] ** 2 * ticks
    require(weighted_ticks == endpoint - start_tick,
            "piecewise-constant intervals do not cover exact physical duration")

    def rms(name: str) -> float:
        return math.sqrt(weighted_squares[name] / weighted_ticks)

    linear = [row["linear_speed_m_s"] for row in rows]
    angular = [row["angular_speed_rad_s"] for row in rows]
    roll = [row["roll_deg"] for row in rows]
    pitch = [row["pitch_deg"] for row in rows]
    return {
        "scope": scope,
        "start_tick": start_tick,
        "end_tick": endpoint,
        "duration_s": weighted_ticks / PHYSICS_HZ,
        "sample_count_including_zero_weight_endpoint": len(rows),
        "body_linear_speed_m_s": {"rms": rms("linear_speed_m_s"), "max": max(linear)},
        "body_angular_speed_rad_s": {"rms": rms("angular_speed_rad_s"), "max": max(angular)},
        "roll_deg": {"rms_about_zero": rms("roll_deg"),
                     "min": min(roll), "max": max(roll)},
        "pitch_deg": {"rms_about_zero": rms("pitch_deg"),
                      "min": min(pitch), "max": max(pitch)},
    }


def public_run(run: dict[str, Any]) -> dict[str, Any]:
    result = {key: value for key, value in run.items()
              if key not in ("samples", "rear_index")}
    result["full_physical_episode"] = window_stats(run, 0, "full_physical_episode")
    result["RR_to_RL_window"] = (None if run["rear_index"] is None else
        window_stats(run, run["rear_index"],
                     "first_recorded_P07_or_later_sample_through_physical_endpoint"))
    return result


def fmt(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.6f}"


def write_new(path: Path, text: str) -> None:
    path = path.resolve()
    require(path.parent == HERE and not path.exists(),
            "output must be one new file in the isolated outputs directory")
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())


def main(args: argparse.Namespace) -> None:
    candidate = read_source(args.candidate_source, args.candidate_manifest_sha256,
        expected_experiment="rr_rl_timing_policy_learning_v1", expected_role="C")
    reference = read_source(args.reference_source, args.reference_manifest_sha256,
        expected_experiment="task_conditioned_hip_wheel_v1", expected_role="B")
    runs = {"CP223616_candidate": public_run(candidate),
            "historical_N_reference": public_run(reference)}
    result = {
        "schema": "wlr50_clean.descriptive_body_stability_two_sealed_runs.v1",
        "runs": runs,
        "method": {
            "signals": "recorded actual base linear/angular velocity and base orientation quaternion",
            "quaternion_order": "wxyz",
            "roll_pitch_conversion": "normalized quaternion; standard world-referenced XYZ roll and pitch",
            "sampling": "sealed height_diagnostics; dominant 15 Hz (8 physics ticks)",
            "RMS": "left-sample piecewise-constant weighting to next diagnostic tick; exact physical endpoint duration; no interpolation",
            "extrema": "all recorded samples including exact zero-weight terminal sample",
            "RR_to_RL_window": "first recorded sample whose phase is P07 or later through the physical endpoint",
            "joint_command_variation": "N/A (not needed for this bounded body-motion table)",
        },
        "claims": {
            "stability_superiority_claimed": False,
            "causal_claimed": False,
            "time_normalized_ranking_claimed": False,
            "same_controller_or_version_claimed": False,
            "video_freeze_counted_as_physics": False,
            "different_task_outcomes_and_durations_retained": True,
        },
    }
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    write_new(output_json, json.dumps(result, indent=2, sort_keys=True,
                                      ensure_ascii=False) + "\n")
    lines = [
        "# CP223616 vs historical N: descriptive body-motion table", "",
        "These are separate sealed physical episodes with different task outcomes and durations. The table is descriptive only: it does **not** establish stability superiority, causality, or a time-normalized ranking. Historical N is not a fresh same-controller B. Video freeze is excluded.", "",
        "| Run / physical window | Outcome | Duration s | Samples | Linear speed RMS / max (m/s) | Angular speed RMS / max (rad/s) | Roll min..max / RMS0 (deg) | Pitch min..max / RMS0 (deg) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for run_name, run in runs.items():
        outcome = "SUCCESS" if run["physical_task_success"] is True else "INCOMPLETE"
        for key in ("full_physical_episode", "RR_to_RL_window"):
            window = run[key]
            if window is None:
                lines.append(f"| {run_name} / {key} | {outcome} | N/A | 0 | N/A | N/A | N/A | N/A |")
                continue
            lin, ang = window["body_linear_speed_m_s"], window["body_angular_speed_rad_s"]
            roll, pitch = window["roll_deg"], window["pitch_deg"]
            lines.append(
                f"| {run_name} / {key} | {outcome} | {fmt(window['duration_s'])} | "
                f"{window['sample_count_including_zero_weight_endpoint']} | "
                f"{fmt(lin['rms'])} / {fmt(lin['max'])} | "
                f"{fmt(ang['rms'])} / {fmt(ang['max'])} | "
                f"{fmt(roll['min'])}..{fmt(roll['max'])} / {fmt(roll['rms_about_zero'])} | "
                f"{fmt(pitch['min'])}..{fmt(pitch['max'])} / {fmt(pitch['rms_about_zero'])} |")
    lines += [
        "", "## Sampling and source boundary", "",
        "Both runs use their sealed 15 Hz `height_diagnostics.jsonl` sensor stream (dominant stride: 8 physics ticks at 120 Hz). RMS uses a documented left-sample piecewise-constant weighting over the exact physical duration; it is an endpoint-weighted approximation, not a 120 Hz reconstruction or interpolation.", "",
        "Only each source manifest and height-diagnostic JSONL are used and SHA-verified. Full source paths and hashes are retained in the JSON report.", "",
        f"Machine-readable report: [{output_json.name}]({output_json.name})", "",
    ]
    write_new(output_md, "\n".join(lines))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument("--candidate-source", type=Path, required=True)
    result.add_argument("--candidate-manifest-sha256", required=True)
    result.add_argument("--reference-source", type=Path, required=True)
    result.add_argument("--reference-manifest-sha256", required=True)
    result.add_argument("--output-json", type=Path, required=True)
    result.add_argument("--output-md", type=Path, required=True)
    return result


if __name__ == "__main__":
    main(parser().parse_args())
