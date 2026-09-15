"""Export real B0/C0 task-window captures, independently of physical success.

Small output-only reuse of the existing video11 diagnostic remux workflow.
Never modifies a source, controller, checkpoint, evaluator, or existing output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
from wlr50_clean.infrastructure.video_capture import find_ffmpeg, validate_mp4
from wlr50_clean.evaluation.video_timeline import decode_frame_timeline, load_viewport_frame_ledger
from wlr50_clean.ppo.semantic_video import task_interval_action_window

FPS = 15
HZ = 120
VALIDATION_KEYS = (
    "sha256", "bytes", "valid", "full_decode", "frame_count", "fps", "resolution",
    "codec", "pixel_format", "duration_s", "container_duration_s", "container_duration_valid",
    "first_pts_s", "last_pts_s", "frame_pts_sha256", "decoded_frame_checksums_sha256",
    "unique_frame_checksums", "black_like_frame_count", "timestamps_monotonic", "timestamps_continuous",
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def run(command):
    result = subprocess.run(command, capture_output=True, text=True, errors="replace")
    require(result.returncode == 0, f"Command failed ({result.returncode}): {result.stderr[-2500:]}")
    return result


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_new_json(path, payload):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)


def compact_validation(validation):
    return {key: validation.get(key) for key in VALIDATION_KEYS}


def classify_physical_result(success, reason, source):
    """Physical result category only; always preserve the original reason separately."""
    if success:
        return "SUCCESS"
    values = {str(value).upper() for value in (reason, source) if value}
    if any("SAFETY_ABORT" in value or value in ("FALL", "HARD_JOINT_LIMIT", "NUMERICAL_FAILURE", "NUMERICAL_ANOMALY") for value in values):
        return "SAFETY_ABORT"
    if any("BODY_COLLISION" in value or "WHEEL_ONLY" in value for value in values):
        return "TASK_FAILURE"
    if not values or any("INCOMPLETE" in value or "TIMEOUT" in value or "TIME_LIMIT" in value or "BUDGET" in value for value in values):
        return "TASK_INCOMPLETE"
    return "UNCLASSIFIED_NON_SUCCESS"


def previews(path, count, ffmpeg):
    pattern = path.with_name(path.stem + "_decoded_%02d.png")
    destinations = [path.with_name(path.stem + f"_decoded_{i:02d}.png") for i in (1, 2)]
    require(not any(item.exists() for item in destinations), "Preview exists; never overwrite")
    run([str(ffmpeg), "-hide_banner", "-nostdin", "-v", "error", "-n", "-i", str(path),
         "-map", "0:v:0", "-vf", f"select=eq(n\\,0)+eq(n\\,{count - 1})",
         "-fps_mode", "passthrough", "-frames:v", "2", "-threads", "1", str(pattern)])
    require(all(item.is_file() and item.stat().st_size > 0 for item in destinations), "Preview missing")
    return [{"path": str(item), "frame_index": index} for item, index in zip(destinations, (0, count - 1))]


def final_phase(source_dir):
    # A physical endpoint may interrupt a decision before step_info exists.
    # The recorded request phase remains explicit in that row; do not infer a later phase.
    last = None
    with (source_dir / "video_policy_decisions.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                last = json.loads(line)
    require(last is not None, "No recorded decisions")
    info = last.get("step_info") or {}
    origin = "final_returned_step_info.end_phase_id" if "end_phase_id" in info else "interrupted_final_decision.request_phase"
    return info.get("end_phase_id", last.get("request_phase")), origin, last


def export(source_dir, output, label):
    source_dir, output = Path(source_dir).resolve(), Path(output).resolve()
    source_path = source_dir / "actual_viewport_video.mp4"
    manifest_path = source_dir / "semantic_video_source_manifest.json"
    capture_path = source_dir / "viewport_buffer_video_manifest.json"
    source_bytes, capture_bytes = manifest_path.read_bytes(), capture_path.read_bytes()
    source, capture = json.loads(source_bytes), json.loads(capture_bytes)
    require(label in ("B0", "C0", "D_RR_OFF"), "Unsupported diagnostic label")
    require(source["role"] == {"B0": "B", "C0": "C", "D_RR_OFF": "C"}[label], "Source role mismatch")
    intervention = source.get("diagnostic_intervention")
    if label == "D_RR_OFF":
        require(isinstance(intervention, dict) and intervention.get("masked_raw_indices") == [6, 7]
                and intervention.get("mode") == "DIAGNOSTIC_RR_RAW_CHANNELS_OFF_FROM_FIRST_DECISION"
                and source.get("formal_ppo_success") is False,
                "D_RR_OFF requires explicit from-reset RR raw-channel intervention provenance")
    else:
        require(intervention is None and not (source.get("checkpoint_load_provenance") or {}).get("diagnostic_intervention"),
                "Intervention must not be published as B0 or C0")
    require(source["from_phase"] == "P01" and source["optimizer_updates"] == 0, "Not an untrained full-P01 evaluation")
    require(source.get("fresh_process_single_episode") is True and source.get("episode_count") == 1, "Not a single episode")
    require(source.get("pre_action_ticks") == 0, "This helper does not reinterpret pre-roll")
    require(capture.get("valid") is True and capture.get("encoder_finalized_before_app_close") is True,
            "Capture did not close and validate; retain source and diagnose separately")
    require(not output.exists() and not output.with_suffix(".media.json").exists(), "Output exists; never overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = find_ffmpeg(capture["full_decode"]["ffmpeg_path"])
    count = capture["frame_count"]
    source_validation = validate_mp4(source_path, ffmpeg=ffmpeg, expected_frame_count=count,
                                     require_sane_container_duration=False)
    require(source_validation["valid"] is True, "Source does not fully decode")
    require(source_validation["sha256"] == capture["video_sha256"], "Source no longer matches recorded capture")
    frames = decode_frame_timeline(source_path, ffmpeg=ffmpeg)
    ledger = load_viewport_frame_ledger(source_dir / "viewport_frame_ledger.jsonl")
    window = task_interval_action_window(source, frames, ledger)
    require(0 < window.output_duration_s <= 200, "Complete source exceeds 200 seconds")
    command = [str(ffmpeg), "-hide_banner", "-nostdin", "-v", "error", "-n", "-copyts", "-start_at_zero",
               "-i", str(source_path), "-map", "0:v:0", "-an", "-sn", "-dn", "-c:v", "copy",
               "-movflags", "+faststart", str(output)]
    source_stat = (source_path.stat().st_size, source_path.stat().st_mtime_ns)
    run(command)
    validation = validate_mp4(output, ffmpeg=ffmpeg, expected_frame_count=count, require_sane_container_duration=True)
    require(validation["valid"] is True and 0 < validation["container_duration_s"] <= 200, "Output failed strict media check")
    output_frames = decode_frame_timeline(output, ffmpeg=ffmpeg)
    require(frames == output_frames, "Stream-copy changed decoded frame, exact PTS, or key-frame sequence")
    require(source_stat == (source_path.stat().st_size, source_path.stat().st_mtime_ns), "Source modified")
    require(manifest_path.read_bytes() == source_bytes and capture_path.read_bytes() == capture_bytes, "Source manifest modified")
    phase, phase_origin, decision = final_phase(source_dir)
    physical = source.get("physical_episode") or {}
    evaluation = physical.get("physical_task_evaluation") or {}
    success = source.get("physical_task_success") is True
    reason = evaluation.get("termination_reason") or (decision.get("step_info") or {}).get("termination_reason")
    physical_result = classify_physical_result(success, reason, evaluation.get("termination_source"))
    receipt = {
        "schema": "wlr50_clean.diagnostic_media.v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "label": label, "mode": intervention["mode"] if intervention else source["mode"],
        "underlying_capture_mode": source["mode"], "diagnostic_intervention": intervention,
        "formal_C0": label == "C0", "after_repair": False, "physical_result": physical_result,
        "physical_task_success": success, "terminal_phase": phase, "terminal_phase_provenance": phase_origin,
        "termination_reason": reason,
        "physical_evaluator_termination_reason": evaluation.get("termination_reason"),
        "termination_source": evaluation.get("termination_source"), "physical_evidence_status": evaluation.get("physical_evidence_status"),
        "media_validity": "COMPLETE_PLAYABLE", "source": str(source_path), "output": str(output),
        "source_manifest": str(manifest_path), "source_run": str(source_dir.parent),
        "source_manifest_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "runtime_contract": source["runtime_contract"], "checkpoint_load_provenance": source.get("checkpoint_load_provenance"),
        "camera": source["camera"], "seed": source["seed"], "natural_reset_proof": source.get("natural_reset_proof"),
        "reset_evidence": source.get("reset_evidence"), "evaluation_configuration": source["evaluation_configuration"],
        "source_acceptance_error_unchanged": source.get("source_acceptance_error"), "optimizer_updates": 0,
        "physical_ticks": source["episode_physics_ticks"], "physical_duration_s": source["episode_physics_ticks"] / HZ,
        "media_duration_s": count / FPS, "frame_count": count, "frame_window": window.as_dict(),
        "first_actual_frame_tick": ledger[0].sim_step, "last_actual_frame_tick": ledger[-1].sim_step,
        "terminal_frame_quantization_s": count / FPS - source["episode_physics_ticks"] / HZ,
        "pre_roll_available": False, "pre_roll_limitation": "Frozen capture starts at the first executed interval endpoint; tick0 physical evidence exists but no preceding encoded frames are available.",
        "original_preserved": True, "reencoded": False, "speed_modified": False, "stitched": False,
        "trimmed": False, "frame_interpolation": False, "decoded_frame_pts_key_sequence_equal": True,
        "remux_command": command, "source_validation": compact_validation(source_validation),
        "output_validation": compact_validation(validation), "previews": previews(output, count, ffmpeg),
    }
    write_new_json(output.with_suffix(".media.json"), receipt)
    print(json.dumps({key: receipt[key] for key in ("output", "physical_result", "media_validity", "terminal_phase", "physical_duration_s", "frame_count", "previews")}, indent=2))


def compare(left_path, right_path, output):
    left, right = read_json(left_path), read_json(right_path)
    require(left["label"] == "B0" and right["label"] == "C0", "Wrong comparison order")
    require(left["first_actual_frame_tick"] == right["first_actual_frame_tick"] == 8,
            "This paired task-window export requires both first actual frames at tick8; no inferred retiming")
    for key in ("camera", "seed", "evaluation_configuration", "runtime_contract"):
        require(left[key] == right[key], f"Paired capture differs in {key}")
    require(left["natural_reset_proof"]["entry"] == right["natural_reset_proof"]["entry"], "Initial logical state differs")
    # Physical initial-state equality is independently measured by the run audit, not inferred from seed.
    require(left["media_validity"] == right["media_validity"] == "COMPLETE_PLAYABLE", "Invalid input media")
    output = Path(output).resolve()
    require(not output.exists() and not output.with_suffix(".media.json").exists(), "Output exists; never overwrite")
    ffmpeg = find_ffmpeg()
    count = max(left["frame_count"], right["frame_count"])
    require(count / FPS <= 200, "Comparison exceeds 200 seconds")
    font = "C\\:/Windows/Fonts/arial.ttf"
    require(Path("C:/Windows/Fonts/arial.ttf").is_file(), "Comparison label font unavailable")
    filters = []
    for index, item in enumerate((left, right)):
        caption = f"{item['label']} | {item['physical_result']} | ends {item['physical_duration_s']:.3f}s | {item['terminal_phase']}"
        require(all(character.isalnum() or character in " |._-" for character in caption), "Unsafe caption")
        filter_text = (f"[{index}:v]setpts=PTS-STARTPTS,scale=960:540,pad=960:610:0:70:black,"
                       f"tpad=stop_mode=clone:stop={count - item['frame_count']},setpts=N/(15*TB),"
                       f"drawtext=fontfile='{font}':text='{caption}':fontcolor=white:fontsize=24:x=15:y=10,"
                       f"drawtext=fontfile='{font}':text='1x real elapsed time - 15 Hz interval endpoints':fontcolor=white:fontsize=18:x=15:y=42,"
                       f"drawtext=fontfile='{font}':text='RUN ENDED - FROZEN TERMINAL FRAME':fontcolor=yellow:fontsize=24:x=15:y=570:"
                       f"enable='gte(n,{item['frame_count']})'[v{index}]")
        filters.append(filter_text)
    filters.append("[v0][v1]hstack=inputs=2:shortest=1,format=yuv420p[out]")
    command = [str(ffmpeg), "-hide_banner", "-nostdin", "-v", "error", "-n", "-i", left["output"], "-i", right["output"],
               "-filter_complex_threads", "1", "-filter_complex", ";".join(filters), "-map", "[out]", "-an", "-frames:v", str(count),
               "-r", "15", "-fps_mode", "cfr", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-threads", "2", "-movflags", "+faststart", str(output)]
    run(command)
    validation = validate_mp4(output, ffmpeg=ffmpeg, expected_frame_count=count, require_sane_container_duration=True,
                              expected_width=1920, expected_height=610)
    require(validation["valid"] is True, "Comparison failed full decode")
    receipt = {
        "schema": "wlr50_clean.diagnostic_comparison.v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output": str(output), "sources": [str(Path(left_path).resolve()), str(Path(right_path).resolve())],
        "physical_result": {item["label"]: item["physical_result"] for item in (left, right)},
        "media_validity": "COMPLETE_PLAYABLE", "media_duration_s": count / FPS, "frame_count": count,
        "alignment": "Same elapsed P01 task-interval time; source ledger verified. No phase/event synchronization or time stretching.",
        "first_actual_frame_ticks": [left["first_actual_frame_tick"], right["first_actual_frame_tick"]],
        "terminal_quantization_s": {item["label"]: item["terminal_frame_quantization_s"] for item in (left, right)},
        "static_terminal_frames_added": {item["label"]: count - item["frame_count"] for item in (left, right)},
        "freeze_is_physical_evidence": False, "freeze_label": "RUN ENDED - FROZEN TERMINAL FRAME",
        "derived_pts": "N/15 after padding, identical to both verified 15 Hz source grids before the explicit static tail; avoids invalid padded PTS from FFmpeg tpad/hstack.",
        "new_physical_ticks": 0, "speed_modified": False, "phase_sync": False, "leg_labels": None,
        "initial_physical_state_equality": "See independently measured pair audit; seed and logical entry alone are not proof.",
        "pre_roll_available": False, "validation": compact_validation(validation), "command": command,
        "previews": previews(output, count, ffmpeg),
    }
    write_new_json(output.with_suffix(".media.json"), receipt)
    print(json.dumps({key: receipt[key] for key in ("output", "media_validity", "media_duration_s", "frame_count", "static_terminal_frames_added", "previews")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    single = sub.add_parser("export")
    single.add_argument("--source", required=True)
    single.add_argument("--output", required=True)
    single.add_argument("--label", choices=("B0", "C0", "D_RR_OFF"), required=True)
    pair = sub.add_parser("compare")
    pair.add_argument("--left", required=True)
    pair.add_argument("--right", required=True)
    pair.add_argument("--output", required=True)
    args = parser.parse_args()
    # Bound CPU media work even inside the reused decoder/validator functions.
    original_run = subprocess.run
    def bounded_run(command, *positional, **keywords):
        if isinstance(command, (list, tuple)) and "ffmpeg" in Path(command[0]).name.lower():
            command = [command[0], "-threads", "2", "-filter_threads", "1", "-filter_complex_threads", "1", *command[1:]]
        return original_run(command, *positional, **keywords)
    subprocess.run = bounded_run
    try:
        if args.command == "export":
            export(args.source, args.output, args.label)
        else:
            compare(args.left, args.right, args.output)
    finally:
        subprocess.run = original_run


if __name__ == "__main__":
    main()
