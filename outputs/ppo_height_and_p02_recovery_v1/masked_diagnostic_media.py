"""Standalone bounded wheels4 intervention remux; never enters formal B/C exporter."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "outputs/ppo_rr_video_diagnosis_v1/diagnostic_media.py"
spec = importlib.util.spec_from_file_location("verified_media_primitives", HELPER)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    directory, output = args.source.resolve(), args.output.resolve()
    source_path = directory / "actual_viewport_video.mp4"
    manifest = directory / "semantic_video_source_manifest.json"
    capture_path = directory / "viewport_buffer_video_manifest.json"
    source_bytes, capture_bytes = manifest.read_bytes(), capture_path.read_bytes()
    source, capture = json.loads(source_bytes), json.loads(capture_bytes)
    run = base.read_json(directory.parent / "run_manifest.json")
    intervention = source.get("diagnostic_intervention") or {}
    base.require(run.get("completed_at_utc") and run.get("runtime_contract") == source.get("runtime_contract"), "Not finalized/runtime-bound")
    base.require(source.get("role") == "C" and intervention.get("masked_raw_indices") == [8, 9, 10, 11]
        and intervention.get("mode") == "DIAGNOSTIC_WHEELS4_RAW_OFF_FROM_FIRST_DECISION"
        and source.get("formal_ppo_success") is False and source.get("formal_P02_evidence") is False,
        "Requires explicit wheels4 diagnostic, never formal C0")
    base.require(source.get("diagnostic_bounded_stop") is True and source.get("diagnostic_stop_reason") == "DIAGNOSTIC_BOUNDED_WINDOW"
        and source.get("episode_physics_ticks") == intervention.get("maximum_diagnostic_physics_ticks") == 1920,
        "This exporter requires the verified 16s bounded diagnostic, not an inferred cut")
    base.require(source.get("optimizer_updates") == 0 and source.get("on_policy_training_samples") == 0
        and source.get("frozen_model_unchanged_after_capture") is True, "Not unchanged frozen diagnostic")
    base.require(source.get("from_phase") == "P01" and source.get("pre_action_ticks") == 0
        and source.get("fresh_process_single_episode") is True and source.get("episode_count") == 1, "Reset/capture mismatch")
    base.require(capture.get("valid") is True and capture.get("encoder_finalized_before_app_close") is True, "Writer not finalized")
    base.require(not output.exists() and not output.with_suffix(".media.json").exists(), "Never overwrite")
    ffmpeg = base.find_ffmpeg(capture["full_decode"]["ffmpeg_path"])
    count = capture["frame_count"]
    before = base.validate_mp4(source_path, ffmpeg=ffmpeg, expected_frame_count=count, require_sane_container_duration=False)
    base.require(before["valid"] and before["sha256"] == capture["video_sha256"], "Original capture mismatch/decode failure")
    frames = base.decode_frame_timeline(source_path, ffmpeg=ffmpeg)
    ledger = base.load_viewport_frame_ledger(directory / "viewport_frame_ledger.jsonl")
    window = base.task_interval_action_window(source, frames, ledger)
    base.require(count == 240 and window.output_duration_s == 16 and ledger[0].sim_step == 8 and ledger[-1].sim_step == 1920, "Incomplete diagnostic timeline")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [str(ffmpeg), "-hide_banner", "-nostdin", "-v", "error", "-n", "-copyts", "-start_at_zero",
        "-i", str(source_path), "-map", "0:v:0", "-an", "-sn", "-dn", "-c:v", "copy", "-movflags", "+faststart", str(output)]
    stat = (source_path.stat().st_size, source_path.stat().st_mtime_ns)
    base.run(command)
    after = base.validate_mp4(output, ffmpeg=ffmpeg, expected_frame_count=count, require_sane_container_duration=True)
    base.require(after["valid"] and after["container_duration_s"] <= 200, "Remux media invalid")
    base.require(frames == base.decode_frame_timeline(output, ffmpeg=ffmpeg), "Decoded/PTS/keyframe sequence changed")
    base.require(stat == (source_path.stat().st_size, source_path.stat().st_mtime_ns)
        and manifest.read_bytes() == source_bytes and capture_path.read_bytes() == capture_bytes, "Original changed")
    phase, phase_origin, decision = base.final_phase(directory)
    receipt = dict(schema="wlr50_clean.masked_diagnostic_media.v1", label="DIAGNOSTIC ONLY | wheel raw8-11 OFF | bounded16s",
        formal_C0=False, formal_P02_evidence=False, formal_ppo_success=False, physical_task_success=source["physical_task_success"],
        physical_result="DIAGNOSTIC_BOUNDED_WINDOW_TASK_INCOMPLETE", termination_reason=source["diagnostic_stop_reason"],
        controller_termination_reason=(decision.get("step_info") or {}).get("termination_reason"),
        terminal_phase=phase, terminal_phase_provenance=phase_origin, optimizer_updates=0, on_policy_training_samples=0,
        diagnostic_intervention=intervention, source=str(source_path), source_manifest=str(manifest), source_run=str(directory.parent),
        source_manifest_sha256=hashlib.sha256(source_bytes).hexdigest(), output=str(output), runtime_contract=source["runtime_contract"],
        checkpoint_load_provenance=source["checkpoint_load_provenance"], media_validity="COMPLETE_PLAYABLE", frame_count=count,
        fps=15, physical_ticks=1920, physical_duration_s=16, media_duration_s=16, first_actual_frame_tick=8, last_actual_frame_tick=1920,
        pre_roll_available=False, pre_roll_limitation="No encoded tick0 or settling frames; frame0 is actual physical tick8.",
        original_preserved=True, reencoded=False, speed_modified=False, stitched=False, trimmed=False, frame_interpolation=False,
        decoded_frame_pts_key_sequence_equal=True, label_presentation="filename/receipt/player caption; individual video remains pure simulation",
        source_validation=base.compact_validation(before), output_validation=base.compact_validation(after),
        previews=base.previews(output, count, ffmpeg), remux_command=command, formal_validator_modified=False,
        exporter=str(Path(__file__).resolve()), exporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    base.write_new_json(output.with_suffix(".media.json"), receipt)
    print(json.dumps({k: receipt[k] for k in ("output", "label", "terminal_phase", "media_validity", "frame_count", "previews")}))


if __name__ == "__main__":
    original_run = subprocess.run
    def bounded_run(command, *args, **kwargs):
        if isinstance(command, (list, tuple)) and "ffmpeg" in Path(command[0]).name.lower():
            command = [command[0], "-threads", "2", "-filter_threads", "1", "-filter_complex_threads", "1", *command[1:]]
        return original_run(command, *args, **kwargs)
    subprocess.run = bounded_run
    try: main()
    finally: subprocess.run = original_run
