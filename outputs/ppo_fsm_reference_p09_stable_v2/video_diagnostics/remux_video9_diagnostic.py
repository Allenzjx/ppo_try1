"""Video9-only copy of the prior diagnostic repair; old helper stays unchanged."""
from __future__ import annotations

import hashlib
import json
import re
import struct
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from fractions import Fraction

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT / "src"))
from wlr50_clean.infrastructure.video_capture import find_ffmpeg, sha256_file, validate_mp4
from wlr50_clean.evaluation.video_timeline import decode_frame_timeline

RUN = "20260910T2235582002787Z_gd4e46006b382_0f64bf3b35f74953994f8ca191952ee8"
SOURCE_DIR = PROJECT / "runs/ppo_fsm_reference_p09_stable_v2/video_eval/validation" / RUN / "source"
SOURCE = SOURCE_DIR / "actual_viewport_video.mp4"
OUTPUT = Path(__file__).resolve().parent / "video9_159104_p01_incomplete_p05.mp4"
EXPECTED_SOURCE_SHA = "3f9a6158657af9fc1bfbd5fe112a3e62a9f06d9912365e2e38747be5f3e74a05"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def delta_hash(rows) -> str:
    return hashlib.sha256(b"".join(
        struct.pack(">d", right.pts_s - left.pts_s)
        for left, right in zip(rows, rows[1:])
    )).hexdigest()


def packet_records(path: Path, ffmpeg: Path) -> tuple[list[dict], dict]:
    """Hash encoded packet payloads without decoding/reencoding; normalize time bases exactly."""
    command = [str(ffmpeg), "-hide_banner", "-nostdin", "-nostats", "-dump",
               "-i", str(path), "-map", "0:v:0", "-an", "-sn", "-dn",
               "-c:v", "copy", "-f", "framehash", "-hash", "sha256", "-"]
    result = subprocess.run(command, capture_output=True, text=True, errors="replace")
    require(result.returncode == 0, "Packet-hash command failed: " + result.stderr[-1200:])
    timebase_match = re.search(r"(?m)^#tb 0:\s*(\d+)/(\d+)\s*$", result.stdout)
    require(timebase_match is not None, "Missing packet time base")
    timebase = Fraction(int(timebase_match[1]), int(timebase_match[2]))
    rows = []
    for line in result.stdout.splitlines():
        if not line or line.startswith("#"):
            continue
        cells = [cell.strip() for cell in line.split(",")]
        require(len(cells) >= 6 and cells[0] == "0", "Unexpected packet hash row")
        rows.append({"dts_s_exact": str(int(cells[1]) * timebase),
                     "pts_s_exact": str(int(cells[2]) * timebase),
                     "duration_s_exact": str(int(cells[3]) * timebase),
                     "bytes": int(cells[4]), "payload_sha256": cells[5]})
    keyflags = [int(value) for value in re.findall(r"\bkeyframe=(\d+)\b", result.stderr)]
    require(len(rows) == 631 and len(keyflags) == len(rows), "Packet/key-flag count mismatch")
    for row, keyflag in zip(rows, keyflags):
        row["key_frame"] = bool(keyflag)
    summary = {"count": len(rows), "time_base": str(timebase), "command": command,
               "sha256_of_ordered_normalized_packet_records": hashlib.sha256(
                   json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest(),
               "sha256_of_ordered_payload_hashes": hashlib.sha256(
                   "\n".join(row["payload_sha256"] for row in rows).encode("ascii")).hexdigest(),
               "key_frames": sum(keyflags),
               "sha256_of_key_flags": hashlib.sha256(bytes(keyflags)).hexdigest(),
               "first": rows[0], "last": rows[-1]}
    return rows, summary


def main() -> None:
    require(sys.argv[1:] in ([], ["--verify-existing"]), "Only --verify-existing is supported")
    verify_existing = sys.argv[1:] == ["--verify-existing"]
    require(OUTPUT.is_file() if verify_existing else not OUTPUT.exists(), "Existing-output mode mismatch; never overwrite")
    require(SOURCE.stat().st_size == 59244888, "Unexpected original byte count")
    source_manifest_path = SOURCE_DIR / "semantic_video_source_manifest.json"
    original_manifest_bytes = source_manifest_path.read_bytes()
    source_manifest = json.loads(original_manifest_bytes)
    require(source_manifest["physical_task_success"] is False, "This repair is restricted to the incomplete diagnostic capture")
    require(source_manifest["diagnostic_only"] is True, "Source must remain diagnostic-only")

    require(source_manifest["checkpoint_load_provenance"]["saved_global_policy_decisions"] == 159104,
            "Unexpected loaded checkpoint counter")
    require(source_manifest["issued_policy_decisions"] == source_manifest["completed_environment_steps"] == 631,
            "Unexpected issued/returned decision count")
    require(source_manifest["physical_episode"]["observed_physics_ticks"] == 5045
            and abs(source_manifest["physical_episode"]["physical_task_duration_s"] - (5045 / 120.0)) < 1e-9,
            "Unexpected actual physical interval")
    capture_manifest_path = SOURCE_DIR / "viewport_buffer_video_manifest.json"
    original_capture_manifest_bytes = capture_manifest_path.read_bytes()
    capture_manifest = json.loads(original_capture_manifest_bytes)
    require(capture_manifest["full_decode"]["container_duration_valid"] is False,
            "Do not remux an already normal container")

    ffmpeg = find_ffmpeg(capture_manifest["full_decode"]["ffmpeg_path"])
    source_validation = validate_mp4(SOURCE, ffmpeg=ffmpeg, expected_frame_count=631, require_sane_container_duration=False)
    require(source_validation["valid"] is True, "Original failed decode/timing/content validation")
    require(source_validation["sha256"] == EXPECTED_SOURCE_SHA, "Unexpected original SHA256")
    source_frames = decode_frame_timeline(SOURCE, ffmpeg=ffmpeg)
    source_packets, source_packet_summary = packet_records(SOURCE, ffmpeg)
    command = [str(ffmpeg), "-hide_banner", "-nostdin", "-v", "error", "-n",
               "-copyts", "-start_at_zero", "-i", str(SOURCE), "-map", "0:v:0",
               "-an", "-sn", "-dn", "-c:v", "copy", "-movflags", "+faststart", str(OUTPUT)]
    completed = None if verify_existing else subprocess.run(command, capture_output=True, text=True, errors="replace")
    require(completed is None or completed.returncode == 0, "Remux failed: " + (completed.stderr[-2000:] if completed else ""))
    output_validation = validate_mp4(OUTPUT, ffmpeg=ffmpeg, expected_frame_count=631, require_sane_container_duration=True)
    require(output_validation["valid"] is True, "Repaired copy failed strict MP4 validation")
    require(0 < output_validation["container_duration_s"] <= 200, "Repaired container exceeds 200 seconds")
    output_frames = decode_frame_timeline(OUTPUT, ffmpeg=ffmpeg)
    output_packets, output_packet_summary = packet_records(OUTPUT, ffmpeg)
    require(source_packets == output_packets, "Encoded payload, exact normalized packet timestamps, or key flags changed")
    require(source_frames == output_frames, "Decoded frame/PTS/key-frame sequence changed")
    require(delta_hash(source_frames) == delta_hash(output_frames), "PTS delta sequence changed")
    require(SOURCE.stat().st_size == 59244888 and sha256_file(SOURCE) == EXPECTED_SOURCE_SHA,
            "Original file no longer matches pre-remux identity")
    require(source_manifest_path.read_bytes() == original_manifest_bytes, "Source manifest changed")
    require(capture_manifest_path.read_bytes() == original_capture_manifest_bytes, "Capture manifest changed")
    preview_paths = [OUTPUT.with_name("video9_159104_decoded_01.png"), OUTPUT.with_name("video9_159104_decoded_02.png")]
    require(all(path.is_file() for path in preview_paths) if verify_existing else not any(path.exists() for path in preview_paths), "Preview mode mismatch; never overwrite")
    preview_command = [str(ffmpeg), "-hide_banner", "-nostdin", "-v", "error", "-n",
                       "-i", str(OUTPUT), "-map", "0:v:0", "-vf", "select=eq(n\\,0)+eq(n\\,630)",
                       "-fps_mode", "passthrough", "-frames:v", "2", "-threads", "1",
                       str(OUTPUT.with_name("video9_159104_decoded_%02d.png"))]
    preview_result = None if verify_existing else subprocess.run(preview_command, capture_output=True, text=True, errors="replace")
    require(preview_result is None or preview_result.returncode == 0, "Preview extraction failed: " + (preview_result.stderr[-1000:] if preview_result else ""))
    preview_records = []
    for path, index in zip(preview_paths, (0, 630)):
        header = path.read_bytes()[:24]
        require(header[:8] == bytes.fromhex("89504e470d0a1a0a") and struct.unpack(">II", header[16:24]) == (1280, 720), "Invalid preview PNG")
        preview_records.append({"path": str(path), "source_frame_index": index,
                                "source_pts_s": output_frames[index].pts_s,
                                "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    compact_keys = ("sha256", "bytes", "valid", "full_decode", "frame_count", "fps", "resolution",
                    "codec", "pixel_format", "duration_s", "container_duration_s", "container_duration_valid",
                    "first_pts_s", "last_pts_s", "frame_pts_sha256", "decoded_frame_checksums_sha256",
                    "unique_frame_checksums", "black_like_frame_count", "timestamps_monotonic", "timestamps_continuous")
    receipt = {
        "schema": "wlr50_clean.diagnostic_packet_copy_receipt.v1",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "diagnostic_only": True, "physical_task_success": False, "success_publication": False,
        "checkpoint_policy_decisions": 159104, "source_run_id": RUN,
        "source": str(SOURCE), "output": str(OUTPUT),
        "source_manifest": str(SOURCE_DIR / "semantic_video_source_manifest.json"),
        "source_acceptance_error_unchanged": source_manifest["source_acceptance_error"],
        "original_preserved": True, "source_manifest_bytes_unchanged": True,
        "source_manifest_sha256": hashlib.sha256(original_manifest_bytes).hexdigest(),
        "source_capture_manifest_bytes_unchanged": True,
        "source_capture_manifest_sha256": hashlib.sha256(original_capture_manifest_bytes).hexdigest(),
        "physical_evaluator_termination_reason": source_manifest["physical_episode"]["physical_task_evaluation"]["termination_reason"],
        "decoded_preview_pngs": preview_records, "preview_extraction_command": preview_command, "reencoded": False, "stitched": False,
        "trimmed": False, "speed_modified": False, "interpolated": False,
        "remux_command": command, "remux_returncode": 0 if completed is None else completed.returncode,
        "remux_executed_this_verification": not verify_existing,
        "source_validation": {k: source_validation[k] for k in compact_keys},
        "output_validation": {k: output_validation[k] for k in compact_keys},
        "decoded_frame_pts_key_flag_sequence_exactly_equal": True,
        "packet_payload_pts_dts_duration_key_flag_sequence_exactly_equal": True,
        "source_packet_summary": source_packet_summary, "output_packet_summary": output_packet_summary,
        "task_outcome": "INCOMPLETE_CONTROLLER_BLOCKED_P05",
        "policy_decisions_issued": 631, "full_eight_tick_decisions": 630, "final_decision_physics_ticks": 5,
        "completed_environment_steps": 631, "interrupted_final_decision_ticks": 0,
        "source_pts_delta_sha256": delta_hash(source_frames),
        "output_pts_delta_sha256": delta_hash(output_frames),
        "physical_duration_s": source_manifest["physical_episode"]["physical_task_duration_s"],
        "physical_ticks": source_manifest["physical_episode"]["observed_physics_ticks"],
        "media_duration_s": len(output_frames) / 15.0,
        "final_frame_quantization_s": len(output_frames) / 15.0 - source_manifest["physical_episode"]["physical_task_duration_s"],
        "note": "Container metadata repair only; the fixed final-frame quantization already existed in the raw capture. No task verdict or production artifact was changed.",
    }
    receipt_path = OUTPUT.with_suffix(".remux_receipt.json")
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps({"receipt": str(receipt_path), "output": str(OUTPUT), "source_validation": receipt["source_validation"], "output_validation": receipt["output_validation"], "packet_records_equal": True, "preview_pngs": preview_records}, indent=2))


if __name__ == "__main__":
    main()





