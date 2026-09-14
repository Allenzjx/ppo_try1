"""Video6-only copy of the prior diagnostic repair; old helper stays unchanged."""
from __future__ import annotations

import hashlib
import json
import struct
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT / "src"))
from wlr50_clean.infrastructure.video_capture import find_ffmpeg, sha256_file, validate_mp4
from wlr50_clean.evaluation.video_timeline import decode_frame_timeline

RUN = "20260910T1528556872035Z_g7db0d17f398d_a5db22d8e4b545969459f6ea059a6c87"
SOURCE_DIR = PROJECT / "runs/ppo_fsm_reference_p09_stable_v2/video_eval/validation" / RUN / "source"
SOURCE = SOURCE_DIR / "actual_viewport_video.mp4"
OUTPUT = Path(__file__).resolve().parent / "video6_151936_p01_incomplete.mp4"
EXPECTED_SOURCE_SHA = "20f8ee06793334e1b66f84b6f46b7198f5d68e1ccd23631c2e3a1673b7c62bdc"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def delta_hash(rows) -> str:
    return hashlib.sha256(b"".join(
        struct.pack(">d", right.pts_s - left.pts_s)
        for left, right in zip(rows, rows[1:])
    )).hexdigest()


def main() -> None:
    require(sys.argv[1:] in ([], ["--verify-existing"]), "Only --verify-existing is supported")
    verify_existing = sys.argv[1:] == ["--verify-existing"]
    require(OUTPUT.is_file() if verify_existing else not OUTPUT.exists(), "Existing-output mode mismatch; never overwrite")
    require(SOURCE.stat().st_size == 110515863, "Unexpected original byte count")
    source_manifest_path = SOURCE_DIR / "semantic_video_source_manifest.json"
    original_manifest_bytes = source_manifest_path.read_bytes()
    source_manifest = json.loads(original_manifest_bytes)
    require(source_manifest["physical_task_success"] is False, "This repair is restricted to the incomplete diagnostic capture")
    require(source_manifest["diagnostic_only"] is True, "Source must remain diagnostic-only")
    capture_manifest = json.loads((SOURCE_DIR / "viewport_buffer_video_manifest.json").read_text(encoding="utf-8"))
    ffmpeg = find_ffmpeg(capture_manifest["full_decode"]["ffmpeg_path"])
    source_validation = validate_mp4(SOURCE, ffmpeg=ffmpeg, expected_frame_count=1176, require_sane_container_duration=False)
    require(source_validation["valid"] is True, "Original failed decode/timing/content validation")
    require(source_validation["sha256"] == EXPECTED_SOURCE_SHA, "Unexpected original SHA256")
    source_frames = decode_frame_timeline(SOURCE, ffmpeg=ffmpeg)
    command = [str(ffmpeg), "-hide_banner", "-nostdin", "-v", "error", "-n",
               "-copyts", "-start_at_zero", "-i", str(SOURCE), "-map", "0:v:0",
               "-an", "-sn", "-dn", "-c:v", "copy", "-movflags", "+faststart", str(OUTPUT)]
    completed = None if verify_existing else subprocess.run(command, capture_output=True, text=True, errors="replace")
    require(completed is None or completed.returncode == 0, "Remux failed: " + (completed.stderr[-2000:] if completed else ""))
    output_validation = validate_mp4(OUTPUT, ffmpeg=ffmpeg, expected_frame_count=1176, require_sane_container_duration=True)
    require(output_validation["valid"] is True, "Repaired copy failed strict MP4 validation")
    require(0 < output_validation["container_duration_s"] <= 200, "Repaired container exceeds 200 seconds")
    output_frames = decode_frame_timeline(OUTPUT, ffmpeg=ffmpeg)
    require(source_frames == output_frames, "Decoded frame/PTS/key-frame sequence changed")
    require(delta_hash(source_frames) == delta_hash(output_frames), "PTS delta sequence changed")
    require(SOURCE.stat().st_size == 110515863 and sha256_file(SOURCE) == EXPECTED_SOURCE_SHA,
            "Original file no longer matches pre-remux identity")
    require(source_manifest_path.read_bytes() == original_manifest_bytes, "Source manifest changed")
    preview_paths = [OUTPUT.with_name("video6_151936_decoded_01.png"), OUTPUT.with_name("video6_151936_decoded_02.png")]
    require(all(path.is_file() for path in preview_paths) if verify_existing else not any(path.exists() for path in preview_paths), "Preview mode mismatch; never overwrite")
    preview_command = [str(ffmpeg), "-hide_banner", "-nostdin", "-v", "error", "-n",
                       "-i", str(OUTPUT), "-map", "0:v:0", "-vf", "select=eq(n\\,0)+eq(n\\,1175)",
                       "-fps_mode", "passthrough", "-frames:v", "2", "-threads", "1",
                       str(OUTPUT.with_name("video6_151936_decoded_%02d.png"))]
    preview_result = None if verify_existing else subprocess.run(preview_command, capture_output=True, text=True, errors="replace")
    require(preview_result is None or preview_result.returncode == 0, "Preview extraction failed: " + (preview_result.stderr[-1000:] if preview_result else ""))
    preview_records = []
    for path, index in zip(preview_paths, (0, 1175)):
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
        "checkpoint_policy_decisions": 151936, "source_run_id": RUN,
        "source": str(SOURCE), "output": str(OUTPUT),
        "source_manifest": str(SOURCE_DIR / "semantic_video_source_manifest.json"),
        "source_acceptance_error_unchanged": source_manifest["source_acceptance_error"],
        "original_preserved": True, "source_manifest_bytes_unchanged": True,
        "source_manifest_sha256": hashlib.sha256(original_manifest_bytes).hexdigest(),
        "decoded_preview_pngs": preview_records, "preview_extraction_command": preview_command, "reencoded": False, "stitched": False,
        "trimmed": False, "speed_modified": False, "interpolated": False,
        "remux_command": command, "remux_returncode": 0 if completed is None else completed.returncode,
        "remux_executed_this_verification": not verify_existing,
        "source_validation": {k: source_validation[k] for k in compact_keys},
        "output_validation": {k: output_validation[k] for k in compact_keys},
        "decoded_frame_pts_key_flag_sequence_exactly_equal": True,
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
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
