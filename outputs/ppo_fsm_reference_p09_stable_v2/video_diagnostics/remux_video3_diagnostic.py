"""Output-only, one-shot packet-copy repair; never publishes task success."""
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

RUN = "20260910T1138465345417Z_g7db0d17f398d_9bd793c700bd4040b4c517a0cd625ebb"
SOURCE_DIR = PROJECT / "runs/ppo_fsm_reference_p09_stable_v2/video_eval/validation" / RUN / "source"
SOURCE = SOURCE_DIR / "actual_viewport_video.mp4"
OUTPUT = Path(__file__).resolve().parent / "video3_145920_p01_incomplete.mp4"
EXPECTED_SOURCE_SHA = "2c770e40365925c1a829a893c5518b152bd614bf26efac8a7438c4582d5afc79"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def delta_hash(rows) -> str:
    return hashlib.sha256(b"".join(
        struct.pack(">d", right.pts_s - left.pts_s)
        for left, right in zip(rows, rows[1:])
    )).hexdigest()


def main() -> None:
    require(not OUTPUT.exists(), "Refusing to overwrite a diagnostic output")
    require(SOURCE.stat().st_size == 71009845, "Unexpected original byte count")
    require(sha256_file(SOURCE) == EXPECTED_SOURCE_SHA, "Unexpected original SHA256")
    source_manifest = json.loads((SOURCE_DIR / "semantic_video_source_manifest.json").read_text(encoding="utf-8"))
    require(source_manifest["physical_task_success"] is False, "This repair is restricted to the incomplete diagnostic capture")
    require(source_manifest["diagnostic_only"] is True, "Source must remain diagnostic-only")
    ffmpeg = find_ffmpeg()
    source_validation = validate_mp4(SOURCE, ffmpeg=ffmpeg, expected_frame_count=758, require_sane_container_duration=False)
    require(source_validation["valid"] is True, "Original failed decode/timing/content validation")
    source_frames = decode_frame_timeline(SOURCE, ffmpeg=ffmpeg)
    command = [str(ffmpeg), "-hide_banner", "-nostdin", "-v", "error", "-n",
               "-copyts", "-start_at_zero", "-i", str(SOURCE), "-map", "0:v:0",
               "-an", "-sn", "-dn", "-c:v", "copy", "-movflags", "+faststart", str(OUTPUT)]
    completed = subprocess.run(command, capture_output=True, text=True, errors="replace")
    require(completed.returncode == 0, "Remux failed: " + completed.stderr[-2000:])
    output_validation = validate_mp4(OUTPUT, ffmpeg=ffmpeg, expected_frame_count=758, require_sane_container_duration=True)
    require(output_validation["valid"] is True, "Repaired copy failed strict MP4 validation")
    require(0 < output_validation["container_duration_s"] <= 200, "Repaired container exceeds 200 seconds")
    output_frames = decode_frame_timeline(OUTPUT, ffmpeg=ffmpeg)
    require(source_frames == output_frames, "Decoded frame/PTS/key-frame sequence changed")
    require(delta_hash(source_frames) == delta_hash(output_frames), "PTS delta sequence changed")
    require(SOURCE.stat().st_size == 71009845 and sha256_file(SOURCE) == EXPECTED_SOURCE_SHA,
            "Original file no longer matches pre-remux identity")
    compact_keys = ("sha256", "bytes", "valid", "full_decode", "frame_count", "fps", "resolution",
                    "codec", "pixel_format", "duration_s", "container_duration_s", "container_duration_valid",
                    "first_pts_s", "last_pts_s", "frame_pts_sha256", "decoded_frame_checksums_sha256",
                    "unique_frame_checksums", "black_like_frame_count", "timestamps_monotonic", "timestamps_continuous")
    receipt = {
        "schema": "wlr50_clean.diagnostic_packet_copy_receipt.v1",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "diagnostic_only": True, "physical_task_success": False, "success_publication": False,
        "checkpoint_policy_decisions": 145920, "source_run_id": RUN,
        "source": str(SOURCE), "output": str(OUTPUT),
        "source_manifest": str(SOURCE_DIR / "semantic_video_source_manifest.json"),
        "source_acceptance_error_unchanged": source_manifest["source_acceptance_error"],
        "original_preserved": True, "reencoded": False, "stitched": False,
        "trimmed": False, "speed_modified": False, "interpolated": False,
        "remux_command": command, "remux_returncode": completed.returncode,
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
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
