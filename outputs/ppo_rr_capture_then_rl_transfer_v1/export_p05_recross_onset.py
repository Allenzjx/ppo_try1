"""Create the bounded CP220544 P05 recross-onset excerpt from its reviewed full video.

This outputs-only helper never reads the raw viewport video or starts Isaac.  It
binds the existing review receipt and sealed frame ledger, cuts one contiguous
25--65 second interval from the already validated HUD video, and validates all
600 decoded frames before publishing a separate receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any

from PIL import Image


FPS = 15
START_FRAME = 375
END_FRAME = 975
FRAME_COUNT = END_FRAME - START_FRAME
OUTPUT_NAME = "CP220544_DET_P05_recross_onset_detail.mp4"
RECEIPT_NAME = "onset_receipt.json"
TITLE = (
    "LOCAL ONSET 25-65s | FL cross 29.533s | includes rejected 50-60s window "
    "| complete tail is in FULL"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}\n"
            f"{completed.stderr[-4000:]}"
        )
    return completed


def selected_ledger_rows(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    selected: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig") as stream:
        for index, line in enumerate(stream):
            if index in (START_FRAME, END_FRAME - 1):
                value = json.loads(line)
                require(isinstance(value, dict), "frame-ledger row must be an object")
                selected[index] = value
            if index >= END_FRAME - 1:
                break
    require(set(selected) == {START_FRAME, END_FRAME - 1},
            "sealed ledger does not cover the requested onset interval")
    for index, row in selected.items():
        require(row.get("encoded_frame_index") == index, "frame ledger index mismatch")
        require(row.get("callback_count") == 1, "onset frame lacks exactly one callback")
    return selected[START_FRAME], selected[END_FRAME - 1]


def decode_validation(path: Path, ffmpeg: Path) -> dict[str, Any]:
    decoded = run([
        str(ffmpeg), "-hide_banner", "-nostdin", "-v", "error", "-threads", "2",
        "-i", str(path), "-map", "0:v:0", "-an", "-f", "framemd5", "-",
    ]).stdout
    lines = [line.strip() for line in decoded.splitlines() if line.strip()]
    require("#tb 0: 1/15" in lines, "decoded time base is not 1/15")
    require("#dimensions 0: 1280x720" in lines, "decoded dimensions changed")
    frames = [line for line in lines if not line.startswith("#")]
    require(len(frames) == FRAME_COUNT, "full decode frame count mismatch")
    pts: list[int] = []
    durations: list[int] = []
    for line in frames:
        fields = [item.strip() for item in line.split(",", 5)]
        require(len(fields) == 6, "unexpected framemd5 row")
        require(int(fields[0]) == 0, "unexpected decoded stream index")
        pts.append(int(fields[2]))
        durations.append(int(fields[3]))
    require(pts == list(range(FRAME_COUNT)), "decoded PTS grid is not continuous 15 fps")
    require(set(durations) == {1}, "decoded frame duration changed")

    black = run([
        str(ffmpeg), "-hide_banner", "-nostdin", "-v", "info", "-threads", "2",
        "-i", str(path), "-vf", "blackframe=amount=98:threshold=16", "-an",
        "-f", "null", os.devnull,
    ]).stderr
    black_count = len(re.findall(r"Parsed_blackframe[^\n]*pblack:", black))
    require(black_count == 0, "derived onset clip contains a black-like frame")
    return {
        "valid": True,
        "full_decode": True,
        "frame_count": FRAME_COUNT,
        "fps": float(FPS),
        "resolution": [1280, 720],
        "duration_s": FRAME_COUNT / FPS,
        "first_pts_s": 0.0,
        "last_pts_s": (FRAME_COUNT - 1) / FPS,
        "timestamps_monotonic": True,
        "timestamps_continuous": True,
        "black_like_frame_count": black_count,
        "decoded_frame_checksums_sha256": hashlib.sha256(
            ("\n".join(frames) + "\n").encode("ascii")
        ).hexdigest(),
    }


def make_previews(path: Path, ffmpeg: Path) -> list[dict[str, Any]]:
    pattern = path.with_name(path.stem + "_decoded_%02d.png")
    preview_paths = [path.with_name(path.stem + f"_decoded_{i:02d}.png") for i in (1, 2)]
    require(not any(item.exists() for item in preview_paths), "onset preview already exists")
    run([
        str(ffmpeg), "-hide_banner", "-nostdin", "-v", "error", "-n", "-threads", "1",
        "-i", str(path), "-vf", f"select=eq(n\\,0)+eq(n\\,{FRAME_COUNT - 1})",
        "-fps_mode", "passthrough", "-frames:v", "2", str(pattern),
    ])
    result: list[dict[str, Any]] = []
    for output, index in zip(preview_paths, (0, FRAME_COUNT - 1)):
        require(output.is_file() and output.stat().st_size > 0, "onset preview is missing")
        with Image.open(output) as image:
            require(image.size == (1280, 720), "onset preview dimensions changed")
        result.append({
            "path": str(output.resolve()),
            "frame_index": index,
            "sha256": sha256(output),
        })
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-dir", type=Path, required=True)
    args = parser.parse_args()
    review_dir = args.review_dir.resolve(strict=True)
    existing_receipt_path = review_dir / "export_receipt.json"
    existing = read_json(existing_receipt_path)
    require(existing.get("schema") == "wlr50_clean.rr_capture_video_export.v1",
            "unexpected parent export receipt")

    full_record = existing.get("full") or {}
    validation = full_record.get("validation") or {}
    full = Path(full_record.get("output", "")).resolve(strict=True)
    require(full.parent == review_dir and full.name == "CP220544_DET_full_RR_RL_attempt.mp4",
            "parent full video identity changed")
    require(full_record.get("frame_count") == 3000 and validation.get("valid") is True and
            validation.get("full_decode") is True and validation.get("frame_count") == 3000 and
            validation.get("fps") == 15.0 and validation.get("duration_s") == 200.0,
            "parent full video was not the reviewed 3000-frame artifact")
    require(full.stat().st_size == validation.get("bytes"), "parent full video size changed")
    parent_sha = validation.get("sha256")
    require(isinstance(parent_sha, str) and len(parent_sha) == 64,
            "parent full SHA is missing from its receipt")

    source = Path(existing.get("source", "")).resolve(strict=True)
    manifest_path = source / "semantic_video_source_manifest.json"
    manifest = read_json(manifest_path)
    require(sha256(manifest_path) == existing.get("source_manifest_sha256"),
            "sealed source manifest differs from the parent receipt")
    ledger_path = source / "viewport_frame_ledger.jsonl"
    ledger_record = (manifest.get("artifacts") or {}).get(ledger_path.name) or {}
    require(ledger_path.is_file() and ledger_path.stat().st_size == ledger_record.get("bytes") and
            sha256(ledger_path) == ledger_record.get("sha256"),
            "sealed frame ledger differs from its manifest")
    first_row, last_row = selected_ledger_rows(ledger_path)

    selected_path = Path(existing.get("selected_response_json", "")).resolve(strict=True)
    require(sha256(selected_path) == existing.get("selected_response_json_sha256"),
            "selected-response evidence differs from the parent receipt")
    selected = read_json(selected_path)
    cross = next((row for row in selected.get("rows", []) if row.get("tick") == 3544), None)
    require(isinstance(cross, dict) and cross.get("phase") == "P05" and
            cross.get("fl_capture_pending") is True and
            abs(float(cross.get("time_s")) - 29.53333333333333) < 1e-9,
            "the sealed selected response lacks the real FL cross onset")

    command_from_parent = full_record.get("command") or []
    require(command_from_parent, "parent receipt lacks its ffmpeg command")
    ffmpeg = Path(command_from_parent[0]).resolve(strict=True)
    output = review_dir / OUTPUT_NAME
    partial = review_dir / (output.stem + ".partial.mp4")
    receipt_path = review_dir / RECEIPT_NAME
    require(not output.exists() and not partial.exists() and not receipt_path.exists(),
            "onset output or receipt already exists")

    title = TITLE.replace(":", "\\:").replace("'", "\\'")
    command = [
        str(ffmpeg), "-hide_banner", "-nostdin", "-v", "error", "-n",
        "-threads", "2", "-filter_threads", "1", "-i", str(full),
        "-vf", (
            f"trim=start_frame={START_FRAME}:end_frame={END_FRAME},setpts=PTS-STARTPTS,"
            "drawbox=x=0:y=0:w=iw:h=20:color=black@0.92:t=fill,"
            "drawtext=fontfile='C\\:/Windows/Fonts/consola.ttf':"
            f"text='{title}':fontcolor=yellow:fontsize=14:x=14:y=2"
        ),
        "-an", "-frames:v", str(FRAME_COUNT), "-r", str(FPS), "-fps_mode", "cfr",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-threads", "2", "-movflags", "+faststart", str(partial),
    ]
    run(command)
    output_validation = decode_validation(partial, ffmpeg)
    output_validation.update({"sha256": sha256(partial), "bytes": partial.stat().st_size})
    partial.replace(output)
    previews = make_previews(output, ffmpeg)

    receipt = {
        "schema": "wlr50_clean.rr_capture_p05_recross_onset_export.v1",
        "classification": "LOCAL_SAME_RUN_P05_RECROSS_ONSET__RR_RL_NOT_REACHED",
        "output": str(output.resolve()),
        "parent_export_receipt": str(existing_receipt_path.resolve()),
        "parent_export_receipt_sha256": sha256(existing_receipt_path),
        "parent_full_video": str(full),
        "parent_full_video_sha256": parent_sha,
        "parent_full_sha256_source": "already_validated_parent_export_receipt_not_rehashed",
        "source_manifest": str(manifest_path.resolve()),
        "source_manifest_sha256": existing.get("source_manifest_sha256"),
        "source_frame_ledger": str(ledger_path.resolve()),
        "source_frame_ledger_sha256": ledger_record.get("sha256"),
        "source_frame_interval_half_open": [START_FRAME, END_FRAME],
        "parent_full_pts_interval_half_open_s": [START_FRAME / FPS, END_FRAME / FPS],
        "source_evidence_endpoints": {
            "first": {
                "encoded_frame_index": first_row.get("encoded_frame_index"),
                "physical_tick": first_row.get("sim_step"),
                "sim_time_s": first_row.get("sim_time_s"),
            },
            "last_included": {
                "encoded_frame_index": last_row.get("encoded_frame_index"),
                "physical_tick": last_row.get("sim_step"),
                "sim_time_s": last_row.get("sim_time_s"),
            },
        },
        "sealed_cross_evidence": {
            "physics_tick": cross.get("tick"),
            "sim_time_s": cross.get("time_s"),
            "phase": cross.get("phase"),
            "fl_capture_pending": cross.get("fl_capture_pending"),
            "fl_front_distance_mm": cross.get("fl_front_distance_mm"),
            "fl_gap_mm": cross.get("fl_gap_mm"),
        },
        "visible_title_overlay": TITLE,
        "same_single_run": True,
        "contiguous": True,
        "stitched": False,
        "normal_speed": True,
        "speed_modified": False,
        "full_failure_tail_location": str(full),
        "validation": output_validation,
        "previews": previews,
        "command": command,
    }
    temporary_receipt = review_dir / (RECEIPT_NAME + ".partial")
    require(not temporary_receipt.exists(), "partial onset receipt already exists")
    temporary_receipt.write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8", newline="\n",
    )
    temporary_receipt.replace(receipt_path)
    print(json.dumps({
        "output": str(output.resolve()),
        "receipt": str(receipt_path.resolve()),
        "validation": output_validation,
        "previews": previews,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
