"""Selected real video frames bound to existing publication receipts and ledgers.

No simulation, no AI images, no source mutation. Contact sheets are sparse
visual viewing aids, not joint/force/CoM measurements or complete frame review.
"""
from __future__ import annotations
import hashlib
import json
import math
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

PROJECT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / "src"))
from wlr50_clean.infrastructure.video_capture import find_ffmpeg


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    afile = Path("C:/robotics_sim/wlr_robot/fsm_50mm_recording_shaped_clean_v1/outputs/final/video_validation.json")
    bfile = PROJECT / "outputs/ppo_rr_video_diagnosis_v1/videos/zero_residual_before.media.json"
    cfile = PROJECT / "outputs/ppo_fsm_reference_p09_stable_v2/video_diagnostics/video11_164736_p01_safety_abort.remux_receipt.json"
    am, bm, cm = read(afile), read(bfile), read(cfile)
    a = am["videos"]["fsm"]
    assert am["fsm_publication"]["decoded_frames_unchanged"] and am["fsm_publication"]["source_frame_range"] == [0, 1618]
    specs = [
        ("A_FSM", "fsm_50mm_physical_success_clean.mp4", afile, a,
         Path(a["source_video"]).parent / "viewport_frame_ledger.jsonl", None,
         [0, 14, 28, 42, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 78, 90, 102, 107.8666667]),
        ("B0_ZERO", "zero_residual_before.mp4", bfile, bm["output_validation"],
         Path(bm["source_manifest"]).parent / "viewport_frame_ledger.jsonl", None,
         [0, 12, 24, 36, 38, 41, 44, 47, 50, 52, 64, 73.8]),
        ("C_HISTORICAL", "video11_164736_p01_safety_abort (1).mp4", cfile, cm["output_validation"],
         Path(cm["source_manifest"]).parent / "viewport_frame_ledger.jsonl", 164736,
         [0, 1.5, 3, 4.5, 6, 7.5, 9, 10.9333333]),
    ]
    ffmpeg = find_ffmpeg()
    folder = OUTPUT / "input_video_samples"
    folder.mkdir(parents=True, exist_ok=False)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 19)
    results = []
    for label, name, receipt, validation, ledger_path, checkpoint, times in specs:
        downloaded = Path("C:/Users/kskzz/Downloads") / name
        digest = sha(downloaded)
        assert digest == validation["sha256"] and downloaded.stat().st_size == validation["bytes"], "Selected download differs from published source"
        ledger = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(ledger) == validation["frame_count"]
        assert [row["encoded_frame_index"] for row in ledger] == list(range(len(ledger)))
        assert all(row["width"] == 1280 and row["height"] == 720 for row in ledger)
        indices = sorted(set(min(len(ledger) - 1, int(round(value * 15))) for value in times))
        pattern = folder / f"{label}_selected_%02d.png"
        select = "+".join(f"eq(n\\,{index})" for index in indices)
        command = [str(ffmpeg), "-hide_banner", "-nostdin", "-threads", "2", "-filter_threads", "1", "-n",
                   "-i", str(downloaded), "-map", "0:v:0", "-vf", f"select={select},showinfo",
                   "-fps_mode", "passthrough", "-frames:v", str(len(indices)), "-threads", "1", str(pattern)]
        process = subprocess.run(command, capture_output=True, text=True, errors="replace")
        assert process.returncode == 0, process.stderr[-1500:]
        pts = [float(value) for value in re.findall(r"\bn:\s*\d+\s+pts:\s*-?\d+\s+pts_time:\s*([-+0-9.eE]+)", process.stderr)]
        assert len(pts) == len(indices) and all(abs(p - i / 15) < 0.00005 for i, p in zip(indices, pts))
        samples = []
        for position, (index, media_pts) in enumerate(zip(indices, pts), 1):
            path = folder / f"{label}_selected_{position:02d}.png"
            with Image.open(path) as decoded:
                assert decoded.size == (1280, 720)
                decoded.load()
            row = ledger[index]
            samples.append({"path": str(path), "source_frame_index": index, "media_pts_s": media_pts,
                            "ledger_sim_tick": row["sim_step"], "ledger_sim_time_s": row["sim_time_s"]})
        sheets = []
        for start in range(0, len(samples), 6):
            group = samples[start:start + 6]
            sheet = Image.new("RGB", (1280, 414 * math.ceil(len(group) / 2) + 32), "#151719")
            draw = ImageDraw.Draw(sheet)
            draw.text((12, 6), f"{label} | Sparse visual samples; NOT physical measurements", font=font, fill="white")
            for number, sample in enumerate(group):
                x, y = (number % 2) * 640, (number // 2) * 414 + 32
                with Image.open(sample["path"]) as source:
                    sheet.paste(source.convert("RGB").resize((640, 360)), (x, y))
                draw.text((x + 8, y + 364), f"frame {sample['source_frame_index']} | media PTS {sample['media_pts_s']:.3f}s", font=font, fill="white")
                draw.text((x + 8, y + 388), f"ledger sim {sample['ledger_sim_time_s']:.3f}s | tick {sample['ledger_sim_tick']}", font=font, fill="#9dd8e8")
            path = folder / f"{label}_contact_{start // 6 + 1:02d}.png"
            sheet.save(path)
            sheets.append(str(path))
        results.append({"label": label, "selected_download": str(downloaded), "download_sha256": digest,
            "bytes": downloaded.stat().st_size, "matches_published_media_bytes": True,
            "original_receipt": str(receipt), "ledger": str(ledger_path), "checkpoint": checkpoint,
            "existing_full_media_decode": {key: validation.get(key) for key in ("valid", "full_decode", "frame_count", "fps", "resolution", "duration_s")},
            "publication_ledger_binding": "all source frames retained in original order; identical published-media bytes",
            "attachment_boundary": "Download matching the named input is byte-identical to the published media. No attachment-specific provider URI was supplied.",
            "samples": samples, "contact_sheets": sheets, "ffmpeg_command": command,
            "ffmpeg_actual_extraction_exit": process.returncode})
    report = {"schema": "wlr50_clean.selected_input_media_samples.v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "purpose": "full-time-domain sparse viewing plus denser requested windows, not complete frame-by-frame review",
              "physical_measurements_from_images": False, "leg_labels_guessed": False,
              "sources_unchanged": True, "new_simulation_runs": 0, "inputs": results}
    target = OUTPUT / "input_video_identity_and_samples.json"
    with target.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps({"receipt": str(target), "inputs": [{"label": x["label"], "byte_match": True, "samples": len(x["samples"]), "sheets": x["contact_sheets"]} for x in results]}, indent=2))


if __name__ == "__main__":
    main()
