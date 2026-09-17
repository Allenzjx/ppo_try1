"""Read only the last complete live decision; no success verdict before finalization."""
import json
from pathlib import Path
import sys

run = Path(sys.argv[1])
path = run / "source/video_policy_decisions.jsonl"
last = {}
if path.exists():
    with path.open("rb") as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - 2 * 1024 * 1024))
        for line in reversed(stream.read().splitlines()):
            try:
                last = json.loads(line)
                break
            except (ValueError, UnicodeDecodeError):
                pass
info = last.get("step_info", {})
task = info.get("semantic_task", {})
features = task.get("goal_features", {})
manifest_path = run / "run_manifest.json"
manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
print(json.dumps({
    "lifecycle": manifest.get("lifecycle", "running_or_not_finalized"),
    "decision": last.get("decision"), "phase": last.get("request_phase"),
    "tick": last.get("end_tick"), "placed_history": task.get("placed_history"),
    "crossed_history": task.get("front_edge_crossed_history"),
    "termination_reason": task.get("termination_reason"),
    "FR_clearance_m": features.get("FR_clearance_m"),
    "FR_front_distance_m": features.get("FR_front_distance_m"),
    "FL_clearance_m": features.get("FL_clearance_m"),
    "FL_front_distance_m": features.get("FL_front_distance_m"),
    "RR_clearance_m": features.get("RR_clearance_m"),
    "RR_front_distance_m": features.get("RR_front_distance_m"),
}, ensure_ascii=False))
