"""Bounded read-only live status; no repeated scan of large active audit logs."""
import json
from pathlib import Path
import sys


def last_record(path):
    if not path.exists():
        return None
    with path.open("rb") as stream:
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(max(0, size-4*1024*1024))
        lines = stream.read().splitlines()
    for line in reversed(lines):
        try:
            result = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            continue
        if isinstance(result, dict):
            return result
    return None


run = Path(sys.argv[1])
decision = last_record(run / "residual_and_projection_audit.jsonl") or {}
info = decision.get("applied_audit") or decision.get("info") or {}
update = last_record(run / "optimizer_updates.jsonl") or {}
episode = last_record(run / "completed_episodes.jsonl") or {}
manifest_path = run / "run_manifest.json"
manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
rr = info.get("semantic_task", {}).get("physical_evaluator", {}).get("current_legs", {}).get("RR", {})
print(json.dumps({"run": str(run), "lifecycle": manifest.get("lifecycle", "running_or_not_finalized"),
    "decision": decision.get("global_policy_decision"),
    "phase": info.get("phase_id"), "tick": info.get("physics_tick"),
    "placed_history": info.get("semantic_task", {}).get("placed_history"),
    "RR": {key: rr.get(key) for key in ("contact_mode", "air", "current_lift_valid",
        "unsupported_free_lift_m", "clearance_m", "front_distance_m", "wheel_bottom_vz_m_s")},
    "reward": decision.get("reward"), "reward_families": (info.get("reward") or {}).get("families"),
    "last_update": {key: update.get(key) for key in ("ppo_update", "global_policy_decisions",
        "optimizer_steps", "kl_mean", "value_loss", "optimizer_learning_rate")},
    "last_episode": {key: episode.get(key) for key in ("episode_index", "policy_decisions",
        "termination_reason", "task_success", "duration_s")}}, ensure_ascii=False))
