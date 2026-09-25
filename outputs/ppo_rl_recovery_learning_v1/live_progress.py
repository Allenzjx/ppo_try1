"""Small read-only progress snapshot for one active semantic PPO run.

Each JSONL is opened independently and snapshotted with seek(END)/tell().  The
reader never uses stat length, never waits for the writer, and ignores a
trailing partial line.  Sampling evidence is not converted into optimizer
credit: update counters/checkpoint come only from optimizer_updates.jsonl.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable


FILES = {
    "prefix": "prefix_evidence.jsonl",
    "optimizer": "optimizer_updates.jsonl",
    "residual": "residual_and_projection_audit.jsonl",
}
COUNTERS = ("global_policy_decisions", "ppo_updates", "optimizer_steps")


def scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    return None


def get_path(value: Any, path: str) -> Any:
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def first_scalar(value: dict[str, Any], paths: Iterable[str]) -> dict[str, Any] | None:
    for path in paths:
        found = scalar(get_path(value, path))
        if found is not None:
            return {"path": path, "value": found}
    return None


def nested_schemas(value: Any, *, max_depth: int = 3) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []

    def visit(item: Any, path: str, depth: int) -> None:
        if len(result) >= 16 or depth > max_depth:
            return
        if isinstance(item, dict):
            schema = item.get("schema")
            if isinstance(schema, str):
                result.append({"path": path or "$", "schema": schema})
            for key, child in item.items():
                if isinstance(child, (dict, list)):
                    visit(child, f"{path}.{key}" if path else str(key), depth + 1)
        elif isinstance(item, list):
            for index, child in enumerate(item[:4]):
                if isinstance(child, (dict, list)):
                    visit(child, f"{path}[{index}]", depth + 1)

    visit(value, "", 0)
    return result


def last_complete_json(path: Path, *, max_row_bytes: int) -> dict[str, Any]:
    if not path.is_file():
        return {"state": "MISSING", "path": str(path)}
    chunk = 64 * 1024
    with path.open("rb") as stream:
        stream.seek(0, 2)
        snapshot_end = stream.tell()
        if snapshot_end == 0:
            return {"state": "EMPTY", "path": str(path),
                    "snapshot_end_offset": 0}
        start = snapshot_end
        data = b""
        while start > 0 and len(data) < max_row_bytes:
            take = min(chunk, start, max_row_bytes - len(data))
            start -= take
            stream.seek(start)
            data = stream.read(take) + data
            newline_positions = [index for index, byte in enumerate(data) if byte == 10]
            for index in range(len(newline_positions) - 1, -1, -1):
                end = newline_positions[index]
                begin = newline_positions[index - 1] + 1 if index else 0
                if begin == 0 and start > 0:
                    continue  # The first buffered segment may begin mid-row.
                raw = data[begin:end].strip()
                if not raw:
                    continue
                try:
                    row = json.loads(raw)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if not isinstance(row, dict):
                    continue
                return {"state": "OK", "path": str(path),
                        "snapshot_end_offset": snapshot_end,
                        "complete_row_end_offset": start + end + 1,
                        "trailing_uncommitted_bytes": snapshot_end - (start + end + 1),
                        "row": row}
        return {"state": "NO_COMPLETE_JSON_WITHIN_LIMIT", "path": str(path),
                "snapshot_end_offset": snapshot_end,
                "max_row_bytes": max_row_bytes}


def progress_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": first_scalar(row, (
            "schema", "applied_audit.schema", "policy_request.schema")),
        "entry_kind": first_scalar(row, ("kind", "task_result_scope")),
        "sim_time_s": first_scalar(row, (
            "sim_time_s", "episode_sim_time_s", "prefix_sim_time_s",
            "step_info.sim_time_s", "semantic_task.sim_time_s",
            "applied_audit.sim_time_s")),
        "phase": first_scalar(row, (
            "phase", "phase_id", "source_phase", "state_id", "stage_id",
            "step_info.phase_id", "step_info.end_phase_id",
            "step_info.semantic_task.stage_id", "applied_audit.phase_id",
            "applied_audit.end_phase_id", "applied_audit.semantic_task.stage_id")),
        "decision": first_scalar(row, (
            "decision", "decision_index", "decision_count",
            "global_policy_decision", "global_policy_decisions",
            "step_info.decision_count", "applied_audit.decision_count")),
        "global_policy_decision": first_scalar(
            row, ("global_policy_decision", "global_policy_decisions")),
        "physics_tick": first_scalar(row, (
            "episode_physics_tick", "physics_tick", "step_info.physics_tick",
            "applied_audit.physics_tick")),
        "top_level_evidence_keys": sorted(row)[:48],
        "nested_schemas": nested_schemas(row),
    }


def find_named_scalars(value: Any, names: set[str], path: str = "",
                       depth: int = 0) -> list[dict[str, Any]]:
    if depth > 4:
        return []
    result: list[dict[str, Any]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            if key in names:
                item_scalar = scalar(item)
                if item_scalar is not None:
                    result.append({"path": child_path, "value": item_scalar})
            if isinstance(item, (dict, list)):
                result.extend(find_named_scalars(item, names, child_path, depth + 1))
    elif isinstance(value, list):
        for index, item in enumerate(value[:8]):
            if isinstance(item, (dict, list)):
                result.extend(find_named_scalars(
                    item, names, f"{path}[{index}]", depth + 1))
    return result[:32]


def optimizer_summary(snapshot: dict[str, Any]) -> dict[str, Any]:
    if snapshot["state"] != "OK":
        return {"has_completed_update_row": False,
                "completed_update_counters": {key: 0 for key in COUNTERS},
                "checkpoint": None,
                "semantics": "no completed optimizer row; sampling is not an update"}
    row = snapshot["row"]
    candidates = find_named_scalars(row, set(COUNTERS) | {
        "checkpoint", "checkpoint_path", "checkpoint_sha256",
        "manifest", "manifest_sha256", "update_index", "completed_update"})
    counters: dict[str, Any] = {}
    for key in COUNTERS:
        aliases = ("ppo_update",) if key == "ppo_updates" else ()
        found = first_scalar(row, (
            key, *aliases, f"counters.{key}", f"lifetime_counters.{key}",
            f"checkpoint_counters.{key}", f"saved_checkpoint.{key}"))
        counters[key] = 0 if found is None else found
    checkpoint = first_scalar(row, (
        "checkpoint", "checkpoint_path", "saved_checkpoint.path",
        "checkpoint.path", "output_checkpoint"))
    return {"has_completed_update_row": True,
            "completed_update_counters": counters,
            "checkpoint": checkpoint,
            "relevant_scalar_paths": candidates,
            "semantics": "values are from the last complete optimizer row only"}


def compact_snapshot(kind: str, snapshot: dict[str, Any]) -> dict[str, Any]:
    result = {key: value for key, value in snapshot.items() if key != "row"}
    if snapshot["state"] == "OK":
        result["latest"] = progress_summary(snapshot["row"])
    result["kind"] = kind
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--max-row-bytes", type=int, default=16 * 1024 * 1024)
    args = parser.parse_args()
    if args.max_row_bytes < 64 * 1024:
        raise SystemExit("--max-row-bytes must be at least 65536")
    run = args.run.resolve(strict=True)
    if not run.is_dir():
        raise SystemExit("--run must be a directory")
    snapshots = {kind: last_complete_json(run / name,
                                          max_row_bytes=args.max_row_bytes)
                 for kind, name in FILES.items()}
    output = {
        "schema": "outputs.rl_recovery_live_progress.v1",
        "run": str(run),
        "snapshot_semantics": (
            "independent open+seek(END)+tell boundary per file; trailing partial JSON ignored"),
        "progress": {kind: compact_snapshot(kind, snapshot)
                     for kind, snapshot in snapshots.items()},
        "last_completed_optimizer_update": optimizer_summary(snapshots["optimizer"]),
        "claims": {"filesystem_stat_length_used": False,
                   "sampling_rows_counted_as_updates": False,
                   "writer_waited_on_or_locked": False},
    }
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
