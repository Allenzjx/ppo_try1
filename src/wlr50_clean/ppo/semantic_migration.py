"""Explicit, immutable semantic checkpoint version boundaries, without Isaac."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA = "wlr50_clean.semantic_checkpoint_migration.v1"
STAGE_SPEC = "configs/ppo_semantic_v2/stage_task_spec.yaml"
INSTRUMENTATION_FILES = frozenset({
    "src/wlr50_clean/ppo/semantic_cli.py", "src/wlr50_clean/ppo/semantic_training.py",
    "src/wlr50_clean/ppo/semantic_migration.py", "src/wlr50_clean/ppo/semantic_legacy_evaluation.py",
    "src/wlr50_clean/ppo/semantic_metrics.py",
    "scripts/run_semantic_ppo.ps1",
})
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint_metadata(checkpoint: Path) -> dict[str, Any]:
    checkpoint = checkpoint.resolve(strict=True)
    sidecar = checkpoint.with_name(checkpoint.stem + "_manifest.json")
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    if (data.get("schema") != "wlr50_clean.semantic_checkpoint.v1"
            or data.get("checkpoint_path") != str(checkpoint)
            or data.get("checkpoint_sha256") != file_sha(checkpoint)
            or data.get("save_load_round_trip") is not True):
        raise ValueError("source semantic checkpoint/sidecar integrity mismatch")
    return data


def _contract(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    files = result.get("files")
    if (not isinstance(files, dict) or not files or any(not isinstance(v, str) or len(v) != 64 for v in files.values())
            or result.get("runtime_content_sha256") != digest(files)
            or len(str(result.get("source_git_commit", ""))) != 40):
        raise ValueError("migration runtime inventory digest/revision is malformed")
    return result


def _geometric_factor(project_root: Path, old: dict, new: dict) -> dict[str, Any]:
    raw = subprocess.run(["git", "-C", str(project_root), "show", f"{old['source_git_commit']}:{STAGE_SPEC}"],
                         check=True, capture_output=True).stdout
    # Git stores LF; the pinned Windows working tree may materialize CRLF.
    candidates = (raw, raw.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    if old["files"][STAGE_SPEC] not in {hashlib.sha256(data).hexdigest() for data in candidates}:
        raise ValueError("old geometric source bytes do not match the checkpoint contract")
    current = project_root / STAGE_SPEC
    if file_sha(current) != new["files"][STAGE_SPEC]:
        raise ValueError("new geometric source bytes do not match the current contract")
    before = raw.decode("utf-8").replace("\r\n", "\n")
    after = current.read_text(encoding="utf-8")
    source, target = "  approach_min_m: -0.18\n", "  approach_min_m: -0.005\n"
    if before.count(source) != 1 or before.replace(source, target) != after:
        raise ValueError("only the declared approach_min_m -0.18 to -0.005 geometric factor is permitted")
    return {"file": STAGE_SPEC, "field": "geometry.approach_min_m", "before": -0.18, "after": -0.005}


def build_migration_plan(checkpoint: Path, current_contract: Mapping[str, Any], *,
                         allowed_changed_files: Sequence[str], reason: str,
                         project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Build a reviewed plan after committing the new runtime; does not write."""
    checkpoint = Path(checkpoint).resolve(strict=True)
    metadata = checkpoint_metadata(checkpoint)
    old, new = _contract(metadata["runtime_contract"]), _contract(current_contract)
    old_fixed = {key: value for key, value in old.items() if key not in ("files", "runtime_content_sha256", "source_git_commit")}
    new_fixed = {key: value for key, value in new.items() if key not in ("files", "runtime_content_sha256", "source_git_commit")}
    if old_fixed != new_fixed:
        raise ValueError("migration cannot change frozen physics, runtime versions, rates or budgets")
    declared = list(allowed_changed_files)
    if not reason.strip() or len(set(declared)) != len(declared):
        raise ValueError("migration requires a reason and unique exact changed-file names")
    delta = sorted(path for path in set(old["files"]) | set(new["files"]) if old["files"].get(path) != new["files"].get(path))
    if sorted(declared) != delta or not set(delta) <= INSTRUMENTATION_FILES | {STAGE_SPEC}:
        raise ValueError("migration delta is undeclared or touches protected observation/action/reward/backend/physics files")
    if any(path not in new["files"] for path in delta):
        raise ValueError("migration cannot delete runtime files")
    geometric = _geometric_factor(Path(project_root), old, new) if STAGE_SPEC in delta else None
    sidecar = checkpoint.with_name(checkpoint.stem + "_manifest.json")
    return {"schema": SCHEMA, "reason": reason.strip(), "source_checkpoint": str(checkpoint),
            "source_checkpoint_sha256": file_sha(checkpoint), "source_manifest_sha256": file_sha(sidecar),
            "source_contract_sha256": digest(old), "target_contract_sha256": digest(new),
            "source_git_commit": old["source_git_commit"], "target_git_commit": new["source_git_commit"],
            "allowed_changed_files": delta, "changed_file_hashes": {path: {"before": old["files"].get(path), "after": new["files"][path]} for path in delta},
            "geometric_factor": geometric, "observation_dimension": 324, "action_dimension": 12,
            "preserve_actor_critic_optimizer_normalizer_rng_and_budget": True,
            "discard_old_rollout_storage": True, "physics_resume": "fresh_legal_P01_reset"}


def validate_migration_plan(checkpoint: Path, current_contract: Mapping[str, Any], plan_path: Path, *,
                            project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    path = Path(plan_path).resolve(strict=True)
    supplied = json.loads(path.read_text(encoding="utf-8"))
    expected = build_migration_plan(checkpoint, current_contract,
                                    allowed_changed_files=supplied.get("allowed_changed_files", []),
                                    reason=supplied.get("reason", ""), project_root=project_root)
    if supplied != expected:
        raise ValueError("migration plan is not exactly bound to this immutable checkpoint and runtime")
    return {"plan_path": str(path), "plan_sha256": file_sha(path), **expected}
