"""Explicit, immutable semantic checkpoint version boundaries, without Isaac."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA = "wlr50_clean.semantic_checkpoint_migration.v1"
STAGE_SPEC = "configs/ppo_semantic_v2/stage_task_spec.yaml"
SUPERVISOR = "src/wlr50_clean/ppo/semantic_supervisor.py"
PRIOR_DIAGNOSTIC_HEAD = "84c607a2ffbba36f46a0e70dbb886227c0c32ec5"
PRIOR_CONFIG_LINES = (
    "  approach_wheel_prior_rad_s: [0.3, 0.3, 0.3, 0.3]\n"
    "  approach_wheel_prior_source: P01\n"
    "  approach_wheel_prior_scope: P02_current_physical_goal_feedback_only\n"
)
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


def _version_text(project_root: Path, contract: dict, relative: str, *, prefer_worktree: bool = False) -> str:
    current = project_root / relative
    expected = contract["files"][relative]
    if prefer_worktree and current.is_file() and file_sha(current) == expected:
        return current.read_text(encoding="utf-8")
    raw = subprocess.run(["git", "-C", str(project_root), "show", f"{contract['source_git_commit']}:{relative}"],
                         check=True, capture_output=True).stdout
    # Git stores LF; the pinned Windows working tree may materialize CRLF.
    candidates = (raw, raw.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    if expected not in {hashlib.sha256(data).hexdigest() for data in candidates}:
        raise ValueError(f"versioned source bytes do not match the checkpoint contract: {relative}")
    return raw.decode("utf-8").replace("\r\n", "\n")


def _geometric_factor(project_root: Path, old: dict, new: dict, *, prior_transition: bool = False) -> dict[str, Any] | None:
    before = _version_text(project_root, old, STAGE_SPEC)
    after = _version_text(project_root, new, STAGE_SPEC, prefer_worktree=True)
    if prior_transition:
        nominal = after.split("\nnominal:\n", 1)
        if (len(nominal) != 2 or PRIOR_CONFIG_LINES in before or after.count(PRIOR_CONFIG_LINES) != 1
                or PRIOR_CONFIG_LINES not in nominal[1].split("\nstage_defaults:", 1)[0] + "\n"):
            raise ValueError("prior transition requires exactly the declared nominal P01 wheel-prior additions")
        after = after.replace(PRIOR_CONFIG_LINES, "")
    source, target = "  approach_min_m: -0.18\n", "  approach_min_m: -0.005\n"
    if prior_transition and before == after and target in before:
        return None
    if before.count(source) != 1 or before.replace(source, target) != after:
        raise ValueError("only the declared approach_min_m -0.18 to -0.005 geometric factor is permitted")
    return {"file": STAGE_SPEC, "field": "geometry.approach_min_m", "before": -0.18, "after": -0.005}


def _prior_factor(project_root: Path, old: dict, new: dict, evidence: Mapping[str, Any] | None,
                  source_checkpoint: Path) -> dict[str, Any]:
    if not isinstance(evidence, Mapping) or set(evidence) != {"B", "C"}:
        raise ValueError("nominal prior transition requires explicit B84c and C84c diagnostic evidence")
    before = _version_text(project_root, old, SUPERVISOR)
    after = _version_text(project_root, new, SUPERVISOR, prefer_worktree=True)
    def split(text):
        start, end = "class NominalMotionProvider:", "class SemanticControllerAdapter:"
        if text.count(start) != 1 or text.count(end) != 1:
            raise ValueError("nominal provider class boundaries are ambiguous")
        prefix, rest = text.split(start, 1)
        body, suffix = rest.split(end, 1)
        return prefix, body, suffix
    old_parts, new_parts = split(before), split(after)
    if old_parts[0] != new_parts[0] or old_parts[2] != new_parts[2] or old_parts[1] == new_parts[1]:
        raise ValueError("only the NominalMotionProvider class block may change; task evaluator and controller adapter remain frozen")
    rows = {}
    for role, mode in (("B", "semantic_prior_eval"), ("C", "semantic_residual_eval")):
        path = Path(evidence[role]).resolve(strict=True)
        if not path.is_relative_to((project_root / "runs/ppo_semantic_v2").resolve()):
            raise ValueError("prior diagnostic must belong to the isolated semantic run root")
        data = json.loads(path.read_text(encoding="utf-8"))
        lifecycle_path = path.with_name("run_manifest.json")
        lifecycle = json.loads(lifecycle_path.read_text(encoding="utf-8"))
        if (path.name != "evaluation_manifest.json" or data.get("mode") != mode
                or data.get("runtime_contract", {}).get("source_git_commit") != PRIOR_DIAGNOSTIC_HEAD
                or data.get("optimizer_updates_during_evaluation") != 0 or data.get("task_success") is not False
                or data.get("termination_reason") != "INCOMPLETE_CONTROLLER_BLOCKED"
                or lifecycle.get("lifecycle") != "SUCCEEDED" or lifecycle.get("result") != data):
            raise ValueError("prior diagnostic is not a finalized zero-update B/C84c physical failure")
        if ((role == "B" and data.get("checkpoint") is not None)
                or (role == "C" and (data.get("checkpoint_sha256") != file_sha(source_checkpoint)
                                     or Path(str(data.get("checkpoint"))).resolve() != source_checkpoint))):
            raise ValueError("prior diagnostic must compare zero B and the unchanged source checkpoint C")
        rows[role] = {"evaluation_manifest": str(path), "evaluation_manifest_sha256": file_sha(path),
                      "run_manifest_sha256": file_sha(lifecycle_path), "mode": mode,
                      "policy_decisions": data["policy_decisions"], "duration_s": data["duration_s"]}
    return {"diagnostic_git_commit": PRIOR_DIAGNOSTIC_HEAD, "source_file": SUPERVISOR,
            "allowed_source_scope": "NominalMotionProvider_class_only",
            "unchanged_prefix_sha256": hashlib.sha256(old_parts[0].encode()).hexdigest(),
            "unchanged_suffix_sha256": hashlib.sha256(old_parts[2].encode()).hexdigest(),
            "wheel_prior_rad_s": [0.3] * 4, "source_nominal_phase": "P01",
            "scope": "P02_current_physical_goal_feedback_only", "evidence": rows}


def build_migration_plan(checkpoint: Path, current_contract: Mapping[str, Any], *,
                         allowed_changed_files: Sequence[str], reason: str,
                         prior_evidence: Mapping[str, Any] | None = None,
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
    if sorted(declared) != delta or not set(delta) <= INSTRUMENTATION_FILES | {STAGE_SPEC, SUPERVISOR}:
        raise ValueError("migration delta is undeclared or touches protected observation/action/reward/backend/physics files")
    if any(path not in new["files"] for path in delta):
        raise ValueError("migration cannot delete runtime files")
    prior_transition = SUPERVISOR in delta
    if prior_transition != (prior_evidence is not None) or (prior_transition and STAGE_SPEC not in delta):
        raise ValueError("nominal source change and explicit prior evidence/config change must occur together")
    geometric = _geometric_factor(Path(project_root), old, new, prior_transition=prior_transition) if STAGE_SPEC in delta else None
    prior = _prior_factor(Path(project_root), old, new, prior_evidence, checkpoint) if prior_transition else None
    sidecar = checkpoint.with_name(checkpoint.stem + "_manifest.json")
    result = {"schema": SCHEMA, "reason": reason.strip(), "source_checkpoint": str(checkpoint),
            "source_checkpoint_sha256": file_sha(checkpoint), "source_manifest_sha256": file_sha(sidecar),
            "source_contract_sha256": digest(old), "target_contract_sha256": digest(new),
            "source_git_commit": old["source_git_commit"], "target_git_commit": new["source_git_commit"],
            "allowed_changed_files": delta, "changed_file_hashes": {path: {"before": old["files"].get(path), "after": new["files"][path]} for path in delta},
            "geometric_factor": geometric, "observation_dimension": 324, "action_dimension": 12,
            "preserve_actor_critic_optimizer_normalizer_rng_and_budget": True,
            "discard_old_rollout_storage": True, "physics_resume": "fresh_legal_P01_reset"}
    if prior is not None:
        result["prior_factor"] = prior
    return result


def validate_migration_plan(checkpoint: Path, current_contract: Mapping[str, Any], plan_path: Path, *,
                            project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    path = Path(plan_path).resolve(strict=True)
    supplied = json.loads(path.read_text(encoding="utf-8"))
    expected = build_migration_plan(checkpoint, current_contract,
                                    allowed_changed_files=supplied.get("allowed_changed_files", []),
                                    reason=supplied.get("reason", ""), project_root=project_root,
                                    prior_evidence=None if "prior_factor" not in supplied else {
                                        role: row["evaluation_manifest"] for role, row in supplied["prior_factor"]["evidence"].items()})
    if supplied != expected:
        raise ValueError("migration plan is not exactly bound to this immutable checkpoint and runtime")
    return {"plan_path": str(path), "plan_sha256": file_sha(path), **expected}
