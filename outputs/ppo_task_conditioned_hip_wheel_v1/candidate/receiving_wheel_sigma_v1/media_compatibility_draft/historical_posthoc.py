"""Run old quantity/AUX proof in its real detached historical runtime.

This helper is read-only and post-hoc.  It never creates or modifies a
worktree.  The caller must supply a clean detached checkout whose real HEAD is
the quantity plan's target commit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


HERE = Path(__file__).resolve().parent
PROFILE = HERE.parent
OUT = PROFILE.parents[1]
ROOT = OUT.parents[1]
AUX = OUT / "candidate/aux_method_label"
MARKER = "WLR50_HISTORICAL_POSTHOC_JSON="


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=check, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )


def validate_historical_root(root: Path, target_contract: Mapping[str, Any],
                             expected_commit: str) -> dict[str, Any]:
    root = Path(root).resolve(strict=True)
    require((root / ".git").exists(), "historical runtime root is not a git worktree")
    head = git(root, "rev-parse", "HEAD").stdout.strip()
    require(head == expected_commit == target_contract.get("source_git_commit"),
            "historical worktree HEAD is not the plan/runtime target commit")
    require(git(root, "symbolic-ref", "-q", "HEAD", check=False).returncode != 0,
            "historical runtime must be a detached checkout, not a moving branch")
    require(not git(root, "status", "--porcelain", "--untracked-files=no").stdout.strip(),
            "historical runtime has tracked modifications")
    files = target_contract.get("files")
    require(isinstance(files, Mapping) and files, "target runtime has no file inventory")
    total = 0
    for relative, expected in files.items():
        path = (root / relative).resolve(strict=True)
        require(path.is_relative_to(root) and sha(path) == expected,
                "historical runtime file differs: " + relative)
        total += path.stat().st_size
    return {"root": str(root), "detached_head": head, "tracked_status_clean": True,
            "runtime_file_count": len(files), "runtime_file_bytes": total,
            "runtime_contract_sha256": digest(target_contract)}


def _expected_evaluation(module, historical_root: Path, presentation_root: Path,
                         factor: Mapping[str, Any], source: Mapping[str, Any],
                         target: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    names = {"action_schema.json": "action_schema_path",
        "execution_profile.yaml": "execution_profile",
        "observation_schema.json": "observation_schema_path",
        "quality_score.yaml": "quality_score_path",
        "reward_config.yaml": "reward_config_path",
        "stage_task_spec.yaml": "task_spec_path"}
    result = []
    for side, contract in (("source", source), ("target", target)):
        rows = {}
        for name, key in names.items():
            binding = factor["configuration_bindings"][name][side]
            rows[key] = {"path": str((presentation_root / binding["path"]).resolve()),
                "sha256": binding["sha256"],
                "bytes": len(module._version_bytes(
                    historical_root, contract, binding["path"], prefer_worktree=True))}
        result.append(rows)
    return result[0], result[1]


def child(historical_root: Path, target_checkpoint: Path,
          auxiliary_receipt: Path, presentation_root: Path) -> dict[str, Any]:
    """Execute inside the checkpoint's pinned Python with historical imports."""
    historical_root = Path(historical_root).resolve(strict=True)
    target_checkpoint = Path(target_checkpoint).resolve(strict=True)
    presentation_root = Path(presentation_root).resolve(strict=True)
    sys.path.insert(0, str(AUX))
    sys.path.insert(0, str(historical_root / "src"))
    from wlr50_clean.ppo import semantic_migration as migration
    require(Path(migration.__file__).resolve().is_relative_to(historical_root / "src"),
            "historical production validator was shadowed")
    target = migration.checkpoint_metadata(target_checkpoint)
    extension = target.get("training_quantity_budget_extension")
    require(isinstance(extension, Mapping), "historical target has no quantity receipt")
    plan_path = Path(extension["plan_path"]).resolve(strict=True)
    require(sha(plan_path) == extension["plan_sha256"], "historical quantity plan changed")
    plan = read(plan_path)
    source_path = Path(plan["source_checkpoint"]).resolve(strict=True)
    source = migration.checkpoint_metadata(source_path)
    verified = migration.validate_migration_plan(
        source_path, target["runtime_contract"], plan_path,
        project_root=historical_root,
    )
    factor = verified.get("training_quantity_budget_factor")
    expected = {"factor": factor, "plan_path": verified["plan_path"],
        "plan_sha256": verified["plan_sha256"],
        "source_checkpoint_sha256": verified["source_checkpoint_sha256"],
        "source_contract_sha256": verified["source_contract_sha256"],
        "target_contract_sha256": verified["target_contract_sha256"]}
    require(extension == expected
            and all(value is None for key, value in verified.items()
                    if key.endswith("_factor") and key != "training_quantity_budget_factor")
            and factor.get("training_quantity_only") is True
            and factor.get("kernel_changed") is False
            and factor.get("same_mdp_claimed") is True,
            "archived validator does not reconstruct the exact saved quantity-only receipt")
    source_eval, target_eval = _expected_evaluation(
        migration, historical_root, presentation_root, factor,
        source["runtime_contract"], target["runtime_contract"],
    )
    import limited_aux_provenance
    require(Path(limited_aux_provenance.__file__).resolve() ==
            (AUX / "limited_aux_provenance.py").resolve(),
            "sealed LIMITED AUX validator was shadowed")
    auxiliary = limited_aux_provenance.validate_auxiliary_checkpoint(
        target_checkpoint, Path(auxiliary_receipt).resolve(strict=True))
    return {"schema": "wlr50_clean.historical_quantity_aux_posthoc.v1",
        "plan_target_git_commit": plan["target_git_commit"],
        "historical_validator": str(Path(migration.__file__).resolve()),
        "historical_validator_sha256": sha(Path(migration.__file__)),
        "quantity_plan": str(plan_path), "quantity_plan_sha256": verified["plan_sha256"],
        "quantity_factor": factor,
        "quantity_source_runtime_contract": source["runtime_contract"],
        "quantity_target_runtime_contract": target["runtime_contract"],
        "source_evaluation_configuration": source_eval,
        "target_evaluation_configuration": target_eval,
        "auxiliary_identity": auxiliary}


def validate(target_checkpoint: Path, auxiliary_receipt: Path, *,
             historical_runtime_root: Path) -> dict[str, Any]:
    """Launch the old proof in an isolated CPU-only subprocess."""
    target_checkpoint = Path(target_checkpoint).resolve(strict=True)
    target = read(target_checkpoint.with_name(target_checkpoint.stem + "_manifest.json"))
    require(target.get("schema") == "wlr50_clean.semantic_checkpoint.v1"
            and target.get("checkpoint_path") == str(target_checkpoint)
            and target.get("checkpoint_sha256") == sha(target_checkpoint)
            and target.get("save_load_round_trip") is True,
            "historical target checkpoint/sidecar integrity mismatch")
    extension = target.get("training_quantity_budget_extension") or {}
    plan_path = Path(extension.get("plan_path", "")).resolve(strict=True)
    plan = read(plan_path)
    historical = validate_historical_root(
        historical_runtime_root, target["runtime_contract"], plan["target_git_commit"],
    )
    python = Path(target["runtime_contract"]["local_runtime_versions"]["python_executable"]).resolve(strict=True)
    environment = dict(os.environ)
    environment.update(CUDA_VISIBLE_DEVICES="-1", PYTHONNOUSERSITE="1")
    command = [str(python), str(Path(__file__).resolve()), "--child",
        "--historical-runtime-root", str(Path(historical_runtime_root).resolve()),
        "--target-checkpoint", str(target_checkpoint),
        "--aux-receipt", str(Path(auxiliary_receipt).resolve(strict=True)),
        "--presentation-root", str(ROOT.resolve())]
    run = subprocess.run(command, env=environment, text=True, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, timeout=120)
    require(run.returncode == 0, "historical post-hoc subprocess failed: " + run.stderr[-2000:])
    lines = [line for line in run.stdout.splitlines() if line.startswith(MARKER)]
    require(len(lines) == 1, "historical post-hoc subprocess returned no unique receipt")
    result = json.loads(lines[0][len(MARKER):])
    require(result.get("plan_target_git_commit") == historical["detached_head"]
            and result.get("quantity_plan_sha256") == extension.get("plan_sha256")
            and result.get("auxiliary_identity", {}).get("accepted_auxiliary_updates_total") == 7
            and result.get("auxiliary_identity", {}).get("attempted_auxiliary_optimizer_steps_total") == 8,
            "historical child result does not bind the checkout/quantity/AUX receipt")
    return {**result, "historical_runtime": historical,
            "subprocess_python": str(python), "CUDA_VISIBLE_DEVICES": "-1",
            "posthoc_only_no_training": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--historical-runtime-root", type=Path, required=True)
    parser.add_argument("--target-checkpoint", type=Path, required=True)
    parser.add_argument("--aux-receipt", type=Path, required=True)
    parser.add_argument("--presentation-root", type=Path, default=ROOT)
    args = parser.parse_args()
    if args.child:
        print(MARKER + json.dumps(child(args.historical_runtime_root,
            args.target_checkpoint, args.aux_receipt, args.presentation_root),
            sort_keys=True, separators=(",", ":"), allow_nan=False))
    else:
        print(json.dumps(validate(args.target_checkpoint, args.aux_receipt,
            historical_runtime_root=args.historical_runtime_root), indent=2, allow_nan=False))
