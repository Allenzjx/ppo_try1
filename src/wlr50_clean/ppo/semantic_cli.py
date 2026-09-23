"""Isolated semantic B/C experiments, without the legacy perfect-probe gates."""
from __future__ import annotations

import argparse
from contextlib import closing, nullcontext
import hashlib
import importlib.metadata
import json
import math
import os
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .semantic_training import (
    STAGE_BUDGETS, training_quantity_budgets, SemanticRslAdapter, construct_semantic_runner, jsonable,
    load_semantic_checkpoint, save_semantic_checkpoint, seed_training_rngs,
    semantic_runner_config, semantic_curriculum_epoch, sha256_file, train_semantic, verified_native_effect, write_json,
)
from .semantic_migration import experiment_namespace, topology, stage_partition
from .semantic_policy_distribution import (
    LEGACY_POLICY, STATE_DEPENDENT_POLICY, HISTORY_POLICY, policy_contract,
    policy_version_from_metadata, build_policy_distribution_migration,
    policy_migration_checkpoint_name, policy_observation_layout_from_metadata,
)
from .semantic_receiving_wheel_profile import RECEIVING_WHEEL_POLICY

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUNS_ROOT = PROJECT_ROOT / "runs/ppo_semantic_v2"
OUTPUT_ROOT = PROJECT_ROOT / "outputs/ppo_semantic_v2"
LOCKED_DISTRIBUTIONS = {"torch": "2.7.0+cu128", "rsl-rl-lib": "5.0.1",
                        "isaacsim": "5.1.0.0", "isaaclab": "0.54.3", "tensordict": "0.12.2"}


def version_paths(version: str, *, experiment_id: str | None = None) -> tuple[Path, Path, Path]:
    namespace = experiment_namespace(version, experiment_id)
    if version == "v2":
        return RUNS_ROOT, OUTPUT_ROOT, PROJECT_ROOT / "configs/ppo_semantic_v2"
    return (PROJECT_ROOT / "runs" / namespace, PROJECT_ROOT / "outputs" / namespace,
            PROJECT_ROOT / "configs" / (namespace if experiment_id in ("all_stage_acceptance_v1", "fsm_reference_p09_stable_v2", "task_first_recovery_v1", "non_residual_refine_v1", "residual_rr_fix_v1", "fl_capture_quality_v1", "task_conditioned_hip_wheel_v1", "p05_hip_only_continuation_v1", "rr_capture_then_rl_transfer_v1", "rr_rl_timing_policy_learning_v1") else "ppo_semantic_v3"))


def _request_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    experiment_id = getattr(args, "experiment_id", None)
    return (version_paths(args.semantic_version) if experiment_id is None else
            version_paths(args.semantic_version, experiment_id=experiment_id))


def _checkpoint_output_root(args: argparse.Namespace) -> Path:
    """Artifact routing only; never change the namespace/config/runtime selection."""
    base = _request_paths(args)[1].resolve()
    name = getattr(args, "checkpoint_output_branch", None)
    if name is None:
        return base
    if (not isinstance(name, str) or re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", name) is None
            or name in {"con", "prn", "aux", "nul", *[f"com{i}" for i in range(1,10)], *[f"lpt{i}" for i in range(1,10)]}
            or args.semantic_version != "v3" or args.num_envs != 1
            or getattr(args, "experiment_id", None) not in ("rr_capture_then_rl_transfer_v1", "rr_rl_timing_policy_learning_v1")
            or args.command not in ("train", "eval") or args.checkpoint is None
            or (args.command == "eval" and args.mode != "semantic_residual_eval")
            or args.new_mdp_warm_start or getattr(args, "policy_distribution_migration", False)):
        raise ValueError("checkpoint output branch requires a safe single name and explicit RR v3 N1 resume/eval")
    result = base / "branches" / name
    for path in (base / "branches", result, result / "checkpoints", result / "checkpoints/history"):
        if path.resolve() != path or not path.resolve().is_relative_to(base):
            raise ValueError("checkpoint output branch cannot use a reparse/symlink alias or escape its namespace")
    return result


def _checkpoint_branch_source_root(args: argparse.Namespace, base: Path, destination: Path) -> Path:
    source = args.checkpoint.resolve(strict=True)
    if source.is_relative_to(destination / "checkpoints"):
        return destination
    if source.is_relative_to(base.resolve() / "checkpoints/history"):
        if args.command == "train" and (any((destination / "checkpoints/history").glob("*.pt"))
                or (destination / "checkpoints/checkpoint_last_pointer.json").exists()):
            raise ValueError("occupied checkpoint output branch requires explicit same-branch resume")
        return base
    raise ValueError("checkpoint branch source must be an immutable parent source or its own branch checkpoint")


def _bind_checkpoint_output_routing(args: argparse.Namespace, base: Path, destination: Path, source_root: Path) -> None:
    from .semantic_migration import digest
    metadata = json.loads(args.checkpoint.with_name(args.checkpoint.stem + "_manifest.json").read_text(encoding="utf-8"))
    if getattr(args,"experiment_id",None) == "rr_rl_timing_policy_learning_v1":
        from .semantic_rear_policy_timing_migration import build_rear_policy_output_routing, validate_rear_policy_namespace
        route = build_rear_policy_output_routing(metadata, metadata["runtime_contract"], destination)
        if source_root.resolve() == destination.resolve() and metadata.get("checkpoint_output_routing") != route:
            raise ValueError("same-branch rear419 checkpoint lacks its inherited output route")
        if source_root.resolve() != destination.resolve() and metadata.get("checkpoint_output_routing") is not None:
            raise ValueError("parent rear419 source cannot replay an existing branch")
        validate_rear_policy_namespace(metadata,metadata["runtime_contract"],destination,checkpoint_output_routing=route)
        args._checkpoint_output_routing=route
        return
    selection = (metadata.get("rr_progress_handoff_v5_migration") or {}).get("source_selection")
    if "rr_capture_reserve_v10_migration" in metadata:
        from .semantic_rr_capture_reserve_migration import validate_v10_branch_receipt
        route = {"schema":"wlr50_clean.checkpoint_output_routing.v1",
                 "branch":args.checkpoint_output_branch,"output_root":str(destination),
                 "main_latest_pointer_promotion":False,"source_selection":jsonable(selection)}
        if source_root.resolve() != destination or metadata.get("checkpoint_output_routing") != route:
            raise ValueError("v10 continuation requires its unchanged existing output branch")
        validate_v10_branch_receipt(metadata, metadata.get("runtime_contract") or {}, route)
        args._checkpoint_output_routing = route
        return
    if "rr_postcapture_wheel_v9_migration" in metadata:
        from .semantic_rr_postcapture_wheel_migration import validate_v9_branch_receipt
        route = {"schema":"wlr50_clean.checkpoint_output_routing.v1",
                 "branch":args.checkpoint_output_branch,"output_root":str(destination),
                 "main_latest_pointer_promotion":False,"source_selection":jsonable(selection)}
        if source_root.resolve() != destination or metadata.get("checkpoint_output_routing") != route:
            raise ValueError("v9 learned continuation requires its unchanged existing output branch")
        validate_v9_branch_receipt(metadata, metadata.get("runtime_contract") or {}, route)
        args._checkpoint_output_routing = route
        return
    wheel_signed = "rr_signed_wheel_v8_migration" in metadata
    signed = "rr_signed_contact_v7_migration" in metadata
    contact = metadata.get("rr_signed_wheel_v8_migration" if wheel_signed else "rr_signed_contact_v7_migration" if signed else "rr_contact_onset_v6_migration") or {}
    contact_schema, contact_factor, contact_feedback = (
        ("wlr50_clean.rr_signed_wheel_same410.v8", "rr_signed_wheel_v8_factor", "signed_band_contact_formation_incremental_v6") if wheel_signed else
        ("wlr50_clean.rr_signed_contact_same410.v7", "rr_signed_contact_v7_factor", "signed_band_contact_formation_incremental_v6") if signed else
        ("wlr50_clean.rr_contact_onset_same410.v6", "rr_contact_onset_v6_factor", "progress_reserve_contact_onset_incremental_v5"))
    runtime = metadata.get("runtime_contract") or {}
    origin = dict(global_policy_decisions=220544, ppo_updates=1688, optimizer_steps=33760)
    if (not isinstance(selection, dict) or selection.get("source_role") != "front_validated_ancestor_control_eval"
            or selection.get("counters") != origin or metadata.get("policy_contract", {}).get("observation_dimension") != 410):
        raise ValueError("checkpoint output branch only accepts the declared front-validated ancestor lineage")
    if (contact.get("schema") != contact_schema
            or contact.get("target_git_commit") != runtime.get("source_git_commit")
            or contact.get("target_contract_sha256") != digest(runtime)
            or contact.get("target_runtime_content_sha256") != runtime.get("runtime_content_sha256")
            or contact.get("source_selection", {}).get("source_role") != selection["source_role"]
            or contact.get("source_selection", {}).get("counters") != origin
            or contact.get(contact_factor, {}).get("target_feedback_revision") != contact_feedback
            or contact.get(contact_factor, {}).get("counter_origin") != origin):
        raise ValueError("checkpoint output branch requires its formally published v6/v7/v8 control receipt and current runtime")
    route = {"schema":"wlr50_clean.checkpoint_output_routing.v1",
             "branch":args.checkpoint_output_branch, "output_root":str(destination),
             "main_latest_pointer_promotion":False, "source_selection":jsonable(selection)}
    inherited = metadata.get("checkpoint_output_routing")
    if source_root.resolve() == destination:
        if inherited != route or any(type(metadata.get(k)) is not int or metadata[k] < v for k,v in origin.items()):
            raise ValueError("same-branch checkpoint routing/source counters are inconsistent")
    elif (inherited is not None or {k:metadata.get(k) for k in origin} != origin
          or metadata.get("rr_capture_transfer_branch_counts") != dict.fromkeys(origin, 0)):
        raise ValueError("initial ancestor branch cannot borrow updates or replay another output branch")
    args._checkpoint_output_routing = route


def local_versions() -> dict[str, Any]:
    versions = {name: importlib.metadata.version(name) for name in LOCKED_DISTRIBUTIONS}
    if versions != LOCKED_DISTRIBUTIONS or sys.version_info[:3] != (3, 11, 15):
        raise RuntimeError("semantic execution must use the unchanged locked local Python/RSL/Torch/Isaac stack")
    return {"python": ".".join(map(str, sys.version_info[:3])), "python_executable": str(Path(sys.executable).resolve()),
            "distributions": versions}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(allow_abbrev=False)
    result.add_argument("command", choices=("preflight", "smoke", "train", "eval"))
    result.add_argument("--run-dir", type=Path, required=True)
    result.add_argument("--expected-head", required=True)
    result.add_argument("--seed", type=int, default=1001)
    result.add_argument("--num-envs", type=int, choices=(1, 8), default=1)
    result.add_argument("--semantic-version", choices=("v2", "v3"), default="v2")
    result.add_argument("--experiment-id", choices=("transfer_roles_v1", "all_stage_acceptance_v1", "fsm_reference_p09_stable_v2", "task_first_recovery_v1", "non_residual_refine_v1", "residual_rr_fix_v1", "fl_capture_quality_v1", "task_conditioned_hip_wheel_v1", "p05_hip_only_continuation_v1", "rr_capture_then_rl_transfer_v1", "rr_rl_timing_policy_learning_v1"))
    result.add_argument("--from-phase", choices=("P01", "P03", "P04", "P05", "P06", "P07", "P08", "P09", "P10", "P11", "P12", "P13"), default="P01")
    result.add_argument("--teacher-offset-decisions", type=int, default=0)
    result.add_argument("--prefix-source", choices=("frozen_fsm", "checkpoint_policy", "successful_nominal"), default="frozen_fsm")
    result.add_argument("--new-mdp-warm-start", action="store_true")
    result.add_argument("--target-policy-version", choices=(HISTORY_POLICY,))
    result.add_argument("--policy-distribution-migration", action="store_true")
    result.add_argument("--vector-smoke-evidence", type=Path)
    result.add_argument("--stage", choices=tuple(STAGE_BUDGETS), default="smoke")
    result.add_argument("--decisions", type=int)
    result.add_argument("--max-decisions", type=int, default=3000)
    result.add_argument("--checkpoint", type=Path)
    result.add_argument("--checkpoint-output-branch")
    result.add_argument("--resume-migration", type=Path)
    result.add_argument("--mode", choices=("legacy_fsm_eval", "semantic_prior_eval", "semantic_residual_eval"), default="semantic_prior_eval")
    result.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    result.add_argument("--checkpoint-interval-updates", type=int, default=10)
    result.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    return result


def _frozen_media_revision(relative, expected, actual, experiment_id):
    """Explicit user-authorized media-only revision; never exempt controller/physics.

    Preserve the original frozen inventory and its original Git blob. This
    exception is bound to exact reviewed callback-wait bytes, not a filename
    wildcard or permission to mutate any other supposedly frozen component.
    """
    if actual == expected:
        return None
    if (experiment_id != "rr_rl_timing_policy_learning_v1"
            or relative != "src/wlr50_clean/infrastructure/video_capture.py"
            or expected != "4de41b906d506cc98165c66e44afa767a9f8111a1483a09492a25df43346dbd6"
            or actual != "6ec218b1a64344516aabeb5c32729314724ad6f175147f61086dfc38570c13ae"):
        raise ValueError(f"frozen A bytes changed: {relative}")
    preserved = "10aced6093c0efcca042bb9645e30140bffd04ff"
    blob = subprocess.run(["git", "-C", str(PROJECT_ROOT), "show", preserved+":"+relative],
                          check=True, capture_output=True).stdout
    if hashlib.sha256(blob).hexdigest() != expected:
        raise ValueError("original frozen media source is not preserved")
    return {"path": relative, "original_sha256": expected, "runtime_sha256": actual,
            "preserved_original_git_commit": preserved,
            "scope": "same_capture_callback_wait_only_no_physics_controller_or_actuator_change",
            "original_frozen_inventory_rewritten": False}


def runtime_contract(*, expected_head: str, semantic_version: str = "v2",
                     experiment_id: str | None = None) -> dict[str, Any]:
    experiment_namespace(semantic_version, experiment_id)
    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(PROJECT_ROOT), *args], check=True,
                              capture_output=True, text=True).stdout.strip()
    head = git("rev-parse", "HEAD")
    if head != expected_head or len(head) != 40:
        raise ValueError("semantic run HEAD differs from its pinned revision")
    paths = ("src/wlr50_clean", "scripts", "configs", "artifacts/ppo_phase_v1_start", "pyproject.toml")
    if git("status", "--porcelain=v1", "--untracked-files=all", "--", *paths):
        raise ValueError("semantic runtime source/config is not clean and committed")
    protected_path = PROJECT_ROOT / "artifacts/ppo_phase_v1_start/frozen_fsm_hashes.json"
    frozen = json.loads(protected_path.read_text(encoding="utf-8"))
    if frozen.get("algorithm") != "sha256" or not isinstance(frozen.get("protected_files"), dict):
        raise ValueError("frozen A inventory is malformed")
    media_revisions = []
    for relative, expected in frozen["protected_files"].items():
        revision = _frozen_media_revision(relative, expected, sha256_file(PROJECT_ROOT / relative), experiment_id)
        if revision is not None:
            media_revisions.append(revision)
    files = {relative: sha256_file(PROJECT_ROOT / relative)
             for relative in sorted(git("ls-files", "--", *paths).splitlines())}
    if not files:
        raise ValueError("semantic runtime inventory is empty")
    data = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    contract = {"schema": "wlr50_clean.semantic_runtime_contract.v1", "source_git_commit": head,
            "runtime_content_sha256": hashlib.sha256(data).hexdigest(), "files": files,
            "frozen_A_files": dict(frozen["protected_files"]), "rsl_rl_version": "5.0.1",
            "physics_hz": 120.0, "decision_hz": 15.0, "task_timeout_s": 200.0,
            "timeout_bootstrap": False, "training_budgets": training_quantity_budgets(experiment_id),
            "local_runtime_versions": local_versions()}
    if media_revisions:
        contract["frozen_A_media_revisions"] = media_revisions
    if semantic_version == "v3":
        config_root = version_paths(semantic_version, experiment_id=experiment_id)[2]
        contract.update(semantic_version="v3", selected_configuration={
            path.name: {"path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                        "sha256": sha256_file(path)} for path in sorted(config_root.iterdir()) if path.is_file()})
    if experiment_id is not None:
        contract["experiment_id"] = experiment_id
    if experiment_id in ("task_conditioned_hip_wheel_v1", "p05_hip_only_continuation_v1", "rr_capture_then_rl_transfer_v1", "rr_rl_timing_policy_learning_v1"):
        import yaml
        profile = yaml.safe_load((config_root / "execution_profile.yaml").read_text(encoding="utf-8"))
        declared = profile.get("training_budgets", {})
        if (declared != contract["training_budgets"]
                or any(type(value) is not int for value in declared.values())):
            raise ValueError("execution profile quantity budget differs from its explicit experiment ceiling")
    return contract


def _resolved_checkpoint(path: Path, *, output_root: Path | None = None) -> Path:
    output_root = OUTPUT_ROOT if output_root is None else output_root
    source = path.resolve(strict=True)
    if not source.is_relative_to((output_root / "checkpoints").resolve()):
        raise ValueError("semantic checkpoint must be inside the new isolated checkpoint root")
    if source.name == "checkpoint_last.pt":
        pointer = json.loads(source.with_name("checkpoint_last_pointer.json").read_text(encoding="utf-8"))
        immutable = Path(pointer["checkpoint"]).resolve(strict=True)
        if (not immutable.is_relative_to((output_root / "checkpoints/history").resolve())
                or sha256_file(source) != pointer["checkpoint_sha256"]
                or sha256_file(immutable) != pointer["checkpoint_sha256"]
                or sha256_file(Path(pointer["manifest"])) != pointer["manifest_sha256"]):
            raise ValueError("semantic last checkpoint pointer/copy is inconsistent")
        source = immutable
    return source


def validate_request(args: argparse.Namespace) -> None:
    _validate_target_policy_request(args)
    runs_root, output_root, _ = _request_paths(args)
    checkpoint_output = _checkpoint_output_root(args)
    branch_requested = getattr(args, "checkpoint_output_branch", None) is not None
    budgets = training_quantity_budgets(getattr(args, "experiment_id", None))
    if getattr(args, "experiment_id", None) in ("residual_rr_fix_v1", "fl_capture_quality_v1", "task_conditioned_hip_wheel_v1", "p05_hip_only_continuation_v1", "rr_capture_then_rl_transfer_v1", "rr_rl_timing_policy_learning_v1") and (
            args.new_mdp_warm_start or getattr(args, "policy_distribution_migration", False)):
        raise ValueError("RR continuation uses its explicit state-preserving task migration or exact resume")
    if getattr(args, "experiment_id", None) == "non_residual_refine_v1":
        if (args.command not in ("preflight", "eval") or args.mode != "semantic_prior_eval"
                or args.checkpoint is not None or args.resume_migration is not None
                or args.new_mdp_warm_start or getattr(args, "policy_distribution_migration", False)
                or args.num_envs != 1 or args.from_phase != "P01" or args.teacher_offset_decisions
                or getattr(args, "prefix_source", "frozen_fsm") != "frozen_fsm"):
            raise ValueError("non-residual refinement is isolated natural-P01 prior-only, without a checkpoint or PPO training")
    directory = args.run_dir.resolve()
    if not directory.is_relative_to(runs_root.resolve()) or directory == runs_root.resolve():
        raise ValueError(f"semantic run directory must be strictly inside {runs_root}")
    if args.semantic_version == "v3" and args.num_envs != 1:
        raise ValueError("v3 continuation currently implements real N1 reset-only suffix sampling")
    if (args.from_phase != "P01" or args.teacher_offset_decisions) and (
            args.semantic_version != "v3" or args.command != "train" or args.from_phase == "P01"):
        raise ValueError("suffix starts and teacher offsets require v3 N1 training; evaluation remains fresh P01")
    if not 0 <= args.teacher_offset_decisions < 1800:
        raise ValueError("teacher offset must be within the 200 second total task budget")
    if getattr(args, "prefix_source", "frozen_fsm") in ("checkpoint_policy", "successful_nominal") and (
            args.semantic_version != "v3" or args.num_envs != 1 or args.command != "train"
            or args.from_phase == "P01" or args.stage != "phase_suffix" or args.checkpoint is None
            or getattr(args, "policy_distribution_migration", False)):
        raise ValueError("checkpoint-policy prefix requires v3 N1 suffix training with an existing unchanged-architecture checkpoint; evaluation stays natural P01")
    if getattr(args, "policy_distribution_migration", False) and (
            args.semantic_version != "v3" or args.num_envs != 1 or args.command != "train"
            or args.checkpoint is None or args.new_mdp_warm_start or args.resume_migration is not None):
        raise ValueError("policy distribution migration requires v3 N1 train with a checkpoint, exclusive of new-MDP or resume migration")
    if args.command == "train" and args.semantic_version == "v3":
        if args.checkpoint is None:
            raise ValueError("v3 continuation requires existing learned checkpoint weights")
        if args.stage == "phase_suffix" and args.from_phase == "P01":
            raise ValueError("v3 phase_suffix must request an actual rear-leg precursor or suffix phase")
        if args.stage == "full_episode" and args.from_phase != "P01":
            raise ValueError("full_episode training must start from fresh P01")
    if args.new_mdp_warm_start and (args.semantic_version != "v3" or args.command != "train"
                                  or args.checkpoint is None or args.resume_migration is not None):
        raise ValueError("new-MDP warm start requires v3 train with a v2 or v3 checkpoint, not exact resume migration")
    if args.decisions is not None and (args.command != "train" or not 1 <= args.decisions <= budgets[args.stage]):
        raise ValueError("--decisions is an additional train request within the stage budget")
    if not 1 <= args.max_decisions <= 3000 or args.checkpoint_interval_updates < 1:
        raise ValueError("semantic window must remain within the 200 second task horizon")
    if args.command == "smoke" and args.max_decisions < 72:
        raise ValueError("functional smoke requires at least 64 decisions then reset and 8 decisions")
    if args.command == "train" and args.seed not in range(1001, 1009):
        raise ValueError("training seed must belong to the declared training split")
    if args.command == "eval" and args.seed not in (*range(2001, 2006), *range(3001, 3006), 4001):
        raise ValueError("evaluation seed must belong to validation, locked-test or video split")
    if args.command == "eval" and args.mode == "semantic_residual_eval" and args.checkpoint is None:
        raise ValueError("residual evaluation requires a saved checkpoint")
    if args.command == "eval" and args.mode in ("legacy_fsm_eval", "semantic_prior_eval") and args.checkpoint is not None:
        raise ValueError("legacy A and prior B evaluation must not load a PPO checkpoint")
    if args.resume_migration is not None and (args.command not in ("train", "eval") or args.checkpoint is None):
        raise ValueError("explicit migration requires a train or residual-eval checkpoint")
    if args.command in ("preflight", "smoke") and args.checkpoint is not None:
        raise ValueError("functional preflight/smoke must not silently load a policy")
    if args.command == "train" and args.checkpoint is None and any(path.exists() for path in (
            output_root / "checkpoints/checkpoint_last_pointer.json",
            output_root / "checkpoints/history/checkpoint_initial_semantic.pt")):
        raise ValueError("training already exists; explicitly resume checkpoint_last instead of reinitializing")
    if args.checkpoint is not None:
        source_root = (_checkpoint_branch_source_root(args, output_root, checkpoint_output)
                       if branch_requested else output_root)
        if (getattr(args,"experiment_id",None) == "rr_rl_timing_policy_learning_v1"
                and args.resume_migration is not None):
            planned=json.loads(args.resume_migration.read_text(encoding="utf-8"))
            initial_append = isinstance(planned.get("rear_policy_timing_factor"),dict)
            same419 = isinstance(planned.get("rear_recapture_same419_factor"),dict)
            if initial_append == same419:
                raise ValueError("rear timing requires exactly one declared migration boundary")
            if initial_append:
                source_root=version_paths("v3",experiment_id="rr_capture_then_rl_transfer_v1")[1]
            else:
                if planned.get("schema") != "wlr50_clean.rear_recapture_same419.v1":
                    raise ValueError("unsupported same419 migration schema")
                source_root=output_root
        if (getattr(args,"experiment_id",None) == "rr_capture_then_rl_transfer_v1" and args.resume_migration is not None
                and not branch_requested
                and not args.checkpoint.resolve(strict=True).is_relative_to((output_root/"checkpoints").resolve())):
            planned=json.loads(args.resume_migration.read_text(encoding="utf-8"))
            if not isinstance(planned.get("rr_capture_transfer_factor"),dict):
                raise ValueError("RR continuation requires its dedicated state-preserving append migration")
            source_root=version_paths("v3",experiment_id="p05_hip_only_continuation_v1")[1]
        if (getattr(args, "experiment_id", None) == "p05_hip_only_continuation_v1"
                and args.resume_migration is not None
                and not args.checkpoint.resolve(strict=True).is_relative_to((output_root / "checkpoints").resolve())):
            planned = json.loads(args.resume_migration.read_text(encoding="utf-8"))
            if not isinstance(planned.get("p05_capture_assist_factor"), dict):
                raise ValueError("P05 capture continuation requires its explicit state-preserving append migration")
            source_root = version_paths("v3", experiment_id="task_conditioned_hip_wheel_v1")[1]
        if (getattr(args, "experiment_id", None) == "task_conditioned_hip_wheel_v1"
                and args.resume_migration is not None
                and not args.checkpoint.resolve(strict=True).is_relative_to((output_root / "checkpoints").resolve())):
            planned = json.loads(args.resume_migration.read_text(encoding="utf-8"))
            if not isinstance(planned.get("task_conditioned_hip_wheel_factor"), dict):
                raise ValueError("task-conditioned cross-root continuation requires its joint reviewed factor")
            source_root = version_paths("v3", experiment_id="fl_capture_quality_v1")[1]
        if (getattr(args, "experiment_id", None) == "fl_capture_quality_v1"
                and args.resume_migration is not None
                and not args.checkpoint.resolve(strict=True).is_relative_to((output_root / "checkpoints").resolve())):
            planned = json.loads(args.resume_migration.read_text(encoding="utf-8"))
            if not isinstance(planned.get("fl_capture_quality_same372_factor"), dict):
                raise ValueError("FL capture cross-root continuation requires its independent reviewed factor")
            source_root = version_paths("v3", experiment_id="residual_rr_fix_v1")[1]
        if (getattr(args, "experiment_id", None) == "residual_rr_fix_v1"
                and args.resume_migration is not None
                and not args.checkpoint.resolve(strict=True).is_relative_to((output_root / "checkpoints").resolve())):
            # Path routing only. The new factor is fully revalidated before native launch.
            planned = json.loads(args.resume_migration.read_text(encoding="utf-8"))
            if not isinstance(planned.get("rr_physical_acceptance_same372_factor"), dict):
                raise ValueError("RR cross-root continuation requires its independent reviewed factor")
            source_root = version_paths("v3", experiment_id="task_first_recovery_v1")[1]
        if (getattr(args, "experiment_id", None) == "task_first_recovery_v1"
                and args.resume_migration is not None
                and not args.checkpoint.resolve(strict=True).is_relative_to((output_root / "checkpoints").resolve())):
            # Only route the explicit old-policy reward migration. Its strict
            # same-N/config validation still runs before any native launch.
            source_root = version_paths("v3", experiment_id="fsm_reference_p09_stable_v2")[1]
        if args.new_mdp_warm_start:
            if getattr(args, "experiment_id", None) == "fsm_reference_p09_stable_v2":
                # A same-experiment nominal boundary remains separately reviewed
                # by build_v3_warm_start_record; this is path routing, not a waiver.
                if not args.checkpoint.resolve(strict=True).is_relative_to((output_root / "checkpoints").resolve()):
                    source_root = version_paths("v3", experiment_id="all_stage_acceptance_v1")[1]
            elif getattr(args, "experiment_id", None) == "all_stage_acceptance_v1":
                source_root = version_paths("v3", experiment_id="transfer_roles_v1")[1]
            elif getattr(args, "experiment_id", None) is not None:
                prior_v3_root = version_paths("v3")[1]
                if args.checkpoint.resolve(strict=True).is_relative_to((prior_v3_root / "checkpoints").resolve()):
                    source_root = prior_v3_root
            elif args.checkpoint.resolve(strict=True).is_relative_to((OUTPUT_ROOT / "checkpoints").resolve()):
                source_root = OUTPUT_ROOT
        args.checkpoint = _resolved_checkpoint(args.checkpoint, output_root=source_root)
        if branch_requested:
            _bind_checkpoint_output_routing(args, output_root, checkpoint_output, source_root)
        if args.command == "train":
            metadata = json.loads(args.checkpoint.with_name(args.checkpoint.stem + "_manifest.json").read_text())
            rear_timing = getattr(args,"experiment_id",None) == "rr_rl_timing_policy_learning_v1"
            if rear_timing and args.resume_migration is None:
                from .semantic_rear_policy_timing_migration import validate_rear_policy_namespace
                validate_rear_policy_namespace(metadata, metadata["runtime_contract"], checkpoint_output,
                    checkpoint_output_routing=getattr(args,"_checkpoint_output_routing",None))
                if "rear_recapture_migration" in metadata and not branch_requested:
                    raise ValueError("published rear419 recapture ancestor training requires its explicit output branch")
            if not rear_timing and not branch_requested and any(k in metadata for k in
                    ("rr_capture_reserve_v10_migration","rr_postcapture_wheel_v9_migration")):
                raise ValueError("v9/v10 learned continuation requires its explicit output branch")
            if (not rear_timing and not branch_requested and any((metadata.get(key) or {}).get("source_selection", {}).get(
                    "source_role") == "front_validated_ancestor_control_eval" for key in (
                        "rr_contact_onset_v6_migration", "rr_signed_contact_v7_migration", "rr_signed_wheel_v8_migration"))):
                raise ValueError("published ancestor training requires its explicit output branch")
            source_version = metadata.get("semantic_version", "v2")
            if args.new_mdp_warm_start:
                expected_version = "v2" if source_root == OUTPUT_ROOT else "v3"
                if source_version != expected_version:
                    raise ValueError("warm-start checkpoint version differs from its isolated source root")
                if source_version == "v2" and any((output_root / relative).exists() for relative in (
                        "checkpoints/checkpoint_last_pointer.json", "checkpoints/history/checkpoint_initial_v3_warm_start.pt")):
                    raise ValueError("v3 training already exists; continue its checkpoint without restarting v3 budgets")
            reset_v3_budget = args.new_mdp_warm_start and source_version == "v2"
            remaining = budgets[args.stage] - (0 if reset_v3_budget else int(metadata["stage_requested_decisions"].get(args.stage, 0)))
            if remaining < 1 or (args.decisions is not None and args.decisions > remaining):
                raise ValueError("additional request exceeds the remaining semantic stage budget")
    if args.num_envs == 8:
        if args.command not in ("preflight", "smoke", "train") or args.device != "cuda:0" or args.seed != 1001:
            raise ValueError("initial N8 entry supports GPU preflight/smoke/train and seed1001 only")
        if args.command == "smoke" and args.max_decisions != 128:
            raise ValueError("N8 smoke requires two explicit 64-decision-per-row probes")
        if args.command == "train":
            if args.checkpoint is None or args.vector_smoke_evidence is None:
                raise ValueError("N8 must load actual learned weights and current live N8 interface proof")
            if args.decisions is not None and args.decisions % 1024:
                raise ValueError("N8 training requests must be whole128x8 rollouts; use N1 for the tail")
            from .semantic_migration import checkpoint_metadata
            from .semantic_migration import stage_partition
            metadata = checkpoint_metadata(args.checkpoint)
            left = budgets[args.stage]-int(metadata["stage_requested_decisions"][args.stage])
            if args.decisions is None and stage_partition(left)["N8_requested"] < 1024:
                raise ValueError("remaining stage budget requires the explicit N1 tail")
    elif args.vector_smoke_evidence is not None:
        raise ValueError("standalone N8 proof is only consumed by N8 training")
    args.run_dir = directory


def _validate_target_policy_request(args: argparse.Namespace) -> None:
    target = getattr(args, "target_policy_version", None)
    if target is not None and (
            target != HISTORY_POLICY or args.semantic_version != "v3" or args.num_envs != 1
            or args.command != "train" or args.checkpoint is None or not args.new_mdp_warm_start
            or args.resume_migration is not None or getattr(args, "policy_distribution_migration", False)
            or getattr(args, "prefix_source", "frozen_fsm") == "checkpoint_policy"):
        raise ValueError("target policy kernel requires exclusive v3 N1 new-MDP training; no checkpoint-policy prefix at this boundary")


def _preflight_checkpoint(args: argparse.Namespace, contract: dict[str, Any]) -> None:
    """Reject stale or unauthorized weights before loading any native library."""
    _validate_target_policy_request(args)
    args._migration_record = None
    args._warm_start_record = None
    args._policy_migration_record = None
    args._policy_version = LEGACY_POLICY
    args._observation_layout = None
    if args.checkpoint is None:
        if getattr(args, "policy_distribution_migration", False):
            raise ValueError("policy distribution migration requires a saved checkpoint")
        return
    from .semantic_observation import load_semantic_observation_schema
    target_schema = load_semantic_observation_schema(_request_paths(args)[2] / "observation_schema.json")
    args._observation_layout = getattr(target_schema, "observation_layout",
                                       getattr(target_schema, "transfer_role_features_version", None))
    from .semantic_migration import checkpoint_metadata, validate_migration_plan
    metadata = checkpoint_metadata(args.checkpoint)
    if (not args.new_mdp_warm_start and args.resume_migration is None
            and not getattr(args, "policy_distribution_migration", False)
            and metadata["runtime_contract"] != contract):
        raise ValueError("checkpoint runtime changed; an explicit reviewed resume migration is required")
    args._policy_version = policy_version_from_metadata(metadata)
    source_layout = policy_observation_layout_from_metadata(metadata)
    if getattr(args,"experiment_id",None) == "rr_rl_timing_policy_learning_v1" and args.resume_migration is not None:
        from .semantic_rear_policy_timing_profile import REAR_POLICY_TIMING_POLICY, REAR_POLICY_TIMING_OBSERVATION_LAYOUT
        args._migration_record=validate_migration_plan(args.checkpoint,contract,args.resume_migration)
        if args._observation_layout != REAR_POLICY_TIMING_OBSERVATION_LAYOUT or metadata["seed"] != args.seed:
            raise ValueError("rear timing requires exact419 layout and original RNG seed")
        args._policy_version=REAR_POLICY_TIMING_POLICY
        return
    if getattr(args,"experiment_id",None) == "rr_capture_then_rl_transfer_v1" and args.resume_migration is not None:
        proposed=json.loads(args.resume_migration.read_text(encoding="utf-8"))
        if proposed.get("rr_capture_transfer_factor") is not None:
            from .semantic_rr_capture_profile import RR_CAPTURE_POLICY, RR_CAPTURE_OBSERVATION_LAYOUT
            args._migration_record=validate_migration_plan(args.checkpoint,contract,args.resume_migration)
            if args._observation_layout != RR_CAPTURE_OBSERVATION_LAYOUT or metadata["seed"] != args.seed:
                raise ValueError("RR append requires exact target layout and original training RNG seed")
            args._policy_version=RR_CAPTURE_POLICY
            return
    if (getattr(args, "experiment_id", None) == "p05_hip_only_continuation_v1"
            and args.resume_migration is not None):
        proposed = json.loads(args.resume_migration.read_text(encoding="utf-8"))
        if proposed.get("p05_capture_assist_factor") is not None:
            from .semantic_p05_capture_profile import P05_CAPTURE_POLICY, P05_CAPTURE_OBSERVATION_LAYOUT
            args._migration_record = validate_migration_plan(args.checkpoint, contract, args.resume_migration)
            if args._observation_layout != P05_CAPTURE_OBSERVATION_LAYOUT:
                raise ValueError("P05 capture migration target observation schema mismatch")
            if args.command == "train" and metadata["seed"] != args.seed:
                raise ValueError("P05 append migration must preserve the training RNG seed")
            args._policy_version = P05_CAPTURE_POLICY
            return
    if not args.new_mdp_warm_start and source_layout != args._observation_layout:
        raise ValueError("checkpoint observation layout differs; explicit new-MDP append migration is required")
    if getattr(args, "policy_distribution_migration", False):
        # Repeat the scope check here: this boundary must remain safe even when
        # called directly by another entry point before any native launch.
        if (getattr(args, "semantic_version", "v2") != "v3" or getattr(args, "num_envs", 1) != 1
                or args.command != "train" or args.new_mdp_warm_start or args.resume_migration is not None):
            raise ValueError("policy distribution migration requires exclusive v3 N1 training")
        if metadata["seed"] != args.seed:
            raise ValueError("policy migration must preserve the recorded training RNG seed")
        args._policy_migration_record = build_policy_distribution_migration(
            args.checkpoint, contract, project_root=PROJECT_ROOT)
        initial = _request_paths(args)[1] / "checkpoints/history" / policy_migration_checkpoint_name(args._policy_migration_record)
        if initial.exists() or initial.with_name(initial.stem + "_manifest.json").exists():
            raise ValueError("this policy migration initial checkpoint already exists; resume its published weights")
        args._policy_version = STATE_DEPENDENT_POLICY
        return
    if args.new_mdp_warm_start:
        from .semantic_migration import build_v3_warm_start_record, v3_warm_start_checkpoint_name
        options = {}
        if getattr(args, "target_policy_version", None) is not None:
            options["target_policy_version"] = args.target_policy_version
        args._warm_start_record = build_v3_warm_start_record(args.checkpoint, contract, project_root=PROJECT_ROOT, **options)
        append = args._warm_start_record.get("observation_append_transition")
        if append is not None and append["target_observation_layout"] != args._observation_layout:
            raise ValueError("append migration target layout differs from the current observation schema")
        same_layout = args._warm_start_record.get("observation_same_layout_transition")
        if same_layout is not None and (
                same_layout["source_observation_layout"] != source_layout
                or same_layout["target_observation_layout"] != args._observation_layout):
            raise ValueError("same372 authority migration layout differs from verified source/target schemas")
        if options:
            args._policy_version = args._warm_start_record["policy_kernel_transition"]["target_policy_version"]
        if metadata["seed"] != args.seed:
            raise ValueError("warm start must preserve the recorded training RNG seed")
        initial = _request_paths(args)[1] / "checkpoints/history" / v3_warm_start_checkpoint_name(args._warm_start_record)
        if initial.exists() or initial.with_name(initial.stem + "_manifest.json").exists():
            raise ValueError("this source/target new-MDP initial checkpoint already exists; resume its published weights")
        return
    if args.resume_migration is None:
        if metadata["runtime_contract"] != contract:
            raise ValueError("checkpoint runtime changed; an explicit reviewed resume migration is required")
    else:
        args._migration_record = validate_migration_plan(args.checkpoint, contract, args.resume_migration)
        if args._migration_record.get("training_quantity_budget_factor") is not None and (
                args.command != "train" or args.semantic_version != "v3" or args.num_envs != 1
                or args.stage != "full_episode" or args.from_phase != "P01" or args.teacher_offset_decisions != 0
                or getattr(args, "prefix_source", "frozen_fsm") != "frozen_fsm"):
            raise ValueError("quantity-only migration first enters through fresh natural-P01 N1 full training")
        from .semantic_training import (_validated_exploration_temperature_factor,
            _validated_request_history_kernel_factor, _validated_physical_innovation_sigma_factor,
            _validated_task_conditioned_hip_wheel_factor, _validated_receiving_wheel_sigma_factor)
        receiving_wheel = _validated_receiving_wheel_sigma_factor(metadata, args._migration_record,
            semantic_version=args.semantic_version, seed=int(metadata["seed"]), device=args.device,
            observation_layout=args._observation_layout)
        task_conditioned = _validated_task_conditioned_hip_wheel_factor(metadata, args._migration_record,
            semantic_version=args.semantic_version, seed=int(metadata["seed"]), device=args.device,
            observation_layout=args._observation_layout)
        physical_innovation = _validated_physical_innovation_sigma_factor(metadata, args._migration_record,
            semantic_version=args.semantic_version, seed=int(metadata["seed"]), device=args.device,
            observation_layout=args._observation_layout)
        request_history = _validated_request_history_kernel_factor(metadata, args._migration_record,
            semantic_version=args.semantic_version, seed=int(metadata["seed"]), device=args.device,
            observation_layout=args._observation_layout)
        temperature = _validated_exploration_temperature_factor(metadata, args._migration_record,
            semantic_version=args.semantic_version, seed=int(metadata["seed"]), device=args.device,
            observation_layout=args._observation_layout)
        if receiving_wheel is not None:
            if (args.command != "train" or args.semantic_version != "v3" or args.num_envs != 1
                    or args.stage != "full_episode" or args.from_phase != "P01" or args.teacher_offset_decisions != 0
                    or getattr(args, "prefix_source", "frozen_fsm") != "frozen_fsm"):
                raise ValueError("receiving-wheel sigma first enters via fresh natural-P01 N1 full training")
            args._policy_version = receiving_wheel["target_policy_contract"]["version"]
        elif task_conditioned is not None:
            if getattr(args, "num_envs", 1) != 1:
                raise ValueError("task-conditioned reward/sigma migration requires N1")
            args._policy_version = task_conditioned["target_policy_contract"]["version"]
        elif physical_innovation is not None:
            if getattr(args, "num_envs", 1) != 1:
                raise ValueError("physical-innovation sigma migration requires N1")
            args._policy_version = physical_innovation["target_policy_contract"]["version"]
        elif request_history is not None:
            if getattr(args, "num_envs", 1) != 1:
                raise ValueError("request-history kernel migration requires N1")
            args._policy_version = request_history["target_policy_contract"]["version"]
        elif temperature is not None:
            if getattr(args, "num_envs", 1) != 1:
                raise ValueError("exploration temperature migration requires N1")
            args._policy_version = temperature["target_policy_contract"]["version"]
        elif metadata.get("runner_config") != semantic_runner_config(seed=int(metadata["seed"]), device=args.device,
                semantic_version=metadata.get("semantic_version", "v2"), policy_version=args._policy_version,
                observation_layout=args._observation_layout):
            raise ValueError("migration cannot change PPO hyperparameters or normalization")
    if args.command == "train" and metadata["seed"] != args.seed:
        raise ValueError("resume must preserve the checkpoint training RNG seed")


def _task_conditioned_prefix_provenance(args, contract, previous):
    """Bind migrated source weights to the actual target task/kernel, including exact archive-only changes."""
    from .semantic_policy_distribution import (TASK_CONDITIONED_HIP_WHEEL_POLICY,
        FR_KNEE_PHYSICAL_INNOVATION_POLICY)
    target = _resolved_policy_contract(args)
    source = previous.get("policy_contract")
    record = getattr(args, "_migration_record", None) or {}
    archive = record.get("archive_only_exact_bytes_factor")
    task = record.get("task_conditioned_hip_wheel_factor")
    receiving = record.get("receiving_wheel_sigma_factor")
    source_runtime = previous["runtime_contract"]["runtime_content_sha256"]
    target_runtime = contract["runtime_content_sha256"]
    factor, key = (archive, "archive_only_exact_bytes_migration") if archive is not None else (task, "task_conditioned_hip_wheel_migration")
    if receiving is not None:
        factor, key = receiving, "receiving_wheel_sigma_migration"
    if factor is not None:
        if (sum(x is not None for x in (task, archive, receiving)) != 1
                or factor.get("source_policy_contract") != source or factor.get("target_policy_contract") != target
                or record.get("source_runtime_content_sha256") != source_runtime
                or record.get("target_runtime_content_sha256") != target_runtime
                or target["version"] not in (FR_KNEE_PHYSICAL_INNOVATION_POLICY, TASK_CONDITIONED_HIP_WHEEL_POLICY, RECEIVING_WHEEL_POLICY)):
            raise ValueError("task/archival prefix lacks exact source-weight/target-kernel provenance")
        migration = {k: record[k] for k in ("plan_path", "plan_sha256", "source_checkpoint_sha256",
            "source_runtime_content_sha256", "target_runtime_content_sha256")}
    else:
        if (source != target or previous["runtime_contract"] != contract
                or target["version"] not in (TASK_CONDITIONED_HIP_WHEEL_POLICY, RECEIVING_WHEEL_POLICY)):
            raise ValueError("task-conditioned prefix must use verified migration or exact target checkpoint resume")
        migration = None
    return {"source_policy_contract": source, "effective_policy_contract": target,
        "effective_runtime_content_sha256": target_runtime, key: migration}


def _request_history_prefix_provenance(args, contract, previous):
    """Do not relabel an old weight file as if it already stored the new kernel."""
    from .semantic_policy_distribution import (HISTORY_REQUEST_CAP_TRANSITION_POLICY,
        HISTORY_QUARTER_TEMPERED_POLICY, FR_KNEE_PHYSICAL_INNOVATION_POLICY, TASK_CONDITIONED_HIP_WHEEL_POLICY,
        supported_heteroscedastic_contract_version)
    target = _resolved_policy_contract(args)
    from .semantic_p05_capture_profile import P05_CAPTURE_POLICY
    from .semantic_rr_capture_profile import RR_CAPTURE_POLICY
    from .semantic_rear_policy_timing_profile import REAR_POLICY_TIMING_POLICY
    if target["version"] in (P05_CAPTURE_POLICY,RR_CAPTURE_POLICY,REAR_POLICY_TIMING_POLICY):
        if (previous.get("policy_contract") != target or previous["runtime_contract"] != contract
                or getattr(args, "_migration_record", None) is not None):
            raise ValueError("P05 prefix requires the saved/reloaded migrated checkpoint in its exact runtime")
        return {"source_policy_contract": target, "effective_policy_contract": target,
                "effective_runtime_content_sha256": contract["runtime_content_sha256"],
                "p05_capture_assist_migration": None}
    if (target["version"] in (TASK_CONDITIONED_HIP_WHEEL_POLICY, RECEIVING_WHEEL_POLICY)
            or (getattr(args, "_migration_record", None) or {}).get("archive_only_exact_bytes_factor") is not None):
        return _task_conditioned_prefix_provenance(args, contract, previous)
    if target["version"] not in (HISTORY_REQUEST_CAP_TRANSITION_POLICY, FR_KNEE_PHYSICAL_INNOVATION_POLICY):
        return {}
    physical_innovation = target["version"] == FR_KNEE_PHYSICAL_INNOVATION_POLICY
    source_kernel = HISTORY_REQUEST_CAP_TRANSITION_POLICY if physical_innovation else HISTORY_QUARTER_TEMPERED_POLICY
    factor_key = "physical_innovation_sigma_factor" if physical_innovation else "request_history_kernel_factor"
    provenance_key = "physical_innovation_sigma_migration" if physical_innovation else "request_history_kernel_migration"
    source = previous.get("policy_contract")
    source_version = supported_heteroscedastic_contract_version(source)
    record = getattr(args, "_migration_record", None)
    factor = (record or {}).get(factor_key)
    migration = None
    if source_version == source_kernel:
        if (factor is None or factor.get("source_policy_contract") != source
                or factor.get("target_policy_contract") != target
                or record.get("target_runtime_content_sha256") != contract["runtime_content_sha256"]
                or record.get("source_runtime_content_sha256") != previous["runtime_contract"]["runtime_content_sha256"]):
            raise ValueError("old checkpoint prefix requires explicit source-weight/new-kernel migration provenance")
        migration = {key: record[key] for key in ("plan_path", "plan_sha256",
            "source_checkpoint_sha256", "source_runtime_content_sha256", "target_runtime_content_sha256")}
    elif (source_version != target["version"] or source != target
            or previous["runtime_contract"] != contract or factor is not None):
        raise ValueError("request-history checkpoint prefix is neither verified migration nor exact new-kernel resume")
    return {"source_policy_contract": source, "effective_policy_contract": target,
        "effective_runtime_content_sha256": contract["runtime_content_sha256"],
        provenance_key: migration}


def _resolved_policy_version(args: argparse.Namespace) -> str:
    """Use preflight's verified choice; direct CPU entry calls verify metadata too."""
    resolved = getattr(args, "_policy_version", None)
    if resolved is not None:
        policy_contract(resolved, observation_layout=_resolved_observation_layout(args))
        # Reject an unsupported internal selection, including a missing role layout.
        return resolved
    if args.checkpoint is None:
        return LEGACY_POLICY
    from .semantic_migration import checkpoint_metadata
    return policy_version_from_metadata(checkpoint_metadata(args.checkpoint))


def _resolved_observation_layout(args: argparse.Namespace) -> str | None:
    if hasattr(args, "_observation_layout"):
        return args._observation_layout
    append = (getattr(args, "_warm_start_record", None) or {}).get("observation_append_transition")
    if append is not None:
        return append["target_observation_layout"]
    if args.checkpoint is None:
        return None
    from .semantic_migration import checkpoint_metadata
    return policy_observation_layout_from_metadata(checkpoint_metadata(args.checkpoint))


def _resolved_policy_contract(args: argparse.Namespace) -> dict[str, Any]:
    return policy_contract(_resolved_policy_version(args),
                           observation_layout=_resolved_observation_layout(args))


def _observation_layout_options(args: argparse.Namespace) -> dict[str, str]:
    layout = _resolved_observation_layout(args)
    return {} if layout is None else {"observation_layout": layout}


def _save_policy_migration_initial(runner: Any, env: Any, args: argparse.Namespace,
                                   contract: dict[str, Any], output_root: Path,
                                   previous: dict[str, Any]) -> None:
    """Publish a separate immutable conversion boundary without spending samples."""
    from .semantic_migration import continuation_topology
    record = args._policy_migration_record
    initial = output_root / "checkpoints/history" / policy_migration_checkpoint_name(record)
    if initial.exists() or initial.with_name(initial.stem + "_manifest.json").exists():
        raise ValueError("policy migration initial checkpoint already exists; refusing to overwrite")
    write_json(args.run_dir / "policy_distribution_migration.json", record)
    initial_infos = {**previous, "runtime_contract": contract, "semantic_version": args.semantic_version,
        "stage": "initial_policy_distribution_migration",
        "policy_distribution_migration": record,
        "policy_contract": _resolved_policy_contract(args),
        "execution_topology": continuation_topology(env.cfg["reset_sampling"], env.cfg.get("prefix_request"),
            observation_layout=_resolved_observation_layout(args)),
        "curriculum_epoch": semantic_curriculum_epoch(env.cfg),
        "sampling": env.cfg["reset_sampling"],
        "implemented_reset_sampling": env.cfg["reset_sampling"],
        "phase_suffix_curriculum_implemented": env.cfg.get("prefix_request") is not None,
        "runner_config": semantic_runner_config(seed=args.seed, device=args.device,
            semantic_version=args.semantic_version, policy_version=_resolved_policy_version(args),
            observation_layout=_resolved_observation_layout(args))}
    save_semantic_checkpoint(runner, initial, initial_infos)


def _live_runtime_identity(core: Any, args: argparse.Namespace, contract: dict[str, Any], *,
                           boundary: str) -> dict[str, Any]:
    """Read actual already-reset objects/modules only; never import/step/re-read sensors."""
    def attributes(value):
        return {} if value is None else getattr(value, "__dict__", {})
    def class_record(cls):
        module = sys.modules.get(cls.__module__)
        spec = None if module is None else getattr(module, "__spec__", None)
        return {"class": cls.__qualname__, "module": cls.__module__,
                "module_file": None if module is None else getattr(module, "__file__", None),
                "module_origin": None if spec is None else spec.origin}
    def identity(value):
        if value is None:
            return None
        return {**class_record(type(value)),
                "mro": [class_record(cls) for cls in type(value).__mro__ if cls is not object]}
    def callable_record(value):
        if value is None:
            return None
        function = getattr(value, "__func__", value)
        module_name = getattr(function, "__module__", type(function).__module__)
        module = sys.modules.get(module_name)
        code = getattr(function, "__code__", None)
        return {"qualname": getattr(function, "__qualname__", type(function).__qualname__),
                "module": module_name,
                "module_file": None if module is None else getattr(module, "__file__", None),
                "source_file": None if code is None else code.co_filename,
                "source_firstlineno": None if code is None else code.co_firstlineno}
    def absolute_local_path(value):
        # Lexical absolutization only: no filesystem/asset-resolver inspection.
        if value is None:
            return None
        text = str(value)
        return None if not text.strip() or "://" in text else os.path.abspath(text)
    backend = getattr(core, "backend", None)
    members = attributes(backend)
    outer = members.get("_controller")
    outer_members = attributes(outer)
    active = outer_members.get("_semantic")
    teacher = outer_members.get("teacher")
    if active is None:
        active = teacher if teacher is not None else outer
    active_members = attributes(active)
    provider = active_members.get("nominal_provider")
    provider_members = attributes(provider)
    adapter = members.get("_adapter")
    reader = members.get("_reader")
    adapter_members, reader_members = attributes(adapter), attributes(reader)
    reference_spec = provider_members.get("_reference_fsm_spec")
    if reference_spec is None:
        reference_spec = active_members.get("spec")
    source_path = attributes(reference_spec).get("path")
    mapper = adapter_members.get("servo_target_mapper")
    recovery = active_members.get("recovery")
    dependencies = members.get("_dependencies")
    factory_names = ("create_scene", "create_sensing_backends", "adapter_from_scene",
                     "reader_from_scene", "controller_from_paths", "capture_reset_state", "reset_scene")
    # BackendDependencies and SceneHandle are slots dataclasses; these are
    # stored references, never calls to scene/sensor/reset factories.
    scene = members.get("_scene")
    robot = getattr(scene, "robot", None)
    robot_cfg = attributes(robot).get("cfg")  # AssetBase stores cfg.copy().
    spawn_cfg = getattr(robot_cfg, "spawn", None)
    configured_usd = getattr(spawn_cfg, "usd_path", None)
    apply_method = getattr(adapter, "apply_full12", None)
    apply_function = getattr(apply_method, "__func__", apply_method)
    adapter_globals = getattr(apply_function, "__globals__", {})
    residual_module = sys.modules.get("wlr50_clean.ppo.semantic_residual_adapter")
    residual_symbols = {} if residual_module is None else vars(residual_module)
    dispatch_class = residual_symbols.get("SemanticActuationDispatch")
    frame = members.get("_authoritative_frame")
    # AuthoritativeFrame is a slots dataclass; these are stored values, not reads.
    frame_fields = {name: getattr(frame, name, None) for name in ("physics_tick", "state_id")}
    initialized = all(value is not None for value in (outer, adapter, reader, frame))
    loaded_reference = {}
    # Distinguish already-imported reference classes from actually selected objects.
    for module_name, symbol in (
            ("wlr50_clean.fsm.controller", "SensorFsmController"),
            ("wlr50_clean.fsm.recovery", "RecoveryPlanner")):
        module = sys.modules.get(module_name)
        cls = None if module is None else vars(module).get(symbol)
        loaded_reference[symbol] = None if cls is None else class_record(cls)
    return {
        "schema": "wlr50_clean.live_runtime_identity.v1",
        "evidence_scope": ("current_process_already_reset_actual_objects_not_offline_reconstruction"
                           if initialized else "incomplete_or_synthetic_objects_not_live_runtime_proof"),
        "boundary": boundary, "backend_initialized": initialized,
        "python_executable": sys.executable, "cwd": os.getcwd(), "sys_path": list(sys.path),
        "process_id": os.getpid(), "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_git_commit": contract.get("source_git_commit"),
        "runtime_content_sha256": contract.get("runtime_content_sha256"),
        "experiment_id": getattr(args, "experiment_id", None),
        "command": getattr(args, "command", None), "requested_from_phase": getattr(args, "from_phase", "P01"),
        "physics_tick": frame_fields.get("physics_tick"), "state_id": frame_fields.get("state_id"),
        "backend": identity(backend), "outer_controller": identity(outer),
        "active_controller": identity(active), "prefix_mode": outer_members.get("mode"),
        "nominal_provider": identity(provider),
        "source_motion_executor": identity(provider_members.get("_source_motion")),
        "adapter": identity(adapter), "mapper": identity(mapper),
        "sensor_reader": identity(reader),
        "contact_backend": identity(reader_members.get("contact_backend")),
        "geometry_backend": identity(reader_members.get("geometry_backend")),
        "source_fsm_spec": identity(reference_spec),
        "loaded_source_fsm_spec_path": absolute_local_path(source_path),
        "backend_fsm_path": absolute_local_path(members.get("fsm_path")),
        "effective_configuration_paths": {
            name: absolute_local_path(members.get(name)) for name in (
                "execution_profile_path", "task_spec_path", "motion_contract_path", "fsm_path")},
        "configuration_path_source": "actual_backend_stored_attributes_not_reconstructed_from_cli",
        "scene": identity(scene), "scene_robot": identity(robot),
        "configured_robot_asset": {
            "source": "actual_backend._scene.robot.cfg.spawn.usd_path",
            "configured_usd_path_as_stored": None if configured_usd is None else str(configured_usd),
            "configured_usd_absolute_path": absolute_local_path(configured_usd),
            "evidence_scope": "active_robot_configured_path_only_not_USD_resolver_dependencies",
            "USD_stage_inspected": False, "resolved_dependency_proof": False},
        "backend_dependencies": identity(dependencies),
        "dependency_factory_callables": {
            name: callable_record(getattr(dependencies, name, None)) for name in factory_names},
        "dependency_factory_scope": "actual_injected_references_not_claim_each_factory_used_by_current_N",
        "semantic_controller_factory_override": callable_record(members.get("_semantic_controller_factory")),
        "active_feedback_method_callables": {
            "provider_source_normal_bias": callable_record(getattr(provider, "_source_normal_bias", None)),
            "provider_start_source_motion": callable_record(getattr(provider, "_start_source_motion", None)),
            "backend_atomic_apply": callable_record(getattr(backend, "_atomic_apply", None)),
            "adapter_apply_full12": callable_record(apply_method),
            "mapper_advance": callable_record(getattr(mapper, "advance", None)),
            "adapter_bound_feedback_step_global": callable_record(adapter_globals.get("bounded_drive_feedback_step")),
            "adapter_bias_validation_global": callable_record(adapter_globals.get("_full12_drive_feedback_bias"))},
        "already_loaded_residual_dispatch": {
            "scope": "already_loaded_symbols_only_transient_dispatch_not_retained_between_ticks",
            "class": None if dispatch_class is None else class_record(dispatch_class),
            "apply_full12": callable_record(getattr(dispatch_class, "apply_full12", None)),
            "apply_semantic_residual": callable_record(residual_symbols.get("apply_semantic_residual")),
            "bound_feedback_step_global": callable_record(residual_symbols.get("bounded_drive_feedback_step"))},
        "recovery": identity(recovery),
        "recovery_absence": (
            "deliberately_not_used_by_successful_fsm_derived_semantic_N"
            if recovery is None and provider_members.get("_reference_nominal") is True
            else ("not_present_on_active_object" if recovery is None else None)),
        "loaded_reference_classes_not_necessarily_selected": loaded_reference,
        "control_calls_performed_by_logger": 0, "new_imports_performed_by_logger": 0,
        "historical_run_identity_backfilled": False,
    }


def _save_live_runtime_identity(core: Any, args: argparse.Namespace, contract: dict[str, Any], *,
                                boundary: str) -> None:
    record = _live_runtime_identity(core, args, contract, boundary=boundary)
    # A fresh per-run artifact: do not overwrite another process's evidence.
    with (args.run_dir / "live_runtime_identity.json").open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _evaluation(core: Any, args: argparse.Namespace, *, contract: dict[str, Any]) -> dict[str, Any]:
    if args.command == "eval":
        from .semantic_legacy_evaluation import PhysicalEvaluationRecorder
        kwargs = {}
        if args.semantic_version == "v3":
            config_root = _request_paths(args)[2]
            kwargs = {"task_spec_path": config_root / "stage_task_spec.yaml",
                      "quality_score_path": config_root / "quality_score.yaml"}
        scope = closing(PhysicalEvaluationRecorder(args.run_dir, **kwargs))
    else:
        scope = nullcontext(None)
    with scope as recorder:
        return _evaluation_body(core, args, contract=contract, recorder=recorder)


def _evaluation_body(core: Any, args: argparse.Namespace, *, contract: dict[str, Any], recorder: Any) -> dict[str, Any]:
    import torch
    from tensordict import TensorDict
    observation = tuple(core.reset(seed=args.seed))
    _save_live_runtime_identity(core, args, contract, boundary="evaluation_after_actual_reset")
    if recorder is not None:
        recorder.start(core.frame)
        core.tick_observer = recorder.observe
    runner = None
    if args.checkpoint is not None:
        class ObservationEnv:
            num_envs, num_actions = 1, 12
            cfg = {"evaluation": True, "semantic_version": args.semantic_version}
            def get_observations(self):
                tensor = torch.tensor([observation], dtype=torch.float32, device=args.device)
                return TensorDict({"policy": tensor, "critic": tensor.clone()}, batch_size=[1], device=args.device)
        metadata = json.loads(args.checkpoint.with_name(args.checkpoint.stem + "_manifest.json").read_text())
        runner, _ = construct_semantic_runner(ObservationEnv(), seed=int(metadata["seed"]), device=args.device,
            policy_version=_resolved_policy_version(args), initialize_actor=False,
            **_observation_layout_options(args))
        load_semantic_checkpoint(runner, args.checkpoint, contract=contract, seed=int(metadata["seed"]),
                                 migration=getattr(args, "_migration_record", None))
        runner.alg.eval_mode()
    total_reward = 0.0
    last_info: dict[str, Any] = {}
    transitions = 0
    reset_count, local_tick, native_effect_decisions = 1, 0, 0
    reset_records = [{"before_policy_decision": 1, "kind": "initial"}]
    with (args.run_dir / "residual_and_projection_audit.jsonl").open("x", encoding="utf-8") as stream:
        for index in range(args.max_decisions):
            if args.command == "smoke" and (index == 64 or bool(core.done)):
                observation = tuple(core.reset(seed=args.seed))
                reset_count += 1
                local_tick = 0
                reset_records.append({"before_policy_decision": index + 1, "kind": "same_backend_reset"})
            if runner is not None:
                tensor = torch.tensor([observation], dtype=torch.float32, device=args.device)
                obs = TensorDict({"policy": tensor, "critic": tensor.clone()}, batch_size=[1], device=args.device)
                with torch.inference_mode():
                    raw = tuple(runner.alg.actor(obs, stochastic_output=False)[0].cpu().tolist())
            elif args.command == "smoke":
                # A functional 5%-of-engineering-cap bipolar excitation, not a
                # full-task-success prerequisite or a pretend learned action.
                raw = tuple(math.atanh(0.05 if (local_tick // 3) % 2 == 0 else -0.05)
                            if local_tick >= 4 and channel == local_tick % 12 else 0.0 for channel in range(12))
            else:
                raw = (0.0,) * 12
            step = core.step(raw)
            observation = tuple(step.observation)
            total_reward += float(step.reward)
            last_info = jsonable(dict(step.info))
            if (len(observation) < 1 or any(not math.isfinite(float(value)) for value in observation)
                    or not math.isfinite(float(step.reward)) or tuple(last_info["raw_policy_action_full12"]) != raw):
                raise RuntimeError("evaluation transition has an invalid observation/reward/raw-action binding")
            if args.command == "smoke":
                changed = verified_native_effect(last_info, raw)
                native = last_info["actuator_target_effect_audit"]
                native_effect_decisions += int(changed > 0)
                if local_tick < 4 and (changed != 0 or any(native["projected_residual_full12"])):
                    raise RuntimeError("reset zero-residual branch unexpectedly changed actual native targets")
            stream.write(json.dumps(last_info, allow_nan=False) + "\n")
            stream.flush()
            transitions += 1
            local_tick += 1
            if step.truncated:
                raise RuntimeError("semantic task timeout must be terminal, not truncated")
            if step.terminated and args.command != "smoke":
                break
    if args.command == "smoke" and (reset_count < 2 or native_effect_decisions < 1):
        raise RuntimeError("functional smoke must verify repeated reset and a resolvable real native target effect")
    result = {"schema": "wlr50_clean.semantic_evaluation.v1", "mode": "interface_smoke" if args.command == "smoke" else args.mode,
              "seed": args.seed, "checkpoint": None if args.checkpoint is None else str(args.checkpoint),
              "checkpoint_sha256": None if args.checkpoint is None else sha256_file(args.checkpoint),
              "deterministic_policy": runner is not None, "from_phase": "P01", "policy_decisions": transitions,
              "duration_s": float(core.frame.sim_time_s), "task_success": last_info.get("task_success") is True,
              "termination_reason": last_info.get("termination_reason"), "window_ended_before_task_terminal": not bool(core.done),
              "reward_total": total_reward, "telemetry": jsonable(core.telemetry_summary()),
              "runtime_contract": contract, "physical_failure_is_not_interface_failure": True,
              "evaluation_seed_interpretation": "deterministic_repetition_without_randomization"}
    result["checkpoint_resume_migration"] = getattr(args, "_migration_record", None)
    result["policy_contract"] = None if runner is None else _resolved_policy_contract(args)
    result["optimizer_updates_during_evaluation"] = 0
    if args.command == "smoke":
        result["interface_smoke"] = {"reset_count": reset_count, "reset_records": reset_records,
                                     "native_effect_decisions": native_effect_decisions,
                                     "all_decisions_native_audit_verified": True, "functional_passed": True}
    if recorder is not None:
        result["controller_task_success"] = result["task_success"]
        result.update(recorder.summary())
        if result["controller_task_success"] and not result["task_success"]:
            result["controller_termination_reason"] = result["termination_reason"]
            result["termination_reason"] = "INCOMPLETE_PHYSICAL_TASK"
    write_json(args.run_dir / "evaluation_manifest.json", result)
    return result


class GpuProbe:
    def __init__(self, run_dir, device):
        import torch
        self.device = device
        self.stream = (Path(run_dir) / "gpu_memory.jsonl").open("x", encoding="utf-8")
        self.started = time.perf_counter()
        self.rows = []
        if str(device).startswith("cuda"):
            # Torch 2.7's allocator-stat reset does not perform lazy CUDA init.
            # Kit may own a CUDA context while Torch's allocator is still cold.
            torch.cuda.init()
            torch.cuda.reset_peak_memory_stats(device)

    def sample(self, event):
        import torch
        row = {"event": event, "wall_time_s": time.perf_counter()-self.started,
               "pid": os.getpid(), "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)
            free, total = torch.cuda.mem_get_info(self.device)
            row.update(torch_allocated_bytes=torch.cuda.memory_allocated(self.device),
                torch_reserved_bytes=torch.cuda.memory_reserved(self.device),
                torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
                torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(self.device),
                device_free_bytes=free, device_total_bytes=total,
                device_used_bytes=total-free, torch_only_is_not_Kit_process_memory=True)
        for label, query in (
            ("device_memory", ["--query-gpu=index,uuid,memory.used,memory.total"]),
            ("process_memory", ["--query-compute-apps=pid,used_gpu_memory"])):
            try:
                result = subprocess.run(["nvidia-smi", *query, "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                    creationflags=0x08000000 if os.name == "nt" else 0)
                row[label] = {"exit_code": result.returncode, "stdout": result.stdout.strip(),
                              "stderr": result.stderr.strip(),
                              "N_A_is_unavailable_not_zero": True}
            except (OSError, subprocess.TimeoutExpired) as exc:
                row[label] = {"available": False, "reason": str(exc)}
        self.rows.append(row)
        self.stream.write(json.dumps(row, allow_nan=False)+"\n")
        self.stream.flush()
        return row

    def summary(self):
        return {"artifact": str(Path(self.stream.name).resolve()), "sample_count": len(self.rows),
                "peak_of_sampled_device_used_bytes": max(
                    (r["device_used_bytes"] for r in self.rows if "device_used_bytes" in r), default=None),
                "process_peak_not_claimed_when_WDDM_reports_N_A": True}

    def close(self):
        self.stream.close()

class _RowProbe:
    def __init__(self, row, stream):
        self.row, self.stream, self.batches = row, stream, []

    def reset(self):
        self.batches.append([])

    def observe(self, before, after, projection):
        from .isaac_fsm_backend import _member
        raw = after.info["raw_observation"]
        measured = tuple(float(v) for v in _member(raw, "actual_full12"))
        position = tuple(float(v) for v in _member(_member(raw, "base"), "position_w_m"))
        if len(measured) != 12 or len(position) != 3 or any(
                not math.isfinite(v) for v in (*measured, *position)):
            raise RuntimeError("N8 probe requires actual finite physical row measurements")
        row = {"env_index": self.row, "reset_index": len(self.batches)-1,
               "physics_tick": after.physics_tick, "sim_time_s": after.sim_time_s,
               "actual_full12": measured, "base_position_m": position,
               "source_phase": before.state_id, "phase": after.state_id}
        self.batches[-1].append(row)
        self.stream.write(json.dumps(row, allow_nan=False)+"\n")

    def summary(self):
        return {"measured_physics_ticks": sum(map(len, self.batches))}

def smoke_actions(index):
    # 0..7 zero; 8..31 row0-only wheel excitation; 32..63 row-identifiable.
    if index < 8:
        return [[0.0]*12 for _ in range(8)]
    if index < 32:
        rows = [[0.0]*12 for _ in range(8)]
        rows[0][8] = math.atanh(0.10 if index < 20 else -0.10)
        return rows
    return [[math.atanh((0.05+0.01*row)*(1 if (row+channel)%2 else -1))
             for channel in range(12)] for row in range(8)]

def _isolation(control, treatment, backend):
    other_servo, other_wheel, other_base = 0.0, 0.0, 0.0
    active_servo, active_wheel, compared = 0.0, 0.0, 0
    for row in range(8):
        first = {r["physics_tick"]:r for r in control[row]}
        second = {r["physics_tick"]:r for r in treatment[row]}
        common = sorted(set(first) & set(second) & set(range(65,257)))
        if len(common) < 32:
            raise RuntimeError("N8 functional isolation probe ended before enough measured common ticks")
        compared += len(common)
        for tick in common:
            a, b = first[tick], second[tick]
            servo = max(abs(x-y) for x,y in zip(a["actual_full12"][:8], b["actual_full12"][:8]))
            wheel = max(abs(x-y) for x,y in zip(a["actual_full12"][8:], b["actual_full12"][8:]))
            position = max(abs(x-y) for x,y in zip(a["base_position_m"], b["base_position_m"]))
            if row == 0:
                active_servo, active_wheel = max(active_servo,servo), max(active_wheel,wheel)
            else:
                other_servo, other_wheel, other_base = max(other_servo,servo), max(other_wheel,wheel), max(other_base,position)
    origins = backend.scene.env_origins.detach().cpu().tolist()
    minimum_spacing = min(math.dist(origins[a], origins[b]) for a in range(8) for b in range(a))
    # Interface repeatability/isolation tolerances; not reward or task-success gates.
    passed = (other_servo <= 0.01 and other_wheel <= 0.002 and other_base <= 0.00002
              and (active_servo > 0.0001 or active_wheel > 0.00001)
              and minimum_spacing >= 6.0 and backend.scene.cfg.filter_collisions is True
              and backend.scene.cfg.replicate_physics is True)
    return {"passed": passed, "compared_physics_rows": compared,
            "other_row_servo_delta_max_deg": other_servo, "other_row_wheel_delta_max_rad_s": other_wheel,
            "other_row_base_delta_max_m": other_base, "active_row_servo_response_deg": active_servo,
            "active_row_wheel_response_rad_s": active_wheel, "minimum_origin_spacing_m": minimum_spacing,
            "collision_filtering": bool(backend.scene.cfg.filter_collisions),
            "same_nominal_zero_control_then_row0_only_intervention": True}

def _smoke(backend, args, contract, gpu):
    import torch
    from .semantic_vector_env import SemanticVectorRslEnv
    effects = [0]*8
    with (args.run_dir/"vector_physical_probe.jsonl").open("x",encoding="utf-8") as physical, \
         (args.run_dir/"vector_native_audit.jsonl").open("x",encoding="utf-8") as audit:
        probes = tuple(_RowProbe(row,physical) for row in range(8))
        env = SemanticVectorRslEnv(backend,device=args.device,tick_observers=probes)
        # No actor/optimizer is constructed for this functional test.
        env.bind_final_value_function(lambda obs: torch.zeros((8,1),device=args.device),gamma=env.gamma)
        selections = []
        gpu.sample("after_explicit_reset_1")
        for segment in range(2):
            if segment:
                env._reset_all()
                gpu.sample("after_explicit_reset_2")
            selected = [len(p.batches)-1 for p in probes]
            selections.append(selected)
            for decision in range(64):
                requested = [[0.0]*12 for _ in range(8)] if segment == 0 else smoke_actions(decision)
                actions = torch.tensor(requested,dtype=torch.float32,device=args.device)
                _, _, dones, extras = env.step(actions)
                if len(extras["semantic_decisions"]) != 8:
                    raise RuntimeError("N8 smoke lost a physical row")
                for row, info in enumerate(extras["semantic_decisions"]):
                    # The real vector kernel has already bound every physical tick,
                    # request, raw sample, target dispatch and four zero write counters.
                    if (tuple(info["raw_policy_action_full12"]) != tuple(actions[row].cpu().tolist())
                            or info["actuator_target_effect_audit_summary"]["all_ticks_verified"] is not True
                            or info["no_in_episode_state_writes_verified"] is not True):
                        raise RuntimeError("N8 smoke row/action/native evidence mismatch")
                    n = info["actuator_target_effect_audit_summary"]["actual_native_effect_tick_count"]
                    if (segment == 0 or decision < 8 or (decision < 32 and row != 0)) and n:
                        raise RuntimeError("unexcited physical row received a residual native target change")
                    if segment == 1 and decision >= 32:
                        effects[row] += n
                    audit.write(json.dumps({"segment":segment,"decision":decision,"env_index":row,
                                            "info":jsonable(info)},allow_nan=False)+"\n")
                if bool(dones.any()):
                    break  # Valid failure; never continue a terminated scene secretly.
            physical.flush(); audit.flush()
        control = [p.batches[selections[0][row]] for row,p in enumerate(probes)]
        treatment = [p.batches[selections[1][row]] for row,p in enumerate(probes)]
        isolation = _isolation(control,treatment,backend)
        gpu.sample("after_zero_identifiable_and_isolation_probes")
        if not isolation["passed"] or any(n < 1 for n in effects):
            raise RuntimeError("N8 interface evidence insufficient; this is not a full-task-success requirement")
        result = {"schema":"wlr50_clean.semantic_vector_smoke.v1","num_envs":8,
            "explicit_reset_count":2,"actual_reset_count":env.reset_count,"optimizer_steps":0,
            "per_row_effect_counts":effects,"all_row_native_audits_verified":True,
            "one_step_write_capture_verified":True,"physical_isolation_verified":True,
            "physical_isolation":isolation,"functional_passed":True,
            "raw_action_range_abs_max":math.atanh(0.12),"runtime_contract":contract,
            "task_success_not_required":True,"telemetry":env.telemetry_summary(),
            "gpu_measurements":gpu.summary()}
        write_json(args.run_dir/"vector_smoke_manifest.json",result)
        return result

def dispatch_vector(app,args,contract,output_root):
    from .semantic_vector_backend import SemanticVectorIsaacBackend
    from .semantic_vector_env import SemanticVectorRslEnv
    from .semantic_vector_training import construct_semantic_vector_runner, train_semantic_vector
    gpu = GpuProbe(args.run_dir,args.device)
    try:
        gpu.sample("before_vector_scene")
        backend = SemanticVectorIsaacBackend(app,num_envs=8)
        gpu.sample("after_vector_scene")
        if args.command == "smoke":
            return _smoke(backend,args,contract,gpu)
        env = SemanticVectorRslEnv(backend,device=args.device,seeds=tuple(range(1001,1009)))
        env.gpu_probe = gpu
        env.cfg["vector_smoke_evidence"] = args._vector_smoke_record
        gpu.sample("after_training_legal_reset")
        runner,_ = construct_semantic_vector_runner(env,seed=args.seed,device=args.device)
        previous = load_semantic_checkpoint(runner,args.checkpoint,contract=contract,seed=args.seed,
                                            migration=args._migration_record)
        if runner.alg.storage.step != 0 or runner.alg.storage.actions.shape != (128,8,12):
            raise RuntimeError("N8 migration must start with fresh complete 128x8 raw rollout storage")
        # Runtime contract binds the same explicit ceiling checked before launch.
        remaining = contract["training_budgets"][args.stage]-int(previous["stage_requested_decisions"][args.stage])
        requested = stage_partition(remaining)["N8_requested"] if args.decisions is None else args.decisions
        gpu.sample("after_actual_checkpoint_reload")
        result = train_semantic_vector(runner,env,decisions=requested,run_dir=args.run_dir,
            output_root=output_root,stage=args.stage,contract=contract,seed=args.seed,
            resume_infos=previous,checkpoint_interval_updates=args.checkpoint_interval_updates)
        return result
    finally:
        gpu.close()


def dispatch_live(args: argparse.Namespace, contract: dict[str, Any]) -> dict[str, Any]:
    # Resolve the installed PyTorch/TensorDict native DLLs before Kit extends
    # the Windows DLL search path. Loading tensordict._C after Kit produced an
    # access violation on this pinned stack; these imports create no scene.
    import torch
    import tensordict
    # No Isaac scene or environment imports before AppLauncher.
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=bool(args.headless), enable_cameras=False).app
    args._live_app = app
    app.update()
    try:
        if args.command == "eval" and args.mode == "legacy_fsm_eval":
            from .semantic_legacy_evaluation import _evaluation_legacy
            kwargs = {}
            if args.semantic_version == "v3":
                config_root = _request_paths(args)[2]
                kwargs = {"task_spec_path": config_root / "stage_task_spec.yaml",
                          "quality_score_path": config_root / "quality_score.yaml"}
            return _evaluation_legacy(app, args, contract, **kwargs)
        if args.num_envs == 8:
            return dispatch_vector(app, args, contract, OUTPUT_ROOT)
        from .semantic_backend import SemanticIsaacBackend
        from .semantic_env import SemanticEpisodeEnv
        seed_training_rngs(args.seed)
        _, output_root, config_root = _request_paths(args)
        backend_options = {"audit_actuator_target_effect": True}
        core_options = {"collect_trace": False}
        if args.semantic_version == "v3":
            backend_options.update(execution_profile=config_root / "execution_profile.yaml",
                                   task_spec_path=config_root / "stage_task_spec.yaml")
            core_options.update(action_config=config_root / "execution_profile.yaml",
                                reward_config_path=config_root / "reward_config.yaml",
                                observation_schema_path=config_root / "observation_schema.json")
        nominal_prefix = getattr(args, "prefix_source", "frozen_fsm") == "successful_nominal"
        checkpoint_prefix = getattr(args, "prefix_source", "frozen_fsm") in ("checkpoint_policy", "successful_nominal")
        if args.from_phase != "P01" and not checkpoint_prefix:
            from .semantic_prefix import PrefixRequest, PrefixSemanticIsaacBackend
            backend = PrefixSemanticIsaacBackend(app, prefix_request=PrefixRequest(
                target_phase=args.from_phase, teacher_offset_decisions=args.teacher_offset_decisions), **backend_options)
        else:
            backend = SemanticIsaacBackend(app, **backend_options)
        core = SemanticEpisodeEnv(backend, **core_options)
        if args.command != "train":
            return _evaluation(core, args, contract=contract)
        if args.from_phase != "P01":
            from .semantic_prefix import PrefixRslAdapter
            prefix_stream = (args.run_dir / "prefix_evidence.jsonl").open("x", encoding="utf-8")
            # Keep the persistent sink open across every reset. Kit's finally
            # path closes it before native shutdown, including failed roll-ins.
            args._prefix_evidence_stream = prefix_stream
            def prefix_evidence(record):
                prefix_stream.write(json.dumps(jsonable(record), allow_nan=False) + "\n")
                prefix_stream.flush()
            if checkpoint_prefix:
                from .semantic_checkpoint_prefix import CheckpointPolicyPrefixRequest, CheckpointPolicyPrefixRslAdapter
                env = CheckpointPolicyPrefixRslAdapter(core, seed=args.seed, device=args.device,
                    evidence_sink=prefix_evidence, request=CheckpointPolicyPrefixRequest(
                        target_phase=args.from_phase, teacher_offset_decisions=args.teacher_offset_decisions,
                        source="successful_nominal" if nominal_prefix else "frozen_checkpoint_policy"))
            else:
                env = PrefixRslAdapter(core, seed=args.seed, device=args.device, evidence_sink=prefix_evidence)
        else:
            env = SemanticRslAdapter(core, seed=args.seed, device=args.device)
        env.cfg["semantic_version"] = args.semantic_version
        runner, _ = construct_semantic_runner(env, seed=args.seed, device=args.device,
            policy_version=_resolved_policy_version(args), initialize_actor=args.checkpoint is None,
            **_observation_layout_options(args))
        previous = None
        if args.checkpoint is not None:
            previous = load_semantic_checkpoint(runner, args.checkpoint, contract=contract, seed=args.seed,
                                                migration=getattr(args, "_migration_record", None),
                                                warm_start=getattr(args, "_warm_start_record", None),
                                                policy_migration=getattr(args, "_policy_migration_record", None))
            if nominal_prefix:
                from .semantic_transfer_roles import ROLE_OBSERVATION_LAYOUT
                # Same N/controller/mapper/history as B and C. Only reset-time
                # zero residual creates this legal training start; no PPO credit.
                env.install_prefix_policy(lambda observation: (0.0,)*12, {
                    "schema": "wlr50_clean.successful_nominal_prefix.v1",
                    "source": "successful_nominal", "raw_action_full12": [0.0]*12,
                    "policy_credit": False,
                    "execution_profile_sha256": sha256_file(config_root / "execution_profile.yaml"),
                    "stage_task_spec_sha256": sha256_file(config_root / "stage_task_spec.yaml"),
                    "runtime_content_sha256": contract["runtime_content_sha256"],
                    "interface_contract": {"observation_dimension": len(core.observation),
                        "observation_layout": _resolved_observation_layout(args), "action_dimension": 12}})
            elif checkpoint_prefix:
                from .semantic_checkpoint_prefix_policy import build_frozen_checkpoint_prefix_policy
                # The loader already verified this exact immutable checkpoint,
                # actor and preprocessing. The independent actor is reset-only;
                # it never replaces/mutates PPO's current transition or storage.
                source = previous["resume_source_checkpoint"]
                frozen_prefix = build_frozen_checkpoint_prefix_policy(runner.alg.actor, {
                    "checkpoint_path": source["checkpoint"], "checkpoint_sha256": source["checkpoint_sha256"],
                    "actor_parameter_sha256": previous["actor_parameter_sha256"],
                    "source_global_policy_decisions": previous["global_policy_decisions"],
                    "source_ppo_updates": previous["ppo_updates"],
                    "policy_contract": _resolved_policy_contract(args),
                    "source_runtime_content_sha256": previous["runtime_contract"]["runtime_content_sha256"],
                    **_request_history_prefix_provenance(args, contract, previous),
                })
                env.install_prefix_policy(frozen_prefix, frozen_prefix.provenance)
            if getattr(args, "_policy_migration_record", None) is not None:
                _save_policy_migration_initial(runner, env, args, contract, output_root, previous)
            if args.new_mdp_warm_start:
                from .semantic_migration import (continuation_topology, v3_warm_start_checkpoint_name,
                                                  warm_start_source_execution_profile)
                from .semantic_training import compare_warm_start_action
                write_json(args.run_dir / "new_mdp_warm_start.json", args._warm_start_record)
                comparison = compare_warm_start_action(runner, env,
                    old_execution_profile=warm_start_source_execution_profile(
                        args._warm_start_record, args.run_dir, project_root=PROJECT_ROOT),
                    new_execution_profile=config_root / "execution_profile.yaml")
                if args._warm_start_record.get("observation_same_layout_transition") is not None:
                    comparison["authority_boundary_scope"] = (
                        "one current nominal/observation and zero-history projectors; the changed cap "
                        "can change projected actions; this does not compare source/target nominal timing")
                    if getattr(args, "experiment_id", None) == "fsm_reference_p09_stable_v2":
                        comparison["authority_boundary_scope"] = (
                            "one current target nominal/observation and zero-history projectors; action ranges "
                            "are unchanged, but this does not compare source/target nominal or observation semantics")
                if args._warm_start_record.get("policy_kernel_transition") is not None:
                    comparison["policy_kernel_context"] = (
                        "both physical-profile projections use the target actor; this is not an old/new-policy comparison; "
                        "see new_mdp_initial_policy_kernel_comparison.json")
                comparison_path = args.run_dir / "new_mdp_initial_action_comparison.json"
                write_json(comparison_path, comparison)
                previous["new_mdp_initial_action_comparison"] = {
                    "path": str(comparison_path.resolve()), "sha256": sha256_file(comparison_path)}
                if args._warm_start_record.get("policy_kernel_transition") is not None:
                    from .semantic_training import compare_warm_start_policy_kernel
                    kernel_comparison = compare_warm_start_policy_kernel(runner, env, record=args._warm_start_record)
                    kernel_path = args.run_dir / "new_mdp_initial_policy_kernel_comparison.json"
                    write_json(kernel_path, kernel_comparison)
                    previous["new_mdp_initial_policy_kernel_comparison"] = {
                        "path": str(kernel_path.resolve()), "sha256": sha256_file(kernel_path)}
                initial_infos = {**previous, "runtime_contract": contract, "stage": "initial_v3_warm_start",
                                 "execution_topology": continuation_topology(env.cfg["reset_sampling"], env.cfg.get("prefix_request"),
                                     observation_layout=_resolved_observation_layout(args)),
                                 "curriculum_epoch": semantic_curriculum_epoch(env.cfg),
                                 "sampling": env.cfg["reset_sampling"],
                                 "implemented_reset_sampling": env.cfg["reset_sampling"],
                                 "phase_suffix_curriculum_implemented": env.cfg.get("prefix_request") is not None,
                                 "policy_contract": _resolved_policy_contract(args),
                                 "runner_config": semantic_runner_config(seed=args.seed, device=args.device,
                                     semantic_version=args.semantic_version, policy_version=_resolved_policy_version(args),
                                     observation_layout=_resolved_observation_layout(args))}
                save_semantic_checkpoint(runner, output_root / "checkpoints/history" /
                    v3_warm_start_checkpoint_name(args._warm_start_record), initial_infos)
        else:
            initial = output_root / "checkpoints/history/checkpoint_initial_semantic.pt"
            save_semantic_checkpoint(runner, initial, {
                "seed": args.seed, "runtime_contract": contract, "stage": "initial",
                "semantic_version": args.semantic_version,
                "policy_contract": _resolved_policy_contract(args),
                "execution_topology": topology(1, observation_layout=_resolved_observation_layout(args)),
                "global_policy_decisions": 0, "ppo_updates": 0, "optimizer_steps": 0,
                "stage_requested_decisions": {stage: 0 for stage in STAGE_BUDGETS},
                "runner_config": semantic_runner_config(seed=args.seed, device=args.device,
                    semantic_version=args.semantic_version, policy_version=_resolved_policy_version(args),
                    observation_layout=_resolved_observation_layout(args)),
            })
        _save_live_runtime_identity(core, args, contract, boundary="training_after_actual_reset_and_checkpoint_load")
        remaining = contract["training_budgets"][args.stage] - int((previous or {}).get("stage_requested_decisions", {}).get(args.stage, 0))
        return train_semantic(runner, env, run_dir=args.run_dir, output_root=_checkpoint_output_root(args),
                              stage=args.stage, decisions=remaining if args.decisions is None else args.decisions,
                              contract=contract, seed=args.seed, resume_infos=previous,
                              checkpoint_interval_updates=args.checkpoint_interval_updates,
                              checkpoint_output_routing=getattr(args, "_checkpoint_output_routing", None))
    finally:
        # main persists the final lifecycle BEFORE closing Kit. Its native
        # immediate-exit path does not return to Python on this Windows stack.
        prefix_stream = getattr(args, "_prefix_evidence_stream", None)
        if prefix_stream is not None:
            prefix_stream.close()
            del args._prefix_evidence_stream


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    validate_request(args)
    contract_options = {"expected_head": args.expected_head}
    if args.semantic_version == "v3":
        contract_options["semantic_version"] = "v3"
    if args.experiment_id is not None:
        contract_options["experiment_id"] = args.experiment_id
    contract = runtime_contract(**contract_options)
    _preflight_checkpoint(args, contract)
    from .semantic_migration import source_num_envs, verified_vector_smoke
    args._vector_smoke_record = None
    if args.command == "train" and args.num_envs == 8:
        args._vector_smoke_record = verified_vector_smoke(args.vector_smoke_evidence, contract, PROJECT_ROOT)
    if args.checkpoint is not None and not args.new_mdp_warm_start:
        from .semantic_migration import checkpoint_metadata
        source_count = source_num_envs(checkpoint_metadata(args.checkpoint))
        factor = (args._migration_record or {}).get("execution_factor")
        if factor is not None and (factor["source_num_envs"] != source_count or factor["target_num_envs"] != args.num_envs):
            raise ValueError("explicit execution plan does not match requested topology")
        if source_count != args.num_envs and (factor is None or
                factor["source_num_envs"] != source_count or factor["target_num_envs"] != args.num_envs):
            raise ValueError("changing checkpoint execution topology requires an explicit exact migration factor")
    args.run_dir.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    lifecycle = {"schema": "wlr50_clean.semantic_run.v1", "command": args.command,
                 "arguments": jsonable(vars(args)), "started_at_utc": started,
                 "runtime_contract": contract, "lifecycle": "RUNNING"}
    write_json(args.run_dir / "run_manifest.started.json", lifecycle)
    try:
        if args.command == "preflight":
            result = {"runtime_valid": True, "isaac_started": False,
                      "legacy_perfect_nonzero_gate_required": False}
        else:
            result = dispatch_live(args, contract)
        if runtime_contract(**contract_options) != contract:
            raise RuntimeError("semantic runtime bytes changed during the run")
        final_status = result.get("lifecycle", "SUCCEEDED")
        lifecycle.update(lifecycle=final_status, result=result, completed_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(args.run_dir / "run_manifest.json", lifecycle)
        print(json.dumps({"semantic_run": str(args.run_dir), "lifecycle": final_status}), flush=True)
        return 0
    except BaseException as exc:
        lifecycle.update(lifecycle="FAILED", error=str(exc), traceback=traceback.format_exc(),
                         completed_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(args.run_dir / "run_manifest.json", lifecycle)
        raise
    finally:
        app = getattr(args, "_live_app", None)
        if app is not None:
            app.close(wait_for_replicator=False, skip_cleanup=True)


if __name__ == "__main__":
    raise SystemExit(main())
