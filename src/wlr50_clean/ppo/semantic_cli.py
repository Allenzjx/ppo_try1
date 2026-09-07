"""Isolated semantic B/C experiments, without the legacy perfect-probe gates."""
from __future__ import annotations

import argparse
from contextlib import closing, nullcontext
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .semantic_training import (
    STAGE_BUDGETS, SemanticRslAdapter, construct_semantic_runner, jsonable,
    load_semantic_checkpoint, save_semantic_checkpoint, seed_training_rngs,
    semantic_runner_config, semantic_curriculum_epoch, sha256_file, train_semantic, verified_native_effect, write_json,
)
from .semantic_migration import experiment_namespace, topology, stage_partition
from .semantic_policy_distribution import (
    LEGACY_POLICY, STATE_DEPENDENT_POLICY, HISTORY_POLICY, policy_contract,
    policy_version_from_metadata, build_policy_distribution_migration,
    policy_migration_checkpoint_name,
)

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
            PROJECT_ROOT / "configs/ppo_semantic_v3")


def _request_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    experiment_id = getattr(args, "experiment_id", None)
    return (version_paths(args.semantic_version) if experiment_id is None else
            version_paths(args.semantic_version, experiment_id=experiment_id))


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
    result.add_argument("--experiment-id", choices=("transfer_roles_v1",))
    result.add_argument("--from-phase", choices=("P01", "P03", "P04", "P05", "P06", "P07", "P08", "P09", "P10", "P11", "P12", "P13"), default="P01")
    result.add_argument("--teacher-offset-decisions", type=int, default=0)
    result.add_argument("--prefix-source", choices=("frozen_fsm", "checkpoint_policy"), default="frozen_fsm")
    result.add_argument("--new-mdp-warm-start", action="store_true")
    result.add_argument("--target-policy-version", choices=(HISTORY_POLICY,))
    result.add_argument("--policy-distribution-migration", action="store_true")
    result.add_argument("--vector-smoke-evidence", type=Path)
    result.add_argument("--stage", choices=tuple(STAGE_BUDGETS), default="smoke")
    result.add_argument("--decisions", type=int)
    result.add_argument("--max-decisions", type=int, default=3000)
    result.add_argument("--checkpoint", type=Path)
    result.add_argument("--resume-migration", type=Path)
    result.add_argument("--mode", choices=("legacy_fsm_eval", "semantic_prior_eval", "semantic_residual_eval"), default="semantic_prior_eval")
    result.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    result.add_argument("--checkpoint-interval-updates", type=int, default=10)
    result.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    return result


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
    for relative, expected in frozen["protected_files"].items():
        if sha256_file(PROJECT_ROOT / relative) != expected:
            raise ValueError(f"frozen A bytes changed: {relative}")
    files = {relative: sha256_file(PROJECT_ROOT / relative)
             for relative in sorted(git("ls-files", "--", *paths).splitlines())}
    if not files:
        raise ValueError("semantic runtime inventory is empty")
    data = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    contract = {"schema": "wlr50_clean.semantic_runtime_contract.v1", "source_git_commit": head,
            "runtime_content_sha256": hashlib.sha256(data).hexdigest(), "files": files,
            "frozen_A_files": dict(frozen["protected_files"]), "rsl_rl_version": "5.0.1",
            "physics_hz": 120.0, "decision_hz": 15.0, "task_timeout_s": 200.0,
            "timeout_bootstrap": False, "training_budgets": dict(STAGE_BUDGETS),
            "local_runtime_versions": local_versions()}
    if semantic_version == "v3":
        config_root = version_paths(semantic_version)[2]
        contract.update(semantic_version="v3", selected_configuration={
            path.name: {"path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                        "sha256": sha256_file(path)} for path in sorted(config_root.iterdir()) if path.is_file()})
    if experiment_id is not None:
        contract["experiment_id"] = experiment_id
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
    if getattr(args, "prefix_source", "frozen_fsm") == "checkpoint_policy" and (
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
    if args.decisions is not None and (args.command != "train" or not 1 <= args.decisions <= STAGE_BUDGETS[args.stage]):
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
        source_root = output_root
        if args.new_mdp_warm_start:
            if getattr(args, "experiment_id", None) is not None:
                prior_v3_root = version_paths("v3")[1]
                if args.checkpoint.resolve(strict=True).is_relative_to((prior_v3_root / "checkpoints").resolve()):
                    source_root = prior_v3_root
            elif args.checkpoint.resolve(strict=True).is_relative_to((OUTPUT_ROOT / "checkpoints").resolve()):
                source_root = OUTPUT_ROOT
        args.checkpoint = _resolved_checkpoint(args.checkpoint, output_root=source_root)
        if args.command == "train":
            metadata = json.loads(args.checkpoint.with_name(args.checkpoint.stem + "_manifest.json").read_text())
            source_version = metadata.get("semantic_version", "v2")
            if args.new_mdp_warm_start:
                expected_version = "v2" if source_root == OUTPUT_ROOT else "v3"
                if source_version != expected_version:
                    raise ValueError("warm-start checkpoint version differs from its isolated source root")
                if source_version == "v2" and any((output_root / relative).exists() for relative in (
                        "checkpoints/checkpoint_last_pointer.json", "checkpoints/history/checkpoint_initial_v3_warm_start.pt")):
                    raise ValueError("v3 training already exists; continue its checkpoint without restarting v3 budgets")
            reset_v3_budget = args.new_mdp_warm_start and source_version == "v2"
            remaining = STAGE_BUDGETS[args.stage] - (0 if reset_v3_budget else int(metadata["stage_requested_decisions"].get(args.stage, 0)))
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
            left = STAGE_BUDGETS[args.stage]-int(metadata["stage_requested_decisions"][args.stage])
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
    if args.checkpoint is None:
        if getattr(args, "policy_distribution_migration", False):
            raise ValueError("policy distribution migration requires a saved checkpoint")
        return
    from .semantic_migration import checkpoint_metadata, validate_migration_plan
    metadata = checkpoint_metadata(args.checkpoint)
    if (not args.new_mdp_warm_start and args.resume_migration is None
            and not getattr(args, "policy_distribution_migration", False)
            and metadata["runtime_contract"] != contract):
        raise ValueError("checkpoint runtime changed; an explicit reviewed resume migration is required")
    args._policy_version = policy_version_from_metadata(metadata)
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
        if metadata.get("runner_config") != semantic_runner_config(seed=int(metadata["seed"]), device=args.device,
                semantic_version=metadata.get("semantic_version", "v2"), policy_version=args._policy_version):
            raise ValueError("migration cannot change PPO hyperparameters or normalization")
    if args.command == "train" and metadata["seed"] != args.seed:
        raise ValueError("resume must preserve the checkpoint training RNG seed")


def _resolved_policy_version(args: argparse.Namespace) -> str:
    """Use preflight's verified choice; direct CPU entry calls verify metadata too."""
    resolved = getattr(args, "_policy_version", None)
    if resolved is not None:
        policy_contract(resolved)  # Reject an unsupported internal selection.
        return resolved
    if args.checkpoint is None:
        return LEGACY_POLICY
    from .semantic_migration import checkpoint_metadata
    return policy_version_from_metadata(checkpoint_metadata(args.checkpoint))


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
        "policy_contract": policy_contract(_resolved_policy_version(args)),
        "execution_topology": continuation_topology(env.cfg["reset_sampling"], env.cfg.get("prefix_request")),
        "curriculum_epoch": semantic_curriculum_epoch(env.cfg),
        "sampling": env.cfg["reset_sampling"],
        "implemented_reset_sampling": env.cfg["reset_sampling"],
        "phase_suffix_curriculum_implemented": env.cfg.get("prefix_request") is not None,
        "runner_config": semantic_runner_config(seed=args.seed, device=args.device,
            semantic_version=args.semantic_version, policy_version=_resolved_policy_version(args))}
    save_semantic_checkpoint(runner, initial, initial_infos)


def _evaluation(core: Any, args: argparse.Namespace, *, contract: dict[str, Any]) -> dict[str, Any]:
    if args.command == "eval":
        from .semantic_legacy_evaluation import PhysicalEvaluationRecorder
        kwargs = {}
        if args.semantic_version == "v3":
            config_root = version_paths("v3")[2]
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
            policy_version=_resolved_policy_version(args), initialize_actor=False)
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
    result["policy_contract"] = None if runner is None else policy_contract(_resolved_policy_version(args))
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
        # CLI already requires a declared stage budget; use canonical constant here.
        from .semantic_training import STAGE_BUDGETS
        remaining = STAGE_BUDGETS[args.stage]-int(previous["stage_requested_decisions"][args.stage])
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
                config_root = version_paths("v3")[2]
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
        checkpoint_prefix = getattr(args, "prefix_source", "frozen_fsm") == "checkpoint_policy"
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
                        target_phase=args.from_phase, teacher_offset_decisions=args.teacher_offset_decisions))
            else:
                env = PrefixRslAdapter(core, seed=args.seed, device=args.device, evidence_sink=prefix_evidence)
        else:
            env = SemanticRslAdapter(core, seed=args.seed, device=args.device)
        env.cfg["semantic_version"] = args.semantic_version
        runner, _ = construct_semantic_runner(env, seed=args.seed, device=args.device,
            policy_version=_resolved_policy_version(args), initialize_actor=args.checkpoint is None)
        previous = None
        if args.checkpoint is not None:
            previous = load_semantic_checkpoint(runner, args.checkpoint, contract=contract, seed=args.seed,
                                                migration=getattr(args, "_migration_record", None),
                                                warm_start=getattr(args, "_warm_start_record", None),
                                                policy_migration=getattr(args, "_policy_migration_record", None))
            if checkpoint_prefix:
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
                    "policy_contract": policy_contract(_resolved_policy_version(args)),
                    "source_runtime_content_sha256": previous["runtime_contract"]["runtime_content_sha256"],
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
                                 "execution_topology": continuation_topology(env.cfg["reset_sampling"], env.cfg.get("prefix_request")),
                                 "curriculum_epoch": semantic_curriculum_epoch(env.cfg),
                                 "sampling": env.cfg["reset_sampling"],
                                 "implemented_reset_sampling": env.cfg["reset_sampling"],
                                 "phase_suffix_curriculum_implemented": env.cfg.get("prefix_request") is not None,
                                 "policy_contract": policy_contract(_resolved_policy_version(args)),
                                 "runner_config": semantic_runner_config(seed=args.seed, device=args.device,
                                     semantic_version=args.semantic_version, policy_version=_resolved_policy_version(args))}
                save_semantic_checkpoint(runner, output_root / "checkpoints/history" /
                    v3_warm_start_checkpoint_name(args._warm_start_record), initial_infos)
        else:
            initial = output_root / "checkpoints/history/checkpoint_initial_semantic.pt"
            save_semantic_checkpoint(runner, initial, {
                "seed": args.seed, "runtime_contract": contract, "stage": "initial",
                "semantic_version": args.semantic_version,
                "policy_contract": policy_contract(_resolved_policy_version(args)),
                "execution_topology": topology(1),
                "global_policy_decisions": 0, "ppo_updates": 0, "optimizer_steps": 0,
                "stage_requested_decisions": {stage: 0 for stage in STAGE_BUDGETS},
                "runner_config": semantic_runner_config(seed=args.seed, device=args.device,
                    semantic_version=args.semantic_version, policy_version=_resolved_policy_version(args)),
            })
        remaining = STAGE_BUDGETS[args.stage] - int((previous or {}).get("stage_requested_decisions", {}).get(args.stage, 0))
        return train_semantic(runner, env, run_dir=args.run_dir, output_root=output_root,
                              stage=args.stage, decisions=remaining if args.decisions is None else args.decisions,
                              contract=contract, seed=args.seed, resume_infos=previous,
                              checkpoint_interval_updates=args.checkpoint_interval_updates)
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
