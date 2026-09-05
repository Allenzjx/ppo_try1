"""Isolated semantic B/C experiments, without the legacy perfect-probe gates."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
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
    semantic_runner_config, sha256_file, train_semantic, verified_native_effect, write_json,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUNS_ROOT = PROJECT_ROOT / "runs/ppo_semantic_v2"
OUTPUT_ROOT = PROJECT_ROOT / "outputs/ppo_semantic_v2"
LOCKED_DISTRIBUTIONS = {"torch": "2.7.0+cu128", "rsl-rl-lib": "5.0.1",
                        "isaacsim": "5.1.0.0", "isaaclab": "0.54.3", "tensordict": "0.12.2"}


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
    result.add_argument("--stage", choices=tuple(STAGE_BUDGETS), default="smoke")
    result.add_argument("--decisions", type=int)
    result.add_argument("--max-decisions", type=int, default=3000)
    result.add_argument("--checkpoint", type=Path)
    result.add_argument("--mode", choices=("semantic_prior_eval", "semantic_residual_eval"), default="semantic_prior_eval")
    result.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    result.add_argument("--checkpoint-interval-updates", type=int, default=10)
    result.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    return result


def runtime_contract(*, expected_head: str) -> dict[str, Any]:
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
    return {"schema": "wlr50_clean.semantic_runtime_contract.v1", "source_git_commit": head,
            "runtime_content_sha256": hashlib.sha256(data).hexdigest(), "files": files,
            "frozen_A_files": dict(frozen["protected_files"]), "rsl_rl_version": "5.0.1",
            "physics_hz": 120.0, "decision_hz": 15.0, "task_timeout_s": 200.0,
            "timeout_bootstrap": False, "training_budgets": dict(STAGE_BUDGETS),
            "local_runtime_versions": local_versions()}


def _resolved_checkpoint(path: Path) -> Path:
    source = path.resolve(strict=True)
    if not source.is_relative_to((OUTPUT_ROOT / "checkpoints").resolve()):
        raise ValueError("semantic checkpoint must be inside the new isolated checkpoint root")
    if source.name == "checkpoint_last.pt":
        pointer = json.loads(source.with_name("checkpoint_last_pointer.json").read_text(encoding="utf-8"))
        immutable = Path(pointer["checkpoint"]).resolve(strict=True)
        if (not immutable.is_relative_to((OUTPUT_ROOT / "checkpoints/history").resolve())
                or sha256_file(source) != pointer["checkpoint_sha256"]
                or sha256_file(immutable) != pointer["checkpoint_sha256"]
                or sha256_file(Path(pointer["manifest"])) != pointer["manifest_sha256"]):
            raise ValueError("semantic last checkpoint pointer/copy is inconsistent")
        source = immutable
    return source


def validate_request(args: argparse.Namespace) -> None:
    directory = args.run_dir.resolve()
    if not directory.is_relative_to(RUNS_ROOT.resolve()) or directory == RUNS_ROOT.resolve():
        raise ValueError("semantic run directory must be strictly inside runs/ppo_semantic_v2")
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
    if args.command == "eval" and args.mode == "semantic_prior_eval" and args.checkpoint is not None:
        raise ValueError("prior B evaluation must not load a PPO checkpoint")
    if args.command in ("preflight", "smoke") and args.checkpoint is not None:
        raise ValueError("functional preflight/smoke must not silently load a policy")
    if args.command == "train" and args.checkpoint is None and any(path.exists() for path in (
            OUTPUT_ROOT / "checkpoints/checkpoint_last_pointer.json",
            OUTPUT_ROOT / "checkpoints/history/checkpoint_initial_semantic.pt")):
        raise ValueError("training already exists; explicitly resume checkpoint_last instead of reinitializing")
    if args.checkpoint is not None:
        args.checkpoint = _resolved_checkpoint(args.checkpoint)
        if args.command == "train":
            metadata = json.loads(args.checkpoint.with_name(args.checkpoint.stem + "_manifest.json").read_text())
            remaining = STAGE_BUDGETS[args.stage] - int(metadata["stage_requested_decisions"].get(args.stage, 0))
            if remaining < 1 or (args.decisions is not None and args.decisions > remaining):
                raise ValueError("additional request exceeds the remaining semantic stage budget")
    args.run_dir = directory


def _evaluation(core: Any, args: argparse.Namespace, *, contract: dict[str, Any]) -> dict[str, Any]:
    import torch
    from tensordict import TensorDict
    observation = tuple(core.reset(seed=args.seed))
    metrics = None
    if args.command == "eval":
        from .semantic_metrics import SemanticMetricsAccumulator
        metrics = SemanticMetricsAccumulator()
        core.tick_observer = metrics.observe
    runner = None
    if args.checkpoint is not None:
        class ObservationEnv:
            num_envs, num_actions = 1, 12
            cfg = {"evaluation": True}
            def get_observations(self):
                tensor = torch.tensor([observation], dtype=torch.float32, device=args.device)
                return TensorDict({"policy": tensor, "critic": tensor.clone()}, batch_size=[1], device=args.device)
        runner, _ = construct_semantic_runner(ObservationEnv(), seed=1001, device=args.device)
        metadata = json.loads(args.checkpoint.with_name(args.checkpoint.stem + "_manifest.json").read_text())
        load_semantic_checkpoint(runner, args.checkpoint, contract=contract, seed=int(metadata["seed"]))
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
    if args.command == "smoke":
        result["interface_smoke"] = {"reset_count": reset_count, "reset_records": reset_records,
                                     "native_effect_decisions": native_effect_decisions,
                                     "all_decisions_native_audit_verified": True, "functional_passed": True}
    if metrics is not None:
        result["quality_metrics"] = metrics.summary()
    write_json(args.run_dir / "evaluation_manifest.json", result)
    return result


def dispatch_live(args: argparse.Namespace, contract: dict[str, Any]) -> dict[str, Any]:
    # Resolve the installed PyTorch/TensorDict native DLLs before Kit extends
    # the Windows DLL search path. Loading tensordict._C after Kit produced an
    # access violation on this pinned stack; these imports create no scene.
    import torch
    import tensordict
    # No Isaac scene or environment imports before AppLauncher.
    from isaaclab.app import AppLauncher
    app = AppLauncher(headless=bool(args.headless), enable_cameras=False).app
    app.update()
    try:
        from .semantic_backend import SemanticIsaacBackend
        from .semantic_env import SemanticEpisodeEnv
        seed_training_rngs(args.seed)
        backend = SemanticIsaacBackend(app, audit_actuator_target_effect=True)
        core = SemanticEpisodeEnv(backend, collect_trace=False)
        if args.command != "train":
            return _evaluation(core, args, contract=contract)
        env = SemanticRslAdapter(core, seed=args.seed, device=args.device)
        runner, _ = construct_semantic_runner(env, seed=args.seed, device=args.device)
        previous = None
        if args.checkpoint is not None:
            previous = load_semantic_checkpoint(runner, args.checkpoint, contract=contract, seed=args.seed)
        else:
            initial = OUTPUT_ROOT / "checkpoints/history/checkpoint_initial_semantic.pt"
            save_semantic_checkpoint(runner, initial, {
                "seed": args.seed, "runtime_contract": contract, "stage": "initial",
                "global_policy_decisions": 0, "ppo_updates": 0, "optimizer_steps": 0,
                "stage_requested_decisions": {stage: 0 for stage in STAGE_BUDGETS},
                "runner_config": semantic_runner_config(seed=args.seed, device=args.device),
            })
        remaining = STAGE_BUDGETS[args.stage] - int((previous or {}).get("stage_requested_decisions", {}).get(args.stage, 0))
        return train_semantic(runner, env, run_dir=args.run_dir, output_root=OUTPUT_ROOT,
                              stage=args.stage, decisions=remaining if args.decisions is None else args.decisions,
                              contract=contract, seed=args.seed, resume_infos=previous,
                              checkpoint_interval_updates=args.checkpoint_interval_updates)
    finally:
        app.close(wait_for_replicator=False, skip_cleanup=True)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    validate_request(args)
    contract = runtime_contract(expected_head=args.expected_head)
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
        if runtime_contract(expected_head=args.expected_head) != contract:
            raise RuntimeError("semantic runtime bytes changed during the run")
        lifecycle.update(lifecycle="SUCCEEDED", result=result, completed_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(args.run_dir / "run_manifest.json", lifecycle)
        print(json.dumps({"semantic_run": str(args.run_dir), "lifecycle": "SUCCEEDED"}), flush=True)
        return 0
    except BaseException as exc:
        lifecycle.update(lifecycle="FAILED", error=str(exc), traceback=traceback.format_exc(),
                         completed_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(args.run_dir / "run_manifest.json", lifecycle)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
