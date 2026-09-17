"""Explicit CPU-only candidate initializer; never run automatically after training.

Only use after the task-first continuation's saved/reloaded formal evaluation
still fails with the adverse mean. This is initialization, not learned success.
No Isaac imports, real observations, policy forward, rollout, or PPO update.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def initialize(checkpoint: Path, *, expected_head: str, branch_id: str, reason: str,
               artifact_suffix: str = "", supersedes_offline_checkpoint: Path | None = None) -> dict:
    import torch
    from wlr50_clean.ppo.semantic_cli import runtime_contract
    from wlr50_clean.ppo.semantic_migration import (
        apply_task_recovery_mean_head, checkpoint_metadata, file_sha,
    )
    from wlr50_clean.ppo.semantic_policy_distribution import policy_version_from_metadata
    from wlr50_clean.ppo.semantic_transfer_roles import ROLE_OBSERVATION_LAYOUT
    from wlr50_clean.ppo.semantic_training import (
        SemanticRslAdapter, construct_semantic_runner, load_semantic_checkpoint,
        save_semantic_checkpoint, seed_training_rngs, state_hash, write_json,
    )
    from wlr50_clean.ppo.rl_library_wrapper import capture_training_rng_state

    checkpoint = checkpoint.resolve(strict=True)
    history = ROOT / "outputs/ppo_task_first_recovery_v1/checkpoints/history"
    if not checkpoint.is_relative_to(history.resolve()):
        raise ValueError("candidate source must be the completed task-first continuation's immutable history checkpoint")
    metadata = checkpoint_metadata(checkpoint)
    contract = runtime_contract(expected_head=expected_head, semantic_version="v3",
                                experiment_id="task_first_recovery_v1")
    if contract != metadata["runtime_contract"]:
        raise ValueError("source and current task-first runtime differ; do not mix initialization with a code/reward migration")
    if metadata.get("task_recovery_branch") is not None:
        raise ValueError("this named recovery is not an unlimited repeated mean reset")
    # The helper validates the branch id before any output path is created.
    import re
    if re.fullmatch(r"[a-z][a-z0-9_]{2,63}", branch_id) is None:
        raise ValueError("invalid branch id")
    if artifact_suffix and re.fullmatch(r"[a-z][a-z0-9_]{2,63}", artifact_suffix) is None:
        raise ValueError("invalid offline artifact suffix")
    suffix = f"_{artifact_suffix}" if artifact_suffix else ""
    output = history / f"checkpoint_initial_{branch_id}_from_{metadata['global_policy_decisions']:09d}{suffix}.pt"
    receipt = ROOT / "outputs/ppo_task_first_recovery_v1" / f"{branch_id}_initialization{suffix}.json"
    if output.exists() or output.with_name(output.stem + "_manifest.json").exists() or receipt.exists():
        raise FileExistsError("immutable recovery candidate or receipt already exists; resume it instead")
    superseded = None
    if supersedes_offline_checkpoint is not None:
        rejected_path = supersedes_offline_checkpoint.resolve(strict=True)
        if not artifact_suffix or not rejected_path.is_relative_to(history.resolve()):
            raise ValueError("offline correction requires a new artifact suffix and an existing same-history initial candidate")
        rejected = checkpoint_metadata(rejected_path)
        rejected_branch = rejected.get("task_recovery_branch", {})
        if (rejected_branch.get("branch_id") != branch_id
                or rejected_branch.get("source_checkpoint", {}).get("checkpoint_sha256") != metadata["checkpoint_sha256"]
                or rejected_branch.get("counter_origin") != {key: metadata[key] for key in
                    ("global_policy_decisions", "ppo_updates", "optimizer_steps")}
                or rejected.get("task_recovery_branch_counts") != {key: 0 for key in
                    ("global_policy_decisions", "ppo_updates", "optimizer_steps")}
                or rejected.get("stage") != "initial_task_recovery_mean_head"):
            raise ValueError("offline correction cannot supersede a trained or different recovery strategy")
        superseded = {"path": str(rejected_path), "sha256": rejected["checkpoint_sha256"],
            "status": "REJECTED_OFFLINE_SERIALIZATION_NOT_FOR_TRAINING",
            "preserved_untouched": True, "physical_or_optimizer_steps_performed": 0,
            "reason": "CPU-only lazy CUDA seed callbacks overwrote queued restored Philox offset on first RNG capture",
            "same_recovery_strategy_not_an_additional_mean_reset_search": True}

    class NoPhysicsCore:
        def reset(self, *, seed=1001):
            return (0.0,) * 372

        def step(self, raw):
            raise RuntimeError("offline initialization cannot execute physics or collect training data")

    seed = int(metadata["seed"])
    seed_training_rngs(seed)
    # CPU model construction need not initialize CUDA. PyTorch defers manual
    # CUDA seeds separately from set_rng_state callbacks; at first lazy init a
    # pending seed can overwrite a previously queued restore. Materialize the
    # generator before loading the checkpoint, then the official loader's RNG
    # restore acts immediately. No GPU samples, tensor dynamics or Isaac here.
    if torch.cuda.is_available():
        torch.cuda.get_rng_state_all()
    env = SemanticRslAdapter(NoPhysicsCore(), seed=seed, device="cpu")
    env.cfg["semantic_version"] = "v3"
    runner, _ = construct_semantic_runner(env, seed=seed, device="cpu",
        policy_version=policy_version_from_metadata(metadata), initialize_actor=False,
        observation_layout=ROLE_OBSERVATION_LAYOUT)
    previous = load_semantic_checkpoint(runner, checkpoint, contract=contract, seed=seed)
    # Serialization device is CPU, not a new training configuration. Restore
    # the exact source runner specification in the saved metadata; next real
    # training/evaluation constructs its recorded device normally.
    source_config = copy.deepcopy(metadata["runner_config"])
    cpu_expected = copy.deepcopy(source_config)
    cpu_expected["device"] = "cpu"
    if runner._semantic_runner_config != cpu_expected:
        raise RuntimeError("offline reconstruction differs beyond serialization device")
    runner._semantic_runner_config = source_config
    runner.cfg["device"] = source_config["device"]
    original_rng = copy.deepcopy(previous["training_rng_state"])
    if capture_training_rng_state(seed=seed) != original_rng:
        raise RuntimeError("source RNG restore differs before initialization; refusing to create candidate")
    infos = apply_task_recovery_mean_head(runner, previous, branch_id=branch_id, reason=reason)
    infos["task_recovery_branch_counts"] = {key: 0 for key in ("global_policy_decisions", "ppo_updates", "optimizer_steps")}
    infos["offline_initialization"] = {
        "execution_device": "cpu", "source_training_device": source_config["device"],
        "source_training_runner_config_preserved": True,
        "synthetic_zero_observation_used_for_model_construction_only": True,
        "physics_steps": 0, "policy_forwards": 0, "optimizer_steps": 0,
        "does_not_update_last_or_best_checkpoint_pointer": True,
        "cuda_generator_materialized_before_checkpoint_rng_restore": bool(torch.cuda.is_available()),
        "all_rng_exact_after_restore_and_before_save": True,
    }
    if capture_training_rng_state(seed=seed) != original_rng:
        raise RuntimeError("mean-head transformation changed RNG before save; refusing to create candidate")
    if superseded is not None:
        infos["offline_serialization_correction"] = superseded
    saved, sidecar = save_semantic_checkpoint(runner, output, infos)
    published = checkpoint_metadata(saved)
    if published["training_rng_state"] != original_rng:
        raise RuntimeError("candidate initialization changed restored training RNG")
    if published["runner_config"] != metadata["runner_config"]:
        raise RuntimeError("candidate serialization changed training configuration")
    if superseded is not None:
        old_payload = torch.load(superseded["path"], map_location="cpu", weights_only=False)
        new_payload = torch.load(saved, map_location="cpu", weights_only=False)
        if state_hash({k:v for k,v in old_payload.items() if k != "infos"}) != state_hash(
                {k:v for k,v in new_payload.items() if k != "infos"}):
            raise RuntimeError("offline RNG correction changed actual model/Adam/iteration tensors")
        superseded["all_non_infos_checkpoint_payload_exactly_equal"] = True
    result = {"schema": "wlr50_clean.offline_mean_head_initialization_receipt.v1",
        "source_checkpoint": str(checkpoint), "source_sha256": metadata["checkpoint_sha256"],
        "candidate_checkpoint": str(saved), "candidate_manifest": str(sidecar),
        "candidate_sha256": published["checkpoint_sha256"],
        "task_recovery_branch": published["task_recovery_branch"],
        "task_recovery_branch_counts": published["task_recovery_branch_counts"],
        "offline_initialization": infos["offline_initialization"],
        "source_checkpoint_unchanged": file_sha(checkpoint) == metadata["checkpoint_sha256"],
        "actual_save_load_round_trip": published["save_load_round_trip"],
        "all_old_non_mean_states_preserved_except_recorded_mean_moments": True,
        "initialization_is_not_PPO_learning_or_task_success": True,
        "offline_serialization_correction": superseded,
    }
    write_json(receipt, result)
    return result


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--branch-id", default="mean_head_recovery_v1")
    parser.add_argument("--reason", required=True)
    parser.add_argument("--artifact-suffix", default="")
    parser.add_argument("--supersedes-offline-checkpoint", type=Path)
    parser.add_argument("--initialize-after-failed-task-first-evaluation", action="store_true", required=True)
    args = parser.parse_args()
    import json
    print(json.dumps(initialize(args.checkpoint, expected_head=args.expected_head,
        branch_id=args.branch_id, reason=args.reason, artifact_suffix=args.artifact_suffix,
        supersedes_offline_checkpoint=args.supersedes_offline_checkpoint), indent=2), flush=True)


if __name__ == "__main__":
    main()
