"""Explicit policy-architecture boundaries, independent of the physical MDP.

Metadata validation does not import torch or start native simulation. Tensor
mapping is a separate, side-effect-free helper for the two pinned RSL models.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .semantic_migration import (
    PROJECT_ROOT, _version_bytes, checkpoint_metadata, digest, file_sha,
    source_num_envs,
)

LEGACY_POLICY = "gaussian_scalar_v1"
STATE_DEPENDENT_POLICY = "heteroscedastic_log_v1"
POLICY_SCHEMA = "wlr50_clean.semantic_policy_distribution.v1"
MIGRATION_SCHEMA = "wlr50_clean.semantic_policy_distribution_migration.v1"
MODULE_PATH = "src/wlr50_clean/ppo/semantic_policy_distribution.py"
ALLOWED_CHANGED_FILES = frozenset({
    "src/wlr50_clean/ppo/semantic_training.py",
    "src/wlr50_clean/ppo/semantic_cli.py",
    "src/wlr50_clean/ppo/semantic_video_cli.py",
    "scripts/run_semantic_ppo.ps1",
    MODULE_PATH,
})
CONFIG_NAMES = frozenset({
    "stage_task_spec.yaml", "execution_profile.yaml", "reward_config.yaml",
    "observation_schema.json", "action_schema.json", "quality_score.yaml",
})
NORMALIZATION = "fixed_versioned_observation_schema; identity_RSL_normalizer"


def policy_contract(version: str) -> dict[str, Any]:
    if version not in (LEGACY_POLICY, STATE_DEPENDENT_POLICY):
        raise ValueError("unsupported semantic policy distribution version")
    dependent = version == STATE_DEPENDENT_POLICY
    return {
        "schema": POLICY_SCHEMA, "version": version,
        "distribution_class": ("HeteroscedasticGaussianDistribution" if dependent else "GaussianDistribution"),
        "std_type": "log" if dependent else "scalar",
        "observation_dimension": 324, "raw_action_dimension": 12,
        "actor_hidden_dims": [256, 256], "activation": "elu",
        "state_dependent_std": dependent, "normalization": NORMALIZATION,
        "raw_action_semantics": "unbounded_Gaussian_latent_before_existing_tanh_projection",
    }


def configure_policy_distribution(config: dict[str, Any], version: str) -> None:
    """Change exactly the distribution class and parameterization in place."""
    contract = policy_contract(version)
    distribution = config["actor"]["distribution_cfg"]
    if not isinstance(distribution, dict) or "init_std" not in distribution:
        raise ValueError("semantic actor configuration requires init_std")
    distribution["class_name"] = contract["distribution_class"]
    distribution["std_type"] = contract["std_type"]


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _same_json(left: Any, right: Any) -> bool:
    # Preserve the distinction between booleans and numeric configuration values.
    try:
        return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError):
        return False


def policy_version_from_metadata(metadata: Mapping[str, Any]) -> str:
    """Validate the complete pinned runner config, including legacy metadata."""
    from .semantic_training import semantic_runner_config

    if not isinstance(metadata, Mapping):
        raise ValueError("policy metadata must be a mapping")
    config = metadata.get("runner_config")
    if not isinstance(config, dict):
        raise ValueError("policy metadata requires the complete runner_config")
    seed = _integer(metadata.get("seed"), "policy seed")
    semantic_version = metadata.get("semantic_version", "v2")
    if semantic_version not in ("v2", "v3"):
        raise ValueError("unsupported checkpoint semantic version")
    try:
        distribution = config["actor"]["distribution_cfg"]
        pair = (distribution["class_name"], distribution["std_type"])
        version = {
            ("GaussianDistribution", "scalar"): LEGACY_POLICY,
            ("HeteroscedasticGaussianDistribution", "log"): STATE_DEPENDENT_POLICY,
        }[pair]
    except (KeyError, TypeError) as error:
        raise ValueError("checkpoint has an unsupported policy distribution") from error
    declared = metadata.get("policy_contract")
    if declared is None:
        if "policy_contract" in metadata or version != LEGACY_POLICY:
            raise ValueError("heteroscedastic checkpoint requires an explicit policy_contract")
    elif not _same_json(declared, policy_contract(version)):
        raise ValueError("checkpoint policy_contract disagrees with its distribution")
    if "policy_version" in metadata and metadata["policy_version"] != version:
        raise ValueError("checkpoint policy_version disagrees with its distribution")
    device = config.get("device")
    if device not in ("cpu", "cuda:0"):
        raise ValueError("checkpoint runner_config has an unsupported device")
    expected = semantic_runner_config(seed=seed, device=device,
                                     semantic_version=semantic_version, policy_version=version)
    if not _same_json(config, expected):
        raise ValueError("checkpoint complete runner_config differs from the pinned semantic policy configuration")
    return version


def _sha(value: Any, label: str, *, length: int = 64) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{%d}" % length, value) is None:
        raise ValueError(f"{label} must be a lowercase SHA digest")
    return value


def _relative(value: Any) -> str:
    if (not isinstance(value, str) or not value or "\\" in value or ":" in value
            or PurePosixPath(value).is_absolute() or ".." in PurePosixPath(value).parts
            or PurePosixPath(value).as_posix() != value):
        raise ValueError("runtime inventory contains an unsafe relative path")
    return value


def _runtime_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("policy migration requires a runtime contract")
    result = copy.deepcopy(dict(value))
    required = {"schema", "files", "source_git_commit", "runtime_content_sha256", "frozen_A_files",
                "physics_hz", "decision_hz", "task_timeout_s", "timeout_bootstrap", "training_budgets",
                "rsl_rl_version", "local_runtime_versions", "semantic_version", "selected_configuration"}
    if not required <= result.keys():
        raise ValueError("policy migration runtime contract is incomplete")
    if (result["schema"] != "wlr50_clean.semantic_runtime_contract.v1" or result["semantic_version"] != "v3"
            or result["rsl_rl_version"] != "5.0.1"):
        raise ValueError("policy migration requires the pinned v3/RSL 5.0.1 runtime")
    _sha(result["source_git_commit"], "runtime source commit", length=40)
    files = result["files"]
    if not isinstance(files, dict) or not files:
        raise ValueError("runtime inventory is empty")
    for path, sha in files.items():
        _relative(path)
        _sha(sha, "runtime file SHA")
    if result["runtime_content_sha256"] != digest(files):
        raise ValueError("runtime inventory digest mismatch")
    frozen = result["frozen_A_files"]
    if not isinstance(frozen, dict) or not frozen or any(files.get(path) != sha for path, sha in frozen.items()):
        raise ValueError("frozen A inventory is missing or inconsistent")
    selected = result["selected_configuration"]
    if not isinstance(selected, dict) or set(selected) != CONFIG_NAMES:
        raise ValueError("policy migration requires all six unchanged v3 configurations")
    for name, row in selected.items():
        path = f"configs/ppo_semantic_v3/{name}"
        if not _same_json(row, {"path": path, "sha256": files.get(path)}) or path not in files:
            raise ValueError("selected configuration differs from runtime inventory")
    return result


def build_policy_distribution_migration(checkpoint: Path, current_contract: Mapping[str, Any], *,
                                        project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Bind a v3 Gaussian-to-heteroscedastic conversion; never a new MDP."""
    checkpoint = Path(checkpoint).resolve(strict=True)
    metadata = checkpoint_metadata(checkpoint)
    if metadata.get("semantic_version") != "v3" or policy_version_from_metadata(metadata) != LEGACY_POLICY:
        raise ValueError("policy migration only accepts a v3 legacy Gaussian source")
    if source_num_envs(metadata) != 1:
        raise ValueError("policy migration requires the existing N1 topology")
    old = _runtime_contract(metadata.get("runtime_contract"))
    new = _runtime_contract(current_contract)
    variable = {"files", "source_git_commit", "runtime_content_sha256"}
    if not _same_json({k: v for k, v in old.items() if k not in variable},
                      {k: v for k, v in new.items() if k not in variable}):
        raise ValueError("policy-only migration cannot change physical/configuration/runtime metadata")
    delta = sorted(path for path in old["files"].keys() | new["files"].keys()
                   if old["files"].get(path) != new["files"].get(path))
    if not set(delta) <= ALLOWED_CHANGED_FILES:
        raise ValueError("policy migration changed a non-whitelisted physical/reward/runtime file")
    if any(path not in new["files"] for path in old["files"]):
        raise ValueError("policy migration cannot delete runtime files")
    if set(new["files"]) - set(old["files"]) - {MODULE_PATH}:
        raise ValueError("only the policy distribution module may be newly added")
    if MODULE_PATH not in new["files"]:
        raise ValueError("target inventory does not bind the policy distribution implementation")
    root = Path(project_root).resolve(strict=True)
    for relative, expected in new["files"].items():
        target = (root / relative).resolve(strict=True)
        if not target.is_relative_to(root) or not target.is_file() or file_sha(target) != expected:
            raise ValueError(f"target runtime bytes differ from inventory: {relative}")
    for relative in delta:
        if relative in old["files"]:
            _version_bytes(root, old, relative)
        else:
            historical = subprocess.run(["git", "-C", str(root), "ls-tree", "-r", "--name-only",
                                         old["source_git_commit"], "--", relative],
                                        check=True, capture_output=True, text=True)
            if historical.stdout.strip():
                raise ValueError("new module was already present but omitted from the source inventory")
        _version_bytes(root, new, relative)
    seed = _integer(metadata.get("seed"), "source seed")
    lifetime = {key: _integer(metadata.get(key), f"source {key}") for key in
                ("global_policy_decisions", "ppo_updates", "optimizer_steps")}
    origin = _integer(metadata.get("new_mdp_origin_global_policy_decisions"), "original v3 origin")
    budgets = old["training_budgets"]
    spent = metadata.get("stage_requested_decisions")
    if (not isinstance(spent, dict) or set(spent) != {"smoke", "phase_suffix", "full_episode"}
            or not isinstance(budgets, dict) or set(budgets) != set(spent)
            or any(type(v) is not int or v < 0 or type(budgets[k]) is not int or v > budgets[k]
                   for k, v in spent.items())
            or origin > lifetime["global_policy_decisions"]
            or sum(spent.values()) > lifetime["global_policy_decisions"] - origin):
        raise ValueError("policy migration must preserve valid existing v3 budget accounting")
    lr = metadata.get("optimizer_learning_rate")
    if type(lr) not in (int, float) or not math.isfinite(lr) or lr <= 0:
        raise ValueError("source effective optimizer learning rate is invalid")
    rng = metadata.get("training_rng_state")
    if (not isinstance(rng, dict) or rng.get("schema") != "wlr50_clean.training_rng_state.v1"
            or type(rng.get("seed")) is not int or rng["seed"] != seed):
        raise ValueError("source training RNG metadata is missing or inconsistent")
    if metadata.get("normalization") != NORMALIZATION:
        raise ValueError("policy migration requires the existing identity normalization")
    source_hashes = {key: _sha(metadata.get(key), key) for key in
                     ("actor_parameter_sha256", "critic_parameter_sha256", "optimizer_state_sha256",
                      "normalizer_state_sha256")}
    sidecar = checkpoint.with_name(checkpoint.stem + "_manifest.json")
    return {
        "schema": MIGRATION_SCHEMA, "physical_mdp_changed": False, "reward_changed": False,
        "policy_architecture_changed": True, "exact_optimizer_resume": False,
        "source_checkpoint": str(checkpoint), "source_checkpoint_sha256": metadata["checkpoint_sha256"],
        "source_manifest_sha256": file_sha(sidecar), "source_semantic_version": "v3", "source_seed": seed,
        "source_runtime_contract": old, "target_runtime_contract": new,
        "source_policy_version": LEGACY_POLICY, "target_policy_version": STATE_DEPENDENT_POLICY,
        "source_policy_contract": policy_contract(LEGACY_POLICY),
        "target_policy_contract": policy_contract(STATE_DEPENDENT_POLICY),
        **{f"source_{k}": v for k, v in lifetime.items()},
        "source_stage_requested_decisions": dict(spent), "target_stage_requested_decisions": dict(spent),
        "new_mdp_origin_global_policy_decisions": origin, "source_state_hashes": source_hashes,
        "runtime_changed_files": delta,
        "changed_file_hashes": {path: {"before": old["files"].get(path), "after": new["files"][path]} for path in delta},
        "optimizer": {"kind": "Adam", "state": "fresh_Adam_preserve_source_effective_lr",
                      "initial_learning_rate": float(lr), "preserve_source_effective_lr": True,
                      "old_moments_inherited": False,
                      "reason": "explicit policy-parameterization boundary; physical MDP and reward unchanged"},
        "rng": "restore_verified_source_training_RNG_after_construction_and_mapping",
        "source_training_rng_state_sha256": digest(rng),
        "old_rollout_buffer_inherited": False, "physical_state_inherited": False,
        "migration_added_policy_decisions": 0, "migration_added_ppo_updates": 0,
        "migration_added_optimizer_steps": 0,
    }


def policy_migration_checkpoint_name(record: Mapping[str, Any]) -> str:
    if (record.get("schema") != MIGRATION_SCHEMA
            or not _same_json(record.get("target_policy_contract"), policy_contract(STATE_DEPENDENT_POLICY))):
        raise ValueError("invalid policy distribution migration record")
    source = _sha(record.get("source_checkpoint_sha256"), "source checkpoint SHA")
    step = _integer(record.get("source_global_policy_decisions"), "source global decisions")
    target = _runtime_contract(record.get("target_runtime_contract"))
    return (f"checkpoint_initial_policy_from_{step:09d}_s{source[:12]}"
            f"_g{target['source_git_commit'][:12]}_{target['runtime_content_sha256']}"
            f"_p{digest(record['target_policy_contract'])[:12]}.pt")


def map_gaussian_actor_state(source_state: Mapping[str, Any], target_state: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a new strict state dict; never mutate either supplied model/state."""
    import torch

    shapes = {"mlp.0.weight": (256, 324), "mlp.0.bias": (256,),
              "mlp.2.weight": (256, 256), "mlp.2.bias": (256,),
              "mlp.4.weight": (12, 256), "mlp.4.bias": (12,)}
    if (set(source_state) != set(shapes) | {"distribution.std_param"}
            or set(target_state) != set(shapes)):
        raise ValueError("actor state keys do not match the pinned Gaussian/heteroscedastic models")
    for state, expected_shapes in (
            (source_state, {**shapes, "distribution.std_param": (12,)}),
            (target_state, {**shapes, "mlp.4.weight": (24, 256), "mlp.4.bias": (24,)})):
        for key, shape in expected_shapes.items():
            value = state[key]
            if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                    or value.dtype != torch.float32 or not bool(torch.isfinite(value).all())):
                raise ValueError(f"actor tensor has invalid shape/dtype/finiteness: {key}")
    sigma = source_state["distribution.std_param"].detach()
    if not bool((sigma > 0).all()):
        raise ValueError("source learned sigma must be finite and strictly positive without clipping")
    mapped = {}
    for key in shapes:
        destination = target_state[key]
        source = source_state[key].detach().to(device=destination.device)
        if key.startswith("mlp.4."):
            value = torch.zeros_like(destination)
            value[:12].copy_(source)
            if key.endswith("bias"):
                value[12:].copy_(sigma.to(device=destination.device).log())
        else:
            value = source.clone()
        mapped[key] = value
    recovered_sigma = mapped["mlp.4.bias"][12:].exp()
    if not bool(torch.isfinite(recovered_sigma).all()) or not bool((recovered_sigma > 0).all()):
        raise ValueError("source learned sigma cannot be represented by the target log-std head")

    def tensor_sha(value: Any) -> str:
        cpu = value.detach().cpu().contiguous()
        head = json.dumps({"shape": list(cpu.shape), "dtype": str(cpu.dtype)}, sort_keys=True).encode()
        return hashlib.sha256(head + cpu.numpy().tobytes()).hexdigest()

    preserved = {}
    for key in shapes:
        target = mapped[key][:12] if key.startswith("mlp.4.") else mapped[key]
        if not torch.equal(source_state[key].detach().cpu(), target.detach().cpu()):
            raise RuntimeError("mapped actor failed exact retained-weight verification")
        preserved[key] = {"source_sha256": tensor_sha(source_state[key]), "target_retained_sha256": tensor_sha(target)}
    evidence = {
        "schema": "wlr50_clean.semantic_policy_tensor_mapping.v1",
        "source_policy_contract": policy_contract(LEGACY_POLICY),
        "target_policy_contract": policy_contract(STATE_DEPENDENT_POLICY),
        "preserved_hidden_and_mean_weights_exact": True, "preserved_tensors": preserved,
        "source_sigma_full12": sigma.cpu().tolist(),
        "target_initial_sigma_full12": recovered_sigma.detach().cpu().tolist(),
        "target_log_std_bias_full12": mapped["mlp.4.bias"][12:].detach().cpu().tolist(),
        "target_std_head_weights_zero": bool(torch.count_nonzero(mapped["mlp.4.weight"][12:]) == 0),
        "removed_source_keys": ["distribution.std_param"], "sigma_clipped": False,
        "input_states_mutated": False, "optimizer_steps": 0,
    }
    return mapped, evidence
