"""Fail-closed post-hoc provenance for receiving-wheel media/receipts.

This output-only draft does not authorize training, checkpoint migration, or
media export.  Formal use deliberately depends on the deployed production
``validate_migration_plan`` implementation; the current pre-deployment runtime
therefore cannot pass :func:`validate_checkpoint`.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping


HERE = Path(__file__).resolve().parent
PROFILE = HERE.parent
OUT = PROFILE.parents[1]
ROOT = OUT.parents[1]
SOURCE_POLICY = "task_conditioned_hip_wheel_sigma_v1"
TARGET_POLICY = "task_conditioned_receiving_wheel_sigma_x3_v1"
FACTOR_SCHEMA = "wlr50_clean.receiving_wheel_sigma_same372.v1"
SIGMA_SEMANTICS = (
    "task_conditioned_B_over_cap_then_P10_P12_RR_placed_history_"
    "FR_RR_wheel_sigma_x3_same372_v1"
)
MIGRATION_KEY = "receiving_wheel_sigma_migration"
AUX_BRANCH = "task_conditioned_hip_wheel_branch"
AUX_LEDGER = "auxiliary_mean_learning"
COUNTERS = ("global_policy_decisions", "ppo_updates", "optimizer_steps")
FALSE_CLAIMS = (
    "reward_changed", "task_acceptance_changed", "nominal_control_changed",
    "action_execution_changed", "physical_scene_changed",
    "actuator_capability_changed", "action_ranges_changed",
    "observation_layout_changed", "new_mdp", "physical_mdp_changed",
    "training_quantity_only", "lifetime_stage_budget_counters_reset",
    "deterministic_mean_changed", "history_kernel_changed",
    "physical_success_claimed",
)


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


def _factor(verified: Mapping[str, Any]) -> Mapping[str, Any]:
    factor = verified.get("receiving_wheel_sigma_factor")
    require(isinstance(factor, Mapping), "official plan has no receiving-wheel sigma factor")
    require(all(value is None for key, value in verified.items()
                if key.endswith("_factor") and key != "receiving_wheel_sigma_factor"),
            "receiving-wheel plan mixes another migration factor")
    return factor


def validate_exact_boundary(source: Mapping[str, Any], target: Mapping[str, Any],
                            verified: Mapping[str, Any]) -> dict[str, Any]:
    """Validate only the exact officially reconstructed sigma-only boundary."""
    factor = _factor(verified)
    gate = factor.get("receiving_continuation_gate")
    expected_gate = {
        "phase_indices": [9, 10, 11],
        "RR_placed_history_observation_index": 157,
        "RR_placed_history_value": 1,
        "channel_indices": [9, 11],
        "channel_names": ["FR_wheel", "RR_wheel"],
        "conditional_sigma_multiplier": 3.0,
        "otherwise_multiplier": 1.0,
        "current_RR_support_required": False,
        "current_RR_lift_required": False,
        "P13_unchanged": True,
    }
    require(factor.get("schema") == FACTOR_SCHEMA
            and factor.get("source_policy_version") == SOURCE_POLICY
            and factor.get("target_policy_version") == TARGET_POLICY
            and factor.get("sigma_scaling_semantics") == SIGMA_SEMANTICS
            and gate == expected_gate,
            "factor is not the exact P10-P12/RR-history157/FR-RR-wheel sigma-x3 profile")
    observation = factor.get("observation_contract") or {}
    require(observation.get("observation_dimension") == 372
            and observation.get("action_dimension") == 12
            and observation.get("num_envs") == 1
            and observation.get("parameter_mapping") == "identity_all_parameters_and_buffers",
            "sigma boundary changes the same372/full12/N1 parameter mapping")
    require(factor.get("kernel_changed") is True
            and factor.get("same_mdp_claimed") is True
            and factor.get("original_branch_origin_preserved") is True
            and factor.get("auxiliary_updates_added") == 0
            and all(factor.get(key) is False for key in FALSE_CLAIMS),
            "sigma boundary carries an unreviewed MDP/control/reward/mean/history claim")
    require(source.get("policy_contract") == factor.get("source_policy_contract")
            and target.get("policy_contract") == factor.get("target_policy_contract")
            and source.get("runner_config") == factor.get("source_runner_config")
            and target.get("runner_config") == factor.get("target_runner_config"),
            "checkpoint policy/runner contracts do not bind the official factor")
    require(source.get("runtime_contract", {}).get("experiment_id")
            == target.get("runtime_contract", {}).get("experiment_id")
            == "task_conditioned_hip_wheel_v1",
            "sigma boundary is outside the task-conditioned experiment")
    variable = {"files", "runtime_content_sha256", "source_git_commit"}
    old = source["runtime_contract"]
    new = target["runtime_contract"]
    require({k: v for k, v in old.items() if k not in variable}
            == {k: v for k, v in new.items() if k not in variable},
            "sigma boundary changes runtime/MDP/configuration/budget metadata")
    ledger = source.get(AUX_BRANCH, {}).get(AUX_LEDGER)
    require(isinstance(ledger, Mapping)
            and ledger.get("accepted_auxiliary_updates_total") == 7
            and ledger.get("attempted_auxiliary_optimizer_steps_total") == 8
            and target.get(AUX_BRANCH) == source.get(AUX_BRANCH)
            and factor.get("preserved_auxiliary_ledger_sha256") == digest(ledger),
            "reviewed AUX 7-accepted/8-attempted lineage was not preserved exactly")
    preserved = factor.get("preserved_training_state") or {}
    for key in ("actor_parameter_sha256", "critic_parameter_sha256",
                "optimizer_state_sha256", "normalizer_state_sha256", "training_rng_state"):
        require(preserved.get(key) == source.get(key),
                "sigma plan does not preserve source state: " + key)
    return {
        "schema": FACTOR_SCHEMA,
        "source_policy_version": SOURCE_POLICY,
        "target_policy_version": TARGET_POLICY,
        "sigma_scaling_semantics": SIGMA_SEMANTICS,
        "gate": expected_gate,
        "same_mdp": True,
        "stochastic_kernel_changed": True,
        "deterministic_mean_changed": False,
        "auxiliary_updates_added": 0,
        "factor_sha256": digest(factor),
    }


def validate_saved_record(current: Mapping[str, Any], source: Mapping[str, Any],
                          verified: Mapping[str, Any]) -> dict[str, Any]:
    """Bind the persisted checkpoint receipt to the official validator result."""
    saved = current.get(MIGRATION_KEY)
    factor = _factor(verified)
    expected = {
        "factor": dict(factor),
        "plan_path": verified["plan_path"],
        "plan_sha256": verified["plan_sha256"],
        "source_checkpoint_sha256": verified["source_checkpoint_sha256"],
        "source_contract_sha256": verified["source_contract_sha256"],
        "target_contract_sha256": verified["target_contract_sha256"],
    }
    require(saved == expected, "persisted receiving-wheel receipt is not the official plan result")
    require(source.get(MIGRATION_KEY) is None,
            "sigma source already contains a receiving-wheel migration receipt")
    require(source.get("checkpoint_sha256") == verified["source_checkpoint_sha256"],
            "official plan does not bind the historical source checkpoint")
    return validate_exact_boundary(source, current, verified)


def checkpoint_identity(path: Path, metadata: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(path).resolve(strict=True)
    sidecar = path.with_name(path.stem + "_manifest.json")
    return {
        "checkpoint": str(path),
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "manifest": str(sidecar),
        "manifest_sha256": sha(sidecar),
        "saved_global_policy_decisions": metadata["global_policy_decisions"],
        "actor_parameter_sha256": metadata["actor_parameter_sha256"],
        "task_conditioned_hip_wheel_branch_counts":
            metadata["task_conditioned_hip_wheel_branch_counts"],
        "policy_version": metadata["policy_contract"]["version"],
        "runtime_contract": metadata["runtime_contract"],
    }


def _official_module():
    production = (ROOT / "src").resolve()
    if str(production) not in sys.path:
        sys.path.insert(0, str(production))
    module = importlib.import_module("wlr50_clean.ppo.semantic_migration")
    require(Path(module.__file__).resolve().is_relative_to(production),
            "official migration validator was shadowed")
    require(getattr(module, "RECEIVING_WHEEL_SIGMA_SCHEMA", None) == FACTOR_SCHEMA,
            "receiving-wheel migration is not deployed in the production validator")
    return module


def _target_ancestry(checkpoint: Path, current: Mapping[str, Any], source_path: Path,
                     source: Mapping[str, Any], verified: Mapping[str, Any], official) -> list[dict[str, Any]]:
    """Follow every fresh-PPO target checkpoint back to the exact sigma source."""
    factor = _factor(verified)
    expected_saved = current[MIGRATION_KEY]
    cursor = Path(checkpoint).resolve()
    metadata = current
    lineage = []
    seen = set()
    for _ in range(64):
        require(cursor not in seen and cursor != source_path,
                "sigma target ancestry cycles or skips the migration boundary")
        seen.add(cursor)
        require(metadata.get(MIGRATION_KEY) == expected_saved
                and metadata.get("runtime_contract") == current.get("runtime_contract")
                and metadata.get("policy_contract") == factor.get("target_policy_contract")
                and metadata.get("runner_config") == factor.get("target_runner_config")
                and metadata.get(AUX_BRANCH) == source.get(AUX_BRANCH),
                "sigma descendant changed its migration, runtime, policy, runner or AUX branch")
        lineage.append({"checkpoint": str(cursor), "sha256": metadata["checkpoint_sha256"]})
        ancestry = metadata.get("resume_ancestry") or {}
        binding = ancestry.get("source_checkpoint") or {}
        parent_path = Path(binding.get("checkpoint", "")).resolve(strict=True)
        parent = official.checkpoint_metadata(parent_path)
        parent_sidecar = parent_path.with_name(parent_path.stem + "_manifest.json")
        expected_hop_migration = verified if parent_path == source_path else None
        require(binding == {
            "checkpoint": str(parent_path),
            "checkpoint_sha256": parent["checkpoint_sha256"],
            "manifest": str(parent_sidecar),
            "manifest_sha256": sha(parent_sidecar),
        } and ancestry.get("source_actor_parameter_sha256") == parent.get("actor_parameter_sha256")
          and ancestry.get("source_runtime_contract") == parent.get("runtime_contract")
          and all(ancestry.get("source_" + key) == parent.get(key) for key in COUNTERS)
          and ancestry.get("resume_migration") == expected_hop_migration,
          "sigma successor is not bound to its actual official predecessor and migration")
        updates = metadata["ppo_updates"] - parent["ppo_updates"]
        require(updates > 0
                and metadata["global_policy_decisions"] - parent["global_policy_decisions"] == 128 * updates
                and metadata["optimizer_steps"] - parent["optimizer_steps"] == 20 * updates
                and all(metadata["stage_requested_decisions"].get(key, -1) >= value
                        for key, value in parent["stage_requested_decisions"].items()),
                "sigma successor is not independent fresh N1 PPO")
        if parent_path == source_path:
            require(parent == source, "sigma ancestry source metadata changed")
            lineage.append({"checkpoint": str(parent_path),
                            "sha256": parent["checkpoint_sha256"]})
            return lineage
        cursor, metadata = parent_path, parent
    raise ValueError("sigma target ancestry exceeds the bounded 64-checkpoint audit")


def validate_checkpoint(checkpoint: Path, auxiliary_receipt: Path, *,
                        historical_runtime_root: Path,
                        project_root: Path = ROOT) -> dict[str, Any]:
    """Full CPU-only formal-media check; intentionally fails before deployment."""
    project_root = Path(project_root).resolve()
    require(project_root == ROOT.resolve(), "draft is bound to this exact output tree")
    checkpoint = Path(checkpoint).resolve(strict=True)
    official = _official_module()
    current = official.checkpoint_metadata(checkpoint)
    import torch
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    persisted = payload.get("infos")
    require(persisted == {k: v for k, v in current.items() if k not in
            ("checkpoint_path", "checkpoint_sha256", "save_load_round_trip")},
            "persisted checkpoint infos differ from the verified manifest")
    saved = current.get(MIGRATION_KEY)
    require(isinstance(saved, Mapping) and set(saved) == {
        "factor", "plan_path", "plan_sha256", "source_checkpoint_sha256",
        "source_contract_sha256", "target_contract_sha256",
    }, "checkpoint has no exact persisted receiving-wheel migration receipt")
    plan_path = Path(saved["plan_path"]).resolve(strict=True)
    require(sha(plan_path) == saved["plan_sha256"], "persisted sigma plan changed")
    plan = read(plan_path)
    source_path = Path(plan["source_checkpoint"]).resolve(strict=True)
    verified = official.validate_migration_plan(
        source_path, current["runtime_contract"], plan_path, project_root=project_root,
    )
    source = official.checkpoint_metadata(source_path)
    boundary = validate_saved_record(current, source, verified)
    target_lineage = _target_ancestry(
        checkpoint, current, source_path, source, verified, official,
    )
    from historical_posthoc import validate as validate_historical
    historical = validate_historical(source_path, Path(auxiliary_receipt),
        historical_runtime_root=Path(historical_runtime_root))
    auxiliary = historical["auxiliary_identity"]
    require(auxiliary.get("method") == "PPO_PLUS_LIMITED_AUX"
            and auxiliary.get("accepted_auxiliary_updates_total") == 7
            and auxiliary.get("attempted_auxiliary_optimizer_steps_total") == 8,
            "historical source does not carry the exact reviewed LIMITED AUX lineage")
    # The official source is the quantity-bound target.  Pairing must feed this
    # identity to the existing quantity-only gate; sigma is a second boundary.
    return {
        **auxiliary,
        "display_label": "PPO + LIMITED AUX",
        "exploration_profile": boundary,
        "receiving_wheel_sigma_migration": dict(saved),
        "historical_quantity_boundary": {
            key: value for key, value in historical.items()
            if key != "auxiliary_identity"
        },
        "sigma_source_checkpoint_identity": checkpoint_identity(source_path, source),
        "sigma_target_checkpoint_identity": checkpoint_identity(checkpoint, current),
        "verified_sigma_target_lineage": target_lineage,
        "comparison_contract": {
            "quantity_boundary_and_sigma_boundary_are_distinct": True,
            "same_N": 1,
            "same_physics_and_MDP": True,
            "same_nominal_mapper_evaluator_reward_control_chain": True,
            "same_policy_distribution": False,
            "deterministic_and_stochastic_C_must_use_same_checkpoint": True,
            "old_B_must_not_be_relabelled_as_target_sigma_profile": True,
        },
        "media_does_not_authorize_optimizer_or_migration": True,
        "task_success_not_inferred": True,
    }
