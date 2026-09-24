"""Exact learned422 -> cooperative-prep422 full-state continuation.

The migration changes reviewed runtime/reward telemetry and the conditional
sigma law, not learned tensors, optimizer state, RNG, normalization, physical
dynamics, actuator targets, HISTORY, or the observation layout.  Its source is
the completed d7e97ee course checkpoint, never a requested budget or pointer.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from .semantic_p02_progress_profile import (
    P02_PROGRESS_POLICY, P02_PROGRESS_OBSERVATION_LAYOUT,
    p02_progress_policy_contract,
)
from .semantic_rear_cooperative_prep_profile import (
    COOPERATIVE_PREP_POLICY, COOPERATIVE_PREP_OBSERVATION_LAYOUT,
    cooperative_prep_policy_contract,
)
from .semantic_rear_policy_timing_migration import (
    COUNTERS, EXPERIMENT, SOURCE_COUNTS, preserved_keys,
)
from .semantic_rear_recapture_migration import BRANCH_NAME
from .semantic_cooperative_preparation import MODE as COOPERATIVE_PREPARATION_MODE


SCHEMA = "wlr50_clean.cooperative_prep_same422.v4"
FACTOR_KEY = "cooperative_prep_same422_factor"
MIGRATION = "cooperative_prep_migration"
SOURCE_HEAD = "d7e97ee7b7e493d4f3ff34f9c8762f73550ad7bd"
SOURCE_SHA = "2739173651e516ab19a5213e6d3206f7c970e7d76ed0f3adbde1bc95b7dc86c7"
SOURCE_MANIFEST_SHA = "c6e0cd978202509cf73d41e5814fe9ad59051a6e3c9f1f14d5f689c8b7c4b550"
REVISION_ORIGIN = dict(zip(COUNTERS, (223232, 1709, 34180)))

# Filled only after root freezes and reviews the exact target runtime. Code is
# authorized by a closed path set and checked against contract/on-disk bytes;
# embedding this module's own target SHA here would create a self-reference.
REVIEWED_CODE_PATHS = frozenset({
    "src/wlr50_clean/ppo/semantic_backend.py",
    "src/wlr50_clean/ppo/semantic_checkpoint_prefix_policy.py",
    "src/wlr50_clean/ppo/semantic_cli.py",
    "src/wlr50_clean/ppo/semantic_cooperative_prep_migration.py",
    "src/wlr50_clean/ppo/semantic_cooperative_preparation.py",
    "src/wlr50_clean/ppo/semantic_height_diagnostics.py",
    "src/wlr50_clean/ppo/semantic_hip_mount_geometry.py",
    "src/wlr50_clean/ppo/semantic_migration.py",
    "src/wlr50_clean/ppo/semantic_policy_distribution.py",
    "src/wlr50_clean/ppo/semantic_rear_cooperative_prep_actor.py",
    "src/wlr50_clean/ppo/semantic_rear_cooperative_prep_profile.py",
    "src/wlr50_clean/ppo/semantic_rear_cooperative_prep_sigma.py",
    "src/wlr50_clean/ppo/semantic_rear_policy_timing_migration.py",
    "src/wlr50_clean/ppo/semantic_reward.py",
    "src/wlr50_clean/ppo/semantic_supervisor.py",
    "src/wlr50_clean/ppo/semantic_training.py",
})
REVIEWED_CONFIG_HASHES = {
    "configs/ppo_rr_rl_timing_policy_learning_v1/execution_profile.yaml": {
        "before": "59cba4e244033e1e09dfccbe4581d3c33867ca9279d895811ee9af64cc02484d",
        "after": "ea1b307e80b07788d7225f5ecade26cf651e3c7c07aadcff19f7dbaa60ff6c3f",
    },
    "configs/ppo_rr_rl_timing_policy_learning_v1/stage_task_spec.yaml": {
        "before": "c26b52503fffc80e43e0f02ed7dd25d75143a1314da58a15919386d16425d2bb",
        "after": "a240de4afc2e578ec8647f8825089aa689b5a14d271f37b3b796353db5572983",
    },
    "configs/ppo_rr_rl_timing_policy_learning_v1/reward_config.yaml": {
        "before": "681f42fabe639edae78746c1fcdb9f7555bd73ac1634891b41bcb548e0d03aba",
        "after": "598905a74059114df5ea80d524a89ad28494693697ae1880df122f4bd1f30c08",
    },
}


def _source_policy():
    return p02_progress_policy_contract(
        observation_layout=P02_PROGRESS_OBSERVATION_LAYOUT)


def _target_policy():
    return cooperative_prep_policy_contract(
        observation_layout=COOPERATIVE_PREP_OBSERVATION_LAYOUT)


def _preserved(metadata):
    """Full source identity checked during the one-time migration load."""
    return tuple(sorted(set(preserved_keys(metadata)) | {
        "runner_config", "policy_contract", "actor_parameter_sha256",
        "critic_parameter_sha256", "optimizer_state_sha256",
        "checkpoint_output_routing", "p02_progress_migration",
    }))


def _runner_configs(metadata):
    """Reconstruct the exact source/target PPO configs around this boundary."""
    from .semantic_return_profile import runner_return_profile
    from .semantic_training import semantic_runner_config
    source = metadata["runner_config"]
    horizon = runner_return_profile(source, semantic_version="v3")["version"]
    common = dict(seed=metadata["seed"], device=source["device"],
                  semantic_version="v3", return_profile=horizon)
    old = semantic_runner_config(
        **common, policy_version=P02_PROGRESS_POLICY,
        observation_layout=P02_PROGRESS_OBSERVATION_LAYOUT)
    new = semantic_runner_config(
        **common, policy_version=COOPERATIVE_PREP_POLICY,
        observation_layout=COOPERATIVE_PREP_OBSERVATION_LAYOUT)
    return old, new


def _reviewed_runtime_delta(old, new, root):
    import yaml
    from .semantic_cooperative_preparation import MODE as PREPARATION_MODE, WORKSPACE_WEIGHTS
    from .semantic_migration import _version_bytes, file_sha
    code = REVIEWED_CODE_PATHS
    if not isinstance(code, frozenset) or not code:
        raise ValueError("cooperative-prep code path review is not frozen")
    expected = code | frozenset(REVIEWED_CONFIG_HASHES)
    variable = {"files", "source_git_commit", "runtime_content_sha256",
                "selected_configuration"}
    if ({key: value for key, value in old.items() if key not in variable}
            != {key: value for key, value in new.items() if key not in variable}):
        raise ValueError("runtime metadata changed beyond the reviewed cooperative-prep bytes")
    changed = sorted(
        path for path in set(old["files"]) | set(new["files"])
        if old["files"].get(path) != new["files"].get(path))
    if set(changed) != set(expected) or any(path not in new["files"] for path in changed):
        raise ValueError("runtime changed outside the exact cooperative-prep review")
    for path, wanted in new["files"].items():
        if file_sha(root/path) != wanted:
            raise ValueError("target runtime bytes differ from inventory: " + path)
    result = {}
    for path in changed:
        pair = {"before": old["files"].get(path), "after": new["files"].get(path)}
        if pair["after"] is None or file_sha(root/path) != pair["after"]:
            raise ValueError("reviewed cooperative-prep bytes differ: " + path)
        if path in REVIEWED_CONFIG_HASHES and pair != REVIEWED_CONFIG_HASHES[path]:
            raise ValueError("reviewed cooperative-prep config hash differs: " + path)
        result[path] = dict(pair)
    if set(old["selected_configuration"]) != set(new["selected_configuration"]):
        raise ValueError("configuration inventory changed")
    previous = {}
    for name, target in new["selected_configuration"].items():
        source = old["selected_configuration"][name]
        if (source["path"] != target["path"]
                or file_sha(root/target["path"]) != target["sha256"]
                or old["files"].get(source["path"]) != source["sha256"]
                or new["files"].get(target["path"]) != target["sha256"]):
            raise ValueError("configuration binding differs: " + name)
        if source != target:
            previous[name] = copy.deepcopy(source)
    by_path = {entry["path"]: name for name, entry in new["selected_configuration"].items()}
    if not set(REVIEWED_CONFIG_HASHES) <= set(by_path):
        raise ValueError("reviewed configs are absent from selected configuration")
    for path in REVIEWED_CONFIG_HASHES:
        name = by_path[path]
        before = yaml.safe_load(_version_bytes(root, old, path))
        after = yaml.safe_load((root/path).read_bytes())
        if name in ("execution_profile.yaml", "stage_task_spec.yaml"):
            if before.get("cooperative_preparation_mode") is not None:
                raise ValueError("source unexpectedly has cooperative preparation mode")
            if after.pop("cooperative_preparation_mode", None) != PREPARATION_MODE:
                raise ValueError("target cooperative preparation mode differs")
        elif name == "reward_config.yaml":
            wanted = {"mode": PREPARATION_MODE, "counterroll_cost_per_s": .01,
                "force_noise_floor_n": .2, "clearance_scale_m": .02,
                "workspace_weights": dict(WORKSPACE_WEIGHTS)}
            if before.get("cooperative_preparation") is not None:
                raise ValueError("source unexpectedly has cooperative preparation reward")
            if after.pop("cooperative_preparation", None) != wanted:
                raise ValueError("target cooperative preparation reward differs")
        else:
            raise ValueError("unexpected reviewed cooperative-prep config: " + name)
        if before != after:
            raise ValueError("configuration changed beyond cooperative preparation: " + name)
    return result, previous


def _previous_contract(current, receipt):
    """Reverse only this receipt, yielding the exact historical d7e contract."""
    from .semantic_migration import digest
    previous = copy.deepcopy(current)
    previous["source_git_commit"] = receipt["source_git_commit"]
    previous["runtime_content_sha256"] = receipt["source_runtime_content_sha256"]
    for path, hashes in receipt["changed_file_hashes"].items():
        if previous["files"].get(path) != hashes["after"]:
            raise ValueError("current runtime differs from cooperative-prep receipt: " + path)
        if hashes["before"] is None:
            previous["files"].pop(path)
        else:
            previous["files"][path] = hashes["before"]
    previous["selected_configuration"].update(receipt["source_changed_configuration"])
    if digest(previous) != receipt["source_contract_sha256"]:
        raise ValueError("historical P02-progress runtime reconstruction is not exact")
    return previous


def build_cooperative_prep_migration(checkpoint, current_contract, *, reason,
                                     project_root=None):
    from .semantic_migration import (
        _contract, checkpoint_metadata, digest, file_sha, source_num_envs,
    )
    from .semantic_p02_progress_migration import validate_p02_progress_lineage
    root = Path(project_root or Path(__file__).resolve().parents[3])
    checkpoint = Path(checkpoint).resolve(strict=True)
    sidecar = checkpoint.with_name(checkpoint.stem + "_manifest.json")
    branch = root / "outputs" / ("ppo_" + EXPERIMENT) / "branches" / BRANCH_NAME
    if (checkpoint.parent != (branch / "checkpoints" / "history").resolve()
            or file_sha(checkpoint) != SOURCE_SHA
            or file_sha(sidecar) != SOURCE_MANIFEST_SHA):
        raise ValueError("requires exact completed d7e learned422 course checkpoint")
    metadata = checkpoint_metadata(checkpoint)
    old, new = _contract(metadata["runtime_contract"]), _contract(current_contract)
    route = metadata.get("checkpoint_output_routing")
    # Validate the real old policy/runtime first.  Never substitute the target
    # cooperative contract into the immutable P02 receipt.
    validate_p02_progress_lineage(
        metadata, old, branch, checkpoint_output_routing=route)
    source_policy, target_policy = _source_policy(), _target_policy()
    source_runner, target_runner = _runner_configs(metadata)
    if (old.get("source_git_commit") != SOURCE_HEAD
            or new.get("source_git_commit") == SOURCE_HEAD
            or old.get("experiment_id") != EXPERIMENT
            or new.get("experiment_id") != EXPERIMENT
            or not isinstance(reason, str) or not reason.strip()
            or MIGRATION in metadata
            or {key: metadata.get(key) for key in COUNTERS} != REVISION_ORIGIN
            or metadata.get("policy_contract") != source_policy
            or metadata.get("runner_config") != source_runner
            or metadata.get("save_load_round_trip") is not True
            or source_num_envs(metadata) != 1):
        raise ValueError("sealed learned422 source/runtime/policy/counts are not intact")
    changed, source_configs = _reviewed_runtime_delta(old, new, root)
    factor = {
        "schema": SCHEMA,
        "source_policy_contract": source_policy,
        "target_policy_contract": target_policy,
        "revision_counter_origin": REVISION_ORIGIN,
        "original_branch_origin": SOURCE_COUNTS,
        "source_effective_learning_rate": metadata["optimizer_learning_rate"],
        "source_runner_config": copy.deepcopy(source_runner),
        "target_runner_config": copy.deepcopy(target_runner),
        "preserved_metadata_sha256": {
            key: digest(metadata[key]) for key in _preserved(metadata)
        },
        "parameter_mapping": "identity_all_actor_critic_buffers_full_Adam_options_steps_and_rng",
        "observation_dimension": 422,
        "observation_layout": COOPERATIVE_PREP_OBSERVATION_LAYOUT,
        "cooperative_preparation_revision": COOPERATIVE_PREPARATION_MODE,
        "num_envs": 1,
        "same_mdp_claimed": False,
        "same_numeric_input_Gaussian_preserved": False,
        "deterministic_mean_and_learned_parameters_preserved": True,
        "conditional_sigma_semantics_changed": True,
        "bounded_preparation_reward_semantics_changed": True,
        "geometry_telemetry_added": True,
        "physical_dynamics_changed": False,
        "source_pose_commands_changed": False,
        "RR_assist_changed": False,
        "wheel_shaping_changed": False,
        "HISTORY_changed": False,
        "learning_rate_changed": False,
        "normalizer_changed": False,
        "old_rollout_inherited": False,
        "added_policy_decisions": 0,
        "added_ppo_updates": 0,
        "added_optimizer_steps": 0,
        "added_auxiliary_updates": 0,
    }
    record = {
        "schema": SCHEMA,
        "reason": reason.strip(),
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": SOURCE_SHA,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA,
        "source_git_commit": old["source_git_commit"],
        "target_git_commit": new["source_git_commit"],
        "source_runtime_content_sha256": old["runtime_content_sha256"],
        "target_runtime_content_sha256": new["runtime_content_sha256"],
        "source_contract_sha256": digest(old),
        "target_contract_sha256": digest(new),
        "changed_file_hashes": changed,
        "source_changed_configuration": source_configs,
        "discard_old_rollout_storage": True,
        FACTOR_KEY: factor,
    }
    if _previous_contract(new, record) != old:
        raise ValueError("runtime changed outside reviewed cooperative-prep delta")
    return record


def validate_cooperative_prep_migration(checkpoint, current_contract, plan_path,
                                        *, project_root=None):
    from .semantic_migration import file_sha
    path = Path(plan_path).resolve(strict=True)
    record = json.loads(path.read_text(encoding="utf-8"))
    expected = build_cooperative_prep_migration(
        checkpoint, current_contract, reason=record.get("reason", ""),
        project_root=project_root)
    if record != expected:
        raise ValueError("cooperative-prep plan/source/runtime differs")
    return {**record, "plan_path": str(path), "plan_sha256": file_sha(path)}


def validate_cooperative_prep_lineage(metadata, contract, output_root,
                                      *, checkpoint_output_routing=None):
    from .semantic_migration import digest
    from .semantic_p02_progress_migration import validate_p02_progress_lineage
    receipt = metadata.get(MIGRATION, {})
    factor = receipt.get(FACTOR_KEY, {})
    route = metadata.get("checkpoint_output_routing")
    try:
        source_runner, target_runner = _runner_configs({
            "seed": metadata.get("seed"),
            "runner_config": factor.get("source_runner_config"),
        })
    except (KeyError, TypeError, ValueError):
        raise ValueError("cooperative-prep runner configuration is malformed") from None
    if (receipt.get("schema") != SCHEMA or factor.get("schema") != SCHEMA
            or receipt.get("source_checkpoint_sha256") != SOURCE_SHA
            or receipt.get("source_manifest_sha256") != SOURCE_MANIFEST_SHA
            or receipt.get("source_git_commit") != SOURCE_HEAD
            or receipt.get("target_git_commit") != contract.get("source_git_commit")
            or receipt.get("target_contract_sha256") != digest(contract)
            or receipt.get("target_runtime_content_sha256") != contract.get("runtime_content_sha256")
            or factor.get("revision_counter_origin") != REVISION_ORIGIN
            or factor.get("original_branch_origin") != SOURCE_COUNTS
            or factor.get("source_policy_contract") != _source_policy()
            or factor.get("target_policy_contract") != _target_policy()
            or factor.get("source_runner_config") != source_runner
            or factor.get("target_runner_config") != target_runner
            or metadata.get("runner_config") != target_runner
            or metadata.get("policy_contract") != factor.get("target_policy_contract")
            or factor.get("observation_dimension") != 422
            or factor.get("observation_layout") != COOPERATIVE_PREP_OBSERVATION_LAYOUT
            or factor.get("cooperative_preparation_revision")
                != COOPERATIVE_PREPARATION_MODE
            or factor.get("parameter_mapping")
                != "identity_all_actor_critic_buffers_full_Adam_options_steps_and_rng"
            or any(type(metadata.get(key)) is not int
                   or metadata[key] < REVISION_ORIGIN[key] for key in COUNTERS)
            or any(factor.get(key) != 0 for key in (
                "added_policy_decisions", "added_ppo_updates",
                "added_optimizer_steps", "added_auxiliary_updates"))
            or factor.get("preserved_metadata_sha256", {}).get("p02_progress_migration")
                != digest(metadata.get("p02_progress_migration"))
            or factor.get("preserved_metadata_sha256", {}).get("checkpoint_output_routing")
                != digest(route)
            or factor.get("preserved_metadata_sha256", {}).get("runner_config")
                != digest(source_runner)
            or factor.get("preserved_metadata_sha256", {}).get("policy_contract")
                != digest(factor.get("source_policy_contract"))):
        raise ValueError("cooperative-prep lineage/policy/counts changed")
    if checkpoint_output_routing is not None and checkpoint_output_routing != route:
        raise ValueError("cooperative-prep output route changed")
    old = _previous_contract(contract, receipt)
    historical = copy.deepcopy(dict(metadata))
    historical.pop(MIGRATION, None)
    historical["runtime_contract"] = old
    historical["policy_contract"] = copy.deepcopy(factor["source_policy_contract"])
    historical["runner_config"] = copy.deepcopy(factor["source_runner_config"])
    validate_p02_progress_lineage(
        historical, old, output_root,
        checkpoint_output_routing=checkpoint_output_routing)
    return receipt


def _verify_target_identity(runner, infos, factor):
    import torch
    from .semantic_migration import digest
    from .semantic_training import parameter_hash, state_hash, _normalizers
    from .rl_library_wrapper import optimizer_learning_rate, capture_training_rng_state
    actual = {
        "actor_parameter_sha256": parameter_hash(runner.alg.actor),
        "critic_parameter_sha256": parameter_hash(runner.alg.critic),
        "optimizer_state_sha256": state_hash(runner.alg.optimizer.state_dict()),
        "normalizer_state_sha256": state_hash(_normalizers(runner)),
        "training_rng_state": capture_training_rng_state(seed=infos["seed"]),
    }
    if (any(type(getattr(runner.alg, role).obs_normalizer) is not torch.nn.Identity
            for role in ("actor", "critic"))
            or runner._semantic_runner_config != factor["target_runner_config"]
            or infos.get("runner_config") != factor["target_runner_config"]
            or optimizer_learning_rate(runner) != factor["source_effective_learning_rate"]
            or runner.alg.learning_rate != factor["source_effective_learning_rate"]
            or any(infos.get(key) != value for key, value in actual.items())
            or runner.alg.storage.step != 0 or runner.alg.transition.actions is not None
            or tuple(runner.alg.storage.actions.shape) != (128, 1, 12)
            or any(tuple(runner.alg.storage.observations[key].shape) != (128, 1, 422)
                   for key in ("policy", "critic"))):
        raise RuntimeError("cooperative-prep full-state identity/fresh422 storage failed")
    source = factor["preserved_metadata_sha256"]
    for key in _preserved(infos):
        if (key not in ("runner_config", "policy_contract")
                and key in source and digest(infos[key]) != source[key]):
            raise RuntimeError("cooperative-prep changed source metadata: " + key)


def load_cooperative_prep_migration(runner, checkpoint, *, contract, seed, record):
    from .semantic_migration import checkpoint_metadata
    from .semantic_rr_capture_migration import _shape_env
    from .semantic_training import (
        _normalizers, _runner_policy_contract, construct_semantic_runner,
        load_semantic_checkpoint, parameter_hash, state_hash,
    )
    from .rl_library_wrapper import optimizer_learning_rate, restore_training_rng_state
    verified = validate_cooperative_prep_migration(
        checkpoint, contract, record["plan_path"])
    if verified != dict(record):
        raise RuntimeError("cooperative-prep plan changed after preflight")
    metadata = checkpoint_metadata(Path(checkpoint))
    factor = verified[FACTOR_KEY]
    if (seed != metadata["seed"]
            or _runner_policy_contract(runner) != factor["target_policy_contract"]
            or tuple(runner.alg.storage.actions.shape) != (128, 1, 12)
            or any(tuple(runner.alg.storage.observations[key].shape) != (128, 1, 422)
                   for key in ("policy", "critic"))
            or runner.alg.storage.step != 0 or runner.alg.transition.actions is not None):
        raise RuntimeError("cooperative-prep requires original seed and empty target422 storage")
    if runner._semantic_runner_config != factor["target_runner_config"]:
        raise RuntimeError("cooperative-prep changed unrelated PPO configuration")
    device = str(runner.device)
    source, _ = construct_semantic_runner(
        _shape_env(422, device), seed=seed, device=device,
        policy_version=P02_PROGRESS_POLICY,
        observation_layout=P02_PROGRESS_OBSERVATION_LAYOUT,
        initialize_actor=False)
    if source._semantic_runner_config != factor["source_runner_config"]:
        raise RuntimeError("cooperative-prep source PPO configuration differs")
    infos = load_semantic_checkpoint(
        source, Path(checkpoint), contract=metadata["runtime_contract"], seed=seed)
    for key, wanted in factor["preserved_metadata_sha256"].items():
        from .semantic_migration import digest
        if key not in infos or digest(infos[key]) != wanted:
            raise RuntimeError("sealed source metadata changed: " + key)
    runner.alg.actor.load_state_dict(source.alg.actor.state_dict(), strict=True)
    runner.alg.critic.load_state_dict(source.alg.critic.state_dict(), strict=True)
    runner.alg.optimizer.load_state_dict(source.alg.optimizer.state_dict())
    runner.alg.learning_rate = source.alg.learning_rate
    runner.current_learning_iteration = source.current_learning_iteration
    if (optimizer_learning_rate(runner) != optimizer_learning_rate(source)
            or parameter_hash(runner.alg.actor) != parameter_hash(source.alg.actor)
            or parameter_hash(runner.alg.critic) != parameter_hash(source.alg.critic)
            or state_hash(runner.alg.optimizer.state_dict())
                != state_hash(source.alg.optimizer.state_dict())
            or state_hash(_normalizers(runner)) != state_hash(_normalizers(source))):
        raise RuntimeError("cooperative-prep identity mapping changed model/Adam/normalizer")
    restore_training_rng_state(metadata["training_rng_state"], expected_seed=seed)
    from .semantic_migration import continuation_topology
    sampling = "P01_full_task_only_initial_version"
    result = {
        **infos,
        "runtime_contract": dict(contract),
        "runner_config": copy.deepcopy(runner._semantic_runner_config),
        "policy_contract": _runner_policy_contract(runner),
        "actor_parameter_sha256": parameter_hash(runner.alg.actor),
        "critic_parameter_sha256": parameter_hash(runner.alg.critic),
        "optimizer_state_sha256": state_hash(runner.alg.optimizer.state_dict()),
        "normalizer_state_sha256": state_hash(_normalizers(runner)),
        "resume_migration": verified,
        MIGRATION: verified,
        "old_rollout_inherited": False,
        "physical_env_state_saved": False,
        "resume_physics": "fresh_legal_reset_not_bitwise_simulator_resume",
        "sampling": sampling,
        "stage": "full_episode",
        "implemented_reset_sampling": sampling,
        "phase_suffix_curriculum_implemented": False,
        "curriculum_epoch": {"reset_sampling": sampling, "prefix_request": None},
        "execution_topology": continuation_topology(
            sampling, None, observation_layout=COOPERATIVE_PREP_OBSERVATION_LAYOUT),
    }
    _verify_target_identity(runner, result, factor)
    return result


def publish_cooperative_prep_checkpoint(checkpoint, contract, plan_path,
                                        output_checkpoint):
    from .semantic_migration import checkpoint_metadata, file_sha
    from .semantic_rr_capture_migration import _shape_env
    from .semantic_training import (
        construct_semantic_runner, load_semantic_checkpoint,
        save_semantic_checkpoint,
    )
    source = checkpoint_metadata(Path(checkpoint))
    record = validate_cooperative_prep_migration(checkpoint, contract, plan_path)
    route = source["checkpoint_output_routing"]
    destination = Path(output_checkpoint).resolve()
    if (destination.parent != Path(route["output_root"]) / "checkpoints" / "history"
            or destination.exists()
            or destination.with_name(destination.stem + "_manifest.json").exists()):
        raise ValueError("unique same-branch cooperative-prep publication required")

    def make():
        device = source["runner_config"]["device"]
        return construct_semantic_runner(
            _shape_env(422, device), seed=source["seed"], device=device,
            policy_version=COOPERATIVE_PREP_POLICY,
            observation_layout=COOPERATIVE_PREP_OBSERVATION_LAYOUT,
            initialize_actor=False)[0]

    runner = make()
    infos = load_cooperative_prep_migration(
        runner, checkpoint, contract=contract, seed=source["seed"], record=record)
    path, sidecar = save_semantic_checkpoint(runner, destination, infos)
    fresh = make()
    loaded = load_semantic_checkpoint(
        fresh, path, contract=contract, seed=source["seed"])
    _verify_target_identity(fresh, loaded, record[FACTOR_KEY])
    validate_cooperative_prep_lineage(
        loaded, contract, Path(route["output_root"]),
        checkpoint_output_routing=route)
    return {
        "checkpoint": str(path), "checkpoint_sha256": file_sha(path),
        "manifest": str(sidecar), "manifest_sha256": file_sha(sidecar),
        "save_load_round_trip": True,
        **{key: loaded[key] for key in COUNTERS},
        "rear_policy_timing_branch_counts": loaded["rear_policy_timing_branch_counts"],
        "added_policy_decisions": 0, "added_ppo_updates": 0,
        "added_optimizer_steps": 0, "added_auxiliary_updates": 0,
    }
