"""Reviewed reward-only same439 continuation; identity learned state, fresh rollout."""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

from .semantic_rear_policy_timing_migration import (
    COUNTERS, EXPERIMENT, SOURCE_COUNTS, preserved_keys,
)
from .semantic_rear_recapture_migration import BRANCH_NAME, CODE
from .semantic_rear_owner_profile import REAR_OWNER_POLICY, REAR_OWNER_OBSERVATION_LAYOUT

SCHEMA = "wlr50_clean.rr_retention_reward_same439.v1"
FACTOR_KEY = "rr_retention_same439_factor"
MIGRATION = "rr_retention_reward_migration"
SOURCE_HEAD = "f6d1d2df8d87d5f3eaaefc2254adb5f81fc52e2b"
ALLOWED = frozenset(CODE + name for name in (
    "semantic_reward.py", "semantic_rr_retention_migration.py",
    "semantic_migration.py", "semantic_cli.py", "semantic_training.py",
    "semantic_rear_policy_timing_migration.py",
))
MAPPING = "identity_all_actor_critic_buffers_full_Adam_options_steps_LR_Identity_rng"


def _policy():
    from .semantic_policy_distribution import policy_contract
    return policy_contract(REAR_OWNER_POLICY, observation_layout=REAR_OWNER_OBSERVATION_LAYOUT)


def _preserved(metadata):
    return tuple(sorted(set(preserved_keys(metadata)) | {
        "runner_config", "policy_contract", "actor_parameter_sha256",
        "critic_parameter_sha256", "optimizer_state_sha256", "checkpoint_output_routing",
    }))


def _reviewed_delta(old, new, root):
    from .semantic_migration import file_sha, digest
    variable = {"files", "source_git_commit", "runtime_content_sha256"}
    if ({k: v for k, v in old.items() if k not in variable}
            != {k: v for k, v in new.items() if k not in variable}):
        raise ValueError("same439 reward boundary changed unrelated runtime metadata/configuration")
    changed = {p for p in set(old["files"]) | set(new["files"])
               if old["files"].get(p) != new["files"].get(p)}
    if changed != ALLOWED or any(p not in new["files"] for p in changed):
        raise ValueError("same439 reward boundary requires exactly six reviewed runtime paths")
    if CODE + "semantic_rr_retention_migration.py" in old["files"]:
        raise ValueError("same439 first reward boundary cannot overwrite an existing migration")
    if any(file_sha(root / p) != sha for p, sha in new["files"].items()):
        raise ValueError("target runtime bytes differ from the frozen contract")
    if any(digest(c["files"]) != c["runtime_content_sha256"] for c in (old, new)):
        raise ValueError("runtime content digest differs from its full file inventory")
    return {p: {"before": old["files"].get(p), "after": new["files"][p]}
            for p in sorted(changed)}


def _previous_contract(current, receipt):
    from .semantic_migration import digest
    previous = copy.deepcopy(current)
    if (receipt.get("target_git_commit") != previous.get("source_git_commit")
            or receipt.get("target_runtime_content_sha256") != previous.get("runtime_content_sha256")
            or receipt.get("target_contract_sha256") != digest(previous)
            or set(receipt.get("changed_file_hashes", {})) != ALLOWED):
        raise ValueError("same439 target contract or reviewed delta changed")
    previous["source_git_commit"] = receipt["source_git_commit"]
    previous["runtime_content_sha256"] = receipt["source_runtime_content_sha256"]
    for path, hashes in receipt["changed_file_hashes"].items():
        if previous["files"].get(path) != hashes["after"]:
            raise ValueError("same439 target file differs from receipt: " + path)
        if hashes["before"] is None:
            previous["files"].pop(path)
        else:
            previous["files"][path] = hashes["before"]
    if (digest(previous) != receipt["source_contract_sha256"]
            or digest(previous["files"]) != previous["runtime_content_sha256"]):
        raise ValueError("same439 historical f6d runtime reconstruction differs")
    return previous


def build_rr_retention_migration(checkpoint, current_contract, *, reason,
        expected_source_sha256, expected_manifest_sha256, project_root=None):
    from .semantic_migration import checkpoint_metadata, _contract, file_sha, digest, source_num_envs
    from .semantic_rear_owner_migration import validate_rear_owner_lineage
    from .semantic_training import semantic_runner_config
    from .semantic_return_profile import runner_return_profile
    root = Path(project_root or Path(__file__).resolve().parents[3])
    checkpoint = Path(checkpoint).resolve(strict=True)
    branch = root / "outputs" / ("ppo_" + EXPERIMENT) / "branches" / BRANCH_NAME
    for value in (expected_source_sha256, expected_manifest_sha256):
        if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value):
            raise ValueError("explicit actual checkpoint and sidecar hashes are required")
    if (checkpoint.parent != (branch / "checkpoints/history").resolve()
            or file_sha(checkpoint) != expected_source_sha256
            or file_sha(checkpoint.with_name(checkpoint.stem + "_manifest.json")) != expected_manifest_sha256):
        raise ValueError("actual source checkpoint/sidecar/branch differs from explicit selection")
    metadata = checkpoint_metadata(checkpoint)
    old, new = _contract(metadata["runtime_contract"]), _contract(current_contract)
    validate_rear_owner_lineage(metadata, old, branch,
        checkpoint_output_routing=metadata.get("checkpoint_output_routing"))
    horizon = runner_return_profile(metadata["runner_config"], semantic_version="v3")["version"]
    runner_config = semantic_runner_config(seed=metadata["seed"], device=metadata["runner_config"]["device"],
        semantic_version="v3", policy_version=REAR_OWNER_POLICY,
        observation_layout=REAR_OWNER_OBSERVATION_LAYOUT, return_profile=horizon)
    last = metadata.get("last_update", {})
    if (old.get("source_git_commit") != SOURCE_HEAD or new.get("source_git_commit") == SOURCE_HEAD
            or not re.fullmatch("[0-9a-f]{40}", new.get("source_git_commit", ""))
            or old.get("experiment_id") != EXPERIMENT or new.get("experiment_id") != EXPERIMENT
            or not isinstance(reason, str) or not reason.strip() or MIGRATION in metadata
            or metadata.get("policy_contract") != _policy() or metadata["runner_config"] != runner_config
            or source_num_envs(metadata) != 1
            or last.get("global_policy_decisions") != metadata.get("global_policy_decisions")
            or last.get("ppo_update") != metadata.get("ppo_updates")
            or last.get("actor_parameter_sha256_after") != metadata.get("actor_parameter_sha256")
            or last.get("optimizer_learning_rate") != metadata.get("optimizer_learning_rate")
            or last.get("optimizer_steps") != runner_config["algorithm"]["num_learning_epochs"]
                * runner_config["algorithm"]["num_mini_batches"]):
        raise ValueError("requires latest explicitly selected complete f6d439 learned update; no ancestor publication")
    changed = _reviewed_delta(old, new, root)
    origin = {k: metadata[k] for k in COUNTERS}
    factor = dict(schema=SCHEMA, source_policy_contract=_policy(), target_policy_contract=_policy(),
        revision_counter_origin=origin, original_branch_origin=SOURCE_COUNTS,
        source_runner_config=copy.deepcopy(runner_config), target_runner_config=copy.deepcopy(runner_config),
        source_effective_learning_rate=metadata["optimizer_learning_rate"],
        source_stage_requested_decisions=copy.deepcopy(metadata["stage_requested_decisions"]),
        preserved_metadata_sha256={k: digest(metadata[k]) for k in _preserved(metadata)},
        parameter_mapping=MAPPING, observation_dimension=439, action_dimension=12, num_envs=1,
        observation_layout=REAR_OWNER_OBSERVATION_LAYOUT, same_mdp_claimed=False,
        reward_only=True, reward_revision="rr_recapture_retention_reward_only_v1",
        reward_local_potential_changed=True, shared_task_potential_observation_changed=False,
        observation_semantics_changed=False,
        same_numeric_input_Gaussian_and_value_preserved=True,
        physical_dynamics_changed=False, source_pose_commands_changed=False,
        control_projection_changed=False, HISTORY_changed=False, conditional_sigma_changed=False,
        old_rollout_inherited=False, added_policy_decisions=0, added_ppo_updates=0,
        added_optimizer_steps=0, added_auxiliary_updates=0)
    record = dict(schema=SCHEMA, reason=reason.strip(), source_checkpoint=str(checkpoint),
        source_checkpoint_sha256=expected_source_sha256, source_manifest_sha256=expected_manifest_sha256,
        source_git_commit=old["source_git_commit"], target_git_commit=new["source_git_commit"],
        source_contract_sha256=digest(old), target_contract_sha256=digest(new),
        source_runtime_content_sha256=old["runtime_content_sha256"],
        target_runtime_content_sha256=new["runtime_content_sha256"], changed_file_hashes=changed,
        discard_old_rollout_storage=True, physics_resume="fresh_legal_reset_not_bitwise_simulator_resume",
        **{FACTOR_KEY: factor})
    if _previous_contract(new, record) != old:
        raise ValueError("same439 reward boundary changed outside its reviewed delta")
    return record


def validate_rr_retention_migration(checkpoint, contract, plan_path, *, project_root=None):
    from .semantic_migration import file_sha
    path = Path(plan_path).resolve(strict=True)
    record = json.loads(path.read_text(encoding="utf-8"))
    expected = build_rr_retention_migration(checkpoint, contract, reason=record.get("reason", ""),
        expected_source_sha256=record.get("source_checkpoint_sha256"),
        expected_manifest_sha256=record.get("source_manifest_sha256"), project_root=project_root)
    if record != expected:
        raise ValueError("same439 reward plan/source/runtime changed")
    return {**record, "plan_path": str(path), "plan_sha256": file_sha(path)}


def validate_rr_retention_lineage(metadata, contract, output_root, *, checkpoint_output_routing=None):
    from .semantic_migration import digest, checkpoint_metadata, file_sha
    from .semantic_rear_owner_migration import validate_rear_owner_lineage
    receipt = metadata.get(MIGRATION, {})
    factor = receipt.get(FACTOR_KEY, {})
    origin = factor.get("revision_counter_origin", {})
    if (receipt.get("schema") != SCHEMA or factor.get("schema") != SCHEMA
            or receipt.get("source_git_commit") != SOURCE_HEAD
            or factor.get("source_policy_contract") != _policy() or factor.get("target_policy_contract") != _policy()
            or metadata.get("policy_contract") != _policy()
            or factor.get("source_runner_config") != factor.get("target_runner_config")
            or metadata.get("runner_config") != factor.get("target_runner_config")
            or factor.get("parameter_mapping") != MAPPING or factor.get("observation_dimension") != 439
            or factor.get("observation_layout") != REAR_OWNER_OBSERVATION_LAYOUT
            or factor.get("original_branch_origin") != SOURCE_COUNTS
            or factor.get("reward_only") is not True
            or factor.get("reward_revision") != "rr_recapture_retention_reward_only_v1"
            or factor.get("reward_local_potential_changed") is not True
            or factor.get("shared_task_potential_observation_changed") is not False
            or factor.get("observation_semantics_changed") is not False
            or factor.get("same_mdp_claimed") is not False
            or factor.get("same_numeric_input_Gaussian_and_value_preserved") is not True
            or factor.get("action_dimension") != 12 or factor.get("num_envs") != 1
            or receipt.get("discard_old_rollout_storage") is not True
            or any(factor.get(k) is not False for k in (
                "physical_dynamics_changed", "source_pose_commands_changed", "control_projection_changed",
                "HISTORY_changed", "conditional_sigma_changed", "old_rollout_inherited"))
            or any(type(origin.get(k)) is not int or type(metadata.get(k)) is not int
                   or metadata[k] < origin[k] for k in COUNTERS)
            or any(factor.get(k) != 0 for k in (
                "added_policy_decisions", "added_ppo_updates", "added_optimizer_steps", "added_auxiliary_updates"))):
        raise ValueError("same439 reward lineage/policy/counts changed")
    source_path = Path(receipt.get("source_checkpoint", "")).resolve(strict=True)
    source = checkpoint_metadata(source_path)
    if (source.get("checkpoint_sha256") != receipt.get("source_checkpoint_sha256")
            or file_sha(source_path.with_name(source_path.stem + "_manifest.json")) != receipt.get("source_manifest_sha256")
            or {k: source.get(k) for k in COUNTERS} != origin
            or source.get("runtime_contract", {}).get("source_git_commit") != SOURCE_HEAD
            or digest(source["runtime_contract"]) != receipt.get("source_contract_sha256")
            or factor.get("preserved_metadata_sha256") != {k: digest(source[k]) for k in _preserved(source)}
            or factor.get("source_stage_requested_decisions") != source.get("stage_requested_decisions")
            or factor.get("source_effective_learning_rate") != source.get("optimizer_learning_rate")
            or factor.get("source_runner_config") != source.get("runner_config")
            or source_path.parent != Path(source["checkpoint_output_routing"]["output_root"]) / "checkpoints/history"):
        raise ValueError("same439 immutable selected-source receipt changed")
    if any(type(metadata.get("stage_requested_decisions", {}).get(k)) is not int
            or metadata["stage_requested_decisions"][k] < value
            for k, value in source["stage_requested_decisions"].items()):
        raise ValueError("same439 used stage quantities cannot reset")
    for key, sha in factor.get("preserved_metadata_sha256", {}).items():
        if key.endswith(("_branch", "_migration")) or key in (
                "checkpoint_output_routing", "runner_config", "policy_contract", "seed", "normalization"):
            if digest(metadata.get(key)) != sha:
                raise ValueError("same439 protected source lineage changed: " + key)
    old = _previous_contract(contract, receipt)
    historical = {k: copy.deepcopy(v) for k, v in metadata.items() if k != MIGRATION}
    historical["runtime_contract"] = old
    validate_rear_owner_lineage(historical, old, output_root,
        checkpoint_output_routing=checkpoint_output_routing)
    return receipt


def _verify_identity(runner, infos, factor):
    import torch
    from .semantic_migration import digest
    from .semantic_training import parameter_hash, state_hash, _normalizers
    from .rl_library_wrapper import optimizer_learning_rate, capture_training_rng_state
    actual = dict(actor_parameter_sha256=parameter_hash(runner.alg.actor),
        critic_parameter_sha256=parameter_hash(runner.alg.critic),
        optimizer_state_sha256=state_hash(runner.alg.optimizer.state_dict()),
        normalizer_state_sha256=state_hash(_normalizers(runner)),
        training_rng_state=capture_training_rng_state(seed=infos["seed"]))
    if (any(type(getattr(runner.alg, role).obs_normalizer) is not torch.nn.Identity for role in ("actor", "critic"))
            or runner._semantic_runner_config != factor["target_runner_config"]
            or optimizer_learning_rate(runner) != factor["source_effective_learning_rate"]
            or runner.alg.learning_rate != factor["source_effective_learning_rate"]
            or runner.current_learning_iteration != infos["ppo_updates"]
            or any(infos.get(k) != value for k, value in actual.items())
            or any(k not in infos or digest(infos[k]) != sha for k, sha in factor["preserved_metadata_sha256"].items())
            or runner.alg.storage.step != 0 or runner.alg.transition.actions is not None
            or tuple(runner.alg.storage.actions.shape) != (128, 1, 12)
            or any(tuple(runner.alg.storage.observations[k].shape) != (128, 1, 439) for k in ("policy", "critic"))):
        raise RuntimeError("same439 full-state identity or fresh rollout verification failed")


def load_rr_retention_migration(runner, checkpoint, *, contract, seed, record):
    from .semantic_migration import checkpoint_metadata, continuation_topology
    from .semantic_training import load_semantic_checkpoint, _runner_policy_contract
    if any(k.endswith("_factor") and k != FACTOR_KEY and v is not None for k, v in record.items()):
        raise ValueError("same439 reward migration cannot mix another factor")
    verified = validate_rr_retention_migration(checkpoint, contract, record["plan_path"])
    metadata = checkpoint_metadata(Path(checkpoint))
    if (verified != dict(record) or seed != metadata["seed"]
            or _runner_policy_contract(runner) != verified[FACTOR_KEY]["target_policy_contract"]):
        raise ValueError("same439 original seed/policy and unchanged verified plan required")
    infos = load_semantic_checkpoint(runner, Path(checkpoint), contract=metadata["runtime_contract"], seed=seed)
    _verify_identity(runner, infos, verified[FACTOR_KEY])
    sampling = "P01_full_task_only_initial_version"
    return {**infos, "runtime_contract": dict(contract), "resume_migration": verified, MIGRATION: verified,
        "old_rollout_inherited": False, "physical_env_state_saved": False,
        "resume_physics": "fresh_legal_reset_not_bitwise_simulator_resume",
        "sampling": sampling, "stage": "full_episode", "implemented_reset_sampling": sampling,
        "phase_suffix_curriculum_implemented": False,
        "curriculum_epoch": {"reset_sampling": sampling, "prefix_request": None},
        "execution_topology": continuation_topology(sampling, None, observation_layout=REAR_OWNER_OBSERVATION_LAYOUT)}


def publish_rr_retention_checkpoint(checkpoint, contract, plan_path, output_checkpoint):
    from .semantic_migration import checkpoint_metadata, file_sha
    from .semantic_training import construct_semantic_runner, load_semantic_checkpoint, save_semantic_checkpoint
    from .semantic_rr_capture_migration import _shape_env
    source = checkpoint_metadata(Path(checkpoint))
    record = validate_rr_retention_migration(checkpoint, contract, plan_path)
    route = source["checkpoint_output_routing"]
    destination = Path(output_checkpoint).resolve()
    if (destination.parent != Path(route["output_root"]) / "checkpoints/history" or destination.exists()
            or destination.with_name(destination.stem + "_manifest.json").exists()):
        raise ValueError("unique immutable same-branch publication required; never promote latest pointer here")
    def make():
        device = source["runner_config"]["device"]
        return construct_semantic_runner(_shape_env(439, device), seed=source["seed"], device=device,
            policy_version=REAR_OWNER_POLICY, observation_layout=REAR_OWNER_OBSERVATION_LAYOUT,
            initialize_actor=False)[0]
    runner = make()
    infos = load_rr_retention_migration(runner, checkpoint, contract=contract, seed=source["seed"], record=record)
    path, sidecar = save_semantic_checkpoint(runner, destination, infos)
    fresh = make()
    loaded = load_semantic_checkpoint(fresh, path, contract=contract, seed=source["seed"])
    _verify_identity(fresh, loaded, record[FACTOR_KEY])
    validate_rr_retention_lineage(loaded, contract, Path(route["output_root"]), checkpoint_output_routing=route)
    return dict(checkpoint=str(path), checkpoint_sha256=file_sha(path), manifest=str(sidecar),
        manifest_sha256=file_sha(sidecar), save_load_round_trip=True, **{k: loaded[k] for k in COUNTERS},
        rear_policy_timing_branch_counts=loaded["rear_policy_timing_branch_counts"],
        added_policy_decisions=0, added_ppo_updates=0, added_optimizer_steps=0, added_auxiliary_updates=0)
