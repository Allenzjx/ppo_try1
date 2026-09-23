"""Explicit initial419 ancestor continuation with identity training-state mapping."""
from __future__ import annotations
import copy
import json
import re
from pathlib import Path

from .semantic_rear_policy_timing_migration import (
    EXPERIMENT, COUNTERS, SOURCE_COUNTS, SOURCE_SHA, preserved_keys,
    validate_rear_policy_namespace)
from .semantic_rear_policy_timing_profile import (
    REAR_POLICY_TIMING_POLICY, REAR_POLICY_TIMING_OBSERVATION_LAYOUT)

SCHEMA = "wlr50_clean.rear_recapture_same419.v1"
FACTOR_KEY = "rear_recapture_same419_factor"
MIGRATION = "rear_recapture_migration"
SOURCE_HEAD = "fa4b98ed506eb2230e61e13bd93ceecdcb6cfdad"
SOURCE_MODE = "rr_capture_before_rl_transfer_v1"
TARGET_MODE = "rr_recapture_current_support_v2"
BRANCH_NAME = "ancestor220544_recapture_v2"
INITIAL_SOURCE_SHA = "e83b7340febb760fdd00aa5cf44c07bbf45809a2f9a175b1eafd0d670c3cef85"
INITIAL_MANIFEST_SHA = "3fba5f7d8b32f98420924fc5103b4ea4f58205cef004651658152ea011dc05f6"
SOURCE_SELECTION = dict(source_role="explicit_initial419_ancestor_recapture_branch",
    checkpoint_sha256=INITIAL_SOURCE_SHA, manifest_sha256=INITIAL_MANIFEST_SHA,
    counters=SOURCE_COUNTS, output_branch=BRANCH_NAME)
CONFIG = "configs/ppo_rr_rl_timing_policy_learning_v1/"
CODE = "src/wlr50_clean/ppo/"
# Exact candidate scope; amend only after root reviews the actual controller delta.
ALLOWED = frozenset(CODE + name for name in (
    "semantic_rear_recapture_migration.py", "semantic_rear_policy_timing_migration.py",
    "semantic_migration.py", "semantic_training.py", "semantic_cli.py",
    "semantic_rear_policy_timing.py", "semantic_supervisor.py", "semantic_backend.py")) | frozenset((
    CONFIG + "execution_profile.yaml", CONFIG + "stage_task_spec.yaml"))

# Separate reviewed media-only delta; never a control-file wildcard.
MEDIA_REVIEW_COMMIT = "a9825050e4f92ac55153c2501fb5af59fc71e9c1"
MEDIA_REVIEW = {
    CODE + "semantic_video.py": dict(
        before="bce0d97a622a9aede1fdd203db9ec1963e39e2a2ea5529b67c612a982de2d548",
        after="facd65f5ce0a83d732de168923380daf816266e391d48a8a6ac9c00799cdab62"),
    CODE + "semantic_video_cli.py": dict(
        before="ceda23cedf8e25d6559d48179a780f766da5ab7c15bdc86f780054845c39956b",
        after="5633338cd9752c0732aad0e47c1e9f4ed90a69e2f7db3a24b4cf2fa7baeca7e2"),
}


def _reviewed_media_delta(old, new):
    changed = {path for path in MEDIA_REVIEW if old["files"].get(path) != new["files"].get(path)}
    if not changed:
        return None
    if changed != set(MEDIA_REVIEW):
        raise ValueError("media compatibility delta must match the complete reviewed two-file revision")
    measured = {p:dict(before=old["files"].get(p),after=new["files"].get(p)) for p in sorted(changed)}
    if any(not isinstance(row["after"],str) or not re.fullmatch("[0-9a-f]{64}",row["after"])
            for row in MEDIA_REVIEW.values()) or measured != MEDIA_REVIEW:
        raise ValueError("media revision is pending review or differs from the exact reviewed before/after bytes")
    return dict(schema="wlr50_clean.reviewed_video_binding_compatibility.v1",
        reviewed_media_commit=MEDIA_REVIEW_COMMIT,
        scope="selected_six_configs_plus_hashed_curriculum_and_explicit_video_only_runtime_compatibility",
        changed_file_hashes=measured, task_control_changed_by_media_revision=False,
        policy_weights_changed_by_media_revision=False, added_learning=0)


def _preserved(metadata):
    return tuple(sorted(set(preserved_keys(metadata)) | {
        "runner_config", "policy_contract", "actor_parameter_sha256",
        "critic_parameter_sha256", "optimizer_state_sha256"}))


def build_rear_recapture_migration(checkpoint, current_contract, *, reason,
        expected_source_sha256, expected_manifest_sha256, project_root=None):
    import yaml
    from .semantic_migration import (
        checkpoint_metadata, digest, file_sha, source_num_envs, _contract, _version_bytes)
    from .semantic_policy_distribution import policy_contract
    root = Path(project_root or Path(__file__).resolve().parents[3])
    checkpoint = Path(checkpoint).resolve(strict=True)
    for value in (expected_source_sha256, expected_manifest_sha256):
        if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{64}", value):
            raise ValueError("explicit actual source and sidecar hashes are required")
    sidecar = checkpoint.with_name(checkpoint.stem + "_manifest.json")
    if file_sha(checkpoint) != expected_source_sha256 or file_sha(sidecar) != expected_manifest_sha256:
        raise ValueError("source checkpoint or sidecar differs from explicit binding")
    metadata = checkpoint_metadata(checkpoint)
    if (expected_source_sha256 != INITIAL_SOURCE_SHA or expected_manifest_sha256 != INITIAL_MANIFEST_SHA
            or {k:metadata.get(k) for k in COUNTERS} != SOURCE_COUNTS
            or metadata.get("rear_policy_timing_branch_counts") != dict.fromkeys(COUNTERS, 0)):
        raise ValueError("recapture branch requires the explicitly registered initial419 ancestor; no borrowed learning")
    old, new = _contract(metadata["runtime_contract"]), _contract(current_contract)
    validate_rear_policy_namespace(metadata, old, root / "outputs" / ("ppo_" + EXPERIMENT))
    policy = policy_contract(REAR_POLICY_TIMING_POLICY,
        observation_layout=REAR_POLICY_TIMING_OBSERVATION_LAYOUT)
    if (old.get("source_git_commit") != SOURCE_HEAD or MIGRATION in metadata
            or old.get("experiment_id") != EXPERIMENT or new.get("experiment_id") != EXPERIMENT
            or metadata.get("policy_contract") != policy or source_num_envs(metadata) != 1
            or old.get("frozen_A_files") != new.get("frozen_A_files")
            or not reason.strip() or not re.fullmatch("[0-9a-f]{40}", new.get("source_git_commit", ""))
            or metadata.get("save_load_round_trip") is not True):
        raise ValueError("requires an intact sealed same-namespace fa4b98ed419 source and explicit target")
    changed = sorted(k for k in set(old["files"]) | set(new["files"])
        if old["files"].get(k) != new["files"].get(k))
    media_review = _reviewed_media_delta(old, new)
    if not changed or not set(changed) <= ALLOWED | set(MEDIA_REVIEW) or any(k not in new["files"] for k in changed):
        raise ValueError("unreviewed runtime delta or removed source file")
    for path, expected in new["files"].items():
        if file_sha(root / path) != expected:
            raise ValueError("target runtime bytes changed: " + path)
    if set(old["selected_configuration"]) != set(new["selected_configuration"]):
        raise ValueError("same419 configuration inventory changed")
    for name, entry in new["selected_configuration"].items():
        old_entry = old["selected_configuration"][name]
        if entry["path"] != old_entry["path"]:
            raise ValueError("same419 migration cannot switch configuration namespace")
        before = _version_bytes(root, old, old_entry["path"])
        after = (root / entry["path"]).read_bytes()
        if file_sha(root / entry["path"]) != entry["sha256"]:
            raise ValueError("target configuration hash differs")
        a, b = yaml.safe_load(before), yaml.safe_load(after)
        if name == "execution_profile.yaml":
            if a.get("rear_policy_timing_mode") != SOURCE_MODE or b.get("rear_policy_timing_mode") != TARGET_MODE:
                raise ValueError("explicit source/target execution mode missing")
            for cfg in (a, b):
                cfg.pop("revision", None); cfg.pop("rear_policy_timing_mode", None)
            if a != b:
                raise ValueError("unrelated execution/assist/physics/caps change")
        elif name == "stage_task_spec.yaml":
            if a["nominal"].get("rear_policy_timing") != SOURCE_MODE or b["nominal"].get("rear_policy_timing") != TARGET_MODE:
                raise ValueError("explicit source/target task mode missing")
            a["nominal"].pop("rear_policy_timing"); b["nominal"].pop("rear_policy_timing")
            if a != b:
                raise ValueError("unrelated physical acceptance/task constants change")
        elif before != after:
            raise ValueError("unrelated config bytes changed: " + name)
    origin = {k: metadata[k] for k in COUNTERS}
    factor = dict(schema=SCHEMA, source_policy_contract=policy, target_policy_contract=policy,
        source_selection=copy.deepcopy(SOURCE_SELECTION),
        source_mode=SOURCE_MODE, target_mode=TARGET_MODE,
        revision_counter_origin=origin, original_branch_origin=SOURCE_COUNTS,
        preserved_metadata_sha256={k:digest(metadata[k]) for k in _preserved(metadata)},
        source_effective_learning_rate=metadata["optimizer_learning_rate"],
        parameter_mapping="identity_all_actor_critic_parameters_buffers_and_full_Adam",
        observation_dimension=419, action_dimension=12, num_envs=1,
        observation_semantics_changed=["17_task_potential", "410_413_public_rear_task_state"],
        physical_dynamics_changed=False, same_mdp_claimed=False,
        reward_potential_semantics_changed="RR_placed_history_0p8_retained_current_0p2_usable_support_or_legal_recapture_until_RL_live_swing",
        task_state_semantics_changed="RR_current_support_loss_reopens_capture_before_RL_live_swing",
        media_revision_review=media_review,
        same_numeric_input_Gaussian_preserved=True, physical_state_Gaussian_equivalence_claimed=False,
        old_rollout_inherited=False, added_policy_decisions=0, added_ppo_updates=0,
        added_optimizer_steps=0, added_auxiliary_updates=0)
    return dict(schema=SCHEMA, reason=reason.strip(), source_checkpoint=str(checkpoint),
        source_selection=copy.deepcopy(SOURCE_SELECTION),
        source_checkpoint_sha256=expected_source_sha256, source_manifest_sha256=expected_manifest_sha256,
        source_contract_sha256=digest(old), target_contract_sha256=digest(new),
        source_runtime_content_sha256=old["runtime_content_sha256"],
        target_runtime_content_sha256=new["runtime_content_sha256"],
        source_git_commit=old["source_git_commit"], target_git_commit=new["source_git_commit"],
        allowed_changed_files=changed,
        changed_file_hashes={p:dict(before=old["files"].get(p),after=new["files"][p]) for p in changed},
        preserve_actor_critic_optimizer_normalizer_rng_and_budget=True,
        discard_old_rollout_storage=True, physics_resume="fresh_legal_reset_not_bitwise_simulator_resume",
        **{FACTOR_KEY:factor})


def validate_rear_recapture_migration(checkpoint, current_contract, plan_path, *, project_root=None):
    from .semantic_migration import file_sha
    path = Path(plan_path).resolve(strict=True)
    plan = json.loads(path.read_text(encoding="utf-8"))
    expected = build_rear_recapture_migration(checkpoint, current_contract,
        reason=plan.get("reason", ""), expected_source_sha256=plan.get("source_checkpoint_sha256"),
        expected_manifest_sha256=plan.get("source_manifest_sha256"), project_root=project_root)
    if plan != expected:
        raise ValueError("same419 plan is not bound to exact source/current runtime")
    return {**plan, "plan_path":str(path), "plan_sha256":file_sha(path)}


def validate_rear_recapture_lineage(metadata, contract, initial_receipt):
    from .semantic_migration import digest
    receipt = metadata.get(MIGRATION, {})
    factor = receipt.get(FACTOR_KEY, {})
    origin = factor.get("revision_counter_origin", {})
    if (receipt.get("schema") != SCHEMA or factor.get("schema") != SCHEMA
            or receipt.get("source_selection") != SOURCE_SELECTION
            or factor.get("source_selection") != SOURCE_SELECTION
            or receipt.get("source_checkpoint_sha256") != INITIAL_SOURCE_SHA
            or receipt.get("source_manifest_sha256") != INITIAL_MANIFEST_SHA
            or receipt.get("target_git_commit") != contract.get("source_git_commit")
            or receipt.get("target_runtime_content_sha256") != contract.get("runtime_content_sha256")
            or receipt.get("source_contract_sha256") != initial_receipt.get("target_contract_sha256")
            or receipt.get("target_contract_sha256") != digest(contract)
            or receipt.get("source_git_commit") != SOURCE_HEAD
            or factor.get("target_mode") != TARGET_MODE
            or factor.get("source_policy_contract") != metadata.get("policy_contract")
            or factor.get("target_policy_contract") != metadata.get("policy_contract")
            or factor.get("original_branch_origin") != SOURCE_COUNTS
            or origin != SOURCE_COUNTS
            or any(type(metadata.get(k)) is not int or origin[k] > metadata[k] for k in COUNTERS)
            or factor.get("preserved_metadata_sha256", {}).get("rear_policy_timing_migration") != digest(initial_receipt)
            or any(factor.get(k) != 0 for k in (
                "added_policy_decisions","added_ppo_updates","added_optimizer_steps","added_auxiliary_updates"))):
        raise ValueError("same419 current revision lacks an intact ancestor/source/target chain")
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
    if (any(type(getattr(runner.alg, role).obs_normalizer) is not torch.nn.Identity for role in ("actor","critic"))
            or optimizer_learning_rate(runner) != factor["source_effective_learning_rate"]
            or runner.alg.learning_rate != factor["source_effective_learning_rate"]
            or runner._semantic_runner_config != infos["runner_config"]
            or any(infos.get(k) != value for k,value in actual.items())
            or any(k not in infos or digest(infos[k]) != expected
                for k,expected in factor["preserved_metadata_sha256"].items())
            or runner.alg.storage.step != 0 or runner.alg.transition.actions is not None
            or tuple(runner.alg.storage.actions.shape) != (128,1,12)
            or any(tuple(runner.alg.storage.observations[k].shape) != (128,1,419) for k in ("policy","critic"))):
        raise RuntimeError("same419 full-state preservation/fresh storage check failed")


def load_rear_recapture_migration(runner, checkpoint, *, contract, seed, record):
    from .semantic_migration import checkpoint_metadata
    from .semantic_training import load_semantic_checkpoint, _runner_policy_contract
    verified = validate_rear_recapture_migration(checkpoint, contract, record["plan_path"])
    if verified != dict(record):
        raise RuntimeError("migration changed after preflight")
    source = checkpoint_metadata(Path(checkpoint))
    if seed != source["seed"] or _runner_policy_contract(runner) != verified[FACTOR_KEY]["target_policy_contract"]:
        raise RuntimeError("same419 original seed/policy contract required")
    infos = load_semantic_checkpoint(runner, Path(checkpoint), contract=source["runtime_contract"], seed=seed)
    _verify_identity(runner, infos, verified[FACTOR_KEY])
    return {**infos, "runtime_contract":dict(contract), "resume_migration":verified, MIGRATION:verified,
        "old_rollout_inherited":False, "physical_env_state_saved":False,
        "resume_physics":"fresh_legal_reset_not_bitwise_simulator_resume"}


def publish_rear_recapture_checkpoint(checkpoint, current_contract, plan_path, output_checkpoint):
    from .semantic_migration import checkpoint_metadata, file_sha
    from .semantic_training import construct_semantic_runner, load_semantic_checkpoint, save_semantic_checkpoint
    from .semantic_rr_capture_migration import _shape_env
    source = checkpoint_metadata(Path(checkpoint))
    record = validate_rear_recapture_migration(checkpoint, current_contract, plan_path)
    device = source["runner_config"]["device"]
    def make():
        return construct_semantic_runner(_shape_env(419,device), seed=source["seed"], device=device,
            policy_version=REAR_POLICY_TIMING_POLICY,
            observation_layout=REAR_POLICY_TIMING_OBSERVATION_LAYOUT, initialize_actor=False)[0]
    runner = make()
    infos = load_rear_recapture_migration(runner, checkpoint, contract=current_contract, seed=source["seed"], record=record)
    path, sidecar = save_semantic_checkpoint(runner, Path(output_checkpoint), infos)
    fresh = make()
    loaded = load_semantic_checkpoint(fresh, path, contract=current_contract, seed=source["seed"])
    _verify_identity(fresh, loaded, record[FACTOR_KEY])
    validate_rear_policy_namespace(loaded, current_contract, path.parents[2])
    return dict(checkpoint=str(path),checkpoint_sha256=file_sha(path),manifest=str(sidecar),
        manifest_sha256=file_sha(sidecar),save_load_round_trip=True, **{k:loaded[k] for k in COUNTERS},
        rear_policy_timing_branch_counts=loaded["rear_policy_timing_branch_counts"],
        added_policy_decisions=0,added_ppo_updates=0,added_optimizer_steps=0,added_auxiliary_updates=0)
