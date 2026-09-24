"""UNAPPLIED, outputs-only cooperative-prep migration boundary.

This module deliberately contains no model, Torch, checkpoint, or publication
code.  It defines the metadata/contract boundary that production may promote
only after the active d7e97ee course has sealed and its exact source identity
and reviewed runtime/policy deltas have been pinned.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Callable, Mapping


SCHEMA = "wlr50_clean.cooperative_prep_same422.v4"
FACTOR_KEY = "cooperative_prep_same422_factor"
MIGRATION = "cooperative_prep_migration"
SOURCE_HEAD = "d7e97ee7b7e493d4f3ff34f9c8762f73550ad7bd"
SOURCE_POLICY = "rear_policy_p02_progress_history_v1"
TARGET_POLICY = "rear_cooperative_prep_history_v1"
OBSERVATION_LAYOUT = "role419_p02_progress_v1"
OBSERVATION_DIMENSION = 422
COUNTERS = ("global_policy_decisions", "ppo_updates", "optimizer_steps")
ZERO_CREDIT = (
    "added_policy_decisions", "added_ppo_updates", "added_optimizer_steps",
    "added_auxiliary_updates",
)

# Intentionally not filled before the active course seals.  Promotion should
# replace these with literal reviewed values; callers may not infer them from a
# latest pointer or a requested budget.
SEALED_SOURCE_BINDING = None
REVIEWED_RUNTIME_DELTA = None
REVIEWED_POLICY_DELTA = None

_SHA256 = re.compile(r"[0-9a-f]{64}")


def digest(value) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _exact_sha(value, label):
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(label + " must be a lowercase SHA256")
    return value


def validate_source_binding(binding):
    """Validate explicit post-seal pins; never derive identity from a pointer."""
    if not isinstance(binding, Mapping):
        raise ValueError("cooperative-prep source is not sealed/pinned")
    required = {
        "checkpoint", "checkpoint_sha256", "manifest_sha256",
        "source_git_commit", *COUNTERS,
    }
    if set(binding) != required:
        raise ValueError("sealed source binding has missing or extra fields")
    if (not isinstance(binding["checkpoint"], str) or not binding["checkpoint"]
            or binding["source_git_commit"] != SOURCE_HEAD):
        raise ValueError("source must be the exact sealed d7e97ee course checkpoint")
    _exact_sha(binding["checkpoint_sha256"], "checkpoint_sha256")
    _exact_sha(binding["manifest_sha256"], "manifest_sha256")
    if any(type(binding[key]) is not int or binding[key] < 0 for key in COUNTERS):
        raise ValueError("sealed source counters must be nonnegative integers")
    return copy.deepcopy(dict(binding))


def _policy_boundary(source, target, reviewed_delta):
    """Require one exact reviewed policy delta; do not guess the sigma schema."""
    if not isinstance(source, Mapping) or not isinstance(target, Mapping):
        raise ValueError("source and target policy contracts are required")
    if (source.get("version") != SOURCE_POLICY
            or target.get("version") != TARGET_POLICY
            or source.get("observation_layout") != OBSERVATION_LAYOUT
            or target.get("observation_layout") != OBSERVATION_LAYOUT
            or source.get("observation_dimension") != OBSERVATION_DIMENSION
            or target.get("observation_dimension") != OBSERVATION_DIMENSION):
        raise ValueError("cooperative-prep requires the exact same422 policy boundary")
    if not isinstance(reviewed_delta, Mapping) or not reviewed_delta:
        raise ValueError("reviewed target policy delta is not frozen")
    changed = {
        key for key in set(source) | set(target)
        if source.get(key) != target.get(key)
    }
    if changed != set(reviewed_delta):
        raise ValueError("policy contract changed outside the reviewed sigma boundary")
    for key, pair in reviewed_delta.items():
        if (not isinstance(pair, Mapping) or set(pair) != {"before", "after"}
                or source.get(key) != pair["before"]
                or target.get(key) != pair["after"]):
            raise ValueError("reviewed policy delta differs at " + key)
    return copy.deepcopy(dict(target))


def _runtime_delta(source, target, reviewed_delta):
    if not isinstance(reviewed_delta, Mapping) or not reviewed_delta:
        raise ValueError("reviewed runtime/config delta is not frozen")
    before, after = source.get("files", {}), target.get("files", {})
    changed = {key for key in set(before) | set(after) if before.get(key) != after.get(key)}
    if changed != set(reviewed_delta) or any(key not in after for key in changed):
        raise ValueError("runtime changed outside the reviewed cooperative-prep delta")
    for key, pair in reviewed_delta.items():
        if (not isinstance(pair, Mapping) or set(pair) != {"before", "after"}
                or before.get(key) != pair["before"] or after.get(key) != pair["after"]):
            raise ValueError("reviewed runtime delta differs at " + key)
        if pair["after"] is None:
            raise ValueError("cooperative-prep migration may not remove runtime files")
    old_cfg, new_cfg = source.get("selected_configuration", {}), target.get("selected_configuration", {})
    if set(old_cfg) != set(new_cfg):
        raise ValueError("configuration inventory changed")
    changed_cfg = {}
    for name in old_cfg:
        if old_cfg[name].get("path") != new_cfg[name].get("path"):
            raise ValueError("configuration path changed: " + name)
        path = new_cfg[name].get("path")
        if (not isinstance(path, str) or before.get(path) != old_cfg[name].get("sha256")
                or after.get(path) != new_cfg[name].get("sha256")):
            raise ValueError("configuration binding differs from runtime files: " + name)
        if old_cfg[name] != new_cfg[name]:
            changed_cfg[name] = copy.deepcopy(old_cfg[name])
    return copy.deepcopy(dict(reviewed_delta)), changed_cfg


def reconstruct_source_contract(current, receipt):
    """Reverse only the reviewed v4 delta, preserving the old P02 contract."""
    previous = copy.deepcopy(current)
    previous["source_git_commit"] = receipt["source_git_commit"]
    previous["runtime_content_sha256"] = receipt["source_runtime_content_sha256"]
    for path, hashes in receipt["changed_file_hashes"].items():
        if previous.get("files", {}).get(path) != hashes["after"]:
            raise ValueError("current runtime no longer matches v4 receipt: " + path)
        if hashes["before"] is None:
            previous["files"].pop(path)
        else:
            previous["files"][path] = hashes["before"]
    previous["selected_configuration"].update(receipt["source_changed_configuration"])
    if digest(previous) != receipt["source_contract_sha256"]:
        raise ValueError("reconstructed P02 source contract is not exact")
    return previous


def _preserved_keys(metadata):
    explicit = {
        "actor_parameter_sha256", "critic_parameter_sha256",
        "optimizer_state_sha256", "optimizer_learning_rate",
        "normalizer_state_sha256", "normalization", "training_rng_state",
        "seed", "checkpoint_output_routing", "p02_progress_migration",
    }
    inherited = {
        key for key in metadata
        if key.endswith(("_migration", "_branch", "_branch_counts", "_auxiliary"))
    }
    result = sorted((explicit | inherited) - {MIGRATION})
    missing = [key for key in explicit if key not in metadata]
    if missing:
        raise ValueError("source full-state metadata missing: " + ",".join(sorted(missing)))
    return result


def build_draft_receipt(
    metadata,
    current_contract,
    *,
    reason,
    source_binding=None,
    reviewed_runtime_delta=None,
    reviewed_policy_delta=None,
    target_policy_contract,
    ancestry_validator: Callable[[dict, dict], object],
):
    """Build a fail-closed metadata receipt without loading a model.

    ``ancestry_validator`` must wrap production ``validate_p02_progress_lineage``
    with the checkpoint's existing route/output root.  It is deliberately
    invoked on the untouched source metadata and source runtime before the new
    target policy contract is considered.
    """
    binding = validate_source_binding(
        SEALED_SOURCE_BINDING if source_binding is None else source_binding)
    runtime_pins = REVIEWED_RUNTIME_DELTA if reviewed_runtime_delta is None else reviewed_runtime_delta
    policy_pins = REVIEWED_POLICY_DELTA if reviewed_policy_delta is None else reviewed_policy_delta
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("nonempty migration reason required")
    if MIGRATION in metadata:
        raise ValueError("cooperative-prep migration is already present")
    old = copy.deepcopy(metadata.get("runtime_contract"))
    if (not isinstance(old, dict) or old.get("source_git_commit") != SOURCE_HEAD
            or metadata.get("policy_contract", {}).get("version") != SOURCE_POLICY
            or metadata.get("p02_progress_migration") is None
            or metadata.get("save_load_round_trip") is not True):
        raise ValueError("source is not an intact sealed P02-progress checkpoint")
    if ({key: metadata.get(key) for key in COUNTERS}
            != {key: binding[key] for key in COUNTERS}):
        raise ValueError("source counters differ from the post-seal binding")
    # Critical ordering: validate the real old policy/runtime, never a target
    # policy substituted into the historical P02 receipt.
    ancestry_validator(copy.deepcopy(dict(metadata)), copy.deepcopy(old))
    target_policy = _policy_boundary(metadata["policy_contract"], target_policy_contract, policy_pins)
    runtime_delta, source_changed_configuration = _runtime_delta(old, current_contract, runtime_pins)
    if current_contract.get("source_git_commit") == SOURCE_HEAD:
        raise ValueError("target runtime commit must differ from the sealed source")
    factor = {
        "schema": SCHEMA,
        "source_policy_contract": copy.deepcopy(metadata["policy_contract"]),
        "target_policy_contract": target_policy,
        "revision_counter_origin": {key: binding[key] for key in COUNTERS},
        "preserved_metadata_sha256": {
            key: digest(metadata[key]) for key in _preserved_keys(metadata)
        },
        "parameter_mapping": "identity_all_actor_critic_buffers_full_Adam_options_steps_and_rng",
        "observation_dimension": OBSERVATION_DIMENSION,
        "observation_layout": OBSERVATION_LAYOUT,
        "observable_overlapping_prep_sigma_only": True,
        "same_numeric_input_Gaussian_preserved": False,
        "learned_parameters_changed_by_migration": False,
        "history_changed": False,
        "normalizer_changed": False,
        "learning_rate_changed": False,
        "physics_changed": False,
        "source_pose_commands_changed": False,
        "RR_assist_changed": False,
        "wheel_shaping_changed": False,
        "bounded_preparation_reward_semantics_changed": True,
        "geometry_telemetry_added": True,
        "other_reward_terms_changed": False,
        "task_acceptance_changed": False,
        "reward_and_geometry_telemetry_reviewed_in_runtime_delta": True,
        "old_rollout_inherited": False,
        **{key: 0 for key in ZERO_CREDIT},
    }
    receipt = {
        "schema": SCHEMA,
        "reason": reason.strip(),
        "source_checkpoint": binding["checkpoint"],
        "source_checkpoint_sha256": binding["checkpoint_sha256"],
        "source_manifest_sha256": binding["manifest_sha256"],
        "source_git_commit": SOURCE_HEAD,
        "target_git_commit": current_contract.get("source_git_commit"),
        "source_runtime_content_sha256": old.get("runtime_content_sha256"),
        "target_runtime_content_sha256": current_contract.get("runtime_content_sha256"),
        "source_contract_sha256": digest(old),
        "target_contract_sha256": digest(current_contract),
        "changed_file_hashes": runtime_delta,
        "source_changed_configuration": source_changed_configuration,
        "discard_old_rollout_storage": True,
        FACTOR_KEY: factor,
    }
    if reconstruct_source_contract(current_contract, receipt) != old:
        raise ValueError("runtime changed outside the reviewed v4 delta")
    return receipt


def validate_draft_lineage(metadata, current_contract, *, ancestry_validator):
    """Validate target lineage by reconstructing and validating old P02 first."""
    receipt = metadata.get(MIGRATION, {})
    factor = receipt.get(FACTOR_KEY, {})
    if (receipt.get("schema") != SCHEMA or factor.get("schema") != SCHEMA
            or receipt.get("target_contract_sha256") != digest(current_contract)
            or metadata.get("policy_contract") != factor.get("target_policy_contract")
            or factor.get("source_policy_contract", {}).get("version") != SOURCE_POLICY
            or factor.get("target_policy_contract", {}).get("version") != TARGET_POLICY
            or factor.get("observation_layout") != OBSERVATION_LAYOUT
            or factor.get("observation_dimension") != OBSERVATION_DIMENSION
            or factor.get("parameter_mapping")
            != "identity_all_actor_critic_buffers_full_Adam_options_steps_and_rng"
            or factor.get("learned_parameters_changed_by_migration") is not False
            or factor.get("history_changed") is not False
            or factor.get("normalizer_changed") is not False
            or factor.get("learning_rate_changed") is not False
            or factor.get("physics_changed") is not False
            or factor.get("source_pose_commands_changed") is not False
            or factor.get("RR_assist_changed") is not False
            or factor.get("wheel_shaping_changed") is not False
            or any(factor.get(key) != 0 for key in ZERO_CREDIT)):
        raise ValueError("cooperative-prep lineage/identity boundary differs")
    origin = factor.get("revision_counter_origin", {})
    if (set(origin) != set(COUNTERS)
            or any(type(origin[key]) is not int or origin[key] < 0 for key in COUNTERS)
            or any(type(metadata.get(key)) is not int or metadata[key] < origin[key] for key in COUNTERS)):
        raise ValueError("cooperative-prep continuation reset or borrowed counters")
    for key, wanted in factor.get("preserved_metadata_sha256", {}).items():
        if key not in metadata or digest(metadata[key]) != wanted:
            raise ValueError("protected source metadata changed: " + key)
    old = reconstruct_source_contract(current_contract, receipt)
    historical = copy.deepcopy(dict(metadata))
    historical.pop(MIGRATION, None)
    historical["runtime_contract"] = old
    historical["policy_contract"] = copy.deepcopy(factor["source_policy_contract"])
    ancestry_validator(historical, old)
    return receipt
