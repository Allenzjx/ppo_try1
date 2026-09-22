"""Offline export for one sealed P05 capture-assist deterministic video source.

The helper never starts Isaac or a policy.  It preserves the full 15 fps,
single-episode source, overlays measured capture evidence without adding intro
frames, emits one truthful contiguous P05 detail excerpt, and builds a clearly labelled
historical-N same-camera comparison.  A failed physical attempt remains a
publishable DIAGNOSTIC_FAILURE; it is never renamed success.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = Path(__file__).resolve().parent
DEFAULT_HISTORICAL_N = (ROOT / "runs" / "ppo_task_conditioned_hip_wheel_v1" / "video_eval" /
    "prior_B" / "20260921T0701155546642Z_gee5a9651591d_64985c63292d4a9b966716c3a54659a1" / "source")
DEFAULT_HISTORICAL_VERSION = "N_ref historical ee5a9651591d 73.808s"
EXPERIMENT = "p05_hip_only_continuation_v1"
CONTROL_METHOD = "PPO_PLUS_CAPTURE_ASSIST_WITH_INHERITED_LIMITED_AUX"
COUNTERS = ("global_policy_decisions", "ppo_updates", "optimizer_steps")
P05_COUNTER_ORIGIN = {"global_policy_decisions": 199680, "ppo_updates": 1525,
                      "optimizer_steps": 30500}
FEEDBACK_SCHEMA = "wlr50_clean.capture_feedback_semantics_same389.v1"
FEEDBACK_REVISION = "hold_to_air_progress_window_v2"
RR_WORKSPACE_SCHEMA = "wlr50_clean.rr_postcross_workspace_same389.v1"
RR_WORKSPACE_SEMANTICS = "current_qualified_RR_over_top_receiver_retirement_v1"
RR_WORKSPACE_MODE_KEY = "rr_postcross_workspace_semantics"
RR_WORKSPACE_SOURCE_SEMANTICS = "RR_receiver_preparation_share_remains_active_after_crossing"
RR_RECEIVER_V2_SCHEMA = "wlr50_clean.rr_postcross_workspace_same389.v2"
RR_RECEIVER_V2_SEMANTICS = "established_RR_over_top_receiver_retirement_v2"
RR_RECEIVER_V2_BRANCH = "rr_receiver_retirement_v2_branch"
RR_RECEIVER_V2_MIGRATION = "rr_receiver_retirement_v2_migration"
RR_RECEIVER_V2_COUNTS = "rr_receiver_retirement_v2_branch_counts"
P05_PREEDGE_SCHEMA = "wlr50_clean.p05_preedge_approach_recovery_same389.v1"
P05_PREEDGE_SEMANTICS = "p05_preedge_approach_recovery_v1"
P05_PREEDGE_SOURCE_SEMANTICS = "authored_P05_endpoint_without_preedge_recovery"
P05_PREEDGE_MODE_KEY = "p05_preedge_approach_recovery"
P05_PREEDGE_FACTOR_KEY = "p05_preedge_approach_recovery_factor"
P05_PREEDGE_BRANCH = "p05_preedge_approach_recovery_branch"
P05_PREEDGE_MIGRATION = "p05_preedge_approach_recovery_migration"
P05_PREEDGE_COUNTS = "p05_preedge_approach_recovery_branch_counts"
P05_PREEDGE_PRIOR_BRANCHES = (
    "p05_capture_assist", "capture_feedback_semantics",
    "rr_postcross_workspace", "rr_receiver_retirement_v2",
)
P05_PREEDGE_REQUIRED_CHANGED_FILES = frozenset({
    "src/wlr50_clean/ppo/semantic_supervisor.py",
    "src/wlr50_clean/ppo/semantic_p05_preedge_migration.py",
    "configs/ppo_p05_hip_only_continuation_v1/stage_task_spec.yaml",
})
P05_PREEDGE_ALLOWED_CHANGED_FILES = P05_PREEDGE_REQUIRED_CHANGED_FILES | frozenset({
    "src/wlr50_clean/ppo/semantic_migration.py",
    "src/wlr50_clean/ppo/semantic_training.py",
})
RR_WORKSPACE_REQUIRED_HISTORY = frozenset({
    *COUNTERS, "stage_requested_decisions", "new_mdp_origin_global_policy_decisions",
    "p05_capture_assist_branch", "p05_capture_assist_migration",
    "capture_feedback_semantics_branch", "capture_feedback_semantics_migration",
    "task_conditioned_hip_wheel_branch", "actor_parameter_sha256",
    "critic_parameter_sha256", "optimizer_state_sha256", "normalizer_state_sha256",
    "normalization", "training_rng_state", "optimizer_learning_rate", "runner_config",
    "policy_contract",
})
FRONT_AUX_KEY = "front_rehearsal_auxiliary"
FRONT_AUX_SCHEMA = "wlr50_clean.front_rehearsal_auxiliary.v1"
FRONT_AUX_REPORT_SCHEMA = "wlr50_clean.finite_front_phase_columns_aux.v1"
FRONT_AUX_DATA_SCHEMA = "wlr50_clean.front_rehearsal_exact_block03_selection.v1"
FRONT_AUX_LINEAGE_LABEL = "PPO_plus_explicit_phase_column_raw_action_rehearsal"
FRONT_AUX_KIND = "finite_supervised_raw_action_phase_columns_not_PPO"
FRONT_AUX_TARGET = "actual_logged_raw_samples_not_stored_mean_or_final_target"
FRONT_AUX_PARAMETERS = ["actor.mlp.0.weight[:,0:2]"]
FRONT_AUX_HELPERS = frozenset({"front_rehearsal.py", "reviewed_data.py", "rehearsal_cli.py"})
FRONT_AUX_DET_P02_SCHEMA = "wlr50_clean.finite_reconstructed_deterministic_P02_aux.v2"
FRONT_AUX_DET_P02_INSPECTION_SCHEMA = (
    "wlr50_clean.deterministic_front_rehearsal_readonly_inspection.v2")
FRONT_AUX_DET_P02_ADMISSION_SCHEMA = "wlr50_clean.det_front_current_semantics_admission.v1"
FRONT_AUX_DET_P02_ADMISSION_RESULT = (
    "PASS_BOUNDED_P02_HISTORICAL_SUPERVISED_DATA_SEMANTICS_ONLY")
FRONT_AUX_DET_P02_KIND = (
    "finite_supervised_reconstructed_deterministic_P02_raw_actions_not_PPO")
FRONT_AUX_DET_P02_TARGET = (
    "actual_executed_deterministic_raw_equal_to_recorded_mu_not_unexecuted_mean_or_final_target")
FRONT_AUX_DET_P02_PARAMETERS = ["actor.mlp.0.weight[:,1]"]
FRONT_AUX_DET_P02_HELPERS = frozenset({
    "execute_det_rehearsal.py", "inspect_det_rehearsal.py",
    "front_rehearsal_v1/front_rehearsal.py",
    "front_rehearsal_v1/reviewed_data.py",
    "det_front_rehearsal_data_v1/read_candidate.py",
    "audit_det_P02_current_semantics.py",
})
FRONT_AUX_MEAN_V3_SCHEMA = "wlr50_clean.finite_existing_mean_head_aux.v3.fit"
FRONT_AUX_MEAN_V3_INSPECTION_SCHEMA = (
    "wlr50_clean.finite_existing_mean_head_aux.v3.readonly")
FRONT_AUX_MEAN_V3_DATA_SCHEMA = (
    "wlr50_clean.front_mean_rehearsal_data_composition.v3.loaded")
FRONT_AUX_MEAN_V3_OBJECTIVE_SCHEMA = (
    "wlr50_clean.finite_existing_mean_head_aux.v3.objective")
FRONT_AUX_MEAN_V3_BUDGET_SCHEMA = (
    "wlr50_clean.finite_existing_mean_head_aux.v3.explicit_budget")
FRONT_AUX_MEAN_V3_KIND = (
    "finite_supervised_existing_mean_head_actual_front_raw_actions_not_PPO")
FRONT_AUX_MEAN_V3_TARGET = (
    "actually_executed_deterministic_raw_equal_to_recorded_conditional_mean")
FRONT_AUX_MEAN_V3_PARAMETERS = [
    "actor.mlp.4.weight[:12,:]", "actor.mlp.4.bias[:12]"]
FRONT_AUX_MEAN_V3_HELPERS = frozenset({
    "front_mean_rehearsal.py", "fit_mean_core.py",
    "execute_mean_rehearsal.py", "data_v3.py",
})
FRONT_AUX_MEAN_V3_HEAD = "5fd88852bf20c94cd74405c791a13a9fd9e0a3d8"
RR_AUX_MEAN_V4_SCHEMA = "wlr50_clean.finite_existing_mean_head_RR_aux.v4.fit"
RR_AUX_MEAN_V4_INSPECTION_SCHEMA = (
    "wlr50_clean.finite_existing_mean_head_RR_aux.v4.readonly")
RR_AUX_MEAN_V4_DATA_SCHEMA = (
    "wlr50_clean.RR_actual_raw_capture_continuation_data.v4.loaded")
RR_AUX_MEAN_V4_OBJECTIVE_SCHEMA = (
    "wlr50_clean.finite_existing_mean_head_RR_aux.v4.objective")
RR_AUX_MEAN_V4_BUDGET_SCHEMA = (
    "wlr50_clean.finite_existing_mean_head_RR_aux.v4.explicit_budget")
RR_AUX_MEAN_V4_KIND = (
    "finite_supervised_existing_mean_head_actual_RR_capture_continuation_raw_actions_not_PPO")
RR_AUX_MEAN_V4_TARGET = (
    "actually_executed_stochastic_raw12_not_stored_conditional_mean")
RR_AUX_MEAN_V4_PARAMETERS = [
    "actor.mlp.4.weight[:12,:]", "actor.mlp.4.bias[:12]"]
RR_AUX_MEAN_V4_HELPERS = frozenset({
    "rr_mean_rehearsal.py", "fit_rr_mean_core.py",
    "execute_rr_mean_rehearsal.py", "data_v4.py",
})
RR_AUX_MEAN_V4_HEAD = "5fd88852bf20c94cd74405c791a13a9fd9e0a3d8"
FPS, HZ, STRIDE, MAX_FRAMES = 15, 120, 8, 3000
SERVO_ORDER = (
    "front_left_hip", "front_left_knee", "front_right_hip", "front_right_knee",
    "rear_left_hip", "rear_left_knee", "rear_right_hip", "rear_right_knee",
)
WHEEL_ORDER = (
    "front_left_ankle", "front_right_ankle", "rear_left_ankle", "rear_right_ankle",
)
WHEEL_LABELS = ("FL", "FR", "RL", "RR")

sys.path.insert(0, str(ROOT / "src"))
from wlr50_clean.evaluation.video_timeline import decode_frame_timeline, load_viewport_frame_ledger
from wlr50_clean.infrastructure.video_capture import find_ffmpeg, validate_mp4


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_new_json(path: Path, payload: Any) -> None:
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_new_text(path: Path, value: str) -> None:
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(value)


def feedback_training_identity(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return an independent feedback-revision counter, if one is declared."""
    branch = metadata.get("capture_feedback_semantics_branch")
    if branch is None:
        require(metadata.get("capture_feedback_semantics_migration") is None
                and metadata.get("capture_feedback_semantics_branch_counts") is None,
                "checkpoint has orphaned capture-feedback migration evidence")
        return {"feedback_branch_present": False, "feedback_revision": None,
                "feedback_counter_origin": None,
                "feedback_added_policy_decisions": None, "feedback_added_ppo_updates": None,
                "feedback_added_optimizer_steps": None}
    require(isinstance(branch, dict) and branch.get("schema") == FEEDBACK_SCHEMA
            and branch.get("feedback_revision") == FEEDBACK_REVISION,
            "checkpoint has an invalid capture-feedback branch revision")
    origin = branch.get("counter_origin")
    require(isinstance(origin, dict) and set(origin) == set(COUNTERS)
            and all(type(origin[key]) is int and origin[key] >= 0 for key in COUNTERS),
            "checkpoint has a missing or invalid capture-feedback counter origin")
    counts: dict[str, int] = {}
    for key in COUNTERS:
        current = metadata.get(key)
        require(type(current) is int and current >= origin[key],
                f"checkpoint has a negative capture-feedback {key} delta")
        counts[key] = current - origin[key]
    persisted_counts = metadata.get("capture_feedback_semantics_branch_counts")
    require(isinstance(persisted_counts, dict) and set(persisted_counts) == set(COUNTERS)
            and all(type(persisted_counts[key]) is int and persisted_counts[key] >= 0
                    for key in COUNTERS) and persisted_counts == counts,
            "checkpoint capture-feedback counts do not match its independent origin")
    p05 = (metadata.get("p05_capture_assist_branch") or {}).get("counter_origin")
    require(p05 == P05_COUNTER_ORIGIN,
            "capture-feedback checkpoint changed the original P05 counter origin")
    p05_counts = {key: metadata[key] - P05_COUNTER_ORIGIN[key] for key in COUNTERS}
    persisted_p05_counts = metadata.get("p05_capture_assist_branch_counts")
    require(isinstance(persisted_p05_counts, dict) and set(persisted_p05_counts) == set(COUNTERS)
            and all(type(persisted_p05_counts[key]) is int and persisted_p05_counts[key] >= 0
                    for key in COUNTERS) and persisted_p05_counts == p05_counts,
            "capture-feedback checkpoint changed the original P05 branch counts")
    migration = metadata.get("capture_feedback_semantics_migration")
    factor = (migration or {}).get("capture_feedback_semantics_factor") or {}
    source_hash = branch.get("source_checkpoint_sha256")
    require(isinstance(migration, dict) and migration.get("schema") == FEEDBACK_SCHEMA
            and factor.get("schema") == FEEDBACK_SCHEMA
            and factor.get("target_feedback_revision") == FEEDBACK_REVISION
            and factor.get("counter_origin") == origin
            and migration.get("source_checkpoint_sha256") == source_hash
            and isinstance(source_hash, str) and len(source_hash) == 64
            and all(character in "0123456789abcdef" for character in source_hash)
            and branch.get("migration_added_updates") == 0,
            "capture-feedback branch does not bind its reviewed migration boundary")
    return {"feedback_branch_present": True, "feedback_revision": FEEDBACK_REVISION,
            "feedback_counter_origin": dict(origin),
            "feedback_added_policy_decisions": counts["global_policy_decisions"],
            "feedback_added_ppo_updates": counts["ppo_updates"],
            "feedback_added_optimizer_steps": counts["optimizer_steps"]}


def rr_workspace_training_identity(metadata: dict[str, Any]) -> dict[str, Any]:
    """Return the independent RR reward-revision lineage, if one is declared."""
    branch = metadata.get("rr_postcross_workspace_branch")
    if branch is None:
        require(metadata.get("rr_postcross_workspace_migration") is None
                and metadata.get("rr_postcross_workspace_branch_counts") is None,
                "checkpoint has orphaned RR workspace migration evidence")
        return {"rr_workspace_branch_present": False, "rr_workspace_semantics": None,
                "rr_workspace_counter_origin": None,
                "rr_workspace_added_policy_decisions": None,
                "rr_workspace_added_ppo_updates": None,
                "rr_workspace_added_optimizer_steps": None}
    require(isinstance(branch, dict) and branch.get("schema") == RR_WORKSPACE_SCHEMA
            and branch.get("semantics") == RR_WORKSPACE_SEMANTICS,
            "checkpoint has an invalid RR workspace branch semantics")
    require(feedback_training_identity(metadata)["feedback_branch_present"] is True,
            "RR workspace checkpoint lost its required feedback-v2 lineage")
    origin = branch.get("counter_origin")
    require(isinstance(origin, dict) and set(origin) == set(COUNTERS)
            and all(type(origin[key]) is int and origin[key] >= 0 for key in COUNTERS),
            "checkpoint has a missing or invalid RR workspace counter origin")
    counts: dict[str, int] = {}
    for key in COUNTERS:
        current = metadata.get(key)
        require(type(current) is int and current >= origin[key],
                f"checkpoint has a negative RR workspace {key} delta")
        counts[key] = current - origin[key]
    persisted_counts = metadata.get("rr_postcross_workspace_branch_counts")
    require(isinstance(persisted_counts, dict) and set(persisted_counts) == set(COUNTERS)
            and all(type(persisted_counts[key]) is int and persisted_counts[key] >= 0
                    for key in COUNTERS) and persisted_counts == counts,
            "checkpoint RR workspace counts do not match its independent origin")

    migration = metadata.get("rr_postcross_workspace_migration")
    factor = (migration or {}).get("rr_postcross_workspace_factor") or {}
    source_hash = branch.get("source_checkpoint_sha256")
    observation_changes = factor.get("observation_semantics_changed")
    require(isinstance(migration, dict) and migration.get("schema") == RR_WORKSPACE_SCHEMA
            and factor.get("schema") == RR_WORKSPACE_SCHEMA
            and factor.get("target_semantics") == RR_WORKSPACE_SEMANTICS
            and factor.get("source_semantics") == RR_WORKSPACE_SOURCE_SEMANTICS
            and factor.get("task_spec_change") == {
                RR_WORKSPACE_MODE_KEY: RR_WORKSPACE_SEMANTICS}
            and factor.get("counter_origin") == origin
            and migration.get("source_checkpoint_sha256") == source_hash
            and isinstance(source_hash, str) and len(source_hash) == 64
            and all(character in "0123456789abcdef" for character in source_hash)
            and branch.get("migration_added_updates") == 0,
            "RR workspace branch does not bind its reviewed source and migration semantics")
    source_checkpoint = Path(migration.get("source_checkpoint", "")).resolve(strict=True)
    source_manifest = source_checkpoint.with_name(source_checkpoint.stem + "_manifest.json")
    require(source_checkpoint.is_file() and source_manifest.is_file()
            and sha256(source_checkpoint) == source_hash
            and sha256(source_manifest) == migration.get("source_manifest_sha256"),
            "RR workspace historical source checkpoint or sidecar binding changed")
    source_metadata = read_json(source_manifest)
    require(Path(source_metadata.get("checkpoint_path", "")).resolve() == source_checkpoint
            and source_metadata.get("checkpoint_sha256") == source_hash
            and all(source_metadata.get(key) == origin[key] for key in COUNTERS)
            and source_metadata.get("rr_postcross_workspace_branch") is None
            and source_metadata.get("rr_postcross_workspace_migration") is None
            and source_metadata.get("rr_postcross_workspace_branch_counts") is None
            and feedback_training_identity(source_metadata)["feedback_branch_present"] is True,
            "RR workspace historical source is not the bound pre-revision lineage")
    preserved = factor.get("preserved_metadata_sha256")
    require(isinstance(preserved, dict) and RR_WORKSPACE_REQUIRED_HISTORY <= set(preserved)
            and all(key in source_metadata and json_digest(source_metadata[key]) == digest
                    for key, digest in preserved.items()),
            "RR workspace historical source metadata does not match the migration factor")
    observation_contract = factor.get("observation_contract") or {}
    require(observation_contract.get("observation_dimension") == 389
            and observation_contract.get("action_dimension") == 12
            and observation_contract.get("observation_layout") == "role372_p05_capture_assist_v1"
            and isinstance(observation_changes, list) and len(observation_changes) == 1
            and observation_changes[0].get("index") == 17
            and observation_changes[0].get("field") == "task_progress_potential",
            "RR workspace migration does not bind the same389 index17 semantic change")
    required_flags = {
        "reward_changed": True,
        "same_mdp_claimed": False,
        "physical_dynamics_changed": False,
        "same_numeric_input_policy_mapping_preserved": True,
        "same_physical_state_action_equivalence_claimed": False,
        "policy_kernel_changed": False,
        "controller_predicates_changed": False,
        "nominal_changed": False,
        "capture_assist_changed": False,
        "caps_changed": False,
        "sigma_changed": False,
        "reward_coefficients_and_return_profile_changed": False,
        "old_values_are_new_reward_ground_truth": False,
        "discard_old_rollout_storage": True,
        "physical_state_inherited": False,
        "added_policy_decisions": 0,
        "added_ppo_updates": 0,
        "added_optimizer_steps": 0,
        "added_auxiliary_updates": 0,
    }
    require(all(factor.get(key) == expected for key, expected in required_flags.items()),
            "RR workspace migration makes an unsupported reward/controller/MDP claim")
    return {"rr_workspace_branch_present": True,
            "rr_workspace_semantics": RR_WORKSPACE_SEMANTICS,
            "rr_workspace_counter_origin": dict(origin),
            "rr_workspace_added_policy_decisions": counts["global_policy_decisions"],
            "rr_workspace_added_ppo_updates": counts["ppo_updates"],
            "rr_workspace_added_optimizer_steps": counts["optimizer_steps"]}


def rr_reward_training_identity(metadata: dict[str, Any]) -> dict[str, Any]:
    """Validate the active RR reward revision while retaining immutable v1 lineage.

    The optional v2 migration hashes its *migration-time* source metadata.  Later
    PPO checkpoints legitimately have different actor/critic/Adam/counters, so
    those hashes are checked against the bound historical source sidecar, never
    against the current learned checkpoint.
    """
    v1 = rr_workspace_training_identity(metadata)
    branch = metadata.get(RR_RECEIVER_V2_BRANCH)
    if branch is None:
        require(metadata.get(RR_RECEIVER_V2_MIGRATION) is None
                and metadata.get(RR_RECEIVER_V2_COUNTS) is None,
                "checkpoint has orphaned RR receiver-v2 migration evidence")
        return v1
    require(v1["rr_workspace_branch_present"] is True,
            "RR receiver-v2 checkpoint lost its immutable v1 RR lineage")
    require(isinstance(branch, dict)
            and branch.get("schema") == RR_RECEIVER_V2_SCHEMA
            and branch.get("semantics") == RR_RECEIVER_V2_SEMANTICS
            and branch.get("migration_added_updates") == 0,
            "checkpoint has an invalid RR receiver-v2 branch semantics")
    origin = branch.get("counter_origin")
    require(isinstance(origin, dict) and set(origin) == set(COUNTERS)
            and all(type(origin[key]) is int and origin[key] >= 0 for key in COUNTERS),
            "checkpoint has a missing or invalid RR receiver-v2 counter origin")
    counts: dict[str, int] = {}
    for key in COUNTERS:
        current = metadata.get(key)
        require(type(current) is int and current >= origin[key],
                f"checkpoint has a negative RR receiver-v2 {key} delta")
        counts[key] = current - origin[key]
    persisted_counts = metadata.get(RR_RECEIVER_V2_COUNTS)
    require(isinstance(persisted_counts, dict) and set(persisted_counts) == set(COUNTERS)
            and all(type(persisted_counts[key]) is int and persisted_counts[key] >= 0
                    for key in COUNTERS) and persisted_counts == counts,
            "checkpoint RR receiver-v2 counts do not match its independent origin")

    migration = metadata.get(RR_RECEIVER_V2_MIGRATION)
    factor = (migration or {}).get("rr_postcross_workspace_factor") or {}
    source_hash = branch.get("source_checkpoint_sha256")
    require(isinstance(migration, dict)
            and migration.get("schema") == RR_RECEIVER_V2_SCHEMA
            and factor.get("schema") == RR_RECEIVER_V2_SCHEMA
            and factor.get("source_semantics") == RR_WORKSPACE_SEMANTICS
            and factor.get("target_semantics") == RR_RECEIVER_V2_SEMANTICS
            and factor.get("task_spec_change") == {
                RR_WORKSPACE_MODE_KEY: RR_RECEIVER_V2_SEMANTICS}
            and factor.get("counter_origin") == origin
            and migration.get("source_checkpoint_sha256") == source_hash
            and isinstance(source_hash, str) and len(source_hash) == 64
            and all(character in "0123456789abcdef" for character in source_hash),
            "RR receiver-v2 branch does not bind its reviewed v1-to-v2 boundary")
    source_checkpoint = Path(migration.get("source_checkpoint", "")).resolve(strict=True)
    source_manifest = source_checkpoint.with_name(source_checkpoint.stem + "_manifest.json")
    require(source_checkpoint.is_file() and source_manifest.is_file()
            and sha256(source_checkpoint) == source_hash
            and sha256(source_manifest) == migration.get("source_manifest_sha256"),
            "RR receiver-v2 historical source checkpoint or sidecar binding changed")
    source_metadata = read_json(source_manifest)
    source_v1 = rr_workspace_training_identity(source_metadata)
    require(source_v1["rr_workspace_branch_present"] is True
            and Path(source_metadata.get("checkpoint_path", "")).resolve() == source_checkpoint
            and source_metadata.get("checkpoint_sha256") == source_hash
            and all(source_metadata.get(key) == origin[key] for key in COUNTERS)
            and source_metadata.get(RR_RECEIVER_V2_BRANCH) is None
            and source_metadata.get(RR_RECEIVER_V2_MIGRATION) is None
            and source_metadata.get(RR_RECEIVER_V2_COUNTS) is None,
            "RR receiver-v2 historical source is not the bound pre-v2 lineage")
    for key in (
            "rr_postcross_workspace_branch", "rr_postcross_workspace_migration",
            "p05_capture_assist_branch", "p05_capture_assist_migration",
            "capture_feedback_semantics_branch", "capture_feedback_semantics_migration",
            "task_conditioned_hip_wheel_branch"):
        require(metadata.get(key) == source_metadata.get(key),
                f"RR receiver-v2 checkpoint changed retained migration-time lineage: {key}")
    preserved = factor.get("preserved_metadata_sha256")
    required_preserved = RR_WORKSPACE_REQUIRED_HISTORY | {
        "rr_postcross_workspace_branch", "rr_postcross_workspace_migration",
        "rr_postcross_workspace_branch_counts"}
    require(isinstance(preserved, dict) and required_preserved <= set(preserved)
            and all(key in source_metadata and json_digest(source_metadata[key]) == digest
                    for key, digest in preserved.items()),
            "RR receiver-v2 migration-time source metadata differs from its factor")
    observation_contract = factor.get("observation_contract") or {}
    observation_changes = factor.get("observation_semantics_changed")
    require(observation_contract.get("observation_dimension") == 389
            and observation_contract.get("action_dimension") == 12
            and observation_contract.get("observation_layout")
                == "role372_p05_capture_assist_v1"
            and isinstance(observation_changes, list) and len(observation_changes) == 1
            and observation_changes[0].get("index") == 17
            and observation_changes[0].get("field") == "task_progress_potential",
            "RR receiver-v2 migration does not bind its same389 index17 semantics")
    required_flags = {
        "reward_changed": True,
        "same_mdp_claimed": False,
        "physical_dynamics_changed": False,
        "same_numeric_input_policy_mapping_preserved": True,
        "same_physical_state_action_equivalence_claimed": False,
        "policy_kernel_changed": False,
        "controller_predicates_changed": False,
        "nominal_changed": False,
        "capture_assist_changed": False,
        "caps_changed": False,
        "sigma_changed": False,
        "reward_coefficients_and_return_profile_changed": False,
        "old_values_are_new_reward_ground_truth": False,
        "discard_old_rollout_storage": True,
        "physical_state_inherited": False,
        "added_policy_decisions": 0,
        "added_ppo_updates": 0,
        "added_optimizer_steps": 0,
        "added_auxiliary_updates": 0,
    }
    require(all(factor.get(key) == expected for key, expected in required_flags.items()),
            "RR receiver-v2 migration makes an unsupported reward/controller/MDP claim")
    return {
        **v1,
        "rr_workspace_v1_semantics": v1["rr_workspace_semantics"],
        "rr_workspace_v1_counter_origin": v1["rr_workspace_counter_origin"],
        "rr_workspace_v1_added_policy_decisions": v1[
            "rr_workspace_added_policy_decisions"],
        "rr_workspace_v1_added_ppo_updates": v1["rr_workspace_added_ppo_updates"],
        "rr_workspace_v1_added_optimizer_steps": v1[
            "rr_workspace_added_optimizer_steps"],
        "rr_receiver_v2_branch_present": True,
        "rr_receiver_v2_semantics": RR_RECEIVER_V2_SEMANTICS,
        "rr_receiver_v2_counter_origin": dict(origin),
        "rr_receiver_v2_added_policy_decisions": counts["global_policy_decisions"],
        "rr_receiver_v2_added_ppo_updates": counts["ppo_updates"],
        "rr_receiver_v2_added_optimizer_steps": counts["optimizer_steps"],
        # Existing overlay fields intentionally name the active reward revision.
        "rr_workspace_semantics": RR_RECEIVER_V2_SEMANTICS,
        "rr_workspace_counter_origin": dict(origin),
        "rr_workspace_added_policy_decisions": counts["global_policy_decisions"],
        "rr_workspace_added_ppo_updates": counts["ppo_updates"],
        "rr_workspace_added_optimizer_steps": counts["optimizer_steps"],
    }


def p05_preedge_training_identity(metadata: dict[str, Any]) -> dict[str, Any]:
    """Validate the optional P05 control-MDP revision and its source boundary.

    Absence returns no fields so legacy receipts and overlays retain their prior
    shape.  Presence requires the complete branch/migration/counts triplet.
    """
    branch = metadata.get(P05_PREEDGE_BRANCH)
    migration = metadata.get(P05_PREEDGE_MIGRATION)
    persisted_counts = metadata.get(P05_PREEDGE_COUNTS)
    if branch is None:
        require(migration is None and persisted_counts is None,
                "checkpoint has orphaned P05 pre-edge recovery evidence")
        return {}
    require(isinstance(branch, dict) and isinstance(migration, dict)
            and isinstance(persisted_counts, dict),
            "P05 pre-edge recovery requires branch/migration/counts together")
    origin = branch.get("counter_origin")
    require(branch.get("schema") == P05_PREEDGE_SCHEMA
            and branch.get("semantics") == P05_PREEDGE_SEMANTICS
            and branch.get("migration_added_updates") == 0
            and isinstance(origin, dict) and set(origin) == set(COUNTERS)
            and all(type(origin[key]) is int and origin[key] >= 0 for key in COUNTERS),
            "checkpoint has an invalid P05 pre-edge recovery branch")
    counts: dict[str, int] = {}
    for key in COUNTERS:
        current = metadata.get(key)
        require(type(current) is int and current >= origin[key],
                f"checkpoint has a negative P05 pre-edge {key} delta")
        counts[key] = current - origin[key]
    require(set(persisted_counts) == set(COUNTERS)
            and all(type(persisted_counts[key]) is int and persisted_counts[key] >= 0
                    for key in COUNTERS) and persisted_counts == counts,
            "checkpoint P05 pre-edge counts do not match their source origin")

    factor = migration.get(P05_PREEDGE_FACTOR_KEY) or {}
    source_hash = branch.get("source_checkpoint_sha256")
    reviewed = factor.get("reviewed_code_sha256")
    changed = migration.get("changed_file_hashes")
    require(migration.get("schema") == P05_PREEDGE_SCHEMA
            and factor.get("schema") == P05_PREEDGE_SCHEMA
            and isinstance(factor.get("review_reason"), str)
            and bool(factor["review_reason"].strip())
            and migration.get("reason") == factor["review_reason"]
            and factor.get("source_semantics") == P05_PREEDGE_SOURCE_SEMANTICS
            and factor.get("target_semantics") == P05_PREEDGE_SEMANTICS
            and factor.get("task_spec_change") == {
                "nominal": {P05_PREEDGE_MODE_KEY: P05_PREEDGE_SEMANTICS}}
            and factor.get("counter_origin") == origin
            and migration.get("source_checkpoint_sha256") == source_hash
            and isinstance(source_hash, str) and len(source_hash) == 64
            and all(character in "0123456789abcdef" for character in source_hash),
            "P05 pre-edge recovery does not bind its reviewed source boundary")
    require(isinstance(reviewed, dict)
            and P05_PREEDGE_REQUIRED_CHANGED_FILES <= set(reviewed)
            and set(reviewed) <= P05_PREEDGE_ALLOWED_CHANGED_FILES
            and all(isinstance(digest, str) and len(digest) == 64
                    and all(character in "0123456789abcdef" for character in digest)
                    for digest in reviewed.values())
            and migration.get("allowed_changed_files") == sorted(reviewed)
            and isinstance(changed, dict) and set(changed) == set(reviewed)
            and all(isinstance(changed[path], dict)
                    and changed[path].get("after") == reviewed[path]
                    for path in reviewed)
            and all(isinstance(factor.get(key), str) and len(factor[key]) == 64
                    for key in ("source_task_spec_sha256", "target_task_spec_sha256"))
            and all(isinstance(migration.get(key), str) and len(migration[key]) == 64
                    for key in ("source_contract_sha256", "target_contract_sha256",
                                "source_runtime_content_sha256",
                                "target_runtime_content_sha256"))
            and isinstance(migration.get("source_git_commit"), str)
            and isinstance(migration.get("target_git_commit"), str)
            and migration["source_git_commit"] != migration["target_git_commit"],
            "P05 pre-edge reviewed code/contract boundary is incomplete")
    source_checkpoint = Path(migration.get("source_checkpoint", "")).resolve(strict=True)
    source_manifest = source_checkpoint.with_name(source_checkpoint.stem + "_manifest.json")
    require(source_checkpoint.is_file() and source_manifest.is_file()
            and sha256(source_checkpoint) == source_hash
            and sha256(source_manifest) == migration.get("source_manifest_sha256"),
            "P05 pre-edge historical source checkpoint or sidecar binding changed")
    source_metadata = read_json(source_manifest)
    require(Path(source_metadata.get("checkpoint_path", "")).resolve() == source_checkpoint
            and source_metadata.get("checkpoint_sha256") == source_hash
            and all(source_metadata.get(key) == origin[key] for key in COUNTERS)
            and source_metadata.get(P05_PREEDGE_BRANCH) is None
            and source_metadata.get(P05_PREEDGE_MIGRATION) is None
            and source_metadata.get(P05_PREEDGE_COUNTS) is None,
            "P05 pre-edge historical source is not the bound pre-revision checkpoint")

    expected_preserved = set(RR_WORKSPACE_REQUIRED_HISTORY) | {
        key for key in source_metadata
        if ((key != "resume_migration" and key.endswith(
                ("_branch", "_branch_counts", "_migration")))
            or key in ("source_stage_requested_decisions",
                       "training_quantity_budget_extension"))
    }
    preserved = factor.get("preserved_metadata_sha256")
    require(isinstance(preserved, dict) and set(preserved) == expected_preserved
            and all(key in source_metadata and json_digest(source_metadata[key]) == digest
                    for key, digest in preserved.items()),
            "P05 pre-edge migration-time source metadata differs from its factor")
    require(factor.get("source_resume_migration") == source_metadata.get("resume_migration")
            and factor.get("source_resume_migration_sha256")
                == json_digest(source_metadata.get("resume_migration")),
            "P05 pre-edge source resume lineage is not hash-bound")

    for name in P05_PREEDGE_PRIOR_BRANCHES:
        branch_key, migration_key, counts_key = (
            name + "_branch", name + "_migration", name + "_branch_counts")
        prior = source_metadata.get(branch_key)
        prior_origin = (prior or {}).get("counter_origin")
        require(isinstance(prior, dict) and isinstance(prior_origin, dict)
                and set(prior_origin) == set(COUNTERS)
                and metadata.get(branch_key) == prior
                and metadata.get(migration_key) == source_metadata.get(migration_key)
                and source_metadata.get(counts_key) == {
                    key: source_metadata[key] - prior_origin[key] for key in COUNTERS}
                and metadata.get(counts_key) == {
                    key: metadata[key] - prior_origin[key] for key in COUNTERS},
                f"P05 pre-edge checkpoint changed prior lineage/origin: {name}")
    source_rr = source_metadata["rr_postcross_workspace_branch"]
    current_rr = metadata["rr_postcross_workspace_branch"]
    source_aux = source_rr.get(FRONT_AUX_KEY)
    historical_aux = ((source_metadata.get("task_conditioned_hip_wheel_branch") or {})
                      .get("auxiliary_mean_learning") or {})
    require(metadata.get("task_conditioned_hip_wheel_branch")
                == source_metadata.get("task_conditioned_hip_wheel_branch")
            and current_rr.get(FRONT_AUX_KEY) == source_aux
            and isinstance(source_aux, dict)
            and source_aux.get("schema") == FRONT_AUX_SCHEMA
            and isinstance(source_aux.get("events"), list)
            and len(source_aux["events"]) >= 4
            and historical_aux.get("accepted_auxiliary_updates_total") == 7
            and historical_aux.get("attempted_auxiliary_optimizer_steps_total") == 8,
            "P05 pre-edge checkpoint lost or rewrote its four-event/new or 7/8 historical AUX")

    observation = factor.get("observation_contract") or {}
    required_flags = {
        "observation_shape_changed": False,
        "observation_codec_changed": False,
        "observation_semantics_changed": [],
        "same_numeric_input_policy_mapping_preserved": True,
        "same_physical_state_action_equivalence_claimed": False,
        "controller_transition_semantics_changed": True,
        "nominal_changed": True,
        "same_mdp_claimed": False,
        "reward_changed": False,
        "physical_dynamics_changed": False,
        "policy_kernel_changed": False,
        "capture_assist_changed": False,
        "caps_changed": False,
        "sigma_changed": False,
        "physical_task_acceptance_rules_changed": False,
        "added_mutable_state": False,
        "discard_old_rollout_storage": True,
        "physical_state_inherited": False,
        "added_policy_decisions": 0,
        "added_ppo_updates": 0,
        "added_optimizer_steps": 0,
        "added_auxiliary_updates": 0,
    }
    require(observation.get("observation_layout") == "role372_p05_capture_assist_v1"
            and observation.get("observation_dimension") == 389
            and observation.get("action_dimension") == 12
            and observation.get("num_envs") == 1
            and observation.get("parameter_mapping")
                == "identity_all_parameters_and_buffers"
            and factor.get("affected_control_phases") == ["P05"]
            and factor.get("parameter_mapping") == "identity_all_parameters_and_buffers"
            and factor.get("optimizer_mapping")
                == "identity_all_Adam_moments_steps_groups_and_effective_LR"
            and factor.get("normalizer_mapping") == "identity_Identity"
            and factor.get("rng_mapping") == "restore_exact_source_training_rng"
            and factor.get("source_effective_learning_rate")
                == source_metadata.get("optimizer_learning_rate")
            and factor.get("target_effective_learning_rate")
                == source_metadata.get("optimizer_learning_rate")
            and all(factor.get(key) == expected
                    for key, expected in required_flags.items())
            and migration.get("observation_dimension") == 389
            and migration.get("action_dimension") == 12
            and migration.get("preserve_actor_critic_optimizer_normalizer_rng_and_budget") is True
            and migration.get("discard_old_rollout_storage") is True
            and migration.get("physics_resume") == "fresh_legal_P01_reset",
            "P05 pre-edge migration overstates a pure-policy/same-MDP transition")
    return {
        "p05_preedge_branch_present": True,
        "p05_preedge_semantics": P05_PREEDGE_SEMANTICS,
        "p05_preedge_counter_origin": dict(origin),
        "p05_preedge_added_policy_decisions": counts["global_policy_decisions"],
        "p05_preedge_added_ppo_updates": counts["ppo_updates"],
        "p05_preedge_added_optimizer_steps": counts["optimizer_steps"],
        "p05_preedge_migration_added_updates": 0,
        "p05_preedge_is_control_mdp_revision": True,
        "p05_preedge_is_pure_policy": False,
    }


def _front_aux_prefix(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not events:
        return None
    return {
        "schema": FRONT_AUX_SCHEMA,
        "events": events,
        "accepted_auxiliary_updates_total": sum(
            event["fit_report"]["accepted_auxiliary_updates"] for event in events),
        "attempted_auxiliary_optimizer_steps_total": sum(
            event["fit_report"]["attempted_auxiliary_optimizer_steps"] for event in events),
        "training_lineage_label": FRONT_AUX_LINEAGE_LABEL,
    }


def _front_aux_bound_file(path_value: Any, digest: Any, description: str, *,
                          expected: Path | None = None) -> Path:
    require(isinstance(path_value, str) and isinstance(digest, str) and len(digest) == 64,
            f"{description} lacks a path/SHA256 binding")
    path = Path(path_value).resolve(strict=True)
    require(path.is_file() and path.is_relative_to(OUTPUT_ROOT.resolve())
            and (expected is None or path == expected.resolve()) and sha256(path) == digest,
            f"{description} bytes or path changed")
    return path


def _front_aux_v1_event(event: dict[str, Any], source_metadata: dict[str, Any]) -> None:
    helper_hashes = event.get("helper_sha256")
    require(isinstance(helper_hashes, dict) and set(helper_hashes) == FRONT_AUX_HELPERS
            and all(isinstance(value, str) and len(value) == 64
                    and sha256(OUTPUT_ROOT / "front_rehearsal_v1" / name) == value
                    for name, value in helper_hashes.items()),
            "front-rehearsal v1 helper hashes differ from the bound reviewed helpers")
    data = event.get("data_receipt")
    require(isinstance(data, dict) and data.get("schema") == FRONT_AUX_DATA_SCHEMA
            and isinstance(data.get("receipt_content_sha256"), str),
            "front-rehearsal v1 data receipt schema is invalid")
    data_without_digest = dict(data)
    data_digest = data_without_digest.pop("receipt_content_sha256")
    groups = data.get("groups") or {}
    require(json_digest(data_without_digest) == data_digest
            and data.get("front_source_count") == 280
            and data.get("raw_targets_unmodified") is True
            and data.get("stored_mean_is_diagnostic_not_executed_target") is True
            and data.get("teacher_deployed") is False
            and data.get("new_PPO_credit") == 0 and data.get("new_auxiliary_credit") == 0
            and set(groups) == {"train", "validation", "invariance"}
            and groups["train"].get("count") == 95
            and groups["validation"].get("count") == 93
            and groups["invariance"].get("count") == 13,
            "front-rehearsal v1 data receipt changed its reviewed raw-action selection")
    source_files = data.get("source_files")
    require(isinstance(source_files, dict) and bool(source_files),
            "front-rehearsal v1 data receipt has no source-file bindings")
    for record in source_files.values():
        require(isinstance(record, dict) and set(record) == {"path", "sha256"},
                "front-rehearsal v1 data source-file binding is malformed")
        data_path = Path(record["path"]).resolve(strict=True)
        require(data_path.is_file() and sha256(data_path) == record["sha256"],
                "front-rehearsal v1 reviewed data source bytes changed")

    report = event.get("fit_report")
    accepted = report.get("accepted_auxiliary_updates") if isinstance(report, dict) else None
    attempted = report.get("attempted_auxiliary_optimizer_steps") if isinstance(report, dict) else None
    require(isinstance(report, dict) and report.get("schema") == FRONT_AUX_REPORT_SCHEMA
            and json_digest(report) == event.get("fit_report_sha256")
            and report.get("budget") == event.get("budget")
            and type(accepted) is int and type(attempted) is int
            and 0 < accepted <= attempted <= 32
            and isinstance(report.get("steps"), list) and len(report["steps"]) == attempted
            and sum(step.get("accepted") is True for step in report["steps"]) == accepted
            and report.get("optimized_parameters") == FRONT_AUX_PARAMETERS
            and report.get("optimized_scalar_count") == 512
            and report.get("P01_P02_mean_and_sigma_may_both_change") is True
            and report.get("kernel_or_sigma_rule_changed") is False
            and report.get("MDP_or_control_changed") is False
            and report.get("PPO_decisions_added") == 0
            and report.get("PPO_updates_added") == 0
            and report.get("PPO_optimizer_steps_added") == 0
            and report.get("teacher_deployed") is False
            and report.get("physical_success_claimed") is False
            and report.get("fresh_PPO_rollout_required") is True
            and report.get("same_input_only_not_same_future_state_trajectory") is True
            and report.get("actual_invariance_phase_ids") == [3, 4, 5, 6]
            and report.get("synthetic_math_probe_phases") == list(range(3, 14))
            and report.get("actor_parameter_sha256_before")
                == source_metadata.get("actor_parameter_sha256")
            and report.get("PPO_Adam_preserved_sha256")
                == source_metadata.get("optimizer_state_sha256")
            and report.get("PPO_LR_preserved")
                == source_metadata.get("optimizer_learning_rate"),
            "front-rehearsal v1 fit report is not the bound finite non-PPO mu/sigma update")


def _front_aux_det_p02_helper_path(name: str) -> Path:
    if name in {"execute_det_rehearsal.py", "inspect_det_rehearsal.py"}:
        return OUTPUT_ROOT / "front_rehearsal_det_v2" / name
    return OUTPUT_ROOT / name


def _front_aux_det_p02_event(event: dict[str, Any], source_metadata: dict[str, Any],
                             source_path: Path) -> None:
    helper_hashes = event.get("helper_sha256")
    require(isinstance(helper_hashes, dict) and set(helper_hashes) == FRONT_AUX_DET_P02_HELPERS
            and all(sha256(_front_aux_det_p02_helper_path(name)) == digest
                    for name, digest in helper_hashes.items()),
            "det-P02 AUX helper hashes differ from the reviewed execution boundary")
    binding = event.get("binding")
    require(isinstance(binding, dict) and set(binding) == {
                "source_checkpoint", "helpers", "inspection", "data_receipt_sha256",
                "data_manifest_sha256", "admission", "expected_head"}
            and binding.get("source_checkpoint") == event.get("source_checkpoint")
            and binding.get("helpers") == helper_hashes
            and binding.get("data_receipt_sha256") == json_digest(event.get("data_receipt"))
            and binding.get("expected_head")
                == (source_metadata.get("runtime_contract") or {}).get("source_git_commit"),
            "det-P02 AUX binding is incomplete or does not bind its source/data/helpers")

    inspection_record = binding.get("inspection") or {}
    require(set(inspection_record) == {"path", "sha256", "schema"}
            and inspection_record.get("schema") == FRONT_AUX_DET_P02_INSPECTION_SCHEMA,
            "det-P02 AUX inspection receipt binding is malformed")
    inspection_path = _front_aux_bound_file(
        inspection_record.get("path"), inspection_record.get("sha256"),
        "det-P02 readonly inspection")
    require(inspection_path.parent == (OUTPUT_ROOT / "front_rehearsal_det_v2").resolve(),
            "det-P02 readonly inspection is outside its isolated directory")
    inspection = read_json(inspection_path)
    inspection_binding = inspection.get("binding") or {}
    require(inspection.get("schema") == FRONT_AUX_DET_P02_INSPECTION_SCHEMA
            and inspection.get("inspection_only") is True
            and inspection.get("automatic_aux_enabled") is False
            and inspection.get("current_MDP_training_admission_claimed") is False
            and inspection.get("optimizer_steps_performed") == 0
            and inspection.get("auxiliary_updates_added") == 0
            and inspection.get("checkpoint_written") is False
            and inspection.get("teacher_deployed") is False
            and inspection.get("physical_success_claimed") is False
            and inspection.get("source_PPO_counters_unchanged")
                == event.get("PPO_counters_unchanged")
            and inspection.get("source_actor_parameter_sha256")
                == source_metadata.get("actor_parameter_sha256")
            and inspection.get("reconstruction_receipt") == event.get("data_receipt")
            and inspection.get("P01_gradient_column_exact_zero") is True
            and inspection.get("P02_mean_and_log_sigma_may_both_change") is True
            and inspection.get("same_input_invariance_is_not_future_trajectory_invariance") is True
            and all((inspection.get("protected_full_Gaussian_algebraic_probe_bitwise_equal")
                     or {}).values())
            and inspection_binding.get("source_checkpoint") == str(source_path)
            and inspection_binding.get("source_sha256")
                == event["source_checkpoint"]["sha256"]
            and inspection_binding.get("source_manifest_sha256")
                == event["source_checkpoint"]["manifest_sha256"]
            and inspection_binding.get("inspection_helper_sha256")
                == helper_hashes["inspect_det_rehearsal.py"]
            and inspection_binding.get("data_sha256") == binding.get("data_manifest_sha256"),
            "det-P02 readonly inspection is stale or overclaims reconstruction/learning")

    admission_record = binding.get("admission") or {}
    require(set(admission_record) == {
                "path", "sha256", "schema", "result", "rows_checked",
                "current_checkpoint_sha256", "candidate_manifest_sha256",
                "candidate_npz_sha256", "audit_script", "inspection_receipt",
                "restrictions", "admission_itself_authorizes_optimization"}
            and admission_record.get("schema") == FRONT_AUX_DET_P02_ADMISSION_SCHEMA
            and admission_record.get("result") == FRONT_AUX_DET_P02_ADMISSION_RESULT
            and admission_record.get("rows_checked") == 254
            and admission_record.get("current_checkpoint_sha256")
                == event["source_checkpoint"]["sha256"]
            and admission_record.get("candidate_manifest_sha256")
                == binding.get("data_manifest_sha256")
            and admission_record.get("admission_itself_authorizes_optimization") is False,
            "det-P02 admission receipt is missing its bounded non-authorization semantics")
    admission_path = _front_aux_bound_file(
        admission_record.get("path"), admission_record.get("sha256"),
        "det-P02 current-semantics admission",
        expected=OUTPUT_ROOT / "det_P02_current_semantics_admission.json")
    audit = admission_record.get("audit_script") or {}
    audit_path = _front_aux_bound_file(
        audit.get("path"), audit.get("sha256"), "det-P02 admission audit",
        expected=OUTPUT_ROOT / "audit_det_P02_current_semantics.py")
    require(admission_record.get("inspection_receipt") == {
                "path": str(inspection_path), "sha256": inspection_record["sha256"]}
            and audit_path.is_file(),
            "det-P02 admission does not bind the reviewed inspection/audit")
    admission = read_json(admission_path)
    required_admission_flags = (
        "both_immutable_plans_and_pure_strict_factors_validated",
        "policy_kernel_action_caps_sigma_contract_equal", "normalization_semantics_equal",
        "capture_numeric_encoder_AST_equal", "physical_runtime_profile_equal",
        "all_float32_X17_exact", "no_data_or_existing_helper_mutation",
    )
    require(admission.get("schema") == FRONT_AUX_DET_P02_ADMISSION_SCHEMA
            and admission.get("result") == FRONT_AUX_DET_P02_ADMISSION_RESULT
            and admission.get("rows_checked") == 254
            and admission.get("current_checkpoint_sha256")
                == event["source_checkpoint"]["sha256"]
            and admission.get("candidate_manifest_sha256") == binding.get("data_manifest_sha256")
            and admission.get("candidate_npz_sha256")
                == admission_record.get("candidate_npz_sha256")
            and all(admission.get(key) is True for key in required_admission_flags)
            and admission.get("maximum_source_phi_abs_error") == 0
            and admission.get("maximum_old_current_phi_abs_difference") == 0
            and admission.get("RR_retirement_active_inputs") == 0
            and admission.get("capture_HOLD_to_AIR_eligible_inputs") == 0
            and admission.get("AUX_updates") == 0 and admission.get("PPO_updates") == 0,
            "det-P02 admission bytes do not prove the bounded unchanged current semantics")

    data = event.get("data_receipt") or {}
    deterministic = data.get("deterministic_source") or {}
    checks = deterministic.get("checks") or {}
    dataset = deterministic.get("dataset") or {}
    provenance = deterministic.get("observation_provenance") or {}
    learning = deterministic.get("learning") or {}
    read_validation = deterministic.get("read_validation") or {}
    require(data.get("candidate_manifest_sha256") == binding.get("data_manifest_sha256")
            and data.get("current_MDP_training_admission_claimed") is False
            and data.get("numeric_input_comparison_only_not_direct_exact389") is True
            and dataset.get("rows") == 254 and dataset.get("phase") == "P02"
            and dataset.get("shapes", {}).get("X389_reconstructed") == [254, 389]
            and dataset.get("shapes", {}).get("raw12_actual_deterministic") == [254, 12]
            and checks.get("all_raw_labels_exactly_recorded_mu_and_executed_raw") is True
            and checks.get("all_previous_raw_and_assist_features_exact") is True
            and checks.get("all_history_center_exact") is True
            and provenance.get("direct389_tensor_was_saved_by_source") is False
            and provenance.get("reconstructed_from_explicit_synchronized_fields") is True
            and provenance.get("missing_fields_guessed_or_filled") is False
            and learning.get("AUX_updates") == 0
            and learning.get("PPO_decisions_added") == 0
            and learning.get("PPO_updates_added") == 0
            and learning.get("checkpoint_written") is False
            and learning.get("current_policy_compatibility_or_training_admission_claimed") is False
            and len(data.get("train_indices") or []) == 85
            and len(data.get("validation_indices") or []) == 85
            and len(data.get("source_only_indices") or []) == 84,
            "det-P02 data receipt overstates its reconstructed deterministic evidence")
    candidate_path = _front_aux_bound_file(
        read_validation.get("manifest_path"), read_validation.get("manifest_sha256"),
        "det-P02 reconstruction candidate",
        expected=OUTPUT_ROOT / "det_front_rehearsal_data_v1" / "candidate_manifest.json")
    require(read_validation.get("manifest_sha256") == binding.get("data_manifest_sha256")
            and read_validation.get("verified_rows") == 254
            and candidate_path.is_file(),
            "det-P02 data receipt does not bind all reconstructed candidate rows")

    budget_receipt = event.get("budget_receipt") or {}
    require(set(budget_receipt) == {"path", "sha256", "envelope"},
            "det-P02 explicit budget receipt is malformed")
    budget_path = _front_aux_bound_file(
        budget_receipt.get("path"), budget_receipt.get("sha256"), "det-P02 explicit budget")
    envelope = budget_receipt.get("envelope") or {}
    require(read_json(budget_path) == envelope
            and set(envelope) == {"schema", "binding", "budget"}
            and envelope.get("schema") == "wlr50_clean.explicit_det_P02_aux_budget.v2"
            and envelope.get("binding") == binding
            and envelope.get("budget") == event.get("budget"),
            "det-P02 budget is not explicit or bound to this exact execution")

    report = event.get("fit_report") or {}
    frozen = report.get("frozen_kernel_fit_report") or {}
    accepted = report.get("accepted_auxiliary_updates")
    attempted = report.get("attempted_auxiliary_optimizer_steps")
    proof = report.get("protected_entire_Gaussian_bitwise_equal") or {}
    require(report.get("schema") == FRONT_AUX_DET_P02_SCHEMA
            and json_digest(report) == event.get("fit_report_sha256")
            and type(accepted) is int and type(attempted) is int
            and 0 < accepted <= attempted <= 32
            and report.get("optimized_parameters") == FRONT_AUX_DET_P02_PARAMETERS
            and report.get("optimized_scalar_count") == 256
            and type(report.get("actually_changed_scalar_count")) is int
            and 0 < report["actually_changed_scalar_count"] <= 256
            and report.get("P01_initial_gradient_exact_zero") is True
            and set(proof) == {"real_P01", "real_P03plus", "synthetic_P01_P03_P13"}
            and all(proof.values())
            and report.get("P02_mean_and_log_sigma_may_both_change") is True
            and report.get("source_observations_directly_saved389") is False
            and report.get("target_semantics") == FRONT_AUX_DET_P02_TARGET
            and report.get("same_input_invariance_is_not_future_trajectory_invariance") is True
            and report.get("nested_kernel_scope_is_capability_not_this_execution_scope") is True
            and report.get("MDP_or_control_changed") is False
            and report.get("kernel_or_sigma_rule_changed") is False
            and report.get("training_rng_preserved") is True
            and report.get("fresh_PPO_rollout_required") is True
            and report.get("teacher_deployed") is False
            and report.get("physical_success_claimed") is False
            and all(report.get(key) == 0 for key in (
                "PPO_decisions_added", "PPO_updates_added", "PPO_optimizer_steps_added"))
            and report.get("actor_parameter_sha256_before")
                == source_metadata.get("actor_parameter_sha256")
            and frozen.get("schema") == FRONT_AUX_REPORT_SCHEMA
            and frozen.get("budget") == event.get("budget")
            and frozen.get("accepted_auxiliary_updates") == accepted
            and frozen.get("attempted_auxiliary_optimizer_steps") == attempted
            and isinstance(frozen.get("steps"), list) and len(frozen["steps"]) == attempted
            and sum(step.get("accepted") is True for step in frozen["steps"]) == accepted
            and frozen.get("optimized_parameters") == FRONT_AUX_PARAMETERS
            and frozen.get("optimized_scalar_count") == 512
            and frozen.get("PPO_Adam_preserved_sha256")
                == source_metadata.get("optimizer_state_sha256")
            and frozen.get("PPO_LR_preserved")
                == source_metadata.get("optimizer_learning_rate")
            and frozen.get("actor_parameter_sha256_before")
                == report.get("actor_parameter_sha256_before")
            and frozen.get("actor_parameter_sha256_after")
                == report.get("actor_parameter_sha256_after"),
            "det-P02 fit report is not the admitted P02-column finite non-PPO update")


def _front_aux_mean_v3_helper_path(name: str) -> Path:
    return OUTPUT_ROOT / "front_mean_rehearsal_v3" / name


def _front_aux_mean_v3_event(event: dict[str, Any], source_metadata: dict[str, Any],
                             source_path: Path) -> None:
    """Validate event3 without assigning it a fixed accepted-step count."""
    helper_hashes = event.get("helper_sha256")
    require(isinstance(helper_hashes, dict) and set(helper_hashes) == FRONT_AUX_MEAN_V3_HELPERS
            and all(isinstance(digest, str) and len(digest) == 64
                    and sha256(_front_aux_mean_v3_helper_path(name)) == digest
                    for name, digest in helper_hashes.items()),
            "mean-v3 AUX helper hashes differ from the bound finite execution helpers")

    binding = event.get("binding")
    require(isinstance(binding, dict) and set(binding) == {
                "source_checkpoint", "helpers", "inspection", "data_receipt_sha256",
                "data_manifest", "objective", "expected_head"}
            and binding.get("source_checkpoint") == event.get("source_checkpoint")
            and binding.get("helpers") == helper_hashes
            and binding.get("data_receipt_sha256") == json_digest(event.get("data_receipt"))
            and binding.get("objective") == event.get("objective")
            and binding.get("expected_head") == FRONT_AUX_MEAN_V3_HEAD
            and binding.get("expected_head")
                == (source_metadata.get("runtime_contract") or {}).get("source_git_commit"),
            "mean-v3 AUX binding does not bind its exact source/data/objective/helpers")

    data_manifest = binding.get("data_manifest") or {}
    require(set(data_manifest) == {"path", "sha256"},
            "mean-v3 data-manifest binding is malformed")
    _front_aux_bound_file(
        data_manifest.get("path"), data_manifest.get("sha256"),
        "mean-v3 data composition manifest",
        expected=OUTPUT_ROOT / "front_mean_rehearsal_v3" / "data_manifest.json")

    objective = binding.get("objective") or {}
    objective_envelope = objective.get("envelope") or {}
    require(set(objective) == {"path", "sha256", "envelope"}
            and set(objective_envelope) == {"schema", "objective"}
            and objective_envelope.get("schema") == FRONT_AUX_MEAN_V3_OBJECTIVE_SCHEMA,
            "mean-v3 explicit objective binding is malformed")
    objective_path = _front_aux_bound_file(
        objective.get("path"), objective.get("sha256"), "mean-v3 explicit objective",
        expected=OUTPUT_ROOT / "front_mean_rehearsal_v3" / "objective_root_phase_balanced.json")
    require(read_json(objective_path) == objective_envelope,
            "mean-v3 explicit objective bytes differ from its bound envelope")

    data = event.get("data_receipt") or {}
    data_without_digest = dict(data)
    data_digest = data_without_digest.pop("receipt_content_sha256", None)
    selection = data.get("selection") or {}
    current = data.get("actual_current_source_binding") or {}
    tensors = data.get("tensor_sha256") or {}
    require(data.get("schema") == FRONT_AUX_MEAN_V3_DATA_SCHEMA
            and isinstance(data_digest, str) and json_digest(data_without_digest) == data_digest
            and data.get("composition_manifest") == data_manifest
            and data.get("loader") == {
                "path": str(_front_aux_mean_v3_helper_path("data_v3.py").resolve()),
                "sha256": helper_hashes["data_v3.py"]}
            and data.get("observations_directly_saved389_for_positive_labels") is False
            and data.get("offpolicy_historical_supervised_only_not_PPO") is True
            and data.get("optimization_authorized_by_loader") is False
            and data.get("loss_weights_selected_by_loader") is False
            and data.get("current_actor_or_future_trajectory_equivalence_claimed") is False
            and all(data.get(key) == 0 for key in (
                "AUX_updates", "PPO_decisions_added", "PPO_updates_added"))
            and selection.get("train_rows") == 86
            and selection.get("P02_validation_rows") == 85
            and selection.get("P01_independent_validation_rows") == 0
            and selection.get("P01_deterministic_training_and_probe_same_row") is True
            and selection.get("P03_P06_protection_rows") == 13
            and selection.get("additional_P07_P12_protection_rows") == 6
            and selection.get("total_protection_rows") == 19
            and selection.get("actual_protection_phases")
                == [f"P{phase:02d}" for phase in range(3, 13)]
            and selection.get("P13_real_coverage_claimed") is False
            and selection.get("train_phase_groups")
                == {"P01": [0], "P02": list(range(1, 86))}
            and isinstance(tensors, dict) and len(tensors) == 8
            and all(isinstance(value, str) and len(value) == 64
                    for value in tensors.values()),
            "mean-v3 data receipt overstates its finite P01/P02 labels or P03-P12 protection")
    source = event["source_checkpoint"]
    require(current.get("actual_caller_checkpoint") == str(source_path)
            and current.get("actual_caller_checkpoint_sha256") == source["sha256"]
            and current.get("actor_sha256") == source["actor_sha256"]
            and current.get("actual_caller_PPO_counters") == source["PPO_counters"]
            and current.get("runtime_contract_sha256") == source["runtime_contract_sha256"]
            and current.get("historical_fixed_state_semantics_compatible") is True
            and current.get("old_admission_was_rebound_to_current_weights") is False
            and current.get("current_weights_or_trajectory_equivalent_to_demonstration_claimed") is False,
            "mean-v3 data receipt is not bound to the actual current source checkpoint")

    inspection_record = binding.get("inspection") or {}
    require(set(inspection_record) == {"path", "sha256", "schema"}
            and inspection_record.get("schema") == FRONT_AUX_MEAN_V3_INSPECTION_SCHEMA,
            "mean-v3 readonly inspection binding is malformed")
    inspection_path = _front_aux_bound_file(
        inspection_record.get("path"), inspection_record.get("sha256"),
        "mean-v3 readonly inspection")
    require(inspection_path.parent == (OUTPUT_ROOT / "front_mean_rehearsal_v3").resolve(),
            "mean-v3 readonly inspection is outside its isolated directory")
    inspection = read_json(inspection_path)
    inspected = inspection.get("inspection") or {}
    base_binding = {key: value for key, value in binding.items() if key != "inspection"}
    require(inspection.get("schema") == FRONT_AUX_MEAN_V3_INSPECTION_SCHEMA
            and inspection.get("binding") == base_binding
            and inspection.get("data_receipt") == data
            and inspection.get("inspection_only") is True
            and inspection.get("optimizer_steps_performed") == 0
            and inspection.get("PPO_updates_added") == 0
            and inspection.get("checkpoint_written") is False
            and inspected.get("schema") == FRONT_AUX_MEAN_V3_SCHEMA.removesuffix(".fit") + ".inspection"
            and inspected.get("optimized_parameters") == FRONT_AUX_MEAN_V3_PARAMETERS
            and inspected.get("optimized_scalar_count") == 3084
            and inspected.get("phase_sample_counts") == {"P01": 1, "P02": 85}
            and inspected.get("actual_protection_phases") == list(range(3, 13))
            and inspected.get("missing_actual_protection_phases") == [13]
            and inspected.get("P13_actual_protection_coverage") is False
            and inspected.get("same_input_sigma_exact_by_frozen_trunk_and_sigma_rows") is True
            and inspected.get("P03plus_mean_bitwise_invariance_claimed") is False
            and inspected.get("future_closed_loop_invariance_claimed") is False
            and inspected.get("optimizer_steps_performed") == 0
            and inspected.get("AUX_updates_added") == 0
            and all(inspected.get(key) == 0 for key in (
                "PPO_decisions_added", "PPO_updates_added", "PPO_optimizer_steps_added"))
            and inspected.get("teacher_deployed") is False
            and inspected.get("physical_success_claimed") is False
            and inspected.get("actor_sha256_unchanged") == source["actor_sha256"],
            "mean-v3 readonly inspection is stale or overclaims scope/protection")

    budget_receipt = event.get("budget_receipt") or {}
    require(set(budget_receipt) == {"path", "sha256", "envelope"},
            "mean-v3 explicit budget receipt is malformed")
    budget_path = _front_aux_bound_file(
        budget_receipt.get("path"), budget_receipt.get("sha256"),
        "mean-v3 explicit budget")
    budget_envelope = budget_receipt.get("envelope") or {}
    require(read_json(budget_path) == budget_envelope
            and set(budget_envelope) == {"schema", "binding", "budget"}
            and budget_envelope.get("schema") == FRONT_AUX_MEAN_V3_BUDGET_SCHEMA
            and budget_envelope.get("binding") == binding
            and budget_envelope.get("budget") == event.get("budget"),
            "mean-v3 budget is not explicit or bound to this exact execution")

    report = event.get("fit_report") or {}
    accepted = report.get("accepted_auxiliary_updates")
    attempted = report.get("attempted_auxiliary_optimizer_steps")
    require(report.get("schema") == FRONT_AUX_MEAN_V3_SCHEMA
            and json_digest(report) == event.get("fit_report_sha256")
            and report.get("objective") == objective_envelope.get("objective")
            and report.get("budget") == event.get("budget")
            and type(accepted) is int and type(attempted) is int
            and 0 < accepted <= attempted <= 32
            and isinstance(report.get("steps"), list) and len(report["steps"]) == attempted
            and sum(step.get("accepted") is True for step in report["steps"]) == accepted
            and report.get("optimized_parameters") == FRONT_AUX_MEAN_V3_PARAMETERS
            and report.get("optimized_scalar_count") == 3084
            and type(report.get("actually_changed_scalar_count")) is int
            and 0 < report["actually_changed_scalar_count"] <= 3084
            and report.get("same_input_sigma_exact") is True
            and report.get("P03plus_mean_bitwise_invariance_claimed") is False
            and report.get("affected_mean_phases") == list(range(1, 14))
            and report.get("positive_label_phases") == ["P01", "P02"]
            and report.get("actual_protection_phases") == list(range(3, 13))
            and report.get("missing_actual_protection_phases") == [13]
            and report.get("P01_independent_validation_rows") == 0
            and report.get("validation_target_loss_is_reported_not_an_accept_rate_gate") is True
            and report.get("same_input_only_not_same_future_state_trajectory") is True
            and report.get("kernel_or_sigma_rule_changed") is False
            and report.get("MDP_or_control_changed") is False
            and report.get("training_rng_preserved") is True
            and report.get("fresh_PPO_rollout_required") is True
            and report.get("teacher_deployed") is False
            and report.get("physical_success_claimed") is False
            and all(report.get(key) == 0 for key in (
                "PPO_decisions_added", "PPO_updates_added", "PPO_optimizer_steps_added"))
            and report.get("actor_parameter_sha256_before") == source["actor_sha256"]
            and source["actor_sha256"] == source_metadata.get("actor_parameter_sha256")
            and isinstance(report.get("actor_parameter_sha256_after"), str)
            and len(report["actor_parameter_sha256_after"]) == 64
            and report["actor_parameter_sha256_after"] != source["actor_sha256"]
            and report.get("PPO_Adam_preserved_sha256")
                == source_metadata.get("optimizer_state_sha256")
            and report.get("PPO_LR_preserved")
                == source_metadata.get("optimizer_learning_rate"),
            "mean-v3 fit report is not the bound finite 3084-parameter mean-only update")


def _rr_aux_mean_v4_helper_path(name: str) -> Path:
    return OUTPUT_ROOT / "rr_mean_rehearsal_v4" / name


def _rr_aux_mean_v4_event(event: dict[str, Any], source_metadata: dict[str, Any],
                          source_path: Path) -> None:
    """Validate event4 as RR-labelled finite AUX, never as more Front AUX/PPO."""
    helper_hashes = event.get("helper_sha256")
    require(isinstance(helper_hashes, dict) and set(helper_hashes) == RR_AUX_MEAN_V4_HELPERS
            and all(isinstance(digest, str) and len(digest) == 64
                    and sha256(_rr_aux_mean_v4_helper_path(name)) == digest
                    for name, digest in helper_hashes.items()),
            "RR-mean-v4 AUX helper hashes differ from the frozen execution boundary")
    binding = event.get("binding")
    require(isinstance(binding, dict) and set(binding) == {
                "source_checkpoint", "helpers", "inspection", "data_receipt_sha256",
                "data_manifest", "objective", "expected_head"}
            and binding.get("source_checkpoint") == event.get("source_checkpoint")
            and binding.get("helpers") == helper_hashes
            and binding.get("data_receipt_sha256") == json_digest(event.get("data_receipt"))
            and binding.get("objective") == event.get("objective")
            and binding.get("expected_head") == RR_AUX_MEAN_V4_HEAD
            and binding.get("expected_head")
                == (source_metadata.get("runtime_contract") or {}).get("source_git_commit"),
            "RR-mean-v4 AUX binding omits its exact source/data/objective/helpers")
    data_manifest = binding.get("data_manifest") or {}
    require(set(data_manifest) == {"path", "sha256"},
            "RR-mean-v4 data-manifest binding is malformed")
    _front_aux_bound_file(
        data_manifest.get("path"), data_manifest.get("sha256"),
        "RR-mean-v4 data manifest",
        expected=OUTPUT_ROOT / "rr_mean_rehearsal_v4" / "data_manifest.json")
    objective = binding.get("objective") or {}
    objective_envelope = objective.get("envelope") or {}
    require(set(objective) == {"path", "sha256", "envelope"}
            and set(objective_envelope) == {"schema", "objective"}
            and objective_envelope.get("schema") == RR_AUX_MEAN_V4_OBJECTIVE_SCHEMA,
            "RR-mean-v4 explicit objective binding is malformed")
    objective_path = _front_aux_bound_file(
        objective.get("path"), objective.get("sha256"), "RR-mean-v4 objective")
    require(objective_path.parent == (OUTPUT_ROOT / "rr_mean_rehearsal_v4").resolve()
            and read_json(objective_path) == objective_envelope,
            "RR-mean-v4 objective bytes differ from its bound envelope")

    data = event.get("data_receipt") or {}
    data_without_digest = dict(data)
    data_digest = data_without_digest.pop("receipt_content_sha256", None)
    selection = data.get("selection") or {}
    x17 = data.get("X17_migration") or {}
    physical = data.get("physical_coverage") or {}
    source_receipt = data.get("source_checkpoint") or {}
    source_cp_receipt = source_receipt.get("checkpoint") or {}
    require(data.get("schema") == RR_AUX_MEAN_V4_DATA_SCHEMA
            and isinstance(data_digest, str) and json_digest(data_without_digest) == data_digest
            and data.get("data_manifest") == data_manifest
            and data.get("loader") == {
                "path": str(_rr_aux_mean_v4_helper_path("data_v4.py").resolve()),
                "sha256": helper_hashes["data_v4.py"]}
            and data.get("current_source_checkpoint_exactly_bound") is True
            and data.get("source_original_direct389") is True
            and data.get("current_unmodified389") is False
            and data.get("offpolicy_historical_supervised_only_not_PPO") is True
            and data.get("optimization_authorized_by_loader") is False
            and data.get("protection_targets_must_be_current_actor_means") is True
            and all(data.get(key) == 0 for key in (
                "AUX_updates_added", "PPO_decisions_added", "PPO_updates_added",
                "optimizer_steps_performed"))
            and data.get("labels") == RR_AUX_MEAN_V4_TARGET
            and source_cp_receipt == {"path": str(source_path), "sha256": event[
                "source_checkpoint"]["sha256"]}
            and source_receipt.get("actor_sha256") == source_metadata.get(
                "actor_parameter_sha256")
            and source_receipt.get("PPO_counters") == event.get("PPO_counters_unchanged")
            and source_receipt.get("entire_existing_front_AUX_ledger", {}).get(
                "accepted_auxiliary_updates_total") == 96
            and source_receipt.get("entire_existing_front_AUX_ledger", {}).get(
                "attempted_auxiliary_optimizer_steps_total") == 96
            and selection.get("source_rows") == 513
            and selection.get("train_rows") == 342
            and selection.get("validation_rows") == 171
            and selection.get("train_phase_counts") == {"P09": 230, "P10": 1, "P11": 111}
            and selection.get("validation_phase_counts") == {"P09": 115, "P11": 56}
            and selection.get("P10_independent_validation_rows") == 0
            and selection.get("all12_raw_channels") is True
            and selection.get("same_episode_correlated_split") is True
            and selection.get("all_AIR_rows_individually_successful_claimed") is False
            and x17.get("only_derived_column") == 17
            and x17.get("source_old_phi_matches_direct_float32") is True
            and x17.get("current_phi_recomputed_from_exact_previous_endpoint") is True
            and x17.get("new_physical_trajectory_or_current_actor_equivalence_claimed") is False
            and physical.get("assist_owned_labels") == 0
            and physical.get("earlier_FL_preparation_pure_policy_claimed") is False
            and isinstance(data.get("tensor_sha256"), dict)
            and len(data["tensor_sha256"]) == 6
            and all(isinstance(value, str) and len(value) == 64
                    for value in data["tensor_sha256"].values()),
            "RR-mean-v4 data receipt overstates its direct389/raw-action evidence")

    inspection_record = binding.get("inspection") or {}
    require(set(inspection_record) == {"path", "sha256", "schema"}
            and inspection_record.get("schema") == RR_AUX_MEAN_V4_INSPECTION_SCHEMA,
            "RR-mean-v4 readonly inspection binding is malformed")
    inspection_path = _front_aux_bound_file(
        inspection_record.get("path"), inspection_record.get("sha256"),
        "RR-mean-v4 readonly inspection")
    require(inspection_path.parent == (OUTPUT_ROOT / "rr_mean_rehearsal_v4").resolve(),
            "RR-mean-v4 inspection is outside its isolated directory")
    inspection = read_json(inspection_path)
    inspected = inspection.get("inspection") or {}
    base_binding = {key: value for key, value in binding.items() if key != "inspection"}
    require(inspection.get("schema") == RR_AUX_MEAN_V4_INSPECTION_SCHEMA
            and inspection.get("binding") == base_binding
            and inspection.get("data_receipt") == data
            and inspection.get("inspection_only") is True
            and inspection.get("optimizer_steps_performed") == 0
            and inspection.get("PPO_updates_added") == 0
            and inspection.get("checkpoint_written") is False
            and inspected.get("schema") == RR_AUX_MEAN_V4_SCHEMA.removesuffix(".fit") + ".inspection"
            and inspected.get("optimized_parameters") == RR_AUX_MEAN_V4_PARAMETERS
            and inspected.get("optimized_scalar_count") == 3084
            and inspected.get("phase_sample_counts") == {"P09": 230, "P10": 1, "P11": 111}
            and inspected.get("validation_phase_sample_counts") == {"P09": 115, "P11": 56}
            and inspected.get("actual_protection_phases") == [1, 2, 3, 4, 5, 6, 7, 8, 12]
            and inspected.get("missing_actual_protection_phases") == [9, 10, 11, 13]
            and inspected.get("P10_independent_validation_rows") == 0
            and inspected.get("P13_actual_protection_coverage") is False
            and inspected.get("single_historical_trajectory_train_validation_correlated") is True
            and inspected.get("same_input_sigma_exact_by_frozen_trunk_and_sigma_rows") is True
            and inspected.get("P03plus_mean_bitwise_invariance_claimed") is False
            and inspected.get("future_closed_loop_invariance_claimed") is False
            and inspected.get("actor_sha256_unchanged") == source_metadata.get(
                "actor_parameter_sha256"),
            "RR-mean-v4 readonly inspection is stale or overclaims its evidence")

    budget_receipt = event.get("budget_receipt") or {}
    require(set(budget_receipt) == {"path", "sha256", "envelope"},
            "RR-mean-v4 explicit budget receipt is malformed")
    budget_path = _front_aux_bound_file(
        budget_receipt.get("path"), budget_receipt.get("sha256"),
        "RR-mean-v4 explicit budget")
    budget_envelope = budget_receipt.get("envelope") or {}
    require(read_json(budget_path) == budget_envelope
            and set(budget_envelope) == {"schema", "binding", "budget"}
            and budget_envelope.get("schema") == RR_AUX_MEAN_V4_BUDGET_SCHEMA
            and budget_envelope.get("binding") == binding
            and budget_envelope.get("budget") == event.get("budget"),
            "RR-mean-v4 budget is not explicit or bound to this execution")

    report = event.get("fit_report") or {}
    accepted = report.get("accepted_auxiliary_updates")
    attempted = report.get("attempted_auxiliary_optimizer_steps")
    breakdown = event.get("auxiliary_credit_breakdown") or {}
    expected_breakdown = {
        "inherited_historical_auxiliary": {"accepted": 7, "attempted": 8},
        "inherited_front_auxiliary": {"accepted": 96, "attempted": 96},
        "this_RR_auxiliary": {"accepted": accepted, "attempted": attempted},
        "cumulative_mixed_ledger_auxiliary": {
            "accepted": 96 + accepted, "attempted": 96 + attempted},
        "legacy_storage_key_does_not_mean_all_events_are_front_AUX": True,
        "all_auxiliary_counts_are_separate_from_PPO": True,
    }
    require(report.get("schema") == RR_AUX_MEAN_V4_SCHEMA
            and json_digest(report) == event.get("fit_report_sha256")
            and report.get("objective") == objective_envelope.get("objective")
            and report.get("budget") == event.get("budget")
            and type(accepted) is int and type(attempted) is int
            and 0 < accepted <= attempted <= 32
            and isinstance(report.get("steps"), list) and len(report["steps"]) == attempted
            and sum(step.get("accepted") is True for step in report["steps"]) == accepted
            and report.get("optimized_parameters") == RR_AUX_MEAN_V4_PARAMETERS
            and report.get("optimized_scalar_count") == 3084
            and type(report.get("actually_changed_scalar_count")) is int
            and 0 < report["actually_changed_scalar_count"] <= 3084
            and report.get("same_input_sigma_exact") is True
            and report.get("P03plus_mean_bitwise_invariance_claimed") is False
            and report.get("affected_mean_phases") == list(range(1, 14))
            and report.get("positive_label_phases") == ["P09", "P10", "P11"]
            and report.get("actual_protection_phases") == [1, 2, 3, 4, 5, 6, 7, 8, 12]
            and report.get("missing_actual_protection_phases") == [9, 10, 11, 13]
            and report.get("P10_independent_validation_rows") == 0
            and report.get("single_historical_trajectory_train_validation_correlated") is True
            and report.get("validation_target_loss_is_reported_not_an_accept_rate_gate") is True
            and report.get("same_input_only_not_same_future_state_trajectory") is True
            and report.get("kernel_or_sigma_rule_changed") is False
            and report.get("MDP_or_control_changed") is False
            and report.get("training_rng_preserved") is True
            and report.get("fresh_PPO_rollout_required") is True
            and report.get("teacher_deployed") is False
            and report.get("physical_success_claimed") is False
            and all(report.get(key) == 0 for key in (
                "PPO_decisions_added", "PPO_updates_added", "PPO_optimizer_steps_added"))
            and report.get("actor_parameter_sha256_before")
                == source_metadata.get("actor_parameter_sha256")
            and isinstance(report.get("actor_parameter_sha256_after"), str)
            and len(report["actor_parameter_sha256_after"]) == 64
            and report["actor_parameter_sha256_after"]
                != source_metadata.get("actor_parameter_sha256")
            and report.get("PPO_Adam_preserved_sha256")
                == source_metadata.get("optimizer_state_sha256")
            and report.get("PPO_LR_preserved")
                == source_metadata.get("optimizer_learning_rate")
            and breakdown == expected_breakdown,
            "RR-mean-v4 fit/credit record is not the bound finite non-PPO update")


def front_rehearsal_training_identity(metadata: dict[str, Any]) -> dict[str, Any]:
    """Validate the legacy-named ledger, including optional distinct RR AUX."""
    require(metadata.get(FRONT_AUX_KEY) is None,
            "front-rehearsal ledger must be nested in the RR workspace branch")
    branch = metadata.get("rr_postcross_workspace_branch") or {}
    ledger = branch.get(FRONT_AUX_KEY)
    if ledger is None:
        return {"front_rehearsal_auxiliary_present": False,
                "front_rehearsal_display_id": None,
                "front_rehearsal_event_index": None,
                "front_rehearsal_accepted_auxiliary_updates_total": None,
                "front_rehearsal_attempted_auxiliary_optimizer_steps_total": None,
                "front_rehearsal_latest_source_PPO_counters": None,
                "front_rehearsal_event_summaries": None}
    require(branch.get("schema") == RR_WORKSPACE_SCHEMA
            and branch.get("semantics") == RR_WORKSPACE_SEMANTICS
            and isinstance(ledger, dict) and ledger.get("schema") == FRONT_AUX_SCHEMA
            and ledger.get("training_lineage_label") == FRONT_AUX_LINEAGE_LABEL,
            "front-rehearsal ledger is outside the exact RR reward lineage")
    inherited_auxiliary = ((metadata.get("task_conditioned_hip_wheel_branch") or {})
                           .get("auxiliary_mean_learning") or {})
    require(inherited_auxiliary.get("accepted_auxiliary_updates_total") == 7
            and inherited_auxiliary.get("attempted_auxiliary_optimizer_steps_total") == 8,
            "front-rehearsal checkpoint lost or relabelled inherited limited AUX 7/8")
    events = ledger.get("events")
    require(isinstance(events, list) and bool(events)
            and [event.get("event_index") for event in events] == list(range(1, len(events) + 1)),
            "front-rehearsal event indices are not a nonempty 1..N sequence")
    v1_event_keys = {
        "event_index", "kind", "source_checkpoint", "helper_sha256", "data_receipt",
        "fit_report", "fit_report_sha256", "budget", "optimized_parameters",
        "target_semantics", "phase_scope", "mean_and_log_sigma_may_change",
        "physical_success_claimed", "teacher_deployed", "PPO_counters_unchanged",
        "stage_requested_decisions_unchanged", "PPO_decisions_added", "PPO_updates_added",
        "PPO_optimizer_steps_added",
    }
    det_p02_event_keys = v1_event_keys | {
        "binding", "budget_receipt", "source_observations_directly_saved389",
        "same_input_P01_P03plus_preserved",
    }
    mean_v3_event_keys = {
        "event_index", "kind", "source_checkpoint", "helper_sha256", "binding",
        "data_receipt", "fit_report", "fit_report_sha256", "objective", "budget",
        "budget_receipt", "optimized_parameters", "optimized_scalar_count",
        "positive_phase_scope", "affected_mean_phase_scope",
        "mean_and_log_sigma_may_change", "mean_may_change_all_phases",
        "same_input_sigma_fixed",
        "protection_is_finite_observed_state_loss_and_trust_not_global_mean_invariance",
        "source_positive_observations_directly_saved389", "target_semantics",
        "P01_positive_rows", "P01_independent_validation_rows",
        "actual_protection_phases", "missing_actual_protection_phases",
        "P13_actual_protection_coverage", "fresh_PPO_rollout_required",
        "physical_success_claimed", "teacher_deployed", "PPO_counters_unchanged",
        "stage_requested_decisions_unchanged", "PPO_decisions_added",
        "PPO_updates_added", "PPO_optimizer_steps_added",
    }
    rr_mean_v4_event_keys = {
        "event_index", "kind", "source_checkpoint", "helper_sha256", "binding",
        "data_receipt", "fit_report", "fit_report_sha256", "objective", "budget",
        "budget_receipt", "optimized_parameters", "optimized_scalar_count",
        "positive_phase_scope", "affected_mean_phase_scope",
        "mean_and_log_sigma_may_change", "mean_may_change_all_phases",
        "same_input_sigma_fixed",
        "protection_is_finite_observed_state_loss_and_trust_not_global_mean_invariance",
        "source_positive_observations_directly_saved389",
        "current_positive_observations_are_original_unmodified389",
        "current_potential_observation_mapping_index", "target_semantics",
        "validation_phase_scope", "P10_positive_rows",
        "P10_independent_validation_rows",
        "single_historical_trajectory_train_validation_correlated",
        "auxiliary_credit_breakdown", "actual_protection_phases",
        "missing_actual_protection_phases", "P13_actual_protection_coverage",
        "fresh_PPO_rollout_required", "teacher_deployed", "physical_success_claimed",
        "PPO_counters_unchanged", "stage_requested_decisions_unchanged",
        "PPO_decisions_added", "PPO_updates_added", "PPO_optimizer_steps_added",
    }
    immutable_branch_keys = (
        "p05_capture_assist_branch", "capture_feedback_semantics_branch",
        "task_conditioned_hip_wheel_branch",
    )
    kinds = [event.get("kind") if isinstance(event, dict) else None for event in events]
    require(kinds in (
                [FRONT_AUX_KIND],
                [FRONT_AUX_KIND, FRONT_AUX_DET_P02_KIND],
                [FRONT_AUX_KIND, FRONT_AUX_DET_P02_KIND, FRONT_AUX_MEAN_V3_KIND],
                [FRONT_AUX_KIND, FRONT_AUX_DET_P02_KIND, FRONT_AUX_MEAN_V3_KIND,
                 RR_AUX_MEAN_V4_KIND]),
            "AUX ledger must preserve events1-3 before optional distinct RR event4")
    event_summaries: list[dict[str, Any]] = []
    for position, event in enumerate(events):
        kind = event.get("kind") if isinstance(event, dict) else None
        is_v1 = kind == FRONT_AUX_KIND
        is_det_p02 = kind == FRONT_AUX_DET_P02_KIND
        is_mean_v3 = kind == FRONT_AUX_MEAN_V3_KIND
        is_rr_mean_v4 = kind == RR_AUX_MEAN_V4_KIND
        expected_keys = (v1_event_keys if is_v1 else
                         det_p02_event_keys if is_det_p02 else
                         mean_v3_event_keys if is_mean_v3 else rr_mean_v4_event_keys)
        require(isinstance(event, dict) and set(event) == expected_keys
                and event["physical_success_claimed"] is False
                and event["teacher_deployed"] is False
                and event["PPO_decisions_added"] == 0
                and event["PPO_updates_added"] == 0
                and event["PPO_optimizer_steps_added"] == 0,
                "front-rehearsal event misstates its bounded non-PPO credit")
        if is_mean_v3:
            require(event["optimized_parameters"] == FRONT_AUX_MEAN_V3_PARAMETERS
                    and event["optimized_scalar_count"] == 3084
                    and event["target_semantics"] == FRONT_AUX_MEAN_V3_TARGET
                    and event["positive_phase_scope"] == ["P01", "P02"]
                    and event["affected_mean_phase_scope"]
                        == [f"P{phase:02d}" for phase in range(1, 14)]
                    and event["mean_and_log_sigma_may_change"] is False
                    and event["mean_may_change_all_phases"] is True
                    and event["same_input_sigma_fixed"] is True
                    and event[
                        "protection_is_finite_observed_state_loss_and_trust_not_global_mean_invariance"] is True
                    and event["source_positive_observations_directly_saved389"] is False
                    and event["P01_positive_rows"] == 1
                    and event["P01_independent_validation_rows"] == 0
                    and event["actual_protection_phases"] == list(range(3, 13))
                    and event["missing_actual_protection_phases"] == [13]
                    and event["P13_actual_protection_coverage"] is False
                    and event["fresh_PPO_rollout_required"] is True,
                    "mean-v3 event misstates its existing-head scope, fixed sigma, or finite protection")
        elif is_rr_mean_v4:
            require(event["optimized_parameters"] == RR_AUX_MEAN_V4_PARAMETERS
                    and event["optimized_scalar_count"] == 3084
                    and event["target_semantics"] == RR_AUX_MEAN_V4_TARGET
                    and event["positive_phase_scope"] == ["P09", "P10", "P11"]
                    and event["affected_mean_phase_scope"]
                        == [f"P{phase:02d}" for phase in range(1, 14)]
                    and event["mean_and_log_sigma_may_change"] is False
                    and event["mean_may_change_all_phases"] is True
                    and event["same_input_sigma_fixed"] is True
                    and event[
                        "protection_is_finite_observed_state_loss_and_trust_not_global_mean_invariance"] is True
                    and event["source_positive_observations_directly_saved389"] is True
                    and event["current_positive_observations_are_original_unmodified389"] is False
                    and event["current_potential_observation_mapping_index"] == 17
                    and event["validation_phase_scope"] == ["P09", "P11"]
                    and event["P10_positive_rows"] == 1
                    and event["P10_independent_validation_rows"] == 0
                    and event["single_historical_trajectory_train_validation_correlated"] is True
                    and event["actual_protection_phases"] == [1, 2, 3, 4, 5, 6, 7, 8, 12]
                    and event["missing_actual_protection_phases"] == [9, 10, 11, 13]
                    and event["P13_actual_protection_coverage"] is False
                    and event["fresh_PPO_rollout_required"] is True,
                    "RR-mean-v4 event misstates its RR labels, fixed sigma, or finite protection")
        else:
            expected_parameters = (FRONT_AUX_PARAMETERS if is_v1 else
                                   FRONT_AUX_DET_P02_PARAMETERS)
            expected_target = FRONT_AUX_TARGET if is_v1 else FRONT_AUX_DET_P02_TARGET
            expected_phases = ["P01", "P02"] if is_v1 else ["P02"]
            require(event["optimized_parameters"] == expected_parameters
                    and event["target_semantics"] == expected_target
                    and event["phase_scope"] == expected_phases
                    and event["mean_and_log_sigma_may_change"] is True
                    and (is_v1 or event.get("source_observations_directly_saved389") is False
                         and event.get("same_input_P01_P03plus_preserved") is True),
                    "front-rehearsal event misstates its bounded non-PPO phase-column scope")
        source = event.get("source_checkpoint")
        source_keys = ({"path", "sha256", "manifest_sha256"} if is_v1 else
                       {"path", "sha256", "manifest_sha256", "PPO_counters",
                        "runtime_contract_sha256"} if is_det_p02 else
                       {"path", "sha256", "manifest_sha256", "actor_sha256",
                        "PPO_counters", "runtime_contract_sha256"})
        require(isinstance(source, dict) and set(source) == source_keys,
                "front-rehearsal event lacks an exact source checkpoint binding")
        source_path = _front_aux_bound_file(
            source.get("path"), source.get("sha256"), "front-rehearsal source checkpoint")
        source_manifest_path = source_path.with_name(source_path.stem + "_manifest.json")
        require(source_manifest_path.is_file()
                and sha256(source_manifest_path) == source.get("manifest_sha256"),
                "front-rehearsal source checkpoint or sidecar bytes changed")
        source_metadata = read_json(source_manifest_path)
        source_counters = event.get("PPO_counters_unchanged")
        require(isinstance(source_counters, dict) and set(source_counters) == set(COUNTERS)
                and all(type(source_counters[key]) is int and source_counters[key] >= 0
                        and source_metadata.get(key) == source_counters[key] for key in COUNTERS)
                and Path(source_metadata.get("checkpoint_path", "")).resolve() == source_path
                and source_metadata.get("checkpoint_sha256") == source["sha256"]
                and event.get("stage_requested_decisions_unchanged")
                    == source_metadata.get("stage_requested_decisions"),
                "front-rehearsal event changed or misbound its source PPO counters")
        if not is_v1:
            require(source.get("PPO_counters") == source_counters
                    and source.get("runtime_contract_sha256")
                        == json_digest(source_metadata.get("runtime_contract")),
                    "later front-rehearsal event source omits its PPO/runtime boundary")
        if is_mean_v3 or is_rr_mean_v4:
            require(source.get("actor_sha256") == source_metadata.get("actor_parameter_sha256"),
                    "mean-head AUX event source omits its exact actor boundary")
        require(rr_workspace_training_identity(source_metadata)["rr_workspace_branch_present"] is True,
                "front-rehearsal source checkpoint lacks the validated RR workspace lineage")
        source_branch = source_metadata.get("rr_postcross_workspace_branch") or {}
        require({key: value for key, value in source_branch.items() if key != FRONT_AUX_KEY}
                    == {key: value for key, value in branch.items() if key != FRONT_AUX_KEY}
                and source_branch.get(FRONT_AUX_KEY) == _front_aux_prefix(events[:position])
                and all(source_metadata.get(key) == metadata.get(key)
                        for key in immutable_branch_keys),
                "front-rehearsal event changed old P05/feedback/RR/task-conditioned AUX lineage")
        if is_v1:
            _front_aux_v1_event(event, source_metadata)
            semantics = "P01/P02 directly-saved raw-action phase columns"
        elif is_det_p02:
            _front_aux_det_p02_event(event, source_metadata, source_path)
            semantics = "P02 reconstructed389 executed deterministic raw actions"
        elif is_mean_v3:
            _front_aux_mean_v3_event(event, source_metadata, source_path)
            semantics = "P01/P02 actual raw labels; existing 3084-scalar mean head only"
        else:
            _rr_aux_mean_v4_event(event, source_metadata, source_path)
            semantics = (
                "P09/P10/P11 actual stochastic raw12; existing mean head; distinct RR AUX")
        summary = {
            "event_index": event["event_index"], "kind": kind,
            "accepted_auxiliary_updates": event["fit_report"]["accepted_auxiliary_updates"],
            "attempted_auxiliary_optimizer_steps": event["fit_report"][
                "attempted_auxiliary_optimizer_steps"],
            "phase_scope": list(event.get("phase_scope") or event["positive_phase_scope"]),
            "optimized_parameters": list(event["optimized_parameters"]),
            "source_global_policy_decisions": source_counters["global_policy_decisions"],
            "mean_and_log_sigma_may_change": event["mean_and_log_sigma_may_change"],
            "mean_may_change_all_phases": event.get("mean_may_change_all_phases", False),
            "same_input_sigma_fixed": event.get("same_input_sigma_fixed", False),
            "source_observations_directly_saved389": bool(is_v1 or is_rr_mean_v4),
            "actual_protection_phases": event.get("actual_protection_phases"),
            "missing_actual_protection_phases": event.get("missing_actual_protection_phases"),
            "P01_independent_validation_rows": event.get("P01_independent_validation_rows"),
            "semantics": semantics,
            "PPO_credit_added": 0, "physical_success_claimed": False,
        }
        if is_rr_mean_v4:
            summary.update({
                "P10_independent_validation_rows": event["P10_independent_validation_rows"],
                "target_semantics": event["target_semantics"],
            })
        event_summaries.append(summary)

    expected = _front_aux_prefix(events)
    require(ledger == expected,
            "front-rehearsal totals or lineage label do not equal the immutable event chain")
    latest = events[-1]
    latest_counters = latest["PPO_counters_unchanged"]
    if all(metadata.get(key) == latest_counters[key] for key in COUNTERS):
        require(metadata.get("actor_parameter_sha256")
                == latest["fit_report"].get("actor_parameter_sha256_after"),
                "same-counter AUX checkpoint actor differs from its accepted fit report")
    decisions = metadata.get("global_policy_decisions")
    require(type(decisions) is int
            and all(type(metadata.get(key)) is int
                    and metadata[key] >= latest_counters[key] for key in COUNTERS),
            "front-rehearsal ledger is attached before its source PPO boundary")
    index = latest["event_index"]
    display_suffix = ({FRONT_AUX_KIND: "AUXFR",
                       FRONT_AUX_DET_P02_KIND: "AUXDET",
                       FRONT_AUX_MEAN_V3_KIND: "AUXMEAN",
                       RR_AUX_MEAN_V4_KIND: "AUXRR"}[latest["kind"]])
    rr_event = latest if latest["kind"] == RR_AUX_MEAN_V4_KIND else None
    front_events = events[:-1] if rr_event is not None else events
    result = {"front_rehearsal_auxiliary_present": True,
            "front_rehearsal_display_id": f"CP{decisions}_{display_suffix}{index}",
            "front_rehearsal_event_index": index,
            "front_rehearsal_accepted_auxiliary_updates_total":
                sum(event["fit_report"]["accepted_auxiliary_updates"]
                    for event in front_events),
            "front_rehearsal_attempted_auxiliary_optimizer_steps_total":
                sum(event["fit_report"]["attempted_auxiliary_optimizer_steps"]
                    for event in front_events),
            "front_rehearsal_inherited_limited_auxiliary_accepted_total": 7,
            "front_rehearsal_inherited_limited_auxiliary_attempted_total": 8,
            "front_rehearsal_training_lineage_label": FRONT_AUX_LINEAGE_LABEL,
            "front_rehearsal_event_summaries": event_summaries,
            "front_rehearsal_latest_source_checkpoint": dict(latest["source_checkpoint"]),
            "front_rehearsal_latest_source_PPO_counters": dict(latest_counters),
            "front_rehearsal_latest_fit_report_sha256": latest["fit_report_sha256"]}
    if rr_event is not None:
        accepted = rr_event["fit_report"]["accepted_auxiliary_updates"]
        attempted = rr_event["fit_report"]["attempted_auxiliary_optimizer_steps"]
        result.update({
            "rr_rehearsal_auxiliary_present": True,
            "rr_rehearsal_accepted_auxiliary_updates_total": accepted,
            "rr_rehearsal_attempted_auxiliary_optimizer_steps_total": attempted,
            "mixed_rehearsal_accepted_auxiliary_updates_total":
                ledger["accepted_auxiliary_updates_total"],
            "mixed_rehearsal_attempted_auxiliary_optimizer_steps_total":
                ledger["attempted_auxiliary_optimizer_steps_total"],
            "legacy_front_rehearsal_storage_key_is_mixed": True,
        })
    return result


def run(command: list[str], *, input_bytes: Iterable[bytes] | None = None) -> None:
    if input_bytes is None:
        completed = subprocess.run(command, capture_output=True, text=True, errors="replace")
        require(completed.returncode == 0, f"command failed ({completed.returncode}): {completed.stderr[-2500:]}")
        return
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        require(process.stdin is not None, "ffmpeg panel pipe is unavailable")
        for block in input_bytes:
            require(process.poll() is None, "ffmpeg exited before every overlay frame was written")
            process.stdin.write(block)
        process.stdin.close()
        error = process.stderr.read().decode(errors="replace") if process.stderr is not None else ""
        require(process.wait() == 0, "overlay encode failed: " + error[-2500:])
    finally:
        if process.poll() is None:
            process.terminate()


def artifact(source: Path, manifest: dict[str, Any], name: str) -> Path:
    record = (manifest.get("artifacts") or {}).get(name)
    require(isinstance(record, dict), f"source manifest lacks {name}")
    path = Path(record.get("path", "")).resolve(strict=True)
    require(path.parent == source and path.name == name, f"{name} is outside the sealed source")
    require(path.stat().st_size == record.get("bytes") and sha256(path) == record.get("sha256"),
            f"{name} changed after source manifest creation")
    return path


def checkpoint_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    proof = manifest.get("checkpoint_load_provenance") or {}
    require(proof.get("checkpoint_loaded_and_verified") is True, "source lacks a verified saved checkpoint load")
    require(manifest.get("policy_sampling_mode") == "deterministic_conditional_mean",
            "main export requires deterministic saved-checkpoint evaluation")
    require(proof.get("stochastic_policy") in (None, False) and proof.get("policy_seed") is None,
            "deterministic source carries stochastic policy provenance")
    binding = proof.get("source") or {}
    checkpoint = Path(binding.get("checkpoint", "")).resolve(strict=True)
    sidecar = Path(binding.get("manifest", "")).resolve(strict=True)
    require(sha256(checkpoint) == binding.get("checkpoint_sha256")
            and sha256(sidecar) == binding.get("manifest_sha256"), "checkpoint binding changed after capture")
    metadata = read_json(sidecar)
    decisions = proof.get("saved_global_policy_decisions")
    require(type(decisions) is int and decisions > 0
            and metadata.get("global_policy_decisions") == decisions
            and Path(metadata.get("checkpoint_path", "")).resolve() == checkpoint
            and metadata.get("checkpoint_sha256") == binding["checkpoint_sha256"],
            "checkpoint sidecar differs from video load provenance")
    actor_hash = (proof.get("parameter_hashes") or {}).get("actor_parameter_sha256")
    require(bool(actor_hash) and metadata.get("actor_parameter_sha256") == actor_hash,
            "loaded actor hash differs from saved checkpoint")
    branch = metadata.get("p05_capture_assist_branch") or {}
    origin = branch.get("counter_origin") or {}
    warm_start = metadata.get("new_mdp_warm_start") or {}
    same_layout = warm_start.get("observation_same_layout_transition") or {}
    require(branch.get("schema") == "wlr50_clean.p05_capture_assist_branch.v1"
            and warm_start.get("schema") == "wlr50_clean.semantic_v3_new_mdp_warm_start.v1"
            and warm_start.get("old_rollout_buffer_inherited") is False
            and same_layout.get("old_rollout_inherited") is False
            and metadata.get("old_rollout_inherited") in (None, False),
            "checkpoint lacks the explicit P05 capture-assist warm-start boundary")
    counters = {}
    for key in COUNTERS:
        current, initial = metadata.get(key), origin.get(key)
        require(type(current) is int and type(initial) is int and current >= initial,
                f"checkpoint has an invalid P05 {key} counter")
        counters[key] = current - initial
    feedback = feedback_training_identity(metadata)
    rr_workspace = rr_reward_training_identity(metadata)
    front_rehearsal = front_rehearsal_training_identity(metadata)
    preedge = p05_preedge_training_identity(metadata)
    return {"decisions": decisions, "checkpoint": str(checkpoint),
            "checkpoint_sha256": binding["checkpoint_sha256"], "manifest": str(sidecar),
            "manifest_sha256": binding["manifest_sha256"], "actor_parameter_sha256": actor_hash,
            "p05_added_policy_decisions": counters["global_policy_decisions"],
            "p05_added_ppo_updates": counters["ppo_updates"],
            "p05_added_optimizer_steps": counters["optimizer_steps"],
            "p05_counter_origin": dict(P05_COUNTER_ORIGIN),
            "migrated_warm_start": counters["ppo_updates"] == 0,
            **feedback, **rr_workspace, **front_rehearsal, **preedge}


def _capture_manifest(source: Path, manifest: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = artifact(source, manifest, "viewport_buffer_video_manifest.json")
    capture = read_json(path)
    require(capture.get("valid") is True and capture.get("encoder_finalized_before_app_close") is True,
            "source writer did not close with a valid full decode")
    return path, capture


def sealed_source(source: Path, *, candidate: bool) -> dict[str, Any]:
    source = Path(source).resolve(strict=True)
    run_path = source.parent / "run_manifest.json"
    require(run_path.is_file(), "sealed parent run manifest is missing")
    run_manifest = read_json(run_path)
    require(bool(run_manifest.get("completed_at_utc")) and run_manifest.get("lifecycle") != "RUNNING",
            "source run is still active")
    manifest_path = source / "semantic_video_source_manifest.json"
    manifest = read_json(manifest_path)
    require(manifest.get("schema") == "wlr50_clean.semantic_video_source.v1"
            and manifest.get("from_phase") == "P01"
            and manifest.get("fresh_process_single_episode") is True
            and manifest.get("episode_count") == 1
            and manifest.get("optimizer_updates") == 0, "source is not one natural P01 evaluation episode")
    require(run_manifest.get("runtime_contract") == manifest.get("runtime_contract"),
            "run/source runtime contract mismatch")
    endpoint = manifest.get("episode_physics_ticks")
    require(type(endpoint) is int and 0 < endpoint <= 24000, "episode endpoint is outside the 200 s task horizon")
    if candidate:
        require(manifest.get("experiment_id") == EXPERIMENT and manifest.get("role") == "C",
                "candidate is not the P05 capture-assist C experiment")
        method = manifest.get("control_method")
        require(method == CONTROL_METHOD, "candidate control method is not PPO + declared capture assist + inherited AUX")
        if "source_control_method" in manifest:
            require(manifest["source_control_method"] == CONTROL_METHOD,
                    "candidate control-method aliases disagree")
        require(manifest.get("capture_assist_enabled_in_training_and_evaluation") is True
                and manifest.get("capture_assist_is_policy_learning") is False,
                "capture assist provenance is absent or mislabelled as policy learning")
        checkpoint = checkpoint_identity(manifest)
        tick_path = artifact(source, manifest, "capture_assist_ticks.jsonl")
    else:
        require(manifest.get("role") == "B" and manifest.get("physical_task_success") is True
                and manifest.get("success_candidate") is True
                and manifest.get("diagnostic_only") is False,
                "historical N source must be the accepted successful B source")
        checkpoint, tick_path = None, None
    video = artifact(source, manifest, "actual_viewport_video.mp4")
    ledger_path = artifact(source, manifest, "viewport_frame_ledger.jsonl")
    _, capture = _capture_manifest(source, manifest)
    return {"source": source, "manifest": manifest, "manifest_path": manifest_path,
            "run_manifest": run_manifest, "run_manifest_path": run_path, "video": video,
            "ledger_path": ledger_path, "capture": capture, "tick_path": tick_path,
            "checkpoint": checkpoint, "endpoint": endpoint}


def checked_media(context: dict[str, Any], *, ffmpeg: Path) -> tuple[list[Any], list[Any], dict[str, Any]]:
    source_frames = decode_frame_timeline(context["video"], ffmpeg=ffmpeg)
    ledger = load_viewport_frame_ledger(context["ledger_path"])
    endpoint = context["endpoint"]
    expected_ticks = [min((index + 1) * STRIDE, endpoint)
                      for index in range((endpoint + STRIDE - 1) // STRIDE)]
    require([row.frame_index for row in ledger] == list(range(len(ledger))), "source frame ledger index gap")
    require([row.sim_step for row in ledger] == expected_ticks, "source frame ledger is not the executed interval grid")
    require(len(source_frames) == len(ledger) == context["capture"].get("frame_count") <= MAX_FRAMES,
            "source frame/decode/ledger count mismatch")
    require(all(abs(frame.pts_s - index / FPS) < 1e-5 for index, frame in enumerate(source_frames)),
            "source native PTS grid is not continuous 15 fps")
    validation = validate_mp4(context["video"], ffmpeg=ffmpeg, expected_fps=FPS,
        expected_frame_count=len(ledger), expected_width=1280, expected_height=720,
        maximum_duration_s=200.0, require_sane_container_duration=False)
    require(validation.get("valid") is True, "source video does not fully decode")
    require(validation.get("sha256") == context["capture"].get("video_sha256"),
            "source video differs from recorder manifest")
    return source_frames, ledger, validation


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _joint(row: dict[str, Any], name: str, field: str) -> float | None:
    return _number(((row.get("joints") or {}).get(name) or {}).get(field))


def _wheel(row: dict[str, Any], name: str) -> float | None:
    return _number(((row.get("wheels") or {}).get(name) or {}).get("velocity_rad_s"))


def frame_summary(row: dict[str, Any]) -> dict[str, Any]:
    state = row.get("capture_assist")
    require(isinstance(state, dict) and state.get("schema") == "wlr50_clean.capture_assist_state.v1",
            "tick lacks the observable top-level capture-assist state")
    require(all(key in state for key in ("mode", "mode_name", "active", "hip_target_deg",
                                         "knee_hold_deg", "hip_entry_deg", "reason")),
            "capture-assist state is incomplete")
    dispatch = row.get("dispatch") or {}
    final = dispatch.get("drive_target_full12")
    require(isinstance(final, list) and len(final) == 12, "tick lacks actual final Full12 target")
    current = row.get("current_legs") or {}
    fl = current.get("FL") or {}
    initialized = bool(state.get("initialized"))
    force = next((_number(fl.get(key)) for key in (
        "obstacle_normal_force_n", "contact_reaction_force_n", "bearing_force_n")
        if _number(fl.get(key)) is not None), None)
    contact = ("TOP" if fl.get("top_surface_contact") is True else
               "AIR" if fl.get("air") is True else str(fl.get("contact_surface") or "NO_TOP"))
    wheels = [_wheel(row, name) for name in WHEEL_ORDER]
    require(all(value is not None for value in wheels), "tick lacks canonical measured four-wheel qd")
    return {
        "tick": row.get("episode_physics_tick"), "time_s": _number(row.get("sim_time_s")),
        "phase": row.get("phase"), "fl_capture_pending": row.get("fl_capture_pending"),
        "assist_active": bool(state["active"]), "assist_mode": state["mode_name"],
        "assist_reason": state["reason"],
        "FL_gap_mm": None if _number(fl.get("clearance_m")) is None else 1000 * float(fl["clearance_m"]),
        "FL_contact": contact, "FL_contact_force_n": force,
        "FL_placed_history": (row.get("placed_history") or {}).get("FL"),
        "FL_hip_assist_target_deg": _number(state.get("hip_target_deg")) if initialized else None,
        "FL_hip_final_target_deg": _number(final[0]), "FL_hip_actual_deg": _joint(row, SERVO_ORDER[0], "position_deg"),
        "FL_knee_hold_target_deg": _number(state.get("knee_hold_deg")) if initialized else None,
        "FL_knee_final_target_deg": _number(final[1]), "FL_knee_actual_deg": _joint(row, SERVO_ORDER[1], "position_deg"),
        "wheel_qd_canonical_rad_s": wheels,
    }


def capture_rows(context: dict[str, Any], ledger: list[Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    wanted = {row.sim_step for row in ledger}
    frames: dict[int, dict[str, Any]] = {}
    selected_candidates: list[dict[str, Any]] = []
    previous_mode = None
    previous_pending = None
    previous_placed = False
    minimum_gap: tuple[float, dict[str, Any]] | None = None
    count = 0
    with context["tick_path"].open("rb") as stream:
        for count, raw in enumerate(stream, 1):
            require(raw.endswith(b"\n"), "capture-assist tick ledger has a partial final row")
            row = json.loads(raw)
            require(row.get("episode_physics_tick") == count
                    and math.isclose(float(row.get("sim_time_s")), count / HZ, abs_tol=1e-10),
                    "capture-assist tick ledger clock gap")
            summary = frame_summary(row)
            mode = summary["assist_mode"]
            pending = summary["fl_capture_pending"]
            placed = summary["FL_placed_history"] is True
            if (count == 1 or mode != previous_mode or pending != previous_pending
                    or placed != previous_placed or summary["phase"] in ("P05", "P06")
                    and not any(item["phase"] == summary["phase"] for item in selected_candidates)):
                selected_candidates.append(summary)
            gap = summary["FL_gap_mm"]
            if gap is not None and (minimum_gap is None or gap < minimum_gap[0]):
                minimum_gap = (gap, summary)
            if count in wanted:
                frames[count] = summary
            previous_mode, previous_pending, previous_placed = mode, pending, placed
    require(count == context["endpoint"], "capture-assist ledger does not cover the entire episode")
    require(set(frames) == wanted, "capture-assist ledger lacks exact encoded frame endpoints")
    rows = [frames[item.sim_step] for item in ledger]
    if minimum_gap is not None:
        selected_candidates.append(minimum_gap[1])
    selected_candidates.append(rows[-1])
    selected: list[dict[str, Any]] = []
    seen: set[int] = set()
    for row in sorted(selected_candidates, key=lambda item: int(item["tick"])):
        if row["tick"] not in seen:
            selected.append(row); seen.add(row["tick"])
    return rows, selected


def outcome(manifest: dict[str, Any]) -> tuple[str, bool, str | None]:
    physical = manifest.get("physical_episode") or {}
    evaluation = physical.get("physical_task_evaluation") or {}
    success = (manifest.get("physical_task_success") is True
               and physical.get("task_success") is True
               and manifest.get("success_candidate") is True
               and manifest.get("diagnostic_only") is False
               and manifest.get("source_acceptance_error") is None)
    reason = evaluation.get("termination_reason") or physical.get("termination_reason") or manifest.get("source_acceptance_error")
    return ("SUCCESS" if success else "DIAGNOSTIC_FAILURE"), success, reason


def detail_artifact_identity(kind: str | None) -> tuple[str, str]:
    """Return the filename token and visible title for the actual detail interval."""
    if kind in (None, "CONTIGUOUS_P05_TO_P06"):
        return "P05_to_P06", "P05_to_P06 | CONTIGUOUS SAME EPISODE"
    require(kind == "P05_THROUGH_FAILURE_ENDPOINT", "unknown P05 detail interval kind")
    return "P05_through_failure", "P05_through_failure | NO P06 OBSERVED"


def output_names(decisions: int, success: bool, *, display_id: str | None = None,
                 detail_kind: str | None = None) -> dict[str, str]:
    suffix = "" if success else "_INCOMPLETE"
    identity = display_id or f"CP{decisions}"
    require(identity == f"CP{decisions}"
            or identity.startswith((f"CP{decisions}_AUXFR", f"CP{decisions}_AUXDET",
                                    f"CP{decisions}_AUXMEAN", f"CP{decisions}_AUXRR")),
            "output display identity does not belong to the checkpoint counters")
    detail_token, _ = detail_artifact_identity(detail_kind)
    return {
        "full": f"{identity}_DET_full_after_P05_fix_capture_assist{suffix}.mp4",
        "detail": f"{identity}_DET_{detail_token}_detail_capture_assist{suffix}.mp4",
        "pair": f"N_vs_{identity}_DET_same_camera.mp4",
    }


def require_aux_destination(destination: Path, display_id: str | None, *,
                            current_decisions: int,
                            latest_source_decisions: int | None) -> None:
    if display_id is None:
        return
    require(display_id.startswith((f"CP{current_decisions}_AUXFR",
                                   f"CP{current_decisions}_AUXDET",
                                   f"CP{current_decisions}_AUXMEAN",
                                   f"CP{current_decisions}_AUXRR"))
            and type(latest_source_decisions) is int
            and current_decisions >= latest_source_decisions,
            "front-rehearsal destination identity lacks its source PPO boundary")
    if current_decisions == latest_source_decisions:
        det_prefix = f"CP{current_decisions}_AUXDET"
        mean_prefix = f"CP{current_decisions}_AUXMEAN"
        rr_prefix = f"CP{current_decisions}_AUXRR"
        if display_id.startswith(det_prefix):
            event_index = display_id.removeprefix(det_prefix)
            reviewed_name = (f"CP{current_decisions}_deterministic_"
                             f"front_rehearsal_det_v{event_index}_review")
            require(destination.name == reviewed_name or destination.name == display_id
                    or destination.name.startswith(display_id + "_"),
                    "same-counter det-P02 AUX directory lacks its derived event identity")
        elif display_id.startswith(mean_prefix):
            event_index = display_id.removeprefix(mean_prefix)
            reviewed_names = {
                f"CP{current_decisions}_deterministic_front_mean_rehearsal_v{event_index}_review",
                f"CP{current_decisions}_deterministic_front_rehearsal_mean_v{event_index}_review",
            }
            require(destination.name in reviewed_names or destination.name == display_id
                    or destination.name.startswith(display_id + "_"),
                    "same-counter mean-head AUX directory lacks its derived event identity")
        elif display_id.startswith(rr_prefix):
            event_index = display_id.removeprefix(rr_prefix)
            reviewed_names = {
                f"CP{current_decisions}_deterministic_rr_mean_rehearsal_v{event_index}_review",
                f"CP{current_decisions}_deterministic_rr_rehearsal_mean_v{event_index}_review",
            }
            require(destination.name in reviewed_names or destination.name == display_id
                    or destination.name.startswith(display_id + "_"),
                    "same-counter RR-mean AUX directory lacks its derived event identity")
        else:
            require(destination.name == display_id or destination.name.startswith(display_id + "_"),
                    "same-counter front-rehearsal candidate directory must begin with its AUX display_id")
        return
    ordinary_identity = f"CP{current_decisions}"
    require(destination.name == ordinary_identity
            or destination.name.startswith(ordinary_identity + "_")
            and not destination.name.startswith(display_id),
            "post-AUX ordinary-PPO export directory must begin with its new CP identity")


def detail_interval(rows: list[dict[str, Any]]) -> tuple[int, int, str] | None:
    p05 = next((index for index, row in enumerate(rows) if row["phase"] == "P05"), None)
    if p05 is None:
        return None
    p06 = next((index for index, row in enumerate(rows[p05:], p05) if row["phase"] == "P06"), None)
    if p06 is None:
        return p05, len(rows), "P05_THROUGH_FAILURE_ENDPOINT"
    after = next((index for index, row in enumerate(rows[p06 + 1:], p06 + 1)
                  if row["phase"] not in ("P05", "P06")), None)
    return p05, (len(rows) if after is None else after), "CONTIGUOUS_P05_TO_P06"


def _fmt(value: Any, digits: int = 2) -> str:
    return "N/A" if value is None else f"{float(value):+.{digits}f}"


def panel_lines(row: dict[str, Any], *, checkpoint: int, p05_ppo_updates: int, result: str,
                feedback_revision: str | None = None,
                feedback_ppo_updates: int | None = None,
                rr_workspace_semantics: str | None = None,
                rr_workspace_ppo_updates: int | None = None,
                p05_preedge_semantics: str | None = None,
                p05_preedge_ppo_updates: int | None = None,
                display_id: str | None = None,
                front_aux_event_index: int | None = None,
                front_aux_accepted_total: int | None = None,
                front_aux_attempted_total: int | None = None,
                front_aux_event_summaries: list[dict[str, Any]] | None = None) -> list[str]:
    wheels = "  ".join(f"{leg} {_fmt(value, 3)}" for leg, value in zip(WHEEL_LABELS, row["wheel_qd_canonical_rad_s"]))
    common = [
        f"assist {row['assist_mode']} active={row['assist_active']} reason={row['assist_reason']} | FL gap {_fmt(row['FL_gap_mm'])}mm {row['FL_contact']} force {_fmt(row['FL_contact_force_n'])}N placed={row['FL_placed_history']}",
        f"FL hip assist/final/actual deg {_fmt(row['FL_hip_assist_target_deg'])}/{_fmt(row['FL_hip_final_target_deg'])}/{_fmt(row['FL_hip_actual_deg'])} | knee HOLD/final/actual {_fmt(row['FL_knee_hold_target_deg'])}/{_fmt(row['FL_knee_final_target_deg'])}/{_fmt(row['FL_knee_actual_deg'])}",
        "actual canonical wheel qd rad/s: " + wheels,
    ]
    if feedback_revision is not None:
        require(feedback_revision == FEEDBACK_REVISION and type(feedback_ppo_updates) is int
                and feedback_ppo_updates >= 0, "invalid capture-feedback overlay identity")
        shown_id = display_id or f"CP{checkpoint}"
        lines = [
            f"PPO + CAPTURE ASSIST + INHERITED LIMITED AUX | DETERMINISTIC {shown_id} | {result}",
            f"P05-origin PPO +{p05_ppo_updates} | feedback-origin PPO +{feedback_ppo_updates} | feedback {feedback_revision}",
        ]
        if rr_workspace_semantics is not None:
            require(rr_workspace_semantics in {
                        RR_WORKSPACE_SEMANTICS, RR_RECEIVER_V2_SEMANTICS}
                    and type(rr_workspace_ppo_updates) is int
                    and rr_workspace_ppo_updates >= 0,
                    "invalid RR workspace overlay identity")
            lines.append(
                f"RR-reward {rr_workspace_semantics} | RR-origin PPO +{rr_workspace_ppo_updates}")
        if p05_preedge_semantics is not None:
            require(p05_preedge_semantics == P05_PREEDGE_SEMANTICS
                    and type(p05_preedge_ppo_updates) is int
                    and p05_preedge_ppo_updates >= 0,
                    "invalid P05 pre-edge control-revision overlay identity")
            lines.append(
                f"P05 control {p05_preedge_semantics} | origin PPO +{p05_preedge_ppo_updates} | CONTROL-MDP, NOT PURE POLICY")
        if front_aux_event_index is not None:
            latest_kind = (None if front_aux_event_summaries is None else
                           front_aux_event_summaries[-1].get("kind"))
            display_suffix = ({FRONT_AUX_KIND: "AUXFR",
                               FRONT_AUX_DET_P02_KIND: "AUXDET",
                               FRONT_AUX_MEAN_V3_KIND: "AUXMEAN",
                               RR_AUX_MEAN_V4_KIND: "AUXRR"}.get(latest_kind, "AUXFR"))
            expected_display = f"CP{checkpoint}_{display_suffix}{front_aux_event_index}"
            require(display_id == expected_display
                    and type(front_aux_accepted_total) is int and front_aux_accepted_total > 0
                    and type(front_aux_attempted_total) is int
                    and front_aux_accepted_total <= front_aux_attempted_total,
                    "invalid front-rehearsal overlay identity")
            if (front_aux_event_summaries is not None
                    and len(front_aux_event_summaries) == 4):
                first, second, third, fourth = front_aux_event_summaries
                front_accepted = sum(item["accepted_auxiliary_updates"]
                                     for item in (first, second, third))
                front_attempted = sum(item["attempted_auxiliary_optimizer_steps"]
                                      for item in (first, second, third))
                require([item.get("kind") for item in (first, second, third, fourth)] == [
                            FRONT_AUX_KIND, FRONT_AUX_DET_P02_KIND,
                            FRONT_AUX_MEAN_V3_KIND, RR_AUX_MEAN_V4_KIND]
                        and front_aux_accepted_total == front_accepted
                        and front_aux_attempted_total == front_attempted
                        and fourth.get("optimized_parameters") == RR_AUX_MEAN_V4_PARAMETERS
                        and fourth.get("phase_scope") == ["P09", "P10", "P11"]
                        and fourth.get("target_semantics") == RR_AUX_MEAN_V4_TARGET
                        and fourth.get("mean_and_log_sigma_may_change") is False
                        and fourth.get("mean_may_change_all_phases") is True
                        and fourth.get("same_input_sigma_fixed") is True
                        and fourth.get("P10_independent_validation_rows") == 0,
                        "RR-AUX #4 overlay summary mixes Front/RR credit or overstates scope")
                lines.extend([
                    f"FR-AUX #1-3 accepted/attempted {front_accepted}/{front_attempted} | inherited Front events unchanged | separate from PPO",
                    f"RR-AUX #4 accepted/attempted {fourth['accepted_auxiliary_updates']}/{fourth['attempted_auxiliary_optimizer_steps']} | P09/P10/P11 actual stochastic raw12 | 3084 mean params; sigma fixed | NOT PPO/SUCCESS",
                ])
            elif front_aux_event_summaries is not None and len(front_aux_event_summaries) in (2, 3):
                first, second = front_aux_event_summaries[:2]
                require(first.get("event_index") == 1 and first.get("kind") == FRONT_AUX_KIND
                        and second.get("event_index") == 2
                        and second.get("kind") == FRONT_AUX_DET_P02_KIND
                        and front_aux_accepted_total == sum(
                            item["accepted_auxiliary_updates"]
                            for item in front_aux_event_summaries)
                        and front_aux_attempted_total == sum(
                            item["attempted_auxiliary_optimizer_steps"]
                            for item in front_aux_event_summaries),
                        "front-rehearsal overlay event counts do not match its ledger")
                lines.extend([
                    f"FR-AUX #1 v1 P01/P02 accepted/attempted {first['accepted_auxiliary_updates']}/{first['attempted_auxiliary_optimizer_steps']} | direct raw | mu+log-sigma",
                    f"FR-AUX #2 det-P02 accepted/attempted {second['accepted_auxiliary_updates']}/{second['attempted_auxiliary_optimizer_steps']} | reconstructed389 W[:,1] | NOT PPO / NOT SUCCESS",
                ])
                if len(front_aux_event_summaries) == 3:
                    third = front_aux_event_summaries[2]
                    require(third.get("event_index") == 3
                            and third.get("kind") == FRONT_AUX_MEAN_V3_KIND
                            and third.get("optimized_parameters") == FRONT_AUX_MEAN_V3_PARAMETERS
                            and third.get("mean_and_log_sigma_may_change") is False
                            and third.get("mean_may_change_all_phases") is True
                            and third.get("same_input_sigma_fixed") is True
                            and third.get("actual_protection_phases") == list(range(3, 13))
                            and third.get("missing_actual_protection_phases") == [13]
                            and third.get("P01_independent_validation_rows") == 0,
                            "mean-v3 overlay summary overstates scope, sigma, or protection coverage")
                    lines.append(
                        f"FR-AUX #3 mean-head {third['accepted_auxiliary_updates']}/{third['attempted_auxiliary_optimizer_steps']} | existing 3084 mu params; sigma fixed | P01 1/no-val; protect P03-P12, no P13 | NOT PPO/SUCCESS")
            else:
                lines.append(
                    f"FR-AUX #{front_aux_event_index} accepted/attempted {front_aux_accepted_total}/{front_aux_attempted_total} | "
                    "P01/P02 phase-column mu+log-sigma may change | NOT PPO / NOT PHYSICAL SUCCESS")
        lines.extend([
            f"1x 15fps | t={row['time_s']:.3f}s tick={row['tick']} {row['phase']} | FL_CAPTURE_PENDING={row['fl_capture_pending']}",
            *common,
        ])
        return lines
    training = ("MIGRATED WARM START | +0 P05 PPO UPDATES" if p05_ppo_updates == 0 else
                f"P05 PPO TRAINED | +{p05_ppo_updates} UPDATES")
    return [
        f"PPO + CAPTURE ASSIST + INHERITED LIMITED AUX | DETERMINISTIC CP{checkpoint} | {result}",
        f"{training} | 1x 15fps | t={row['time_s']:.3f}s tick={row['tick']} {row['phase']} | FL_CAPTURE_PENDING={row['fl_capture_pending']}",
        *common,
    ]


def rgba_panels(rows: list[dict[str, Any]], *, checkpoint: int, p05_ppo_updates: int,
                result: str, feedback_revision: str | None = None,
                feedback_ppo_updates: int | None = None,
                rr_workspace_semantics: str | None = None,
                rr_workspace_ppo_updates: int | None = None,
                p05_preedge_semantics: str | None = None,
                p05_preedge_ppo_updates: int | None = None,
                display_id: str | None = None,
                front_aux_event_index: int | None = None,
                front_aux_accepted_total: int | None = None,
                front_aux_attempted_total: int | None = None,
                front_aux_event_summaries: list[dict[str, Any]] | None = None) -> Iterable[bytes]:
    font_path = Path("C:/Windows/Fonts/consola.ttf")
    require(font_path.is_file(), "overlay font is unavailable")
    for row in rows:
        lines = panel_lines(row, checkpoint=checkpoint, p05_ppo_updates=p05_ppo_updates,
                            result=result, feedback_revision=feedback_revision,
                            feedback_ppo_updates=feedback_ppo_updates,
                            rr_workspace_semantics=rr_workspace_semantics,
                            rr_workspace_ppo_updates=rr_workspace_ppo_updates,
                            p05_preedge_semantics=p05_preedge_semantics,
                            p05_preedge_ppo_updates=p05_preedge_ppo_updates,
                            display_id=display_id,
                            front_aux_event_index=front_aux_event_index,
                            front_aux_accepted_total=front_aux_accepted_total,
                            front_aux_attempted_total=front_aux_attempted_total,
                            front_aux_event_summaries=front_aux_event_summaries)
        styles = {
            5: ((19, 18, 17, 17, 17), 28),
            6: ((18, 16, 16, 16, 16, 16), 24),
            7: ((16, 14, 14, 15, 15, 15, 15), 20),
            8: ((15, 13, 13, 12, 14, 14, 14, 14), 18),
            9: ((15, 13, 13, 12, 12, 13, 13, 13, 13), 16),
            10: ((14, 12, 12, 11, 11, 11, 12, 12, 12, 12), 14),
            11: ((13, 11, 11, 10, 10, 10, 11, 11, 11, 11, 11), 13),
        }
        require(len(lines) in styles, "unexpected capture overlay line count")
        sizes, spacing = styles[len(lines)]
        fonts = [ImageFont.truetype(str(font_path), size) for size in sizes]
        image = Image.new("RGBA", (1280, 150), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 1279, 149), fill=(0, 0, 0, 178))
        for index, line in enumerate(lines):
            require(draw.textlength(line, font=fonts[index]) <= 1250, "overlay line exceeds fixed top panel")
            draw.text((14, 3 + spacing * index), line, font=fonts[index],
                      fill=(255, 255, 128, 255) if index == 0 else (255, 255, 255, 255))
        yield image.tobytes()


def validate_output(path: Path, count: int, width: int, height: int, ffmpeg: Path) -> dict[str, Any]:
    validation = validate_mp4(path, ffmpeg=ffmpeg, expected_fps=FPS, expected_frame_count=count,
        expected_width=width, expected_height=height, maximum_duration_s=200.0,
        require_sane_container_duration=True)
    require(validation.get("valid") is True, "derived video failed full decode")
    decoded = decode_frame_timeline(path, ffmpeg=ffmpeg)
    require(len(decoded) == count and all(abs(frame.pts_s - index / FPS) < 1e-5
            for index, frame in enumerate(decoded)), "derived video lost frames or changed the 15 fps PTS grid")
    return {key: validation.get(key) for key in (
        "sha256", "bytes", "valid", "full_decode", "frame_count", "fps", "resolution",
        "codec", "pixel_format", "duration_s", "container_duration_s", "first_pts_s", "last_pts_s",
        "frame_pts_sha256", "decoded_frame_checksums_sha256", "black_like_frame_count",
        "timestamps_monotonic", "timestamps_continuous")}


def preview(path: Path, count: int, ffmpeg: Path) -> list[dict[str, Any]]:
    pattern = path.with_name(path.stem + "_decoded_%02d.png")
    run([str(ffmpeg), "-hide_banner", "-nostdin", "-v", "error", "-n", "-i", str(path),
         "-vf", f"select=eq(n\\,0)+eq(n\\,{count - 1})", "-fps_mode", "passthrough",
         "-frames:v", "2", "-threads", "1", str(pattern)])
    paths = [path.with_name(path.stem + f"_decoded_{index:02d}.png") for index in (1, 2)]
    require(all(item.is_file() and item.stat().st_size > 0 for item in paths), "decoded previews are missing")
    return [{"path": str(item), "frame_index": frame} for item, frame in zip(paths, (0, count - 1))]


def encode_overlay(context: dict[str, Any], rows: list[dict[str, Any]], output: Path, *,
                   checkpoint: int, p05_ppo_updates: int, result: str, ffmpeg: Path,
                   feedback_revision: str | None = None,
                   feedback_ppo_updates: int | None = None,
                   rr_workspace_semantics: str | None = None,
                   rr_workspace_ppo_updates: int | None = None,
                   p05_preedge_semantics: str | None = None,
                   p05_preedge_ppo_updates: int | None = None,
                   display_id: str | None = None,
                   front_aux_event_index: int | None = None,
                   front_aux_accepted_total: int | None = None,
                   front_aux_attempted_total: int | None = None,
                   front_aux_event_summaries: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    count = len(rows)
    command = [str(ffmpeg), "-hide_banner", "-nostdin", "-v", "error", "-n",
        "-threads", "2", "-filter_threads", "1", "-filter_complex_threads", "1",
        "-i", str(context["video"]), "-f", "rawvideo", "-pixel_format", "rgba",
        "-video_size", "1280x150", "-framerate", str(FPS), "-i", "pipe:0",
        "-filter_complex", "[0:v]setpts=PTS-STARTPTS[v];[1:v]format=rgba,setpts=PTS-STARTPTS[p];[v][p]overlay=0:0:shortest=1,format=yuv420p[out]",
        "-map", "[out]", "-an", "-frames:v", str(count), "-r", str(FPS), "-fps_mode", "cfr",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-threads", "2", "-movflags", "+faststart", str(output)]
    run(command, input_bytes=rgba_panels(rows, checkpoint=checkpoint,
                                         p05_ppo_updates=p05_ppo_updates, result=result,
                                         feedback_revision=feedback_revision,
                                         feedback_ppo_updates=feedback_ppo_updates,
                                         rr_workspace_semantics=rr_workspace_semantics,
                                         rr_workspace_ppo_updates=rr_workspace_ppo_updates,
                                         p05_preedge_semantics=p05_preedge_semantics,
                                         p05_preedge_ppo_updates=p05_preedge_ppo_updates,
                                         display_id=display_id,
                                         front_aux_event_index=front_aux_event_index,
                                         front_aux_accepted_total=front_aux_accepted_total,
                                         front_aux_attempted_total=front_aux_attempted_total,
                                         front_aux_event_summaries=front_aux_event_summaries))
    return {"output": str(output), "frame_count": count, "normal_speed": True,
            "no_intro_frames": True, "top_overlay_no_canvas_growth": True,
            "validation": validate_output(output, count, 1280, 720, ffmpeg),
            "previews": preview(output, count, ffmpeg), "command": command}


def encode_detail(full: Path, rows: list[dict[str, Any]], output: Path, *, ffmpeg: Path) -> dict[str, Any]:
    interval = detail_interval(rows)
    if interval is None:
        return {
            "output": None,
            "omitted": True,
            "reason": "NO_P05_OBSERVED",
            "not_substituted_from_another_checkpoint": True,
            "full_source_preserved_at": str(full),
        }
    start, end, kind = interval
    count = end - start
    require(count > 0, "detail interval is empty")
    detail_token, detail_title = detail_artifact_identity(kind)
    require(f"_DET_{detail_token}_detail_" in output.name,
            "detail filename does not describe its actual interval")
    source_identity = full.name.split("_DET_full_", 1)[0]
    require(bool(source_identity) and all(character.isalnum() or character == "_"
                                          for character in source_identity),
            "detail source identity is not safe for a visible overlay title")
    visible_title = f"DETAIL | {source_identity} | {detail_title}"
    video_filter = (
        f"trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS,"
        "drawbox=x=0:y=0:w=iw:h=20:color=black@0.92:t=fill,"
        "drawtext=fontfile='C\\:/Windows/Fonts/consola.ttf':"
        f"text='{visible_title}':fontcolor=yellow:fontsize=14:x=14:y=2")
    command = [str(ffmpeg), "-hide_banner", "-nostdin", "-v", "error", "-n",
        "-threads", "2", "-filter_threads", "1", "-i", str(full), "-vf",
        video_filter, "-an",
        "-frames:v", str(count), "-r", str(FPS), "-fps_mode", "cfr", "-c:v", "libx264",
        "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p", "-threads", "2",
        "-movflags", "+faststart", str(output)]
    run(command)
    return {"output": str(output), "kind": kind, "source_frame_interval_half_open": [start, end],
            "actual_tick_endpoints": [rows[start]["tick"], rows[end - 1]["tick"]],
            "same_single_episode": True, "contiguous": True, "internal_stalls_removed": False,
            "visible_title_overlay": visible_title, "no_intro_frames": True,
            "full_source_preserved_at": str(full), "frame_count": count,
            "validation": validate_output(output, count, 1280, 720, ffmpeg),
            "previews": preview(output, count, ffmpeg), "command": command}


def encode_pair(baseline: dict[str, Any], candidate: dict[str, Any], full: Path, output: Path, *,
                historical_version: str, ffmpeg: Path,
                candidate_display_id: str | None = None,
                p05_preedge_semantics: str | None = None) -> dict[str, Any]:
    require(baseline["manifest"]["camera"] == candidate["manifest"]["camera"],
            "historical N and candidate camera definitions differ")
    _, b_ledger, _ = checked_media(baseline, ffmpeg=ffmpeg)
    c_count, b_count = candidate["frame_count"], len(b_ledger)
    count = max(b_count, c_count)
    require(count <= MAX_FRAMES, "comparison exceeds 200 seconds")
    safe_version = "".join(character if character.isalnum() or character in " ._-" else "_"
                           for character in historical_version)
    require(safe_version.strip() == historical_version and bool(safe_version.strip()),
            "historical N version contains unsafe label characters")
    font = "C\\:/Windows/Fonts/arial.ttf"
    auxiliary_label = ("RR-AUX NON-PPO" if candidate_display_id is not None
                       and "_AUXRR" in candidate_display_id else "FR-AUX NON-PPO")
    if p05_preedge_semantics is not None:
        require(p05_preedge_semantics == P05_PREEDGE_SEMANTICS,
                "comparison received an unknown P05 control revision")
        candidate_label = (
            f"{candidate_display_id or 'PPO'} | PPO+ASSIST+AUX | "
            "PREEDGE CONTROL-MDP, NOT PURE POLICY")
    else:
        candidate_label = ("PPO + CAPTURE ASSIST | DETERMINISTIC"
            if candidate_display_id is None else
            f"PPO+CAPTURE ASSIST+LIMITED AUX | {candidate_display_id} | {auxiliary_label}")
    labels = (f"HISTORICAL N_REF | {safe_version}", candidate_label)
    inputs = (baseline["video"], full)
    counts = (b_count, c_count)
    filters = []
    for index, (label, frames) in enumerate(zip(labels, counts)):
        filters.append(
            f"[{index}:v]setpts=PTS-STARTPTS,scale=960:540,pad=960:610:0:70:black,"
            f"tpad=stop_mode=clone:stop=-1,setpts=N/({FPS}*TB),"
            f"drawtext=fontfile='{font}':text='{label}':fontcolor=white:fontsize=23:x=14:y=13,"
            f"drawtext=fontfile='{font}':text='RUN ENDED - FROZEN, NOT NEW PHYSICS':fontcolor=yellow:fontsize=20:x=14:y=44:enable='gte(n,{frames})'[v{index}]"
        )
    filters.append("[v0][v1]hstack=inputs=2:shortest=1,format=yuv420p[out]")
    command = [str(ffmpeg), "-hide_banner", "-nostdin", "-v", "error", "-n", "-threads", "2",
        "-filter_complex_threads", "1", "-i", str(inputs[0]), "-i", str(inputs[1]),
        "-filter_complex", ";".join(filters), "-map", "[out]", "-an", "-frames:v", str(count),
        "-r", str(FPS), "-fps_mode", "cfr", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-threads", "2", "-movflags", "+faststart", str(output)]
    run(command)
    result = {"output": str(output), "frame_count": count,
            "historical_N_version": historical_version,
            "historical_N_source": str(baseline["source"]), "same_camera": True,
            "alignment": "same elapsed natural-P01 origin; no phase/event retiming",
            "same_controller_or_runtime_claimed": False, "historical_N_is_not_same_controller_control": True,
            "freeze_added_frames": {"historical_N": count - b_count, "candidate": count - c_count},
            "freeze_is_physical_evidence": False, "normal_speed": True,
            "validation": validate_output(output, count, 1920, 610, ffmpeg),
            "previews": preview(output, count, ffmpeg), "command": command}
    if candidate_display_id is not None:
        result["candidate_display_id"] = candidate_display_id
    if p05_preedge_semantics is not None:
        result["candidate_control_revision"] = p05_preedge_semantics
        result["candidate_control_revision_is_pure_policy"] = False
    return result


def response_markdown(rows: list[dict[str, Any]]) -> str:
    lines = ["# P05 capture-assist selected real ticks", "",
        "Selected state transitions and extrema only; the continuous full video and sealed per-tick ledger remain authoritative.", "",
        "| tick | phase | pending | assist | hip target/final/actual deg | knee hold/final/actual deg | FL gap mm | contact | force N | wheel qd FL/FR/RL/RR |",
        "|---:|:---:|:---:|:---|:---|:---|---:|:---|---:|:---|" ]
    for row in rows:
        wheels = "/".join(_fmt(value, 3) for value in row["wheel_qd_canonical_rad_s"])
        lines.append(f"| {row['tick']} | {row['phase']} | {row['fl_capture_pending']} | {row['assist_mode']} | "
            f"{_fmt(row['FL_hip_assist_target_deg'])}/{_fmt(row['FL_hip_final_target_deg'])}/{_fmt(row['FL_hip_actual_deg'])} | "
            f"{_fmt(row['FL_knee_hold_target_deg'])}/{_fmt(row['FL_knee_final_target_deg'])}/{_fmt(row['FL_knee_actual_deg'])} | "
            f"{_fmt(row['FL_gap_mm'])} | {row['FL_contact']} | {_fmt(row['FL_contact_force_n'])} | {wheels} |")
    lines.extend(["", "`allow_capture_continuation`/`FL_CAPTURE_PENDING` is not a placed/contact claim.", ""])
    return "\n".join(lines)


def receipt_checkpoint_identity(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Keep pre-AUX receipt shape byte-for-byte compatible when no ledger exists."""
    if checkpoint["front_rehearsal_auxiliary_present"]:
        return dict(checkpoint)
    return {key: value for key, value in checkpoint.items()
            if not key.startswith("front_rehearsal_")}


def front_aux_receipt_lineage(checkpoint: dict[str, Any]) -> dict[str, Any] | None:
    if not checkpoint["front_rehearsal_auxiliary_present"]:
        return None
    events = checkpoint["front_rehearsal_event_summaries"]
    latest = events[-1]
    if latest.get("kind") == RR_AUX_MEAN_V4_KIND:
        front_events, rr_event = events[:3], events[3]
        return {
            "display_id": checkpoint["front_rehearsal_display_id"],
            "latest_event_index": checkpoint["front_rehearsal_event_index"],
            "legacy_storage_key": FRONT_AUX_KEY,
            "legacy_storage_key_is_mixed": True,
            "events": events,
            "front_auxiliary": {
                "events": front_events,
                "accepted_auxiliary_updates_total": checkpoint[
                    "front_rehearsal_accepted_auxiliary_updates_total"],
                "attempted_auxiliary_optimizer_steps_total": checkpoint[
                    "front_rehearsal_attempted_auxiliary_optimizer_steps_total"],
            },
            "rr_auxiliary": {
                "event": rr_event,
                "accepted_auxiliary_updates_total": checkpoint[
                    "rr_rehearsal_accepted_auxiliary_updates_total"],
                "attempted_auxiliary_optimizer_steps_total": checkpoint[
                    "rr_rehearsal_attempted_auxiliary_optimizer_steps_total"],
                "all_counts_separate_from_PPO": True,
            },
            "mixed_ledger_auxiliary": {
                "accepted_auxiliary_updates_total": checkpoint[
                    "mixed_rehearsal_accepted_auxiliary_updates_total"],
                "attempted_auxiliary_optimizer_steps_total": checkpoint[
                    "mixed_rehearsal_attempted_auxiliary_optimizer_steps_total"],
                "must_not_be_called_all_front_auxiliary": True,
            },
            "training_lineage_label": checkpoint["front_rehearsal_training_lineage_label"],
            "latest_source_checkpoint": checkpoint[
                "front_rehearsal_latest_source_checkpoint"],
            "latest_source_PPO_counters": checkpoint[
                "front_rehearsal_latest_source_PPO_counters"],
            "latest_fit_report_sha256": checkpoint[
                "front_rehearsal_latest_fit_report_sha256"],
            "inherited_limited_auxiliary": {
                "accepted_auxiliary_updates_total": checkpoint[
                    "front_rehearsal_inherited_limited_auxiliary_accepted_total"],
                "attempted_auxiliary_optimizer_steps_total": checkpoint[
                    "front_rehearsal_inherited_limited_auxiliary_attempted_total"],
                "is_same_ledger_as_front_or_RR_rehearsal": False,
            },
            "latest_event": {
                "kind": latest.get("kind"),
                "phase_scope": latest.get("phase_scope"),
                "target_semantics": latest.get("target_semantics"),
                "optimized_parameters": latest.get("optimized_parameters"),
                "accepted_auxiliary_updates": latest["accepted_auxiliary_updates"],
                "attempted_auxiliary_optimizer_steps": latest[
                    "attempted_auxiliary_optimizer_steps"],
                "mean_and_log_sigma_may_change": latest.get(
                    "mean_and_log_sigma_may_change"),
                "mean_may_change_all_phases": latest.get("mean_may_change_all_phases"),
                "same_input_sigma_fixed": latest.get("same_input_sigma_fixed"),
                "actual_protection_phases": latest.get("actual_protection_phases"),
                "missing_actual_protection_phases": latest.get(
                    "missing_actual_protection_phases"),
                "P10_independent_validation_rows": latest.get(
                    "P10_independent_validation_rows"),
            },
            "PPO_credit_added": 0,
            "physical_success_claimed": False,
        }
    return {
        "display_id": checkpoint["front_rehearsal_display_id"],
        "latest_event_index": checkpoint["front_rehearsal_event_index"],
        "events": events,
        "accepted_auxiliary_updates_total": checkpoint[
            "front_rehearsal_accepted_auxiliary_updates_total"],
        "attempted_auxiliary_optimizer_steps_total": checkpoint[
            "front_rehearsal_attempted_auxiliary_optimizer_steps_total"],
        "training_lineage_label": checkpoint["front_rehearsal_training_lineage_label"],
        "latest_source_checkpoint": checkpoint[
            "front_rehearsal_latest_source_checkpoint"],
        "latest_source_PPO_counters": checkpoint[
            "front_rehearsal_latest_source_PPO_counters"],
        "latest_fit_report_sha256": checkpoint[
            "front_rehearsal_latest_fit_report_sha256"],
        "inherited_limited_auxiliary": {
            "accepted_auxiliary_updates_total": checkpoint[
                "front_rehearsal_inherited_limited_auxiliary_accepted_total"],
            "attempted_auxiliary_optimizer_steps_total": checkpoint[
                "front_rehearsal_inherited_limited_auxiliary_attempted_total"],
            "is_same_ledger_as_front_rehearsal": False,
        },
        "phase_scope": ["P01", "P02"],
        "mean_and_log_sigma_may_change": any(
            event.get("mean_and_log_sigma_may_change", True) for event in events),
        "latest_event": {
            "kind": latest.get("kind"),
            "optimized_parameters": latest.get("optimized_parameters"),
            "accepted_auxiliary_updates": latest["accepted_auxiliary_updates"],
            "attempted_auxiliary_optimizer_steps": latest[
                "attempted_auxiliary_optimizer_steps"],
            "mean_and_log_sigma_may_change": latest.get(
                "mean_and_log_sigma_may_change"),
            "mean_may_change_all_phases": latest.get("mean_may_change_all_phases"),
            "same_input_sigma_fixed": latest.get("same_input_sigma_fixed"),
            "actual_protection_phases": latest.get("actual_protection_phases"),
            "missing_actual_protection_phases": latest.get(
                "missing_actual_protection_phases"),
            "P01_independent_validation_rows": latest.get(
                "P01_independent_validation_rows"),
        },
        "PPO_credit_added": 0,
        "physical_success_claimed": False,
    }


def export(source: Path, destination: Path, historical_n_source: Path, historical_version: str) -> dict[str, Any]:
    destination = Path(destination).resolve()
    require(destination.is_relative_to(OUTPUT_ROOT.resolve()), "destination must remain in the isolated P05 output tree")
    require(not destination.exists(), "destination already exists; exports are immutable")
    candidate = sealed_source(source, candidate=True)
    baseline = sealed_source(historical_n_source, candidate=False)
    ffmpeg = find_ffmpeg(candidate["capture"].get("full_decode", {}).get("ffmpeg_path"))
    _, ledger, source_validation = checked_media(candidate, ffmpeg=ffmpeg)
    rows, selected = capture_rows(candidate, ledger)
    capture_observations = {
        "assist_active_observed": any(row["assist_active"] for row in selected),
        "assist_modes_observed": sorted({row["assist_mode"] for row in selected}),
        "fl_capture_pending_observed": any(row["fl_capture_pending"] is True for row in selected),
        "pending_branch_physical_validation_claimed": False,
    }
    result_label, success, termination = outcome(candidate["manifest"])
    checkpoint = candidate["checkpoint"]
    display_id = checkpoint["front_rehearsal_display_id"]
    latest_source = checkpoint["front_rehearsal_latest_source_PPO_counters"]
    require_aux_destination(destination, display_id,
                            current_decisions=checkpoint["decisions"],
                            latest_source_decisions=(None if latest_source is None else
                                latest_source["global_policy_decisions"]))
    interval = detail_interval(rows)
    detail_kind = None if interval is None else interval[2]
    names = output_names(checkpoint["decisions"], success, display_id=display_id,
                         detail_kind=detail_kind)
    destination.mkdir(parents=True)
    full = encode_overlay(candidate, rows, destination / names["full"], checkpoint=checkpoint["decisions"],
                          p05_ppo_updates=checkpoint["p05_added_ppo_updates"],
                          result=result_label, ffmpeg=ffmpeg,
                          feedback_revision=checkpoint["feedback_revision"],
                          feedback_ppo_updates=checkpoint["feedback_added_ppo_updates"],
                          rr_workspace_semantics=checkpoint["rr_workspace_semantics"],
                          rr_workspace_ppo_updates=checkpoint["rr_workspace_added_ppo_updates"],
                          p05_preedge_semantics=checkpoint.get("p05_preedge_semantics"),
                          p05_preedge_ppo_updates=checkpoint.get(
                              "p05_preedge_added_ppo_updates"),
                          display_id=display_id,
                          front_aux_event_index=checkpoint["front_rehearsal_event_index"],
                          front_aux_accepted_total=checkpoint[
                              "front_rehearsal_accepted_auxiliary_updates_total"],
                          front_aux_attempted_total=checkpoint[
                              "front_rehearsal_attempted_auxiliary_optimizer_steps_total"],
                          front_aux_event_summaries=checkpoint[
                              "front_rehearsal_event_summaries"])
    detail = encode_detail(Path(full["output"]), rows, destination / names["detail"], ffmpeg=ffmpeg)
    pair_candidate = {**candidate, "frame_count": len(rows)}
    pair = encode_pair(baseline, pair_candidate, Path(full["output"]), destination / names["pair"],
                       historical_version=historical_version, ffmpeg=ffmpeg,
                       candidate_display_id=display_id,
                       p05_preedge_semantics=checkpoint.get("p05_preedge_semantics"))
    response_json = destination / "capture_assist_selected_response.json"
    response_md = destination / "capture_assist_selected_response.md"
    write_new_json(response_json, {"schema": "wlr50_clean.p05_capture_assist_selected_response.v1",
        "source": str(candidate["source"]), "source_capture_assist_ticks_sha256": sha256(candidate["tick_path"]),
        "selection": "state transitions, first P05/P06, minimum FL gap, endpoint; no interpolation",
        "observations": capture_observations,
        "rows": selected, "full_per_tick_ledger": str(candidate["tick_path"])})
    write_new_text(response_md, response_markdown(selected))
    receipt_checkpoint = receipt_checkpoint_identity(checkpoint)
    training_lineage = {
        "p05_capture_assist": {
            "counter_origin": checkpoint["p05_counter_origin"],
            "added_policy_decisions": checkpoint["p05_added_policy_decisions"],
            "added_ppo_updates": checkpoint["p05_added_ppo_updates"],
            "added_optimizer_steps": checkpoint["p05_added_optimizer_steps"],
        },
        "capture_feedback_semantics": (None if not checkpoint["feedback_branch_present"] else {
            "feedback_revision": checkpoint["feedback_revision"],
            "counter_origin": checkpoint["feedback_counter_origin"],
            "added_policy_decisions": checkpoint["feedback_added_policy_decisions"],
            "added_ppo_updates": checkpoint["feedback_added_ppo_updates"],
            "added_optimizer_steps": checkpoint["feedback_added_optimizer_steps"],
        }),
        "rr_postcross_workspace": (None if not checkpoint["rr_workspace_branch_present"] else {
            "reward_semantics": checkpoint["rr_workspace_semantics"],
            "counter_origin": checkpoint["rr_workspace_counter_origin"],
            "added_policy_decisions": checkpoint["rr_workspace_added_policy_decisions"],
            "added_ppo_updates": checkpoint["rr_workspace_added_ppo_updates"],
            "added_optimizer_steps": checkpoint["rr_workspace_added_optimizer_steps"],
        }),
    }
    if checkpoint.get("rr_receiver_v2_branch_present") is True:
        training_lineage["rr_postcross_workspace"]["active_revision"] = (
            "rr_receiver_retirement_v2")
        training_lineage["rr_postcross_workspace_v1_preserved"] = {
            "reward_semantics": checkpoint["rr_workspace_v1_semantics"],
            "counter_origin": checkpoint["rr_workspace_v1_counter_origin"],
            "added_policy_decisions": checkpoint[
                "rr_workspace_v1_added_policy_decisions"],
            "added_ppo_updates": checkpoint["rr_workspace_v1_added_ppo_updates"],
            "added_optimizer_steps": checkpoint[
                "rr_workspace_v1_added_optimizer_steps"],
            "entire_branch_retained": True,
        }
        training_lineage["rr_receiver_retirement_v2"] = {
            "reward_semantics": checkpoint["rr_receiver_v2_semantics"],
            "counter_origin": checkpoint["rr_receiver_v2_counter_origin"],
            "added_policy_decisions": checkpoint[
                "rr_receiver_v2_added_policy_decisions"],
            "added_ppo_updates": checkpoint["rr_receiver_v2_added_ppo_updates"],
            "added_optimizer_steps": checkpoint[
                "rr_receiver_v2_added_optimizer_steps"],
        }
    if checkpoint.get("p05_preedge_branch_present") is True:
        training_lineage["p05_preedge_approach_recovery"] = {
            "control_semantics": checkpoint["p05_preedge_semantics"],
            "counter_origin": checkpoint["p05_preedge_counter_origin"],
            "added_policy_decisions": checkpoint[
                "p05_preedge_added_policy_decisions"],
            "added_ppo_updates": checkpoint["p05_preedge_added_ppo_updates"],
            "added_optimizer_steps": checkpoint[
                "p05_preedge_added_optimizer_steps"],
            "migration_added_updates": checkpoint[
                "p05_preedge_migration_added_updates"],
            "control_mdp_revision": True,
            "pure_policy_contribution": False,
        }
    if checkpoint["front_rehearsal_auxiliary_present"]:
        auxiliary_lineage = front_aux_receipt_lineage(checkpoint)
        if checkpoint["front_rehearsal_event_summaries"][-1]["kind"] == RR_AUX_MEAN_V4_KIND:
            training_lineage["mixed_front_and_RR_rehearsal_auxiliary"] = auxiliary_lineage
        else:
            training_lineage["front_rehearsal_auxiliary"] = auxiliary_lineage
    claims = {"capture_assist_is_policy_learning": False,
        "pending_is_placement": False, "phase_entry_is_task_success": False,
        "unobserved_pending_branch_is_physically_validated": False,
        "historical_N_is_same_controller_control": False,
        "rr_reward_revision_is_pure_policy": False,
        "rr_reward_revision_is_full_task_success": False,
        "failure_video_is_success": False}
    if checkpoint["front_rehearsal_auxiliary_present"]:
        latest_aux = checkpoint["front_rehearsal_event_summaries"][-1]
        claims.update({
            "front_rehearsal_auxiliary_is_PPO": False,
            "front_rehearsal_auxiliary_is_mean_only": False,
            "front_rehearsal_auxiliary_is_physical_success": False,
            "front_rehearsal_P03_plus_same_input_is_same_trajectory": False,
            "front_rehearsal_reconstructed389_is_direct_saved389": False,
            "front_rehearsal_v1_data_quality_applies_to_det_P02_event": False,
            "front_rehearsal_v1_or_det_P02_data_quality_applies_to_mean_v3_event": False,
            "front_rehearsal_latest_event_is_mean_only": (
                latest_aux["kind"] == FRONT_AUX_MEAN_V3_KIND),
            "front_rehearsal_latest_event_fixed_same_input_sigma": (
                latest_aux["same_input_sigma_fixed"]),
        })
        if latest_aux["kind"] == RR_AUX_MEAN_V4_KIND:
            claims.pop("front_rehearsal_latest_event_is_mean_only")
            claims.pop("front_rehearsal_latest_event_fixed_same_input_sigma")
            claims.update({
                "legacy_front_rehearsal_storage_key_means_all_events_are_front_AUX": False,
                "rr_rehearsal_auxiliary_is_PPO": False,
                "rr_rehearsal_auxiliary_is_physical_success": False,
                "rr_rehearsal_auxiliary_targets_are_stored_conditional_mean": False,
                "rr_rehearsal_auxiliary_current_positive_observations_are_unmodified389": False,
                "rr_rehearsal_auxiliary_same_input_sigma_fixed": True,
            })
    if checkpoint.get("rr_receiver_v2_branch_present") is True:
        claims.update({
            "rr_receiver_v2_is_pure_policy": False,
            "rr_receiver_v2_is_full_task_success": False,
            "rr_receiver_v2_replaces_or_rewrites_v1_lineage": False,
        })
    if checkpoint.get("p05_preedge_branch_present") is True:
        claims.update({
            "p05_preedge_recovery_is_pure_policy": False,
            "p05_preedge_recovery_is_same_mdp": False,
            "p05_preedge_recovery_is_physical_success": False,
            "p05_preedge_recovery_rewrites_prior_four_origins_or_AUX": False,
        })
    receipt = {"schema": "wlr50_clean.p05_capture_video_export.v1",
        "source": str(candidate["source"]), "source_manifest": str(candidate["manifest_path"]),
        "source_manifest_sha256": sha256(candidate["manifest_path"]),
        "source_run_manifest_sha256": sha256(candidate["run_manifest_path"]),
        "source_video_validation": {key: source_validation.get(key) for key in (
            "sha256", "bytes", "valid", "full_decode", "frame_count", "fps", "resolution",
            "duration_s", "frame_pts_sha256", "decoded_frame_checksums_sha256")},
        "checkpoint": receipt_checkpoint,
        "training_lineage": training_lineage,
        "control_method": CONTROL_METHOD,
        "policy_sampling_mode": "deterministic_conditional_mean",
        "physical_result": result_label, "physical_task_success": success,
        "termination_reason": termination, "source_acceptance_error": candidate["manifest"].get("source_acceptance_error"),
        "capture_observations": capture_observations,
        "full_episode_continuous": True, "normal_speed": True, "single_episode": True,
        "stitched": False, "speed_modified": False, "extra_intro_frames": 0,
        "full": full, "detail": detail, "historical_N_comparison": pair,
        "selected_response_json": str(response_json), "selected_response_json_sha256": sha256(response_json),
        "selected_response_markdown": str(response_md), "selected_response_markdown_sha256": sha256(response_md),
        "claims": claims}
    write_new_json(destination / "export_receipt.json", receipt)
    summary = {"physical_result": result_label, "checkpoint": checkpoint["decisions"],
        "full": full["output"], "detail": detail.get("output"), "comparison": pair["output"],
        "receipt": str(destination / "export_receipt.json")}
    if display_id is not None:
        summary["display_id"] = display_id
    print(json.dumps(summary, indent=2))
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--historical-n-source", type=Path, default=DEFAULT_HISTORICAL_N)
    parser.add_argument("--historical-n-version", default=DEFAULT_HISTORICAL_VERSION)
    args = parser.parse_args()
    export(args.source, args.destination, args.historical_n_source, args.historical_n_version)


if __name__ == "__main__":
    main()
