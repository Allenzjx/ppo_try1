"""Export one sealed rear-policy evaluation without starting Isaac.

The source must be one deterministic natural-P01 evaluation from
``ppo_rr_rl_timing_policy_learning_v1``.  The exporter keeps every real frame
on the native 15 fps timeline, labels the retained FL assist separately from
the disabled rear task assist, preserves an incomplete tail, and compares it
with the accepted historical N only as a visibly frozen reference.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = Path(__file__).resolve().parent
SHARED_EXPORTER = (ROOT / "outputs" / "ppo_p05_hip_only_continuation_v1" /
                   "export_p05_capture_video.py")
DEFAULT_HISTORICAL_N = (ROOT / "runs" / "ppo_task_conditioned_hip_wheel_v1" /
    "video_eval" / "prior_B" /
    "20260921T0701155546642Z_gee5a9651591d_64985c63292d4a9b966716c3a54659a1" /
    "source")
DEFAULT_HISTORICAL_VERSION = "N_ref historical ee5a9651591d 73.808s"

EXPERIMENT = "rr_rl_timing_policy_learning_v1"
CONTROL_METHOD = "PPO_PLUS_FL_CAPTURE_ASSIST_REAR_TASK_ASSIST_OFF_WITH_INHERITED_LIMITED_AUX"
MEDIA_COMPAT_SCHEMA = "wlr50_clean.semantic_video_media_only_checkpoint_compatibility.v1"
MEDIA_COMPAT_FILES = {
    "src/wlr50_clean/ppo/semantic_video.py",
    "src/wlr50_clean/ppo/semantic_video_cli.py",
}
MIGRATION_SCHEMA = "wlr50_clean.rear_policy_timing_append.v1"
RECAPTURE_SCHEMA = "wlr50_clean.rear_recapture_same419.v1"
RECAPTURE_FACTOR_KEY = "rear_recapture_same419_factor"
RECAPTURE_MODE = "rr_recapture_current_support_v2"
RECAPTURE_BRANCH = "ancestor220544_recapture_v2"
RECAPTURE_SOURCE_SELECTION = {
    "source_role": "explicit_initial419_ancestor_recapture_branch",
    "checkpoint_sha256": "e83b7340febb760fdd00aa5cf44c07bbf45809a2f9a175b1eafd0d670c3cef85",
    "manifest_sha256": "3fba5f7d8b32f98420924fc5103b4ea4f58205cef004651658152ea011dc05f6",
    "counters": {"global_policy_decisions": 220544,
                 "ppo_updates": 1688, "optimizer_steps": 33760},
    "output_branch": RECAPTURE_BRANCH,
}
LIVE_SWING_SCHEMA = "wlr50_clean.rear_live_swing_same419.v3"
LIVE_SWING_FACTOR_KEY = "rear_live_swing_same419_factor"
LIVE_SWING_MODE = "rr_live_swing_evidence_v3"
LIVE_SWING_SOURCE_HEAD = "44219b4fdc4d36d33be489b833c03b897766045b"
LIVE_SWING_SOURCE_SHA = "1827b5d59935b31e1e27cc7ae0c364dd0203a9d7f77a7d9e9502189bbd31c7d5"
LIVE_SWING_SOURCE_MANIFEST_SHA = "b27d373edb11ed5d9455bc3d6b32dda4657c9b08b08f91d9d42de0d425686b75"
LIVE_SWING_ORIGIN = {"global_policy_decisions": 221568,
                     "ppo_updates": 1696, "optimizer_steps": 33920}
P02_PROGRESS_SCHEMA = "wlr50_clean.p02_progress_append.v1"
P02_PROGRESS_FACTOR_KEY = "p02_progress_append_factor"
P02_PROGRESS_MIGRATION = "p02_progress_migration"
P02_PROGRESS_SOURCE_HEAD = "45862675a18a5bb2b75f0d4883e0cee2a3be24e3"
P02_PROGRESS_SOURCE_SHA = "da63ac820ab7b260e161a116b5c925370090c087b4cce222859626b8faec24e1"
P02_PROGRESS_SOURCE_MANIFEST_SHA = "956737aab9e91c3fc11d77a237f4ec03613877c53081f16c4b6b81e82c5501b6"
P02_PROGRESS_ORIGIN = {"global_policy_decisions": 221696,
                       "ppo_updates": 1697, "optimizer_steps": 33940}
P02_PROGRESS_POLICY_VERSION = "rear_policy_p02_progress_history_v1"
P02_PROGRESS_OBSERVATION_LAYOUT = "role419_p02_progress_v1"
P02_PROGRESS_MODE = "p02_measured_progress_credit_v1"
P02_PROGRESS_GROUP = "p02_progress_credit_full3"
P02_PROGRESS_FIELDS = (
    "p02_best_remaining_m", "p02_progress_credit_fraction",
    "p02_current_eligible",
)
COOPERATIVE_PREP_SCHEMA = "wlr50_clean.cooperative_prep_same422.v4"
COOPERATIVE_PREP_FACTOR_KEY = "cooperative_prep_same422_factor"
COOPERATIVE_PREP_MIGRATION = "cooperative_prep_migration"
COOPERATIVE_PREP_SOURCE_HEAD = "d7e97ee7b7e493d4f3ff34f9c8762f73550ad7bd"
COOPERATIVE_PREP_SOURCE_SHA = "2739173651e516ab19a5213e6d3206f7c970e7d76ed0f3adbde1bc95b7dc86c7"
COOPERATIVE_PREP_SOURCE_MANIFEST_SHA = "c6e0cd978202509cf73d41e5814fe9ad59051a6e3c9f1f14d5f689c8b7c4b550"
COOPERATIVE_PREP_ORIGIN = {"global_policy_decisions": 223232,
                           "ppo_updates": 1709, "optimizer_steps": 34180}
COOPERATIVE_PREP_POLICY_VERSION = "rear_cooperative_prep_history_v1"
COOPERATIVE_PREP_ACTOR_CLASS = (
    "wlr50_clean.ppo.semantic_rear_cooperative_prep_actor:"
    "SemanticRearCooperativePrepHistoryMLPModel")
COOPERATIVE_PREPARATION_REVISION = "rr_capture_cooperative_preparation_v4"
COOPERATIVE_PREP_CHANGED_PATHS = frozenset({
    "configs/ppo_rr_rl_timing_policy_learning_v1/execution_profile.yaml",
    "configs/ppo_rr_rl_timing_policy_learning_v1/reward_config.yaml",
    "configs/ppo_rr_rl_timing_policy_learning_v1/stage_task_spec.yaml",
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
COOPERATIVE_PREP_CHANGED_CONFIGS = frozenset({
    "execution_profile.yaml", "reward_config.yaml", "stage_task_spec.yaml",
})
POLICY_VERSION = "rr_rl_timing_policy_learning_history_v1"
OBSERVATION_LAYOUT = "role410_rear_policy_timing_v1"
NOMINAL_TIMING = "rr_capture_before_rl_transfer_v1"
FL_ASSIST_MODE = "p05_hip_only_continuation_v1"
CAPTURE_ABORT_SOURCE_ERROR = (
    "VideoArtifactError: VIDEO_OR_ARTIFACT_ERROR: encode failed: RuntimeError: "
    "viewport callback_count=0, expected 1")
CAPTURE_ABORT_ENCODER_ERROR = (
    "encode failed: RuntimeError: viewport callback_count=0, expected 1")
COUNTERS = ("global_policy_decisions", "ppo_updates", "optimizer_steps")
FPS, HZ, STRIDE, MAX_FRAMES = 15, 120, 8, 3000
PANEL_HEIGHT = 232
SERVO_ORDER = (
    "front_left_hip", "front_left_knee", "front_right_hip", "front_right_knee",
    "rear_left_hip", "rear_left_knee", "rear_right_hip", "rear_right_knee",
)
WHEEL_ORDER = (
    "front_left_ankle", "front_right_ankle", "rear_left_ankle", "rear_right_ankle",
)
WHEEL_FORWARD_SIGN = {
    "front_left_ankle": -1., "front_right_ankle": 1.,
    "rear_left_ankle": -1., "rear_right_ankle": 1.,
}
REAR_TIMING_FLAGS = (
    "rr_carry_capture", "rr_support_handoff", "rl_prep_transfer",
    "rl_swing_capture", "p09_dependency_wait", "p12_dependency_wait",
)
REAR_TIMING_CLOCKS = (
    "p09_source_time_s", "p12_source_time_s", "p12_rl_source_time_s",
)

_SHARED: Any = None


def shared() -> Any:
    global _SHARED
    if _SHARED is None:
        spec = importlib.util.spec_from_file_location("_rear_policy_media_shared", SHARED_EXPORTER)
        require(spec is not None and spec.loader is not None,
                "shared media helper cannot be loaded")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _SHARED = module
    return _SHARED


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def sha256(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"),
                     allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def checked_sha(value: str, label: str) -> str:
    result = str(value).lower()
    require(len(result) == 64 and all(ch in "0123456789abcdef" for ch in result),
            f"{label} must be one lowercase SHA-256")
    return result


def checked_head(value: str, label: str = "--expected-head") -> str:
    result = str(value).lower()
    require(len(result) == 40 and all(ch in "0123456789abcdef" for ch in result),
            f"{label} must be one 40-hex frozen commit")
    return result


def checkpoint_runtime_head(args: argparse.Namespace) -> str:
    value = args.checkpoint_runtime_head or args.expected_head
    return checked_head(value, "--checkpoint-runtime-head")


def write_new_json(path: Path, value: Any) -> None:
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def artifact(source: Path, manifest: dict[str, Any], name: str) -> Path:
    record = (manifest.get("artifacts") or {}).get(name)
    require(isinstance(record, dict), f"source manifest lacks {name}")
    path = Path(record.get("path", "")).resolve(strict=True)
    require(path.parent == source and path.name == name,
            f"{name} is outside the sealed source")
    require(path.stat().st_size == record.get("bytes") and
            sha256(path) == record.get("sha256"),
            f"{name} changed after source manifest creation")
    return path


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        return float(value)
    return None


def _vector(value: Any, size: int, label: str) -> list[float]:
    require(isinstance(value, list) and len(value) == size, f"tick lacks {label}")
    result = [_number(item) for item in value]
    require(all(item is not None for item in result), f"{label} is non-finite")
    return [float(item) for item in result]


def _joint(row: dict[str, Any], name: str) -> float | None:
    return _number(((row.get("joints") or {}).get(name) or {}).get("position_deg"))


def _contact(row: dict[str, Any]) -> tuple[str, float | None]:
    if row.get("top_surface_contact") is True:
        label = "TOP"
    elif row.get("ground_contact") is True:
        label = "GROUND"
    elif row.get("air") is True:
        label = "AIR"
    else:
        label = str(row.get("contact_surface") or "NO_TOP")
    force = next((_number(row.get(key)) for key in (
        "obstacle_normal_force_n", "contact_reaction_force_n", "bearing_force_n")
        if _number(row.get(key)) is not None), None)
    return label, force


def _leg(row: dict[str, Any], name: str) -> dict[str, Any]:
    leg = (row.get("current_legs") or {}).get(name)
    require(isinstance(leg, dict), f"tick lacks current {name} physical state")
    return leg


def _gap_mm(leg: dict[str, Any]) -> float | None:
    value = _number(leg.get("clearance_m"))
    return None if value is None else 1000. * value


def _front_mm(leg: dict[str, Any]) -> float | None:
    value = _number(leg.get("front_distance_m"))
    return None if value is None else 1000. * value


def inherited_auxiliary(metadata: dict[str, Any], preserved: dict[str, Any]) -> dict[str, Any]:
    legacy_branch = metadata.get("task_conditioned_hip_wheel_branch")
    mixed_branch = metadata.get("rr_postcross_workspace_branch")
    require(isinstance(legacy_branch, dict) and isinstance(mixed_branch, dict),
            "checkpoint lost inherited limited-AUX lineage")
    require(json_digest(legacy_branch) == preserved.get("task_conditioned_hip_wheel_branch")
            and json_digest(mixed_branch) == preserved.get("rr_postcross_workspace_branch"),
            "inherited AUX branches differ from the migration-time bindings")
    legacy = legacy_branch.get("auxiliary_mean_learning")
    mixed = mixed_branch.get("front_rehearsal_auxiliary")
    require(isinstance(legacy, dict) and isinstance(mixed, dict),
            "inherited AUX ledgers are missing")
    values = {
        "legacy_limited_AUX": {
            "accepted": legacy.get("accepted_auxiliary_updates_total"),
            "attempted": legacy.get("attempted_auxiliary_optimizer_steps_total")},
        "mixed_front_and_RR_AUX": {
            "accepted": mixed.get("accepted_auxiliary_updates_total"),
            "attempted": mixed.get("attempted_auxiliary_optimizer_steps_total")},
    }
    require(all(type(value) is int and value >= 0
                for group in values.values() for value in group.values()),
            "inherited AUX counts are malformed")
    return values


def rear_recapture_identity(metadata: dict[str, Any], runtime: dict[str, Any],
                            initial_migration: dict[str, Any], policy: dict[str, Any],
                            counters: dict[str, int], checkpoint: Path
                            ) -> dict[str, Any] | None:
    receipt = metadata.get("rear_recapture_migration")
    if receipt is None:
        require(metadata.get("checkpoint_output_routing") is None,
                "legacy rear-policy checkpoint has unexpected branch routing")
        return None
    factor = receipt.get(RECAPTURE_FACTOR_KEY) if isinstance(receipt, dict) else None
    require(receipt.get("schema") == RECAPTURE_SCHEMA and
            isinstance(factor, dict) and factor.get("schema") == RECAPTURE_SCHEMA and
            receipt.get("source_selection") == RECAPTURE_SOURCE_SELECTION and
            factor.get("source_selection") == RECAPTURE_SOURCE_SELECTION and
            receipt.get("source_checkpoint_sha256") ==
                RECAPTURE_SOURCE_SELECTION["checkpoint_sha256"] and
            receipt.get("source_manifest_sha256") ==
                RECAPTURE_SOURCE_SELECTION["manifest_sha256"],
            "checkpoint lacks the exact registered rear-recapture source")
    origin = factor.get("revision_counter_origin")
    require(origin == RECAPTURE_SOURCE_SELECTION["counters"] and
            factor.get("original_branch_origin") == origin and
            factor.get("source_mode") == NOMINAL_TIMING and
            factor.get("target_mode") == RECAPTURE_MODE and
            factor.get("source_policy_contract") == policy and
            factor.get("target_policy_contract") == policy and
            factor.get("observation_dimension") == 419 and
            factor.get("action_dimension") == 12 and factor.get("num_envs") == 1 and
            factor.get("parameter_mapping") ==
                "identity_all_actor_critic_parameters_buffers_and_full_Adam" and
            factor.get("physical_dynamics_changed") is False and
            factor.get("same_mdp_claimed") is False and
            factor.get("old_rollout_inherited") is False and
            all(factor.get("added_" + key) == 0 for key in
                ("policy_decisions", "ppo_updates", "optimizer_steps", "auxiliary_updates")),
            "rear-recapture factor changes policy shape/state or invents migration learning")
    require(receipt.get("source_git_commit") ==
                initial_migration.get("target_git_commit") and
            receipt.get("source_runtime_content_sha256") ==
                initial_migration.get("target_runtime_content_sha256") and
            receipt.get("source_contract_sha256") ==
                initial_migration.get("target_contract_sha256") and
            receipt.get("target_git_commit") == runtime.get("source_git_commit") and
            receipt.get("target_runtime_content_sha256") ==
                runtime.get("runtime_content_sha256") and
            receipt.get("target_contract_sha256") == json_digest(runtime) and
            (factor.get("preserved_metadata_sha256") or {}).get(
                "rear_policy_timing_migration") == json_digest(initial_migration),
            "rear-recapture ancestor/source/target runtime chain differs")
    revision_counts = {}
    for key in COUNTERS:
        require(type(counters[key]) is int and counters[key] >= origin[key],
                f"invalid rear-recapture {key} origin")
        revision_counts[key] = counters[key] - origin[key]
    plan = Path(receipt.get("plan_path", "")).resolve(strict=True)
    require(sha256(plan) == receipt.get("plan_sha256") and
            read_json(plan) == {key: value for key, value in receipt.items()
                                if key not in ("plan_path", "plan_sha256")},
            "rear-recapture migration plan changed after publication")
    source_checkpoint = Path(receipt.get("source_checkpoint", "")).resolve(strict=True)
    source_manifest = source_checkpoint.with_name(
        source_checkpoint.stem + "_manifest.json").resolve(strict=True)
    require(sha256(source_checkpoint) == RECAPTURE_SOURCE_SELECTION["checkpoint_sha256"] and
            sha256(source_manifest) == RECAPTURE_SOURCE_SELECTION["manifest_sha256"],
            "rear-recapture registered initial419 source bytes changed")
    changed = receipt.get("changed_file_hashes") or {}
    media_changed = set(changed) & MEDIA_COMPAT_FILES
    media_review = factor.get("media_revision_review")
    if media_changed:
        require(media_changed == MEDIA_COMPAT_FILES and isinstance(media_review, dict) and
                media_review.get("schema") ==
                    "wlr50_clean.reviewed_video_binding_compatibility.v1" and
                media_review.get("reviewed_media_commit") ==
                    "a9825050e4f92ac55153c2501fb5af59fc71e9c1" and
                media_review.get("changed_file_hashes") ==
                    {path: changed[path] for path in sorted(MEDIA_COMPAT_FILES)} and
                media_review.get("task_control_changed_by_media_revision") is False and
                media_review.get("policy_weights_changed_by_media_revision") is False and
                media_review.get("added_learning") == 0,
                "rear-recapture media review differs from its exact independent pair")
    else:
        require(media_review is None,
                "rear-recapture claims a media review without a media delta")
    branch_root = (OUTPUT_ROOT / "branches" / RECAPTURE_BRANCH).resolve()
    routing = metadata.get("checkpoint_output_routing")
    require(routing == {"schema": "wlr50_clean.checkpoint_output_routing.v1",
            "branch": RECAPTURE_BRANCH, "output_root": str(branch_root),
            "main_latest_pointer_promotion": False,
            "source_selection": RECAPTURE_SOURCE_SELECTION} and
            checkpoint.parent == branch_root / "checkpoints" / "history",
            "rear-recapture checkpoint lacks exact isolated output routing")
    return {"schema": RECAPTURE_SCHEMA, "target_mode": RECAPTURE_MODE,
        "revision_counter_origin": origin, "revision_branch_counts": revision_counts,
        "migration_plan": str(plan), "migration_plan_sha256": receipt["plan_sha256"],
        "source_selection": RECAPTURE_SOURCE_SELECTION,
        "checkpoint_output_routing": routing, "new_auxiliary_updates": 0,
        "media_revision_review": media_review}


def reconstruct_live_swing_source_runtime(runtime: dict[str, Any],
                                          receipt: dict[str, Any]) -> dict[str, Any]:
    """Reverse the sealed v3 delta so the inherited v2 receipt sees its own runtime."""
    require(isinstance(receipt, dict) and receipt.get("schema") == LIVE_SWING_SCHEMA and
            receipt.get("target_git_commit") == runtime.get("source_git_commit") and
            receipt.get("target_runtime_content_sha256") ==
                runtime.get("runtime_content_sha256") and
            receipt.get("target_contract_sha256") == json_digest(runtime),
            "live-swing receipt does not bind the current checkpoint runtime")
    previous = copy.deepcopy(runtime)
    previous["source_git_commit"] = receipt.get("source_git_commit")
    previous["runtime_content_sha256"] = receipt.get("source_runtime_content_sha256")
    changed = receipt.get("changed_file_hashes")
    require(isinstance(changed, dict) and changed,
            "live-swing receipt lacks its finite runtime delta")
    files = previous.get("files")
    require(isinstance(files, dict), "live-swing runtime lacks a file inventory")
    for path, binding in changed.items():
        require(isinstance(path, str) and isinstance(binding, dict) and
                set(binding) == {"before", "after"} and
                files.get(path) == binding.get("after"),
                "live-swing changed-file binding differs from current runtime")
        before = binding.get("before")
        if before is None:
            files.pop(path, None)
        else:
            require(isinstance(before, str) and len(before) == 64,
                    "live-swing source file hash is malformed")
            files[path] = before
    source_configuration = receipt.get("source_changed_configuration")
    selected = previous.get("selected_configuration")
    require(isinstance(source_configuration, dict) and
            set(source_configuration) == {"execution_profile.yaml", "stage_task_spec.yaml"} and
            isinstance(selected, dict) and set(source_configuration) <= set(selected),
            "live-swing source configuration delta differs")
    selected.update(copy.deepcopy(source_configuration))
    require(previous.get("source_git_commit") == LIVE_SWING_SOURCE_HEAD and
            json_digest(previous) == receipt.get("source_contract_sha256"),
            "live-swing historical v2 runtime reconstruction is not exact")
    return previous


def rear_live_swing_identity(metadata: dict[str, Any], runtime: dict[str, Any],
                             source_runtime: dict[str, Any], policy: dict[str, Any],
                             counters: dict[str, int], checkpoint: Path,
                             recapture: dict[str, Any] | None
                             ) -> dict[str, Any] | None:
    receipt = metadata.get("rear_live_swing_migration")
    if receipt is None:
        return None
    factor = receipt.get(LIVE_SWING_FACTOR_KEY) if isinstance(receipt, dict) else None
    require(recapture is not None and receipt.get("schema") == LIVE_SWING_SCHEMA and
            isinstance(factor, dict) and factor.get("schema") == LIVE_SWING_SCHEMA and
            receipt.get("source_role") ==
                "latest_sealed_learned_recapture_branch_not_ancestor" and
            receipt.get("source_checkpoint_sha256") == LIVE_SWING_SOURCE_SHA and
            receipt.get("source_manifest_sha256") == LIVE_SWING_SOURCE_MANIFEST_SHA and
            receipt.get("source_git_commit") == LIVE_SWING_SOURCE_HEAD and
            receipt.get("source_contract_sha256") == json_digest(source_runtime) and
            receipt.get("source_runtime_content_sha256") ==
                source_runtime.get("runtime_content_sha256") and
            receipt.get("target_git_commit") == runtime.get("source_git_commit") and
            receipt.get("target_contract_sha256") == json_digest(runtime) and
            receipt.get("target_runtime_content_sha256") ==
                runtime.get("runtime_content_sha256") and
            receipt.get("discard_old_rollout_storage") is True,
            "checkpoint lacks the exact learned-v2 to live-swing runtime boundary")
    require(factor.get("source_mode") == RECAPTURE_MODE and
            factor.get("target_mode") == LIVE_SWING_MODE and
            factor.get("revision_counter_origin") == LIVE_SWING_ORIGIN and
            factor.get("original_branch_origin") == RECAPTURE_SOURCE_SELECTION["counters"] and
            factor.get("source_policy_contract") == policy and
            factor.get("target_policy_contract") == policy and
            factor.get("observation_dimension") == 419 and
            factor.get("action_dimension") == 12 and factor.get("num_envs") == 1 and
            factor.get("parameter_mapping") ==
                "identity_all_actor_critic_buffers_full_Adam_and_rng" and
            factor.get("same_numeric_input_Gaussian_preserved") is True and
            factor.get("same_mdp_claimed") is False and
            factor.get("physical_dynamics_changed") is False and
            factor.get("old_rollout_inherited") is False and
            all(factor.get("added_" + key) == 0 for key in
                ("policy_decisions", "ppo_updates", "optimizer_steps", "auxiliary_updates")),
            "live-swing factor changes learned state or invents migration learning")
    preserved = factor.get("preserved_metadata_sha256") or {}
    routing = metadata.get("checkpoint_output_routing")
    require(preserved.get("rear_recapture_migration") ==
                json_digest(metadata.get("rear_recapture_migration")) and
            preserved.get("checkpoint_output_routing") == json_digest(routing) and
            routing == recapture.get("checkpoint_output_routing"),
            "live-swing migration lost the parent recapture receipt or output route")
    revision_counts = {}
    for key in COUNTERS:
        require(type(counters[key]) is int and counters[key] >= LIVE_SWING_ORIGIN[key],
                f"invalid live-swing {key} origin")
        revision_counts[key] = counters[key] - LIVE_SWING_ORIGIN[key]
    plan = Path(receipt.get("plan_path", "")).resolve(strict=True)
    require(sha256(plan) == receipt.get("plan_sha256") and
            read_json(plan) == {key:value for key,value in receipt.items()
                                if key not in ("plan_path", "plan_sha256")},
            "live-swing migration plan changed after publication")
    source_checkpoint = Path(receipt.get("source_checkpoint", "")).resolve(strict=True)
    source_manifest = source_checkpoint.with_name(
        source_checkpoint.stem + "_manifest.json").resolve(strict=True)
    branch_root = (OUTPUT_ROOT / "branches" / RECAPTURE_BRANCH).resolve()
    require(source_checkpoint.parent == branch_root / "checkpoints" / "history" and
            sha256(source_checkpoint) == LIVE_SWING_SOURCE_SHA and
            sha256(source_manifest) == LIVE_SWING_SOURCE_MANIFEST_SHA and
            checkpoint.parent == branch_root / "checkpoints" / "history",
            "live-swing source/current checkpoint escaped its exact branch binding")
    return {"schema":LIVE_SWING_SCHEMA, "target_mode":LIVE_SWING_MODE,
        "revision_counter_origin":LIVE_SWING_ORIGIN,
        "revision_branch_counts":revision_counts,
        "migration_plan":str(plan), "migration_plan_sha256":receipt["plan_sha256"],
        "source_checkpoint":str(source_checkpoint),
        "source_checkpoint_sha256":LIVE_SWING_SOURCE_SHA,
        "source_manifest_sha256":LIVE_SWING_SOURCE_MANIFEST_SHA,
        "checkpoint_output_routing":routing, "added_auxiliary_updates":0}


def reconstruct_p02_progress_source_runtime(runtime: dict[str, Any],
                                            receipt: dict[str, Any]) -> dict[str, Any]:
    """Reverse only the sealed 419->422 append for inherited-lineage checks."""
    require(isinstance(receipt, dict) and
            receipt.get("schema") == P02_PROGRESS_SCHEMA and
            receipt.get("target_git_commit") == runtime.get("source_git_commit") and
            receipt.get("target_runtime_content_sha256") ==
                runtime.get("runtime_content_sha256") and
            receipt.get("target_contract_sha256") == json_digest(runtime) and
            receipt.get("discard_old_rollout_storage") is True,
            "P02-progress receipt does not bind the current checkpoint runtime")
    previous = copy.deepcopy(runtime)
    previous["source_git_commit"] = receipt.get("source_git_commit")
    previous["runtime_content_sha256"] = receipt.get("source_runtime_content_sha256")
    changed = receipt.get("changed_file_hashes")
    require(isinstance(changed, dict) and changed,
            "P02-progress receipt lacks its finite runtime delta")
    files = previous.get("files")
    require(isinstance(files, dict), "P02-progress runtime lacks a file inventory")
    for path, binding in changed.items():
        require(isinstance(path, str) and isinstance(binding, dict) and
                set(binding) == {"before", "after"} and
                files.get(path) == binding.get("after"),
                "P02-progress changed-file binding differs from current runtime")
        before = binding.get("before")
        if before is None:
            files.pop(path, None)
        else:
            require(isinstance(before, str) and len(before) == 64 and
                    all(ch in "0123456789abcdef" for ch in before),
                    "P02-progress source file hash is malformed")
            files[path] = before
    source_configuration = receipt.get("source_changed_configuration")
    required_configuration = {
        "execution_profile.yaml", "stage_task_spec.yaml",
        "observation_schema.json", "curriculum_plan.json",
    }
    selected = previous.get("selected_configuration")
    require(isinstance(source_configuration, dict) and
            set(source_configuration) == required_configuration and
            isinstance(selected, dict) and required_configuration <= set(selected),
            "P02-progress source configuration delta differs")
    selected.update(copy.deepcopy(source_configuration))
    require(previous.get("source_git_commit") == P02_PROGRESS_SOURCE_HEAD and
            json_digest(previous) == receipt.get("source_contract_sha256"),
            "P02-progress historical 419 runtime reconstruction is not exact")
    return previous


def p02_progress_identity(metadata: dict[str, Any], runtime: dict[str, Any],
                          source_runtime: dict[str, Any],
                          source_policy: dict[str, Any], policy: dict[str, Any],
                          counters: dict[str, int], checkpoint: Path,
                          live_swing: dict[str, Any] | None
                          ) -> dict[str, Any] | None:
    receipt = metadata.get(P02_PROGRESS_MIGRATION)
    if receipt is None:
        return None
    factor = (receipt.get(P02_PROGRESS_FACTOR_KEY)
              if isinstance(receipt, dict) else None)
    require(live_swing is not None and
            receipt.get("schema") == P02_PROGRESS_SCHEMA and
            isinstance(factor, dict) and
            factor.get("schema") == P02_PROGRESS_SCHEMA and
            receipt.get("source_checkpoint_sha256") == P02_PROGRESS_SOURCE_SHA and
            receipt.get("source_manifest_sha256") ==
                P02_PROGRESS_SOURCE_MANIFEST_SHA and
            receipt.get("source_git_commit") == P02_PROGRESS_SOURCE_HEAD and
            receipt.get("source_contract_sha256") == json_digest(source_runtime) and
            receipt.get("source_runtime_content_sha256") ==
                source_runtime.get("runtime_content_sha256") and
            receipt.get("target_git_commit") == runtime.get("source_git_commit") and
            receipt.get("target_contract_sha256") == json_digest(runtime) and
            receipt.get("target_runtime_content_sha256") ==
                runtime.get("runtime_content_sha256") and
            receipt.get("discard_old_rollout_storage") is True,
            "checkpoint lacks the exact learned-419 to P02-progress-422 boundary")
    require(factor.get("revision_counter_origin") == P02_PROGRESS_ORIGIN and
            factor.get("original_branch_origin") ==
                RECAPTURE_SOURCE_SELECTION["counters"] and
            factor.get("source_policy_contract") == source_policy and
            factor.get("target_policy_contract") == policy and
            factor.get("actor_critic_mapping") ==
                "old419_columns_exact_new3_zero" and
            factor.get("Adam_mapping") ==
                "all_old_moments_steps_options_exact_new3_first_weight_moment_columns_zero" and
            factor.get("same_mdp_claimed") is False and
            factor.get("physical_dynamics_changed") is False and
            factor.get("rear_control_changed") is False and
            factor.get("raw_sigma_kernel") ==
                "unchanged_rear419_slice_0_419; new_features_only_enter_MLP" and
            factor.get("old_rollout_inherited") is False and
            all(factor.get("added_" + key) == 0 for key in
                ("policy_decisions", "ppo_updates", "optimizer_steps",
                 "auxiliary_updates")),
            "P02-progress factor changes old learned state or invents migration learning")
    preserved = factor.get("preserved_metadata_sha256") or {}
    protected = (
        "rear_live_swing_migration", "rear_recapture_migration",
        "rear_policy_timing_migration", "rear_policy_timing_branch",
        "checkpoint_output_routing",
    )
    require(all(preserved.get(key) == json_digest(metadata.get(key))
                for key in protected),
            "P02-progress migration lost its inherited source lineage")
    routing = metadata.get("checkpoint_output_routing")
    require(routing == live_swing.get("checkpoint_output_routing"),
            "P02-progress migration changed the isolated branch route")
    revision_counts = {}
    for key in COUNTERS:
        require(type(counters[key]) is int and counters[key] >= P02_PROGRESS_ORIGIN[key],
                f"invalid P02-progress {key} origin")
        revision_counts[key] = counters[key] - P02_PROGRESS_ORIGIN[key]
    plan = Path(receipt.get("plan_path", "")).resolve(strict=True)
    require(sha256(plan) == receipt.get("plan_sha256") and
            read_json(plan) == {key: value for key, value in receipt.items()
                                if key not in ("plan_path", "plan_sha256")},
            "P02-progress migration plan changed after publication")
    source_checkpoint = Path(receipt.get("source_checkpoint", "")).resolve(strict=True)
    source_manifest = source_checkpoint.with_name(
        source_checkpoint.stem + "_manifest.json").resolve(strict=True)
    branch_root = (OUTPUT_ROOT / "branches" / RECAPTURE_BRANCH).resolve()
    require(source_checkpoint.parent == branch_root / "checkpoints" / "history" and
            sha256(source_checkpoint) == P02_PROGRESS_SOURCE_SHA and
            sha256(source_manifest) == P02_PROGRESS_SOURCE_MANIFEST_SHA and
            checkpoint.parent == branch_root / "checkpoints" / "history",
            "P02-progress source/current checkpoint escaped its exact branch binding")
    return {"schema": P02_PROGRESS_SCHEMA,
        "policy_version": P02_PROGRESS_POLICY_VERSION,
        "observation_layout": P02_PROGRESS_OBSERVATION_LAYOUT,
        "observation_dimension": 422, "preserved_prefix_dimension": 419,
        "p02_progress_credit_mode": P02_PROGRESS_MODE,
        "revision_counter_origin": P02_PROGRESS_ORIGIN,
        "revision_branch_counts": revision_counts,
        "migration_plan": str(plan), "migration_plan_sha256": receipt["plan_sha256"],
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": P02_PROGRESS_SOURCE_SHA,
        "source_manifest_sha256": P02_PROGRESS_SOURCE_MANIFEST_SHA,
        "checkpoint_output_routing": routing, "added_auxiliary_updates": 0}


def reconstruct_cooperative_prep_source_runtime(runtime: dict[str, Any],
                                                receipt: dict[str, Any]) -> dict[str, Any]:
    """Reverse only the sealed same422 cooperative-prep boundary."""
    require(isinstance(receipt, dict) and
            receipt.get("schema") == COOPERATIVE_PREP_SCHEMA and
            receipt.get("target_git_commit") == runtime.get("source_git_commit") and
            receipt.get("target_runtime_content_sha256") ==
                runtime.get("runtime_content_sha256") and
            receipt.get("target_contract_sha256") == json_digest(runtime) and
            receipt.get("discard_old_rollout_storage") is True,
            "cooperative-prep receipt does not bind the current checkpoint runtime")
    previous = copy.deepcopy(runtime)
    previous["source_git_commit"] = receipt.get("source_git_commit")
    previous["runtime_content_sha256"] = receipt.get("source_runtime_content_sha256")
    changed = receipt.get("changed_file_hashes")
    require(isinstance(changed, dict) and
            set(changed) == COOPERATIVE_PREP_CHANGED_PATHS,
            "cooperative-prep runtime delta differs from the reviewed finite set")
    files = previous.get("files")
    require(isinstance(files, dict), "cooperative-prep runtime lacks a file inventory")
    for path, binding in changed.items():
        require(isinstance(binding, dict) and set(binding) == {"before", "after"} and
                files.get(path) == binding.get("after"),
                "cooperative-prep changed-file binding differs from current runtime")
        before = binding.get("before")
        if before is None:
            files.pop(path, None)
        else:
            require(isinstance(before, str) and len(before) == 64 and
                    all(ch in "0123456789abcdef" for ch in before),
                    "cooperative-prep source file hash is malformed")
            files[path] = before
    source_configuration = receipt.get("source_changed_configuration")
    selected = previous.get("selected_configuration")
    require(isinstance(source_configuration, dict) and
            set(source_configuration) == COOPERATIVE_PREP_CHANGED_CONFIGS and
            isinstance(selected, dict) and
            COOPERATIVE_PREP_CHANGED_CONFIGS <= set(selected),
            "cooperative-prep source configuration delta differs")
    selected.update(copy.deepcopy(source_configuration))
    require(previous.get("source_git_commit") == COOPERATIVE_PREP_SOURCE_HEAD and
            json_digest(previous) == receipt.get("source_contract_sha256"),
            "cooperative-prep historical P02 runtime reconstruction is not exact")
    return previous


def cooperative_prep_identity(metadata: dict[str, Any], runtime: dict[str, Any],
                              source_runtime: dict[str, Any],
                              source_policy: dict[str, Any], policy: dict[str, Any],
                              counters: dict[str, int], checkpoint: Path,
                              p02_progress: dict[str, Any] | None
                              ) -> dict[str, Any] | None:
    receipt = metadata.get(COOPERATIVE_PREP_MIGRATION)
    if receipt is None:
        return None
    factor = (receipt.get(COOPERATIVE_PREP_FACTOR_KEY)
              if isinstance(receipt, dict) else None)
    require(p02_progress is not None and
            receipt.get("schema") == COOPERATIVE_PREP_SCHEMA and
            isinstance(factor, dict) and
            factor.get("schema") == COOPERATIVE_PREP_SCHEMA and
            receipt.get("source_checkpoint_sha256") == COOPERATIVE_PREP_SOURCE_SHA and
            receipt.get("source_manifest_sha256") ==
                COOPERATIVE_PREP_SOURCE_MANIFEST_SHA and
            receipt.get("source_git_commit") == COOPERATIVE_PREP_SOURCE_HEAD and
            receipt.get("source_contract_sha256") == json_digest(source_runtime) and
            receipt.get("source_runtime_content_sha256") ==
                source_runtime.get("runtime_content_sha256") and
            receipt.get("target_git_commit") == runtime.get("source_git_commit") and
            receipt.get("target_contract_sha256") == json_digest(runtime) and
            receipt.get("target_runtime_content_sha256") ==
                runtime.get("runtime_content_sha256") and
            receipt.get("discard_old_rollout_storage") is True,
            "checkpoint lacks the exact learned422 cooperative-prep boundary")
    require(factor.get("revision_counter_origin") == COOPERATIVE_PREP_ORIGIN and
            factor.get("original_branch_origin") ==
                RECAPTURE_SOURCE_SELECTION["counters"] and
            factor.get("source_policy_contract") == source_policy and
            factor.get("target_policy_contract") == policy and
            factor.get("observation_dimension") == 422 and
            factor.get("observation_layout") == P02_PROGRESS_OBSERVATION_LAYOUT and
            factor.get("cooperative_preparation_revision") ==
                COOPERATIVE_PREPARATION_REVISION and
            factor.get("parameter_mapping") ==
                "identity_all_actor_critic_buffers_full_Adam_options_steps_and_rng" and
            factor.get("deterministic_mean_and_learned_parameters_preserved") is True and
            factor.get("conditional_sigma_semantics_changed") is True and
            factor.get("bounded_preparation_reward_semantics_changed") is True and
            factor.get("physical_dynamics_changed") is False and
            factor.get("source_pose_commands_changed") is False and
            factor.get("RR_assist_changed") is False and
            factor.get("wheel_shaping_changed") is False and
            factor.get("HISTORY_changed") is False and
            factor.get("old_rollout_inherited") is False and
            all(factor.get("added_" + key) == 0 for key in
                ("policy_decisions", "ppo_updates", "optimizer_steps",
                 "auxiliary_updates")),
            "cooperative-prep factor changes protected state or invents learning")
    preserved = factor.get("preserved_metadata_sha256") or {}
    route = metadata.get("checkpoint_output_routing")
    require(preserved.get(P02_PROGRESS_MIGRATION) ==
                json_digest(metadata.get(P02_PROGRESS_MIGRATION)) and
            preserved.get("checkpoint_output_routing") == json_digest(route) and
            route == p02_progress.get("checkpoint_output_routing"),
            "cooperative-prep migration lost P02 ancestry or branch routing")
    revision_counts = {}
    for key in COUNTERS:
        require(type(counters[key]) is int and
                counters[key] >= COOPERATIVE_PREP_ORIGIN[key],
                f"invalid cooperative-prep {key} origin")
        revision_counts[key] = counters[key] - COOPERATIVE_PREP_ORIGIN[key]
    plan = Path(receipt.get("plan_path", "")).resolve(strict=True)
    require(sha256(plan) == receipt.get("plan_sha256") and
            read_json(plan) == {key:value for key,value in receipt.items()
                                if key not in ("plan_path", "plan_sha256")},
            "cooperative-prep migration plan changed after publication")
    source_checkpoint = Path(receipt.get("source_checkpoint", "")).resolve(strict=True)
    source_manifest = source_checkpoint.with_name(
        source_checkpoint.stem + "_manifest.json").resolve(strict=True)
    branch_root = (OUTPUT_ROOT / "branches" / RECAPTURE_BRANCH).resolve()
    require(source_checkpoint.parent == branch_root / "checkpoints" / "history" and
            sha256(source_checkpoint) == COOPERATIVE_PREP_SOURCE_SHA and
            sha256(source_manifest) == COOPERATIVE_PREP_SOURCE_MANIFEST_SHA and
            checkpoint.parent == branch_root / "checkpoints" / "history",
            "cooperative-prep source/current checkpoint escaped its branch binding")
    return {"schema": COOPERATIVE_PREP_SCHEMA,
        "policy_version": COOPERATIVE_PREP_POLICY_VERSION,
        "observation_layout": P02_PROGRESS_OBSERVATION_LAYOUT,
        "observation_dimension": 422,
        "cooperative_preparation_revision": COOPERATIVE_PREPARATION_REVISION,
        "revision_counter_origin": COOPERATIVE_PREP_ORIGIN,
        "revision_branch_counts": revision_counts,
        "migration_plan": str(plan), "migration_plan_sha256": receipt["plan_sha256"],
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": COOPERATIVE_PREP_SOURCE_SHA,
        "source_manifest_sha256": COOPERATIVE_PREP_SOURCE_MANIFEST_SHA,
        "checkpoint_output_routing": route, "added_auxiliary_updates": 0}


def media_only_runtime_compatibility(manifest: dict[str, Any], metadata: dict[str, Any],
                                     binding: dict[str, Any], args: argparse.Namespace
                                     ) -> dict[str, Any] | None:
    """Validate the capture-runtime/checkpoint-runtime split recorded live."""
    proof = manifest.get("checkpoint_load_provenance") or {}
    source = metadata.get("runtime_contract") or {}
    evaluation = manifest.get("runtime_contract") or {}
    source_head = checkpoint_runtime_head(args)
    evaluation_head = checked_head(args.expected_head)
    require(source.get("source_git_commit") == source_head and
            evaluation.get("source_git_commit") == evaluation_head,
            "checkpoint/evaluation runtime heads differ from explicit invocation")
    receipt = proof.get("reviewed_media_only_runtime_compatibility")
    if source == evaluation:
        require(source_head == evaluation_head and receipt is None,
                "exact-runtime video has a spurious media compatibility receipt")
        return None
    require(source_head != evaluation_head and isinstance(receipt, dict) and
            receipt.get("schema") == MEDIA_COMPAT_SCHEMA and
            receipt.get("scope") == "semantic_video_cli_eval_only",
            "cross-runtime video lacks the reviewed media-only receipt")
    source_identity = {"source_git_commit": source_head,
                       "runtime_content_sha256": source.get("runtime_content_sha256")}
    evaluation_identity = {"source_git_commit": evaluation_head,
                           "runtime_content_sha256": evaluation.get("runtime_content_sha256")}
    require(receipt.get("source_runtime") == source_identity and
            receipt.get("evaluation_runtime") == evaluation_identity and
            proof.get("checkpoint_source_runtime") == source_identity and
            proof.get("capture_evaluation_runtime") == evaluation_identity,
            "media-only receipt runtime identities differ from source/checkpoint")
    source_files, evaluation_files = source.get("files"), evaluation.get("files")
    require(isinstance(source_files, dict) and isinstance(evaluation_files, dict) and
            set(source_files) == set(evaluation_files) and
            {path for path in source_files if source_files[path] != evaluation_files[path]}
                == MEDIA_COMPAT_FILES,
            "media-only receipt does not isolate the exact two media files")
    require(source.get("runtime_content_sha256") == json_digest(source_files) and
            evaluation.get("runtime_content_sha256") == json_digest(evaluation_files),
            "media-only runtime file inventory digest differs")
    ignored = {"source_git_commit", "runtime_content_sha256", "files"}
    require({key: value for key, value in source.items() if key not in ignored} ==
            {key: value for key, value in evaluation.items() if key not in ignored},
            "media-only runtime changed non-file contract metadata")
    delta = receipt.get("reviewed_media_delta")
    require(isinstance(delta, dict) and set(delta) == MEDIA_COMPAT_FILES and
            all(delta[path] == {"source_sha256": source_files[path],
                                "evaluation_sha256": evaluation_files[path]}
                for path in MEDIA_COMPAT_FILES),
            "media-only receipt hashes differ from the two runtime inventories")
    require(receipt.get("selected_configuration") == source.get("selected_configuration") ==
                evaluation.get("selected_configuration") and
            receipt.get("selected_configuration_unchanged") is True and
            receipt.get("all_other_runtime_files_identical") is True and
            receipt.get("build_video_core_ast_unchanged") is True and
            receipt.get("official_checkpoint_load_uses_source_runtime") is True and
            receipt.get("capture_manifest_uses_evaluation_runtime") is True and
            receipt.get("mdp_changed") is False and
            receipt.get("control_changed") is False and
            receipt.get("policy_distribution_changed") is False and
            receipt.get("training_allowed") is False,
            "media-only receipt scope/unchanged declarations differ")
    checkpoint_record = receipt.get("source_checkpoint") or {}
    require(checkpoint_record == {
        "path": binding.get("checkpoint"),
        "sha256": binding.get("checkpoint_sha256"),
        "manifest_path": binding.get("manifest"),
        "manifest_sha256": binding.get("manifest_sha256")},
        "media-only receipt checkpoint binding differs from official load proof")
    return receipt


def checkpoint_identity(manifest: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    proof = manifest.get("checkpoint_load_provenance") or {}
    require(proof.get("checkpoint_loaded_and_verified") is True and
            manifest.get("policy_sampling_mode") == "deterministic_conditional_mean" and
            proof.get("stochastic_policy") in (None, False) and proof.get("policy_seed") is None,
            "source is not deterministic evaluation of a verified saved checkpoint")
    binding = proof.get("source") or {}
    checkpoint = Path(binding.get("checkpoint", "")).resolve(strict=True)
    sidecar = Path(binding.get("manifest", "")).resolve(strict=True)
    expected_checkpoint = Path(args.checkpoint).resolve(strict=True)
    expected_sidecar = expected_checkpoint.with_name(expected_checkpoint.stem + "_manifest.json").resolve(strict=True)
    checkpoint_sha = checked_sha(args.checkpoint_sha256, "--checkpoint-sha256")
    sidecar_sha = checked_sha(args.checkpoint_manifest_sha256,
                              "--checkpoint-manifest-sha256")
    require(checkpoint == expected_checkpoint and sidecar == expected_sidecar and
            sha256(checkpoint) == checkpoint_sha == binding.get("checkpoint_sha256") and
            sha256(sidecar) == sidecar_sha == binding.get("manifest_sha256"),
            "video checkpoint binding differs from explicit checkpoint bytes")
    metadata = read_json(sidecar)
    counters = {key: getattr(args, "expected_" + key) for key in COUNTERS}
    require(all(type(value) is int and value >= 0 for value in counters.values()) and
            all(metadata.get(key) == value for key, value in counters.items()) and
            proof.get("saved_global_policy_decisions") == counters["global_policy_decisions"] and
            metadata.get("checkpoint_sha256") == checkpoint_sha and
            Path(metadata.get("checkpoint_path", "")).resolve() == checkpoint,
            "checkpoint counters/path differ from explicit invocation or video load proof")
    expected_head = checkpoint_runtime_head(args)
    runtime = metadata.get("runtime_contract") or {}
    require(runtime.get("experiment_id") == EXPERIMENT and
            runtime.get("source_git_commit") == expected_head,
            "checkpoint is from another experiment/runtime")
    media_compatibility = media_only_runtime_compatibility(
        manifest, metadata, binding, args)
    policy = metadata.get("policy_contract") or {}
    progress_receipt = metadata.get(P02_PROGRESS_MIGRATION)
    cooperative_receipt = metadata.get(COOPERATIVE_PREP_MIGRATION)
    if progress_receipt is None:
        require(cooperative_receipt is None,
                "cooperative-prep checkpoint lost its inherited P02 receipt")
        base_runtime = runtime
        base_policy = policy
        progress_runtime = None
        progress_policy = None
        require(policy.get("version") == POLICY_VERSION and
                policy.get("observation_layout") == OBSERVATION_LAYOUT and
                policy.get("observation_dimension") == 419,
                "checkpoint lacks the exact 419-column rear-policy contract")
    else:
        progress_runtime = (reconstruct_cooperative_prep_source_runtime(
            runtime, cooperative_receipt) if cooperative_receipt is not None else runtime)
        base_runtime = reconstruct_p02_progress_source_runtime(
            progress_runtime, progress_receipt)
        progress_factor = progress_receipt.get(P02_PROGRESS_FACTOR_KEY) or {}
        base_policy = progress_factor.get("source_policy_contract") or {}
        progress_policy = (cooperative_receipt.get(
            COOPERATIVE_PREP_FACTOR_KEY, {}).get("source_policy_contract", {})
            if cooperative_receipt is not None else policy)
        require(progress_policy.get("version") == P02_PROGRESS_POLICY_VERSION and
                progress_policy.get("actor_class") ==
                    "wlr50_clean.ppo.semantic_p02_progress_actor:SemanticP02ProgressHistoryMLPModel" and
                progress_policy.get("observation_layout") ==
                    P02_PROGRESS_OBSERVATION_LAYOUT and
                progress_policy.get("observation_dimension") == 422 and
                progress_policy.get("preserved_observation_prefix_dimension") == 419 and
                progress_policy.get("p02_progress_mode") == P02_PROGRESS_MODE and
                progress_policy.get("p02_progress_observation_group") ==
                    P02_PROGRESS_GROUP and
                progress_policy.get("p02_progress_observation_slice") == [419, 422] and
                progress_policy.get("p02_progress_observation_fields") ==
                    list(P02_PROGRESS_FIELDS) and
                progress_policy.get("p02_progress_feature_scales") == [1.0, 1.0, 1.0] and
                progress_policy.get("old_419_numerical_codec_preserved") is True and
                progress_policy.get("sigma_kernel_observation_slice") == [0, 419],
                "checkpoint lacks the exact zero-appended 422 P02-progress contract")
        if cooperative_receipt is None:
            require(policy == progress_policy,
                    "plain P02-progress checkpoint policy changed without a boundary")
        else:
            require(policy.get("version") == COOPERATIVE_PREP_POLICY_VERSION and
                    policy.get("actor_class") == COOPERATIVE_PREP_ACTOR_CLASS and
                    policy.get("source_policy_version") ==
                        P02_PROGRESS_POLICY_VERSION and
                    policy.get("observation_layout") ==
                        P02_PROGRESS_OBSERVATION_LAYOUT and
                    policy.get("observation_dimension") == 422 and
                    policy.get("preserved_observation_prefix_dimension") == 422 and
                    policy.get("observation_schema_change") is False and
                    policy.get("parameter_topology_change") is False and
                    policy.get("deterministic_mean_change") is False and
                    policy.get("stochastic_kernel_change") is True and
                    policy.get("sigma_kernel_observation_slice") == [0, 422] and
                    policy.get("extra_filter_or_sampling_rejection") is False,
                    "checkpoint lacks the exact same422 cooperative-prep policy contract")
    require(policy.get("raw_action_dimension") == 12 and
            "rear_task_assists_and_rear_task_wheel_projection_OFF" in
                str(policy.get("action_transform", "")),
            "checkpoint changes the rear-OFF action contract")
    branch = metadata.get("rear_policy_timing_branch")
    counts = metadata.get("rear_policy_timing_branch_counts")
    migration = metadata.get("rear_policy_timing_migration")
    factor = (migration or {}).get("rear_policy_timing_factor")
    require(isinstance(branch, dict) and branch.get("schema") == MIGRATION_SCHEMA and
            branch.get("migration_added_updates") == 0 and
            isinstance(migration, dict) and migration.get("schema") == MIGRATION_SCHEMA and
            isinstance(factor, dict) and factor.get("schema") == MIGRATION_SCHEMA,
            "checkpoint lacks the rear-policy migration/branch boundary")
    origin = branch.get("counter_origin")
    require(isinstance(origin, dict) and factor.get("counter_origin") == origin,
            "rear-policy branch origin differs from its migration")
    computed = {}
    for key in COUNTERS:
        require(type(origin.get(key)) is int and counters[key] >= origin[key],
                f"invalid rear-policy {key} origin")
        computed[key] = counters[key] - origin[key]
    require(counts == computed, "rear-policy branch counts do not match lifetime counters")
    require(factor.get("rear_task_assists_enabled") is False and
            factor.get("FL_capture_assist_enabled") is True and
            factor.get("old_rollout_inherited") is False and
            factor.get("physical_dynamics_changed") is False and
            all(factor.get("added_" + key) == 0 for key in
                ("policy_decisions", "ppo_updates", "optimizer_steps", "auxiliary_updates")),
            "rear-policy migration control/credit declaration differs")
    require(args.expected_new_auxiliary_updates == 0 and
            metadata.get("rear_policy_timing_auxiliary") is None,
            "this exporter accepts no new rear-policy AUX credit")
    plan = Path(migration.get("plan_path", "")).resolve(strict=True)
    require(sha256(plan) == migration.get("plan_sha256"),
            "rear-policy migration plan changed after publication")
    plan_payload = read_json(plan)
    recapture_receipt = metadata.get("rear_recapture_migration")
    initial_target_head = (recapture_receipt.get("source_git_commit")
        if isinstance(recapture_receipt, dict) else expected_head)
    require(plan_payload == {key: value for key, value in migration.items()
            if key not in ("plan_path", "plan_sha256")} and
            migration.get("target_git_commit") == initial_target_head and
            factor.get("target_policy_contract") == base_policy,
            "rear-policy migration receipt differs from its sealed plan or target")
    source_checkpoint = Path(migration.get("source_checkpoint", "")).resolve(strict=True)
    source_sidecar = source_checkpoint.with_name(
        source_checkpoint.stem + "_manifest.json").resolve(strict=True)
    require(sha256(source_checkpoint) == migration.get("source_checkpoint_sha256") ==
                branch.get("source_checkpoint_sha256") and
            sha256(source_sidecar) == migration.get("source_manifest_sha256"),
            "rear-policy migration source checkpoint binding changed")
    preserved = factor.get("preserved_metadata_sha256")
    require(isinstance(preserved, dict), "rear-policy migration lacks preserved lineage hashes")
    auxiliary = inherited_auxiliary(metadata, preserved)
    live_receipt = metadata.get("rear_live_swing_migration")
    recapture_runtime = (reconstruct_live_swing_source_runtime(base_runtime, live_receipt)
                         if live_receipt is not None else base_runtime)
    recapture = rear_recapture_identity(
        metadata, recapture_runtime, migration, base_policy, counters, checkpoint)
    live_swing = rear_live_swing_identity(
        metadata, base_runtime, recapture_runtime, base_policy, counters,
        checkpoint, recapture)
    p02_progress = p02_progress_identity(
        metadata, progress_runtime or runtime, base_runtime, base_policy,
        progress_policy or policy, counters,
        checkpoint, live_swing)
    cooperative_prep = cooperative_prep_identity(
        metadata, runtime, progress_runtime or runtime,
        progress_policy or policy, policy, counters, checkpoint, p02_progress)
    nominal_timing = LIVE_SWING_MODE if live_swing is not None else NOMINAL_TIMING
    actor_hash = (proof.get("parameter_hashes") or {}).get("actor_parameter_sha256")
    require(isinstance(actor_hash, str) and metadata.get("actor_parameter_sha256") == actor_hash,
            "loaded actor differs from the checkpoint sidecar")
    return {
        "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha,
        "manifest": str(sidecar), "manifest_sha256": sidecar_sha,
        "actor_parameter_sha256": actor_hash, "policy_version": policy["version"],
        "observation_layout": policy["observation_layout"],
        "observation_dimension": policy["observation_dimension"],
        "lifetime_counters": counters, "branch_counter_origin": origin,
        "rear_policy_branch_counts": computed,
        "rear_recapture_revision": recapture,
        "rear_live_swing_revision": live_swing,
        "p02_progress_revision": p02_progress,
        "cooperative_preparation_revision": cooperative_prep,
        "migration_added_learning": {"policy_decisions": 0, "ppo_updates": 0,
                                     "optimizer_steps": 0, "auxiliary_updates": 0},
        "inherited_auxiliary": auxiliary, "this_run_auxiliary_updates": 0,
        "FL_capture_assist_enabled": True, "rear_task_assists_enabled": False,
        "nominal_timing": nominal_timing,
        "runtime_head": checked_head(args.expected_head),
        "checkpoint_runtime_head": expected_head,
        "evaluation_runtime_head": checked_head(args.expected_head),
        "reviewed_media_only_runtime_compatibility": media_compatibility,
        "migration_plan": str(plan), "migration_plan_sha256": migration["plan_sha256"],
        "migration_source_checkpoint": str(source_checkpoint),
        "migration_source_checkpoint_sha256": migration["source_checkpoint_sha256"],
        "migration_source_manifest_sha256": migration["source_manifest_sha256"],
    }


def sealed_source(source: Path, args: argparse.Namespace) -> dict[str, Any]:
    source = Path(source).resolve(strict=True)
    run_path = source.parent / "run_manifest.json"
    manifest_path = source / "semantic_video_source_manifest.json"
    require(run_path.is_file() and manifest_path.is_file(), "sealed source manifests are missing")
    require(sha256(manifest_path) == checked_sha(args.source_manifest_sha256,
            "--source-manifest-sha256") and
            sha256(run_path) == checked_sha(args.source_run_manifest_sha256,
            "--source-run-manifest-sha256"),
            "sealed source/run manifest differs from explicit invocation")
    run_manifest, manifest = read_json(run_path), read_json(manifest_path)
    require(bool(run_manifest.get("completed_at_utc")) and
            run_manifest.get("lifecycle") != "RUNNING" and
            run_manifest.get("runtime_contract") == manifest.get("runtime_contract"),
            "source is active or run/source runtime contracts differ")
    endpoint = manifest.get("episode_physics_ticks")
    require(manifest.get("schema") == "wlr50_clean.semantic_video_source.v1" and
            manifest.get("experiment_id") == EXPERIMENT and
            manifest.get("role") == "C" and manifest.get("from_phase") == "P01" and
            manifest.get("fresh_process_single_episode") is True and
            manifest.get("episode_count") == 1 and manifest.get("optimizer_updates") == 0 and
            type(endpoint) is int and 0 < endpoint <= 24000,
            "source is not one complete natural-P01 rear-policy evaluation")
    runtime = manifest.get("runtime_contract") or {}
    require(runtime.get("source_git_commit") == checked_head(args.expected_head) and
            runtime.get("experiment_id") == EXPERIMENT,
            "source runtime differs from explicit frozen runtime")
    require(manifest.get("control_method") == CONTROL_METHOD and
            manifest.get("capture_assist_enabled_in_training_and_evaluation") is True and
            manifest.get("capture_assist_is_policy_learning") is False and
            manifest.get("front_fl_capture_assist") == {
                "enabled": True, "mode": FL_ASSIST_MODE, "is_policy_learning": False},
            "source lacks the exact declared FL capture-assist provenance")
    checkpoint = checkpoint_identity(manifest, args)
    # The 422 append retains the original public rear-assist manifest label;
    # its inherited live-swing timing is proven by the checkpoint lineage above.
    source_nominal_timing = (NOMINAL_TIMING
        if checkpoint.get("p02_progress_revision") is not None else
        checkpoint["nominal_timing"])
    require(manifest.get("rear_task_assist") == {
        "enabled": False, "rr_capture_assist_mode": None,
        "rr_capture_wheel_mode": "off", "nominal_geometry_advisory": None,
        "nominal_timing": source_nominal_timing,
        "policy_controls_rear_task_actions": True,
        "rear_action_request": "residual_policy"},
        "source does not prove rear task assist was disabled")
    error = manifest.get("source_acceptance_error")
    diagnostic_partial = bool(args.diagnostic_capture_abort)
    if diagnostic_partial:
        require(run_manifest.get("lifecycle") == "DIAGNOSTIC_FAILURE" and
                manifest.get("diagnostic_only") is True and
                manifest.get("success_candidate") is False and
                manifest.get("physical_task_success") is not True and
                error == CAPTURE_ABORT_SOURCE_ERROR,
                "--diagnostic-capture-abort accepts only the exact closed callback abort")
    else:
        require(error is None or (isinstance(error, str) and
                "episode did not meet common physical task" in error and
                "VIDEO_OR_ARTIFACT_ERROR" not in error and "VideoArtifactError" not in error),
                "artifact/infrastructure failure needs explicit --diagnostic-capture-abort")
    require(run_manifest.get("checkpoint_runtime_compatibility") ==
            checkpoint["reviewed_media_only_runtime_compatibility"],
            "run/source media-only compatibility receipts differ")
    video = artifact(source, manifest, "actual_viewport_video.mp4")
    ledger_path = artifact(source, manifest, "viewport_frame_ledger.jsonl")
    tick_path = artifact(source, manifest, "capture_assist_ticks.jsonl")
    native_tick_path = artifact(source, manifest, "native_tick_audit.jsonl")
    decisions_path = artifact(source, manifest, "video_policy_decisions.jsonl")
    capture_path = artifact(source, manifest, "viewport_buffer_video_manifest.json")
    capture = read_json(capture_path)
    if diagnostic_partial:
        require(capture.get("valid") is False and
                capture.get("status") == "VIDEO_OR_ARTIFACT_ERROR" and
                capture.get("error") == CAPTURE_ABORT_ENCODER_ERROR and
                capture.get("encoder_finalized_before_app_close") is True and
                capture.get("frame_ledger_complete") is True and
                capture.get("one_callback_per_render") is True and
                capture.get("active_render_product_identity_proven") is True and
                (capture.get("full_decode") or {}).get("valid") is True,
                "callback-abort source did not retain a finalized decodable prefix")
    else:
        require(capture.get("valid") is True and
                capture.get("encoder_finalized_before_app_close") is True,
                "source writer did not close with a valid full decode")
    return {"source": source, "manifest": manifest, "manifest_path": manifest_path,
        "run_manifest": run_manifest, "run_manifest_path": run_path, "endpoint": endpoint,
        "video": video, "ledger_path": ledger_path, "tick_path": tick_path,
        "native_tick_path": native_tick_path, "decisions_path": decisions_path,
        "capture": capture, "capture_path": capture_path, "checkpoint": checkpoint,
        "diagnostic_partial": diagnostic_partial}


def checked_capture_abort_media(context: dict[str, Any], ffmpeg: Path
                                ) -> tuple[list[Any], list[Any], dict[str, Any]]:
    """Validate one continuous encoded prefix with exactly one missing callback frame."""
    helper = shared()
    decoded = helper.decode_frame_timeline(context["video"], ffmpeg=ffmpeg)
    ledger = helper.load_viewport_frame_ledger(context["ledger_path"])
    actual = context["capture"].get("frame_count")
    endpoint = context["endpoint"]
    declared = (endpoint + STRIDE - 1) // STRIDE
    require(type(actual) is int and actual > 0 and actual == len(decoded) == len(ledger) and
            declared == actual + 1 and endpoint == actual * STRIDE + STRIDE,
            "callback abort is not exactly one absent final render interval")
    require([item.frame_index for item in ledger] == list(range(actual)) and
            [item.sim_step for item in ledger] == [STRIDE * (index + 1)
                                                    for index in range(actual)] and
            all(abs(frame.pts_s - index / FPS) < 1e-5
                for index, frame in enumerate(decoded)),
            "callback-abort prefix lost its continuous real frame/PTS lineage")
    interval = context["manifest"].get("task_interval_window") or {}
    require(interval.get("frame_count") == declared and
            interval.get("endpoint_episode_tick") == endpoint and
            interval.get("extra_physics_ticks") == 0,
            "declared task interval differs from the one missing callback boundary")
    validation = helper.validate_mp4(context["video"], ffmpeg=ffmpeg,
        expected_fps=FPS, expected_frame_count=actual, expected_width=1280,
        expected_height=720, maximum_duration_s=200.0,
        require_sane_container_duration=False)
    require(validation.get("valid") is True and
            validation.get("sha256") == context["capture"].get("video_sha256"),
            "callback-abort encoded prefix does not fully decode or changed after closure")
    validation.update({
        "diagnostic_partial": True,
        "declared_episode_frame_count": declared,
        "actual_encoded_frame_count": actual,
        "last_encoded_physics_tick": ledger[-1].sim_step,
        "physical_episode_endpoint_tick": endpoint,
        "missing_physical_interval_s": [ledger[-1].sim_step / HZ, endpoint / HZ],
        "missing_final_frame_fabricated": False,
    })
    return decoded, ledger, validation


def com_to_fr_window_summary(row: dict[str, Any]) -> dict[str, Any]:
    """Logged RL-role receiver FR projection, not accumulated transfer or load."""
    result = {"valid": False, "reference_tick": None, "endpoint_tick": row.get("episode_physics_tick"),
        "window_s": None, "com_toward_fr_m": None, "fixed_direction_world": None,
        "semantics": "local_transfer_roles_window_start_CoM_to_FR_wheel_horizontal_direction",
        "total_transfer_or_FR_bearing_claimed": False}
    role = (row.get("transfer_roles") or {}).get("RL") or {}
    window = role.get("transfer_direction_context") or {}
    tick, first = result["endpoint_tick"], window.get("reference_tick")
    duration = _number(window.get("window_s"))
    value = _number(window.get("com_toward_receiver_m"))
    direction = window.get("fixed_direction_world")
    displacement = window.get("com_world_displacement_m")
    if (role.get("valid") is not True or role.get("diagonal_receiving_side") != "FR"
            or (row.get("center_of_mass") or {}).get("valid") is not True
            or type(tick) is not int or type(first) is not int or not 0 <= first <= tick
            or duration is None or not 0. <= duration <= .5+1e-8 or value is None
            or not math.isclose(duration, (tick-first)/HZ, rel_tol=0., abs_tol=1e-8)
            or not isinstance(direction, (tuple,list)) or len(direction) != 3
            or not isinstance(displacement, (tuple,list)) or len(displacement) != 3
            or any(_number(x) is None for x in (*direction,*displacement))):
        return result
    if (not math.isclose(sum(x*x for x in direction), 1., rel_tol=0., abs_tol=1e-7)
            or abs(direction[2]) > 1e-9
            or not math.isclose(value, sum(x*y for x,y in zip(direction,displacement)),
                                rel_tol=0., abs_tol=1e-8)):
        return result
    result.update(valid=True, reference_tick=first, window_s=duration,
                  com_toward_fr_m=value, fixed_direction_world=list(direction))
    return result


def overlay_decision_receipt(decision: dict[str, Any]) -> dict[str, Any]:
    """Compact one committed decision without conflating its post-step hint."""
    step = decision.get("step_info") or {}
    tick = decision.get("end_tick")
    require(type(tick) is int and step.get("physics_tick") == tick and
            decision.get("environment_step_returned") is True,
            "video decision lacks one completed endpoint")
    nominal = _vector(step.get("nominal_action_full12"), 12,
                      "committed decision nominal Full12")
    final = _vector(step.get("actual_drive_target_full12"), 12,
                    "committed decision final Full12")
    audit = step.get("actuator_target_effect_audit") or {}
    require(audit.get("schema") == "wlr50_clean.actuator_target_effect_audit.v1" and
            audit.get("verified") is True and
            audit.get("actual_mapping_matches_dispatch") is True and
            audit.get("setter_dispatch_targets_equal") is True and
            type(audit.get("physics_tick")) is int,
            "decision lacks a verified committed actuator receipt")
    mapped = _vector(audit.get("native_drive_target_full12"), 12,
                     "receipt native baseline Full12")
    baseline = _vector((audit.get("policy_headroom_evidence") or {}).get(
        "baseline_native_plus_controller_full12"), 12,
        "receipt baseline native plus controller Full12")
    counterfactual_physical = _vector((audit.get("counterfactual_native_targets") or {}).get(
        "wheel_velocity_rad_s"), 4, "receipt counterfactual physical wheel target")
    actual_physical = _vector((audit.get("actual_native_targets") or {}).get(
        "wheel_velocity_rad_s"), 4, "receipt actual physical wheel target")
    require(all(math.isclose(nominal[8+i], mapped[8+i], rel_tol=0., abs_tol=1e-9) and
                math.isclose(mapped[8+i], baseline[8+i], rel_tol=0., abs_tol=1e-9) and
                math.isclose(counterfactual_physical[i],
                    WHEEL_FORWARD_SIGN[name] * mapped[8+i], rel_tol=0., abs_tol=2e-6) and
                math.isclose(actual_physical[i],
                    WHEEL_FORWARD_SIGN[name] * final[8+i], rel_tol=0., abs_tol=2e-6)
                for i, name in enumerate(WHEEL_ORDER)),
            "decision wheel baseline/final differs from the committed actuator receipt")
    task = (step.get("semantic_task") or {})
    partial = ((task.get("nominal_provider_diagnostics") or {}).get(
        "source_partial_order") or {})
    phase = step.get("end_phase_id")
    layers = [value for value in partial.get("layers", [])
              if isinstance(value, dict) and value.get("stage") == phase and
              value.get("observation_tick") == tick]
    require(len(layers) <= 1, "source partial-order has duplicate current-phase rows")
    layer = layers[0] if layers else {}
    source_clock = {
        "available": bool(layers), "stage": phase,
        "status": layer.get("status"), "source_ticks": layer.get("source_ticks"),
        "late_group_start_tick": layer.get("late_group_start_tick"),
        "observation_tick": layer.get("observation_tick"),
        "wait_reason": layer.get("wait_reason"),
        "semantics": "source_partial_order_current_phase_not_public_dependency_flag",
    }
    return {
        "episode_end_tick": tick,
        "decision": decision.get("decision"),
        "actuator_receipt_raw_tick": audit["physics_tick"],
        "mapped_N_baseline_canonical_full12": mapped,
        "mapped_N_baseline_physical_wheels": counterfactual_physical,
        "executed_final_canonical_full12": final,
        "executed_final_physical_wheels": actual_physical,
        "N_baseline_is_counterfactual_not_executed_alone": True,
        "source_partial_order": source_clock,
    }


def decision_receipts(context: dict[str, Any]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    last_step = None
    expected_decision = 1
    with context["decisions_path"].open("rb") as stream:
        for raw in stream:
            require(raw.endswith(b"\n"), "policy decision ledger has a partial row")
            decision = json.loads(raw)
            require(decision.get("decision") == expected_decision,
                    "policy decision ledger has a numbering gap")
            receipt = overlay_decision_receipt(decision)
            tick = receipt["episode_end_tick"]
            require(tick not in result, "policy decision ledger repeats an endpoint")
            result[tick] = receipt
            last_step = decision.get("step_info")
            expected_decision += 1
    require(result, "policy decision ledger is empty")
    context["_last_step_info"] = last_step
    return result


def rl_contact_summary(leg: dict[str, Any]) -> dict[str, Any]:
    mode = leg.get("contact_mode")
    surface = leg.get("contact_surface")
    bearing = leg.get("bearing_verified")
    fallback, force = _contact(leg)
    return {
        "contact_mode": mode if isinstance(mode, str) and mode else None,
        "contact_surface": surface if isinstance(surface, str) and surface else None,
        "bearing_verified_raw": bearing if type(bearing) is bool else None,
        "display_fallback_when_contact_mode_absent": fallback,
        "contact_force_n": force,
        "front_schema_fallback_is_explicit_not_invented": mode is None,
        "ground_and_obstacle_is_body_collision_or_pure_wheel_climb_claim": False,
    }


def frame_summary(row: dict[str, Any], native_row: dict[str, Any],
                  decision_receipt: dict[str, Any] | None = None) -> dict[str, Any]:
    tick = row.get("episode_physics_tick")
    require(row.get("rear_task_assist_disabled") is True,
            "tick does not prove rear task assist is disabled")
    rr_state = row.get("rr_capture_assist")
    require(isinstance(rr_state, dict) and rr_state.get("mode_name") == "WAIT" and
            rr_state.get("active") is False and rr_state.get("owner_indices") == [],
            "disabled rear-assist snapshot is not inactive WAIT with no owners")
    timing = row.get("rear_policy_timing")
    require(isinstance(timing, dict) and set(timing) ==
            set(REAR_TIMING_FLAGS + REAR_TIMING_CLOCKS) and
            all(type(timing[key]) is bool for key in REAR_TIMING_FLAGS) and
            all(_number(timing[key]) is not None and 0. <= float(timing[key]) <= 200.
                for key in REAR_TIMING_CLOCKS),
            "tick lacks the exact visible rear timing state")
    fl_state = row.get("capture_assist")
    require(isinstance(fl_state, dict) and
            fl_state.get("schema") == "wlr50_clean.capture_assist_state.v1",
            "tick lacks observable FL capture-assist state")
    dispatch = row.get("dispatch") or {}
    require(dispatch.get("rr_capture_assist_evidence") is None,
            "rear task assist unexpectedly wrote this dispatch")
    source_suggestion = _vector(row.get("nominal_full12"), 12,
                                "post-step source suggestion Full12")
    final = _vector(dispatch.get("drive_target_full12"), 12, "final target Full12")
    actual = _vector(row.get("actual_full12"), 12, "actual Full12")
    policy = _vector(dispatch.get("independent_policy_residual_requested_full12"),
                     12, "raw policy request Full12")
    require(native_row.get("episode_physics_tick") == tick,
            "native audit/tick identity mismatch")
    audit = native_row.get("native_audit") or {}
    require(audit.get("verified") is True and
            audit.get("canonical_order") == list(SERVO_ORDER + WHEEL_ORDER),
            "tick lacks verified canonical/native actuator audit")
    native_targets = audit.get("actual_native_targets") or {}
    physical_wheels = _vector(native_targets.get("wheel_velocity_rad_s"), 4,
                              "native wheel targets")
    expected_physical = [WHEEL_FORWARD_SIGN[name] * final[8 + index]
                         for index, name in enumerate(WHEEL_ORDER)]
    require(all(math.isclose(one, two, rel_tol=0., abs_tol=2e-6)
                for one, two in zip(physical_wheels, expected_physical)),
            "native wheel targets violate the locked axis mapping")
    mapped_nominal = _vector(native_row.get("nominal_full12"), 12,
                             "native committed nominal Full12")
    audit_baseline = _vector(audit.get("native_drive_target_full12"), 12,
                             "native audit baseline Full12")
    require(all(math.isclose(mapped_nominal[8+i], audit_baseline[8+i],
                             rel_tol=0., abs_tol=1e-9)
                for i in range(4)),
            "native committed wheel baseline differs from its actuator receipt")
    if decision_receipt is not None:
        require(decision_receipt.get("episode_end_tick") == tick and
                decision_receipt.get("actuator_receipt_raw_tick") == audit.get("physics_tick") and
                all(math.isclose(decision_receipt["mapped_N_baseline_canonical_full12"][8+i],
                                 mapped_nominal[8+i], rel_tol=0., abs_tol=1e-9) and
                    math.isclose(decision_receipt["executed_final_canonical_full12"][8+i],
                                 final[8+i], rel_tol=0., abs_tol=1e-9)
                    for i in range(4)),
                "frame/native/decision receipt clocks or wheel targets differ")
    measured = []
    for index, name in enumerate(WHEEL_ORDER):
        value = _number(((row.get("wheels") or {}).get(name) or {}).get("velocity_rad_s"))
        require(value is not None and math.isclose(value, actual[8 + index],
                rel_tol=0., abs_tol=1e-9), f"{name} qdot differs from actual Full12")
        measured.append(float(value))
    legs = {name: _leg(row, name) for name in ("FL", "RR", "RL")}
    contacts = {name: _contact(leg) for name, leg in legs.items()}
    rl_contact = rl_contact_summary(legs["RL"])
    initialized = bool(fl_state.get("initialized"))
    return {
        "tick": tick, "time_s": _number(row.get("sim_time_s")),
        "phase": row.get("phase"), "rear_policy_timing": timing,
        "fl_assist_mode": fl_state.get("mode_name"),
        "fl_assist_active": bool(fl_state.get("active")),
        "fl_assist_target_deg": (_number(fl_state.get("hip_target_deg"))
                                  if initialized else None),
        "fl_gap_mm": _gap_mm(legs["FL"]), "fl_contact": contacts["FL"][0],
        "fl_force_n": contacts["FL"][1],
        "fl_placed": (row.get("placed_history") or {}).get("FL"),
        "rr_gap_mm": _gap_mm(legs["RR"]), "rr_front_mm": _front_mm(legs["RR"]),
        "rr_contact": contacts["RR"][0], "rr_force_n": contacts["RR"][1],
        "rr_placed": (row.get("placed_history") or {}).get("RR"),
        "rl_gap_mm": _gap_mm(legs["RL"]), "rl_front_mm": _front_mm(legs["RL"]),
        "rl_contact": contacts["RL"][0], "rl_force_n": contacts["RL"][1],
        "rl_contact_mode": rl_contact["contact_mode"],
        "rl_contact_surface": rl_contact["contact_surface"],
        "rl_bearing_verified_raw": rl_contact["bearing_verified_raw"],
        "rl_contact_schema_fallback": rl_contact["front_schema_fallback_is_explicit_not_invented"],
        "rl_placed": (row.get("placed_history") or {}).get("RL"),
        "rr_hip_final_deg": final[6], "rr_hip_actual_deg": _joint(row, SERVO_ORDER[6]),
        "rr_knee_final_deg": final[7], "rr_knee_actual_deg": _joint(row, SERVO_ORDER[7]),
        "rl_hip_final_deg": final[4], "rl_hip_actual_deg": _joint(row, SERVO_ORDER[4]),
        "rl_knee_final_deg": final[5], "rl_knee_actual_deg": _joint(row, SERVO_ORDER[5]),
        "rear_policy_request": policy[4:8],
        "wheel_nominal": mapped_nominal[8:],
        "wheel_mapped_N_baseline": mapped_nominal[8:],
        "wheel_source_suggestion_poststep": source_suggestion[8:],
        "N_baseline_is_counterfactual_not_executed_alone": True,
        "actuator_receipt_raw_tick": audit.get("physics_tick"),
        "source_partial_order": ((decision_receipt or {}).get("source_partial_order") or {
            "available": False, "stage": row.get("phase"), "status": None,
            "source_ticks": None, "late_group_start_tick": None,
            "observation_tick": None,
            "semantics": "no_exact_decision_endpoint_for_this_selected_tick"}),
        "wheel_final": final[8:], "wheel_native_physical": physical_wheels,
        "wheel_actual": measured,
        "com_to_fr_local_window": com_to_fr_window_summary(row),
    }


def unavailable_mounts(reason: str) -> dict[str, Any]:
    return {"valid": False, "source_verified": False, "reason": reason,
        "physics_tick": None, "left_minus_right_mean_z_m": None,
        "rear_minus_front_mean_z_m": None, "FR_world_z_m": None,
        "diagnostic_only_not_bearing_or_readiness": True}


def mount_frame_summary(row: dict[str, Any], expected_tick: int) -> dict[str, Any]:
    """Use same-tick logged geometry only; never fill from neighboring frames."""
    measured = (row.get("same_rigid_body_hip_mount_geometry") or {}).get("value")
    if not isinstance(measured, dict) or measured.get("valid") is not True or measured.get("source_verified") is not True:
        reason = measured.get("reason") if isinstance(measured, dict) else None
        return unavailable_mounts(reason or "actual four-mount measurement unavailable")
    resolution = measured.get("source_resolution") or {}
    if (measured.get("schema") != "readonly.same_rigid_body_hip_mount_world_geometry.v1"
            or resolution.get("source_verified") is not True
            or resolution.get("parent_body_name") != "base_link"
            or row.get("physics_tick") != expected_tick
            or row.get("frame_raw_physics_tick") != expected_tick
            or measured.get("physics_tick") != expected_tick
            or (measured.get("parent_pose") or {}).get("observation_tick") != expected_tick
            or _number(measured.get("simulation_time_s")) is None
            or not math.isclose(measured["simulation_time_s"], expected_tick/HZ, rel_tol=0., abs_tol=1e-8)):
        return unavailable_mounts("four-mount source or exact frame clock mismatch")
    fields = ("left_minus_right_mean_z_m", "rear_minus_front_mean_z_m", "FR_world_z_m")
    values = {name: _number(measured.get(name)) for name in fields}
    if any(value is None for value in values.values()):
        return unavailable_mounts("nonfinite or missing four-mount measurement")
    return {"valid": True, "source_verified": True, "reason": None,
        "physics_tick": expected_tick, **values,
        "diagnostic_only_not_bearing_or_readiness": True,
        "sign_semantics": "positive_left_higher;positive_rear_higher;world_FR_z"}


def mount_evidence(context: dict[str, Any]) -> dict[int, dict[str, Any]]:
    name = "height_diagnostics.jsonl"
    if name not in (context["manifest"].get("artifacts") or {}):
        return {}
    # An optional unavailable sensor is N/A; a changed sealed evidence file
    # still fails the existing artifact identity check, like other evidence.
    path = artifact(context["source"], context["manifest"], name)
    result: dict[int, dict[str, Any]] = {}
    with path.open("rb") as stream:
        for raw in stream:
            try:
                value = json.loads(raw)
                tick = value.get("physics_tick")
                if type(tick) is not int:
                    continue
                result[tick] = (unavailable_mounts("duplicate diagnostic frame") if tick in result
                                else mount_frame_summary(value, tick))
            except (ValueError, TypeError, AttributeError):
                continue  # No interpolation or inferred zero for a malformed row.
    return result


def capture_rows(context: dict[str, Any], ledger: list[Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    wanted = {item.sim_step for item in ledger}
    mounts = mount_evidence(context)
    receipts = decision_receipts(context)
    frames: dict[int, dict[str, Any]] = {}
    selected: list[dict[str, Any]] = []
    prior_signature = None
    count = 0
    with context["tick_path"].open("rb") as stream, \
            context["native_tick_path"].open("rb") as native_stream:
        for count, raw in enumerate(stream, 1):
            native_raw = native_stream.readline()
            require(raw.endswith(b"\n") and native_raw.endswith(b"\n"),
                    "per-tick evidence has a partial or missing final row")
            row, native = json.loads(raw), json.loads(native_raw)
            require(row.get("episode_physics_tick") == count and
                    math.isclose(float(row.get("sim_time_s")), count / HZ, abs_tol=1e-10),
                    "capture-assist tick ledger clock gap")
            receipt = receipts.get(count)
            require(count not in wanted or receipt is not None,
                    "encoded frame endpoint lacks its completed decision receipt")
            item = frame_summary(row, native, receipt)
            item["hip_mount_geometry"] = mounts.get(count, unavailable_mounts(
                "no exact-tick verified four-mount diagnostic in this source"))
            signature = (item["phase"], tuple(item["rear_policy_timing"].items()),
                         item["fl_assist_mode"], item["fl_placed"],
                         item["rr_contact"], item["rr_placed"],
                         item["rl_contact"], item["rl_contact_mode"],
                         item["rl_contact_surface"], item["rl_placed"])
            if signature != prior_signature:
                selected.append(item)
            if count in wanted:
                frames[count] = item
            prior_signature = signature
        require(native_stream.readline() == b"", "native audit has extra rows")
    require(count == context["endpoint"], "per-tick evidence does not cover the episode")
    require(set(frames) == wanted, "per-tick evidence lacks encoded frame endpoints")
    require(wanted.issubset(receipts),
            "encoded frame endpoints are not a subset of completed decisions")
    rows = [frames[item.sim_step] for item in ledger]
    if context["diagnostic_partial"]:
        last_encoded_tick = ledger[-1].sim_step
        selected = [item for item in selected if item["tick"] <= last_encoded_tick]
    selected.append(rows[-1])
    unique = {int(item["tick"]): item for item in selected}
    return rows, [unique[key] for key in sorted(unique)]


def terminal_evidence(context: dict[str, Any]) -> dict[str, Any]:
    last_step = context.get("_last_step_info")
    if not isinstance(last_step, dict):
        with context["decisions_path"].open("rb") as stream:
            for raw in stream:
                require(raw.endswith(b"\n"), "policy decision ledger has a partial row")
                row = json.loads(raw)
                if isinstance(row.get("step_info"), dict):
                    last_step = row["step_info"]
    task = ((last_step or {}).get("semantic_task") or {})
    physical = context["manifest"].get("physical_episode") or {}
    evaluation = physical.get("physical_task_evaluation") or {}
    reason = (task.get("termination_reason") or evaluation.get("termination_reason") or
              physical.get("termination_reason"))
    source_name = task.get("termination_source") or evaluation.get("termination_source")
    if context["diagnostic_partial"]:
        last_encoded_tick = int(context["capture"]["frame_count"]) * STRIDE
        require((last_step or {}).get("physics_tick") == last_encoded_tick and
                reason is None and source_name is None,
                "callback-abort prefix must end at a non-terminal complete policy decision")
        return {"result": "CAPTURE_ABORT", "success": False,
            "task_terminal_observed": False, "termination_reason": None,
            "termination_source": None,
            "source_acceptance_error": context["manifest"].get("source_acceptance_error")}
    success = (context["manifest"].get("physical_task_success") is True and
               physical.get("task_success") is True and
               context["manifest"].get("success_candidate") is True and
               context["manifest"].get("diagnostic_only") is False and
               context["manifest"].get("source_acceptance_error") is None)
    return {"result": "SUCCESS" if success else "INCOMPLETE", "success": success,
        "task_terminal_observed": True,
        "termination_reason": reason, "termination_source": source_name,
        "source_acceptance_error": context["manifest"].get("source_acceptance_error")}


def rr_reached(rows: list[dict[str, Any]]) -> bool:
    return any(item["phase"] in ("P09", "P10", "P11", "P12", "P13") or
               item["rear_policy_timing"]["rr_carry_capture"] or
               item["rear_policy_timing"]["rr_support_handoff"] for item in rows)


def rl_reached(rows: list[dict[str, Any]]) -> bool:
    return any(item["rear_policy_timing"]["rl_prep_transfer"] or
               item["rear_policy_timing"]["rl_swing_capture"] for item in rows)


def detail_plan(rows: list[dict[str, Any]], step: int, success: bool,
                diagnostic_partial: bool = False, *, cooperative_preparation: bool = False) -> dict[str, Any]:
    if diagnostic_partial:
        start = max(0, len(rows) - 60 * FPS)
        phase = str(rows[-1]["phase"])
        return {"kind": "VIDEO_CAPTURE_ABORT_AVAILABLE_TAIL", "start": start,
            "end": len(rows),
            "filename": f"CP{step}_DET_CAPTURE_ABORT_{phase}_available_tail.mp4",
            "title": (f"DETAIL | CP{step} | CAPTURE ABORT IN {phase} | "
                      "NOT TASK TERMINAL | REAL FRAMES ONLY"),
            "requested_RR_detail_unavailable_reason":
                "VIDEO_CAPTURE_ABORT_BEFORE_TASK_TERMINAL"}
    suffix = "" if success else "_INCOMPLETE"
    if cooperative_preparation and rr_reached(rows):
        start = next(index for index, item in enumerate(rows) if
            item["phase"] in ("P07", "P08", "P09", "P10", "P11", "P12", "P13") or
            item["rear_policy_timing"]["rr_carry_capture"] or
            item["rear_policy_timing"]["rr_support_handoff"])
        status = "RL PREP/SWING OBSERVED" if rl_reached(rows) else "RL NOT REACHED"
        return {"kind": "RR_TO_RL_PREPARATION_ACTUAL_WINDOW", "start": start,
            "end": len(rows),
            "filename": f"CP{step}_DET_RR_to_RL_preparation_REAR_OFF_FL_ON{suffix}.mp4",
            "title": f"DETAIL | CP{step} | RR to RL PREPARATION | REAR OFF FL ON | {'SUCCESS' if success else 'INCOMPLETE'} | {status}",
            "requested_RR_detail_unavailable_reason": None,
            "scope": "contiguous_actual_P07_preparation_through_this_episode_endpoint",
            "contact_success_inferred_from_preparation": False}
    if rr_reached(rows):
        start = next(index for index, item in enumerate(rows) if
            item["phase"] in ("P09", "P10", "P11", "P12", "P13") or
            item["rear_policy_timing"]["rr_carry_capture"] or
            item["rear_policy_timing"]["rr_support_handoff"])
        status = "actual RL window observed" if rl_reached(rows) else "RL NOT REACHED"
        kind = "RR_CAPTURE_TO_RL" if success else "RR_ATTEMPT_TO_ACTUAL_TAIL"
        filename = (f"CP{step}_DET_RR_capture_to_RL_detail.mp4" if success else
                    f"CP{step}_DET_RR_attempt_to_actual_tail_INCOMPLETE.mp4")
        title = (f"DETAIL | CP{step} | RR capture to RL | SUCCESS" if success else
                 f"DETAIL | CP{step} | RR ATTEMPT WINDOW | INCOMPLETE | {status}")
        return {"kind": kind, "start": start, "end": len(rows),
            "filename": filename, "title": title,
            "requested_RR_detail_unavailable_reason": None}
    start = max(0, len(rows) - 60 * FPS)
    phase = str(rows[-1]["phase"])
    return {"kind": "RR_NOT_REACHED_PREDECESSOR_FAILURE", "start": start,
        "end": len(rows),
        "filename": f"CP{step}_DET_RR_NOT_REACHED_{phase}_failure_detail{suffix}.mp4",
        "title": f"DETAIL | CP{step} | {phase} predecessor failure tail | RR/RL NOT REACHED",
        "requested_RR_detail_unavailable_reason": "THIS_EPISODE_DID_NOT_REACH_RR_WINDOW"}


def _fmt(value: Any, digits: int = 2) -> str:
    return "N/A" if value is None else f"{float(value):.{digits}f}"


def _wheel(prefix: str, values: list[float]) -> str:
    return prefix + " ".join(f"{name}={_fmt(value, 3)}" for name, value in
                             zip(("FL", "FR", "RL", "RR"), values))


def hip_mount_panel_line(row: dict[str, Any]) -> str:
    geometry = row.get("hip_mount_geometry") or unavailable_mounts("unavailable")
    def mm(key: str) -> str:
        value = _number(geometry.get(key)) if geometry.get("valid") is True else None
        return _fmt(None if value is None else 1000.*value, 2)
    return ("HIP mounts mm (live USD): dL-R="+mm("left_minus_right_mean_z_m")+
        " dRear-Front="+mm("rear_minus_front_mean_z_m")+" FRz="+mm("FR_world_z_m")+
        " | geometry, NOT CoM/bearing")


def com_to_fr_panel_line(row: dict[str, Any]) -> str:
    value = row.get("com_to_fr_local_window") or {}
    valid = value.get("valid") is True
    projection = _number(value.get("com_toward_fr_m")) if valid else None
    duration = _number(value.get("window_s")) if valid else None
    reference = value.get("reference_tick") if valid else None
    endpoint = value.get("endpoint_tick") if valid else None
    return (f"CoM->FR local {_fmt(duration,3)}s (max0.5s), fixed start-dir [{_fmt(reference,0)}->{_fmt(endpoint,0)}]: "
        f"{_fmt(None if projection is None else projection*1000.,3)}mm | NOT total transfer / FR bearing")


def panel_lines(row: dict[str, Any], outcome: dict[str, Any],
                identity: dict[str, Any]) -> list[str]:
    counts = identity["rear_policy_branch_counts"]
    legacy_aux = identity["inherited_auxiliary"]["legacy_limited_AUX"]
    mixed_aux = identity["inherited_auxiliary"]["mixed_front_and_RR_AUX"]
    recapture = identity.get("rear_recapture_revision")
    live_swing = identity.get("rear_live_swing_revision")
    p02_progress = identity.get("p02_progress_revision")
    cooperative = identity.get("cooperative_preparation_revision")
    timing = row["rear_policy_timing"]
    source_clock = row["source_partial_order"]
    source_status = (source_clock.get("status")
                     if source_clock.get("available") else "N/A")
    source_ticks = (source_clock.get("source_ticks")
                    if source_clock.get("available") else None)
    late_tick = (source_clock.get("late_group_start_tick")
                 if source_clock.get("available") else None)
    source_suggestion = row["wheel_source_suggestion_poststep"]
    baseline = row["wheel_mapped_N_baseline"]
    rl_mode = row["rl_contact_mode"] or "N/A(front-schema)"
    rl_surface = row["rl_contact_surface"] or row["rl_contact"]
    state = ("RR_CARRY" if timing["rr_carry_capture"] else
             "RR_HANDOFF" if timing["rr_support_handoff"] else
             "RL_PREP" if timing["rl_prep_transfer"] else
             "RL_SWING" if timing["rl_swing_capture"] else "FRONT/PREDECESSOR")
    return [
        f"DETERMINISTIC RESIDUAL PPO | {outcome['result']} | CP{identity['lifetime_counters']['global_policy_decisions']} | g{identity['runtime_head'][:12]}",
        "FRONT FL ASSIST: ON (p05 hip-only) | REAR TASK ASSIST: OFF",
        ("CAPTURE ABORT | NOT TASK RESULT | REAL ENCODED PREFIX | NO FRAME FILLED"
         if outcome["result"] == "CAPTURE_ABORT" else
         (f"COOPERATIVE PREP: {cooperative['cooperative_preparation_revision']} | sigma/reward boundary +0 learning"
          if cooperative is not None else
         (f"P02-PROGRESS 422: {p02_progress['p02_progress_credit_mode']} | inherited live-swing v3 | AUX +0"
          if p02_progress is not None else
          (f"LIVE-SWING MODE: {live_swing['target_mode']} | inherited v2 recapture | AUX +0"
          if live_swing is not None else
          (f"RECAPTURE MODE: {recapture['target_mode']} | BRANCH: {RECAPTURE_BRANCH} | AUX +0"
           if recapture is not None else
           f"NOMINAL TIMING: {identity['nominal_timing']} | TRAINING: PPO | this branch AUX +0"))))),
        f"inherited AUX (not PPO): legacy {legacy_aux['accepted']}/{legacy_aux['attempted']} | mixed front+RR {mixed_aux['accepted']}/{mixed_aux['attempted']}",
        f"rear-policy branch +{counts['global_policy_decisions']} decisions / +{counts['ppo_updates']} PPO / +{counts['optimizer_steps']} Adam",
        (f"t={_fmt(row['time_s'],3)}s ep={row['tick']} {row['phase']} | {state} | "
         f"srcPO={source_status} src={_fmt(source_ticks,0)} late={_fmt(late_tick,0)} | "
         f"publicDep P09={timing.get('p09_dependency_wait', 'N/A')} "
         f"P12={timing.get('p12_dependency_wait', 'N/A')}"),
        f"FL gap={_fmt(row['fl_gap_mm'],3)}mm {row['fl_contact']} placed_hist={row['fl_placed']} assist={row['fl_assist_mode']} active={row['fl_assist_active']}",
        f"RR gap/front={_fmt(row['rr_gap_mm'],3)}/{_fmt(row['rr_front_mm'],3)}mm {row['rr_contact']} placed_hist={row['rr_placed']} force={_fmt(row['rr_force_n'],2)}N",
        f"RR hip target/actual={_fmt(row['rr_hip_final_deg'])}/{_fmt(row['rr_hip_actual_deg'])}deg knee={_fmt(row['rr_knee_final_deg'])}/{_fmt(row['rr_knee_actual_deg'])}deg",
        (f"RL gap/front={_fmt(row['rl_gap_mm'],3)}/{_fmt(row['rl_front_mm'],3)}mm "
         f"mode={rl_mode} surface={rl_surface} "
         f"bearing={row['rl_bearing_verified_raw'] if row['rl_bearing_verified_raw'] is not None else 'N/A'} "
         f"| policy rear={','.join(_fmt(v,2) for v in row['rear_policy_request'])}"),
        (f"N receipt CF-not-solo [FL,FR,RL,RR] end ep{row['tick']}/raw{row['actuator_receipt_raw_tick']}: "
         f"{','.join(_fmt(v,2) for v in baseline)} | source hint poststep: "
         f"{','.join(_fmt(v,2) for v in source_suggestion)}"),
        _wheel("FINAL wheel target: ", row["wheel_final"]),
        _wheel("MEASURED wheel qd:  ", row["wheel_actual"]),
        hip_mount_panel_line(row),
        com_to_fr_panel_line(row),
    ]


def rgba_panels(rows: list[dict[str, Any]], outcome: dict[str, Any],
                identity: dict[str, Any]) -> Iterable[bytes]:
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    for row in rows:
        image = Image.new("RGBA", (1280, PANEL_HEIGHT), (0, 0, 0, 222))
        draw = ImageDraw.Draw(image)
        for index, line in enumerate(panel_lines(row, outcome, identity)):
            color = (255, 226, 80, 255) if index < 3 else (238, 244, 250, 255)
            draw.text((10, 3 + 15 * index), line, font=font, fill=color)
        yield image.tobytes()


def encode_full(context: dict[str, Any], rows: list[dict[str, Any]], output: Path,
                outcome: dict[str, Any], identity: dict[str, Any], ffmpeg: Path) -> dict[str, Any]:
    command = [str(ffmpeg), "-hide_banner", "-nostdin", "-v", "error", "-n",
        "-threads", "2", "-filter_threads", "1", "-filter_complex_threads", "1",
        "-i", str(context["video"]), "-f", "rawvideo", "-pixel_format", "rgba",
        "-video_size", f"1280x{PANEL_HEIGHT}", "-framerate", str(FPS), "-i", "pipe:0",
        "-filter_complex", "[0:v]setpts=PTS-STARTPTS[v];[1:v]format=rgba,setpts=PTS-STARTPTS[p];[v][p]overlay=0:0:shortest=1,format=yuv420p[out]",
        "-map", "[out]", "-an", "-frames:v", str(len(rows)), "-r", str(FPS),
        "-fps_mode", "cfr", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-threads", "2", "-movflags", "+faststart", str(output)]
    shared().run(command, input_bytes=rgba_panels(rows, outcome, identity))
    partial = outcome["result"] == "CAPTURE_ABORT"
    return {"output": str(output), "frame_count": len(rows), "normal_speed": True,
        "no_intro_frames": True, "full_failure_tail_preserved": not partial,
        "all_available_encoded_tail_preserved": True,
        "diagnostic_partial": partial,
        "top_overlay_no_canvas_growth": True,
        "validation": shared().validate_output(output, len(rows), 1280, 720, ffmpeg),
        "previews": shared().preview(output, len(rows), ffmpeg), "command": command}


def encode_detail(full: Path, rows: list[dict[str, Any]], plan: dict[str, Any],
                  destination: Path, ffmpeg: Path) -> dict[str, Any]:
    start, end = plan["start"], plan["end"]
    count, output = end - start, destination / plan["filename"]
    safe_title = "".join(ch if ch.isalnum() or ch in " _|/-" else "_"
                         for ch in plan["title"])
    video_filter = (f"trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS,"
        "drawbox=x=0:y=0:w=iw:h=20:color=black@0.92:t=fill,"
        "drawtext=fontfile='C\\:/Windows/Fonts/consola.ttf':"
        f"text='{safe_title}':fontcolor=yellow:fontsize=14:x=14:y=2")
    command = [str(ffmpeg), "-hide_banner", "-nostdin", "-v", "error", "-n",
        "-threads", "2", "-filter_threads", "1", "-i", str(full), "-vf", video_filter,
        "-an", "-frames:v", str(count), "-r", str(FPS), "-fps_mode", "cfr",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-threads", "2", "-movflags", "+faststart", str(output)]
    shared().run(command)
    return {"output": str(output), "kind": plan["kind"], "frame_count": count,
        "front_FL_assist": "ON", "rear_task_assist": "OFF",
        "preparation_is_contact_or_full_task_success": False,
        "requested_RR_detail_unavailable_reason":
            plan["requested_RR_detail_unavailable_reason"],
        "source_frame_interval_half_open": [start, end],
        "actual_tick_endpoints": [rows[start]["tick"], rows[end - 1]["tick"]],
        "same_single_episode": True, "contiguous": True,
        "internal_stalls_removed": False, "normal_speed": True,
        "validation": shared().validate_output(output, count, 1280, 720, ffmpeg),
        "previews": shared().preview(output, count, ffmpeg), "command": command}


def encode_pair(baseline: dict[str, Any], candidate: dict[str, Any], full: Path,
                output: Path, step: int, result: str, ffmpeg: Path) -> dict[str, Any]:
    require(baseline["manifest"]["camera"] == candidate["manifest"]["camera"],
            "historical N and candidate camera definitions differ")
    _, ledger, _ = shared().checked_media(baseline, ffmpeg=ffmpeg)
    b_count, c_count = len(ledger), candidate["frame_count"]
    count = max(b_count, c_count)
    require(count <= MAX_FRAMES, "comparison exceeds 200 seconds")
    candidate_label = (f"CP{step} DET PPO | CAPTURE ABORT | NOT TASK RESULT"
        if result == "CAPTURE_ABORT" else
        f"CP{step} DET PPO | FL ASSIST ON | REAR TASK ASSIST OFF | {result}")
    labels = (
        f"HISTORICAL N_REF (NOT FRESH B) | {DEFAULT_HISTORICAL_VERSION}",
        candidate_label)
    filters = []
    for index, (label, frames) in enumerate(zip(labels, (b_count, c_count))):
        filters.append(f"[{index}:v]setpts=PTS-STARTPTS,scale=960:540,pad=960:610:0:70:black,"
            f"tpad=stop_mode=clone:stop=-1,setpts=N/({FPS}*TB),"
            "drawtext=fontfile='C\\:/Windows/Fonts/arial.ttf':"
            f"text='{label}':fontcolor=white:fontsize=20:x=14:y=13,"
            "drawtext=fontfile='C\\:/Windows/Fonts/arial.ttf':"
            f"text='RUN ENDED - FROZEN, NOT NEW PHYSICS':fontcolor=yellow:fontsize=20:x=14:y=44:enable='gte(n,{frames})'[v{index}]")
    filters.append("[v0][v1]hstack=inputs=2:shortest=1,format=yuv420p[out]")
    command = [str(ffmpeg), "-hide_banner", "-nostdin", "-v", "error", "-n",
        "-threads", "2", "-filter_complex_threads", "1", "-i", str(baseline["video"]),
        "-i", str(full), "-filter_complex", ";".join(filters), "-map", "[out]", "-an",
        "-frames:v", str(count), "-r", str(FPS), "-fps_mode", "cfr", "-c:v", "libx264",
        "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-threads", "2",
        "-movflags", "+faststart", str(output)]
    shared().run(command)
    return {"output": str(output), "frame_count": count,
        "historical_N_version": DEFAULT_HISTORICAL_VERSION,
        "historical_N_source": str(baseline["source"]),
        "historical_N_is_fresh_B": False, "same_camera": True,
        "same_controller_or_runtime_claimed": False,
        "alignment": "same elapsed natural-P01 origin; no phase/event retiming",
        "freeze_added_frames": {"historical_N": count - b_count,
                                "candidate": count - c_count},
        "freeze_is_physical_evidence": False, "normal_speed": True,
        "validation": shared().validate_output(output, count, 1920, 610, ffmpeg),
        "previews": shared().preview(output, count, ffmpeg), "command": command}


def export(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.source).resolve(strict=True)
    destination = Path(args.destination).resolve()
    require(destination.is_relative_to(OUTPUT_ROOT.resolve()) and not destination.exists(),
            "destination must be a new directory in the isolated rear-policy output tree")
    candidate = sealed_source(source, args)
    baseline = shared().sealed_source(Path(args.historical_n_source), candidate=False)
    ffmpeg = shared().find_ffmpeg(candidate["capture"].get("full_decode", {}).get("ffmpeg_path"))
    if candidate["diagnostic_partial"]:
        _, ledger, source_validation = checked_capture_abort_media(candidate, ffmpeg)
    else:
        _, ledger, source_validation = shared().checked_media(candidate, ffmpeg=ffmpeg)
    rows, selected = capture_rows(candidate, ledger)
    require(0 < len(rows) <= MAX_FRAMES, "source exceeds one 200-second episode")
    outcome = terminal_evidence(candidate)
    identity = candidate["checkpoint"]
    step = identity["lifetime_counters"]["global_policy_decisions"]
    suffix = "" if outcome["success"] else "_INCOMPLETE"
    plan = detail_plan(rows, step, outcome["success"], candidate["diagnostic_partial"],
        cooperative_preparation=identity.get("cooperative_preparation_revision") is not None)
    full_name = (f"CP{step}_DET_policy_rear_no_assist_CAPTURE_ABORT_"
                 f"{rows[-1]['phase']}_INCOMPLETE.mp4"
                 if candidate["diagnostic_partial"] else
                 f"CP{step}_DET_full_policy_rear_no_assist{suffix}.mp4")
    destination.mkdir(parents=True)
    full = encode_full(candidate, rows,
        destination / full_name,
        outcome, identity, ffmpeg)
    print("FULL_PLAYABLE_VALIDATED " + full["output"], flush=True)
    detail = encode_detail(Path(full["output"]), rows, plan, destination, ffmpeg)
    pair = encode_pair(baseline, {**candidate, "frame_count": len(rows)},
        Path(full["output"]), destination / f"N_vs_CP{step}_DET_same_camera.mp4",
        step, outcome["result"], ffmpeg)
    evidence_path = destination / "selected_real_events.json"
    write_new_json(evidence_path, {
        "schema": "wlr50_clean.rear_policy_video_selected_events.v1",
        "source": str(source),
        "selection": ("actual encoded-prefix phase/rear-timing/contact transitions; "
            "missing callback endpoint excluded; no interpolation"
            if candidate["diagnostic_partial"] else
            "actual phase/rear-timing/contact transitions plus endpoint; no interpolation"),
        "rows": selected, "full_per_tick_evidence": str(candidate["tick_path"])})
    receipt = {
        "schema": "wlr50_clean.rear_policy_no_assist_video_export.v2",
        "source": str(source), "source_manifest": str(candidate["manifest_path"]),
        "source_manifest_sha256": sha256(candidate["manifest_path"]),
        "source_run_manifest_sha256": sha256(candidate["run_manifest_path"]),
        "source_video_validation": {key: source_validation.get(key) for key in (
            "sha256", "bytes", "valid", "full_decode", "frame_count", "fps",
            "resolution", "duration_s", "frame_pts_sha256",
            "decoded_frame_checksums_sha256", "black_like_frame_count",
            "timestamps_monotonic", "timestamps_continuous", "diagnostic_partial",
            "declared_episode_frame_count", "actual_encoded_frame_count",
            "last_encoded_physics_tick", "physical_episode_endpoint_tick",
            "missing_physical_interval_s", "missing_final_frame_fabricated")},
        "checkpoint_identity": identity, "experiment_id": EXPERIMENT,
        "rear_recapture_revision": identity.get("rear_recapture_revision"),
        "rear_live_swing_revision": identity.get("rear_live_swing_revision"),
        "p02_progress_revision": identity.get("p02_progress_revision"),
        "cooperative_preparation_revision":
            identity.get("cooperative_preparation_revision"),
        "control_method": CONTROL_METHOD,
        "visible_labels": {"deterministic_residual_PPO": True,
            "front_FL_assist": "ON", "front_FL_assist_mode": FL_ASSIST_MODE,
            "rear_task_assist": "OFF", "nominal_timing": identity["nominal_timing"],
            "rear_recapture_mode": (None if identity.get("rear_recapture_revision") is None
                else identity["rear_recapture_revision"]["target_mode"]),
            "rear_live_swing_mode": (None if identity.get("rear_live_swing_revision") is None
                else identity["rear_live_swing_revision"]["target_mode"]),
            "p02_progress_credit_mode": (None
                if identity.get("p02_progress_revision") is None else
                identity["p02_progress_revision"]["p02_progress_credit_mode"]),
            "cooperative_preparation_mode": (None
                if identity.get("cooperative_preparation_revision") is None else
                identity["cooperative_preparation_revision"]
                    ["cooperative_preparation_revision"]),
            "policy_version": identity["policy_version"],
            "observation_layout": identity["observation_layout"],
            "observation_dimension": identity["observation_dimension"],
            "checkpoint_output_branch": (None if identity.get("rear_recapture_revision") is None
                else identity["rear_recapture_revision"]["checkpoint_output_routing"]["branch"]),
            "training": "PPO + inherited limited AUX", "this_run_AUX_updates": 0,
            "capture_abort_not_task_result": candidate["diagnostic_partial"]},
        "physical_result": outcome["result"],
        "physical_task_success": outcome["success"],
        "termination_reason": outcome["termination_reason"],
        "termination_source": outcome["termination_source"],
        "source_acceptance_error": outcome["source_acceptance_error"],
        "task_terminal_observed": outcome["task_terminal_observed"],
        "RR_window_reached": rr_reached(rows), "RL_window_reached": rl_reached(rows),
        "requested_RR_detail_unavailable_reason":
            plan["requested_RR_detail_unavailable_reason"],
        "full_episode_continuous": not candidate["diagnostic_partial"],
        "continuous_encoded_prefix": True,
        "full_failure_tail_preserved": not candidate["diagnostic_partial"],
        "all_available_encoded_tail_preserved": True,
        "diagnostic_capture_abort": candidate["diagnostic_partial"],
        "capture_abort": ({
            "classification": "VIDEO_ARTIFACT_ERROR_NOT_TASK_FAILURE",
            "declared_episode_physics_ticks": candidate["endpoint"],
            "last_encoded_physics_tick": source_validation["last_encoded_physics_tick"],
            "actual_encoded_frames": source_validation["actual_encoded_frame_count"],
            "declared_interval_frames": source_validation["declared_episode_frame_count"],
            "missing_physical_interval_s": source_validation["missing_physical_interval_s"],
            "missing_final_frame_fabricated": False, "source_modified": False,
        } if candidate["diagnostic_partial"] else None),
        "normal_speed": True, "single_episode": True, "stitched": False,
        "speed_modified": False, "extra_intro_frames": 0,
        "overlay_semantics": {
            "schema": "wlr50_clean.rear_policy_video_overlay_alignment.v2",
            "encoded_frame_clock": "episode decision end_tick == step_info.physics_tick",
            "mapped_N_baseline_key":
                "step_info.actuator_target_effect_audit.native_drive_target_full12[8:12]",
            "mapped_N_crosscheck_key":
                "step_info.actuator_target_effect_audit.policy_headroom_evidence.baseline_native_plus_controller_full12[8:12]",
            "mapped_N_physical_axis_key":
                "step_info.actuator_target_effect_audit.counterfactual_native_targets.wheel_velocity_rad_s",
            "mapped_N_semantics":
                "same-decision counterfactual nominal/controller baseline; not executed alone",
            "executed_final_key": "step_info.actual_drive_target_full12[8:12]",
            "poststep_source_hint_key": "sourcecapture_assist_ticks.nominal_full12[8:12]",
            "poststep_source_hint_semantics":
                "post-step source suggestion shown separately; not relabeled as executed mapped N",
            "actuator_raw_clock_key":
                "step_info.actuator_target_effect_audit.physics_tick",
            "source_partial_order_key":
                "step_info.semantic_task.nominal_provider_diagnostics.source_partial_order.layers[current phase, observation_tick=end_tick]",
            "public_dependency_flag_key": "sourcecapture_assist_ticks.rear_policy_timing.*_dependency_wait",
            "RL_contact_keys": ["current_legs.RL.contact_mode",
                                "current_legs.RL.contact_surface",
                                "current_legs.RL.bearing_verified"],
            "RL_contact_fallback":
                "explicit front-schema fallback only when contact_mode is unavailable",
        },
        "full": full, "detail": detail, "historical_N_comparison": pair,
        "historical_N_is_fresh_same_controller_B": False,
        "historical_N_freezes_after_its_own_endpoint": True,
        "selected_real_events": str(evidence_path),
        "selected_real_events_sha256": sha256(evidence_path),
        "claims": {"front_FL_assist_is_policy_learning": False,
            "rear_task_assist_active": False,
            "rear_policy_branch_AUX_added": False,
            "rear_recapture_migration_added_learning": False,
            "rear_live_swing_migration_added_learning": False,
            "p02_progress_migration_added_learning": False,
            "cooperative_prep_migration_added_learning": False,
            "branch_output_promoted_main_pointer": False,
            "RR_placed_history_is_current_capture_success": False,
            "capture_abort_is_task_failure_or_success": False,
            "missing_callback_frame_fabricated": False,
            "historical_N_is_same_version_or_same_controller": False,
            "frozen_comparison_frames_are_physical_evidence": False,
            "failure_video_is_success": False,
            "unreached_RR_detail_is_substituted_from_another_run": False,
            "mapped_N_baseline_was_executed_alone": False,
            "poststep_source_hint_is_executed_mapped_N": False,
            "public_dependency_flag_is_source_partial_order_wait": False,
            "RL_ground_and_obstacle_is_body_collision_or_pure_wheel_climb": False},
        "code_binding": {"exporter_sha256": sha256(Path(__file__)),
                         "shared_media_helper_sha256": sha256(SHARED_EXPORTER)},
    }
    receipt_path = destination / "export_receipt.json"
    write_new_json(receipt_path, receipt)
    print(json.dumps({"physical_result": outcome["result"], "full": full["output"],
        "detail": detail["output"], "comparison": pair["output"],
        "receipt": str(receipt_path)}, indent=2))
    return receipt


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument("--source", type=Path, required=True)
    result.add_argument("--destination", type=Path, required=True)
    result.add_argument("--source-manifest-sha256", required=True)
    result.add_argument("--source-run-manifest-sha256", required=True)
    result.add_argument("--diagnostic-capture-abort", action="store_true",
        help=("accept only a sealed callback_count=0 artifact abort with one missing "
              "final render interval; never fabricate that frame or a task result"))
    result.add_argument("--expected-head", required=True)
    result.add_argument("--checkpoint-runtime-head",
        help=("saved checkpoint runtime; defaults to --expected-head for exact-runtime "
              "evaluation and is required explicitly only for a reviewed media-only split"))
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--checkpoint-sha256", required=True)
    result.add_argument("--checkpoint-manifest-sha256", required=True)
    result.add_argument("--expected-global-policy-decisions", type=int, required=True)
    result.add_argument("--expected-ppo-updates", type=int, required=True)
    result.add_argument("--expected-optimizer-steps", type=int, required=True)
    result.add_argument("--expected-new-auxiliary-updates", type=int, default=0)
    result.add_argument("--historical-n-source", type=Path, default=DEFAULT_HISTORICAL_N)
    return result


if __name__ == "__main__":
    export(parser().parse_args())
