"""Export one sealed RR-capture/RL-transfer deterministic diagnostic video.

This outputs-only helper starts neither Isaac nor a policy.  It accepts only a
sealed ``rr_capture_then_rl_transfer_v1`` source loaded from the explicit
389->410 migration, keeps every source frame on the native 15 fps timeline,
adds a transparent evidence HUD, emits one contiguous RR-capture-to-RL excerpt
when that window was actually observed, and makes a visibly labelled
historical-N comparison.  Task failure stays failure; the historical N is not
presented as a fresh same-controller B run.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
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

EXPERIMENT = "rr_capture_then_rl_transfer_v1"
CONTROL_METHOD = "PPO_PLUS_FL_RR_CAPTURE_ASSISTS_WITH_INHERITED_LIMITED_AUX"
MIGRATION_SCHEMA = "wlr50_clean.rr_capture_transfer_append.v1"
FEEDBACK_V2_SCHEMA = "wlr50_clean.rr_capture_feedback_same410.v2"
FEEDBACK_V2_FACTOR = "rr_capture_feedback_peak_v2_factor"
FEEDBACK_V2_KEY = "rr_capture_feedback_peak_v2_migration"
FEEDBACK_V2_SOURCE_REVISION = "implicit_window_start_net_progress_v1"
FEEDBACK_V2_REVISION = "window_peak_progress_v2"
FEEDBACK_V2_WINDOW = (
    "public_window_peak_gap_reuses_existing_window_start_gap_scalar_"
    "upward_motion_never_resets_elapsed"
)
POLICY_VERSION = "rr_capture_then_rl_transfer_history_v1"
OBSERVATION_LAYOUT = "role389_rr_capture_transfer_v1"
RR_ASSIST_SCHEMA = "wlr50_clean.rr_capture_assist_state.v1"
RR_ASSIST_MODE = "rr_hip_only_capture_v1"
COUNTERS = ("global_policy_decisions", "ppo_updates", "optimizer_steps")
EXPECTED_CHECKPOINT_DECISIONS = 220544
FPS, HZ, STRIDE, MAX_FRAMES = 15, 120, 8, 3000

# These names/signs are part of the sealed 26db source contract.  Keep them
# local: importing the live runtime here would silently apply later assist
# validators (for example feedback-v3) to this historical feedback-v2 source.
SERVO_ORDER = (
    "front_left_hip", "front_left_knee", "front_right_hip", "front_right_knee",
    "rear_left_hip", "rear_left_knee", "rear_right_hip", "rear_right_knee",
)
WHEEL_ORDER = (
    "front_left_ankle", "front_right_ankle", "rear_left_ankle", "rear_right_ankle",
)
FULL12_ORDER = SERVO_ORDER + WHEEL_ORDER
WHEEL_FORWARD_SIGN = {
    "front_left_ankle": -1.0, "front_right_ankle": 1.0,
    "rear_left_ankle": -1.0, "rear_right_ankle": 1.0,
}
RR_TASK_FIELDS = (
    "rr_lift_carry", "rr_top_reachable", "rr_top_contact", "rr_current_bearing",
    "rl_transfer_ready", "rr_capture_recovery_allowed", "fl_wheel_guidance_active",
)
RR_V2_FEATURE_NAMES = (
    "mode", "initialized", "knee_hold_deg", "hip_entry_deg", "hip_target_deg",
    "travel_used_deg", "descent_elapsed_s", "window_start_gap_m",
    "window_elapsed_s", "hold_elapsed_s", "release_fraction", "contact_seen",
    "retired", "blocked_reason",
)


_SHARED: Any = None


def shared() -> Any:
    """Load the already-used media/decode helpers without copying release logic."""
    global _SHARED
    if _SHARED is None:
        spec = importlib.util.spec_from_file_location("_p05_media_shared", SHARED_EXPORTER)
        require(spec is not None and spec.loader is not None,
                "shared P05 media helper cannot be loaded")
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


def write_new_json(path: Path, value: Any) -> None:
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def write_new_text(path: Path, value: str) -> None:
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(value.rstrip() + "\n")


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


def _sidecar_from_checkpoint(checkpoint: Path) -> Path:
    return checkpoint.with_name(checkpoint.stem + "_manifest.json")


def checkpoint_identity(manifest: dict[str, Any]) -> dict[str, Any]:
    """Validate the sealed RR410 checkpoint and optional feedback-v2 boundary.

    This intentionally validates the historical checkpoint/plan bytes.  It
    does not import the current runtime's assist or migration validators.
    """
    inherited = shared().checkpoint_identity(manifest)
    proof = manifest.get("checkpoint_load_provenance") or {}
    binding = proof.get("source") or {}
    checkpoint = Path(binding.get("checkpoint", "")).resolve(strict=True)
    sidecar = Path(binding.get("manifest", "")).resolve(strict=True)
    metadata = read_json(sidecar)

    require(metadata.get("global_policy_decisions") == EXPECTED_CHECKPOINT_DECISIONS,
            "this initial export must use CP220544")
    require((metadata.get("runtime_contract") or {}).get("experiment_id") == EXPERIMENT,
            "checkpoint runtime is not the RR-capture experiment")
    policy = metadata.get("policy_contract") or {}
    require(policy.get("version") == POLICY_VERSION and
            policy.get("observation_layout") == OBSERVATION_LAYOUT and
            policy.get("observation_dimension") == 410 and
            policy.get("raw_action_dimension") == 12,
            "checkpoint lacks the exact RR410 policy contract")

    migration = metadata.get("rr_capture_transfer_migration")
    branch = metadata.get("rr_capture_transfer_branch")
    counts = metadata.get("rr_capture_transfer_branch_counts")
    require(isinstance(migration, dict) and migration.get("schema") == MIGRATION_SCHEMA,
            "checkpoint lacks the explicit RR migration")
    require(isinstance(branch, dict) and branch.get("schema") == MIGRATION_SCHEMA,
            "checkpoint lacks the RR transfer branch")
    factor = migration.get("rr_capture_transfer_factor")
    require(isinstance(factor, dict) and factor.get("schema") == MIGRATION_SCHEMA,
            "RR migration factor is missing")
    origin = factor.get("counter_origin")
    require(isinstance(origin, dict) and branch.get("counter_origin") == origin,
            "RR branch origin differs from the migration")
    expected_counts: dict[str, int] = {}
    for key in COUNTERS:
        current, initial = metadata.get(key), origin.get(key)
        require(type(current) is int and type(initial) is int and current >= initial,
                f"invalid RR branch {key}")
        expected_counts[key] = current - initial
    require(counts == expected_counts and all(value == 0 for value in expected_counts.values()),
            "initial RR410 video must have zero new PPO decisions/updates/Adam steps")
    require(branch.get("source_checkpoint_sha256") == migration.get("source_checkpoint_sha256")
            and branch.get("migration_added_updates") == 0,
            "RR branch is not bound to its migration source")
    require(all(factor.get(f"added_{name}") == 0 for name in
                ("policy_decisions", "ppo_updates", "optimizer_steps", "auxiliary_updates")),
            "RR migration itself must add no optimization credit")
    require(factor.get("control_semantics_changed") is True and
            factor.get("same_mdp_claimed") is False and
            factor.get("physical_dynamics_changed") is False and
            factor.get("old_rollout_inherited") is False and
            factor.get("physical_state_inherited") is False,
            "RR migration semantic boundary is incomplete")
    require(factor.get("target_policy_contract") == policy and
            factor.get("target_observation_dimension") == 410 and
            factor.get("target_observation_layout") == OBSERVATION_LAYOUT,
            "RR migration target contract differs from the loaded checkpoint")
    feedback_migration = metadata.get(FEEDBACK_V2_KEY)
    if feedback_migration is None:
        require(migration.get("target_git_commit") ==
                (metadata.get("runtime_contract") or {}).get("source_git_commit"),
                "RR migration target commit differs from the checkpoint runtime")
        require(metadata.get("resume_migration") == migration,
                "initial RR checkpoint must retain its exact resume migration")

    plan_path = Path(migration.get("plan_path", "")).resolve(strict=True)
    require(sha256(plan_path) == migration.get("plan_sha256"),
            "RR migration plan changed after publication")
    supplied = read_json(plan_path)
    require({key: value for key, value in migration.items()
             if key not in ("plan_path", "plan_sha256")} == supplied,
            "published RR migration differs from its immutable plan")

    source_checkpoint = Path(migration.get("source_checkpoint", "")).resolve(strict=True)
    source_sidecar = _sidecar_from_checkpoint(source_checkpoint).resolve(strict=True)
    require(sha256(source_checkpoint) == migration.get("source_checkpoint_sha256") and
            sha256(source_sidecar) == migration.get("source_manifest_sha256"),
            "RR migration source checkpoint binding changed")
    source_metadata = read_json(source_sidecar)
    preserved = factor.get("preserved_metadata_sha256")
    require(isinstance(preserved, dict) and preserved,
            "RR migration lacks preserved metadata bindings")
    for key, digest in preserved.items():
        require(key in source_metadata and json_digest(source_metadata[key]) == digest,
                f"RR migration source metadata changed: {key}")
        require(metadata.get(key) == source_metadata[key],
                f"RR publication failed to preserve source metadata: {key}")

    require(checkpoint == Path(inherited["checkpoint"]).resolve() and
            sidecar == Path(inherited["manifest"]).resolve(),
            "inherited lineage validator used another checkpoint")
    result = {
        **inherited,
        "checkpoint": str(checkpoint),
        "manifest": str(sidecar),
        "rr_capture_transfer_counter_origin": dict(origin),
        "rr_capture_transfer_branch_counts": expected_counts,
        "rr_capture_transfer_migration": {
            "schema": MIGRATION_SCHEMA,
            "plan_path": str(plan_path),
            "plan_sha256": migration["plan_sha256"],
            "source_checkpoint": str(source_checkpoint),
            "source_checkpoint_sha256": migration["source_checkpoint_sha256"],
            "source_manifest_sha256": migration["source_manifest_sha256"],
            "source_git_commit": migration["source_git_commit"],
            "target_git_commit": migration["target_git_commit"],
            "observation_transition": [389, 410],
            "migration_added_policy_decisions": 0,
            "migration_added_ppo_updates": 0,
            "migration_added_optimizer_steps": 0,
            "migration_added_auxiliary_updates": 0,
        },
    }
    if feedback_migration is None:
        result["rr_capture_control_revision"] = "initial_rr410_v1"
        result["rr_capture_feedback_v2_present"] = False
        return result

    require(isinstance(feedback_migration, dict) and
            feedback_migration.get("schema") == FEEDBACK_V2_SCHEMA,
            "checkpoint has an invalid feedback-v2 migration")
    feedback_factor = feedback_migration.get(FEEDBACK_V2_FACTOR)
    require(isinstance(feedback_factor, dict) and
            feedback_factor.get("schema") == FEEDBACK_V2_SCHEMA,
            "feedback-v2 factor is missing")
    feedback_origin = feedback_factor.get("counter_origin")
    require(isinstance(feedback_origin, dict) and set(feedback_origin) == set(COUNTERS)
            and all(metadata.get(key) == feedback_origin[key] for key in COUNTERS),
            "feedback-v2 checkpoint counters differ from the migration origin")
    require(all(feedback_factor.get(f"added_{name}") == 0 for name in
                ("policy_decisions", "ppo_updates", "optimizer_steps",
                 "auxiliary_updates")),
            "feedback-v2 migration must add zero learning credit")
    require(feedback_factor.get("source_feedback_revision") ==
                FEEDBACK_V2_SOURCE_REVISION and
            feedback_factor.get("target_feedback_revision") == FEEDBACK_V2_REVISION and
            feedback_factor.get("window_reference_semantics") == FEEDBACK_V2_WINDOW,
            "feedback-v2 revision semantics differ from the sealed boundary")
    expected_flags = {
        "creates_new_branch": False,
        "same_mdp_claimed": False,
        "physical_dynamics_changed": False,
        "reward_changed": False,
        "policy_kernel_changed": False,
        "same_numeric_input_policy_mapping_preserved": True,
        "same_physical_state_action_equivalence_claimed": False,
        "discard_old_rollout_storage": True,
        "physical_state_inherited": False,
        "pre_cross_descent_allowed": False,
    }
    require(all(feedback_factor.get(key) == expected
                for key, expected in expected_flags.items()),
            "feedback-v2 migration makes an unsupported policy/physics/MDP claim")
    observation = feedback_factor.get("observation_contract") or {}
    require(observation.get("observation_dimension") == 410 and
            observation.get("action_dimension") == 12 and
            observation.get("observation_layout") == OBSERVATION_LAYOUT and
            observation.get("source_policy_contract") == policy and
            observation.get("target_policy_contract") == policy,
            "feedback-v2 must retain the exact RR410 v1 policy contract/kernel")

    feedback_plan = Path(feedback_migration.get("plan_path", "")).resolve(strict=True)
    require(sha256(feedback_plan) == feedback_migration.get("plan_sha256"),
            "feedback-v2 immutable plan changed after publication")
    supplied_feedback = read_json(feedback_plan)
    require({key: value for key, value in feedback_migration.items()
             if key not in ("plan_path", "plan_sha256")} == supplied_feedback,
            "published feedback-v2 migration differs from its immutable plan")
    feedback_source = Path(feedback_migration.get("source_checkpoint", "")).resolve(
        strict=True)
    feedback_source_sidecar = _sidecar_from_checkpoint(feedback_source).resolve(strict=True)
    require(sha256(feedback_source) == feedback_migration.get("source_checkpoint_sha256")
            and sha256(feedback_source_sidecar) ==
                feedback_migration.get("source_manifest_sha256"),
            "feedback-v2 source checkpoint binding changed")
    feedback_source_metadata = read_json(feedback_source_sidecar)
    require(Path(feedback_source_metadata.get("checkpoint_path", "")).resolve() ==
                feedback_source and
            feedback_source_metadata.get("checkpoint_sha256") ==
                feedback_migration.get("source_checkpoint_sha256") and
            all(feedback_source_metadata.get(key) == feedback_origin[key]
                for key in COUNTERS) and
            feedback_source_metadata.get(FEEDBACK_V2_KEY) is None,
            "feedback-v2 source is not the exact pre-v2 CP220544 checkpoint")
    require(feedback_factor.get("source_resume_migration") ==
                feedback_source_metadata.get("resume_migration") and
            feedback_factor.get("source_resume_migration") == migration and
            metadata.get("resume_migration") == feedback_migration,
            "feedback-v2 resume ancestry does not retain the original RR migration")
    feedback_preserved = feedback_factor.get("preserved_metadata_sha256")
    require(isinstance(feedback_preserved, dict) and feedback_preserved,
            "feedback-v2 factor lacks migration-time preserved metadata")
    for key, digest in feedback_preserved.items():
        require(key in feedback_source_metadata and
                json_digest(feedback_source_metadata[key]) == digest,
                f"feedback-v2 source metadata changed: {key}")
        require(metadata.get(key) == feedback_source_metadata[key],
                f"feedback-v2 publication changed preserved metadata: {key}")

    runtime = metadata.get("runtime_contract") or {}
    require(feedback_migration.get("source_git_commit") ==
                (feedback_source_metadata.get("runtime_contract") or {}).get(
                    "source_git_commit") and
            feedback_migration.get("target_git_commit") == runtime.get("source_git_commit") and
            feedback_migration.get("target_runtime_content_sha256") ==
                runtime.get("runtime_content_sha256"),
            "feedback-v2 checkpoint runtime does not match its frozen target")
    reviewed = feedback_factor.get("reviewed_code_sha256")
    runtime_files = runtime.get("files") or {}
    require(isinstance(reviewed, dict) and reviewed and
            feedback_migration.get("allowed_changed_files") == sorted(reviewed) and
            all(runtime_files.get(path) == digest for path, digest in reviewed.items()),
            "sealed 26db runtime files differ from feedback-v2 reviewed hashes")

    result.update({
        "rr_capture_control_revision": FEEDBACK_V2_REVISION,
        "rr_capture_feedback_v2_present": True,
        "rr_capture_feedback_v2": {
            "schema": FEEDBACK_V2_SCHEMA,
            "plan_path": str(feedback_plan),
            "plan_sha256": feedback_migration["plan_sha256"],
            "source_checkpoint": str(feedback_source),
            "source_checkpoint_sha256": feedback_migration["source_checkpoint_sha256"],
            "source_manifest_sha256": feedback_migration["source_manifest_sha256"],
            "source_git_commit": feedback_migration["source_git_commit"],
            "target_git_commit": feedback_migration["target_git_commit"],
            "source_feedback_revision": FEEDBACK_V2_SOURCE_REVISION,
            "target_feedback_revision": FEEDBACK_V2_REVISION,
            "window_reference_semantics": FEEDBACK_V2_WINDOW,
            "counter_origin": dict(feedback_origin),
            "migration_added_policy_decisions": 0,
            "migration_added_ppo_updates": 0,
            "migration_added_optimizer_steps": 0,
            "migration_added_auxiliary_updates": 0,
            "policy_contract_unchanged": True,
            "policy_version": POLICY_VERSION,
        },
    })
    return result


def sealed_source(source: Path, *, candidate: bool) -> dict[str, Any]:
    source = Path(source).resolve(strict=True)
    run_path = source.parent / "run_manifest.json"
    require(run_path.is_file(), "sealed parent run manifest is missing")
    run_manifest = read_json(run_path)
    require(bool(run_manifest.get("completed_at_utc")) and
            run_manifest.get("lifecycle") != "RUNNING",
            "source run is still active")
    manifest_path = source / "semantic_video_source_manifest.json"
    manifest = read_json(manifest_path)
    require(manifest.get("schema") == "wlr50_clean.semantic_video_source.v1" and
            manifest.get("from_phase") == "P01" and
            manifest.get("fresh_process_single_episode") is True and
            manifest.get("episode_count") == 1 and
            manifest.get("optimizer_updates") == 0,
            "source is not one natural-P01 evaluation episode")
    require(run_manifest.get("runtime_contract") == manifest.get("runtime_contract"),
            "run/source runtime contract mismatch")
    endpoint = manifest.get("episode_physics_ticks")
    require(type(endpoint) is int and 0 < endpoint <= 24000,
            "episode endpoint is outside the 200 s task horizon")
    if candidate:
        require(manifest.get("experiment_id") == EXPERIMENT and manifest.get("role") == "C",
                "candidate is not the RR-capture C experiment")
        require(manifest.get("control_method") == CONTROL_METHOD,
                "candidate control method lacks both declared capture assists and inherited AUX")
        if "source_control_method" in manifest:
            require(manifest["source_control_method"] == CONTROL_METHOD,
                    "candidate control-method aliases disagree")
        require(manifest.get("capture_assist_enabled_in_training_and_evaluation") is True and
                manifest.get("capture_assist_is_policy_learning") is False,
                "FL capture assist provenance is incomplete")
        rr = manifest.get("rr_capture_assist")
        require(rr == {
            "mode": RR_ASSIST_MODE,
            "training_and_evaluation_identical": True,
            "first_revision_wheel_guidance": "off",
            "is_policy_learning": False,
            "declared_owner_indices": [6, 7],
            "changes_physical_contact_evidence": False,
        }, "RR hip-only/wheel-off source declaration differs")
        error = manifest.get("source_acceptance_error")
        require(error is None or (isinstance(error, str) and
                "episode did not meet common physical task" in error and
                "VIDEO_OR_ARTIFACT_ERROR" not in error),
                "artifact/infrastructure error is not a task-attempt video")
        checkpoint = checkpoint_identity(manifest)
        tick_path = artifact(source, manifest, "capture_assist_ticks.jsonl")
        native_tick_path = artifact(source, manifest, "native_tick_audit.jsonl")
    else:
        require(manifest.get("role") == "B" and
                manifest.get("physical_task_success") is True and
                manifest.get("success_candidate") is True and
                manifest.get("diagnostic_only") is False,
                "historical N source must be the accepted successful B source")
        checkpoint, tick_path, native_tick_path = None, None, None
    video = artifact(source, manifest, "actual_viewport_video.mp4")
    ledger_path = artifact(source, manifest, "viewport_frame_ledger.jsonl")
    capture_path = artifact(source, manifest, "viewport_buffer_video_manifest.json")
    capture = read_json(capture_path)
    require(capture.get("valid") is True and
            capture.get("encoder_finalized_before_app_close") is True,
            "source writer did not close with a valid full decode")
    return {"source": source, "manifest": manifest, "manifest_path": manifest_path,
        "run_manifest": run_manifest, "run_manifest_path": run_path, "video": video,
        "ledger_path": ledger_path, "capture": capture, "capture_path": capture_path,
        "tick_path": tick_path, "native_tick_path": native_tick_path,
        "checkpoint": checkpoint, "endpoint": endpoint}


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        return float(value)
    return None


def _joint(row: dict[str, Any], name: str) -> float | None:
    return _number(((row.get("joints") or {}).get(name) or {}).get("position_deg"))


def _vector(value: Any, label: str) -> list[float]:
    require(isinstance(value, list) and len(value) == 12,
            f"tick lacks {label} Full12")
    result = [_number(item) for item in value]
    require(all(item is not None for item in result), f"{label} Full12 is non-finite")
    return [float(item) for item in result]


def _contact(row: dict[str, Any]) -> tuple[str, float | None]:
    if row.get("top_surface_contact") is True:
        label = "TOP"
    elif row.get("ground_contact") is True:
        label = "GROUND"
    elif row.get("air") is True:
        label = "AIR"
    else:
        label = str(row.get("contact_surface") or "NO_TOP")
    force = next((_number(row.get(key)) for key in
                  ("obstacle_normal_force_n", "contact_reaction_force_n", "bearing_force_n")
                  if _number(row.get(key)) is not None), None)
    return label, force


def _vec3(value: Any, label: str) -> list[float]:
    require(isinstance(value, list) and len(value) == 3,
            f"tick lacks {label} xyz")
    result = [_number(item) for item in value]
    require(all(item is not None for item in result), f"{label} xyz is non-finite")
    return [float(item) for item in result]


def validate_historical_rr_snapshot(state: dict[str, Any]) -> None:
    """Validate historical v1/v2 snapshots without importing live runtime."""
    require(state.get("schema") == RR_ASSIST_SCHEMA and
            state.get("version") == RR_ASSIST_MODE,
            "tick has another RR assist schema/version")
    revision = state.get("feedback_revision")
    if revision is not None:
        require(revision == FEEDBACK_V2_REVISION and
                state.get("window_reference_semantics") == FEEDBACK_V2_WINDOW,
                "tick is not the sealed feedback-v2 RR assist revision")
    require(state.get("feature_names") == list(RR_V2_FEATURE_NAMES),
            "feedback-v2 snapshot has another 14-scalar state layout")
    require(all(_number(state.get(name)) is not None for name in RR_V2_FEATURE_NAMES),
            "feedback-v2 snapshot has a non-finite state scalar")
    require(isinstance(state.get("mode_name"), str) and
            isinstance(state.get("reason"), str) and
            type(state.get("active")) is bool and
            isinstance(state.get("owners"), list) and len(state["owners"]) == 2 and
            isinstance(state.get("owner_indices"), list) and
            set(state["owner_indices"]).issubset({6, 7}) and
            type(state.get("last_dispatch_physics_tick")) is int,
            "feedback-v2 snapshot ownership/clock fields are malformed")


def frame_summary(row: dict[str, Any], native_row: dict[str, Any]) -> dict[str, Any]:
    rr_state = row.get("rr_capture_assist")
    require(isinstance(rr_state, dict), "tick lacks RR capture-assist state")
    validate_historical_rr_snapshot(rr_state)
    fl_state = row.get("capture_assist") or {}
    require(fl_state.get("schema") == "wlr50_clean.capture_assist_state.v1",
            "tick lacks FL capture-assist state")
    context = row.get("rr_capture_transfer_context")
    require(isinstance(context, dict) and set(context) == set(RR_TASK_FIELDS) and
            all(type(context[key]) is bool for key in RR_TASK_FIELDS),
            "tick lacks the exact seven RR transfer-context booleans")
    diagnostics = row.get("rr_capture_transfer_diagnostics") or {}
    require(diagnostics.get("mode") == EXPERIMENT and
            diagnostics.get("air_is_support") is False,
            "tick lacks RR transfer diagnostic semantics")
    tick = row.get("episode_physics_tick")
    dispatch = row.get("dispatch") or {}
    dispatch_tick = dispatch.get("physics_tick")
    require(type(dispatch_tick) is int and dispatch_tick >= tick,
            "dispatch clock is not a valid monotonic physical tick")
    native = _vector(dispatch.get("native_drive_target_full12"), "native drive target")
    final = _vector(dispatch.get("drive_target_full12"), "canonical final target")
    actual = _vector(row.get("actual_full12"), "canonical actual readback")
    require(dispatch.get("rr_capture_assist_evidence") is not None,
            "tick lacks RR actuator evidence")
    rr_evidence = dispatch["rr_capture_assist_evidence"]
    correction = _vector(rr_evidence.get("assist_correction_full12"), "RR assist correction")
    require(rr_evidence.get("policy_request_unchanged") is True and
            set(rr_evidence.get("owner_indices") or ()).issubset({6, 7}) and
            all(abs(correction[index]) <= 1e-12 for index in range(12) if index not in (6, 7)),
            "RR assist wrote outside its declared hip/knee owner")
    pre = rr_evidence.get("context") or {}
    require(pre.get("dispatch_physics_tick") == dispatch_tick and
            pre.get("source_observation_tick") == tick - 1,
            "RR assist pre-dispatch evidence is not the adjacent source observation")
    require(rr_state == rr_evidence.get("state_after"),
            "post-step RR state differs from this tick's committed assist state")
    require(rr_state.get("last_dispatch_physics_tick") == dispatch_tick,
            "top-level RR state has another dispatch clock")

    require(native_row.get("episode_physics_tick") == tick,
            "native audit/tick identity mismatch")
    native_audit = native_row.get("native_audit") or {}
    require(native_audit.get("verified") is True and
            native_audit.get("physics_tick") == dispatch_tick and
            native_audit.get("canonical_order") == list(FULL12_ORDER),
            "tick lacks the verified native actuator audit")
    require(native_audit.get("rr_capture_assist_state_transition_independently_reconstructed") is True and
            native_audit.get("rr_capture_assist_evidence") == rr_evidence,
            "native audit did not independently reconstruct this RR assist transition")
    audited_mapper_target = _vector(native_audit.get("native_drive_target_full12"),
                                    "native-audit mapper target")
    require(all(math.isclose(a, b, rel_tol=0., abs_tol=1e-12)
                for a, b in zip(audited_mapper_target, native)),
            "capture/native ledgers disagree on the pre-bias mapper target")
    native_targets = native_audit.get("actual_native_targets") or {}
    physical_wheels_raw = native_targets.get("wheel_velocity_rad_s")
    require(isinstance(physical_wheels_raw, list) and len(physical_wheels_raw) == 4,
            "native audit lacks four actual PhysX wheel targets")
    physical_wheels = [_number(value) for value in physical_wheels_raw]
    require(all(value is not None for value in physical_wheels),
            "native PhysX wheel targets are non-finite")
    physical_wheels = [float(value) for value in physical_wheels]
    physical_servos_raw = native_targets.get("servo_position_rad")
    require(isinstance(physical_servos_raw, list) and len(physical_servos_raw) == 8,
            "native audit lacks eight actual PhysX servo targets")
    physical_servos = [_number(value) for value in physical_servos_raw]
    require(all(value is not None for value in physical_servos),
            "native PhysX servo targets are non-finite")
    physical_servos = [float(value) for value in physical_servos]
    expected_physical = [WHEEL_FORWARD_SIGN[name] * final[8 + index]
                         for index, name in enumerate(WHEEL_ORDER)]
    require(all(math.isclose(actual_value, expected_value, rel_tol=0., abs_tol=2e-6)
                for actual_value, expected_value in zip(physical_wheels, expected_physical)),
            "native PhysX wheel targets violate the locked per-axis sign mapping")
    wheel_rows = row.get("wheels") or {}
    measured = []
    for index, name in enumerate(WHEEL_ORDER):
        value = _number((wheel_rows.get(name) or {}).get("velocity_rad_s"))
        require(value is not None and math.isclose(value, actual[8 + index], rel_tol=0., abs_tol=1e-9),
                f"{name} measured velocity differs from canonical actual_full12")
        measured.append(float(value))
    legs = row.get("current_legs") or {}
    rr_leg = legs.get("RR") or {}
    fl_leg = legs.get("FL") or {}
    body_position = _vec3((row.get("body") or {}).get("position_w_m"),
                          "body world position")
    center_of_mass = _vec3((row.get("center_of_mass") or {}).get("position_w_m"),
                           "center-of-mass world position")
    rr_wheel = (row.get("wheels") or {}).get(WHEEL_ORDER[3]) or {}
    rr_center = _vec3(rr_wheel.get("center_w_m"), "RR wheel-center world position")
    rr_center_body = [rr_center[index] - body_position[index] for index in range(3)]
    contact, force = _contact(rr_leg)
    pre_contact = ("TOP" if pre.get("top_surface_contact") is True else
                   "GROUND" if pre.get("ground_contact") is True else
                   "AIR" if pre.get("air") is True else
                   "OBSTACLE" if pre.get("obstacle_pair_active") is True else "NO_TOP")
    continuation = row.get("capture_continuation") or {}
    require(type(continuation.get("fl_capture_pending")) is bool,
            "tick lacks explicit FL capture-pending history/current status")
    initialized = bool(rr_state.get("initialized"))
    return {
        "tick": tick,
        "dispatch_physics_tick": dispatch_tick,
        "dispatch_to_episode_tick_offset": dispatch_tick - tick,
        "time_s": _number(row.get("sim_time_s")),
        "phase": row.get("phase"),
        "fl_assist_mode": fl_state.get("mode_name"),
        "fl_assist_active": bool(fl_state.get("active")),
        "rr_assist_mode": rr_state.get("mode_name"),
        "rr_assist_active": bool(rr_state.get("active")),
        "rr_assist_reason": rr_state.get("reason"),
        "rr_assist_owner_indices": list(rr_state.get("owner_indices") or []),
        "rr_assist_feedback_revision": rr_state.get("feedback_revision"),
        "rr_assist_window_reference_semantics": rr_state.get(
            "window_reference_semantics"),
        "rr_assist_travel_used_deg": _number(rr_state.get("travel_used_deg")),
        "rr_assist_descent_elapsed_s": _number(rr_state.get("descent_elapsed_s")),
        "rr_assist_window_elapsed_s": _number(rr_state.get("window_elapsed_s")),
        "rr_hip_assist_target_deg": _number(rr_state.get("hip_target_deg")) if initialized else None,
        "rr_hip_final_target_deg": final[6],
        "rr_hip_actual_deg": _joint(row, SERVO_ORDER[6]),
        "rr_hip_native_target_rad": physical_servos[6],
        "rr_knee_hold_target_deg": _number(rr_state.get("knee_hold_deg")) if initialized else None,
        "rr_knee_final_target_deg": final[7],
        "rr_knee_actual_deg": _joint(row, SERVO_ORDER[7]),
        "rr_knee_native_target_rad": physical_servos[7],
        "rr_pre_observation_tick": pre.get("source_observation_tick"),
        "rr_post_actual_episode_tick": tick,
        "rr_pre_gap_mm": (None if _number(pre.get("gap_m")) is None
                          else 1000. * float(pre["gap_m"])),
        "rr_pre_contact": pre_contact,
        "rr_pre_hip_actual_deg": _number(pre.get("hip_actual_deg")),
        "rr_pre_knee_actual_deg": _number(pre.get("knee_actual_deg")),
        "rr_pre_current_bearing": pre.get("current_top_bearing"),
        "rr_post_gap_mm": (None if _number(rr_leg.get("clearance_m")) is None
                      else 1000. * float(rr_leg["clearance_m"])),
        "rr_post_contact": contact,
        "rr_contact_reaction_force_n": _number(rr_leg.get("contact_reaction_force_n")),
        "rr_bearing_force_n": _number(rr_leg.get("bearing_force_n")),
        "rr_bearing_verified": rr_leg.get("bearing_verified"),
        "rr_obstacle_normal_force_n": _number(rr_leg.get("obstacle_normal_force_n")),
        "rr_display_force_n": force,
        "rr_placed_history": (row.get("placed_history") or {}).get("RR"),
        "rr_front_edge_crossed": rr_leg.get("front_edge_crossed"),
        "rr_front_distance_mm": (None if _number(rr_leg.get("front_distance_m")) is None
                                  else 1000. * float(rr_leg["front_distance_m"])),
        "rr_lift_carry": context["rr_lift_carry"],
        "rr_top_reachable": context["rr_top_reachable"],
        "rr_top_contact": context["rr_top_contact"],
        "rr_current_bearing": context["rr_current_bearing"],
        "rl_transfer_ready": context["rl_transfer_ready"],
        "rr_capture_recovery_allowed": context["rr_capture_recovery_allowed"],
        "fl_wheel_guidance_active": context["fl_wheel_guidance_active"],
        "fl_capture_pending": continuation["fl_capture_pending"],
        "fl_within_top_xy": fl_leg.get("within_top_xy"),
        "fl_front_distance_mm": (None if _number(fl_leg.get("front_distance_m")) is None
                                 else 1000. * float(fl_leg["front_distance_m"])),
        "fl_gap_mm": (None if _number(fl_leg.get("clearance_m")) is None
                      else 1000. * float(fl_leg["clearance_m"])),
        "body_position_w_m": body_position,
        "center_of_mass_position_w_m": center_of_mass,
        "rr_wheel_center_w_m": rr_center,
        "rr_wheel_center_body_relative_m": rr_center_body,
        "wheel_prebias_mapper_canonical_rad_s": native[8:],
        "wheel_final_canonical_rad_s": final[8:],
        "wheel_native_physical_rad_s": physical_wheels,
        "wheel_actual_canonical_rad_s": measured,
    }


def capture_rows(context: dict[str, Any], ledger: list[Any]) -> tuple[
        list[dict[str, Any]], list[dict[str, Any]]]:
    wanted = {item.sim_step for item in ledger}
    frames: dict[int, dict[str, Any]] = {}
    milestones: dict[str, dict[str, Any]] = {}
    count = 0
    dispatch_offset = None
    last_item: dict[str, Any] | None = None
    with (context["tick_path"].open("rb") as stream,
          context["native_tick_path"].open("rb") as native_stream):
        for count, raw in enumerate(stream, 1):
            require(raw.endswith(b"\n"), "capture-assist tick ledger has a partial final row")
            native_raw = native_stream.readline()
            require(native_raw.endswith(b"\n"),
                    "native tick audit is missing or has a partial final row")
            row = json.loads(raw)
            native_row = json.loads(native_raw)
            require(row.get("episode_physics_tick") == count and
                    math.isclose(float(row.get("sim_time_s")), count / HZ, abs_tol=1e-10),
                    "capture-assist tick ledger clock gap")
            item = frame_summary(row, native_row)
            if dispatch_offset is None:
                dispatch_offset = item["dispatch_to_episode_tick_offset"]
            require(item["dispatch_to_episode_tick_offset"] == dispatch_offset,
                    "dispatch/episode physical-clock offset changed within the run")
            expected_revision = (FEEDBACK_V2_REVISION if
                context["checkpoint"]["rr_capture_feedback_v2_present"] else None)
            require(item["rr_assist_feedback_revision"] == expected_revision,
                    "per-tick RR feedback revision differs from checkpoint lineage")

            def keep(name: str, condition: bool) -> None:
                if condition and name not in milestones:
                    milestones[name] = {"milestone": name, **item}

            owned = bool(item["rr_assist_owner_indices"])
            crossed = item["rr_front_edge_crossed"] is True
            travel = item["rr_assist_travel_used_deg"] or 0.0
            keep("FIRST_ASSIST_OWNERSHIP", owned)
            keep("FIRST_RR_FRONT_EDGE_CROSS", crossed)
            keep("FIRST_POST_CROSS_DESCENT", crossed and travel > 1e-9)
            for degrees in (5, 10, 15, 20):
                keep(f"HIP_DESCENT_MINUS_{degrees}_DEG", travel >= degrees - 1e-9)
            if count in wanted:
                frames[count] = item
            last_item = item
        require(native_stream.readline() == b"", "native tick audit extends beyond the episode")
    require(count == context["endpoint"],
            "capture-assist ledger does not cover the entire episode")
    require(set(frames) == wanted,
            "capture-assist ledger lacks exact encoded-frame endpoints")
    rows = [frames[item.sim_step] for item in ledger]
    require(last_item is not None, "capture-assist ledger is empty")
    milestones["TERMINAL"] = {"milestone": "TERMINAL", **last_item}
    order = ("FIRST_ASSIST_OWNERSHIP", "FIRST_RR_FRONT_EDGE_CROSS",
             "FIRST_POST_CROSS_DESCENT", "HIP_DESCENT_MINUS_5_DEG",
             "HIP_DESCENT_MINUS_10_DEG", "HIP_DESCENT_MINUS_15_DEG",
             "HIP_DESCENT_MINUS_20_DEG", "TERMINAL")
    if context["checkpoint"]["rr_capture_feedback_v2_present"]:
        require(set(milestones) == set(order),
                "feedback-v2 run lacks one of the eight declared RR milestones")
        return rows, [milestones[name] for name in order]
    return rows, [milestones[name] for name in order if name in milestones]


def outcome(manifest: dict[str, Any]) -> tuple[str, bool, str | None]:
    physical = manifest.get("physical_episode") or {}
    evaluation = physical.get("physical_task_evaluation") or {}
    success = (manifest.get("physical_task_success") is True and
               physical.get("task_success") is True and
               manifest.get("success_candidate") is True and
               manifest.get("diagnostic_only") is False and
               manifest.get("source_acceptance_error") is None)
    reason = (evaluation.get("termination_reason") or physical.get("termination_reason") or
              manifest.get("source_acceptance_error"))
    return ("SUCCESS" if success else "DIAGNOSTIC_FAILURE / INCOMPLETE"), success, reason


def _fmt(value: Any, digits: int = 2) -> str:
    return "N/A" if value is None else f"{float(value):+.{digits}f}"


def _wheel_line(prefix: str, values: list[float]) -> str:
    labels = ("FL", "FR", "RL", "RR")
    return prefix + "  ".join(f"{label} {_fmt(value, 3)}" for label, value in zip(labels, values))


def panel_lines(row: dict[str, Any], result: str, *, rr_reached: bool,
                feedback_v2: bool) -> list[str]:
    attempt = ("RR WINDOW REACHED | RL NOT REACHED" if rr_reached else
               "RR/RL NOT REACHED - P05 PREDECESSOR FAILURE")
    boundary = ("RR FEEDBACK window_peak_progress_v2 | SAME 410 + SAME v1 POLICY/KERNEL"
                if feedback_v2 else "389->410 CONTROL/OBS MIGRATION")
    return [
        f"PPO + FL/RR CAPTURE ASSISTS + INHERITED AUX | DET CP220544 | {result}",
        f"{attempt} | {boundary} | new PPO +0 / AUX +0",
        f"RR hip-only {row['rr_assist_mode']} active={row['rr_assist_active']} owner={row['rr_assist_owner_indices']} reason={row['rr_assist_reason']} | wheel intervention OFF | FL assist {row['fl_assist_mode']}",
        f"RR hip assist/final/actual deg {_fmt(row['rr_hip_assist_target_deg'])}/{_fmt(row['rr_hip_final_target_deg'])}/{_fmt(row['rr_hip_actual_deg'])} native-target {_fmt(row['rr_hip_native_target_rad'], 4)}rad | knee HOLD/final/actual {_fmt(row['rr_knee_hold_target_deg'])}/{_fmt(row['rr_knee_final_target_deg'])}/{_fmt(row['rr_knee_actual_deg'])} native {_fmt(row['rr_knee_native_target_rad'], 4)}rad",
        f"RR PRE(tick {row['rr_pre_observation_tick']}) gap {_fmt(row['rr_pre_gap_mm'])}mm {row['rr_pre_contact']} -> POST gap {_fmt(row['rr_post_gap_mm'])}mm {row['rr_post_contact']} | bearing={row['rr_current_bearing']} placed={row['rr_placed_history']} RL-ready={row['rl_transfer_ready']}",
        f"RR POST forces reaction/bearing/obstacle N {_fmt(row['rr_contact_reaction_force_n'])}/{_fmt(row['rr_bearing_force_n'])}/{_fmt(row['rr_obstacle_normal_force_n'])} | bearing_verified={row['rr_bearing_verified']} | FL pending/XY/frontmm={row['fl_capture_pending']}/{row['fl_within_top_xy']}/{_fmt(row['fl_front_distance_mm'])}",
        _wheel_line("wheel mapper pre-bias target, canonical axis: ", row["wheel_prebias_mapper_canonical_rad_s"]),
        _wheel_line("wheel FINAL target, canonical axis rad/s:    ", row["wheel_final_canonical_rad_s"]),
        _wheel_line("wheel actual PhysX native-axis target rad/s: ", row["wheel_native_physical_rad_s"]),
        _wheel_line("wheel measured canonical actual rad/s:       ", row["wheel_actual_canonical_rad_s"]),
        f"1x 15fps | t={row['time_s']:.3f}s tick={row['tick']} {row['phase']} | same-run measured evidence; assists are not PPO credit",
    ]


def rgba_panels(rows: list[dict[str, Any]], result: str, *, rr_reached: bool,
                feedback_v2: bool) -> Iterable[bytes]:
    font_path = Path("C:/Windows/Fonts/consola.ttf")
    require(font_path.is_file(), "overlay font is unavailable")
    fonts = [ImageFont.truetype(str(font_path), size)
             for size in (15, 12, 12, 12, 11, 11, 11, 11, 11, 11, 11)]
    for row in rows:
        lines = panel_lines(row, result, rr_reached=rr_reached,
                            feedback_v2=feedback_v2)
        image = Image.new("RGBA", (1280, 187), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 1279, 186), fill=(0, 0, 0, 180))
        for index, line in enumerate(lines):
            require(draw.textlength(line, font=fonts[index]) <= 1252,
                    "RR overlay line exceeds fixed top panel")
            draw.text((14, 3 + index * 16), line, font=fonts[index],
                      fill=(255, 255, 128, 255) if index == 0 else (255, 255, 255, 255))
        yield image.tobytes()


def run(command: list[str], *, input_bytes: Iterable[bytes] | None = None) -> None:
    if input_bytes is None:
        completed = subprocess.run(command, capture_output=True, text=True, errors="replace")
        require(completed.returncode == 0,
                f"command failed ({completed.returncode}): {completed.stderr[-2500:]}")
        return
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        require(process.stdin is not None, "ffmpeg panel pipe is unavailable")
        for block in input_bytes:
            require(process.poll() is None,
                    "ffmpeg exited before every overlay frame was written")
            process.stdin.write(block)
        process.stdin.close()
        error = process.stderr.read().decode(errors="replace") if process.stderr else ""
        require(process.wait() == 0, "overlay encode failed: " + error[-2500:])
    finally:
        if process.poll() is None:
            process.terminate()


def encode_full(candidate: dict[str, Any], rows: list[dict[str, Any]], output: Path,
                *, result: str, rr_reached: bool, feedback_v2: bool,
                ffmpeg: Path) -> dict[str, Any]:
    count = len(rows)
    command = [str(ffmpeg), "-hide_banner", "-nostdin", "-v", "error", "-n",
        "-threads", "2", "-filter_threads", "1", "-filter_complex_threads", "1",
        "-i", str(candidate["video"]), "-f", "rawvideo", "-pixel_format", "rgba",
        "-video_size", "1280x187", "-framerate", str(FPS), "-i", "pipe:0",
        "-filter_complex", "[0:v]setpts=PTS-STARTPTS[v];[1:v]format=rgba,setpts=PTS-STARTPTS[p];[v][p]overlay=0:0:shortest=1,format=yuv420p[out]",
        "-map", "[out]", "-an", "-frames:v", str(count), "-r", str(FPS),
        "-fps_mode", "cfr", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-threads", "2", "-movflags", "+faststart", str(output)]
    run(command, input_bytes=rgba_panels(rows, result, rr_reached=rr_reached,
                                         feedback_v2=feedback_v2))
    helper = shared()
    return {"output": str(output), "frame_count": count, "normal_speed": True,
        "no_intro_frames": True, "full_failure_tail_preserved": True,
        "top_overlay_no_canvas_growth": True,
        "validation": helper.validate_output(output, count, 1280, 720, ffmpeg),
        "previews": helper.preview(output, count, ffmpeg), "command": command}


def rr_window_reached(rows: list[dict[str, Any]]) -> bool:
    phases = {"P09", "P10", "P11", "P12", "P13"}
    return any(row["phase"] in phases or row["rr_lift_carry"] or
               row["rr_assist_mode"] != "WAIT" for row in rows)


def detail_plan(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if rr_window_reached(rows):
        phases = {"P09", "P10", "P11", "P12", "P13"}
        start = next(index for index, row in enumerate(rows)
                     if row["phase"] in phases or row["rr_lift_carry"] or
                     row["rr_assist_mode"] != "WAIT")
        return {"kind": "RR_CAPTURE_THROUGH_TERMINAL", "start": start,
            "end": len(rows),
            "filename": "CP220544_DET_RR_capture_to_RL_detail.mp4",
            "title": "DETAIL | CP220544 | RR feedback-v2 capture attempt through terminal | RL NOT REACHED",
            "requested_RR_detail_unavailable_reason": None}

    # The alternative is not an RR clip.  It requires actual P05 history-crossed
    # pending state plus current geometric rollback, all from this same episode.
    rollback = [index for index, row in enumerate(rows)
                if row["phase"] == "P05" and row["fl_capture_pending"] is True
                and row["fl_within_top_xy"] is False
                and row["fl_front_distance_mm"] is not None
                and row["fl_front_distance_mm"] < 0.]
    if not rollback:
        return None
    floor = max(0, len(rows) - 60 * FPS)
    in_tail = [index for index in rollback if index >= floor]
    anchor = in_tail[0] if in_tail else rollback[-1]
    start = (max(floor, anchor - 2 * FPS) if in_tail else
             max(0, anchor - 2 * FPS))
    # Prefer a compact 25-60 s truthful tail; never invent frames or delete an
    # internal stall.  A shorter real remainder is retained as-is.
    if len(rows) - start < 25 * FPS:
        start = max(0, len(rows) - 25 * FPS)
    return {"kind": "P05_RECROSS_FAILURE", "start": start, "end": len(rows),
        "filename": "CP220544_DET_P05_recross_failure_detail.mp4",
        "title": "DETAIL | CP220544 | P05 historical-cross pending / current-XY rollback | RR/RL NOT REACHED",
        "requested_RR_detail_unavailable_reason":
            "NO_P09_OR_RR_CAPTURE_OBSERVED__REAL_P05_PREDECESSOR_RECROSS_FAILURE_INSTEAD"}


def encode_detail(full: Path, rows: list[dict[str, Any]], destination: Path,
                  *, ffmpeg: Path) -> dict[str, Any]:
    plan = detail_plan(rows)
    if plan is None:
        return {"output": None, "omitted": True,
            "reason": "NO_RR_WINDOW_AND_NO_VERIFIED_P05_RECROSS_FAILURE_WINDOW",
            "requested_RR_detail_unavailable_reason":
                "NO_P09_OR_RR_CAPTURE_OBSERVED",
            "not_substituted_from_another_checkpoint": True,
            "full_source_preserved_at": str(full)}
    start, end = plan["start"], plan["end"]
    count = end - start
    output = destination / plan["filename"]
    title = plan["title"]
    video_filter = (f"trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS,"
        "drawbox=x=0:y=0:w=iw:h=20:color=black@0.92:t=fill,"
        "drawtext=fontfile='C\\:/Windows/Fonts/consola.ttf':"
        f"text='{title}':fontcolor=yellow:fontsize=14:x=14:y=2")
    command = [str(ffmpeg), "-hide_banner", "-nostdin", "-v", "error", "-n",
        "-threads", "2", "-filter_threads", "1", "-i", str(full), "-vf", video_filter,
        "-an", "-frames:v", str(count), "-r", str(FPS), "-fps_mode", "cfr",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-threads", "2", "-movflags", "+faststart", str(output)]
    run(command)
    helper = shared()
    return {"output": str(output), "kind": plan["kind"], "frame_count": count,
        "requested_RR_detail_unavailable_reason":
            plan["requested_RR_detail_unavailable_reason"],
        "source_frame_interval_half_open": [start, end],
        "actual_tick_endpoints": [rows[start]["tick"], rows[end - 1]["tick"]],
        "same_single_episode": True, "contiguous": True,
        "internal_stalls_removed": False, "full_failure_tail_preserved": True,
        "visible_title_overlay": title, "normal_speed": True,
        "validation": helper.validate_output(output, count, 1280, 720, ffmpeg),
        "previews": helper.preview(output, count, ffmpeg), "command": command}


def encode_pair(baseline: dict[str, Any], candidate: dict[str, Any], full: Path,
                output: Path, *, historical_version: str, feedback_v2: bool,
                ffmpeg: Path) -> dict[str, Any]:
    require(baseline["manifest"]["camera"] == candidate["manifest"]["camera"],
            "historical N and candidate camera definitions differ")
    _, b_ledger, _ = shared().checked_media(baseline, ffmpeg=ffmpeg)
    b_count, c_count = len(b_ledger), candidate["frame_count"]
    count = max(b_count, c_count)
    require(count <= MAX_FRAMES, "comparison exceeds 200 seconds")
    safe = "".join(ch if ch.isalnum() or ch in " ._-" else "_" for ch in historical_version)
    require(safe.strip() == historical_version and bool(safe.strip()),
            "historical N label contains unsafe characters")
    candidate_label = ("CP220544 | SAME v1 POLICY + RR FEEDBACK-v2 CONTROL | "
                       "FL/RR ASSISTS + AUX | +0 PPO" if feedback_v2 else
                       "CP220544 | PPO + FL/RR ASSISTS + INHERITED AUX | "
                       "RR wheel intervention OFF | +0 new PPO")
    labels = (f"HISTORICAL N_REF (NOT FRESH B) | {safe}", candidate_label)
    counts = (b_count, c_count)
    font = "C\\:/Windows/Fonts/arial.ttf"
    filters = []
    for index, (label, frames) in enumerate(zip(labels, counts)):
        filters.append(f"[{index}:v]setpts=PTS-STARTPTS,scale=960:540,pad=960:610:0:70:black,"
            f"tpad=stop_mode=clone:stop=-1,setpts=N/({FPS}*TB),"
            f"drawtext=fontfile='{font}':text='{label}':fontcolor=white:fontsize=20:x=14:y=13,"
            f"drawtext=fontfile='{font}':text='RUN ENDED - FROZEN, NOT NEW PHYSICS':fontcolor=yellow:fontsize=20:x=14:y=44:enable='gte(n,{frames})'[v{index}]")
    filters.append("[v0][v1]hstack=inputs=2:shortest=1,format=yuv420p[out]")
    command = [str(ffmpeg), "-hide_banner", "-nostdin", "-v", "error", "-n",
        "-threads", "2", "-filter_complex_threads", "1", "-i", str(baseline["video"]),
        "-i", str(full), "-filter_complex", ";".join(filters), "-map", "[out]", "-an",
        "-frames:v", str(count), "-r", str(FPS), "-fps_mode", "cfr", "-c:v", "libx264",
        "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-threads", "2",
        "-movflags", "+faststart", str(output)]
    run(command)
    helper = shared()
    return {"output": str(output), "frame_count": count,
        "historical_N_version": historical_version,
        "historical_N_source": str(baseline["source"]),
        "historical_N_is_fresh_B": False, "same_camera": True,
        "alignment": "same elapsed natural-P01 origin; no phase/event retiming",
        "same_controller_or_runtime_claimed": False,
        "freeze_added_frames": {"historical_N": count - b_count, "candidate": count - c_count},
        "freeze_is_physical_evidence": False, "normal_speed": True,
        "validation": helper.validate_output(output, count, 1920, 610, ffmpeg),
        "previews": helper.preview(output, count, ffmpeg), "command": command}


def response_markdown(rows: list[dict[str, Any]]) -> str:
    lines = ["# RR feedback-v2 selected real ticks", "",
        "Eight exact decision-grid milestones from this sealed run; no interpolation. `pre` is the adjacent source observation and `post` is the measured episode tick after dispatch.", "",
        "| milestone | pre/post tick | phase | RR assist | travel deg | hip final/actual/native-rad | knee final/actual/native-rad | gap/front mm | contact/placed | body xyz m | COM xyz m | RR center xyz m | wheels source | wheels target | wheels actual |",
        "|:---|:---|:---:|:---|---:|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|"]
    for row in rows:
        vector = lambda key: "/".join(_fmt(value, 3) for value in row[key])
        xyz = lambda key: "/".join(_fmt(value, 4) for value in row[key])
        lines.append(f"| {row['milestone']} | {row['rr_pre_observation_tick']}/{row['rr_post_actual_episode_tick']} | {row['phase']} | {row['rr_assist_mode']} | "
            f"{_fmt(row['rr_assist_travel_used_deg'])} | "
            f"{_fmt(row['rr_hip_final_target_deg'])}/{_fmt(row['rr_hip_actual_deg'])}/{_fmt(row['rr_hip_native_target_rad'], 4)} | "
            f"{_fmt(row['rr_knee_final_target_deg'])}/{_fmt(row['rr_knee_actual_deg'])}/{_fmt(row['rr_knee_native_target_rad'], 4)} | "
            f"{_fmt(row['rr_post_gap_mm'])}/{_fmt(row['rr_front_distance_mm'])} | {row['rr_post_contact']}/{row['rr_placed_history']} | "
            f"{xyz('body_position_w_m')} | {xyz('center_of_mass_position_w_m')} | {xyz('rr_wheel_center_w_m')} | "
            f"{vector('wheel_prebias_mapper_canonical_rad_s')} | {vector('wheel_final_canonical_rad_s')} | {vector('wheel_actual_canonical_rad_s')} |")
    lines.extend(["", "The 45.703 mm terminal gap versus 63.599 mm in sealed a546 is a descriptive cross-run, cross-control-revision comparison. The 17.896 mm difference is not attributed wholly or independently to RR hip feedback.", "",
        "RR current contact and placed history remain distinct. Assist/AUX activity is not PPO credit or task success.", ""])
    return "\n".join(lines)


def export(source: Path, destination: Path, historical_n_source: Path,
           historical_version: str) -> dict[str, Any]:
    destination = Path(destination).resolve()
    require(destination.is_relative_to(OUTPUT_ROOT.resolve()),
            "destination must remain in the isolated RR output tree")
    require(not destination.exists(), "destination already exists; exports are immutable")
    candidate = sealed_source(source, candidate=True)
    baseline = sealed_source(historical_n_source, candidate=False)
    helper = shared()
    ffmpeg = helper.find_ffmpeg(candidate["capture"].get("full_decode", {}).get("ffmpeg_path"))
    _, ledger, source_validation = helper.checked_media(candidate, ffmpeg=ffmpeg)
    rows, selected = capture_rows(candidate, ledger)
    rr_reached = rr_window_reached(rows)
    result_label, success, termination = outcome(candidate["manifest"])
    checkpoint = candidate["checkpoint"]
    require(checkpoint["decisions"] == EXPECTED_CHECKPOINT_DECISIONS,
            "source was not loaded from CP220544")
    feedback_v2 = checkpoint["rr_capture_feedback_v2_present"] is True
    preflight_detail = detail_plan(rows)
    if feedback_v2:
        require(rr_reached is True and success is False and
                preflight_detail is not None and
                preflight_detail["kind"] == "RR_CAPTURE_THROUGH_TERMINAL" and
                rows[-1]["phase"] == "P09" and
                rows[-1]["rr_placed_history"] is not True,
                "sealed feedback-v2 run is not the declared incomplete P09 RR attempt")
    else:
        require(rr_reached is False and preflight_detail is not None and
                preflight_detail["kind"] == "P05_RECROSS_FAILURE",
                "sealed initial RR410 run does not match the declared P05 predecessor failure")
    print("PREFLIGHT_COMPLETE_SAFE_TO_EDIT_RUNTIME", flush=True)

    names = {
        "full": "CP220544_DET_full_RR_RL_attempt.mp4",
        "pair": "N_vs_CP220544_DET_same_camera.mp4",
    }
    destination.mkdir(parents=True)
    full = encode_full(candidate, rows, destination / names["full"],
                       result=result_label, rr_reached=rr_reached,
                       feedback_v2=feedback_v2, ffmpeg=ffmpeg)
    print("FULL_PLAYABLE_VALIDATED " + full["output"], flush=True)
    detail = encode_detail(Path(full["output"]), rows, destination, ffmpeg=ffmpeg)
    pair_candidate = {**candidate, "frame_count": len(rows)}
    pair = encode_pair(baseline, pair_candidate, Path(full["output"]),
                       destination / names["pair"], historical_version=historical_version,
                       feedback_v2=feedback_v2, ffmpeg=ffmpeg)

    response_json = destination / "rr_capture_selected_response.json"
    response_md = destination / "rr_capture_selected_response.md"
    write_new_json(response_json, {
        "schema": "wlr50_clean.rr_capture_video_selected_response.v2",
        "source": str(candidate["source"]),
        "source_capture_assist_ticks_sha256": sha256(candidate["tick_path"]),
        "source_native_tick_audit_sha256": sha256(candidate["native_tick_path"]),
        "selection": "eight exact decision-grid rows: first ownership, first RR cross, first post-cross descent, first -5/-10/-15/-20deg travel thresholds, terminal; no interpolation",
        "pre_context_and_post_actual_tick_are_explicit": True,
        "terminal_gap_cross_run_context": {
            "this_feedback_v2_run_mm": selected[-1]["rr_post_gap_mm"],
            "sealed_a546_run_mm": 63.599,
            "difference_mm_descriptive_only": (None if selected[-1]["rr_post_gap_mm"] is None
                else 63.599 - selected[-1]["rr_post_gap_mm"]),
            "runs_have_different_control_revisions": True,
            "difference_wholly_attributed_to_RR_hip": False,
            "causal_counterfactual_claimed": False,
        },
        "rr_capture_or_RL_window_reached": rr_reached,
        "requested_RR_detail_unavailable_reason":
            detail.get("requested_RR_detail_unavailable_reason"),
        "rows": selected,
        "full_per_tick_ledger": str(candidate["tick_path"]),
    })
    write_new_text(response_md, response_markdown(selected))

    receipt = {
        "schema": "wlr50_clean.rr_capture_video_export.v2",
        "source": str(candidate["source"]),
        "source_manifest": str(candidate["manifest_path"]),
        "source_manifest_sha256": sha256(candidate["manifest_path"]),
        "source_run_manifest_sha256": sha256(candidate["run_manifest_path"]),
        "source_video_validation": {key: source_validation.get(key) for key in (
            "sha256", "bytes", "valid", "full_decode", "frame_count", "fps", "resolution",
            "duration_s", "frame_pts_sha256", "decoded_frame_checksums_sha256")},
        "checkpoint": checkpoint,
        "experiment_id": EXPERIMENT,
        "control_method": CONTROL_METHOD,
        "policy_sampling_mode": "deterministic_conditional_mean",
        "policy_contract_version": POLICY_VERSION,
        "policy_kernel_changed_by_feedback_v2": False,
        "rr_capture_control_revision": checkpoint["rr_capture_control_revision"],
        "rr_capture_feedback_v2": checkpoint.get("rr_capture_feedback_v2"),
        "physical_result": result_label,
        "physical_task_success": success,
        "termination_reason": termination,
        "source_acceptance_error": candidate["manifest"].get("source_acceptance_error"),
        "rr_capture_or_RL_window_reached": rr_reached,
        "attempt_classification": ("RR_CAPTURE_ATTEMPT__RL_NOT_REACHED" if rr_reached else
                                   "P05_PREDECESSOR_FAILURE__RR_RL_NOT_REACHED"),
        "requested_RR_detail_unavailable_reason":
            detail.get("requested_RR_detail_unavailable_reason"),
        "rr_capture_assist": {
            "mode": RR_ASSIST_MODE,
            "owner_indices": [6, 7],
            "owner_names": [SERVO_ORDER[6], SERVO_ORDER[7]],
            "wheel_intervention": "off",
            "is_policy_learning": False,
        },
        "wheel_axis_mapping": {
            "canonical_order": list(WHEEL_ORDER),
            "short_labels": ["FL", "FR", "RL", "RR"],
            "physical_forward_sign": {name: WHEEL_FORWARD_SIGN[name] for name in WHEEL_ORDER},
            "prebias_mapper_target_source": "dispatch.native_drive_target_full12[8:12] in canonical axes; not a PhysX-native-axis target",
            "final_canonical_target_source": "dispatch.drive_target_full12[8:12]",
            "actual_native_physical_target_source":
                "native_tick_audit.native_audit.actual_native_targets.wheel_velocity_rad_s",
            "measured_canonical_actual_source": "actual_full12[8:12], cross-checked against wheels.*.velocity_rad_s",
        },
        "full_episode_continuous": True,
        "full_failure_tail_preserved": True,
        "normal_speed": True,
        "single_episode": True,
        "stitched": False,
        "speed_modified": False,
        "extra_intro_frames": 0,
        "full": full,
        "detail": detail,
        "historical_N_comparison": pair,
        "selected_response_json": str(response_json),
        "selected_response_json_sha256": sha256(response_json),
        "selected_response_markdown": str(response_md),
        "selected_response_markdown_sha256": sha256(response_md),
        "claims": {
            "rr_capture_assist_is_policy_learning": False,
            "fl_capture_assist_is_policy_learning": False,
            "inherited_auxiliary_is_PPO": False,
            "rr_transfer_branch_has_new_PPO_updates": False,
            "feedback_v2_migration_has_new_PPO_updates": False,
            "feedback_v2_is_new_policy_kernel": False,
            "rr_wheel_intervention_active": False,
            "reachability_is_contact_or_bearing": False,
            "historical_N_is_fresh_same_controller_B": False,
            "failure_video_is_success": False,
            "migration_is_same_MDP": False,
            "P05_failure_detail_is_RR_capture_detail": False,
            "missing_RR_window_is_substituted_from_another_run": False,
            "terminal_gap_difference_vs_a546_is_independent_hip_causality": False,
        },
    }
    write_new_json(destination / "export_receipt.json", receipt)
    print(json.dumps({"physical_result": result_label,
        "full": full["output"], "detail": detail.get("output"),
        "comparison": pair["output"],
        "receipt": str(destination / "export_receipt.json")}, indent=2))
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--historical-n-source", type=Path, default=DEFAULT_HISTORICAL_N)
    parser.add_argument("--historical-n-version", default=DEFAULT_HISTORICAL_VERSION)
    args = parser.parse_args()
    export(args.source, args.destination, args.historical_n_source,
           args.historical_n_version)


if __name__ == "__main__":
    main()
