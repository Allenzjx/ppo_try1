"""Derive a formal local-RR video export from one sealed direct manifest.

The default mode only validates immutable metadata and prints the exact child
command.  ``--execute`` is required to invoke the existing media exporter.
Torch, PXR and Isaac are never imported.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
_COORDINATE_SPEC = importlib.util.spec_from_file_location(
    '_rr_capture_coordinate_video', OUTPUT / 'coordinate_video_validator.py')
if _COORDINATE_SPEC is None or _COORDINATE_SPEC.loader is None:
    raise RuntimeError('coordinate video validator cannot be loaded')
COORDINATES = importlib.util.module_from_spec(_COORDINATE_SPEC)
_COORDINATE_SPEC.loader.exec_module(COORDINATES)
EXPORTER = OUTPUT / "export_rr_capture_first_video.py"
EXPERIMENT = "ppo_rr_capture_first_cp225280_v1"
SOURCE_SCHEMA = "wlr50_clean.frozen_prior_rr_capture_video.v1"
CHECKPOINT_SCHEMA = "wlr50_clean.frozen_prior_rr_capture_checkpoint.v1"
ORIGIN = 225280
VERSION_FAMILIES = {
    "v1": {
        "source_schema": SOURCE_SCHEMA,
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "request_schema": "wlr50_clean.actual_rr_capture_local_request.v1",
        "task_schema": "wlr50_clean.rr_capture_local_task.v1",
        "actor_policy": "frozen_cp225280_rr_capture_local_history_v1",
        "profile_policy": "frozen_cp225280_raw_head_plus_local447_v1",
        "profile_schema": "wlr50_clean.rr_capture_local_contract.v1",
        "observation_dimension": 447,
        "observation_layout": "role439_rr_capture_local_v1",
        "capture_features": 8,
    },
    "v2": {
        "source_schema": "wlr50_clean.frozen_prior_rr_capture_video.v2",
        "checkpoint_schema": "wlr50_clean.frozen_prior_rr_capture_checkpoint.v2",
        "request_schema": "wlr50_clean.actual_rr_capture_local_request.v2",
        "task_schema": "wlr50_clean.rr_capture_local_task.v2",
        "actor_policy": "frozen_cp225280_rr_capture_local_history_v2",
        "profile_policy": "frozen_cp225280_raw_head_plus_local448_lineage_v2",
        "profile_schema": "wlr50_clean.rr_capture_local_contract.v2",
        "observation_dimension": 448,
        "observation_layout": "role439_rr_capture_local_v2",
        "capture_features": 9,
    },
}
LOCAL_FIELDS_V1 = ["active", "activation_age_norm", "entry_rr_hip_norm", "entry_rr_knee_norm",
    "entry_gap_norm", "current_top_contact", "current_top_bearing", "capture_hold_progress"]
ELIGIBILITY_FIELD = "current_attempt_capture_eligible"
SOURCE_DISPATCH_V2 = "rr_local_defer_p09_late_and_new_p12_until_terminal_v2"
SOURCE_TRACKING_INHERITANCE = "pending_source_tracking_inheritance_v1"
AUX_TRAINING_LABEL = "finite RR AUX training lineage / rear helpers OFF"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked_hash(value: Any, label: str) -> str:
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            f"{label} must be lowercase SHA-256")
    return value


def version_family(schema: Any, *, kind: str = "source_schema") -> dict[str, Any]:
    matches = [(name, value) for name, value in VERSION_FAMILIES.items()
               if value.get(kind) == schema]
    require(len(matches) == 1, "unknown or ambiguous RR-capture version/schema")
    name, version = matches[0]
    return dict(version, version=name)


def source_tracking_disclosure(version: str, revision: Any) -> dict[str, Any]:
    """Name the recorded control revision, never infer it from a CP number/head."""
    require(version in ("v1", "v2"), "unknown tracking version family")
    require(revision is None or
            (version == "v2" and revision == SOURCE_TRACKING_INHERITANCE),
            "unknown or incompatible source tracking owner revision")
    if revision is not None:
        label = "pending-source inherited"
        description = ("Pending P09 late inherits tracking responsibility from the previous "
            "consumed source sample, not the deferred late group. A wheel-only stop creates "
            "no new servo tracking. This is a source-control correction, not learned descent.")
    elif version == "v2":
        label = "legacy v2 late-derived"
        description = ("Sealed v2 without the pending-source tracking inheritance revision; "
            "not relabelled as the corrected source controller.")
    else:
        label = "original v1"
        description = "Sealed original v1 source controller; no v2 deferred-source revision."
    return {"source_tracking_owner_revision": revision, "hud_label": label,
        "description": description,
        "training_count_scope": "Exact checkpoint counters retained; no tracking-revision-local counts inferred."}


def auxiliary_training_disclosure(source: dict[str, Any], checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Check saved training provenance, not an evaluation-time intervention."""
    steps = (checkpoint.get("counts") or {}).get("auxiliary_updates", 0)
    require(type(steps) is int and steps >= 0, "invalid AUX optimizer step count")
    control = source.get("control_contributions") or {}
    events = checkpoint.get("local_auxiliary_events", [])
    source_events = control.get("local_auxiliary_events", [])
    require(isinstance(events, list) and isinstance(source_events, list) and
            json.dumps(events, sort_keys=True, allow_nan=False) ==
            json.dumps(source_events, sort_keys=True, allow_nan=False),
            "source/checkpoint local AUX training events differ")
    require(all(isinstance(event, dict) and
                type(event.get("actual_optimizer_steps")) is int and
                event["actual_optimizer_steps"] > 0 for event in events) and
            sum(event["actual_optimizer_steps"] for event in events) == steps,
            "AUX training event steps do not match checkpoint counts")
    source_steps = control.get("local_auxiliary_optimizer_steps", 0)
    eval_steps = control.get("AUX_updates_during_evaluation", 0)
    require(type(source_steps) is int and source_steps == steps and
            type(eval_steps) is int and eval_steps == 0 and
            (steps == 0 or "AUX_updates_during_evaluation" in control),
            "source AUX training/evaluation step accounting differs")
    return {"local_auxiliary_events": copy.deepcopy(events),
        "actual_training_optimizer_steps": steps, "AUX_updates_during_evaluation": 0,
        "label": AUX_TRAINING_LABEL if steps else None,
        "realtime_AUX_assist": False,
        "pure_PPO_training_claimed": False if steps else None}


def validate_version_bindings(source: dict[str, Any], checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Pure metadata contract; does not infer compatibility from dimensions alone."""
    version = version_family(source.get("schema"))
    control = source.get("control_contributions") or {}
    runtime = checkpoint.get("runtime_contract") or {}
    profile = runtime.get("local_contract") or {}
    actor = (checkpoint.get("runner_config") or {}).get("actor") or {}
    expected_features = LOCAL_FIELDS_V1 + ([ELIGIBILITY_FIELD] if version["version"] == "v2" else [])
    require(checkpoint.get("schema") == version["checkpoint_schema"] and
            control.get("policy_version") == version["profile_policy"] and
            control.get("observation_dimension") == version["observation_dimension"] and
            profile.get("version") == version["profile_policy"] and
            profile.get("schema") == version["profile_schema"] and
            profile.get("observation_dimension") == version["observation_dimension"] and
            profile.get("local_features") == expected_features and
            actor.get("observation_layout") == version["observation_layout"] and
            actor.get("legacy447_migration_only", False) is False,
            "source/checkpoint/profile/actor versions are mixed or legacy migration-only")
    task = source.get("local_task") or {}
    require(task.get("schema") == version["task_schema"],
            "source local-task schema differs from its version family")
    if version["version"] == "v2":
        require(control.get("capture_source_dispatch") == SOURCE_DISPATCH_V2 and
                profile.get("capture_source_dispatch") == SOURCE_DISPATCH_V2,
                "v2 source dispatch rule is absent or differs from its runtime profile")
        context = task.get("obs9")
        eligible = (task.get("metrics") or {}).get(ELIGIBILITY_FIELD)
        require(isinstance(context, list) and len(context) == 9 and
                type(eligible) is bool and type(context[8]) in (int, float) and
                context[8] == float(eligible),
                "v2 source endpoint obs9/current-attempt eligibility is inconsistent")
        counts = checkpoint.get("counts") or {}
        require(all(type(counts.get(key)) is int and type(counts.get(total)) is int and
                    0 <= counts[key] <= counts[total]
                    for key, total in (("task_v2_policy_decisions", "local_policy_decisions"),
                                       ("task_v2_ppo_updates", "local_ppo_updates"))),
                "v2 checkpoint must separately count its post-migration decisions/updates")
    else:
        require(control.get("capture_source_dispatch") is None and
                profile.get("capture_source_dispatch") is None,
                "v1 source may not silently use the v2 dispatch rule")
    tracking_revision = profile.get("source_tracking_owner_revision")
    require(control.get("source_tracking_owner_revision") == tracking_revision,
            "source/checkpoint source tracking owner revision differs")
    version["source_tracking"] = source_tracking_disclosure(version["version"], tracking_revision)
    version["auxiliary_training"] = auxiliary_training_disclosure(source, checkpoint)
    version["mean_coordinates"] = COORDINATES.validate_coordinate_video_binding(source, checkpoint)
    return version


def build_plan(source: Path, destination: Path) -> dict[str, Any]:
    source = source.resolve(strict=True)
    experiment_root = (ROOT / "runs" / EXPERIMENT).resolve(strict=True)
    require(source.name == "source" and source.is_relative_to(experiment_root),
            "formal source is outside the isolated experiment")
    source_manifest_path = source / "source_manifest.json"
    run_manifest_path = source.parent / "run_manifest.json"
    source_manifest = read_json(source_manifest_path)
    run_manifest = read_json(run_manifest_path)
    require(run_manifest.get("lifecycle") == "COMPLETE" and
            run_manifest.get("result") == source_manifest,
            "formal source parent run is not sealed COMPLETE")
    version = version_family(source_manifest.get("schema"))
    require(source_manifest.get("mode") == "DETERMINISTIC_COMPOSITE_POLICY" and
            source_manifest.get("diagnostic_intervention") is False and
            (source_manifest.get("control_contributions") or {}).get(
                "diagnostic_only_RR_joint_override") is False and
            source_manifest.get("checkpoint_model_unchanged") is True and
            source_manifest.get("PPO_updates") == 0,
            "formal wrapper rejects diagnostic, mutated, or update-during-eval sources")
    require(run_manifest.get("mode") == "eval",
            "formal source parent run mode is not eval")

    proof = source_manifest.get("checkpoint_load_provenance") or {}
    require(proof.get("strict_actual_composite_load_verified") is True and
            proof.get("actor_critic_optimizer_hashes_verified") is True,
            "formal source lacks actual composite reload proof")
    checkpoint = Path(proof.get("checkpoint", "")).resolve(strict=True)
    sidecar = Path(proof.get("manifest", "")).resolve(strict=True)
    require(sidecar == checkpoint.with_name(checkpoint.stem + "_manifest.json"),
            "checkpoint provenance names a non-paired sidecar")
    checkpoint_sha = checked_hash(proof.get("checkpoint_sha256"), "checkpoint SHA")
    sidecar_sha = checked_hash(proof.get("manifest_sha256"), "checkpoint sidecar SHA")
    require(sha256(checkpoint) == checkpoint_sha and sha256(sidecar) == sidecar_sha,
            "actual formal checkpoint or sidecar differs from load provenance")
    metadata = read_json(sidecar)
    counts = metadata.get("counts") or {}
    count_keys = ("local_policy_decisions", "local_ppo_updates", "local_optimizer_steps")
    require(metadata.get("schema") == version["checkpoint_schema"] and
            metadata.get("checkpoint_sha256") == checkpoint_sha and
            metadata.get("save_load_round_trip") is True and
            all(type(counts.get(key)) is int and counts[key] >= 0 for key in count_keys),
            "formal checkpoint sidecar schema/counts/round-trip is invalid")
    version = validate_version_bindings(source_manifest, metadata)
    require((source_manifest.get("control_contributions") or {}).get(
                "local_branch_counters") == counts and
            source_manifest.get("counts") == counts,
            "formal source did not evaluate the sidecar's exact cumulative counts")
    runtime = metadata.get("runtime_contract") or {}
    expected_head = runtime.get("source_git_commit")
    require(isinstance(expected_head, str) and
            re.fullmatch(r"[0-9a-f]{40}", expected_head) is not None and
            source_manifest.get("runtime_contract") == runtime,
            "source/checkpoint runtime contract differs")

    destination = destination.resolve()
    require(destination.is_relative_to(OUTPUT.resolve()) and not destination.exists(),
            "destination must be a new directory below the isolated output namespace")
    local = counts["local_policy_decisions"]
    display_step = ORIGIN + local
    command = [sys.executable, str(EXPORTER),
        "--source", str(source), "--destination", str(destination),
        "--source-manifest-sha256", sha256(source_manifest_path),
        "--source-run-manifest-sha256", sha256(run_manifest_path),
        "--expected-head", expected_head,
        "--checkpoint", str(checkpoint), "--checkpoint-sha256", checkpoint_sha,
        "--checkpoint-manifest-sha256", sidecar_sha,
        "--expected-local-policy-decisions", str(local),
        "--expected-local-ppo-updates", str(counts["local_ppo_updates"]),
        "--expected-local-optimizer-steps", str(counts["local_optimizer_steps"])]
    return {"schema": "wlr50_clean.formal_rr_capture_export_plan." + version["version"],
        "source": str(source), "destination": str(destination),
        "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha,
        "checkpoint_manifest": str(sidecar), "checkpoint_manifest_sha256": sidecar_sha,
        "version_family": version,
        "source_tracking_owner_revision": version["source_tracking"]["source_tracking_owner_revision"],
        "source_tracking": version["source_tracking"],
        "auxiliary_training": version["auxiliary_training"],
        "mean_coordinates": version["mean_coordinates"],
        "runtime_head": expected_head, "counts": counts,
        "display_id": f"CP{display_step}_local{local:06d}",
        "numeric_step_collision_note": (
            "isolated local-RR branch identity; not the historical soft-KL checkpoint "
            f"that may share numeric CP{display_step}"),
        "diagnostic": False, "formal_deterministic_result_claimed_before_export": False,
        "command": command}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = build_plan(args.source, args.destination)
    print(json.dumps(plan, indent=2), flush=True)
    if args.execute:
        subprocess.run(plan["command"], check=True)


if __name__ == "__main__":
    main()
