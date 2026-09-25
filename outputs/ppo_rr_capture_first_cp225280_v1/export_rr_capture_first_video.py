"""Export one sealed frozen-CP225280 + local-RR evaluation.

This outputs-only adapter reads the direct recorder produced by
``semantic_rr_capture_local``.  It deliberately does not reuse or fabricate
the old semantic-video ``official_load`` provenance.  The composite
checkpoint, its sidecar, the direct source manifest and the parent run
manifest are all supplied by hash.  Torch, Isaac and PXR are never imported.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = Path(__file__).resolve().parent
MEDIA_PATH = (ROOT / "outputs" / "ppo_rr_rl_timing_policy_learning_v1" /
              "export_policy_rear_no_assist_video.py")

EXPERIMENT = "ppo_rr_capture_first_cp225280_v1"
SOURCE_SCHEMA = "wlr50_clean.frozen_prior_rr_capture_video.v1"
CHECKPOINT_SCHEMA = "wlr50_clean.frozen_prior_rr_capture_checkpoint.v1"
POLICY_VERSION = "frozen_cp225280_raw_head_plus_local447_v1"
FL_ASSIST_MODE = "p05_hip_only_continuation_v1"
PRIOR_HEAD = "f6d1d2df8d87d5f3eaaefc2254adb5f81fc52e2b"
PRIOR_SHA = "fbe28e3718a5796afc2271e7b10aa04017369593b5c62e0bde1973a20f68032f"
PRIOR_MANIFEST_SHA = "f501b7a735c0aaa42be00114e09f80d46f7f009d7717aa416fc30ad3cc837d6d"
ORIGINAL_422_SHA = "21897a4b2d2a85f01092c187c1b1ac824d8e61b01edaf732a6385ca951ab1dcf"
COUNTER_ORIGIN = 225280

REFERENCE_SOURCE = (ROOT / "runs" / "ppo_rr_rl_timing_policy_learning_v1" /
    "video_eval" / "validation" /
    "20260924T0214316632402Z_g49eb23163a6e_530cc3c96d834dc3ba84b0b19d8db4f1" /
    "source")
REFERENCE_SOURCE_MANIFEST_SHA = "6414be0c81b091effc2b07fbc8287d0b4e4866d18a4d83ac4b095e3f7bc69310"
REFERENCE_RUN_MANIFEST_SHA = "0b745890596accf262e0ffebc6b90c46ac4222cc8c9739beb8e4a135568872cc"
REFERENCE_VIDEO_SHA = "408a50761b420d3eeb5fa70748cbbcab8da63911671b641a4591f37ae83b64db"
REFERENCE_LEDGER_SHA = "2b7b8815a488be4f1b03eb77ad929c5eb4f40d0f80aa705cf76352fd69b6b984"


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("validated media helper cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


MEDIA = _load(MEDIA_PATH, "_rr_capture_first_media")
VERSIONS = _load(OUTPUT_ROOT / "export_formal_from_sealed.py", "_rr_capture_first_versions")
require = MEDIA.require


def _vector(value: Any, size: int, label: str) -> list[float]:
    require(isinstance(value, (list, tuple)) and len(value) == size and
            all(type(item) in (int, float) and math.isfinite(float(item))
                for item in value), label + " must be finite Full" + str(size))
    return [float(item) for item in value]


def _record(source: Path, manifest: dict[str, Any], name: str) -> Path:
    record = (manifest.get("sealed_files") or {}).get(name)
    require(isinstance(record, dict) and set(record) == {"path", "sha256", "bytes"},
            "source manifest lacks sealed file: " + name)
    path = Path(record["path"]).resolve(strict=True)
    expected = (source / name).resolve(strict=True)
    require(path == expected and MEDIA.checked_sha(record["sha256"], name) ==
            MEDIA.sha256(path) and record["bytes"] == path.stat().st_size,
            "sealed source file changed: " + name)
    return path


def _checkpoint_identity(manifest: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    proof = manifest.get("checkpoint_load_provenance") or {}
    require(proof.get("strict_actual_composite_load_verified") is True and
            proof.get("actor_critic_optimizer_hashes_verified") is True,
            "source lacks the actual strict composite reload proof")
    checkpoint = Path(proof.get("checkpoint", "")).resolve(strict=True)
    sidecar = Path(proof.get("manifest", "")).resolve(strict=True)
    expected = Path(args.checkpoint).resolve(strict=True)
    expected_sidecar = expected.with_name(expected.stem + "_manifest.json").resolve(strict=True)
    checkpoint_sha = MEDIA.checked_sha(args.checkpoint_sha256, "--checkpoint-sha256")
    sidecar_sha = MEDIA.checked_sha(args.checkpoint_manifest_sha256,
                                    "--checkpoint-manifest-sha256")
    require(checkpoint == expected and sidecar == expected_sidecar and
            MEDIA.sha256(checkpoint) == checkpoint_sha == proof.get("checkpoint_sha256") and
            MEDIA.sha256(sidecar) == sidecar_sha == proof.get("manifest_sha256"),
            "source composite checkpoint differs from explicit immutable bytes")
    metadata = MEDIA.read_json(sidecar)
    counts = metadata.get("counts") or {}
    expected_counts = {
        "local_policy_decisions": args.expected_local_policy_decisions,
        "local_ppo_updates": args.expected_local_ppo_updates,
        "local_optimizer_steps": args.expected_local_optimizer_steps,
    }
    version = VERSIONS.validate_version_bindings(manifest, metadata)
    require(metadata.get("schema") == version["checkpoint_schema"] and
            metadata.get("checkpoint_sha256") == checkpoint_sha and
            Path(metadata.get("checkpoint", "")).resolve() == checkpoint and
            metadata.get("save_load_round_trip") is True and
            all(counts.get(key) == value and type(value) is int and value >= 0
                for key, value in expected_counts.items()),
            "composite checkpoint schema/path/counters differ")
    runtime = metadata.get("runtime_contract") or {}
    expected_head = MEDIA.checked_head(args.expected_head)
    require(runtime.get("experiment_id") == EXPERIMENT and
            runtime.get("source_git_commit") == expected_head and
            manifest.get("runtime_contract") == runtime,
            "checkpoint/source runtime differs from the isolated experiment")
    prior = metadata.get("prior") or {}
    require(prior.get("path") and prior.get("sha256") == PRIOR_SHA and
            prior.get("manifest_sha256") == PRIOR_MANIFEST_SHA and
            prior.get("original422_sha256") == ORIGINAL_422_SHA and
            prior.get("original422_tensor_identity_verified") is True and
            prior.get("prior_optimizer_loaded") is False and
            prior.get("prior_weights_fully_frozen") is True,
            "composite checkpoint does not bind the frozen CP225280 prior")
    control = manifest.get("control_contributions") or {}
    require(control.get("policy_version") == version["profile_policy"] and
            control.get("observation_dimension") == version["observation_dimension"] and
            control.get("local_branch_counters") == counts and
            control.get("frozen_prior") == prior and
            control.get("FL_capture_assist_mode") == FL_ASSIST_MODE and
            control.get("rear_owner_projection") is False and
            control.get("rr_capture_assist_mode") is None and
            control.get("nominal_geometry_advisory") is None and
            control.get("rr_capture_wheel_mode") == "off",
            "source control-contribution labels differ from the composite checkpoint")
    display_step = COUNTER_ORIGIN + counts["local_policy_decisions"]
    return {"checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha,
        "manifest": str(sidecar), "manifest_sha256": sidecar_sha,
        "runtime_head": expected_head, "counts": counts,
        "display_step": display_step, "prior": prior,
        "policy_version": version["profile_policy"], "version_family": version,
        "source_tracking_owner_revision": version["source_tracking"]["source_tracking_owner_revision"],
        "source_tracking": version["source_tracking"],
        "auxiliary_training": version["auxiliary_training"],
        "diagnostic": manifest.get("diagnostic_intervention") is True}


def _time_receipt(manifest: dict[str, Any]) -> None:
    ticks = manifest.get("actual_ticks")
    receipt = manifest.get("time_receipt") or {}
    require(type(ticks) is int and 0 < ticks <= 24000 and
            receipt.get("schema") == "wlr50_clean.task_interval_video_window.v1" and
            receipt.get("experiment_id") == EXPERIMENT and
            receipt.get("endpoint_episode_tick") == ticks and
            receipt.get("frame_count") == (ticks + 7) // 8 and
            receipt.get("physical_duration_s") == ticks / 120.0 and
            receipt.get("encoded_duration_s") == ((ticks + 7) // 8) / 15.0 and
            receipt.get("extra_physics_ticks") == 0 and
            receipt.get("extra_pre_frames") == 0 and
            receipt.get("extra_post_frames") == 0,
            "source has a wrong or non-local task interval receipt")


def sealed_source(args: argparse.Namespace) -> dict[str, Any]:
    source = Path(args.source).resolve(strict=True)
    # The direct route permits an immutable run directory anywhere below its
    # own namespace (for example video_eval/validation/<run-id>), but never a
    # source from a different experiment.
    experiment_runs = (ROOT / "runs" / EXPERIMENT).resolve(strict=True)
    require(source.name == "source" and source.is_relative_to(experiment_runs) and
            source.parent != experiment_runs,
            "source is outside the isolated RR-capture experiment")
    source_manifest = source / "source_manifest.json"
    run_manifest = source.parent / "run_manifest.json"
    require(MEDIA.sha256(source_manifest) == MEDIA.checked_sha(
                args.source_manifest_sha256, "--source-manifest-sha256") and
            MEDIA.sha256(run_manifest) == MEDIA.checked_sha(
                args.source_run_manifest_sha256, "--source-run-manifest-sha256"),
            "sealed source/run manifest differs from explicit invocation")
    manifest, run = MEDIA.read_json(source_manifest), MEDIA.read_json(run_manifest)
    version = VERSIONS.version_family(manifest.get("schema"))
    require(run.get("lifecycle") == "COMPLETE" and run.get("mode") in ("eval", "diagnostic") and
            run.get("result") == manifest and
            manifest.get("continuous_natural_P01") is True and
            manifest.get("checkpoint_model_unchanged") is True and
            manifest.get("policy_switches") == 0 and manifest.get("PPO_updates") == 0 and
            manifest.get("front_FL_assist") is True and
            manifest.get("rear_task_assists") is False and
            manifest.get("rear_owner_projection") is False and
            manifest.get("ignored_legacy_sampling_profile") is True and
            manifest.get("error") is None,
            "source is not one sealed unchanged natural-P01 composite evaluation")
    diagnostic = manifest.get("mode") == "INDEPENDENT_DIRECTION_DIAGNOSTIC"
    require(diagnostic == bool(manifest.get("diagnostic_intervention")) and
            diagnostic == bool((manifest.get("control_contributions") or {}).get(
                "diagnostic_only_RR_joint_override")) and
            (not diagnostic or args.allow_diagnostic) and
            (diagnostic or not args.allow_diagnostic),
            "diagnostic/formal source requires the matching explicit CLI mode")
    _time_receipt(manifest)
    camera = manifest.get("camera") or {}
    require(camera == {"eye_m": [1.85, -1.65, 1.15],
                       "target_m": [0.7, -0.15, 0.15]},
            "source camera differs from the accepted CP225280 camera")
    identity = _checkpoint_identity(manifest, args)
    recorder_path = _record(source, manifest, "viewport_buffer_video_manifest.json")
    recorder = MEDIA.read_json(recorder_path)
    video = _record(source, manifest, "actual_viewport_video.mp4")
    ledger = _record(source, manifest, "viewport_frame_ledger.jsonl")
    decisions = _record(source, manifest, "video_policy_decisions.jsonl")
    ticks = _record(source, manifest, "capture_assist_ticks.jsonl")
    native_ticks = _record(source, manifest, "native_tick_audit.jsonl")
    require(recorder.get("valid") is True and
            recorder.get("encoder_finalized_before_app_close") is True and
            recorder.get("video_path") == str(video) and
            recorder.get("video_sha256") == MEDIA.sha256(video) and
            recorder.get("ledger_path") == str(ledger) and
            recorder.get("ledger_sha256") == MEDIA.sha256(ledger) and
            recorder.get("frame_count") == manifest["time_receipt"]["frame_count"],
            "source recorder did not close on the bound video/ledger")
    return {"source": source, "manifest": manifest, "manifest_path": source_manifest,
        "run_manifest": run, "run_manifest_path": run_manifest,
        "video": video, "ledger_path": ledger, "decisions_path": decisions,
        "tick_path": ticks, "native_tick_path": native_ticks,
        "capture": recorder, "capture_path": recorder_path,
        "endpoint": manifest["actual_ticks"], "checkpoint": identity, "version_family": version,
        "diagnostic_partial": False}


def _leg(ev: dict[str, Any], name: str) -> dict[str, Any]:
    value = (ev.get("current_legs") or {}).get(name)
    return value if isinstance(value, dict) else {}


def _contact(leg: dict[str, Any]) -> str:
    if leg.get("top_surface_contact") is True:
        return "TOP"
    if leg.get("ground_contact") is True:
        return "GROUND"
    if leg.get("air") is True:
        return "AIR"
    return str(leg.get("contact_mode") or leg.get("contact_surface") or "N/A")


def _force(leg: dict[str, Any]) -> float | None:
    for key in ("bearing_force_n", "obstacle_normal_force_n", "contact_reaction_force_n"):
        value = leg.get(key)
        if type(value) in (int, float) and math.isfinite(float(value)):
            return float(value)
    return None


def _decision_row(item: dict[str, Any], version: dict[str, Any] | None = None) -> dict[str, Any]:
    step = item.get("step_info") or {}
    tick = item.get("end_tick")
    request_phase = item.get("request_phase")
    require(type(tick) is int and step.get("physics_tick") == tick and
            isinstance(request_phase, str) and step.get("phase_id") == request_phase and
            item.get("environment_step_returned") is True,
            "decision ledger lacks one completed physical endpoint")
    request = item.get("policy_request") or {}
    version = (VERSIONS.version_family(request.get("schema"), kind="request_schema")
               if version is None else version)
    require(request.get("schema") == version["request_schema"] and
            request.get("policy_version") == version["actor_policy"] and
            request.get("sampling_draws") == 0 and
            request.get("selected_raw_log_probability") is None,
            "formal/diagnostic video decision is not deterministic composite inference")
    # The compact completed-decision record intentionally omits raw_observation
    # and atomic_ack.  Their authoritative after-step evidence is joined by
    # episode tick from capture_assist_ticks/native_tick_audit below.
    source_nominal = _vector(step.get("nominal_action_full12"), 12, "step source nominal")
    request_physical = _vector(step.get("projected_residual_full12"), 12,
                               "step projected physical REQUEST")
    final = _vector(step.get("actual_drive_target_full12"), 12, "step FINAL target")
    issued_raw = _vector(step.get("raw_policy_action_full12"), 12, "issued raw latent")
    step_audit = step.get("actuator_target_effect_audit") or {}
    require(step_audit.get("schema") == "wlr50_clean.actuator_target_effect_audit.v1" and
            step_audit.get("verified") is True and
            isinstance(step_audit.get("source_phase_id"), str) and
            step_audit.get("policy_request_phase") == request_phase,
            "completed step lacks its exact source/request phase audit")
    selected = _vector(request.get("selected_raw_full12"), 12, "actor selected raw")
    selected_tanh = _vector(request.get("selected_tanh_full12"), 12, "actor selected tanh")
    prior = _vector(request.get("prior_same_observation_conditional_mean_full12"),
                    12, "frozen-prior conditional raw")
    local = _vector(request.get("applied_local_raw_mean_delta_full12"),
                    12, "local raw correction")
    diagnostic = request.get("independent_diagnostic")
    require(diagnostic is None or isinstance(diagnostic, dict),
            "diagnostic intervention evidence must be an object")
    # Actor raw is dimensionless.  The environment-issued raw may intentionally
    # differ in the separately-labelled diagnostic, while a formal row must be
    # the actor output.  Neither is compared to the physical-unit REQUEST.
    require(diagnostic is not None or issued_raw == selected,
            "formal evaluation issued raw differs from actor selected raw")
    task = step.get("semantic_task") or {}
    local_task = step.get("rr_capture_local") or {}
    require(local_task.get("schema") == version["task_schema"],
            "decision lacks the observable RR-local task state")
    local_metrics = local_task.get("metrics") or {}
    require(local_metrics.get("tick") == tick and
            type(local_metrics.get("current_top_contact")) is bool and
            type(local_metrics.get("current_top_bearing")) is bool,
            "decision lacks same-endpoint current RR contact/bearing evidence")
    require(not local_metrics["current_top_bearing"] or
            local_metrics["current_top_contact"],
            "current RR bearing cannot be true without current top contact")
    request_eligible = endpoint_eligible = None
    if version["version"] == "v2":
        field = VERSIONS.ELIGIBILITY_FIELD
        request_eligible = (request.get("capture_context") or {}).get(field)
        endpoint_eligible = local_metrics.get(field)
        require(type(request_eligible) in (int, float) and request_eligible in (0, 1) and
                type(endpoint_eligible) is bool,
                "v2 decision is missing boolean request/endpoint current-attempt eligibility")
        obs9 = local_task.get("obs9")
        require(isinstance(obs9, list) and len(obs9) == 9 and
                obs9[8] == float(endpoint_eligible),
                "v2 decision obs9 disagrees with endpoint current-attempt eligibility")
        # Request context belongs to the start of the eight-tick decision;
        # endpoint task evidence can legitimately differ after contact/GROUND.
        require(not local_task.get("local_success") or
                (endpoint_eligible and local_metrics["current_top_contact"] and
                 local_metrics["current_top_bearing"]),
                "v2 local success lacks eligible current TOP bearing")
    return {"tick": tick, "time_s": float(tick) / 120.0,
        "version": version["version"],
        "request_phase": request_phase,
        "phase": step.get("end_phase_id") or task.get("stage_id"),
        "source_nominal_step": source_nominal,
        "request_physical_step": request_physical, "final_step": final,
        "step_audit_source_phase": step_audit["source_phase_id"],
        "step_audit_policy_request_phase": step_audit["policy_request_phase"],
        "step_audit_raw_physics_tick": step_audit.get("physics_tick"),
        "prior_raw": prior, "local_raw": local, "selected_raw": selected,
        "selected_tanh": selected_tanh, "issued_raw": issued_raw,
        "diagnostic_intervention": diagnostic,
        "local_active": local_task.get("active") is True,
        "local_success": local_task.get("local_success") is True,
        "local_hold_s": local_task.get("hold_elapsed_s"),
        "capture_progress": local_task.get("capture_hold_progress"),
        "request_capture_eligible": request_eligible,
        "endpoint_capture_eligible": endpoint_eligible,
        "local_metrics": local_metrics}


def _join_endpoint(row: dict[str, Any], tick_row: dict[str, Any],
                   native_row: dict[str, Any]) -> dict[str, Any]:
    """Join one completed 15 Hz decision to its saved 120 Hz endpoint."""
    dispatch = tick_row.get("dispatch") or {}
    native = native_row.get("native_audit") or {}
    require(tick_row.get("episode_physics_tick") == row["tick"] and
            native_row.get("episode_physics_tick") == row["tick"],
            "decision and native evidence use different episode endpoints")
    require(native.get("schema") == "wlr50_clean.actuator_target_effect_audit.v1" and
            native.get("verified") is True and
            native.get("actual_mapping_matches_dispatch") is True and
            native.get("setter_dispatch_targets_equal") is True,
            "native endpoint lacks verified actuator evidence")
    raw_tick = dispatch.get("physics_tick")
    require(type(raw_tick) is int and raw_tick == native.get("physics_tick") ==
                row["step_audit_raw_physics_tick"] and
            tick_row.get("phase") == row["phase"] and
            native.get("source_phase_id") == row["step_audit_source_phase"] and
            native.get("policy_request_phase") == row["request_phase"] ==
                row["step_audit_policy_request_phase"],
            "episode/native endpoint clocks or phases disagree")
    # The capture row's top-level nominal is the post-step source suggestion
    # for the next interval.  The command actually dispatched for this endpoint
    # is the native-audit nominal, independently matched to step_info.
    poststep_source_hint = _vector(tick_row.get("nominal_full12"), 12,
                                   "post-step source hint")
    native_nominal = _vector(native_row.get("nominal_full12"), 12,
                             "native-audit dispatched source command")
    mapped = _vector(dispatch.get("native_drive_target_full12"), 12,
                     "tick mapped N")
    native_mapped = _vector(native.get("native_drive_target_full12"), 12,
                            "native-audit mapped N")
    requested = _vector(dispatch.get("independent_policy_residual_requested_full12"),
                        12, "tick projected physical REQUEST")
    native_projected = _vector(native_row.get("projected_residual_full12"), 12,
                               "native projected physical REQUEST")
    final = _vector(dispatch.get("drive_target_full12"), 12, "tick FINAL")
    actual = _vector(tick_row.get("actual_full12"), 12, "tick measured actual")
    native_issued_raw = _vector(native.get("raw_policy_action_full12"), 12,
                                "native issued raw latent")
    effective_value = dispatch.get("independent_policy_residual_effective_full12")
    effective = (None if effective_value is None else
                 _vector(effective_value, 12, "tick effective physical residual"))
    headroom_effective_value = ((native.get("policy_headroom_evidence") or {}).get(
        "effective_policy_residual_full12"))
    headroom_effective = (None if headroom_effective_value is None else
                          _vector(headroom_effective_value, 12,
                                  "native effective physical residual"))
    if effective is not None and headroom_effective is not None:
        require(effective == headroom_effective,
                "tick/native effective physical residual differs")
    effective = effective if effective is not None else headroom_effective
    require(native_nominal == row["source_nominal_step"] and
            mapped == native_mapped and
            requested == native_projected == row["request_physical_step"] and
            final == row["final_step"] and native_issued_raw == row["issued_raw"],
            "decision/tick/native command lineage differs at encoded endpoint")
    rr, fl, fr, rl = (_leg(tick_row, name) for name in ("RR", "FL", "FR", "RL"))
    placed = tick_row.get("placed_history") or {}
    gap, front = rr.get("clearance_m"), rr.get("front_distance_m")
    row.update(source_nominal=native_nominal, poststep_source_hint=poststep_source_hint,
               mapped_nominal=mapped,
               request_physical=requested, effective_physical=effective,
               final=final, actual=actual, raw_physics_tick=raw_tick,
               evidence_alignment="episode endpoint -> same saved native dispatch/readback",
               rr_gap_mm=None if not isinstance(gap, (int, float)) else 1000.0*float(gap),
               rr_front_mm=None if not isinstance(front, (int, float)) else 1000.0*float(front),
               rr_contact=_contact(rr), rr_force_n=_force(rr),
               rr_current_top_contact=row["local_metrics"]["current_top_contact"],
               rr_current_top_bearing=row["local_metrics"]["current_top_bearing"],
               rr_sensor_bearing_flag=rr.get("bearing_verified") is True,
               rr_placed=placed.get("RR") is True,
               support={name: {"contact": _contact(leg), "force_n": _force(leg)}
                        for name, leg in (("FL", fl), ("FR", fr), ("RL", rl))})
    return row


def capture_rows(context: dict[str, Any], ledger: list[Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    decisions: dict[int, dict[str, Any]] = {}
    expected = 1
    with context["decisions_path"].open("rb") as stream:
        for raw in stream:
            require(raw.endswith(b"\n"), "decision ledger has a partial row")
            item = json.loads(raw)
            require(item.get("decision") == expected, "decision numbering gap")
            row = _decision_row(item, context["version_family"])
            row["local_mean_coordinate_gain_full12"] = VERSIONS.COORDINATES.validate_coordinate_video_request(
                item.get("policy_request") or {}, context["checkpoint"]["version_family"]["mean_coordinates"])
            require(context["checkpoint"]["diagnostic"] or row["diagnostic_intervention"] is None,
                    "formal source contains a diagnostic action override")
            require(row["tick"] not in decisions, "duplicate decision endpoint")
            decisions[row["tick"]] = row
            expected += 1
    wanted = {item.sim_step for item in ledger}
    tick_rows: dict[int, dict[str, Any]] = {}
    native_rows: dict[int, dict[str, Any]] = {}
    count = 0
    with context["tick_path"].open("rb") as stream:
        for count, raw in enumerate(stream, 1):
            require(raw.endswith(b"\n"), "native tick ledger has a partial row")
            value = json.loads(raw)
            require(value.get("episode_physics_tick") == count and
                    math.isclose(float(value.get("sim_time_s")), count / 120.0,
                                 rel_tol=0.0, abs_tol=1e-10),
                    "native tick ledger clock gap")
            if count in wanted:
                tick_rows[count] = value
    native_count = 0
    with context["native_tick_path"].open("rb") as stream:
        for native_count, raw in enumerate(stream, 1):
            require(raw.endswith(b"\n"), "native audit ledger has a partial row")
            value = json.loads(raw)
            require(value.get("episode_physics_tick") == native_count,
                    "native audit ledger episode clock gap")
            if native_count in wanted:
                native_rows[native_count] = value
    require(count == context["endpoint"] and native_count == context["endpoint"] and
            set(tick_rows) == wanted and set(native_rows) == wanted and
            wanted <= set(decisions),
            "sealed evidence does not cover every encoded decision endpoint")
    rows = []
    for item in ledger:
        row = decisions[item.sim_step]
        tick_row = tick_rows[item.sim_step]
        native_row = native_rows[item.sim_step]
        rows.append(_join_endpoint(row, tick_row, native_row))
    selected = []
    prior = None
    for row in rows:
        signature = (row["phase"], row["local_active"], row["local_success"],
                     row["rr_contact"], row["rr_placed"], row["endpoint_capture_eligible"])
        if signature != prior:
            selected.append(row)
        prior = signature
    if not selected or selected[-1]["tick"] != rows[-1]["tick"]:
        selected.append(rows[-1])
    return rows, selected


def outcome(context: dict[str, Any]) -> dict[str, Any]:
    manifest = context["manifest"]
    local_success = manifest.get("local_RR_success") is True
    full_success = manifest.get("full_task_success") is True
    terminal = manifest.get("terminal_info") or {}
    task = terminal.get("semantic_task") or {}
    reason = terminal.get("termination_reason") or task.get("termination_reason")
    if manifest.get("schema") == VERSIONS.VERSION_FAMILIES["v2"]["source_schema"]:
        local = manifest.get("local_task") or {}
        metrics = local.get("metrics") or {}
        require(local.get("schema") == VERSIONS.VERSION_FAMILIES["v2"]["task_schema"] and
                local.get("local_success") is local_success and
                terminal.get("local_task_success") is local_success,
                "v2 source/terminal current local result differs")
        require(not local_success or
                (metrics.get(VERSIONS.ELIGIBILITY_FIELD) is True and
                 metrics.get("current_top_contact") is True and
                 metrics.get("current_top_bearing") is True and
                 metrics.get("ground_contact") is False),
                "v2 source claims success without current eligible non-GROUND TOP bearing")
    return {"result": "RR_LOCAL_SUCCESS" if local_success else "INCOMPLETE",
        "success": local_success, "full_task_success": full_success,
        "termination_reason": reason,
        "termination_source": task.get("termination_source"),
        "source_acceptance_error": None}


def _fmt(value: Any, digits: int = 2) -> str:
    return "N/A" if not isinstance(value, (int, float)) else f"{float(value):.{digits}f}"


def _v(values: list[float], indices: tuple[int, ...], digits: int = 2) -> str:
    return ",".join(_fmt(values[index], digits) for index in indices)


def _maybe_v(values: list[float] | None, indices: tuple[int, ...], digits: int = 2) -> str:
    return "N/A" if values is None else _v(values, indices, digits)


def panel_lines(row: dict[str, Any], result: dict[str, Any], identity: dict[str, Any]) -> list[str]:
    counts = identity["counts"]
    mode = "DIAGNOSTIC PROBE / PPO CREDIT 0" if identity["diagnostic"] else "DETERMINISTIC COMPOSITE POLICY"
    local_result = "RR CAPTURE SUCCESS (LOCAL)" if result["success"] else "RR CAPTURE INCOMPLETE"
    version = (identity.get("version_family") or {}).get("version", "v1")
    tracking = VERSIONS.source_tracking_disclosure(version, identity.get("source_tracking_owner_revision"))
    eligibility = (f" qualified request/end={row.get('request_capture_eligible')}/{row.get('endpoint_capture_eligible')}"
                   if version == "v2" else "")
    ledger_line = (f"training ledger +{counts['local_policy_decisions']} decisions / +{counts['local_ppo_updates']} PPO / "
        f"+{counts['local_optimizer_steps']} Adam | training-prefix={counts.get('prefix_decisions','N/A')} decisions, PPO credit 0 | AUX +0")
    if version == "v2":
        ledger_line = (f"training total +{counts['local_policy_decisions']} decisions / {counts['local_ppo_updates']} PPO / "
            f"{counts['local_optimizer_steps']} Adam | task-v2 +{counts['task_v2_policy_decisions']} / {counts['task_v2_ppo_updates']} PPO | "
            f"prefix {counts.get('prefix_decisions','N/A')} not PPO | AUX +{counts.get('auxiliary_updates',0)}")
    module_line = (f"FROZEN PRIOR: CP225280 | LEARNED: RR LOCAL {version} | TRACKING: {tracking['hud_label']} | "
                   "ONE COMPOSITE CHECKPOINT")
    if counts.get("auxiliary_updates", 0):
        module_line = (f"CP225280 FROZEN | RR LOCAL {version} | TRACKING: {tracking['hud_label']} | "
                       + VERSIONS.AUX_TRAINING_LABEL)
    return [
        f"{mode} | {local_result} | CP225280+local{counts['local_policy_decisions']} | g{identity['runtime_head'][:12]}",
        module_line,
        ("FL ASSIST: ON (p05 hip-only) | REAR ASSISTS: OFF | OWNER PROJECTION: OFF | "
         + ("SOURCE: RR-first pending P09 late / new P12" if version == "v2" else "SOURCE: original v1")),
        ledger_line,
        (f"t={_fmt(row['time_s'],3)}s tick={row['tick']} {row['request_phase']}->{row['phase']} | local active={row['local_active']} "
         f"hold={_fmt(row['local_hold_s'],3)}s progress={_fmt(row['capture_progress'],3)}"),
        (f"RR gap/front={_fmt(row['rr_gap_mm'],3)}/{_fmt(row['rr_front_mm'],3)}mm {row['rr_contact']} "
         f"force={_fmt(row['rr_force_n'],2)}N currentTOP={row['rr_current_top_contact']} "
         f"currentBearing={row['rr_current_top_bearing']} placed_hist={row['rr_placed']}" + eligibility),
        (f"RR deg dispatched source/mappedN -> FINAL/actual: {_v(row['source_nominal'],(6,7),2)}/"
         f"{_v(row['mapped_nominal'],(6,7),2)} -> {_v(row['final'],(6,7),2)}/"
         f"{_v(row['actual'],(6,7),2)} | poststep hint={_v(row['poststep_source_hint'],(6,7),2)}"),
        (f"RR actor unitless selected/tanh -> issued raw: {_v(row['selected_raw'],(6,7),3)} / "
         f"{_v(row['selected_tanh'],(6,7),3)} -> {_v(row['issued_raw'],(6,7),3)} | local={_v(row['local_raw'],(6,7),3)}"),
        (f"RR physical REQUEST/effective [deg]: {_v(row['request_physical'],(6,7),3)} / "
         f"{_maybe_v(row['effective_physical'],(6,7),3)}"),
        (f"wheel dispatched source/mappedN [FL,FR,RL,RR]: {_v(row['source_nominal'],(8,9,10,11),2)}/"
         f"{_v(row['mapped_nominal'],(8,9,10,11),2)} | poststep hint={_v(row['poststep_source_hint'],(8,9,10,11),2)}"),
        (f"wheel actor unitless selected/tanh -> issued raw: {_v(row['selected_raw'],(8,9,10,11),3)} / "
         f"{_v(row['selected_tanh'],(8,9,10,11),3)} -> {_v(row['issued_raw'],(8,9,10,11),3)}"),
        (f"wheel physical REQUEST -> FINAL [rad/s]: {_v(row['request_physical'],(8,9,10,11),3)} -> "
         f"{_v(row['final'],(8,9,10,11),3)}"),
        f"wheel measured actual qd [FL,FR,RL,RR]: {_v(row['actual'],(8,9,10,11),3)} rad/s",
        ("RAW OVERRIDE: DIAGNOSTIC, PPO CREDIT 0 | mapper deltas are not all policy causality"
         if row["diagnostic_intervention"] is not None else
         "RAW OVERRIDE: none; actor raw issued | mapper deltas are not all policy causality"),
        (f"support FL={row['support']['FL']['contact']}/{_fmt(row['support']['FL']['force_n'],2)}N "
         f"FR={row['support']['FR']['contact']}/{_fmt(row['support']['FR']['force_n'],2)}N "
         f"RL={row['support']['RL']['contact']}/{_fmt(row['support']['RL']['force_n'],2)}N | full task success={result['full_task_success']}"),
    ]


def detail_plan(rows: list[dict[str, Any]], identity: dict[str, Any],
                result: dict[str, Any]) -> dict[str, Any]:
    step = identity["display_step"]
    active = next((index for index, row in enumerate(rows) if row["local_active"]), None)
    suffix = "" if result["success"] else "_INCOMPLETE"
    diagnostic = "_DIAGNOSTIC" if identity["diagnostic"] else ""
    mode_label = ("DIRECTION DIAGNOSTIC | PPO CREDIT 0 | "
                  if identity["diagnostic"] else "DETERMINISTIC COMPOSITE POLICY | ")
    if identity["counts"].get("auxiliary_updates", 0):
        mode_label = (("DIAGNOSTIC / PPO CREDIT 0 | " if identity["diagnostic"] else "DET | ")
                      + VERSIONS.AUX_TRAINING_LABEL + " | ")
    if active is None:
        return {"kind": "RR_CAPTURE_WINDOW_NOT_REACHED", "start": max(0, len(rows)-900),
            "end": len(rows),
            "filename": f"CP{step}{diagnostic}_DET_RR_capture_not_reached{suffix}.mp4",
            "title": (f"DETAIL | {mode_label}CP225280+local{identity['counts']['local_policy_decisions']} | "
                      "RR CAPTURE WINDOW NOT REACHED"),
            "requested_RR_detail_unavailable_reason": "THIS_EPISODE_DID_NOT_ACTIVATE_RR_LOCAL_POLICY"}
    return {"kind": "RR_CAPTURE_LOCAL_ACTUAL_WINDOW", "start": active, "end": len(rows),
        "filename": f"CP{step}{diagnostic}_DET_RR_capture_detail{suffix}.mp4",
        "title": (f"DETAIL | {mode_label}CP225280+local{identity['counts']['local_policy_decisions']} | "
                  f"{'RR LOCAL SUCCESS' if result['success'] else 'RR CAPTURE INCOMPLETE'} | FULL TASK {result['full_task_success']}"),
        "requested_RR_detail_unavailable_reason": None}


def fixed_cp225280() -> dict[str, Any]:
    source = REFERENCE_SOURCE.resolve(strict=True)
    manifest_path = source / "semantic_video_source_manifest.json"
    run_path = source.parent / "run_manifest.json"
    require(MEDIA.sha256(manifest_path) == REFERENCE_SOURCE_MANIFEST_SHA and
            MEDIA.sha256(run_path) == REFERENCE_RUN_MANIFEST_SHA,
            "accepted CP225280 reference manifests changed")
    manifest = MEDIA.read_json(manifest_path)
    video = MEDIA.artifact(source, manifest, "actual_viewport_video.mp4")
    ledger = MEDIA.artifact(source, manifest, "viewport_frame_ledger.jsonl")
    capture_path = MEDIA.artifact(source, manifest, "viewport_buffer_video_manifest.json")
    capture = MEDIA.read_json(capture_path)
    require(MEDIA.sha256(video) == REFERENCE_VIDEO_SHA and
            MEDIA.sha256(ledger) == REFERENCE_LEDGER_SHA and
            manifest.get("camera") == {"eye_m": [1.85, -1.65, 1.15],
                                       "fps": 15, "resolution": [1280, 720],
                                       "target_m": [0.7, -0.15, 0.15]},
            "accepted CP225280 reference video/ledger/camera changed")
    return {"source": source, "manifest": manifest, "manifest_path": manifest_path,
        "run_manifest_path": run_path, "video": video, "ledger_path": ledger,
        "capture": capture, "capture_path": capture_path,
        "endpoint": manifest["episode_physics_ticks"]}


def encode_pair(reference: dict[str, Any], candidate: dict[str, Any], full: Path,
                output: Path, identity: dict[str, Any], result: dict[str, Any], ffmpeg: Path) -> dict[str, Any]:
    require(reference["manifest"]["camera"] == {
                **candidate["manifest"]["camera"], "fps": 15, "resolution": [1280, 720]},
            "candidate and accepted CP225280 camera definitions differ")
    _, ledger, validation = MEDIA.shared().checked_media(reference, ffmpeg=ffmpeg)
    left, right = len(ledger), candidate["capture"]["frame_count"]
    count = max(left, right)
    candidate_label = (f"CP225280+local{identity['counts']['local_policy_decisions']} DET | "
        f"{'RR LOCAL SUCCESS' if result['success'] else 'RR CAPTURE INCOMPLETE'} | FL ON / REAR ASSIST OFF")
    if identity["counts"].get("auxiliary_updates", 0):
        candidate_label = f"CP{identity['display_step']} DET | " + VERSIONS.AUX_TRAINING_LABEL
    labels = (
        "ACCEPTED CP225280 ORIGINAL49eb | FROZEN REFERENCE | NOT NEW PHYSICS",
        candidate_label)
    filters = []
    for index, (label, frames) in enumerate(zip(labels, (left, right))):
        filters.append(f"[{index}:v]setpts=PTS-STARTPTS,scale=960:540,pad=960:610:0:70:black,"
            f"tpad=stop_mode=clone:stop=-1,setpts=N/(15*TB),"
            "drawtext=fontfile='C\\:/Windows/Fonts/arial.ttf':"
            f"text='{label}':fontcolor=white:fontsize={16 if index == 1 and identity['counts'].get('auxiliary_updates', 0) else 20}:x=14:y=13,"
            "drawtext=fontfile='C\\:/Windows/Fonts/arial.ttf':"
            f"text='RUN ENDED - FROZEN, NOT NEW PHYSICS':fontcolor=yellow:fontsize=20:x=14:y=44:enable='gte(n,{frames})'[v{index}]")
    filters.append("[v0][v1]hstack=inputs=2:shortest=1,format=yuv420p[out]")
    command = [str(ffmpeg), "-hide_banner", "-nostdin", "-v", "error", "-n",
        "-threads", "2", "-filter_complex_threads", "1", "-i", str(reference["video"]),
        "-i", str(full), "-filter_complex", ";".join(filters), "-map", "[out]", "-an",
        "-frames:v", str(count), "-r", "15", "-fps_mode", "cfr", "-c:v", "libx264",
        "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-threads", "2",
        "-movflags", "+faststart", str(output)]
    MEDIA.shared().run(command)
    return {"output": str(output), "frame_count": count,
        "reference": "accepted_CP225280_original49eb", "reference_source": str(reference["source"]),
        "reference_source_manifest_sha256": REFERENCE_SOURCE_MANIFEST_SHA,
        "reference_video_sha256": REFERENCE_VIDEO_SHA,
        "candidate_freeze_frames": count-right, "reference_freeze_frames": count-left,
        "freeze_is_physical_evidence": False, "same_camera": True,
        "same_controller_claimed": False, "normal_speed": True,
        "candidate_title": candidate_label,
        "reference_source_validation": validation,
        "validation": MEDIA.shared().validate_output(output, count, 1920, 610, ffmpeg),
        "previews": MEDIA.shared().preview(output, count, ffmpeg), "command": command}


def export(args: argparse.Namespace) -> dict[str, Any]:
    source = sealed_source(args)
    destination = Path(args.destination).resolve()
    require(destination.is_relative_to(OUTPUT_ROOT.resolve()) and not destination.exists(),
            "destination must be a new directory under the isolated output namespace")
    reference = fixed_cp225280()
    require(source["manifest"]["camera"] == {
                "eye_m": reference["manifest"]["camera"]["eye_m"],
                "target_m": reference["manifest"]["camera"]["target_m"]},
            "new source is not the accepted CP225280 same camera")
    ffmpeg = MEDIA.shared().find_ffmpeg(
        (source["capture"].get("full_decode") or {}).get("ffmpeg_path"))
    _, ledger, validation = MEDIA.shared().checked_media(source, ffmpeg=ffmpeg)
    rows, selected = capture_rows(source, ledger)
    result, identity = outcome(source), source["checkpoint"]
    plan = detail_plan(rows, identity, result)
    destination.mkdir(parents=True)
    previous_panel = MEDIA.panel_lines
    MEDIA.panel_lines = panel_lines
    suffix = "" if result["success"] else "_INCOMPLETE"
    diagnostic = "_DIAGNOSTIC" if identity["diagnostic"] else ""
    full_path = destination / (
        f"CP{identity['display_step']}{diagnostic}_DET_P01_to_RR_placement{suffix}.mp4")
    try:
        full = MEDIA.encode_full(source, rows, full_path, result, identity, ffmpeg)
    finally:
        MEDIA.panel_lines = previous_panel
    print("FULL_PLAYABLE_VALIDATED " + full["output"], flush=True)
    detail = MEDIA.encode_detail(Path(full["output"]), rows, plan, destination, ffmpeg)
    pair = encode_pair(reference, source, Path(full["output"]),
        destination / f"CP225280_vs_CP{identity['display_step']}{diagnostic}_DET_same_camera.mp4",
        identity, result, ffmpeg)
    selected_path = destination / "selected_real_events.json"
    MEDIA.write_new_json(selected_path, {"schema": "wlr50_clean.rr_capture_first_selected_events." + identity["version_family"]["version"],
        "source": str(source["source"]), "selection": "actual encoded endpoints and state transitions; no interpolation",
        "rows": selected})
    receipt = {"schema": "wlr50_clean.rr_capture_first_video_export." + identity["version_family"]["version"],
        "source": str(source["source"]), "source_manifest": str(source["manifest_path"]),
        "source_manifest_sha256": MEDIA.sha256(source["manifest_path"]),
        "source_run_manifest_sha256": MEDIA.sha256(source["run_manifest_path"]),
        "source_video_validation": validation, "checkpoint_identity": identity,
        "version_family": identity["version_family"],
        "source_tracking_owner_revision": identity["source_tracking_owner_revision"],
        "source_tracking": identity["source_tracking"],
        "auxiliary_training": identity["auxiliary_training"],
        "capture_source_dispatch": (source["manifest"].get("control_contributions") or {}).get("capture_source_dispatch"),
        "visible_labels": {"frozen_prior": "CP225280", "learned_module": "RR local " + identity["version_family"]["version"],
            "FL_assist": "ON p05 hip-only non-learning", "rear_task_assists": "OFF",
            "source_tracking": identity["source_tracking"]["hud_label"],
            "auxiliary_training": identity["auxiliary_training"]["label"],
            "owner_projection": "OFF", "diagnostic": identity["diagnostic"]},
        "local_RR_success": result["success"], "full_task_success": result["full_task_success"],
        "termination_reason": result["termination_reason"],
        "full_episode_continuous": True, "full_failure_tail_preserved": True,
        "normal_speed": True, "stitched": False, "extra_intro_frames": 0,
        "full": full, "detail": detail, "CP225280_comparison": pair,
        "selected_real_events": str(selected_path),
        "selected_real_events_sha256": MEDIA.sha256(selected_path),
        "claims": {"frozen_prior_counted_as_new_learning": False,
            "prefix_counted_as_local_PPO_credit": False,
            "rear_task_teacher_or_completion_assist_active": False,
            "RR_local_success_is_full_task_success": False,
            "CP225280_reference_is_historical_N": False,
            "frozen_comparison_tail_is_physical_evidence": False,
            "diagnostic_override_is_formal_PPO_success": False},
        "code_binding": {"exporter": str(Path(__file__).resolve()),
            "exporter_sha256": MEDIA.sha256(Path(__file__)),
            "version_contract_helper": str(OUTPUT_ROOT / "export_formal_from_sealed.py"),
            "version_contract_helper_sha256": MEDIA.sha256(OUTPUT_ROOT / "export_formal_from_sealed.py"),
            "media_helper": str(MEDIA_PATH), "media_helper_sha256": MEDIA.sha256(MEDIA_PATH)}}
    receipt_path = destination / "export_receipt.json"
    MEDIA.write_new_json(receipt_path, receipt)
    print(json.dumps({"result": result, "full": full["output"],
        "detail": detail["output"], "comparison": pair["output"],
        "receipt": str(receipt_path)}, indent=2))
    return receipt


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument("--source", type=Path, required=True)
    result.add_argument("--destination", type=Path, required=True)
    result.add_argument("--source-manifest-sha256", required=True)
    result.add_argument("--source-run-manifest-sha256", required=True)
    result.add_argument("--expected-head", required=True)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--checkpoint-sha256", required=True)
    result.add_argument("--checkpoint-manifest-sha256", required=True)
    result.add_argument("--expected-local-policy-decisions", type=int, required=True)
    result.add_argument("--expected-local-ppo-updates", type=int, required=True)
    result.add_argument("--expected-local-optimizer-steps", type=int, required=True)
    result.add_argument("--allow-diagnostic", action="store_true",
        help="Required only for explicitly-labelled non-PPO direction-diagnostic sources.")
    return result


if __name__ == "__main__":
    export(parser().parse_args())
