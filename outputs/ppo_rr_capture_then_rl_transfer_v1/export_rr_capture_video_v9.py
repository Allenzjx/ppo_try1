"""Strict outputs-only exporter for a future learned v9 branch checkpoint.

This adapter is intentionally dormant until an actual checkpoint, its direct
parent, and a sealed natural-P01 video source are supplied explicitly.  It
does not discover ``latest`` state.  The v9 migration is zero-learning and
changes only P12 post-authored-stop wheel retention; it does not claim that a
front-leg regression was repaired, that RR/RL was reached, or that controller
credit is PPO.

An optional ``rr_capture_transfer_branch.front_retention_auxiliary`` ledger is
accepted only when its complete canonical JSON bytes and separate accepted /
attempted counts are explicitly pinned.  Historical front AUX and all PPO
counters remain separate.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
V8_ADAPTER = HERE / "export_rr_capture_video_v8.py"
V8_ADAPTER_SHA = "b97d33d8fd5a5b2ae9f143f8a818d38a31ca3db86d6da872eb44b13c3f5cf0db"

SCHEMA = "wlr50_clean.rr_postcapture_wheel_same410.v9"
FACTOR_KEY = "rr_postcapture_wheel_v9_factor"
MIGRATION_KEY = "rr_postcapture_wheel_v9_migration"
PRIOR_MIGRATION_KEY = "rr_signed_wheel_v8_migration"
SOURCE_HEAD = "d1871df37d6ea909657511d0e43e7435198f6ccd"
TARGET_HEAD = "3edda51732f4ff85717fcb3491bf5c8c5766474d"
BRANCH_NAME = "ancestor220544_signed_wheel_v8"
POLICY_VERSION = "rr_capture_then_rl_transfer_history_v1"
OBSERVATION_LAYOUT = "role389_rr_capture_transfer_v1"
FEEDBACK_REVISION = "signed_band_contact_formation_incremental_v6"
CONTROL_REVISION = "v9_P12_postcapture_wheel_retention"
METHOD = (
    "PPO_PLUS_FL_RR_CAPTURE_ASSISTS_PLUS_SIGNED_SUPPORT_WHEEL_V8_PLUS_"
    "POSTCAPTURE_WHEEL_V9_WITH_INHERITED_LIMITED_AUX"
)
COUNTER_KEYS = ("global_policy_decisions", "ppo_updates", "optimizer_steps")
BRANCH_ORIGIN = dict(zip(COUNTER_KEYS, (220544, 1688, 33760)))
V9_ORIGIN = dict(zip(COUNTER_KEYS, (221952, 1699, 33980)))
SOURCE_SELECTION = {
    "checkpoint_sha256": "746c3abb9aa6bfced2c194a45b7c1c9db2cf543a25ab1c033108233bce1c685a",
    "manifest_sha256": "8e4caab3cc7a1bf66c0b1e41617413c693405367efa68f627a461d99f51a629e",
    "source_git_commit": SOURCE_HEAD,
    "counters": V9_ORIGIN,
    "source_role": "learned_ancestor_branch_continuation",
    "checkpoint_output_branch": BRANCH_NAME,
    "branch_origin": BRANCH_ORIGIN,
}
SOURCE_WHEEL_SEMANTICS = (
    "P09_post_source_committed_stop_current_RR_AIR_Q_cross_supported_FL_FR_RL_"
    "signed_task_band_nonnegative_depth_gap_floor_previous_FINAL_1p8_slew_TOP_release"
)
WHEEL_SEMANTICS = (
    "P09_v8_unchanged_P12_RR_placed_RL_unplaced_post_authored_wheel_stop_"
    "current_TOP_or_qualified_signed_AIR_bearing_only_depth_floor_previous_FINAL_1p8_slew"
)

_V8 = None
_MEDIA = None
_V8_FRAME_SUMMARY = None
_V8_DETAIL_PLAN = None
_PINS = None


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sidecar(checkpoint):
    checkpoint = Path(checkpoint)
    return checkpoint.with_name(checkpoint.stem + "_manifest.json")


def checked_sha(value, label):
    value = str(value).lower()
    require(re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            f"{label} must be an explicit lowercase SHA256")
    return value


def checked_file(path, expected_sha, label):
    path = Path(path).resolve(strict=True)
    require(path.is_file() and sha256(path) == expected_sha,
            f"{label} is missing or differs from its explicit SHA256")
    return path


def configure(arguments):
    """Freeze all mutable paths before any reviewed media module is loaded."""
    global _PINS
    require(_PINS is None, "v9 exporter pins may be configured only once")
    require(str(arguments.expected_head).lower() == TARGET_HEAD,
            "--expected-head differs from the frozen v9 runtime")
    require(arguments.checkpoint_output_branch == BRANCH_NAME,
            "v9 continuation must stay in the existing isolated branch")
    names = (
        "checkpoint_sha256", "checkpoint_manifest_sha256",
        "published_checkpoint_sha256", "published_checkpoint_manifest_sha256",
        "migration_plan_sha256", "publication_sha256", "publisher_script_sha256",
        "migration_source_checkpoint_sha256", "migration_source_manifest_sha256",
        "expected_parent_checkpoint_sha256", "expected_parent_checkpoint_manifest_sha256",
    )
    hashes = {name: checked_sha(getattr(arguments, name),
        "--" + name.replace("_", "-")) for name in names}
    require(hashes["migration_source_checkpoint_sha256"] ==
                SOURCE_SELECTION["checkpoint_sha256"]
            and hashes["migration_source_manifest_sha256"] ==
                SOURCE_SELECTION["manifest_sha256"],
            "v9 migration source is not exact learned CP221952")
    files = {
        "checkpoint": checked_file(arguments.checkpoint,
            hashes["checkpoint_sha256"], "evaluated v9 checkpoint"),
        "published_checkpoint": checked_file(arguments.published_v9_checkpoint,
            hashes["published_checkpoint_sha256"], "published zero-update v9 checkpoint"),
        "migration_plan": checked_file(arguments.migration_plan,
            hashes["migration_plan_sha256"], "v9 migration plan"),
        "publication": checked_file(arguments.publication,
            hashes["publication_sha256"], "v9 publication"),
        "publisher_script": checked_file(arguments.publisher_script,
            hashes["publisher_script_sha256"], "v9 publisher"),
        "migration_source_checkpoint": checked_file(arguments.migration_source_checkpoint,
            hashes["migration_source_checkpoint_sha256"], "learned v8 source"),
        "expected_parent_checkpoint": checked_file(arguments.expected_parent_checkpoint,
            hashes["expected_parent_checkpoint_sha256"], "direct v9 parent"),
    }
    manifests = {
        "checkpoint_manifest": sidecar(files["checkpoint"]).resolve(strict=True),
        "published_manifest": sidecar(files["published_checkpoint"]).resolve(strict=True),
        "migration_source_manifest": sidecar(files["migration_source_checkpoint"]).resolve(strict=True),
        "expected_parent_checkpoint_manifest": sidecar(
            files["expected_parent_checkpoint"]).resolve(strict=True),
    }
    require(sha256(manifests["checkpoint_manifest"]) ==
                hashes["checkpoint_manifest_sha256"],
            "evaluated v9 checkpoint manifest differs")
    require(sha256(manifests["published_manifest"]) ==
                hashes["published_checkpoint_manifest_sha256"],
            "published v9 manifest differs")
    require(sha256(manifests["migration_source_manifest"]) ==
                hashes["migration_source_manifest_sha256"],
            "v9 source manifest differs")
    require(sha256(manifests["expected_parent_checkpoint_manifest"]) ==
                hashes["expected_parent_checkpoint_manifest_sha256"],
            "direct v9 parent manifest differs")
    current = {
        "global_policy_decisions": arguments.expected_global_policy_decisions,
        "ppo_updates": arguments.expected_ppo_updates,
        "optimizer_steps": arguments.expected_optimizer_steps,
    }
    require(all(type(current[key]) is int and current[key] >= V9_ORIGIN[key]
                for key in COUNTER_KEYS)
            and any(current[key] > V9_ORIGIN[key] for key in COUNTER_KEYS),
            "evaluated v9 checkpoint must contain actual post-migration PPO credit")
    ledger_args = (arguments.expected_front_retention_ledger,
                   arguments.expected_front_retention_ledger_sha256,
                   arguments.expected_front_retention_accepted,
                   arguments.expected_front_retention_attempted)
    require(all(value is None for value in ledger_args)
            or all(value is not None for value in ledger_args),
            "front-retention ledger path/SHA/count pins must be supplied together")
    ledger_path = None
    ledger_sha = None
    if arguments.expected_front_retention_ledger is not None:
        ledger_sha = checked_sha(arguments.expected_front_retention_ledger_sha256,
                                 "--expected-front-retention-ledger-sha256")
        ledger_path = checked_file(arguments.expected_front_retention_ledger,
                                   ledger_sha, "front-retention AUX ledger")
        require(arguments.expected_front_retention_accepted >= 0
                and arguments.expected_front_retention_attempted >=
                    arguments.expected_front_retention_accepted,
                "front-retention AUX counts are invalid")
    _PINS = {
        **files, **manifests, **hashes,
        "expected_head": TARGET_HEAD,
        "checkpoint_output_branch": BRANCH_NAME,
        "current_counters": current,
        "front_retention_ledger": ledger_path,
        "front_retention_ledger_sha256": ledger_sha,
        "front_retention_accepted": arguments.expected_front_retention_accepted,
        "front_retention_attempted": arguments.expected_front_retention_attempted,
        "publication_state_proof_key": arguments.publication_state_proof_key,
    }
    return _PINS


def pins():
    require(isinstance(_PINS, dict), "configure explicit v9 pins before export")
    return _PINS


def v8():
    global _V8, _V8_FRAME_SUMMARY, _V8_DETAIL_PLAN
    if _V8 is not None:
        return _V8
    require(sha256(V8_ADAPTER) == V8_ADAPTER_SHA,
            "reviewed v8 exporter bytes changed; review required")
    spec = importlib.util.spec_from_file_location("_rr_v9_reviewed_v8_media", V8_ADAPTER)
    require(spec is not None and spec.loader is not None,
            "cannot load reviewed v8 exporter")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _V8_FRAME_SUMMARY = module.frame_summary
    _V8_DETAIL_PLAN = module.detail_plan
    _V8 = module
    return module


def display_id():
    return f"CP{pins()['current_counters']['global_policy_decisions']} v9"


def _front_retention(metadata, source):
    branch = metadata.get("rr_capture_transfer_branch") or {}
    ledger = branch.get("front_retention_auxiliary")
    expected_path = pins()["front_retention_ledger"]
    if expected_path is None:
        require(ledger is None,
                "unbound front-retention AUX ledger must not be silently accepted")
        summary = {"present": False, "accepted": 0, "attempted": 0,
                   "separate_from_PPO": True}
    else:
        expected = read_json(expected_path)
        require(ledger == expected and isinstance(ledger, dict)
                and isinstance(ledger.get("schema"), str)
                and isinstance(ledger.get("events"), list),
                "front-retention AUX ledger differs from its exact external receipt")
        accepted = ledger.get("accepted_auxiliary_updates_total")
        attempted = ledger.get("attempted_auxiliary_optimizer_steps_total")
        require(accepted == pins()["front_retention_accepted"]
                and attempted == pins()["front_retention_attempted"],
                "front-retention AUX ledger totals differ from explicit pins")
        summary = {"present": True, "schema": ledger["schema"],
                   "events": len(ledger["events"]), "accepted": accepted,
                   "attempted": attempted, "separate_from_PPO": True,
                   "receipt": str(expected_path),
                   "receipt_sha256": pins()["front_retention_ledger_sha256"]}
    old = ((metadata.get("rr_postcross_workspace_branch") or {})
           .get("front_rehearsal_auxiliary"))
    source_old = ((source.get("rr_postcross_workspace_branch") or {})
                  .get("front_rehearsal_auxiliary"))
    require(old == source_old,
            "historical finite AUX ledger changed across v9 continuation")
    return summary


def checkpoint_identity(manifest):
    """Validate exact v9 migration, branch continuation, and video checkpoint."""
    p = pins()
    metadata = read_json(p["checkpoint_manifest"])
    published = read_json(p["published_manifest"])
    source = read_json(p["migration_source_manifest"])
    parent = read_json(p["expected_parent_checkpoint_manifest"])
    plan = read_json(p["migration_plan"])
    publication = read_json(p["publication"])
    factor = plan.get(FACTOR_KEY) or {}
    receipt = published.get(MIGRATION_KEY) or {}
    require(plan.get("schema") == SCHEMA and factor.get("schema") == SCHEMA
            and receipt.get("schema") == SCHEMA,
            "v9 plan/factor/publication receipt schema differs")
    require(plan.get("source_selection") == SOURCE_SELECTION
            and factor.get("source_selection") == SOURCE_SELECTION
            and factor.get("counter_origin") == V9_ORIGIN,
            "v9 source identity or counter origin differs")
    require(factor.get("source_feedback_revision") == FEEDBACK_REVISION
            and factor.get("target_feedback_revision") == FEEDBACK_REVISION
            and factor.get("observation_shape_changed") is False
            and factor.get("observation_codec_changed") is False
            and factor.get("capture_search_budget_changed") is False
            and factor.get("observation_semantics_changed") == [
                "X409 current envelope armed timing extends to explicitly observed P12 postcapture scope"]
            and factor.get("policy_kernel_changed") is False
            and factor.get("sigma_changed") is False
            and factor.get("caps_changed") is False
            and factor.get("reward_changed") is False
            and factor.get("same_mdp_claimed") is False
            and factor.get("same_numeric_input_policy_mapping_preserved") is True
            and factor.get("existing_branch_origins_preserved") is True
            and factor.get("creates_new_branch") is False,
            "v9 declared semantic boundary differs")
    effective = factor.get("effective_execution_semantics") or {}
    require(effective.get("source_wheel_semantics") == SOURCE_WHEEL_SEMANTICS
            and effective.get("wheel_semantics") == WHEEL_SEMANTICS
            and effective.get("P09_original_path_preserved") is True
            and effective.get("P12_requires_its_own_adjacent_committed_authored_stop") is True
            and effective.get("P12_selected_wheels_require_measured_current_bearing_including_RR_only_when_bearing") is True
            and effective.get("RR_assist_budget_and_state_unchanged") is True
            and effective.get("raw_Gaussian_is_not_transformed_target") is True,
            "v9 postcapture wheel scope differs")
    require(all(factor.get("added_" + name) == 0 for name in
                ("policy_decisions", "ppo_updates", "optimizer_steps", "auxiliary_updates")),
            "v9 migration was assigned learning credit")
    require(receipt.get("plan_sha256") == p["migration_plan_sha256"]
            and Path(receipt.get("plan_path", "")).resolve() == p["migration_plan"]
            and {key: value for key, value in receipt.items()
                 if key not in ("plan_path", "plan_sha256")} == plan
            and published.get("resume_migration") == receipt
            and metadata.get(MIGRATION_KEY) == receipt,
            "v9 checkpoint does not inherit the exact publication receipt")
    preserved = factor.get("preserved_metadata_sha256") or {}
    require(isinstance(preserved, dict) and preserved,
            "v9 factor lacks migration-time preserved metadata bindings")
    for key, digest in preserved.items():
        require(key in source and json_digest(source[key]) == digest
                and published.get(key) == source[key],
                f"v9 zero-update publication changed {key}")
    proof = p["publication_state_proof_key"]
    require(re.fullmatch(r"[A-Za-z0-9_]{1,96}", proof or "") is not None
            and publication.get(proof) is True
            and publication.get("checkpoint_sha256") == p["published_checkpoint_sha256"]
            and publication.get("source_checkpoint_sha256") ==
                p["migration_source_checkpoint_sha256"]
            and publication.get("target_git_commit") == TARGET_HEAD
            and publication.get("save_load_round_trip") is True,
            "v9 official publication proof differs")
    current = p["current_counters"]
    require(all(metadata.get(key) == current[key] for key in COUNTER_KEYS),
            "evaluated checkpoint counters differ from explicit pins")
    learned = {key: current[key] - BRANCH_ORIGIN[key] for key in COUNTER_KEYS}
    require(metadata.get("rr_capture_transfer_branch_counts") == learned,
            "whole RR branch counts do not equal current counters minus original origin")
    route = metadata.get("checkpoint_output_routing")
    require(route == source.get("checkpoint_output_routing")
            and route == parent.get("checkpoint_output_routing")
            and route.get("schema") == "wlr50_clean.checkpoint_output_routing.v1"
            and route.get("branch") == BRANCH_NAME
            and route.get("main_latest_pointer_promotion") is False,
            "v9 branch route changed or permits pointer promotion")
    ancestry = metadata.get("resume_ancestry") or {}
    direct_parent = ancestry.get("source_checkpoint") or {}
    require(direct_parent.get("checkpoint_sha256") ==
                p["expected_parent_checkpoint_sha256"]
            and direct_parent.get("manifest_sha256") ==
                p["expected_parent_checkpoint_manifest_sha256"],
            "evaluated v9 checkpoint did not load its explicitly pinned parent")
    runtime = metadata.get("runtime_contract") or {}
    load = manifest.get("checkpoint_load_provenance") or {}
    load_source = load.get("source") or {}
    require(runtime == manifest.get("runtime_contract")
            and runtime.get("source_git_commit") == TARGET_HEAD
            and metadata.get("policy_contract") == source.get("policy_contract")
            and load.get("policy_contract") == metadata.get("policy_contract")
            and load.get("observation_dimension") == 410
            and load.get("observation_layout") == OBSERVATION_LAYOUT
            and load_source.get("checkpoint_sha256") == p["checkpoint_sha256"]
            and load_source.get("manifest_sha256") == p["checkpoint_manifest_sha256"]
            and manifest.get("saved_global_policy_decisions") ==
                current["global_policy_decisions"]
            and manifest.get("policy_version") == POLICY_VERSION
            and manifest.get("policy_sampling_mode") == "deterministic_conditional_mean",
            "sealed video did not load the exact deterministic v9 checkpoint")
    parameter_hashes = load.get("parameter_hashes") or {}
    require(all(parameter_hashes.get(key) == metadata.get(key) for key in
                ("actor_parameter_sha256", "critic_parameter_sha256",
                 "optimizer_state_sha256", "normalizer_state_sha256")),
            "video load parameter hashes differ from the evaluated checkpoint")
    front_retention = _front_retention(metadata, source)
    return {
        "checkpoint": str(p["checkpoint"]),
        "checkpoint_sha256": p["checkpoint_sha256"],
        "checkpoint_manifest_sha256": p["checkpoint_manifest_sha256"],
        "checkpoint_output_branch": BRANCH_NAME,
        "counters": current,
        "whole_RR_branch_learning": learned,
        "v9_learning": {key: current[key] - V9_ORIGIN[key] for key in COUNTER_KEYS},
        "front_retention_auxiliary": front_retention,
        "control_revision": CONTROL_REVISION,
        "wheel_semantics": WHEEL_SEMANTICS,
    }


def frame_summary(row, native_row):
    out = _V8_FRAME_SUMMARY(row, native_row)
    phase = out.get("phase")
    out.update(
        rr_v9_control_revision=CONTROL_REVISION,
        rr_v9_wheel_semantics=WHEEL_SEMANTICS,
        rr_v9_P12_scope_observed=phase == "P12",
        rr_v9_postcapture_context_observed=(phase == "P12"
            and out.get("rr_placed_history") is True),
    )
    return out


def _last_json_line(path):
    with Path(path).open("rb") as stream:
        stream.seek(0, 2)
        end = stream.tell()
        require(end > 0, "policy decision stream is empty")
        data = b""
        cursor = end
        while cursor > 0 and b"\n" not in data.rstrip(b"\r\n"):
            size = min(65536, cursor)
            cursor -= size
            stream.seek(cursor)
            data = stream.read(size) + data
        line = data.rstrip(b"\r\n").splitlines()[-1]
    return json.loads(line)


def capture_rows(candidate, ledger):
    rows, selected = v8().capture_rows(candidate, ledger)
    decision_path = candidate["source"] / "video_policy_decisions.jsonl"
    terminal = _last_json_line(decision_path)
    step = terminal.get("step_info") or {}
    task = step.get("semantic_task") or {}
    require(terminal.get("decision") == len(rows)
            and step.get("physics_tick") == rows[-1]["tick"]
            and isinstance(task.get("termination_reason"), str),
            "sealed terminal policy decision does not align with video rows")
    rows[-1]["semantic_terminal_reason"] = task["termination_reason"]
    rows[-1]["semantic_terminal_source"] = task.get("termination_source")
    selected.append({
        "milestone": "TERMINAL_SEMANTIC_TASK",
        "decision": terminal["decision"],
        "tick": rows[-1]["tick"],
        "phase": rows[-1]["phase"],
        "termination_reason": task["termination_reason"],
        "termination_source": task.get("termination_source"),
    })
    return rows, selected


def panel_lines(row, result, *, rr_reached, feedback_v2):
    m, f = media(), media()._fmt
    current = pins()["current_counters"]
    credit = {key: current[key] - V9_ORIGIN[key] for key in COUNTER_KEYS}
    retention = checkpoint_identity_cache()["front_retention_auxiliary"]
    retention_text = (f"retention AUX {retention['accepted']}/{retention['attempted']}"
                      if retention["present"] else "retention AUX none")
    return [
        f"{display_id()} DET | PPO + FL/RR ASSISTS + WHEEL v8/v9 + INHERITED AUX | {result}",
        f"v9 PPO +{credit['global_policy_decisions']}/+{credit['ppo_updates']}/+{credit['optimizer_steps']} | {retention_text} (not PPO) | RR/P10/RL={row['run_RR_window_reached']}/{row['run_P10_reached']}/{row['run_RL_window_reached']}",
        f"P12 post-stop wheel retention scope={row['rr_v9_P12_scope_observed']} context={row['rr_v9_postcapture_context_observed']} | P09 v8 path unchanged",
        f"RR {row['rr_assist_mode']} | gap {f(row['rr_post_gap_mm'])}mm depth {f(row['rr_front_distance_mm'])}mm {row['rr_post_contact']} | TOP-ever={row['run_RR_TOP_observed']} placed={row['run_RR_placed']}",
        "controller wheel contribution is not PPO; no front-support repair or RL success is inferred",
        m._wheel_line("source nominal wheel rad/s: ", row["source_nominal_wheel_rad_s"]),
        m._wheel_line("FINAL target canonical rad/s: ", row["wheel_final_canonical_rad_s"]),
        m._wheel_line("MEASURED canonical qdot rad/s: ", row["wheel_actual_canonical_rad_s"]),
        m._wheel_line("controller delta rad/s: ", row["wheel_controller_delta"]),
        f"1x 15fps | t={row['time_s']:.3f}s tick={row['tick']} {row['phase']} | sensor result controls labels",
    ]


_IDENTITY_CACHE = None


def checkpoint_identity_cache():
    require(isinstance(_IDENTITY_CACHE, dict), "checkpoint identity not validated")
    return _IDENTITY_CACHE


def detail_plan(rows):
    plan = dict(_V8_DETAIL_PLAN(rows))
    plan["filename"] = plan["filename"].replace("_v8_", "_v9_")
    plan["title"] = f"DETAIL | {display_id()} | " + plan["title"].split(" | ", 2)[-1]
    return plan


def media():
    global _MEDIA
    if _MEDIA is not None:
        return _MEDIA
    adapter = v8()
    require(adapter._PINS is None,
            "reviewed v8 adapter was configured before v9 compatibility setup")
    adapter._PINS = pins()
    module = adapter.media()
    adapter.frame_summary = frame_summary
    adapter.v7().frame_summary = frame_summary
    adapter.panel_lines = panel_lines
    adapter.detail_plan = detail_plan
    module.checkpoint_identity = _cached_checkpoint_identity
    module.panel_lines = panel_lines
    module.detail_plan = detail_plan
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    node = next(value for value in tree.body
                if isinstance(value, ast.FunctionDef) and value.name == "encode_pair")
    assignments = [value for value in ast.walk(node) if isinstance(value, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "candidate_label"
                for target in value.targets)]
    require(len(assignments) == 1, "comparison label AST differs")
    credit = pins()["current_counters"]["ppo_updates"] - V9_ORIGIN["ppo_updates"]
    assignments[0].value = ast.Constant(
        f"{display_id()} | WHEEL v8/v9 + FL/RR/AUX | v9 PPO +{credit}")
    ast.fix_missing_locations(node)
    exec(compile(ast.Module(body=[node], type_ignores=[]),
                 str(module.__file__) + ":v9_label_only", "exec"), module.__dict__)
    _MEDIA = module
    return module


def _cached_checkpoint_identity(manifest):
    global _IDENTITY_CACHE
    require(_IDENTITY_CACHE is None, "checkpoint identity validated more than once")
    _IDENTITY_CACHE = checkpoint_identity(manifest)
    return _IDENTITY_CACHE


def export(source, destination):
    m = media()
    source = Path(source).resolve(strict=True)
    destination = Path(destination).resolve()
    m.require(destination.is_relative_to(HERE.resolve()) and not destination.exists(),
              "new isolated destination required")
    candidate = m.sealed_source(source, candidate=True)
    baseline = m.sealed_source(m.DEFAULT_HISTORICAL_N, candidate=False)
    helper = m.shared()
    ffmpeg = helper.find_ffmpeg(candidate["capture"].get("full_decode", {}).get(
        "ffmpeg_path"))
    _, ledger, source_validation = helper.checked_media(candidate, ffmpeg=ffmpeg)
    rows, selected = capture_rows(candidate, ledger)
    m.require(0 < len(rows) <= 3000, "source exceeds one normal-speed 200 s episode")
    result, success, acceptance = m.outcome(candidate["manifest"])
    step = pins()["current_counters"]["global_policy_decisions"]
    suffix = "SUCCESS" if success else "INCOMPLETE"
    destination.mkdir(parents=True)
    full = m.encode_full(candidate, rows,
        destination / f"CP{step}_DET_v9_full_attempt_{suffix}.mp4",
        result=result, rr_reached=rows[-1]["run_RR_window_reached"],
        feedback_v2=True, ffmpeg=ffmpeg)
    print("FULL_PLAYABLE_VALIDATED " + full["output"], flush=True)
    detail = m.encode_detail(Path(full["output"]), rows, destination, ffmpeg=ffmpeg)
    pair = m.encode_pair(baseline, {**candidate, "frame_count": len(rows)},
        Path(full["output"]), destination / f"historical_N_vs_CP{step}_DET_v9.mp4",
        historical_version=m.DEFAULT_HISTORICAL_VERSION,
        feedback_v2=True, ffmpeg=ffmpeg)
    evidence = destination / f"CP{step}_v9_selected_evidence.json"
    m.write_new_json(evidence, {
        "schema": "wlr50_clean.rr_postcapture_wheel_v9_video_evidence.v1",
        "source": str(source),
        "selection": "actual phase/contact/wheel/terminal milestones; absent events are not invented",
        "rows": selected,
    })
    identity = checkpoint_identity_cache()
    receipt = {
        "schema": "wlr50_clean.rr_postcapture_wheel_v9_video_export.v1",
        "source": str(source),
        "source_manifest_sha256": m.sha256(candidate["manifest_path"]),
        "source_run_manifest_sha256": m.sha256(candidate["run_manifest_path"]),
        "checkpoint_identity": identity,
        "control_method": METHOD,
        "physical_result": result,
        "physical_task_success": success,
        "semantic_terminal_reason": rows[-1]["semantic_terminal_reason"],
        "semantic_terminal_source": rows[-1]["semantic_terminal_source"],
        "source_acceptance_detail": acceptance,
        "RR_window_reached": rows[-1]["run_RR_window_reached"],
        "P10_reached": rows[-1]["run_P10_reached"],
        "RL_window_reached": rows[-1]["run_RL_window_reached"],
        "full_episode_continuous": True,
        "full_failure_tail_preserved": True,
        "normal_speed": True,
        "single_episode": True,
        "stitched": False,
        "extra_intro_frames": 0,
        "full": full,
        "detail": detail,
        "historical_N_comparison": pair,
        "historical_N_is_fresh_same_controller_B": False,
        "historical_N_freezes_after_its_own_endpoint": True,
        "source_validation": source_validation,
        "selected_evidence": str(evidence),
        "selected_evidence_sha256": m.sha256(evidence),
        "migration_added_learning": False,
        "controller_intervention_is_PPO": False,
        "front_support_repair_claimed": False,
        "missing_events_are_not_invented": True,
        "code_binding": {"adapter_sha256": m.sha256(Path(__file__)),
                         "reviewed_v8_adapter_sha256": V8_ADAPTER_SHA},
    }
    m.write_new_json(destination / "export_receipt.json", receipt)
    print(json.dumps({"physical_result": result, "full": full["output"],
        "detail": detail["output"], "historical_N_pair": pair["output"],
        "receipt": str(destination / "export_receipt.json")}, indent=2))
    return receipt


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument("--source", type=Path, required=True)
    result.add_argument("--destination", type=Path, required=True)
    result.add_argument("--expected-head", required=True)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--checkpoint-sha256", required=True)
    result.add_argument("--checkpoint-manifest-sha256", required=True)
    result.add_argument("--published-v9-checkpoint", type=Path, required=True)
    result.add_argument("--published-checkpoint-sha256", required=True)
    result.add_argument("--published-checkpoint-manifest-sha256", required=True)
    result.add_argument("--migration-plan", type=Path, required=True)
    result.add_argument("--migration-plan-sha256", required=True)
    result.add_argument("--publication", type=Path, required=True)
    result.add_argument("--publication-sha256", required=True)
    result.add_argument("--publication-state-proof-key", required=True)
    result.add_argument("--publisher-script", type=Path, required=True)
    result.add_argument("--publisher-script-sha256", required=True)
    result.add_argument("--migration-source-checkpoint", type=Path, required=True)
    result.add_argument("--migration-source-checkpoint-sha256", required=True)
    result.add_argument("--migration-source-manifest-sha256", required=True)
    result.add_argument("--checkpoint-output-branch", required=True)
    result.add_argument("--expected-parent-checkpoint", type=Path, required=True)
    result.add_argument("--expected-parent-checkpoint-sha256", required=True)
    result.add_argument("--expected-parent-checkpoint-manifest-sha256", required=True)
    result.add_argument("--expected-global-policy-decisions", type=int, required=True)
    result.add_argument("--expected-ppo-updates", type=int, required=True)
    result.add_argument("--expected-optimizer-steps", type=int, required=True)
    result.add_argument("--expected-front-retention-ledger", type=Path)
    result.add_argument("--expected-front-retention-ledger-sha256")
    result.add_argument("--expected-front-retention-accepted", type=int)
    result.add_argument("--expected-front-retention-attempted", type=int)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    configure(arguments)
    export(arguments.source, arguments.destination)
