"""Strict dormant outputs-only exporter for learned v8 branch descendants.

This adapter deliberately rejects the zero-update CP220544 migration ancestor
as an evaluated video checkpoint. It accepts only a real ordinary-PPO child in
the formal ``ancestor220544_signed_wheel_v8`` branch, with explicit immutable
HEAD/checkpoint/manifest/plan/publication/publisher/immediate-parent pins and
actual checkpoint counters. No ``latest`` lookup or planned counter is used.

The v8 controller changes only the existing signed-gap wheel continuity
transform. The RR assist state, feedback revision, 53-degree/45-second budget,
sensor TOP criterion, policy kernel, and task acceptance remain unchanged.
Signed AIR is not contact, bearing, placement, P10/RL progress, or success.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
V7_ADAPTER = HERE / "export_rr_capture_video_v7.py"
V7_ADAPTER_SHA = "b6fad48bf4a72b21be5e980988cf104b5fb393c0497ff5be05af4e33a581ffdd"
PUBLISHER = HERE / "publish_rr_signed_wheel_v8.py"
PUBLISHER_SHA = "4cb0f244b00d533603cbcab02a6ab9321b10de0843a19f1f1549a6f23b4627aa"

SCHEMA = "wlr50_clean.rr_signed_wheel_same410.v8"
FACTOR_KEY = "rr_signed_wheel_v8_factor"
MIGRATION_KEY = "rr_signed_wheel_v8_migration"
PRIOR_MIGRATION_KEY = "rr_signed_contact_v7_migration"
PRIOR_FACTOR_KEY = "rr_signed_contact_v7_factor"
SOURCE_SCHEMA = "wlr50_clean.rr_signed_contact_same410.v7"
SOURCE_HEAD = "60abc00957c0c988da6689e2fb3cfd5c8da22a47"
TARGET_HEAD = "d1871df37d6ea909657511d0e43e7435198f6ccd"
SOURCE_REVISION = "rr_capture_signed_contact_formation_v7"
TARGET_REVISION = "rr_capture_signed_wheel_continuity_v8"
FEEDBACK_REVISION = "signed_band_contact_formation_incremental_v6"
ASSIST_SEARCH = (
    "hip20_knee20_progress_earned_signed_task_band_knee12_then_1deg_"
    "public_peak_gap_le1mm_total53_exposure45_no_recharge_AIR_not_contact_"
    "sensor_TOP_unchanged_captured_issued_N_request_deltas_"
    "RL_current_TOP_retirement"
)
SOURCE_WHEEL_SEMANTICS = (
    "P09_post_source_committed_stop_current_RR_AIR_Q_cross_supported_FL_FR_RL_"
    "nonnegative_depth_gap_floor_previous_FINAL_1p8_slew_TOP_release"
)
WHEEL_SEMANTICS = (
    "P09_post_source_committed_stop_current_RR_AIR_Q_cross_supported_FL_FR_RL_"
    "signed_task_band_nonnegative_depth_gap_floor_previous_FINAL_1p8_slew_TOP_release"
)
BRANCH_NAME = "ancestor220544_signed_wheel_v8"
WHEEL_MODE = "rr_capture_support_forward_projection_v1"
METHOD = (
    "PPO_PLUS_FL_RR_CAPTURE_ASSISTS_PLUS_SIGNED_SUPPORT_WHEEL_V8_"
    "WITH_INHERITED_LIMITED_AUX"
)
COUNTER_KEYS = ("global_policy_decisions", "ppo_updates", "optimizer_steps")
ANCESTOR_COUNTERS = dict(zip(COUNTER_KEYS, (220544, 1688, 33760)))
PUBLISHED_ANCESTOR_SHA = "339c2c779e0a9916b30af1d5ba0e76dfd76708f8d3d23b3284aca56fd5dcf072"
ANCESTOR_SOURCE = {
    "checkpoint_sha256": "47fdec0614ed2683a0eae6fdef736598400f3c6833473b551a8c93f6f0d8f430",
    "manifest_sha256": "a0906988337c97f5d67108c289b4ec46d6497716cac56df39bedf8d9597362a1",
    "counters": ANCESTOR_COUNTERS,
    "source_git_commit": SOURCE_HEAD,
    "source_role": "front_validated_ancestor_control_eval",
}
BRANCH_ROUTE_SOURCE = {
    "checkpoint_sha256": "308122a3c8e760733fbe8370bfdf399a25718148c82333deb41a1265e4e13895",
    "manifest_sha256": "4fbf49e50fe7ab7903462478e45778b79d9cc0f757a99c0127aacb84a5fbb2e8",
    "source_git_commit": "c53119ab332fe048668e66f0543889a128443ca5",
    "counters": ANCESTOR_COUNTERS,
    "source_role": "front_validated_ancestor_control_eval",
}
EXPECTED_EXECUTION = {
    "source_wheel_semantics": SOURCE_WHEEL_SEMANTICS,
    "wheel_semantics": WHEEL_SEMANTICS,
    "configured_capture_band_m": [-.015, .025],
    "gain_clips_negative_gap_to_zero": True,
    "nonnegative_support_floor_retained": True,
    "RR_capture_feedback_unchanged": FEEDBACK_REVISION,
    "RR_assist_budget_and_state_unchanged": True,
    "real_TOP_support_phase_and_source_stop_release_unchanged": True,
    "raw_Gaussian_is_not_transformed_target": True,
}
EXPECTED_ROUTING = {
    "optional_ancestor_only": True,
    "default_output_behavior_changed": False,
    "first_parent_source_requires_published_v8": True,
    "same_branch_resume_and_pointer_verified": True,
    "main_latest_pointer_promotion_authorized": False,
    "network_reset_or_borrowed_credit": False,
}
OBSERVED_TRANSFORM_CHANGE = (
    "already observed signed gap/contact now select forward_floor instead of "
    "release_slew inside the existing band; armed bit meaning unchanged"
)

_V7 = None
_V7_FRAME_SUMMARY = None
_V7_DETAIL_PLAN = None
_BASE_IDENTITY = None
_MEDIA = None
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


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")).hexdigest()


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
            f"{label} is missing or differs from its declared SHA256")
    return path


def configure(arguments):
    """Resolve every mutable identity before loading the reviewed media code."""
    global _PINS
    require(_PINS is None, "v8 exporter pins may be configured only once")
    require(str(arguments.expected_head).lower() == TARGET_HEAD,
            "--expected-head differs from the frozen d1871df v8 runtime")
    require(str(arguments.migration_source_head).lower() == SOURCE_HEAD,
            "--migration-source-head differs from the frozen 60abc v7 runtime")
    require(arguments.migration_source_role == ANCESTOR_SOURCE["source_role"],
            "v8 branch exporter accepts only its registered ancestor source")
    require(arguments.checkpoint_output_branch == BRANCH_NAME,
            "v8 learned video must use the exact formal ancestor branch")
    hashes = {
        name: checked_sha(getattr(arguments, name), "--" + name.replace("_", "-"))
        for name in (
            "checkpoint_sha256", "checkpoint_manifest_sha256",
            "published_checkpoint_sha256",
            "published_checkpoint_manifest_sha256", "migration_plan_sha256",
            "publication_sha256", "migration_source_checkpoint_sha256",
            "migration_source_manifest_sha256", "expected_parent_checkpoint_sha256",
            "expected_parent_checkpoint_manifest_sha256", "publisher_script_sha256",
        )
    }
    require(hashes["published_checkpoint_sha256"] == PUBLISHED_ANCESTOR_SHA,
            "published v8 ancestor checkpoint differs from the formal publication")
    require(hashes["migration_source_checkpoint_sha256"] ==
            ANCESTOR_SOURCE["checkpoint_sha256"]
            and hashes["migration_source_manifest_sha256"] ==
            ANCESTOR_SOURCE["manifest_sha256"],
            "v8 migration source differs from the immutable v7 ancestor registry")
    require(hashes["publisher_script_sha256"] == PUBLISHER_SHA,
            "publisher SHA differs from the frozen reviewed publisher")
    files = {
        "checkpoint": checked_file(arguments.checkpoint,
            hashes["checkpoint_sha256"], "evaluated v8 branch checkpoint"),
        "published_checkpoint": checked_file(arguments.published_v8_checkpoint,
            hashes["published_checkpoint_sha256"], "published v8 ancestor"),
        "migration_plan": checked_file(arguments.migration_plan,
            hashes["migration_plan_sha256"], "v8 migration plan"),
        "publication": checked_file(arguments.publication,
            hashes["publication_sha256"], "v8 publication"),
        "migration_source_checkpoint": checked_file(
            arguments.migration_source_checkpoint,
            hashes["migration_source_checkpoint_sha256"],
            "registered v7 migration source"),
        "publisher_script": checked_file(arguments.publisher_script,
            hashes["publisher_script_sha256"], "v8 publisher script"),
        "expected_parent_checkpoint": checked_file(
            arguments.expected_parent_checkpoint,
            hashes["expected_parent_checkpoint_sha256"],
            "actual loaded v8 branch parent checkpoint"),
    }
    require(files["publisher_script"] == PUBLISHER.resolve(),
            "v8 publisher must be the reviewed outputs-only file")
    published_manifest = sidecar(files["published_checkpoint"]).resolve(strict=True)
    source_manifest = sidecar(files["migration_source_checkpoint"]).resolve(strict=True)
    checkpoint_manifest = sidecar(files["checkpoint"]).resolve(strict=True)
    parent_manifest = sidecar(files["expected_parent_checkpoint"]).resolve(strict=True)
    require(sha256(checkpoint_manifest) == hashes["checkpoint_manifest_sha256"],
            "evaluated v8 branch checkpoint manifest differs from its explicit pin")
    require(sha256(parent_manifest) ==
            hashes["expected_parent_checkpoint_manifest_sha256"],
            "actual loaded parent manifest differs from its explicit pin")
    require(sha256(published_manifest) == hashes["published_checkpoint_manifest_sha256"],
            "published v8 ancestor manifest differs from its explicit pin")
    require(sha256(source_manifest) == hashes["migration_source_manifest_sha256"],
            "registered v7 source manifest differs from its explicit pin")
    current = {
        "global_policy_decisions": arguments.expected_global_policy_decisions,
        "ppo_updates": arguments.expected_ppo_updates,
        "optimizer_steps": arguments.expected_optimizer_steps,
    }
    require(all(type(value) is int and value >= ANCESTOR_COUNTERS[key]
                for key, value in current.items())
            and any(current[key] > ANCESTOR_COUNTERS[key] for key in COUNTER_KEYS),
            "evaluated checkpoint must contain actual post-migration branch learning")
    proof_key = str(arguments.publication_state_proof_key)
    require(re.fullmatch(r"[A-Za-z0-9_]{1,96}", proof_key) is not None,
            "publication state proof key is malformed")
    _PINS = {
        **files, **hashes, "published_manifest": published_manifest,
        "migration_source_manifest": source_manifest,
        "checkpoint_manifest": checkpoint_manifest,
        "expected_parent_checkpoint_manifest": parent_manifest,
        "expected_head": TARGET_HEAD, "migration_source_head": SOURCE_HEAD,
        "migration_source_role": arguments.migration_source_role,
        "checkpoint_role": "ancestor-branch-descendant",
        "checkpoint_output_branch": BRANCH_NAME,
        "current_counters": current,
        "publication_state_proof_key": proof_key,
    }
    return _PINS


def pins():
    require(isinstance(_PINS, dict), "configure explicit v8 pins before export")
    return _PINS


def v7():
    global _V7, _V7_FRAME_SUMMARY, _V7_DETAIL_PLAN, _BASE_IDENTITY
    if _V7 is not None:
        return _V7
    require(sha256(V7_ADAPTER) == V7_ADAPTER_SHA,
            "reviewed v7 exporter bytes changed; review required")
    spec = importlib.util.spec_from_file_location("_rr_v8_reviewed_v7_media", V7_ADAPTER)
    require(spec is not None and spec.loader is not None,
            "cannot load reviewed v7 exporter")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.v6()
    _BASE_IDENTITY = module._V6_IDENTITY
    _V7_FRAME_SUMMARY = module.frame_summary
    _V7_DETAIL_PLAN = module.detail_plan
    _V7 = module
    return module


def display_id():
    step = pins()["current_counters"]["global_policy_decisions"]
    return f"CP{step} BRANCH {BRANCH_NAME} v8"


def _normalized_v8_view(value):
    """Add aliases solely for the already-reviewed common identity checker."""
    out = copy.deepcopy(value)
    factor = out.get(FACTOR_KEY) if isinstance(out, dict) else None
    if isinstance(factor, dict):
        effective = factor.get("effective_execution_semantics")
        if isinstance(effective, dict):
            effective.update({
                "search_semantics": ASSIST_SEARCH,
                "maximum_total_travel_deg": 53,
                "maximum_total_exposure_s": 45,
                "contact_onset_extra_deg": 1,
                "contact_onset_extra_active_s": 1,
                "required_public_progress_peak_maximum_m": .001,
                "budget_recharges": False,
                "sensor_TOP_confirmation_unchanged": True,
                "RL_air_releases_capture": False,
            })
    receipt = out.get(MIGRATION_KEY) if isinstance(out, dict) else None
    if isinstance(receipt, dict):
        out[MIGRATION_KEY] = _normalized_v8_view(receipt)
    resume = out.get("resume_migration") if isinstance(out, dict) else None
    if isinstance(resume, dict) and resume.get("schema") == SCHEMA:
        out["resume_migration"] = _normalized_v8_view(resume)
    route = out.get("checkpoint_output_routing") if isinstance(out, dict) else None
    if (isinstance(route, dict) and route.get("branch") == BRANCH_NAME
            and route.get("main_latest_pointer_promotion") is False):
        # The raw route was already checked byte-for-byte against the pinned
        # direct-parent manifest.  The v6 common checker predates this branch
        # and expects the current migration source in this slot, so present a
        # compatibility alias without changing the raw strict validation.
        route["source_selection"] = copy.deepcopy(ANCESTOR_SOURCE)
    return out


def checkpoint_identity(manifest):
    """Validate v8 wheel scope, then the shared full-state/ancestry invariants."""
    p, m, prior = pins(), media(), v7()
    raw_metadata = read_json(sidecar(p["checkpoint"]))
    raw_published = read_json(p["published_manifest"])
    raw_source = read_json(p["migration_source_manifest"])
    raw_plan = read_json(p["migration_plan"])
    factor = raw_plan.get(FACTOR_KEY) or {}
    receipt = raw_published.get(MIGRATION_KEY) or {}
    old_receipt = raw_source.get(PRIOR_MIGRATION_KEY) or {}
    require(raw_plan.get("schema") == SCHEMA and factor.get("schema") == SCHEMA
            and receipt.get("schema") == SCHEMA,
            "v8 plan/factor/publication receipt schema differs")
    require(factor.get("source_selection") == ANCESTOR_SOURCE
            and raw_plan.get("source_selection") == ANCESTOR_SOURCE
            and factor.get("counter_origin") == ANCESTOR_COUNTERS,
            "v8 immutable source selection or counter origin differs")
    require(factor.get("source_feedback_revision") == FEEDBACK_REVISION
            and factor.get("target_feedback_revision") == FEEDBACK_REVISION
            and factor.get("observation_semantics_changed") == []
            and factor.get("existing_observed_state_controls_changed_transform") ==
                OBSERVED_TRANSFORM_CHANGE
            and factor.get("effective_execution_semantics") == EXPECTED_EXECUTION
            and factor.get("checkpoint_output_routing") == EXPECTED_ROUTING,
            "v8 signed-wheel continuity scope differs")
    exact_flags = {
        "observation_shape_changed": False,
        "observation_codec_changed": False,
        "capture_search_budget_changed": False,
        "legacy_action_transform_is_current_execution_semantics": False,
        "controller_transition_semantics_changed": True,
        "same_mdp_claimed": False,
        "same_numeric_input_policy_mapping_preserved": True,
        "same_physical_state_action_equivalence_claimed": False,
        "policy_kernel_changed": False,
        "sigma_changed": False,
        "caps_changed": False,
        "reward_changed": False,
        "physical_dynamics_changed": False,
        "physical_task_acceptance_rules_changed": False,
        "existing_branch_origins_preserved": True,
        "creates_new_branch": False,
        "discard_old_rollout_storage": True,
        "old_rollout_is_new_MDP_onpolicy": False,
        "physical_state_inherited": False,
        "candidate_evaluation_only": False,
        "ancestor_training_requires_explicit_output_branch": True,
        "latest_learned_weights_preserved": False,
        "latest_learned_policy_equivalence_claimed": False,
        "latest_pointer_promotion_authorized": False,
    }
    require(all(factor.get(key) is value for key, value in exact_flags.items())
            and all(factor.get("added_" + name) == 0 for name in
                    ("policy_decisions", "ppo_updates", "optimizer_steps",
                     "auxiliary_updates")),
            "v8 semantic scope, source role, or zero-credit migration flags differ")
    require(factor.get("source_v7_receipt_sha256") == json_digest(old_receipt)
            and factor.get("source_resume_migration_sha256") ==
                json_digest(raw_source.get("resume_migration")),
            "v8 source receipt/resume binding differs")
    old_factor = old_receipt.get(PRIOR_FACTOR_KEY) or {}
    require(old_receipt.get("schema") == SOURCE_SCHEMA
            and old_receipt.get("target_git_commit") == SOURCE_HEAD
            and old_factor.get("counter_origin") == ANCESTOR_COUNTERS
            and old_factor.get("target_feedback_revision") == FEEDBACK_REVISION
            and old_receipt.get("source_selection", {}).get("source_role") ==
                ANCESTOR_SOURCE["source_role"],
            "v8 source is not the exact formally published v7 ancestor")
    require(raw_metadata.get(MIGRATION_KEY) == receipt,
            "evaluated branch checkpoint does not inherit the exact v8 receipt")
    ancestry = raw_metadata.get("resume_ancestry") or {}
    parent = ancestry.get("source_checkpoint") or {}
    require(parent.get("checkpoint_sha256") ==
                p["expected_parent_checkpoint_sha256"],
            "v8 branch checkpoint did not load the explicitly pinned parent")
    # Ordinary saves retain the branch's original routing source, which predates
    # the v8 migration ancestor.  Bind that immutable historical route through
    # the explicitly SHA-pinned immediate-parent manifest instead of replacing
    # it with the v5/v7 migration source identity.
    parent_metadata = read_json(p["expected_parent_checkpoint_manifest"])
    route = raw_metadata.get("checkpoint_output_routing")
    parent_route = parent_metadata.get("checkpoint_output_routing")
    require(route == parent_route and isinstance(route, dict),
            "v8 branch checkpoint route differs from its pinned direct parent")
    require(route.get("schema") == "wlr50_clean.checkpoint_output_routing.v1"
            and route.get("branch") == BRANCH_NAME
            and route.get("output_root") ==
                str((HERE / "branches" / BRANCH_NAME).resolve())
            and route.get("main_latest_pointer_promotion") is False,
            "v8 branch checkpoint route is not the isolated no-promotion route")
    route_source = route.get("source_selection") or {}
    require(route_source.get("source_role") ==
                ANCESTOR_SOURCE["source_role"]
            and route_source.get("counters") == ANCESTOR_COUNTERS
            and route_source.get("registry_bridge_required") is False,
            "v8 branch route source is not the preserved front ancestor route")

    original_read = m.read_json
    m.read_json = lambda path: _normalized_v8_view(original_read(path))
    try:
        inherited = _BASE_IDENTITY(manifest)
    finally:
        m.read_json = original_read
    for key in ("v6_plan", "v6_plan_sha256", "v6_publication",
                "v6_publication_sha256"):
        inherited.pop(key, None)
    inherited.update({
        "rr_capture_control_revision": TARGET_REVISION,
        "rr_capture_feedback_revision": FEEDBACK_REVISION,
        "v8_wheel_semantics": WHEEL_SEMANTICS,
        "v8_plan": str(p["migration_plan"]),
        "v8_plan_sha256": p["migration_plan_sha256"],
        "v8_publication": str(p["publication"]),
        "v8_publication_sha256": p["publication_sha256"],
        "migration_added_policy_decisions": 0,
        "migration_added_PPO_updates": 0,
        "migration_added_optimizer_steps": 0,
        "signed_AIR_permission_is_contact_or_bearing": False,
        "weak_contact_controller_added": False,
    })
    return inherited


def frame_summary(row, native_row):
    out = _V7_FRAME_SUMMARY(row, native_row)
    gap_mm = float(out["rr_post_gap_mm"])
    out.update(
        rr_v8_control_revision=TARGET_REVISION,
        rr_v8_wheel_semantics=WHEEL_SEMANTICS,
        rr_v8_current_gap_in_signed_band=-15.0 <= gap_mm <= 25.0,
        rr_v8_observed_wheel_action=out["wheel_action"],
        rr_v8_observed_wheel_envelope_active=out["wheel_envelope_active"],
        rr_v8_observed_wheel_projection_changed=out["wheel_projection_changed"],
    )
    return out


def capture_rows(candidate, ledger):
    rows, selected = v7().capture_rows(candidate, ledger)
    observed = next((value for value in rows
        if value["phase"] == "P09" and value["rr_post_contact"] == "AIR"
        and value["rr_v8_current_gap_in_signed_band"]
        and value["rr_v8_observed_wheel_envelope_active"]), None)
    if observed is not None:
        selected.append({"milestone": "FIRST_V8_SIGNED_WHEEL_CONTEXT", **observed})
    return rows, selected


def panel_lines(row, result, *, rr_reached, feedback_v2):
    m, f = media(), media()._fmt
    credit = {key: pins()["current_counters"][key] - ANCESTOR_COUNTERS[key]
              for key in COUNTER_KEYS}
    step = pins()["current_counters"]["global_policy_decisions"]
    assist_reason = str(row["rr_assist_reason"])
    if len(assist_reason) > 42:
        assist_reason = assist_reason[:39] + "..."
    return [
        f"CP{step} v8 DET | PPO + FL/RR ASSISTS + SIGNED WHEEL v8 + INHERITED LIMITED AUX | {result}",
        f"branch {BRANCH_NAME} | migration +0 | learned +{credit['global_policy_decisions']}/+{credit['ppo_updates']}/+{credit['optimizer_steps']} | RR/P10/RL={row['run_RR_window_reached']}/{row['run_P10_reached']}/{row['run_RL_window_reached']}",
        f"RR {row['rr_assist_mode']} mode={row['rr_v7_mode_index']} reason={assist_reason} | unchanged budget | travel={f(row['rr_v7_total_travel_deg'])}deg",
        f"RR gap {f(row['rr_post_gap_mm'])}mm depth {f(row['rr_front_distance_mm'])}mm {row['rr_post_contact']} | signed-band={row['rr_v8_current_gap_in_signed_band']} TOP-ever={row['run_RR_TOP_observed']} placed={row['run_RR_placed']}",
        f"wheel-v8 action={row['rr_v8_observed_wheel_action']} envelope={row['rr_v8_observed_wheel_envelope_active']} changed={row['rr_v8_observed_wheel_projection_changed']} gain={f(row['wheel_gain'])} | non-PPO controller",
        f"signed AIR is not contact/bearing; no weak-contact controller | P10={row['run_P10_reached']} RL={row['run_RL_window_reached']} RL-placed={row['run_RL_placed']}",
        m._wheel_line("source nominal wheel rad/s: ", row["source_nominal_wheel_rad_s"]),
        m._wheel_line("FINAL target canonical rad/s: ", row["wheel_final_canonical_rad_s"]),
        m._wheel_line("MEASURED canonical qdot rad/s: ", row["wheel_actual_canonical_rad_s"]),
        m._wheel_line("wheel-v8 controller delta rad/s: ", row["wheel_controller_delta"]),
        f"1x 15fps | t={row['time_s']:.3f}s tick={row['tick']} {row['phase']} | current contact != placed history; sensor result controls success labels",
    ]


def detail_plan(rows):
    plan = dict(_V7_DETAIL_PLAN(rows))
    plan["filename"] = plan["filename"].replace("_v7_", "_v8_")
    plan["title"] = f"DETAIL | {display_id()} | " + plan["title"].split(" | ", 2)[-1]
    return plan


def media():
    global _MEDIA
    if _MEDIA is not None:
        return _MEDIA
    prior = v7()
    prior._PINS = pins()
    module = prior.media()
    base = prior.v6()
    base._PINS = pins()
    base.MIGRATION_KEY = MIGRATION_KEY
    base.FACTOR_KEY = FACTOR_KEY
    base.PRIOR_MIGRATION_KEY = PRIOR_MIGRATION_KEY
    base.SCHEMA = SCHEMA
    base.TARGET_REVISION = TARGET_REVISION
    base.SOURCE_FEEDBACK = FEEDBACK_REVISION
    base.REVISION = FEEDBACK_REVISION
    base.SEARCH = ASSIST_SEARCH
    base.ANCESTOR_COUNTERS = ANCESTOR_COUNTERS
    base.ANCESTOR_SOURCE = ANCESTOR_SOURCE
    base.display_id = display_id
    base.checkpoint_identity = checkpoint_identity
    prior.frame_summary = frame_summary
    prior.panel_lines = panel_lines
    prior.detail_plan = detail_plan
    module.checkpoint_identity = checkpoint_identity
    module.validate_historical_rr_snapshot = prior.validate_snapshot
    module.FEEDBACK_V2_REVISION = FEEDBACK_REVISION
    module.panel_lines = panel_lines
    module.detail_plan = detail_plan

    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    node = next(value for value in tree.body
                if isinstance(value, ast.FunctionDef) and value.name == "encode_pair")
    assignments = [value for value in ast.walk(node) if isinstance(value, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "candidate_label"
                for target in value.targets)]
    require(len(assignments) == 1, "comparison label AST differs")
    credit = {key: pins()["current_counters"][key] - ANCESTOR_COUNTERS[key]
              for key in COUNTER_KEYS}
    assignments[0].value = ast.Constant(
        f"{display_id()} | SIGNED-WHEEL v8 + FL/RR/AUX | "
        f"actual branch PPO +{credit['ppo_updates']}")
    ast.fix_missing_locations(node)
    exec(compile(ast.Module(body=[node], type_ignores=[]),
                 str(module.__file__) + ":v8_label_only", "exec"), module.__dict__)
    _MEDIA = module
    return module


def export(source, destination):
    m = media()
    source, destination = Path(source).resolve(strict=True), Path(destination).resolve()
    m.require(destination.is_relative_to(HERE.resolve()) and not destination.exists(),
              "new isolated destination required")
    candidate = m.sealed_source(source, candidate=True)
    baseline = m.sealed_source(m.DEFAULT_HISTORICAL_N, candidate=False)
    helper = m.shared()
    ffmpeg = helper.find_ffmpeg(candidate["capture"].get("full_decode", {}).get(
        "ffmpeg_path"))
    _, ledger, source_validation = helper.checked_media(candidate, ffmpeg=ffmpeg)
    rows, selected = capture_rows(candidate, ledger)
    m.require(0 < len(rows) <= 3000, "full source must remain <=200 s at native 15fps")
    result, success, source_acceptance_detail = m.outcome(candidate["manifest"])
    step = pins()["current_counters"]["global_policy_decisions"]
    suffix = "SUCCESS" if success else "INCOMPLETE"
    destination.mkdir(parents=True)
    full = m.encode_full(candidate, rows,
        destination / f"CP{step}_DET_v8_full_attempt_{suffix}.mp4",
        result=result, rr_reached=rows[-1]["run_RR_window_reached"],
        feedback_v2=True, ffmpeg=ffmpeg)
    print("FULL_PLAYABLE_VALIDATED " + full["output"], flush=True)
    detail = m.encode_detail(Path(full["output"]), rows, destination, ffmpeg=ffmpeg)
    pair = m.encode_pair(baseline, {**candidate, "frame_count": len(rows)},
        Path(full["output"]), destination / f"historical_N_vs_CP{step}_DET_v8.mp4",
        historical_version=m.DEFAULT_HISTORICAL_VERSION,
        feedback_v2=True, ffmpeg=ffmpeg)
    evidence = destination / f"CP{step}_v8_selected_fourwheel_evidence.json"
    m.write_new_json(evidence, {
        "schema": "wlr50_clean.rr_signed_wheel_v8_video_evidence.v1",
        "source": str(source),
        "capture_assist_ticks_sha256": m.sha256(candidate["tick_path"]),
        "native_tick_audit_sha256": m.sha256(candidate["native_tick_path"]),
        "selection": "actual v8/RR/P10/RL/wheel/contact milestones plus terminal; absent events are not invented",
        "pre_and_post_ticks_separate": True,
        "signed_AIR_permission_is_contact_or_bearing": False,
        "weak_contact_controller_added": False,
        "rows": selected,
    })
    credit = {key: pins()["current_counters"][key] - ANCESTOR_COUNTERS[key]
              for key in COUNTER_KEYS}
    receipt = {
        "schema": "wlr50_clean.rr_signed_wheel_v8_video_export.v1",
        "source": str(source),
        "source_manifest_sha256": m.sha256(candidate["manifest_path"]),
        "source_run_manifest_sha256": m.sha256(candidate["run_manifest_path"]),
        "checkpoint": candidate["checkpoint"],
        "checkpoint_role": "ancestor-branch-descendant",
        "checkpoint_output_branch": BRANCH_NAME,
        "control_method": METHOD,
        "source_legacy_control_method": candidate["manifest"].get("control_method"),
        "effective_wheel_mode": WHEEL_MODE,
        "v8_wheel_semantics": WHEEL_SEMANTICS,
        "RR_assist_feedback_revision": FEEDBACK_REVISION,
        "historical_ancestor_weight_counters": ANCESTOR_COUNTERS,
        "actual_RR_branch_learning": credit,
        "evaluation_added_PPO_updates": 0,
        "migration_added_optimization": 0,
        "inherited_limited_AUX_is_PPO": False,
        "controller_intervention_is_PPO_learning": False,
        "same_MDP_claimed": False,
        "signed_AIR_permission_is_contact_or_bearing": False,
        "weak_contact_controller_added": False,
        "physical_result": result,
        "physical_task_success": success,
        "semantic_terminal_reason": rows[-1].get("semantic_terminal_reason"),
        "semantic_terminal_source": rows[-1].get("semantic_terminal_source"),
        "source_acceptance_detail": source_acceptance_detail,
        "source_acceptance_error": candidate["manifest"].get("source_acceptance_error"),
        "RR_window_reached": rows[-1]["run_RR_window_reached"],
        "RR_TOP_observed": rows[-1]["run_RR_TOP_observed"],
        "RR_placed": rows[-1]["run_RR_placed"],
        "P10_reached": rows[-1]["run_P10_reached"],
        "RL_window_reached": rows[-1]["run_RL_window_reached"],
        "RL_placed": rows[-1]["run_RL_placed"],
        "RL_success_claimed": success and rows[-1]["run_RL_placed"],
        "attempt_classification": ("TASK_SUCCESS" if success else
            "RL_WINDOW_REACHED_INCOMPLETE" if rows[-1]["run_RL_window_reached"] else
            "P10_REACHED_RL_NOT_REACHED" if rows[-1]["run_P10_reached"] else
            "RR_WINDOW_ONLY" if rows[-1]["run_RR_window_reached"] else
            "PREDECESSOR_ONLY"),
        "full_episode_continuous": True,
        "full_failure_tail_preserved": True,
        "normal_speed": True,
        "single_episode": True,
        "stitched": False,
        "extra_intro_frames": 0,
        "source_container_duration_is_not_physics_duration": True,
        "derived_output_PTS_rebuilt_from_decoded_frame_order_at_15fps": True,
        "source_validation": source_validation,
        "full": full,
        "detail": detail,
        "historical_N_comparison": pair,
        "historical_N_is_fresh_same_controller_B": False,
        "historical_N_freezes_after_its_own_endpoint": True,
        "selected_fourwheel_evidence": str(evidence),
        "selected_fourwheel_evidence_sha256": m.sha256(evidence),
        "explicit_cli_binding": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in pins().items() if key != "current_counters"
        },
        "expected_checkpoint_counters": pins()["current_counters"],
        "code_binding": {
            "adapter_sha256": m.sha256(Path(__file__)),
            "v7_media_adapter_sha256": V7_ADAPTER_SHA,
            "v8_publisher_sha256": PUBLISHER_SHA,
        },
        "missing_events_are_not_invented": True,
        "migration_ancestor_is_not_labeled_as_new_learning": True,
        "targets_are_not_actual_velocity": True,
    }
    m.write_new_json(destination / "export_receipt.json", receipt)
    print(json.dumps({"physical_result": result, "full": full["output"],
        "detail": detail["output"], "historical_N_pair": pair["output"],
        "receipt": str(destination / "export_receipt.json")}, indent=2))
    return receipt


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument("--source", type=Path, required=True,
                        help="exact sealed v8 learned-branch semantic-video source")
    result.add_argument("--destination", type=Path, required=True)
    result.add_argument("--expected-head", required=True)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--checkpoint-sha256", required=True)
    result.add_argument("--checkpoint-manifest-sha256", required=True)
    result.add_argument("--published-v8-checkpoint", type=Path, required=True)
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
    result.add_argument("--migration-source-head", required=True)
    result.add_argument("--migration-source-role", required=True)
    result.add_argument("--checkpoint-output-branch", required=True)
    result.add_argument("--expected-parent-checkpoint", type=Path, required=True)
    result.add_argument("--expected-parent-checkpoint-sha256", required=True)
    result.add_argument("--expected-parent-checkpoint-manifest-sha256", required=True)
    result.add_argument("--expected-global-policy-decisions", type=int, required=True)
    result.add_argument("--expected-ppo-updates", type=int, required=True)
    result.add_argument("--expected-optimizer-steps", type=int, required=True)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    configure(arguments)
    export(arguments.source, arguments.destination)
