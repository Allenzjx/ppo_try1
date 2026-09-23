"""Dormant strict outputs-only exporter for signed-AIR RR v7 evaluations.

This adapter is intentionally unusable without explicit immutable CLI pins.  It
loads the reviewed v6 media renderer by SHA, validates the formal v7 migration
publication (or an explicitly routed ordinary-PPO descendant), and derives all
contact, placement, P10, RL, and success labels from the sealed episode.  The
signed AIR band is permission for the existing bounded action only: it is not a
contact signal, weak-contact controller, bearing event, or physical success.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import math
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
V6_ADAPTER = HERE / "export_rr_capture_video_v6.py"
V6_ADAPTER_SHA = "870c1085c933f0d4360d20f0c08e3851c2f7421482993e370cdba18a06676c62"

MIGRATION_KEY = "rr_signed_contact_v7_migration"
FACTOR_KEY = "rr_signed_contact_v7_factor"
PRIOR_MIGRATION_KEY = "rr_contact_onset_v6_migration"
PRIOR_FACTOR_KEY = "rr_contact_onset_v6_factor"
SCHEMA = "wlr50_clean.rr_signed_contact_same410.v7"
SOURCE_SCHEMA = "wlr50_clean.rr_contact_onset_same410.v6"
SOURCE_HEAD = "97c4367ee293cb4dd191944ef663bbda63a5b235"
SOURCE_REVISION = "rr_capture_contact_onset_v6"
TARGET_REVISION = "rr_capture_signed_contact_formation_v7"
SOURCE_FEEDBACK = "progress_reserve_contact_onset_incremental_v5"
REVISION = "signed_band_contact_formation_incremental_v6"
SEARCH = (
    "hip20_knee20_progress_earned_signed_task_band_knee12_then_1deg_"
    "public_peak_gap_le1mm_total53_exposure45_no_recharge_AIR_not_contact_"
    "sensor_TOP_unchanged_captured_issued_N_request_deltas_"
    "RL_current_TOP_retirement"
)
WHEEL_MODE = "rr_capture_support_forward_projection_v1"
METHOD = (
    "PPO_PLUS_FL_RR_CAPTURE_ASSISTS_PLUS_SUPPORT_WHEEL_V4_"
    "WITH_INHERITED_LIMITED_AUX"
)
COUNTER_KEYS = ("global_policy_decisions", "ppo_updates", "optimizer_steps")
ANCESTOR_COUNTERS = dict(zip(COUNTER_KEYS, (220544, 1688, 33760)))
ANCESTOR_SOURCE = {
    "checkpoint_sha256": "0a61fb608210828ff9b3edb95701a25dd8232f2a8e0c207179ea5ae6bd142e01",
    "manifest_sha256": "5672a9446485d09da35aae818dc010fc128d2334165194c2cad773d74ac3fc4c",
    "counters": ANCESTOR_COUNTERS,
    "source_git_commit": SOURCE_HEAD,
    "source_role": "front_validated_ancestor_control_eval",
}
# Ordinary PPO descendants retain the pre-v6 branch router's original c531
# selection.  This is intentionally distinct from the immediate v7 migration
# source above and is validated before the common v6 checker receives an alias.
BRANCH_ROUTE_SOURCE = {
    "checkpoint_sha256": "308122a3c8e760733fbe8370bfdf399a25718148c82333deb41a1265e4e13895",
    "manifest_sha256": "4fbf49e50fe7ab7903462478e45778b79d9cc0f757a99c0127aacb84a5fbb2e8",
    "source_git_commit": "c53119ab332fe048668e66f0543889a128443ca5",
    "counters": ANCESTOR_COUNTERS,
    "source_role": "front_validated_ancestor_control_eval",
}
EXPECTED_EXECUTION = {
    "search_semantics": SEARCH,
    "configured_capture_band_m": [-.015, .025],
    "AIR_is_not_contact_or_bearing": True,
    "maximum_total_travel_deg": 53,
    "maximum_total_exposure_s": 45,
    "signed_contact_extra_deg": 1,
    "signed_contact_extra_active_s": 1,
    "required_public_progress_peak_maximum_m": .001,
    "budget_recharges": False,
    "sensor_TOP_confirmation_unchanged": True,
    "RL_air_releases_capture": False,
    "raw_Gaussian_is_not_transformed_target": True,
}
EXPECTED_ROUTING = {
    "optional_ancestor_only": True,
    "default_output_behavior_changed": False,
    "first_parent_source_requires_published_v7": True,
    "same_branch_resume_and_pointer_verified": True,
    "main_latest_pointer_promotion_authorized": False,
    "network_reset_or_borrowed_credit": False,
}

_V6 = None
_V6_IDENTITY = None
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
    """Resolve all mutable identities before loading the reviewed renderer."""
    global _PINS
    require(_PINS is None, "v7 exporter pins may be configured only once")
    head = str(arguments.expected_head).lower()
    source_head = str(arguments.migration_source_head).lower()
    require(re.fullmatch(r"[0-9a-f]{40}", head) is not None,
            "--expected-head must be the exact frozen v7 40-hex commit")
    require(source_head == SOURCE_HEAD,
            "--migration-source-head must bind the registered 97c4367 v6 runtime")
    require(arguments.migration_source_role == ANCESTOR_SOURCE["source_role"],
            "v7 ancestor exporter accepts only the registered front-validated source role")
    hashes = {
        name: checked_sha(getattr(arguments, name), "--" + name.replace("_", "-"))
        for name in (
            "checkpoint_sha256", "published_checkpoint_sha256",
            "published_checkpoint_manifest_sha256", "migration_plan_sha256",
            "publication_sha256", "migration_source_checkpoint_sha256",
            "migration_source_manifest_sha256",
        )
    }
    require(hashes["migration_source_checkpoint_sha256"] ==
            ANCESTOR_SOURCE["checkpoint_sha256"]
            and hashes["migration_source_manifest_sha256"] ==
            ANCESTOR_SOURCE["manifest_sha256"],
            "v7 ancestor source checkpoint/manifest differ from the immutable registry")
    files = {
        "checkpoint": checked_file(arguments.checkpoint,
            hashes["checkpoint_sha256"], "evaluated checkpoint"),
        "published_checkpoint": checked_file(arguments.published_v7_checkpoint,
            hashes["published_checkpoint_sha256"], "published v7 checkpoint"),
        "migration_plan": checked_file(arguments.migration_plan,
            hashes["migration_plan_sha256"], "v7 migration plan"),
        "publication": checked_file(arguments.publication,
            hashes["publication_sha256"], "v7 publication"),
        "migration_source_checkpoint": checked_file(
            arguments.migration_source_checkpoint,
            hashes["migration_source_checkpoint_sha256"],
            "registered v6 migration source"),
    }
    published_manifest = sidecar(files["published_checkpoint"]).resolve(strict=True)
    source_manifest = sidecar(files["migration_source_checkpoint"]).resolve(strict=True)
    require(sha256(published_manifest) == hashes["published_checkpoint_manifest_sha256"],
            "published v7 checkpoint manifest differs from its explicit pin")
    require(sha256(source_manifest) == hashes["migration_source_manifest_sha256"],
            "registered v6 source manifest differs from its explicit pin")
    current = {
        "global_policy_decisions": arguments.expected_global_policy_decisions,
        "ppo_updates": arguments.expected_ppo_updates,
        "optimizer_steps": arguments.expected_optimizer_steps,
    }
    require(all(type(value) is int and value >= ANCESTOR_COUNTERS[key]
                for key, value in current.items()),
            "declared checkpoint counters cannot precede the v7 ancestor")
    branch = arguments.checkpoint_output_branch
    parent_sha = arguments.expected_parent_checkpoint_sha256
    if arguments.checkpoint_role == "published-ancestor-zero-learning":
        require(branch is None and parent_sha is None
                and files["checkpoint"] == files["published_checkpoint"]
                and hashes["checkpoint_sha256"] == hashes["published_checkpoint_sha256"]
                and current == ANCESTOR_COUNTERS,
                "zero-learning v7 ancestor must be the exact publication")
    else:
        require(isinstance(branch, str)
                and re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", branch) is not None,
                "v7 branch descendant requires its exact safe output-branch name")
        parent_sha = checked_sha(parent_sha, "--expected-parent-checkpoint-sha256")
        require(any(current[key] > ANCESTOR_COUNTERS[key] for key in COUNTER_KEYS),
                "v7 branch descendant must carry real counters beyond its ancestor")
    proof_key = str(arguments.publication_state_proof_key)
    require(re.fullmatch(r"[A-Za-z0-9_]{1,96}", proof_key) is not None,
            "publication state proof key is malformed")
    _PINS = {
        **files, **hashes, "published_manifest": published_manifest,
        "migration_source_manifest": source_manifest, "expected_head": head,
        "migration_source_head": source_head,
        "migration_source_role": arguments.migration_source_role,
        "checkpoint_role": arguments.checkpoint_role,
        "checkpoint_output_branch": branch,
        "expected_parent_checkpoint_sha256": parent_sha,
        "current_counters": current,
        "publication_state_proof_key": proof_key,
    }
    return _PINS


def pins():
    require(isinstance(_PINS, dict), "configure explicit v7 pins before export")
    return _PINS


def v6():
    global _V6, _V6_IDENTITY
    if _V6 is not None:
        return _V6
    require(sha256(V6_ADAPTER) == V6_ADAPTER_SHA,
            "reviewed v6 exporter bytes changed; review required")
    spec = importlib.util.spec_from_file_location("_rr_v7_reviewed_v6_media", V6_ADAPTER)
    require(spec is not None and spec.loader is not None,
            "cannot load reviewed v6 exporter")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _V6_IDENTITY = module.checkpoint_identity
    _V6 = module
    return module


def display_id():
    p = pins()
    step = p["current_counters"]["global_policy_decisions"]
    return (f"CP{step} ANCESTOR v7" if p["checkpoint_role"] ==
            "published-ancestor-zero-learning" else
            f"CP{step} BRANCH {p['checkpoint_output_branch']} v7")


def _normalized_v7_view(value):
    """Supply only the v6 common-validator aliases; raw v7 is checked first."""
    out = copy.deepcopy(value)
    factor = out.get(FACTOR_KEY) if isinstance(out, dict) else None
    if isinstance(factor, dict):
        effective = factor.get("effective_execution_semantics")
        if isinstance(effective, dict):
            effective["contact_onset_extra_deg"] = effective.get("signed_contact_extra_deg")
            effective["contact_onset_extra_active_s"] = effective.get(
                "signed_contact_extra_active_s")
    receipt = out.get(MIGRATION_KEY) if isinstance(out, dict) else None
    if isinstance(receipt, dict):
        out[MIGRATION_KEY] = _normalized_v7_view(receipt)
    resume = out.get("resume_migration") if isinstance(out, dict) else None
    if isinstance(resume, dict) and resume.get("schema") == SCHEMA:
        out["resume_migration"] = _normalized_v7_view(resume)
    route = out.get("checkpoint_output_routing") if isinstance(out, dict) else None
    if isinstance(route, dict) and route.get("source_selection") == BRANCH_ROUTE_SOURCE:
        route["source_selection"] = copy.deepcopy(ANCESTOR_SOURCE)
    return out


def checkpoint_identity(manifest):
    """Validate raw v7-specific scope, then reuse strict common v6 checks."""
    p, m, base = pins(), media(), v6()
    raw_metadata = read_json(sidecar(p["checkpoint"]))
    raw_published = read_json(p["published_manifest"])
    raw_source = read_json(p["migration_source_manifest"])
    raw_plan = read_json(p["migration_plan"])
    factor = raw_plan.get(FACTOR_KEY) or {}
    receipt = raw_published.get(MIGRATION_KEY) or {}
    prior = raw_source.get(PRIOR_MIGRATION_KEY) or {}
    require(raw_plan.get("schema") == SCHEMA and factor.get("schema") == SCHEMA
            and receipt.get("schema") == SCHEMA,
            "v7 plan/factor/publication receipt schema differs")
    require(factor.get("source_selection") == ANCESTOR_SOURCE
            and raw_plan.get("source_selection") == ANCESTOR_SOURCE
            and factor.get("counter_origin") == ANCESTOR_COUNTERS,
            "v7 immutable ancestor selection or counter origin differs")
    require(factor.get("source_feedback_revision") == SOURCE_FEEDBACK
            and factor.get("target_feedback_revision") == REVISION
            and factor.get("effective_execution_semantics") == EXPECTED_EXECUTION,
            "v7 signed-AIR execution semantics differ")
    require(factor.get("observation_semantics_changed") == [
                "existing RR recovery permission now includes configured signed AIR capture band; no contact/support/placement inference"]
            and factor.get("checkpoint_output_routing") == EXPECTED_ROUTING,
            "v7 action scope or branch routing declaration differs")
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
            "v7 semantic scope, source role, or zero-credit migration flags differ")
    require(factor.get("source_v6_receipt_sha256") == json_digest(prior)
            and factor.get("source_resume_migration_sha256") ==
                json_digest(raw_source.get("resume_migration"))
            and factor.get("existing_branch_origins_preserved") is True,
            "v7 source receipt/ancestry preservation binding differs")
    prior_factor = prior.get(PRIOR_FACTOR_KEY) or {}
    require(prior.get("schema") == SOURCE_SCHEMA
            and prior.get("target_git_commit") == SOURCE_HEAD
            and prior_factor.get("target_feedback_revision") == SOURCE_FEEDBACK
            and prior_factor.get("counter_origin") == ANCESTOR_COUNTERS
            and prior.get("source_selection", {}).get("source_role") ==
                ANCESTOR_SOURCE["source_role"],
            "v7 source is not the exact formally published v6 ancestor")
    require(raw_metadata.get(MIGRATION_KEY) == receipt,
            "evaluated checkpoint does not inherit the exact v7 receipt")
    if p["checkpoint_role"] == "ancestor-branch-descendant":
        name = p["checkpoint_output_branch"]
        expected_route = {
            "schema": "wlr50_clean.checkpoint_output_routing.v1",
            "branch": name,
            "output_root": str((HERE / "branches" / name).resolve()),
            "main_latest_pointer_promotion": False,
            "source_selection": BRANCH_ROUTE_SOURCE,
        }
        require(raw_metadata.get("checkpoint_output_routing") == expected_route,
                "v7 descendant did not preserve the original c531 branch route")

    original_read = m.read_json
    m.read_json = lambda path: _normalized_v7_view(original_read(path))
    try:
        inherited = _V6_IDENTITY(manifest)
    finally:
        m.read_json = original_read
    inherited.pop("v6_plan", None)
    inherited.pop("v6_plan_sha256", None)
    inherited.pop("v6_publication", None)
    inherited.pop("v6_publication_sha256", None)
    inherited.update({
        "rr_capture_control_revision": TARGET_REVISION,
        "rr_capture_feedback_revision": REVISION,
        "v7_plan": str(p["migration_plan"]),
        "v7_plan_sha256": p["migration_plan_sha256"],
        "v7_publication": str(p["publication"]),
        "v7_publication_sha256": p["publication_sha256"],
        "signed_AIR_permission_is_contact_or_bearing": False,
        "weak_contact_controller_added": False,
    })
    return inherited


def validate_snapshot(state):
    """Validate the unchanged 14-scalar public snapshot with v7 signed bounds."""
    base, m = v6(), media()
    base.v5().base()._SNAPSHOT_VALIDATOR(state)
    feature_names = tuple(m.RR_V2_FEATURE_NAMES)
    metadata = {"schema", "version", "feedback_revision",
        "window_reference_semantics", "capture_search_semantics", "mode_name",
        "reason", "active", "owners", "owner_indices",
        "last_dispatch_physics_tick", "feature_names"}
    m.require(set(state) == set(feature_names) | metadata
        and state.get("feedback_revision") == REVISION
        and state.get("capture_search_semantics") == SEARCH,
        "tick is not the exact reviewed v7 public snapshot")
    mode = state.get("mode")
    travel = state.get("travel_used_deg")
    elapsed = state.get("descent_elapsed_s")
    m.require(type(mode) in (int, float) and float(mode).is_integer()
        and 0 <= mode <= 7 and 0 <= travel <= 53 + 1e-9
        and 0 <= elapsed <= 45 + 1e-9,
        "v7 public mode/travel/exposure is malformed")
    knee_elapsed = max(travel - 20.0, 0.0)
    hip_elapsed = elapsed - knee_elapsed
    hip_travel = min(travel, 20.0)
    m.require(hip_travel / 2.0 - 1e-9 <= hip_elapsed
        <= min(12.0, hip_travel) + 1e-9,
        "v7 public hip/knee travel and exposure are inconsistent")
    if mode == 6:
        m.require(travel >= 20 and -.015 <= state.get("window_start_gap_m") <= .025
            and state.get("window_elapsed_s") < 2,
            "v7 DESCEND_PROGRESS lacks public fresh signed-band credit")
    if travel > 52 + 1e-9:
        m.require(-.015 <= state.get("window_start_gap_m") <= .001,
            "v7 signed terminal degree lacks its public peak-gap evidence")
    if mode == 7:
        m.require(state.get("contact_seen") == 1 and state.get("retired") == 0,
                  "v7 CAPTURED_FOLLOW lacks contact history or live ownership")


def frame_summary(row, native_row):
    out = v6()._V5_FRAME_SUMMARY(row, native_row)
    state = row["rr_capture_assist"]
    semantic_task = ((row.get("step_info") or {}).get("semantic_task") or {})
    travel = float(state["travel_used_deg"])
    gap_mm = float(out["rr_post_gap_mm"])
    out.update(
        rr_v7_mode_index=int(state["mode"]),
        rr_v7_feedback_revision=state["feedback_revision"],
        rr_v7_capture_search_semantics=state["capture_search_semantics"],
        rr_v7_total_travel_deg=travel,
        rr_v7_base_search_used_deg=min(travel, 40.0),
        rr_v7_progress_reserve_used_deg=min(max(travel - 40.0, 0.0), 12.0),
        rr_v7_signed_contact_used_deg=max(travel - 52.0, 0.0),
        rr_v7_total_exposure_s=float(state["descent_elapsed_s"]),
        rr_v7_descend_progress=state["mode_name"] == "DESCEND_PROGRESS",
        rr_v7_captured_follow=state["mode_name"] == "CAPTURED_FOLLOW",
        rr_v7_public_captured_hip_target_deg=float(state["hip_target_deg"]),
        rr_v7_public_captured_knee_target_deg=float(state["knee_hold_deg"]),
        rr_v7_current_gap_inside_configured_signed_band=-15.0 <= gap_mm <= 25.0,
        rr_v7_signed_AIR_permission_observed=(state["mode_name"] == "DESCEND_PROGRESS"
            and out["rr_post_contact"] == "AIR" and -15.0 <= gap_mm <= 25.0),
        semantic_terminal_reason=semantic_task.get("termination_reason"),
        semantic_terminal_source=semantic_task.get("termination_source"),
    )
    return out


def capture_rows(candidate, ledger):
    m = media()
    wanted = {value.sim_step for value in ledger}
    frames, milestones = {}, {}
    count, offset, last = 0, None, None
    with candidate["tick_path"].open("rb") as ticks, \
            candidate["native_tick_path"].open("rb") as natives:
        for count, raw in enumerate(ticks, 1):
            native_raw = natives.readline()
            m.require(raw.endswith(b"\n") and native_raw.endswith(b"\n"),
                      "unsealed/partial v7 tick ledger")
            row, native = json.loads(raw), json.loads(native_raw)
            m.require(row.get("episode_physics_tick") == count
                and math.isclose(float(row["sim_time_s"]), count / 120., abs_tol=1e-10),
                "v7 tick sequence gap")
            item = frame_summary(row, native)
            offset = (item["dispatch_to_episode_tick_offset"] if offset is None else offset)
            m.require(item["dispatch_to_episode_tick_offset"] == offset,
                      "v7 dispatch offset changed")
            events = {
                "FIRST_RR_OWNER": bool(item["rr_assist_owner_indices"]),
                "FIRST_RR_CROSS": item["rr_front_edge_crossed"] is True,
                "FIRST_DESCEND_PROGRESS": item["rr_v7_descend_progress"],
                "FIRST_SIGNED_AIR_PERMISSION": item["rr_v7_signed_AIR_permission_observed"],
                "FIRST_SIGNED_TERMINAL_INCREMENT": item["rr_v7_signed_contact_used_deg"] > 0,
                "FIRST_CAPTURED_FOLLOW": item["rr_v7_captured_follow"],
                "FIRST_WHEEL_ENVELOPE": item["wheel_envelope_active"],
                "FIRST_WHEEL_PROJECTION": item["wheel_projection_changed"],
                "FIRST_RR_TOP": item["rr_post_contact"] == "TOP",
                "FIRST_RR_PLACED": item["rr_placed_history"] is True,
                "FIRST_P10": item["phase"] == "P10",
                "FIRST_RL_PREPARATION": item["phase"] == "P11",
                "FIRST_RL_CAPTURE_PHASE": item["phase"] == "P12",
                "FIRST_RL_PLACED": item["rl_placed_history"] is True,
                "FIRST_P13": item["phase"] == "P13",
            }
            for key, observed in events.items():
                if observed and key not in milestones:
                    milestones[key] = {"milestone": key, **item}
            if count in wanted:
                frames[count] = item
            last = item
        m.require(natives.readline() == b"", "native ledger extends past v7 episode")
    m.require(count == candidate["endpoint"] and set(frames) == wanted and last is not None,
              "exact full v7 episode/frame coverage unavailable")
    milestones["TERMINAL"] = {"milestone": "TERMINAL", **last}
    rows = [frames[value.sim_step] for value in ledger]
    facts = {
        "run_RR_window_reached": m.rr_window_reached(rows),
        "run_RR_TOP_observed": any(value["rr_post_contact"] == "TOP" for value in rows),
        "run_RR_placed": any(value["rr_placed_history"] is True for value in rows),
        "run_P10_reached": any(value["phase"] in ("P10", "P11", "P12", "P13") for value in rows),
        "run_RL_window_reached": any(value["phase"] in ("P11", "P12", "P13") for value in rows),
        "run_RL_placed": any(value["rl_placed_history"] is True for value in rows),
    }
    for item in rows:
        item.update(facts)
    return rows, list(milestones.values())


def panel_lines(row, result, *, rr_reached, feedback_v2):
    m, f = media(), media()._fmt
    credit = {key: pins()["current_counters"][key] - ANCESTOR_COUNTERS[key]
              for key in COUNTER_KEYS}
    return [
        f"{display_id()} DET | PPO + FL/RR CAPTURE ASSISTS + SUPPORT-WHEEL v4 + INHERITED LIMITED AUX | {result}",
        f"v7 SIGNED-AIR PERMISSION | migration +0 | branch learning +{credit['global_policy_decisions']}/+{credit['ppo_updates']}/+{credit['optimizer_steps']} | RR/P10/RL={row['run_RR_window_reached']}/{row['run_P10_reached']}/{row['run_RL_window_reached']}",
        f"RR {row['rr_assist_mode']} ({row['rr_assist_reason']}) mode={row['rr_v7_mode_index']} | travel base/reserve/signed={f(row['rr_v7_base_search_used_deg'])}/{f(row['rr_v7_progress_reserve_used_deg'])}/{f(row['rr_v7_signed_contact_used_deg'])}deg",
        f"public RR hip target/actual {f(row['rr_v7_public_captured_hip_target_deg'])}/{f(row['rr_hip_actual_deg'])}deg | knee target/actual {f(row['rr_v7_public_captured_knee_target_deg'])}/{f(row['rr_knee_actual_deg'])}deg | captured-follow={row['rr_v7_captured_follow']}",
        f"RR gap {f(row['rr_post_gap_mm'])}mm depth {f(row['rr_front_distance_mm'])}mm {row['rr_post_contact']} | TOP-ever={row['run_RR_TOP_observed']} placed={row['run_RR_placed']} RL-placed={row['run_RL_placed']}",
        f"signed band permission={row['rr_v7_signed_AIR_permission_observed']} | AIR is not contact/bearing; no weak-contact controller | controller credit is not PPO",
        m._wheel_line("source nominal wheel rad/s: ", row["source_nominal_wheel_rad_s"]),
        m._wheel_line("FINAL target canonical rad/s: ", row["wheel_final_canonical_rad_s"]),
        m._wheel_line("MEASURED canonical qdot rad/s: ", row["wheel_actual_canonical_rad_s"]),
        m._wheel_line("wheel-v4 controller delta rad/s: ", row["wheel_controller_delta"]),
        f"1x 15fps | t={row['time_s']:.3f}s tick={row['tick']} {row['phase']} | current contact != placed history; sensor result controls success labels",
    ]


def detail_plan(rows):
    m = media()
    if rows[-1]["run_RR_window_reached"]:
        start = next(index for index, value in enumerate(rows)
            if value["phase"] in ("P09", "P10", "P11", "P12", "P13")
            or value["rr_lift_carry"] or value["rr_assist_mode"] != "WAIT")
        if rows[-1]["run_RL_window_reached"]:
            name, title = "RR_window_through_observed_RL_tail", \
                "RR window through actually observed RL tail"
        elif rows[-1]["run_P10_reached"]:
            name, title = "RR_window_through_P10_RL_not_reached", \
                "RR window through actual P10 tail - RL NOT REACHED"
        else:
            name, title = "RR_window_RL_not_reached", \
                "RR window through terminal - P10/RL NOT REACHED"
        reason = None
    else:
        start = max(0, len(rows) - 60 * m.FPS)
        name, title = "predecessor_failure_tail", \
            "predecessor terminal tail - RR/P10/RL NOT REACHED"
        reason = "THIS_EPISODE_DID_NOT_REACH_RR_WINDOW"
    step = pins()["current_counters"]["global_policy_decisions"]
    return {"kind": name, "start": start, "end": len(rows),
        "filename": f"CP{step}_DET_v7_{name}_detail.mp4",
        "title": f"DETAIL | {display_id()} | {title}",
        "requested_RR_detail_unavailable_reason": reason}


def media():
    """Install only the narrow v7 callbacks into the SHA-pinned v6 renderer."""
    global _MEDIA
    if _MEDIA is not None:
        return _MEDIA
    base = v6()
    base._PINS = pins()
    base.MIGRATION_KEY = MIGRATION_KEY
    base.FACTOR_KEY = FACTOR_KEY
    base.PRIOR_MIGRATION_KEY = PRIOR_MIGRATION_KEY
    base.SCHEMA = SCHEMA
    base.TARGET_REVISION = TARGET_REVISION
    base.SOURCE_FEEDBACK = SOURCE_FEEDBACK
    base.REVISION = REVISION
    base.SEARCH = SEARCH
    base.METHOD = METHOD
    base.ANCESTOR_COUNTERS = ANCESTOR_COUNTERS
    base.ANCESTOR_SOURCE = ANCESTOR_SOURCE
    base.display_id = display_id
    base.checkpoint_identity = checkpoint_identity
    base.validate_snapshot = validate_snapshot
    base.frame_summary = frame_summary
    base.panel_lines = panel_lines
    base.detail_plan = detail_plan
    module = base.media()
    module.checkpoint_identity = checkpoint_identity
    module.validate_historical_rr_snapshot = validate_snapshot
    module.FEEDBACK_V2_REVISION = REVISION
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
        f"{display_id()} | SIGNED-AIR v7 + FL/RR/WHEELv4 + AUX | "
        f"branch PPO +{credit['ppo_updates']}")
    ast.fix_missing_locations(node)
    exec(compile(ast.Module(body=[node], type_ignores=[]),
                 str(module.__file__) + ":v7_label_only", "exec"), module.__dict__)
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
        destination / f"CP{step}_DET_v7_full_attempt_{suffix}.mp4",
        result=result, rr_reached=rows[-1]["run_RR_window_reached"],
        feedback_v2=True, ffmpeg=ffmpeg)
    print("FULL_PLAYABLE_VALIDATED " + full["output"], flush=True)
    detail = m.encode_detail(Path(full["output"]), rows, destination, ffmpeg=ffmpeg)
    pair = m.encode_pair(baseline, {**candidate, "frame_count": len(rows)},
        Path(full["output"]), destination / f"historical_N_vs_CP{step}_DET_v7.mp4",
        historical_version=m.DEFAULT_HISTORICAL_VERSION,
        feedback_v2=True, ffmpeg=ffmpeg)
    evidence = destination / f"CP{step}_v7_selected_fourwheel_evidence.json"
    m.write_new_json(evidence, {
        "schema": "wlr50_clean.rr_signed_contact_v7_video_evidence.v1",
        "source": str(source),
        "capture_assist_ticks_sha256": m.sha256(candidate["tick_path"]),
        "native_tick_audit_sha256": m.sha256(candidate["native_tick_path"]),
        "selection": "first actually observed v7/RR/P10/RL/wheel/signed-band/contact milestones plus terminal; absent milestones are not invented",
        "pre_and_post_ticks_separate": True,
        "signed_AIR_permission_is_contact_or_bearing": False,
        "weak_contact_controller_added": False,
        "rows": selected,
    })
    credit = {key: pins()["current_counters"][key] - ANCESTOR_COUNTERS[key]
              for key in COUNTER_KEYS}
    receipt = {
        "schema": "wlr50_clean.rr_signed_contact_v7_video_export.v1",
        "source": str(source),
        "source_manifest_sha256": m.sha256(candidate["manifest_path"]),
        "source_run_manifest_sha256": m.sha256(candidate["run_manifest_path"]),
        "checkpoint": candidate["checkpoint"],
        "checkpoint_role": pins()["checkpoint_role"],
        "checkpoint_output_branch": pins()["checkpoint_output_branch"],
        "control_method": METHOD,
        "source_legacy_control_method": candidate["manifest"].get("control_method"),
        "effective_wheel_mode": WHEEL_MODE,
        "RR_assist_feedback_revision": REVISION,
        "RR_assist_capture_search_semantics": SEARCH,
        "historical_ancestor_weight_counters": ANCESTOR_COUNTERS,
        "RR_branch_learning": credit,
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
            "v6_media_adapter_sha256": V6_ADAPTER_SHA,
        },
        "missing_events_are_not_invented": True,
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
                        help="exact sealed v7 semantic-video source directory")
    result.add_argument("--destination", type=Path, required=True)
    result.add_argument("--expected-head", required=True)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--checkpoint-sha256", required=True)
    result.add_argument("--published-v7-checkpoint", type=Path, required=True)
    result.add_argument("--published-checkpoint-sha256", required=True)
    result.add_argument("--published-checkpoint-manifest-sha256", required=True)
    result.add_argument("--migration-plan", type=Path, required=True)
    result.add_argument("--migration-plan-sha256", required=True)
    result.add_argument("--publication", type=Path, required=True)
    result.add_argument("--publication-sha256", required=True)
    result.add_argument("--publication-state-proof-key", required=True,
        help="exact boolean publication field proving same410 full-state preservation")
    result.add_argument("--migration-source-checkpoint", type=Path, required=True)
    result.add_argument("--migration-source-checkpoint-sha256", required=True)
    result.add_argument("--migration-source-manifest-sha256", required=True)
    result.add_argument("--migration-source-head", required=True)
    result.add_argument("--migration-source-role", required=True)
    result.add_argument("--checkpoint-role", required=True,
        choices=("published-ancestor-zero-learning", "ancestor-branch-descendant"))
    result.add_argument("--checkpoint-output-branch")
    result.add_argument("--expected-parent-checkpoint-sha256")
    result.add_argument("--expected-global-policy-decisions", type=int, required=True)
    result.add_argument("--expected-ppo-updates", type=int, required=True)
    result.add_argument("--expected-optimizer-steps", type=int, required=True)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    configure(arguments)
    export(arguments.source, arguments.destination)
