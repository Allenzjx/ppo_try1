"""Outputs-only exporter for the exact learned v10 CP222720 evaluation.

This is a narrow adapter over the already decoded/validated RR media path.  It
does not import the live runtime, discover ``latest`` state, start Isaac, or
award controller/AUX work to PPO.  Invocation is intentionally impossible
until the caller supplies hashes for a *sealed* source and parent run.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
V8_ADAPTER = HERE / "export_rr_capture_video_v8.py"
V8_ADAPTER_SHA = "b97d33d8fd5a5b2ae9f143f8a818d38a31ca3db86d6da872eb44b13c3f5cf0db"

SCHEMA = "wlr50_clean.rr_capture_reserve_same410.v10"
FACTOR_KEY = "rr_capture_reserve_v10_factor"
MIGRATION_KEY = "rr_capture_reserve_v10_migration"
TARGET_HEAD = "99dff5fd366e9cad80899176bf8b008e02112daf"
SOURCE_HEAD = "3edda51732f4ff85717fcb3491bf5c8c5766474d"
BRANCH = "ancestor220544_signed_wheel_v8"
POLICY_VERSION = "rr_capture_then_rl_transfer_history_v1"
OBSERVATION_LAYOUT = "role389_rr_capture_transfer_v1"
CONTROL_REVISION = "rr_capture_progress_reserve_v10"
FEEDBACK_REVISION = "progress_earned_capture_reserve_incremental_v10"
WINDOW_REFERENCE = (
    "public_window_peak_gap_reuses_existing_window_start_gap_scalar_"
    "upward_motion_never_resets_elapsed"
)
SEARCH = (
    "hip20_knee20_progress_earned_current_XY_support_tracking_gap_ge_minus15mm_"
    "knee12_then_1deg_public_peak_gap_le1mm_total53_exposure45_no_recharge_"
    "AIR_not_contact_sensor_TOP_unchanged_captured_issued_N_request_deltas_"
    "RL_current_TOP_retirement"
)
METHOD = (
    "PPO_PLUS_FL_RR_CAPTURE_ASSISTS_PLUS_V10_CAPTURE_RESERVE_PLUS_"
    "INHERITED_WHEEL_V9_AND_LIMITED_AUX"
)
COUNTER_KEYS = ("global_policy_decisions", "ppo_updates", "optimizer_steps")
BRANCH_ORIGIN = dict(zip(COUNTER_KEYS, (220544, 1688, 33760)))
V10_ORIGIN = dict(zip(COUNTER_KEYS, (222592, 1704, 34080)))
CURRENT_COUNTERS = dict(zip(COUNTER_KEYS, (222720, 1705, 34100)))
SOURCE_SELECTION = {
    "checkpoint_sha256": "8aaf255c118fa9199467b8e4016ad6d8ac235d28a31f25335423c45e93e5d86e",
    "source_git_commit": SOURCE_HEAD,
    "checkpoint_output_branch": BRANCH,
    "branch_origin": BRANCH_ORIGIN,
    "manifest_sha256": "32aa8dcd6eef618b6262342911f6971a85e1217102bb149ab0be93f08ad6cf67",
    "counters": V10_ORIGIN,
    "source_role": "latest_sealed_v9_branch_with_official_front_retention_AUX",
    "front_retention_auxiliary_sha256":
        "52aad294be62450ec44fb3c28b4f20a0b3cf9d54b3cb653c23fddad5047ac815",
}
CURRENT_CP_SHA = "069a71f547427b69491ff749dccc14fcf403565d81b3c37db3c6ac9b1e739e55"
CURRENT_MANIFEST_SHA = "314f4e25189d895f24c51c590dbe3aa284c20cf9de7a049fad526bd13b2054e5"
PUBLISHED_CP_SHA = "8f1634e58d5303a53a1abef39c3ef8cc459223b1a2ca582c85d6d978cdf0b223"
PUBLISHED_MANIFEST_SHA = "35a6c9453f0eda960b58ad6b61898c8b5a625e637373d320b63fba4c0a86a671"
PLAN_SHA = "4fa505698d5fe094c4ea6c98b40a6e131e61a2b14443bef1e0e2fe0fe1fa883a"
PUBLICATION_SHA = "ce49e5e3c1cb5492ba908ec13e212b0810b878698aeeb75f4104648fb04e7c8"
PUBLISHER_SHA = "3a3c1f46748ad68283e79896f026681e12c84cba567463d19c6f9e5d7b05aa3e"
LEDGER_SHA = SOURCE_SELECTION["front_retention_auxiliary_sha256"]
RR_FEATURE_NAMES = (
    "mode", "initialized", "knee_hold_deg", "hip_entry_deg", "hip_target_deg",
    "travel_used_deg", "descent_elapsed_s", "window_start_gap_m",
    "window_elapsed_s", "hold_elapsed_s", "release_fraction", "contact_seen",
    "retired", "blocked_reason",
)
RR_MODES = ("WAIT", "DESCEND", "HOLD", "BLOCKED", "RELEASE", "RELEASED",
            "DESCEND_PROGRESS", "CAPTURED_FOLLOW")
RR_REASONS = ("none", "physical_invalid", "capture_XY_unavailable",
    "other_support_unavailable", "gap_not_improving",
    "finite_search_travel_or_margin", "waiting_actual_tracking",
    "current_AIR_unavailable", "finite_descent_exposure",
    "qualified_crossing_unavailable", "fresh_capture_progress_required")

_PINS = None
_V8 = None
_MEDIA = None
_V8_FRAME_SUMMARY = None
_IDENTITY = None


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
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def sidecar(checkpoint):
    checkpoint = Path(checkpoint)
    return checkpoint.with_name(checkpoint.stem + "_manifest.json")


def checked_sha(value, label):
    value = str(value).lower()
    require(re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            f"{label} must be an explicit lowercase SHA256")
    return value


def checked_file(path, expected, label):
    path = Path(path).resolve(strict=True)
    require(path.is_file() and sha256(path) == expected,
            f"{label} differs from its explicit SHA256")
    return path


def configure(args):
    global _PINS
    require(_PINS is None, "v10 exporter may be configured only once")
    require(str(args.expected_head).lower() == TARGET_HEAD,
            "--expected-head is not the frozen v10 runtime")
    expected = {
        "checkpoint_sha256": CURRENT_CP_SHA,
        "checkpoint_manifest_sha256": CURRENT_MANIFEST_SHA,
        "published_checkpoint_sha256": PUBLISHED_CP_SHA,
        "published_checkpoint_manifest_sha256": PUBLISHED_MANIFEST_SHA,
        "migration_source_checkpoint_sha256": SOURCE_SELECTION["checkpoint_sha256"],
        "migration_source_manifest_sha256": SOURCE_SELECTION["manifest_sha256"],
        "migration_plan_sha256": PLAN_SHA,
        "publication_sha256": PUBLICATION_SHA,
        "publisher_script_sha256": PUBLISHER_SHA,
        "front_retention_ledger_sha256": LEDGER_SHA,
    }
    hashes = {name: checked_sha(getattr(args, name), "--" + name.replace("_", "-"))
              for name in expected}
    require(hashes == expected, "one immutable v10/checkpoint/AUX pin differs")
    require((args.expected_global_policy_decisions, args.expected_ppo_updates,
             args.expected_optimizer_steps) == tuple(CURRENT_COUNTERS[k]
                                                       for k in COUNTER_KEYS),
            "checkpoint counters are not actual CP222720 counters")
    require(args.expected_front_retention_accepted == 32 and
            args.expected_front_retention_attempted == 32,
            "front-retention AUX credit must remain the actual separate 32/32")
    files = {
        "checkpoint": checked_file(args.checkpoint, CURRENT_CP_SHA,
                                   "evaluated CP222720"),
        "published_checkpoint": checked_file(args.published_v10_checkpoint,
                                             PUBLISHED_CP_SHA,
                                             "zero-update v10 publication"),
        "migration_source_checkpoint": checked_file(args.migration_source_checkpoint,
            SOURCE_SELECTION["checkpoint_sha256"], "exact AUX CP222592 source"),
        "migration_plan": checked_file(args.migration_plan, PLAN_SHA, "v10 plan"),
        "publication": checked_file(args.publication, PUBLICATION_SHA,
                                    "v10 publication receipt"),
        "publisher_script": checked_file(args.publisher_script, PUBLISHER_SHA,
                                         "reviewed v10 publisher"),
    }
    manifests = {
        "checkpoint_manifest": sidecar(files["checkpoint"]).resolve(strict=True),
        "published_manifest": sidecar(files["published_checkpoint"]).resolve(strict=True),
        "migration_source_manifest": sidecar(
            files["migration_source_checkpoint"]).resolve(strict=True),
    }
    require(sha256(manifests["checkpoint_manifest"]) == CURRENT_MANIFEST_SHA and
            sha256(manifests["published_manifest"]) == PUBLISHED_MANIFEST_SHA and
            sha256(manifests["migration_source_manifest"]) ==
                SOURCE_SELECTION["manifest_sha256"],
            "checkpoint sidecar differs from its immutable pin")
    _PINS = {**files, **manifests, **hashes,
        "source_manifest_sha256": checked_sha(args.source_manifest_sha256,
                                                "--source-manifest-sha256"),
        "source_run_manifest_sha256": checked_sha(args.source_run_manifest_sha256,
                                                    "--source-run-manifest-sha256"),
        "current_counters": dict(CURRENT_COUNTERS),
    }
    return _PINS


def pins():
    require(isinstance(_PINS, dict), "configure v10 pins before export")
    return _PINS


def v8():
    global _V8
    if _V8 is None:
        require(sha256(V8_ADAPTER) == V8_ADAPTER_SHA,
                "reviewed v8 media adapter bytes changed")
        spec = importlib.util.spec_from_file_location("_rr_v10_reviewed_v8_media",
                                                       V8_ADAPTER)
        require(spec is not None and spec.loader is not None,
                "cannot load reviewed v8 media adapter")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _V8 = module
    return _V8


def display_id():
    return "CP222720 v10"


def _front_retention(metadata):
    ledger = ((metadata.get("rr_capture_transfer_branch") or {})
              .get("front_retention_auxiliary"))
    require(isinstance(ledger, dict) and json_digest(ledger) == LEDGER_SHA and
            ledger.get("schema") == "wlr50_clean.RR410_front_retention_auxiliary.v1" and
            ledger.get("accepted_auxiliary_updates_total") == 32 and
            ledger.get("attempted_auxiliary_optimizer_steps_total") == 32,
            "front-retention AUX ledger is not the exact independent 32/32 record")
    return ledger


def checkpoint_identity(manifest):
    """Validate only the exact v10 boundary, direct child, and video load."""
    p = pins()
    metadata = read_json(p["checkpoint_manifest"])
    published = read_json(p["published_manifest"])
    source = read_json(p["migration_source_manifest"])
    plan = read_json(p["migration_plan"])
    publication = read_json(p["publication"])
    factor = plan.get(FACTOR_KEY) or {}
    receipt = published.get(MIGRATION_KEY) or {}
    require(plan.get("schema") == SCHEMA and factor.get("schema") == SCHEMA and
            receipt.get("schema") == SCHEMA,
            "v10 plan/factor/publication schema differs")
    require(plan.get("source_selection") == SOURCE_SELECTION and
            factor.get("source_selection") == SOURCE_SELECTION and
            factor.get("counter_origin") == V10_ORIGIN and
            plan.get("source_checkpoint_sha256") == SOURCE_SELECTION["checkpoint_sha256"] and
            plan.get("source_manifest_sha256") == SOURCE_SELECTION["manifest_sha256"] and
            plan.get("source_git_commit") == SOURCE_HEAD and
            plan.get("target_git_commit") == TARGET_HEAD,
            "v10 migration source or target identity differs")
    effective = factor.get("effective_execution_semantics") or {}
    require(factor.get("source_feedback_revision") ==
                "signed_band_contact_formation_incremental_v6" and
            factor.get("target_feedback_revision") == FEEDBACK_REVISION and
            effective.get("search_semantics") == SEARCH and
            effective.get("removed_upper_gap_checks") ==
                ["reserve_earned", "reserve_continuation", "mode6_snapshot_validation"] and
            effective.get("signed_lower_gap_m") == -.015 and
            effective.get("sensor_TOP_upper_gap_rule_changed") is False and
            effective.get("total_maximum_travel_deg") == 53.0 and
            effective.get("total_maximum_active_exposure_s") == 45.0 and
            effective.get("reserve_budget_recharged") is False and
            effective.get("current_XY_Q_cross_AIR_support_tracking_still_required") is True and
            effective.get("v9_wheel_bytes_unchanged") is True and
            effective.get("raw_Gaussian_is_not_transformed_target") is True,
            "v10 finite reserve semantics differ")
    for key, expected in {
        "observation_shape_changed": False, "observation_codec_changed": False,
        "capture_search_budget_changed": False, "policy_kernel_changed": False,
        "sigma_changed": False, "caps_changed": False, "reward_changed": False,
        "physical_dynamics_changed": False,
        "physical_task_acceptance_rules_changed": False,
        "same_mdp_claimed": False, "existing_branch_origins_preserved": True,
        "creates_new_branch": False, "latest_branch_learned_weights_preserved": True,
        "latest_pointer_promotion_authorized": False,
    }.items():
        require(factor.get(key) is expected, f"v10 semantic flag differs: {key}")
    require(all(factor.get("added_" + name) == 0 for name in
                ("policy_decisions", "ppo_updates", "optimizer_steps",
                 "auxiliary_updates")), "v10 migration was assigned learning credit")
    observation = factor.get("observation_contract") or {}
    require(observation.get("observation_dimension") == 410 and
            observation.get("action_dimension") == 12 and
            observation.get("observation_layout") == OBSERVATION_LAYOUT and
            observation.get("source_policy_contract") ==
                observation.get("target_policy_contract") == source.get("policy_contract"),
            "v10 changed the 410 policy contract/kernel")

    require(Path(receipt.get("plan_path", "")).resolve() == p["migration_plan"] and
            receipt.get("plan_sha256") == PLAN_SHA and
            {key: value for key, value in receipt.items()
             if key not in ("plan_path", "plan_sha256")} == plan and
            metadata.get(MIGRATION_KEY) == receipt,
            "current checkpoint does not inherit the exact v10 publication receipt")
    preserved = factor.get("preserved_metadata_sha256") or {}
    require(isinstance(preserved, dict) and preserved,
            "v10 factor lacks migration-time preserved bindings")
    for key, digest in preserved.items():
        require(key in source and json_digest(source[key]) == digest and
                published.get(key) == source[key],
                f"zero-update v10 publication changed preserved field {key}")
    source_ledger = _front_retention(source)
    require(_front_retention(published) == source_ledger and
            _front_retention(metadata) == source_ledger,
            "front-retention AUX changed across migration/PPO save")
    require(all(source.get(key) == V10_ORIGIN[key] for key in COUNTER_KEYS) and
            all(published.get(key) == V10_ORIGIN[key] for key in COUNTER_KEYS) and
            all(metadata.get(key) == CURRENT_COUNTERS[key] for key in COUNTER_KEYS),
            "v10 source/publication/current counters differ")
    branch_learning = {key: CURRENT_COUNTERS[key] - BRANCH_ORIGIN[key]
                       for key in COUNTER_KEYS}
    require(metadata.get("rr_capture_transfer_branch_counts") == branch_learning,
            "whole RR branch counters differ from original branch origin")
    route = metadata.get("checkpoint_output_routing")
    require(route == source.get("checkpoint_output_routing") ==
                published.get("checkpoint_output_routing") and
            route.get("branch") == BRANCH and
            route.get("main_latest_pointer_promotion") is False,
            "v10 checkpoint left the isolated no-promotion branch")
    ancestry = metadata.get("resume_ancestry") or {}
    parent = ancestry.get("source_checkpoint") or {}
    require(parent.get("checkpoint_sha256") == PUBLISHED_CP_SHA and
            parent.get("manifest_sha256") == PUBLISHED_MANIFEST_SHA and
            ancestry.get("resume_migration") == receipt,
            "CP222720 is not the direct ordinary-PPO child of the v10 publication")
    require(publication.get("checkpoint_sha256") == PUBLISHED_CP_SHA and
            publication.get("source_checkpoint_sha256") ==
                SOURCE_SELECTION["checkpoint_sha256"] and
            publication.get("target_git_commit") == TARGET_HEAD and
            publication.get("save_load_round_trip") is True and
            publication.get("no_pointer_promotion") is True and
            publication.get("front_retention_auxiliary_counts") ==
                {"accepted": 32, "attempted": 32} and
            all(publication.get("migration_added_" + name) == 0 for name in
                ("policy_decisions", "ppo_updates", "optimizer_steps",
                 "auxiliary_updates")),
            "official v10 save/reload publication differs")

    runtime = metadata.get("runtime_contract") or {}
    load = manifest.get("checkpoint_load_provenance") or {}
    loaded = load.get("source") or {}
    require(runtime == manifest.get("runtime_contract") and
            runtime.get("source_git_commit") == TARGET_HEAD and
            load.get("checkpoint_loaded_and_verified") is True and
            load.get("policy_sampling_mode") == "deterministic_conditional_mean" and
            load.get("stochastic_policy") is False and
            load.get("optimizer_updates") == 0 and
            load.get("saved_global_policy_decisions") ==
                CURRENT_COUNTERS["global_policy_decisions"] and
            load.get("observation_dimension") == 410 and
            load.get("observation_layout") == OBSERVATION_LAYOUT and
            load.get("policy_version") == POLICY_VERSION and
            load.get("policy_contract") == metadata.get("policy_contract") and
            loaded.get("checkpoint_sha256") == CURRENT_CP_SHA and
            loaded.get("manifest_sha256") == CURRENT_MANIFEST_SHA,
            "sealed video did not deterministically load exact CP222720")
    parameters = load.get("parameter_hashes") or {}
    require(all(parameters.get(key) == metadata.get(key) for key in
                ("actor_parameter_sha256", "critic_parameter_sha256",
                 "optimizer_state_sha256", "normalizer_state_sha256")),
            "video load parameter hashes differ from CP222720")
    return {
        "checkpoint": str(p["checkpoint"]),
        "checkpoint_sha256": CURRENT_CP_SHA,
        "checkpoint_manifest_sha256": CURRENT_MANIFEST_SHA,
        "checkpoint_output_branch": BRANCH,
        "counters": dict(CURRENT_COUNTERS),
        "whole_RR_branch_learning": branch_learning,
        "v10_ordinary_PPO_learning": {key: CURRENT_COUNTERS[key] - V10_ORIGIN[key]
                                      for key in COUNTER_KEYS},
        "v10_migration_added_learning": False,
        "front_retention_auxiliary": {"accepted": 32, "attempted": 32,
                                       "digest": LEDGER_SHA,
                                       "separate_from_PPO": True},
        "control_revision": CONTROL_REVISION,
        "feedback_revision": FEEDBACK_REVISION,
    }


def validate_snapshot(state):
    metadata = {"schema", "version", "feedback_revision", "window_reference_semantics",
        "capture_search_semantics", "mode_name", "reason", "active", "owners",
        "owner_indices", "last_dispatch_physics_tick", "feature_names"}
    require(isinstance(state, dict) and set(state) == set(RR_FEATURE_NAMES) | metadata and
            state.get("schema") == "wlr50_clean.rr_capture_assist_state.v1" and
            state.get("version") == "rr_hip_only_capture_v1" and
            state.get("feedback_revision") == FEEDBACK_REVISION and
            state.get("window_reference_semantics") == WINDOW_REFERENCE and
            state.get("capture_search_semantics") == SEARCH and
            state.get("feature_names") == list(RR_FEATURE_NAMES),
            "tick is not the exact public v10 assist snapshot")
    values = {}
    for key in RR_FEATURE_NAMES:
        value = state.get(key)
        require(not isinstance(value, bool) and isinstance(value, (int, float)) and
                math.isfinite(value), f"v10 snapshot {key} is non-finite")
        values[key] = float(value)
    mode, reason = int(values["mode"]), int(values["blocked_reason"])
    require(values["mode"] == mode and 0 <= mode < len(RR_MODES) and
            values["blocked_reason"] == reason and 0 <= reason < len(RR_REASONS) and
            state.get("mode_name") == RR_MODES[mode] and
            state.get("reason") == RR_REASONS[reason],
            "v10 snapshot mode/reason differs")
    active = mode in (1, 2, 3, 4, 6, 7)
    require(state.get("active") is active and
            state.get("owner_indices") == ([6, 7] if active else []) and
            isinstance(state.get("owners"), list) and len(state["owners"]) == 2 and
            type(state.get("last_dispatch_physics_tick")) is int and
            0 <= values["travel_used_deg"] <= 53.0 + 1e-9 and
            0 <= values["descent_elapsed_s"] <= 45.0 + 1e-9,
            "v10 snapshot ownership/budget differs")
    if mode == 6:
        require(values["travel_used_deg"] >= 20.0 and
                values["window_start_gap_m"] >= -.015 and
                values["window_elapsed_s"] < 2.0,
                "v10 DESCEND_PROGRESS lacks lower-band earned-progress evidence")
    if mode == 7:
        require(values["contact_seen"] == 1.0 and values["retired"] == 0.0,
                "v10 CAPTURED_FOLLOW lacks contact history/live ownership")


def frame_summary(row, native_row):
    out = _V8_FRAME_SUMMARY(row, native_row)
    state = row["rr_capture_assist"]
    gap = float(out["rr_post_gap_mm"])
    out.update(
        rr_v10_control_revision=CONTROL_REVISION,
        rr_v10_feedback_revision=state["feedback_revision"],
        rr_v10_search_semantics=state["capture_search_semantics"],
        rr_v10_descend_progress=state["mode_name"] == "DESCEND_PROGRESS",
        rr_v10_progress_lower_gap_satisfied=gap >= -15.0,
        rr_v10_above_removed_legacy_upper=gap > 25.0,
    )
    return out


def _last_json_line(path):
    with Path(path).open("rb") as stream:
        stream.seek(0, 2)
        end = stream.tell()
        require(end > 0, "policy decision stream is empty")
        data, cursor = b"", end
        while cursor > 0 and b"\n" not in data.rstrip(b"\r\n"):
            size = min(65536, cursor)
            cursor -= size
            stream.seek(cursor)
            data = stream.read(size) + data
    return json.loads(data.rstrip(b"\r\n").splitlines()[-1])


def capture_rows(candidate, ledger):
    rows, selected = v8().capture_rows(candidate, ledger)
    first = next((row for row in rows if row["rr_v10_descend_progress"]), None)
    if first is not None:
        selected.append({"milestone": "FIRST_V10_DESCEND_PROGRESS", **first})
    above = next((row for row in rows if row["rr_v10_descend_progress"] and
                  row["rr_v10_above_removed_legacy_upper"]), None)
    if above is not None:
        selected.append({"milestone": "FIRST_V10_PROGRESS_ABOVE_OLD_25MM_UPPER",
                         **above})
    terminal = _last_json_line(candidate["source"] / "video_policy_decisions.jsonl")
    task = ((terminal.get("step_info") or {}).get("semantic_task") or {})
    require((terminal.get("step_info") or {}).get("physics_tick") == rows[-1]["tick"] and
            isinstance(task.get("termination_reason"), str),
            "terminal semantic-task decision does not align with final video row")
    rows[-1]["semantic_terminal_reason"] = task["termination_reason"]
    rows[-1]["semantic_terminal_source"] = task.get("termination_source")
    selected.append({"milestone": "TERMINAL_SEMANTIC_TASK", "tick": rows[-1]["tick"],
        "phase": rows[-1]["phase"], "termination_reason": task["termination_reason"],
        "termination_source": task.get("termination_source")})
    return rows, selected


def identity_cache():
    require(isinstance(_IDENTITY, dict), "checkpoint identity was not validated")
    return _IDENTITY


def panel_lines(row, result, *, rr_reached, feedback_v2):
    m, f = media(), media()._fmt
    credit = identity_cache()["v10_ordinary_PPO_learning"]
    reason = str(row["rr_assist_reason"])
    if len(reason) > 37:
        reason = reason[:34] + "..."
    return [
        f"{display_id()} DET | PPO + FL/RR ASSISTS + WHEEL v9 + INHERITED AUX | {result}",
        f"PPO v10 +{credit['global_policy_decisions']}/+{credit['ppo_updates']}/+{credit['optimizer_steps']} | controller migration +0 PPO | retention AUX 32/32 (not PPO)",
        f"RR/P10/RL={row['run_RR_window_reached']}/{row['run_P10_reached']}/{row['run_RL_window_reached']} | RR {row['rr_assist_mode']} reason={reason} travel={f(row['rr_v7_total_travel_deg'])}deg",
        f"v10 earned reserve: lower gap -15mm, no old +25mm permission ceiling | 53deg/45s | AIR != contact",
        f"RR gap {f(row['rr_post_gap_mm'])}mm front {f(row['rr_front_distance_mm'])}mm {row['rr_post_contact']} | TOP-ever={row['run_RR_TOP_observed']} placed={row['run_RR_placed']}",
        f"RR hip assist/final/actual {f(row['rr_hip_assist_target_deg'])}/{f(row['rr_hip_final_target_deg'])}/{f(row['rr_hip_actual_deg'])}deg | knee hold/final/actual {f(row['rr_knee_hold_target_deg'])}/{f(row['rr_knee_final_target_deg'])}/{f(row['rr_knee_actual_deg'])}",
        m._wheel_line("source nominal wheel rad/s: ", row["source_nominal_wheel_rad_s"]),
        m._wheel_line("FINAL canonical target rad/s: ", row["wheel_final_canonical_rad_s"]),
        m._wheel_line("MEASURED canonical qdot rad/s: ", row["wheel_actual_canonical_rad_s"]),
        "sensor TOP/placement unchanged | controller and retention AUX are separate from PPO credit",
        f"1x 15fps | t={row['time_s']:.3f}s tick={row['tick']} {row['phase']} | actual sensor/result controls labels",
    ]


def detail_plan(rows):
    last = rows[-1]
    if last["run_RR_window_reached"]:
        start = next(index for index, row in enumerate(rows)
            if row["phase"] in ("P09", "P10", "P11", "P12", "P13") or
            row["rr_lift_carry"] or row["rr_assist_mode"] != "WAIT")
        if last["run_RL_window_reached"]:
            name, title = "RR_window_through_observed_RL_tail", \
                "RR window through actually observed RL tail"
        elif last["run_P10_reached"]:
            name, title = "RR_window_through_P10_RL_not_reached", \
                "RR window through actual P10 tail - RL NOT REACHED"
        else:
            name, title = "RR_window_RL_not_reached", \
                "RR window through terminal - P10/RL NOT REACHED"
        unavailable = None
    else:
        start = max(0, len(rows) - 60 * media().FPS)
        name, title = "predecessor_failure_tail", \
            "predecessor terminal tail - RR/P10/RL NOT REACHED"
        unavailable = "THIS_EPISODE_DID_NOT_REACH_RR_WINDOW"
    return {"kind": name, "start": start, "end": len(rows),
        "filename": f"CP222720_DET_v10_{name}_detail.mp4",
        "title": f"DETAIL | {display_id()} | {title}",
        "requested_RR_detail_unavailable_reason": unavailable}


def _cached_identity(manifest):
    global _IDENTITY
    require(_IDENTITY is None, "checkpoint identity validated more than once")
    _IDENTITY = checkpoint_identity(manifest)
    return _IDENTITY


def media():
    global _MEDIA, _V8_FRAME_SUMMARY
    if _MEDIA is not None:
        return _MEDIA
    adapter = v8()
    require(adapter._PINS is None, "v8 adapter was configured outside v10")
    adapter._PINS = pins()
    module = adapter.media()
    _V8_FRAME_SUMMARY = adapter.frame_summary
    module.validate_historical_rr_snapshot = validate_snapshot
    module.FEEDBACK_V2_REVISION = FEEDBACK_REVISION
    module.checkpoint_identity = _cached_identity
    module.panel_lines = panel_lines
    module.detail_plan = detail_plan
    adapter.v7().frame_summary = frame_summary
    adapter.frame_summary = frame_summary
    adapter.panel_lines = panel_lines
    adapter.detail_plan = detail_plan
    _MEDIA = module
    return module


def encode_pair(baseline, candidate, full, output, *, ffmpeg):
    m = media()
    m.require(baseline["manifest"]["camera"] == candidate["manifest"]["camera"],
              "historical N and candidate camera definitions differ")
    _, b_ledger, _ = m.shared().checked_media(baseline, ffmpeg=ffmpeg)
    b_count, c_count = len(b_ledger), candidate["frame_count"]
    count = max(b_count, c_count)
    m.require(count <= m.MAX_FRAMES, "comparison exceeds 200 seconds")
    labels = (f"HISTORICAL N_REF (NOT FRESH B) | {m.DEFAULT_HISTORICAL_VERSION}",
        f"{display_id()} | v10 RESERVE + FL/RR/WHEELv9 | PPO +1 | AUX 32/32")
    font, filters = "C\\:/Windows/Fonts/arial.ttf", []
    for index, (label, frames) in enumerate(zip(labels, (b_count, c_count))):
        filters.append(f"[{index}:v]setpts=PTS-STARTPTS,scale=960:540,pad=960:610:0:70:black,"
            f"tpad=stop_mode=clone:stop=-1,setpts=N/({m.FPS}*TB),"
            f"drawtext=fontfile='{font}':text='{label}':fontcolor=white:fontsize=20:x=14:y=13,"
            f"drawtext=fontfile='{font}':text='RUN ENDED - FROZEN, NOT NEW PHYSICS':"
            f"fontcolor=yellow:fontsize=20:x=14:y=44:enable='gte(n,{frames})'[v{index}]")
    filters.append("[v0][v1]hstack=inputs=2:shortest=1,format=yuv420p[out]")
    command = [str(ffmpeg), "-hide_banner", "-nostdin", "-v", "error", "-n",
        "-threads", "2", "-filter_complex_threads", "1", "-i", str(baseline["video"]),
        "-i", str(full), "-filter_complex", ";".join(filters), "-map", "[out]", "-an",
        "-frames:v", str(count), "-r", str(m.FPS), "-fps_mode", "cfr", "-c:v",
        "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-threads", "2", "-movflags", "+faststart", str(output)]
    m.run(command)
    return {"output": str(output), "frame_count": count,
        "historical_N_version": m.DEFAULT_HISTORICAL_VERSION,
        "historical_N_source": str(baseline["source"]),
        "historical_N_is_fresh_B": False, "same_camera": True,
        "alignment": "same elapsed natural-P01 origin; no phase/event retiming",
        "same_controller_or_runtime_claimed": False,
        "freeze_added_frames": {"historical_N": count - b_count,
                                "candidate": count - c_count},
        "freeze_is_physical_evidence": False, "normal_speed": True,
        "validation": m.shared().validate_output(output, count, 1920, 610, ffmpeg),
        "previews": m.shared().preview(output, count, ffmpeg), "command": command}


def export(source, destination):
    m, p = media(), pins()
    source, destination = Path(source).resolve(strict=True), Path(destination).resolve()
    m.require(destination.is_relative_to(HERE.resolve()) and not destination.exists(),
              "new isolated destination under this output namespace is required")
    candidate = m.sealed_source(source, candidate=True)
    m.require(sha256(candidate["manifest_path"]) == p["source_manifest_sha256"] and
              sha256(candidate["run_manifest_path"]) == p["source_run_manifest_sha256"],
              "sealed source/run manifest differs from explicit invocation")
    baseline = m.sealed_source(m.DEFAULT_HISTORICAL_N, candidate=False)
    ffmpeg = m.shared().find_ffmpeg(candidate["capture"].get("full_decode", {}).get(
        "ffmpeg_path"))
    _, ledger, source_validation = m.shared().checked_media(candidate, ffmpeg=ffmpeg)
    rows, selected = capture_rows(candidate, ledger)
    m.require(0 < len(rows) <= 3000, "source exceeds one normal-speed 200 s episode")
    result, success, acceptance = m.outcome(candidate["manifest"])
    suffix = "SUCCESS" if success else "INCOMPLETE"
    destination.mkdir(parents=True)
    full = m.encode_full(candidate, rows,
        destination / f"CP222720_DET_v10_full_attempt_{suffix}.mp4",
        result=result, rr_reached=rows[-1]["run_RR_window_reached"],
        feedback_v2=True, ffmpeg=ffmpeg)
    print("FULL_PLAYABLE_VALIDATED " + full["output"], flush=True)
    detail = m.encode_detail(Path(full["output"]), rows, destination, ffmpeg=ffmpeg)
    pair = encode_pair(baseline, {**candidate, "frame_count": len(rows)},
        Path(full["output"]), destination / "historical_N_vs_CP222720_DET_v10.mp4",
        ffmpeg=ffmpeg)
    evidence = destination / "CP222720_v10_selected_evidence.json"
    m.write_new_json(evidence, {"schema": "wlr50_clean.rr_capture_reserve_v10_video_evidence.v1",
        "source": str(source),
        "selection": "actual phase/contact/v10-progress/terminal milestones; absent events are not invented",
        "rows": selected})
    identity = identity_cache()
    last = rows[-1]
    receipt = {
        "schema": "wlr50_clean.rr_capture_reserve_v10_video_export.v1",
        "source": str(source),
        "source_manifest_sha256": p["source_manifest_sha256"],
        "source_run_manifest_sha256": p["source_run_manifest_sha256"],
        "checkpoint_identity": identity,
        "control_method_display": METHOD,
        "credit_labels": {
            "ordinary_PPO_after_v10": identity["v10_ordinary_PPO_learning"],
            "v10_controller_migration": {"policy_decisions": 0, "ppo_updates": 0,
                                         "optimizer_steps": 0, "is_PPO": False},
            "front_retention_auxiliary": identity["front_retention_auxiliary"],
        },
        "physical_result": result, "physical_task_success": success,
        "semantic_terminal_reason": last["semantic_terminal_reason"],
        "semantic_terminal_source": last["semantic_terminal_source"],
        "source_acceptance_detail": acceptance,
        "RR_window_reached": last["run_RR_window_reached"],
        "RR_TOP_observed": last["run_RR_TOP_observed"],
        "RR_placed": last["run_RR_placed"],
        "P10_reached": last["run_P10_reached"],
        "RL_window_reached": last["run_RL_window_reached"],
        "RL_placed": last["run_RL_placed"],
        "RL_success_claimed": bool(success and last["run_RL_placed"]),
        "attempt_classification": ("TASK_SUCCESS" if success else
            "RL_WINDOW_REACHED_INCOMPLETE" if last["run_RL_window_reached"] else
            "P10_REACHED_RL_NOT_REACHED" if last["run_P10_reached"] else
            "RR_WINDOW_ONLY" if last["run_RR_window_reached"] else
            "PREDECESSOR_ONLY"),
        "full_episode_continuous": True, "full_failure_tail_preserved": True,
        "normal_speed": True, "single_episode": True, "stitched": False,
        "extra_intro_frames": 0,
        "source_validation": source_validation,
        "full": full, "detail": detail, "historical_N_comparison": pair,
        "historical_N_is_fresh_same_controller_B": False,
        "historical_N_freezes_after_its_own_endpoint": True,
        "selected_evidence": str(evidence),
        "selected_evidence_sha256": m.sha256(evidence),
        "controller_intervention_is_PPO": False,
        "front_retention_auxiliary_is_PPO": False,
        "missing_events_are_not_invented": True,
        "code_binding": {"adapter_sha256": m.sha256(Path(__file__)),
                         "reviewed_v8_media_adapter_sha256": V8_ADAPTER_SHA},
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
    result.add_argument("--source-manifest-sha256", required=True)
    result.add_argument("--source-run-manifest-sha256", required=True)
    result.add_argument("--expected-head", required=True)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--checkpoint-sha256", required=True)
    result.add_argument("--checkpoint-manifest-sha256", required=True)
    result.add_argument("--published-v10-checkpoint", type=Path, required=True)
    result.add_argument("--published-checkpoint-sha256", required=True)
    result.add_argument("--published-checkpoint-manifest-sha256", required=True)
    result.add_argument("--migration-source-checkpoint", type=Path, required=True)
    result.add_argument("--migration-source-checkpoint-sha256", required=True)
    result.add_argument("--migration-source-manifest-sha256", required=True)
    result.add_argument("--migration-plan", type=Path, required=True)
    result.add_argument("--migration-plan-sha256", required=True)
    result.add_argument("--publication", type=Path, required=True)
    result.add_argument("--publication-sha256", required=True)
    result.add_argument("--publisher-script", type=Path, required=True)
    result.add_argument("--publisher-script-sha256", required=True)
    result.add_argument("--front-retention-ledger-sha256", required=True)
    result.add_argument("--expected-front-retention-accepted", type=int, required=True)
    result.add_argument("--expected-front-retention-attempted", type=int, required=True)
    result.add_argument("--expected-global-policy-decisions", type=int, required=True)
    result.add_argument("--expected-ppo-updates", type=int, required=True)
    result.add_argument("--expected-optimizer-steps", type=int, required=True)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    configure(arguments)
    export(arguments.source, arguments.destination)
