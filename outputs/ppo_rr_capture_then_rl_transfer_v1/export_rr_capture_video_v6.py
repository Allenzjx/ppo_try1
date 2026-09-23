"""Strict outputs-only v6 RR contact-onset video exporter.

This file is a dormant text draft until an explicitly named video source has
sealed.  It reuses the SHA-pinned v5 media implementation, never imports the
current runtime's assist implementation, and accepts no implicit ``latest``
checkpoint.  Every mutable identity (target HEAD, evaluated checkpoint,
migration plan, publication, published zero-update checkpoint, and immediate
branch parent when applicable) is an explicit CLI binding.

The two deliberately narrow checkpoint roles are:

* ``published-ancestor-zero-learning``: the independently published CP220544
  v6 migration checkpoint; and
* ``ancestor-branch-descendant``: an ordinary PPO checkpoint written beneath
  the explicit ancestor output branch, with an exact parent SHA and counters.

Physical success, RR contact/TOP/placement, P10, and RL claims are derived only
from the sealed episode.  Missing stages produce truthful failure-tail media.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V5_ADAPTER = HERE / "export_rr_capture_video_v5_CP220544.py"
V5_ADAPTER_SHA = "189088dbd5bdfd5386954fcc677d2bbcaadd690376e8123ebd9c15aedcd40fd4"

MIGRATION_KEY = "rr_contact_onset_v6_migration"
FACTOR_KEY = "rr_contact_onset_v6_factor"
PRIOR_MIGRATION_KEY = "rr_progress_handoff_v5_migration"
SCHEMA = "wlr50_clean.rr_contact_onset_same410.v6"
TARGET_REVISION = "rr_capture_contact_onset_v6"
SOURCE_FEEDBACK = "progress_reserve_captured_incremental_v4"
REVISION = "progress_reserve_contact_onset_incremental_v5"
SEARCH = (
    "hip20_knee20_progress_earned_near_top_knee12_then_1deg_contact_onset_only_"
    "public_peak_gap_le1mm_total53_exposure45_no_recharge_sensor_TOP_unchanged_"
    "captured_issued_N_request_deltas_RL_current_TOP_retirement"
)
WHEEL_MODE = "rr_capture_support_forward_projection_v1"
METHOD = (
    "PPO_PLUS_FL_RR_CAPTURE_ASSISTS_PLUS_SUPPORT_WHEEL_V4_"
    "WITH_INHERITED_LIMITED_AUX"
)
COUNTER_KEYS = ("global_policy_decisions", "ppo_updates", "optimizer_steps")
ANCESTOR_COUNTERS = dict(zip(COUNTER_KEYS, (220544, 1688, 33760)))
ANCESTOR_SOURCE = {
    "checkpoint_sha256": "308122a3c8e760733fbe8370bfdf399a25718148c82333deb41a1265e4e13895",
    "manifest_sha256": "4fbf49e50fe7ab7903462478e45778b79d9cc0f757a99c0127aacb84a5fbb2e8",
    "source_git_commit": "c53119ab332fe048668e66f0543889a128443ca5",
    "counters": ANCESTOR_COUNTERS,
    "source_role": "front_validated_ancestor_control_eval",
}

_V5 = None
_V5_FRAME_SUMMARY = None
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
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
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
    """Resolve all declared pins before importing any media implementation."""
    global _PINS
    require(_PINS is None, "v6 exporter pins may be configured only once")
    head = str(arguments.expected_head).lower()
    require(re.fullmatch(r"[0-9a-f]{40}", head) is not None,
            "--expected-head must be the exact frozen 40-hex commit")
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
            "v6 control validator accepts only the registered CP220544 v5 ancestor source")
    files = {
        "checkpoint": checked_file(arguments.checkpoint,
            hashes["checkpoint_sha256"], "evaluated checkpoint"),
        "published_checkpoint": checked_file(arguments.published_v6_checkpoint,
            hashes["published_checkpoint_sha256"], "published v6 checkpoint"),
        "migration_plan": checked_file(arguments.migration_plan,
            hashes["migration_plan_sha256"], "v6 migration plan"),
        "publication": checked_file(arguments.publication,
            hashes["publication_sha256"], "v6 publication"),
        "migration_source_checkpoint": checked_file(
            arguments.migration_source_checkpoint,
            hashes["migration_source_checkpoint_sha256"],
            "registered v5 migration source"),
    }
    published_manifest = sidecar(files["published_checkpoint"]).resolve(strict=True)
    source_manifest = sidecar(files["migration_source_checkpoint"]).resolve(strict=True)
    require(sha256(published_manifest) == hashes["published_checkpoint_manifest_sha256"],
            "published v6 checkpoint manifest differs from its explicit pin")
    require(sha256(source_manifest) == hashes["migration_source_manifest_sha256"],
            "registered v5 source manifest differs from its explicit pin")
    current = {
        "global_policy_decisions": arguments.expected_global_policy_decisions,
        "ppo_updates": arguments.expected_ppo_updates,
        "optimizer_steps": arguments.expected_optimizer_steps,
    }
    require(all(type(value) is int and value >= ANCESTOR_COUNTERS[key]
                for key, value in current.items()),
            "declared checkpoint counters cannot precede the ancestor")
    branch = arguments.checkpoint_output_branch
    parent_sha = arguments.expected_parent_checkpoint_sha256
    if arguments.checkpoint_role == "published-ancestor-zero-learning":
        require(branch is None and parent_sha is None
                and files["checkpoint"] == files["published_checkpoint"]
                and hashes["checkpoint_sha256"] == hashes["published_checkpoint_sha256"]
                and current == ANCESTOR_COUNTERS,
                "zero-learning ancestor must be the published CP220544 with no branch/parent")
    else:
        require(isinstance(branch, str)
                and re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", branch) is not None,
                "branch descendant requires its exact safe output-branch name")
        parent_sha = checked_sha(parent_sha, "--expected-parent-checkpoint-sha256")
        require(any(current[key] > ANCESTOR_COUNTERS[key] for key in COUNTER_KEYS),
                "branch descendant must carry real counters beyond its ancestor")
    proof_key = str(arguments.publication_state_proof_key)
    require(re.fullmatch(r"[A-Za-z0-9_]{1,96}", proof_key) is not None,
            "publication state proof key is malformed")
    _PINS = {
        **files, **hashes, "published_manifest": published_manifest,
        "migration_source_manifest": source_manifest, "expected_head": head,
        "checkpoint_role": arguments.checkpoint_role,
        "checkpoint_output_branch": branch,
        "expected_parent_checkpoint_sha256": parent_sha,
        "current_counters": current,
        "publication_state_proof_key": proof_key,
    }
    return _PINS


def pins():
    require(isinstance(_PINS, dict), "configure explicit v6 pins before export")
    return _PINS


def v5():
    global _V5, _V5_FRAME_SUMMARY
    if _V5 is not None:
        return _V5
    require(sha256(V5_ADAPTER) == V5_ADAPTER_SHA,
            "reviewed v5 media adapter bytes changed; review required")
    spec = importlib.util.spec_from_file_location("_rr_v6_reviewed_v5_media", V5_ADAPTER)
    require(spec is not None and spec.loader is not None,
            "cannot load reviewed v5 media adapter")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _V5 = module
    _V5_FRAME_SUMMARY = module.frame_summary
    return module


def display_id():
    p = pins()
    step = p["current_counters"]["global_policy_decisions"]
    return (f"CP{step} ANCESTOR v6" if p["checkpoint_role"] ==
            "published-ancestor-zero-learning" else
            f"CP{step} BRANCH {p['checkpoint_output_branch']} v6")


def media():
    """Narrowly adapt only the SHA-pinned outputs media module."""
    global _MEDIA
    if _MEDIA is not None:
        return _MEDIA
    prior = v5()
    module = prior.media()
    _MEDIA = module
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
    branch_credit = {key: pins()["current_counters"][key] - ANCESTOR_COUNTERS[key]
                     for key in COUNTER_KEYS}
    assignments[0].value = ast.Constant(
        f"{display_id()} | CONTACT-ONSET v6 + FL/RR/WHEELv4 + AUX | "
        f"branch PPO +{branch_credit['ppo_updates']}")
    ast.fix_missing_locations(node)
    exec(compile(ast.Module(body=[node], type_ignores=[]),
                 str(module.__file__) + ":v6_label_only", "exec"), module.__dict__)
    return module


def checkpoint_identity(manifest):
    """Bind v6 publication or its exact explicitly routed PPO descendant."""
    p, m = pins(), media()
    inherited = m.shared().checkpoint_identity(manifest)
    require(Path(inherited["checkpoint"]).resolve() == p["checkpoint"]
            and inherited["checkpoint_sha256"] == p["checkpoint_sha256"]
            and inherited["decisions"] == p["current_counters"]["global_policy_decisions"],
            "sealed video does not load the explicitly pinned v6 checkpoint")
    metadata = m.read_json(Path(inherited["manifest"]))
    published = m.read_json(p["published_manifest"])
    source = m.read_json(p["migration_source_manifest"])
    plan = m.read_json(p["migration_plan"])
    publication = m.read_json(p["publication"])
    runtime = metadata.get("runtime_contract") or {}
    published_runtime = published.get("runtime_contract") or {}
    policy = metadata.get("policy_contract") or {}
    require(all(metadata.get(key) == value
                for key, value in p["current_counters"].items())
            and runtime.get("source_git_commit") == p["expected_head"]
            and runtime == manifest.get("runtime_contract")
            and published_runtime == runtime
            and policy.get("version") == m.POLICY_VERSION
            and policy.get("observation_layout") == m.OBSERVATION_LAYOUT
            and policy.get("observation_dimension") == 410
            and policy.get("raw_action_dimension") == 12,
            "checkpoint counters/runtime or unchanged same410 policy differ")

    receipt = published.get(MIGRATION_KEY)
    require(isinstance(receipt, dict) and receipt.get("schema") == SCHEMA
            and Path(receipt.get("plan_path", "")).resolve() == p["migration_plan"]
            and receipt.get("plan_sha256") == p["migration_plan_sha256"]
            and {key: value for key, value in receipt.items()
                 if key not in ("plan_path", "plan_sha256")} == plan
            and published.get("resume_migration") == receipt
            and metadata.get(MIGRATION_KEY) == receipt,
            "v6 receipt is not the exact immutable migration publication")
    factor = plan.get(FACTOR_KEY) or {}
    expected_selection = dict(ANCESTOR_SOURCE)
    require(plan.get("schema") == SCHEMA and factor.get("schema") == SCHEMA
            and plan.get("source_selection") == expected_selection
            and factor.get("source_selection") == expected_selection
            and factor.get("counter_origin") == ANCESTOR_COUNTERS
            and plan.get("source_checkpoint_sha256") ==
                p["migration_source_checkpoint_sha256"]
            and plan.get("source_manifest_sha256") ==
                p["migration_source_manifest_sha256"]
            and Path(plan.get("source_checkpoint", "")).resolve() ==
                p["migration_source_checkpoint"]
            and plan.get("target_git_commit") == p["expected_head"],
            "v6 plan source role, source bytes, counters, or target HEAD differ")
    require(source.get("checkpoint_sha256") == ANCESTOR_SOURCE["checkpoint_sha256"]
            and all(source.get(key) == value for key, value in ANCESTOR_COUNTERS.items())
            and (source.get("runtime_contract") or {}).get("source_git_commit") ==
                ANCESTOR_SOURCE["source_git_commit"]
            and source.get(MIGRATION_KEY) is None
            and isinstance(source.get(PRIOR_MIGRATION_KEY), dict),
            "migration source is not the registered immutable c531 v5 ancestor")

    flags = {
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
    require(all(factor.get(key) is value for key, value in flags.items())
            and all(factor.get("added_" + name) == 0 for name in
                    ("policy_decisions", "ppo_updates", "optimizer_steps",
                     "auxiliary_updates")),
            "v6 semantic scope or zero-credit migration claims differ")
    observation = factor.get("observation_contract") or {}
    require(observation.get("source_policy_contract") == policy
            and observation.get("target_policy_contract") == policy
            and observation.get("observation_layout") == m.OBSERVATION_LAYOUT
            and observation.get("observation_dimension") == 410
            and observation.get("action_dimension") == 12,
            "v6 changed the saved policy kernel or observation shape")
    effective = factor.get("effective_execution_semantics") or {}
    require(factor.get("source_feedback_revision") == SOURCE_FEEDBACK
            and factor.get("target_feedback_revision") == REVISION
            and effective.get("search_semantics") == SEARCH
            and effective.get("maximum_total_travel_deg") == 53
            and effective.get("maximum_total_exposure_s") == 45
            and effective.get("contact_onset_extra_deg") == 1
            and effective.get("contact_onset_extra_active_s") == 1
            and effective.get("required_public_progress_peak_maximum_m") == .001
            and effective.get("budget_recharges") is False
            and effective.get("sensor_TOP_confirmation_unchanged") is True
            and effective.get("RL_air_releases_capture") is False,
            "v6 contact-onset limits or unchanged TOP criterion differ")
    reviewed = factor.get("reviewed_code_sha256") or {}
    require(reviewed and plan.get("allowed_changed_files") == sorted(reviewed)
            and all((runtime.get("files") or {}).get(path) == digest
                    for path, digest in reviewed.items())
            and plan.get("target_runtime_content_sha256") ==
                runtime.get("runtime_content_sha256"),
            "frozen v6 runtime/plan file binding differs")
    preserved = factor.get("preserved_metadata_sha256")
    require(isinstance(preserved, dict) and preserved,
            "v6 factor lacks migration-time preserved metadata bindings")
    for key, digest in preserved.items():
        require(key in source and m.json_digest(source[key]) == digest,
                f"v6 source metadata differs at {key}")
        require(published.get(key) == source[key],
                f"zero-update v6 publication changed {key}")

    require(Path(publication.get("checkpoint", "")).resolve() ==
                p["published_checkpoint"]
            and publication.get("checkpoint_sha256") ==
                p["published_checkpoint_sha256"]
            and publication.get("source_checkpoint_sha256") ==
                p["migration_source_checkpoint_sha256"]
            and publication.get("target_git_commit") == p["expected_head"]
            and publication.get("save_load_round_trip") is True
            and publication.get(p["publication_state_proof_key"]) is True,
            "v6 save/reload publication proof differs from explicit pins")

    branch = metadata.get("rr_capture_transfer_branch") or {}
    learned = {key: p["current_counters"][key] - ANCESTOR_COUNTERS[key]
               for key in COUNTER_KEYS}
    require(branch.get("counter_origin") == ANCESTOR_COUNTERS
            and metadata.get("rr_capture_transfer_branch_counts") == learned,
            "RR branch origin/counts do not equal the evaluated checkpoint counters")
    if p["checkpoint_role"] == "published-ancestor-zero-learning":
        require(metadata == published and learned == dict.fromkeys(COUNTER_KEYS, 0)
                and metadata.get("checkpoint_output_routing") is None,
                "published ancestor was assigned branch learning or routing")
    else:
        name = p["checkpoint_output_branch"]
        output_root = (HERE / "branches" / name).resolve()
        route = metadata.get("checkpoint_output_routing")
        expected_route = {
            "schema": "wlr50_clean.checkpoint_output_routing.v1",
            "branch": name,
            "output_root": str(output_root),
            "main_latest_pointer_promotion": False,
            "source_selection": expected_selection,
        }
        ancestry = metadata.get("resume_ancestry") or {}
        parent = ancestry.get("source_checkpoint") or {}
        require(p["checkpoint"].is_relative_to(output_root / "checkpoints/history")
                and route == expected_route
                and parent.get("checkpoint_sha256") ==
                    p["expected_parent_checkpoint_sha256"],
                "branch checkpoint path/routing/immediate parent binding differs")
    return {
        **inherited,
        "rr_capture_control_revision": TARGET_REVISION,
        "rr_capture_feedback_revision": REVISION,
        "checkpoint_role": p["checkpoint_role"],
        "checkpoint_output_branch": p["checkpoint_output_branch"],
        "historical_ancestor_weight_counters": ANCESTOR_COUNTERS,
        "RR_branch_learning": learned,
        "evaluation_added_policy_decisions": 0,
        "evaluation_added_PPO_updates": 0,
        "migration_added_optimization": 0,
        "v6_plan": str(p["migration_plan"]),
        "v6_plan_sha256": p["migration_plan_sha256"],
        "v6_publication": str(p["publication"]),
        "v6_publication_sha256": p["publication_sha256"],
        "effective_execution_semantics": effective,
        "inherited_AUX_separate_from_PPO": True,
    }


def validate_snapshot(state):
    """Validate the sealed 14-scalar v6 snapshot without live imports."""
    prior, m = v5(), media()
    prior.base()._SNAPSHOT_VALIDATOR(state)
    feature_names = tuple(m.RR_V2_FEATURE_NAMES)
    metadata = {"schema", "version", "feedback_revision",
        "window_reference_semantics", "capture_search_semantics", "mode_name",
        "reason", "active", "owners", "owner_indices",
        "last_dispatch_physics_tick", "feature_names"}
    m.require(set(state) == set(feature_names) | metadata
        and state.get("feedback_revision") == REVISION
        and state.get("capture_search_semantics") == SEARCH,
        "tick is not the exact reviewed v6 public snapshot")
    mode = state.get("mode")
    travel = state.get("travel_used_deg")
    elapsed = state.get("descent_elapsed_s")
    m.require(type(mode) in (int, float) and float(mode).is_integer()
        and 0 <= mode <= 7 and 0 <= travel <= 53 + 1e-9
        and 0 <= elapsed <= 45 + 1e-9,
        "v6 public mode/travel/exposure is malformed")
    knee_elapsed = max(travel - 20.0, 0.0)
    hip_elapsed = elapsed - knee_elapsed
    hip_travel = min(travel, 20.0)
    m.require(hip_travel / 2.0 - 1e-9 <= hip_elapsed
        <= min(12.0, hip_travel) + 1e-9,
        "v6 public hip/knee travel and exposure are inconsistent")
    if mode == 6:
        m.require(travel >= 20 and state.get("window_start_gap_m") <= .025
            and state.get("window_elapsed_s") < 2,
            "v6 DESCEND_PROGRESS lacks public fresh near-top credit")
    if travel > 52 + 1e-9:
        m.require(state.get("window_start_gap_m") <= .001,
            "v6 contact-onset degree lacks its public <=1mm peak-gap evidence")
    if mode == 7:
        m.require(state.get("contact_seen") == 1 and state.get("retired") == 0,
                  "v6 CAPTURED_FOLLOW lacks contact history or live ownership")


def frame_summary(row, native_row):
    out = _V5_FRAME_SUMMARY(row, native_row)
    state = row["rr_capture_assist"]
    travel = float(state["travel_used_deg"])
    out.update(
        rr_v6_mode_index=int(state["mode"]),
        rr_v6_feedback_revision=state["feedback_revision"],
        rr_v6_capture_search_semantics=state["capture_search_semantics"],
        rr_v6_total_travel_deg=travel,
        rr_v6_base_search_used_deg=min(travel, 40.0),
        rr_v6_progress_reserve_used_deg=min(max(travel - 40.0, 0.0), 12.0),
        rr_v6_contact_onset_used_deg=max(travel - 52.0, 0.0),
        rr_v6_total_exposure_s=float(state["descent_elapsed_s"]),
        rr_v6_descend_progress=state["mode_name"] == "DESCEND_PROGRESS",
        rr_v6_captured_follow=state["mode_name"] == "CAPTURED_FOLLOW",
        rr_v6_public_captured_hip_target_deg=float(state["hip_target_deg"]),
        rr_v6_public_captured_knee_target_deg=float(state["knee_hold_deg"]),
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
                      "unsealed/partial v6 tick ledger")
            row, native = json.loads(raw), json.loads(native_raw)
            m.require(row.get("episode_physics_tick") == count
                and math.isclose(float(row["sim_time_s"]), count / 120., abs_tol=1e-10),
                "v6 tick sequence gap")
            item = frame_summary(row, native)
            offset = (item["dispatch_to_episode_tick_offset"]
                      if offset is None else offset)
            m.require(item["dispatch_to_episode_tick_offset"] == offset,
                      "v6 dispatch offset changed")
            events = {
                "FIRST_RR_OWNER": bool(item["rr_assist_owner_indices"]),
                "FIRST_RR_CROSS": item["rr_front_edge_crossed"] is True,
                "FIRST_DESCEND_PROGRESS": item["rr_v6_descend_progress"],
                "FIRST_CONTACT_ONSET_INCREMENT":
                    item["rr_v6_contact_onset_used_deg"] > 0,
                "FIRST_CAPTURED_FOLLOW": item["rr_v6_captured_follow"],
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
        m.require(natives.readline() == b"", "native ledger extends past v6 episode")
    m.require(count == candidate["endpoint"] and set(frames) == wanted
              and last is not None, "exact full v6 episode/frame coverage unavailable")
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
        f"v6 CONTACT-ONSET | migration +0 | this branch learning +{credit['global_policy_decisions']}/+{credit['ppo_updates']}/+{credit['optimizer_steps']} | RR/P10/RL={row['run_RR_window_reached']}/{row['run_P10_reached']}/{row['run_RL_window_reached']}",
        f"RR {row['rr_assist_mode']} ({row['rr_assist_reason']}) mode={row['rr_v6_mode_index']} | travel base/reserve/onset={f(row['rr_v6_base_search_used_deg'])}/{f(row['rr_v6_progress_reserve_used_deg'])}/{f(row['rr_v6_contact_onset_used_deg'])}deg",
        f"public RR hip target/actual {f(row['rr_v6_public_captured_hip_target_deg'])}/{f(row['rr_hip_actual_deg'])}deg | knee target/actual {f(row['rr_v6_public_captured_knee_target_deg'])}/{f(row['rr_knee_actual_deg'])}deg | captured-follow={row['rr_v6_captured_follow']}",
        f"RR gap {f(row['rr_post_gap_mm'])}mm depth {f(row['rr_front_distance_mm'])}mm {row['rr_post_contact']} | TOP-ever={row['run_RR_TOP_observed']} placed={row['run_RR_placed']} RL-placed={row['run_RL_placed']}",
        f"wheel-v4 PRE envelope={row['wheel_envelope_active']} changed={row['wheel_projection_changed']} {row['wheel_action']} gain={f(row['wheel_gain'])} | controller credit is not PPO",
        m._wheel_line("source nominal wheel rad/s: ", row["source_nominal_wheel_rad_s"]),
        m._wheel_line("FINAL target canonical rad/s: ", row["wheel_final_canonical_rad_s"]),
        m._wheel_line("MEASURED canonical qdot rad/s: ", row["wheel_actual_canonical_rad_s"]),
        m._wheel_line("wheel-v4 controller delta rad/s: ", row["wheel_controller_delta"]),
        f"1x 15fps | t={row['time_s']:.3f}s tick={row['tick']} {row['phase']} | current contact != placed history; sensor result controls success labels",
    ]


def detail_plan(rows):
    m = media()
    rr = rows[-1]["run_RR_window_reached"]
    if rr:
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
        "filename": f"CP{step}_DET_v6_{name}_detail.mp4",
        "title": f"DETAIL | {display_id()} | {title}",
        "requested_RR_detail_unavailable_reason": reason}


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
    m.require(0 < len(rows) <= 3000,
              "full source must remain <=200 s at native 15fps")
    result, success, termination = m.outcome(candidate["manifest"])
    step = pins()["current_counters"]["global_policy_decisions"]
    suffix = "SUCCESS" if success else "INCOMPLETE"
    destination.mkdir(parents=True)
    full = m.encode_full(candidate, rows,
        destination / f"CP{step}_DET_v6_full_attempt_{suffix}.mp4",
        result=result, rr_reached=rows[-1]["run_RR_window_reached"],
        feedback_v2=True, ffmpeg=ffmpeg)
    print("FULL_PLAYABLE_VALIDATED " + full["output"], flush=True)
    detail = m.encode_detail(Path(full["output"]), rows, destination, ffmpeg=ffmpeg)
    pair = m.encode_pair(baseline, {**candidate, "frame_count": len(rows)},
        Path(full["output"]), destination / f"historical_N_vs_CP{step}_DET_v6.mp4",
        historical_version=m.DEFAULT_HISTORICAL_VERSION,
        feedback_v2=True, ffmpeg=ffmpeg)
    evidence = destination / f"CP{step}_v6_selected_fourwheel_evidence.json"
    m.write_new_json(evidence, {
        "schema": "wlr50_clean.rr_contact_onset_v6_video_evidence.v1",
        "source": str(source),
        "capture_assist_ticks_sha256": m.sha256(candidate["tick_path"]),
        "native_tick_audit_sha256": m.sha256(candidate["native_tick_path"]),
        "selection": "first actually observed v6/RR/P10/RL/wheel/contact milestones plus terminal; absent milestones are not invented",
        "pre_and_post_ticks_separate": True,
        "rows": selected,
    })
    credit = {key: pins()["current_counters"][key] - ANCESTOR_COUNTERS[key]
              for key in COUNTER_KEYS}
    receipt = {
        "schema": "wlr50_clean.rr_contact_onset_v6_video_export.v1",
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
        "physical_result": result,
        "physical_task_success": success,
        "termination_reason": termination,
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
            for key, value in pins().items()
            if key not in ("current_counters",)
        },
        "expected_checkpoint_counters": pins()["current_counters"],
        "code_binding": {
            "adapter_sha256": m.sha256(Path(__file__)),
            "v5_media_adapter_sha256": V5_ADAPTER_SHA,
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
                        help="exact sealed v6 semantic-video source directory")
    result.add_argument("--destination", type=Path, required=True)
    result.add_argument("--expected-head", required=True)
    result.add_argument("--checkpoint", type=Path, required=True)
    result.add_argument("--checkpoint-sha256", required=True)
    result.add_argument("--published-v6-checkpoint", type=Path, required=True)
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
