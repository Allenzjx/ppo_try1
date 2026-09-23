"""DRAFT: sealed CP220544 ancestor/v5 video export; never runs physics.

This outputs-only adapter is intentionally dormant until its explicit source
has sealed.  It pins the reviewed v4 media adapter, validates the new v5
same410 full-state migration against its immutable ancestor, then reuses only
the established frame/decode/encode machinery.  It never imports the live
runtime's assist validator, never assigns the later CP221184 learning credit
to these ancestor weights, and derives RR/RL claims only from this episode.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V4_ADAPTER = HERE / "export_rr_capture_video_v4_CP221184.py"
V4_ADAPTER_SHA = "c48d210060a17617bab6cbc80e2fac26d30980309753d4f76c6b136ded840f68"
HEAD = "c53119ab332fe048668e66f0543889a128443ca5"
CP = (HERE / "checkpoints/history/"
      "checkpoint_rr_progress_handoff_v5_ancestor_step_000220544_gc53119ab332f.pt")
CP_SHA = "308122a3c8e760733fbe8370bfdf399a25718148c82333deb41a1265e4e13895"
PLAN = HERE / "CP220544_RR410_progress_handoff_v5_ancestor_gc53119ab332f_migration.json"
PLAN_SHA = "78d741c6aeec61e28989853b5a96cc61467db3bdda540a41b9282b03d3730d6f"
PUBLICATION = HERE / "CP220544_RR410_progress_handoff_v5_ancestor_gc53119ab332f_publication.json"
PUBLICATION_SHA = "a14295451d7d3fc347a358ad2f5a7046b02d1a167b7c2b476de1157beeb54187"
MIGRATION_KEY = "rr_progress_handoff_v5_migration"
FACTOR_KEY = "rr_progress_handoff_v5_factor"
PRIOR_MIGRATION_KEY = "rr_carry_handoff_v4_migration"
SCHEMA = "wlr50_clean.rr_progress_handoff_same410.v5"
REVISION = "progress_reserve_captured_incremental_v4"
SEARCH = ("hip20_knee20_then_single_progress_earned_near_top_knee12_total52_"
          "exposure44_no_recharge_public_mode6_credit_captured_targets_add_issued_"
          "N_and_requested_residual_deltas_RL_current_TOP_retirement")
WHEEL_MODE = "rr_capture_support_forward_projection_v1"
METHOD = ("PPO_PLUS_FL_RR_CAPTURE_ASSISTS_PLUS_SUPPORT_WHEEL_V4_"
          "WITH_INHERITED_LIMITED_AUX")
COUNTERS = {"global_policy_decisions": 220544,
            "ppo_updates": 1688, "optimizer_steps": 33760}
RR_BRANCH_LEARNING = {key: 0 for key in COUNTERS}
SOURCE_CP_SHA = "facb915397f06251bf3e9c4f32cc2f2389262ff4c36f20c3a97b051bbe046892"
SOURCE_MANIFEST_SHA = "a10975d12695745ddfe6873fcc6213c9fb7ac2443957664914af598f68ac29ec"
SOURCE_HEAD = "57ae41eb4b57f63d4258e79b6083053166cc3cb3"

_BASE = None
_MEDIA = None


def base():
    global _BASE
    if _BASE is not None:
        return _BASE
    import hashlib
    if hashlib.sha256(V4_ADAPTER.read_bytes()).hexdigest() != V4_ADAPTER_SHA:
        raise RuntimeError("reviewed v4 media adapter bytes changed; review required")
    spec = importlib.util.spec_from_file_location("_rr_v5_reviewed_v4_media", V4_ADAPTER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load reviewed v4 media adapter")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _BASE = module
    return module


def media():
    """Load and narrowly adapt the SHA-pinned outputs-only media code."""
    global _MEDIA
    if _MEDIA is not None:
        return _MEDIA
    b = base()
    module = b.media()
    _MEDIA = module  # Set before callbacks can re-enter this function.
    module.checkpoint_identity = checkpoint_identity
    module.validate_historical_rr_snapshot = validate_snapshot
    module.FEEDBACK_V2_REVISION = REVISION
    module.panel_lines = panel_lines
    module.detail_plan = detail_plan

    # Reuse the established comparison encoder and change only its candidate
    # label in the pinned AST.  Historical N and freeze behavior stay intact.
    tree = ast.parse(module.Path(module.__file__).read_text(encoding="utf-8"))
    node = next(value for value in tree.body
                if isinstance(value, ast.FunctionDef) and value.name == "encode_pair")
    assignments = [value for value in ast.walk(node) if isinstance(value, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "candidate_label"
                for target in value.targets)]
    module.require(len(assignments) == 1, "comparison label AST differs")
    assignments[0].value = ast.Constant(
        "CP220544 ANCESTOR | v5 CONTROL + FL/RR/WHEELv4 + AUX | RR PPO +0")
    ast.fix_missing_locations(node)
    exec(compile(ast.Module(body=[node], type_ignores=[]),
                 str(module.__file__) + ":v5_label_only", "exec"), module.__dict__)
    return module


def checkpoint_identity(manifest):
    """Bind the exact v5 ancestor weights, lineage, zero credit and runtime."""
    m = media()
    inherited = m.shared().checkpoint_identity(manifest)
    m.require(Path(inherited["checkpoint"]).resolve() == CP.resolve()
        and inherited["checkpoint_sha256"] == CP_SHA
        and inherited["decisions"] == COUNTERS["global_policy_decisions"],
        "this adapter accepts only the frozen CP220544 v5 ancestor checkpoint")
    metadata = m.read_json(Path(inherited["manifest"]))
    m.require(all(metadata.get(key) == value for key, value in COUNTERS.items()),
              "v5 ancestor checkpoint counters differ")
    runtime = metadata.get("runtime_contract") or {}
    policy = metadata.get("policy_contract") or {}
    m.require(runtime.get("source_git_commit") == HEAD
        and runtime == manifest.get("runtime_contract")
        and policy.get("version") == m.POLICY_VERSION
        and policy.get("observation_layout") == m.OBSERVATION_LAYOUT
        and policy.get("observation_dimension") == 410
        and policy.get("raw_action_dimension") == 12,
        "sealed source is not the exact same410/c531 v5 runtime")
    m.require(m.sha256(PLAN) == PLAN_SHA and m.sha256(PUBLICATION) == PUBLICATION_SHA,
              "immutable v5 plan/publication changed")
    plan, publication = m.read_json(PLAN), m.read_json(PUBLICATION)
    receipt = metadata.get(MIGRATION_KEY)
    m.require(isinstance(receipt, dict) and receipt.get("schema") == SCHEMA
        and Path(receipt.get("plan_path", "")).resolve() == PLAN.resolve()
        and receipt.get("plan_sha256") == PLAN_SHA
        and {key: value for key, value in receipt.items()
             if key not in ("plan_path", "plan_sha256")} == plan
        and metadata.get("resume_migration") == receipt,
        "saved v5 receipt differs from its immutable plan")
    factor = plan.get(FACTOR_KEY) or {}
    source_selection = factor.get("source_selection")
    expected_selection = {
        "checkpoint_sha256": SOURCE_CP_SHA,
        "manifest_sha256": SOURCE_MANIFEST_SHA,
        "source_git_commit": SOURCE_HEAD,
        "counters": COUNTERS,
        "source_role": "front_validated_ancestor_control_eval",
        "registry_bridge_required": False,
    }
    m.require(factor.get("schema") == SCHEMA
        and factor.get("counter_origin") == COUNTERS
        and source_selection == expected_selection
        and plan.get("source_selection") == expected_selection
        and metadata.get("rr_capture_transfer_branch_counts") == RR_BRANCH_LEARNING,
        "v5 ancestor source role/counts or zero RR learning credit differs")
    rr_branch = metadata.get("rr_capture_transfer_branch") or {}
    m.require(rr_branch.get("counter_origin") == COUNTERS
        and all(COUNTERS[key] - rr_branch["counter_origin"][key] == 0
                for key in COUNTERS),
        "v5 ancestor RR branch was reset, recreated or assigned learned credit")

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
        "candidate_evaluation_only": True,
        "latest_pointer_promotion_authorized": False,
        "latest_learned_weights_preserved": False,
        "latest_learned_policy_equivalence_claimed": False,
    }
    m.require(all(factor.get(key) is value for key, value in flags.items())
        and all(factor.get("added_" + name) == 0 for name in
                ("policy_decisions", "ppo_updates", "optimizer_steps",
                 "auxiliary_updates")),
        "v5 semantic/learning-credit claims differ")
    observation = factor.get("observation_contract") or {}
    m.require(observation.get("source_policy_contract") == policy
        and observation.get("target_policy_contract") == policy
        and observation.get("observation_layout") == m.OBSERVATION_LAYOUT
        and observation.get("observation_dimension") == 410
        and observation.get("action_dimension") == 12,
        "v5 must retain the exact same410 v1 policy/kernel")
    effective = factor.get("effective_execution_semantics") or {}
    m.require(factor.get("source_feedback_revision") == "window_peak_hip_then_knee_v3"
        and factor.get("target_feedback_revision") == REVISION
        and effective.get("search_semantics") == SEARCH
        and effective.get("maximum_total_travel_deg") == 52
        and effective.get("maximum_total_exposure_s") == 44
        and effective.get("earned_reserve_degrees") == 12
        and effective.get("earned_reserve_exposure_s") == 12
        and effective.get("reserve_recharges") is False
        and effective.get("RL_air_releases_capture") is False,
        "v5 progress reserve/captured-follow semantics differ")
    reviewed = factor.get("reviewed_code_sha256") or {}
    m.require(reviewed and plan.get("allowed_changed_files") == sorted(reviewed)
        and all((runtime.get("files") or {}).get(path) == digest
                for path, digest in reviewed.items())
        and plan.get("target_runtime_content_sha256") == runtime.get("runtime_content_sha256")
        and plan.get("target_git_commit") == HEAD,
        "frozen c531 runtime/plan hash binding differs")

    source = Path(plan.get("source_checkpoint", "")).resolve(strict=True)
    source_manifest = m._sidecar_from_checkpoint(source).resolve(strict=True)
    m.require(m.sha256(source) == SOURCE_CP_SHA
        and m.sha256(source_manifest) == SOURCE_MANIFEST_SHA
        and plan.get("source_checkpoint_sha256") == SOURCE_CP_SHA
        and plan.get("source_manifest_sha256") == SOURCE_MANIFEST_SHA
        and plan.get("source_git_commit") == SOURCE_HEAD,
        "v5 immutable v4 ancestor source binding changed")
    source_metadata = m.read_json(source_manifest)
    m.require(Path(source_metadata.get("checkpoint_path", "")).resolve() == source
        and source_metadata.get("checkpoint_sha256") == SOURCE_CP_SHA
        and all(source_metadata.get(key) == value for key, value in COUNTERS.items())
        and (source_metadata.get("runtime_contract") or {}).get("source_git_commit") == SOURCE_HEAD
        and source_metadata.get(MIGRATION_KEY) is None
        and source_metadata.get("resume_migration") ==
            source_metadata.get(PRIOR_MIGRATION_KEY)
        and m.json_digest(source_metadata.get(PRIOR_MIGRATION_KEY)) ==
            factor.get("source_v4_receipt_sha256")
        and m.json_digest(source_metadata.get("resume_migration")) ==
            factor.get("source_resume_migration_sha256"),
        "v5 source is not the exact registered v4 ancestor lineage")
    preserved = factor.get("preserved_metadata_sha256")
    m.require(isinstance(preserved, dict) and preserved,
              "v5 factor lacks migration-time preserved lineage")
    for key, digest in preserved.items():
        m.require(key in source_metadata and m.json_digest(source_metadata[key]) == digest,
                  f"v5 source metadata changed: {key}")
        m.require(metadata.get(key) == source_metadata[key],
                  f"v5 publication changed preserved lineage/state: {key}")

    m.require(publication.get("checkpoint_sha256") == CP_SHA
        and Path(publication.get("checkpoint", "")).resolve() == CP.resolve()
        and publication.get("source_checkpoint_sha256") == SOURCE_CP_SHA
        and publication.get("target_git_commit") == HEAD
        and publication.get("save_load_round_trip") is True
        and publication.get("exact_same410_source_state_preserved") is True
        and all(publication.get(key) == value for key, value in COUNTERS.items())
        and publication.get("rr_capture_transfer_branch_counts") == RR_BRANCH_LEARNING
        and all(publication.get("migration_added_" + name) == 0 for name in
                ("policy_decisions", "ppo_updates", "optimizer_steps",
                 "auxiliary_updates"))
        and publication.get("physical_evaluation") == "NOT_YET_EVALUATED",
        "actual v5 save/reload publication differs")
    return {**inherited,
        "rr_capture_control_revision": "v5_progress_incremental_captured_handoff",
        "rr_capture_transfer_branch_counts": RR_BRANCH_LEARNING,
        "historical_ancestor_weight_counters": COUNTERS,
        "later_CP221184_learning_credit_borrowed": False,
        "evaluation_added_policy_decisions": 0,
        "evaluation_added_PPO_updates": 0,
        "migration_added_optimization": 0,
        "v5_plan": str(PLAN), "v5_plan_sha256": PLAN_SHA,
        "v5_publication": str(PUBLICATION),
        "v5_publication_sha256": PUBLICATION_SHA,
        "effective_execution_semantics": effective,
        "inherited_AUX_separate_from_PPO": True}


def validate_snapshot(state):
    """Validate the sealed v5 public 14-scalar codec without live imports."""
    b, m = base(), media()
    b._SNAPSHOT_VALIDATOR(state)
    feature_names = tuple(m.RR_V2_FEATURE_NAMES)
    metadata = {"schema", "version", "feedback_revision",
        "window_reference_semantics", "capture_search_semantics", "mode_name",
        "reason", "active", "owners", "owner_indices",
        "last_dispatch_physics_tick", "feature_names"}
    m.require(set(state) == set(feature_names) | metadata
        and state.get("feedback_revision") == REVISION
        and state.get("capture_search_semantics") == SEARCH,
        "tick is not the exact reviewed v5 public snapshot")
    mode = state.get("mode")
    travel = state.get("travel_used_deg")
    elapsed = state.get("descent_elapsed_s")
    m.require(type(mode) in (int, float) and float(mode).is_integer()
        and 0 <= mode <= 7 and 0 <= travel <= 52 + 1e-9
        and 0 <= elapsed <= 44 + 1e-9,
        "v5 public mode/travel/exposure is malformed")
    knee_elapsed = max(travel - 20.0, 0.0)
    hip_elapsed = elapsed - knee_elapsed
    hip_travel = min(travel, 20.0)
    m.require(hip_travel / 2.0 - 1e-9 <= hip_elapsed
        <= min(12.0, hip_travel) + 1e-9,
        "v5 public hip/knee travel and exposure are inconsistent")
    if mode == 6:
        m.require(travel >= 20 and state.get("window_start_gap_m") <= .025
            and state.get("window_elapsed_s") < 2,
            "DESCEND_PROGRESS lacks its public fresh near-top credit")
    if mode == 7:
        m.require(state.get("contact_seen") == 1 and state.get("retired") == 0,
                  "CAPTURED_FOLLOW lacks contact history or live ownership")


def frame_summary(row, native_row):
    b = base()
    out = b.frame_summary(row, native_row)
    state = row["rr_capture_assist"]
    travel = float(state["travel_used_deg"])
    out.update(
        rr_v5_mode_index=int(state["mode"]),
        rr_v5_feedback_revision=state["feedback_revision"],
        rr_v5_capture_search_semantics=state["capture_search_semantics"],
        rr_v5_total_travel_deg=travel,
        rr_v5_base_search_used_deg=min(travel, 40.0),
        rr_v5_progress_reserve_used_deg=max(travel - 40.0, 0.0),
        rr_v5_total_exposure_s=float(state["descent_elapsed_s"]),
        rr_v5_descend_progress=state["mode_name"] == "DESCEND_PROGRESS",
        rr_v5_captured_follow=state["mode_name"] == "CAPTURED_FOLLOW",
        rr_v5_public_captured_hip_target_deg=float(state["hip_target_deg"]),
        rr_v5_public_captured_knee_target_deg=float(state["knee_hold_deg"]),
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
                      "unsealed/partial v5 tick ledger")
            row, native = json.loads(raw), json.loads(native_raw)
            m.require(row.get("episode_physics_tick") == count
                and math.isclose(float(row["sim_time_s"]), count / 120., abs_tol=1e-10),
                "v5 tick sequence gap")
            item = frame_summary(row, native)
            offset = (item["dispatch_to_episode_tick_offset"]
                      if offset is None else offset)
            m.require(item["dispatch_to_episode_tick_offset"] == offset,
                      "v5 dispatch offset changed")
            events = {
                "FIRST_RR_OWNER": bool(item["rr_assist_owner_indices"]),
                "FIRST_RR_CROSS": item["rr_front_edge_crossed"] is True,
                "FIRST_DESCEND_PROGRESS": item["rr_v5_descend_progress"],
                "FIRST_CAPTURED_FOLLOW": item["rr_v5_captured_follow"],
                "FIRST_WHEEL_ENVELOPE": item["wheel_envelope_active"],
                "FIRST_WHEEL_PROJECTION": item["wheel_projection_changed"],
                "FIRST_RR_TOP": item["rr_post_contact"] == "TOP",
                "FIRST_RR_PLACED": item["rr_placed_history"] is True,
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
        m.require(natives.readline() == b"", "native ledger extends past v5 episode")
    m.require(count == candidate["endpoint"] and set(frames) == wanted
              and last is not None, "exact full v5 episode/frame coverage unavailable")
    milestones["TERMINAL"] = {"milestone": "TERMINAL", **last}
    rows = [frames[value.sim_step] for value in ledger]
    rr_reached = m.rr_window_reached(rows)
    rl_reached = any(value["phase"] in ("P11", "P12", "P13") for value in rows)
    rl_placed = any(value["rl_placed_history"] is True for value in rows)
    for item in rows:
        item.update(run_RR_window_reached=rr_reached,
                    run_RL_window_reached=rl_reached,
                    run_RL_placed=rl_placed)
    return rows, list(milestones.values())


def panel_lines(row, result, *, rr_reached, feedback_v2):
    m, f = media(), media()._fmt
    return [
        f"CP220544 ANCESTOR DET | PPO + FL/RR ASSISTS + SUPPORT-WHEEL v4 + INHERITED LIMITED AUX | {result}",
        f"v5 CONTROL progress-reserve/captured-follow | weights 220544/1688/33760 | RR learning +0/+0/+0 | RR/RL reached={rr_reached}/{row['run_RL_window_reached']}",
        f"RR {row['rr_assist_mode']} ({row['rr_assist_reason']}) mode={row['rr_v5_mode_index']} | total/base/reserve travel {f(row['rr_v5_total_travel_deg'])}/{f(row['rr_v5_base_search_used_deg'])}/{f(row['rr_v5_progress_reserve_used_deg'])}deg",
        f"public captured hip target/actual {f(row['rr_v5_public_captured_hip_target_deg'])}/{f(row['rr_hip_actual_deg'])}deg | knee target/actual {f(row['rr_v5_public_captured_knee_target_deg'])}/{f(row['rr_knee_actual_deg'])}deg | follow={row['rr_v5_captured_follow']}",
        f"RR gap {f(row['rr_post_gap_mm'])}mm depth {f(row['rr_front_distance_mm'])}mm {row['rr_post_contact']} | bearing={row['rr_current_bearing']} placed-history={row['rr_placed_history']} RL-placed={row['run_RL_placed']}",
        f"wheel-v4 PRE envelope={row['wheel_envelope_active']} changed={row['wheel_projection_changed']} {row['wheel_action']} gain={f(row['wheel_gain'])} | no controller credit is PPO",
        m._wheel_line("source nominal wheel rad/s: ", row["source_nominal_wheel_rad_s"]),
        m._wheel_line("FINAL target canonical rad/s: ", row["wheel_final_canonical_rad_s"]),
        m._wheel_line("MEASURED canonical qdot rad/s: ", row["wheel_actual_canonical_rad_s"]),
        m._wheel_line("wheel-v4 controller delta rad/s: ", row["wheel_controller_delta"]),
        f"1x 15fps | t={row['time_s']:.3f}s tick={row['tick']} {row['phase']} | current contact != history; ancestor weights != later learned CP221184",
    ]


def detail_plan(rows):
    m = media()
    reached = m.rr_window_reached(rows)
    if reached:
        start = next(index for index, value in enumerate(rows)
            if value["phase"] in ("P09", "P10", "P11", "P12", "P13")
            or value["rr_lift_carry"] or value["rr_assist_mode"] != "WAIT")
        rl = any(value["phase"] in ("P11", "P12", "P13") for value in rows)
        name = "RR_capture_to_RL" if rl else "RR_capture_RL_not_reached"
        title = ("RR capture through observed RL window and terminal" if rl else
                 "RR capture through terminal - RL NOT REACHED")
        reason = None
    else:
        start = max(0, len(rows) - 60 * m.FPS)
        name, title = "predecessor_failure_tail", \
            "predecessor terminal tail - RR AND RL NOT REACHED"
        reason = "THIS_EPISODE_DID_NOT_REACH_RR_WINDOW"
    return {"kind": name, "start": start, "end": len(rows),
        "filename": f"CP220544_DET_v5_{name}_detail.mp4",
        "title": f"DETAIL | CP220544 ancestor v5 | {title}",
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
    rr, rl = rows[-1]["run_RR_window_reached"], rows[-1]["run_RL_window_reached"]
    suffix = "SUCCESS" if success else "INCOMPLETE"
    destination.mkdir(parents=True)
    full = m.encode_full(candidate, rows,
        destination / f"CP220544_DET_v5_full_attempt_{suffix}.mp4",
        result=result, rr_reached=rr, feedback_v2=True, ffmpeg=ffmpeg)
    print("FULL_PLAYABLE_VALIDATED " + full["output"], flush=True)
    detail = m.encode_detail(Path(full["output"]), rows, destination, ffmpeg=ffmpeg)
    pair = m.encode_pair(baseline, {**candidate, "frame_count": len(rows)},
        Path(full["output"]), destination / "historical_N_vs_CP220544_DET_v5.mp4",
        historical_version=m.DEFAULT_HISTORICAL_VERSION,
        feedback_v2=True, ffmpeg=ffmpeg)
    evidence = destination / "CP220544_v5_selected_fourwheel_evidence.json"
    m.write_new_json(evidence, {
        "schema": "wlr50_clean.rr_capture_v5_video_evidence.v1",
        "source": str(source),
        "capture_assist_ticks_sha256": m.sha256(candidate["tick_path"]),
        "native_tick_audit_sha256": m.sha256(candidate["native_tick_path"]),
        "selection": "first actually observed v5/RR/RL/wheel milestones plus terminal; absent milestones are not invented",
        "pre_and_post_ticks_separate": True,
        "rows": selected,
    })
    receipt = {
        "schema": "wlr50_clean.rr_capture_v5_video_export.v1",
        "source": str(source),
        "source_manifest_sha256": m.sha256(candidate["manifest_path"]),
        "source_run_manifest_sha256": m.sha256(candidate["run_manifest_path"]),
        "checkpoint": candidate["checkpoint"],
        "control_method": METHOD,
        "source_legacy_control_method": candidate["manifest"].get("control_method"),
        "effective_wheel_mode": WHEEL_MODE,
        "RR_assist_feedback_revision": REVISION,
        "RR_assist_capture_search_semantics": SEARCH,
        "historical_ancestor_weight_counters": COUNTERS,
        "RR_branch_learning": RR_BRANCH_LEARNING,
        "later_CP221184_learning_credit_borrowed": False,
        "evaluation_added_PPO_updates": 0,
        "migration_added_optimization": 0,
        "inherited_limited_AUX_is_PPO": False,
        "controller_intervention_is_PPO_learning": False,
        "same_MDP_claimed": False,
        "physical_result": result,
        "physical_task_success": success,
        "termination_reason": termination,
        "source_acceptance_error": candidate["manifest"].get("source_acceptance_error"),
        "RR_window_reached": rr,
        "RL_window_reached": rl,
        "RL_placed": rows[-1]["run_RL_placed"],
        "RL_success_claimed": success and rows[-1]["run_RL_placed"],
        "attempt_classification": ("TASK_SUCCESS" if success else
            "RL_WINDOW_REACHED_INCOMPLETE" if rl else
            "RR_CAPTURE_WINDOW_ONLY" if rr else "PREDECESSOR_ONLY"),
        "full_episode_continuous": True,
        "full_failure_tail_preserved": True,
        "normal_speed": True,
        "single_episode": True,
        "stitched": False,
        "extra_intro_frames": 0,
        "source_container_duration_is_not_physics_duration": True,
        "derived_output_PTS_rebuilt_from_decoded_frame_order_at_15fps": True,
        "source_validation": source_validation,
        "full": full, "detail": detail, "historical_N_comparison": pair,
        "historical_N_is_fresh_same_controller_B": False,
        "selected_fourwheel_evidence": str(evidence),
        "selected_fourwheel_evidence_sha256": m.sha256(evidence),
        "code_binding": {"adapter_sha256": m.sha256(Path(__file__)),
                         "v4_media_adapter_sha256": V4_ADAPTER_SHA},
        "missing_events_are_not_invented": True,
        "targets_are_not_actual_velocity": True,
    }
    m.write_new_json(destination / "export_receipt.json", receipt)
    print(json.dumps({"physical_result": result, "full": full["output"],
        "detail": detail["output"], "historical_N_pair": pair["output"],
        "receipt": str(destination / "export_receipt.json")}, indent=2))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="exact sealed v5 semantic-video source directory")
    parser.add_argument("--destination", type=Path, required=True)
    arguments = parser.parse_args()
    export(arguments.source, arguments.destination)
