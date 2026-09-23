"""DRAFT: sealed CP221184/v4 full/detail/historical-N export, never physics.

Prepared by text inspection only; not executed, decoded, encoded or tested.
The old 26db exporter and all existing movies remain immutable. This adapter
reuses its sealed-source, full-frame/decode, HUD and encoder implementations.
New labels and provenance explicitly disclose controllers and inherited AUX.
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
LEGACY = HERE / "export_rr_capture_video.py"
LEGACY_SHA = "e14d175b1e8d176e7161822950cea99d1b4f67084d026973f5fa905fa51f2e4e"
HEAD = "632295bc49cf56e120db426f106a58ee7e78c3ff"
CP_SHA = "2fed192f9c5141a40f769893d0266ec3b95c91d55a4b3091acc411cd99672f90"
CP = HERE / "checkpoints/history/checkpoint_rr_carry_handoff_v4_step_000221184_g632295bc49cf.pt"
SOURCE = ROOT / "runs/ppo_rr_capture_then_rl_transfer_v1/video_eval/validation/20260923T0546444511040Z_g632295bc49cf_9a60bc74b9b14f6eb7debd0d2e92d914/source"
PLAN = HERE / "CP221184_RR410_carry_handoff_v4_g632295bc49cf_migration.json"
PLAN_SHA = "41101037c95108e8f1044aad4919ddf28308b6e0e25ce7494e54c9c7c5d6cd21"
PUBLICATION = HERE / "CP221184_RR410_carry_handoff_v4_g632295bc49cf_publication.json"
PUBLICATION_SHA = "a6a9bc24cddb332c0616df23ad794cd8d98e124203cb1e0565d0f95be3433481"
MIGRATION_KEY = "rr_carry_handoff_v4_migration"
FACTOR_KEY = "rr_carry_handoff_v4_factor"
SCHEMA = "wlr50_clean.rr_carry_handoff_same410.v4"
MODE = "rr_capture_support_forward_projection_v1"
METHOD = "PPO_PLUS_FL_RR_CAPTURE_ASSISTS_PLUS_SUPPORT_WHEEL_V4_WITH_INHERITED_LIMITED_AUX"
REVISION = "window_peak_hip_then_knee_v3"
SEARCH = "combined_travel_0_40_first20_negative_hip_then20_positive_knee_at1deg_per_s_knee_hold_is_dynamic_target_axis_from_travel_hip_elapsed_equals_total_elapsed_minus_max_travel_minus20_0"
COUNTERS = {"global_policy_decisions": 221184, "ppo_updates": 1693, "optimizer_steps": 33860}
LEARNED = {"global_policy_decisions": 640, "ppo_updates": 5, "optimizer_steps": 100}
_MEDIA = None
_FRAME_SUMMARY = None
_SNAPSHOT_VALIDATOR = None


def media():
    global _MEDIA, _FRAME_SUMMARY, _SNAPSHOT_VALIDATOR
    if _MEDIA is not None:
        return _MEDIA
    import hashlib
    if hashlib.sha256(LEGACY.read_bytes()).hexdigest() != LEGACY_SHA:
        raise RuntimeError("reviewed historical exporter bytes changed; review required")
    spec = importlib.util.spec_from_file_location("_rr_v4_sealed_media", LEGACY)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load reviewed historical exporter")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _MEDIA = module
    _FRAME_SUMMARY = module.frame_summary
    _SNAPSHOT_VALIDATOR = module.validate_historical_rr_snapshot
    # Only this isolated imported module is adapted; no old file is rewritten.
    module.checkpoint_identity = checkpoint_identity
    module.validate_historical_rr_snapshot = validate_snapshot
    module.FEEDBACK_V2_REVISION = REVISION  # Legacy snapshot validator's constant name.
    module.panel_lines = panel_lines
    module.detail_plan = detail_plan
    # Preserve every comparison encoder/freeze/quality rule; replace just its
    # candidate-label assignment, in the SHA-pinned function AST.
    tree = ast.parse(LEGACY.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "encode_pair")
    assignments = [n for n in ast.walk(node) if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "candidate_label" for t in n.targets)]
    module.require(len(assignments) == 1, "comparison label AST differs")
    assignments[0].value = ast.Constant("CP221184 | FL/RR + WHEELv4 + AUX | trained +640/5 | migration 0")
    ast.fix_missing_locations(node)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(LEGACY)+":v4_label_only", "exec"), module.__dict__)
    return module


def checkpoint_identity(manifest):
    m = media()
    inherited = m.shared().checkpoint_identity(manifest)
    m.require(Path(inherited["checkpoint"]).resolve() == CP.resolve()
        and inherited["checkpoint_sha256"] == CP_SHA and inherited["decisions"] == 221184,
        "this adapter accepts only the frozen migrated CP221184")
    metadata = m.read_json(Path(inherited["manifest"]))
    m.require(all(metadata.get(k) == v for k, v in COUNTERS.items()), "checkpoint counters differ")
    runtime, policy = metadata["runtime_contract"], metadata["policy_contract"]
    m.require(runtime.get("source_git_commit") == HEAD
        and runtime == manifest.get("runtime_contract")
        and policy.get("version") == m.POLICY_VERSION
        and policy.get("observation_layout") == m.OBSERVATION_LAYOUT
        and policy.get("observation_dimension") == 410 and policy.get("raw_action_dimension") == 12,
        "sealed source is not the declared same410/v4 runtime")
    m.require(m.sha256(PLAN) == PLAN_SHA and m.sha256(PUBLICATION) == PUBLICATION_SHA,
        "reviewed immutable migration/publication changed")
    plan, publication = m.read_json(PLAN), m.read_json(PUBLICATION)
    receipt = metadata.get(MIGRATION_KEY)
    m.require(isinstance(receipt, dict) and receipt.get("schema") == SCHEMA
        and Path(receipt.get("plan_path", "")).resolve() == PLAN.resolve()
        and receipt.get("plan_sha256") == PLAN_SHA
        and {k:v for k,v in receipt.items() if k not in ("plan_path", "plan_sha256")} == plan
        and metadata.get("resume_migration") == receipt,
        "saved v4 receipt differs from its reviewed publication")
    factor = plan[FACTOR_KEY]
    m.require(factor.get("schema") == SCHEMA and factor.get("counter_origin") == COUNTERS
        and metadata.get("rr_capture_transfer_branch_counts") == LEARNED,
        "prior real 640/5/100 or zero-credit migration origin differs")
    branch = metadata.get("rr_capture_transfer_branch") or {}
    m.require(all(COUNTERS[k] - branch.get("counter_origin", {}).get(k, -1) == LEARNED[k]
        for k in COUNTERS), "RR branch was reset/recreated")
    flags = {"creates_new_branch":False, "same_mdp_claimed":False,
        "policy_kernel_changed":False, "sigma_changed":False, "caps_changed":False,
        "reward_changed":False, "physical_dynamics_changed":False,
        "physical_task_acceptance_rules_changed":False, "discard_old_rollout_storage":True,
        "same_numeric_input_policy_mapping_preserved":True,
        "same_physical_state_action_equivalence_claimed":False, "physical_state_inherited":False}
    m.require(all(factor.get(k) is v for k,v in flags.items()) and all(factor.get("added_"+k) == 0
        for k in ("policy_decisions", "ppo_updates", "optimizer_steps", "auxiliary_updates")),
        "v4 semantic/learning-credit claims differ")
    effective = factor.get("effective_execution_semantics") or {}
    m.require(effective.get("wheel_mode") == MODE
        and effective.get("RR_capture_feedback") == REVISION
        and effective.get("handoff") == "current_TOP_cumulative_HOLD_next_decision_v1"
        and effective.get("raw_Gaussian_is_not_projected_target") is True,
        "v4 wheel/contact handoff is not explicitly declared")
    reviewed = factor.get("reviewed_code_sha256") or {}
    m.require(reviewed and plan.get("allowed_changed_files") == sorted(reviewed)
        and all(runtime["files"].get(k) == v for k,v in reviewed.items())
        and plan.get("target_runtime_content_sha256") == runtime["runtime_content_sha256"]
        and plan.get("target_git_commit") == HEAD, "frozen runtime/plan hash binding differs")
    for key, digest in factor["preserved_metadata_sha256"].items():
        m.require(key in metadata and m.json_digest(metadata[key]) == digest,
            f"preserved origins/AUX/ancestor receipt changed: {key}")
    source = Path(plan["source_checkpoint"])
    m.require(m.sha256(source) == plan["source_checkpoint_sha256"]
        and m.sha256(m._sidecar_from_checkpoint(source)) == plan["source_manifest_sha256"],
        "migration's actual learned-source binding changed")
    m.require(publication.get("checkpoint_sha256") == CP_SHA
        and publication.get("save_load_round_trip") is True
        and publication.get("exact_same410_learned_state_preserved") is True
        and publication.get("rr_capture_transfer_branch_counts") == LEARNED,
        "actual same410 save/reload publication differs")
    return {**inherited, "rr_capture_control_revision":"v4_support_forward_contact_handoff",
        "rr_capture_transfer_branch_counts":LEARNED, "actual_prior_training":LEARNED,
        "evaluation_added_policy_decisions":0, "evaluation_added_PPO_updates":0,
        "migration_added_optimization":0, "v4_plan":str(PLAN), "v4_plan_sha256":PLAN_SHA,
        "v4_publication":str(PUBLICATION), "v4_publication_sha256":PUBLICATION_SHA,
        "effective_execution_semantics":effective, "inherited_AUX_separate_from_PPO":True}


def validate_snapshot(state):
    m = media()
    # The pinned old validator checks metadata/types only: it has NO 20-degree,
    # 12-second-total or immutable-knee restriction. The v3 budgets follow here.
    _SNAPSHOT_VALIDATOR(state)
    m.require(state.get("feedback_revision") == REVISION and state.get("capture_search_semantics") == SEARCH,
        "tick is not the reviewed v3 hip-then-knee assist")
    travel, elapsed = state["travel_used_deg"], state["descent_elapsed_s"]
    m.require(0. <= travel <= 40.+1e-8 and 0. <= elapsed <= 32.+1e-8
        and -1e-8 <= elapsed-max(travel-20.,0.) <= 12.+1e-8,
        "v3 combined search budget/state is malformed")


def frame_summary(row, native_row):
    m = media()
    out = _FRAME_SUMMARY(row, native_row)
    audit = native_row["native_audit"]
    wheel = audit.get("rr_carry_wheel_evidence")
    m.require(isinstance(wheel,dict) and wheel.get("schema") == "wlr50_clean.rr_carry_wheel_evidence.v1"
        and wheel.get("mode") == MODE
        and audit.get("rr_carry_wheel_context_and_previous_FINAL_independently_verified") is True,
        "tick lacks independently reconstructed wheel-v4 evidence")
    ctx = wheel.get("context") or {}
    m.require(ctx.get("dispatch_physics_tick") == out["dispatch_physics_tick"]
        and ctx.get("source_control_tick") == out["tick"]-1
        and wheel.get("source_evidence") == ctx.get("source_evidence")
        and wheel.get("previous_final_wheel_rad_s") == audit.get("previous_final_drive_wheel_rad_s")
        and wheel.get("output_full12") == row["dispatch"]["drive_target_full12"]
        and wheel.get("raw_policy_and_log_probability_unchanged") is True,
        "wheel source/prestate/final target evidence is not same-tick")
    mask = audit.get("phase_mask_full12")
    raw = m._vector(audit.get("raw_policy_action_full12"), "raw policy")
    m.require(mask == [1]*12, "this declared all12 source has a non-all12 residual permission mask")
    rl = (row.get("current_legs") or {}).get("RL") or {}
    out.update(rr_assist_capture_search_semantics=state_search(row),
        rr_hip_travel_deg=min(out["rr_assist_travel_used_deg"],20.),
        rr_knee_travel_deg=max(out["rr_assist_travel_used_deg"]-20.,0.),
        rl_current_contact=m._contact(rl)[0], rl_placed_history=(row.get("placed_history") or {}).get("RL"),
        # PhysicalEvaluationRecorder writes nominal_full12 at JSON-row TOP LEVEL,
        # as a sibling of native_audit, not inside that audit object.
        source_nominal_wheel_rad_s=m._vector(native_row.get("nominal_full12"),"source nominal")[8:],
        source_nominal_wheel_source="native_tick_audit.jsonl row['nominal_full12'][8:12]",
        policy_raw_wheel=raw[8:], residual_permission_wheel=mask[8:],
        requested_policy_residual_wheel=row["dispatch"]["independent_policy_residual_requested_full12"][8:],
        effective_policy_residual_wheel=m._vector((audit.get("policy_headroom_evidence") or {}).get(
            "effective_policy_residual_full12"),"headroom effective policy residual")[8:],
        effective_policy_residual_source="native_audit.policy_headroom_evidence.effective_policy_residual_full12; before wheel projection",
        wheel_envelope_active=wheel["envelope_active"], wheel_projection_changed=wheel["actual_projection_changed"],
        wheel_action=ctx.get("action"), wheel_gain=ctx.get("gain"),
        wheel_controller_delta=wheel["controller_delta_full12"][8:],
        wheel_pre_projection_candidate=wheel["candidate_full12"][8:],
        wheel_desired_before_slew=wheel["desired_before_slew_full12"][8:],
        previous_final_wheel=wheel["previous_final_wheel_rad_s"],
        wheel_native_counterfactual_no_current_policy=audit["counterfactual_native_targets"]["wheel_velocity_rad_s"],
        wheel_native_policy_effect=audit["native_target_delta"]["wheel_velocity_rad_s"],
        wheel_source_evidence=wheel["source_evidence"], wheel_context=ctx,
        wheel_actuator_joint_ids=None, actuator_joint_ids_missing_reason="not persisted by these two writer schemas",
        wheel_last_writer=audit.get("actual_target_source"),
        mask_scope="PPO residual permission, not multiplication of nominal",
        wheel_receipt_from="native_tick_audit.native_audit.rr_carry_wheel_evidence",
        next_state_X409=out["fl_wheel_guidance_active"])
    # X409 is recomputed on the resulting state for the NEXT dispatch. It need
    # not equal the pre-dispatch receipt's envelope bit on transition ticks.
    return out


def state_search(row):
    return row["rr_capture_assist"]["capture_search_semantics"]


def capture_rows(candidate, ledger):
    m = media()
    wanted, frames, milestones = {v.sim_step for v in ledger}, {}, {}
    count, offset, last = 0, None, None
    with candidate["tick_path"].open("rb") as ticks, candidate["native_tick_path"].open("rb") as natives:
        for count, raw in enumerate(ticks,1):
            native_raw = natives.readline()
            m.require(raw.endswith(b"\n") and native_raw.endswith(b"\n"), "unsealed/partial tick ledger")
            row, native = json.loads(raw), json.loads(native_raw)
            m.require(row.get("episode_physics_tick") == count
                and math.isclose(float(row["sim_time_s"]),count/120.,abs_tol=1e-10),"tick sequence gap")
            item = frame_summary(row,native)
            offset = item["dispatch_to_episode_tick_offset"] if offset is None else offset
            m.require(item["dispatch_to_episode_tick_offset"] == offset,"dispatch offset changed")
            events = {"FIRST_RR_OWNER":bool(item["rr_assist_owner_indices"]),
                "FIRST_RR_CROSS":item["rr_front_edge_crossed"] is True,
                "FIRST_WHEEL_ENVELOPE":item["wheel_envelope_active"],
                "FIRST_WHEEL_PROJECTION":item["wheel_projection_changed"],
                "FIRST_KNEE_CONTINUATION":item["rr_knee_travel_deg"]>0.,
                "FIRST_RR_TOP":item["rr_post_contact"]=="TOP",
                "FIRST_RR_PLACED":item["rr_placed_history"] is True,
                "FIRST_RL_PREPARATION":item["phase"]=="P11",
                "FIRST_RL_CAPTURE_PHASE":item["phase"]=="P12",
                "FIRST_RL_PLACED":item["rl_placed_history"] is True,
                "FIRST_P13":item["phase"]=="P13"}
            for key, yes in events.items():
                if yes and key not in milestones: milestones[key]={"milestone":key,**item}
            if count in wanted: frames[count]=item
            last=item
        m.require(natives.readline()==b"", "native ledger extends past episode")
    m.require(count==candidate["endpoint"] and set(frames)==wanted and last is not None,
        "exact full episode/frame coverage unavailable")
    milestones["TERMINAL"]={"milestone":"TERMINAL",**last}
    rows=[frames[v.sim_step] for v in ledger]
    rr_reached=m.rr_window_reached(rows)
    rl_reached=any(v["phase"] in ("P11","P12","P13") for v in rows)
    for item in rows:
        item.update(run_RR_window_reached=rr_reached,run_RL_window_reached=rl_reached)
    return rows,list(milestones.values())


def panel_lines(row,result,*,rr_reached,feedback_v2):
    m=media(); f=m._fmt
    return [
        f"CP221184 DET | PPO + FL/RR ASSISTS + SUPPORT-WHEEL v4 + HISTORICAL LIMITED AUX | {result}",
        f"prior real training +640 decisions / 5 PPO / 100 Adam | migration 0 | RR/RL reached={rr_reached}/{row['run_RL_window_reached']}",
        f"RR hip-then-knee {row['rr_assist_mode']} ({row['rr_assist_reason']}) | travel hip/knee {f(row['rr_hip_travel_deg'])}/{f(row['rr_knee_travel_deg'])}deg | FL {row['fl_assist_mode']}",
        f"RR hip target/actual {f(row['rr_hip_final_target_deg'])}/{f(row['rr_hip_actual_deg'])}deg | knee dynamic target/actual {f(row['rr_knee_final_target_deg'])}/{f(row['rr_knee_actual_deg'])}deg",
        f"RR gap {f(row['rr_post_gap_mm'])}mm depth {f(row['rr_front_distance_mm'])}mm {row['rr_post_contact']} | bearing={row['rr_current_bearing']} placed-history={row['rr_placed_history']} RL-ready={row['rl_transfer_ready']}",
        f"wheel PRE envelope={row['wheel_envelope_active']} changed={row['wheel_projection_changed']} {row['wheel_action']} gain={f(row['wheel_gain'])} | next-state X409={row['next_state_X409']}",
        m._wheel_line("source nominal wheel rad/s: ",row["source_nominal_wheel_rad_s"]),
        m._wheel_line("FINAL target canonical rad/s: ",row["wheel_final_canonical_rad_s"]),
        m._wheel_line("MEASURED canonical qdot rad/s: ",row["wheel_actual_canonical_rad_s"]),
        m._wheel_line("wheel controller delta rad/s: ",row["wheel_controller_delta"]),
        f"1x 15fps | t={row['time_s']:.3f}s tick={row['tick']} {row['phase']} | current contact != historic placed; controller != PPO learning",
    ]


def detail_plan(rows):
    m=media()
    reached=m.rr_window_reached(rows)
    if reached:
        start=next(i for i,v in enumerate(rows) if v["phase"] in ("P09","P10","P11","P12","P13")
            or v["rr_lift_carry"] or v["rr_assist_mode"]!="WAIT")
        rl=any(v["phase"] in ("P11","P12","P13") for v in rows)
        name="RR_capture_RL_window" if rl else "RR_capture_RL_not_reached"
        title="RR capture through RL window and terminal" if rl else "RR capture through terminal - RL NOT REACHED"
    else:
        start=max(0,len(rows)-60*m.FPS)
        name,title="predecessor_failure_tail","predecessor terminal tail - RR AND RL NOT REACHED"
    return dict(kind=name,start=start,end=len(rows),filename=f"CP221184_DET_v4_{name}_detail.mp4",
        title=f"DETAIL | CP221184 v4 | {title}",
        requested_RR_detail_unavailable_reason=None if reached else "THIS_EPISODE_DID_NOT_REACH_RR_WINDOW")


def export(destination):
    m=media()
    destination=Path(destination).resolve()
    m.require(destination.is_relative_to(HERE.resolve()) and not destination.exists(),"new isolated destination required")
    candidate=m.sealed_source(SOURCE,candidate=True)  # Rejects RUNNING/unsealed/artifact failures.
    baseline=m.sealed_source(m.DEFAULT_HISTORICAL_N,candidate=False)
    helper=m.shared()
    ffmpeg=helper.find_ffmpeg(candidate["capture"].get("full_decode",{}).get("ffmpeg_path"))
    _,ledger,source_validation=helper.checked_media(candidate,ffmpeg=ffmpeg)
    rows,selected=capture_rows(candidate,ledger)
    m.require(0<len(rows)<=3000,"full source must remain <=200 s at native 15fps")
    result,success,termination=m.outcome(candidate["manifest"])
    rr,rl=rows[-1]["run_RR_window_reached"],rows[-1]["run_RL_window_reached"]
    destination.mkdir(parents=True)
    full=m.encode_full(candidate,rows,destination/"CP221184_DET_v4_full_attempt.mp4",
        result=result,rr_reached=rr,feedback_v2=True,ffmpeg=ffmpeg)
    detail=m.encode_detail(Path(full["output"]),rows,destination,ffmpeg=ffmpeg)
    pair=m.encode_pair(baseline,{**candidate,"frame_count":len(rows)},Path(full["output"]),
        destination/"historical_N_vs_CP221184_DET_v4.mp4",historical_version=m.DEFAULT_HISTORICAL_VERSION,
        feedback_v2=True,ffmpeg=ffmpeg)
    evidence=destination/"CP221184_v4_selected_fourwheel_evidence.json"
    m.write_new_json(evidence,dict(schema="wlr50_clean.rr_capture_v4_video_evidence.v1",source=str(SOURCE),
        capture_assist_ticks_sha256=m.sha256(candidate["tick_path"]),
        native_tick_audit_sha256=m.sha256(candidate["native_tick_path"]),
        selection="first observed events plus real terminal; absent milestones are not invented",
        pre_and_post_ticks_separate=True,rows=selected))
    receipt=dict(schema="wlr50_clean.rr_capture_v4_video_export.v1",source=str(SOURCE),
        source_manifest_sha256=m.sha256(candidate["manifest_path"]),
        source_run_manifest_sha256=m.sha256(candidate["run_manifest_path"]),
        checkpoint=candidate["checkpoint"],control_method=METHOD,
        source_legacy_control_method=candidate["manifest"].get("control_method"),
        source_legacy_first_revision_wheel_guidance=candidate["manifest"]["rr_capture_assist"]["first_revision_wheel_guidance"],
        legacy_first_revision_field_is_effective_v4_mode=False,effective_wheel_mode=MODE,
        RR_assist_feedback_revision=REVISION,RR_assist_capture_search_semantics=SEARCH,
        actual_prior_training=LEARNED,evaluation_added_PPO_updates=0,migration_added_optimization=0,
        inherited_limited_AUX_is_PPO=False,controller_intervention_is_PPO_learning=False,
        same_MDP_claimed=False,physical_result=result,physical_task_success=success,
        termination_reason=termination,source_acceptance_error=candidate["manifest"].get("source_acceptance_error"),
        RR_window_reached=rr,RL_window_reached=rl,
        attempt_classification=("RL_WINDOW_REACHED" if rl else "RR_CAPTURE_WINDOW_ONLY" if rr else "PREDECESSOR_ONLY"),
        full_episode_continuous=True,full_failure_tail_preserved=True,normal_speed=True,
        single_episode=True,stitched=False,extra_intro_frames=0,source_validation=source_validation,
        full=full,detail=detail,historical_N_comparison=pair,
        historical_N_is_fresh_same_controller_B=False,
        selected_fourwheel_evidence=str(evidence),selected_fourwheel_evidence_sha256=m.sha256(evidence),
        code_binding=dict(adapter_sha256=m.sha256(Path(__file__)),legacy_exporter_sha256=LEGACY_SHA),
        missing_native_joint_ids_not_invented=True,targets_are_not_actual_velocity=True,
        independent_zero_current_policy_audit_is_not_new_physics=True)
    m.write_new_json(destination/"export_receipt.json",receipt)
    print(json.dumps(dict(physical_result=result,full=full["output"],detail=detail["output"],
        historical_N_pair=pair["output"],receipt=str(destination/"export_receipt.json")),indent=2))
    return receipt


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination",type=Path,required=True)
    args=parser.parse_args()
    export(args.destination)
