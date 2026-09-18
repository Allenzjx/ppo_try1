"""Sealed P01 CP178432: bounded FL execution-chain evidence, no new physics."""
import itertools
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "runs/ppo_residual_rr_fix_v1/video_eval/validation/20260917T0510347247858Z_g6c2121b68654_35383f247031424e9b890de765bc2b13"
SOURCE = RUN / "source"
OUT = ROOT / "outputs/diagnostics_v1/formal_P01_CP178432_failure.json"
START, END = 2500, 5907
NAMES = ("front_left", "front_right", "rear_left", "rear_right")


def load(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def window(path, start, stop):
    with path.open(encoding="utf-8") as stream:
        yield from (json.loads(line) for line in itertools.islice(stream, start, stop))


def stats(values):
    values = list(values)
    return {"min": min(values), "max": max(values), "mean": sum(values)/len(values)}


def main():
    run = load(RUN / "run_manifest.json")
    assert run["completed_at_utc"] and run["lifecycle"] == "DIAGNOSTIC_FAILURE"
    manifest = load(SOURCE / "semantic_video_source_manifest.json")
    assert manifest["episode_physics_ticks"] == END and manifest["from_phase"] == "P01"
    assert manifest["physical_task_success"] is False
    decisions, prior_raw = {}, None
    last_decision = None
    for d in window(SOURCE / "video_policy_decisions.jsonl", 0, None):
        tick = d["end_tick"]
        raw = d["raw_policy_action_full12"]
        if START <= tick <= END:
            info = d["step_info"]
            task = info["semantic_task"]
            ev = task["physical_evaluator"]
            rho = manifest["checkpoint_load_provenance"]["policy_contract"]["rho"]
            clipped_prior = [max(-20., min(20., x)) for x in prior_raw]
            carry = [rho*x for x in clipped_prior]
            decisions[tick] = {
                "tick": tick, "start_tick": d["start_tick"], "physics_ticks": d["physics_ticks"],
                "request_phase": d["request_phase"], "stage_id": task["stage_id"],
                "raw_selected_conditional_mean": raw, "previous_saved_raw": prior_raw,
                "HISTORY_carry_inferred_from_recorded_previous_raw": carry,
                "base_mean_algebraic_from_raw_and_history": [(v-h)/(1.-rho) for v, h in zip(raw, carry)],
                "algebraic_decomposition_caveat": "Uses recorded deterministic conditional-mean contract and prior issued raw, not a new actor forward or direct stored observation read.",
                "FL_current": ev["current_legs"]["FL"],
                "other_legs": {leg: ev["current_legs"][leg] for leg in ("FR", "RL", "RR")},
                "FL_transfer_role": ev.get("transfer_roles", {}).get("FL"),
                "completion_values": task["completion_values"],
                "phase_progress": task["phase_progress"], "task_progress_potential": task["task_progress_potential"],
                "stage_age_s": task["stage_age_s"], "local_timeout": task.get("local_timeout"),
                "nominal_diagnostics": task["nominal_provider_diagnostics"],
                "N_full12": info["nominal_action_full12"],
                "projected_residual_full12": info["projected_residual_full12"],
                "actual_drive_target_full12": info["actual_drive_target_full12"],
                "reward_breakdown": info["reward_breakdown"]}
        prior_raw, last_decision = raw, d
    assert last_decision["end_tick"] == END
    final_info = last_decision["step_info"]
    task = final_info["semantic_task"]
    physical_final = manifest["physical_episode"]["physical_task_evaluation"]
    rows = []
    for p, n in zip(window(SOURCE / "physical_observations.jsonl", START, END+1),
                    window(SOURCE / "native_tick_audit.jsonl", START-1, END), strict=True):
        tick = p["physics_tick"]
        assert n["episode_physics_tick"] == tick
        a = n["native_audit"]
        tr = a["tracking_reference_evidence"]
        mapper = tr["mapper_pre_state"]
        wheel = p["wheels"]["front_left_ankle"]
        contact = p["contacts"]["front_left_wheel"]
        joints = [p["joints"]["front_left_"+j] for j in ("hip", "knee")]
        rows.append({
            "tick": tick, "simulation_time_s": p["simulation_time_s"], "source_phase": n["source_phase_id"],
            "FL_front_distance_m": wheel["center_w_m"][0] - p["obstacle"]["front_x_m"],
            "FL_gap_m": wheel["bottom_w_m"][2] - p["obstacle"]["top_z_m"],
            "FL_center_w_m": wheel["center_w_m"], "FL_bottom_w_m": wheel["bottom_w_m"],
            "FL_contact_class": contact["contact_class"],
            "FL_ground": {k: contact["ground"][k] for k in ("active", "pair_verified", "force_w_n", "normal_force_n")},
            "FL_obstacle": {k: contact["obstacle"][k] for k in ("active", "pair_verified", "force_w_n", "normal_force_n", "contact_point_w_m")},
            "FL_N_hip_knee_deg": n["nominal_full12"][:2],
            "N_wheels_rad_s": n["nominal_full12"][8:],
            "FL_raw": a["raw_policy_action_full12"][:2],
            "raw_wheels": a["raw_policy_action_full12"][8:],
            "FL_mapped_N_deg": a["native_drive_target_full12"][:2],
            "FL_controller_deg": a["controller_drive_bias_full12"][:2],
            "controller_full12": a["controller_drive_bias_full12"],
            "FL_projected_residual_deg": n["projected_residual_full12"][:2],
            "FL_previous_final_servo_deg": a["previous_final_drive_servo_deg"][:2],
            "FL_final_target_deg": [j["command_deg"] for j in joints],
            "FL_actual_deg": [j["position_deg"] for j in joints],
            "FL_actual_velocity_deg_s": [j["velocity_deg_s"] for j in joints],
            "FL_target_minus_actual_deg": [j["command_deg"] - j["position_deg"] for j in joints],
            "FL_mapper_pre_state": {k: v[:2] for k, v in mapper.items()},
            "tracking_servo_names": tr["tracking_servo_names"],
            "FL_tracking_channels": tr["channels"][:2],
            "FL_geometry_evidence": a.get("nominal_geometry_evidence"),
            "mask_full12": a["phase_mask_full12"], "verified_dispatch": a["verified"],
            "physical_write_pair_matches": a["setter_dispatch_targets_equal"] and a["actual_mapping_matches_dispatch"],
            "actual_fourwheel_qd_rad_s": [p["wheels"][name+"_ankle"]["velocity_rad_s"] for name in NAMES],
            "actual_fourwheel_command_rad_s": [p["wheels"][name+"_ankle"]["command_rad_s"] for name in NAMES],
            "base": p["base"], "center_of_mass": p["center_of_mass"],
            "body_collision": p["body_collision"], "body_bounds_w_m": p["body_bounds_w_m"]})
    assert [r["tick"] for r in rows] == list(range(START, END+1))
    tail = [r for r in rows if r["tick"] >= 3000]
    source_changes = []
    previous = None
    for r in rows:
        key = (tuple(r["FL_N_hip_knee_deg"]), tuple(r["N_wheels_rad_s"]))
        if key != previous:
            source_changes.append({"tick": r["tick"], "FL_N_deg": key[0], "N_wheels_rad_s": key[1]})
            previous = key
    contact_rows = [r for r in rows if r["FL_ground"]["active"] or r["FL_obstacle"]["active"]]
    summary = {
        "run_lifecycle": run["lifecycle"], "sealed_at": run["completed_at_utc"],
        "physical_task_success": False, "full_task_first_incomplete": "P05_FL_placement",
        "RR_reached_in_full_evaluation": False,
        "terminal_tick": END, "duration_s": END/120.,
        "semantic_termination_reason": task["termination_reason"],
        "semantic_termination_source": task["termination_source"],
        "semantic_local_timeout": task["local_timeout"],
        "semantic_stage_age_s": task["stage_age_s"],
        "physical_termination_reason": physical_final["termination_reason"],
        "physical_termination_source": physical_final["termination_source"],
        "physical_run_validity": physical_final["run_validity"],
        "physical_evidence_status": physical_final["physical_evidence_status"],
        "original_source_acceptance_error": manifest["source_acceptance_error"],
        "completed_stage_ids": task["completed_stage_ids"],
        "completion_values_not_boolean_success": task["completion_values"],
        "final_FL": physical_final["current_legs"]["FL"],
        "event_ticks": physical_final["history"]["event_ticks"],
        "actual_physics_rows": len(rows), "decision_snapshots": len(decisions),
        "FL_contact_active_physics_ticks": [r["tick"] for r in contact_rows],
        "FL_gap_window_m": stats(r["FL_gap_m"] for r in rows),
        "FL_gap_3000_to_endpoint_m": stats(r["FL_gap_m"] for r in tail),
        "FL_hip_abs_target_error_tail_deg": stats(abs(r["FL_target_minus_actual_deg"][0]) for r in tail),
        "FL_knee_abs_target_error_tail_deg": stats(abs(r["FL_target_minus_actual_deg"][1]) for r in tail),
        "FL_hip_target_tail_deg": stats(r["FL_final_target_deg"][0] for r in tail),
        "FL_hip_actual_tail_deg": stats(r["FL_actual_deg"][0] for r in tail),
        "FL_knee_target_tail_deg": stats(r["FL_final_target_deg"][1] for r in tail),
        "FL_knee_actual_tail_deg": stats(r["FL_actual_deg"][1] for r in tail),
        "FL_raw_tail": [stats(r["FL_raw"][j] for r in tail) for j in range(2)],
        "FL_effective_projected_residual_tail_deg": [stats(r["FL_projected_residual_deg"][j] for r in tail) for j in range(2)],
        "controller_full12_always_zero": all(r["controller_full12"] == [0.] * 12 for r in rows),
        "masks_all12_one": all(r["mask_full12"] == [1.] * 12 for r in rows),
        "dispatch_always_verified": all(r["verified_dispatch"] and r["physical_write_pair_matches"] for r in rows),
        "FL_capture_owner_present_in_any_decision": any("FL" in d["nominal_diagnostics"]["capture_owner_hold"]["captures"] for d in decisions.values()),
        "base_position_change_tail_m": [b-a for a,b in zip(tail[0]["base"]["position_w_m"], tail[-1]["base"]["position_w_m"])],
        "CoM_position_change_tail_m": [b-a for a,b in zip(tail[0]["center_of_mass"]["position_w_m"], tail[-1]["center_of_mass"]["position_w_m"])]}
    report = {
        "schema": "wlr50_clean.formal_P01_CP178432_FL_failure.v1", "source": str(SOURCE),
        "bounds": [START, END], "summary": summary,
        "no_new_simulation_policy_forward_GAE_or_physical_counterfactual": True,
        "checkpoint_load_provenance": manifest["checkpoint_load_provenance"],
        "runtime_contract": manifest["runtime_contract"],
        "terminal_semantic_task": {k:v for k,v in task.items() if k not in ("physical_evaluator", "transfer_roles", "history", "transition_evidence", "transfer_role_context")},
        "source_N_changes": source_changes, "decision_snapshots": list(decisions.values()),
        "physics_native_rows": rows,
        "attribution_caveats": [
            "native mapped N is the actual same-dispatch computation, not an independently recomputed zero rollout",
            "separate counterfactual nominal-history trace is not a physical trajectory; mapper pre-state and previous final targets are retained",
            "raw/HISTORY algebra does not establish a physical derivative or causal benefit of zeroing one residual",
            "an inactive contact sensor may retain contact_point_w_m; active pair and force determine actual contact",
            "final_controlled is only current speed condition, not platform/placement/full-task success"]}
    with OUT.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"output": str(OUT), "summary": summary, "source_N_changes": source_changes},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
