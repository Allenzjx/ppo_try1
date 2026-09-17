"""Bounded ended-run P06/P09 comparison; no reference Recording scan or physics."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
RUNS = {
    "C176768": ("runs/ppo_task_first_recovery_v1/video_eval/validation/20260916T0428416958414Z_gb0438f66ec63_292464ef72094ece8a8e494452b8aa23/source", 2352, 5984, 6000, 6272, 6311),
    "retained_zero": ("runs/ppo_fsm_reference_p09_stable_v2/video_eval/prior_B/20260915T0525295619176Z_g4a58c0190ef7_615f8fe3cfc64cff81f73ac6a0d302e6/source", 2680, 5160, 5176, 5434, 5511),
}
ORDER = ["front_left_hip","front_left_knee","front_right_hip","front_right_knee","rear_left_hip","rear_left_knee","rear_right_hip","rear_right_knee","front_left_ankle","front_right_ankle","rear_left_ankle","rear_right_ankle"]


def rows(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def main():
    from wlr50_clean.ppo.semantic_training import write_json
    result, physical, native = {}, {}, {}
    for label, (relative, p06, p07, p09, qualified, stop) in RUNS.items():
        source = ROOT / relative
        low = p07-120
        obs, targets, decisions, wrapper_terminal = {}, {}, [], None
        for row in rows(source / "physical_observations.jsonl"):
            tick = row["physics_tick"]
            if tick > stop: break
            if not (low <= tick <= stop or tick in (p06,p06+1)): continue
            obs[tick] = {
                "tick": tick, "time_s": row["simulation_time_s"],
                "body_min_z_m": row["body_bounds_w_m"]["base_link"]["minimum_m"][2],
                "body_gap_above_obstacle_top_m": row["body_bounds_w_m"]["base_link"]["minimum_m"][2]-row["obstacle"]["top_z_m"],
                "body_collision": row["body_collision"], "body_position_w_m": row["base"]["position_w_m"],
                "body_linear_velocity_w_m_s": row["base"]["linear_velocity_w_m_s"],
                "body_angular_velocity_w_rad_s": row["base"]["angular_velocity_w_rad_s"],
                "com_w_m": row["center_of_mass"]["position_w_m"],
                "support_bodies": row["support"]["active_bodies"],
                "actual_full12": row["actual_full12"], "commanded_full12": row["commanded_full12"],
                "joint_tracking_error_deg": [row["joints"][name]["error_deg"] for name in ORDER[:8]],
                "wheel_bottom_z_m": [row["wheels"][name]["bottom_w_m"][2] for name in ORDER[8:]],
            }
        for row in rows(source / "native_tick_audit.jsonl"):
            tick = row["episode_physics_tick"]
            if tick > stop: break
            if not (low <= tick <= stop or tick == p06+1): continue
            audit = row["native_audit"]
            targets[tick] = {"tick": tick, "source_phase": row["source_phase_id"],
                "source_N_full12": row["nominal_full12"], "mapped_N_full12": audit["native_drive_target_full12"],
                "raw_full12": audit["raw_policy_action_full12"],
                "projected_residual_full12": row["projected_residual_full12"],
                "same_state_zero_residual_counterfactual_native": audit["counterfactual_native_targets"],
                "native_target_delta": audit["native_target_delta"],
                "final_native_targets": audit["actual_native_targets"],
                "physical_actual_and_command_same_tick": obs.get(tick),
            }
        for row in rows(source / "video_policy_decisions.jsonl"):
            if row["start_tick"] > stop: break
            if not (low <= row["end_tick"] <= stop or row["end_tick"] in (p06,p06+8)): continue
            if "step_info" not in row:
                wrapper_terminal = row
                continue
            task = row["step_info"]["semantic_task"]
            evaluator = task["physical_evaluator"]
            provider = task.get("nominal_provider_diagnostics",{})
            decision = {"start_tick": row["start_tick"], "end_tick": row["end_tick"],
                "request_phase": row["request_phase"], "task_stage": task["stage_id"],
                "current_legs": {leg:{k:status.get(k) for k in (
                    "air","support","contact_mode","contact_surface","bearing_force_n","load_fraction",
                    "clearance_m","front_distance_m","initial_now","current_lift_valid","ground_relative_lift_m")}
                    for leg,status in evaluator["current_legs"].items()},
                "source_layers": provider.get("source_partial_order",{}).get("layers"),
                "capture_owner_hold": provider.get("capture_owner_hold"),
                "height_recovery": provider.get("height_recovery"),
                "reward_breakdown": row["step_info"]["reward_breakdown"],
                "rr_events": [event for event in task["history"].get("lift_attempt_events",[]) if event["leg"]=="RR"],
                "history_event_ticks": task["history"]["event_ticks"],
            }
            decisions.append(decision)
        selected = sorted({p06,p07,p07+8,p09,p09+8,p09+64,p09+128,p09+192,p09+256,qualified-8,qualified,qualified+8,stop})
        samples = []
        for tick in selected:
            if tick not in obs: continue
            previous = next((d for d in reversed(decisions) if d["end_tick"] <= tick),None)
            samples.append({"physical":obs[tick], "same_tick_native":targets.get(tick),
                "next_tick_native":targets.get(tick+1), "previous_decision_task":previous})
        boundaries = {}
        for threshold in (.08,.06,.04,.02,.01,0.):
            boundaries[str(threshold)] = next((r for t,r in obs.items() if t>=p07 and r["body_gap_above_obstacle_top_m"]<=threshold),None)
        layer_changes = []
        seen = {}
        for d in decisions:
            for layer in d["source_layers"] or []:
                identity = (layer.get("status"),layer.get("wait_reason"))
                if seen.get(layer["stage"]) != identity:
                    layer_changes.append({"decision_end_tick":d["end_tick"], **layer})
                    seen[layer["stage"]] = identity
        result[label] = {"source":str(source),"p06_entry_tick":p06,"p07_entry_tick":p07,"p09_entry_tick":p09,
            "rr_qualified_lift_tick":qualified,"window_stop_tick":stop,
            "samples":samples,"source_layer_status_changes":layer_changes,
            "first_body_gap_below_descriptive_thresholds_after_p07":boundaries,
            "window_max_abs_projected_servo_residual_deg":max(abs(x) for t,row in targets.items() if t>=p07 for x in row["projected_residual_full12"][:8]),
            "window_max_abs_projected_wheel_residual_rad_s":max(abs(x) for t,row in targets.items() if t>=p07 for x in row["projected_residual_full12"][8:]),
            "terminal_or_window_end":obs[stop],
            "last_decision":decisions[-1],
            "wrapper_terminal_without_environment_return":wrapper_terminal,
        }
        physical[label], native[label] = obs,targets
    c_entry,b_entry=RUNS["C176768"][2],RUNS["retained_zero"][2]
    diffs=[]
    for offset in range(1,min(RUNS["C176768"][5]-c_entry,RUNS["retained_zero"][5]-b_entry)+1):
        c,b=native["C176768"].get(c_entry+offset),native["retained_zero"].get(b_entry+offset)
        if c is None or b is None: continue
        changes={name:c["source_N_full12"][i]-b["source_N_full12"][i] for i,name in enumerate(ORDER)
                 if abs(c["source_N_full12"][i]-b["source_N_full12"][i])>1e-9}
        if changes:
            diffs.append({"offset_from_P07_entry":offset,"C_tick":c_entry+offset,"zero_tick":b_entry+offset,
                          "source_N_differences":changes,"C_N":c["source_N_full12"],"zero_N":b["source_N_full12"]})
            break
    output={"schema":"wlr50_clean.c176768_rear_entry_bounded_comparison.v1", "order":ORDER,
        "units":"first8 canonical deg, last4 canonical-forward rad/s; explicit native arrays retain actuator signs",
        "runs":result,"first_P07_event_aligned_source_N_difference":diffs,
        "comparison_caveat":"different earlier closed-loop trajectories and actual task-entry states; same N code need not produce same captured/feedback-dependent requests; not a one-factor physics intervention",
        "reward_credit_caveat":"formal deterministic evaluation has zero optimizer updates; its terminal collision was not in first128 training rollout; no GAE/advantages reconstructed here",
        "threshold_caveat":"body-gap thresholds are reporting markers only, not changed physical acceptance"}
    out=ROOT/"outputs/ppo_task_first_recovery_v1/c176768_p09_failure_window.json"
    write_json(out,output)
    print(json.dumps({"path":str(out),"first_N_difference":diffs,"runs":{label:{
        "source_status_changes":r["source_layer_status_changes"],
        "first_body_gap_ticks":{key:None if value is None else value["tick"] for key,value in r["first_body_gap_below_descriptive_thresholds_after_p07"].items()},
        "max_servo_residual":r["window_max_abs_projected_servo_residual_deg"],
        "max_wheel_residual":r["window_max_abs_projected_wheel_residual_rad_s"],
        "sample_summary":[{"tick":s["physical"]["tick"],"gap_mm":1000*s["physical"]["body_gap_above_obstacle_top_m"],
            "qRR":s["physical"]["actual_full12"][6:8],"targetRR":s["physical"]["commanded_full12"][6:8],
            "bodyz":s["physical"]["body_position_w_m"][2],"support":s["physical"]["support_bodies"]} for s in r["samples"]]}
        for label,r in result.items()}},indent=2))


if __name__=="__main__":main()
