"""Read-only stopped-run final-window/command-ownership audit; no physics."""
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
SOURCES = {
    "C177024": "runs/ppo_task_first_recovery_v1/video_eval/validation/20260916T0510507849173Z_gad0c1328f177_2bdf12ef355f483cb214a6112220df2b/source",
    "retained_zero": "runs/ppo_fsm_reference_p09_stable_v2/video_eval/prior_B/20260915T0525295619176Z_g4a58c0190ef7_615f8fe3cfc64cff81f73ac6a0d302e6/source",
}


def rows(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def norm(row):
    return math.sqrt(sum(v*v for v in row))


def main():
    from wlr50_clean.ppo.semantic_training import write_json
    result = {}
    for label, relative in SOURCES.items():
        source = ROOT / relative
        decisions, last_wrapper = [], None
        for row in rows(source / "video_policy_decisions.jsonl"):
            if not row.get("step_info"):
                last_wrapper = row
                continue
            info = row["step_info"]
            task = info["semantic_task"]
            ev = task["physical_evaluator"]
            if task["stage_id"] not in ("P12", "P13"):
                continue
            decisions.append({"tick": row["end_tick"], "stage": task["stage_id"],
                "time_s": ev["simulation_time_s"], "elapsed_stage_s": task["stage_elapsed_s"],
                "history": task["history"]["event_ticks"],
                "post": {k:ev.get(k) for k in ("traversal_event_time_s", "final_controlled",
                    "final_region_valid", "final_support_available", "task_completed_controlled",
                    "post_completion_observation_started", "post_completion_elapsed_s",
                    "post_completion_observation_complete", "post_completion_loss_observed",
                    "maximum_commanded_wheel_speed_rad_s", "home_maximum_servo_error_deg",
                    "strict_recovery_quality", "goal_features", "reason", "termination_source")},
                "owner": task.get("nominal_provider_diagnostics", {}).get("final_stop_owner"),
                "nominal_wheels": info["nominal_action_full12"][8:],
                "residual_wheels": info["projected_residual_full12"][8:],
                "final_wheels": info["actual_drive_target_full12"][8:]})
        first_post = next(d for d in decisions if d["post"]["post_completion_observation_started"])
        post_tick = round((first_post["time_s"]-first_post["post"]["post_completion_elapsed_s"])*120)
        phase_entry = next(d["tick"] for d in decisions if d["stage"] == "P13")
        all_placed = max(decisions[-1]["history"]["placed"].values())
        states, last_state = [], None
        for d in decisions:
            if d["tick"] < min(all_placed, post_tick)-16:
                continue
            state = (d["stage"], d["post"]["post_completion_observation_started"],
                d["post"]["final_controlled"], d["post"]["final_region_valid"],
                d["post"]["final_support_available"], bool((d["owner"] or {}).get("active")))
            if state != last_state:
                states.append(d)
                last_state = state
        changes, last_wheels = [], None
        for row in rows(source / "native_tick_audit.jsonl"):
            tick = row["episode_physics_tick"]
            if tick < min(all_placed, phase_entry)-8:
                continue
            wheel = row["nominal_full12"][8:]
            if wheel != last_wheels:
                changes.append({"tick":tick,"phase":row["source_phase_id"],"N_wheels":wheel,
                    "residual_wheels":row["projected_residual_full12"][8:]})
                last_wheels = wheel
        selected = set([all_placed, post_tick-1, post_tick, post_tick+1,
                        phase_entry, phase_entry+1, post_tick+119, post_tick+120])
        stop = next((r for r in changes if r["tick"] >= phase_entry
                     and max(map(abs,r["N_wheels"])) <= .02),None)
        if stop:
            selected.update((stop["tick"]-1,stop["tick"],stop["tick"]+1))
        physical, last = [], None
        first_post_speed_loss = None
        for row in rows(source / "physical_observations.jsonl"):
            tick = row["physics_tick"]
            if tick < min(all_placed,post_tick)-1:
                continue
            last = {"tick":tick,"time_s":row["simulation_time_s"],
                "wheel_actual_rad_s":row["actual_full12"][8:],
                "wheel_final_target_rad_s":row["commanded_full12"][8:],
                "body_linear_speed_m_s":norm(row["base"]["linear_velocity_w_m_s"]),
                "body_angular_speed_rad_s":norm(row["base"]["angular_velocity_w_rad_s"]),
                "supports":row["support"]["active_bodies"], "body_collision":row["body_collision"],
                "home_error_deg_diagnostic":max(map(abs,row["actual_full12"][:8]))}
            last["measured_speed_pass"] = bool(max(map(abs,last["wheel_actual_rad_s"])) <= .25
                and last["body_linear_speed_m_s"] <= .05 and last["body_angular_speed_rad_s"] <= .3)
            if tick > post_tick and first_post_speed_loss is None and not last["measured_speed_pass"]:
                first_post_speed_loss = last
            if tick in selected:
                physical.append(last)
        result[label] = {"source":str(source), "all_placed_tick":all_placed,
            "P13_task_entry_tick":phase_entry, "post_start_tick":post_tick,
            "post_start_stage":"P12" if post_tick < phase_entry else "P13",
            "first_P13_nominal_stop":stop,"N_wheel_owner_changes":changes,
            "decision_flag_changes":states,"last_completed_decision":decisions[-1],
            "first_post_measured_speed_loss":first_post_speed_loss,
            "selected_physical":physical,"final_physical":last,"terminal_wrapper":last_wrapper}
    contract = json.loads((ROOT/"configs/recording_motion_contract.json").read_text())
    p13 = next(p for p in contract["phases"] if p["state_id"]=="P13")
    result["P13_authored_wheel_events"] = [{"source_elapsed_s":w["time_s"],"wheels":w["full12"][8:],
        "commands":w.get("source_commands")} for w in p13["waypoints"] if w.get("source_commands")]
    path=ROOT/"outputs/ppo_task_first_recovery_v1/c177024_p13_stop_window.json"
    write_json(path,result)
    print(json.dumps({"path":str(path),**{label:{k:r[k] for k in ("all_placed_tick","P13_task_entry_tick",
        "post_start_tick","post_start_stage","first_P13_nominal_stop","N_wheel_owner_changes",
        "first_post_measured_speed_loss","final_physical")} for label,r in result.items() if label in SOURCES},
        "P13_authored_wheel_events":result["P13_authored_wheel_events"]},indent=2))


if __name__ == "__main__":
    main()
