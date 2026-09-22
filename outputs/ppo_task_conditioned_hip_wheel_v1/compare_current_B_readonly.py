"""Join saved event-window outputs and source identities; no simulator/raw rescan."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = {"B":"currentB_event_windows.json", "deterministic":"CP187904_deterministic_event_windows.json",
         "stochastic":"CP187904_stochastic_event_windows.json"}


def main():
    records = {key:json.loads((HERE/name).read_text()) for key,name in FILES.items()}
    identities = {}
    for key, record in records.items():
        m = json.loads(Path(record["manifest"]).read_text())
        load = m.get("checkpoint_load_provenance")
        identities[key] = dict(source=record["source"], manifest=record["manifest"], role=m["role"],
            experiment_id=m.get("experiment_id"), physical_seed=m["seed"], policy_seed=m.get("policy_seed"),
            policy_sampling_mode=m["policy_sampling_mode"], optimizer_updates=m["optimizer_updates"],
            git_commit=m["runtime_contract"]["source_git_commit"],
            runtime_content_sha256=m["runtime_contract"]["runtime_content_sha256"],
            configuration_sha256={k:v["sha256"] for k,v in m["evaluation_configuration"].items()},
            checkpoint_loaded_and_verified=None if load is None else load["checkpoint_loaded_and_verified"],
            checkpoint_parameter_hashes=None if load is None else load["parameter_hashes"])
    windows = {}
    for name in records["B"]["windows"]:
        comparisons = {}
        for key, record in records.items():
            w=record["windows"][name]; p=w["physical"]; s=w.get("current_support_samples")
            metrics = None if p is None else {k:p[k] for k in ("duration_s","rate_RMS_rad_s","peak_tilt_rad",
                "body_collider_min_world_z_m","conservative_body_obstacle_AABB_separation_m",
                "body_forward_displacement_m","wheel_center_forward_displacement_m","wheel_bottom_gap_above_top_m",
                "RR_mount_world_z_m_sampled","actual_contact_ticks")}
            comparisons[key] = dict(status=w["status"], start_tick=w["start_tick"], goal_tick=w["goal_tick"],
                observed_end_tick=w["observed_end_tick"], metrics=metrics,
                current_FL_support=None if s is None else dict(samples=len(s["sample_ticks"]),**s["FL"]),
                current_RR_support=None if s is None else dict(samples=len(s["sample_ticks"]),**s["RR"]))
        for key in ("deterministic","stochastic"):
            c,b=comparisons[key],comparisons["B"]
            complete=c["status"]==b["status"]=="COMPLETE" and c["metrics"] is not None and b["metrics"] is not None
            c["completed_window_difference_vs_B"] = None if not complete else dict(
                duration_s=c["metrics"]["duration_s"]-b["metrics"]["duration_s"],
                rate_RMS_fraction=c["metrics"]["rate_RMS_rad_s"]/b["metrics"]["rate_RMS_rad_s"]-1,
                peak_tilt_fraction=c["metrics"]["peak_tilt_rad"]/b["metrics"]["peak_tilt_rad"]-1)
            c["comparison_caveat"] = ("Same completed physical task window, single run; not a whole-task improvement or population result."
                if complete else "Not rankable against completed B: target incomplete, start not reached or measurement missing.")
        windows[name]=comparisons
    old=json.loads((HERE/"event_windows_selfcheck_success_B.json").read_text())
    b=records["B"]
    out=dict(schema="current_B_CP187904_event_comparison.v1",identities=identities,
        identity_checks={key:len(set(json.dumps(v[key],sort_keys=True) for v in identities.values()))==1
            for key in ("experiment_id","physical_seed","git_commit","runtime_content_sha256","configuration_sha256")},
        two_C_checkpoint_parameter_hashes_identical=identities["deterministic"]["checkpoint_parameter_hashes"]==identities["stochastic"]["checkpoint_parameter_hashes"],
        outcomes={key:record["outcome"] for key,record in records.items()},
        deterministic_semantic_termination=json.loads((HERE/"CP187904_deterministic_terminal_appendix.json").read_text())["semantic_task"],
        stochastic_RR_retention=json.loads((HERE/"CP187904_stochastic_RR_retention_appendix.json").read_text()),
        history_event_ticks={key:record["history_event_ticks"] for key,record in records.items()},
        windows=windows,
        current_B_vs_saved_old_B=dict(old_evidence="event_windows_selfcheck_success_B.json; no old raw log rescan",
            current_major_events=b["history_event_ticks"],old_major_events=old["history_event_ticks"],
            all_saved_major_events_equal=b["history_event_ticks"]==old["history_event_ticks"],
            final_tick_equal=b["outcome"]["final_tick"]==old["outcome"]["final_tick"],
            final_time_equal=b["outcome"]["final_time_s"]==old["outcome"]["final_time_s"],
            all_saved_stage_entries_equal=b["stage_entry_ticks"]==old["stage_entry_ticks"],
            entire_trajectory_identical=None,entire_trajectory_comparison_performed=False),
        semantics=dict(RMS=records["B"]["metric_semantics"]["rate_RMS"],
            units="seconds, radians, radians/second, metres; raw source signs retained",
            incomplete_window_ranking=False,missing_values_zero_filled=False,
            success_priority="B alone completed the whole task in these three runs; lower partial-window RMS does not qualify PPO as better overall.",
            trajectory_claim="Equal event times do not prove equal complete trajectories; no raw full-trajectory comparison performed.",
            limitations="One run per mode; matched environment/configuration but not statistical success rates. AABB separation is a lower bound, not exact mesh clearance; collider z is not base z; CoM is not support; wheel motion does not prove traction."))
    with (HERE/"CP187904_current_B_comparison.json").open("x",encoding="utf-8") as stream:
        json.dump(out,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write("\n")
    print(json.dumps({"identity_checks":out["identity_checks"],"old_B_major_events_equal":out["current_B_vs_saved_old_B"]["all_saved_major_events_equal"],
        "FR_differences":{k:windows["FR_P01_to_capture"][k]["completed_window_difference_vs_B"] for k in ("deterministic","stochastic")}},ensure_ascii=False))


if __name__ == "__main__":
    main()
