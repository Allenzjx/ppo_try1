"""Compact summary from already extracted windows and the sealed manifest only."""
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent
extracted=OUT/"CP192512_stochastic_event_windows.json"
d=json.loads(extracted.read_text(encoding="utf-8"))
m=json.loads(Path(d["manifest"]).read_text(encoding="utf-8"))
e=m["physical_episode"]["physical_task_evaluation"]
assert d["outcome"]["final_tick"]==5775 and e["termination_reason"]=="TASK_FAILURE_BODY_COLLISION"
small={}
for name in ("FL_cross_to_capture","FL_capture_hold_until_RR_preparation","P06_rear_approach","RR_preparation_to_place"):
    w=d["windows"][name];p=w["physical"];s=w["current_support_samples"]
    small[name]=dict(status=w["status"],ticks=[w["start_tick"],w["observed_end_tick"]],
        duration_s=p["duration_s"],body_forward_displacement_m=p["body_forward_displacement_m"],
        rate_RMS_rad_s=p["rate_RMS_rad_s"],peak_tilt_rad=p["peak_tilt_rad"],
        body_collider_min_z_m=p["body_collider_min_world_z_m"]["minimum"],
        AABB_separation_lower_bound_min_m=p["conservative_body_obstacle_AABB_separation_m"]["minimum"],
        FL_gap_m=p["wheel_bottom_gap_above_top_m"]["FL"],RR_gap_m=p["wheel_bottom_gap_above_top_m"]["RR"],
        raw_contact_counts=p["actual_contact_ticks"],evaluator_endpoint_count=len(s["sample_ticks"]),
        FL_current_support_summary=s["FL"],RR_current_support_summary=s["RR"])
summary=dict(schema="CP192512.stochastic_compact_physical_summary.v1",source=d["source"],
    extracted_source=str(extracted),outcome=d["outcome"],identity={k:m[k] for k in ("seed","policy_seed","policy_sampling_mode","optimizer_updates","interrupted_final_decision_ticks")},
    history_event_ticks=d["history_event_ticks"],stage_entry_ticks=d["stage_entry_ticks"],
    RR_attempt_events=[x for x in e["history"]["lift_attempt_events"] if x["leg"]=="RR"],
    RR_final_current={k:e["current_legs"]["RR"][k] for k in ("current_lift_valid","active_attempt","contact_mode","support","front_edge_crossed","placed_on_top","free_air_reference_tick","consecutive_free_air_samples","unsupported_free_lift_m","clearance_m","front_distance_m")},
    RR_history_active_boolean=e["history"]["active_lift"]["RR"],
    first_unmet_explanation="RR qualified at4861 but revoked after ground contact4936; current lift false, never crossed/placed. Need maintained valid RR lift and controlled crossing/placement, not claim RR never qualified.",
    windows=small,first_exact_FL_air_after_capture_tick=None,
    first_exact_FL_air_absence_reason="Not extracted by existing window helper; no additional raw scan; counts preserve actual recorded contact classification.",
    limitations="Fixed stochastic checkpoint evaluation only; 0 optimizer updates. Completed window endpoint means event reached, not successful contact retention. Endpoint support counts are not continuous support durations. RR window incomplete and cannot outrank a completed run. No extra old-run scan, model forward, physics or production edit.")
with (OUT/"CP192512_stochastic_summary.json").open("x",encoding="utf-8") as stream:
    json.dump(summary,stream,ensure_ascii=False,indent=2,allow_nan=False)
print("compact summary saved; no additional raw trace scan")
