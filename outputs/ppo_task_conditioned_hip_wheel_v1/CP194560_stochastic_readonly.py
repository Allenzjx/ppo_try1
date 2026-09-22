"""Naturally sealed current video only; CPU/std-lib physical evidence summary."""
import json
from pathlib import Path
from video_event_windows_readonly import analyze
from rr_probe_readonly import lines

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SOURCE = ROOT / "runs/ppo_task_conditioned_hip_wheel_v1/video_eval/validation/20260921T1136245327456Z_g97ecd305afb5_1e424676283e49bdbee680a08111d59e/source"
manifest_path = SOURCE / "semantic_video_source_manifest.json"
assert manifest_path.is_file(), "Wait for natural seal, never fabricate finality"
d = analyze(SOURCE)
m = json.loads(manifest_path.read_text(encoding="utf-8"))
e = m["physical_episode"]["physical_task_evaluation"]
assert m["optimizer_updates"] == 0
d["outcome"]["physical_valid"] = e["valid"]
for row in lines(SOURCE / "video_policy_decisions.jsonl"):
    last = row
assert last["end_tick"] == m["episode_physics_ticks"]
s = last["step_info"]
small = {}
for name in ("FL_cross_to_capture","FL_capture_hold_until_RR_preparation","P06_rear_approach","RR_preparation_to_place"):
    w = d["windows"][name]; p = w["physical"]; support = w.get("current_support_samples")
    small[name] = dict(status=w["status"], ticks=[w["start_tick"],w["observed_end_tick"]], reason=w.get("reason"))
    if p is None:
        small[name]["physical"] = None
        continue
    small[name].update(duration_s=p["duration_s"],body_forward_displacement_m=p["body_forward_displacement_m"],
        rate_RMS_rad_s=p["rate_RMS_rad_s"],peak_tilt_rad=p["peak_tilt_rad"],
        body_collider_min_z_m=p["body_collider_min_world_z_m"]["minimum"],
        AABB_separation_lower_bound_min_m=p["conservative_body_obstacle_AABB_separation_m"]["minimum"],
        FL_gap_m=p["wheel_bottom_gap_above_top_m"]["FL"],RR_gap_m=p["wheel_bottom_gap_above_top_m"]["RR"],
        raw_contact_counts=p["actual_contact_ticks"],evaluator_endpoint_count=len(support["sample_ticks"]),
        FL_current_support_summary=support["FL"],RR_current_support_summary=support["RR"])
def leg_current(leg):
    return {k:e["current_legs"][leg].get(k) for k in ("air","ground_contact","contact_mode","support","bearing_force_n", "top_surface_contact",
        "current_lift_valid","active_attempt","front_edge_crossed","placed_on_top","clearance_m","front_distance_m")}
summary = dict(schema="CP194560.stochastic_compact_physical_readonly.v1",source=str(SOURCE),training_label="PPO+LIMITEDAUX",
    outcome=d["outcome"],semantic_terminal=dict(reason=s["termination_reason"],source=s["semantic_task"]["termination_source"],
        phase=s["end_phase_id"],time_outs=s["time_outs"],bootstrap_allowed=s["terminal_bootstrap_allowed"]),
    identity={k:m[k] for k in ("seed","policy_seed","policy_sampling_mode","optimizer_updates","interrupted_final_decision_ticks")},
    history_event_ticks=d["history_event_ticks"],stage_entry_ticks=d["stage_entry_ticks"],
    RR_attempt_events=[x for x in e["history"]["lift_attempt_events"] if x["leg"]=="RR"],
    current_legs={leg:leg_current(leg) for leg in ("FL","FR","RL","RR")},
    RR_history_active_boolean=e["history"]["active_lift"].get("RR"),windows=small,
    first_unmet_explanation="RR qualified at4887, regrounded/revoked before crossing at4928, and never crossed/placed. First unmet task is maintained valid RR lift followed by crossing/controlled placement, not 'RR never qualified'.",
    first_exact_FL_air_after_capture_tick=None,
    limitations="Fixed stochastic checkpoint, not a training rollout. Current endpoint support counts are not contact durations. Complete FL stage window does not prove continuous retention. Incomplete RR cannot outrank successful completion. No wheel traction causality, fabricated geometry/zero filling, old raw scan, GPU import, physics or production edits.")
with (OUT / "CP194560_stochastic_readonly.json").open("x",encoding="utf-8") as stream:
    json.dump(summary,stream,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps(summary,ensure_ascii=False,indent=2))
