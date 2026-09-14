"""Single bounded audit pass per existing run; stop N+0 at completed P05."""
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "runs/ppo_fsm_reference_p09_stable_v2"
N0_ID = "20260910T1427462398013Z_g7db0d17f398d_6d154f2f0ebc4d2a9cf08176a804299b"
C_ID = "20260910T1411052557957Z_g7db0d17f398d_fda5b92927ef45d7aa2b048e98015a8e"


def summary(info):
    task = info["semantic_task"]
    fl = task["physical_evaluator"]["current_legs"]["FL"]
    mode = "TOP" if fl["top_contact"] else "GROUND" if fl["ground_contact"] else "AIR" if fl["air"] else "OTHER"
    return {"decision":info["decision_count"], "tick":info["physics_tick"], "phase":info["phase_id"],
        "end_phase":info["end_phase_id"], "time_s":info["sim_time_s"],
        "FL_contact":mode, "FL_front_m":fl["front_distance_m"], "FL_clearance_m":fl["clearance_m"],
        "FL_top_geometry":fl["top_geometry"], "FL_bearing_n":fl["bearing_force_n"], "FL_support":fl["support"],
        "FL_Q":task["active_lift_history"]["FL"], "FL_C":task["front_edge_crossed_history"]["FL"],
        "FL_P":task["placed_history"]["FL"],
        "nominal_FL_hip_knee_deg":info["nominal_action_full12"][:2], "nominal_wheels_rad_s":info["nominal_action_full12"][8:],
        "residual_FL_hip_knee_deg":info["projected_residual_full12"][:2], "residual_wheels_rad_s":info["projected_residual_full12"][8:],
        "target_FL_hip_knee_deg":info["actual_drive_target_full12"][:2], "target_wheels_rad_s":info["actual_drive_target_full12"][8:]}


def inspect(path, video=False):
    seen, milestones, counts, post_modes = 0, {}, Counter(), Counter()
    p05_points, precursor = [], None
    native_verified = native_count = 0
    raw_zero_all_read = residual_zero_all_read = True
    masks_open, p05_count = 0, 0
    largest_simple_difference = {"absolute_difference":-1}
    max_composed_error = 0.0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            info = row["step_info"] if video else row
            seen += 1
            task = info["semantic_task"]
            point = summary(info)
            raw_zero_all_read &= not any(info["raw_policy_action_full12"])
            residual_zero_all_read &= not any(info["projected_residual_full12"])
            if info["phase_id"] == "P04":
                precursor = point
            if info["phase_id"] == "P05":
                p05_count += 1
                p05_points.append(point)
                milestones.setdefault("P05_first_action_endpoint",point)
                masks_open += info["actuator_target_effect_audit"]["phase_mask_full12"] == [1]*12
                for tick in info["actuator_target_effect_audit_ticks"]:
                    native_count += 1
                    native_verified += tick["verified"] is True
                counts[point["FL_contact"]] += 1
                if point["FL_Q"]:
                    milestones.setdefault("first_qualified_endpoint",point)
                if point["FL_C"]:
                    post_modes[point["FL_contact"]] += 1
                    milestones.setdefault("first_post_cross_endpoint",point)
                    if point["FL_top_geometry"]:
                        milestones.setdefault("first_post_cross_top_geometry_endpoint",point)
                    if point["FL_contact"] == "TOP":
                        milestones.setdefault("first_post_cross_TOP_endpoint",point)
                if point["FL_P"]:
                    milestones.setdefault("first_placed_endpoint",point)
                native = info["actuator_target_effect_audit"]
                for channel,(base,res,target) in enumerate(zip(native["native_drive_target_full12"], info["projected_residual_full12"],info["actual_drive_target_full12"])):
                    error = abs(base+res-target)
                    if error > largest_simple_difference["absolute_difference"]:
                        largest_simple_difference = {"absolute_difference":error, "channel_index0":channel,
                            "channel_name":native["canonical_order"][channel], "tick":info["physics_tick"],
                            "native_base":base, "residual":res, "final_target":target,
                            "controller_bias":native["controller_drive_bias_full12"][channel],
                            "combined_bias":native["combined_post_mapper_bias_full12"][channel]}
                    max_composed_error=max(max_composed_error,abs(base+native["combined_post_mapper_bias_full12"][channel]-target))
                last_info=info
                if task["placed_history"]["FL"] or info["end_phase_id"] != "P05":
                    break
            elif p05_count:
                break
    history=last_info["semantic_task"]["physical_evaluator"]["history"]
    milestones["P04_last_action_endpoint"] = precursor
    milestones["P05_last_read_endpoint"] = summary(last_info)
    return {"audit_path":str(path), "rows_read":seen, "P05_decisions":p05_count,
        "read_ended_at_tick":last_info["physics_tick"], "all_raw_zero_in_read_prefix":raw_zero_all_read,
        "all_residual_zero_in_read_prefix":residual_zero_all_read,
        "FL_events":[e for e in history["lift_attempt_events"] if e["leg"]=="FL"],
        "FL_event_ticks":{k:v.get("FL") for k,v in history["event_ticks"].items()},
        "P05_contact_endpoints":dict(counts), "post_C_contact_endpoints":dict(post_modes),
        "native_ticks_in_P05":native_count, "native_verified_in_P05":native_verified,
        "open_12_masks_in_P05":masks_open, "milestones":milestones,
        "largest_simple_native_base_plus_residual_difference":largest_simple_difference,
        "maximum_native_base_plus_combined_bias_vs_final_error":max_composed_error}


n0=inspect(BASE / "diagnostics" / N0_ID / "residual_and_projection_audit.jsonl")
video5=inspect(BASE / "video_eval/validation" / C_ID / "source/video_policy_decisions.jsonl",True)
assert n0["milestones"]["P05_last_read_endpoint"]["FL_P"] is True
assert video5["milestones"]["P05_last_read_endpoint"]["FL_P"] is False
print(json.dumps({"schema":"wlr50_clean.bounded_progress_matched_comparison.v1", "checked_at_utc":datetime.now(timezone.utc).isoformat(),
    "alignment":"P05 entry / measured Q / front C / top-geometry / true TOP / controlled P; not clock- or pose-identical trajectories",
    "N_plus_zero":n0,"video5_checkpoint149888":video5,
    "scope":"One pass per named existing audit, N+0 stopped at its already-completed P05; no P06+ conclusions, physics scan, reward recomputation, model hash, media or production change."},indent=2))
