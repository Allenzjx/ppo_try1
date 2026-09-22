"""Small sealed-video/probe event-window adapter; no simulator or evaluator replay."""
import argparse
import json
import math
from pathlib import Path
from rr_probe_readonly import LEGS, lines, stats, compact_physical, physical_window


def finite_tree(value):
    if isinstance(value, dict):
        return "nonfinite_raw_float" not in value and all(finite_tree(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def support_summary(samples, lo, hi):
    rows = [(tick, ev) for tick, ev in samples if lo <= tick <= hi]
    if not rows:
        return None
    result = dict(sample_ticks=[tick for tick, _ in rows], sampling="completed decision endpoints plus exact manifest terminal; not 120Hz contact durations")
    for leg in LEGS:
        cs = [ev["current_legs"][leg] for _, ev in rows]
        result[leg] = dict(air_samples=sum(c["air"] for c in cs),
            ground_samples=sum(c["ground_contact"] for c in cs),
            top_verified_bearing_samples=sum(c["top_surface_contact"] and c["bearing_verified"] and c["support"] for c in cs),
            support_samples=sum(c["support"] for c in cs),
            current_lift_valid_samples=(sum(c["current_lift_valid"] for c in cs) if all("current_lift_valid" in c for c in cs) else None))
    return result


def measured_window(raw, errors, samples, heights, start, end, endpoint):
    if start is None:
        return dict(status="NOT_REACHED", start_tick=None, goal_tick=end, observed_end_tick=None, physical=None, reason="start event not observed")
    stop = endpoint if end is None else min(end, endpoint)
    result = dict(status="COMPLETE" if end is not None and end <= endpoint else "INCOMPLETE",
                  start_tick=start, goal_tick=end, observed_end_tick=stop)
    if stop < start:
        return dict(**result, physical=None, reason="event order inconsistent")
    selected = [r for r in raw if start <= r["tick"] <= stop]
    bad = {tick: why for tick, why in errors.items() if start <= tick <= stop}
    if bad or [r["tick"] for r in selected] != list(range(start, stop+1)):
        return dict(**result, physical=None, reason="missing/nonfinite physical samples; no interpolation, zero-fill, or partial-window RMS", invalid_ticks=bad)
    if any(b["t"] <= a["t"] for a,b in zip(selected,selected[1:])):
        return dict(**result, physical=None, reason="non-increasing physical timestamps; no synthetic dt")
    # Reuse the exact existing adjacent-tick, wrapped Euler-rate RMS implementation.
    full = physical_window(selected, start, stop)
    p = {key:full[key] for key in ("duration_s", "physics_samples", "rate_RMS_rad_s", "peak_tilt_rad",
         "body_collider_min_world_z_m", "conservative_body_obstacle_AABB_separation_m", "com_world_displacement_m",
         "com_displacement_toward_initial_FL_direction_m", "actual_contact_ticks", "joints")}
    p["body_forward_displacement_m"] = selected[-1]["base"]["position_w_m"][0]-selected[0]["base"]["position_w_m"][0]
    p["wheel_bottom_gap_above_top_m"] = {leg:stats(r["gap"][leg] for r in selected) for leg in LEGS}
    p["wheel_center_forward_displacement_m"] = {leg:selected[-1]["wheels"][leg]["center_w_m"][0]-selected[0]["wheels"][leg]["center_w_m"][0] for leg in LEGS}
    p["wheel_measured_canonical_qd_rad_s"] = {leg:stats(r["wheels"][leg]["velocity_rad_s"] for r in selected) for leg in LEGS}
    p["wheel_command_canonical_rad_s"] = {leg:stats(r["wheels"][leg]["command_rad_s"] for r in selected) for leg in LEGS}
    mounts = [h for h in heights if start <= h["physics_tick"] <= stop
              and h.get("clock_unchanged") is True and h.get("frame_raw_physics_tick") == h["physics_tick"]]
    p["RR_mount_world_z_m_sampled"] = stats(h["rr_hip_mount_w_m"]["value"][2] for h in mounts
        if h.get("rr_hip_mount_w_m",{}).get("value") is not None and finite_tree(h["rr_hip_mount_w_m"]["value"]))
    p["other_hip_mount_world_z_m"] = None
    p["hip_absence_reason"] = "Only independently recorded RR USD mount is read; other mounts are not inferred from base/CoM."
    result.update(physical=p, current_support_samples=support_summary(samples,start,stop),
                  reason=None, stable_success_claim=False)
    return result


def analyze(run):
    manifest_path = run / "semantic_video_source_manifest.json"
    is_video = manifest_path.is_file()
    if not is_video:
        manifest_path = run / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not is_video and manifest.get("lifecycle") not in ("DIAGNOSTIC_SEALED", "DIAGNOSTIC_ERROR"):
        raise ValueError("Probe must naturally seal before analysis")
    summary = manifest.get("physical_episode" if is_video else "physical_summary")
    if not isinstance(summary,dict) or not isinstance(summary.get("physical_task_evaluation"),dict):
        raise ValueError("No authoritative physical summary; do not substitute the last complete decision for terminal outcome")
    final = summary["physical_task_evaluation"]
    endpoint = manifest["episode_physics_ticks" if is_video else "endpoint_tick"]
    if endpoint is None or final["physics_tick"] != endpoint:
        raise ValueError("Manifest endpoint and independent evaluator clocks disagree")
    events = final["history"]["event_ticks"]
    stage_entry = {"P01":0}
    for row in lines(run / "stage_transition_evidence.jsonl"):
        if row["from_stage"] != row["to_stage"]:
            stage_entry.setdefault(row["to_stage"], row["physics_tick"])
    decisions = []
    for row in lines(run / ("video_policy_decisions.jsonl" if is_video else "probe_decisions.jsonl")):
        if "step_info" in row:
            decisions.append((row["end_tick"],row["step_info"]["semantic_task"]["physical_evaluator"]))
    if not decisions or decisions[-1][0] != endpoint:
        decisions.append((endpoint,final))  # The partial terminal decision is real, not discarded.
    raw, errors = [], {}
    for row in lines(run / "physical_observations.jsonl"):
        tick = row["physics_tick"]
        if tick > endpoint:
            break
        try:
            compact = compact_physical(row)
            if not finite_tree(compact):
                raise ValueError("nonfinite measured field")
            if row.get("all_finite") is not True:
                raise ValueError("raw observation all_finite is not true")
            raw.append(compact)
        except (KeyError,TypeError,ValueError,ZeroDivisionError) as exc:
            errors[tick] = f"{type(exc).__name__}: {exc}"
    height_path = run / "height_diagnostics.jsonl"
    heights = list(lines(height_path)) if height_path.is_file() else []
    lift, cross, placed = (events.get(k,{}) for k in ("active_lift","front_edge_crossed","placed"))
    ranges = dict(FR_P01_to_capture=(0,placed.get("FR")),
        FR_qualified_to_cross=(lift.get("FR"),cross.get("FR")),
        FL_cross_to_capture=(cross.get("FL"),placed.get("FL")),
        FL_capture_hold_until_RR_preparation=(placed.get("FL"),stage_entry.get("P07")),
        P06_rear_approach=(stage_entry.get("P06"),stage_entry.get("P07")),
        RR_preparation_to_place=(stage_entry.get("P07"),placed.get("RR")),
        RR_qualified_to_place=(lift.get("RR"),placed.get("RR")))
    first = next((f"{leg}:{event}" for leg in ("FR","FL","RR","RL")
        for event in ("active_lift","front_edge_crossed","placed") if not final["history"][event].get(leg)),None)
    if first is None and not final["success"]:
        first = next((key for key in ("final_region_valid","final_controlled","post_completion_observation_complete") if not final.get(key)),"terminal task result unresolved")
    return dict(schema="sealed_physical_event_windows.v1", source=str(run), manifest=str(manifest_path),
        role=manifest.get("role", "diagnostic_probe"), sampling_mode=manifest.get("policy_sampling_mode",manifest.get("baseline")),
        checkpoint_load_provenance=manifest.get("checkpoint_load_provenance"),
        outcome=dict(physical_task_success=final["success"], termination_reason=final.get("termination_reason"),
            reason=final.get("reason"), final_tick=endpoint, final_time_s=final["simulation_time_s"],
            first_unfinished_derived_from_physical_history=first,
            source_acceptance_error=manifest.get("source_acceptance_error"), external_budget_stop=manifest.get("external_budget_stop"),
            classification="SUCCESS" if final["success"] else ("INCOMPLETE" if final.get("termination_reason") is None else "TERMINATED_UNSUCCESSFUL")),
        history_event_ticks=events, stage_entry_ticks=stage_entry,
        metric_semantics=dict(rate_RMS="sqrt(integral((wrapped adjacent roll rate squared + pitch rate squared)/2 *dt)/T); rad/s, NOT RMS/T",
            comparison="Compare like physical event windows; report duration/progress separately. Incomplete windows must not outrank completed tasks on smaller RMS.",
            exact_mesh_clearance=None, CoM_direction_is_not_receiver_support=True, wheel_rotation_is_not_traction=True,
            measurement_failure="Any missing/nonfinite physical sample makes that window's physical aggregate null; failure outcome remains authoritative."),
        windows={name:measured_window(raw,errors,decisions,heights,lo,hi,endpoint) for name,(lo,hi) in ranges.items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    result = analyze(args.run)
    with args.output.open("x",encoding="utf-8") as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write("\n")
    print(json.dumps(dict(outcome=result["outcome"], windows={k:dict(status=v["status"],start=v["start_tick"],end=v["observed_end_tick"],
        RMS=None if v["physical"] is None else v["physical"]["rate_RMS_rad_s"]) for k,v in result["windows"].items()}),ensure_ascii=False))
