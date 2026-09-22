"""Bounded read-only diagnostic comparison. Writes analysis only, never sim state."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from wlr50_clean.ppo.semantic_observation import _quaternion, _rpy
from wlr50_clean.infrastructure.command_batch import servo_limits_deg

NAMES = ("FR_RL_minus3_CP185856_20260921_01", "FL_minus3_CP185856_20260921_01")
RUNS = ROOT / "runs/ppo_task_conditioned_hip_wheel_v1/diagnostics"
JOINTS = ("rear_left_hip", "rear_right_hip", "rear_right_knee", "front_left_hip", "front_left_knee")


def lines(path, end=None):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.endswith("\n"): break  # An unfinished active write is not a sample.
            row = json.loads(line)
            if end is not None and row.get("physics_tick", row.get("end_tick", -1)) > end: break
            yield row


def stats(values):
    values = list(values)
    return None if not values else dict(n=len(values), minimum=min(values), maximum=max(values),
        mean=sum(values)/len(values), first=values[0], last=values[-1])


def compact(row):
    info, req = row["step_info"], row["policy_request_not_necessarily_applied"]
    task = info["semantic_task"]
    native = info["actuator_target_effect_audit"]
    headroom = native["policy_headroom_evidence"]
    return dict(start=row["start_tick"], end=row["end_tick"], phase=info["phase_id"],
        intervention=row["intervention"], baseline=row["baseline_raw"], injected=row["injected_raw"],
        nominal=info["nominal_action_full12"], residual=info["projected_residual_full12"],
        final=info["actual_drive_target_full12"], mapped=headroom["baseline_native_plus_controller_full12"],
        effective=headroom["effective_policy_residual_full12"], clipped=headroom["clipped_servo_indices"],
        verified=native["verified"], mask=native["phase_mask_full12"], physical=task["physical_evaluator"],
        history=task["history"], base_mean=req["base_mean_full12"], conditional=req["conditional_mean_full12"],
        history_center=req["history_center_full12"], sigma=req["effective_sigma_full12"],
        four_hip=row.get("four_hip_geometry"), termination=info["termination_reason"])


def raw_rows(path, end):
    rows = []
    for row in lines(path / "physical_observations.jsonl", end):
        bounds, ob = row["body_bounds_w_m"]["base_link"], row["obstacle"]
        low, high = bounds["minimum_m"], bounds["maximum_m"]
        o_low = (ob["front_x_m"], ob["right_y_m"], ob["bottom_z_m"])
        o_high = (ob["back_x_m"], ob["left_y_m"], ob["top_z_m"])
        sep = math.sqrt(sum(max(a-y, x-b, 0.)**2 for a,b,x,y in zip(low,high,o_low,o_high)))
        rows.append(dict(tick=row["physics_tick"], t=row["simulation_time_s"],
            rpy=_rpy(_quaternion(row["base"]["orientation_wxyz"])), base=row["base"],
            q={name: row["joints"][name]["position_deg"] for name in JOINTS},
            body_min_z=low[2], separation_lower_bound=sep,
            FRgap=row["wheels"]["front_right_ankle"]["bottom_w_m"][2]-ob["top_z_m"],
            FLgap=row["wheels"]["front_left_ankle"]["bottom_w_m"][2]-ob["top_z_m"],
            FL_obstacle_active=row["contacts"]["front_left_wheel"]["obstacle"]["active"],
            FL_ground_active=row["contacts"]["front_left_wheel"]["ground"]["active"],
            full12=row["actual_full12"], com=row["center_of_mass"]))
    return rows


def fr_window(path, decisions):
    events = decisions[-1]["history"]["event_ticks"]
    end = events["placed"]["FR"]
    rows = raw_rows(path, end)
    assert [r["tick"] for r in rows] == list(range(end+1)), "missing real FR physics interval"
    total, integral = 0., 0.
    for a,b in zip(rows, rows[1:]):
        dt = b["t"]-a["t"]
        rates = [math.atan2(math.sin(y-x), math.cos(y-x))/dt for x,y in zip(a["rpy"][:2],b["rpy"][:2])]
        integral += .5*sum(v*v for v in rates)*dt
        total += dt
    result = dict(window="natural_P01_to_first_real_FR_placed", start_tick=0,end_tick=end,
        actual_intervals=end, duration_s=total, rate_rms_rad_s=math.sqrt(integral/total),
        tilt_peak_rad=max(math.hypot(*r["rpy"][:2]) for r in rows),
        collider_world_min_z_m=stats(r["body_min_z"] for r in rows),
        conservative_body_obstacle_separation_m=stats(r["separation_lower_bound"] for r in rows),
        FR_gap_m=stats(r["FRgap"] for r in rows), FR_events={k:v.get("FR") for k,v in events.items()},
        FR_gap_after_tick72_before_cross_m=stats(r["FRgap"] for r in rows if 72 <= r["tick"] < events["front_edge_crossed"]["FR"]),
        joints={}, learned_or_optimizer_updates=0)
    for name in JOINTS[:3]:
        lo,hi = servo_limits_deg(name)
        result["joints"][name] = dict(actual_deg=stats(r["q"][name] for r in rows),
            minimum_negative_margin_deg=min(r["q"][name]-lo for r in rows),
            minimum_positive_margin_deg=min(hi-r["q"][name] for r in rows), hard_limits_deg=[lo,hi])
    hs = list(lines(path / "height_diagnostics.jsonl", end))
    result["RR_mount_world_z_m_15Hz"] = stats(h["rr_hip_mount_w_m"]["value"][2] for h in hs if h["rr_hip_mount_w_m"]["value"] is not None)
    return result, rows


def first_difference(a, b):
    for x,y in zip(a,b):
        if x["tick"] != y["tick"]: return dict(reason="clock_mismatch")
        keys = []
        if x["full12"] != y["full12"]: keys.append("actual_full12")
        if x["base"]["position_w_m"] != y["base"]["position_w_m"]: keys.append("base_position")
        if x["base"]["orientation_wxyz"] != y["base"]["orientation_wxyz"]: keys.append("base_orientation")
        if keys: return dict(tick=x["tick"],fields=keys,actual_full12_max_abs_difference=max(abs(i-j) for i,j in zip(x["full12"],y["full12"])))
    return None


def action_example(r, physical_rows, channel, name):
    actual = next((x for x in physical_rows if x["tick"] == r["end"]), None)
    return dict(start_tick=r["start"], end_tick=r["end"], phase=r["phase"], intervention=r["intervention"],
        nominal_deg=r["nominal"][channel], mapped_nominal_same_dispatch_deg=r["mapped"][channel],
        filtered_request_deg=r["residual"][channel], effective_residual_deg=r["effective"][channel], final_deg=r["final"][channel],
        actual_post_step_deg=None if actual is None else actual["q"][name],
        network_mean=r["base_mean"][channel], history_center=r["history_center"][channel],
        conditional_mean=r["conditional"][channel], baseline_raw=r["baseline"][channel], applied_raw=r["injected"][channel])


def fl_followup(path, decisions):
    manifest_path=path/"run_manifest.json"
    if not manifest_path.exists(): return dict(available=False,reason="still active")
    manifest=json.loads(manifest_path.read_text())
    if manifest["lifecycle"] != "DIAGNOSTIC_SEALED": return dict(available=False,reason=manifest["lifecycle"])
    raw=raw_rows(path,manifest["endpoint_tick"])
    active=[r for r in decisions if r["intervention"] and "index" in r["intervention"]]
    start=active[0]["start"]
    release=next(r["start"] for r in active if r["intervention"]["release_age"] is not None)
    bounds={"pre":(start-24,start),"ramp":(start+1,start+96),"hold":(start+97,release),
            "release":(release+1,release+96),"follow":(release+97,manifest["endpoint_tick"])}
    windows={}
    for name,(lo,hi) in bounds.items():
        rows=[r for r in raw if lo<=r["tick"]<=hi]
        hips=[r["four_hip"] for r in decisions if lo<=r["end"]<=hi]
        windows[name]=dict(ticks=[lo,hi],physics_samples=len(rows),FL_gap_m=stats(r["FLgap"] for r in rows),
            FL_obstacle_contact_ticks=sum(r["FL_obstacle_active"] for r in rows),
            FL_ground_contact_ticks=sum(r["FL_ground_active"] for r in rows),
            FL_actual_hip_deg=stats(r["q"]["front_left_hip"] for r in rows),
            FL_actual_knee_deg=stats(r["q"]["front_left_knee"] for r in rows),
            body_min_world_z_m=stats(r["body_min_z"] for r in rows),
            body_obstacle_conservative_separation_m=stats(r["separation_lower_bound"] for r in rows),
            CoM_world_z_m=stats(r["com"]["position_w_m"][2] for r in rows),
            four_hip_world_z_m={leg:stats(h["hip_mount_world_m"][leg]["value"][2] for h in hips
                if h["hip_mount_world_m"][leg]["value"] is not None) for leg in ("FL","FR","RL","RR")})
    observed=[r for r in raw if start<=r["tick"]<=manifest["endpoint_tick"]]
    best=min(observed,key=lambda r:r["FLgap"])
    near_end=((best["tick"]+7)//8)*8
    selected=sorted(set((start,start+96,near_end,release,release+96,active[-1]["end"],manifest["endpoint_tick"])))
    examples=[action_example(r,raw,0,"front_left_hip") for r in decisions if r["end"] in selected]
    return dict(available=True,lifecycle=manifest["lifecycle"],endpoint_tick=manifest["endpoint_tick"],
        original_task_reason=manifest["original_task_reason"], external_budget_stop=manifest["external_budget_stop"],
        final_phase=manifest["final_phase"],learned_state_unchanged=manifest["learned_state_unchanged"],
        FL_events={k:v.get("FL") for k,v in decisions[-1]["history"]["event_ticks"].items()},
        first_P06_tick=next((r["end"] for r in decisions if r["phase"]=="P06"),None),
        intervention_start_tick=start,release_start_tick=release,
        minimum_post_intervention_gap=dict(tick=best["tick"],gap_m=best["FLgap"],
            hip_deg=best["q"]["front_left_hip"],knee_deg=best["q"]["front_left_knee"]),
        action_examples=examples,windows=windows,
        all_selected_native_verified=all(r["verified"] for r in active),
        intervention_channel_clipped_count=sum(0 in r["clipped"] for r in active),
        all_masks_full12=all(r["mask"]==[1]*12 for r in active),
        maximum_REQUEST_vs_effective_difference=max(abs(r["residual"][0]-r["effective"][0]) for r in active),
        note="finite external diagnostic, no on-policy credit; no unperturbed FL control run at this entry")


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True)
    args=parser.parse_args()
    ds = {name:[compact(r) for r in lines(RUNS/name/"probe_decisions.jsonl")] for name in NAMES}
    fr, a = fr_window(RUNS/NAMES[0], ds[NAMES[0]])
    control, b = fr_window(RUNS/NAMES[1], ds[NAMES[1]])
    result=dict(schema="task_direction_probe_readonly_pair.v1", checkpoint="CP185856",
        comparison="same_weight_deterministic_FR_finite_RL_intervention_vs_other_run_pre_FL_intervention_not_formal_PPO",
        rate_metric="sqrt(time_mean((roll_rate^2+pitch_rate^2)/2)), wrapped adjacent120Hz Euler derivatives",
        FR_RL_minus3=fr, no_FR_intervention=control, first_physical_difference=first_difference(a,b),
        rate_change_fraction=fr["rate_rms_rad_s"]/control["rate_rms_rad_s"]-1,
        peak_tilt_change_fraction=fr["tilt_peak_rad"]/control["tilt_peak_rad"]-1,
        duration_change_s=fr["duration_s"]-control["duration_s"])
    active=[r for r in ds[NAMES[0]] if r["intervention"] and "index" in r["intervention"]]
    choices=[active[i] for i in (0,11,74,75,86)]
    result["FR_probe_action_examples"]=[action_example(r,a,4,"rear_left_hip") for r in choices]
    result["FR_probe_finite_window"]=dict(first_start_tick=active[0]["start"], last_indexed_end_tick=active[-1]["end"],
        ramp_decisions=12, release_begins_at_index=75, release_decisions=12,
        note="5 seconds ramp/hold from anchored REQUEST, then .8 second release to live policy; not constant -3 through FR capture")
    result["control_FR_has_no_intervention"]=all(r["intervention"] is None for r in ds[NAMES[1]] if r["start"] < control["end_tick"])
    hip_rows=[r["four_hip"] for r in ds[NAMES[1]] if r["four_hip"] is not None]
    startup=json.loads((RUNS/NAMES[1]/"four_hip_geometry_startup.json").read_text())
    result["four_hip_geometry_validation"] = dict(startup=startup,
        sampled_decisions=len(hip_rows), last_sample_tick=hip_rows[-1]["physics_tick"],
        clocks_all_aligned=all(r["clock_unchanged"] and r["frame_clock_aligned"] for r in hip_rows),
        all_four_available=all(all(v["value"] is not None for v in r["hip_mount_world_m"].values()) for r in hip_rows),
        no_four_hip_stream_in_FR_probe=True)
    fl_path=RUNS/NAMES[1]; sealed=(fl_path/"run_manifest.json").exists()
    result["FL_run_state"]=dict(sealed=sealed, complete_decisions_read=len(ds[NAMES[1]]),last_tick=ds[NAMES[1]][-1]["end"],
        analyzed_FL_outcome=False, reason="FR-only bounded report; FL follow-up requires natural run seal")
    result["FL_followup"] = fl_followup(fl_path,ds[NAMES[1]])
    result["FL_run_state"]["analyzed_FL_outcome"] = result["FL_followup"]["available"]
    result["FL_run_state"]["reason"] = ("natural DIAGNOSTIC_SEALED; FL follow-up analyzed" if result["FL_followup"]["available"]
        else result["FL_followup"]["reason"])
    hs={h["physics_tick"]:h for h in lines(fl_path/"height_diagnostics.jsonl",ds[NAMES[1]][-1]["end"])}
    result["four_hip_geometry_validation"]["RR_mount_max_error_vs_independent_height_stream_m"]=max(
        abs(r["hip_mount_world_m"]["RR"]["value"][i]-hs[r["physics_tick"]]["rr_hip_mount_w_m"]["value"][i])
        for r in hip_rows for i in range(3) if r["physics_tick"] in hs)
    with args.output.open("x", encoding="utf-8") as stream: json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({k:result[k] for k in ("rate_change_fraction","peak_tilt_change_fraction","duration_change_s","first_physical_difference","FL_run_state")},indent=2))
    print(json.dumps(dict(FR=fr,control=control),indent=2))


if __name__=="__main__": main()
