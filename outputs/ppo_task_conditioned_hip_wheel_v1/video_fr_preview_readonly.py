"""Bounded FR preview from a live video's already-written physical window."""
import argparse
import json
import math
from pathlib import Path
from rr_probe_readonly import B, compact_physical, lines, physical_window, stats


def extract(source,end,with_kinematics=False):
    raw=[]
    rear_names=("rear_left_hip","rear_left_knee","rear_right_hip","rear_right_knee")
    for r in lines(source/"physical_observations.jsonl"):
        if r["physics_tick"]>end:break
        compact=compact_physical(r)
        if with_kinematics:
            compact["rear_actual"]={name:r["joints"][name]["position_deg"] for name in rear_names}
            compact["rear_command"]={name:r["joints"][name]["command_deg"] for name in rear_names}
        raw.append(compact)
        if r["physics_tick"]==end:break
    assert [r["tick"] for r in raw]==list(range(end+1)),"FR physical window not fully written"
    history=None
    for r in lines(source/"stage_transition_evidence.jsonl"):
        if r["physics_tick"]>end:break
        history=r["physical_history"]
        if r["physics_tick"]==end:break
    assert history["event_ticks"]["placed"].get("FR")==end,"end must be verified FR capture"
    full=physical_window(raw,0,end)
    events={k:v.get("FR") for k,v in history["event_ticks"].items()}
    safe=next((r["tick"] for r in raw if r["gap"]["FR"]>=.015 and not any(r["contacts"]["FR"][p]["active"] for p in ("ground","obstacle"))),None)
    result=dict(source=str(source),physical_window_ticks=[0,end],actual_physics_intervals=end,
        FR_events=events,duration_s=full["duration_s"],rate_RMS_rad_s=full["rate_RMS_rad_s"],
        peak_tilt_rad=full["peak_tilt_rad"],body_collider_minimum_world_z_m=full["body_collider_min_world_z_m"],
        conservative_body_obstacle_AABB_separation_m=full["conservative_body_obstacle_AABB_separation_m"],
        FR_gap_all_window_m=stats(r["gap"]["FR"] for r in raw),first_AIR_gap_at_least_15mm_tick=safe,
        FR_gap_first_safe_before_cross_m=stats(r["gap"]["FR"] for r in raw if safe is not None and safe<=r["tick"]<events["front_edge_crossed"]))
    if with_kinematics:
        duration=full["duration_s"]
        result["kinematics"]=dict(
            angle_RMS_rad={axis:math.sqrt(sum(.5*(a["rpy"][i]**2+b["rpy"][i]**2)*(b["t"]-a["t"])
                for a,b in zip(raw,raw[1:]))/duration) for i,axis in enumerate(("roll","pitch"))},
            angle_RMS_semantics="trapezoidal time mean of squared measured Euler angle, rad",
            body_forward_displacement_m=raw[-1]["base"]["position_w_m"][0]-raw[0]["base"]["position_w_m"][0],
            rear_joints={name:dict(actual_deg=stats(r["rear_actual"][name] for r in raw),
                final_drive_command_deg=stats(r["rear_command"][name] for r in raw)) for name in rear_names},
            rear_wheels={leg:dict(actual_qd_rad_s=stats(r["wheels"][leg]["velocity_rad_s"] for r in raw),
                command_rad_s=stats(r["wheels"][leg]["command_rad_s"] for r in raw)) for leg in ("RL","RR")},
            RR_world_mount_z_m=None,mount_absence_reason="Not derived from base or CoM; this bounded reader reads physical observations, not USD mounts.")
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);p.add_argument("--end",type=int,required=True);p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();a=extract(args.run/"source",args.end);b=extract(B,1502)
    started=json.loads((args.run/"run_manifest.started.json").read_text())
    request=next(lines(args.run/"source/video_policy_decisions.jsonl")).get("policy_request")
    result=dict(schema="live_video_bounded_FR_preview.v1",classification="PRELIMINARY_REFERENCE_NOT_CURRENT_SAME_VERSION_PAIRED_RESULT",
        video_still_may_be_running=True,whole_episode_outcome=None,
        started_arguments=started["arguments"],actual_first_policy_sampling_mode=None if request is None else request.get("mode"),
        candidate=a,historical_reference=b,
        differences=dict(duration_s=a["duration_s"]-b["duration_s"],rate_RMS_fraction=a["rate_RMS_rad_s"]/b["rate_RMS_rad_s"]-1,
            peak_tilt_fraction=a["peak_tilt_rad"]/b["peak_tilt_rad"]-1,
            collider_minimum_z_m=a["body_collider_minimum_world_z_m"]["minimum"]-b["body_collider_minimum_world_z_m"]["minimum"]),
        metric="sqrt(integral((roll_rate squared+pitch_rate squared)/2*dt)/T), wrapped adjacent120Hz Euler derivatives; not RMS divided by duration again",
        limitations="Different saved versions; oldB is preliminary reference only. No final manifest required/fabricated. Window ends at FR capture; no claim about P05/full task success.")
    with args.output.open("x",encoding="utf-8") as stream:json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(result,ensure_ascii=False))
