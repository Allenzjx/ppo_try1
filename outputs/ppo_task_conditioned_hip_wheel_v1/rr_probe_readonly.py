"""Analyze already-sealed RR diagnostic logs; no simulator or policy execution."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from wlr50_clean.ppo.semantic_observation import _quaternion, _rpy
from wlr50_clean.infrastructure.command_batch import servo_limits_deg

B = ROOT / "runs/ppo_fl_capture_quality_v1/video_eval/prior_B/20260918T0555572376368Z_gf2e552406ea7_1894338b350241fcbbac4868c5f7f3fb/source"
LEGS = {"FL":"front_left", "FR":"front_right", "RL":"rear_left", "RR":"rear_right"}


def lines(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.endswith("\n"): break
            yield json.loads(line)


def stats(values):
    values=list(values)
    return None if not values else dict(n=len(values),minimum=min(values),maximum=max(values),mean=sum(values)/len(values),first=values[0],last=values[-1])


def compact_decision(r):
    s=r["step_info"]; p=s["semantic_task"]["physical_evaluator"]
    native=s["actuator_target_effect_audit"]; head=native["policy_headroom_evidence"]
    return dict(start=r["start_tick"],end=r["end_tick"],phase=s["phase_id"],end_phase=s["end_phase_id"],
        ev=p,history=s["semantic_task"]["history"],transitions=s["stage_transition_evidence"],
        intervention=r.get("intervention"),four_hip=r.get("four_hip_geometry"),
        nominal=s["nominal_action_full12"],mapped=head["baseline_native_plus_controller_full12"],
        request=s["projected_residual_full12"],effective=head["effective_policy_residual_full12"],
        final=s["actual_drive_target_full12"],mask=native["phase_mask_full12"],verified=native["verified"],
        clipped=head["clipped_servo_indices"],reason=s["termination_reason"],success=s["full_task_success"])


def compact_physical(r):
    bounds=r["body_bounds_w_m"]["base_link"]; ob=r["obstacle"]
    lo,hi=bounds["minimum_m"],bounds["maximum_m"]
    ol=(ob["front_x_m"],ob["right_y_m"],ob["bottom_z_m"])
    oh=(ob["back_x_m"],ob["left_y_m"],ob["top_z_m"])
    contacts={leg:r["contacts"][name+"_wheel"] for leg,name in LEGS.items()}
    return dict(tick=r["physics_tick"],t=r["simulation_time_s"],base=r["base"],full12=r["actual_full12"],
        rpy=_rpy(_quaternion(r["base"]["orientation_wxyz"])),
        body_min_z=lo[2],body_aabb= bounds,
        separation=math.sqrt(sum(max(a-y,x-b,0.)**2 for a,b,x,y in zip(lo,hi,ol,oh))),
        com=r["center_of_mass"],contacts=contacts,
        q={name:r["joints"][name]["position_deg"] for name in
           ("front_left_hip","rear_left_hip","rear_right_hip","rear_right_knee")},
        wheels={leg:r["wheels"][name+"_ankle"] for leg,name in LEGS.items()},
        gap={leg:r["wheels"][name+"_ankle"]["bottom_w_m"][2]-ob["top_z_m"] for leg,name in LEGS.items()})


def contact_active(row,leg):
    c=row["contacts"][leg]
    assert c["ground"]["pair_verified"] and c["obstacle"]["pair_verified"], "unverified pair"
    return c["ground"]["active"] or c["obstacle"]["active"]


def bool_runs(rows, predicate):
    runs=[]; start=None; last=None
    for r in rows:
        if predicate(r):
            if start is None: start=r["tick"]
            last=r["tick"]
        elif start is not None:
            runs.append(dict(first_tick=start,last_tick=last,samples=last-start+1)); start=None
    if start is not None: runs.append(dict(first_tick=start,last_tick=last,samples=last-start+1))
    return dict(count=len(runs),total_samples=sum(r["samples"] for r in runs),
        longest_observed_duration_s=max((r["samples"]-1 for r in runs),default=0)/120,
        runs=runs,semantics="inclusive 120Hz observations; duration last-first, not an unseen interval")


def events(ds):
    transitions=[t for d in ds for t in d["transitions"]]
    hist=ds[-1]["history"]
    rr_events=[e for e in hist.get("lift_attempt_events",[]) if e.get("leg")=="RR"]
    return dict(P08=next((t["physics_tick"] for t in transitions if t["to_stage"]=="P08"),None),
        P09=next((t["physics_tick"] for t in transitions if t["to_stage"]=="P09"),None),
        P10=next((t["physics_tick"] for t in transitions if t["to_stage"]=="P10"),None),
        initial=next((e["physics_tick"] for e in rr_events if e["event"]=="whole_body_initial_clearance"),None),
        qualified=hist["event_ticks"]["active_lift"].get("RR"),
        crossed=hist["event_ticks"]["front_edge_crossed"].get("RR"),
        placed=hist["event_ticks"]["placed"].get("RR"),lift_attempt_events=rr_events)


def physical_window(rows,lo,hi):
    rs=[r for r in rows if lo<=r["tick"]<=hi]
    if not rs: return None
    assert [r["tick"] for r in rs]==list(range(lo,hi+1)), "missing physical interval"
    dt=rs[-1]["t"]-rs[0]["t"]; integral=0.
    for a,b in zip(rs,rs[1:]):
        step=b["t"]-a["t"]
        rates=[math.atan2(math.sin(y-x),math.cos(y-x))/step for x,y in zip(a["rpy"][:2],b["rpy"][:2])]
        integral+=.5*sum(v*v for v in rates)*step
    origin=rs[0]["com"]["position_w_m"]
    toward=[rs[0]["wheels"]["FL"]["center_w_m"][i]-origin[i] for i in range(2)]
    norm=math.hypot(*toward); toward=[v/norm for v in toward]
    displacement=[rs[-1]["com"]["position_w_m"][i]-origin[i] for i in range(3)]
    joints={}
    for name in rs[0]["q"]:
        lower,upper=servo_limits_deg(name)
        joints[name]=dict(actual_deg=stats(r["q"][name] for r in rs),
            minimum_negative_margin_deg=min(r["q"][name]-lower for r in rs),
            minimum_positive_margin_deg=min(upper-r["q"][name] for r in rs),hard_limits_deg=[lower,upper])
    return dict(ticks=[lo,hi],duration_s=dt,physics_samples=len(rs),
        rate_RMS_rad_s=math.sqrt(integral/dt) if dt else None,peak_tilt_rad=max(math.hypot(*r["rpy"][:2]) for r in rs),
        RR_gap_m=stats(r["gap"]["RR"] for r in rs), FL_gap_m=stats(r["gap"]["FL"] for r in rs),
        body_collider_min_world_z_m=stats(r["body_min_z"] for r in rs),
        conservative_body_obstacle_AABB_separation_m=stats(r["separation"] for r in rs),
        com_world_z_m=stats(r["com"]["position_w_m"][2] for r in rs),
        com_world_displacement_m=displacement,initial_com_toward_FL_fixed_world_xy=toward,
        com_displacement_toward_initial_FL_direction_m=sum(x*y for x,y in zip(displacement,toward)),
        com_displacement_does_not_imply_FL_bearing=True,
        actual_contact_ticks={leg:dict(ground=sum(r["contacts"][leg]["ground"]["active"] for r in rs),
             obstacle=sum(r["contacts"][leg]["obstacle"]["active"] for r in rs),
             air=sum(not contact_active(r,leg) for r in rs)) for leg in LEGS},
        RR_contact_free_runs=bool_runs(rs,lambda r:not contact_active(r,"RR")),
        RR_ground_contact_runs=bool_runs(rs,lambda r:r["contacts"]["RR"]["ground"]["active"]),
        FL_obstacle_normal_force_n=stats(r["contacts"]["FL"]["obstacle"]["normal_force_n"] for r in rs),
        RR_total_normal_force_n=stats(sum(r["contacts"]["RR"][p]["normal_force_n"] for p in ("ground","obstacle")) for r in rs),
        joints=joints,exact_cartesian_feasibility=None,exact_mesh_obstacle_clearance_m=None)


def sampled_window(ds,hs,lo,hi):
    selected=[d for d in ds if lo<=d["end"]<=hi]
    if not selected: return None
    legs=[d["ev"]["current_legs"] for d in selected]
    rr=[a["RR"] for a in legs]; fl=[a["FL"] for a in legs]
    mounts=[h for h in hs if lo<=h["physics_tick"]<=hi and h["rr_hip_mount_w_m"]["value"] is not None]
    four=[d["four_hip"] for d in selected if d["four_hip"] is not None]
    contexts=[d["ev"]["transfer_roles"]["RR"]["transfer_direction_context"] for d in selected]
    return dict(decision_samples=len(selected),RR_current_lift_valid_samples=sum(r["current_lift_valid"] for r in rr),
        RR_air_samples=sum(r["air"] for r in rr),RR_current_support_samples=sum(r["support"] for r in rr),
        RR_load_fraction_valid_samples=sum(r["load_fraction_valid"] for r in rr),
        RR_load_fraction=stats(r["load_fraction"] for r in rr if r["load_fraction_valid"]),
        RR_contact_mode_counts={mode:sum(r["contact_mode"]==mode for r in rr) for mode in sorted(set(r["contact_mode"] for r in rr))},
        FL_current_support_samples=sum(r["support"] for r in fl),FL_top_bearing_samples=sum(
            r["support"] and r["top_surface_contact"] and r["bearing_verified"] and not r["air"] for r in fl),
        FL_air_samples=sum(r["air"] for r in fl),FL_bearing_force_n=stats(r["bearing_force_n"] for r in fl),
        FL_toward_CoM_rolling_context_m=stats(c["com_toward_receiver_m"] for c in contexts),
        positive_CoM_toward_FL_while_FL_air_samples=sum(c["com_toward_receiver_m"]>0 and r["air"] for c,r in zip(contexts,fl)),
        RR_mount_world_z_m=stats(h["rr_hip_mount_w_m"]["value"][2] for h in mounts),
        four_hip_world_z_m=({leg:stats(h["hip_mount_world_m"][leg]["value"][2] for h in four if h["hip_mount_world_m"][leg]["value"] is not None) for leg in LEGS} if four else None),
        last_RR_current_state={k:rr[-1][k] for k in ("air","current_lift_valid","contact_mode","clearance_m","front_distance_m","support","bearing_force_n","load_fraction","load_fraction_valid","placed_on_top")})


def analyze(path,decision_name,budget_end):
    ds=[compact_decision(r) for r in lines(path/decision_name) if "step_info" in r]
    endpoint=min(budget_end,ds[-1]["end"])
    raw=[compact_physical(r) for r in lines(path/"physical_observations.jsonl") if r["physics_tick"]<=endpoint]
    hs=list(lines(path/"height_diagnostics.jsonl")); ev=events(ds)
    start=ev["P08"]
    ranges={}
    if start is not None and start<=endpoint:
        ranges["P08_to_placed_or_budget"]=(start,min(ev["placed"] or endpoint,endpoint))
        ranges["P08_to_common_budget"]=(start,endpoint)
    if ev["qualified"] is not None and ev["qualified"]<=endpoint:
        if start is not None: ranges["P08_to_first_qualified"]=(start,ev["qualified"])
        ranges["qualified_to_placed_or_budget"]=(ev["qualified"],min(ev["placed"] or endpoint,endpoint))
    if ev["placed"] is not None and ev["placed"]<=endpoint:
        ranges["post_placed_to_common_budget"]=(ev["placed"],endpoint)
    active=[d for d in ds if d["intervention"] and "index" in d["intervention"]]
    result=dict(path=str(path),events=ev,analysis_endpoint_tick=endpoint,
        final_history_event_ticks=ds[-1]["history"]["event_ticks"],
        analysis_endpoint_semantics="common bounded horizon no longer than 60s; B's full success retained independently",
        actual_run_final_tick=ds[-1]["end"],actual_run_final_phase=ds[-1]["end_phase"],
        actual_run_reason=ds[-1]["reason"],actual_run_full_success=ds[-1]["success"],
        windows={name:dict(physical_120Hz=physical_window(raw,lo,hi),sampled_15Hz=sampled_window(ds,hs,lo,hi)) for name,(lo,hi) in ranges.items()})
    # A video may stop on a real physical terminal inside the last 8-tick decision.
    # Its last returned step is NOT the authoritative full-episode result.
    source_manifest=path/"semantic_video_source_manifest.json"
    if source_manifest.exists():
        source=json.loads(source_manifest.read_text())
        result["last_completed_decision_tick"]=ds[-1]["end"]
        result["actual_run_final_tick"]=source["episode_physics_ticks"]
        result["actual_run_full_success"]=source["physical_task_success"]
        result["actual_run_reason"]=source["physical_episode"]["physical_task_evaluation"]["termination_reason"]
        result["full_episode_result_provenance"]=str(source_manifest)
    four=[d["four_hip"] for d in ds if d["four_hip"] is not None]
    hmap={h["physics_tick"]:h for h in hs}
    result["four_hip_validation"]=(dict(samples=len(four),
        all_clocks_aligned=all(h["clock_unchanged"] and h["frame_clock_aligned"] for h in four),
        all_four_available=all(all(p["value"] is not None for p in h["hip_mount_world_m"].values()) for h in four),
        RR_max_xyz_error_vs_independent_height_stream_m=max(abs(h["hip_mount_world_m"]["RR"]["value"][i]
            -hmap[h["physics_tick"]]["rr_hip_mount_w_m"]["value"][i]) for h in four for i in range(3))) if four else None)
    first_ground=next((r for r in raw if ev["qualified"] is not None and ev["qualified"]<r["tick"]<=(ev["placed"] or endpoint)
        and r["contacts"]["RR"]["ground"]["active"]),None)
    seen_valid=False; first_invalid=None
    for d in ds:
        if d["end"]>(ev["placed"] or endpoint):break
        rr=d["ev"]["current_legs"]["RR"]
        if rr["current_lift_valid"]:seen_valid=True
        elif seen_valid:
            first_invalid=dict(tick=d["end"],RR=rr);break
    result["first_sampled_current_lift_loss_before_placed"]=first_invalid
    result["first_raw_ground_after_qualified_before_placed_tick"]=None if first_ground is None else first_ground["tick"]
    snapshot_ticks={k:v for k,v in ev.items() if isinstance(v,int)}
    if first_ground is not None:snapshot_ticks["first_ground_after_qualified"]=first_ground["tick"]
    by_tick={r["tick"]:r for r in raw}
    result["physical_event_snapshots"]={name:dict(tick=tick,body_min_world_z_m=by_tick[tick]["body_min_z"],
        conservative_body_obstacle_AABB_separation_m=by_tick[tick]["separation"],com_world_m=by_tick[tick]["com"]["position_w_m"],
        joint_actual_deg=by_tick[tick]["q"],RR_gap_m=by_tick[tick]["gap"]["RR"],FL_gap_m=by_tick[tick]["gap"]["FL"],
        contacts={leg:dict(ground=by_tick[tick]["contacts"][leg]["ground"]["active"],
            obstacle=by_tick[tick]["contacts"][leg]["obstacle"]["active"]) for leg in LEGS})
        for name,tick in snapshot_ticks.items() if tick in by_tick}
    if active:
        channels=[int(i) for i in active[0]["intervention"]["delta_deg"]]
        release=next((d for d in active if d["intervention"]["release_age"] is not None),None)
        sample_indices=sorted(set((0,min(11,len(active)-1),min(13,len(active)-1),len(active)-1,
            next((i-1 for i,d in enumerate(active) if d is release),len(active)-1))))
        actual={r["tick"]:r for r in raw}
        names={0:"front_left_hip",4:"rear_left_hip"}
        result["intervention"]=dict(start_tick=active[0]["start"],release_start_tick=release["start"] if release else None,
            last_indexed_end_tick=active[-1]["end"],first_receipt=active[0]["intervention"],last_receipt=active[-1]["intervention"],
            full12_masks_all_one=all(d["mask"]==[1]*12 for d in active),native_verified=all(d["verified"] for d in active),
            selected_clip_count=sum(any(c in d["clipped"] for c in channels) for d in active),
            max_REQUEST_vs_effective_difference=max(abs(d["request"][c]-d["effective"][c]) for c in channels for d in active),
            examples=[dict(start=d["start"],end=d["end"],receipt=d["intervention"],
                channels={names[c]:dict(nominal=d["nominal"][c],mapped_same_dispatch=d["mapped"][c],REQUEST=d["request"][c],
                    effective=d["effective"][c],final=d["final"][c],actual=actual[d["end"]]["q"][names[c]]) for c in channels})
                for d in [active[i] for i in sample_indices] if d["end"] in actual])
    else: result["intervention"]=None
    return result,raw


def first_difference(a,b):
    for x,y in zip(a,b):
        assert x["tick"]==y["tick"]
        keys=[]
        for key in ("full12","base"):
            if x[key]!=y[key]: keys.append(key)
        if keys:return dict(tick=x["tick"],fields=keys,
            max_full12_absolute_difference_mixed_units=max(abs(i-j) for i,j in zip(x["full12"],y["full12"])))
    return None


def first_dispatch_difference(probe_path):
    probe=(r for r in lines(probe_path/"probe_decisions.jsonl") if "step_info" in r)
    reference=(r for r in lines(B/"video_policy_decisions.jsonl") if "step_info" in r)
    fields=("nominal_action_full12","projected_residual_full12","actual_drive_target_full12","phase_id","end_phase_id")
    for a,b in zip(probe,reference):
        assert a["end_tick"]==b["end_tick"], "decision clocks differ"
        x,y=a["step_info"],b["step_info"]
        changed=[k for k in fields if x[k]!=y[k]]
        if changed:return dict(start_tick=a["start_tick"],end_tick=a["end_tick"],fields=changed,
            values={k:dict(probe=x[k],B=y[k]) for k in changed},all_preceding_fields_identical=True)
    return None


def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();manifest=json.loads((args.run/"run_manifest.json").read_text())
    assert manifest["lifecycle"] in ("DIAGNOSTIC_SEALED","DIAGNOSTIC_ERROR"), "run must naturally seal first"
    probe,pr=analyze(args.run,"probe_decisions.jsonl",7200)
    reference,br=analyze(B,"video_policy_decisions.jsonl",min(7200,probe["analysis_endpoint_tick"]))
    manifest_keys=("schema","case","baseline","checkpoint","checkpoint_sha256","actual_head","source_head",
        "all_checkpoint_runtime_files_identical","new_PPO_decisions","new_PPO_updates","new_optimizer_steps",
        "diagnostic_only","source_N_changed","state_injection","max_seconds","harness_sha256","lifecycle",
        "endpoint_tick","final_phase","original_task_reason","probe_started","probe_complete",
        "actual_diagnostic_decisions","external_budget_stop","learned_state_unchanged","completed_at_utc")
    compact_manifest={k:manifest.get(k) for k in manifest_keys}
    result=dict(schema="RR_direction_diagnostic_readonly.v1",source_manifest_path=str(args.run/"run_manifest.json"),
        source_manifest=compact_manifest,
        diagnostic_only=True,new_PPO_decisions=0,new_PPO_updates=0,reference=reference,probe=probe,
        first_physical_difference_vs_B=first_difference(pr,br),
        first_dispatch_difference_vs_B=first_dispatch_difference(args.run),
        rate_metric="sqrt(time_mean((roll_rate^2+pitch_rate^2)/2)), wrapped adjacent120Hz Euler derivatives",
        qualification_not_equal_to_clearance_or_placement=True,CoM_direction_not_equal_to_FL_bearing=True)
    with args.output.open("x",encoding="utf-8") as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(events=probe["events"],B_events=reference["events"],
        first_difference=result["first_physical_difference_vs_B"],manifest=compact_manifest),indent=2))


if __name__=="__main__":main()
