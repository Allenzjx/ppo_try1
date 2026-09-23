"""Text-prepared, output-only sealed v5 video physics analysis. NOT a launcher.

Execute only after explicit owner confirmation that the fixed source sealed
and its Isaac process exited. No runtime, model, Torch, image or video imports.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "runs/ppo_rr_capture_then_rl_transfer_v1/video_eval/validation/20260923T0718224682359Z_gc53119ab332f_886689c9a357416293bc8242856d0be6/source"
HEAD = "c53119ab332fe048668e66f0543889a128443ca5"
CP_SHA = "308122a3c8e760733fbe8370bfdf399a25718148c82333deb41a1265e4e13895"
BASE = HERE / "diagnose_57ae41e_sealed_fixed_com.py"
BASE_SHA = "b521aabc6ddc208ab65d2357aed97a8b2e3cadd42e2b6433e5129992a781ade1"


def shared():
    if hashlib.sha256(BASE.read_bytes()).hexdigest() != BASE_SHA:
        raise RuntimeError("previous reviewed diagnostic helper changed")
    spec = importlib.util.spec_from_file_location("_sealed_fixed_COM_reader",BASE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def yaw(raw):
    q = (raw.get("base") or {}).get("orientation_wxyz")
    if not isinstance(q,list) or len(q)!=4 or not all(type(v) in (float,int) and math.isfinite(v) for v in q):
        return None
    if not math.isclose(sum(v*v for v in q),1.,abs_tol=1e-4):
        return None
    w,x,y,z=q
    return math.degrees(math.atan2(2*(w*z+x*y),1-2*(y*y+z*z)))


def rr_relative_position(m,raw):
    """Actual wheel center in measured base frame; NOT a CoM-fixed frame."""
    wheel=((raw.get("wheels") or {}).get("rear_right_ankle") or {}).get("center_w_m")
    base=raw.get("base") or {}; origin=base.get("position_w_m"); q=base.get("orientation_wxyz")
    com=(raw.get("center_of_mass") or {}).get("position_w_m")
    result={"RR_wheel_center_relative_base_frame_m":None,
        "RR_wheel_center_minus_mass_COM_world_axes_m":None,"coordinate_origin_is_not_COM":True}
    if m.finite_vector(wheel) and m.finite_vector(com):
        result["RR_wheel_center_minus_mass_COM_world_axes_m"]=[wheel[i]-com[i] for i in range(3)]
    if not (m.finite_vector(wheel) and m.finite_vector(origin) and m.finite_vector(q,4)
            and math.isclose(sum(v*v for v in q),1.,abs_tol=1e-4)):
        return result
    w,x,y,z=q; x,y,z=-x,-y,-z
    vx,vy,vz=[wheel[i]-origin[i] for i in range(3)]
    tx,ty,tz=2*(y*vz-z*vy),2*(z*vx-x*vz),2*(x*vy-y*vx)
    result["RR_wheel_center_relative_base_frame_m"]=[vx+w*tx+y*tz-z*ty,vy+w*ty+z*tx-x*tz,vz+w*tz+x*ty-y*tx]
    return result


def main(destination):
    m=shared(); inventory={}
    run=m.read_json(SOURCE.parent/"run_manifest.json",inventory)
    source=m.read_json(SOURCE/"semantic_video_source_manifest.json",inventory)
    m.require(run.get("completed_at_utc") and run.get("lifecycle") in ("FAILED","SUCCEEDED","DIAGNOSTIC_FAILURE"),"source not sealed")
    if run.get("lifecycle")=="DIAGNOSTIC_FAILURE":
        m.require(source.get("physical_task_success") is False and bool(source.get("source_acceptance_error")),
            "diagnostic failure must retain its actual incomplete/error outcome")
    m.require(source.get("schema")=="wlr50_clean.semantic_video_source.v1" and source.get("role")=="C"
        and source.get("experiment_id")=="rr_capture_then_rl_transfer_v1" and source.get("from_phase")=="P01"
        and source.get("fresh_process_single_episode") is True and source.get("episode_count")==1
        and source.get("optimizer_updates")==0,"not one natural-P01, frozen-policy video episode")
    contract=source.get("runtime_contract") or {}
    m.require(contract.get("source_git_commit")==HEAD and run.get("runtime_contract")==contract,"wrong v5 runtime")
    load=source.get("checkpoint_load_provenance") or {}
    m.require(load.get("checkpoint_loaded_and_verified") is True and load.get("official_load_semantic_checkpoint") is True
        and load.get("optimizer_updates")==0 and (load.get("source") or {}).get("checkpoint_sha256")==CP_SHA,
        "actual loader/source CP differs; ancestry must not be inferred from an arbitrary matching nested hash")
    terminal=source.get("episode_physics_ticks")
    m.require(type(terminal) is int and 0<terminal<=24000,"invalid physical episode extent")
    milestones,entries,stages={}, {}, {}
    def add(tick,label):
        if type(tick) is int and 0<=tick<=terminal and label not in milestones.setdefault(tick,[]):
            milestones[tick].append(label)
    add(terminal,"actual_terminal_or_capture_end")
    add(terminal-240,"terminal_minus_2s")
    add(terminal-120,"terminal_minus_1s")
    for seconds in (3,4,5): add(terminal-seconds*120,f"terminal_minus_{seconds}s")
    for row in m.lines(SOURCE/"stage_transition_evidence.jsonl",inventory):
        tick,phase=row["physics_tick"],row.get("to_stage")
        if phase in ("P07","P09","P10","P11","P12","P13") and phase not in entries:
            entries[phase]=tick; add(tick,"first_"+phase); stages[tick]=row
        for event in ("active_lift","front_edge_crossed","placed"):
            exact=row.get("physical_history",{}).get("event_ticks",{}).get(event,{}).get("RR")
            if type(exact) is int:
                add(exact,"RR_history_"+event)
    decisions={}; issued=completed=last_end=0; partial=None
    for row in m.lines(SOURCE/"video_policy_decisions.jsonl",inventory):
        issued+=1
        m.require(row.get("decision")==issued and row.get("start_tick")==last_end,"non-contiguous video decision envelopes")
        end=row.get("end_tick")
        m.require(type(end) is int and last_end<end<=terminal,"invalid video decision end")
        last_end=end
        if row.get("environment_step_returned") is not True:
            m.require(partial is None and end==terminal and "step_info" not in row,"invalid interrupted final decision")
            partial=row; continue
        completed+=1; info=row.get("step_info")
        m.require(isinstance(info,dict) and info.get("physics_tick")==end,"missing/misaligned actual step_info")
        task=info.get("semantic_task") or {}
        decisions[end]={"video_envelope":m.fields(row,("decision","request_phase","start_tick","end_tick","physics_ticks")),
            "phase_start":info.get("phase_id"),"phase_end":info.get("end_phase_id"),"termination_reason":info.get("termination_reason"),
            "history":task.get("history"),"current_legs":(task.get("physical_evaluator") or {}).get("current_legs"),
            "raw_policy_full12":info.get("raw_policy_action_full12"),"policy_request":row.get("policy_request"),
            "final_drive_full12":info.get("actual_drive_target_full12")}
    m.require(last_end==terminal and issued==source.get("issued_policy_decisions")
        and completed==source.get("completed_environment_steps"),"source/decision counts or tail disagree")
    natives,seen,previous={},set(),None; ncount=0; last_pre_top=False; native_top_seen=False
    placed_tick=next((tick for tick,labels in milestones.items() if "RR_history_placed" in labels),None)
    for row in m.lines(SOURCE/"native_tick_audit.jsonl",inventory):
        tick=row["episode_physics_tick"]; m.require(tick==ncount+1,"native ticks not contiguous"); ncount=tick
        audit=row["native_audit"]; rr=audit.get("rr_capture_assist_evidence") or {}; wheel=audit.get("rr_carry_wheel_evidence") or {}
        state,ctx=rr.get("state_after") or {},rr.get("context") or {}; labels=[]
        if wheel.get("envelope_active") is True: labels.append("first_actual_wheel_envelope")
        if wheel.get("actual_projection_changed") is True: labels.append("first_actual_wheel_projection")
        if row.get("source_phase_id") in ("P09","P10","P11","P12","P13"):
            mode=state.get("mode_name")
            if mode not in (None,"WAIT"): labels.append("first_RR_mode_"+mode)
            if state.get("travel_used_deg",0)>20.: labels.append("first_knee_search")
            if state.get("travel_used_deg",0)>40.: labels.append("first_progress_earned_reserve")
            if state.get("retired")==1.: labels.append("first_RR_owner_retired")
            increment=rr.get("captured_incremental_target")
            if increment is not None:
                labels.append("first_captured_increment_receipt")
                for key in ("nominal_delta_rr_deg","requested_residual_delta_rr_deg"):
                    if any(v!=0 for v in increment.get(key,[])): labels.append("first_captured_nonzero_"+key)
            for field,label in (("top_surface_contact","TOP"),("current_top_bearing","TOP_bearing")):
                if ctx.get(field) is True:
                    event="first_pre_dispatch_RR_"+label
                    labels.append(event)
                    if event not in seen:
                        pt=ctx.get("source_observation_tick"); add(pt,"first_observed_RR_"+label)
                        if previous and previous["episode_physics_tick"]==pt:
                            natives[pt]=native_summary(m,previous)
            lost=native_top_seen and last_pre_top and ctx.get("top_surface_contact") is False
            after_placed=(placed_tick is not None and ctx.get("source_observation_tick",-1)>placed_tick
                and ctx.get("top_surface_contact") is False)
            for occurs,event in ((lost,"first_pre_dispatch_RR_TOP_loss"),(after_placed,"first_pre_dispatch_nonTOP_after_placed")):
                if occurs:
                    labels.append(event)
                    if event not in seen:
                        pt=ctx.get("source_observation_tick"); add(pt,event.replace("pre_dispatch","observed"))
                        if previous and previous["episode_physics_tick"]==pt: natives[pt]=native_summary(m,previous)
            last_pre_top=ctx.get("top_surface_contact") is True
            native_top_seen=native_top_seen or last_pre_top
        for label in labels:
            if label not in seen:
                seen.add(label); add(tick,label)
        if tick in milestones: natives[tick]=native_summary(m,row)
        previous=row
    m.require(ncount==terminal,"native tail differs from episode end")
    captures={}; count=0; top_seen=False; prior_top=False; placed_seen=False; first_lost=False; first_air=False; first_placed_loss=False
    cross_tick=next((tick for tick,labels in milestones.items() if "RR_history_front_edge_crossed" in labels),None)
    contact_counts={"post_RR_cross_ticks":0,"ground_contact":0,"obstacle_pair_active":0,
        "top_surface_contact":0,"support":0,"max_bearing_force_n":0.}
    for row in m.lines(SOURCE/"capture_assist_ticks.jsonl",inventory):
        tick=row["episode_physics_tick"]; m.require(tick==count+1,"capture ticks not contiguous"); count=tick
        rr=(row.get("current_legs") or {}).get("RR") or {}; top=rr.get("top_surface_contact") is True
        if cross_tick is not None and tick>=cross_tick:
            contact_counts["post_RR_cross_ticks"]+=1
            for key in ("ground_contact","obstacle_pair_active","top_surface_contact","support"):
                contact_counts[key]+=int(rr.get(key) is True)
            force=rr.get("bearing_force_n")
            if type(force) in (int,float) and math.isfinite(force):
                contact_counts["max_bearing_force_n"]=max(contact_counts["max_bearing_force_n"],force)
        if top and not top_seen: add(tick,"first_post_step_RR_TOP"); top_seen=True
        if top_seen and prior_top and not top and not first_lost:
            add(tick,"first_RR_TOP_loss"); first_lost=True
        if top_seen and rr.get("air") is True and not first_air:
            add(tick,"first_RR_AIR_after_TOP"); first_air=True
        placed_seen=placed_seen or (row.get("placed_history") or {}).get("RR") is True
        if placed_seen and not top and not first_placed_loss:
            add(tick,"first_nonTOP_after_history_placed"); first_placed_loss=True
        if tick in milestones:
            captures[tick]=m.fields(row,("episode_physics_tick","sim_time_s","phase","current_legs","placed_history",
                "rr_capture_assist","rr_capture_transfer_context","rr_capture_transfer_diagnostics","rr_capture_continuation",
                "policy_permission_mask_full12","nominal_full12","actual_full12","dispatch"))
        prior_top=top
    m.require(count==terminal,"capture evidence tail differs from episode end")
    # Optional 15Hz/forced-terminal height stream: exact alignment only. A nearby
    # sample retains its own tick and is NOT silently substituted at an event.
    heights={}; prior=None; targets=iter(sorted(milestones)); target=next(targets,None)
    if (SOURCE/"height_diagnostics.jsonl").is_file():
        for row in m.lines(SOURCE/"height_diagnostics.jsonl",inventory):
            tick=row["physics_tick"]
            while target is not None and target<tick:
                heights[target]={"exact":None,"nearest_prior":prior}; target=next(targets,None)
            if target==tick:
                heights[target]={"exact":row,"nearest_prior":None}; target=next(targets,None)
            prior=row
        while target is not None:
            heights[target]={"exact":None,"nearest_prior":prior}; target=next(targets,None)
    startup=None
    if (SOURCE/"height_diagnostics_startup.json").is_file():
        startup=m.read_json(SOURCE/"height_diagnostics_startup.json",inventory)
    anchors,samples={},[]; pcount=-1
    min_gap=None; tail_ranges={seconds:{"rows":0,"minimum_gap_m":None,"maximum_gap_m":None,
        "first":None,"last":None} for seconds in (2,5)}
    for raw in m.lines(SOURCE/"physical_observations.jsonl",inventory):
        tick=raw["physics_tick"]; m.require(tick==pcount+1,"physical ticks not contiguous"); pcount=tick
        for phase,receiver in (("P07","FL"),("P10","FR")):
            if entries.get(phase)==tick:
                anchors[phase]=m.make_anchor(raw,phase,receiver); anchors[phase]["body_yaw0_deg"]=yaw(raw)
        if cross_tick is not None and tick>=cross_tick:
            rr=(raw.get("wheels") or {}).get("rear_right_ankle") or {}
            bottom=rr.get("bottom_w_m"); top=(raw.get("obstacle") or {}).get("top_z_m")
            gap=bottom[2]-top if m.finite_vector(bottom) and type(top) in (float,int) else None
            if gap is not None and (min_gap is None or gap<min_gap["RR_gap_m"]):
                min_gap=m.physical_summary(raw)
            for seconds,stats in tail_ranges.items():
                if tick<terminal-seconds*120 or gap is None: continue
                stats["rows"]+=1
                stats["minimum_gap_m"]=gap if stats["minimum_gap_m"] is None else min(stats["minimum_gap_m"],gap)
                stats["maximum_gap_m"]=gap if stats["maximum_gap_m"] is None else max(stats["maximum_gap_m"],gap)
                compact={"tick":tick,"time_s":raw.get("simulation_time_s"),"gap_m":gap,
                    "actual_full12":raw.get("actual_full12"),"commanded_full12":raw.get("commanded_full12"),
                    "RR_contact_pairs":(raw.get("contacts") or {}).get("rear_right_wheel")}
                if stats["first"] is None: stats["first"]=compact
                stats["last"]=compact
        if tick not in milestones: continue
        physical=m.physical_summary(raw); physical["body_yaw_deg"]=yaw(raw)
        physical.update(rr_relative_position(m,raw))
        height=heights.get(tick) or {}; exact=height.get("exact") or {}; mount=exact.get("rr_hip_mount_w_m") or {}
        if exact.get("clock_unchanged") is True and exact.get("frame_raw_physics_tick")==tick and m.finite_vector(mount.get("value")):
            physical.update(RR_hip_mount_w_m=mount["value"],RR_hip_mount_unavailable_reason=None)
        else:
            physical["RR_hip_mount_unavailable_reason"]="no valid same-tick height diagnostic; nearby sample is separately timestamped"
        projections={}
        for phase in ("P07","P10"):
            anchor=anchors.get(phase); projection=m.fixed_projection(raw,anchor)
            if projection:
                com=(raw.get("center_of_mass") or {}).get("position_w_m")
                projection["COM_world_delta_xy_m"]=[com[i]-anchor["COM0_w_m"][i] for i in range(2)] if m.finite_vector(com) else None
                a0,a1=anchor.get("body_yaw0_deg"),yaw(raw)
                projection["body_yaw_delta_deg"]=(a1-a0+180)%360-180 if a0 is not None and a1 is not None else None
            projections[phase]=projection
        samples.append({"labels":milestones[tick],"physical_post_step":physical,"dispatch_pre_step":natives.get(tick),
            "current_capture_post_step":captures.get(tick),"exact_decision_endpoint":decisions.get(tick),
            "stage_event":stages.get(tick),"fixed_reference_projections":projections,"height_diagnostic":height})
    m.require(pcount==terminal,"physical tail differs from source endpoint")
    for phase in ("P07","P10"):
        anchors.setdefault(phase,{"valid":False,"reason":"not reached; no receiver direction inferred"})
    # Check only the evidence already read, not video/full-checkpoint bytes.
    for name,record in inventory.items():
        sealed=(source.get("artifacts") or {}).get(name)
        if sealed: m.require(sealed.get("sha256")==record["sha256"],"sealed evidence digest differs: "+name)
    physical=source.get("physical_episode") or {}
    result={"schema":"output_only.sealed_v5_capture_follow_fixed_COM.v1","source":str(SOURCE),"runtime_HEAD":HEAD,
        "checkpoint_load_provenance":load,"task_success":physical.get("task_success"),
        "physical_task_evaluation":physical.get("physical_task_evaluation"),"episode_ticks":terminal,"duration_s":terminal/120.,
        "issued_eval_decisions":issued,"completed_environment_steps":completed,"interrupted_final_decision":partial,
        "source_acceptance_error":source.get("source_acceptance_error"),"physical_analysis_is_not_media_validation":True,
        "new_optimizer_updates":0,"new_physics_ticks":0,"inventory":inventory,"anchors":anchors,
        "RR_TOP_ever_observed":top_seen,"RR_placed_ever_observed":placed_seen,"RR_TOP_loss_observed":first_lost,
        "post_cross_contact_counts":contact_counts,"minimum_post_cross_gap_actual_row":min_gap,"last_seconds_ranges":tail_ranges,
        "height_startup":startup,"milestones":samples,
        "limits":["Current contact/bearing is separate from placed history; actual sensor predicates are not relabelled.",
            "Native global dispatch, episode pre-observation, and post-step ticks stay separate.",
            "Fixed d uses true mass COM and entry receiver position, never rolling reanchored or moved-foot credit.",
            "World-fixed diagonal progress contains forward traversal and can change interpretation with yaw; not lateral-load proof.",
            "P10 missing means no RL-to-FR direction. Hip mount is exact height evidence or null, not upper-link origin.",
            "No task success, policy improvement, mean attribution, GAE or learning credit is inferred.",
            "Final video decision can be physically interrupted without step_info; no fake completion row is synthesized."]}
    destination.mkdir(parents=True,exist_ok=False)
    with (destination/"v5_capture_follow_fixed_COM.json").open("x",encoding="utf-8") as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    rows=["# Sealed v5 capture/follow evidence","",f"Source: `{SOURCE}`",f"Physical success: `{result['task_success']}`; {terminal}/120 = {terminal/120.} s.",
        "", "| Tick | Milestone | RR gap mm | Front mm | P07 CoM mm | P10 CoM mm |", "|---:|---|---:|---:|---:|---:|"]
    def mm(v): return None if v is None else round(v*1000,5)
    for sample in samples:
        p=sample["physical_post_step"]; f=sample["fixed_reference_projections"]
        rows.append(f"| {p['post_tick']} | {', '.join(sample['labels'])} | {mm(p['RR_gap_m'])} | {mm(p['RR_front_distance_m'])} | {mm((f['P07'] or {}).get('COM_toward_receiver_m'))} | {mm((f['P10'] or {}).get('COM_toward_receiver_m'))} |")
    rows += [""]+result["limits"]
    (destination/"v5_capture_follow_fixed_COM.md").write_text("\n".join(rows)+"\n",encoding="utf-8")
    print(json.dumps({"destination":str(destination),"ticks":terminal,"milestones":len(samples),"success":result["task_success"]}))


def native_summary(m,row):
    result=m.native_summary(row); audit=row["native_audit"]
    result.update(rr_capture_assist_branch_replays=audit.get("rr_capture_assist_branch_replays"),
        issued_inputs_independently_verified=audit.get("rr_capture_issued_inputs_independently_verified"))
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner-confirmed-sealed",action="store_true",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(); main(args.output)
