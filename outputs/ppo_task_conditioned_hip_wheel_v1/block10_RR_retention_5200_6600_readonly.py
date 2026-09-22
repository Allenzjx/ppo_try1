"""One closed active-training window, never read past the requested tick6600."""
import hashlib
import json
from pathlib import Path
from rr_probe_readonly import ROOT, stats

OUT=Path(__file__).resolve().parent
RUN=ROOT/"runs/ppo_task_conditioned_hip_wheel_v1/train/20260921T1043344171428Z_g97ecd305afb5_0168bb1943e74402accb92ea31b10de3"
path=RUN/"residual_and_projection_audit.jsonl"
rows=[]; digest=hashlib.sha256();prefix_bytes=0;prefix_lines=0
with path.open("rb") as stream:
    for line in stream:
        assert line.endswith(b"\n"),"closed window not yet written"
        r=json.loads(line);s=r["applied_audit"];t=s["physics_tick"]
        assert t<=6600,"refuse future record"
        digest.update(line);prefix_bytes+=len(line);prefix_lines+=1
        if t>=5200:
            ev=s["semantic_task"]["physical_evaluator"]; a=s["actuator_target_effect_audit"];h=a["policy_headroom_evidence"]
            nd=s["semantic_task"]["nominal_provider_diagnostics"]
            rows.append(dict(tick=t,phase=s["phase_id"],end_phase=s["end_phase_id"],global_decision=r["global_policy_decision"],terminal=r["terminal"],
                ev=ev,history=s["semantic_task"]["history"],transitions=s["stage_transition_evidence"],
                body_front_m=ev["goal_features"]["body_forward_m"],RR=ev["current_legs"]["RR"],FL=ev["current_legs"]["FL"],
                nominal_wheels=s["nominal_action_full12"][8:],mapped_wheels=h["baseline_native_plus_controller_full12"][8:],
                wheel_policy_raw=r["raw_policy_action_full12"][8:],wheel_REQUEST=s["projected_residual_full12"][8:],
                wheel_effective=h["effective_policy_residual_full12"][8:],wheel_final=s["actual_drive_target_full12"][8:],
                wheel_measured_canonical=ev["measured_wheel_velocity_rad_s"],mask=a["phase_mask_full12"],
                dispatch_verified=a["verified"] and a["setter_dispatch_targets_equal"] and a["actual_mapping_matches_dispatch"],
                tick_dispatch=s["actuator_target_effect_audit_ticks"],
                owner= {k:nd.get(k) for k in ("final_stop_owner","rr_carry_continuation","p06_rolling_retirement","p06_wheel_tail")},
                target_source=a["actual_target_source"]))
        if t==6600:break
assert [r["tick"] for r in rows]==list(range(5200,6601,8))
placed=rows[-1]["history"]["event_ticks"]["placed"]["RR"]
assert placed==5542
post=[r for r in rows if r["tick"]>=placed]
def top(r):
    a=r["RR"];return a["top_surface_contact"] and a["bearing_verified"] and a["support"]
def compact(r):
    return dict(tick=r["tick"],phase=r["phase"],end_phase=r["end_phase"],body_front_m=r["body_front_m"],
        RR={k:r["RR"].get(k) for k in ("contact_mode","current_lift_valid","support","bearing_force_n","within_top_xy","front_distance_m","clearance_m","placed_on_top")},
        RR_current_TOP_verified_bearing=top(r),
        FL={k:r["FL"].get(k) for k in ("air","support","top_surface_contact","bearing_force_n","front_distance_m","clearance_m")},
        nominal_wheels=r["nominal_wheels"],mapped_wheels=r["mapped_wheels"],policy_raw_wheels=r["wheel_policy_raw"],
        REQUEST_wheels=r["wheel_REQUEST"],effective_wheels=r["wheel_effective"],final_wheels=r["wheel_final"],
        measured_canonical_wheels=r["wheel_measured_canonical"],
        final_stop_owner_active=r["owner"]["final_stop_owner"]["active"],
        source_rr_wheel_owner=r["owner"].get("rr_carry_continuation"),
        p06_retirement=r["owner"]["p06_rolling_retirement"],
        last_target_source=r["target_source"])
first_non_top=next((r for r in post if not top(r)),None)
first_air=next((r for r in post if r["RR"]["air"]),None)
first_ground=next((r for r in post if r["RR"]["ground_contact"]),None)
first_outside=next((r for r in post if not r["RR"]["within_top_xy"]),None)
changes=[compact(r) for i,r in enumerate(rows) if i==0 or (r["nominal_wheels"],r["phase"],r["end_phase"])!=(rows[i-1]["nominal_wheels"],rows[i-1]["phase"],rows[i-1]["end_phase"])]
contact_changes=[compact(r) for i,r in enumerate(post) if i==0 or (r["RR"]["contact_mode"],top(r),r["RR"]["within_top_xy"],r["RR"]["current_lift_valid"])!=(post[i-1]["RR"]["contact_mode"],top(post[i-1]),post[i-1]["RR"]["within_top_xy"],post[i-1]["RR"]["current_lift_valid"])]
selected={5200,post[0]["tick"],6600,max(post,key=lambda r:r["body_front_m"])["tick"],max(post,key=lambda r:r["RR"]["front_distance_m"])["tick"]}
for r in (first_non_top,first_air,first_ground,first_outside):
    if r is not None:selected.update((r["tick"]-8,r["tick"]))
windows=[]
for lo,hi in ((5200,post[0]["tick"]),(post[0]["tick"],6600)):
    w=[r for r in rows if lo<=r["tick"]<=hi]
    windows.append(dict(ticks=[lo,hi],body_net_advance_m=w[-1]["body_front_m"]-w[0]["body_front_m"],
        wheel_center_net_forward_m={leg:w[-1]["ev"]["current_legs"][leg]["front_distance_m"]-w[0]["ev"]["current_legs"][leg]["front_distance_m"] for leg in ("FL","FR","RL","RR")},
        RR_TOP_bearing_endpoints=sum(top(r) for r in w),RR_ground_endpoints=sum(r["RR"]["ground_contact"] for r in w),RR_AIR_endpoints=sum(r["RR"]["air"] for r in w),
        FL_TOP_bearing_endpoints=sum(r["FL"]["top_surface_contact"] and r["FL"]["bearing_verified"] and r["FL"]["support"] for r in w),FL_AIR_endpoints=sum(r["FL"]["air"] for r in w),n=len(w),
        RR_target=stats(r["wheel_final"][3] for r in w),RR_actual_canonical_qd=stats(r["wheel_measured_canonical"][3] for r in w),
        RR_negative_target_count=sum(r["wheel_final"][3]<0 for r in w),RR_negative_actual_count=sum(r["wheel_measured_canonical"][3]<0 for r in w)))
result=dict(schema="block10.closed_RR_retention_5200_6600.v1",source=str(path),
    bounded_prefix=dict(end_tick=6600,complete_lines=prefix_lines,bytes=prefix_bytes,sha256=digest.hexdigest(),not_whole_active_file_hash=True),
    classification="ongoing stochastic training with updates; not fixed checkpoint comparison or episode final result",
    wheel_order=["FL","FR","RL","RR"],wheel_velocity_units="rad/s canonical; raw native measured qd absent/null",
    event_ticks=rows[-1]["history"]["event_ticks"],
    RR_events=[e for e in rows[-1]["history"]["lift_attempt_events"] if e["leg"]=="RR"],
    transitions=[t for r in rows for t in r["transitions"]],
    first_postcapture_saved=dict(capture_tick=placed,first_endpoint=post[0]["tick"],first_non_TOP_bearing_tick=None if first_non_top is None else first_non_top["tick"],
        first_AIR_tick=None if first_air is None else first_air["tick"],first_GROUND_tick=None if first_ground is None else first_ground["tick"],first_outside_top_xy_tick=None if first_outside is None else first_outside["tick"]),
    postcapture_body_front_peak=compact(max(post,key=lambda r:r["body_front_m"])),
    postcapture_RR_front_peak=compact(max(post,key=lambda r:r["RR"]["front_distance_m"])),
    source_or_phase_changes=changes,postcapture_contact_or_validity_changes=contact_changes,
    selected_snapshots=[compact(r) for r in rows if r["tick"] in selected],windows=windows,
    dispatch=dict(all_full12_masks_one=all(r["mask"]==[1]*12 for r in rows),all_final_dispatch_verified=all(r["dispatch_verified"] for r in rows),
        all_interval_tick_dispatch_verified=all(t["verified"] for r in rows[1:] for t in r["tick_dispatch"]),
        handoff_hold_used_count=sum(t["handoff_hold_used"] for r in rows[1:] for t in r["tick_dispatch"]),
        final_stop_owner_active_count=sum(r["owner"]["final_stop_owner"]["active"] for r in rows),
        final_minus_same_dispatch_mapped_effective_max_abs=max(abs(f-n-e) for r in rows for f,n,e in zip(r["wheel_final"],r["mapped_wheels"],r["wheel_effective"])),
        full_last_source_event_owner_ledger=None,raw_native_measured_qd=None),
    endpoint_physical=dict(tick=6600,terminal=rows[-1]["terminal"],valid=rows[-1]["ev"]["valid"],termination=rows[-1]["ev"]["termination_reason"],success=rows[-1]["ev"]["success"]),
    limitations="15Hz current evaluator endpoints are not full contact time. Exact current contact loss lies between last prior and first changed endpoint unless explicit event tick exists. Wheel spin/body translation/leg endpoint motion distinct; no traction causality or optimizer-start gate. Future ticks not read.")
with (OUT/"block10_RR_retention_5200_6600.json").open("x",encoding="utf-8") as stream:
    json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps({k:result[k] for k in ("event_ticks","RR_events","transitions","first_postcapture_saved","windows","dispatch","endpoint_physical")},indent=2))
for r in result["selected_snapshots"]:
    print("SNAP",r["tick"],r["phase"],r["end_phase"],r["RR"],r["body_front_m"],"N",r["nominal_wheels"],"eff",r["effective_wheels"],"final",r["final_wheels"],"qd",r["measured_canonical_wheels"])
print("CHANGES",[(r["tick"],r["phase"],r["end_phase"],r["nominal_wheels"]) for r in changes])
