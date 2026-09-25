"""Stdlib, completed-row-only FR-placement aligned diagnostic; no model imports.

Explicit --cutoff is the last already-observed complete episode tick. This
script does not poll/wait, decode video or read beyond that fixed window. It
returns JSON to stdout; the caller publishes reviewed artifacts with apply_patch.
"""
import argparse
import json
import math
from pathlib import Path

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
SOURCE=ROOT/'runs/ppo_rr_rl_timing_policy_learning_v1/video_eval/validation/20260924T1641350708339Z_g892385cba8a7_de3f11b3a669411abf3e6658002008cc/source'
JOINTS=['front_left_hip','front_left_knee','front_right_hip','front_right_knee',
        'rear_left_hip','rear_left_knee','rear_right_hip','rear_right_knee']


def complete_rows(path):
    with path.open(encoding='utf-8-sig') as stream:
        for line in stream:
            if not line.endswith('\n'):
                break
            yield json.loads(line)


def pitch(q):
    w,x,y,z=q
    return math.degrees(math.asin(max(-1.,min(1.,2*(w*y-z*x)))))


def read_window(cutoff, *, source=SOURCE, checkpoint=226304, sealed=False):
    selected=[]; fr=None; count=0
    for row in complete_rows(source/'video_policy_decisions.jsonl'):
        tick=row['end_tick']
        if tick>cutoff:
            break
        s=row['step_info']; e=s['semantic_task']['physical_evaluator']
        fr=e['history']['event_ticks']['placed'].get('FR',fr)
        selected.append(row);count+=1
        if fr is None and len(selected)>9:
            selected.pop(0)
        if fr is not None and tick>=fr+720:
            break
    if fr is None or cutoff<fr+720:
        raise ValueError('FR placed +6s window not yet complete; do not claim a failed capture')
    rows=[r for r in selected if fr-60<=r['end_tick']<=fr+720]
    picks=[min(rows,key=lambda r:abs(r['end_tick']-(fr+off*120))) for off in (-.5,0,.5,1,2,4,6)]
    ticks={r['end_tick'] for r in picks}; heights={}
    for row in complete_rows(source/'height_diagnostics.jsonl'):
        tick=row['physics_tick']
        if tick>max(ticks):break
        if tick in ticks:heights[tick]=row
    if set(heights)!=ticks:
        raise ValueError('missing exact-tick height receipt; wait for matching completed diagnostic rows')
    points=[]
    for off,r in zip((-.5,0,.5,1,2,4,6),picks):
        tick=r['end_tick'];s=r['step_info'];a=s['actuator_target_effect_audit']
        h=a['policy_headroom_evidence'];tr=a['tracking_reference_evidence'];d=heights[tick]
        e=s['semantic_task']['physical_evaluator']; geom=d['same_rigid_body_hip_mount_geometry']['value']
        if not geom['valid'] or not geom['source_verified']:
            raise ValueError('unverified actual rigid-body mount geometry')
        actual=[d['joints_canonical']['value'][name]['position_deg'] for name in JOINTS]
        n=s['nominal_action_full12'];mn=a['native_drive_target_full12']
        req=h['requested_policy_residual_full12'];eff=h['effective_policy_residual_full12'];final=s['actual_drive_target_full12']
        wheel_actual=e['measured_wheel_velocity_rad_s']
        assist=a['capture_assist_evidence'];p=r['policy_request']
        points.append(dict(requested_offset_s=off,offset_s=(tick-fr)/120,tick=tick,stage=s['end_phase_id'],
            FR_hip_chain=[n[2],mn[2],req[2],eff[2],final[2],actual[2]],
            FR_knee_chain=[n[3],mn[3],req[3],eff[3],final[3],actual[3]],
            FL_hip_final_actual=[final[0],actual[0]],FL_knee_final_actual=[final[1],actual[1]],
            wheel_chain=[n[8:],mn[8:],req[8:],eff[8:],final[8:],wheel_actual],
            FR_raw_hip_knee=r['raw_policy_action_full12'][2:4],
            FR_mean_hip_knee=p['base_mean_full12'][2:4],FR_conditional_mean=p['conditional_mean_full12'][2:4],
            headroom_clip_indices=h['clipped_servo_indices'],phase_mask=a['phase_mask_full12'],
            mapping_verified=a['actual_mapping_matches_dispatch'],
            assist_mode=assist['state_after']['mode'],assist_owner_indices=assist['owner_indices'],
            assist_correction=assist['assist_correction_full12'],rear_owner17=p['rear_owner_observed_features'],
            base_xyz_m=d['base']['value']['position_w_m'],COM_xyz_m=d['center_of_mass']['value']['position_w_m'],
            pitch_deg=pitch(d['base']['value']['orientation_wxyz']),
            mounts_world=geom['mount_world_m'],left_minus_right_m=geom['left_minus_right_mean_z_m'],
            rear_minus_front_m=geom['rear_minus_front_mean_z_m'],mount_valid=True,mount_source_verified=True,
            legs={leg:{k:v.get(k) for k in ('clearance_m','front_distance_m','air','ground_contact','top_contact',
                'within_top_xy','bearing_force_n','contact_reaction_force_n','current_lift_valid')} for leg,v in e['current_legs'].items()}))
    old=json.loads((OUT/'CP225792_DET_FRplaced_front_space_6s.json').read_text(encoding='utf-8'))
    latest=selected[-1]['step_info']; pe=latest['semantic_task']['physical_evaluator']
    return dict(schema='bounded_live_eval_FR_placed_6sec_comparison_v1',checkpoint=checkpoint,
        source=str(source),cutoff_tick=cutoff,FR_placed_tick=fr,
        chain_order=['source_N','same_tick_mapped_N','filtered_REQUEST','headroom_effective','FINAL','actual'],
        wheel_order=['FL','FR','RL','RR'],endpoints=points,
        comparison_prior=[dict(label='CP225792',FR_placed_tick=old['FR_placed_tick'],endpoints=old['endpoints'])]+old['prior'],
        complete_window_decisions=len(rows),latest_read=dict(tick=latest['physics_tick'],sim_s=latest['sim_time_s'],
            phase=latest['end_phase_id'],termination=latest['termination_reason'],events=pe['history']['event_ticks']),
        full_linkage_clearance='N/A: wheel gap/mount geometry are not entire linkage swept-clearance proof',
        claims=dict(partial_active_evidence=not sealed,full_task_success=False,physical_retention_from_KL=False,
            pure_network_FL_capture=False,production_changed=False),
        method='exact completed decision/height ticks; actual USD body0 mount definitions + same rigid body pose; no joint-angle proxy',
        scopes=['Event aligned, not common wall time; all checkpoint/prehistory differences retained.',
            'Original FL assist separately declared; rear task assists remain off.',
            'Policy contribution comes from same-tick audit, not independent nominal subtraction.',
            'An unobserved future event is pending, not a declared task failure.'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cutoff',type=int,required=True)
    p.add_argument('--source',type=Path,default=SOURCE);p.add_argument('--checkpoint',type=int,default=226304)
    args=p.parse_args();print(json.dumps(read_window(args.cutoff,source=args.source,checkpoint=args.checkpoint),ensure_ascii=False))
