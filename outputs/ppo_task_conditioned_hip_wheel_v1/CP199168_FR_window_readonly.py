"""Completed FR prefix only; existing physical-window metric, no simulation."""
import hashlib
import json
import math
from pathlib import Path
from rr_probe_readonly import compact_physical, physical_window, stats
from video_fr_preview_readonly import extract

OUT=Path(__file__).resolve().parent; ROOT=OUT.parents[1]
RUN=ROOT/'runs/ppo_task_conditioned_hip_wheel_v1/video_eval/validation/20260921T1429210894242Z_g649ccd906421_0c5fc0498bbd4acca26ca3dde5d10193'
OLD=json.loads((OUT/'CP194560_FR_window_readonly.json').read_text())
B=json.loads((OUT/'currentB_event_windows.json').read_text())
started=json.loads((RUN/'run_manifest.started.json').read_text())
assert started['arguments']['seed']==4001 and started['arguments']['stochastic_policy'] is False
assert Path(started['arguments']['checkpoint']).name=='checkpoint_step_000199168.pt'

def read_closed(path,end):
    digest=hashlib.sha256();size=count=0;rows=[]
    with path.open('rb') as f:
        for line in f:
            assert line.endswith(b'\n')
            r=json.loads(line);tick=r['physics_tick'];assert tick<=end
            digest.update(line);size+=len(line);count+=1
            rows.append(compact_physical(r))
            if tick==end:break
    assert [r['tick'] for r in rows]==list(range(end+1))
    return rows,dict(path=str(path),through_tick=end,complete_lines=count,consumed_bytes=size,
        consumed_prefix_sha256=digest.hexdigest(),whole_file_hash=False)

def window(rows,lo,hi):
    rs=rows[lo:hi+1];p=physical_window(rows,lo,hi);duration=p['duration_s']
    angles={name:math.sqrt(sum(.5*(a['rpy'][i]**2+b['rpy'][i]**2)*(b['t']-a['t'])
        for a,b in zip(rs,rs[1:]))/duration) for i,name in enumerate(('roll','pitch'))}
    return dict(ticks=[lo,hi],duration_s=duration,angle_RMS_rad=angles,
        tilt_RMS_rad=math.sqrt(sum(x*x for x in angles.values())),rate_RMS_rad_s=p['rate_RMS_rad_s'],
        peak_tilt_rad=p['peak_tilt_rad'],body_collider_minimum_world_z_m=p['body_collider_min_world_z_m'],
        conservative_obstacle_AABB_distance_m=p['conservative_body_obstacle_AABB_separation_m'],
        FR_gap_m=stats(r['gap']['FR'] for r in rs),
        body_advance_m=rs[-1]['base']['position_w_m'][0]-rs[0]['base']['position_w_m'][0],
        RL_actual_hip_deg=p['joints']['rear_left_hip']['actual_deg'],
        RR_actual_hip_deg=p['joints']['rear_right_hip']['actual_deg'],
        RR_actual_pair_contact_samples=p['actual_contact_ticks']['RR'],
        RR_total_normal_force_n=p['RR_total_normal_force_n'],
        RR_independently_classified_support_count=None,
        RR_support_note='120Hz verified pair contacts/normal reaction are observed; evaluator support bit not read or inferred')

def b_window(name,old_full=None):
    w=B['windows'][name];p=w['physical'];angle=None if old_full is None else old_full['kinematics']['angle_RMS_rad']
    return dict(ticks=[w['start_tick'],w['observed_end_tick']],duration_s=p['duration_s'],angle_RMS_rad=angle,
        tilt_RMS_rad=None if angle is None else math.sqrt(sum(x*x for x in angle.values())),
        rate_RMS_rad_s=p['rate_RMS_rad_s'],peak_tilt_rad=p['peak_tilt_rad'],
        body_collider_minimum_world_z_m=p['body_collider_min_world_z_m'],
        conservative_obstacle_AABB_distance_m=p['conservative_body_obstacle_AABB_separation_m'],
        FR_gap_m=p['wheel_bottom_gap_above_top_m']['FR'],body_advance_m=p['body_forward_displacement_m'],
        RL_actual_hip_deg=p['joints']['rear_left_hip']['actual_deg'],RR_actual_hip_deg=p['joints']['rear_right_hip']['actual_deg'],
        RR_actual_pair_contact_samples=p['actual_contact_ticks']['RR'],RR_total_normal_force_n=None,
        RR_independently_classified_support_samples=w['current_support_samples']['RR'],
        missing_qualified_angle_reason=None if angle is not None else 'not in existing B summary; old B raw not reread')

new_full=extract(RUN/'source',2074,with_kinematics=True)
assert new_full['FR_events']==dict(active_lift=23,front_edge_crossed=2064,placed=2074)
new,proof=read_closed(RUN/'source/physical_observations.jsonl',2074)
old_full=OLD['records']['CP194560_det_PPO_LIMITEDAUX']
old,old_proof=read_closed(Path(old_full['source'])/'physical_observations.jsonl',1795)
assert abs(window(old,0,1795)['rate_RMS_rad_s']-old_full['rate_RMS_rad_s'])<1e-14
records=dict(current_B=dict(FR_P01_to_capture=b_window('FR_P01_to_capture',OLD['records']['current_B']),
                           FR_qualified_to_cross=b_window('FR_qualified_to_cross')),
    CP194560_det=dict(FR_P01_to_capture=window(old,0,1795),FR_qualified_to_cross=window(old,23,1785)),
    CP199168_det_receiving=dict(FR_P01_to_capture=window(new,0,2074),FR_qualified_to_cross=window(new,23,2064)))
result=dict(schema='CP199168.provisional_closed_FR_windows.v1',run=str(RUN),checkpoint='CP199168',
    candidate_read_through_tick=2074,whole_episode_outcome=None,full_run_sealed_or_success_claim=False,
    label='fixed CP receiving profile deterministic; training lineage may include limited auxiliary data, no aux-specific attribution',
    FR_events=new_full['FR_events'],records=records,
    FR_safe_gap={k:v['FR_gap_first_safe_before_cross_m'] for k,v in
        (('current_B',OLD['records']['current_B']),('CP194560_det',old_full),('CP199168_det_receiving',new_full))},
    source_prefix_proof=proof,CP194560_new_subwindow_prefix_proof=old_proof,
    reused_reference_files={n:hashlib.sha256((OUT/n).read_bytes()).hexdigest()
        for n in ('CP194560_FR_window_readonly.json','currentB_event_windows.json')},
    semantics='120Hz wrapped finite-difference roll/pitch joint rate RMS=sqrt(integral((rdot^2+pdot^2)/2)dt/T); angles trapezoidal RMS; tilt=hypot(roll,pitch). No extra division by duration. Collider minimum world z not base_z or exact mesh clearance.',
    comparison_limits='Same physical event endpoints and reset seed, not matched speed or single-factor intervention. Receiving gate inactive in FR and deterministic kernel itself unchanged for identical weights; training/trajectory differs. B runtime older; no global whole-run or causal ranking.')
with (OUT/'CP199168_FR_window_readonly.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
for name,ws in records.items():
    for label,w in ws.items():
        print(name,label,json.dumps(w))
print('SAFE_GAP',json.dumps(result['FR_safe_gap']))
print('PREFIX',json.dumps(proof))
