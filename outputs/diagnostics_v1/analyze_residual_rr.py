"""Focused sealed RR windows; no policy forward, relabeling, or simulation."""
import itertools
import json
import math
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from wlr50_clean.ppo.semantic_physical_sensing import physical_contact_surface

RUNS={
 'CP177152':('runs/ppo_task_first_recovery_v1/video_eval/validation/20260916T0550249947408Z_g00050a2b1452_086805789a6e491392ef4656c7708c2c/source',4292,5464,4940,5428),
 'retained_zero':('runs/ppo_fsm_reference_p09_stable_v2/video_eval/prior_B/20260915T0525295619176Z_g4a58c0190ef7_615f8fe3cfc64cff81f73ac6a0d302e6/source',5100,6184,5434,6155),
}
LEGS=['FL','FR','RL','RR']; NAMES=['front_left','front_right','rear_left','rear_right']

def bounded_lines(path,start,stop):
    with path.open(encoding='utf8') as stream:
        yield from (json.loads(line) for line in itertools.islice(stream,start,stop))

def segments(rows,predicate):
    groups=[]
    for r in rows:
        if not predicate(r):continue
        if not groups or groups[-1][-1]['tick']+1!=r['tick']:groups.append([r])
        else:groups[-1].append(r)
    return [{'start':g[0]['tick'],'end':g[-1]['tick'],'duration_s':len(g)/120,
             'RR_z_gain_m':g[-1]['RR_bottom_z']-g[0]['RR_bottom_z'],
             'RR_gap_first_last_m':[g[0]['RR_gap'],g[-1]['RR_gap']],
             'RR_qd_range':[min(r['RR_qd'] for r in g),max(r['RR_qd'] for r in g)],
             'RR_command_range':[min(r['final'][11] for r in g),max(r['final'][11] for r in g)]} for g in groups]

results={}
for label,(relative,start,end,qualified,cross) in RUNS.items():
    source=ROOT/relative
    physical={r['physics_tick']:r for r in bounded_lines(source/'physical_observations.jsonl',start,end+1)}
    native={r['episode_physics_tick']:r for r in bounded_lines(source/'native_tick_audit.jsonl',start-1,end)}
    assert set(physical)==set(native)==set(range(start,end+1))
    decisions={}
    for d in bounded_lines(source/'video_policy_decisions.jsonl',start//8-2,end//8+1):
        if start<=d['end_tick']<=end:
            a=d['step_info']; e=a['semantic_task']['physical_evaluator']
            rr=e['current_legs']['RR']; role=e['transfer_roles']['RR']; direction=role['transfer_direction_context']
            decisions[d['end_tick']]={
                'phase':a['phase_id'],'end_phase':a['end_phase_id'],
                'RR_current':rr,'RR_role':role,
                'other_current':{leg:e['current_legs'][leg] for leg in LEGS[:-1]},
                'CoM_to_FL_past_window_m':direction['com_toward_receiver_m'],
                'CoM_velocity_to_FL_m_s':direction['com_velocity_toward_receiver_m_s'],
                'CoM_direction_reference_tick':direction['reference_tick'],
                'history_active_RR':e['history']['active_lift']['RR'],
                'history_crossed_RR':e['history']['front_edge_crossed']['RR'],
                'history_placed_RR':e['history']['placed']['RR']}
    data=[]
    for tick,p in physical.items():
        n=native[tick]; a=n['native_audit']; obstacle=p['obstacle']
        surf={}; force={}; verified={}
        for leg,name in zip(LEGS,NAMES):
            c=p['contacts'][name+'_wheel']
            values=[physical_contact_surface(c[kind],kind=kind,obstacle=obstacle,
                tolerance_m=.005,force_noise_floor_n=.2) for kind in ('ground','obstacle')]
            surf[leg]=values[1]['surface']
            force[leg]=sum(v['bearing_force_n'] for v in values if v['bearing_verified'])
            verified[leg]=all(v['bearing_verified'] for v in values)
        total=sum(force.values()); valid=all(verified.values()) and total>0
        wheel=p['wheels']['rear_right_ankle']; rr=p['contacts']['rear_right_wheel']
        fl=p['contacts']['front_left_wheel']; com=p['center_of_mass']['position_w_m']
        r={'tick':tick,'time_s':p['simulation_time_s'],'phase':n['source_phase_id'],
           'RR_surface':surf['RR'],'RR_ground':rr['ground']['active'],'RR_obstacle':rr['obstacle']['active'],
           'RR_air':not rr['ground']['active'] and not rr['obstacle']['active'],
           'RR_bearing_N':force['RR'],'RR_bearing_verified':verified['RR'],
           'RR_load_fraction':force['RR']/total if valid else None,'all_load_fractions_valid':valid,
           'RR_obstacle_force_w_n':rr['obstacle']['force_w_n'],
           'RR_obstacle_contact_point_w_m':rr['obstacle']['contact_point_w_m'],
           'RR_center':wheel['center_w_m'],'RR_bottom_z':wheel['bottom_w_m'][2],
           'RR_gap':wheel['bottom_w_m'][2]-obstacle['top_z_m'],
           'RR_front_distance':wheel['center_w_m'][0]-obstacle['front_x_m'],
           'RR_qd':wheel['velocity_rad_s'],'RR_hip_actual':p['joints']['rear_right_hip']['position_deg'],
           'RR_knee_actual':p['joints']['rear_right_knee']['position_deg'],
           'FL_air':not fl['ground']['active'] and not fl['obstacle']['active'],
           'FL_surface':surf['FL'],'FL_bearing_N':force['FL'],
           'FL_load_fraction':force['FL']/total if valid else None,'bearing_by_leg':force,
           'CoM_world':com,'FL_center':p['wheels']['front_left_ankle']['center_w_m'],
           'body_position':p['base']['position_w_m'],
           'raw':a['raw_policy_action_full12'],'mask':a['phase_mask_full12'],
           'N':n['nominal_full12'],'mapped_N':a['native_drive_target_full12'],
           'residual':n['projected_residual_full12'],
           'final':[p['joints'][name+'_'+joint]['command_deg'] for name in NAMES for joint in ('hip','knee')]
                    +[p['wheels'][name+'_ankle']['command_rad_s'] for name in NAMES],
           'dispatch_verified':a['verified'],'same_tick_decision_diagnostic':decisions.get(tick)}
        previous=physical.get(tick-1)
        r['RR_bottom_velocity_finite_difference_m_s']=None if previous is None else (
            wheel['bottom_w_m'][2]-previous['wheels']['rear_right_ankle']['bottom_w_m'][2])*120
        data.append(r)
    qwindow=[r for r in data if qualified<=r['tick']<=cross]
    transitions=[r for r in bounded_lines(source/'stage_transition_evidence.jsonl',0,None)
                 if start<=r['physics_tick']<=end]
    keyticks={start,qualified-1,qualified,qualified+1,cross-1,cross,cross+1}
    for r in transitions:keyticks.add(r['physics_tick'])
    contact_segments={kind:segments(qwindow,lambda r,kind=kind:r['RR_surface']==kind and r['RR_obstacle'])
        for kind in ('FRONT_WALL','OBSTACLE_AMBIGUOUS','TOP')}
    for intervals in contact_segments.values():
        for interval in intervals:keyticks.update((interval['start'],interval['end']))
    for tick in list(keyticks):
        keyticks.add((tick//8)*8);keyticks.add((tick//8+1)*8)
    results[label]={'source':str(source),'bounds':[start,end],'q_to_cross_bounds':[qualified,cross],
        'source_label_preserved':'original sealed SUCCESS; no relabeling by this diagnostic',
        'phase_intervals':{phase:segments(data,lambda r,p=phase:r['phase']==p) for phase in ('P06','P07','P08','P09','P10')},
        'Q_to_cross_surface_counts':{kind:sum(r['RR_surface']==kind for r in qwindow)
            for kind in ('NONE','FRONT_WALL','OBSTACLE_AMBIGUOUS','TOP')},
        'Q_to_cross_air_segments':segments(qwindow,lambda r:r['RR_air']),
        'Q_to_cross_ground_segments':segments(qwindow,lambda r:r['RR_ground']),
        'Q_to_cross_contact_segments':contact_segments,
        'Q_to_cross_max_AIR_gap_m':max((r['RR_gap'] for r in qwindow if r['RR_air']),default=None),
        'Q_to_cross_max_raw':max(abs(x) for r in qwindow for x in r['raw']),
        'all12_mask_one':all(r['mask']==[1]*12 for r in data),
        'all_dispatch_verified':all(r['dispatch_verified'] for r in data),
        'transitions':transitions,'key_rows':[r for r in data if r['tick'] in keyticks],
        'rows':data}
out=ROOT/'outputs/diagnostics_v1/residual_RR_diagnosis.json'
with out.open('x',encoding='utf8') as stream:
    json.dump({'schema':'wlr50_clean.focused_RR_contact_audit.v1','runs':results,
        'no_policy_forward':True,'no_physics':True,'no_original_labels_changed':True,
        'contact_classification':'same existing pure physical_contact_surface applied to saved exact pairs; no re-evaluation of task success',
        'current_lift_and_transfer_cadence':'original decision-end diagnostic only; not interpolated to other physics ticks'},stream,ensure_ascii=False,indent=2)
for label,r in results.items():
    print(json.dumps({'run':label,**{k:v for k,v in r.items() if k not in ('rows','key_rows','transitions')}},indent=2))
