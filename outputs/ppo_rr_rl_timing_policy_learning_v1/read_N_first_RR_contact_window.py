"""JSON-only binary-seek to N_ref +/-0.3 s RR capture window; stdout only."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
N=ROOT/'runs/ppo_task_conditioned_hip_wheel_v1/video_eval/prior_B/20260921T0701155546642Z_gee5a9651591d_64985c63292d4a9b966716c3a54659a1/source'
lo_tick,hi_tick=6110,6191
inventory={}

def window(name,key):
    path=N/name; stat=path.stat(); probes=0; rows=[]
    with path.open('rb') as stream:
        low,high=0,stat.st_size
        while high-low>65536:
            mid=(low+high)//2;stream.seek(mid);stream.readline();raw=stream.readline()
            if not raw:high=mid;continue
            row=json.loads(raw);probes+=1
            if row[key]<lo_tick:low=stream.tell()
            else:high=mid
        stream.seek(low)
        scanned=0
        for raw in stream:
            scanned+=1;row=json.loads(raw)
            if row[key]>hi_tick:break
            if row[key]>=lo_tick:rows.append(row)
            assert scanned<1000
    after=path.stat();assert (stat.st_size,stat.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    inventory[name]=dict(binary_probe_rows=probes,local_scan_rows=scanned,selected_rows=len(rows))
    return rows

physics=window('physical_observations.jsonl','physics_tick')
native=window('native_tick_audit.jsonl','episode_physics_tick')
policy=window('video_policy_decisions.jsonl','end_tick')
assert len(physics)==len(native)==82
nd={r['episode_physics_tick']:r for r in native}

def contact(row,prefix):
    pair=row['contacts'][prefix+'_wheel'];g=pair['ground'];o=pair['obstacle']
    return dict(raw_class=pair['contact_class'],ground_active=g['active'],ground_force_n=g['normal_force_n'],
        obstacle_active=o['active'],obstacle_force_n=o['normal_force_n'],obstacle_force_w_n=o['force_w_n'],
        consecutive_obstacle_active=o['consecutive_active_ticks'],obstacle_point_w_m=o['contact_point_w_m'])

physical=[]
for row in physics:
    tick=row['physics_tick']; nr=nd[tick]
    physical.append(dict(tick=tick,t=row['simulation_time_s'],RR_gap_m=row['wheels']['rear_right_ankle']['bottom_w_m'][2]-row['obstacle']['top_z_m'],
        RR_front_m=row['wheels']['rear_right_ankle']['center_w_m'][0]-row['obstacle']['front_x_m'],
        RR=contact(row,'rear_right'),FL=contact(row,'front_left'),RL=contact(row,'rear_left'),FR=contact(row,'front_right'),
        source_phase=nr['source_phase_id'],N_RR=nr['nominal_full12'][6:8],N_FL_wheel=nr['nominal_full12'][8],
        RR_final=row['commanded_full12'][6:8],RR_actual=row['actual_full12'][6:8]))
decisions=[]
for row in policy:
    a=row['step_info'];s=a['semantic_task'];legs=s['physical_evaluator']['current_legs']
    keys=['clearance_m','top_contact','air','support','bearing_force_n','bearing_verified','consecutive_top_samples','ground_contact','top_surface_contact','contact_surface','placed_on_top']
    decisions.append(dict(tick=row['end_tick'],t=a['sim_time_s'],phase=row['request_phase'],
        legs={leg:{k:v.get(k) for k in keys} for leg,v in legs.items()},event_ticks=s['history']['event_ticks']))
post=[r for r in physical if r['tick']>=6155]
summary=dict(RR_post_capture_obstacle_active_count=sum(r['RR']['obstacle_active'] for r in post),post_capture_rows=len(post),
    RR_post_capture_obstacle_force_range_n=[min(r['RR']['obstacle_force_n'] for r in post),max(r['RR']['obstacle_force_n'] for r in post)],
    RR_post_capture_gap_range_m=[min(r['RR_gap_m'] for r in post),max(r['RR_gap_m'] for r in post)],
    first_RR_obstacle_active_tick_in_window=next((r['tick'] for r in physical if r['RR']['obstacle_active']),None),
    first_late_source_tick=next((r['tick'] for r in physical if r['N_FL_wheel']==-1.07),None),
    first_P10_knee_change_tick=next((r['tick'] for r in physical if r['N_RR'][1]!=-37.8 and r['tick']>=6155),None))
selected_ticks={6110,6111,6112,6113,6119,6152,6153,6154,6155,6156,6160,6161,6168,6175,6176,6184,6191}
result=dict(schema='readonly.accepted_N_first_RR_contact_window.v1',source=str(N),tick_window=[lo_tick,hi_tick],
    read_inventory=inventory,scope='sealed N_ref, non-paired old control; zero physical/model/PPO execution; raw obstacle force is not automatically evaluator TOP',
    summary=summary,selected_physics=[r for r in physical if r['tick'] in selected_ticks],decision_evaluator=decisions,
    all_RR_force_gap_trace=[{k:r[k] for k in ['tick','RR_gap_m','RR_front_m','N_RR','N_FL_wheel']}|
        {'RR_obstacle_force_n':r['RR']['obstacle_force_n'],'RR_obstacle_active':r['RR']['obstacle_active'],
         'RR_consecutive_obstacle_active':r['RR']['consecutive_obstacle_active']} for r in physical])
print(json.dumps(result,indent=2))
