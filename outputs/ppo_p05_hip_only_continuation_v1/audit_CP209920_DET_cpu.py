"""Bounded read-only sealed CP209920 evidence extraction (stdlib only)."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'runs/ppo_p05_hip_only_continuation_v1/video_eval/validation/20260922T1004553890489Z_g5fd88852bf20_a9c7b7134b6c4b448234ac34c0a3dc08/source'
rows=[json.loads(s) for s in (SRC/'video_policy_decisions.jsonl').open(encoding='utf-8')]
ns={}
wheel_changes=[]
last=None
for line in (SRC/'native_tick_audit.jsonl').open(encoding='utf-8'):
    r=json.loads(line); ns[r['episode_physics_tick']]=r
    if r['nominal_full12'][8:]!=last:
        wheel_changes.append({'tick':r['episode_physics_tick'],'N_wheels':r['nominal_full12'][8:]})
        last=r['nominal_full12'][8:]
obs={}
for line in (SRC/'physical_observations.jsonl').open(encoding='utf-8'):
    r=json.loads(line); obs[r['physics_tick']]=r
last=rows[-1]['step_info']; task=last['semantic_task']; ev=task['physical_evaluator']
ep=[(r['end_tick'],r['step_info']['semantic_task']['physical_evaluator']) for r in rows]
peak=max(ep,key=lambda p:p[1]['current_legs']['FR']['clearance_m'])
raw_peak=max(obs.values(),key=lambda o:o['wheels']['front_right_ankle']['bottom_w_m'][2]-o['obstacle']['top_z_m'])
legs=('FL','FR','RL','RR'); names=('front_left','front_right','rear_left','rear_right')
def summarize(tick):
    r=ns[tick]; n=r['native_audit']; h=n['policy_headroom_evidence']; o=obs[tick]
    nearest=min(ep,key=lambda p:abs(p[0]-tick))
    return dict(tick=tick,time=o['simulation_time_s'],phase=r['source_phase_id'],
        N=r['nominal_full12'],mapper=n['native_drive_target_full12'],baseline=h['baseline_native_plus_controller_full12'],
        residual=r['projected_residual_full12'],raw=n['raw_policy_action_full12'],
        effective=h['effective_policy_residual_full12'],final=o['commanded_full12'],actual_post=o['actual_full12'],
        wheel_joint_actual=[o['wheels'][name+'_ankle']['velocity_rad_s'] for name in names],
        wheel_centers=[o['wheels'][name+'_ankle']['center_w_m'] for name in names],
        wheel_bottoms=[o['wheels'][name+'_ankle']['bottom_w_m'] for name in names],
        contacts={leg:{'class':o['contacts'][name+'_wheel']['contact_class'],
          'ground_N':o['contacts'][name+'_wheel']['ground']['normal_force_n'],
          'obstacle_N':o['contacts'][name+'_wheel']['obstacle']['normal_force_n']} for leg,name in zip(legs,names)},
        obstacle=o['obstacle'],base=o['base']['position_w_m'],
        nearest_endpoint_tick=nearest[0],nearest_FR=nearest[1]['current_legs']['FR'],
        same_tick_verified=n['verified'],mapping_verified=n['actual_mapping_matches_dispatch'],
        native_physical_targets=n['actual_native_targets'],last_target_source=n['actual_target_source'],
        clip=h['clipped_servo_indices'],final_clip=n['capture_assist_evidence']['final_slew_or_clamp_indices'])
selected=sorted(set([16,22,40,peak[0],128,192,196,200,288,904,1592,1593,1600,1864]+[r['tick'] for r in wheel_changes]))
print(json.dumps(dict(decisions=len(rows),ticks=len(ns),physical_observations=len(obs),
    source_phase_change=[{'tick':t,'phase':r['source_phase_id']} for t,r in ns.items() if t==1 or r['source_phase_id']!=ns[t-1]['source_phase_id']],
    wheel_changes=wheel_changes,
    all_native_verified=all(r['native_audit']['verified'] and r['native_audit']['actual_mapping_matches_dispatch'] for r in ns.values()),
    all_masks_open=all(r['native_audit']['phase_mask_full12']==[1]*12 for r in ns.values()),
    all_no_assist=all(not r['native_audit']['capture_assist_evidence']['owner_indices'] for r in ns.values()),
    native_last_writer_sources=list(set(r['native_audit']['actual_target_source'] for r in ns.values())),
    first_FR_N_endpoint=next(t for t,r in ns.items() if r['nominal_full12'][3]==45.9),
    raw_peak={'tick':raw_peak['physics_tick'],'gap_m':raw_peak['wheels']['front_right_ankle']['bottom_w_m'][2]-raw_peak['obstacle']['top_z_m']},
    nominal_diagnostics=task['nominal_provider_diagnostics'],
    final_current_legs=ev['current_legs'],
    clips=sum(bool(r['native_audit']['policy_headroom_evidence']['clipped_servo_indices']) for r in ns.values()),
    all_physically_valid=all(e['valid'] and e['termination_reason'] is None for _,e in ep),
    max_gap_endpoint={'tick':peak[0],'gap_m':peak[1]['current_legs']['FR']['clearance_m']},
    first_clear_endpoint=next((t for t,e in ep if e['current_legs']['FR']['clearance_m']>=.015),None),
    all_physical_finite=all(o['all_finite'] for o in obs.values()),
    events=ev['history']['lift_attempt_events'],last_history=ev['history'],
    terminal={k:last.get(k) for k in ('termination_reason','task_success','full_task_success','no_in_episode_state_writes_verified')},
    terminal_task={k:task.get(k) for k in ('termination_source','entry_valid','entry_reasons','completion_values','stage_age_s','local_timeout','source_partial_order')},
    milestones=[summarize(t) for t in selected]),ensure_ascii=False,separators=(',',':')))
