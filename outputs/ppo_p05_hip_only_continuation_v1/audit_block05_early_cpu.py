"""One bounded stdlib-only snapshot; no imports/replay of live runtime."""
import json
from pathlib import Path
from collections import Counter

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/ppo_p05_hip_only_continuation_v1/train/20260922T0926100383683Z_g5fd88852bf20_344b6773d9854b41948d1632e1f0fc3f'
CUTOFF=208653
episodes={}
for line in (RUN/'residual_and_projection_audit.jsonl').open(encoding='utf-8'):
    try: row=json.loads(line)
    except json.JSONDecodeError: break
    if row['global_policy_decision']>CUTOFF: break
    a=row['applied_audit']; t=a['semantic_task']; ev=t['physical_evaluator']; n=a['actuator_target_effect_audit']
    if a['decision_count']==1: episode=len(episodes); episodes[episode]=[]
    fr=ev['current_legs']['FR']; h=n['policy_headroom_evidence']
    channels=n['tracking_reference_evidence']['channels']
    actual=[v['nominal_deg']-v['current_actual_canonical_error_deg'] for v in channels]
    legs={leg:{k:v.get(k) for k in ('air','support','ground_contact','top_contact','bearing_force_n','clearance_m','front_distance_m')}
          for leg,v in ev['current_legs'].items()}
    episodes[episode].append(dict(global_decision=row['global_policy_decision'],tick=a['physics_tick'],
        time=a['sim_time_s'],phase=a['end_phase_id'],terminal=row['terminal'],
        FR_front_mm=fr['front_distance_m']*1000,FR_gap_mm=fr['clearance_m']*1000,
        Q=ev['history']['active_lift']['FR'],C=ev['history']['front_edge_crossed']['FR'],P=ev['history']['placed']['FR'],
        FR_ground=fr['ground_contact'],FR_air=fr['air'],entry_valid=t['entry_valid'],completion=t['completion_values'],
        body_forward=ev['goal_features']['body_forward_m'],legs=legs,
        N=a['nominal_action_full12'],mapper=n['native_drive_target_full12'],baseline=h['baseline_native_plus_controller_full12'],
        effective=h['effective_policy_residual_full12'],final=a['actual_drive_target_full12'],actual_pre=actual,
        wheel_actual=ev['measured_wheel_velocity_rad_s'],mask=n['phase_mask_full12'],
        native_verified=n['verified'] and n['actual_mapping_matches_dispatch'],
        verified_tick_count=a['actuator_target_effect_audit_summary']['verified_tick_count'],
        physical_ticks=a['physics_ticks'],all_ticks_verified=a['actuator_target_effect_audit_summary']['all_ticks_verified'],
        no_state_writes=a['no_in_episode_state_writes_verified'],clip=h['clipped_servo_indices'],
        assist_owner=n['capture_assist_evidence']['owner_indices'],
        physical_valid=ev['valid'],evidence=ev['physical_evidence_status'],physical_failure=ev['termination_reason'],
        events=ev['history']['lift_attempt_events'],local_timeout=t['local_timeout'],
        stage_age=t['stage_age_s'],termination_source=t['termination_source']))
    if row['global_policy_decision']==CUTOFF: break

result={'snapshot_cutoff_global_policy_decision':CUTOFF,'episodes':{}}
for index,rows in episodes.items():
    first_clear=next((r for r in rows if r['phase']=='P02' and r['completion'].get('clear_FR')==1),None)
    first_final_knee=next((r for r in rows if r['N'][3]==45.9),None)
    milestones=[min(rows,key=lambda r:abs(r['tick']-tick)) for tick in ((16,48,288,904) if index==2 else (16,48,1592,1600))]
    for row in (first_clear,first_final_knee,rows[-1]):
        if row and row not in milestones: milestones.append(row)
    milestones.sort(key=lambda r:r['tick'])
    wheel_changes=[]; last=None
    for r in rows:
        wheels=r['N'][8:]
        if wheels!=last: wheel_changes.append({'tick':r['tick'],'N_wheels':wheels}); last=wheels
    result['episodes'][index]=dict(decisions=len(rows),last_global_decision=rows[-1]['global_decision'],
        terminal=rows[-1]['terminal'],all_native_verified=all(r['native_verified'] and r['all_ticks_verified'] for r in rows),
        all12_masks_open=all(r['mask']==[1]*12 for r in rows),no_assist=all(not r['assist_owner'] for r in rows),
        no_headroom_clips=all(not r['clip'] for r in rows),no_state_writes=all(r['no_state_writes'] for r in rows),
        physical_ticks=sum(r['physical_ticks'] for r in rows),verified_ticks=sum(r['verified_tick_count'] for r in rows),
        all_physical_valid=all(r['physical_valid'] and r['physical_failure'] is None for r in rows),
        evidence_status_counts=dict(Counter(r['evidence'] for r in rows)),
        FR_lift_events=[r for r in rows[-1]['events'] if r['leg']=='FR'],
        terminal_local_timeout=rows[-1]['local_timeout'],last_stage_age=rows[-1]['stage_age'],
        terminal_source=rows[-1]['termination_source'],last_completion=rows[-1]['completion'],
        closest_FR_front_endpoint={k:max(rows,key=lambda r:r['FR_front_mm'])[k] for k in ('global_decision','tick','FR_front_mm','FR_gap_mm')},
        max_FR_gap_mm=max(r['FR_gap_mm'] for r in rows),
        max_FR_front_mm=max(r['FR_front_mm'] for r in rows),first_clear_tick=first_clear['tick'] if first_clear else None,
        first_source_final_knee_tick=first_final_knee['tick'] if first_final_knee else None,
        FR_ground_endpoints=sum(r['FR_ground'] for r in rows),FR_cross_endpoints=sum(r['C'] for r in rows),
        N_wheel_change_endpoints=wheel_changes,
        post1600_N_fourwheel_positive_endpoints=sum(r['tick']>=1600 and min(r['N'][8:])>0 for r in rows),
        milestones=milestones)
    for r in result['episodes'][index]['milestones']:
        for key in ('mask','completion','native_verified','verified_tick_count','physical_ticks',
                    'all_ticks_verified','no_state_writes','physical_valid','physical_failure','events','local_timeout','stage_age','termination_source'):
            r.pop(key,None)
print(json.dumps(result,ensure_ascii=False,separators=(',',':'),allow_nan=False))
