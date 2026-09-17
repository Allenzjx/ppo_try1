"""Read only the first 33 complete persisted rows; no live actor/storage access."""
import itertools
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from wlr50_clean.infrastructure.command_batch import SERVO_ORDER, SERVO_COMMAND_SIGN

RUN=ROOT/'runs/ppo_task_first_recovery_v1/train/20260916T0628168580885Z_gfc14a68c037a_fe364dae15434293be866a2fe834d015'
with (RUN/'residual_and_projection_audit.jsonl').open(encoding='utf8') as stream:
    rows=[json.loads(line) for line in itertools.islice(stream,33)]
assert len(rows)==33 and rows[-1]['terminal'] and all(not r['terminal'] for r in rows[:-1])
assert [r['global_policy_decision'] for r in rows]==list(range(177281,177314))
last=rows[-1]; audit=last['applied_audit']; evaluation=audit['semantic_task']['physical_evaluator']
schema=json.loads((ROOT/'configs/ppo_task_first_recovery_v1/observation_schema.json').read_text())
terminal=last['terminal_observation']['policy'][0]
decoded={}; offset=0
for group in schema['feature_groups']:
    scale=group['scale']; n=group['size']
    scale=scale if isinstance(scale,list) else [scale]*n
    decoded[group['name']]=[x*s for x,s in zip(terminal[offset:offset+n],scale)]
    offset+=n
assert offset==len(terminal)==372
com_world=evaluation['transfer_roles']['RR']['transfer_direction_context']['mass_weighted_com_position_w_m']
base_z=com_world[2]-decoded['com_position_relative_base'][2]
snapshots=[]
for r in rows:
    a=r['applied_audit']; e=a['semantic_task']['physical_evaluator']
    tr=a['actuator_target_effect_audit']['tracking_reference_evidence']
    actual=[(math.degrees(q)-standing)/SERVO_COMMAND_SIGN[name]
        for name,q,standing in zip(SERVO_ORDER,tr['actual_measured_physical_rad'],tr['standing_pose_deg'])]
    snapshots.append({'decision':r['global_policy_decision'],'tick':a['physics_tick'],
        'request_phase':a['phase_id'],'end_phase':a['end_phase_id'],
        'raw':r['raw_policy_action_full12'],'stored_old_mean':r['old_distribution_mean_full12'],
        'stored_effective_sigma':r['old_distribution_std_full12'],
        'stored_innovation':[x-m for x,m in zip(r['raw_policy_action_full12'],r['old_distribution_mean_full12'])],
        'nominal':a['nominal_action_full12'],'projected_residual':a['projected_residual_full12'],
        'final':a['actual_drive_target_full12'],'actual_servo_before_last_dispatch_deg':actual,
        'actual_servo_sample_episode_tick':a['physics_tick']-1,
        'actual_wheel_rad_s':e['measured_wheel_velocity_rad_s'],
        'supports':[leg for leg,v in e['current_legs'].items() if v['support']],
        'body_minimum_world_z_m':e['body_traversal_geometry']['minimum_w_m'][2],
        'linear_speed_m_s':e['goal_features']['body_linear_speed_m_s'],
        'angular_speed_rad_s':e['goal_features']['body_angular_speed_rad_s'],
        'RR_front_distance_m':e['current_legs']['RR']['front_distance_m'],
        'RR_clearance_above_top_m':e['current_legs']['RR']['clearance_m'],
        'current_RR_lift_valid':e['current_legs']['RR'].get('current_lift_valid'),
        'transitions':a['stage_transition_evidence'],'terminal':r['terminal']})
result={'schema':'wlr50_clean.quarter_first33_persisted_failure.v1','source_run':str(RUN),
    'read_first_complete_rows':33,'no_live_memory_rollout_or_actor_read':True,
    'no_GAE_returns_advantages_or_network_forward_recomputed':True,
    'stored_distribution_available_in_original_audit':True,
    'request_phase_counts':{p:sum(s['request_phase']==p for s in snapshots) for p in ('P06','P07','P08','P09')},
    'raw_absmax':max(abs(v) for r in rows for v in r['raw_policy_action_full12']),
    'decisions_any_raw_abs_gt3':sum(any(abs(v)>3 for v in r['raw_policy_action_full12']) for r in rows),
    'effective_sigma_absmax':max(r['old_distribution_std_full12'][j] for r in rows for j in range(12)),
    'terminal':{'reason':audit['termination_reason'],'source':evaluation['termination_source'],
        'evaluator_reason':evaluation['reason'],'base_z_decoded_from_saved_float32_observation_m':base_z,
        'base_z_decode':'saved exact CoM world z minus schema-denormalized saved terminal CoM-relative-base z',
        'chassis_rpy_rad_decoded':decoded['chassis_rpy_rad'],
        'chassis_rpy_deg_decoded':[math.degrees(v) for v in decoded['chassis_rpy_rad']],
        'projected_gravity_decoded':decoded['projected_gravity_chassis'],
        'base_height_fall_guard':base_z<.015,
        'gravity_fall_guard':decoded['projected_gravity_chassis'][2]>-.30,
        'explosion_values_triggered':base_z>1 or evaluation['goal_features']['body_linear_speed_m_s']>5 or evaluation['goal_features']['body_angular_speed_rad_s']>20,
        'physical_body_minimum_world_z_m':evaluation['body_traversal_geometry']['minimum_w_m'][2],
        'history':evaluation['history'],'reward_breakdown':audit['reward_breakdown'],
        'raw_guard_values_not_persisted':True,'finite_float32_decode_not_new_simulation':True},
    'snapshots':snapshots}
out=ROOT/'outputs/ppo_task_first_recovery_v1/quarter512_first33_FALL_diagnosis.json'
with out.open('x',encoding='utf8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2)
print(json.dumps({'path':str(out),'counts':result['request_phase_counts'],
    'raw_absmax':result['raw_absmax'],'raw_gt3':result['decisions_any_raw_abs_gt3'],
    'sigma_max':result['effective_sigma_absmax'],'terminal':{k:v for k,v in result['terminal'].items() if k not in ('history','reward_breakdown')},
    'samples':[{'tick':s['tick'],'FLhip_target':s['final'][0],'FLhip_actual':s['actual_servo_before_last_dispatch_deg'][0],
        'FRk_target':s['final'][3],'FRk_actual':s['actual_servo_before_last_dispatch_deg'][3],
        'wheel_target':s['final'][8:],'wheel_actual':s['actual_wheel_rad_s'],'supports':s['supports'],
        'body_min_z':s['body_minimum_world_z_m']} for s in snapshots if s['tick'] in (2688,2704,2720,2744,2752,2760,2784,2880,2944)]},indent=2))
