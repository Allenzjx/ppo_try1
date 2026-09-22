"""One sealed training log; compact 2-window physical summary, CPU only."""
from collections import Counter
import json
import math
from pathlib import Path
from rr_probe_readonly import lines, servo_limits_deg

OUT=Path(__file__).resolve().parent; ROOT=OUT.parents[1]
RUN=ROOT/'runs/ppo_task_conditioned_hip_wheel_v1/train/20260921T1200165967039Z_g97ecd305afb5_4ee14f50dcfc4e2d9d5787e6fb08ecbf'
LEGS=('FL','FR','RL','RR')
NAMES=('front_left_hip','front_left_knee','front_right_hip','front_right_knee','rear_left_hip','rear_left_knee','rear_right_hip','rear_right_knee')

def agg(v):
    v=list(v); return dict(min=min(v),max=max(v),mean=sum(v)/len(v))

def compact(r):
    s=r['applied_audit']; ev=s['semantic_task']['physical_evaluator']; native=s['actuator_target_effect_audit']
    h=native['policy_headroom_evidence']; positions={}; margins={}
    for role in ev['transfer_roles'].values():
        for name,m in role['receiver_workspace_state']['joint_range_margin_deg'].items():
            lo,hi=servo_limits_deg(name); q=lo+m['negative_deg']
            assert abs(hi-q-m['positive_deg'])<1e-8
            if name in positions: assert abs(positions[name]-q)<1e-8
            positions[name]=q; margins[name]=min(m.values())
    assert set(positions)==set(NAMES)
    geometry=s['reward_breakdown']['task_space_quality_sample_audit']
    return dict(tick=s['physics_tick'],time=s['sim_time_s'],phase=s['phase_id'],end_phase=s['end_phase_id'],
        global_decision=r['global_policy_decision'],terminal=r['terminal'],termination=s.get('termination_reason'),
        body=ev['goal_features']['body_forward_m'],legs=ev['current_legs'],events=s['semantic_task']['history']['event_ticks'],
        positions=positions,margins=margins,N=s['nominal_action_full12'],mapped=h['baseline_native_plus_controller_full12'],
        residual=h['effective_policy_residual_full12'],target=s['actual_drive_target_full12'],qd=ev['measured_wheel_velocity_rad_s'],
        mask=native['phase_mask_full12'],verified=native['verified'] and native['setter_dispatch_targets_equal'] and native['actual_mapping_matches_dispatch'],
        ticks_verified=all(t['verified'] for t in s['actuator_target_effect_audit_ticks']),
        tick_count=len(s['actuator_target_effect_audit_ticks']),stop=s['semantic_task']['nominal_provider_diagnostics']['final_stop_owner']['active'],
        owner_source=native['actual_target_source'],geometry=geometry,
        body_min_z=min((g['body_collider_minimum_w_m'][2] for g in geometry if g.get('eligible') and g.get('valid')),default=None))

groups=[]; previous=None; episode=0; final_rows=[]
for r in lines(RUN/'residual_and_projection_audit.jsonl'):
    s=r['applied_audit']; tick=s['physics_tick']
    if previous is not None and tick<previous['tick']: episode+=1
    c=compact(r)
    if c['phase']=='P06':
        if previous is None or previous['phase']!='P06':
            assert previous is not None and previous['end_phase']=='P06'
            groups.append(dict(episode_index=episode,anchor=previous,actions=[]))
        groups[-1]['actions'].append(c)
    if r['terminal']: final_rows.append(dict(episode_index=episode,tick=tick,termination=c['termination']))
    previous=c
assert len(groups)==2 and sum(len(g['actions']) for g in groups)==443
results=[]
for group in groups:
    actions=group['actions']; samples=[group['anchor']]+actions
    times=[s['time'] for s in samples]; dts=[b-a for a,b in zip(times,times[1:])]
    assert all(dt>0 for dt in dts)
    duration=times[-1]-times[0]; assert abs(sum(dts)-duration)<1e-10
    contacts={}
    for leg in ('FL','FR'):
        cs=[s['legs'][leg] for s in samples]
        modes=['AIR' if c['air'] else 'TOP_BEARING' if c['top_surface_contact'] and c['bearing_verified'] and c['support'] else 'GROUND' if c['ground_contact'] else 'OTHER' for c in cs]
        contacts[leg]=dict(endpoint_samples=len(cs),mode_counts=dict(Counter(modes)),
            mode_change_count=sum(a!=b for a,b in zip(modes,modes[1:])),
            first_AIR_saved_tick=next((s['tick'] for s in samples if s['legs'][leg]['air']),None),
            gap_m=agg(c['clearance_m'] for c in cs),final_gap_m=cs[-1]['clearance_m'],
            final_bearing_n=cs[-1]['bearing_force_n'],final_mode=modes[-1])
    wheels={}
    for i,leg in enumerate(LEGS):
        target=[s['target'][8+i] for s in actions]; qd=[s['qd'][i] for s in actions]
        wheels[leg]=dict(N_rad_s=agg(s['N'][8+i] for s in actions),mapped_rad_s=agg(s['mapped'][8+i] for s in actions),
            effective_residual_rad_s=agg(s['residual'][8+i] for s in actions),final_target_rad_s=agg(target),actual_canonical_qd_rad_s=agg(qd),
            target_negative_endpoints=sum(x<0 for x in target),actual_negative_endpoints=sum(x<0 for x in qd),
            endpoint_tracking_RMS_rad_s=math.sqrt(sum((a-b)**2*dt for a,b,dt in zip(qd,target,dts))/duration),
            final_minus_mapped_effective_max_abs=max(abs(s['target'][8+i]-s['mapped'][8+i]-s['residual'][8+i]) for s in actions))
    joints={}
    for name in NAMES:
        values=[s['positions'][name] for s in samples]; dq=[b-a for a,b in zip(values,values[1:])]
        signs=[1 if x>0 else -1 for x in dq if abs(x)>=.05]
        joints[name]=dict(min_deg=min(values),max_deg=max(values),range_deg=max(values)-min(values),
            net_deg=values[-1]-values[0],sampled_TV_deg=sum(map(abs,dq)),
            direction_reversals_over_0_05_deg=sum(a!=b for a,b in zip(signs,signs[1:])),
            sampled_derivative_RMS_deg_s=math.sqrt(sum(delta*delta/dt for delta,dt in zip(dq,dts))/duration),
            minimum_physical_limit_margin_deg=min(s['margins'][name] for s in samples))
    results.append(dict(episode_index=group['episode_index'],FL_capture_tick=actions[-1]['events']['placed'].get('FL'),
        ticks=[samples[0]['tick'],samples[-1]['tick']],duration_s=duration,policy_actions=len(actions),endpoint_samples=len(samples),
        actual_dt_range_s=[min(dts),max(dts)],global_action_decisions=[actions[0]['global_decision'],actions[-1]['global_decision']],
        completion='P06 event complete; this is not full task success' if actions[-1]['end_phase']=='P07' else 'P06 incomplete at block budget, nonterminal',
        body_net_forward_m=samples[-1]['body']-samples[0]['body'],
        body_collider_min_world_z_m=min(s['body_min_z'] for s in actions if s['body_min_z'] is not None),
        current_front_support=contacts,wheels=wheels,actual_joints=joints,
        N_servo_unique_vectors=sorted(set(tuple(s['N'][:8]) for s in actions)),
        dispatch=dict(full12_masks_all_one=all(s['mask']==[1]*12 for s in actions),all_endpoints_verified=all(s['verified'] for s in actions),
            all_physics_ticks_verified=all(s['ticks_verified'] for s in actions),physics_ticks=sum(s['tick_count'] for s in actions),
            explicit_final_stop_count=sum(s['stop'] for s in actions),last_target_sources=sorted(set(s['owner_source'] for s in actions))),
        raw_native_actual_qd=None,hip_mount_world_z=None,full_120Hz_joint_TV_or_derivative_RMS=None,traction_power_fraction=None))
B=json.loads((OUT/'currentB_event_windows.json').read_text(encoding='utf-8'))['windows']['P06_rear_approach']
old=json.loads((OUT/'block02_P06_wheel_window.json').read_text(encoding='utf-8'))
reference=dict(successful_N=dict(artifact='currentB_event_windows.json',ticks=[B['start_tick'],B['observed_end_tick']],
    duration_s=B['physical']['duration_s'],body_net_forward_m=B['physical']['body_forward_displacement_m'],
    FL_TOP_bearing_endpoints=B['current_support_samples']['FL']['top_verified_bearing_samples'],
    FL_endpoint_samples=len(B['current_support_samples']['sample_ticks']),
    available_joint_range_deg={name:v['actual_deg']['maximum']-v['actual_deg']['minimum'] for name,v in B['physical']['joints'].items()},
    joint_TV_and_derivative_RMS=None),
    old_training_P06_fixed_window=dict(artifact='block02_P06_wheel_window.json',ticks=[2800,4000],duration_s=10.,
        body_net_forward_m=old['body_and_progress']['body_forward_displacement_m'],
        FL_TOP_bearing_endpoints=old['current_contact_and_wheel_end_motion']['FL']['actual_TOP_verified_bearing_count'],
        FL_endpoint_samples=old['current_contact_and_wheel_end_motion']['FL']['samples'],
        joint_sampled_TV_deg={name:v['sampled_total_variation'] for name,v in old['linkage']['measured_positions_deg'].items()}))
result=dict(schema='block11.P06_two_compact_windows.v1',source_run=str(RUN),
    classification='successful_nominal P04 initialized suffix, stochastic training with updates; not fixed CP or natural P01 success',
    actual_P06_policy_actions=443,windows=results,episode_terminals=final_rows,
    last_block_endpoint=dict(tick=previous['tick'],phase=previous['end_phase'],terminal=previous['terminal']),references_reused=reference,
    semantics=dict(actual_joint_source='measured joint_range_margin_deg plus physical lower limit, independently checked against upper margin; not targets',
        joint_derivative_RMS='sqrt(sum((delta_q/dt)^2*dt)/sum(dt)), actual recorded 15Hz dt; NOT RMS/T',
        TV='sampled 15Hz total variation, lower-resolution path statistic; not 120Hz work, periodic gait or pure waste',
        reversals='sign changes after excluding individual absolute delta_q below0.05deg; not gait-cycle count',
        contacts='current evaluator endpoints, not contact durations; placed history is not load-bearing',
        wheels='canonical actual joint qd, target distinct; no torque/slip/intervention evidence of traction fraction',
        comparison='references have differing durations/state/CP; compare stated ranges only, no stable ranking or imaginary N TV',
        missing='raw native actual qd, hip USD heights and full120Hz joint derivatives remain null'),
    no_new_physics_optimizer_or_production_edit=True)
with (OUT/'block11_P06_compact_readonly.json').open('x',encoding='utf-8') as stream:
    json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
for w in results:
    print('WINDOW',json.dumps({k:w[k] for k in ('episode_index','FL_capture_tick','ticks','duration_s','policy_actions','completion','body_net_forward_m','body_collider_min_world_z_m','current_front_support','N_servo_unique_vectors','dispatch')}))
    print('WHEELS',json.dumps(w['wheels']))
    print('JOINTS',json.dumps(w['actual_joints']))
print('REFERENCES',json.dumps(reference))
