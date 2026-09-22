"""One bounded CPU-only read of the first <=64 complete learner records."""
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
RUN = ROOT / 'runs/ppo_task_conditioned_hip_wheel_v1/train/20260921T1314224958148Z_g97ecd305afb5_728292550a8b4d38a844358c483ac74c'
PATH = RUN / 'residual_and_projection_audit.jsonl'

def compact_leg(c):
    return {k: c.get(k) for k in ('contact_mode', 'air', 'ground_contact', 'obstacle_pair_active',
        'top_surface_contact', 'bearing_verified', 'support', 'bearing_force_n', 'within_top_xy',
        'front_distance_m', 'clearance_m', 'current_lift_valid', 'active_attempt')}

def top(c):
    return c['top_surface_contact'] and c['bearing_verified'] and c['support']

def stats(values):
    v = list(values)
    return dict(min=min(v), mean=sum(v)/len(v), max=max(v))

def bounded_tail(path):
    with path.open('rb') as f:
        f.seek(0, 2); end = f.tell(); start = max(0, end-262144)
        f.seek(start); data = f.read(end-start)
    complete = data.splitlines(keepends=True)
    if start: complete = complete[1:]
    return [json.loads(line) for line in complete[-4:] if line.endswith(b'\n')]

prefix = bounded_tail(RUN/'prefix_evidence.jsonl')
handoff = next(r['start'] for r in prefix if r['kind']=='policy_credit_start')
prefix_result = next(r for r in prefix if r['kind']=='checkpoint_prefix_result')
assert handoff['physics_tick']==6176 and handoff['actual_phase']=='P12'
rows=[]; digest=hashlib.sha256(); consumed=0
with PATH.open('rb') as f:
    for i in range(64):
        line=f.readline()
        if not line or not line.endswith(b'\n'): break
        digest.update(line); consumed+=len(line)
        r=json.loads(line); s=r['applied_audit']; task=s['semantic_task']; ev=task['physical_evaluator']
        a=s['actuator_target_effect_audit']; h=a['policy_headroom_evidence']; nd=task['nominal_provider_diagnostics']
        assert s['physics_tick']==6176+8*(i+1)
        assert not s['prefix_teacher_data_in_ppo_storage'] and not s['prefix_checkpoint_policy_data_in_ppo_storage']
        rows.append(dict(action=i+1, global_decision=r['global_policy_decision'], tick=s['physics_tick'],
            phase=s['phase_id'], end_phase=s['end_phase_id'], body_forward_m=ev['goal_features']['body_forward_m'],
            RR=compact_leg(ev['current_legs']['RR']), RL=compact_leg(ev['current_legs']['RL']),
            FL=compact_leg(ev['current_legs']['FL']),
            RR_historical_placed=task['placed_history']['RR'], RL_historical_lift=task['active_lift_history']['RL'],
            N=s['nominal_action_full12'][8:], mapped=h['baseline_native_plus_controller_full12'][8:],
            REQUEST=h['requested_policy_residual_full12'][8:], effective=h['effective_policy_residual_full12'][8:],
            final=s['actual_drive_target_full12'][8:], actual_canonical=ev['measured_wheel_velocity_rad_s'],
            native_target=a['actual_native_targets']['wheel_velocity_rad_s'],
            current_mask=a['phase_mask_full12'], source=a['actual_target_source'],
            final_stop=nd['final_stop_owner']['active'],
            carry={k:nd.get('rr_carry_continuation',{}).get(k) for k in ('enabled','active','status')},
            verified=a['verified'] and a['setter_dispatch_targets_equal'] and a['actual_mapping_matches_dispatch']
                and all(x['verified'] for x in s['actuator_target_effect_audit_ticks']),
            terminal=r['terminal'], physical_valid=ev['valid'], physical_termination=ev['termination_reason']))
        last_history=task['history']
assert rows
first_loss=next((r for r in rows if not top(r['RR'])),None)
first_rl=next((r for r in rows if r['RL_historical_lift']),None)
first_out=next((r for r in rows if not r['RR']['within_top_xy']),None)
selected={rows[0]['tick'],rows[-1]['tick']}
for r in (first_loss,first_rl,first_out):
    if r:selected.update((r['tick']-8,r['tick']))
snaps=[{k:v for k,v in r.items() if k not in ('current_mask','verified','source','carry','final_stop','terminal','physical_valid','physical_termination')}
       for r in rows if r['tick'] in selected]
contacts=[]; previous=None
for r in rows:
    key=(r['RR']['contact_mode'],top(r['RR']),r['RR']['air'],r['RR']['ground_contact'],r['RR']['within_top_xy'])
    if key!=previous:contacts.append(dict(tick=r['tick'],mode=r['RR']['contact_mode'],TOP=top(r['RR']),
        AIR=r['RR']['air'],ground=r['RR']['ground_contact'],within_top_xy=r['RR']['within_top_xy']))
    previous=key
events=last_history['event_ticks']
result=dict(schema='block13.first_at_most_64_learner_actions.v1',run=str(RUN),
    source_binding=dict(path=str(PATH),first_complete_lines=len(rows),end_tick=rows[-1]['tick'],
        consumed_bytes=consumed,sha256_of_consumed_prefix=digest.hexdigest(),whole_active_file_hash=False),
    scope='N-initialized P12 training suffix from CP196608, old profile; not natural-P01 fixed-checkpoint evaluation; no future records read',
    handoff={k:handoff[k] for k in ('mode','physics_tick','sim_time_s','actual_phase','from_P01_current_policy','target_first_observed_decision')},
    prefix={k:prefix_result[k] for k in ('prefix_decisions','prefix_physics_ticks','policy_credit','accepted','target_first_observed_decision')},
    RL_at_handoff=dict(qualified=False,evidence='first learner endpoint6184 has historical lift false and no prior RL qualified event; handoff6176 direct contact snapshot not in prefix summary',current_contact_at_6176=None),
    wheel_order=['FL','FR','RL','RR'],wheel_units='canonical rad/s except native_target uses actual actuator axis signs',
    nominal_wheel_changes=[dict(tick=r['tick'],N=r['N']) for i,r in enumerate(rows) if i==0 or r['N']!=rows[i-1]['N']],
    RL_gap_extrema_ticks=dict(min=min(rows,key=lambda r:r['RL']['clearance_m'])['tick'],
                             max=max(rows,key=lambda r:r['RL']['clearance_m'])['tick']),
    events=events, RL_attempt_events=[x for x in last_history['lift_attempt_events'] if x['leg']=='RL' and x['physics_tick']>=6176],
    first_observed=dict(RR_not_TOP=None if first_loss is None else first_loss['tick'],
        RR_TOP_loss_interval=None if first_loss is None else [max(6176,first_loss['tick']-8),first_loss['tick']],
        RL_qualified=None if first_rl is None else events['active_lift'].get('RL'),
        RR_outside_top_xy=None if first_out is None else first_out['tick']),
    selected_endpoints=snaps,RR_contact_changes=contacts,
    aggregate=dict(actions=len(rows),ticks=[rows[0]['tick'],rows[-1]['tick']],
        body_net_forward_m=rows[-1]['body_forward_m']-rows[0]['body_forward_m'],
        RR_center_net_forward_m=rows[-1]['RR']['front_distance_m']-rows[0]['RR']['front_distance_m'],
        RL_center_net_forward_m=rows[-1]['RL']['front_distance_m']-rows[0]['RL']['front_distance_m'],
        RL_gap_m=stats(r['RL']['clearance_m'] for r in rows),
        RR_TOP_count=sum(top(r['RR']) for r in rows),RR_AIR_count=sum(r['RR']['air'] for r in rows),
        RR_ground_count=sum(r['RR']['ground_contact'] for r in rows),
        FL_TOP_count=sum(top(r['FL']) for r in rows),
        wheels={name:{field:stats(r[field][j] for r in rows) for field in ('N','mapped','REQUEST','effective','final','actual_canonical')}
                for j,name in enumerate(('FL','FR','RL','RR'))}),
    dispatch=dict(all_verified=all(r['verified'] for r in rows),all_full12_masks_one=all(r['current_mask']==[1]*12 for r in rows),
        final_stop_count=sum(r['final_stop'] for r in rows),target_readback_sources=sorted({r['source'] for r in rows}),
        final_minus_mapped_effective_max_abs=max(abs(f-n-e) for r in rows for f,n,e in zip(r['final'],r['mapped'],r['effective'])),
        native_measured_qd=None,source_channel_owner_ledger=None),
    endpoint={k:rows[-1][k] for k in ('tick','end_phase','terminal','physical_valid','physical_termination')},
    limitations='15Hz endpoint contacts only; exact events retained where logged. No measured native qd or full per-channel owner ledger. Wheel rotation is not wheel-end displacement or traction proof. No fixed-checkpoint/causal improvement claim.')
with (OUT/'block13_receiving_first64_readonly.json').open('x',encoding='utf-8') as f:
    json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps({k:result[k] for k in ('handoff','prefix','first_observed','aggregate','dispatch','endpoint')},indent=2))
for r in snaps:print('SNAP',json.dumps(r))
print('RL_EVENTS',json.dumps(result['RL_attempt_events']))
