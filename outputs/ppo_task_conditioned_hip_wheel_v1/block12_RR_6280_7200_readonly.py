"""Bounded active suffix window. Stops at exact complete tick7200, CPU only."""
import hashlib
import json
from pathlib import Path
from rr_probe_readonly import lines

OUT=Path(__file__).resolve().parent; ROOT=OUT.parents[1]
RUN=ROOT/'runs/ppo_task_conditioned_hip_wheel_v1/train/20260921T1223490758141Z_g97ecd305afb5_3f4b1021c62b4daf84d7c5e53a91e9bf'
PATH=RUN/'residual_and_projection_audit.jsonl'
rows=[]; digest=hashlib.sha256();size=count=0
def leg(c):
    return {k:c.get(k) for k in ('contact_mode','air','ground_contact','obstacle_pair_active','contact_surface','top_surface_contact','bearing_verified',
        'support','bearing_force_n','within_top_xy','front_distance_m','clearance_m','current_lift_valid','placed_on_top')}
with PATH.open('rb') as stream:
    for line in stream:
        assert line.endswith(b'\n')
        r=json.loads(line);s=r['applied_audit'];t=s['physics_tick'];assert t<=7200
        digest.update(line);size+=len(line);count+=1
        if t>=6280:
            task=s['semantic_task'];ev=task['physical_evaluator'];a=s['actuator_target_effect_audit'];h=a['policy_headroom_evidence'];nd=task['nominal_provider_diagnostics']
            rows.append(dict(tick=t,phase=s['phase_id'],end_phase=s['end_phase_id'],body_front_m=ev['goal_features']['body_forward_m'],
                RR=leg(ev['current_legs']['RR']),FL=leg(ev['current_legs']['FL']),
                N=s['nominal_action_full12'][8:],mapped=h['baseline_native_plus_controller_full12'][8:],
                effective=h['effective_policy_residual_full12'][8:],target=s['actual_drive_target_full12'][8:],
                actual_canonical=ev['measured_wheel_velocity_rad_s'],mask=a['phase_mask_full12'],
                verified=a['verified'] and a['setter_dispatch_targets_equal'] and a['actual_mapping_matches_dispatch'],
                tick_verified=all(x['verified'] for x in s['actuator_target_effect_audit_ticks']),
                stop_owner=nd['final_stop_owner']['active'],carry_owner=nd.get('rr_carry_continuation'),
                p06_wheel_tail=nd.get('p06_wheel_tail'),physical=dict(valid=ev['valid'],termination=ev['termination_reason'],success=ev['success']),
                terminal=r['terminal']))
            last_history=task['history']
        if t==7200:break
assert [r['tick'] for r in rows]==list(range(6280,7201,8))
assert last_history['event_ticks']['placed']['RR']==6285
def top(r):
    c=r['RR'];return c['top_surface_contact'] and c['bearing_verified'] and c['support']
def small(r):
    return {k:v for k,v in r.items() if k not in ('mask','verified','tick_verified','carry_owner','p06_wheel_tail','physical','terminal')}
post=[r for r in rows if r['tick']>=6285]
first_loss=next((r for r in post if not top(r)),None)
first_out=next((r for r in post if not r['RR']['within_top_xy']),None)
first_ground=next((r for r in post if r['RR']['ground_contact']),None)
first_negative_front=next((r for r in post if r['RR']['front_distance_m']<0),None)
ticks={6280,post[0]['tick'],7200}
for r in (first_loss,first_out,first_ground,first_negative_front):
    if r:ticks.update((r['tick']-8,r['tick']))
source_changes=[r for i,r in enumerate(rows) if i==0 or r['N']!=rows[i-1]['N'] or (r['phase'],r['end_phase'])!=(rows[i-1]['phase'],rows[i-1]['end_phase'])]
contacts=[];before=None
for r in post:
    c=r['RR'];key=(c['contact_mode'],c['air'],c['ground_contact'],top(r),c['within_top_xy'],c['current_lift_valid'])
    if key!=before:contacts.append(dict(tick=r['tick'],RR=c))
    before=key
def agg(v):
    v=list(v);return dict(min=min(v),max=max(v),mean=sum(v)/len(v))
result=dict(schema='block12.closed_RR_retention_6280_7200.v1',source=str(PATH),
    bounded_prefix=dict(end_tick=7200,complete_lines=count,consumed_bytes=size,sha256=digest.hexdigest(),not_whole_active_file_hash=True),
    scope='successful_nominal real prefix handoff5456 then live PPO suffix with updates; not natural P01 policy evaluation or final episode result',
    wheel_order=['FL','FR','RL','RR'],units='wheel canonical rad/s; position m',
    events=last_history['event_ticks'],RR_attempt_events=[x for x in last_history['lift_attempt_events'] if x['leg']=='RR'],
    first_observed=dict(capture_tick=6285,first_postcapture_endpoint=post[0]['tick'],
        first_non_TOP_bearing_endpoint=None if first_loss is None else first_loss['tick'],
        first_outside_top_XY_endpoint=None if first_out is None else first_out['tick'],
        first_negative_front_distance_endpoint=None if first_negative_front is None else first_negative_front['tick'],
        first_ground_endpoint=None if first_ground is None else first_ground['tick']),
    selected_snapshots=[small(r) for r in rows if r['tick'] in ticks],
    source_or_phase_changes=[dict(tick=r['tick'],phase=r['phase'],end_phase=r['end_phase'],N=r['N'],
        carry_owner=r['carry_owner'],p06_wheel_tail=r['p06_wheel_tail']) for r in source_changes],
    RR_contact_changes=contacts,
    postcapture_window=dict(ticks=[post[0]['tick'],7200],endpoints=len(post),
        body_net_forward_m=post[-1]['body_front_m']-post[0]['body_front_m'],
        RR_center_net_forward_m=post[-1]['RR']['front_distance_m']-post[0]['RR']['front_distance_m'],
        RR_TOP_bearing_endpoints=sum(top(r) for r in post),RR_AIR_endpoints=sum(r['RR']['air'] for r in post),
        RR_ground_endpoints=sum(r['RR']['ground_contact'] for r in post),
        RR_target=agg(r['target'][3] for r in post),RR_actual_canonical=agg(r['actual_canonical'][3] for r in post),
        RR_negative_target_count=sum(r['target'][3]<0 for r in post),RR_negative_actual_count=sum(r['actual_canonical'][3]<0 for r in post)),
    dispatch=dict(full12_masks_all_one=all(r['mask']==[1]*12 for r in rows),all_verified=all(r['verified'] and r['tick_verified'] for r in rows),
        stop_owner_count=sum(r['stop_owner'] for r in rows),
        final_minus_mapped_effective_max_abs=max(abs(f-n-e) for r in rows for f,n,e in zip(r['target'],r['mapped'],r['effective'])),
        raw_native_measured_qd=None,full_source_stop_event_owner_ledger=None),
    endpoint=dict(tick=7200,phase=rows[-1]['end_phase'],terminal=rows[-1]['terminal'],physical=rows[-1]['physical']),
    limitations='15Hz saved contact endpoints not exact continuous loss times except explicit events. AIR=false or support=false does not imply ground. Source-stop changes before6280 unobserved. No torque/slip or counterfactual traction attribution. Future records not read.')
with (OUT/'block12_RR_6280_7200_readonly.json').open('x',encoding='utf-8') as stream:
    json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps({k:result[k] for k in ('first_observed','postcapture_window','dispatch','endpoint')},indent=2))
for r in result['selected_snapshots']:print('SNAP',json.dumps(r))
print('SOURCES',json.dumps(result['source_or_phase_changes']))
print('CONTACTS',json.dumps(result['RR_contact_changes']))
print('RR_EVENTS',json.dumps(result['RR_attempt_events']))
