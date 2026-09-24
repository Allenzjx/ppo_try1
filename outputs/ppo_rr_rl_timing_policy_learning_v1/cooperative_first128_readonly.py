"""Only first sealed128 of the explicitly named cooperative run; stdlib, stdout."""
from collections import Counter
import itertools
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/ppo_rr_rl_timing_policy_learning_v1/train/20260924T0038084596548Z_g49eb23163a6e_baf0b6006fea46a3a731db4c4d63d633'
CHECKPOINT=ROOT/'outputs/ppo_rr_rl_timing_policy_learning_v1/branches/ancestor220544_recapture_v2/checkpoints/history/checkpoint_step_000223360.pt'
MANIFEST=CHECKPOINT.with_name(CHECKPOINT.stem+'_manifest.json')


def first_rows(path,count):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in itertools.islice(stream,count)]


def logp(raw,mean,std):
    return sum(-.5*((x-m)/s)**2-math.log(s)-.5*math.log(2*math.pi) for x,m,s in zip(raw,mean,std))


def maximum_error(a,b):return max(abs(x-y) for x,y in zip(a,b))


def marker(row):
    a=row['applied_audit'];task=a['semantic_task'];rr=task['physical_evaluator']['current_legs']['RR']
    p=row['policy_request']
    return dict(decision=row['global_policy_decision'],request_start_tick=a['physics_tick']-a['physics_ticks'],
        endpoint_tick=a['physics_tick'],endpoint_time_s=a['sim_time_s'],request_phase=a['phase_id'],
        RR_endpoint={key:rr[key] for key in ('current_lift_valid','air','top_contact','ground_contact','clearance_m','front_distance_m','bearing_force_n')},
        prep_endpoint=task.get('cooperative_preparation'),
        observed_gates={k:p[k] for k in ('cooperative_observed_rr_carry_capture','cooperative_observed_rr_top_reachable',
            'cooperative_observed_rl_prep_transfer','cooperative_parent_receiving_continuation_active',
            'cooperative_prep_allowed','cooperative_fl_wheel_extra_active')},
        total_rear_multiplier=p['rear_local_sigma_multiplier_full12'],raw_sigma=p['effective_sigma_full12'])


if not MANIFEST.exists() or not CHECKPOINT.exists() or not (RUN/'optimizer_updates.jsonl').exists():
    print(json.dumps(dict(status='WAITING_FOR_FIRST_SEALED128',checkpoint=str(CHECKPOINT))))
    raise SystemExit
metadata=json.loads(MANIFEST.read_text())
update=first_rows(RUN/'optimizer_updates.jsonl',1)[0]
assert metadata['global_policy_decisions']==223360 and update['global_policy_decisions']==223360
assert metadata['ppo_updates']==update['ppo_update']==1710 and update['optimizer_steps']==20
assert metadata['policy_contract']['version']=='rear_cooperative_prep_history_v1'
audit=first_rows(RUN/'residual_and_projection_audit.jsonl',128)
assert len(audit)==128 and [r['global_policy_decision'] for r in audit]==list(range(223233,223361))
counts=Counter(); maxima=Counter(); gate_first=None; distributions=[]; counterroll=Counter()
capability={i:dict(name=name,rows=[]) for i,name in ((3,'FR_knee'),(1,'FL_knee'),(4,'RL_hip'),(5,'RL_knee'),(8,'FL_wheel'))}
for row in audit:
    p=row['policy_request'];a=row['applied_audit'];task=a['semantic_task'];ev=task['physical_evaluator'];rr=ev['current_legs']['RR']
    assert p['policy_version']=='rear_cooperative_prep_history_v1'
    assert p['selected_raw_full12']==row['raw_policy_action_full12']==a['raw_policy_action_full12']
    assert p['selected_raw_log_probability']==row['old_log_probability']
    assert p['conditional_mean_full12']==row['old_distribution_mean_full12']
    assert p['effective_sigma_full12']==row['old_distribution_std_full12']
    assert p['extra_model_forwards']==p['extra_random_draws']==0 and p['sampling_draws']==1
    assert not a['prefix_teacher_data_in_ppo_storage'] and not a['prefix_checkpoint_policy_data_in_ppo_storage']
    assert p['rear_task_assists_enabled'] is False and p['rr_capture_assist_observed_features']==[0.]*14
    carry=p['cooperative_observed_rr_carry_capture'];reachable=p['cooperative_observed_rr_top_reachable']
    prep=p['cooperative_observed_rl_prep_transfer'];receiving=p['cooperative_parent_receiving_continuation_active']
    assert carry==bool(p['rear_policy_timing_observed_features'][0])
    assert reachable==bool(p['rr_capture_transfer_observed_features'][1])
    assert prep==bool(p['rear_policy_timing_observed_features'][2]) and receiving==p['receiving_continuation_active']
    allowed=(carry and reachable) or prep;wheel=carry and reachable and not receiving
    expected=[1.]*12
    if carry and reachable:expected[6]=4.
    if allowed:
        for i,value in ((3,16.),(1,2.),(4,2.),(5,8.)):expected[i]=value
    if wheel:expected[8]=2.
    assert p['cooperative_prep_allowed']==allowed and p['cooperative_fl_wheel_extra_active']==wheel
    assert p['rear_local_sigma_multiplier_full12']==expected
    assert p['cooperative_support_transfer_permission_modified'] is False and p['cooperative_target_or_mean_modified'] is False
    expected_all=[b*r*c for b,r,c in zip(p['innovation_sigma_multiplier_full12'],p['receiving_sigma_multiplier_full12'],expected)]
    assert maximum_error(expected_all,p['effective_innovation_sigma_multiplier_full12'])<1e-5
    sigma=[s*.25*m for s,m in zip(p['learned_sigma_full12'],expected_all)]
    mean=[.9*h+.1*b for h,b in zip(p['history_center_full12'],p['base_mean_full12'])]
    maxima['sample_logp_abs_error']=max(maxima['sample_logp_abs_error'],abs(logp(p['selected_raw_full12'],p['conditional_mean_full12'],p['effective_sigma_full12'])-p['selected_raw_log_probability']))
    maxima['sample_sigma_abs_error']=max(maxima['sample_sigma_abs_error'],maximum_error(sigma,p['effective_sigma_full12']))
    maxima['sample_mean_abs_error']=max(maxima['sample_mean_abs_error'],maximum_error(mean,p['conditional_mean_full12']))
    counts.update({f'phase_{a["phase_id"]}':1,'cooperative_gate':int(allowed),'FL_wheel_extra':int(wheel),
        'RR_qualified_endpoints':int(rr['current_lift_valid']),'RR_TOP_endpoints':int(rr['top_contact']),
        'RR_bearing_endpoints':int(task['rr_capture_continuation']['rr_current_bearing']),
        'RL_qualified_endpoints':int(ev['current_legs']['RL']['current_lift_valid']),
        'prep_relevant_endpoints':int(task.get('cooperative_preparation',{}).get('relevant',False))})
    if allowed and gate_first is None:gate_first=marker(row)
    if allowed:
        headroom=a['actuator_target_effect_audit']['policy_headroom_evidence']
        for i,item in capability.items():
            final=a['actual_drive_target_full12'][i]
            boundaries=(headroom['servo_safety_limits_deg'][i]+headroom['servo_hard_limits_deg'][i]) if i<8 else []
            item['rows'].append(dict(decision=row['global_policy_decision'],raw=p['selected_raw_full12'][i],
                conditional_mean=p['conditional_mean_full12'][i],sigma=p['effective_sigma_full12'][i],
                requested_residual=headroom['requested_policy_residual_full12'][i],
                effective_headroom_residual=headroom['effective_policy_residual_full12'][i],
                actual_final_target=final,headroom_clipped=i in headroom['clipped_servo_indices'],
                logged_boundary=next((value for value in boundaries if abs(final-value)<1e-6),None)))
    for c in a['reward_breakdown'].get('cooperative_preparation_sample_audit',[]):
        assert 0<=c['raw_cost']<=1
        counterroll.update(samples=1,eligible=int(c.get('eligible',False)),positive_cost=int(c['raw_cost']>0))
    native=a['actuator_target_effect_audit_summary']
    assert native['all_ticks_verified'] is True and native['verified_tick_count']==a['physics_ticks']
    counts['verified_native_physics_ticks']+=native['verified_tick_count']
    distributions.append(p)
likelihood=json.loads((RUN/'rollouts/update_001710_likelihood.json').read_text())
assert len(likelihood['minibatches'])==20 and likelihood['extra_model_forwards']==likelihood['extra_random_draws']==0
exposures=Counter()
for batch in likelihood['minibatches']:
    assert 'cooperative' in batch['sigma_source']
    for j,ids in enumerate(batch['rollout_flat_indices']):
        assert len(ids)==1 and 0<=ids[0]<128
        exposures[ids[0]]+=1;p=distributions[ids[0]]
        assert batch['old_log_probability'][j]==p['selected_raw_log_probability']
        assert batch['rear_local_sigma_multiplier_full12'][j]==p['rear_local_sigma_multiplier_full12']
        assert batch['cooperative_prep_allowed'][j]==p['cooperative_prep_allowed']
        lp=logp(p['selected_raw_full12'],batch['current_conditional_mean'][j],batch['current_conditional_sigma'][j])
        maxima['minibatch_logp_abs_error']=max(maxima['minibatch_logp_abs_error'],abs(lp-batch['optimization_log_probability'][j]))
        sigma=[math.exp(s)*.25*m for s,m in zip(batch['current_network_log_sigma_full12'][j],p['effective_innovation_sigma_multiplier_full12'])]
        maxima['minibatch_sigma_abs_error']=max(maxima['minibatch_sigma_abs_error'],maximum_error(sigma,batch['current_conditional_sigma'][j]))
assert exposures==Counter({i:5 for i in range(128)}) and max(maxima.values())<.001
advantage=first_rows(RUN/'advantage_audit.jsonl',1)[0]
assert advantage['sample_count']==128 and advantage['teacher_prefix_samples_included'] is False
prefix=None
with (RUN/'prefix_evidence.jsonl').open() as stream:
    for line in stream:
        row=json.loads(line)
        assert row.get('policy_credit') is False
        if row.get('kind')=='policy_credit_start':prefix=row['start'];break
assert prefix is not None and prefix['mode']=='successful_nominal_initialized_suffix'
capability_summary=[]
for i,item in capability.items():
    values=item['rows'];ranges={}
    for field in ('raw','conditional_mean','sigma','requested_residual','effective_headroom_residual','actual_final_target'):
        ranges[field]=None if not values else dict(minimum=min(r[field] for r in values),maximum=max(r[field] for r in values))
    boundary_rows={}
    for value in sorted({r['logged_boundary'] for r in values if r['logged_boundary'] is not None}):
        selected=[r for r in values if r['logged_boundary']==value]
        boundary_rows[str(value)]=dict(count=len(selected),distinct_raw_values=len({r['raw'] for r in selected}),
            raw_minimum=min(r['raw'] for r in selected),raw_maximum=max(r['raw'] for r in selected))
    capability_summary.append(dict(channel_index=i,name=item['name'],prep_gate_samples=len(values),
        units='raw/mean/sigma dimensionless; residual/final deg' if i<8 else 'raw/mean/sigma dimensionless; residual/final canonical rad/s',
        ranges=ranges,headroom_clipped_endpoint_count=sum(r['headroom_clipped'] for r in values),
        distinct_raw_values=len({r['raw'] for r in values}),distinct_final_targets_rounded_6dp=len({round(r['actual_final_target'],6) for r in values}),
        repeated_logged_servo_boundary_targets=boundary_rows,
        wheel_hard_boundary_not_supplied_in_this_headroom_receipt=i>=8,
        source='same decision policy_request and last physics actuator_target_effect_audit.policy_headroom_evidence plus actual_drive_target_full12; no independently recomputed nominal subtraction'))
print(json.dumps(dict(status='SEALED128_AUDITED',run_dir=str(RUN),checkpoint=str(CHECKPOINT),
    checkpoint_sha256=metadata['checkpoint_sha256'],learning=dict(decisions=128,ppo_updates=1,optimizer_steps=20,first=223233,last=223360),
    prefix={k:prefix[k] for k in ('mode','physics_tick','sim_time_s','target_first_observed_decision','from_P01_current_policy')},
    counts=dict(counts),counterroll_physical_samples=dict(counterroll),max_abs_errors=dict(maxima),
    actual_capability_on_prep_gate_only=capability_summary,
    original_raw_logp_exact_rows=128,minibatches=20,exposures_per_sample=5,
    first_gate=gate_first,last_endpoint=marker(audit[-1]),update=update,
    raw_GAE_mean=advantage['overall']['raw_gae_returns_minus_old_values']['mean'],
    scope='Only sealed first128; endpoint contact counts not120Hz contact duration; original raw Gaussian not projected-target likelihood; no PT/Torch/model reads'),indent=2))
