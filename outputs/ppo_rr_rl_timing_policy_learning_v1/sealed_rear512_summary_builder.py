"""Bounded stdlib read-only audit of two explicitly sealed d7 blocks; stdout only."""
import collections
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
LEDGER = ROOT/'runs/ppo_rr_rl_timing_policy_learning_v1/curriculum/p02_progress1536_gd7e97ee7b7e4/curriculum.json'
HEAD = 'd7e97ee7b7e493d4f3ff34f9c8762f73550ad7bd'
EXPECTED = {
    'P07': ('20260923T2341477711792Z_gd7e97ee7b7e4_d0bb8df8bc8c4cfb9f9c620a3a4e6890',222721,222976),
    'P10': ('20260923T2359308297105Z_gd7e97ee7b7e4_b9e3f44fc10f468b9792649c280aef23',222977,223232)}
spec = subprocess.check_output(['git','show',HEAD+':configs/ppo_rr_rl_timing_policy_learning_v1/stage_task_spec.yaml'],cwd=ROOT,text=True)
THRESHOLD = float(re.search(r'(?m)^  unloaded_leg_maximum_load_fraction: ([0-9.]+)$',spec)[1])


def rows(path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip(): yield json.loads(line)


def normal_logp(raw, mean, sigma):
    return sum(-.5*((x-m)/s)**2-math.log(s)-.5*math.log(2*math.pi) for x,m,s in zip(raw,mean,sigma))


def error(a,b):
    return max((abs(x-y) for x,y in zip(a,b)),default=0.)


def leg_value(task, name):
    ev=task['physical_evaluator']; row=ev['current_legs'][name]; history=ev['history']
    value={key:row.get(key) for key in ('current_lift_valid','motion_continuation_allowed','air','ground_contact',
        'top_contact','top_surface_contact','support','bearing_verified','bearing_force_n','contact_surface',
        'front_distance_m','clearance_m','load_fraction','load_fraction_valid','within_top_xy')}
    value.update(active_lift_history=history['active_lift'][name], crossed_history=history['front_edge_crossed'][name],
        placed_history=history['placed'][name], measured_unloaded=bool(row.get('load_fraction_valid') is True
            and row['load_fraction'] <= THRESHOLD))
    if name=='RR': value['current_bearing']=task['rr_capture_continuation']['rr_current_bearing']
    return value


def compact(row):
    app=row['applied_audit']; task=app['semantic_task']; ev=task['physical_evaluator']
    layers=task['nominal_provider_diagnostics']['source_partial_order']['layers']
    request_fields=row['policy_request']['rear_policy_timing_observed_features']
    return dict(decision=row['global_policy_decision'],tick=ev['physics_tick'],time_s=ev['simulation_time_s'],
        request_start_tick=ev['physics_tick']-app['physics_ticks'],
        request_phase=app['phase_id'],end_phase=app['end_phase_id'],terminal=row['terminal'],
        task_success=app['task_success'],full_task_success=app['full_task_success'],
        termination_reason=app['termination_reason'],RR=leg_value(task,'RR'),RL=leg_value(task,'RL'),
        source_layers=[{k:l.get(k) for k in ('stage','source_ticks','late_group_start_tick',
            'status','wait_reason','rr_current_bearing','support_transfer_permitted','rl_current_swing',
            'wheel_source_clock_continues_after_start')} for l in layers if l['stage'] in ('P09','P12')],
        request_rear_observed_features=request_fields,
        recorded_RR_capture_tick=task['nominal_provider_diagnostics'].get('capture_owner_hold',{}).get('captures',{}).get('RR',{}).get('observed_capture_tick'),
        request_start_source_ticks_decoded={k:round(request_fields[i]*200*120) for i,k in
            ((4,'P09'),(5,'P12_wheel'),(6,'P12_RL_joint'))},
        nominal=app['nominal_action_full12'],final_target=app['actual_drive_target_full12'])


def summarize(block):
    phase=block['from_phase']; name,first,last=EXPECTED[phase]; run=Path(block['run_dir'])
    assert run.name==name and block['lifecycle']=='SUCCEEDED' and block['prefix_policy_credit'] is False
    manifest=json.loads((run/'training_manifest.json').read_text())
    assert manifest['lifecycle']=='SUCCEEDED' and manifest['actual_policy_decisions']==256
    updates=list(rows(run/'optimizer_updates.jsonl')); advantages=list(rows(run/'advantage_audit.jsonl'))
    assert len(updates)==len(advantages)==2 and sum(u['optimizer_steps'] for u in updates)==40
    prefix=list(rows(run/'prefix_evidence.jsonl'))
    prefix_start=next(r['start'] for r in prefix if r['kind']=='policy_credit_start')
    assert all(r.get('policy_credit') is False for r in prefix)
    audit=[]; requests=[]; sample_checks=collections.Counter()
    max_request_logp_error=max_request_sigma_error=max_request_mean_error=0.
    for r in rows(run/'residual_and_projection_audit.jsonl'):
        a=r['applied_audit']; p=r['policy_request']; audit.append(compact(r)); requests.append(p)
        for key,ok in dict(raw_exact=p['selected_raw_full12']==r['raw_policy_action_full12']==a['raw_policy_action_full12'],
            logp_exact=p['selected_raw_log_probability']==r['old_log_probability'],
            mean_exact=p['conditional_mean_full12']==r['old_distribution_mean_full12'],
            sigma_exact=p['effective_sigma_full12']==r['old_distribution_std_full12'],
            single_forward_draw=p['sampling_draws']==1 and p['extra_model_forwards']==p['extra_random_draws']==0,
            rear_assists_off=p['rear_task_assists_enabled'] is False and p['rr_capture_assist_observed_features']==[0.]*14,
            prefix_excluded=a['prefix_teacher_data_in_ppo_storage'] is False and a['prefix_checkpoint_policy_data_in_ppo_storage'] is False,
            original_kernel=p['policy_version']=='rear_policy_p02_progress_history_v1').items():
            sample_checks[key]+=bool(ok)
        max_request_logp_error=max(max_request_logp_error,abs(normal_logp(r['raw_policy_action_full12'],
            r['old_distribution_mean_full12'],r['old_distribution_std_full12'])-r['old_log_probability']))
        sigma=[s*p['exploration_std_temperature']*m for s,m in zip(p['learned_sigma_full12'],p['effective_innovation_sigma_multiplier_full12'])]
        mean=[.9*h+.1*b for h,b in zip(p['history_center_full12'],p['base_mean_full12'])]
        max_request_sigma_error=max(max_request_sigma_error,error(sigma,p['effective_sigma_full12']))
        max_request_mean_error=max(max_request_mean_error,error(mean,p['conditional_mean_full12']))
    assert len(audit)==256 and [r['decision'] for r in audit]==list(range(first,last+1))
    assert all(count==256 for count in sample_checks.values())
    for current, following in zip(audit,audit[1:]):
        if not current['terminal'] and current['tick']==following['request_start_tick']:
            current['end_clocks_from_next_same_tick_request']=following['request_start_source_ticks_decoded']
            current['end_rear_flags_from_next_same_tick_request']={k:bool(following['request_rear_observed_features'][i])
                for i,k in ((0,'rr_carry_capture'),(2,'rl_prep_transfer'),(3,'rl_swing_capture'),(8,'p12_dependency_wait'))}
    phase_counts=dict(collections.Counter(r['request_phase'] for r in audit))
    assert phase_counts==block['actual_phase_samples']
    likelihood=[]
    for n,(update,adv) in enumerate(zip(updates,advantages)):
        assert update['ppo_update']==adv['ppo_update_intended']
        assert adv['sample_count']==128 and adv['teacher_prefix_samples_included'] is False
        assert adv['first_global_policy_decision']==first+n*128 and adv['last_global_policy_decision']==first+(n+1)*128-1
        path=run/'rollouts'/f"update_{update['ppo_update']:06d}_likelihood.json"
        data=json.loads(path.read_text()); assert data['extra_model_forwards']==data['extra_random_draws']==0
        exposures=collections.Counter(); maxima=dict(logp=0.,sigma=0.,mean=0.,ratio=0.,old_logp=0.)
        assert len(data['minibatches'])==20
        gradients_finite=True
        for batch in data['minibatches']:
            assert 'observed_rear_local_sigma' in batch['sigma_source']
            for index,aliases in enumerate(batch['rollout_flat_indices']):
                assert len(aliases)==1 and 0<=aliases[0]<128
                local=aliases[0]; exposures[local]+=1; p=requests[n*128+local]
                lp=normal_logp(p['selected_raw_full12'],batch['current_conditional_mean'][index],batch['current_conditional_sigma'][index])
                maxima['logp']=max(maxima['logp'],abs(lp-batch['optimization_log_probability'][index]))
                expected_sigma=[math.exp(x)*p['exploration_std_temperature']*m for x,m in
                    zip(batch['current_network_log_sigma_full12'][index],p['effective_innovation_sigma_multiplier_full12'])]
                expected_mean=[.9*h+.1*b for h,b in zip(p['history_center_full12'],batch['current_network_mean_full12'][index])]
                maxima['sigma']=max(maxima['sigma'],error(expected_sigma,batch['current_conditional_sigma'][index]))
                maxima['mean']=max(maxima['mean'],error(expected_mean,batch['current_conditional_mean'][index]))
                maxima['old_logp']=max(maxima['old_logp'],abs(p['selected_raw_log_probability']-batch['old_log_probability'][index]))
                ratio=math.exp(batch['optimization_log_probability'][index]-batch['old_log_probability'][index])
                maxima['ratio']=max(maxima['ratio'],abs(ratio-batch['ratio'][index]))
            for key in ('loss_gradient_wrt_network_mean_full12','loss_gradient_wrt_network_log_sigma_full12'):
                gradients_finite &= all(math.isfinite(x) for row in batch[key] for x in row)
        assert exposures==collections.Counter({i:5 for i in range(128)}) and gradients_finite
        assert maxima['old_logp']==0. and max(maxima.values())<.001
        likelihood.append(dict(update=update['ppo_update'],minibatches=20,exposures_per_sample=5,
            max_abs_errors_float64_scalar_vs_recorded_float32=maxima,all_head_gradients_finite=gradients_finite,
            raw_GAE_mean=adv['overall']['raw_gae_returns_minus_old_values']['mean'],
            standardized_advantage_positive=adv['overall']['stored_advantages']['positive_count'],
            standardized_advantage_negative=adv['overall']['stored_advantages']['negative_count'],
            tail_bootstrap=adv['tail_bootstrap'],ordinary_phase_changes=adv['ordinary_phase_change_samples']))
    first_loss=next((i for i in range(1,256) if audit[i-1]['RR']['current_bearing'] and not audit[i]['RR']['current_bearing']),None)
    counts={}
    for leg in ('RR','RL'):
        values=[r[leg] for r in audit]
        keys=('current_lift_valid','air','ground_contact','top_contact','top_surface_contact','support','measured_unloaded','crossed_history','placed_history')
        if leg=='RR': keys+=('current_bearing',)
        counts[leg]={key:sum(v[key] is True for v in values) for key in keys}
        counts[leg].update(gap_min_m=min(v['clearance_m'] for v in values),gap_max_m=max(v['clearance_m'] for v in values),
            final_gap_m=values[-1]['clearance_m'],max_bearing_force_n=max(v['bearing_force_n'] for v in values),
            load_fraction_min=min(v['load_fraction'] for v in values if v['load_fraction_valid']))
        qualified_top_air=[r for r in audit if r[leg]['current_lift_valid'] and r[leg]['air'] and r[leg]['within_top_xy']]
        tops=[r for r in audit if r[leg]['top_contact']]
        first_air=next((r for r in audit if r[leg]['air']),None)
        counts[leg].update(qualified_top_air_count=len(qualified_top_air),
            qualified_top_air_min_gap_m=min((r[leg]['clearance_m'] for r in qualified_top_air),default=None),
            first_air_endpoint=None if first_air is None else {k:first_air[k] for k in ('decision','tick','time_s',leg)},
            last_TOP_endpoint=None if not tops else {k:tops[-1][k] for k in ('decision','tick','time_s',leg)})
    return dict(from_phase=phase,run_dir=str(run),checkpoint=block['checkpoint'],checkpoint_sha256=block['checkpoint_sha256'],
        sealed_lifecycle=manifest['lifecycle'],learning=dict(first_decision=first,last_decision=last,decisions=256,ppo_updates=2,optimizer_steps=40,
            actual_phase_samples=phase_counts,update_ids=[u['ppo_update'] for u in updates]),
        prefix_zero_credit=dict(source='successful_nominal',**{k:prefix_start[k] for k in ('physics_tick','sim_time_s','from_P01_current_policy')},
            prefix_decisions=prefix_start['target_first_observed_decision'],all_policy_credit_false=True,
            RR_placement_at_first_learner_endpoint=audit[0]['RR']['placed_history'],
            RR_recorded_capture_tick=audit[0]['recorded_RR_capture_tick']),
        first_endpoint=audit[0],last_endpoint=audit[-1],counts_at_256_decision_endpoints=counts,
        first_observed_RR_bearing_loss_window=None if first_loss is None else audit[max(0,first_loss-1):first_loss+2],
        sample_checks=dict(sample_checks),request_max_abs_errors=dict(logp=max_request_logp_error,sigma=max_request_sigma_error,mean=max_request_mean_error),
        optimizer_updates=updates,advantage_and_likelihood=likelihood,
        terminal_decisions=sum(r['terminal'] for r in audit),full_task_success_decisions=sum(r['full_task_success'] for r in audit),
        analysis_limits='Decision-endpoint contact samples only; first lost bearing is bracketed, not an exact120Hz sensor onset. Scalar recomputation uses logged gates/scales and is not a new model forward or a PT tensor reload.')


ledger=json.loads(LEDGER.read_text())
assert ledger['runtime_head']==HEAD and ledger['lifecycle']=='SUCCEEDED'
blocks=[summarize(b) for b in ledger['completed_blocks'] if b['from_phase'] in EXPECTED]
assert len(blocks)==2
print(json.dumps(dict(schema='sealed_rear512_readonly.v1',source_head=HEAD,course_ledger=str(LEDGER),
    scope='Only the two sealed P07/P10 suffix256 blocks; no new course or old history scan',
    combined_learning=dict(decisions=512,ppo_updates=4,optimizer_steps=80),
    unload_measurement_threshold=THRESHOLD,blocks=blocks),ensure_ascii=False,indent=2,allow_nan=False))
