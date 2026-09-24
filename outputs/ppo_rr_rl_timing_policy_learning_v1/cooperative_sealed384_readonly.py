"""Bounded sealed384 audit; stdout only, stdlib, never imports training runtime."""
from collections import Counter
import itertools
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/ppo_rr_rl_timing_policy_learning_v1/train/20260924T0038084596548Z_g49eb23163a6e_baf0b6006fea46a3a731db4c4d63d633'
CP=ROOT/'outputs/ppo_rr_rl_timing_policy_learning_v1/branches/ancestor220544_recapture_v2/checkpoints/history/checkpoint_step_000223616.pt'
def rows(path,count=None):
    with path.open(encoding='utf-8') as f:
        return [json.loads(s) for s in (itertools.islice(f,count) if count is not None else f)]
def span(values):
    return None if not values else dict(minimum=min(values),maximum=max(values))
def selected(d,keys):
    return {k:d.get(k) for k in keys}
def mark(row):
    a=row['applied_audit']; t=a['semantic_task']; ev=t['physical_evaluator']; c=t['rr_capture_continuation']
    legs=ev['current_legs']; diag=t['nominal_provider_diagnostics']
    fields=('contact_mode','current_lift_valid','air','top_contact','ground_contact','bearing_force_n',
            'load_fraction','within_top_xy','within_lateral_span','front_distance_m','clearance_m',
            'consecutive_top_samples','consecutive_air_samples','current_lift_qualified_tick')
    return dict(decision=row['global_policy_decision'],tick=a['physics_tick'],time_s=a['sim_time_s'],phase=a['phase_id'],
        request_tick=a['physics_tick']-8,request_rear_timing_features=row['policy_request']['rear_policy_timing_observed_features'],
        request_cooperative_flags=selected(row['policy_request'],('cooperative_observed_rr_carry_capture','cooperative_observed_rr_top_reachable',
            'cooperative_observed_rl_prep_transfer','cooperative_parent_receiving_continuation_active','cooperative_prep_allowed','cooperative_fl_wheel_extra_active')),
        RR=selected(legs['RR'],fields),RL=selected(legs['RL'],fields),
        capture=selected(c,('rr_lift_carry','rr_top_reachable','rr_top_contact','rr_current_bearing',
            'rl_transfer_ready','other_measured_supports','rr_actual_joint_margins_deg')),
        source_layers=[selected(x,('stage','source_ticks','status','wait_reason','late_group_start_tick',
            'rr_current_bearing','support_transfer_permitted','rl_current_swing',
            'wheel_source_clock_continues_after_start')) for x in diag['source_partial_order']['layers'] if x['stage'] in ('P09','P12')],
        completion_values=t['completion_values'],stage_age_s=t['stage_age_s'],
        terminal=a['termination_reason'],task_success=a['task_success'],body_forward_m=ev['goal_features']['body_forward_m'],
        current_supports=[k for k,v in legs.items() if v['support']],
        applied_final_full12=a['actual_drive_target_full12'],
        measured_wheel_velocity_rad_s=ev['measured_wheel_velocity_rad_s'],
        measured_wheel_order=ev['stop_progress_wheel_order'])
def lp(raw,mean,std):
    return sum(-.5*((x-m)/s)**2-math.log(s)-.5*math.log(2*math.pi) for x,m,s in zip(raw,mean,std))
def contact_marker(row):
    m=mark(row)
    return dict(decision=m['decision'],tick=m['tick'],time_s=m['time_s'],phase=m['phase'],
        RR=selected(m['RR'],('contact_mode','bearing_force_n','within_top_xy','clearance_m','consecutive_top_samples')),
        RL=selected(m['RL'],('contact_mode','current_lift_valid')))

manifest=json.loads(CP.with_name(CP.stem+'_manifest.json').read_text())
assert CP.exists() and manifest['global_policy_decisions']==223616 and manifest['ppo_updates']==1712
assert manifest['checkpoint_sha256']=='327f62688c41cb0d6137fb35de64592c12030389aa352c7199fd1f780079d632'
run=json.loads((RUN/'training_manifest.json').read_text())
assert run['lifecycle']=='SUCCEEDED' and run['actual_policy_decisions']==384 and run['ppo_updates_this_run']==3
assert run['optimizer_steps_this_run']==60
audit=rows(RUN/'residual_and_projection_audit.jsonl',384)
assert len(audit)==384 and [r['global_policy_decision'] for r in audit]==list(range(223233,223617))
assert [r['applied_audit']['physics_tick'] for r in audit]==list(range(5168,8233,8))
updates=rows(RUN/'optimizer_updates.jsonl')
advantages=rows(RUN/'advantage_audit.jsonl')
assert len(advantages)==3
assert len(updates)==3 and [u['ppo_update'] for u in updates]==[1710,1711,1712]
assert all(u['optimizer_steps']==20 and u['actor_parameters_changed'] for u in updates)
assert all(updates[i]['actor_parameter_sha256_after']==updates[i+1]['actor_parameter_sha256_before'] for i in (0,1))
allcounts=Counter(); maxima=Counter(); blocks=[]; exposures=Counter()
for b in range(3):
    block=audit[b*128:(b+1)*128]; counts=Counter(); counterroll=Counter()
    for r in block:
        p=r['policy_request'];a=r['applied_audit']; t=a['semantic_task'];ev=t['physical_evaluator']; rr=ev['current_legs']['RR'];rl=ev['current_legs']['RL']
        assert p['selected_raw_full12']==r['raw_policy_action_full12']==a['raw_policy_action_full12']
        assert p['selected_raw_log_probability']==r['old_log_probability']
        assert p['conditional_mean_full12']==r['old_distribution_mean_full12']
        assert p['effective_sigma_full12']==r['old_distribution_std_full12']
        assert p['rear_task_assists_enabled'] is False and p['rr_capture_assist_observed_features']==[0.]*14
        assert p['sampling_draws']==1 and p['extra_model_forwards']==p['extra_random_draws']==0
        assert not a['prefix_teacher_data_in_ppo_storage'] and not a['prefix_checkpoint_policy_data_in_ppo_storage']
        assert a['no_in_episode_state_writes_verified'] is True
        carry=p['cooperative_observed_rr_carry_capture']; reachable=p['cooperative_observed_rr_top_reachable']
        prep=p['cooperative_observed_rl_prep_transfer']; receiving=p['cooperative_parent_receiving_continuation_active']
        allowed=(carry and reachable) or prep
        mult=[1.]*12
        if carry and reachable: mult[6]=4.
        if allowed:
            for i,v in ((3,16.),(1,2.),(4,2.),(5,8.)):mult[i]=v
        if carry and reachable and not receiving:mult[8]=2.
        assert p['rear_local_sigma_multiplier_full12']==mult
        assert p['cooperative_prep_allowed']==allowed
        counts.update({f'phase_{a["phase_id"]}':1,'prep_gate':int(allowed),'FL_wheel_extra':int(mult[8]==2),
            'RR_qualified':int(rr['current_lift_valid']),'RR_TOP':int(rr['top_contact']),
            'RR_bearing':int(t['rr_capture_continuation']['rr_current_bearing']),
            'RR_AIR':int(rr['air']),'RR_GROUND':int(rr['ground_contact']),
            'RL_qualified':int(rl['current_lift_valid']),'RL_AIR':int(rl['air']),
            'RL_GROUND':int(rl['ground_contact']),'RL_TOP':int(rl['top_contact']),
            'RL_cross_history':int(ev['history']['front_edge_crossed']['RL']),
            'RL_placed_history':int(ev['history']['placed']['RL']),
            'task_success':int(a['task_success']),'task_terminal':int(a['termination_reason'] is not None)})
        for layer in t['nominal_provider_diagnostics']['source_partial_order']['layers']:
            if layer['stage']=='P12':
                counts['P12_RL_lane_held']+=int(layer['status']=='holding_RL_joint_lane')
        native=a['actuator_target_effect_audit_summary']
        assert native['all_ticks_verified'] and native['verified_tick_count']==8
        counts['native_verified_ticks']+=8
        maxima['sample_logp_error']=max(maxima['sample_logp_error'],abs(lp(p['selected_raw_full12'],p['conditional_mean_full12'],p['effective_sigma_full12'])-r['old_log_probability']))
        for c in a['reward_breakdown'].get('cooperative_preparation_sample_audit',[]):
            counterroll.update(samples=1,eligible=int(c.get('eligible',False)),positive_cost=int(c['raw_cost']>0))
        cost=a['reward_breakdown']['cooperative_counterroll_cost']
        expected_cost=sum(c['raw_cost']*c['dt_s']*c['coefficient_per_s'] for c in a['reward_breakdown']['cooperative_preparation_sample_audit'])
        assert abs(cost-expected_cost)<1e-12
        counterroll['charged_cost_sum']+=cost
        counterroll['positive_decisions']+=int(cost>0)
    like=json.loads((RUN/f'rollouts/update_{1710+b:06d}_likelihood.json').read_text())
    assert len(like['minibatches'])==20 and like['extra_model_forwards']==like['extra_random_draws']==0
    seen=Counter()
    for mb in like['minibatches']:
        for j,inds in enumerate(mb['rollout_flat_indices']):
            assert len(inds)==1 and 0<=inds[0]<128
            i=inds[0];seen[i]+=1;p=block[i]['policy_request']
            assert mb['old_log_probability'][j]==p['selected_raw_log_probability']
            assert mb['rear_local_sigma_multiplier_full12'][j]==p['rear_local_sigma_multiplier_full12']
            assert mb['cooperative_prep_allowed'][j]==p['cooperative_prep_allowed']
            maxima['minibatch_logp_error']=max(maxima['minibatch_logp_error'],abs(lp(p['selected_raw_full12'],mb['current_conditional_mean'][j],mb['current_conditional_sigma'][j])-mb['optimization_log_probability'][j]))
    assert seen==Counter({i:5 for i in range(128)})
    allcounts.update(counts)
    legs=[r['applied_audit']['semantic_task']['physical_evaluator']['current_legs'] for r in block]
    blocks.append(dict(block=b+1,generated_with_network_CP=[223232,223360,223488][b],
        generated_before_update=1710+b,credit_start=block[0]['global_policy_decision'],credit_end=block[-1]['global_policy_decision'],
        physical_start_s=block[0]['applied_audit']['sim_time_s']-8/120,physical_end_s=block[-1]['applied_audit']['sim_time_s'],
        counts=dict(counts),counterroll=dict(counterroll),
        logged_FL_nominal_endpoint_rad_s=span([r['applied_audit']['nominal_action_full12'][8] for r in block]),
        RR_TOP_force_n=span([x['RR']['bearing_force_n'] for x in legs if x['RR']['top_contact']]),
        RR_qualified_AIR_in_XY_gap_m=span([x['RR']['clearance_m'] for x in legs if x['RR']['current_lift_valid'] and x['RR']['air'] and x['RR']['within_top_xy']]),
        RL_front_m=span([x['RL']['front_distance_m'] for x in legs]),
        optimizer=updates[b],advantage_audit=selected(advantages[b],('sample_count','teacher_prefix_samples_included','overall','tail_bootstrap','ordinary_phase_change_samples')),
        last=mark(block[-1])))
assert max(maxima.values())<.001
contact_transitions=[]
for prev,r in zip(audit,audit[1:]):
    pv=prev['applied_audit']['semantic_task']['rr_capture_continuation']['rr_current_bearing']
    now=r['applied_audit']['semantic_task']['rr_capture_continuation']['rr_current_bearing']
    if pv!=now:
        contact_transitions.append(dict(change='GAIN' if now else 'LOSS',physics_tick_interval=[prev['applied_audit']['physics_tick'],r['applied_audit']['physics_tick']],
            last_before=contact_marker(prev),first_after=contact_marker(r)))
last=audit[-1]['applied_audit']['semantic_task'];ev=last['physical_evaluator']
late_window=[]
for r in audit:
    a=r['applied_audit']; tick=a['physics_tick']
    if tick not in (5680,5688,5696,5760,5768,5816,5824,6184,6192,6200,6208,6216,6224,6232):continue
    m=mark(r); t=a['semantic_task'];e=t['physical_evaluator'];effect=a['actuator_target_effect_audit']
    late_window.append(dict(decision=r['global_policy_decision'],tick=tick,time_s=a['sim_time_s'],phase=a['phase_id'],
        legs={leg:selected(e['current_legs'][leg],('contact_mode','air','ground_contact','top_contact','bearing_force_n','support','load_fraction','consecutive_top_samples')) for leg in ('FR','FL','RR','RL')},
        source_layers=m['source_layers'],source_nominal_full12=a['nominal_action_full12'],
        mapped_native_N_full12=effect['native_drive_target_full12'],final_target_full12=a['actual_drive_target_full12']))
prefix=run['telemetry']['core']
print(json.dumps(dict(status='SEALED384_AUDITED',run_dir=str(RUN),checkpoint=str(CP),checkpoint_sha256=manifest['checkpoint_sha256'],
    learning=dict(policy_decisions=384,ppo_updates=3,optimizer_steps=60,first=223233,last=223616),
    prefix=dict(decisions=prefix['prefix_behavior_decisions'],physics_ticks=prefix['prefix_physics_ticks'],PPO_credit=0),
    aggregate_counts=dict(allcounts),blocks=blocks,contact_transitions=contact_transitions,
    P09_late_bounded_window=late_window,
    max_logp_errors=dict(maxima),raw_likelihood_rows_verified=384,minibatches=60,exposures_per_sample=5,
    history=selected(ev['history'],('active_lift','front_edge_crossed','placed','event_ticks')),
    learner_rear_attempt_events=[e for e in ev['history']['lift_attempt_events'] if e['leg'] in ('RR','RL') and e['physics_tick']>5160],
    final_stall_diagnostic=last['stall_diagnostic'],final_substage=last['substage'],
    task_terminal_count=allcounts['task_terminal'],task_success_count=allcounts['task_success'],
    run_end='requested384_consumed_full_update_boundary;not_task_terminal_or_task_success',
    first128_attribution='CP223232 parameters + cooperative_sigma/control49eb stochastic execution before update1710; not learned by that update',
    later_attribution='Block2 uses update1710 parameters; block3 uses update1711 parameters in same continuing physical episode. State/time differ; not causal paired proof of learning benefit. CP223616 update1712 is unexecuted within this run.',
    scope='Only sealed named384. Contact counts are15Hz endpoints; changes bracket120Hz transitions, not exact transitions. No Torch/Isaac/model imports. Physical target verification is not tracking-success proof.'),indent=2))
