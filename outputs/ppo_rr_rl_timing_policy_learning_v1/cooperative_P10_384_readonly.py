"""Read-only sealed P10/384 evidence; no Torch/Isaac, no writes."""
import collections,hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/ppo_rr_rl_timing_policy_learning_v1/train/20260924T0130272021180Z_g49eb23163a6e_5e6b94b24cce4a2fb9edc117565be855'
CP=ROOT/'outputs/ppo_rr_rl_timing_policy_learning_v1/branches/ancestor220544_recapture_v2/checkpoints/history/checkpoint_step_000224000.pt'
def readrows(p):
    with p.open(encoding='utf-8') as f:return [json.loads(l) for l in f]
def take(d,ks):return {k:d.get(k) for k in ks}
def stats(xs):
    return None if not xs else dict(n=len(xs),minimum=min(xs),maximum=max(xs),mean=sum(xs)/len(xs))
def logp(x,m,s):return sum(-.5*((a-b)/c)**2-math.log(c)-.5*math.log(2*math.pi) for a,b,c in zip(x,m,s))
def marker(r):
    a=r['applied_audit'];t=a['semantic_task'];e=t['physical_evaluator'];p=r['policy_request'];h=a['actuator_target_effect_audit']['policy_headroom_evidence']
    fields=('contact_mode','current_lift_valid','air','ground_contact','top_contact','within_top_xy','within_lateral_span',
        'bearing_force_n','load_fraction','front_distance_m','clearance_m','consecutive_air_samples','consecutive_top_samples')
    return dict(decision=r['global_policy_decision'],tick=a['physics_tick'],time_s=a['sim_time_s'],phase=a['phase_id'],
        RR=take(e['current_legs']['RR'],fields),RL=take(e['current_legs']['RL'],fields),
        other_forces_n={k:e['current_legs'][k]['bearing_force_n'] for k in ('FR','FL')},
        completion_values=t['completion_values'],terminal=a['termination_reason'],task_success=a['task_success'],
        capture=take(t['rr_capture_continuation'],('rr_lift_carry','rr_top_reachable','rr_top_contact','rr_current_bearing','rl_transfer_ready','other_measured_supports')),
        p12_source=[take(x,('stage','source_ticks','status','wait_reason','support_transfer_permitted','rl_current_swing','wheel_source_clock_continues_after_start'))
            for x in t['nominal_provider_diagnostics']['source_partial_order']['layers'] if x['stage']=='P12'],
        p09_source=[take(x,('stage','source_ticks','status','wait_reason','late_group_start_tick','late_group_source_tick'))
            for x in t['nominal_provider_diagnostics']['source_partial_order']['layers'] if x['stage']=='P09'],
        request_start_tick=a['physics_tick']-8,observed_rear9=p['rear_policy_timing_observed_features'],
        observed_cooperative=take(p,('cooperative_observed_rr_carry_capture','cooperative_observed_rr_top_reachable','cooperative_observed_rl_prep_transfer',
            'cooperative_parent_receiving_continuation_active','cooperative_prep_allowed','cooperative_fl_wheel_extra_active')),
        final_target_full12=a['actual_drive_target_full12'],
        FL=dict(nominal=a['nominal_action_full12'][8],mapped_native=h['geometry_corrected_native_full12'][8],
            base_mean=p['base_mean_full12'][8],conditional_mean=p['conditional_mean_full12'][8],sigma=p['effective_sigma_full12'][8],
            raw=p['selected_raw_full12'][8],desired_residual=math.tanh(p['selected_raw_full12'][8])*p['current_cap_full12'][8],
            previous_filtered_request=p['previous_filtered_request_full12'][8],requested_after_rate=h['requested_policy_residual_full12'][8],
            after_headroom=h['effective_policy_residual_full12'][8],final=a['actual_drive_target_full12'][8],
            actual=e['measured_wheel_velocity_rad_s'][0]),
        body_forward_m=e['goal_features']['body_forward_m'])
audit=readrows(RUN/'residual_and_projection_audit.jsonl')
assert len(audit)==384 and [r['global_policy_decision'] for r in audit]==list(range(223617,224001))
assert [r['applied_audit']['physics_tick'] for r in audit]==list(range(6144,9209,8))
m=json.loads(CP.with_name(CP.stem+'_manifest.json').read_text())
manifestsha=hashlib.sha256(CP.with_name(CP.stem+'_manifest.json').read_bytes()).hexdigest()
assert CP.exists() and m['checkpoint_sha256']=='24fa50fba9212ddbda123380f9746769654af169c65fc48e09380c6d60ef5d9d'
assert manifestsha=='ac2424852f57a9a8953c4054f80ba6ffd293994d337424e6febd472b803a3eb3'
assert m['global_policy_decisions']==224000 and m['ppo_updates']==1715
training=json.loads((RUN/'training_manifest.json').read_text())
assert training['actual_policy_decisions']==384 and training['lifecycle']=='SUCCEEDED'
updates=readrows(RUN/'optimizer_updates.jsonl');adv=readrows(RUN/'advantage_audit.jsonl')
assert len(updates)==len(adv)==3 and sum(u['optimizer_steps'] for u in updates)==60
assert [u['global_policy_decisions'] for u in updates]==[223744,223872,224000]
channels={3:'FR_knee',1:'FL_knee',4:'RL_hip',5:'RL_knee',8:'FL_wheel'}
channel_rows={i:[] for i in channels};maxerr=collections.Counter();blocks=[];allcounts=collections.Counter()
for b in range(3):
    part=audit[128*b:128*(b+1)];counts=collections.Counter();cost=0.;fee=collections.Counter()
    for r in part:
        a=r['applied_audit'];t=a['semantic_task'];e=t['physical_evaluator'];p=r['policy_request'];effect=a['actuator_target_effect_audit'];h=effect['policy_headroom_evidence']
        assert p['selected_raw_full12']==r['raw_policy_action_full12']==a['raw_policy_action_full12']
        assert p['selected_raw_log_probability']==r['old_log_probability']
        assert p['conditional_mean_full12']==r['old_distribution_mean_full12'] and p['effective_sigma_full12']==r['old_distribution_std_full12']
        assert not a['prefix_teacher_data_in_ppo_storage'] and not a['prefix_checkpoint_policy_data_in_ppo_storage']
        assert not p['rear_task_assists_enabled'] and p['rr_capture_assist_observed_features']==[0.]*14
        assert p['extra_model_forwards']==p['extra_random_draws']==0 and p['sampling_draws']==1
        carry=p['cooperative_observed_rr_carry_capture'];reach=p['cooperative_observed_rr_top_reachable']
        prep=p['cooperative_observed_rl_prep_transfer'];recv=p['cooperative_parent_receiving_continuation_active']
        mult=[1.]*12
        if carry and reach:mult[6]=4.
        if (carry and reach) or prep:
            for i,v in ((3,16.),(1,2.),(4,2.),(5,8.)):mult[i]=v
        if carry and reach and not recv:mult[8]=2.
        assert p['rear_local_sigma_multiplier_full12']==mult
        counts.update({f'phase_{a["phase_id"]}':1,'prep_gate':int(p['cooperative_prep_allowed']),'FL_extra':int(mult[8]==2)})
        for leg in ('RR','RL'):
            legrow=e['current_legs'][leg]
            for name,key in [('qualified','current_lift_valid'),('TOP','top_contact'),('AIR','air'),('GROUND','ground_contact')]:
                counts[leg+'_'+name]+=int(legrow[key])
            counts[leg+'_cross_history']+=int(e['history']['front_edge_crossed'][leg])
            counts[leg+'_placed_history']+=int(e['history']['placed'][leg])
        counts['RR_bearing']+=int(t['rr_capture_continuation']['rr_current_bearing'])
        counts['terminal']+=int(a['termination_reason'] is not None)
        counts['task_success']+=int(a['task_success'])
        held=sum(bool(x['handoff_hold_used']) for x in a['actuator_target_effect_audit_ticks'])
        counts['phase_handoff_held_physics_ticks']+=held
        native=a['actuator_target_effect_audit_summary']
        assert native['all_ticks_verified'] and native['verified_tick_count']==8
        maxerr['sample_raw_logp']=max(maxerr['sample_raw_logp'],abs(logp(p['selected_raw_full12'],p['conditional_mean_full12'],p['effective_sigma_full12'])-r['old_log_probability']))
        for i,name in channels.items():
            desired=math.tanh(p['selected_raw_full12'][i])*p['current_cap_full12'][i]
            previous=p['previous_filtered_request_full12'][i];maximum=(60. if i<8 else 1.8)*(8-held)/120
            rate_expected=max(previous-maximum,min(previous+maximum,desired))
            request=h['requested_policy_residual_full12'][i]
            maxerr['request_rate_reconstruction']=max(maxerr['request_rate_reconstruction'],abs(rate_expected-request))
            assert abs(rate_expected-request)<1e-4
            candidate=h['candidate_native_target_before_final_slew_full12'][i];final=a['actual_drive_target_full12'][i]
            slew=False;hard=False
            if i<8:
                lo,hi=h['servo_hard_limits_deg'][i];desiredfinal=max(lo,min(hi,candidate));old=effect['previous_final_drive_servo_deg'][i]
                delta=effect['tracking_reference_evidence']['maximum_delta_deg']
                expected=max(lo,min(hi,old+max(-delta,min(delta,desiredfinal-old))))
                maxerr['final_servo_reconstruction']=max(maxerr['final_servo_reconstruction'],abs(expected-final))
                assert abs(expected-final)<1e-8
                slew=abs(desiredfinal-old)>delta+1e-9;hard=abs(candidate-desiredfinal)>1e-9
            channel_rows[i].append(dict(block=b+1,prep=bool(p['cooperative_prep_allowed']),raw=p['selected_raw_full12'][i],
                base_mean=p['base_mean_full12'][i],conditional_mean=p['conditional_mean_full12'][i],sigma=p['effective_sigma_full12'][i],
                desired_residual=desired,requested_after_rate=request,effective_headroom=h['effective_policy_residual_full12'][i],
                final=final,rate_clipped=abs(request-desired)>1e-5,headroom_clipped=i in h['clipped_servo_indices'],
                final_slew_clipped=slew,final_hard_clipped=hard,
                boundary_target=(i<8 and any(abs(final-v)<1e-8 for v in h['servo_safety_limits_deg'][i]+h['servo_hard_limits_deg'][i])),
                N=a['nominal_action_full12'][i],N_mapped=h['geometry_corrected_native_full12'][i],
                actual=(e['measured_wheel_velocity_rad_s'][0] if i==8 else None)))
        for c in a['reward_breakdown']['cooperative_preparation_sample_audit']:
            fee.update(physical_samples=1,eligible=int(c['eligible']),positive=int(c['raw_cost']>0))
        cost+=a['reward_breakdown']['cooperative_counterroll_cost']
    like=json.loads((RUN/f'rollouts/update_{1713+b:06d}_likelihood.json').read_text());seen=collections.Counter()
    assert len(like['minibatches'])==20
    for batch in like['minibatches']:
        for j,inds in enumerate(batch['rollout_flat_indices']):
            assert len(inds)==1;idx=inds[0];seen[idx]+=1;p=part[idx]['policy_request']
            assert batch['old_log_probability'][j]==p['selected_raw_log_probability']
            assert batch['rear_local_sigma_multiplier_full12'][j]==p['rear_local_sigma_multiplier_full12']
            maxerr['current_raw_logp']=max(maxerr['current_raw_logp'],abs(logp(p['selected_raw_full12'],batch['current_conditional_mean'][j],batch['current_conditional_sigma'][j])-batch['optimization_log_probability'][j]))
    assert seen==collections.Counter({i:5 for i in range(128)})
    allcounts.update(counts)
    blocks.append(dict(block=b+1,network_used_CP=[223616,223744,223872][b],first_decision=part[0]['global_policy_decision'],last_decision=part[-1]['global_policy_decision'],
        physical_start_s=part[0]['applied_audit']['sim_time_s']-8/120,physical_end_s=part[-1]['applied_audit']['sim_time_s'],
        counts=dict(counts),optimizer=updates[b],advantage_audit=take(adv[b],('overall','tail_bootstrap','ordinary_phase_change_samples','teacher_prefix_samples_included')),
        counterroll=dict(fee),counterroll_cost=cost,last=marker(part[-1])))
authority=[]
for i,rs in channel_rows.items():
    scopes={}
    for label,subset in [('all',rs),('prep_active',[v for v in rs if v['prep']]),('N_zero',[v for v in rs if abs(v['N'])<1e-12])]:
        if not subset:scopes[label]={'n':0};continue
        fields=['raw','base_mean','conditional_mean','sigma','desired_residual','requested_after_rate','effective_headroom','final']
        scopes[label]=dict(n=len(subset),ranges={k:stats([v[k] for v in subset]) for k in fields},
            counts={k:sum(v[k] for v in subset) for k in ('rate_clipped','headroom_clipped','final_slew_clipped','final_hard_clipped','boundary_target')},
            negative_counts={k:sum(v[k]<-1e-8 for v in subset) for k in ('raw','conditional_mean','desired_residual','requested_after_rate','final')},
            distinct_raw=len({v['raw'] for v in subset}),distinct_final_6dp=len({round(v['final'],6) for v in subset}))
        if i==8:
            scopes[label]['actual_rad_s']=stats([v['actual'] for v in subset])
            scopes[label]['negative_actual_count']=sum(v['actual']<-1e-8 for v in subset)
    authority.append(dict(index=i,name=channels[i],scopes=scopes))
last=audit[-1]['applied_audit']['semantic_task'];ev=last['physical_evaluator']
transitions=[]
for before,after in zip(audit,audit[1:]):
    x=before['applied_audit']['semantic_task']['physical_evaluator']['current_legs']['RR']
    y=after['applied_audit']['semantic_task']['physical_evaluator']['current_legs']['RR']
    if (x['contact_mode'],x['current_lift_valid'])!=(y['contact_mode'],y['current_lift_valid']):
        transitions.append(dict(interval_ticks=[before['applied_audit']['physics_tick'],after['applied_audit']['physics_tick']],
            before=take(x,('contact_mode','current_lift_valid','bearing_force_n')),
            after=take(y,('contact_mode','current_lift_valid','bearing_force_n')),
            end_time_s=after['applied_audit']['sim_time_s']))
first_loss=next((i for i,r in enumerate(audit) if not r['applied_audit']['semantic_task']['rr_capture_continuation']['rr_current_bearing']),None)
first_ground=next(i for i,r in enumerate(audit) if r['applied_audit']['semantic_task']['physical_evaluator']['current_legs']['RR']['ground_contact'])
first_stop=next(i for i,r in enumerate(audit) if abs(r['applied_audit']['nominal_action_full12'][8])<1e-12)
first_reverse=next(i for i,r in enumerate(audit) if abs(r['applied_audit']['nominal_action_full12'][8])<1e-12 and r['applied_audit']['actual_drive_target_full12'][8]<-1e-8)
def window(i):return [marker(r) for r in audit[max(0,i-1):min(384,i+2)]]
loss_derivation=[]
for end_tick in (6208,7392):
    r=next(r for r in audit if r['applied_audit']['physics_tick']==end_tick)
    count=r['applied_audit']['semantic_task']['physical_evaluator']['current_legs']['RR']['consecutive_air_samples']
    loss_derivation.append(dict(endpoint_tick=end_tick,consecutive_air_samples=count,
        first_AIR_tick=end_tick-count+1,first_AIR_time_s=(end_tick-count+1)/120,
        method='Counter-derived: evaluator contiguous 120Hz update and reset-on-non-AIR; no instantaneous first-AIR force receipt claimed.'))
prefix=training['telemetry']['core']
print(json.dumps(dict(schema='readonly.cooperative_P10_sealed384.v1',run_dir=str(RUN),checkpoint=str(CP),
 checkpoint_sha256=m['checkpoint_sha256'],manifest_sha256=manifestsha,learning=dict(decisions=384,updates=3,optimizer_steps=60,cumulative_decisions=224000,cumulative_updates=1715),
 prefix=dict(decisions=prefix['prefix_behavior_decisions'],physics_ticks=prefix['prefix_physics_ticks'],credit=0,RR_placed_tick=6133),
 counts=dict(allcounts),blocks=blocks,authority=authority,raw_likelihood_max_errors=dict(maxerr),minibatches=60,each_sample_exposures=5,
 first_loss_window=window(first_loss),first_ground_window=window(first_ground),first_N_FL_zero_window=window(first_stop),first_N_zero_FL_reverse_window=window(first_reverse),
 RR_first_and_last_AIR_onset=loss_derivation,last_TOP_loss_window=window(next(i for i,r in enumerate(audit) if r['applied_audit']['physics_tick']==7392)),
 RR_contact_qualification_transitions=transitions,history=take(ev['history'],('active_lift','front_edge_crossed','placed','event_ticks')),
 learner_rear_events=[x for x in ev['history']['lift_attempt_events'] if x['leg'] in ('RR','RL') and x['physics_tick']>6136],
 termination='budget384_sealed_not_task_terminal;all_rollout_tails_nonterminal_bootstrap',
 attribution='Block1 CP223616 before update1713; block2 CP223744; block3 CP223872. Final CP224000 update1715 not physically executed within this run. Prefix RR placement not learner achievement.',
 rate_audit_semantics='Endpoint original raw/currentcap and observed previous request, subtract actual handoff-hold ticks from8 before60deg/s or1.8rad/s2 bound. Same-tick requested→headroom→candidate→previous-final hard/slew independently checked. No independent nominal counterfactual or claim everyinterior tick logged.',
 scope='Only sealed namedP10/384 plus current frozen code/config reads. No Torch/Isaac/model loads/prod writes/activeP01logs.'),separators=(',',':')))
