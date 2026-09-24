"""Read only the named natural course AFTER both final manifests exist.

Stdlib only; no physical runs, policy imports, writes, or unsealed credit.
"""
import collections
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'runs/ppo_rr_rl_timing_policy_learning_v1/train/20260924T0149526403839Z_g49eb23163a6e_b74e867ba96841109c0616c26526f9ba'

def rows(path):
    with path.open(encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)

def take(d, keys):
    return {k: d.get(k) for k in keys}

def logp(x, mean, sigma):
    return sum(-.5*((a-b)/c)**2-math.log(c)-.5*math.log(2*math.pi)
               for a,b,c in zip(x,mean,sigma))

def mark(r, episode):
    a=r['applied_audit']; t=a['semantic_task']; e=t['physical_evaluator']
    legkeys=('contact_mode','current_lift_valid','air','ground_contact','top_contact',
        'within_top_xy','bearing_force_n','load_fraction','front_distance_m','clearance_m',
        'consecutive_air_samples','consecutive_top_samples')
    return dict(episode=episode, decision=r['global_policy_decision'], tick=a['physics_tick'],
        time_s=a['sim_time_s'], request_phase=a['phase_id'], current_phase=t['stage_id'],
        legs={leg:take(e['current_legs'][leg],legkeys) for leg in ('FR','FL','RR','RL')},
        history=take(e['history'],('active_lift','front_edge_crossed','placed','event_ticks')),
        capture=take(t.get('rr_capture_continuation',{}),('rr_lift_carry','rr_top_reachable',
            'rr_top_contact','rr_current_bearing','rl_transfer_ready','other_measured_supports')),
        rear_task_public=t.get('rear_policy_timing',{}),
        observed_rear9=r['policy_request'].get('rear_policy_timing_observed_features'),
        source=[take(x,('stage','source_ticks','status','wait_reason','late_group_start_tick',
            'support_transfer_permitted','rl_current_swing','wheel_source_clock_continues_after_start'))
            for x in t['nominal_provider_diagnostics']['source_partial_order']['layers'] if x['stage'] in ('P09','P12')],
        final_target_full12=a['actual_drive_target_full12'],
        measured_canonical_wheel_rad_s=e.get('measured_wheel_velocity_rad_s'),
        body_forward_m=e['goal_features']['body_forward_m'],
        completion_values=t['completion_values'],terminal=a['termination_reason'],success=a['task_success'],
        termination_source=t.get('termination_source'),stage_age_s=t['stage_age_s'],local_timeout=t['local_timeout'],
        terminal_bootstrap_allowed=a['terminal_bootstrap_allowed'],
        reward=take(a['reward_breakdown'],('total','potential_shaping','terminal_event','elapsed_physics_s')))

if not all((RUN/f).is_file() for f in ('run_manifest.json','training_manifest.json')):
    print(json.dumps(dict(status='WAIT_FINAL_MANIFESTS',run_dir=str(RUN),credited=0,
        note='No optimizer or physical audit was read; zero here is audit credit, not an active-run count.')))
    raise SystemExit(0)
run=json.loads((RUN/'run_manifest.json').read_text())
m=json.loads((RUN/'training_manifest.json').read_text())
assert run['lifecycle']!='RUNNING' and m['lifecycle']!='RUNNING'
updates=list(rows(RUN/'optimizer_updates.jsonl'))
adv=list(rows(RUN/'advantage_audit.jsonl'))
assert updates and len(updates)==len(adv)
last_sealed=updates[-1]['global_policy_decisions']
core=m['telemetry']['core']
assert m['phase_suffix_curriculum_implemented'] is False
assert m['implemented_sampling']=='P01_full_task_only_initial_version'
prefix_receipt_fields=collections.Counter()
expected=m['actual_policy_decisions']
assert expected==len(updates)*128
assert last_sealed==m['global_policy_decisions']
counter=collections.Counter(); blockcounter=collections.Counter(); stages=[]; markers=[]
field_presence=collections.Counter()
episodes=[]; events=[]; event_seen=set(); likelihood=collections.Counter()
block=[]; blocks=[]; first_decision=None; previous=None; episode=0; episode_first=None
last_episode_marker=None; last_physical_contact={}; terminal_rows=[]; observed_count=0
firsts={}; physical_fee=collections.Counter(); block_fee=collections.Counter()
fee_cost=0.; block_cost=0.; sparse_transitions=[]

for r in rows(RUN/'residual_and_projection_audit.jsonl'):
    decision=r['global_policy_decision']
    if decision>last_sealed:
        continue
    a=r['applied_audit']; t=a['semantic_task']; e=t['physical_evaluator']; p=r['policy_request']
    if first_decision is None:
        first_decision=decision
    assert decision==first_decision+observed_count
    observed_count+=1
    new_episode=previous is None or a['physics_tick']<=previous['tick']
    if new_episode:
        if last_episode_marker:
            episodes.append(dict(index=episode,first=episode_first,last=last_episode_marker))
        episode+=1; episode_first=mark(r,episode); last_physical_contact={}
        assert a['physics_tick']==8 and a['phase_id']=='P01'
    endpoint=mark(r,episode)
    if previous is None or new_episode or t['stage_id']!=previous['current_phase']:
        stages.append(take(endpoint,('episode','decision','tick','time_s','request_phase','current_phase')))
    assert p['selected_raw_full12']==r['raw_policy_action_full12']==a['raw_policy_action_full12']
    assert p['selected_raw_log_probability']==r['old_log_probability']
    # Natural core is not wrapped in a prefix core, so these wrapper-only fields
    # are absent. Verify positive final sampling metadata and tick8 P01 resets;
    # do not silently default absent storage flags or prefix counters to false/0.
    for prefix_key in ('prefix_teacher_data_in_ppo_storage','prefix_checkpoint_policy_data_in_ppo_storage'):
        if prefix_key in a:
            assert a[prefix_key] is False
            prefix_receipt_fields[prefix_key+'_present_false']+=1
        else:
            prefix_receipt_fields[prefix_key+'_absent']+=1
    assert p['extra_model_forwards']==p['extra_random_draws']==0 and p['sampling_draws']==1
    assert not p['rear_task_assists_enabled'] and p['rr_capture_assist_observed_features']==[0.]*14
    assert a['actuator_target_effect_audit_summary']['all_ticks_verified']
    likelihood['sample_raw_logp_max_error']=max(likelihood['sample_raw_logp_max_error'],
        abs(logp(p['selected_raw_full12'],p['conditional_mean_full12'],p['effective_sigma_full12'])-r['old_log_probability']))
    carry=p['cooperative_observed_rr_carry_capture']; reach=p['cooperative_observed_rr_top_reachable']
    prep=p['cooperative_observed_rl_prep_transfer']; recv=p['cooperative_parent_receiving_continuation_active']
    mult=[1.]*12
    if carry and reach: mult[6]=4.
    if (carry and reach) or prep:
        for i,v in ((3,16.),(1,2.),(4,2.),(5,8.)): mult[i]=v
    if carry and reach and not recv: mult[8]=2.
    assert p['rear_local_sigma_multiplier_full12']==mult
    tickcount=collections.Counter({f'request_phase_{a["phase_id"]}':1,f'current_phase_{t["stage_id"]}':1,
        'prep_gate':int(p['cooperative_prep_allowed']),'FL_extra':int(mult[8]==2),
        'RR_current_bearing':int(t.get('rr_capture_continuation',{}).get('rr_current_bearing',False)),
        'task_success':int(a['task_success']),'terminal':int(a['termination_reason'] is not None)})
    for leg in ('FR','FL','RR','RL'):
        cur=e['current_legs'][leg]
        for label,key in (('qualified','current_lift_valid'),('TOP','top_contact'),('AIR','air'),('GROUND','ground_contact')):
            field_presence[f'{leg}_{key}_'+('present' if key in cur else 'absent')]+=1
            if key in cur:
                tickcount[f'{leg}_{label}']+=int(cur[key])
        for kind in ('front_edge_crossed','placed'):
            tickcount[f'{leg}_{kind}_history']+=int(e['history'][kind][leg])
            if e['history'][kind][leg]:
                k=(episode,leg,kind)
                if k not in firsts:
                    firsts[k]=dict(episode=episode,leg=leg,event=kind,
                        exact_event_tick=e['history']['event_ticks'][kind].get(leg),first_sample=endpoint)
        if leg in ('RR','RL'):
            key=(cur['contact_mode'],bool(cur.get('current_lift_valid')),bool(cur.get('within_top_xy')))
            if last_physical_contact.get(leg)!=key:
                sparse_transitions.append(dict(episode=episode,leg=leg,decision=decision,tick=a['physics_tick'],
                    time_s=a['sim_time_s'],**take(cur,('contact_mode','current_lift_valid','bearing_force_n',
                        'within_top_xy','front_distance_m','clearance_m','consecutive_air_samples','consecutive_top_samples'))))
                last_physical_contact[leg]=key
    for ev in e['history']['lift_attempt_events']:
        key=(episode,json.dumps(ev,sort_keys=True))
        if key not in event_seen:
            event_seen.add(key);events.append(dict(episode=episode,**ev))
    for c in a['reward_breakdown']['cooperative_preparation_sample_audit']:
        z=dict(physical_samples=1,eligible=int(c['eligible']),positive=int(c['raw_cost']>0))
        physical_fee.update(z);block_fee.update(z)
    fee_cost+=a['reward_breakdown']['cooperative_counterroll_cost']; block_cost+=a['reward_breakdown']['cooperative_counterroll_cost']
    counter.update(tickcount);blockcounter.update(tickcount)
    if a['termination_reason'] is not None:
        terminal_rows.append(endpoint)
    block.append(dict(raw=p['selected_raw_full12'],old_logp=r['old_log_probability'],multipliers=mult))
    previous=endpoint;last_episode_marker=endpoint
    if len(block)==128:
        b=len(blocks);u=updates[b]
        assert decision==u['global_policy_decisions']
        lr=json.loads((RUN/f'rollouts/update_{u["ppo_update"]:06d}_likelihood.json').read_text())
        exposure=collections.Counter()
        assert len(lr['minibatches'])==u['optimizer_steps']
        for batch in lr['minibatches']:
            for j,indices in enumerate(batch['rollout_flat_indices']):
                assert len(indices)==1
                idx=indices[0]; exposure[idx]+=1;old=block[idx]
                assert batch['old_log_probability'][j]==old['old_logp']
                assert batch['rear_local_sigma_multiplier_full12'][j]==old['multipliers']
                likelihood['current_raw_logp_max_error']=max(likelihood['current_raw_logp_max_error'],
                    abs(logp(old['raw'],batch['current_conditional_mean'][j],batch['current_conditional_sigma'][j])-batch['optimization_log_probability'][j]))
        assert exposure==collections.Counter({i:5 for i in range(128)})
        blocks.append(dict(index=b+1,first_decision=decision-127,last_decision=decision,
            sampling_network_previous_CP=decision-128,counts=dict(blockcounter),
            optimizer=take(u,('ppo_update','optimizer_steps','actor_parameters_changed','finite_nonzero_gradient_observed',
                'kl_mean','clip_fraction','value_loss','surrogate_loss','optimizer_learning_rate')),
            advantage=take(adv[b],('overall','tail_bootstrap','terminal_samples','ordinary_phase_change_samples','teacher_prefix_samples_included')),
            counterroll=dict(block_fee),counterroll_cost=block_cost,last=endpoint))
        block=[];blockcounter.clear();block_fee.clear();block_cost=0.
assert observed_count==expected and not block
episodes.append(dict(index=episode,first=episode_first,last=last_episode_marker))
cpinfo=m['checkpoints'][-1];cp=Path(cpinfo['checkpoint']);sidepath=Path(cpinfo['manifest'])
side=json.loads(sidepath.read_text()); assert cp.exists()
assert side['global_policy_decisions']==last_sealed
print(json.dumps(dict(schema='readonly.cooperative_natural1280.v1',run_dir=str(RUN),
    run_lifecycle=run['lifecycle'],training_lifecycle=m['lifecycle'],
    learning=dict(actual_decisions=observed_count,updates=len(updates),optimizer_steps=sum(u['optimizer_steps'] for u in updates),
        cumulative_decisions=last_sealed,cumulative_updates=updates[-1]['ppo_update'],
        first_decision=first_decision,unconsumed=m.get('unconsumed_requested_policy_decisions')),
    checkpoint=dict(path=str(cp),sha256_from_sealed_sidecar=side['checkpoint_sha256'],
        manifest_path=str(sidepath),manifest_sha256=hashlib.sha256(sidepath.read_bytes()).hexdigest()),
    prefix=dict(implemented_sampling=m['implemented_sampling'],phase_suffix_curriculum_implemented=False,
        core_prefix_counter_fields_present=[k for k in core if k.startswith('prefix_')],
        wrapper_storage_field_presence=dict(prefix_receipt_fields),
        each_episode_first_endpoint_is_tick8_P01=True,
        interpretation='No prefix path used; wrapper-only flags/counters absent, not false-defaulted. All1280 are natural policy decisions.'),
    core_phase_decisions=core['phase_decisions'],counts=dict(counter),blocks=blocks,
    executed_physics_ticks=core['physics_ticks'],schema_field_presence=dict(field_presence),
    episodes=episodes,phase_changes=stages,first_physical_history_events=list(firsts.values()),
    lift_attempt_events=events,rear_contact_qualification_transitions=sparse_transitions,
    terminals=terminal_rows,counterroll=dict(physical_fee),counterroll_cost=fee_cost,
    likelihood=dict(likelihood),attribution='Initial behavior and each block precede that block optimizer update; final saved checkpoint not re-evaluated within this training run.',
    scope='Named sealed natural run only; stdlib and existing receipts; no Torch/Isaac, model queries, production edits, or active rollout credit.'),separators=(',',':')))
