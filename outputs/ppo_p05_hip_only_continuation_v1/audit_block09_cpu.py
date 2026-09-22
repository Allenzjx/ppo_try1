"""Sealed block09 only: real rollouts, episodes, adaptive LR and lineage."""
from collections import Counter
from copy import deepcopy
import itertools
import json
from pathlib import Path

import torch
import yaml
import audit_block08_first_carry_cpu as h
from wlr50_clean.ppo.semantic_training import state_hash
from wlr50_clean.ppo.semantic_supervisor import TaskStageSupervisor, _current_rr_receiver_preparation_retired

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
RUN = ROOT / 'runs/ppo_p05_hip_only_continuation_v1/train/20260922T1428131775302Z_g336b7c56d2f0_23b26a4c40a24a178d190def03381413'
SOURCE = OUT / 'checkpoints/history/checkpoint_rr_receiver_v2_step_000214400.pt'
TARGET = OUT / 'checkpoints/history/checkpoint_step_000216448.pt'
SOURCE_SHA = 'a039f071def736bcb8d2f9b6fb0c7e4691d9aeb74715918521b2268e9a521d9a'
TARGET_SHA = '8ec7784a9078ae7e9bb5f1d7298a4aa655c0d979ddb906e8a4d577bf8079a8f4'
COUNTERS = ('global_policy_decisions', 'ppo_updates', 'optimizer_steps')
PHASES = [f'P{i:02}' for i in range(1, 14)]
PRIOR_PHASE_COUNTS = [52,6799,12,5,1012,2107,19,9,3939,2,303,461,0]


def main():
    assert not torch.cuda.is_available(); torch.set_num_threads(1)
    manifest = h.read(RUN / 'training_manifest.json')
    assert manifest['lifecycle'] == 'SUCCEEDED' and manifest['actual_policy_decisions'] == 2048
    assert manifest['requested_policy_decisions'] == 2048 and manifest['unconsumed_requested_policy_decisions'] == 0
    sm, tm = (h.read(p.with_name(p.stem + '_manifest.json')) for p in (SOURCE,TARGET))
    assert h.sha(SOURCE) == sm['checkpoint_sha256'] == SOURCE_SHA
    assert h.sha(TARGET) == tm['checkpoint_sha256'] == TARGET_SHA and tm['source_run'] == str(RUN) and tm['save_load_round_trip']
    assert tuple(tm[k]-sm[k] for k in COUNTERS) == (2048,16,320)
    assert tuple(tm[k] for k in COUNTERS) == (216448,1656,33120)
    spec_path = ROOT / 'configs/ppo_p05_hip_only_continuation_v1/stage_task_spec.yaml'
    for path in (spec_path,ROOT/'src/wlr50_clean/ppo/semantic_supervisor.py'):
        assert h.sha(path) == tm['runtime_contract']['files'][path.relative_to(ROOT).as_posix()]
    current = object.__new__(TaskStageSupervisor); current.spec = yaml.safe_load(spec_path.read_text())
    old = object.__new__(TaskStageSupervisor); old.spec = deepcopy(current.spec)
    old.spec['rr_postcross_workspace_semantics'] = 'current_qualified_RR_over_top_receiver_retirement_v1'
    assert current.spec['rr_postcross_workspace_semantics'] == 'established_RR_over_top_receiver_retirement_v2'

    def retirement(ev):
        v2 = bool(_current_rr_receiver_preparation_retired(current.spec,'RR',ev))
        v1 = bool(_current_rr_receiver_preparation_retired(old.spec,'RR',ev))
        unplaced = not ev['history']['placed']['RR']
        delta = current.physical_potential(ev)-old.physical_potential(ev)
        assert delta >= -1e-12
        return dict(v2_gate=v2,v1_gate_same_state=v1,v2_only_gate=v2 and not v1,
            v2_unplaced_workspace_consumption=v2 and unplaced,v2_only_unplaced_consumption=v2 and not v1 and unplaced,
            v2_vs_v1_effective_Phi_difference=abs(delta)>1e-12,delta=delta)

    prefixes=[]; prefix_total=Counter(); prefix_phases=Counter(); local=Counter(); phases=Counter(); last_native=None
    for row in h.lines(RUN/'prefix_evidence.jsonl'):
        assert row['policy_credit'] is False
        if row['kind']=='checkpoint_prefix_decision':
            local['decisions']+=1;local['physics_ticks']+=row['physics_ticks'];phases[row['phase_id']]+=1
            assert row['actuator_target_effect_audit_summary']['all_ticks_verified'] and row['no_in_episode_state_writes_verified']
            last_native=row['actuator_target_effect_audit']
        elif row['kind']=='checkpoint_prefix_result':
            assert row['accepted']
        elif row['kind']=='policy_credit_start':
            start=row['start'];p=start['prefix_policy_provenance']
            assert start['actual_phase']=='P04' and p['checkpoint_sha256']==SOURCE_SHA
            assert p['frozen_actor_parameter_sha256']==sm['actor_parameter_sha256']
            assert p['frozen_for_entire_training_block'] and p['independent_parameter_and_buffer_storage_verified']
            assert p['source_policy_contract']==p['effective_policy_contract']==tm['policy_contract']
            assert p['source_runtime_content_sha256']==p['effective_runtime_content_sha256']==tm['runtime_contract']['runtime_content_sha256']
            prefixes.append({'counts':dict(local),'phases':dict(phases),'start_tick':start['physics_tick'],
                'start_time_s':start['sim_time_s'],'last_native':last_native})
            prefix_total.update(local);prefix_phases.update(phases);local=Counter();phases=Counter()
    assert len(prefixes)==3 and not local
    updates=list(h.lines(RUN/'optimizer_updates.jsonl'))
    assert [u['ppo_update'] for u in updates]==list(range(1641,1657))
    assert all(u['optimizer_steps']==20 and u['actor_parameters_changed'] and u['finite_nonzero_gradient_observed'] for u in updates)
    assert updates[0]['actor_parameter_sha256_before']==sm['actor_parameter_sha256']
    assert updates[-1]['actor_parameter_sha256_after']==tm['actor_parameter_sha256']
    assert all(a['actor_parameter_sha256_after']==b['actor_parameter_sha256_before'] for a,b in zip(updates,updates[1:]))
    lr=[{'update':u['ppo_update'],'optimizer_learning_rate':u['optimizer_learning_rate']} for u in updates]
    assert updates[-1]['optimizer_learning_rate']==tm['optimizer_learning_rate']==1e-5
    assert any(u['optimizer_learning_rate']!=1e-5 for u in updates), 'do not misreport adaptive LR as constant'
    requested=Counter();endpoints=Counter();execution=Counter();rr_inputs=Counter();rr_ends=Counter();ri=Counter();re=Counter()
    episodes=[];previous_ev=None;previous_native=None;previous_row=None;rollouts=[];max_logp=0.
    iterator=h.lines(RUN/'residual_and_projection_audit.jsonl')
    for block,update in enumerate(range(1641,1657)):
        rows=list(itertools.islice(iterator,128));assert len(rows)==128
        batch=torch.load(RUN/f'rollouts/rollout_{update:06}.pt',map_location='cpu',weights_only=False)
        obs=batch['observations']['policy'];actions=batch['actions'];means,stds=batch['distribution_params']
        assert obs.shape==(128,1,389) and actions.shape==(128,1,12) and torch.equal(obs,batch['observations']['critic'])
        assert batch['runtime_contract']==tm['runtime_contract'] and batch['policy_contract']==tm['policy_contract']
        assert batch['curriculum_epoch']['prefix_policy_provenance']['checkpoint_sha256']==SOURCE_SHA
        local_phases=Counter()
        for offset,row in enumerate(rows):
            index=block*128+offset;a=row['applied_audit'];p=row['policy_request'];native=a['actuator_target_effect_audit']
            ev=a['semantic_task']['physical_evaluator'];assist=native['capture_assist_evidence'];state=assist['state_after']
            if a['decision_count']==1:
                assert not episodes or episodes[-1]['terminal']
                epno=len(episodes);prefix=prefixes[epno]
                episodes.append({'episode_index':epno,'decisions':0,'physics_ticks':0,'input_phases':Counter(),
                    'assist_owned_endpoints':0,'assist_initialized_endpoints':0,'first_RR_placed_endpoint':None,
                    'first_RR_ground_after_placement':None,'RR_placed_TOP_endpoints':0,'RR_placed_ground_endpoints':0,
                    'prefix_decisions':prefix['counts']['decisions'],'prefix_ticks':prefix['counts']['physics_ticks']})
                previous_ev=None;previous_row=None;previous_native=prefix['last_native']
            ep=episodes[-1]
            assert row['global_policy_decision']==214401+index and a['decision_count']==ep['decisions']+1
            assert a['physical_core_decision_count_including_prefix']==ep['prefix_decisions']+a['decision_count']
            assert not a['prefix_teacher_data_in_ppo_storage'] and not a['prefix_checkpoint_policy_data_in_ppo_storage']
            assert a['task_result_scope']=='checkpoint_policy_initialized_suffix'
            assert p['sampling_draws']==1 and p['extra_random_draws']==0
            assert row['raw_policy_action_full12']==p['selected_raw_full12']==native['raw_policy_action_full12']
            assert row['old_distribution_mean_full12']==p['conditional_mean_full12'] and row['old_distribution_std_full12']==p['effective_sigma_full12']
            assert native['verified'] and native['actual_mapping_matches_dispatch'] and native['phase_mask_full12']==[1]*12
            assert assist['owner_indices'] in ([],[0,1]) and assist['candidate_before_assist_full12'][2:]==assist['candidate_after_assist_full12'][2:]
            assert torch.equal(obs[offset,0,372:384],torch.tensor(h.capture_assist_features(previous_native['capture_assist_evidence']['state_after']),dtype=obs.dtype))
            assert a['actuator_target_effect_audit_summary']['all_ticks_verified'] and a['no_in_episode_state_writes_verified']
            assert len(a['actuator_target_effect_audit_ticks'])==a['physics_ticks'] and all(t['verified'] for t in a['actuator_target_effect_audit_ticks'])
            if previous_row is not None:
                assert not previous_row['terminal'] and previous_row['applied_audit']['end_phase_id']==a['phase_id']
                assert previous_row['applied_audit']['physics_tick']==a['physics_tick']-a['physics_ticks']
            if a['phase_id']!=a['end_phase_id']:
                execution['ordinary_phase_handoffs']+=1;assert not row['terminal']
            ep['decisions']+=1;ep['physics_ticks']+=a['physics_ticks'];ep['input_phases'][a['phase_id']]+=1
            ep['assist_owned_endpoints']+=bool(assist['owner_indices']);ep['assist_initialized_endpoints']+=bool(state['initialized'])
            rr=ev['current_legs']['RR'];placed=bool(ev['history']['placed']['RR'])
            def event_record():
                return {'global_decision':row['global_policy_decision'],'endpoint_tick':a['physics_tick'],
                    'RR_placed_event_tick':ev['history']['event_ticks']['placed']['RR'],'phase':a['end_phase_id'],
                    'current_TOP':bool(rr['top_contact']),'current_ground':bool(rr['ground_contact']),
                    'current_lift_valid':bool(rr['current_lift_valid']),'gap_m':rr['clearance_m'],
                    'assist_owner_indices':assist['owner_indices']}
            if placed and ep['first_RR_placed_endpoint'] is None:ep['first_RR_placed_endpoint']=event_record()
            if placed and rr['ground_contact'] and ep['first_RR_ground_after_placement'] is None:ep['first_RR_ground_after_placement']=event_record()
            ep['RR_placed_TOP_endpoints']+=placed and bool(rr['top_contact']);ep['RR_placed_ground_endpoints']+=placed and bool(rr['ground_contact'])
            ep.update(last_global_decision=row['global_policy_decision'],physical_duration_s=a['sim_time_s'],last_phase=a['end_phase_id'],
                terminal=bool(row['terminal']),termination_reason=a['termination_reason'],full_task_success=bool(a['full_task_success']),
                placed_history=ev['history']['placed'],event_ticks=ev['history']['event_ticks'],
                final_RR={'gap_m':rr['clearance_m'],'current_TOP':rr['top_contact'],'ground_contact':rr['ground_contact'],
                    'current_lift_valid':rr['current_lift_valid'],'front_distance_m':rr['front_distance_m']})
            requested[a['phase_id']]+=1;local_phases[a['phase_id']]+=1;endpoints[a['end_phase_id']]+=1
            execution['credited_physics_ticks']+=a['physics_ticks'];execution['native_verified_ticks']+=len(a['actuator_target_effect_audit_ticks'])
            execution['assist_owned_endpoints']+=bool(assist['owner_indices']);execution['assist_initialized_endpoints']+=bool(state['initialized'])
            execution['terminal_samples']+=bool(row['terminal'])
            for key,column in (('qualified_current',149),('crossed_history',153),('placed_history',157)):
                rr_inputs[key]+=int(obs[offset,0,column]==1)
            if previous_ev is not None:
                input_rr=previous_ev['current_legs']['RR']
                rr_inputs['TOP']+=bool(input_rr['top_contact']);rr_inputs['ground']+=bool(input_rr['ground_contact'])
                before_rs=retirement(previous_ev)
                for key,value in before_rs.items():
                    if key!='delta':ri[key]+=int(value)
            else:
                assert not bool(obs[offset,0,149]) and not bool(obs[offset,0,153])
                ri['prefix_entry_gate_false_by_unearned_RR_bits']+=1
            for key,value in (('qualified_current',rr['current_lift_valid']),('crossed_history',ev['history']['front_edge_crossed']['RR']),
                ('placed_history',placed),('TOP',rr['top_contact']),('ground',rr['ground_contact'])):rr_ends[key]+=bool(value)
            end_rs=retirement(ev)
            for key,value in end_rs.items():
                if key!='delta':re[key]+=int(value)
            assert abs(current.physical_potential(ev)-a['semantic_task']['task_progress_potential'])<1e-12
            previous_ev=ev;previous_row=row;previous_native=native
        for tensor,key in ((actions,'raw_policy_action_full12'),(means,'old_distribution_mean_full12'),(stds,'old_distribution_std_full12'),
            (batch['actions_log_prob'],'old_log_probability'),(batch['values'],'old_value'),(batch['rewards'],'reward'),(batch['dones'],'terminal')):
            assert torch.isfinite(tensor).all() and torch.equal(tensor,torch.tensor([r[key] for r in rows],dtype=tensor.dtype).reshape(tensor.shape))
        assert [PHASES[i] for i in obs[:,0,:13].argmax(-1)]==[r['applied_audit']['phase_id'] for r in rows]
        for begin,end,key in ((372,384,'capture_assist_observed_features'),(384,389,'capture_continuation_observed_features')):
            assert torch.equal(obs[:,0,begin:end],torch.tensor([r['policy_request'][key] for r in rows],dtype=obs.dtype))
        assert torch.equal(obs[:,0,17],torch.tensor([r['applied_audit']['reward_breakdown']['potential_before'] for r in rows],dtype=obs.dtype))
        likelihood=h.read(RUN/f'rollouts/update_{update:06}_likelihood.json')
        uses=Counter(i for mini in likelihood['minibatches'] for ids in mini['rollout_flat_indices'] for i in ids)
        assert len(likelihood['minibatches'])==20 and uses==Counter({i:5 for i in range(128)})
        max_logp=max(max_logp,float((torch.distributions.Normal(means,stds).log_prob(actions).sum(-1).unsqueeze(-1)-batch['actions_log_prob']).abs().max()))
        rollouts.append({'update':update,'phase_counts':dict(local_phases),'effective_LR':updates[block]['optimizer_learning_rate']})
    assert next(iterator,None) is None and max_logp<=1e-5
    complete=list(h.lines(RUN/'completed_episodes.jsonl'));ended=[e for e in episodes if e['terminal']]
    assert len(episodes)==3 and len(ended)==len(complete)==2
    for ep,row in zip(ended,complete):
        assert ep['decisions']==row['policy_decisions'] and ep['termination_reason']==row['termination_reason']
        assert ep['physical_duration_s']==row['duration_s'] and ep['full_task_success']==row['full_task_success']
    assert [e['decisions'] for e in episodes]==[484,904,660]
    assert episodes[2]['first_RR_placed_endpoint']['RR_placed_event_tick']==5759 and episodes[2]['first_RR_ground_after_placement']
    assert episodes[2]['physical_duration_s']==61.6 and not episodes[2]['terminal'] and episodes[2]['last_phase']=='P12'
    assert execution['credited_physics_ticks']==execution['native_verified_ticks']==manifest['telemetry']['core']['physics_ticks']
    assert execution['credited_physics_ticks']+prefix_total['physics_ticks']==manifest['telemetry']['core']['physical_core_including_prefix']['physics_ticks']
    source,target=(torch.load(p,map_location='cpu',weights_only=False) for p in (SOURCE,TARGET))
    for payload,meta in ((source,sm),(target,tm)):
        assert h.parameter_sha(payload['actor_state_dict'])==meta['actor_parameter_sha256']
        assert h.parameter_sha(payload['critic_state_dict'])==meta['critic_parameter_sha256']
        assert state_hash(payload['optimizer_state_dict'])==meta['optimizer_state_sha256']
        assert all(payload['infos'][k]==v for k,v in meta.items() if k in payload['infos'])
    deltas=[float(target['optimizer_state_dict']['state'][k]['step']-v['step']) for k,v in source['optimizer_state_dict']['state'].items()]
    assert len(deltas)==12 and deltas==[320.]*12
    preserved=[k for k in sm if k!='resume_migration' and k.endswith(('_branch','_migration'))]
    preserved+=['normalization','normalizer_state_sha256','runner_config','policy_contract','runtime_contract']
    assert all(sm[k]==tm[k] for k in preserved)
    ledger=tm['rr_postcross_workspace_branch']['front_rehearsal_auxiliary']
    assert len(ledger['events'])==4 and [ledger[k] for k in ('accepted_auxiliary_updates_total','attempted_auxiliary_optimizer_steps_total')]==[103,104]
    assert [(e['fit_report']['accepted_auxiliary_updates'],e['fit_report']['attempted_auxiliary_optimizer_steps']) for e in ledger['events']]==[(32,32),(32,32),(32,32),(7,8)]
    older=tm['task_conditioned_hip_wheel_branch']['auxiliary_mean_learning']
    assert [older[k] for k in ('accepted_auxiliary_updates_total','attempted_auxiliary_optimizer_steps_total')]==[7,8]
    branches=('p05_capture_assist','capture_feedback_semantics','rr_postcross_workspace','rr_receiver_retirement_v2')
    for name in branches:assert tm[name+'_branch_counts']=={k:tm[k]-v for k,v in tm[name+'_branch']['counter_origin'].items()}
    srng,trng=sm['training_rng_state'],tm['training_rng_state']
    assert set(srng)==set(trng) and srng['seed']==trng['seed']==1001
    assert srng['torch_cuda_device_count']==trng['torch_cuda_device_count']==len(trng['torch_cuda'])==1
    assert state_hash(srng)!=state_hash(trng)
    for ep in episodes:
        ep['input_phases']={p:ep['input_phases'][p] for p in PHASES};ep['credited_duration_s']=ep['physics_ticks']/120
    for prefix in prefixes:del prefix['last_native']
    assert sum(PRIOR_PHASE_COUNTS)==214400-199680
    cumulative={p:PRIOR_PHASE_COUNTS[i]+requested[p] for i,p in enumerate(PHASES)}
    assert sum(cumulative.values())==216448-199680==16768 and tm['ppo_updates']-1525==115+16==131
    result={'schema':'wlr50_clean.block09_sealed_training_audit.v1','result':'PASS','run':str(RUN),
        'source_checkpoint':str(SOURCE),'source_sha256':SOURCE_SHA,'checkpoint':str(TARGET),'checkpoint_sha256':TARGET_SHA,
        'new_counts':dict(policy_decisions=2048,PPO_updates=16,Adam_steps=320),'lifetime_counts':{k:tm[k] for k in COUNTERS},
        'planned':2048,'unconsumed':0,'training_lifecycle':'SUCCEEDED_not_task_success',
        'request_phase_counts':{p:requested[p] for p in PHASES},'endpoint_phase_counts':dict(endpoints),
        'prefix_counts':dict(prefix_total),'prefix_phase_counts':dict(prefix_phases),'prefix_starts':prefixes,'prefix_PPO_credit':0,
        'execution_counts':dict(execution),'episodes':episodes,'ended_episodes':2,'sampling_boundary_partial_episodes':1,
        'RR_input_counts':dict(rr_inputs),'RR_endpoint_counts':dict(rr_ends),
        'RR_v2_actual_input_gate_counts':dict(ri),'RR_v2_actual_endpoint_gate_counts':dict(re),
        'gate_comparison_scope':'v2 and v1 recomputed on SAME actual block09 state; not old/new trajectory comparison; consumption excludes already placed',
        'per_rollout':rollouts,'raw_storage_and_five_PPO_uses_verified':True,'CPU_Normal_logp_max_error':max_logp,
        'all16_updates_actor_changed_finite_gradient':True,'all12_Adam_step_deltas':deltas,'Identity_preserved':True,
        'adaptive_LR_by_update':lr,'adaptive_LR_min':min(u['optimizer_learning_rate'] for u in updates),
        'adaptive_LR_max':max(u['optimizer_learning_rate'] for u in updates),'final_effective_LR':tm['optimizer_learning_rate'],
        'four_events_full_exact_source_carry':True,'front_AUX':[96,96],'RR_AUX':[7,8],'mixed_AUX':[103,104],'old_separate_AUX':[7,8],'AUX_added':0,
        'counter_origins':{b:tm[b+'_branch']['counter_origin'] for b in branches},'branch_added_counts':{b:tm[b+'_branch_counts'] for b in branches},
        'RNG_full_schema_CUDA_count_retained_and_state_advanced':True,'RNG_source_hash':state_hash(srng),'RNG_final_hash':state_hash(trng),
        'checkpoint_sidecar_embedded_equal':True,'actual_official_save_reload_recorded':True,
        'cumulative_since_P05_origin199680':{'prior_through_block08_phase_counts':dict(zip(PHASES,PRIOR_PHASE_COUNTS)),
            'phase_counts':cumulative,'policy_decisions':16768,'PPO_updates':131,'previous_PPO_updates':115},
        'physical_full_P01_PPO_success_claimed':False,'audit_physics_fit_or_checkpoint_writes':0,
        'limitations':['Suffix initialization prefixes are real but uncredited; not full P01 learned success.',
            'RR placed history survives current contact loss and ground recontact; those states are separately counted.',
            'Third episode ends only at sampling boundary; no terminal failure/success is invented.',
            'First prefix-entry current TOP/ground categories are absent from the learner endpoint stream; gate false is proved by unearned RR bits.']}
    (OUT/'block09_training_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    table='\n'.join(f"| {e['episode_index']} | {e['decisions']} | {e['physical_duration_s']:.6f} | {e['credited_duration_s']:.6f} | {e['last_phase']} | {e['termination_reason'] or 'sampling boundary; done=false'} |" for e in episodes)
    md=f'''# Block09 sealed genuine PPO audit

**PASS: actual +2048 decisions /16 PPO /320 Adam**, cumulative **216448 /1656 /33120**. Planned2048, unconsumed0. Training process SUCCEEDED is not task success. Final checkpoint SHA `{TARGET_SHA}`.

Credited phase inputs: **{dict(requested)}**; other P01–P13 phases0. Three frozen migrated-source prefixes total **{prefix_total['decisions']} actions/{prefix_total['physics_ticks']} ticks**, all credit0. Learner native-verified physics ticks **{execution['credited_physics_ticks']}**, assist-owned endpoints **{execution['assist_owned_endpoints']}**. Every original389/raw12/μ/σ/logp/value/reward/done matches sealed storage, and each sample is used5 times; CPU logp max error {max_logp:.9g}.

| Episode | Learner decisions | Total seconds | Credited seconds | First unfinished/current stage | Result |
| --- | ---: | ---: | ---: | --- | --- |
{table}

Episode2 really earned RR placement at tick5759 and continued through P10/P11 to P12. It later returned RR to ground (first sampled ground endpoint tick{episodes[2]['first_RR_ground_after_placement']['endpoint_tick']}); placement history is not current support. At61.6s it remains an unfinished P12 sampling-boundary partial, not a full PPO success. The other two actual results remain BODY_COLLISION and INCOMPLETE_CONTROLLER_BLOCKED.

Actual RR input counts: {dict(rr_inputs)}. New-v2 same-state gate counts: {dict(ri)}. Gate, unplaced workspace consumption and actual v2-v1 potential difference are distinct; these counts do not replace the prior same-state report or claim causal outcome improvement.

All16 actor updates have finite nonzero gradients; all12 Adam states advance320. **LR is adaptive, not constant**: min{result['adaptive_LR_min']:.9g}, max{result['adaptive_LR_max']:.9g}, final1e-5; per-update values are in JSON. Identity/full RNG schema and CUDA count persist, RNG advances normally. All four full event records and origins/migrations carry exactly: front96/96, RR7/8, mixed103/104, older separate7/8; no new AUX. Actual checkpoint hashes/embedded metadata and official reload verify.

Since P05 origin199680: **16768 policy decisions /131 PPO updates**; cumulative phase counts **{cumulative}** (prior115 + current16 updates). Prefix/AUX credit is excluded. CPU-only bounded audit; no simulation, fit, production/frozen-helper edit or checkpoint write. Helper exits after reports.
'''
    (OUT/'block09_training_audit.md').write_text(md,encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('result','new_counts','lifetime_counts','request_phase_counts','prefix_counts','execution_counts',
        'RR_input_counts','RR_v2_actual_input_gate_counts','adaptive_LR_by_update','cumulative_since_P05_origin199680')},indent=2))


if __name__=='__main__':
    main()
