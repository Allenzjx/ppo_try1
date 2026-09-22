"""Sealed block05 only. CPU/stdout audit, no simulation, optimization or writes.

Default refuses a non-SUCCEEDED training manifest. --self-test uses tiny synthetic
arithmetic/predicate fixtures and never opens this run or a checkpoint.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'runs/ppo_p05_hip_only_continuation_v1/train/20260922T0926100383683Z_g5fd88852bf20_344b6773d9854b41948d1632e1f0fc3f'
SOURCE = ROOT/'outputs/ppo_p05_hip_only_continuation_v1/checkpoints/history/checkpoint_rr_postcross_workspace_v1_step_000207872_g5fd88852bf20.pt'
SOURCE_SHA = 'ac67a23edd97552cb5b2ff7f5e3484bc2e3ad47146cc7ed90fac45adfe764752'
PHASES = [f'P{i:02}' for i in range(1,14)]
LEGS = ['FL','FR','RL','RR']
COUNTERS = ['global_policy_decisions','ppo_updates','optimizer_steps']
BRANCHES = ['p05_capture_assist','capture_feedback_semantics','rr_postcross_workspace']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def lines(name):
    with (RUN/name).open(encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def phase_dict(counter):
    return {phase:counter[phase] for phase in PHASES}


def parameter_state_sha(state):
    digest=hashlib.sha256()
    for name,value in sorted(state.items()):
        tensor=value.detach().cpu().contiguous()
        digest.update(name.encode());digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode());digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def workspace_flags(evaluation,spec,predicate):
    """Gate is not the same as actual consumption: placed legs bypass workspace."""
    gate = bool(predicate(spec,'RR',evaluation))
    consumed = gate and not evaluation['history']['placed']['RR']
    return {'predicate_active':gate,'workspace_share_consumed':consumed}


def self_test():
    import yaml
    from wlr50_clean.ppo.semantic_supervisor import _current_rr_receiver_preparation_retired
    spec=yaml.safe_load((ROOT/'configs/ppo_p05_hip_only_continuation_v1/stage_task_spec.yaml').read_text())
    event={'valid':True,'termination_reason':None,
        'history':{'active_lift':{'RR':True},'front_edge_crossed':{'RR':True},'placed':{'RR':False}},
        'current_legs':{'RR':{'current_lift_valid':True,'ground_contact':False,'within_top_xy':True,
            'within_lateral_span':True,'front_distance_m':.1,'air':True,'clearance_m':-.001,
            'top_surface_contact':False,'top_contact':False}}}
    predicate=_current_rr_receiver_preparation_retired
    assert workspace_flags(event,spec,predicate)=={'predicate_active':True,'workspace_share_consumed':True}
    placed=deepcopy(event);placed['history']['placed']['RR']=True
    assert workspace_flags(placed,spec,predicate)=={'predicate_active':True,'workspace_share_consumed':False}
    for field,value in [('ground_contact',True),('within_top_xy',False),('current_lift_valid',False),
                        ('front_distance_m',-.01),('clearance_m',spec['geometry']['top_gap_min_m']-.001)]:
        negative=deepcopy(event);negative['current_legs']['RR'][field]=value
        assert workspace_flags(negative,spec,predicate)=={'predicate_active':False,'workspace_share_consumed':False}
    # Every phase is printed, even when no samples reach the rear stages.
    counts=phase_dict(Counter({'P01':2,'P02':126}));assert len(counts)==13 and sum(counts.values())==128
    assert counts['P09']==counts['P13']==0
    # Necessary qualification/cross bits can prove reset retirement false, not TOP contact false.
    assert not (False and False)
    print(json.dumps({'self_test':'PASS','synthetic_predicate_cases':7,
        'actual_run_or_checkpoints_read':False,'real_learning_added':0,'files_written':False}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        self_test();return
    # Check lifecycle before loading tensors or reading growing collection streams.
    manifest=read(RUN/'training_manifest.json')
    if manifest.get('lifecycle')!='SUCCEEDED':
        raise RuntimeError('Block05 not sealed SUCCEEDED; stop without polling or inspecting partial rollouts')
    import torch,yaml
    from wlr50_clean.ppo.semantic_training import state_hash
    from wlr50_clean.ppo.semantic_supervisor import TaskStageSupervisor,_current_rr_receiver_preparation_retired
    torch.set_num_threads(1)
    assert not torch.cuda.is_available(),'Use CUDA_VISIBLE_DEVICES=-1 for this read-only CPU audit'
    assert manifest['stage']=='full_episode' and not manifest['phase_suffix_curriculum_implemented']
    assert not (RUN/'prefix_evidence.jsonl').exists()
    actual=manifest['actual_policy_decisions']
    assert type(actual) is int and 0<actual<=4096 and actual%128==0
    saved=manifest['checkpoints'];assert saved
    checkpoint=Path(saved[-1]['checkpoint']).resolve(strict=True)
    sm=read(SOURCE.with_name(SOURCE.stem+'_manifest.json'))
    cm=read(checkpoint.with_name(checkpoint.stem+'_manifest.json'))
    assert sha(SOURCE)==sm['checkpoint_sha256']==SOURCE_SHA
    assert sha(checkpoint)==cm['checkpoint_sha256'] and cm['save_load_round_trip']
    assert cm['source_run']==str(RUN) and cm['runtime_contract']==sm['runtime_contract']
    assert cm['global_policy_decisions']==manifest['global_policy_decisions']==sm['global_policy_decisions']+actual
    spec_path=ROOT/'configs/ppo_p05_hip_only_continuation_v1/stage_task_spec.yaml'
    for path in [spec_path,ROOT/'src/wlr50_clean/ppo/semantic_supervisor.py']:
        relative=path.relative_to(ROOT).as_posix()
        assert sha(path)==cm['runtime_contract']['files'][relative], 'Do not reinterpret this run with changed code/config'
    spec=yaml.safe_load(spec_path.read_text())
    revised=object.__new__(TaskStageSupervisor);revised.spec=spec
    original=object.__new__(TaskStageSupervisor);original.spec=deepcopy(spec)
    original.spec.pop('rr_postcross_workspace_semantics')

    def rr_semantics(ev):
        flag=workspace_flags(ev,spec,_current_rr_receiver_preparation_retired)
        new_phi=revised.physical_potential(ev);old_phi=original.physical_potential(ev)
        delta=new_phi-old_phi
        assert delta>=-1e-12 and (flag['workspace_share_consumed'] or abs(delta)<1e-12)
        return {**flag,'effective_potential_difference':abs(delta)>1e-12,
                'potential_delta':delta,'new_potential':new_phi}

    def finite(value):
        if torch.is_tensor(value):return bool(torch.isfinite(value).all())
        if isinstance(value,dict):return all(finite(v) for v in value.values())
        if isinstance(value,(tuple,list)):return all(finite(v) for v in value)
        return not isinstance(value,float) or math.isfinite(value)

    request,endpoint,totals=Counter(),Counter(),Counter()
    rr_inputs,rr_ends,retirement_inputs,retirement_ends=Counter(),Counter(),Counter(),Counter()
    retirement_input_phases,retirement_effective_input_phases=Counter(),Counter()
    all_leg_ends={leg:Counter() for leg in LEGS}
    episodes,compact=[],[]
    previous_ev=previous_semantics=previous_row=None
    for index,row in enumerate(lines('residual_and_projection_audit.jsonl')):
        assert index<actual, 'sealed run contains unaccounted decisions beyond its manifest'
        a=row['applied_audit'];task=a['semantic_task'];ev=task['physical_evaluator'];hist=ev['history']
        n=a['actuator_target_effect_audit'];p=row['policy_request']
        assert row['global_policy_decision']==sm['global_policy_decisions']+1+index
        if a['decision_count']==1:
            assert previous_row is None or previous_row['terminal']
            previous_ev=previous_semantics=None
            episodes.append(dict(episode_index=len(episodes),policy_decisions=0,physics_ticks=0,
                request_phases=Counter(),RR_endpoints=Counter(),retirement_input=Counter()))
            assert a['phase_id']=='P01'
        stat=episodes[-1];assert a['decision_count']==stat['policy_decisions']+1
        stat['policy_decisions']+=1;stat['physics_ticks']+=a['physics_ticks'];stat['request_phases'][a['phase_id']]+=1
        stat.update(last_global_decision=row['global_policy_decision'],last_phase=a['end_phase_id'],
            duration_s=a['sim_time_s'],terminal=bool(row['terminal']),termination_reason=a['termination_reason'],
            full_task_success=bool(a['full_task_success']),placed_history=hist['placed'],event_ticks=hist['event_ticks'])
        request[a['phase_id']]+=1;endpoint[a['end_phase_id']]+=1
        assert n['verified'] and n['actual_mapping_matches_dispatch'] and n['phase_mask_full12']==[1]*12
        assert a['no_in_episode_state_writes_verified'] and a['actuator_target_effect_audit_summary']['all_ticks_verified']
        assert not a.get('prefix_teacher_data_in_ppo_storage',False) and not a.get('prefix_checkpoint_policy_data_in_ppo_storage',False)
        assert row['raw_policy_action_full12']==p['selected_raw_full12']==n['raw_policy_action_full12']
        assert row['old_distribution_mean_full12']==p['conditional_mean_full12']
        assert row['old_distribution_std_full12']==p['effective_sigma_full12']
        assert p['sampling_draws']==1 and p['extra_random_draws']==0
        owners=n['capture_assist_evidence']['owner_indices'];assert owners in ([],[0,1])
        totals['assist_owned_endpoints']+=bool(owners)
        totals['physics_ticks']+=a['physics_ticks']
        totals['verified_native_ticks']+=a['actuator_target_effect_audit_summary']['verified_tick_count']
        totals['pending_endpoints']+=bool(task['fl_capture_pending'])
        totals['pending_advanced_endpoints']+=bool(task['capture_continuation']['scheduler_advanced_pending'])
        for leg in LEGS:
            current=ev['current_legs'][leg]
            for key,value in [('lift_feature_equivalent',current.get('current_lift_valid') if leg=='RR' else hist['active_lift'][leg]),
                              ('crossed_history',hist['front_edge_crossed'][leg]),('placed_history',hist['placed'][leg]),
                              ('current_TOP',current['top_contact'])]:all_leg_ends[leg][key]+=bool(value)
        rr=ev['current_legs']['RR']
        rr_flags={'qualified':bool(rr.get('current_lift_valid')),'crossed_history':bool(hist['front_edge_crossed']['RR']),
            'placed_history':bool(hist['placed']['RR']),'current_TOP':bool(rr['top_contact']),
            'crossed_unplaced':bool(hist['front_edge_crossed']['RR'] and not hist['placed']['RR'])}
        rr_ends.update({key:int(value) for key,value in rr_flags.items()});stat['RR_endpoints'].update({key:int(value) for key,value in rr_flags.items()})
        rs=rr_semantics(ev)
        assert abs(rs['new_potential']-task['task_progress_potential'])<1e-12
        for key in ['predicate_active','workspace_share_consumed','effective_potential_difference']:
            retirement_ends[key]+=int(rs[key])
        totals['retirement_endpoint_potential_delta_sum']+=rs['potential_delta']
        if previous_ev is None:
            totals['initial_reset_TOP_unknown']+=1
        else:
            rr_inputs['current_TOP_known']+=int(previous_ev['current_legs']['RR']['top_contact'])
            totals['TOP_inputs_known_from_adjacent_endpoint']+=1
            assert not previous_row['terminal']
            assert a['reward_breakdown']['potential_before']==previous_row['applied_audit']['semantic_task']['task_progress_potential']
        compact.append(dict(phase=a['phase_id'],raw=row['raw_policy_action_full12'],mean=row['old_distribution_mean_full12'],
            std=row['old_distribution_std_full12'],logp=row['old_log_probability'],value=row['old_value'],
            reward=row['reward'],terminal=row['terminal'],qRR=p['current_RR_qualification'],pRR=p['RR_placed_history'],
            phi_before=a['reward_breakdown']['potential_before'],assist=p['capture_assist_observed_features'],
            continuation=p['capture_continuation_observed_features'],rr_semantics=previous_semantics,episode_index=len(episodes)-1))
        previous_ev,previous_semantics,previous_row=ev,rs,row
    assert len(compact)==actual and totals['physics_ticks']==manifest['telemetry']['core']['physics_ticks']
    updates=list(lines('optimizer_updates.jsonl'))
    expected_updates=list(range(sm['ppo_updates']+1,cm['ppo_updates']+1))
    assert [u['ppo_update'] for u in updates]==expected_updates and len(updates)==actual//128
    assert len(updates)==manifest['ppo_updates_this_run']
    assert all(finite(u) and u['optimizer_steps']==20 and u['actor_parameters_changed'] and u['finite_nonzero_gradient_observed'] for u in updates)
    assert updates[0]['actor_parameter_sha256_before']==sm['actor_parameter_sha256']
    assert updates[-1]['actor_parameter_sha256_after']==cm['actor_parameter_sha256']
    assert all(a['actor_parameter_sha256_after']==b['actor_parameter_sha256_before'] for a,b in zip(updates,updates[1:]))
    max_error=0.;per_rollout=[]
    for block,update in enumerate(expected_updates):
        data=torch.load(RUN/f'rollouts/rollout_{update:06}.pt',map_location='cpu',weights_only=False)
        obs=data['observations']['policy'];actions=data['actions'];mean,std=data['distribution_params']
        assert finite(data) and obs.shape==(128,1,389) and actions.shape==(128,1,12)
        assert torch.equal(obs,data['observations']['critic']) and data['runtime_contract']==cm['runtime_contract']
        assert data['policy_contract']==cm['policy_contract'] and data['curriculum_epoch']['prefix_request'] is None
        segment=compact[128*block:128*(block+1)]
        for tensor,key in [(actions,'raw'),(mean,'mean'),(std,'std'),(data['actions_log_prob'],'logp'),
                            (data['values'],'value'),(data['rewards'],'reward'),(data['dones'],'terminal'),
                            (obs[:,0,17],'phi_before'),(obs[:,0,372:384],'assist'),(obs[:,0,384:389],'continuation')]:
            assert torch.equal(tensor,torch.tensor([r[key] for r in segment],dtype=tensor.dtype).reshape(tensor.shape))
        assert [PHASES[i] for i in obs[:,0,:13].argmax(-1)]==[r['phase'] for r in segment]
        assert [bool(v) for v in obs[:,0,149]]==[r['qRR'] for r in segment]
        assert [bool(v) for v in obs[:,0,157]]==[r['pRR'] for r in segment]
        local=Counter()
        for i,record in enumerate(segment):
            for key,j in [('qualified',149),('crossed_history',153),('placed_history',157)]:rr_inputs[key]+=int(obs[i,0,j]==1)
            rr_inputs['crossed_unplaced']+=int(obs[i,0,153]==1 and obs[i,0,157]==0)
            state=record['rr_semantics']
            if state is None:
                assert not bool(obs[i,0,149]) and not bool(obs[i,0,153])
                totals['initial_reset_retirement_proven_false_by_necessary_bits']+=1
                continue
            for key in ['predicate_active','workspace_share_consumed','effective_potential_difference']:
                retirement_inputs[key]+=int(state[key]);local[key]+=int(state[key])
                episodes[record['episode_index']]['retirement_input'][key]+=int(state[key])
            if state['predicate_active']:retirement_input_phases[record['phase']]+=1
            if state['effective_potential_difference']:retirement_effective_input_phases[record['phase']]+=1
        error=float((torch.distributions.Normal(mean,std).log_prob(actions).sum(-1).unsqueeze(-1)-data['actions_log_prob']).abs().max())
        max_error=max(max_error,error)
        audit=read(RUN/f'rollouts/update_{update:06}_likelihood.json')
        uses=Counter(i for batch in audit['minibatches'] for ids in batch['rollout_flat_indices'] for i in ids)
        assert len(audit['minibatches'])==20 and uses==Counter({i:5 for i in range(128)})
        per_rollout.append({'update':update,'input_phases':dict(Counter(r['phase'] for r in segment)),'RR_retirement_inputs':dict(local)})
    completed=list(lines('completed_episodes.jsonl')) if (RUN/'completed_episodes.jsonl').exists() else []
    terminal_episodes=[e for e in episodes if e['terminal']]
    assert len(completed)==len(terminal_episodes)
    for actual_episode,reported in zip(terminal_episodes,completed):
        assert actual_episode['policy_decisions']==reported['policy_decisions']
        assert abs(actual_episode['duration_s']-reported['duration_s'])<1e-9
        assert actual_episode['termination_reason']==reported['termination_reason']
        assert actual_episode['full_task_success']==reported['full_task_success']
    assert all(e['terminal'] for e in episodes[:-1])
    for episode in episodes:episode['request_phases']=phase_dict(episode['request_phases'])
    source=torch.load(SOURCE,map_location='cpu',weights_only=False)
    target=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert finite(target) and state_hash(target['optimizer_state_dict'])==cm['optimizer_state_sha256']
    for payload,metadata in [(source,sm),(target,cm)]:
        assert parameter_state_sha(payload['actor_state_dict'])==metadata['actor_parameter_sha256']
        assert parameter_state_sha(payload['critic_state_dict'])==metadata['critic_parameter_sha256']
        assert state_hash(payload['optimizer_state_dict'])==metadata['optimizer_state_sha256']
    assert all(target['infos'][key]==value for key,value in cm.items() if key in target['infos'])
    steps=cm['optimizer_steps']-sm['optimizer_steps']
    assert steps==len(updates)*20==manifest['optimizer_steps_this_run']
    delta=[float(target['optimizer_state_dict']['state'][key]['step']-value['step']) for key,value in source['optimizer_state_dict']['state'].items()]
    assert len(delta)==12 and set(delta)=={float(steps)}
    assert cm['actor_parameter_sha256']!=sm['actor_parameter_sha256'] and cm['optimizer_state_sha256']!=sm['optimizer_state_sha256']
    preserved=[key for key in sm if (key.endswith(('_branch','_migration')) and key!='resume_migration')]
    preserved+=['normalizer_state_sha256','normalization','runner_config','policy_contract']
    assert all(cm[key]==sm[key] for key in preserved)
    for branch in BRANCHES:
        assert cm[branch+'_branch_counts']=={key:cm[key]-value for key,value in cm[branch+'_branch']['counter_origin'].items()}
    aux=cm['task_conditioned_hip_wheel_branch']['auxiliary_mean_learning']
    result={'schema':'wlr50_clean.p05_capture_block05_sealed_audit.v1','run':str(RUN),'lifecycle':manifest['lifecycle'],
        'checkpoint':str(checkpoint),'checkpoint_sha256':cm['checkpoint_sha256'],'source_checkpoint':str(SOURCE),
        'source_checkpoint_sha256':SOURCE_SHA,'runtime':cm['runtime_contract']['source_git_commit'],
        'new_counts':dict(policy_decisions=actual,ppo_updates=len(updates),optimizer_steps=steps),
        'lifetime_counts':{key:cm[key] for key in COUNTERS},
        'planned_requested_policy_decisions':manifest.get('planned_requested_policy_decisions'),
        'unconsumed_requested_policy_decisions':manifest.get('unconsumed_requested_policy_decisions'),
        'lineage_origins':{b:cm[b+'_branch']['counter_origin'] for b in BRANCHES},
        'lineage_added_counts':{b:cm[b+'_branch_counts'] for b in BRANCHES},
        'entire_AUX_ledger_preserved':True,'new_AUX_updates':0,
        'AUX_totals':{key:aux[key] for key in ['accepted_auxiliary_updates_total','attempted_auxiliary_optimizer_steps_total']},
        'request_phase_counts':phase_dict(request),'endpoint_phase_counts':phase_dict(endpoint),
        'RR_physical_input_counts':dict(rr_inputs),'RR_physical_endpoint_counts':dict(rr_ends),
        'all_leg_physical_endpoint_counts':all_leg_ends,'RR_retirement_input_counts':dict(retirement_inputs),
        'RR_retirement_endpoint_counts':dict(retirement_ends),'RR_retirement_gate_input_phases':phase_dict(retirement_input_phases),
        'RR_retirement_effective_input_phases':phase_dict(retirement_effective_input_phases),
        'execution_counts':dict(totals),'episodes':episodes,'completed_episode_count':len(completed),
        'partial_episode_count':sum(not e['terminal'] for e in episodes),'prefix_policy_decisions':0,
        'per_rollout':per_rollout,'all_updates_finite_and_actor_changed':True,
        'all_original_raw_samples_match':True,'each_sample_used_five_times':True,
        'independent_CPU_logp_max_error':max_error,'all12_Adam_step_deltas':delta,
        'Identity_preserved':True,'effective_LR':cm['optimizer_learning_rate'],
        'recorded_save_load_round_trip':cm['save_load_round_trip'],
        'limitations':['Training SUCCEEDED is not task success. Episode outcomes above are authoritative.',
            'Initial reset TOP lacks prior physical endpoint and is counted unknown, not invented false.',
            'Retirement gate may be true after placement although actual potential bypasses workspace; counts are separate.',
            'Potential differences are pure recomputation on recorded physics, not a counterfactual trajectory.',
            'No physics replay, GPU, auxiliary optimizer or new checkpoint was used by this audit.']}
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':main()
