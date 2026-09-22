"""Synthetic persistence and metadata tests only; never optimize or encode media."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parents[3]/'src'))
import limited_aux_provenance as v
import limited_aux_media as media


@pytest.fixture
def proof(tmp_path,monkeypatch):
    import torch
    from wlr50_clean.ppo import semantic_training as t
    from wlr50_clean.ppo.semantic_policy_distribution import policy_contract, TASK_CONDITIONED_HIP_WHEEL_POLICY
    from wlr50_clean.ppo.semantic_migration import continuation_topology
    monkeypatch.setattr(v,'OUT',tmp_path)
    history=tmp_path/'checkpoints/history';history.mkdir(parents=True)
    source=history/'checkpoint_step_000000128.pt';aux=history/'checkpoint_aux_flmean_SYNTHETIC.pt'
    successor=history/'checkpoint_step_000000256.pt'
    sd={'mlp.0.weight':torch.zeros(2,2),'mlp.0.bias':torch.zeros(2),
        'mlp.4.weight':torch.zeros(24,256),'mlp.4.bias':torch.zeros(24)}
    adam={'state':{0:{'step':torch.tensor(20.),'exp_avg':torch.tensor(.2),'exp_avg_sq':torch.tensor(.04)}},
          'param_groups':[{'params':[0],'lr':1e-5}]}
    origin={k:0 for k in v.COUNTERS}
    meta={'schema':'wlr50_clean.semantic_checkpoint.v1','semantic_version':'v3','seed':1001,'stage':'full_episode',
        'runtime_contract':{'experiment_id':'task_conditioned_hip_wheel_v1'},
        'policy_contract':policy_contract(TASK_CONDITIONED_HIP_WHEEL_POLICY,observation_layout='diagonal_transfer_state_v1'),
        'runner_config':t.semantic_runner_config(seed=1001,device='cpu',semantic_version='v3',
            policy_version=TASK_CONDITIONED_HIP_WHEEL_POLICY,observation_layout='diagonal_transfer_state_v1'),
        'sampling':'P01_full_task_only_initial_version',
        'execution_topology':continuation_topology('P01_full_task_only_initial_version',None,
            observation_layout='diagonal_transfer_state_v1'),
        'global_policy_decisions':128,'ppo_updates':1,'optimizer_steps':20,
        'stage_requested_decisions':{'full_episode':128,'phase_suffix':0,'smoke':0},
        v.BRANCH:{'schema':'wlr50_clean.task_conditioned_hip_wheel_branch.v1',
            'branch_id':'task_conditioned_hip_wheel_v1','counter_origin':origin},
        'task_conditioned_hip_wheel_branch_counts':{'global_policy_decisions':128,'ppo_updates':1,'optimizer_steps':20},
        'optimizer_learning_rate':1e-5,'normalizer_state_sha256':t.state_hash({'actor':{},'critic':{}}),
        'training_rng_state':{'SYNTHETIC':True}}
    def writecp(path,info,actor):
        info=copy.deepcopy(info)
        for role,state in (('actor',actor),('critic',sd)):
            info[role+'_parameter_sha256']=t.parameter_hash(SimpleNamespace(named_parameters=lambda:list(state.items())))
        info['optimizer_state_sha256']=t.state_hash(adam)
        torch.save({'infos':info,'actor_state_dict':actor,'critic_state_dict':sd,'optimizer_state_dict':adam},path)
        side=path.with_name(path.stem+'_manifest.json')
        side.write_text(json.dumps({**info,'checkpoint_path':str(path),'checkpoint_sha256':v.sha(path),
            'save_load_round_trip':True}),encoding='utf-8')
        return info
    source_infos=writecp(source,meta,sd)
    source_binding={'path':str(source),'sha256':v.sha(source),'manifest_sha256':v.sha(source.with_name(source.stem+'_manifest.json'))}
    selection=tmp_path/'SYNTHETIC_selection.json';selection.write_text('{"synthetic":true}')
    data={'selection_path':str(selection),'selection_sha256':v.sha(selection),'label_scope':v.SCOPE,
        'PPO_credit':0,'capture_labels':False,'other_channels_supervised':False,'helper_bundle_sha256':v.APPROVED_HELPERS}
    changed={k:x.clone() for k,x in sd.items()};changed['mlp.4.weight'][0,0]=.01
    fit={'schema':'wlr50_clean.finite_FL_conditional_mean_aux.v1','label_scope':v.SCOPE,'budget':v.BUDGET,
        'optimized_parameters':['actor.mlp.4.weight[0,:]','actor.mlp.4.bias[0]'],'optimized_scalar_count':257,
        'auxiliary_optimizer':'independent two-leaf SGD; momentum=0; weight_decay=0',
        'PPO_decisions_added':0,'PPO_updates_added':0,'PPO_optimizer_steps_added':0,
        'MDP_or_control_changed':False,'kernel_or_sigma_changed':False,'teacher_deployed':False,'physical_success_claimed':False,
        'training_rng_preserved':True,'fresh_PPO_rollout_required':True,
        'steps':[{'attempt':i+1,'learning_rate':.05*(16-i)/16,'loss_before':1/(i+1),'loss_after_candidate':1/(i+2),'accepted':True,
            'max_train_request_shift_deg':.3,'max_holdout_request_shift_deg':.1,'maximum_per_state_conditional_kl':.03,'selected_gradient_l2':.1} for i in range(16)],
        'accepted_auxiliary_updates':16,'attempted_auxiliary_optimizer_steps':16,'stop_reason':'finite_budget_exhausted',
        'actor_parameter_sha256_before':source_infos['actor_parameter_sha256'],
        'actor_parameter_sha256_after':t.parameter_hash(SimpleNamespace(named_parameters=lambda:list(changed.items()))),
        'PPO_Adam_preserved_sha256':source_infos['optimizer_state_sha256'],'PPO_LR_preserved':1e-5}
    event={'event_index':1,'kind':'limited_supervised_FL_conditional_mean_not_PPO','source_checkpoint':source_binding,
        'helper_sha256':v.APPROVED_HELPERS['finite_auxiliary_mean.py'],'data_receipt':data,'report':fit,
        'PPO_counters_unchanged':{k:source_infos[k] for k in v.COUNTERS},
        'stage_requested_decisions_unchanged':source_infos['stage_requested_decisions']}
    ledger={'schema':'wlr50_clean.auxiliary_mean_learning_ledger.v1','events':[event],
        'accepted_auxiliary_updates_total':16,'attempted_auxiliary_optimizer_steps_total':16,
        'training_lineage_label':'PPO_plus_explicit_finite_auxiliary_mean_supervision'}
    info=copy.deepcopy(source_infos);info[v.BRANCH][v.LEDGER]=ledger;info['stage']='auxiliary_mean_pretrain_not_PPO'
    info['resume_ancestry']={'operation':'explicit_auxiliary_mean_supervision_not_PPO','source_checkpoint':source_binding,
        'source_actor_parameter_sha256':source_infos['actor_parameter_sha256'],'source_runtime_contract':meta['runtime_contract'],
        **{'source_'+k:source_infos[k] for k in v.COUNTERS}}
    info['resume_source_checkpoint']={'checkpoint':str(source),'checkpoint_sha256':source_binding['sha256'],
        'manifest':str(source.with_name(source.stem+'_manifest.json')),'manifest_sha256':source_binding['manifest_sha256']}
    aux_infos=writecp(aux,info,changed)
    binding={'source_checkpoint':source_binding,'selection_sha256':data['selection_sha256'],
        'helper_sha256':v.APPROVED_HELPERS,'runtime_contract_sha256':v.helpers_digest if hasattr(v,'helpers_digest') else ''}
    from wlr50_clean.ppo.semantic_migration import digest
    binding['runtime_contract_sha256']=digest(meta['runtime_contract'])
    inspection=tmp_path/'SYNTHETIC_inspection.json'
    inspection.write_text(json.dumps({'mode':'read_only_current_checkpoint_on_real_saved_states','binding':binding,'auxiliary_updates':0}))
    execution={'schema':'wlr50_clean.finite_auxiliary_candidate_receipt.v1','mode':'explicit_finite_auxiliary_not_PPO',
        'fit':fit,'auxiliary_updates':16,'data_receipt':data,'PPO_decisions_added':0,'PPO_updates_added':0,
        'PPO_optimizer_steps_added':0,'teacher_deployed':False,'capture_claimed':False,'physical_success_claimed':False,
        'binding':binding,'original_PPO_counters':{k:source_infos[k] for k in v.COUNTERS},
        'data_runtime_compatibility':{'SYNTHETIC_CHECK_ONLY':True},
        'reviewed_inspection':{'path':str(inspection),'sha256':v.sha(inspection)},
        'auxiliary_checkpoint':{'path':str(aux),'manifest':str(aux.with_name(aux.stem+'_manifest.json')),'sha256':v.sha(aux)}}
    receipt=tmp_path/'SYNTHETIC_execution.json';receipt.write_text(json.dumps(execution))
    info=copy.deepcopy(aux_infos);info['stage']='full_episode'
    info.pop('resume_source_checkpoint')  # Official train rebuilds saved infos for fresh PPO.
    info.update(global_policy_decisions=256,ppo_updates=2,optimizer_steps=40)
    info['stage_requested_decisions']['full_episode']=256
    info['task_conditioned_hip_wheel_branch_counts']={k:info[k] for k in v.COUNTERS}
    info['resume_ancestry']={'source_checkpoint':{'checkpoint':str(aux),'checkpoint_sha256':v.sha(aux),
        'manifest':str(aux.with_name(aux.stem+'_manifest.json')),'manifest_sha256':v.sha(aux.with_name(aux.stem+'_manifest.json'))},
        'source_actor_parameter_sha256':aux_infos['actor_parameter_sha256'],'source_runtime_contract':meta['runtime_contract'],
        **{'source_'+k:aux_infos[k] for k in v.COUNTERS}}
    after_ppo={k:x.clone() for k,x in changed.items()};after_ppo['mlp.4.weight'][1,0]=.02
    successor_infos=writecp(successor,info,after_ppo)
    def data_load(_):return None,None,None,{k:value for k,value in data.items() if k!='helper_bundle_sha256'}
    fake=SimpleNamespace(load_reviewed_selection=data_load,verify_data_compatibility=lambda *a:{'SYNTHETIC_CHECK_ONLY':True})
    monkeypatch.setattr(v,'checked_helpers',lambda:fake)
    return locals()


def test_immediate_aux_and_fresh_PPO_inheritance_both_validate(proof):
    f=proof
    immediate=v.validate_auxiliary_checkpoint(f['aux'],f['receipt'])
    successor=v.validate_auxiliary_checkpoint(f['successor'],f['receipt'])
    assert immediate['method']==successor['method']=='PPO_PLUS_LIMITED_AUX'
    assert immediate['PPO_counters']['ppo_updates']==1 and successor['PPO_counters']['ppo_updates']==2
    assert immediate['accepted_auxiliary_updates_total']==successor['accepted_auxiliary_updates_total']==16
    assert len(successor['verified_lineage'])==2 and successor['auxiliary_PPO_credit']==0


@pytest.mark.parametrize('fault',['missing_ledger','two_events','stale_report','stale_inspection','helper','data',
    'PPO_credit','budget','trust','linear_lr','count','source_hash','aux_hash','wrong_channel',
    'successor_ledger','successor_ancestry','successor_counts','hidden_aux','ledger_extra_aux','event_extra_aux','nonfinite'])
def test_provenance_refusals(proof,fault):
    f=proof
    if fault in ('stale_report','PPO_credit','budget','trust','linear_lr','count','source_hash','aux_hash','helper','data'):
        x=copy.deepcopy(f['execution'])
        if fault=='stale_report':x['fit']['actor_parameter_sha256_before']='bad'
        elif fault=='PPO_credit':x['PPO_updates_added']=1
        elif fault=='budget':x['fit']['budget']['max_steps']=17
        elif fault=='trust':x['fit']['steps'][0]['max_train_request_shift_deg']=2.
        elif fault=='linear_lr':x['fit']['steps'][0]['learning_rate']=.04
        elif fault=='count':x['auxiliary_updates']=2
        elif fault=='source_hash':x['binding']['source_checkpoint']['sha256']='bad'
        elif fault=='aux_hash':x['auxiliary_checkpoint']['sha256']='bad'
        elif fault=='helper':x['binding']['helper_sha256']['finite_auxiliary_mean.py']='bad'
        elif fault=='data':x['data_receipt']['selection_sha256']='bad'
        f['receipt'].write_text(json.dumps(x))
    elif fault=='stale_inspection':f['inspection'].write_text('{}')
    else:
        x=copy.deepcopy(f['successor_infos']);actor=f['after_ppo']
        if fault=='missing_ledger':x[v.BRANCH].pop(v.LEDGER)
        elif fault=='two_events':x[v.BRANCH][v.LEDGER]['events']*=2
        elif fault=='successor_ledger':x[v.BRANCH][v.LEDGER]['accepted_auxiliary_updates_total']=3
        elif fault=='successor_ancestry':x['resume_ancestry']['source_checkpoint']['checkpoint_sha256']='bad'
        elif fault=='successor_counts':x['optimizer_steps']=41
        elif fault=='hidden_aux':x['resume_ancestry']['unreviewed_auxiliary']={'updates':9}
        elif fault=='ledger_extra_aux':x[v.BRANCH][v.LEDGER]['unreviewed_auxiliary']={'updates':9}
        elif fault=='event_extra_aux':x[v.BRANCH][v.LEDGER]['events'][0]['another_auxiliary']={'updates':9}
        elif fault=='nonfinite':
            actor={k:value.clone() for k,value in actor.items()};actor['mlp.4.bias'][0]=float('nan')
        elif fault=='wrong_channel':
            bad={k:value.clone() for k,value in f['changed'].items()};bad['mlp.4.weight'][1,0]=.02
            f['writecp'](f['aux'],f['aux_infos'],bad)
        if fault!='wrong_channel':f['writecp'](f['successor'],x,actor)
    with pytest.raises((ValueError,KeyError)):
        v.validate_auxiliary_checkpoint(f['successor'],f['receipt'])


@pytest.mark.parametrize('field,value',[('accepted_auxiliary_updates',True),('attempted_auxiliary_optimizer_steps',0)])
def test_step_report_requires_independent_integer_counts(proof,field,value):
    fit=copy.deepcopy(proof['fit']);fit[field]=value
    with pytest.raises(ValueError):v.validate_fit(fit)


@pytest.mark.parametrize('field,value',[('max_train_request_shift_deg',1.001),('max_holdout_request_shift_deg',.251),
    ('maximum_per_state_conditional_kl',.101),('learning_rate',.04),('loss_after_candidate',2.),
    ('selected_gradient_l2',float('nan')),('attempt',2)])
def test_direct_finite_step_guards(proof,field,value):
    fit=copy.deepcopy(proof['fit']);fit['steps'][0][field]=value
    with pytest.raises(ValueError):v.validate_fit(fit)


def test_known_ledger_is_only_recursive_auxiliary_exception():
    v.reject_unknown_auxiliary({v.BRANCH:{v.LEDGER:{'accepted_auxiliary_updates_total':1}}})
    with pytest.raises(ValueError):v.reject_unknown_auxiliary({v.BRANCH:{v.LEDGER:{}}},allow_ledger=False)
    with pytest.raises(ValueError):v.reject_unknown_auxiliary({'old':[{'auxiliary_mean_learning':None}]})


def test_fit_unknown_nested_auxiliary_is_rejected(proof):
    fit=copy.deepcopy(proof['fit']);fit['before']={'prediction':{'unreviewed_auxiliary':9}}
    with pytest.raises(ValueError):v.validate_fit(fit)


def test_actual_sealed_helper_import_is_read_only_and_hash_bound():
    module=v.checked_helpers()
    assert module.a.Budget().max_steps==16
    assert module.a.Budget().learning_rate==.05


def test_AST_adapter_isolated_labels_and_receipt_no_encoding(monkeypatch):
    adapter=media.make_adapter(Path('not_an_existing_aux_report.json'))
    assert adapter.export.__globals__ is not vars(media.r)
    assert adapter.export.__globals__['output_name']('C','deterministic_conditional_mean',
        {'saved_global_policy_decisions':192512})=='PPO_PLUS_LIMITED_AUX_CP192512_deterministic_P01_full.mp4'
    constants=adapter.export.__code__.co_consts
    assert 'PPO + LIMITED AUX CP' in constants and 'PPO FULL12 CP' not in constants
    assert media.SCHEMA in constants and 'learning_method' in constants
    assert media.r.sha256(media.OUT/'review_video.py')==media.REVIEW_HASH
    assert media.r.sha256(media.OUT/'pair_quantity_only_v2.py')==media.QUANTITY_HASH
    assert len((HERE/'limited_aux_media.py').read_text().splitlines())<150


def test_invalid_aux_stops_export_before_directory_or_encoder(tmp_path,monkeypatch):
    monkeypatch.setattr(media.r,'sealed_source',lambda source:{'role':'C','checkpoint':{'checkpoint':'synthetic'}})
    def reject(*args):raise ValueError('synthetic invalid auxiliary proof')
    monkeypatch.setattr(media,'validate_auxiliary_checkpoint',reject)
    def noencode(*args,**kwargs):raise AssertionError('encoder must not run')
    monkeypatch.setattr(media.r.base,'export',noencode)
    adapter=media.make_adapter(tmp_path/'report.json');destination=tmp_path/'must_not_exist'
    with pytest.raises(ValueError,match='synthetic invalid auxiliary proof'):adapter.export(tmp_path,destination)
    assert not destination.exists()
