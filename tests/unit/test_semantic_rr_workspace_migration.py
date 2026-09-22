"""CPU/synthetic RR workspace migration; never physical or learning evidence."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT/'src'))


from wlr50_clean.ppo import semantic_rr_workspace_migration as m


def boundary():
    from wlr50_clean.ppo.semantic_policy_distribution import CONFIG_NAMES,policy_contract
    from wlr50_clean.ppo.semantic_training import semantic_runner_config
    from wlr50_clean.ppo.semantic_migration import continuation_topology
    paths = {f'configs/ppo_{m.EXPERIMENT}/{name}':'a'*64 for name in CONFIG_NAMES}
    selected = {name:{'path':f'configs/ppo_{m.EXPERIMENT}/{name}','sha256':'a'*64} for name in CONFIG_NAMES}
    old = {'experiment_id':m.EXPERIMENT,'semantic_version':'v3','selected_configuration':selected,
           'files':{**paths,m.SUPERVISOR:'b'*64},'source_git_commit':'a'*40,'runtime_content_sha256':'c'*64}
    new = deepcopy(old); new['source_git_commit']='f'*40; new['runtime_content_sha256']='f'*64
    new['files'].update({m.SUPERVISOR:'d'*64,m.TASK_SPEC:'e'*64,m.MODULE:'f'*64})
    new['selected_configuration']['stage_task_spec.yaml']['sha256']='e'*64
    canonical = policy_contract(m.P05_CAPTURE_POLICY,observation_layout=m.P05_CAPTURE_OBSERVATION_LAYOUT)
    sampling = 'P01_full_task_only_initial_version'
    meta = {'semantic_version':'v3','runtime_contract':old,'policy_contract':canonical,'seed':1001,
        'runner_config':semantic_runner_config(seed=1001,device='cpu',semantic_version='v3',
            policy_version=m.P05_CAPTURE_POLICY,observation_layout=m.P05_CAPTURE_OBSERVATION_LAYOUT),
        'sampling':sampling,'stage':'full_episode','curriculum_epoch':{},
        'implemented_reset_sampling':sampling,'phase_suffix_curriculum_implemented':False,
        'execution_topology':continuation_topology(sampling,None,observation_layout=m.P05_CAPTURE_OBSERVATION_LAYOUT),
        'global_policy_decisions':300000,'ppo_updates':2000,'optimizer_steps':40000,'optimizer_learning_rate':1e-5,
        'stage_requested_decisions':{'full_episode':4096,'phase_suffix':2048,'smoke':0},
        'new_mdp_origin_global_policy_decisions':10112,
        'p05_capture_assist_branch':{'counter_origin':dict(global_policy_decisions=199680,ppo_updates=1525,optimizer_steps=30500)},
        'p05_capture_assist_migration':{'immutable':'P05 source migration'},
        'capture_feedback_semantics_branch':{'feedback_revision':'hold_to_air_progress_window_v2',
            'counter_origin':dict(global_policy_decisions=203776,ppo_updates=1557,optimizer_steps=31140)},
        'capture_feedback_semantics_migration':{'immutable':'feedback v2 source migration'},
        'task_conditioned_hip_wheel_branch':{'auxiliary_mean_learning':{'accepted':7,'attempted':8}},
        'actor_parameter_sha256':'a'*64,'critic_parameter_sha256':'b'*64,'optimizer_state_sha256':'c'*64,
        'normalizer_state_sha256':'d'*64,'normalization':'Identity','training_rng_state':{}}
    source_spec={'episode_maximum_duration_s':200.,'physical_acceptance_version':'all_stage_v1',
                 'geometry':{'top_gap_min_m':-.015},'stages':{'P01':{'unchanged':True}}}
    target_spec={**deepcopy(source_spec),m.MODE_KEY:m.MODE}
    reviewed={path:sha for path,sha in new['files'].items() if old['files'].get(path)!=sha}
    return meta,old,new,reviewed,source_spec,target_spec


def factor(values):
    meta,old,new,review,source_spec,target_spec=values
    return m.rr_workspace_factor(meta,old,new,reason='synthetic narrow workspace test',
        reviewed_code_sha256=review,source_task_spec=source_spec,target_task_spec=target_spec)


def test_exact_factor_and_three_origins_without_new_credit():
    values=boundary();meta=values[0];f=factor(values)
    assert f['counter_origin']=={key:meta[key] for key in m.COUNTERS}
    assert f['reward_changed'] and not f['same_mdp_claimed']
    assert f['observation_semantics_changed'][0]['index']==17
    assert not f['same_physical_state_action_equivalence_claimed']
    assert f['observation_contract']['source_policy_contract']==f['observation_contract']['target_policy_contract']
    state={**meta,'rr_postcross_workspace_branch':{'counter_origin':f['counter_origin']}}
    assert m.rr_workspace_branch_counts(state)['rr_postcross_workspace_branch_counts']==dict.fromkeys(m.COUNTERS,0)
    assert m.rr_workspace_branch_counts({**state,'global_policy_decisions':300128,'ppo_updates':2001,'optimizer_steps':40020})[
        'rr_postcross_workspace_branch_counts']==dict(global_policy_decisions=128,ppo_updates=1,optimizer_steps=20)


@pytest.mark.parametrize('bad',['reward','task_duration','physical_rule','unknown_mode','source_mode','missing_review',
                              'kernel','controller','feedback','repeat','counter','LR'])
def test_strict_rejects_unreviewed_changes(bad):
    v=boundary();meta,old,new,review,s,t=v
    if bad=='reward':new['selected_configuration']['reward_config.yaml']['sha256']='8'*64
    elif bad=='task_duration':t['episode_maximum_duration_s']=250.
    elif bad=='physical_rule':t['geometry']['top_gap_min_m']=-.050
    elif bad=='unknown_mode':t[m.MODE_KEY]='anything'
    elif bad=='source_mode':s[m.MODE_KEY]=m.MODE
    elif bad=='missing_review':review.pop(m.SUPERVISOR)
    elif bad=='kernel':meta['policy_contract']['rho']=.8
    elif bad=='controller':new['files']['src/wlr50_clean/ppo/semantic_backend.py']='9'*64
    elif bad=='feedback':meta.pop('capture_feedback_semantics_migration')
    elif bad=='repeat':meta['rr_postcross_workspace_branch']={}
    elif bad=='counter':meta['global_policy_decisions']=-1
    else:meta['optimizer_learning_rate']=0
    with pytest.raises(ValueError):factor(v)


def test_cpu_synthetic_publish_generic_loader_and_later_save_preserve_all_state(tmp_path,monkeypatch):
    import torch
    from wlr50_clean.ppo import semantic_training as training,semantic_migration as shared
    from wlr50_clean.ppo.semantic_p05_capture_migration import _ObservationOnlyEnv
    assert not torch.cuda.is_available()
    def make():return training.construct_semantic_runner(_ObservationOnlyEnv(389),seed=1001,device='cpu',
        policy_version=m.P05_CAPTURE_POLICY,observation_layout=m.P05_CAPTURE_OBSERVATION_LAYOUT,initialize_actor=False)[0]
    runner=make()
    for parameter in (*runner.alg.actor.parameters(),*runner.alg.critic.parameters()):
        parameter.grad=torch.randn_like(parameter)*.01
    runner.alg.optimizer.step();runner.alg.optimizer.zero_grad()
    runner.alg.optimizer.param_groups[0]['lr']=1e-5;runner.alg.learning_rate=1e-5
    meta,old,new,review,s,t=boundary()
    source,_=training.save_semantic_checkpoint(runner,tmp_path/'synthetic_source.pt',meta)
    sm=shared.checkpoint_metadata(source)
    f=m.rr_workspace_factor(sm,old,new,reason='synthetic identity publication',reviewed_code_sha256=review,
        source_task_spec=s,target_task_spec=t)
    record={'schema':m.SCHEMA,'plan_path':str(tmp_path/'synthetic_mock_plan.json'),
        'source_checkpoint_sha256':sm['checkpoint_sha256'],'observation_dimension':389,'action_dimension':12,m.FACTOR_KEY:f}
    monkeypatch.setattr(m,'validate_rr_workspace_migration',lambda *a,**k:deepcopy(record))
    monkeypatch.setattr(shared,'validate_migration_plan',lambda *a,**k:deepcopy(record))
    receipt=m.publish_rr_workspace_checkpoint(source,new,tmp_path/'synthetic_mock_plan.json',tmp_path/'synthetic_target.pt')
    tm=shared.checkpoint_metadata(Path(receipt['checkpoint']))
    assert tm['rr_postcross_workspace_branch_counts']==dict.fromkeys(m.COUNTERS,0)
    for key in m.preserved_keys(sm):assert tm[key]==sm[key]
    fresh=make();loaded=training.load_semantic_checkpoint(fresh,Path(receipt['checkpoint']),contract=new,seed=1001)
    loaded.update(global_policy_decisions=300128,ppo_updates=2001,optimizer_steps=40020)
    # Synthetic counter-propagation fixture only, not an extra real PPO update.
    later,_=training.save_semantic_checkpoint(fresh,tmp_path/'synthetic_later.pt',loaded)
    lm=shared.checkpoint_metadata(later)
    assert lm['rr_postcross_workspace_migration']==record
    assert lm['rr_postcross_workspace_branch_counts']==dict(global_policy_decisions=128,ppo_updates=1,optimizer_steps=20)
    assert lm['p05_capture_assist_branch']==sm['p05_capture_assist_branch']
    assert lm['capture_feedback_semantics_branch']==sm['capture_feedback_semantics_branch']
    assert lm['task_conditioned_hip_wheel_branch']==sm['task_conditioned_hip_wheel_branch']
