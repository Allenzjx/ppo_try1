"""Synthetic CPU only: finite core and immutable-checkpoint publication contract."""
from copy import deepcopy
from dataclasses import asdict,replace
from pathlib import Path
import json
import pytest
import torch
import rr_mean_rehearsal as m
import fit_rr_mean_core as core
import execute_rr_mean_rehearsal as cli
from test_rr_mean_rehearsal import data,objective,budget
from wlr50_clean.ppo.semantic_p05_capture_migration import _ObservationOnlyEnv


@pytest.fixture(autouse=True)
def cpu_only():
    assert not torch.cuda.is_available(),'CUDA_VISIBLE_DEVICES=-1 required'
    before=torch.get_rng_state();threads=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_rng_state(before);torch.set_num_threads(threads)


def runner():
    run,_=cli.training.construct_semantic_runner(_ObservationOnlyEnv(389),seed=1001,device='cpu',initialize_actor=False,
        policy_version=cli.P05_CAPTURE_POLICY,observation_layout=cli.P05_CAPTURE_OBSERVATION_LAYOUT)
    # Populate synthetic Adam state before fit; never an actual training source.
    for p in [*run.alg.actor.parameters(),*run.alg.critic.parameters()]:p.grad=torch.linspace(-.03,.05,p.numel()).reshape_as(p)
    run.alg.optimizer.step();run.alg.optimizer.zero_grad(set_to_none=True)
    for p in run.alg.actor.parameters():p.grad=torch.full_like(p,.001)
    return run


def protected(run):
    return {'critic':cli.training.parameter_hash(run.alg.critic),
        'Adam':cli.training.state_hash(run.alg.optimizer.state_dict()),
        'Identity':cli.training.state_hash(cli.training._normalizers(run)),
        'LR':(run.alg.learning_rate,cli.training.optimizer_learning_rate(run)),
        'RNG':cli.training.capture_training_rng_state(seed=1001),
        'config':deepcopy(run._semantic_runner_config)}


def gradients(run):
    return {role:{key:None if value.grad is None else value.grad.detach().clone()
        for key,value in getattr(run.alg,role).named_parameters()} for role in ('actor','critic')}


def assert_gradients_equal(run,before):
    for role,values in before.items():
        for key,value in getattr(run.alg,role).named_parameters():
            assert value.grad is None if values[key] is None else (
                value.grad is not None and torch.equal(value.grad,values[key]))


def fit(run,d,**kwargs):return core.fit_mean_head(run,d,objective=objective(),budget=budget(**kwargs),authorized=True)


def test_two_accept_only_existing_mean_rows_all_other_states_gradients_preserved(monkeypatch):
    run=runner();d=data(run.alg.actor);before=deepcopy(run.alg.actor.state_dict());other=protected(run)
    grads={k:p.grad.clone() for k,p in run.alg.actor.named_parameters()}
    def forbid(*args,**kwargs):raise AssertionError('PPO optimizer called')
    monkeypatch.setattr(run.alg.optimizer,'step',forbid)
    report=fit(run,d)
    assert report['accepted_auxiliary_updates']==report['attempted_auxiliary_optimizer_steps']==2
    assert 0<report['actually_changed_scalar_count']<=3084
    assert report['same_input_sigma_exact'] and not report['P03plus_mean_bitwise_invariance_claimed']
    for k,v in run.alg.actor.state_dict().items():
        assert torch.equal(v[12:],before[k][12:]) if k in ('mlp.4.weight','mlp.4.bias') else torch.equal(v,before[k])
    assert protected(run)==other and all(torch.equal(p.grad,grads[k]) for k,p in run.alg.actor.named_parameters())
    assert report['PPO_decisions_added']==report['PPO_updates_added']==report['PPO_optimizer_steps_added']==0


def test_first_reject_is_finite_and_preserves_source():
    run=runner();d=data(run.alg.actor);before=cli.training.parameter_hash(run.alg.actor);other=protected(run)
    report=fit(run,d,max_attempts=32,maximum_protection_request_shift_full12=(1e-14,)*12)
    assert report['accepted_auxiliary_updates']==0 and report['attempted_auxiliary_optimizer_steps']==1
    assert report['steps'][0]['rejected_full3084_leaf_discarded']
    assert cli.training.parameter_hash(run.alg.actor)==before and protected(run)==other


def test_second_reject_keeps_first_accepted_leaf_not_source(monkeypatch):
    run=runner();d=data(run.alg.actor);saved=deepcopy(run.alg.actor.state_dict())
    expected=fit(run,d,max_attempts=1);expected_state=deepcopy(run.alg.actor.state_dict());run.alg.actor.load_state_dict(saved)
    original=m.guard_candidate;calls=0
    def reject_second(*args,**kwargs):
        nonlocal calls
        calls+=1; result=original(*args,**kwargs)
        if calls==2:result['accepted_by_trust_only']=False;result['rejection_reasons'].append('synthetic_second_rejection')
        return result
    monkeypatch.setattr(m,'guard_candidate',reject_second)
    report=fit(run,d,max_attempts=32)
    assert report['accepted_auxiliary_updates']==1 and report['attempted_auxiliary_optimizer_steps']==2
    assert all(torch.equal(v,expected_state[k]) for k,v in run.alg.actor.state_dict().items())
    assert report['actor_parameter_sha256_after']==expected['actor_parameter_sha256_after']


@pytest.mark.parametrize('bad',['trunk','sigma','gradient_nan','RNG','critic','Adam','LR',
                              'normalizer','config','actor_gradient','critic_gradient'])
def test_fail_closed_full_state_restored_on_unexpected_state_or_nonfinite(monkeypatch,bad):
    run=runner();d=data(run.alg.actor);before=cli.training.parameter_hash(run.alg.actor)
    # Nonempty critic gradients prove restoration beyond the all-None case.
    for p in run.alg.critic.parameters():p.grad=torch.full_like(p,.002)
    other=protected(run);grads=gradients(run)
    normalizers=(type(run.alg.actor.obs_normalizer),type(run.alg.critic.obs_normalizer))
    original=m.loss_terms;calls=0
    def corrupt(*args,**kwargs):
        nonlocal calls
        result=original(*args,**kwargs);calls+=1
        if calls==2:
            if bad=='gradient_nan':result['total']=result['total']*float('nan')
            elif bad=='RNG':torch.rand(1)
            elif bad=='LR':
                run.alg.learning_rate+=.1
                for group in run.alg.optimizer.param_groups:group['lr']+=.2
            elif bad=='normalizer':run.alg.actor.obs_normalizer=torch.nn.Linear(389,389)
            elif bad=='config':run._semantic_runner_config['algorithm']['learning_rate']+=.3
            elif bad in ('actor_gradient','critic_gradient'):
                model=run.alg.actor if bad=='actor_gradient' else run.alg.critic
                next(model.parameters()).grad.add_(.5)
            else:
                with torch.no_grad():
                    if bad=='trunk':run.alg.actor.mlp[0].bias.add_(.1)
                    elif bad=='sigma':run.alg.actor.mlp[4].bias[12:].add_(.1)
                    elif bad=='critic':next(run.alg.critic.parameters()).add_(.1)
                    elif bad=='Adam':next(iter(run.alg.optimizer.state.values()))['exp_avg'].add_(.1)
        return result
    monkeypatch.setattr(m,'loss_terms',corrupt)
    with pytest.raises((ValueError,RuntimeError)):fit(run,d)
    assert cli.training.parameter_hash(run.alg.actor)==before
    assert protected(run)==other
    assert_gradients_equal(run,grads)
    assert (type(run.alg.actor.obs_normalizer),type(run.alg.critic.obs_normalizer))==normalizers


def test_initial_loss_exception_is_inside_full_rollback(monkeypatch):
    run=runner();d=data(run.alg.actor)
    for p in run.alg.critic.parameters():p.grad=torch.full_like(p,.002)
    actor=cli.training.parameter_hash(run.alg.actor);other=protected(run);grads=gradients(run)
    def injected_initial_exception(*args,**kwargs):
        torch.rand(1)
        with torch.no_grad():next(run.alg.critic.parameters()).add_(.2)
        next(run.alg.actor.parameters()).grad.add_(.3)
        raise RuntimeError('synthetic initial-loss fault')
    monkeypatch.setattr(m,'loss_terms',injected_initial_exception)
    with pytest.raises(RuntimeError,match='synthetic initial-loss fault'):fit(run,d)
    assert cli.training.parameter_hash(run.alg.actor)==actor and protected(run)==other
    assert_gradients_equal(run,grads)


@pytest.mark.parametrize('bad',['permission','storage','transition'])
def test_no_authorization_or_unfinished_rollout_refused(bad):
    run=runner();d=data(run.alg.actor)
    if bad=='storage':run.alg.storage.step=1
    if bad=='transition':run.alg.transition.actions=torch.zeros(1,12)
    with pytest.raises(ValueError):core.fit_mean_head(run,d,objective=objective(),budget=budget(),authorized=bad!='permission')


def metadata():
    origin=dict(global_policy_decisions=0,ppo_updates=0,optimizer_steps=0)
    events=[{'event_index':i,'fit_report':{'accepted_auxiliary_updates':32,'attempted_auxiliary_optimizer_steps':32},
             'original_immutable_event':i} for i in (1,2,3)]
    return {'seed':1001,'runtime_contract':{'synthetic_CPU_fixture_not_real_physics':True},
        'global_policy_decisions':128,'ppo_updates':1,'optimizer_steps':20,
        'stage_requested_decisions':{'smoke':0,'full_episode':128,'phase_suffix':0},
        'stage':'full_episode','sampling':'P01_full_task_only_initial_version',
        'execution_topology':cli.semantic_migration.continuation_topology('P01_full_task_only_initial_version',None,
            observation_layout=cli.P05_CAPTURE_OBSERVATION_LAYOUT),
        'new_mdp_origin_global_policy_decisions':0,
        'task_conditioned_hip_wheel_branch':{'counter_origin':deepcopy(origin),'auxiliary_mean_learning':{
            'accepted_auxiliary_updates_total':7,'attempted_auxiliary_optimizer_steps_total':8}},
        'p05_capture_assist_branch':{'counter_origin':deepcopy(origin)},'capture_feedback_semantics_branch':{'counter_origin':deepcopy(origin)},
        'capture_feedback_semantics_migration':{'original':True},'p05_capture_assist_migration':{'original':True},
        'rr_postcross_workspace_migration':{'original':True},
        'rr_postcross_workspace_branch':{'schema':'wlr50_clean.rr_postcross_workspace_same389.v1',
            'semantics':'current_qualified_RR_over_top_receiver_retirement_v1','counter_origin':deepcopy(origin),
            'migration_added_updates':0,'source_checkpoint_sha256':'a'*64,
            cli.LEDGER_KEY:{'schema':cli.LEDGER_SCHEMA,'events':events,'accepted_auxiliary_updates_total':96,
                           'attempted_auxiliary_optimizer_steps_total':96,'training_lineage_label':'unchanged'}}}


def binding(infos,d):
    return {'source_checkpoint':{'path':'synthetic.pt','sha256':'b'*64,'PPO_counters':{k:infos[k] for k in cli.COUNTERS}},
        'helpers':{'synthetic':'c'*64},'data_receipt_sha256':cli.digest(d['receipt']),
        'objective':{'path':'synthetic_objective.json','sha256':'d'*64,
                     'envelope':{'schema':cli.SCHEMA+'.objective','objective':cli.json_value(asdict(objective()))}}}


def budget_receipt():return {'path':'synthetic_budget.json','sha256':'e'*64,'envelope':{'budget':cli.json_value(asdict(budget()))}}


def test_append_event4_preserves_old_ledger7_8_and_all_origins():
    run=runner();d=data(run.alg.actor);report=fit(run,d);infos=metadata();saved=deepcopy(infos)
    out=cli.append_event(infos,report=report,binding=binding(infos,d),data_receipt=d['receipt'],budget_receipt=budget_receipt())
    assert infos==saved and out['task_conditioned_hip_wheel_branch']==infos['task_conditioned_hip_wheel_branch']
    ledger=out['rr_postcross_workspace_branch'][cli.LEDGER_KEY]
    assert ledger['events'][:3]==infos['rr_postcross_workspace_branch'][cli.LEDGER_KEY]['events']
    assert ledger['accepted_auxiliary_updates_total']==98 and ledger['training_lineage_label']=='unchanged'
    event=ledger['events'][3]
    assert event['event_index']==4 and event['kind']==cli.EVENT_KIND
    assert event['positive_phase_scope']==['P09','P10','P11'] and len(event['affected_mean_phase_scope'])==13
    assert event['same_input_sigma_fixed'] and not event['mean_and_log_sigma_may_change']
    credit=event['auxiliary_credit_breakdown']
    assert credit['inherited_front_auxiliary']=={'accepted':96,'attempted':96}
    assert credit['this_RR_auxiliary']=={'accepted':2,'attempted':2}
    assert credit['cumulative_mixed_ledger_auxiliary']=={'accepted':98,'attempted':98}
    assert credit['inherited_historical_auxiliary']=={'accepted':7,'attempted':8}
    assert event['fit_report_sha256']==cli.digest(event['fit_report'])
    with pytest.raises(ValueError):cli.append_event(out,report=report,binding=binding(out,d),data_receipt=d['receipt'],budget_receipt=budget_receipt())


def test_synthetic_official_save_fresh_reload_no_latest_aliases(tmp_path,monkeypatch):
    monkeypatch.setattr(cli,'OUT',tmp_path)
    run=runner();d=data(run.alg.actor)
    cp,_=cli.training.save_semantic_checkpoint(run,tmp_path/'synthetic_source.pt',metadata())
    meta=cli.semantic_migration.checkpoint_metadata(cp)
    infos=cli.training.load_semantic_checkpoint(run,cp,contract=meta['runtime_contract'],seed=1001)
    report=fit(run,d)
    out=cli.save_candidate(run,tmp_path/'checkpoint_aux_meanrr_step_000214400_v4.pt',infos,data=d,report=report,
        binding=binding(infos,d),budget_receipt=budget_receipt())
    assert out['independent_official_reload_verified'] and out['source_device_preserved']=='cpu'
    assert not out['latest_pointer_published']
    new=cli.semantic_migration.checkpoint_metadata(Path(out['path']))
    assert all(new[k]==infos[k] for k in cli.COUNTERS)
    for key in ('optimizer_state_sha256','critic_parameter_sha256','normalizer_state_sha256','training_rng_state'):
        assert new[key]==infos[key]
    assert len(new['rr_postcross_workspace_branch'][cli.LEDGER_KEY]['events'])==4
    assert not (tmp_path/'checkpoint_last_pointer.json').exists()


def test_cpu_inspection_restores_rng_and_does_not_load_or_step_PPO_Adam(tmp_path,monkeypatch):
    run=runner();d=data(run.alg.actor);cp,_=cli.training.save_semantic_checkpoint(run,tmp_path/'source.pt',metadata())
    meta=cli.semantic_migration.checkpoint_metadata(cp);rng=cli.training.capture_training_rng_state(seed=1001)
    def forbid(*args,**kwargs):raise AssertionError('readonly inspection touched PPO Adam/fit')
    monkeypatch.setattr(torch.optim.Adam,'step',forbid);monkeypatch.setattr(torch.optim.Adam,'load_state_dict',forbid)
    monkeypatch.setattr(core,'fit_mean_head',forbid)
    report=cli.cpu_inspection(cp,meta,d,objective())
    assert report['optimizer_steps_performed']==0 and report['actor_sha256_unchanged']==meta['actor_parameter_sha256']
    assert cli.training.capture_training_rng_state(seed=1001)==rng


@pytest.mark.parametrize('bad',['binding','auto_retry','max_attempts','missing_lr'])
def test_explicit_budget_rejects_stale_or_automatic(monkeypatch,bad):
    b={'synthetic':True};value={'schema':cli.SCHEMA+'.explicit_budget','binding':b,'budget':asdict(budget())}
    if bad=='binding':value['binding']={'stale':True}
    if bad=='auto_retry':value['retry']=True
    if bad=='max_attempts':value['budget']['max_attempts']=33
    if bad=='missing_lr':del value['budget']['learning_rate']
    monkeypatch.setattr(cli,'read',lambda _:value)
    with pytest.raises((ValueError,TypeError)):cli.load_budget('unused',b)


def test_inspection_binding_rejects_stale_source_and_false_exact_mean(monkeypatch):
    base={'source_checkpoint':{'actor_sha256':'a'},'objective':{'envelope':{'objective':{}}}}
    report={'schema':cli.SCHEMA+'.readonly','binding':base,'inspection_only':True,'optimizer_steps_performed':0,
        'PPO_updates_added':0,'checkpoint_written':False,'inspection':{'schema':cli.SCHEMA+'.inspection',
        'optimized_parameters':m.PARAMETERS,'optimized_scalar_count':3084,'optimizer_steps_performed':0,
        'same_input_sigma_exact_by_frozen_trunk_and_sigma_rows':True,'P03plus_mean_bitwise_invariance_claimed':True}}
    monkeypatch.setattr(cli,'read',lambda _:report)
    with pytest.raises(ValueError):cli.verify_inspection('unused',base)
    with pytest.raises(ValueError):cli.verify_inspection('unused',{'stale':True})


@pytest.mark.parametrize('field,value',[('positive_label_phases',['P01','P02']),
    ('actual_protection_phases',[3,4,5,6]),('P10_independent_validation_rows',1),
    ('single_historical_trajectory_train_validation_correlated',False)])
def test_event4_rejects_front_scope_or_invented_independent_validation(field,value):
    run=runner();d=data(run.alg.actor);report=fit(run,d);report[field]=value
    with pytest.raises(ValueError):cli._validate_fit(report)


def test_source_is_fixed_and_old_three_event_totals_cannot_be_relabelled():
    contract={'source_git_commit':cli.HEAD}
    with pytest.raises(ValueError,match='CP214400'):
        cli.prepare_context('unused',{'runtime_contract':contract,'checkpoint_sha256':'0'*64,
            'global_policy_decisions':214400},contract,objective_path='unused')
    run=runner();d=data(run.alg.actor);report=fit(run,d);infos=metadata()
    infos['rr_postcross_workspace_branch'][cli.LEDGER_KEY]['accepted_auxiliary_updates_total']=95
    with pytest.raises(ValueError,match='96/96'):
        cli.append_event(infos,report=report,binding=binding(infos,d),data_receipt=d['receipt'],budget_receipt=budget_receipt())


def test_destination_is_exact_unique_v4_without_alias_publication(tmp_path,monkeypatch):
    monkeypatch.setattr(cli,'OUT',tmp_path)
    for name in ('checkpoint_last.pt','checkpoint_aux_meanfront_step_000214400_v3.pt','checkpoint_aux_meanrr_step_000213376_v4.pt'):
        with pytest.raises(ValueError):cli._new_path(tmp_path/name,checkpoint=True)
    name=tmp_path/'checkpoint_aux_meanrr_step_000214400_v4.pt'
    assert cli._new_path(name,checkpoint=True)==name
    name.write_bytes(b'synthetic existing file, never overwrite')
    with pytest.raises(ValueError):cli._new_path(name,checkpoint=True)
