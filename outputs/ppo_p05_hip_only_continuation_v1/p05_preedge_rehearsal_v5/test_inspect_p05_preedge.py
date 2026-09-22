"""Synthetic CPU inspection tests; no real checkpoint optimization or success credit."""
from copy import deepcopy

import pytest
import torch
import inspect_p05_preedge as a
from wlr50_clean.ppo.semantic_p05_capture_profile import P05_CAPTURE_POLICY, P05_CAPTURE_OBSERVATION_LAYOUT
from wlr50_clean.ppo.semantic_p05_capture_migration import _ObservationOnlyEnv


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    rng=a.training.capture_training_rng_state(seed=0)
    threads=torch.get_num_threads();torch.set_num_threads(1);torch.manual_seed(1729)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    yield
    a.training.restore_training_rng_state(rng,expected_seed=0);torch.set_num_threads(threads)


def runner():
    run=a.training.construct_semantic_runner(_ObservationOnlyEnv(389),seed=1001,device='cpu',
        initialize_actor=False,policy_version=P05_CAPTURE_POLICY,
        observation_layout=P05_CAPTURE_OBSERVATION_LAYOUT)[0]
    # Populated synthetic Adam fixture, never any real checkpoint learning.
    for p in [*run.alg.actor.parameters(),*run.alg.critic.parameters()]:
        p.grad=torch.linspace(-.03,.05,p.numel()).reshape_as(p)
    run.alg.optimizer.step();run.alg.optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        run.alg.actor.mlp[4].weight[12:]+=torch.linspace(-.03,.03,12*256).reshape(12,256)
    return run


def data(actor):
    x=torch.zeros(6,389);x[:,4]=1;x[:,20]=.01
    x[:,30]=torch.linspace(-.1,.1,6)
    x[:,195:207]=torch.linspace(-.2,.2,72).reshape(6,12)
    v=x.clone();v[:,30]+=.023;v[:,195:207]*=.7
    h=torch.zeros(8,389);h[torch.arange(8),torch.tensor([4,6,7,8,9,10,11,12])-1]=1
    h[:,20]=.01;h[:,157]=1
    h[:,195:207]=torch.linspace(-.1,.1,96).reshape(8,12)
    with torch.no_grad():
        target=a.distribution(actor,a.kernel.tensors(x))['mean']+torch.linspace(.02,.05,12)
        vt=a.distribution(actor,a.kernel.tensors(v))['mean']+torch.linspace(.02,.05,12)
    return {'train_observations':x,'train_raw_targets':target,
        'validation_observations':v,'validation_raw_targets':vt,
        'protection_observations':h,'receipt':{'synthetic_fixture':True}}


def protected(run):
    return {'actor':a.training.state_hash(run.alg.actor.state_dict()),
        'critic':a.training.state_hash(run.alg.critic.state_dict()),
        'Adam':a.training.state_hash(run.alg.optimizer.state_dict()),
        'normalizers':a.training.state_hash(a.training._normalizers(run)),
        'rng':a.training.capture_training_rng_state(seed=1001),
        'grads':a.training.state_hash({f'{who}/{k}':p.grad for who,obj in (
            ('actor',run.alg.actor),('critic',run.alg.critic)) for k,p in obj.named_parameters()})}


def test_exact_official_distribution_and_true_point_one_history_chain():
    run=runner();d=data(run.alg.actor);obs=a.kernel.tensors(d['train_observations'])
    leaf=run.alg.actor.mlp[0].weight[:,4].detach().clone().requires_grad_(True)
    actual=a.distribution(run.alg.actor,obs,leaf)
    assert a.kernel._exact_invariance(actual,a.distribution(run.alg.actor,obs))
    assert torch.equal(actual['mean'],run.alg.actor(obs,stochastic_output=False))
    rng=torch.get_rng_state().clone();run.alg.actor(obs,stochastic_output=True)
    assert torch.equal(actual['sigma'],run.alg.actor.output_std);torch.set_rng_state(rng)
    network,=torch.autograd.grad(actual['network_mean'].sum(),leaf,retain_graph=True)
    conditioned,=torch.autograd.grad(actual['mean'].sum(),leaf)
    torch.testing.assert_close(conditioned,.1*network,rtol=1e-4,atol=1e-7)
    assert float(conditioned.norm()/network.norm())==pytest.approx(.1,rel=1e-6)


def test_nonP05_exact_but_P05_mean_and_sigma_can_change():
    run=runner();d=data(run.alg.actor)
    leaf=run.alg.actor.mlp[0].weight[:,4].detach().clone()+torch.linspace(-.03125,.03125,256)
    x=torch.zeros(12,389);x[torch.arange(12),torch.tensor(a.NON_P05_PHASES)-1]=1
    for obs in (a.kernel.tensors(x),a.kernel.tensors(d['protection_observations'])):
        assert a.kernel._exact_invariance(a.distribution(run.alg.actor,obs),a.distribution(run.alg.actor,obs,leaf))
    obs=a.kernel.tensors(d['train_observations'])
    before=a.distribution(run.alg.actor,obs);after=a.distribution(run.alg.actor,obs,leaf)
    assert not torch.equal(before['mean'],after['mean'])
    assert not torch.equal(before['log_sigma'],after['log_sigma'])


def test_inspect_has_no_step_preserves_complete_runner_and_existing_grads(monkeypatch):
    run=runner();d=data(run.alg.actor)
    for p in [*run.alg.actor.parameters(),*run.alg.critic.parameters()]:p.grad=torch.full_like(p,.031)
    before=protected(run)
    def forbid(*args,**kwargs):raise AssertionError('readonly inspector stepped an optimizer')
    monkeypatch.setattr(torch.optim.SGD,'step',forbid);monkeypatch.setattr(torch.optim.Adam,'step',forbid)
    r=a.inspect_actor(run.alg.actor,d)
    assert protected(run)==before
    assert r['gradient_l2']>0 and len(r['gradient_full256'])==256
    assert torch.tensor(r['per_channel_gradient_full12x256']).shape==(12,256)
    assert r['actual_nonP05_holdout_phases']==[4,6,7,8,9,10,11,12]
    assert r['missing_actual_nonP05_holdout_phases']==[1,2,3,13]
    assert r['negative_frozen_gradient_unit_lr_JVP']['protection']['all_derivatives_exact_zero']
    assert any(r['negative_frozen_gradient_unit_lr_JVP']['train']['maximum_abs_log_sigma_per_unit_lr_per_channel'])
    assert r['P05_mean_and_log_sigma_may_both_change']
    assert r['optimizer_steps_performed']==r['AUX_updates_added']==r['PPO_updates_added']==0
    assert not r['checkpoint_written'] and not r['physical_success_claimed']
    assert not hasattr(a,'fit') and not hasattr(a,'Budget')


@pytest.mark.parametrize('bad',['positive_phase','protection_phase','mode','initialized','overlap','nan','targets','old372','soft_phase'])
def test_invalid_scope_and_data_fail_closed(bad):
    run=runner();d=data(run.alg.actor);before=protected(run)
    if bad=='positive_phase':d['train_observations'][0,4]=0;d['train_observations'][0,1]=1
    elif bad=='protection_phase':d['protection_observations'][0,:13]=0;d['protection_observations'][0,4]=1
    elif bad=='mode':d['train_observations'][0,372]=.2
    elif bad=='initialized':d['validation_observations'][0,373]=1
    elif bad=='overlap':d['validation_observations']=d['train_observations'].clone()
    elif bad=='nan':d['train_raw_targets'][0,0]=float('nan')
    elif bad=='targets':d['train_raw_targets']=d['train_raw_targets'][:,:11]
    elif bad=='old372':d['train_observations']=d['train_observations'][:,:372]
    elif bad=='soft_phase':d['train_observations'][0,4]=.9
    with pytest.raises(ValueError):a.inspect_actor(run.alg.actor,d)
    assert protected(run)==before


@pytest.mark.parametrize('fault',['rng','parameter','grad'])
def test_unexpected_mutation_is_restored_and_refuses_report(monkeypatch,fault):
    run=runner();d=data(run.alg.actor);before=protected(run)
    def sabotage(actor,data):
        if fault=='rng':torch.rand(1)
        elif fault=='parameter':
            with torch.no_grad():actor.mlp[0].weight[0,0]+=1
        else:actor.mlp[0].weight.grad=torch.ones_like(actor.mlp[0].weight)
        return {'must_not_publish':True}
    monkeypatch.setattr(a,'_inspect',sabotage)
    with pytest.raises(RuntimeError,match='restored'):a.inspect_actor(run.alg.actor,d)
    assert protected(run)==before


def test_cpu_checkpoint_actor_copy_preserves_whole_source_and_process_rng(tmp_path):
    run=runner();d=data(run.alg.actor)
    infos={'seed':1001,'actor_parameter_sha256':a.training.parameter_hash(run.alg.actor),
        'optimizer_state_sha256':a.training.state_hash(run.alg.optimizer.state_dict()),
        'normalizer_state_sha256':a.training.state_hash(a.training._normalizers(run)),
        'training_rng_state':a.training.capture_training_rng_state(seed=1001),
        'global_policy_decisions':128,'ppo_updates':1,'optimizer_steps':20}
    payload={'actor_state_dict':run.alg.actor.state_dict(),'critic_state_dict':run.alg.critic.state_dict(),
        'optimizer_state_dict':run.alg.optimizer.state_dict(),'infos':infos}
    path=tmp_path/'synthetic.pt';torch.save(payload,path)
    metadata={**deepcopy(infos),'checkpoint_path':str(path),'checkpoint_sha256':a.sha(path)}
    before=protected(run);payloadhash=a.training.state_hash(payload)
    r=a.cpu_inspection(path,metadata,d)
    assert a.training.state_hash(torch.load(path,weights_only=False))==payloadhash
    assert protected(run)==before
    assert r['source_checkpoint_sha256_unchanged']==metadata['checkpoint_sha256']
    assert r['source_device_metadata_unchanged'] and not r['checkpoint_written']


def test_cli_rejects_execute_or_budget_options():
    for option in ('--execute-aux','--budget'):
        with pytest.raises(SystemExit):a.main([option])
