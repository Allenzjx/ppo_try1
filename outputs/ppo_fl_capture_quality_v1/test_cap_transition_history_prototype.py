"""CPU-only prototype checks. Synthetic optimizer steps have zero real credit."""
from __future__ import annotations
import copy
import io
import json
import math
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from wlr50_clean.ppo.semantic_history_actor import SemanticQuarterTemperedHistoryMLPModel

from cap_transition_history_prototype import (
    CAPS, REQUEST_SCALES, OutputOnlyCapTransitionQuarterActor, request_transition_history,
)

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT/'runs/ppo_fl_capture_quality_v1/train/20260918T0412242524179Z_gf2e552406ea7_541c9a6d6c8e436d811fa7bcd64a885d'
CP = ROOT/'outputs/ppo_fl_capture_quality_v1/checkpoints/history/checkpoint_step_000178688.pt'


@pytest.fixture(autouse=True)
def cpu_scope(monkeypatch):
    previous_threads, rng = torch.get_num_threads(), torch.get_rng_state()
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    torch.manual_seed(55193)
    yield
    torch.set_rng_state(rng)
    torch.set_num_threads(previous_threads)


@pytest.fixture(scope='module')
def saved():
    return (torch.load(CP, map_location='cpu', weights_only=False),
        torch.load(RUN/'rollouts/rollout_001362.pt', map_location='cpu', weights_only=False))


def obs(value):
    return TensorDict({'policy':value, 'critic':value.clone()}, batch_size=value.shape[:-1])


def actor(saved, cls=OutputOnlyCapTransitionQuarterActor):
    checkpoint, rollout = saved
    config = copy.deepcopy(checkpoint['infos']['runner_config']['actor'])
    config.pop('class_name')
    model = cls(obs(rollout['observations']['policy'][:1, 0]), {'actor':['policy']}, 'actor', 12, **config)
    before = torch.get_rng_state()
    model.load_state_dict(checkpoint['actor_state_dict'], strict=True)
    assert torch.equal(before, torch.get_rng_state())
    assert type(model.obs_normalizer) is torch.nn.Identity
    assert all(t.device.type == 'cpu' for t in model.parameters())
    return model


def exact(a,b):
    if isinstance(a,torch.Tensor): assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: exact(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert type(a) is type(b) and len(a)==len(b)
        for x,y in zip(a,b): exact(x,y)
    else: assert a==b


def test_real_saved_handoff_mean_sigma_logprob_and_no_private_history(saved):
    old, new = actor(saved,SemanticQuarterTemperedHistoryMLPModel), actor(saved)
    checkpoint, rollout = saved
    values = rollout['observations']['policy'][:,0].clone()
    original = values.clone()
    observation = obs(values)
    history, changed = request_transition_history(values)
    assert changed[14].tolist()==[True,True,True,True,True,True,True,True,True,True,False,False]
    assert int(changed.sum())==10  # One real P05->P06 handoff in this saved batch.
    assert values[14,20]==0 and values[15,20]>0
    torch.testing.assert_close(history[14,3],torch.tensor(.18112131367929543),rtol=0,atol=1e-7)
    rng=torch.get_rng_state()
    old_mean,new_mean=old(observation),new(observation)
    assert torch.equal(rng,torch.get_rng_state())
    assert torch.equal(old_mean[~changed],new_mean[~changed])
    torch.testing.assert_close(new_mean[14,3],torch.tensor(.15066612506634977),rtol=0,atol=2e-7)
    old(observation,stochastic_output=True)
    old_sigma=old.output_std.clone()
    torch.testing.assert_close(old.output_mean,rollout['distribution_params'][0][:,0],rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(old_sigma,rollout['distribution_params'][1][:,0],rtol=1e-5,atol=1e-6)
    new(observation,stochastic_output=True)
    assert torch.equal(old_sigma,new.output_std)
    expected=torch.distributions.Normal(new_mean,new.output_std)
    original_actions=rollout['actions'][:,0]
    assert torch.equal(new.get_output_log_prob(original_actions),expected.log_prob(original_actions).sum(-1))
    assert new.get_output_log_prob(original_actions)[14] != rollout['actions_log_prob'][14,0,0]
    before=torch.get_rng_state()
    actions=new(observation,stochastic_output=True)
    after=torch.get_rng_state()
    torch.set_rng_state(before)
    assert torch.equal(actions,expected.sample())
    assert torch.equal(after,torch.get_rng_state())
    cache=new.distribution._distribution
    new.reset(torch.ones(128,dtype=torch.bool))
    assert torch.equal(new(observation),new_mean)
    assert new.distribution._distribution is cache
    assert torch.equal(values,original)
    exact(old.state_dict(),new.state_dict())
    assert set(vars(old))==set(vars(new))


def fixture(phase=5,age=0.,predecessor=True):
    values=torch.zeros(1,372)
    values[0,phase]=1
    values[0,20]=age/200.
    if predecessor and phase: values[0,158:158+phase]=1
    values[0,195:207]=torch.linspace(-1,1,12)
    values[0,207:219]=torch.tensor(CAPS[max(0,phase-1)])*.6/torch.tensor(REQUEST_SCALES)
    return values


@pytest.mark.parametrize('phase,age,predecessor',[(5,1/15,True),(5,0.,False),(4,0.,True),(0,0.,False)])
def test_samephase_or_ineligible_is_bitwise_old_kernel(saved,phase,age,predecessor):
    old,new=actor(saved,SemanticQuarterTemperedHistoryMLPModel),actor(saved)
    observation=obs(fixture(phase,age,predecessor))
    assert torch.equal(old(observation),new(observation))
    rng=torch.get_rng_state()
    before=old(observation,stochastic_output=True)
    after=torch.get_rng_state()
    torch.set_rng_state(rng)
    assert torch.equal(before,new(observation,stochastic_output=True))
    assert torch.equal(after,torch.get_rng_state())


def test_zero_history_and_P02_to_P03_only_two_wheels(saved):
    old,new=actor(saved,SemanticQuarterTemperedHistoryMLPModel),actor(saved)
    values=fixture(5)
    values[:,195:231]=0
    assert torch.equal(old(obs(values)),new(obs(values)))
    values=fixture(2)
    history,mask=request_transition_history(values)
    assert mask[0].tolist()==[False]*8+[True,False,True,False]
    assert torch.equal(history[~mask],values[:,195:207][~mask])
    assert torch.equal(torch.tanh(history[mask]),torch.tensor([.36,.36]))
    # At predecessor bounds, new larger caps still make inverse tanh finite.
    values[0,207:219]=torch.tensor(CAPS[1])/torch.tensor(REQUEST_SCALES)
    history,mask=request_transition_history(values)
    assert torch.isfinite(history).all()
    assert torch.allclose(torch.tanh(history[mask]),torch.tensor([.6,.6]))


@pytest.mark.parametrize('fault',['nan','clip','no_stage','double_stage','fractional_completed','negative_age','request_outside_old_cap','wrong_dimension'])
def test_invalid_observable_contract_rejected(fault):
    values=fixture()
    if fault=='nan': values[0,210]=float('nan')
    elif fault=='clip': values[0,195]=20.01
    elif fault=='no_stage': values[0,:13]=0
    elif fault=='double_stage': values[0,0]=1
    elif fault=='fractional_completed': values[0,162]=.5
    elif fault=='negative_age': values[0,20]=-.001
    elif fault=='request_outside_old_cap': values[0,210]=25/6
    elif fault=='wrong_dimension': values=values[:,:371]
    with pytest.raises(ValueError): request_transition_history(values)


def make_ppo(saved):
    checkpoint,rollout=saved
    observation=obs(rollout['observations']['policy'][:1,0])
    config=copy.deepcopy(checkpoint['infos']['runner_config']['critic'])
    config.pop('class_name')
    critic=MLPModel(observation,{'critic':['critic']},'critic',1,**config)
    storage=RolloutStorage('rl',1,32,observation,[12],device='cpu')
    model=actor(saved)
    algorithm=PPO(model,critic,storage,num_learning_epochs=5,num_mini_batches=4,
        learning_rate=1e-5,schedule='fixed',gamma=.9985,lam=.99,device='cpu')
    rng=torch.get_rng_state()
    algorithm.load(checkpoint,load_cfg=None,strict=True)
    assert torch.equal(rng,torch.get_rng_state())
    return algorithm


def test_official_CPU_PPO_synthetic_update_save_reload_preserves_real_source_state(saved,tmp_path):
    algorithm=make_ppo(saved)
    checkpoint,rollout=saved
    assert algorithm.storage.step==0 and algorithm.transition.actions is None
    exact(algorithm.optimizer.state_dict(),checkpoint['optimizer_state_dict'])
    assert all(g['lr']==1e-5 for g in algorithm.optimizer.param_groups)
    before=copy.deepcopy(algorithm.actor.state_dict())
    for k in range(32):
        observation=obs(rollout['observations']['policy'][k:k+1,0].clone())
        with torch.no_grad():
            algorithm.act(observation)
            algorithm.process_env_step(observation,torch.tensor([math.sin(k)*.02]),torch.tensor([k==31]),{})
    with torch.no_grad(): algorithm.compute_returns(observation)
    oldlogp=algorithm.storage.actions_log_prob.clone()
    with torch.no_grad():
        flatobs=algorithm.storage.observations['policy'].flatten(0,1)
        algorithm.actor(obs(flatobs),stochastic_output=True)
        ratio=(algorithm.actor.get_output_log_prob(algorithm.storage.actions.flatten(0,1))-oldlogp.flatten()).exp()
    torch.testing.assert_close(ratio,torch.ones_like(ratio),rtol=2e-5,atol=2e-5)
    losses=algorithm.update()
    assert all(math.isfinite(float(x)) for x in losses.values())
    assert any(not torch.equal(before[k],v) for k,v in algorithm.actor.state_dict().items())
    assert algorithm.storage.step==0
    stream=io.BytesIO()
    torch.save({'models':algorithm.save(),'CPU_rng':torch.get_rng_state(),
        'prototype_only':True,'real_policy_decisions_added':0,'real_optimizer_steps_added':0},stream)
    stream.seek(0)
    state=torch.load(stream,map_location='cpu',weights_only=False)
    reloaded=make_ppo(saved)
    reloaded.load(state['models'],load_cfg=None,strict=True)
    torch.set_rng_state(state['CPU_rng'])
    exact(algorithm.save(),reloaded.save())
    assert torch.equal(algorithm.actor(observation),reloaded.actor(observation))
    assert reloaded.storage.step==0 and reloaded.transition.actions is None
    assert type(reloaded.actor.obs_normalizer) is type(reloaded.critic.obs_normalizer) is torch.nn.Identity
