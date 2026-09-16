"""Small synthetic CPU checks; never optimize saved robot rollouts."""
from __future__ import annotations
import json
from types import SimpleNamespace
import pytest
import torch
from tensordict import TensorDict
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from wlr50_clean.ppo.semantic_history_actor import SemanticTemperedHistoryMLPModel
from wlr50_clean.ppo.semantic_training import audited_ppo_update


@pytest.fixture(autouse=True)
def preserve_cpu_state():
    rng=torch.get_rng_state(); threads=torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_rng_state(rng); torch.set_num_threads(threads)


def synthetic_runner():
    torch.manual_seed(50721)
    obs=TensorDict({'policy':torch.randn(1,372)},batch_size=[1])
    actor=SemanticTemperedHistoryMLPModel(obs,{'actor':['policy']},'actor',12,[16,16],'elu',False,
        {'class_name':'HeteroscedasticGaussianDistribution','init_std':.15,'std_type':'log'},
        'diagonal_transfer_state_v1',.5)
    critic=MLPModel(obs,{'critic':['policy']},'critic',1,[16,16],'elu',False)
    storage=RolloutStorage('rl',1,16,obs,[12],'cpu')
    alg=PPO(actor,critic,storage,num_learning_epochs=5,num_mini_batches=4,
        learning_rate=3e-5,gamma=.9985,lam=.99,device='cpu')
    with torch.no_grad():
        for i in range(16):
            raw=alg.act(obs)
            nxt=torch.randn(1,372)
            nxt[:,195:207]=raw.clamp(-20,20)
            obs=TensorDict({'policy':nxt},batch_size=[1])
            alg.process_env_step(obs,torch.tensor([(-1.)**i*.1]),torch.tensor([i==8]),{'time_outs':torch.tensor([False])})
        alg.compute_returns(obs)
    return SimpleNamespace(alg=alg)


def recursive_equal(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(recursive_equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(recursive_equal(x,y) for x,y in zip(a,b))
    return a==b


def test_likelihood_instrumentation_has_bitwise_identical_updates_optimizer_rng_and_storage(tmp_path):
    plain=synthetic_runner(); observed=synthetic_runner()
    pre=torch.get_rng_state().clone()
    plain_result=audited_ppo_update(plain)
    plain_after_rng=torch.get_rng_state().clone()
    torch.set_rng_state(pre)
    target=tmp_path/'actual_likelihood.json'
    observed_result=audited_ppo_update(observed,likelihood_audit_path=target)
    assert torch.equal(plain_after_rng,torch.get_rng_state())
    assert plain_result==observed_result
    assert recursive_equal(plain.alg.actor.state_dict(),observed.alg.actor.state_dict())
    assert recursive_equal(plain.alg.critic.state_dict(),observed.alg.critic.state_dict())
    assert recursive_equal(plain.alg.optimizer.state_dict(),observed.alg.optimizer.state_dict())
    for key in ('actions','advantages','returns','actions_log_prob','values','rewards','dones'):
        assert torch.equal(getattr(plain.alg.storage,key),getattr(observed.alg.storage,key))
    receipt=json.loads(target.read_text())
    assert receipt['extra_model_forwards']==receipt['extra_random_draws']==0
    assert len(receipt['minibatches'])==20
    appearances=[0]*16
    for batch in receipt['minibatches']:
        for j,indices in enumerate(batch['rollout_flat_indices']):
            assert len(indices)==1
            appearances[indices[0]]+=1
            ratio=batch['ratio'][j]; adv=batch['actual_advantage'][j]
            assert batch['clipped_branch_strictly_active'][j] == (-adv*min(1.2,max(.8,ratio)) > -adv*ratio)
        assert batch['history_source']=='this_saved_observation_195_207_not_shuffled_neighbor'
    assert appearances==[5]*16
    # Only the first minibatch is pre-update: later first-epoch batches already see changed weights.
    assert max(abs(x-1.) for x in receipt['minibatches'][0]['ratio'])<2e-5


@pytest.mark.parametrize('advantage',[-2.,2.])
@pytest.mark.parametrize('action',[-.4,.4])
def test_local_ppo_gradient_changes_sample_probability_with_advantage(advantage,action):
    mean=torch.tensor(0.,requires_grad=True)
    sample=torch.tensor(action)
    old_logp=torch.distributions.Normal(mean.detach(),.2).log_prob(sample)
    logp=torch.distributions.Normal(mean,.2).log_prob(sample)
    ratio=(logp-old_logp).exp()
    loss=torch.maximum(-advantage*ratio,-advantage*ratio.clamp(.8,1.2))
    gradient,=torch.autograd.grad(loss,mean)
    new_mean=mean.detach()-.0001*gradient
    new_logp=torch.distributions.Normal(new_mean,.2).log_prob(sample)
    assert float((new_logp-old_logp)*advantage)>0


@pytest.mark.parametrize('advantage,ratio,active',[(2.,1.3,True),(2.,.7,False),(-2.,.7,True),(-2.,1.3,False),(2.,1.,False)])
def test_clip_branch_is_sign_sensitive_not_any_ratio_outside_band(advantage,ratio,active):
    loss=-advantage*ratio
    clipped=-advantage*min(1.2,max(.8,ratio))
    assert (clipped>loss)==active
