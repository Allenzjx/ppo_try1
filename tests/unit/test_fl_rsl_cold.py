# Adaptation draft only: install under tests/unit after production modules exist.
"""COLD CPU ONLY; not executed while Isaac runs. Synthetic, not training credit.

Uses the already DET-validated A0 c039 package, loaded through its strict route.
Actual future B publication must choose the latest DET-validated A, not blindly
the model with the highest count. All new samples here are synthetic CPU tests.
CUDA RNG is an opaque CPU byte bank: no CUDA initialization/execution is allowed.
Python/NumPy/Torch CPU RNG and all CPU model/Adam computations remain actual.
"""
import copy
import json
import hashlib
import sys
from pathlib import Path
from functools import lru_cache
from types import SimpleNamespace
import pytest
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from rsl_rl.algorithms import PPO
from rsl_rl.storage import RolloutStorage
from wlr50_clean.ppo.semantic_rr_capture_local_actor import tensor_state_sha256
from wlr50_clean.ppo import semantic_finish_advance as accepted_route
from wlr50_clean.ppo.semantic_fl_forward_actor import WindowFLActor, WindowFLDistribution, audited_fl_request, audited_fl_ppo_update
from wlr50_clean.ppo.semantic_fl_forward_window import FL, B_FIELDS, scalar_logp, scalar_kl, scalar_entropy
from wlr50_clean.ppo.rl_library_wrapper import (capture_training_rng_state,
    restore_training_rng_state, _decode_torch_rng_state)


ROOT = next(p for p in Path(__file__).resolve().parents if (p/'src/wlr50_clean').is_dir())
sys.path.insert(0,str(ROOT/'tests/unit'))
from test_semantic_finish_advance_actor import recorded490


@lru_cache(maxsize=1)
def accepted_A():
    path=ROOT/'outputs/ppo_finish_advance_then_home_v1/checkpoints/history/checkpoint_CP232960_finish000000_gc039c74281c1.pt'
    side=path.with_name(path.stem+'_manifest.json')
    assert hashlib.sha256(path.read_bytes()).hexdigest()=='c5f2d8ae756296f9392e71955ea0bacd41397cd8983b3a98fbb1dea75566b845'
    assert hashlib.sha256(side.read_bytes()).hexdigest()=='825b6b70e3ba754fa613fb18a49ea722bbbad4b45f5c40d0f0b5d02944238d37'
    meta=json.loads(side.read_text())
    runner=accepted_route.make_runner('cpu',1001,saved_configuration=meta['runner_config'])
    accepted_route.load(runner,path,meta['runtime_contract'])
    return runner.alg.actor,meta['runner_config']['actor']


@pytest.fixture(autouse=True)
def rng_and_threads(monkeypatch):
    path=ROOT/'outputs/ppo_finish_advance_then_home_v1/checkpoints/history/checkpoint_CP232960_finish000000_gc039c74281c1_manifest.json'
    meta=json.loads(path.read_text())
    bank=[_decode_torch_rng_state(value,label='test-only opaque CUDA state')
          for value in meta['training_rng']['torch_cuda']]
    def set_bank(values): bank[:]=[value.detach().cpu().clone() for value in values]
    monkeypatch.setattr(torch.cuda,'is_available',lambda:bool(bank))
    monkeypatch.setattr(torch.cuda,'device_count',lambda:len(bank))
    monkeypatch.setattr(torch.cuda,'get_rng_state_all',lambda:[v.clone() for v in bank])
    monkeypatch.setattr(torch.cuda,'set_rng_state_all',set_bank)
    monkeypatch.setattr(torch.cuda,'manual_seed',lambda *a,**k:None)
    monkeypatch.setattr(torch.cuda,'manual_seed_all',lambda *a,**k:None)
    def no_cuda(*a,**k): raise AssertionError('GPU initialization forbidden in CPU test')
    monkeypatch.setattr(torch.cuda,'init',no_cuda)
    monkeypatch.setattr(torch.cuda,'_lazy_init',no_cuda)
    original_rng=capture_training_rng_state(seed=1001)
    n=torch.get_num_threads()
    torch.set_num_threads(1); torch.manual_seed(1001)
    try:
        yield
    finally:
        try:
            restore_training_rng_state(original_rng,expected_seed=1001)
        finally:
            torch.set_num_threads(n)


def create(flags=(1,0,1,0)):
    x=torch.zeros(len(flags),531+len(B_FIELDS))
    x[:,:490]=recorded490()[torch.arange(len(flags))%32]
    x[:,531]=torch.tensor(flags)
    obs=TensorDict({'policy':x},batch_size=[len(flags)])
    frozen,cfg=accepted_A()
    digest=tensor_state_sha256(frozen.state_dict())
    actor=WindowFLActor(obs,{'actor':['policy']},'actor',12,
        frozen_A_configuration=cfg,expected_frozen_A_sha256=digest)
    actor.load_A_state(frozen.state_dict(),expected_sha256=digest)
    return actor,obs


def test_only_FL_samples_and_zero_delta_preserves_entire_frozen_mean():
    actor,obs=create()
    baseline=actor.frozen_A({'policy':obs['policy'][:,:531]},stochastic_output=False)
    assert torch.equal(actor(obs,stochastic_output=False),baseline)
    sample=actor(obs,stochastic_output=True)
    other=[i for i in range(12) if i!=FL]
    assert torch.equal(sample[:,other],baseline[:,other])
    assert torch.equal(sample[[1,3],FL],baseline[[1,3],FL])
    assert all(p.shape==(4,1) for p in actor.output_distribution_params)
    assert actor.distribution.std.shape==(4,1)
    with torch.no_grad(): actor.mlp.mean_delta.fill_(.4)
    actual=actor(obs,stochastic_output=False)
    assert torch.equal(actual[:,other],baseline[:,other])
    assert torch.allclose(actual[[0,2],FL]-baseline[[0,2],FL],torch.tensor([.4,.4]))


def test_inactive_no_new_head_no_rng_no_hidden_history_or_filter_reset():
    actor,obs=create((0,0))
    seen=[]; handle=actor.mlp.register_forward_hook(lambda *args:seen.append(True))
    before=torch.get_rng_state().clone()
    sample=actor(obs,stochastic_output=True)
    handle.remove()
    baseline=actor.frozen_A({'policy':obs['policy'][:,:531]},stochastic_output=False)
    assert not seen and torch.equal(torch.get_rng_state(),before)
    assert torch.equal(sample,baseline)
    assert torch.equal(actor.get_output_log_prob(sample),torch.zeros(2))
    assert torch.equal(actor.output_entropy,torch.zeros(2))


def test_scalar_old_current_logp_KL_entropy_exact_dimension_and_negatives():
    actor,obs=create()
    sample=actor(obs,stochastic_output=True).detach()
    old=tuple(p.detach().clone() for p in actor.output_distribution_params)
    oldlog=actor.get_output_log_prob(sample).detach().clone()
    for i in range(4):
        assert float(oldlog[i])==pytest.approx(scalar_logp(float(sample[i,FL]),float(old[0][i,0]),
                float(old[1][i,0]),bool(old[2][i,0])),abs=2.e-6)
    with torch.no_grad(): actor.mlp.mean_delta.add_(.01); actor.mlp.log_std.add_(.02)
    actor(obs,stochastic_output=True)
    new=actor.output_distribution_params
    now=actor.get_output_log_prob(sample)
    kl=actor.get_kl_divergence(old,new)
    for i in range(4):
        on=bool(old[2][i,0])
        assert float(now[i])==pytest.approx(scalar_logp(float(sample[i,FL]),float(new[0][i,0]),float(new[1][i,0]),on),abs=2.e-6)
        assert float(kl[i])==pytest.approx(scalar_kl(float(old[0][i,0]),float(old[1][i,0]),float(new[0][i,0]),float(new[1][i,0]),old_active=on,new_active=on),abs=2.e-6)
        assert float(actor.output_entropy[i])==pytest.approx(scalar_entropy(float(new[1][i,0]),on),abs=2.e-6)
    bad=sample.clone();bad[0,3]+=.01
    with pytest.raises(ValueError,match='deterministic support'): actor.get_output_log_prob(bad)
    badflag=new[2].clone();badflag[0]=0.
    with pytest.raises(ValueError,match='window mask'): actor.get_kl_divergence(old,(new[0],new[1],badflag))
    dist=WindowFLDistribution(12)
    with pytest.raises(ValueError): dist.update((torch.zeros(1,12),torch.zeros(1,1),torch.ones(1,1)))


def test_new_parameters_only_optimizer_strict_save_reload_and_frozen_A(tmp_path):
    actor,obs=create(); frozen=copy.deepcopy(actor.frozen_A.state_dict())
    assert sum(p.numel() for p in actor.trainable_parameters())==2
    optimizer=torch.optim.Adam(actor.trainable_parameters(),lr=1.e-4,weight_decay=.01)
    sample=actor(obs,stochastic_output=True).detach()
    loss=-actor.get_output_log_prob(sample).mean()-.01*actor.output_entropy.mean()
    loss.backward();optimizer.step();actor.assert_frozen_state(optimizer)
    assert all(torch.equal(v,frozen[k]) for k,v in actor.frozen_A.state_dict().items())
    path=tmp_path/'unit_candidate_only.pt';torch.save(actor.state_dict(),path)
    restored=WindowFLActor(obs,{'actor':['policy']},'actor',12,frozen_A_configuration=accepted_A()[1],
        expected_frozen_A_sha256=actor.expected_frozen_A_sha256)
    restored.load_state_dict(torch.load(path,map_location='cpu',weights_only=False))
    assert torch.equal(actor(obs,stochastic_output=False),restored(obs,stochastic_output=False))
    broken=copy.deepcopy(actor.state_dict());broken['frozen_A.mlp.4.bias'][0]+=.01
    with pytest.raises(ValueError): restored.load_state_dict(broken)


def test_actual_RSL_full12_storage_scalar_params_contiguous_GAE_and_audited_update(tmp_path):
    actor,batch=create((1,)); first=batch
    critic=MLPModel(first,{'critic':['policy']},'critic',1,(8,),'elu',False,None)
    storage=RolloutStorage('rl',1,16,first,(12,),'cpu')
    alg=PPO(actor,critic,storage,num_learning_epochs=1,num_mini_batches=4,
            learning_rate=1.e-5,gamma=.9985,lam=.99,device='cpu',schedule='adaptive')
    # Official default includes all parameters; explicitly exclude frozen A.
    alg.optimizer=torch.optim.Adam((*actor.trainable_parameters(),*critic.parameters()),lr=1.e-5)
    rawrows=[];logs=[]
    for i in range(16):
        obs=first.clone();obs['policy'][:,531]=float(i%4==0)
        obs['policy'][:,532]=.8-i*.001
        with torch.no_grad():
            raw=alg.act(obs);rawrows.append(raw.clone());logs.append(alg.transition.actions_log_prob.clone())
            alg.process_env_step(obs,torch.tensor([-.01]),torch.tensor([False]),{})
    assert storage.actions.shape==(16,1,12)
    assert all(p.shape==(16,1,1) for p in storage.distribution_params)
    assert int(storage.distribution_params[2].sum())==4  #4 FL rows +12 critic-only; no splicing.
    assert not storage.dones.any()
    assert torch.equal(storage.actions,torch.stack(rawrows))
    assert torch.equal(storage.actions_log_prob.view(16),torch.stack(logs).view(16))
    with torch.no_grad():alg.compute_returns(obs)
    frozen=copy.deepcopy(actor.frozen_A.state_dict())
    runner=SimpleNamespace(alg=alg,_semantic_policy_version=actor.policy_version,current_learning_iteration=0)
    report=audited_fl_ppo_update(runner,likelihood_audit_path=tmp_path/'scalar_likelihood.json')
    assert report['optimizer_steps']==4
    actor.assert_frozen_state(alg.optimizer)
    assert all(torch.equal(v,frozen[k]) for k,v in actor.frozen_A.state_dict().items())
    audit=json.loads((tmp_path/'scalar_likelihood.json').read_text())
    assert len(audit['minibatches'])==4
    assert all(len(row)==1 for mb in audit['minibatches'] for row in mb['mean_FL'])
    assert report['total_continuous_storage_rows']==16 and report['stochastic_FL_rows']==4
    assert report['critic_only_context_rows']==12


def test_all_inactive_PPO_batch_leaves_existing_scalar_Adam_momentum_untouched(tmp_path):
    actor,obs=create((1,))
    critic=MLPModel(obs,{'critic':['policy']},'critic',1,(8,),'elu',False,None)
    storage=RolloutStorage('rl',1,4,obs,(12,),'cpu')
    alg=PPO(actor,critic,storage,num_learning_epochs=1,num_mini_batches=1,learning_rate=1.e-5,
            device='cpu',schedule='adaptive')
    alg.optimizer=torch.optim.Adam((*actor.trainable_parameters(),*critic.parameters()),lr=1.e-5)
    # Deliberately establish nonzero existing Adam moments before inactive PPO.
    loss=actor.mlp.mean_delta+actor.mlp.log_std
    alg.optimizer.zero_grad();loss.backward();alg.optimizer.step();alg.optimizer.zero_grad(set_to_none=True)
    before=copy.deepcopy({name:(p.detach().clone(),alg.optimizer.state[p]) for name,p in actor.mlp.named_parameters()})
    obs['policy'][:,531]=0.
    for _ in range(4):
        with torch.no_grad():
            alg.act(obs)
            alg.process_env_step(obs,torch.tensor([-.01]),torch.tensor([False]),{})
    with torch.no_grad():alg.compute_returns(obs)
    runner=SimpleNamespace(alg=alg)
    report=audited_fl_ppo_update(runner,likelihood_audit_path=tmp_path/'inactive.json')
    assert report['stochastic_FL_rows']==0 and report['critic_only_context_rows']==4
    assert report['actor_gradient_steps']==0 and report['inactive_only_steps']==1
    assert report['steps'][0]['kl_all']==0. and report['actual_learning_rate']==1.e-5
    for name,p in actor.mlp.named_parameters():
        value,state=before[name]
        assert torch.equal(value,p.detach()) and p.grad is None
        for key,old in state.items():
            assert torch.equal(old,alg.optimizer.state[p][key]) if torch.is_tensor(old) else old==alg.optimizer.state[p][key]


def test_one_forward_audited_request_keeps_raw_likelihood_and_no_inactive_credit():
    actor,obs=create((1,))
    raw,audit=audited_fl_request(actor,obs,lambda:actor(obs,stochastic_output=True),stochastic=True)
    assert audit['sampling_draws']==1 and audit['density_dimension']==1
    assert audit['selected_raw_full12']==raw[0].tolist()
    assert audit['selected_raw_log_probability']==float(actor.get_output_log_prob(raw)[0])
    obs['policy'][:,531]=0.;before=torch.get_rng_state().clone()
    _,audit=audited_fl_request(actor,obs,lambda:actor(obs,stochastic_output=True),stochastic=True)
    assert audit['sampling_draws']==0 and audit['stochastic_FL_credit']==0
    assert audit['selected_raw_log_probability']==0. and torch.equal(before,torch.get_rng_state())
