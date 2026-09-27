"""CPU synthetic wiring only; actual04 package, no Isaac or new PPO credit.

Recorded inputs are real487 T1 rollout rows before the old finish endpoint,
with the three published zero-column finish fields appended to obtain490.
The added finish41 tail is explicitly counterfactual test input, not new data.
"""
import copy
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'outputs/ppo_post_rr_front_pair_rl_v1/checkpoints/history/checkpoint_CP232960_postRR000512_g04a8001b901f.pt'
SOURCE_SHA = 'b24126a4df16c780f5ce65039ac4fce0c03c7d57c838b8ce5fc75fc477fa7bd9'
MANIFEST_SHA = 'f292d84bfa2f302373360855711dadd73bc4ea4e5c556ac8e1562dcee5cc1e45'


@pytest.fixture(autouse=True)
def cpu_rng_only():
    state = torch.get_rng_state().clone()
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(1001)
    yield
    torch.set_rng_state(state)
    torch.set_num_threads(threads)


@lru_cache(maxsize=1)
def source_data():
    from wlr50_clean.ppo.semantic_training import state_hash
    sidecar = SOURCE.with_name(SOURCE.stem+'_manifest.json')
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == SOURCE_SHA
    assert hashlib.sha256(sidecar.read_bytes()).hexdigest() == MANIFEST_SHA
    meta = json.loads(sidecar.read_text())
    saved = torch.load(SOURCE, map_location='cpu', weights_only=False)
    assert meta['observation_dimension'] == 490
    assert all(meta.get(k) == v for k,v in saved['infos'].items())
    assert all(state_hash(saved[k]) == v for k,v in meta['state_hashes'].items())
    return saved


@lru_cache(maxsize=1)
def recorded490():
    rollout = torch.load(ROOT/'runs/ppo_post_rr_front_pair_rl_v1/RL_T1_train512_475eb1f/rollouts/rollout_0001.pt',
                         map_location='cpu', weights_only=False)
    # All are actual first-episode rows before76s, not the later finish failure.
    old = rollout['observations']['policy'].reshape(-1,487)[torch.linspace(0,95,32).long()].clone()
    return torch.cat((old,torch.zeros(len(old),3)),dim=-1)


def obs_from(x):
    return TensorDict({'policy':x,'critic':x.clone()},batch_size=[len(x)])


def make_actor(*, bind=True):
    from wlr50_clean.ppo.semantic_finish_advance_actor import SemanticFinishAdvanceMLPModel, OBSERVATION_DIMENSION
    from wlr50_clean.ppo.semantic_rr_capture_local_actor import tensor_state_sha256
    saved = source_data()
    old = recorded490().clone()
    assert OBSERVATION_DIMENSION == 531
    x = torch.cat((old,torch.zeros(len(old),OBSERVATION_DIMENSION-490)),dim=-1)
    cfg = copy.deepcopy(saved['infos']['runner_config']['actor'])
    kwargs = dict(hidden_dims=(128,128),obs_normalization=False,
        distribution_cfg={'class_name':'HeteroscedasticGaussianDistribution','init_std':.04,'std_type':'log'},
        frozen_accepted_configuration=cfg,
        expected_accepted_state_sha256=tensor_state_sha256(saved['actor_state_dict']),
        initial_finish_std=(.04,)*12)
    actor = SemanticFinishAdvanceMLPModel(obs_from(x),{'actor':['policy']},'actor',12,**kwargs)
    if bind:
        actor.load_accepted_state(saved['actor_state_dict'],expected_sha256=kwargs['expected_accepted_state_sha256'])
    return actor,obs_from(x),kwargs


def active_observation(obs):
    from wlr50_clean.ppo.semantic_finish_advance_task import FINISH_FIELDS
    active = obs.clone()
    x = active['policy']
    x[:,490] = 1.
    # Distinct nonzero committed REQUEST reference per channel. This proves
    # handoff uses observed physical-reference ratios, not a raw-zero reset.
    reference = torch.linspace(-.55,.45,12)
    for i in range(12):
        x[:,490+FINISH_FIELDS.index('finish_request_ref_ratio_'+str(i))] = reference[i]
    active['critic'] = x.clone()
    return active,reference


def make_critic(obs):
    cfg = copy.deepcopy(source_data()['infos']['runner_config']['critic'])
    cfg.pop('class_name')
    critic = MLPModel(obs,{'critic':['critic']},'critic',1,**cfg)
    old = source_data()['critic_state_dict']
    expanded = {}
    changes = []
    for k,v in old.items():
        target = critic.state_dict()[k]
        if target.shape == v.shape:
            expanded[k] = v.clone()
        else:
            assert v.ndim == 2 and v.shape[1] == 490 and target.shape == (v.shape[0],531)
            expanded[k] = torch.zeros_like(target)
            expanded[k][:,:490] = v
            changes.append(k)
    assert changes == ['mlp.0.weight']
    critic.load_state_dict(expanded,strict=True)
    return critic


def test_real_complete_prefix_bitexact_after_joint_new_head_critic_Adam_step():
    from wlr50_clean.ppo.semantic_post_rr_front_prep_actor import SemanticPostRRHistoryMLPModel
    actor,prefix,_ = make_actor()
    old_cfg = copy.deepcopy(source_data()['infos']['runner_config']['actor']); old_cfg.pop('class_name')
    accepted = SemanticPostRRHistoryMLPModel(obs_from(recorded490()),{'actor':['policy']},'actor',12,**old_cfg)
    accepted.load_state_dict(source_data()['actor_state_dict'],strict=True)
    with torch.no_grad():
        expected = accepted(obs_from(recorded490()),stochastic_output=False)
        before = actor(prefix,stochastic_output=False)
    assert torch.equal(expected,before)
    frozen = copy.deepcopy(actor.frozen_accepted.state_dict())
    critic = make_critic(prefix)
    critic_before = copy.deepcopy(critic.state_dict())
    optimizer = torch.optim.Adam(list(actor.trainable_parameters())+list(critic.parameters()),
                                 lr=1.e-5,weight_decay=.01)
    actor.train(); critic.train()
    actor.assert_frozen_state(optimizer)
    active,_ = active_observation(prefix)
    current = actor(active,stochastic_output=False)
    loss = (current-.2).square().mean()+(critic(active)-.3).square().mean()
    optimizer.zero_grad();loss.backward();optimizer.step()
    actor.assert_frozen_state(optimizer)
    assert all(torch.equal(v,actor.frozen_accepted.state_dict()[k]) for k,v in frozen.items())
    assert all(not p.requires_grad and p.grad is None for p in actor.frozen_accepted.parameters())
    assert not actor.frozen_accepted.training
    assert any(not torch.equal(v,critic.state_dict()[k]) for k,v in critic_before.items())
    with torch.no_grad():
        assert torch.equal(actor(prefix,stochastic_output=False),before)
        assert not torch.equal(actor(active,stochastic_output=False),current)
    assert type(actor.obs_normalizer) is torch.nn.Identity
    assert type(actor.frozen_accepted.obs_normalizer) is torch.nn.Identity


def test_prefix_no_sampling_no_rng_draw_and_no_new_head_forward(monkeypatch):
    from wlr50_clean.ppo.semantic_finish_advance_actor import audited_finish_request
    actor,prefix,_ = make_actor()
    def forbidden(*args,**kwargs):
        raise AssertionError('new head or distribution sampling reached before legal RL event')
    monkeypatch.setattr(actor.mlp,'forward',forbidden)
    monkeypatch.setattr(actor.distribution,'sample',forbidden)
    before = torch.get_rng_state().clone()
    one = prefix[:1]
    with torch.no_grad():
        _,audit = audited_finish_request(actor,one,lambda:actor(one,stochastic_output=False),stochastic=False)
    assert not audit['finish_active'] and audit['prefix_excluded_from_new_PPO_credit']
    assert audit['sampling_draws'] == 0 and audit['selected_raw_log_probability'] is None
    with pytest.raises(ValueError,match='no new sampling'):
        actor(prefix,stochastic_output=True)
    assert torch.equal(before,torch.get_rng_state())


def test_active_reference_HISTORY_once_and_real_Gaussian_old_current_logp(monkeypatch):
    from wlr50_clean.ppo.semantic_finish_advance_actor import audited_finish_request
    from wlr50_clean.ppo.semantic_p05_capture_actor import p05_capture_request_history
    from wlr50_clean.ppo.semantic_history_actor import HISTORY_RHO
    actor,prefix,_ = make_actor()
    active,reference = active_observation(prefix[:1])
    center,_ = p05_capture_request_history(active['policy'][:,:389])
    expected = HISTORY_RHO*center+(1.-HISTORY_RHO)*torch.atanh(reference)
    rng = torch.get_rng_state().clone()
    with torch.no_grad():
        deterministic = actor(active,stochastic_output=False)
    assert torch.equal(deterministic,expected)
    assert not torch.equal(deterministic,torch.zeros_like(deterministic))
    assert not torch.equal(deterministic,HISTORY_RHO*center)
    assert torch.equal(rng,torch.get_rng_state())
    calls = []
    original_sample = actor.distribution.sample
    def sample_once():
        calls.append(1)
        return original_sample()
    monkeypatch.setattr(actor.distribution,'sample',sample_once)
    sample,audit = audited_finish_request(actor,active,lambda:actor(active,stochastic_output=True),stochastic=True)
    assert len(calls) == audit['sampling_draws'] == audit['history_kernel_applications'] == 1
    assert not audit['gain10_applied_again']
    raw = sample.detach().clone()
    old_mean,old_std = (t.detach().clone() for t in actor.output_distribution_params)
    old_logp = actor.get_output_log_prob(raw).detach().clone()
    manual = torch.distributions.Normal(old_mean.double(),old_std.double()).log_prob(raw.double()).sum(-1)
    assert torch.isfinite(old_logp).all() and torch.allclose(old_logp.double(),manual,atol=4.e-6,rtol=0.)
    assert audit['selected_raw_log_probability'] == float(old_logp[0])
    optimizer = torch.optim.Adam(actor.trainable_parameters(),lr=1.e-5)
    optimizer.zero_grad()
    (actor.output_distribution_params[0]-.2).square().mean().backward()
    optimizer.step()
    # Official model interface updates its conditional distribution; old sample
    # likelihood remains evaluated on old raw, never projected/tanh targets.
    actor(active,stochastic_output=True)
    mean,std = actor.output_distribution_params
    current_logp = actor.get_output_log_prob(raw)
    manual_current = torch.distributions.Normal(mean.double(),std.double()).log_prob(raw.double()).sum(-1)
    assert torch.allclose(current_logp.double(),manual_current,atol=4.e-6,rtol=0.)
    assert torch.isfinite(torch.exp(current_logp-old_logp)).all()
    assert not torch.equal(mean.detach(),old_mean)
    actor.assert_frozen_state(optimizer)


def test_strict_roundtrip_binding_and_positive_normalizer_freeze_checks(tmp_path):
    from wlr50_clean.ppo.semantic_rr_capture_local_actor import tensor_state_sha256
    actor,prefix,kwargs = make_actor()
    active,_ = active_observation(prefix)
    with torch.no_grad():
        expected = actor(active,stochastic_output=False)
    path = tmp_path/'actor_only_synthetic.pt'
    torch.save(actor.state_dict(),path)
    restored,_,_ = make_actor(bind=False)
    restored.load_state_dict(torch.load(path,map_location='cpu',weights_only=False),strict=True)
    restored.assert_frozen_state()
    with torch.no_grad():
        assert torch.equal(expected,restored(active,stochastic_output=False))
        assert torch.equal(actor(prefix,stochastic_output=False),restored(prefix,stochastic_output=False))
    assert restored.expected_accepted_state_sha256 == kwargs['expected_accepted_state_sha256']
    with pytest.raises(ValueError,match='strict stable'):
        restored.load_state_dict(actor.state_dict(),strict=False)
    tampered = copy.deepcopy(actor.state_dict())
    key = next(k for k in tampered if k.startswith('frozen_accepted.') and tampered[k].dtype.is_floating_point)
    tampered[key].reshape(-1)[0] += .01
    with pytest.raises(ValueError,match='binding differs'):
        restored.load_state_dict(tampered,strict=True)
    with pytest.raises(ValueError,match='one immutable'):
        actor.load_accepted_state(source_data()['actor_state_dict'],expected_sha256=tensor_state_sha256(source_data()['actor_state_dict']))
    bad_optimizer = torch.optim.Adam(actor.parameters(),lr=1.e-5)
    with pytest.raises(RuntimeError,match='excluded'):
        actor.assert_frozen_state(bad_optimizer)
