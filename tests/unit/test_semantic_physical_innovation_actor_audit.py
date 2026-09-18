"""Production actor/request audit only: CPU evidence, zero real training credit."""
from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict

from wlr50_clean.ppo.semantic_history_actor import (
    SemanticCapTransitionQuarterHistoryMLPModel,
    SemanticFRKneePhysicalInnovationHistoryMLPModel,
    physical_innovation_effective_log_std,
)
from wlr50_clean.ppo.semantic_policy_distribution import (
    FR_KNEE_PHYSICAL_INNOVATION_POLICY, FR_KNEE_PHYSICAL_SIGMA_SEMANTICS,
    HISTORY_REQUEST_CAP_TRANSITION_POLICY, REQUEST_HISTORY_CAPS, REQUEST_HISTORY_SCALES,
    policy_contract, policy_version_from_metadata, supported_heteroscedastic_contract_version,
)
from wlr50_clean.ppo.semantic_training import audited_history_policy_request, semantic_runner_config
from wlr50_clean.ppo.semantic_transfer_roles import ROLE_OBSERVATION_LAYOUT


@pytest.fixture(autouse=True)
def cpu_scope():
    state, threads = torch.get_rng_state(), torch.get_num_threads()
    torch.manual_seed(24112)
    torch.set_num_threads(1)
    yield
    torch.set_rng_state(state)
    torch.set_num_threads(threads)


def observation(phase, *, age=0.):
    x = torch.zeros(1, 372)
    x[0, phase] = 1
    x[0, 158:158+phase] = 1
    x[0, 20] = age
    x[0, 195:207] = torch.linspace(-.31, .23, 12)
    x[0, 207:219] = torch.tensor(REQUEST_HISTORY_CAPS[max(phase-1, 0)])*.15/torch.tensor(REQUEST_HISTORY_SCALES)
    return TensorDict({'policy': x, 'critic': x.clone()}, batch_size=[1])


def actor(cls, obs):
    return cls(obs, {'actor':['policy']}, 'actor', 12, observation_layout=ROLE_OBSERVATION_LAYOUT,
        exploration_std_temperature=.25,
        distribution_cfg={'class_name':'HeteroscedasticGaussianDistribution', 'std_type':'log', 'init_std':.15})


@pytest.mark.parametrize('phase', range(13))
def test_all_phases_one_MLP_one_draw_exact_mean_history_other_channels_and_actual_sigma_audit(phase):
    obs = observation(phase)
    old = actor(SemanticCapTransitionQuarterHistoryMLPModel, obs)
    new = actor(SemanticFRKneePhysicalInnovationHistoryMLPModel, obs)
    new.load_state_dict(old.state_dict(), strict=True)
    calls = []
    handle = new.mlp.register_forward_hook(lambda *args: calls.append(1))
    before = torch.get_rng_state()
    expected = old(obs, stochastic_output=True)
    after = torch.get_rng_state()
    torch.set_rng_state(before)
    result, audit = audited_history_policy_request(new, obs, lambda:new(obs, stochastic_output=True), stochastic=True)
    handle.remove()
    assert len(calls) == 1 and torch.equal(after, torch.get_rng_state())
    assert torch.equal(old.output_mean, new.output_mean)
    keep = torch.ones(12, dtype=torch.bool)
    if phase >= 5: keep[3] = False
    assert torch.equal(expected[:,keep], result[:,keep])
    assert torch.equal(old.output_std[:,keep], new.output_std[:,keep])
    if phase >= 5:
        torch.testing.assert_close(new.output_std[:,3], old.output_std[:,3]*(24/112), rtol=1e-6, atol=0.)
    assert torch.equal(torch.tensor(audit['effective_sigma_full12']), new.output_std[0])
    assert torch.equal(torch.tensor(audit['conditional_mean_full12']), new.output_mean[0])
    assert audit['selected_raw_log_probability'] == new.get_output_log_prob(result)[0].item()
    assert audit['policy_version'] == FR_KNEE_PHYSICAL_INNOVATION_POLICY
    assert audit['sigma_scaling_semantics'] == FR_KNEE_PHYSICAL_SIGMA_SEMANTICS
    assert audit['sigma_scaling_gate_full12'] == [phase>=5 and i==3 for i in range(12)]
    assert audit['sampling_draws'] == 1 and audit['extra_random_draws'] == audit['extra_model_forwards'] == 0
    normal = torch.distributions.Normal(new.output_mean, new.output_std)
    assert torch.equal(new.get_output_log_prob(result), normal.log_prob(result).sum(-1))
    assert torch.equal(new.output_entropy, normal.entropy().sum(-1))
    cached = tuple(x.clone() for x in new.output_distribution_params)
    before = torch.get_rng_state()
    fixed, fixed_audit = audited_history_policy_request(new, obs, lambda:new(obs), stochastic=False)
    assert torch.equal(fixed, old(obs)) and torch.equal(before, torch.get_rng_state())
    assert all(torch.equal(x,y) for x,y in zip(cached,new.output_distribution_params))
    assert fixed_audit['sampling_draws'] == 0 and fixed_audit['effective_sigma_full12'] == audit['effective_sigma_full12']
    assert set(dict(old.named_parameters())) == set(dict(new.named_parameters()))
    assert set(dict(old.named_buffers())) == set(dict(new.named_buffers()))
    assert all(torch.equal(v,new.state_dict()[k]) for k,v in old.state_dict().items())


def test_same_phase_age_gate_no_new_history_cache_and_full_raw_support():
    obs = observation(5, age=.02)
    model = actor(SemanticFRKneePhysicalInnovationHistoryMLPModel, obs)
    _, audit = audited_history_policy_request(model, obs, lambda:model(obs, stochastic_output=True), stochastic=True)
    assert not any(audit['cap_transition_gate_full12'])
    assert audit['history_center_full12'] == audit['previous_raw_from_current_observation_full12']
    assert audit['sigma_scaling_gate_full12'][3] is True
    tails = torch.zeros(2,12); tails[:,3] = torch.tensor([-5.,5.])
    assert torch.isfinite(model.get_output_log_prob(tails)).all()
    assert (tails[:,3].tanh().abs()*112>111).all()


@pytest.mark.parametrize('fault', ('nan','zero_stage','two_stages','negative_age','wrong_temperature','wrong_shape','overflow','underflow'))
def test_shared_sigma_rule_rejects_invalid_input_without_silent_clipping(fault):
    x = observation(5)['policy']; log_std=torch.zeros(1,12); temp=.25
    if fault=='nan': x[0,0]=float('nan')
    elif fault=='zero_stage':x[0,:13]=0
    elif fault=='two_stages':x[0,0]=1
    elif fault=='negative_age':x[0,20]=-.1
    elif fault=='wrong_temperature':temp=.5
    elif fault=='wrong_shape':log_std=torch.zeros(1,11)
    elif fault=='overflow':log_std[0,3]=1000
    else:log_std[0,3]=-1000
    with pytest.raises(ValueError): physical_innovation_effective_log_std(log_std,x,temp)


def test_exact_class_allowlist_does_not_accept_arbitrary_subclass():
    class Unreviewed(SemanticFRKneePhysicalInnovationHistoryMLPModel):pass
    obs=observation(5); model=actor(Unreviewed,obs)
    with pytest.raises(ValueError,match='exact supported'):
        audited_history_policy_request(model,obs,lambda:model(obs,stochastic_output=True),stochastic=True)


def test_complete_contract_runner_and_real_source_state_shape_are_explicit():
    source=semantic_runner_config(seed=1001,device='cpu',semantic_version='v3',
        policy_version=HISTORY_REQUEST_CAP_TRANSITION_POLICY,observation_layout=ROLE_OBSERVATION_LAYOUT)
    target=semantic_runner_config(seed=1001,device='cpu',semantic_version='v3',
        policy_version=FR_KNEE_PHYSICAL_INNOVATION_POLICY,observation_layout=ROLE_OBSERVATION_LAYOUT)
    old=copy.deepcopy(source); new=copy.deepcopy(target)
    old['actor'].pop('class_name');new['actor'].pop('class_name');assert old==new
    contract=policy_contract(FR_KNEE_PHYSICAL_INNOVATION_POLICY,observation_layout=ROLE_OBSERVATION_LAYOUT)
    assert supported_heteroscedastic_contract_version(contract)==FR_KNEE_PHYSICAL_INNOVATION_POLICY
    metadata={'seed':1001,'semantic_version':'v3','runner_config':target,'policy_contract':contract}
    assert policy_version_from_metadata(metadata)==FR_KNEE_PHYSICAL_INNOVATION_POLICY
    malformed=copy.deepcopy(metadata);malformed['policy_contract']['innovation_scale_channel_index']=7
    with pytest.raises(ValueError):policy_version_from_metadata(malformed)
    with pytest.raises(ValueError):semantic_runner_config(seed=1001,semantic_version='v2',policy_version=FR_KNEE_PHYSICAL_INNOVATION_POLICY)
    checkpoint=Path(__file__).resolve().parents[2]/'outputs/ppo_fl_capture_quality_v1/checkpoints/history/checkpoint_step_000184320.pt'
    if checkpoint.exists():
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert payload['infos']['global_policy_decisions']==184320
        model=actor(SemanticFRKneePhysicalInnovationHistoryMLPModel,observation(5))
        model.load_state_dict(payload['actor_state_dict'],strict=True)
        assert all(torch.equal(v,model.state_dict()[k]) for k,v in payload['actor_state_dict'].items())
