"""DEFERRED CPU tensor/actor tests. Do not collect during active Isaac.

No real PPO samples or optimizer credit. One synthetic Adam step below only
populates CPU unit-test moments to verify full-state copy compatibility.
"""
import os
import importlib.util
from copy import deepcopy
from pathlib import Path
import sys
import pytest

if os.environ.get('COOPERATIVE_PREP_TENSOR_TESTS_IDLE') != '1':
    pytest.skip('root must confirm idle Isaac before Torch testing', allow_module_level=True)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'src/wlr50_clean').is_dir())
sys.path.insert(0, str(ROOT/'src'))
import torch
from tensordict import TensorDict

# Only in this explicit future test process, load draft modules under their
# proposed package names. No production file is overwritten or imported anew.
for stem in ('semantic_rear_cooperative_prep_sigma', 'semantic_rear_cooperative_prep_profile',
             'semantic_rear_cooperative_prep_actor'):
    name = 'wlr50_clean.ppo.'+stem
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(stem+'.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

from wlr50_clean.ppo.semantic_rear_cooperative_prep_sigma import (
    cooperative_prep_effective_log_std, cooperative_prep_multipliers)
from wlr50_clean.ppo.semantic_rear_cooperative_prep_actor import SemanticRearCooperativePrepHistoryMLPModel
from wlr50_clean.ppo.semantic_rear_cooperative_prep_profile import (
    cooperative_prep_policy_contract, COOPERATIVE_PREP_POLICY)
from wlr50_clean.ppo.semantic_p02_progress_actor import (
    SemanticP02ProgressHistoryMLPModel, p02_progress_effective_log_std)
from wlr50_clean.ppo.semantic_p02_progress_profile import P02_PROGRESS_OBSERVATION_LAYOUT
from wlr50_clean.ppo.semantic_receiving_wheel_sigma import receiving_wheel_effective_log_std


@pytest.fixture(autouse=True)
def cpu_scope():
    state, threads = torch.get_rng_state(), torch.get_num_threads()
    torch.manual_seed(20260924)
    torch.set_num_threads(1)
    yield
    torch.set_rng_state(state)
    torch.set_num_threads(threads)


def latent(phase=8, count=2):
    x = torch.zeros(count, 422)
    x[:, phase] = 1.
    x[:, 20] = .025
    x[:, 158:158+phase] = 1.
    x[:, 195:207] = torch.linspace(-.7, .9, 12)
    return x


def obs(x):
    return TensorDict({'policy': x, 'critic': x.clone()}, batch_size=list(x.shape[:-1]))


def model(x, cls):
    return cls(obs(x), {'actor': ['policy']}, 'actor', 12,
        observation_layout=P02_PROGRESS_OBSERVATION_LAYOUT,
        exploration_std_temperature=.25, obs_normalization=False,
        distribution_cfg={'class_name': 'HeteroscedasticGaussianDistribution',
                          'std_type': 'log', 'init_std': .15})


@pytest.mark.parametrize('carry,reachable,prep,receiving', [
    (False, False, False, False), (True, False, False, False),
    (True, True, False, False), (False, False, True, True), (True, True, True, True)])
def test_tensor_matches_scalar_parent_totals_and_preserves_old_unselected_channels(carry, reachable, prep, receiving):
    x = latent(9 if receiving else 8)
    x[:, 410], x[:, 404], x[:, 412], x[:, 157] = float(carry), float(reachable), float(prep), float(receiving)
    head = torch.linspace(-3., -.2, x.shape[0]*12).reshape(-1, 12)
    saved, rng = x.clone(), torch.get_rng_state()
    parent, parent_ev = receiving_wheel_effective_log_std(head, x[:, :372], .25)
    old, _ = p02_progress_effective_log_std(head, x, .25)
    new, ev = cooperative_prep_effective_log_std(head, x, .25)
    expected, _ = cooperative_prep_multipliers(rr_carry_capture=carry,
        rr_top_reachable=reachable, rl_prep_transfer=prep,
        receiving_continuation_active=receiving)
    assert bool((parent_ev['receiving_continuation_active'] == receiving).all())
    torch.testing.assert_close(new, parent+torch.tensor(expected).log(), rtol=0, atol=0)
    assert torch.equal(new[:, [0, 2, 6, 7, 9, 10, 11]], old[:, [0, 2, 6, 7, 9, 10, 11]])
    assert torch.equal(x, saved) and torch.equal(torch.get_rng_state(), rng)
    assert ev['cooperative_support_transfer_permission_modified'] is False


@pytest.mark.parametrize('phase', range(6))
def test_early_front_offgate_sigma_bitwise_equal_old422(phase):
    x = latent(phase)
    head = torch.linspace(-3., -.2, x.shape[0]*12).reshape(-1, 12)
    old, _ = p02_progress_effective_log_std(head, x, .25)
    new, _ = cooperative_prep_effective_log_std(head, x, .25)
    assert torch.equal(new, old)


def test_same_state_dict_and_full_adam_survive_without_mean_change():
    x = latent(); x[:, 410] = x[:, 404] = 1.
    old = model(x, SemanticP02ProgressHistoryMLPModel)
    old_optimizer = torch.optim.Adam(old.parameters(), lr=.00003)
    old(obs(x)).square().mean().backward()
    old_optimizer.step()  # CPU synthetic fixture only; not physical PPO credit.
    state, adam = deepcopy(old.state_dict()), deepcopy(old_optimizer.state_dict())
    new = model(x, SemanticRearCooperativePrepHistoryMLPModel)
    new.load_state_dict(state, strict=True)
    new_optimizer = torch.optim.Adam(new.parameters(), lr=.00003)
    new_optimizer.load_state_dict(adam)
    assert [(n, tuple(p.shape)) for n, p in old.named_parameters()] == [
        (n, tuple(p.shape)) for n, p in new.named_parameters()]
    assert old.state_dict().keys() == new.state_dict().keys()
    assert all(torch.equal(v, new.state_dict()[k]) for k, v in state.items())
    assert torch.equal(old(obs(x)), new(obs(x)))
    other = new_optimizer.state_dict()
    assert other['param_groups'] == adam['param_groups'] and other['state'].keys() == adam['state'].keys()
    for parameter, row in adam['state'].items():
        for key, value in row.items():
            assert torch.equal(value, other['state'][parameter][key]) if isinstance(value, torch.Tensor) else value == other['state'][parameter][key]


def test_actual_actor_sample_log_probability_uses_shared_kernel_and_ratio_one_without_update():
    x = latent(); x[:, 410] = x[:, 404] = 1.
    actor = model(x, SemanticRearCooperativePrepHistoryMLPModel)
    heads = []
    handle = actor.mlp.register_forward_hook(lambda _m, _a, output: heads.append(output.detach().clone()))
    selected = actor(obs(x), stochastic_output=True)
    handle.remove()
    assert len(heads) == 1
    expected_log_std, _ = cooperative_prep_effective_log_std(heads[0][..., 1, :], x, .25)
    torch.testing.assert_close(actor.output_std, expected_log_std.exp(), rtol=0, atol=0)
    old_logp = actor.get_output_log_prob(selected).detach()
    manual = torch.distributions.Normal(actor.output_mean, actor.output_std).log_prob(selected).sum(-1)
    torch.testing.assert_close(old_logp, manual)
    actor(obs(x), stochastic_output=True)
    current_logp = actor.get_output_log_prob(selected)
    torch.testing.assert_close(torch.exp(current_logp-old_logp), torch.ones_like(old_logp))


def test_contract_explicitly_versions_only_kernel_not_observations_or_control():
    contract = cooperative_prep_policy_contract(observation_layout=P02_PROGRESS_OBSERVATION_LAYOUT)
    assert contract['version'] == COOPERATIVE_PREP_POLICY
    assert contract['observation_dimension'] == 422
    assert contract['observation_layout'] == 'role419_p02_progress_v1'
    assert contract['deterministic_mean_change'] is False
    assert contract['control_or_reward_change'] is False
    assert contract['stochastic_kernel_change'] is True
    assert contract['support_transfer_permission_modified'] is False
