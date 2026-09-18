"""Exact half/quarter prefix compatibility; CPU only, no physical rollout credit."""
from __future__ import annotations

import copy
from types import MethodType
import pytest

from wlr50_clean.ppo import semantic_training as t
from wlr50_clean.ppo.semantic_checkpoint_prefix_policy import build_frozen_checkpoint_prefix_policy
from wlr50_clean.ppo.semantic_policy_distribution import (
    HISTORY_POLICY, HISTORY_TEMPERED_POLICY, HISTORY_QUARTER_TEMPERED_POLICY, policy_contract)
from wlr50_clean.ppo.semantic_transfer_roles import ROLE_OBSERVATION_LAYOUT
from test_semantic_quarter_temperature_migration import make_runner


@pytest.fixture
def cpu(monkeypatch):
    torch = pytest.importorskip('torch')
    threads, state = torch.get_num_threads(), torch.get_rng_state()
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    yield torch
    torch.set_rng_state(state)
    torch.set_num_threads(threads)


def provenance(actor, version):
    return {'checkpoint_path': 'synthetic_CPU_constructor_only.pt', 'checkpoint_sha256': 'a'*64,
        'actor_parameter_sha256': t.parameter_hash(actor), 'source_global_policy_decisions': 0,
        'source_ppo_updates': 0, 'policy_contract': policy_contract(version, observation_layout=ROLE_OBSERVATION_LAYOUT)}


@pytest.mark.parametrize('version', [HISTORY_POLICY, HISTORY_TEMPERED_POLICY, HISTORY_QUARTER_TEMPERED_POLICY])
def test_exact_class_frozen_mean_no_rng_shared_state_or_source_cache_change(cpu, version):
    runner, env = make_runner(version)
    actor = runner.alg.actor
    obs = env.get_observations()
    obs['policy'][0, 195:207] = cpu.linspace(-.7, .8, 12)
    with cpu.inference_mode():
        actor(obs, stochastic_output=True)
        expected = actor(obs, stochastic_output=False)[0].tolist()
    cache = actor.distribution._distribution
    state = cpu.get_rng_state().clone()
    parameters = t.parameter_hash(actor)
    optimizer = t.state_hash(runner.alg.optimizer.state_dict())
    frozen = build_frozen_checkpoint_prefix_policy(actor, provenance(actor, version))
    result = frozen(tuple(obs['policy'][0].tolist()))
    assert result == tuple(expected)
    assert cpu.equal(state, cpu.get_rng_state())
    assert actor.distribution._distribution is cache
    assert t.parameter_hash(actor) == parameters
    assert t.state_hash(runner.alg.optimizer.state_dict()) == optimizer
    assert frozen.provenance['independent_parameter_and_buffer_storage_verified'] is True
    assert frozen.provenance['distribution_cache_copied'] is False
    assert frozen.provenance['frozen_for_entire_training_block'] is True
    assert runner.alg.storage.step == 0 and runner.alg.transition.actions is None


@pytest.mark.parametrize('fault', ['wrong_declared_version', 'instance_forward', 'identity_normalizer'])
def test_rejects_wrong_kernel_or_preprocessing(cpu, fault):
    runner, _ = make_runner(HISTORY_QUARTER_TEMPERED_POLICY)
    actor = runner.alg.actor
    source = provenance(actor, HISTORY_QUARTER_TEMPERED_POLICY)
    if fault == 'wrong_declared_version':
        source['policy_contract'] = policy_contract(HISTORY_TEMPERED_POLICY, observation_layout=ROLE_OBSERVATION_LAYOUT)
    elif fault == 'instance_forward':
        original = actor.forward
        actor.forward = MethodType(lambda self, *a, **kw: original(*a, **kw), actor)
    else:
        actor.obs_normalizer = cpu.nn.Sequential(cpu.nn.Identity())
    with pytest.raises(ValueError):
        build_frozen_checkpoint_prefix_policy(actor, source)


def test_frozen_quarter_retains_observation_history_after_source_parameter_update(cpu):
    runner, env = make_runner(HISTORY_QUARTER_TEMPERED_POLICY)
    actor = runner.alg.actor
    frozen = build_frozen_checkpoint_prefix_policy(actor, provenance(actor, HISTORY_QUARTER_TEMPERED_POLICY))
    obs = env.get_observations()
    vector = tuple(obs['policy'][0].tolist())
    before = frozen(vector)
    with cpu.no_grad():
        next(actor.parameters()).add_(.3)
    assert frozen(vector) == before
    changed = list(vector)
    changed[195:207] = [v+.5 for v in changed[195:207]]
    assert frozen(tuple(changed)) != before
