"""Bounded CPU runner wiring; synthetic PPO computations carry ZERO run credit.

The source is the actual sealed B1 payload. Only training RNG restoration is
mocked (and its arguments checked); no GPU RNG restoration is claimed. No source
payload, run counter, checkpoint pointer or production output is written.
"""
import copy
import sys
from pathlib import Path

import pytest
import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'src/wlr50_clean').is_dir())
sys.path.insert(0, str(ROOT/'src'))
from wlr50_clean.ppo import semantic_fl_prior_retirement as route
from wlr50_clean.ppo import rl_library_wrapper
from wlr50_clean.ppo.semantic_fl_prior_retirement_actor import PriorRetirementActor
from wlr50_clean.ppo.semantic_training import state_hash


@pytest.fixture
def migrated(monkeypatch):
    def no_cuda(*args, **kwargs):
        raise AssertionError('GPU initialization forbidden in synthetic CPU test')
    monkeypatch.setattr(torch.cuda, 'init', no_cuda)
    monkeypatch.setattr(torch.cuda, '_lazy_init', no_cuda)
    restores = []
    def record_only_restore(value, *, expected_seed):
        restores.append((copy.deepcopy(value), expected_seed))
    monkeypatch.setattr(rl_library_wrapper, 'restore_training_rng_state', record_only_restore)
    previous_threads = torch.get_num_threads()
    previous_rng = torch.get_rng_state().clone()
    torch.set_num_threads(1)
    try:
        runner = route.make_runner('cpu')
        lineage, counts = route.migrate_from_B535(runner)
        source, meta = route.source_metadata()
        payload = torch.load(source, map_location='cpu', weights_only=False)
        assert restores == [(meta['training_rng'],1001)]*2  # strict old loader + migration boundary
        yield runner, lineage, counts, payload, meta
        assert not torch.cuda.is_initialized()
    finally:
        torch.set_rng_state(previous_rng)
        torch.set_num_threads(previous_threads)


def test_actual_B_actor_critic_named_Adam_steps_LR_and_counts(migrated):
    runner, lineage, counts, payload, meta = migrated
    assert type(runner.alg.actor) is PriorRetirementActor
    assert runner.alg.actor.obs_dim == 536
    assert state_hash(runner.alg.actor.state_dict()) == state_hash(payload['actor_state_dict'])
    collapsed = copy.deepcopy(runner.alg.critic.state_dict())
    changes = lineage['prior_retirement_migration']['critic_zero_columns']
    assert changes == ['mlp.0.weight']
    assert not torch.count_nonzero(collapsed[changes[0]][:,535:])
    collapsed[changes[0]] = collapsed[changes[0]][:,:535].clone()
    assert state_hash(collapsed) == state_hash(payload['critic_state_dict'])
    oldopt = payload['optimizer_state_dict']
    newopt = copy.deepcopy(runner.alg.optimizer.state_dict())
    assert newopt['param_groups'] == oldopt['param_groups']
    names = route._parameters(runner)
    ids = newopt['param_groups'][0]['params']
    for (name, parameter), pid in zip(names, ids, strict=True):
        for key, old in oldopt['state'][pid].items():
            new = newopt['state'][pid][key]
            if torch.is_tensor(old) and new.shape != old.shape:
                assert name == 'critic.mlp.0.weight' and old.ndim == 2
                assert not torch.count_nonzero(new[:,535:])
                newopt['state'][pid][key] = new[:,:535].clone()
    assert state_hash(newopt) == state_hash(oldopt)
    assert runner.alg.learning_rate == meta['learning_rate'] == .01
    assert all(g['lr'] == .01 for g in runner.alg.optimizer.param_groups)
    assert counts == meta['counts'] and not any(route.retirement_counts(lineage,counts).values())
    assert runner.current_learning_iteration == counts['ppo_updates']
    assert runner.alg.storage.step == 0 and runner.alg.transition.actions is None
    runner.alg.actor.assert_frozen_state(runner.alg.optimizer)


def sample_observation(index, active):
    values = [0.]*536
    values[5 if active else 6] = 1.
    values[20] = .1
    values[203] = -1.1747
    values[531:535] = [float(active),.5,1.,0.]
    values[535] = min(1.,index/15.) if active else 0.
    return route.tensor_observation(values,'cpu')


def collect_synthetic512(runner, *, all_inactive):
    raw_rows, log_rows, flags = [], [], []
    for i in range(512):
        active = not all_inactive and i%4 == 0
        observation = sample_observation(i,active)
        next_observation = sample_observation(i+1,not all_inactive and (i+1)%4 == 0)
        with torch.no_grad():
            raw, audit = route.request(runner,observation,stochastic=True)
            assert audit['active'] == active and audit['density_dimension'] == 1
            assert audit['sampling_draws'] == int(active)
            assert audit['public_prior_retirement_is_not_learned_delta']
            assert audit['selected_raw_log_probability'] == float(runner.alg.transition.actions_log_prob[0])
            raw_rows.append(raw.clone())
            log_rows.append(runner.alg.transition.actions_log_prob.clone())
            flags.append(active)
            runner.alg.process_env_step(next_observation,torch.tensor([-.001+.0001*(i%5)]),
                                        torch.tensor([False]),{})
    storage = runner.alg.storage
    assert storage.step == 512 and not storage.dones.any()
    assert storage.observations['policy'].shape[-1] == 536
    assert torch.equal(storage.actions,torch.stack(raw_rows))
    assert torch.equal(storage.actions_log_prob.view(512),torch.stack(log_rows).view(512))
    assert storage.distribution_params[2][:,0,0].bool().tolist() == flags
    assert all(p.shape == (512,1,1) for p in storage.distribution_params)
    with torch.no_grad(): runner.alg.compute_returns(next_observation)
    assert torch.isfinite(storage.returns).all() and torch.isfinite(storage.advantages).all()
    return flags


def test_contiguous_mixed512_actual_new_request_and_official_PPO(migrated):
    runner, lineage, counts, _, _ = migrated
    counts_before = copy.deepcopy(counts)
    frozen = state_hash(runner.alg.actor.frozen_A.state_dict())
    flags = collect_synthetic512(runner,all_inactive=False)
    report = route.update_complete_rollout(runner)
    assert sum(flags) == report['stochastic_FL_rows'] == 128
    assert report['total_continuous_storage_rows'] == 512
    assert report['critic_only_context_rows'] == 384
    assert report['optimizer_steps'] == 20 and report['actor_optimizer_steps'] > 0
    assert report['extra_model_forwards'] == report['extra_random_draws'] == 0
    assert state_hash(runner.alg.actor.frozen_A.state_dict()) == frozen
    assert runner.alg.storage.step == 0 and runner.alg.transition.actions is None
    assert counts == counts_before  # Ephemeral synthetic update, zero experiment credit.
    assert not any(route.retirement_counts(lineage,counts).values())


def test_critic_only512_preserves_existing_actor_Adam_momentum(migrated):
    runner, _, counts, _, _ = migrated
    actor = runner.alg.actor
    before_actor = state_hash(actor.state_dict())
    before_adam = {name:copy.deepcopy(runner.alg.optimizer.state[p])
                   for name,p in actor.named_parameters() if p.requires_grad}
    critic_steps = {name:float(runner.alg.optimizer.state[p]['step'])
                    for name,p in runner.alg.critic.named_parameters()}
    counts_before = copy.deepcopy(counts)
    collect_synthetic512(runner,all_inactive=True)
    report = route.update_complete_rollout(runner)
    assert report['stochastic_FL_rows'] == report['actor_optimizer_steps'] == 0
    assert report['critic_only_context_rows'] == 512 and report['inactive_only_steps'] == 20
    assert state_hash(actor.state_dict()) == before_actor
    for name,p in actor.named_parameters():
        if p.requires_grad:
            assert p.grad is None and state_hash(runner.alg.optimizer.state[p]) == state_hash(before_adam[name])
    assert all(float(runner.alg.optimizer.state[p]['step']) == critic_steps[name]+20
               for name,p in runner.alg.critic.named_parameters())
    assert counts == counts_before
