"""Bounded CPU evidence. Synthetic optimizer steps earn ZERO physical training credit."""
from __future__ import annotations
import copy
import math
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import unpad_trajectories
from wlr50_clean.ppo.semantic_history_actor import SemanticCapTransitionQuarterHistoryMLPModel
from wlr50_clean.ppo.semantic_policy_distribution import REQUEST_HISTORY_CAPS, REQUEST_HISTORY_SCALES
from candidate_actor import PhysicalInnovationCandidate, FR_KNEE_SIGMA_SCALE, CANDIDATE_VERSION

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CP = ROOT / 'outputs/ppo_fl_capture_quality_v1/checkpoints/history/checkpoint_step_000183552.pt'


@pytest.fixture(autouse=True)
def cpu_scope(monkeypatch):
    rng, threads = torch.get_rng_state(), torch.get_num_threads()
    torch.manual_seed(924112)
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    yield
    torch.set_rng_state(rng)
    torch.set_num_threads(threads)


@pytest.fixture(scope='module')
def checkpoint():
    result = torch.load(CP, map_location='cpu', weights_only=False)
    assert result['infos']['global_policy_decisions'] == 183552
    return result


def td(data):
    return TensorDict({'policy': data, 'critic': data.clone()}, batch_size=data.shape[:-1])


def data_for(phases):
    data = torch.zeros(len(phases), 372)
    for i, phase in enumerate(phases):
        data[i, phase] = 1
        data[i, 158:158 + phase] = 1
        data[i, 30] = i  # Unique bounded synthetic-observation ID for minibatch alignment.
        data[i, 195:207] = torch.linspace(-.31, .23, 12)
        data[i, 207:219] = (torch.tensor(REQUEST_HISTORY_CAPS[max(0, phase - 1)])
            * .15 / torch.tensor(REQUEST_HISTORY_SCALES))
    return data


def actor(checkpoint, cls=PhysicalInnovationCandidate):
    cfg = copy.deepcopy(checkpoint['infos']['runner_config']['actor'])
    cfg.pop('class_name')
    result = cls(td(data_for([4])), {'actor': ['policy']}, 'actor', 12, **cfg)
    result.load_state_dict(checkpoint['actor_state_dict'], strict=True)
    return result


def exact(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left: exact(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right): exact(a, b)
    else:
        assert left == right


def test_CP183552_all_weights_topology_normalizer_and_deterministic_cache_rng_unchanged(checkpoint):
    old, new = actor(checkpoint, SemanticCapTransitionQuarterHistoryMLPModel), actor(checkpoint)
    exact(old.state_dict(), new.state_dict())
    assert set(dict(old.named_parameters())) == set(dict(new.named_parameters()))
    assert set(dict(old.named_buffers())) == set(dict(new.named_buffers()))
    assert set(vars(old)) == set(vars(new))
    obs = td(data_for(list(range(13))))
    old(obs, stochastic_output=True); new(obs, stochastic_output=True)
    cached = tuple(x.clone() for x in new.output_distribution_params)
    rng = torch.get_rng_state()
    assert torch.equal(old(obs), new(obs))
    assert torch.equal(rng, torch.get_rng_state())
    exact(cached, new.output_distribution_params)
    exact(checkpoint['actor_state_dict'], new.state_dict())


@pytest.mark.parametrize('phase', range(13))
def test_all_phase_single_delta_sampling_one_draw_and_rng_exact(checkpoint, phase, monkeypatch):
    old, new = actor(checkpoint, SemanticCapTransitionQuarterHistoryMLPModel), actor(checkpoint)
    obs = td(data_for([phase]))
    counts = {'mlp': 0, 'sample': 0}
    hook = new.mlp.register_forward_hook(lambda *args: counts.__setitem__('mlp', counts['mlp'] + 1))
    sample = new.distribution.sample
    def counted_sample():
        counts['sample'] += 1
        return sample()
    monkeypatch.setattr(new.distribution, 'sample', counted_sample)
    rng = torch.get_rng_state()
    expected = old(obs, stochastic_output=True)
    after = torch.get_rng_state()
    torch.set_rng_state(rng)
    actual = new(obs, stochastic_output=True)
    hook.remove()
    assert counts == {'mlp': 1, 'sample': 1}
    assert torch.equal(after, torch.get_rng_state())
    assert torch.equal(old.output_mean, new.output_mean)
    if phase < 5:
        assert torch.equal(expected, actual)
        assert torch.equal(old.output_std, new.output_std)
        assert torch.equal(old.get_output_log_prob(actual), new.get_output_log_prob(actual))
    else:
        unchanged = torch.arange(12) != 3
        assert torch.equal(expected[:, unchanged], actual[:, unchanged])
        assert torch.equal(old.output_std[:, unchanged], new.output_std[:, unchanged])
        # log-space addition and exp have float32 rounding distinct from direct
        # multiplication. Unselected channels above still require exact equality.
        torch.testing.assert_close(new.output_std[:, 3], old.output_std[:, 3] * FR_KNEE_SIGMA_SCALE, rtol=1e-6, atol=0)
        old_z = (expected[:, 3] - old.output_mean[:, 3]) / old.output_std[:, 3]
        new_z = (actual[:, 3] - new.output_mean[:, 3]) / new.output_std[:, 3]
        torch.testing.assert_close(old_z, new_z, rtol=2e-5, atol=2e-6)


def test_mixed_batch_age_repeat_padding_likelihood_entropy_KL_and_full_support(checkpoint):
    old, new = actor(checkpoint, SemanticCapTransitionQuarterHistoryMLPModel), actor(checkpoint)
    values = data_for([4, 5, 12, 0, 5, 7])
    values[4, 20] = 1/3000  # P06 beyond entry: scaling persists, mean gate unchanged.
    obs = td(values)
    old(obs, stochastic_output=True)
    sampled = new(obs, stochastic_output=True)
    mean, std = (x.clone() for x in new.output_distribution_params)
    old_logp = new.get_output_log_prob(sampled).clone()
    assert torch.equal(mean, old.output_mean)
    normal = torch.distributions.Normal(mean, std)
    assert torch.equal(new.get_output_log_prob(sampled), normal.log_prob(sampled).sum(-1))
    assert torch.equal(new.output_entropy, normal.entropy().sum(-1))
    assert torch.equal(new.get_kl_divergence((mean, std), (mean, std)), torch.zeros(6))
    shifted = (mean + .01, std * 1.1)
    expected_kl = torch.distributions.kl_divergence(normal, torch.distributions.Normal(*shifted)).sum(-1)
    assert torch.equal(new.get_kl_divergence((mean, std), shifted), expected_kl)
    # A positive Gaussian sigma retains unbounded raw support. This does not
    # assert that sampled physical target/actual displacement is unbounded.
    tails = sampled.clone(); tails[:, 3] = torch.tensor([-3., 3., -4., 4., -5., 5.])
    assert torch.isfinite(new.get_output_log_prob(tails)).all()
    assert (tails[1:3, 3].tanh().abs() * 112 > 111).all()  # P06/P13 only.
    before = torch.get_rng_state(); deterministic = new(obs)
    assert torch.equal(before, torch.get_rng_state())
    assert torch.equal(deterministic, new(obs))
    order = torch.tensor([2, 4, 0, 5, 3, 1])
    new(td(values[order]), stochastic_output=True)
    torch.testing.assert_close(new.output_mean, mean[order], rtol=1e-6, atol=2e-7)
    torch.testing.assert_close(new.output_std, std[order], rtol=1e-6, atol=2e-7)
    ratio = (new.get_output_log_prob(sampled[order]) - old_logp[order]).exp()
    torch.testing.assert_close(ratio, torch.ones_like(ratio), rtol=0, atol=2e-5)
    padded = values.reshape(2, 3, 372).clone()
    mask = torch.tensor([[True, True, True], [True, False, False]])
    padded[~mask] = float('nan')
    unpadded = unpad_trajectories(td(padded), mask)
    rng = torch.get_rng_state(); expected = new(unpadded, stochastic_output=True)
    after = torch.get_rng_state(); torch.set_rng_state(rng)
    actual = new(td(padded), masks=mask, stochastic_output=True)
    assert torch.equal(expected, actual) and torch.equal(after, torch.get_rng_state())


@pytest.mark.parametrize('fault', ['nan', 'no_stage', 'two_stages', 'bad_age'])
def test_invalid_stage_is_not_silently_treated_as_a_sigma_gate(checkpoint, fault):
    model = actor(checkpoint); values = data_for([5])
    if fault == 'nan': values[0, 3] = float('nan')
    elif fault == 'no_stage': values[0, :13] = 0
    elif fault == 'two_stages': values[0, 0] = 1
    else: values[0, 20] = -.1
    with pytest.raises(ValueError): model(td(values), stochastic_output=True)


def make_ppo(checkpoint):
    initial = td(data_for([4]))
    cfg = copy.deepcopy(checkpoint['infos']['runner_config']['critic']); cfg.pop('class_name')
    critic = MLPModel(initial, {'critic': ['critic']}, 'critic', 1, **cfg)
    algo_cfg = copy.deepcopy(checkpoint['infos']['runner_config']['algorithm'])
    algo_cfg.pop('class_name'); algo_cfg.pop('share_cnn_encoders')
    result = PPO(actor(checkpoint), critic, RolloutStorage('rl', 1, 16, initial, [12], device='cpu'),
        device='cpu', **algo_cfg)
    result.load(copy.deepcopy(checkpoint), load_cfg=None, strict=True)
    # Mirror existing resume's effective LR restore without changing saved config.
    result.learning_rate = checkpoint['infos']['optimizer_learning_rate']
    return result


def test_one_official_synthetic_PPO_update_and_output_only_save_reload(checkpoint, tmp_path, record_property):
    algorithm = make_ppo(checkpoint)
    before = copy.deepcopy(algorithm.save())
    for key in ('actor_state_dict', 'critic_state_dict', 'optimizer_state_dict'):
        exact(before[key], checkpoint[key])
    assert algorithm.learning_rate == 1e-5
    assert all(group['lr'] == 1e-5 for group in algorithm.optimizer.param_groups)
    values = data_for([0, 4, 5, 6, 12, 2, 5, 4] * 2)
    with torch.no_grad():
        for i in range(16):
            observation = td(values[i:i+1])
            algorithm.act(observation)
            algorithm.process_env_step(observation, torch.tensor([math.sin(i)*.01]), torch.tensor([i == 15]), {})
        algorithm.compute_returns(observation)
    frozen_storage = copy.deepcopy(algorithm.storage)
    first_max_ratio_error = None
    minibatches, uses = [], torch.zeros(16, dtype=torch.int64)
    original_generator = algorithm.storage.mini_batch_generator
    def checked_generator(*args):
        nonlocal first_max_ratio_error
        for batch in original_generator(*args):
            ids = batch.observations['policy'][:, 30].long()
            assert torch.equal(batch.actions, frozen_storage.actions[ids, 0])
            assert torch.equal(batch.old_actions_log_prob, frozen_storage.actions_log_prob[ids, 0])
            for a, b in zip(batch.old_distribution_params, frozen_storage.distribution_params):
                assert torch.equal(a, b[ids, 0])
            uses[ids] += 1
            minibatches.append(ids.tolist())
            yield batch
    algorithm.storage.mini_batch_generator = checked_generator
    original_log_prob = algorithm.actor.get_output_log_prob
    def checked_log_prob(actions):
        nonlocal first_max_ratio_error
        result = original_log_prob(actions)
        if first_max_ratio_error is None:
            ids = torch.tensor(minibatches[-1])
            old = frozen_storage.actions_log_prob[ids, 0].squeeze(-1)
            first_max_ratio_error = float(((result - old).exp() - 1).abs().max())
        return result
    algorithm.actor.get_output_log_prob = checked_log_prob
    losses = algorithm.update()
    assert len(minibatches) == 20 and bool((uses == 5).all())
    assert first_max_ratio_error < 2e-5
    assert all(math.isfinite(float(x)) for x in losses.values())
    assert any(not torch.equal(v, before['actor_state_dict'][k]) for k, v in algorithm.actor.state_dict().items())
    assert algorithm.storage.step == 0
    payload = copy.deepcopy(algorithm.save())
    payload['candidate_only'] = {'version': CANDIDATE_VERSION, 'not_for_physics_or_production_resume': True,
        'real_policy_decisions': 0, 'real_PPO_updates': 0, 'synthetic_samples': 16, 'synthetic_optimizer_steps': 20,
        'effective_learning_rate_after_synthetic_update': algorithm.learning_rate}
    payload['test_rng_state'] = torch.get_rng_state().clone()
    path = tmp_path / 'NOT_FOR_PRODUCTION.synthetic.pt'
    torch.save(payload, path)
    restored = make_ppo(checkpoint)
    reloaded = torch.load(path, map_location='cpu', weights_only=False)
    restored.load(reloaded, load_cfg=None, strict=True)
    restored.learning_rate = reloaded['candidate_only']['effective_learning_rate_after_synthetic_update']
    assert restored.learning_rate == algorithm.learning_rate
    assert all(group['lr'] == restored.learning_rate for group in restored.optimizer.param_groups)
    exact(algorithm.save(), restored.save())
    check_obs = td(values)
    assert torch.equal(algorithm.actor(check_obs), restored.actor(check_obs))
    torch.set_rng_state(payload['test_rng_state'])
    expected = algorithm.actor(check_obs, stochastic_output=True)
    after = torch.get_rng_state(); torch.set_rng_state(payload['test_rng_state'])
    assert torch.equal(expected, restored.actor(check_obs, stochastic_output=True))
    assert torch.equal(after, torch.get_rng_state())
    record_property('first_minibatch_max_ratio_error', first_max_ratio_error)
    record_property('synthetic_optimizer_steps', 20)
    record_property('real_training_credit', 0)
