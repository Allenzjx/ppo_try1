"""Synthetic CPU actor/metadata proofs; no real checkpoint or physical evaluation."""
from __future__ import annotations

import copy
import io
import json
import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rsl_rl")
from tensordict import TensorDict
from rsl_rl.utils import resolve_callable, unpad_trajectories

from wlr50_clean.ppo import semantic_policy_distribution as policy
from wlr50_clean.ppo.semantic_history_actor import (
    SemanticHistoryMLPModel, SemanticTemperedHistoryMLPModel, history_conditioned_head,
)
from wlr50_clean.ppo.semantic_training import semantic_runner_config
from wlr50_clean.ppo.semantic_transfer_roles import ROLE_OBSERVATION_LAYOUT


@pytest.fixture(autouse=True)
def cpu_only_rng_thread_scope():
    previous_threads, previous_rng = torch.get_num_threads(), torch.get_rng_state()
    torch.set_num_threads(1)
    torch.manual_seed(92361)
    yield
    torch.set_rng_state(previous_rng)
    torch.set_num_threads(previous_threads)


def observations(n=7):
    values = torch.randn(n, 372, device="cpu")
    values[:, 195:207] = torch.linspace(-2., 2., n * 12).reshape(n, 12)
    return TensorDict({"policy": values}, batch_size=[n])


def metadata(version=policy.HISTORY_TEMPERED_POLICY):
    return {"semantic_version": "v3", "seed": 1001, "policy_version": version,
            "runner_config": semantic_runner_config(seed=1001, device="cpu", semantic_version="v3",
                policy_version=version, observation_layout=ROLE_OBSERVATION_LAYOUT),
            "policy_contract": policy.policy_contract(version, observation_layout=ROLE_OBSERVATION_LAYOUT)}


def actor(version, obs):
    config = copy.deepcopy(metadata(version)["runner_config"]["actor"])
    cls = resolve_callable(config.pop("class_name"))
    return cls(obs, {"actor": ["policy"]}, "actor", 12, **config)


def pair():
    obs = observations()
    torch.manual_seed(93511)
    old = actor(policy.HISTORY_POLICY, obs)
    torch.manual_seed(93511)
    new = actor(policy.HISTORY_TEMPERED_POLICY, obs)
    return old, new, obs


def exact_tree(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            exact_tree(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for x, y in zip(left, right):
            exact_tree(x, y)
    else:
        assert type(left) is type(right) and left == right


def test_same_parameter_topology_strict_saved_reload_and_adam_fixture_state():
    old, new, obs = pair()
    assert type(old) is SemanticHistoryMLPModel and type(new) is SemanticTemperedHistoryMLPModel
    assert list(old.state_dict()) == list(new.state_dict())
    assert list(dict(old.named_parameters())) == list(dict(new.named_parameters()))
    assert list(dict(old.named_buffers())) == list(dict(new.named_buffers())) == []
    exact_tree(old.state_dict(), new.state_dict())
    assert set(vars(new)) == set(vars(old))  # Fixed property adds no mutable state.
    old_optimizer = torch.optim.Adam(old.parameters(), lr=1e-5)
    # Populate synthetic Adam moments only, not a PPO/critic/real-checkpoint update.
    sum(parameter.square().mean() for parameter in old.parameters()).backward()
    old_optimizer.step()
    old_optimizer.zero_grad(set_to_none=True)
    saved = io.BytesIO()
    torch.save({"actor": old.state_dict(), "optimizer": old_optimizer.state_dict(),
                "normalizer": old.obs_normalizer.state_dict(), "rng": torch.get_rng_state()}, saved)
    saved.seek(0)
    restored = torch.load(saved, weights_only=True)
    new.load_state_dict(restored["actor"], strict=True)
    new_optimizer = torch.optim.Adam(new.parameters(), lr=3e-5)
    new_optimizer.load_state_dict(restored["optimizer"])
    exact_tree(old.state_dict(), new.state_dict())
    exact_tree(old_optimizer.state_dict(), new_optimizer.state_dict())
    exact_tree(restored["normalizer"], new.obs_normalizer.state_dict())
    assert new_optimizer.param_groups[0]["lr"] == 1e-5
    torch.set_rng_state(restored["rng"])
    assert torch.equal(new(obs), old(obs))
    assert torch.equal(torch.get_rng_state(), restored["rng"])
    assert not new.is_recurrent and new.get_hidden_state() is None


def test_frozen_deterministic_mean_history_cache_rng_and_inputs_bitwise_unchanged():
    old, new, obs = pair()
    new(obs, stochastic_output=True)
    cache = new.distribution._distribution
    state, original, rng = copy.deepcopy(new.state_dict()), obs.clone(), torch.get_rng_state()
    new.train()
    for values in (obs, obs.flip(0)):
        assert torch.equal(new(values), old(values))
    assert torch.equal(rng, torch.get_rng_state())
    assert new.distribution._distribution is cache and new.training
    exact_tree(state, new.state_dict())
    assert torch.equal(obs["policy"], original["policy"])
    # Existing raw history, not actor-private history, still drives the mean.
    changed = obs.clone()
    changed["policy"][:, 195:207].neg_()
    assert torch.equal(new(changed), old(changed))
    assert not torch.equal(new(changed), new(obs))
    new.reset(torch.ones(len(obs), dtype=torch.bool))
    assert torch.equal(new(obs), old(obs))


def test_all_stochastic_consumers_use_one_scaled_cached_distribution_and_rng_draw():
    old, new, obs = pair()
    latent = old.get_latent(obs)
    head = history_conditioned_head(old.mlp(latent), latent[:, 195:207])
    mean, learned_sigma = head[:, 0], head[:, 1].exp()
    effective_sigma = (head[:, 1] + math.log(.5)).exp()
    expected = torch.distributions.Normal(mean, effective_sigma)
    rng = torch.get_rng_state()
    actions = new(obs, stochastic_output=True)
    after = torch.get_rng_state()
    torch.set_rng_state(rng)
    assert torch.equal(actions, expected.sample())
    assert torch.equal(torch.get_rng_state(), after)
    assert torch.equal(new.output_mean, mean) and torch.equal(new.output_std, effective_sigma)
    torch.testing.assert_close(effective_sigma, .5 * learned_sigma, rtol=1e-6, atol=0.)
    assert torch.equal(new.get_output_log_prob(actions), expected.log_prob(actions).sum(-1))
    assert torch.equal(new.output_entropy, expected.entropy().sum(-1))
    exact_tree(new.output_distribution_params, (mean, effective_sigma))
    old_params = tuple(value.detach().clone() for value in new.output_distribution_params)
    old_log_prob = new.get_output_log_prob(actions).detach().clone()
    new(obs, stochastic_output=True)  # Same official update-side cache rebuild.
    assert torch.equal(torch.exp(new.get_output_log_prob(actions) - old_log_prob), torch.ones(len(obs)))
    assert torch.equal(new.get_kl_divergence(old_params, new.output_distribution_params), torch.zeros(len(obs)))
    unscaled = torch.distributions.Normal(mean, learned_sigma)
    assert not torch.equal(new.get_output_log_prob(actions), unscaled.log_prob(actions).sum(-1))
    torch.testing.assert_close(new.output_entropy - unscaled.entropy().sum(-1),
                               torch.full((len(obs),), 12 * math.log(.5)), rtol=1e-6, atol=2e-6)
    with torch.no_grad():
        new.mlp[4].bias[:12].add_(.12)
        new.mlp[4].bias[12:].add_(.025)
    new(obs, stochastic_output=True)
    expected_kl = torch.distributions.kl_divergence(expected,
        torch.distributions.Normal(new.output_mean, new.output_std)).sum(-1)
    assert torch.equal(new.get_kl_divergence(old_params, new.output_distribution_params), expected_kl)
    assert bool((expected_kl > 0).all())


def test_shuffled_stored_history_likelihood_and_padded_minibatch_path():
    _, new, obs = pair()
    actions = new(obs, stochastic_output=True).detach()
    mean, std = (value.detach().clone() for value in new.output_distribution_params)
    lp = new.get_output_log_prob(actions).detach().clone()
    order = torch.tensor([6, 2, 0, 5, 3, 1, 4])
    new(obs[order], stochastic_output=True)
    torch.testing.assert_close(new.output_mean, mean[order], rtol=1e-6, atol=2e-7)
    torch.testing.assert_close(new.output_std, std[order], rtol=1e-6, atol=2e-7)
    torch.testing.assert_close(new.get_output_log_prob(actions[order]), lp[order], rtol=1e-5, atol=1e-5)
    padded = TensorDict({"policy": torch.randn(4, 3, 372)}, batch_size=[4, 3])
    masks = torch.tensor([[True, True, True], [True, True, True], [True, True, False], [False, False, False]])
    unpadded = unpad_trajectories(padded, masks)
    rng = torch.get_rng_state()
    expected = new(unpadded, stochastic_output=True)
    torch.set_rng_state(rng)
    assert torch.equal(new(padded, masks=masks, stochastic_output=True), expected)


def test_learned_log_std_keeps_finite_nonzero_gradient_without_state_or_action_clamp():
    _, new, obs = pair()
    new(obs, stochastic_output=True)
    actions = new.output_mean.detach() + torch.linspace(.01, .3, len(obs))[:, None]
    loss = -new.get_output_log_prob(actions).mean()
    loss.backward()
    gradient = new.mlp[4].weight.grad[12:]
    assert bool(torch.isfinite(gradient).all()) and int(torch.count_nonzero(gradient)) > 0
    assert new.exploration_std_temperature == .5
    with pytest.raises(AttributeError):
        new.exploration_std_temperature = .25


@pytest.mark.parametrize("value", [None, True, False, .25, 1., 0., -.5, "0.5", float("nan"), float("inf")])
def test_actor_requires_explicit_exact_temperature(value):
    obs = observations()
    cfg = copy.deepcopy(metadata()["runner_config"]["actor"])
    cfg.pop("class_name")
    cfg["exploration_std_temperature"] = value
    with pytest.raises(ValueError, match="temperature 0.5"):
        SemanticTemperedHistoryMLPModel(obs, {"actor": ["policy"]}, "actor", 12, **cfg)


def test_actor_rejects_missing_temperature_and_non_role_layout():
    obs = observations()
    cfg = copy.deepcopy(metadata()["runner_config"]["actor"])
    cfg.pop("class_name")
    cfg.pop("exploration_std_temperature")
    with pytest.raises(ValueError):
        SemanticTemperedHistoryMLPModel(obs, {"actor": ["policy"]}, "actor", 12, **cfg)
    cfg.update(exploration_std_temperature=.5, observation_layout=None)
    with pytest.raises(ValueError):
        SemanticTemperedHistoryMLPModel(obs, {"actor": ["policy"]}, "actor", 12, **cfg)


def test_effective_underflow_is_rejected_not_clipped(monkeypatch):
    _, new, obs = pair()
    head = torch.zeros(len(obs), 2, 12)
    head[:, 1] = -103.5  # Learned float32 sigma >0, halving falls below representability.
    assert bool((head[:, 1].exp() > 0).all())
    monkeypatch.setattr(new.mlp, "forward", lambda latent: head)
    with pytest.raises(ValueError, match="effective conditional sigma"):
        new(obs, stochastic_output=True)


def test_new_metadata_factory_and_contract_only_change_explicit_temperature_fields():
    old, new = metadata(policy.HISTORY_POLICY), metadata()
    assert policy.policy_version_from_metadata(new) == policy.HISTORY_TEMPERED_POLICY
    assert policy.supported_heteroscedastic_contract_version(json.loads(json.dumps(new["policy_contract"]))) == policy.HISTORY_TEMPERED_POLICY
    assert policy.policy_observation_layout_from_metadata(new) == ROLE_OBSERVATION_LAYOUT
    expected_config = copy.deepcopy(old["runner_config"])
    expected_config["actor"].update(class_name=policy.HISTORY_TEMPERED_ACTOR_CLASS,
                                    exploration_std_temperature=.5)
    assert new["runner_config"] == expected_config
    expected_contract = copy.deepcopy(old["policy_contract"])
    expected_contract.update(version=policy.HISTORY_TEMPERED_POLICY,
        actor_class=policy.HISTORY_TEMPERED_ACTOR_CLASS, exploration_std_temperature=.5,
        conditional_std="0.5*learned_sigma_as_innovation; no_stationary_rescaling",
        effective_log_std="learned_log_std+log(0.5)",
        temperature_scope="all_stochastic_sample_logprob_entropy_KL_calls; deterministic_mean_unchanged")
    assert new["policy_contract"] == expected_contract
    assert policy.policy_version_from_metadata(old) == policy.HISTORY_POLICY


@pytest.mark.parametrize("change", ["missing_tau", "wrong_tau", "bool_tau", "extra_tau", "contract_tau",
    "missing_contract_tau", "old_class", "old_contract", "wrong_mean", "wrong_rho", "wrong_std",
    "wrong_layout", "extra_config", "v2", "wrong_version"])
def test_metadata_rejects_mixed_unbound_temperature_or_semantics(change):
    info = metadata()
    cfg, contract = info["runner_config"]["actor"], info["policy_contract"]
    if change == "missing_tau": cfg.pop("exploration_std_temperature")
    elif change == "wrong_tau": cfg["exploration_std_temperature"] = .25
    elif change == "bool_tau": cfg["exploration_std_temperature"] = True
    elif change == "extra_tau": cfg["distribution_cfg"]["exploration_std_temperature"] = .5
    elif change == "contract_tau": contract["exploration_std_temperature"] = .25
    elif change == "missing_contract_tau": contract.pop("exploration_std_temperature")
    elif change == "old_class": cfg["class_name"] = policy.HISTORY_ACTOR_CLASS
    elif change == "old_contract": info["policy_contract"] = policy.policy_contract(policy.HISTORY_POLICY, observation_layout=ROLE_OBSERVATION_LAYOUT)
    elif change == "wrong_mean": contract["conditional_mean"] = "base_mean"
    elif change == "wrong_rho": contract["rho"] = .5
    elif change == "wrong_std": contract["effective_log_std"] = "learned_log_std"
    elif change == "wrong_layout": contract["observation_layout"] = None
    elif change == "extra_config": info["runner_config"]["algorithm"]["temperature"] = .5
    elif change == "v2": info["semantic_version"] = "v2"
    else: info["policy_version"] = policy.HISTORY_POLICY
    with pytest.raises(ValueError):
        policy.policy_version_from_metadata(info)


def test_no_new_contract_without_role_layout_or_hidden_temperature_in_old_config():
    with pytest.raises(ValueError, match="372"):
        policy.policy_contract(policy.HISTORY_TEMPERED_POLICY)
    for version in (policy.LEGACY_POLICY, policy.STATE_DEPENDENT_POLICY, policy.HISTORY_POLICY):
        cfg = semantic_runner_config(seed=1001, device="cpu", semantic_version="v3", policy_version=version)
        cfg["actor"]["exploration_std_temperature"] = .5
        with pytest.raises(ValueError, match="temperature"):
            policy.configure_policy_distribution(cfg, version)


@pytest.mark.parametrize("kind", ["jit", "onnx"])
def test_new_actor_keeps_explicit_unsupported_export_rejection(kind):
    _, new, _ = pair()
    with pytest.raises(NotImplementedError, match="export"):
        getattr(new, f"as_{kind}")()
