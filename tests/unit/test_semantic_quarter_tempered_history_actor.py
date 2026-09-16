"""CPU-only .5 -> .25 conditional-innovation proofs, not physical success."""
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
    SemanticTemperedHistoryMLPModel, SemanticQuarterTemperedHistoryMLPModel,
    history_conditioned_head,
)
from wlr50_clean.ppo.semantic_training import semantic_runner_config
from wlr50_clean.ppo.semantic_transfer_roles import ROLE_OBSERVATION_LAYOUT

QUARTER = policy.HISTORY_QUARTER_TEMPERED_POLICY


@pytest.fixture(autouse=True)
def cpu_rng_thread_scope():
    threads, rng = torch.get_num_threads(), torch.get_rng_state()
    torch.set_num_threads(1)
    torch.manual_seed(92361)
    yield
    torch.set_rng_state(rng)
    torch.set_num_threads(threads)


def metadata(version=QUARTER):
    return {"semantic_version": "v3", "seed": 1001, "policy_version": version,
            "runner_config": semantic_runner_config(seed=1001, device="cpu", semantic_version="v3",
                policy_version=version, observation_layout=ROLE_OBSERVATION_LAYOUT),
            "policy_contract": policy.policy_contract(version, observation_layout=ROLE_OBSERVATION_LAYOUT)}


def pair():
    values = torch.randn(7, 372)
    values[:, 195:207] = torch.linspace(-2., 2., 84).reshape(7, 12)
    obs = TensorDict({"policy": values}, batch_size=[7])
    models = []
    for version in (policy.HISTORY_TEMPERED_POLICY, QUARTER):
        config = copy.deepcopy(metadata(version)["runner_config"]["actor"])
        cls = resolve_callable(config.pop("class_name"))
        torch.manual_seed(93511)
        models.append(cls(obs, {"actor": ["policy"]}, "actor", 12, **config))
    return *models, obs


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


def test_strict_state_adam_normalizer_and_rng_reload_preserves_mean():
    old, new, obs = pair()
    assert type(old) is SemanticTemperedHistoryMLPModel
    assert type(new) is SemanticQuarterTemperedHistoryMLPModel
    assert list(dict(old.named_parameters())) == list(dict(new.named_parameters()))
    assert list(dict(old.named_buffers())) == list(dict(new.named_buffers())) == []
    assert set(vars(old)) == set(vars(new))
    exact_tree(old.state_dict(), new.state_dict())
    opt_old = torch.optim.Adam(old.parameters(), lr=1e-5)
    sum(parameter.square().mean() for parameter in old.parameters()).backward()
    opt_old.step()  # Synthetic populated Adam fixture, not a real PPO update.
    opt_old.zero_grad(set_to_none=True)
    stream = io.BytesIO()
    torch.save({"actor": old.state_dict(), "optimizer": opt_old.state_dict(),
                "normalizer": old.obs_normalizer.state_dict(), "rng": torch.get_rng_state()}, stream)
    stream.seek(0)
    restored = torch.load(stream, weights_only=True)
    new.load_state_dict(restored["actor"], strict=True)
    opt_new = torch.optim.Adam(new.parameters(), lr=3e-5)
    opt_new.load_state_dict(restored["optimizer"])
    exact_tree(old.state_dict(), new.state_dict())
    exact_tree(opt_old.state_dict(), opt_new.state_dict())
    exact_tree(restored["normalizer"], new.obs_normalizer.state_dict())
    assert opt_new.param_groups[0]["lr"] == 1e-5
    torch.set_rng_state(restored["rng"])
    assert torch.equal(new(obs), old(obs))
    assert torch.equal(torch.get_rng_state(), restored["rng"])


def test_deterministic_mean_history_cache_inputs_rng_are_exactly_unchanged():
    old, new, obs = pair()
    new(obs, stochastic_output=True)
    cache, state, original = new.distribution._distribution, copy.deepcopy(new.state_dict()), obs.clone()
    rng = torch.get_rng_state()
    for values in (obs, obs.flip(0)):
        assert torch.equal(new(values), old(values))
    changed = obs.clone()
    changed["policy"][:, 195:207].neg_()
    assert torch.equal(new(changed), old(changed))
    assert not torch.equal(new(changed), new(obs))
    new.reset(torch.ones(len(obs), dtype=torch.bool))
    assert torch.equal(new(obs), old(obs))
    assert new.distribution._distribution is cache
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.equal(obs["policy"], original["policy"])
    exact_tree(state, new.state_dict())


def test_common_gaussian_path_has_half_old_sigma_and_correct_sample_logprob_entropy_kl():
    old, new, obs = pair()
    old(obs, stochastic_output=True)
    latent = new.get_latent(obs)
    head = history_conditioned_head(new.mlp(latent), latent[:, 195:207])
    mean, sigma = head[:, 0], (head[:, 1] + math.log(.25)).exp()
    expected = torch.distributions.Normal(mean, sigma)
    torch.testing.assert_close(sigma, old.output_std * .5, rtol=1e-6, atol=0.)
    rng = torch.get_rng_state()
    actions = new(obs, stochastic_output=True)
    after = torch.get_rng_state()
    torch.set_rng_state(rng)
    assert torch.equal(actions, expected.sample())
    assert torch.equal(after, torch.get_rng_state())
    assert torch.equal(new.output_mean, old.output_mean)
    exact_tree(new.output_distribution_params, (mean, sigma))
    assert torch.equal(new.get_output_log_prob(actions), expected.log_prob(actions).sum(-1))
    assert torch.equal(new.output_entropy, expected.entropy().sum(-1))
    torch.testing.assert_close(new.output_entropy - old.output_entropy,
                               torch.full((len(obs),), 12 * math.log(.5)), rtol=1e-6, atol=2e-6)
    params = tuple(value.detach().clone() for value in new.output_distribution_params)
    log_prob = new.get_output_log_prob(actions).detach().clone()
    new(obs, stochastic_output=True)
    assert torch.equal((new.get_output_log_prob(actions) - log_prob).exp(), torch.ones(len(obs)))
    assert torch.equal(new.get_kl_divergence(params, new.output_distribution_params), torch.zeros(len(obs)))
    with torch.no_grad():
        new.mlp[4].bias[:12].add_(.12)
        new.mlp[4].bias[12:].add_(.025)
    new(obs, stochastic_output=True)
    kl = torch.distributions.kl_divergence(expected,
        torch.distributions.Normal(new.output_mean, new.output_std)).sum(-1)
    assert torch.equal(new.get_kl_divergence(params, new.output_distribution_params), kl)
    assert bool((kl > 0).all())


def test_stored_history_shuffle_padded_minibatch_and_learned_sigma_gradient():
    _, new, obs = pair()
    actions = new(obs, stochastic_output=True).detach()
    mean, std = (value.detach().clone() for value in new.output_distribution_params)
    log_prob = new.get_output_log_prob(actions).detach().clone()
    order = torch.tensor([6, 2, 0, 5, 3, 1, 4])
    new(obs[order], stochastic_output=True)
    torch.testing.assert_close(new.output_mean, mean[order], rtol=1e-6, atol=2e-7)
    torch.testing.assert_close(new.output_std, std[order], rtol=1e-6, atol=2e-7)
    torch.testing.assert_close(new.get_output_log_prob(actions[order]), log_prob[order], rtol=1e-5, atol=1e-5)
    padded = TensorDict({"policy": torch.randn(4, 3, 372)}, batch_size=[4, 3])
    masks = torch.tensor([[True, True, True], [True, True, True], [True, True, False], [False, False, False]])
    unpadded = unpad_trajectories(padded, masks)
    rng = torch.get_rng_state()
    expected = new(unpadded, stochastic_output=True)
    torch.set_rng_state(rng)
    assert torch.equal(new(padded, masks=masks, stochastic_output=True), expected)
    actions = new.output_mean.detach() + .1
    (-new.get_output_log_prob(actions).mean()).backward()
    gradient = new.mlp[4].weight.grad[12:]
    assert bool(torch.isfinite(gradient).all()) and int(torch.count_nonzero(gradient)) > 0
    assert new.exploration_std_temperature == .25
    with pytest.raises(AttributeError):
        new.exploration_std_temperature = .5


@pytest.mark.parametrize("value", [None, True, False, .5, 1., 0., -.25, "0.25", float("nan"), float("inf")])
def test_quarter_class_rejects_other_temperature_or_missing_role_layout(value):
    _, _, obs = pair()
    config = copy.deepcopy(metadata()["runner_config"]["actor"])
    config.pop("class_name")
    config["exploration_std_temperature"] = value
    with pytest.raises(ValueError, match="temperature 0.25"):
        SemanticQuarterTemperedHistoryMLPModel(obs, {"actor": ["policy"]}, "actor", 12, **config)
    config.update(exploration_std_temperature=.25, observation_layout=None)
    with pytest.raises(ValueError, match="372"):
        SemanticQuarterTemperedHistoryMLPModel(obs, {"actor": ["policy"]}, "actor", 12, **config)


def test_effective_underflow_rejected_without_clipping_and_no_export(monkeypatch):
    _, new, obs = pair()
    head = torch.zeros(len(obs), 2, 12)
    head[:, 1] = -103.5
    assert bool((head[:, 1].exp() > 0).all())
    monkeypatch.setattr(new.mlp, "forward", lambda latent: head)
    with pytest.raises(ValueError, match="effective conditional sigma"):
        new(obs, stochastic_output=True)
    for method in (new.as_jit, new.as_onnx):
        with pytest.raises(NotImplementedError, match="export"):
            method()


def test_only_explicit_temperature_metadata_fields_change_and_old_version_still_loads():
    old, new = metadata(policy.HISTORY_TEMPERED_POLICY), metadata()
    expected_config = copy.deepcopy(old["runner_config"])
    expected_config["actor"].update(class_name=policy.HISTORY_QUARTER_TEMPERED_ACTOR_CLASS,
                                     exploration_std_temperature=.25)
    assert new["runner_config"] == expected_config
    expected_contract = copy.deepcopy(old["policy_contract"])
    expected_contract.update(version=QUARTER, actor_class=policy.HISTORY_QUARTER_TEMPERED_ACTOR_CLASS,
        exploration_std_temperature=.25,
        conditional_std="0.25*learned_sigma_as_innovation; no_stationary_rescaling",
        effective_log_std="learned_log_std+log(0.25)")
    assert new["policy_contract"] == expected_contract
    assert policy.policy_version_from_metadata(old) == policy.HISTORY_TEMPERED_POLICY
    assert policy.policy_version_from_metadata(new) == QUARTER
    assert policy.supported_heteroscedastic_contract_version(json.loads(json.dumps(new["policy_contract"]))) == QUARTER
    assert policy.policy_observation_layout_from_metadata(new) == ROLE_OBSERVATION_LAYOUT
    with pytest.raises(ValueError, match="372"):
        policy.policy_contract(QUARTER)


@pytest.mark.parametrize("change", ["tau", "missing_tau", "class", "contract", "contract_tau", "rho",
                                     "mean", "std", "version", "semantic_v2", "layout", "extra_config"])
def test_mixed_half_quarter_or_unbound_semantics_are_rejected(change):
    info = metadata()
    actor, contract = info["runner_config"]["actor"], info["policy_contract"]
    if change == "tau": actor["exploration_std_temperature"] = .5
    elif change == "missing_tau": actor.pop("exploration_std_temperature")
    elif change == "class": actor["class_name"] = policy.HISTORY_TEMPERED_ACTOR_CLASS
    elif change == "contract": info["policy_contract"] = metadata(policy.HISTORY_TEMPERED_POLICY)["policy_contract"]
    elif change == "contract_tau": contract["exploration_std_temperature"] = .5
    elif change == "rho": contract["rho"] = .5
    elif change == "mean": contract["conditional_mean"] = "base_mean"
    elif change == "std": contract["effective_log_std"] = "learned_log_std"
    elif change == "version": info["policy_version"] = policy.HISTORY_TEMPERED_POLICY
    elif change == "semantic_v2": info["semantic_version"] = "v2"
    elif change == "layout": contract["observation_layout"] = None
    else: info["runner_config"]["algorithm"]["temperature"] = .25
    with pytest.raises(ValueError):
        policy.policy_version_from_metadata(info)
