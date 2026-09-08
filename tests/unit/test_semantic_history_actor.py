"""CPU-only proofs of the conditional kernel, not a physical-control experiment."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rsl_rl")
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from rsl_rl.modules.distribution import HeteroscedasticGaussianDistribution
from rsl_rl.utils import resolve_callable, unpad_trajectories

from wlr50_clean.ppo import semantic_policy_distribution as policy
from wlr50_clean.ppo.semantic_history_actor import (
    HISTORY_RHO, SemanticHistoryMLPModel, history_conditioned_head,
)
from wlr50_clean.ppo.semantic_observation import load_semantic_observation_schema
from wlr50_clean.ppo.semantic_training import semantic_runner_config


@pytest.fixture(autouse=True)
def cpu_rng_and_thread_scope():
    previous_threads = torch.get_num_threads()
    previous_rng = torch.get_rng_state()
    torch.set_num_threads(1)
    torch.manual_seed(61203)
    yield
    torch.set_rng_state(previous_rng)
    torch.set_num_threads(previous_threads)


def observations(n=7):
    values = torch.randn(n, 324, device="cpu")
    values[:, 195:207] = torch.linspace(-2., 2., n * 12).reshape(n, 12)
    return TensorDict({"policy": values}, batch_size=[n])


def actor_pair(obs=None):
    obs = observations() if obs is None else obs
    cfg = semantic_runner_config(seed=1001, device="cpu", semantic_version="v3",
                                 policy_version=policy.STATE_DEPENDENT_POLICY)["actor"]
    cfg = copy.deepcopy(cfg)
    cfg.pop("class_name")
    torch.manual_seed(71204)
    old = MLPModel(obs, {"actor": ["policy"]}, "actor", 12, **copy.deepcopy(cfg))
    torch.manual_seed(71204)
    new = SemanticHistoryMLPModel(obs, {"actor": ["policy"]}, "actor", 12, **copy.deepcopy(cfg))
    return old, new, obs


def metadata(version):
    result = {"semantic_version": "v3", "seed": 1001,
              "runner_config": semantic_runner_config(seed=1001, device="cpu", semantic_version="v3",
                                                       policy_version=version),
              "policy_contract": policy.policy_contract(version)}
    return result


def test_helper_mixes_only_mean_and_preserves_log_sigma_and_inputs():
    head = torch.randn(5, 2, 12, dtype=torch.float64)
    history = torch.linspace(-20., 20., 60, dtype=torch.float64).reshape(5, 12)
    before, history_before = head.clone(), history.clone()
    actual = history_conditioned_head(head, history)
    assert torch.equal(actual[:, 0], (1. - HISTORY_RHO) * head[:, 0] + HISTORY_RHO * history)
    assert torch.equal(actual[:, 1], head[:, 1])
    assert torch.equal(actual[:, 1].exp(), head[:, 1].exp())
    assert torch.equal(head, before) and torch.equal(history, history_before)
    assert actual.data_ptr() != head.data_ptr()


@pytest.mark.parametrize("rho", [True, False, None, "0.9", -.1, 1., 1.1, float("nan"),
                                  float("inf"), -float("inf"), 10 ** 400])
def test_helper_rejects_invalid_rho(rho):
    with pytest.raises(ValueError, match="rho"):
        history_conditioned_head(torch.zeros(2, 2, 12), torch.zeros(2, 12), rho)


@pytest.mark.parametrize("change", ["head_list", "history_list", "head_int", "history_bool",
                                     "head_width", "head_rank", "history_shape", "dtype",
                                     "head_nan", "head_inf", "history_nan", "history_inf",
                                     "history_above", "history_below"])
def test_helper_rejects_invalid_tensor_contract(change):
    head, history = torch.zeros(2, 2, 12), torch.zeros(2, 12)
    if change == "head_list": head = head.tolist()
    elif change == "history_list": history = history.tolist()
    elif change == "head_int": head = head.to(torch.int64)
    elif change == "history_bool": history = history.to(torch.bool)
    elif change == "head_width": head = torch.zeros(2, 2, 11)
    elif change == "head_rank": head = torch.zeros(24)
    elif change == "history_shape": history = torch.zeros(1, 12)
    elif change == "dtype": history = history.double()
    elif change == "head_nan": head[0, 0, 0] = float("nan")
    elif change == "head_inf": head[0, 1, 0] = float("inf")
    elif change == "history_nan": history[0, 0] = float("nan")
    elif change == "history_inf": history[0, 0] = float("inf")
    elif change == "history_above": history[0, 0] = 20.01
    else: history[0, 0] = -20.01
    with pytest.raises(ValueError):
        history_conditioned_head(head, history)


def test_rho_zero_is_exact_head_and_official_sampling_rng_degeneracy():
    head = torch.randn(3, 2, 12)
    head[0, 0, 0] = -0.
    history = torch.full((3, 12), 20.)
    before = torch.get_rng_state()
    zero = history_conditioned_head(head, history, rho=0.)
    assert zero is head and torch.signbit(zero[0, 0, 0])
    assert torch.equal(before, torch.get_rng_state())
    old = HeteroscedasticGaussianDistribution(12, init_std=.15, std_type="log")
    new = HeteroscedasticGaussianDistribution(12, init_std=.15, std_type="log")
    old.update(head)
    new.update(zero)
    torch.set_rng_state(before)
    expected = old.sample()
    after = torch.get_rng_state()
    torch.set_rng_state(before)
    actual = new.sample()
    assert torch.equal(actual, expected) and torch.equal(after, torch.get_rng_state())
    assert torch.equal(new.log_prob(actual), old.log_prob(actual))
    assert torch.equal(new.entropy, old.entropy)


@pytest.mark.parametrize("log_sigma", [100., 1000., -104., -1000.])
@pytest.mark.parametrize("rho", [0., .9])
def test_unrepresentable_conditional_sigma_rejected_without_floor_or_clamp(log_sigma, rho):
    head = torch.zeros(2, 2, 12)
    head[:, 1] = log_sigma
    before = head.clone()
    with pytest.raises(ValueError, match="sigma must be finite and strictly positive"):
        history_conditioned_head(head, torch.zeros(2, 12), rho)
    assert torch.equal(head, before)


@pytest.mark.parametrize("log_sigma", [-90., 80.])
def test_representable_extreme_sigma_is_not_rescaled_or_clamped(log_sigma):
    head = torch.zeros(2, 2, 12)
    head[:, 1] = log_sigma
    actual = history_conditioned_head(head, torch.zeros(2, 12))
    assert torch.equal(actual[:, 1], head[:, 1])
    assert torch.equal(actual[:, 1].exp(), head[:, 1].exp())


def test_actual_actor_state_keys_initialization_weights_buffers_and_strict_load_are_identical():
    old, new, _ = actor_pair()
    assert list(old.state_dict()) == list(new.state_dict())
    assert list(dict(old.named_parameters())) == list(dict(new.named_parameters()))
    assert list(dict(old.named_buffers())) == list(dict(new.named_buffers())) == []
    assert all(torch.equal(v, new.state_dict()[k]) for k, v in old.state_dict().items())
    assert all(v.data_ptr() != new.state_dict()[k].data_ptr() for k, v in old.state_dict().items())
    assert not new.is_recurrent and new.get_hidden_state() is None
    new.load_state_dict(old.state_dict(), strict=True)
    assert all(torch.equal(v, new.state_dict()[k]) for k, v in old.state_dict().items())
    assert set(vars(new)) == set(vars(old))  # No private action history or extra model state.


def test_real_rsl_sample_logprob_entropy_kl_use_actual_conditional_normal():
    old, new, obs = actor_pair()
    base = old.mlp(old.get_latent(obs))
    mean = (1. - HISTORY_RHO) * base[:, 0] + HISTORY_RHO * obs["policy"][:, 195:207]
    sigma = base[:, 1].exp()
    expected = torch.distributions.Normal(mean, sigma)
    before = torch.get_rng_state()
    actions = new(obs, stochastic_output=True)
    after = torch.get_rng_state()
    torch.set_rng_state(before)
    assert torch.equal(actions, expected.sample())
    assert torch.equal(after, torch.get_rng_state())
    assert torch.equal(new.output_mean, mean) and torch.equal(new.output_std, sigma)
    assert torch.equal(new.get_output_log_prob(actions), expected.log_prob(actions).sum(-1))
    assert torch.equal(new.output_entropy, expected.entropy().sum(-1))
    old_params = (mean.detach().clone(), sigma.detach().clone())
    with torch.no_grad():
        new.mlp[4].bias[:12].add_(.12)
        new.mlp[4].bias[12:].add_(.025)
    new(obs, stochastic_output=True)
    actual_kl = new.get_kl_divergence(old_params, new.output_distribution_params)
    expected_kl = torch.distributions.kl_divergence(expected, torch.distributions.Normal(
        new.output_mean, new.output_std)).sum(-1)
    assert torch.equal(actual_kl, expected_kl)
    assert torch.isfinite(actual_kl).all() and bool((actual_kl > 0).all())


def test_positive_log_density_is_valid_and_base_mean_is_not_the_likelihood_center():
    _, new, obs = actor_pair()
    new(obs, stochastic_output=True)
    assert bool((new.get_output_log_prob(new.output_mean) > 0).all())
    base_mean = new.mlp(new.get_latent(obs))[:, 0]
    assert not torch.equal(base_mean, new.output_mean)
    assert bool((new.get_output_log_prob(base_mean) < new.get_output_log_prob(new.output_mean)).all())


def test_shuffled_minibatches_rebuild_each_stored_history_not_other_row_or_cache():
    _, new, obs = actor_pair()
    actions = new(obs, stochastic_output=True).detach()
    mean, std = [v.detach().clone() for v in new.output_distribution_params]
    lp, entropy = new.get_output_log_prob(actions).detach(), new.output_entropy.detach()
    permutation = torch.tensor([6, 2, 0, 5, 3, 1, 4])
    new(obs[permutation], stochastic_output=True)
    # Permuting GEMM rows can change reduction scheduling; compare at float32 accuracy.
    torch.testing.assert_close(new.output_mean, mean[permutation], rtol=1e-6, atol=2e-7)
    torch.testing.assert_close(new.output_std, std[permutation], rtol=1e-6, atol=2e-7)
    torch.testing.assert_close(new.get_output_log_prob(actions[permutation]), lp[permutation], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(new.output_entropy, entropy[permutation], rtol=1e-6, atol=2e-7)


def test_deterministic_forward_is_conditional_mean_without_cache_rng_or_mode_mutation():
    _, new, obs = actor_pair()
    new.train()
    new(obs, stochastic_output=True)
    cache = new.distribution._distribution
    before = torch.get_rng_state()
    model_before = {k: v.clone() for k, v in new.state_dict().items()}
    different = obs.clone()
    different["policy"][:, 195:207].neg_()
    head = new.mlp(new.get_latent(different))
    expected = history_conditioned_head(head, different["policy"][:, 195:207])[:, 0]
    assert torch.equal(new(different, stochastic_output=False), expected)
    assert new.distribution._distribution is cache and new.training
    assert torch.equal(before, torch.get_rng_state())
    assert all(torch.equal(v, new.state_dict()[k]) for k, v in model_before.items())


def test_history_is_stateless_reset_zero_and_phase_change_does_not_clear_input_history():
    _, new, obs = actor_pair()
    initial = new(obs).detach().clone()
    unrelated = observations()
    unrelated["policy"][:, 195:207] = 19.
    new(unrelated)
    new.reset(torch.ones(len(obs), dtype=torch.bool))
    assert torch.equal(new(obs), initial)
    reset_obs = obs.clone()
    reset_obs["policy"][:, 195:207] = 0.
    base = new.mlp(new.get_latent(reset_obs))[:, 0]
    assert torch.equal(new(reset_obs), (1. - HISTORY_RHO) * base)
    changed_phase = obs.clone()
    changed_phase["policy"][:, :13] = 0.
    changed_phase["policy"][:, 8] = 1.
    changed_head = new.mlp(new.get_latent(changed_phase))
    expected = history_conditioned_head(changed_head, obs["policy"][:, 195:207])[:, 0]
    assert torch.equal(new(changed_phase), expected)


def test_real_schema_provides_existing_clipped_raw_slice_not_residual_history(tmp_path):
    root = Path(__file__).resolve().parents[2]
    # Keep the original 324 ABI assertions explicit after the current opt-in
    # schema appends role state; no original feature or scale is changed here.
    data = json.loads((root / "configs/ppo_semantic_v3/observation_schema.json").read_text())
    data.pop("transfer_role_features_version", None)
    data["feature_groups"] = [row for row in data["feature_groups"] if row["name"] != "transfer_role_context_full48"]
    legacy_path = tmp_path / "legacy_observation_schema.json"
    legacy_path.write_text(json.dumps(data), encoding="utf-8")
    schema = load_semantic_observation_schema(legacy_path)
    offset = 0
    located = []
    for row in schema.groups:
        if row["name"] == "previous_raw_full12":
            located.append((offset, row["size"], row["scale"]))
        offset += row["size"]
    assert located == [(195, 12, 1.0)]
    assert offset == 324 and schema.clip == 20.0
    groups = {row["name"]: [0.] * row["size"] for row in schema.groups}
    groups["previous_raw_full12"] = [-100., 100., -20., 20.] + list(range(8))
    groups["previous_residual_full12"] = [1.] * 12
    encoded = schema.encode(groups)
    assert schema.dimension == 324 and schema.clip == 20.
    assert encoded[195:207] == (-20., 20., -20., 20., *range(8))
    assert encoded[207:219] != encoded[195:207]
    _, new, _ = actor_pair()
    obs = TensorDict({"policy": torch.tensor([encoded])}, batch_size=[1])
    actual = new(obs)
    base = new.mlp(new.get_latent(obs))
    assert torch.equal(actual, history_conditioned_head(base, obs["policy"][:, 195:207])[:, 0])


def test_masks_follow_official_unpad_before_history_selection():
    _, new, _ = actor_pair()
    values = torch.randn(4, 3, 324)
    padded = TensorDict({"policy": values}, batch_size=[4, 3])
    masks = torch.tensor([[True, True, True], [True, True, True], [True, True, False], [False, False, False]])
    unpadded = unpad_trajectories(padded, masks)
    expected = new(unpadded)
    actual = new(padded, masks=masks)
    assert torch.equal(actual, expected) and tuple(actual.shape) == (4, 2, 12)


def test_mean_gradient_scaled_but_learned_log_sigma_gradient_not_rescaled():
    head = torch.randn(2, 2, 12, requires_grad=True)
    history = torch.ones(2, 12)
    mixed = history_conditioned_head(head, history)
    mixed.sum().backward()
    assert torch.equal(head.grad[:, 0], torch.full((2, 12), 1. - HISTORY_RHO))
    assert torch.equal(head.grad[:, 1], torch.ones(2, 12))


@pytest.mark.parametrize("kind", ["jit", "onnx"])
def test_unsupported_export_is_explicitly_rejected(kind):
    _, new, _ = actor_pair()
    with pytest.raises(NotImplementedError, match="export"):
        getattr(new, f"as_{kind}")()


@pytest.mark.parametrize("version", [policy.LEGACY_POLICY, policy.STATE_DEPENDENT_POLICY, policy.HISTORY_POLICY])
def test_real_factory_configuration_and_strict_metadata(version):
    info = metadata(version)
    assert policy.policy_version_from_metadata(info) == version
    config = copy.deepcopy(info["runner_config"]["actor"])
    cls = resolve_callable(config.pop("class_name"))
    instance = cls(observations(), {"actor": ["policy"]}, "actor", 12, **config)
    assert type(instance) is (SemanticHistoryMLPModel if version == policy.HISTORY_POLICY else MLPModel)
    if version == policy.HISTORY_POLICY:
        assert info["runner_config"]["actor"]["class_name"] == policy.HISTORY_ACTOR_CLASS
        assert info["policy_contract"]["rho"] == .9


@pytest.mark.parametrize("change", ["missing_contract", "null_contract", "old_contract", "old_class",
                                    "unknown_class", "wrong_distribution", "rho", "rho_bool",
                                    "history_slice", "history_clip", "std_scaling", "extra_contract",
                                    "extra_config", "missing_config", "version", "v2"])
def test_history_metadata_rejects_missing_mixed_or_unrecognized_version(change):
    info = metadata(policy.HISTORY_POLICY)
    if change == "missing_contract": info.pop("policy_contract")
    elif change == "null_contract": info["policy_contract"] = None
    elif change == "old_contract": info["policy_contract"] = policy.policy_contract(policy.STATE_DEPENDENT_POLICY)
    elif change == "old_class": info["runner_config"]["actor"]["class_name"] = "MLPModel"
    elif change == "unknown_class": info["runner_config"]["actor"]["class_name"] = "OtherHistoryActor"
    elif change == "wrong_distribution": info["runner_config"]["actor"]["distribution_cfg"]["std_type"] = "scalar"
    elif change == "rho": info["policy_contract"]["rho"] = .8
    elif change == "rho_bool": info["policy_contract"]["rho"] = True
    elif change == "history_slice": info["policy_contract"]["history_slice"] = [207, 219]
    elif change == "history_clip": info["policy_contract"]["history_clip"] = 21.
    elif change == "std_scaling": info["policy_contract"]["conditional_std"] = "sqrt(1-rho**2)*sigma"
    elif change == "extra_contract": info["policy_contract"]["unreviewed"] = 1
    elif change == "extra_config": info["runner_config"]["actor"]["rho"] = .9
    elif change == "missing_config": info["runner_config"].pop("save_interval")
    elif change == "version": info["policy_version"] = policy.STATE_DEPENDENT_POLICY
    else: info["semantic_version"] = "v2"
    with pytest.raises(ValueError):
        policy.policy_version_from_metadata(info)


@pytest.mark.parametrize("version", [policy.STATE_DEPENDENT_POLICY, policy.HISTORY_POLICY])
def test_public_prefix_contract_validation_is_exact_and_json_roundtrip_safe(version):
    contract = policy.policy_contract(version)
    assert policy.supported_heteroscedastic_contract_version(contract) == version
    assert policy.supported_heteroscedastic_contract_version(json.loads(json.dumps(contract))) == version
    contract["observation_dimension"] = 332
    with pytest.raises(ValueError): policy.supported_heteroscedastic_contract_version(contract)


@pytest.mark.parametrize("contract", [None, {}, [], {"version": policy.HISTORY_POLICY},
                                        policy.policy_contract(policy.LEGACY_POLICY)])
def test_public_prefix_validator_does_not_widen_to_partial_or_legacy_scalar(contract):
    with pytest.raises(ValueError): policy.supported_heteroscedastic_contract_version(contract)


def test_old_contracts_remain_identical_and_missing_contract_legacy_remains_supported():
    for version in (policy.LEGACY_POLICY, policy.STATE_DEPENDENT_POLICY):
        dependent = version == policy.STATE_DEPENDENT_POLICY
        expected = {"schema": policy.POLICY_SCHEMA, "version": version,
                    "distribution_class": "HeteroscedasticGaussianDistribution" if dependent else "GaussianDistribution",
                    "std_type": "log" if dependent else "scalar", "observation_dimension": 324,
                    "raw_action_dimension": 12, "actor_hidden_dims": [256, 256], "activation": "elu",
                    "state_dependent_std": dependent, "normalization": policy.NORMALIZATION,
                    "raw_action_semantics": "unbounded_Gaussian_latent_before_existing_tanh_projection"}
        assert policy.policy_contract(version) == expected
    info = metadata(policy.LEGACY_POLICY)
    info.pop("policy_contract")
    assert policy.policy_version_from_metadata(info) == policy.LEGACY_POLICY


@pytest.mark.parametrize("change", ["normalizer", "obs_groups", "width", "distribution", "std_type", "output"])
def test_actor_rejects_incompatible_history_or_distribution_interface(change):
    obs = observations()
    config = copy.deepcopy(metadata(policy.HISTORY_POLICY)["runner_config"]["actor"])
    config.pop("class_name")
    groups, output = {"actor": ["policy"]}, 12
    if change == "normalizer": config["obs_normalization"] = True
    elif change == "obs_groups": groups = {"actor": ["critic"]}
    elif change == "width": obs = TensorDict({"policy": torch.zeros(2, 332)}, batch_size=[2])
    elif change == "distribution": config["distribution_cfg"]["class_name"] = "GaussianDistribution"
    elif change == "std_type": config["distribution_cfg"]["std_type"] = "scalar"
    else: output = 8
    with pytest.raises(ValueError):
        SemanticHistoryMLPModel(obs, groups, "actor", output, **config)
