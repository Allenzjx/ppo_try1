"""CPU-only explicit appended-layout contracts, kernel and frozen-prefix checks."""
from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rsl_rl")
from tensordict import TensorDict

from wlr50_clean.ppo import semantic_policy_distribution as policy
from wlr50_clean.ppo.semantic_history_actor import SemanticHistoryMLPModel, HISTORY_RHO
from wlr50_clean.ppo.semantic_checkpoint_prefix_policy import build_frozen_checkpoint_prefix_policy
from wlr50_clean.ppo.semantic_checkpoint_prefix import (
    _core_observation_layout, CheckpointPolicyPrefixRequest, CheckpointPolicyPrefixRslAdapter,
)
from wlr50_clean.ppo.semantic_training import parameter_hash, semantic_runner_config
from wlr50_clean.ppo.semantic_observation import load_semantic_observation_schema
from wlr50_clean.ppo.semantic_transfer_roles import (
    LEGS, ROLE_OBSERVATION_LAYOUT as LAYOUT, ROLE_OBSERVATION_GROUP,
    ROLE_OBSERVATION_BASE_DIM, ROLE_OBSERVATION_DIM, ROLE_OBSERVATION_FIELDS,
)
from test_semantic_checkpoint_prefix import Core
from test_semantic_checkpoint_prefix_policy import rng_state, assert_rng_equal


@pytest.fixture(autouse=True)
def cpu_scope():
    threads, state = torch.get_num_threads(), torch.get_rng_state()
    torch.set_num_threads(1)
    torch.manual_seed(918372)
    yield
    torch.set_rng_state(state)
    torch.set_num_threads(threads)


def observations(dimension=372, n=4):
    values = torch.randn(n, dimension) * .2
    values[:, 195:207] = torch.linspace(-2., 2., n * 12).reshape(n, 12)
    if dimension == 372:
        values[:, 324:] = torch.rand(n, 48)
    return TensorDict({"policy": values, "critic": values.clone()}, batch_size=[n])


def config(layout=None):
    options = {} if layout is None else {"observation_layout": layout}
    return semantic_runner_config(seed=1001, device="cpu", semantic_version="v3",
                                  policy_version=policy.HISTORY_POLICY, **options)


def actor(layout=None, obs=None):
    options = copy.deepcopy(config(layout)["actor"])
    options.pop("class_name")
    obs = observations(324 if layout is None else 372) if obs is None else obs
    return SemanticHistoryMLPModel(obs, {"actor": ["policy"]}, "actor", 12, **options)


def metadata(layout=None):
    return {"semantic_version": "v3", "seed": 1001,
            "runner_config": config(layout),
            "policy_contract": policy.policy_contract(policy.HISTORY_POLICY, observation_layout=layout)}


def source(model, layout=None):
    return {"checkpoint_path": "verified/cpu_layout_source.pt", "checkpoint_sha256": "a" * 64,
            "actor_parameter_sha256": parameter_hash(model),
            "source_global_policy_decisions": 127232, "source_ppo_updates": 959,
            "policy_contract": policy.policy_contract(policy.HISTORY_POLICY, observation_layout=layout)}


def test_explicit_role_contract_keeps_legacy_bytes_and_exact_history_slice():
    old = policy.policy_contract(policy.HISTORY_POLICY)
    expected = copy.deepcopy(old)
    new = policy.policy_contract(policy.HISTORY_POLICY, observation_layout=LAYOUT)
    assert old == policy.policy_contract(policy.HISTORY_POLICY, observation_layout=None)
    assert old["observation_dimension"] == 324 and "observation_layout" not in old
    assert new["observation_dimension"] == ROLE_OBSERVATION_DIM == 372
    assert new["base_observation_dimension"] == ROLE_OBSERVATION_BASE_DIM == 324
    assert new["role_observation_slice"] == [324, 372]
    assert new["role_observation_group"] == ROLE_OBSERVATION_GROUP
    assert new["role_observation_leg_order"] == list(LEGS) == ["FL", "FR", "RL", "RR"]
    assert new["role_observation_fields"] == list(ROLE_OBSERVATION_FIELDS)
    assert len(ROLE_OBSERVATION_FIELDS) == 12
    for key in expected.keys() - {"observation_dimension"}:
        assert new[key] == expected[key]
    assert new["rho"] == .9 and new["history_slice"] == [195, 207] and new["history_clip"] == 20.
    assert "observation_layout" not in config()["actor"]
    assert config(LAYOUT)["actor"]["observation_layout"] == LAYOUT


@pytest.mark.parametrize("version", [policy.LEGACY_POLICY, policy.STATE_DEPENDENT_POLICY])
def test_role_layout_cannot_expand_old_gaussian_distribution_lanes(version):
    cfg = copy.deepcopy(config())
    before = copy.deepcopy(cfg)
    with pytest.raises(ValueError):
        policy.policy_contract(version, observation_layout=LAYOUT)
    with pytest.raises(ValueError):
        policy.configure_policy_distribution(cfg, version, observation_layout=LAYOUT)
    assert cfg == before


@pytest.mark.parametrize("layout", [False, True, 372, "372", "other", "", {}])
def test_unknown_or_implicit_layout_is_not_accepted(layout):
    with pytest.raises(ValueError):
        policy.policy_contract(policy.HISTORY_POLICY, observation_layout=layout)


@pytest.mark.parametrize("layout", [None, LAYOUT])
def test_complete_metadata_resolves_layout_without_changing_kernel(layout):
    info = metadata(layout)
    assert policy.policy_version_from_metadata(info) == policy.HISTORY_POLICY
    assert policy.policy_observation_layout_from_metadata(info) == layout
    assert policy.supported_heteroscedastic_contract_version(info["policy_contract"]) == policy.HISTORY_POLICY


@pytest.mark.parametrize("fault", ["missing_marker", "wrong_dim", "wrong_base", "fields", "leg_order",
                                   "slice", "rho", "history_slice", "extra", "runner_missing", "runner_wrong"])
def test_mixed_role_metadata_and_layout_contract_fail_closed(fault):
    info = metadata(LAYOUT)
    contract = info["policy_contract"]
    if fault == "missing_marker": contract.pop("observation_layout")
    elif fault == "wrong_dim": contract["observation_dimension"] = 324
    elif fault == "wrong_base": contract["base_observation_dimension"] = 323
    elif fault == "fields": contract["role_observation_fields"].reverse()
    elif fault == "leg_order": contract["role_observation_leg_order"].reverse()
    elif fault == "slice": contract["role_observation_slice"] = [323, 371]
    elif fault == "rho": contract["rho"] = .8
    elif fault == "history_slice": contract["history_slice"] = [243, 255]
    elif fault == "extra": contract["unsupported"] = True
    elif fault == "runner_missing": info["runner_config"]["actor"].pop("observation_layout")
    else: info["runner_config"]["actor"]["observation_layout"] = None
    with pytest.raises(ValueError):
        policy.policy_observation_layout_from_metadata(info)
    if not fault.startswith("runner"):
        with pytest.raises(ValueError):
            policy.supported_heteroscedastic_contract_version(contract)


@pytest.mark.parametrize("dimension,layout", [(372, None), (324, LAYOUT), (371, LAYOUT), (373, LAYOUT)])
def test_actual_actor_requires_matching_explicit_layout(dimension, layout):
    with pytest.raises(ValueError, match="layout"):
        actor(layout, observations(dimension))


def test_372_actual_kernel_is_same_conditional_gaussian_and_uses_stored_history():
    obs = observations()
    model = actor(LAYOUT, obs)
    head = model.mlp(model.get_latent(obs))
    mean = (1.-HISTORY_RHO)*head[:, 0] + HISTORY_RHO*obs["policy"][:, 195:207]
    sigma = head[:, 1].exp()
    expected = torch.distributions.Normal(mean, sigma)
    before = torch.get_rng_state()
    sampled = model(obs, stochastic_output=True)
    after = torch.get_rng_state()
    torch.set_rng_state(before)
    assert torch.equal(sampled, expected.sample())
    assert torch.equal(after, torch.get_rng_state())
    assert torch.equal(model.output_mean, mean) and torch.equal(model.output_std, sigma)
    assert torch.equal(model.get_output_log_prob(sampled), expected.log_prob(sampled).sum(-1))
    cache, before = model.distribution._distribution, rng_state()
    assert torch.equal(model(obs), mean)
    assert model.distribution._distribution is cache
    assert_rng_equal(before)
    permutation = torch.tensor([3, 0, 2, 1])
    torch.testing.assert_close(model(obs[permutation]), mean[permutation], rtol=1e-6, atol=2e-7)
    wrong_width = observations(324)
    with pytest.raises((ValueError, RuntimeError)):
        model(wrong_width)


def test_zero_appended_input_weights_preserve_initial_outputs_but_receive_gradient():
    old_obs = observations(324)
    new_obs = observations(372)
    new_obs["policy"][:, :324] = old_obs["policy"]
    new_obs["critic"] = new_obs["policy"].clone()
    old, new = actor(None, old_obs), actor(LAYOUT, new_obs)
    mapped = {key: value.detach().clone() for key, value in old.state_dict().items()}
    mapped["mlp.0.weight"] = torch.cat((mapped["mlp.0.weight"], torch.zeros(256, 48)), dim=1)
    new.load_state_dict(mapped, strict=True)
    assert torch.equal(new.mlp[0].weight[:, :324], old.mlp[0].weight)
    assert torch.count_nonzero(new.mlp[0].weight[:, 324:]) == 0
    torch.testing.assert_close(new(new_obs), old(old_obs), rtol=1e-6, atol=2e-7)
    old_head, new_head = old.mlp(old_obs["policy"]), new.mlp(new_obs["policy"])
    torch.testing.assert_close(new_head[:, 1].exp(), old_head[:, 1].exp(), rtol=1e-6, atol=2e-7)
    altered = new_obs.clone()
    altered["policy"][:, 324:] = -7.
    torch.testing.assert_close(new(altered), new(new_obs), rtol=1e-6, atol=2e-7)
    new.mlp(new_obs["policy"]).square().mean().backward()
    gradient = new.mlp[0].weight.grad[:, 324:]
    assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient) > 0
    assert set(old.state_dict()) == set(new.state_dict()) and not list(new.named_buffers())


def test_current_schema_is_exact_append_and_preserves_all_old_encoded_positions(tmp_path):
    path = Path(__file__).resolve().parents[2] / 'configs/ppo_semantic_v3/observation_schema.json'
    current = load_semantic_observation_schema(path)
    assert current.dimension == 372 and current.transfer_role_features_version == LAYOUT
    data = json.loads(path.read_text())
    assert data['feature_groups'][-1] == {'name': ROLE_OBSERVATION_GROUP, 'size': 48, 'scale': 1.0}
    data.pop('transfer_role_features_version')
    data['feature_groups'].pop()
    old_path = tmp_path / 'legacy324.json'
    old_path.write_text(json.dumps(data), encoding='utf-8')
    old = load_semantic_observation_schema(old_path)
    old_groups = {row['name']: [float(index+1)]*row['size'] for index, row in enumerate(old.groups)}
    old_groups['previous_raw_full12'] = [-100., 100., -20., 20.] + list(range(8))
    new_groups = {**old_groups, ROLE_OBSERVATION_GROUP: [.25]*48}
    legacy_encoded, encoded = old.encode(old_groups), current.encode(new_groups)
    assert encoded[:324] == legacy_encoded
    assert encoded[195:207] == (-20., 20., -20., 20., *range(8))
    assert encoded[324:] == (.25,)*48 and len(encoded) == 372


def test_372_masked_minibatch_unpads_before_selecting_the_original_history_slice():
    from rsl_rl.utils import unpad_trajectories
    model = actor(LAYOUT)
    values = torch.randn(4, 3, 372) * .2
    padded = TensorDict({'policy': values, 'critic': values.clone()}, batch_size=[4, 3])
    masks = torch.tensor([[True, True, True], [True, True, True],
                          [True, True, False], [False, False, False]])
    expected = model(unpad_trajectories(padded, masks))
    assert torch.equal(model(padded, masks=masks), expected)


def test_372_frozen_prefix_is_private_and_preserves_source_cache_rng_and_history():
    obs = observations(n=1)
    model = actor(LAYOUT, obs)
    expected = tuple(model(obs)[0].detach().tolist())
    model(obs, stochastic_output=True)
    cache = model.distribution._distribution
    before = rng_state()
    prefix = build_frozen_checkpoint_prefix_policy(model, source(model, LAYOUT))
    values = tuple(obs["policy"][0].tolist())
    assert prefix(values) == expected
    assert model.distribution._distribution is cache and prefix._actor.distribution._distribution is None
    assert not any(p.requires_grad for p in prefix._actor.parameters())
    original = {p.untyped_storage().data_ptr() for p in model.parameters()}
    assert all(p.untyped_storage().data_ptr() not in original for p in prefix._actor.parameters())
    assert_rng_equal(before)
    with pytest.raises(ValueError, match="372"):
        prefix(values[:324])
    for changed in (source(model), {**source(model, LAYOUT), "actor_parameter_sha256": "0"*64}):
        with pytest.raises(ValueError):
            build_frozen_checkpoint_prefix_policy(model, changed)
    model.observation_layout = None
    with pytest.raises(ValueError, match="layout"):
        build_frozen_checkpoint_prefix_policy(model, source(model, LAYOUT))


class RoleCore(Core):
    observation_dimension = 372

    def __init__(self):
        super().__init__([[('P06', 8, None), ('P07', 8, None), ('P08', 8, None)]])
        self.observation_schema = SimpleNamespace(transfer_role_features_version=LAYOUT)

    def reset(self, seed=1001, options=None):
        super().reset(seed, options)
        self.observation += (.2,)*48
        return self.observation

    def step(self, raw):
        result = super().step(raw)
        self.observation += (.3,)*48
        return replace(result, observation=self.observation)


def test_372_prefix_facade_retains_credit_exclusion_and_cross_phase_history():
    core, evidence = RoleCore(), []
    env = CheckpointPolicyPrefixRslAdapter(core, seed=1001, device="cpu", evidence_sink=evidence.append,
                                         request=CheckpointPolicyPrefixRequest('P07'))
    assert env.get_observations()["policy"].shape == (1, 372)
    model = actor(LAYOUT)
    frozen = build_frozen_checkpoint_prefix_policy(model, source(model, LAYOUT))
    with pytest.raises(ValueError, match="live observation layout"):
        env.install_prefix_policy(lambda obs: (0.,)*12, source(model))
    assert not core.actions and env.core._policy is None
    start = env.install_prefix_policy(frozen, frozen.provenance)
    assert start["actual_phase"] == 'P07' and start["physics_tick"] == 16
    assert env.core.prefix_decisions == 2 and env.total_decisions == 0
    assert all(row['policy_credit'] is False for row in evidence)
    prefix_last = core.actions[-1]
    _, _, done, extras = env.step(torch.zeros(1, 12))
    assert not done.item() and core.frame.state_id == 'P08'
    assert core.pre_action_history[-1]['residual'] == prefix_last
    assert env.total_decisions == 1 and core.resets == 1
    assert extras['semantic_decisions'][0]['prefix_checkpoint_policy_data_in_ppo_storage'] is False
    assert len(env._observation) == 372


@pytest.mark.parametrize('dimension,marker', [(372, None), (324, LAYOUT), (371, LAYOUT),
                                            (373, LAYOUT), (372, 'other'), (True, None)])
def test_prefix_facade_rejects_implicit_or_wrong_schema_before_reset(dimension, marker):
    core = RoleCore()
    core.observation_dimension = dimension
    core.observation_schema.transfer_role_features_version = marker
    with pytest.raises(ValueError, match='layout'):
        _core_observation_layout(core)
    assert core.resets == 0
