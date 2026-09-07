"""CPU-only frozen-copy ABI and installed RSL inference regressions; no Isaac."""
from __future__ import annotations

import copy
import random
from types import MethodType

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
pytest.importorskip("rsl_rl")
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from rsl_rl.modules.distribution import HeteroscedasticGaussianDistribution

from wlr50_clean.ppo.semantic_checkpoint_prefix_policy import build_frozen_checkpoint_prefix_policy
from wlr50_clean.ppo.semantic_policy_distribution import STATE_DEPENDENT_POLICY, policy_contract
from wlr50_clean.ppo.semantic_training import parameter_hash, semantic_runner_config


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class FixedNormalizer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("offset", torch.full((324,), 0.125))
        self.register_buffer("scale", torch.full((324,), 1.25))
        self.register_buffer("count", torch.tensor(17))
        self.calls = 0

    def forward(self, values):
        self.calls += 1
        return (values - self.offset) / self.scale


def actual_actor():
    """Official kernel, with explicit stateful-normalizer preservation fixture."""
    cfg = copy.deepcopy(semantic_runner_config(seed=1001, device="cpu", semantic_version="v3",
                                               policy_version=STATE_DEPENDENT_POLICY)["actor"])
    cfg.pop("class_name")
    actor = MLPModel(batch(), {"actor": ["policy"]}, "actor", 12, **cfg)
    actor.obs_normalizer = FixedNormalizer()
    return actor


def final_linear(actor):
    return [module for module in actor.mlp.modules() if isinstance(module, torch.nn.Linear)][-1]


def record(actor):
    return {
        "checkpoint_path": "verified/source_000089728.pt",
        "checkpoint_sha256": "a" * 64,
        "actor_parameter_sha256": parameter_hash(actor),
        "source_global_policy_decisions": 89728,
        "source_ppo_updates": 664,
        "policy_contract": policy_contract(STATE_DEPENDENT_POLICY),
        "source_runtime_content_sha256": "b" * 64,
    }


def observation():
    return tuple((i - 162) / 324 for i in range(324))


def batch(values=None):
    tensor = torch.tensor([observation() if values is None else values], dtype=torch.float32)
    return TensorDict({"policy": tensor, "critic": tensor.clone()}, batch_size=[1])


def rng_state():
    return torch.get_rng_state().clone(), random.getstate(), copy.deepcopy(np.random.get_state())


def assert_rng_equal(before):
    after = rng_state()
    assert torch.equal(before[0], after[0])
    assert before[1] == after[1]
    assert before[2][0] == after[2][0]
    assert np.array_equal(before[2][1], after[2][1])
    assert before[2][2:] == after[2][2:]


def test_actual_actor_copy_preserves_buffers_modes_rng_and_raw_mean_without_source_calls():
    actor = actual_actor()
    with torch.no_grad():
        final_linear(actor).bias[:12].fill_(2.0)  # Returned latent must not be tanh-clipped.
    expected = tuple(actor(batch(), stochastic_output=False)[0].detach().tolist())
    actor.obs_normalizer.eval()  # Source mixed training modes must also survive.
    modes = [m.training for m in actor.modules()]
    original_state = {k: v.clone() for k, v in actor.state_dict().items()}
    original_calls = actor.obs_normalizer.calls
    before = rng_state()
    policy = build_frozen_checkpoint_prefix_policy(actor, record(actor))
    assert policy(observation()) == expected
    assert policy(observation()) == expected
    assert any(abs(v) > 1 for v in expected)
    assert actor.obs_normalizer.calls == original_calls
    assert [m.training for m in actor.modules()] == modes
    assert all(torch.equal(v, actor.state_dict()[k]) for k, v in original_state.items())
    assert all(p.requires_grad for p in actor.parameters())
    assert not any(p.requires_grad for p in policy._actor.parameters())
    assert not any(m.training for m in policy._actor.modules())
    assert_rng_equal(before)


def test_prefix_stays_fixed_when_main_optimizer_updates_and_normalizer_changes():
    actor = actual_actor()
    optimizer = torch.optim.Adam(actor.parameters(), lr=0.01)
    policy = build_frozen_checkpoint_prefix_policy(actor, record(actor))
    old_output = policy(observation())
    frozen_hash = parameter_hash(policy._actor)
    optimizer.zero_grad()
    actor(batch(), stochastic_output=False).square().mean().backward()
    optimizer.step()
    with torch.no_grad():
        actor.obs_normalizer.offset.add_(1)
        actor.obs_normalizer.count.add_(1)
    assert parameter_hash(actor) != frozen_hash
    assert parameter_hash(policy._actor) == frozen_hash
    assert policy(observation()) == old_output
    frozen_ids = {id(p) for p in policy._actor.parameters()}
    assert not any(id(p) in frozen_ids for g in optimizer.param_groups for p in g["params"])


def test_actual_official_heteroscedastic_nonleaf_cache_and_learned_std_are_preserved():
    cfg = copy.deepcopy(semantic_runner_config(seed=1001, device="cpu", semantic_version="v3",
                                               policy_version=STATE_DEPENDENT_POLICY)["actor"])
    cfg.pop("class_name")
    actor = MLPModel(batch(), {"actor": ["policy"]}, "actor", 12, **cfg)
    final_linear = [m for m in actor.mlp.modules() if isinstance(m, torch.nn.Linear)][-1]
    # Mimic an already learned std head, not the constructor's zero std weights.
    with torch.no_grad():
        final_linear.weight[12:].fill_(0.013)
        final_linear.bias[12:].copy_(torch.linspace(-2, -1, 12))
    expected = tuple(actor(batch(), stochastic_output=False)[0].detach().tolist())
    actor(batch(), stochastic_output=True)
    cache = actor.distribution._distribution
    assert cache.loc.grad_fn is not None and cache.scale.grad_fn is not None
    with pytest.raises(RuntimeError, match="deepcopy"):
        copy.deepcopy(actor)
    mean, std = cache.loc.detach().clone(), cache.scale.detach().clone()
    before = rng_state()
    policy = build_frozen_checkpoint_prefix_policy(actor, record(actor))
    assert actor.distribution._distribution is cache
    assert policy._actor.distribution._distribution is None
    assert parameter_hash(policy._actor) == parameter_hash(actor)
    assert isinstance(policy._actor.distribution, HeteroscedasticGaussianDistribution)
    frozen_final = [m for m in policy._actor.mlp.modules() if isinstance(m, torch.nn.Linear)][-1]
    assert torch.equal(frozen_final.weight[12:], final_linear.weight[12:])
    assert policy(observation()) == expected
    assert policy(observation()) == expected
    assert policy._actor.distribution._distribution is None
    assert actor.distribution._distribution is cache
    assert torch.equal(cache.loc, mean) and torch.equal(cache.scale, std)
    assert actor.training and all(p.requires_grad for p in actor.parameters())
    assert_rng_equal(before)


def test_provenance_is_json_and_detached_from_both_input_and_returned_dict():
    actor = actual_actor()
    source = record(actor)
    policy = build_frozen_checkpoint_prefix_policy(actor, source)
    source["policy_contract"]["raw_action_dimension"] = 99
    obtained = policy.provenance
    obtained["policy_contract"]["observation_dimension"] = 0
    assert policy.provenance["policy_contract"] == policy_contract(STATE_DEPENDENT_POLICY)
    assert policy.provenance["source_global_policy_decisions"] == 89728
    assert policy.provenance["source_ppo_updates"] == 664
    assert policy.provenance["frozen_actor_parameter_sha256"] == parameter_hash(actor)
    assert policy.provenance["independent_parameter_and_buffer_storage_verified"] is True


@pytest.mark.parametrize("key", ["checkpoint_path", "checkpoint_sha256", "actor_parameter_sha256",
                                "source_global_policy_decisions", "source_ppo_updates", "policy_contract"])
def test_required_provenance_missing_is_rejected(key):
    actor = actual_actor()
    source = record(actor)
    del source[key]
    with pytest.raises(ValueError):
        build_frozen_checkpoint_prefix_policy(actor, source)


@pytest.mark.parametrize("key,value", [
    ("checkpoint_path", " "), ("checkpoint_sha256", "x" * 64),
    ("actor_parameter_sha256", "0" * 64), ("source_global_policy_decisions", True),
    ("source_ppo_updates", -1), ("source_ppo_updates", 1.5),
    ("source_runtime_content_sha256", "invalid"), ("extra", float("nan")),
])
def test_invalid_or_mismatched_source_provenance_is_rejected(key, value):
    actor = actual_actor()
    source = record(actor)
    source[key] = value
    with pytest.raises(ValueError):
        build_frozen_checkpoint_prefix_policy(actor, source)


@pytest.mark.parametrize("field,value", [("observation_dimension", 332), ("raw_action_dimension", 13),
                                       ("state_dependent_std", 1), ("version", "gaussian_scalar_v1")])
def test_wrong_policy_contract_is_rejected(field, value):
    actor = actual_actor()
    source = record(actor)
    source["policy_contract"][field] = value
    with pytest.raises(ValueError, match="contract"):
        build_frozen_checkpoint_prefix_policy(actor, source)


@pytest.mark.parametrize("invalid", [(0.0,) * 323, (0.0,) * 325, "x", None,
                                    (float("nan"),) * 324, (float("inf"),) * 324,
                                    (True,) * 324, ("0",) * 324, (1e100,) * 324])
def test_invalid_observation_is_rejected_without_source_or_prefix_forward(invalid):
    actor = actual_actor()
    policy = build_frozen_checkpoint_prefix_policy(actor, record(actor))
    with pytest.raises((ValueError, RuntimeError)):
        policy(invalid)
    assert actor.obs_normalizer.calls == 0 and policy._actor.obs_normalizer.calls == 0


@pytest.mark.parametrize("kind", ["nan", "shape", "list"])
def test_nonfinite_or_malformed_action_is_rejected(kind):
    actor = actual_actor()
    policy = build_frozen_checkpoint_prefix_policy(actor, record(actor))
    # Exercise the output guard independently after a valid real-kernel copy;
    # a forged source kernel is rejected at construction in separate tests.
    def malformed(self, observations, *, stochastic_output=False):
        result = MLPModel.forward(self, observations, stochastic_output=stochastic_output)
        if kind == "nan":
            return result * float("nan")
        if kind == "shape":
            return result[:, :11]
        return result.tolist()
    policy._actor.forward = MethodType(malformed, policy._actor)
    with pytest.raises(ValueError):
        policy(observation())


def test_unknown_nonleaf_state_fails_without_mutating_source():
    actor = actual_actor()
    actor.unrecognized_cached_tensor = final_linear(actor).weight * 2
    before = parameter_hash(actor)
    with pytest.raises(ValueError, match="without resetting"):
        build_frozen_checkpoint_prefix_policy(actor, record(actor))
    assert actor.training and parameter_hash(actor) == before
    assert actor.unrecognized_cached_tensor.grad_fn is not None


def test_deepcopy_returning_source_is_rejected_before_freezing_source(monkeypatch):
    actor = actual_actor()
    monkeypatch.setattr(MLPModel, "__deepcopy__", lambda self, memo: self, raising=False)
    with pytest.raises(ValueError, match="shares a source module"):
        build_frozen_checkpoint_prefix_policy(actor, record(actor))
    assert actor.training and all(p.requires_grad for p in actor.parameters())


@pytest.mark.parametrize("mutation", ["recurrent", "wrong_dimension", "double", "cache"])
def test_unsupported_actor_interface_fails_closed(mutation):
    actor = actual_actor()
    if mutation == "recurrent":
        actor.is_recurrent = True
    elif mutation == "wrong_dimension":
        actor.obs_dim = 8
    elif mutation == "double":
        actor.double()
    else:
        actor.distribution._distribution = object()
    with pytest.raises(ValueError):
        build_frozen_checkpoint_prefix_policy(actor, record(actor))
