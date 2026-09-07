"""CPU real-kernel prefix contracts; fake core checks credit, not live physics."""
from __future__ import annotations

import copy
from types import MethodType

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rsl_rl")
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from wlr50_clean.ppo.semantic_checkpoint_prefix import _provenance
from wlr50_clean.ppo.semantic_checkpoint_prefix_policy import (
    _source_record, build_frozen_checkpoint_prefix_policy,
)
from wlr50_clean.ppo.semantic_history_actor import SemanticHistoryMLPModel
from wlr50_clean.ppo.semantic_policy_distribution import (
    HISTORY_POLICY, STATE_DEPENDENT_POLICY, policy_contract,
)
from wlr50_clean.ppo.semantic_training import parameter_hash, semantic_runner_config
from test_semantic_checkpoint_prefix import Core, Request, adapter
from test_semantic_checkpoint_prefix_policy import assert_rng_equal, rng_state

VERSIONS = (STATE_DEPENDENT_POLICY, HISTORY_POLICY)


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def observation(history=0.):
    values = torch.linspace(-.3, .3, 324)
    values[195:207] = history
    return tuple(values.tolist())


def batch(values):
    policy = torch.tensor([values], dtype=torch.float32)
    return TensorDict({"policy": policy, "critic": policy.clone()}, batch_size=[1])


def actual_actor(version):
    cfg = copy.deepcopy(semantic_runner_config(
        seed=1001, device="cpu", semantic_version="v3", policy_version=version)["actor"])
    cfg.pop("class_name")
    cls = SemanticHistoryMLPModel if version == HISTORY_POLICY else MLPModel
    actor = cls(batch(observation()), {"actor": ["policy"]}, "actor", 12, **cfg)
    # Exercise learned, nonconstant log-std head values with the official model.
    last = [m for m in actor.mlp.modules() if isinstance(m, torch.nn.Linear)][-1]
    with torch.no_grad():
        last.weight[12:].fill_(.013)
        last.bias[12:].copy_(torch.linspace(-2., -1., 12))
    return actor


def source_record(actor, version):
    # These are explicit CPU fixture identities, never claims about a live CP.
    return {"checkpoint_path": "verified/cpu_source.pt", "checkpoint_sha256": "a"*64,
            "actor_parameter_sha256": parameter_hash(actor),
            "source_global_policy_decisions": 118016, "source_ppo_updates": 887,
            "source_runtime_content_sha256": "b"*64,
            "policy_contract": policy_contract(version)}


@pytest.mark.parametrize("version", VERSIONS)
def test_real_old_and_history_prefix_are_private_equal_deterministic_and_rng_neutral(version):
    actor = actual_actor(version)
    obs = observation(2.)
    expected = tuple(actor(batch(obs), stochastic_output=False)[0].detach().tolist())
    actor(batch(obs), stochastic_output=True)
    cache = actor.distribution._distribution
    assert cache.loc.grad_fn is not None and cache.scale.grad_fn is not None
    mean, std = cache.loc.detach().clone(), cache.scale.detach().clone()
    state = {k: v.clone() for k, v in actor.state_dict().items()}
    modes = [m.training for m in actor.modules()]
    source = source_record(actor, version)
    before = rng_state()
    prefix = build_frozen_checkpoint_prefix_policy(actor, source)
    assert type(prefix._actor) is type(actor)
    assert prefix(obs) == expected
    prefix(observation(-3.))
    assert prefix(obs) == expected  # No private previous-call action memory.
    assert prefix._actor.distribution._distribution is None
    assert actor.distribution._distribution is cache
    assert torch.equal(cache.loc, mean) and torch.equal(cache.scale, std)
    assert all(torch.equal(actor.state_dict()[k], v) for k, v in state.items())
    assert [m.training for m in actor.modules()] == modes
    assert all(p.requires_grad for p in actor.parameters())
    assert not any(p.requires_grad for p in prefix._actor.parameters())
    original = dict(actor.named_parameters()) | dict(actor.named_buffers())
    copied = dict(prefix._actor.named_parameters()) | dict(prefix._actor.named_buffers())
    assert original.keys() == copied.keys()
    source_storages = {v.untyped_storage().data_ptr() for v in original.values() if v.numel()}
    assert all(torch.equal(v, copied[k]) for k, v in original.items())
    assert all(v.untyped_storage().data_ptr() not in source_storages for v in copied.values() if v.numel())
    assert prefix.provenance["policy_contract"] == policy_contract(version)
    assert prefix.provenance["frozen_actor_parameter_sha256"] == source["actor_parameter_sha256"]
    assert_rng_equal(before)


def test_history_prefix_uses_observation_raw_slice_and_preserves_innovation_std():
    actor = actual_actor(HISTORY_POLICY)
    obs = observation(20.)
    tensors = batch(obs)
    with torch.no_grad():
        head = actor.mlp(actor.get_latent(tensors))
        expected = (1.-.9)*head[:, 0, :] + .9*tensors["policy"][:, 195:207]
        base_std = head[:, 1, :].exp()
    actor(tensors, stochastic_output=True)
    assert torch.equal(actor.output_mean, expected)
    assert torch.equal(actor.output_std, base_std)
    prefix = build_frozen_checkpoint_prefix_policy(actor, source_record(actor, HISTORY_POLICY))
    before = rng_state()
    assert prefix(obs) == tuple(expected[0].tolist())
    assert min(prefix(obs)) > 1.  # Raw latent, not an extra tanh/projected action.
    assert_rng_equal(before)
    too_large = list(obs)
    too_large[195] = 20.01
    with pytest.raises(ValueError, match="schema-clipped"):
        prefix(too_large)


@pytest.mark.parametrize("version", VERSIONS)
def test_prefix_stays_at_source_when_actual_main_adam_changes_learned_weights(version):
    actor = actual_actor(version)
    prefix = build_frozen_checkpoint_prefix_policy(actor, source_record(actor, version))
    values = observation(1.)
    expected = prefix(values)
    frozen_hash = parameter_hash(prefix._actor)
    optimizer = torch.optim.Adam(actor.parameters(), lr=.01)
    optimizer.zero_grad()
    actor(batch(values), stochastic_output=False).square().mean().backward()
    optimizer.step()
    assert parameter_hash(actor) != frozen_hash
    assert parameter_hash(prefix._actor) == frozen_hash
    assert prefix(values) == expected
    assert not ({id(p) for p in prefix._actor.parameters()}
                & {id(p) for g in optimizer.param_groups for p in g["params"]})


@pytest.mark.parametrize("actual_version,declared_version", [
    (STATE_DEPENDENT_POLICY, HISTORY_POLICY), (HISTORY_POLICY, STATE_DEPENDENT_POLICY),
])
def test_identical_tensor_hash_cannot_relabel_actual_old_or_history_kernel(actual_version, declared_version):
    actor = actual_actor(actual_version)
    other = actual_actor(declared_version)
    other.load_state_dict(actor.state_dict(), strict=True)
    assert parameter_hash(other) == parameter_hash(actor)
    with pytest.raises(ValueError, match="kernel/class"):
        build_frozen_checkpoint_prefix_policy(actor, source_record(actor, declared_version))


@pytest.mark.parametrize("version", VERSIONS)
def test_actor_subclass_is_not_the_exact_verified_kernel(version):
    cls = SemanticHistoryMLPModel if version == HISTORY_POLICY else MLPModel
    class OtherKernel(cls):
        pass
    actor = actual_actor(version)
    actor.__class__ = OtherKernel
    with pytest.raises(ValueError, match="kernel/class"):
        build_frozen_checkpoint_prefix_policy(actor, source_record(actor, version))


@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("method", ["forward", "get_latent", "deterministic_output"])
def test_instance_kernel_replacement_is_rejected_even_with_equal_parameters(version, method):
    actor = actual_actor(version)
    record = source_record(actor, version)
    owner = actor.distribution if method == "deterministic_output" else actor
    setattr(owner, method, MethodType(lambda self, *args, **kwargs: None, owner))
    assert parameter_hash(actor) == record["actor_parameter_sha256"]
    with pytest.raises(ValueError):
        build_frozen_checkpoint_prefix_policy(actor, record)


@pytest.mark.parametrize("change", ["normalizer", "flag"])
def test_history_kernel_cannot_reinterpret_schema_history_with_another_normalizer(change):
    actor = actual_actor(HISTORY_POLICY)
    if change == "normalizer":
        actor.obs_normalizer = torch.nn.ReLU()
    else:
        actor.obs_normalization = True
    with pytest.raises(ValueError, match="identity-normalized"):
        build_frozen_checkpoint_prefix_policy(actor, source_record(actor, HISTORY_POLICY))


@pytest.mark.parametrize("field,value", [
    ("rho", .8), ("rho", True), ("history_slice", [194, 206]),
    ("history_clip", 21.), ("state_dependent_std", 1), ("actor_class", "MLPModel"),
])
def test_both_prefix_contract_boundaries_reject_changed_history_semantics(field, value):
    actor = actual_actor(HISTORY_POLICY)
    record = source_record(actor, HISTORY_POLICY)
    record["policy_contract"][field] = value
    for validate in (_source_record, _provenance):
        with pytest.raises(ValueError, match="contract"):
            validate(record)


@pytest.mark.parametrize("version", VERSIONS)
def test_ordinary_matching_checkpoint_prefix_runs_without_migration_and_excludes_credit(version):
    core = Core([[('P06', 8, None), ('P07', 8, None), ('P08', 8, None)]])
    env, core, evidence = adapter(core, Request('P07'))
    actor = actual_actor(version)
    prefix = build_frozen_checkpoint_prefix_policy(actor, source_record(actor, version))
    source = copy.deepcopy(prefix.provenance)
    before = rng_state()
    start = env.install_prefix_policy(prefix, source)
    assert start["actual_phase"] == 'P07' and start["physics_tick"] == 16
    assert core.resets == 1 and len(core.actions) == 2
    assert env.total_decisions == 0 and env.core.credited_ticks == 0
    assert env.core.prefix_decisions == 2 and env.core.prefix_ticks == 16
    assert all(row["policy_credit"] is False for row in evidence)
    assert env.cfg["prefix_policy_provenance"]["policy_contract"] == policy_contract(version)
    source["policy_contract"]["version"] = 'tampered'
    assert prefix.provenance["policy_contract"]["version"] == version
    prefix_last = core.actions[-1]
    _, _, done, extras = env.step(torch.zeros(1, 12))
    assert not done.item() and env.total_decisions == 1
    assert core.pre_action_history[-1]["residual"] == prefix_last
    assert extras["semantic_decisions"][0]["prefix_checkpoint_policy_data_in_ppo_storage"] is False
    assert env.core.prefix_ticks == 16 and env.core.credited_ticks == 8
    assert actor.distribution._distribution is None
    assert_rng_equal(before)
