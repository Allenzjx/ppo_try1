"""CPU-only candidate tests; no production overlay or physical/PPO credit.

Run with --confcutdir equal to this directory (not the older candidate parent).
The complete candidate package is selected once by conftest; production source
is untouched and the semantic loader is not bypassed in integration tests.
"""
from __future__ import annotations

import copy
import importlib.util
import math
from pathlib import Path
import sys

import pytest
import torch
from tensordict import TensorDict

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "src"))
from wlr50_clean.ppo import semantic_history_actor as old
from wlr50_clean.ppo import semantic_policy_distribution as policy
from wlr50_clean.ppo.semantic_training import state_hash


def load_new_module(name):
    fullname = "wlr50_clean.ppo." + name
    path = HERE / "src/wlr50_clean/ppo" / (name + ".py")
    if fullname in sys.modules:
        assert Path(sys.modules[fullname].__file__).resolve() == path.resolve()
        return sys.modules[fullname]
    spec = importlib.util.spec_from_file_location(fullname, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[fullname] = module
    spec.loader.exec_module(module)
    return module


profile = load_new_module("semantic_receiving_wheel_profile")
new = load_new_module("semantic_receiving_wheel_sigma")
LAYOUT = "diagonal_transfer_state_v1"


@pytest.fixture(autouse=True)
def isolated_cpu_rng():
    state, threads = torch.get_rng_state(), torch.get_num_threads()
    torch.manual_seed(917)
    torch.set_num_threads(1)
    yield
    torch.set_rng_state(state)
    torch.set_num_threads(threads)


def values(stage, placed):
    x = torch.zeros(1, 372)
    x[0, stage] = 1
    x[0, 20] = .01
    x[0, 158:158 + stage] = 1
    x[0, 157] = float(placed)
    x[0, 195:207] = torch.linspace(-.4, .2, 12)
    return x


def obs(x):
    return TensorDict({"policy": x, "critic": x.clone()}, batch_size=[x.shape[0]])


def model(cls, observation):
    return cls(observation, {"actor": ["policy"]}, "actor", 12,
        observation_layout=LAYOUT, exploration_std_temperature=.25,
        distribution_cfg={"class_name": "HeteroscedasticGaussianDistribution",
                          "std_type": "log", "init_std": .15})


@pytest.mark.parametrize("stage", range(13))
@pytest.mark.parametrize("placed", [False, True])
def test_exact_gate_only_FR_RR_sigma(stage, placed):
    x = values(stage, placed)
    logs = torch.linspace(-3., -1., 12).reshape(1, 12)
    baseline, before = old.task_conditioned_effective_log_std(logs, x, .25)
    rng = torch.get_rng_state()
    result, audit = new.receiving_wheel_effective_log_std(logs, x, .25)
    assert torch.equal(rng, torch.get_rng_state())
    gate = torch.zeros(1, 12, dtype=torch.bool)
    if stage in (9, 10, 11) and placed:
        gate[:, [9, 11]] = True
    expected = torch.where(gate, baseline + math.log(3.), baseline)
    assert torch.equal(result, expected)
    assert torch.equal(audit["receiving_sigma_gate_full12"], gate)
    assert torch.equal(result[~gate], baseline[~gate])
    assert torch.equal(audit["current_cap_full12"], before["current_cap_full12"])
    assert torch.equal(audit["physical_equivalent_B_full12"], before["physical_equivalent_B_full12"])
    assert bool((torch.isfinite(result.exp()) & (result.exp() > 0)).all())
    if bool(gate.any()):
        torch.testing.assert_close(result.exp()[gate], 3 * baseline.exp()[gate], rtol=4e-7, atol=0)


@pytest.mark.parametrize("contact_pair,current_lift", [(0, 0), (1, 0), (0, 1)])
def test_placed_history_is_not_current_support_or_current_lift(contact_pair, current_lift):
    x = values(10, True)
    x[0, 137] = contact_pair  # RR ground pair; active/ambiguous support does not define this gate.
    x[0, 149] = current_lift
    _, audit = new.receiving_wheel_effective_log_std(torch.zeros(1, 12), x, .25)
    assert audit["receiving_sigma_gate_full12"].nonzero().tolist() == [[0, 9], [0, 11]]


def test_mixed_batch_uses_each_stored_state_without_broadcast_leak():
    states = [(s, p) for s in range(13) for p in (False, True)]
    x = torch.cat([values(s, p) for s, p in states])
    logs = torch.zeros(len(states), 12)
    result, audit = new.receiving_wheel_effective_log_std(logs, x, .25)
    for i, (s, p) in enumerate(states):
        single, one = new.receiving_wheel_effective_log_std(logs[i:i+1], x[i:i+1], .25)
        assert torch.equal(result[i:i+1], single)
        assert torch.equal(audit["receiving_sigma_gate_full12"][i:i+1], one["receiving_sigma_gate_full12"])


@pytest.mark.parametrize("bad", ["nonbinary", "nonfinite", "shape", "temperature", "overflow"])
def test_parent_contract_and_positive_sigma_fail_closed(bad):
    x, logs, temperature = values(10, True), torch.zeros(1, 12), .25
    if bad == "nonbinary": x[0, 157] = .5
    if bad == "nonfinite": x[0, 157] = float("nan")
    if bad == "shape": logs = logs[:, :11]
    if bad == "temperature": temperature = .5
    if bad == "overflow": logs[:] = 89.5  # parent exp finite, selected x3 exp overflows.
    with pytest.raises(ValueError):
        new.receiving_wheel_effective_log_std(logs, x, temperature)


def test_complete_registered_profile_exact_diff():
    baseline = policy.policy_contract(policy.TASK_CONDITIONED_HIP_WHEEL_POLICY, observation_layout=LAYOUT)
    contract = profile.receiving_wheel_policy_contract(observation_layout=LAYOUT)
    changed = {k for k in baseline if baseline[k] != contract[k]}
    assert changed == {"version", "actor_class", "sigma_scaling_semantics", "conditional_std", "effective_log_std"}
    assert contract["rho"] == .9 and contract["directional_bias"] is False
    assert contract["receiving_continuation_gate"]["channel_indices"] == [9, 11]
    assert contract["phase_caps_full12"] == baseline["phase_caps_full12"]
    with pytest.raises(ValueError): profile.receiving_wheel_policy_contract()
    assert policy.supported_heteroscedastic_contract_version(contract) == profile.RECEIVING_WHEEL_POLICY


@pytest.mark.parametrize("stage,placed", [(4, True), (9, False), (9, True), (10, True), (11, True), (12, True)])
def test_official_Gaussian_one_draw_likelihood_entropy_KL_and_mean_exact(stage, placed):
    observation = obs(values(stage, placed))
    baseline = model(old.SemanticTaskConditionedHipWheelHistoryMLPModel, observation)
    candidate = model(new.SemanticReceivingWheelSigmaHistoryMLPModel, observation)
    candidate.load_state_dict(baseline.state_dict(), strict=True)
    calls = []
    handle = candidate.mlp.register_forward_hook(lambda *_: calls.append(1))
    rng = torch.get_rng_state()
    action = candidate(observation, stochastic_output=True)
    handle.remove()
    after = torch.get_rng_state()
    assert len(calls) == 1
    gaussian = torch.distributions.Normal(candidate.output_mean, candidate.output_std)
    torch.set_rng_state(rng)
    assert torch.equal(action, gaussian.sample()) and torch.equal(after, torch.get_rng_state())
    saved_log_prob = candidate.get_output_log_prob(action).detach().clone()
    saved_params = tuple(v.detach().clone() for v in candidate.output_distribution_params)
    assert torch.equal(saved_log_prob, gaussian.log_prob(action).sum(-1))
    assert torch.equal(candidate.output_entropy, gaussian.entropy().sum(-1))
    candidate(observation, stochastic_output=True)  # unchanged weights, same profile, current likelihood.
    assert torch.equal(candidate.get_output_log_prob(action), saved_log_prob)
    assert torch.equal(torch.exp(candidate.get_output_log_prob(action) - saved_log_prob), torch.ones(1))
    assert torch.equal(candidate.distribution.kl_divergence(saved_params, candidate.output_distribution_params), torch.zeros(1))
    cached = tuple(v.clone() for v in candidate.output_distribution_params)
    rng = torch.get_rng_state()
    assert torch.equal(candidate(observation), baseline(observation))
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(x, y) for x, y in zip(cached, candidate.output_distribution_params))
    baseline(observation, stochastic_output=True)
    assert torch.equal(candidate.output_mean, baseline.output_mean)
    if stage in (9, 10, 11) and placed:
        # A legacy stored likelihood cannot be reused at this profile boundary.
        assert not torch.equal(candidate.get_output_log_prob(action), baseline.get_output_log_prob(action))
    else:
        assert torch.equal(candidate.output_std, baseline.output_std)


def test_direct_state_roundtrip_preserves_all_weights_full_Adam_LR_Identity_RNG_and_method_ledger(tmp_path):
    """Topology feasibility only: NOT an official semantic migration receipt."""
    observation = obs(values(10, True))
    source = model(old.SemanticTaskConditionedHipWheelHistoryMLPModel, observation)
    source_critic = torch.nn.Linear(372, 1)
    optimizer = torch.optim.Adam([{"params": list(source.parameters())},
                                 {"params": list(source_critic.parameters())}], lr=2.25e-5)
    # Synthetic population of full moments, not real PPO or auxiliary credit.
    source(observation, stochastic_output=True)
    loss = source.output_mean.square().sum() + source.output_std.square().sum() + source_critic(observation["critic"]).square().sum()
    loss.backward(); optimizer.step(); optimizer.zero_grad()
    infos = {"CPU_SYNTHETIC_NOT_REAL_CHECKPOINT": True,
        "global_policy_decisions": 1024, "ppo_updates": 8, "optimizer_steps": 160,
        "branch": {"counter_origin": {"global_policy_decisions": 0},
                   "auxiliary_mean_learning": {"accepted": 7, "attempted": 8}},
        "stage_spent": {"full_episode": 1024},
        "profile": profile.receiving_wheel_policy_contract(observation_layout=LAYOUT)}
    payload = {"actor": source.state_dict(), "critic": source_critic.state_dict(),
               "optimizer": optimizer.state_dict(), "infos": infos,
               "rng": torch.get_rng_state(), "normalizer": source.obs_normalizer.state_dict()}
    path = tmp_path / "synthetic_topology_feasibility.pt"
    torch.save(payload, path)
    restored = torch.load(path, map_location="cpu", weights_only=False)
    target = model(new.SemanticReceivingWheelSigmaHistoryMLPModel, observation)
    target_critic = torch.nn.Linear(372, 1)
    target_optimizer = torch.optim.Adam([{"params": list(target.parameters())},
                                        {"params": list(target_critic.parameters())}], lr=.0003)
    target.load_state_dict(restored["actor"], strict=True)
    target_critic.load_state_dict(restored["critic"], strict=True)
    target_optimizer.load_state_dict(restored["optimizer"])
    target.obs_normalizer.load_state_dict(restored["normalizer"], strict=True)
    assert type(source.obs_normalizer) is torch.nn.Identity and type(target.obs_normalizer) is torch.nn.Identity
    assert state_hash(target.state_dict()) == state_hash(source.state_dict())
    assert state_hash(target_critic.state_dict()) == state_hash(source_critic.state_dict())
    assert state_hash(target_optimizer.state_dict()) == state_hash(optimizer.state_dict())
    assert all(g["lr"] == 2.25e-5 for g in target_optimizer.param_groups)
    assert restored["infos"] == infos
    clone = model(new.SemanticReceivingWheelSigmaHistoryMLPModel, observation)
    clone.load_state_dict(target.state_dict(), strict=True)
    torch.set_rng_state(restored["rng"])
    selected = target(observation, stochastic_output=True)
    selected_log_prob, after = target.get_output_log_prob(selected).detach(), torch.get_rng_state()
    torch.set_rng_state(restored["rng"])
    assert torch.equal(selected, clone(observation, stochastic_output=True))
    assert torch.equal(selected_log_prob, clone.get_output_log_prob(selected))
    assert torch.equal(after, torch.get_rng_state())
