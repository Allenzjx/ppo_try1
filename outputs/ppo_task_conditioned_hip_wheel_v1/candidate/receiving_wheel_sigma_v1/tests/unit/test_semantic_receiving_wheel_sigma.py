"""Receiving-wheel profile regression, CPU/synthetic only; no checkpoint or Isaac."""
import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rsl_rl")
TensorDict = pytest.importorskip("tensordict").TensorDict

from wlr50_clean.ppo.semantic_history_actor import (
    SemanticTaskConditionedHipWheelHistoryMLPModel, task_conditioned_effective_log_std,
)
from wlr50_clean.ppo.semantic_receiving_wheel_profile import (
    RECEIVING_WHEEL_POLICY, RECEIVING_WHEEL_SIGMA_SEMANTICS,
)
from wlr50_clean.ppo.semantic_receiving_wheel_sigma import (
    SemanticReceivingWheelSigmaHistoryMLPModel, receiving_wheel_effective_log_std,
)
from wlr50_clean.ppo.semantic_training import audited_history_policy_request


@pytest.fixture(autouse=True)
def preserve_cpu_rng_and_threads():
    state, threads = torch.get_rng_state(), torch.get_num_threads()
    torch.manual_seed(917)
    torch.set_num_threads(1)
    yield
    torch.set_rng_state(state)
    torch.set_num_threads(threads)


def _latent(phase, placed):
    x = torch.zeros(1, 372)
    x[0, phase], x[0, 20], x[0, 157] = 1, .01, float(placed)
    x[0, 158:158 + phase] = 1
    x[0, 195:207] = torch.linspace(-.4, .2, 12)
    return x


def _actor(cls, observation):
    return cls(observation, {"actor": ["policy"]}, "actor", 12,
        observation_layout="diagonal_transfer_state_v1", exploration_std_temperature=.25,
        distribution_cfg={"class_name": "HeteroscedasticGaussianDistribution",
                          "std_type": "log", "init_std": .15})


@pytest.mark.parametrize("phase", range(13))
@pytest.mark.parametrize("placed", [False, True])
def test_all_phases_and_placement_history_change_only_two_receiving_sigmas(phase, placed):
    latent, learned = _latent(phase, placed), torch.linspace(-3., -1., 12).reshape(1, 12)
    old, before = task_conditioned_effective_log_std(learned, latent, .25)
    state = torch.get_rng_state()
    new, evidence = receiving_wheel_effective_log_std(learned, latent, .25)
    gate = torch.zeros(1, 12, dtype=torch.bool)
    if phase in (9, 10, 11) and placed:
        gate[:, [9, 11]] = True
    assert torch.equal(new, torch.where(gate, old + math.log(3.), old))
    assert torch.equal(evidence["receiving_sigma_gate_full12"], gate)
    assert torch.equal(new[~gate], old[~gate])
    assert torch.equal(evidence["current_cap_full12"], before["current_cap_full12"])
    assert torch.equal(evidence["physical_equivalent_B_full12"], before["physical_equivalent_B_full12"])
    assert torch.equal(state, torch.get_rng_state())
    assert bool((torch.isfinite(new.exp()) & (new.exp() > 0)).all())


def test_rr_placement_history_does_not_claim_current_support_or_lift():
    latent = _latent(10, True)
    for ground, current_lift in ((0, 0), (1, 0), (0, 1)):
        latent[0, 137], latent[0, 149] = ground, current_lift
        _, evidence = receiving_wheel_effective_log_std(torch.zeros(1, 12), latent, .25)
        assert evidence["receiving_sigma_gate_full12"].nonzero().tolist() == [[0, 9], [0, 11]]
    latent[0, 157] = .5
    with pytest.raises(ValueError):
        receiving_wheel_effective_log_std(torch.zeros(1, 12), latent, .25)


@pytest.mark.parametrize("phase,placed", [(4, True), (9, False), (9, True), (10, True), (11, True), (12, True)])
def test_same_weight_mean_one_gaussian_current_likelihood_and_actual_request_audit(phase, placed):
    latent = _latent(phase, placed)
    observation = TensorDict({"policy": latent, "critic": latent.clone()}, batch_size=[1])
    old = _actor(SemanticTaskConditionedHipWheelHistoryMLPModel, observation)
    actor = _actor(SemanticReceivingWheelSigmaHistoryMLPModel, observation)
    actor.load_state_dict(old.state_dict(), strict=True)
    calls = []
    handle = actor.mlp.register_forward_hook(lambda *_: calls.append(1))
    state = torch.get_rng_state()
    try:
        raw, audit = audited_history_policy_request(actor, observation,
            lambda: actor(observation, stochastic_output=True), stochastic=True)
    finally:
        handle.remove()
    after = torch.get_rng_state()
    assert len(calls) == 1
    assert audit["policy_version"] == RECEIVING_WHEEL_POLICY
    assert audit["sigma_scaling_semantics"] == RECEIVING_WHEEL_SIGMA_SEMANTICS
    assert audit["sampling_draws"] == 1 and audit["extra_model_forwards"] == audit["extra_random_draws"] == 0
    assert torch.equal(torch.tensor(audit["effective_sigma_full12"]), actor.output_std[0])
    normal = torch.distributions.Normal(actor.output_mean, actor.output_std)
    torch.set_rng_state(state)
    assert torch.equal(raw, normal.sample()) and torch.equal(after, torch.get_rng_state())
    log_prob = actor.get_output_log_prob(raw).detach().clone()
    assert torch.equal(log_prob, normal.log_prob(raw).sum(-1))
    assert audit["selected_raw_log_probability"] == log_prob[0].item()
    assert torch.equal(actor.output_entropy, normal.entropy().sum(-1))
    actor(observation, stochastic_output=True)
    assert torch.equal(actor.get_output_log_prob(raw), log_prob)
    assert torch.equal(torch.exp(actor.get_output_log_prob(raw) - log_prob), torch.ones(1))
    cache = tuple(v.clone() for v in actor.output_distribution_params)
    state = torch.get_rng_state()
    fixed, deterministic_audit = audited_history_policy_request(
        actor, observation, lambda: actor(observation), stochastic=False)
    assert torch.equal(fixed, old(observation))
    assert torch.equal(state, torch.get_rng_state())
    assert all(torch.equal(a, b) for a, b in zip(cache, actor.output_distribution_params))
    assert deterministic_audit["effective_sigma_full12"] == audit["effective_sigma_full12"]
    assert deterministic_audit["sampling_draws"] == 0
