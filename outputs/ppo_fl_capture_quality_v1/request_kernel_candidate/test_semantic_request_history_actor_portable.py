"""Synthetic CPU request-history regression; no files, checkpoints or physics.

Portable to tests/unit: imports only installed dependencies and project modules.
"""
from __future__ import annotations

import copy
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rsl_rl")
TensorDict = pytest.importorskip("tensordict").TensorDict

from wlr50_clean.ppo.semantic_history_actor import (
    SemanticQuarterTemperedHistoryMLPModel,
    SemanticCapTransitionQuarterHistoryMLPModel,
    cap_transition_request_history,
)
from wlr50_clean.ppo.semantic_transfer_roles import ROLE_OBSERVATION_LAYOUT

SCALES = (4., 4., 4., 6., 4., 4., 4., 4., .12, .12, .12, .12)
P05_CAP = (18., 24., 18., 24., 12., 18., 12., 18., 1., .6, 1., .6)
P06_CAP = (32., 36., 24., 112., 24., 36., 24., 36., 1.2, 1.2, 1., .6)


@pytest.fixture(autouse=True)
def cpu_rng_scope():
    rng, threads = torch.get_rng_state(), torch.get_num_threads()
    torch.manual_seed(4187)
    torch.set_num_threads(1)
    yield
    torch.set_rng_state(rng)
    torch.set_num_threads(threads)


def values(phase=5, *, age_s=0., predecessor=True):
    """Zero-based phase; prior REQUEST is deliberately unlike prior raw."""
    data = torch.zeros(1, 372)
    data[0, phase] = 1.
    data[0, 20] = age_s / 200.
    if predecessor and phase:
        data[0, 158:158 + phase] = 1.
    data[0, 195:207] = torch.linspace(-.8, .8, 12)
    data[0, 207:219] = torch.tensor([1., -2., 3., -4., 2., -3., 2., -3., .2, -.2, .2, -.2]) / torch.tensor(SCALES)
    return data


def observation(data):
    return TensorDict({"policy": data}, batch_size=[data.shape[0]])


def actors(data):
    kwargs = dict(hidden_dims=[16], activation="elu", obs_normalization=False,
        distribution_cfg={"class_name": "HeteroscedasticGaussianDistribution", "std_type": "log", "init_std": .15},
        observation_layout=ROLE_OBSERVATION_LAYOUT, exploration_std_temperature=.25)
    old = SemanticQuarterTemperedHistoryMLPModel(observation(data), {"actor": ["policy"]}, "actor", 12, **copy.deepcopy(kwargs))
    new = SemanticCapTransitionQuarterHistoryMLPModel(observation(data), {"actor": ["policy"]}, "actor", 12, **kwargs)
    new.load_state_dict(old.state_dict(), strict=True)
    return old, new


@pytest.mark.parametrize("phase,age,predecessor", [(0, 0., False), (1, 0., True),
    (4, 0., True), (5, 1 / 15, True), (5, 0., False), (6, 0., True)])
def test_nontransition_or_reset_without_completed_predecessor_keeps_raw(phase, age, predecessor):
    data = values(phase, age_s=age, predecessor=predecessor)
    center, evidence = cap_transition_request_history(data)
    assert not evidence["gate_full12"].any()
    assert torch.equal(center, data[:, 195:207])
    old, new = actors(data)
    assert torch.equal(old(observation(data)), new(observation(data)))


@pytest.mark.parametrize("phase,channels", [(2, [8, 10]), (5, list(range(10)))])
def test_only_increased_cap_channels_replace_center(phase, channels):
    data = values(phase)
    original, rng = data.clone(), torch.get_rng_state()
    center, evidence = cap_transition_request_history(data)
    gate = evidence["gate_full12"]
    assert gate[0].nonzero().flatten().tolist() == channels
    assert torch.equal(center[~gate], data[:, 195:207][~gate])
    recovered = center.tanh() * evidence["current_cap_full12"]
    torch.testing.assert_close(recovered[gate], evidence["previous_filtered_request_full12"][gate], rtol=1e-6, atol=1e-6)
    assert torch.equal(data, original) and torch.equal(torch.get_rng_state(), rng)


def test_p06_request_coordinates_preserved_without_sigma_or_parameter_change():
    data = values()
    request = torch.tensor(P05_CAP) * torch.linspace(-1., 1., 12)
    data[0, 207:219] = request / torch.tensor(SCALES)
    center, evidence = cap_transition_request_history(data)
    assert torch.equal(evidence["current_cap_full12"][0], torch.tensor(P06_CAP))
    torch.testing.assert_close(center[0, :10].tanh() * torch.tensor(P06_CAP)[:10], request[:10], rtol=1e-6, atol=1e-6)
    old, new = actors(data)
    before = {name: tensor.clone() for name, tensor in new.state_dict().items()}
    inputs = observation(data)
    expected = .1 * new.mlp(data)[:, 0, :] + .9 * center
    torch.testing.assert_close(new(inputs), expected, rtol=1e-6, atol=1e-6)
    old(inputs, stochastic_output=True)
    new(inputs, stochastic_output=True)
    assert torch.equal(old.output_std, new.output_std)
    assert new.exploration_std_temperature == .25
    assert old.state_dict().keys() == before.keys() == new.state_dict().keys()
    for name, tensor in before.items():
        assert torch.equal(tensor, old.state_dict()[name])
        assert torch.equal(tensor, new.state_dict()[name])
    # This preserves the history-center representation, not final physical motion.
    assert not torch.equal(old.output_mean[:, :10], new.output_mean[:, :10])


def test_repeated_same_observation_and_mixed_batch_have_no_consumed_entry_flag():
    data = torch.cat([values(2), values(), values(age_s=1 / 15), values(0, predecessor=False)])
    center, evidence = cap_transition_request_history(data)
    assert torch.equal(center, cap_transition_request_history(data)[0])
    order = torch.tensor([2, 0, 3, 1])
    shuffled, shuffled_evidence = cap_transition_request_history(data[order])
    assert torch.equal(shuffled, center[order])
    assert torch.equal(shuffled_evidence["gate_full12"], evidence["gate_full12"][order])
    _, model = actors(data)
    expected = model(observation(data))
    assert torch.equal(expected, model(observation(data)))
    torch.testing.assert_close(model(observation(data[order])), expected[order], rtol=1e-6, atol=1e-6)


def test_zero_request_and_raw_history_still_match_old_kernel_at_real_gate():
    data = values()
    data[:, 195:219] = 0.
    assert cap_transition_request_history(data)[1]["gate_full12"].any()
    old, new = actors(data)
    assert torch.equal(old(observation(data)), new(observation(data)))


@pytest.mark.parametrize("fault", ["nan", "clip", "no_stage", "two_stages", "completed", "negative_age", "request_cap", "dimension"])
def test_invalid_observation_is_rejected_without_clipping_or_fabrication(fault):
    data = values()
    if fault == "nan": data[0, 210] = float("nan")
    elif fault == "clip": data[0, 195] = 20.01
    elif fault == "no_stage": data[0, :13] = 0.
    elif fault == "two_stages": data[0, 0] = 1.
    elif fault == "completed": data[0, 162] = .5
    elif fault == "negative_age": data[0, 20] = -.001
    elif fault == "request_cap": data[0, 210] = 25. / 6.  # FR-knee prior cap is24, not112.
    elif fault == "dimension": data = data[:, :371]
    with pytest.raises(ValueError):
        cap_transition_request_history(data)
