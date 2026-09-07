"""CPU checks of the installed official return estimator, not a replacement GAE.

These isolate PPO's return boundary. They do not construct Isaac, optimize a
network, load checkpoints, or claim to test the environment's phase producer.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rsl_rl")
from rsl_rl.algorithms.ppo import PPO
from tensordict import TensorDict

from wlr50_clean.ppo.semantic_training import semantic_runner_config


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


class SuccessorCritic:
    """A value depending on the supplied successor, with a call receipt."""

    def __init__(self):
        self.observations = []

    def __call__(self, observations):
        self.observations.append(observations)
        x = observations["critic"]
        assert x.device.type == "cpu"
        return x[:, :1] + 2.0 * x[:, 1:2]


def official_returns(rewards, values, dones, *, successor=(1.0, 2.0), lam=None):
    """Supply only the real compute_returns ABI; no optimizer is constructed."""
    config = semantic_runner_config(seed=1001, device="cpu", semantic_version="v3")
    algorithm = object.__new__(PPO)
    algorithm.gamma = config["algorithm"]["gamma"]
    algorithm.lam = config["algorithm"]["lam"] if lam is None else lam
    algorithm.normalize_advantage_per_mini_batch = config["algorithm"]["normalize_advantage_per_mini_batch"]
    assert len(rewards) == len(values) == len(dones) and len(rewards) >= 2
    tensor = lambda rows: torch.tensor(rows, dtype=torch.float32, device="cpu").reshape(-1, 1, 1)
    storage = SimpleNamespace(
        num_transitions_per_env=len(rewards),
        rewards=tensor(rewards),
        values=tensor(values),
        dones=torch.tensor(dones, dtype=torch.bool, device="cpu").reshape(-1, 1, 1),
        returns=torch.zeros((len(rewards), 1, 1), dtype=torch.float32, device="cpu"),
    )
    algorithm.storage = storage
    algorithm.critic = SuccessorCritic()
    observations = TensorDict(
        {"critic": torch.tensor([successor], dtype=torch.float32, device="cpu")},
        batch_size=[1], device="cpu",
    )
    PPO.compute_returns(algorithm, observations)
    assert len(algorithm.critic.observations) == 1
    assert algorithm.critic.observations[0] is observations
    assert storage.returns.device.type == storage.advantages.device.type == "cpu"
    assert torch.isfinite(storage.returns).all()
    assert torch.isfinite(storage.advantages).all()
    return storage


def assert_returns(storage, expected):
    torch.testing.assert_close(
        storage.returns[:, 0, 0],
        torch.tensor(expected, dtype=torch.float32, device="cpu"),
        rtol=3e-6, atol=3e-6,
    )


@pytest.mark.parametrize("version,gamma,lam", [("v3", .9985, .99), ("v2", .995, .95)])
def test_versioned_production_parameters_keep_128_steps(version, gamma, lam):
    config = semantic_runner_config(seed=1001, device="cpu", semantic_version=version)
    assert config["algorithm"]["gamma"] == gamma
    assert config["algorithm"]["lam"] == lam
    assert config["num_steps_per_env"] == 128
    assert config["algorithm"]["normalize_advantage_per_mini_batch"] is False


def test_three_step_finite_gae_matches_expanded_hand_calculation():
    config = semantic_runner_config(seed=1001, device="cpu", semantic_version="v3")
    g, lam = config["algorithm"]["gamma"], config["algorithm"]["lam"]
    trace = g * lam
    # Successor critic returns 1 + 2*2 = 5, not the last stored value .7.
    d0, d1, d2 = .25 + g * -.2 - .4, -.1 + g * .7 + .2, .3 + g * 5 - .7
    expected = [.4 + d0 + trace * d1 + trace**2 * d2,
                -.2 + d1 + trace * d2, .7 + d2]
    storage = official_returns([.25, -.1, .3], [.4, -.2, .7], [False] * 3)
    assert_returns(storage, expected)
    raw_advantage = storage.returns - storage.values
    expected_normalized = ((raw_advantage - raw_advantage.mean())
                           / (raw_advantage.std() + 1e-8))
    torch.testing.assert_close(storage.advantages, expected_normalized)


@pytest.mark.parametrize("terminal_event", [-40.0, 40.0])
def test_true_terminal_does_not_import_reset_episode_value(terminal_event):
    config = semantic_runner_config(seed=1001, device="cpu", semantic_version="v3")
    g, lam = config["algorithm"]["gamma"], config["algorithm"]["lam"]
    first = official_returns([.25, terminal_event, 123.0], [.4, -.2, 50_000.0],
                             [False, True, False], successor=(100_000.0, 2.0))
    second = official_returns([.25, terminal_event, -999.0], [.4, -.2, -7_000.0],
                              [False, True, False], successor=(-20_000.0, 3.0))
    expected_before = .25 + g * -.2 + g * lam * (terminal_event + .2)
    for storage in (first, second):
        assert storage.returns[1, 0, 0].item() == terminal_event
        assert storage.returns[0, 0, 0].item() == pytest.approx(expected_before, abs=3e-6)
    torch.testing.assert_close(first.returns[:2], second.returns[:2], rtol=0, atol=0)
    # Whole-batch normalized advantages may change with reset-episode statistics;
    # that is not value bootstrap or TD-trace leakage across done.
    assert first.returns[2, 0, 0] != second.returns[2, 0, 0]


def test_nonterminal_128_boundary_uses_actual_successor_critic():
    config = semantic_runner_config(seed=1001, device="cpu", semantic_version="v3")
    g, lam = config["algorithm"]["gamma"], config["algorithm"]["lam"]
    trace = g * lam
    first = official_returns([0.0] * 128, [0.0] * 128, [False] * 128, successor=(2.0, 3.0))
    second = official_returns([0.0] * 128, [0.0] * 128, [False] * 128, successor=(5.0, 3.0))
    # Only delta_127 is nonzero; the expected trace is a geometric closed form.
    assert_returns(first, [g * 8.0 * trace**(127 - index) for index in range(128)])
    assert_returns(second, [g * 11.0 * trace**(127 - index) for index in range(128)])
    assert first.returns[0, 0, 0] > 0
    assert second.returns[-1, 0, 0].item() == pytest.approx(g * 11, abs=3e-6)


def test_true_terminal_at_128_boundary_masks_successor_value():
    config = semantic_runner_config(seed=1001, device="cpu", semantic_version="v3")
    trace = config["algorithm"]["gamma"] * config["algorithm"]["lam"]
    storage = official_returns([0.0] * 127 + [-40.0], [0.0] * 128,
                               [False] * 127 + [True], successor=(1_000_000.0, 2.0))
    assert_returns(storage, [-40.0 * trace**(127 - index) for index in range(128)])
    assert storage.returns[-1, 0, 0].item() == -40.0


def test_ordinary_phase_labels_without_done_do_not_cut_return_trace():
    config = semantic_runner_config(seed=1001, device="cpu", semantic_version="v3")
    trace = config["algorithm"]["gamma"] * config["algorithm"]["lam"]
    # Labels identify the scenario only. Official GAE has no phase argument:
    # P07 -> P08 -> P09 remains one trace when the producer supplies done=False.
    phases = ("P07", "P08", "P09")
    dones = [False] * len(phases)
    storage = official_returns([0.0, 0.0, 1.0], [0.0] * 3, dones, successor=(0.0, 0.0))
    assert_returns(storage, [trace**2, trace, 1.0])
    assert storage.returns[0, 0, 0] > 0
    # Counterexample proves the test observes the actual done-dependent boundary.
    cut = official_returns([0.0, 0.0, 1.0], [0.0] * 3, [True, False, False], successor=(0.0, 0.0))
    assert cut.returns[0, 0, 0].item() == 0.0


def test_lambda_changes_trace_not_one_step_gamma_bootstrap():
    config = semantic_runner_config(seed=1001, device="cpu", semantic_version="v3")
    g, lam = config["algorithm"]["gamma"], config["algorithm"]["lam"]
    actual = official_returns([0.0, 0.0], [0.0, 0.0], [False, False], successor=(3.0, 0.0))
    counterfactual = official_returns([0.0, 0.0], [0.0, 0.0], [False, False],
                                      successor=(3.0, 0.0), lam=.95)
    assert_returns(actual, [g * lam * (g * 3), g * 3])
    assert_returns(counterfactual, [g * .95 * (g * 3), g * 3])
    assert actual.returns[-1, 0, 0].item() == counterfactual.returns[-1, 0, 0].item()
    assert actual.returns[0, 0, 0] > counterfactual.returns[0, 0, 0]
