"""Official CPU PPO regressions with synthetic core observations; no Isaac claim."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rsl_rl")

from test_semantic_training import Core
from wlr50_clean.ppo import semantic_training as training
from wlr50_clean.ppo.semantic_policy_distribution import HISTORY_POLICY
from wlr50_clean.ppo.semantic_transfer_roles import ROLE_OBSERVATION_LAYOUT


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    yield
    torch.set_num_threads(previous)


class HistoryCore(Core):
    """Finite explicit 372 observations; reset is deterministic and RNG-neutral."""

    def __init__(self, *, terminal_at=128):
        super().__init__(terminal_at=terminal_at)
        self.before_reset = None

    def reset(self, **kwargs):
        if self.resets and self.before_reset is not None:
            self.before_reset()
        return super().reset(**kwargs) + (0.,) * 364

    def step(self, raw):
        step = super().step(raw)
        values = list(step.observation) + [0.] * 364
        values[195:207] = [max(-20., min(20., value)) for value in raw]
        step.observation = tuple(values)
        # A synthetic partial terminal interval verifies metadata preservation,
        # not a claim to have simulated these physical ticks.
        step.info["physics_ticks"] = 3 if step.terminated else 8
        if step.terminated:
            self.frame.sim_time_s -= 5 / 120
            step.info["termination_reason"] = "HARD_JOINT_LIMIT"
        return step


def make(core=None):
    training.seed_training_rngs(1001)
    env = training.SemanticRslAdapter(core or HistoryCore(), seed=1001, device="cpu")
    env.cfg["semantic_version"] = "v3"
    runner, _ = training.construct_semantic_runner(
        env, seed=1001, device="cpu", policy_version=HISTORY_POLICY,
        observation_layout=ROLE_OBSERVATION_LAYOUT)
    return runner, env


def run(runner, env, directory, *, decisions=128):
    return training.train_semantic(
        runner, env, run_dir=directory / "run", output_root=directory / "output",
        stage="full_episode", decisions=decisions, contract={"tail_reset_cpu": True},
        seed=1001, checkpoint_interval_updates=10)


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def rollout(directory, update=1):
    return torch.load(directory / f"run/rollouts/rollout_{update:06d}.pt", weights_only=False)


def checkpoint(directory, step):
    return directory / f"output/checkpoints/history/checkpoint_step_{step:09d}.pt"


def assert_terminal_return_without_bootstrap(data, index):
    reward = data["rewards"][index]
    value = data["values"][index]
    # Match the installed official float32 operation order, not an artificial
    # exact r copy: fl(fl(r - V) + V), with no terminal next-value contribution.
    expected = (reward - value) + value
    assert torch.equal(data["returns"][index], expected)
    eps = torch.finfo(torch.float32).eps
    tolerance = eps / (1 - eps) * (reward.double().abs() + 2 * value.double().abs())
    assert torch.all((data["returns"][index].double() - reward.double()).abs() <= tolerance)


def test_deferred_adapter_returns_true_finite_terminal_with_false_timeout_and_no_hidden_reset():
    core = HistoryCore(terminal_at=1)
    _, env = make(core)
    captured = []
    with training._terminal_evidence_before_reset(
            env, lambda *args: captured.append(args), defer_terminal_reset=True):
        obs, reward, done, extras = env.step(torch.zeros(1, 12))
    assert done.item() is True and extras["time_outs"].item() is False
    assert extras["terminal_reset_deferred"] is True
    assert extras["terminal_evidence_persisted_before_reset"] is True
    assert len(captured) == 1 and core.resets == 1 and core.calls == 1
    assert captured[0][1] is extras["terminal_observation"]
    assert torch.equal(obs["policy"], extras["terminal_observation"]["policy"])
    assert bool(torch.isfinite(obs["policy"]).all())
    assert torch.equal(reward, captured[0][2])
    assert env.get_observations()["policy"][0, 0].item() == pytest.approx(.1)
    assert core.resets == 1 and env.episode_reset_pending
    assert env._terminal_evidence_writer is None and env._defer_terminal_reset is False


def test_last_terminal_saves_without_unused_reset_and_keeps_real_terminal_obs(tmp_path):
    core = HistoryCore()
    core.before_reset = lambda: pytest.fail("unused reset/prefix after final decision")
    runner, env = make(core)
    result = run(runner, env, tmp_path)
    assert result["actual_policy_decisions"] == 128
    assert result["ppo_updates_this_run"] == 1 and result["optimizer_steps_this_run"] == 20
    assert core.resets == 1 and core.calls == 128
    assert env.episode_reset_pending is True
    assert env._terminal_evidence_writer is None and env._defer_terminal_reset is False
    audit = rows(tmp_path / "run/residual_and_projection_audit.jsonl")
    episodes = rows(tmp_path / "run/completed_episodes.jsonl")
    assert len(audit) == 128 and len(episodes) == 1
    assert audit[-1]["terminal"] is True
    assert audit[-1]["applied_audit"]["physics_ticks"] == 3
    assert episodes[0]["termination_reason"] == "HARD_JOINT_LIMIT"
    final = env.get_observations()
    assert final["policy"].shape == (1, 372) and bool(torch.isfinite(final["policy"]).all())
    assert final["policy"][0, 0].item() == pytest.approx(12.8)
    assert final["policy"].tolist() == audit[-1]["terminal_observation"]["policy"]
    env.get_observations()  # Read-only even with a pending reset.
    assert core.resets == 1
    with pytest.raises(RuntimeError, match="pending terminal reset"):
        env.step(torch.zeros(1, 12))
    assert core.calls == 128
    data = rollout(tmp_path)
    for index, row in enumerate(audit):
        assert row["raw_policy_action_full12"] == data["actions"][index, 0].tolist()
        assert row["old_log_probability"] == data["actions_log_prob"][index, 0].item()
        assert row["old_distribution_mean_full12"] == data["distribution_params"][0][index, 0].tolist()
        assert row["old_distribution_std_full12"] == data["distribution_params"][1][index, 0].tolist()
    assert_terminal_return_without_bootstrap(data, 127)
    manifest = json.loads(checkpoint(tmp_path, 128).with_name("checkpoint_step_000000128_manifest.json").read_text())
    assert manifest["save_load_round_trip"] is True
    assert manifest["physical_env_state_saved"] is False


def test_next_rollout_resets_after_saved_update_before_sampling_and_uses_new_history(tmp_path):
    core = HistoryCore()
    runner, env = make(core)

    def before_reset():
        assert core.calls == 128
        assert checkpoint(tmp_path, 128).is_file()
        assert runner.alg.storage.step == 0
        assert env._terminal_evidence_writer is None and env._defer_terminal_reset is False

    core.before_reset = before_reset
    result = run(runner, env, tmp_path, decisions=256)
    assert result["ppo_updates_this_run"] == 2
    assert core.calls == 256 and core.resets == 2
    assert env.episode_reset_pending
    second = rollout(tmp_path, 2)
    assert second["observations"]["policy"][0, 0, 0].item() == 0.
    assert torch.equal(second["observations"]["policy"][0, 0, 195:207], torch.zeros(12))
    assert_terminal_return_without_bootstrap(second, 127)


@pytest.mark.parametrize("fault", ["reset_failure", "curriculum_mutation"])
def test_failed_next_reset_preserves_completed_update_and_checkpoint_even_off_cadence(tmp_path, fault):
    core = HistoryCore(terminal_at=256)
    runner, env = make(core)
    calls_at_act = []
    actual_act = runner.alg.act

    def act(obs):
        calls_at_act.append(core.calls)
        return actual_act(obs)

    runner.alg.act = act

    def before_reset():
        # Update2 is not the default cadence10, first, or last planned update.
        # A deferred terminal forces a verified checkpoint before any new reset.
        assert checkpoint(tmp_path, 256).is_file()
        assert runner.alg.storage.step == 0
        assert env._terminal_evidence_writer is None and env._defer_terminal_reset is False
        if fault == "reset_failure":
            raise RuntimeError("synthetic next prefix failure")
        env.cfg["reset_sampling"] = "unexpected_source_mutation"

    core.before_reset = before_reset
    message = "next prefix failure" if fault == "reset_failure" else "curriculum changed during deferred"
    with pytest.raises(RuntimeError, match=message):
        run(runner, env, tmp_path, decisions=384)
    assert core.calls == 256 and len(calls_at_act) == 256
    assert runner.alg.storage.step == 0
    assert checkpoint(tmp_path, 256).is_file()
    failure = json.loads((tmp_path / "run/training_failure.json").read_text())
    assert failure["completed_updates_this_run"] == 2
    assert failure["last_verified_checkpoint"] is not None
    assert env._terminal_evidence_writer is None and env._defer_terminal_reset is False
    assert len(rows(tmp_path / "run/residual_and_projection_audit.jsonl")) == 256


def test_middle_terminal_still_resets_immediately_and_final_nonterminal_does_not(tmp_path):
    core = HistoryCore(terminal_at=127)
    runner, env = make(core)
    run(runner, env, tmp_path)
    data = rollout(tmp_path)
    assert core.resets == 2 and core.calls == 128
    assert not env.episode_reset_pending
    assert data["dones"][126].item() == 1 and data["dones"][127].item() == 0
    assert data["observations"]["policy"][127, 0, 0].item() == 0.
    assert_terminal_return_without_bootstrap(data, 126)


def test_nonterminal_last_tick_phase_change_does_not_reset_or_cut_gae(tmp_path):
    core = HistoryCore(terminal_at=999)
    original = core.step

    def step(raw):
        result = original(raw)
        result.info.update(phase_id="P08", end_phase_id="P09")
        return result

    core.step = step
    runner, env = make(core)
    run(runner, env, tmp_path)
    assert core.resets == 1 and not env.episode_reset_pending
    assert not bool(rollout(tmp_path)["dones"].any())


def test_stop_after_complete_update_does_not_consume_pending_reset(tmp_path, monkeypatch):
    core = HistoryCore()
    core.before_reset = lambda: pytest.fail("reset after an accepted complete-update stop")
    runner, env = make(core)
    monkeypatch.setattr(training, "_stop_request", lambda *args: {"reason": "test complete-update stop"})
    result = run(runner, env, tmp_path, decisions=256)
    assert result["lifecycle"] == "STOPPED_AT_VERIFIED_UPDATE_BOUNDARY"
    assert result["actual_policy_decisions"] == 128
    assert result["unconsumed_requested_policy_decisions"] == 128
    assert core.resets == 1 and env.episode_reset_pending
    assert checkpoint(tmp_path, 128).is_file()


def test_sink_failure_on_final_tick_restores_scope_and_prevents_reset_update_save(tmp_path, monkeypatch):
    core = HistoryCore()
    runner, env = make(core)
    actor = training.parameter_hash(runner.alg.actor)

    def fail_fsync(fd):
        raise OSError("synthetic terminal fsync failure")

    monkeypatch.setattr(training.os, "fsync", fail_fsync)
    with pytest.raises(OSError, match="terminal fsync failure"):
        run(runner, env, tmp_path)
    assert core.calls == 128 and core.resets == 1
    assert env._terminal_evidence_writer is None and env._defer_terminal_reset is False
    assert training.parameter_hash(runner.alg.actor) == actor
    assert not checkpoint(tmp_path, 128).exists()


@pytest.mark.parametrize("fault", ["normalization", "normalizer_type", "recurrent", "rnd", "vector"])
def test_unsupported_stateful_or_vector_path_keeps_original_autoreset(fault, monkeypatch):
    runner, env = make()
    assert training._supports_deferred_terminal_reset(runner, env)
    if fault == "normalization":
        monkeypatch.setattr(runner.alg.critic, "obs_normalization", True)
    elif fault == "normalizer_type":
        monkeypatch.setattr(runner.alg.actor, "obs_normalizer", torch.nn.LayerNorm(372))
    elif fault == "recurrent":
        monkeypatch.setattr(runner.alg.actor, "is_recurrent", True)
    elif fault == "rnd":
        monkeypatch.setattr(runner.alg, "rnd", object())
    else:
        monkeypatch.setattr(env, "num_envs", 8)
    assert not training._supports_deferred_terminal_reset(runner, env)


def test_identity_HISTORY372_official_rollouts_and_updates_equal_eager_for_rng_neutral_reset(tmp_path, monkeypatch):
    support = training._supports_deferred_terminal_reset
    before, eager_env = make()
    monkeypatch.setattr(training, "_supports_deferred_terminal_reset", lambda *args: False)
    run(before, eager_env, tmp_path / "eager", decisions=256)
    eager_rng = torch.get_rng_state().clone()
    after, deferred_env = make()
    monkeypatch.setattr(training, "_supports_deferred_terminal_reset", support)
    run(after, deferred_env, tmp_path / "deferred", decisions=256)
    assert eager_env.core.resets == 3 and deferred_env.core.resets == 2
    assert torch.equal(torch.get_rng_state(), eager_rng)
    for update in (1, 2):
        a, b = rollout(tmp_path / "eager", update), rollout(tmp_path / "deferred", update)
        for key in ("actions", "actions_log_prob", "values", "rewards", "dones", "returns", "advantages"):
            assert torch.equal(a[key], b[key]), (update, key)
        assert training.state_hash(a["observations"]) == training.state_hash(b["observations"])
        assert training.state_hash(a["distribution_params"]) == training.state_hash(b["distribution_params"])
    for role in ("actor", "critic"):
        assert training.parameter_hash(getattr(before.alg, role)) == training.parameter_hash(getattr(after.alg, role))
    assert training.state_hash(before.alg.optimizer.state_dict()) == training.state_hash(after.alg.optimizer.state_dict())
    assert training.state_hash(training._normalizers(before)) == training.state_hash(training._normalizers(after))
    assert before.alg.learning_rate == after.alg.learning_rate
