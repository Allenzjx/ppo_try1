"""A terminal is recorded before autoreset, without moving its PPO boundary."""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rsl_rl")

from test_semantic_training import Core, make_runner
from wlr50_clean.ppo import semantic_training as training


def json_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_every_terminal_is_durable_before_reset_and_written_exactly_once(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    fsync_calls = []
    real_fsync = training.os.fsync

    def fsync(fd):
        fsync_calls.append(fd)
        return real_fsync(fd)

    monkeypatch.setattr(training.os, "fsync", fsync)

    class InspectReset(Core):
        last_sync_count = 0

        def reset(self, **kwargs):
            if self.resets:
                # These reads execute inside reset, before any next-prefix step
                # or return to the training collector can occur.
                rows = json_lines(run_dir / "residual_and_projection_audit.jsonl")
                episodes = json_lines(run_dir / "completed_episodes.jsonl")
                assert len(rows) == self.calls
                assert len(episodes) == self.resets
                assert episodes[-1]["episode_index"] == self.resets - 1
                assert episodes[-1]["policy_decisions"] == self.terminal_at
                assert rows[-1]["terminal"] is True
                assert rows[-1]["global_policy_decision"] == self.calls
                assert rows[-1]["applied_audit"] == episodes[-1]["terminal_info"]
                assert rows[-1]["terminal_observation"]["policy"][0][0] == pytest.approx(0.7)
                assert rows[-1]["old_log_probability"] == float(runner.alg.transition.actions_log_prob[0])
                assert rows[-1]["raw_policy_action_full12"] == runner.alg.transition.actions[0].tolist()
                assert len(fsync_calls) >= self.last_sync_count + 2
                self.last_sync_count = len(fsync_calls)
            return super().reset(**kwargs)

    runner, env = make_runner(InspectReset())
    result = training.train_semantic(
        runner, env, run_dir=run_dir, output_root=tmp_path / "output",
        stage="smoke", decisions=128, contract={}, seed=1001)
    rows = json_lines(run_dir / "residual_and_projection_audit.jsonl")
    episodes = json_lines(run_dir / "completed_episodes.jsonl")
    assert len(rows) == 128
    assert len(episodes) == 18
    assert sum(row["terminal"] for row in rows) == 18
    assert [row["episode_index"] for row in episodes] == list(range(18))
    assert result["actual_policy_decisions"] == 128
    assert result["ppo_updates_this_run"] == 1
    assert result["optimizer_steps_this_run"] == 20
    assert env._terminal_evidence_writer is None
    rollout = torch.load(run_dir / "rollouts/rollout_000001.pt", weights_only=False)
    for tick, row in enumerate(rows):
        assert row["raw_policy_action_full12"] == rollout["actions"][tick, 0].tolist()
        assert row["old_log_probability"] == rollout["actions_log_prob"][tick, 0].item()
        assert row["old_distribution_mean_full12"] == rollout["distribution_params"][0][tick, 0].tolist()
        assert row["old_distribution_std_full12"] == rollout["distribution_params"][1][tick, 0].tolist()
    terminal = rollout["dones"].bool()
    assert torch.equal(rollout["returns"][terminal], rollout["rewards"][terminal])


@pytest.mark.parametrize("reason", ["FALL", "BODY_COLLISION", "task_timeout"])
def test_prefix_failure_does_not_hide_terminal_or_create_resumable_rollout(tmp_path, reason):
    run_dir = tmp_path / "run"

    class FailingPrefix(Core):
        def step(self, raw):
            step = super().step(raw)
            step.info["termination_reason"] = reason
            return step

        def reset(self, **kwargs):
            if self.resets:
                rows = json_lines(run_dir / "residual_and_projection_audit.jsonl")
                episodes = json_lines(run_dir / "completed_episodes.jsonl")
                assert len(rows) == len(episodes) == 1
                assert episodes[0]["termination_reason"] == reason
                assert rows[0]["applied_audit"]["termination_reason"] == reason
                assert rows[0]["terminal_observation"]["policy"][0][0] == pytest.approx(0.1)
                raise RuntimeError("next teacher prefix failed after terminal evidence")
            return super().reset(**kwargs)

    runner, env = make_runner(FailingPrefix(terminal_at=1))
    actor_before = training.parameter_hash(runner.alg.actor)
    with pytest.raises(RuntimeError, match="next teacher prefix failed"):
        training.train_semantic(
            runner, env, run_dir=run_dir, output_root=tmp_path / "output",
            stage="smoke", decisions=128, contract={}, seed=1001)
    row = json_lines(run_dir / "residual_and_projection_audit.jsonl")[0]
    assert row["raw_policy_action_full12"] == runner.alg.transition.actions[0].tolist()
    assert row["old_log_probability"] == float(runner.alg.transition.actions_log_prob[0])
    assert row["terminal"] is True
    assert len(json_lines(run_dir / "completed_episodes.jsonl")) == 1
    failure = json.loads((run_dir / "training_failure.json").read_text())
    assert failure["completed_updates_this_run"] == 0
    assert failure["last_verified_checkpoint"] is None
    assert failure["partial_rollout_or_update_is_not_resumable"] is True
    assert training.parameter_hash(runner.alg.actor) == actor_before
    assert not (tmp_path / "output").exists()
    assert env._terminal_evidence_writer is None


def test_sink_failure_prevents_prefix_and_restores_scoped_writer():
    env = training.SemanticRslAdapter(Core(terminal_at=1), seed=1001, device="cpu")

    def fail_write(summary, final_observation, reward):
        assert env.core.resets == 1
        assert final_observation["policy"][0, 0].item() == pytest.approx(0.1)
        raise OSError("terminal evidence disk failure")

    with pytest.raises(OSError, match="terminal evidence disk failure"):
        with training._terminal_evidence_before_reset(env, fail_write):
            env.step(torch.zeros((1, 12)))
    assert env.core.resets == 1
    assert env.total_decisions == 1
    assert len(env.completed_episodes) == 1
    assert env._terminal_evidence_writer is None


def test_sink_does_not_change_reset_observation_terminal_observation_or_timeout():
    env = training.SemanticRslAdapter(Core(terminal_at=1), seed=1001, device="cpu")
    seen = []

    def capture(summary, final_observation, reward):
        assert env.core.resets == 1
        seen.append((summary, final_observation, reward.clone()))

    with training._terminal_evidence_before_reset(env, capture):
        obs, reward, done, extras = env.step(torch.zeros((1, 12)))
    assert len(seen) == 1
    assert extras["terminal_evidence_persisted_before_reset"] is True
    assert extras["episode_summaries"] == [seen[0][0]]
    assert extras["terminal_observation"] is seen[0][1]
    assert torch.equal(reward, seen[0][2])
    assert done.item() is True
    assert extras["time_outs"].item() is False
    assert obs["policy"][0, 0].item() == 0.0
    assert extras["terminal_observation"]["policy"][0, 0].item() == pytest.approx(0.1)
    assert env.core.resets == 2
    assert env._terminal_evidence_writer is None


def test_nonterminal_phase_handoff_does_not_invoke_terminal_writer():
    core = Core(terminal_at=3)
    original_step = core.step

    def handoff(raw):
        step = original_step(raw)
        step.info["phase_id"] = "P08"
        step.info["phase_after_step"] = "P09"
        return step

    core.step = handoff
    env = training.SemanticRslAdapter(core, seed=1001, device="cpu")

    def unexpected(*args):
        pytest.fail("ordinary phase handoff invoked terminal writer")

    with training._terminal_evidence_before_reset(env, unexpected):
        obs, reward, done, extras = env.step(torch.zeros((1, 12)))
    assert done.item() is False
    assert extras["time_outs"].item() is False
    assert extras["episode_summaries"] == []
    assert "terminal_observation" not in extras
    assert core.resets == 1
    assert env.total_decisions == 1
