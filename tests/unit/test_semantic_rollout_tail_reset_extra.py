"""Official CPU PPO regressions with synthetic core observations; no Isaac claim."""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rsl_rl")

from test_semantic_rollout_tail_reset import (
    HistoryCore,
    assert_terminal_return_without_bootstrap,
    checkpoint,
    cpu_only,  # Imported autouse CPU fixture; no CUDA-dependent test path.
    make,
    rollout,
    rows,
    run,
)
from wlr50_clean.ppo import semantic_training as training


@pytest.mark.parametrize("fault", ["nan", "wrong_dimension"])
def test_invalid_pending_reset_observation_preserves_checkpoint_without_new_act(
    tmp_path, monkeypatch, fault
):
    core = HistoryCore(terminal_at=128)
    runner, env = make(core)
    original_reset = core.reset
    original_act = runner.alg.act
    act_calls = []

    def act(obs):
        act_calls.append(core.calls)
        return original_act(obs)

    def invalid_reset(**kwargs):
        # Only the deferred reset after the first complete update reaches here.
        assert core.calls == 128
        assert checkpoint(tmp_path, 128).is_file()
        assert runner.alg.storage.step == 0
        values = list(original_reset(**kwargs))
        if fault == "nan":
            values[0] = float("nan")
        else:
            values.pop()
        return tuple(values)

    monkeypatch.setattr(runner.alg, "act", act)
    monkeypatch.setattr(core, "reset", invalid_reset)
    with pytest.raises(RuntimeError, match="semantic observation is non-finite or changed dimension"):
        run(runner, env, tmp_path, decisions=256)

    assert core.calls == 128 and len(act_calls) == 128
    assert core.resets == 2  # Physical reset returned; its observation was rejected.
    assert runner.alg.storage.step == 0
    assert env.episode_reset_pending is True
    assert env._terminal_evidence_writer is None and env._defer_terminal_reset is False
    assert checkpoint(tmp_path, 128).is_file()
    assert not checkpoint(tmp_path, 256).exists()
    assert len(rows(tmp_path / "run/residual_and_projection_audit.jsonl")) == 128
    assert len(rows(tmp_path / "run/optimizer_updates.jsonl")) == 1
    failure = json.loads((tmp_path / "run/training_failure.json").read_text())
    assert failure["completed_updates_this_run"] == 1
    assert failure["last_verified_checkpoint"] is not None
    manifest = json.loads(
        checkpoint(tmp_path, 128).with_name("checkpoint_step_000000128_manifest.json").read_text()
    )
    assert training.parameter_hash(runner.alg.actor) == manifest["actor_parameter_sha256"]
    assert training.state_hash(runner.alg.optimizer.state_dict()) == manifest["optimizer_state_sha256"]


def test_forced_capability_false_really_eager_resets_final_terminal(tmp_path, monkeypatch):
    core = HistoryCore(terminal_at=128)
    runner, env = make(core)
    # Keep the real supported current actor/critic. Do not spoof RNN attributes
    # and accidentally change the official algorithm's storage/update route.
    assert training._supports_deferred_terminal_reset(runner, env)
    monkeypatch.setattr(training, "_supports_deferred_terminal_reset", lambda *args: False)
    reset_points = []

    def before_reset():
        reset_points.append((core.calls, runner.alg.storage.step))
        # Eager reset is inside env.step, before process_env_step stores the tail
        # transition and before the ensuing complete update/checkpoint.
        assert core.calls == 128 and runner.alg.storage.step == 127
        assert not checkpoint(tmp_path, 128).exists()
        assert env._defer_terminal_reset is False

    core.before_reset = before_reset
    result = run(runner, env, tmp_path, decisions=128)
    assert reset_points == [(128, 127)]
    assert core.calls == 128 and core.resets == 2
    assert not env.episode_reset_pending
    assert env._terminal_evidence_writer is None and env._defer_terminal_reset is False
    assert env.get_observations()["policy"][0, 0].item() == 0.0
    assert result["actual_policy_decisions"] == 128 and result["ppo_updates_this_run"] == 1
    audit = rows(tmp_path / "run/residual_and_projection_audit.jsonl")
    assert len(audit) == 128 and audit[-1]["terminal"] is True
    # Eager reset still preserves the separate real terminal tensor in the audit.
    assert audit[-1]["terminal_observation"]["policy"][0][0] == pytest.approx(12.8)
    assert_terminal_return_without_bootstrap(rollout(tmp_path), 127)
    assert checkpoint(tmp_path, 128).is_file()
