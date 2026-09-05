from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rsl_rl")

from wlr50_clean.ppo.semantic_training import (
    SemanticRslAdapter, construct_semantic_runner, train_semantic,
    load_semantic_checkpoint, save_semantic_checkpoint, seed_training_rngs,
    state_hash, parameter_hash,
)


class Core:
    """Strict core ABI; failures are genuine terminals, not bad transitions."""
    def __init__(self, *, terminal_at=7, fail_at=None):
        self.terminal_at, self.fail_at = terminal_at, fail_at
        self.resets, self.calls = 0, 0
        self.frame = SimpleNamespace(sim_time_s=0.0)

    def reset(self, *, seed=1001, options=None):
        self.resets += 1
        self.tick = 0
        self.done = False
        self.frame.sim_time_s = 0.0
        return (0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

    def step(self, raw):
        assert isinstance(raw, tuple) and len(raw) == 12
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("injected sensor interface failure")
        self.tick += 1
        self.frame.sim_time_s = self.tick / 15
        self.done = self.tick == self.terminal_at
        observation = (self.tick / 10, 1.0, -0.2, 0.3, 0.0, 0.2, 0.0, 1 - self.tick / 3000)
        return SimpleNamespace(
            observation=observation, reward=1.0 + raw[0] - raw[1] ** 2,
            terminated=self.done, truncated=False,
            info={"phase_id": "P01", "raw_policy_action_full12": raw,
                  "applied_action_full12": tuple(0.1 * value for value in raw),
                  "actuator_target_effect_audit": native_audit(raw),
                  "termination_reason": "task_timeout" if self.done else None,
                  "task_success": False},
        )

    def telemetry_summary(self):
        return {"calls": self.calls, "resets": self.resets}


def native_audit(raw):
    return {"schema": "wlr50_clean.actuator_target_effect_audit.v1", "verified": True,
            "actual_mapping_matches_dispatch": True, "setter_dispatch_targets_equal": True,
            "same_tick_counterfactual": True, "raw_policy_action_full12": raw,
            "target_dtype": "torch.float32", "changed_target_channel_count": int(any(raw))}


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_runner(core=None):
    seed_training_rngs(1001)
    env = SemanticRslAdapter(core or Core(), seed=1001, device="cpu")
    runner, _ = construct_semantic_runner(env, seed=1001, device="cpu")
    return runner, env


def test_real_official_update_failed_episodes_and_actual_checkpoint_roundtrip(tmp_path):
    runner, env = make_runner()
    result = train_semantic(runner, env, run_dir=tmp_path / "run", output_root=tmp_path / "output",
                            stage="smoke", decisions=128, contract={"revision": "test"}, seed=1001)
    assert result["actual_policy_decisions"] == 128
    assert result["ppo_updates_this_run"] == 1
    assert result["optimizer_steps_this_run"] == 20
    assert result["actor_parameter_sha256_before"] != result["actor_parameter_sha256_after"]
    assert result["finite_nonzero_gradient_observed"] is True
    assert result["telemetry"]["completed_episode_count"] == 18
    assert result["telemetry"]["success_count"] == 0
    rows = [json.loads(line) for line in (tmp_path / "run/residual_and_projection_audit.jsonl").read_text().splitlines()]
    rollout = torch.load(tmp_path / "run/rollouts/rollout_000001.pt", weights_only=False)
    for tick, row in enumerate(rows):
        assert row["raw_policy_action_full12"] == rollout["actions"][tick, 0].tolist()
        assert row["raw_policy_action_full12"] != row["applied_audit"]["applied_action_full12"]
        assert row["old_log_probability"] == rollout["actions_log_prob"][tick, 0].item()
    # A task-terminal reward is not contaminated by reset-state value bootstrap.
    terminal = rollout["dones"].bool()
    assert torch.equal(rollout["returns"][terminal], rollout["rewards"][terminal])
    update = json.loads((tmp_path / "run/optimizer_updates.jsonl").read_text())
    for key in ("kl_mean", "clip_fraction", "entropy", "value_loss", "gradient_norm_max"):
        assert torch.isfinite(torch.tensor(update[key]))
    checkpoint = tmp_path / "output/checkpoints/history/checkpoint_step_000000128.pt"
    original_optimizer = state_hash(runner.alg.optimizer.state_dict())
    original_actor = parameter_hash(runner.alg.actor)
    # The checkpoint captured this exact global RNG state, independent of new runner initialization.
    expected_sample = torch.rand(8)
    other, _ = make_runner()
    infos = load_semantic_checkpoint(other, checkpoint, contract={"revision": "test"}, seed=1001)
    assert parameter_hash(other.alg.actor) == original_actor
    assert state_hash(other.alg.optimizer.state_dict()) == original_optimizer
    assert torch.equal(torch.rand(8), expected_sample)
    assert infos["optimizer_steps"] == 20
    assert infos["physical_env_state_saved"] is False


def test_adapter_preserves_final_observation_and_disables_task_timeout_bootstrap():
    env = SemanticRslAdapter(Core(terminal_at=1), seed=1001, device="cpu")
    obs, reward, done, extras = env.step(torch.zeros((1, 12)))
    assert done.item() is True and extras["time_outs"].item() is False
    assert extras["terminal_observation"]["policy"][0, 0].item() == pytest.approx(0.1)
    assert obs["policy"][0, 0].item() == 0.0
    assert extras["semantic_decisions"][0]["raw_policy_action_full12"] == [0.0] * 12


def test_failed_partial_rollout_never_updates_or_gets_checkpoint(tmp_path):
    runner, env = make_runner(Core(fail_at=2))
    before = parameter_hash(runner.alg.actor)
    with pytest.raises(RuntimeError, match="sensor interface failure"):
        train_semantic(runner, env, run_dir=tmp_path / "run", output_root=tmp_path / "output",
                       stage="smoke", decisions=128, contract={}, seed=1001)
    failure = json.loads((tmp_path / "run/training_failure.json").read_text())
    assert failure["completed_updates_this_run"] == 0
    assert failure["last_verified_checkpoint"] is None
    assert parameter_hash(runner.alg.actor) == before
    assert not (tmp_path / "output").exists()


def test_checkpoint_refuses_overwrite_and_runtime_mismatch(tmp_path):
    runner, _ = make_runner()
    path = tmp_path / "initial.pt"
    save_semantic_checkpoint(runner, path, {"seed": 1001, "runtime_contract": {"v": 1}})
    with pytest.raises(FileExistsError):
        save_semantic_checkpoint(runner, path, {"seed": 1001})
    with pytest.raises(RuntimeError, match="contract mismatch"):
        load_semantic_checkpoint(runner, path, contract={"v": 2}, seed=1001)


def test_remaining_budget_is_checked_before_any_rollout(tmp_path):
    runner, env = make_runner()
    with pytest.raises(ValueError, match="remaining"):
        train_semantic(runner, env, run_dir=tmp_path / "run", output_root=tmp_path / "output",
                       stage="smoke", decisions=129, contract={}, seed=1001,
                       resume_infos={"stage_requested_decisions": {"smoke": 9872}})
    assert env.core.calls == 0


def test_actual_semantic_core_observation_reward_projection_to_official_rsl(tmp_path):
    from test_semantic_observation_reward_env import Backend
    from wlr50_clean.ppo.semantic_env import SemanticEpisodeEnv
    class AuditedBackend(Backend):
        def step_physics(self, applied_action_full12):
            frame = super().step_physics(applied_action_full12)
            frame.info["actuator_target_effect_audit"] = native_audit(self.requests[-1][1])
            return frame
    core = SemanticEpisodeEnv(AuditedBackend(terminal_tick=56, reason="FALL"), collect_trace=False)
    runner, env = make_runner(core)
    assert env.observation_dimension == core.observation_schema.dimension
    result = train_semantic(runner, env, run_dir=tmp_path / "real_core", output_root=tmp_path / "output",
                            stage="smoke", decisions=128, contract={}, seed=1001)
    assert result["optimizer_steps_this_run"] == 20
    assert result["telemetry"]["core"]["terminations"]["FALL"] == 18
    assert result["actor_parameter_sha256_before"] != result["actor_parameter_sha256_after"]


def test_explicit_resume_preserves_counts_and_never_reinitializes_actor(tmp_path):
    runner, env = make_runner()
    output = tmp_path / "output"
    train_semantic(runner, env, run_dir=tmp_path / "first", output_root=output,
                   stage="smoke", decisions=128, contract={}, seed=1001)
    first = output / "checkpoints/history/checkpoint_step_000000128.pt"
    resumed_runner, resumed_env = make_runner()
    previous = load_semantic_checkpoint(resumed_runner, first, contract={}, seed=1001)
    before = parameter_hash(resumed_runner.alg.actor)
    result = train_semantic(resumed_runner, resumed_env, run_dir=tmp_path / "second", output_root=output,
                            stage="smoke", decisions=128, contract={}, seed=1001, resume_infos=previous)
    manifest = json.loads((output / "checkpoints/history/checkpoint_step_000000256_manifest.json").read_text())
    assert result["actor_parameter_sha256_before"] == before
    assert manifest["global_policy_decisions"] == 256 and manifest["ppo_updates"] == 2
    assert manifest["optimizer_steps"] == 40
    assert manifest["stage_requested_decisions"]["smoke"] == 256


def test_invalid_native_audit_stops_before_storing_transition():
    core = Core()
    original = core.step
    def corrupt(raw):
        step = original(raw)
        step.info["actuator_target_effect_audit"] = None
        return step
    core.step = corrupt
    env = SemanticRslAdapter(core, seed=1001, device="cpu")
    with pytest.raises(RuntimeError, match="native target audit"):
        env.step(torch.zeros((1, 12)))
