"""Real official PPO loop with fake physical core; no Isaac/physics claims."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rsl_rl")

from test_semantic_checkpoint_prefix import Core, Request, adapter, provenance
from wlr50_clean.ppo import semantic_training as training
from wlr50_clean.ppo.semantic_checkpoint_prefix_policy import build_frozen_checkpoint_prefix_policy
from wlr50_clean.ppo.semantic_policy_distribution import (
    HISTORY_POLICY, STATE_DEPENDENT_POLICY, policy_contract,
)


@pytest.fixture(autouse=True)
def one_cpu_thread(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    # This algorithm-only fixture verifies CPU RNG/optimizer behavior, not CUDA.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    yield
    torch.set_num_threads(previous)


def make(policy_version=STATE_DEPENDENT_POLICY):
    training.seed_training_rngs(1001)
    # One excluded roll-in decision, then one credited terminal per reset.
    core = Core([[('P06', 8, None), ('P09', 3, 'BODY_COLLISION')]])
    # Core.step returns a synthetic unit reward and does not implement PBRS.
    # Explicitly absent calculator selects the supported runner-only check;
    # do not disguise the old object() identity stand-in as a real calculator.
    core.reward_calculator = None
    env, core, evidence = adapter(core, Request('P06'))
    runner, _ = training.construct_semantic_runner(env, seed=1001, device="cpu",
                                                  policy_version=policy_version)
    source = provenance()
    source["actor_parameter_sha256"] = training.parameter_hash(runner.alg.actor)
    source["policy_contract"] = policy_contract(policy_version)
    frozen = build_frozen_checkpoint_prefix_policy(runner.alg.actor, source)
    env.install_prefix_policy(frozen, frozen.provenance)
    return runner, env, core, evidence, frozen


@pytest.mark.parametrize("policy_version", [STATE_DEPENDENT_POLICY, HISTORY_POLICY])
def test_terminal_reset_provenance_mutation_aborts_before_storage_optimizer_or_save(tmp_path, monkeypatch, policy_version):
    runner, env, core, evidence, _ = make(policy_version)
    assert core.resets == 1 and env.core.prefix_decisions == 1
    original_reset = core.reset

    def changed_provenance_during_reset(*args, **kwargs):
        observation = original_reset(*args, **kwargs)
        # Simulate a reset-side source replacement after the pre-step binding
        # check passed. The loop must independently reject the returned step.
        env.cfg["prefix_policy_provenance"]["checkpoint_sha256"] = "c" * 64
        return observation

    monkeypatch.setattr(core, "reset", changed_provenance_during_reset)
    calls = {name: 0 for name in ("process", "returns", "update", "save", "torch_save", "optimizer")}

    def forbidden(name):
        def fail(*args, **kwargs):
            calls[name] += 1
            raise AssertionError(f"{name} must not run after reset changes provenance")
        return fail

    monkeypatch.setattr(runner.alg, "process_env_step", forbidden("process"))
    monkeypatch.setattr(runner.alg, "compute_returns", forbidden("returns"))
    monkeypatch.setattr(runner.alg.optimizer, "step", forbidden("optimizer"))
    monkeypatch.setattr(training, "audited_ppo_update", forbidden("update"))
    monkeypatch.setattr(training, "save_semantic_checkpoint", forbidden("save"))
    monkeypatch.setattr(torch, "save", forbidden("torch_save"))
    before_actor = training.parameter_hash(runner.alg.actor)
    before_critic = training.parameter_hash(runner.alg.critic)
    before_optimizer = training.state_hash(runner.alg.optimizer.state_dict())
    with pytest.raises(RuntimeError, match="curriculum changed during a physical step/reset"):
        training.train_semantic(runner, env, run_dir=tmp_path / "run", output_root=tmp_path / "output",
                                stage="phase_suffix", decisions=128, contract={"cpu": "fixture"}, seed=1001)
    assert core.resets == 2  # The terminal-triggered reset really executed.
    assert env.core.prefix_decisions == 2 and env.total_decisions == 1
    assert len(env.completed_episodes) == 1
    assert all(value == 0 for value in calls.values())
    assert runner.alg.storage.step == 0
    assert runner.alg.transition.actions is not None  # Rejected pending act; never resumable.
    assert training.parameter_hash(runner.alg.actor) == before_actor
    assert training.parameter_hash(runner.alg.critic) == before_critic
    assert training.state_hash(runner.alg.optimizer.state_dict()) == before_optimizer
    for filename in ("residual_and_projection_audit.jsonl", "optimizer_updates.jsonl", "completed_episodes.jsonl"):
        assert (tmp_path / "run" / filename).read_text() == ""
    assert not list((tmp_path / "run/rollouts").iterdir())
    assert not (tmp_path / "output").exists()
    failure = json.loads((tmp_path / "run/training_failure.json").read_text())
    assert failure["lifecycle"] == "FAILED"
    assert failure["completed_updates_this_run"] == 0
    assert failure["last_verified_checkpoint"] is None
    assert failure["partial_rollout_or_update_is_not_resumable"] is True
    assert sum(row["kind"] == "policy_credit_start" for row in evidence) == 2


@pytest.mark.parametrize("policy_version", [STATE_DEPENDENT_POLICY, HISTORY_POLICY])
def test_real_frozen_prefix_actions_never_enter_official_storage_across_terminal_resets(tmp_path, policy_version):
    runner, env, core, evidence, frozen = make(policy_version)
    frozen_hash = training.parameter_hash(frozen._actor)
    result = training.train_semantic(runner, env, run_dir=tmp_path / "run", output_root=tmp_path / "output",
                                    stage="phase_suffix", decisions=128, contract={"cpu": "fixture"}, seed=1001)
    assert result["actual_policy_decisions"] == 128
    assert result["ppo_updates_this_run"] == 1 and result["optimizer_steps_this_run"] == 20
    assert result["finite_nonzero_gradient_observed"] is True
    assert result["actor_parameter_sha256_before"] != result["actor_parameter_sha256_after"]
    assert env.total_decisions == 128 and len(env.completed_episodes) == 128
    assert env.core.prefix_decisions == 129  # Initial + each post-terminal reset.
    assert env.core.prefix_ticks == 129 * 8
    assert env.core.credited_ticks == 128 * 3
    assert len(core.actions) == 257
    prefix_actions, credited_actions = core.actions[0::2], core.actions[1::2]
    assert len(prefix_actions) == 129 and len(credited_actions) == 128
    assert all(action == (0.,) * 12 for action in prefix_actions)
    assert training.parameter_hash(frozen._actor) == frozen_hash
    assert frozen._actor.distribution._distribution is None
    rollout = torch.load(tmp_path / "run/rollouts/rollout_000001.pt", weights_only=False)
    assert tuple(rollout["actions"].shape) == (128, 1, 12)
    assert torch.equal(rollout["actions"][:, 0], torch.tensor(credited_actions))
    assert rollout["policy_contract"] == policy_contract(policy_version)
    # Every reset's excluded zero-history roll-in ends at Core's observation1.
    # The actual stored conditional mean must use that credited observation,
    # not the prefix's observation0 or a private previous-episode action cache.
    assert torch.equal(rollout["observations"]["policy"][:, :, 195:207],
                       torch.ones(128, 1, 12))
    expected_mean = .9 if policy_version == HISTORY_POLICY else 0.
    assert torch.equal(rollout["distribution_params"][0],
                       torch.full((128, 1, 12), expected_mean))
    assert torch.all(rollout["dones"].bool())
    assert torch.equal(rollout["rewards"], torch.ones_like(rollout["rewards"]))
    # Official terminal GAE performs fl(fl(r - V) + V), not a direct copy
    # of r. Check that exact operation order: no reset-value bootstrap term.
    rewards, values = rollout["rewards"], rollout["values"]
    assert rewards.dtype == values.dtype == rollout["returns"].dtype == torch.float32
    expected_returns = (rewards - values) + values
    assert torch.equal(rollout["returns"], expected_returns)
    # Two rounded operations: gamma_2 = 2u/(1-2u), u = float32 eps/2.
    # This absolute operand-scale bound admits cancellation roundoff only,
    # rather than an arbitrary reward tolerance or altered terminal semantics.
    eps = torch.finfo(torch.float32).eps
    gamma_2 = eps / (1.0 - eps)
    tolerance = gamma_2 * (rewards.double().abs() + 2 * values.double().abs())
    assert torch.all((rollout["returns"].double() - rewards.double()).abs() <= tolerance)
    rows = [json.loads(line) for line in (tmp_path / "run/residual_and_projection_audit.jsonl").read_text().splitlines()]
    assert len(rows) == 128 and [row["global_policy_decision"] for row in rows] == list(range(1, 129))
    for index, row in enumerate(rows):
        assert row["raw_policy_action_full12"] == rollout["actions"][index, 0].tolist()
        assert row["old_log_probability"] == rollout["actions_log_prob"][index, 0].item()
        assert row["applied_audit"]["prefix_checkpoint_policy_data_in_ppo_storage"] is False
    assert sum(row["kind"] == "checkpoint_prefix_decision" for row in evidence) == 129
    assert all(row["policy_credit"] is False for row in evidence)
    checkpoint = json.loads(Path(result["checkpoints"][-1]["manifest"]).read_text())
    assert checkpoint["global_policy_decisions"] == 128
    assert checkpoint["policy_contract"] == policy_contract(policy_version)
    assert checkpoint["curriculum_epoch"]["prefix_policy_provenance"] == frozen.provenance
    assert checkpoint["save_load_round_trip"] is True
