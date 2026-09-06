from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rsl_rl")

from test_semantic_v3_continuation import Core324
from wlr50_clean.ppo import semantic_policy_distribution as policies
from wlr50_clean.ppo.semantic_training import (
    SemanticRslAdapter, construct_semantic_runner, load_semantic_checkpoint,
    parameter_hash, save_semantic_checkpoint, seed_training_rngs, state_hash,
    train_semantic,
)


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make(policy=policies.LEGACY_POLICY, *, initialize=True):
    env = SemanticRslAdapter(Core324(), seed=1001, device="cpu")
    env.cfg["semantic_version"] = "v3"
    runner, config = construct_semantic_runner(env, seed=1001, device="cpu",
        policy_version=policy, initialize_actor=initialize)
    return runner, env, config


def test_fresh_log_head_keeps_official_sigma_and_zeroes_only_mean():
    runner, env, _ = make(policies.STATE_DEPENDENT_POLICY)
    observations = env.get_observations()
    assert torch.equal(runner.alg.actor(observations), torch.zeros(1, 12))
    runner.alg.act(observations)
    assert torch.allclose(runner.alg.actor.output_std, torch.full((1, 12), .15), atol=2e-7)
    assert not torch.allclose(runner.alg.actor.output_std, torch.ones(1, 12))


def test_actual_official_policy_migration_update_exact_resume_and_raw_audit(tmp_path, monkeypatch):
    seed_training_rngs(1001)
    source, source_env, _ = make()
    old_contract, target_contract = {"revision": "source-CPU"}, {"revision": "target-CPU"}
    old_result = train_semantic(source, source_env, run_dir=tmp_path / "old_run",
        output_root=tmp_path / "old_output", stage="full_episode", decisions=128,
        contract=old_contract, seed=1001)
    previous = json.loads(Path(old_result["checkpoints"][-1]["manifest"]).read_text())
    for publication_key in ("checkpoint_path", "checkpoint_sha256", "save_load_round_trip"):
        previous.pop(publication_key)
    previous["new_mdp_origin_global_policy_decisions"] = 0
    # Deliberately different from factory initial LR: migration must use actual
    # restored Adam settings, not the configured 3e-5 default.
    source.alg.learning_rate = 1e-5
    source.alg.optimizer.param_groups[0]["lr"] = 1e-5
    source_path = tmp_path / "source_adjusted.pt"
    save_semantic_checkpoint(source, source_path, previous)
    source_state = {name: value.clone() for name, value in source.alg.actor.state_dict().items()}
    critic_hash = parameter_hash(source.alg.critic)
    expected_rng = torch.rand(9)
    target, env, config = make(policies.STATE_DEPENDENT_POLICY, initialize=False)
    record = {"source_runtime_contract": old_contract, "target_runtime_contract": target_contract,
              "optimizer": {"initial_learning_rate": 1e-5},
              "physical_mdp_changed": False, "reward_changed": False}
    # Pure-file/hash plan validation has separate positive/negative tests. This
    # seam exercises real RSL tensor load, mapping, RNG and optimizer behavior.
    monkeypatch.setattr(policies, "build_policy_distribution_migration", lambda path, contract: record)
    migrated = load_semantic_checkpoint(target, source_path, contract=target_contract,
        seed=1001, policy_migration=record)
    assert torch.equal(torch.rand(9), expected_rng)
    assert env.core.calls == 0 and env.core.resets == 1
    assert target.alg.optimizer.state_dict()["state"] == {}
    assert target.alg.learning_rate == 1e-5
    assert target.alg.optimizer.param_groups[0]["lr"] == 1e-5
    assert parameter_hash(target.alg.critic) == critic_hash
    assert migrated["stage_requested_decisions"] == previous["stage_requested_decisions"]
    assert migrated["new_mdp_origin_global_policy_decisions"] == 0
    assert (migrated["global_policy_decisions"], migrated["ppo_updates"], migrated["optimizer_steps"]) == (128, 1, 20)
    assert migrated["runner_config"] == config
    assert target.alg.storage.step == 0 and target.alg.transition.actions is None
    mapped = target.alg.actor.state_dict()
    for name, value in source_state.items():
        if name == "distribution.std_param":
            assert torch.allclose(mapped["mlp.4.bias"][12:].exp(), value, atol=1e-7)
        elif name in ("mlp.4.weight", "mlp.4.bias"):
            assert torch.equal(mapped[name][:12], value)
        else:
            assert torch.equal(mapped[name], value)
    assert torch.count_nonzero(mapped["mlp.4.weight"][12:]) == 0
    initial = tmp_path / "converted_initial.pt"
    save_semantic_checkpoint(target, initial, {**migrated, "runtime_contract": target_contract})
    reloaded, _, _ = make(policies.STATE_DEPENDENT_POLICY, initialize=False)
    load_semantic_checkpoint(reloaded, initial, contract=target_contract, seed=1001)
    assert parameter_hash(reloaded.alg.actor) == parameter_hash(target.alg.actor)
    before_std_head = target.alg.actor.state_dict()["mlp.4.weight"][12:].clone()
    result = train_semantic(target, env, run_dir=tmp_path / "new_run", output_root=tmp_path / "new_output",
        stage="full_episode", decisions=128, contract=target_contract, seed=1001, resume_infos=migrated)
    assert result["global_policy_decisions"] == 256
    assert result["ppo_updates_this_run"] == 1 and result["optimizer_steps_this_run"] == 20
    assert not torch.equal(before_std_head, target.alg.actor.state_dict()["mlp.4.weight"][12:])
    rows = [json.loads(line) for line in (tmp_path / "new_run/residual_and_projection_audit.jsonl").read_text().splitlines()]
    rollout = torch.load(tmp_path / "new_run/rollouts/rollout_000002.pt", weights_only=False)
    for index, row in enumerate(rows):
        assert row["raw_policy_action_full12"] == rollout["actions"][index, 0].tolist()
        assert row["old_distribution_mean_full12"] == rollout["distribution_params"][0][index, 0].tolist()
        assert row["old_distribution_std_full12"] == rollout["distribution_params"][1][index, 0].tolist()
    expected_lp = torch.distributions.Normal(*rollout["distribution_params"]).log_prob(rollout["actions"]).sum(-1)
    torch.testing.assert_close(expected_lp, rollout["actions_log_prob"].squeeze(-1))
    latest = Path(result["checkpoints"][-1]["checkpoint"])
    fresh, _, _ = make(policies.STATE_DEPENDENT_POLICY, initialize=False)
    final_infos = load_semantic_checkpoint(fresh, latest, contract=target_contract, seed=1001)
    assert parameter_hash(fresh.alg.actor) == parameter_hash(target.alg.actor)
    assert state_hash(fresh.alg.optimizer.state_dict()) == state_hash(target.alg.optimizer.state_dict())
    assert final_infos["policy_distribution_migration"] == record
    assert final_infos["stage_requested_decisions"]["full_episode"] == 256
    wrong, _, _ = make()
    with pytest.raises(RuntimeError, match="policy distribution"):
        load_semantic_checkpoint(wrong, latest, contract=target_contract, seed=1001)


def test_partial_rollout_or_mixed_migrations_are_rejected_before_conversion(tmp_path, monkeypatch):
    target, env, _ = make(policies.STATE_DEPENDENT_POLICY, initialize=False)
    record = {"source_runtime_contract": {"old": 1}, "optimizer": {"initial_learning_rate": 1e-5}}
    monkeypatch.setattr(policies, "build_policy_distribution_migration", lambda path, contract: record)
    with pytest.raises(ValueError, match="separate operations"):
        load_semantic_checkpoint(target, tmp_path / "absent.pt", contract={}, seed=1001,
                                 policy_migration=record, warm_start={})
    target.alg.act(env.get_observations())
    with pytest.raises(RuntimeError, match="fresh"):
        load_semantic_checkpoint(target, tmp_path / "absent.pt", contract={}, seed=1001,
                                 policy_migration=record)
