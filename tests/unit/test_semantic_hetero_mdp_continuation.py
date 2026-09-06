"""Official CPU optimizer/loader regression; the core is not Isaac evidence."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rsl_rl")

from test_semantic_policy_training import make
from test_semantic_v3_continuation import contracts
from wlr50_clean.ppo import semantic_migration as migration
from wlr50_clean.ppo.semantic_policy_distribution import (
    STATE_DEPENDENT_POLICY, policy_contract, policy_version_from_metadata,
)
from wlr50_clean.ppo.semantic_training import (
    load_semantic_checkpoint, parameter_hash, save_semantic_checkpoint,
    seed_training_rngs, state_hash, train_semantic,
)


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_learned_hetero_new_mdp_initial_roundtrip_and_fresh_update(tmp_path, monkeypatch):
    seed_training_rngs(1001)
    _, old_contract = contracts(tmp_path)
    source, source_env, _ = make(STATE_DEPENDENT_POLICY)
    before_head = source.alg.actor.state_dict()["mlp.4.weight"][12:].clone()
    source_result = train_semantic(source, source_env,
        run_dir=tmp_path / "source_run", output_root=tmp_path / "source_output",
        stage="full_episode", decisions=128, contract=old_contract, seed=1001)
    learned_head = source.alg.actor.state_dict()["mlp.4.weight"][12:].clone()
    assert not torch.equal(before_head, learned_head)
    assert torch.count_nonzero(learned_head) > 0
    infos = json.loads(Path(source_result["checkpoints"][-1]["manifest"]).read_text())
    for key in ("checkpoint_path", "checkpoint_sha256", "save_load_round_trip"):
        infos.pop(key)
    infos["new_mdp_origin_global_policy_decisions"] = 0
    infos["policy_distribution_migration"] = {"lineage": "CPU fixture earlier architecture boundary"}
    # This boundary deliberately resets LR to 3e-5, unlike policy-only migration.
    source.alg.learning_rate = 1e-5
    for group in source.alg.optimizer.param_groups:
        group["lr"] = 1e-5
    checkpoint = tmp_path / "learned_hetero_source.pt"
    save_semantic_checkpoint(source, checkpoint, infos)
    source_sha = migration.file_sha(checkpoint)
    expected_rng = torch.rand(11)
    actor_hash, critic_hash = parameter_hash(source.alg.actor), parameter_hash(source.alg.critic)

    new_contract = copy.deepcopy(old_contract)
    new_contract["source_git_commit"] = "3" * 40
    # A real temporary config-byte change, not a mock validator. This checks
    # migration plumbing, not the physics or reward effectiveness of that change.
    relative = "configs/ppo_semantic_v3/reward_config.yaml"
    reward_file = tmp_path / relative
    reward_file.write_text(reward_file.read_text().replace("potential_weight: 5.0", "potential_weight: 5.1"))
    new_contract["files"][relative] = migration.file_sha(reward_file)
    new_contract["runtime_content_sha256"] = migration.digest(new_contract["files"])
    record = migration.build_v3_warm_start_record(checkpoint, new_contract, project_root=tmp_path)
    assert record["runtime_changed_files"] == [relative]
    actual_validator = migration.build_v3_warm_start_record
    monkeypatch.setattr(migration, "build_v3_warm_start_record",
        lambda path, contract: actual_validator(path, contract, project_root=tmp_path))

    target, env, target_config = make(STATE_DEPENDENT_POLICY, initialize=False)
    resumed = load_semantic_checkpoint(target, checkpoint, contract=new_contract,
        seed=1001, warm_start=record)
    assert torch.equal(torch.rand(11), expected_rng)
    assert parameter_hash(target.alg.actor) == actor_hash
    assert parameter_hash(target.alg.critic) == critic_hash
    assert torch.equal(target.alg.actor.state_dict()["mlp.4.weight"][12:], learned_head)
    assert target.alg.optimizer.state_dict()["state"] == {}
    assert target.alg.learning_rate == 3e-5
    assert all(group["lr"] == 3e-5 for group in target.alg.optimizer.param_groups)
    assert target.alg.storage.step == 0 and target.alg.transition.actions is None
    assert tuple(target.alg.storage.actions.shape) == (128, 1, 12)
    assert env.core.calls == 0 and env.core.resets == 1
    assert (resumed["global_policy_decisions"], resumed["ppo_updates"], resumed["optimizer_steps"]) == (128, 1, 20)
    assert resumed["stage_requested_decisions"] == infos["stage_requested_decisions"]
    assert resumed["new_mdp_origin_global_policy_decisions"] == 0
    assert resumed["policy_distribution_migration"] == infos["policy_distribution_migration"]
    normalizers = {role: getattr(target.alg, role).obs_normalizer.state_dict() for role in ("actor", "critic")}
    assert not any(normalizers.values())
    assert state_hash(normalizers) == resumed["normalizer_state_sha256"]

    initial = tmp_path / migration.v3_warm_start_checkpoint_name(record)
    _, initial_sidecar = save_semantic_checkpoint(target, initial,
        {**resumed, "runtime_contract": new_contract})
    saved_initial = json.loads(initial_sidecar.read_text())
    assert saved_initial["runner_config"] == target_config
    assert saved_initial["policy_contract"] == policy_contract(STATE_DEPENDENT_POLICY)
    assert policy_version_from_metadata(saved_initial) == STATE_DEPENDENT_POLICY
    assert saved_initial["save_load_round_trip"] is True
    reloaded, _, _ = make(STATE_DEPENDENT_POLICY, initialize=False)
    loaded_initial = load_semantic_checkpoint(reloaded, initial, contract=new_contract, seed=1001)
    assert parameter_hash(reloaded.alg.actor) == actor_hash
    assert parameter_hash(reloaded.alg.critic) == critic_hash
    assert state_hash(reloaded.alg.optimizer.state_dict()) == state_hash(target.alg.optimizer.state_dict())
    assert loaded_initial["new_mdp_warm_start"] == record

    continued = train_semantic(target, env, run_dir=tmp_path / "continued_run",
        output_root=tmp_path / "continued_output", stage="full_episode", decisions=128,
        contract=new_contract, seed=1001, resume_infos=resumed)
    assert continued["global_policy_decisions"] == 256
    assert (continued["ppo_updates_this_run"], continued["optimizer_steps_this_run"]) == (1, 20)
    final_checkpoint = Path(continued["checkpoints"][-1]["checkpoint"])
    fresh, _, _ = make(STATE_DEPENDENT_POLICY, initialize=False)
    final_infos = load_semantic_checkpoint(fresh, final_checkpoint, contract=new_contract, seed=1001)
    assert (final_infos["global_policy_decisions"], final_infos["ppo_updates"], final_infos["optimizer_steps"]) == (256, 2, 40)
    assert final_infos["stage_requested_decisions"]["full_episode"] == 256
    assert final_infos["new_mdp_origin_global_policy_decisions"] == 0
    assert final_infos["new_mdp_warm_start"] == record
    assert parameter_hash(fresh.alg.actor) == parameter_hash(target.alg.actor)
    assert parameter_hash(fresh.alg.critic) == parameter_hash(target.alg.critic)
    assert state_hash(fresh.alg.optimizer.state_dict()) == state_hash(target.alg.optimizer.state_dict())
    assert migration.file_sha(checkpoint) == source_sha
