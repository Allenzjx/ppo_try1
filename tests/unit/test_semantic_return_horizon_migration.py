"""Actual official CPU PPO migration; the small core is not physics evidence.

Only the historical runner factory's return-profile selection is patched. The
actor (including learned log std), critic, Adam, storage and save/load are real.
The attached real reward calculator tests configuration binding, not Core324's
synthetic reward formula or any physical task outcome.
"""
from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

torch = pytest.importorskip("torch")
pytest.importorskip("rsl_rl")

from test_semantic_v3_continuation import Core324, contracts
from wlr50_clean.ppo import semantic_migration as migration
from wlr50_clean.ppo import semantic_training as training
from wlr50_clean.ppo.semantic_policy_distribution import (
    STATE_DEPENDENT_POLICY, policy_contract, policy_version_from_metadata,
)
from wlr50_clean.ppo.semantic_return_profile import (
    LEGACY_RETURN_PROFILE, RETURN_PROFILE, RUNNER_PROFILE_KEY,
    profile_parameters, runner_return_profile,
)
from wlr50_clean.ppo.semantic_reward import (
    SemanticRewardCalculator, load_semantic_reward_config,
)


REWARD = "configs/ppo_semantic_v3/reward_config.yaml"
COUNTERS = ("global_policy_decisions", "ppo_updates", "optimizer_steps")
PUBLICATION_FIELDS = ("checkpoint_path", "checkpoint_sha256", "save_load_round_trip")


@pytest.fixture(scope="module", autouse=True)
def cpu_only_single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    # The production RNG helper otherwise captures CUDA RNG on a GPU host even
    # for a CPU runner. This test proves CPU RNG restoration only, without CUDA
    # initialization, and does not replace any official model/optimizer code.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch.cuda, "is_available", lambda: False)
        try:
            yield
        finally:
            torch.set_num_threads(previous)


def _git(root: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(root), *arguments], check=True,
        capture_output=True,
    ).stdout


def _commit_configs(root: Path, message: str) -> str:
    _git(root, "-c", "core.autocrlf=false", "add", "configs")
    _git(root, "-c", "user.name=Semantic CPU test",
         "-c", "user.email=semantic-cpu@example.invalid",
         "-c", "commit.gpgsign=false", "commit", "-m", message)
    return _git(root, "rev-parse", "HEAD").decode().strip()


def _contract_at(root: Path, template: dict, revision: str) -> dict:
    result = copy.deepcopy(template)
    result["source_git_commit"] = revision
    result["files"] = {relative: migration.file_sha(root / relative)
                       for relative in result["files"]}
    result["runtime_content_sha256"] = migration.digest(result["files"])
    return result


def _runner(reward_config, *, legacy: bool = False, initialize: bool = False):
    core = Core324()
    core.reward_calculator = SemanticRewardCalculator(reward_config)
    env = training.SemanticRslAdapter(core, seed=1001, device="cpu")
    env.cfg["semantic_version"] = "v3"
    if legacy:
        actual_factory = training.semantic_runner_config

        def historical_factory(**kwargs):
            assert kwargs.get("semantic_version") == "v3"
            return actual_factory(**{**kwargs, "return_profile": LEGACY_RETURN_PROFILE})

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(training, "semantic_runner_config", historical_factory)
            runner, config = training.construct_semantic_runner(
                env, seed=1001, device="cpu", policy_version=STATE_DEPENDENT_POLICY,
                initialize_actor=initialize,
            )
    else:
        runner, config = training.construct_semantic_runner(
            env, seed=1001, device="cpu", policy_version=STATE_DEPENDENT_POLICY,
            initialize_actor=initialize,
        )
    return runner, env, config


def _metadata(checkpoint: Path) -> dict:
    return json.loads(checkpoint.with_name(checkpoint.stem + "_manifest.json").read_text())


def _infos(metadata: dict) -> dict:
    return {key: copy.deepcopy(value) for key, value in metadata.items()
            if key not in PUBLICATION_FIELDS}


def _normalizers(runner) -> dict:
    return {role: getattr(runner.alg, role).obs_normalizer.state_dict()
            for role in ("actor", "critic")}


def _assert_weights_and_identity(runner, source) -> None:
    for role in ("actor", "critic"):
        actual = getattr(runner.alg, role).state_dict()
        expected = getattr(source.alg, role).state_dict()
        assert actual.keys() == expected.keys()
        for key in actual:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        assert getattr(runner.alg, role).obs_normalization is False
    assert _normalizers(runner) == {"actor": {}, "critic": {}}
    assert training.state_hash(_normalizers(runner)) == training.state_hash(_normalizers(source))


@pytest.fixture(scope="module")
def historical_source(tmp_path_factory, cpu_only_single_thread):
    root = tmp_path_factory.mktemp("return_horizon_git")
    _, template = contracts(root)
    reward_path = root / REWARD
    target_bytes = reward_path.read_bytes()
    legacy_values = yaml.safe_load(target_bytes)
    legacy_values.pop("return_profile")
    legacy_values["gamma"] = 0.995
    source_bytes = yaml.safe_dump(legacy_values, sort_keys=False).encode("utf-8")
    reward_path.write_bytes(source_bytes)
    _git(root, "init")
    source_revision = _commit_configs(root, "CPU historical legacy return configuration")
    old_contract = _contract_at(root, template, source_revision)
    old_reward = load_semantic_reward_config(reward_path)

    training.seed_training_rngs(1001)
    source, env, old_runner_config = _runner(old_reward, legacy=True, initialize=True)
    initial_std_head = source.alg.actor.state_dict()["mlp.4.weight"][12:].clone()
    result = training.train_semantic(
        source, env, run_dir=root / "source_run", output_root=root / "source_output",
        stage="full_episode", decisions=128, contract=old_contract, seed=1001,
    )
    assert (result["actual_policy_decisions"], result["ppo_updates_this_run"],
            result["optimizer_steps_this_run"]) == (128, 1, 20)
    assert result["finite_nonzero_gradient_observed"] is True
    learned_std_head = source.alg.actor.state_dict()["mlp.4.weight"][12:]
    assert not torch.equal(initial_std_head, learned_std_head)
    assert torch.count_nonzero(learned_std_head) > 0
    assert type(source.alg.optimizer) is torch.optim.Adam
    assert source.alg.optimizer.state_dict()["state"]
    infos = _infos(json.loads(Path(result["checkpoints"][-1]["manifest"]).read_text()))
    infos["new_mdp_origin_global_policy_decisions"] = 0
    infos["policy_distribution_migration"] = {"lineage": "CPU fixture prior hetero architecture boundary"}
    source.alg.learning_rate = 1e-5
    for group in source.alg.optimizer.param_groups:
        group["lr"] = 1e-5
    checkpoint = root / "learned_legacy_hetero_source.pt"
    training.save_semantic_checkpoint(source, checkpoint, infos)
    source_metadata = _metadata(checkpoint)

    # Source bytes now exist only in this real Git history. Both revisions and
    # their file hashes are genuine; no fake SHA or source-validator bypass.
    reward_path.write_bytes(target_bytes)
    target_revision = _commit_configs(root, "CPU target versioned gamma and lambda")
    new_contract = _contract_at(root, template, target_revision)
    assert source_revision != target_revision
    assert reward_path.read_bytes() != source_bytes
    assert _git(root, "show", f"{source_revision}:{REWARD}") == source_bytes
    assert migration._version_bytes(root, old_contract, REWARD, prefer_worktree=True) == source_bytes
    yield SimpleNamespace(
        root=root, source=source, checkpoint=checkpoint, metadata=source_metadata,
        old_contract=old_contract, new_contract=new_contract, old_reward=old_reward,
        new_reward=load_semantic_reward_config(reward_path), old_config=old_runner_config,
        source_bytes=source_bytes,
    )


def test_legacy_hetero_return_change_initial_exact_reload_then_actual_update(
        historical_source, tmp_path, monkeypatch):
    fixture = historical_source
    source = fixture.source
    source_checkpoint_bytes = fixture.checkpoint.read_bytes()
    source_sidecar_bytes = fixture.checkpoint.with_name(
        fixture.checkpoint.stem + "_manifest.json").read_bytes()
    assert RUNNER_PROFILE_KEY not in fixture.old_config
    assert fixture.old_config == training.semantic_runner_config(
        seed=1001, device="cpu", semantic_version="v3",
        policy_version=STATE_DEPENDENT_POLICY, return_profile=LEGACY_RETURN_PROFILE,
    )
    assert (source.alg.gamma, source.alg.lam) == (0.995, 0.95)
    assert policy_version_from_metadata(fixture.metadata) == STATE_DEPENDENT_POLICY
    record = migration.build_v3_warm_start_record(
        fixture.checkpoint, fixture.new_contract, project_root=fixture.root)
    transition = {
        "source": profile_parameters(LEGACY_RETURN_PROFILE, semantic_version="v3"),
        "target": profile_parameters(RETURN_PROFILE, semantic_version="v3"),
        "reward_configuration": {
            "source_path": REWARD, "source_sha256": fixture.old_contract["files"][REWARD],
            "target_path": REWARD, "target_sha256": fixture.new_contract["files"][REWARD],
        },
        "reward_discount_changed": True, "gae_estimator_changed": True,
        "rollout_storage_inherited": False,
    }
    assert record["return_horizon_transition"] == transition
    assert record["runtime_changed_files"] == [REWARD]
    assert record["old_rollout_buffer_inherited"] is False
    assert record["physical_state_inherited"] is False
    assert record["source_stage_requested_decisions"] == fixture.metadata["stage_requested_decisions"]
    assert record["target_stage_requested_decisions"] == fixture.metadata["stage_requested_decisions"]
    # Redirect only the default project root: the same full real validator runs
    # again inside load, including Git recovery of the old reward configuration.
    actual_validator = migration.build_v3_warm_start_record
    monkeypatch.setattr(migration, "build_v3_warm_start_record",
        lambda checkpoint, contract: actual_validator(checkpoint, contract, project_root=fixture.root))
    target, env, target_config = _runner(fixture.new_reward)
    resumed = training.load_semantic_checkpoint(
        target, fixture.checkpoint, contract=fixture.new_contract, seed=1001, warm_start=record)
    _assert_weights_and_identity(target, source)
    assert training.capture_training_rng_state(seed=1001) == fixture.metadata["training_rng_state"]
    assert (target.alg.gamma, target.alg.lam) == (0.9985, 0.99)
    assert training.assert_semantic_return_consistency(target, env) == transition["target"]
    assert target_config[RUNNER_PROFILE_KEY] == RETURN_PROFILE
    assert type(target.alg.optimizer) is torch.optim.Adam
    assert target.alg.optimizer.state_dict()["state"] == {}
    assert target.alg.learning_rate == 3e-5
    assert all(group["lr"] == 3e-5 for group in target.alg.optimizer.param_groups)
    assert fixture.metadata["optimizer_learning_rate"] == 1e-5
    assert target.alg.storage is not source.alg.storage
    assert tuple(target.alg.storage.actions.shape) == (128, 1, 12)
    assert target.alg.storage.observations["policy"].shape[-1] == 324
    assert target.alg.storage.step == 0 and target.alg.transition.actions is None
    assert env.core.calls == 0 and env.core.resets == 1
    assert tuple(resumed[key] for key in COUNTERS) == (128, 1, 20)
    assert resumed["stage_requested_decisions"] == {"smoke": 0, "phase_suffix": 0, "full_episode": 128}
    assert resumed["new_mdp_origin_global_policy_decisions"] == 0
    assert resumed["policy_distribution_migration"] == fixture.metadata["policy_distribution_migration"]

    initial = tmp_path / migration.v3_warm_start_checkpoint_name(record)
    training.save_semantic_checkpoint(target, initial, {**resumed, "runtime_contract": fixture.new_contract})
    saved = _metadata(initial)
    embedded = torch.load(initial, map_location="cpu", weights_only=False)["infos"]
    assert embedded == _infos(saved)
    assert saved["runner_config"] == target_config
    assert runner_return_profile(saved["runner_config"], semantic_version="v3") == transition["target"]
    assert policy_version_from_metadata(saved) == STATE_DEPENDENT_POLICY
    assert saved["policy_contract"] == policy_contract(STATE_DEPENDENT_POLICY)
    assert saved["runtime_contract"] == fixture.new_contract
    assert saved["new_mdp_warm_start"]["return_horizon_transition"] == transition
    assert saved["training_rng_state"] == fixture.metadata["training_rng_state"]
    assert saved["save_load_round_trip"] is True
    assert tuple(saved[key] for key in COUNTERS) == (128, 1, 20)
    with pytest.raises(FileExistsError):
        training.save_semantic_checkpoint(target, initial, {**resumed, "runtime_contract": fixture.new_contract})

    reloaded, fresh_env, _ = _runner(fixture.new_reward)
    loaded = training.load_semantic_checkpoint(reloaded, initial, contract=fixture.new_contract, seed=1001)
    _assert_weights_and_identity(reloaded, source)
    assert training.capture_training_rng_state(seed=1001) == saved["training_rng_state"]
    assert (reloaded.alg.gamma, reloaded.alg.lam) == (0.9985, 0.99)
    assert reloaded.alg.optimizer.state_dict()["state"] == {}
    assert training.state_hash(reloaded.alg.optimizer.state_dict()) == saved["optimizer_state_sha256"]
    assert all(group["lr"] == 3e-5 for group in reloaded.alg.optimizer.param_groups)
    assert reloaded.alg.storage.step == 0 and reloaded.alg.transition.actions is None
    assert fresh_env.total_decisions == 0 and fresh_env.core.calls == 0
    assert loaded["new_mdp_warm_start"] == record
    continued = training.train_semantic(
        reloaded, fresh_env, run_dir=tmp_path / "continued_run", output_root=tmp_path / "continued_output",
        stage="full_episode", decisions=128, contract=fixture.new_contract, seed=1001, resume_infos=loaded,
    )
    assert (continued["actual_policy_decisions"], continued["ppo_updates_this_run"],
            continued["optimizer_steps_this_run"]) == (128, 1, 20)
    assert continued["global_policy_decisions"] == 256
    assert continued["finite_nonzero_gradient_observed"] is True
    assert continued["actor_parameter_sha256_before"] == saved["actor_parameter_sha256"]
    assert continued["actor_parameter_sha256_after"] != saved["actor_parameter_sha256"]
    assert fresh_env.core.calls == 128 and fresh_env.total_decisions == 128
    assert continued["training_success_is_not_task_success"] is True
    final_path = Path(continued["checkpoints"][-1]["checkpoint"])
    final_runner, _, _ = _runner(fixture.new_reward)
    final = training.load_semantic_checkpoint(final_runner, final_path, contract=fixture.new_contract, seed=1001)
    assert tuple(final[key] for key in COUNTERS) == (256, 2, 40)
    assert final["stage_requested_decisions"] == {"smoke": 0, "phase_suffix": 0, "full_episode": 256}
    assert final["new_mdp_origin_global_policy_decisions"] == 0
    assert final["runner_config"] == target_config
    assert final["new_mdp_warm_start"] == record
    assert (final_runner.alg.gamma, final_runner.alg.lam) == (0.9985, 0.99)
    assert final_runner.alg.optimizer.state_dict()["state"]
    assert training.state_hash(final_runner.alg.optimizer.state_dict()) == training.state_hash(reloaded.alg.optimizer.state_dict())
    _assert_weights_and_identity(final_runner, reloaded)
    assert fixture.checkpoint.read_bytes() == source_checkpoint_bytes
    assert fixture.checkpoint.with_name(fixture.checkpoint.stem + "_manifest.json").read_bytes() == source_sidecar_bytes


def test_return_change_without_explicit_new_mdp_is_rejected_before_credit(historical_source):
    fixture = historical_source
    target, env, _ = _runner(fixture.new_reward)
    before_actor = training.parameter_hash(target.alg.actor)
    before_optimizer = training.state_hash(target.alg.optimizer.state_dict())
    with pytest.raises(RuntimeError, match="return-profile changes require explicit new-MDP warm start"):
        training.load_semantic_checkpoint(target, fixture.checkpoint, contract=fixture.new_contract, seed=1001)
    assert env.core.calls == 0 and env.total_decisions == 0
    assert target.alg.storage.step == 0 and target.alg.transition.actions is None
    assert training.parameter_hash(target.alg.actor) == before_actor
    assert training.state_hash(target.alg.optimizer.state_dict()) == before_optimizer


@pytest.mark.parametrize("runner_is_legacy", [True, False], ids=["legacy-runner-new-reward", "new-runner-legacy-reward"])
def test_hash_bound_historical_reward_runner_mix_is_rejected(historical_source, tmp_path, runner_is_legacy):
    fixture = historical_source
    reward = fixture.old_reward if runner_is_legacy else fixture.new_reward
    runner, env, _ = _runner(reward, legacy=runner_is_legacy)
    # Restore actual trained networks/Adam with the unchanged official loader,
    # then publish a deliberately wrong historical *contract*. Runner and its
    # live calculator remain consistent, so rejection must inspect real source
    # reward bytes rather than just accept a valid-looking runner marker.
    infos = dict(training.load_checkpoint_round_trip(runner, fixture.checkpoint))
    wrong_contract = fixture.new_contract if runner_is_legacy else fixture.old_contract
    mixed = tmp_path / "mixed_source.pt"
    training.save_semantic_checkpoint(runner, mixed, {**infos, "runtime_contract": wrong_contract})
    assert policy_version_from_metadata(_metadata(mixed)) == STATE_DEPENDENT_POLICY
    with pytest.raises(ValueError, match="source runner discount/profile differs from its hash-bound historical reward config"):
        migration.build_v3_warm_start_record(mixed, fixture.new_contract, project_root=fixture.root)
    assert env.core.calls == 0


@pytest.mark.parametrize("mismatch", ["gamma", "lambda", "calculator"])
def test_actual_ppo_or_reward_mismatch_rejected_before_rollout(historical_source, tmp_path, mismatch):
    fixture = historical_source
    runner, env, _ = _runner(fixture.new_reward)
    if mismatch == "gamma":
        runner.alg.gamma = 0.995
    elif mismatch == "lambda":
        runner.alg.lam = 0.95
    else:
        env.core.reward_calculator = SemanticRewardCalculator(fixture.old_reward)
    actor_before = training.parameter_hash(runner.alg.actor)
    message = "actual PPO gamma/lambda" if mismatch != "calculator" else "actual semantic PBRS return profiles"
    with pytest.raises(RuntimeError, match=message):
        training.train_semantic(
            runner, env, run_dir=tmp_path / "rejected_run", output_root=tmp_path / "rejected_output",
            stage="full_episode", decisions=128, contract=fixture.new_contract, seed=1001,
        )
    assert env.core.calls == 0 and env.total_decisions == 0
    assert runner.alg.storage.step == 0 and runner.alg.transition.actions is None
    assert training.parameter_hash(runner.alg.actor) == actor_before
    assert runner.alg.optimizer.state_dict()["state"] == {}
    assert not (tmp_path / "rejected_run").exists()
