"""Real official CPU networks/Adam; no physics or actual training-source mutation."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rsl_rl")
from tensordict import TensorDict

from test_semantic_policy_training import make
from test_semantic_return_horizon_migration import _commit_configs, _contract_at, _git, _infos, _metadata
from test_semantic_v3_continuation import contracts
from wlr50_clean.ppo import semantic_migration as migration
from wlr50_clean.ppo import semantic_training as training
from wlr50_clean.ppo.semantic_history_actor import HISTORY_RHO, SemanticHistoryMLPModel
from wlr50_clean.ppo.semantic_policy_distribution import HISTORY_POLICY, policy_contract

OBSERVATION = "configs/ppo_semantic_v3/observation_schema.json"
COLUMNS = (210, 222)
COUNTERS = ("global_policy_decisions", "ppo_updates", "optimizer_steps")


@pytest.fixture(scope="module", autouse=True)
def cpu_only_one_thread():
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch.cuda, "is_available", lambda: False)
        yield
    torch.set_num_threads(old_threads)


@pytest.fixture(scope="module")
def source_fixture(tmp_path_factory, cpu_only_one_thread):
    root = tmp_path_factory.mktemp("transfer_scale_git")
    _, template = contracts(root)
    schema_path = root / OBSERVATION
    schema = json.loads(schema_path.read_text())
    for group in schema["feature_groups"]:
        if group["name"] in ("previous_residual_full12", "previous_previous_residual_full12"):
            group["scale"][3] = 4
    schema_path.write_text(json.dumps(schema), encoding="utf-8")
    source_schema_bytes = schema_path.read_bytes()
    _git(root, "init")
    old_contract = _contract_at(root, template, _commit_configs(root, "real scale4 source"))
    training.seed_training_rngs(1001)
    source, env, _ = make(HISTORY_POLICY)
    initial_std = source.alg.actor.state_dict()["mlp.4.weight"][12:].clone()
    result = training.train_semantic(source, env, run_dir=root / "old_run",
        output_root=root / "old_output", stage="full_episode", decisions=128,
        contract=old_contract, seed=1001)
    assert result["finite_nonzero_gradient_observed"]
    assert not torch.equal(initial_std, source.alg.actor.state_dict()["mlp.4.weight"][12:])
    infos = _infos(json.loads(Path(result["checkpoints"][-1]["manifest"]).read_text()))
    infos["new_mdp_origin_global_policy_decisions"] = 0
    source.alg.learning_rate = 1e-5
    for group in source.alg.optimizer.param_groups:
        group["lr"] = 1e-5
    checkpoint = root / "learned_history_source.pt"
    training.save_semantic_checkpoint(source, checkpoint, infos)
    for group in schema["feature_groups"]:
        if group["name"] in ("previous_residual_full12", "previous_previous_residual_full12"):
            group["scale"][3] = 6
    schema_path.write_text(json.dumps(schema), encoding="utf-8")
    new_contract = _contract_at(root, template, _commit_configs(root, "reviewed scale6 target"))
    new_contract["experiment_id"] = "transfer_roles_v1"
    assert _git(root, "show", f"{old_contract['source_git_commit']}:{OBSERVATION}") == source_schema_bytes
    assert migration._version_bytes(root, old_contract, OBSERVATION, prefer_worktree=True) == source_schema_bytes
    return SimpleNamespace(root=root, source=source, checkpoint=checkpoint,
        metadata=_metadata(checkpoint), old_contract=old_contract, new_contract=new_contract)


def _load_target(fixture, monkeypatch):
    record = migration.build_v3_warm_start_record(fixture.checkpoint, fixture.new_contract,
                                                project_root=fixture.root)
    actual_validator = migration.build_v3_warm_start_record
    monkeypatch.setattr(migration, "build_v3_warm_start_record",
        lambda path, contract, **options: actual_validator(path, contract, project_root=fixture.root, **options))
    target, env, config = make(HISTORY_POLICY, initialize=False)
    infos = training.load_semantic_checkpoint(target, fixture.checkpoint,
        contract=fixture.new_contract, seed=1001, warm_start=record)
    return target, env, config, record, infos


def _assert_only_reviewed_columns(source, target):
    for role in ("actor", "critic"):
        before = getattr(source.alg, role).state_dict()
        after = getattr(target.alg, role).state_dict()
        assert before.keys() == after.keys()
        for name, value in before.items():
            expected = value.clone()
            if name == "mlp.0.weight":
                expected[:, list(COLUMNS)] *= 1.5
            torch.testing.assert_close(after[name], expected, rtol=0, atol=0)
        assert getattr(target.alg, role).obs_normalizer.state_dict() == {}
        assert getattr(target.alg, role).obs_normalization is False


def _outputs(runner, tensor):
    obs = TensorDict({"policy": tensor, "critic": tensor}, batch_size=[tensor.shape[0]])
    with torch.inference_mode():
        mean = runner.alg.actor(obs, stochastic_output=False)
        head = runner.alg.actor.mlp(runner.alg.actor.get_latent(obs))
        std = head[..., 1, :].exp()
        value = runner.alg.critic(obs)
    return mean, std, value


def test_real_history_scale_migration_preserves_unclipped_physical_function_and_state(source_fixture, monkeypatch):
    f = source_fixture
    target, env, config, record, infos = _load_target(f, monkeypatch)
    assert type(target.alg.actor) is SemanticHistoryMLPModel and HISTORY_RHO == .9
    assert record["observation_scale_transition"]["columns"] == [210, 222]
    assert record["observation_scale_transition"]["factors"] == [1.5, 1.5]
    assert infos["training_rng_state"] == f.metadata["training_rng_state"]
    assert training.capture_training_rng_state(seed=1001) == f.metadata["training_rng_state"]
    _assert_only_reviewed_columns(f.source, target)
    assert tuple(infos[key] for key in COUNTERS) == (128, 1, 20)
    assert infos["stage_requested_decisions"] == f.metadata["stage_requested_decisions"]
    assert infos["new_mdp_origin_global_policy_decisions"] == 0
    assert config == f.metadata["runner_config"]
    assert (target.alg.gamma, target.alg.lam) == (.9985, .99)
    assert target.alg.storage.step == 0 and target.alg.transition.actions is None
    assert target.alg.optimizer.state_dict()["state"] == {}
    assert target.alg.learning_rate == 3e-5
    assert all(group["lr"] == 3e-5 for group in target.alg.optimizer.param_groups)
    assert f.metadata["optimizer_learning_rate"] == 1e-5
    assert env.core.calls == 0 and env.core.resets == 1
    assert infos["actor_parameter_sha256"] == training.parameter_hash(target.alg.actor)
    assert infos["critic_parameter_sha256"] == training.parameter_hash(target.alg.critic)
    evidence = infos["observation_scale_compensation_evidence"]
    assert evidence["source_actor_parameter_sha256"] == f.metadata["actor_parameter_sha256"]
    assert evidence["source_critic_parameter_sha256"] == f.metadata["critic_parameter_sha256"]
    assert evidence["compensated_actor_parameter_sha256"] != evidence["source_actor_parameter_sha256"]
    assert evidence["compensated_critic_parameter_sha256"] != evidence["source_critic_parameter_sha256"]
    assert evidence["normalizer_state_sha256"] == f.metadata["normalizer_state_sha256"]

    generator = torch.Generator().manual_seed(711)
    old = torch.randn(257, 324, generator=generator)
    physical = torch.linspace(-80., 80., old.shape[0])
    old[:, 210], old[:, 222] = physical / 4, physical.flip(0) / 4
    new = old.clone()
    new[:, 210], new[:, 222] = physical / 6, physical.flip(0) / 6
    source_cache = f.source.alg.actor.distribution._distribution
    target_cache = target.alg.actor.distribution._distribution
    rng = training.capture_training_rng_state(seed=1001)
    for expected, actual in zip(_outputs(f.source, old), _outputs(target, new)):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=3e-6)
    assert f.source.alg.actor.distribution._distribution is source_cache
    assert target.alg.actor.distribution._distribution is target_cache
    assert training.capture_training_rng_state(seed=1001) == rng
    assert torch.equal(old[:, 195:207], new[:, 195:207])


def test_immutable_initial_exact_resume_fresh_update_and_already6_no_double_compensation(
        source_fixture, tmp_path, monkeypatch):
    f = source_fixture
    source_bytes, source_sidecar = f.checkpoint.read_bytes(), f.checkpoint.with_name(f.checkpoint.stem + "_manifest.json").read_bytes()
    target, env, _, record, infos = _load_target(f, monkeypatch)
    initial = tmp_path / migration.v3_warm_start_checkpoint_name(record)
    training.save_semantic_checkpoint(target, initial, {**infos, "runtime_contract": f.new_contract})
    metadata = _metadata(initial)
    assert metadata["save_load_round_trip"] is True
    assert metadata["policy_contract"] == policy_contract(HISTORY_POLICY)
    assert metadata["observation_scale_compensation_evidence"] == infos["observation_scale_compensation_evidence"]
    fresh, _, _ = make(HISTORY_POLICY, initialize=False)
    reloaded = training.load_semantic_checkpoint(fresh, initial, contract=f.new_contract, seed=1001)
    for role in ("actor", "critic"):
        assert training.parameter_hash(getattr(fresh.alg, role)) == training.parameter_hash(getattr(target.alg, role))
    assert reloaded["observation_scale_compensation_evidence"] == infos["observation_scale_compensation_evidence"]
    no_op = migration.build_v3_warm_start_record(initial, f.new_contract)
    assert no_op["observation_scale_transition"]["changed"] is False
    again, _, _ = make(HISTORY_POLICY, initialize=False)
    second = training.load_semantic_checkpoint(again, initial, contract=f.new_contract, seed=1001, warm_start=no_op)
    assert second["observation_scale_compensation_evidence"]["compensation_applied"] is False
    for role in ("actor", "critic"):
        assert training.parameter_hash(getattr(again.alg, role)) == training.parameter_hash(getattr(target.alg, role))

    result = training.train_semantic(target, env, run_dir=tmp_path / "new_run", output_root=tmp_path / "new_output",
        stage="full_episode", decisions=128, contract=f.new_contract, seed=1001, resume_infos=infos)
    assert result["actual_policy_decisions"] == 128
    assert result["ppo_updates_this_run"] == 1 and result["optimizer_steps_this_run"] == 20
    assert result["finite_nonzero_gradient_observed"] is True
    final = _metadata(Path(result["checkpoints"][-1]["checkpoint"]))
    assert tuple(final[key] for key in COUNTERS) == (256, 2, 40)
    assert final["stage_requested_decisions"]["full_episode"] == 256
    assert final["new_mdp_origin_global_policy_decisions"] == 0
    assert final["observation_scale_compensation_evidence"] == infos["observation_scale_compensation_evidence"]
    assert final["save_load_round_trip"] is True
    assert f.checkpoint.read_bytes() == source_bytes
    assert f.checkpoint.with_name(f.checkpoint.stem + "_manifest.json").read_bytes() == source_sidecar


def test_previously_clipped_physical_history_is_not_a_global_equivalence_claim(source_fixture, monkeypatch):
    f = source_fixture
    target, _, _, _, _ = _load_target(f, monkeypatch)
    old, new = torch.zeros(2, 324), torch.zeros(2, 324)
    physical = torch.tensor([96., -96.])
    old[:, 210] = (physical / 4).clamp(-20, 20)
    new[:, 210] = (physical / 6).clamp(-20, 20)
    with torch.inference_mode():
        source_pre = f.source.alg.actor.mlp[0](old)
        target_pre = target.alg.actor.mlp[0](new)
    expected_difference = (physical / 4 - old[:, 210]).unsqueeze(-1) * f.source.alg.actor.mlp[0].weight[:, 210]
    torch.testing.assert_close(target_pre - source_pre, expected_difference, rtol=2e-5, atol=3e-6)
    assert torch.count_nonzero(target_pre - source_pre) > 0


@pytest.mark.parametrize("key,value", [
    ("columns", [209, 222]), ("columns", [210, 221]), ("factors", [1., 1.]),
    ("source_scales", [4, 6]), ("target_scales", [6, 7]), ("changed", False),
    ("experiment_id", "other"), ("observation_dimension", 325),
])
def test_unreviewed_scale_record_rejected_before_loading(source_fixture, monkeypatch, key, value):
    f = source_fixture
    record = migration.build_v3_warm_start_record(f.checkpoint, f.new_contract, project_root=f.root)
    bad = copy.deepcopy(record)
    bad["observation_scale_transition"][key] = value
    actual_validator = migration.build_v3_warm_start_record
    monkeypatch.setattr(migration, "build_v3_warm_start_record",
        lambda path, contract, **options: actual_validator(path, contract, project_root=f.root, **options))
    target, _, _ = make(HISTORY_POLICY, initialize=False)
    before = training.parameter_hash(target.alg.actor)
    with pytest.raises(RuntimeError, match="binding changed"):
        training.load_semantic_checkpoint(target, f.checkpoint, contract=f.new_contract, seed=1001, warm_start=bad)
    assert training.parameter_hash(target.alg.actor) == before


def test_source_fingerprint_failure_occurs_before_any_scale_mutation(source_fixture, monkeypatch):
    f = source_fixture
    record = migration.build_v3_warm_start_record(f.checkpoint, f.new_contract, project_root=f.root)
    actual_validator = migration.build_v3_warm_start_record
    monkeypatch.setattr(migration, "build_v3_warm_start_record",
        lambda path, contract, **options: actual_validator(path, contract, project_root=f.root, **options))
    actual_load = training.load_checkpoint_round_trip
    def corrupt_after_real_official_load(runner, checkpoint):
        infos = actual_load(runner, checkpoint)
        with torch.no_grad():
            runner.alg.actor.mlp[0].weight[0, 0] += 1.
        return infos
    monkeypatch.setattr(training, "load_checkpoint_round_trip", corrupt_after_real_official_load)
    def must_not_compensate(*args, **kwargs):
        pytest.fail("compensation ran before source fingerprint verification")
    monkeypatch.setattr(training, "_compensate_observation_input_scales", must_not_compensate)
    target, _, _ = make(HISTORY_POLICY, initialize=False)
    with pytest.raises(RuntimeError, match="source actor_parameter_sha256"):
        training.load_semantic_checkpoint(target, f.checkpoint, contract=f.new_contract, seed=1001, warm_start=record)


def test_ordinary_resume_does_not_silently_apply_new_schema(source_fixture):
    target, _, _ = make(HISTORY_POLICY, initialize=False)
    with pytest.raises(RuntimeError, match="contract mismatch"):
        training.load_semantic_checkpoint(target, source_fixture.checkpoint,
            contract=source_fixture.new_contract, seed=1001)
