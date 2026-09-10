"""Exact HISTORY372 instrumentation resume and actual runtime identity tests.

No top-level Torch import. The explicit CPU test may run only after Isaac exits,
with CUDA_VISIBLE_DEVICES empty; it is not a physical task success test.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest

from wlr50_clean.ppo import semantic_cli as cli, semantic_migration as migration
from wlr50_clean.ppo import semantic_policy_distribution as policies
from wlr50_clean.ppo import semantic_training as training
from wlr50_clean.ppo.semantic_transfer_roles import ROLE_OBSERVATION_LAYOUT as LAYOUT

NAMES = ("stage_task_spec.yaml", "execution_profile.yaml", "reward_config.yaml",
         "observation_schema.json", "action_schema.json", "quality_score.yaml")
CODE = "src/wlr50_clean/ppo/semantic_cli.py"
NAMESPACE = "ppo_fsm_reference_p09_stable_v2"


def refresh(contract):
    contract["runtime_content_sha256"] = migration.digest(contract["files"])


def changed(old, new):
    return sorted(key for key in old["files"].keys() | new["files"].keys()
                  if old["files"].get(key) != new["files"].get(key))


def save_metadata(checkpoint, metadata):
    checkpoint.with_name(checkpoint.stem + "_manifest.json").write_text(
        json.dumps(metadata), encoding="utf-8")


def fixture372(root):
    files, selected = {}, {}
    for name in NAMES:
        relative = f"configs/{NAMESPACE}/{name}"
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((migration.PROJECT_ROOT / relative).read_bytes())
        files[relative] = migration.file_sha(target)
        selected[name] = {"path": relative, "sha256": files[relative]}
    target = root / CODE
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("# exact source fixture\n", encoding="utf-8")
    files[CODE] = migration.file_sha(target)
    old = {"source_git_commit": "1" * 40, "files": files,
           "semantic_version": "v3", "experiment_id": "fsm_reference_p09_stable_v2",
           "selected_configuration": selected, "physics_hz": 120., "decision_hz": 15.,
           "task_timeout_s": 200., "timeout_bootstrap": False,
           "frozen_A_files": {"reference": "a" * 64}}
    refresh(old)
    new = copy.deepcopy(old)
    new["source_git_commit"] = "2" * 40
    target.write_text("# reviewed source instrumentation fixture\n", encoding="utf-8")
    new["files"][CODE] = migration.file_sha(target)
    refresh(new)
    checkpoint = root / "source.pt"
    checkpoint.write_bytes(b"metadata-only fixture; never passed to torch.load")
    metadata = {
        "schema": "wlr50_clean.semantic_checkpoint.v1",
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": migration.file_sha(checkpoint),
        "save_load_round_trip": True, "runtime_contract": old,
        "semantic_version": "v3", "seed": 1001,
        "sampling": "P01_full_task_only_initial_version",
        "execution_topology": migration.continuation_topology(
            "P01_full_task_only_initial_version", None, observation_layout=LAYOUT),
        "runner_config": training.semantic_runner_config(seed=1001, device="cpu",
            semantic_version="v3", policy_version=policies.HISTORY_POLICY, observation_layout=LAYOUT),
        "policy_contract": policies.policy_contract(policies.HISTORY_POLICY, observation_layout=LAYOUT),
    }
    save_metadata(checkpoint, metadata)
    return checkpoint, old, new


def plan(root, source, old, new, **kwargs):
    return migration.build_migration_plan(source, new, allowed_changed_files=changed(old, new),
        reason="Reviewed logging and exact372-resume metadata only; no control change",
        project_root=root, **kwargs)


def test_exact372_instrumentation_binds_schema_and_preserves_learning_contract(tmp_path):
    source, old, new = fixture372(tmp_path)
    before = source.read_bytes()
    record = plan(tmp_path, source, old, new)
    factor = record["instrumentation_observation_contract"]
    canonical = policies.policy_contract(policies.HISTORY_POLICY, observation_layout=LAYOUT)
    assert factor["source_policy_contract"] == factor["target_policy_contract"] == canonical
    assert factor["selected_configuration"] == old["selected_configuration"]
    assert record["observation_dimension"] == 372 and record["action_dimension"] == 12
    assert factor["parameter_mapping"] is None and factor["num_envs"] == 1
    assert record["preserve_actor_critic_optimizer_normalizer_rng_and_budget"] is True
    assert record["discard_old_rollout_storage"] is True
    assert "video_instrumentation_factor" not in record and "execution_factor" not in record
    path = tmp_path / "resume_plan.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    assert migration.validate_migration_plan(source, new, path, project_root=tmp_path)[
        "instrumentation_observation_contract"] == factor
    assert source.read_bytes() == before


@pytest.mark.parametrize("fault", [
    "missing_policy_contract", "wrong_layout", "mixed_runner", "topology",
    "selected_path", "selected_extra", "schema_marker", "changed_schema", "protected_runtime",
])
def test_372_never_inferred_from_width_or_used_as_general_permission(tmp_path, fault):
    source, old, new = fixture372(tmp_path)
    metadata = migration.checkpoint_metadata(source)
    if fault == "missing_policy_contract":
        del metadata["policy_contract"]
    elif fault == "wrong_layout":
        metadata["policy_contract"]["observation_layout"] = "unreviewed372"
    elif fault == "mixed_runner":
        metadata["runner_config"]["actor"].pop("observation_layout")
    elif fault == "topology":
        metadata["execution_topology"]["num_envs"] = 8
    elif fault in ("selected_path", "selected_extra"):
        for contract in (old, new):
            if fault == "selected_path":
                contract["selected_configuration"]["observation_schema.json"]["path"] = "unverified/schema.json"
            else:
                contract["selected_configuration"]["extra"] = {"path": "extra", "sha256": "0" * 64}
    elif fault in ("schema_marker", "changed_schema"):
        relative = f"configs/{NAMESPACE}/observation_schema.json"
        schema_path = tmp_path / relative
        schema = json.loads(schema_path.read_text())
        if fault == "schema_marker":
            schema.pop("transfer_role_features_version")
        else:
            schema["clip"] += .5
        schema_path.write_text(json.dumps(schema), encoding="utf-8")
        new["files"][relative] = migration.file_sha(schema_path)
        new["selected_configuration"]["observation_schema.json"]["sha256"] = new["files"][relative]
        if fault == "schema_marker":
            old["files"][relative] = new["files"][relative]
            old["selected_configuration"]["observation_schema.json"]["sha256"] = new["files"][relative]
    elif fault == "protected_runtime":
        new["files"]["src/wlr50_clean/ppo/semantic_backend.py"] = "b" * 64
    refresh(old)
    refresh(new)
    metadata["runtime_contract"] = old
    save_metadata(source, metadata)
    with pytest.raises(ValueError):
        plan(tmp_path, source, old, new)


@pytest.mark.parametrize("other_factor", [
    "prior_evidence", "qualification_evidence", "evaluator_review", "execution_evidence", "video_review",
])
def test_role_instrumentation_is_a_separate_boundary(tmp_path, other_factor):
    source, old, new = fixture372(tmp_path)
    with pytest.raises(ValueError, match="cannot mix"):
        plan(tmp_path, source, old, new, **{other_factor: {}})


def test_cli_preflight_preserves_explicit_history372_layout(tmp_path, monkeypatch):
    source, old, new = fixture372(tmp_path)
    path = tmp_path / "resume_plan.json"
    path.write_text(json.dumps(plan(tmp_path, source, old, new)), encoding="utf-8")
    real_validate = migration.validate_migration_plan
    monkeypatch.setattr(migration, "validate_migration_plan", lambda cp, contract, p:
        real_validate(cp, contract, p, project_root=tmp_path))
    monkeypatch.setattr(cli, "_request_paths", lambda args:
        (tmp_path / "runs", tmp_path / "outputs", tmp_path / f"configs/{NAMESPACE}"))
    args = NS(checkpoint=source, resume_migration=path, new_mdp_warm_start=False,
        policy_distribution_migration=False, target_policy_version=None,
        seed=1001, device="cpu", semantic_version="v3", num_envs=1, command="train")
    cli._preflight_checkpoint(args, new)
    assert args._observation_layout == LAYOUT and args._policy_version == policies.HISTORY_POLICY
    assert args._warm_start_record is None and args._policy_migration_record is None


def test_identity_reads_actual_objects_including_slot_frame_without_control_calls(tmp_path, monkeypatch):
    class Sentinel:
        def step(self, *a, **k):
            raise AssertionError("logger must not step")
        read = reset = step
        apply_full12 = advance = step
    class Provider(Sentinel):
        _source_normal_bias = _start_source_motion = Sentinel.step
    class Robot(Sentinel):
        @property
        def stage(self):
            raise AssertionError("configured USD path must not inspect the USD stage")
    @dataclass(slots=True)
    class Frame:
        physics_tick: int = 192
        state_id: str = "P06"
    @dataclass(slots=True)
    class Scene:
        robot: object
    @dataclass(slots=True)
    class Dependencies:
        create_scene: object
        create_sensing_backends: object
        adapter_from_scene: object
        reader_from_scene: object
        controller_from_paths: object
        capture_reset_state: object
        reset_scene: object
    # Capture the actual globals bound to this synthetic apply method; no
    # function is executed merely to obtain its module/source identity.
    monkeypatch.setattr(sys.modules[__name__], "bounded_drive_feedback_step", Sentinel.step, raising=False)
    monkeypatch.setattr(sys.modules[__name__], "_full12_drive_feedback_bias", Sentinel.step, raising=False)
    provider = Provider()
    provider._reference_nominal = True
    provider._source_motion = Sentinel()
    provider._reference_fsm_spec = NS(path=tmp_path / "actual_loaded_fsm.yaml")
    active = NS(nominal_provider=provider)
    mapper, adapter, reader = Sentinel(), Sentinel(), Sentinel()
    adapter.servo_target_mapper = mapper
    reader.contact_backend, reader.geometry_backend = Sentinel(), Sentinel()
    outer = NS(_semantic=active, teacher=None, mode="READY")
    robot = Robot()
    robot.cfg = NS(spawn=NS(usd_path=str(tmp_path / "not_read_or_opened_robot.usd")))
    paths = {name: tmp_path / filename for name, filename in (
        ("execution_profile_path", "effective_profile.yaml"),
        ("task_spec_path", "effective_task.yaml"),
        ("motion_contract_path", "effective_motion.json"),
        ("fsm_path", "backend_fsm.yaml"))}
    dependencies = Dependencies(*([Sentinel.step] * 7))
    backend = NS(_controller=outer, _adapter=adapter, _reader=reader,
        _authoritative_frame=Frame(), _scene=Scene(robot),
        _dependencies=dependencies, _atomic_apply=adapter.step, **paths)
    core = NS(backend=backend)
    args = NS(command="train", from_phase="P06", experiment_id="fsm_reference_p09_stable_v2", run_dir=tmp_path)
    before_objects = (backend._controller, backend._adapter, backend._reader, backend._authoritative_frame)
    before_modules = dict(sys.modules)
    result = cli._live_runtime_identity(core, args, {}, boundary="synthetic_unit_test")
    assert result["backend_initialized"] is True and result["physics_tick"] == 192
    assert result["nominal_provider"]["module_file"] == sys.modules[Provider.__module__].__file__
    assert result["mapper"]["class"].endswith("Sentinel")
    assert result["loaded_source_fsm_spec_path"] == str(provider._reference_fsm_spec.path)
    assert result["effective_configuration_paths"] == {key: str(value) for key, value in paths.items()}
    assert all(os.path.isabs(value) for value in result["effective_configuration_paths"].values())
    asset = result["configured_robot_asset"]
    assert asset["configured_usd_path_as_stored"] == robot.cfg.spawn.usd_path
    assert asset["configured_usd_absolute_path"] == robot.cfg.spawn.usd_path
    assert asset["source"] == "actual_backend._scene.robot.cfg.spawn.usd_path"
    assert asset["USD_stage_inspected"] is False and asset["resolved_dependency_proof"] is False
    assert result["scene"]["class"].endswith("Scene") and result["scene_robot"]["class"].endswith("Robot")
    for factory in result["dependency_factory_callables"].values():
        assert factory["module_file"] == sys.modules[__name__].__file__
        assert factory["source_file"] == Sentinel.step.__code__.co_filename
        assert factory["source_firstlineno"] == Sentinel.step.__code__.co_firstlineno
    for method in result["active_feedback_method_callables"].values():
        assert method["source_file"] == Sentinel.step.__code__.co_filename
        assert method["qualname"] == Sentinel.step.__qualname__
    assert result["recovery"] is None
    assert result["recovery_absence"] == "deliberately_not_used_by_successful_fsm_derived_semantic_N"
    assert result["sys_path"] == sys.path and result["python_executable"] == sys.executable
    assert result["control_calls_performed_by_logger"] == result["new_imports_performed_by_logger"] == 0
    assert before_objects == (backend._controller, backend._adapter, backend._reader, backend._authoritative_frame)
    assert before_modules == dict(sys.modules)
    cli._save_live_runtime_identity(core, args, {}, boundary="synthetic_unit_test")
    with pytest.raises(FileExistsError):
        cli._save_live_runtime_identity(core, args, {}, boundary="synthetic_unit_test")
    # Incomplete fixtures must not be represented as initialized live proof.
    missing = cli._live_runtime_identity(NS(), args, {}, boundary="synthetic_unit_test")
    assert not missing["backend_initialized"]
    assert all(value is None for value in missing["effective_configuration_paths"].values())
    assert missing["configured_robot_asset"]["configured_usd_absolute_path"] is None
    assert all(value is None for value in missing["dependency_factory_callables"].values())
    # Preserve configured URI identity without fabricating a local or resolved path.
    robot.cfg.spawn.usd_path = "omniverse://configured-only/robot.usd"
    uri = cli._live_runtime_identity(core, args, {}, boundary="synthetic_unit_test")["configured_robot_asset"]
    assert uri["configured_usd_path_as_stored"] == robot.cfg.spawn.usd_path
    assert uri["configured_usd_absolute_path"] is None and uri["resolved_dependency_proof"] is False
    robot.cfg.spawn.usd_path = ""
    empty = cli._live_runtime_identity(core, args, {}, boundary="synthetic_unit_test")["configured_robot_asset"]
    assert empty["configured_usd_absolute_path"] is None


def test_real_CPU_history372_exact_reload_preserves_nonempty_Adam_rng_counters_and_kernel(tmp_path, monkeypatch):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    import torch
    from tensordict import TensorDict
    from test_semantic_v3_continuation import Core324

    class Core372(Core324):
        def reset(self, **kwargs):
            return super().reset(**kwargs) + (0.,) * 48
        def step(self, raw):
            result = super().step(raw)
            values = list(result.observation) + [0.] * 48
            values[195:207] = [max(-20., min(20., float(x))) for x in raw]
            result.observation = tuple(values)
            return result

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        _, old, new = fixture372(tmp_path)
        training.seed_training_rngs(1001)
        env = training.SemanticRslAdapter(Core372(), seed=1001, device="cpu")
        env.cfg["semantic_version"] = "v3"
        runner, _ = training.construct_semantic_runner(env, seed=1001, device="cpu",
            policy_version=policies.HISTORY_POLICY, observation_layout=LAYOUT, initialize_actor=False)
        trained = training.train_semantic(runner, env, run_dir=tmp_path / "cpu_source_run",
            output_root=tmp_path / "cpu_source_outputs", stage="full_episode",
            decisions=128, contract=old, seed=1001)
        assert runner.alg.optimizer.state_dict()["state"]
        source = Path(trained["checkpoints"][-1]["checkpoint"])
        metadata = migration.checkpoint_metadata(source)
        record = plan(tmp_path, source, old, new)
        path = tmp_path / "resume_plan.json"
        path.write_text(json.dumps(record), encoding="utf-8")
        real_validate = migration.validate_migration_plan
        monkeypatch.setattr(migration, "validate_migration_plan", lambda cp, contract, p:
            real_validate(cp, contract, p, project_root=tmp_path))
        verified = migration.validate_migration_plan(source, new, path)
        target_env = training.SemanticRslAdapter(Core372(), seed=1001, device="cpu")
        target_env.cfg["semantic_version"] = "v3"
        target, _ = training.construct_semantic_runner(target_env, seed=1001, device="cpu",
            policy_version=policies.HISTORY_POLICY, observation_layout=LAYOUT, initialize_actor=False)
        infos = training.load_semantic_checkpoint(target, source, contract=new, seed=1001, migration=verified)
        assert target.alg.storage.step == 0 and target.alg.transition.actions is None
        for name in ("actor", "critic"):
            expected, actual = getattr(runner.alg, name).state_dict(), getattr(target.alg, name).state_dict()
            assert expected.keys() == actual.keys()
            assert all(torch.equal(expected[k], actual[k]) for k in expected)
        assert training.state_hash(target.alg.optimizer.state_dict()) == metadata["optimizer_state_sha256"]
        assert training.state_hash(training._normalizers(target)) == metadata["normalizer_state_sha256"]
        assert training.capture_training_rng_state(seed=1001) == metadata["training_rng_state"]
        assert training.optimizer_learning_rate(target) == metadata["optimizer_learning_rate"]
        for key in ("global_policy_decisions", "ppo_updates", "optimizer_steps",
                    "stage_requested_decisions", "training_rng_state"):
            assert infos[key] == metadata[key]
        for history in (0., .7):
            observation = [0.] * 372
            observation[195:207] = [history] * 12
            tensor = torch.tensor([observation], dtype=torch.float32)
            inputs = TensorDict({"policy": tensor, "critic": tensor.clone()}, batch_size=[1])
            with torch.inference_mode():
                assert torch.equal(runner.alg.actor(inputs, stochastic_output=False),
                                   target.alg.actor(inputs, stochastic_output=False))
        target.alg.storage.step = 1
        with pytest.raises(RuntimeError, match="partial old rollout"):
            training.load_semantic_checkpoint(target, source, contract=new, seed=1001, migration=verified)
        target.alg.storage.step = 0
    finally:
        torch.set_num_threads(previous_threads)
