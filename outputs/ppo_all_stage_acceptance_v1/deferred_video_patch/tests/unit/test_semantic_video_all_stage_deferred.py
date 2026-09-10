"""DEFERRED DRAFT: run only after reviewed production application, never live Isaac.

Imports the applied package, NOT these mirrored source files. Synthetic routing
and real CPU state reload are not a physics/video-success certification.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from wlr50_clean.ppo import semantic_cli, semantic_migration as migration
from wlr50_clean.ppo import semantic_video as video, semantic_video_cli as cli
from wlr50_clean.ppo import semantic_policy_distribution as policies
from wlr50_clean.ppo import semantic_training as training
from wlr50_clean.ppo.semantic_transfer_roles import ROLE_OBSERVATION_LAYOUT as LAYOUT
from test_semantic_video_migration import fixture, refresh, delta, REVIEW


NAMES = ("stage_task_spec.yaml", "execution_profile.yaml", "reward_config.yaml",
         "observation_schema.json", "action_schema.json", "quality_score.yaml")


def contracts372(root):
    source, old, new = fixture(root, existing=True)
    selected = {}
    for name in NAMES:
        relative = f"configs/ppo_all_stage_acceptance_v1/{name}"
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((migration.PROJECT_ROOT / relative).read_bytes())
        sha = migration.file_sha(destination)
        old["files"][relative] = new["files"][relative] = sha
        selected[name] = {"path": relative, "sha256": sha}
    for contract in (old, new):
        contract.update(semantic_version="v3", experiment_id="all_stage_acceptance_v1",
                        selected_configuration=copy.deepcopy(selected), observation_dimension=372)
        refresh(contract)
    metadata = migration.checkpoint_metadata(source)
    metadata.update(runtime_contract=old, semantic_version="v3", seed=1001,
        sampling="P01_full_task_only_initial_version",
        execution_topology=migration.continuation_topology(
            "P01_full_task_only_initial_version", None, observation_layout=LAYOUT),
        runner_config=training.semantic_runner_config(seed=1001, device="cpu", semantic_version="v3",
            policy_version=policies.HISTORY_POLICY, observation_layout=LAYOUT),
        policy_contract=policies.policy_contract(policies.HISTORY_POLICY, observation_layout=LAYOUT))
    source.with_name(source.stem + "_manifest.json").write_text(json.dumps(metadata), encoding="utf-8")
    return source, old, new


def plan(root, source, old, new):
    return migration.build_migration_plan(source, new, allowed_changed_files=delta(old, new),
        reason="Deferred exact same-layout video integration", video_review=REVIEW, project_root=root)


def test_explicit_experiment_routes_capture_and_replay_without_changing_default():
    default = video.video_configuration("v3")
    selected = video.video_configuration("v3", experiment_id="all_stage_acceptance_v1")
    assert all(path.parent.name == "ppo_semantic_v3" for path in default.values())
    assert all(path.parent.name == "ppo_all_stage_acceptance_v1" for path in selected.values())
    source = {"semantic_version": "v3", "experiment_id": "all_stage_acceptance_v1",
        "runtime_contract": {"semantic_version": "v3", "experiment_id": "all_stage_acceptance_v1"},
        "evaluation_configuration": {key: video.file_record(path) for key, path in selected.items()}}
    assert video.validate_video_configuration(source) == selected
    source["evaluation_configuration"]["task_spec_path"] = video.file_record(default["task_spec_path"])
    with pytest.raises(video.SemanticVideoError, match="configuration"):
        video.validate_video_configuration(source)
    source["experiment_id"] = "transfer_roles_v1"
    with pytest.raises(video.SemanticVideoError, match="experiment"):
        video.validate_video_configuration(source)
    with pytest.raises(ValueError):
        video.video_configuration("v2", experiment_id="all_stage_acceptance_v1")


def test_A_only_reader_dependency_changes_and_BC_select_same_experiment(monkeypatch):
    from wlr50_clean.ppo import isaac_fsm_backend as frozen, residual_direct_env
    from wlr50_clean.ppo import semantic_backend, semantic_env, semantic_physical_sensing
    @dataclass
    class Dependencies:
        reader_from_scene: object
        controller_sentinel: object
    sentinel = object()
    original = Dependencies(lambda *a, **k: "legacy", sentinel)
    monkeypatch.setattr(frozen, "_load_live_dependencies", lambda: original)
    monkeypatch.setattr(frozen, "IsaacFSMBackend", lambda app, **kw: NS(options=kw))
    monkeypatch.setattr(residual_direct_env, "ResidualEpisodeEnv", lambda backend, **kw: NS(backend=backend, options=kw))
    monkeypatch.setattr(semantic_physical_sensing.SemanticSensorReader, "from_live_scene",
                        lambda scene, adapter, **kw: (scene, adapter, kw))
    current = cli.build_video_core(None, role="A", semantic_version="v3", experiment_id="all_stage_acceptance_v1")
    actual = current.backend.options["dependencies"]
    assert actual is not original and actual.controller_sentinel is sentinel
    assert actual.reader_from_scene(1, 2, 3) == (1, 2, {"backends": 3})
    assert original.reader_from_scene() == "legacy"
    old = cli.build_video_core(None, role="A", semantic_version="v3")
    assert old.backend.options == {"audit_actuator_target_effect": True}
    monkeypatch.setattr(semantic_backend, "SemanticIsaacBackend", lambda app, **kw: NS(options=kw))
    monkeypatch.setattr(semantic_env, "SemanticEpisodeEnv", lambda backend, **kw: NS(backend=backend, options=kw))
    for role in ("B", "C"):
        core = cli.build_video_core(None, role=role, semantic_version="v3", experiment_id="all_stage_acceptance_v1")
        assert core.backend.options["task_spec_path"].parent.name == "ppo_all_stage_acceptance_v1"
        assert core.options["observation_schema_path"].parent.name == "ppo_all_stage_acceptance_v1"


def test_video372_plan_is_same_layout_not_tensor_mapping(tmp_path):
    source, old, new = contracts372(tmp_path)
    record = plan(tmp_path, source, old, new)
    assert (record["observation_dimension"], record["observation_layout"]) == (372, LAYOUT)
    contract = record["video_instrumentation_factor"]["observation_contract"]
    assert contract["source_policy_contract"] == contract["target_policy_contract"]
    assert contract["parameter_mapping"] == "none_exact_same_layout"
    assert record["preserve_actor_critic_optimizer_normalizer_rng_and_budget"] is True
    assert record["geometric_factor"] is None and "execution_factor" not in record
    saved = tmp_path / "video_plan.json"
    saved.write_text(json.dumps(record), encoding="utf-8")
    assert migration.validate_migration_plan(source, new, saved, project_root=tmp_path)["observation_dimension"] == 372


def test_preflight_uses_verified372_layout_for_the_exact_runner_comparison(tmp_path, monkeypatch):
    source, old, new = contracts372(tmp_path)
    saved = tmp_path / "video_plan.json"
    saved.write_text(json.dumps(plan(tmp_path, source, old, new)), encoding="utf-8")
    real_validate = migration.validate_migration_plan
    monkeypatch.setattr(migration, "validate_migration_plan", lambda checkpoint, contract, path:
        real_validate(checkpoint, contract, path, project_root=tmp_path))
    monkeypatch.setattr(semantic_cli, "PROJECT_ROOT", tmp_path)
    args = semantic_cli.parser().parse_args(["eval", "--run-dir", str(tmp_path / "video_run"),
        "--expected-head", new["source_git_commit"], "--semantic-version", "v3",
        "--experiment-id", "all_stage_acceptance_v1", "--seed", "4001", "--device", "cpu",
        "--checkpoint", str(source), "--resume-migration", str(saved),
        "--mode", "semantic_residual_eval", "--no-headless"])
    semantic_cli._preflight_checkpoint(args, new)
    assert args._observation_layout == LAYOUT and args._policy_version == policies.HISTORY_POLICY
    assert args._migration_record["observation_dimension"] == 372
    assert args._warm_start_record is None and args._policy_migration_record is None


@pytest.mark.parametrize("fault", ["missing_contract", "wrong_layout", "mixed_runner", "target_schema",
                                 "schema_marker", "selected_path", "selected_extra", "mixed_factor"])
def test_video372_rejects_unverified_layout_or_nonvideo_changes(tmp_path, fault):
    source, old, new = contracts372(tmp_path)
    sidecar = source.with_name(source.stem + "_manifest.json")
    metadata = json.loads(sidecar.read_text())
    if fault == "missing_contract":
        metadata.pop("policy_contract")
    elif fault == "wrong_layout":
        metadata["policy_contract"]["observation_layout"] = "unknown"
    elif fault == "mixed_runner":
        metadata["runner_config"]["algorithm"]["gamma"] = .995
    elif fault == "target_schema":
        new["selected_configuration"]["observation_schema.json"]["sha256"] = "0" * 64
    elif fault == "schema_marker":
        relative = "configs/ppo_all_stage_acceptance_v1/observation_schema.json"
        path = tmp_path / relative
        schema = json.loads(path.read_text())
        schema["transfer_role_features_version"] = "unverified_role_layout"
        path.write_text(json.dumps(schema), encoding="utf-8")
        sha = migration.file_sha(path)
        # Even matching dimensions and internally consistent source/target file
        # receipts do not permit an unrecognized role marker.
        for runtime in (old, new):
            runtime["files"][relative] = sha
            runtime["selected_configuration"]["observation_schema.json"]["sha256"] = sha
            refresh(runtime)
        metadata["runtime_contract"] = copy.deepcopy(old)
    elif fault in ("selected_path", "selected_extra"):
        for runtime in (old, new):
            if fault == "selected_path":
                runtime["selected_configuration"]["observation_schema.json"]["path"] = "unverified/schema.json"
            else:
                runtime["selected_configuration"]["unverified"] = {"path": "extra", "sha256": "0" * 64}
        metadata["runtime_contract"] = copy.deepcopy(old)
    sidecar.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError):
        migration.build_migration_plan(source, new, allowed_changed_files=delta(old, new),
            reason="must reject", video_review=REVIEW, project_root=tmp_path,
            **({"execution_evidence": {}} if fault == "mixed_factor" else {}))


def test_real_CPU_history372_video_exact_reload_retains_Adam_rng_and_kernel(tmp_path, monkeypatch):
    # Future execution only. Never reuse real CUDA-source RNG on a CPU fixture.
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    import torch
    from tensordict import TensorDict
    from test_semantic_v3_continuation import Core324
    class Core372(Core324):
        def reset(self, **kw):
            return super().reset(**kw) + (0.,) * 48
        def step(self, raw):
            result = super().step(raw)
            values = list(result.observation) + [0.] * 48
            values[195:207] = [max(-20., min(20., float(x))) for x in raw]
            result.observation = tuple(values)
            return result
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        _, old, new = contracts372(tmp_path)
        training.seed_training_rngs(1001)
        env = training.SemanticRslAdapter(Core372(), seed=1001, device="cpu")
        env.cfg["semantic_version"] = "v3"
        runner, _ = training.construct_semantic_runner(env, seed=1001, device="cpu",
            policy_version=policies.HISTORY_POLICY, observation_layout=LAYOUT, initialize_actor=False)
        trained = training.train_semantic(runner, env, run_dir=tmp_path / "cpu_source_run",
            output_root=tmp_path / "cpu_source_outputs", stage="full_episode", decisions=128,
            contract=old, seed=1001)
        assert runner.alg.optimizer.state_dict()["state"]
        source = Path(trained["checkpoints"][-1]["checkpoint"])
        metadata = migration.checkpoint_metadata(source)
        record = plan(tmp_path, source, old, new)
        saved = tmp_path / "video_plan.json"
        saved.write_text(json.dumps(record), encoding="utf-8")
        real_validate = migration.validate_migration_plan
        monkeypatch.setattr(migration, "validate_migration_plan", lambda checkpoint, contract, path:
            real_validate(checkpoint, contract, path, project_root=tmp_path))
        verified = migration.validate_migration_plan(source, new, saved)
        loaded = []
        real_construct = cli.construct_semantic_runner
        def construct(*a, **kw):
            result = real_construct(*a, **kw)
            loaded.append(result[0])
            return result
        monkeypatch.setattr(cli, "construct_semantic_runner", construct)
        args = NS(checkpoint=source, seed=4001, semantic_version="v3", device="cpu",
            _policy_version=policies.HISTORY_POLICY, _observation_layout=LAYOUT, _migration_record=verified)
        observation = [0.] * 372
        action, proof, unchanged = cli.checkpoint_loader(args, new)(tuple(observation))
        target = loaded[0]
        assert target.alg.storage.step == 0 and target.alg.transition.actions is None
        assert proof["optimizer_updates"] == 0 and proof["observation_dimension"] == 372
        for part in ("actor", "critic"):
            expected, actual = getattr(runner.alg, part).state_dict(), getattr(target.alg, part).state_dict()
            assert expected.keys() == actual.keys()
            assert all(torch.equal(expected[k], actual[k]) for k in expected)
        assert training.state_hash(target.alg.optimizer.state_dict()) == metadata["optimizer_state_sha256"]
        assert training.state_hash(training._normalizers(target)) == metadata["normalizer_state_sha256"]
        assert training.capture_training_rng_state(seed=1001) == metadata["training_rng_state"]
        assert training.optimizer_learning_rate(target) == metadata["optimizer_learning_rate"]
        assert proof["saved_global_policy_decisions"] == metadata["global_policy_decisions"]
        for history in (0., .7):
            observation[195:207] = [history] * 12
            tensor = torch.tensor([observation], dtype=torch.float32)
            inputs = TensorDict({"policy": tensor, "critic": tensor.clone()}, batch_size=[1])
            with torch.inference_mode():
                expected = tuple(runner.alg.actor(inputs, stochastic_output=False)[0].tolist())
            assert action(observation, 1) == expected
            unchanged()
    finally:
        torch.set_num_threads(previous_threads)
