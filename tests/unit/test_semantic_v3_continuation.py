from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rsl_rl")

from test_semantic_training import Core
from wlr50_clean.ppo import semantic_cli as cli
from wlr50_clean.ppo import semantic_migration as migration
from wlr50_clean.ppo.semantic_training import (
    SemanticRslAdapter, construct_semantic_runner, load_semantic_checkpoint,
    parameter_hash, save_semantic_checkpoint, semantic_runner_config, train_semantic,
    compare_warm_start_action,
)


class Core324(Core):
    def reset(self, *, seed=1001, options=None):
        return super().reset(seed=seed, options=options) + (0.0,) * 316

    def step(self, raw):
        step = super().step(raw)
        step.observation += (0.0,) * 316
        return step


@pytest.fixture(autouse=True)
def single_cpu_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def runner(version="v2"):
    env = SemanticRslAdapter(Core324(), seed=1001, device="cpu")
    env.cfg["semantic_version"] = version
    actual, _ = construct_semantic_runner(env, seed=1001, device="cpu")
    return actual, env


def contracts(tmp_path):
    names = ("stage_task_spec.yaml", "execution_profile.yaml", "reward_config.yaml",
             "observation_schema.json", "action_schema.json", "quality_score.yaml")
    old_files, new_files = {}, {}
    for name in names:
        for version, files in (("v2", old_files), ("v3", new_files)):
            relative = f"configs/ppo_semantic_{version}/{name}"
            destination = tmp_path / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((cli.PROJECT_ROOT / relative).read_bytes())
            files[relative] = migration.file_sha(destination)
    invariant = {"frozen_A_files": {"frozen": "f" * 64}, "physics_hz": 120.0,
                 "decision_hz": 15.0, "task_timeout_s": 200.0, "timeout_bootstrap": False,
                 "rsl_rl_version": "5.0.1", "local_runtime_versions": {"test": "CPU"}}
    old = {**invariant, "source_git_commit": "1" * 40, "files": old_files,
           "runtime_content_sha256": migration.digest(old_files)}
    new = {**invariant, "source_git_commit": "2" * 40, "files": {**old_files, **new_files}, "semantic_version": "v3"}
    new["runtime_content_sha256"] = migration.digest(new["files"])
    return old, new


def test_real_rsl_v2_networks_to_v3_fresh_adam_rollout_and_exact_resume(tmp_path, monkeypatch):
    old, new = contracts(tmp_path)
    source_runner, source_env = runner()
    trained = train_semantic(source_runner, source_env, run_dir=tmp_path / "source_run",
        output_root=tmp_path / "source_outputs", stage="smoke", decisions=128, contract=old, seed=1001)
    source = Path(trained["checkpoints"][-1]["checkpoint"])
    record = migration.build_v3_warm_start_record(source, new, project_root=tmp_path)
    assert record["exact_mdp_resume"] is False
    assert record["old_rollout_buffer_inherited"] is False
    # Use the actual validator at its normal project-root call seam.
    original = migration.build_v3_warm_start_record
    monkeypatch.setattr(migration, "build_v3_warm_start_record",
        lambda checkpoint, contract: original(checkpoint, contract, project_root=tmp_path))
    target, env = runner("v3")
    previous = load_semantic_checkpoint(target, source, contract=new, seed=1001, warm_start=record)
    assert parameter_hash(target.alg.actor) == parameter_hash(source_runner.alg.actor)
    assert parameter_hash(target.alg.critic) == parameter_hash(source_runner.alg.critic)
    assert target.alg.optimizer.state_dict()["state"] == {}
    assert target.alg.learning_rate == 3e-5
    assert all(group["lr"] == 3e-5 for group in target.alg.optimizer.param_groups)
    assert target.alg.storage.step == 0 and target.alg.transition.actions is None
    assert tuple(target.alg.storage.actions.shape) == (128, 1, 12)
    assert previous["stage_requested_decisions"] == dict.fromkeys(("smoke", "phase_suffix", "full_episode"), 0)
    assert previous["global_policy_decisions"] == 128
    assert previous["source_stage_requested_decisions"]["smoke"] == 128
    continued = train_semantic(target, env, run_dir=tmp_path / "v3_run", output_root=tmp_path / "v3_outputs",
        stage="smoke", decisions=128, contract=new, seed=1001, resume_infos=previous)
    assert continued["global_policy_decisions"] == 256
    assert continued["optimizer_steps_this_run"] == 20
    new_checkpoint = Path(continued["checkpoints"][-1]["checkpoint"])
    fresh, _ = runner("v3")
    restored = load_semantic_checkpoint(fresh, new_checkpoint, contract=new, seed=1001)
    assert restored["stage_requested_decisions"]["smoke"] == 128
    assert restored["new_mdp_origin_global_policy_decisions"] == 128
    assert restored["new_mdp_warm_start"] == record
    assert parameter_hash(fresh.alg.actor) == parameter_hash(target.alg.actor)


def test_new_mdp_rejects_changed_encoder_or_physics_and_corrupt_source(tmp_path):
    old, new = contracts(tmp_path)
    actual, _ = runner()
    source = tmp_path / "source.pt"
    save_semantic_checkpoint(actual, source, {"seed": 1001, "runtime_contract": old})
    bad = copy.deepcopy(new)
    bad["physics_hz"] = 60.0
    with pytest.raises(ValueError, match="physical/runtime"):
        migration.build_v3_warm_start_record(source, bad, project_root=tmp_path)
    path = tmp_path / "configs/ppo_semantic_v3/observation_schema.json"
    schema = json.loads(path.read_text())
    schema["feature_groups"][0]["scale"] = 2.0
    path.write_text(json.dumps(schema))
    new["files"]["configs/ppo_semantic_v3/observation_schema.json"] = migration.file_sha(path)
    new["runtime_content_sha256"] = migration.digest(new["files"])
    with pytest.raises(ValueError, match="preprocessing"):
        migration.build_v3_warm_start_record(source, new, project_root=tmp_path)
    source.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        migration.build_v3_warm_start_record(source, new, project_root=tmp_path)


@pytest.mark.parametrize("phase", ["P06", "P07", "P08", "P09", "P10", "P11", "P12", "P13"])
def test_v3_suffix_cli_and_topology_bind_real_prefix_request(tmp_path, monkeypatch, phase):
    from wlr50_clean.ppo.semantic_prefix import PrefixRequest, sampling_label
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cli, "OUTPUT_ROOT", tmp_path / "outputs/ppo_semantic_v2")
    source = cli.OUTPUT_ROOT / "checkpoints/history/source.pt"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"only path validation")
    source.with_name("source_manifest.json").write_text(json.dumps({"stage_requested_decisions": {"smoke": 10000}}))
    args = cli.parser().parse_args(["train", "--run-dir", str(tmp_path / "runs/ppo_semantic_v3/train/epoch"),
        "--expected-head", "a" * 40, "--semantic-version", "v3", "--from-phase", phase,
        "--new-mdp-warm-start", "--checkpoint", str(source), "--stage", "phase_suffix", "--decisions", "128"])
    cli.validate_request(args)
    request = PrefixRequest(target_phase=phase)
    sampling = sampling_label(request)
    metadata = {"semantic_version": "v3", "sampling": sampling,
        "curriculum_epoch": {"prefix_request": request.as_dict()},
        "execution_topology": migration.continuation_topology(sampling, request.as_dict())}
    assert migration.source_num_envs(metadata) == 1
    metadata["execution_topology"]["reset_sampling"] = "P01_only"
    with pytest.raises(ValueError, match="malformed"):
        migration.source_num_envs(metadata)


@pytest.mark.parametrize("extra", [[], ["--from-phase", "P06"], ["--num-envs", "8"], ["--new-mdp-warm-start"]])
def test_v3_cannot_silently_reinitialize_actor_or_use_unimplemented_vector(tmp_path, monkeypatch, extra):
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    args = cli.parser().parse_args(["train", "--run-dir", str(tmp_path / "runs/ppo_semantic_v3/train/epoch"),
        "--expected-head", "a" * 40, "--semantic-version", "v3", *extra])
    with pytest.raises(ValueError):
        cli.validate_request(args)


def test_v2_default_configuration_and_paths_are_unchanged():
    assert cli.parser().get_default("semantic_version") == "v2"
    assert cli.version_paths("v2")[:2] == (cli.RUNS_ROOT, cli.OUTPUT_ROOT)
    assert semantic_runner_config(seed=1001)["algorithm"]["learning_rate"] == 3e-4
    assert semantic_runner_config(seed=1001, semantic_version="v3")["algorithm"]["learning_rate"] == 3e-5


def test_same_state_action_comparison_uses_real_actor_projectors_without_sampling_or_history_mutation():
    from test_semantic_observation_reward_env import _frame
    actual, env = runner("v3")
    env.core.frame = _frame(stage="P06")
    with torch.no_grad():
        layers = [layer for layer in actual.alg.actor.modules() if isinstance(layer, torch.nn.Linear)]
        layers[-1].bias.fill_(0.2)
    before_rng = torch.get_rng_state().clone()
    before_actor = parameter_hash(actual.alg.actor)
    mode = actual.alg.actor.training
    record = compare_warm_start_action(actual, env,
        old_execution_profile=cli.version_paths("v2")[2] / "execution_profile.yaml",
        new_execution_profile=cli.version_paths("v3")[2] / "execution_profile.yaml")
    assert record["phase_id"] == "P06" and record["observation_dimension"] == 324
    assert any(record["new_minus_old"]["scaled_residual_full12"])
    assert record["old"]["bounded_residual_full12"] == record["new"]["bounded_residual_full12"]
    assert record["actual_environment_or_bridge_history_modified"] is False
    assert actual.alg.storage.step == 0 and actual.alg.transition.actions is None
    assert torch.equal(before_rng, torch.get_rng_state())
    assert parameter_hash(actual.alg.actor) == before_actor and actual.alg.actor.training == mode
    assert env.core.calls == 0
