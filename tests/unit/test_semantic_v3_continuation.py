from __future__ import annotations

import copy
import json
import subprocess
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
    compare_warm_start_action, state_hash,
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
    # A later v3 code/config boundary is not a second allocation of v3 budgets.
    patched_contract = {**new, "source_git_commit": "3" * 40}
    patched_record = original(new_checkpoint, patched_contract, project_root=tmp_path)
    assert patched_record["source_semantic_version"] == "v3"
    assert patched_record["configuration_transition"]["execution_profile.yaml"]["source_path"].startswith("configs/ppo_semantic_v3/")
    patched, patched_env = runner("v3")
    patch_previous = load_semantic_checkpoint(patched, new_checkpoint, contract=patched_contract,
                                              seed=1001, warm_start=patched_record)
    assert patch_previous["stage_requested_decisions"]["smoke"] == 128
    assert patch_previous["new_mdp_origin_global_policy_decisions"] == 128
    assert (patch_previous["global_policy_decisions"], patch_previous["ppo_updates"],
            patch_previous["optimizer_steps"]) == (256, 2, 40)
    assert parameter_hash(patched.alg.actor) == parameter_hash(target.alg.actor)
    assert parameter_hash(patched.alg.critic) == parameter_hash(target.alg.critic)
    assert patched.alg.optimizer.state_dict()["state"] == {}
    next_result = train_semantic(patched, patched_env, run_dir=tmp_path / "patched_run",
        output_root=tmp_path / "patched_outputs", stage="smoke", decisions=128,
        contract=patched_contract, seed=1001, resume_infos=patch_previous)
    next_metadata = migration.checkpoint_metadata(Path(next_result["checkpoints"][-1]["checkpoint"]))
    assert next_metadata["stage_requested_decisions"]["smoke"] == 256
    assert next_metadata["new_mdp_origin_global_policy_decisions"] == 128
    assert (next_metadata["global_policy_decisions"], next_metadata["ppo_updates"],
            next_metadata["optimizer_steps"]) == (384, 3, 60)
    assert next_metadata["resume_ancestry"]["source_checkpoint"]["checkpoint"] == str(new_checkpoint.resolve())
    assert migration.v3_warm_start_checkpoint_name(record) != migration.v3_warm_start_checkpoint_name(patched_record)
    invalid_source = json.loads(new_checkpoint.with_name(new_checkpoint.stem + "_manifest.json").read_text())
    invalid_source.pop("new_mdp_origin_global_policy_decisions")
    new_checkpoint.with_name(new_checkpoint.stem + "_manifest.json").write_text(json.dumps(invalid_source))
    with pytest.raises(ValueError, match="intact original budget accounting"):
        original(new_checkpoint, patched_contract, project_root=tmp_path)


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


def test_v3_source_cli_preserves_existing_budget_and_allows_new_boundary(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(cli, "OUTPUT_ROOT", tmp_path / "outputs/ppo_semantic_v2")
    source = cli.version_paths("v3")[1] / "checkpoints/history/checkpoint_step_000011648.pt"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"path/argument validation only")
    source.with_name(source.stem + "_manifest.json").write_text(json.dumps({
        "semantic_version": "v3", "stage_requested_decisions": {"phase_suffix": 1536}}))
    (source.parent / "checkpoint_initial_v3_warm_start.pt").write_bytes(b"previous initial must remain")
    (source.parent.parent / "checkpoint_last_pointer.json").write_text("{}")
    argv = ["train", "--run-dir", str(tmp_path / "runs/ppo_semantic_v3/train/patch"),
            "--expected-head", "a" * 40, "--semantic-version", "v3", "--from-phase", "P06",
            "--new-mdp-warm-start", "--checkpoint", str(source), "--stage", "phase_suffix"]
    args = cli.parser().parse_args([*argv, "--decisions", "98464"])
    cli.validate_request(args)
    assert args.checkpoint == source.resolve()
    with pytest.raises(ValueError, match="remaining semantic stage budget"):
        cli.validate_request(cli.parser().parse_args([*argv, "--decisions", "98465"]))
    old_source = cli.OUTPUT_ROOT / "checkpoints/history/old_v2.pt"
    old_source.parent.mkdir(parents=True)
    old_source.write_bytes(b"old v2 path validation")
    old_source.with_name(old_source.stem + "_manifest.json").write_text(json.dumps({"semantic_version": "v2"}))
    old_args = cli.parser().parse_args([*argv, "--decisions", "128"])
    old_args.checkpoint = old_source
    with pytest.raises(ValueError, match="v3 training already exists"):
        cli.validate_request(old_args)


def test_warm_start_profile_recovers_hash_bound_git_bytes_not_current_v3_or_v2(tmp_path):
    _, source_contract = contracts(tmp_path)
    relative = "configs/ppo_semantic_v3/execution_profile.yaml"
    current = tmp_path / relative
    # Model the real Windows checkout: manifest hashes CRLF while Git stores LF.
    source_bytes = current.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    current.write_bytes(source_bytes)
    source_contract["files"][relative] = migration.file_sha(current)
    source_contract["runtime_content_sha256"] = migration.digest(source_contract["files"])
    def git(*args):
        return subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True).stdout.decode().strip()
    git("init")
    git("-c", "core.autocrlf=true", "add", "configs")
    git("-c", "user.name=CPU Test", "-c", "user.email=cpu@example.invalid", "commit", "-m", "source profile")
    source_contract["source_git_commit"] = git("rev-parse", "HEAD")
    record = {"source_runtime_contract": source_contract, "configuration_transition": {
        "execution_profile.yaml": {"source_path": relative, "source_sha256": migration.file_sha(current)}}}
    assert migration.warm_start_source_execution_profile(record, tmp_path / "run", project_root=tmp_path) == current.resolve()
    current.write_bytes(source_bytes + b"\r\n# changed target profile\r\n")
    restored = migration.warm_start_source_execution_profile(record, tmp_path / "run", project_root=tmp_path)
    assert restored != current.resolve() and restored.read_bytes() == source_bytes
    assert migration.file_sha(restored) == record["configuration_transition"]["execution_profile.yaml"]["source_sha256"]
    assert migration.warm_start_source_execution_profile(record, tmp_path / "run", project_root=tmp_path) == restored
    restored.write_bytes(b"corrupt materialized source")
    with pytest.raises(ValueError, match="immutable source execution profile"):
        migration.warm_start_source_execution_profile(record, tmp_path / "run", project_root=tmp_path)


def test_real_local_v3_11648_checkpoint_official_load_retains_networks_and_spent_budget(tmp_path, monkeypatch):
    """Optional local evidence: the actual stopped live checkpoint, not a tensor stub."""
    import wlr50_clean.ppo.semantic_training as training
    source = cli.PROJECT_ROOT / "outputs/ppo_semantic_v3/checkpoints/history/checkpoint_step_000011648.pt"
    if not source.is_file():
        pytest.skip("local immutable live 11648 checkpoint is not present")
    metadata = migration.checkpoint_metadata(source)
    target_contract = copy.deepcopy(metadata["runtime_contract"])
    # CPU test inventory only: runtime_contract() intentionally forbids uncommitted
    # worktrees. Bind today's actual bytes without claiming a live frozen revision.
    for relative in target_contract["files"]:
        target_contract["files"][relative] = migration.file_sha(cli.PROJECT_ROOT / relative)
    target_contract["runtime_content_sha256"] = migration.digest(target_contract["files"])
    record = migration.build_v3_warm_start_record(source, target_contract)
    actual, env = runner("v3")
    source_loads = []
    official_load = training.load_checkpoint_round_trip
    def observe_actual_load(runner, path):
        source_loads.append(Path(path).resolve())
        return official_load(runner, path)
    monkeypatch.setattr(training, "load_checkpoint_round_trip", observe_actual_load)
    previous = load_semantic_checkpoint(actual, source, contract=target_contract, seed=1001, warm_start=record)
    assert source_loads == [source.resolve()]
    assert (previous["global_policy_decisions"], previous["ppo_updates"], previous["optimizer_steps"]) == (11648, 56, 1120)
    assert previous["stage_requested_decisions"] == {"smoke": 0, "phase_suffix": 1536, "full_episode": 0}
    assert previous["new_mdp_origin_global_policy_decisions"] == 10112
    assert parameter_hash(actual.alg.actor) == metadata["actor_parameter_sha256"]  # Includes learned std.
    assert parameter_hash(actual.alg.critic) == metadata["critic_parameter_sha256"]
    normalizers = {role: getattr(actual.alg, role).obs_normalizer.state_dict() for role in ("actor", "critic")}
    assert not any(normalizers.values()) and state_hash(normalizers) == metadata["normalizer_state_sha256"]
    assert actual.alg.optimizer.state_dict()["state"] == {}
    assert state_hash(actual.alg.optimizer.state_dict()) != metadata["optimizer_state_sha256"]
    assert actual.alg.learning_rate == 3e-5 and all(group["lr"] == 3e-5 for group in actual.alg.optimizer.param_groups)
    assert actual.alg.storage.step == 0 and actual.alg.transition.actions is None
    assert tuple(actual.alg.storage.actions.shape) == (128, 1, 12)
    name = migration.v3_warm_start_checkpoint_name(record)
    assert "000011648" in name and target_contract["source_git_commit"][:12] in name
    assert target_contract["runtime_content_sha256"] in name
    changed = copy.deepcopy(record)
    changed["target_runtime_contract"]["source_git_commit"] = "f" * 40
    assert migration.v3_warm_start_checkpoint_name(changed) != name
    from test_semantic_observation_reward_env import _frame
    env.core.frame = _frame(stage="P06")
    old_profile = migration.warm_start_source_execution_profile(record, tmp_path / "run")
    assert migration.file_sha(old_profile) == record["configuration_transition"]["execution_profile.yaml"]["source_sha256"]
    comparison = compare_warm_start_action(actual, env, old_execution_profile=old_profile,
        new_execution_profile=cli.version_paths("v3")[2] / "execution_profile.yaml")
    if comparison["old"]["sha256"] == comparison["new"]["sha256"]:
        assert not any(comparison["new_minus_old"]["scaled_residual_full12"])
    # Real preflight branch refuses the exact already-published boundary before
    # live dispatch, while the earlier differently named initial is harmless.
    output_root = tmp_path / "isolated_v3_outputs"
    config_root = cli.version_paths("v3")[2]
    monkeypatch.setattr(cli, "version_paths", lambda version: (tmp_path / "runs", output_root, config_root))
    args = cli.parser().parse_args(["train", "--run-dir", str(tmp_path / "runs/epoch"),
        "--expected-head", target_contract["source_git_commit"], "--semantic-version", "v3",
        "--new-mdp-warm-start", "--checkpoint", str(source)])
    cli._preflight_checkpoint(args, target_contract)
    assert args._warm_start_record == record
    initial = output_root / "checkpoints/history" / name
    initial.parent.mkdir(parents=True)
    initial.write_bytes(b"existing immutable publication")
    with pytest.raises(ValueError, match="initial checkpoint already exists"):
        cli._preflight_checkpoint(args, target_contract)
    assert initial.read_bytes() == b"existing immutable publication"
