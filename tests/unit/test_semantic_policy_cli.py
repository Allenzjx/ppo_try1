"""CPU entry-point contracts; no Isaac launch or physical-success assertions."""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from wlr50_clean.ppo import semantic_cli as cli
from wlr50_clean.ppo import semantic_migration as migration
from wlr50_clean.ppo import semantic_video_cli as video_cli
from wlr50_clean.ppo.semantic_policy_distribution import (
    LEGACY_POLICY, STATE_DEPENDENT_POLICY, policy_contract,
)


def metadata(version=LEGACY_POLICY, *, contract=None, explicit=True):
    result = {"seed": 1001, "semantic_version": "v3", "runtime_contract": contract or {"version": 1},
        "runner_config": cli.semantic_runner_config(seed=1001, device="cpu", semantic_version="v3",
                                                     policy_version=version),
        "stage_requested_decisions": {"smoke": 0, "phase_suffix": 15360, "full_episode": 12800}}
    if explicit:
        result["policy_contract"] = policy_contract(version)
    return result


@pytest.fixture
def context(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    checkpoint = tmp_path / "outputs/ppo_semantic_v3/checkpoints/history/source.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"CLI plumbing checkpoint, tensor integrity tested by loader suite")
    checkpoint.with_name("source_manifest.json").write_text(json.dumps(metadata()), encoding="utf-8")
    monkeypatch.setattr(migration, "checkpoint_metadata", lambda path: json.loads(
        Path(path).with_name(Path(path).stem + "_manifest.json").read_text(encoding="utf-8")))
    args = cli.parser().parse_args(["train", "--run-dir", str(tmp_path / "runs/ppo_semantic_v3/train/test"),
        "--expected-head", "a" * 40, "--semantic-version", "v3", "--stage", "full_episode",
        "--decisions", "128", "--checkpoint", str(checkpoint), "--device", "cpu"])
    return args, checkpoint


@pytest.mark.parametrize("changes", [
    {"semantic_version": "v2"}, {"num_envs": 8}, {"command": "eval"}, {"command": "smoke"},
    {"command": "preflight"}, {"checkpoint": None}, {"new_mdp_warm_start": True},
    {"resume_migration": Path("review.json")},
])
def test_conversion_scope_rejected_before_checkpoint_or_app(context, changes):
    args, _ = context
    args.policy_distribution_migration = True
    for key, value in changes.items():
        setattr(args, key, value)
    with pytest.raises(ValueError):
        cli.validate_request(args)


def test_explicit_conversion_builds_preflight_record_and_no_new_budget(context, monkeypatch):
    args, checkpoint = context
    args.policy_distribution_migration = True
    calls = []
    record = {"source_policy_version": LEGACY_POLICY, "target_policy_version": STATE_DEPENDENT_POLICY}
    monkeypatch.setattr(cli, "build_policy_distribution_migration", lambda path, contract, *, project_root:
                        calls.append((path, contract, project_root)) or record)
    monkeypatch.setattr(cli, "policy_migration_checkpoint_name", lambda record: "checkpoint_initial_policy_test.pt")
    cli.validate_request(args)
    cli._preflight_checkpoint(args, {"version": 2})
    assert calls == [(checkpoint, {"version": 2}, cli.PROJECT_ROOT)]
    assert args._policy_version == STATE_DEPENDENT_POLICY and args._policy_migration_record == record
    assert args._migration_record is None and args._warm_start_record is None
    assert args.decisions == 128


def test_main_conversion_validation_finishes_before_any_live_dispatch(context, monkeypatch):
    args, checkpoint = context
    monkeypatch.setattr(cli, "runtime_contract", lambda **kw: {"version": 2})
    monkeypatch.setattr(cli, "dispatch_live", lambda *a: pytest.fail("Isaac dispatch preceded policy validation"))
    def reject(*args, **kwargs):
        raise ValueError("policy-only protected configuration changed")
    monkeypatch.setattr(cli, "build_policy_distribution_migration", reject)
    with pytest.raises(ValueError, match="policy-only protected"):
        cli.main(["train", "--run-dir", str(args.run_dir), "--expected-head", "a" * 40,
            "--semantic-version", "v3", "--stage", "full_episode", "--decisions", "128",
            "--checkpoint", str(checkpoint), "--device", "cpu", "--policy-distribution-migration"])
    assert not args.run_dir.exists()


@pytest.mark.parametrize("sidecar_only", [False, True])
def test_conversion_cannot_republish_existing_initial(context, monkeypatch, sidecar_only):
    args, checkpoint = context
    args.policy_distribution_migration = True
    monkeypatch.setattr(cli, "build_policy_distribution_migration", lambda *a, **kw: {"record": "verified"})
    monkeypatch.setattr(cli, "policy_migration_checkpoint_name", lambda record: "checkpoint_initial_policy_test.pt")
    initial = checkpoint.parent / "checkpoint_initial_policy_test.pt"
    existing = initial.with_name(initial.stem + "_manifest.json") if sidecar_only else initial
    existing.write_bytes(b"immutable prior conversion")
    with pytest.raises(ValueError, match="already exists"):
        cli._preflight_checkpoint(args, {"version": 2})
    assert existing.read_bytes() == b"immutable prior conversion"


@pytest.mark.parametrize("version,explicit", [(LEGACY_POLICY, False), (LEGACY_POLICY, True),
                                             (STATE_DEPENDENT_POLICY, True)])
def test_checkpoint_type_auto_selected_without_conversion_flag(context, version, explicit):
    args, checkpoint = context
    checkpoint.with_name("source_manifest.json").write_text(json.dumps(metadata(version, explicit=explicit)))
    cli._preflight_checkpoint(args, {"version": 1})
    assert args._policy_version == version and args._policy_migration_record is None
    del args._policy_version
    assert cli._resolved_policy_version(args) == version


def test_incomplete_legacy_configuration_is_not_silently_assumed(context):
    args, checkpoint = context
    value = metadata(explicit=False)
    value["runner_config"]["actor"].pop("obs_normalization")
    checkpoint.with_name("source_manifest.json").write_text(json.dumps(value))
    with pytest.raises(ValueError):
        cli._preflight_checkpoint(args, {"version": 1})


def test_missing_new_namespace_flag_defaults_to_no_conversion(context):
    args, _ = context
    del args.policy_distribution_migration
    cli._preflight_checkpoint(args, {"version": 1})
    assert args._policy_version == LEGACY_POLICY and args._policy_migration_record is None


def test_actual_38272_legacy_checkpoint_metadata_auto_selects_old_distribution():
    source = Path(cli.__file__).resolve().parents[3] / "outputs/ppo_semantic_v3/checkpoints/history/checkpoint_step_000038272.pt"
    if not source.is_file():
        pytest.skip("immutable local 38272 checkpoint not present")
    value = migration.checkpoint_metadata(source)
    args = cli.parser().parse_args(["eval", "--run-dir", "unused_cpu_preflight", "--expected-head", "a" * 40,
        "--semantic-version", "v3", "--checkpoint", str(source), "--mode", "semantic_residual_eval", "--seed", "2001"])
    cli._preflight_checkpoint(args, value["runtime_contract"])
    assert args._policy_version == LEGACY_POLICY and args._policy_migration_record is None
    assert value["global_policy_decisions"] == 38272


@pytest.mark.parametrize("mode", ["legacy_fsm_eval", "semantic_prior_eval"])
def test_A_B_no_checkpoint_remain_actorless_legacy_default(context, mode):
    args, _ = context
    args.command, args.mode, args.checkpoint, args.seed, args.decisions = "eval", mode, None, 2001, None
    cli.validate_request(args)
    cli._preflight_checkpoint(args, {"version": 2})
    assert args._policy_version == LEGACY_POLICY and args._policy_migration_record is None


def test_exact_resume_comparison_uses_v3_and_resolved_policy(context, monkeypatch):
    args, checkpoint = context
    args.resume_migration = Path("review.json")
    value = metadata(STATE_DEPENDENT_POLICY)
    checkpoint.with_name("source_manifest.json").write_text(json.dumps(value))
    monkeypatch.setattr(migration, "validate_migration_plan", lambda *a: {"verified": True})
    cli._preflight_checkpoint(args, {"version": 2})
    assert args._policy_version == STATE_DEPENDENT_POLICY and args._migration_record == {"verified": True}


def fake_evaluation_runner():
    import torch
    class Actor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mean = torch.nn.Parameter(torch.zeros(12))
            self.obs_normalizer = torch.nn.Identity()
        def forward(self, observations, *, stochastic_output):
            assert stochastic_output is False
            return self.mean.expand(observations["policy"].shape[0], -1)
    actor, critic = Actor(), torch.nn.Linear(324, 1)
    critic.obs_normalizer = torch.nn.Identity()
    optimizer = torch.optim.Adam([*actor.parameters(), *critic.parameters()], lr=1e-5)
    return NS(alg=NS(actor=actor, critic=critic, optimizer=optimizer, eval_mode=lambda: None))


@pytest.mark.parametrize("version", [LEGACY_POLICY, STATE_DEPENDENT_POLICY])
@pytest.mark.parametrize("training_seed", [1001, 1002])
def test_eval_constructs_resolved_policy_without_initializing_or_converting(context, monkeypatch, version, training_seed):
    args, checkpoint = context
    value = metadata(version)
    value["seed"] = training_seed
    value["runner_config"] = cli.semantic_runner_config(seed=training_seed, device="cpu", semantic_version="v3",
                                                        policy_version=version)
    checkpoint.with_name("source_manifest.json").write_text(json.dumps(value))
    args.command, args.mode, args.seed, args.max_decisions = "eval", "semantic_residual_eval", 2001, 1
    args.run_dir.mkdir(parents=True)
    cli._preflight_checkpoint(args, {"version": 1})
    runner, calls = fake_evaluation_runner(), []
    def construct(env, *, seed, device, policy_version, initialize_actor):
        calls.append((seed, policy_version, initialize_actor, env.cfg["semantic_version"]))
        return runner, {}
    monkeypatch.setattr(cli, "construct_semantic_runner", construct)
    monkeypatch.setattr(cli, "load_semantic_checkpoint", lambda actual, path, *, contract, seed, migration:
                        {"global_policy_decisions": 38272})
    core = NS(frame=NS(sim_time_s=0.0), done=False)
    core.reset = lambda *, seed: (0.0,) * 324
    def step(raw):
        core.done = True
        core.frame.sim_time_s = 1 / 15
        return NS(observation=(0.0,) * 324, reward=0.0, terminated=True, truncated=False,
                  info={"raw_policy_action_full12": raw, "task_success": False, "termination_reason": "FALL"})
    core.step, core.telemetry_summary = step, lambda: {}
    result = cli._evaluation_body(core, args, contract={"version": 1}, recorder=None)
    assert calls == [(training_seed, version, False, "v3")]
    assert result["optimizer_updates_during_evaluation"] == 0 and result["deterministic_policy"]
    assert result["policy_contract"] == policy_contract(version)


@pytest.mark.parametrize("version", [LEGACY_POLICY, STATE_DEPENDENT_POLICY])
def test_video_auto_selects_saved_actor_type_without_conversion(context, monkeypatch, version):
    args, checkpoint = context
    checkpoint.with_name("source_manifest.json").write_text(json.dumps(metadata(version)))
    args.command, args.mode, args.seed = "eval", "semantic_residual_eval", 4001
    cli._preflight_checkpoint(args, {"version": 1})
    runner, calls = fake_evaluation_runner(), []
    def construct(env, *, seed, device, policy_version, initialize_actor):
        calls.append((seed, policy_version, initialize_actor))
        return runner, {}
    monkeypatch.setattr(video_cli, "construct_semantic_runner", construct)
    monkeypatch.setattr(video_cli, "load_semantic_checkpoint", lambda actual, path, *, contract, seed, migration:
                        {"resume_source_checkpoint": str(path), "global_policy_decisions": 38272})
    action, proof, unchanged = video_cli.checkpoint_loader(args, {"version": 1})((0.0,) * 324)
    assert action((1.0,) * 324, 1) == (0.0,) * 12
    unchanged()
    assert calls == [(1001, version, False)]
    assert proof["optimizer_updates"] == 0 and proof["policy_version"] == version


def test_video_explicitly_forbids_conversion_before_source_validation(context):
    args, _ = context
    args.policy_distribution_migration = True
    with pytest.raises(RuntimeError, match="cannot convert"):
        video_cli.validate_video_args(args)


def test_train_migration_wiring_saves_immutable_initial_with_unchanged_ledger(context, monkeypatch):
    from wlr50_clean.ppo import semantic_backend, semantic_env
    args, checkpoint = context
    args.policy_distribution_migration = True
    args._policy_version = STATE_DEPENDENT_POLICY
    args._policy_migration_record = {"source_policy_version": LEGACY_POLICY, "target_policy_version": STATE_DEPENDENT_POLICY}
    args._migration_record = args._warm_start_record = None
    args.run_dir.mkdir(parents=True)
    app = NS(update=lambda: None)
    monkeypatch.setitem(sys.modules, "isaaclab.app", NS(AppLauncher=lambda **kw: NS(app=app)))
    monkeypatch.setattr(semantic_backend, "SemanticIsaacBackend", lambda *a, **kw: NS())
    monkeypatch.setattr(semantic_env, "SemanticEpisodeEnv", lambda *a, **kw: NS())
    env = NS(cfg={"reset_sampling": "P01_full_task_only_initial_version"})
    monkeypatch.setattr(cli, "SemanticRslAdapter", lambda *a, **kw: env)
    actual_runner, observed, saved = NS(), {}, []
    def construct(env, *, seed, device, policy_version, initialize_actor):
        assert policy_version == STATE_DEPENDENT_POLICY and initialize_actor is False
        assert env.cfg["semantic_version"] == "v3"
        return actual_runner, {}
    monkeypatch.setattr(cli, "construct_semantic_runner", construct)
    previous = {"seed": 1001, "semantic_version": "v3", "global_policy_decisions": 38272,
        "ppo_updates": 264, "optimizer_steps": 5280,
        "stage_requested_decisions": {"smoke": 0, "phase_suffix": 15360, "full_episode": 12800},
        "new_mdp_origin_global_policy_decisions": 10112,
        "policy_distribution_migration": args._policy_migration_record,
        "policy_distribution_migration_evidence": {"real_tensor_mapping_verified": True},
        "resume_source_checkpoint": str(checkpoint)}
    before = copy.deepcopy(previous)
    def load(runner, path, *, contract, seed, migration, warm_start, policy_migration):
        assert migration is None and warm_start is None and policy_migration == args._policy_migration_record
        return previous
    monkeypatch.setattr(cli, "load_semantic_checkpoint", load)
    monkeypatch.setattr(cli, "policy_migration_checkpoint_name", lambda record: "checkpoint_initial_policy_test.pt")
    monkeypatch.setattr(cli, "save_semantic_checkpoint", lambda runner, path, infos: saved.append((path, infos)))
    monkeypatch.setattr(cli, "train_semantic", lambda runner, env, **kw: observed.update(kw) or {"called": True})
    result = cli.dispatch_live(args, {"version": 2})
    assert result == {"called": True} and len(saved) == 1
    assert previous == before and observed["resume_infos"] == previous and observed["decisions"] == 128
    path, infos = saved[0]
    assert path.name == "checkpoint_initial_policy_test.pt" and infos["stage"] == "initial_policy_distribution_migration"
    for key in ("global_policy_decisions", "ppo_updates", "optimizer_steps", "stage_requested_decisions",
                "new_mdp_origin_global_policy_decisions", "policy_distribution_migration_evidence"):
        assert infos[key] == previous[key]
    assert infos["runtime_contract"] == {"version": 2}
    assert infos["policy_contract"] == policy_contract(STATE_DEPENDENT_POLICY)
    assert infos["runner_config"] == cli.semantic_runner_config(seed=1001, device="cpu", semantic_version="v3",
                                                                policy_version=STATE_DEPENDENT_POLICY)
    assert json.loads((args.run_dir / "policy_distribution_migration.json").read_text()) == args._policy_migration_record
    assert not (args.run_dir / "new_mdp_initial_action_comparison.json").exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"already published")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        cli._save_policy_migration_initial(actual_runner, env, args, {"version": 2}, cli.version_paths("v3")[1], previous)


@pytest.mark.skipif(sys.platform != "win32", reason="native PowerShell argument binding")
def test_powershell_conversion_scope_fails_before_native_launch():
    script = Path(cli.__file__).resolve().parents[3] / "scripts/run_semantic_ppo.ps1"
    result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-File", str(script),
        "-Command", "eval", "-ExpectedHead", "a" * 40, "-SemanticVersion", "v3",
        "-PolicyDistributionMigration", "-Checkpoint", "never_loaded.pt"], capture_output=True, text=True)
    assert result.returncode != 0
    assert "PolicyDistributionMigration requires v3 N1 train" in result.stderr
    assert "if ($PolicyDistributionMigration) { $arguments += '--policy-distribution-migration' }" in script.read_text()
