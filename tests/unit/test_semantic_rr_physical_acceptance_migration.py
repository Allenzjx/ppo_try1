"""Bounded RR task transition: real temporary Git and official CPU PPO, no Isaac."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
import yaml

from wlr50_clean.ppo import semantic_cli as cli, semantic_migration as m, semantic_training as t
from wlr50_clean.ppo import semantic_video as video
from wlr50_clean.ppo.semantic_policy_distribution import HISTORY_QUARTER_TEMPERED_POLICY, policy_contract
from wlr50_clean.ppo.semantic_transfer_roles import ROLE_OBSERVATION_LAYOUT
from test_semantic_role_observation_migration import NAMES, commit, git, source_metadata, write_json
from test_semantic_quarter_temperature_migration import make_runner

SOURCE = "configs/ppo_task_first_recovery_v1"
TARGET = "configs/ppo_residual_rr_fix_v1"
ZERO = "configs/ppo_non_residual_refine_v1"
FACTOR = "rr_physical_acceptance_same372_factor"
PROTECTED = {f"src/wlr50_clean/ppo/{name}.py" for name in (
    "semantic_reward", "semantic_nominal_geometry", "semantic_history_actor", "semantic_policy_distribution")}


def bind(root, contract):
    contract["files"] = {p: m.file_sha(root / p) for p in contract["files"]}
    contract["runtime_content_sha256"] = m.digest(contract["files"])
    namespace = f"configs/ppo_{contract['experiment_id']}"
    contract["selected_configuration"] = {n: {"path": f"{namespace}/{n}",
        "sha256": contract["files"][f"{namespace}/{n}"]} for n in NAMES}


@pytest.fixture(scope="module")
def prototype(tmp_path_factory):
    root = tmp_path_factory.mktemp("rr_acceptance_git")
    paths = m.RR_PHYSICAL_ACCEPTANCE_FILES | PROTECTED | {f"{SOURCE}/{n}" for n in NAMES}
    for relative in paths:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((m.PROJECT_ROOT / relative).read_bytes())
    git(root, "init")
    git(root, "config", "user.name", "RR acceptance CPU fixture")
    git(root, "config", "user.email", "rr-acceptance@example.invalid")
    old = {"source_git_commit": commit(root, "old quarter task"), "files": dict.fromkeys(sorted(paths)),
        "semantic_version": "v3", "experiment_id": "task_first_recovery_v1",
        "frozen_A_files": {"frozen": "a" * 64}, "physics_hz": 120., "decision_hz": 15.,
        "task_timeout_s": 200., "timeout_bootstrap": False, "rsl_rl_version": "5.0.1",
        "local_runtime_versions": {"fixture": "CPU only, not Isaac"}}
    bind(root, old)
    for namespace in (TARGET, ZERO):
        for name in NAMES:
            dest = root / namespace / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes((m.PROJECT_ROOT / namespace / name).read_bytes())
    source_text = (root / m.SUPERVISOR).read_text(encoding="utf-8")
    (root / m.SUPERVISOR).write_text(source_text + "\n# reviewed RR target fixture\n", encoding="utf-8")
    new = copy.deepcopy(old)
    new["experiment_id"] = "residual_rr_fix_v1"
    new["files"].update(dict.fromkeys(f"{namespace}/{n}" for namespace in (TARGET, ZERO) for n in NAMES))
    new["source_git_commit"] = commit(root, "new RR config and reviewed core")
    bind(root, new)
    return SimpleNamespace(root=root, old=old, new=new)


@pytest.fixture
def f(prototype, tmp_path):
    root = tmp_path / "case"
    shutil.copytree(prototype.root, root)
    checkpoint = root / "outputs/ppo_task_first_recovery_v1/checkpoints/history/source.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"metadata-only fixture; no tensor loader")
    metadata = source_metadata(checkpoint, prototype.old, layout=ROLE_OBSERVATION_LAYOUT,
        policy=HISTORY_QUARTER_TEMPERED_POLICY)
    sidecar = checkpoint.with_name("source_manifest.json")
    write_json(sidecar, metadata)
    return SimpleNamespace(root=root, old=copy.deepcopy(prototype.old), new=copy.deepcopy(prototype.new),
        checkpoint=checkpoint, sidecar=sidecar, metadata=metadata)


def record(f, **options):
    delta = sorted(p for p in f.old["files"].keys() | f.new["files"].keys()
        if f.old["files"].get(p) != f.new["files"].get(p))
    return m.build_migration_plan(f.checkpoint, f.new, allowed_changed_files=delta,
        reason="Strict free-air RR qualification, current carry and source-home continuation",
        rr_physical_acceptance_review={"reason": "Reviewed task-derived observation and nominal transition; same quarter full12 state",
            "reviewed_code_sha256": {p: f.new["files"][p] for p in delta if p in m.RR_PHYSICAL_ACCEPTANCE_FILES}},
        project_root=f.root, **options)


def verified_plan(f, monkeypatch):
    path = f.root / "plan.json"
    write_json(path, record(f))
    original = m.validate_migration_plan
    monkeypatch.setattr(m, "validate_migration_plan", lambda cp, c, p:
        original(cp, c, p, project_root=f.root))
    return m.validate_migration_plan(f.checkpoint, f.new, path)


def request(f, **options):
    args = cli.parser().parse_args(["train", "--semantic-version", "v3", "--experiment-id", "residual_rr_fix_v1",
        "--run-dir", str(f.root / "runs/ppo_residual_rr_fix_v1/train/new"),
        "--expected-head", f.new["source_git_commit"], "--checkpoint", str(f.checkpoint),
        "--resume-migration", str(f.root / "plan.json"), "--stage", "full_episode", "--decisions", "128",
        "--seed", "1001", "--device", "cpu"])
    for key, value in options.items():
        setattr(args, key, value)
    return args


def test_exact_config_delta_and_new_paths():
    for name in set(NAMES) - {"stage_task_spec.yaml"}:
        assert (m.PROJECT_ROOT / SOURCE / name).read_bytes() == (m.PROJECT_ROOT / TARGET / name).read_bytes()
    before = yaml.safe_load((m.PROJECT_ROOT / SOURCE / "stage_task_spec.yaml").read_text())
    expected = copy.deepcopy(before)
    expected.update(revision="residual_rr_fix_v1", p09_lift_semantics="functional_free_air_lift_v3")
    expected["nominal"].update(rr_carry_source_semantics="current_free_lift_before_pending_knee_and_roll_v1",
        final_stop_owner="source_home_after_physical_stop_v2")
    assert yaml.safe_load((m.PROJECT_ROOT / TARGET / "stage_task_spec.yaml").read_text()) == expected
    runs, outputs, configs = cli.version_paths("v3", experiment_id="residual_rr_fix_v1")
    assert [p.name for p in (runs, outputs, configs)] == ["ppo_residual_rr_fix_v1"] * 3
    assert all(path.parent == configs for path in video.video_configuration("v3", experiment_id="residual_rr_fix_v1").values())
    assert video.task_interval_receipt(6116, experiment_id="residual_rr_fix_v1")["frame_count"] == 765
    assert video.camera_for_experiment("residual_rr_fix_v1") == video.camera_for_experiment("non_residual_refine_v1")


def test_real_plan_roundtrip_same_policy_changed_semantics_and_no_source_write(f):
    before = f.checkpoint.read_bytes(), f.sidecar.read_bytes()
    plan = record(f)
    factor = plan[FACTOR]
    assert factor["schema"] == m.RR_PHYSICAL_ACCEPTANCE_SCHEMA
    assert factor["branch_id"] == "residual_rr_fix_v1"
    assert factor["counter_origin"] == {key: f.metadata[key] for key in ("global_policy_decisions", "ppo_updates", "optimizer_steps")}
    assert factor["task_acceptance_changed"] and factor["action_execution_changed"]
    assert factor["physical_mdp_changed"] and factor["task_reward_event_semantics_changed"]
    assert not any(factor[k] for k in ("physical_scene_changed", "actuator_capability_changed", "kernel_changed",
        "reward_code_and_weights_changed", "nominal_geometry_changed", "action_ranges_changed", "observation_layout_changed"))
    assert factor["quality_epsilon"] == 0. and factor["observation_semantics_changed"]
    ob = factor["observation_contract"]
    assert ob["source_policy_contract"] == ob["target_policy_contract"] == f.metadata["policy_contract"]
    assert (ob["observation_dimension"], ob["action_dimension"], ob["num_envs"]) == (372, 12, 1)
    assert factor["source_runner_config"] == factor["target_runner_config"] == f.metadata["runner_config"]
    assert plan["discard_old_rollout_storage"] and factor["migration_added_updates"] == 0
    assert len(factor["inherited_zero_configuration_bindings"]) == 6
    path = f.root / "plan.json"
    write_json(path, plan)
    assert m.validate_migration_plan(f.checkpoint, f.new, path, project_root=f.root)[FACTOR] == factor
    assert before == (f.checkpoint.read_bytes(), f.sidecar.read_bytes())


@pytest.mark.parametrize("kind", ["physics", "namespace", "policy_temperature", "rho", "runner", "stage_extra",
    "reward_config", "caps", "observation", "source_config", "geometry_code", "actor_code", "reward_code",
    "zero_extra", "missing_delta", "deleted_file", "no_core_fix"])
def test_rejects_unrelated_or_incompatible_change(f, kind):
    if kind == "physics":
        f.new["physics_hz"] = 60.
    elif kind == "namespace":
        f.new["experiment_id"] = "non_residual_refine_v1"
    elif kind in ("policy_temperature", "rho", "runner"):
        if kind == "runner": f.metadata["runner_config"]["algorithm"]["clip_param"] = .3
        else: f.metadata["policy_contract"]["exploration_std_temperature" if kind == "policy_temperature" else "rho"] = .5
        write_json(f.sidecar, f.metadata)
    elif kind == "missing_delta":
        f.new["files"]["src/unreviewed.py"] = "b" * 64
        f.new["runtime_content_sha256"] = m.digest(f.new["files"])
    elif kind == "deleted_file":
        f.new["files"].pop(next(iter(PROTECTED)))
        f.new["runtime_content_sha256"] = m.digest(f.new["files"])
    elif kind == "no_core_fix":
        f.new["files"][m.SUPERVISOR] = f.old["files"][m.SUPERVISOR]
        f.new["runtime_content_sha256"] = m.digest(f.new["files"])
    else:
        relative = {"stage_extra": f"{TARGET}/stage_task_spec.yaml", "reward_config": f"{TARGET}/reward_config.yaml",
            "caps": f"{TARGET}/execution_profile.yaml", "observation": f"{TARGET}/observation_schema.json",
            "source_config": f"{SOURCE}/stage_task_spec.yaml", "zero_extra": f"{ZERO}/stage_task_spec.yaml",
            "geometry_code": "src/wlr50_clean/ppo/semantic_nominal_geometry.py",
            "actor_code": "src/wlr50_clean/ppo/semantic_history_actor.py",
            "reward_code": "src/wlr50_clean/ppo/semantic_reward.py"}[kind]
        file = f.root / relative
        if kind in ("stage_extra", "zero_extra"):
            data = yaml.safe_load(file.read_text())
            data["history"]["minimum_lift_gain_m"] = .001
            file.write_text(yaml.safe_dump(data), encoding="utf-8")
        else:
            file.write_bytes(file.read_bytes() + b"\n# not this boundary\n")
        bind(f.root, f.new)
    with pytest.raises(ValueError):
        record(f)


@pytest.mark.parametrize("other", ["exploration_temperature_review", "task_first_reward_review",
    "execution_composition_review", "final_stop_handoff_review", "video_review", "execution_evidence"])
def test_factor_is_exclusive(f, other):
    with pytest.raises(ValueError, match="cannot mix"):
        record(f, **{other: {}})


@pytest.mark.parametrize("field", ["optimizer", "counter_origin", "observation_semantics_changed", "quality_epsilon"])
def test_receipt_tamper_rejected(f, field):
    plan = record(f)
    plan[FACTOR][field] = "unreviewed"
    path = f.root / "tamper.json"
    write_json(path, plan)
    with pytest.raises(ValueError, match="exactly bound"):
        m.validate_migration_plan(f.checkpoint, f.new, path, project_root=f.root)


def test_stale_target_bytes_and_omitted_delta_rejected(f):
    plan = record(f)
    with pytest.raises(ValueError, match="inventory"):
        m.build_migration_plan(f.checkpoint, f.new, allowed_changed_files=[], reason=plan["reason"],
            rr_physical_acceptance_review={"reason": "review", "reviewed_code_sha256": plan[FACTOR]["reviewed_code_sha256"]},
            project_root=f.root)
    file = f.root / m.SUPERVISOR
    file.write_bytes(file.read_bytes() + b"\n# edit after review\n")
    with pytest.raises(ValueError, match="target bytes"):
        record(f)


def test_cli_real_plan_selects_same_quarter_and_allows_prefix_not_legacy_factor(f, monkeypatch):
    monkeypatch.setattr(cli, "PROJECT_ROOT", f.root)
    verified = verified_plan(f, monkeypatch)
    args = request(f)
    cli.validate_request(args)
    cli._preflight_checkpoint(args, f.new)
    assert args._migration_record == verified
    assert cli._resolved_policy_version(args) == HISTORY_QUARTER_TEMPERED_POLICY
    suffix = request(f, stage="phase_suffix", from_phase="P06", decisions=512, prefix_source="successful_nominal")
    cli.validate_request(suffix)
    cli._preflight_checkpoint(suffix, f.new)
    with pytest.raises(ValueError, match="checkpoint root"):
        cli.validate_request(request(f, resume_migration=None))
    write_json(f.root / "plan.json", {"final_stop_handoff_factor": {}})
    with pytest.raises(ValueError, match="independent reviewed factor"):
        cli.validate_request(request(f))


def test_rr_cannot_initialize_fresh_weights_but_prior_eval_needs_no_checkpoint(f, monkeypatch):
    monkeypatch.setattr(cli, "PROJECT_ROOT", f.root)
    with pytest.raises(ValueError, match="v3 continuation requires existing learned checkpoint"):
        cli.validate_request(request(f, checkpoint=None, resume_migration=None))
    cli.validate_request(request(f, command="eval", checkpoint=None, resume_migration=None,
        decisions=None, seed=4001, mode="semantic_prior_eval"))


@pytest.fixture
def cpu(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("rsl_rl")
    count, rng = torch.get_num_threads(), torch.get_rng_state()
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    yield torch
    torch.set_rng_state(rng)
    torch.set_num_threads(count)


def test_official_cpu_load_update_save_reload_full_state_and_branch_counts(prototype, tmp_path, monkeypatch, cpu):
    root = tmp_path / "official_cpu"
    shutil.copytree(prototype.root, root)
    t.seed_training_rngs(1001)
    source, source_env = make_runner(HISTORY_QUARTER_TEMPERED_POLICY)
    initial = t.train_semantic(source, source_env, run_dir=root / "old_run", output_root=root / "old_output",
        stage="full_episode", decisions=128, contract=prototype.old, seed=1001)
    metadata = json.loads(Path(initial["checkpoints"][-1]["manifest"]).read_text())
    for key in ("checkpoint_path", "checkpoint_sha256", "save_load_round_trip"):
        metadata.pop(key)
    # An old ancestry marker must survive without being mistaken for this branch's origin.
    metadata["task_recovery_branch"] = {"counter_origin": {"global_policy_decisions": 0, "ppo_updates": 0, "optimizer_steps": 0}}
    source.alg.learning_rate = 1e-5
    for group in source.alg.optimizer.param_groups:
        group.update(lr=1e-5, betas=(.87, .996), eps=2e-8)
    cp = root / "learned_quarter.pt"
    t.save_semantic_checkpoint(source, cp, metadata)
    saved = m.checkpoint_metadata(cp)
    assert source.alg.optimizer.state_dict()["state"]
    f = SimpleNamespace(root=root, old=prototype.old, new=prototype.new, checkpoint=cp)
    verified = verified_plan(f, monkeypatch)
    target, env = make_runner(HISTORY_QUARTER_TEMPERED_POLICY)
    loaded = t.load_semantic_checkpoint(target, cp, contract=f.new, seed=1001, migration=verified)
    for role in ("actor", "critic"):
        assert t.parameter_hash(getattr(target.alg, role)) == t.parameter_hash(getattr(source.alg, role))
    assert t.state_hash(target.alg.optimizer.state_dict()) == saved["optimizer_state_sha256"]
    assert t.state_hash(t._normalizers(target)) == saved["normalizer_state_sha256"]
    assert target.alg.actor.obs_normalizer.state_dict() == target.alg.critic.obs_normalizer.state_dict() == {}
    assert t.capture_training_rng_state(seed=1001) == saved["training_rng_state"]
    assert target.alg.learning_rate == t.optimizer_learning_rate(target) == 1e-5
    assert target.alg.storage.step == 0 and target.alg.transition.actions is None and env.core.calls == 0
    for key in ("global_policy_decisions", "ppo_updates", "optimizer_steps", "stage_requested_decisions"):
        assert loaded[key] == saved[key]
    assert loaded["rr_task_branch"]["counter_origin"] == {"global_policy_decisions": 128, "ppo_updates": 1, "optimizer_steps": 20}
    assert loaded["task_recovery_branch"] == saved["task_recovery_branch"]
    for fault in ("partial", "pending"):
        target.alg.storage.step = int(fault == "partial")
        target.alg.transition.actions = cpu.zeros(1, 12) if fault == "pending" else None
        with pytest.raises(RuntimeError, match="partial old rollout"):
            t.load_semantic_checkpoint(target, cp, contract=f.new, seed=1001, migration=verified)
    target.alg.storage.step = 0
    target.alg.transition.actions = None
    continued = t.train_semantic(target, env, run_dir=root / "new_run", output_root=root / "outputs/ppo_residual_rr_fix_v1",
        stage="full_episode", decisions=128, contract=f.new, seed=1001, resume_infos=loaded)
    final_cp = Path(continued["checkpoints"][-1]["checkpoint"])
    final, _ = make_runner(HISTORY_QUARTER_TEMPERED_POLICY)
    info = t.load_semantic_checkpoint(final, final_cp, contract=f.new, seed=1001)
    assert (info["global_policy_decisions"], info["ppo_updates"], info["optimizer_steps"]) == (256, 2, 40)
    assert info["rr_task_branch_counts"] == {"global_policy_decisions": 128, "ppo_updates": 1, "optimizer_steps": 20}
    assert info["resume_ancestry"]["resume_migration"][FACTOR] == verified[FACTOR]
    assert t.parameter_hash(final.alg.actor) == t.parameter_hash(target.alg.actor) != saved["actor_parameter_sha256"]
    assert t.state_hash(final.alg.optimizer.state_dict()) == t.state_hash(target.alg.optimizer.state_dict())
    assert info["policy_contract"] == policy_contract(HISTORY_QUARTER_TEMPERED_POLICY, observation_layout=ROLE_OBSERVATION_LAYOUT)
    assert list((root / "new_run").rglob("update_000002_likelihood.json"))
    monkeypatch.setattr(cli, "PROJECT_ROOT", root)
    same = request(f, checkpoint=final_cp, resume_migration=None)
    cli.validate_request(same)
    cli._preflight_checkpoint(same, f.new)
    assert cli._resolved_policy_version(same) == HISTORY_QUARTER_TEMPERED_POLICY
