"""Narrow nominal timing boundary; synthetic CPU evidence, never Isaac or live CPs."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
import yaml

from wlr50_clean.ppo import semantic_cli as cli, semantic_migration as m
from wlr50_clean.ppo.semantic_transfer_roles import ROLE_OBSERVATION_LAYOUT
from test_semantic_role_observation_migration import NAMES, PROTECTED, commit, git, source_metadata, write_json

CONFIG = "configs/ppo_fsm_reference_p09_stable_v2"
SPEC = f"{CONFIG}/stage_task_spec.yaml"


def bind(root, contract):
    contract["files"] = {p: m.file_sha(root / p) for p in contract["files"]}
    contract["runtime_content_sha256"] = m.digest(contract["files"])
    contract["selected_configuration"] = {
        name: {"path": f"{CONFIG}/{name}", "sha256": contract["files"][f"{CONFIG}/{name}"]}
        for name in NAMES}


@pytest.fixture(scope="module")
def prototype(tmp_path_factory):
    root = tmp_path_factory.mktemp("timing_only_git")
    paths = {f"{CONFIG}/{name}" for name in NAMES} | {m.SUPERVISOR, PROTECTED} | {
        f"src/wlr50_clean/ppo/{name}.py" for name in ("semantic_migration", "semantic_training")}
    for relative in sorted(paths):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((m.PROJECT_ROOT / relative).read_bytes())
    spec = yaml.safe_load((root / SPEC).read_bytes())
    spec["nominal"].pop("sequence_semantics", None)
    (root / SPEC).write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    provider = root / m.SUPERVISOR
    provider.write_text("LOCKED = 1\n\nclass NominalMotionProvider:\n    timing = 'before'\n\nclass TaskEvaluator:\n    pass\n", encoding="utf-8")
    git(root, "init")
    git(root, "config", "user.name", "Timing fixture")
    git(root, "config", "user.email", "timing@example.invalid")
    old = {"source_git_commit": commit(root, "before timing"),
        "files": dict.fromkeys(sorted(paths)), "semantic_version": "v3",
        "experiment_id": "fsm_reference_p09_stable_v2",
        "frozen_A_files": {"frozen": "a" * 64}, "physics_hz": 120., "decision_hz": 15.,
        "task_timeout_s": 200., "timeout_bootstrap": False, "rsl_rl_version": "5.0.1",
        "local_runtime_versions": {"fixture": "not Isaac"}}
    bind(root, old)
    spec["nominal"]["sequence_semantics"] = m.TIMING_ONLY_MODE
    (root / SPEC).write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    provider.write_bytes(provider.read_bytes().replace(b"timing = 'before'", b"timing = 'after'"))
    new = copy.deepcopy(old)
    new["source_git_commit"] = commit(root, "only timing opt-in and class")
    bind(root, new)
    return SimpleNamespace(root=root, old=old, new=new)


@pytest.fixture
def f(prototype, tmp_path):
    root = tmp_path / "case"
    shutil.copytree(prototype.root, root)
    checkpoint = root / "outputs/ppo_fsm_reference_p09_stable_v2/checkpoints/history/source.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"opaque fixture, not torch loaded")
    metadata = source_metadata(checkpoint, prototype.old, layout=ROLE_OBSERVATION_LAYOUT)
    sidecar = checkpoint.with_name("source_manifest.json")
    write_json(sidecar, metadata)
    return SimpleNamespace(root=root, old=copy.deepcopy(prototype.old), new=copy.deepcopy(prototype.new),
        checkpoint=checkpoint, sidecar=sidecar, metadata=metadata)


def record(f, **options):
    delta = sorted(p for p in set(f.old["files"]) | set(f.new["files"])
                   if f.old["files"].get(p) != f.new["files"].get(p))
    return m.build_migration_plan(f.checkpoint, f.new, allowed_changed_files=delta,
        reason="Review source partial-order timing, not equivalent physical behavior",
        timing_review={"reason": "Only nominal start order changes; preserve learned state"},
        project_root=f.root, **options)


def test_explicit_timing_only_preserves_layout_and_full_state_contract(f):
    immutable = f.checkpoint.read_bytes(), f.sidecar.read_bytes()
    r = record(f)
    t = r["nominal_timing_factor"]
    assert t["schema"] == m.TIMING_ONLY_SCHEMA
    assert t["physical_mdp_changed"] and t["nominal_control_changed"]
    assert t["nominal_provider"]["outside_class_bytes_identical"]
    assert not any(t[k] for k in ("reward_changed", "task_acceptance_changed", "physical_actuators_changed",
                                  "action_ranges_changed", "kernel_changed"))
    assert t["observation_contract"]["source_policy_contract"] == f.metadata["policy_contract"]
    assert t["observation_contract"]["source_policy_contract"] == t["observation_contract"]["target_policy_contract"]
    assert r["observation_dimension"] == 372 and r["action_dimension"] == 12
    assert r["preserve_actor_critic_optimizer_normalizer_rng_and_budget"] and r["discard_old_rollout_storage"]
    assert "preserve_complete_verified_Adam" in t["optimizer"]
    assert "instrumentation_observation_contract" not in r and "prior_factor" not in r
    path = f.root / "plan.json"
    write_json(path, r)
    verified = m.validate_migration_plan(f.checkpoint, f.new, path, project_root=f.root)
    assert verified["nominal_timing_factor"] == t
    assert immutable == (f.checkpoint.read_bytes(), f.sidecar.read_bytes())
    r["nominal_timing_factor"]["physical_mdp_changed"] = False
    write_json(path, r)
    with pytest.raises(ValueError, match="exactly bound"):
        m.validate_migration_plan(f.checkpoint, f.new, path, project_root=f.root)


@pytest.mark.parametrize("name", [n for n in NAMES if n != "stage_task_spec.yaml"])
def test_other_five_config_bytes_cannot_change(f, name):
    path = f.root / CONFIG / name
    path.write_bytes(path.read_bytes() + b"\n")
    bind(f.root, f.new)
    with pytest.raises(ValueError):
        record(f)


@pytest.mark.parametrize("case", ["missing_optin", "wrong_optin", "other_task", "outside_before",
    "outside_after", "same_class", "reward_source", "metadata_physics", "metadata_layout", "selected_hash", "removed_file"])
def test_scope_is_not_a_generic_same372_waiver(f, case):
    provider = f.root / m.SUPERVISOR
    if case in ("missing_optin", "wrong_optin", "other_task"):
        path = f.root / SPEC
        spec = yaml.safe_load(path.read_bytes())
        if case == "missing_optin": del spec["nominal"]["sequence_semantics"]
        elif case == "wrong_optin": spec["nominal"]["sequence_semantics"] = "unreviewed"
        else: spec["rear_leg_order"] = "RL_FIRST"
        path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    elif case == "outside_before": provider.write_bytes(provider.read_bytes().replace(b"LOCKED = 1", b"LOCKED = 2"))
    elif case == "outside_after": provider.write_bytes(provider.read_bytes().replace(b"    pass", b"    changed = True"))
    elif case == "same_class": provider.write_bytes(m._version_bytes(f.root, f.old, m.SUPERVISOR))
    elif case == "reward_source":
        path = f.root / PROTECTED
        path.write_bytes(path.read_bytes() + b"\n# unreviewed reward change\n")
    elif case == "metadata_physics": f.new["decision_hz"] = 30.
    elif case == "metadata_layout":
        f.metadata["policy_contract"]["rho"] = .8
        write_json(f.sidecar, f.metadata)
    bind(f.root, f.new)
    if case == "selected_hash": f.new["selected_configuration"]["stage_task_spec.yaml"]["sha256"] = "0" * 64
    elif case == "removed_file": del f.new["files"][PROTECTED]
    with pytest.raises(ValueError):
        record(f)


def test_other_factors_and_implicit_migration_are_rejected(f):
    with pytest.raises(ValueError, match="cannot mix"):
        record(f, video_review={"reason": "not mixed with control"})
    with pytest.raises(ValueError):
        m.build_migration_plan(f.checkpoint, f.new, allowed_changed_files=[SPEC, m.SUPERVISOR],
            reason="No explicit timing review", project_root=f.root)


def test_existing_cli_resume_migration_path_accepts_factor(f, monkeypatch):
    path = f.root / "plan.json"
    write_json(path, record(f))
    monkeypatch.setattr(cli, "PROJECT_ROOT", f.root)
    real_validate = m.validate_migration_plan
    monkeypatch.setattr(m, "validate_migration_plan", lambda cp, contract, p:
        real_validate(cp, contract, p, project_root=f.root))
    args = cli.parser().parse_args(["train", "--semantic-version", "v3", "--experiment-id",
        "fsm_reference_p09_stable_v2", "--run-dir", str(f.root / "runs/ppo_fsm_reference_p09_stable_v2/train/new"),
        "--expected-head", f.new["source_git_commit"], "--checkpoint", str(f.checkpoint),
        "--resume-migration", str(path), "--stage", "full_episode", "--from-phase", "P01",
        "--decisions", "128", "--seed", "1001", "--device", "cpu"])
    cli.validate_request(args)
    cli._preflight_checkpoint(args, f.new)
    assert args._migration_record["nominal_timing_factor"]["schema"] == m.TIMING_ONLY_SCHEMA
    assert args._warm_start_record is None and args._policy_migration_record is None


def test_real_cpu372_preserves_nonempty_adam_and_continues_one_update(prototype, tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("rsl_rl")
    from test_semantic_role_append_training import make
    from wlr50_clean.ppo import semantic_training as training
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    try:
        root = tmp_path / "actual"
        shutil.copytree(prototype.root, root)
        training.seed_training_rngs(1001)
        source, source_env = make(True)
        result = training.train_semantic(source, source_env, run_dir=root / "source_run",
            output_root=root / "source_output", stage="full_episode", decisions=128,
            contract=prototype.old, seed=1001)
        metadata = json.loads(Path(result["checkpoints"][-1]["manifest"]).read_text())
        for key in ("checkpoint_path", "checkpoint_sha256", "save_load_round_trip"): metadata.pop(key)
        source.alg.learning_rate = 1e-5
        for group in source.alg.optimizer.param_groups:
            group.update(lr=1e-5, betas=(.87, .996), eps=2e-8, weight_decay=1e-6)
        checkpoint = root / "learned372.pt"
        _, sidecar = training.save_semantic_checkpoint(source, checkpoint, metadata)
        saved = m.checkpoint_metadata(checkpoint)
        assert source.alg.optimizer.state_dict()["state"]
        immutable = checkpoint.read_bytes(), sidecar.read_bytes()
        fixture = SimpleNamespace(root=root, old=prototype.old, new=prototype.new, checkpoint=checkpoint)
        path = root / "plan.json"
        write_json(path, record(fixture))
        real_validate = m.validate_migration_plan
        monkeypatch.setattr(m, "validate_migration_plan", lambda cp, contract, p:
            real_validate(cp, contract, p, project_root=root))
        verified = m.validate_migration_plan(checkpoint, prototype.new, path)
        target, env = make(True)
        loaded = training.load_semantic_checkpoint(target, checkpoint, contract=prototype.new,
            seed=1001, migration=verified)
        for role in ("actor", "critic"):
            before, after = getattr(source.alg, role).state_dict(), getattr(target.alg, role).state_dict()
            assert before.keys() == after.keys() and all(torch.equal(v, after[k]) for k, v in before.items())
        assert training.state_hash(target.alg.optimizer.state_dict()) == saved["optimizer_state_sha256"]
        assert training.state_hash(training._normalizers(target)) == saved["normalizer_state_sha256"]
        assert training.capture_training_rng_state(seed=1001) == saved["training_rng_state"]
        assert target.alg.learning_rate == training.optimizer_learning_rate(target) == 1e-5
        assert target.alg.storage.step == 0 and target.alg.transition.actions is None and env.core.calls == 0
        assert tuple(target.alg.storage.observations["policy"].shape) == (128, 1, 372)
        assert tuple(target.alg.storage.actions.shape) == (128, 1, 12)
        for key in ("global_policy_decisions", "ppo_updates", "optimizer_steps", "stage_requested_decisions"):
            assert loaded[key] == saved[key]
        target.alg.storage.step = 1
        with pytest.raises(RuntimeError, match="partial old rollout"):
            training.load_semantic_checkpoint(target, checkpoint, contract=prototype.new, seed=1001, migration=verified)
        target.alg.storage.step = 0
        continued = training.train_semantic(target, env, run_dir=root / "target_run",
            output_root=root / "target_output", stage="full_episode", decisions=128,
            contract=prototype.new, seed=1001, resume_infos=loaded)
        final, _ = make(True)
        infos = training.load_semantic_checkpoint(final, Path(continued["checkpoints"][-1]["checkpoint"]),
            contract=prototype.new, seed=1001)
        assert (infos["global_policy_decisions"], infos["ppo_updates"], infos["optimizer_steps"]) == (256, 2, 40)
        assert training.state_hash(final.alg.optimizer.state_dict()) == training.state_hash(target.alg.optimizer.state_dict())
        assert infos["resume_ancestry"]["resume_migration"]["nominal_timing_factor"]["schema"] == m.TIMING_ONLY_SCHEMA
        assert immutable == (checkpoint.read_bytes(), sidecar.read_bytes())
    finally:
        torch.set_num_threads(threads)
