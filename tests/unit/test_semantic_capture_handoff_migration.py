"""One-factor FL nominal handoff migration: real byte guards and CPU372 restore."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
import yaml

from wlr50_clean.ppo import semantic_cli as cli
from wlr50_clean.ppo import semantic_migration as m
from wlr50_clean.ppo.semantic_transfer_roles import ROLE_OBSERVATION_LAYOUT
from test_semantic_role_observation_migration import NAMES, PROTECTED, commit, git, source_metadata, write_json

CONFIG = "configs/ppo_fsm_reference_p09_stable_v2"
SPEC = f"{CONFIG}/stage_task_spec.yaml"
V1 = "current_FL_capture_wheel_continuation_v1"
V2 = "current_FL_capture_wheel_continuation_to_handoff_v2"


def bind(root, contract):
    contract["files"] = {p: m.file_sha(root / p) for p in contract["files"]}
    contract["runtime_content_sha256"] = m.digest(contract["files"])
    contract["selected_configuration"] = {
        name: {"path": f"{CONFIG}/{name}", "sha256": contract["files"][f"{CONFIG}/{name}"]}
        for name in NAMES}


@pytest.fixture(scope="module")
def prototype(tmp_path_factory):
    root = tmp_path_factory.mktemp("capture_handoff_git")
    paths = {f"{CONFIG}/{name}" for name in NAMES} | {m.SUPERVISOR, PROTECTED} | {
        f"src/wlr50_clean/ppo/{name}.py" for name in ("semantic_migration", "semantic_cli", "semantic_training")}
    for relative in sorted(paths):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((m.PROJECT_ROOT / relative).read_bytes())
    spec = yaml.safe_load((root / SPEC).read_bytes())
    # This fixture explicitly represents the old v1 boundary even after v2 lands.
    spec["nominal"]["p05_pending_capture"] = V1
    (root / SPEC).write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    provider = root / m.SUPERVISOR
    provider.write_text("LOCKED = 1\n\nclass NominalMotionProvider:\n    capture = 'v1'\n\nclass TaskEvaluator:\n    pass\n", encoding="utf-8")
    git(root, "init")
    git(root, "config", "user.name", "Capture handoff fixture")
    git(root, "config", "user.email", "capture-handoff@example.invalid")
    old = {"source_git_commit": commit(root, "explicit old nominal v1"),
        "files": dict.fromkeys(sorted(paths)), "semantic_version": "v3",
        "experiment_id": "fsm_reference_p09_stable_v2",
        "frozen_A_files": {"frozen": "a" * 64}, "physics_hz": 120., "decision_hz": 15.,
        "task_timeout_s": 200., "timeout_bootstrap": False, "rsl_rl_version": "5.0.1",
        "local_runtime_versions": {"fixture": "not Isaac"}}
    bind(root, old)
    spec["nominal"]["p05_pending_capture"] = V2
    (root / SPEC).write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    provider.write_bytes(provider.read_bytes().replace(b"capture = 'v1'", b"capture = 'v2'"))
    new = copy.deepcopy(old)
    new["source_git_commit"] = commit(root, "only nominal capture handoff v2")
    bind(root, new)
    return SimpleNamespace(root=root, old=old, new=new)


@pytest.fixture
def f(prototype, tmp_path, monkeypatch):
    root = tmp_path / "case"
    shutil.copytree(prototype.root, root)
    checkpoint = root / "outputs/ppo_fsm_reference_p09_stable_v2/checkpoints/history/source.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"opaque record-only fixture; never torch loaded")
    metadata = source_metadata(checkpoint, prototype.old, layout=ROLE_OBSERVATION_LAYOUT)
    metadata["optimizer_learning_rate"] = 1e-5
    sidecar = checkpoint.with_name("source_manifest.json")
    write_json(sidecar, metadata)
    # Only the fixture's real historical Git commit substitutes for reviewed 7db.
    monkeypatch.setattr(m, "CAPTURE_HANDOFF_SOURCE_HEAD", prototype.old["source_git_commit"])
    return SimpleNamespace(root=root, old=copy.deepcopy(prototype.old), new=copy.deepcopy(prototype.new),
                           checkpoint=checkpoint, sidecar=sidecar, metadata=metadata)


def record(f, **kwargs):
    return m.build_v3_warm_start_record(f.checkpoint, f.new, project_root=f.root, **kwargs)


def test_true_nominal_mdp_same372_identity_and_source_lr(f):
    original = (f.checkpoint.read_bytes(), f.sidecar.read_bytes())
    r = record(f)
    t = r["observation_same_layout_transition"]
    assert t["schema"] == m.CAPTURE_HANDOFF_SAME372_SCHEMA
    assert t["physical_mdp_changed"] and t["nominal_control_changed"]
    assert t["nominal_provider"]["outside_class_bytes_identical"]
    assert not any(t[k] for k in ("kernel_changed", "reward_changed", "task_acceptance_changed",
                                  "physical_actuators_changed", "action_ranges_changed"))
    assert t["source_policy_contract"] == t["target_policy_contract"] == f.metadata["policy_contract"]
    assert t["source_schema_sha256"] == t["target_schema_sha256"]
    assert t["parameter_mapping"] == "identity_all_parameters_and_buffers"
    assert r["network"]["observation_dimension"] == 372
    assert r["optimizer"]["initial_learning_rate"] == 1e-5
    assert r["optimizer"]["learning_rate_policy"] == "preserve_verified_source_effective_learning_rate"
    assert r["optimizer"]["preserve_source_group_options"] is True
    assert r["optimizer"]["state"] == "reset_all_moments"
    assert r["target_stage_requested_decisions"] == f.metadata["stage_requested_decisions"]
    assert r["new_mdp_origin_global_policy_decisions"] == f.metadata["new_mdp_origin_global_policy_decisions"]
    assert r["source_global_policy_decisions"] == f.metadata["global_policy_decisions"]
    assert not r["exact_mdp_resume"] and not r["old_rollout_buffer_inherited"] and not r["physical_state_inherited"]
    assert not {"observation_append_transition", "observation_scale_transition", "policy_kernel_transition"} & r.keys()
    assert record(f) == r and original == (f.checkpoint.read_bytes(), f.sidecar.read_bytes())


@pytest.mark.parametrize("case", ["head", "experiment", "kernel", "lr_missing", "lr_boolean", "lr_negative", "budget", "runtime"])
def test_source_or_runtime_metadata_cannot_be_relaxed(f, case):
    options = {}
    if case == "head": f.metadata["runtime_contract"]["source_git_commit"] = "0" * 40
    elif case == "experiment": f.metadata["runtime_contract"]["experiment_id"] = "transfer_roles_v1"
    elif case == "kernel": options["target_policy_version"] = f.metadata["policy_contract"]["version"]
    elif case == "lr_missing": f.metadata.pop("optimizer_learning_rate")
    elif case == "lr_boolean": f.metadata["optimizer_learning_rate"] = True
    elif case == "lr_negative": f.metadata["optimizer_learning_rate"] = -1
    elif case == "budget": f.metadata["stage_requested_decisions"] = {"full_episode": 0}
    else: f.new["decision_hz"] = 30.
    write_json(f.sidecar, f.metadata)
    with pytest.raises(ValueError): record(f, **options)


@pytest.mark.parametrize("name", [n for n in NAMES if n != "stage_task_spec.yaml"])
def test_other_five_config_bytes_remain_exact(f, name):
    path = f.root / CONFIG / name
    path.write_bytes(path.read_bytes() + b"\n")
    bind(f.root, f.new)
    with pytest.raises(ValueError): record(f)


@pytest.mark.parametrize("case", ["no_optin", "wrong_optin", "other_task", "outside_before", "outside_after",
                                  "same_class", "unrelated_source", "selected_hash", "removed_file"])
def test_one_flag_class_only_scope(f, case):
    provider = f.root / m.SUPERVISOR
    if case in ("no_optin", "wrong_optin", "other_task"):
        path = f.root / SPEC
        spec = yaml.safe_load(path.read_bytes())
        if case == "no_optin": spec["nominal"]["p05_pending_capture"] = V1
        elif case == "wrong_optin": spec["nominal"]["p05_pending_capture"] = "unreviewed_v3"
        else: spec["rear_leg_order"] = "RL_FIRST"
        path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    elif case == "outside_before": provider.write_bytes(provider.read_bytes().replace(b"LOCKED = 1", b"LOCKED = 2"))
    elif case == "outside_after":
        original = provider.read_bytes()
        changed = original.replace(b"    pass", b"    changed = True")
        assert changed != original  # Exercise the mutation on both LF and CRLF.
        provider.write_bytes(changed)
    elif case == "same_class": provider.write_bytes(m._version_bytes(f.root, f.old, m.SUPERVISOR))
    elif case == "unrelated_source":
        path = f.root / PROTECTED
        path.write_bytes(path.read_bytes() + b"\n# no unrelated semantic change\n")
    bind(f.root, f.new)
    if case == "selected_hash": f.new["selected_configuration"]["stage_task_spec.yaml"]["sha256"] = "0" * 64
    elif case == "removed_file": del f.new["files"][PROTECTED]
    with pytest.raises(ValueError): record(f)


def arguments(f, *, warm_start=True):
    values = ["train", "--semantic-version", "v3", "--experiment-id", "fsm_reference_p09_stable_v2",
        "--run-dir", str(f.root / "runs/ppo_fsm_reference_p09_stable_v2/train/new"),
        "--expected-head", f.new["source_git_commit"], "--checkpoint", str(f.checkpoint),
        "--stage", "full_episode", "--from-phase", "P01", "--decisions", "128", "--seed", "1001"]
    return cli.parser().parse_args(values + (["--new-mdp-warm-start"] if warm_start else []))


def test_cli_same_namespace_is_explicit_and_initial_is_immutable(f, monkeypatch):
    monkeypatch.setattr(cli, "PROJECT_ROOT", f.root)
    args = arguments(f)
    cli.validate_request(args)
    cli._preflight_checkpoint(args, f.new)
    assert args._warm_start_record["observation_same_layout_transition"]["schema"] == m.CAPTURE_HANDOFF_SAME372_SCHEMA
    implicit = arguments(f, warm_start=False)
    cli.validate_request(implicit)
    with pytest.raises(ValueError, match="runtime changed"):
        cli._preflight_checkpoint(implicit, f.new)
    initial = f.root / "outputs/ppo_fsm_reference_p09_stable_v2/checkpoints/history" / m.v3_warm_start_checkpoint_name(args._warm_start_record)
    initial.write_bytes(b"immutable already-published initial fixture")
    with pytest.raises(ValueError, match="already exists"):
        cli._preflight_checkpoint(arguments(f), f.new)


def test_real_cpu372_load_keeps_all_tensors_rng_lr_options_and_counters(prototype, tmp_path, monkeypatch):
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
        monkeypatch.setattr(m, "CAPTURE_HANDOFF_SOURCE_HEAD", prototype.old["source_git_commit"])
        training.seed_training_rngs(1001)
        source, source_env = make(True)
        result = training.train_semantic(source, source_env, run_dir=root / "source_run",
            output_root=root / "source_output", stage="full_episode", decisions=128, contract=prototype.old, seed=1001)
        metadata = json.loads(Path(result["checkpoints"][-1]["manifest"]).read_text())
        for key in ("checkpoint_path", "checkpoint_sha256", "save_load_round_trip"): metadata.pop(key)
        metadata["new_mdp_origin_global_policy_decisions"] = 0
        source.alg.learning_rate = 1e-5
        for group in source.alg.optimizer.param_groups:
            group.update(lr=1e-5, betas=(.87, .996), eps=2e-8, weight_decay=1e-6)
        checkpoint = root / "learned372.pt"
        _, sidecar = training.save_semantic_checkpoint(source, checkpoint, metadata)
        immutable = (checkpoint.read_bytes(), sidecar.read_bytes())
        assert source.alg.optimizer.state_dict()["state"]
        plan = m.build_v3_warm_start_record(checkpoint, prototype.new, project_root=root)
        original_build = m.build_v3_warm_start_record
        monkeypatch.setattr(m, "build_v3_warm_start_record",
            lambda cp, contract, **kwargs: original_build(cp, contract, project_root=root, **kwargs))
        target, env = make(True)
        loaded = training.load_semantic_checkpoint(target, checkpoint, contract=prototype.new, seed=1001, warm_start=plan)
        assert training.capture_training_rng_state(seed=1001) == loaded["training_rng_state"]
        for role in ("actor", "critic"):
            before, after = getattr(source.alg, role).state_dict(), getattr(target.alg, role).state_dict()
            assert before.keys() == after.keys() and all(torch.equal(v, after[k]) for k, v in before.items())
        assert training._normalizers(target) == {"actor": {}, "critic": {}}
        assert target.alg.optimizer.state_dict()["state"] == {}
        old_group, new_group = source.alg.optimizer.param_groups[0], target.alg.optimizer.param_groups[0]
        assert all(new_group[k] == old_group[k] for k in source.alg.optimizer.defaults)
        assert target.alg.learning_rate == training.optimizer_learning_rate(target) == 1e-5
        assert target.alg.storage.step == 0 and target.alg.transition.actions is None
        assert tuple(target.alg.storage.actions.shape) == (128, 1, 12)
        assert tuple(target.alg.storage.observations["policy"].shape) == (128, 1, 372)
        assert (loaded["global_policy_decisions"], loaded["ppo_updates"], loaded["optimizer_steps"]) == (128, 1, 20)
        assert loaded["stage_requested_decisions"] == {"smoke": 0, "full_episode": 128, "phase_suffix": 0}
        assert env.core.calls == 0
        initial = root / m.v3_warm_start_checkpoint_name(plan)
        training.save_semantic_checkpoint(target, initial, {**loaded, "runtime_contract": prototype.new})
        fresh, _ = make(True)
        initial_infos = training.load_semantic_checkpoint(fresh, initial, contract=prototype.new, seed=1001)
        assert initial_infos["new_mdp_warm_start"] == plan
        assert training.state_hash(fresh.alg.optimizer.state_dict()) == training.state_hash(target.alg.optimizer.state_dict())
        # Neither an old partial rollout nor implicit exact resume may cross this boundary.
        stale, _ = make(True)
        stale.alg.storage.step = 1
        with pytest.raises(RuntimeError, match="fresh N1"):
            training.load_semantic_checkpoint(stale, checkpoint, contract=prototype.new, seed=1001, warm_start=plan)
        implicit, _ = make(True)
        with pytest.raises(RuntimeError, match="contract mismatch"):
            training.load_semantic_checkpoint(implicit, checkpoint, contract=prototype.new, seed=1001)
        continued = training.train_semantic(target, env, run_dir=root / "target_run",
            output_root=root / "target_output", stage="full_episode", decisions=128,
            contract=prototype.new, seed=1001, resume_infos=loaded)
        assert (continued["global_policy_decisions"], continued["ppo_updates_this_run"], continued["optimizer_steps_this_run"]) == (256, 1, 20)
        final, _ = make(True)
        final_infos = training.load_semantic_checkpoint(final, Path(continued["checkpoints"][-1]["checkpoint"]),
                                                       contract=prototype.new, seed=1001)
        assert (final_infos["global_policy_decisions"], final_infos["ppo_updates"], final_infos["optimizer_steps"]) == (256, 2, 40)
        assert final_infos["stage_requested_decisions"]["full_episode"] == 256
        assert final_infos["new_mdp_origin_global_policy_decisions"] == 0
        assert training.state_hash(final.alg.optimizer.state_dict()) == training.state_hash(target.alg.optimizer.state_dict())
        assert immutable == (checkpoint.read_bytes(), sidecar.read_bytes())
    finally:
        torch.set_num_threads(threads)
