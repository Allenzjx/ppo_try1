"""Strict record guards plus official CPU372 load/update, never Isaac evidence."""
from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from wlr50_clean.ppo import semantic_migration as m
from wlr50_clean.ppo.semantic_transfer_roles import ROLE_OBSERVATION_LAYOUT
from test_semantic_role_observation_migration import (
    NAMES, OBS, PROTECTED, bind, commit, git, source_metadata, write_json,
)

SPEC = "configs/ppo_semantic_v3/stage_task_spec.yaml"
PROFILE = "configs/ppo_semantic_v3/execution_profile.yaml"
PHASES = tuple(f"P{i:02}" for i in range(6, 14))
OVERRIDES = {phase: {"front_left_hip": 60.0} for phase in PHASES}


def write_yaml(path, value):
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


@pytest.fixture(scope="module")
def prototype(tmp_path_factory):
    root = tmp_path_factory.mktemp("same372_authority_git")
    paths = sorted(m.SAME372_AUTHORITY_RUNTIME_FILES | {OBS, PROTECTED} |
                   {f"configs/ppo_semantic_v3/{name}" for name in NAMES})
    for relative in paths:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((m.PROJECT_ROOT / relative).read_bytes())
    # Explicit historical source, even when tests run after target configs land.
    spec = yaml.safe_load((root / SPEC).read_bytes())
    spec["revision"] = "diagonal_transfer_roles_v1"
    spec["nominal"].pop("phase_servo_rate_overrides_deg_s", None)
    profile = yaml.safe_load((root / PROFILE).read_bytes())
    profile["revision"] = "continuous_transfer_roles_v1_residual_authority"
    for phase in PHASES:
        profile["residual"]["phase_caps_full12"][phase][0] = 24
    write_yaml(root / SPEC, spec)
    write_yaml(root / PROFILE, profile)
    # Tiny syntactically real class boundary tests the byte guard only.
    (root / m.SUPERVISOR).write_bytes(
        b"LOCKED = 1\n\nclass NominalMotionProvider:\n    rate = 150\n\nclass TaskEvaluator:\n    pass\n")
    git(root, "init")
    git(root, "config", "user.name", "Same372 record test")
    git(root, "config", "user.email", "same372@example.invalid")
    old = {"source_git_commit": commit(root, "real same372 source bytes"),
        "files": dict.fromkeys(paths), "semantic_version": "v3", "experiment_id": "transfer_roles_v1",
        "frozen_A_files": {"frozen": "a"*64}, "physics_hz": 120., "decision_hz": 15.,
        "task_timeout_s": 200., "timeout_bootstrap": False, "rsl_rl_version": "5.0.1",
        "local_runtime_versions": {"fixture": "record_only"}}
    bind(root, old)
    spec["revision"] = "diagonal_transfer_roles_v2_fl_advisory_counterauthority"
    spec["nominal"]["phase_servo_rate_overrides_deg_s"] = copy.deepcopy(OVERRIDES)
    profile["revision"] = "continuous_transfer_roles_v2_fl_advisory_counterauthority"
    for phase in PHASES:
        profile["residual"]["phase_caps_full12"][phase][0] = 32
    write_yaml(root / SPEC, spec)
    write_yaml(root / PROFILE, profile)
    provider = root / m.SUPERVISOR
    provider.write_bytes(provider.read_bytes().replace(b"rate = 150", b"rate = 60"))
    new = copy.deepcopy(old)
    new["source_git_commit"] = commit(root, "real same372 target bytes")
    bind(root, new)
    checkpoint = root / "record_only.pt"
    checkpoint.write_bytes(b"opaque record validation fixture, not torch")
    return SimpleNamespace(root=root, old=old, new=new)


@pytest.fixture
def f(prototype, tmp_path):
    root = tmp_path / "case"
    shutil.copytree(prototype.root, root)
    cp = root / "record_only.pt"
    metadata = source_metadata(cp, prototype.old, layout=ROLE_OBSERVATION_LAYOUT)
    sidecar = cp.with_name("record_only_manifest.json")
    write_json(sidecar, metadata)
    return SimpleNamespace(root=root, checkpoint=cp, sidecar=sidecar, metadata=metadata,
                           old=copy.deepcopy(prototype.old), new=copy.deepcopy(prototype.new))


def record(f, **kwargs):
    return m.build_v3_warm_start_record(f.checkpoint, f.new, project_root=f.root, **kwargs)


def test_same372_is_identity_recipe_not_a_second_append_or_scale_migration(f):
    frozen = (f.checkpoint.read_bytes(), f.sidecar.read_bytes())
    r = record(f)
    t = r["observation_same_layout_transition"]
    assert t["schema"] == m.SAME372_AUTHORITY_SCHEMA
    assert t["source_observation_dimension"] == t["target_observation_dimension"] == 372
    assert t["source_observation_layout"] == t["target_observation_layout"] == ROLE_OBSERVATION_LAYOUT
    assert t["source_policy_contract"] == t["target_policy_contract"] == f.metadata["policy_contract"]
    assert t["source_schema_sha256"] == t["target_schema_sha256"]
    assert t["parameter_mapping"] == "identity_all_parameters_and_buffers"
    assert t["nominal_provider"]["outside_class_bytes_identical"] is True
    assert t["authority_changes"]["nominal.phase_servo_rate_overrides_deg_s"] == OVERRIDES
    assert r["network"]["observation_dimension"] == 372
    assert r["optimizer"]["initial_learning_rate"] == 3e-5
    assert r["source_stage_requested_decisions"] == r["target_stage_requested_decisions"] == f.metadata["stage_requested_decisions"]
    assert r["source_global_policy_decisions"] == f.metadata["global_policy_decisions"]
    assert r["new_mdp_origin_global_policy_decisions"] == f.metadata["new_mdp_origin_global_policy_decisions"]
    assert r["old_rollout_buffer_inherited"] is False and r["physical_state_inherited"] is False
    assert not {"observation_append_transition", "observation_scale_transition", "policy_kernel_transition"} & r.keys()
    assert record(f) == r and frozen == (f.checkpoint.read_bytes(), f.sidecar.read_bytes())


@pytest.mark.parametrize("case", [
    "global150", "one_override_missing", "wrong_phase", "other_servo", "rate_bool", "rate61", "task",
    "FLcap33", "earlyFLcap", "other_cap", "residual_rate", "wheel_rate", "headroom", "revision",
])
def test_only_exact_two_config_changes_are_allowed(f, case):
    path = f.root / (SPEC if case in {"global150", "one_override_missing", "wrong_phase", "other_servo", "rate_bool", "rate61", "task"} else PROFILE)
    value = yaml.safe_load(path.read_bytes())
    if case == "global150": value["nominal"]["servo_handoff_rate_deg_s"] = 60.
    elif case == "one_override_missing": del value["nominal"]["phase_servo_rate_overrides_deg_s"]["P13"]
    elif case == "wrong_phase": value["nominal"]["phase_servo_rate_overrides_deg_s"]["P05"] = {"front_left_hip": 60.}
    elif case == "other_servo": value["nominal"]["phase_servo_rate_overrides_deg_s"]["P07"]["rear_right_hip"] = 60.
    elif case == "rate_bool": value["nominal"]["phase_servo_rate_overrides_deg_s"]["P07"]["front_left_hip"] = True
    elif case == "rate61": value["nominal"]["phase_servo_rate_overrides_deg_s"]["P07"]["front_left_hip"] = 61.
    elif case == "task": value["geometry"]["workspace_min_m"] = -.3
    elif case == "FLcap33": value["residual"]["phase_caps_full12"]["P07"][0] = 33
    elif case == "earlyFLcap": value["residual"]["phase_caps_full12"]["P05"][0] = 32
    elif case == "other_cap": value["residual"]["phase_caps_full12"]["P07"][1] += 1
    elif case == "residual_rate": value["residual"]["servo_rate_deg_s"] = 70.
    elif case == "wheel_rate": value["residual"]["wheel_rate_rad_s2"] = 2.
    elif case == "headroom": value["residual"]["policy_headroom_mode"] = None
    else: value["revision"] = "unreviewed"
    write_yaml(path, value)
    bind(f.root, f.new)
    with pytest.raises(ValueError, match="same372"): record(f)


@pytest.mark.parametrize("name", ["observation_schema.json", "reward_config.yaml", "action_schema.json", "quality_score.yaml"])
def test_other_four_configuration_bytes_are_unchanged_even_comments(f, name):
    path = f.root / f"configs/ppo_semantic_v3/{name}"
    path.write_bytes(path.read_bytes() + b"\n")
    bind(f.root, f.new)
    with pytest.raises(ValueError, match="same372"): record(f)


@pytest.mark.parametrize("case", ["outside_before", "outside_after", "no_provider_change", "unrelated_file", "metadata", "selection_hash", "selection_path", "selection_extra", "kernel"])
def test_provider_and_runtime_scope_is_not_a_blanket_waiver(f, case):
    path = f.root / m.SUPERVISOR
    if case == "outside_before": path.write_bytes(path.read_bytes().replace(b"LOCKED = 1", b"LOCKED = 2"))
    elif case == "outside_after": path.write_bytes(path.read_bytes().replace(b"class TaskEvaluator:\n    pass", b"class TaskEvaluator:\n    changed = True"))
    elif case == "no_provider_change": path.write_bytes(m._version_bytes(f.root, f.old, m.SUPERVISOR))
    elif case == "unrelated_file":
        path = f.root / PROTECTED
        path.write_bytes(path.read_bytes() + b"\n# unreviewed\n")
    bind(f.root, f.new)
    if case == "metadata": f.new["extra"] = True
    elif case == "selection_hash": f.new["selected_configuration"]["stage_task_spec.yaml"]["sha256"] = "0"*64
    elif case == "selection_path": f.new["selected_configuration"]["stage_task_spec.yaml"]["path"] = PROFILE
    elif case == "selection_extra": f.new["selected_configuration"]["extra"] = {}
    with pytest.raises(ValueError):
        record(f, **({"target_policy_version": f.metadata["policy_contract"]["version"]} if case == "kernel" else {}))


@pytest.fixture(scope="module")
def official_cpu_runtime():
    torch = pytest.importorskip("torch")
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    # Generate a CPU-only source receipt, not a CUDA-to-CPU restore adapter.
    # This also keeps broad CPU-suite collection independent of GPU visibility.
    with pytest.MonkeyPatch.context() as cpu:
        cpu.setattr(torch.cuda, "is_available", lambda: False)
        try:
            yield
        finally:
            torch.set_num_threads(threads)


@pytest.fixture(scope="module")
def learned_cpu_source(prototype, tmp_path_factory, official_cpu_runtime):
    """Real learned372 checkpoint and real hash-bound temporary git configs.

    Only the core's physics is fake; RSL actor/std/critic, Adam, GAE, storage,
    checkpoint IO and migration validation below are the production paths.
    """
    torch = pytest.importorskip("torch")
    from test_semantic_role_append_training import make
    from wlr50_clean.ppo import semantic_training as training
    assert not torch.cuda.is_available()
    root = tmp_path_factory.mktemp("same372_actual_cpu") / "case"
    shutil.copytree(prototype.root, root)
    training.seed_training_rngs(1001)
    source, env = make(True)
    first_weights = {role: getattr(source.alg, role).mlp[0].weight.detach().clone()
                     for role in ("actor", "critic")}
    std_weights = source.alg.actor.state_dict()["mlp.4.weight"][12:].clone()
    result = training.train_semantic(source, env, run_dir=root / "source_run",
        output_root=root / "source_output", stage="full_episode", decisions=128,
        contract=prototype.old, seed=1001)
    for role in ("actor", "critic"):
        current = getattr(source.alg, role).mlp[0].weight.detach()
        assert torch.count_nonzero(current[:, 324:]) > 0
        assert not torch.equal(current[:, 324:], first_weights[role][:, 324:])
    assert not torch.equal(source.alg.actor.state_dict()["mlp.4.weight"][12:], std_weights)
    assert source.alg.optimizer.state_dict()["state"]
    metadata = json.loads(Path(result["checkpoints"][-1]["manifest"]).read_text())
    for key in ("checkpoint_path", "checkpoint_sha256", "save_load_round_trip"):
        metadata.pop(key)
    metadata["new_mdp_origin_global_policy_decisions"] = 0
    source.alg.learning_rate = 1e-5
    for group in source.alg.optimizer.param_groups:
        group["lr"] = 1e-5
    checkpoint = root / "actual_learned372.pt"
    _, sidecar = training.save_semantic_checkpoint(source, checkpoint, metadata)
    metadata = json.loads(sidecar.read_text())
    plan = m.build_v3_warm_start_record(checkpoint, prototype.new, project_root=root)
    yield SimpleNamespace(root=root, old=prototype.old, new=prototype.new,
        source=source, checkpoint=checkpoint, sidecar=sidecar, metadata=metadata, plan=plan)


def actual_project_validator(monkeypatch, source):
    original = m.build_v3_warm_start_record
    # Redirect only project-root lookup, never bypass the real record validator.
    monkeypatch.setattr(m, "build_v3_warm_start_record",
        lambda checkpoint, contract, **kwargs: original(
            checkpoint, contract, project_root=source.root, **kwargs))


def test_actual372_weights_std_adam_rng_initial_exact_reload_and_fresh_update(
        learned_cpu_source, monkeypatch, tmp_path):
    import torch
    from test_semantic_role_append_training import make
    from wlr50_clean.ppo import semantic_training as training
    s = learned_cpu_source
    actual_project_validator(monkeypatch, s)
    immutable = (s.checkpoint.read_bytes(), s.sidecar.read_bytes())
    observed_source_adam = []
    original_load = training.load_checkpoint_round_trip

    def checked_official_load(runner, checkpoint):
        infos = original_load(runner, checkpoint)
        if checkpoint == s.checkpoint:
            state = runner.alg.optimizer.state_dict()
            assert state["state"] and all(g["lr"] == 1e-5 for g in state["param_groups"])
            observed_source_adam.append(training.state_hash(state))
        return infos

    monkeypatch.setattr(training, "load_checkpoint_round_trip", checked_official_load)
    target, env = make(True)
    loaded = training.load_semantic_checkpoint(target, s.checkpoint, contract=s.new,
        seed=1001, warm_start=s.plan)
    assert observed_source_adam == [s.metadata["optimizer_state_sha256"]]
    assert training.capture_training_rng_state(seed=1001) == s.metadata["training_rng_state"]
    for role in ("actor", "critic"):
        before, after = getattr(s.source.alg, role).state_dict(), getattr(target.alg, role).state_dict()
        assert before.keys() == after.keys()
        assert all(torch.equal(before[key], after[key]) for key in before)
        assert torch.count_nonzero(after["mlp.0.weight"][:, 324:]) > 0
    assert loaded["normalizer_state_sha256"] == training.state_hash(training._normalizers(target))
    assert training._normalizers(target) == {"actor": {}, "critic": {}}
    assert (loaded["global_policy_decisions"], loaded["ppo_updates"], loaded["optimizer_steps"]) == (128, 1, 20)
    assert loaded["stage_requested_decisions"] == s.metadata["stage_requested_decisions"]
    assert loaded["new_mdp_origin_global_policy_decisions"] == 0
    assert loaded["training_rng_state"] == s.metadata["training_rng_state"]
    assert target.alg.optimizer.state_dict()["state"] == {}
    assert target.alg.learning_rate == 3e-5
    assert all(g["lr"] == 3e-5 for g in target.alg.optimizer.param_groups)
    assert target.alg.storage.step == 0 and target.alg.transition.actions is None
    assert tuple(target.alg.storage.observations["policy"].shape) == (128, 1, 372)
    assert env.core.calls == 0 and env.core.resets == 1
    # Same input kernel/value identity, not old/new projected-action equivalence.
    observations = env.get_observations()
    with torch.inference_mode():
        for role in ("actor", "critic"):
            assert torch.equal(getattr(s.source.alg, role)(observations),
                               getattr(target.alg, role)(observations))
    initial = tmp_path / m.v3_warm_start_checkpoint_name(s.plan)
    _, initial_sidecar = training.save_semantic_checkpoint(target, initial,
        {**loaded, "runtime_contract": s.new})
    initial_metadata = json.loads(initial_sidecar.read_text())
    assert initial_metadata["save_load_round_trip"] is True
    assert initial_metadata["runner_config"] == s.metadata["runner_config"]
    assert initial_metadata["policy_contract"] == s.metadata["policy_contract"]
    reloaded, _ = make(True)
    reloaded_infos = training.load_semantic_checkpoint(reloaded, initial, contract=s.new, seed=1001)
    assert reloaded_infos["new_mdp_warm_start"] == s.plan
    assert training.state_hash(reloaded.alg.optimizer.state_dict()) == training.state_hash(target.alg.optimizer.state_dict())
    assert training.parameter_hash(reloaded.alg.actor) == training.parameter_hash(s.source.alg.actor)
    result = training.train_semantic(target, env, run_dir=tmp_path / "target_run",
        output_root=tmp_path / "target_output", stage="full_episode", decisions=128,
        contract=s.new, seed=1001, resume_infos=loaded)
    assert (result["global_policy_decisions"], result["ppo_updates_this_run"], result["optimizer_steps_this_run"]) == (256, 1, 20)
    assert result["finite_nonzero_gradient_observed"] is True
    assert training.parameter_hash(target.alg.actor) != s.metadata["actor_parameter_sha256"]
    fresh, _ = make(True)
    final = training.load_semantic_checkpoint(fresh, Path(result["checkpoints"][-1]["checkpoint"]), contract=s.new, seed=1001)
    assert (final["global_policy_decisions"], final["ppo_updates"], final["optimizer_steps"]) == (256, 2, 40)
    assert final["stage_requested_decisions"]["full_episode"] == 256
    assert final["new_mdp_origin_global_policy_decisions"] == 0
    assert training.state_hash(fresh.alg.optimizer.state_dict()) == training.state_hash(target.alg.optimizer.state_dict())
    assert immutable == (s.checkpoint.read_bytes(), s.sidecar.read_bytes())


@pytest.mark.parametrize("case", ["target324", "partial_rollout", "mixed_append", "mixed_scale", "ordinary_new_contract"])
def test_actual372_loader_rejects_wrong_layout_history_and_implicit_migration(
        learned_cpu_source, monkeypatch, case):
    from test_semantic_role_append_training import make
    from wlr50_clean.ppo import semantic_training as training
    s = learned_cpu_source
    actual_project_validator(monkeypatch, s)
    target, _ = make(case != "target324")
    plan = copy.deepcopy(s.plan)
    if case == "partial_rollout": target.alg.storage.step = 1
    elif case == "mixed_append": plan["observation_append_transition"] = {"target_observation_dimension": 372}
    elif case == "mixed_scale": plan["observation_scale_transition"] = {"columns": [210, 222]}
    before = training.parameter_hash(target.alg.actor)
    expected = {"target324": "same372 authority", "partial_rollout": "fresh N1",
        "mixed_append": "binding changed", "mixed_scale": "binding changed",
        "ordinary_new_contract": "contract mismatch"}[case]
    with pytest.raises(RuntimeError, match=expected):
        training.load_semantic_checkpoint(target, s.checkpoint, contract=s.new,
            seed=1001, warm_start=None if case == "ordinary_new_contract" else plan)
    assert training.parameter_hash(target.alg.actor) == before
