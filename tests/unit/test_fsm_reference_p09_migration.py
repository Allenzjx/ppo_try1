"""Bounded SAME372 reference/P09 migration guards and official CPU restoration."""
from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from wlr50_clean.ppo import semantic_cli as cli
from wlr50_clean.ppo import semantic_migration as migration
from wlr50_clean.ppo.semantic_transfer_roles import ROLE_OBSERVATION_LAYOUT
from test_all_stage_migration import prototype as all_stage_prototype, same372_prototype
from test_semantic_role_observation_migration import NAMES, PROTECTED, commit, source_metadata, write_json

SOURCE_CONFIG = "configs/ppo_all_stage_acceptance_v1"
TARGET_CONFIG = "configs/ppo_fsm_reference_p09_stable_v2"


def bind_target(root, contract):
    contract["files"] = {path: migration.file_sha(root / path) for path in contract["files"]}
    contract["runtime_content_sha256"] = migration.digest(contract["files"])
    contract["selected_configuration"] = {
        name: {"path": f"{TARGET_CONFIG}/{name}", "sha256": contract["files"][f"{TARGET_CONFIG}/{name}"]}
        for name in NAMES}


@pytest.fixture(scope="module")
def prototype(all_stage_prototype, tmp_path_factory):
    root = tmp_path_factory.mktemp("fsm_p09_record") / "case"
    shutil.copytree(all_stage_prototype.root, root)
    old = copy.deepcopy(all_stage_prototype.new)
    new = copy.deepcopy(old)
    new["experiment_id"] = "fsm_reference_p09_stable_v2"
    for name in NAMES:
        target = root / TARGET_CONFIG / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / SOURCE_CONFIG / name).read_bytes())
        new["files"][f"{TARGET_CONFIG}/{name}"] = None
    for name, changed in (
            ("stage_task_spec.yaml", {"revision": "fsm_reference_p09_stable_v2",
                "p09_lift_semantics": "functional_lift_edge_v2",
                "reference_nominal_semantics": "successful_fsm_derived_v2"}),
            ("execution_profile.yaml", {"revision": "fsm_reference_p09_stable_v2_shared_source_derived_nominal",
                "nominal_geometry_advisory": "functional_rr_preplace_nominal_advisory_v2"})):
        path = root / TARGET_CONFIG / name
        value = yaml.safe_load(path.read_bytes())
        value.update(changed)
        path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    supervisor = root / migration.SUPERVISOR
    supervisor.write_bytes(supervisor.read_bytes() + b"\n# explicit functional-lift fixture revision\n")
    new["source_git_commit"] = commit(root, "bounded functional P09 target")
    bind_target(root, new)
    return SimpleNamespace(root=root, old=old, new=new)


@pytest.fixture
def f(prototype, tmp_path):
    root = tmp_path / "case"
    shutil.copytree(prototype.root, root)
    checkpoint = root / "outputs/ppo_all_stage_acceptance_v1/checkpoints/history/source.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"opaque metadata-only checkpoint fixture; not a torch load")
    metadata = source_metadata(checkpoint, prototype.old, layout=ROLE_OBSERVATION_LAYOUT)
    metadata.update(global_policy_decisions=140544, ppo_updates=1063, optimizer_steps=21260,
        optimizer_learning_rate=1e-5, new_mdp_origin_global_policy_decisions=10112,
        stage_requested_decisions={"smoke": 0, "full_episode": 65024, "phase_suffix": 65408})
    sidecar = checkpoint.with_name("source_manifest.json")
    write_json(sidecar, metadata)
    return SimpleNamespace(root=root, old=copy.deepcopy(prototype.old), new=copy.deepcopy(prototype.new),
        checkpoint=checkpoint, sidecar=sidecar, metadata=metadata)


def record(f, **kwargs):
    return migration.build_v3_warm_start_record(f.checkpoint, f.new, project_root=f.root, **kwargs)


def test_explicit_same372_recipe_preserves_source_lr_counters_and_budget(f):
    result = record(f)
    transition = result["observation_same_layout_transition"]
    assert transition["schema"] == migration.FSM_REFERENCE_P09_SCHEMA
    assert transition["source_observation_dimension"] == transition["target_observation_dimension"] == 372
    assert transition["source_observation_layout"] == transition["target_observation_layout"] == ROLE_OBSERVATION_LAYOUT
    assert transition["source_schema_sha256"] == transition["target_schema_sha256"]
    assert transition["source_policy_contract"] == transition["target_policy_contract"] == f.metadata["policy_contract"]
    assert transition["parameter_mapping"] == "identity_all_parameters_and_buffers"
    assert transition["kernel_changed"] is False and transition["action_ranges_changed"] is False
    assert result["optimizer"]["initial_learning_rate"] == 1e-5
    assert result["optimizer"]["state"] == "reset_all_moments"
    assert result["optimizer"]["learning_rate_policy"] == "preserve_verified_source_effective_learning_rate"
    assert result["source_global_policy_decisions"] == 140544
    assert result["target_stage_requested_decisions"] == f.metadata["stage_requested_decisions"]
    assert result["new_mdp_origin_global_policy_decisions"] == 10112
    assert result["old_rollout_buffer_inherited"] is False
    assert result["physical_state_inherited"] is False
    assert result["experiment_transition"]["target_configuration_namespace"] == "ppo_fsm_reference_p09_stable_v2"
    assert not result["return_horizon_transition"]["reward_discount_changed"]
    assert not result["return_horizon_transition"]["gae_estimator_changed"]


@pytest.mark.parametrize("case", ["source", "lr_missing", "lr_boolean", "lr_negative", "budget", "frozen", "kernel"])
def test_wrong_source_lr_budget_or_protected_runtime_is_rejected(f, case):
    options = {}
    if case == "source": f.metadata["runtime_contract"]["experiment_id"] = "transfer_roles_v1"
    elif case == "lr_missing": f.metadata.pop("optimizer_learning_rate")
    elif case == "lr_boolean": f.metadata["optimizer_learning_rate"] = True
    elif case == "lr_negative": f.metadata["optimizer_learning_rate"] = -1.
    elif case == "budget": f.metadata["stage_requested_decisions"] = {"full_episode": 0}
    elif case == "frozen":
        path = f.root / PROTECTED
        path.write_bytes(path.read_bytes() + b"\n# forbidden physical mutation\n")
        bind_target(f.root, f.new)
    elif case == "kernel": options["target_policy_version"] = f.metadata["policy_contract"]["version"]
    write_json(f.sidecar, f.metadata)
    with pytest.raises(ValueError):
        record(f, **options)


@pytest.mark.parametrize("case", ["reward_bytes", "observation_bytes", "cap", "rate", "task", "hold", "selected_hash"])
def test_no_extra_config_or_hidden_hold_gate_waiver(f, case):
    if case == "selected_hash":
        f.new["selected_configuration"]["action_schema.json"]["sha256"] = "0" * 64
    elif case.endswith("_bytes"):
        name = "reward_config.yaml" if case == "reward_bytes" else "observation_schema.json"
        path = f.root / TARGET_CONFIG / name
        path.write_bytes(path.read_bytes() + b"\n")
        bind_target(f.root, f.new)
    else:
        name = "execution_profile.yaml" if case in ("cap", "rate") else "stage_task_spec.yaml"
        path = f.root / TARGET_CONFIG / name
        value = yaml.safe_load(path.read_bytes())
        if case == "cap": value["residual"]["phase_caps_full12"]["P09"][0] += 1
        elif case == "rate": value["decision_hz"] = 30.
        elif case == "task": value["rear_leg_order"] = "RL_FIRST"
        else: value["stable_duration_s"] = 0.1
        path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
        bind_target(f.root, f.new)
    with pytest.raises(ValueError):
        record(f)


def arguments(f, *, warm_start):
    values = ["train", "--semantic-version", "v3", "--experiment-id", "fsm_reference_p09_stable_v2",
        "--run-dir", str(f.root / "runs/ppo_fsm_reference_p09_stable_v2/train/new"),
        "--expected-head", f.new["source_git_commit"], "--checkpoint", str(f.checkpoint),
        "--stage", "full_episode", "--from-phase", "P01", "--decisions", "128", "--seed", "1001"]
    return cli.parser().parse_args(values + (["--new-mdp-warm-start"] if warm_start else []))


def test_cli_isolates_target_namespace_and_requires_explicit_all_stage_warmstart(f, monkeypatch):
    monkeypatch.setattr(cli, "PROJECT_ROOT", f.root)
    assert cli.version_paths("v3", experiment_id="fsm_reference_p09_stable_v2") == (
        f.root / "runs/ppo_fsm_reference_p09_stable_v2",
        f.root / "outputs/ppo_fsm_reference_p09_stable_v2", f.root / TARGET_CONFIG)
    args = arguments(f, warm_start=True)
    cli.validate_request(args)
    assert args.checkpoint == f.checkpoint.resolve()
    with pytest.raises(ValueError, match="isolated checkpoint root"):
        cli.validate_request(arguments(f, warm_start=False))


def test_actual_cpu372_warmstart_preserves_weights_rng_lr_and_updates(prototype, tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("rsl_rl")
    from test_semantic_role_append_training import make
    from wlr50_clean.ppo import semantic_training as training
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    try:
        root = tmp_path / "actual_cpu"
        shutil.copytree(prototype.root, root)
        training.seed_training_rngs(1001)
        source, source_env = make(True)
        first = training.train_semantic(source, source_env, run_dir=root / "source_run",
            output_root=root / "source_output", stage="full_episode", decisions=128,
            contract=prototype.old, seed=1001)
        metadata = json.loads(Path(first["checkpoints"][-1]["manifest"]).read_text())
        for key in ("checkpoint_path", "checkpoint_sha256", "save_load_round_trip"):
            metadata.pop(key)
        metadata["new_mdp_origin_global_policy_decisions"] = 0
        source.alg.learning_rate = 1e-5
        for group in source.alg.optimizer.param_groups:
            group["lr"] = 1e-5
        assert source.alg.optimizer.state_dict()["state"]
        checkpoint = root / "learned372.pt"
        training.save_semantic_checkpoint(source, checkpoint, metadata)
        plan = migration.build_v3_warm_start_record(checkpoint, prototype.new, project_root=root)
        original = migration.build_v3_warm_start_record
        monkeypatch.setattr(migration, "build_v3_warm_start_record",
            lambda checkpoint, contract, **kwargs: original(checkpoint, contract, project_root=root, **kwargs))
        target, target_env = make(True)
        infos = training.load_semantic_checkpoint(target, checkpoint,
            contract=prototype.new, seed=1001, warm_start=plan)
        for role in ("actor", "critic"):
            before, after = getattr(source.alg, role).state_dict(), getattr(target.alg, role).state_dict()
            assert before.keys() == after.keys()
            assert all(torch.equal(value, after[key]) for key, value in before.items())
        assert target.alg.optimizer.state_dict()["state"] == {}
        assert target.alg.learning_rate == training.optimizer_learning_rate(target) == 1e-5
        assert target.alg.storage.step == 0 and target.alg.transition.actions is None
        assert training.capture_training_rng_state(seed=1001) == infos["training_rng_state"]
        assert target.alg.actor.obs_normalizer.state_dict() == target.alg.critic.obs_normalizer.state_dict() == {}
        assert (infos["global_policy_decisions"], infos["ppo_updates"], infos["optimizer_steps"]) == (128, 1, 20)
        result = training.train_semantic(target, target_env, run_dir=root / "target_run",
            output_root=root / "target_output", stage="full_episode", decisions=128,
            contract=prototype.new, seed=1001, resume_infos=infos)
        assert result["global_policy_decisions"] == 256
        assert result["ppo_updates_this_run"] == 1 and result["optimizer_steps_this_run"] == 20
        fresh, _ = make(True)
        restored = training.load_semantic_checkpoint(fresh, Path(result["checkpoints"][-1]["checkpoint"]),
            contract=prototype.new, seed=1001)
        assert (restored["global_policy_decisions"], restored["ppo_updates"], restored["optimizer_steps"]) == (256, 2, 40)
        assert training.state_hash(fresh.alg.optimizer.state_dict()) == training.state_hash(target.alg.optimizer.state_dict())
    finally:
        torch.set_num_threads(threads)
