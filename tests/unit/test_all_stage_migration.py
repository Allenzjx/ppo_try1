"""All-stage same372 record/CLI guards, not tensor restore or physics evidence.

Uses genuine temporary Git/config bytes and an explicitly opaque checkpoint.
Counter/RNG/Adam assertions validate the recipe only; no Torch load is performed.
"""
from __future__ import annotations

import copy
import json
import shutil
from types import SimpleNamespace

import pytest
import yaml

from wlr50_clean.ppo import semantic_cli as cli
from wlr50_clean.ppo import semantic_migration as m
from wlr50_clean.ppo.semantic_transfer_roles import ROLE_OBSERVATION_LAYOUT
from test_semantic_same372_authority_migration import prototype as same372_prototype
from test_semantic_role_observation_migration import (
    NAMES, PROTECTED, bind, commit, source_metadata, write_json,
)

SOURCE_CONFIG = "configs/ppo_semantic_v3"
TARGET_CONFIG = "configs/ppo_all_stage_acceptance_v1"
NEW_SENSOR = "src/wlr50_clean/ppo/semantic_physical_sensing.py"
FROZEN_PATHS = (
    PROTECTED,
    "src/wlr50_clean/infrastructure/robot_adapter.py",
    "src/wlr50_clean/infrastructure/scene_factory.py",
    "src/wlr50_clean/sensing/geometry.py",
)


def bind_target(root, contract):
    contract["files"] = {p: m.file_sha(root / p) for p in contract["files"]}
    contract["runtime_content_sha256"] = m.digest(contract["files"])
    contract["selected_configuration"] = {
        name: {"path": f"{TARGET_CONFIG}/{name}",
               "sha256": contract["files"][f"{TARGET_CONFIG}/{name}"]}
        for name in NAMES}


@pytest.fixture(scope="module")
def prototype(same372_prototype, tmp_path_factory):
    root = tmp_path_factory.mktemp("all_stage_record_git") / "case"
    shutil.copytree(same372_prototype.root, root)
    old = copy.deepcopy(same372_prototype.new)
    for relative in FROZEN_PATHS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((m.PROJECT_ROOT / relative).read_bytes())
        old["files"][relative] = None
    old["source_git_commit"] = commit(root, "fixed source HISTORY372 physical bytes")
    bind(root, old)
    new = copy.deepcopy(old)
    new["experiment_id"] = "all_stage_acceptance_v1"
    for name in NAMES:
        target = root / TARGET_CONFIG / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((root / SOURCE_CONFIG / name).read_bytes())
        new["files"][f"{TARGET_CONFIG}/{name}"] = None
    target_spec = root / TARGET_CONFIG / "stage_task_spec.yaml"
    spec = yaml.safe_load(target_spec.read_bytes())
    spec["physical_acceptance_version"] = "all_stage_v1"
    spec["final"]["post_completion_observation_s"] = 1.
    target_spec.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    # Real source/target Git-bound implementation delta, not a mocked validator.
    path = root / m.SUPERVISOR
    path.write_bytes(path.read_bytes() + b"\n# explicit acceptance fixture revision\n")
    sensor = root / NEW_SENSOR
    sensor.write_bytes(b'"""Opaque source-inventory fixture, not imported."""\n')
    new["files"][NEW_SENSOR] = None
    new["source_git_commit"] = commit(root, "isolated all-stage target bytes")
    bind_target(root, new)
    return SimpleNamespace(root=root, old=old, new=new)


@pytest.fixture
def f(prototype, tmp_path):
    root = tmp_path / "case"
    shutil.copytree(prototype.root, root)
    checkpoint = root / "outputs/ppo_transfer_roles_v1/checkpoints/history/source.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"opaque metadata guard fixture; deliberately not a torch checkpoint")
    metadata = source_metadata(checkpoint, prototype.old, layout=ROLE_OBSERVATION_LAYOUT)
    metadata.update(global_policy_decisions=138496, ppo_updates=1047, optimizer_steps=20940,
        new_mdp_origin_global_policy_decisions=10112,
        stage_requested_decisions={"smoke": 0, "full_episode": 60000, "phase_suffix": 68384})
    sidecar = checkpoint.with_name("source_manifest.json")
    write_json(sidecar, metadata)
    return SimpleNamespace(root=root, old=copy.deepcopy(prototype.old), new=copy.deepcopy(prototype.new),
                           checkpoint=checkpoint, sidecar=sidecar, metadata=metadata)


def record(f, **kwargs):
    return m.build_v3_warm_start_record(f.checkpoint, f.new, project_root=f.root, **kwargs)


def test_same372_new_acceptance_recipe_preserves_learning_ledger_not_physical_state(f):
    before = (f.checkpoint.read_bytes(), f.sidecar.read_bytes())
    result = record(f)
    transition = result["observation_same_layout_transition"]
    assert transition["schema"] == m.ALL_STAGE_SCHEMA
    assert transition["parameter_mapping"] == "identity_all_parameters_and_buffers"
    assert transition["source_policy_contract"] == transition["target_policy_contract"] == f.metadata["policy_contract"]
    assert transition["source_observation_dimension"] == transition["target_observation_dimension"] == 372
    assert transition["source_observation_layout"] == transition["target_observation_layout"] == ROLE_OBSERVATION_LAYOUT
    assert transition["source_schema_sha256"] == transition["target_schema_sha256"]
    assert transition["kernel_changed"] is False and transition["physical_actuators_changed"] is False
    assert transition["measurement_and_task_acceptance_changed"] is True
    assert transition["reward_changed"] is True  # Changed task/Phi semantics despite identical reward YAML.
    assert transition["training_rng_preserved"] is True
    assert transition["lifetime_counters_and_spent_budgets_preserved"] is True
    assert result["rng"] == "restore_verified_training_rng_then_sample_fresh_new_MDP_rollouts"
    assert result["source_global_policy_decisions"] == 138496
    assert result["new_mdp_origin_global_policy_decisions"] == 10112
    assert result["source_stage_requested_decisions"] == result["target_stage_requested_decisions"] == f.metadata["stage_requested_decisions"]
    assert result["optimizer"]["state"] == "reset_all_moments"
    assert result["optimizer"]["initial_learning_rate"] == 3e-5
    assert result["old_rollout_buffer_inherited"] is False and result["physical_state_inherited"] is False
    assert result["exact_mdp_resume"] is False
    assert result["network"]["observation_dimension"] == 372
    assert not {"observation_append_transition", "observation_scale_transition", "policy_kernel_transition"} & result.keys()
    assert not result["return_horizon_transition"]["reward_discount_changed"]
    assert not result["return_horizon_transition"]["gae_estimator_changed"]
    assert result["experiment_transition"] == {
        "source_experiment_id": "transfer_roles_v1", "target_experiment_id": "all_stage_acceptance_v1",
        "source_artifact_namespace": "ppo_transfer_roles_v1", "target_artifact_namespace": "ppo_all_stage_acceptance_v1",
        "target_configuration_namespace": "ppo_all_stage_acceptance_v1", "v3_budgets_reset": False}
    assert NEW_SENSOR in result["runtime_changed_files"]
    assert record(f) == result and before == (f.checkpoint.read_bytes(), f.sidecar.read_bytes())
    receipt = m.checkpoint_metadata(f.checkpoint)
    assert (receipt["ppo_updates"], receipt["optimizer_steps"]) == (1047, 20940)
    # These are preserved sidecar numbers, not proof a real optimizer was loaded.
    assert m.v3_warm_start_checkpoint_name(result).startswith("checkpoint_initial_v3_from_000138496_")


@pytest.mark.parametrize("relative", FROZEN_PATHS)
def test_frozen_sensing_scene_adapter_reward_bytes_cannot_change(f, relative):
    path = f.root / relative
    path.write_bytes(path.read_bytes() + b"\n# forbidden physical or unrelated mutation\n")
    bind_target(f.root, f.new)
    with pytest.raises(ValueError, match="non-reviewed runtime"):
        record(f)


@pytest.mark.parametrize("side", ["source", "target"])
@pytest.mark.parametrize("name", NAMES)
def test_all_six_selected_configuration_hash_bindings_are_checked(f, side, name):
    if side == "source":
        f.metadata["runtime_contract"]["selected_configuration"][name]["sha256"] = "0"*64
        write_json(f.sidecar, f.metadata)
    else:
        f.new["selected_configuration"][name]["sha256"] = "0"*64
    with pytest.raises(ValueError, match="binding"):
        record(f)


@pytest.mark.parametrize("case", ["missing", "extra", "wrong_path"])
def test_selected_configuration_set_and_paths_are_not_optional(f, case):
    selected = f.new["selected_configuration"]
    if case == "missing": selected.pop("quality_score.yaml")
    elif case == "extra": selected["not_a_seventh_config"] = {}
    else:
        selected["quality_score.yaml"]["path"] = f"{SOURCE_CONFIG}/quality_score.yaml"
        # Keep the corrupted inventory internally consistent: refusal must not
        # depend on an orphan newly-added target file accidentally remaining.
        del f.new["files"][f"{TARGET_CONFIG}/quality_score.yaml"]
        f.new["runtime_content_sha256"] = m.digest(f.new["files"])
    with pytest.raises(ValueError, match="binding|six|isolated namespace"):
        record(f)


@pytest.mark.parametrize("experiment", [None, "all_stage_acceptance_v1"])
def test_wrong_source_experiment_is_not_relabelled_transfer_roles(f, experiment):
    f.metadata["runtime_contract"]["experiment_id"] = experiment
    write_json(f.sidecar, f.metadata)
    with pytest.raises(ValueError, match="existing N1 HISTORY372"):
        record(f)


@pytest.mark.parametrize("case", ["schema_bytes", "schema_scale", "reward", "action", "quality", "physical_metadata", "delete", "inventory_tamper", "kernel"])
def test_same_layout_boundary_is_not_an_unrestricted_new_mdp_waiver(f, case):
    name = {"reward": "reward_config.yaml", "action": "action_schema.json", "quality": "quality_score.yaml"}.get(case)
    if name:
        path = f.root / TARGET_CONFIG / name
        path.write_bytes(path.read_bytes() + b"\n")
        bind_target(f.root, f.new)
    elif case.startswith("schema"):
        path = f.root / TARGET_CONFIG / "observation_schema.json"
        if case == "schema_bytes": path.write_bytes(path.read_bytes()+b"\n")
        else:
            data = json.loads(path.read_bytes()); data["feature_groups"][0]["scale"] = 2.
            write_json(path, data)
        bind_target(f.root, f.new)
    elif case == "physical_metadata": f.new["physics_hz"] = 60.
    elif case == "delete":
        del f.new["files"][PROTECTED]
        f.new["runtime_content_sha256"] = m.digest(f.new["files"])
    elif case == "inventory_tamper":
        (f.root / NEW_SENSOR).write_bytes(b"unbound post-inventory content")
    options = {"target_policy_version": f.metadata["policy_contract"]["version"]} if case == "kernel" else {}
    with pytest.raises(ValueError):
        record(f, **options)


@pytest.mark.parametrize("field,value", [("new_mdp_origin_global_policy_decisions", None),
                                        ("stage_requested_decisions", {"full_episode": 0})])
def test_missing_original_budget_cannot_silently_restart_all_stage_budget(f, field, value):
    f.metadata[field] = value
    write_json(f.sidecar, f.metadata)
    with pytest.raises(ValueError, match="intact original budget"):
        record(f)


def arguments(f, checkpoint, *, warm_start):
    values = ["train", "--semantic-version", "v3", "--experiment-id", "all_stage_acceptance_v1",
        "--run-dir", str(f.root / "runs/ppo_all_stage_acceptance_v1/train/new"),
        "--expected-head", f.new["source_git_commit"], "--checkpoint", str(checkpoint),
        "--stage", "full_episode", "--from-phase", "P01", "--decisions", "128", "--seed", "1001"]
    return cli.parser().parse_args(values + (["--new-mdp-warm-start"] if warm_start else []))


def test_cli_routes_configs_and_initial_checkpoint_source_without_changing_physical_version(f, monkeypatch):
    monkeypatch.setattr(cli, "PROJECT_ROOT", f.root)
    runs, outputs, config = cli.version_paths("v3", experiment_id="all_stage_acceptance_v1")
    assert (runs, outputs, config) == (f.root/"runs/ppo_all_stage_acceptance_v1",
        f.root/"outputs/ppo_all_stage_acceptance_v1", f.root/TARGET_CONFIG)
    assert cli.version_paths("v3", experiment_id="transfer_roles_v1")[2] == f.root/SOURCE_CONFIG
    args = arguments(f, f.checkpoint, warm_start=True)
    cli.validate_request(args)
    assert args.checkpoint == f.checkpoint.resolve() and args.num_envs == 1 and args.from_phase == "P01"
    with pytest.raises(ValueError, match="isolated checkpoint root"):
        cli.validate_request(arguments(f, f.checkpoint, warm_start=False))


def test_cli_ordinary_resume_accepts_only_new_output_root(f, monkeypatch):
    monkeypatch.setattr(cli, "PROJECT_ROOT", f.root)
    checkpoint = f.root/"outputs/ppo_all_stage_acceptance_v1/checkpoints/history/target.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"path validation only")
    write_json(checkpoint.with_name("target_manifest.json"), f.metadata)
    args = arguments(f, checkpoint, warm_start=False)
    cli.validate_request(args)
    assert args.checkpoint == checkpoint.resolve() and not args.new_mdp_warm_start


def test_cli_disallows_old_generic_v3_source_for_new_all_stage_boundary(f, monkeypatch):
    monkeypatch.setattr(cli, "PROJECT_ROOT", f.root)
    checkpoint = f.root/"outputs/ppo_semantic_v3/checkpoints/history/generic.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"path validation only")
    write_json(checkpoint.with_name("generic_manifest.json"), f.metadata)
    with pytest.raises(ValueError, match="isolated checkpoint root"):
        cli.validate_request(arguments(f, checkpoint, warm_start=True))
