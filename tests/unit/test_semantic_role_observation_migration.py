"""Append48 record/topology tests, not tensor, optimizer or physics evidence.

Source config/runtime bytes have real temporary git history. The opaque .pt
fixture only exercises the sidecar/digest validator; root tests the real loader.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from wlr50_clean.ppo import semantic_migration as m
from wlr50_clean.ppo.semantic_policy_distribution import (
    HISTORY_POLICY, STATE_DEPENDENT_POLICY, policy_contract,
)
from wlr50_clean.ppo.semantic_training import semantic_runner_config
from wlr50_clean.ppo.semantic_transfer_roles import (
    ROLE_OBSERVATION_LAYOUT, ROLE_OBSERVATION_GROUP,
    ROLE_OBSERVATION_BASE_DIM, ROLE_OBSERVATION_DIM, ROLE_OBSERVATION_FIELDS,
)

OBS = "configs/ppo_semantic_v3/observation_schema.json"
ENCODER = "src/wlr50_clean/ppo/semantic_observation.py"
PROTECTED = "src/wlr50_clean/ppo/semantic_reward.py"
NAMES = ("stage_task_spec.yaml", "execution_profile.yaml", "reward_config.yaml",
         "observation_schema.json", "action_schema.json", "quality_score.yaml")
SAMPLING = "P01_full_task_only_initial_version"
GROUP = {"name": ROLE_OBSERVATION_GROUP, "size": 48, "scale": 1.0}


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True,
                          capture_output=True).stdout.decode().strip()


def commit(root, message):
    git(root, "-c", "core.autocrlf=false", "add", "configs", "src")
    git(root, "-c", "commit.gpgsign=false", "commit", "-m", message)
    return git(root, "rev-parse", "HEAD")


def bind(root, contract):
    contract["files"] = {p: m.file_sha(root / p) for p in contract["files"]}
    contract["runtime_content_sha256"] = m.digest(contract["files"])
    contract["selected_configuration"] = {
        name: {"path": f"configs/ppo_semantic_v3/{name}",
               "sha256": contract["files"][f"configs/ppo_semantic_v3/{name}"]}
        for name in NAMES}


def source_metadata(checkpoint, contract, *, layout=None, policy=HISTORY_POLICY):
    return {
        "schema": "wlr50_clean.semantic_checkpoint.v1", "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": m.file_sha(checkpoint), "save_load_round_trip": True,
        "semantic_version": "v3", "runtime_contract": copy.deepcopy(contract), "seed": 1001,
        "runner_config": semantic_runner_config(seed=1001, device="cpu", semantic_version="v3",
            policy_version=policy, observation_layout=layout),
        "policy_contract": policy_contract(policy, observation_layout=layout),
        "global_policy_decisions": 128, "ppo_updates": 1, "optimizer_steps": 20,
        "new_mdp_origin_global_policy_decisions": 0,
        "stage_requested_decisions": {"smoke": 0, "full_episode": 128, "phase_suffix": 0},
        "sampling": SAMPLING,
        "execution_topology": m.continuation_topology(SAMPLING, None, observation_layout=layout),
    }


@pytest.fixture(scope="module")
def prototype(tmp_path_factory):
    root = tmp_path_factory.mktemp("role_append_git")
    paths = sorted(m.ROLE_APPEND_RUNTIME_FILES | {PROTECTED} |
                   {f"configs/ppo_semantic_v3/{name}" for name in NAMES})
    for relative in paths:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((m.PROJECT_ROOT / relative).read_bytes())
    before = json.loads((root / OBS).read_text())
    before.pop("transfer_role_features_version", None)
    before["feature_groups"] = [g for g in before["feature_groups"] if g["name"] != ROLE_OBSERVATION_GROUP]
    assert sum(g["size"] for g in before["feature_groups"]) == 324
    write_json(root / OBS, before)
    git(root, "init")
    git(root, "config", "user.name", "Append48 record test")
    git(root, "config", "user.email", "append48@example.invalid")
    old = {
        "source_git_commit": commit(root, "real original324 bytes"),
        "files": dict.fromkeys(paths), "semantic_version": "v3", "experiment_id": "transfer_roles_v1",
        "frozen_A_files": {"frozen": "a" * 64}, "physics_hz": 120., "decision_hz": 15.,
        "task_timeout_s": 200., "timeout_bootstrap": False, "rsl_rl_version": "5.0.1",
        "local_runtime_versions": {"fixture": "metadata_only"},
    }
    bind(root, old)
    after = copy.deepcopy(before)
    after["revision"] = "explicit_role_append48"
    after["transfer_role_features_version"] = ROLE_OBSERVATION_LAYOUT
    after["feature_groups"].append(copy.deepcopy(GROUP))
    write_json(root / OBS, after)
    # Make source recovery and target runtime verification actually exercise
    # a changed allowed .py file, rather than only the observation JSON.
    with (root / ENCODER).open("a", encoding="utf-8") as stream:
        stream.write("\n# metadata-only target test revision\n")
    new = copy.deepcopy(old)
    new["source_git_commit"] = commit(root, "real appended48 bytes")
    bind(root, new)
    checkpoint = root / "record_only.pt"
    checkpoint.write_bytes(b"opaque record-validation fixture; not torch tensors")
    metadata = source_metadata(checkpoint, old)
    write_json(checkpoint.with_name("record_only_manifest.json"), metadata)
    return SimpleNamespace(root=root, old=old, new=new, before=before, after=after)


@pytest.fixture
def f(prototype, tmp_path):
    root = tmp_path / "isolated"
    shutil.copytree(prototype.root, root)
    checkpoint = root / "record_only.pt"
    old, new = copy.deepcopy(prototype.old), copy.deepcopy(prototype.new)
    metadata = source_metadata(checkpoint, old)
    sidecar = checkpoint.with_name("record_only_manifest.json")
    write_json(sidecar, metadata)
    return SimpleNamespace(root=root, old=old, new=new, before=copy.deepcopy(prototype.before),
        after=copy.deepcopy(prototype.after), checkpoint=checkpoint, sidecar=sidecar, metadata=metadata)


def record(f, **kwargs):
    return m.build_v3_warm_start_record(f.checkpoint, f.new, project_root=f.root, **kwargs)


def save_target_schema(f):
    write_json(f.root / OBS, f.after)
    bind(f.root, f.new)


def test_exact_append_record_preserves_ledger_and_separates_zero_padding_from_scale(f):
    old_bytes, sidecar_bytes = f.checkpoint.read_bytes(), f.sidecar.read_bytes()
    actual = record(f)
    a = actual["observation_append_transition"]
    assert a["schema"] == "wlr50_clean.transfer_roles_observation_append_transition.v1"
    assert (a["source_observation_dimension"], a["target_observation_dimension"]) == (324, 372)
    assert a["source_observation_layout"] is None and a["target_observation_layout"] == ROLE_OBSERVATION_LAYOUT
    assert a["first_layer_parameter"] == "mlp.0.weight" and a["appended_columns"] == [324, 372]
    assert a["first_layer_mapping"] == "append_zero_actor_and_critic_first_layer"
    assert a["fields_per_leg"] == list(ROLE_OBSERVATION_FIELDS)
    assert a["source_policy_contract"] == policy_contract(HISTORY_POLICY)
    assert a["target_policy_contract"] == policy_contract(HISTORY_POLICY, observation_layout=ROLE_OBSERVATION_LAYOUT)
    assert a["normalizers"] == "identity_RSL_state_preserved"
    assert actual["optimizer"] == {"kind": "Adam", "state": "reset_all_moments", "initial_learning_rate": 3e-5,
        "reason": "explicit role-observation boundary; verify source Adam then reset moments"}
    assert actual["source_stage_requested_decisions"] == actual["target_stage_requested_decisions"] == f.metadata["stage_requested_decisions"]
    assert actual["source_global_policy_decisions"] == 128 and actual["new_mdp_origin_global_policy_decisions"] == 0
    assert actual["old_rollout_buffer_inherited"] is False and actual["physical_state_inherited"] is False
    assert actual["network"]["observation_dimension"] == 372
    assert "observation_scale_transition" not in actual and "policy_kernel_transition" not in actual
    assert actual["return_horizon_transition"]["reward_discount_changed"] is False
    assert actual["return_horizon_transition"]["gae_estimator_changed"] is False
    assert record(f) == actual  # Pure recipe; no hidden mutation or second padding.
    assert f.checkpoint.read_bytes() == old_bytes and f.sidecar.read_bytes() == sidecar_bytes
    assert m._version_bytes(f.root, f.old, OBS) != (f.root / OBS).read_bytes()
    assert m.v3_warm_start_checkpoint_name(actual).endswith(".pt")
    assert f.old["selected_configuration"]["observation_schema.json"]["sha256"] != f.new["selected_configuration"]["observation_schema.json"]["sha256"]
    assert {k: v for k, v in f.old["selected_configuration"].items() if k != "observation_schema.json"} == {
        k: v for k, v in f.new["selected_configuration"].items() if k != "observation_schema.json"}


@pytest.mark.parametrize("side", ["source", "target"])
@pytest.mark.parametrize("case", ["missing", "extra_selection", "path", "hash", "extra_record_key", "unrelated_hash"])
def test_selected_configuration_is_not_a_blanket_metadata_exemption(f, side, case):
    contract = f.metadata["runtime_contract"] if side == "source" else f.new
    selected = contract["selected_configuration"]
    if case == "missing": contract.pop("selected_configuration")
    elif case == "extra_selection": selected["unknown.yaml"] = {"path": "unknown.yaml", "sha256": "a"*64}
    elif case == "path": selected["observation_schema.json"]["path"] = "configs/ppo_semantic_v2/observation_schema.json"
    elif case == "hash": selected["observation_schema.json"]["sha256"] = "0"*64
    elif case == "extra_record_key": selected["observation_schema.json"]["allowed"] = True
    else: selected["execution_profile.yaml"]["sha256"] = "0"*64
    write_json(f.sidecar, f.metadata)
    with pytest.raises(ValueError, match="selected_configuration"): record(f)


@pytest.mark.parametrize("case", ["missing_marker", "unknown_marker", "null_marker", "size", "size_bool", "scale",
    "scale_bool", "extra_group_key", "extra_group", "group_order", "old_scale", "raw_slice", "clip", "normalization",
    "unknown_preprocessing"])
def test_append_rejects_any_prefix_or_appended_layout_drift(f, case):
    groups = f.after["feature_groups"]
    if case == "missing_marker": f.after.pop("transfer_role_features_version")
    elif case == "unknown_marker": f.after["transfer_role_features_version"] = "unreviewed"
    elif case == "null_marker": f.after["transfer_role_features_version"] = None
    elif case == "size": groups[-1]["size"] = 47
    elif case == "size_bool": groups[-1]["size"] = True
    elif case == "scale": groups[-1]["scale"] = 2.
    elif case == "scale_bool": groups[-1]["scale"] = True
    elif case == "extra_group_key": groups[-1]["fields"] = list(ROLE_OBSERVATION_FIELDS)
    elif case == "extra_group": groups.append(copy.deepcopy(GROUP))
    elif case == "group_order": groups[-1], groups[-2] = groups[-2], groups[-1]
    elif case == "old_scale": next(g for g in groups if g["name"] == "previous_residual_full12")["scale"][3] = 4
    elif case == "raw_slice": next(g for g in groups if g["name"] == "previous_raw_full12")["scale"] = 2.
    elif case == "clip": f.after["clip"] = 21.
    elif case == "normalization": f.after["normalization"] = "adaptive"
    else: f.after["new_preprocessing"] = "not declared"
    save_target_schema(f)
    with pytest.raises(ValueError): record(f)


@pytest.mark.parametrize("name", [x for x in NAMES if x != "observation_schema.json"])
def test_all_other_configuration_bytes_remain_exact(f, name):
    path = f.root / f"configs/ppo_semantic_v3/{name}"
    with path.open("a", encoding="utf-8") as stream: stream.write("\n# forbidden simultaneous config change\n")
    bind(f.root, f.new)
    with pytest.raises(ValueError, match="cannot change task/action/reward/quality"): record(f)


@pytest.mark.parametrize("case", ["source_experiment", "target_experiment", "kernel", "actor", "normalizer", "source372",
    "physical_metadata", "protected_runtime", "deleted_runtime", "target_bytes", "source_bytes", "checkpoint_bytes"])
def test_append_source_and_runtime_boundaries_fail_closed(f, case):
    kwargs = {}
    if case == "source_experiment": f.metadata["runtime_contract"].pop("experiment_id")
    elif case == "target_experiment": f.new.pop("experiment_id")
    elif case == "kernel": kwargs["target_policy_version"] = HISTORY_POLICY
    elif case == "actor":
        f.metadata = source_metadata(f.checkpoint, f.old, policy=STATE_DEPENDENT_POLICY)
    elif case == "normalizer": f.metadata["runner_config"]["actor"]["obs_normalization"] = True
    elif case == "source372": f.metadata = source_metadata(f.checkpoint, f.old, layout=ROLE_OBSERVATION_LAYOUT)
    elif case == "physical_metadata": f.new["decision_hz"] = 30.
    elif case == "protected_runtime":
        with (f.root / PROTECTED).open("a", encoding="utf-8") as stream: stream.write("\n# forbidden\n")
        bind(f.root, f.new)
    elif case == "deleted_runtime":
        f.new["files"].pop(ENCODER); f.new["runtime_content_sha256"] = m.digest(f.new["files"])
    elif case == "target_bytes":
        with (f.root / ENCODER).open("a", encoding="utf-8") as stream: stream.write("\n# unbound\n")
    elif case == "source_bytes":
        f.metadata["runtime_contract"]["files"][ENCODER] = "0" * 64
        f.metadata["runtime_contract"]["runtime_content_sha256"] = m.digest(f.metadata["runtime_contract"]["files"])
    else: f.checkpoint.write_bytes(b"corrupt")
    write_json(f.sidecar, f.metadata)
    with pytest.raises(ValueError): record(f, **kwargs)


def test_legacy_scale_transition_unchanged_and_not_combined_with_append(f):
    old = copy.deepcopy(f.before)
    for g in old["feature_groups"]:
        if g["name"] in ("previous_residual_full12", "previous_previous_residual_full12"): g["scale"][3] = 4
    result = m._transfer_observation_scale_transition(old, f.before, {"source_sha256": "a"*64, "target_sha256": "b"*64})
    assert result["columns"] == [210, 222] and result["factors"] == [1.5, 1.5] and result["changed"] is True
    unchanged = m._transfer_observation_scale_transition(f.before, f.before, {"source_sha256": "a"*64, "target_sha256": "a"*64})
    assert unchanged["changed"] is False and unchanged["factors"] == [1., 1.]


def test_topology_default_stays_324_and_explicit_layout_is_bound_to_metadata(f):
    assert m.topology(1) == m.topology(1, observation_layout=None)
    assert "observation_layout" not in m.topology(1)
    assert m.topology(8)["observation_dimension"] == 324
    metadata = source_metadata(f.checkpoint, f.old, layout=ROLE_OBSERVATION_LAYOUT)
    assert metadata["execution_topology"]["observation_dimension"] == 372
    assert m.source_num_envs(metadata) == 1
    metadata["execution_topology"]["observation_dimension"] = 324
    with pytest.raises(ValueError, match="topology"): m.source_num_envs(metadata)


@pytest.mark.parametrize("case", ["missing_topology", "missing_contract", "incomplete_contract", "wrong_marker", "N8", "bool"])
def test_role_topology_cannot_silently_revert_to324_or_expand_N8(f, case):
    metadata = source_metadata(f.checkpoint, f.old, layout=ROLE_OBSERVATION_LAYOUT)
    if case == "missing_topology": metadata.pop("execution_topology")
    elif case == "missing_contract": metadata.pop("policy_contract")
    elif case == "incomplete_contract": metadata["policy_contract"].pop("role_observation_fields")
    elif case == "wrong_marker": metadata["policy_contract"]["observation_layout"] = "unknown"
    elif case == "N8":
        with pytest.raises(ValueError): m.topology(8, observation_layout=ROLE_OBSERVATION_LAYOUT)
        return
    else:
        with pytest.raises(ValueError): m.topology(1, observation_layout=True)
        return
    with pytest.raises(ValueError): m.source_num_envs(metadata)
