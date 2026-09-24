"""Focused CPU checks for the reviewed same422 cooperative-prep boundary."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]

from wlr50_clean.ppo import semantic_cooperative_prep_migration as m
from wlr50_clean.ppo import semantic_training as training
from wlr50_clean.ppo.semantic_migration import digest, file_sha
from wlr50_clean.ppo.semantic_p02_progress_profile import (
    P02_PROGRESS_OBSERVATION_LAYOUT, P02_PROGRESS_POLICY,
)
from wlr50_clean.ppo.semantic_rear_cooperative_prep_profile import (
    COOPERATIVE_PREP_OBSERVATION_LAYOUT, COOPERATIVE_PREP_POLICY,
)
from wlr50_clean.ppo.semantic_rr_capture_migration import _shape_env


SOURCE = (ROOT / "outputs/ppo_rr_rl_timing_policy_learning_v1/branches"
          "/ancestor220544_recapture_v2/checkpoints/history/checkpoint_step_000223232.pt")


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    state, threads = torch.get_rng_state(), torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_rng_state(state)
    torch.set_num_threads(threads)


def _source_metadata():
    return json.loads(SOURCE.with_name(SOURCE.stem + "_manifest.json").read_text())


def _worktree_target(old):
    result = copy.deepcopy(old)
    result["source_git_commit"] = "f" * 40
    for relative in m.REVIEWED_CODE_PATHS | frozenset(m.REVIEWED_CONFIG_HASHES):
        result["files"][relative] = file_sha(ROOT / relative)
    result["runtime_content_sha256"] = digest(result["files"])
    for row in result["selected_configuration"].values():
        if row["path"] in m.REVIEWED_CONFIG_HASHES:
            row["sha256"] = result["files"][row["path"]]
    return result


def _runner(policy):
    layout = (P02_PROGRESS_OBSERVATION_LAYOUT
              if policy == P02_PROGRESS_POLICY else COOPERATIVE_PREP_OBSERVATION_LAYOUT)
    return training.construct_semantic_runner(
        _shape_env(422, "cpu"), seed=1001, device="cpu",
        policy_version=policy, observation_layout=layout,
        initialize_actor=False)[0]


def test_actual_cp223232_build_and_historical_lineage_are_exact():
    source = _source_metadata()
    target = _worktree_target(source["runtime_contract"])
    record = m.build_cooperative_prep_migration(
        SOURCE, target, reason="focused zero-credit CPU review")
    factor = record[m.FACTOR_KEY]
    assert record["source_checkpoint_sha256"] == m.SOURCE_SHA
    assert record["source_manifest_sha256"] == m.SOURCE_MANIFEST_SHA
    assert set(record["changed_file_hashes"]) == (
        m.REVIEWED_CODE_PATHS | frozenset(m.REVIEWED_CONFIG_HASHES))
    assert m._previous_contract(target, record) == source["runtime_contract"]
    assert factor["source_policy_contract"] == m._source_policy()
    assert factor["target_policy_contract"] == m._target_policy()
    assert factor["added_policy_decisions"] == factor["added_ppo_updates"] == 0

    migrated = copy.deepcopy(source)
    migrated.update(runtime_contract=target,
                    policy_contract=copy.deepcopy(factor["target_policy_contract"]),
                    runner_config=copy.deepcopy(factor["target_runner_config"]),
                    cooperative_prep_migration=copy.deepcopy(record))
    route = migrated["checkpoint_output_routing"]
    assert m.validate_cooperative_prep_lineage(
        migrated, target, Path(route["output_root"]),
        checkpoint_output_routing=route) == record

    # A later ordinary PPO save may advance counters/weights, but cannot rewrite
    # the boundary, its old P02 ancestry, route, or runner/policy contract.
    later = copy.deepcopy(migrated)
    for key, amount in zip(m.COUNTERS, (128, 1, 20), strict=True):
        later[key] += amount
    from wlr50_clean.ppo.semantic_rear_policy_timing_migration import rear_policy_timing_branch_counts
    later = rear_policy_timing_branch_counts(later)
    later["actor_parameter_sha256"] = "a" * 64
    m.validate_cooperative_prep_lineage(
        later, target, Path(route["output_root"]),
        checkpoint_output_routing=route)

    bad = copy.deepcopy(migrated)
    bad["runner_config"]["algorithm"]["gamma"] = .5
    with pytest.raises(ValueError, match="lineage|configuration"):
        m.validate_cooperative_prep_lineage(
            bad, target, Path(route["output_root"]),
            checkpoint_output_routing=route)
    bad = copy.deepcopy(migrated)
    bad["p02_progress_migration"]["reason"] += " tampered"
    with pytest.raises(ValueError):
        m.validate_cooperative_prep_lineage(
            bad, target, Path(route["output_root"]),
            checkpoint_output_routing=route)


def test_runtime_delta_rejects_unreviewed_contract_metadata():
    source = _source_metadata()
    target = _worktree_target(source["runtime_contract"])
    target["physics_hz"] = 119.0
    with pytest.raises(ValueError, match="runtime metadata"):
        m.build_cooperative_prep_migration(
            SOURCE, target, reason="must reject unrelated metadata")


def test_cpu_full_state_identity_fresh_storage_and_normal_save(tmp_path, monkeypatch):
    source = _runner(P02_PROGRESS_POLICY)
    for parameter in list(source.alg.actor.parameters()) + list(source.alg.critic.parameters()):
        parameter.grad = torch.ones_like(parameter)
    source.alg.optimizer.step()
    source.alg.optimizer.zero_grad()
    source.alg.learning_rate = 1e-5
    for group in source.alg.optimizer.param_groups:
        group["lr"] = 1e-5
    source.current_learning_iteration = m.REVISION_ORIGIN["ppo_updates"]

    base = _source_metadata()
    for key in ("checkpoint_path", "checkpoint_sha256", "save_load_round_trip"):
        base.pop(key, None)
    source_path, source_sidecar = training.save_semantic_checkpoint(
        source, tmp_path / "source422.pt", base)
    source_info = json.loads(source_sidecar.read_text())
    target = _runner(COOPERATIVE_PREP_POLICY)
    old_runner, target_runner = m._runner_configs(source_info)
    assert old_runner == source_info["runner_config"]
    current = copy.deepcopy(source_info["runtime_contract"])
    current.update(source_git_commit="e" * 40, runtime_content_sha256="d" * 64)
    factor = {
        "schema": m.SCHEMA,
        "source_policy_contract": copy.deepcopy(source_info["policy_contract"]),
        "target_policy_contract": m._target_policy(),
        "source_runner_config": old_runner,
        "target_runner_config": target_runner,
        "source_effective_learning_rate": 1e-5,
        "preserved_metadata_sha256": {
            key: digest(source_info[key]) for key in m._preserved(source_info)
        },
        "parameter_mapping": "identity_all_actor_critic_buffers_full_Adam_options_steps_and_rng",
    }
    record = {"schema": m.SCHEMA, "plan_path": str(tmp_path / "plan.json"),
              m.FACTOR_KEY: factor}
    monkeypatch.setattr(m, "validate_cooperative_prep_migration",
                        lambda *args, **kwargs: record)

    before = {
        "actor": training.parameter_hash(source.alg.actor),
        "critic": training.parameter_hash(source.alg.critic),
        "adam": training.state_hash(source.alg.optimizer.state_dict()),
    }
    infos = training.load_semantic_checkpoint(
        target, source_path, contract=current, seed=1001, migration=record)
    assert training.parameter_hash(target.alg.actor) == before["actor"]
    assert training.parameter_hash(target.alg.critic) == before["critic"]
    assert training.state_hash(target.alg.optimizer.state_dict()) == before["adam"]
    assert target.alg.storage.step == 0 and target.alg.transition.actions is None
    assert infos["sampling"] == "P01_full_task_only_initial_version"
    assert infos["phase_suffix_curriculum_implemented"] is False
    assert infos[m.MIGRATION] == record and infos["resume_migration"] == record

    saved, _ = training.save_semantic_checkpoint(
        target, tmp_path / "migrated422.pt", infos)
    fresh = _runner(COOPERATIVE_PREP_POLICY)
    loaded = training.load_semantic_checkpoint(
        fresh, saved, contract=current, seed=1001)
    assert training.parameter_hash(fresh.alg.actor) == before["actor"]
    assert training.parameter_hash(fresh.alg.critic) == before["critic"]
    assert training.state_hash(fresh.alg.optimizer.state_dict()) == before["adam"]
    assert loaded[m.MIGRATION] == record


def test_missing_source_is_fail_closed(tmp_path):
    with pytest.raises((FileNotFoundError, ValueError)):
        m.build_cooperative_prep_migration(
            tmp_path / "absent.pt", {}, reason="not a source")
