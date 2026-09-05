from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from wlr50_clean.ppo import semantic_migration as migration
from wlr50_clean.ppo.semantic_training import write_json
from test_semantic_migration import checkpoint, contract


TEST_SOURCE = "tests/unit/test_semantic_supervisor.py"
NODE = TEST_SOURCE + "::test_early_top_then_center_cross[missing_point]"


def fixture(tmp_path, monkeypatch):
    before = ("# unchanged imports and task configuration\n"
              "class TaskEvaluator:\n    immediately_previous_air_only = True\n\n"
              "class TaskStageSupervisor:\n    unchanged_stage = True\n\n"
              "class NominalMotionProvider:\n    unchanged_prior = True\n\n"
              "class SemanticControllerAdapter:\n    unchanged_adapter = True\n")
    after = before.replace("immediately_previous_air_only", "continuous_qualified_air_top_chain")
    old, new = contract(1), contract(1)
    new["source_git_commit"] = "2" * 40
    source_file = tmp_path / migration.SUPERVISOR
    source_file.parent.mkdir(parents=True)
    source_file.write_text(after, encoding="utf-8")
    old["files"][migration.SUPERVISOR] = migration.hashlib.sha256(before.encode()).hexdigest()
    new["files"][migration.SUPERVISOR] = migration.file_sha(source_file)
    for value in (old, new):
        value["files"][migration.STAGE_SPEC] = "f" * 64
        value["runtime_content_sha256"] = migration.digest(value["files"])
    test_file = tmp_path / TEST_SOURCE
    test_file.parent.mkdir(parents=True)
    test_file.write_text("def test_early_top_then_center_cross():\n    pass\n", encoding="utf-8")
    monkeypatch.setattr(migration.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout=before.encode()))
    review = {"reason": "Reviewed legitimate AIR to early TOP to later center crossing counterexample",
              "counterexample_tests": [NODE]}
    return checkpoint(tmp_path, old), old, new, review


def build(source, new, review, root, **kwargs):
    return migration.build_migration_plan(source, new, allowed_changed_files=[migration.SUPERVISOR],
        reason="Explicit reviewed evaluator-only boundary from retained weights", evaluator_review=review,
        project_root=root, **kwargs)


def test_reviewer_bound_evaluator_only_factor_needs_no_old_prior_or_new_hash_pin(tmp_path, monkeypatch):
    source, old, new, review = fixture(tmp_path, monkeypatch)
    original = source.read_bytes()
    plan = build(source, new, review, tmp_path)
    factor = plan["reviewed_evaluator_factor"]
    assert plan["geometric_factor"] is None
    assert "prior_factor" not in plan and "qualification_factor" not in plan
    assert factor["before_class_sha256"] != factor["after_class_sha256"]
    assert factor["test_sources"][TEST_SOURCE]["sha256"] == migration.file_sha(tmp_path / TEST_SOURCE)
    assert factor["test_nodes_are_review_references_not_pass_certification"] is True
    assert factor["target_contract_sha256"] == migration.digest(new)
    assert factor["source_git_commit"] == old["source_git_commit"]
    assert plan["preserve_actor_critic_optimizer_normalizer_rng_and_budget"] is True
    assert plan["discard_old_rollout_storage"] is True
    assert plan["physics_resume"] == "fresh_legal_P01_reset"
    path = tmp_path / "reviewed.json"
    write_json(path, plan)
    assert migration.validate_migration_plan(source, new, path, project_root=tmp_path)["reviewed_evaluator_factor"] == factor
    assert source.read_bytes() == original


@pytest.mark.parametrize("fault", ["prefix", "stage", "nominal", "adapter", "no_class_change", "duplicate_boundary"])
def test_every_byte_outside_evaluator_and_unambiguous_changed_class_is_required(tmp_path, monkeypatch, fault):
    source, old, new, review = fixture(tmp_path, monkeypatch)
    path = tmp_path / migration.SUPERVISOR
    text = path.read_text()
    if fault == "prefix":
        text = text.replace("unchanged imports", "changed imports")
    elif fault == "no_class_change":
        text = text.replace("continuous_qualified_air_top_chain", "immediately_previous_air_only")
    elif fault == "duplicate_boundary":
        text += "\nclass TaskEvaluator:\n    extra = True\n"
    else:
        text = text.replace("unchanged_" + {"stage": "stage", "nominal": "prior", "adapter": "adapter"}[fault], "changed")
    path.write_text(text)
    new["files"][migration.SUPERVISOR] = migration.file_sha(path)
    new["runtime_content_sha256"] = migration.digest(new["files"])
    with pytest.raises(ValueError):
        build(source, new, review, tmp_path)


@pytest.mark.parametrize("fault", ["missing", "empty_reason", "no_tests", "duplicate_test", "foreign_test", "extra_key"])
def test_explicit_review_is_not_inferred_and_stays_exact(tmp_path, monkeypatch, fault):
    source, old, new, review = fixture(tmp_path, monkeypatch)
    if fault == "missing":
        review = None
    elif fault == "empty_reason":
        review["reason"] = " "
    elif fault == "no_tests":
        review["counterexample_tests"] = []
    elif fault == "duplicate_test":
        review["counterexample_tests"] *= 2
    elif fault == "foreign_test":
        review["counterexample_tests"] = [migration.SUPERVISOR + "::test_escape"]
    else:
        review["automatic_test_certification"] = True
    with pytest.raises(ValueError):
        build(source, new, review, tmp_path)


@pytest.mark.parametrize("mixed", ["config", "prior", "historical_qualification"])
def test_reviewed_boundary_cannot_combine_mdp_factors(tmp_path, monkeypatch, mixed):
    source, old, new, review = fixture(tmp_path, monkeypatch)
    kwargs = {}
    changed = [migration.SUPERVISOR]
    if mixed == "config":
        new["files"][migration.STAGE_SPEC] = "0" * 64
        new["runtime_content_sha256"] = migration.digest(new["files"])
        changed.append(migration.STAGE_SPEC)
    elif mixed == "prior":
        kwargs["prior_evidence"] = {}
    else:
        kwargs["qualification_evidence"] = tmp_path / "old.json"
    with pytest.raises(ValueError, match="cannot mix"):
        migration.build_migration_plan(source, new, allowed_changed_files=changed, reason="reviewed",
            evaluator_review=review, project_root=tmp_path, **kwargs)


@pytest.mark.parametrize("fault", ["class_sha", "target_head", "checkpoint_sha", "test_source"])
def test_plan_hash_binding_rejects_changed_boundary_or_counterexample_source(tmp_path, monkeypatch, fault):
    source, old, new, review = fixture(tmp_path, monkeypatch)
    plan = build(source, new, review, tmp_path)
    if fault == "class_sha":
        plan["reviewed_evaluator_factor"]["after_class_sha256"] = "0" * 64
    elif fault == "target_head":
        plan["target_git_commit"] = "a" * 40
    elif fault == "checkpoint_sha":
        plan["source_checkpoint_sha256"] = "b" * 64
    else:
        (tmp_path / TEST_SOURCE).write_text("# changed reviewed counterexample\n")
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="not exactly bound"):
        migration.validate_migration_plan(source, new, path, project_root=tmp_path)


def test_explicit_class_review_does_not_authorize_frozen_physics_or_reward(tmp_path, monkeypatch):
    source, old, new, review = fixture(tmp_path, monkeypatch)
    bad = copy.deepcopy(new)
    bad["frozen_A_files"]["A"] = "0" * 64
    with pytest.raises(ValueError, match="frozen physics"):
        build(source, bad, review, tmp_path)
    new["files"]["src/wlr50_clean/ppo/semantic_reward.py"] = "0" * 64
    new["runtime_content_sha256"] = migration.digest(new["files"])
    with pytest.raises(ValueError, match="protected"):
        migration.build_migration_plan(source, new,
            allowed_changed_files=[migration.SUPERVISOR, "src/wlr50_clean/ppo/semantic_reward.py"],
            reason="invalid", evaluator_review=review, project_root=tmp_path)
