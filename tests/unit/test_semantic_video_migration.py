from __future__ import annotations

import copy
import json

import pytest

from wlr50_clean.ppo import semantic_migration as migration


CODE = "src/wlr50_clean/ppo/semantic_migration.py"
VIDEO = "src/wlr50_clean/ppo/semantic_video.py"
REVIEW = {"reason": "Reviewed exact source diff: video capture instrumentation only; task and learning code unchanged"}
PROTECTED = (
    migration.SUPERVISOR, migration.STAGE_SPEC,
    "src/wlr50_clean/ppo/semantic_reward.py",
    "src/wlr50_clean/ppo/semantic_backend.py",
    "configs/ppo_semantic_v2/observation_schema.json",
    "configs/ppo_semantic_v2/action_schema.json",
    "configs/ppo_semantic_v2/execution_profile.yaml",
    "configs/ppo_semantic_v2/quality_score.yaml",
    *sorted(migration.VECTOR_FILES),
)


def refresh(contract):
    contract["runtime_content_sha256"] = migration.digest(contract["files"])


def save_source(root, contract):
    checkpoint = root / "source.pt"
    checkpoint.write_bytes(b"immutable checkpoint bytes; no optimizer operation in migration")
    checkpoint.with_name("source_manifest.json").write_text(json.dumps({
        "schema": "wlr50_clean.semantic_checkpoint.v1",
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": migration.file_sha(checkpoint),
        "save_load_round_trip": True,
        "runtime_contract": contract,
        "execution_topology": migration.topology(1),
        "global_policy_decisions": 10112,
        "optimizer_steps": 880,
    }), encoding="utf-8")
    return checkpoint


def fixture(root, *, existing=False):
    files = {name: "f" * 64 for name in PROTECTED}
    files[CODE] = "1" * 64
    old = {"source_git_commit": "1" * 40, "files": files,
           "frozen_A_files": {"A": "a" * 64}, "observation_dimension": 324,
           "action_dimension": 12, "physics_hz": 120, "decision_hz": 15,
           "task_timeout_s": 200, "versions": {"torch": "locked"}}
    for relative in sorted(migration.VIDEO_FILES):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# reviewed video source\n", encoding="utf-8")
        if existing:
            old["files"][relative] = migration.file_sha(path)
    refresh(old)
    new = copy.deepcopy(old)
    new["source_git_commit"] = "2" * 40
    new["files"][CODE] = "2" * 64
    if existing:
        (root / VIDEO).write_text("# separately reviewed video repair\n", encoding="utf-8")
    for relative in sorted(migration.VIDEO_FILES):
        new["files"][relative] = migration.file_sha(root / relative)
    refresh(new)
    return save_source(root, old), old, new


def delta(old, new):
    return sorted(path for path in old["files"].keys() | new["files"].keys()
                  if old["files"].get(path) != new["files"].get(path))


def build(source, old, new, root, **kwargs):
    return migration.build_migration_plan(source, new, allowed_changed_files=delta(old, new),
        reason="Independent video source boundary", video_review=REVIEW, project_root=root, **kwargs)


@pytest.mark.parametrize("existing", [False, True])
def test_new_and_repaired_video_bind_exact_source_and_target_without_changing_topology(tmp_path, existing):
    source, old, new = fixture(tmp_path, existing=existing)
    original_checkpoint = source.read_bytes()
    plan = build(source, old, new, tmp_path)
    factor = plan["video_instrumentation_factor"]
    assert factor["added_files"] == ([] if existing else sorted(migration.VIDEO_FILES))
    assert factor["modified_files"] == ([VIDEO] if existing else [])
    assert factor["video_file_hashes"] == {
        path: {"before": old["files"].get(path), "after": new["files"][path]}
        for path in sorted(migration.VIDEO_FILES)}
    assert factor["scope_is_reviewer_assertion_not_semantic_equivalence_proof"] is True
    assert factor["video_success_or_improvement_certified"] is False
    assert "training_or_task_behavior_changed" not in factor
    assert plan["source_contract_sha256"] == migration.digest(old)
    assert plan["target_contract_sha256"] == migration.digest(new)
    assert plan["changed_file_hashes"] == {
        path: {"before": old["files"].get(path), "after": new["files"][path]}
        for path in delta(old, new)}
    assert "execution_factor" not in plan and plan["geometric_factor"] is None
    assert plan["preserve_actor_critic_optimizer_normalizer_rng_and_budget"] is True
    assert plan["discard_old_rollout_storage"] is True
    assert plan["physics_resume"] == "fresh_legal_P01_reset"
    path = tmp_path / "video_plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    assert migration.validate_migration_plan(source, new, path, project_root=tmp_path)["video_instrumentation_factor"] == factor
    assert migration.source_num_envs(migration.checkpoint_metadata(source)) == 1
    assert source.read_bytes() == original_checkpoint


@pytest.mark.parametrize("factor", ["prior_evidence", "qualification_evidence", "evaluator_review", "execution_evidence"])
def test_video_review_cannot_mix_task_or_topology_permission(tmp_path, factor):
    source, old, new = fixture(tmp_path)
    with pytest.raises(ValueError, match="separate reviewed boundaries"):
        build(source, old, new, tmp_path, **{factor: {}})


@pytest.mark.parametrize("protected", PROTECTED)
def test_video_review_cannot_change_protected_runtime_or_profile(tmp_path, protected):
    source, old, new = fixture(tmp_path)
    new["files"][protected] = "b" * 64
    refresh(new)
    with pytest.raises(ValueError, match="cannot change"):
        build(source, old, new, tmp_path)


@pytest.mark.parametrize("fault", ["partial_source", "missing_target", "deleted_instrumentation", "disk_mismatch",
                                  "undeclared_video", "missing_review", "frozen_A"])
def test_video_publication_requires_complete_retained_sources_and_exact_contract(tmp_path, fault):
    source, old, new = fixture(tmp_path, existing=fault == "missing_target")
    if fault == "partial_source":
        old["files"][VIDEO] = new["files"][VIDEO]
        refresh(old)
        save_source(tmp_path, old)
    elif fault == "missing_target":
        del new["files"][VIDEO]
    elif fault == "deleted_instrumentation":
        del new["files"][CODE]
    elif fault == "disk_mismatch":
        (tmp_path / VIDEO).write_text("# unreviewed later mutation\n", encoding="utf-8")
    elif fault == "frozen_A":
        new["frozen_A_files"]["A"] = "0" * 64
    refresh(new)
    with pytest.raises(ValueError):
        migration.build_migration_plan(source, new,
            allowed_changed_files=[CODE] if fault == "undeclared_video" else delta(old, new),
            reason="reviewed video", video_review=None if fault == "missing_review" else REVIEW,
            project_root=tmp_path)


@pytest.mark.parametrize("review", [{}, {"reason": " "}, {"reason": 123}, {"reason": "review", "waive": True}])
def test_video_review_reason_is_explicit_and_exact(tmp_path, review):
    source, old, new = fixture(tmp_path)
    with pytest.raises(ValueError, match="explicit review reason"):
        migration.build_migration_plan(source, new, allowed_changed_files=delta(old, new),
            reason="reviewed", video_review=review, project_root=tmp_path)


@pytest.mark.parametrize("fault", ["before_hash", "after_hash", "added_files", "modified_files", "target_contract", "source_bytes"])
def test_plan_and_checkpoint_tampering_is_rejected(tmp_path, fault):
    source, old, new = fixture(tmp_path, existing=True)
    plan = build(source, old, new, tmp_path)
    factor = plan["video_instrumentation_factor"]
    if fault in ("before_hash", "after_hash"):
        factor["video_file_hashes"][VIDEO][fault.split("_")[0]] = "0" * 64
    elif fault in ("added_files", "modified_files"):
        factor[fault] = ["unreviewed.py"]
    elif fault == "target_contract":
        plan["target_contract_sha256"] = "0" * 64
    else:
        source.write_bytes(b"replaced checkpoint")
    path = tmp_path / "bad_plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    with pytest.raises(ValueError):
        migration.validate_migration_plan(source, new, path, project_root=tmp_path)


@pytest.mark.parametrize("num_envs", [1, 8])
def test_existing_topology_v1_is_unchanged(num_envs):
    assert migration.topology(num_envs) == {
        "schema": "wlr50_clean.semantic_execution_topology.v1", "num_envs": num_envs,
        "observation_dimension": 324, "action_dimension": 12,
        "rollout_decisions_per_env": 128, "reset_sampling": "P01_only",
        "phase_suffix_curriculum_implemented": False,
        "peer_reset": "none" if num_envs == 1 else "synchronous_done_with_gamma_V_actual_final_obs",
        "task_timeout_bootstrap": False, "physical_state_saved": False}


def test_pre_video_instrumentation_plan_retains_exact_old_schema(tmp_path):
    source, old, new = fixture(tmp_path)
    for relative in migration.VIDEO_FILES:
        del new["files"][relative]
    refresh(new)
    sidecar = source.with_name("source_manifest.json")
    # Golden shape from the existing plan writer, before optional video support.
    legacy = {"schema": migration.SCHEMA, "reason": "old instrumentation boundary",
        "source_checkpoint": str(source.resolve()), "source_checkpoint_sha256": migration.file_sha(source),
        "source_manifest_sha256": migration.file_sha(sidecar),
        "source_contract_sha256": migration.digest(old), "target_contract_sha256": migration.digest(new),
        "source_git_commit": old["source_git_commit"], "target_git_commit": new["source_git_commit"],
        "allowed_changed_files": [CODE], "changed_file_hashes": {CODE: {"before": "1" * 64, "after": "2" * 64}},
        "geometric_factor": None, "observation_dimension": 324, "action_dimension": 12,
        "preserve_actor_critic_optimizer_normalizer_rng_and_budget": True,
        "discard_old_rollout_storage": True, "physics_resume": "fresh_legal_P01_reset"}
    path = tmp_path / "old_plan.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")
    verified = migration.validate_migration_plan(source, new, path, project_root=tmp_path)
    assert verified == {"plan_path": str(path.resolve()), "plan_sha256": migration.file_sha(path), **legacy}
