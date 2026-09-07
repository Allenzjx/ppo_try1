"""Explicit, immutable semantic checkpoint version boundaries, without Isaac."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA = "wlr50_clean.semantic_checkpoint_migration.v1"
STAGE_SPEC = "configs/ppo_semantic_v2/stage_task_spec.yaml"
SUPERVISOR = "src/wlr50_clean/ppo/semantic_supervisor.py"
PRIOR_DIAGNOSTIC_HEAD = "84c607a2ffbba36f46a0e70dbb886227c0c32ec5"
QUALIFICATION_DIAGNOSTIC_HEAD = "38276d12f9dc1bdc3c81f3d0130e59ff162d2597"
# Deliberately exact reviewed class revisions, not a general evaluator exemption.
QUALIFICATION_OLD_CLASS_SHA256 = "7b22b0c4eae20a448cca23218aab3aadfd585dc64aa0197af568859276c303c6"
QUALIFICATION_NEW_CLASS_SHA256 = "8eaebeec34ac32d79d3eb7cfa810a8a966a92a6aa10095d32c6e2efeae351d36"
PRIOR_CONFIG_LINES = (
    "  approach_wheel_prior_rad_s: [0.3, 0.3, 0.3, 0.3]\n"
    "  approach_wheel_prior_source: P01\n"
    "  approach_wheel_prior_scope: P02_current_physical_goal_feedback_only\n"
)
INSTRUMENTATION_FILES = frozenset({
    "src/wlr50_clean/ppo/semantic_cli.py", "src/wlr50_clean/ppo/semantic_training.py",
    "src/wlr50_clean/ppo/semantic_migration.py", "src/wlr50_clean/ppo/semantic_legacy_evaluation.py",
    "src/wlr50_clean/ppo/semantic_metrics.py",
    "scripts/run_semantic_ppo.ps1",
})
VIDEO_FILES = frozenset({
    "src/wlr50_clean/ppo/semantic_video.py",
    "src/wlr50_clean/ppo/semantic_video_cli.py",
    "scripts/run_semantic_video.ps1",
})
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def checkpoint_metadata(checkpoint: Path) -> dict[str, Any]:
    checkpoint = checkpoint.resolve(strict=True)
    sidecar = checkpoint.with_name(checkpoint.stem + "_manifest.json")
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    if (data.get("schema") != "wlr50_clean.semantic_checkpoint.v1"
            or data.get("checkpoint_path") != str(checkpoint)
            or data.get("checkpoint_sha256") != file_sha(checkpoint)
            or data.get("save_load_round_trip") is not True):
        raise ValueError("source semantic checkpoint/sidecar integrity mismatch")
    return data


def build_v3_warm_start_record(checkpoint: Path, current_contract: Mapping[str, Any], *,
                                project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Explicit new-MDP boundary; never a relaxation of v2 exact-resume rules."""
    metadata = checkpoint_metadata(checkpoint)
    from .semantic_policy_distribution import policy_version_from_metadata
    from .semantic_return_profile import (
        LEGACY_RETURN_PROFILE, RETURN_PROFILE, reward_return_profile, runner_return_profile,
    )
    from .semantic_reward import load_semantic_reward_config
    import yaml
    # Do not reinterpret source actor metadata using the target factory's new
    # discount. Validate its complete, explicitly recognized historical config.
    policy_version_from_metadata(metadata)
    old, new = _contract(metadata["runtime_contract"]), _contract(current_contract)
    source_version = metadata.get("semantic_version", "v2")
    if source_version not in ("v2", "v3") or new.get("semantic_version") != "v3":
        raise ValueError("new-MDP warm start requires a v2 or v3 source checkpoint and a v3 target")
    if old.get("semantic_version", "v2") != source_version:
        raise ValueError("new-MDP source semantic version differs from its runtime contract")
    source_global = int(metadata.get("global_policy_decisions", 0))
    source_spent = dict(metadata.get("stage_requested_decisions", {}))
    origin = source_global
    if source_version == "v3":
        origin = metadata.get("new_mdp_origin_global_policy_decisions")
        if (type(origin) is not int or not 0 <= origin <= source_global
                or set(source_spent) != {"smoke", "phase_suffix", "full_episode"}
                or any(type(value) is not int or value < 0 for value in source_spent.values())):
            raise ValueError("v3 new-MDP continuation requires intact original budget accounting")
    for key in ("frozen_A_files", "physics_hz", "decision_hz", "task_timeout_s",
                "timeout_bootstrap", "rsl_rl_version", "local_runtime_versions"):
        if old.get(key) != new.get(key):
            raise ValueError(f"new-MDP continuation cannot change physical/runtime contract: {key}")
    config_names = ("stage_task_spec.yaml", "execution_profile.yaml", "reward_config.yaml",
                    "observation_schema.json", "action_schema.json", "quality_score.yaml")
    config_records = {}
    for name in config_names:
        source, target = f"configs/ppo_semantic_{source_version}/{name}", f"configs/ppo_semantic_v3/{name}"
        if source not in old["files"] or target not in new["files"]:
            raise ValueError(f"new-MDP requires both versioned configuration records: {name}")
        if file_sha(project_root / target) != new["files"][target]:
            raise ValueError(f"new-MDP target config bytes changed: {name}")
        config_records[name] = {"source_path": source, "source_sha256": old["files"][source],
                                "target_path": target, "target_sha256": new["files"][target]}
    reward_record = config_records["reward_config.yaml"]
    source_reward = yaml.safe_load(_version_bytes(
        project_root, old, reward_record["source_path"], prefer_worktree=True))
    source_return = reward_return_profile(source_reward, semantic_version=source_version)
    if source_return != runner_return_profile(metadata["runner_config"], semantic_version=source_version):
        raise ValueError("source runner discount/profile differs from its hash-bound historical reward config")
    target_reward = load_semantic_reward_config(project_root / reward_record["target_path"])
    target_return = reward_return_profile(target_reward.values, semantic_version="v3")
    if source_return != target_return and (source_return["version"], target_return["version"]) != (
            LEGACY_RETURN_PROFILE, RETURN_PROFILE):
        raise ValueError("unsupported new-MDP return-profile transition")
    return_transition = {
        "source": source_return, "target": target_return,
        "reward_configuration": dict(reward_record),
        "reward_discount_changed": source_return["gamma"] != target_return["gamma"],
        "gae_estimator_changed": source_return["lambda"] != target_return["lambda"],
        "rollout_storage_inherited": False,
    }
    # Identity learned normalizers do not imply identical fixed observation
    # preprocessing. Require the complete old/new group ordering and scales to
    # match; an incompatible encoder needs a separate reviewed transformation.
    before = json.loads(_version_text(project_root, old, config_records["observation_schema.json"]["source_path"], prefer_worktree=True))
    after = json.loads((project_root / "configs/ppo_semantic_v3/observation_schema.json").read_text(encoding="utf-8"))
    for key in ("feature_groups", "clip", "maximum_task_duration_s", "fixed_chassis_to_body_wxyz",
                "level_reference", "normalization"):
        if before.get(key) != after.get(key):
            raise ValueError(f"warm-start actor observation preprocessing changed: {key}")
    if sum(group["size"] for group in before.get("feature_groups", ())) != 324:
        raise ValueError("warm start requires the existing 324-observation network")
    # Verify the comparison's historical source before AppLauncher/reset; defer
    # only immutable materialization, not source availability, to publication.
    _version_bytes(project_root, old, config_records["execution_profile.yaml"]["source_path"], prefer_worktree=True)
    checkpoint = checkpoint.resolve(strict=True)
    sidecar = checkpoint.with_name(checkpoint.stem + "_manifest.json")
    return {"schema": "wlr50_clean.semantic_v3_new_mdp_warm_start.v1", "exact_mdp_resume": False,
            "source_checkpoint": str(checkpoint), "source_checkpoint_sha256": file_sha(checkpoint),
            "source_manifest_sha256": file_sha(sidecar),
            "source_semantic_version": source_version, "source_global_policy_decisions": source_global,
            "source_stage_requested_decisions": source_spent,
            "target_stage_requested_decisions": (source_spent if source_version == "v3" else
                                                    dict.fromkeys(("smoke", "phase_suffix", "full_episode"), 0)),
            "new_mdp_origin_global_policy_decisions": origin,
            "source_runtime_contract": old, "target_runtime_contract": new,
            "configuration_transition": config_records,
            "return_horizon_transition": return_transition,
            "runtime_changed_files": sorted(key for key in set(old["files"]) | set(new["files"])
                                             if old["files"].get(key) != new["files"].get(key)),
            "network": {"observation_dimension": 324, "raw_action_dimension": 12,
                        "actor": "preserve_all_parameters_including_learned_std",
                        "critic": "preserve_weights_then_online_recalibration_on_current_task_distribution",
                        "normalizers": "identity_RSL_state; identical_fixed_schema_preprocessing"},
            "optimizer": {"kind": "Adam", "state": "reset_all_moments", "initial_learning_rate": 3e-5,
                          "reason": "explicit versioned new-MDP boundary; changed files recorded above"},
            "old_rollout_buffer_inherited": False, "physical_state_inherited": False,
            "rng": "restore_verified_training_rng_then_sample_fresh_new_MDP_rollouts",
            "action_output_semantics": "same raw actor output uses hash-bound source/target physical profiles; ranges may be unchanged",
            "stage_accounting": ("preserve existing v3 spent budgets and original v3 origin; preserve lifetime counters"
                                 if source_version == "v3" else
                                 "new v3 requested budgets; preserve lifetime global/update counters"),
            "reset_sampling": "explicit_fixed_from_phase_per_run; reset_only_rollin_excluded_from_PPO_credit"}


def v3_warm_start_checkpoint_name(record: Mapping[str, Any]) -> str:
    """Immutable revision-bound initial publication, separate from every prior boundary."""
    target = _contract(record["target_runtime_contract"])
    return (f"checkpoint_initial_v3_from_{int(record['source_global_policy_decisions']):09d}"
            f"_s{record['source_checkpoint_sha256'][:12]}"
            f"_g{target['source_git_commit'][:12]}_{target['runtime_content_sha256']}.pt")


def warm_start_source_execution_profile(record: Mapping[str, Any], output_directory: Path, *,
                                        project_root: Path = PROJECT_ROOT) -> Path:
    """Resolve exact source bytes, never substitute today's configuration silently."""
    source = _contract(record["source_runtime_contract"])
    binding = record["configuration_transition"]["execution_profile.yaml"]
    relative, expected = binding["source_path"], binding["source_sha256"]
    if source["files"].get(relative) != expected:
        raise ValueError("source execution profile binding differs from checkpoint runtime")
    current = project_root / relative
    if current.is_file() and file_sha(current) == expected:
        return current.resolve()
    raw = _version_bytes(project_root, source, relative)
    destination = output_directory / "source_configuration" / f"execution_profile_{expected}.yaml"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if file_sha(destination) != expected:
            raise ValueError("immutable source execution profile snapshot differs from checkpoint")
    else:
        with destination.open("xb") as stream:
            stream.write(raw)
    return destination.resolve()


def _contract(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    files = result.get("files")
    if (not isinstance(files, dict) or not files or any(not isinstance(v, str) or len(v) != 64 for v in files.values())
            or result.get("runtime_content_sha256") != digest(files)
            or len(str(result.get("source_git_commit", ""))) != 40):
        raise ValueError("migration runtime inventory digest/revision is malformed")
    return result


def _version_text(project_root: Path, contract: dict, relative: str, *, prefer_worktree: bool = False) -> str:
    return _version_bytes(project_root, contract, relative, prefer_worktree=prefer_worktree).decode("utf-8").replace("\r\n", "\n")


def _version_bytes(project_root: Path, contract: dict, relative: str, *, prefer_worktree: bool = False) -> bytes:
    current = project_root / relative
    expected = contract["files"][relative]
    if prefer_worktree and current.is_file() and file_sha(current) == expected:
        return current.read_bytes()
    raw = subprocess.run(["git", "-C", str(project_root), "show", f"{contract['source_git_commit']}:{relative}"],
                         check=True, capture_output=True).stdout
    # Git stores LF; the pinned Windows working tree may materialize CRLF.
    candidates = (raw, raw.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    for data in candidates:
        if hashlib.sha256(data).hexdigest() == expected:
            return data
    raise ValueError(f"versioned source bytes do not match the checkpoint contract: {relative}")


def _geometric_factor(project_root: Path, old: dict, new: dict, *, prior_transition: bool = False) -> dict[str, Any] | None:
    before = _version_text(project_root, old, STAGE_SPEC)
    after = _version_text(project_root, new, STAGE_SPEC, prefer_worktree=True)
    if prior_transition:
        nominal = after.split("\nnominal:\n", 1)
        if (len(nominal) != 2 or PRIOR_CONFIG_LINES in before or after.count(PRIOR_CONFIG_LINES) != 1
                or PRIOR_CONFIG_LINES not in nominal[1].split("\nstage_defaults:", 1)[0] + "\n"):
            raise ValueError("prior transition requires exactly the declared nominal P01 wheel-prior additions")
        after = after.replace(PRIOR_CONFIG_LINES, "")
    source, target = "  approach_min_m: -0.18\n", "  approach_min_m: -0.005\n"
    if prior_transition and before == after and target in before:
        return None
    if before.count(source) != 1 or before.replace(source, target) != after:
        raise ValueError("only the declared approach_min_m -0.18 to -0.005 geometric factor is permitted")
    return {"file": STAGE_SPEC, "field": "geometry.approach_min_m", "before": -0.18, "after": -0.005}


def _prior_factor(project_root: Path, old: dict, new: dict, evidence: Mapping[str, Any] | None,
                  source_checkpoint: Path, *, qualification_transition: bool = False) -> dict[str, Any]:
    if not isinstance(evidence, Mapping) or set(evidence) != {"B", "C"}:
        raise ValueError("nominal prior transition requires explicit B84c and C84c diagnostic evidence")
    before = _version_text(project_root, old, SUPERVISOR)
    after = _version_text(project_root, new, SUPERVISOR, prefer_worktree=True)
    if qualification_transition:
        _, _, after = _qualified_class_delta(before, after)
    def split(text):
        start, end = "class NominalMotionProvider:", "class SemanticControllerAdapter:"
        if text.count(start) != 1 or text.count(end) != 1:
            raise ValueError("nominal provider class boundaries are ambiguous")
        prefix, rest = text.split(start, 1)
        body, suffix = rest.split(end, 1)
        return prefix, body, suffix
    old_parts, new_parts = split(before), split(after)
    if old_parts[0] != new_parts[0] or old_parts[2] != new_parts[2] or old_parts[1] == new_parts[1]:
        raise ValueError("only the NominalMotionProvider class block may change; task evaluator and controller adapter remain frozen")
    rows = {}
    for role, mode in (("B", "semantic_prior_eval"), ("C", "semantic_residual_eval")):
        path = Path(evidence[role]).resolve(strict=True)
        if not path.is_relative_to((project_root / "runs/ppo_semantic_v2").resolve()):
            raise ValueError("prior diagnostic must belong to the isolated semantic run root")
        data = json.loads(path.read_text(encoding="utf-8"))
        lifecycle_path = path.with_name("run_manifest.json")
        lifecycle = json.loads(lifecycle_path.read_text(encoding="utf-8"))
        if (path.name != "evaluation_manifest.json" or data.get("mode") != mode
                or data.get("runtime_contract", {}).get("source_git_commit") != PRIOR_DIAGNOSTIC_HEAD
                or data.get("optimizer_updates_during_evaluation") != 0 or data.get("task_success") is not False
                or data.get("termination_reason") != "INCOMPLETE_CONTROLLER_BLOCKED"
                or lifecycle.get("lifecycle") != "SUCCEEDED" or lifecycle.get("result") != data):
            raise ValueError("prior diagnostic is not a finalized zero-update B/C84c physical failure")
        if ((role == "B" and data.get("checkpoint") is not None)
                or (role == "C" and (data.get("checkpoint_sha256") != file_sha(source_checkpoint)
                                     or Path(str(data.get("checkpoint"))).resolve() != source_checkpoint))):
            raise ValueError("prior diagnostic must compare zero B and the unchanged source checkpoint C")
        rows[role] = {"evaluation_manifest": str(path), "evaluation_manifest_sha256": file_sha(path),
                      "run_manifest_sha256": file_sha(lifecycle_path), "mode": mode,
                      "policy_decisions": data["policy_decisions"], "duration_s": data["duration_s"]}
    return {"diagnostic_git_commit": PRIOR_DIAGNOSTIC_HEAD, "source_file": SUPERVISOR,
            "allowed_source_scope": "NominalMotionProvider_class_only",
            "unchanged_prefix_sha256": hashlib.sha256(old_parts[0].encode()).hexdigest(),
            "unchanged_suffix_sha256": hashlib.sha256(old_parts[2].encode()).hexdigest(),
            "wheel_prior_rad_s": [0.3] * 4, "source_nominal_phase": "P01",
            "scope": "P02_current_physical_goal_feedback_only", "evidence": rows}


def _qualified_class_delta(before: str, after: str) -> tuple[str, str, str]:
    start, end = "class TaskEvaluator:", "class TaskStageSupervisor:"
    def split(text):
        if text.count(start) != 1 or text.count(end) != 1:
            raise ValueError("physical qualification class boundaries are ambiguous")
        prefix, rest = text.split(start, 1)
        body, suffix = rest.split(end, 1)
        return prefix, start + body, end + suffix
    old, new = split(before), split(after)
    old_sha, new_sha = (hashlib.sha256(value[1].encode()).hexdigest() for value in (old, new))
    if old_sha != QUALIFICATION_OLD_CLASS_SHA256 or new_sha != QUALIFICATION_NEW_CLASS_SHA256:
        raise ValueError("physical qualification change is not the exact reviewed TaskEvaluator revision")
    return old_sha, new_sha, new[0] + old[1] + new[2]


def _qualification_factor(project_root: Path, old: dict, new: dict, evidence: Path) -> dict[str, Any]:
    before = _version_text(project_root, old, SUPERVISOR)
    after = _version_text(project_root, new, SUPERVISOR, prefer_worktree=True)
    old_sha, new_sha, _ = _qualified_class_delta(before, after)
    path = Path(evidence).resolve(strict=True)
    if not path.is_relative_to((project_root / "runs/ppo_semantic_v2/prior_B").resolve()) or path.name != "evaluation_manifest.json":
        raise ValueError("qualification evidence must be the isolated B382 physical evaluation")
    data = json.loads(path.read_text(encoding="utf-8"))
    run_path = path.with_name("run_manifest.json")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    if (data.get("mode") != "semantic_prior_eval" or data.get("checkpoint") is not None
            or data.get("runtime_contract", {}).get("source_git_commit") != QUALIFICATION_DIAGNOSTIC_HEAD
            or data.get("optimizer_updates_during_evaluation") != 0 or data.get("task_success") is not False
            or data.get("termination_reason") != "INCOMPLETE_CONTROLLER_BLOCKED"
            or data.get("policy_decisions") != 1140 or data.get("observed_physics_ticks") != 9120
            or abs(float(data.get("duration_s", -1)) - 76.0) > 1e-8
            or run.get("lifecycle") != "SUCCEEDED" or run.get("result") != data):
        raise ValueError("qualification evidence is not the finalized 76s B382 zero-update failure")
    artifacts = {}
    for name in ("physical_observations.jsonl", "native_tick_audit.jsonl", "stage_transition_evidence.jsonl"):
        source = Path(data["evaluation_artifacts"][name]).resolve(strict=True)
        if source != path.with_name(name):
            raise ValueError("qualification raw evidence is not owned by the diagnostic run")
        artifacts[name] = {"path": str(source), "sha256": file_sha(source), "bytes": source.stat().st_size}
    raw_path = Path(artifacts["physical_observations.jsonl"]["path"])
    with raw_path.open("rb") as stream:
        count = sum(1 for _ in stream)
    if count != 9121:
        raise ValueError("qualification evidence must retain initial plus all 9120 real physics observations")
    return {"diagnostic_git_commit": QUALIFICATION_DIAGNOSTIC_HEAD,
            "evaluation_manifest": str(path), "evaluation_manifest_sha256": file_sha(path),
            "run_manifest_sha256": file_sha(run_path), "raw_artifacts": artifacts,
            "source_file": SUPERVISOR, "allowed_source_scope": "exact_reviewed_TaskEvaluator_physical_qualification_revision",
            "before_class_sha256": old_sha, "after_class_sha256": new_sha,
            "semantics": ["positive_temporal_lift_not_descent_range", "pre_cross_ground_contact_invalidates_lift_eligibility",
                          "crossing_requires_current_air_clearance_or_real_top_contact"],
            "old_rollouts_are_not_reused": True, "old_checkpoint_is_not_a_full_success_claim": True}


def _reviewed_evaluator_factor(project_root: Path, old: dict, new: dict,
                               review: Mapping[str, Any]) -> dict[str, Any]:
    """Bind a separately reviewed evaluator-only repair, not a general MDP waiver."""
    if not isinstance(review, Mapping) or set(review) != {"reason", "counterexample_tests"}:
        raise ValueError("evaluator review requires exact reason and counterexample tests")
    reason, nodes = review["reason"], review["counterexample_tests"]
    if (not isinstance(reason, str) or not reason.strip() or not isinstance(nodes, (list, tuple))
            or not nodes or any(not isinstance(node, str) for node in nodes) or len(set(nodes)) != len(nodes)):
        raise ValueError("evaluator review needs a reason and unique counterexample test nodes")
    before = _version_text(project_root, old, SUPERVISOR)
    after = _version_text(project_root, new, SUPERVISOR, prefer_worktree=True)
    def split(text):
        start, end = "class TaskEvaluator:", "class TaskStageSupervisor:"
        if text.count(start) != 1 or text.count(end) != 1:
            raise ValueError("reviewed TaskEvaluator class boundaries are ambiguous")
        prefix, rest = text.split(start, 1)
        body, suffix = rest.split(end, 1)
        return prefix, start + body, end + suffix
    old_parts, new_parts = split(before), split(after)
    if old_parts[0] != new_parts[0] or old_parts[2] != new_parts[2] or old_parts[1] == new_parts[1]:
        raise ValueError("reviewed evaluator repair must change only TaskEvaluator; all outside bytes remain identical")
    sources = {}
    for node in nodes:
        parts = node.split("::")
        if len(parts) != 2:
            raise ValueError("counterexample test must be an explicit file::function node")
        relative, function = parts
        path = (project_root / relative).resolve(strict=True)
        if (not relative.startswith("tests/unit/") or "\\" in relative or path.suffix != ".py"
                or not path.is_relative_to((project_root / "tests/unit").resolve())
                or not function.startswith("test_")):
            raise ValueError("counterexample source must be a named unit test inside the project")
        sources[relative] = {"path": str(path), "sha256": file_sha(path)}
    sha = lambda text: hashlib.sha256(text.encode()).hexdigest()
    return {"schema": "wlr50_clean.reviewed_task_evaluator_factor.v1",
            "source_file": SUPERVISOR, "allowed_source_scope": "TaskEvaluator_class_only",
            "review_reason": reason.strip(), "counterexample_tests": list(nodes),
            "test_sources": sources, "test_nodes_are_review_references_not_pass_certification": True,
            "before_class_sha256": sha(old_parts[1]), "after_class_sha256": sha(new_parts[1]),
            "unchanged_prefix_sha256": sha(old_parts[0]), "unchanged_suffix_sha256": sha(old_parts[2]),
            "source_git_commit": old["source_git_commit"], "target_git_commit": new["source_git_commit"],
            "source_contract_sha256": digest(old), "target_contract_sha256": digest(new),
            "old_rollouts_are_not_reused": True, "old_checkpoint_is_not_a_full_success_claim": True}


def _video_instrumentation_factor(old, new, review, delta, project_root):
    """Bind reviewed video source changes; never prove behavior or certify an outcome."""
    if (not isinstance(review, Mapping) or set(review) != {"reason"}
            or not isinstance(review["reason"], str) or not review["reason"].strip()):
        raise ValueError("video instrumentation requires an explicit review reason")
    if set(delta) - INSTRUMENTATION_FILES - VIDEO_FILES:
        raise ValueError("video publication cannot change task/nominal/reward/action/backend or execution modules")
    source_files = VIDEO_FILES & set(old["files"])
    if source_files not in (set(), VIDEO_FILES) or not VIDEO_FILES <= set(new["files"]):
        raise ValueError("video source must be absent or complete and target must retain all three reviewed files")
    hashes = {path: file_sha(project_root / path) for path in sorted(VIDEO_FILES)}
    if any(hashes[path] != new["files"][path] for path in VIDEO_FILES):
        raise ValueError("video source bytes differ from reviewed target contract")
    return {"schema": "wlr50_clean.semantic_video_instrumentation_migration.v1",
            "review_reason": review["reason"].strip(),
            "video_file_hashes": {path: {"before": old["files"].get(path), "after": hashes[path]}
                                  for path in sorted(VIDEO_FILES)},
            "added_files": sorted(VIDEO_FILES - source_files),
            "modified_files": sorted(path for path in source_files if old["files"][path] != hashes[path]),
            "reviewed_scope": "video_instrumentation_only_no_task_reward_or_topology_change",
            "scope_is_reviewer_assertion_not_semantic_equivalence_proof": True,
            "video_success_or_improvement_certified": False}


def build_migration_plan(checkpoint: Path, current_contract: Mapping[str, Any], *,
                         allowed_changed_files: Sequence[str], reason: str,
                         prior_evidence: Mapping[str, Any] | None = None,
                         qualification_evidence: Path | None = None,
                         evaluator_review: Mapping[str, Any] | None = None,
                         execution_evidence: Mapping[str, Any] | None = None,
                         video_review: Mapping[str, Any] | None = None,
                         project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Build a reviewed plan after committing the new runtime; does not write."""
    checkpoint = Path(checkpoint).resolve(strict=True)
    metadata = checkpoint_metadata(checkpoint)
    old, new = _contract(metadata["runtime_contract"]), _contract(current_contract)
    old_fixed = {key: value for key, value in old.items() if key not in ("files", "runtime_content_sha256", "source_git_commit")}
    new_fixed = {key: value for key, value in new.items() if key not in ("files", "runtime_content_sha256", "source_git_commit")}
    if old_fixed != new_fixed:
        raise ValueError("migration cannot change frozen physics, runtime versions, rates or budgets")
    declared = list(allowed_changed_files)
    if not reason.strip() or len(set(declared)) != len(declared):
        raise ValueError("migration requires a reason and unique exact changed-file names")
    delta = sorted(path for path in set(old["files"]) | set(new["files"]) if old["files"].get(path) != new["files"].get(path))
    video = None
    if video_review is not None:
        if any(value is not None for value in (
                prior_evidence, qualification_evidence, evaluator_review, execution_evidence)):
            raise ValueError("video instrumentation and task/execution factors require separate reviewed boundaries")
        video = _video_instrumentation_factor(old, new, video_review, delta, Path(project_root))
    execution = None
    additional = set(VIDEO_FILES) if video is not None else set()
    if execution_evidence is not None:
        if (set(delta) - INSTRUMENTATION_FILES - VECTOR_FILES or prior_evidence is not None
                or qualification_evidence is not None or evaluator_review is not None):
            raise ValueError("execution migration cannot mix task/nominal/configuration factors")
        execution = build_execution_factor(metadata, new, execution_evidence, project_root=Path(project_root))
        additional = set(VECTOR_FILES)
    if sorted(declared) != delta or not set(delta) <= INSTRUMENTATION_FILES | {STAGE_SPEC, SUPERVISOR} | additional:
        raise ValueError("migration delta is undeclared or touches protected observation/action/reward/backend/physics files")
    if any(path not in new["files"] for path in delta):
        raise ValueError("migration cannot delete runtime files")
    evaluator = None
    if evaluator_review is not None:
        if (SUPERVISOR not in delta or STAGE_SPEC in delta or prior_evidence is not None
                or qualification_evidence is not None):
            raise ValueError("reviewed evaluator repair cannot mix prior/configuration or historical qualification factors")
        evaluator = _reviewed_evaluator_factor(Path(project_root), old, new, evaluator_review)
    prior_transition = SUPERVISOR in delta and evaluator is None
    if prior_transition != (prior_evidence is not None) or (prior_transition and STAGE_SPEC not in delta):
        raise ValueError("nominal source change and explicit prior evidence/config change must occur together")
    if qualification_evidence is not None and not prior_transition:
        raise ValueError("qualification permission requires the explicitly declared supervisor source change")
    geometric = _geometric_factor(Path(project_root), old, new, prior_transition=prior_transition) if STAGE_SPEC in delta else None
    prior = _prior_factor(Path(project_root), old, new, prior_evidence, checkpoint,
                           qualification_transition=qualification_evidence is not None) if prior_transition else None
    qualification = _qualification_factor(Path(project_root), old, new, qualification_evidence) if qualification_evidence is not None else None
    sidecar = checkpoint.with_name(checkpoint.stem + "_manifest.json")
    result = {"schema": SCHEMA, "reason": reason.strip(), "source_checkpoint": str(checkpoint),
            "source_checkpoint_sha256": file_sha(checkpoint), "source_manifest_sha256": file_sha(sidecar),
            "source_contract_sha256": digest(old), "target_contract_sha256": digest(new),
            "source_git_commit": old["source_git_commit"], "target_git_commit": new["source_git_commit"],
            "allowed_changed_files": delta, "changed_file_hashes": {path: {"before": old["files"].get(path), "after": new["files"][path]} for path in delta},
            "geometric_factor": geometric, "observation_dimension": 324, "action_dimension": 12,
            "preserve_actor_critic_optimizer_normalizer_rng_and_budget": True,
            "discard_old_rollout_storage": True, "physics_resume": "fresh_legal_P01_reset"}
    if prior is not None:
        result["prior_factor"] = prior
    if qualification is not None:
        result["qualification_factor"] = qualification
    if evaluator is not None:
        result["reviewed_evaluator_factor"] = evaluator
    if execution is not None:
        result["execution_factor"] = execution
    if video is not None:
        result["video_instrumentation_factor"] = video
    return result


def validate_migration_plan(checkpoint: Path, current_contract: Mapping[str, Any], plan_path: Path, *,
                            project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    path = Path(plan_path).resolve(strict=True)
    supplied = json.loads(path.read_text(encoding="utf-8"))
    expected = build_migration_plan(checkpoint, current_contract,
                                    allowed_changed_files=supplied.get("allowed_changed_files", []),
                                    reason=supplied.get("reason", ""), project_root=project_root,
                                    prior_evidence=None if "prior_factor" not in supplied else {
                                        role: row["evaluation_manifest"] for role, row in supplied["prior_factor"]["evidence"].items()},
                                    qualification_evidence=None if "qualification_factor" not in supplied else Path(supplied["qualification_factor"]["evaluation_manifest"]),
                                    evaluator_review=None if "reviewed_evaluator_factor" not in supplied else {
                                        "reason": supplied["reviewed_evaluator_factor"]["review_reason"],
                                        "counterexample_tests": supplied["reviewed_evaluator_factor"]["counterexample_tests"]},
                                    execution_evidence=None if "execution_factor" not in supplied else {
                                        "target_num_envs": supplied["execution_factor"]["target_num_envs"],
                                        "vector_smoke": supplied["execution_factor"]["vector_smoke"]["manifest"]},
                                    video_review=None if "video_instrumentation_factor" not in supplied else {
                                        "reason": supplied["video_instrumentation_factor"]["review_reason"]})
    if supplied != expected:
        raise ValueError("migration plan is not exactly bound to this immutable checkpoint and runtime")
    return {"plan_path": str(path), "plan_sha256": file_sha(path), **expected}


VECTOR_FILES = frozenset({
    "src/wlr50_clean/ppo/semantic_vector_backend.py",
    "src/wlr50_clean/ppo/semantic_vector_env.py",
    "src/wlr50_clean/ppo/semantic_vector_training.py",
})

def topology(num_envs: int) -> dict[str, Any]:
    if type(num_envs) is not int or num_envs not in (1, 8):
        raise ValueError("only single or eight-row semantic execution is reviewed")
    return {"schema": "wlr50_clean.semantic_execution_topology.v1",
            "num_envs": num_envs, "observation_dimension": 324, "action_dimension": 12,
            "rollout_decisions_per_env": 128, "reset_sampling": "P01_only",
            "phase_suffix_curriculum_implemented": False,
            "peer_reset": "none" if num_envs == 1 else "synchronous_done_with_gamma_V_actual_final_obs",
            "task_timeout_bootstrap": False, "physical_state_saved": False}

def source_num_envs(metadata: Mapping[str, Any]) -> int:
    declared = metadata.get("execution_topology")
    if declared is not None:
        count = declared.get("num_envs")
        expected = topology(count)
        if metadata.get("semantic_version") == "v3":
            if count != 1:
                raise ValueError("v3 continuation topology must be N1")
            expected = continuation_topology(metadata.get("sampling"),
                                             metadata.get("curriculum_epoch", {}).get("prefix_request"))
        if declared != expected:
            raise ValueError("checkpoint execution topology is malformed")
        return count
    # Legacy semantic checkpoints existed only before vector code was added.
    # Never infer one row for a checkpoint whose runtime already had vector code.
    if any(path in metadata["runtime_contract"].get("files", {}) for path in VECTOR_FILES):
        raise ValueError("vector-capable checkpoint lacks explicit execution topology")
    return 1


def continuation_topology(sampling: str, prefix_request: Mapping[str, Any] | None) -> dict[str, Any]:
    if prefix_request is None:
        if sampling != "P01_full_task_only_initial_version":
            raise ValueError("v3 P01 sampling metadata is malformed")
    elif prefix_request.get("schema") == "wlr50_clean.checkpoint_policy_prefix_request.v1":
        from .semantic_checkpoint_prefix import CheckpointPolicyPrefixRequest, sampling_label
        request = CheckpointPolicyPrefixRequest(**{name: prefix_request[name] for name in (
            "target_phase", "maximum_prefix_decisions", "teacher_offset_decisions")})
        if request.as_dict() != dict(prefix_request) or sampling != sampling_label(request):
            raise ValueError("v3 checkpoint-policy prefix sampling metadata is malformed")
    else:
        from .semantic_prefix import PrefixRequest, sampling_label
        request = PrefixRequest(**{name: prefix_request[name] for name in (
            "target_phase", "maximum_prefix_decisions", "maximum_takeover_decisions", "teacher_offset_decisions")})
        if request.as_dict() != dict(prefix_request) or sampling != sampling_label(request):
            raise ValueError("v3 fixed suffix sampling metadata is malformed")
    return {**topology(1), "schema": "wlr50_clean.semantic_execution_topology.v2",
            "reset_sampling": sampling, "phase_suffix_curriculum_implemented": prefix_request is not None}

def stage_partition(remaining_requested: int) -> dict[str, int]:
    if type(remaining_requested) is not int or remaining_requested < 1:
        raise ValueError("remaining requested budget must be positive")
    vector = remaining_requested // 1024 * 1024
    tail = remaining_requested - vector
    tail_actual = ((tail + 127) // 128) * 128
    return {"N8_requested": vector, "N8_actual": vector, "N8_updates": vector // 1024,
            "N1_requested": tail, "N1_actual": tail_actual, "N1_updates": tail_actual // 128,
            "total_requested": remaining_requested, "total_actual": vector + tail_actual,
            "rounding_overrun": tail_actual - tail}

def verified_vector_smoke(path: Path, contract: Mapping[str, Any], project_root: Path) -> dict[str, Any]:
    path = Path(path).resolve(strict=True)
    if path.name != "vector_smoke_manifest.json" or not path.is_relative_to(
            (project_root / "runs/ppo_semantic_v2/interface_smoke").resolve()):
        raise ValueError("N8 proof must be an isolated live interface-smoke manifest")
    data = json.loads(path.read_text(encoding="utf-8"))
    run_path = path.with_name("run_manifest.json")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    required = ("functional_passed", "all_row_native_audits_verified",
                "one_step_write_capture_verified", "physical_isolation_verified")
    if (data.get("schema") != "wlr50_clean.semantic_vector_smoke.v1"
            or data.get("runtime_contract") != dict(contract) or data.get("num_envs") != 8
            or data.get("explicit_reset_count") != 2
            or data.get("optimizer_steps") != 0
            or any(data.get(key) is not True for key in required)
            or len(data.get("per_row_effect_counts", [])) != 8
            or any(type(n) is not int or n < 1 for n in data["per_row_effect_counts"])
            or run.get("lifecycle") != "SUCCEEDED" or run.get("result") != data):
        raise ValueError("N8 proof lacks current-runtime real reset/native/isolation evidence")
    files = {}
    for name in ("vector_native_audit.jsonl", "vector_physical_probe.jsonl", "gpu_memory.jsonl"):
        source = path.with_name(name)
        if not source.is_file() or source.stat().st_size < 1:
            raise ValueError("N8 proof omitted an actual measured artifact")
        files[name] = {"path": str(source), "sha256": file_sha(source)}
    return {"manifest": str(path), "manifest_sha256": file_sha(path),
            "run_manifest_sha256": file_sha(run_path), "artifacts": files}

def build_execution_factor(metadata, new, evidence, *, project_root: Path):
    if not isinstance(evidence, Mapping) or set(evidence) != {"target_num_envs", "vector_smoke"}:
        raise ValueError("execution boundary requires exact target count and current live N8 proof")
    source, target = source_num_envs(metadata), evidence["target_num_envs"]
    topology(target)
    if (source, target) not in ((1, 8), (8, 1)):
        raise ValueError("execution factor permits only explicit 1-to-8 or 8-to-1 boundaries")
    old = metadata["runtime_contract"]
    added = {path for path in VECTOR_FILES if path not in old["files"]}
    if added not in (set(), set(VECTOR_FILES)) or not VECTOR_FILES <= set(new["files"]):
        raise ValueError("vector execution must be a complete reviewed additive module set")
    if source == 8 and added:
        raise ValueError("an eight-row source checkpoint must already bind every vector module")
    reviewed_files = {}
    for relative in sorted(VECTOR_FILES):
        path = project_root / relative
        if file_sha(path) != new["files"][relative]:
            raise ValueError("vector source bytes disagree with current runtime")
        reviewed_files[relative] = new["files"][relative]
        if relative in old["files"] and old["files"][relative] != new["files"][relative]:
            raise ValueError("topology change cannot silently revise existing vector implementation")
    smoke = verified_vector_smoke(Path(evidence["vector_smoke"]), new, project_root)
    return {"schema": "wlr50_clean.semantic_execution_migration.v1",
            "source_num_envs": source, "target_num_envs": target,
            "reviewed_vector_source_sha256": reviewed_files, "vector_smoke": smoke,
            "target_topology": topology(target), "old_storage_reused": False,
            "restore_actor_critic_Adam_normalizer_RNG_and_global_budget": True,
            "reset": "fresh_legal_P01_all_target_rows", "observation_dimension": 324,
            "policy_action_dimension": 12, "PPO_algorithm_and_hyperparameters_changed": False}
