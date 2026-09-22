"""Explicit, immutable semantic checkpoint version boundaries, without Isaac."""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA = "wlr50_clean.semantic_checkpoint_migration.v1"
REQUEST_HISTORY_KERNEL_SCHEMA = "wlr50_clean.cap_transition_request_history_same372.v1"
REQUEST_HISTORY_KERNEL_FILES = frozenset(f"src/wlr50_clean/ppo/{name}.py" for name in (
    "semantic_history_actor", "semantic_policy_distribution", "semantic_training",
    "semantic_migration", "semantic_cli", "semantic_checkpoint_prefix_policy"))
PHYSICAL_INNOVATION_SIGMA_SCHEMA = "wlr50_clean.FR_knee_phase_physical_innovation_sigma.v1"
PHYSICAL_INNOVATION_SIGMA_FILES = REQUEST_HISTORY_KERNEL_FILES
TASK_CONDITIONED_HIP_WHEEL_SCHEMA = "wlr50_clean.task_conditioned_hip_wheel_same372.v1"
ARCHIVE_ONLY_EXACT_BYTES_SCHEMA = "wlr50_clean.archive_only_exact_bytes_same372.v1"
TRAINING_QUANTITY_BUDGET_SCHEMA = "wlr50_clean.training_quantity_budget_same372.v1"
TASK_CONDITIONED_HIP_WHEEL_FILES = REQUEST_HISTORY_KERNEL_FILES | frozenset({
    "src/wlr50_clean/ppo/semantic_reward.py", "src/wlr50_clean/ppo/semantic_supervisor.py",
    "src/wlr50_clean/ppo/semantic_task_quality.py", "src/wlr50_clean/ppo/semantic_observation.py",
    "src/wlr50_clean/ppo/semantic_video.py", "src/wlr50_clean/ppo/semantic_video_cli.py",
    "scripts/run_semantic_ppo.ps1", "scripts/run_semantic_video.ps1"})
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
ROLE_APPEND_RUNTIME_FILES = frozenset({
    "configs/ppo_semantic_v3/observation_schema.json",
    *(f"src/wlr50_clean/ppo/{name}.py" for name in (
        "semantic_transfer_roles", "semantic_observation", "semantic_policy_distribution",
        "semantic_history_actor", "semantic_checkpoint_prefix_policy", "semantic_checkpoint_prefix",
        "semantic_migration", "semantic_training", "semantic_cli")),
})
SAME372_AUTHORITY_SCHEMA = "wlr50_clean.transfer_roles_same_layout_authority_transition.v1"
ALL_STAGE_SCHEMA = "wlr50_clean.all_stage_same372_acceptance_transition.v1"
FSM_REFERENCE_P09_SCHEMA = "wlr50_clean.fsm_reference_p09_functional_same372_transition.v2"
CAPTURE_HANDOFF_SAME372_SCHEMA = "wlr50_clean.p05_capture_to_handoff_same372_nominal_transition.v1"
CAPTURE_HANDOFF_SOURCE_HEAD = "7db0d17f398d393ce026b6990bd2566d53366407"
TIMING_ONLY_SCHEMA = "wlr50_clean.nominal_timing_same372_continuation.v1"
TIMING_ONLY_SPEC = "configs/ppo_fsm_reference_p09_stable_v2/stage_task_spec.yaml"
TIMING_ONLY_MODE = "source_partial_order_physical_ready_v1"
BODY_REWARD_SCHEMA = "wlr50_clean.functional_carry_body_reward_same372_continuation.v1"
BODY_REWARD_MODE = "current_functional_carry_and_capture_settle_v1"
BODY_REWARD_CODE = "src/wlr50_clean/ppo/semantic_reward.py"
BODY_REWARD_CONFIG = "configs/ppo_fsm_reference_p09_stable_v2/reward_config.yaml"
HEIGHT_RECOVERY_SCHEMA = "wlr50_clean.height_and_p02_recovery_same372_continuation.v1"
HEIGHT_RECOVERY_MODE = "source_segment_reduction_and_live_recovery_v1"
HEIGHT_GEOMETRY_MODE = "contact_aware_bounded_rr_nominal_v3"
HEIGHT_FINAL_STOP_MODE = "current_physical_stop_nominal_owner_v1"
HEIGHT_EXECUTION = "configs/ppo_fsm_reference_p09_stable_v2/execution_profile.yaml"
HEIGHT_NEW_FILES = frozenset(f"src/wlr50_clean/ppo/{name}.py" for name in (
    "semantic_height_recovery", "semantic_height_diagnostics"))
HEIGHT_CODE_FILES = HEIGHT_NEW_FILES | {SUPERVISOR} | frozenset(
    f"src/wlr50_clean/ppo/{name}.py" for name in (
        "semantic_nominal_geometry", "semantic_backend", "semantic_video",
        "semantic_migration", "semantic_training"))
HEIGHT_FILES = HEIGHT_CODE_FILES | {TIMING_ONLY_SPEC, HEIGHT_EXECUTION}
EXPLORATION_TEMPERATURE_SCHEMA = "wlr50_clean.history372_innovation_temperature_continuation.v1"
TASK_FIRST_REWARD_SCHEMA = "wlr50_clean.task_first_reward_same372_continuation.v1"
TASK_RECOVERY_BRANCH_SCHEMA = "wlr50_clean.task_recovery_mean_head_branch.v1"
RR_PHYSICAL_ACCEPTANCE_SCHEMA = "wlr50_clean.rr_physical_acceptance_same372_continuation.v1"
FL_CAPTURE_QUALITY_SCHEMA = "wlr50_clean.fl_capture_quality_same372_continuation.v1"
FL_CAPTURE_QUALITY_FILES = frozenset({
    *(f"src/wlr50_clean/ppo/{name}.py" for name in (
        "semantic_reward", "semantic_supervisor", "semantic_migration", "semantic_training",
        "semantic_cli", "semantic_video_cli", "semantic_video", "semantic_checkpoint_prefix_policy")),
    "scripts/run_semantic_ppo.ps1", "scripts/run_semantic_video.ps1",
})
RR_PHYSICAL_ACCEPTANCE_FILES = frozenset({
    *(f"src/wlr50_clean/ppo/{name}.py" for name in (
        "semantic_supervisor", "semantic_transfer_roles", "semantic_backend",
        "semantic_migration", "semantic_training", "semantic_cli", "semantic_video_cli",
        "semantic_video", "isaac_fsm_backend")),
    "scripts/run_semantic_ppo.ps1", "scripts/run_semantic_video.ps1",
})
FINAL_STOP_HANDOFF_SCHEMA = "wlr50_clean.final_stop_handoff_same372_fix.v1"
FINAL_STOP_HANDOFF_FILES = frozenset({SUPERVISOR, "src/wlr50_clean/ppo/semantic_migration.py",
    "src/wlr50_clean/ppo/semantic_training.py"})
FINAL_STOP_HANDOFF_CORE_FILES = frozenset({SUPERVISOR})
EXECUTION_COMPOSITION_SCHEMA = "wlr50_clean.independent_post_mapper_residual_same372_fix.v1"
EXECUTION_COMPOSITION_FILES = frozenset(f"src/wlr50_clean/ppo/{name}.py" for name in (
    "semantic_residual_adapter", "semantic_nominal_geometry", "semantic_migration",
    "semantic_training", "semantic_cli"))
EXECUTION_COMPOSITION_CORE_FILES = frozenset(f"src/wlr50_clean/ppo/{name}.py" for name in (
    "semantic_residual_adapter", "semantic_nominal_geometry"))

TASK_FIRST_REVIEW_FILES = frozenset({
    *(f"src/wlr50_clean/ppo/{name}.py" for name in (
        "semantic_reward", "semantic_training", "semantic_migration", "semantic_cli", "semantic_video_cli", "semantic_video",
        "semantic_checkpoint_prefix")),
    "scripts/run_semantic_ppo.ps1", "scripts/run_semantic_video.ps1",
    "scripts/initialize_task_recovery_branch.py",
})
EXPLORATION_TEMPERATURE_FILES = frozenset(f"src/wlr50_clean/ppo/{name}.py" for name in (
    "semantic_history_actor", "semantic_policy_distribution", "semantic_training",
    "semantic_cli", "semantic_migration"))
SAME372_AUTHORITY_RUNTIME_FILES = frozenset({
    "configs/ppo_semantic_v3/stage_task_spec.yaml",
    "configs/ppo_semantic_v3/execution_profile.yaml",
    SUPERVISOR,
    *(f"src/wlr50_clean/ppo/{name}.py" for name in (
        "semantic_migration", "semantic_training", "semantic_cli")),
})


def experiment_namespace(semantic_version: str, experiment_id: str | None = None) -> str:
    """Artifact routing only; the experiment keeps the versioned v3 configs."""
    if semantic_version not in ("v2", "v3"):
        raise ValueError("unsupported semantic runtime version")
    if experiment_id is None:
        return f"ppo_semantic_{semantic_version}"
    if experiment_id not in ("transfer_roles_v1", "all_stage_acceptance_v1", "fsm_reference_p09_stable_v2", "task_first_recovery_v1", "non_residual_refine_v1", "residual_rr_fix_v1", "fl_capture_quality_v1", "task_conditioned_hip_wheel_v1") or semantic_version != "v3":
        raise ValueError("isolated semantic experiment requires semantic version v3")
    return f"ppo_{experiment_id}"


def _fsm_reference_p09_same372_transition(before, after, binding, *, metadata, old, new,
                                         config_records, project_root, target_policy_version):
    """One explicit functional-lift/reference boundary, not a generic372 waiver."""
    import yaml
    from .semantic_policy_distribution import (HISTORY_POLICY, policy_contract,
        policy_observation_layout_from_metadata, policy_version_from_metadata)
    from .semantic_transfer_roles import ROLE_OBSERVATION_LAYOUT
    if (old.get("experiment_id") != "all_stage_acceptance_v1"
            or new.get("experiment_id") != "fsm_reference_p09_stable_v2"
            or old.get("semantic_version") != "v3" or new.get("semantic_version") != "v3"
            or policy_version_from_metadata(metadata) != HISTORY_POLICY
            or policy_observation_layout_from_metadata(metadata) != ROLE_OBSERVATION_LAYOUT
            or source_num_envs(metadata) != 1 or target_policy_version is not None):
        raise ValueError("FSM/P09 boundary requires existing all-stage N1 HISTORY372 without a kernel change")
    if (before != after or binding["source_sha256"] != binding["target_sha256"]
            or sum(row["size"] for row in after["feature_groups"]) != 372):
        raise ValueError("FSM/P09 boundary must preserve all372 schema bytes and fixed preprocessing")
    variable = {"files", "source_git_commit", "runtime_content_sha256", "selected_configuration", "experiment_id"}
    if ({k:v for k,v in old.items() if k not in variable}
            != {k:v for k,v in new.items() if k not in variable}):
        raise ValueError("FSM/P09 boundary cannot change other runtime/physical metadata")
    for name in ("action_schema.json", "quality_score.yaml", "reward_config.yaml", "observation_schema.json"):
        if config_records[name]["source_sha256"] != config_records[name]["target_sha256"]:
            raise ValueError(f"FSM/P09 boundary cannot change configuration bytes: {name}")
    for side, contract in (("source", old), ("target", new)):
        selected = contract.get("selected_configuration", {})
        if set(selected) != set(config_records):
            raise ValueError("FSM/P09 requires exactly six selected configuration bindings")
        namespace = "ppo_all_stage_acceptance_v1" if side == "source" else "ppo_fsm_reference_p09_stable_v2"
        for name, row in config_records.items():
            expected = {"path": f"configs/{namespace}/{name}", "sha256": row[f"{side}_sha256"]}
            if (row[f"{side}_path"] != expected["path"] or selected[name] != expected
                    or contract["files"].get(expected["path"]) != expected["sha256"]):
                raise ValueError("FSM/P09 configuration namespace/hash binding differs")
    configs = {}
    for name in ("stage_task_spec.yaml", "execution_profile.yaml"):
        row = config_records[name]
        source = yaml.safe_load(_version_bytes(project_root, old, row["source_path"], prefer_worktree=True))
        target = yaml.safe_load((project_root / row["target_path"]).read_text(encoding="utf-8"))
        if not isinstance(source, dict) or not isinstance(target, dict):
            raise ValueError("FSM/P09 configuration root must be a mapping")
        configs[name] = (source, target)
    source_spec, target_spec = configs["stage_task_spec.yaml"]
    expected_spec = {**source_spec, "revision": "fsm_reference_p09_stable_v2",
                     "p09_lift_semantics": "functional_lift_edge_v2",
                     "reference_nominal_semantics": "successful_fsm_derived_v2"}
    if (target_spec != expected_spec or source_spec.get("physical_acceptance_version") != "all_stage_v1"
            or source_spec.get("rear_leg_order") != "RR_FIRST"
            or "p09_lift_semantics" in source_spec or "reference_nominal_semantics" in source_spec):
        raise ValueError("FSM/P09 task config permits only its revision and two explicit functional-lift/nominal opt-ins")
    source_profile, target_profile = configs["execution_profile.yaml"]
    expected_profile = {**source_profile,
        "revision": "fsm_reference_p09_stable_v2_shared_source_derived_nominal",
        "nominal_geometry_advisory": "functional_rr_preplace_nominal_advisory_v2"}
    if target_profile != expected_profile:
        raise ValueError("FSM/P09 execution profile permits only the reviewed nominal geometry advisory")
    allowed = {SUPERVISOR, "scripts/run_semantic_ppo.ps1",
        *(f"src/wlr50_clean/ppo/{name}.py" for name in (
            "semantic_cli", "semantic_migration", "semantic_training", "semantic_backend",
            "semantic_transfer_roles", "semantic_nominal_geometry", "semantic_prefix")),
        *(row["target_path"] for row in config_records.values())}
    delta = {p for p in old["files"].keys() | new["files"].keys() if old["files"].get(p) != new["files"].get(p)}
    if not delta <= allowed or old["files"].keys() - new["files"].keys():
        raise ValueError(f"FSM/P09 changed non-reviewed runtime files: {sorted(delta - allowed)}")
    for relative in delta:
        if relative in old["files"]:
            _version_bytes(project_root, old, relative, prefer_worktree=True)
        if file_sha(project_root / relative) != new["files"][relative]:
            raise ValueError("FSM/P09 target runtime bytes differ from inventory")
    source_lr = metadata.get("optimizer_learning_rate")
    if type(source_lr) not in (int, float) or not math.isfinite(source_lr) or source_lr <= 0:
        raise ValueError("FSM/P09 requires the recorded finite positive source effective Adam learning rate")
    contract = policy_contract(HISTORY_POLICY, observation_layout=ROLE_OBSERVATION_LAYOUT)
    return {"schema": FSM_REFERENCE_P09_SCHEMA,
        "source_observation_dimension": 372, "target_observation_dimension": 372,
        "source_observation_layout": ROLE_OBSERVATION_LAYOUT, "target_observation_layout": ROLE_OBSERVATION_LAYOUT,
        "source_schema_sha256": binding["source_sha256"], "target_schema_sha256": binding["target_sha256"],
        "source_policy_contract": contract, "target_policy_contract": dict(contract),
        "parameter_mapping": "identity_all_parameters_and_buffers", "observation_bytes_unchanged": True,
        "observation_semantics_changed": ["RR active_lift_history slot represents current valid lift evidence, not permanent historical qualification"],
        "kernel_changed": False, "reward_changed": True, "physical_actuators_changed": False,
        "action_ranges_changed": False, "nominal_and_task_acceptance_changed": True,
        "normalizers": "identity_RSL_state_preserved", "old_rollout_inherited": False,
        "training_rng_preserved": True, "lifetime_counters_and_spent_budgets_preserved": True,
        "optimizer": {"kind": "Adam", "state": "reset_all_moments",
            "initial_learning_rate": source_lr, "learning_rate_policy": "preserve_verified_source_effective_learning_rate"},
        "equivalence_scope": "identical372-input learned function only; changed functional-lift observation/task and successful-FSM-derived nominal, not trajectory equivalence"}


def _capture_handoff_same372_transition(before, after, binding, *, metadata, old, new,
                                       config_records, project_root, target_policy_version):
    """The reviewed FL capture-to-handoff change, not a generic same-MDP waiver."""
    import yaml
    from .semantic_policy_distribution import (HISTORY_POLICY, policy_contract,
        policy_observation_layout_from_metadata, policy_version_from_metadata)
    from .semantic_transfer_roles import ROLE_OBSERVATION_LAYOUT
    experiment = "fsm_reference_p09_stable_v2"
    if (old.get("experiment_id") != experiment or new.get("experiment_id") != experiment
            or old.get("semantic_version") != "v3" or new.get("semantic_version") != "v3"
            or old.get("source_git_commit") != CAPTURE_HANDOFF_SOURCE_HEAD
            or new.get("source_git_commit") == old.get("source_git_commit")
            or policy_version_from_metadata(metadata) != HISTORY_POLICY
            or policy_observation_layout_from_metadata(metadata) != ROLE_OBSERVATION_LAYOUT
            or source_num_envs(metadata) != 1 or target_policy_version is not None):
        raise ValueError("capture handoff requires the reviewed same-experiment source HEAD and N1 HISTORY372")
    if (before != after or binding["source_sha256"] != binding["target_sha256"]
            or sum(row["size"] for row in after["feature_groups"]) != 372):
        raise ValueError("capture handoff must preserve all372 schema bytes and fixed preprocessing")
    variable = {"files", "source_git_commit", "runtime_content_sha256", "selected_configuration"}
    if ({k:v for k,v in old.items() if k not in variable}
            != {k:v for k,v in new.items() if k not in variable}):
        raise ValueError("capture handoff cannot change other runtime/physical metadata")
    for side, contract in (("source", old), ("target", new)):
        selected = contract.get("selected_configuration", {})
        if set(selected) != set(config_records):
            raise ValueError("capture handoff requires exactly six selected configuration bindings")
        for name, row in config_records.items():
            path = f"configs/ppo_fsm_reference_p09_stable_v2/{name}"
            expected = {"path": path, "sha256": row[f"{side}_sha256"]}
            if (row[f"{side}_path"] != path or selected[name] != expected
                    or contract["files"].get(path) != expected["sha256"]):
                raise ValueError("capture handoff configuration namespace/hash binding differs")
            if name != "stage_task_spec.yaml" and row["source_sha256"] != row["target_sha256"]:
                raise ValueError(f"capture handoff cannot change configuration bytes: {name}")
    spec_row = config_records["stage_task_spec.yaml"]
    source_spec = yaml.safe_load(_version_bytes(project_root, old, spec_row["source_path"], prefer_worktree=True))
    target_spec = yaml.safe_load((project_root / spec_row["target_path"]).read_bytes())
    if (not isinstance(source_spec, dict) or not isinstance(target_spec, dict)
            or not isinstance(source_spec.get("nominal"), dict)
            or source_spec["nominal"].get("p05_pending_capture") != "current_FL_capture_wheel_continuation_v1"):
        raise ValueError("capture handoff source must have the exact v1 nominal capture opt-in")
    expected_spec = json.loads(json.dumps(source_spec, allow_nan=False))
    expected_spec["nominal"]["p05_pending_capture"] = "current_FL_capture_wheel_continuation_to_handoff_v2"
    if target_spec != expected_spec:
        raise ValueError("capture handoff permits only the exact nominal p05_pending_capture v1-to-v2 change")
    if old["files"].keys() != new["files"].keys():
        raise ValueError("capture handoff cannot add or remove runtime files")
    delta = {p for p in old["files"] if old["files"][p] != new["files"][p]}
    required = {SUPERVISOR, spec_row["target_path"]}
    allowed = required | {f"src/wlr50_clean/ppo/{name}.py" for name in
                          ("semantic_migration", "semantic_training", "semantic_cli")}
    if not required <= delta or not delta <= allowed:
        raise ValueError(f"capture handoff changed non-reviewed runtime files: {sorted(delta - allowed)}")
    for relative in delta:
        _version_bytes(project_root, old, relative, prefer_worktree=True)
        if file_sha(project_root / relative) != new["files"][relative]:
            raise ValueError("capture handoff target runtime bytes differ from inventory")
    old_regions = _nominal_provider_byte_regions(_version_bytes(project_root, old, SUPERVISOR, prefer_worktree=True))
    new_regions = _nominal_provider_byte_regions((project_root / SUPERVISOR).read_bytes())
    if old_regions[0] != new_regions[0] or old_regions[2] != new_regions[2] or old_regions[1] == new_regions[1]:
        raise ValueError("capture handoff may change only the NominalMotionProvider class bytes")
    source_lr = metadata.get("optimizer_learning_rate")
    if type(source_lr) not in (int, float) or not math.isfinite(source_lr) or source_lr <= 0:
        raise ValueError("capture handoff requires the verified positive source effective Adam learning rate")
    policy = policy_contract(HISTORY_POLICY, observation_layout=ROLE_OBSERVATION_LAYOUT)
    return {"schema": CAPTURE_HANDOFF_SAME372_SCHEMA,
        "source_observation_dimension": 372, "target_observation_dimension": 372,
        "source_observation_layout": ROLE_OBSERVATION_LAYOUT, "target_observation_layout": ROLE_OBSERVATION_LAYOUT,
        "source_schema_sha256": binding["source_sha256"], "target_schema_sha256": binding["target_sha256"],
        "source_policy_contract": policy, "target_policy_contract": dict(policy),
        "parameter_mapping": "identity_all_parameters_and_buffers", "observation_bytes_unchanged": True,
        "observation_semantics_changed": [], "kernel_changed": False, "reward_changed": False,
        "physical_mdp_changed": True, "nominal_control_changed": True, "task_acceptance_changed": False,
        "physical_actuators_changed": False, "action_ranges_changed": False,
        "nominal_provider": {"file": SUPERVISOR,
            "source_class_sha256": hashlib.sha256(old_regions[1]).hexdigest(),
            "target_class_sha256": hashlib.sha256(new_regions[1]).hexdigest(),
            "outside_class_sha256": hashlib.sha256(old_regions[0] + b"\0" + old_regions[2]).hexdigest(),
            "outside_class_bytes_identical": True},
        "normalizers": "identity_RSL_state_preserved", "old_rollout_inherited": False,
        "physical_state_inherited": False, "training_rng_preserved": True,
        "lifetime_counters_and_spent_budgets_preserved": True,
        "optimizer": {"kind": "Adam", "state": "reset_all_moments",
            "initial_learning_rate": source_lr, "learning_rate_policy": "preserve_verified_source_effective_learning_rate",
            "preserve_source_group_options": True},
        "equivalence_scope": "identical372-input learned policy/value function only; changed FL nominal capture handoff, not projected actions or physical trajectory equivalence"}


def _all_stage_same372_transition(before, after, binding, *, metadata, old, new,
                                  config_records, project_root, target_policy_version):
    """Explicit acceptance/measurement boundary; never a generic resume waiver."""
    import yaml
    from .semantic_policy_distribution import (HISTORY_POLICY, policy_contract,
        policy_observation_layout_from_metadata, policy_version_from_metadata)
    from .semantic_transfer_roles import ROLE_OBSERVATION_LAYOUT
    if (old.get("experiment_id") != "transfer_roles_v1"
            or new.get("experiment_id") != "all_stage_acceptance_v1"
            or old.get("semantic_version") != "v3" or new.get("semantic_version") != "v3"
            or policy_version_from_metadata(metadata) != HISTORY_POLICY
            or policy_observation_layout_from_metadata(metadata) != ROLE_OBSERVATION_LAYOUT
            or source_num_envs(metadata) != 1 or target_policy_version is not None):
        raise ValueError("all-stage boundary requires existing N1 HISTORY372 transfer-role policy")
    if (before != after or binding["source_sha256"] != binding["target_sha256"]
            or sum(row["size"] for row in after["feature_groups"]) != 372):
        raise ValueError("all-stage boundary must preserve all372 schema bytes and fixed preprocessing")
    variable = {"files", "source_git_commit", "runtime_content_sha256", "selected_configuration", "experiment_id"}
    if ({k:v for k,v in old.items() if k not in variable}
            != {k:v for k,v in new.items() if k not in variable}):
        raise ValueError("all-stage boundary cannot change other runtime/physical metadata")
    for name in ("action_schema.json", "quality_score.yaml", "reward_config.yaml"):
        if config_records[name]["source_sha256"] != config_records[name]["target_sha256"]:
            raise ValueError(f"all-stage boundary cannot silently change {name}")
    for side, contract in (("source", old), ("target", new)):
        selected = contract.get("selected_configuration", {})
        if set(selected) != set(config_records):
            raise ValueError("all-stage requires exactly six selected configuration bindings")
        for name, row in config_records.items():
            namespace = "ppo_semantic_v3" if side == "source" else "ppo_all_stage_acceptance_v1"
            if row[f"{side}_path"] != f"configs/{namespace}/{name}":
                raise ValueError("all-stage configuration must use its exact isolated namespace")
            expected = {"path": row[f"{side}_path"], "sha256": row[f"{side}_sha256"]}
            if selected[name] != expected or contract["files"].get(expected["path"]) != expected["sha256"]:
                raise ValueError("all-stage configuration binding differs")
    target_spec = yaml.safe_load((project_root / config_records["stage_task_spec.yaml"]["target_path"]).read_text(encoding="utf-8"))
    if target_spec.get("physical_acceptance_version") != "all_stage_v1" or target_spec.get("rear_leg_order") != "RR_FIRST":
        raise ValueError("all-stage requires explicit physical acceptance version and RR_FIRST")
    # Assets, scene, actuation physics, frozen FSM and all unrelated source bytes
    # remain exact. The explicit set permits measurement fixes, not dynamics edits.
    allowed = {SUPERVISOR, "scripts/run_semantic_ppo.ps1",
        *(f"src/wlr50_clean/ppo/{name}.py" for name in (
            "semantic_cli", "semantic_migration", "semantic_training", "semantic_backend",
            "semantic_transfer_roles", "semantic_legacy_evaluation", "semantic_physical_sensing")),
        *(row["target_path"] for row in config_records.values())}
    delta = {p for p in old["files"].keys() | new["files"].keys() if old["files"].get(p) != new["files"].get(p)}
    if not delta <= allowed or old["files"].keys() - new["files"].keys():
        raise ValueError(f"all-stage changed non-reviewed runtime files: {sorted(delta - allowed)}")
    for relative in delta:
        if relative in old["files"]:
            _version_bytes(project_root, old, relative, prefer_worktree=True)
        if file_sha(project_root / relative) != new["files"][relative]:
            raise ValueError("all-stage target runtime bytes differ from inventory")
    contract = policy_contract(HISTORY_POLICY, observation_layout=ROLE_OBSERVATION_LAYOUT)
    return {"schema": ALL_STAGE_SCHEMA,
        "source_observation_dimension": 372, "target_observation_dimension": 372,
        "source_observation_layout": ROLE_OBSERVATION_LAYOUT, "target_observation_layout": ROLE_OBSERVATION_LAYOUT,
        "source_schema_sha256": binding["source_sha256"], "target_schema_sha256": binding["target_sha256"],
        "source_policy_contract": contract, "target_policy_contract": dict(contract),
        "parameter_mapping": "identity_all_parameters_and_buffers", "observation_bytes_unchanged": True,
        "observation_semantics_changed": ["continued_response_fraction counts actual transfer maturity",
            "P13 phase_progress reversibly encodes traversal event, fixed post timer and region-loss latch",
            "global physical Phi finish uses current body rectangle, measured rates and post observation"],
        "kernel_changed": False, "reward_changed": True, "physical_actuators_changed": False,
        "effective_action_mapping_changed": True, "measurement_and_task_acceptance_changed": True,
        "normalizers": "identity_RSL_state_preserved", "old_rollout_inherited": False,
        "training_rng_preserved": True, "lifetime_counters_and_spent_budgets_preserved": True,
        "optimizer": {"kind": "Adam", "state": "reset_all_moments", "initial_learning_rate": 3e-5},
        "equivalence_scope": "identical372-input learned function only; changed sensing/task/PBRS/nominal/residual mapping, not trajectory equivalence"}


def _transfer_observation_scale_transition(before: Mapping[str, Any], after: Mapping[str, Any],
                                           binding: Mapping[str, Any]) -> dict[str, Any]:
    """The sole reviewed encoder change; all other preprocessing stays exact."""
    normalized_before = json.loads(json.dumps(before, allow_nan=False))
    groups = ("previous_residual_full12", "previous_previous_residual_full12")
    columns, source_scales, target_scales = [], [], []
    offset = 0
    for old_group, new_group in zip(normalized_before.get("feature_groups", ()), after.get("feature_groups", ())):
        if old_group.get("name") in groups:
            if (old_group.get("size") != 12 or new_group.get("name") != old_group["name"]
                    or new_group.get("size") != 12
                    or not isinstance(old_group.get("scale"), list) or len(old_group["scale"]) != 12
                    or not isinstance(new_group.get("scale"), list) or len(new_group["scale"]) != 12):
                raise ValueError("transfer-role observation scale group layout differs")
            previous, target = old_group["scale"][3], new_group["scale"][3]
            if type(previous) not in (int, float) or previous not in (4, 6) or type(target) not in (int, float) or target != 6:
                raise ValueError("transfer-role observation scales allow only 4-to-6 or already-6")
            columns.append(offset + 3)
            source_scales.append(previous)
            target_scales.append(target)
            old_group["scale"][3] = target
        offset += old_group.get("size", 0)
    if columns != [210, 222] or source_scales not in ([4, 4], [6, 6]):
        raise ValueError("transfer-role observation scale columns must be exactly 210 and 222")
    for key in ("feature_groups", "clip", "maximum_task_duration_s", "fixed_chassis_to_body_wxyz",
                "level_reference", "normalization"):
        if normalized_before.get(key) != after.get(key):
            raise ValueError(f"warm-start actor observation preprocessing changed outside reviewed scales: {key}")
    return {
        "schema": "wlr50_clean.transfer_roles_observation_scale_transition.v1",
        "experiment_id": "transfer_roles_v1", "observation_dimension": 324,
        "columns": columns, "feature_groups": list(groups), "channel_index": 3,
        "source_scales": source_scales, "target_scales": target_scales,
        "factors": [target / source for source, target in zip(source_scales, target_scales)],
        "changed": source_scales != target_scales,
        "source_schema_sha256": binding["source_sha256"], "target_schema_sha256": binding["target_sha256"],
        "first_layer_compensation": "multiply_actor_and_critic_input_columns_by_target_over_source",
        "equivalence_scope": "unclipped source inputs only; previously clipped history can expose new information",
    }


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _transfer_observation_append_transition(before: Mapping[str, Any], after: Mapping[str, Any],
                                            binding: Mapping[str, Any], *, metadata: Mapping[str, Any],
                                            old: Mapping[str, Any], new: Mapping[str, Any],
                                            config_records: Mapping[str, Any], project_root: Path,
                                            target_policy_version: str | None) -> dict[str, Any]:
    """Only the reviewed HISTORY324 -> HISTORY372 observation boundary.

    This validates the immutable mapping recipe, not any tensor or simulator
    state. The loader must verify the source first, then zero-pad exactly the
    two first-layer weights and prove all remaining learned state is retained.
    """
    from .semantic_policy_distribution import (
        HISTORY_POLICY, policy_contract, policy_observation_layout_from_metadata,
        policy_version_from_metadata,
    )
    from .semantic_transfer_roles import (
        LEGS, ROLE_OBSERVATION_LAYOUT, ROLE_OBSERVATION_GROUP, ROLE_OBSERVATION_FIELDS,
        ROLE_OBSERVATION_BASE_DIM, ROLE_OBSERVATION_DIM,
    )
    if (old.get("semantic_version") != "v3" or new.get("semantic_version") != "v3"
            or old.get("experiment_id") != "transfer_roles_v1"
            or new.get("experiment_id") != "transfer_roles_v1"
            or policy_version_from_metadata(metadata) != HISTORY_POLICY
            or policy_observation_layout_from_metadata(metadata) is not None
            or source_num_envs(metadata) != 1 or target_policy_version is not None):
        raise ValueError("role observation append requires transfer_roles_v1 N1 HISTORY324 without a simultaneous kernel transition")
    marker = "transfer_role_features_version"
    if marker in before or after.get(marker) != ROLE_OBSERVATION_LAYOUT:
        raise ValueError("role observation append requires the exact new layout marker and an unmarked source")
    source_groups, target_groups = before.get("feature_groups"), after.get("feature_groups")
    if (not isinstance(source_groups, list) or not isinstance(target_groups, list)
            or any(not isinstance(g, dict) or type(g.get("size")) is not int or g["size"] <= 0
                   for g in source_groups + target_groups)
            or sum(g["size"] for g in source_groups) != ROLE_OBSERVATION_BASE_DIM
            or sum(g["size"] for g in target_groups) != ROLE_OBSERVATION_DIM
            or len(target_groups) != len(source_groups) + 1
            or digest(target_groups[:-1]) != digest(source_groups)
            or digest(target_groups[-1]) != digest({"name": ROLE_OBSERVATION_GROUP,
                "size": ROLE_OBSERVATION_DIM - ROLE_OBSERVATION_BASE_DIM, "scale": 1.0})
            or any(g.get("name") == ROLE_OBSERVATION_GROUP for g in source_groups)):
        raise ValueError("role observation append must preserve all source324 groups and append only the exact48 group")
    # Revision is descriptive. Every other top-level preprocessing/deployability
    # field is preserved, including unknown future fields (fail closed).
    source_fixed = {k: v for k, v in before.items() if k != "revision"}
    target_fixed = {k: v for k, v in after.items() if k not in ("revision", marker)}
    target_fixed["feature_groups"] = target_groups[:-1]
    if digest(source_fixed) != digest(target_fixed):
        raise ValueError("role observation append cannot change existing observation preprocessing")
    for name, row in config_records.items():
        if name != "observation_schema.json" and row["source_sha256"] != row["target_sha256"]:
            raise ValueError(f"role observation append cannot change task/action/reward/quality configuration: {name}")
    # Actual v3 contracts bind the selected config twice: inventory and named
    # path/SHA records. Validate both copies before normalizing ONLY the one
    # reviewed observation SHA; do not exempt the entire selection mapping.
    old_selected, new_selected = old.get("selected_configuration"), new.get("selected_configuration")
    for selected, contract, side in ((old_selected, old, "source"), (new_selected, new, "target")):
        if not isinstance(selected, Mapping) or set(selected) != set(config_records):
            raise ValueError("role observation append selected_configuration must contain exactly the six bound configurations")
        for name, row in config_records.items():
            expected = {"path": row[f"{side}_path"], "sha256": row[f"{side}_sha256"]}
            if (digest(selected[name]) != digest(expected)
                    or contract["files"].get(expected["path"]) != expected["sha256"]):
                raise ValueError(f"role observation append selected_configuration {side} binding differs: {name}")
    variable = {"files", "source_git_commit", "runtime_content_sha256"}
    target_fixed_metadata = {k: v for k, v in new.items() if k not in variable}
    target_fixed_metadata["selected_configuration"] = dict(new_selected)
    target_fixed_metadata["selected_configuration"]["observation_schema.json"] = old_selected["observation_schema.json"]
    if digest({k: v for k, v in old.items() if k not in variable}) != digest(target_fixed_metadata):
        raise ValueError("role observation append cannot change physical/runtime metadata")
    if set(old["files"]) != set(new["files"]):
        raise ValueError("role observation append cannot add or remove runtime files")
    delta = {p for p in old["files"] if old["files"][p] != new["files"][p]}
    if not delta <= ROLE_APPEND_RUNTIME_FILES:
        raise ValueError("role observation append changed a non-whitelisted runtime file")
    for relative in delta:
        _version_bytes(project_root, dict(old), relative, prefer_worktree=True)
        if file_sha(project_root / relative) != new["files"][relative]:
            raise ValueError("role observation append target bytes differ from runtime inventory")
    return {
        "schema": "wlr50_clean.transfer_roles_observation_append_transition.v1",
        "experiment_id": "transfer_roles_v1", "changed": True,
        "source_observation_dimension": ROLE_OBSERVATION_BASE_DIM,
        "target_observation_dimension": ROLE_OBSERVATION_DIM,
        "source_observation_layout": None, "target_observation_layout": ROLE_OBSERVATION_LAYOUT,
        "source_schema_sha256": binding["source_sha256"], "target_schema_sha256": binding["target_sha256"],
        "source_policy_contract": policy_contract(HISTORY_POLICY),
        "target_policy_contract": policy_contract(HISTORY_POLICY, observation_layout=ROLE_OBSERVATION_LAYOUT),
        "feature_group": ROLE_OBSERVATION_GROUP, "leg_order": list(LEGS),
        "fields_per_leg": list(ROLE_OBSERVATION_FIELDS),
        "first_layer_parameter": "mlp.0.weight",
        "appended_columns": [ROLE_OBSERVATION_BASE_DIM, ROLE_OBSERVATION_DIM],
        "first_layer_mapping": "append_zero_actor_and_critic_first_layer",
        "existing_columns_preserved": True, "all_other_parameters_and_buffers_preserved": True,
        "learned_std_preserved": True, "normalizers": "identity_RSL_state_preserved",
        "optimizer": {"kind": "Adam", "state": "reset_all_moments", "initial_learning_rate": 3e-5},
        "training_rng_preserved": True, "lifetime_counters_and_spent_budgets_preserved": True,
        "old_rollout_inherited": False, "physical_state_inherited": False,
        "kernel_changed": False, "reward_changed": False, "physical_dynamics_changed": False,
        "equivalence_scope": "initial zero-appended-column function only; dimension-dependent GEMM rounding may differ; not physical trajectory equivalence",
    }


def _nominal_provider_byte_regions(raw: bytes) -> tuple[bytes, bytes, bytes]:
    """Exclude exactly the one top-level class, retaining outside bytes."""
    import ast
    try:
        tree = ast.parse(raw.decode("utf-8"))
    except (SyntaxError, UnicodeDecodeError) as exc:
        raise ValueError("same372 nominal-provider source is not valid UTF-8 Python") from exc
    matches = [node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == "NominalMotionProvider"]
    if len(matches) != 1 or matches[0].decorator_list:
        raise ValueError("same372 requires exactly one undecorated NominalMotionProvider class")
    node = matches[0]
    lines = raw.splitlines(keepends=True)
    return (b"".join(lines[:node.lineno-1]), b"".join(lines[node.lineno-1:node.end_lineno]),
            b"".join(lines[node.end_lineno:]))


def _transfer_same372_authority_transition(before, after, binding, *, metadata, old, new,
                                           config_records, project_root, target_policy_version):
    """Only the reviewed FL-hip advisory/cap change; no generic372 waiver."""
    import yaml
    from .semantic_observation import load_semantic_observation_schema
    from .semantic_policy_distribution import (
        HISTORY_POLICY, policy_contract, policy_observation_layout_from_metadata,
        policy_version_from_metadata,
    )
    from .semantic_transfer_roles import ROLE_OBSERVATION_LAYOUT, ROLE_OBSERVATION_DIM
    if (old.get("semantic_version") != "v3" or new.get("semantic_version") != "v3"
            or old.get("experiment_id") != "transfer_roles_v1"
            or new.get("experiment_id") != "transfer_roles_v1"
            or policy_version_from_metadata(metadata) != HISTORY_POLICY
            or policy_observation_layout_from_metadata(metadata) != ROLE_OBSERVATION_LAYOUT
            or source_num_envs(metadata) != 1 or target_policy_version is not None):
        raise ValueError("same372 authority requires transfer_roles_v1 N1 HISTORY372 with no kernel transition")
    schema = load_semantic_observation_schema(project_root / binding["target_path"])
    if (schema.dimension != ROLE_OBSERVATION_DIM
            or schema.transfer_role_features_version != ROLE_OBSERVATION_LAYOUT
            or before.get("transfer_role_features_version") != ROLE_OBSERVATION_LAYOUT
            or digest(before) != digest(after) or binding["source_sha256"] != binding["target_sha256"]):
        raise ValueError("same372 authority must preserve the complete observation schema bytes and layout")
    changed_configs = {"stage_task_spec.yaml", "execution_profile.yaml"}
    for name, row in config_records.items():
        if name not in changed_configs and row["source_sha256"] != row["target_sha256"]:
            raise ValueError(f"same372 authority cannot change observation/reward/action/quality bytes: {name}")
    source_cfg, target_cfg = {}, {}
    for name in changed_configs:
        row = config_records[name]
        source_cfg[name] = yaml.safe_load(_version_bytes(project_root, old, row["source_path"], prefer_worktree=True))
        target_cfg[name] = yaml.safe_load((project_root / row["target_path"]).read_bytes())
        if not isinstance(source_cfg[name], dict) or not isinstance(target_cfg[name], dict):
            raise ValueError("same372 authority requires mapping configuration roots")
    phases = tuple(f"P{i:02}" for i in range(6, 14))
    overrides = {phase: {"front_left_hip": 60.0} for phase in phases}
    spec = json.loads(json.dumps(source_cfg["stage_task_spec.yaml"], allow_nan=False))
    nominal = spec.get("nominal", {})
    if (spec.get("revision") != "diagonal_transfer_roles_v1"
            or type(nominal.get("servo_handoff_rate_deg_s")) not in (int, float)
            or nominal["servo_handoff_rate_deg_s"] != 150.0
            or "phase_servo_rate_overrides_deg_s" in nominal):
        raise ValueError("same372 authority source nominal revision/rates are not the reviewed baseline")
    spec["revision"] = "diagonal_transfer_roles_v2_fl_advisory_counterauthority"
    nominal["phase_servo_rate_overrides_deg_s"] = overrides
    if digest(spec) != digest(target_cfg["stage_task_spec.yaml"]):
        raise ValueError("same372 authority permits only the exact P06-P13 FL-hip60 override and stage revision")
    profile = json.loads(json.dumps(source_cfg["execution_profile.yaml"], allow_nan=False))
    residual = profile.get("residual", {})
    if (profile.get("revision") != "continuous_transfer_roles_v1_residual_authority"
            or type(residual.get("servo_rate_deg_s")) not in (int, float)
            or residual["servo_rate_deg_s"] != 60.0):
        raise ValueError("same372 authority source residual slew/revision differs")
    for phase in phases:
        caps = residual.get("phase_caps_full12", {}).get(phase)
        if (not isinstance(caps, list) or len(caps) != 12
                or type(caps[0]) not in (int, float) or caps[0] != 24):
            raise ValueError("same372 authority requires original FL-hip24 caps in P06-P13")
        caps[0] = 32
    profile["revision"] = "continuous_transfer_roles_v2_fl_advisory_counterauthority"
    if digest(profile) != digest(target_cfg["execution_profile.yaml"]):
        raise ValueError("same372 authority permits only FL-hip24-to32 P06-P13 caps and execution revision")
    old_selected, new_selected = old.get("selected_configuration"), new.get("selected_configuration")
    for selected, contract, side in ((old_selected, old, "source"), (new_selected, new, "target")):
        if not isinstance(selected, Mapping) or set(selected) != set(config_records):
            raise ValueError("same372 selected_configuration requires exactly the six bound configs")
        for name, row in config_records.items():
            expected = {"path": row[f"{side}_path"], "sha256": row[f"{side}_sha256"]}
            if digest(selected[name]) != digest(expected) or contract["files"].get(expected["path"]) != expected["sha256"]:
                raise ValueError(f"same372 selected_configuration binding differs: {side}/{name}")
    variable = {"files", "source_git_commit", "runtime_content_sha256"}
    old_fixed = {k: v for k, v in old.items() if k not in variable}
    new_fixed = {k: v for k, v in new.items() if k not in variable}
    new_fixed["selected_configuration"] = dict(new_selected)
    for name in changed_configs:
        new_fixed["selected_configuration"][name] = old_selected[name]
    if digest(old_fixed) != digest(new_fixed):
        raise ValueError("same372 authority cannot change physical or other runtime metadata")
    if set(old["files"]) != set(new["files"]):
        raise ValueError("same372 authority cannot add/remove runtime files")
    delta = {path for path in old["files"] if old["files"][path] != new["files"][path]}
    if not delta <= SAME372_AUTHORITY_RUNTIME_FILES or SUPERVISOR not in delta:
        raise ValueError("same372 authority requires only its reviewed runtime files and changed nominal provider")
    for relative in delta:
        _version_bytes(project_root, old, relative, prefer_worktree=True)
        if file_sha(project_root / relative) != new["files"][relative]:
            raise ValueError("same372 authority target runtime bytes differ")
    provider_before = _version_bytes(project_root, old, SUPERVISOR, prefer_worktree=True)
    provider_after = (project_root / SUPERVISOR).read_bytes()
    left, source_class, right = _nominal_provider_byte_regions(provider_before)
    target_left, target_class, target_right = _nominal_provider_byte_regions(provider_after)
    if left != target_left or right != target_right or source_class == target_class:
        raise ValueError("same372 authority may change only the NominalMotionProvider class bytes")
    contract = policy_contract(HISTORY_POLICY, observation_layout=ROLE_OBSERVATION_LAYOUT)
    return {
        "schema": SAME372_AUTHORITY_SCHEMA,
        "source_observation_dimension": ROLE_OBSERVATION_DIM, "target_observation_dimension": ROLE_OBSERVATION_DIM,
        "source_observation_layout": ROLE_OBSERVATION_LAYOUT, "target_observation_layout": ROLE_OBSERVATION_LAYOUT,
        "source_schema_sha256": binding["source_sha256"], "target_schema_sha256": binding["target_sha256"],
        "source_policy_contract": contract, "target_policy_contract": dict(contract),
        "parameter_mapping": "identity_all_parameters_and_buffers",
        "observation_bytes_unchanged": True, "kernel_changed": False, "reward_changed": False,
        "physical_actuators_changed": False, "effective_action_mapping_changed": True,
        "authority_changes": {"nominal.phase_servo_rate_overrides_deg_s": overrides,
            "FL_hip_phase_caps": {"phases": list(phases), "channel": 0, "source": 24, "target": 32}},
        "nominal_provider": {"file": SUPERVISOR,
            "source_class_sha256": hashlib.sha256(source_class).hexdigest(),
            "target_class_sha256": hashlib.sha256(target_class).hexdigest(),
            "outside_class_sha256": hashlib.sha256(left + b"\0" + right).hexdigest(),
            "outside_class_bytes_identical": True},
        "normalizers": "identity_RSL_state_preserved",
        "optimizer": {"kind": "Adam", "state": "reset_all_moments", "initial_learning_rate": 3e-5},
        "training_rng_preserved": True, "lifetime_counters_and_spent_budgets_preserved": True,
        "old_rollout_inherited": False, "physical_state_inherited": False,
        "equivalence_scope": "same-input learned policy/value function only; not projected-action, nominal timing or trajectory equivalence",
    }


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
                                project_root: Path = PROJECT_ROOT,
                                target_policy_version: str | None = None) -> dict[str, Any]:
    """Explicit new-MDP boundary; never a relaxation of v2 exact-resume rules."""
    metadata = checkpoint_metadata(checkpoint)
    from .semantic_policy_distribution import (
        HISTORY_POLICY, STATE_DEPENDENT_POLICY, policy_contract, policy_version_from_metadata,
    )
    from .semantic_return_profile import (
        LEGACY_RETURN_PROFILE, RETURN_PROFILE, reward_return_profile, runner_return_profile,
    )
    from .semantic_reward import load_semantic_reward_config
    import yaml
    # Do not reinterpret source actor metadata using the target factory's new
    # discount. Validate its complete, explicitly recognized historical config.
    source_policy_version = policy_version_from_metadata(metadata)
    old, new = _contract(metadata["runtime_contract"]), _contract(current_contract)
    source_version = metadata.get("semantic_version", "v2")
    if source_version not in ("v2", "v3") or new.get("semantic_version") != "v3":
        raise ValueError("new-MDP warm start requires a v2 or v3 source checkpoint and a v3 target")
    if old.get("semantic_version", "v2") != source_version:
        raise ValueError("new-MDP source semantic version differs from its runtime contract")
    source_experiment, target_experiment = old.get("experiment_id"), new.get("experiment_id")
    source_namespace = experiment_namespace(source_version, source_experiment)
    target_namespace = experiment_namespace("v3", target_experiment)
    if target_experiment == "transfer_roles_v1" and source_version != "v3":
        raise ValueError("transfer_roles_v1 warm start requires an existing v3 checkpoint")
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
        source = old.get("selected_configuration", {}).get(name, {}).get("path", f"configs/ppo_semantic_{source_version}/{name}")
        target = new.get("selected_configuration", {}).get(name, {}).get("path", f"configs/ppo_semantic_v3/{name}")
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
    # preprocessing. Only the explicit transfer-role experiment has a reviewed
    # two-column scale compensation; historical/default paths remain exact.
    before = json.loads(_version_text(project_root, old, config_records["observation_schema.json"]["source_path"], prefer_worktree=True))
    after = json.loads((project_root / config_records["observation_schema.json"]["target_path"]).read_text(encoding="utf-8"))
    observation_transition = None
    observation_append_transition = None
    observation_same_layout_transition = None
    if target_experiment == "fsm_reference_p09_stable_v2":
        transition_builder = (_capture_handoff_same372_transition
            if source_experiment == "fsm_reference_p09_stable_v2" else _fsm_reference_p09_same372_transition)
        observation_same_layout_transition = transition_builder(
            before, after, config_records["observation_schema.json"], metadata=metadata,
            old=old, new=new, config_records=config_records, project_root=project_root,
            target_policy_version=target_policy_version)
    elif target_experiment == "all_stage_acceptance_v1":
        observation_same_layout_transition = _all_stage_same372_transition(
            before, after, config_records["observation_schema.json"], metadata=metadata,
            old=old, new=new, config_records=config_records, project_root=project_root,
            target_policy_version=target_policy_version)
    elif "transfer_role_features_version" in before and "transfer_role_features_version" in after:
        observation_same_layout_transition = _transfer_same372_authority_transition(
            before, after, config_records["observation_schema.json"], metadata=metadata,
            old=old, new=new, config_records=config_records, project_root=project_root,
            target_policy_version=target_policy_version)
    elif "transfer_role_features_version" in after:
        observation_append_transition = _transfer_observation_append_transition(
            before, after, config_records["observation_schema.json"], metadata=metadata,
            old=old, new=new, config_records=config_records, project_root=project_root,
            target_policy_version=target_policy_version)
    elif target_experiment == "transfer_roles_v1":
        observation_transition = _transfer_observation_scale_transition(
            before, after, config_records["observation_schema.json"])
    else:
        for key in ("feature_groups", "clip", "maximum_task_duration_s", "fixed_chassis_to_body_wxyz",
                    "level_reference", "normalization"):
            if before.get(key) != after.get(key):
                raise ValueError(f"warm-start actor observation preprocessing changed: {key}")
    source_dimension = 324 if observation_same_layout_transition is None else 372
    if sum(group["size"] for group in before.get("feature_groups", ())) != source_dimension:
        raise ValueError("warm start source dimension differs from its explicit reviewed transition")
    # Verify the comparison's historical source before AppLauncher/reset; defer
    # only immutable materialization, not source availability, to publication.
    _version_bytes(project_root, old, config_records["execution_profile.yaml"]["source_path"], prefer_worktree=True)
    kernel_transition = None
    if target_policy_version is not None:
        if (source_version != "v3" or source_policy_version != STATE_DEPENDENT_POLICY
                or target_policy_version != HISTORY_POLICY or source_num_envs(metadata) != 1):
            raise ValueError("history-kernel migration requires the existing v3 N1 heteroscedastic source")
        # One factor only: do not hide an action/reward/physics change in a
        # distribution migration. General new-MDP continuation remains separate.
        if any(row["source_sha256"] != row["target_sha256"] for row in config_records.values()):
            raise ValueError("history-kernel migration cannot change any physical/reward/observation configuration")
        variable = {"files", "source_git_commit", "runtime_content_sha256"}
        if {k: v for k, v in old.items() if k not in variable} != {k: v for k, v in new.items() if k not in variable}:
            raise ValueError("history-kernel migration cannot change runtime or physical metadata")
        module = "src/wlr50_clean/ppo/semantic_history_actor.py"
        allowed = {module, "src/wlr50_clean/ppo/semantic_policy_distribution.py",
                   "src/wlr50_clean/ppo/semantic_training.py", "src/wlr50_clean/ppo/semantic_migration.py",
                   "src/wlr50_clean/ppo/semantic_cli.py", "scripts/run_semantic_ppo.ps1",
                   "src/wlr50_clean/ppo/semantic_checkpoint_prefix.py",
                   "src/wlr50_clean/ppo/semantic_checkpoint_prefix_policy.py"}
        delta = {p for p in old["files"].keys() | new["files"].keys()
                 if old["files"].get(p) != new["files"].get(p)}
        if (not delta <= allowed or module not in new["files"]
                or old["files"].keys() - new["files"].keys()
                or new["files"].keys() - old["files"].keys() - {module}):
            raise ValueError("history-kernel migration changed a non-whitelisted runtime file")
        for relative in delta:
            if relative in old["files"]:
                _version_bytes(project_root, old, relative, prefer_worktree=True)
            if file_sha(project_root / relative) != new["files"][relative]:
                raise ValueError("history-kernel target implementation bytes differ from runtime inventory")
        kernel_transition = {
            "schema": "wlr50_clean.semantic_history_kernel_transition.v1",
            "source_policy_version": source_policy_version, "target_policy_version": target_policy_version,
            "source_policy_contract": policy_contract(source_policy_version),
            "target_policy_contract": policy_contract(target_policy_version),
            "physical_mdp_changed": False, "reward_changed": False,
            "conditional_policy_changed": True, "fixed_mean_control_changed": True,
            "parameter_layout_changed": False, "learned_parameters_preserved": True,
            "old_rollout_inherited": False, "checkpoint_policy_prefix_during_migration": False,
        }
    checkpoint = checkpoint.resolve(strict=True)
    sidecar = checkpoint.with_name(checkpoint.stem + "_manifest.json")
    result = {"schema": "wlr50_clean.semantic_v3_new_mdp_warm_start.v1", "exact_mdp_resume": False,
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
    if kernel_transition is not None:
        result["policy_kernel_transition"] = kernel_transition
        result["action_output_semantics"] = "same learned tensors; changed history-conditioned Gaussian and fixed mean; unchanged physical projection"
        result["optimizer"]["reason"] = "explicit conditional-policy boundary; source moments verified then reset, learned weights preserved"
    if source_experiment is not None or target_experiment is not None:
        result["experiment_transition"] = {
            "source_experiment_id": source_experiment, "target_experiment_id": target_experiment,
            "source_artifact_namespace": source_namespace, "target_artifact_namespace": target_namespace,
            "target_configuration_namespace": (target_namespace if target_experiment in ("all_stage_acceptance_v1", "fsm_reference_p09_stable_v2") else "ppo_semantic_v3"), "v3_budgets_reset": False,
        }
    if observation_transition is not None:
        result["observation_scale_transition"] = observation_transition
        result["network"].update({
            "actor": "preserve_source_then_compensate_only_reviewed_first_layer_input_columns",
            "critic": "preserve_source_then_compensate_only_reviewed_first_layer_input_columns",
            "normalizers": "identity_RSL_state_preserved; explicit_two_column_fixed_scale_transition",
        })
        result["action_output_semantics"] = (
            "hash-bound source/target physical profiles; compensated input scaling preserves unclipped-domain "
            "network functions, not previously clipped history or subsequent physical trajectories")
    if observation_append_transition is not None:
        result["observation_append_transition"] = observation_append_transition
        result["network"].update({
            "source_observation_dimension": observation_append_transition["source_observation_dimension"],
            "observation_dimension": observation_append_transition["target_observation_dimension"],
            "actor": "preserve_source_parameters_then_append_zero_first_layer_columns_including_std_path",
            "critic": "preserve_source_parameters_then_append_zero_first_layer_columns",
            "normalizers": "identity_RSL_state_preserved; existing_fixed_preprocessing_unchanged",
        })
        result["optimizer"]["reason"] = "explicit role-observation boundary; verify source Adam then reset moments"
        result["action_output_semantics"] = observation_append_transition["equivalence_scope"]
    if observation_same_layout_transition is not None:
        result["observation_same_layout_transition"] = observation_same_layout_transition
        result["network"].update(source_observation_dimension=372, observation_dimension=372,
            actor="preserve_all_parameters_and_buffers_including_learned_std",
            critic="preserve_all_parameters_and_buffers",
            normalizers="identity_RSL_state_preserved; identical_complete372_schema_bytes")
        result["optimizer"]["reason"] = "explicit same372 semantic/authority boundary; verify source Adam then reset moments"
        if observation_same_layout_transition["schema"] in (FSM_REFERENCE_P09_SCHEMA, CAPTURE_HANDOFF_SAME372_SCHEMA):
            result["optimizer"].update(observation_same_layout_transition["optimizer"])
        result["action_output_semantics"] = observation_same_layout_transition["equivalence_scope"]
    return result


def v3_warm_start_checkpoint_name(record: Mapping[str, Any]) -> str:
    """Immutable revision-bound initial publication, separate from every prior boundary."""
    target = _contract(record["target_runtime_contract"])
    kernel = record.get("policy_kernel_transition")
    suffix = "" if kernel is None else "_p" + digest(kernel["target_policy_contract"])[:12]
    return (f"checkpoint_initial_v3_from_{int(record['source_global_policy_decisions']):09d}"
            f"_s{record['source_checkpoint_sha256'][:12]}"
            f"_g{target['source_git_commit'][:12]}_{target['runtime_content_sha256']}{suffix}.pt")


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


def _instrumentation_observation_contract(metadata, old, new, delta, project_root, *, mixed_factors):
    """Explicit unchanged HISTORY372 metadata/schema; no new MDP permission.

    Legacy 324 plans retain their byte-for-byte plan schema. This additional
    factor only opens exact resume for the validated N1 role layout, and only
    for the existing reviewed instrumentation allowlist.
    """
    from .semantic_policy_distribution import (
        CONFIG_NAMES, HISTORY_POLICY, policy_contract,
        policy_observation_layout_from_metadata, policy_version_from_metadata,
    )
    from .semantic_observation import load_semantic_observation_schema
    from .semantic_transfer_roles import ROLE_OBSERVATION_LAYOUT
    declared = metadata.get("policy_contract") or {}
    actor = (metadata.get("runner_config") or {}).get("actor", {})
    candidate = (declared.get("observation_layout") is not None
                 or declared.get("observation_dimension") == 372
                 or actor.get("observation_layout") is not None
                 or old.get("observation_dimension") == 372
                 or new.get("observation_dimension") == 372)
    if not candidate:
        return None
    if (policy_version_from_metadata(metadata) != HISTORY_POLICY
            or policy_observation_layout_from_metadata(metadata) != ROLE_OBSERVATION_LAYOUT
            or metadata.get("semantic_version") != "v3"
            or old.get("semantic_version") != "v3" or new.get("semantic_version") != "v3"
            or source_num_envs(metadata) != 1):
        raise ValueError("instrumentation372 requires verified v3 N1 HISTORY role-layout metadata")
    if mixed_factors or set(delta) - INSTRUMENTATION_FILES:
        raise ValueError("instrumentation372 cannot mix task/execution/video or other runtime factors")
    canonical = policy_contract(HISTORY_POLICY, observation_layout=ROLE_OBSERVATION_LAYOUT)
    selected = old.get("selected_configuration")
    if (not isinstance(selected, Mapping) or set(selected) != CONFIG_NAMES
            or new.get("selected_configuration") != selected):
        raise ValueError("instrumentation372 requires identical complete six selected configuration bindings")
    namespace = experiment_namespace("v3", old.get("experiment_id"))
    config_namespace = (namespace if old.get("experiment_id") in (
        "all_stage_acceptance_v1", "fsm_reference_p09_stable_v2") else "ppo_semantic_v3")
    records = {}
    for name, binding in selected.items():
        relative = f"configs/{config_namespace}/{name}"
        if (not isinstance(binding, Mapping) or set(binding) != {"path", "sha256"}
                or binding["path"] != relative
                or old["files"].get(relative) != binding["sha256"]
                or new["files"].get(relative) != binding["sha256"]):
            raise ValueError("instrumentation372 configuration path/hash binding differs")
        before = _version_bytes(project_root, old, relative, prefer_worktree=True)
        target = project_root / relative
        after = target.read_bytes()
        if before != after or hashlib.sha256(after).hexdigest() != binding["sha256"]:
            raise ValueError("instrumentation372 cannot change configuration or preprocessing bytes")
        records[name] = dict(binding)
    schema = load_semantic_observation_schema(project_root / selected["observation_schema.json"]["path"])
    if (schema.transfer_role_features_version != ROLE_OBSERVATION_LAYOUT
            or schema.dimension != canonical["observation_dimension"]):
        raise ValueError("instrumentation372 schema differs from the complete verified policy layout")
    for runtime in (old, new):
        if (runtime.get("observation_dimension", canonical["observation_dimension"])
                != canonical["observation_dimension"]
                or runtime.get("action_dimension", canonical["raw_action_dimension"])
                != canonical["raw_action_dimension"]):
            raise ValueError("instrumentation372 declared runtime dimensions disagree")
    return {
        "schema": "wlr50_clean.instrumentation_same_observation_contract.v1",
        "source_policy_contract": canonical, "target_policy_contract": canonical,
        "observation_layout": ROLE_OBSERVATION_LAYOUT,
        "selected_configuration": records,
        "observation_dimension": canonical["observation_dimension"],
        "action_dimension": canonical["raw_action_dimension"],
        "num_envs": 1, "parameter_mapping": None,
        "scope": "reviewed_instrumentation_only_no_control_or_MDP_change",
        "scope_is_reviewer_assertion_not_semantic_equivalence_proof": True,
    }


def _video_observation_contract(metadata, old, new, delta, project_root):
    """Same372 video receipt only after the separate strict video review.

    The instrumentation-only validator above stays unchanged. This branch
    additionally permits only existing VIDEO_FILES, never MDP/config changes.
    Legacy324 video plan shape is preserved.
    """
    from .semantic_policy_distribution import (
        CONFIG_NAMES, HISTORY_POLICY, policy_contract,
        policy_observation_layout_from_metadata, policy_version_from_metadata,
    )
    from .semantic_observation import load_semantic_observation_schema
    from .semantic_transfer_roles import ROLE_OBSERVATION_LAYOUT
    declared = metadata.get("policy_contract") or {}
    actor = (metadata.get("runner_config") or {}).get("actor", {})
    candidate = (declared.get("observation_layout") is not None
                 or declared.get("observation_dimension") == 372
                 or actor.get("observation_layout") is not None
                 or old.get("observation_dimension") == 372
                 or new.get("observation_dimension") == 372)
    if not candidate:
        return None
    if (policy_version_from_metadata(metadata) != HISTORY_POLICY
            or policy_observation_layout_from_metadata(metadata) != ROLE_OBSERVATION_LAYOUT
            or metadata.get("semantic_version") != "v3"
            or old.get("semantic_version") != "v3" or new.get("semantic_version") != "v3"
            or source_num_envs(metadata) != 1):
        raise ValueError("video372 requires verified v3 N1 HISTORY role-layout metadata")
    if set(delta) - INSTRUMENTATION_FILES - VIDEO_FILES:
        raise ValueError("video372 cannot change task/execution/configuration runtime files")
    canonical = policy_contract(HISTORY_POLICY, observation_layout=ROLE_OBSERVATION_LAYOUT)
    selected = old.get("selected_configuration")
    if (not isinstance(selected, Mapping) or set(selected) != CONFIG_NAMES
            or new.get("selected_configuration") != selected):
        raise ValueError("video372 requires identical complete six selected configuration bindings")
    namespace = experiment_namespace("v3", old.get("experiment_id"))
    config_namespace = (namespace if old.get("experiment_id") in (
        "all_stage_acceptance_v1", "fsm_reference_p09_stable_v2") else "ppo_semantic_v3")
    records = {}
    for name, binding in selected.items():
        relative = f"configs/{config_namespace}/{name}"
        if (not isinstance(binding, Mapping) or set(binding) != {"path", "sha256"}
                or binding["path"] != relative
                or old["files"].get(relative) != binding["sha256"]
                or new["files"].get(relative) != binding["sha256"]):
            raise ValueError("video372 configuration path/hash binding differs")
        before = _version_bytes(project_root, old, relative, prefer_worktree=True)
        target = project_root / relative
        after = target.read_bytes()
        if before != after or hashlib.sha256(after).hexdigest() != binding["sha256"]:
            raise ValueError("video372 cannot change configuration or preprocessing bytes")
        records[name] = dict(binding)
    schema = load_semantic_observation_schema(project_root / selected["observation_schema.json"]["path"])
    if (schema.transfer_role_features_version != ROLE_OBSERVATION_LAYOUT
            or schema.dimension != canonical["observation_dimension"]):
        raise ValueError("video372 schema differs from the complete verified policy layout")
    for runtime in (old, new):
        if (runtime.get("observation_dimension", canonical["observation_dimension"])
                != canonical["observation_dimension"]
                or runtime.get("action_dimension", canonical["raw_action_dimension"])
                != canonical["raw_action_dimension"]):
            raise ValueError("video372 declared runtime dimensions disagree")
    return {
        "schema": "wlr50_clean.video_same_observation_contract.v1",
        "source_policy_contract": canonical, "target_policy_contract": canonical,
        "observation_layout": ROLE_OBSERVATION_LAYOUT,
        "selected_configuration": records,
        "observation_dimension": canonical["observation_dimension"],
        "action_dimension": canonical["raw_action_dimension"],
        "num_envs": 1, "parameter_mapping": None,
        "scope": "reviewed_video_only_no_control_or_MDP_change",
        "scope_is_reviewer_assertion_not_semantic_equivalence_proof": True,
    }


def _timing_only_factor(metadata, old, new, delta, review, project_root):
    """One nominal timing opt-in, explicitly not equivalent physical trajectories."""
    import yaml
    from .semantic_observation import load_semantic_observation_schema
    from .semantic_policy_distribution import (
        CONFIG_NAMES, HISTORY_POLICY, policy_contract,
        policy_observation_layout_from_metadata, policy_version_from_metadata,
    )
    from .semantic_transfer_roles import ROLE_OBSERVATION_LAYOUT
    if (not isinstance(review, Mapping) or set(review) != {"reason"}
            or not isinstance(review["reason"], str) or not review["reason"].strip()):
        raise ValueError("timing-only migration requires an explicit review reason")
    if (metadata.get("semantic_version") != "v3"
            or policy_version_from_metadata(metadata) != HISTORY_POLICY
            or policy_observation_layout_from_metadata(metadata) != ROLE_OBSERVATION_LAYOUT
            or source_num_envs(metadata) != 1
            or any(c.get("semantic_version") != "v3"
                   or c.get("experiment_id") != "fsm_reference_p09_stable_v2" for c in (old, new))):
        raise ValueError("timing-only migration requires same-experiment v3 N1 HISTORY372")
    required = {SUPERVISOR, TIMING_ONLY_SPEC}
    allowed = required | {f"src/wlr50_clean/ppo/{name}.py" for name in
                          ("semantic_migration", "semantic_training")}
    if old["files"].keys() != new["files"].keys() or not required <= set(delta) or not set(delta) <= allowed:
        raise ValueError("timing-only migration permits only nominal class/config and its two compatibility modules")
    canonical = policy_contract(HISTORY_POLICY, observation_layout=ROLE_OBSERVATION_LAYOUT)
    if metadata.get("policy_contract") != canonical:
        raise ValueError("timing-only source policy contract differs from canonical HISTORY372")
    records = {}
    for name in sorted(CONFIG_NAMES):
        relative = f"configs/ppo_fsm_reference_p09_stable_v2/{name}"
        for contract in (old, new):
            selected = contract.get("selected_configuration", {})
            if (set(selected) != CONFIG_NAMES or selected[name] != {
                    "path": relative, "sha256": contract["files"].get(relative)}):
                raise ValueError("timing-only selected configuration binding differs")
        before = _version_bytes(project_root, old, relative, prefer_worktree=True)
        after = _version_bytes(project_root, new, relative, prefer_worktree=True)
        if hashlib.sha256(after).hexdigest() != new["files"][relative]:
            raise ValueError("timing-only target configuration bytes differ from inventory")
        if name == "stage_task_spec.yaml":
            source_spec, target_spec = yaml.safe_load(before), yaml.safe_load(after)
            if (not isinstance(source_spec, dict) or not isinstance(source_spec.get("nominal"), dict)
                    or "sequence_semantics" in source_spec["nominal"]):
                raise ValueError("timing-only source must not already contain the sequence opt-in")
            expected = json.loads(json.dumps(source_spec, allow_nan=False))
            expected["nominal"]["sequence_semantics"] = TIMING_ONLY_MODE
            if target_spec != expected:
                raise ValueError("timing-only permits only nominal.sequence_semantics addition")
        elif before != after:
            raise ValueError(f"timing-only cannot change configuration bytes: {name}")
        records[name] = {"path": relative, "source_sha256": old["files"][relative],
                         "target_sha256": new["files"][relative]}
    for relative in delta:
        _version_bytes(project_root, old, relative, prefer_worktree=True)
        if hashlib.sha256(_version_bytes(project_root, new, relative, prefer_worktree=True)).hexdigest() != new["files"][relative]:
            raise ValueError("timing-only target runtime bytes differ from inventory")
    before = _nominal_provider_byte_regions(_version_bytes(project_root, old, SUPERVISOR, prefer_worktree=True))
    after = _nominal_provider_byte_regions(_version_bytes(project_root, new, SUPERVISOR, prefer_worktree=True))
    if before[0] != after[0] or before[2] != after[2] or before[1] == after[1]:
        raise ValueError("timing-only may change only the NominalMotionProvider class bytes")
    schema = load_semantic_observation_schema(project_root / records["observation_schema.json"]["path"])
    if schema.dimension != 372 or schema.transfer_role_features_version != ROLE_OBSERVATION_LAYOUT:
        raise ValueError("timing-only requires the unchanged complete372 observation schema")
    for runtime in (old, new):
        if runtime.get("observation_dimension", 372) != 372 or runtime.get("action_dimension", 12) != 12:
            raise ValueError("timing-only runtime dimensions differ")
    return {"schema": TIMING_ONLY_SCHEMA, "review_reason": review["reason"].strip(),
        "sequence_semantics": TIMING_ONLY_MODE, "configuration_bindings": records,
        "physical_mdp_changed": True, "nominal_control_changed": True,
        "reward_changed": False, "task_acceptance_changed": False,
        "observation_semantics_changed": [], "physical_actuators_changed": False,
        "action_ranges_changed": False, "kernel_changed": False,
        "nominal_provider": {"source_class_sha256": hashlib.sha256(before[1]).hexdigest(),
            "target_class_sha256": hashlib.sha256(after[1]).hexdigest(),
            "outside_class_sha256": hashlib.sha256(before[0] + b"\0" + before[2]).hexdigest(),
            "outside_class_bytes_identical": True},
        "observation_contract": {"source_policy_contract": canonical, "target_policy_contract": dict(canonical),
            "observation_layout": ROLE_OBSERVATION_LAYOUT, "observation_dimension": 372,
            "action_dimension": 12, "num_envs": 1,
            "parameter_mapping": "identity_all_parameters_and_buffers"},
        "optimizer": "preserve_complete_verified_Adam_state_and_effective_learning_rate",
        "normalizers": "preserve_verified_identity_RSL_state",
        "scope": "reviewed_nominal_timing_only_not_physical_MDP_or_trajectory_equivalence",
        "scope_is_reviewer_assertion_not_semantic_equivalence_proof": True}


def _body_reward_source_delta(before: str, after: str) -> dict[str, Any]:
    """Reverse only the reviewed helper/loader/body-weight insertions."""
    import ast
    marker = 'ROLE_TRANSFER_VERSION = "diagonal_transfer_roles_v1"\n'
    boundary = '@dataclass(frozen=True)\nclass SemanticRewardConfig:'
    if before.count(marker) != 1 or after.count(marker) != 1:
        raise ValueError("body reward role marker is not unique")
    source_prefix, source_tail = before.split(marker)
    target_prefix, target_tail = after.split(marker)
    source_gap, source_rest = source_tail.split(boundary, 1)
    added, target_rest = target_tail.split(boundary, 1)
    if source_prefix != target_prefix or source_gap.strip():
        raise ValueError("body reward may not change the module prefix")
    tree = ast.parse(added)
    helper_names = {"_validate_carry_body_allowance", "_current_functional_body_allowance"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    constants = [node for node in tree.body if isinstance(node, ast.Assign)]
    if (len(tree.body) != 3 or len(functions) != 2 or {n.name for n in functions} != helper_names
            or any(n.decorator_list for n in functions) or len(constants) != 1
            or len(constants[0].targets) != 1 or not isinstance(constants[0].targets[0], ast.Name)
            or constants[0].targets[0].id != "CARRY_BODY_ALLOWANCE_MODE"
            or ast.literal_eval(constants[0].value) != BODY_REWARD_MODE):
        raise ValueError("body reward permits exactly the named constant and two helpers")
    restored = target_prefix + marker + source_gap + boundary + target_rest
    loader = "    _validate_carry_body_allowance(v)\n"
    body = ('            body_transfer_fraction = transfer_fraction\n'
            '            if v.get("carry_body_allowance") == CARRY_BODY_ALLOWANCE_MODE:\n'
            '                body_transfer_fraction = max(body_transfer_fraction,\n'
            '                    _current_functional_body_allowance(sample.current.task, v["capture_settle_window_s"]))\n')
    new_weight = '            motion_weight = 1.-(1.-v["transfer_attitude_weight"])*body_transfer_fraction\n'
    old_weight = '            motion_weight = 1.-(1.-v["transfer_attitude_weight"])*transfer_fraction\n'
    for text in (loader, body, new_weight):
        if restored.count(text) != 1:
            raise ValueError("body reward consumer insertion differs from the reviewed form")
    restored = restored.replace(loader, "").replace(body, "").replace(new_weight, old_weight)
    if restored != before:
        raise ValueError("body reward changed another loader/evaluate/contact/smoothness/potential path")
    return {"source_normalized_sha256": hashlib.sha256(before.encode()).hexdigest(),
        "target_normalized_sha256": hashlib.sha256(after.encode()).hexdigest(),
        "added_helpers_sha256": hashlib.sha256(added.encode()).hexdigest(),
        "outside_reviewed_body_consumption_unchanged": True}


def _body_reward_factor(checkpoint, metadata, old, new, delta, review, project_root):
    import yaml
    from .semantic_policy_distribution import CONFIG_NAMES, HISTORY_POLICY, policy_contract
    from .semantic_transfer_roles import ROLE_OBSERVATION_LAYOUT
    required_review = {"reason", "nominal_reference_manifest", "timing_plan"}
    if (not isinstance(review, Mapping) or set(review) != required_review
            or not isinstance(review["reason"], str) or not review["reason"].strip()):
        raise ValueError("body reward requires the exact explicit review and same-N reference")
    reference_path = Path(review["nominal_reference_manifest"]).resolve(strict=True)
    record = json.loads(reference_path.read_text(encoding="utf-8"))
    result = record.get("result")
    if (record.get("lifecycle") not in ("SUCCEEDED", "DIAGNOSTIC_FAILURE")
            or not record.get("completed_at_utc") or not isinstance(result, Mapping)
            or result.get("mode") not in ("semantic_prior_eval", "semantic_residual_eval")):
        raise ValueError("body reward reference must be a completed same-N evaluation; success is not required")
    updates = [row[key] for row in (record, result) for key in
               ("optimizer_updates", "optimizer_updates_during_evaluation") if key in row]
    if not updates or any(type(n) is not int or n != 0 for n in updates):
        raise ValueError("body reward same-N reference must have zero optimizer updates")
    reference = _contract(record["runtime_contract"])
    if result.get("runtime_contract") != reference:
        raise ValueError("body reward reference runtime bindings disagree")
    variable = {"files", "runtime_content_sha256", "source_git_commit", "selected_configuration"}
    for contract in (old, reference):
        if ({k:v for k,v in contract.items() if k not in variable}
                != {k:v for k,v in new.items() if k not in variable}):
            raise ValueError("body reward cannot change fixed physical/runtime metadata")
    canonical = policy_contract(HISTORY_POLICY, observation_layout=ROLE_OBSERVATION_LAYOUT)
    if (metadata.get("semantic_version") != "v3" or metadata.get("policy_contract") != canonical
            or source_num_envs(metadata) != 1 or new.get("semantic_version") != "v3"
            or new.get("experiment_id") != "fsm_reference_p09_stable_v2"):
        raise ValueError("body reward requires same-experiment v3 N1 HISTORY372")
    if old["files"].keys() != reference["files"].keys() or reference["files"].keys() != new["files"].keys():
        raise ValueError("body reward cannot add or remove runtime files")
    required = {BODY_REWARD_CODE, BODY_REWARD_CONFIG}
    allowed = required | {f"src/wlr50_clean/ppo/{n}.py" for n in ("semantic_migration", "semantic_training")}
    final_delta = {p for p in reference["files"] if reference["files"][p] != new["files"][p]}
    if not required <= final_delta or not final_delta <= allowed:
        raise ValueError("body reward changed N or a file outside the exact body/compatibility scope")
    configs = {}
    for name in sorted(CONFIG_NAMES):
        relative = f"configs/ppo_fsm_reference_p09_stable_v2/{name}"
        for contract in (old, reference, new):
            selected = contract.get("selected_configuration", {})
            if (set(selected) != CONFIG_NAMES or selected[name] != {
                    "path": relative, "sha256": contract["files"].get(relative)}):
                raise ValueError("body reward selected configuration binding differs")
        before = _version_bytes(project_root, reference, relative, prefer_worktree=True)
        after = _version_bytes(project_root, new, relative, prefer_worktree=True)
        if name == "reward_config.yaml":
            source_cfg, target_cfg = yaml.safe_load(before), yaml.safe_load(after)
            if any(k in source_cfg for k in ("carry_body_allowance", "capture_settle_window_s")):
                raise ValueError("body reward opt-in already exists in the reference")
            expected = dict(source_cfg, carry_body_allowance=BODY_REWARD_MODE, capture_settle_window_s=.5)
            if target_cfg != expected:
                raise ValueError("body reward permits only the exact two config additions")
        elif before != after:
            raise ValueError("body reward cannot change the other five selected config bytes")
        configs[name] = {"path": relative, "source_sha256": reference["files"][relative],
                         "target_sha256": new["files"][relative]}
    reference_spec = yaml.safe_load(_version_bytes(project_root, reference, TIMING_ONLY_SPEC, prefer_worktree=True))
    source_spec = yaml.safe_load(_version_bytes(project_root, old, TIMING_ONLY_SPEC, prefer_worktree=True))
    if (reference_spec.get("nominal", {}).get("sequence_semantics") != TIMING_ONLY_MODE
            or reference_spec.get("transfer_roles", {}).get("window_s") != .5):
        raise ValueError("body reward reference must retain reviewed timing and the existing0.5s transfer window")
    timing = None
    if source_spec.get("nominal", {}).get("sequence_semantics") == TIMING_ONLY_MODE:
        if review["timing_plan"] is not None or set(delta) - allowed:
            raise ValueError("already-timed body reward source cannot change nominal or add a timing ancestor")
        for relative in (SUPERVISOR, TIMING_ONLY_SPEC):
            if old["files"][relative] != reference["files"][relative]:
                raise ValueError("body reward source N differs from its evaluated reference")
    else:
        if not isinstance(review["timing_plan"], str) or not review["timing_plan"]:
            raise ValueError("pre-timing body reward source requires its existing verified timing plan")
        timing = validate_migration_plan(checkpoint, reference, Path(review["timing_plan"]), project_root=project_root)
        if timing.get("nominal_timing_factor", {}).get("schema") != TIMING_ONLY_SCHEMA:
            raise ValueError("body reward ancestor must be the narrow timing-only migration")
    for relative in required:
        if old["files"][relative] != reference["files"][relative]:
            raise ValueError("body reward source and same-N reference already differ in reward")
    before = _version_text(project_root, reference, BODY_REWARD_CODE, prefer_worktree=True)
    after = _version_text(project_root, new, BODY_REWARD_CODE, prefer_worktree=True)
    scope = _body_reward_source_delta(before, after)
    for relative in final_delta:
        if file_sha(project_root / relative) != new["files"][relative]:
            raise ValueError("body reward target runtime bytes differ from the current inventory")
    return {"schema": BODY_REWARD_SCHEMA, "review_reason": review["reason"].strip(),
        "nominal_reference_manifest": str(reference_path), "nominal_reference_manifest_sha256": file_sha(reference_path),
        "nominal_reference_contract_sha256": digest(reference), "nominal_reference_commit": reference["source_git_commit"],
        "timing_ancestor": None if timing is None else {"plan_path": timing["plan_path"],
            "plan_sha256": timing["plan_sha256"], "nominal_timing_factor": timing["nominal_timing_factor"]},
        "configuration_bindings": configs, "reward_source_scope": scope,
        "reward_changed": True, "physical_mdp_changed": True,
        "nominal_changed_since_checkpoint": timing is not None, "nominal_changed_since_reference": False,
        "observation_semantics_changed": [], "task_acceptance_changed": False,
        "physical_actuators_changed": False, "action_ranges_changed": False, "kernel_changed": False,
        "observation_contract": {"source_policy_contract": canonical, "target_policy_contract": dict(canonical),
            "observation_layout": ROLE_OBSERVATION_LAYOUT, "observation_dimension": 372,
            "action_dimension": 12, "num_envs": 1, "parameter_mapping": "identity_all_parameters_and_buffers"},
        "optimizer": "preserve_complete_verified_Adam_state_and_effective_learning_rate",
        "normalizers": "preserve_verified_identity_RSL_state",
        "scope_is_reviewer_assertion_not_semantic_equivalence_proof": True}


def _height_candidate(value):
    """Independent migration bounds: do not trust the candidate's own validator."""
    keys = {"mode", "candidate_id", "preparation_reduction_deg", "post_lift_recovery_deg",
            "recovery_rate_deg_s"}
    if (not isinstance(value, dict) or set(value) != keys
            or value["mode"] != HEIGHT_RECOVERY_MODE
            or not isinstance(value["candidate_id"], str) or not value["candidate_id"].strip()):
        raise ValueError("height recovery requires the exact versioned candidate schema")
    for name in ("preparation_reduction_deg", "post_lift_recovery_deg"):
        row = value[name]
        if (not isinstance(row, dict) or set(row) != {"front_left_hip", "rear_left_hip"}
                or any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 10
                       for v in row.values())):
            raise ValueError("height recovery permits only separate FL/RL amounts in 0..10 degrees")
    rate = value["recovery_rate_deg_s"]
    if type(rate) not in (int, float) or not math.isfinite(rate) or not 0 < rate <= 60:
        raise ValueError("height recovery rate must be finite, positive and at most60 deg/s")
    return value


def _height_source_scope(before: str, after: str, *, functions=(), methods=(), constant=False,
                         geometry_import=False):
    """Compare protected AST outside exact reviewed function/method regions.

    Bodies remain a code-review assertion, not a proof of physical equivalence.
    Module additions and physical setter/step calls are not opened by this scope.
    """
    import ast
    def stripped(text):
        tree = ast.parse(text)
        seen = set()
        class Scope(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                if node.name in functions:
                    seen.add(node.name)
                    return None
                return node

            def visit_ClassDef(self, node):
                for item in list(node.body):
                    key = f"{node.name}.{getattr(item, 'name', '')}"
                    if isinstance(item, ast.FunctionDef) and key in methods:
                        seen.add(key)
                        node.body.remove(item)
                return node

            def visit_Assign(self, node):
                if constant and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "BOUNDED_RR_MODE":
                    if ast.literal_eval(node.value) != HEIGHT_GEOMETRY_MODE:
                        raise ValueError("height geometry mode constant changed")
                    return None
                return node

            def visit_ImportFrom(self, node):
                if geometry_import and node.module == "semantic_nominal_geometry":
                    node.names = [n for n in node.names if not (n.name == "BOUNDED_RR_MODE" and n.asname is None)]
                return node
        tree = Scope().visit(tree)
        return ast.dump(tree, include_attributes=False), seen
    source, old_seen = stripped(before)
    target, new_seen = stripped(after)
    if source != target or not old_seen <= new_seen:
        raise ValueError("height recovery changed code outside reviewed named regions")
    def physical_calls(text):
        calls = []
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Call):
                name = getattr(node.func, "attr", getattr(node.func, "id", ""))
                if (name.startswith(("set_", "write_joint", "write_root", "apply_force"))
                        or name in {"reset", "step", "step_physics", "simulate", "render", "write_data_to_sim", "apply_action"}):
                    calls.append(ast.dump(node, include_attributes=False))
        return sorted(calls)
    if physical_calls(before) != physical_calls(after):
        raise ValueError("height recovery may not change physical setter/reset/step/render calls")
    return {"protected_ast_sha256": digest(source), "reviewed_regions": sorted(new_seen),
            "protected_ast_identical": True}


def _height_no_physics_writes(text: str):
    """Conservative syntactic veto, not a claim that arbitrary Python is pure."""
    import ast
    forbidden = {"setattr", "exec", "eval", "step", "step_physics", "simulate", "reset",
                 "render", "update", "write_data_to_sim", "apply_torque", "apply_action"}
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "attr", getattr(node.func, "id", ""))
            if name in forbidden or name.startswith(("set_", "write_joint", "write_root", "apply_force")):
                raise ValueError(f"height read-only helper contains forbidden physical/dynamic call: {name}")
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                for child in ast.walk(target):
                    if isinstance(child, ast.Attribute) and isinstance(child.ctx, ast.Store) and not (
                            isinstance(child.value, ast.Name) and child.value.id == "self"):
                        raise ValueError("height helper may not assign external object attributes")


def _height_recovery_factor(metadata, old, new, delta, review, project_root):
    import ast
    import yaml
    from .semantic_observation import load_semantic_observation_schema
    from .semantic_policy_distribution import CONFIG_NAMES, HISTORY_POLICY, policy_contract
    from .semantic_transfer_roles import ROLE_OBSERVATION_LAYOUT
    if (not isinstance(review, Mapping) or set(review) != {"reason", "reviewed_code_sha256"}
            or not isinstance(review["reason"], str) or not review["reason"].strip()
            or not isinstance(review["reviewed_code_sha256"], Mapping)):
        raise ValueError("height recovery requires an explicit reason and exact reviewed code hashes")
    canonical = policy_contract(HISTORY_POLICY, observation_layout=ROLE_OBSERVATION_LAYOUT)
    if (metadata.get("semantic_version") != "v3" or metadata.get("policy_contract") != canonical
            or source_num_envs(metadata) != 1
            or any(c.get("semantic_version") != "v3" or c.get("experiment_id") != "fsm_reference_p09_stable_v2"
                   for c in (old, new))):
        raise ValueError("height recovery requires the same experiment N1 canonical HISTORY372")
    if (set(old["files"]) - set(new["files"]) or set(new["files"]) - set(old["files"]) - HEIGHT_NEW_FILES
            or not HEIGHT_FILES <= set(new["files"]) or not set(delta) <= HEIGHT_FILES):
        raise ValueError("height recovery runtime delta exceeds its exact candidate boundary")
    hashes = {p: new["files"][p] for p in sorted(HEIGHT_CODE_FILES)}
    if dict(review["reviewed_code_sha256"]) != hashes:
        raise ValueError("height recovery code review is not bound to every exact candidate code file")
    raw_old, raw_new = {}, {}
    for path in sorted(HEIGHT_FILES):
        raw_new[path] = _version_bytes(project_root, new, path, prefer_worktree=True)
        if path in old["files"]:
            raw_old[path] = _version_bytes(project_root, old, path, prefer_worktree=True)
    records, source_candidate, target_candidate = {}, None, None
    source_final_stop, target_final_stop = None, None
    for name in sorted(CONFIG_NAMES):
        path = f"configs/ppo_fsm_reference_p09_stable_v2/{name}"
        for contract in (old, new):
            selected = contract.get("selected_configuration", {})
            if set(selected) != CONFIG_NAMES or selected[name] != {"path": path, "sha256": contract["files"].get(path)}:
                raise ValueError("height recovery selected configuration bindings differ")
        before = _version_bytes(project_root, old, path, prefer_worktree=True)
        after = _version_bytes(project_root, new, path, prefer_worktree=True)
        source, target = yaml.safe_load(before), yaml.safe_load(after)
        if name == "stage_task_spec.yaml":
            if (source.get("nominal", {}).get("sequence_semantics") != TIMING_ONLY_MODE
                    or target.get("nominal", {}).get("sequence_semantics") != TIMING_ONLY_MODE):
                raise ValueError("height recovery requires existing source partial-order timing")
            source_candidate = source["nominal"].pop("height_recovery", None)
            target_candidate = _height_candidate(target["nominal"].pop("height_recovery", None))
            if source_candidate is not None:
                _height_candidate(source_candidate)
            source_final_stop = source["nominal"].pop("final_stop_owner", None)
            target_final_stop = target["nominal"].pop("final_stop_owner", None)
            if (source_final_stop not in (None, HEIGHT_FINAL_STOP_MODE)
                    or target_final_stop not in (None, HEIGHT_FINAL_STOP_MODE)
                    or source_final_stop is not None and target_final_stop is None):
                raise ValueError("height recovery permits only the reviewed current-physical final-stop owner")
            if source != target:
                raise ValueError("height recovery may change only its nominal height/final-stop candidates in task spec")
        elif name == "execution_profile.yaml":
            prior_mode = source.pop("nominal_geometry_advisory", None)
            if (prior_mode not in ("functional_rr_preplace_nominal_advisory_v2", HEIGHT_GEOMETRY_MODE)
                    or target.pop("nominal_geometry_advisory", None) != HEIGHT_GEOMETRY_MODE
                    or source != target):
                raise ValueError("height recovery may change only the reviewed geometry advisory version")
        elif before != after:
            raise ValueError(f"height recovery cannot change protected config bytes: {name}")
        records[name] = {"path": path, "source_sha256": old["files"][path], "target_sha256": new["files"][path]}
    source_helpers = HEIGHT_NEW_FILES & set(old["files"])
    if (source_helpers not in (set(), HEIGHT_NEW_FILES)
            or (source_candidate is None) != (not source_helpers)):
        raise ValueError("height recovery source candidate and new-module inventory disagree")
    scopes = {}
    before = _nominal_provider_byte_regions(raw_old[SUPERVISOR])
    after = _nominal_provider_byte_regions(raw_new[SUPERVISOR])
    if before[0] != after[0] or before[2] != after[2]:
        raise ValueError("height recovery may not change supervisor outside NominalMotionProvider")
    scopes[SUPERVISOR] = {"outside_class_bytes_identical": True,
        "source_class_sha256": hashlib.sha256(before[1]).hexdigest(),
        "target_class_sha256": hashlib.sha256(after[1]).hexdigest()}
    for name, functions, methods, constant, geometry_import in (
        ("semantic_nominal_geometry", ("capture_nominal_geometry_context", "correct_nominal_geometry", "_bounded_rr_correction"), (), True, False),
        ("semantic_backend", ("load_execution_profile",), ("SemanticIsaacBackend.__init__", "SemanticIsaacBackend._atomic_apply"), False, True),
        ("semantic_video", ("capture_semantic_video",), ("EndpointObserver.__init__", "EndpointObserver.__call__"), False, False)):
        path = f"src/wlr50_clean/ppo/{name}.py"
        scopes[path] = _height_source_scope(raw_old[path].decode("utf-8"), raw_new[path].decode("utf-8"),
            functions=functions, methods=methods, constant=constant, geometry_import=geometry_import)
    for path in HEIGHT_NEW_FILES:
        text = raw_new[path].decode("utf-8")
        tree = ast.parse(text)
        names = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
        expected = ({"validate_height_candidate", "source_owner_height_target", "current_rr_recovery_permission"}
            if path.endswith("semantic_height_recovery.py") else
            {"member", "plain", "read_value", "transform_point", "lowest_collider_point", "resolve_rr_mount", "HeightDiagnostics"})
        if names != expected:
            raise ValueError("height helper top-level definitions exceed reviewed scope")
        _height_no_physics_writes(text)
    schema = load_semantic_observation_schema(project_root / records["observation_schema.json"]["path"])
    if schema.dimension != 372 or schema.transfer_role_features_version != ROLE_OBSERVATION_LAYOUT:
        raise ValueError("height recovery requires the unchanged complete372 observation schema")
    result = {"schema": HEIGHT_RECOVERY_SCHEMA, "review_reason": review["reason"].strip(),
        "reviewed_code_sha256": hashes, "configuration_bindings": records, "code_scope": scopes,
        "source_candidate": source_candidate, "target_candidate": target_candidate,
        "physical_mdp_changed": True, "nominal_control_changed": True, "reward_changed": False,
        "task_acceptance_changed": False, "observation_semantics_changed": [],
        "physical_actuators_changed": False, "action_ranges_changed": False, "kernel_changed": False,
        "observation_contract": {"source_policy_contract": canonical, "target_policy_contract": dict(canonical),
            "observation_layout": ROLE_OBSERVATION_LAYOUT, "observation_dimension": 372,
            "action_dimension": 12, "num_envs": 1, "parameter_mapping": "identity_all_parameters_and_buffers"},
        "optimizer": "preserve_complete_verified_Adam_state_and_effective_learning_rate",
        "normalizers": "preserve_verified_identity_RSL_state",
        "scope_is_reviewer_assertion_not_semantic_equivalence_proof": True}
    # Leave earlier immutable plans byte-structurally compatible when neither
    # side has this opt-in. The new candidate is nominal ownership, never an
    # exemption from the protected TaskEvaluator or final observation window.
    if source_final_stop is not None or target_final_stop is not None:
        result["final_stop_nominal_owner"] = {
            "source_mode": source_final_stop, "target_mode": target_final_stop,
            "task_acceptance_and_post_completion_window_changed": False,
            "policy_channels_or_action_distribution_changed": False,
        }
    return result


def _temperature_protected_scope(before, after, *, functions=(), classes=(), constants=()):
    """Protect old control/learner code outside the explicit sampling boundary."""
    import ast
    def protected(text):
        tree = ast.parse(text)
        seen, retained = set(), []
        for node in tree.body:
            key = getattr(node, "name", None)
            if (isinstance(node, ast.FunctionDef) and key in functions
                    or isinstance(node, ast.ClassDef) and key in classes):
                seen.add(key)
                continue
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name) and node.targets[0].id in constants):
                seen.add(node.targets[0].id)
                continue
            retained.append(node)
        tree.body = retained
        return ast.dump(tree, include_attributes=False), seen
    source, old_seen = protected(before)
    target, new_seen = protected(after)
    if source != target or not old_seen <= new_seen:
        raise ValueError("temperature continuation changed code outside reviewed sampling regions")
    def physical_calls(text):
        calls = []
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Call):
                name = getattr(node.func, "attr", getattr(node.func, "id", ""))
                if (name.startswith(("set_", "write_joint", "write_root", "apply_force"))
                        or name in {"reset", "step", "step_physics", "simulate", "render", "write_data_to_sim", "apply_action"}):
                    calls.append(ast.dump(node, include_attributes=False))
        return sorted(calls)
    if physical_calls(before) != physical_calls(after):
        raise ValueError("temperature continuation cannot change physical or optimizer step calls")
    return {"protected_ast_sha256": digest(source), "protected_ast_identical": True,
            "reviewed_regions": sorted(new_seen)}


def _build_exploration_temperature_plan(checkpoint, metadata, old, new, *,
        allowed_changed_files, reason, review, project_root):
    from .semantic_policy_distribution import (CONFIG_NAMES, HISTORY_POLICY,
        HISTORY_TEMPERED_POLICY, HISTORY_QUARTER_TEMPERED_POLICY,
        policy_contract, policy_version_from_metadata, _same_json)
    from .semantic_training import semantic_runner_config
    from .semantic_return_profile import runner_return_profile
    from .semantic_transfer_roles import ROLE_OBSERVATION_LAYOUT
    if (not isinstance(review, Mapping) or set(review) != {"reason", "reviewed_code_sha256"}
            or not isinstance(review["reason"], str) or not review["reason"].strip()
            or not isinstance(review["reviewed_code_sha256"], Mapping)
            or not isinstance(reason, str) or not reason.strip()):
        raise ValueError("temperature continuation requires an explicit exact-code review")
    source_version = policy_version_from_metadata(metadata)
    quarter = (source_version == HISTORY_TEMPERED_POLICY
        and old.get("experiment_id") == "task_first_recovery_v1")
    namespace = "task_first_recovery_v1" if quarter else "fsm_reference_p09_stable_v2"
    expected_source = HISTORY_TEMPERED_POLICY if quarter else HISTORY_POLICY
    target_version = HISTORY_QUARTER_TEMPERED_POLICY if quarter else HISTORY_TEMPERED_POLICY
    source_temperature, target_temperature = (.5, .25) if quarter else (1., .5)
    source_policy = policy_contract(expected_source, observation_layout=ROLE_OBSERVATION_LAYOUT)
    target_policy = policy_contract(target_version, observation_layout=ROLE_OBSERVATION_LAYOUT)
    if (source_version != expected_source or not _same_json(metadata.get("policy_contract"), source_policy)
            or metadata.get("semantic_version") != "v3" or source_num_envs(metadata) != 1
            or any(c.get("semantic_version") != "v3" or c.get("experiment_id") != namespace
                   for c in (old, new))):
        raise ValueError("temperature continuation requires existing same-experiment N1 HISTORY372")
    variable = {"files", "runtime_content_sha256", "source_git_commit"}
    if ({k:v for k,v in old.items() if k not in variable}
            != {k:v for k,v in new.items() if k not in variable}):
        raise ValueError("temperature continuation cannot change runtime, physics, budgets or configuration bindings")
    delta = sorted(p for p in set(old["files"]) | set(new["files"]) if old["files"].get(p) != new["files"].get(p))
    declared = list(allowed_changed_files)
    if (len(set(declared)) != len(declared) or sorted(declared) != delta
            or set(old["files"]) != set(new["files"])
            or not EXPLORATION_TEMPERATURE_FILES <= set(new["files"])
            or not set(delta) <= EXPLORATION_TEMPERATURE_FILES):
        raise ValueError("temperature continuation delta exceeds its exact five-file sampling boundary")
    if quarter:
        # The existing CLI already selects the verified target contract. This
        # continuation needs no routing, nominal, evaluator or learner-loop edit.
        core = {"src/wlr50_clean/ppo/semantic_history_actor.py",
                "src/wlr50_clean/ppo/semantic_policy_distribution.py"}
        if not core <= set(delta) or "src/wlr50_clean/ppo/semantic_cli.py" in delta:
            raise ValueError("quarter temperature requires its two kernel files and unchanged CLI")
    hashes = {p: new["files"][p] for p in sorted(EXPLORATION_TEMPERATURE_FILES)}
    if dict(review["reviewed_code_sha256"]) != hashes:
        raise ValueError("temperature review hashes do not bind the exact current code")
    records = {}
    for name in sorted(CONFIG_NAMES):
        path = f"configs/ppo_{namespace}/{name}"
        for c in (old, new):
            selected = c.get("selected_configuration", {})
            if set(selected) != CONFIG_NAMES or selected[name] != {"path": path, "sha256": c["files"].get(path)}:
                raise ValueError("temperature selected configuration binding differs")
        before = _version_bytes(project_root, old, path, prefer_worktree=True)
        after = _version_bytes(project_root, new, path, prefer_worktree=True)
        if before != after:
            raise ValueError("temperature continuation cannot change any nominal/reward/action/observation/quality config bytes")
        records[name] = {"path": path, "source_sha256": old["files"][path], "target_sha256": new["files"][path]}
    scopes = {}
    allowed = {
        "semantic_history_actor": {"classes": ("SemanticTemperedHistoryMLPModel",)},
        "semantic_policy_distribution": {
            "functions": ("policy_contract", "configure_policy_distribution", "supported_heteroscedastic_contract_version", "policy_version_from_metadata"),
            "constants": ("HISTORY_TEMPERED_POLICY", "HISTORY_TEMPERED_ACTOR_CLASS")},
        "semantic_training": {"functions": ("semantic_runner_config", "load_semantic_checkpoint", "_validated_exploration_temperature_factor")},
        "semantic_cli": {"functions": ("_preflight_checkpoint", "_resolved_policy_version")},
    }
    if quarter:
        allowed["semantic_history_actor"] = {"classes": ("SemanticQuarterTemperedHistoryMLPModel",)}
        allowed["semantic_policy_distribution"]["constants"] = (
            "HISTORY_QUARTER_TEMPERED_POLICY", "HISTORY_QUARTER_TEMPERED_ACTOR_CLASS")
        allowed["semantic_training"] = {"functions": (
            "semantic_runner_config", "_validated_exploration_temperature_factor")}
        allowed["semantic_cli"] = {}
    for name, options in allowed.items():
        path = f"src/wlr50_clean/ppo/{name}.py"
        before = _version_bytes(project_root, old, path, prefer_worktree=True).decode("utf-8")
        after = _version_bytes(project_root, new, path, prefer_worktree=True).decode("utf-8")
        scopes[path] = _temperature_protected_scope(before, after, **options)
    _version_bytes(project_root, new, "src/wlr50_clean/ppo/semantic_migration.py", prefer_worktree=True)
    horizon = runner_return_profile(metadata["runner_config"], semantic_version="v3")
    options = {"seed": metadata["seed"], "device": metadata["runner_config"]["device"],
        "semantic_version": "v3", "return_profile": horizon["version"], "observation_layout": ROLE_OBSERVATION_LAYOUT}
    source_config = semantic_runner_config(policy_version=source_version, **options)
    target_config = semantic_runner_config(policy_version=target_version, **options)
    if not _same_json(metadata["runner_config"], source_config):
        raise ValueError("temperature source runner configuration is not canonical")
    # The actor selector/explicit innovation scale are the only configuration changes.
    source_rest = json.loads(json.dumps(source_config))
    target_rest = json.loads(json.dumps(target_config))
    source_rest["actor"].pop("class_name")
    target_rest["actor"].pop("class_name")
    source_scale = source_rest["actor"].pop("exploration_std_temperature", 1.)
    target_scale = target_rest["actor"].pop("exploration_std_temperature", None)
    if source_scale != source_temperature or target_scale != target_temperature or source_rest != target_rest:
        raise ValueError("temperature continuation may change only actor selection and its reviewed fixed innovation scale")
    observation = {"source_policy_contract": source_policy, "target_policy_contract": target_policy,
        "observation_layout": ROLE_OBSERVATION_LAYOUT, "observation_dimension": 372,
        "action_dimension": 12, "num_envs": 1, "parameter_mapping": "identity_all_parameters_and_buffers"}
    factor = {"schema": EXPLORATION_TEMPERATURE_SCHEMA, "review_reason": review["reason"].strip(),
        "reviewed_code_sha256": hashes, "configuration_bindings": records, "code_scope": scopes,
        "source_policy_version": source_version, "target_policy_version": target_version,
        "source_policy_contract": source_policy, "target_policy_contract": target_policy,
        "source_runner_config": source_config, "target_runner_config": target_config,
        "source_exploration_std_temperature": source_temperature, "target_exploration_std_temperature": target_temperature,
        "kernel_changed": True, "physical_mdp_changed": False, "nominal_control_changed": False,
        "reward_changed": False, "task_acceptance_changed": False, "action_ranges_changed": False,
        "observation_semantics_changed": [], "observation_contract": observation,
        "deterministic_same_weights_same_observation": "exact_original_conditional_mean_path",
        "stochastic_likelihood": "same_effective_std_for_sampling_old_params_logprob_entropy_KL_and_update",
        "optimizer": "preserve_complete_verified_Adam_state_and_effective_learning_rate",
        "normalizers": "preserve_verified_identity_RSL_state", "migration_added_updates": 0,
        "scope_is_reviewer_assertion_not_semantic_equivalence_proof": True}
    sidecar = checkpoint.with_name(checkpoint.stem + "_manifest.json")
    return {"schema": SCHEMA, "reason": reason.strip(), "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": file_sha(checkpoint), "source_manifest_sha256": file_sha(sidecar),
        "source_contract_sha256": digest(old), "target_contract_sha256": digest(new),
        "source_git_commit": old["source_git_commit"], "target_git_commit": new["source_git_commit"],
        "allowed_changed_files": delta,
        "changed_file_hashes": {p: {"before": old["files"][p], "after": new["files"][p]} for p in delta},
        "geometric_factor": None, "observation_dimension": 372, "action_dimension": 12,
        "preserve_actor_critic_optimizer_normalizer_rng_and_budget": True,
        "discard_old_rollout_storage": True, "physics_resume": "fresh_legal_P01_reset",
        "exploration_temperature_factor": factor}


def _build_task_first_reward_plan(checkpoint, metadata, old, new, *,
        allowed_changed_files, reason, review, project_root):
    """Reviewed reward-only objective boundary; successful N and physical task stay fixed."""
    import copy
    import yaml
    from .semantic_policy_distribution import CONFIG_NAMES, HISTORY_TEMPERED_POLICY, policy_contract
    from .semantic_transfer_roles import ROLE_OBSERVATION_LAYOUT
    if (not isinstance(review, Mapping) or set(review) != {"reason", "reviewed_code_sha256"}
            or not isinstance(review["reason"], str) or not review["reason"].strip()
            or not isinstance(reason, str) or not reason.strip()):
        raise ValueError("task-first reward requires explicit reason and exact reviewed code hashes")
    if (old.get("experiment_id") not in ("fsm_reference_p09_stable_v2", "task_first_recovery_v1")
            or new.get("experiment_id") != "task_first_recovery_v1"
            or old.get("semantic_version") != "v3" or new.get("semantic_version") != "v3"
            or source_num_envs(metadata) != 1):
        raise ValueError("task-first reward requires existing same372 v3 N1 and its isolated namespace")
    canonical = policy_contract(HISTORY_TEMPERED_POLICY, observation_layout=ROLE_OBSERVATION_LAYOUT)
    if metadata.get("policy_contract") != canonical:
        raise ValueError("task-first recovery preserves the verified tempered HISTORY372 policy")
    variable = {"files", "runtime_content_sha256", "source_git_commit", "selected_configuration", "experiment_id"}
    if ({k: v for k, v in old.items() if k not in variable}
            != {k: v for k, v in new.items() if k not in variable}):
        raise ValueError("task-first reward cannot change physical/runtime metadata")
    delta = sorted(p for p in old["files"].keys() | new["files"].keys()
                   if old["files"].get(p) != new["files"].get(p))
    if sorted(allowed_changed_files) != delta or len(set(allowed_changed_files)) != len(allowed_changed_files):
        raise ValueError("task-first migration requires the exact unique changed-file inventory")
    if old["files"].keys() - new["files"].keys():
        raise ValueError("task-first migration cannot delete runtime files")
    source_namespace = f"configs/ppo_{old['experiment_id']}"
    target_namespace = "configs/ppo_task_first_recovery_v1"
    targets = {f"{target_namespace}/{n}" for n in CONFIG_NAMES}
    code_delta = set(delta) - targets
    if not code_delta <= TASK_FIRST_REVIEW_FILES:
        raise ValueError(f"task-first migration changed protected N/physics/policy files: {sorted(code_delta - TASK_FIRST_REVIEW_FILES)}")
    hashes = review["reviewed_code_sha256"]
    if not isinstance(hashes, Mapping) or dict(hashes) != {p: new["files"][p] for p in sorted(code_delta)}:
        raise ValueError("task-first reviewed code hashes must bind every changed non-config file")
    records = {}
    for side, contract, namespace in (("source", old, source_namespace), ("target", new, target_namespace)):
        if set(contract.get("selected_configuration", {})) != CONFIG_NAMES:
            raise ValueError("task-first requires exactly six selected configuration bindings")
        for name in CONFIG_NAMES:
            relative = f"{namespace}/{name}"
            expected = {"path": relative, "sha256": contract["files"].get(relative)}
            if contract["selected_configuration"][name] != expected or expected["sha256"] is None:
                raise ValueError("task-first configuration binding differs from runtime inventory")
            records.setdefault(name, {})[side] = expected
    for name, row in records.items():
        before_bytes = _version_bytes(project_root, old, row["source"]["path"], prefer_worktree=True)
        after_bytes = _version_bytes(project_root, new, row["target"]["path"], prefer_worktree=True)
        before, after = yaml.safe_load(before_bytes), yaml.safe_load(after_bytes)
        if name == "reward_config.yaml":
            expected = copy.deepcopy(before)
            expected.update(revision="task_first_recovery_epsilon_zero_v1",
                            objective_profile="task_first_recovery_v1", quality_epsilon=0.0)
            for family in ("body_stability", "contact_motion_quality", "control_smoothness", "control_regularization"):
                expected["family_weights"][family] = 0.0
            if after != expected or before.get("family_weights", {}).get("control_regularization") != 0.0:
                raise ValueError("task-first permits only explicit epsilon-zero quality weights and profile markers")
        elif before != after:
            raise ValueError(f"task-first cannot change non-reward configuration semantics: {name}")
        row["bytes_identical"] = before_bytes == after_bytes
        row["semantics_identical"] = before == after
        row["serialization_only_change"] = before == after and before_bytes != after_bytes
    for relative in delta:
        if file_sha(project_root / relative) != new["files"][relative]:
            raise ValueError("task-first target bytes differ from current inventory")
    observation = {"source_policy_contract": canonical, "target_policy_contract": dict(canonical),
        "observation_layout": ROLE_OBSERVATION_LAYOUT, "observation_dimension": 372,
        "action_dimension": 12, "num_envs": 1, "parameter_mapping": "identity_all_parameters_and_buffers"}
    factor = {"schema": TASK_FIRST_REWARD_SCHEMA, "review_reason": review["reason"].strip(),
        "reviewed_code_sha256": dict(hashes), "configuration_bindings": records,
        "reward_changed": records["reward_config.yaml"]["semantics_identical"] is False,
        "physical_mdp_changed": False, "nominal_control_changed": False, "task_acceptance_changed": False,
        "physical_actuators_changed": False, "action_ranges_changed": False, "kernel_changed": False,
        "observation_semantics_changed": [], "observation_contract": observation,
        "normalizers": "preserve_verified_identity_RSL_state",
        "optimizer": "preserve_complete_verified_Adam_state_and_effective_learning_rate",
        "critic": "preserve_parameters_and_Adam_then_refit_only_on_new_objective_on_policy_data; old_values_not_new_objective_truth",
        "quality_epsilon": 0.0, "old_rollout_inherited": False, "migration_added_updates": 0,
        "scope_is_reviewer_assertion_not_semantic_equivalence_proof": True}
    sidecar = checkpoint.with_name(checkpoint.stem + "_manifest.json")
    return {"schema": SCHEMA, "reason": reason.strip(), "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": file_sha(checkpoint), "source_manifest_sha256": file_sha(sidecar),
        "source_contract_sha256": digest(old), "target_contract_sha256": digest(new),
        "source_git_commit": old["source_git_commit"], "target_git_commit": new["source_git_commit"],
        "allowed_changed_files": delta,
        "changed_file_hashes": {p: {"before": old["files"].get(p), "after": new["files"][p]} for p in delta},
        "geometric_factor": None, "observation_dimension": 372, "action_dimension": 12,
        "preserve_actor_critic_optimizer_normalizer_rng_and_budget": True,
        "discard_old_rollout_storage": True, "physics_resume": "fresh_legal_P01_reset",
        "task_first_reward_factor": factor}


def _build_execution_composition_plan(checkpoint, metadata, old, new, *,
        allowed_changed_files, reason, review, project_root):
    """Fix residual double-application; retain tensors, not trajectory equivalence."""
    import yaml
    from .semantic_policy_distribution import CONFIG_NAMES, HISTORY_TEMPERED_POLICY, policy_contract
    from .semantic_transfer_roles import ROLE_OBSERVATION_LAYOUT
    if (not isinstance(review, Mapping) or set(review) != {"reason", "reviewed_code_sha256"}
            or not isinstance(review["reason"], str) or not review["reason"].strip()
            or not isinstance(reason, str) or not reason.strip()):
        raise ValueError("execution composition requires explicit reason and exact reviewed hashes")
    if (old.get("experiment_id") != "task_first_recovery_v1"
            or new.get("experiment_id") != old.get("experiment_id")
            or old.get("semantic_version") != "v3" or new.get("semantic_version") != "v3"
            or source_num_envs(metadata) != 1):
        raise ValueError("execution composition requires same task-first v3 N1")
    canonical = policy_contract(HISTORY_TEMPERED_POLICY, observation_layout=ROLE_OBSERVATION_LAYOUT)
    if metadata.get("policy_contract") != canonical:
        raise ValueError("execution composition preserves canonical tempered HISTORY372/12")
    variable = {"files", "runtime_content_sha256", "source_git_commit"}
    if ({k: v for k, v in old.items() if k not in variable}
            != {k: v for k, v in new.items() if k not in variable}):
        raise ValueError("execution composition cannot change physical metadata or configuration bindings")
    if old["files"].keys() != new["files"].keys():
        raise ValueError("execution composition cannot add or remove runtime files")
    delta = sorted(p for p in old["files"] if old["files"][p] != new["files"][p])
    if sorted(allowed_changed_files) != delta or len(set(allowed_changed_files)) != len(allowed_changed_files):
        raise ValueError("execution composition requires the exact unique changed-file inventory")
    if not set(delta) <= EXECUTION_COMPOSITION_FILES or not EXECUTION_COMPOSITION_CORE_FILES <= set(delta):
        raise ValueError("execution composition requires its reviewed core fix and cannot change protected files")
    hashes = review["reviewed_code_sha256"]
    if not isinstance(hashes, Mapping) or dict(hashes) != {p: new["files"][p] for p in delta}:
        raise ValueError("execution composition hashes must bind every changed runtime file")
    bindings = old.get("selected_configuration", {})
    if set(bindings) != CONFIG_NAMES:
        raise ValueError("execution composition requires all six unchanged configuration bindings")
    for name, row in bindings.items():
        relative = f"configs/ppo_task_first_recovery_v1/{name}"
        if row != {"path": relative, "sha256": old["files"].get(relative)} or row["sha256"] is None:
            raise ValueError("execution composition configuration binding differs from inventory")
        if (_version_bytes(project_root, old, relative, prefer_worktree=True)
                != _version_bytes(project_root, new, relative, prefer_worktree=True)):
            raise ValueError("execution composition requires byte-identical configurations")
    for relative in delta:
        _version_bytes(project_root, old, relative, prefer_worktree=True)
        if file_sha(project_root / relative) != new["files"][relative]:
            raise ValueError("execution composition target bytes differ from current inventory")
    reward = yaml.safe_load(_version_bytes(project_root, new,
        bindings["reward_config.yaml"]["path"], prefer_worktree=True))
    if (reward.get("objective_profile") != "task_first_recovery_v1" or reward.get("quality_epsilon") != 0.0
            or any(reward.get("family_weights", {}).get(name) != 0.0 for name in (
                "body_stability", "contact_motion_quality", "control_smoothness", "control_regularization"))):
        raise ValueError("execution composition requires the unchanged task-first epsilon-zero objective")
    observation = {"source_policy_contract": canonical, "target_policy_contract": dict(canonical),
        "observation_layout": ROLE_OBSERVATION_LAYOUT, "observation_dimension": 372,
        "action_dimension": 12, "num_envs": 1, "parameter_mapping": "identity_all_parameters_and_buffers"}
    factor = {"schema": EXECUTION_COMPOSITION_SCHEMA, "review_reason": review["reason"].strip(),
        "reviewed_code_sha256": dict(hashes), "configuration_bindings": dict(bindings),
        "action_execution_changed": True, "transition_execution_semantics_changed": True,
        "nominal_history_semantics": "same_actual_state_nominal_command_history_excludes_policy_v1",
        "nominal_history_initialization": "fresh_legal_reset_then_existing_settle_or_zero_bootstrap; not_old_combined_final",
        "nominal_history_resets_on_phase_transition": False,
        "physical_scene_changed": False, "actuator_capability_changed": False,
        "nominal_source_schedule_changed": False, "task_acceptance_changed": False,
        "reward_changed": False, "quality_epsilon": 0.0, "action_ranges_changed": False,
        "kernel_changed": False, "observation_semantics_changed": [], "observation_contract": observation,
        "normalizers": "preserve_verified_identity_RSL_state",
        "optimizer": "preserve_complete_verified_Adam_state_and_effective_learning_rate",
        "critic": "preserve_parameters_and_Adam_then_refit_on_corrected_execution_on_policy_data",
        "training_rng": "preserve_verified_training_rng", "lifetime_counters_preserved": True,
        "old_rollout_inherited": False, "migration_added_updates": 0,
        "trajectory_equivalence_claimed": False,
        "scope_is_reviewer_assertion_not_semantic_equivalence_proof": True}
    sidecar = checkpoint.with_name(checkpoint.stem + "_manifest.json")
    return {"schema": SCHEMA, "reason": reason.strip(), "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": file_sha(checkpoint), "source_manifest_sha256": file_sha(sidecar),
        "source_contract_sha256": digest(old), "target_contract_sha256": digest(new),
        "source_git_commit": old["source_git_commit"], "target_git_commit": new["source_git_commit"],
        "allowed_changed_files": delta,
        "changed_file_hashes": {p: {"before": old["files"][p], "after": new["files"][p]} for p in delta},
        "geometric_factor": None, "observation_dimension": 372, "action_dimension": 12,
        "preserve_actor_critic_optimizer_normalizer_rng_and_budget": True,
        "discard_old_rollout_storage": True, "physics_resume": "fresh_legal_P01_reset",
        "execution_composition_factor": factor}


def _final_stop_owner_scope(before: bytes, after: bytes) -> dict[str, Any]:
    """AST-locate exactly one owner method; keep evaluator/other source identical."""
    import ast

    def split(raw):
        text = raw.decode("utf-8").replace("\r\n", "\n")
        tree = ast.parse(text)
        classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "NominalMotionProvider"]
        if len(classes) != 1:
            raise ValueError("final stop handoff requires one NominalMotionProvider")
        methods = [n for n in classes[0].body if isinstance(n, ast.FunctionDef)
                   and n.name == "_observe_final_stop_owner"]
        if len(methods) != 1 or methods[0].decorator_list:
            raise ValueError("final stop handoff requires one undecorated stop-owner method")
        method = methods[0]
        lines = text.splitlines(keepends=True)
        outside = "".join(lines[:method.lineno-1]) + "<reviewed stop-owner method>\n" + "".join(lines[method.end_lineno:])
        body = "".join(lines[method.lineno-1:method.end_lineno])
        signature = (ast.dump(method.args), ast.dump(method.returns) if method.returns else None)
        return outside, body, signature

    old_outside, old_method, old_signature = split(before)
    new_outside, new_method, new_signature = split(after)
    if old_outside != new_outside or old_signature != new_signature:
        raise ValueError("final stop handoff changed evaluator or code outside the exact stop-owner method")
    if old_method == new_method:
        raise ValueError("final stop handoff requires an actual owner-method change")
    sha = lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()
    return {"method": "NominalMotionProvider._observe_final_stop_owner",
        "unchanged_outside_method_sha256": sha(old_outside),
        "source_method_sha256": sha(old_method), "target_method_sha256": sha(new_method),
        "method_signature_unchanged": True, "evaluator_and_other_methods_unchanged": True}


def _build_final_stop_handoff_plan(checkpoint, metadata, old, new, *,
        allowed_changed_files, reason, review, project_root):
    """Repair final-stop ownership, not the evaluator or task success criteria."""
    import yaml
    from .semantic_policy_distribution import CONFIG_NAMES, HISTORY_TEMPERED_POLICY, policy_contract
    from .semantic_transfer_roles import ROLE_OBSERVATION_LAYOUT
    if (not isinstance(review, Mapping) or set(review) != {"reason", "reviewed_code_sha256"}
            or not isinstance(review["reason"], str) or not review["reason"].strip()
            or not isinstance(reason, str) or not reason.strip()):
        raise ValueError("final stop handoff requires explicit reason and exact reviewed hashes")
    if (old.get("experiment_id") != "task_first_recovery_v1"
            or new.get("experiment_id") != old.get("experiment_id")
            or old.get("semantic_version") != "v3" or new.get("semantic_version") != "v3"
            or source_num_envs(metadata) != 1):
        raise ValueError("final stop handoff requires same task-first v3 N1")
    canonical = policy_contract(HISTORY_TEMPERED_POLICY, observation_layout=ROLE_OBSERVATION_LAYOUT)
    if metadata.get("policy_contract") != canonical:
        raise ValueError("final stop handoff preserves canonical tempered HISTORY372/12")
    variable = {"files", "runtime_content_sha256", "source_git_commit"}
    if ({k: v for k, v in old.items() if k not in variable}
            != {k: v for k, v in new.items() if k not in variable}):
        raise ValueError("final stop handoff cannot change physical metadata or configuration bindings")
    if old["files"].keys() != new["files"].keys():
        raise ValueError("final stop handoff cannot add or remove runtime files")
    delta = sorted(p for p in old["files"] if old["files"][p] != new["files"][p])
    if sorted(allowed_changed_files) != delta or len(set(allowed_changed_files)) != len(allowed_changed_files):
        raise ValueError("final stop handoff requires the exact unique changed-file inventory")
    if not set(delta) <= FINAL_STOP_HANDOFF_FILES or not FINAL_STOP_HANDOFF_CORE_FILES <= set(delta):
        raise ValueError("final stop handoff requires its reviewed core fix and cannot change protected files")
    hashes = review["reviewed_code_sha256"]
    if not isinstance(hashes, Mapping) or dict(hashes) != {p: new["files"][p] for p in delta}:
        raise ValueError("final stop handoff hashes must bind every changed runtime file")
    bindings = old.get("selected_configuration", {})
    if set(bindings) != CONFIG_NAMES:
        raise ValueError("final stop handoff requires all six unchanged configuration bindings")
    for name, row in bindings.items():
        relative = f"configs/ppo_task_first_recovery_v1/{name}"
        if row != {"path": relative, "sha256": old["files"].get(relative)} or row["sha256"] is None:
            raise ValueError("final stop handoff configuration binding differs from inventory")
        if (_version_bytes(project_root, old, relative, prefer_worktree=True)
                != _version_bytes(project_root, new, relative, prefer_worktree=True)):
            raise ValueError("final stop handoff requires byte-identical configurations")
    for relative in delta:
        _version_bytes(project_root, old, relative, prefer_worktree=True)
        if file_sha(project_root / relative) != new["files"][relative]:
            raise ValueError("final stop handoff target bytes differ from current inventory")
    scope = _final_stop_owner_scope(
        _version_bytes(project_root, old, SUPERVISOR, prefer_worktree=True),
        _version_bytes(project_root, new, SUPERVISOR, prefer_worktree=True))
    reward = yaml.safe_load(_version_bytes(project_root, new,
        bindings["reward_config.yaml"]["path"], prefer_worktree=True))
    if (reward.get("objective_profile") != "task_first_recovery_v1" or reward.get("quality_epsilon") != 0.0
            or any(reward.get("family_weights", {}).get(name) != 0.0 for name in (
                "body_stability", "contact_motion_quality", "control_smoothness", "control_regularization"))):
        raise ValueError("final stop handoff requires the unchanged task-first epsilon-zero objective")
    observation = {"source_policy_contract": canonical, "target_policy_contract": dict(canonical),
        "observation_layout": ROLE_OBSERVATION_LAYOUT, "observation_dimension": 372,
        "action_dimension": 12, "num_envs": 1, "parameter_mapping": "identity_all_parameters_and_buffers"}
    factor = {"schema": FINAL_STOP_HANDOFF_SCHEMA, "review_reason": review["reason"].strip(),
        "reviewed_code_sha256": dict(hashes), "configuration_bindings": dict(bindings),
        "action_execution_changed": True, "transition_execution_semantics_changed": True,
        "stop_owner_acquisition_semantics": "post_window_triggered_nominal_stop_takeover_v2",
        "supervisor_scope": scope, "task_evaluator_changed": False,
        "fixed_post_completion_window_changed": False,
        "residual_composition_changed": False, "controller_history_reset_on_phase_transition": False,
        "physical_scene_changed": False, "actuator_capability_changed": False,
        "recorded_FSM_source_changed": False, "task_acceptance_changed": False,
        "reward_changed": False, "quality_epsilon": 0.0, "action_ranges_changed": False,
        "kernel_changed": False, "observation_semantics_changed": [], "observation_contract": observation,
        "normalizers": "preserve_verified_identity_RSL_state",
        "optimizer": "preserve_complete_verified_Adam_state_and_effective_learning_rate",
        "critic": "preserve_parameters_and_Adam_then_refit_on_corrected_execution_on_policy_data",
        "training_rng": "preserve_verified_training_rng", "lifetime_counters_preserved": True,
        "old_rollout_inherited": False, "migration_added_updates": 0,
        "trajectory_equivalence_claimed": False,
        "scope_is_reviewer_assertion_not_semantic_equivalence_proof": True}
    sidecar = checkpoint.with_name(checkpoint.stem + "_manifest.json")
    return {"schema": SCHEMA, "reason": reason.strip(), "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": file_sha(checkpoint), "source_manifest_sha256": file_sha(sidecar),
        "source_contract_sha256": digest(old), "target_contract_sha256": digest(new),
        "source_git_commit": old["source_git_commit"], "target_git_commit": new["source_git_commit"],
        "source_runtime_content_sha256": old["runtime_content_sha256"],
        "target_runtime_content_sha256": new["runtime_content_sha256"],
        "allowed_changed_files": delta,
        "changed_file_hashes": {p: {"before": old["files"][p], "after": new["files"][p]} for p in delta},
        "geometric_factor": None, "observation_dimension": 372, "action_dimension": 12,
        "preserve_actor_critic_optimizer_normalizer_rng_and_budget": True,
        "discard_old_rollout_storage": True, "physics_resume": "fresh_legal_P01_reset",
        "final_stop_handoff_factor": factor}


def apply_task_recovery_mean_head(runner: Any, source_infos: Mapping[str, Any], *,
        branch_id: str, reason: str) -> dict[str, Any]:
    """Explicit in-memory branch initialization after verified checkpoint loading.

    Does not save a checkpoint or perform a forward/optimizer/physics step. The
    caller publishes under a new immutable name, then trains fresh on-policy
    data. Absolute lifetime counters stay intact; branch counters use the
    recorded origin. Adam's scalar step belongs to the shared mean/sigma tensor
    and must remain intact: only mean-row first/second moments are cleared.
    """
    import copy
    import re
    import torch
    from .semantic_policy_distribution import HISTORY_TEMPERED_POLICY, policy_contract
    from .semantic_transfer_roles import ROLE_OBSERVATION_LAYOUT
    from .semantic_training import parameter_hash, state_hash
    if (not isinstance(branch_id, str) or re.fullmatch(r"[a-z][a-z0-9_]{2,63}", branch_id) is None
            or not isinstance(reason, str) or not reason.strip()):
        raise ValueError("mean-head branch requires an explicit safe branch id and reason")
    if source_infos.get("task_recovery_branch") is not None:
        raise ValueError("mean-head recovery must not silently reinitialize an existing recovery branch")
    canonical = policy_contract(HISTORY_TEMPERED_POLICY, observation_layout=ROLE_OBSERVATION_LAYOUT)
    if (source_infos.get("policy_contract") != canonical
            or source_infos.get("runtime_contract", {}).get("experiment_id") != "task_first_recovery_v1"
            or getattr(runner, "_semantic_policy_version", None) != HISTORY_TEMPERED_POLICY
            or getattr(runner, "_semantic_observation_layout", None) != ROLE_OBSERVATION_LAYOUT
            or runner.alg.storage.step != 0 or runner.alg.transition.actions is not None):
        raise ValueError("mean-head recovery requires a loaded task-first HISTORY372 checkpoint and fresh rollout")
    counters = {key: source_infos.get(key) for key in ("global_policy_decisions", "ppo_updates", "optimizer_steps")}
    if any(type(value) is not int or value < 1 for value in counters.values()):
        raise ValueError("mean-head recovery requires explicit positive ancestry counters")
    source = source_infos.get("resume_source_checkpoint")
    if not isinstance(source, Mapping) or not source.get("checkpoint_sha256") or not source.get("manifest_sha256"):
        raise ValueError("mean-head recovery requires the verified immutable source checkpoint binding")
    actor, critic, optimizer = runner.alg.actor, runner.alg.critic, runner.alg.optimizer
    if type(optimizer) is not torch.optim.Adam:
        raise ValueError("mean-head recovery only supports the existing verified Adam")
    before_actor = parameter_hash(actor)
    before_critic = parameter_hash(critic)
    before_optimizer = state_hash(optimizer.state_dict())
    if (before_actor != source_infos.get("actor_parameter_sha256")
            or before_critic != source_infos.get("critic_parameter_sha256")
            or before_optimizer != source_infos.get("optimizer_state_sha256")):
        raise ValueError("mean-head recovery requires exact source actor/critic/Adam hashes")
    named = dict(actor.named_parameters())
    keys = ("mlp.4.weight", "mlp.4.bias")
    if (any(key not in named for key in keys) or tuple(named[keys[0]].shape) != (24, 256)
            or tuple(named[keys[1]].shape) != (24,)):
        raise ValueError("mean-head recovery requires the existing 24-row 256-feature output layer")
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    if len({id(p) for p in parameters}) != len(parameters):
        raise ValueError("mean-head recovery cannot map duplicate optimizer parameters")
    old_optimizer = optimizer.state_dict()
    saved_ids = [p for group in old_optimizer["param_groups"] for p in group["params"]]
    parameter_ids = {id(p): saved_id for p, saved_id in zip(parameters, saved_ids, strict=True)}
    mapped_actor = copy.deepcopy(actor.state_dict())
    mapped_optimizer = copy.deepcopy(old_optimizer)
    resets = []
    for key in keys:
        parameter = named[key]
        if id(parameter) not in parameter_ids:
            raise ValueError("mean-head parameter is absent from Adam")
        saved_id = parameter_ids[id(parameter)]
        state = mapped_optimizer["state"].get(saved_id)
        if not isinstance(state, dict) or not {"step", "exp_avg", "exp_avg_sq"} <= set(state):
            raise ValueError("mean-head recovery requires existing Adam moment state")
        moment_keys = set(state) - {"step"}
        if not moment_keys <= {"exp_avg", "exp_avg_sq", "max_exp_avg_sq"}:
            raise ValueError("mean-head recovery encountered unsupported Adam state")
        step = state["step"]
        if (not isinstance(step, torch.Tensor) or step.numel() != 1
                or not bool(torch.isfinite(step).all()) or float(step) <= 0):
            raise ValueError("mean-head recovery requires finite positive shared Adam step")
        if not bool(torch.isfinite(mapped_actor[key]).all()):
            raise ValueError("mean-head source parameters must be finite")
        mapped_actor[key][:12].zero_()
        for moment in sorted(moment_keys):
            value = state[moment]
            if not isinstance(value, torch.Tensor) or value.shape != parameter.shape or not bool(torch.isfinite(value).all()):
                raise ValueError("mean-head Adam moment dimensions/finiteness differ")
            value[:12].zero_()
        resets.append({"parameter": key, "mean_rows": [0, 12], "preserved_log_sigma_rows": [12, 24],
                       "cleared_moment_fields": sorted(moment_keys), "shared_Adam_step_preserved": float(step)})
    # All checks and both transforms finish before either live object is changed.
    actor.load_state_dict(mapped_actor, strict=True)
    optimizer.load_state_dict(mapped_optimizer)
    if state_hash(actor.state_dict()) != state_hash(mapped_actor) or state_hash(optimizer.state_dict()) != state_hash(mapped_optimizer):
        raise RuntimeError("mean-head branch failed exact in-memory transformed-state round trip")
    branch = {"schema": TASK_RECOVERY_BRANCH_SCHEMA, "branch_id": branch_id, "reason": reason.strip(),
        "source_checkpoint": copy.deepcopy(dict(source)), "counter_origin": counters,
        "branch_policy_decisions": 0, "branch_ppo_updates": 0, "branch_optimizer_steps": 0,
        "branch_counter_semantics": "absolute_checkpoint_counters_minus_origin_lifetime_counters",
        "old_model_and_old_optimizer_preserved_on_disk": True,
        "actor_feature_parameters": "preserved_exactly", "mean_output_parameters": "zero_first12_rows_only",
        "log_sigma_parameters": "preserved_exactly_not_exp0", "changed_parameter_rows": resets,
        "optimizer": "preserve_all_other_states_and_groups; clear_only_mean_row_moments",
        "Adam_shared_scalar_step": "preserved_for_sigma_and_shared_parameter; mean_moment_bias_correction_uses_existing_age",
        "critic": "preserve_parameters_and_Adam_then_refit_only_on_new_reward_on_policy_data; source_values_not_ground_truth",
        "history": "kernel_unchanged; fresh_legal_episode_reset_has_zero_previous_raw",
        "normalizer": "identity_state_preserved", "training_rng": "preserved_no_sampling_during_initialization",
        "old_rollout_inherited": False, "all12_channels_remain_learnable": True,
        "initialization_is_not_PPO_learning_success": True,
        "actor_parameter_sha256_before": before_actor, "actor_parameter_sha256_after": parameter_hash(actor),
        "critic_parameter_sha256_before": before_critic, "critic_parameter_sha256_after": parameter_hash(critic),
        "optimizer_state_sha256_before": before_optimizer, "optimizer_state_sha256_after": state_hash(optimizer.state_dict())}
    return {**copy.deepcopy(dict(source_infos)), "stage": "initial_task_recovery_mean_head",
            "task_recovery_branch": branch}


def _build_rr_physical_acceptance_plan(checkpoint, metadata, old, new, *,
        allowed_changed_files, reason, review, project_root):
    """One reviewed task/nominal semantic boundary; no tensor or kernel change.

    Exact file receipts express the reviewer's scope, not a claim that differing
    source code is equivalent. Old migration factors retain their own rules.
    """
    import copy
    import yaml
    from .semantic_policy_distribution import CONFIG_NAMES, HISTORY_QUARTER_TEMPERED_POLICY, policy_contract
    from .semantic_transfer_roles import ROLE_OBSERVATION_LAYOUT
    from .semantic_training import semantic_runner_config
    from .semantic_return_profile import runner_return_profile
    if (not isinstance(review, Mapping) or set(review) != {"reason", "reviewed_code_sha256"}
            or not isinstance(review["reason"], str) or not review["reason"].strip()
            or not isinstance(reason, str) or not reason.strip()):
        raise ValueError("RR acceptance requires an explicit reason and exact reviewed hashes")
    if (old.get("experiment_id") != "task_first_recovery_v1"
            or new.get("experiment_id") != "residual_rr_fix_v1"
            or old.get("semantic_version") != "v3" or new.get("semantic_version") != "v3"
            or source_num_envs(metadata) != 1):
        raise ValueError("RR acceptance requires task-first to isolated RR-fix v3 N1")
    canonical = policy_contract(HISTORY_QUARTER_TEMPERED_POLICY, observation_layout=ROLE_OBSERVATION_LAYOUT)
    if metadata.get("policy_contract") != canonical:
        raise ValueError("RR acceptance preserves the complete quarter HISTORY372 full12 policy")
    horizon = runner_return_profile(metadata["runner_config"], semantic_version="v3")["version"]
    config = semantic_runner_config(seed=metadata["seed"], device=metadata["runner_config"]["device"],
        semantic_version="v3", policy_version=HISTORY_QUARTER_TEMPERED_POLICY,
        observation_layout=ROLE_OBSERVATION_LAYOUT, return_profile=horizon)
    if metadata["runner_config"] != config:
        raise ValueError("RR acceptance cannot change PPO hyperparameters or normalization")
    variable = {"files", "runtime_content_sha256", "source_git_commit", "selected_configuration", "experiment_id"}
    if ({k: v for k, v in old.items() if k not in variable}
            != {k: v for k, v in new.items() if k not in variable}):
        raise ValueError("RR acceptance cannot change frozen physical/runtime metadata")
    delta = sorted(p for p in old["files"].keys() | new["files"].keys()
                   if old["files"].get(p) != new["files"].get(p))
    if sorted(allowed_changed_files) != delta or len(set(allowed_changed_files)) != len(allowed_changed_files):
        raise ValueError("RR acceptance requires the exact unique changed-file inventory")
    if old["files"].keys() - new["files"].keys():
        raise ValueError("RR acceptance cannot delete runtime files")
    source_namespace, target_namespace = "configs/ppo_task_first_recovery_v1", "configs/ppo_residual_rr_fix_v1"
    zero_namespace = "configs/ppo_non_residual_refine_v1"
    targets = {f"{target_namespace}/{name}" for name in CONFIG_NAMES}
    zero_configs = {f"{zero_namespace}/{name}" for name in CONFIG_NAMES}
    code_delta = set(delta) - targets - zero_configs
    if not code_delta <= RR_PHYSICAL_ACCEPTANCE_FILES or SUPERVISOR not in code_delta:
        raise ValueError(f"RR acceptance changed protected files or lacks core fix: {sorted(code_delta - RR_PHYSICAL_ACCEPTANCE_FILES)}")
    hashes = review["reviewed_code_sha256"]
    if not isinstance(hashes, Mapping) or dict(hashes) != {p: new["files"][p] for p in sorted(code_delta)}:
        raise ValueError("RR acceptance reviewed hashes must bind every changed code file")
    records = {}
    for side, contract, namespace in (("source", old, source_namespace), ("target", new, target_namespace)):
        if set(contract.get("selected_configuration", {})) != CONFIG_NAMES:
            raise ValueError("RR acceptance requires exactly six selected configuration bindings")
        for name in CONFIG_NAMES:
            relative = f"{namespace}/{name}"
            expected = {"path": relative, "sha256": contract["files"].get(relative)}
            if contract["selected_configuration"][name] != expected or expected["sha256"] is None:
                raise ValueError("RR acceptance configuration binding differs from inventory")
            records.setdefault(name, {})[side] = expected
    zero_records = {}
    if zero_configs & new["files"].keys() and not zero_configs <= new["files"].keys():
        raise ValueError("RR acceptance cannot inherit a partial zero configuration")
    for name, row in records.items():
        before_bytes = _version_bytes(project_root, old, row["source"]["path"], prefer_worktree=True)
        after_bytes = _version_bytes(project_root, new, row["target"]["path"], prefer_worktree=True)
        before, after = yaml.safe_load(before_bytes), yaml.safe_load(after_bytes)
        if name == "stage_task_spec.yaml":
            if (before.get("p09_lift_semantics") != "functional_lift_edge_v2"
                    or before.get("physical_acceptance_version") != "all_stage_v1"
                    or before["nominal"].get("final_stop_owner") != "current_physical_stop_nominal_owner_v1"
                    or "rr_carry_source_semantics" in before["nominal"]):
                raise ValueError("RR acceptance requires the declared original lift/source/stop semantics")
            expected = copy.deepcopy(before)
            expected.update(revision="residual_rr_fix_v1", p09_lift_semantics="functional_free_air_lift_v3")
            expected["nominal"].update(rr_carry_source_semantics="current_free_lift_before_pending_knee_and_roll_v1",
                final_stop_owner="source_home_after_physical_stop_v2")
            if after != expected:
                raise ValueError("RR acceptance permits only the four declared stage-spec fields")
        elif before_bytes != after_bytes:
            raise ValueError(f"RR acceptance requires byte-identical non-stage configuration: {name}")
        row.update(bytes_identical=before_bytes == after_bytes, semantics_identical=before == after)
        # These six public zero files were introduced after the source checkpoint.
        # Bind their narrow existing home-mode delta instead of ignoring all new configs.
        relative = f"{zero_namespace}/{name}"
        if relative in new["files"]:
            zero_bytes = _version_bytes(project_root, new, relative, prefer_worktree=True)
            expected_zero = copy.deepcopy(before)
            if name == "stage_task_spec.yaml":
                expected_zero["nominal"]["final_stop_owner"] = "source_home_after_physical_stop_v2"
                if yaml.safe_load(zero_bytes) != expected_zero:
                    raise ValueError("RR acceptance inherited zero profile changed beyond home mode")
            elif zero_bytes != before_bytes:
                raise ValueError("RR acceptance inherited zero non-stage config must remain identical")
            zero_records[name] = {"path": relative, "sha256": new["files"][relative]}
    for relative in delta:
        if relative in old["files"]:
            _version_bytes(project_root, old, relative, prefer_worktree=True)
        if file_sha(project_root / relative) != new["files"][relative]:
            raise ValueError("RR acceptance target bytes differ from current inventory")
    reward = yaml.safe_load(_version_bytes(project_root, new, records["reward_config.yaml"]["target"]["path"], prefer_worktree=True))
    if (reward.get("objective_profile") != "task_first_recovery_v1" or reward.get("quality_epsilon") != 0.0
            or any(reward.get("family_weights", {}).get(name) != 0.0 for name in (
                "body_stability", "contact_motion_quality", "control_smoothness", "control_regularization"))):
        raise ValueError("RR acceptance retains the unchanged task-first epsilon-zero reward")
    origin = {key: metadata[key] for key in ("global_policy_decisions", "ppo_updates", "optimizer_steps")}
    observation = {"source_policy_contract": canonical, "target_policy_contract": dict(canonical),
        "observation_layout": ROLE_OBSERVATION_LAYOUT, "observation_dimension": 372,
        "action_dimension": 12, "num_envs": 1, "parameter_mapping": "identity_all_parameters_and_buffers"}
    factor = {"schema": RR_PHYSICAL_ACCEPTANCE_SCHEMA, "review_reason": review["reason"].strip(),
        "branch_id": "residual_rr_fix_v1", "counter_origin": origin,
        "reviewed_code_sha256": dict(hashes), "configuration_bindings": records,
        "inherited_zero_configuration_bindings": zero_records,
        "source_runner_config": config, "target_runner_config": copy.deepcopy(config),
        "p09_lift_semantics": "functional_free_air_lift_v3",
        "rr_carry_source_semantics": "current_free_lift_before_pending_knee_and_roll_v1",
        "final_stop_owner": "source_home_after_physical_stop_v2",
        "task_acceptance_changed": True, "nominal_control_changed": True,
        "action_execution_changed": True, "physical_mdp_changed": True,
        "physical_scene_changed": False, "actuator_capability_changed": False,
        "recorded_FSM_source_changed": False, "nominal_geometry_changed": False,
        "action_ranges_changed": False, "kernel_changed": False,
        "reward_code_and_weights_changed": False, "task_reward_event_semantics_changed": True,
        "quality_epsilon": 0.0, "observation_layout_changed": False,
        "observation_semantics_changed": ["RR_current_free_air_lift_and_qualified_history",
            "task_goal_and_completed_stage_derived_values", "transfer_role_current_lift_readiness",
            "nominal_source_carry_and_terminal_action_values"],
        "observation_contract": observation,
        "controller_history_reset_on_phase_transition": False,
        "normalizers": "preserve_verified_identity_RSL_state",
        "optimizer": "preserve_complete_verified_Adam_state_and_effective_learning_rate",
        "critic": "preserve_parameters_and_Adam_then_refit_on_new_task_semantics_on_policy_data",
        "training_rng": "preserve_verified_training_rng", "lifetime_counters_preserved": True,
        "old_rollout_inherited": False, "migration_added_updates": 0,
        "video_camera_scope": "viewport_only_same_review_camera_as_isolated_zero",
        "trajectory_equivalence_claimed": False,
        "scope_is_reviewer_assertion_not_semantic_equivalence_proof": True}
    sidecar = checkpoint.with_name(checkpoint.stem + "_manifest.json")
    return {"schema": SCHEMA, "reason": reason.strip(), "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": file_sha(checkpoint), "source_manifest_sha256": file_sha(sidecar),
        "source_contract_sha256": digest(old), "target_contract_sha256": digest(new),
        "source_git_commit": old["source_git_commit"], "target_git_commit": new["source_git_commit"],
        "source_runtime_content_sha256": old["runtime_content_sha256"],
        "target_runtime_content_sha256": new["runtime_content_sha256"],
        "allowed_changed_files": delta,
        "changed_file_hashes": {p: {"before": old["files"].get(p), "after": new["files"][p]} for p in delta},
        "geometric_factor": None, "observation_dimension": 372, "action_dimension": 12,
        "preserve_actor_critic_optimizer_normalizer_rng_and_budget": True,
        "discard_old_rollout_storage": True, "physics_resume": "fresh_legal_P01_reset",
        "rr_physical_acceptance_same372_factor": factor}


def _build_fl_capture_quality_plan(checkpoint, metadata, old, new, *,
        allowed_changed_files, reason, review, project_root):
    """Explicit reward/potential-feature boundary; preserve the learned kernel."""
    import ast
    import copy
    import yaml
    from .semantic_policy_distribution import CONFIG_NAMES, HISTORY_QUARTER_TEMPERED_POLICY, policy_contract
    from .semantic_transfer_roles import ROLE_OBSERVATION_LAYOUT
    from .semantic_training import semantic_runner_config
    from .semantic_return_profile import runner_return_profile
    if (not isinstance(review, Mapping) or set(review) != {"reason", "reviewed_code_sha256"}
            or not isinstance(review["reason"], str) or not review["reason"].strip()
            or not isinstance(reason, str) or not reason.strip()):
        raise ValueError("FL capture requires an explicit reason and exact reviewed hashes")
    if (old.get("experiment_id") != "residual_rr_fix_v1"
            or new.get("experiment_id") != "fl_capture_quality_v1"
            or old.get("semantic_version") != "v3" or new.get("semantic_version") != "v3"
            or source_num_envs(metadata) != 1):
        raise ValueError("FL capture requires isolated RR-fix to FL-quality v3 N1")
    canonical = policy_contract(HISTORY_QUARTER_TEMPERED_POLICY, observation_layout=ROLE_OBSERVATION_LAYOUT)
    horizon = runner_return_profile(metadata["runner_config"], semantic_version="v3")["version"]
    config = semantic_runner_config(seed=metadata["seed"], device=metadata["runner_config"]["device"],
        semantic_version="v3", policy_version=HISTORY_QUARTER_TEMPERED_POLICY,
        observation_layout=ROLE_OBSERVATION_LAYOUT, return_profile=horizon)
    if metadata.get("policy_contract") != canonical or metadata["runner_config"] != config:
        raise ValueError("FL capture preserves quarter HISTORY372 full12 PPO and normalization")
    variable = {"files", "runtime_content_sha256", "source_git_commit", "selected_configuration", "experiment_id"}
    if ({k: v for k, v in old.items() if k not in variable}
            != {k: v for k, v in new.items() if k not in variable}):
        raise ValueError("FL capture cannot change frozen physical/runtime metadata")
    delta = sorted(p for p in old["files"].keys() | new["files"].keys()
                   if old["files"].get(p) != new["files"].get(p))
    if sorted(allowed_changed_files) != delta or len(set(allowed_changed_files)) != len(allowed_changed_files):
        raise ValueError("FL capture requires the exact unique changed-file inventory")
    if old["files"].keys() - new["files"].keys():
        raise ValueError("FL capture cannot delete runtime files")
    source_ns, target_ns = "configs/ppo_residual_rr_fix_v1", "configs/ppo_fl_capture_quality_v1"
    targets = {f"{target_ns}/{name}" for name in CONFIG_NAMES}
    code_delta = set(delta) - targets
    if not code_delta <= FL_CAPTURE_QUALITY_FILES or not {SUPERVISOR, BODY_REWARD_CODE} <= code_delta:
        raise ValueError("FL capture changed protected files or lacks the reviewed reward/potential change")
    hashes = review["reviewed_code_sha256"]
    if not isinstance(hashes, Mapping) or dict(hashes) != {p: new["files"][p] for p in sorted(code_delta)}:
        raise ValueError("FL capture reviewed hashes must bind every changed code file")
    records = {}
    for side, contract, namespace in (("source", old, source_ns), ("target", new, target_ns)):
        if set(contract.get("selected_configuration", {})) != CONFIG_NAMES:
            raise ValueError("FL capture requires all six selected configuration bindings")
        for name in CONFIG_NAMES:
            relative = f"{namespace}/{name}"
            expected = {"path": relative, "sha256": contract["files"].get(relative)}
            if expected["sha256"] is None or contract["selected_configuration"][name] != expected:
                raise ValueError("FL capture selected configuration differs from inventory")
            records.setdefault(name, {})[side] = expected
    for name, row in records.items():
        before_bytes = _version_bytes(project_root, old, row["source"]["path"], prefer_worktree=True)
        after_bytes = _version_bytes(project_root, new, row["target"]["path"], prefer_worktree=True)
        before, after = yaml.safe_load(before_bytes), yaml.safe_load(after_bytes)
        expected = copy.deepcopy(before)
        if name == "stage_task_spec.yaml":
            if (before.get("p09_lift_semantics") != "functional_free_air_lift_v3"
                    or before.get("capture_approach_semantics") != "post_cross_current_surface_proximity_plus_real_contact_v1"
                    or before["nominal"].get("final_stop_owner") != "source_home_after_physical_stop_v2"):
                raise ValueError("FL capture source lacks the preserved current RR/home/capture semantics")
            expected.update(revision="fl_capture_quality_v1",
                capture_approach_semantics="post_cross_FL_multiscale_positive_gap_plus_real_contact_v1",
                fl_capture_potential={"coarse_gap_scale_m": .025, "fine_gap_scale_m": .003, "fine_fraction": .5})
        elif name == "reward_config.yaml":
            if before.get("objective_profile") != "task_first_recovery_v1" or before.get("quality_epsilon") != 0.:
                raise ValueError("FL capture source is not the preserved task-first reward")
            expected.update(revision="fl_capture_front_body_quality_v1", objective_profile="fl_capture_front_body_quality_v1",
                quality_epsilon=.03, euler_rate_scale_rad_s=.5,
                front_body_quality={"phases": ["P01", "P02"], "transfer_weight_floor": .5,
                    "attitude_fraction": .5, "rate_fraction": .5, "sample_audit": True})
            expected["family_weights"]["body_stability"] = .03
            expected["signal_ownership"]["body_stability"] = ["gravity_attitude", "euler_roll_pitch_derivative"]
        elif before_bytes != after_bytes:
            raise ValueError(f"FL capture requires byte-identical non-reward/task configuration: {name}")
        if after != expected:
            raise ValueError(f"FL capture configuration exceeds reviewed changes: {name}")
        row.update(bytes_identical=before_bytes == after_bytes, semantics_identical=before == after)
    # Protect the actual task evaluator and nominal controller, not just a flag.
    def without_mode_constants(text):
        tree = ast.parse(text)
        tree.body = [n for n in tree.body if not (isinstance(n, ast.Assign) and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name) and n.targets[0].id in {"FL_CAPTURE_APPROACH_MODE", "CAPTURE_APPROACH_MODES"})]
        return ast.unparse(tree)
    scope = _height_source_scope(
        without_mode_constants(_version_text(project_root, old, SUPERVISOR, prefer_worktree=True)),
        without_mode_constants(_version_text(project_root, new, SUPERVISOR, prefer_worktree=True)),
        functions=("_capture_approach_enabled",),
        methods=("TaskStageSupervisor.physical_potential", "TaskStageSupervisor._current_capture_progress"))
    prefix_path = "src/wlr50_clean/ppo/semantic_checkpoint_prefix_policy.py"
    prefix_scope = None
    if prefix_path in code_delta:
        prefix_scope = _height_source_scope(
            _version_text(project_root, old, prefix_path, prefer_worktree=True),
            _version_text(project_root, new, prefix_path, prefer_worktree=True),
            methods=("FrozenCheckpointPrefixPolicy.__init__",))
    for relative in delta:
        if relative in old["files"]:
            _version_bytes(project_root, old, relative, prefer_worktree=True)
        if file_sha(project_root / relative) != new["files"][relative]:
            raise ValueError("FL capture target bytes differ from current inventory")
    origin = {key: metadata[key] for key in ("global_policy_decisions", "ppo_updates", "optimizer_steps")}
    factor = {"schema": FL_CAPTURE_QUALITY_SCHEMA, "review_reason": review["reason"].strip(),
        "branch_id": "fl_capture_quality_v1", "counter_origin": origin,
        "reviewed_code_sha256": dict(hashes), "configuration_bindings": records,
        "supervisor_scope": scope, "source_runner_config": config, "target_runner_config": copy.deepcopy(config),
        "task_acceptance_changed": False, "nominal_control_changed": False, "action_execution_changed": False,
        "physical_scene_changed": False, "actuator_capability_changed": False, "action_ranges_changed": False,
        "kernel_changed": False, "reward_changed": True, "same_mdp_claimed": False,
        "reset_prefix_class_acceptance_changed": prefix_scope is not None,
        "checkpoint_prefix_scope": prefix_scope,
        "reset_prefix_class_acceptance_semantics": "exact_declared_history_half_quarter_classes_same_deterministic_forward",
        "prefix_policy_kernel_changed": False, "prefix_samples_have_optimizer_credit": False,
        "observation_layout_changed": False,
        "observation_semantics_changed": ["existing_global_physical_potential_feature_FL_capture_soft_progress"],
        "observation_contract": {"source_policy_contract": canonical, "target_policy_contract": dict(canonical),
            "observation_layout": ROLE_OBSERVATION_LAYOUT, "observation_dimension": 372,
            "action_dimension": 12, "num_envs": 1, "parameter_mapping": "identity_all_parameters_and_buffers"},
        "normalizers": "preserve_verified_identity_RSL_state",
        "optimizer": "preserve_complete_verified_Adam_state_and_effective_learning_rate",
        "critic": "preserve_then_refit_on_fresh_reward_data", "training_rng": "preserve_verified_training_rng",
        "lifetime_counters_preserved": True, "old_rollout_inherited": False, "migration_added_updates": 0,
        "policy_evaluation_modes": ["deterministic_conditional_mean", "training_style_conditional_gaussian"],
        "trajectory_equivalence_claimed": False}
    return {"schema": SCHEMA, "reason": reason.strip(), "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": file_sha(checkpoint),
        "source_manifest_sha256": file_sha(checkpoint.with_name(checkpoint.stem + "_manifest.json")),
        "source_contract_sha256": digest(old), "target_contract_sha256": digest(new),
        "source_git_commit": old["source_git_commit"], "target_git_commit": new["source_git_commit"],
        "source_runtime_content_sha256": old["runtime_content_sha256"],
        "target_runtime_content_sha256": new["runtime_content_sha256"], "allowed_changed_files": delta,
        "changed_file_hashes": {p: {"before": old["files"].get(p), "after": new["files"][p]} for p in delta},
        "geometric_factor": None, "observation_dimension": 372, "action_dimension": 12,
        "preserve_actor_critic_optimizer_normalizer_rng_and_budget": True,
        "discard_old_rollout_storage": True, "physics_resume": "fresh_legal_P01_reset",
        "fl_capture_quality_same372_factor": factor}


def _request_history_runtime_binding(spec, execution, observation, target_policy):
    """Bind the pure entry predicate to this exact observable control graph."""
    from .semantic_policy_distribution import REQUEST_HISTORY_CAPS, REQUEST_HISTORY_SCALES
    phases = [f"P{i:02}" for i in range(1, 14)]
    if (set(spec.get("stages", {})) != set(phases)
            or any(spec["stages"][p].get("next_phase") !=
                   (phases[i+1] if i < 12 else "SUCCESS") for i, p in enumerate(phases))):
        raise ValueError("request-history kernel requires the unchanged forward adjacent P01-P13 graph")
    expected_caps = {p: list(REQUEST_HISTORY_CAPS[i]) for i, p in enumerate(phases)}
    if (execution.get("physics_hz") != 120. or execution.get("decision_hz") != 15.
            or execution.get("residual", {}).get("phase_caps_full12") != expected_caps
            or target_policy.get("phase_caps_full12") != list(expected_caps.values())):
        raise ValueError("request-history kernel clock or exact per-phase caps differ")
    groups, offset = {}, 0
    for group in observation.get("feature_groups", []):
        name = group.get("name")
        if name in groups:
            raise ValueError("request-history observation contains duplicate feature groups")
        groups[name] = (offset, group.get("size"), group.get("scale"))
        offset += group.get("size", 0)
    expected = {"stage_one_hot": (0, 13, 1.), "task_times": (18, 3, 200.),
        "completed_stages": (158, 13, 1.), "previous_raw_full12": (195, 12, 1.),
        "previous_residual_full12": (207, 12, list(REQUEST_HISTORY_SCALES))}
    if offset != 372 or any(groups.get(k) != v for k, v in expected.items()):
        raise ValueError("request-history schema indices or decoded REQUEST scales differ")
    return {"phase_graph": {p: spec["stages"][p]["next_phase"] for p in phases},
        "physics_hz": 120., "decision_hz": 15., "feature_bindings": {k: list(v) for k, v in expected.items()},
        "phase_caps_full12": expected_caps,
        "request_scope": "previous_filtered_REQUEST_not_post_mapper_effective_or_actual_motion"}


def _request_history_code_scope(before, after, *, functions=(), classes=(), constants=(), methods=(),
                                prefix_provenance_unpack=False):
    """Exact named AST boundary; the CLI loop permits only one provenance unpack."""
    import ast
    def protected(text, target):
        tree = ast.parse(text)
        seen, unpack_count = set(), 0
        retained = []
        for node in tree.body:
            name = getattr(node, "name", None)
            if ((isinstance(node, ast.FunctionDef) and name in functions)
                    or (isinstance(node, ast.ClassDef) and name in classes)):
                seen.add(name)
                continue
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in constants:
                seen.add(node.targets[0].id)
                continue
            if isinstance(node, ast.ClassDef):
                for method in list(node.body):
                    key = f"{name}.{getattr(method, 'name', '')}"
                    if isinstance(method, ast.FunctionDef) and key in methods:
                        node.body.remove(method)
                        seen.add(key)
            retained.append(node)
        tree.body = retained
        if prefix_provenance_unpack:
            expected = ast.dump(ast.parse("_request_history_prefix_provenance(args, contract, previous)", mode="eval").body, include_attributes=False)
            for call in ast.walk(tree):
                if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                        and call.func.id == "build_frozen_checkpoint_prefix_policy" and len(call.args) == 2
                        and isinstance(call.args[1], ast.Dict)):
                    continue
                record = call.args[1]
                for i in range(len(record.keys)-1, -1, -1):
                    if record.keys[i] is None and ast.dump(record.values[i], include_attributes=False) == expected:
                        del record.keys[i]; del record.values[i]
                        unpack_count += 1
            if unpack_count != (1 if target else 0):
                raise ValueError("request-history CLI must add exactly one verified prefix provenance unpack")
        return ast.dump(tree, include_attributes=False), seen
    old_ast, old_seen = protected(before, False)
    new_ast, new_seen = protected(after, True)
    if old_ast != new_ast or not old_seen <= new_seen:
        raise ValueError("request-history changed code outside its named review scope")
    def physical_calls(text):
        calls = []
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Call):
                name = getattr(node.func, "attr", getattr(node.func, "id", ""))
                if name.startswith(("set_", "write_joint", "write_root", "apply_force")) or name in {
                        "reset", "step", "step_physics", "simulate", "render", "write_data_to_sim", "apply_action"}:
                    calls.append(ast.dump(node, include_attributes=False))
        return sorted(calls)
    if physical_calls(before) != physical_calls(after):
        raise ValueError("request-history cannot alter physical setter/reset/step calls")
    return {"protected_ast_sha256": digest(old_ast), "protected_ast_identical": True,
        "reviewed_regions": sorted(new_seen), "prefix_provenance_unpack_only": prefix_provenance_unpack}


def _build_request_history_kernel_plan(checkpoint, metadata, old, new, *,
        allowed_changed_files, reason, review, project_root):
    """One same372 mean-kernel factor; no reward, temperature or physical change."""
    import copy
    import yaml
    from .semantic_policy_distribution import (CONFIG_NAMES, HISTORY_QUARTER_TEMPERED_POLICY,
        HISTORY_REQUEST_CAP_TRANSITION_POLICY, REQUEST_HISTORY_SEMANTICS, policy_contract)
    from .semantic_transfer_roles import ROLE_OBSERVATION_LAYOUT
    from .semantic_return_profile import runner_return_profile
    from .semantic_training import semantic_runner_config
    if (not isinstance(review, Mapping) or set(review) != {"reason", "reviewed_code_sha256"}
            or not isinstance(review["reason"], str) or not review["reason"].strip()
            or not isinstance(reason, str) or not reason.strip()):
        raise ValueError("request-history requires an explicit exact-code review")
    if (metadata.get("semantic_version") != "v3" or source_num_envs(metadata) != 1
            or any(c.get("semantic_version") != "v3" or c.get("experiment_id") != "fl_capture_quality_v1"
                   for c in (old, new))
            or old["source_git_commit"] == new["source_git_commit"]):
        raise ValueError("request-history requires a versioned FL-quality same-experiment v3 N1 boundary")
    variable = {"files", "runtime_content_sha256", "source_git_commit"}
    if {k:v for k,v in old.items() if k not in variable} != {k:v for k,v in new.items() if k not in variable}:
        raise ValueError("request-history cannot change configuration, physics, rates or budgets")
    delta = sorted(p for p in old["files"].keys() | new["files"].keys() if old["files"].get(p) != new["files"].get(p))
    if (set(old["files"]) != set(new["files"]) or set(delta) != REQUEST_HISTORY_KERNEL_FILES
            or sorted(allowed_changed_files) != delta or len(set(allowed_changed_files)) != len(allowed_changed_files)):
        raise ValueError("request-history requires exactly the reviewed six runtime files")
    hashes = {p: new["files"][p] for p in sorted(REQUEST_HISTORY_KERNEL_FILES)}
    if not isinstance(review["reviewed_code_sha256"], Mapping) or dict(review["reviewed_code_sha256"]) != hashes:
        raise ValueError("request-history review does not bind every exact target runtime byte hash")
    source_policy = policy_contract(HISTORY_QUARTER_TEMPERED_POLICY, observation_layout=ROLE_OBSERVATION_LAYOUT)
    target_policy = policy_contract(HISTORY_REQUEST_CAP_TRANSITION_POLICY, observation_layout=ROLE_OBSERVATION_LAYOUT)
    if metadata.get("policy_contract") != source_policy or metadata.get("optimizer_learning_rate") != 1e-5:
        raise ValueError("request-history requires the unchanged quarter HISTORY372 and verified effective Adam LR1e-5")
    records, configs = {}, {}
    for name in sorted(CONFIG_NAMES):
        path = f"configs/ppo_fl_capture_quality_v1/{name}"
        for c in (old, new):
            if set(c.get("selected_configuration", {})) != CONFIG_NAMES or c["selected_configuration"][name] != {"path": path, "sha256": c["files"].get(path)}:
                raise ValueError("request-history requires exact unchanged six-configuration bindings")
        before = _version_bytes(project_root, old, path, prefer_worktree=True)
        after = _version_bytes(project_root, new, path, prefer_worktree=True)
        if before != after:
            raise ValueError("request-history cannot change any of the six configuration files")
        records[name] = {"path": path, "source_sha256": old["files"][path], "target_sha256": new["files"][path], "bytes_identical": True}
        configs[name] = yaml.safe_load(after)
    binding = _request_history_runtime_binding(configs["stage_task_spec.yaml"], configs["execution_profile.yaml"], configs["observation_schema.json"], target_policy)
    scopes = {}
    allowed = {
        "semantic_history_actor": {"functions": ("cap_transition_request_history",), "classes": ("SemanticCapTransitionQuarterHistoryMLPModel",)},
        "semantic_policy_distribution": {"functions": ("policy_contract", "configure_policy_distribution", "supported_heteroscedastic_contract_version", "policy_version_from_metadata"),
            "constants": ("HISTORY_REQUEST_CAP_TRANSITION_POLICY", "HISTORY_REQUEST_CAP_TRANSITION_ACTOR_CLASS", "REQUEST_HISTORY_CAPS", "REQUEST_HISTORY_SCALES", "REQUEST_HISTORY_SEMANTICS")},
        "semantic_training": {"functions": ("semantic_runner_config", "load_semantic_checkpoint", "_validated_request_history_kernel_factor", "audited_history_policy_request", "audited_ppo_update")},
        "semantic_cli": {"functions": ("_preflight_checkpoint", "_request_history_prefix_provenance"), "prefix_provenance_unpack": True},
        "semantic_checkpoint_prefix_policy": {"functions": ("_source_record",), "methods": ("FrozenCheckpointPrefixPolicy.__init__",)},
        "semantic_migration": {"functions": ("_request_history_runtime_binding", "_request_history_code_scope", "_build_request_history_kernel_plan", "build_migration_plan", "validate_migration_plan"),
            "constants": ("REQUEST_HISTORY_KERNEL_SCHEMA", "REQUEST_HISTORY_KERNEL_FILES")},
    }
    for name, options in allowed.items():
        path = f"src/wlr50_clean/ppo/{name}.py"
        before = _version_text(project_root, old, path, prefer_worktree=True)
        after = _version_text(project_root, new, path, prefer_worktree=True)
        scopes[path] = _request_history_code_scope(before, after, **options)
    horizon = runner_return_profile(metadata["runner_config"], semantic_version="v3")["version"]
    options = dict(seed=metadata["seed"], device=metadata["runner_config"]["device"], semantic_version="v3", return_profile=horizon, observation_layout=ROLE_OBSERVATION_LAYOUT)
    source_config = semantic_runner_config(policy_version=source_policy["version"], **options)
    target_config = semantic_runner_config(policy_version=target_policy["version"], **options)
    source_rest, target_rest = copy.deepcopy(source_config), copy.deepcopy(target_config)
    source_rest["actor"].pop("class_name"); target_rest["actor"].pop("class_name")
    if metadata["runner_config"] != source_config or source_rest != target_rest:
        raise ValueError("request-history may change only the actor class selector, not PPO/normalization/temperature")
    observation = {"source_policy_contract": source_policy, "target_policy_contract": target_policy,
        "observation_layout": ROLE_OBSERVATION_LAYOUT, "observation_dimension": 372,
        "action_dimension": 12, "num_envs": 1, "parameter_mapping": "identity_all_parameters_and_buffers"}
    factor = {"schema": REQUEST_HISTORY_KERNEL_SCHEMA, "review_reason": review["reason"].strip(),
        "reviewed_code_sha256": hashes, "configuration_bindings": records, "code_scope": scopes,
        "source_policy_version": source_policy["version"], "target_policy_version": target_policy["version"],
        "source_policy_contract": source_policy, "target_policy_contract": target_policy,
        "source_runner_config": source_config, "target_runner_config": target_config,
        "history_center_semantics": REQUEST_HISTORY_SEMANTICS, "runtime_binding": binding,
        "kernel_changed": True, "physical_mdp_changed": False, "nominal_control_changed": False,
        "reward_changed": False, "task_acceptance_changed": False, "action_ranges_changed": False,
        "observation_semantics_changed": [], "observation_contract": observation,
        "deterministic_same_weights_same_observation": "changed_only_at_observed_cap_increase_entry_gate; not_behavior_equivalent",
        "sigma_same_weights_same_observation": "unchanged_learned_sigma_times_fixed_0.25",
        "stochastic_likelihood": "same_new_conditional_mean_and_sigma_for_sample_old_params_logprob_entropy_KL_update",
        "optimizer": "preserve_complete_verified_Adam_state_and_effective_learning_rate",
        "source_effective_learning_rate": 1e-5, "target_effective_learning_rate": 1e-5,
        "normalizers": "preserve_verified_identity_RSL_state", "training_rng": "preserve_verified_training_rng",
        "counter_origin": {k: metadata[k] for k in ("global_policy_decisions", "ppo_updates", "optimizer_steps")},
        "old_rollout_inherited": False, "migration_added_updates": 0, "migration_added_policy_decisions": 0,
        "prefix_effective_kernel": "source_checkpoint_weights_plus_explicit_target_kernel_and_plan_provenance",
        "prefix_samples_have_optimizer_credit": False, "trajectory_equivalence_claimed": False,
        "scope_is_reviewer_assertion_not_semantic_equivalence_proof": True}
    return {"schema": SCHEMA, "reason": reason.strip(), "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": file_sha(checkpoint), "source_manifest_sha256": file_sha(checkpoint.with_name(checkpoint.stem+"_manifest.json")),
        "source_contract_sha256": digest(old), "target_contract_sha256": digest(new),
        "source_git_commit": old["source_git_commit"], "target_git_commit": new["source_git_commit"],
        "source_runtime_content_sha256": old["runtime_content_sha256"], "target_runtime_content_sha256": new["runtime_content_sha256"],
        "allowed_changed_files": delta, "changed_file_hashes": {p: {"before": old["files"][p], "after": new["files"][p]} for p in delta},
        "geometric_factor": None, "observation_dimension": 372, "action_dimension": 12,
        "preserve_actor_critic_optimizer_normalizer_rng_and_budget": True,
        "discard_old_rollout_storage": True, "physics_resume": "fresh_legal_P01_reset", "request_history_kernel_factor": factor}


def _build_physical_innovation_sigma_plan(checkpoint, metadata, old, new, *,
        allowed_changed_files, reason, review, project_root):
    """Same REQUEST mean and full action support; only one phase sigma scale."""
    import copy
    import yaml
    from .semantic_policy_distribution import (CONFIG_NAMES, HISTORY_REQUEST_CAP_TRANSITION_POLICY,
        FR_KNEE_PHYSICAL_INNOVATION_POLICY, FR_KNEE_PHYSICAL_SIGMA_SEMANTICS, policy_contract)
    from .semantic_transfer_roles import ROLE_OBSERVATION_LAYOUT
    from .semantic_return_profile import runner_return_profile
    from .semantic_training import semantic_runner_config
    if (not isinstance(review, Mapping) or set(review) != {"reason", "reviewed_code_sha256"}
            or not isinstance(review["reason"], str) or not review["reason"].strip()
            or not isinstance(reason, str) or not reason.strip()):
        raise ValueError("physical-innovation sigma requires an explicit exact-code review")
    if (metadata.get("semantic_version") != "v3" or source_num_envs(metadata) != 1
            or any(c.get("semantic_version") != "v3" or c.get("experiment_id") != "fl_capture_quality_v1"
                   for c in (old, new)) or old["source_git_commit"] == new["source_git_commit"]):
        raise ValueError("physical-innovation sigma requires a versioned FL-quality v3 N1 boundary")
    variable = {"files", "runtime_content_sha256", "source_git_commit"}
    if {k:v for k,v in old.items() if k not in variable} != {k:v for k,v in new.items() if k not in variable}:
        raise ValueError("physical-innovation sigma cannot change configuration, physics, rates or budgets")
    delta = sorted(p for p in old["files"].keys() | new["files"].keys() if old["files"].get(p) != new["files"].get(p))
    if (set(old["files"]) != set(new["files"]) or set(delta) != PHYSICAL_INNOVATION_SIGMA_FILES
            or sorted(allowed_changed_files) != delta or len(set(allowed_changed_files)) != len(allowed_changed_files)):
        raise ValueError("physical-innovation sigma requires exactly the reviewed six runtime files")
    hashes = {p: new["files"][p] for p in sorted(PHYSICAL_INNOVATION_SIGMA_FILES)}
    if not isinstance(review["reviewed_code_sha256"], Mapping) or dict(review["reviewed_code_sha256"]) != hashes:
        raise ValueError("physical-innovation review must bind exact target code hashes")
    source_policy = policy_contract(HISTORY_REQUEST_CAP_TRANSITION_POLICY, observation_layout=ROLE_OBSERVATION_LAYOUT)
    target_policy = policy_contract(FR_KNEE_PHYSICAL_INNOVATION_POLICY, observation_layout=ROLE_OBSERVATION_LAYOUT)
    learning_rate = metadata.get("optimizer_learning_rate")
    if (metadata.get("policy_contract") != source_policy or isinstance(learning_rate, bool)
            or not isinstance(learning_rate, (int, float)) or not math.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError("physical-innovation requires exact REQUEST372 source and finite positive saved effective Adam LR")
    records, configs = {}, {}
    for name in sorted(CONFIG_NAMES):
        path = f"configs/ppo_fl_capture_quality_v1/{name}"
        for c in (old, new):
            if set(c.get("selected_configuration", {})) != CONFIG_NAMES or c["selected_configuration"][name] != {"path": path, "sha256": c["files"].get(path)}:
                raise ValueError("physical-innovation requires exact six configuration bindings")
        before = _version_bytes(project_root, old, path, prefer_worktree=True)
        after = _version_bytes(project_root, new, path, prefer_worktree=True)
        if before != after:
            raise ValueError("physical-innovation cannot change any of the six configuration bytes")
        records[name] = {"path": path, "source_sha256": old["files"][path], "target_sha256": new["files"][path], "bytes_identical": True}
        configs[name] = yaml.safe_load(after)
    binding = _request_history_runtime_binding(configs["stage_task_spec.yaml"], configs["execution_profile.yaml"], configs["observation_schema.json"], target_policy)
    allowed = {
        "semantic_history_actor": {"functions": ("physical_innovation_effective_log_std",), "classes": ("SemanticFRKneePhysicalInnovationHistoryMLPModel",)},
        "semantic_policy_distribution": {"functions": ("policy_contract", "configure_policy_distribution", "supported_heteroscedastic_contract_version", "policy_version_from_metadata"),
            "constants": ("FR_KNEE_PHYSICAL_INNOVATION_POLICY", "FR_KNEE_PHYSICAL_INNOVATION_ACTOR_CLASS", "FR_KNEE_PHYSICAL_SIGMA_SEMANTICS")},
        "semantic_training": {"functions": ("semantic_runner_config", "load_semantic_checkpoint", "_validated_physical_innovation_sigma_factor", "audited_history_policy_request", "audited_ppo_update")},
        "semantic_cli": {"functions": ("_preflight_checkpoint", "_request_history_prefix_provenance")},
        "semantic_checkpoint_prefix_policy": {"functions": ("_source_record",), "methods": ("FrozenCheckpointPrefixPolicy.__init__",)},
        "semantic_migration": {"functions": ("_build_physical_innovation_sigma_plan", "build_migration_plan", "validate_migration_plan"),
            "constants": ("PHYSICAL_INNOVATION_SIGMA_SCHEMA", "PHYSICAL_INNOVATION_SIGMA_FILES")},
    }
    scopes = {}
    for name, options in allowed.items():
        path = f"src/wlr50_clean/ppo/{name}.py"
        scopes[path] = _request_history_code_scope(
            _version_text(project_root, old, path, prefer_worktree=True),
            _version_text(project_root, new, path, prefer_worktree=True), **options)
    horizon = runner_return_profile(metadata["runner_config"], semantic_version="v3")["version"]
    options = dict(seed=metadata["seed"], device=metadata["runner_config"]["device"], semantic_version="v3", return_profile=horizon, observation_layout=ROLE_OBSERVATION_LAYOUT)
    source_config = semantic_runner_config(policy_version=source_policy["version"], **options)
    target_config = semantic_runner_config(policy_version=target_policy["version"], **options)
    source_rest, target_rest = copy.deepcopy(source_config), copy.deepcopy(target_config)
    source_rest["actor"].pop("class_name"); target_rest["actor"].pop("class_name")
    if metadata["runner_config"] != source_config or source_rest != target_rest:
        raise ValueError("physical-innovation may change only actor selector, not PPO/normalization/temperature")
    observation = {"source_policy_contract": source_policy, "target_policy_contract": target_policy,
        "observation_layout": ROLE_OBSERVATION_LAYOUT, "observation_dimension": 372,
        "action_dimension": 12, "num_envs": 1, "parameter_mapping": "identity_all_parameters_and_buffers"}
    factor = {"schema": PHYSICAL_INNOVATION_SIGMA_SCHEMA, "review_reason": review["reason"].strip(),
        "reviewed_code_sha256": hashes, "configuration_bindings": records, "code_scope": scopes,
        "source_policy_version": source_policy["version"], "target_policy_version": target_policy["version"],
        "source_policy_contract": source_policy, "target_policy_contract": target_policy,
        "source_runner_config": source_config, "target_runner_config": target_config,
        "sigma_scaling_semantics": FR_KNEE_PHYSICAL_SIGMA_SEMANTICS, "runtime_binding": binding,
        "kernel_changed": True, "physical_mdp_changed": False, "nominal_control_changed": False,
        "reward_changed": False, "task_acceptance_changed": False, "action_ranges_changed": False,
        "observation_semantics_changed": [], "observation_contract": observation,
        "deterministic_same_weights_same_observation": "unchanged_parent_REQUEST_history_mean_bitwise",
        "sigma_same_weights_same_observation": "P01_P05_all12_unchanged;P06_P13_only_FR_knee_index3_times_24_over_112",
        "stochastic_likelihood": "same_scaled_official_Gaussian_cache_for_sample_logprob_entropy_KL_update",
        "optimizer": "preserve_complete_verified_Adam_state_and_effective_learning_rate",
        "source_effective_learning_rate": learning_rate, "target_effective_learning_rate": learning_rate,
        "normalizers": "preserve_verified_identity_RSL_state", "training_rng": "preserve_verified_training_rng",
        "counter_origin": {k: metadata[k] for k in ("global_policy_decisions", "ppo_updates", "optimizer_steps")},
        "old_rollout_inherited": False, "migration_added_updates": 0, "migration_added_policy_decisions": 0,
        "prefix_effective_kernel": "source_checkpoint_weights_plus_explicit_target_kernel_and_plan_provenance",
        "prefix_samples_have_optimizer_credit": False, "physical_trajectory_equivalence_claimed": False,
        "scope_is_reviewer_assertion_not_semantic_equivalence_proof": True}
    return {"schema": SCHEMA, "reason": reason.strip(), "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": file_sha(checkpoint), "source_manifest_sha256": file_sha(checkpoint.with_name(checkpoint.stem+"_manifest.json")),
        "source_contract_sha256": digest(old), "target_contract_sha256": digest(new),
        "source_git_commit": old["source_git_commit"], "target_git_commit": new["source_git_commit"],
        "source_runtime_content_sha256": old["runtime_content_sha256"], "target_runtime_content_sha256": new["runtime_content_sha256"],
        "allowed_changed_files": delta, "changed_file_hashes": {p: {"before": old["files"][p], "after": new["files"][p]} for p in delta},
        "geometric_factor": None, "observation_dimension": 372, "action_dimension": 12,
        "preserve_actor_critic_optimizer_normalizer_rng_and_budget": True,
        "discard_old_rollout_storage": True, "physics_resume": "fresh_legal_P01_reset", "physical_innovation_sigma_factor": factor}


def _task_conditioned_plan_envelope(checkpoint, old, new, delta, reason, factor_key, factor):
    return {"schema": SCHEMA, "reason": reason.strip(), "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": file_sha(checkpoint),
        "source_manifest_sha256": file_sha(checkpoint.with_name(checkpoint.stem+"_manifest.json")),
        "source_contract_sha256": digest(old), "target_contract_sha256": digest(new),
        "source_git_commit": old["source_git_commit"], "target_git_commit": new["source_git_commit"],
        "source_runtime_content_sha256": old["runtime_content_sha256"],
        "target_runtime_content_sha256": new["runtime_content_sha256"],
        "allowed_changed_files": delta,
        "changed_file_hashes": {p: {"before": old["files"].get(p), "after": new["files"][p]} for p in delta},
        "geometric_factor": None, "observation_dimension": 372, "action_dimension": 12,
        "preserve_actor_critic_optimizer_normalizer_rng_and_budget": True,
        "discard_old_rollout_storage": True, "physics_resume": "fresh_legal_P01_reset", factor_key: factor}


def _task_conditioned_source_binding(metadata, old, new, project_root, *, archive=False):
    from .semantic_policy_distribution import (FR_KNEE_PHYSICAL_INNOVATION_POLICY,
        TASK_CONDITIONED_HIP_WHEEL_POLICY, policy_contract)
    from .semantic_transfer_roles import ROLE_OBSERVATION_LAYOUT
    from .semantic_training import semantic_runner_config
    from .semantic_return_profile import runner_return_profile
    source_version = metadata.get("policy_contract", {}).get("version")
    versions = {FR_KNEE_PHYSICAL_INNOVATION_POLICY, TASK_CONDITIONED_HIP_WHEEL_POLICY} if archive else {FR_KNEE_PHYSICAL_INNOVATION_POLICY}
    if (metadata.get("semantic_version") != "v3" or source_num_envs(metadata) != 1
            or source_version not in versions or old.get("semantic_version") != "v3"
            or new.get("semantic_version") != "v3" or old["source_git_commit"] == new["source_git_commit"]):
        raise ValueError("task/archival migration requires a versioned exact supported v3 N1 source")
    actual_head = subprocess.run(["git", "-C", str(project_root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True).stdout.strip()
    if actual_head != new["source_git_commit"]:
        raise ValueError("task/archival migration target HEAD is not the actual project HEAD")
    target_version = source_version if archive else TASK_CONDITIONED_HIP_WHEEL_POLICY
    source = policy_contract(source_version, observation_layout=ROLE_OBSERVATION_LAYOUT)
    target = policy_contract(target_version, observation_layout=ROLE_OBSERVATION_LAYOUT)
    horizon = runner_return_profile(metadata["runner_config"], semantic_version="v3")["version"]
    options = dict(seed=metadata["seed"], device=metadata["runner_config"]["device"],
        semantic_version="v3", return_profile=horizon, observation_layout=ROLE_OBSERVATION_LAYOUT)
    source_config = semantic_runner_config(policy_version=source_version, **options)
    target_config = semantic_runner_config(policy_version=target_version, **options)
    import copy
    a, b = copy.deepcopy(source_config), copy.deepcopy(target_config)
    a["actor"].pop("class_name"); b["actor"].pop("class_name")
    lr = metadata.get("optimizer_learning_rate")
    if (metadata.get("policy_contract") != source or metadata["runner_config"] != source_config or a != b
            or type(lr) not in (int, float) or not math.isfinite(lr) or lr <= 0):
        raise ValueError("task migration must preserve canonical PPO/Identity/source effective Adam LR")
    origin = {key: metadata[key] for key in ("global_policy_decisions", "ppo_updates", "optimizer_steps")}
    if any(type(v) is not int or v < 0 for v in origin.values()):
        raise ValueError("task migration source counters must be nonnegative integers")
    return {"source_policy_version": source_version, "target_policy_version": target_version,
        "source_policy_contract": source, "target_policy_contract": target,
        "source_runner_config": source_config, "target_runner_config": target_config,
        "source_effective_learning_rate": lr, "target_effective_learning_rate": lr,
        "counter_origin": origin, "observation_contract": {
            "source_policy_contract": source, "target_policy_contract": target,
            "observation_layout": ROLE_OBSERVATION_LAYOUT, "observation_dimension": 372,
            "action_dimension": 12, "num_envs": 1, "parameter_mapping": "identity_all_parameters_and_buffers"},
        "normalizers": "preserve_verified_identity_RSL_state",
        "optimizer": "preserve_complete_verified_Adam_state_and_effective_learning_rate",
        "optimizer_lr_scope": "all_parameter_groups_and_algorithm_learning_rate_scalar",
        "training_rng": "preserve_verified_training_rng", "lifetime_counters_preserved": True,
        "old_rollout_inherited": False, "migration_added_updates": 0, "migration_added_policy_decisions": 0,
        "prefix_samples_have_optimizer_credit": False, "physical_trajectory_equivalence_claimed": False}


def _build_archive_only_exact_bytes_plan(checkpoint, metadata, old, new, *,
        allowed_changed_files, reason, review, project_root):
    """Explicit known-policy HEAD-only archive boundary, never a source exemption."""
    from .semantic_policy_distribution import CONFIG_NAMES
    if (not isinstance(review, Mapping) or set(review) != {"reason"}
            or not isinstance(review["reason"], str) or not review["reason"].strip()
            or not isinstance(reason, str) or not reason.strip() or list(allowed_changed_files)):
        raise ValueError("archive-only migration needs its explicit reason and no changed runtime file")
    factor = _task_conditioned_source_binding(metadata, old, new, project_root, archive=True)
    expected_experiment = ("fl_capture_quality_v1" if factor["source_policy_version"] ==
        "request_history_FR_knee_P06plus_physical_innovation_sigma_v1" else "task_conditioned_hip_wheel_v1")
    if (old.get("experiment_id") != expected_experiment or new.get("experiment_id") != expected_experiment
            or {k:v for k,v in old.items() if k != "source_git_commit"}
            != {k:v for k,v in new.items() if k != "source_git_commit"}
            or set(old.get("selected_configuration", {})) != CONFIG_NAMES):
        raise ValueError("archive-only migration permits only HEAD, not runtime/configuration/namespace changes")
    for relative, expected in new["files"].items():
        if file_sha(project_root / relative) != expected:
            raise ValueError("archive-only current runtime bytes differ from the source checkpoint")
        _version_bytes(project_root, old, relative, prefer_worktree=True)
    for name, row in old["selected_configuration"].items():
        path = f"configs/ppo_{expected_experiment}/{name}"
        if row != {"path": path, "sha256": old["files"].get(path)} or row["sha256"] is None:
            raise ValueError("archive-only migration requires exact six namespace/hash bindings")
    factor.update(schema=ARCHIVE_ONLY_EXACT_BYTES_SCHEMA, review_reason=review["reason"].strip(),
        kernel_changed=False, reward_changed=False, task_acceptance_changed=False,
        nominal_control_changed=False, action_execution_changed=False,
        configuration_bindings=old["selected_configuration"], runtime_bytes_identical=True,
        observation_semantics_changed=[], same_mdp_claimed=True)
    return _task_conditioned_plan_envelope(checkpoint, old, new, [], reason,
        "archive_only_exact_bytes_factor", factor)


def _build_task_conditioned_hip_wheel_plan(checkpoint, metadata, old, new, *,
        allowed_changed_files, reason, review, project_root):
    """Joint task-reward/potential and state-sigma boundary; not a sigma-only waiver."""
    import ast
    import copy
    import yaml
    from .semantic_policy_distribution import CONFIG_NAMES, TASK_CONDITIONED_HIP_WHEEL_SIGMA_SEMANTICS
    if (not isinstance(review, Mapping) or set(review) != {"reason", "reviewed_code_sha256"}
            or not isinstance(review["reason"], str) or not review["reason"].strip()
            or not isinstance(reason, str) or not reason.strip()):
        raise ValueError("task-conditioned migration needs explicit reason and exact reviewed code hashes")
    factor = _task_conditioned_source_binding(metadata, old, new, project_root)
    if old.get("experiment_id") != "fl_capture_quality_v1" or new.get("experiment_id") != "task_conditioned_hip_wheel_v1":
        raise ValueError("task-conditioned migration requires the isolated FL-quality source and task target")
    variable = {"files", "runtime_content_sha256", "source_git_commit", "selected_configuration", "experiment_id"}
    if {k:v for k,v in old.items() if k not in variable} != {k:v for k,v in new.items() if k not in variable}:
        raise ValueError("task-conditioned migration cannot alter physical/runtime metadata")
    delta = sorted(p for p in old["files"].keys() | new["files"].keys() if old["files"].get(p) != new["files"].get(p))
    if (sorted(allowed_changed_files) != delta or len(set(allowed_changed_files)) != len(allowed_changed_files)
            or old["files"].keys() - new["files"].keys()):
        raise ValueError("task-conditioned migration requires exact changed inventory without deletions")
    target_paths = {f"configs/ppo_task_conditioned_hip_wheel_v1/{n}" for n in CONFIG_NAMES}
    code_delta = set(delta) - target_paths
    required = REQUEST_HISTORY_KERNEL_FILES | {SUPERVISOR, BODY_REWARD_CODE}
    if not code_delta <= TASK_CONDITIONED_HIP_WHEEL_FILES or not required <= code_delta:
        raise ValueError("task-conditioned migration changed protected files or omitted its joint implementation")
    hashes = {p: new["files"][p] for p in sorted(code_delta)}
    if not isinstance(review["reviewed_code_sha256"], Mapping) or dict(review["reviewed_code_sha256"]) != hashes:
        raise ValueError("task-conditioned exact code review does not bind the entire changed implementation")
    records = {}
    for side, contract, namespace in (("source", old, "ppo_fl_capture_quality_v1"),
                                     ("target", new, "ppo_task_conditioned_hip_wheel_v1")):
        if set(contract.get("selected_configuration", {})) != CONFIG_NAMES:
            raise ValueError("task-conditioned migration requires exactly six selected configurations")
        for name in CONFIG_NAMES:
            path = f"configs/{namespace}/{name}"
            row = {"path": path, "sha256": contract["files"].get(path)}
            if row["sha256"] is None or contract["selected_configuration"][name] != row:
                raise ValueError("task-conditioned selected configuration is not bound to its namespace and bytes")
            records.setdefault(name, {})[side] = row
    configs = {}
    for name, row in records.items():
        before_raw = _version_bytes(project_root, old, row["source"]["path"], prefer_worktree=True)
        after_raw = _version_bytes(project_root, new, row["target"]["path"], prefer_worktree=True)
        before, after = yaml.safe_load(before_raw), yaml.safe_load(after_raw)
        expected = copy.deepcopy(before)
        if name == "stage_task_spec.yaml":
            if (before.get("capture_approach_semantics") != "post_cross_FL_multiscale_positive_gap_plus_real_contact_v1"
                    or before.get("p09_lift_semantics") != "functional_free_air_lift_v3"
                    or before["nominal"].get("final_stop_owner") != "source_home_after_physical_stop_v2"):
                raise ValueError("task-conditioned source lacks preserved FL/RR/home semantics")
            expected.update(revision="task_conditioned_hip_wheel_v1",
                rolling_capture_retention={"rear_preparation_near_m": -.22, "blend_distance_m": .05,
                    "contact_fraction": .5, "positive_gap_scale_m": .003})
        elif name == "reward_config.yaml":
            if before.get("objective_profile") != "fl_capture_front_body_quality_v1" or before.get("quality_epsilon") != .03:
                raise ValueError("task-conditioned source must preserve the genuinely nonzero front-quality objective")
            expected.update(revision="task_conditioned_hip_wheel_quality_v1",
                objective_profile="task_conditioned_hip_wheel_quality_v1", quality_epsilon=.06,
                task_space_quality={"phases": ["P01", "P02", "P05", "P06", "P07", "P08", "P09"],
                    "clearance_margin_m": .020, "geometry_fraction": .5, "front_fraction": .5,
                    "sample_audit": True})
            expected["family_weights"]["body_stability"] = .06
            expected["signal_ownership"]["body_stability"] = ["gravity_attitude", "euler_roll_pitch_derivative",
                "conservative_collider_obstacle_separation_deficit"]
        elif before_raw != after_raw:
            raise ValueError(f"task-conditioned migration requires byte-identical protected configuration: {name}")
        if after != expected:
            raise ValueError(f"task-conditioned configuration exceeds its explicitly reviewed soft fields: {name}")
        row.update(bytes_identical=before_raw == after_raw, semantics_identical=before == after)
        configs[name] = after
    runtime_binding = _request_history_runtime_binding(configs["stage_task_spec.yaml"],
        configs["execution_profile.yaml"], configs["observation_schema.json"], factor["target_policy_contract"])
    spec = configs["stage_task_spec.yaml"]
    geometry = spec.get("geometry", {})
    if (any(geometry.get(k) != v for k,v in {"airborne_clearance_above_top_m": .015,
            "top_gap_min_m": -.015, "top_gap_max_m": .025, "xy_measurement_tolerance_m": .005}.items())
            or spec.get("support", {}).get("force_noise_floor_n") != .2
            or spec.get("fl_capture_potential", {}).get("fine_gap_scale_m") != .003):
        raise ValueError("task sigma observable proxies differ from the fixed current gap/contact geometry")
    runtime_binding["exploration_proxy_not_acceptance"] = {
        "FR_approach_gap_m": .015, "FL_fine_gap_m": .003, "top_gap_m": [-.015,.025],
        "contact_force_floor_N": .2, "xy_tolerance_m": .005,
        "rear_exploration_blend_m": [-.27,-.22], "rear_retention_blend_m": [-.27,-.22]}
    # Named soft scopes only; permit the new pure-quality helper import, not other module changes.
    def without_quality_import_and_mode(text):
        tree = ast.parse(text)
        tree.body = [n for n in tree.body if not (
            isinstance(n, ast.ImportFrom) and n.level == 1 and n.module == "semantic_task_quality")
            and not (isinstance(n, ast.Assign) and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "TASK_CONDITIONED_QUALITY_OBJECTIVE")]
        return ast.unparse(tree)
    scopes = {}
    for relative, functions, methods in (
            (SUPERVISOR, ("load_task_spec",), ("TaskStageSupervisor._current_capture_retention",)),
            (BODY_REWARD_CODE, ("_validate_task_priority",), ("SemanticRewardCalculator.evaluate",)),
            ("src/wlr50_clean/ppo/semantic_observation.py", (), ("SemanticObservationBuilder.build",))):
        scopes[relative] = _height_source_scope(
            without_quality_import_and_mode(_version_text(project_root, old, relative, prefer_worktree=True)),
            without_quality_import_and_mode(_version_text(project_root, new, relative, prefer_worktree=True)),
            functions=functions, methods=methods)
    helper = "src/wlr50_clean/ppo/semantic_task_quality.py"
    if helper in code_delta:
        helper_tree = ast.parse(_version_text(project_root, new, helper, prefer_worktree=True))
        # This reviewed helper's local JSON result.update is not app.update.
        # Keep the existing veto untouched; only normalize that exact receiver.
        for node in ast.walk(helper_tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "result"
                    and node.func.attr == "update"):
                node.func = ast.Name(id="local_result_mapping_update", ctx=ast.Load())
        _height_no_physics_writes(ast.unparse(helper_tree))
    # Exact protected ASTs keep hard events and the entire nominal/dispatch classes unchanged.
    def protected_supervisor(text):
        tree = ast.parse(text)
        names = {"TaskEvaluator", "NominalMotionProvider", "SemanticControllerAdapter"}
        selected = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
        if {node.name for node in selected} != names:
            raise ValueError("task-conditioned supervisor omitted a protected event/control class")
        return ast.dump(ast.Module(body=selected, type_ignores=[]), include_attributes=False)
    before_ast = protected_supervisor(_version_text(project_root, old, SUPERVISOR, prefer_worktree=True))
    after_ast = protected_supervisor(_version_text(project_root, new, SUPERVISOR, prefer_worktree=True))
    if before_ast != after_ast:
        raise ValueError("task-conditioned migration cannot change hard events, nominal motion or controller dispatch")
    for relative in delta:
        if relative in old["files"]:
            _version_bytes(project_root, old, relative, prefer_worktree=True)
        if file_sha(project_root / relative) != new["files"][relative]:
            raise ValueError("task-conditioned target bytes differ from the reviewed runtime")
    factor.update(schema=TASK_CONDITIONED_HIP_WHEEL_SCHEMA, review_reason=review["reason"].strip(),
        branch_id="task_conditioned_hip_wheel_v1", reviewed_code_sha256=hashes,
        configuration_bindings=records, protected_supervisor_ast_sha256=digest(before_ast),
        quality_code_scope=scopes,
        protected_hard_evaluator_nominal_dispatch_ast_identical=True,
        sigma_scaling_semantics=TASK_CONDITIONED_HIP_WHEEL_SIGMA_SEMANTICS,
        runtime_binding=runtime_binding,
        kernel_changed=True, reward_changed=True, same_mdp_claimed=False, physical_mdp_changed=False,
        task_acceptance_changed=False, nominal_control_changed=False, action_execution_changed=False,
        physical_scene_changed=False, actuator_capability_changed=False, action_ranges_changed=False,
        observation_layout_changed=False,
        observation_semantics_changed=["existing_global_physical_potential_current_front_rolling_retention"],
        critic="preserve_then_refit_on_fresh_reward_data",
        stochastic_likelihood="same_task_scaled_official_Gaussian_cache_for_sample_logprob_entropy_KL_update",
        prefix_effective_kernel="source_weights_plus_explicit_target_kernel_and_task_config",
        scope_is_reviewer_assertion_not_semantic_equivalence_proof=True)
    return _task_conditioned_plan_envelope(checkpoint, old, new, delta, reason,
        "task_conditioned_hip_wheel_factor", factor)


def _training_quantity_budget_code_scope(before, after, *, relative):
    """Pin reviewed quantity plumbing, not arbitrary whole-function exemptions."""
    import ast

    def tree_hash(node):
        return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()

    # These are Python 3.11 AST pairs for the pinned official runtime's reviewed quantity
    # diff. A changed LR, entropy formula, loss, reset, control call or receipt
    # inside a listed function cannot pass by merely editing the review hashes.
    bindings = {
        "semantic_training.py": {
            "training_quantity_budgets": (None, "2ba090cfe6659c62acf38f1aa87c798f27f29f5cf9931e691a338c7f6f113855"),
            "load_semantic_checkpoint": ("ffd7f35d9e5a3e40ff484815aff99306e1801b7d4823e27699e10b3682a33a35", "3dba6ef72f595a737fafd1dd2f0ea9b90f0af1c63600c6e83c4eccb17967134e"),
            "train_semantic": ("c08e29bb05e65555af58a23c10f2bcee4613422f1e2a49f56b5e8268b7c823b1", "a6beba47341cdbf3f348a61912fda1867a53447161f0f9806be94f5647ff04dd"),
        },
        "semantic_cli.py": {
            "validate_request": ("0312487aa05813b5235174823d4d98833855ec3fa574f2edfd52d357d2673bb4", "8bb59f1ba95c3df64fe51398799a87dc9589325ae618179f4ed9a456a816972d"),
            "dispatch_vector": ("ab82830e4b91746e7bad7926c2341f25cbadd6a9ea0a7bcc6505fb6be984133b", "b77892b89f96cbe6736feaba24ce5ecb97ccc318817b59b90973c02a0f05487e"),
            "runtime_contract": ("bf3af97f8bfb7acb9f961cab2f9cf09c4540e1b3e31bd25b39e7382d4a77de89", "0f8d37f76cc53ab9939a63c0a7e2fe4353e647d39f94cd6a2526fd8c9e8b6336"),
            "_preflight_checkpoint": ("e242a01d0e79f877e93edc889afbb48cb1a50c20490db9c9744ca7a1225c1cec", "6409d586389423ac2be321b3d100089e726c87fd2d2714f079409655557cddbe"),
            "dispatch_live": ("d353a50b81c1801fcc2a95f65437e675265021db43661a748d26dd548733b3de", "3243056158bc6b8ffcb2fc45db5400ea934d2bdc137ba71d28e1dac181faa30a"),
        },
    }
    trees = [ast.parse(before), ast.parse(after)]
    filename = Path(relative).name
    if filename in bindings:
        expected = bindings[filename]
        for side, tree in enumerate(trees):
            nodes = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
            for name, pair in expected.items():
                actual = tree_hash(nodes[name]) if name in nodes else None
                if actual != pair[side]:
                    raise ValueError(f"quantity-only reviewed AST changed: {filename}:{name}")
            tree.body = [n for n in tree.body if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) or n.name not in expected]
        if filename == "semantic_cli.py":
            imports = [n for n in trees[1].body if isinstance(n, ast.ImportFrom) and n.module == "semantic_training"]
            if len(imports) != 1 or sum(x.name == "training_quantity_budgets" and x.asname is None for x in imports[0].names) != 1:
                raise ValueError("quantity-only CLI must import exactly its budget helper")
            imports[0].names = [n for n in imports[0].names if n.name != "training_quantity_budgets"]
        regions = {name: {"before": pair[0], "after": pair[1]} for name, pair in expected.items()}
    elif filename == "semantic_migration.py":
        added = {"_training_quantity_budget_code_scope", "_build_training_quantity_budget_plan"}
        present = [{n.name for n in t.body if isinstance(n, ast.FunctionDef)} & added for t in trees]
        if present != [set(), added]:
            raise ValueError("quantity-only migration adds only its explicit new factor helpers")
        trees[1].body = [n for n in trees[1].body if not isinstance(n, ast.FunctionDef) or n.name not in added]
        constants = [n for n in trees[1].body if isinstance(n, ast.Assign) and any(isinstance(x, ast.Name) and x.id == "TRAINING_QUANTITY_BUDGET_SCHEMA" for x in n.targets)]
        if len(constants) != 1 or ast.literal_eval(constants[0].value) != TRAINING_QUANTITY_BUDGET_SCHEMA:
            raise ValueError("quantity-only migration schema declaration changed")
        trees[1].body.remove(constants[0])
        build = next(n for n in trees[1].body if isinstance(n, ast.FunctionDef) and n.name == "build_migration_plan")
        indexes = [i for i, n in enumerate(build.args.kwonlyargs) if n.arg == "training_quantity_budget_review"]
        if len(indexes) != 1 or ast.dump(build.args.kw_defaults[indexes[0]]) != "Constant(value=None)":
            raise ValueError("quantity review must remain an optional explicit keyword")
        del build.args.kwonlyargs[indexes[0]]
        del build.args.kw_defaults[indexes[0]]
        route = ast.parse('''
if training_quantity_budget_review is not None:
    if any(v is not None for v in (prior_evidence, qualification_evidence, evaluator_review,
            execution_evidence, video_review, timing_review, body_reward_review, height_recovery_review,
            exploration_temperature_review, task_first_reward_review, execution_composition_review,
            final_stop_handoff_review, rr_physical_acceptance_review, fl_capture_quality_review,
            request_history_kernel_review, physical_innovation_sigma_review,
            task_conditioned_hip_wheel_review, archive_only_exact_bytes_review)):
        raise ValueError("quantity-only budget continuation cannot mix another migration factor")
    return _build_training_quantity_budget_plan(checkpoint, metadata, old, new,
        allowed_changed_files=allowed_changed_files, reason=reason,
        review=training_quantity_budget_review, project_root=Path(project_root))
''').body[0]
        routes = [n for n in build.body if ast.dump(n, include_attributes=False) == ast.dump(route, include_attributes=False)]
        if len(routes) != 1:
            raise ValueError("quantity-only migration route differs from the exclusive reviewed route")
        build.body.remove(routes[0])
        validate = next(n for n in trees[1].body if isinstance(n, ast.FunctionDef) and n.name == "validate_migration_plan")
        calls = [n for n in ast.walk(validate) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "build_migration_plan"]
        if len(calls) != 1:
            raise ValueError("quantity-only validator lost its canonical builder")
        added_keywords = [k for k in calls[0].keywords if k.arg == "training_quantity_budget_review"]
        expected_value = ast.parse('''None if "training_quantity_budget_factor" not in supplied else {
            "reason": supplied["training_quantity_budget_factor"]["review_reason"],
            "reviewed_code_sha256": supplied["training_quantity_budget_factor"]["reviewed_code_sha256"]}''', mode="eval").body
        if len(added_keywords) != 1 or ast.dump(added_keywords[0].value, include_attributes=False) != ast.dump(expected_value, include_attributes=False):
            raise ValueError("quantity-only validator must reconstruct the exact explicit review")
        calls[0].keywords.remove(added_keywords[0])
        regions = {"new_factor_helpers": sorted(added), "existing_routes_preserved": True}
    else:
        raise ValueError("quantity-only migration cannot modify this code file")
    if ast.dump(trees[0], include_attributes=False) != ast.dump(trees[1], include_attributes=False):
        raise ValueError(f"quantity-only migration changed protected AST: {relative}")
    return {"protected_ast_sha256": tree_hash(trees[0]), "exact_reviewed_regions": regions}


def _build_training_quantity_budget_plan(checkpoint, metadata, old, new, *,
        allowed_changed_files, reason, review, project_root):
    """Extend one lifetime quantity ceiling; preserve the MDP, kernel and Adam."""
    import copy
    import yaml
    from .semantic_policy_distribution import CONFIG_NAMES, TASK_CONDITIONED_HIP_WHEEL_POLICY
    if (not isinstance(review, Mapping) or set(review) != {"reason", "reviewed_code_sha256"}
            or not isinstance(review["reason"], str) or not review["reason"].strip()
            or not isinstance(reason, str) or not reason.strip()):
        raise ValueError("quantity-only migration requires a reason and exact reviewed code hashes")
    factor = _task_conditioned_source_binding(metadata, old, new, project_root, archive=True)
    experiment = "task_conditioned_hip_wheel_v1"
    source_budgets = {"smoke": 10000, "phase_suffix": 100000, "full_episode": 100000}
    target_budgets = {**source_budgets, "full_episode": 131072}
    if (factor["source_policy_version"] != TASK_CONDITIONED_HIP_WHEEL_POLICY
            or old.get("experiment_id") != experiment or new.get("experiment_id") != experiment
            or old["source_git_commit"] != "ee5a9651591d20bea48be8ceba075fa66594cb36"
            or metadata.get("training_quantity_budget_extension") is not None
            or old.get("training_budgets") != source_budgets or new.get("training_budgets") != target_budgets
            or any(type(v) is not int for c in (old, new) for v in c.get("training_budgets", {}).values())):
        raise ValueError("quantity-only migration requires the first exact task-policy 100000-to-131072 boundary")
    variable = {"files", "runtime_content_sha256", "source_git_commit", "selected_configuration", "training_budgets"}
    if {k: v for k, v in old.items() if k not in variable} != {k: v for k, v in new.items() if k not in variable}:
        raise ValueError("quantity-only migration cannot change runtime/MDP/control metadata")
    profile = f"configs/ppo_{experiment}/execution_profile.yaml"
    code = {f"src/wlr50_clean/ppo/{name}.py" for name in ("semantic_cli", "semantic_training", "semantic_migration")}
    delta = sorted(p for p in old["files"].keys() | new["files"].keys() if old["files"].get(p) != new["files"].get(p))
    if (set(old["files"]) != set(new["files"]) or set(delta) != code | {profile}
            or sorted(allowed_changed_files) != delta or len(set(allowed_changed_files)) != len(allowed_changed_files)):
        raise ValueError("quantity-only migration requires exactly three code files and one execution profile")
    hashes = {p: new["files"][p] for p in sorted(code)}
    if not isinstance(review["reviewed_code_sha256"], Mapping) or dict(review["reviewed_code_sha256"]) != hashes:
        raise ValueError("quantity-only review must bind every changed code byte")
    source_bytes = {}
    for path, expected in new["files"].items():
        if file_sha(project_root / path) != expected:
            raise ValueError(f"quantity-only target bytes are not the reviewed runtime: {path}")
        source_bytes[path] = _version_bytes(project_root, old, path, prefer_worktree=True)
    bindings = {}
    if any(set(c.get("selected_configuration", {})) != CONFIG_NAMES for c in (old, new)):
        raise ValueError("quantity-only migration requires all six selected configurations")
    for name in sorted(CONFIG_NAMES):
        path = f"configs/ppo_{experiment}/{name}"
        rows = []
        for contract in (old, new):
            row = contract["selected_configuration"][name]
            if row != {"path": path, "sha256": contract["files"].get(path)} or row["sha256"] is None:
                raise ValueError("quantity-only migration selected configuration/hash mismatch")
            rows.append(dict(row))
        before, after = source_bytes[path], (project_root / path).read_bytes()
        if path == profile:
            marker = b"  full_episode: 100000"
            if before.count(marker) != 1 or before.replace(marker, b"  full_episode: 131072") != after:
                raise ValueError("quantity-only profile permits only the single full_episode integer delta")
            a, b = yaml.safe_load(before), yaml.safe_load(after)
            expected = copy.deepcopy(a)
            expected["training_budgets"] = target_budgets
            if a.get("training_budgets") != source_budgets or b != expected:
                raise ValueError("quantity-only profile altered a non-budget field")
        elif before != after:
            raise ValueError("quantity-only migration must preserve five configuration files byte-for-byte")
        bindings[name] = {"source": rows[0], "target": rows[1], "bytes_identical": before == after}
    scopes = {path: _training_quantity_budget_code_scope(source_bytes[path].decode("utf-8"),
        (project_root / path).read_text(encoding="utf-8"), relative=path) for path in sorted(code)}
    spent = metadata.get("stage_requested_decisions")
    branch = metadata.get("task_conditioned_hip_wheel_branch", {})
    origin = branch.get("counter_origin", {})
    counters = factor["counter_origin"]
    counts = metadata.get("task_conditioned_hip_wheel_branch_counts")
    budget_origin = metadata.get("new_mdp_origin_global_policy_decisions")
    if (not isinstance(spent, Mapping) or set(spent) != set(source_budgets)
            or any(type(v) is not int or not 0 <= v <= source_budgets[k] for k, v in spent.items())
            or branch.get("schema") != "wlr50_clean.task_conditioned_hip_wheel_branch.v1"
            or branch.get("branch_id") != experiment or set(origin) != set(counters)
            or any(type(v) is not int or not 0 <= v <= counters[k] for k, v in origin.items())
            or not isinstance(counts, Mapping) or any(type(v) is not int for v in counts.values())
            or counts != {k: counters[k] - origin[k] for k in counters}
            or type(budget_origin) is not int or not 0 <= budget_origin <= counters["global_policy_decisions"]
            or sum(spent.values()) > counters["global_policy_decisions"] - budget_origin):
        raise ValueError("quantity-only continuation requires exact lifetime budgets and original task branch accounting")
    preserved = {k: copy.deepcopy(v) for k, v in metadata.items() if k.endswith("_branch")
        or k.endswith("_branch_counts") or k in ("source_stage_requested_decisions", "new_mdp_origin_global_policy_decisions")}
    factor.update(schema=TRAINING_QUANTITY_BUDGET_SCHEMA, review_reason=review["reason"].strip(),
        reviewed_code_sha256=hashes, code_scope=scopes, source_training_budgets=source_budgets,
        target_training_budgets=target_budgets, source_stage_requested_decisions=dict(spent),
        target_stage_requested_decisions=dict(spent), preserved_branch_metadata=preserved,
        configuration_bindings=bindings, budget_field="training_budgets.full_episode",
        source_remaining_full_episode=source_budgets["full_episode"] - spent["full_episode"],
        target_remaining_full_episode=target_budgets["full_episode"] - spent["full_episode"],
        entropy_schedule_denominator=210000, training_quantity_only=True,
        first_target_execution="train_v3_N1_full_episode_fresh_natural_P01_no_offset_no_prefix",
        kernel_changed=False, reward_changed=False, task_acceptance_changed=False,
        nominal_control_changed=False, action_execution_changed=False,
        observation_semantics_changed=[], same_mdp_claimed=True, new_mdp=False,
        lifetime_stage_budget_counters_reset=False, original_branch_origin_preserved=True)
    return _task_conditioned_plan_envelope(checkpoint, old, new, delta, reason,
        "training_quantity_budget_factor", factor)


def build_migration_plan(checkpoint: Path, current_contract: Mapping[str, Any], *,
                         allowed_changed_files: Sequence[str], reason: str,
                         prior_evidence: Mapping[str, Any] | None = None,
                         qualification_evidence: Path | None = None,
                         evaluator_review: Mapping[str, Any] | None = None,
                         execution_evidence: Mapping[str, Any] | None = None,
                         video_review: Mapping[str, Any] | None = None,
                         timing_review: Mapping[str, Any] | None = None,
                         body_reward_review: Mapping[str, Any] | None = None,
                         height_recovery_review: Mapping[str, Any] | None = None,
                         exploration_temperature_review: Mapping[str, Any] | None = None,
                         task_first_reward_review: Mapping[str, Any] | None = None,
                         execution_composition_review: Mapping[str, Any] | None = None,
                         final_stop_handoff_review: Mapping[str, Any] | None = None,
                         rr_physical_acceptance_review: Mapping[str, Any] | None = None,
                         fl_capture_quality_review: Mapping[str, Any] | None = None,
                         request_history_kernel_review: Mapping[str, Any] | None = None,
                         physical_innovation_sigma_review: Mapping[str, Any] | None = None,
                         task_conditioned_hip_wheel_review: Mapping[str, Any] | None = None,
                         archive_only_exact_bytes_review: Mapping[str, Any] | None = None,
                         training_quantity_budget_review: Mapping[str, Any] | None = None,
                         project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Build a reviewed plan after committing the new runtime; does not write."""
    checkpoint = Path(checkpoint).resolve(strict=True)
    metadata = checkpoint_metadata(checkpoint)
    old, new = _contract(metadata["runtime_contract"]), _contract(current_contract)
    if training_quantity_budget_review is not None:
        if any(v is not None for v in (prior_evidence, qualification_evidence, evaluator_review,
                execution_evidence, video_review, timing_review, body_reward_review, height_recovery_review,
                exploration_temperature_review, task_first_reward_review, execution_composition_review,
                final_stop_handoff_review, rr_physical_acceptance_review, fl_capture_quality_review,
                request_history_kernel_review, physical_innovation_sigma_review,
                task_conditioned_hip_wheel_review, archive_only_exact_bytes_review)):
            raise ValueError("quantity-only budget continuation cannot mix another migration factor")
        return _build_training_quantity_budget_plan(checkpoint, metadata, old, new,
            allowed_changed_files=allowed_changed_files, reason=reason,
            review=training_quantity_budget_review, project_root=Path(project_root))
    if task_conditioned_hip_wheel_review is not None or archive_only_exact_bytes_review is not None:
        if (task_conditioned_hip_wheel_review is not None and archive_only_exact_bytes_review is not None
                or any(v is not None for v in (prior_evidence, qualification_evidence, evaluator_review,
                    execution_evidence, video_review, timing_review, body_reward_review, height_recovery_review,
                    exploration_temperature_review, task_first_reward_review, execution_composition_review,
                    final_stop_handoff_review, rr_physical_acceptance_review, fl_capture_quality_review,
                    request_history_kernel_review, physical_innovation_sigma_review))):
            raise ValueError("task-conditioned or archive-only boundary cannot mix another migration factor")
        builder = (_build_task_conditioned_hip_wheel_plan if task_conditioned_hip_wheel_review is not None
                   else _build_archive_only_exact_bytes_plan)
        return builder(checkpoint, metadata, old, new, allowed_changed_files=allowed_changed_files,
            reason=reason, review=(task_conditioned_hip_wheel_review if task_conditioned_hip_wheel_review is not None
                                  else archive_only_exact_bytes_review), project_root=Path(project_root))
    if physical_innovation_sigma_review is not None:
        if any(v is not None for v in (prior_evidence, qualification_evidence, evaluator_review,
                execution_evidence, video_review, timing_review, body_reward_review,
                height_recovery_review, exploration_temperature_review, task_first_reward_review,
                execution_composition_review, final_stop_handoff_review, rr_physical_acceptance_review,
                fl_capture_quality_review, request_history_kernel_review)):
            raise ValueError("physical-innovation sigma cannot mix any other migration factor")
        return _build_physical_innovation_sigma_plan(checkpoint, metadata, old, new,
            allowed_changed_files=allowed_changed_files, reason=reason,
            review=physical_innovation_sigma_review, project_root=Path(project_root))
    if request_history_kernel_review is not None:
        if any(v is not None for v in (prior_evidence, qualification_evidence, evaluator_review,
                execution_evidence, video_review, timing_review, body_reward_review,
                height_recovery_review, exploration_temperature_review, task_first_reward_review,
                execution_composition_review, final_stop_handoff_review, rr_physical_acceptance_review,
                fl_capture_quality_review)):
            raise ValueError("request-history kernel cannot mix any other migration factor")
        return _build_request_history_kernel_plan(checkpoint, metadata, old, new,
            allowed_changed_files=allowed_changed_files, reason=reason,
            review=request_history_kernel_review, project_root=Path(project_root))
    if fl_capture_quality_review is not None:
        if any(v is not None for v in (prior_evidence, qualification_evidence, evaluator_review,
                execution_evidence, video_review, timing_review, body_reward_review,
                height_recovery_review, exploration_temperature_review, task_first_reward_review,
                execution_composition_review, final_stop_handoff_review, rr_physical_acceptance_review)):
            raise ValueError("FL capture cannot mix any other migration factor")
        return _build_fl_capture_quality_plan(checkpoint, metadata, old, new,
            allowed_changed_files=allowed_changed_files, reason=reason,
            review=fl_capture_quality_review, project_root=Path(project_root))
    if rr_physical_acceptance_review is not None:
        if any(v is not None for v in (prior_evidence, qualification_evidence, evaluator_review,
                execution_evidence, video_review, timing_review, body_reward_review,
                height_recovery_review, exploration_temperature_review, task_first_reward_review,
                execution_composition_review, final_stop_handoff_review)):
            raise ValueError("RR acceptance cannot mix any other migration factor")
        return _build_rr_physical_acceptance_plan(checkpoint, metadata, old, new,
            allowed_changed_files=allowed_changed_files, reason=reason,
            review=rr_physical_acceptance_review, project_root=Path(project_root))
    if final_stop_handoff_review is not None:
        if any(v is not None for v in (prior_evidence, qualification_evidence, evaluator_review,
                execution_evidence, video_review, timing_review, body_reward_review,
                height_recovery_review, exploration_temperature_review, task_first_reward_review,
                execution_composition_review)):
            raise ValueError("final stop handoff cannot mix any other migration factor")
        return _build_final_stop_handoff_plan(checkpoint, metadata, old, new,
            allowed_changed_files=allowed_changed_files, reason=reason,
            review=final_stop_handoff_review, project_root=Path(project_root))
    if execution_composition_review is not None:
        if any(v is not None for v in (prior_evidence, qualification_evidence, evaluator_review,
                execution_evidence, video_review, timing_review, body_reward_review,
                height_recovery_review, exploration_temperature_review, task_first_reward_review)):
            raise ValueError("execution composition cannot mix any other migration factor")
        return _build_execution_composition_plan(checkpoint, metadata, old, new,
            allowed_changed_files=allowed_changed_files, reason=reason,
            review=execution_composition_review, project_root=Path(project_root))
    if task_first_reward_review is not None:
        if any(v is not None for v in (prior_evidence, qualification_evidence, evaluator_review,
                execution_evidence, video_review, timing_review, body_reward_review,
                height_recovery_review, exploration_temperature_review)):
            raise ValueError("task-first reward cannot mix any other migration factor")
        return _build_task_first_reward_plan(checkpoint, metadata, old, new,
            allowed_changed_files=allowed_changed_files, reason=reason,
            review=task_first_reward_review, project_root=Path(project_root))
    if exploration_temperature_review is not None:
        if any(v is not None for v in (prior_evidence, qualification_evidence, evaluator_review,
                execution_evidence, video_review, timing_review, body_reward_review, height_recovery_review)):
            raise ValueError("temperature continuation cannot mix any other migration factor")
        return _build_exploration_temperature_plan(checkpoint, metadata, old, new,
            allowed_changed_files=allowed_changed_files, reason=reason,
            review=exploration_temperature_review, project_root=Path(project_root))
    variable = {"files", "runtime_content_sha256", "source_git_commit"}
    if body_reward_review is not None and timing_review is not None:
        raise ValueError("body reward uses a verified timing ancestor, not a mixed top-level timing factor")
    if height_recovery_review is not None and any(v is not None for v in (
            timing_review, body_reward_review, prior_evidence, qualification_evidence,
            evaluator_review, execution_evidence, video_review)):
        raise ValueError("height recovery cannot mix other migration factors")
    if timing_review is not None or body_reward_review is not None or height_recovery_review is not None:
        if any(v is not None for v in (prior_evidence, qualification_evidence, evaluator_review,
                                       execution_evidence, video_review)):
            raise ValueError("timing-only migration cannot mix other reviewed factors")
        variable.add("selected_configuration")  # Bound field-by-field below, not exempted.
    old_fixed = {key: value for key, value in old.items() if key not in variable}
    new_fixed = {key: value for key, value in new.items() if key not in variable}
    if old_fixed != new_fixed:
        raise ValueError("migration cannot change frozen physics, runtime versions, rates or budgets")
    declared = list(allowed_changed_files)
    if not reason.strip() or len(set(declared)) != len(declared):
        raise ValueError("migration requires a reason and unique exact changed-file names")
    delta = sorted(path for path in set(old["files"]) | set(new["files"]) if old["files"].get(path) != new["files"].get(path))
    observation_contract = None
    timing = (_timing_only_factor(metadata, old, new, delta, timing_review, Path(project_root))
              if timing_review is not None else None)
    body_reward = (_body_reward_factor(checkpoint, metadata, old, new, delta, body_reward_review, Path(project_root))
                   if body_reward_review is not None else None)
    height = (_height_recovery_factor(metadata, old, new, delta, height_recovery_review, Path(project_root))
              if height_recovery_review is not None else None)
    if timing is None and body_reward is None and height is None and (video_review is None or not (set(delta) & VIDEO_FILES)):
        # Existing instrumentation-only permission is unchanged; video has
        # its own explicit review and receipt, never a widened allowlist.
        observation_contract = _instrumentation_observation_contract(
            metadata, old, new, delta, Path(project_root),
            mixed_factors=any(value is not None for value in (
                prior_evidence, qualification_evidence, evaluator_review, execution_evidence, video_review)))
    video = None
    if video_review is not None:
        if any(value is not None for value in (
                prior_evidence, qualification_evidence, evaluator_review, execution_evidence)):
            raise ValueError("video instrumentation and task/execution factors require separate reviewed boundaries")
        video = _video_instrumentation_factor(old, new, video_review, delta, Path(project_root))
        video_observation = _video_observation_contract(metadata, old, new, delta, Path(project_root))
        if video_observation is not None:
            video["observation_contract"] = video_observation
    execution = None
    additional = set(VIDEO_FILES) if video is not None else ({TIMING_ONLY_SPEC} if timing is not None else set())
    if body_reward is not None:
        additional = {TIMING_ONLY_SPEC, BODY_REWARD_CODE, BODY_REWARD_CONFIG}
    if height is not None:
        additional = set(HEIGHT_FILES)
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
    prior_transition = SUPERVISOR in delta and evaluator is None and timing is None and body_reward is None and height is None
    if prior_transition != (prior_evidence is not None) or (prior_transition and STAGE_SPEC not in delta):
        raise ValueError("nominal source change and explicit prior evidence/config change must occur together")
    if qualification_evidence is not None and not prior_transition:
        raise ValueError("qualification permission requires the explicitly declared supervisor source change")
    geometric = _geometric_factor(Path(project_root), old, new, prior_transition=prior_transition) if STAGE_SPEC in delta else None
    prior = _prior_factor(Path(project_root), old, new, prior_evidence, checkpoint,
                           qualification_transition=qualification_evidence is not None) if prior_transition else None
    qualification = _qualification_factor(Path(project_root), old, new, qualification_evidence) if qualification_evidence is not None else None
    layout_contract = observation_contract or (video or timing or body_reward or height or {}).get("observation_contract")
    sidecar = checkpoint.with_name(checkpoint.stem + "_manifest.json")
    result = {"schema": SCHEMA, "reason": reason.strip(), "source_checkpoint": str(checkpoint),
            "source_checkpoint_sha256": file_sha(checkpoint), "source_manifest_sha256": file_sha(sidecar),
            "source_contract_sha256": digest(old), "target_contract_sha256": digest(new),
            "source_git_commit": old["source_git_commit"], "target_git_commit": new["source_git_commit"],
            "allowed_changed_files": delta, "changed_file_hashes": {path: {"before": old["files"].get(path), "after": new["files"][path]} for path in delta},
            "geometric_factor": geometric,
            "observation_dimension": (324 if layout_contract is None else layout_contract["observation_dimension"]),
            "action_dimension": (12 if layout_contract is None else layout_contract["action_dimension"]),
            "preserve_actor_critic_optimizer_normalizer_rng_and_budget": True,
            "discard_old_rollout_storage": True, "physics_resume": "fresh_legal_P01_reset"}
    if observation_contract is not None:
        result["instrumentation_observation_contract"] = observation_contract
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
    if timing is not None:
        result["nominal_timing_factor"] = timing
    if body_reward is not None:
        result["body_reward_factor"] = body_reward
    if height is not None:
        result["height_recovery_factor"] = height
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
                                        "reason": supplied["video_instrumentation_factor"]["review_reason"]},
                                    timing_review=None if "nominal_timing_factor" not in supplied else {
                                        "reason": supplied["nominal_timing_factor"]["review_reason"]},
                                    body_reward_review=None if "body_reward_factor" not in supplied else {
                                        "reason": supplied["body_reward_factor"]["review_reason"],
                                        "nominal_reference_manifest": supplied["body_reward_factor"]["nominal_reference_manifest"],
                                        "timing_plan": (supplied["body_reward_factor"].get("timing_ancestor") or {}).get("plan_path")},
                                    height_recovery_review=None if "height_recovery_factor" not in supplied else {
                                        "reason": supplied["height_recovery_factor"]["review_reason"],
                                        "reviewed_code_sha256": supplied["height_recovery_factor"]["reviewed_code_sha256"]},
                                    exploration_temperature_review=None if "exploration_temperature_factor" not in supplied else {
                                        "reason": supplied["exploration_temperature_factor"]["review_reason"],
                                        "reviewed_code_sha256": supplied["exploration_temperature_factor"]["reviewed_code_sha256"]},
                                    task_first_reward_review=None if "task_first_reward_factor" not in supplied else {
                                        "reason": supplied["task_first_reward_factor"]["review_reason"],
                                        "reviewed_code_sha256": supplied["task_first_reward_factor"]["reviewed_code_sha256"]},
                                    execution_composition_review=None if "execution_composition_factor" not in supplied else {
                                        "reason": supplied["execution_composition_factor"]["review_reason"],
                                        "reviewed_code_sha256": supplied["execution_composition_factor"]["reviewed_code_sha256"]},
                                    final_stop_handoff_review=None if "final_stop_handoff_factor" not in supplied else {
                                        "reason": supplied["final_stop_handoff_factor"]["review_reason"],
                                        "reviewed_code_sha256": supplied["final_stop_handoff_factor"]["reviewed_code_sha256"]},
                                    rr_physical_acceptance_review=None if "rr_physical_acceptance_same372_factor" not in supplied else {
                                        "reason": supplied["rr_physical_acceptance_same372_factor"]["review_reason"],
                                        "reviewed_code_sha256": supplied["rr_physical_acceptance_same372_factor"]["reviewed_code_sha256"]},
                                    fl_capture_quality_review=None if "fl_capture_quality_same372_factor" not in supplied else {
                                        "reason": supplied["fl_capture_quality_same372_factor"]["review_reason"],
                                        "reviewed_code_sha256": supplied["fl_capture_quality_same372_factor"]["reviewed_code_sha256"]},
                                    request_history_kernel_review=None if "request_history_kernel_factor" not in supplied else {
                                        "reason": supplied["request_history_kernel_factor"]["review_reason"],
                                        "reviewed_code_sha256": supplied["request_history_kernel_factor"]["reviewed_code_sha256"]},
                                    physical_innovation_sigma_review=None if "physical_innovation_sigma_factor" not in supplied else {
                                        "reason": supplied["physical_innovation_sigma_factor"]["review_reason"],
                                        "reviewed_code_sha256": supplied["physical_innovation_sigma_factor"]["reviewed_code_sha256"]},
                                    task_conditioned_hip_wheel_review=None if "task_conditioned_hip_wheel_factor" not in supplied else {
                                        "reason": supplied["task_conditioned_hip_wheel_factor"]["review_reason"],
                                        "reviewed_code_sha256": supplied["task_conditioned_hip_wheel_factor"]["reviewed_code_sha256"]},
                                    archive_only_exact_bytes_review=None if "archive_only_exact_bytes_factor" not in supplied else {
                                        "reason": supplied["archive_only_exact_bytes_factor"]["review_reason"]},
                                    training_quantity_budget_review=None if "training_quantity_budget_factor" not in supplied else {
                                        "reason": supplied["training_quantity_budget_factor"]["review_reason"],
                                        "reviewed_code_sha256": supplied["training_quantity_budget_factor"]["reviewed_code_sha256"]})
    if supplied != expected:
        raise ValueError("migration plan is not exactly bound to this immutable checkpoint and runtime")
    return {"plan_path": str(path), "plan_sha256": file_sha(path), **expected}


VECTOR_FILES = frozenset({
    "src/wlr50_clean/ppo/semantic_vector_backend.py",
    "src/wlr50_clean/ppo/semantic_vector_env.py",
    "src/wlr50_clean/ppo/semantic_vector_training.py",
})

def topology(num_envs: int, *, observation_layout: str | None = None) -> dict[str, Any]:
    if type(num_envs) is not int or num_envs not in (1, 8):
        raise ValueError("only single or eight-row semantic execution is reviewed")
    result = {"schema": "wlr50_clean.semantic_execution_topology.v1",
            "num_envs": num_envs, "observation_dimension": 324, "action_dimension": 12,
            "rollout_decisions_per_env": 128, "reset_sampling": "P01_only",
            "phase_suffix_curriculum_implemented": False,
            "peer_reset": "none" if num_envs == 1 else "synchronous_done_with_gamma_V_actual_final_obs",
            "task_timeout_bootstrap": False, "physical_state_saved": False}
    if observation_layout is not None:
        from .semantic_transfer_roles import ROLE_OBSERVATION_LAYOUT, ROLE_OBSERVATION_DIM
        if type(observation_layout) is not str or observation_layout != ROLE_OBSERVATION_LAYOUT or num_envs != 1:
            raise ValueError("role observation topology requires the explicit supported N1 layout")
        result.update(observation_layout=observation_layout, observation_dimension=ROLE_OBSERVATION_DIM)
    return result

def source_num_envs(metadata: Mapping[str, Any]) -> int:
    # This public validator does not call source_num_envs; topology validation
    # cannot infer372 from a dimension alone or trust an incomplete contract.
    observation_layout = None
    actor_config = (metadata.get("runner_config") or {}).get("actor", {})
    if "policy_contract" in metadata or "observation_layout" in actor_config:
        from .semantic_policy_distribution import policy_observation_layout_from_metadata
        observation_layout = policy_observation_layout_from_metadata(metadata)
    declared = metadata.get("execution_topology")
    if declared is not None:
        count = declared.get("num_envs")
        expected = topology(count, observation_layout=observation_layout)
        if metadata.get("semantic_version") == "v3":
            if count != 1:
                raise ValueError("v3 continuation topology must be N1")
            expected = continuation_topology(metadata.get("sampling"),
                                             metadata.get("curriculum_epoch", {}).get("prefix_request"),
                                             observation_layout=observation_layout)
        if declared != expected:
            raise ValueError("checkpoint execution topology is malformed")
        return count
    if observation_layout is not None:
        raise ValueError("role observation checkpoint lacks explicit execution topology")
    # Legacy semantic checkpoints existed only before vector code was added.
    # Never infer one row for a checkpoint whose runtime already had vector code.
    if any(path in metadata["runtime_contract"].get("files", {}) for path in VECTOR_FILES):
        raise ValueError("vector-capable checkpoint lacks explicit execution topology")
    return 1


def continuation_topology(sampling: str, prefix_request: Mapping[str, Any] | None, *,
                          observation_layout: str | None = None) -> dict[str, Any]:
    if prefix_request is None:
        if sampling != "P01_full_task_only_initial_version":
            raise ValueError("v3 P01 sampling metadata is malformed")
    elif prefix_request.get("schema") == "wlr50_clean.checkpoint_policy_prefix_request.v1":
        from .semantic_checkpoint_prefix import CheckpointPolicyPrefixRequest, sampling_label
        request = CheckpointPolicyPrefixRequest(**{name: prefix_request[name] for name in (
            "target_phase", "maximum_prefix_decisions", "teacher_offset_decisions", "source")})
        if request.as_dict() != dict(prefix_request) or sampling != sampling_label(request):
            raise ValueError("v3 checkpoint-policy prefix sampling metadata is malformed")
    else:
        from .semantic_prefix import PrefixRequest, sampling_label
        request = PrefixRequest(**{name: prefix_request[name] for name in (
            "target_phase", "maximum_prefix_decisions", "maximum_takeover_decisions", "teacher_offset_decisions")})
        if request.as_dict() != dict(prefix_request) or sampling != sampling_label(request):
            raise ValueError("v3 fixed suffix sampling metadata is malformed")
    return {**topology(1, observation_layout=observation_layout), "schema": "wlr50_clean.semantic_execution_topology.v2",
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
