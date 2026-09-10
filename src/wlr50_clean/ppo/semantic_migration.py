"""Explicit, immutable semantic checkpoint version boundaries, without Isaac."""
from __future__ import annotations

import hashlib
import json
import math
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
    if experiment_id not in ("transfer_roles_v1", "all_stage_acceptance_v1", "fsm_reference_p09_stable_v2") or semantic_version != "v3":
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
        observation_same_layout_transition = _fsm_reference_p09_same372_transition(
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
        if observation_same_layout_transition["schema"] == FSM_REFERENCE_P09_SCHEMA:
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
            "target_phase", "maximum_prefix_decisions", "teacher_offset_decisions")})
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
