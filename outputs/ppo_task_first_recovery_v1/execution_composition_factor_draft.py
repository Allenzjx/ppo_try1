"""DRAFT ONLY: insertion candidate for semantic_migration.py after safe boundary.

This file is not imported by production and performs no I/O at import time.
The unresolved helpers/types below intentionally refer to the production module.
"""

EXECUTION_COMPOSITION_SCHEMA = "wlr50_clean.independent_post_mapper_residual_same372_fix.v1"
EXECUTION_COMPOSITION_FILES = frozenset(f"src/wlr50_clean/ppo/{name}.py" for name in (
    "semantic_residual_adapter", "semantic_nominal_geometry", "semantic_migration",
    "semantic_training", "semantic_cli"))
EXECUTION_COMPOSITION_CORE_FILES = frozenset(f"src/wlr50_clean/ppo/{name}.py" for name in (
    "semantic_residual_adapter", "semantic_nominal_geometry"))


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
