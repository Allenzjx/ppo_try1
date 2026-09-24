"""Outputs-only synthetic checks for strict rear-policy export lineage."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "_rear_no_assist_export_tested", HERE / "export_policy_rear_no_assist_video.py")
assert SPEC is not None and SPEC.loader is not None
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


def fixture(tmp_path):
    initial_checkpoint = (HERE / "checkpoints/history/"
        "checkpoint_initial_rear_policy_step_000220544_gfa4b98ed506e.pt").resolve(strict=True)
    initial_sidecar = initial_checkpoint.with_name(
        initial_checkpoint.stem + "_manifest.json").resolve(strict=True)
    initial = exporter.read_json(initial_sidecar)
    old_migration = initial["rear_policy_timing_migration"]
    policy = initial["policy_contract"]
    runtime = {"source_git_commit": "4" * 40,
               "runtime_content_sha256": "5" * 64,
               "experiment_id": exporter.EXPERIMENT}
    counters = {key: exporter.RECAPTURE_SOURCE_SELECTION["counters"][key] + delta
                for key, delta in zip(exporter.COUNTERS, (128, 1, 20))}
    factor = {"schema": exporter.RECAPTURE_SCHEMA,
        "source_selection": copy.deepcopy(exporter.RECAPTURE_SOURCE_SELECTION),
        "source_mode": exporter.NOMINAL_TIMING,
        "target_mode": exporter.RECAPTURE_MODE,
        "revision_counter_origin": copy.deepcopy(
            exporter.RECAPTURE_SOURCE_SELECTION["counters"]),
        "original_branch_origin": copy.deepcopy(
            exporter.RECAPTURE_SOURCE_SELECTION["counters"]),
        "source_policy_contract": copy.deepcopy(policy),
        "target_policy_contract": copy.deepcopy(policy),
        "observation_dimension": 419, "action_dimension": 12, "num_envs": 1,
        "parameter_mapping": "identity_all_actor_critic_parameters_buffers_and_full_Adam",
        "physical_dynamics_changed": False, "same_mdp_claimed": False,
        "old_rollout_inherited": False, "added_policy_decisions": 0,
        "added_ppo_updates": 0, "added_optimizer_steps": 0,
        "added_auxiliary_updates": 0, "media_revision_review": None,
        "preserved_metadata_sha256": {
            "rear_policy_timing_migration": exporter.json_digest(old_migration)}}
    receipt = {"schema": exporter.RECAPTURE_SCHEMA, "reason": "synthetic outputs-only test",
        "source_selection": copy.deepcopy(exporter.RECAPTURE_SOURCE_SELECTION),
        "source_checkpoint": str(initial_checkpoint),
        "source_checkpoint_sha256": exporter.sha256(initial_checkpoint),
        "source_manifest_sha256": exporter.sha256(initial_sidecar),
        "source_contract_sha256": old_migration["target_contract_sha256"],
        "target_contract_sha256": exporter.json_digest(runtime),
        "source_runtime_content_sha256": old_migration["target_runtime_content_sha256"],
        "target_runtime_content_sha256": runtime["runtime_content_sha256"],
        "source_git_commit": old_migration["target_git_commit"],
        "target_git_commit": runtime["source_git_commit"],
        "allowed_changed_files": ["synthetic"],
        "changed_file_hashes": {"synthetic": {"before": "7" * 64, "after": "8" * 64}},
        "preserve_actor_critic_optimizer_normalizer_rng_and_budget": True,
        "discard_old_rollout_storage": True,
        "physics_resume": "fresh_legal_reset_not_bitwise_simulator_resume",
        exporter.RECAPTURE_FACTOR_KEY: factor}
    plan = tmp_path / "migration.json"
    plan.write_text(json.dumps(receipt), encoding="utf-8")
    receipt.update(plan_path=str(plan.resolve()), plan_sha256=exporter.sha256(plan))
    branch_root = (HERE / "branches" / exporter.RECAPTURE_BRANCH).resolve()
    routing = {"schema": "wlr50_clean.checkpoint_output_routing.v1",
        "branch": exporter.RECAPTURE_BRANCH, "output_root": str(branch_root),
        "main_latest_pointer_promotion": False,
        "source_selection": copy.deepcopy(exporter.RECAPTURE_SOURCE_SELECTION)}
    metadata = {"rear_recapture_migration": receipt,
                "checkpoint_output_routing": routing}
    checkpoint = branch_root / "checkpoints/history/checkpoint_step_000220672.pt"
    return metadata, runtime, old_migration, policy, counters, checkpoint


def reseal(receipt, path):
    payload = {key:value for key,value in receipt.items()
               if key not in ("plan_path", "plan_sha256")}
    path.write_text(json.dumps(payload), encoding="utf-8")
    receipt["plan_path"] = str(path.resolve())
    receipt["plan_sha256"] = exporter.sha256(path)


def overlay_decision_fixture():
    baseline = [0.0] * 8 + [0.3, 0.3, 0.3, 0.3]
    final = [0.0] * 8 + [-0.7, 0.24, 0.29, 0.19]
    baseline_physical = [
        exporter.WHEEL_FORWARD_SIGN[name] * baseline[8 + index]
        for index, name in enumerate(exporter.WHEEL_ORDER)]
    final_physical = [
        exporter.WHEEL_FORWARD_SIGN[name] * final[8 + index]
        for index, name in enumerate(exporter.WHEEL_ORDER)]
    return {"decision": 7, "end_tick": 44, "environment_step_returned": True,
        "step_info": {"physics_tick": 44, "end_phase_id": "P09",
            "nominal_action_full12": baseline,
            "actual_drive_target_full12": final,
            "actuator_target_effect_audit": {
                "schema": "wlr50_clean.actuator_target_effect_audit.v1",
                "verified": True, "actual_mapping_matches_dispatch": True,
                "setter_dispatch_targets_equal": True, "physics_tick": 223,
                "native_drive_target_full12": baseline,
                "policy_headroom_evidence": {
                    "baseline_native_plus_controller_full12": baseline},
                "counterfactual_native_targets": {
                    "wheel_velocity_rad_s": baseline_physical},
                "actual_native_targets": {
                    "wheel_velocity_rad_s": final_physical}},
            "semantic_task": {"nominal_provider_diagnostics": {
                "source_partial_order": {"layers": [{"stage": "P09",
                    "observation_tick": 44, "status": "holding",
                    "source_ticks": 648, "late_group_start_tick": None,
                    "wait_reason": "no_live_physical_readiness"}]}}}}}


def test_overlay_decision_receipt_separates_mapped_N_and_source_clock():
    receipt = exporter.overlay_decision_receipt(overlay_decision_fixture())
    assert receipt["episode_end_tick"] == 44
    assert receipt["actuator_receipt_raw_tick"] == 223
    assert receipt["mapped_N_baseline_canonical_full12"][8:] == [0.3] * 4
    assert receipt["N_baseline_is_counterfactual_not_executed_alone"] is True
    assert receipt["source_partial_order"] == {
        "available": True, "stage": "P09", "status": "holding",
        "source_ticks": 648, "late_group_start_tick": None,
        "observation_tick": 44, "wait_reason": "no_live_physical_readiness",
        "semantics": "source_partial_order_current_phase_not_public_dependency_flag"}


def test_overlay_decision_receipt_rejects_tampered_baseline():
    decision = overlay_decision_fixture()
    decision["step_info"]["actuator_target_effect_audit"][
        "native_drive_target_full12"][8] = 0.2
    with pytest.raises(RuntimeError):
        exporter.overlay_decision_receipt(decision)


def test_RL_contact_mode_is_tristate_and_never_reclassified():
    actual = exporter.rl_contact_summary({"contact_mode": "GROUND_AND_OBSTACLE",
        "contact_surface": "FRONT_WALL", "bearing_verified": False,
        "obstacle_normal_force_n": 3.0, "top_surface_contact": False,
        "ground_contact": True})
    assert actual["contact_mode"] == "GROUND_AND_OBSTACLE"
    assert actual["contact_surface"] == "FRONT_WALL"
    assert actual["bearing_verified_raw"] is False
    assert actual["ground_and_obstacle_is_body_collision_or_pure_wheel_climb_claim"] is False
    missing = exporter.rl_contact_summary({"ground_contact": True})
    assert missing["contact_mode"] is None
    assert missing["bearing_verified_raw"] is None
    assert missing["front_schema_fallback_is_explicit_not_invented"] is True


def live_fixture(tmp_path):
    metadata, runtime, old_migration, policy, counters, checkpoint = fixture(tmp_path)
    source_runtime = copy.deepcopy(runtime)
    source_runtime.update(source_git_commit=exporter.LIVE_SWING_SOURCE_HEAD,
        files={"src/base.py":"1" * 64, "src/live.py":"2" * 64},
        selected_configuration={
            "execution_profile.yaml":{"path":"configs/execution_profile.yaml",
                                      "sha256":"3" * 64},
            "stage_task_spec.yaml":{"path":"configs/stage_task_spec.yaml",
                                    "sha256":"4" * 64}})
    source_runtime["runtime_content_sha256"] = exporter.json_digest(source_runtime["files"])
    recapture = metadata["rear_recapture_migration"]
    recapture.update(target_git_commit=source_runtime["source_git_commit"],
        target_runtime_content_sha256=source_runtime["runtime_content_sha256"],
        target_contract_sha256=exporter.json_digest(source_runtime))
    reseal(recapture, tmp_path / "recapture_migration.json")

    runtime = copy.deepcopy(source_runtime)
    runtime["source_git_commit"] = "8" * 40
    runtime["files"]["src/live.py"] = "5" * 64
    runtime["selected_configuration"]["execution_profile.yaml"] = {
        "path":"configs/execution_profile.yaml", "sha256":"6" * 64}
    runtime["selected_configuration"]["stage_task_spec.yaml"] = {
        "path":"configs/stage_task_spec.yaml", "sha256":"7" * 64}
    runtime["runtime_content_sha256"] = exporter.json_digest(runtime["files"])
    counters = {key:exporter.LIVE_SWING_ORIGIN[key] + delta
                for key,delta in zip(exporter.COUNTERS, (128, 1, 20))}
    source = (HERE / "branches" / exporter.RECAPTURE_BRANCH / "checkpoints" /
        "history" / "checkpoint_step_000221568.pt").resolve(strict=True)
    source_manifest = source.with_name(source.stem + "_manifest.json").resolve(strict=True)
    factor = {"schema":exporter.LIVE_SWING_SCHEMA,
        "source_mode":exporter.RECAPTURE_MODE,
        "target_mode":exporter.LIVE_SWING_MODE,
        "revision_counter_origin":copy.deepcopy(exporter.LIVE_SWING_ORIGIN),
        "original_branch_origin":copy.deepcopy(
            exporter.RECAPTURE_SOURCE_SELECTION["counters"]),
        "source_policy_contract":copy.deepcopy(policy),
        "target_policy_contract":copy.deepcopy(policy),
        "observation_dimension":419, "action_dimension":12, "num_envs":1,
        "parameter_mapping":"identity_all_actor_critic_buffers_full_Adam_and_rng",
        "same_numeric_input_Gaussian_preserved":True,
        "same_mdp_claimed":False, "physical_dynamics_changed":False,
        "old_rollout_inherited":False, "added_policy_decisions":0,
        "added_ppo_updates":0, "added_optimizer_steps":0,
        "added_auxiliary_updates":0,
        "preserved_metadata_sha256":{
            "rear_recapture_migration":exporter.json_digest(recapture),
            "checkpoint_output_routing":exporter.json_digest(
                metadata["checkpoint_output_routing"])}}
    receipt = {"schema":exporter.LIVE_SWING_SCHEMA,
        "reason":"synthetic outputs-only live-swing test",
        "source_checkpoint":str(source),
        "source_checkpoint_sha256":exporter.sha256(source),
        "source_manifest_sha256":exporter.sha256(source_manifest),
        "source_git_commit":source_runtime["source_git_commit"],
        "target_git_commit":runtime["source_git_commit"],
        "source_runtime_content_sha256":source_runtime["runtime_content_sha256"],
        "target_runtime_content_sha256":runtime["runtime_content_sha256"],
        "source_contract_sha256":exporter.json_digest(source_runtime),
        "target_contract_sha256":exporter.json_digest(runtime),
        "changed_file_hashes":{"src/live.py":{
            "before":source_runtime["files"]["src/live.py"],
            "after":runtime["files"]["src/live.py"]}},
        "source_changed_configuration":copy.deepcopy(
            source_runtime["selected_configuration"]),
        "discard_old_rollout_storage":True,
        "source_role":"latest_sealed_learned_recapture_branch_not_ancestor",
        exporter.LIVE_SWING_FACTOR_KEY:factor}
    reseal(receipt, tmp_path / "live_swing_migration.json")
    metadata["rear_live_swing_migration"] = receipt
    checkpoint = (HERE / "branches" / exporter.RECAPTURE_BRANCH / "checkpoints" /
                  "history" / "checkpoint_step_000221696.pt")
    return metadata, runtime, source_runtime, old_migration, policy, counters, checkpoint


def test_exact_runtime_recapture_v2_receipt_and_branch_routing(tmp_path):
    values = fixture(tmp_path)
    result = exporter.rear_recapture_identity(*values)
    assert result["schema"] == exporter.RECAPTURE_SCHEMA
    assert result["target_mode"] == exporter.RECAPTURE_MODE
    assert result["revision_branch_counts"] == {
        "global_policy_decisions": 128, "ppo_updates": 1, "optimizer_steps": 20}
    assert result["checkpoint_output_routing"]["branch"] == exporter.RECAPTURE_BRANCH
    assert result["new_auxiliary_updates"] == 0


@pytest.mark.parametrize("fault", [
    "selection", "learning", "runtime", "source_runtime", "route", "path",
    "counts", "media", "missing_migration",
])
def test_recapture_v2_export_fails_closed(tmp_path, fault):
    values = list(fixture(tmp_path))
    metadata, runtime = values[0], values[1]
    receipt = metadata["rear_recapture_migration"]
    factor = receipt[exporter.RECAPTURE_FACTOR_KEY]
    if fault == "selection":
        factor["source_selection"]["output_branch"] = "other"
    elif fault == "learning":
        factor["added_ppo_updates"] = 1
    elif fault == "runtime":
        runtime["source_git_commit"] = "9" * 40
    elif fault == "source_runtime":
        receipt["source_runtime_content_sha256"] = "9" * 64
    elif fault == "route":
        metadata["checkpoint_output_routing"]["main_latest_pointer_promotion"] = True
    elif fault == "path":
        values[-1] = HERE / "checkpoints/history/checkpoint_step_000220672.pt"
    elif fault == "counts":
        values[4]["ppo_updates"] = exporter.RECAPTURE_SOURCE_SELECTION["counters"]["ppo_updates"] - 1
    elif fault == "missing_migration":
        metadata.pop("rear_recapture_migration")
    else:
        factor["media_revision_review"] = {"schema": "invented"}
    with pytest.raises(RuntimeError):
        exporter.rear_recapture_identity(*values)


def test_exact_runtime_defaults_checkpoint_head_and_needs_no_media_receipt():
    runtime = {"source_git_commit": "4" * 40,
               "runtime_content_sha256": "5" * 64}
    args = NS(expected_head="4" * 40, checkpoint_runtime_head=None)
    manifest = {"runtime_contract": runtime,
                "checkpoint_load_provenance": {}}
    metadata = {"runtime_contract": copy.deepcopy(runtime)}
    assert exporter.checkpoint_runtime_head(args) == "4" * 40
    assert exporter.media_only_runtime_compatibility(
        manifest, metadata, {}, args) is None


def test_split_runtime_still_requires_explicit_checkpoint_head():
    args = NS(expected_head="4" * 40, checkpoint_runtime_head=None)
    manifest = {"runtime_contract": {"source_git_commit": "4" * 40,
        "runtime_content_sha256": "5" * 64}, "checkpoint_load_provenance": {}}
    metadata = {"runtime_contract": {"source_git_commit": "3" * 40,
        "runtime_content_sha256": "6" * 64}}
    with pytest.raises(RuntimeError):
        exporter.media_only_runtime_compatibility(manifest, metadata, {}, args)


def test_live_swing_reconstructs_parent_runtime_then_accepts_descendant(tmp_path):
    metadata, runtime, wanted_source, old_migration, policy, counters, checkpoint = live_fixture(tmp_path)
    source = exporter.reconstruct_live_swing_source_runtime(
        runtime, metadata["rear_live_swing_migration"])
    assert source == wanted_source
    recapture = exporter.rear_recapture_identity(
        metadata, source, old_migration, policy, counters, checkpoint)
    live = exporter.rear_live_swing_identity(
        metadata, runtime, source, policy, counters, checkpoint, recapture)
    assert live["target_mode"] == exporter.LIVE_SWING_MODE
    assert live["revision_branch_counts"] == {
        "global_policy_decisions":128, "ppo_updates":1, "optimizer_steps":20}
    assert live["added_auxiliary_updates"] == 0


@pytest.mark.parametrize("fault", ["delta", "source", "mode", "learning", "parent"])
def test_live_swing_export_fails_closed(tmp_path, fault):
    metadata, runtime, source, old_migration, policy, counters, checkpoint = live_fixture(tmp_path)
    receipt = metadata["rear_live_swing_migration"]
    factor = receipt[exporter.LIVE_SWING_FACTOR_KEY]
    if fault == "delta":
        runtime["files"]["src/live.py"] = "9" * 64
    elif fault == "source":
        receipt["source_checkpoint_sha256"] = "9" * 64
    elif fault == "mode":
        factor["target_mode"] = "invented"
    elif fault == "learning":
        factor["added_ppo_updates"] = 1
    else:
        factor["preserved_metadata_sha256"]["rear_recapture_migration"] = "9" * 64
    with pytest.raises(RuntimeError):
        historical = exporter.reconstruct_live_swing_source_runtime(runtime, receipt)
        recapture = exporter.rear_recapture_identity(
            metadata, historical, old_migration, policy, counters, checkpoint)
        exporter.rear_live_swing_identity(
            metadata, runtime, historical, policy, counters, checkpoint, recapture)


def p02_progress_fixture(tmp_path):
    source_checkpoint = (HERE / "branches" / exporter.RECAPTURE_BRANCH /
        "checkpoints" / "history" / "checkpoint_step_000221696.pt").resolve(strict=True)
    source_manifest = source_checkpoint.with_name(
        source_checkpoint.stem + "_manifest.json").resolve(strict=True)
    source_policy = exporter.read_json(source_manifest)["policy_contract"]
    target_policy = copy.deepcopy(source_policy)
    target_policy.update(
        version=exporter.P02_PROGRESS_POLICY_VERSION,
        actor_class=("wlr50_clean.ppo.semantic_p02_progress_actor:"
                     "SemanticP02ProgressHistoryMLPModel"),
        observation_layout=exporter.P02_PROGRESS_OBSERVATION_LAYOUT,
        observation_dimension=422,
        preserved_observation_prefix_dimension=419,
        p02_progress_mode=exporter.P02_PROGRESS_MODE,
        p02_progress_observation_group=exporter.P02_PROGRESS_GROUP,
        p02_progress_observation_slice=[419, 422],
        p02_progress_observation_fields=list(exporter.P02_PROGRESS_FIELDS),
        p02_progress_feature_scales=[1.0, 1.0, 1.0],
        old_419_numerical_codec_preserved=True,
        sigma_kernel_observation_slice=[0, 419],
    )
    configuration_names = (
        "execution_profile.yaml", "stage_task_spec.yaml",
        "observation_schema.json", "curriculum_plan.json",
    )
    source_configuration = {name: {
        "path": "configs/" + name, "sha256": format(index + 1, "064x")}
        for index, name in enumerate(configuration_names)}
    source_files = {entry["path"]: entry["sha256"]
                    for entry in source_configuration.values()}
    source_files["src/wlr50_clean/ppo/semantic_p02_progress_actor.py"] = "a" * 64
    source_runtime = {
        "source_git_commit": exporter.P02_PROGRESS_SOURCE_HEAD,
        "runtime_content_sha256": "b" * 64,
        "experiment_id": exporter.EXPERIMENT,
        "files": source_files,
        "selected_configuration": copy.deepcopy(source_configuration),
    }
    runtime = copy.deepcopy(source_runtime)
    runtime["source_git_commit"] = "d" * 40
    runtime["runtime_content_sha256"] = "e" * 64
    changed = {}
    for index, (path, before) in enumerate(source_files.items()):
        after = format(index + 20, "064x")
        runtime["files"][path] = after
        changed[path] = {"before": before, "after": after}
    for name in configuration_names:
        runtime["selected_configuration"][name]["sha256"] = runtime["files"][
            source_configuration[name]["path"]]
    route = {"schema": "wlr50_clean.checkpoint_output_routing.v1",
        "branch": exporter.RECAPTURE_BRANCH,
        "output_root": str((HERE / "branches" / exporter.RECAPTURE_BRANCH).resolve()),
        "main_latest_pointer_promotion": False,
        "source_selection": copy.deepcopy(exporter.RECAPTURE_SOURCE_SELECTION)}
    metadata = {
        "rear_live_swing_migration": {"preserved": "live"},
        "rear_recapture_migration": {"preserved": "recapture"},
        "rear_policy_timing_migration": {"preserved": "initial"},
        "rear_policy_timing_branch": {"preserved": "branch"},
        "checkpoint_output_routing": route,
    }
    protected = {key: exporter.json_digest(metadata[key]) for key in (
        "rear_live_swing_migration", "rear_recapture_migration",
        "rear_policy_timing_migration", "rear_policy_timing_branch",
        "checkpoint_output_routing")}
    factor = {"schema": exporter.P02_PROGRESS_SCHEMA,
        "source_policy_contract": source_policy,
        "target_policy_contract": target_policy,
        "revision_counter_origin": copy.deepcopy(exporter.P02_PROGRESS_ORIGIN),
        "original_branch_origin": copy.deepcopy(
            exporter.RECAPTURE_SOURCE_SELECTION["counters"]),
        "preserved_metadata_sha256": protected,
        "actor_critic_mapping": "old419_columns_exact_new3_zero",
        "Adam_mapping": ("all_old_moments_steps_options_exact_new3_"
                         "first_weight_moment_columns_zero"),
        "same_mdp_claimed": False, "physical_dynamics_changed": False,
        "rear_control_changed": False,
        "raw_sigma_kernel": ("unchanged_rear419_slice_0_419; "
                             "new_features_only_enter_MLP"),
        "old_rollout_inherited": False, "added_policy_decisions": 0,
        "added_ppo_updates": 0, "added_optimizer_steps": 0,
        "added_auxiliary_updates": 0}
    receipt = {"schema": exporter.P02_PROGRESS_SCHEMA,
        "reason": "synthetic outputs-only 419-to-422 test",
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": exporter.sha256(source_checkpoint),
        "source_manifest_sha256": exporter.sha256(source_manifest),
        "source_git_commit": source_runtime["source_git_commit"],
        "target_git_commit": runtime["source_git_commit"],
        "source_runtime_content_sha256": source_runtime["runtime_content_sha256"],
        "target_runtime_content_sha256": runtime["runtime_content_sha256"],
        "source_contract_sha256": exporter.json_digest(source_runtime),
        "target_contract_sha256": exporter.json_digest(runtime),
        "changed_file_hashes": changed,
        "source_changed_configuration": source_configuration,
        "discard_old_rollout_storage": True,
        exporter.P02_PROGRESS_FACTOR_KEY: factor}
    reseal(receipt, tmp_path / "p02_progress_migration.json")
    metadata[exporter.P02_PROGRESS_MIGRATION] = receipt
    counters = {key: exporter.P02_PROGRESS_ORIGIN[key] + delta
                for key, delta in zip(exporter.COUNTERS, (128, 1, 20))}
    checkpoint = (HERE / "branches" / exporter.RECAPTURE_BRANCH /
                  "checkpoints" / "history" / "checkpoint_step_000221824.pt")
    live_swing = {"checkpoint_output_routing": route}
    return (metadata, runtime, source_runtime, source_policy, target_policy,
            counters, checkpoint, live_swing)


def test_p02_progress_reconstructs_419_parent_and_accepts_422_descendant(tmp_path):
    values = p02_progress_fixture(tmp_path)
    metadata, runtime, wanted_source = values[:3]
    source = exporter.reconstruct_p02_progress_source_runtime(
        runtime, metadata[exporter.P02_PROGRESS_MIGRATION])
    assert source == wanted_source
    result = exporter.p02_progress_identity(
        metadata, runtime, source, *values[3:])
    assert result["policy_version"] == exporter.P02_PROGRESS_POLICY_VERSION
    assert result["observation_dimension"] == 422
    assert result["revision_branch_counts"] == {
        "global_policy_decisions": 128, "ppo_updates": 1, "optimizer_steps": 20}
    assert result["added_auxiliary_updates"] == 0


@pytest.mark.parametrize("fault", [
    "delta", "source", "policy", "learning", "parent", "route", "counts",
])
def test_p02_progress_export_fails_closed(tmp_path, fault):
    values = list(p02_progress_fixture(tmp_path))
    metadata, runtime = values[0], values[1]
    receipt = metadata[exporter.P02_PROGRESS_MIGRATION]
    factor = receipt[exporter.P02_PROGRESS_FACTOR_KEY]
    if fault == "delta":
        runtime["files"][next(iter(receipt["changed_file_hashes"]))] = "f" * 64
    elif fault == "source":
        receipt["source_checkpoint_sha256"] = "f" * 64
    elif fault == "policy":
        values[4]["observation_dimension"] = 419
    elif fault == "learning":
        factor["added_ppo_updates"] = 1
    elif fault == "parent":
        factor["preserved_metadata_sha256"]["rear_live_swing_migration"] = "f" * 64
    elif fault == "route":
        values[7] = {"checkpoint_output_routing": {"branch": "other"}}
    else:
        values[5]["ppo_updates"] = exporter.P02_PROGRESS_ORIGIN["ppo_updates"] - 1
    with pytest.raises(RuntimeError):
        source = exporter.reconstruct_p02_progress_source_runtime(runtime, receipt)
        exporter.p02_progress_identity(
            metadata, runtime, source, *values[3:])


def cooperative_checkpoint_fixture():
    checkpoint = (HERE / "branches" / exporter.RECAPTURE_BRANCH / "checkpoints" /
        "history" / "checkpoint_cooperative_prep_CP223232_g49eb23163a6e.pt").resolve(strict=True)
    sidecar = checkpoint.with_name(checkpoint.stem + "_manifest.json").resolve(strict=True)
    metadata = exporter.read_json(sidecar)
    runtime = metadata["runtime_contract"]
    manifest = {
        "policy_sampling_mode": "deterministic_conditional_mean",
        "runtime_contract": copy.deepcopy(runtime),
        "checkpoint_load_provenance": {
            "checkpoint_loaded_and_verified": True,
            "stochastic_policy": False,
            "policy_seed": None,
            "saved_global_policy_decisions": metadata["global_policy_decisions"],
            "source": {"checkpoint": str(checkpoint),
                "checkpoint_sha256": exporter.sha256(checkpoint),
                "manifest": str(sidecar),
                "manifest_sha256": exporter.sha256(sidecar)},
            "parameter_hashes": {
                "actor_parameter_sha256": metadata["actor_parameter_sha256"]},
        },
    }
    args = NS(checkpoint=checkpoint,
        checkpoint_sha256=exporter.sha256(checkpoint),
        checkpoint_manifest_sha256=exporter.sha256(sidecar),
        expected_head=runtime["source_git_commit"], checkpoint_runtime_head=None,
        expected_global_policy_decisions=metadata["global_policy_decisions"],
        expected_ppo_updates=metadata["ppo_updates"],
        expected_optimizer_steps=metadata["optimizer_steps"],
        expected_new_auxiliary_updates=0)
    return manifest, args, metadata


def test_actual_published_cooperative_checkpoint_identity_is_strict():
    manifest, args, metadata = cooperative_checkpoint_fixture()
    result = exporter.checkpoint_identity(manifest, args)
    revision = result["cooperative_preparation_revision"]
    assert result["policy_version"] == exporter.COOPERATIVE_PREP_POLICY_VERSION
    assert result["p02_progress_revision"] is not None
    assert revision["schema"] == exporter.COOPERATIVE_PREP_SCHEMA
    assert revision["cooperative_preparation_revision"] == (
        exporter.COOPERATIVE_PREPARATION_REVISION)
    assert revision["revision_branch_counts"] == dict.fromkeys(exporter.COUNTERS, 0)
    assert revision["source_checkpoint_sha256"] == exporter.COOPERATIVE_PREP_SOURCE_SHA
    assert metadata[exporter.COOPERATIVE_PREP_MIGRATION][
        exporter.COOPERATIVE_PREP_FACTOR_KEY]["added_auxiliary_updates"] == 0


@pytest.mark.parametrize("fault", ["policy", "learning", "source", "parent", "counts"])
def test_cooperative_checkpoint_identity_fails_closed(fault):
    manifest, args, metadata = cooperative_checkpoint_fixture()
    receipt = metadata[exporter.COOPERATIVE_PREP_MIGRATION]
    factor = receipt[exporter.COOPERATIVE_PREP_FACTOR_KEY]
    if fault == "policy":
        factor["target_policy_contract"]["version"] = "invented"
    elif fault == "learning":
        factor["added_ppo_updates"] = 1
    elif fault == "source":
        receipt["source_checkpoint_sha256"] = "f" * 64
    elif fault == "parent":
        factor["preserved_metadata_sha256"][exporter.P02_PROGRESS_MIGRATION] = "f" * 64
    else:
        args.expected_ppo_updates = exporter.COOPERATIVE_PREP_ORIGIN["ppo_updates"] - 1
    # Write no altered sidecar: exercise the strict leaf directly on the copy.
    runtime = metadata["runtime_contract"]
    source_runtime = exporter.reconstruct_cooperative_prep_source_runtime(runtime, receipt)
    progress_receipt = metadata[exporter.P02_PROGRESS_MIGRATION]
    base_runtime = exporter.reconstruct_p02_progress_source_runtime(
        source_runtime, progress_receipt)
    progress_factor = progress_receipt[exporter.P02_PROGRESS_FACTOR_KEY]
    counters = {key: getattr(args, "expected_" + key) for key in exporter.COUNTERS}
    with pytest.raises(RuntimeError):
        exporter.cooperative_prep_identity(
            metadata, runtime, source_runtime, factor.get("source_policy_contract", {}),
            metadata["policy_contract"], counters, Path(args.checkpoint),
            {"checkpoint_output_routing": metadata["checkpoint_output_routing"]})
