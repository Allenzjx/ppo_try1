from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from copy import deepcopy
from unittest.mock import patch


MODULE_PATH = Path(__file__).with_name("export_p05_capture_video.py")
SPEC = importlib.util.spec_from_file_location("export_p05_capture_video", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


def tick_row(tick: int, *, phase: str = "P05", pending: bool = False,
             mode: int = 1, mode_name: str = "DESCEND") -> dict:
    state = {
        "schema": "wlr50_clean.capture_assist_state.v1",
        "mode": float(mode),
        "mode_name": mode_name,
        "active": mode in (1, 2, 3, 4),
        "initialized": 1.0 if mode else 0.0,
        "hip_target_deg": -12.5,
        "knee_hold_deg": 31.0,
        "hip_entry_deg": -10.0,
        "reason": "none",
    }
    joints = {
        name: {"position_deg": float(index)}
        for index, name in enumerate(exporter.SERVO_ORDER)
    }
    wheels = {
        name: {"velocity_rad_s": value}
        for name, value in zip(exporter.WHEEL_ORDER, (-0.1, 0.2, -0.3, 0.4), strict=True)
    }
    return {
        "episode_physics_tick": tick,
        "sim_time_s": tick / exporter.HZ,
        "phase": phase,
        "capture_assist": state,
        "capture_continuation": pending,
        "fl_capture_pending": pending,
        "placed_history": {"FL": False},
        "current_legs": {"FL": {
            "clearance_m": 0.0075,
            "air": True,
            "top_surface_contact": False,
            "contact_surface": "NONE",
            "contact_reaction_force_n": 0.0,
        }},
        "dispatch": {
            "drive_target_full12": [float(index) for index in range(12)],
            # Deliberately conflicting nested state: overlays must use the
            # observable top-level state captured after the physical tick.
            "capture_assist_evidence": {"state_after": {"hip_target_deg": 999.0}},
        },
        "actual_full12": [0.0] * 12,
        "joints": joints,
        "wheels": wheels,
    }


class ExportP05CaptureVideoTests(unittest.TestCase):
    def feedback_metadata(self) -> dict:
        origin = {"global_policy_decisions": 203776, "ppo_updates": 1557,
                  "optimizer_steps": 31140}
        current = {"global_policy_decisions": 203776, "ppo_updates": 1557,
                   "optimizer_steps": 31140}
        source_hash = "a" * 64
        factor = {"schema": exporter.FEEDBACK_SCHEMA,
                  "target_feedback_revision": exporter.FEEDBACK_REVISION,
                  "counter_origin": deepcopy(origin)}
        return {**current,
            "p05_capture_assist_branch": {"counter_origin": deepcopy(exporter.P05_COUNTER_ORIGIN)},
            "p05_capture_assist_branch_counts": {
                key: current[key] - exporter.P05_COUNTER_ORIGIN[key] for key in exporter.COUNTERS},
            "capture_feedback_semantics_branch": {
                "schema": exporter.FEEDBACK_SCHEMA,
                "feedback_revision": exporter.FEEDBACK_REVISION,
                "counter_origin": deepcopy(origin),
                "source_checkpoint_sha256": source_hash,
                "migration_added_updates": 0,
            },
            "capture_feedback_semantics_branch_counts": {key: 0 for key in exporter.COUNTERS},
            "capture_feedback_semantics_migration": {
                "schema": exporter.FEEDBACK_SCHEMA,
                "source_checkpoint_sha256": source_hash,
                "capture_feedback_semantics_factor": factor,
            }}

    def rr_workspace_metadata(self, directory: Path) -> dict:
        metadata = self.feedback_metadata()
        origin = {key: metadata[key] for key in exporter.COUNTERS}
        metadata.update({
            "stage_requested_decisions": 128,
            "new_mdp_origin_global_policy_decisions": exporter.P05_COUNTER_ORIGIN[
                "global_policy_decisions"],
            "p05_capture_assist_migration": {"immutable": "p05 source migration"},
            "task_conditioned_hip_wheel_branch": {
                "immutable": "task branch",
                "auxiliary_mean_learning": {
                    "accepted_auxiliary_updates_total": 7,
                    "attempted_auxiliary_optimizer_steps_total": 8,
                },
            },
            "actor_parameter_sha256": "1" * 64,
            "critic_parameter_sha256": "2" * 64,
            "optimizer_state_sha256": "3" * 64,
            "normalizer_state_sha256": "4" * 64,
            "normalization": {"actor": "Identity", "critic": "Identity"},
            "training_rng_state": {"cpu": "synthetic unit fixture"},
            "optimizer_learning_rate": 1e-5,
            "runner_config": {"device": "cpu"},
            "policy_contract": {"observation_dimension": 389},
        })
        source_checkpoint = directory / "source_checkpoint.pt"
        source_checkpoint.write_bytes(b"synthetic checkpoint bytes")
        source_hash = exporter.sha256(source_checkpoint)
        source_manifest = source_checkpoint.with_name(source_checkpoint.stem + "_manifest.json")
        source_metadata = deepcopy(metadata)
        source_metadata.update({
            "checkpoint_path": str(source_checkpoint.resolve()),
            "checkpoint_sha256": source_hash,
        })
        source_manifest.write_text(json.dumps(source_metadata), encoding="utf-8")
        preserved = {
            key: exporter.json_digest(source_metadata[key])
            for key in exporter.RR_WORKSPACE_REQUIRED_HISTORY
        }
        factor = {
            "schema": exporter.RR_WORKSPACE_SCHEMA,
            "target_semantics": exporter.RR_WORKSPACE_SEMANTICS,
            "source_semantics": exporter.RR_WORKSPACE_SOURCE_SEMANTICS,
            "task_spec_change": {
                exporter.RR_WORKSPACE_MODE_KEY: exporter.RR_WORKSPACE_SEMANTICS},
            "counter_origin": deepcopy(origin),
            "preserved_metadata_sha256": preserved,
            "observation_contract": {
                "observation_dimension": 389,
                "action_dimension": 12,
                "observation_layout": "role372_p05_capture_assist_v1",
            },
            "observation_semantics_changed": [{
                "index": 17, "field": "task_progress_potential",
                "change": "same physical state may encode corrected RR post-cross workspace potential",
            }],
            "reward_changed": True,
            "same_mdp_claimed": False,
            "physical_dynamics_changed": False,
            "same_numeric_input_policy_mapping_preserved": True,
            "same_physical_state_action_equivalence_claimed": False,
            "policy_kernel_changed": False,
            "controller_predicates_changed": False,
            "nominal_changed": False,
            "capture_assist_changed": False,
            "caps_changed": False,
            "sigma_changed": False,
            "reward_coefficients_and_return_profile_changed": False,
            "old_values_are_new_reward_ground_truth": False,
            "discard_old_rollout_storage": True,
            "physical_state_inherited": False,
            "added_policy_decisions": 0,
            "added_ppo_updates": 0,
            "added_optimizer_steps": 0,
            "added_auxiliary_updates": 0,
        }
        metadata.update({
            "rr_postcross_workspace_branch": {
                "schema": exporter.RR_WORKSPACE_SCHEMA,
                "semantics": exporter.RR_WORKSPACE_SEMANTICS,
                "counter_origin": deepcopy(origin),
                "source_checkpoint_sha256": source_hash,
                "migration_added_updates": 0,
            },
            "rr_postcross_workspace_branch_counts": {
                key: 0 for key in exporter.COUNTERS},
            "rr_postcross_workspace_migration": {
                "schema": exporter.RR_WORKSPACE_SCHEMA,
                "source_checkpoint": str(source_checkpoint.resolve()),
                "source_checkpoint_sha256": source_hash,
                "source_manifest_sha256": exporter.sha256(source_manifest),
                "rr_postcross_workspace_factor": factor,
            },
        })
        return metadata

    def front_rehearsal_metadata(self, directory: Path) -> dict:
        source_metadata = self.rr_workspace_metadata(directory)
        source_checkpoint = directory / "front_rehearsal_source.pt"
        source_checkpoint.write_bytes(b"synthetic front rehearsal source checkpoint")
        source_hash = exporter.sha256(source_checkpoint)
        source_metadata.update({
            "checkpoint_path": str(source_checkpoint.resolve()),
            "checkpoint_sha256": source_hash,
        })
        source_manifest = source_checkpoint.with_name(
            source_checkpoint.stem + "_manifest.json")
        source_manifest.write_text(json.dumps(source_metadata), encoding="utf-8")

        data_file = directory / "reviewed_raw_actions.jsonl"
        data_file.write_text("{}\n", encoding="utf-8")
        data_receipt = {
            "schema": exporter.FRONT_AUX_DATA_SCHEMA,
            "front_source_count": 280,
            "raw_targets_unmodified": True,
            "stored_mean_is_diagnostic_not_executed_target": True,
            "teacher_deployed": False,
            "new_PPO_credit": 0,
            "new_auxiliary_credit": 0,
            "groups": {
                "train": {"count": 95},
                "validation": {"count": 93},
                "invariance": {"count": 13},
            },
            "source_files": {"reviewed_raw_actions.jsonl": {
                "path": str(data_file.resolve()), "sha256": exporter.sha256(data_file)}},
        }
        data_receipt["receipt_content_sha256"] = exporter.json_digest(data_receipt)
        budget = {"max_attempts": 1, "learning_rate": 0.01}
        fit_report = {
            "schema": exporter.FRONT_AUX_REPORT_SCHEMA,
            "budget": deepcopy(budget),
            "accepted_auxiliary_updates": 1,
            "attempted_auxiliary_optimizer_steps": 1,
            "steps": [{"accepted": True}],
            "optimized_parameters": deepcopy(exporter.FRONT_AUX_PARAMETERS),
            "optimized_scalar_count": 512,
            "P01_P02_mean_and_sigma_may_both_change": True,
            "kernel_or_sigma_rule_changed": False,
            "MDP_or_control_changed": False,
            "PPO_decisions_added": 0,
            "PPO_updates_added": 0,
            "PPO_optimizer_steps_added": 0,
            "teacher_deployed": False,
            "physical_success_claimed": False,
            "fresh_PPO_rollout_required": True,
            "same_input_only_not_same_future_state_trajectory": True,
            "actual_invariance_phase_ids": [3, 4, 5, 6],
            "synthetic_math_probe_phases": list(range(3, 14)),
            "actor_parameter_sha256_before": source_metadata["actor_parameter_sha256"],
            "actor_parameter_sha256_after": "5" * 64,
            "PPO_Adam_preserved_sha256": source_metadata["optimizer_state_sha256"],
            "PPO_LR_preserved": source_metadata["optimizer_learning_rate"],
        }
        helper_hashes = {
            name: exporter.sha256(exporter.OUTPUT_ROOT / "front_rehearsal_v1" / name)
            for name in exporter.FRONT_AUX_HELPERS
        }
        event = {
            "event_index": 1,
            "kind": exporter.FRONT_AUX_KIND,
            "source_checkpoint": {
                "path": str(source_checkpoint.resolve()),
                "sha256": source_hash,
                "manifest_sha256": exporter.sha256(source_manifest),
            },
            "helper_sha256": helper_hashes,
            "data_receipt": data_receipt,
            "fit_report": fit_report,
            "fit_report_sha256": exporter.json_digest(fit_report),
            "budget": deepcopy(budget),
            "optimized_parameters": deepcopy(exporter.FRONT_AUX_PARAMETERS),
            "target_semantics": exporter.FRONT_AUX_TARGET,
            "phase_scope": ["P01", "P02"],
            "mean_and_log_sigma_may_change": True,
            "physical_success_claimed": False,
            "teacher_deployed": False,
            "PPO_counters_unchanged": {
                key: source_metadata[key] for key in exporter.COUNTERS},
            "stage_requested_decisions_unchanged": source_metadata[
                "stage_requested_decisions"],
            "PPO_decisions_added": 0,
            "PPO_updates_added": 0,
            "PPO_optimizer_steps_added": 0,
        }
        target = deepcopy(source_metadata)
        target["actor_parameter_sha256"] = fit_report["actor_parameter_sha256_after"]
        target["rr_postcross_workspace_branch"][exporter.FRONT_AUX_KEY] = {
            "schema": exporter.FRONT_AUX_SCHEMA,
            "events": [event],
            "accepted_auxiliary_updates_total": 1,
            "attempted_auxiliary_optimizer_steps_total": 1,
            "training_lineage_label": exporter.FRONT_AUX_LINEAGE_LABEL,
        }
        return target

    def front_rehearsal_det_p02_metadata(self, directory: Path) -> dict:
        source_checkpoint = (exporter.OUTPUT_ROOT / "checkpoints" / "history" /
                             "checkpoint_step_000211968.pt").resolve()
        source_manifest = source_checkpoint.with_name(
            source_checkpoint.stem + "_manifest.json")
        source_metadata = exporter.read_json(source_manifest)
        source_counters = {key: source_metadata[key] for key in exporter.COUNTERS}
        source = {
            "path": str(source_checkpoint),
            "sha256": exporter.sha256(source_checkpoint),
            "manifest_sha256": exporter.sha256(source_manifest),
            "PPO_counters": deepcopy(source_counters),
            "runtime_contract_sha256": exporter.json_digest(
                source_metadata["runtime_contract"]),
        }
        inspection_path = (exporter.OUTPUT_ROOT / "front_rehearsal_det_v2" /
                           "CP211968_readonly_candidate_v2.json").resolve()
        inspection = exporter.read_json(inspection_path)
        data_receipt = deepcopy(inspection["reconstruction_receipt"])
        helper_hashes = {
            name: exporter.sha256(exporter._front_aux_det_p02_helper_path(name))
            for name in exporter.FRONT_AUX_DET_P02_HELPERS
        }
        admission_path = (exporter.OUTPUT_ROOT /
                          "det_P02_current_semantics_admission.json").resolve()
        admission = exporter.read_json(admission_path)
        audit_path = (exporter.OUTPUT_ROOT /
                      "audit_det_P02_current_semantics.py").resolve()
        inspection_sha = exporter.sha256(inspection_path)
        admission_record = {
            "path": str(admission_path),
            "sha256": exporter.sha256(admission_path),
            "schema": exporter.FRONT_AUX_DET_P02_ADMISSION_SCHEMA,
            "result": exporter.FRONT_AUX_DET_P02_ADMISSION_RESULT,
            "rows_checked": 254,
            "current_checkpoint_sha256": source["sha256"],
            "candidate_manifest_sha256": data_receipt["candidate_manifest_sha256"],
            "candidate_npz_sha256": admission["candidate_npz_sha256"],
            "audit_script": {"path": str(audit_path), "sha256": exporter.sha256(audit_path)},
            "inspection_receipt": {"path": str(inspection_path), "sha256": inspection_sha},
            "restrictions": deepcopy(admission["restrictions"]),
            "admission_itself_authorizes_optimization": False,
        }
        binding = {
            "source_checkpoint": deepcopy(source),
            "helpers": helper_hashes,
            "inspection": {"path": str(inspection_path), "sha256": inspection_sha,
                           "schema": exporter.FRONT_AUX_DET_P02_INSPECTION_SCHEMA},
            "data_receipt_sha256": exporter.json_digest(data_receipt),
            "data_manifest_sha256": data_receipt["candidate_manifest_sha256"],
            "admission": admission_record,
            "expected_head": source_metadata["runtime_contract"]["source_git_commit"],
        }
        budget = {"learning_rate": 0.01, "max_attempts": 1}
        envelope = {"schema": "wlr50_clean.explicit_det_P02_aux_budget.v2",
                    "binding": deepcopy(binding), "budget": deepcopy(budget)}
        budget_path = directory / "synthetic_det_P02_budget.json"
        budget_path.write_text(json.dumps(envelope), encoding="utf-8")
        budget_receipt = {"path": str(budget_path.resolve()),
                          "sha256": exporter.sha256(budget_path), "envelope": envelope}
        actor_after = "6" * 64
        frozen = {
            "schema": exporter.FRONT_AUX_REPORT_SCHEMA,
            "budget": deepcopy(budget),
            "steps": [{"accepted": True}],
            "accepted_auxiliary_updates": 1,
            "attempted_auxiliary_optimizer_steps": 1,
            "optimized_parameters": deepcopy(exporter.FRONT_AUX_PARAMETERS),
            "optimized_scalar_count": 512,
            "PPO_Adam_preserved_sha256": source_metadata["optimizer_state_sha256"],
            "PPO_LR_preserved": source_metadata["optimizer_learning_rate"],
            "actor_parameter_sha256_before": source_metadata["actor_parameter_sha256"],
            "actor_parameter_sha256_after": actor_after,
        }
        fit_report = {
            "schema": exporter.FRONT_AUX_DET_P02_SCHEMA,
            "accepted_auxiliary_updates": 1,
            "attempted_auxiliary_optimizer_steps": 1,
            "optimized_parameters": deepcopy(exporter.FRONT_AUX_DET_P02_PARAMETERS),
            "optimized_scalar_count": 256,
            "actually_changed_scalar_count": 128,
            "P01_initial_gradient_exact_zero": True,
            "protected_entire_Gaussian_bitwise_equal": {
                "real_P01": True, "real_P03plus": True,
                "synthetic_P01_P03_P13": True},
            "P02_mean_and_log_sigma_may_both_change": True,
            "source_observations_directly_saved389": False,
            "target_semantics": exporter.FRONT_AUX_DET_P02_TARGET,
            "same_input_invariance_is_not_future_trajectory_invariance": True,
            "nested_kernel_scope_is_capability_not_this_execution_scope": True,
            "MDP_or_control_changed": False,
            "kernel_or_sigma_rule_changed": False,
            "training_rng_preserved": True,
            "fresh_PPO_rollout_required": True,
            "teacher_deployed": False,
            "physical_success_claimed": False,
            "PPO_decisions_added": 0,
            "PPO_updates_added": 0,
            "PPO_optimizer_steps_added": 0,
            "actor_parameter_sha256_before": source_metadata["actor_parameter_sha256"],
            "actor_parameter_sha256_after": actor_after,
            "frozen_kernel_fit_report": frozen,
        }
        event = {
            "event_index": 2,
            "kind": exporter.FRONT_AUX_DET_P02_KIND,
            "source_checkpoint": deepcopy(source),
            "helper_sha256": helper_hashes,
            "binding": binding,
            "data_receipt": data_receipt,
            "fit_report": fit_report,
            "fit_report_sha256": exporter.json_digest(fit_report),
            "budget": budget,
            "budget_receipt": budget_receipt,
            "optimized_parameters": deepcopy(exporter.FRONT_AUX_DET_P02_PARAMETERS),
            "phase_scope": ["P02"],
            "mean_and_log_sigma_may_change": True,
            "source_observations_directly_saved389": False,
            "target_semantics": exporter.FRONT_AUX_DET_P02_TARGET,
            "same_input_P01_P03plus_preserved": True,
            "physical_success_claimed": False,
            "teacher_deployed": False,
            "PPO_counters_unchanged": deepcopy(source_counters),
            "stage_requested_decisions_unchanged": deepcopy(
                source_metadata["stage_requested_decisions"]),
            "PPO_decisions_added": 0,
            "PPO_updates_added": 0,
            "PPO_optimizer_steps_added": 0,
        }
        target = deepcopy(source_metadata)
        ledger = target["rr_postcross_workspace_branch"][exporter.FRONT_AUX_KEY]
        ledger["events"].append(event)
        ledger["accepted_auxiliary_updates_total"] += 1
        ledger["attempted_auxiliary_optimizer_steps_total"] += 1
        target["actor_parameter_sha256"] = actor_after
        return target

    def front_mean_v3_actual_metadata(self) -> dict:
        manifest = (exporter.OUTPUT_ROOT / "checkpoints" / "history" /
                    "checkpoint_aux_meanfront_step_000213376_v3_manifest.json")
        return exporter.read_json(manifest)

    def rr_mean_v4_actual_metadata(self) -> dict:
        manifest = (exporter.OUTPUT_ROOT / "checkpoints" / "history" /
                    "checkpoint_aux_meanrr_step_000214400_v4_manifest.json")
        return exporter.read_json(manifest)

    def rr_receiver_v2_actual_metadata(self) -> dict:
        manifest = (exporter.OUTPUT_ROOT / "checkpoints" / "history" /
                    "checkpoint_rr_receiver_v2_step_000214400_manifest.json")
        return exporter.read_json(manifest)

    def p05_preedge_fixture(self) -> tuple[dict, dict]:
        source_manifest = (exporter.OUTPUT_ROOT / "checkpoints" / "history" /
                           "checkpoint_step_000216448_manifest.json")
        source = exporter.read_json(source_manifest)
        source_checkpoint = Path(source["checkpoint_path"]).resolve(strict=True)
        expected_preserved = set(exporter.RR_WORKSPACE_REQUIRED_HISTORY) | {
            key for key in source
            if ((key != "resume_migration" and key.endswith(
                    ("_branch", "_branch_counts", "_migration")))
                or key in ("source_stage_requested_decisions",
                           "training_quantity_budget_extension"))
        }
        reviewed = {
            path: f"{index + 1:x}" * 64
            for index, path in enumerate(sorted(
                exporter.P05_PREEDGE_REQUIRED_CHANGED_FILES))
        }
        factor = {
            "schema": exporter.P05_PREEDGE_SCHEMA,
            "review_reason": "synthetic reviewed P05-only control revision",
            "source_semantics": exporter.P05_PREEDGE_SOURCE_SEMANTICS,
            "target_semantics": exporter.P05_PREEDGE_SEMANTICS,
            "reviewed_code_sha256": reviewed,
            "task_spec_change": {"nominal": {
                exporter.P05_PREEDGE_MODE_KEY: exporter.P05_PREEDGE_SEMANTICS}},
            "source_task_spec_sha256": "4" * 64,
            "target_task_spec_sha256": "5" * 64,
            "observation_contract": {
                "observation_layout": "role372_p05_capture_assist_v1",
                "observation_dimension": 389,
                "action_dimension": 12,
                "num_envs": 1,
                "parameter_mapping": "identity_all_parameters_and_buffers",
            },
            "observation_shape_changed": False,
            "observation_codec_changed": False,
            "observation_semantics_changed": [],
            "same_numeric_input_policy_mapping_preserved": True,
            "same_physical_state_action_equivalence_claimed": False,
            "controller_transition_semantics_changed": True,
            "nominal_changed": True,
            "affected_control_phases": ["P05"],
            "same_mdp_claimed": False,
            "reward_changed": False,
            "physical_dynamics_changed": False,
            "policy_kernel_changed": False,
            "capture_assist_changed": False,
            "caps_changed": False,
            "sigma_changed": False,
            "physical_task_acceptance_rules_changed": False,
            "added_mutable_state": False,
            "preserved_metadata_sha256": {
                key: exporter.json_digest(source[key]) for key in expected_preserved},
            "source_resume_migration": deepcopy(source.get("resume_migration")),
            "source_resume_migration_sha256": exporter.json_digest(
                source.get("resume_migration")),
            "counter_origin": {key: source[key] for key in exporter.COUNTERS},
            "source_effective_learning_rate": source["optimizer_learning_rate"],
            "target_effective_learning_rate": source["optimizer_learning_rate"],
            "parameter_mapping": "identity_all_parameters_and_buffers",
            "optimizer_mapping": "identity_all_Adam_moments_steps_groups_and_effective_LR",
            "normalizer_mapping": "identity_Identity",
            "rng_mapping": "restore_exact_source_training_rng",
            "discard_old_rollout_storage": True,
            "physical_state_inherited": False,
            "added_policy_decisions": 0,
            "added_ppo_updates": 0,
            "added_optimizer_steps": 0,
            "added_auxiliary_updates": 0,
        }
        migration = {
            "schema": exporter.P05_PREEDGE_SCHEMA,
            "reason": factor["review_reason"],
            "source_checkpoint": str(source_checkpoint),
            "source_checkpoint_sha256": source["checkpoint_sha256"],
            "source_manifest_sha256": exporter.sha256(source_manifest),
            "source_contract_sha256": "6" * 64,
            "target_contract_sha256": "7" * 64,
            "source_git_commit": "1" * 40,
            "target_git_commit": "2" * 40,
            "source_runtime_content_sha256": "8" * 64,
            "target_runtime_content_sha256": "9" * 64,
            "allowed_changed_files": sorted(reviewed),
            "changed_file_hashes": {
                path: {"before": "0" * 64, "after": digest}
                for path, digest in reviewed.items()},
            "observation_dimension": 389,
            "action_dimension": 12,
            "preserve_actor_critic_optimizer_normalizer_rng_and_budget": True,
            "discard_old_rollout_storage": True,
            "physics_resume": "fresh_legal_P01_reset",
            exporter.P05_PREEDGE_FACTOR_KEY: factor,
        }
        target = deepcopy(source)
        target[exporter.P05_PREEDGE_BRANCH] = {
            "schema": exporter.P05_PREEDGE_SCHEMA,
            "semantics": exporter.P05_PREEDGE_SEMANTICS,
            "counter_origin": deepcopy(factor["counter_origin"]),
            "source_checkpoint_sha256": source["checkpoint_sha256"],
            "migration_added_updates": 0,
        }
        target[exporter.P05_PREEDGE_MIGRATION] = migration
        target[exporter.P05_PREEDGE_COUNTS] = {
            key: 0 for key in exporter.COUNTERS}
        return source, target

    def test_summary_uses_top_level_state_and_real_pending_flag(self) -> None:
        summary = exporter.frame_summary(tick_row(8, phase="P05", pending=False))
        self.assertEqual(summary["FL_hip_assist_target_deg"], -12.5)
        self.assertFalse(summary["fl_capture_pending"])
        self.assertEqual(summary["FL_contact"], "AIR")
        self.assertEqual(summary["FL_gap_mm"], 7.5)
        self.assertEqual(summary["wheel_qd_canonical_rad_s"], [-0.1, 0.2, -0.3, 0.4])

    def test_overlay_has_truthful_method_and_no_pending_phase_inference(self) -> None:
        summary = exporter.frame_summary(tick_row(8, phase="P05", pending=False))
        lines = exporter.panel_lines(summary, checkpoint=199680, p05_ppo_updates=0,
                                     result="DIAGNOSTIC_FAILURE")
        self.assertIn("PPO + CAPTURE ASSIST + INHERITED LIMITED AUX", lines[0])
        self.assertIn("DETERMINISTIC CP199680", lines[0])
        self.assertIn("MIGRATED WARM START | +0 P05 PPO UPDATES", lines[1])
        self.assertIn("FL_CAPTURE_PENDING=False", lines[1])
        self.assertNotIn("FL_CAPTURE_PENDING=True", "\n".join(lines))
        panel = next(iter(exporter.rgba_panels([summary], checkpoint=199680,
                                               p05_ppo_updates=0,
                                               result="DIAGNOSTIC_FAILURE")))
        self.assertEqual(len(panel), 1280 * 150 * 4)

    def test_wait_state_does_not_claim_zero_degree_assist_targets(self) -> None:
        summary = exporter.frame_summary(tick_row(8, phase="P05", pending=True,
                                                  mode=0, mode_name="WAIT"))
        self.assertIsNone(summary["FL_hip_assist_target_deg"])
        self.assertIsNone(summary["FL_knee_hold_target_deg"])
        lines = exporter.panel_lines(summary, checkpoint=199680, p05_ppo_updates=0,
                                     result="DIAGNOSTIC_FAILURE")
        self.assertIn("assist WAIT active=False", lines[2])
        self.assertIn("hip assist/final/actual deg N/A/", lines[3])
        self.assertIn("knee HOLD/final/actual N/A/", lines[3])

    def test_feedback_revision_has_an_independent_zero_update_label(self) -> None:
        identity = exporter.feedback_training_identity(self.feedback_metadata())
        self.assertTrue(identity["feedback_branch_present"])
        self.assertEqual(identity["feedback_counter_origin"]["ppo_updates"], 1557)
        self.assertEqual(identity["feedback_added_ppo_updates"], 0)
        summary = exporter.frame_summary(tick_row(8))
        lines = exporter.panel_lines(summary, checkpoint=203776, p05_ppo_updates=32,
            result="DIAGNOSTIC_FAILURE", feedback_revision=identity["feedback_revision"],
            feedback_ppo_updates=identity["feedback_added_ppo_updates"])
        self.assertIn("P05-origin PPO +32", lines[1])
        self.assertIn("feedback-origin PPO +0", lines[1])
        self.assertIn(exporter.FEEDBACK_REVISION, lines[1])
        panel = next(iter(exporter.rgba_panels([summary], checkpoint=203776,
            p05_ppo_updates=32, result="DIAGNOSTIC_FAILURE",
            feedback_revision=identity["feedback_revision"], feedback_ppo_updates=0)))
        self.assertEqual(len(panel), 1280 * 150 * 4)

    def test_feedback_branch_rejects_missing_negative_and_tampered_origins(self) -> None:
        cases = []
        missing = self.feedback_metadata()
        missing["capture_feedback_semantics_branch"].pop("counter_origin")
        cases.append(missing)
        negative = self.feedback_metadata()
        negative["capture_feedback_semantics_branch"]["counter_origin"]["ppo_updates"] = -1
        cases.append(negative)
        tampered = self.feedback_metadata()
        tampered["capture_feedback_semantics_branch"]["counter_origin"]["ppo_updates"] += 1
        cases.append(tampered)
        missing_counts = self.feedback_metadata()
        missing_counts.pop("capture_feedback_semantics_branch_counts")
        cases.append(missing_counts)
        wrong_revision = self.feedback_metadata()
        wrong_revision["capture_feedback_semantics_branch"]["feedback_revision"] = "tampered_v3"
        cases.append(wrong_revision)
        for metadata in cases:
            with self.subTest(metadata=metadata):
                with self.assertRaises(RuntimeError):
                    exporter.feedback_training_identity(metadata)

    def test_orphaned_feedback_migration_cannot_look_like_an_old_checkpoint(self) -> None:
        metadata = self.feedback_metadata()
        metadata.pop("capture_feedback_semantics_branch")
        with self.assertRaises(RuntimeError):
            exporter.feedback_training_identity(metadata)

    def test_rr_workspace_revision_has_independent_reward_update_label(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            identity = exporter.rr_workspace_training_identity(
                self.rr_workspace_metadata(Path(directory)))
        self.assertTrue(identity["rr_workspace_branch_present"])
        self.assertEqual(identity["rr_workspace_added_ppo_updates"], 0)
        summary = exporter.frame_summary(tick_row(8))
        lines = exporter.panel_lines(
            summary, checkpoint=1, p05_ppo_updates=0,
            feedback_revision=exporter.FEEDBACK_REVISION, feedback_ppo_updates=0,
            rr_workspace_semantics=identity["rr_workspace_semantics"],
            rr_workspace_ppo_updates=identity["rr_workspace_added_ppo_updates"],
            result="DIAGNOSTIC_FAILURE")
        self.assertEqual(len(lines), 7)
        self.assertIn(exporter.RR_WORKSPACE_SEMANTICS, lines[2])
        self.assertIn("RR-origin PPO +0", lines[2])
        panel = next(iter(exporter.rgba_panels(
            [summary], checkpoint=1, p05_ppo_updates=0,
            feedback_revision=exporter.FEEDBACK_REVISION, feedback_ppo_updates=0,
            rr_workspace_semantics=identity["rr_workspace_semantics"],
            rr_workspace_ppo_updates=0, result="DIAGNOSTIC_FAILURE")))
        self.assertEqual(len(panel), 1280 * 150 * 4)

    def test_rr_workspace_rejects_orphan_tamper_and_count_mismatch(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            base = self.rr_workspace_metadata(Path(directory))
            cases = []
            orphan = deepcopy(base)
            orphan.pop("rr_postcross_workspace_branch")
            cases.append(orphan)
            wrong_mode = deepcopy(base)
            wrong_mode["rr_postcross_workspace_branch"]["semantics"] = "wrong"
            cases.append(wrong_mode)
            wrong_source = deepcopy(base)
            wrong_source["rr_postcross_workspace_migration"]["source_checkpoint_sha256"] = "c" * 64
            cases.append(wrong_source)
            wrong_history = deepcopy(base)
            wrong_history["rr_postcross_workspace_migration"]["rr_postcross_workspace_factor"][
                "source_semantics"] = "wrong"
            cases.append(wrong_history)
            wrong_index = deepcopy(base)
            wrong_index["rr_postcross_workspace_migration"]["rr_postcross_workspace_factor"][
                "observation_semantics_changed"][0]["index"] = 18
            cases.append(wrong_index)
            false_same_mdp = deepcopy(base)
            false_same_mdp["rr_postcross_workspace_migration"]["rr_postcross_workspace_factor"][
                "same_mdp_claimed"] = True
            cases.append(false_same_mdp)
            missing_counts = deepcopy(base)
            missing_counts.pop("rr_postcross_workspace_branch_counts")
            cases.append(missing_counts)
            negative_delta = deepcopy(base)
            negative_delta["rr_postcross_workspace_branch"]["counter_origin"]["ppo_updates"] += 1
            cases.append(negative_delta)
            for metadata in cases:
                with self.subTest(metadata=metadata):
                    with self.assertRaises(RuntimeError):
                        exporter.rr_workspace_training_identity(metadata)

    def test_rr_workspace_rejects_changed_historical_source_sidecar(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            metadata = self.rr_workspace_metadata(Path(directory))
            migration = metadata["rr_postcross_workspace_migration"]
            source = Path(migration["source_checkpoint"])
            source_manifest = source.with_name(source.stem + "_manifest.json")
            source_manifest.write_text("{}", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                exporter.rr_workspace_training_identity(metadata)

    def test_old_checkpoint_has_no_rr_workspace_lineage(self) -> None:
        identity = exporter.rr_workspace_training_identity(self.feedback_metadata())
        self.assertFalse(identity["rr_workspace_branch_present"])
        self.assertIsNone(identity["rr_workspace_semantics"])

    def test_front_rehearsal_aux_has_derived_identity_and_distinct_paths(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory(dir=exporter.OUTPUT_ROOT) as directory:
            identity = exporter.front_rehearsal_training_identity(
                self.front_rehearsal_metadata(Path(directory)))
            self.assertEqual(identity["front_rehearsal_display_id"], "CP203776_AUXFR1")
            self.assertEqual(identity["front_rehearsal_accepted_auxiliary_updates_total"], 1)
            self.assertEqual(identity["front_rehearsal_attempted_auxiliary_optimizer_steps_total"], 1)
            good = exporter.OUTPUT_ROOT / "CP203776_AUXFR1_deterministic_review"
            exporter.require_aux_destination(
                good, identity["front_rehearsal_display_id"],
                current_decisions=203776, latest_source_decisions=203776)
            with self.assertRaises(RuntimeError):
                exporter.require_aux_destination(
                    exporter.OUTPUT_ROOT / "CP203776_deterministic_review",
                    identity["front_rehearsal_display_id"],
                    current_decisions=203776, latest_source_decisions=203776)
            exporter.require_aux_destination(
                exporter.OUTPUT_ROOT / "CP205824_deterministic_front_rehearsal_v1_review",
                "CP205824_AUXFR1", current_decisions=205824,
                latest_source_decisions=203776)
            with self.assertRaises(RuntimeError):
                exporter.require_aux_destination(
                    exporter.OUTPUT_ROOT / "CP205824_AUXFR1_deterministic_review",
                    "CP205824_AUXFR1", current_decisions=205824,
                    latest_source_decisions=203776)
        names = exporter.output_names(203776, False, display_id="CP203776_AUXFR1")
        self.assertTrue(names["full"].startswith("CP203776_AUXFR1_"))
        self.assertEqual(names["pair"], "N_vs_CP203776_AUXFR1_DET_same_camera.mp4")

    def test_front_rehearsal_aux_overlay_and_receipt_are_explicitly_non_ppo(self) -> None:
        summary = exporter.frame_summary(tick_row(8))
        lines = exporter.panel_lines(
            summary, checkpoint=203776, p05_ppo_updates=32,
            feedback_revision=exporter.FEEDBACK_REVISION, feedback_ppo_updates=0,
            rr_workspace_semantics=exporter.RR_WORKSPACE_SEMANTICS,
            rr_workspace_ppo_updates=0, display_id="CP203776_AUXFR1",
            front_aux_event_index=1, front_aux_accepted_total=1,
            front_aux_attempted_total=1, result="DIAGNOSTIC_FAILURE")
        self.assertEqual(len(lines), 8)
        self.assertIn("P01/P02 phase-column mu+log-sigma may change", lines[3])
        self.assertIn("NOT PPO / NOT PHYSICAL SUCCESS", lines[3])
        checkpoint = {
            "front_rehearsal_auxiliary_present": True,
            "front_rehearsal_display_id": "CP203776_AUXFR1",
            "front_rehearsal_event_index": 1,
            "front_rehearsal_accepted_auxiliary_updates_total": 1,
            "front_rehearsal_attempted_auxiliary_optimizer_steps_total": 1,
            "front_rehearsal_inherited_limited_auxiliary_accepted_total": 7,
            "front_rehearsal_inherited_limited_auxiliary_attempted_total": 8,
            "front_rehearsal_training_lineage_label": exporter.FRONT_AUX_LINEAGE_LABEL,
            "front_rehearsal_latest_source_checkpoint": {"sha256": "a" * 64},
            "front_rehearsal_latest_source_PPO_counters": {
                "global_policy_decisions": 203776,
                "ppo_updates": 1557,
                "optimizer_steps": 31140,
            },
            "front_rehearsal_event_summaries": [{
                "event_index": 1,
                "accepted_auxiliary_updates": 1,
                "attempted_auxiliary_optimizer_steps": 1,
            }],
            "front_rehearsal_latest_fit_report_sha256": "b" * 64,
        }
        receipt = exporter.front_aux_receipt_lineage(checkpoint)
        self.assertEqual(receipt["PPO_credit_added"], 0)
        self.assertFalse(receipt["physical_success_claimed"])
        self.assertTrue(receipt["mean_and_log_sigma_may_change"])

    def test_front_rehearsal_aux_rejects_tampered_lineage(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory(dir=exporter.OUTPUT_ROOT) as directory:
            base = self.front_rehearsal_metadata(Path(directory))
            mutations = []
            wrong_index = deepcopy(base)
            wrong_index["rr_postcross_workspace_branch"][exporter.FRONT_AUX_KEY][
                "events"][0]["event_index"] = 2
            mutations.append(wrong_index)
            wrong_helper = deepcopy(base)
            wrong_helper["rr_postcross_workspace_branch"][exporter.FRONT_AUX_KEY][
                "events"][0]["helper_sha256"]["front_rehearsal.py"] = "0" * 64
            mutations.append(wrong_helper)
            wrong_data = deepcopy(base)
            wrong_data["rr_postcross_workspace_branch"][exporter.FRONT_AUX_KEY][
                "events"][0]["data_receipt"]["receipt_content_sha256"] = "0" * 64
            mutations.append(wrong_data)
            wrong_report = deepcopy(base)
            wrong_report["rr_postcross_workspace_branch"][exporter.FRONT_AUX_KEY][
                "events"][0]["fit_report_sha256"] = "0" * 64
            mutations.append(wrong_report)
            wrong_total = deepcopy(base)
            wrong_total["rr_postcross_workspace_branch"][exporter.FRONT_AUX_KEY][
                "accepted_auxiliary_updates_total"] = 2
            mutations.append(wrong_total)
            wrong_scope = deepcopy(base)
            wrong_scope["rr_postcross_workspace_branch"][exporter.FRONT_AUX_KEY][
                "events"][0]["mean_and_log_sigma_may_change"] = False
            mutations.append(wrong_scope)
            wrong_success = deepcopy(base)
            wrong_success["rr_postcross_workspace_branch"][exporter.FRONT_AUX_KEY][
                "events"][0]["physical_success_claimed"] = True
            mutations.append(wrong_success)
            wrong_old_aux = deepcopy(base)
            wrong_old_aux["task_conditioned_hip_wheel_branch"] = {"tampered": True}
            mutations.append(wrong_old_aux)
            for metadata in mutations:
                with self.subTest(metadata=metadata):
                    with self.assertRaises(RuntimeError):
                        exporter.front_rehearsal_training_identity(metadata)

    def test_det_p02_event2_is_separate_and_derived_from_the_ledger(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory(dir=exporter.OUTPUT_ROOT) as directory:
            identity = exporter.front_rehearsal_training_identity(
                self.front_rehearsal_det_p02_metadata(Path(directory)))
        self.assertEqual(identity["front_rehearsal_event_index"], 2)
        self.assertEqual(identity["front_rehearsal_display_id"], "CP211968_AUXDET2")
        self.assertEqual(identity["front_rehearsal_accepted_auxiliary_updates_total"], 33)
        self.assertEqual(identity["front_rehearsal_attempted_auxiliary_optimizer_steps_total"], 33)
        events = identity["front_rehearsal_event_summaries"]
        self.assertEqual([(event["event_index"], event["accepted_auxiliary_updates"])
                          for event in events], [(1, 32), (2, 1)])
        self.assertTrue(events[0]["source_observations_directly_saved389"])
        self.assertFalse(events[1]["source_observations_directly_saved389"])
        self.assertEqual(events[1]["phase_scope"], ["P02"])
        self.assertEqual(events[1]["optimized_parameters"],
                         exporter.FRONT_AUX_DET_P02_PARAMETERS)
        exporter.require_aux_destination(
            exporter.OUTPUT_ROOT /
                "CP211968_deterministic_front_rehearsal_det_v2_review",
            identity["front_rehearsal_display_id"], current_decisions=211968,
            latest_source_decisions=211968)

        summary = exporter.frame_summary(tick_row(8))
        lines = exporter.panel_lines(
            summary, checkpoint=211968, p05_ppo_updates=96,
            feedback_revision=exporter.FEEDBACK_REVISION, feedback_ppo_updates=64,
            rr_workspace_semantics=exporter.RR_WORKSPACE_SEMANTICS,
            rr_workspace_ppo_updates=32, display_id="CP211968_AUXDET2",
            front_aux_event_index=2, front_aux_accepted_total=33,
            front_aux_attempted_total=33, front_aux_event_summaries=events,
            result="DIAGNOSTIC_FAILURE")
        self.assertEqual(len(lines), 9)
        self.assertIn("#1 v1 P01/P02 accepted/attempted 32/32", lines[3])
        self.assertIn("#2 det-P02 accepted/attempted 1/1", lines[4])
        self.assertIn("reconstructed389 W[:,1]", lines[4])
        self.assertIn("NOT PPO / NOT SUCCESS", lines[4])

    def test_det_p02_event2_rejects_provenance_and_claim_tampering(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory(dir=exporter.OUTPUT_ROOT) as directory:
            base = self.front_rehearsal_det_p02_metadata(Path(directory))
            ledger = base["rr_postcross_workspace_branch"][exporter.FRONT_AUX_KEY]
            mutations = []
            wrong_kind = deepcopy(base)
            wrong_kind["rr_postcross_workspace_branch"][exporter.FRONT_AUX_KEY][
                "events"][1]["kind"] = "not_the_reviewed_det_P02_kind"
            mutations.append(wrong_kind)
            direct389 = deepcopy(base)
            direct389["rr_postcross_workspace_branch"][exporter.FRONT_AUX_KEY][
                "events"][1]["source_observations_directly_saved389"] = True
            mutations.append(direct389)
            false_success = deepcopy(base)
            false_success["rr_postcross_workspace_branch"][exporter.FRONT_AUX_KEY][
                "events"][1]["physical_success_claimed"] = True
            mutations.append(false_success)
            wrong_scope = deepcopy(base)
            wrong_scope["rr_postcross_workspace_branch"][exporter.FRONT_AUX_KEY][
                "events"][1]["optimized_parameters"] = deepcopy(exporter.FRONT_AUX_PARAMETERS)
            mutations.append(wrong_scope)
            wrong_helper = deepcopy(base)
            wrong_helper["rr_postcross_workspace_branch"][exporter.FRONT_AUX_KEY][
                "events"][1]["helper_sha256"]["execute_det_rehearsal.py"] = "0" * 64
            mutations.append(wrong_helper)
            wrong_admission = deepcopy(base)
            wrong_admission["rr_postcross_workspace_branch"][exporter.FRONT_AUX_KEY][
                "events"][1]["binding"]["admission"][
                    "admission_itself_authorizes_optimization"] = True
            mutations.append(wrong_admission)
            wrong_inspection = deepcopy(base)
            wrong_inspection["rr_postcross_workspace_branch"][exporter.FRONT_AUX_KEY][
                "events"][1]["binding"]["inspection"]["sha256"] = "0" * 64
            mutations.append(wrong_inspection)
            wrong_nested_scope = deepcopy(base)
            wrong_nested_scope["rr_postcross_workspace_branch"][exporter.FRONT_AUX_KEY][
                "events"][1]["fit_report"][
                    "nested_kernel_scope_is_capability_not_this_execution_scope"] = False
            mutations.append(wrong_nested_scope)
            changed_event1 = deepcopy(base)
            changed_event1["rr_postcross_workspace_branch"][exporter.FRONT_AUX_KEY][
                "events"][0]["phase_scope"] = ["P02"]
            mutations.append(changed_event1)
            wrong_total = deepcopy(base)
            wrong_total["rr_postcross_workspace_branch"][exporter.FRONT_AUX_KEY][
                "accepted_auxiliary_updates_total"] += 1
            mutations.append(wrong_total)
            self.assertEqual(ledger["accepted_auxiliary_updates_total"], 33)
            for metadata in mutations:
                with self.subTest(metadata=metadata):
                    with self.assertRaises(RuntimeError):
                        exporter.front_rehearsal_training_identity(metadata)

    def test_mean_v3_event3_is_mean_only_and_identity_is_ledger_derived(self) -> None:
        identity = exporter.front_rehearsal_training_identity(
            self.front_mean_v3_actual_metadata())
        self.assertEqual(identity["front_rehearsal_event_index"], 3)
        self.assertEqual(identity["front_rehearsal_display_id"],
                         "CP213376_AUXMEAN3")
        self.assertEqual(identity["front_rehearsal_accepted_auxiliary_updates_total"], 96)
        self.assertEqual(identity["front_rehearsal_attempted_auxiliary_optimizer_steps_total"], 96)
        latest = identity["front_rehearsal_event_summaries"][-1]
        self.assertEqual(latest["kind"], exporter.FRONT_AUX_MEAN_V3_KIND)
        self.assertEqual(latest["optimized_parameters"],
                         exporter.FRONT_AUX_MEAN_V3_PARAMETERS)
        self.assertFalse(latest["mean_and_log_sigma_may_change"])
        self.assertTrue(latest["mean_may_change_all_phases"])
        self.assertTrue(latest["same_input_sigma_fixed"])
        self.assertEqual(latest["actual_protection_phases"], list(range(3, 13)))
        self.assertEqual(latest["missing_actual_protection_phases"], [13])
        self.assertEqual(latest["P01_independent_validation_rows"], 0)
        exporter.require_aux_destination(
            exporter.OUTPUT_ROOT /
                "CP213376_deterministic_front_mean_rehearsal_v3_review",
            identity["front_rehearsal_display_id"], current_decisions=213376,
            latest_source_decisions=213376)
        names = exporter.output_names(
            213376, False, display_id=identity["front_rehearsal_display_id"])
        self.assertTrue(names["full"].startswith("CP213376_AUXMEAN3_"))

        lines = exporter.panel_lines(
            exporter.frame_summary(tick_row(8)), checkpoint=213376,
            p05_ppo_updates=107, feedback_revision=exporter.FEEDBACK_REVISION,
            feedback_ppo_updates=75,
            rr_workspace_semantics=exporter.RR_WORKSPACE_SEMANTICS,
            rr_workspace_ppo_updates=43,
            display_id=identity["front_rehearsal_display_id"],
            front_aux_event_index=3,
            front_aux_accepted_total=identity[
                "front_rehearsal_accepted_auxiliary_updates_total"],
            front_aux_attempted_total=identity[
                "front_rehearsal_attempted_auxiliary_optimizer_steps_total"],
            front_aux_event_summaries=identity["front_rehearsal_event_summaries"],
            result="DIAGNOSTIC_FAILURE")
        self.assertEqual(len(lines), 10)
        self.assertIn("#3 mean-head 32/32", lines[5])
        self.assertIn("3084 mu params; sigma fixed", lines[5])
        self.assertIn("P01 1/no-val; protect P03-P12, no P13", lines[5])
        receipt = exporter.front_aux_receipt_lineage(identity)
        self.assertEqual(receipt["latest_event"]["kind"],
                         exporter.FRONT_AUX_MEAN_V3_KIND)
        self.assertFalse(receipt["latest_event"]["mean_and_log_sigma_may_change"])
        self.assertTrue(receipt["latest_event"]["same_input_sigma_fixed"])

    def test_mean_v3_event3_rejects_scope_provenance_and_count_tampering(self) -> None:
        base = self.front_mean_v3_actual_metadata()
        mutations = []
        wrong_kind = deepcopy(base)
        wrong_kind["rr_postcross_workspace_branch"][exporter.FRONT_AUX_KEY][
            "events"][2]["kind"] = "not_the_reviewed_mean_v3_kind"
        mutations.append(wrong_kind)
        wrong_sigma = deepcopy(base)
        wrong_sigma["rr_postcross_workspace_branch"][exporter.FRONT_AUX_KEY][
            "events"][2]["same_input_sigma_fixed"] = False
        mutations.append(wrong_sigma)
        wrong_protection = deepcopy(base)
        wrong_protection["rr_postcross_workspace_branch"][exporter.FRONT_AUX_KEY][
            "events"][2]["P13_actual_protection_coverage"] = True
        mutations.append(wrong_protection)
        wrong_helper = deepcopy(base)
        wrong_helper["rr_postcross_workspace_branch"][exporter.FRONT_AUX_KEY][
            "events"][2]["helper_sha256"]["fit_mean_core.py"] = "0" * 64
        mutations.append(wrong_helper)
        false_ppo = deepcopy(base)
        false_ppo["rr_postcross_workspace_branch"][exporter.FRONT_AUX_KEY][
            "events"][2]["PPO_updates_added"] = 1
        mutations.append(false_ppo)
        wrong_total = deepcopy(base)
        wrong_total["rr_postcross_workspace_branch"][exporter.FRONT_AUX_KEY][
            "accepted_auxiliary_updates_total"] += 1
        mutations.append(wrong_total)
        for metadata in mutations:
            with self.subTest(kind=metadata["rr_postcross_workspace_branch"][
                    exporter.FRONT_AUX_KEY]["events"][2].get("kind")):
                with self.assertRaises((RuntimeError, ValueError)):
                    exporter.front_rehearsal_training_identity(metadata)

    def test_actual_cp214400_three_event_regression_is_unchanged(self) -> None:
        metadata = exporter.read_json(
            exporter.OUTPUT_ROOT / "checkpoints" / "history" /
            "checkpoint_step_000214400_manifest.json")
        identity = exporter.front_rehearsal_training_identity(metadata)
        reward = exporter.rr_reward_training_identity(metadata)
        self.assertEqual(identity["front_rehearsal_display_id"],
                         "CP214400_AUXMEAN3")
        self.assertEqual(identity["front_rehearsal_event_index"], 3)
        self.assertEqual(identity["front_rehearsal_accepted_auxiliary_updates_total"], 96)
        self.assertEqual(identity["front_rehearsal_attempted_auxiliary_optimizer_steps_total"], 96)
        self.assertNotIn("rr_rehearsal_auxiliary_present", identity)
        self.assertEqual(reward["rr_workspace_semantics"],
                         exporter.RR_WORKSPACE_SEMANTICS)
        self.assertNotIn("rr_receiver_v2_branch_present", reward)
        self.assertTrue(exporter.output_names(
            214400, False, display_id=identity["front_rehearsal_display_id"]
        )["full"].startswith("CP214400_AUXMEAN3_"))

    def test_actual_event4_derives_auxrr4_and_splits_front_rr_credit(self) -> None:
        identity = exporter.front_rehearsal_training_identity(
            self.rr_mean_v4_actual_metadata())
        self.assertEqual(identity["front_rehearsal_display_id"], "CP214400_AUXRR4")
        self.assertEqual(identity["front_rehearsal_event_index"], 4)
        self.assertEqual(identity["front_rehearsal_accepted_auxiliary_updates_total"], 96)
        self.assertEqual(identity["front_rehearsal_attempted_auxiliary_optimizer_steps_total"], 96)
        self.assertEqual(identity["rr_rehearsal_accepted_auxiliary_updates_total"], 7)
        self.assertEqual(identity["rr_rehearsal_attempted_auxiliary_optimizer_steps_total"], 8)
        self.assertEqual(identity["mixed_rehearsal_accepted_auxiliary_updates_total"], 103)
        self.assertEqual(identity["mixed_rehearsal_attempted_auxiliary_optimizer_steps_total"], 104)
        latest = identity["front_rehearsal_event_summaries"][-1]
        self.assertEqual(latest["kind"], exporter.RR_AUX_MEAN_V4_KIND)
        self.assertEqual(latest["phase_scope"], ["P09", "P10", "P11"])
        self.assertEqual(latest["target_semantics"], exporter.RR_AUX_MEAN_V4_TARGET)
        self.assertTrue(latest["source_observations_directly_saved389"])
        self.assertEqual(latest["P10_independent_validation_rows"], 0)
        exporter.require_aux_destination(
            exporter.OUTPUT_ROOT /
                "CP214400_deterministic_rr_mean_rehearsal_v4_review",
            identity["front_rehearsal_display_id"], current_decisions=214400,
            latest_source_decisions=214400)
        names = exporter.output_names(
            214400, False, display_id=identity["front_rehearsal_display_id"])
        self.assertTrue(names["full"].startswith("CP214400_AUXRR4_"))

        lines = exporter.panel_lines(
            exporter.frame_summary(tick_row(8)), checkpoint=214400,
            p05_ppo_updates=115, feedback_revision=exporter.FEEDBACK_REVISION,
            feedback_ppo_updates=83,
            rr_workspace_semantics=exporter.RR_WORKSPACE_SEMANTICS,
            rr_workspace_ppo_updates=51,
            display_id=identity["front_rehearsal_display_id"],
            front_aux_event_index=4,
            front_aux_accepted_total=identity[
                "front_rehearsal_accepted_auxiliary_updates_total"],
            front_aux_attempted_total=identity[
                "front_rehearsal_attempted_auxiliary_optimizer_steps_total"],
            front_aux_event_summaries=identity["front_rehearsal_event_summaries"],
            result="DIAGNOSTIC_FAILURE")
        self.assertEqual(len(lines), 9)
        self.assertIn("FR-AUX #1-3 accepted/attempted 96/96", lines[3])
        self.assertIn("RR-AUX #4 accepted/attempted 7/8", lines[4])
        self.assertIn("actual stochastic raw12", lines[4])
        panel = next(iter(exporter.rgba_panels(
            [exporter.frame_summary(tick_row(8))], checkpoint=214400,
            p05_ppo_updates=115, feedback_revision=exporter.FEEDBACK_REVISION,
            feedback_ppo_updates=83,
            rr_workspace_semantics=exporter.RR_WORKSPACE_SEMANTICS,
            rr_workspace_ppo_updates=51,
            display_id=identity["front_rehearsal_display_id"],
            front_aux_event_index=4,
            front_aux_accepted_total=96, front_aux_attempted_total=96,
            front_aux_event_summaries=identity["front_rehearsal_event_summaries"],
            result="DIAGNOSTIC_FAILURE")))
        self.assertEqual(len(panel), 1280 * 150 * 4)
        receipt = exporter.front_aux_receipt_lineage(identity)
        self.assertEqual(receipt["front_auxiliary"][
            "accepted_auxiliary_updates_total"], 96)
        self.assertEqual(receipt["rr_auxiliary"][
            "accepted_auxiliary_updates_total"], 7)
        self.assertEqual(receipt["mixed_ledger_auxiliary"][
            "accepted_auxiliary_updates_total"], 103)
        self.assertTrue(receipt["mixed_ledger_auxiliary"][
            "must_not_be_called_all_front_auxiliary"])

    def test_event4_rejects_credit_scope_and_helper_tampering(self) -> None:
        metadata = self.rr_mean_v4_actual_metadata()
        ledger = metadata["rr_postcross_workspace_branch"][exporter.FRONT_AUX_KEY]
        event = ledger["events"][3]
        cases = [
            (event, "target_semantics", "not_actual_executed_raw12"),
            (event["auxiliary_credit_breakdown"],
             "legacy_storage_key_does_not_mean_all_events_are_front_AUX", False),
            (event["helper_sha256"], "execute_rr_mean_rehearsal.py", "0" * 64),
            (ledger, "accepted_auxiliary_updates_total", 104),
        ]
        for owner, key, replacement in cases:
            original = owner[key]
            try:
                owner[key] = replacement
                with self.subTest(key=key):
                    with self.assertRaises((RuntimeError, ValueError)):
                        exporter.front_rehearsal_training_identity(metadata)
            finally:
                owner[key] = original

    def test_actual_rr_receiver_v2_is_active_and_preserves_v1(self) -> None:
        metadata = self.rr_receiver_v2_actual_metadata()
        identity = exporter.rr_reward_training_identity(metadata)
        self.assertEqual(identity["rr_workspace_semantics"],
                         exporter.RR_RECEIVER_V2_SEMANTICS)
        self.assertEqual(identity["rr_workspace_v1_semantics"],
                         exporter.RR_WORKSPACE_SEMANTICS)
        self.assertEqual(identity["rr_receiver_v2_added_ppo_updates"], 0)
        self.assertEqual(identity["rr_workspace_added_ppo_updates"], 0)
        lines = exporter.panel_lines(
            exporter.frame_summary(tick_row(8)), checkpoint=214400,
            p05_ppo_updates=115, feedback_revision=exporter.FEEDBACK_REVISION,
            feedback_ppo_updates=83,
            rr_workspace_semantics=identity["rr_workspace_semantics"],
            rr_workspace_ppo_updates=identity["rr_workspace_added_ppo_updates"],
            result="DIAGNOSTIC_FAILURE")
        self.assertIn(exporter.RR_RECEIVER_V2_SEMANTICS, lines[2])
        self.assertNotIn(exporter.RR_WORKSPACE_SEMANTICS, lines[2])

    def test_rr_receiver_v2_rejects_orphan_count_and_source_hash_tampering(self) -> None:
        metadata = self.rr_receiver_v2_actual_metadata()
        cases = [
            (metadata[exporter.RR_RECEIVER_V2_BRANCH], "semantics", "wrong_v2"),
            (metadata[exporter.RR_RECEIVER_V2_COUNTS], "ppo_updates", 1),
            (metadata[exporter.RR_RECEIVER_V2_MIGRATION][
                "rr_postcross_workspace_factor"]["preserved_metadata_sha256"],
             "rr_postcross_workspace_branch", "0" * 64),
        ]
        for owner, key, replacement in cases:
            original = owner[key]
            try:
                owner[key] = replacement
                with self.subTest(key=key):
                    with self.assertRaises((RuntimeError, ValueError)):
                        exporter.rr_reward_training_identity(metadata)
            finally:
                owner[key] = original
        orphan = self.rr_receiver_v2_actual_metadata()
        orphan.pop(exporter.RR_RECEIVER_V2_BRANCH)
        with self.assertRaises(RuntimeError):
            exporter.rr_reward_training_identity(orphan)

    def test_post_v2_ordinary_ppo_uses_current_counts_not_migration_hashes(self) -> None:
        metadata = self.rr_receiver_v2_actual_metadata()
        deltas = {"global_policy_decisions": 128, "ppo_updates": 1,
                  "optimizer_steps": 20}
        for key, delta in deltas.items():
            metadata[key] += delta
        for branch_key, count_key in (
                ("p05_capture_assist_branch", "p05_capture_assist_branch_counts"),
                ("capture_feedback_semantics_branch",
                 "capture_feedback_semantics_branch_counts"),
                ("rr_postcross_workspace_branch", "rr_postcross_workspace_branch_counts"),
                (exporter.RR_RECEIVER_V2_BRANCH, exporter.RR_RECEIVER_V2_COUNTS)):
            origin = metadata[branch_key]["counter_origin"]
            metadata[count_key] = {
                key: metadata[key] - origin[key] for key in exporter.COUNTERS}
        metadata["actor_parameter_sha256"] = "f" * 64
        reward = exporter.rr_reward_training_identity(metadata)
        auxiliary = exporter.front_rehearsal_training_identity(metadata)
        self.assertEqual(reward["rr_receiver_v2_added_ppo_updates"], 1)
        self.assertEqual(reward["rr_workspace_added_ppo_updates"], 1)
        self.assertEqual(auxiliary["front_rehearsal_display_id"],
                         "CP214528_AUXRR4")
        exporter.require_aux_destination(
            exporter.OUTPUT_ROOT / "CP214528_deterministic_rr_v2_review",
            auxiliary["front_rehearsal_display_id"], current_decisions=214528,
            latest_source_decisions=214400)

    def test_optional_preedge_absence_keeps_old_path_and_complete_triplet_is_strict(self) -> None:
        source, migrated = self.p05_preedge_fixture()
        self.assertEqual(exporter.p05_preedge_training_identity(source), {})
        identity = exporter.p05_preedge_training_identity(migrated)
        self.assertTrue(identity["p05_preedge_branch_present"])
        self.assertEqual(identity["p05_preedge_semantics"],
                         exporter.P05_PREEDGE_SEMANTICS)
        self.assertEqual(identity["p05_preedge_added_policy_decisions"], 0)
        self.assertEqual(identity["p05_preedge_added_ppo_updates"], 0)
        self.assertEqual(identity["p05_preedge_added_optimizer_steps"], 0)
        self.assertFalse(identity["p05_preedge_is_pure_policy"])
        lines = exporter.panel_lines(
            exporter.frame_summary(tick_row(8)), checkpoint=216448,
            p05_ppo_updates=131, feedback_revision=exporter.FEEDBACK_REVISION,
            feedback_ppo_updates=99,
            rr_workspace_semantics=exporter.RR_RECEIVER_V2_SEMANTICS,
            rr_workspace_ppo_updates=67,
            p05_preedge_semantics=identity["p05_preedge_semantics"],
            p05_preedge_ppo_updates=identity["p05_preedge_added_ppo_updates"],
            result="DIAGNOSTIC_FAILURE")
        self.assertTrue(any("CONTROL-MDP, NOT PURE POLICY" in line for line in lines))

        missing_counts = migrated.pop(exporter.P05_PREEDGE_COUNTS)
        try:
            with self.assertRaises(RuntimeError):
                exporter.p05_preedge_training_identity(migrated)
        finally:
            migrated[exporter.P05_PREEDGE_COUNTS] = missing_counts
        original_hash = migrated[exporter.P05_PREEDGE_MIGRATION][
            "source_checkpoint_sha256"]
        try:
            migrated[exporter.P05_PREEDGE_MIGRATION][
                "source_checkpoint_sha256"] = "0" * 64
            with self.assertRaises(RuntimeError):
                exporter.p05_preedge_training_identity(migrated)
        finally:
            migrated[exporter.P05_PREEDGE_MIGRATION][
                "source_checkpoint_sha256"] = original_hash

    def test_preedge_later_ordinary_ppo_uses_branch_counts_not_source_state_hashes(self) -> None:
        _, metadata = self.p05_preedge_fixture()
        deltas = {"global_policy_decisions": 128, "ppo_updates": 1,
                  "optimizer_steps": 20}
        for key, delta in deltas.items():
            metadata[key] += delta
        for name in (*exporter.P05_PREEDGE_PRIOR_BRANCHES,
                     "p05_preedge_approach_recovery"):
            branch_key, counts_key = name + "_branch", name + "_branch_counts"
            origin = metadata[branch_key]["counter_origin"]
            metadata[counts_key] = {
                key: metadata[key] - origin[key] for key in exporter.COUNTERS}
        metadata["actor_parameter_sha256"] = "f" * 64
        metadata["critic_parameter_sha256"] = "e" * 64
        metadata["optimizer_state_sha256"] = "d" * 64
        identity = exporter.p05_preedge_training_identity(metadata)
        self.assertEqual(identity["p05_preedge_added_policy_decisions"], 128)
        self.assertEqual(identity["p05_preedge_added_ppo_updates"], 1)
        self.assertEqual(identity["p05_preedge_added_optimizer_steps"], 20)
        self.assertEqual(identity["p05_preedge_migration_added_updates"], 0)

    def test_absent_front_ledger_preserves_pre_aux_output_and_receipt_shape(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            metadata = self.rr_workspace_metadata(Path(directory))
            identity = exporter.front_rehearsal_training_identity(metadata)
        self.assertFalse(identity["front_rehearsal_auxiliary_present"])
        combined = {"decisions": 203776, "stable": "old", **identity}
        receipt_checkpoint = exporter.receipt_checkpoint_identity(combined)
        self.assertEqual(receipt_checkpoint, {"decisions": 203776, "stable": "old"})
        self.assertIsNone(exporter.front_aux_receipt_lineage(combined))
        self.assertEqual(
            exporter.output_names(203776, False)["pair"],
            "N_vs_CP203776_DET_same_camera.mp4")
        lines = exporter.panel_lines(
            exporter.frame_summary(tick_row(8)), checkpoint=203776,
            p05_ppo_updates=32, feedback_revision=exporter.FEEDBACK_REVISION,
            feedback_ppo_updates=0, rr_workspace_semantics=exporter.RR_WORKSPACE_SEMANTICS,
            rr_workspace_ppo_updates=0, result="DIAGNOSTIC_FAILURE")
        self.assertEqual(len(lines), 7)

    def test_output_names_cannot_misname_failure_as_success(self) -> None:
        failed = exporter.output_names(
            199680, False, detail_kind="P05_THROUGH_FAILURE_ENDPOINT")
        passed = exporter.output_names(199680, True)
        self.assertEqual(failed["full"],
                         "CP199680_DET_full_after_P05_fix_capture_assist_INCOMPLETE.mp4")
        self.assertIn("_INCOMPLETE.mp4", failed["detail"])
        self.assertIn("_P05_through_failure_detail_", failed["detail"])
        self.assertIn("_P05_to_P06_detail_", passed["detail"])
        self.assertNotIn("INCOMPLETE", passed["full"])
        self.assertEqual(failed["pair"], "N_vs_CP199680_DET_same_camera.mp4")

    def test_source_error_overrides_physical_success_label(self) -> None:
        manifest = {
            "physical_task_success": True,
            "physical_episode": {"task_success": True},
            "success_candidate": False,
            "diagnostic_only": True,
            "source_acceptance_error": "RuntimeError: source validation failed",
        }
        self.assertEqual(exporter.outcome(manifest),
                         ("DIAGNOSTIC_FAILURE", False, "RuntimeError: source validation failed"))

    def test_detail_is_contiguous_and_failure_tail_is_not_cut(self) -> None:
        success_rows = [{"phase": phase, "tick": index * 8}
                        for index, phase in enumerate(("P04", "P05", "P05", "P06", "P06", "P07"), 1)]
        self.assertEqual(exporter.detail_interval(success_rows), (1, 5, "CONTIGUOUS_P05_TO_P06"))
        failed_rows = [{"phase": phase, "tick": index * 8}
                       for index, phase in enumerate(("P04", "P05", "P05", "P05"), 1)]
        self.assertEqual(exporter.detail_interval(failed_rows),
                         (1, 4, "P05_THROUGH_FAILURE_ENDPOINT"))

    def test_failure_detail_filename_and_visible_title_match_interval(self) -> None:
        rows = [{"phase": phase, "tick": index * 8}
                for index, phase in enumerate(("P04", "P05", "P05", "P05"), 1)]
        full = Path("CP216448_AUXRR4_DET_full_after_P05_fix_capture_assist_INCOMPLETE.mp4")
        output = Path(
            "CP216448_AUXRR4_DET_P05_through_failure_detail_capture_assist_INCOMPLETE.mp4")
        with patch.object(exporter, "run") as run, \
                patch.object(exporter, "validate_output", return_value={"valid": True}), \
                patch.object(exporter, "preview", return_value=[]):
            detail = exporter.encode_detail(full, rows, output, ffmpeg=Path("ffmpeg.exe"))
        video_filter = run.call_args.args[0][run.call_args.args[0].index("-vf") + 1]
        self.assertIn("P05_through_failure", video_filter)
        self.assertNotIn("P05_to_P06", video_filter)
        self.assertEqual(
            detail["visible_title_overlay"],
            "DETAIL | CP216448_AUXRR4 | P05_through_failure | NO P06 OBSERVED")
        self.assertTrue(detail["no_intro_frames"])

    def test_detail_is_omitted_when_failure_precedes_p05(self) -> None:
        rows = [{"phase": phase, "tick": index * 8}
                for index, phase in enumerate(("P01", "P02", "P03", "P04"), 1)]
        self.assertIsNone(exporter.detail_interval(rows))

        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "must_not_exist.mp4"
            detail = exporter.encode_detail(
                Path(directory) / "full.mp4", rows, output,
                ffmpeg=Path(directory) / "ffmpeg-must-not-run.exe")
        self.assertIsNone(detail["output"])
        self.assertTrue(detail["omitted"])
        self.assertEqual(detail["reason"], "NO_P05_OBSERVED")
        self.assertTrue(detail["not_substituted_from_another_checkpoint"])
        self.assertFalse(output.exists())

    def test_tick_ledger_selection_uses_exact_frame_endpoints(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "capture_assist_ticks.jsonl"
            with path.open("w", encoding="utf-8", newline="\n") as stream:
                for tick in range(1, 13):
                    phase = "P04" if tick < 5 else "P05" if tick < 10 else "P06"
                    stream.write(json.dumps(tick_row(tick, phase=phase, pending=tick >= 5)) + "\n")
            context = {"tick_path": path, "endpoint": 12}
            rows, selected = exporter.capture_rows(
                context, [SimpleNamespace(sim_step=8), SimpleNamespace(sim_step=12)])
        self.assertEqual([row["tick"] for row in rows], [8, 12])
        self.assertTrue(any(row["phase"] == "P05" for row in selected))
        self.assertTrue(any(row["phase"] == "P06" for row in selected))

    def test_response_states_noncausal_pending_limitation(self) -> None:
        markdown = exporter.response_markdown([exporter.frame_summary(tick_row(8))])
        self.assertIn("is not a placed/contact claim", markdown)


if __name__ == "__main__":
    unittest.main()
