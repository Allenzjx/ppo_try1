"""Pure-stdlib tests.  They do not import Torch, PXR, or simulator modules."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


HERE = Path(__file__).resolve().parent


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


DATA = load("staged_front_retention439_data", "data_contract.py")
KERNEL = load("staged_front_retention439_kernel", "front_retention439.py")


def phase_for(index: int) -> str:
    if index < 10:
        return "P01"
    if index < 30:
        return "P02"
    if index < 40:
        return "P03"
    if index < 60:
        return "P05"
    if index < 80:
        return "P06"
    if index < 90:
        return "P07"
    return "P09"


def observation(phase: str) -> list[float]:
    result = [0.0] * DATA.OBSERVATION_DIMENSION
    result[int(phase[1:]) - 1] = 1.0
    return result


def row(index: int) -> dict:
    # Post-cutoff payload is deliberately unusable; only its contiguous
    # zero-based decision_index may be considered by the validator.
    if index > DATA.CUTOFF_INCLUSIVE:
        return {"decision_index": index, "phase": "P09",
                "overridden_indices": [1, 3, 6]}
    phase = phase_for(index)
    raw = [index / 10000.0 + channel / 1000.0 for channel in range(12)]
    return {"decision_index": index, "phase": phase,
        "policy_observation_vector": observation(phase),
        "original_student_raw_full12": raw,
        "original_student_request_audit": {"selected_raw_full12": list(raw)},
        "applied_raw_full12": list(raw), "overridden_indices": [],
        "probe_active": False, "new_PPO_samples": 0, "new_PPO_updates": 0,
        "new_optimizer_steps": 0, "new_AUX_accepted_updates": 0,
        "new_AUX_attempted_steps": 0, "teacher_actions_deployed": False,
        "diagnostic_rows_entered_in_on_policy_storage": False,
        "formal_deterministic_policy_result": False,
        "step_info": {"raw_policy_action_full12": list(raw),
                      "applied_action_full12": list(raw)}}


def rows() -> list[dict]:
    return [row(index) for index in range(DATA.SOURCE_DECISION_ROWS)]


class DataContractTest(unittest.TestCase):
    def test_balanced_contiguous_pre_intervention_plan(self):
        result = DATA.plan_probe_rows(rows(), DATA.reviewed_source_binding())
        receipt = result["receipt"]
        self.assertEqual(receipt["decision_index_cutoff_inclusive"], 997)
        self.assertEqual(receipt["post_cutoff_rows_strictly_excluded"], 370)
        self.assertEqual(receipt["phase_columns"], [1, 4, 5, 8])
        self.assertEqual(receipt["rows_per_phase_per_partition"], 10)
        self.assertEqual(len(receipt["train_row_ids"]), 40)
        self.assertEqual(len(receipt["holdout_row_ids"]), 40)
        self.assertTrue(set(receipt["train_row_ids"]).isdisjoint(
            receipt["holdout_row_ids"]))
        self.assertTrue(all(index <= 997 for index in
                            receipt["train_row_ids"] + receipt["holdout_row_ids"]))
        self.assertFalse(receipt["synthetic_or_intervention_rows_used"])

    def test_pre_intervention_raw_mismatch_fails(self):
        value = rows()
        value[997]["applied_raw_full12"][0] += 1.0
        with self.assertRaisesRegex(ValueError, "raw/applied/request audit"):
            DATA.plan_probe_rows(value, DATA.reviewed_source_binding())

    def test_legal_physical_transform_is_not_a_raw_override(self):
        value = rows()
        value[0]["step_info"]["applied_action_full12"] = [-.10862, 11., 2., 3., 8.53369] + [0.] * 7
        result = DATA.plan_probe_rows(value, DATA.reviewed_source_binding())
        self.assertEqual(result["receipt"]["step_applied_action_validated_as"],
                         "finite_physical_full12_not_raw_label")
        self.assertTrue(all(row["target_raw"] != tuple(value[0]["step_info"]["applied_action_full12"])
                            for row in result["train_rows"] + result["holdout_rows"]))

    def test_actual_raw_audit_or_step_replacement_still_fails(self):
        for container, key in (("original_student_request_audit", "selected_raw_full12"),
                               ("step_info", "raw_policy_action_full12")):
            value = rows()
            value[0][container][key] = list(value[0][container][key])
            value[0][container][key][0] += .001
            with self.subTest(container=container), self.assertRaisesRegex(ValueError, "raw/applied/request audit"):
                DATA.plan_probe_rows(value, DATA.reviewed_source_binding())

    def test_nonfinite_physical_action_still_fails(self):
        value = rows()
        value[0]["step_info"]["applied_action_full12"] = [float("nan")] + [0.] * 11
        with self.assertRaisesRegex(ValueError, "physical applied action must be finite"):
            DATA.plan_probe_rows(value, DATA.reviewed_source_binding())

    def test_wrong_zero_based_index_fails(self):
        value = rows()
        value[13]["decision_index"] = 14
        with self.assertRaisesRegex(ValueError, "contiguous zero-based"):
            DATA.plan_probe_rows(value, DATA.reviewed_source_binding())

    def test_source_binding_is_exact(self):
        value = DATA.reviewed_source_binding()
        value["source_decisions"] = dict(value["source_decisions"])
        value["source_decisions"]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "binding differs"):
            DATA.plan_probe_rows(rows(), value)

    def test_real_invariance_requires_available_P10_P12_not_unreached_P13(self):
        values = [{"row_id": phase, "phase": phase,
                   "policy_observation_vector": observation(phase)}
                  for phase in DATA.REAL_INVARIANCE_PHASES]
        result = DATA.validate_invariance_rows(values)
        self.assertEqual(result["phases"], ["P10", "P11", "P12"])
        self.assertEqual(result["actual_rows_by_phase"], {"P10":1,"P11":1,"P12":1,"P13":0})
        self.assertFalse(result["real_P13_validation_claimed"])
        with self.assertRaisesRegex(ValueError, "available P10-P12"):
            DATA.validate_invariance_rows(values[:-1])
        with self.assertRaisesRegex(ValueError, "P13 is synthetic-only"):
            DATA.validate_invariance_rows(values + [{"row_id":"synthetic_P13", "phase":"P13",
                "policy_observation_vector":observation("P13")}])


class KernelStdlibContractTest(unittest.TestCase):
    def test_sparse_whitelist_and_budget(self):
        self.assertEqual(KERNEL.PHASE_COLUMNS, (1, 4, 5, 8))
        self.assertEqual(KERNEL.OPTIMIZED_SCALAR_COUNT, 1024)
        self.assertEqual(KERNEL.OPTIMIZED_PARAMETERS,
                         ("actor.mlp.0.weight[:,[1,4,5,8]]",))
        KERNEL.Budget(32, 1e-5, (0.1,) * 12, (0.1,) * 12,
                      0.01, 0.02).validate()
        with self.assertRaisesRegex(ValueError, "1..32"):
            KERNEL.Budget(33, 1e-5, (0.1,) * 12, (0.1,) * 12,
                          0.01, 0.02).validate()

    def test_ledger_facing_report_contract(self):
        protected = {"critic_parameter_sha256": "a" * 64,
            "optimizer_state_sha256": "b" * 64,
            "normalizer_state_sha256": "c" * 64,
            "training_rng_state_sha256": "d" * 64,
            "optimizer_learning_rate": 1e-5}
        report = {"schema": KERNEL.FIT_SCHEMA,
            "data_receipt": {"path": "C:\\sealed\\selected_rows.json",
                             "sha256": "0" * 64},
            "budget": {"max_attempts": 32, "learning_rate": 1e-5,
                "maximum_train_request_shift_full12": (0.1,) * 12,
                "maximum_validation_request_shift_full12": (0.1,) * 12,
                "maximum_per_state_full_gaussian_kl": 0.01,
                "maximum_abs_log_sigma_change": 0.02},
            "optimized_parameters": list(KERNEL.OPTIMIZED_PARAMETERS),
            "optimized_scalar_count": 1024,
            "accepted_auxiliary_updates": 1,
            "attempted_auxiliary_optimizer_steps": 1,
            "stop_reason": "finite_budget_exhausted",
            "first_rejected_proposal_restored_and_stopped": False,
            "actor_parameter_sha256_before": "e" * 64,
            "actor_parameter_sha256_after": "f" * 64,
            "protected_state_before": protected,
            "protected_state_after": dict(protected),
            "PPO_decisions_added": 0, "PPO_updates_added": 0,
            "PPO_optimizer_steps_added": 0,
            "real_P10_P12_full_Gaussian_bitwise_unchanged": True,
            "synthetic_P13_full_Gaussian_bitwise_unchanged": True,
            "zero_selected_phase_columns_P10_P13_algebra_verified": True,
            "invariance_evidence": {
                "schema":KERNEL.INVARIANCE_SCHEMA,
                "actual_rows_by_phase":{"P10":1,"P11":1,"P12":1,"P13":0},
                "real_P13_validation_claimed":False,"synthetic_P13_rows":3,
                "synthetic_P13_origin":"first_real_row_per_P10_P12_with_only_phase_onehot_replaced_test_fixture",
                "synthetic_rows_used_for_fit":False,"selected_phase_columns":[1,4,5,8],
                "real_observations_float32_sha256":"a"*64,
                "synthetic_P13_observations_float32_sha256":"b"*64,
                "same_future_trajectory_claimed":False},
            "nonselected_actor_state_unchanged": True,
            "fresh_PPO_rollout_required": True,
            "targets_were_executed_student_raw_requests": True,
            "targets_were_success_or_teacher_labels": False,
            "teacher_deployed": False, "physical_success_claimed": False}
        KERNEL.validate_fit_report(report, publishable=True)
        report["optimized_parameters"] = ["actor.mlp.0.weight[:,0:9]"]
        with self.assertRaisesRegex(ValueError, "sparse whitelist"):
            KERNEL.validate_fit_report(report, publishable=True)

    def test_import_does_not_load_numeric_stack_and_has_no_writer(self):
        self.assertNotIn("torch", sys.modules)
        source = (HERE / "front_retention439.py").read_text(encoding="utf-8")
        self.assertNotIn("save_semantic_checkpoint", source)
        self.assertNotIn("publish_", source)
        self.assertNotIn("argparse", source)


if __name__ == "__main__":
    unittest.main()
