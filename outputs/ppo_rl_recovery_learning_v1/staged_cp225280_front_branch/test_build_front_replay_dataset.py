"""Small stdlib tests for the CP225280 front replay selector."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys
import unittest


MODULE_PATH = Path(__file__).with_name("build_front_replay_dataset.py")
SPEC = importlib.util.spec_from_file_location("_front_replay_builder", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


def row(index: int, phase: str) -> dict:
    observation = [0.0] * 439
    observation[int(phase[1:]) - 1] = 1.0
    mean = [index / 1000.0 + channel / 100.0 for channel in range(12)]
    sigma = [0.05 + channel / 1000.0 for channel in range(12)]
    return {"new_PPO_samples": 0, "new_PPO_updates": 0,
        "new_optimizer_steps": 0, "new_AUX_accepted_updates": 0,
        "new_AUX_attempted_steps": 0, "teacher_actions_deployed": False,
        "diagnostic_rows_entered_in_on_policy_storage": False,
        "formal_deterministic_policy_result": False,
        "decision_index": index, "phase": phase,
        "policy_observation_vector": observation,
        "original_student_raw_full12": list(mean),
        "original_student_request_audit": {
            "mode": "deterministic_conditional_mean",
            "policy_version": M.SOURCE_POLICY, "rho": 0.9,
            "exploration_std_temperature": 0.25,
            "conditional_mean_full12": list(mean),
            "effective_sigma_full12": list(sigma),
            "effective_log_std_full12": [math.log(value) for value in sigma],
            "selected_raw_full12": list(mean)},
        "applied_raw_full12": list(mean), "overridden_indices": [],
        "probe_active": False,
        "step_info": {"raw_policy_action_full12": list(mean)}}


class FrontReplayTests(unittest.TestCase):
    def source(self) -> list[dict]:
        phases = (["P01"] * 2 + ["P02"] * 100 + ["P03"] * 4 +
                  ["P04"] + ["P05"] * 80 + ["P06"] * 70)
        return [row(index, phase) for index, phase in enumerate(phases)]

    def test_uniform_positions_include_entire_range(self) -> None:
        values = M.uniform_positions(340, 64)
        self.assertEqual((len(values), values[0], values[-1]), (64, 0, 339))
        self.assertEqual(len(values), len(set(values)))

    def test_stratified_selection_and_interleaved_holdout(self) -> None:
        source = self.source()
        result = M.select_rows(source, cutoff_inclusive=len(source) - 1,
                               expected_total_rows=len(source))
        counts = {phase: result["phase_selection"][phase]["selected_count"]
                  for phase in M.FRONT_PHASES}
        self.assertEqual(counts, {"P01": 2, "P02": 64, "P03": 4,
                                  "P04": 1, "P05": 64, "P06": 64})
        self.assertEqual(len(result["training_rows"]), 100)
        self.assertEqual(len(result["heldout_rows"]), 99)
        self.assertEqual(result["phase_selection"]["P02"]
                         ["selected_last_decision_index"], 101)
        self.assertFalse(result["phase_selection"]["P04"]
                         ["phase_specific_holdout_available"])

    def test_rejects_intervention(self) -> None:
        value = row(0, "P01")
        value["probe_active"] = True
        with self.assertRaisesRegex(ValueError, "intervention"):
            M.compact_front_row(value, 0)

    def test_rejects_wrong_gaussian_or_history_identity(self) -> None:
        value = row(0, "P01")
        value["original_student_request_audit"]["effective_sigma_full12"][0] = 2.0
        with self.assertRaisesRegex(ValueError, "sigma and log std"):
            M.compact_front_row(value, 0)
        value = row(0, "P01")
        value["original_student_request_audit"]["rho"] = 0.8
        with self.assertRaisesRegex(ValueError, "HISTORY identity"):
            M.compact_front_row(value, 0)

    def test_rejects_mean_raw_mismatch(self) -> None:
        value = row(0, "P01")
        value["original_student_raw_full12"][0] += 0.1
        with self.assertRaisesRegex(ValueError, "mean/raw"):
            M.compact_front_row(value, 0)


if __name__ == "__main__":
    unittest.main()
