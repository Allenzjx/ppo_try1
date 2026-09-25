"""Pure-stdlib reader checks using one real saved row plus labelled mutations."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

MODULE = Path(__file__).with_name("analyze_actual_metrics.py")
SPEC = importlib.util.spec_from_file_location("_actual_metrics_reader", MODULE)
reader = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reader)
REAL_SOURCE = reader.ROOT / (
    "runs/ppo_rr_capture_first_cp225280_v1/"
    "diagnostic_hipminus25_kneeplus20_ecf205e/source/video_policy_decisions.jsonl")


class ReaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with REAL_SOURCE.open("rb") as stream:
            cls.line = stream.readline()
        cls.row = json.loads(cls.line)

    def temporary_log(self, payload):
        directory = tempfile.TemporaryDirectory(prefix="rr_metrics_reader_test_")
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "decisions.jsonl"
        path.write_bytes(payload)
        return path

    def test_real_row_prefix_zero_and_exact_reference(self):
        path = self.temporary_log(self.line)
        baseline, index = reader.analyze(path)
        report, _ = reader.analyze(path, reference=index)
        self.assertTrue(report["prefix"]["strict_zero_verified"])
        self.assertEqual(report["prefix"]["reference_compared_rows"], 1)
        self.assertEqual(report["prefix"]["reference_raw_max_abs_delta"], 0.)
        self.assertEqual(report["prefix"]["reference_final_max_abs_delta"], 0.)
        self.assertIsNone(baseline["gate_entry"])

    def test_live_ignores_only_unfinished_tail(self):
        path = self.temporary_log(self.line + b'{"decision":2')
        status = {}
        self.assertEqual(len(list(reader.decisions(path, True, status))), 1)
        self.assertEqual(status["ignored_incomplete_tail_bytes"], len(b'{"decision":2'))
        with self.assertRaisesRegex(ValueError, "unfinished"):
            list(reader.decisions(path, False, {}))
        malformed = self.temporary_log(self.line + b'{"bad":\n')
        with self.assertRaisesRegex(ValueError, "malformed complete line"):
            list(reader.decisions(malformed, True, {}))

    def test_snapshot_does_not_follow_append(self):
        path = self.temporary_log(self.line)
        status = {}
        rows = reader.decisions(path, True, status)
        next(rows)
        with path.open("ab") as stream:
            stream.write(self.line)
        self.assertEqual(list(rows), [])
        self.assertEqual(status["complete_rows"], 1)

    def test_snapshot_uses_handle_eof_not_directory_stat(self):
        path = self.temporary_log(self.line)
        real_stat = Path.stat
        def stale_stat(candidate, *args, **kwargs):
            if candidate == path:
                raise AssertionError("reader must not use possibly stale directory size")
            return real_stat(candidate, *args, **kwargs)
        status = {}
        with patch.object(Path, "stat", stale_stat):
            self.assertEqual(len(list(reader.decisions(path, True, status))), 1)
        self.assertEqual(status["snapshot_bytes"], len(self.line))
        self.assertEqual(status["snapshot_size_source"], "open_handle_seek_END_tell")

    def test_nonzero_or_missing_local_is_not_silent_zero(self):
        mutated = copy.deepcopy(self.row)
        mutated["policy_request"]["local_raw_mean_delta_full12"][6] = .001
        path = self.temporary_log(json.dumps(mutated).encode() + b"\n")
        report, _ = reader.analyze(path)
        self.assertFalse(report["prefix"]["strict_zero_verified"])
        del mutated["policy_request"]["capture_active"]
        path = self.temporary_log(json.dumps(mutated).encode() + b"\n")
        report, _ = reader.analyze(path)
        self.assertFalse(report["prefix"]["strict_zero_verified"])
        self.assertEqual(report["prefix"]["unknown_request_gate_rows"], 1)

    def test_real_measurements_preserve_clock_and_not_target(self):
        result = reader.sample(self.row)
        info = self.row["step_info"]
        self.assertEqual(result["rr"]["actual_rr_deg"], info["rr_capture_local"]["metrics"]["actual_rr_hip_knee_deg"])
        self.assertNotEqual(result["rr"]["actual_rr_deg"], result["rr"]["final_rr_deg"])
        self.assertEqual(result["clock"]["rr_actual_observation_tick"], result["tick"])
        self.assertNotEqual(result["clock"]["native_dispatch_tick"], result["tick"])
        self.assertEqual(result["fl_wheel"]["measured_canonical_velocity_rad_s"],
                         info["semantic_task"]["physical_evaluator"]["measured_wheel_velocity_rad_s"][0])

    def test_diagnostic_candidate_is_separate_from_actual_final(self):
        mutated = copy.deepcopy(self.row)
        mutated["policy_request"]["independent_diagnostic"] = {
            "candidate_absolute_targets": {"6": -20., "7": -38.}}
        result = reader.sample(mutated)
        self.assertEqual(result["rr"]["diagnostic_candidate_rr_deg"], [-20., -38.])
        self.assertEqual(result["rr"]["final_rr_deg"], reader.sample(self.row)["rr"]["final_rr_deg"])
        path = self.temporary_log(json.dumps(mutated).encode() + b"\n")
        report, _ = reader.analyze(path)
        self.assertFalse(report["prefix"]["no_diagnostic_override"])
        self.assertFalse(report["prefix"]["strict_zero_verified"])


if __name__ == "__main__":
    unittest.main()
