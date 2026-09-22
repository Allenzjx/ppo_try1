"""CPU tests and assertions against already-saved real runs; never launches Isaac."""
import json
import math
from pathlib import Path
import unittest
from video_event_windows_readonly import finite_tree, measured_window
from rr_probe_readonly import B, compact_physical, lines, physical_window

HERE = Path(__file__).resolve().parent


class EventWindowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.b = json.loads((HERE / "event_windows_selfcheck_success_B.json").read_text())
        cls.a = json.loads((HERE / "event_windows_selfcheck_incomplete_RRA.json").read_text())
        cls.first = compact_physical(next(lines(B / "physical_observations.jsonl")))

    def test_authoritative_partial_terminal_is_retained(self):
        self.assertEqual(self.b["outcome"]["final_tick"],8857)
        self.assertTrue(self.b["outcome"]["physical_task_success"])
        self.assertIsNone(self.b["outcome"]["first_unfinished_derived_from_physical_history"])

    def test_same_real_prefix_same_metrics(self):
        for name in ("FR_P01_to_capture","FL_cross_to_capture","P06_rear_approach"):
            self.assertEqual(self.a["windows"][name],self.b["windows"][name])

    def test_old_RR_metric_exact_agreement(self):
        saved = json.loads((HERE / "RR_lift_FL_RL_minus3_N_analysis.json").read_text())
        expected = saved["reference"]["windows"]["qualified_to_placed_or_budget"]["physical_120Hz"]["rate_RMS_rad_s"]
        self.assertEqual(self.b["windows"]["RR_qualified_to_place"]["physical"]["rate_RMS_rad_s"],expected)

    def test_real_incomplete_is_not_task_failure_or_success(self):
        self.assertEqual(self.a["outcome"]["classification"],"INCOMPLETE")
        self.assertIsNone(self.a["outcome"]["termination_reason"])
        window = self.a["windows"]["RR_preparation_to_place"]
        self.assertEqual(window["status"],"INCOMPLETE")
        self.assertIsNone(window["goal_tick"])
        self.assertEqual(window["observed_end_tick"],7200)

    def test_missing_start_is_not_zero_duration(self):
        result = measured_window([],{},[],[],None,None,99)
        self.assertEqual(result["status"],"NOT_REACHED")
        self.assertIsNone(result["physical"])

    def test_missing_or_nonfinite_sample_is_not_zero(self):
        for error in ({},{1:"nonfinite terminal geometry"}):
            result = measured_window([],error,[],[],0,None,2)
            self.assertIsNone(result["physical"])
            self.assertEqual(result["status"],"INCOMPLETE")
        self.assertFalse(finite_tree({"rpy":[float("nan")]}))
        self.assertFalse(finite_tree({"nonfinite_raw_float":"nan"}))

    def test_RMS_not_divided_by_duration_again(self):
        def run(n,dt):
            data = [dict(self.first,tick=i,t=i*dt,rpy=(i*.01,0.,0.)) for i in range(n+1)]
            return physical_window(data,0,n)["rate_RMS_rad_s"]
        self.assertAlmostEqual(run(2,1.),.01/math.sqrt(2.))
        self.assertAlmostEqual(run(20,1.),run(2,1.))
        self.assertAlmostEqual(run(20,2.),run(20,1.)/2.)


if __name__ == "__main__":
    unittest.main()
