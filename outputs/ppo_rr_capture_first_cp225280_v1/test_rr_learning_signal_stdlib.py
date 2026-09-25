"""No tensor imports: safe while the real simulator continues."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

path=Path(__file__).with_name("analyze_rr_learning_signal.py")
spec=importlib.util.spec_from_file_location("learning_signal",path)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def metrics(**changes):
    value=dict(current_top_contact=False,current_top_bearing=False,
               ground_contact=False,free_air=True,within_top_xy=True,gap_m=.05)
    value.update(changes)
    return value


class SignalTests(unittest.TestCase):
    def test_import_has_no_tensor_or_robot_modules(self):
        self.assertNotIn("torch",sys.modules)
        self.assertNotIn("pxr",sys.modules)
        self.assertFalse(any(k.startswith("wlr50_clean") for k in sys.modules))

    def test_requires_explicit_exit_ack_before_any_io_or_tensor_import(self):
        with self.assertRaisesRegex(ValueError,"Isaac exit"):
            m.analyze(Path("does_not_exist"))
        self.assertNotIn("torch",sys.modules)

    def test_air_gap_descent(self):
        self.assertEqual(m.physical_bucket(metrics(),metrics(gap_m=.049),False),"AIR_legal_gap_decreasing")
        self.assertEqual(m.physical_bucket(metrics(),metrics(gap_m=.051),False),"AIR_other")

    def test_first_contact_and_reacquisition_distinct(self):
        after=metrics(current_top_contact=True,current_top_bearing=True,free_air=False)
        self.assertEqual(m.physical_bucket(metrics(),after,False),"first_TOP_endpoint")
        self.assertEqual(m.physical_bucket(metrics(),after,True),"TOP_reacquisition")

    def test_hold_and_drop_distinct(self):
        bearing=metrics(current_top_contact=True,current_top_bearing=True,free_air=False)
        self.assertEqual(m.physical_bucket(bearing,bearing,True),"TOP_bearing_hold")
        self.assertEqual(m.physical_bucket(bearing,metrics(),True),"bearing_drop_endpoint")

    def test_ground_not_old_air_success(self):
        self.assertEqual(m.physical_bucket(metrics(),metrics(ground_contact=True,free_air=False),True),"GROUND_not_RR_success")

    def test_air_after_top_is_recapture(self):
        self.assertEqual(m.physical_bucket(metrics(),metrics(),True),"AIR_recapture_after_TOP")

    def test_statistics_keep_negative_advantage(self):
        self.assertEqual(m.stats([-2.,1.])["mean"],-.5)
        self.assertEqual(m.stats([-2.,1.])["negative"],1)
        self.assertEqual(m.stats([]),{"n":0})
        with self.assertRaises(ValueError):
            m.stats([float("nan")])

    def test_incomplete_tail_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/"rows.jsonl"
            p.write_bytes(b'{"a":1}')
            with self.assertRaisesRegex(ValueError,"incomplete"):
                list(m.json_rows(p))

    def test_identical_gaussian_zero_KL(self):
        self.assertEqual(m.gaussian_kl_parts(.7,.3,.7,.3),dict(mean=0.,scale=0.,total=0.))

    def test_mean_shift_KL_sensitive_to_sigma(self):
        large=m.gaussian_kl_parts(0.,.3,.03,.3)
        small=m.gaussian_kl_parts(0.,.03,.03,.03)
        self.assertAlmostEqual(large["total"],.005)
        self.assertAlmostEqual(small["total"],.5)
        self.assertEqual(small["scale"],0.)

    def test_std_KL_is_separate_and_directional(self):
        a=m.gaussian_kl_parts(0.,.3,0.,.6)
        b=m.gaussian_kl_parts(0.,.6,0.,.3)
        self.assertEqual(a["mean"],0.)
        self.assertGreater(a["scale"],0.)
        self.assertNotEqual(a["total"],b["total"])


if __name__=="__main__":
    unittest.main()
