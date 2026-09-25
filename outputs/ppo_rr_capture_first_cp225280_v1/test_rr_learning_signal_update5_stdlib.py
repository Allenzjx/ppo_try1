"""Small stdlib coverage for the bounded resumed-update reader."""
import sys
import unittest
from pathlib import Path
import analyze_rr_learning_signal_update5 as m


class Update5Tests(unittest.TestCase):
    def test_import_is_stdlib_only(self):
        self.assertNotIn('torch',sys.modules)
        self.assertNotIn('pxr',sys.modules)

    def test_explicit_exit_guard_precedes_any_path_read(self):
        with self.assertRaisesRegex(ValueError,'Isaac exit'):
            m.analyze(Path('does_not_exist'))
        self.assertNotIn('torch',sys.modules)

    def test_positive_knee_deviation_negative_advantage_not_relabelled(self):
        raw,mean,sigma = [0.]*12,[0.]*12,[1.]*12
        raw[7],mean[7],sigma[7] = -.4,-.7,.3
        row = dict(policy={'selected_raw_full12':raw,'conditional_mean_full12':mean,
                   'active_conditional_std_full12':sigma},
                   reward=1.,raw_gae=-4.,advantage=-2.,**{'return':-3.},
                   local_reward={'potential_shaping':1.,'terminal_event':0.},
                   before={'gap_m':.03,'actual_rr_hip_knee_deg':[0.,-30.]},
                   after={'gap_m':.02,'actual_rr_hip_knee_deg':[0.,-29.]})
        result = m.knee_signal([row])['groups']['above_collection_mean']
        self.assertLess(result['mean_score_times_advantage']['mean'],0.)
        self.assertEqual(result['normalized_advantage']['mean'],-2.)
        self.assertGreater(result['actual_knee_delta_deg']['mean'],0.)


if __name__=='__main__':
    unittest.main(verbosity=2)

