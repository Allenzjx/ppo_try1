"""Standard-library-only rules/contract/syntax, no Torch or physical credit."""
import ast
import itertools
import math
from pathlib import Path
import sys
import unittest

from semantic_rear_cooperative_prep_sigma import (
    cooperative_prep_multipliers, scalar_effective_log_std)


def flags(carry=False, reachable=False, prep=False, receiving=False):
    return dict(rr_carry_capture=carry, rr_top_reachable=reachable,
                rl_prep_transfer=prep, receiving_continuation_active=receiving)


class CooperativeScalarTests(unittest.TestCase):
    def test_complete_boolean_truth_table_exact_selected_totals(self):
        for carry, reachable, prep, receiving in itertools.product((False, True), repeat=4):
            m, ev = cooperative_prep_multipliers(**flags(carry, reachable, prep, receiving))
            expected = [1.]*12
            if carry and reachable:
                expected[6] = 4.
            if carry and reachable or prep:
                for i, value in ((3, 16.), (1, 2.), (4, 2.), (5, 8.)):
                    expected[i] = value
            if carry and reachable and not receiving:
                expected[8] = 2.
            self.assertEqual(m, tuple(expected))
            self.assertEqual(ev['prep_allowed'], carry and reachable or prep)
            self.assertFalse(ev['support_transfer_permission_modified'])

    def test_precontact_sigma_from_reported_checkpoint_not_double_rr(self):
        prior = {1: .11675, 3: .01902, 4: .08211, 5: .005988, 6: 1.13082, 8: .20696}
        expected = {1: .2335, 3: .30432, 4: .16422, 5: .047904, 6: 1.13082, 8: .41392}
        m, _ = cooperative_prep_multipliers(**flags(True, True))
        for i, value in prior.items():
            # Existing RR value ALREADY includes ×4. Other AIR rear totals were1.
            self.assertAlmostEqual(value*m[i]/(4. if i == 6 else 1.), expected[i])

    def test_postcontact_fr_total_is_16_not_old2_times16(self):
        m, _ = cooperative_prep_multipliers(**flags(prep=True, receiving=True))
        self.assertEqual(m[3], 16.)
        self.assertEqual(m[3]/2., 8.)
        self.assertEqual(m[1]/2., 1.)
        self.assertEqual(m[8], 1.)

    def test_parent_receiving_blocks_fl_extra_even_during_recapture(self):
        m, ev = cooperative_prep_multipliers(**flags(True, True, False, True))
        self.assertTrue(ev['prep_allowed'])
        self.assertFalse(ev['fl_wheel_extra_active'])
        self.assertEqual(m[8], 1.)
        self.assertEqual(m[6], 4.)

    def test_unselected_channels_and_early_front_exact_identity(self):
        for combination in itertools.product((False, True), repeat=4):
            m, _ = cooperative_prep_multipliers(**flags(*combination))
            self.assertTrue(all(m[i] == 1. for i in (0, 2, 7, 9, 10, 11)))
        old = tuple(-2.+i/15 for i in range(12))
        actual, _ = scalar_effective_log_std(old, **flags())
        self.assertEqual(actual, old)

    def test_invalid_flags_and_nonfinite_parent_not_silently_coerced(self):
        for value in (0, 1, None, .5, 'true'):
            bad = flags(); bad['rr_carry_capture'] = value
            with self.assertRaises(ValueError):
                cooperative_prep_multipliers(**bad)
        with self.assertRaises(ValueError):
            scalar_effective_log_std([math.nan]*12, **flags())

    def test_profile_actor_and_deferred_tests_parse_without_loading_torch(self):
        for name in ('semantic_rear_cooperative_prep_profile.py', 'semantic_rear_cooperative_prep_actor.py',
                     'test_cooperative_prep_tensor.py'):
            ast.parse(Path(__file__).with_name(name).read_text(encoding='utf-8'))
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('isaaclab', sys.modules)


if __name__ == '__main__':
    unittest.main()
