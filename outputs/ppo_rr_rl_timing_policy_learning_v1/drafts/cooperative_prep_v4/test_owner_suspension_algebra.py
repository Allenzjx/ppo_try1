"""Pure design counterexamples; arbitrary angles are not robot targets.

This does not import/modify the real mapper or claim physical safety. The
stateful candidate is an explicit relative-input transform, not decomposition.
"""
import math
import unittest


def slew(previous, candidate, rate=1.25):
    return previous+max(-rate,min(rate,candidate-previous))


def held_candidate(final_anchor, request_at_entry, current_request, capacity=36.):
    if not all(math.isfinite(x) for x in (final_anchor,request_at_entry,current_request,capacity)) or capacity <= 0:
        raise ValueError('finite explicit anchor/request/capacity required')
    return final_anchor+max(-capacity,min(capacity,current_request-request_at_entry))


class Counterexamples(unittest.TestCase):
    def test_frozen_source_or_real_mapped_N_does_not_hold_executed_target(self):
        final, fixed_mapped, request = 20., 10., -2.
        self.assertLess(slew(final,fixed_mapped+request),final)
    def test_separate_diagnostic_nominal_cannot_be_called_executed_hold(self):
        final, diagnostic_trace, request = 20., 12., -2.
        self.assertNotEqual(slew(final,diagnostic_trace+request),final)
    def test_constant_request_hold_no_final_minus_request_attribution(self):
        for anchor,request in ((20.,-2.),(-57.,-36.),(130.,24.)):
            self.assertEqual(held_candidate(anchor,request,request),anchor)
    def test_policy_changes_keep_both_directions(self):
        self.assertGreater(slew(20.,held_candidate(20.,-2.,1.)),20.)
        self.assertLess(slew(20.,held_candidate(20.,-2.,-4.)),20.)
    def test_difference_does_not_silently_double_capacity(self):
        self.assertEqual(held_candidate(0.,-36.,36.),36.)
    def test_fixed_anchor_allows_slew_to_finish_policy_change_not_disappear(self):
        candidate=held_candidate(20.,-2.,1.)
        first=slew(20.,candidate); second=slew(first,candidate)
        self.assertGreater(second,first)
        self.assertLessEqual(second,candidate)
    def test_unknown_anchor_not_fabricated(self):
        with self.assertRaises(ValueError): held_candidate(float('nan'),0.,0.)


if __name__=='__main__': unittest.main()
