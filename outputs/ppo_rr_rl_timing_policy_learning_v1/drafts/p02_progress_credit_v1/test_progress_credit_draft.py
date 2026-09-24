"""Pure outputs-only design tests; not proof of Isaac success or production wiring."""
import unittest

from progress_credit_draft import FrontProgressCredit


class CreditDraftTests(unittest.TestCase):
    def step(self, state, r, *, eligible=True, p02=True, age=22.4, abort=False):
        return state.observe(remaining_m=r, dt_s=1 / 120, eligible=eligible,
                             p02=p02, episode_age_s=age, physical_abort=abort)

    def earn(self, state):
        self.step(state, .050)
        for i in range(1, 481):
            result = self.step(state, .050 - i / 120 * .005)
        return result

    def test_real_forward_progress_can_continue_after_legacy_budget(self):
        result = self.earn(FrontProgressCredit())
        self.assertTrue(result["continuation_allowed"])
        self.assertAlmostEqual(result["credit_m"], .0075)

    def test_stall_cannot_renew_and_credit_is_bounded(self):
        state = FrontProgressCredit()
        self.earn(state)
        for _ in range(721):
            result = self.step(state, .030)
        self.assertFalse(result["continuation_allowed"])
        self.assertEqual(result["credit_m"], 0.)

    def test_repeated_old_space_never_earns_new_credit(self):
        state = FrontProgressCredit()
        self.earn(state)
        earned = 0.
        for i in range(1200):
            result = self.step(state, .030 + .003 * (i % 2))
            earned += result["earned_progress_m"]
        self.assertAlmostEqual(earned, 0., places=14)
        self.assertFalse(result["continuation_allowed"])

    def test_wheel_motion_or_near_air_without_fr_progress_earns_nothing(self):
        state = FrontProgressCredit()
        for _ in range(1000):
            result = self.step(state, .010)
        self.assertEqual(result["credit_m"], 0.)

    def test_backward_motion_earns_nothing(self):
        state = FrontProgressCredit()
        for i in range(1000):
            result = self.step(state, .010 + i * .00001)
        self.assertFalse(result["continuation_allowed"])

    def test_small_noise_or_subfloor_speed_cannot_live_forever(self):
        state = FrontProgressCredit()
        self.earn(state)
        for i in range(1200):
            result = self.step(state, .030 - i / 120 * .0001)
        self.assertEqual(result["credit_m"], 0.)

    def test_ineligible_ground_old_lift_clearance_or_support_cannot_extend(self):
        # Integration must independently exercise each real evaluator gate;
        # this kernel intentionally does not invent support/contact evidence.
        state = FrontProgressCredit()
        self.earn(state)
        result = self.step(state, .029, eligible=False)
        self.assertFalse(result["continuation_allowed"])
        self.assertEqual(result["earned_progress_m"], 0.)

    def test_ineligible_progress_cannot_be_recredited_after_return(self):
        state = FrontProgressCredit()
        self.step(state, .05, eligible=False)
        self.step(state, .02, eligible=False)
        result = self.step(state, .02, eligible=True)
        self.assertEqual(result["credit_m"], 0.)

    def test_safety_and_global_deadline_override_credit(self):
        for kwargs in ({"abort": True}, {"age": 200.0}):
            state = FrontProgressCredit()
            self.earn(state)
            self.assertFalse(self.step(state, .029, **kwargs)["continuation_allowed"])

    def test_no_phase_completion_or_contact_is_awarded(self):
        state = FrontProgressCredit()
        self.earn(state)
        result = self.step(state, 0.)
        self.assertNotIn("placed_FR", result)
        self.assertNotIn("stage_id", result)
        self.assertNotIn("success", result)

    def test_phase_exit_resets_only_this_credit_not_action_or_gae(self):
        state = FrontProgressCredit()
        self.earn(state)
        result = self.step(state, .01, p02=False)
        self.assertEqual(result["observation"], (0., 0., 0.))
        self.assertIsNone(state.best_remaining_m)

    def test_public_state_and_nonfinite_rejection(self):
        result = self.earn(FrontProgressCredit())
        self.assertEqual(len(result["observation"]), 3)
        self.assertAlmostEqual(result["observation"][0], .03)
        self.assertEqual(result["observation"][1:], (1., 1.))
        with self.assertRaises(ValueError):
            self.step(FrontProgressCredit(), float("nan"))


if __name__ == "__main__":
    unittest.main()
