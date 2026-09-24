"""No Torch/Isaac/actuator: real PURE projector + private namespace wrapper tests."""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "src/wlr50_clean").is_dir())
sys.path.insert(0, str(ROOT / "src"))
from wlr50_clean.ppo.semantic_headroom import project_semantic_servo_headroom as original
from isolated_headroom_wrapper import IsolatedHeadroomProbe
from test_target_plan import context


class IsolatedWrapperTests(unittest.TestCase):
    def make(self):
        tick = {"value": 99}
        wrapper = IsolatedHeadroomProbe(original, read_committed_adapter_tick=lambda: tick["value"],
                                       diagnostic_only=True)
        c = context()
        c["residual_caps_full12"][6:8] = [24., 36.]
        wrapper.arm_from_observer(c)
        return wrapper, tick, c

    def args(self, c, request=None):
        return dict(native_full12=tuple(c["same_tick_mapped_nominal_full12"]),
            controller_bias_full12=tuple(c["controller_bias_full12"]),
            projected_residual_full12=tuple(c["policy_projected_residual_full12"] if request is None else request))

    def test_real_projector_used_and_audit_actual_recomputes_without_state_advance(self):
        w, tick, c = self.make()
        dispatch = w.project(**self.args(c))
        tick["value"] = 100
        audit = w.project(**self.args(c))
        self.assertEqual(dispatch, audit)
        self.assertEqual(w.plan_calls, 1)
        self.assertEqual(w.planner.last_tick, 100)
        self.assertEqual(audit["candidate_native_target_before_final_slew_full12"][6:8], [5., -20.])
        expected = original(**self.args(c))
        for i in range(12):
            if i not in (6, 7):
                self.assertEqual(audit["effective_combined_post_mapper_bias_full12"][i],
                                 expected["effective_combined_post_mapper_bias_full12"][i])

    def test_zero_policy_counterfactual_keeps_same_exogenous_rr_targets(self):
        w, tick, c = self.make()
        actual = w.project(**self.args(c))
        tick["value"] = 100
        cf = w.project(**self.args(c, (0.,) * 12))
        self.assertEqual(cf["candidate_native_target_before_final_slew_full12"][6:8],
                         actual["candidate_native_target_before_final_slew_full12"][6:8])
        self.assertTrue(all(cf["effective_policy_residual_full12"][i] == 0.
                            for i in range(12) if i not in (6, 7)))
        self.assertEqual(w.plan_calls, 1)

    def test_original_zero_request_has_no_dispatch_counterfactual_ambiguity(self):
        w, tick, c = self.make()
        c["policy_projected_residual_full12"] = [0.] * 12
        actual = w.project(**self.args(c))
        tick["value"] = 100
        self.assertEqual(w.project(**self.args(c)), actual)
        self.assertEqual(w.project(**self.args(c, (0.,) * 12)), actual)
        self.assertEqual(w.plan_calls, 1)

    def test_prefix_or_released_plan_is_exact_original_projector(self):
        w, tick, c = self.make()
        w.pending["context"]["phase"] = "P06"
        self.assertEqual(w.project(**self.args(c)), original(**self.args(c)))
        tick["value"] = 100
        self.assertEqual(w.project(**self.args(c, (0.,) * 12)), original(**self.args(c, (0.,) * 12)))

    def test_new_decision_receipt_can_be_bound_without_advancing_planner(self):
        w, tick, c = self.make()
        raw = tuple(.007 * i for i in range(12))
        w.bind_decision_proposal(raw, -8.)
        self.assertEqual(w.plan_calls, 0)
        receipt = w.project(**self.args(c))["explicit_RR_direction_diagnostic"]
        self.assertEqual(receipt["original_policy_raw_full12"], list(raw))
        self.assertEqual(receipt["original_policy_log_probability"], -8.)
        with self.assertRaises(ValueError):
            w.bind_decision_proposal(raw, -8.)

    def test_audit_cannot_start_probe_or_supply_a_different_mapper(self):
        w, tick, c = self.make()
        tick["value"] = 100
        with self.assertRaises(ValueError):
            w.project(**self.args(c))
        tick["value"] = 99
        w.project(**self.args(c))
        tick["value"] = 100
        bad = self.args(c); bad["native_full12"] = (1.,) * 12
        with self.assertRaises(ValueError):
            w.project(**bad)
        bad = self.args(c, (.7,) * 12)
        with self.assertRaises(ValueError):
            w.project(**bad)

    def test_context_epoch_and_duplicate_prewrite_are_strict(self):
        w, tick, c = self.make()
        first = w.project(**self.args(c))
        self.assertEqual(w.project(**self.args(c)), first)
        self.assertEqual(w.plan_calls, 1)
        with self.assertRaises(ValueError):
            w.project(**self.args(c, (.7,) * 12))
        tick["value"] = 110
        with self.assertRaises(ValueError):
            w.project(**self.args(c))

    def test_scoped_install_restores_even_exception_no_real_module_patched(self):
        w, tick, c = self.make()
        private = SimpleNamespace(project_semantic_servo_headroom=original)
        with self.assertRaises(RuntimeError):
            with w.installed_on(private):
                self.assertIsNot(private.project_semantic_servo_headroom, original)
                private.project_semantic_servo_headroom(**self.args(c))
                raise RuntimeError("synthetic downstream error")
        self.assertIs(private.project_semantic_servo_headroom, original)
        from wlr50_clean.ppo import semantic_headroom
        self.assertIs(semantic_headroom.project_semantic_servo_headroom, original)

    def test_training_context_is_rejected_and_no_torch_loaded(self):
        with self.assertRaises(ValueError):
            IsolatedHeadroomProbe(original, read_committed_adapter_tick=lambda: 0,
                                  diagnostic_only=True, training_allowed=True)
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("isaaclab", sys.modules)


if __name__ == "__main__":
    unittest.main()
