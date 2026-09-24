"""No production/Torch imports: runner syntax and bounded ownership bookkeeping."""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

from diagnostic_runner import source_owners, deterministic_proposal_log_density, ATTRIBUTION


class RunnerContractPureTests(unittest.TestCase):
    def test_artifacts_parse_without_importing_native_audit_or_physics(self):
        for name in ("diagnostic_runner.py", "test_native_audit_draft.py"):
            ast.parse(Path(__file__).with_name(name).read_text(encoding="utf-8"))
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("isaaclab", sys.modules)
        self.assertFalse(ATTRIBUTION["formal_C_or_pure_policy_success"])
        self.assertTrue(ATTRIBUTION["native_actual_readback_audit_required"])

    def test_real_layer_event_identity_stable_until_new_event_or_later_owner(self):
        names = tuple("joint"+str(i) for i in range(8))
        groups = [NS(time_s=0., channels=(names[6], names[7])),
                  NS(time_s=1., channels=(names[7],))]
        motion = NS(phase=NS(atomic_groups=groups), _scaled_source_tick=lambda t: round(t*120))
        first = dict(stage="P09", sample=NS(tick_index=100), touched={6, 7}, motion=motion)
        provider = NS(_continuous_layers=[first])
        initial = source_owners(provider, names)
        first["sample"].tick_index = 101
        self.assertEqual(source_owners(provider, names), initial)
        first["sample"].tick_index = 120
        self.assertEqual(source_owners(provider, names)[0], initial[0])
        self.assertNotEqual(source_owners(provider, names)[1], initial[1])
        later = dict(stage="P10", sample=NS(tick_index=1), touched={7}, motion=motion)
        provider._continuous_layers.append(later)
        self.assertEqual(source_owners(provider, names)[1][1], "P10")
        later["capture_retired_servo_indices"] = {7}
        self.assertEqual(source_owners(provider, names)[1][1], "P09")

    def test_density_uses_original_request_not_injected_target_or_second_forward(self):
        import math
        receipt = dict(selected_raw_full12=[.2]*12, conditional_mean_full12=[.2]*12,
                       effective_sigma_full12=[.3]*12)
        self.assertAlmostEqual(deterministic_proposal_log_density(receipt),
            12*(-math.log(.3)-.5*math.log(2*math.pi)))
        receipt["effective_sigma_full12"][0] = 0.
        with self.assertRaises(ValueError):
            deterministic_proposal_log_density(receipt)


if __name__ == "__main__":
    unittest.main()
