"""Integrated tracking-only tests; stdlib and real source executor."""
import ast
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
from wlr50_clean.reference.motion_contract import load_motion_contract

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

PRODUCTION = ROOT / "src/wlr50_clean/ppo/semantic_rr_capture_deferred_late.py"
candidate = load("integrated_tracking_owner_fix", PRODUCTION)
motion = load("readonly_tracking_test_executor", ROOT / "src/wlr50_clean/fsm/motion_executor.py")
CONTRACT = load_motion_contract(ROOT / "configs/recording_motion_contract.json")
PHASE = CONTRACT.phase("P09")

def executor(phase=PHASE):
    result = motion.MotionExecutor(physics_hz=120., servo_rate_limit_deg_s=150.,
                                  initial_full12=phase.start_full12)
    result.start_phase(phase)
    return result

def carrier(previous_tracking=None):
    source = executor()
    for _ in range(648):
        before = source.tick()
    if previous_tracking is not None:
        before = replace(before, tracking_servo_names=previous_tracking)
    return candidate.DeferredLateCarrier(source, previous_sample=before,
                                         pending_event_tick=648), before

class TrackingOwnerFixTests(unittest.TestCase):
    def test_real_647_empty_and_unconsumed_648_full12_names_not_inherited(self):
        c, before = carrier()
        self.assertEqual(before.tick_index, 647)
        self.assertEqual(before.tracking_servo_names, ())
        baseline = executor()
        for _ in range(649):
            raw = baseline.tick()
        self.assertEqual(set(raw.tracking_servo_names), set(candidate.SERVO_NAMES))
        self.assertEqual(c.tick().tracking_servo_names, ())
        self.assertFalse(c.receipt()["pending_event_consumed"])

    def test_pending_and_after_wheel_stop_keep_actual_prelate_empty_tracking(self):
        c, before = carrier()
        for _ in range(500):
            result = c.tick()
            self.assertEqual(result.tracking_servo_names, before.tracking_servo_names)
            for index in candidate.DEFERRED_INDICES:
                self.assertEqual(result.full12[index], before.full12[index])
            self.assertFalse(result.endpoint_issued)
        self.assertEqual(c.stop_event_ticks, [864])
        self.assertEqual(c.receipt()["deferred_source_tracking_servo_names"], [])

    def test_original_stop_dispatch_is_exactly_once_and_has_only_wheel_owners(self):
        c, _ = carrier()
        emitted = []
        for _ in range(500):
            sample = c.tick()
            if sample.atomic_groups:
                emitted.append(sample)
        self.assertEqual([x.tick_index for x in emitted], [864])
        self.assertEqual(set(emitted[0].atomic_groups[0].channels), set(candidate.WHEEL_NAMES))
        self.assertEqual(emitted[0].full12[8:], (0.,)*4)
        self.assertEqual(emitted[0].tracking_servo_names, ())

    def test_prior_legal_tracking_is_preserved_not_global_servo_disable(self):
        # Separate ownership unit case; not a claim that actual sample647 has
        # these names. The wrapper preserves input responsibility, not zero().
        existing = ("rear_right_knee",)
        c, _ = carrier(existing)
        for _ in range(250):
            self.assertEqual(c.tick().tracking_servo_names, existing)

    def test_independent_p10_p11_source_layers_are_not_modified(self):
        c, _ = carrier()
        live = [executor(CONTRACT.phase(p)) for p in ("P10", "P11")]
        controls = [executor(CONTRACT.phase(p)) for p in ("P10", "P11")]
        changed = [False, False]
        for _ in range(250):
            c.tick()
            for i, (a, b) in enumerate(zip(live, controls)):
                x, y = a.tick(), b.tick()
                self.assertEqual(x, y)
                joint = 7 if i == 0 else 2
                changed[i] |= x.full12[joint] != a.phase.start_full12[joint]
        self.assertEqual(changed, [True, True])

    def test_policy_history_and_dispatch_authority_unchanged(self):
        c, _ = carrier()
        receipt = c.receipt()
        self.assertEqual(receipt["policy_channels_restricted"], [])
        self.assertEqual(receipt["mapper_or_history_resets"], 0)
        self.assertEqual(receipt["extra_actuator_writes"], 0)
        tree = ast.parse(PRODUCTION.read_text())
        calls = {n.func.attr for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        self.assertFalse(calls & {"set_joint_position_target", "set_joint_velocity_target",
                                 "project_tick", "backward", "zero_grad"})
        imports = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        imports += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        self.assertFalse(any(n.split(".")[0] in ("torch", "pxr", "isaaclab", "numpy") for n in imports))

if __name__ == "__main__":
    unittest.main(verbosity=2)
