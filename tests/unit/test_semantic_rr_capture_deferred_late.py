"""Pure wiring tests plus a YAML-backed factory test; no Torch or simulator."""
from dataclasses import replace
import ast
import importlib.util
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wlr50_clean.ppo.semantic_rr_capture_deferred_late import (DeferredLateCarrier,
    DEFERRED_INDICES, DEFERRED_SERVOS, WHEEL_NAMES, SUPPORTED_CARRY_MODE,
    controller_factory, provider_type, public_active)
from wlr50_clean.reference.motion_contract import load_motion_contract

# Load the pure executor file directly: importing fsm/__init__ would pull its
# unrelated YAML parser. The tested file itself only imports stdlib modules and
# the standard-library-only contract/constants modules.
SPEC = importlib.util.spec_from_file_location("deferred_pure_motion_executor",
    ROOT / "src/wlr50_clean/fsm/motion_executor.py")
motion_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = motion_module
SPEC.loader.exec_module(motion_module)
MotionExecutor = motion_module.MotionExecutor
CONTRACT = load_motion_contract(ROOT / "configs/recording_motion_contract.json")
PHASE = CONTRACT.phase("P09")
LATE_TIME = next(g.time_s for g in PHASE.atomic_groups if g.source_full12_atomic)


def pending(phase=PHASE):
    motion = MotionExecutor(physics_hz=120., servo_rate_limit_deg_s=150.,
                            initial_full12=phase.start_full12)
    motion.start_phase(phase)
    tick = motion._scaled_source_tick(LATE_TIME)
    before = None
    for _ in range(tick):
        before = motion.tick()
    return motion, before, tick


class BaseProvider:
    def __init__(self):
        self._p09_late_source = (5.333333333334394, LATE_TIME)
        self._rr_carry_source_mode = None
        self._rear_policy_timing_mode = "rr_live_swing_evidence_v3"
        self.spec = {"support": {"force_noise_floor_n": .2}}
        self._continuous_layers = []
        self.calls = []

    def _sequence_permission(self, layer, task, observation):
        self.calls.append((layer["stage"], observation))
        return task.get("base_permission", False)

    def _rr_waiting_late_group(self, layer):
        return False

    @property
    def nominal_suggestion_diagnostics(self):
        return {"base_unmodified": True}


def make_provider(active=True):
    state = {"active": active}
    provider = provider_type(BaseProvider)(read_local_active=lambda: state["active"])
    motion, sample, tick = pending()
    layer = dict(stage="P09", motion=motion, sample=sample, ticks=tick,
                 sequence_diagnostic={})
    provider._continuous_layers = [layer]
    task = {"physical_evaluator": {"valid": True, "physics_tick": 9000}}
    return provider, state, layer, task


class DeferredSourceTests(unittest.TestCase):
    def test_actual_source_changes_exact_five_not_rr_or_fr(self):
        motion, before, tick = pending()
        original = motion.tick()
        changed = {i for i in range(12) if before.full12[i] != original.full12[i]}
        self.assertEqual(changed, set(DEFERRED_INDICES))
        self.assertEqual(tick, 648)
        self.assertEqual(before.full12[8:], (0.,) * 4)
        self.assertEqual(original.full12[6:8], before.full12[6:8])
        self.assertEqual(original.full12[2:4], before.full12[2:4])

    def test_pending_never_acquires_five_channel_goals_or_tracking(self):
        motion, before, tick = pending()
        control, _, _ = pending()
        carrier = DeferredLateCarrier(motion, previous_sample=before, pending_event_tick=tick)
        for _ in range(240):
            actual, baseline = carrier.tick(), control.tick()
            for i in range(12):
                self.assertEqual(actual.full12[i], before.full12[i] if i in DEFERRED_INDICES
                                 else baseline.full12[i])
            self.assertFalse(DEFERRED_SERVOS.intersection(actual.tracking_servo_names))
            self.assertFalse(actual.endpoint_issued)
        self.assertFalse(carrier.receipt()["pending_event_consumed"])
        self.assertEqual(carrier.pending_event_tick, 648)
        self.assertGreater(carrier.receipt()["carrier_next_tick"], 864)

    def test_original_explicit_stop_delivered_once_not_hidden_as_pending(self):
        motion, before, tick = pending()
        carrier = DeferredLateCarrier(motion, previous_sample=before, pending_event_tick=tick)
        emitted = []
        for _ in range(240):
            sample = carrier.tick()
            if sample.atomic_groups:
                emitted.append(sample)
        self.assertEqual([row.tick_index for row in emitted], [864])
        self.assertEqual(set(emitted[0].atomic_groups[0].channels), set(WHEEL_NAMES))
        self.assertEqual(emitted[0].full12[8:], (0.,) * 4)
        self.assertEqual(carrier.source_atomic_emitted, 0)
        self.assertEqual(carrier.original.source_atomic_emitted, 1)

    def test_no_retrofit_after_event_and_no_stale_previous_sample(self):
        motion, before, tick = pending()
        with self.assertRaisesRegex(ValueError, "preceding"):
            DeferredLateCarrier(motion, previous_sample=replace(before, tick_index=tick-2),
                                pending_event_tick=tick)
        motion.tick()
        with self.assertRaisesRegex(ValueError, "unconsumed"):
            DeferredLateCarrier(motion, previous_sample=before, pending_event_tick=tick)

    def test_nonzero_pre_stop_rejected(self):
        motion, before, tick = pending()
        bad = replace(before, full12=before.full12[:8] + (1., 0., 0., 0.))
        with self.assertRaisesRegex(ValueError, "wheel stop"):
            DeferredLateCarrier(motion, previous_sample=bad, pending_event_tick=tick)

    def test_extra_capture_joint_after_late_rejected_not_silently_blocked(self):
        last = PHASE.waypoints[-1]
        bad = replace(last, changed_channels=("rear_right_knee",))
        phase = replace(PHASE, waypoints=PHASE.waypoints[:-1] + (bad,))
        motion, before, tick = pending(phase)
        with self.assertRaisesRegex(ValueError, "new post-late action"):
            DeferredLateCarrier(motion, previous_sample=before, pending_event_tick=tick)

    def test_mixed_later_stop_joint_group_rejected(self):
        last = PHASE.atomic_groups[-1]
        phase = replace(PHASE, atomic_groups=PHASE.atomic_groups[:-1] +
                        (replace(last, channels=last.channels + ("rear_right_knee",)),))
        motion, before, tick = pending(phase)
        with self.assertRaisesRegex(ValueError, "mixed"):
            DeferredLateCarrier(motion, previous_sample=before, pending_event_tick=tick)


class PermissionTests(unittest.TestCase):
    def test_existing_carry_mode_is_preserved_with_pre_late_events(self):
        class ExistingCarry(BaseProvider):
            def __init__(self):
                super().__init__()
                self._rr_carry_source_mode = SUPPORTED_CARRY_MODE
                self._rr_carry_knee_source_times = (.6666666667, 1.)
                self._rr_carry_roll_source_time = 1.9333333333
        provider = provider_type(ExistingCarry)(read_local_active=lambda: True)
        self.assertEqual(provider._rr_carry_source_mode, SUPPORTED_CARRY_MODE)
        self.assertEqual(provider._rr_carry_knee_source_times, (.6666666667, 1.))
        self.assertEqual(provider._rr_carry_roll_source_time, 1.9333333333)

    def test_unknown_carry_or_missing_late_source_rejected(self):
        for late, mode in ((None, None), ((5.3, 5.4), "unknown_carry")):
            class InvalidCarry(BaseProvider):
                def __init__(self):
                    super().__init__()
                    self._p09_late_source, self._rr_carry_source_mode = late, mode
            with self.assertRaisesRegex(ValueError, "supported carry"):
                provider_type(InvalidCarry)(read_local_active=lambda: True)

    def test_missing_nonfinite_or_post_late_carry_events_rejected(self):
        for knees, roll in (((), 1.9), ((float("nan"),), 1.9), ((5.4,), 1.9),
                            ((.7,), 5.4), ((.7,), None), ((-.1,), 1.9)):
            class InvalidCarry(BaseProvider):
                def __init__(self):
                    super().__init__()
                    self._p09_late_source = (5.3, 5.4)
                    self._rr_carry_source_mode = SUPPORTED_CARRY_MODE
                    self._rr_carry_knee_source_times = knees
                    self._rr_carry_roll_source_time = roll
            with self.assertRaisesRegex(ValueError, "must precede"):
                provider_type(InvalidCarry)(read_local_active=lambda: True)

    def test_inactive_prefix_all_stages_delegate_without_mutation(self):
        for phase in (f"P{i:02d}" for i in range(1, 14)):
            provider, state, layer, task = make_provider(False)
            layer["stage"] = phase
            original = layer["motion"]
            self.assertFalse(provider._sequence_permission(layer, task, "observation"))
            self.assertIs(layer["motion"], original)
            self.assertEqual(provider.calls, [(phase, "observation")])

    def test_rr_carry_pre_late_p10_p11_and_started_p12_delegate(self):
        for phase in ("P09", "P10", "P11", "P12"):
            provider, state, layer, task = make_provider()
            layer["stage"] = phase
            if phase == "P09":
                layer["ticks"] -= 1
            task["base_permission"] = True
            self.assertTrue(provider._sequence_permission(layer, task, None))
            self.assertEqual(len(provider.calls), 1)
            self.assertNotIsInstance(layer["motion"], DeferredLateCarrier)

    def test_active_pending_source_deferred_even_with_instantaneous_hold_one(self):
        provider, state, layer, task = make_provider()
        task.update(capture_hold_progress=1., local_success=True)
        self.assertTrue(provider._sequence_permission(layer, task, None))
        sample = layer["motion"].tick()
        layer.update(sample=sample, ticks=layer["ticks"] + 1)
        self.assertTrue(provider._sequence_permission(layer, task, None))
        self.assertTrue(provider._rr_waiting_late_group(layer))
        self.assertNotIn("late_group_start_tick", layer["sequence_diagnostic"])
        self.assertEqual(layer["motion"].receipt()["pending_event_tick"], 648)

    def test_carrier_fresh_stop_wins_existing_ordered_wheel_owner_rule(self):
        provider, state, layer, task = make_provider()
        provider._sequence_permission(layer, task, None)
        while layer["motion"]._tick_index <= 864:
            layer["sample"] = layer["motion"].tick()
        layer["advanced_this_tick"] = True
        # Exact Boolean structure of production's ordered_wheel_owner clause.
        owner = ((layer["stage"] in ("P07", "P09") and not layer["sample"].endpoint_issued
                  and not provider._rr_waiting_late_group(layer)) or
                 (layer["advanced_this_tick"] and any(set(g.channels).intersection(WHEEL_NAMES)
                  for g in layer["sample"].atomic_groups)))
        self.assertTrue(owner)

    def test_active_after_old_late_started_rejected(self):
        provider, state, layer, task = make_provider()
        layer["motion"].tick()
        layer["ticks"] += 1
        with self.assertRaisesRegex(ValueError, "retrofit"):
            provider._sequence_permission(layer, task, None)

    def test_gate_cannot_unlatch_without_episode_reset(self):
        provider, state, layer, task = make_provider()
        provider._sequence_permission(layer, task, None)
        state["active"] = False
        with self.assertRaisesRegex(ValueError, "unlatch"):
            provider._sequence_permission(layer, task, None)

    def test_invalid_physics_and_safety_remain_base_responsibility(self):
        for condition in ("invalid", "task_terminal", "physical_terminal"):
            provider, state, layer, task = make_provider()
            if condition == "invalid":
                task["physical_evaluator"]["valid"] = False
            elif condition == "task_terminal":
                task["termination_reason"] = "BODY_COLLISION"
            else:
                task["physical_evaluator"]["termination_reason"] = "HARD_JOINT_LIMIT"
            self.assertFalse(provider._sequence_permission(layer, task, None))
            self.assertEqual(len(provider.calls), 1)
            self.assertNotIsInstance(layer["motion"], DeferredLateCarrier)

    def test_public_gate_rejects_coerced_or_missing_boolean(self):
        for value in (None, 0, 1, "true"):
            with self.assertRaises(ValueError):
                public_active(lambda: value)

    def test_no_action_or_history_writing_api_and_no_learning_import(self):
        source = (ROOT / "src/wlr50_clean/ppo/semantic_rr_capture_deferred_late.py").read_text()
        tree = ast.parse(source)
        imported = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        imported += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
        self.assertFalse(any(x.split(".")[0] in {"torch", "pxr", "numpy", "isaaclab"}
                             for x in imported))
        calls = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute)}
        self.assertFalse(calls.intersection({"set_joint_position_target", "set_joint_velocity_target",
            "step_physics", "zero_grad", "backward", "reset", "project_tick"}))

    def test_p12_pending_ground_unload_waits_even_with_current_rr_bearing(self):
        provider, state, layer, task = make_provider()
        layer.update(stage="P12", ticks=0, rl_ticks=0)
        task["base_permission"] = True
        task["physical_evaluator"]["current_legs"] = {
            "RR": dict(ground_contact=False, air=False, top_contact=True,
                top_surface_contact=True, obstacle_pair_active=True, within_top_xy=True,
                contact_surface="TOP", support=True, bearing_verified=True, bearing_force_n=10.),
            "FL": dict(ground_contact=True, air=False, support=True,
                       bearing_verified=True, bearing_force_n=5.),
            "RL": dict(ground_contact=True, air=False, current_lift_valid=False,
                       motion_continuation_allowed=True)}
        self.assertFalse(provider._sequence_permission(layer, task, None))
        self.assertTrue(layer["rl_dependency_wait"])
        self.assertTrue(layer["sequence_diagnostic"]["support_transfer_permitted"])
        self.assertEqual(layer["ticks"], 0)
        self.assertEqual(layer["rl_ticks"], 0)
        self.assertEqual(provider.calls, [])

    def test_p12_pending_existing_qualified_air_uses_actual_dependency_helper(self):
        provider, state, layer, task = make_provider()
        layer.update(stage="P12", ticks=0)
        task["base_permission"] = True
        task["physical_evaluator"]["current_legs"] = {
            "RL": dict(ground_contact=False, air=True, current_lift_valid=True,
                       motion_continuation_allowed=True)}
        self.assertTrue(provider._sequence_permission(layer, task, None))
        self.assertEqual(len(provider.calls), 1)

    def test_p12_pending_old_history_or_weakened_air_never_bypasses_helper(self):
        for change in ({"current_lift_valid": False}, {"motion_continuation_allowed": False},
                       {"air": False}, {"ground_contact": True}):
            provider, state, layer, task = make_provider()
            layer.update(stage="P12", ticks=0)
            task["base_permission"] = True
            rl = dict(ground_contact=False, air=True, current_lift_valid=True,
                      motion_continuation_allowed=True)
            rl.update(change)
            task["physical_evaluator"].update(current_legs={"RL": rl},
                history={"active_lift": {"RL": True}, "placed": {"RR": True, "RL": True}})
            self.assertFalse(provider._sequence_permission(layer, task, None))
            self.assertEqual(provider.calls, [])

    def test_p12_already_started_delegates_existing_clock_stop_and_joint_pause(self):
        for base_permission in (False, True):
            provider, state, layer, task = make_provider()
            layer.update(stage="P12", ticks=65, rl_ticks=60, rl_dependency_wait=True)
            task["base_permission"] = base_permission
            self.assertEqual(provider._sequence_permission(layer, task, None), base_permission)
            self.assertEqual(len(provider.calls), 1)
            self.assertEqual((layer["ticks"], layer["rl_ticks"]), (65, 60))

    def test_p12_pending_before_local_gate_preserves_original_permission(self):
        provider, state, layer, task = make_provider(False)
        layer.update(stage="P12", ticks=0)
        task["base_permission"] = True
        self.assertTrue(provider._sequence_permission(layer, task, None))
        self.assertEqual(len(provider.calls), 1)


@unittest.skipUnless(importlib.util.find_spec("yaml"), "real controller factory requires PyYAML")
class AcceptedControllerFactoryTests(unittest.TestCase):
    def test_real_accepted_factory_preserves_source_configuration_and_carry_times(self):
        from wlr50_clean.ppo.semantic_supervisor import SemanticControllerAdapter
        task_path = ROOT / "configs/ppo_rr_capture_first_cp225280_v1/stage_task_spec.yaml"
        paths = (ROOT / "configs/fsm_states.yaml", ROOT / "configs/recording_motion_contract.json")
        original = SemanticControllerAdapter.from_paths(*paths, task_spec_path=task_path)
        wrapped = controller_factory(task_spec_path=task_path,
            read_local_active=lambda: False)(*paths)
        a, b = original.nominal_provider, wrapped.nominal_provider
        self.assertEqual(a.spec, b.spec)
        self.assertEqual(a.contract, b.contract)
        self.assertEqual(a._reference_fsm_spec, b._reference_fsm_spec)
        self.assertEqual(a._p09_late_source, b._p09_late_source)
        self.assertEqual(a._rr_carry_source_mode, b._rr_carry_source_mode)
        self.assertEqual(b._rr_carry_source_mode, SUPPORTED_CARRY_MODE)
        self.assertEqual(a._rr_carry_knee_source_times, b._rr_carry_knee_source_times)
        self.assertEqual(a._rr_carry_roll_source_time, b._rr_carry_roll_source_time)
        self.assertEqual([round(t*120) for t in b._rr_carry_knee_source_times],
                         [80, 88, 96, 104, 112, 120])
        self.assertEqual(round(b._rr_carry_roll_source_time*120), 232)
        self.assertEqual(round(b._p09_late_source[1]*120), 648)
        # The real FSM normal corrections/time scaling are included here,
        # unlike a stand-in BaseProvider. Stop remains singular after endpoint.
        motion = MotionExecutor(physics_hz=b.physics_hz,
            servo_rate_limit_deg_s=b.servo_rate_limit_deg_s,
            initial_full12=b.contract.phase("P09").start_full12)
        b._start_source_motion(motion, b.contract.phase("P09"))
        late_tick = motion._scaled_source_tick(b._p09_late_source[1])
        for _ in range(late_tick):
            before = motion.tick()
        carrier = DeferredLateCarrier(motion, previous_sample=before, pending_event_tick=late_tick)
        endpoint = motion._endpoint_tick(motion.phase)
        while carrier._tick_index <= endpoint + 240:
            carrier.tick()
        self.assertEqual(carrier.stop_event_ticks, [864])
        self.assertFalse(carrier.receipt()["pending_event_consumed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
