# Adaptation draft only: install under tests/unit after production modules exist.
"""Production-import adaptation: pure-CPU B wiring, no Torch/Isaac imports."""
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

from wlr50_clean.ppo.semantic_fl_forward_task import (FlWindowConfig, FlWindowCore, config_from_sources,
                            read_window, transition_evidence)

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'src/wlr50_clean').is_dir())

CFG = FlWindowConfig(3064)


def frame(tick=6000, *, phase='P06', x=.6, final_fl=-.6, actual_fl=-.6, bearing=False, cursor=1000):
    contacts = {}
    for index, name in enumerate(('front_left_wheel', 'front_right_wheel', 'rear_left_wheel', 'rear_right_wheel')):
        active = bearing if index == 0 else True
        contacts[name] = dict(ground=dict(active=active, pair_verified=active, normal_force_n=3. if active else 0.),
                              obstacle=dict(active=False, pair_verified=True, normal_force_n=0.))
    raw = NS(base=NS(position_w_m=(x, 0., .25)), actual_full12=[0.]*8+[actual_fl, .3, .3, .3], contacts=contacts)
    provider = dict(capture_continuation=dict(source_observation_tick=tick, P06_source_tick=cursor,
        P06_layer_present=True, P06_source_advanced=True, P06_wheel_contribution_enabled=True,
        P06_wheel_gain=1., fl_capture_pending=False),
        p06_rolling_retirement=dict(layer_present=True, wheel_gain=1.),
        p06_wheel_tail=dict(layer_present=True, status='finite_source_before_endpoint', source_endpoint_issued=False))
    ev = dict(valid=True, physics_tick=tick, history=dict(placed=dict(FL=True, FR=True)), termination_reason=None)
    final = [0.]*8+[final_fl, .3, .3, .3]
    ack = dict(physics_tick=tick+179, drive_target_full12=final)
    audit = dict(verified=True, actual_mapping_matches_dispatch=True, setter_dispatch_targets_equal=True,
        physics_tick=tick+179, source_phase_id=phase,
        actual_native_targets=dict(wheel_velocity_rad_s=[-final_fl, .3, -.3, .3]))
    return NS(physics_tick=tick, state_id=phase, nominal_action_full12=[0.]*8+[.3]*4,
        safety_projection=NS(force_wheels_zero=False, residual_enabled=True),
        info=dict(raw_observation=raw, semantic_task=dict(physical_evaluator=ev,
            nominal_provider_diagnostics=provider), atomic_ack=ack, actuator_target_effect_audit=audit,
            drive_target_full12=final))


@dataclass(frozen=True)
class Step:
    observation: tuple
    reward: float
    terminated: bool
    truncated: bool
    info: dict


class FakeCore:
    def __init__(self):
        self.done = True
        self.task = NS(finish_active=False, config=NS(failure_cost=40.))
        self.backend = object()
        self.tick_observer = None
        self.frame = None
        self.original_reward = 0.
        self.next_frame = None

    def reset(self, seed=1001):
        self.done = False
        self.frame = frame()
        return (0.,)*531

    def step(self, raw):
        before = self.frame
        self.frame = self.next_frame or frame(before.physics_tick+1, x=.601)
        self.tick_observer(before, self.frame, None)
        return Step((0.,)*531, self.original_reward, self.done, False,
                    dict(phase_id=before.state_id, termination_reason='BODY_COLLISION' if self.done else None))


class WindowTest(unittest.TestCase):
    def test_source_contract_real_files_only(self):
        root = ROOT
        cfg = config_from_sources(root/'configs/recording_motion_contract.json', root/'configs/fsm_states.yaml')
        self.assertEqual(cfg.p06_stop_source_tick, 3064)
        self.assertEqual(cfg.normal_time_scale, 1.)
        self.assertEqual(len(cfg.source_receipt['contract_sha256']), 64)

    def test_air_and_negative_final_do_not_disable_learning(self):
        s = read_window(frame(), CFG)
        self.assertEqual(s['schema'], 'wlr50_clean.p06_fl_forward_window.v1')
        self.assertTrue(s['active'])
        self.assertFalse(s['current_FL_bearing'])
        self.assertEqual(len(s['observation']), 4)

    def test_owner_stop_pending_recoil_and_atomic_horizon(self):
        cases = []
        a = frame(); a.info['semantic_task']['nominal_provider_diagnostics']['capture_continuation']['P06_source_advanced'] = False; cases.append(a)
        a = frame(); a.info['semantic_task']['nominal_provider_diagnostics']['capture_continuation']['fl_capture_pending'] = True; cases.append(a)
        a = frame(); a.info['semantic_task']['nominal_provider_diagnostics']['p06_wheel_tail']['source_endpoint_issued'] = True; cases.append(a)
        a = frame(); a.nominal_action_full12[8:] = [0.]*4; cases.append(a)
        a = frame(); a.nominal_action_full12[8:] = [-.3]*4; cases.append(a)
        a = frame(); a.safety_projection.force_wheels_zero = True; cases.append(a)
        a = frame(); a.info['semantic_task']['nominal_provider_diagnostics']['p06_rolling_retirement']['wheel_gain'] = .9; cases.append(a)
        cases += [frame(phase='P09'), frame(cursor=3056)]
        for f in cases:
            self.assertFalse(read_window(f, CFG)['active'])
        self.assertTrue(read_window(frame(cursor=3055), CFG)['active'])

    def test_same_tick_owner_and_verified_evaluator_required(self):
        a = frame(); a.info['semantic_task']['nominal_provider_diagnostics']['capture_continuation']['source_observation_tick'] -= 1
        self.assertFalse(read_window(a, CFG)['active'])
        a = frame(); a.info['semantic_task']['physical_evaluator']['valid'] = False
        self.assertFalse(read_window(a, CFG)['active'])

    def test_air_negative_command_not_claimed_loaded_drag(self):
        r = transition_evidence(frame(), frame(6001), CFG, request_active=True)
        self.assertEqual(r['progress_reward'], 0.)
        self.assertGreater(r['negative_FINAL_command_cost'], 0.)
        self.assertEqual(r['nonprogress_loaded_counterdrive_cost'], 0.)
        r = transition_evidence(frame(), frame(6001, bearing=True), CFG, request_active=True)
        self.assertGreater(r['nonprogress_loaded_counterdrive_cost'], 0.)

    def test_positive_final_no_residual_sign_cost_and_no_speed_reward(self):
        a = frame()
        low = transition_evidence(a, frame(6001, final_fl=.1, actual_fl=.1), CFG, request_active=True)
        high = transition_evidence(a, frame(6001, final_fl=3., actual_fl=3.), CFG, request_active=True)
        self.assertEqual(low['local_reward'], 0.)
        self.assertEqual(high['local_reward'], 0.)

    def test_soft_cost_retains_gradient_at_actual_baseline(self):
        a = frame()
        larger = transition_evidence(a, frame(6001, final_fl=-.7, actual_fl=-.7, bearing=True), CFG, request_active=True)
        smaller = transition_evidence(a, frame(6001, final_fl=-.6, actual_fl=-.6, bearing=True), CFG, request_active=True)
        self.assertGreater(larger['negative_FINAL_command_cost'], smaller['negative_FINAL_command_cost'])
        self.assertGreater(larger['nonprogress_loaded_counterdrive_cost'], smaller['nonprogress_loaded_counterdrive_cost'])
        self.assertLess(larger['negative_FINAL_command_cost'], CFG.negative_final_cost_per_s/120.)
        zero = transition_evidence(a, frame(6001, final_fl=0., actual_fl=0., bearing=True), CFG, request_active=True)
        self.assertEqual(zero['negative_FINAL_command_cost'], 0.)
        self.assertEqual(zero['nonprogress_loaded_counterdrive_cost'], 0.)

    def test_signed_body_progress_not_wheel_integral(self):
        a = frame(final_fl=.1)
        b = frame(6001, x=.601, final_fl=.1)
        c = frame(6002, x=.6, final_fl=.1)
        one = transition_evidence(a, b, CFG, request_active=True)
        two = transition_evidence(b, c, CFG, request_active=True)
        self.assertAlmostEqual(one['progress_reward'], .005)
        self.assertAlmostEqual(one['progress_reward']+two['progress_reward'], 0.)

    def test_unverified_contact_or_progress_no_counterdrive_cost(self):
        a, b = frame(), frame(6001, bearing=True)
        b.info['raw_observation'].contacts['front_left_wheel']['ground']['pair_verified'] = False
        self.assertEqual(transition_evidence(a, b, CFG, request_active=True)['nonprogress_loaded_counterdrive_cost'], 0.)
        b = frame(6001, x=.601, bearing=True)
        self.assertEqual(transition_evidence(a, b, CFG, request_active=True)['nonprogress_loaded_counterdrive_cost'], 0.)

    def test_adjacent_ack_audit_and_native_mapping_not_ack_alone(self):
        for corrupt in ('stale', 'native', 'verified'):
            b = frame(6001)
            if corrupt == 'stale': b.info['atomic_ack']['physics_tick'] -= 1
            if corrupt == 'native': b.info['actuator_target_effect_audit']['actual_native_targets']['wheel_velocity_rad_s'][0] = 0.
            if corrupt == 'verified': b.info['actuator_target_effect_audit']['verified'] = False
            r = transition_evidence(frame(), b, CFG, request_active=True)
            self.assertFalse(r['eligible'])
            self.assertEqual(r['local_reward'], 0.)

    def test_request_mask_and_native_retirement_reported_separately(self):
        a = frame(); a.info['semantic_task']['nominal_provider_diagnostics']['p06_rolling_retirement']['wheel_gain'] = .9
        r = transition_evidence(a, frame(6001), CFG, request_active=True)
        self.assertTrue(r['request_active'])
        self.assertFalse(r['native_source_window_active'])
        self.assertEqual(r['local_reward'], 0.)

    def test_core_preserves_control_closure_and_external_recorder(self):
        inner = FakeCore(); core = FlWindowCore(inner, CFG)
        seen = []; core.tick_observer = lambda a, b, p: seen.append(b.physics_tick)
        self.assertTrue(core.done)
        self.assertIs(core.task, inner.task)
        self.assertEqual(len(core.reset()), 535)
        s = core.step([0.]*12)
        self.assertEqual(seen, [6001])
        self.assertTrue(s.info['fl_forward_window']['actor_sample'])
        self.assertFalse(s.terminated)
        inner.next_frame = frame(6002, phase='P09')
        core.step([0.]*12)
        inner.original_reward = 12.345
        inner.next_frame = frame(6003, phase='P10')
        s = core.step([0.]*12)
        self.assertEqual(s.reward, 12.345)
        self.assertTrue(s.info['fl_forward_window']['critic_only_sample'])
        self.assertFalse(s.terminated)  # ordinary phase change cannot cut GAE
        inner.done = True
        inner.original_reward = 0.
        inner.next_frame = frame(6004, phase='P10')
        s = core.step([0.]*12)
        self.assertTrue(s.terminated)
        self.assertEqual(s.info['termination_reason'], 'BODY_COLLISION')
        self.assertEqual(s.reward, -40.)
        self.assertEqual(s.info['fl_forward_window']['inherited_pre_finish_terminal_event'], -40.)

    def test_finish_terminal_reward_not_duplicated(self):
        inner = FakeCore(); core = FlWindowCore(inner, CFG)
        core.reset(); core.step([0.]*12)
        inner.task.finish_active = True
        inner.original_reward = -40.
        inner.done = True
        inner.next_frame = frame(6002, phase='P13')
        core.fl_window['active'] = False
        s = core.step([0.]*12)
        self.assertEqual(s.reward, -40.)
        self.assertEqual(s.info['fl_forward_window']['inherited_pre_finish_terminal_event'], 0.)


if __name__ == '__main__':
    unittest.main()
