"""Independent stdlib tests of root's production pure module, no physical credit.

No monkeypatch of production and no model imports. Safety/data-contract tests
are ordinary assertions: failures expose draft gaps rather than hiding them.
"""
import copy
import math
from pathlib import Path
import sys
import types
import unittest

from wlr50_clean.ppo import semantic_cooperative_preparation as candidate


def observation():
    return dict(physics_tick=120, simulation_time_s=1.,
        joints={'front_left_knee': {'position_deg': -45.}},
        body_bounds_w_m={'rear_left_wheel': {
            'minimum_m': [.1, -.05, .06], 'maximum_m': [.2, .05, .16]}},
        geometry_pose_aware=True,
        obstacle=dict(front_x_m=0., back_x_m=2., right_y_m=-1., left_y_m=1.,
                      bottom_z_m=0., top_z_m=.05))


def evaluation():
    return dict(physics_tick=120, simulation_time_s=1., valid=True,
        termination_reason=None,
        history={'active_lift': {'RR': True}, 'front_edge_crossed': {'RR': True},
                 'placed': {'RR': False, 'RL': False}},
        current_legs={
            'RR': dict(current_lift_valid=True, air=True, ground_contact=False,
                       within_top_xy=True, within_lateral_span=True,
                       clearance_m=.04, front_distance_m=.2),
            'RL': dict(current_lift_valid=False, air=False, ground_contact=True,
                       motion_continuation_allowed=True),
            'FL': dict(air=False, ground_contact=False, top_surface_contact=True,
                       top_contact=True, obstacle_pair_active=True, support=True,
                       bearing_verified=True, bearing_force_n=2.)})


def measure(obs=None, ev=None, context=None):
    return candidate.measure_preparation(observation=obs or observation(),
        evaluation=ev or evaluation(),
        rr_context=context or dict(rr_top_reachable=True, rr_current_bearing=False),
        support_spec={'force_noise_floor_n': .2}, joint_limits=(-60., 210.),
        clearance_scale_m=.02)


def frames():
    ev = evaluation(); diag = measure(ev=ev)
    current = types.SimpleNamespace(task={'stage_id': 'P09',
        'physical_evaluator': ev, 'cooperative_preparation': diag},
        groups={'actual_wheel_velocity_rad_s': [-.6, .3, .3, .3]},
        metrics={'sim_time_s': 1., 'wheels': [{'center': [.2, 0., .1]} for _ in range(4)]})
    previous = copy.deepcopy(current)
    previous.metrics['sim_time_s'] = 1.-1/120
    previous.task['physical_evaluator']['physics_tick'] = 119
    previous.task['physical_evaluator']['simulation_time_s'] = 1.-1/120
    return previous, current


def counter(previous=None, current=None, nominal=.3, final=-.6, dt=1/120):
    before, after = frames()
    n, d, cap = [0.]*12, [0.]*12, [1.]*12
    n[8], d[8], cap[8] = nominal, final, 1.2
    return candidate.fl_counterroll_sample(previous=previous or before,
        current=current or after, nominal=n, actual_drive=d, dt_s=dt,
        residual_caps=cap, force_noise_floor_n=.2)


class GeometryAndWorkspaceTests(unittest.TestCase):
    def test_actual_raw_bounds_schema_and_gap(self):
        d = measure()
        self.assertTrue(d['relevant'])
        self.assertAlmostEqual(d['rl_wheel_clearance_lower_bound_m'], .01)
        self.assertAlmostEqual(d['rl_space_credit'], .5)
        self.assertFalse(d['rl_whole_linkage_clearance_verified'])
        self.assertIsNone(d['rl_whole_linkage_clearance_m'])
        self.assertFalse(d['air_is_support'])
    def test_AABB_overlap_zero_is_not_task_failure(self):
        o = observation(); o['body_bounds_w_m']['rear_left_wheel']['minimum_m'][2] = .04
        d = measure(obs=o)
        self.assertEqual(d['rl_space_credit'], 0.)
        self.assertTrue(d['valid'])
        self.assertTrue(d['strong_transfer_permission_unchanged'])
    def test_AABB_diagonal_distance(self):
        o = observation(); b = {'minimum_m': [-.04, -.05, .08], 'maximum_m': [-.03, .05, .2]}
        self.assertAlmostEqual(candidate.aabb_clearance(b, o['obstacle']), math.sqrt(2)*.03)
    def test_missing_bounds_not_zero(self):
        o = observation(); o['body_bounds_w_m'].clear()
        with self.assertRaises(ValueError): measure(obs=o)
    def test_unverified_pose_rejected(self):
        o = observation(); o['geometry_pose_aware'] = False
        with self.assertRaises(ValueError): measure(obs=o)
    def test_nonfinite_or_reversed_bounds_rejected(self):
        for x in [float('nan'), .3]:
            o = observation(); o['body_bounds_w_m']['rear_left_wheel']['minimum_m'][0] = x
            with self.assertRaises(ValueError): measure(obs=o)
    def test_clock_disagreement_rejected(self):
        e = evaluation(); e['physics_tick'] -= 1
        with self.assertRaises(ValueError): measure(ev=e)
    def test_no_angle_target_large_interior_plateau(self):
        for knee in [-45., 0., 60., 180.]:
            o = observation(); o['joints']['front_left_knee']['position_deg'] = knee
            self.assertEqual(measure(obs=o)['fl_range_credit'], 1.)
    def test_out_of_bound_margin_is_zero_not_new_safety_termination(self):
        o = observation(); o['joints']['front_left_knee']['position_deg'] = -60.01
        self.assertEqual(measure(obs=o)['fl_range_credit'], 0.)
    def test_relevance_requires_current_geometry_and_real_history(self):
        for name in ['ground_contact', 'within_top_xy', 'within_lateral_span']:
            e = evaluation(); e['current_legs']['RR'][name] = name == 'ground_contact'
            self.assertFalse(measure(ev=e)['relevant'])
        e = evaluation(); e['history']['front_edge_crossed']['RR'] = False
        self.assertFalse(measure(ev=e)['relevant'])
    def test_termination_cannot_be_relevant(self):
        e = evaluation(); e['termination_reason'] = 'BODY_COLLISION'
        self.assertFalse(measure(ev=e)['relevant'])
    def test_legal_current_RL_swing_retires_range_and_space(self):
        e = evaluation(); e['current_legs']['RL'].update(
            current_lift_valid=True, air=True, ground_contact=False)
        d = measure(ev=e)
        self.assertTrue(d['range_space_credit_retired_after_live_swing'])
        self.assertEqual((d['fl_range_credit'], d['rl_space_credit']), (1., 1.))
    def test_RL_ground_or_unqualified_AIR_does_not_retire(self):
        for patch in [dict(current_lift_valid=True, air=False, ground_contact=True),
                      dict(current_lift_valid=False, air=True, ground_contact=False),
                      dict(current_lift_valid=True, air=True, ground_contact=False, motion_continuation_allowed=False)]:
            e = evaluation(); e['current_legs']['RL'].update(patch)
            self.assertFalse(measure(ev=e)['range_space_credit_retired_after_live_swing'])
    def test_workspace_weights_bounded_and_legacy_when_irrelevant(self):
        d = measure()
        self.assertAlmostEqual(sum(d['weights'].values()), 1.)
        self.assertAlmostEqual(candidate.workspace_progress(.6, .4, d), .5375)
        self.assertAlmostEqual(candidate.workspace_progress(.6, .4, None), .5)
    def test_no_mutation(self):
        o, e = observation(), evaluation(); before = copy.deepcopy((o, e))
        measure(obs=o, ev=e)
        self.assertEqual((o, e), before)


class CounterrollTests(unittest.TestCase):
    def test_true_supported_counterroll_no_progress(self):
        d = counter()
        self.assertTrue(d['eligible']); self.assertAlmostEqual(d['raw_cost'], .5)
        self.assertFalse(d['alters_action'])
        self.assertTrue(d['task_prior_not_proven_causal_traction'])
    def test_source_stop_or_reverse_untouched(self):
        for n in (0., -.3):
            d = counter(nominal=n)
            self.assertFalse(d['eligible']); self.assertEqual(d['raw_cost'], 0.)
    def test_actual_forward_target_or_motion_does_not_get_reverse_cost(self):
        self.assertEqual(counter(final=.1)['raw_cost'], 0.)
        a, b = frames(); b.groups['actual_wheel_velocity_rad_s'][0] = .1
        self.assertEqual(counter(a, b)['raw_cost'], 0.)
    def test_FL_air_unverified_or_low_force_excluded(self):
        for patch in [dict(air=True), dict(bearing_verified=False), dict(support=False),
                      dict(bearing_force_n=.199), dict(top_surface_contact=False, ground_contact=False)]:
            a, b = frames(); b.task['physical_evaluator']['current_legs']['FL'].update(patch)
            self.assertFalse(counter(a, b)['eligible'])
    def test_RR_bearing_or_not_qualified_excluded(self):
        a, b = frames(); b.task['cooperative_preparation']['current_rr_bearing'] = True
        self.assertFalse(counter(a, b)['eligible'])
        a, b = frames(); b.task['physical_evaluator']['current_legs']['RR']['current_lift_valid'] = False
        self.assertFalse(counter(a, b)['eligible'])
    def test_RL_swing_excluded(self):
        a, b = frames(); b.task['cooperative_preparation']['rl_actual_swing'] = True
        self.assertFalse(counter(a, b)['eligible'])
    def test_real_forward_progress_retires_cost(self):
        a, b = frames(); b.metrics['wheels'][3]['center'][0] += .3*.04998999834060672/120
        self.assertAlmostEqual(counter(a, b)['raw_cost'], 0., places=10)
    def test_legal_measured_gap_descent_retires_cost(self):
        a, b = frames()
        b.task['physical_evaluator']['current_legs']['RR']['clearance_m'] -= .3*.04998999834060672/120
        self.assertAlmostEqual(counter(a, b)['raw_cost'], 0., places=10)
    def test_center_descent_alone_not_capture_progress(self):
        a, b = frames(); b.metrics['wheels'][3]['center'][2] -= .001
        self.assertAlmostEqual(counter(a, b)['raw_cost'], .5)
    def test_descent_outside_top_XY_not_credited(self):
        a, b = frames(); b.task['physical_evaluator']['current_legs']['RR']['clearance_m'] -= .001
        b.task['physical_evaluator']['current_legs']['RR']['within_top_xy'] = False
        self.assertAlmostEqual(counter(a, b)['raw_cost'], .5)
    def test_previous_outside_XY_lateral_or_ground_not_legal_descent(self):
        for field, value in [('within_top_xy', False), ('within_lateral_span', False), ('ground_contact', True)]:
            a, b = frames()
            a.task['physical_evaluator']['current_legs']['RR'][field] = value
            b.task['physical_evaluator']['current_legs']['RR']['clearance_m'] -= .001
            self.assertAlmostEqual(counter(a, b)['raw_cost'], .5)
    def test_deeper_negative_gap_not_progress(self):
        a, b = frames()
        a.task['physical_evaluator']['current_legs']['RR']['clearance_m'] = -.001
        b.task['physical_evaluator']['current_legs']['RR']['clearance_m'] = -.002
        self.assertAlmostEqual(counter(a, b)['raw_cost'], .5)
    def test_nonfinite_measured_gap_rejected(self):
        a, b = frames(); b.task['physical_evaluator']['current_legs']['RR']['clearance_m'] = float('nan')
        with self.assertRaises(ValueError): counter(a, b)
    def test_nonadjacent_supplied_interval_rejected(self):
        with self.assertRaises(ValueError): counter(dt=.1)
    def test_missing_center_rejected_not_zero(self):
        a, b = frames(); del b.metrics['wheels'][3]['center']
        with self.assertRaises((ValueError, KeyError)): counter(a, b)
    def test_nonfinite_center_rejected_not_zero_progress(self):
        a, b = frames(); b.metrics['wheels'][3]['center'][0] = float('nan')
        with self.assertRaises(ValueError): counter(a, b)
    def test_nonfinite_final_target_rejected_not_zero_counterroll(self):
        with self.assertRaises(ValueError): counter(final=float('nan'))
    def test_actual_observation_time_must_match_supplied_dt(self):
        a, b = frames(); a.metrics['sim_time_s'] = .5
        a.task['physical_evaluator']['simulation_time_s'] = .5
        a.task['physical_evaluator']['physics_tick'] = 60
        with self.assertRaises(ValueError): counter(a, b)
    def test_front_phase_not_penalized_even_if_stale_relevance_supplied(self):
        a, b = frames(); b.task['stage_id'] = 'P02'
        self.assertFalse(counter(a, b)['eligible'])


if __name__ == '__main__':
    unittest.main()
