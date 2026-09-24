"""Measured proxy v5: no simulator, network, commands or task credit."""
import copy
import math
import unittest

from wlr50_clean.ppo import semantic_cooperative_preparation as p


def inputs():
    return dict(knee_deg=-45., joint_limits=(-60., 210.), servo_reserve_deg=2.,
        fl_action_capacity_deg=36., rl_bounds=dict(minimum_m=[-.3, -.05, 0.],
        maximum_m=[-.2, .05, .1]), obstacle=dict(front_x_m=0., back_x_m=2.,
        right_y_m=-1., left_y_m=1., bottom_z_m=0., top_z_m=.05),
        clearance_scale_m=.02)


def fixture():
    args = inputs()
    observation = dict(physics_tick=120, simulation_time_s=1.,
        joints={'front_left_knee': {'position_deg': -45., 'command_deg': 100.}},
        body_bounds_w_m={'rear_left_wheel': args['rl_bounds']},
        geometry_pose_aware=True, obstacle=args['obstacle'])
    evaluation = dict(physics_tick=120, simulation_time_s=1., valid=True,
        termination_reason=None,
        history={'active_lift': {'RR': True}, 'front_edge_crossed': {'RR': True},
                 'placed': {'RR': False, 'RL': False}},
        current_legs={
            'RR': dict(current_lift_valid=True, air=True, ground_contact=False,
                within_top_xy=True, within_lateral_span=True),
            'RL': dict(current_lift_valid=False, air=False, ground_contact=True,
                motion_continuation_allowed=True)})
    return dict(observation=observation, evaluation=evaluation,
        rr_context=dict(rr_top_reachable=True, rr_current_bearing=False),
        support_spec={'force_noise_floor_n': .2}, joint_limits=(-60., 210.),
        clearance_scale_m=.02, mode=p.TASK_PROXY_MODE,
        fl_action_capacity_deg=36., servo_reserve_deg=2.)


class FiniteMeasuredProxyTests(unittest.TestCase):
    def test_FL_no_early_ten_degree_saturation(self):
        d = p.measured_task_proxies(**inputs())
        self.assertAlmostEqual(d['fl_range_credit'], 13./36.)
        self.assertEqual(d['fl_negative_executable_reserve_deg'], 13.)

    def test_more_current_negative_reserve_is_continuous(self):
        args = inputs(); values = []
        for q in (-58., -50., -45., -30., -22.):
            args['knee_deg'] = q
            values.append(p.measured_task_proxies(**args)['fl_range_credit'])
        self.assertEqual(values, sorted(values))
        self.assertEqual(values[0], 0.)
        self.assertEqual(values[-1], 1.)

    def test_capacity_is_actual_parameter_not_angle_template(self):
        args = inputs(); a = p.measured_task_proxies(**args)
        args['fl_action_capacity_deg'] = 72.
        b = p.measured_task_proxies(**args)
        self.assertEqual(a['fl_range_credit'], b['fl_range_credit'] * 2.)

    def test_protection_margin_not_asset_hard_edge(self):
        args = inputs(); args['knee_deg'] = -58.
        self.assertEqual(p.measured_task_proxies(**args)['fl_range_credit'], 0.)
        args['servo_reserve_deg'] = 0.
        self.assertGreater(p.measured_task_proxies(**args)['fl_range_credit'], 0.)

    def test_positive_escape_bound_prevents_infinite_positive_knee(self):
        args = inputs(); args['knee_deg'] = 209.
        self.assertEqual(p.measured_task_proxies(**args)['fl_range_credit'], 0.)

    def test_outside_actual_hard_range_no_new_abort(self):
        for q in (-61., 211.):
            args = inputs(); args['knee_deg'] = q
            self.assertEqual(p.measured_task_proxies(**args)['fl_range_credit'], 0.)

    def test_RL_far_ground_cannot_score_clearance_by_retreat(self):
        args = inputs()
        for x in (-.3, -10., .4):
            args['rl_bounds']['minimum_m'][0] = x
            args['rl_bounds']['maximum_m'][0] = x + .1
            self.assertEqual(p.measured_task_proxies(**args)['rl_space_credit'], 0.)

    def test_RL_raising_actual_bottom_improves_corridor_not_center_only(self):
        args = inputs(); scores = []
        for z in (0., .02, .04, .06, .07):
            args['rl_bounds']['minimum_m'][2] = z
            args['rl_bounds']['maximum_m'][2] = z + .1
            scores.append(p.measured_task_proxies(**args)['rl_space_credit'])
        self.assertEqual(scores, sorted(scores))
        self.assertEqual(scores[0], 0.)
        self.assertEqual(scores[-1], 1.)
        args['rl_bounds']['maximum_m'][2] += .5
        self.assertEqual(p.measured_task_proxies(**args)['rl_space_credit'], scores[-1])

    def test_RL_outside_lateral_corridor_not_full_credit(self):
        args = inputs(); args['rl_bounds'].update(minimum_m=[0., .99, .07], maximum_m=[.1, 1.04, .17])
        d = p.measured_task_proxies(**args)
        self.assertAlmostEqual(d['rl_top_corridor_lateral_deficit_m'], .04)
        self.assertLess(d['rl_space_credit'], 1.)

    def test_contacting_allowed_edge_is_not_failure_or_negative_bonus(self):
        args = inputs(); args['rl_bounds'].update(minimum_m=[-.05, -.05, .02], maximum_m=[.05, .05, .12])
        d = p.measured_task_proxies(**args)
        self.assertGreater(d['rl_space_credit'], 0.)
        self.assertNotIn('termination_reason', d)
        self.assertNotIn('target', d)

    def test_rigid_world_translation_preserves_score(self):
        args = inputs(); before = p.measured_task_proxies(**args)
        for index, delta, names in ((0, 2., ('front_x_m', 'back_x_m')),
                                    (1, -3., ('right_y_m', 'left_y_m')),
                                    (2, .5, ('bottom_z_m', 'top_z_m'))):
            for bound in ('minimum_m', 'maximum_m'): args['rl_bounds'][bound][index] += delta
            for name in names: args['obstacle'][name] += delta
        after = p.measured_task_proxies(**args)
        self.assertAlmostEqual(before['rl_space_credit'], after['rl_space_credit'])

    def test_missing_nonfinite_invalid_geometry_never_fabricates_zero(self):
        for field, value in (('fl_action_capacity_deg', None), ('fl_action_capacity_deg', 0.),
                             ('servo_reserve_deg', float('nan')), ('knee_deg', float('inf'))):
            args = inputs(); args[field] = value
            with self.assertRaises(ValueError): p.measured_task_proxies(**args)
        args = inputs(); args['rl_bounds']['minimum_m'][0] = math.nan
        with self.assertRaises(ValueError): p.measured_task_proxies(**args)

    def test_no_hidden_accumulator_or_loop_bonus(self):
        args = inputs(); before = copy.deepcopy(args)
        a = p.measured_task_proxies(**args)
        args['rl_bounds']['minimum_m'][2] = .03
        p.measured_task_proxies(**args)
        args = before
        self.assertEqual(a, p.measured_task_proxies(**args))


class VersionedIntegrationTests(unittest.TestCase):
    def test_v4_explicit_and_default_remain_identical(self):
        args = fixture(); args.pop('mode')
        default = p.measure_preparation(**args)
        self.assertEqual(default, p.measure_preparation(**args, mode=p.MODE))
        self.assertEqual(default['fl_range_credit'], 1.)
        self.assertEqual(default['rl_space_credit'], 1.)

    def test_v5_current_actual_not_issued_target_and_no_whole_link_claim(self):
        args = fixture(); d = p.measure_preparation(**args)
        self.assertAlmostEqual(d['fl_range_credit'], 13./36.)
        self.assertEqual(d['rl_space_credit'], 0.)
        self.assertFalse(d['rl_whole_linkage_clearance_verified'])
        self.assertFalse(d['potential_uses_rl_euclidean_obstacle_clearance'])
        self.assertIsNone(d['rl_whole_linkage_clearance_m'])
        self.assertTrue(d['strong_transfer_permission_unchanged'])

    def test_current_qualified_RL_swing_and_placed_retire_proxies(self):
        for placed in (False, True):
            args = fixture()
            if placed: args['evaluation']['history']['placed']['RL'] = True
            else: args['evaluation']['current_legs']['RL'].update(current_lift_valid=True, air=True, ground_contact=False)
            d = p.measure_preparation(**args)
            self.assertEqual((d['fl_range_credit'], d['rl_space_credit']), (1., 1.))

    def test_old_qualification_or_edge_contact_alone_not_retired(self):
        args = fixture(); args['evaluation']['current_legs']['RL'].update(current_lift_valid=True, air=False, ground_contact=False)
        self.assertFalse(p.measure_preparation(**args)['range_space_credit_retired_after_live_swing'])

    def test_irrelevant_front_no_new_geometry_or_capacity_dependency(self):
        args = fixture(); args['evaluation']['history']['front_edge_crossed']['RR'] = False
        args['observation']['body_bounds_w_m'].clear(); args['fl_action_capacity_deg'] = None
        self.assertFalse(p.measure_preparation(**args)['relevant'])

    def test_same_existing_budget_no_extra_task_share(self):
        d = p.measure_preparation(**fixture())
        self.assertEqual(d['weights'], p.WORKSPACE_WEIGHTS)
        self.assertAlmostEqual(sum(d['weights'].values()), 1.)
        self.assertAlmostEqual(p.workspace_progress(.6, .4, d), .25*.6+.5*.4+.125*13/36)
        self.assertEqual(p.TASK_PROXY_REWARD_CONFIG, dict(p.REWARD_CONFIG, mode=p.TASK_PROXY_MODE))

    def test_unknown_version_and_missing_current_capacity_fail_closed(self):
        args = fixture(); args['mode'] = 'unknown'
        with self.assertRaises(ValueError): p.measure_preparation(**args)
        args = fixture(); args['fl_action_capacity_deg'] = None
        with self.assertRaises(ValueError): p.measure_preparation(**args)


if __name__ == '__main__':
    unittest.main()
