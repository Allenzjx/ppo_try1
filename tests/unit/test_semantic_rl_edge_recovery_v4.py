"""Real TaskEvaluator + pure control math; no Torch/Isaac or physical success.

The unchanged v3 evaluator fixture produces the current, ground-revoked RL
evidence consumed by opt-in v4. The owner-level integration is tested separately;
EDGE permission must not silently open the strong-unload source lane.
"""
from copy import deepcopy

import pytest

from test_semantic_rl_live_swing_evidence_v3 import prefix, rl_lift, setup, step, task
from test_semantic_all_stage_physical_acceptance import set_leg
from test_semantic_supervisor import advance
from wlr50_clean.infrastructure.command_batch import SERVO_ORDER, WHEEL_ORDER, servo_limits_deg
from wlr50_clean.infrastructure.robot_adapter import bounded_drive_feedback_step
from wlr50_clean.ppo.semantic_backend import build_semantic_projector
from wlr50_clean.ppo.semantic_headroom import project_semantic_servo_headroom
from wlr50_clean.ppo.semantic_rear_policy_timing import (
    MODE, RECAPTURE_MODE, LIVE_SWING_MODE, EDGE_RECOVERY_MODE,
    rear_dependency, public_timing,
)
from test_semantic_rl_live_swing_evidence_v3 import CONFIG

ZERO = (0.,) * 12


def edge(surface='WALL'):
    ev, obs = prefix()
    obs, _ = rl_lift(ev, obs)
    obs, _ = step(ev, obs, 'RR', x=.52, bottom=.075)
    obs = advance(obs)
    set_leg(obs, 'RL', x=.498, bottom=.014, surface=surface)
    obs['wheels']['rear_left_ankle']['command_rad_s'] = .3
    return ev, obs, ev.observe(obs)


def dependency(ev, snap, mode=EDGE_RECOVERY_MODE):
    return rear_dependency(task(snap), ev.spec['support'], mode=mode)


@pytest.mark.parametrize('surface', ['WALL', 'AMBIGUOUS'])
def test_same_attempt_edge_not_terminal_or_support_and_has_public_recovery(surface):
    ev, obs, snap = edge(surface)
    before = deepcopy(snap)
    rl = snap['current_legs']['RL']
    assert snap['valid'] and snap['termination_reason'] is None
    assert rl['current_lift_valid'] and rl['active_attempt'] and not rl['air']
    assert rl['obstacle_pair_active'] and rl['contact_reaction']
    assert rl['bearing_force_n'] == 0. and not rl['support']
    assert not snap['history']['front_edge_crossed']['RL']
    assert not snap['history']['placed']['RL']
    dep = dependency(ev, snap)
    assert not dep['rr_current_bearing'] and not dep['support_transfer_permitted']
    assert not dep['rl_current_swing']
    assert dep['rl_edge_recovery_permitted'] and dep['rl_motion_continuation_permitted']
    public = public_timing(task(snap), [], ev.spec['support'], 120., mode=EDGE_RECOVERY_MODE)
    assert public['rl_swing_capture'] and not public['rr_carry_capture']
    assert snap == before  # No contact, placement, force or task result mutation.
    obs, air = step(ev, obs, 'RL', x=.49, bottom=.02)
    assert dependency(ev, air)['rl_current_swing']
    assert not dependency(ev, air)['rl_edge_recovery_permitted']
    assert air['current_legs']['RL']['current_lift_qualified_tick'] == rl['current_lift_qualified_tick']


@pytest.mark.parametrize('old_mode', [None, MODE, RECAPTURE_MODE, LIVE_SWING_MODE])
def test_old_modes_retain_air_only_contact_permission(old_mode):
    ev, _, snap = edge()
    dep = dependency(ev, snap, old_mode)
    assert not dep['rl_current_swing'] and not dep['rl_edge_recovery_permitted']
    assert not dep['rl_motion_continuation_permitted']


def test_top_is_capture_not_edge_and_does_not_fake_air():
    ev, obs = prefix()
    obs, _ = rl_lift(ev, obs)
    obs, snap = step(ev, obs, 'RL', x=.501, bottom=.049, surface='TOP')
    assert snap['current_legs']['RL']['top_surface_contact']
    assert not dependency(ev, snap)['rl_edge_recovery_permitted']
    assert not dependency(ev, snap)['rl_current_swing']


@pytest.mark.parametrize('placed_before_ground', [False, True])
def test_ground_revokes_current_edge_even_with_old_history_then_fresh_work_restores(placed_before_ground):
    ev, obs = prefix()
    obs, _ = rl_lift(ev, obs)
    if placed_before_ground:
        for _ in range(2):
            obs, _ = step(ev, obs, 'RL', x=.501, bottom=.049, surface='TOP')
    old = deepcopy(ev.snapshot['history'])
    obs, ground = step(ev, obs, 'RL', x=.49, bottom=0., surface='GROUND')
    assert not dependency(ev, ground)['rl_motion_continuation_permitted']
    assert ground['current_legs']['RL']['current_lift_qualified_tick'] is None
    assert ground['history']['placed'] == old['placed']
    assert ground['history']['event_ticks'] == old['event_ticks']
    for surface in ('AIR', 'WALL', 'AMBIGUOUS'):
        obs, stale = step(ev, obs, 'RL', x=.49, bottom=.001, surface=surface)
        assert not stale['current_legs']['RL']['current_lift_valid']
        assert not dependency(ev, stale)['rl_motion_continuation_permitted']
    obs, _ = step(ev, obs, 'RL', x=.49, bottom=.004, hip=4.5)
    obs, fresh = step(ev, obs, 'RL', x=.49, bottom=.014, hip=6.)
    assert fresh['current_legs']['RL']['current_lift_valid']
    assert fresh['current_legs']['RL']['current_lift_qualified_tick'] > old['event_ticks']['active_lift']['RL']
    obs, recovered = step(ev, obs, 'RL', x=.498, bottom=.015, surface='WALL')
    assert dependency(ev, recovered)['rl_edge_recovery_permitted']


@pytest.mark.parametrize('field,value', [
    ('active_attempt', False), ('current_lift_valid', False), ('motion_continuation_allowed', False),
    ('current_lift_qualified_tick', None), ('current_lift_qualified_tick', True),
    ('current_lift_qualified_tick', 1000000), ('ground_contact', True),
    ('obstacle_pair_active', False), ('contact_reaction', False),
    ('contact_reaction_force_n', float('nan')), ('contact_reaction_force_n', -1.),
    ('contact_mode', 'GROUND_AND_OBSTACLE'), ('contact_surface', 'TOP'),
])
def test_missing_or_contradictory_current_edge_evidence_fails_closed(field, value):
    ev, _, snap = edge()
    snap['current_legs']['RL'][field] = value
    assert not dependency(ev, snap)['rl_edge_recovery_permitted']


@pytest.mark.parametrize('fault', ['body', 'numerical', 'sensor'])
def test_real_safety_or_invalid_sensor_overrides_recovery(fault):
    ev, obs, _ = edge()
    obs = advance(obs)
    if fault == 'body': obs['body_collision']['detected'] = True
    elif fault == 'numerical': obs['all_finite'] = False
    else: obs['contacts']['rear_left_wheel']['obstacle'].pop('force_w_n')
    snap = ev.observe(obs)
    assert snap['termination_reason'] is not None
    assert not dependency(ev, snap)['rl_motion_continuation_permitted']


def test_single_driven_wall_contact_is_not_failure_but_unqualified_powered_ascent_remains_failure():
    ev, obs = setup()
    for index, (x, bottom) in enumerate(((.47,.01),(.48,.025),(.501,.049))):
        obs = advance(obs)
        set_leg(obs, 'RL', x=x, bottom=bottom, surface='WALL')
        obs['wheels']['rear_left_ankle']['command_rad_s'] = .3
        snap = ev.observe(obs)
        assert not dependency(ev, snap)['rl_edge_recovery_permitted']
        if index == 0:
            assert snap['termination_reason'] is None
    assert snap['termination_reason'] == 'TASK_FAILURE_WHEEL_ONLY_CLIMB'
    assert snap['termination_source'] == 'WHEEL_ONLY_PROCESS'


def test_edge_recovery_retains_knee_and_backward_wheel_residual_target_path_without_torch():
    ev, _, snap = edge()
    assert dependency(ev, snap)['rl_edge_recovery_permitted']
    assert not dependency(ev, snap)['support_transfer_permitted']
    projector = build_semantic_projector(CONFIG/'execution_profile.yaml')
    raw = [0.] * 12
    raw[5] = .02  # Candidate positive RL knee adjustment, not a fixed controller.
    raw[8:] = [-.02] * 4  # A bounded reverse residual remains expressible.
    projected = projector.project(raw, state_id='P12', nominal_action_full12=ZERO,
        reference_action_full12=ZERO, reference_delta_full12=ZERO,
        previous_projected_residual_full12=ZERO)
    residual = projected.safe_projected_residual_full12
    assert residual[5] > 0 and all(value < 0 for value in residual[8:])
    headroom = project_semantic_servo_headroom(ZERO, ZERO, residual)
    candidate = headroom['candidate_native_target_before_final_slew_full12']
    lo, hi = servo_limits_deg(SERVO_ORDER[5])
    final_knee = bounded_drive_feedback_step(previous_deg=0., native_deg=0.,
        bias_deg=headroom['effective_combined_post_mapper_bias_full12'][5],
        maximum_delta_deg=1.25, lower_deg=lo, upper_deg=hi)
    assert 0. < final_knee <= 1.25 and candidate[5] > 0.
    assert all(value < 0. for value in candidate[8:])
    # This proves pure projection/final-servo arithmetic paths, not actuator
    # dispatch, real movement, improved clearance or completed recovery.
