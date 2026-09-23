"""Signed geometry permission is not contact. Fixtures/replay are not physics."""
from copy import deepcopy

import pytest

from test_semantic_rr_capture_assist import DT, step, top
from test_semantic_rr_capture_context import measured, facts
from test_semantic_rr_contact_onset import near_budget
from test_semantic_rr_contact_handoff import ContactTrace
from test_semantic_rr_capture_supervisor import airborne_rr
from test_semantic_rr_carry_wheel import SPEC
from wlr50_clean.ppo.semantic_rr_capture_assist import (
    apply_rr_capture_assist_snapshot, validate_rr_capture_assist_snapshot,
)
from wlr50_clean.ppo.semantic_rr_capture_context import rr_contact_handoff_window_s


def live_credit():
    assist = near_budget(peak=.000138727)
    assist.state.update(mode=6., blocked_reason=0., travel_used_deg=52.041666666665,
                        descent_elapsed_s=42.041666666665, window_elapsed_s=.208333333333)
    return assist


@pytest.mark.parametrize('gap', [2.610613e-6, -3.537004e-6, -.0003, -.0149, -.015])
def test_signed_air_preserves_permission_without_top_bearing_or_placement(gap):
    task, obs = measured()
    task['physical_evaluator']['current_legs']['RR']['clearance_m'] = gap
    assist = live_credit()
    result = facts(task, obs, assist_snapshot=assist.snapshot())
    assert result['rr_top_reachable'] and result['rr_capture_recovery_allowed']
    assert not result['rr_top_contact'] and not result['rr_current_bearing']
    assert not result['rr_history_placed'] and not result['rl_transfer_ready']
    before = deepcopy(assist.state)
    step(assist, 2, gap_m=gap)
    assert assist.snapshot()['mode_name'] == 'DESCEND_PROGRESS'
    assert assist.state['travel_used_deg'] == pytest.approx(before['travel_used_deg'] + DT)
    assert assist.state['descent_elapsed_s'] == pytest.approx(before['descent_elapsed_s'] + DT)
    assert assist.state['contact_seen'] == 0.
    command = apply_rr_capture_assist_snapshot((0.,) * 12, assist.snapshot())
    assert command[7] == pytest.approx(before['knee_hold_deg'] + DT)
    assert command[:6] == (0.,) * 6 and command[8:] == (0.,) * 4


def test_negative_public_peak_is_valid_but_does_not_refresh_absolute_budgets():
    assist = live_credit()
    step(assist, 2, gap_m=-.0003)  # genuine >0.2mm progress resets local window only
    assert assist.state['window_start_gap_m'] == -.0003
    assert assist.state['travel_used_deg'] > 52.
    validate_rr_capture_assist_snapshot(assist.snapshot())
    for tick in range(3, 140):
        step(assist, tick, gap_m=-.0003 - tick * .000004)
    assert assist.state['travel_used_deg'] == pytest.approx(53.)
    assert assist.state['descent_elapsed_s'] == pytest.approx(43.)
    assert assist.snapshot()['mode_name'] == 'BLOCKED'
    assert assist.state['contact_seen'] == 0.


@pytest.mark.parametrize('change', [dict(gap_m=-.015001), dict(within_top_xy=False),
    dict(other_support_count=1), dict(qualified_RR=False), dict(crossed_RR=False),
    dict(physical_valid=False), dict(hip_actual_deg=10.), dict(knee_actual_deg=10.)])
def test_signed_band_cannot_override_safety_tracking_support_or_qualification(change):
    assist = live_credit(); before = deepcopy(assist.state)
    step(assist, 2, **{'gap_m': -.0003, **change})
    assert assist.snapshot()['mode_name'] == 'BLOCKED'
    assert assist.state['travel_used_deg'] == before['travel_used_deg']
    assert assist.state['knee_hold_deg'] == before['knee_hold_deg']


@pytest.mark.parametrize('gap', [-.015001, -.2])
def test_below_existing_task_band_is_not_a_capture_candidate(gap):
    task, obs = measured()
    task['physical_evaluator']['current_legs']['RR']['clearance_m'] = gap
    result = facts(task, obs, assist_snapshot=live_credit().snapshot())
    assert not result['rr_top_reachable'] and not result['rr_capture_recovery_allowed']


def test_actual_top_stops_signed_descent_and_ambiguous_contact_never_grants_support():
    assist = live_credit(); step(assist, 2, gap_m=-.0003)
    target, travel = assist.state['knee_hold_deg'], assist.state['travel_used_deg']
    step(assist, 3, air=False, obstacle_pair_active=True, gap_m=-.0004)
    assert assist.snapshot()['mode_name'] == 'HOLD'
    assert assist.state['contact_seen'] == 0.
    assert assist.state['knee_hold_deg'] == target and assist.state['travel_used_deg'] == travel
    task, obs = measured()
    task['physical_evaluator']['current_legs']['RR'].update(
        air=False, obstacle_pair_active=True, contact_surface='OBSTACLE_AMBIGUOUS', clearance_m=-.0004)
    result = facts(task, obs, assist_snapshot=assist.snapshot())
    assert not result['rr_top_contact'] and not result['rr_current_bearing']
    assert not result['rr_capture_recovery_allowed']  # No unobserved weak-contact extension in v7.
    step(assist, 4, **top())
    assert assist.state['contact_seen'] == 1. and assist.state['travel_used_deg'] == travel


@pytest.mark.parametrize('gap', [-3.537004e-6, -.0003])
def test_real_supervisor_path_keeps_signed_air_at_expired_local_deadline(airborne_rr, gap):
    trace = ContactTrace(airborne_rr)
    trace.step(local_age=20.)
    # Place fixture's measured wheel bottom in the signed band; exact pairs stay
    # inactive. The normal evaluator and postphysics deadline remain in use.
    from test_semantic_all_stage_physical_acceptance import set_leg
    from test_semantic_supervisor import advance
    obs = advance(trace.obs)
    set_leg(obs, 'RR', x=.53, bottom=.05 + gap, surface='AIR')
    trace.sup.stage_started_s = obs['simulation_time_s'] - 52.533333
    trace.sup.episode_started_s = obs['simulation_time_s'] - 121.666667
    trace.sup.rr_capture_feedback = dict(episode_observation_tick=obs['physics_tick'],
                                       state=live_credit().snapshot())
    result = trace.sup.observe_and_update(obs)
    rr = result['physical_evaluator']['current_legs']['RR']
    assert rr['air'] and not rr['top_contact'] and not result['placed_history']['RR']
    assert result['termination_reason'] is None
    assert result['rr_capture_continuation']['local_warning_only']


@pytest.mark.parametrize('key,value', [('top_gap_min_m', -.02), ('top_gap_max_m', .03)])
def test_runtime_rejects_unmigrated_geometry_band(key, value):
    spec = deepcopy(SPEC); spec['geometry'][key] = value
    with pytest.raises(ValueError, match='frozen task geometry band'):
        rr_contact_handoff_window_s(spec)
