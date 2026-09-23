"""Near-plane permission is an action budget, never TOP or task acceptance."""
from copy import deepcopy

import pytest

from test_semantic_rr_capture_assist import DT, step, top
from wlr50_clean.ppo.semantic_rr_capture_assist import RRHipOnlyCaptureAssist


def near_budget(*, peak=.00024, elapsed=.1):
    assist = RRHipOnlyCaptureAssist()
    step(assist, 1, gap_m=.00024)
    assist.state.update(mode=3., blocked_reason=5., travel_used_deg=52.,
                        descent_elapsed_s=42., hip_target_deg=-1.785913,
                        knee_hold_deg=-13.361176, window_start_gap_m=peak,
                        window_elapsed_s=elapsed)
    return assist


def test_fresh_measured_descent_can_execute_last_degree_but_nearness_is_not_contact():
    assist = near_budget()
    step(assist, 2, gap_m=.000027)
    assert assist.snapshot()['mode_name'] == 'DESCEND_PROGRESS'
    assert assist.state['travel_used_deg'] == pytest.approx(52. + DT)
    assert assist.state['contact_seen'] == 0.
    assert assist.state['knee_hold_deg'] == pytest.approx(-13.361176 + DT)


def test_continuous_earned_mode_survives_old_boundary_without_inventing_new_progress():
    assist = near_budget(peak=.000138727)
    assist.state.update(mode=6., blocked_reason=0., travel_used_deg=52. - DT,
                        descent_elapsed_s=42. - DT)
    step(assist, 2, gap_m=.000026958)
    assert assist.snapshot()['mode_name'] == 'DESCEND_PROGRESS'
    assert assist.state['travel_used_deg'] == pytest.approx(52.)
    step(assist, 3, gap_m=.000026958)
    assert assist.state['travel_used_deg'] == pytest.approx(52. + DT)
    assert assist.state['contact_seen'] == 0.


def test_sealed_old_blocked_state_does_not_regain_mode6_from_nearness_alone():
    assist = near_budget(peak=.000138727)
    step(assist, 2, gap_m=.000026958)
    assert assist.snapshot()['mode_name'] == 'BLOCKED'
    assert assist.state['travel_used_deg'] == 52.
    assert assist.state['contact_seen'] == 0.


@pytest.mark.parametrize('peak,gap', [(.002, .0017), (.000027, .000027)])
def test_far_gap_or_reset_near_window_cannot_grant_terminal_approach(peak, gap):
    assist = near_budget(peak=peak, elapsed=0.)
    step(assist, 2, gap_m=gap)
    assert assist.snapshot()['mode_name'] == 'BLOCKED'
    assert assist.state['travel_used_deg'] == 52.
    assert assist.state['contact_seen'] == 0.


@pytest.mark.parametrize('change', [dict(physical_valid=False), dict(within_top_xy=False),
    dict(other_support_count=1), dict(qualified_RR=False), dict(crossed_RR=False),
    dict(hip_actual_deg=20.), dict(knee_actual_deg=10.)])
def test_previous_safety_and_tracking_gates_apply_to_terminal_approach(change):
    assist = near_budget()
    step(assist, 2, gap_m=.000027, **change)
    assert assist.state['travel_used_deg'] == 52.
    assert assist.state['contact_seen'] == 0.


def test_one_degree_absolute_budget_never_recharges_even_with_continued_feedback():
    assist = near_budget(peak=.001)
    for tick in range(2, 125):
        step(assist, tick, gap_m=max(0., .0007 - (tick-2) * .000004))
    assert assist.state['travel_used_deg'] == pytest.approx(53.)
    assert assist.state['descent_elapsed_s'] == pytest.approx(43.)
    before = deepcopy(assist.state)
    step(assist, 125, **top())
    step(assist, 126, gap_m=.0008)
    step(assist, 127, gap_m=.0002)
    assert assist.state['travel_used_deg'] == before['travel_used_deg']
    assert assist.state['descent_elapsed_s'] == before['descent_elapsed_s']


def test_actual_top_stops_terminal_descent_and_keeps_captured_target():
    assist = near_budget(); step(assist, 2, gap_m=.000027)
    budget, target = assist.state['travel_used_deg'], assist.state['knee_hold_deg']
    step(assist, 3, **top())
    assert assist.snapshot()['mode_name'] == 'HOLD'
    assert assist.state['contact_seen'] == 1.
    assert assist.state['travel_used_deg'] == budget and assist.state['knee_hold_deg'] == target
