"""Synthetic earned-reserve wiring checks; not physical capture evidence.

STAGED ONLY. Normal production imports intentionally require the v10 patch.
No tests were run by the staging author while the sole Isaac remained active.
"""
from copy import deepcopy
import json

import pytest

from test_semantic_rr_capture_assist import DT, ZERO, step, top
from test_semantic_rr_capture_context import measured, facts
from wlr50_clean.infrastructure.command_batch import SERVO_ORDER, servo_limits_deg
from wlr50_clean.ppo.semantic_rr_capture_assist import (
    RRHipOnlyCaptureAssist, RR_CAPTURE_ASSIST_FEATURE_NAMES,
    RR_CAPTURE_FEEDBACK_REVISION, RR_CAPTURE_SEARCH_SEMANTICS,
    _search_limits, rr_capture_assist_features, validate_rr_capture_assist_snapshot,
)


def boundary(*, travel=40., exposure=30., peak=.0306, elapsed=0.):
    assist = RRHipOnlyCaptureAssist()
    step(assist, 1, gap_m=peak)
    assist.state.update(mode=3., blocked_reason=5., travel_used_deg=travel,
        descent_elapsed_s=exposure, hip_target_deg=-4.65054,
        knee_hold_deg=-24.76254, window_start_gap_m=peak,
        window_elapsed_s=elapsed)
    validate_rr_capture_assist_snapshot(assist.snapshot())
    return assist


def test_real_terminal_values_do_not_by_themselves_invent_new_point_two_mm_credit():
    assist = boundary(peak=.0302553, elapsed=DT)
    step(assist, 2, gap_m=.0301154, other_support_count=3,
         hip_actual_deg=-4.94657, knee_actual_deg=-24.67888)
    assert assist.state['travel_used_deg'] == 40.
    assert assist.snapshot()['mode_name'] == 'BLOCKED'
    assert assist.snapshot()['reason'] == 'fresh_capture_progress_required'


def test_measured_78point6s_gap_pair_can_earn_credit_without_claiming_full_physical_replay():
    # Sealed v9 first episode global222486: this peak/current pair decreased
    # by 0.3314303947 mm while remaining above25 mm. The rest is a synthetic
    # valid fixture, not a fabricated complete context/physics replay.
    assist = boundary(travel=39.83333333333221, exposure=29.83333333333221,
        peak=.030055098814890588, elapsed=DT)
    step(assist, 2, gap_m=.02972366842018978)
    assert assist.snapshot()['mode_name'] == 'DESCEND_PROGRESS'
    assert assist.state['window_start_gap_m'] == .02972366842018978
    assert assist.state['window_elapsed_s'] == pytest.approx(DT)
    assert assist.state['travel_used_deg'] == pytest.approx(39.84166666666554)
    assert assist.state['contact_seen'] == 0.


def test_earned_descent_before_40_degree_boundary_carries_across_it_above_25mm():
    assist = boundary(travel=40.-2.*DT, exposure=30.-2.*DT)
    step(assist, 2, gap_m=.0302553, other_support_count=3,
         hip_actual_deg=-4.94657, knee_actual_deg=-24.67888)
    assert assist.snapshot()['mode_name'] == 'DESCEND_PROGRESS'
    assert assist.state['window_start_gap_m'] == .0302553
    step(assist, 3, gap_m=.0301154)
    assert assist.state['travel_used_deg'] == pytest.approx(40.)
    assert assist.snapshot()['mode_name'] == 'DESCEND_PROGRESS'
    step(assist, 4, gap_m=.029968)
    assert assist.state['travel_used_deg'] == pytest.approx(40.+DT)
    assert assist.state['descent_elapsed_s'] == pytest.approx(30.+DT)
    assert assist.state['hip_target_deg'] == -4.65054
    assert assist.state['knee_hold_deg'] == pytest.approx(-24.76254+3.*DT)
    assert not assist.state['contact_seen']


@pytest.mark.parametrize('peak,gap', [(.0306, .0303), (.04, .0397), (.08, .0797)])
def test_positive_gap_is_not_a_new_hard_ceiling_if_other_gates_and_progress_hold(peak, gap):
    assist = boundary(peak=peak)
    step(assist, 2, gap_m=gap)
    assert assist.snapshot()['mode_name'] == 'DESCEND_PROGRESS'
    assert assist.state['travel_used_deg'] == pytest.approx(40.+DT)
    assert assist.state['contact_seen'] == 0.


@pytest.mark.parametrize('change', [dict(physical_valid=False), dict(within_top_xy=False),
    dict(other_support_count=1), dict(qualified_RR=False), dict(crossed_RR=False),
    dict(air=False), dict(hip_actual_deg=0.), dict(knee_actual_deg=0.)])
def test_current_invalid_geometry_support_lift_or_tracking_cannot_earn_reserve(change):
    assist = boundary()
    before = deepcopy(assist.state)
    step(assist, 2, gap_m=.0303, **change)
    assert assist.snapshot()['mode_name'] == 'BLOCKED'
    for key in ('travel_used_deg', 'descent_elapsed_s', 'hip_target_deg', 'knee_hold_deg',
                'window_start_gap_m', 'window_elapsed_s'):
        assert assist.state[key] == before[key]


@pytest.mark.parametrize('gap,permitted', [(-.015, True), (-.0150001, False)])
def test_unchanged_negative_gap_floor(gap, permitted):
    assist = boundary(peak=-.014)
    step(assist, 2, gap_m=gap)
    assert (assist.state['travel_used_deg'] > 40.) is permitted
    assert assist.state['contact_seen'] == 0.


def test_above_band_credit_still_stalls_after_two_active_seconds_and_never_self_renews():
    assist = boundary(); step(assist, 2, gap_m=.0303)
    for tick in range(3, 270):
        step(assist, tick, gap_m=.0303)
    assert assist.snapshot()['mode_name'] == 'BLOCKED'
    assert assist.state['travel_used_deg'] == pytest.approx(42.)
    before = deepcopy(assist.state)
    step(assist, 270, gap_m=.0303)
    assert assist.state == before


@pytest.mark.parametrize('contact', [dict(air=False, ground_contact=True),
    dict(air=False, obstacle_pair_active=True), top()])
def test_any_real_contact_stops_descent_and_never_recharges_budget(contact):
    assist = boundary(); step(assist, 2, gap_m=.0303)
    used = assist.state['travel_used_deg']
    step(assist, 3, **contact)
    assert assist.snapshot()['mode_name'] == 'HOLD'
    assert assist.state['travel_used_deg'] == used
    step(assist, 4, gap_m=.0303)
    assert assist.state['travel_used_deg'] == used
    assert assist.snapshot()['mode_name'] == 'BLOCKED'


def test_unchanged_12_degree_reserve_exhausts_at_52_without_near_plane_bonus():
    assist = boundary()
    for tick in range(2, 1460):
        step(assist, tick, gap_m=.0303-(tick-2)*.000003)
    assert assist.state['travel_used_deg'] == pytest.approx(52.)
    assert assist.state['descent_elapsed_s'] == pytest.approx(42.)
    assert assist.snapshot()['mode_name'] == 'BLOCKED'
    assert _search_limits({**assist.state, 'mode': 6.}) == (52., 44.)


def test_final_one_degree_and_absolute_exposure_still_require_public_peak_within_one_mm():
    assist = boundary(travel=52., exposure=44., peak=.0014)
    step(assist, 2, gap_m=.0011)
    assert assist.state['travel_used_deg'] == 52.
    assert assist.snapshot()['mode_name'] == 'BLOCKED'
    step(assist, 3, gap_m=.0008)
    assert assist.snapshot()['mode_name'] == 'DESCEND_PROGRESS'
    assert _search_limits(assist.state) == (53., 45.)
    for tick in range(4, 125):
        step(assist, tick, gap_m=.0008-(tick-3)*.000003)
    assert assist.state['travel_used_deg'] == pytest.approx(53.)
    assert assist.state['descent_elapsed_s'] == pytest.approx(45.)
    assert assist.snapshot()['mode_name'] == 'BLOCKED'
    used = deepcopy(assist.state)
    step(assist, 125, **top()); step(assist, 126, gap_m=.0002)
    step(assist, 127, gap_m=-.0001)
    assert assist.state['travel_used_deg'] == used['travel_used_deg']
    assert assist.state['descent_elapsed_s'] == used['descent_elapsed_s']


def test_knee_margin_is_still_two_degrees_even_with_fresh_above_band_progress():
    assist = boundary()
    assist.state['knee_hold_deg'] = servo_limits_deg(SERVO_ORDER[7])[1]-2.
    step(assist, 2, gap_m=.0303)
    assert assist.state['travel_used_deg'] == 40.
    assert assist.snapshot()['mode_name'] == 'BLOCKED'


def test_new_mode6_snapshot_roundtrip_keeps_exact_fourteen_features():
    assist = boundary(); step(assist, 2, gap_m=.0303)
    snapshot = assist.snapshot()
    assert snapshot['feedback_revision'] == RR_CAPTURE_FEEDBACK_REVISION == 'progress_earned_capture_reserve_incremental_v10'
    assert snapshot['capture_search_semantics'] == RR_CAPTURE_SEARCH_SEMANTICS
    assert len(RR_CAPTURE_ASSIST_FEATURE_NAMES) == len(rr_capture_assist_features(snapshot)) == 14
    replay = RRHipOnlyCaptureAssist.from_snapshot(json.loads(json.dumps(snapshot)))
    assert step(assist, 3, gap_m=.0302) == step(replay, 3, gap_m=.0302)
    for change in (dict(window_start_gap_m=-.0151), dict(window_elapsed_s=2.),
                   dict(travel_used_deg=53.1), dict(descent_elapsed_s=45.1)):
        with pytest.raises(ValueError):
            validate_rr_capture_assist_snapshot({**snapshot, **change})


def test_scheduler_context_can_continue_true_progress_without_contact_or_rl_credit():
    assist = boundary(); step(assist, 2, gap_m=.0303)
    task, observation = measured()
    task['physical_evaluator']['current_legs']['RR']['clearance_m'] = .0303
    result = facts(task, observation, assist_snapshot=assist.snapshot())
    assert result['rr_top_reachable'] and result['rr_capture_recovery_allowed']
    assert not result['rr_top_contact'] and not result['rr_current_bearing']
    assert not result['rl_transfer_ready']
    assert not task['physical_evaluator']['history']['placed']['RR']


def test_no_fresh_credit_cannot_borrow_scheduler_recovery():
    assist = boundary(peak=.0302553)
    step(assist, 2, gap_m=.0301154)
    task, observation = measured()
    task['physical_evaluator']['current_legs']['RR']['clearance_m'] = .0301154
    result = facts(task, observation, assist_snapshot=assist.snapshot())
    assert result['rr_top_reachable'] and not result['rr_capture_recovery_allowed']
