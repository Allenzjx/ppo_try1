"""Versioned finite hip/knee controller tests, not physical-success evidence."""
from copy import deepcopy
import json

import pytest

from test_semantic_rr_capture_assist import DT, ZERO, step, top
from wlr50_clean.infrastructure.command_batch import SERVO_ORDER, servo_limits_deg
from wlr50_clean.ppo.semantic_rr_capture_assist import (
    RRHipOnlyCaptureAssist, RR_CAPTURE_ASSIST_FEATURE_NAMES,
    RR_CAPTURE_FEEDBACK_REVISION, RR_CAPTURE_SEARCH_SEMANTICS,
    apply_rr_capture_assist_snapshot, rr_capture_assist_features,
    validate_rr_capture_assist_snapshot,
)


def advance_search(assist, *, gap=None, **changes):
    tick = 1 if assist.last_tick is None else assist.last_tick + 1
    gap = changes.pop('gap_m', gap)
    return step(assist, tick, gap_m=.08 - tick * .000003 if gap is None else gap, **changes)


@pytest.fixture(scope="module")
def hip_boundary_snapshot():
    assist = RRHipOnlyCaptureAssist()
    for _ in range(1210):
        advance_search(assist)
        if assist.state['travel_used_deg'] == 20.:
            return assist.snapshot()
    pytest.fail('hip search did not reach the exact, finite axis boundary')


@pytest.fixture
def knee_assist(hip_boundary_snapshot):
    return RRHipOnlyCaptureAssist.from_snapshot(deepcopy(hip_boundary_snapshot))


def test_original_hip_direction_then_exact_boundary_starts_only_positive_knee(knee_assist):
    assist = knee_assist
    before = assist.snapshot()
    assert before['hip_target_deg'] == pytest.approx(-10.)
    assert before['knee_hold_deg'] == -8.
    assert before['travel_used_deg'] == 20.
    assert before['descent_elapsed_s'] == pytest.approx(10.)
    assert before['window_elapsed_s'] == 0.
    assert before['mode_name'] == 'DESCEND'
    advance_search(assist)
    assert assist.state['hip_target_deg'] == before['hip_target_deg']
    assert assist.state['knee_hold_deg'] == pytest.approx(-8. + DT)
    assert assist.state['travel_used_deg'] == pytest.approx(20. + DT)
    assert assist.state['descent_elapsed_s'] == pytest.approx(10. + DT)
    candidate = tuple(float(i) for i in range(12))
    target = apply_rr_capture_assist_snapshot(candidate, assist.snapshot())
    assert target[6:8] == (assist.state['hip_target_deg'], assist.state['knee_hold_deg'])
    assert all(target[i] == candidate[i] for i in range(12) if i not in (6, 7))
    assert assist.snapshot()['owner_indices'] == [6, 7]


def test_boundary_is_not_a_resettable_latch(knee_assist):
    # Explicit public state counterexample: sitting exactly at the boundary
    # must not grant repeated fresh windows on tracking pauses or resume.
    knee_assist.state['window_elapsed_s'] = .75
    gap = knee_assist.state['window_start_gap_m']
    advance_search(knee_assist, gap=gap, hip_actual_deg=0.)
    assert knee_assist.state['blocked_reason'] == 6.
    assert knee_assist.state['window_elapsed_s'] == .75
    advance_search(knee_assist, gap=gap)
    assert knee_assist.state['window_elapsed_s'] == pytest.approx(.75 + DT)


@pytest.mark.parametrize('change,reason', [
    ({'physical_valid': False}, 1), ({'within_top_xy': False}, 2),
    ({'other_support_count': 1}, 3), ({'qualified_RR': False}, 9),
    ({'crossed_RR': False}, 9), ({'air': False}, 7),
    ({'hip_actual_deg': 0.}, 6), ({'knee_actual_deg': 0.}, 6),
])
def test_knee_retains_all_physical_and_tracking_gates(knee_assist, change, reason):
    before = deepcopy(knee_assist.state)
    advance_search(knee_assist, gap=.1, **change)
    assert knee_assist.state['blocked_reason'] == reason
    for key in ('hip_target_deg', 'knee_hold_deg', 'travel_used_deg', 'descent_elapsed_s',
                'window_start_gap_m', 'window_elapsed_s'):
        assert knee_assist.state[key] == before[key]


def test_flat_or_rising_knee_gap_stops_after_two_seconds_without_budget_recharge(knee_assist):
    assist = knee_assist
    for i in range(260):
        advance_search(assist, gap=.09 + i * .000001)
    held = deepcopy(assist.state)
    assert held['travel_used_deg'] == pytest.approx(22.)
    assert held['descent_elapsed_s'] == pytest.approx(12.)
    assert held['blocked_reason'] == 4.
    advance_search(assist, gap=held['window_start_gap_m'] - .0001)
    assert assist.state['travel_used_deg'] == held['travel_used_deg']
    advance_search(assist, gap=held['window_start_gap_m'] - .00021)
    assert assist.state['travel_used_deg'] == pytest.approx(held['travel_used_deg'] + DT)
    assert assist.state['descent_elapsed_s'] == pytest.approx(held['descent_elapsed_s'] + DT)


@pytest.mark.parametrize('contact', [dict(top_surface_contact=True, obstacle_pair_active=True),
    dict(obstacle_pair_active=True), dict(ground_contact=True)])
def test_any_contact_stops_knee_immediately_and_loss_resumes_same_axis(knee_assist, contact):
    assist = knee_assist
    advance_search(assist)
    before = deepcopy(assist.state)
    advance_search(assist, air=False, **contact)
    assert assist.snapshot()['mode_name'] == 'HOLD'
    for key in ('hip_target_deg', 'knee_hold_deg', 'travel_used_deg', 'descent_elapsed_s'):
        assert assist.state[key] == before[key]
    advance_search(assist)
    assert assist.state['hip_target_deg'] == before['hip_target_deg']
    assert assist.state['knee_hold_deg'] == pytest.approx(before['knee_hold_deg'] + DT)
    assert assist.state['travel_used_deg'] == pytest.approx(before['travel_used_deg'] + DT)


def test_contact_before_axis_boundary_does_not_spend_or_switch():
    assist = RRHipOnlyCaptureAssist(); advance_search(assist)
    assist.state.update(travel_used_deg=20. - DT, descent_elapsed_s=(20. - DT) / 2.,
                        hip_target_deg=-10. + DT)
    before = deepcopy(assist.state)
    advance_search(assist, **top())
    assert assist.state['travel_used_deg'] == before['travel_used_deg']
    assert assist.state['knee_hold_deg'] == before['knee_hold_deg']
    advance_search(assist)
    assert assist.state['travel_used_deg'] == 20.
    assert assist.state['knee_hold_deg'] == before['knee_hold_deg']
    assert assist.state['descent_elapsed_s'] == pytest.approx(10.)


@pytest.mark.parametrize('long_contact', [False, True])
def test_contact_loss_preserves_pending_target_and_never_restarts_hip(knee_assist, long_contact):
    assist = knee_assist; advance_search(assist)
    budget = (assist.state['travel_used_deg'], assist.state['descent_elapsed_s'])
    advance_search(assist, stage_id='P10', **top())
    for _ in range(90 if long_contact else 3):
        advance_search(assist, stage_id='P10', **top())
    assert assist.snapshot()['mode_name'] == 'CAPTURED_FOLLOW'
    pending = (assist.state['hip_target_deg'], assist.state['knee_hold_deg'])
    final = ZERO[:6] + (-9., -6.) + (.3,) * 4
    advance_search(assist, stage_id='P11', previous=final)
    assert assist.state['hip_target_deg'] == pending[0]
    assert assist.state['knee_hold_deg'] == pytest.approx(pending[1] + DT)
    assert assist.state['travel_used_deg'] == pytest.approx(budget[0] + DT)
    assert assist.state['descent_elapsed_s'] == pytest.approx(budget[1] + DT)
    advance_search(assist, stage_id='P11')
    assert assist.state['hip_target_deg'] == pending[0]
    assert assist.state['knee_hold_deg'] == pytest.approx(pending[1] + 2*DT)
    assert assist.state['travel_used_deg'] == pytest.approx(budget[0] + 2*DT)


def test_knee_upper_margin_remains_two_degrees_and_partial_step_counts_actual_exposure(knee_assist):
    assist = knee_assist
    upper = servo_limits_deg(SERVO_ORDER[7])[1] - 2.
    assist.state['knee_hold_deg'] = upper - DT / 2.
    before = deepcopy(assist.state)
    advance_search(assist)
    assert assist.state['knee_hold_deg'] == upper
    assert assist.state['blocked_reason'] == 5.
    assert assist.state['descent_elapsed_s'] == pytest.approx(before['descent_elapsed_s'] + DT / 2.)
    advance_search(assist, gap=.01)
    assert assist.state['knee_hold_deg'] == upper


def test_hip_exposure_exhausted_before_twenty_does_not_unlock_knee():
    assist = RRHipOnlyCaptureAssist(); advance_search(assist)
    assist.state.update(travel_used_deg=12. - DT, descent_elapsed_s=12. - DT,
                        hip_target_deg=-2. + DT)
    advance_search(assist, gap=.002)
    assert assist.state['travel_used_deg'] == pytest.approx(12.)
    assert assist.state['blocked_reason'] == 8.
    advance_search(assist, gap=.001)
    assert assist.state['travel_used_deg'] == pytest.approx(12.)
    assert assist.state['knee_hold_deg'] == -8.


def test_full_twenty_knee_and_total_exposure_limits_cannot_be_recharged(knee_assist):
    assist = knee_assist
    for _ in range(2420):
        advance_search(assist)
    assert assist.state['travel_used_deg'] == pytest.approx(40.)
    assert assist.state['descent_elapsed_s'] == pytest.approx(30.)
    assert assist.state['hip_target_deg'] == pytest.approx(-10.)
    assert assist.state['knee_hold_deg'] == pytest.approx(12.)
    assert assist.snapshot()['mode_name'] == 'BLOCKED'
    assert assist.state['blocked_reason'] == 10.
    budget = (assist.state['travel_used_deg'], assist.state['descent_elapsed_s'])
    advance_search(assist, stage_id='P10', **top())
    for _ in range(90):
        advance_search(assist, stage_id='P10', **top())
    advance_search(assist, stage_id='P11')
    assert assist.snapshot()['mode_name'] == 'BLOCKED'
    assert (assist.state['travel_used_deg'], assist.state['descent_elapsed_s']) == budget


def test_slow_hip_maximum_twelve_plus_knee_twenty_has_exact_thirtytwo_exposure(knee_assist):
    assist = knee_assist
    # Valid public history: 8 degrees at 1/s and 12 degrees at 2/s would
    # exceed the hip budget; 4 at 1/s plus 16 at 2/s uses exactly 12 seconds.
    assist.state.update(travel_used_deg=40. - DT, descent_elapsed_s=32. - DT,
                        knee_hold_deg=12. - DT)
    advance_search(assist)
    assert assist.state['travel_used_deg'] == pytest.approx(40.)
    assert assist.state['descent_elapsed_s'] == pytest.approx(32.)
    assert assist.snapshot()['mode_name'] == 'BLOCKED'


def test_public_layout_scales_and_serialized_replay_are_versioned_not_expanded(knee_assist):
    advance_search(knee_assist)
    snapshot = knee_assist.snapshot()
    assert snapshot['feedback_revision'] == RR_CAPTURE_FEEDBACK_REVISION == 'progress_reserve_captured_incremental_v4'
    assert snapshot['capture_search_semantics'] == RR_CAPTURE_SEARCH_SEMANTICS
    assert len(RR_CAPTURE_ASSIST_FEATURE_NAMES) == 14
    values = rr_capture_assist_features(snapshot)
    assert values[2] == snapshot['knee_hold_deg'] / 180.
    assert values[5] == snapshot['travel_used_deg'] / 20.
    assert values[6] == snapshot['descent_elapsed_s'] / 12.
    restored = RRHipOnlyCaptureAssist.from_snapshot(json.loads(json.dumps(snapshot)))
    assert advance_search(knee_assist) == advance_search(restored)


@pytest.mark.parametrize('change', [
    {'capture_search_semantics': 'hip_only'}, {'feedback_revision': 'window_peak_progress_v2'},
    {'travel_used_deg': 40.01}, {'descent_elapsed_s': 32.01},
    {'travel_used_deg': 25., 'descent_elapsed_s': 4.},
    {'travel_used_deg': 25., 'descent_elapsed_s': 17.01},
])
def test_incompatible_metadata_and_impossible_derived_exposure_fail_closed(knee_assist, change):
    snapshot = knee_assist.snapshot(); snapshot.update(change)
    with pytest.raises(ValueError):
        validate_rr_capture_assist_snapshot(snapshot)


def test_knee_axis_RL_real_placement_releases_once_and_never_reacquires(knee_assist):
    assist = knee_assist; advance_search(assist)
    budget = (assist.state['travel_used_deg'], assist.state['descent_elapsed_s'])
    advance_search(assist, stage_id='P13', rl_placed_current_top_support=True, **top())
    for _ in range(90):
        advance_search(assist, stage_id='P13', rl_placed_current_top_support=True, **top())
    assert assist.snapshot()['mode_name'] == 'RELEASED'
    advance_search(assist)
    assert assist.snapshot()['mode_name'] == 'RELEASED'
    assert (assist.state['travel_used_deg'], assist.state['descent_elapsed_s']) == budget
