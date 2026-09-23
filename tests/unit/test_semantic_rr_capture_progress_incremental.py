"""Finite synthetic controller counterexamples, not physical-success evidence."""
from copy import deepcopy
import json

import pytest

from test_semantic_rr_capture_assist import DT, ZERO, PREVIOUS, step, top, context
from wlr50_clean.infrastructure.command_batch import SERVO_ORDER, servo_limits_deg
from wlr50_clean.ppo.semantic_rr_capture_assist import (
    RRHipOnlyCaptureAssist, RR_CAPTURE_ASSIST_FEATURE_NAMES, RR_CAPTURE_FEEDBACK_REVISION,
    apply_rr_capture_assist_snapshot, rr_capture_assist_features, validate_rr_capture_assist_snapshot,
)


def boundary():
    assist = RRHipOnlyCaptureAssist()
    step(assist, 1, gap_m=.014)
    assist.state.update(mode=3., blocked_reason=5., travel_used_deg=40., descent_elapsed_s=30.,
                        hip_target_deg=-1.785913, knee_hold_deg=-25.361176)
    validate_rr_capture_assist_snapshot(assist.snapshot())
    return assist


def captured():
    assist = boundary()
    step(assist, 2, **top())
    assert assist.state['contact_seen'] == 1.
    return assist


def test_reset_window_is_not_progress_credit_at_the_original_budget_boundary():
    assist = boundary()
    step(assist, 2, gap_m=.014)
    assert assist.snapshot()['mode_name'] == 'BLOCKED'
    assert assist.snapshot()['reason'] == 'fresh_near_top_progress_required'
    assert assist.state['travel_used_deg'] == 40.
    step(assist, 3, gap_m=.01379)
    assert assist.snapshot()['mode_name'] == 'DESCEND_PROGRESS'
    assert assist.state['travel_used_deg'] == pytest.approx(40. + DT)


def test_public_progress_credit_survives_physics_ticks_not_contact_or_support_reset():
    assist = boundary(); step(assist, 2, gap_m=.0137)
    budget = assist.state['travel_used_deg']
    step(assist, 3, gap_m=.01369)
    assert assist.state['travel_used_deg'] == pytest.approx(budget + DT)
    step(assist, 4, **top())
    saved = assist.state['travel_used_deg']
    step(assist, 5, gap_m=.014)
    assert assist.state['travel_used_deg'] == saved
    assert assist.snapshot()['mode_name'] == 'BLOCKED'
    step(assist, 6, gap_m=.0137)
    assert assist.snapshot()['mode_name'] == 'DESCEND_PROGRESS'
    assert assist.state['travel_used_deg'] == pytest.approx(saved + DT)


@pytest.mark.parametrize('change', [dict(physical_valid=False), dict(within_top_xy=False),
    dict(other_support_count=1), dict(qualified_RR=False), dict(crossed_RR=False),
    dict(air=False), dict(hip_actual_deg=20.), dict(knee_actual_deg=0.)])
def test_invalid_actual_state_cannot_earn_or_spend_reserve(change):
    assist = boundary(); step(assist, 2, gap_m=.0137, **change)
    assert assist.state['travel_used_deg'] == 40.
    assert assist.snapshot()['mode_name'] == 'BLOCKED'


def test_near_top_gate_rejects_far_gap_even_after_true_drop():
    assist = boundary(); assist.state['window_start_gap_m'] = .040
    step(assist, 2, gap_m=.039)
    assert assist.state['travel_used_deg'] == 40.
    assert assist.snapshot()['mode_name'] == 'BLOCKED'


def test_one_twelve_degree_reserve_finishes_at_52_without_recharging():
    assist = boundary()
    for tick in range(2, 1460):
        step(assist, tick, gap_m=.0137 - (tick - 2) * .000003)
    assert assist.state['travel_used_deg'] == pytest.approx(52.)
    assert assist.state['descent_elapsed_s'] == pytest.approx(42.)
    assert assist.state['knee_hold_deg'] == pytest.approx(-13.361176)
    assert assist.snapshot()['reason'] == 'finite_search_travel_or_margin'
    before = deepcopy(assist.state)
    step(assist, 1460, **top())
    step(assist, 1461, gap_m=.003)
    step(assist, 1462, gap_m=.0027)
    assert assist.state['travel_used_deg'] == before['travel_used_deg']
    assert assist.state['descent_elapsed_s'] == before['descent_elapsed_s']


def test_flat_reserve_stops_in_two_active_seconds_and_never_renews_itself():
    assist = boundary(); step(assist, 2, gap_m=.0137)
    for tick in range(3, 270):
        step(assist, tick, gap_m=.0137)
    assert assist.state['travel_used_deg'] == pytest.approx(42.)
    assert assist.snapshot()['mode_name'] == 'BLOCKED'
    before = deepcopy(assist.state)
    step(assist, 270, gap_m=.0137)
    assert assist.state == before


@pytest.mark.parametrize('contact', [dict(ground_contact=True), dict(obstacle_pair_active=True)])
def test_ground_or_wall_stops_search_without_becoming_TOP_capture_history(contact):
    assist = boundary(); step(assist, 2, air=False, **contact)
    assert assist.snapshot()['mode_name'] == 'HOLD'
    assert assist.state['contact_seen'] == 0.
    step(assist, 3, stage_id='P10', air=False, issued_nominal_delta_rr_deg=(0., 7.4), **contact)
    assert assist.state['knee_hold_deg'] == -25.361176


def test_captured_target_integrates_each_source_and_policy_delta_once_not_absolute_candidate():
    assist = captured(); start = (assist.state['hip_target_deg'], assist.state['knee_hold_deg'])
    receipt = step(assist, 3, stage_id='P10', issued_nominal_delta_rr_deg=(0., 5.3),
                   issued_requested_residual_delta_rr_deg=(.1, -.4), **top())
    assert assist.snapshot()['mode_name'] == 'CAPTURED_FOLLOW'
    assert assist.state['hip_target_deg'] == pytest.approx(start[0] + .1)
    assert assist.state['knee_hold_deg'] == pytest.approx(start[1] + 4.9)
    output = apply_rr_capture_assist_snapshot(ZERO[:6] + (-.3235, -58.) + ZERO[8:], assist.snapshot())
    assert output[7] == pytest.approx(start[1] + 4.9)
    previous = ZERO[:6] + start + ZERO[8:]  # final slew has not caught up
    step(assist, 4, previous=previous, stage_id='P10', issued_nominal_delta_rr_deg=(0., 2.1), **top())
    assert assist.state['knee_hold_deg'] == pytest.approx(start[1] + 7.)
    target = assist.state['knee_hold_deg']
    step(assist, 5, previous=previous, stage_id='P10', **top())
    assert assist.state['knee_hold_deg'] == target
    assert receipt['captured_incremental_target']['requested_residual_delta_rr_deg'] == [.1, -.4]


def test_two_real_supports_RR_plus_FL_are_admissible_while_RL_lifts():
    assist = captured()
    step(assist, 3, stage_id='P12', other_support_count=1, rl_qualified_lift=True,
         issued_requested_residual_delta_rr_deg=(-.2, .3), **top())
    assert assist.snapshot()['mode_name'] == 'CAPTURED_FOLLOW'
    assert assist.state['retired'] == 0.
    assert assist.state['hip_target_deg'] == pytest.approx(-1.985913)


def test_support_loss_retains_all_pending_source_deltas_but_physical_output_holds_FINAL():
    assist = captured(); before = assist.state['knee_hold_deg']
    previous = ZERO[:6] + (-1.8, -25.4) + ZERO[8:]
    step(assist, 3, previous=previous, stage_id='P10', other_support_count=0,
         issued_nominal_delta_rr_deg=(0., 7.4), **top())
    assert assist.state['knee_hold_deg'] == pytest.approx(before + 7.4)
    assert assist.snapshot()['mode_name'] == 'BLOCKED'
    with pytest.raises(ValueError, match='previous FINAL'):
        apply_rr_capture_assist_snapshot(ZERO, assist.snapshot())
    assert apply_rr_capture_assist_snapshot(ZERO, assist.snapshot(), previous_final_full12=previous)[6:8] == previous[6:8]
    step(assist, 4, previous=previous, stage_id='P10', other_support_count=1, **top())
    assert assist.state['knee_hold_deg'] == pytest.approx(before + 7.4)
    assert apply_rr_capture_assist_snapshot(ZERO, assist.snapshot())[7] == pytest.approx(before + 7.4)


def test_AIR_contact_loss_does_not_reanchor_or_reset_spent_budget():
    assist = captured()
    step(assist, 3, stage_id='P10', **top())
    target = (assist.state['hip_target_deg'], assist.state['knee_hold_deg'])
    budget = (assist.state['travel_used_deg'], assist.state['descent_elapsed_s'])
    step(assist, 4, previous=PREVIOUS, stage_id='P11', gap_m=.014,
         issued_requested_residual_delta_rr_deg=(-.2, .3))
    assert assist.state['hip_target_deg'] == pytest.approx(target[0] - .2)
    assert assist.state['knee_hold_deg'] == pytest.approx(target[1] + .3)
    assert (assist.state['travel_used_deg'], assist.state['descent_elapsed_s']) == budget
    assert assist.state['release_fraction'] == 0.


def test_positive_and_negative_increment_respect_unchanged_two_degree_margins():
    assist = captured()
    step(assist, 3, stage_id='P10', issued_requested_residual_delta_rr_deg=(1000., -1000.), **top())
    assert assist.state['hip_target_deg'] == servo_limits_deg(SERVO_ORDER[6])[1] - 2.
    assert assist.state['knee_hold_deg'] == servo_limits_deg(SERVO_ORDER[7])[0] + 2.
    step(assist, 4, stage_id='P10', issued_requested_residual_delta_rr_deg=(-.5, .5), **top())
    assert assist.state['hip_target_deg'] == servo_limits_deg(SERVO_ORDER[6])[1] - 2.5
    assert assist.state['knee_hold_deg'] == servo_limits_deg(SERVO_ORDER[7])[0] + 2.5


def test_actual_and_zero_current_policy_branches_must_advance_from_same_before_independently():
    actual = captured(); zero = RRHipOnlyCaptureAssist.from_snapshot(actual.snapshot())
    # Previous issued request=+1; actual current=+1.2, zero current=0.
    step(actual, 3, stage_id='P10', issued_requested_residual_delta_rr_deg=(0., .2), **top())
    step(zero, 3, stage_id='P10', issued_requested_residual_delta_rr_deg=(0., -1.), **top())
    assert actual.state['knee_hold_deg'] - zero.state['knee_hold_deg'] == pytest.approx(1.2)


def test_RL_lift_does_not_retire_but_current_TOP_placement_permits_finite_release():
    assist = captured()
    step(assist, 3, stage_id='P12', rl_qualified_lift=True, other_support_count=1, **top())
    assert assist.state['retired'] == 0.
    step(assist, 4, stage_id='P13', rl_placed_current_top_support=True, other_support_count=1, **top())
    assert assist.snapshot()['mode_name'] == 'RELEASE' and assist.state['release_fraction'] == 0.
    for tick in range(5, 95):
        step(assist, tick, stage_id='P13', rl_placed_current_top_support=True,
             other_support_count=1, release_candidate_rr_deg=(-1., -20.), **top())
    assert assist.snapshot()['mode_name'] == 'RELEASED'
    assert assist.state['release_fraction'] == 1.
    before = deepcopy(assist.state)
    step(assist, 95, stage_id='P09', gap_m=.01)
    assert assist.state == before


def test_release_contact_loss_holds_FINAL_and_fraction_without_reanchor_or_candidate_leak():
    assist = captured()
    step(assist, 3, stage_id='P13', rl_placed_current_top_support=True, **top())
    step(assist, 4, stage_id='P13', rl_placed_current_top_support=True,
         release_candidate_rr_deg=(-1., -20.), **top())
    before = deepcopy(assist.state)
    previous = ZERO[:6] + (-1.8, -25.4) + ZERO[8:]
    step(assist, 5, previous=previous, stage_id='P13', rl_placed_current_top_support=True,
         release_candidate_rr_deg=(90., -58.))
    for key in ('hip_target_deg', 'knee_hold_deg', 'release_fraction', 'travel_used_deg', 'descent_elapsed_s'):
        assert assist.state[key] == before[key]
    assert apply_rr_capture_assist_snapshot((90.,) * 12, assist.snapshot(), previous_final_full12=previous)[6:8] == previous[6:8]
    step(assist, 6, stage_id='P13', rl_placed_current_top_support=True,
         release_candidate_rr_deg=(-1., -20.), **top())
    assert assist.state['release_fraction'] == pytest.approx(before['release_fraction'] + DT/.75)


def test_all_action_relevant_state_replays_and_keeps_exact_fourteen_unclipped_features():
    assist = boundary(); step(assist, 2, gap_m=.0137)
    snapshot = assist.snapshot()
    assert snapshot['feedback_revision'] == RR_CAPTURE_FEEDBACK_REVISION == 'signed_band_contact_formation_incremental_v6'
    assert len(RR_CAPTURE_ASSIST_FEATURE_NAMES) == 14
    features = rr_capture_assist_features(snapshot)
    assert features[0] == 6./5. and features[5] > 2.
    replay = RRHipOnlyCaptureAssist.from_snapshot(json.loads(json.dumps(snapshot)))
    assert step(assist, 3, gap_m=.0136) == step(replay, 3, gap_m=.0136)


@pytest.mark.parametrize('change', [dict(issued_nominal_delta_rr_deg=None),
    dict(issued_requested_residual_delta_rr_deg=(float('nan'), 0.)),
    dict(rl_placed_current_top_support=1)])
def test_invalid_follow_context_is_transactional(change):
    assist = captured(); before = assist.snapshot()
    with pytest.raises(ValueError):
        step(assist, 3, stage_id='P10', **top(), **change)
    assert assist.snapshot() == before
