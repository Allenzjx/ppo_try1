"""Finite feedback/handoff counterexamples; synthetic, not physical success."""
from copy import deepcopy

import pytest

from test_semantic_rr_capture_assist import DT, PREVIOUS, ZERO, step, top
from wlr50_clean.ppo.semantic_rr_capture_assist import (
    RRHipOnlyCaptureAssist, RR_CAPTURE_FEEDBACK_REVISION,
    RR_CAPTURE_WINDOW_REFERENCE_SEMANTICS, RR_CAPTURE_ASSIST_FEATURE_NAMES,
    apply_rr_capture_assist_snapshot, rr_capture_assist_features,
    validate_rr_capture_assist_snapshot,
)


def test_xy_handoff_holds_previous_final_without_crossing_or_descent_credit():
    assist = RRHipOnlyCaptureAssist()
    entry = ZERO[:6] + (18.18, -45.37) + (.3,) * 4
    first = step(assist, 1, previous=entry, crossed_RR=False, gap_m=.028,
                 hip_actual_deg=18., knee_actual_deg=-45.2)
    assert first['state_after']['mode_name'] == 'BLOCKED'
    assert first['state_after']['reason'] == 'qualified_crossing_unavailable'
    assert assist.state['travel_used_deg'] == assist.state['descent_elapsed_s'] == 0.
    candidate = ZERO[:6] + (11.9, -51.6) + (-1.07, 0., 0., 0.)
    target = apply_rr_capture_assist_snapshot(candidate, assist.snapshot())
    assert target[6:8] == entry[6:8]
    assert target[:6] == candidate[:6] and target[8:] == candidate[8:]
    for tick in range(2, 9):
        step(assist, tick, crossed_RR=False, gap_m=.03)
    assert assist.state['hip_target_deg'] == entry[6]
    assert assist.state['travel_used_deg'] == assist.state['descent_elapsed_s'] == 0.
    receipt = step(assist, 9, crossed_RR=True, gap_m=.03)
    assert receipt['context']['crossed_RR'] is True
    assert receipt['state_after']['mode_name'] == 'DESCEND'
    assert assist.state['hip_target_deg'] == pytest.approx(entry[6] - 2 * DT)
    assert assist.state['knee_hold_deg'] == entry[7]
    assert not assist.state['contact_seen']


def test_post_reconfiguration_local_descent_is_visible_not_entry_low_watermark():
    assist = RRHipOnlyCaptureAssist()
    step(assist, 1, gap_m=.028662)
    step(assist, 2, gap_m=.069092)
    assert assist.state['window_start_gap_m'] == .069092
    assert assist.state['window_elapsed_s'] == DT  # worsening did not reset/extend
    assist.state['window_elapsed_s'] = 2. - DT
    used = assist.state['travel_used_deg']
    step(assist, 3, gap_m=.066853)
    assert assist.snapshot()['mode_name'] == 'DESCEND'
    assert assist.state['window_start_gap_m'] == .066853
    assert assist.state['window_elapsed_s'] == DT
    assert assist.state['travel_used_deg'] == pytest.approx(used + 2 * DT)
    assert not assist.state['contact_seen']  # neither local descent nor owner is TOP


def test_monotonic_worsening_and_subthreshold_noise_do_not_renew_window():
    assist = RRHipOnlyCaptureAssist(); step(assist, 1, gap_m=.028662)
    for tick in range(2, 242):
        step(assist, tick, gap_m=.028662 + tick * .0001)
    held = deepcopy(assist.state)
    assert held['blocked_reason'] == 4. and held['travel_used_deg'] == pytest.approx(4.)
    step(assist, 242, gap_m=held['window_start_gap_m'] - .0001)
    assert assist.state['travel_used_deg'] == held['travel_used_deg']
    assert assist.state['window_elapsed_s'] == held['window_elapsed_s']


@pytest.mark.parametrize('bad', [dict(physical_valid=False), dict(within_top_xy=False),
    dict(other_support_count=1), dict(qualified_RR=False), dict(crossed_RR=False),
    dict(hip_actual_deg=30.), dict(knee_actual_deg=30.), dict(air=False)])
def test_invalid_or_untracked_samples_cannot_build_peak_or_renew_exposure(bad):
    assist = RRHipOnlyCaptureAssist(); step(assist, 1, gap_m=.03); step(assist, 2, gap_m=.03)
    before = deepcopy(assist.state)
    step(assist, 3, gap_m=.08, **bad)
    step(assist, 4, gap_m=.06, **bad)
    for key in ('window_start_gap_m', 'window_elapsed_s', 'travel_used_deg', 'descent_elapsed_s'):
        assert assist.state[key] == before[key]


def test_many_peak_drops_cannot_refresh_total_travel_budget():
    assist = RRHipOnlyCaptureAssist(); step(assist, 1, gap_m=.03)
    # Above-band measured progress earns v10 reserve, but repeated peak drops
    # cannot exceed its existing52 degree cap without the <=1 mm final rule.
    for tick in range(2, 5200):
        step(assist, tick, gap_m=.06 if tick % 2 else .059)
    assert assist.state['travel_used_deg'] == pytest.approx(52.)
    assert assist.state['descent_elapsed_s'] == pytest.approx(42.)
    assert assist.snapshot()['reason'] == 'finite_search_travel_or_margin'
    assert not assist.state['contact_seen']
    target = assist.state['hip_target_deg']
    knee_target = assist.state['knee_hold_deg']
    step(assist, 5200, gap_m=.04)
    assert assist.state['hip_target_deg'] == target
    assert assist.state['knee_hold_deg'] == knee_target
    assert assist.state['travel_used_deg'] == pytest.approx(52.)
    assert assist.state['descent_elapsed_s'] == pytest.approx(42.)


def test_contact_and_retirement_still_preempt_peak_descent():
    assist = RRHipOnlyCaptureAssist(); step(assist, 1, gap_m=.08)
    step(assist, 2, **top())
    assert assist.snapshot()['mode_name'] == 'HOLD'
    assert assist.state['travel_used_deg'] == 0.
    step(assist, 3, stage_id='P13', rl_placed_current_top_support=True, **top())
    assert assist.snapshot()['mode_name'] == 'RELEASE' and assist.state['retired'] == 1.


def test_public_peak_scalar_and_snapshot_replay_are_exact_and_versioned():
    assist = RRHipOnlyCaptureAssist(); step(assist, 1, gap_m=.028); step(assist, 2, gap_m=.07)
    snapshot = assist.snapshot()
    assert snapshot['feedback_revision'] == RR_CAPTURE_FEEDBACK_REVISION == 'progress_earned_capture_reserve_incremental_v10'
    assert snapshot['window_reference_semantics'] == RR_CAPTURE_WINDOW_REFERENCE_SEMANTICS
    assert RR_CAPTURE_ASSIST_FEATURE_NAMES[7] == 'window_start_gap_m'
    assert rr_capture_assist_features(snapshot)[7] == pytest.approx(.7)
    replay = RRHipOnlyCaptureAssist.from_snapshot(snapshot)
    assert step(assist, 3, gap_m=.068) == step(replay, 3, gap_m=.068)
    for key in ('feedback_revision', 'window_reference_semantics'):
        bad = deepcopy(snapshot); bad[key] = 'implicit_v1'
        with pytest.raises(ValueError):
            validate_rr_capture_assist_snapshot(bad)
