"""Synthetic state-machine/dispatch checks, never physical success evidence."""
from __future__ import annotations

import pytest

from wlr50_clean.infrastructure.command_batch import SERVO_ORDER, servo_limits_deg
from wlr50_clean.ppo.semantic_capture_assist import (
    CAPTURE_ASSIST_FEEDBACK_REVISION, CAPTURE_ASSIST_FEATURE_NAMES,
    HipOnlyCaptureAssist, apply_capture_assist_snapshot, capture_assist_features,
)

ZERO = (0.,) * 12


def context(assist, tick, gap, *, contact=False, pair=None, phase="P06", **changes):
    return {
        "dispatch_physics_tick": tick, "stage_id": phase, "physical_valid": True,
        "within_top_xy": True, "other_support_count": 2,
        "top_surface_contact": contact,
        "obstacle_pair_active": contact if pair is None else pair,
        "placed_FL": tick > 1, "gap_m": gap,
        "hip_actual_deg": assist.state["hip_target_deg"] if assist.state["initialized"] else 10.,
        "qualified_FL": True, "crossed_FL": True, "source_unfold_dispatched": True,
        **changes,
    }


def step(assist, tick, gap, **changes):
    return assist.advance(
        context=context(assist, tick, gap, **changes),
        previous_final_full12=(10., -8.) + ZERO[2:], physics_dt_s=1./120.,
    )


def after_contact(contact_gap):
    assist = HipOnlyCaptureAssist()
    step(assist, 1, .010, phase="P05")
    step(assist, 2, contact_gap, contact=True)
    assert assist.snapshot()["mode_name"] == "HOLD"
    return assist


@pytest.mark.parametrize("contact_gap", [-.001, .0005])
def test_reopened_improving_gap_uses_new_air_window_not_old_contact_height(contact_gap):
    assist = after_contact(contact_gap)
    for index in range(300):
        gap = .010 - .005 * index / 299.
        receipt = step(assist, index + 3, gap)
        if index == 0:
            assert receipt["state_before"]["window_start_gap_m"] == contact_gap
            assert assist.state["window_start_gap_m"] == gap
            assert assist.state["window_elapsed_s"] == pytest.approx(1./120.)
        assert assist.state["blocked_reason"] == 0.
        assert assist.state["hip_entry_deg"] == 10.
        assert assist.state["knee_hold_deg"] == -8.
    assert assist.snapshot()["mode_name"] == "DESCEND"
    assert assist.state["hip_target_deg"] == pytest.approx(5.)


@pytest.mark.parametrize("end_gap", [.010, .015])
def test_flat_or_worsening_gap_still_expires_and_blocked_does_not_renew_window(end_gap):
    assist = after_contact(-.001)
    first_block = None
    for index in range(300):
        gap = .010 + (end_gap - .010) * index / 299.
        step(assist, index + 3, gap)
        if assist.state["blocked_reason"] == 4. and first_block is None:
            first_block = dict(assist.state)
    assert first_block is not None
    assert assist.state["window_elapsed_s"] >= 2.
    assert assist.state["window_start_gap_m"] == .010
    assert assist.state["hip_target_deg"] == first_block["hip_target_deg"]
    assert assist.state["window_elapsed_s"] == first_block["window_elapsed_s"]
    # Both a stage label change and another blocked AIR frame retain the
    # expired local window. Only a real new contact-loss edge may rebase it.
    step(assist, 303, end_gap, phase="P05")
    step(assist, 304, end_gap, phase="P06")
    assert assist.state["blocked_reason"] == 4.
    assert assist.state["window_elapsed_s"] == first_block["window_elapsed_s"]
    assert assist.state["hip_target_deg"] == first_block["hip_target_deg"]


def test_descending_air_and_normal_P05_P06_handoff_do_not_reset_elapsed():
    assist = after_contact(-.001)
    step(assist, 3, .010, phase="P05")
    step(assist, 4, .010, phase="P05")
    before = dict(assist.state)
    step(assist, 5, .010, phase="P06")
    assert assist.state["window_start_gap_m"] == before["window_start_gap_m"]
    assert assist.state["window_elapsed_s"] == pytest.approx(before["window_elapsed_s"] + 1./120.)
    assert assist.state["hip_entry_deg"] == before["hip_entry_deg"]
    assert assist.state["knee_hold_deg"] == before["knee_hold_deg"]


def test_long_improving_reapproach_retains_original_total_twenty_degree_limit():
    assist = after_contact(-.001)
    for index in range(1000):
        step(assist, index + 3, .100 - .080 * index / 999.)
    lower = max(10. - 20., servo_limits_deg(SERVO_ORDER[0])[0] + 2.)
    assert assist.state["hip_target_deg"] == pytest.approx(lower)
    assert assist.state["hip_entry_deg"] == 10.
    assert assist.state["knee_hold_deg"] == -8.
    assert assist.state["blocked_reason"] == 5.


def test_repeated_contact_losses_do_not_reanchor_target_or_replenish_total_travel():
    assist = after_contact(-.001)
    for index in range(2500):
        # Every real contact stops accumulation; each AIR interval may
        # request one small step but never obtains another twenty degrees.
        step(assist, 3 + 2 * index, .001, contact=False)
        target = assist.state["hip_target_deg"]
        step(assist, 4 + 2 * index, -.001, contact=True)
        assert assist.state["hip_target_deg"] == target
        assert assist.state["hip_entry_deg"] == 10.
        assert assist.state["knee_hold_deg"] == -8.
    assert assist.state["hip_target_deg"] == pytest.approx(-10.)
    step(assist, 5003, .001)
    assert assist.state["blocked_reason"] == 5.
    assert assist.state["hip_target_deg"] == pytest.approx(-10.)


def test_any_real_obstacle_pair_stops_pressing_without_claiming_top_placement():
    assist = after_contact(-.001)
    step(assist, 3, .010)
    target = assist.state["hip_target_deg"]
    step(assist, 4, .008, contact=False, pair=True)
    assert assist.snapshot()["mode_name"] == "HOLD"
    assert assist.state["hip_target_deg"] == target
    assert assist.state["window_elapsed_s"] == 0.
    assert assist.state["window_start_gap_m"] == .008
    # Actual pair disappears: rebase once, retain useful output and anchors.
    step(assist, 5, .012, contact=False, pair=False)
    assert assist.state["window_start_gap_m"] == .012
    assert assist.state["window_elapsed_s"] == pytest.approx(1./120.)
    assert assist.state["hip_entry_deg"] == 10.


@pytest.mark.parametrize("changes,reason", [
    ({"physical_valid": False}, 1.),
    ({"within_top_xy": False}, 2.),
    ({"other_support_count": 1}, 3.),
    ({"hip_actual_deg": 25.}, 6.),
])
def test_contact_loss_window_rebase_never_bypasses_existing_safety_or_tracking(changes, reason):
    assist = after_contact(-.001)
    target = assist.state["hip_target_deg"]
    step(assist, 3, .010, **changes)
    assert assist.state["blocked_reason"] == reason
    assert assist.state["hip_target_deg"] == target
    assert assist.state["knee_hold_deg"] == -8.


def test_feedback_revision_is_explicit_nonnumeric_without_changing_twelve_features():
    assist = HipOnlyCaptureAssist()
    snapshot = assist.snapshot()
    assert snapshot["feedback_revision"] == CAPTURE_ASSIST_FEEDBACK_REVISION == "hold_to_air_progress_window_v2"
    assert isinstance(snapshot["feedback_revision"], str)
    assert len(CAPTURE_ASSIST_FEATURE_NAMES) == 12
    assert capture_assist_features(snapshot) == (0.,) * 12
    assert "feedback_revision" not in CAPTURE_ASSIST_FEATURE_NAMES
    assert apply_capture_assist_snapshot((1.,) * 12, snapshot) == (1.,) * 12


def test_native_auditor_independently_reconstructs_new_recontact_state_transition():
    from test_actuator_target_effect import _adapter
    from wlr50_clean.ppo.actuator_target_effect import build_actuator_target_effect_audit
    from wlr50_clean.ppo.isaac_fsm_backend import build_residual_actuation_plan
    from wlr50_clean.ppo.semantic_residual_adapter import apply_semantic_residual

    adapter, assist = _adapter(), HipOnlyCaptureAssist()
    nominal, residual = (1., -1.) + ZERO[2:], (.25, .125) + ZERO[2:]
    plan = build_residual_actuation_plan(
        tuple(a + b for a, b in zip(nominal, residual)),
        frozen_nominal_full12=nominal,
        drive_feedback_bias_full12=ZERO, normal_drive_bias_full12=ZERO,
    )
    for tick, gap, contact in ((1, .010, False), (2, -.001, True),
                               (3, .010, False), (4, .009, False)):
        previous = tuple(adapter._final_drive_servo_deg[name] for name in SERVO_ORDER)
        ack = apply_semantic_residual(
            adapter, nominal, physics_tick=tick, tracking_servo_names=(),
            controller_bias_full12=ZERO, projected_residual_full12=residual,
            capture_assist=assist,
            capture_assist_context=context(assist, tick, gap, contact=contact, hip_actual_deg=previous[0]),
        )
        audit = build_actuator_target_effect_audit(
            adapter=adapter, actuation=plan, raw_ack=ack,
            previous_final_drive_servo_deg=previous,
            source_phase_id="P06", policy_request=None,
        )
        assert audit["verified"]
        assert audit["capture_assist_state_transition_independently_reconstructed"]
        assert audit["capture_assist_evidence"]["state_after"]["feedback_revision"] == CAPTURE_ASSIST_FEEDBACK_REVISION
        assert ack["independent_policy_residual_requested_full12"] == list(residual)
        if tick == 3:
            assert assist.state["window_start_gap_m"] == .010
            assert assist.state["window_elapsed_s"] == pytest.approx(1./120.)
