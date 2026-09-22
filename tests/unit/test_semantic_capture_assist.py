from __future__ import annotations

from types import SimpleNamespace

import pytest

from test_actuator_target_effect import _adapter
from wlr50_clean.infrastructure.command_batch import SERVO_ORDER, servo_limits_deg
from wlr50_clean.ppo.semantic_capture_assist import (
    HipOnlyCaptureAssist, CAPTURE_ASSIST_FEATURE_NAMES, capture_assist_features,
    apply_capture_assist_snapshot, capture_assist_context,
)
from wlr50_clean.ppo.semantic_residual_adapter import apply_semantic_residual
from wlr50_clean.ppo.isaac_fsm_backend import build_residual_actuation_plan
from wlr50_clean.ppo.actuator_target_effect import build_actuator_target_effect_audit

ZERO = (0.,)*12


def context(tick, **changes):
    return {"dispatch_physics_tick":tick, "stage_id":"P05", "physical_valid":True,
        "within_top_xy":True, "other_support_count":2, "top_surface_contact":False,
        "obstacle_pair_active":False, "placed_FL":False, "gap_m":.008,
        "hip_actual_deg":10., "qualified_FL":True, "crossed_FL":True,
        "source_unfold_dispatched":True, **changes}


def advance(assist, tick, previous=(10.,-8.)+ZERO[2:], **changes):
    return assist.advance(context=context(tick,**changes), previous_final_full12=previous,
                          physics_dt_s=1./120.)


def test_feedback_descent_holds_final_knee_and_never_reintegrates_nominal():
    assist = HipOnlyCaptureAssist()
    advance(assist,1)
    anchor=assist.snapshot()["knee_hold_deg"]
    for tick in range(2,241):
        gap=.008-(tick-1)*.000005
        advance(assist,tick,hip_actual_deg=assist.state["hip_target_deg"],gap_m=gap)
        transformed=apply_capture_assist_snapshot((100.,50.)+(2.,)*10,assist.snapshot())
        assert transformed[1] == anchor == -8.
        assert transformed[2:] == (2.,)*10
    assert assist.state["hip_target_deg"] == pytest.approx(10.-239*2./120.)
    assert len(capture_assist_features(assist.snapshot())) == len(CAPTURE_ASSIST_FEATURE_NAMES) == 12


def test_real_contact_stops_accumulation_holds_P06_then_releases_P07_continuously():
    assist=HipOnlyCaptureAssist()
    advance(assist,1)
    advance(assist,2)
    hip=assist.state["hip_target_deg"]
    for tick in range(3,80):
        advance(assist,tick,stage_id="P06",top_surface_contact=True,placed_FL=True)
        assert assist.state["hip_target_deg"] == hip
        assert apply_capture_assist_snapshot(ZERO,assist.snapshot())[:2] == (hip,-8.)
    released=[]
    for tick in range(80,171):
        advance(assist,tick,stage_id="P07",top_surface_contact=True,placed_FL=True)
        released.append(apply_capture_assist_snapshot(ZERO,assist.snapshot())[0])
    assert all(a >= b for a,b in zip(released,released[1:]))
    assert released[-1] == 0. and assist.snapshot()["mode_name"] == "RELEASED"


@pytest.mark.parametrize("changes,reason",[(dict(physical_valid=False),1),
    (dict(within_top_xy=False),2),(dict(other_support_count=1),3),
    (dict(hip_actual_deg=30.),6)])
def test_block_never_keeps_pressing_or_changes_other_channels(changes,reason):
    assist=HipOnlyCaptureAssist();advance(assist,1)
    advance(assist,2,**changes)
    assert assist.state["hip_target_deg"] == 10.
    assert assist.state["blocked_reason"] == reason
    assert apply_capture_assist_snapshot((20.,20.)+(3.,)*10,assist.snapshot())[2:] == (3.,)*10


def test_no_gap_progress_stops_finite_direction_until_body_or_wheels_help():
    assist=HipOnlyCaptureAssist();advance(assist,1)
    for tick in range(2,302):
        advance(assist,tick,hip_actual_deg=assist.state["hip_target_deg"])
    held=assist.state["hip_target_deg"]
    assert assist.state["blocked_reason"] == 4
    advance(assist,302,hip_actual_deg=held,gap_m=.0075,stage_id="P06")
    assert assist.state["hip_target_deg"] < held
    assert assist.state["knee_hold_deg"] == -8.


def test_long_initial_tracking_wait_does_not_consume_descent_window():
    assist=HipOnlyCaptureAssist();advance(assist,1)
    for tick in range(2,400):
        advance(assist,tick,hip_actual_deg=20.)
    assert assist.state["hip_target_deg"] == 10. and assist.state["window_elapsed_s"] == 0.
    advance(assist,400,hip_actual_deg=10.)
    assert assist.state["hip_target_deg"] < 10. and assist.state["mode"] == 1


def test_total_search_cannot_exceed_twenty_degrees_or_original_reserve():
    assist=HipOnlyCaptureAssist();advance(assist,1,gap_m=.05)
    for tick in range(2,1800):
        advance(assist,tick,hip_actual_deg=assist.state["hip_target_deg"],gap_m=.050-tick*.000015)
    assert assist.state["hip_target_deg"] == pytest.approx(max(10.-20.,servo_limits_deg(SERVO_ORDER[0])[0]+2.))
    assert assist.state["blocked_reason"] == 5


@pytest.mark.parametrize("field",["source_unfold_dispatched","qualified_FL","crossed_FL","within_top_xy"])
def test_assist_does_not_steal_unfinished_source_unfold_or_uncrossed_leg(field):
    assist=HipOnlyCaptureAssist();advance(assist,1,**{field:False})
    assert assist.snapshot()["mode_name"] == "WAIT"
    assert apply_capture_assist_snapshot((2.,)*12,assist.snapshot()) == (2.,)*12


def test_real_dispatch_knee_fixed_under_nominal_and_policy_changes_and_audit_passes():
    adapter=_adapter()
    for tick in range(1,25):
        apply_semantic_residual(adapter,(10.,-8.)+ZERO[2:],physics_tick=tick,
            tracking_servo_names=(),controller_bias_full12=ZERO,projected_residual_full12=ZERO)
    assist=HipOnlyCaptureAssist()
    knee=adapter._final_drive_servo_deg[SERVO_ORDER[1]]
    nominal_history=[]
    for tick in range(25,60):
        previous=tuple(adapter._final_drive_servo_deg[name] for name in SERVO_ORDER)
        nominal=(20.,20.)+ZERO[2:]
        residual=(2.,3.)+(.2,)*10
        ack=apply_semantic_residual(adapter,nominal,physics_tick=tick,tracking_servo_names=(),
            controller_bias_full12=ZERO,projected_residual_full12=residual,capture_assist=assist,
            capture_assist_context=context(tick,hip_actual_deg=previous[0],gap_m=.008-(tick-25)*.00001))
        assert ack["drive_target_full12"][1] == knee
        assert ack["independent_policy_residual_requested_full12"] == list(residual)
        assert ack["capture_assist_evidence"]["knee_hold_final_verified"]
        nominal_history.append(ack["nominal_command_servo_deg"][1])
        actuation=build_residual_actuation_plan(tuple(a+b for a,b in zip(nominal,residual)),
            frozen_nominal_full12=nominal,drive_feedback_bias_full12=ZERO,normal_drive_bias_full12=ZERO)
        audit=build_actuator_target_effect_audit(adapter=adapter,actuation=actuation,raw_ack=ack,
            previous_final_drive_servo_deg=previous,source_phase_id="P05",policy_request=None)
        assert audit["verified"] and audit["capture_assist_state_transition_independently_reconstructed"]
        assert audit["changed_channels_full12"][:2] == [False,False]
        assert audit["counterfactual_scope"] == "same_pre_tick_state_and_same_capture_assist_without_current_ppo_residual"
    assert nominal_history[-1] > nominal_history[0]  # no assist feedback into nominal history


def test_reset_new_assist_has_no_old_recovery_targets():
    assist=HipOnlyCaptureAssist();advance(assist,1)
    fresh=HipOnlyCaptureAssist()
    assert capture_assist_features(fresh.snapshot()) == (0.,)*12


def test_inactive_assist_zero_has_exact_existing_targets_and_nominal_history():
    plain, enabled = _adapter(), _adapter()
    assist=HipOnlyCaptureAssist()
    for tick in range(1,80):
        nominal=(10.,-8.)+(3.,)*6+(.2,)*4
        args=dict(command=nominal,physics_tick=tick,tracking_servo_names=(),
            controller_bias_full12=(1.,)*8+ZERO[8:],projected_residual_full12=ZERO)
        before=apply_semantic_residual(plain,**args)
        after=apply_semantic_residual(enabled,**args,capture_assist=assist,
            capture_assist_context=context(tick,stage_id="P01",source_unfold_dispatched=False))
        assert before["drive_target_full12"] == after["drive_target_full12"]
        assert before["nominal_command_servo_deg"] == after["nominal_command_servo_deg"]
        assert after["capture_assist_evidence"]["owner_indices"] == []


def test_source_endpoint_must_have_reached_previous_real_dispatch():
    legs={leg:dict(support=True,bearing_verified=True) for leg in ("FR","RR","RL")}
    legs["FL"]=dict(within_top_xy=True,clearance_m=.008)
    task={"physical_evaluator":dict(valid=True,current_legs=legs,physics_tick=1,
        history={"active_lift":{"FL":True},"front_edge_crossed":{"FL":True},"placed":{"FL":False}})}
    frame=SimpleNamespace(state_id="P05",endpoint_issued=True,full12=(10.,-8.)+ZERO[2:])
    args=dict(task=task,observation={"joints":{SERVO_ORDER[0]:{"position_deg":10.}}},
        source_frame=frame,nominal_provider=None,physics_tick=2)
    missing=capture_assist_context(**args,previous_ack={"requested_full12":ZERO})
    present=capture_assist_context(**args,previous_ack={"requested_full12":frame.full12})
    assert not missing["source_unfold_dispatched"] and present["source_unfold_dispatched"]
