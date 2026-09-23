"""Final dispatch/replay tests, not evidence of physical task success."""
import pytest

from test_actuator_target_effect import _adapter
from test_semantic_capture_assist import context as fl_context
from test_semantic_rr_capture_assist import context as rr_context
from wlr50_clean.infrastructure.command_batch import SERVO_ORDER
from wlr50_clean.ppo.semantic_capture_assist import HipOnlyCaptureAssist
from wlr50_clean.ppo.semantic_rr_capture_assist import RRHipOnlyCaptureAssist
from wlr50_clean.ppo.semantic_residual_adapter import apply_semantic_residual
from wlr50_clean.ppo.isaac_fsm_backend import build_residual_actuation_plan
from wlr50_clean.ppo.actuator_target_effect import build_actuator_target_effect_audit
from wlr50_clean.ppo.semantic_tracking_reference import MODE, capture_tracking_reference_context
from wlr50_clean.ppo.semantic_headroom import HEADROOM_MODE

ZERO = (0.,) * 12


@pytest.mark.parametrize("with_fl", [False, True])
def test_rr_final_hold_and_independent_replay_preserve_other_ten_channels(with_fl):
    actual, plain = _adapter(), _adapter()
    for adapter in (actual, plain):
        adapter.apply_full12(ZERO, physics_tick=0, tracking_servo_names=(), drive_feedback_bias_full12=ZERO)
    entry = (10., -8., 0., 0., 0., 0., 10., -8., .2, .3, .4, .5)
    for tick in range(1, 25):
        for adapter in (actual, plain):
            apply_semantic_residual(adapter, entry, physics_tick=tick,
                tracking_servo_names=(), controller_bias_full12=ZERO,
                projected_residual_full12=ZERO, tracking_reference_mode=MODE,
                tracking_reference_bootstrap_tick=1, policy_headroom_mode=HEADROOM_MODE)
    rr = RRHipOnlyCaptureAssist()
    fl = HipOnlyCaptureAssist() if with_fl else None
    knee = actual._final_drive_servo_deg[SERVO_ORDER[7]]
    history = []
    for tick in range(25, 55):
        previous = tuple(actual._final_drive_servo_deg[n] for n in SERVO_ORDER)
        nominal = (20., 20., 3., 3., 3., 3., 20., 20., .2, .3, .4, .5)
        residual = (2., 3., .2, .2, .2, .2, 2., 3., .01, .02, .03, .04)
        plan = build_residual_actuation_plan(tuple(a+b for a,b in zip(nominal, residual)),
            frozen_nominal_full12=nominal, drive_feedback_bias_full12=ZERO,
            normal_drive_bias_full12=ZERO)
        residual = tuple(plan.projected_residual_full12)
        common = dict(command=nominal, physics_tick=tick, tracking_servo_names=(),
            controller_bias_full12=ZERO, projected_residual_full12=residual,
            tracking_reference_mode=MODE, tracking_reference_bootstrap_tick=1, policy_headroom_mode=HEADROOM_MODE)
        independent = capture_tracking_reference_context(actual, physics_tick=tick, bootstrap_physics_tick=1)
        independent["source_tracking_servo_names"] = []
        extra = {} if fl is None else dict(capture_assist=fl,
            capture_assist_context=fl_context(tick, stage_id="P09", source_unfold_dispatched=False))
        ack = apply_semantic_residual(actual, **common, **extra, rr_capture_assist=rr,
            rr_capture_assist_context=rr_context(rr, tick, hip_actual_deg=previous[6],
                knee_actual_deg=previous[7], gap_m=.02-(tick-25)*.00001))
        base = apply_semantic_residual(plain, **common)
        assert ack["drive_target_full12"][7] == knee
        assert all(ack["drive_target_full12"][i] == base["drive_target_full12"][i]
                   for i in range(12) if i not in (6, 7))
        assert ack["independent_policy_residual_requested_full12"] == list(residual)
        history.append(ack["nominal_command_servo_deg"][7])
        audit = build_actuator_target_effect_audit(adapter=actual, actuation=plan,
            raw_ack=ack, previous_final_drive_servo_deg=previous, source_phase_id="P09",
            policy_request=None, tracking_reference_mode=MODE, tracking_reference_context=independent,
            policy_headroom_mode=HEADROOM_MODE)
        assert audit["verified"] and audit["rr_capture_assist_state_transition_independently_reconstructed"]
        assert audit["changed_channels_full12"][6:8] == [False, False]
        assert audit["rr_capture_assist_owned_channels_full12"] == [False]*6+[True, True]+[False]*4
    assert history[-1] > history[0]  # no assist target fed back into nominal mapper
    assert ack["drive_target_full12"][6] < entry[6]


def test_rr_wait_keeps_zero_residual_nominal_wheels_and_all_targets_exact():
    plain, enabled = _adapter(), _adapter()
    for adapter in (plain, enabled):
        adapter.apply_full12(ZERO, physics_tick=0, tracking_servo_names=(), drive_feedback_bias_full12=ZERO)
    rr = RRHipOnlyCaptureAssist()
    for tick in range(1, 40):
        args = dict(command=(3.,)*8+(.2,.3,.4,.5), physics_tick=tick,
            tracking_servo_names=(), controller_bias_full12=ZERO,
            projected_residual_full12=ZERO, tracking_reference_mode=MODE,
            tracking_reference_bootstrap_tick=1, policy_headroom_mode=HEADROOM_MODE)
        base = apply_semantic_residual(plain, **args)
        ack = apply_semantic_residual(enabled, **args, rr_capture_assist=rr,
            rr_capture_assist_context=rr_context(rr, tick, stage_id="P02", crossed_RR=False))
        assert ack["drive_target_full12"] == base["drive_target_full12"]
        assert ack["nominal_command_servo_deg"] == base["nominal_command_servo_deg"]
        assert rr.snapshot()["mode_name"] == "WAIT"
