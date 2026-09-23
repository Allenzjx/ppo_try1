"""Synthetic wiring checks, never physical capture or learning credit."""
from copy import deepcopy

import pytest

from test_actuator_target_effect import _adapter
from wlr50_clean.infrastructure.command_batch import SERVO_ORDER, WHEEL_FORWARD_SIGN, WHEEL_ORDER
from wlr50_clean.ppo.semantic_rr_carry_wheel import MODE, CONTEXT_SCHEMA, SEMANTICS
from wlr50_clean.ppo.semantic_residual_adapter import apply_semantic_residual
from wlr50_clean.ppo.isaac_fsm_backend import build_residual_actuation_plan
from wlr50_clean.ppo.actuator_target_effect import build_actuator_target_effect_audit, ActuatorTargetEffectError

ZERO = (0.,) * 12


def context(tick, previous, *, active=True, action="forward_floor"):
    # The pure builder's source/contact evidence has separate positive/negative
    # tests. Here exercise actual canonical/native dispatch and independent audit.
    return dict(schema=CONTEXT_SCHEMA, mode=MODE, semantics=SEMANTICS,
        dispatch_physics_tick=tick, phase="P09", envelope_active=active,
        action=action if active else "bypass", selected_full12_indices=[8, 9, 10] if active else [],
        forward_floor_rad_s=.25 if action == "forward_floor" else 0.,
        previous_final_wheel_rad_s=list(previous), wheel_rate_rad_s2=1.8,
        source_evidence={"dispatch_physics_tick":tick, "synthetic_source_fixture":True})


def seeded(wheels=(-.93, .06, -.01, -.08)):
    adapter = _adapter()
    apply_semantic_residual(adapter, ZERO[:8]+tuple(wheels), physics_tick=1,
        tracking_servo_names=(), controller_bias_full12=ZERO, projected_residual_full12=ZERO)
    return adapter


def dispatch(adapter, tick, nominal, residual, ctx):
    previous_servo = tuple(adapter._final_drive_servo_deg[n] for n in SERVO_ORDER)
    previous_wheels = tuple(adapter.last_ack["drive_target_full12"][8:])
    plan = build_residual_actuation_plan(tuple(a+b for a,b in zip(nominal,residual)),
        frozen_nominal_full12=nominal, drive_feedback_bias_full12=ZERO, normal_drive_bias_full12=ZERO)
    ack = apply_semantic_residual(adapter, nominal, physics_tick=tick, tracking_servo_names=(),
        controller_bias_full12=ZERO, projected_residual_full12=plan.projected_residual_full12,
        rr_carry_wheel_context=ctx)
    args = dict(adapter=adapter, actuation=plan, raw_ack=ack,
        previous_final_drive_servo_deg=previous_servo, previous_final_drive_wheel_rad_s=previous_wheels,
        source_phase_id="P09", policy_request=None, rr_carry_wheel_context=deepcopy(ctx))
    return ack, args


def test_support_forward_real_dispatch_slew_native_signs_and_counterfactual():
    adapter = seeded()
    for tick in range(2, 87):
        previous = tuple(adapter.last_ack["drive_target_full12"][8:])
        ctx = context(tick, previous)
        nominal = (3.,)*8+(0.,)*4
        residual = (1.,)*8+(-.93,.06,-.01,-.08)
        ack, args = dispatch(adapter,tick,nominal,residual,ctx)
        audit = build_actuator_target_effect_audit(**args)
        assert audit["verified"] and audit["rr_carry_wheel_context_and_previous_FINAL_independently_verified"]
        assert ack["articulation_writes_this_call"] == 1
        assert ack["independent_policy_residual_requested_full12"] == list(residual)
        actual = ack["drive_target_full12"][8:]
        for i in range(3):
            assert abs(actual[i]-previous[i]) <= 1.8/120 + 1e-12
        assert actual[3] == -.08  # AIR RR wheel is not selected as traction.
        assert actual[0] >= previous[0]
        for i,name in enumerate(WHEEL_ORDER):
            native = adapter.robot._joint_vel_target_sim[0,adapter.joint_map.wheel_ids[i]].item()
            assert native == pytest.approx(actual[i]*WHEEL_FORWARD_SIGN[name], abs=1e-7)
    assert all(v > 0 for v in actual[:3])
    assert audit["changed_channels_full12"][8:11] == [False]*3
    assert audit["changed_channels_full12"][11] is True


def test_zero_residual_preserves_existing_forward_nominal():
    adapter = seeded((.3,)*4)
    ack,args = dispatch(adapter,2,ZERO[:8]+(.3,)*4,ZERO,context(2,(.3,)*4))
    assert ack["drive_target_full12"][8:] == [.3]*4
    assert build_actuator_target_effect_audit(**args)["changed_target_channel_count"] == 0


def test_source_stop_bypass_does_not_keep_new_forward_floor_or_slew():
    adapter = seeded((.3,)*4)
    ctx = context(2,(.3,)*4,active=False)
    ack,args = dispatch(adapter,2,ZERO,ZERO,ctx)
    assert ack["drive_target_full12"][8:] == [0.]*4
    assert not ack["rr_carry_wheel_evidence"]["envelope_active"]
    assert build_actuator_target_effect_audit(**args)["verified"]


@pytest.mark.parametrize("corruption", ["missing_previous", "wrong_previous", "wrong_context", "forged_output"])
def test_independent_audit_rejects_corrupt_wheel_proof(corruption):
    adapter = seeded()
    ctx = context(2,(-.93,.06,-.01,-.08))
    ack,args = dispatch(adapter,2,ZERO,ZERO[:8]+(-.93,.06,-.01,-.08),ctx)
    if corruption == "missing_previous":
        args["previous_final_drive_wheel_rad_s"] = None
    elif corruption == "wrong_previous":
        args["previous_final_drive_wheel_rad_s"] = (0.,)*4
    elif corruption == "wrong_context":
        args["rr_carry_wheel_context"]["forward_floor_rad_s"] = .1
    else:
        ack["rr_carry_wheel_evidence"]["output_full12"][8] += .1
    with pytest.raises((ActuatorTargetEffectError, ValueError, TypeError)):
        build_actuator_target_effect_audit(**args)
