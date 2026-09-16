"""Frozen dispatch/audit seam, including real geometry-helper integration."""
from __future__ import annotations

import copy
import math
import sys
from types import ModuleType

import pytest
import torch

from test_actuator_target_effect import _adapter
from test_semantic_residual_adapter import plan
from wlr50_clean.infrastructure.command_batch import SERVO_ORDER
from wlr50_clean.infrastructure.robot_adapter import RobotAdapterError, bounded_drive_feedback_step
from wlr50_clean.ppo.actuator_target_effect import (
    ActuatorTargetEffectError, build_actuator_target_effect_audit,
)
from wlr50_clean.ppo.isaac_fsm_backend import IsaacFSMBackend
from wlr50_clean.ppo.semantic_residual_adapter import SemanticActuationDispatch, apply_semantic_residual


ZERO = (0.,) * 12
RR_HIP = 6


def full(value, channel=RR_HIP):
    result = list(ZERO)
    result[channel] = value
    return tuple(result)


def valid_context(tick=21):
    return {"dispatch_physics_tick": tick, "canonical_servo_indices": (6, 7)}


def dispatch(adapter, actuation, tick, context=None):
    return IsaacFSMBackend._atomic_apply(
        None, SemanticActuationDispatch(adapter, actuation, nominal_geometry_context=context),
        actuation.frozen_nominal_full12, physics_tick=tick, tracking_servo_names=(),
        drive_feedback_bias_full12=actuation.combined_post_mapper_bias_full12)


def saturated_history():
    adapter = _adapter()
    # Real prior commands/residuals leave native=20 but actual final=0. There
    # is no artificial mapper implementation or second history in dispatch.
    for tick in range(1, 21):
        dispatch(adapter, plan(full(-20.), nominal=full(20.)), tick)
    assert adapter.last_ack["native_drive_target_full12"][RR_HIP] == 20.
    assert adapter._final_drive_servo_deg[SERVO_ORDER[RR_HIP]] == 0.
    return adapter


def install_helper(monkeypatch, callback):
    module = ModuleType("wlr50_clean.ppo.semantic_nominal_geometry")
    module.correct_nominal_geometry = callback
    monkeypatch.setitem(sys.modules, module.__name__, module)


def audit(adapter, actuation, ack, previous):
    return build_actuator_target_effect_audit(
        adapter=adapter, actuation=actuation, raw_ack=ack,
        previous_final_drive_servo_deg=previous, source_phase_id="P09", policy_request=None)


@pytest.mark.parametrize("controller", [0., 2.])
@pytest.mark.parametrize("residual, expected, policy_effect", [(0., 0., False), (.5, .5, True),
                                                              (2., 1.25, True), (1.e-12, 1.e-12, False)])
def test_active_inverse_composition_one_advance_write_and_three_targets(monkeypatch, controller, residual,
                                                                      expected, policy_effect):
    adapter = saturated_history()
    context = valid_context()
    previous = tuple(adapter._final_drive_servo_deg.values())
    calls = []

    def geometry(*, adapter, native_full12, controller_bias_full12, context, nominal_previous_servo_deg):
        calls.append((tuple(native_full12), tuple(controller_bias_full12), context))
        adjusted = list(native_full12)
        # Desired no-current-policy final=0. Invert the full raw nominal+c,
        # not the previously bounded 1.25-degree target.
        adjusted[RR_HIP] = -controller_bias_full12[RR_HIP]
        return tuple(adjusted), {"status": "projected", "desired_nominal_final_deg": 0.}

    install_helper(monkeypatch, geometry)
    advance = adapter.servo_target_mapper.advance
    advances = []

    def counted(*args, **kwargs):
        advances.append(tuple(args[0]))
        return advance(*args, **kwargs)

    monkeypatch.setattr(adapter.servo_target_mapper, "advance", counted)
    event_count = len(adapter.robot.events)
    actuation = plan(full(residual), full(controller), full(20.))
    ack = dispatch(adapter, actuation, 21, context)
    assert advances == [full(20.)[:8]]
    assert len(calls) == 1 and calls[0] == (full(20.), full(controller), context)
    assert adapter.robot.events[event_count:] == ["position.setter", "velocity.setter", "dispatch"]
    assert adapter.write_count == 21 and ack["articulation_writes_this_call"] == 1
    assert ack["native_drive_target_full12"] == list(full(20.))
    assert ack["servo_native_drive_command_deg"] == list(full(20.)[:8])
    assert ack["geometry_adjusted_native_full12"] == list(full(-controller))
    assert ack["nominal_geometry_adjustment_full12"] == list(full(-20.-controller))
    assert ack["drive_target_full12"][RR_HIP] == pytest.approx(expected, abs=1.e-14)
    assert ack["drive_feedback_bias_requested_full12"] == list(actuation.combined_post_mapper_bias_full12)
    assert ack["bounded_controller_bias_requested_full12"] == list(full(controller))
    assert ack["independent_policy_residual_requested_full12"] == list(actuation.projected_residual_full12)
    assert tuple(adapter._final_drive_servo_deg.values()) == tuple(ack["drive_target_full12"][:8])

    effect = audit(adapter, actuation, ack, previous)
    assert effect["verified"] and effect["setter_dispatch_targets_equal"]
    assert effect["geometry_nominal_native_targets"] == effect["counterfactual_native_targets"]
    assert effect["geometry_nominal_native_targets"]["servo_position_rad"][RR_HIP] == .75
    assert effect["raw_nominal_native_targets"]["servo_position_rad"][RR_HIP] == pytest.approx(
        float(torch.tensor(.75-math.radians(1.25), dtype=torch.float32)))
    assert effect["nominal_geometry_changed_target_channel_count"] == 1
    assert effect["nominal_geometry_changed_channels_full12"] == [i == RR_HIP for i in range(12)]
    assert effect["changed_target_channel_count"] == int(policy_effect)
    assert effect["nominal_geometry_native_target_delta"]["servo_position_rad"][RR_HIP] > 0.
    # No audit advances the mapper or stages a second target.
    assert len(advances) == 1 and len(adapter.robot.events) == event_count+3


@pytest.mark.parametrize("status", ["unchanged", "degraded_bypass"])
@pytest.mark.parametrize("residual", [0., -.5, 2.])
def test_identity_and_degraded_keep_raw_native_and_old_residual_mapping(monkeypatch, status, residual):
    adapter, reference = saturated_history(), saturated_history()

    def geometry(*, adapter, native_full12, controller_bias_full12, context, nominal_previous_servo_deg):
        return native_full12, {"status": status}

    install_helper(monkeypatch, geometry)
    actuation = plan(full(residual), nominal=full(20.))
    previous = tuple(adapter._final_drive_servo_deg.values())
    ack = dispatch(adapter, actuation, 21, valid_context())
    original = dispatch(reference, actuation, 21)
    for key, value in original.items():
        assert ack[key] == value
    assert ack["geometry_adjusted_native_full12"] == list(full(20.))
    assert ack["nominal_geometry_adjustment_full12"] == list(ZERO)
    assert ack["drive_target_full12"][RR_HIP] == 1.25
    assert torch.equal(adapter.robot._joint_pos_target_sim, reference.robot._joint_pos_target_sim)
    effect = audit(adapter, actuation, ack, previous)
    assert effect["nominal_geometry_changed_target_channel_count"] == 0
    old_effect = audit(reference, actuation, original, previous)
    for key, value in old_effect.items():
        assert effect[key] == value
    assert not any("geometry" in key for key in original)
    assert not any("geometry" in key for key in old_effect)


def test_bounded_delta_is_not_a_correct_inverse_and_identity_must_not_recenter():
    def bound(native, residual=0.):
        return bounded_drive_feedback_step(previous_deg=0., native_deg=native, bias_deg=residual,
            maximum_delta_deg=1.25, lower_deg=-135., upper_deg=135.)
    assert bound(20.) == 1.25
    assert bound(20.+(0.-1.25)) == 1.25  # Wrong delta is swallowed by slew.
    assert bound(20.+(0.-20.)) == 0.
    assert bound(20., -.5) == 1.25
    assert bound(1.25, -.5) == .75  # Re-centering identity changes the old map.


@pytest.mark.parametrize("bad_channel", [0, 4, 8])
def test_geometry_cannot_change_front_other_rear_or_wheel_channels(monkeypatch, bad_channel):
    adapter = _adapter()

    def geometry(**kwargs):
        return full(1., bad_channel), {"status": "invalid_fixture"}

    install_helper(monkeypatch, geometry)
    with pytest.raises(RobotAdapterError, match="declared rear"):
        dispatch(adapter, plan(ZERO), 1, valid_context(1))
    assert adapter.write_count == 0 and adapter.robot.events == []


def test_geometry_does_not_bypass_original_controller_bias_cap(monkeypatch):
    install_helper(monkeypatch, lambda **kwargs: pytest.fail("helper must not run"))
    adapter = _adapter()
    with pytest.raises(RobotAdapterError, match="bounded mapper envelope"):
        apply_semantic_residual(adapter, ZERO, physics_tick=1, tracking_servo_names=(),
            controller_bias_full12=full(10.01), projected_residual_full12=full(-40.),
            nominal_geometry_context=valid_context(1))
    assert adapter.servo_target_mapper.feedback_tick == 0 and adapter.robot.events == []


@pytest.mark.parametrize("tamper", ["missing_adjustment", "wrong_adjustment", "wrong_corrected", "missing_proof"])
def test_audit_rejects_incomplete_or_inconsistent_geometry_records(monkeypatch, tamper):
    adapter = saturated_history()
    install_helper(monkeypatch, lambda **kwargs: (ZERO, {"status": "projected"}))
    previous = tuple(adapter._final_drive_servo_deg.values())
    actuation = plan(full(.5), nominal=full(20.))
    ack = copy.deepcopy(dispatch(adapter, actuation, 21, valid_context()))
    if tamper == "missing_adjustment": del ack["nominal_geometry_adjustment_full12"]
    elif tamper == "wrong_adjustment": ack["nominal_geometry_adjustment_full12"][RR_HIP] = -1.25
    elif tamper == "missing_proof": del ack["nominal_geometry_evidence"]
    else:
        ack["geometry_adjusted_native_full12"][RR_HIP] = .25
        ack["nominal_geometry_adjustment_full12"][RR_HIP] = -19.75
    with pytest.raises(ActuatorTargetEffectError):
        audit(adapter, actuation, ack, previous)


@pytest.mark.parametrize("context_tick", [None, False, True, 1., 0, 2])
def test_missing_invalid_or_stale_context_tick_never_advances_or_writes(monkeypatch, context_tick):
    adapter = _adapter()
    install_helper(monkeypatch, lambda **kwargs: pytest.fail("helper must not run"))
    monkeypatch.setattr(adapter.servo_target_mapper, "advance",
                        lambda *args, **kwargs: pytest.fail("mapper must not advance"))
    context = valid_context(1)
    if context_tick is None:
        del context["dispatch_physics_tick"]
    else:
        context["dispatch_physics_tick"] = context_tick
    with pytest.raises(RobotAdapterError, match="dispatch physics tick"):
        dispatch(adapter, plan(ZERO), 1, context)
    assert adapter.write_count == 0 and adapter.robot.events == []
    assert adapter._last_physics_tick is None


@pytest.mark.parametrize("indices", [None, (4, 7), (7, 6), (6, 7, 8), (6., 7), (True, 7)])
def test_invalid_declared_pair_rejected_before_mapper(monkeypatch, indices):
    adapter = _adapter()
    install_helper(monkeypatch, lambda **kwargs: pytest.fail("helper must not run"))
    monkeypatch.setattr(adapter.servo_target_mapper, "advance",
                        lambda *args, **kwargs: pytest.fail("mapper must not advance"))
    context = valid_context(1)
    context["canonical_servo_indices"] = indices
    with pytest.raises(RobotAdapterError, match="ordered rear"):
        dispatch(adapter, plan(ZERO), 1, context)
    assert adapter.write_count == 0 and adapter.robot.events == []


@pytest.mark.parametrize("controller", [0., 2.])
@pytest.mark.parametrize("residual", [0., .5, -.5, 2., 1.e-12])
def test_real_geometry_projection_mapper_dispatch_and_three_branch_audit(controller, residual):
    # No helper/mapper/projection mock. The local J is an explicit synthetic
    # model evaluated at actual standing q; live capture/COM identities have
    # separate coverage in test_semantic_nominal_geometry_context.py.
    from wlr50_clean.ppo.semantic_nominal_geometry import CONTEXT_SCHEMA, MODE

    # Start with true zero nominal history, not a cancelled +20 nominal whose
    # previous final=0 used to leak the policy back into geometry. That distinct
    # history case is covered by the policy-excluded-history regression tests.
    adapter, zero_policy, raw_reference = (_adapter() for _ in range(3))
    for item in (adapter, zero_policy, raw_reference):
        for tick in range(1, 21):
            dispatch(item, plan(ZERO), tick)
    nominal = (3., -2., .25, -.25, .4, -.4, 20., .5, .1, -.2, .3, -.4)
    policy = (residual*.2, 0., 0., 0., 0., 0., residual, 0., residual*.1, 0., 0., 0.)
    controller_bias = full(controller)
    context = {
        "schema": CONTEXT_SCHEMA, "mode": MODE, "source_phase_id": "P09",
        "source_control_tick": 20, "source_sim_time_s": 20./120.,
        "dispatch_physics_tick": 21, "active_leg": "RR", "canonical_servo_indices": (6, 7),
        "physical_q_rad": tuple(float(adapter.robot.data.joint_pos[0, adapter.joint_map.servo_ids[i]])
                                for i in (6, 7)),
        "jacobian_x_m_per_rad": (0., -1.), "jacobian_z_m_per_rad": (1., 0.),
        "clearance_m": .015, "clearance_margin_m": .015,
        "place_xy": False, "ground_contact": False,
        "physical_motion_guaranteed": False,
    }
    original_context = copy.deepcopy(context)
    previous = tuple(adapter._final_drive_servo_deg.values())
    actuation = plan(policy, controller_bias, nominal)
    ack = dispatch(adapter, actuation, 21, context)
    zero_actuation = plan(ZERO, controller_bias, nominal)
    zero_ack = dispatch(zero_policy, zero_actuation, 21, context)
    raw_ack = dispatch(raw_reference, actuation, 21)

    assert context == original_context
    assert adapter.servo_target_mapper.feedback_tick == 21
    assert adapter.write_count == 21 and ack["articulation_writes_this_call"] == 1
    assert adapter.robot.events == ["position.setter", "velocity.setter", "dispatch"]*21
    assert ack["native_drive_target_full12"] == raw_ack["native_drive_target_full12"]
    assert ack["servo_native_drive_command_deg"] == raw_ack["servo_native_drive_command_deg"]
    assert ack["native_drive_target_full12"][RR_HIP] == 1.25
    assert ack["nominal_geometry_adjustment_full12"] == zero_ack["nominal_geometry_adjustment_full12"]
    assert ack["geometry_adjusted_native_full12"] == zero_ack["geometry_adjusted_native_full12"]
    assert ack["geometry_adjusted_native_full12"][RR_HIP] == pytest.approx(-controller, abs=1.e-10)
    assert ack["nominal_geometry_adjustment_full12"][RR_HIP] == pytest.approx(-1.25-controller, abs=1.e-10)
    assert zero_ack["drive_target_full12"][RR_HIP] == pytest.approx(0., abs=1.e-10)
    assert ack["drive_target_full12"][RR_HIP] == pytest.approx(max(-1.25, min(1.25, residual)), abs=1.e-10)
    assert ack["drive_feedback_bias_requested_full12"] == list(actuation.combined_post_mapper_bias_full12)
    assert ack["independent_policy_residual_requested_full12"] == list(actuation.projected_residual_full12)
    evidence = ack["nominal_geometry_evidence"]
    assert evidence["current_policy_residual_used"] is False
    assert evidence["physical_motion_guaranteed"] is False
    assert evidence["projection"]["mathematical_contract_verified"] is True
    # The independent x row is knee-only. Its nominal and current residual
    # effects, plus every other leg and all four wheels, remain unchanged.
    for i in range(12):
        if i != RR_HIP:
            assert ack["drive_target_full12"][i] == pytest.approx(raw_ack["drive_target_full12"][i], abs=1.e-10)
    other_servo_ids = [adapter.joint_map.servo_ids[i] for i in range(8) if i != RR_HIP]
    assert torch.equal(adapter.robot._joint_pos_target_sim[:, other_servo_ids],
                       raw_reference.robot._joint_pos_target_sim[:, other_servo_ids])
    assert torch.equal(adapter.robot._joint_vel_target_sim, raw_reference.robot._joint_vel_target_sim)

    effect = audit(adapter, actuation, ack, previous)
    zero_effect = audit(zero_policy, zero_actuation, zero_ack, previous)
    assert effect["verified"] and effect["actual_mapping_matches_dispatch"]
    assert effect["geometry_nominal_native_targets"] == zero_effect["actual_native_targets"]
    assert effect["geometry_nominal_native_targets"] == effect["counterfactual_native_targets"]
    assert effect["nominal_geometry_changed_target_channel_count"] == 1
    assert zero_effect["changed_target_channel_count"] == 0
    assert effect["raw_nominal_native_targets"]["servo_position_rad"][RR_HIP] == pytest.approx(
        float(torch.tensor(.75-math.radians(1.25), dtype=torch.float32)))
    assert effect["geometry_nominal_native_targets"]["servo_position_rad"][RR_HIP] == .75
    assert effect["changed_channels_full12"][RR_HIP] is (abs(residual) > 1.e-9)
    if residual == 0.:
        assert effect["changed_target_channel_count"] == 0
        assert effect["actual_native_targets"] == effect["geometry_nominal_native_targets"]
    # Audits are read-only: no helper-induced second update/dispatch/history.
    assert adapter.servo_target_mapper.feedback_tick == adapter.write_count == 21
    assert adapter.robot.events == ["position.setter", "velocity.setter", "dispatch"]*21
