"""Real frozen mapper/dispatch and float32 audit for opt-in servo headroom."""
from __future__ import annotations

import copy
import json
import math

import pytest
import torch

from test_actuator_target_effect import _adapter
from test_semantic_residual_adapter import plan
from test_semantic_nominal_geometry_dispatch import saturated_history
from wlr50_clean.infrastructure.command_batch import SERVO_ORDER, servo_limits_deg
from wlr50_clean.infrastructure.robot_adapter import RobotAdapterError
from wlr50_clean.ppo.actuator_target_effect import ActuatorTargetEffectError, build_actuator_target_effect_audit
from wlr50_clean.ppo.isaac_fsm_backend import IsaacFSMBackend
from wlr50_clean.ppo.semantic_headroom import HEADROOM_MODE, project_semantic_servo_headroom
from wlr50_clean.ppo.semantic_prefix import DispatchReceipt
from wlr50_clean.ppo.semantic_residual_adapter import SemanticActuationDispatch, apply_semantic_residual


ZERO = (0.,)*12


def full(value, channel=7):
    result = list(ZERO)
    result[channel] = value
    return tuple(result)


def dispatch(adapter, actuation, tick, *, mode=HEADROOM_MODE, geometry=None, tracking=()):
    return IsaacFSMBackend._atomic_apply(
        None, SemanticActuationDispatch(adapter, actuation,
            nominal_geometry_context=geometry, policy_headroom_mode=mode),
        actuation.frozen_nominal_full12, physics_tick=tick, tracking_servo_names=tracking,
        drive_feedback_bias_full12=actuation.combined_post_mapper_bias_full12)


def audit(adapter, actuation, ack, previous, *, mode=HEADROOM_MODE):
    return build_actuator_target_effect_audit(adapter=adapter, actuation=actuation, raw_ack=ack,
        previous_final_drive_servo_deg=previous, source_phase_id="P09", policy_request=None,
        policy_headroom_mode=mode)


def warm(nominal, controller=ZERO, *, tracking=(), measured_offset_deg=None):
    adapter = _adapter()
    if measured_offset_deg is not None:
        joint = adapter.joint_map.servo_ids[7]
        adapter.robot.data.joint_pos[0, joint] = .75 + math.radians(measured_offset_deg)
    for tick in range(1, 181):
        dispatch(adapter, plan(ZERO, controller, nominal), tick, mode=None, tracking=tracking)
    return adapter


@pytest.mark.parametrize("requested", [-24.98, -22.71])
def test_real_tracking_mapper_recovers_lost_native_headroom_without_new_mapper_input(requested):
    nominal = full(-37.8)
    tracking = (SERVO_ORDER[7],)
    adapter = warm(nominal, tracking=tracking, measured_offset_deg=60.)
    assert adapter.last_ack["native_drive_target_full12"][7] == pytest.approx(-27.8)
    previous = tuple(adapter._final_drive_servo_deg.values())
    actuation = plan(full(requested), nominal=nominal)
    ack = dispatch(adapter, actuation, 181, tracking=tracking)
    proof = ack["policy_headroom_evidence"]
    assert ack["applied_full12"] == list(nominal)
    assert proof["effective_policy_residual_full12"][7] == pytest.approx(requested)
    assert proof["policy_residual_intervals_servo_deg"][7][0] == pytest.approx(-30.2)
    assert proof["candidate_native_target_before_final_slew_full12"][7] == pytest.approx(-27.8+requested)
    # The recovered static target is NOT claimed to execute in one tick.
    assert ack["drive_target_full12"][7] == pytest.approx(previous[7]-1.25)
    assert ack["drive_feedback_bias_requested_full12"] == list(actuation.combined_post_mapper_bias_full12)
    assert ack["independent_policy_residual_requested_full12"] == list(actuation.projected_residual_full12)
    assert adapter.servo_target_mapper.feedback_tick == adapter.write_count == 181
    assert audit(adapter, actuation, ack, previous)["verified"]


@pytest.mark.parametrize("channel", range(8))
@pytest.mark.parametrize("side", ["lower", "upper"])
def test_zero_outside_reserved_band_preserves_baseline_and_native_conversion(channel, side):
    lo, hi = servo_limits_deg(SERVO_ORDER[channel])
    nominal = full(lo+1. if side == "lower" else hi-1., channel)
    adapter, reference = warm(nominal), warm(nominal)
    previous = tuple(adapter._final_drive_servo_deg.values())
    actuation = plan(ZERO, nominal=nominal)
    ack = dispatch(adapter, actuation, 181)
    old = dispatch(reference, actuation, 181, mode=None)
    for key, value in old.items():
        assert ack[key] == value
    assert ack["policy_headroom_evidence"]["effective_policy_residual_full12"] == list(ZERO)
    assert channel in ack["policy_headroom_evidence"]["baseline_outside_reserved_servo_indices"]
    assert torch.equal(adapter.robot._joint_pos_target_sim, reference.robot._joint_pos_target_sim)
    assert torch.equal(adapter.robot._joint_vel_target_sim, reference.robot._joint_vel_target_sim)
    effect = audit(adapter, actuation, ack, previous)
    assert effect["changed_target_channel_count"] == 0
    assert effect["actual_native_targets"] == effect["counterfactual_native_targets"]


@pytest.mark.parametrize("residual,expected_effective", [(-.5, 0.), (0., 0.), (.5, .5), (1.e-12, 1.e-12)])
def test_controller_included_and_inward_small_request_not_relocated(residual, expected_effective):
    nominal, controller = full(-56.), full(-3.)
    adapter = warm(nominal, controller)
    previous = tuple(adapter._final_drive_servo_deg.values())
    actuation = plan(full(residual), controller, nominal)
    ack = dispatch(adapter, actuation, 181)
    proof = ack["policy_headroom_evidence"]
    assert proof["baseline_native_plus_controller_full12"][7] == -59.
    assert proof["effective_policy_residual_full12"][7] == pytest.approx(expected_effective, abs=5.e-15)
    assert ack["drive_target_full12"][7] == pytest.approx(-59.+expected_effective, abs=5.e-14)
    effect = audit(adapter, actuation, ack, previous)
    assert effect["changed_target_channel_count"] == int(residual == .5)
    # Prefix receipt must still bind the original c + requested residual.
    receipt = DispatchReceipt.from_ack(source_tick=180, command=nominal, tracking=(),
        bias=actuation.combined_post_mapper_bias_full12, ack=ack)
    assert receipt.bias == actuation.combined_post_mapper_bias_full12


def test_outward_request_rejected_by_native_reserve_but_old_mode_unchanged():
    nominal = full(-58.)
    adapter, old_adapter = warm(nominal), warm(nominal)
    previous = tuple(adapter._final_drive_servo_deg.values())
    actuation = plan(full(-12.), nominal=nominal)
    ack = dispatch(adapter, actuation, 181)
    old = dispatch(old_adapter, actuation, 181, mode=None)
    assert ack["drive_target_full12"][7] == -58.
    assert old["drive_target_full12"][7] == -59.25
    assert ack["policy_headroom_evidence"]["effective_policy_residual_full12"][7] == 0.
    assert ack["independent_policy_residual_requested_full12"][7] == -12.
    assert audit(adapter, actuation, ack, previous)["changed_target_channel_count"] == 0
    old_effect = audit(old_adapter, actuation, old, previous, mode=None)
    assert not any("headroom" in key for key in old)
    assert not any("headroom" in key for key in old_effect)


def test_zero_without_geometry_calls_original_adapter_before_readonly_headroom(monkeypatch):
    from wlr50_clean.ppo import semantic_headroom
    adapter = _adapter()
    original_apply = adapter.apply_full12
    original_project = semantic_headroom.project_semantic_servo_headroom
    calls = []

    def original(*args, **kwargs):
        calls.append("original_apply")
        return original_apply(*args, **kwargs)

    def headroom(*args, **kwargs):
        assert adapter.write_count == adapter.servo_target_mapper.feedback_tick == 1
        assert adapter.robot.events == ["position.setter", "velocity.setter", "dispatch"]
        calls.append("readonly_receipt")
        return original_project(*args, **kwargs)

    monkeypatch.setattr(adapter, "apply_full12", original)
    monkeypatch.setattr(semantic_headroom, "project_semantic_servo_headroom", headroom)
    actuation = plan(ZERO, full(2.), full(-37.8))
    ack = dispatch(adapter, actuation, 1)
    assert calls == ["original_apply", "readonly_receipt"]
    assert ack["policy_headroom_evidence"]["geometry_corrected_native_full12"] == ack["native_drive_target_full12"]


@pytest.mark.parametrize("policy", [.5, 134., 1.e-12])
def test_real_geometry_precedes_headroom_three_branches_and_wheels_unchanged(policy):
    from wlr50_clean.ppo.semantic_nominal_geometry import CONTEXT_SCHEMA, MODE
    adapter, zero_adapter, reference = (saturated_history() for _ in range(3))
    nominal = full(20., 6)
    controller = full(2., 6)
    residual = full(policy, 6)[:8] + (.1, -.2, .3, -.4)
    context = {"schema": CONTEXT_SCHEMA, "mode": MODE, "source_phase_id": "P09",
        "source_control_tick": 20, "source_sim_time_s": 20./120.,
        "dispatch_physics_tick": 21, "active_leg": "RR", "canonical_servo_indices": (6, 7),
        "physical_q_rad": tuple(float(adapter.robot.data.joint_pos[0, adapter.joint_map.servo_ids[i]])
                                for i in (6, 7)),
        "jacobian_x_m_per_rad": (0., -1.), "jacobian_z_m_per_rad": (1., 0.),
        "clearance_m": .015, "clearance_margin_m": .015,
        "place_xy": False, "ground_contact": False, "physical_motion_guaranteed": False}
    before = copy.deepcopy(context)
    previous = tuple(adapter._final_drive_servo_deg.values())
    actuation = plan(residual, controller, nominal)
    ack = dispatch(adapter, actuation, 21, geometry=context)
    zero_plan = plan(ZERO, controller, nominal)
    zero_ack = dispatch(zero_adapter, zero_plan, 21, geometry=context)
    old = dispatch(reference, actuation, 21, geometry=context, mode=None)
    assert context == before
    assert ack["nominal_geometry_evidence"] == zero_ack["nominal_geometry_evidence"]
    assert ack["policy_headroom_evidence"]["geometry_corrected_native_full12"][6] == pytest.approx(-2.)
    assert ack["policy_headroom_evidence"]["baseline_native_plus_controller_full12"][6] == pytest.approx(0.)
    assert ack["policy_headroom_evidence"]["effective_policy_residual_full12"][6] == pytest.approx(min(133.,policy))
    assert ack["drive_target_full12"][6] == pytest.approx(min(1.25,policy))
    assert torch.equal(adapter.robot._joint_vel_target_sim, reference.robot._joint_vel_target_sim)
    assert ack["drive_target_full12"][8:] == old["drive_target_full12"][8:]
    effect = audit(adapter, actuation, ack, previous)
    zero_effect = audit(zero_adapter, zero_plan, zero_ack, previous)
    assert effect["geometry_nominal_native_targets"] == zero_effect["actual_native_targets"]
    assert effect["nominal_geometry_changed_channels_full12"][6] is True
    assert effect["changed_channels_full12"][6] is (policy > 1.e-9)
    assert zero_effect["changed_target_channel_count"] == 0
    assert adapter.servo_target_mapper.feedback_tick == adapter.write_count == 21
    assert adapter.robot.events == ["position.setter", "velocity.setter", "dispatch"]*21


@pytest.mark.parametrize("mode", ["unknown", True, 1])
def test_invalid_mode_never_advances_or_writes(mode):
    adapter = _adapter()
    with pytest.raises(RobotAdapterError, match="headroom mode"):
        dispatch(adapter, plan(ZERO), 1, mode=mode)
    assert adapter.servo_target_mapper.feedback_tick == adapter.write_count == 0
    assert adapter.robot.events == []


def test_headroom_does_not_relax_controller_envelope():
    adapter = _adapter()
    with pytest.raises(RobotAdapterError, match="bounded mapper envelope"):
        apply_semantic_residual(adapter, ZERO, physics_tick=1, tracking_servo_names=(),
            controller_bias_full12=full(10.01), projected_residual_full12=full(-50.),
            policy_headroom_mode=HEADROOM_MODE)
    assert adapter.servo_target_mapper.feedback_tick == adapter.write_count == 0


@pytest.mark.parametrize("tamper", ["missing_mode", "wrong_mode", "missing_evidence", "request",
    "effective", "combined", "interval", "native", "reserve", "extra", "bool", "nan",
    "old_combined", "old_controller", "old_request"])
def test_audit_rejects_unbound_headroom_and_request_fields(tamper):
    adapter = _adapter()
    actuation = plan(full(-24.))
    previous = tuple(adapter._final_drive_servo_deg.values())
    ack = copy.deepcopy(dispatch(adapter, actuation, 1))
    proof = ack["policy_headroom_evidence"]
    if tamper == "missing_mode": del ack["policy_headroom_mode"]
    elif tamper == "wrong_mode": ack["policy_headroom_mode"] = "old"
    elif tamper == "missing_evidence": del ack["policy_headroom_evidence"]
    elif tamper == "request": proof["requested_policy_residual_full12"][7] += 1.
    elif tamper == "effective": proof["effective_policy_residual_full12"][7] += 1.
    elif tamper == "combined": proof["effective_combined_post_mapper_bias_full12"][7] += 1.
    elif tamper == "interval": proof["policy_residual_intervals_servo_deg"][7][0] -= 1.
    elif tamper == "native": proof["geometry_corrected_native_full12"][7] += 1.
    elif tamper == "reserve": proof["servo_reserve_deg"] = 0.
    elif tamper == "extra": proof["unreviewed"] = True
    elif tamper == "bool": proof["effective_policy_residual_full12"][0] = False
    elif tamper == "nan": proof["effective_policy_residual_full12"][7] = float("nan")
    elif tamper == "old_combined": ack["drive_feedback_bias_requested_full12"][7] += 1.
    elif tamper == "old_controller": ack["bounded_controller_bias_requested_full12"][7] += 1.
    else: ack["independent_policy_residual_requested_full12"][7] += 1.
    with pytest.raises(ActuatorTargetEffectError):
        audit(adapter, actuation, ack, previous)


def test_expected_mode_is_required_and_json_roundtrip_is_supported(monkeypatch):
    adapter = _adapter()
    actuation = plan(full(.5))
    previous = tuple(adapter._final_drive_servo_deg.values())
    ack = dispatch(adapter, actuation, 1)
    with pytest.raises(ActuatorTargetEffectError, match="unexpected"):
        audit(adapter, actuation, ack, previous, mode=None)
    for owner, name in ((adapter.servo_target_mapper, "advance"), (adapter, "apply_full12"),
                        (adapter.robot, "set_joint_position_target"),
                        (adapter.robot, "set_joint_velocity_target"), (adapter.robot, "write_data_to_sim")):
        monkeypatch.setattr(owner, name, lambda *args, **kwargs: pytest.fail("audit is read-only"))
    result = audit(adapter, actuation, json.loads(json.dumps(ack)), previous)
    expected = project_semantic_servo_headroom(ack["native_drive_target_full12"],
        actuation.controller_drive_bias_full12, actuation.projected_residual_full12)
    assert result["policy_headroom_evidence"] == expected
    assert adapter.write_count == adapter.servo_target_mapper.feedback_tick == 1


@pytest.mark.parametrize("owner,field", [("data", "joint_pos_target"), ("data", "joint_vel_target"),
                                        ("robot", "_joint_pos_target_sim"), ("robot", "_joint_vel_target_sim")])
def test_new_headroom_mode_still_validates_each_native_float32_buffer(owner, field):
    adapter = _adapter()
    actuation = plan(full(.5))
    previous = tuple(adapter._final_drive_servo_deg.values())
    ack = dispatch(adapter, actuation, 1)
    target = adapter.robot.data if owner == "data" else adapter.robot
    setattr(target, field, getattr(target, field).double())
    with pytest.raises(ActuatorTargetEffectError, match="float32"):
        audit(adapter, actuation, ack, previous)
