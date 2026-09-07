"""CPU integration: real mapper, post-mapper headroom, four float32 buffers.

The articulation is the existing CPU tensor fixture, not an Isaac simulation.
Tracking context is captured independently before dispatch, never from the
current ACK's self-reported evidence. No production helper is mocked.
"""
from __future__ import annotations

import copy
import json
import math

import pytest
import torch

from test_actuator_target_effect import _adapter
from test_semantic_residual_adapter import plan
from wlr50_clean.infrastructure.command_batch import SERVO_COMMAND_SIGN, SERVO_ORDER, servo_limits_deg
from wlr50_clean.infrastructure.robot_adapter import RobotAdapterError
from wlr50_clean.ppo.actuator_target_effect import (
    ActuatorTargetEffectError, actuator_target_audit_request, build_actuator_target_effect_audit,
)
from wlr50_clean.ppo.isaac_fsm_backend import IsaacFSMBackend
from wlr50_clean.ppo.semantic_headroom import HEADROOM_MODE
from wlr50_clean.ppo.semantic_residual_adapter import SemanticActuationDispatch, apply_semantic_residual
from wlr50_clean.ppo.semantic_tracking_reference import (
    capture_tracking_reference_context, build_tracking_reference,
)

MODE = "previous_ack_requested_servo_reference_v1"
ZERO = (0.,) * 12
BOOTSTRAP_TICK = 1


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    # Scope is this pytest process; no parent shell/device setting is modified.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def full(value, channel=7):
    values = list(ZERO)
    values[channel] = value
    return tuple(values)


def adapter():
    result = _adapter()
    # The live backend also requires an explicit prior, ordinary zero prime ACK.
    result.apply_full12(ZERO, physics_tick=0, tracking_servo_names=(),
                        drive_feedback_bias_full12=ZERO)
    return result


def buffers(a):
    return tuple(t.detach().clone() for t in (
        a.robot.data.joint_pos_target, a.robot.data.joint_vel_target,
        a.robot._joint_pos_target_sim, a.robot._joint_vel_target_sim,
    ))


def context(a, tick):
    return capture_tracking_reference_context(
        a, physics_tick=tick, bootstrap_physics_tick=BOOTSTRAP_TICK)


def dispatch(a, p, tick, *, tracking=(), mode=MODE):
    return IsaacFSMBackend._atomic_apply(
        None, SemanticActuationDispatch(
            a, p, policy_headroom_mode=HEADROOM_MODE,
            tracking_reference_mode=mode,
            tracking_reference_bootstrap_tick=BOOTSTRAP_TICK),
        p.frozen_nominal_full12, physics_tick=tick, tracking_servo_names=tracking,
        drive_feedback_bias_full12=p.combined_post_mapper_bias_full12)


def audit(a, p, ack, before, ctx, *, phase="P09", mode=MODE):
    return build_actuator_target_effect_audit(
        adapter=a, actuation=p, raw_ack=ack,
        previous_final_drive_servo_deg=before, source_phase_id=phase,
        policy_request=actuator_target_audit_request(phase, p.projected_residual_full12, (1,) * 12),
        policy_headroom_mode=HEADROOM_MODE,
        tracking_reference_mode=mode, tracking_reference_context=ctx)


def checked_step(a, p, tick, *, tracking=(), phase="P09"):
    ctx = context(a, tick)
    # This is independently bound to the source frame, not to current ACK names.
    ctx["source_tracking_servo_names"] = list(tracking)
    before = tuple(a._final_drive_servo_deg.values())
    frozen_ctx = copy.deepcopy(ctx)
    ack = dispatch(a, p, tick, tracking=tracking)
    helper_context = {key: value for key, value in ctx.items() if key != "source_tracking_servo_names"}
    proof = build_tracking_reference(helper_context, requested_command_deg=p.frozen_nominal_full12[:8],
                                     tracking_servo_names=tracking)
    assert ack["tracking_reference_mode"] == MODE
    assert ack["tracking_reference_evidence"] == proof
    effect = audit(a, p, ack, before, ctx, phase=phase)
    assert effect["verified"] and effect["setter_dispatch_targets_equal"]
    assert effect["actual_mapping_matches_dispatch"]
    assert ctx == frozen_ctx
    for tensor in buffers(a):
        assert tensor.device.type == "cpu" and tensor.dtype == torch.float32
    assert torch.equal(a.robot.data.joint_pos_target, a.robot._joint_pos_target_sim)
    assert torch.equal(a.robot.data.joint_vel_target, a.robot._joint_vel_target_sim)
    for i, name in enumerate(SERVO_ORDER):
        lo, hi = servo_limits_deg(name)
        assert lo <= ack["drive_target_full12"][i] <= hi
        assert abs(ack["drive_target_full12"][i] - before[i]) <= 1.25 + 1.e-12
    return ctx, before, ack, effect


@pytest.mark.parametrize("tracking", [(), (SERVO_ORDER[0],), SERVO_ORDER])
def test_zero_request_history_preserves_original_mapper_and_four_buffers(tracking):
    a, original = adapter(), adapter()
    nominal = (2., -3., 1., 2., -1., 3., 2., -2., .1, -.2, .3, -.4)
    for tick in range(1, 18):
        _, _, ack, effect = checked_step(a, plan(ZERO, nominal=nominal), tick, tracking=tracking)
        old = original.apply_full12(nominal, physics_tick=tick,
                                    tracking_servo_names=tracking, drive_feedback_bias_full12=ZERO)
        for key, value in old.items():
            assert ack[key] == value
        for actual, expected in zip(buffers(a), buffers(original), strict=True):
            assert torch.equal(actual, expected)
        assert effect["changed_target_channel_count"] == 0
    assert a.servo_target_mapper.feedback_tick == original.servo_target_mapper.feedback_tick == 18
    assert a.write_count == 18


@pytest.mark.parametrize("channel", [0, 7])
def test_zero_current_request_retains_previous_request_reference_and_one_real_advance(channel, monkeypatch):
    a = adapter()
    for tick in range(1, 4):
        checked_step(a, plan(full(5., channel)), tick, tracking=(SERVO_ORDER[channel],))
    # Follow the prior policy target in real physical coordinates, including rear sign.
    joint = a.joint_map.servo_ids[channel]
    a.robot.data.joint_pos[0, joint] = .75 + SERVO_COMMAND_SIGN[SERVO_ORDER[channel]] * math.radians(5.)
    measured = a.robot.data.joint_pos.clone()
    original_advance = a.servo_target_mapper.advance
    original_apply = a.apply_full12
    calls, fast = [], []
    def observed(*args, **kwargs):
        calls.append((tuple(args[0]), tuple(args[1]), kwargs))
        return original_advance(*args, **kwargs)
    def old_fast(*args, **kwargs):
        fast.append(True)
        return original_apply(*args, **kwargs)
    monkeypatch.setattr(a.servo_target_mapper, "advance", observed)
    monkeypatch.setattr(a, "apply_full12", old_fast)
    ticks, writes, event_count = a.servo_target_mapper.feedback_tick, a.write_count, len(a.robot.events)
    ctx, _, ack, effect = checked_step(a, plan(ZERO), 4, tracking=(SERVO_ORDER[channel],))
    assert ctx["previous_requested_full12"][channel] == 5.
    assert not fast, "r_current=0 must not bypass a nonzero previous requested reference"
    assert len(calls) == 1 and calls[0][0] == ZERO[:8]
    assert calls[0][1] == tuple(ack["tracking_reference_evidence"]["mapper_computational_reference_rad"])
    assert calls[0][1][channel] != ctx["actual_measured_physical_rad"][channel]
    assert torch.equal(a.robot.data.joint_pos, measured), "computational reference must not rewrite sensing"
    assert a.servo_target_mapper.feedback_tick == ticks + 1 and a.write_count == writes + 1
    assert a.robot.events[event_count:] == ["position.setter", "velocity.setter", "dispatch"]
    assert ack["independent_policy_residual_requested_full12"] == list(ZERO)
    assert effect["changed_target_channel_count"] == 0  # zero CURRENT-r counterfactual shares this history


def test_previous_requested_is_used_even_when_previous_headroom_effective_was_zero():
    a = adapter()
    nominal = full(-58.)
    for tick in range(1, 67):
        checked_step(a, plan(ZERO, nominal=nominal), tick)
    _, _, clipped, _ = checked_step(a, plan(full(-24.), nominal=nominal), 67)
    assert clipped["policy_headroom_evidence"]["effective_policy_residual_full12"][7] == 0.
    assert clipped["independent_policy_residual_requested_full12"][7] == -24.
    ctx, _, ack, _ = checked_step(a, plan(ZERO, nominal=nominal), 68, tracking=(SERVO_ORDER[7],))
    assert ctx["previous_requested_full12"][7] == -24.
    evidence = ack["tracking_reference_evidence"]
    assert evidence["mapper_computational_reference_rad"][7] != ctx["actual_measured_physical_rad"][7]
    assert ack["applied_full12"] == list(nominal)
    assert ack["policy_headroom_evidence"]["requested_policy_residual_full12"] == list(ZERO)


def test_ended_tracking_uses_real_q_for_original_stale_bias_retirement():
    a = adapter()
    channel, name = 7, SERVO_ORDER[7]
    a.robot.data.joint_pos[0, a.joint_map.servo_ids[channel]] = .75 - SERVO_COMMAND_SIGN[name] * math.radians(3.)
    for tick in range(1, 42):
        checked_step(a, plan(full(2.)), tick, tracking=(name,))
    assert abs(a.servo_target_mapper._compensation[name]) > 2.
    # Real q has converged to nominal, so the original ended-tracking criterion applies.
    a.robot.data.joint_pos[0, a.joint_map.servo_ids[channel]] = .75
    expected = copy.deepcopy(a.servo_target_mapper)
    real = tuple(float(a.robot.data.joint_pos[0, index]) for index in a.joint_map.servo_ids)
    expected_map = expected.advance(ZERO[:8], real, tracking_servo_names=())
    ctx, _, ack, _ = checked_step(a, plan(ZERO), 42, tracking=())
    assert ack["tracking_reference_evidence"]["mapper_computational_reference_rad"] == list(real)
    assert ack["servo_tracking_compensation_deg"] == list(expected_map.tracking_compensation_deg)
    assert a.servo_target_mapper._retiring_stale_bias == expected._retiring_stale_bias
    assert ctx["previous_requested_full12"][7] == 2.


def test_phase_change_keeps_request_history_nominal_and_feedback_clock():
    a = adapter()
    nominal = full(2., 0)
    previous = None
    for tick in range(1, 10):
        phase = "P08" if tick <= 4 else "P09"
        ctx, _, ack, _ = checked_step(a, plan(full(.3, 0), nominal=nominal), tick,
                                      tracking=(SERVO_ORDER[0],), phase=phase)
        assert ack["requested_full12"] == list(nominal)
        assert ack["applied_full12"] == list(nominal)
        assert a.servo_target_mapper.feedback_tick == tick + 1
        if previous is not None:
            assert ctx["previous_requested_full12"] == previous["independent_policy_residual_requested_full12"]
            assert ctx["previous_ack_physics_tick"] == tick - 1
        previous = ack


def prepared():
    a = adapter()
    for tick in range(1, 4):
        checked_step(a, plan(full(2., 0)), tick, tracking=(SERVO_ORDER[0],))
    ctx, before, ack, _ = checked_step(a, plan(full(.3, 0)), 4, tracking=(SERVO_ORDER[0],))
    return a, plan(full(.3, 0)), ctx, before, ack


@pytest.mark.parametrize("tamper", [
    "missing_mode", "unknown_mode", "missing_evidence", "reference", "history", "q",
    "previous_tick", "dispatch_tick", "feedback_clock", "bool_history", "nan_reference",
    "context_history", "context_q", "context_stale", "source_tracking", "missing_source_tracking",
])
def test_independent_audit_rejects_forged_current_ack_or_pre_dispatch_context(tamper):
    a, p, ctx, before, ack = prepared()
    ack, ctx = copy.deepcopy(ack), copy.deepcopy(ctx)
    evidence = ack["tracking_reference_evidence"]
    if tamper == "missing_mode": del ack["tracking_reference_mode"]
    elif tamper == "unknown_mode": ack["tracking_reference_mode"] = "unknown"
    elif tamper == "missing_evidence": del ack["tracking_reference_evidence"]
    elif tamper == "reference": evidence["mapper_computational_reference_rad"][0] += .01
    elif tamper == "history": evidence["previous_requested_full12"][0] += 1.
    elif tamper == "q": evidence["actual_measured_physical_rad"][0] += .01
    elif tamper == "previous_tick": evidence["previous_ack_physics_tick"] -= 1
    elif tamper == "dispatch_tick": ack["physics_tick"] += 1
    elif tamper == "feedback_clock": evidence["mapper_feedback_tick"] += 1
    elif tamper == "bool_history": evidence["previous_requested_full12"][0] = True
    elif tamper == "nan_reference": evidence["mapper_computational_reference_rad"][0] = float("nan")
    elif tamper == "context_history": ctx["previous_requested_full12"][0] += 1.
    elif tamper == "context_q": ctx["actual_measured_physical_rad"][0] += .01
    elif tamper == "context_stale": ctx["dispatch_physics_tick"] -= 1
    elif tamper == "source_tracking": ctx["source_tracking_servo_names"] = []
    elif tamper == "missing_source_tracking": del ctx["source_tracking_servo_names"]
    state, targets = copy.deepcopy(a.servo_target_mapper.__dict__), buffers(a)
    with pytest.raises(ActuatorTargetEffectError):
        audit(a, p, ack, before, ctx)
    assert a.servo_target_mapper.__dict__ == state
    for left, right in zip(buffers(a), targets, strict=True):
        assert torch.equal(left, right)


def test_audit_requires_independent_context_and_rejects_unexpected_mode():
    a, p, ctx, before, ack = prepared()
    for supplied_mode, supplied_context in ((MODE, None), (None, ctx), ("unknown", ctx)):
        with pytest.raises(ActuatorTargetEffectError):
            audit(a, p, ack, before, supplied_context, mode=supplied_mode)


@pytest.mark.parametrize("buffer_index", range(4))
def test_real_cpu_float32_setter_and_dispatch_buffer_tamper_is_detected(buffer_index):
    a, p, ctx, before, ack = prepared()
    owners = ((a.robot.data, "joint_pos_target", a.joint_map.servo_ids[0]),
              (a.robot.data, "joint_vel_target", a.joint_map.wheel_ids[0]),
              (a.robot, "_joint_pos_target_sim", a.joint_map.servo_ids[0]),
              (a.robot, "_joint_vel_target_sim", a.joint_map.wheel_ids[0]))
    owner, key, index = owners[buffer_index]
    getattr(owner, key)[0, index] += .125
    with pytest.raises(ActuatorTargetEffectError):
        audit(a, p, ack, before, ctx)


@pytest.mark.parametrize("bad_mode", ["unknown", True, 1])
def test_bad_mode_does_not_advance_or_write(bad_mode):
    a = adapter()
    before, writes, clock = buffers(a), a.write_count, a.servo_target_mapper.feedback_tick
    with pytest.raises(RobotAdapterError):
        dispatch(a, plan(ZERO), 1, mode=bad_mode)
    assert a.write_count == writes and a.servo_target_mapper.feedback_tick == clock
    for left, right in zip(buffers(a), before, strict=True):
        assert torch.equal(left, right)


def test_controller_ten_degree_envelope_not_relaxed_by_reference_mode():
    a = adapter()
    with pytest.raises(RobotAdapterError, match="bounded mapper envelope"):
        apply_semantic_residual(a, ZERO, physics_tick=1, tracking_servo_names=(),
            controller_bias_full12=full(10.01), projected_residual_full12=full(-50.),
            policy_headroom_mode=HEADROOM_MODE, tracking_reference_mode=MODE,
            tracking_reference_bootstrap_tick=BOOTSTRAP_TICK)
    assert a.write_count == a.servo_target_mapper.feedback_tick == 1


def test_native_audit_json_round_trip_keeps_strict_independent_evidence():
    a, p, ctx, before, ack = prepared()
    effect = audit(a, p, json.loads(json.dumps(ack)), before, json.loads(json.dumps(ctx)))
    assert effect["verified"] and effect["target_dtype"] == "torch.float32"


@pytest.mark.parametrize("tamper", ["absent", "tick", "feedback", "request_composition",
                                    "missing_mode", "missing_receipt"])
def test_invalid_previous_live_ack_fails_before_mapper_or_buffer_mutation(tamper):
    a = adapter()
    checked_step(a, plan(full(2., 0)), 1, tracking=(SERVO_ORDER[0],))
    if tamper == "absent": a.last_ack = None
    elif tamper == "tick": a.last_ack["physics_tick"] = 0
    elif tamper == "feedback": a.last_ack["servo_tracking_feedback_sample_tick"] += 1
    elif tamper == "request_composition": a.last_ack["independent_policy_residual_requested_full12"][0] += 1.
    elif tamper == "missing_mode": del a.last_ack["tracking_reference_mode"]
    elif tamper == "missing_receipt": del a.last_ack["tracking_reference_evidence"]
    state, targets = copy.deepcopy(a.servo_target_mapper.__dict__), buffers(a)
    writes, events = a.write_count, list(a.robot.events)
    with pytest.raises(RobotAdapterError):
        dispatch(a, plan(ZERO), 2, tracking=(SERVO_ORDER[0],))
    assert a.servo_target_mapper.__dict__ == state
    assert a.write_count == writes and a.robot.events == events
    for left, right in zip(buffers(a), targets, strict=True):
        assert torch.equal(left, right)
