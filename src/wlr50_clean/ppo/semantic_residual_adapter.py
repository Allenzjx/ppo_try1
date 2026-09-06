"""PPO-only post-mapper composition; the frozen A adapter is not modified.

Controller correction and policy residual have different bounds. Only the
former is a bounded tracking correction. Both share the original final-drive
hard limits and slew, after exactly one nominal mapper advance. This module
never steps physics, resets state, or adds a second articulation write.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from wlr50_clean.infrastructure.command_batch import (
    FULL12_ORDER, SERVO_ORDER, WHEEL_VELOCITY_LIMIT_RAD_S, Full12Command,
    build_physical_batch, servo_limits_deg,
)
from wlr50_clean.infrastructure.robot_adapter import (
    RobotAdapterError, _clone_tensor, _full12_drive_feedback_bias,
    _joint_matrix, _row_values, bounded_drive_feedback_step,
)
from wlr50_clean.infrastructure.servo_target_mapper import ServoTargetMapperError


def apply_semantic_residual(adapter: Any, command: Sequence[float], *,
                            physics_tick: int, tracking_servo_names: Sequence[str],
                            controller_bias_full12: Sequence[float],
                            projected_residual_full12: Sequence[float],
                            nominal_geometry_context: Mapping[str, Any] | None = None,
                            policy_headroom_mode: str | None = None) -> dict[str, Any]:
    """Dispatch independent policy residual without treating it as tracking bias."""
    # Keep the original controller envelope, including for the exact-zero path.
    controller = _full12_drive_feedback_bias(controller_bias_full12)
    residual = tuple(float(value) for value in projected_residual_full12)
    if len(residual) != 12 or any(not math.isfinite(value) for value in residual):
        raise RobotAdapterError("PPO residual must contain twelve finite values")
    combined = tuple(a + b for a, b in zip(controller, residual, strict=True))
    if any(not math.isfinite(value) for value in combined):
        raise RobotAdapterError("combined PPO target offset is non-finite")
    if policy_headroom_mode is not None:
        from .semantic_headroom import HEADROOM_MODE, project_semantic_servo_headroom
        if policy_headroom_mode != HEADROOM_MODE:
            raise RobotAdapterError("unknown semantic policy headroom mode")
    evidence = {
        "semantic_residual_composition": "independent_post_mapper_residual.v1",
        "bounded_controller_bias_requested_full12": list(controller),
        "independent_policy_residual_requested_full12": list(residual),
        "controller_bias_original_envelope_verified": True,
        # Existing backend and native-audit readers consume this compatibility
        # field as the TOTAL post-mapper offset, not the controller-only bias.
        "drive_feedback_bias_requested_semantics": "combined_controller_plus_independent_policy_residual",
    }
    if not any(residual) and nominal_geometry_context is None:
        ack = adapter.apply_full12(command, physics_tick=physics_tick,
            tracking_servo_names=tracking_servo_names, drive_feedback_bias_full12=controller)
        if policy_headroom_mode is not None:
            # Keep exact zero/teacher execution on the original adapter. Its
            # just-completed native ACK supplies this read-only receipt; no
            # duplicate mapper, history update or target dispatch is needed.
            headroom = project_semantic_servo_headroom(
                native_full12=tuple(ack["native_drive_target_full12"]),
                controller_bias_full12=controller, projected_residual_full12=residual)
            evidence.update(policy_headroom_mode=policy_headroom_mode,
                            policy_headroom_evidence=headroom)
        ack.update(evidence)
        adapter.last_ack = dict(ack)
        return ack

    requested = adapter._coerce_command(command)
    tick = adapter._validate_tick(physics_tick)
    if nominal_geometry_context is not None:
        if not isinstance(nominal_geometry_context, Mapping):
            raise RobotAdapterError("nominal geometry context must be a mapping")
        context_tick = nominal_geometry_context.get("dispatch_physics_tick")
        if type(context_tick) is not int or context_tick != tick:
            raise RobotAdapterError("nominal geometry context does not match the dispatch physics tick")
        context_indices = nominal_geometry_context.get("canonical_servo_indices")
        if (not isinstance(context_indices, (tuple, list))
                or any(type(index) is not int for index in context_indices)
                or tuple(context_indices) not in ((4, 5), (6, 7))):
            raise RobotAdapterError("nominal geometry context must identify one ordered rear hip/knee pair")
        geometry_indices = frozenset(context_indices)
    logical_applied = requested.clamped()
    measured = _row_values(_joint_matrix(adapter.robot, "joint_pos")[:, list(adapter.joint_map.servo_ids)])
    try:
        mapping = adapter.servo_target_mapper.advance(logical_applied.servo_deg, measured,
            tracking_servo_names=tracking_servo_names)
    except ServoTargetMapperError as exc:
        raise RobotAdapterError(f"invalid servo target mapping: {exc}") from exc
    native = mapping.applied_drive_command_deg
    native_drive = Full12Command(native, logical_applied.wheel_rad_s)
    corrected_native = native_drive.to_full12()
    if nominal_geometry_context is not None:
        from .semantic_nominal_geometry import correct_nominal_geometry

        # The helper sees the unique real mapper result and the bounded
        # controller correction, never this tick's raw/projected PPO action.
        corrected, geometry_evidence = correct_nominal_geometry(
            adapter=adapter, native_full12=native_drive.to_full12(),
            controller_bias_full12=controller, context=nominal_geometry_context)
        corrected_native = tuple(float(value) for value in corrected)
        if len(corrected_native) != 12 or any(not math.isfinite(value) for value in corrected_native):
            raise RobotAdapterError("nominal geometry must return twelve finite native targets")
        changed = {index for index, (before, after) in enumerate(
            zip(native_drive.to_full12(), corrected_native, strict=True)) if before != after}
        if not changed <= geometry_indices:
            raise RobotAdapterError("nominal geometry changed channels outside its declared rear hip/knee pair")
        if not isinstance(geometry_evidence, Mapping):
            raise RobotAdapterError("nominal geometry evidence must be a mapping")
        evidence.update({
            "geometry_adjusted_native_full12": list(corrected_native),
            "nominal_geometry_adjustment_full12": [after-before for before, after in
                zip(native_drive.to_full12(), corrected_native, strict=True)],
            "nominal_geometry_evidence": dict(geometry_evidence),
        })
    effective_combined = combined
    if policy_headroom_mode is not None:
        # Headroom belongs to this unique real mapper/geometry result. The
        # request remains unchanged for the backend, bridge and prefix receipt.
        headroom = project_semantic_servo_headroom(
            native_full12=corrected_native, controller_bias_full12=controller,
            projected_residual_full12=residual)
        effective_combined = tuple(headroom["effective_combined_post_mapper_bias_full12"])
        evidence.update(policy_headroom_mode=policy_headroom_mode,
                        policy_headroom_evidence=headroom)
    final_servo = []
    for name, target, bias in zip(SERVO_ORDER, corrected_native[:8], effective_combined[:8], strict=True):
        lower, upper = servo_limits_deg(name)
        final_servo.append(bounded_drive_feedback_step(
            previous_deg=adapter._final_drive_servo_deg[name], native_deg=target,
            bias_deg=bias, maximum_delta_deg=adapter.servo_target_mapper.maximum_delta_deg,
            lower_deg=lower, upper_deg=upper))
    final_wheels = tuple(max(-WHEEL_VELOCITY_LIMIT_RAD_S,
        min(WHEEL_VELOCITY_LIMIT_RAD_S, target + bias))
        for target, bias in zip(corrected_native[8:], effective_combined[8:], strict=True))
    drive = Full12Command(tuple(final_servo), final_wheels)
    physical = build_physical_batch(drive, adapter.standing_pose_deg)
    positions = _clone_tensor(adapter._standing_servo_tensor)
    for index, value in enumerate(physical.servo_target_rad):
        positions[:, index] = float(value)
    velocities = _clone_tensor(_joint_matrix(adapter.robot, "joint_vel")[:, list(adapter.joint_map.wheel_ids)])
    for index, value in enumerate(physical.wheel_target_rad_s):
        velocities[:, index] = float(value)
    adapter.robot.set_joint_position_target(positions, joint_ids=list(adapter.joint_map.servo_ids))
    adapter.robot.set_joint_velocity_target(velocities, joint_ids=list(adapter.joint_map.wheel_ids))
    adapter.robot.write_data_to_sim()
    # This is the one actual final-drive history, also used by the read-only
    # same-tick counterfactual. No second zero-trajectory history is introduced.
    adapter._final_drive_servo_deg.update(zip(SERVO_ORDER, final_servo, strict=True))
    adapter.write_count += 1
    adapter._last_physics_tick = tick
    ack = {
        "schema": "wlr50_clean.atomic_full12_ack.v1", "physics_tick": tick,
        "physics_dt_s": adapter.physics_dt_s, "write_count": adapter.write_count,
        "articulation_writes_this_call": 1, "canonical_order": list(FULL12_ORDER),
        "requested_full12": list(requested.to_full12()),
        "applied_full12": list(logical_applied.to_full12()),
        "drive_target_full12": list(drive.to_full12()),
        "native_drive_target_full12": list(native_drive.to_full12()),
        "drive_feedback_bias_requested_full12": list(combined),
        "drive_feedback_bias_realized_full12": [a-b for a,b in zip(drive.to_full12(),native_drive.to_full12(),strict=True)],
        "drive_feedback_final_slew_limit_deg_per_tick": adapter.servo_target_mapper.maximum_delta_deg,
        "command_was_clamped": requested != logical_applied,
        "servo_applied_drive_command_deg": final_servo,
        "servo_native_drive_command_deg": list(native),
        "servo_tracking_compensation_deg": list(mapping.tracking_compensation_deg),
        "servo_nominal_target_reached": list(mapping.nominal_target_reached),
        "servo_tracking_active": list(mapping.tracking_active),
        "tracking_servo_names": list(tracking_servo_names),
        "servo_tracking_feedback_sample_tick": mapping.feedback_sample_tick,
        "servo_tracking_feedback_sampled": mapping.feedback_sampled,
        "servo_joint_ids": list(adapter.joint_map.servo_ids),
        "wheel_joint_ids": list(adapter.joint_map.wheel_ids),
        "servo_target_physical_rad": list(physical.servo_target_rad),
        "wheel_target_physical_rad_s": list(physical.wheel_target_rad_s),
        "motion_start_skew_s": 0.0, **evidence,
    }
    adapter.last_ack = dict(ack)
    return ack


class SemanticActuationDispatch:
    """Per-call view retaining the existing backend's atomic ACK validation."""

    def __init__(self, adapter: Any, plan: Any, *, nominal_geometry_context: Mapping[str, Any] | None = None,
                 policy_headroom_mode: str | None = None):
        self.adapter, self.plan = adapter, plan
        self.nominal_geometry_context = nominal_geometry_context
        self.policy_headroom_mode = policy_headroom_mode

    def __getattr__(self, name):
        return getattr(self.adapter, name)

    def apply_full12(self, command, *, physics_tick, tracking_servo_names, drive_feedback_bias_full12):
        if (tuple(command) != self.plan.frozen_nominal_full12 or
                tuple(drive_feedback_bias_full12) != self.plan.combined_post_mapper_bias_full12):
            raise RobotAdapterError("semantic dispatch differs from the current actuation plan")
        return apply_semantic_residual(self.adapter, command, physics_tick=physics_tick,
            tracking_servo_names=tracking_servo_names,
            controller_bias_full12=self.plan.controller_drive_bias_full12,
            projected_residual_full12=self.plan.projected_residual_full12,
            nominal_geometry_context=self.nominal_geometry_context,
            policy_headroom_mode=self.policy_headroom_mode)
