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


NOMINAL_HISTORY_SEMANTICS = "same_actual_state_nominal_command_history_excludes_policy_v1"


def _nominal_previous(adapter):
    """Read a command trace, initialized only before the first policy dispatch.

    This is not a second mapper, observation, or simulated trajectory. The
    trace uses the one live mapper result and current measured geometry. It
    excludes the additive policy offset so a geometry hold cannot integrate it.
    Its eight values are controller-internal command memory, logged in ACKs,
    not new/reinterpreted features in the existing 372-dimensional observation.
    Existing observations are not claimed to expose the full controller state.
    """
    trace = getattr(adapter, "_semantic_nominal_command_history", None)
    writes = adapter.write_count
    if trace is None:
        ack = getattr(adapter, "last_ack", None)
        actual = tuple(float(adapter._final_drive_servo_deg[name]) for name in SERVO_ORDER)
        fresh = writes == 0 and ack is None and not any(actual)
        settled = (isinstance(ack, Mapping) and
            "semantic_residual_composition" not in ack and
            ack.get("write_count") == writes and
            ack.get("physics_tick") == adapter._last_physics_tick and
            not any(actual) and
            all(tuple(ack.get(key, ())) == (0.,)*12 for key in
                ("requested_full12", "applied_full12", "drive_target_full12",
                 "native_drive_target_full12", "drive_feedback_bias_requested_full12")))
        if not (fresh or settled):
            raise RobotAdapterError("nominal command history missing outside fresh/settled zero reset")
        return actual, "fresh_adapter_zero" if fresh else "settled_original_zero_ack"
    if (not isinstance(trace, Mapping) or trace.get("write_count") != writes or
            trace.get("physics_tick") != adapter._last_physics_tick):
        raise RobotAdapterError("nominal command history is not adjacent to the actual dispatch")
    try:
        previous = tuple(float(value) for value in trace["servo_deg"])
    except (TypeError, ValueError, KeyError) as exc:
        raise RobotAdapterError("invalid nominal command history") from exc
    if len(previous) != 8 or any(not math.isfinite(value) for value in previous):
        raise RobotAdapterError("invalid nominal command history")
    return previous, "previous_semantic_nominal_command"


def _nominal_step(adapter, previous, native, controller):
    result = []
    for name, before, target, bias in zip(SERVO_ORDER, previous, native[:8], controller[:8], strict=True):
        lower, upper = servo_limits_deg(name)
        result.append(bounded_drive_feedback_step(previous_deg=before, native_deg=target,
            bias_deg=bias, maximum_delta_deg=adapter.servo_target_mapper.maximum_delta_deg,
            lower_deg=lower, upper_deg=upper))
    return result


def _store_nominal_history(adapter, ack, previous, nominal, source):
    adapter._semantic_nominal_command_history = {
        "servo_deg": tuple(nominal), "write_count": adapter.write_count,
        "physics_tick": adapter._last_physics_tick,
    }
    ack.update(nominal_history_semantics=NOMINAL_HISTORY_SEMANTICS,
        nominal_history_initialization=source,
        nominal_command_history_added_to_policy_observation=False,
        previous_nominal_command_servo_deg=list(previous),
        nominal_command_servo_deg=list(nominal))


def apply_semantic_residual(adapter: Any, command: Sequence[float], *,
                            physics_tick: int, tracking_servo_names: Sequence[str],
                            controller_bias_full12: Sequence[float],
                            projected_residual_full12: Sequence[float],
                            nominal_geometry_context: Mapping[str, Any] | None = None,
                            policy_headroom_mode: str | None = None,
                            tracking_reference_mode: str | None = None,
                            tracking_reference_bootstrap_tick: int | None = None,
                            capture_assist: Any = None,
                            capture_assist_context: Mapping[str, Any] | None = None,
                            rr_capture_assist: Any = None,
                            rr_capture_assist_context: Mapping[str, Any] | None = None,
                            rr_carry_wheel_context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Dispatch independent policy residual without treating it as tracking bias."""
    # Keep the original controller envelope, including for the exact-zero path.
    controller = _full12_drive_feedback_bias(controller_bias_full12)
    residual = tuple(float(value) for value in projected_residual_full12)
    if len(residual) != 12 or any(not math.isfinite(value) for value in residual):
        raise RobotAdapterError("PPO residual must contain twelve finite values")
    combined = tuple(a + b for a, b in zip(controller, residual, strict=True))
    if any(not math.isfinite(value) for value in combined):
        raise RobotAdapterError("combined PPO target offset is non-finite")
    nominal_previous, nominal_history_source = _nominal_previous(adapter)
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
    tracking_reference = None
    if tracking_reference_mode is not None:
        from .semantic_tracking_reference import (
            MODE, build_tracking_reference, capture_tracking_reference_context,
        )
        if tracking_reference_mode != MODE or policy_headroom_mode is None:
            raise RobotAdapterError("semantic tracking reference requires its known mode and same-tick headroom")
        try:
            reference_context = capture_tracking_reference_context(
                adapter, physics_tick=physics_tick,
                bootstrap_physics_tick=tracking_reference_bootstrap_tick)
            tracking_reference = build_tracking_reference(
                reference_context,
                requested_command_deg=adapter._coerce_command(command).clamped().servo_deg,
                tracking_servo_names=tracking_servo_names)
        except (ValueError, TypeError, KeyError) as exc:
            raise RobotAdapterError(f"invalid semantic tracking reference: {exc}") from exc
        evidence.update(tracking_reference_mode=tracking_reference_mode,
                        tracking_reference_evidence=tracking_reference)
    # A zero CURRENT action need not have zero reference history. Ordinary
    # phase changes and a policy withdrawal never reset that shared history.
    reference_history_nonzero = (tracking_reference is not None and
        any(tracking_reference["previous_requested_full12"][:8]))
    if (not any(residual) and nominal_geometry_context is None and not reference_history_nonzero
            and capture_assist is None and rr_capture_assist is None and rr_carry_wheel_context is None):
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
        nominal_servo = _nominal_step(adapter, nominal_previous,
            ack["native_drive_target_full12"], controller)
        _store_nominal_history(adapter, ack, nominal_previous, nominal_servo, nominal_history_source)
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
    measured = (_row_values(_joint_matrix(adapter.robot, "joint_pos")[:, list(adapter.joint_map.servo_ids)])
                if tracking_reference is None else
                tuple(tracking_reference["mapper_computational_reference_rad"]))
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
            controller_bias_full12=controller, context=nominal_geometry_context,
            nominal_previous_servo_deg=nominal_previous)
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
    nominal_servo = _nominal_step(adapter, nominal_previous, corrected_native, controller)
    assist_targets = None
    if capture_assist is not None:
        from .semantic_capture_assist import apply_capture_assist_snapshot
        if capture_assist_context is None or capture_assist_context.get("dispatch_physics_tick") != tick:
            raise RobotAdapterError("capture assist requires matching current physical context")
        previous_final = tuple(adapter._final_drive_servo_deg[name] for name in SERVO_ORDER)
        previous_wheels = tuple((getattr(adapter, "last_ack", None) or {}).get("drive_target_full12", (0.,)*12)[8:])
        receipt = capture_assist.advance(context=capture_assist_context,
            previous_final_full12=previous_final+previous_wheels, physics_dt_s=adapter.physics_dt_s)
        candidate = tuple(a+b for a,b in zip(corrected_native, effective_combined, strict=True))
        assist_targets = apply_capture_assist_snapshot(candidate, receipt["state_after"])
        receipt.update(candidate_before_assist_full12=list(candidate),
            candidate_after_assist_full12=list(assist_targets),
            assist_correction_full12=[a-b for a,b in zip(assist_targets,candidate,strict=True)],
            policy_request_unchanged=True, owner_indices=[0,1] if receipt["state_after"]["active"] else [],
            transform_semantics="observable_state_dependent_FL_final_target_before_existing_hard_clamp_and_slew",
            nominal_history_receives_assist=False)
        evidence["capture_assist_evidence"] = receipt
    rr_assist_targets = None
    if rr_capture_assist is not None:
        from .semantic_rr_capture_assist import apply_rr_capture_assist_snapshot
        if rr_capture_assist_context is None or rr_capture_assist_context.get("dispatch_physics_tick") != tick:
            raise RobotAdapterError("RR capture assist requires matching current physical context")
        previous_final = tuple(adapter._final_drive_servo_deg[name] for name in SERVO_ORDER)
        previous_wheels = tuple((getattr(adapter, "last_ack", None) or {}).get("drive_target_full12", (0.,)*12)[8:])
        candidate = (assist_targets if assist_targets is not None else
                     tuple(a+b for a,b in zip(corrected_native,effective_combined,strict=True)))
        if tracking_reference is None:
            raise RobotAdapterError("RR incremental capture requires verified previous issued N/request history")
        # Both values are the actual adjacent dispatch inputs, already exposed
        # by semantic_env HISTORY. Neither mapper compensation nor the clipped
        # absolute candidate is a baseline for captured-target continuation.
        rr_context = dict(rr_capture_assist_context)
        rr_context.update(
            issued_nominal_delta_rr_deg=[logical_applied.servo_deg[i]
                - tracking_reference["mapper_pre_state"]["requested_servo_deg"][i] for i in (6,7)],
            issued_requested_residual_delta_rr_deg=[residual[i]
                - tracking_reference["previous_requested_full12"][i] for i in (6,7)],
            release_candidate_rr_deg=[candidate[i] for i in (6,7)])
        receipt = rr_capture_assist.advance(context=rr_context,
            previous_final_full12=previous_final+previous_wheels, physics_dt_s=adapter.physics_dt_s)
        rr_assist_targets = apply_rr_capture_assist_snapshot(candidate,receipt["state_after"],
            previous_final_full12=previous_final+previous_wheels)
        receipt.update(candidate_before_assist_full12=list(candidate),
            candidate_after_assist_full12=list(rr_assist_targets),
            assist_correction_full12=[a-b for a,b in zip(rr_assist_targets,candidate,strict=True)],
            policy_request_unchanged=True, owner_indices=[6,7] if receipt["state_after"]["active"] else [],
            transform_semantics="observable_state_dependent_RR_final_target_before_existing_hard_clamp_and_slew",
            nominal_history_receives_assist=False)
        evidence["rr_capture_assist_evidence"] = receipt
    final_servo = []
    for index, (name, target, bias) in enumerate(zip(SERVO_ORDER, corrected_native[:8], effective_combined[:8], strict=True)):
        if assist_targets is not None and index in evidence["capture_assist_evidence"]["owner_indices"]:
            target, bias = assist_targets[index], 0.
        if rr_assist_targets is not None and index in evidence["rr_capture_assist_evidence"]["owner_indices"]:
            target, bias = rr_assist_targets[index], 0.
        lower, upper = servo_limits_deg(name)
        final_servo.append(bounded_drive_feedback_step(
            previous_deg=adapter._final_drive_servo_deg[name], native_deg=target,
            bias_deg=bias, maximum_delta_deg=adapter.servo_target_mapper.maximum_delta_deg,
            lower_deg=lower, upper_deg=upper))
    final_wheels = tuple(max(-WHEEL_VELOCITY_LIMIT_RAD_S,
        min(WHEEL_VELOCITY_LIMIT_RAD_S, target + bias))
        for target, bias in zip(corrected_native[8:], effective_combined[8:], strict=True))
    if rr_carry_wheel_context is not None:
        from .semantic_rr_carry_wheel import project_rr_carry_wheels
        if rr_carry_wheel_context.get("dispatch_physics_tick") != tick:
            raise RobotAdapterError("RR wheel projection requires current dispatch context")
        previous_ack = getattr(adapter, "last_ack", None)
        if (not isinstance(previous_ack, Mapping) or previous_ack.get("write_count") != adapter.write_count
                or previous_ack.get("physics_tick") != adapter._last_physics_tick):
            raise RobotAdapterError("RR wheel projection requires adjacent committed final-wheel history")
        previous_wheels = tuple(previous_ack["drive_target_full12"][8:])
        wheel_candidate = tuple(final_servo) + final_wheels
        receipt = project_rr_carry_wheels(wheel_candidate, context=rr_carry_wheel_context,
            previous_final_wheel_rad_s=previous_wheels, physics_dt_s=adapter.physics_dt_s)
        final_wheels = tuple(receipt["output_full12"][8:])
        evidence["rr_carry_wheel_evidence"] = receipt
    if "capture_assist_evidence" in evidence:
        evidence["capture_assist_evidence"].update(final_servo_target_deg=list(final_servo),
            final_slew_or_clamp_indices=[i for i in (0,1) if final_servo[i] != assist_targets[i]],
            knee_hold_final_verified=(not evidence["capture_assist_evidence"]["state_after"]["active"]
                or evidence["capture_assist_evidence"]["state_after"]["mode"] == 4
                or final_servo[1] == evidence["capture_assist_evidence"]["state_after"]["knee_hold_deg"]))
    if "rr_capture_assist_evidence" in evidence:
        evidence["rr_capture_assist_evidence"].update(final_servo_target_deg=list(final_servo),
            final_slew_or_clamp_indices=[i for i in (6,7) if final_servo[i] != rr_assist_targets[i]],
            knee_hold_final_verified=(not evidence["rr_capture_assist_evidence"]["state_after"]["active"]
                or evidence["rr_capture_assist_evidence"]["state_after"]["mode"] == 4
                or final_servo[7] == evidence["rr_capture_assist_evidence"]["state_after"]["knee_hold_deg"]))
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
    # Only actual targets enter the actuator slew/history and counterfactual
    # audit. The separate nominal command trace is never sent to the robot and
    # never substitutes for actual state or advances a second mapper.
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
    _store_nominal_history(adapter, ack, nominal_previous, nominal_servo, nominal_history_source)
    adapter.last_ack = dict(ack)
    return ack


class SemanticActuationDispatch:
    """Per-call view retaining the existing backend's atomic ACK validation."""

    def __init__(self, adapter: Any, plan: Any, *, nominal_geometry_context: Mapping[str, Any] | None = None,
                 policy_headroom_mode: str | None = None,
                 tracking_reference_mode: str | None = None,
                 tracking_reference_bootstrap_tick: int | None = None,
                 capture_assist: Any = None, capture_assist_context: Mapping[str, Any] | None = None,
                 rr_capture_assist: Any = None, rr_capture_assist_context: Mapping[str, Any] | None = None,
                 rr_carry_wheel_context: Mapping[str, Any] | None = None):
        self.adapter, self.plan = adapter, plan
        self.nominal_geometry_context = nominal_geometry_context
        self.policy_headroom_mode = policy_headroom_mode
        self.tracking_reference_mode = tracking_reference_mode
        self.tracking_reference_bootstrap_tick = tracking_reference_bootstrap_tick
        self.capture_assist = capture_assist
        self.capture_assist_context = capture_assist_context
        self.rr_capture_assist = rr_capture_assist
        self.rr_capture_assist_context = rr_capture_assist_context
        self.rr_carry_wheel_context = rr_carry_wheel_context

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
            policy_headroom_mode=self.policy_headroom_mode,
            tracking_reference_mode=self.tracking_reference_mode,
            tracking_reference_bootstrap_tick=self.tracking_reference_bootstrap_tick,
            capture_assist=self.capture_assist, capture_assist_context=self.capture_assist_context,
            rr_capture_assist=self.rr_capture_assist, rr_capture_assist_context=self.rr_capture_assist_context,
            rr_carry_wheel_context=self.rr_carry_wheel_context)
