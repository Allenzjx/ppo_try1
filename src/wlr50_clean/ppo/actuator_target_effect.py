"""Read-only, same-tick actuator effect evidence for the bounded smoke gate.

The mature mapper is advanced only by the real RobotAdapter dispatch.  Its
reported native result is shared by the actual and zero-current-residual
branches below.  Frozen scalar slew/clamp and physical conversion functions
are reused; neither branch writes targets or changes mapper state.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from wlr50_clean.infrastructure.command_batch import (
    FULL12_ORDER,
    SERVO_ORDER,
    Full12Command,
    build_physical_batch,
    servo_limits_deg,
)
from wlr50_clean.infrastructure.robot_adapter import bounded_drive_feedback_step


ACTUATOR_TARGET_EFFECT_SCHEMA = "wlr50_clean.actuator_target_effect_audit.v1"


class ActuatorTargetEffectError(RuntimeError):
    """The actual float32 dispatch could not be independently bound to its audit."""


def actuator_target_audit_request(
    phase_id: str,
    raw_policy_action_full12: Sequence[float],
    phase_mask_full12: Sequence[int],
) -> dict[str, Any]:
    if phase_id not in {f"P{index:02d}" for index in range(1, 14)}:
        raise ActuatorTargetEffectError("policy request phase must be P01-P13")
    action = _finite_values(raw_policy_action_full12, 12, "raw policy action")
    mask = tuple(phase_mask_full12)
    if len(mask) != 12 or any(value not in (0, 1) for value in mask):
        raise ActuatorTargetEffectError("policy request mask must contain twelve binary values")
    return {
        "policy_request_phase": phase_id,
        "raw_policy_action_full12": list(action),
        "phase_mask_full12": [int(value) for value in mask],
    }


def build_actuator_target_effect_audit(
    *,
    adapter: Any,
    actuation: Any,
    raw_ack: Mapping[str, Any],
    previous_final_drive_servo_deg: Sequence[float],
    source_phase_id: str,
    policy_request: Mapping[str, Any] | None,
    policy_headroom_mode: str | None = None,
    tracking_reference_mode: str | None = None,
    tracking_reference_context: Mapping[str, Any] | None = None,
    previous_final_drive_wheel_rad_s: Sequence[float] | None = None,
    rr_carry_wheel_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Inspect the already completed dispatch, never simulate another write.

    The counterfactual removes only this tick's PPO residual from the same
    measured state and t-1 final drive.  It is not an independent FSM rollout.
    This function is intentionally called only by explicitly enabled audits.
    """

    import torch

    if source_phase_id not in {f"P{index:02d}" for index in range(1, 14)}:
        raise ActuatorTargetEffectError("source phase must be P01-P13")
    if raw_ack.get("articulation_writes_this_call") != 1:
        raise ActuatorTargetEffectError("audit requires one completed articulation dispatch")
    previous = _finite_values(previous_final_drive_servo_deg, 8, "previous final drive")
    native = _finite_values(raw_ack["native_drive_target_full12"], 12, "native drive")
    reference_keys = ("tracking_reference_mode", "tracking_reference_evidence")
    reference_evidence = None
    if tracking_reference_mode is None:
        if tracking_reference_context is not None or any(key in raw_ack for key in reference_keys):
            raise ActuatorTargetEffectError("unexpected tracking reference dispatch evidence")
    else:
        import json
        from .semantic_tracking_reference import MODE, build_tracking_reference
        if (tracking_reference_mode != MODE or not isinstance(tracking_reference_context, Mapping)
                or raw_ack.get(reference_keys[0]) != MODE
                or not isinstance(raw_ack.get(reference_keys[1]), Mapping)):
            raise ActuatorTargetEffectError("missing or inconsistent tracking reference evidence")
        context = dict(tracking_reference_context)
        source_tracking = context.pop("source_tracking_servo_names", None)
        if (not isinstance(source_tracking, (list, tuple)) or
                list(source_tracking) != raw_ack.get("tracking_servo_names")):
            raise ActuatorTargetEffectError("tracking reference source channels differ from dispatch")
        try:
            reference_evidence = build_tracking_reference(
                context, requested_command_deg=actuation.frozen_nominal_full12[:8],
                tracking_servo_names=source_tracking)
            declared = json.dumps(dict(raw_ack[reference_keys[1]]), sort_keys=True, allow_nan=False)
            expected = json.dumps(reference_evidence, sort_keys=True, allow_nan=False)
        except (ValueError, TypeError, KeyError) as exc:
            raise ActuatorTargetEffectError(f"invalid tracking reference evidence: {exc}") from exc
        if declared != expected:
            raise ActuatorTargetEffectError("tracking reference differs from independent pre-dispatch reconstruction")
        if (type(raw_ack.get("physics_tick")) is not int or
                type(raw_ack.get("servo_tracking_feedback_sample_tick")) is not int or
                raw_ack.get("physics_tick") != context["dispatch_physics_tick"] or
                raw_ack.get("servo_tracking_feedback_sample_tick") != context["mapper_feedback_tick"] or
                adapter.servo_target_mapper.feedback_tick != context["mapper_feedback_tick"] + 1):
            raise ActuatorTargetEffectError("tracking reference mapper clock is not exactly one advance")
    geometry_keys = ("geometry_adjusted_native_full12", "nominal_geometry_adjustment_full12",
                     "nominal_geometry_evidence")
    geometry_enabled = any(key in raw_ack for key in geometry_keys)
    corrected_native = native
    if geometry_enabled:
        if not all(key in raw_ack for key in geometry_keys):
            raise ActuatorTargetEffectError("incomplete nominal geometry dispatch evidence")
        corrected_native = _finite_values(raw_ack[geometry_keys[0]], 12, "geometry adjusted native")
        adjustment = _finite_values(raw_ack[geometry_keys[1]], 12, "nominal geometry adjustment")
        if adjustment != tuple(after-before for before, after in zip(native, corrected_native, strict=True)):
            raise ActuatorTargetEffectError("nominal geometry adjustment differs from native targets")
        changed_geometry = {i for i, (before, after) in enumerate(zip(native, corrected_native, strict=True))
                            if before != after}
        if not (changed_geometry <= {4, 5} or changed_geometry <= {6, 7}):
            raise ActuatorTargetEffectError("nominal geometry changed channels outside one rear pair")
        if not isinstance(raw_ack[geometry_keys[2]], Mapping):
            raise ActuatorTargetEffectError("invalid nominal geometry evidence")
    controller_bias = _finite_values(actuation.controller_drive_bias_full12, 12, "controller bias")
    combined_bias = _finite_values(actuation.combined_post_mapper_bias_full12, 12, "combined bias")
    if _finite_values(raw_ack["drive_feedback_bias_requested_full12"], 12, "ack bias") != combined_bias:
        raise ActuatorTargetEffectError("actual dispatch bias differs from the actuation plan")
    headroom_keys = ("policy_headroom_mode", "policy_headroom_evidence")
    actual_bias, zero_policy_bias = combined_bias, controller_bias
    headroom = None
    if policy_headroom_mode is None:
        if any(key in raw_ack for key in headroom_keys):
            raise ActuatorTargetEffectError("unexpected policy headroom dispatch evidence")
    else:
        import json
        from .semantic_headroom import HEADROOM_MODE, project_semantic_servo_headroom
        if policy_headroom_mode != HEADROOM_MODE:
            raise ActuatorTargetEffectError("unknown expected policy headroom mode")
        if (not all(key in raw_ack for key in headroom_keys)
                or raw_ack[headroom_keys[0]] != policy_headroom_mode
                or not isinstance(raw_ack[headroom_keys[1]], Mapping)):
            raise ActuatorTargetEffectError("missing or inconsistent policy headroom dispatch evidence")
        requested_residual = _finite_values(actuation.projected_residual_full12, 12, "projected residual")
        if combined_bias != tuple(c+r for c,r in zip(controller_bias, requested_residual, strict=True)):
            raise ActuatorTargetEffectError("headroom request composition differs from the actuation plan")
        for key, expected in (("bounded_controller_bias_requested_full12", controller_bias),
                              ("independent_policy_residual_requested_full12", requested_residual)):
            if key not in raw_ack or _finite_values(raw_ack[key], 12, key) != expected:
                raise ActuatorTargetEffectError("headroom request receipt differs from the actuation plan")
        # Recompute, do not trust the dispatch's effective residual or margins.
        # JSON comparison preserves numeric/bool distinctions and accepts the
        # same list representation after a receipt has been serialized.
        headroom = project_semantic_servo_headroom(
            native_full12=corrected_native, controller_bias_full12=controller_bias,
            projected_residual_full12=requested_residual)
        try:
            declared = json.dumps(dict(raw_ack[headroom_keys[1]]), sort_keys=True, allow_nan=False)
            expected = json.dumps(headroom, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ActuatorTargetEffectError("invalid policy headroom evidence representation") from exc
        if declared != expected:
            raise ActuatorTargetEffectError("policy headroom evidence differs from independent reconstruction")
        actual_bias = tuple(headroom["effective_combined_post_mapper_bias_full12"])
        zero_headroom = project_semantic_servo_headroom(
            native_full12=corrected_native, controller_bias_full12=controller_bias,
            projected_residual_full12=(0.,)*12)
        zero_policy_bias = tuple(zero_headroom["effective_combined_post_mapper_bias_full12"])
    maximum_delta = float(raw_ack["drive_feedback_final_slew_limit_deg_per_tick"])
    if maximum_delta != float(adapter.servo_target_mapper.maximum_delta_deg):
        raise ActuatorTargetEffectError("actual dispatch slew limit differs from the frozen mapper")

    assist_snapshot = None
    assist_receipt = raw_ack.get("capture_assist_evidence")
    if assist_receipt is not None:
        import json
        from .semantic_capture_assist import (CAPTURE_ASSIST_FEATURE_NAMES, HipOnlyCaptureAssist,
            apply_capture_assist_snapshot, capture_assist_features)
        if not isinstance(assist_receipt, Mapping):
            raise ActuatorTargetEffectError("invalid capture assist receipt")
        try:
            before = assist_receipt["state_before"]
            capture_assist_features(before)
            replay = HipOnlyCaptureAssist()
            replay.state = {key: float(before[key]) for key in CAPTURE_ASSIST_FEATURE_NAMES}
            replay.last_tick = int(raw_ack["physics_tick"])-1
            context = assist_receipt["context"]
            if context["dispatch_physics_tick"] != raw_ack["physics_tick"]:
                raise ValueError("assist dispatch clock differs")
            # Knee/hip anchors come from independent pre-dispatch final state.
            # Wheel values are unused by this deliberately two-channel layer.
            recomputed = replay.advance(context=context,
                previous_final_full12=previous+(0.,)*4, physics_dt_s=float(raw_ack["physics_dt_s"]))
            assist_snapshot = recomputed["state_after"]
            if json.dumps(assist_snapshot,sort_keys=True,allow_nan=False) != json.dumps(
                    assist_receipt["state_after"],sort_keys=True,allow_nan=False):
                raise ValueError("assist state transition differs from reconstruction")
            requested_candidate = tuple(a+b for a,b in zip(corrected_native,actual_bias,strict=True))
            assisted_candidate = apply_capture_assist_snapshot(requested_candidate,assist_snapshot)
            if (_finite_values(assist_receipt["candidate_before_assist_full12"],12,"assist input") != requested_candidate
                    or _finite_values(assist_receipt["candidate_after_assist_full12"],12,"assist output") != assisted_candidate
                    or _finite_values(assist_receipt["assist_correction_full12"],12,"assist correction") !=
                    tuple(a-b for a,b in zip(assisted_candidate,requested_candidate,strict=True))):
                raise ValueError("assist transformation differs from independent reconstruction")
            owner_indices = [0,1] if assist_snapshot["active"] else []
            if assist_receipt["owner_indices"] != owner_indices:
                raise ValueError("assist owners differ from current state")
        except (TypeError,ValueError,KeyError) as exc:
            raise ActuatorTargetEffectError(f"invalid capture assist reconstruction: {exc}") from exc

    rr_assist_snapshot = None
    rr_replayed_branches = {}
    rr_assist_receipt = raw_ack.get("rr_capture_assist_evidence")
    if rr_assist_receipt is not None:
        import json
        from .semantic_rr_capture_assist import (RRHipOnlyCaptureAssist,
            apply_rr_capture_assist_snapshot)
        if not isinstance(rr_assist_receipt, Mapping):
            raise ActuatorTargetEffectError("invalid RR capture assist receipt")
        if reference_evidence is None:
            raise ActuatorTargetEffectError("RR capture audit requires independently captured previous issued inputs")
        try:
            rr_before = rr_assist_receipt["state_before"]
            rr_context = rr_assist_receipt["context"]
            if not isinstance(rr_context, Mapping) or rr_context["dispatch_physics_tick"] != raw_ack["physics_tick"]:
                raise ValueError("RR assist dispatch clock differs")
            rr_requested = _finite_values(actuation.projected_residual_full12, 12, "RR current REQUEST")
            if _finite_values(raw_ack["independent_policy_residual_requested_full12"], 12, "RR ACK REQUEST") != rr_requested:
                raise ValueError("RR current requested residual differs from actuation plan")
            current_nominal = Full12Command.from_full12(actuation.frozen_nominal_full12).clamped().to_full12()
            if _finite_values(raw_ack["applied_full12"], 12, "RR issued nominal") != current_nominal:
                raise ValueError("RR current clamped nominal differs from dispatch")
            # This context was captured BEFORE the real dispatch, validated
            # against the adjacent ACK and mapper, then independently rebuilt
            # above. Do not use the new ACK's self-reported past inputs.
            rr_previous_nominal = tracking_reference_context["mapper_pre_state"]["requested_servo_deg"]
            rr_previous_requested = tracking_reference_context["previous_requested_full12"]
            rr_nominal_delta = [current_nominal[i]-rr_previous_nominal[i] for i in (6,7)]

            def replay_rr_branch(bias, native_targets, requested_residual, branch):
                candidate = tuple(a+b for a,b in zip(native_targets,bias,strict=True))
                if assist_snapshot is not None:
                    candidate = apply_capture_assist_snapshot(candidate,assist_snapshot)
                expected_inputs = {
                    "issued_nominal_delta_rr_deg": rr_nominal_delta,
                    "issued_requested_residual_delta_rr_deg": [requested_residual[i]-rr_previous_requested[i] for i in (6,7)],
                    "release_candidate_rr_deg": [candidate[i] for i in (6,7)],
                }
                if branch == "actual":
                    for key, expected in expected_inputs.items():
                        declared = rr_context.get(key)
                        if (not isinstance(declared, (list, tuple)) or len(declared) != 2
                                or any(isinstance(v,bool) for v in declared)
                                or _finite_values(declared, 2, "RR "+key) != tuple(expected)):
                            raise ValueError("RR issued-input evidence differs from independent prestate: "+key)
                context = dict(rr_context)
                context.update(expected_inputs)
                replay = RRHipOnlyCaptureAssist.from_snapshot(rr_before)
                if (replay.last_tick not in (None, int(raw_ack["physics_tick"])-1)
                        or (replay.last_tick is None and rr_before["initialized"] != 0.)):
                    raise ValueError("RR before snapshot is not adjacent to actual dispatch")
                recomputed = replay.advance(context=context,
                    previous_final_full12=previous+(0.,)*4, physics_dt_s=float(raw_ack["physics_dt_s"]))
                snapshot = recomputed["state_after"]
                assisted = apply_rr_capture_assist_snapshot(candidate,snapshot,
                    previous_final_full12=previous+(0.,)*4)
                rr_replayed_branches[branch] = {
                    "same_before_snapshot": True, "before_dispatch_tick": rr_before["last_dispatch_physics_tick"],
                    "issued_inputs": expected_inputs, "state_after": snapshot,
                    "captured_incremental_target": recomputed.get("captured_incremental_target"),
                }
                return snapshot, candidate, assisted

            rr_assist_snapshot, requested_candidate, assisted_candidate = replay_rr_branch(
                actual_bias,corrected_native,rr_requested,"actual")
            if json.dumps(rr_assist_snapshot,sort_keys=True,allow_nan=False) != json.dumps(
                    rr_assist_receipt["state_after"],sort_keys=True,allow_nan=False):
                raise ValueError("RR assist state transition differs from reconstruction")
            if ("captured_incremental_target" not in rr_assist_receipt or json.dumps(
                    rr_replayed_branches["actual"]["captured_incremental_target"],sort_keys=True,allow_nan=False)
                    != json.dumps(rr_assist_receipt["captured_incremental_target"],sort_keys=True,allow_nan=False)):
                raise ValueError("RR captured target increment differs from independent reconstruction")
            if (_finite_values(rr_assist_receipt["candidate_before_assist_full12"],12,"RR assist input") != requested_candidate
                    or _finite_values(rr_assist_receipt["candidate_after_assist_full12"],12,"RR assist output") != assisted_candidate
                    or _finite_values(rr_assist_receipt["assist_correction_full12"],12,"RR assist correction") !=
                    tuple(a-b for a,b in zip(assisted_candidate,requested_candidate,strict=True))):
                raise ValueError("RR assist transformation differs from independent reconstruction")
            if rr_assist_receipt["owner_indices"] != ([6,7] if rr_assist_snapshot["active"] else []):
                raise ValueError("RR assist owner differs from its current state")
        except (TypeError,ValueError,KeyError) as exc:
            raise ActuatorTargetEffectError(f"invalid RR capture assist reconstruction: {exc}") from exc

    wheel_receipt = raw_ack.get("rr_carry_wheel_evidence")
    previous_wheels = None
    if rr_carry_wheel_context is None:
        if wheel_receipt is not None or previous_final_drive_wheel_rad_s is not None:
            raise ActuatorTargetEffectError("unexpected RR support-wheel projection evidence")
    else:
        import json
        from .semantic_rr_carry_wheel import project_rr_carry_wheels
        previous_wheels = _finite_values(previous_final_drive_wheel_rad_s, 4, "previous FINAL wheels")
        if (not isinstance(wheel_receipt, Mapping)
                or rr_carry_wheel_context.get("dispatch_physics_tick") != raw_ack["physics_tick"]
                or wheel_receipt.get("context") != rr_carry_wheel_context
                or tuple(wheel_receipt.get("previous_final_wheel_rad_s", ())) != previous_wheels):
            raise ActuatorTargetEffectError("RR wheel evidence differs from independently captured pre-dispatch state")

    owner_receipt = raw_ack.get("rear_owner_recovery_evidence")
    owner_state = None
    if owner_receipt is not None:
        from .semantic_rear_owner_recovery import MODE as OWNER_MODE, transition as owner_transition, apply_state as apply_owner_state
        try:
            owner_pre = adapter._semantic_owner_pre_dispatch
            expected_context = dict(owner_pre["context"])
            if assist_snapshot is not None and assist_snapshot["active"]:
                expected_context["winning_late_owner"] = [False, False, *expected_context["winning_late_owner"][2:]]
            if (owner_receipt["state_before"] != owner_pre["state_before"]
                    or owner_receipt["context"] != expected_context
                    or owner_receipt["previous_final_full12"] != owner_pre["previous_ack"]["drive_target_full12"]
                    or owner_receipt["previous_requested_full12"] != owner_pre["previous_ack"].get(
                        "independent_policy_residual_requested_full12", [0.] * 12)):
                raise ValueError("owner receipt differs from independent pre-dispatch state")
            owner_state = owner_transition(owner_receipt["state_before"], owner_receipt["context"],
                owner_receipt["previous_final_full12"], owner_receipt["previous_requested_full12"])
            owner_input = tuple(a+b for a,b in zip(corrected_native, actual_bias, strict=True))
            if assist_snapshot is not None:
                owner_input = apply_capture_assist_snapshot(owner_input, assist_snapshot)
            owner_output = apply_owner_state(owner_input, actuation.projected_residual_full12,
                owner_state, owner_receipt["capacities_full12"])
            if (owner_receipt["mode"] != OWNER_MODE or owner_receipt["state_after"] != owner_state
                    or tuple(owner_receipt["previous_final_full12"][:8]) != tuple(previous)
                    or tuple(owner_receipt["requested_full12"]) != tuple(actuation.projected_residual_full12)
                    or tuple(owner_receipt["candidate_before_full12"]) != owner_input
                    or tuple(owner_receipt["candidate_after_full12"]) != owner_output):
                raise ValueError("owner recovery differs from independent reconstruction")
        except (TypeError, KeyError, ValueError, AttributeError) as exc:
            raise ActuatorTargetEffectError(str(exc)) from exc

    def physical_targets(bias: Sequence[float], native_targets: Sequence[float] = corrected_native,
                         *, verify_wheel_receipt: bool = False, rr_branch: str = "zero_current_policy") -> Any:
        servo = []
        assisted = None
        branch_rr_snapshot = None
        if assist_snapshot is not None:
            assisted = apply_capture_assist_snapshot(
                tuple(a+b for a,b in zip(native_targets,bias,strict=True)),assist_snapshot)
        if rr_assist_snapshot is not None:
            if rr_branch == "actual":
                branch_rr_snapshot = rr_assist_snapshot
                assisted = assisted_candidate
            else:
                try:
                    branch_rr_snapshot, _, assisted = replay_rr_branch(bias,native_targets,(0.,)*12,rr_branch)
                except (TypeError,ValueError,KeyError) as exc:
                    raise ActuatorTargetEffectError(f"invalid RR {rr_branch} reconstruction: {exc}") from exc
        owner_candidate = None
        if owner_state is not None:
            base_candidate = assisted if assisted is not None else tuple(a+b for a,b in zip(native_targets,bias,strict=True))
            owner_candidate = apply_owner_state(base_candidate,
                actuation.projected_residual_full12 if rr_branch == "actual" else (0.,)*12,
                owner_state, owner_receipt["capacities_full12"])
        for index, name in enumerate(SERVO_ORDER):
            lower, upper = servo_limits_deg(name)
            native_target, effective_bias = native_targets[index], bias[index]
            if assist_snapshot is not None and index < 2 and assist_snapshot["active"]:
                native_target, effective_bias = assisted[index], 0.
            if branch_rr_snapshot is not None and index in (6,7) and branch_rr_snapshot["active"]:
                native_target, effective_bias = assisted[index], 0.
            if owner_candidate is not None and index in owner_receipt["owner_indices"]:
                native_target, effective_bias = owner_candidate[index], 0.
            servo.append(bounded_drive_feedback_step(
                previous_deg=previous[index],
                native_deg=native_target,
                bias_deg=effective_bias,
                maximum_delta_deg=maximum_delta,
                lower_deg=lower,
                upper_deg=upper,
            ))
        # Full12Command owns hard wheel limits; build_physical_batch owns all
        # standing offsets, joint signs, and degree-to-radian conversion.
        candidate = Full12Command(tuple(servo), tuple(native_targets[index] + bias[index]
            for index in range(8, 12))).clamped().to_full12()
        if rr_carry_wheel_context is not None:
            reconstructed = project_rr_carry_wheels(candidate, context=rr_carry_wheel_context,
                previous_final_wheel_rad_s=previous_wheels, physics_dt_s=float(raw_ack["physics_dt_s"]))
            if verify_wheel_receipt and json.dumps(reconstructed, sort_keys=True, allow_nan=False) != json.dumps(
                    wheel_receipt, sort_keys=True, allow_nan=False):
                raise ActuatorTargetEffectError("RR wheel projection differs from independent reconstruction")
            candidate = tuple(reconstructed["output_full12"])
        command = Full12Command.from_full12(candidate).clamped()
        return build_physical_batch(command, adapter.standing_pose_deg)

    actual_physical = physical_targets(actual_bias, verify_wheel_receipt=True, rr_branch="actual")
    counterfactual_physical = physical_targets(zero_policy_bias)
    robot = adapter.robot
    servo_ids = list(adapter.joint_map.servo_ids)
    wheel_ids = list(adapter.joint_map.wheel_ids)
    if len(servo_ids) != 8 or len(wheel_ids) != 4 or len(set(servo_ids + wheel_ids)) != 12:
        raise ActuatorTargetEffectError("audit requires twelve distinct canonical joint IDs")

    def read_targets(owner: Any, attribute: str, ids: list[int]) -> Any:
        tensor = getattr(owner, attribute, None)
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2 or tensor.shape[0] != 1:
            raise ActuatorTargetEffectError(f"missing single-articulation target tensor: {attribute}")
        if tensor.dtype != torch.float32:
            raise ActuatorTargetEffectError(f"target tensor is not float32: {attribute}")
        selected = tensor[:, ids].detach().clone()
        if not bool(torch.isfinite(selected).all().item()):
            raise ActuatorTargetEffectError(f"non-finite actual target tensor: {attribute}")
        return selected

    staged_servo = read_targets(robot.data, "joint_pos_target", servo_ids)
    staged_wheel = read_targets(robot.data, "joint_vel_target", wheel_ids)
    dispatched_servo = read_targets(robot, "_joint_pos_target_sim", servo_ids)
    dispatched_wheel = read_targets(robot, "_joint_vel_target_sim", wheel_ids)
    if not torch.equal(staged_servo, dispatched_servo) or not torch.equal(staged_wheel, dispatched_wheel):
        raise ActuatorTargetEffectError("setter targets differ from completed PhysX dispatch buffers")

    def cast_targets(physical: Any) -> tuple[Any, Any]:
        return (
            dispatched_servo.new_tensor([physical.servo_target_rad]),
            dispatched_wheel.new_tensor([physical.wheel_target_rad_s]),
        )

    expected_servo, expected_wheel = cast_targets(actual_physical)
    if not torch.equal(expected_servo, dispatched_servo) or not torch.equal(expected_wheel, dispatched_wheel):
        raise ActuatorTargetEffectError("frozen mapping reconstruction differs from actual dispatch")
    nominal_servo, nominal_wheel = cast_targets(counterfactual_physical)
    # Numeric inequality intentionally does not count +0 versus -0 as effect.
    changed = torch.cat((dispatched_servo != nominal_servo, dispatched_wheel != nominal_wheel), dim=1)
    changed_channels = [bool(value) for value in changed.cpu().tolist()[0]]

    def record(servo: Any, wheel: Any) -> dict[str, list[float]]:
        return {
            "servo_position_rad": servo.cpu().tolist()[0],
            "wheel_velocity_rad_s": wheel.cpu().tolist()[0],
        }

    request = {
        "policy_request_phase": None,
        "raw_policy_action_full12": None,
        "phase_mask_full12": None,
    }
    if policy_request is not None:
        request = actuator_target_audit_request(
            policy_request["policy_request_phase"],
            policy_request["raw_policy_action_full12"],
            policy_request["phase_mask_full12"],
        )
    result = {
        "schema": ACTUATOR_TARGET_EFFECT_SCHEMA,
        "verified": True,
        "source_phase_id": source_phase_id,
        **request,
        "physics_tick": int(raw_ack["physics_tick"]),
        "canonical_order": list(FULL12_ORDER),
        "projected_residual_full12": list(_finite_values(actuation.projected_residual_full12, 12, "projected residual")),
        "changed_channels_full12": changed_channels,
        "changed_target_channel_count": sum(changed_channels),
        "actual_native_targets": record(dispatched_servo, dispatched_wheel),
        "counterfactual_native_targets": record(nominal_servo, nominal_wheel),
        "native_target_delta": record(dispatched_servo - nominal_servo, dispatched_wheel - nominal_wheel),
        "target_dtype": str(dispatched_servo.dtype),
        "servo_target_dtype": str(dispatched_servo.dtype),
        "wheel_target_dtype": str(dispatched_wheel.dtype),
        "setter_dispatch_targets_equal": True,
        "actual_mapping_matches_dispatch": True,
        "same_tick_counterfactual": True,
        "counterfactual_scope": "same_pre_tick_state_without_current_ppo_residual",
        "actual_target_source": "robot._joint_pos_target_sim/robot._joint_vel_target_sim_after_existing_write_data_to_sim",
        "previous_final_drive_servo_deg": list(previous),
        "native_drive_target_full12": list(native),
        "controller_drive_bias_full12": list(controller_bias),
        "combined_post_mapper_bias_full12": list(combined_bias),
    }
    if headroom is not None:
        result.update(policy_headroom_mode=policy_headroom_mode,
                      policy_headroom_evidence=headroom)
    if reference_evidence is not None:
        result.update(tracking_reference_mode=tracking_reference_mode,
                      tracking_reference_evidence=reference_evidence,
                      tracking_reference_previous_ack_independently_verified=True)
    if assist_snapshot is not None:
        result.update(capture_assist_evidence=dict(assist_receipt),
            capture_assist_state_transition_independently_reconstructed=True,
            capture_assist_owned_channels_full12=[bool(i<2 and assist_snapshot["active"]) for i in range(12)],
            counterfactual_scope="same_pre_tick_state_and_same_capture_assist_without_current_ppo_residual",
            policy_request_execution_semantics="sample_unchanged_state_dependent_FL_target_transform",
            all12_policy_channels_unmodified_at_actuator=not assist_snapshot["active"])
    if rr_assist_snapshot is not None:
        result.update(rr_capture_assist_evidence=dict(rr_assist_receipt),
            rr_capture_assist_state_transition_independently_reconstructed=True,
            rr_capture_issued_inputs_independently_verified=True,
            rr_capture_assist_branch_replays=rr_replayed_branches,
            rr_capture_counterfactual_state_semantics="same_prestate_distinct_actual_zero_current_policy_and_geometry_zero_replays",
            rr_capture_assist_owned_channels_full12=[bool(i in (6,7) and rr_assist_snapshot["active"]) for i in range(12)],
            counterfactual_scope="same_pre_tick_state_and_same_FL_RR_assists_without_current_ppo_residual",
            policy_request_execution_semantics="sample_unchanged_declared_state_dependent_FL_RR_target_transforms",
            all12_policy_channels_unmodified_at_actuator=not (
                rr_assist_snapshot["active"] or (assist_snapshot is not None and assist_snapshot["active"])))
    if wheel_receipt is not None:
        result.update(rr_carry_wheel_evidence=dict(wheel_receipt),
            rr_carry_wheel_context_and_previous_FINAL_independently_verified=True,
            previous_final_drive_wheel_rad_s=list(previous_wheels),
            counterfactual_scope="same_pre_tick_state_same_FL_RR_assists_and_support_wheel_projection_without_current_ppo_residual",
            policy_request_execution_semantics="raw_sample_unchanged_declared_servo_and_support_wheel_transforms",
            all12_policy_channels_unmodified_at_actuator=bool(
                result.get("all12_policy_channels_unmodified_at_actuator", True)
                and not wheel_receipt["envelope_active"]))
    if geometry_enabled:
        # Remove geometry only in this third, zero-current-policy branch.
        # The existing actual-minus-counterfactual fields remain PPO-only.
        raw_zero_bias = controller_bias
        if headroom is not None:
            raw_zero = project_semantic_servo_headroom(
                native_full12=native, controller_bias_full12=controller_bias,
                projected_residual_full12=(0.,)*12)
            raw_zero_bias = tuple(raw_zero["effective_combined_post_mapper_bias_full12"])
        raw_nominal_servo, raw_nominal_wheel = cast_targets(physical_targets(raw_zero_bias, native,
            rr_branch="zero_current_policy_without_geometry"))
        geometry_changed = torch.cat((nominal_servo != raw_nominal_servo,
                                      nominal_wheel != raw_nominal_wheel), dim=1)
        geometry_channels = [bool(value) for value in geometry_changed.cpu().tolist()[0]]
        result.update({
            "geometry_adjusted_native_full12": list(corrected_native),
            "nominal_geometry_adjustment_full12": list(adjustment),
            "nominal_geometry_evidence": dict(raw_ack["nominal_geometry_evidence"]),
            "raw_nominal_native_targets": record(raw_nominal_servo, raw_nominal_wheel),
            "geometry_nominal_native_targets": record(nominal_servo, nominal_wheel),
            "nominal_geometry_native_target_delta": record(nominal_servo-raw_nominal_servo,
                                                            nominal_wheel-raw_nominal_wheel),
            "nominal_geometry_changed_channels_full12": geometry_channels,
            "nominal_geometry_changed_target_channel_count": sum(geometry_channels),
            "nominal_geometry_counterfactual_scope": "same_pre_tick_state_zero_current_ppo_without_nominal_geometry",
        })
    return result


def _finite_values(values: Sequence[float], size: int, label: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != size or any(not math.isfinite(value) for value in result):
        raise ActuatorTargetEffectError(f"{label} must contain {size} finite values")
    return result
