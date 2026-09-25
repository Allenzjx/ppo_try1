"""Stateless, declared pause of an already-issued dependent source pursuit.

Only FL hip/knee and RL hip are eligible. RR capture and all wheel/stop targets
are untouched. A paused axis keeps adjacent FINAL plus the student's requested
residual change, projected through enabled original headroom before unchanged
hard clamp/slew. It does not freeze the
policy, synthesize support, write nominal history, or advance a second mapper.
"""
from __future__ import annotations

from collections.abc import Mapping
import copy
import math

MODE = "issued_continuation_source_pause_requested_delta_v1"
INDICES = (0, 1, 4)
FEATURE_NAMES = ("pause_source_FL_hip", "pause_source_FL_knee", "pause_source_RL_hip")
FLAGS = ("active", "live", "source_strong_started", "rr_current_bearing",
         "rr_support_continuation_valid", "rl_current_swing")


def vector(value, name):
    if (not isinstance(value, (list, tuple)) or len(value) != 12
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in value)):
        raise ValueError("source pause requires finite full12 " + name)
    return tuple(float(v) for v in value)


def validate_context(context):
    if not isinstance(context, Mapping) or context.get("mode") != MODE:
        raise ValueError("unknown continuation source pause context")
    if any(type(context.get(key)) is not bool for key in FLAGS):
        raise ValueError("continuation source pause flags must be Boolean")
    owners = context.get("owned_indices")
    if (not isinstance(owners, (list, tuple)) or len(owners) != len(set(owners))
            or any(type(i) is not int or i not in INDICES for i in owners)):
        raise ValueError("source pause ownership must be a subset of FL hip/knee and RL hip")
    if type(context.get("dispatch_physics_tick")) is not int:
        raise ValueError("source pause requires an exact dispatch tick")


def make_context(inputs, *, dispatch_physics_tick):
    context = dict(inputs, mode=MODE, dispatch_physics_tick=dispatch_physics_tick)
    validate_context(context)
    return context


def paused_indices(context):
    validate_context(context)
    pause = (context["active"] and context["live"] and context["source_strong_started"]
        and not context["rr_current_bearing"] and not context["rr_support_continuation_valid"]
        and not context["rl_current_swing"])
    return tuple(i for i in INDICES if pause and i in context["owned_indices"])


def features(context):
    if context is None:
        return (0.,) * 3
    selected = paused_indices(context)
    return tuple(float(i in selected) for i in INDICES)


def project(candidate, request, *, context, previous_ack, previous_tick, write_count,
            policy_headroom_mode=None):
    """Pure replayable target transform from independently captured pre-state."""
    validate_context(context)
    candidate, request = vector(candidate, "candidate"), vector(request, "policy request")
    if (type(previous_tick) is not int or type(write_count) is not int
            or not isinstance(previous_ack, Mapping)
            or previous_ack.get("physics_tick") != previous_tick
            or previous_ack.get("write_count") != write_count
            or context["dispatch_physics_tick"] != previous_tick + 1):
        raise ValueError("source pause requires the adjacent committed ACK")
    previous_final = vector(previous_ack.get("drive_target_full12"), "previous FINAL")
    # The original reset/settle ACK legitimately has no learned residual.
    previous_request = vector(previous_ack.get("independent_policy_residual_requested_full12", (0.,)*12),
                              "previous issued policy request")
    indices = paused_indices(context)
    output = list(candidate)
    for i in indices:
        output[i] = previous_final[i] + request[i] - previous_request[i]
    unbounded_output = list(output)
    delta_headroom = None
    if policy_headroom_mode is not None:
        from .semantic_headroom import HEADROOM_MODE, project_semantic_servo_headroom
        if policy_headroom_mode != HEADROOM_MODE:
            raise ValueError("unknown source pause headroom mode")
        # This is a change of the requested residual, not a second policy
        # residual or mapper advance. The committed FINAL is the held baseline.
        # Preserve original headroom semantics: outside-band zero stays zero,
        # inward recovery is allowed, and additional outward motion is blocked.
        requested_delta = [0.] * 12
        for i in indices:
            requested_delta[i] = request[i] - previous_request[i]
        delta_headroom = project_semantic_servo_headroom(
            native_full12=previous_final, controller_bias_full12=(0.,)*12,
            projected_residual_full12=requested_delta)
        projected = delta_headroom["candidate_native_target_before_final_slew_full12"]
        for i in indices:
            output[i] = projected[i]
    if any(not math.isfinite(v) for v in output):
        raise ValueError("source pause produced a nonfinite candidate")
    receipt = dict(mode=MODE, context=copy.deepcopy(dict(context)), pause_indices=list(indices),
        observation_features=list(features(context)),
        previous_ack_physics_tick=previous_tick, previous_ack_write_count=write_count,
        previous_final_full12=list(previous_final), previous_requested_full12=list(previous_request),
        requested_full12=list(request), candidate_before_full12=list(candidate),
        candidate_after_full12=output,
        correction_full12=[a-b for a,b in zip(output,candidate)],
        target_semantics="adjacent_FINAL_plus_current_requested_residual_delta_before_original_clamp_slew",
        policy_raw_sample_and_logp_unchanged=True, policy_changes_permitted=True,
        nominal_history_modified=False, second_mapper_advance=False,
        wheel_stop_modified=False, RR_target_modified=False, task_contact_credit=False,
        task_assist=False)
    if delta_headroom is not None:
        receipt.update(
            headroom_delta_projection_revision="adjacent_FINAL_requested_delta_headroom_v1",
            policy_headroom_mode=policy_headroom_mode,
            candidate_before_delta_headroom_full12=unbounded_output,
            requested_policy_change_delta_full12=requested_delta,
            effective_policy_change_delta_full12=delta_headroom["effective_policy_residual_full12"],
            delta_headroom_evidence=delta_headroom,
            delta_semantics="change_of_requested_residual_around_adjacent_FINAL_not_absolute_PPO_residual",
            target_semantics="adjacent_FINAL_plus_headroom_bounded_requested_delta_before_original_clamp_slew")
    return receipt
