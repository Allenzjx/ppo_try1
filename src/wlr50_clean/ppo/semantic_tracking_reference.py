"""V3-only, read-only previous-ACK REQUEST tracking reference.

No mapper advance, physical write, sensor replacement, or new evolving history.
The reserve bounds the desired mapper correction, not its old slewed state or
subsequent geometry/controller composition. It is not a physical guarantee.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import json
import math
from numbers import Real

from wlr50_clean.infrastructure.command_batch import FULL12_ORDER, SERVO_ORDER, SERVO_COMMAND_SIGN, servo_limits_deg
from wlr50_clean.infrastructure.robot_adapter import RobotAdapterError, _joint_matrix, _row_values

MODE = "previous_ack_requested_servo_reference_v1"
CONTEXT_SCHEMA = "wlr50_clean.semantic_tracking_reference_context.v1"
EVIDENCE_SCHEMA = "wlr50_clean.semantic_tracking_reference.v1"


class SemanticTrackingReferenceError(ValueError):
    """Missing same-tick provenance or invalid reference arithmetic."""


def _require(condition, message):
    if not condition:
        raise SemanticTrackingReferenceError(message)


def _number(value, label):
    _require(not isinstance(value, bool) and isinstance(value, Real), f"{label} must be a finite real")
    value = float(value)
    _require(math.isfinite(value), f"{label} must be finite")
    return value


def _integer(value, label, minimum=0):
    _require(type(value) is int and value >= minimum, f"{label} must be an integer >= {minimum}")
    return value


def _vector(values, count, label):
    _require(isinstance(values, Sequence) and not isinstance(values, (str, bytes)) and len(values) == count,
             f"{label} must contain {count} values")
    return [_number(value, label) for value in values]


def _booleans(values, label):
    _require(isinstance(values, Sequence) and len(values) == 8 and all(type(x) is bool for x in values),
             f"{label} must contain eight booleans")
    return list(values)


def _names(values):
    _require(isinstance(values, Sequence) and not isinstance(values, (str, bytes)), "tracking names must be a sequence")
    _require(all(type(x) is str and x in SERVO_ORDER for x in values) and len(set(values)) == len(values),
             "tracking names must be unique canonical servo names")
    return list(values)


def _state(mapper):
    result = {}
    for name, attribute, boolean in (
        ("requested_servo_deg", "_requested", False),
        ("applied_drive_command_deg", "_applied", False),
        ("tracking_compensation_deg", "_compensation", False),
        ("nominal_target_reached", "_nominal_reached", True),
        ("tracking_active", "_tracking_active", True),
        ("retiring_stale_bias", "_retiring_stale_bias", True),
    ):
        values = getattr(mapper, attribute, None)
        _require(isinstance(values, Mapping) and set(values) == set(SERVO_ORDER), f"mapper {attribute} unavailable")
        row = [values[key] for key in SERVO_ORDER]
        result[name] = _booleans(row, name) if boolean else _vector(row, 8, name)
    return result


def capture_tracking_reference_context(adapter, *, physics_tick: int,
                                       bootstrap_physics_tick: int | None = None) -> dict:
    """Bind live q/prestate to the immediately preceding, already-written ACK.

    Bootstrap is only an explicitly identified reset/prime dispatch, never a
    missing-history fallback. The caller derives its fixed tick from reset.
    """
    tick = _integer(physics_tick, "physics_tick", 1)
    if bootstrap_physics_tick is not None:
        _integer(bootstrap_physics_tick, "bootstrap_physics_tick", 1)
    ack = getattr(adapter, "last_ack", None)
    _require(isinstance(ack, Mapping), "previous ACK is required, including at bootstrap")
    _require(ack.get("schema") == "wlr50_clean.atomic_full12_ack.v1", "invalid previous ACK schema")
    previous_tick = _integer(ack.get("physics_tick"), "previous ACK physics_tick")
    _require(previous_tick == _integer(getattr(adapter, "_last_physics_tick", None), "adapter last tick") == tick - 1,
             "previous ACK/native dispatch ticks are not adjacent")
    writes = _integer(getattr(adapter, "write_count", None), "adapter write_count", 1)
    _require(_integer(ack.get("write_count"), "ACK write_count", 1) == writes,
             "previous ACK write count differs from adapter")
    _require(type(ack.get("articulation_writes_this_call")) is int and ack["articulation_writes_this_call"] == 1,
             "previous ACK is not one completed write")
    _require(list(ack.get("canonical_order", ())) == list(FULL12_ORDER), "previous ACK order differs")
    mapper = adapter.servo_target_mapper
    feedback_tick = _integer(mapper.feedback_tick, "mapper feedback_tick", 1)
    sample_tick = _integer(ack.get("servo_tracking_feedback_sample_tick"), "previous feedback sample tick")
    _require(sample_tick + 1 == feedback_tick == writes, "mapper/ACK feedback clock or write count differs")
    state = _state(mapper)
    for field, ack_field in (("requested_servo_deg", "applied_full12"),
                             ("applied_drive_command_deg", "servo_native_drive_command_deg"),
                             ("tracking_compensation_deg", "servo_tracking_compensation_deg")):
        count = 12 if ack_field == "applied_full12" else 8
        _require(state[field] == _vector(ack.get(ack_field), count, ack_field)[:8], "previous ACK differs from live mapper " + field)
    for field, ack_field in (("nominal_target_reached", "servo_nominal_target_reached"),
                             ("tracking_active", "servo_tracking_active")):
        _require(state[field] == _booleans(ack.get(ack_field), ack_field), "previous ACK differs from live mapper " + field)
    _names(ack.get("tracking_servo_names"))
    semantic_keys = ("semantic_residual_composition", "independent_policy_residual_requested_full12",
                     "policy_headroom_mode", "policy_headroom_evidence", "tracking_reference_mode", "tracking_reference_evidence")
    semantic = any(key in ack for key in semantic_keys)
    if semantic:
        _require(ack.get("semantic_residual_composition") == "independent_post_mapper_residual.v1",
                 "previous semantic ACK has an incompatible residual composition")
        _require(ack.get("tracking_reference_mode") == MODE, "previous semantic ACK lacks the required reference mode")
        receipt = ack.get("tracking_reference_evidence")
        _require(isinstance(receipt, Mapping) and receipt.get("schema") == EVIDENCE_SCHEMA and receipt.get("mode") == MODE,
                 "previous semantic ACK lacks reference evidence")
        _require(type(receipt.get("dispatch_physics_tick")) is int and receipt["dispatch_physics_tick"] == previous_tick,
                 "previous reference receipt dispatch tick differs")
        _require(type(receipt.get("mapper_feedback_tick")) is int and receipt["mapper_feedback_tick"] == sample_tick,
                 "previous reference receipt feedback tick differs")
        previous = _vector(ack.get("independent_policy_residual_requested_full12"), 12, "previous REQUEST residual")
        controller = _vector(ack.get("bounded_controller_bias_requested_full12"), 12, "previous controller bias")
        combined = _vector(ack.get("drive_feedback_bias_requested_full12"), 12, "previous combined bias")
        _require(combined == [c+r for c, r in zip(controller, previous)], "previous requested residual composition differs")
        for optional in ("ppo_projected_residual_full12",):
            if optional in ack:
                _require(_vector(ack[optional], 12, optional) == previous, "previous policy request aliases differ")
        source = "previous_semantic_ack_requested_residual"
    else:
        _require(bootstrap_physics_tick is not None and tick == bootstrap_physics_tick,
                 "original ACK allowed only at the explicit bootstrap dispatch tick")
        for key in ("requested_full12", "applied_full12", "native_drive_target_full12", "drive_target_full12",
                    "drive_feedback_bias_requested_full12"):
            _require(all(x == 0.0 for x in _vector(ack.get(key), 12, key)), "bootstrap requires original zero-command ACK")
        _require(not ack["tracking_servo_names"] and not any(state["tracking_active"])
                 and not any(state["retiring_stale_bias"]) and all(state["nominal_target_reached"])
                 and not any(state["tracking_compensation_deg"]), "bootstrap requires settled zero tracking state")
        _require(sample_tick == previous_tick and writes == tick, "bootstrap native and mapper origin differ")
        previous = [0.0]*12
        source = "explicit_original_zero_bootstrap"
    standing = getattr(adapter, "standing_pose_deg", None)
    _require(isinstance(standing, Mapping) and set(standing) == set(SERVO_ORDER), "standing pose unavailable")
    standing_values = [_number(standing[name], "standing pose") for name in SERVO_ORDER]
    _require(getattr(mapper, "standing_pose_deg", None) == dict(standing), "adapter and mapper standing poses differ")
    ids = tuple(adapter.joint_map.servo_ids)
    try:
        matrix = _joint_matrix(adapter.robot, "joint_pos")
    except RobotAdapterError as exc:
        raise SemanticTrackingReferenceError("actual joint positions are unavailable") from exc
    _require(matrix.shape[0] == 1 and len(ids) == 8 and len(set(ids)) == 8
             and all(type(i) is int and 0 <= i < matrix.shape[1] for i in ids), "reference capture requires one valid canonical servo row")
    try:
        actual = _vector(_row_values(matrix[:, list(ids)]), 8, "actual measured physical radians")
    except RobotAdapterError as exc:
        raise SemanticTrackingReferenceError("actual joint positions are invalid") from exc
    context = {"schema": CONTEXT_SCHEMA, "mode": MODE,
        "dispatch_physics_tick": tick, "previous_ack_physics_tick": previous_tick,
        "previous_ack_write_count": writes, "mapper_feedback_tick": feedback_tick,
        "previous_feedback_sample_tick": sample_tick, "bootstrap_physics_tick": bootstrap_physics_tick,
        "previous_request_source": source, "previous_requested_full12": previous,
        "actual_measured_physical_rad": actual, "standing_pose_deg": standing_values,
        "tracking_gain": _number(mapper.tracking_gain, "tracking_gain"),
        "tracking_limit_deg": _number(mapper.tracking_limit_deg, "tracking_limit_deg"),
        "maximum_delta_deg": _number(mapper.maximum_delta_deg, "maximum_delta_deg"),
        "feedback_interval_ticks": _integer(mapper.feedback_interval_ticks, "feedback interval", 1),
        "mapper_pre_state": state}
    _validate_context(context)
    return context


def _validate_context(context):
    _require(isinstance(context, Mapping) and context.get("schema") == CONTEXT_SCHEMA and context.get("mode") == MODE,
             "invalid tracking reference context")
    tick = _integer(context.get("dispatch_physics_tick"), "dispatch tick", 1)
    _require(_integer(context.get("previous_ack_physics_tick"), "previous ACK tick") == tick-1, "context ACK tick is stale")
    feedback = _integer(context.get("mapper_feedback_tick"), "feedback tick", 1)
    _require(_integer(context.get("previous_feedback_sample_tick"), "previous sample") + 1 == feedback
             == _integer(context.get("previous_ack_write_count"), "write count", 1), "context feedback clock differs")
    _integer(context.get("feedback_interval_ticks"), "feedback interval", 1)
    for key in ("tracking_gain", "tracking_limit_deg", "maximum_delta_deg"):
        _require(_number(context.get(key), key) >= 0, key + " must be nonnegative")
    _vector(context.get("previous_requested_full12"), 12, "previous requested residual")
    _vector(context.get("actual_measured_physical_rad"), 8, "actual q")
    _vector(context.get("standing_pose_deg"), 8, "standing pose")
    state = context.get("mapper_pre_state")
    _require(isinstance(state, Mapping), "mapper prestate missing")
    for key in ("requested_servo_deg", "applied_drive_command_deg", "tracking_compensation_deg"):
        _vector(state.get(key), 8, key)
    for key in ("nominal_target_reached", "tracking_active", "retiring_stale_bias"):
        _booleans(state.get(key), key)


def build_tracking_reference(context: Mapping, *, requested_command_deg: Sequence,
                             tracking_servo_names: Sequence) -> dict:
    """Compute a local reference only where frozen advance consumes feedback."""
    _validate_context(context)
    requested = _vector(requested_command_deg, 8, "requested command")
    scheduled = _names(tracking_servo_names)
    actual = _vector(context["actual_measured_physical_rad"], 8, "actual q")
    previous = _vector(context["previous_requested_full12"], 12, "previous request")
    gain, limit = context["tracking_gain"], context["tracking_limit_deg"]
    state = context["mapper_pre_state"]
    sample = context["mapper_feedback_tick"] % context["feedback_interval_ticks"] == 0
    reference, channels = list(actual), []
    for i, name in enumerate(SERVO_ORDER):
        hard_lo, hard_hi = servo_limits_deg(name)
        nominal = max(hard_lo, min(hard_hi, requested[i]))
        sign = SERVO_COMMAND_SIGN[name]
        nominal_physical = context["standing_pose_deg"][i] + sign*nominal
        e0 = (nominal_physical-math.degrees(actual[i]))/sign
        changed = not math.isclose(nominal, state["requested_servo_deg"][i], rel_tol=0.0, abs_tol=1e-9)
        eligible = name in scheduled and not changed and state["nominal_target_reached"][i] and sample
        r = previous[i] if eligible and gain > 0 else 0.0
        raw0, raw1 = gain*e0, gain*(e0+r)
        _require(all(math.isfinite(x) for x in (nominal_physical,e0,raw0,raw1)), "reference arithmetic overflow")
        c0, c1 = max(-limit, min(limit, raw0)), max(-limit, min(limit, raw1))
        lower = max(-limit, min(c0, 0.0, hard_lo+2.0-nominal))
        upper = min(limit, max(c0, 0.0, hard_hi-2.0-nominal))
        bounded = max(lower, min(upper, c1))
        clipped = bounded != c1
        used = eligible and gain > 0 and r != 0.0
        if used:
            reference[i] = (math.radians(nominal_physical-sign*(bounded/gain)) if clipped
                            else actual[i]-sign*math.radians(r))
        _require(math.isfinite(reference[i]), "reference angle overflow")
        achieved_raw = gain*((nominal_physical-math.degrees(reference[i]))/sign)
        _require(math.isfinite(achieved_raw), "reference reconstruction overflow")
        achieved = max(-limit, min(limit, achieved_raw))
        channels.append({"servo":name, "scheduled":name in scheduled,
            "ended_tracking":state["tracking_active"][i] and name not in scheduled,
            "retiring_stale_bias_before":state["retiring_stale_bias"][i],
            "nominal_changed":changed, "nominal_reached_before":state["nominal_target_reached"][i],
            "feedback_sample_tick":sample, "feedback_eligible":eligible, "reference_used":used,
            "previous_requested_deg":previous[i], "active_reference_deg":r,
            "nominal_deg":nominal, "current_actual_canonical_error_deg":e0,
            "desired_original_c0_deg":c0, "desired_reference_c1_deg":c1,
            "allowed_desired_interval_deg":[lower,upper], "bounded_desired_c1_deg":bounded,
            "desired_reference_clipped":clipped, "reconstructed_desired_deg":achieved,
            "reconstruction_error_deg":achieved-bounded if used else None,
            "reserved_native_band_deg":[hard_lo+2.0,hard_hi-2.0]})
    result = copy.deepcopy(dict(context))
    result.update(schema=EVIDENCE_SCHEMA, mapper_computational_reference_rad=reference,
                  requested_command_deg=requested, tracking_servo_names=scheduled, channels=channels,
                  reference_reads_current_policy_residual=False,
                  reference_semantics="previous_ACK_filtered_REQUEST_not_effective_or_realized_residual",
                  reserve_scope="desired_mapper_correction_only; original_baseline_exception_and_old_slew_and_later_geometry_controller_composition_retained",
                  physical_sensor_modified=False, mapper_advances=0, articulation_writes=0)
    try:
        json.dumps(result, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise SemanticTrackingReferenceError("reference evidence must be finite JSON") from exc
    return result
