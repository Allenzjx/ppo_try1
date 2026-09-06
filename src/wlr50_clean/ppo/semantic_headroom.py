"""Pure post-mapper residual headroom projection; no mapper, slew, or dispatch.

The caller must supply the unique same-tick geometry-corrected native target.
All servo numbers use canonical logical degrees, not absolute PhysX radians.
This calculation cannot itself verify the caller's clock/geometry provenance.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real

from wlr50_clean.infrastructure.command_batch import (
    FULL12_ORDER, SERVO_ORDER, WHEEL_VELOCITY_LIMIT_RAD_S, servo_limits_deg,
)
from wlr50_clean.infrastructure.servo_target_mapper import SERVO_TRACKING_COMPENSATION_MAX_DEG

HEADROOM_MODE = "same_tick_post_mapper_servo_margin_v1"
SERVO_RESERVE_DEG = 2.0


class SemanticHeadroomError(ValueError):
    """Invalid pure arithmetic input or incompatible opt-in configuration."""


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise SemanticHeadroomError(f"{label} must be a finite real number, not bool/string")
    result = float(value)
    if not math.isfinite(result):
        raise SemanticHeadroomError(f"{label} must be finite")
    return result


def _full12(values: Sequence[float], label: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or len(values) != 12:
        raise SemanticHeadroomError(f"{label} must be a twelve-value sequence")
    return tuple(_number(value, f"{label}[{index}]") for index, value in enumerate(values))


def validate_semantic_servo_headroom_config(mode: str, joint_safety_margin_deg: Mapping[str, object]) -> None:
    """Startup-loader seam: validate the opt-in against the actual base config.

    Pure validation only. The caller reads the base config; this module does
    not read files or silently assume a differently configured margin is safe.
    """
    if type(mode) is not str or mode != HEADROOM_MODE:
        raise SemanticHeadroomError("unknown post-mapper headroom mode")
    if not isinstance(joint_safety_margin_deg, Mapping):
        raise SemanticHeadroomError("joint_safety_margin_deg must be the base-config mapping")
    for name in ("hip", "knee"):
        if name not in joint_safety_margin_deg or _number(joint_safety_margin_deg[name], name) != SERVO_RESERVE_DEG:
            raise SemanticHeadroomError("this mode requires the original 2-degree hip and knee reserves")


def _add_offset(base: float, offset: float) -> float:
    # Exact zero never causes clamp-to-reserve or signed-zero relocation.
    result = base if offset == 0.0 else base + offset
    if not math.isfinite(result):
        raise SemanticHeadroomError("composition overflowed to a non-finite value")
    return result


def project_semantic_servo_headroom(
    native_full12: Sequence[float],
    controller_bias_full12: Sequence[float],
    projected_residual_full12: Sequence[float],
) -> dict:
    """Clip only requested servo residual around ``geometry_native + c``.

    For servo i: L/U = frozen hard limits inset by the original 2 degrees;
    interval = [min(0,L-(m_g+c)), max(0,U-(m_g+c))]. Wheels pass through.
    Baselines outside the reserved band are not moved: zero remains zero and
    a small inward request need not instantly return to that band. This is not
    a hard target/slew projection, nor a guarantee of actual physical motion.
    """
    native = _full12(native_full12, "geometry-corrected native_full12")
    controller = _full12(controller_bias_full12, "bounded controller_bias_full12")
    requested = _full12(projected_residual_full12, "projected_residual_full12")
    # Preserve the original controller envelope; policy residual is not subject
    # to that controller-only 10-degree bound. Match its existing 1e-12 tolerance.
    if any(abs(value) > SERVO_TRACKING_COMPENSATION_MAX_DEG + 1e-12 for value in controller[:8]):
        raise SemanticHeadroomError("controller servo bias exceeds the original envelope")
    if any(abs(value) > WHEEL_VELOCITY_LIMIT_RAD_S + 1e-12 for value in controller[8:]):
        raise SemanticHeadroomError("controller wheel bias exceeds the original envelope")
    hard_limits = tuple(servo_limits_deg(name) for name in SERVO_ORDER)
    safe_limits = tuple((lo + SERVO_RESERVE_DEG, hi - SERVO_RESERVE_DEG) for lo, hi in hard_limits)
    if any(not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi for lo, hi in safe_limits):
        raise SemanticHeadroomError("invalid frozen servo limits")
    baseline = tuple(_add_offset(m, c) for m, c in zip(native, controller, strict=True))
    intervals = tuple((min(0.0, lo-base), max(0.0, hi-base))
                      for base, (lo,hi) in zip(baseline[:8], safe_limits, strict=True))
    if any(not math.isfinite(lo) or not math.isfinite(hi) for lo, hi in intervals):
        raise SemanticHeadroomError("headroom interval overflowed")
    effective = []
    for request, (lo,hi) in zip(requested[:8], intervals, strict=True):
        # Preserve signed zero and tiny residuals without subtracting two nearly
        # equal absolute targets to recover the effective residual.
        effective.append(request if request == 0.0 else max(lo, min(hi, request)))
    effective.extend(requested[8:])
    combined = tuple(_add_offset(c, r) if i < 8 else c + r
                     for i, (c, r) in enumerate(zip(controller, effective, strict=True)))
    if any(not math.isfinite(value) for value in combined):
        raise SemanticHeadroomError("composition overflowed to a non-finite value")
    # This is only an arithmetic diagnostic. The real adapter still owns final
    # hard clamping, the one existing slew history, dtype cast, and dispatch.
    # Wheels retain the original arithmetic, including IEEE signed-zero sums.
    candidate = tuple(_add_offset(m, bias) if i < 8 else m + bias
                      for i, (m, bias) in enumerate(zip(native, combined, strict=True)))
    if any(not math.isfinite(value) for value in candidate):
        raise SemanticHeadroomError("composition overflowed to a non-finite value")
    return {
        "mode": HEADROOM_MODE,
        "schema": "wlr50_clean.semantic_servo_headroom.v1",
        "canonical_order": list(FULL12_ORDER),
        "servo_reserve_deg": SERVO_RESERVE_DEG,
        "servo_hard_limits_deg": [list(pair) for pair in hard_limits],
        "servo_safety_limits_deg": [list(pair) for pair in safe_limits],
        "geometry_corrected_native_full12": list(native),
        "bounded_controller_bias_full12": list(controller),
        "baseline_native_plus_controller_full12": list(baseline),
        "requested_policy_residual_full12": list(requested),
        "effective_policy_residual_full12": list(effective),
        "effective_combined_post_mapper_bias_full12": list(combined),
        "policy_residual_intervals_servo_deg": [list(pair) for pair in intervals],
        "clipped_servo_indices": [i for i in range(8) if effective[i] != requested[i]],
        "baseline_outside_reserved_servo_indices": [i for i,(base,(lo,hi)) in
            enumerate(zip(baseline[:8],safe_limits,strict=True)) if not lo <= base <= hi],
        "candidate_native_target_before_final_slew_full12": list(candidate),
        "candidate_target_semantics": "native_plus_effective_combined_bias_no_final_clamp_slew_or_dtype_cast",
        "wheel_residual_unchanged": True,
        "same_tick_provenance": "caller_responsibility_not_validated_by_this_pure_function",
    }
