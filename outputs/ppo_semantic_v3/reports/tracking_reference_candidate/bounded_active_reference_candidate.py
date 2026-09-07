"""UNWIRED report variant 02: bound the reference's desired correction increment.

The comparator c0 uses CURRENT real measurement and the SAME mapper history.
It is not a separate B rollout, a contact estimator, or a physical guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Mapping, Sequence

from active_reference_candidate import _eight, active_tracking_reference
from wlr50_clean.infrastructure.command_batch import SERVO_COMMAND_SIGN, SERVO_ORDER, servo_limits_deg


@dataclass(frozen=True)
class BoundedReference:
    actual_measured_physical_rad: tuple[float, ...]
    mapper_computational_reference_rad: tuple[float, ...]
    channels: tuple[dict, ...]


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be a finite real non-boolean number")
    return float(value)


def bounded_active_tracking_reference(*, measured_physical_rad: Sequence[float],
        previous_effective_residual_deg: Sequence[float], tracking_servo_names: Sequence[str],
        requested_command_deg: Sequence[float], standing_pose_deg: Mapping[str, float],
        tracking_gain: float, tracking_limit_deg: float) -> BoundedReference:
    ordinary = active_tracking_reference(measured_physical_rad=measured_physical_rad,
        previous_effective_residual_deg=previous_effective_residual_deg,
        tracking_servo_names=tracking_servo_names)
    requested = _eight(requested_command_deg, "requested_command_deg")
    gain, limit = _number(tracking_gain, "tracking_gain"), _number(tracking_limit_deg, "tracking_limit_deg")
    if gain < 0. or limit < 0.:
        raise ValueError("gain and correction limit must be nonnegative")
    if not isinstance(standing_pose_deg, Mapping) or set(standing_pose_deg) != set(SERVO_ORDER):
        raise ValueError("standing pose must contain exactly the canonical eight servos")
    standing = {name:_number(standing_pose_deg[name], name) for name in SERVO_ORDER}
    reference = list(ordinary.mapper_computational_reference_rad)
    records = []
    for i, name in enumerate(SERVO_ORDER):
        hard_lo, hard_hi = servo_limits_deg(name)
        nominal = max(hard_lo, min(hard_hi, requested[i]))
        q = ordinary.actual_measured_physical_rad[i]
        r = ordinary.active_reference_deg[i]
        sign = SERVO_COMMAND_SIGN[name]
        if gain == 0. or r == 0.:
            reference[i] = q
            records.append({"servo":name, "status":"original_no_reference", "clipped":False})
            continue
        # Use exactly the frozen physical-angle subtraction ordering for e0.
        # Algebraically this is n - (degrees(q)-standing)/sign.
        nominal_physical = standing[name] + sign*nominal
        e0 = (nominal_physical-math.degrees(q))/sign
        raw0, raw1 = gain*e0, gain*(e0+r)
        if not all(math.isfinite(v) for v in (nominal_physical,e0,raw0,raw1)):
            raise ValueError("reference correction arithmetic overflow")
        c0, c1 = max(-limit,min(limit,raw0)), max(-limit,min(limit,raw1))
        safe_lo, safe_hi = hard_lo+2., hard_hi-2.
        lower = max(-limit,min(c0,0.,safe_lo-nominal))
        upper = min(limit,max(c0,0.,safe_hi-nominal))
        bounded = max(lower,min(upper,c1))
        clipped = bounded != c1
        if clipped:
            # Invert only the clipped desired feedback error. This is a LOCAL
            # computational reference, not a change to measured q or nominal.
            reference[i] = math.radians(nominal_physical-sign*(bounded/gain))
        if not math.isfinite(reference[i]):
            raise ValueError("reference angle overflow")
        achieved_raw = gain*((nominal_physical-math.degrees(reference[i]))/sign)
        if not math.isfinite(achieved_raw):
            raise ValueError("reference reconstruction overflow")
        achieved = max(-limit,min(limit,achieved_raw))
        records.append({"servo":name, "status":"reference_desired_clipped" if clipped else "original_active_reference",
            "clipped":clipped, "nominal_deg":nominal, "current_actual_canonical_error_deg":e0,
            "previous_effective_deg":r, "desired_original_c0_deg":c0,
            "desired_reference_c1_deg":c1, "allowed_desired_interval_deg":[lower,upper],
            "bounded_desired_c1_deg":bounded, "reconstructed_desired_deg":achieved,
            "reconstruction_error_deg":achieved-bounded if clipped else None,
            "reserved_native_band_deg":[safe_lo,safe_hi]})
    return BoundedReference(ordinary.actual_measured_physical_rad, tuple(reference), tuple(records))
