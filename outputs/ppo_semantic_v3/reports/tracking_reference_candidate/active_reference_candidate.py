"""UNWIRED REPORT CANDIDATE: pure active-only computational reference.

This is not a sensor, a mapper replacement, or a production policy mode.
The caller must supply a verified previous-tick EFFECTIVE residual. This
prototype deliberately does not implement ACK/clock/reset/audit provenance.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Sequence

from wlr50_clean.infrastructure.command_batch import SERVO_COMMAND_SIGN, SERVO_ORDER


@dataclass(frozen=True)
class ActiveReference:
    actual_measured_physical_rad: tuple[float, ...]
    mapper_computational_reference_rad: tuple[float, ...]
    previous_effective_residual_deg: tuple[float, ...]
    active_reference_deg: tuple[float, ...]
    scheduled_servo_names: tuple[str, ...]


def _eight(values: Sequence[float], label: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{label} must be a numeric eight-vector")
    try:
        values = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{label} must be iterable") from exc
    if len(values) != 8 or any(isinstance(x, bool) or not isinstance(x, Real) for x in values):
        raise ValueError(f"{label} must contain eight real, non-boolean numbers")
    result = tuple(float(x) for x in values)
    if not all(math.isfinite(x) for x in result):
        raise ValueError(f"{label} must be finite")
    return result


def active_tracking_reference(*, measured_physical_rad: Sequence[float],
                              previous_effective_residual_deg: Sequence[float],
                              tracking_servo_names: Sequence[str]) -> ActiveReference:
    """Return local reference values; never mutate caller data or mapper state.

    Frozen advance consumes measured values in active feedback and in
    ended-tracking retirement. Ended names are disjoint from current scheduled
    names, so only the former can see the computational adjustment below.
    Standing cancels in the delta; rear canonical command signs are negative.
    """
    actual = _eight(measured_physical_rad, "measured_physical_rad")
    previous = _eight(previous_effective_residual_deg, "previous_effective_residual_deg")
    if isinstance(tracking_servo_names, (str, bytes)):
        raise ValueError("tracking_servo_names must be a sequence of names")
    try:
        scheduled = tuple(tracking_servo_names)
    except TypeError as exc:
        raise ValueError("tracking_servo_names must be iterable") from exc
    if (any(not isinstance(name, str) or name not in SERVO_ORDER for name in scheduled)
            or len(set(scheduled)) != len(scheduled)):
        raise ValueError("tracking names must be unique canonical servo names")
    selected = frozenset(scheduled)
    active, reference = [], []
    for name, q, previous_r in zip(SERVO_ORDER, actual, previous, strict=True):
        r = previous_r if name in selected else 0.0
        # Preserve actual signed zero / bit pattern when no reference applies.
        value = q if r == 0.0 else q - SERVO_COMMAND_SIGN[name] * math.radians(r)
        if not math.isfinite(value):
            raise ValueError("computational reference overflow")
        active.append(r)
        reference.append(value)
    return ActiveReference(actual, tuple(reference), previous, tuple(active), scheduled)
