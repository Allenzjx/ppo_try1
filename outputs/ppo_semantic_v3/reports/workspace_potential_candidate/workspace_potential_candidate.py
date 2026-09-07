"""UNWIRED soft-potential candidate; never a task/entry/completion predicate.

Reciprocal distance to the existing current workspace interval, using the old
0.25 m decay scale. Extreme finite distances may underflow to zero; an extremely
small positive outside distance may round to one. Neither is a hard decision.
"""
from __future__ import annotations

import math
from numbers import Real


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real, not a boolean/string")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def interval_distance_progress(x, lower, upper, within_lateral_span, scale=.25):
    """Return bounded soft progress; validate even the lateral-outside path.

    This function reads no phase, historical entry, physical state or controller.
    It does not grant qualification, crossing, placement or workspace completion.
    """
    x, lower, upper, scale = (_finite(value, name) for value, name in
        ((x,"x"), (lower,"lower"), (upper,"upper"), (scale,"scale")))
    if type(within_lateral_span) is not bool:
        raise ValueError("within_lateral_span must be an actual boolean")
    if not lower < upper:
        raise ValueError("workspace lower must be less than upper")
    if scale <= 0.0:
        raise ValueError("scale must be positive")
    if not within_lateral_span:
        return 0.0
    if lower <= x <= upper:
        return 1.0
    boundary = lower if x < lower else upper
    distance = abs(boundary-x)
    if math.isfinite(distance):
        # Avoid overflow in scale+distance and division of a large distance by
        # a tiny scale. Only divide a smaller nonnegative number by the larger.
        if distance <= scale:
            return 1.0/(1.0+distance/scale)
        ratio = scale/distance
        return ratio/(1.0+ratio)
    # Finite endpoints can have an unrepresentable opposite-sign difference.
    # Normalize BEFORE subtraction; here the scaled distance is in (1,2].
    normalizer = max(abs(boundary),abs(x))
    scaled_distance = abs(boundary/normalizer-x/normalizer)
    scaled_scale = scale/normalizer
    return scaled_scale/(scaled_scale+scaled_distance)
