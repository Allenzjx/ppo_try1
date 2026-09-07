"""REVIEW DRAFT ONLY: nominal-only two-DOF, fixed-base linear projection.

Not imported by runtime. Not physically validated. No residual, phase, joint-sign,
historical-pose, wheel-control or motion-clock input exists here.

Coordinates must already be *physical* joint radians: nominal_delta_rad equals
the proposed nominal target minus the same-tick measured actual q. The box is the
caller-computed intersection of physical joint and this-tick nominal target
limits, translated by that same actual q. The proposed nominal must be in it.
Jx/Jz are world derivatives in m/rad at this actual state, for the point selected
by the caller. This helper cannot validate Jacobian indices, mapper provenance,
wheel-bottom offsets, floating-base/contact effects or subsequent actuator slew.

available_descent_m is supplied as max(0, current_clearance - existing_margin).
This helper invents neither a margin nor a success/contact criterion. place_xy
is current physical eligibility, not append-only placement history.

When a correction is necessary, the mathematical priority is:
  1. box AND Jz*dq >= -available_descent;
  2. do not reverse requested Jx direction (for zero request: no backward x);
  3. minimize |Jx*dq - Jx*dq_nominal|;
  4. among those solutions minimize Euclidean joint-radian target change.
An exact original Jx value is preferred whenever feasible. Otherwise its value
is clamped to the feasible interval: a lexicographic relaxation, not a weighted
penalty or an ill-conditioned Jacobian inverse. If every feasible non-reversing
x magnitude exceeds the original, the smallest feasible increase is allowed and
reported: this is NOT a no-amplification guarantee. Empty feasible sets return NO
target. The integration must explicitly handle that failure; this draft does
not quietly send an unsafe original target, zero a leg, or alter a residual.

The finite 2-D polygon and its linear slice are solved by edge interpolation and
segment projection. Degenerate boxes/rows are supported. Float-rounding slack is
reported separately; it is not a new physical tolerance. A successful result
proves only these linearized nominal constraints, NOT actual next-tick motion.

No Python or tests were executed while writing this live-barrier draft.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from typing import Sequence


Pair = tuple[float, float]


class ProjectionNumericalError(ValueError):
    """Finite inputs could not be represented/verified safely in float math."""


@dataclass(frozen=True)
class NominalProjectionResult:
    feasible: bool
    status: str
    nominal_delta_rad: Pair
    corrected_delta_rad: Pair | None
    nominal_correction_rad: Pair | None
    proof: dict[str, object]


def _scalar(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number, not bool/string")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _pair(value: Sequence[float], name: str) -> Pair:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must contain exactly two real values")
    try:
        items = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must contain exactly two real values") from exc
    if len(items) != 2:
        raise ValueError(f"{name} must contain exactly two real values")
    return (_scalar(items[0], name), _scalar(items[1], name))


def _finite(value: float) -> float:
    if not math.isfinite(value):
        raise ProjectionNumericalError("nonfinite intermediate; no target emitted")
    return value


def _dot(row: Pair, point: Pair) -> float:
    try:
        return _finite(math.fsum((row[0] * point[0], row[1] * point[1])))
    except (OverflowError, ValueError) as exc:
        raise ProjectionNumericalError("dot product overflow; no target emitted") from exc


def _unit(row: Pair) -> tuple[Pair, float]:
    norm = _finite(math.hypot(*row))
    return (((row[0] / norm, row[1] / norm) if norm else (0.0, 0.0)), norm)


def _deduplicate(points: list[Pair]) -> list[Pair]:
    # Exact equality only: do not erase a genuinely narrow feasible polygon.
    result: list[Pair] = []
    for point in points:
        if not result or point != result[-1]:
            result.append(point)
    if len(result) > 1 and result[0] == result[-1]:
        result.pop()
    return result


def _interpolate(a: Pair, b: Pair, t: float) -> Pair:
    # A convex combination avoids both extrapolation and tiny-determinant solves.
    t = min(1.0, max(0.0, t))
    return (
        _finite((1.0 - t) * a[0] + t * b[0]),
        _finite((1.0 - t) * a[1] + t * b[1]),
    )


def _crossing(a: Pair, b: Pair, fa: float, fb: float) -> Pair:
    # Called only when signed distances straddle the line. Scaling avoids an
    # overflow in fa-fb for large, opposite-signed finite distances.
    scale = max(abs(fa), abs(fb))
    if not scale:
        return a
    ua, ub = fa / scale, fb / scale
    denominator = ua - ub
    if denominator == 0.0:
        raise ProjectionNumericalError("unresolved line crossing; no target emitted")
    return _interpolate(a, b, ua / denominator)


def _clip(points: list[Pair], row: Pair, bound: float) -> list[Pair]:
    """Intersect a convex polygon/segment/point with row*p >= bound."""
    if not points:
        return []
    if len(points) == 1:
        return points if _dot(row, points[0]) >= bound else []
    output: list[Pair] = []
    previous = points[-1]
    fp = _finite(_dot(row, previous) - bound)
    for current in points:
        fc = _finite(_dot(row, current) - bound)
        if (fp >= 0.0) != (fc >= 0.0):
            output.append(_crossing(previous, current, fp, fc))
        if fc >= 0.0:
            output.append(current)
        previous, fp = current, fc
    return _deduplicate(output)


def _slice(points: list[Pair], row: Pair, level: float, eps: float) -> list[Pair]:
    """Intersect with row*p == level; returned points lie on a convex segment."""
    output: list[Pair] = []
    previous = points[-1]
    fp = _finite(_dot(row, previous) - level)
    for current in points:
        fc = _finite(_dot(row, current) - level)
        if abs(fc) <= eps:
            output.append(current)
        if (fp < 0.0 < fc) or (fc < 0.0 < fp):
            output.append(_crossing(previous, current, fp, fc))
        previous, fp = current, fc
    return _deduplicate(output)


def _segment_projection(nominal: Pair, a: Pair, b: Pair) -> Pair:
    delta = (_finite(b[0] - a[0]), _finite(b[1] - a[1]))
    unit, length = _unit(delta)
    if length == 0.0:
        return a
    offset = (_finite(nominal[0] - a[0]), _finite(nominal[1] - a[1]))
    along = _dot(unit, offset)
    if along <= 0.0:
        return a
    if along >= length:
        return b
    return _interpolate(a, b, along / length)


def _nearest(nominal: Pair, points: list[Pair]) -> Pair:
    # For a polygon the caller's nominal is outside it (violated downward row).
    # For a slice the set is a segment, possibly represented by repeated points.
    candidates = list(points)
    for index, point in enumerate(points):
        candidates.append(_segment_projection(nominal, point, points[index - 1]))
    return min(
        candidates,
        key=lambda point: (
            _finite(math.hypot(point[0] - nominal[0], point[1] - nominal[1])),
            point[0], point[1],
        ),
    )


def project_nominal_downward(
    *,
    nominal_delta_rad: Sequence[float],
    jacobian_x_m_per_rad: Sequence[float],
    jacobian_z_m_per_rad: Sequence[float],
    delta_lower_rad: Sequence[float],
    delta_upper_rad: Sequence[float],
    available_descent_m: float,
    place_xy: bool,
) -> NominalProjectionResult:
    """Return an identity/correction, or explicit infeasibility with no target.

    Invalid shapes/nonfinite values/invalid boxes/out-of-box proposals raise
    ValueError even in the place-XY identity branch. Numeric failure raises
    ProjectionNumericalError, never a fabricated feasible target.
    """
    nominal = _pair(nominal_delta_rad, "nominal_delta_rad")
    jx = _pair(jacobian_x_m_per_rad, "jacobian_x_m_per_rad")
    jz = _pair(jacobian_z_m_per_rad, "jacobian_z_m_per_rad")
    lower = _pair(delta_lower_rad, "delta_lower_rad")
    upper = _pair(delta_upper_rad, "delta_upper_rad")
    allowance = _scalar(available_descent_m, "available_descent_m")
    if type(place_xy) is not bool:
        raise ValueError("place_xy must be bool")
    if allowance < 0.0:
        raise ValueError("available_descent_m must be nonnegative")
    if any(lo > hi for lo, hi in zip(lower, upper)):
        raise ValueError("empty/inverted target box")
    if any(not lo <= q <= hi for lo, q, hi in zip(lower, nominal, upper)):
        raise ValueError("nominal must already obey caller's allowed target box")

    x_unit, x_norm = _unit(jx)
    z_unit, z_norm = _unit(jz)
    old_x, old_z = _dot(jx, nominal), _dot(jz, nominal)
    coordinate_scale = max(1.0, *(abs(v) for v in (*nominal, *lower, *upper)))
    eps = _finite(128.0 * math.ulp(coordinate_scale))
    proof: dict[str, object] = {
        "schema": "nominal_projection_math_draft.v1",
        "scope": "nominal_only_fixed_base_first_order_two_dof",
        "physical_motion_guaranteed": False,
        "residual_constraint_applied": False,
        "place_xy": place_xy,
        "downward_constraint_enforced": not place_xy,
        "available_descent_m": allowance,
        "jacobian_x_m_per_rad": jx,
        "jacobian_z_m_per_rad": jz,
        "delta_lower_rad": lower,
        "delta_upper_rad": upper,
        "original_linear_x_m": old_x,
        "original_linear_z_m": old_z,
        "numerical_coordinate_tolerance_rad": eps,
        "input_target_within_box": True,
    }

    def finish(candidate: Pair, status: str, x_level: float | None = None) -> NominalProjectionResult:
        new_x, new_z = _dot(jx, candidate), _dot(jz, candidate)
        box_ok = all(lo - eps <= q <= hi + eps for lo, q, hi in zip(lower, candidate, upper))
        z_tolerance = _finite(eps * z_norm)
        z_ok = new_z >= -allowance - z_tolerance
        x_normalized = _dot(x_unit, candidate)
        old_x_normalized = _dot(x_unit, nominal)
        # Preserve the requested direction, not the arbitrary sign of a joint.
        direction = -1.0 if old_x_normalized < 0.0 else 1.0
        direction_ok = not x_norm or direction * x_normalized >= -eps
        level_ok = x_level is None or abs(x_normalized - x_level) <= 4.0 * eps
        if not (box_ok and (place_xy or z_ok) and direction_ok and level_ok):
            raise ProjectionNumericalError("post-projection check failed; no target emitted")
        correction = (
            _finite(candidate[0] - nominal[0]),
            _finite(candidate[1] - nominal[1]),
        )
        proof.update({
            "mathematical_contract_verified": True,
            "corrected_linear_x_m": new_x,
            "corrected_linear_z_m": new_z,
            "linear_z_slack_m": _finite(new_z + allowance),
            "linear_z_rounding_tolerance_m": z_tolerance,
            "downward_halfspace_satisfied": z_ok,
            "box_satisfied": box_ok,
            "requested_x_direction_not_reversed": direction_ok,
            "forward_error_m": abs(_finite(new_x - old_x)),
            "requested_x_magnitude_increase_m": max(0.0, _finite(abs(new_x) - abs(old_x))),
            "no_amplification_guarantee": False,
            "box_lower_slack_rad": tuple(_finite(q - lo) for q, lo in zip(candidate, lower)),
            "box_upper_slack_rad": tuple(_finite(hi - q) for q, hi in zip(candidate, upper)),
            "joint_correction_norm_rad": _finite(math.hypot(*correction)),
        })
        return NominalProjectionResult(True, status, nominal, candidate, correction, proof)

    if place_xy:
        return finish(nominal, "identity_place_xy")
    if old_z >= -allowance:
        return finish(nominal, "identity_within_descent_allowance")
    # A zero Jz cannot violate a nonnegative allowance in finite arithmetic.
    if z_norm == 0.0:
        raise ProjectionNumericalError("zero vertical row reported a downward violation")

    polygon = _deduplicate([
        (lower[0], lower[1]), (upper[0], lower[1]),
        (upper[0], upper[1]), (lower[0], upper[1]),
    ])
    polygon = _clip(polygon, z_unit, _finite(-allowance / z_norm))
    if not polygon:
        proof.update({"mathematical_contract_verified": False, "conflict": "box_vs_downward_halfspace"})
        return NominalProjectionResult(False, "infeasible_box_downward", nominal, None, None, proof)

    if x_norm == 0.0:
        proof["forward_priority"] = "zero_horizontal_row_euclidean_projection"
        return finish(_nearest(nominal, polygon), "projected_exact_forward", 0.0)

    old_level = _dot(x_unit, nominal)
    direction = -1.0 if old_level < 0.0 else 1.0
    proof["nonreversal_direction"] = "negative_requested_x" if direction < 0.0 else "nonnegative_world_x"
    proof["box_downward_feasible_x_interval_m"] = [
        min(_dot(jx, p) for p in polygon), max(_dot(jx, p) for p in polygon),
    ]
    polygon = _clip(polygon, (direction * x_unit[0], direction * x_unit[1]), 0.0)
    if not polygon:
        proof.update({"mathematical_contract_verified": False, "conflict": "all_box_downward_solutions_reverse_x"})
        return NominalProjectionResult(False, "infeasible_nonreversal", nominal, None, None, proof)

    low = min(_dot(x_unit, point) for point in polygon)
    high = max(_dot(x_unit, point) for point in polygon)
    level = min(high, max(low, old_level))
    exact_forward = low <= old_level <= high
    proof["forward_priority"] = "exact_original_x" if exact_forward else "closest_feasible_x_then_euclidean"
    proof["selected_normalized_x_level_rad"] = level
    section = _slice(polygon, x_unit, level, eps)
    if not section:
        raise ProjectionNumericalError("nonempty polygon slice unresolved; no target emitted")
    return finish(
        _nearest(nominal, section),
        "projected_exact_forward" if exact_forward else "projected_relaxed_forward",
        level,
    )
