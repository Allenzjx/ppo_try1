"""Report-only objective candidate; standard library, no runtime integration.

After all four hard placement-history flags, blend the existing applied-drive
derivative cost with actual canonical wheel-command effort. No phase/region,
nominal, policy residual, measured body speed, or success flag is an input.
"""
from __future__ import annotations

import math
from numbers import Real
from typing import Mapping, Sequence

LEG_KEYS = frozenset(("FR", "FL", "RR", "RL"))
ORIGINAL_WHEEL_HARD_LIMIT = 2.0943951023931953
BLEND_WEIGHT = 0.5
FAMILY_WEIGHT = 0.1


def _finite(value, label):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be a finite non-boolean number")
    return float(value)


def _twelve(values: Sequence[float], label):
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{label} must be a twelve-vector")
    try:
        values=tuple(values)
    except TypeError as exc:
        raise ValueError(f"{label} must be iterable") from exc
    if len(values) != 12:
        raise ValueError(f"{label} must contain twelve values")
    return tuple(_finite(value,label) for value in values)


def _square_cost(values, scales):
    return min(1.0,sum(min(1.0,(x/s)**2) for x,s in zip(values,scales,strict=True))/max(1,len(values)))


def physical_wheel_stop_effort(*, actual_drive_full12: Sequence[float],
        previous_actual_drive_full12: Sequence[float],
        previous_previous_actual_drive_full12: Sequence[float],
        dt_s: float, placed_history: Mapping[str,bool]) -> dict:
    actual=_twelve(actual_drive_full12,"actual drive")
    previous=_twelve(previous_actual_drive_full12,"previous actual drive")
    older=_twelve(previous_previous_actual_drive_full12,"older actual drive")
    dt=_finite(dt_s,"dt_s")
    if not 0. < dt <= 1./120.+1e-9:
        raise ValueError("sample must be an actual positive 120Hz interval")
    if (not isinstance(placed_history,Mapping) or set(placed_history) != LEG_KEYS
            or any(type(value) is not bool for value in placed_history.values())):
        raise ValueError("exact four boolean hard placed-history flags required")
    if any(abs(value) > ORIGINAL_WHEEL_HARD_LIMIT+1e-12 for value in actual[8:]):
        raise ValueError("actual canonical wheel command exceeds the original hard limit")
    rates=(60.,)*8+(1.8,)*4
    first=tuple((a-b)/dt for a,b in zip(actual,previous,strict=True))
    second=tuple((a-2*b+c)/(dt*dt) for a,b,c in zip(actual,previous,older,strict=True))
    old_smooth=(_square_cost(first,rates)+_square_cost(second,tuple(x*120. for x in rates)))/2.
    effort=sum(min(1.,abs(value)/ORIGINAL_WHEEL_HARD_LIMIT) for value in actual[8:])/4.
    enabled=all(placed_history.values())
    cost=(1.-BLEND_WEIGHT)*old_smooth+BLEND_WEIGHT*effort if enabled else old_smooth
    # This is the existing family ceiling, not an additional family or weight.
    cost=min(1.,max(0.,cost))
    return {"all_four_placed_history":enabled,"old_applied_derivative_cost":old_smooth,
        "actual_wheel_effort_cost":effort,"candidate_control_smoothness_cost":cost,
        "old_control_smoothness_reward":-FAMILY_WEIGHT*old_smooth*dt,
        "candidate_control_smoothness_reward":-FAMILY_WEIGHT*cost*dt,
        "reward_delta":-FAMILY_WEIGHT*(cost-old_smooth)*dt,
        "dt_s":dt,"wheel_order":["FL","FR","RL","RR"]}
