"""Unwired variant 02: retain all old derivative cost, use remaining capacity.

This is an objective change, not PBRS and not an additional success predicate.
"""
from __future__ import annotations

from typing import Mapping, Sequence
from wheel_stop_effort_candidate import (physical_wheel_stop_effort, _finite,
    BLEND_WEIGHT, FAMILY_WEIGHT)


def remaining_capacity_cost(*, old_smoothness: float, wheel_effort: float,
                            all_placed_history: bool) -> float:
    smooth=_finite(old_smoothness,"old smoothness")
    effort=_finite(wheel_effort,"wheel effort")
    if not 0. <= smooth <= 1. or not 0. <= effort <= 1. or type(all_placed_history) is not bool:
        raise ValueError("bounded costs and a boolean history gate are required")
    if not all_placed_history:
        return smooth
    return smooth+BLEND_WEIGHT*(1.-smooth)*effort


def physical_wheel_stop_effort_v2(*, actual_drive_full12: Sequence[float],
        previous_actual_drive_full12: Sequence[float],
        previous_previous_actual_drive_full12: Sequence[float],
        dt_s: float, placed_history: Mapping[str,bool]) -> dict:
    result=physical_wheel_stop_effort(actual_drive_full12=actual_drive_full12,
        previous_actual_drive_full12=previous_actual_drive_full12,
        previous_previous_actual_drive_full12=previous_previous_actual_drive_full12,
        dt_s=dt_s,placed_history=placed_history)
    old=result["old_applied_derivative_cost"]
    cost=remaining_capacity_cost(old_smoothness=old,wheel_effort=result["actual_wheel_effort_cost"],
        all_placed_history=result["all_four_placed_history"])
    result.update(candidate_variant="retain_old_smoothness_remaining_capacity_v2",
        candidate_control_smoothness_cost=cost,
        candidate_control_smoothness_reward=-FAMILY_WEIGHT*cost*result["dt_s"],
        reward_delta=-FAMILY_WEIGHT*(cost-old)*result["dt_s"])
    return result
