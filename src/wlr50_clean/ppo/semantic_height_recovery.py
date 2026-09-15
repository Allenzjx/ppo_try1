"""Bounded source-owner hip candidates; no physics/state writes or task credit."""
from __future__ import annotations

import math
from collections.abc import Mapping

MODE = "source_segment_reduction_and_live_recovery_v1"
OWNERS = {"P07": (0, "front_left_hip"), "P08": (4, "rear_left_hip")}


def validate_height_candidate(value):
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {
            "mode", "candidate_id", "preparation_reduction_deg", "post_lift_recovery_deg",
            "recovery_rate_deg_s"} or value["mode"] != MODE:
        raise ValueError("invalid versioned height candidate")
    if not isinstance(value["candidate_id"], str) or not value["candidate_id"].strip():
        raise ValueError("height candidate requires an explicit identity")
    result = dict(value)
    for key in ("preparation_reduction_deg", "post_lift_recovery_deg"):
        row = value[key]
        if not isinstance(row, Mapping) or set(row) != {v[1] for v in OWNERS.values()}:
            raise ValueError("height candidate requires separate FL/RL hip amounts")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
               not math.isfinite(v) or not 0. <= v <= 10. for v in row.values()):
            raise ValueError("height candidate is a bounded 0..10 degree source reduction")
        result[key] = dict(row)
    rate = value["recovery_rate_deg_s"]
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or not 0 < rate <= 60:
        raise ValueError("height recovery rate must be positive and within existing residual slew")
    return result


def source_owner_height_target(*, source_value, source_entry, reduction_deg, recovery_deg):
    """Subtract once from the fixed source excursion, not the previous target.

    An originally nonpositive excursion is untouched. The original motion
    executor still owns its clock/atomic groups and the final mapper its slew.
    """
    values = (source_value, source_entry, reduction_deg, recovery_deg)
    if any(not math.isfinite(float(v)) for v in values) or min(reduction_deg, recovery_deg) < 0:
        raise ValueError("invalid height target operands")
    excursion = max(0., source_value-source_entry)
    return source_value-min(excursion, reduction_deg+recovery_deg)


def current_rr_recovery_permission(task, *, support_spec):
    """A live attained carry can permit recovery; no static/elapsed-time gate."""
    ev = task.get("physical_evaluator", {})
    legs = ev.get("current_legs", {})
    rr = legs.get("RR", {})
    def bearing(row):
        force = row.get("bearing_force_n")
        return (row.get("support") is True and row.get("bearing_verified") is True
            and isinstance(force, (int, float)) and math.isfinite(force)
            and force >= support_spec["force_noise_floor_n"]
            and row.get("air") is False
            and (row.get("ground_contact") is True or row.get("top_surface_contact") is True))
    return bool(task.get("stage_id") in ("P08", "P09") and ev.get("valid") is True
        and task.get("termination_reason") is None and ev.get("termination_reason") is None
        and rr.get("current_lift_valid") is True and rr.get("motion_continuation_allowed") is True
        and rr.get("ground_contact") is False and rr.get("within_lateral_span") is True
        and ev.get("history", {}).get("active_lift", {}).get("RR") is True
        and sum(bearing(row) for leg, row in legs.items() if leg != "RR")
            >= support_spec["minimum_other_supports"])
