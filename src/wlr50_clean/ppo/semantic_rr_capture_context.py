"""Measured rear task facts; a reachable candidate is never a measured support.

This is a bounded workspace proxy, not inverse-kinematics or dynamic feasibility.
It does not write simulator state, completion history, phase IDs or reward.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

from wlr50_clean.infrastructure.command_batch import SERVO_ORDER, servo_limits_deg

RR_CONTINUATION_MODE = "rr_capture_then_rl_transfer_v1"


def _get(obj, key):
    return obj[key] if isinstance(obj, Mapping) else getattr(obj, key)


def verified_current_support(row, force_noise_floor_n):
    force = row.get("bearing_force_n")
    return bool(row.get("support") is True and row.get("bearing_verified") is True
        and isinstance(force, (int,float)) and not isinstance(force,bool) and math.isfinite(force)
        and force >= force_noise_floor_n and row.get("air") is False
        and (row.get("ground_contact") is True or row.get("top_surface_contact") is True))


def rr_capture_transfer_context(*, task, observation, support_spec, assist_snapshot=None, wheel_mode="off"):
    if wheel_mode not in ("off", "rr_carry_forward_v1"):
        raise ValueError("unknown RR carry wheel intervention")
    ev = task["physical_evaluator"]
    legs, history = ev["current_legs"], ev["history"]
    rr = legs["RR"]
    valid = bool(ev["valid"] and ev.get("termination_reason") is None
                 and task.get("termination_reason") is None)
    floor = float(support_spec["force_noise_floor_n"])
    if not math.isfinite(floor) or floor < 0.:
        raise ValueError("RR context requires the existing finite support noise floor")

    def bearing(leg):
        row = legs[leg]
        return verified_current_support(row, floor)

    top_contact = bool(rr.get("top_contact") is True and rr.get("top_surface_contact") is True
        and rr.get("ground_contact") is False and rr.get("contact_surface") == "TOP"
        and rr.get("obstacle_pair_active") is True and rr.get("within_top_xy") is True)
    current_bearing = bool(top_contact and bearing("RR") and rr.get("ground_contact") is False
        and rr.get("contact_surface") == "TOP" and rr.get("obstacle_pair_active") is True
        and rr.get("within_top_xy") is True)
    other_supports = tuple(leg for leg in ("FR", "FL", "RL") if bearing(leg))
    qualified = bool(rr.get("current_lift_valid") is True
                     and history["active_lift"]["RR"] and not rr["ground_contact"])
    joints = _get(observation, "joints")
    margins = []
    for name in SERVO_ORDER[6:8]:
        actual = float(_get(joints[name], "position_deg"))
        lo, hi = servo_limits_deg(name)
        if not math.isfinite(actual):
            raise ValueError("RR workspace needs finite actual joint positions")
        margins.append((actual-lo, hi-actual))
    gap = float(rr["clearance_m"])
    if not math.isfinite(gap):
        raise ValueError("RR workspace needs finite measured gap")
    # One available direction is a candidate, not proof that hip-negative works.
    # Actual descent, tracking and gap trend are checked by the local actuator layer.
    available = any(max(pair) > 2. for pair in margins)
    air_candidate = bool(qualified and rr.get("air") is True and gap >= 0.)
    reachable = bool(valid and (air_candidate or current_bearing) and rr["within_top_xy"] and rr["within_lateral_span"]
                     and available and len(other_supports) >= 2)
    rl_role = ev.get("transfer_roles", {}).get("RL", {})
    # Alternative real support arrangements remain admissible; preferred FL+RR
    # bridges are logged by TransferRoleTracker, never invented from this flag.
    rl_ready = bool(valid and current_bearing and len(other_supports) >= 1
                    and rl_role.get("preparation_ready") is True)
    snapshot = assist_snapshot or {}
    if snapshot:
        # Local import keeps the sensor helper usable by the actuator module.
        from .semantic_rr_capture_assist import validate_rr_capture_assist_snapshot
        validate_rr_capture_assist_snapshot(snapshot)
    live_descent = bool(snapshot.get("mode_name") == "DESCEND"
                        and not snapshot.get("retired"))
    carry = bool(valid and qualified and not current_bearing and not history["placed"]["RL"])
    guide = bool(wheel_mode == "rr_carry_forward_v1" and carry and bearing("FL"))
    return {
        "rr_lift_carry": carry,
        "rr_top_reachable": reachable,
        "rr_top_contact": bool(valid and top_contact),
        "rr_current_bearing": bool(valid and current_bearing),
        "rl_transfer_ready": rl_ready,
        "rr_capture_recovery_allowed": bool(reachable and live_descent),
        "fl_wheel_guidance_active": guide,
        "mode": RR_CONTINUATION_MODE,
        "workspace_semantics": "current_XY_joint_margin_and_measured_other_support_candidate_not_feasibility_proof",
        "other_measured_supports": list(other_supports),
        "rr_actual_joint_margins_deg": margins,
        "rr_gap_m": gap,
        "rr_history_placed": bool(history["placed"]["RR"]),
        "air_is_support": False,
    }
