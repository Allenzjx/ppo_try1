"""Measured rear task dependencies, never actuator targets or placement events.

The late P09 FL/RL source group is a support transfer, not RR joint descent.
It waits for actual RR support. Pending RR carry and policy remain independent.
"""
from __future__ import annotations

import math
from typing import Mapping, Any

MODE = "rr_capture_before_rl_transfer_v1"
RECAPTURE_MODE = "rr_recapture_current_support_v2"
LIVE_SWING_MODE = "rr_live_swing_evidence_v3"
EDGE_RECOVERY_MODE = "rr_rl_edge_recovery_v4"
LIVE_SWING_MODES = (LIVE_SWING_MODE, EDGE_RECOVERY_MODE)
RECAPTURE_MODES = (RECAPTURE_MODE, *LIVE_SWING_MODES)
MODES = (MODE, *RECAPTURE_MODES)


def verified_bearing(row: Mapping[str, Any], support: Mapping[str, Any]) -> bool:
    force = row.get("bearing_force_n")
    return bool(type(force) in (int, float) and math.isfinite(force)
        and force >= support["force_noise_floor_n"]
        and row.get("support") is True and row.get("bearing_verified") is True
        and row.get("air") is False
        and (row.get("ground_contact") is True or row.get("top_surface_contact") is True))


def rear_dependency(task: Mapping[str, Any], support: Mapping[str, Any], *,
                    mode: str | None = MODE) -> dict[str, bool]:
    """Separate load-dependent transfer from current RL recovery permission.

    EDGE permission does not authorize the P12 strong-unload source lane or
    assert a support force. It exposes an already-qualified attempt's legal
    recovery path; final actuator limits and physical safety remain elsewhere.
    Legacy modes retain their AIR-only continuation semantics.
    """
    if mode not in (None, *MODES):
        raise ValueError("unknown rear policy timing mode")
    ev = task.get("physical_evaluator", {})
    legs = ev.get("current_legs", {})
    rr, rl = legs.get("RR", {}), legs.get("RL", {})
    live = bool(ev.get("valid") is True and task.get("termination_reason") is None
        and ev.get("termination_reason") is None)
    top = bool(live and rr.get("ground_contact") is False
        and rr.get("air") is False and rr.get("top_contact") is True
        and rr.get("top_surface_contact") is True
        and rr.get("obstacle_pair_active") is True and rr.get("within_top_xy") is True
        and rr.get("contact_surface") == "TOP")
    bearing = top and verified_bearing(rr, support)
    bridge = any(verified_bearing(legs.get(leg, {}), support) for leg in ("FR", "FL"))
    # A live RL swing must retain its lowering/carry path if RR drops load.
    # Old placed/active_lift history cannot grant this after RL returns to ground.
    rl_continuing = bool(live and rl.get("current_lift_valid") is True
        and rl.get("motion_continuation_allowed") is True and rl.get("air") is True
        and rl.get("ground_contact") is False)
    # Consume the real evaluator's *current*, ground-revoked same-attempt
    # evidence, never history.active_lift/placed. An ambiguous exact obstacle
    # pair can permit disengagement without becoming verified bearing or TOP.
    qualified_tick = rl.get("current_lift_qualified_tick")
    tick = ev.get("physics_tick")
    reaction = rl.get("contact_reaction_force_n")
    edge_recovery = bool(mode == EDGE_RECOVERY_MODE and live
        and rl.get("current_lift_valid") is True
        and rl.get("motion_continuation_allowed") is True
        and rl.get("active_attempt") is True
        and type(tick) is int and type(qualified_tick) is int
        and 0 <= qualified_tick <= tick
        and rl.get("air") is False and rl.get("ground_contact") is False
        and rl.get("obstacle_pair_active") is True
        and rl.get("contact_reaction") is True
        and type(reaction) in (int, float) and math.isfinite(reaction) and reaction >= 0.
        and rl.get("top_contact") is False and rl.get("top_surface_contact") is False
        and rl.get("contact_mode") in ("FRONT_WALL", "OBSTACLE_AMBIGUOUS")
        and rl.get("contact_surface") == rl.get("contact_mode"))
    return dict(rr_top_contact=top, rr_current_bearing=bool(bearing),
        support_transfer_permitted=bool(live and bearing and bridge),
        rl_current_swing=rl_continuing,
        rl_edge_recovery_permitted=edge_recovery,
        rl_motion_continuation_permitted=bool(rl_continuing or edge_recovery))


def public_timing(task, layers, support, physics_hz, *, mode=MODE):
    if mode not in (None, *MODES):
        raise ValueError("unknown rear policy timing mode")
    dep = rear_dependency(task, support, mode=mode)
    by_phase = {layer["stage"]: layer for layer in layers}
    p09, p12 = by_phase.get("P09", {}), by_phase.get("P12", {})
    late_started = "late_group_start_tick" in p09.get("sequence_diagnostic", {})
    rear = task.get("stage_id") in ("P07", "P08", "P09", "P10", "P11", "P12")
    ev = task.get("physical_evaluator", {})
    rr = ev.get("current_legs", {}).get("RR", {})
    # The existing public task flag means swing/capture *work*, not AIR or
    # bearing. V4 includes same-attempt edge recovery; the separate dependency
    # flags retain the exact AIR/EDGE distinction for source-owner consumers.
    swing = rear and dep["rl_motion_continuation_permitted"]
    placed = ev.get("history", {}).get("placed", {})
    live = bool(ev.get("valid") is True and ev.get("termination_reason") is None
                and task.get("termination_reason") is None)
    # A task request, not new AIR/lift/support evidence. After first capture,
    # old source clocks and placed history cannot hide current loss of load.
    # Preserve a real ongoing RL swing and the already-placed RL continuation.
    recapture = bool(mode in RECAPTURE_MODES and rear and live and not swing
        and placed.get("RL") is not True and not dep["rr_current_bearing"]
        and (late_started or placed.get("RR") is True
             or task.get("stage_id") in ("P10", "P11", "P12")))
    prep = rear and not swing and not recapture and (
        late_started or task.get("stage_id") in ("P10", "P11", "P12"))
    handoff = rear and not swing and not prep and not recapture and dep["rr_top_contact"]
    carry = recapture or (rear and not swing and not prep and not handoff and bool(
        rr.get("current_lift_valid") or task.get("stage_id") == "P09"))
    clock = lambda layer, key: min(200., float(layer.get(key, 0)) / physics_hz)
    return dict(rr_carry_capture=bool(carry), rr_support_handoff=bool(handoff),
        rl_prep_transfer=bool(prep), rl_swing_capture=bool(swing),
        p09_source_time_s=clock(p09, "ticks"), p12_source_time_s=clock(p12, "ticks"),
        p12_rl_source_time_s=clock(p12, "rl_ticks"),
        p09_dependency_wait=p09.get("sequence_diagnostic", {}).get("wait_reason")
            == "current_RR_bearing_before_FL_RL_transfer",
        p12_dependency_wait=bool(p12.get("rl_dependency_wait", False)))
