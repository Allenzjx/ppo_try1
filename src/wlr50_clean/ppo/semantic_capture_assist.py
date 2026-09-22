"""Opt-in, observable FL capture actuator transform; never a policy sample.

The cumulative search target is separate from nominal/mapper history.  Only
FL hip/knee can be owned here; the existing final clamp, slew and single
physical dispatch remain authoritative.  No force or physical state is set.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from wlr50_clean.infrastructure.command_batch import SERVO_ORDER, servo_limits_deg

CAPTURE_ASSIST_MODE = "p05_hip_only_continuation_v1"
CAPTURE_ASSIST_FEATURE_NAMES = (
    "mode", "initialized", "knee_hold_deg", "hip_entry_deg", "hip_target_deg",
    "best_gap_m", "window_start_gap_m", "window_elapsed_s", "hold_elapsed_s",
    "release_fraction", "blocked_reason", "contact_seen",
)
_SCALES = (5., 1., 180., 180., 180., .1, .1, 2., 1., 1., 7., 1.)
_MODES = ("WAIT", "DESCEND", "HOLD", "BLOCKED", "RELEASE", "RELEASED")
_REASONS = ("none", "physical_invalid", "capture_XY_unavailable", "other_support_unavailable",
            "gap_not_improving", "finite_hip_travel_or_margin", "waiting_actual_tracking", "source_unfold_pending")


def _finite(value, label):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"capture assist {label} must be finite")
    return value


def capture_assist_features(snapshot: Mapping) -> tuple[float, ...]:
    if not isinstance(snapshot, Mapping) or snapshot.get("schema") != "wlr50_clean.capture_assist_state.v1":
        raise ValueError("capture assist observation requires its explicit state snapshot")
    return tuple(_finite(snapshot[key], key) / scale
                 for key, scale in zip(CAPTURE_ASSIST_FEATURE_NAMES, _SCALES, strict=True))


def apply_capture_assist_snapshot(candidate_full12: Sequence[float], snapshot: Mapping) -> tuple[float, ...]:
    """Pure reconstruction for real dispatch AND zero-current-policy audit.

    State was updated from pre-dispatch physical feedback, not from the raw
    policy request.  Replaying this transform never advances recovery state.
    """
    candidate = tuple(_finite(x, "candidate") for x in candidate_full12)
    if len(candidate) != 12:
        raise ValueError("capture assist candidate must contain twelve values")
    capture_assist_features(snapshot)
    mode = int(snapshot["mode"])
    if mode not in range(len(_MODES)) or mode != snapshot["mode"]:
        raise ValueError("capture assist mode is invalid")
    if mode in (0, 5):
        return candidate
    fraction = _finite(snapshot["release_fraction"], "release fraction") if mode == 4 else 0.
    if not 0. <= fraction <= 1.:
        raise ValueError("capture assist release fraction must be within [0,1]")
    targets = (snapshot["hip_target_deg"], snapshot["knee_hold_deg"])
    return tuple((1.-fraction)*float(targets[i])+fraction*value if i < 2 else value
                 for i, value in enumerate(candidate))


class HipOnlyCaptureAssist:
    """Finite feedback descent with contact hold and continuous P07 release.

    Engineering bounds: 20 degrees total negative hip travel, original two
    degree reserve, 4/2/1 degree/s far/middle/last-three-mm descent, maximum
    three-degree actual tracking lag, two-second/0.2-mm progress window.
    A blocked search retains its target while shared wheel/body actions can
    improve geometry; it never loops a nominal-relative decrement.
    """
    def __init__(self):
        self.state = dict.fromkeys(CAPTURE_ASSIST_FEATURE_NAMES, 0.)
        self.last_tick = None  # adjacency audit only, not an action input

    def snapshot(self) -> dict:
        state = dict(self.state)
        mode = int(state["mode"])
        return {"schema": "wlr50_clean.capture_assist_state.v1", "version": CAPTURE_ASSIST_MODE,
                **state, "mode_name": _MODES[mode],
                "reason": _REASONS[int(state["blocked_reason"])],
                "active": mode in (1, 2, 3, 4),
                "owners": ["capture_assist" if mode in (1, 2, 3) else
                           "capture_assist_policy_blend" if mode == 4 else "nominal_plus_policy"] * 2,
                "last_dispatch_physics_tick": self.last_tick,
                "feature_names": list(CAPTURE_ASSIST_FEATURE_NAMES)}

    def advance(self, *, context: Mapping, previous_final_full12: Sequence[float], physics_dt_s: float) -> dict:
        tick = context["dispatch_physics_tick"]
        if type(tick) is not int or (self.last_tick is not None and tick != self.last_tick+1):
            raise ValueError("capture assist requires adjacent real dispatch ticks")
        self.last_tick = tick
        dt = _finite(physics_dt_s, "physics dt")
        if not math.isclose(dt, 1./120., abs_tol=1e-12):
            raise ValueError("capture assist retains the 120 Hz physical clock")
        state = self.state
        stage = str(context["stage_id"])
        window = stage in ("P05", "P06")
        safe = bool(context["physical_valid"])
        xy = bool(context.get("within_top_xy"))
        support = int(context.get("other_support_count", 0)) >= 2
        contact = bool(context.get("top_surface_contact") or context.get("obstacle_pair_active"))
        placed = bool(context.get("placed_FL"))
        gap = _finite(context.get("gap_m", 0.), "gap")
        before = self.snapshot()
        if state["mode"] == 0:
            ready = bool(window and safe and xy and support and not placed and not contact
                         and context.get("qualified_FL") and context.get("crossed_FL")
                         and context.get("source_unfold_dispatched"))
            if not ready:
                return {"state_before": before, "state_after": self.snapshot(), "context": dict(context)}
            previous = tuple(_finite(x, "previous final") for x in previous_final_full12)
            if len(previous) != 12:
                raise ValueError("capture assist must anchor the actual previous Full12 final target")
            state.update(mode=1., initialized=1., knee_hold_deg=previous[1], hip_entry_deg=previous[0],
                         hip_target_deg=previous[0], best_gap_m=gap, window_start_gap_m=gap,
                         window_elapsed_s=0., blocked_reason=0.)
            # This dispatch only takes ownership. No new step before observing
            # the anchored target's actual response on the following tick.
            return {"state_before": before, "state_after": self.snapshot(), "context": dict(context)}
        if state["mode"] == 5:
            return {"state_before": before, "state_after": self.snapshot(), "context": dict(context)}
        if not window and state["mode"] != 4:
            state.update(mode=4., blocked_reason=0.)
        if state["mode"] == 4:
            state["release_fraction"] = min(1., state["release_fraction"]+dt/.75)
            if state["release_fraction"] >= 1.:
                state["mode"] = 5.
        elif contact:
            # Any real obstacle reaction stops accumulation, even before the
            # evaluator confirms bearing/top placement; no force threshold is
            # invented and no amount of target error authorizes more pressing.
            state.update(mode=2., contact_seen=1., blocked_reason=0.)
            state["hold_elapsed_s"] += dt
            state["window_elapsed_s"] = 0.
            state["window_start_gap_m"] = gap
        else:
            state["best_gap_m"] = min(state["best_gap_m"], gap)
            improved = state["window_start_gap_m"]-gap >= .0002
            if improved:
                state.update(window_start_gap_m=gap, window_elapsed_s=0.)
            actual = _finite(context.get("hip_actual_deg", state["hip_target_deg"]), "actual hip")
            lo, _ = servo_limits_deg(SERVO_ORDER[0])
            lower = max(lo+2., state["hip_entry_deg"]-20.)
            reason = (1 if not safe else 2 if not xy else 3 if not support else
                      5 if state["hip_target_deg"] <= lower+1e-9 else
                      6 if abs(actual-state["hip_target_deg"]) > 3. else
                      4 if state["window_elapsed_s"] >= 2. and not improved else 0)
            state.update(mode=3. if reason else 1., blocked_reason=float(reason))
            if reason == 0:
                rate = 4. if gap > .015 else 2. if gap > .003 else 1.
                state["hip_target_deg"] = max(lower, state["hip_target_deg"]-rate*dt)
                state["window_elapsed_s"] += dt
            # Waiting for initial actual tracking is not descent exposure.
            # Do not expire the progress window before any bounded step could
            # be tried, nor integrate a request while the servo catches up.
        return {"state_before": before, "state_after": self.snapshot(), "context": dict(context)}


def capture_assist_context(*, task: Mapping, observation, source_frame, previous_ack: Mapping,
                           nominal_provider, physics_tick: int) -> dict:
    def member(obj, name, default=None):
        return obj.get(name, default) if isinstance(obj, Mapping) else getattr(obj, name, default)
    ev = task.get("physical_evaluator", {})
    current = ev.get("current_legs", {})
    fl = current.get("FL", {})
    history = ev.get("history", {})
    source_complete = bool(source_frame.state_id == "P05" and source_frame.endpoint_issued)
    for layer in getattr(nominal_provider, "_continuous_layers", ()):
        if layer.get("stage") == "P05" and member(layer.get("sample"), "endpoint_issued", False):
            source_complete = True
    # Verify the source endpoint request had already reached the real dispatch
    # path. Merely evaluating a new endpoint is not enough to freeze its knee.
    previous_request = previous_ack.get("requested_full12", ())
    source_issued = bool(source_complete and len(previous_request) == 12
                         and tuple(previous_request[:2]) == tuple(source_frame.full12[:2]))
    joint = member(observation, "joints", {}).get(SERVO_ORDER[0])
    return {"dispatch_physics_tick": physics_tick, "source_observation_tick": ev.get("physics_tick"),
            "stage_id": source_frame.state_id, "physical_valid": bool(ev.get("valid") is True
                and ev.get("termination_reason") is None and task.get("termination_reason") is None),
            "within_top_xy": bool(fl.get("within_top_xy")),
            "other_support_count": sum(bool(value.get("support") and value.get("bearing_verified"))
                                       for leg, value in current.items() if leg != "FL"),
            "qualified_FL": bool(history.get("active_lift", {}).get("FL")),
            "crossed_FL": bool(history.get("front_edge_crossed", {}).get("FL")),
            "placed_FL": bool(history.get("placed", {}).get("FL")),
            "top_surface_contact": bool(fl.get("top_surface_contact")),
            "obstacle_pair_active": bool(fl.get("obstacle_pair_active")),
            "gap_m": fl.get("clearance_m", 0.), "hip_actual_deg": member(joint, "position_deg", 0.),
            "source_unfold_dispatched": source_issued,
            "source_endpoint_issued": source_complete}
