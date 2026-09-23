"""Opt-in finite RR hip-only capture transform, not a policy or task event.

Only canonical RR hip/knee (6/7) are owned. The existing final hard limits,
slew and unique physical dispatch remain authoritative. Negative hip is an
explicit bounded diagnostic direction, not a claim about Cartesian motion.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from wlr50_clean.infrastructure.command_batch import SERVO_ORDER, servo_limits_deg
from wlr50_clean.ppo.semantic_rr_capture_context import verified_current_support

RR_CAPTURE_ASSIST_MODE = "rr_hip_only_capture_v1"
RR_CAPTURE_ASSIST_SCHEMA = "wlr50_clean.rr_capture_assist_state.v1"
RR_CAPTURE_FEEDBACK_REVISION = "window_peak_progress_v2"
RR_CAPTURE_WINDOW_REFERENCE_SEMANTICS = "public_window_peak_gap_reuses_existing_window_start_gap_scalar_upward_motion_never_resets_elapsed"
RR_CAPTURE_ASSIST_FEATURE_NAMES = (
    "mode", "initialized", "knee_hold_deg", "hip_entry_deg", "hip_target_deg",
    "travel_used_deg", "descent_elapsed_s", "window_start_gap_m", "window_elapsed_s",
    "hold_elapsed_s", "release_fraction", "contact_seen", "retired", "blocked_reason",
)
_SCALES = (5., 1., 180., 180., 180., 20., 12., .1, 2., 1., 1., 1., 1., 9.)
_MODES = ("WAIT", "DESCEND", "HOLD", "BLOCKED", "RELEASE", "RELEASED")
_REASONS = ("none", "physical_invalid", "capture_XY_unavailable", "other_support_unavailable",
            "gap_not_improving", "finite_hip_travel_or_margin", "waiting_actual_tracking",
            "current_AIR_unavailable", "finite_descent_exposure", "qualified_crossing_unavailable")
_BOOL_CONTEXT = ("physical_valid", "within_top_xy", "qualified_RR", "crossed_RR",
                 "air", "ground_contact", "top_surface_contact", "obstacle_pair_active",
                 "current_top_bearing", "rl_qualified_lift")


def _finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"RR capture assist {label} must be a finite number")
    return float(value)


def _vector(value, size, label):
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != size:
        raise ValueError(f"RR capture assist {label} requires {size} values")
    return tuple(_finite(x, label) for x in value)


def _snapshot(state, tick):
    mode = int(state["mode"])
    active = mode in (1, 2, 3, 4)
    owner = "rr_capture_assist_policy_blend" if mode == 4 else "rr_capture_assist" if active else "nominal_plus_policy"
    return {"schema": RR_CAPTURE_ASSIST_SCHEMA, "version": RR_CAPTURE_ASSIST_MODE,
        "feedback_revision": RR_CAPTURE_FEEDBACK_REVISION,
        "window_reference_semantics": RR_CAPTURE_WINDOW_REFERENCE_SEMANTICS,
        **state, "mode_name": _MODES[mode], "reason": _REASONS[int(state["blocked_reason"])],
        "active": active, "owners": [owner, owner], "owner_indices": [6, 7] if active else [],
        "last_dispatch_physics_tick": tick, "feature_names": list(RR_CAPTURE_ASSIST_FEATURE_NAMES)}


def _exhaustion_reason(state):
    if (state["travel_used_deg"] >= 20. - 1e-9
            or state["hip_target_deg"] <= servo_limits_deg(SERVO_ORDER[6])[0] + 2. + 1e-9):
        return 5
    if state["descent_elapsed_s"] >= 12. - 1e-9:
        return 8
    if state["window_elapsed_s"] >= 2. - 1e-9:
        return 4
    return 0


def validate_rr_capture_assist_snapshot(snapshot: Mapping) -> dict:
    """Strict finite state/metadata validation for observation and replay."""
    metadata = {"schema", "version", "feedback_revision", "window_reference_semantics",
                "mode_name", "reason", "active", "owners", "owner_indices",
                "last_dispatch_physics_tick", "feature_names"}
    if not isinstance(snapshot, Mapping) or set(snapshot) != set(RR_CAPTURE_ASSIST_FEATURE_NAMES) | metadata:
        raise ValueError("RR capture assist requires its complete explicit snapshot")
    state = {k: _finite(snapshot[k], k) for k in RR_CAPTURE_ASSIST_FEATURE_NAMES}
    for key, maximum in (("mode", 5), ("blocked_reason", 9)):
        if state[key] != int(state[key]) or not 0 <= state[key] <= maximum:
            raise ValueError(f"invalid RR capture assist {key}")
    for key in ("initialized", "contact_seen", "retired"):
        if state[key] not in (0., 1.):
            raise ValueError(f"invalid RR capture assist binary {key}")
    for key, maximum in (("travel_used_deg", 20.), ("descent_elapsed_s", 12.),
                          ("release_fraction", 1.)):
        if not 0. <= state[key] <= maximum + 1e-9:
            raise ValueError(f"invalid RR capture assist bounded {key}")
    if state["window_elapsed_s"] < 0. or state["hold_elapsed_s"] < 0.:
        raise ValueError("RR capture assist elapsed state cannot be negative")
    for key, index in (("hip_entry_deg", 6), ("hip_target_deg", 6), ("knee_hold_deg", 7)):
        lo, hi = servo_limits_deg(SERVO_ORDER[index])
        if not lo <= state[key] <= hi:
            raise ValueError(f"RR capture assist {key} violates canonical hard limits")
    if (state["mode"] == 0) != (state["initialized"] == 0):
        raise ValueError("RR capture assist initialization differs from mode")
    if state["mode"] not in (4., 5.) and state["release_fraction"] != 0.:
        raise ValueError("RR capture assist non-release state has a blend fraction")
    if state["mode"] == 5. and state["release_fraction"] != 1.:
        raise ValueError("RR capture assist completed release must be complete")
    if state["mode"] == 1. and (_exhaustion_reason(state) or state["retired"] or state["blocked_reason"]):
        raise ValueError("RR capture assist DESCEND cannot advertise exhausted/blocked/retired recovery")
    tick = snapshot["last_dispatch_physics_tick"]
    if tick is not None and (type(tick) is not int or tick < 0):
        raise ValueError("RR capture assist snapshot dispatch tick is invalid")
    expected = _snapshot(state, tick)
    if type(snapshot["active"]) is not bool or dict(snapshot) != expected:
        raise ValueError("RR capture assist snapshot metadata differs from its state")
    return state


def rr_capture_assist_features(snapshot: Mapping) -> tuple[float, ...]:
    state = validate_rr_capture_assist_snapshot(snapshot)
    return tuple(state[key] / scale for key, scale in zip(RR_CAPTURE_ASSIST_FEATURE_NAMES, _SCALES, strict=True))


def apply_rr_capture_assist_snapshot(candidate_full12: Sequence[float], snapshot: Mapping) -> tuple[float, ...]:
    """Pure shared real/zero-current-policy transform; never advances state."""
    candidate = _vector(candidate_full12, 12, "candidate")
    state = validate_rr_capture_assist_snapshot(snapshot)
    if not snapshot["active"]:
        return candidate
    fraction = state["release_fraction"] if state["mode"] == 4 else 0.
    result = list(candidate)
    for index, key in ((6, "hip_target_deg"), (7, "knee_hold_deg")):
        result[index] = (1. - fraction) * state[key] + fraction * candidate[index]
    return tuple(result)


class RRHipOnlyCaptureAssist:
    """Bounded physical-feedback search; all action-relevant state is public.

    Total assist descent is <=20 degrees and <=12 seconds, never renewed by
    contact loss or phase changes. At most 2 degrees/s (1 in the last 3 mm),
    2-second/0.2-mm local peak-to-current gap progress windows, with 3-degree
    hip AND knee tracking gates. The publicly versioned window_start_gap_m
    scalar (feature7 / observation396) stores the current window's peak gap.
    Rising gap never resets its clock or either total search budget.
    At the qualified XY candidate, ownership can bridge geometry retirement
    before the crossing event is latched; this only holds the previous FINAL
    target. Negative descent still requires the actual crossing history.
    Any physical contact stops descent; only real current TOP bearing starts
    P10/P11 release. First qualified RL lift retires this local controller.
    """
    def __init__(self):
        self.state = dict.fromkeys(RR_CAPTURE_ASSIST_FEATURE_NAMES, 0.)
        self.last_tick = None

    def snapshot(self) -> dict:
        result = _snapshot(dict(self.state), self.last_tick)
        validate_rr_capture_assist_snapshot(result)
        return result

    @classmethod
    def from_snapshot(cls, snapshot: Mapping):
        result = cls()
        result.state = validate_rr_capture_assist_snapshot(snapshot)
        result.last_tick = snapshot["last_dispatch_physics_tick"]
        return result

    @staticmethod
    def _anchor(state, previous, gap, *, first=False):
        state.update(initialized=1., mode=1., hip_target_deg=previous[6], knee_hold_deg=previous[7],
            release_fraction=0., window_start_gap_m=gap, window_elapsed_s=0., blocked_reason=0.)
        if first:
            state["hip_entry_deg"] = previous[6]

    @staticmethod
    def _start_release(state, previous):
        # Releasing from the last actual final target avoids replaying a stale
        # anchor after partial blends; it never restores old nominal entries.
        state.update(mode=4., hip_target_deg=previous[6], knee_hold_deg=previous[7],
                     release_fraction=0., blocked_reason=0.)

    @staticmethod
    def _release_step(state, dt):
        state["release_fraction"] = min(1., state["release_fraction"] + dt / .75)
        if state["release_fraction"] >= 1. - 1e-12:
            state.update(mode=5., release_fraction=1.)

    def advance(self, *, context: Mapping, previous_final_full12: Sequence[float], physics_dt_s: float) -> dict:
        if not isinstance(context, Mapping):
            raise ValueError("RR capture assist requires explicit physical context")
        tick = context.get("dispatch_physics_tick")
        if type(tick) is not int or tick < 0 or (self.last_tick is not None and tick != self.last_tick + 1):
            raise ValueError("RR capture assist requires adjacent real dispatch ticks")
        dt = _finite(physics_dt_s, "physics dt")
        if not math.isclose(dt, 1. / 120., rel_tol=0., abs_tol=1e-12):
            raise ValueError("RR capture assist retains the 120 Hz physical clock")
        stage = context.get("stage_id")
        if stage not in {f"P{i:02d}" for i in range(1, 14)}:
            raise ValueError("RR capture assist stage is invalid")
        for key in _BOOL_CONTEXT:
            if type(context.get(key)) is not bool:
                raise ValueError(f"RR capture assist requires explicit bool {key}")
        support = context.get("other_support_count")
        if type(support) is not int or not 0 <= support <= 3:
            raise ValueError("RR capture assist other support count is invalid")
        gap = _finite(context.get("gap_m"), "gap")
        hip = _finite(context.get("hip_actual_deg"), "actual hip")
        knee = _finite(context.get("knee_actual_deg"), "actual knee")
        previous = _vector(previous_final_full12, 12, "previous FINAL Full12")
        for i in (6, 7):
            lo, hi = servo_limits_deg(SERVO_ORDER[i])
            if not lo <= previous[i] <= hi:
                raise ValueError("RR capture assist previous FINAL violates its hard limits")
        contact = any(context[k] for k in ("top_surface_contact", "obstacle_pair_active", "ground_contact"))
        bearing = context["current_top_bearing"]
        if bearing and (not context["top_surface_contact"] or not context["obstacle_pair_active"]
                        or not context["within_top_xy"] or context["air"] or context["ground_contact"]):
            raise ValueError("RR current TOP bearing contradicts contact evidence")
        before = self.snapshot()
        state = dict(self.state)
        if context["rl_qualified_lift"]:
            state["retired"] = 1.
        window = stage in ("P09", "P10", "P11")
        reachable = (context["physical_valid"] and context["within_top_xy"] and support >= 2
                     and context["qualified_RR"] and context["air"] and not contact)
        ready = reachable and context["crossed_RR"]
        mode = state["mode"]
        if mode == 0:
            if not state["retired"] and stage == "P09" and reachable:
                self._anchor(state, previous, gap, first=True)
                if not context["crossed_RR"]:
                    # Bridge the existing clearance-helper's XY exit without
                    # spending descent budget or inventing crossing/support.
                    state.update(mode=3., blocked_reason=9.)
        elif mode == 5:
            if not state["retired"] and window and ready:
                self._anchor(state, previous, gap)  # finite budgets deliberately retained
        elif state["retired"] or not window:
            if mode != 4:
                self._start_release(state, previous)
            else:
                self._release_step(state, dt)
        elif mode == 4:
            if bearing and context["physical_valid"] and context["within_top_xy"] and support >= 2:
                self._release_step(state, dt)
            else:
                # Loss during release freezes this dispatch at its actual
                # previous final targets. No jump back to the old search pose.
                self._anchor(state, previous, gap)
                reason = (1 if not context["physical_valid"] else 2 if not context["within_top_xy"] else
                          3 if support < 2 else 9 if not (context["qualified_RR"] and context["crossed_RR"]) else
                          7 if not contact and not context["air"] else 0)
                state.update(mode=2. if contact else 3. if reason else 1., blocked_reason=float(reason))
                if contact:
                    state["contact_seen"] = 1.
        elif contact:
            state.update(mode=2., contact_seen=1., blocked_reason=0., window_start_gap_m=gap, window_elapsed_s=0.)
            state["hold_elapsed_s"] += dt
            if bearing and stage in ("P10", "P11") and context["physical_valid"] and context["within_top_xy"] and support >= 2:
                self._start_release(state, previous)
        else:
            if mode == 2:
                state.update(window_start_gap_m=gap, window_elapsed_s=0.)
            lower = servo_limits_deg(SERVO_ORDER[6])[0] + 2.
            reason = (1 if not context["physical_valid"] else 2 if not context["within_top_xy"] else
                3 if support < 2 else 9 if not (context["qualified_RR"] and context["crossed_RR"]) else
                7 if not context["air"] else
                5 if state["travel_used_deg"] >= 20. - 1e-9 or state["hip_target_deg"] <= lower + 1e-9 else
                8 if state["descent_elapsed_s"] >= 12. - 1e-9 else
                6 if max(abs(hip - state["hip_target_deg"]), abs(knee - state["knee_hold_deg"])) > 3. else 0)
            if not reason:
                # A whole-body reconfiguration can first raise the swing foot.
                # Recognize subsequent measured descent from the public local
                # peak, not only a new low below the earlier whole-body pose.
                # Invalid/support-lost/untracked samples cannot renew permission.
                state["window_start_gap_m"] = max(state["window_start_gap_m"], gap)
                if state["window_start_gap_m"] - gap >= .0002:
                    state.update(window_start_gap_m=gap, window_elapsed_s=0.)
                elif state["window_elapsed_s"] >= 2. - 1e-9:
                    reason = 4
            state.update(mode=3. if reason else 1., blocked_reason=float(reason))
            if not reason:
                exposure = min(dt, 12. - state["descent_elapsed_s"])
                amount = min((2. if gap > .003 else 1.) * exposure,
                             20. - state["travel_used_deg"], state["hip_target_deg"] - lower)
                state["hip_target_deg"] -= amount
                state["travel_used_deg"] += amount
                state["descent_elapsed_s"] += exposure
                state["window_elapsed_s"] += exposure
        # The last permitted bounded step may consume a budget. Advertise the
        # resulting hold immediately, rather than exposing one stale DESCEND
        # tick to downstream recovery-permission features. No extra step is
        # taken, and subsequent physical gap progress may reopen only the
        # local progress window, never total travel/exposure budgets.
        if state["mode"] == 1.:
            exhausted = _exhaustion_reason(state)
            if exhausted:
                state.update(mode=3., blocked_reason=float(exhausted))
        after = _snapshot(state, tick)
        validate_rr_capture_assist_snapshot(after)
        self.state, self.last_tick = state, tick
        return {"state_before": before, "state_after": after, "context": dict(context)}


def rr_capture_assist_context(*, task: Mapping, observation, source_frame, physics_tick: int,
                              support_spec: Mapping, previous_ack: Mapping | None = None) -> dict:
    """Build feedback from evaluator/current canonical joints, never raw policy.

    ``history.placed`` is intentionally absent: current measured bearing, not
    an old placement latch, controls release and contact-loss recovery.
    """
    def member(obj, name, default=None):
        return obj.get(name, default) if isinstance(obj, Mapping) else getattr(obj, name, default)
    floor = _finite(support_spec["force_noise_floor_n"], "support noise floor")
    if floor < 0.:
        raise ValueError("RR capture assist support floor cannot be negative")
    ev = task.get("physical_evaluator", {})
    legs, history = ev.get("current_legs", {}), ev.get("history", {})
    rr = legs.get("RR", {})
    def verified_support(row):
        return verified_current_support(row, floor)
    top_bearing = bool(verified_support(rr) and rr.get("ground_contact") is False
        and rr.get("contact_surface") == "TOP"
        and rr.get("top_contact") is True and rr.get("top_surface_contact") is True
        and rr.get("obstacle_pair_active") is True and rr.get("within_top_xy") is True)
    joints = member(observation, "joints", {})
    return {"dispatch_physics_tick": physics_tick, "source_observation_tick": ev.get("physics_tick"),
        "stage_id": member(source_frame, "state_id"),
        "physical_valid": bool(ev.get("valid") is True and ev.get("termination_reason") is None and task.get("termination_reason") is None),
        "within_top_xy": rr.get("within_top_xy") is True,
        "other_support_count": sum(verified_support(row) for leg, row in legs.items() if leg != "RR"),
        "qualified_RR": bool(history.get("active_lift", {}).get("RR") is True
                             and rr.get("current_lift_valid") is True),
        "crossed_RR": history.get("front_edge_crossed", {}).get("RR") is True,
        "rl_qualified_lift": history.get("active_lift", {}).get("RL") is True,
        "air": rr.get("air") is True, "ground_contact": rr.get("ground_contact") is True,
        "top_surface_contact": rr.get("top_surface_contact") is True,
        "obstacle_pair_active": rr.get("obstacle_pair_active") is True, "current_top_bearing": top_bearing,
        "gap_m": rr.get("clearance_m"),
        "hip_actual_deg": member(joints.get(SERVO_ORDER[6]), "position_deg"),
        "knee_actual_deg": member(joints.get(SERVO_ORDER[7]), "position_deg")}
