"""Opt-in finite RR hip-then-knee capture transform, not a policy or task event.

Only canonical RR hip/knee (6/7) are owned. The existing final hard limits,
slew and unique physical dispatch remain authoritative. Negative hip is an
explicit bounded diagnostic direction, followed by bounded positive knee;
neither direction is a claim about Cartesian motion.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from wlr50_clean.infrastructure.command_batch import SERVO_ORDER, servo_limits_deg
from wlr50_clean.ppo.semantic_rr_capture_context import (
    verified_current_support, RR_CAPTURE_GAP_MIN_M,
)

RR_CAPTURE_ASSIST_MODE = "rr_hip_only_capture_v1"
RR_CAPTURE_ASSIST_SCHEMA = "wlr50_clean.rr_capture_assist_state.v1"
RR_CAPTURE_FEEDBACK_REVISION = "progress_earned_capture_reserve_incremental_v10"
RR_CAPTURE_WINDOW_REFERENCE_SEMANTICS = "public_window_peak_gap_reuses_existing_window_start_gap_scalar_upward_motion_never_resets_elapsed"
RR_CAPTURE_SEARCH_SEMANTICS = "hip20_knee20_progress_earned_current_XY_support_tracking_gap_ge_minus15mm_knee12_then_1deg_public_peak_gap_le1mm_total53_exposure45_no_recharge_AIR_not_contact_sensor_TOP_unchanged_captured_issued_N_request_deltas_RL_current_TOP_retirement"
RR_CAPTURE_ASSIST_FEATURE_NAMES = (
    "mode", "initialized", "knee_hold_deg", "hip_entry_deg", "hip_target_deg",
    "travel_used_deg", "descent_elapsed_s", "window_start_gap_m", "window_elapsed_s",
    "hold_elapsed_s", "release_fraction", "contact_seen", "retired", "blocked_reason",
)
_SCALES = (5., 1., 180., 180., 180., 20., 12., .1, 2., 1., 1., 1., 1., 9.)
_MODES = ("WAIT", "DESCEND", "HOLD", "BLOCKED", "RELEASE", "RELEASED",
          "DESCEND_PROGRESS", "CAPTURED_FOLLOW")
_REASONS = ("none", "physical_invalid", "capture_XY_unavailable", "other_support_unavailable",
            "gap_not_improving", "finite_search_travel_or_margin", "waiting_actual_tracking",
            "current_AIR_unavailable", "finite_descent_exposure", "qualified_crossing_unavailable",
            "fresh_capture_progress_required")
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
    active = mode in (1, 2, 3, 4, 6, 7)
    owner = ("rr_capture_assist_finite_release" if mode == 4 else
             "rr_capture_incremental_follow" if mode == 7 else
             "rr_capture_assist" if active else "nominal_plus_policy")
    return {"schema": RR_CAPTURE_ASSIST_SCHEMA, "version": RR_CAPTURE_ASSIST_MODE,
        "feedback_revision": RR_CAPTURE_FEEDBACK_REVISION,
        "window_reference_semantics": RR_CAPTURE_WINDOW_REFERENCE_SEMANTICS,
        "capture_search_semantics": RR_CAPTURE_SEARCH_SEMANTICS,
        **state, "mode_name": _MODES[mode], "reason": _REASONS[int(state["blocked_reason"])],
        "active": active, "owners": [owner, owner], "owner_indices": [6, 7] if active else [],
        "last_dispatch_physics_tick": tick, "feature_names": list(RR_CAPTURE_ASSIST_FEATURE_NAMES)}


def _search_limits(state):
    if state["mode"] != 6.:
        return 40., 32.
    # A one-degree terminal approach is available only with the same earned
    # progress credit and a public measured window peak within 1 mm. This is
    # permission to keep executing descent, NEVER a substitute for contact.
    # The 52-degree sealed run ended 0.027 mm above TOP with no sensor force;
    # unchanged counters prevent contact toggles from recharging this budget.
    return ((53., 45.) if RR_CAPTURE_GAP_MIN_M <= state["window_start_gap_m"] <= .001
            else (52., 44.))


def _search_exhaustion_reason(state):
    # Axis and elapsed exposure are derived from the public cumulative
    # counters, never from a resettable anchor or a hidden phase latch.
    knee_axis = state["travel_used_deg"] >= 20.
    travel_end, exposure_end = _search_limits(state)
    if (state["travel_used_deg"] >= travel_end - 1e-9
            or (knee_axis and state["knee_hold_deg"] >= servo_limits_deg(SERVO_ORDER[7])[1] - 2. - 1e-9)
            or (not knee_axis and state["hip_target_deg"] <= servo_limits_deg(SERVO_ORDER[6])[0] + 2. + 1e-9)):
        return 5
    if state["descent_elapsed_s"] >= (exposure_end if knee_axis else 12.) - 1e-9:
        return 8
    return 0


def _exhaustion_reason(state):
    permanent = _search_exhaustion_reason(state)
    if permanent:
        return permanent
    if state["window_elapsed_s"] >= 2. - 1e-9:
        return 4
    return 0


def validate_rr_capture_assist_snapshot(snapshot: Mapping) -> dict:
    """Strict finite state/metadata validation for observation and replay."""
    metadata = {"schema", "version", "feedback_revision", "window_reference_semantics", "capture_search_semantics",
                "mode_name", "reason", "active", "owners", "owner_indices",
                "last_dispatch_physics_tick", "feature_names"}
    if not isinstance(snapshot, Mapping) or set(snapshot) != set(RR_CAPTURE_ASSIST_FEATURE_NAMES) | metadata:
        raise ValueError("RR capture assist requires its complete explicit snapshot")
    state = {k: _finite(snapshot[k], k) for k in RR_CAPTURE_ASSIST_FEATURE_NAMES}
    for key, maximum in (("mode", 7), ("blocked_reason", 10)):
        if state[key] != int(state[key]) or not 0 <= state[key] <= maximum:
            raise ValueError(f"invalid RR capture assist {key}")
    for key in ("initialized", "contact_seen", "retired"):
        if state[key] not in (0., 1.):
            raise ValueError(f"invalid RR capture assist binary {key}")
    for key, maximum in (("travel_used_deg", 53.), ("descent_elapsed_s", 45.),
                          ("release_fraction", 1.)):
        if not 0. <= state[key] <= maximum + 1e-9:
            raise ValueError(f"invalid RR capture assist bounded {key}")
    if state["window_elapsed_s"] < 0. or state["hold_elapsed_s"] < 0.:
        raise ValueError("RR capture assist elapsed state cannot be negative")
    knee_elapsed = max(state["travel_used_deg"] - 20., 0.)  # fixed 1 degree/s
    hip_elapsed = state["descent_elapsed_s"] - knee_elapsed
    hip_travel = min(state["travel_used_deg"], 20.)
    if not hip_travel / 2. - 1e-9 <= hip_elapsed <= min(12., hip_travel) + 1e-9:
        raise ValueError("RR capture assist DESCEND counters have inconsistent derived hip/knee exposure")
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
    if state["mode"] == 6. and (state["travel_used_deg"] < 20.
            or state["window_start_gap_m"] < RR_CAPTURE_GAP_MIN_M
            or state["window_elapsed_s"] >= 2.):
        raise ValueError("RR capture assist DESCEND_PROGRESS lacks public fresh capture progress credit")
    if state["mode"] in (1., 6.) and (_exhaustion_reason(state) or state["retired"] or state["blocked_reason"]):
        raise ValueError("RR capture assist DESCEND cannot advertise exhausted/blocked/retired recovery")
    if state["mode"] == 7. and (not state["contact_seen"] or state["retired"]):
        raise ValueError("RR captured follow needs actual contact history and live ownership")
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


def apply_rr_capture_assist_snapshot(candidate_full12: Sequence[float], snapshot: Mapping,
                                    *, previous_final_full12: Sequence[float] | None = None) -> tuple[float, ...]:
    """Pure shared real/zero-current-policy transform; never advances state."""
    candidate = _vector(candidate_full12, 12, "candidate")
    state = validate_rr_capture_assist_snapshot(snapshot)
    if not snapshot["active"]:
        return candidate
    result = list(candidate)
    paused_capture = ((state["contact_seen"] == 1. and state["mode"] == 3.) or state["mode"] == 4.) \
        and state["blocked_reason"] in (1., 2., 3., 7., 9.)
    previous = (_vector(previous_final_full12, 12, "paused capture previous FINAL")
                if paused_capture else None)
    for index, key in ((6, "hip_target_deg"), (7, "knee_hold_deg")):
        # RELEASE is integrated in advance, too: a paused release must not
        # move merely because its external absolute candidate changes.
        result[index] = previous[index] if previous is not None else state[key]
    return tuple(result)


class RRHipOnlyCaptureAssist:
    """Bounded physical-feedback search; all action-relevant state is public.

    Negative hip search is <=20 degrees and <=12 seconds; only after its
    full 20 degrees may positive knee search spend 20 degrees at 1 degree/s.
    One further <=12 degrees requires public DESCEND_PROGRESS mode, current
    gap >= -15 mm and all existing XY/support/tracking/fresh-progress gates.
    The sensor TOP band's 25 mm upper bound is not a descent permission cap:
    earned progress may spend the existing reserve while still above it.
    No positive gap or AIR sample is contact. Public
    travel is normally <=52 degrees and active exposure <=44 seconds. A final
    <=1 degree/1 second is available only under that same earned progress and
    public window peak <=1 mm; absolute totals are <=53 degrees/45 seconds.
    No task/contact threshold is relaxed. Knee exposure
    is max(travel-20, 0); hip exposure is elapsed
    minus knee exposure. Neither is renewed by contact loss or phase changes.
    Hip speed is at most 2 degrees/s (1 in the last 3 mm), with
    2-second/0.2-mm local peak-to-current gap progress windows, with 3-degree
    hip AND knee tracking gates. The publicly versioned window_start_gap_m
    scalar (feature7 / observation396) stores the current window's peak gap.
    Rising gap never resets its clock or either total search budget. The
    single actual hip-to-knee boundary starts a fresh local progress window.
    The legacy knee_hold_deg scalar is now the current knee capture target.
    At the qualified XY candidate, ownership can bridge geometry retirement
    before the crossing event is latched; this only holds the previous FINAL
    target. Negative descent still requires the actual crossing history.
    Any physical contact stops descent. After actual contact, P10-P13 add
    issued source/request deltas to persistent public targets, not to FINAL
    and not an absolute saturated candidate. Current RL TOP placement and
    support, never RL lift history, permits finite feedback-gated release.
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
    def _start_release(state):
        # Keep any pending issued target. The public fraction is never renewed
        # on support/contact pauses, and no absolute target is applied at entry.
        state.update(mode=4., release_fraction=0., blocked_reason=0., retired=1., contact_seen=1.)

    @staticmethod
    def _release_step(state, candidate, dt):
        before = state["release_fraction"]
        after = min(1., before + dt / .75)
        weight = (after - before) / (1. - before)
        for index, key, value in zip((6, 7), ("hip_target_deg", "knee_hold_deg"), candidate, strict=True):
            lo, hi = servo_limits_deg(SERVO_ORDER[index])
            # Original hard limits remain final authority; this owner retains
            # the original two-degree reserve during the controlled release.
            value = max(lo + 2., min(hi - 2., value))
            state[key] += weight * (value - state[key])
        state["release_fraction"] = after
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
        top_seen = bool(context["top_surface_contact"] and context["obstacle_pair_active"]
                        and context["within_top_xy"] and not context["air"] and not context["ground_contact"])
        bearing = context["current_top_bearing"]
        if bearing and (not context["top_surface_contact"] or not context["obstacle_pair_active"]
                        or not context["within_top_xy"] or context["air"] or context["ground_contact"]):
            raise ValueError("RR current TOP bearing contradicts contact evidence")
        before = self.snapshot()
        state = dict(self.state)
        rl_finished = context.get("rl_placed_current_top_support", False)
        if type(rl_finished) is not bool:
            raise ValueError("RR capture assist requires bool rl_placed_current_top_support")
        window = stage in ("P09", "P10", "P11", "P12", "P13")
        reachable = (context["physical_valid"] and context["within_top_xy"] and support >= 2
                     and gap >= RR_CAPTURE_GAP_MIN_M
                     and context["qualified_RR"] and context["air"] and not contact)
        ready = reachable and context["crossed_RR"]
        mode = state["mode"]
        tracking = max(abs(hip - state["hip_target_deg"]), abs(knee - state["knee_hold_deg"])) <= 3.
        # An AIR capture needs two other measured supports. After actual RR
        # bearing, a real two-point bridge (RR plus one other) is admissible;
        # this count is a necessary gate, never a dynamic stability proof.
        support_ready = context["physical_valid"] and context["within_top_xy"] and support >= 1
        # Current support is never inferred from contact_seen or placed history.
        follow = bool(state["initialized"] and not state["retired"]
            and stage in ("P10", "P11", "P12", "P13")
            and (state["contact_seen"] or top_seen))
        follow_safe = support_ready and (bearing or ready)
        increment_receipt = None
        if follow:
            nominal_delta = _vector(context.get("issued_nominal_delta_rr_deg"), 2, "issued nominal RR delta")
            residual_delta = _vector(context.get("issued_requested_residual_delta_rr_deg"), 2,
                                     "issued requested residual RR delta")
            old_target = (state["hip_target_deg"], state["knee_hold_deg"])
            requested_target = tuple(a + n + r for a, n, r in
                                     zip(old_target, nominal_delta, residual_delta, strict=True))
            if any(not math.isfinite(value) for value in requested_target):
                raise ValueError("RR captured target increment overflowed")
            for index, key, value in zip((6, 7), ("hip_target_deg", "knee_hold_deg"), requested_target, strict=True):
                lo, hi = servo_limits_deg(SERVO_ORDER[index])
                state[key] = max(lo + 2., min(hi - 2., value))
            increment_receipt = {"nominal_delta_rr_deg": list(nominal_delta),
                "requested_residual_delta_rr_deg": list(residual_delta),
                "previous_requested_target_rr_deg": list(old_target),
                "requested_target_before_margin_rr_deg": list(requested_target),
                "target_after_margin_rr_deg": [state["hip_target_deg"], state["knee_hold_deg"]],
                "nominal_mapper_or_raw_log_probability_rewritten": False}
            # Tracking gates only additional capture search, not retention of
            # issued deltas whose pending target has yet to complete final slew.
            tracking = max(abs(hip - state["hip_target_deg"]), abs(knee - state["knee_hold_deg"])) <= 3.
        if mode == 0:
            if not state["retired"] and not rl_finished and stage == "P09" and reachable:
                self._anchor(state, previous, gap, first=True)
                if not context["crossed_RR"]:
                    # Bridge the existing clearance-helper's XY exit without
                    # spending descent budget or inventing crossing/support.
                    state.update(mode=3., blocked_reason=9.)
        elif mode == 5:
            pass  # retired; neither phase changes nor AIR can renew ownership
        elif mode == 4:
            if rl_finished and bearing and support_ready and tracking:
                candidate = _vector(context.get("release_candidate_rr_deg"), 2, "release RR candidate")
                state["blocked_reason"] = 0.
                self._release_step(state, candidate, dt)
            else:
                # No reanchor, renewed timer, or moving external blend target.
                state["blocked_reason"] = float(1 if not context["physical_valid"] else
                    2 if not context["within_top_xy"] else 3 if support < 1 else
                    6 if not tracking else 7)
        elif rl_finished and bearing and support_ready and tracking:
            self._start_release(state)
        elif not window:
            state.update(mode=3., blocked_reason=1.)
        elif follow and not follow_safe:
            # Preserve every issued source/request delta in public pending
            # target, while apply_snapshot holds the previous physical FINAL.
            # No source event is silently discarded or replayed after recovery.
            state.update(mode=3., contact_seen=float(bool(state["contact_seen"] or top_seen)),
                blocked_reason=float(1 if not context["physical_valid"] else
                    2 if not context["within_top_xy"] else 3 if support < (1 if bearing else 2) else
                    9 if context["air"] and not (context["qualified_RR"] and context["crossed_RR"]) else 7))
        elif contact:
            state.update(mode=7. if follow and bearing else 2.,
                         contact_seen=float(bool(state["contact_seen"] or top_seen)), blocked_reason=0.,
                         window_start_gap_m=gap, window_elapsed_s=0.)
            state["hold_elapsed_s"] += dt
        else:
            progress_credit = mode == 6.
            if mode in (2, 7):
                # Contact loss retains target/budgets, but a reset window is
                # not evidence of descent and must not grant reserve credit.
                state.update(window_start_gap_m=gap, window_elapsed_s=0.)
            reason = (1 if not context["physical_valid"] else
                2 if not context["within_top_xy"] or gap < RR_CAPTURE_GAP_MIN_M else
                3 if support < 2 else 9 if not (context["qualified_RR"] and context["crossed_RR"]) else
                7 if not context["air"] else
                6 if not tracking else 0)
            if not reason:
                # A whole-body reconfiguration can first raise the swing foot.
                # Recognize subsequent measured descent from the public local
                # peak, not only a new low below the earlier whole-body pose.
                # Invalid/support-lost/untracked samples cannot renew permission.
                state["window_start_gap_m"] = max(state["window_start_gap_m"], gap)
                if state["window_start_gap_m"] - gap >= .0002:
                    state.update(window_start_gap_m=gap, window_elapsed_s=0.)
                    progress_credit = (state["travel_used_deg"] >= 20.
                        and gap >= RR_CAPTURE_GAP_MIN_M)
                elif state["window_elapsed_s"] >= 2. - 1e-9:
                    reason = 4
                progress_credit = bool(progress_credit and gap >= RR_CAPTURE_GAP_MIN_M
                                       and state["window_elapsed_s"] < 2. - 1e-9)
            else:
                progress_credit = False  # invalid samples cannot keep reserve permission alive
            state.update(mode=6. if progress_credit and not reason else 1., blocked_reason=0.)
            if not reason:
                reason = _search_exhaustion_reason(state)
                reserve_available = (state["travel_used_deg"] < 52. - 1e-9 or
                    (RR_CAPTURE_GAP_MIN_M <= state["window_start_gap_m"] <= .001
                     and state["travel_used_deg"] < 53. - 1e-9))
                if reason == 5 and state["travel_used_deg"] >= 40. - 1e-9 and reserve_available and not progress_credit:
                    reason = 10
            state.update(mode=3. if reason else state["mode"], blocked_reason=float(reason))
            if not reason:
                knee_axis = state["travel_used_deg"] >= 20.
                rate = 1. if knee_axis or gap <= .003 else 2.
                travel_end, elapsed_end = _search_limits(state) if knee_axis else (20., 12.)
                key = "knee_hold_deg" if knee_axis else "hip_target_deg"
                margin = (servo_limits_deg(SERVO_ORDER[7])[1] - 2. - state[key] if knee_axis
                          else state[key] - servo_limits_deg(SERVO_ORDER[6])[0] - 2.)
                remaining = travel_end - state["travel_used_deg"]
                amount = min(rate * min(dt, elapsed_end - state["descent_elapsed_s"]), remaining, margin)
                exposure = amount / rate
                state[key] += amount if knee_axis else -amount
                # Exact boundary only after the commanded remaining amount;
                # no near-boundary jump or repeated axis/window reset.
                state["travel_used_deg"] = travel_end if amount == remaining else state["travel_used_deg"] + amount
                state["descent_elapsed_s"] += exposure
                state["window_elapsed_s"] += exposure
                if not knee_axis and state["travel_used_deg"] == 20.:
                    state.update(mode=1., window_start_gap_m=gap, window_elapsed_s=0.)
        # The last permitted bounded step may consume a budget. Advertise the
        # resulting hold immediately, rather than exposing one stale DESCEND
        # tick to downstream recovery-permission features. No extra step is
        # taken, and subsequent physical gap progress may reopen only the
        # local progress window, never total travel/exposure budgets.
        if state["mode"] in (1., 6.):
            exhausted = _exhaustion_reason(state)
            if exhausted:
                state.update(mode=3., blocked_reason=float(exhausted))
        after = _snapshot(state, tick)
        validate_rr_capture_assist_snapshot(after)
        self.state, self.last_tick = state, tick
        return {"state_before": before, "state_after": after, "context": dict(context),
                "captured_incremental_target": increment_receipt}


def rr_capture_assist_context(*, task: Mapping, observation, source_frame, physics_tick: int,
                              support_spec: Mapping, previous_ack: Mapping | None = None) -> dict:
    """Build feedback from evaluator/current canonical joints, never raw policy.

    Old RR placement is never current support. RL retirement requires its
    placement history AND current verified, geometrically legal TOP bearing.
    The adapter adds same-dispatch issued N/request deltas and release candidate;
    this sensor helper never guesses them from an independently rerun mapper.
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
    def top_bearing_of(row):
        return bool(verified_support(row) and row.get("ground_contact") is False
            and row.get("contact_surface") == "TOP"
            and row.get("top_contact") is True and row.get("top_surface_contact") is True
            and row.get("obstacle_pair_active") is True and row.get("within_top_xy") is True)
    top_bearing = top_bearing_of(rr)
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
        "rl_placed_current_top_support": bool(history.get("placed", {}).get("RL") is True
                                               and top_bearing_of(legs.get("RL", {}))),
        "air": rr.get("air") is True, "ground_contact": rr.get("ground_contact") is True,
        "top_surface_contact": rr.get("top_surface_contact") is True,
        "obstacle_pair_active": rr.get("obstacle_pair_active") is True, "current_top_bearing": top_bearing,
        "gap_m": rr.get("clearance_m"),
        "hip_actual_deg": member(joints.get(SERVO_ORDER[6]), "position_deg"),
        "knee_actual_deg": member(joints.get(SERVO_ORDER[7]), "position_deg")}
