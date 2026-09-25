"""Continuous RR capture -> RL traversal task; pure measured state and reward.

Staged v3 candidate, not a claim of physical success. Reuses the v2 gate,
committed entry, same-attempt qualification, and native hold measurements.
Short RR TOP is a once-per-episode milestone, NEVER a local terminal. Near-top
and post-touch grace permit preparation only: neither is measured support.
The source controller remains responsible for current-bearing dependencies.
No targets, phase clocks, history filters, or simulator state are written here.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import math

from .semantic_rr_capture_local_task import (
    OBSERVATION_FIELDS as LEGACY_FIELDS,
    RRCaptureLocalTask, RRCaptureLocalTaskConfig, _finite,
)


SCHEMA = "wlr50_clean.rr_capture_continuation_task.v3"
SUCCESS = "FULL_TRAVERSAL_SUCCESS"
CONTINUATION_FIELDS = (
    "rr_touch_seen", "rr_touch_current_attempt", "rr_support_continuation_valid",
    "rr_geometry_ready_for_RL_prep", "rl_lift_seen", "rl_cross_seen", "rl_touch_seen",
)
OBSERVATION_FIELDS = LEGACY_FIELDS + CONTINUATION_FIELDS
OBSERVATION_DIM = len(OBSERVATION_FIELDS)


@dataclass(frozen=True)
class RRCaptureContinuationTaskConfig(RRCaptureLocalTaskConfig):
    near_enter_gap_m: float = .008
    near_exit_gap_m: float = .012
    touch_grace_enter_gap_m: float = .010
    touch_grace_exit_gap_m: float = .012
    rr_touch_reward: float = 10.
    rl_lift_reward: float = 10.
    rl_cross_reward: float = 15.
    rl_touch_reward: float = 20.
    full_success_reward: float = 80.
    global_deadline_s: float = 200.

    def __post_init__(self):
        super().__post_init__()
        positive = (self.near_enter_gap_m, self.near_exit_gap_m,
                    self.touch_grace_enter_gap_m, self.touch_grace_exit_gap_m,
                    self.rr_touch_reward, self.rl_lift_reward, self.rl_cross_reward,
                    self.rl_touch_reward, self.full_success_reward)
        if not all(_finite(x) and x > 0. for x in positive):
            raise ValueError("continuation scales must be positive finite values")
        if (self.near_enter_gap_m >= self.near_exit_gap_m
                or self.touch_grace_enter_gap_m >= self.touch_grace_exit_gap_m):
            raise ValueError("continuation exit gaps must exceed entry gaps")
        if self.global_deadline_s != 200.:
            raise ValueError("continuous traversal retains the global 200 second deadline")


def _evaluator(source):
    if hasattr(source, "frame"):
        source = source.frame
    info = source if isinstance(source, Mapping) else source.info
    return info["semantic_task"]["physical_evaluator"]


def _unit(value):
    return min(1., max(0., float(value))) if _finite(value) else 0.


def _continuation_evidence(source, floor):
    """Only current measured RL qualification can validate old crossing flags."""
    ev = _evaluator(source)
    row = (ev.get("current_legs") or {}).get("RL")
    available = isinstance(row, Mapping)
    row = row if available else {}
    history = ev.get("history") or {}
    live = ev.get("valid") is True and ev.get("termination_reason") is None
    free_air = bool(row.get("air") is True and row.get("ground_contact") is False
                    and row.get("obstacle_pair_active") is False)
    # The current RL v3 evaluator exposes a qualified latch/tick, cleared on
    # EVERY GROUND, not RR's lift_established field. Its current_lift_valid
    # remains true at legitimate TOP. Keep an explicit establishment field
    # authoritative when provided, so future AIR-validity refinements cannot
    # accidentally reject an established attempt at touchdown (or resurrect
    # one explicitly revoked after ground).
    established = row.get("lift_established")
    qualified_tick = row.get("current_lift_qualified_tick")
    current_tick = ev.get("physics_tick")
    tick_proof = bool(row.get("current_lift_evidence_semantics")
        == "same_attempt_measured_lift_revoked_by_any_ground_v3"
        and type(qualified_tick) is int and type(current_tick) is int
        and 0 <= qualified_tick <= current_tick)
    if type(established) is bool:
        establishment_source = "explicit_current_lift_established"
    else:
        established = bool(row.get("current_lift_valid") is True or tick_proof)
        establishment_source = ("current_RL_qualified_latch" if row.get("current_lift_valid") is True
                                else "current_RL_qualified_tick_v3" if tick_proof else "unavailable")
    qualified = bool(live and established
        and row.get("motion_continuation_allowed") is True
        and row.get("ground_contact") is False)
    legal = row.get("within_top_xy") is True and row.get("within_lateral_span") is True
    top = bool(live and legal and row.get("ground_contact") is False
        and row.get("air") is False and row.get("top_contact") is True
        and row.get("top_surface_contact") is True
        and row.get("obstacle_pair_active") is True and row.get("contact_surface") == "TOP")
    force = row.get("bearing_force_n")
    samples = row.get("consecutive_top_samples")
    role = (ev.get("transfer_roles") or {}).get("RL")
    role = role if isinstance(role, Mapping) else {}
    correct_role = bool(live and role.get("valid") is True
                       and role.get("diagonal_receiving_side") == "FR")
    return dict(rl_metrics=dict(
        available=available, current_qualified=qualified,
        current_lift_valid=row.get("current_lift_valid") is True,
        current_air_lift_valid=bool(qualified and free_air and row.get("current_lift_valid") is True),
        lift_established=established, current_attempt_capture_eligible=qualified,
        establishment_source=establishment_source,
        current_lift_qualified_tick=qualified_tick if type(qualified_tick) is int else None,
        free_air=free_air,
        ground_contact=row.get("ground_contact") is True,
        within_top_xy=row.get("within_top_xy") is True,
        within_lateral_span=row.get("within_lateral_span") is True,
        crossed=(history.get("front_edge_crossed") or {}).get("RL") is True,
        placed=(history.get("placed") or {}).get("RL") is True,
        current_top_contact=top,
        current_top_bearing=bool(top and row.get("support") is True
            and row.get("bearing_verified") is True and _finite(force) and force >= floor),
        bearing_force_n=float(force) if _finite(force) else None,
        gap_m=float(row["clearance_m"]) if _finite(row.get("clearance_m")) else None,
        consecutive_top_samples=samples if type(samples) is int and samples >= 0 else 0,
        receiver_role_valid=correct_role, diagonal_receiving_side=role.get("diagonal_receiving_side"),
        workspace_progress=_unit(role.get("workspace_progress")) if correct_role else None,
        preparation_progress=_unit(role.get("preparation_progress")) if correct_role else None,
        transfer_progress=_unit(role.get("transfer_progress")) if correct_role else None,
        transfer_direction_context=deepcopy(role.get("transfer_direction_context")) if correct_role else None),
        # Delegate success to the existing physical evaluator, not stage P13,
        # historical placed, a source clock, or either local RR predicate.
        full_task_success=bool(ev.get("valid") is True and ev.get("success") is True
                               and ev.get("termination_reason") in (None, "SUCCESS", SUCCESS)),
        physical_valid=ev.get("valid") is True,
        final_region_valid=ev.get("final_region_valid") is True,
        final_controlled=ev.get("final_controlled") is True,
        final_support_available=ev.get("final_support_available") is True)


class RRCaptureContinuationTask:
    """All seven new state bits are visible through obs16 and snapshot.

    The old nine inputs keep their positions. ``rr_touch_seen`` is episode
    history for once-only reward. ``rr_touch_current_attempt`` is revocable.
    Existing evaluator within_top_xy already includes its configured 5 mm
    measurement tolerance; this task adds no new lateral/edge forgiveness.
    Same-tick reads are idempotent; the callback consumes the last completed
    native frame. It must not claim the following, not-yet-stepped frame.
    """
    def __init__(self, config=None):
        self.config = RRCaptureContinuationTaskConfig() if config is None else config
        if not isinstance(self.config, RRCaptureContinuationTaskConfig):
            raise ValueError("validated continuous task config required")
        self._capture = RRCaptureLocalTask(self.config)
        self.reset()

    def __getattr__(self, name):
        # Preserve read-only legacy gate/entry/hold access used by the core.
        if name in ("active", "activation_tick", "activation_time_s", "metrics",
                    "entry_rr_hip_deg", "entry_rr_knee_deg", "entry_gap_m",
                    "hold_elapsed_s", "hold_started_s"):
            return getattr(self._capture, name)
        raise AttributeError(name)

    @property
    def local_success(self):
        # Compatibility safeguard against an old local-done consumer.
        return False

    def reset(self):
        self._capture.reset()
        for field in CONTINUATION_FIELDS:
            setattr(self, field, False)
        self._evidence = None

    def observe(self, source):
        evidence = _continuation_evidence(source, self.config.force_noise_floor_n)
        tick = _evaluator(source)["physics_tick"]
        if self.metrics is not None and tick == self.metrics["tick"]:
            if evidence != self._evidence:
                raise ValueError("conflicting continuous evidence at the same observation tick")
            self._capture.observe(source)  # Includes committed ACK/RR checks.
            return self.snapshot()
        self._capture.observe(source)
        self._evidence = evidence
        m, cfg = self.metrics, self.config
        legal = bool(self.active and m["live"] and m["geometry_valid"]
            and m["current_attempt_capture_eligible"] and not m["ground_contact"]
            and m["within_top_xy"] and m["within_lateral_span"])
        if not legal:
            self.rr_touch_current_attempt = False
            self.rr_support_continuation_valid = False
            self.rr_geometry_ready_for_RL_prep = False
        else:
            touch = bool(m["current_top_contact"]
                         and m["consecutive_top_samples"] >= cfg.minimum_top_samples)
            if touch:
                self.rr_touch_seen = self.rr_touch_current_attempt = True
            descent_region = bool(m["free_air"] and m["gap_m"] >= cfg.minimum_legal_gap_m)
            grace_limit = (cfg.touch_grace_exit_gap_m if self.rr_support_continuation_valid
                           else cfg.touch_grace_enter_gap_m)
            self.rr_support_continuation_valid = bool(m["current_top_contact"]
                or (self.rr_touch_current_attempt and descent_region and m["gap_m"] <= grace_limit))
            near_limit = (cfg.near_exit_gap_m if self.rr_geometry_ready_for_RL_prep
                          else cfg.near_enter_gap_m)
            self.rr_geometry_ready_for_RL_prep = bool(m["current_top_contact"]
                or self.rr_support_continuation_valid
                or (descent_region and m["gap_m"] <= near_limit))
        rl = evidence["rl_metrics"]
        if self.active and m["live"] and rl["current_qualified"]:
            self.rl_lift_seen |= rl["current_air_lift_valid"]
            self.rl_cross_seen |= rl["crossed"] and rl["within_top_xy"]
            self.rl_touch_seen |= (rl["crossed"] and rl["current_top_contact"]
                                  and rl["consecutive_top_samples"] >= cfg.minimum_top_samples)
        return self.snapshot()

    def obs9(self):
        return self._capture.obs9()

    def obs16(self):
        return self.obs9() + tuple(float(getattr(self, field)) for field in CONTINUATION_FIELDS)

    def potential_components(self):
        result = dict(rr_capture=0., rl_workspace=0., rl_preparation=0., rl_transfer=0.,
                      rl_qualified_lift=0., rl_cross=0., rl_contact=0., full_finish=0.)
        if not (self.active and self.metrics and self.metrics["live"] and self._evidence):
            return result
        result["rr_capture"] = .45 * sum(self._capture.potential_components().values())
        rl = self._evidence["rl_metrics"]
        # Only the RL->FR role is used. There is no simultaneous reward for
        # continuing the old RR->FL transfer or for zero roll/pitch.
        if self.rr_geometry_ready_for_RL_prep or self.rr_touch_seen or rl["current_qualified"]:
            if rl["receiver_role_valid"] and not self.rl_touch_seen:
                result.update(rl_workspace=.04 * rl["workspace_progress"],
                              rl_preparation=.04 * rl["preparation_progress"],
                              rl_transfer=.07 * rl["transfer_progress"])
            if rl["current_qualified"]:
                result["rl_qualified_lift"] = .10
                result["rl_cross"] = .10 * float(rl["crossed"] and rl["within_top_xy"])
                result["rl_contact"] = .10 * float(rl["crossed"] and rl["current_top_contact"])
        ev = self._evidence
        result["full_finish"] = .1 * float(ev["final_region_valid"]
            and ev["final_controlled"] and ev["final_support_available"])
        return result

    def snapshot(self):
        state = self._capture.snapshot()
        parts = self.potential_components()
        m, ev = self.metrics, self._evidence
        state.update(schema=SCHEMA, local_success=False,
            rr_capture_hold_quality=self._capture.local_success,
            rr_milestone=self.rr_touch_seen,
            rr_contact_now=bool(m and m["current_top_contact"]),
            rr_bearing_now=bool(m and m["current_top_bearing"]),
            rr_transient_grace=bool(self.rr_support_continuation_valid and m
                                    and not m["current_top_contact"]),
            observation_tick=m["tick"] if m else None,
            observation_time_s=m["time_s"] if m else None,
            obs16=self.obs16(), potential=sum(parts.values()), potential_components=parts,
            rl_metrics=deepcopy(ev["rl_metrics"]) if ev else None,
            full_task_success=bool(ev and ev["full_task_success"]),
            support_continuation_semantics="task_permission_not_measured_force",
            physical_state_or_target_writes=0)
        state.update({field: bool(getattr(self, field)) for field in CONTINUATION_FIELDS})
        return state

    def read_local_state(self):
        return self.snapshot()

    def reward(self, before, *, termination_reason=None):
        if not isinstance(before, Mapping) or before.get("schema") != SCHEMA:
            raise ValueError("continuous reward requires its decision-start snapshot")
        after = self.snapshot()
        previous, current = before.get("metrics"), self.metrics
        if (not isinstance(previous, Mapping) or current is None
                or current["tick"] <= previous["tick"] or current["time_s"] <= previous["time_s"]):
            raise ValueError("continuous reward requires a positive real decision interval")
        if termination_reason is not None and (not isinstance(termination_reason, str) or not termination_reason):
            raise ValueError("physical terminal reason must be a nonempty string")
        reason = termination_reason or current["termination_reason"]
        if reason in ("RR_CAPTURE_HOLD_SUCCESS", "RR_CAPTURE_HOLD_LOCAL_SUCCESS"):
            raise ValueError("RR milestone is not a continuous-task terminal")
        success = after["full_task_success"]
        if reason in ("SUCCESS", SUCCESS) and not success:
            raise ValueError("global success requires the real physical evaluator")
        if success and reason not in (None, "SUCCESS", SUCCESS):
            success = False  # Never override a real safety/physical terminal.
        if reason is None and not success and current["time_s"] >= self.config.global_deadline_s:
            reason = "GLOBAL_TASK_DEADLINE_200S"
        terminal = success or reason is not None
        credited = before.get("active") is True
        phi_before = before.get("potential")
        if not _finite(phi_before) or not 0. <= phi_before <= 1. + 1e-12:
            raise ValueError("invalid continuous decision-start potential")
        phi_after = 0. if terminal else after["potential"]
        event_weights = dict(rr_touch_seen=self.config.rr_touch_reward,
            rl_lift_seen=self.config.rl_lift_reward, rl_cross_seen=self.config.rl_cross_reward,
            rl_touch_seen=self.config.rl_touch_reward)
        events = {key: weight if after[key] and not before.get(key, False) else 0.
                  for key, weight in event_weights.items()}
        terminal_event = (self.config.full_success_reward if success
                          else -self.config.failure_cost if reason else 0.)
        dt = current["time_s"]-previous["time_s"]
        shaping = self.config.potential_weight*(self.config.gamma*phi_after-phi_before)
        cost = self.config.time_cost_per_s*dt
        return dict(schema=SCHEMA, reward=(shaping+sum(events.values())+terminal_event-cost) if credited else 0.,
            on_policy_rr_sample=credited, on_policy_continuation_sample=credited, prefix_excluded=not credited,
            potential_before=phi_before, potential_after=phi_after,
            potential_shaping=shaping if credited else 0.,
            milestone_events=events if credited else {key: 0. for key in events},
            terminal_event=terminal_event if credited else 0., time_cost=cost if credited else 0.,
            elapsed_physics_s=dt, gamma=self.config.gamma,
            discount_convention="once_per_issued_policy_decision",
            terminated=terminal, truncated=False, terminal_bootstrap_allowed=not terminal,
            termination_reason=reason or (SUCCESS if success else None),
            rr_subtask_success=self.rr_touch_seen, rr_milestone=self.rr_touch_seen,
            full_task_success=success, current_potential_components=after["potential_components"])
