"""Pure measured RR capture task; no action, simulator, or learning imports.

Call ``observe(core_or_frame_or_info)`` on every native physics observation,
including the frozen prefix. It never changes the supplied objects. Capture
``before = task.snapshot()`` before issuing a policy decision, then call
``task.reward(before)`` after its native ticks. Gamma is applied ONCE per
decision, not once per observation. The caller owns actual episode termination
and bootstrap; ordinary phase changes and rollout budgets are not terminals.

The policy gate latches at qualified AIR after the existing RR crossing. It
survives contact and loss of load. A success is current verified capture plus
continuous measured bearing observation, not a historical placed flag alone.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import math


SCHEMA = "wlr50_clean.rr_capture_local_task.v1"
SUCCESS = "RR_CAPTURE_HOLD_SUCCESS"
OBSERVATION_FIELDS = (
    "active", "activation_age_norm", "entry_rr_hip_norm", "entry_rr_knee_norm",
    "entry_gap_norm", "current_top_contact", "current_top_bearing",
    "capture_hold_progress",
)
OBSERVATION_DIM = len(OBSERVATION_FIELDS)


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _vector12(value):
    if not isinstance(value, (tuple, list)) or len(value) != 12 or not all(map(_finite, value)):
        raise ValueError("committed FINAL must contain twelve finite canonical values")
    return tuple(float(x) for x in value)


@dataclass(frozen=True)
class RRCaptureLocalTaskConfig:
    gamma: float = .9985
    physics_dt_s: float = 1. / 120.
    force_noise_floor_n: float = .2
    minimum_top_samples: int = 2
    hold_duration_s: float = .5
    gap_scale_m: float = .025
    minimum_legal_gap_m: float = -.015
    xy_scale_m: float = .25
    potential_weight: float = 5.
    success_reward: float = 40.
    failure_cost: float = 40.
    time_cost_per_s: float = .02
    activation_age_scale_s: float = 200.
    entry_target_scale_deg: float = 180.
    entry_gap_scale_m: float = .1

    def __post_init__(self):
        if self.gamma != .9985:
            raise ValueError("RR local return version requires gamma=.9985")
        positive = (self.physics_dt_s, self.force_noise_floor_n, self.hold_duration_s,
                    self.gap_scale_m, self.xy_scale_m, self.potential_weight,
                    self.success_reward, self.failure_cost, self.activation_age_scale_s,
                    self.entry_target_scale_deg, self.entry_gap_scale_m)
        if not all(_finite(x) and x > 0 for x in positive):
            raise ValueError("RR local task scales must be finite and positive")
        if not math.isclose(self.physics_dt_s, 1./120., rel_tol=0., abs_tol=1e-12):
            raise ValueError("RR local task requires the existing 120 Hz observation clock")
        if (not _finite(self.time_cost_per_s) or self.time_cost_per_s < 0
                or not _finite(self.minimum_legal_gap_m)
                or type(self.minimum_top_samples) is not int or self.minimum_top_samples < 2):
            raise ValueError("invalid RR local task cost/contact confirmation")


def extract_rr_metrics(source, *, force_noise_floor_n=.2):
    """Read SemanticEpisodeEnv, AuthoritativeFrame, or its ``info`` mapping.

    Physical fields come from info.semantic_task.physical_evaluator. The entry
    targets come only from the last committed info.drive_target_full12 (or the
    decision receipt's actual_drive_target_full12), never nominal or actual q.
    If present, atomic_ack must agree with that FINAL. Raw actual q is optional
    diagnostic evidence. Missing/invalid physical geometry cannot qualify.
    """
    if hasattr(source, "frame"):
        source = source.frame
    frame = None if isinstance(source, Mapping) else source
    info = source if frame is None else frame.info
    if not isinstance(info, Mapping):
        raise ValueError("RR task requires a frame info mapping")
    task = info.get("semantic_task")
    if not isinstance(task, Mapping) or not isinstance(task.get("physical_evaluator"), Mapping):
        raise ValueError("RR task requires the real physical evaluator")
    ev = task["physical_evaluator"]
    rr = (ev.get("current_legs") or {}).get("RR")
    if not isinstance(rr, Mapping):
        raise ValueError("RR current measured leg is missing")
    tick, now = ev.get("physics_tick"), ev.get("simulation_time_s")
    if type(tick) is not int or tick < 0 or not _finite(now) or now < 0:
        raise ValueError("RR evaluator requires its real observation clock")
    if frame is not None and (frame.physics_tick != tick
            or not math.isclose(frame.sim_time_s, now, rel_tol=0., abs_tol=1e-8)):
        raise ValueError("RR evaluator/frame clocks differ")
    final_key = "drive_target_full12" if "drive_target_full12" in info else "actual_drive_target_full12"
    final = _vector12(info.get(final_key))
    ack = info.get("atomic_ack")
    if ack is not None and (not isinstance(ack, Mapping)
            or _vector12(ack.get("drive_target_full12")) != final):
        raise ValueError("RR entry FINAL differs from committed ACK")
    reason = task.get("termination_reason") or ev.get("termination_reason")
    if reason is not None and (not isinstance(reason, str) or not reason):
        raise ValueError("RR physical terminal reason must be a nonempty string")
    history = ev.get("history") or {}
    crossed = (history.get("front_edge_crossed") or {}).get("RR") is True
    placed = (history.get("placed") or {}).get("RR") is True
    gap, outside = rr.get("clearance_m"), rr.get("top_xy_outside_distance_m")
    geometry_valid = _finite(gap) and _finite(outside) and outside >= 0
    live = ev.get("valid") is True and reason is None
    top = bool(live and geometry_valid and rr.get("ground_contact") is False
        and rr.get("air") is False and rr.get("top_contact") is True
        and rr.get("top_surface_contact") is True and rr.get("obstacle_pair_active") is True
        and rr.get("contact_surface") == "TOP" and rr.get("within_top_xy") is True
        and rr.get("within_lateral_span") is True)
    force = rr.get("bearing_force_n")
    bearing = bool(top and rr.get("support") is True and rr.get("bearing_verified") is True
        and _finite(force) and force >= force_noise_floor_n)
    free_air = bool(rr.get("air") is True and rr.get("ground_contact") is False
                    and rr.get("obstacle_pair_active") is False)
    eligible = bool(live and geometry_valid and crossed and free_air
        and rr.get("current_lift_valid") is True and rr.get("motion_continuation_allowed") is True
        and rr.get("within_top_xy") is True and rr.get("within_lateral_span") is True)
    raw = info.get("raw_observation")
    actual = raw.get("actual_full12") if isinstance(raw, Mapping) else getattr(raw, "actual_full12", None)
    # Numerical physical failure must remain reportable as a real terminal;
    # optional invalid measured q is unavailable, never fabricated zero.
    actual_rr = (list(_vector12(actual)[6:8]) if isinstance(actual, (tuple, list))
                 and len(actual) == 12 and all(map(_finite, actual)) else None)
    top_samples = rr.get("consecutive_top_samples")
    return dict(tick=tick, time_s=float(now), phase_id=task.get("stage_id"), live=live,
        termination_reason=reason, activation_eligible=eligible, geometry_valid=geometry_valid,
        crossed=crossed, placed=placed, free_air=free_air,
        ground_contact=rr.get("ground_contact") is True,
        within_top_xy=rr.get("within_top_xy") is True,
        within_lateral_span=rr.get("within_lateral_span") is True,
        gap_m=float(gap) if _finite(gap) else None,
        top_xy_outside_distance_m=float(outside) if _finite(outside) else None,
        current_top_contact=top, current_top_bearing=bearing,
        bearing_force_n=float(force) if _finite(force) else None,
        load_fraction=rr.get("load_fraction") if _finite(rr.get("load_fraction")) else None,
        load_fraction_valid=rr.get("load_fraction_valid") is True,
        consecutive_top_samples=top_samples if type(top_samples) is int and top_samples >= 0 else 0,
        final_rr_hip_deg=final[6], final_rr_knee_deg=final[7], actual_rr_hip_knee_deg=actual_rr,
        final_target_source=final_key, final_ack_verified=ack is not None,
        current_lift_valid=rr.get("current_lift_valid") is True,
        motion_continuation_allowed=rr.get("motion_continuation_allowed") is True)


class RRCaptureLocalTask:
    """One episode's public gate and measured continuous capture evidence.

    ``local_success`` is current evidence; the caller commits its true local
    terminal at the decision boundary. Continuing physics and losing support
    before that boundary withdraws success, but never the active policy gate.
    No deadline is added here. A caller's budget cutoff must not be passed as
    ``termination_reason``; a versioned genuine local deadline may be passed.
    """
    def __init__(self, config=None):
        self.config = RRCaptureLocalTaskConfig() if config is None else config
        if not isinstance(self.config, RRCaptureLocalTaskConfig):
            raise ValueError("validated RR local task config required")
        self.reset()

    def reset(self):
        self.active = False
        self.activation_tick = self.activation_time_s = None
        self.entry_rr_hip_deg = self.entry_rr_knee_deg = self.entry_gap_m = None
        self.hold_started_s = None
        self.hold_elapsed_s = 0.
        self.local_success = False
        self.metrics = None

    def observe(self, source):
        current = extract_rr_metrics(source, force_noise_floor_n=self.config.force_noise_floor_n)
        previous = self.metrics
        if previous is not None and current["tick"] == previous["tick"]:
            if current != previous:
                raise ValueError("conflicting RR evidence at the same observation tick")
            return self.snapshot()
        if previous is not None and (current["tick"] < previous["tick"]
                or current["time_s"] <= previous["time_s"]):
            raise ValueError("reset RR local task before resetting its physical episode")
        adjacent = bool(previous is not None and current["tick"] == previous["tick"] + 1
            and math.isclose(current["time_s"]-previous["time_s"], self.config.physics_dt_s,
                             rel_tol=0., abs_tol=1e-8))
        if not self.active and current["activation_eligible"]:
            self.active = True
            self.activation_tick, self.activation_time_s = current["tick"], current["time_s"]
            self.entry_rr_hip_deg = current["final_rr_hip_deg"]
            self.entry_rr_knee_deg = current["final_rr_knee_deg"]
            self.entry_gap_m = current["gap_m"]
        self.metrics = current
        if self.active and current["current_top_bearing"]:
            if (not adjacent or previous is None or not previous["current_top_bearing"]
                    or self.hold_started_s is None):
                self.hold_started_s = current["time_s"]
            self.hold_elapsed_s = current["time_s"]-self.hold_started_s
        else:
            self.hold_started_s, self.hold_elapsed_s = None, 0.
        self.local_success = bool(self.active and current["current_top_bearing"]
            and current["placed"] and current["crossed"]
            and current["consecutive_top_samples"] >= self.config.minimum_top_samples
            and self.hold_elapsed_s + 1e-10 >= self.config.hold_duration_s)
        return self.snapshot()

    def obs8(self):
        cfg, current = self.config, self.metrics
        active = float(self.active)
        return (active,
            min(1., (current["time_s"]-self.activation_time_s)/cfg.activation_age_scale_s) if self.active else 0.,
            self.entry_rr_hip_deg/cfg.entry_target_scale_deg if self.active else 0.,
            self.entry_rr_knee_deg/cfg.entry_target_scale_deg if self.active else 0.,
            self.entry_gap_m/cfg.entry_gap_scale_m if self.active else 0.,
            float(current["current_top_contact"]) if current else 0.,
            float(current["current_top_bearing"]) if current else 0.,
            min(1., self.hold_elapsed_s/cfg.hold_duration_s))

    def potential_components(self):
        current, cfg = self.metrics, self.config
        parts = dict(legal_xy=0., gap_closure=0., real_contact=0., bearing_hold=0.)
        if not (self.active and current["live"] and current["geometry_valid"]
                and not current["ground_contact"]):
            return parts
        parts["legal_xy"] = .2 / (1.+current["top_xy_outside_distance_m"]/cfg.xy_scale_m)
        legal_descent = (current["within_top_xy"] and current["within_lateral_span"]
            and (current["free_air"] or current["current_top_contact"])
            and current["gap_m"] >= cfg.minimum_legal_gap_m)
        if legal_descent:
            parts["gap_closure"] = .4 / (1.+max(0., current["gap_m"])/cfg.gap_scale_m)
        parts["real_contact"] = .1 * (current["current_top_contact"] + current["current_top_bearing"])
        parts["bearing_hold"] = .2 * min(1., self.hold_elapsed_s/cfg.hold_duration_s)
        return parts

    def snapshot(self):
        parts = self.potential_components()
        return dict(schema=SCHEMA, active=self.active, activation_tick=self.activation_tick,
            activation_time_s=self.activation_time_s, entry_rr_hip_deg=self.entry_rr_hip_deg,
            entry_rr_knee_deg=self.entry_rr_knee_deg, entry_gap_m=self.entry_gap_m,
            hold_elapsed_s=self.hold_elapsed_s, capture_hold_progress=self.obs8()[7],
            local_success=self.local_success, metrics=deepcopy(self.metrics),
            obs8=self.obs8(), potential=sum(parts.values()), potential_components=parts,
            full_task_success=False, physical_state_or_target_writes=0)

    def reward(self, before, *, termination_reason=None):
        """Return RR-only decision reward without mutating task or physics.

        ``before`` must be the snapshot at the actual action's decision start.
        A decision starting inactive remains excluded even if it opens the gate.
        Success and real failure use Phi(next)=0 and disallow bootstrap. A
        sampling cutoff with no physical/local terminal remains bootstrappable.
        """
        if not isinstance(before, Mapping) or before.get("schema") != SCHEMA:
            raise ValueError("RR reward requires a local-task decision-start snapshot")
        after = self.snapshot()
        previous, current = before.get("metrics"), after["metrics"]
        if (not isinstance(previous, Mapping) or current is None
                or current["tick"] <= previous["tick"] or current["time_s"] <= previous["time_s"]):
            raise ValueError("RR reward requires a positive real decision interval")
        if termination_reason is not None and (not isinstance(termination_reason, str) or not termination_reason):
            raise ValueError("RR terminal reason must be a nonempty string")
        reason = termination_reason or current["termination_reason"]
        if reason in ("SUCCESS", SUCCESS):
            raise ValueError("RR success must be established from current measured hold, not an external label")
        success = bool(self.local_success and reason is None)
        terminal = reason is not None or success
        credited = before.get("active") is True
        phi_before = before.get("potential")
        if not _finite(phi_before) or not 0. <= phi_before <= 1. + 1e-12:
            raise ValueError("invalid RR decision-start potential")
        phi_after = 0. if terminal else after["potential"]
        dt = current["time_s"]-previous["time_s"]
        shaping = self.config.potential_weight*(self.config.gamma*phi_after-phi_before)
        event = self.config.success_reward if success else -self.config.failure_cost if reason else 0.
        time_cost = self.config.time_cost_per_s*dt
        return dict(schema=SCHEMA, reward=shaping+event-time_cost if credited else 0.,
            on_policy_rr_sample=credited, prefix_excluded=not credited,
            potential_before=phi_before, potential_after=phi_after,
            potential_shaping=shaping if credited else 0., terminal_event=event if credited else 0.,
            time_cost=time_cost if credited else 0., elapsed_physics_s=dt,
            gamma=self.config.gamma, discount_convention="once_per_issued_policy_decision",
            terminated=terminal, truncated=False, terminal_bootstrap_allowed=not terminal,
            termination_reason=reason or (SUCCESS if success else None),
            rr_subtask_success=success, full_task_success=False,
            current_potential_components=after["potential_components"])
