"""Opt-in measured post-RL advance/home task; no simulator or target writes.

The evaluator owns the physical task clock. The wrapper exposes that clock and
the adjacent committed command references to actor, critic and public source
control; it never creates contact, placement history or a success label.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
import math

from .semantic_rr_capture_local_task import _finite, _vector12

SCHEMA = "wlr50_clean.finish_advance_task.v1"
MODE = "measured_post_RL_advance_then_actual_home_v1"
ADVANCE, HOME, SETTLE, DONE = (
    "ADVANCE_FOR_HOME_CLEARANCE", "HOME_RECOVERY", "FINAL_SETTLE", "DONE")
FINISH_FIELDS = (
    "finish_active",
    *(f"finish_request_ref_ratio_{i}" for i in range(12)),
    *(f"finish_reference_final_norm_{i}" for i in range(12)),
    "finish_advancing", "finish_homing", "finish_settling", "finish_done",
    "finish_advance_progress_norm", "finish_advance_goal_norm", "finish_advance_remaining_norm",
    "finish_home_permitted", "finish_home_elapsed_norm", "finish_home_duration_norm", "finish_home_fraction",
    "finish_home_actual_error_norm", "finish_settle_clean_elapsed_norm", "finish_remaining_global_norm",
    "finish_rear_clearance_norm", "finish_far_edge_clearance_norm",
)
FINISH_OBSERVATION_DIM = len(FINISH_FIELDS)
assert FINISH_OBSERVATION_DIM == 41


def _get(obj, key, default=None):
    return obj.get(key, default) if isinstance(obj, Mapping) else getattr(obj, key, default)


def _vec(value, size=3):
    if not isinstance(value, (tuple, list)) or len(value) != size or not all(map(_finite, value)):
        raise ValueError(f"finish requires finite vector{size}")
    return tuple(map(float, value))


def _clip(value):
    return max(0., min(1., float(value)))


@dataclass(frozen=True)
class FinishAdvanceConfig:
    advance_distance_m: float = .20
    forward_axis_world: tuple = (1., 0., 0.)
    wheel_speed_rad_s: float = .30
    rear_clearance_m: float = .05
    far_edge_clearance_m: float = .10
    home_target_servo_deg: tuple = (.5, -.7, 3.7, .4, -2.6, -3.9, -.5, -6.)
    home_rate_deg_s: float = 20.
    home_minimum_duration_s: float = 2.
    wheel_rate_rad_s2: float = 1.8

    def __post_init__(self):
        positive = (self.advance_distance_m, self.wheel_speed_rad_s, self.rear_clearance_m,
                    self.far_edge_clearance_m, self.home_rate_deg_s,
                    self.home_minimum_duration_s, self.wheel_rate_rad_s2)
        if not all(_finite(x) and x > 0. for x in positive):
            raise ValueError("finish distances/rates must be positive and finite")
        if (self.advance_distance_m > 1. or self.wheel_speed_rad_s > .5
                or self.home_rate_deg_s > 150. or self.wheel_rate_rad_s2 > 1.8):
            raise ValueError("finish configuration exceeds declared existing control bounds")
        # This robot's verified obstacle planes are world-X aligned. Do not
        # invent an orientation or integrate body yaw into the finish axis.
        if _vec(self.forward_axis_world) != (1., 0., 0.):
            raise ValueError("finish axis must match the verified obstacle forward +X")
        target = _vec(self.home_target_servo_deg, 8)
        if any(abs(x) > 8. for x in target):
            raise ValueError("source home target must remain inside unchanged actual home tolerance")


def load_finish_advance_config(value):
    if isinstance(value, FinishAdvanceConfig):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("explicit finish advance configuration required")
    values = dict(value)
    if values.pop("enabled", True) is not True:
        raise ValueError("use None, not a disabled object, for legacy finish behavior")
    if values.pop("mode", MODE) != MODE:
        raise ValueError("unknown requested finish mode")
    return FinishAdvanceConfig(**values)


def home_duration(config, final_servo):
    """Quintic peak derivative is1.875; use actual adjacent command travel."""
    start = _vec(final_servo, 8)
    return max(config.home_minimum_duration_s,
        1.875*max(abs(a-b) for a, b in zip(start, config.home_target_servo_deg))/config.home_rate_deg_s)


class FinishAdvanceTracker:
    """Single measured clock shared by live and independent physical evaluator."""
    def __init__(self, config):
        self.config = load_finish_advance_config(config)
        self.active = False
        self.mode = None
        self.origin = self.activation_tick = self.activation_time_s = None
        self.home_tick = self.home_time_s = self.home_duration_s = None
        self.clean_since = None
        self.clean_elapsed_s = 0.
        self.clean_restarts = 0
        self.forward_completed = self.home_completed = self.settle_completed = False
        self.advance_progress_m = 0.
        self.rear_clearance = self.far_clearance = None
        self.geometry_valid = False
        self.home_error = 0.
        self.home_start_error = None
        self.now = 0.
        self.tick = None
        self.failure = None
        self.current_support_count = 0
        self.current_rl_contact = False
        self.last_reference_body_position = None

    def observe(self, *, tick, now, base_position, current, history, body_bounds,
                front, back, final_servo, home_error, support, region, evidence_ok,
                measured_controlled, commands_controlled, failure, final, episode_limit):
        self.now, self.tick, self.failure = now, tick, failure
        self.home_error = float(home_error)
        self.current_support_count = sum(bool(v.get("support") and v.get("top_surface_contact"))
                                         for v in current.values())
        rl = current["RL"]
        self.current_rl_contact = bool(rl.get("top_contact") and rl.get("top_surface_contact")
            and rl.get("bearing_verified") and not rl.get("air") and rl.get("bearing_force_n", 0.) > 0.)
        qualified = bool(failure is None and all(history["placed"].values())
            and history["front_edge_crossed"].get("RL") and self.current_rl_contact)
        if not self.active and qualified:
            self.active, self.mode = True, ADVANCE
            self.origin = _vec(base_position)
            self.activation_tick, self.activation_time_s = tick, now
        if not self.active:
            return self.snapshot(final, episode_limit)
        self.last_reference_body_position = _vec(base_position)
        self.advance_progress_m = sum((a-b)*d for a, b, d in
            zip(self.last_reference_body_position, self.origin, self.config.forward_axis_world))
        try:
            rear = [_vec(_get(body_bounds[name], "minimum_m"))[0]
                    for name in ("rear_left_wheel", "rear_right_wheel")]
            highs = [_vec(_get(body_bounds[name], "maximum_m"))[0] for name in
                     ("base_link", "front_left_wheel", "front_right_wheel", "rear_left_wheel", "rear_right_wheel")]
            self.rear_clearance, self.far_clearance = min(rear)-front, back-max(highs)
            self.geometry_valid = True
        except (KeyError, TypeError, ValueError):
            self.geometry_valid = False
            self.rear_clearance = self.far_clearance = None
        geometry_ready = bool(self.geometry_valid and self.rear_clearance >= self.config.rear_clearance_m
            and self.far_clearance >= self.config.far_edge_clearance_m)
        if self.mode == ADVANCE and failure is None and evidence_ok and support and region and geometry_ready:
            if self.advance_progress_m + 1.e-12 >= self.config.advance_distance_m:
                self.forward_completed = True
                self.mode = HOME
                self.home_tick, self.home_time_s = tick, now
                self.home_duration_s = home_duration(self.config, final_servo)
                self.home_start_error = self.home_error
        home_now = bool(self.forward_completed and self.home_error <= final["home_tolerance_deg"])
        if self.mode in (HOME, SETTLE) and home_now:
            self.home_completed = True
            self.mode = SETTLE
        clean = bool(self.mode in (SETTLE, DONE) and home_now and geometry_ready
            and region and evidence_ok and support and measured_controlled and commands_controlled
            and failure is None and now < episode_limit)
        if clean:
            if self.clean_since is None:
                self.clean_since = now
            self.clean_elapsed_s = max(0., now-self.clean_since)
        else:
            self.clean_restarts += int(self.clean_since is not None)
            self.clean_since, self.clean_elapsed_s = None, 0.
        self.settle_completed = bool(clean and self.clean_elapsed_s+1.e-12 >= final["post_completion_observation_s"])
        if self.settle_completed:
            self.mode = DONE
        return self.snapshot(final, episode_limit)

    def snapshot(self, final, episode_limit):
        elapsed = max(0., self.now-self.home_time_s) if self.home_time_s is not None else 0.
        return dict(schema=SCHEMA, mode_version=MODE, active=self.active, mode=self.mode,
            activation_tick=self.activation_tick, activation_time_s=self.activation_time_s,
            observation_tick=self.tick, observation_time_s=self.now,
            origin_body_position_w_m=self.origin, reference_body="base_link.position_w_m",
            forward_axis_world=self.config.forward_axis_world,
            current_body_position_w_m=self.last_reference_body_position,
            advance_progress_m=self.advance_progress_m, advance_goal_m=self.config.advance_distance_m,
            advance_remaining_m=max(0., self.config.advance_distance_m-self.advance_progress_m),
            rear_clearance_m=self.rear_clearance, far_edge_clearance_m=self.far_clearance,
            geometry_valid=self.geometry_valid, home_permitted=self.forward_completed,
            home_entry_tick=self.home_tick, home_entry_time_s=self.home_time_s,
            home_elapsed_s=elapsed, home_duration_s=self.home_duration_s,
            home_fraction=_clip(elapsed/self.home_duration_s) if self.home_duration_s else 0.,
            home_maximum_servo_error_deg=self.home_error, home_start_error_deg=self.home_start_error,
            actual_home_tolerance_deg=final["home_tolerance_deg"],
            current_home_within_tolerance=bool(self.active and self.forward_completed
                                             and self.home_error <= final["home_tolerance_deg"]),
            clean_since_s=self.clean_since, clean_elapsed_s=self.clean_elapsed_s,
            clean_required_s=final["post_completion_observation_s"], clean_restarts=self.clean_restarts,
            remaining_global_s=max(0., episode_limit-self.now), current_rl_top_contact=self.current_rl_contact,
            current_top_support_count=self.current_support_count,
            traversal_contacts_complete=self.active, post_RL_forward_completed=self.forward_completed,
            home_recovery_completed=self.home_completed, controlled_settle_completed=self.settle_completed,
            requested_finish_completed=bool(self.settle_completed and self.failure is None),
            termination_reason=self.failure, config=asdict(self.config), physical_state_writes=0)


class FinishAdvanceTask:
    """Preserve accepted PostRR task, append explicit control-reference memory."""
    def __init__(self, config=None, *, accepted_task, phase_caps_full12):
        self.finish_config = load_finish_advance_config(config or FinishAdvanceConfig())
        self.accepted = accepted_task
        self.phase_caps_full12 = deepcopy(phase_caps_full12)
        for caps in self.phase_caps_full12.values():
            if any(x <= 0. for x in _vector12(caps)):
                raise ValueError("finish requires positive current phase capacities")
        self.reset()

    def __getattr__(self, name):
        return getattr(self.accepted, name)

    def reset(self):
        self.accepted.reset()
        self.finish_active = False
        self.finish_state = None
        self.reference_final = self.reference_request = self.reference_caps = None
        self.reference_tick = self.reference_command_tick = self.reference_phase = None
        self._reference_mode = self._last_finish_tick = None
        self.reference_history = []

    def _capture(self, info, state):
        ack = info.get("atomic_ack")
        if not isinstance(ack, Mapping) or type(ack.get("physics_tick")) is not int:
            raise ValueError("finish reference requires adjacent committed atomic ACK")
        audit = info.get("actuator_target_effect_audit") or {}
        if audit.get("physics_tick") != ack["physics_tick"]:
            raise ValueError("finish ACK and action audit clocks differ")
        final = _vector12(ack.get("drive_target_full12"))
        if final != _vector12(info.get("drive_target_full12", info.get("actual_drive_target_full12"))):
            raise ValueError("finish FINAL does not match committed ACK")
        request = _vector12(ack.get("independent_policy_residual_requested_full12"))
        phase = audit.get("policy_request_phase") or audit.get("source_phase_id") or info["semantic_task"]["stage_id"]
        caps = _vector12(self.phase_caps_full12[phase])
        if any(not -1. < a/b < 1. for a, b in zip(request, caps)):
            raise ValueError("finish REQUEST inverse saturated; no hidden inverse clamping")
        self.reference_final, self.reference_request, self.reference_caps = final, request, caps
        self.reference_tick, self.reference_command_tick, self.reference_phase = state["observation_tick"], ack["physics_tick"], phase
        self._reference_mode = state["mode"]
        if state["home_permitted"]:
            expected = home_duration(self.finish_config, final[:8])
            if not math.isclose(expected, state["home_duration_s"], rel_tol=0., abs_tol=1.e-10):
                raise ValueError("home duration physical command and adjacent ACK disagree")
        self.reference_history.append(dict(mode=state["mode"], observation_tick=self.reference_tick,
            command_tick=self.reference_command_tick, phase=phase, final_full12=final, request_full12=request,
            capacities_full12=caps, source="adjacent_committed_atomic_ACK", HISTORY_reset=False))

    def observe(self, source):
        self.accepted.observe(source)
        source = source.frame if hasattr(source, "frame") else source
        info = source if isinstance(source, Mapping) else source.info
        ev = info["semantic_task"]["physical_evaluator"]
        state = ev.get("requested_finish")
        if not isinstance(state, Mapping) or state.get("mode_version") != MODE:
            raise ValueError("finish task requires matching common physical evaluator")
        if state["observation_tick"] == self._last_finish_tick:
            if self.finish_state != state:
                raise ValueError("conflicting requested finish at same native tick")
            return self.snapshot()
        self._last_finish_tick = state["observation_tick"]
        self.finish_state = deepcopy(state)
        if state["active"]:
            if not self.finish_active or (state["home_permitted"] and self._reference_mode == ADVANCE):
                self._capture(info, state)
            self.finish_active = True
        return self.snapshot()

    def finish_observation(self):
        if not self.finish_active:
            return (0.,)*FINISH_OBSERVATION_DIM
        s = self.finish_state
        values = [1., *[x/y for x, y in zip(self.reference_request, self.reference_caps)],
            *[x/(90. if i < 8 else 5.) for i, x in enumerate(self.reference_final)],
            *[float(s["mode"] == name) for name in (ADVANCE, HOME, SETTLE, DONE)],
            s["advance_progress_m"], s["advance_goal_m"], s["advance_remaining_m"],
            float(s["home_permitted"]), s["home_elapsed_s"]/10., (s["home_duration_s"] or 0.)/10., s["home_fraction"],
            s["home_maximum_servo_error_deg"]/90., s["clean_elapsed_s"]/s["clean_required_s"],
            s["remaining_global_s"]/200., s["rear_clearance_m"] or 0., s["far_edge_clearance_m"] or 0.]
        if not all(map(_finite, values)) or len(values) != FINISH_OBSERVATION_DIM:
            raise ValueError("invalid finite finish observation")
        return tuple(values)

    def potential_components(self):
        if not self.finish_active:
            return dict(advance=0., home=0., settle=0.)
        s = self.finish_state
        advance = .35*_clip(s["advance_progress_m"]/s["advance_goal_m"])
        if s["post_RL_forward_completed"]:
            advance = .35
        start = max(8., s["home_start_error_deg"] or 8.)
        home = .4*_clip((start-s["home_maximum_servo_error_deg"])/(start-8. or 1.)) if s["home_permitted"] else 0.
        if s["current_home_within_tolerance"]:
            home = .4
        return dict(advance=advance, home=home, settle=.25*_clip(s["clean_elapsed_s"]/s["clean_required_s"]))

    def snapshot(self):
        old = self.accepted.snapshot()
        context = None
        if self.finish_active:
            context = dict(self.finish_state, schema="wlr50_clean.requested_finish_context.v1",
                reference_final_full12=self.reference_final,
                reference_request_full12=self.reference_request, reference_capacities_full12=self.reference_caps,
                reference_tick=self.reference_tick, reference_command_tick=self.reference_command_tick,
                reference_phase=self.reference_phase, reference_mode=self._reference_mode,
                home_target_servo_deg=self.finish_config.home_target_servo_deg,
                advance_wheel_rad_s=self.finish_config.wheel_speed_rad_s,
                wheel_rate_rad_s2=self.finish_config.wheel_rate_rad_s2)
        parts = self.potential_components()
        old.update(finish_task_schema=SCHEMA, finish_active=self.finish_active,
            requested_finish=deepcopy(self.finish_state), finish_context=context,
            finish_obs=self.finish_observation(), finish_potential=sum(parts.values()),
            finish_potential_components=parts, finish_reference_history=deepcopy(self.reference_history))
        return old

    def read_local_state(self):
        return self.snapshot()

    def reward(self, before, *, termination_reason=None):
        legacy = self.accepted.reward(before, termination_reason=termination_reason)
        credited = before.get("finish_active") is True
        previous = before.get("finish_potential", 0.)
        current = 0. if legacy["terminated"] else sum(self.potential_components().values())
        cfg = self.accepted.config
        shaping = cfg.potential_weight*(cfg.gamma*current-previous)
        cost = cfg.time_cost_per_s*legacy["elapsed_physics_s"]
        result = dict(legacy, finish_task_schema=SCHEMA,
            reward=shaping+legacy["terminal_event"]-cost if credited else 0.,
            on_policy_finish_sample=credited, on_policy_rr_sample=False,
            on_policy_post_rr_sample=False, on_policy_continuation_sample=credited, prefix_excluded=not credited,
            potential_before=previous, potential_after=current, potential_shaping=shaping if credited else 0.,
            milestone_events={}, terminal_event=legacy["terminal_event"] if credited else 0.,
            time_cost=cost if credited else 0., current_potential_components=self.potential_components(),
            accepted_prefix_reward_diagnostic=legacy["reward"], internal_state_change_is_terminal=False)
        return result
