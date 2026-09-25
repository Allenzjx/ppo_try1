"""UNAPPLIED candidate: preserve source tracking while deferring late events.

Only standard-library imports here. This is a source-event adapter, not an
actuator write, residual transform, recapture controller, or full-task route.
The original source carrier can still deliver explicit stops. Its one deferred
event remains pending until the enclosing RR-local episode has terminated;
there is deliberately no within-episode release or catch-up API.
"""
from __future__ import annotations

from dataclasses import replace
import math

MODE = "rr_local_defer_p09_late_and_new_p12_until_terminal_v2"
SOURCE_REVISIONS = (
    "rr_local_p09_five_channel_pending_with_independent_stop_carrier_v2",
    "rr_local_p12_pending_start_preserve_qualified_RL_AIR_and_started_clock_v2",
)
SERVO_NAMES = ("front_left_hip", "front_left_knee", "front_right_hip",
               "front_right_knee", "rear_left_hip", "rear_left_knee",
               "rear_right_hip", "rear_right_knee")
WHEEL_NAMES = ("front_left_ankle", "front_right_ankle", "rear_left_ankle", "rear_right_ankle")
DEFERRED_INDICES = (0, 1, 4, 5, 8)
DEFERRED_SERVOS = frozenset(SERVO_NAMES[i] for i in DEFERRED_INDICES if i < 8)
DEFERRED_NAMES = DEFERRED_SERVOS | {WHEEL_NAMES[0]}
SUPPORTED_CARRY_MODE = "current_free_lift_before_pending_knee_and_roll_v1"


def public_active(read_active):
    result = read_active()
    if type(result) is not bool:
        raise ValueError("the published RR-local active feature must be Boolean")
    return result


class DeferredLateCarrier:
    """Wrap the original executor immediately before its late group, once.

    ``pending_event_tick`` never advances. ``carrier_next_tick`` is separate,
    allowing later original all-wheel stops; no motor/history APIs are used.
    The four source servo goals and FL wheel pulse of the pending event are
    never acquired. Existing pre-late logical values are not measured q, not
    FINAL, and not a new mapped-nominal hold.
    """
    def __init__(self, motion, *, previous_sample, pending_event_tick):
        phase = motion.phase
        if phase.state_id != "P09" or motion._tick_index != pending_event_tick:
            raise ValueError("deferred source requires the unconsumed P09 late event")
        if previous_sample is None or previous_sample.tick_index != pending_event_tick - 1:
            raise ValueError("deferred source requires its immediately preceding source sample")
        waypoints = [w for w in phase.waypoints
                     if motion._scaled_source_tick(w.time_s) == pending_event_tick]
        groups = [g for g in phase.atomic_groups
                  if motion._scaled_source_tick(g.time_s) == pending_event_tick]
        if (len(waypoints) != 1 or len(groups) != 1
                or set(waypoints[0].changed_channels) != DEFERRED_NAMES
                or groups[0].source_full12_atomic is not True
                or tuple(previous_sample.full12[8:]) != (0.,) * 4):
            raise ValueError("expected exact five-channel late group after authored wheel stop")
        # Deliberately bounded to this source suffix: later groups must be
        # wheel-only explicit stops. Never silently discard another new action.
        for waypoint in phase.waypoints:
            if motion._scaled_source_tick(waypoint.time_s) <= pending_event_tick:
                continue
            changed = set(waypoint.changed_channels)
            if not changed.issubset(WHEEL_NAMES) or any(waypoint.full12[8:]):
                raise ValueError("new post-late action needs a separately reviewed lane")
        for group in phase.atomic_groups:
            if motion._scaled_source_tick(group.time_s) > pending_event_tick:
                if not set(group.channels).issubset(WHEEL_NAMES):
                    raise ValueError("cannot split a later mixed source group")
        self.original = motion
        self.pending_event_tick = pending_event_tick
        self.pending_event = waypoints[0]
        self.pending_atomic_group = groups[0]
        self.before = tuple(previous_sample.full12)
        self.before_tracking = tuple(previous_sample.tracking_servo_names)
        self.last = tuple(previous_sample.full12)
        self.last_carrier_tick = pending_event_tick - 1
        self.last_emitted_groups = ()
        self.stop_event_ticks = []
        self._source_atomic_emitted = motion.source_atomic_emitted

    def __getattr__(self, key):
        return getattr(self.original, key)

    @property
    def source_atomic_emitted(self):
        return self._source_atomic_emitted

    def tick(self):
        sample = self.original.tick()
        if sample.tick_index != self.last_carrier_tick + 1:
            raise ValueError("source carrier skipped or repeated a tick")
        self.last_carrier_tick = sample.tick_index
        values, nominal = list(sample.full12), list(sample.nominal_full12)
        for index in DEFERRED_INDICES:
            values[index] = self.before[index]
            nominal[index] = self.before[index]
        groups = tuple(g for g in sample.atomic_groups if g != self.pending_atomic_group)
        # Every allowed later source event is a verified explicit wheel stop.
        # Preserve its original channel ownership and single atomic dispatch.
        for group in groups:
            if not set(group.channels).issubset(WHEEL_NAMES) or any(sample.full12[8:]):
                raise ValueError("unexpected non-stop group in deferred source suffix")
            for name in group.channels:
                index = 8 + WHEEL_NAMES.index(name)
                values[index] = sample.full12[index]
                nominal[index] = sample.nominal_full12[index]
            self.stop_event_ticks.append(sample.tick_index)
        result = replace(sample, full12=tuple(values), nominal_full12=tuple(nominal),
            atomic_groups=groups,
            # The late atomic sample is NOT consumed. It cannot acquire FR/RR
            # tracking merely because their logical goal values are unchanged.
            # Keep this layer's real pre-late responsibility; wheel-only stops
            # do not create servo tracking. Other layers remain untouched.
            tracking_servo_names=self.before_tracking,
            endpoint_issued=False,
            target_progressed=any(abs(a-b) > 1e-12 for a, b in zip(values, self.last)))
        self.last, self.last_emitted_groups = result.full12, result.atomic_groups
        self._source_atomic_emitted += sum(g.source_full12_atomic for g in groups)
        return result

    def receipt(self):
        return dict(mode=MODE, pending_event_tick=self.pending_event_tick,
            source_control_revisions=list(SOURCE_REVISIONS),
            pending_changed_channels=list(self.pending_event.changed_channels),
            pending_event_consumed=False, pending_event_discarded=False,
            carrier_next_tick=self.original._tick_index,
            carrier_proposed_atomic_groups=self.original.source_atomic_emitted,
            actually_emitted_atomic_groups=self.source_atomic_emitted,
            source_stop_event_ticks=list(self.stop_event_ticks),
            deferred_source_values=[self.before[i] for i in DEFERRED_INDICES],
            reference_kind="pre_late_original_logical_source_not_FINAL_or_actual_q",
            deferred_source_tracking_servo_names=list(self.before_tracking),
            deferred_tracking_source="previous_sample_before_unconsumed_late",
            policy_channels_restricted=[], mapper_or_history_resets=0,
            extra_actuator_writes=0, within_episode_release_supported=False)


def provider_type(base):
    """Factory seam keeps this adapter testable without importing YAML/Isaac."""
    class LocalDeferredProvider(base):
        def __init__(self, *args, read_local_active, **kwargs):
            self._read_local_active = read_local_active
            super().__init__(*args, **kwargs)
            if (self._p09_late_source is None
                    or self._rr_carry_source_mode not in (None, SUPPORTED_CARRY_MODE)):
                raise ValueError("deferred source requires the selected pending-late and supported carry source")
            if self._rr_carry_source_mode is not None:
                # The accepted controller already uses this readiness mode.
                # Keep its earlier knee/roll events and their original base
                # permission checks; only the later five-channel event differs.
                knee_times = self._rr_carry_knee_source_times
                carry_times = (*knee_times, self._rr_carry_roll_source_time)
                if not knee_times or any(type(t) not in (int, float)
                        or not math.isfinite(t) or not 0. <= t < self._p09_late_source[1]
                        for t in carry_times):
                    raise ValueError("pending carry knee and roll events must precede the deferred late event")

        def _sequence_permission(self, layer, task, observation):
            active = public_active(self._read_local_active)
            if active and layer["stage"] == "P12" and layer["ticks"] == 0:
                ev = task.get("physical_evaluator", {})
                if (ev.get("valid") is True and task.get("termination_reason") is None
                        and ev.get("termination_reason") is None):
                    # Use the existing CURRENT ground-revoked RL AIR test,
                    # never a weaker local replica or history. This intercepts
                    # only a new lane; once started, base wheel-stop and RL
                    # joint-pause behavior remains entirely unchanged.
                    from wlr50_clean.ppo.semantic_rear_policy_timing import rear_dependency
                    dep = rear_dependency(task, self.spec["support"],
                                          mode=self._rear_policy_timing_mode)
                    if not dep["rl_current_swing"]:
                        layer["rl_dependency_wait"] = True
                        layer["sequence_diagnostic"] = dict(dep,
                            status="local_RR_capture_holding_new_P12_start",
                            wait_reason="active_local_capture_before_new_RL_unload",
                            local_active=True, local_hold_does_not_release_source_event=True,
                            observation_tick=ev.get("physics_tick"),
                            wheel_source_clock_continues_after_start=False,
                            source_control_revisions=list(SOURCE_REVISIONS))
                        return False
            if layer["stage"] != "P09" or not active:
                if isinstance(layer["motion"], DeferredLateCarrier):
                    raise ValueError("active cannot unlatch inside a deferred source episode")
                return super()._sequence_permission(layer, task, observation)
            ev = task.get("physical_evaluator", {})
            if (ev.get("valid") is not True or task.get("termination_reason") is not None
                    or ev.get("termination_reason") is not None):
                return super()._sequence_permission(layer, task, observation)
            pending_tick = layer["motion"]._scaled_source_tick(self._p09_late_source[1])
            if layer["ticks"] < pending_tick:
                return super()._sequence_permission(layer, task, observation)
            if not isinstance(layer["motion"], DeferredLateCarrier):
                if layer["ticks"] != pending_tick or "late_group_start_tick" in layer.get("sequence_diagnostic", {}):
                    raise ValueError("cannot retrofit this guard after the late event has been issued")
                layer["motion"] = DeferredLateCarrier(layer["motion"],
                    previous_sample=layer.get("sample"), pending_event_tick=pending_tick)
            diag = layer.setdefault("sequence_diagnostic", {})
            diag.update(status="local_RR_capture_late_event_deferred_carrier_active",
                wait_reason="active_local_capture_before_committed_terminal",
                local_active=True, local_hold_does_not_release_source_event=True,
                observation_tick=ev.get("physics_tick"),
                deferred_late_source=layer["motion"].receipt())
            # Advance only the sanitized carrier, not the pending late event.
            return True

        def _rr_waiting_late_group(self, layer):
            if isinstance(layer["motion"], DeferredLateCarrier):
                # Preserve existing measured RR carry eligibility. The base
                # ordered_wheel_owner rule still recognizes a fresh stop group
                # and prevents carry from overwriting that stop on its tick.
                return True
            return super()._rr_waiting_late_group(layer)

        @property
        def nominal_suggestion_diagnostics(self):
            result = dict(super().nominal_suggestion_diagnostics)
            result["rr_local_deferred_late_v2"] = dict(
                mode=MODE, local_active=public_active(self._read_local_active),
                source_control_revisions=list(SOURCE_REVISIONS),
                public_permission_expression="local.active; P12 pending only, except existing rear_dependency.rl_current_swing",
                p12_pending_start=[dict(layer.get("sequence_diagnostic", {}))
                                   for layer in self._continuous_layers if layer["stage"] == "P12"],
                layers=[layer["motion"].receipt() for layer in self._continuous_layers
                        if isinstance(layer["motion"], DeferredLateCarrier)])
            return result
    return LocalDeferredProvider


def controller_factory(*, task_spec_path, read_local_active):
    """Use the existing backend controller_factory argument before first reset.

    Imported only by the physical route; tests use
    the pure executor and factory seam, never this runtime import.
    """
    from pathlib import Path
    from .semantic_supervisor import (NominalMotionProvider, SemanticControllerAdapter,
        TaskStageSupervisor, load_fsm_spec, load_motion_contract)
    cls = provider_type(NominalMotionProvider)
    def build(fsm_path, motion_contract_path):
        spec = load_fsm_spec(Path(fsm_path))
        contract = load_motion_contract(Path(motion_contract_path))
        supervisor = TaskStageSupervisor(task_spec_path)
        provider = cls(contract, spec=supervisor.spec, fsm_spec=spec,
                       read_local_active=read_local_active)
        return SemanticControllerAdapter(spec, contract, supervisor=supervisor,
                                         nominal_provider=provider)
    return build
