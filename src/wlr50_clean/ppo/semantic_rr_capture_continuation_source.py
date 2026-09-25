"""Continuous RR-capture / RL-preparation source scheduling (stdlib only).

This is an opt-in successor to the retained RR-local v2 carrier. It changes
source ownership, not physical contact, action likelihood, mapper or HISTORY.
P09's late full12 group is deliberately split: RL knee is preparation; FL
hip/knee + RL hip + FL wheel are load-dependent transfer. All physical writes
remain the existing single full12 dispatch. The pending source clock resumes
where it stopped. Once a wheel pulse starts, its authored stop cannot pause.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import math

from wlr50_clean.ppo.semantic_rear_policy_timing import rear_dependency, verified_bearing
from wlr50_clean.ppo.semantic_rr_capture_deferred_late import (
    DeferredLateCarrier, SERVO_NAMES, WHEEL_NAMES, SUPPORTED_CARRY_MODE,
)

MODE = "rr_touch_neartop_continuous_source_v3"
PREP_INDEX = 5
STRONG_SERVO_INDICES = (0, 1, 4)
STRONG_NAMES = tuple(SERVO_NAMES[i] for i in STRONG_SERVO_INDICES) + (WHEEL_NAMES[0],)
PUBLIC_FIELDS = ("active", "rr_touch_seen", "rr_support_continuation_valid",
                 "rr_geometry_ready_for_RL_prep")


def local_state(read_local_state):
    result = read_local_state()
    if not isinstance(result, Mapping):
        raise ValueError("continuous source requires a public local task snapshot")
    if any(type(result.get(key)) is not bool for key in PUBLIC_FIELDS):
        raise ValueError("continuous source task flags must be explicit Booleans")
    return dict(result)


def continuation_permission(state, evaluation):
    """Read adjacent committed task memory, recheck current physical geometry.

The local task observer commits after the semantic supervisor's same physical
frame. Thus its memory can be current or exactly one native tick old here;
anything older fails closed. This never substitutes old force for current force.
    """
    tick = evaluation.get("physics_tick")
    observed = state.get("observation_tick")
    if observed is None:
        observed = (state.get("metrics") or {}).get("tick")
    live = (evaluation.get("valid") is True
            and evaluation.get("termination_reason") is None)
    rr = evaluation.get("current_legs", {}).get("RR", {})
    fresh = (type(tick) is int and type(observed) is int and 0 <= tick-observed <= 1)
    current_region = (rr.get("within_top_xy") is True
        and rr.get("within_lateral_span") is True and rr.get("ground_contact") is False)
    ready = bool(state["active"] and live and fresh and current_region
        and (state["rr_geometry_ready_for_RL_prep"]
             or state["rr_support_continuation_valid"]))
    return dict(ready=ready, live=bool(live), fresh=bool(fresh),
                local_state_observation_tick=observed, physical_observation_tick=tick,
                current_top_region=bool(current_region),
                physical_support_awarded=False)


def lane_permissions(state, task, support, *, rear_mode):
    ev = task.get("physical_evaluator", {})
    permission = continuation_permission(state, ev)
    dep = rear_dependency(task, support, mode=rear_mode)
    legs = ev.get("current_legs", {})
    bearing = lambda leg: verified_bearing(legs.get(leg, {}), support)
    # A measured two-contact transition is allowed. This is permission for a
    # bounded authored knee preparation, not proof of stability or RL unload.
    rl_still_supported = bearing("RL")
    another_real_support = any(bearing(leg) for leg in ("FL", "FR", "RR"))
    live = permission["live"] and task.get("termination_reason") is None
    prep = bool(live and permission["ready"] and (
        (rl_still_supported and another_real_support)
        or dep["support_transfer_permitted"] or dep["rl_current_swing"]))
    strong = bool(live and dep["support_transfer_permitted"])
    swing = bool(live and dep["rl_current_swing"])
    return dict(permission, **dep, preparation_permitted=prep,
        strong_transfer_permitted=strong, rl_lane_permitted=strong or swing,
        rl_still_supported=bool(rl_still_supported),
        another_measured_support=bool(another_real_support))


class ContinuousLateCarrier:
    """One pending P09 event with an independently permitted preparation lane.

No source tick is consumed before load-dependent release. During the wait,
the prior all-wheel stop stays owned; there is no pulse needing a future stop.
After release the original source clock always advances through its stop even
if bearing is lost. Paused servo requests hold current *logical nominal*, not
old entry or FINAL. This does NOT by itself stop a mapper still chasing a
previously issued distant goal: an independently declared dispatch projection
is required for that guarantee. The residual branch remains active throughout.
    """
    def __init__(self, motion, *, previous_sample, pending_event_tick, read_nominal):
        checked = DeferredLateCarrier(motion, previous_sample=previous_sample,
                                      pending_event_tick=pending_event_tick)
        self.original = motion
        self.pending_event_tick = pending_event_tick
        self.pending_event = checked.pending_event
        self.pending_atomic_group = checked.pending_atomic_group
        self.before = tuple(previous_sample.full12)
        self.before_tracking = tuple(previous_sample.tracking_servo_names)
        self.previous = previous_sample
        self._read_nominal = read_nominal
        self.preparation_permitted = False
        self.strong_permitted = False
        self.preparation_issued = False
        self.strong_started = False
        self.wait_ticks = 0
        self.source_stop_ticks = []
        self._source_atomic_emitted = motion.source_atomic_emitted
        self._receipt_count = 0

    def __getattr__(self, name):
        return getattr(self.original, name)

    @property
    def source_atomic_emitted(self):
        return self._source_atomic_emitted

    def permissions(self, *, preparation, strong):
        if type(preparation) is not bool or type(strong) is not bool:
            raise ValueError("source lane permissions must be Boolean")
        self.preparation_permitted, self.strong_permitted = preparation, strong

    def _hold(self):
        values = tuple(self._read_nominal())
        if len(values) != 12 or any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
            raise ValueError("continuous source requires finite current nominal full12")
        return values

    def tick(self):
        held = self._hold()
        groups = ()
        if not self.strong_started and not self.strong_permitted:
            self.wait_ticks += 1
            values = list(self.before)
            values[PREP_INDEX] = (self.original._apply_correction(
                self.original.phase, self.pending_event.full12)[PREP_INDEX]
                if self.preparation_permitted else held[PREP_INDEX] if self.preparation_issued
                else self.before[PREP_INDEX])
            if self.preparation_permitted and not self.preparation_issued:
                groups = (replace(self.pending_atomic_group,
                    channels=(SERVO_NAMES[PREP_INDEX],), source_full12_atomic=False),)
                self.preparation_issued = True
            tracking = set(self.before_tracking)
            if self.preparation_permitted:
                tracking.add(SERVO_NAMES[PREP_INDEX])
            else:
                tracking.discard(SERVO_NAMES[PREP_INDEX])
            sample = replace(self.previous, tick_index=self.pending_event_tick,
                elapsed_s=self.pending_event_tick*self.original.dt_s,
                full12=tuple(values), nominal_full12=tuple(values),
                atomic_groups=groups, tracking_servo_names=tuple(n for n in SERVO_NAMES if n in tracking),
                endpoint_issued=False,
                target_progressed=any(abs(a-b)>1e-12 for a,b in zip(values,self.previous.full12)))
        else:
            first = not self.strong_started
            self.strong_started = True
            sample = self.original.tick()
            values, nominal = list(sample.full12), list(sample.nominal_full12)
            groups = sample.atomic_groups
            if first:
                # A new split-source contract, not the old full12 reference
                # event. Unchanged FR/RR cannot silently acquire tracking.
                channels = STRONG_NAMES + (() if self.preparation_issued else (SERVO_NAMES[PREP_INDEX],))
                groups = tuple(replace(g, channels=channels, source_full12_atomic=False)
                    if g == self.pending_atomic_group else g for g in groups)
                self.preparation_issued = True
            for group in groups:
                if set(group.channels).issubset(WHEEL_NAMES):
                    if any(sample.full12[8:]):
                        raise ValueError("unexpected non-stop after split P09 late event")
                    self.source_stop_ticks.append(sample.tick_index)
            tracking = set(self.before_tracking)
            if not sample.endpoint_issued:
                for i in STRONG_SERVO_INDICES:
                    if self.strong_permitted:
                        tracking.add(SERVO_NAMES[i])
                    else:
                        values[i] = nominal[i] = held[i]
                        tracking.discard(SERVO_NAMES[i])
                if self.preparation_permitted or self.strong_permitted:
                    tracking.add(SERVO_NAMES[PREP_INDEX])
                else:
                    values[PREP_INDEX] = nominal[PREP_INDEX] = held[PREP_INDEX]
                    tracking.discard(SERVO_NAMES[PREP_INDEX])
            elif not self.strong_permitted:
                for i in (*STRONG_SERVO_INDICES, PREP_INDEX):
                    values[i] = nominal[i] = held[i]
            sample = replace(sample, full12=tuple(values), nominal_full12=tuple(nominal),
                atomic_groups=groups, tracking_servo_names=tuple(n for n in SERVO_NAMES if n in tracking),
                target_progressed=any(abs(a-b)>1e-12 for a,b in zip(values,self.previous.full12)))
        self.previous = sample
        self._receipt_count += len(groups)
        return sample

    def receipt(self):
        return dict(mode=MODE, pending_event_tick=self.pending_event_tick,
            source_next_tick=self.original._tick_index, paused_wait_ticks=self.wait_ticks,
            preparation_issued=self.preparation_issued, strong_started=self.strong_started,
            preparation_permitted=self.preparation_permitted, strong_permitted=self.strong_permitted,
            pending_event_consumed=self.strong_started, within_episode_release_supported=True,
            source_clock_catch_up=False, replayed_source_pulses=0,
            source_stop_event_ticks=list(self.source_stop_ticks),
            reference_full12_group_deliberately_split=True,
            preparation_channels=[SERVO_NAMES[PREP_INDEX]], strong_channels=list(STRONG_NAMES),
            dispatched_split_or_stop_groups=self._receipt_count,
            source_full12_atomic_groups_emitted=self.source_atomic_emitted,
            original_full12_atomic_groups_proposed=self.original.source_atomic_emitted,
            held_value_kind="current_continuous_nominal_not_old_entry_or_FINAL",
            pauses_previously_issued_mapper_pursuit=False,
            policy_channels_restricted=[], mapper_or_history_resets=0, extra_actuator_writes=0)


def provider_type(base):
    class ContinuousProvider(base):
        def __init__(self, *args, read_local_state, **kwargs):
            self._read_local_state = read_local_state
            super().__init__(*args, **kwargs)
            if (self._p09_late_source is None
                    or self._rr_carry_source_mode not in (None, SUPPORTED_CARRY_MODE)):
                raise ValueError("continuous source requires accepted P09 pending/carry source")

        def _sequence_permission(self, layer, task, observation):
            state = local_state(self._read_local_state)
            if not state["active"]:
                if isinstance(layer["motion"], ContinuousLateCarrier):
                    raise ValueError("local active must remain latched during continuation")
                return super()._sequence_permission(layer, task, observation)
            ev = task.get("physical_evaluator", {})
            if (ev.get("valid") is not True or task.get("termination_reason") is not None
                    or ev.get("termination_reason") is not None):
                return False
            lanes = lane_permissions(state, task, self.spec["support"], rear_mode=self._rear_policy_timing_mode)
            if layer["stage"] == "P12":
                phase = layer["motion"].phase
                hip_waypoints = [w for w in phase.waypoints if "rear_left_hip" in w.changed_channels]
                roll_groups = [g for g in phase.atomic_groups if set(g.channels).intersection(WHEEL_NAMES)]
                if not hip_waypoints or not roll_groups:
                    raise ValueError("P12 requires identifiable knee preparation then hip/wheel source")
                strong_tick = layer["motion"]._scaled_source_tick(hip_waypoints[0].time_s)
                roll_tick = layer["motion"]._scaled_source_tick(roll_groups[0].time_s)
                knee_prep = layer.get("rl_ticks", 0) < strong_tick
                joint_permitted = bool(lanes["rl_lane_permitted"]
                    or (knee_prep and lanes["preparation_permitted"]))
                layer["rl_dependency_wait"] = not joint_permitted
                # Before the pulse, keep wheel and knee/hip cadence together.
                # After its first write, the independent stop clock is inexorable.
                pulse_started = layer["ticks"] > roll_tick
                advance = bool(pulse_started or (joint_permitted and
                    (layer["ticks"] < roll_tick or layer.get("rl_ticks", 0) >= roll_tick)))
                if not joint_permitted and layer.get("rl_sample") is not None:
                    old = layer["rl_sample"]
                    values = list(old.full12)
                    values[4:6] = self.nominal_full12[4:6]
                    layer["rl_sample"] = replace(old, full12=tuple(values),
                        nominal_full12=tuple(values), tracking_servo_names=())
                layer["sequence_diagnostic"] = dict(lanes, mode=MODE,
                    status="preparation" if knee_prep and joint_permitted else "active" if joint_permitted else "holding_only_RL_joint_lane",
                    wait_reason=None if joint_permitted else "current_support_for_strong_RL_action",
                    knee_preparation_source_end_tick=strong_tick, wheel_launch_source_tick=roll_tick,
                    wheel_pulse_started=pulse_started, wheel_stop_clock_unpaused=pulse_started,
                    local_policy_still_active=True, source_clock_catch_up=False)
                return advance
            if layer["stage"] != "P09":
                return super()._sequence_permission(layer, task, observation)
            motion = layer["motion"]
            pending = motion._scaled_source_tick(self._p09_late_source[1])
            if not isinstance(motion, ContinuousLateCarrier):
                if layer["ticks"] < pending:
                    return super()._sequence_permission(layer, task, observation)
                if layer["ticks"] != pending:
                    raise ValueError("continuous source cannot retrofit an already-issued late event")
                motion = layer["motion"] = ContinuousLateCarrier(motion,
                    previous_sample=layer.get("sample"), pending_event_tick=pending,
                    read_nominal=lambda: self.nominal_full12)
            motion.permissions(preparation=lanes["preparation_permitted"],
                               strong=lanes["strong_transfer_permitted"])
            diag = layer.setdefault("sequence_diagnostic", {})
            diag.update(lanes, mode=MODE, status="strong_source_released" if motion.strong_started
                else "near_top_preparation" if lanes["preparation_permitted"] else "pending_late_transfer",
                wait_reason=None if lanes["strong_transfer_permitted"] else "current_RR_bearing_for_strong_FL_RL_transfer",
                local_policy_still_active=True, deferred_late_source=motion.receipt())
            if lanes["strong_transfer_permitted"] and "late_group_start_tick" not in diag:
                diag["late_group_start_tick"] = ev.get("physics_tick")
            return True

        def _rr_waiting_late_group(self, layer):
            motion = layer["motion"]
            if isinstance(motion, ContinuousLateCarrier):
                return not motion.strong_started
            return super()._rr_waiting_late_group(layer)

        def _source_normal_bias(self, motion, sample):
            result = list(super()._source_normal_bias(motion, sample))
            if isinstance(motion, ContinuousLateCarrier):
                if not motion.strong_permitted:
                    for i in STRONG_SERVO_INDICES:
                        result[i] = 0.
                if not (motion.preparation_permitted or motion.strong_permitted):
                    result[PREP_INDEX] = 0.
            for layer in self._continuous_layers:
                if layer.get("rl_motion") is motion and layer.get("rl_dependency_wait"):
                    result[4:6] = [0., 0.]
            return tuple(result)

        def continuation_source_state(self):
            return {layer["stage"]: dict(ticks=layer["ticks"],
                endpoint=bool(layer.get("sample") and layer["sample"].endpoint_issued),
                rl_ticks=layer.get("rl_ticks")) for layer in self._continuous_layers}

        def continuation_pause_inputs(self, task):
            state = local_state(self._read_local_state)
            ev = task.get("physical_evaluator", {})
            dep = rear_dependency(task, self.spec["support"], mode=self._rear_policy_timing_mode)
            source_started = any(isinstance(layer["motion"], ContinuousLateCarrier)
                and layer["motion"].strong_started for layer in self._continuous_layers)
            for layer in self._continuous_layers:
                if layer["stage"] == "P12":
                    first_hip = next(w for w in layer["motion"].phase.waypoints
                                     if "rear_left_hip" in w.changed_channels)
                    source_started = source_started or layer.get("rl_ticks", 0) > layer["motion"]._scaled_source_tick(first_hip.time_s)
            owners = self.rear_late_owner_bits()
            return dict(active=state["active"],
                live=bool(ev.get("valid") is True and ev.get("termination_reason") is None
                          and task.get("termination_reason") is None),
                source_strong_started=bool(source_started),
                rr_current_bearing=bool(dep["rr_current_bearing"]),
                rr_support_continuation_valid=state["rr_support_continuation_valid"],
                rl_current_swing=bool(dep["rl_current_swing"]),
                owned_indices=[i for j,i in enumerate((0,1,4,5)) if i in (0,1,4) and owners[j]])

        def rear_policy_timing(self, task):
            result = dict(super().rear_policy_timing(task))
            state = local_state(self._read_local_state)
            permission = continuation_permission(state, task.get("physical_evaluator", {}))
            if permission["ready"] and task.get("termination_reason") is None:
                # Capture work and receiving-side preparation can overlap.
                # No AIR/force/history flag is synthesized by these task roles.
                result["rl_prep_transfer"] = not result.get("rl_swing_capture", False)
                result["rr_continuation_mode"] = MODE
            for layer in self._continuous_layers:
                motion = layer["motion"]
                if isinstance(motion, ContinuousLateCarrier):
                    # Existing observable clock slot exposes the clock that
                    # actually consumes source events, not the wall wait count.
                    result["p09_source_time_s"] = min(200., motion.original._tick_index/self.physics_hz)
                    result["p09_dependency_wait"] = not motion.strong_started
            return result

        @property
        def nominal_suggestion_diagnostics(self):
            result = dict(super().nominal_suggestion_diagnostics)
            result["rr_continuous_source_v3"] = dict(mode=MODE,
                local_state=local_state(self._read_local_state),
                layers=[layer["motion"].receipt() for layer in self._continuous_layers
                        if isinstance(layer["motion"], ContinuousLateCarrier)],
                p12=[dict(layer.get("sequence_diagnostic", {})) for layer in self._continuous_layers
                     if layer["stage"] == "P12"],
                runtime_rear_owner_projection_required=False,
                physical_support_awarded=False, extra_actuator_writes=0)
            return result
    return ContinuousProvider


def supervisor_type(base):
    class ContinuousSupervisor(base):
        def __init__(self, *args, read_local_state, **kwargs):
            self._read_local_state = read_local_state
            self.read_continuation_source_state = lambda: {}
            super().__init__(*args, **kwargs)

        def entry_report(self, stage_id, evaluation=None):
            result = dict(super().entry_report(stage_id, evaluation))
            ev = self.evaluator.snapshot if evaluation is None else evaluation
            permission = continuation_permission(local_state(self._read_local_state), ev)
            if stage_id in ("P10", "P11", "P12") and permission["ready"]:
                # Keep physical predicates and history untouched: only the
                # scheduler may enter preparation before measured placement.
                conditions = self.spec["stages"][stage_id]["valid_start_conditions"]
                actual = {name: self.predicate(name, ev) for name in conditions}
                remaining_reasons = [name for name in result["reasons"] if name != "placed_RR"]
                result.update(valid=not remaining_reasons, reasons=remaining_reasons,
                    rr_continuation_permission=permission, actual_condition_values=actual,
                    rr_placement_waived_for_scheduling_only=actual.get("placed_RR", 1.) < 1.,
                    physical_completion_awarded=False)
            return result

        def observe_and_update(self, observation, *, sim_time_s=None):
            observed_tick = (observation.get("physics_tick") if isinstance(observation, Mapping)
                             else getattr(observation, "physics_tick", None))
            if observed_tick == getattr(self, "_continuation_processed_tick", -1):
                return dict(self._snapshot)
            before_stage = self.stage_id
            result = super().observe_and_update(observation, sim_time_s=sim_time_s)
            self._continuation_processed_tick = observed_tick
            ev = result["physical_evaluator"]
            state = local_state(self._read_local_state)
            permission = continuation_permission(state, ev)
            tick, now = ev.get("physics_tick"), ev.get("simulation_time_s")
            if ev.get("termination_reason") is not None:
                return result
            local_expired = self.termination_source in ("LOCAL_TASK_DEADLINE", "LOCAL_BOUNDED_RECOVERY_EXHAUSTED")
            if self.termination_reason is not None and not local_expired:
                return result
            if (state["active"] and ev.get("valid") is True
                    and self.stage_id in ("P09", "P10", "P11", "P12", "P13") and local_expired):
                self.termination_reason = self.termination_source = None
                result.update(termination_reason=None, termination_source=None, success=False)
                result["local_timeout"] = dict(result["local_timeout"],
                    classification="continuous_rear_recovery_warning_global200_and_safety_authoritative",
                    local_episode_terminal_enabled=False)
                self._snapshot = dict(result)
            if not permission["ready"]:
                return result
            sources = self.read_continuation_source_state()
            source_complete = bool(sources.get(self.stage_id, {}).get("endpoint"))
            # Never add a second forward transition to the same native frame.
            handoff = (self.stage_id == before_stage and type(tick) is int and tick % 8 == 0
                and (self.stage_id == "P09" or (self.stage_id in ("P10", "P11") and source_complete)))
            if handoff:
                previous = self.stage_id
                self.stage_id = self.spec["stages"][previous]["next_phase"]
                self.stage_started_s = now
                self._progress_samples.clear()
                self.transition_evidence.append(dict(from_stage=previous, to_stage=self.stage_id,
                    sim_time_s=now, physics_tick=tick, scheduler_transition=True,
                    physical_completion_awarded=False,
                    reason="RR near-top/touch continuation; completed source preparation can hand over",
                    source_endpoint_before_handoff=source_complete, rr_continuation_permission=permission))
                stage = self.spec["stages"][self.stage_id]
                entry = self.entry_report(self.stage_id, ev)
                goals = {name: self.predicate(name, ev) for name in stage["completion_predicates"]}
                progress = sum(goals.values())/len(goals)
                limit = float(stage["maximum_task_duration"])
                result.update(stage_id=self.stage_id, purpose=stage["purpose"],
                    stage_age_s=0., stage_elapsed_s=0., phase_progress=progress,
                    entry_valid=entry["valid"], entry_reasons=entry["reasons"],
                    completion_values=goals, substage="TRANSFER" if self.stage_id in ("P10", "P11") else "EXECUTION",
                    transition_evidence=list(self.transition_evidence), stall_diagnostic=False,
                    local_timeout=dict(nominal_limit_s=limit, current_progress_allowance_s=0.,
                        fixed_post_window_allowance_s=0., effective_limit_s=limit,
                        nominal_limit_exceeded=False, classification="continuous_task_not_milestone_terminal",
                        timer_inputs_observable_in_existing372=True))
                if self.spec.get("transfer_roles"):
                    active_role = ev.get("transfer_roles", {}).get(stage["active_leg"], {})
                    result.update(transfer_role_context=active_role,
                        pending_capture=bool(active_role.get("pending_capture") or result.get("fl_capture_pending")))
            result["rr_continuation_scheduling"] = dict(permission, mode=MODE,
                local_policy_still_active=state["active"], rr_touch_seen=state["rr_touch_seen"],
                physical_completion_awarded=False, local_success_is_not_done=True)
            self._snapshot = dict(result)
            return result
    return ContinuousSupervisor


def controller_factory(*, task_spec_path, read_local_state):
    from pathlib import Path
    from wlr50_clean.ppo.semantic_supervisor import (NominalMotionProvider,
        SemanticControllerAdapter, TaskStageSupervisor, load_fsm_spec, load_motion_contract)
    def build(fsm_path, motion_contract_path):
        spec = load_fsm_spec(Path(fsm_path))
        contract = load_motion_contract(Path(motion_contract_path))
        supervisor = supervisor_type(TaskStageSupervisor)(task_spec_path,
            read_local_state=read_local_state)
        provider = provider_type(NominalMotionProvider)(contract, spec=supervisor.spec,
            fsm_spec=spec, read_local_state=read_local_state)
        supervisor.read_continuation_source_state = provider.continuation_source_state
        return SemanticControllerAdapter(spec, contract, supervisor=supervisor,
                                         nominal_provider=provider)
    return build
