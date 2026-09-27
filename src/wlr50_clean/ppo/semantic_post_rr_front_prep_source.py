"""Declared post-touch front-pair preparation and residual-reference owner.

No simulator state, policy sample, likelihood, or HISTORY is changed here.
Before the accepted RR touch the wrapper delegates byte-for-byte. Afterward
only four axes use a new, observed residual reference. The finite preparation
is a controller contribution, not a claim that PPO discovered these targets.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import replace
import math

from .semantic_rr_capture_continuation_source import (
    ContinuousLateCarrier, local_state, lane_permissions,
    provider_type as continuous_provider_type,
    supervisor_type as continuous_supervisor_type,
)
from .semantic_rr_capture_deferred_late import SERVO_NAMES, WHEEL_NAMES

MODE = "post_RR_touch_front_pair_final_reference_v1"
PREP_INDICES = (1, 3, 8, 9)  # FL knee, FR knee, FL wheel, FR wheel; canonical.
HELD_STRONG_INDICES = (0, 4)  # Already-issued FL/RL hip pursuits, not RR.
FLAGS = ("post_rr_active", "post_rr_preparing", "post_rr_prepared", "post_rr_prep_exhausted")


def vector(values, name):
    if (not isinstance(values, (list, tuple)) or len(values) != 12
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in values)):
        raise ValueError("post-RR preparation requires finite full12 " + name)
    return tuple(float(x) for x in values)


def read_state(callback):
    state = callback()
    if not isinstance(state, Mapping) or any(type(state.get(k)) is not bool for k in FLAGS):
        raise ValueError("post-RR preparation requires explicit task flags")
    if state["post_rr_active"]:
        if (type(state.get("post_rr_touch_tick")) is not int
                or type(state.get("observation_tick")) is not int
                or not 0 <= state["post_rr_touch_tick"] <= state["observation_tick"]):
            raise ValueError("post-RR preparation requires a latched measured touch tick")
    elif state["post_rr_preparing"] or state["post_rr_prepared"]:
        raise ValueError("preparation cannot precede accepted RR touch")
    return dict(state)


def make_context(state, *, dispatch_physics_tick, physics_hz=120.,
                 knee_rate_deg_s=30., wheel_rate_rad_s2=1.8):
    if not state["post_rr_active"]:
        return None
    context = {k: state[k] for k in FLAGS}
    for key in ("post_rr_touch_tick", "observation_tick", "post_rr_entry_final_full12",
                "post_rr_entry_residual_full12"):
        context[key] = deepcopy(state[key])
    context.update(mode=MODE, dispatch_physics_tick=dispatch_physics_tick,
        physics_hz=physics_hz,
        candidate_knee_deg=state.get("post_rr_candidate_knee_deg", 30.),
        candidate_wheel_rad_s=state.get("post_rr_candidate_wheel_rad_s", 1.),
        post_rr_source_pause_indices=[],
        post_rr_current_rl_swing=False,
        knee_rate_deg_s=state.get("post_rr_knee_rate_deg_s", knee_rate_deg_s),
        wheel_rate_rad_s2=min(state.get("post_rr_wheel_rate_rad_s2", wheel_rate_rad_s2), wheel_rate_rad_s2))
    validate_context(context)
    return context


def validate_context(context):
    if not isinstance(context, Mapping) or context.get("mode") != MODE:
        raise ValueError("unknown post-RR front preparation context")
    if any(type(context.get(k)) is not bool for k in FLAGS) or not context["post_rr_active"]:
        raise ValueError("post-RR context requires active explicit task flags")
    if (type(context.get("dispatch_physics_tick")) is not int
            or type(context.get("post_rr_touch_tick")) is not int
            or type(context.get("observation_tick")) is not int
            or not 0 <= context["post_rr_touch_tick"] <= context["observation_tick"]):
        raise ValueError("post-RR context requires exact completed observation and dispatch ticks")
    for key in ("physics_hz", "knee_rate_deg_s", "wheel_rate_rad_s2"):
        x = context.get(key)
        if type(x) not in (int, float) or not math.isfinite(x) or x <= 0.:
            raise ValueError("post-RR rates must be positive finite")
    if context["knee_rate_deg_s"] > 150. or context["wheel_rate_rad_s2"] > 1.8:
        raise ValueError("preparation cannot increase existing physical command rates")
    for key,low,high in (("candidate_knee_deg",-58.,208.),("candidate_wheel_rad_s",-2.0943951023931953,2.0943951023931953)):
        value=context.get(key)
        if type(value) not in (int,float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError("preparation candidate must remain within original command limits")
    vector(context.get("post_rr_entry_final_full12"), "touch FINAL")
    vector(context.get("post_rr_entry_residual_full12"), "touch REQUEST")
    pause=context.get("post_rr_source_pause_indices", [])
    if (not isinstance(pause,(tuple,list)) or len(pause)!=len(set(pause))
            or any(type(i) is not int or i not in (0,1,4) for i in pause)):
        raise ValueError("post-RR reference requires explicit dependent source pause indices")
    if type(context.get("post_rr_current_rl_swing",False)) is not bool:
        raise ValueError("post-RR reference requires current RL swing Boolean")


def _toward(start, target, distance):
    return start + max(-distance, min(distance, target-start))


def final_reference_candidate(candidate, *, native_full12, controller_full12,
        residual_full12, context, previous_ack, previous_tick, write_count,
        policy_headroom_mode=None):
    """Pure independent-replay transform, before the original final servo slew.

The four old residual offsets are subtracted from the *actual current filtered
REQUEST*, before a fresh headroom projection around the new reference. Old
absolute-request clipping is deliberately not reused. Unselected axes preserve
the incoming candidate. The same transform applies to actual/zero-policy audit
branches, with identical committed reference and task state.
    """
    validate_context(context)
    candidate = vector(candidate, "candidate")
    native = vector(native_full12, "native")
    controller = vector(controller_full12, "controller")
    request = vector(residual_full12, "REQUEST")
    if (not isinstance(previous_ack, Mapping) or type(previous_tick) is not int
            or type(write_count) is not int or previous_ack.get("physics_tick") != previous_tick
            or previous_ack.get("write_count") != write_count
            or context["dispatch_physics_tick"] != previous_tick + 1):
        raise ValueError("post-RR reference requires adjacent committed ACK")
    previous_final = vector(previous_ack.get("drive_target_full12"), "previous FINAL")
    previous_request = vector(previous_ack.get("independent_policy_residual_requested_full12", (0.,)*12),
                              "previous REQUEST")
    entry = vector(context["post_rr_entry_final_full12"], "touch FINAL")
    reference = vector(context["post_rr_entry_residual_full12"], "touch REQUEST")
    elapsed = (context["observation_tick"] - context["post_rr_touch_tick"] + 1) / context["physics_hz"]
    baseline = [a+b for a,b in zip(native, controller)]
    relative = [0.] * 12
    preparing = context["post_rr_preparing"] and not context["post_rr_prep_exhausted"]
    for i in PREP_INDICES:
        relative[i] = request[i] - reference[i]
        if preparing:
            target, rate = (context["candidate_knee_deg"], context["knee_rate_deg_s"]) if i < 8 else (
                context["candidate_wheel_rad_s"], context["wheel_rate_rad_s2"])
            baseline[i] = _toward(entry[i], target, rate*elapsed)
        elif context["post_rr_prep_exhausted"] and not context["post_rr_prepared"]:
            # Diagnostic budget exhaustion is not functional preparation. Stop
            # this module's wheel pulse; do not restore the old negative knee
            # goal and accidentally execute the deferred recoil anyway.
            baseline[i] = previous_final[i] if i < 8 else 0.
            if i < 8:
                relative[i] = request[i] - previous_request[i]
    hold = tuple(i for i in HELD_STRONG_INDICES if not context["post_rr_prepared"]
        and not (i == 4 and context.get("post_rr_current_rl_swing",False)))
    for i in hold:
        baseline[i] = previous_final[i]
        relative[i] = request[i] - previous_request[i]
    reference_paused = tuple(i for i in PREP_INDICES
        if context["post_rr_prepared"] and i in context.get("post_rr_source_pause_indices", ()))
    for i in reference_paused:
        baseline[i] = previous_final[i]
        relative[i] = request[i] - previous_request[i]
    indices = tuple(sorted(set(PREP_INDICES + hold)))
    projected = [a+b for a,b in zip(baseline, relative)]
    headroom = None
    if policy_headroom_mode is not None:
        from .semantic_headroom import HEADROOM_MODE, project_semantic_servo_headroom
        if policy_headroom_mode != HEADROOM_MODE:
            raise ValueError("unknown post-RR headroom mode")
        headroom = project_semantic_servo_headroom(native_full12=baseline,
            controller_bias_full12=(0.,)*12, projected_residual_full12=relative)
        projected = list(headroom["candidate_native_target_before_final_slew_full12"])
    result = list(candidate)
    for i in indices:
        result[i] = projected[i]
    # The declared preparation ramp and its exit have the same bounded wheel
    # rate. It is not an extra global forward-only projection or abs(speed).
    for i in (8, 9):
        result[i] = _toward(previous_final[i], result[i],
                            context["wheel_rate_rad_s2"] / context["physics_hz"])
    return dict(mode=MODE, context=deepcopy(dict(context)), owner_indices=list(indices),
        preparation_indices=list(PREP_INDICES), paused_strong_indices=list(hold),
        rebased_axes_with_current_support_pause=list(reference_paused),
        previous_ack_physics_tick=previous_tick, previous_ack_write_count=write_count,
        previous_final_full12=list(previous_final), previous_requested_full12=list(previous_request),
        requested_full12=list(request), reference_requested_full12=list(reference),
        reference_baseline_full12=baseline, rebased_requested_full12=relative,
        candidate_before_full12=list(candidate), candidate_after_full12=result,
        correction_full12=[a-b for a,b in zip(result,candidate)],
        relative_headroom_evidence=headroom, preparation_elapsed_s=elapsed,
        target_semantics="declared_front_pair_reference_plus_current_filtered_REQUEST_minus_touch_REQUEST",
        wheel_coordinates="canonical_forward_positive_rad_s_FL_native_minus_FR_native_plus",
        front_pair_nominal_candidate=preparing,
        exhausted_recovery_hold=bool(context["post_rr_prep_exhausted"] and not context["post_rr_prepared"]),
        raw_sample_and_log_probability_unchanged=True, HISTORY_reset=False,
        mapper_advanced_again=False, nominal_history_modified=False,
        RR_target_modified=False, support_credit_awarded=False,
        runtime_preparation_controller_contribution=True)


def provider_type(base):
    class PostRRFrontPreparationProvider(base):
        def __init__(self, *args, read_post_rr_state, **kwargs):
            self._read_post_rr_state = read_post_rr_state
            super().__init__(*args, **kwargs)

        def post_rr_preparation_inputs(self):
            return read_state(self._read_post_rr_state)

        def _sequence_permission(self, layer, task, observation):
            post = self.post_rr_preparation_inputs()
            block = post["post_rr_active"] and not post["post_rr_prepared"]
            if not block:
                return super()._sequence_permission(layer, task, observation)
            ev = task.get("physical_evaluator", {})
            if (ev.get("valid") is not True or ev.get("termination_reason") is not None
                    or task.get("termination_reason") is not None):
                return False
            if layer["stage"] == "P09":
                motion = layer["motion"]
                pending = motion._scaled_source_tick(self._p09_late_source[1])
                if not isinstance(motion, ContinuousLateCarrier):
                    if layer["ticks"] < pending:
                        return super()._sequence_permission(layer, task, observation)
                    if layer["ticks"] != pending:
                        raise ValueError("post-RR preparation cannot replay an already-issued late source")
                    motion = layer["motion"] = ContinuousLateCarrier(motion,
                        previous_sample=layer["sample"], pending_event_tick=pending,
                        read_nominal=lambda: self.nominal_full12)
                lanes = lane_permissions(local_state(self._read_local_state), task,
                    self.spec["support"], rear_mode=self._rear_policy_timing_mode)
                motion.permissions(preparation=lanes["preparation_permitted"], strong=False)
                layer.setdefault("sequence_diagnostic", {}).update(post_rr_front_preparation=dict(
                    active=True, strong_source_permitted=False,
                    started_wheel_stop_clock_unpaused=motion.strong_started,
                    pending_source_clock_catch_up=False, source_clock=motion.original._tick_index))
                # Carrier holds its unconsumed source clock, but a previously
                # started wheel pulse always advances through its authored stop.
                return True
            if layer["stage"] == "P12":
                advance = super()._sequence_permission(layer, task, observation)
                lanes = lane_permissions(local_state(self._read_local_state), task,
                    self.spec["support"], rear_mode=self._rear_policy_timing_mode)
                if lanes["rl_current_swing"]:
                    return advance  # Never block a real qualified AIR recovery.
                phase = layer["motion"].phase
                roll_groups = [g for g in phase.atomic_groups if set(g.channels).intersection(WHEEL_NAMES)]
                if not roll_groups:
                    raise ValueError("P12 source has no identifiable wheel pulse/stop")
                launch = layer["motion"]._scaled_source_tick(roll_groups[0].time_s)
                started = layer["ticks"] > launch
                layer["rl_dependency_wait"] = True
                old = layer.get("rl_sample")
                if old is not None:
                    values = list(old.full12); values[4:6] = self.nominal_full12[4:6]
                    layer["rl_sample"] = replace(old, full12=tuple(values),
                        nominal_full12=tuple(values), tracking_servo_names=())
                layer.setdefault("sequence_diagnostic", {}).update(post_rr_front_preparation=dict(
                    active=True, RL_source_joint_clock_paused=True,
                    started_wheel_stop_clock_unpaused=started, source_clock_catch_up=False))
                return started
            return super()._sequence_permission(layer, task, observation)

        def continuation_pause_inputs(self, task):
            result = super().continuation_pause_inputs(task)
            post = self.post_rr_preparation_inputs()
            if post["post_rr_active"] and not post["post_rr_prepared"]:
                # New front reference owns FL knee; the old absolute-reference
                # pause must not cancel it. Other old pause axes stay intact.
                result = dict(result, owned_indices=[i for i in result["owned_indices"] if i != 1])
            return result

        @property
        def nominal_suggestion_diagnostics(self):
            result = dict(super().nominal_suggestion_diagnostics)
            result["post_rr_front_preparation"] = dict(mode=MODE,
                state=self.post_rr_preparation_inputs(), declared_reference_indices=list(PREP_INDICES),
                final_reference_applied_after_nominal_mapper=True, extra_actuator_writes=0,
                sample_or_history_reset=False, support_credit_awarded=False)
            return result
    return PostRRFrontPreparationProvider


def controller_factory(*, task_spec_path, read_local_state, read_post_rr_state):
    from pathlib import Path
    from .semantic_supervisor import (NominalMotionProvider, SemanticControllerAdapter,
        TaskStageSupervisor, load_fsm_spec, load_motion_contract)
    def build(fsm_path, motion_contract_path):
        spec = load_fsm_spec(Path(fsm_path)); contract = load_motion_contract(Path(motion_contract_path))
        supervisor = continuous_supervisor_type(TaskStageSupervisor)(task_spec_path,
            read_local_state=read_local_state)
        provider = provider_type(continuous_provider_type(NominalMotionProvider))(
            contract, spec=supervisor.spec, fsm_spec=spec, read_local_state=read_local_state,
            read_post_rr_state=read_post_rr_state)
        supervisor.read_continuation_source_state = provider.continuation_source_state
        return SemanticControllerAdapter(spec, contract, supervisor=supervisor, nominal_provider=provider)
    return build
