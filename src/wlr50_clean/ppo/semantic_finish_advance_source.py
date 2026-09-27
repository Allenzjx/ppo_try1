"""Declared, opt-in RL-placement -> measured advance -> home reference owner.

The task owns all physical permissions and clocks. This module owns only
nominal scheduling and a replayable FINAL-reference projection; it writes no
measured state, does not award support/success, and never resets HISTORY.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math

from wlr50_clean.infrastructure.command_batch import Full12Command

MODE = "post_RL_advance_then_home_final_reference_v1"
CONTEXT_SCHEMA = "wlr50_clean.requested_finish_context.v1"
ADVANCE = "ADVANCE_FOR_HOME_CLEARANCE"
HOME = "HOME_RECOVERY"
SETTLE = "FINAL_SETTLE"
DONE = "DONE"
MODES = (ADVANCE, HOME, SETTLE, DONE)


def _vector(value, size, name):
    if (not isinstance(value, (tuple, list)) or len(value) != size
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in value)):
        raise ValueError("finish reference requires finite " + name)
    return tuple(float(x) for x in value)


def _positive(value, name, maximum=None):
    if (type(value) not in (int, float) or not math.isfinite(value) or value <= 0.
            or (maximum is not None and value > maximum)):
        raise ValueError("invalid bounded finish " + name)
    return float(value)


def home_duration_s(entry_final, target_servo, *, maximum_rate_deg_s=20., minimum_s=2.):
    """Quintic position profile: peak s'(u)=1.875, below existing drive rates."""
    from types import SimpleNamespace
    from .semantic_finish_advance_task import home_duration
    entry = _vector(entry_final, 12, "home entry FINAL12")
    target = _vector(target_servo, 8, "source home8")
    rate = _positive(maximum_rate_deg_s, "home rate", 150.)
    return home_duration(SimpleNamespace(home_minimum_duration_s=_positive(minimum_s,"home minimum time"),
        home_target_servo_deg=target,home_rate_deg_s=rate),entry[:8])


def validate_context(context):
    if (not isinstance(context, Mapping) or context.get("schema") != CONTEXT_SCHEMA
            or context.get("active") is not True or context.get("mode") not in MODES):
        raise ValueError("unknown active requested-finish context")
    ticks = [context.get(k) for k in ("activation_tick", "reference_tick", "observation_tick")]
    if any(type(x) is not int for x in ticks) or not 0 <= ticks[0] <= ticks[1] <= ticks[2]:
        raise ValueError("finish requires ordered physical event/reference/observation ticks")
    _vector(context.get("reference_final_full12"), 12, "entry FINAL12")
    _vector(context.get("reference_request_full12"), 12, "entry REQUEST12")
    caps = _vector(context.get("reference_capacities_full12"), 12, "entry capacities12")
    if any(x <= 0. for x in caps):
        raise ValueError("finish residual capacities must remain positive")
    _vector(context.get("home_target_servo_deg"), 8, "source home8")
    _positive(context.get("advance_wheel_rad_s"), "advance wheel", 2.0943951023931953)
    _positive(context.get("wheel_rate_rad_s2"), "wheel slew", 1.8)
    if context["mode"] != ADVANCE:
        entry_tick = context.get("home_entry_tick")
        if type(entry_tick) is not int or entry_tick != context["reference_tick"]:
            raise ValueError("home/settle must use the committed HOME-entry reference")
        _positive(context.get("home_duration_s"), "home duration")


def _toward(start, target, delta):
    return target if abs(target-start) <= delta else start + math.copysign(delta,target-start)


def reference_baseline(context, *, physics_hz=120.):
    validate_context(context)
    hz = _positive(physics_hz, "physics frequency")
    entry = _vector(context["reference_final_full12"], 12, "entry FINAL12")
    elapsed = (context["observation_tick"]-context["reference_tick"]) / hz
    fraction = 0.
    if context["mode"] == ADVANCE:
        servos = entry[:8]
        wheel_target = context["advance_wheel_rad_s"]
    else:
        u = min(1., max(0., elapsed/context["home_duration_s"]))
        fraction = u*u*u*(10.+u*(-15.+6.*u))
        servos = (tuple(context["home_target_servo_deg"]) if u >= 1. else
            tuple(a+fraction*(b-a) for a,b in zip(entry[:8], context["home_target_servo_deg"])))
        wheel_target = 0.
    wheels = tuple(_toward(x, wheel_target, context["wheel_rate_rad_s2"]*elapsed) for x in entry[8:])
    return servos+wheels, elapsed, fraction


def final_reference_candidate(candidate, *, native_full12, controller_full12,
        residual_full12, context, previous_ack, previous_tick, write_count,
        policy_headroom_mode=None):
    """Apply the declared final reference, then unchanged physical projection.

The wrapper context is independently captured before dispatch. Replay and the
zero-current-policy counterfactual use the same committed reference; only the
current REQUEST differs. Raw actor actions/log likelihood remain unchanged.
    """
    finish = context.get("finish_context")
    validate_context(finish)
    if finish["observation_tick"] != context.get("observation_tick"):
        raise ValueError("finish/post-RR observation ticks must identify the same committed frame")
    candidate = _vector(candidate, 12, "candidate12")
    native = _vector(native_full12, 12, "mapped native12")
    controller = _vector(controller_full12, 12, "controller12")
    request = _vector(residual_full12, 12, "REQUEST12")
    if (not isinstance(previous_ack, Mapping) or type(previous_tick) is not int
            or type(write_count) is not int or previous_ack.get("physics_tick") != previous_tick
            or previous_ack.get("write_count") != write_count
            or context.get("dispatch_physics_tick") != previous_tick+1):
        raise ValueError("finish requires adjacent committed ACK, not reconstructed history")
    previous = _vector(previous_ack.get("drive_target_full12"), 12, "previous FINAL12")
    reference = _vector(finish["reference_request_full12"], 12, "reference REQUEST12")
    baseline, elapsed, fraction = reference_baseline(finish, physics_hz=context["physics_hz"])
    relative = tuple(a-b for a,b in zip(request,reference))
    headroom = None
    projected = tuple(a+b for a,b in zip(baseline,relative))
    if policy_headroom_mode is not None:
        from .semantic_headroom import HEADROOM_MODE, project_semantic_servo_headroom
        if policy_headroom_mode != HEADROOM_MODE:
            raise ValueError("unknown finish headroom mode")
        headroom = project_semantic_servo_headroom(native_full12=baseline,
            controller_bias_full12=(0.,)*12, projected_residual_full12=relative)
        projected = tuple(headroom["candidate_native_target_before_final_slew_full12"])
    result = list(projected)
    # All four wheel commands use the existing bounded slew. Canonical/native
    # signs are transformed only by the original final atomic write.
    for i in range(8,12):
        result[i] = _toward(previous[i], result[i], finish["wheel_rate_rad_s2"]/context["physics_hz"])
    return dict(mode=MODE, context=deepcopy(dict(context)), owner_indices=list(range(12)),
        preparation_indices=[], paused_strong_indices=[], rebased_axes_with_current_support_pause=[],
        previous_ack_physics_tick=previous_tick, previous_ack_write_count=write_count,
        previous_final_full12=list(previous),
        previous_requested_full12=list(_vector(previous_ack.get("independent_policy_residual_requested_full12"),12,"previous REQUEST12")),
        requested_full12=list(request), reference_requested_full12=list(reference),
        reference_baseline_full12=list(baseline),
        rebased_requested_full12=list(relative),
        additional_local_residual_projection=False,
        candidate_before_full12=list(candidate), candidate_after_full12=result,
        correction_full12=[a-b for a,b in zip(result,candidate)],
        bypassed_mapped_native_full12=list(native), bypassed_controller_full12=list(controller),
        relative_headroom_evidence=headroom, preparation_elapsed_s=elapsed,
        finish_reference_elapsed_s=elapsed, finish_home_fraction=fraction,
        target_semantics="post_RL_declared_FINAL_reference_plus_current_REQUEST_minus_latched_REQUEST_then_original_physical_projection",
        wheel_coordinates="canonical_forward_positive_rad_s_original_native_axis_mapping_unchanged",
        front_pair_nominal_candidate=False, exhausted_recovery_hold=False,
        raw_sample_and_log_probability_unchanged=True, HISTORY_reset=False,
        mapper_advanced_again=False, nominal_history_modified=False,
        RR_target_modified=any(result[i]!=candidate[i] for i in (6,7)),
        support_credit_awarded=False, runtime_preparation_controller_contribution=False,
        runtime_finish_controller_contribution=True,
        old_source_pulses_replayed=False, actual_state_written=False)


def provider_type(base):
    class FinishAdvanceProvider(base):
        def __init__(self, *args, **kwargs):
            self._requested_finish_diagnostic = None
            super().__init__(*args, **kwargs)

        def evaluate(self, stage, observation=None):
            post = self.post_rr_preparation_inputs()
            finish = post.get("finish_context")
            if not isinstance(finish, Mapping) or finish.get("active") is not True:
                return super().evaluate(stage, observation)
            validate_context(finish)
            # No call into old evaluate: this uniformly pauses every source
            # lane, final-stop acquisition/home and endpoint fallback, without
            # consuming paused source time or clearing physical history.
            baseline, elapsed, fraction = reference_baseline(finish, physics_hz=self.physics_hz)
            self.state_id = stage if isinstance(stage,str) else str(stage["stage_id"])
            self.nominal_full12 = Full12Command.from_full12(baseline).clamped().to_full12()
            self.tracking_servo_names = ()
            self.normal_drive_bias_full12 = (0.,)*12
            self.endpoint_issued = False
            self._requested_finish_diagnostic = dict(mode=MODE, task=deepcopy(dict(finish)),
                task_owns_home_and_success_permission=True, old_source_clocks_paused=True,
                old_source_clock_catch_up=False, old_final_stop_owner_bypassed=True,
                old_endpoint_home_bypassed=True, observed_reference_elapsed_s=elapsed,
                nominal_home_fraction=fraction, final_reference_applied_after_existing_mapper=True,
                extra_actuator_writes=0, HISTORY_reset=False, success_awarded=False)
            return self.nominal_full12

        @property
        def nominal_suggestion_diagnostics(self):
            result = super().nominal_suggestion_diagnostics
            if self._requested_finish_diagnostic is None:
                return result
            return dict(result, requested_finish=deepcopy(self._requested_finish_diagnostic))
    return FinishAdvanceProvider


def controller_factory(*, task_spec_path, read_local_state, read_post_rr_state,
                       finish_advance_config):
    from pathlib import Path
    from .semantic_supervisor import (NominalMotionProvider, SemanticControllerAdapter,
        TaskEvaluator, TaskStageSupervisor, load_fsm_spec, load_motion_contract)
    from .semantic_rr_capture_continuation_source import provider_type as continuous_provider_type
    from .semantic_rr_capture_continuation_source import supervisor_type as continuous_supervisor_type
    from .semantic_post_rr_front_prep_source import provider_type as post_provider_type
    def build(fsm_path, motion_contract_path):
        spec=load_fsm_spec(Path(fsm_path));contract=load_motion_contract(Path(motion_contract_path))
        evaluator=TaskEvaluator(task_spec_path,finish_recovery_enabled=True,
            finish_advance_config=finish_advance_config)
        supervisor=continuous_supervisor_type(TaskStageSupervisor)(task_spec_path,
            read_local_state=read_local_state,evaluator=evaluator)
        provider=provider_type(post_provider_type(continuous_provider_type(NominalMotionProvider)))(
            contract,spec=supervisor.spec,fsm_spec=spec,read_local_state=read_local_state,
            read_post_rr_state=read_post_rr_state)
        supervisor.read_continuation_source_state=provider.continuation_source_state
        return SemanticControllerAdapter(spec,contract,supervisor=supervisor,nominal_provider=provider)
    return build
