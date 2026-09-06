"""One task-semantic episode: 15 Hz policy, authoritative 120 Hz physics."""
from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Mapping, Sequence

from .phase_action_masks_v2 import PhaseTransitionBridge
from .residual_direct_env import ResidualStep
from .semantic_observation import (
    HISTORY_GROUPS, SemanticNonFiniteObservation, SemanticObservationBuilder,
    SemanticObservationError, load_semantic_observation_schema, semantic_task, vector,
)
from .semantic_reward import SemanticRewardCalculator, SemanticRewardSample, load_semantic_reward_config

ZERO12 = (0.0,)*12
PHYSICS_DT_S = 1/120
WRITE_COUNTERS = ("in_episode_root_pose_writes", "in_episode_root_velocity_writes",
                  "in_episode_force_or_impulse_writes", "in_episode_gravity_writes")


def _native_tick_evidence(source: Any, current: Any, *, request_phase: str,
                          raw: tuple[float,...], handoff_hold: bool) -> dict[str,Any]:
    audit = current.info.get("actuator_target_effect_audit")
    audit = audit if isinstance(audit,Mapping) else {}
    ack = current.info.get("atomic_ack")
    ack = ack if isinstance(ack,Mapping) else {}
    changed = audit.get("changed_channels_full12")
    count = audit.get("changed_target_channel_count")
    changed_valid = (isinstance(changed,(list,tuple)) and len(changed)==12
                     and all(type(value) is bool for value in changed)
                     and type(count) is int and count==sum(changed))
    verified = bool(audit.get("schema")=="wlr50_clean.actuator_target_effect_audit.v1"
        and all(audit.get(key) is True for key in ("verified","actual_mapping_matches_dispatch",
                 "setter_dispatch_targets_equal","same_tick_counterfactual"))
        and type(audit.get("physics_tick")) is int and audit["physics_tick"]==ack.get("physics_tick")
        and audit.get("source_phase_id")==source.state_id
        and audit.get("policy_request_phase")==request_phase
        and audit.get("raw_policy_action_full12") is not None
        and tuple(audit["raw_policy_action_full12"])==raw and changed_valid)
    return {"episode_physics_tick":current.physics_tick,"command_physics_tick":audit.get("physics_tick"),
            "source_phase_id":source.state_id,"verified":verified,
            "changed_target_channel_count":count if changed_valid else None,
            "actual_native_effect":verified and count>0,
            "own_phase_request_effect":verified and count>0 and source.state_id==request_phase and not handoff_hold,
            "handoff_hold_used":handoff_hold}


def _terminal_reason(frame: Any) -> str | None:
    signals = frame.termination_signals
    for name,reason in (("nan_inf","NAN_INF"),("physics_explosion","PHYSICS_EXPLOSION"),
                        ("body_collision","BODY_COLLISION"),("wheel_only_climb","WHEEL_ONLY_CLIMB"),
                        ("fall","FALL"),("hard_joint_limit","HARD_JOINT_LIMIT")):
        if getattr(signals,name,False):
            return reason
    task = semantic_task(frame)
    reason = task.get("termination_reason")
    result = frame.info.get("controller_task_result")
    if hasattr(result,"value"):
        result = result.value
    if result in ("SAFETY_ABORT","INFRASTRUCTURE_ERROR","VIDEO_OR_ARTIFACT_ERROR"):
        raise RuntimeError(f"semantic runtime aborted: {result}: {reason}")
    if task["success"]:
        return "SUCCESS"
    if reason is not None:
        if not isinstance(reason,str) or not reason:
            raise SemanticObservationError("invalid task termination reason")
        if reason == "SUCCESS":
            raise SemanticObservationError("success reason lacks task success")
        return reason
    if getattr(signals,"success",False):
        raise SemanticObservationError("backend success lacks task evaluator success")
    if getattr(signals,"timeout",False) or frame.sim_time_s >= 200.0-1e-10:
        return "TASK_TIMEOUT"
    if result not in (None,"RUNNING","NOT_STARTED","IN_PROGRESS"):
        raise RuntimeError(f"unmapped terminal controller result: {result}")
    return None


class SemanticEpisodeEnv:
    def __init__(self, backend: Any, *, projector: Any = None, action_config: Any = None,
                 reward_config_path: Any = None, observation_schema_path: Any = None,
                 collect_trace: bool = True, tick_observer: Callable | None = None):
        if projector is not None and action_config is not None:
            raise ValueError("choose projector or action_config, not both")
        if projector is None:
            from .semantic_backend import build_semantic_projector
            projector = build_semantic_projector() if action_config is None else build_semantic_projector(action_config)
        self.backend = backend
        self.projector = projector
        self.bridge = PhaseTransitionBridge(projector)
        self.observation_schema = load_semantic_observation_schema() if observation_schema_path is None else load_semantic_observation_schema(observation_schema_path)
        reward_config = load_semantic_reward_config() if reward_config_path is None else load_semantic_reward_config(reward_config_path)
        if self.observation_schema.maximum_task_duration_s != reward_config.values["maximum_task_duration_s"]:
            raise ValueError("actor and reward task horizons differ")
        self.reward_calculator = SemanticRewardCalculator(reward_config)
        self.observation_builder = SemanticObservationBuilder(self.observation_schema)
        self.collect_trace = bool(collect_trace)
        self.tick_observer = tick_observer
        self.frame = None
        self.observation = None
        self.done = True
        self.decision_count = 0
        self.trace: list[dict[str,Any]] = []
        self._total_decisions = self._total_physics_ticks = self._episodes = 0
        self._phase_decisions: Counter = Counter()
        self._terminations: Counter = Counter()

    @property
    def observation_dimension(self) -> int:
        return self.observation_schema.dimension

    def reset(self, seed: int = 1001, options: Mapping[str,Any] | None = None) -> tuple[float,...]:
        frame = self.backend.reset(seed=seed,options=dict(options or {}))
        if _terminal_reason(frame) is not None:
            raise SemanticObservationError("reset must produce a valid nonterminal physical state")
        self.observation_builder.reset()
        self.reward_calculator.reset()
        nominal = vector(frame.nominal_action_full12,12,"reset nominal")
        drive = vector(frame.info["drive_target_full12"],12,"reset actual drive")
        self._history = {name:ZERO12 for name in HISTORY_GROUPS}
        self._history.update(previous_applied_full12=drive,previous_previous_applied_full12=drive,
                             previous_nominal_full12=nominal)
        self._semantic_frame = self.observation_builder.build(frame,self._history)
        self.bridge.reset(state_id=frame.state_id,applied_action_full12=nominal)
        self.frame = frame
        self.observation = self.observation_schema.encode(self._semantic_frame.groups)
        self.done = False
        self.decision_count = 0
        self.trace = []
        self._episodes += 1
        self._episode_return = 0.0
        self._transition_evidence_count = len(semantic_task(frame).get("transition_evidence",()))
        return self.observation

    def step(self, raw_policy_action_full12: Sequence[float]) -> ResidualStep:
        if self.done or self.frame is None:
            raise RuntimeError("reset is required before stepping a semantic episode")
        raw = vector(raw_policy_action_full12,12,"raw policy action")
        start_frame, start_semantic = self.frame,self._semantic_frame
        if hasattr(self.backend,"set_actuator_target_audit_request"):
            self.backend.set_actuator_target_audit_request(start_frame.state_id,raw,
                                                          self.projector.config.mask_for(start_frame.state_id))
        samples, transitions, native_ticks = [],[],[]
        write_samples = {name:[] for name in WRITE_COUNTERS}
        reason = None
        invalid_terminal_observation = False
        for _ in range(8):
            source = self.frame
            before = self._semantic_frame
            bridged = self.bridge.project_tick(raw,state_id=source.state_id,
                nominal_action_full12=source.nominal_action_full12,
                reference_action_full12=source.nominal_action_full12,reference_delta_full12=ZERO12,
                runtime_action_mask_full12=source.action_mask_full12,safety=source.safety_projection,dt_s=PHYSICS_DT_S)
            projection = bridged.projection
            current = self.backend.step_physics(projection.applied_action_full12)
            if current.physics_tick != source.physics_tick+1 or abs(current.sim_time_s-source.sim_time_s-PHYSICS_DT_S)>1e-8:
                raise RuntimeError("backend must execute exactly one 120 Hz physics tick")
            reason = _terminal_reason(current)
            drive = vector(current.info["drive_target_full12"],12,"actual dispatched drive")
            nominal = vector(source.nominal_action_full12,12,"source nominal")
            residual = vector(projection.safe_projected_residual_full12,12,"projected residual")
            old_history = dict(self._history)
            self._history.update(previous_raw_full12=raw,previous_residual_full12=residual,
                previous_previous_residual_full12=old_history["previous_residual_full12"],
                previous_applied_full12=drive,previous_previous_applied_full12=old_history["previous_applied_full12"],
                previous_nominal_full12=nominal)
            try:
                after = self.observation_builder.build(current,self._history)
            except SemanticNonFiniteObservation:
                if reason not in ("NAN_INF","PHYSICS_EXPLOSION"):
                    raise
                # Numerical physical failure has no value bootstrap. Preserve
                # last finite encoding explicitly, never credit it as live state.
                after = before
                invalid_terminal_observation = True
            caps = tuple(a*b for a,b in zip(self.projector.config.scale_for(source.state_id),
                                            self.projector.config.physical_residual_scale_full12,strict=True))
            samples.append(SemanticRewardSample(before,after,PHYSICS_DT_S,nominal,
                old_history["previous_nominal_full12"],residual,old_history["previous_residual_full12"],
                drive,old_history["previous_applied_full12"],old_history["previous_previous_applied_full12"],caps))
            self.frame,self._semantic_frame = current,after
            self._total_physics_ticks += 1
            native_ticks.append(_native_tick_evidence(source,current,request_phase=start_frame.state_id,raw=raw,
                handoff_hold=bool(bridged.transition_metric and bridged.transition_metric.handoff_hold_used)))
            for name in WRITE_COUNTERS:
                write_samples[name].append(current.info.get(name))
            if bridged.transition_metric is not None:
                transitions.append(bridged.transition_metric.as_dict())
            if self.tick_observer is not None:
                self.tick_observer(source,current,projection)
            if reason is not None:
                break
        reward = self.reward_calculator.evaluate(start_semantic,self._semantic_frame,samples,
                                                termination_reason=reason,task_success=reason=="SUCCESS")
        self.observation = self.observation_schema.encode(self._semantic_frame.groups)
        self.done = reason is not None
        self.decision_count += 1
        self._total_decisions += 1
        self._phase_decisions[start_frame.state_id] += 1
        self._episode_return += reward["total"]
        if reason:
            self._terminations[reason] += 1
        write_counters = {name:max(values) if all(type(value) is int and value>=0 for value in values) else None
                          for name,values in write_samples.items()}
        all_stage_evidence = semantic_task(self.frame).get("transition_evidence",())
        if len(all_stage_evidence) < self._transition_evidence_count:
            raise SemanticObservationError("semantic stage evidence cannot shrink during an episode")
        stage_evidence = list(all_stage_evidence[self._transition_evidence_count:])
        self._transition_evidence_count = len(all_stage_evidence)
        info = {"schema":"wlr50_clean.semantic_decision.v2","phase_id":start_frame.state_id,
            "end_phase_id":self.frame.state_id,"physics_tick":self.frame.physics_tick,
            "sim_time_s":self.frame.sim_time_s,"physics_ticks":len(samples),"decision_count":self.decision_count,
            "raw_policy_action_full12":raw,"applied_action_full12":projection.applied_action_full12,
            "nominal_action_full12":samples[-1].nominal,"projected_residual_full12":samples[-1].residual,
            "actual_drive_target_full12":samples[-1].actual_drive,
            "actuator_target_effect_audit":self.frame.info.get("actuator_target_effect_audit"),
            "actuator_target_effect_audit_ticks":native_ticks,
            "actuator_target_effect_audit_summary":{
                "physics_ticks":len(native_ticks),"verified_tick_count":sum(row["verified"] for row in native_ticks),
                "all_ticks_verified":all(row["verified"] for row in native_ticks),
                "actual_native_effect_tick_count":sum(row["actual_native_effect"] for row in native_ticks),
                "own_phase_request_effect_tick_count":sum(row["own_phase_request_effect"] for row in native_ticks)},
            **write_counters,"no_in_episode_state_writes_verified":all(value==0 for value in write_counters.values()),
            "stage_transition_evidence":stage_evidence,
            "semantic_task":dict(semantic_task(self.frame)),"reward":reward,"reward_breakdown":reward,
            "termination_reason":reason,"task_success":reason=="SUCCESS","time_outs":False,
            "task_outcome_label":"FULL_TASK_SUCCESS" if reason=="SUCCESS" else reason,
            "full_task_success":reason=="SUCCESS",
            "terminal_bootstrap_allowed":False if reason else True,
            "discount_convention":"gamma_once_per_issued_policy_action; task_terminal_no_bootstrap",
            "terminal_observation_finite_fallback":invalid_terminal_observation,
            "phase_transition_action_jump":transitions,"episode_return":self._episode_return,
            "episode_elapsed_s":self.frame.sim_time_s}
        if self.collect_trace:
            self.trace.append(info)
        return ResidualStep(self.observation,reward["total"],self.done,False,info)

    def telemetry_summary(self) -> dict[str,Any]:
        return {"decisions":self._total_decisions,"physics_ticks":self._total_physics_ticks,
                "episodes":self._episodes,"phase_decisions":dict(self._phase_decisions),
                "terminations":dict(self._terminations),"observation_dimension":self.observation_dimension}
