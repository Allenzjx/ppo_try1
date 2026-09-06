"""N=1 reset-only frozen teacher with continuous measured state and a PPO credit seam.

The teacher creates a training distribution, not a current-policy success.
Ordinary stage transitions never reset this physical episode or its GAE stream.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any, Callable, Mapping

from wlr50_clean.fsm.controller import ControllerFrame, SensorFsmController
from wlr50_clean.fsm.state_spec import Lifecycle, load_fsm_spec
from wlr50_clean.fsm.task_result import TaskResult, TaskTermination
from wlr50_clean.infrastructure.command_batch import SERVO_ORDER
from wlr50_clean.reference.motion_contract import load_motion_contract
from .semantic_backend import SemanticIsaacBackend
from .semantic_env import ZERO12
from .semantic_legacy_evaluation import measured_observation
from .semantic_observation import vector
from .semantic_supervisor import SemanticControllerAdapter, TaskStageSupervisor
from .semantic_training import SemanticRslAdapter, jsonable, verified_native_effect

SAMPLING = "natural_P01_A_teacher_prefix_then_semantic_suffix_N1.v2"
PREFIX_TARGETS = tuple(f"P{index:02}" for index in range(6, 14))


def sampling_label(request):
    return f"{SAMPLING}:{request.target_phase}:offset_{request.teacher_offset_decisions}"


class PrefixUnavailable(RuntimeError):
    """A recorded initialization miss, never a discarded credited PPO failure."""
    def __init__(self, reason: str, *, tick: int, task: Mapping[str, Any], observation=None, receipt=None):
        super().__init__(reason)
        self.record = {"reason":reason,"physics_tick":tick,"semantic_task":dict(task),
            "last_observation":None if observation is None else measured_observation(observation),
            "last_executed_dispatch":None if receipt is None else receipt.record()}


@dataclass(frozen=True)
class PrefixRequest:
    target_phase: str = "P06"
    maximum_prefix_decisions: int = 1800
    maximum_takeover_decisions: int = 30
    teacher_offset_decisions: int = 0

    def __post_init__(self):
        if self.target_phase not in PREFIX_TARGETS:
            raise ValueError("prefix target must include a supported P06-P13 preparation/suffix stage")
        for value in (self.maximum_prefix_decisions,self.maximum_takeover_decisions):
            if type(value) is not int or not 1 <= value <= 3000:
                raise ValueError("prefix limits must be bounded positive decision counts")
        if type(self.teacher_offset_decisions) is not int or not 0 <= self.teacher_offset_decisions < self.maximum_prefix_decisions:
            raise ValueError("teacher offset must fit inside the total prefix decision limit")
        if self.maximum_takeover_decisions > self.maximum_prefix_decisions:
            raise ValueError("takeover is part of, not additional to, the total prefix budget")

    def as_dict(self):
        return {"target_phase":self.target_phase,
                "maximum_prefix_decisions":self.maximum_prefix_decisions,
                "maximum_takeover_decisions":self.maximum_takeover_decisions,
                "teacher_offset_decisions":self.teacher_offset_decisions,
                "teacher":"unmodified_SensorFsmController_reset_only",
                "fallback":"one_fresh_P01","task_horizon_includes_prefix":True,
                "entry_is_exact_historical_snapshot":False,
                "ordinary_phase_transitions_end_episode":False}


@dataclass(frozen=True)
class DispatchReceipt:
    source_tick: int
    command_tick: int
    nominal: tuple[float, ...]
    tracking: tuple[str, ...]
    bias: tuple[float, ...]
    ack: Mapping[str, Any]

    @classmethod
    def from_ack(cls, *, source_tick, command, tracking, bias, ack):
        if type(source_tick) is not int or source_tick < 0:
            raise ValueError("receipt requires actual source control tick")
        nominal = vector(command,12,"receipt nominal")
        total_bias = vector(bias,12,"receipt post-mapper bias")
        names = tuple(tracking)
        if len(set(names))!=len(names) or any(name not in SERVO_ORDER for name in names):
            raise ValueError("invalid dispatch tracking names")
        if (type(ack.get("physics_tick")) is not int
                or ack.get("articulation_writes_this_call")!=1
                or tuple(ack.get("applied_full12",()))!=nominal
                or tuple(ack.get("drive_feedback_bias_requested_full12",()))!=total_bias):
            raise ValueError("handoff seed is not bound to the actual atomic dispatch")
        vector(ack["native_drive_target_full12"],12,"receipt native")
        vector(ack["drive_target_full12"],12,"receipt final drive")
        return cls(source_tick,ack["physics_tick"],nominal,names,total_bias,dict(ack))

    def record(self):
        return {"source_control_tick":self.source_tick,"command_physics_tick":self.command_tick,
                "nominal_full12":self.nominal,"tracking_servo_names":self.tracking,
                "controller_bias_full12":self.bias,"atomic_ack":dict(self.ack)}


def _task_frame(tick, now, task, nominal, tracking=(), bias=ZERO12):
    reason = task["termination_reason"]
    lifecycle = Lifecycle.DONE if reason else (Lifecycle.EXECUTE_MOTION if task["entry_valid"] else Lifecycle.WAIT_ENTRY)
    termination = None if reason is None else TaskTermination(
        TaskResult(reason),task["stage_id"],lifecycle.value,now,
        task["physical_evaluator"].get("reason") or reason,task)
    command = tuple(nominal) if reason is None else tuple(nominal[:8])+ZERO12[8:]
    return ControllerFrame(tick,now,task["stage_id"],lifecycle,command,tick%8==0,
        True,False,tuple(tracking),ZERO12,tuple(bias),
        {"mode":"reset_only_teacher_semantic_task_owner","semantic_task":task},
        False,termination,None,())


class ResetOnlyPrefixController:
    """The semantic supervisor owns task truth from tick zero; only one command source."""
    def __init__(self, teacher, supervisor, spec, contract, request: PrefixRequest):
        self.teacher,self.supervisor,self.spec,self.contract = teacher,supervisor,spec,contract
        self.evaluator = supervisor.evaluator
        self.request = request
        self.motion = teacher.motion
        self.physics_tick = 0
        self.mode = "TEACHER"
        self._receipt = None
        self._semantic = None
        self._bias = ZERO12
        self._handoff_tick = None
        self.handoff_record = None
        self.teacher_calls = 0
        self._target_enter_tick = None

    @property
    def task_snapshot(self):
        return self.supervisor.snapshot

    @property
    def task_progress(self):
        return float(self.task_snapshot.get("phase_progress",0.))

    def record_verified_dispatch(self, receipt: DispatchReceipt):
        if receipt.source_tick != self.physics_tick-1:
            raise ValueError("receipt source does not match the emitted controller frame")
        self._receipt = receipt

    def step(self, observation, *, sim_time_s=None):
        tick = self.physics_tick
        now = tick/120 if sim_time_s is None else float(sim_time_s)
        observed_tick = observation.get("physics_tick") if isinstance(observation,Mapping) else observation.physics_tick
        if observed_tick != tick or not math.isclose(now,tick/120,rel_tol=0.,abs_tol=1e-9):
            raise ValueError("teacher/shadow observation clock is not the real continuous clock")
        if tick and (self._receipt is None or self._receipt.source_tick!=tick-1):
            raise ValueError("new observation has no matching executed dispatch receipt")
        if self._semantic is not None:
            frame = self._semantic.step(observation,sim_time_s=now)
            if self.mode!="READY":
                rates=(self.supervisor.spec["nominal"]["servo_handoff_rate_deg_s"],)*8+(
                    self.supervisor.spec["nominal"]["wheel_handoff_rate_rad_s2"],)*4
                self._bias=tuple(x-math.copysign(min(abs(x),rate/120),x) if x else 0.
                                 for x,rate in zip(self._bias,rates,strict=True))
                # Both the generated command and the last actual dispatch must
                # be bias-free before ordinary B/C credit can begin.
                if self._bias==ZERO12 and self._receipt.bias==ZERO12:
                    self.mode="READY"
                elif tick-self._handoff_tick > self.request.maximum_takeover_decisions*8:
                    raise PrefixUnavailable("bounded takeover did not retire teacher bias",tick=tick,task=self.task_snapshot,observation=observation,receipt=self._receipt)
            frame=replace(frame,normal_drive_bias_full12=self._bias,
                drive_feedback_details={**frame.drive_feedback_details,"reset_only_teacher_takeover":self.mode!="READY"})
        else:
            task=self.supervisor.observe_and_update(observation,sim_time_s=now)
            if task["stage_id"] == self.request.target_phase and self._target_enter_tick is None:
                self._target_enter_tick = tick
            if (self._target_enter_tick is not None and task["stage_id"] != self.request.target_phase
                    and task["termination_reason"] is None):
                raise PrefixUnavailable("requested phase ended before its offset handoff",tick=tick,
                                        task=task,observation=observation,receipt=self._receipt)
            if task["termination_reason"] is not None:
                frame=_task_frame(tick,now,task,self._receipt.nominal if self._receipt else ZERO12)
            elif (tick and tick%8==0 and task["stage_id"]==self.request.target_phase
                  and task["physical_evaluator"]["valid"]
                  and tick-self._target_enter_tick >= self.request.teacher_offset_decisions*8):
                self._semantic,frame=SemanticControllerAdapter.from_live_prefix(
                    self.spec,self.contract,supervisor=self.supervisor,
                    nominal_full12=self._receipt.nominal,tracking_servo_names=self._receipt.tracking,
                    physics_tick=tick,sim_time_s=now)
                self.motion=self._semantic.motion
                self._bias=self._receipt.bias
                self._handoff_tick=tick
                self.mode="READY" if self._bias==ZERO12 else "TAKEOVER"
                self.handoff_record={**self._receipt.record(),"handoff_tick":tick,
                    "handoff_time_s":now,"actual_phase":task["stage_id"],
                    "semantic_task":dict(task),"teacher_calls":self.teacher_calls,
                    "target_first_observed_tick":self._target_enter_tick,
                    "teacher_offset_decisions":self.request.teacher_offset_decisions,
                    "entry_observation":measured_observation(observation)}
                frame=replace(frame,normal_drive_bias_full12=self._bias)
                # Releasing the object is not editing/forcing its old FSM state.
                self.teacher=None
            else:
                teacher_frame=self.teacher.step(observation,sim_time_s=now)
                self.teacher_calls+=1
                if teacher_frame.termination is not None:
                    miss=PrefixUnavailable("teacher ended before requested semantic start: "
                        +str(teacher_frame.termination.result)+": "+str(teacher_frame.termination.reason),
                        tick=tick,task=task,observation=observation,receipt=self._receipt)
                    miss.record["teacher_termination"]=jsonable(teacher_frame.termination)
                    raise miss
                if not teacher_frame.full12_atomic_write_required:
                    raise ValueError("teacher failed the established atomic Full12 interface")
                frame=replace(teacher_frame,state_id=task["stage_id"],
                    lifecycle=Lifecycle.EXECUTE_MOTION if task["entry_valid"] else Lifecycle.WAIT_ENTRY,
                    drive_feedback_details={**teacher_frame.drive_feedback_details,
                        "mode":"reset_only_frozen_teacher","semantic_task":task,
                        "teacher_phase":teacher_frame.state_id,"teacher_lifecycle":teacher_frame.lifecycle.value})
        self.physics_tick+=1
        return frame


class PrefixSemanticIsaacBackend(SemanticIsaacBackend):
    """Reuse physical reset and execution; capture the fresh ACK before controller.step."""
    def __init__(self, simulation_app=None, *, prefix_request=None, **kwargs):
        if "controller_factory" in kwargs or kwargs.get("audit_actuator_target_effect",True) is not True:
            raise ValueError("prefix owns explicit controller injection and requires native auditing")
        kwargs["audit_actuator_target_effect"]=True
        self.prefix_request=prefix_request or PrefixRequest()
        if not isinstance(self.prefix_request,PrefixRequest):
            raise ValueError("prefix_request must be a validated PrefixRequest")
        self._prefix_enabled=True
        super().__init__(simulation_app,controller_factory=self._new_prefix_controller,**kwargs)

    def configure_prefix(self, enabled: bool):
        if type(enabled) is not bool:
            raise ValueError("explicit prefix mode boolean required")
        self._prefix_enabled=enabled

    def _new_prefix_controller(self,fsm_path,contract_path):
        if not self._prefix_enabled:
            return SemanticControllerAdapter.from_paths(fsm_path,contract_path,
                task_spec_path=self.task_spec_path)
        teacher=SensorFsmController.from_paths(fsm_path,contract_path)
        return ResetOnlyPrefixController(teacher,TaskStageSupervisor(self.task_spec_path),
            load_fsm_spec(fsm_path),load_motion_contract(contract_path),self.prefix_request)

    @property
    def prefix_controller(self):
        return self._controller if isinstance(self._controller,ResetOnlyPrefixController) else None

    def _build_authoritative_frame(self,observation,controller_frame,*,previous_frame):
        frame=super()._build_authoritative_frame(observation,controller_frame,previous_frame=previous_frame)
        controller=self.prefix_controller
        mode=controller.mode if controller is not None else "FRESH_P01"
        frame.info.update(reset_sampling=sampling_label(self.prefix_request),reset_only_prefix_mode=mode,
            execution_mode={"TEACHER":"reset_only_A_teacher_prime",
                "TAKEOVER":"reset_only_semantic_takeover","READY":"semantic_teacher_initialized_suffix",
                "FRESH_P01":"semantic_fresh_P01_fallback"}[mode])
        return frame

    def _atomic_apply(self,adapter,command,*,physics_tick,tracking_servo_names,drive_feedback_bias_full12):
        controller=self.prefix_controller
        if controller is not None and controller.mode!="READY":
            request=self._actuator_target_audit_request
            if request is None or any(request["raw_policy_action_full12"]):
                raise ValueError("teacher/takeover initialization must use zero raw residual")
        ack=super()._atomic_apply(adapter,command,physics_tick=physics_tick,
            tracking_servo_names=tracking_servo_names,drive_feedback_bias_full12=drive_feedback_bias_full12)
        if controller is not None:
            controller.record_verified_dispatch(DispatchReceipt.from_ack(
                source_tick=self._controller_frame.physics_tick,command=command,
                tracking=tracking_servo_names,bias=drive_feedback_bias_full12,ack=ack))
        return ack


class PrefixCreditCore:
    """Proxy whose reset executes teacher initialization but whose step is only C data."""
    def __init__(self, core, *, evidence_sink: Callable[[Mapping[str,Any]],None]):
        if not isinstance(core.backend,PrefixSemanticIsaacBackend) or not callable(evidence_sink):
            raise ValueError("prefix requires the explicit backend and a persistent evidence sink")
        self.core,self.sink=core,evidence_sink
        self._initializing=False
        self._credit_open=False
        self.prefix_decisions=0
        self.prefix_ticks=0
        self.attempts=[]
        self.credited_decisions=0
        self.credited_ticks=0
        self._credited_phase_decisions=Counter()
        self._credited_terminations=Counter()
        self.reset_wall_time_s=0.
        self.roll_in_wall_time_s=0.
        self._credit_return_offset=0.
        self._previous_observer=core.tick_observer
        core.tick_observer=self._observe

    @property
    def frame(self):
        return self.core.frame

    def __getattr__(self,name):
        # Preserve the real core's read interface without resetting its state.
        return getattr(self.core,name)

    def _observe(self,before,after,projection):
        if self._initializing:
            self.prefix_ticks+=1
        if self._previous_observer is not None:
            self._previous_observer(before,after,projection)

    def reset(self,seed=1001,options=None):
        if options:
            raise ValueError("suffix source is natural P01; snapshot options are unsupported")
        self._credit_open=False
        self._initializing=True
        reset_started=perf_counter()
        backend=self.core.backend
        request=backend.prefix_request
        backend.configure_prefix(True)
        start_decisions,start_ticks=self.prefix_decisions,self.prefix_ticks
        last_info={}
        miss=None
        try:
            observation=self.core.reset(seed=seed)
            roll_in_started=perf_counter()
            self.reset_wall_time_s += roll_in_started-reset_started
            self.sink({"kind":"reset_only_prefix_start","policy_credit":False,
                "seed":seed,"request":request.as_dict(),
                "observation":measured_observation(self.frame.info["raw_observation"])})
            for _ in range(request.maximum_prefix_decisions):
                step=self.core.step(ZERO12)
                self.prefix_decisions+=1
                last_info=dict(step.info)
                verified_native_effect(last_info,ZERO12)
                if (last_info["actuator_target_effect_audit_summary"]["all_ticks_verified"] is not True
                        or last_info["no_in_episode_state_writes_verified"] is not True):
                    raise RuntimeError("prefix native audit/state-write evidence is incomplete")
                self.sink({"kind":"reset_only_prefix_decision","policy_credit":False,
                    **{key:last_info[key] for key in ("phase_id","end_phase_id","physics_tick","physics_ticks",
                        "sim_time_s","raw_policy_action_full12","projected_residual_full12",
                        "actual_drive_target_full12","actuator_target_effect_audit_summary",
                        "no_in_episode_state_writes_verified","termination_reason")}})
                if step.terminated:
                    miss={"reason":"physical_prefix_terminal","terminal_info":last_info}
                    break
                controller=backend.prefix_controller
                if controller.mode=="READY":
                    self.start_record={"mode":"teacher_initialized_suffix","requested_phase":request.target_phase,
                        "actual_phase":self.frame.state_id,"physics_tick":self.frame.physics_tick,
                        "sim_time_s":self.frame.sim_time_s,"remaining_task_time_s":self.frame.info["semantic_task"]["remaining_task_time_s"],
                        "handoff":controller.handoff_record,"from_P01_current_policy":False,
                        "credit_start_observation":measured_observation(self.frame.info["raw_observation"]),
                        "requested_phase_still_active_at_credit":self.frame.state_id==request.target_phase}
                    break
            else:
                miss={"reason":"bounded_prefix_exhausted"}
            if miss is None and backend.prefix_controller.mode!="READY":
                miss={"reason":"prefix_did_not_open_credit"}
        except PrefixUnavailable as exc:
            miss=exc.record
        finally:
            self._initializing=False
            if "roll_in_started" in locals():
                self.roll_in_wall_time_s += perf_counter()-roll_in_started
        attempt={"request":request.as_dict(),"prefix_decisions":self.prefix_decisions-start_decisions,
            "prefix_physics_ticks":self.prefix_ticks-start_ticks,"accepted":miss is None,"miss":miss}
        self.attempts.append(attempt)
        self.sink({"kind":"reset_only_prefix_result","policy_credit":False,**attempt})
        if miss is not None:
            backend.configure_prefix(False)
            fallback_started=perf_counter()
            observation=self.core.reset(seed=seed)
            self.reset_wall_time_s += perf_counter()-fallback_started
            last_info={}
            self.start_record={"mode":"fresh_P01_fallback","actual_phase":self.frame.state_id,
                "physics_tick":self.frame.physics_tick,"sim_time_s":self.frame.sim_time_s,
                "from_P01_current_policy":True,"prefix_miss":miss}
        else:
            observation=self.core.observation
        self._credit_return_offset=float(last_info.get("episode_return",0.))
        self._episode_credit_decisions=0
        self._credit_open=True
        self._start_reference={key:self.start_record[key] for key in (
            "mode","requested_phase","actual_phase","physics_tick","sim_time_s",
            "from_P01_current_policy","requested_phase_still_active_at_credit") if key in self.start_record}
        self._start_reference["prefix_attempt_index"]=len(self.attempts)-1
        self.sink({"kind":"policy_credit_start","policy_credit":False,"start":self.start_record})
        return tuple(observation)

    def step(self,raw):
        if not self._credit_open:
            raise RuntimeError("prefix reset must finish before policy collection")
        step=self.core.step(raw)
        info=dict(step.info)
        self.credited_decisions+=1
        self._episode_credit_decisions+=1
        self.credited_ticks+=int(info["physics_ticks"])
        self._credited_phase_decisions[info["phase_id"]]+=1
        if step.terminated:
            self._credited_terminations[info["termination_reason"]]+=1
        info["physical_core_decision_count_including_prefix"]=info["decision_count"]
        info["decision_count"]=self._episode_credit_decisions
        info["curriculum_start"]=dict(self._start_reference)
        info["physical_episode_return_including_prefix"]=info["episode_return"]
        info["episode_return"]=info["episode_return"]-self._credit_return_offset
        info["prefix_teacher_data_in_ppo_storage"]=False
        info["task_result_scope"]="full_task" if self.start_record["from_P01_current_policy"] else "teacher_initialized_suffix"
        info["task_outcome_label"]=("FULL_TASK_SUCCESS" if self.start_record["from_P01_current_policy"] else "SUFFIX_SUCCESS") if info["task_success"] else info["termination_reason"]
        info["full_task_success"]=bool(info["task_success"] and self.start_record["from_P01_current_policy"])
        return replace(step,info=info)

    def telemetry_summary(self):
        return {"reset_sampling":sampling_label(self.core.backend.prefix_request),"decisions":self.credited_decisions,"physics_ticks":self.credited_ticks,
            "phase_decisions":dict(self._credited_phase_decisions),"terminations":dict(self._credited_terminations),
            "prefix_behavior_decisions":self.prefix_decisions,
            "prefix_physics_ticks":self.prefix_ticks,"prefix_attempts":jsonable(self.attempts),
            "reset_wall_time_s":self.reset_wall_time_s,"roll_in_wall_time_s":self.roll_in_wall_time_s,
            "physical_core_including_prefix":self.core.telemetry_summary()}


class PrefixRslAdapter(SemanticRslAdapter):
    """Ordinary N=1 RSL storage; teacher work happens inside reset, never alg.act."""
    def __init__(self,core,*,seed,device="cuda:0",evidence_sink):
        super().__init__(PrefixCreditCore(core,evidence_sink=evidence_sink),seed=seed,device=device)
        self.cfg.update(reset_sampling=sampling_label(core.backend.prefix_request),prefix_request=core.backend.prefix_request.as_dict(),
            teacher_data_in_ppo_storage=False,external_window_truncation_supported=False)

    def step(self,actions):
        observations,rewards,dones,extras=super().step(actions)
        for summary in extras["episode_summaries"]:
            summary["task_outcome_label"]=summary["terminal_info"]["task_outcome_label"]
            summary["full_task_success"]=summary["terminal_info"]["full_task_success"]
        return observations,rewards,dones,extras

    def telemetry_summary(self):
        result=super().telemetry_summary()
        suffix_success=sum(row["task_success"] and not row["terminal_info"]["curriculum_start"]["from_P01_current_policy"]
                           for row in self.completed_episodes)
        result.update(reset_sampling=self.cfg["reset_sampling"],teacher_initialized_task_success_count=suffix_success,
            success_count=result["success_count"]-suffix_success,
            success_count_scope="fresh_P01_current_policy_only")
        return result
