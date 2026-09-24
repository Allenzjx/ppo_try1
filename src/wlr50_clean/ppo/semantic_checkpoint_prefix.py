"""N=1 checkpoint-policy roll-in on the same ordinary semantic episode.

Initialization is real deterministic policy execution, never PPO storage.
The caller supplies a frozen, separately owned actor callback; this module
does not receive or update the optimizer's actor or replace any controller.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
import json
import math
import re
from time import perf_counter
from typing import Any, Mapping

from .semantic_backend import SemanticIsaacBackend
from .semantic_env import WRITE_COUNTERS
from .semantic_observation import vector
from .semantic_policy_distribution import supported_heteroscedastic_contract_version
from .semantic_prefix import PREFIX_TARGETS, PrefixCreditCore, PrefixRslAdapter
from .semantic_training import SemanticRslAdapter, jsonable, verified_native_effect
from .semantic_transfer_roles import (
    ROLE_OBSERVATION_LAYOUT, ROLE_OBSERVATION_BASE_DIM, ROLE_OBSERVATION_DIM,
)
from .semantic_p05_capture_profile import P05_CAPTURE_OBSERVATION_DIM, P05_CAPTURE_OBSERVATION_LAYOUT
from .semantic_rr_capture_profile import RR_CAPTURE_OBSERVATION_DIM, RR_CAPTURE_OBSERVATION_LAYOUT
from .semantic_p02_progress_profile import P02_PROGRESS_OBSERVATION_LAYOUT
from .semantic_rear_policy_timing_profile import REAR_POLICY_TIMING_OBSERVATION_DIM, REAR_POLICY_TIMING_OBSERVATION_LAYOUT
from .semantic_rear_owner_profile import REAR_OWNER_OBSERVATION_DIM, REAR_OWNER_OBSERVATION_LAYOUT

SAMPLING = "natural_P01_frozen_checkpoint_policy_prefix_then_semantic_suffix_N1.v1"
RESULT_SCOPE = "checkpoint_policy_initialized_suffix"


@dataclass(frozen=True)
class CheckpointPolicyPrefixRequest:
    target_phase: str = "P06"
    teacher_offset_decisions: int = 0
    maximum_prefix_decisions: int = 1800
    source: str = "frozen_checkpoint_policy"

    def __post_init__(self):
        if self.source not in ("frozen_checkpoint_policy", "successful_nominal"):
            raise ValueError("unsupported reset-only prefix source")
        if self.target_phase not in PREFIX_TARGETS:
            raise ValueError("checkpoint prefix target must be P03-P13")
        if type(self.maximum_prefix_decisions) is not int or not 1 <= self.maximum_prefix_decisions <= 3000:
            raise ValueError("maximum prefix decisions must be an integer in [1,3000]")
        if (type(self.teacher_offset_decisions) is not int
                or not 0 <= self.teacher_offset_decisions < self.maximum_prefix_decisions):
            raise ValueError("target offset must fit inside the prefix decision budget")

    def as_dict(self):
        return {"schema": "wlr50_clean.checkpoint_policy_prefix_request.v1",
                "source": self.source, "target_phase": self.target_phase,
                "teacher_offset_decisions": self.teacher_offset_decisions,
                "maximum_prefix_decisions": self.maximum_prefix_decisions,
                "fallback": "one_fresh_P01", "task_horizon_includes_prefix": True,
                "entry_is_exact_historical_snapshot": False,
                "ordinary_phase_transitions_end_episode": False}


def sampling_label(request: CheckpointPolicyPrefixRequest) -> str:
    if not isinstance(request, CheckpointPolicyPrefixRequest):
        raise ValueError("validated checkpoint prefix request required")
    name = (SAMPLING if request.source == "frozen_checkpoint_policy" else
            "natural_P01_successful_nominal_prefix_then_semantic_suffix_N1.v1")
    return f"{name}:{request.target_phase}:offset_{request.teacher_offset_decisions}"


def _json_copy(value):
    def check(item):
        if isinstance(item, Mapping):
            if any(type(key) is not str for key in item):
                raise ValueError("provenance JSON keys must be strings")
            return {key: check(v) for key, v in item.items()}
        if isinstance(item, list):
            return [check(v) for v in item]
        if item is None or type(item) in (str, bool, int):
            return item
        if type(item) is float and math.isfinite(item):
            return item
        raise ValueError("provenance must contain only finite JSON values")
    return json.loads(json.dumps(check(value), allow_nan=False, sort_keys=True))


def _provenance(value):
    if not isinstance(value, Mapping):
        raise ValueError("checkpoint prefix provenance mapping required")
    result = _json_copy(value)
    if result.get("source") == "successful_nominal":
        if result.get("schema") != "wlr50_clean.successful_nominal_prefix.v1":
            raise ValueError("nominal prefix requires explicit non-policy provenance")
        for key in ("execution_profile_sha256", "stage_task_spec_sha256", "runtime_content_sha256"):
            if not isinstance(result.get(key), str) or not re.fullmatch("[0-9a-f]{64}", result[key]):
                raise ValueError("nominal prefix lacks live configuration binding")
        if result.get("interface_contract") not in (
                {"observation_dimension": ROLE_OBSERVATION_DIM,"observation_layout": ROLE_OBSERVATION_LAYOUT, "action_dimension": 12},
                {"observation_dimension": P05_CAPTURE_OBSERVATION_DIM,"observation_layout": P05_CAPTURE_OBSERVATION_LAYOUT,"action_dimension":12},
                {"observation_dimension": RR_CAPTURE_OBSERVATION_DIM,"observation_layout": RR_CAPTURE_OBSERVATION_LAYOUT,"action_dimension":12},
                {"observation_dimension": REAR_POLICY_TIMING_OBSERVATION_DIM,"observation_layout": REAR_POLICY_TIMING_OBSERVATION_LAYOUT,"action_dimension":12},
                {"observation_dimension":422,"observation_layout":P02_PROGRESS_OBSERVATION_LAYOUT,"action_dimension":12},
                {"observation_dimension":REAR_OWNER_OBSERVATION_DIM,"observation_layout":REAR_OWNER_OBSERVATION_LAYOUT,"action_dimension":12}):
            raise ValueError("nominal prefix interface differs from supported explicit layout/Full12")
        if result.get("raw_action_full12") != [0.0]*12 or result.get("policy_credit") is not False:
            raise ValueError("nominal prefix must be zero-residual with no PPO credit")
        return result
    if not isinstance(result.get("checkpoint_path"), str) or not result["checkpoint_path"].strip():
        raise ValueError("verified source checkpoint_path required")
    for key in ("checkpoint_sha256", "actor_parameter_sha256"):
        if not isinstance(result.get(key), str) or not re.fullmatch("[0-9a-f]{64}", result[key]):
            raise ValueError(f"verified source {key} required")
    for key in ("source_global_policy_decisions", "source_ppo_updates"):
        if type(result.get(key)) is not int or result[key] < 0:
            raise ValueError(f"source {key} must be a nonnegative integer")
    try:
        supported_heteroscedastic_contract_version(result.get("policy_contract"))
    except ValueError as error:
        raise ValueError("checkpoint prefix requires the exact heteroscedastic Full12 policy contract and supported layout") from error
    if "source_runtime_content_sha256" in result and (
            not isinstance(result["source_runtime_content_sha256"], str)
            or not re.fullmatch("[0-9a-f]{64}", result["source_runtime_content_sha256"])):
        raise ValueError("invalid source runtime content hash")
    return result


def _core_observation_layout(core):
    dimension = core.observation_dimension
    schema = getattr(core,"observation_schema",None)
    layout = getattr(schema,"observation_layout",getattr(schema,"transfer_role_features_version",None))
    if (type(dimension) is not int
            or not ((dimension == ROLE_OBSERVATION_BASE_DIM and layout is None)
                    or (dimension == ROLE_OBSERVATION_DIM and type(layout) is str
                        and layout == ROLE_OBSERVATION_LAYOUT)
                    or (dimension == P05_CAPTURE_OBSERVATION_DIM and layout == P05_CAPTURE_OBSERVATION_LAYOUT)
                    or (dimension == RR_CAPTURE_OBSERVATION_DIM and layout == RR_CAPTURE_OBSERVATION_LAYOUT)
                    or (dimension == REAR_POLICY_TIMING_OBSERVATION_DIM and layout == REAR_POLICY_TIMING_OBSERVATION_LAYOUT)
                    or (dimension == 422 and layout == P02_PROGRESS_OBSERVATION_LAYOUT)
                    or (dimension == REAR_OWNER_OBSERVATION_DIM and layout == REAR_OWNER_OBSERVATION_LAYOUT))):
        raise ValueError("checkpoint prefix requires the explicit supported observation layout")
    return dimension, layout


class _CheckpointPolicyCreditCore(PrefixCreditCore):
    """Reuse credited-step accounting, replacing only initialization ownership."""
    def __init__(self, core, *, request, evidence_sink):
        # In particular, reject the legacy teacher backend subclass. No
        # configure_prefix/controller factory is called by this module.
        if type(core.backend) is not SemanticIsaacBackend or not callable(evidence_sink):
            raise ValueError("checkpoint prefix requires the ordinary SemanticIsaacBackend and evidence sink")
        self._observation_dimension, self._observation_layout = _core_observation_layout(core)
        self.core, self.request, self.sink = core, request, evidence_sink
        self._initializing = self._credit_open = False
        self.prefix_decisions = self.prefix_ticks = 0
        self.credited_decisions = self.credited_ticks = 0
        self.attempts = []
        self._credited_phase_decisions, self._credited_terminations = Counter(), Counter()
        self.reset_wall_time_s = self.roll_in_wall_time_s = 0.
        self._credit_return_offset = 0.
        self._previous_observer = core.tick_observer
        core.tick_observer = self._observe
        self._policy = None
        self._bound_provenance = None
        self._bootstrap = None

    def _fresh_reset(self, seed):
        started = perf_counter()
        observation = tuple(self.core.reset(seed=seed))
        self.reset_wall_time_s += perf_counter()-started
        if (self.frame.state_id != "P01" or type(self.frame.physics_tick) is not int
                or self.frame.physics_tick != 0 or self.frame.sim_time_s != 0.
                or self.core.decision_count != 0 or self.core.done):
            raise RuntimeError("checkpoint prefix reset must be a fresh nonterminal P01 at t0")
        if _core_observation_layout(self.core) != (self._observation_dimension, self._observation_layout):
            raise RuntimeError("checkpoint prefix observation layout changed after construction")
        vector(observation, self._observation_dimension, "bootstrap observation")
        return observation

    def _bootstrap_signature(self):
        return (self.frame, self.frame.state_id, self.frame.physics_tick, self.frame.sim_time_s,
                self.core.decision_count, self.core.done, tuple(self.core.observation),
                json.dumps(jsonable(self.core._history), sort_keys=True, allow_nan=False),
                json.dumps(jsonable(self.frame.info["semantic_task"]), sort_keys=True, allow_nan=False))

    def reset(self, seed=1001, options=None):
        if options:
            raise ValueError("checkpoint roll-in starts at natural P01; reset options are unsupported")
        self._credit_open = False
        if self._policy is None:
            if self._bootstrap is not None:
                raise RuntimeError("install prefix policy before another reset or policy step")
            observation = self._fresh_reset(seed)
            self._bootstrap = self._bootstrap_signature()
            self.sink({"kind": "checkpoint_prefix_bootstrap", "policy_credit": False,
                       "seed": seed, "request": self.request.as_dict(), "physics_tick": 0})
            return observation
        self._fresh_reset(seed)
        return self._roll_in(seed)

    def install(self, policy_callable, provenance, *, seed):
        if self._policy is not None:
            raise RuntimeError("checkpoint prefix policy is immutable after installation")
        if not callable(policy_callable):
            raise ValueError("frozen deterministic prefix callable required")
        binding = _provenance(provenance)
        nominal = self.request.source == "successful_nominal"
        if nominal != (binding.get("source") == "successful_nominal"):
            raise ValueError("prefix source and provenance differ")
        contract = binding["interface_contract"] if nominal else binding["policy_contract"]
        if ((contract["observation_dimension"], contract.get("observation_layout"))
                != (self._observation_dimension, self._observation_layout)):
            raise ValueError("checkpoint prefix policy contract differs from the live observation layout")
        if self._bootstrap is None or self.frame is not self._bootstrap[0] or self._bootstrap_signature()[1:] != self._bootstrap[1:]:
            raise RuntimeError("prefix policy must be installed on the untouched original bootstrap t0")
        if hasattr(policy_callable, "provenance") and _provenance(policy_callable.provenance) != binding:
            raise ValueError("prefix callable provenance differs from installed source")
        self._policy, self._bound_provenance = policy_callable, binding
        return self._roll_in(seed)

    def _verified_prefix_step(self, step, raw, before_tick, before_time, before_phase):
        info = dict(step.info)
        verified_native_effect(info, raw)
        count = info.get("physics_ticks")
        if (type(count) is not int or not 1 <= count <= 8
                or info.get("phase_id") != before_phase or info.get("end_phase_id") != self.frame.state_id
                or info.get("physics_tick") != self.frame.physics_tick
                or self.frame.physics_tick != before_tick+count
                or not math.isclose(self.frame.sim_time_s-before_time, count/120., rel_tol=0., abs_tol=1e-8)
                or info.get("sim_time_s") != self.frame.sim_time_s
                or tuple(info.get("raw_policy_action_full12", ())) != raw):
            raise RuntimeError("prefix decision is not bound to its continuous physical interval")
        if step.truncated or bool(step.terminated) != bool(info.get("termination_reason")):
            raise RuntimeError("prefix requires genuine task termination, not external truncation")
        summary = info.get("actuator_target_effect_audit_summary", {})
        ticks = info.get("actuator_target_effect_audit_ticks", ())
        if (summary.get("all_ticks_verified") is not True
                or type(summary.get("physics_ticks")) is not int or summary["physics_ticks"] != count
                or type(summary.get("verified_tick_count")) is not int or summary["verified_tick_count"] != count
                or len(ticks) != count or any(row.get("verified") is not True for row in ticks)
                or [row.get("episode_physics_tick") for row in ticks] != list(range(before_tick+1, before_tick+count+1))
                or ticks[-1].get("command_physics_tick") != info["actuator_target_effect_audit"].get("physics_tick")):
            raise RuntimeError("prefix all-tick native audit is incomplete")
        for key, flag in (("actual_native_effect_tick_count", "actual_native_effect"),
                          ("own_phase_request_effect_tick_count", "own_phase_request_effect")):
            if (any(type(row.get(flag)) is not bool for row in ticks)
                    or type(summary.get(key)) is not int or summary[key] != sum(row[flag] for row in ticks)):
                raise RuntimeError("prefix native effect counts differ from verified ticks")
        if (info.get("no_in_episode_state_writes_verified") is not True
                or any(type(info.get(key)) is not int or info[key] != 0 for key in WRITE_COUNTERS)):
            raise RuntimeError("prefix in-episode root/velocity/force/gravity writes are forbidden")
        for key in ("projected_residual_full12", "actual_drive_target_full12", "applied_action_full12"):
            vector(info[key], 12, key)
        return info

    def _roll_in(self, seed):
        self._credit_open = False
        self._initializing = True
        started = perf_counter()
        first_decision, first_tick = self.prefix_decisions, self.prefix_ticks
        target_entry = None
        miss, last_info = None, {}
        self.sink({"kind": "checkpoint_prefix_start", "policy_credit": False,
                   "seed": seed, "request": self.request.as_dict(),
                   "prefix_policy_provenance": _json_copy(self._bound_provenance)})
        try:
            for index in range(1, self.request.maximum_prefix_decisions+1):
                before_tick, before_time, before_phase = self.frame.physics_tick, self.frame.sim_time_s, self.frame.state_id
                raw = vector(self._policy(tuple(self.core.observation)), 12, "frozen checkpoint raw action")
                if self.request.source == "successful_nominal" and raw != (0.0,)*12:
                    raise RuntimeError("nominal prefix produced a nonzero residual")
                before_count = self.prefix_ticks
                step = self.core.step(raw)
                self.prefix_decisions += 1
                last_info = self._verified_prefix_step(step, raw, before_tick, before_time, before_phase)
                if self.prefix_ticks-before_count != last_info["physics_ticks"]:
                    raise RuntimeError("prefix tick observer did not see every executed physical tick")
                # Preserve real physical success evidence, but never label an
                # initialization terminal as a FULL/SUFFIX PPO success sample.
                evidence = {key: last_info[key] for key in (
                    "phase_id", "end_phase_id", "physics_tick", "physics_ticks", "sim_time_s",
                    "raw_policy_action_full12", "applied_action_full12", "projected_residual_full12",
                    "actual_drive_target_full12", "actuator_target_effect_audit",
                    "actuator_target_effect_audit_ticks", "actuator_target_effect_audit_summary",
                    "no_in_episode_state_writes_verified", "termination_reason", *WRITE_COUNTERS)}
                self.sink({**jsonable(evidence), "kind": "checkpoint_prefix_decision", "policy_credit": False,
                    "task_result_scope": "checkpoint_policy_initialization_excluded",
                    "physical_task_success_during_initialization": last_info.get("task_success") is True,
                    "full_task_success": False})
                if step.terminated:
                    miss = {"reason": "physical_prefix_terminal", "termination_reason": last_info["termination_reason"],
                            "physical_task_success_during_initialization": last_info.get("task_success") is True,
                            "physics_tick": self.frame.physics_tick, "actual_phase": self.frame.state_id}
                    break
                phase = self.frame.state_id
                if target_entry is None and phase == self.request.target_phase:
                    target_entry = index
                if target_entry is not None and phase != self.request.target_phase:
                    miss = {"reason": "requested_phase_ended_before_offset", "actual_phase": phase,
                            "target_first_observed_decision": target_entry, "physics_tick": self.frame.physics_tick}
                    break
                if target_entry is None and int(phase[1:]) > int(self.request.target_phase[1:]):
                    miss = {"reason": "target_not_observed_at_decision_end", "actual_phase": phase,
                            "physics_tick": self.frame.physics_tick}
                    break
                if target_entry is not None and index-target_entry >= self.request.teacher_offset_decisions:
                    break
            else:
                miss = {"reason": "bounded_prefix_exhausted"}
        finally:
            self._initializing = False
            self.roll_in_wall_time_s += perf_counter()-started
        attempt = {"request": self.request.as_dict(), "prefix_decisions": self.prefix_decisions-first_decision,
                   "prefix_physics_ticks": self.prefix_ticks-first_tick, "accepted": miss is None,
                   "miss": miss, "target_first_observed_decision": target_entry}
        self.attempts.append(attempt)
        self.sink({"kind": "checkpoint_prefix_result", "policy_credit": False, **jsonable(attempt)})
        if miss is not None:
            self._fresh_reset(seed)  # Exactly one fallback; no recursive retry.
            last_info = {}
            self.start_record = {"mode": "fresh_P01_fallback", "from_P01_current_policy": True,
                                 "prefix_miss": miss}
        else:
            scope = (RESULT_SCOPE if self.request.source == "frozen_checkpoint_policy"
                     else "successful_nominal_initialized_suffix")
            self.start_record = {"mode": scope, "from_P01_current_policy": False,
                                 "target_first_observed_decision": target_entry,
                                 "requested_phase_still_active_at_credit": True}
        self.start_record.update(requested_phase=self.request.target_phase, actual_phase=self.frame.state_id,
            physics_tick=self.frame.physics_tick, sim_time_s=self.frame.sim_time_s,
            remaining_task_time_s=self.frame.info["semantic_task"]["remaining_task_time_s"],
            prefix_policy_provenance=_json_copy(self._bound_provenance))
        self._credit_return_offset = float(last_info.get("episode_return", 0.))
        self._episode_credit_decisions = 0
        self._start_reference = {key: self.start_record[key] for key in (
            "mode", "requested_phase", "actual_phase", "physics_tick", "sim_time_s",
            "from_P01_current_policy", "requested_phase_still_active_at_credit") if key in self.start_record}
        self._start_reference["prefix_attempt_index"] = len(self.attempts)-1
        self.sink({"kind": "policy_credit_start", "policy_credit": False, "start": jsonable(self.start_record)})
        self._credit_open = True
        return tuple(self.core.observation)

    def step(self, raw):
        step = super().step(raw)
        info = dict(step.info)
        info["prefix_checkpoint_policy_data_in_ppo_storage"] = False
        info["task_result_scope"] = ("full_task" if self.start_record["from_P01_current_policy"]
                                     else self.start_record["mode"])
        return replace(step, info=info)

    def telemetry_summary(self):
        return {"reset_sampling": sampling_label(self.request), "decisions": self.credited_decisions,
                "physics_ticks": self.credited_ticks, "phase_decisions": dict(self._credited_phase_decisions),
                "terminations": dict(self._credited_terminations), "prefix_behavior_decisions": self.prefix_decisions,
                "prefix_physics_ticks": self.prefix_ticks, "prefix_attempts": jsonable(self.attempts),
                "prefix_policy_provenance": _json_copy(self._bound_provenance),
                "reset_wall_time_s": self.reset_wall_time_s, "roll_in_wall_time_s": self.roll_in_wall_time_s,
                "physical_core_including_prefix": self.core.telemetry_summary()}


class CheckpointPolicyPrefixRslAdapter(PrefixRslAdapter):
    """Bootstrap observations may construct/load RSL; sampling awaits binding."""
    def __init__(self, core, *, seed, device="cuda:0", evidence_sink, request):
        if not isinstance(request, CheckpointPolicyPrefixRequest):
            raise ValueError("validated checkpoint prefix request required")
        credit = _CheckpointPolicyCreditCore(core, request=request, evidence_sink=evidence_sink)
        SemanticRslAdapter.__init__(self, credit, seed=seed, device=device)
        self.cfg.update(semantic_version="v3", reset_sampling=sampling_label(request),
            prefix_request=request.as_dict(), prefix_policy_provenance=None,
            teacher_data_in_ppo_storage=False, prefix_checkpoint_policy_data_in_ppo_storage=False,
            external_window_truncation_supported=False)

    def install_prefix_policy(self, policy_callable, provenance_dict):
        observation = self.core.install(policy_callable, provenance_dict, seed=self.seed)
        self._observation = tuple(observation)
        self.cfg["prefix_policy_provenance"] = _json_copy(self.core._bound_provenance)
        return dict(self.core.start_record)

    def step(self, actions):
        if self.core._policy is None or not self.core._credit_open:
            raise RuntimeError("install prefix policy before PPO collection")
        if (json.dumps(_provenance(self.cfg.get("prefix_policy_provenance")), sort_keys=True)
                != json.dumps(self.core._bound_provenance, sort_keys=True)
                or json.dumps(self.cfg["prefix_request"], sort_keys=True) != json.dumps(self.core.request.as_dict(), sort_keys=True)
                or self.cfg["reset_sampling"] != sampling_label(self.core.request)):
            raise RuntimeError("checkpoint prefix curriculum binding changed")
        return super().step(actions)

    def telemetry_summary(self):
        result = SemanticRslAdapter.telemetry_summary(self)
        suffix = sum(row["task_success"] and not row["terminal_info"]["curriculum_start"]["from_P01_current_policy"]
                     for row in self.completed_episodes)
        result.update(reset_sampling=self.cfg["reset_sampling"], checkpoint_policy_initialized_task_success_count=suffix,
                      success_count=result["success_count"]-suffix,
                      success_count_scope="fresh_P01_current_policy_only")
        return result
