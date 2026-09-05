"""Task semantics independent of the frozen FSM's imitation guards.

No Isaac imports, legacy-controller state mutation, Recording observations, or
clock-derived completion. The same live TaskEvaluator can accompany A, B and C.
"""
from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import yaml

from wlr50_clean.fsm.controller import ControllerEvent, ControllerFrame
from wlr50_clean.fsm.motion_executor import MotionExecutor
from wlr50_clean.fsm.state_spec import Lifecycle, load_fsm_spec
from wlr50_clean.fsm.task_result import TaskResult, TaskTermination
from wlr50_clean.infrastructure.command_batch import (
    FULL12_ORDER, SERVO_ORDER, WHEEL_ORDER, Full12Command, servo_limits_deg,
)
from wlr50_clean.reference.motion_contract import load_motion_contract

DEFAULT_TASK_SPEC_PATH = Path(__file__).resolve().parents[3] / "configs/ppo_semantic_v2/stage_task_spec.yaml"
LEG_ORDER = ("FL", "FR", "RL", "RR")
PHASE_IDS = tuple(f"P{i:02d}" for i in range(1, 14))
GOAL_FEATURE_KEYS = tuple(
    f"{leg}_{feature}" for leg in LEG_ORDER
    for feature in ("clearance_m", "front_distance_m", "load_fraction")
) + ("support_count", "body_forward_m", "body_linear_speed_m_s",
     "body_angular_speed_rad_s", "maximum_wheel_speed_rad_s")
ZERO12 = (0.0,) * 12


class SemanticObservationError(ValueError):
    """Missing/unverified physical data is an interface failure, not feasibility."""


def _get(value: Any, key: str, default: Any = None) -> Any:
    return value.get(key, default) if isinstance(value, Mapping) else getattr(value, key, default)


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise SemanticObservationError(f"{name} must be a finite measurement")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SemanticObservationError(f"missing finite measurement: {name}") from exc
    if not math.isfinite(result):
        raise SemanticObservationError(f"nonfinite measurement: {name}")
    return result


def _vector(value: Any, size: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, (tuple, list)) or len(value) != size:
        raise SemanticObservationError(f"{name} requires {size} measured components")
    return tuple(_number(item, name) for item in value)


def _norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(v * v for v in values))


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))


def load_task_spec(path: Path | str = DEFAULT_TASK_SPEC_PATH) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        spec = yaml.safe_load(stream)
    if not isinstance(spec, dict) or spec.get("schema") != "wlr50_clean.semantic_stage_task_spec.v2":
        raise ValueError("invalid semantic task schema")
    if spec.get("rear_leg_order") != "RR_FIRST" or tuple(spec.get("stages", {})) != PHASE_IDS:
        raise ValueError("semantic task must contain ordered P01-P13 and RR_FIRST")
    if spec.get("physics_hz") != 120.0 or spec.get("decision_hz") != 15.0:
        raise ValueError("semantic timing must remain 120/15 Hz")
    if not 0 < float(spec["episode_maximum_duration_s"]) <= 200:
        raise ValueError("semantic task horizon must be at most 200 seconds")
    required = {"purpose", "valid_start_conditions", "goal_features", "completion_predicates",
                "progress_potential", "allowed_action_channels", "physical_limits",
                "stall_diagnostic", "maximum_task_duration", "next_phase", "active_leg"}
    predicates = {"physical_valid", "whole_task_success", "rear_approach"} | {
        f"{kind}_{leg}" for kind in ("placed", "lifted", "clear", "approach", "workspace", "support", "load_ready")
        for leg in LEG_ORDER
    }
    for index, phase in enumerate(PHASE_IDS):
        row = spec["stages"][phase]
        if not required <= row.keys():
            raise ValueError(f"{phase} lacks required semantic fields")
        if row["next_phase"] != (PHASE_IDS[index + 1] if index < 12 else "SUCCESS"):
            raise ValueError("semantic graph must be forward-only; repeated stage credit is forbidden")
        if not row["completion_predicates"] or not row["valid_start_conditions"]:
            raise ValueError("entry and completion require physical predicates")
        for key in ("completion_predicates", "valid_start_conditions", "goal_features"):
            if not set(row[key]) <= predicates:
                raise ValueError(f"unknown semantic predicate in {phase}")
        if not row["allowed_action_channels"] or not set(row["allowed_action_channels"]) <= set(FULL12_ORDER):
            raise ValueError(f"invalid action channels in {phase}")
        if not 0 < _number(row["maximum_task_duration"], "stage duration") <= 200:
            raise ValueError("invalid task duration")
    # P10 prepares workspace: workspace cannot be required before entering it.
    if any("workspace" in item for item in spec["stages"]["P10"]["valid_start_conditions"]):
        raise ValueError("P10 must create, not presuppose, RL workspace")
    return spec


class TaskEvaluator:
    """Independent episode-local measured lift -> crossing -> placement history.

    Initializing this class does not import snapshot success latches. Curriculum
    history must be established by real prefix observations; no history setter
    is exposed. Invalid geometry/contact cannot be replaced by knee numbers.
    """

    def __init__(self, task_spec_path: Path | str = DEFAULT_TASK_SPEC_PATH, *, spec: Mapping[str, Any] | None = None):
        self.spec = dict(spec) if spec is not None else load_task_spec(task_spec_path)
        self._last_tick: int | None = None
        self._last_time: float | None = None
        self._stable_since: float | None = None
        self._samples = {leg: deque() for leg in LEG_ORDER}
        self._air_count = dict.fromkeys(LEG_ORDER, 0)
        self._top_count = dict.fromkeys(LEG_ORDER, 0)
        self._history = {key: dict.fromkeys(LEG_ORDER, False) for key in ("active_lift", "front_edge_crossed", "placed")}
        self._event_ticks = {key: {} for key in self._history}
        self._snapshot: dict[str, Any] = {"valid": False, "success": False, "termination_reason": None,
            "reason": "no live observation", "goal_features": dict.fromkeys(GOAL_FEATURE_KEYS, 0.0)}
        self._failure: str | None = None
        self._failure_reason = ""

    @property
    def snapshot(self) -> dict[str, Any]:
        return {**self._snapshot, "goal_features": dict(self._snapshot["goal_features"]),
            "history": {**{k: dict(v) for k, v in self._history.items()},
                        "event_ticks": {k: dict(v) for k, v in self._event_ticks.items()}}}

    def _fail(self, result: TaskResult, reason: str) -> None:
        if self._failure is None:
            self._failure, self._failure_reason = result.value, reason

    def observe(self, observation: Any) -> dict[str, Any]:
        tick = _get(observation, "physics_tick")
        if isinstance(tick, bool) or not isinstance(tick, int) or tick < 0:
            raise SemanticObservationError("physics_tick must be a nonnegative integer")
        now = _number(_get(observation, "simulation_time_s"), "simulation time")
        if self._last_tick == tick:
            if self._last_time != now:
                raise SemanticObservationError("same observation tick has inconsistent time")
            return self.snapshot
        if self._last_tick is not None and (tick != self._last_tick + 1 or not math.isclose(now-self._last_time,1/120.,rel_tol=0.,abs_tol=1e-9)):
            raise SemanticObservationError("task history requires contiguous advancing physical observations")
        if _get(observation, "all_finite") is not True:
            self._fail(TaskResult.SAFETY_ABORT, "nonfinite authoritative observation")
            self._snapshot.update(valid=False, success=False, termination_reason=self._failure, reason=self._failure_reason)
            self._last_tick, self._last_time = tick, now
            return self.snapshot
        base = _get(observation, "base")
        base_position = _vector(_get(base, "position_w_m"), 3, "base position")
        base_linear = _vector(_get(base, "linear_velocity_w_m_s"), 3, "base linear velocity")
        base_angular = _vector(_get(base, "angular_velocity_w_rad_s"), 3, "base angular velocity")
        _vector(_get(base, "orientation_wxyz"), 4, "base quaternion")
        gravity = _vector(_get(_get(observation, "imu"), "projected_gravity_b"), 3, "gravity")
        obstacle = _get(observation, "obstacle")
        front, top = (_number(_get(obstacle, key), key) for key in ("front_x_m", "top_z_m"))
        back, left, right = (_number(_get(obstacle,key),key) for key in ("back_x_m","left_y_m","right_y_m"))
        if back <= front or left <= right:
            raise SemanticObservationError("invalid measured obstacle plane ordering")
        joints, wheels, contacts = (_get(observation, key) for key in ("joints", "wheels", "contacts"))
        if not all(isinstance(v, Mapping) for v in (joints, wheels, contacts)):
            raise SemanticObservationError("full joint/wheel/exact-contact measurements are required")
        positions = {}
        for name in SERVO_ORDER:
            joint = joints.get(name)
            positions[name] = _number(_get(joint, "position_deg"), name)
            _number(_get(joint, "velocity_deg_s"), f"{name} velocity")
            lo, hi = servo_limits_deg(name)
            if not lo <= positions[name] <= hi:
                self._fail(TaskResult.SAFETY_ABORT, f"hard joint limit: {name}")
        collision = _get(_get(observation, "body_collision"), "detected")
        if not isinstance(collision, bool):
            raise SemanticObservationError("authoritative body collision status is required")
        if collision:
            self._fail(TaskResult.TASK_FAILURE_BODY_COLLISION, "central body/obstacle collision")
        if base_position[2] < .015 or base_position[2] > 1 or _norm(base_linear) > 5 or _norm(base_angular) > 20 or gravity[2] > -.30:
            self._fail(TaskResult.SAFETY_ABORT, "fall or physics explosion")
        hist_cfg, geo = self.spec["history"], self.spec["geometry"]
        current = {}; forces = {}; speeds = []; commands = []
        for index, leg in enumerate(LEG_ORDER):
            wheel = wheels.get(WHEEL_ORDER[index]); body_name = _get(wheel, "body_name")
            if _get(wheel, "geometry_verified") is not True:
                raise SemanticObservationError(f"unverified {leg} wheel geometry")
            center = _vector(_get(wheel, "center_w_m"), 3, f"{leg} center")
            bottom = _vector(_get(wheel, "bottom_w_m"), 3, f"{leg} bottom")
            speeds.append(_number(_get(wheel, "velocity_rad_s"), f"{leg} wheel speed"))
            commands.append(_number(_get(wheel, "command_rad_s"), f"{leg} wheel command"))
            contact = contacts.get(body_name); ground = _get(contact, "ground"); obstacle_pair = _get(contact, "obstacle")
            for pair in (ground, obstacle_pair):
                if _get(pair, "pair_verified") is not True or not isinstance(_get(pair, "active"), bool):
                    raise SemanticObservationError(f"{leg} requires verified exact ground and obstacle contact pairs")
                _number(_get(pair, "normal_force_n"), f"{leg} pair normal force")
            ground_active, top_active = _get(ground, "active"), _get(obstacle_pair, "active")
            force = sum(max(0., _get(pair, "normal_force_n")) for pair in (ground, obstacle_pair) if _get(pair, "active"))
            forces[leg] = force
            air = not ground_active and not top_active
            self._air_count[leg] = self._air_count[leg] + 1 if air else 0
            samples = self._samples[leg]
            samples.append((now, bottom[2], positions[SERVO_ORDER[index*2]], positions[SERVO_ORDER[index*2+1]]))
            while len(samples) > 1 and now - samples[0][0] > hist_cfg["window_s"]:
                samples.popleft()
            gain = max(v[1] for v in samples) - min(v[1] for v in samples)
            movement = sum(abs(b[j]-a[j]) for a,b in zip(samples, list(samples)[1:]) for j in (2,3))
            distance = center[0] - front
            lift = (hist_cfg["near_front_min_m"] <= distance <= hist_cfg["near_front_max_m"]
                and self._air_count[leg] >= hist_cfg["minimum_air_samples"]
                and gain >= hist_cfg["minimum_lift_gain_m"] and movement >= hist_cfg["minimum_joint_motion_deg"])
            if lift and not self._history["active_lift"][leg]:
                self._history["active_lift"][leg] = True; self._event_ticks["active_lift"][leg] = tick
            if distance >= 0 and not self._history["front_edge_crossed"][leg]:
                if not self._history["active_lift"][leg]:
                    self._fail(TaskResult.TASK_FAILURE_WHEEL_ONLY_CLIMB, f"{leg} crossed front without measured active lift")
                elif leg == "RL" and not self._history["placed"]["RR"]:
                    self._fail(TaskResult.INCOMPLETE_CONTROLLER_BLOCKED, "RR_FIRST order violated: RL crossed before RR placement")
                else:
                    self._history["front_edge_crossed"][leg] = True; self._event_ticks["front_edge_crossed"][leg] = tick
            xy_tolerance = geo["xy_measurement_tolerance_m"]
            within_lateral_span = right-xy_tolerance <= center[1] <= left+xy_tolerance
            within_top_xy = within_lateral_span and front-xy_tolerance <= center[0] <= back+xy_tolerance
            top_geometry = within_top_xy and geo["top_gap_min_m"] <= bottom[2]-top <= geo["top_gap_max_m"]
            loaded = bool(top_active and top_geometry and distance >= 0)
            self._top_count[leg] = self._top_count[leg]+1 if loaded else 0
            if self._history["front_edge_crossed"][leg] and self._top_count[leg] >= hist_cfg["minimum_top_samples"] and not self._history["placed"][leg]:
                self._history["placed"][leg] = True; self._event_ticks["placed"][leg] = tick
            current[leg] = {"front_distance_m": distance, "clearance_m": bottom[2]-top,
                "top_geometry": top_geometry, "top_contact": loaded, "air": air,
                "recent_joint_motion_deg": movement, "recent_clearance_gain_m": gain,
                "consecutive_air_samples": self._air_count[leg], "consecutive_top_samples": self._top_count[leg],
                "within_top_xy": within_top_xy, "within_lateral_span": within_lateral_span,
                "support": force >= self.spec["support"]["force_noise_floor_n"]}
        total_force = sum(forces.values())
        features = {}
        for leg in LEG_ORDER:
            fraction = forces[leg]/total_force if total_force > 0 else 0.
            current[leg]["load_fraction"] = fraction
            features.update({f"{leg}_clearance_m": current[leg]["clearance_m"],
                             f"{leg}_front_distance_m": current[leg]["front_distance_m"],
                             f"{leg}_load_fraction": fraction})
        features.update(support_count=float(sum(v["support"] for v in current.values())),
                        body_forward_m=base_position[0]-front, body_linear_speed_m_s=_norm(base_linear),
                        body_angular_speed_rad_s=_norm(base_angular), maximum_wheel_speed_rad_s=max(map(abs,speeds)))
        final = self.spec["final"]
        base_region = front+final["minimum_body_forward_m"] <= base_position[0] <= back+geo["xy_measurement_tolerance_m"] and right-geo["xy_measurement_tolerance_m"] <= base_position[1] <= left+geo["xy_measurement_tolerance_m"]
        final_region = all(v["top_geometry"] and v["front_distance_m"] >= final["minimum_rear_wheel_forward_m"] for v in current.values()) and base_region
        home_error = max(abs(positions[name]-target) for name,target in zip(SERVO_ORDER,final["home_servo_pose_deg"]))
        controlled = (_norm(base_linear) <= final["maximum_body_linear_speed_m_s"]
            and _norm(base_angular) <= final["maximum_body_angular_speed_rad_s"]
            and max(map(abs,speeds)) <= final["maximum_wheel_speed_rad_s"]
            and max(map(abs,commands)) <= final["maximum_commanded_wheel_speed_rad_s"]
            and all(abs(positions[name]-target) <= final["home_tolerance_deg"] for name,target in zip(SERVO_ORDER,final["home_servo_pose_deg"])))
        current_support = sum(v["support"] and v["top_contact"] for v in current.values()) >= self.spec["support"]["minimum_other_supports"]
        eligible = self._failure is None and all(self._history["placed"].values()) and final_region and controlled and current_support
        self._stable_since = (now if self._stable_since is None else self._stable_since) if eligible else None
        success = self._stable_since is not None and now-self._stable_since+1e-12 >= final["stable_duration_s"]
        self._snapshot = {"valid": True, "success": success, "termination_reason": self._failure,
            "reason": self._failure_reason, "goal_features": features, "current_legs": current,
            "final_region_valid": final_region, "final_controlled": controlled, "final_support_available": current_support,
            "home_maximum_servo_error_deg": home_error, "maximum_commanded_wheel_speed_rad_s": max(map(abs,commands)),
            "final_stable_for_s": 0. if self._stable_since is None else now-self._stable_since,
            "source": "current_episode_live_joint_geometry_exact_contact_history",
            "physics_tick": tick, "simulation_time_s": now}
        self._last_tick, self._last_time = tick, now
        return self.snapshot

    observe_and_update = observe


class TaskStageSupervisor:
    """Forward-only semantic stages; motion endpoints never veto exploration."""
    def __init__(self, task_spec_path: Path | str = DEFAULT_TASK_SPEC_PATH, *,
                 evaluator: TaskEvaluator | None = None, initial_stage_id: str = "P01"):
        self.spec = load_task_spec(task_spec_path)
        if initial_stage_id not in PHASE_IDS:
            raise ValueError("unknown initial task stage")
        self.evaluator = evaluator or TaskEvaluator(spec=self.spec)
        self.stage_id = initial_stage_id
        self.completed_stage_ids: list[str] = []
        self.transition_evidence: list[dict[str, Any]] = []
        self.stage_started_s: float | None = None
        self.episode_started_s: float | None = None
        self.termination_reason: str | None = None
        self._snapshot: dict[str, Any] = {}
        self._last_observation_tick: int | None = None
        self._progress_samples: deque = deque()

    def predicate(self, name: str, evaluation: Mapping[str, Any]) -> float:
        if name == "physical_valid":
            return float(evaluation["valid"] and evaluation["termination_reason"] is None)
        if name == "whole_task_success":
            if evaluation["success"]: return 1.
            final=self.spec["final"]; features=evaluation["goal_features"]
            history_fraction=sum(evaluation["history"]["placed"].values())/4.
            forward=min(_clip(features["body_forward_m"]/final["minimum_body_forward_m"]),
                        min(_clip(features[f"{leg}_front_distance_m"]/final["minimum_rear_wheel_forward_m"]) for leg in LEG_ORDER))
            stop=sum((_clip(1.-features["maximum_wheel_speed_rad_s"]/final["maximum_wheel_speed_rad_s"]),
                      _clip(1.-features["body_linear_speed_m_s"]/final["maximum_body_linear_speed_m_s"]),
                      _clip(1.-features["body_angular_speed_rad_s"]/final["maximum_body_angular_speed_rad_s"]),
                      _clip(1.-evaluation["home_maximum_servo_error_deg"]/final["home_tolerance_deg"]))) / 4.
            settle=_clip(evaluation["final_stable_for_s"]/final["stable_duration_s"])
            return min(.99,.4*history_fraction+.3*forward+.2*stop+.1*settle)
        if name == "rear_approach":
            return min(self.predicate("workspace_RL", evaluation), self.predicate("workspace_RR", evaluation))
        kind, leg = name.rsplit("_", 1)
        history = evaluation["history"]
        current = evaluation["current_legs"][leg]; geo = self.spec["geometry"]
        if kind == "lifted":
            if history["active_lift"][leg]: return 1.
            cfg = self.spec["history"]
            # Dense measurements provide a gradient of task progress, but only
            # the joint+clearance+AIR chronology can produce completion (=1).
            return .99 * ( _clip(current["recent_clearance_gain_m"]/cfg["minimum_lift_gain_m"])
                + _clip(current["recent_joint_motion_deg"]/cfg["minimum_joint_motion_deg"])
                + _clip(current["consecutive_air_samples"]/cfg["minimum_air_samples"]) ) / 3.
        if kind == "placed":
            if history["placed"][leg]: return 1.
            lift = self.predicate(f"lifted_{leg}", evaluation)
            crossing = 1. if history["front_edge_crossed"][leg] else _clip(1.+current["front_distance_m"]/.20)
            captured = (.5 * float(current["top_geometry"])
                + .5 * _clip(current["consecutive_top_samples"]/self.spec["history"]["minimum_top_samples"]))
            return min(.99, .35*lift + .35*crossing + .30*captured)
        support = sum(v["support"] for key,v in evaluation["current_legs"].items() if key != leg)
        support_available = support >= self.spec["support"]["minimum_other_supports"]
        if kind == "support": return float(support_available)
        if kind == "load_ready":
            if not support_available: return 0.
            limit = self.spec["support"]["unloaded_leg_maximum_load_fraction"]
            return _clip((1.-current["load_fraction"])/(1.-limit))
        if kind in ("approach", "workspace"):
            if not current["within_lateral_span"]: return 0.
            lower, upper = geo[f"{kind}_min_m"], geo[f"{kind}_max_m"]
            x = current["front_distance_m"]
            return 1. if lower <= x <= upper else _clip(1.-min(abs(x-lower),abs(x-upper))/.25)
        if kind == "clear": return _clip(max(0.,current["clearance_m"])/geo["airborne_clearance_above_top_m"])
        raise ValueError(f"unknown task predicate {name}")

    def entry_report(self, stage_id: str, evaluation: Mapping[str, Any] | None = None) -> dict[str, Any]:
        ev = self.evaluator.snapshot if evaluation is None else evaluation
        conditions = self.spec["stages"][stage_id]["valid_start_conditions"]
        values = {name: self.predicate(name, ev) if ev["valid"] else 0. for name in conditions}
        return {"valid": all(v >= 1. for v in values.values()), "reasons": [k for k,v in values.items() if v < 1.], "values": values}

    def observe_and_update(self, observation: Any, *, sim_time_s: float | None = None) -> dict[str, Any]:
        evaluation = self.evaluator.observe(observation)
        if self._last_observation_tick == _get(observation, "physics_tick"):
            return dict(self._snapshot)
        now = _number(_get(observation,"simulation_time_s") if sim_time_s is None else sim_time_s,"task time")
        if self.stage_started_s is None: self.stage_started_s = now
        if self.episode_started_s is None: self.episode_started_s = now
        stage = self.spec["stages"][self.stage_id]
        entry = self.entry_report(self.stage_id, evaluation)
        goal_values = {name: self.predicate(name,evaluation) if evaluation["valid"] else 0. for name in stage["completion_predicates"]}
        progress = sum(goal_values.values())/len(goal_values)
        if evaluation["termination_reason"] is not None:
            self.termination_reason = evaluation["termination_reason"]
        # Each physical observation can credit at most one unique task. No loops,
        # label-only success, reference clock or endpoint participates here.
        if self.termination_reason is None and entry["valid"] and all(v >= 1. for v in goal_values.values()) and _get(observation,"physics_tick") % 8 == 0:
            previous = self.stage_id
            if previous not in self.completed_stage_ids:
                self.completed_stage_ids.append(previous)
            next_stage = stage["next_phase"]
            self.transition_evidence.append({"from_stage": previous, "to_stage": next_stage, "sim_time_s": now,
                "physics_tick": _get(observation,"physics_tick"), "reason": "current physical goal set satisfied",
                "entry": entry, "completion_values": goal_values, "history": evaluation["history"]})
            if next_stage == "SUCCESS": self.termination_reason = "SUCCESS"
            else:
                self.stage_id = next_stage; self.stage_started_s = now; self._progress_samples.clear()
                stage = self.spec["stages"][self.stage_id]; entry = self.entry_report(self.stage_id,evaluation)
                goal_values = {name:self.predicate(name,evaluation) for name in stage["completion_predicates"]}
                progress = sum(goal_values.values())/len(goal_values)
        age = now-self.stage_started_s; episode_age = now-self.episode_started_s
        if self.termination_reason is None and (age >= stage["maximum_task_duration"] or episode_age >= self.spec["episode_maximum_duration_s"]):
            self.termination_reason = TaskResult.INCOMPLETE_CONTROLLER_BLOCKED.value
        self._progress_samples.append((now,progress))
        stall = stage["stall_diagnostic"]
        while len(self._progress_samples)>1 and now-self._progress_samples[0][0]>stall["window_s"]:
            self._progress_samples.popleft()
        stalled = bool(self._progress_samples and now-self._progress_samples[0][0] >= stall["window_s"]-1/120
            and max(v for _,v in self._progress_samples)-min(v for _,v in self._progress_samples)<stall["minimum_potential_change"])
        histories=evaluation["history"]
        self._snapshot={"schema":"wlr50_clean.semantic_task.v2", "stage_id":self.stage_id,"purpose":stage["purpose"],
            "phase_progress":progress,"task_progress_potential":min(1.,(len(self.completed_stage_ids)+(0. if self.termination_reason=="SUCCESS" else progress))/13),
            "goal_features":evaluation["goal_features"],"history":histories,
            "active_lift_history":histories["active_lift"],"front_edge_crossed_history":histories["front_edge_crossed"],"placed_history":histories["placed"],
            "completed_stage_ids":list(self.completed_stage_ids),"success":self.termination_reason=="SUCCESS",
            "termination_reason":self.termination_reason,"entry_valid":entry["valid"],"entry_reasons":entry["reasons"],
            "completion_values":goal_values,"stage_age_s":age,"stage_elapsed_s":age,
            "remaining_task_time_s":max(0.,self.spec["episode_maximum_duration_s"]-episode_age),
            "substage":"CAPTURE" if progress >= .8 else ("TRANSFER" if self.stage_id in ("P01","P04","P08","P10","P11") else "EXECUTION"),
            "stall_diagnostic":stalled,"transition_evidence":list(self.transition_evidence),"physical_evaluator":evaluation}
        self._last_observation_tick = _get(observation, "physics_tick")
        return dict(self._snapshot)

    @property
    def snapshot(self) -> dict[str, Any]: return dict(self._snapshot)


class NominalMotionProvider:
    """Read-only compact actions as finite suggestions, without entry vetoes.

    The existing motion executor owns source waypoint/tracking scheduling only.
    A persistent nominal approaches these advisory targets under physical slew
    limits; it is never reset to a historical entry anchor. The first handoff
    sample retains both nominal and tracking. Adapter compensation is never
    reset. After the suggestion tail, PPO remains active under task deadlines.
    """
    def __init__(self, contract: Any, *, spec: Mapping[str, Any] | None = None):
        self.contract=contract; self.spec=dict(spec) if spec is not None else load_task_spec()
        self.physics_hz=float(contract.physics_hz)
        self.servo_rate_limit_deg_s=float(contract.servo_rate_limit_deg_s)
        self.nominal_full12=tuple(contract.phases[0].start_full12)
        self.state_id: str | None=None; self.elapsed_s=0.; self.endpoint_issued=False
        self.tracking_servo_names: tuple[str,...]=()
        self._source_motion=MotionExecutor(physics_hz=self.physics_hz,
            servo_rate_limit_deg_s=self.servo_rate_limit_deg_s,initial_full12=self.nominal_full12)

    def evaluate(self, stage: str | Mapping[str,Any], observation: Any=None) -> tuple[float,...]:
        stage_id=stage if isinstance(stage,str) else str(stage["stage_id"])
        phase=self.contract.phase(stage_id)
        handoff=self.state_id is not None and self.state_id!=stage_id
        if self.state_id!=stage_id:
            self.state_id=stage_id
            self._source_motion.start_phase(phase)
        source=self._source_motion.tick()
        self.elapsed_s=source.elapsed_s
        proposed=source.full12
        self.endpoint_issued=source.endpoint_issued
        if stage_id=="P13" and self.endpoint_issued:
            proposed=tuple(self.spec["final"]["home_servo_pose_deg"])+(0.,)*4
        if not handoff:
            rates=(self.spec["nominal"]["servo_handoff_rate_deg_s"],)*8+(self.spec["nominal"]["wheel_handoff_rate_rad_s2"],)*4
            self.nominal_full12=Full12Command.from_full12(tuple(old+max(-rate/self.physics_hz,min(rate/self.physics_hz,target-old)) for old,target,rate in zip(self.nominal_full12,proposed,rates))).clamped().to_full12()
            self.tracking_servo_names=source.tracking_servo_names
        return self.nominal_full12


class SemanticControllerAdapter:
    """Explicit startup replacement producing the established ControllerFrame."""
    def __init__(self, spec: Any, contract: Any, *, task_spec_path: Path | str = DEFAULT_TASK_SPEC_PATH):
        self.spec=spec; self.contract=contract
        self.supervisor=TaskStageSupervisor(task_spec_path)
        self.evaluator=self.supervisor.evaluator
        self.nominal_provider=NominalMotionProvider(contract,spec=self.supervisor.spec)
        self.motion=self.nominal_provider
        self.physics_tick=0; self.lifecycle=Lifecycle.EXECUTE_MOTION
        self.history:list[ControllerEvent]=[]; self.termination:TaskTermination|None=None
        self.first_blocker=None; self._last_time:float|None=None

    @classmethod
    def from_paths(cls, fsm_path: Path | str, motion_contract_path: Path | str, *, task_spec_path: Path | str=DEFAULT_TASK_SPEC_PATH):
        return cls(load_fsm_spec(Path(fsm_path)),load_motion_contract(Path(motion_contract_path)),task_spec_path=task_spec_path)

    @property
    def state(self): return SimpleNamespace(state_id=self.supervisor.stage_id)
    @property
    def phase(self): return self.contract.phase(self.supervisor.stage_id)
    @property
    def task_progress(self): return float(self.supervisor.snapshot.get("phase_progress",0.))
    @property
    def task_snapshot(self): return self.supervisor.snapshot

    def step(self, observation: Any, *, sim_time_s: float | None=None) -> ControllerFrame:
        now=self.physics_tick/120. if sim_time_s is None else float(sim_time_s)
        if self._last_time is not None and now <= self._last_time:
            raise SemanticObservationError("controller time must advance monotonically")
        old_events=len(self.supervisor.transition_evidence)
        task=self.supervisor.observe_and_update(observation,sim_time_s=now)
        command=self.nominal_provider.evaluate(task,observation)
        events=[]
        for row in self.supervisor.transition_evidence[old_events:]:
            event=ControllerEvent(now,row["from_stage"],Lifecycle.EXECUTE_MOTION.value,Lifecycle.DONE.value,row["reason"],row)
            events.append(event); self.history.append(event)
        self.lifecycle=Lifecycle.EXECUTE_MOTION if task["entry_valid"] else Lifecycle.WAIT_ENTRY
        if task["termination_reason"] is not None:
            self.lifecycle=Lifecycle.DONE
            self.termination=TaskTermination(TaskResult(task["termination_reason"]),task["stage_id"],self.lifecycle.value,now,
                "physical task success" if task["success"] else (task["physical_evaluator"].get("reason") or "semantic task duration exhausted"),task)
            command=command[:8]+(0.,)*4
        frame=ControllerFrame(self.physics_tick,now,task["stage_id"],self.lifecycle,command,self.physics_tick%8==0,True,False,
            self.nominal_provider.tracking_servo_names,ZERO12,ZERO12,{"mode":"semantic_no_reference_feedback","semantic_task":task},
            self.nominal_provider.endpoint_issued,self.termination,None,tuple(events))
        self._last_time=now; self.physics_tick+=1
        return frame
