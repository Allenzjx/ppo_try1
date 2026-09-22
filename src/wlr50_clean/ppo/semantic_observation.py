"""Task-state actor features, independent of legacy reference/error observations."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from wlr50_clean.infrastructure.command_batch import SERVO_ORDER, WHEEL_ORDER
from wlr50_clean.infrastructure.servo_target_mapper import SERVO_TRACKING_FEEDBACK_INTERVAL_TICKS
from .observation_schema_v2 import _quat_rotate_inverse
from .semantic_transfer_roles import (
    LEGS as ROLE_LEGS, ROLE_OBSERVATION_LAYOUT, ROLE_OBSERVATION_GROUP,
    ROLE_OBSERVATION_BASE_DIM, ROLE_OBSERVATION_DIM, ROLE_OBSERVATION_FIELDS,
)
from .semantic_p05_capture_profile import (
    P05_CAPTURE_OBSERVATION_LAYOUT, P05_CAPTURE_OBSERVATION_DIM,
    P05_CAPTURE_ASSIST_GROUP, P05_CAPTURE_CONTINUATION_GROUP,
)

CONFIG_ROOT = Path(__file__).resolve().parents[3] / "configs" / "ppo_semantic_v2"
DEFAULT_OBSERVATION_SCHEMA = CONFIG_ROOT / "observation_schema.json"
STAGES = tuple(f"P{i:02d}" for i in range(1, 14))
LEGS = ("FL", "FR", "RL", "RR")
SUBSTAGES = ("TRANSFER", "CAPTURE", "EXECUTION")
GOAL_KEYS = tuple(f"{leg}_{key}" for leg in LEGS for key in
                  ("clearance_m", "front_distance_m", "load_fraction")) + (
    "support_count", "body_forward_m", "body_linear_speed_m_s",
    "body_angular_speed_rad_s", "maximum_wheel_speed_rad_s",
)
HISTORY_GROUPS = ("previous_raw_full12", "previous_residual_full12",
                  "previous_previous_residual_full12", "previous_applied_full12",
                  "previous_previous_applied_full12", "previous_nominal_full12")


class SemanticObservationError(ValueError):
    pass


class SemanticNonFiniteObservation(SemanticObservationError):
    pass


def field(value: Any, key: str) -> Any:
    try:
        return value[key] if isinstance(value, Mapping) else getattr(value, key)
    except (KeyError, AttributeError, TypeError) as exc:
        raise SemanticObservationError(f"missing live field {key}") from exc


def finite(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise SemanticObservationError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise SemanticNonFiniteObservation(f"{label} is nonfinite")
    return result


def vector(value: Sequence[float], size: int, label: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        raise SemanticObservationError(f"{label} must be a vector")
    try:
        result = tuple(finite(item, label) for item in value)
    except TypeError as exc:
        raise SemanticObservationError(f"{label} must be a vector") from exc
    if len(result) != size:
        raise SemanticObservationError(f"{label} needs {size} values")
    return result


def semantic_task(frame: Any) -> Mapping[str, Any]:
    task = field(field(frame, "info"), "semantic_task")
    if not isinstance(task, Mapping) or task.get("schema") != "wlr50_clean.semantic_task.v2":
        raise SemanticObservationError("semantic task metadata is required")
    if task.get("stage_id") not in STAGES or task["stage_id"] != field(frame, "state_id"):
        raise SemanticObservationError("semantic task and frame stage differ")
    completed = task.get("completed_stage_ids")
    continuation = task.get("capture_continuation", {})
    pending_mode = (isinstance(continuation, Mapping)
                    and continuation.get("mode") == "p05_hip_only_continuation_v1")
    valid_completed = (isinstance(completed, (list, tuple))
                       and tuple(completed) == tuple(p for p in STAGES if p in completed)
                       if pending_mode else isinstance(completed, (list, tuple))
                       and tuple(completed) == STAGES[:len(completed)])
    if not valid_completed:
        raise SemanticObservationError("completed stages must be a unique ordered prefix")
    for key in ("phase_progress", "task_progress_potential"):
        if not 0.0 <= finite(field(task, key), key) <= 1.0:
            raise SemanticObservationError(f"{key} outside [0,1]")
    if type(task.get("success")) is not bool:
        raise SemanticObservationError("task success must be boolean")
    if task.get("substage") not in SUBSTAGES:
        raise SemanticObservationError("unknown task substage")
    return task


def _quaternion(value: Sequence[float]) -> tuple[float, ...]:
    q = vector(value, 4, "quaternion")
    norm = math.sqrt(sum(x*x for x in q))
    if norm < 1.0e-12:
        raise SemanticObservationError("zero quaternion")
    return tuple(x/norm for x in q)


def _multiply(a: Sequence[float], b: Sequence[float]) -> tuple[float, ...]:
    w,x,y,z = a; v,i,j,k = b
    return (w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j,
            w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v)


def _rpy(q: Sequence[float]) -> tuple[float, float, float]:
    w,x,y,z = q
    return (math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y)),
            math.asin(max(-1.0, min(1.0, 2*(w*y-z*x)))),
            math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z)))


@dataclass(frozen=True)
class SemanticObservationSchema:
    groups: tuple[Mapping[str, Any], ...]
    fixed_chassis_to_body_wxyz: tuple[float, ...]
    maximum_task_duration_s: float
    clip: float
    path: Path
    transfer_role_features_version: str | None = None
    capture_assist_features_version: str | None = None

    @property
    def observation_layout(self) -> str | None:
        return self.capture_assist_features_version or self.transfer_role_features_version

    @property
    def dimension(self) -> int:
        return sum(int(row["size"]) for row in self.groups)

    def encode(self, groups: Mapping[str, Sequence[float]], *, normalized: bool = True) -> tuple[float, ...]:
        if tuple(groups) != tuple(row["name"] for row in self.groups):
            raise SemanticObservationError("feature groups differ from versioned schema")
        result = []
        for spec in self.groups:
            values = vector(groups[spec["name"]], int(spec["size"]), str(spec["name"]))
            scale = spec["scale"]
            scales = vector(scale, len(values), "scale") if isinstance(scale, list) else (finite(scale, "scale"),)*len(values)
            if any(x <= 0 for x in scales):
                raise SemanticObservationError("feature scales must be positive")
            result.extend(max(-self.clip, min(self.clip, x/s)) if normalized else x
                          for x,s in zip(values, scales, strict=True))
        return tuple(result)


def load_semantic_observation_schema(path: Path | str = DEFAULT_OBSERVATION_SCHEMA) -> SemanticObservationSchema:
    selected = Path(path).resolve()
    data = json.loads(selected.read_text(encoding="utf-8"))
    if data.get("schema") != "wlr50_clean.semantic_observation.v2" or data.get("dimension_policy") != "sum_feature_sizes":
        raise SemanticObservationError("invalid semantic observation schema")
    groups = tuple(data["feature_groups"])
    if not groups or len({x["name"] for x in groups}) != len(groups) or any(type(x["size"]) is not int or x["size"] <= 0 for x in groups):
        raise SemanticObservationError("invalid feature sizes/names")
    role_layout = data.get("transfer_role_features_version")
    capture_layout = data.get("capture_assist_features_version")
    role_groups = groups
    if capture_layout is not None:
        expected_tail = (
            {"name":P05_CAPTURE_ASSIST_GROUP,"size":12,"scale":1.0},
            {"name":P05_CAPTURE_CONTINUATION_GROUP,"size":5,"scale":1.0})
        if (capture_layout != P05_CAPTURE_OBSERVATION_LAYOUT or role_layout != ROLE_OBSERVATION_LAYOUT
                or groups[-2:] != expected_tail
                or sum(row["size"] for row in groups) != P05_CAPTURE_OBSERVATION_DIM):
            raise SemanticObservationError("capture layout must append exactly 17 declared state features after role372")
        role_groups = groups[:-2]
    elif any(row["name"] in (P05_CAPTURE_ASSIST_GROUP,P05_CAPTURE_CONTINUATION_GROUP) for row in groups):
        raise SemanticObservationError("capture state requires its explicit version marker")
    if role_layout is None:
        if any(row["name"] == ROLE_OBSERVATION_GROUP for row in groups):
            raise SemanticObservationError("role observation group requires its explicit layout marker")
    elif role_layout != ROLE_OBSERVATION_LAYOUT:
        raise SemanticObservationError("unknown transfer role observation layout")
    else:
        expected = {"name": ROLE_OBSERVATION_GROUP, "size": len(ROLE_LEGS)*len(ROLE_OBSERVATION_FIELDS), "scale": 1.0}
        if (role_groups[-1] != expected or type(role_groups[-1].get("scale")) not in (int, float)
                or sum(row["size"] for row in role_groups[:-1]) != ROLE_OBSERVATION_BASE_DIM
                or sum(row["size"] for row in role_groups) != ROLE_OBSERVATION_DIM):
            raise SemanticObservationError("role observation layout must append exactly 48 scale-one features after 324")
    duration, clip = finite(data["maximum_task_duration_s"], "timeout"), finite(data["clip"], "clip")
    if duration != 200.0 or clip <= 0.0:
        raise SemanticObservationError("semantic observation needs 200 second task horizon and positive clipping")
    return SemanticObservationSchema(groups, _quaternion(data["fixed_chassis_to_body_wxyz"]),
                                     duration, clip, selected, role_layout, capture_layout)


def transfer_role_observation_features(task: Mapping[str, Any]) -> tuple[float, ...]:
    """Expose current role consumers and window maturity, not the full deque.

    Missing optional rows and explicitly invalid rows have a validity bit of
    zero and unavailable zero fillers. A declared-valid malformed row is an
    observation error, not a valid zero transfer or support measurement.
    """
    roles = task.get("transfer_roles")
    if roles is None:
        roles = {}
    if not isinstance(roles, Mapping) or not set(roles) <= set(ROLE_LEGS):
        raise SemanticObservationError("invalid transfer role mapping")

    def number(value: Any, label: str) -> float:
        if type(value) not in (int, float):
            raise SemanticObservationError(f"{label} must be a number, not a boolean/string")
        return finite(value, label)

    def unit(value: Any, label: str, lower: float = 0.0) -> float:
        result = number(value, label)
        if not lower <= result <= 1.0:
            raise SemanticObservationError(f"{label} outside [{lower},1]")
        return result

    result = []
    for leg in ROLE_LEGS:
        role = roles.get(leg)
        if role is None:
            result.extend((0.0,)*len(ROLE_OBSERVATION_FIELDS))
            continue
        if not isinstance(role, Mapping) or type(role.get("valid")) is not bool:
            raise SemanticObservationError(f"{leg} role validity must be boolean")
        if not role["valid"]:
            result.extend((0.0,)*len(ROLE_OBSERVATION_FIELDS))
            continue
        values = [1.0]
        for key in ROLE_OBSERVATION_FIELDS[1:5]:
            values.append(unit(field(role, key), f"{leg}.{key}"))
        for key in ROLE_OBSERVATION_FIELDS[5:7]:
            value = field(role, key)
            if type(value) is not bool:
                raise SemanticObservationError(f"{leg}.{key} must be boolean")
            values.append(float(value))
        context = field(role, "transfer_direction_context")
        if not isinstance(context, Mapping):
            raise SemanticObservationError(f"{leg} transfer direction context must be a mapping")
        direction = field(context, "fixed_direction_world")
        if not isinstance(direction, (list, tuple)) or len(direction) != 3:
            raise SemanticObservationError(f"{leg} transfer direction must have three components")
        direction = tuple(unit(x, f"{leg}.fixed_direction_world", -1.0) for x in direction)
        if direction[2] != 0.0:
            raise SemanticObservationError(f"{leg} transfer direction must be planar")
        values.extend(direction[:2])
        values.append(unit(field(context, "short_support_continuity_fraction"), f"{leg}.short_support_continuity_fraction"))
        minimum = number(field(context, "minimum_evidence_s"), f"{leg}.minimum_evidence_s")
        if minimum <= 0.0:
            raise SemanticObservationError(f"{leg} minimum evidence time must be positive")
        for key in ("continued_response_duration_s", "window_s"):
            elapsed = number(field(context, key), f"{leg}.{key}")
            if elapsed < 0.0:
                raise SemanticObservationError(f"{leg}.{key} must be nonnegative")
            # Compare first: very long finite durations need not overflow a ratio.
            values.append(1.0 if elapsed >= minimum else elapsed/minimum)
        result.extend(values)
    return tuple(result)


@dataclass(frozen=True)
class SemanticObservationFrame:
    groups: Mapping[str, tuple[float, ...]]
    task: Mapping[str, Any]
    metrics: Mapping[str, Any]


class SemanticObservationBuilder:
    """Quaternion finite differences use actual tick time, never episode leveling."""
    def __init__(self, schema: SemanticObservationSchema):
        self.schema = schema
        self.reset()

    def reset(self) -> None:
        self.previous_time = None
        self.previous_rpy = None
        self.previous_omega = None

    def build(self, frame: Any, history: Mapping[str, Sequence[float]]) -> SemanticObservationFrame:
        info = field(frame, "info")
        task = semantic_task(frame)
        raw = field(info, "raw_observation")
        if field(raw, "all_finite") is not True:
            raise SemanticNonFiniteObservation("raw observation all_finite is false")
        base, imu, com, obstacle = (field(raw, key) for key in ("base", "imu", "center_of_mass", "obstacle"))
        q = _quaternion(field(base, "orientation_wxyz"))
        chassis_q = _quaternion(_multiply(q, self.schema.fixed_chassis_to_body_wxyz))
        rpy = _rpy(chassis_q)
        omega = _quat_rotate_inverse(q, field(base, "angular_velocity_w_rad_s"))
        velocity = _quat_rotate_inverse(q, field(base, "linear_velocity_w_m_s"))
        now = finite(field(frame, "sim_time_s"), "simulation time")
        valid = self.previous_time is not None
        rates, angular_acceleration = (0.0, 0.0), (0.0,)*3
        if valid:
            dt = now - self.previous_time
            if dt <= 0.0:
                raise SemanticObservationError("observation time must increase")
            rates = tuple(math.atan2(math.sin(a-b), math.cos(a-b))/dt
                          for a,b in zip(rpy[:2], self.previous_rpy[:2], strict=True))
            angular_acceleration = tuple((a-b)/dt for a,b in zip(omega, self.previous_omega, strict=True))
        base_position = vector(field(base, "position_w_m"), 3, "base position")
        planes = tuple(finite(field(obstacle, key), key) for key in
                       ("front_x_m", "back_x_m", "left_y_m", "right_y_m", "bottom_z_m", "top_z_m"))
        if not (planes[0] < planes[1] and planes[2] > planes[3] and planes[4] < planes[5]):
            raise SemanticObservationError("invalid obstacle planes")
        joints, wheels, contacts, bodies = (field(raw, key) for key in ("joints", "wheels", "contacts", "bodies"))
        forces, active, geometry, wheel_velocity, wheel_metrics = [], [], [], [], []
        for name in WHEEL_ORDER:
            wheel = wheels[name]
            if field(wheel, "geometry_verified") is not True:
                raise SemanticObservationError("wheel geometry is unverified")
            bottom = vector(field(wheel, "bottom_w_m"), 3, "wheel bottom")
            center = vector(field(wheel, "center_w_m"), 3, "wheel center")
            geometry.extend((bottom[0]-planes[0], bottom[0]-planes[1], bottom[2]-planes[5]))
            contact = contacts[str(field(wheel, "body_name"))]
            body = bodies[str(field(wheel, "body_name"))]
            wheel_v = vector(field(body, "linear_velocity_w_m_s"), 3, "wheel velocity")
            wheel_velocity.extend(wheel_v)
            row_forces, row_active, chatter = [], [], []
            for pair_name in ("ground", "obstacle"):
                pair = field(contact, pair_name)
                if field(pair, "pair_verified") is not True or type(field(pair, "active")) is not bool:
                    raise SemanticObservationError("contact pair unverified or active flag invalid")
                force = finite(field(pair, "normal_force_n"), "contact normal force")
                if force < 0.0:
                    raise SemanticObservationError("negative contact force")
                row_forces.append(force); row_active.append(bool(field(pair, "active")))
                sample_history = tuple(field(pair, "active_history"))
                chatter.append(sum(a != b for a,b in zip(sample_history, sample_history[1:]))/max(1,len(sample_history)-1))
            forces.extend(row_forces); active.extend(float(x) for x in row_active)
            speed = finite(field(wheel, "velocity_rad_s"), "wheel speed")
            forward_speed = _quat_rotate_inverse(q, wheel_v)[0]
            wheel_metrics.append({"force": sum(row_forces), "contact": any(row_active),
                                  "vertical_velocity": wheel_v[2], "chatter": max(chatter),
                                  "slip_speed": abs(0.04998999834060672*speed-forward_speed),
                                  "center": center})
        total_force = sum(forces)
        loads = tuple((forces[2*i]+forces[2*i+1])/total_force if total_force > 1e-9 else 0.0 for i in range(4))
        support = field(raw, "support")
        support_valid = bool(field(support, "valid"))
        margin = field(support, "signed_margin_m")
        mapper = field(info, "mapper_state_summary")
        goal = field(task, "goal_features")
        if set(goal) != set(GOAL_KEYS):
            raise SemanticObservationError("goal feature keys differ from supervisor contract")
        groups = {
            "stage_one_hot": tuple(float(task["stage_id"] == phase) for phase in STAGES),
            "substage_one_hot": tuple(float(task["substage"] == value) for value in SUBSTAGES),
            "task_progress": (float(task["phase_progress"]), float(task["task_progress_potential"])),
            "task_times": (now, finite(field(task,"remaining_task_time_s"),"remaining time"), finite(field(task,"stage_elapsed_s"),"stage elapsed")),
            "goal_features": tuple(finite(goal[key], key) for key in GOAL_KEYS),
            "actual_joint_position_deg": tuple(finite(field(joints[name],"position_deg"),name) for name in SERVO_ORDER),
            "actual_joint_velocity_deg_s": tuple(finite(field(joints[name],"velocity_deg_s"),name) for name in SERVO_ORDER),
            "actual_wheel_velocity_rad_s": tuple(finite(field(wheels[name],"velocity_rad_s"),name) for name in WHEEL_ORDER),
            "raw_body_orientation_wxyz": q, "chassis_orientation_wxyz": chassis_q,
            "projected_gravity_chassis": _quat_rotate_inverse(chassis_q,(0.0,0.0,-1.0)),
            "chassis_rpy_rad": rpy, "euler_roll_pitch_rate_rad_s": rates,
            "orientation_derivative_valid": (float(valid),), "body_linear_velocity": velocity,
            "body_angular_velocity": omega, "body_angular_acceleration": angular_acceleration,
            "imu_linear_acceleration_body": vector(field(imu,"linear_acceleration_b_m_s2"),3,"imu acceleration"),
            "com_position_relative_base": tuple(a-b for a,b in zip(vector(field(com,"position_w_m"),3,"com position"),base_position,strict=True)),
            "com_velocity_world": vector(field(com,"velocity_w_m_s"),3,"com velocity"),
            "obstacle_planes_relative_base": tuple(value-base_position[i//2] for i,value in enumerate(planes)),
            "wheel_geometry_relative_obstacle": tuple(geometry), "wheel_body_velocity": tuple(wheel_velocity),
            "wheel_pair_normal_force": tuple(forces), "wheel_pair_active": tuple(active),
            "wheel_load_fraction": loads,
            "support_diagnostics": (finite(field(support,"support_count"),"support count"), float(support_valid), 0.0 if margin is None else finite(margin,"support margin")),
        }
        for key in ("active_lift_history", "front_edge_crossed_history", "placed_history"):
            values = field(task,key)
            if set(values) != set(LEGS) or any(type(values[leg]) is not bool for leg in LEGS):
                raise SemanticObservationError(f"invalid physical history {key}")
            groups[key] = tuple(float(values[leg]) for leg in LEGS)
        groups["completed_stages"] = tuple(float(phase in task["completed_stage_ids"]) for phase in STAGES)
        groups["nominal_action_full12"] = vector(field(frame,"nominal_action_full12"),12,"nominal")
        groups["mapped_nominal_previous_dispatch_full12"] = vector(field(info,"mapped_nominal_full12"),12,"mapped nominal")
        for key in HISTORY_GROUPS:
            groups[key] = vector(field(history,key),12,key)
        for key in ("requested_servo_deg", "tracking_compensation_deg", "applied_drive_command_deg", "final_drive_servo_deg", "nominal_target_reached", "tracking_active", "retiring_stale_bias"):
            groups["mapper_"+key] = vector(field(mapper,key),8,key)
        groups["mapper_feedback_phase"] = (float(int(field(mapper,"feedback_tick")) % SERVO_TRACKING_FEEDBACK_INTERVAL_TICKS),)
        if self.schema.transfer_role_features_version == ROLE_OBSERVATION_LAYOUT:
            groups[ROLE_OBSERVATION_GROUP] = transfer_role_observation_features(task)
        if self.schema.capture_assist_features_version == P05_CAPTURE_OBSERVATION_LAYOUT:
            from .semantic_capture_assist import capture_assist_features
            groups[P05_CAPTURE_ASSIST_GROUP] = capture_assist_features(field(info,"capture_assist"))
            continuation = field(task,"capture_continuation")
            bits = [field(task,"fl_capture_pending"),field(task,"allow_capture_continuation"),
                    field(continuation,"scheduler_advanced_pending"),field(task,"p05_local_deadline_warning")]
            if any(type(value) is not bool for value in bits):
                raise SemanticObservationError("capture scheduling flags must be actual booleans")
            elapsed = finite(field(task,"capture_pending_elapsed_s"),"capture pending elapsed")
            if elapsed < 0:
                raise SemanticObservationError("capture pending elapsed must be nonnegative")
            groups[P05_CAPTURE_CONTINUATION_GROUP] = tuple(float(value) for value in bits)+(elapsed/200.0,)
        self.schema.encode(groups)
        mass = finite(field(com,"total_mass_kg"),"robot mass")
        if mass <= 0.0 or field(com,"valid") is not True:
            raise SemanticObservationError("robot mass/center of mass must be valid")
        self.previous_time, self.previous_rpy, self.previous_omega = now, rpy, omega
        return SemanticObservationFrame(groups, dict(task), {"sim_time_s":now,"rpy":rpy,
            "euler_roll_pitch_rate":rates,"body_angular_velocity":omega,
            "body_angular_acceleration":angular_acceleration,"wheels":tuple(wheel_metrics),
            "mass_kg":mass,"raw_orientation_wxyz":q,"obstacle_planes_world_m":planes})
