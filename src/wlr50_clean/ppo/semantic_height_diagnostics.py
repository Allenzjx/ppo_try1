"""Optional read-only height/drive evidence; never an observation or evaluator.

Live body poses transform an independently loaded collider asset, not a cached
world AABB. Getter failures remain null and cannot stop physical execution.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
import math
from pathlib import Path
import sys

from wlr50_clean.sensing.geometry import BASE_BODY, WHEEL_BODIES, UsdCollisionBoundsProvider
from .semantic_physical_sensing import _body_local_point_array, _world_bounds_from_body_local_array
from .semantic_hip_mount_geometry import resolve_hip_mounts, measure_hip_mounts


def member(value, key, default=None):
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def plain(value, invalid=None, path=()):
    """Keep unknown/nonfinite distinct from zero, including nested tensor values."""
    invalid = [] if invalid is None else invalid
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return {str(k): plain(v, invalid, (*path, str(k))) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v, invalid, (*path, i)) for i, v in enumerate(value)]
    if isinstance(value, float) and not math.isfinite(value):
        invalid.append({"index": list(path), "reported": str(value)})
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported diagnostic value {type(value).__name__}")


def read_value(getter, source):
    try:
        invalid = []
        value = plain(getter(), invalid)
        return {"value": value, "source": source, "nonfinite_entries": invalid,
                "reason": "nonfinite entries replaced with null" if invalid else
                          "getter returned unavailable value" if value is None else None}
    except Exception as exc:
        return {"value": None, "source": source, "reason": f"{type(exc).__name__}: {exc}"}


def transform_point(point, position, quaternion):
    bounds = _world_bounds_from_body_local_array(_body_local_point_array([point]),
        position_w_m=position, orientation_wxyz=quaternion)
    if bounds is None:
        raise ValueError("invalid local point/current rigid-body pose")
    return list(bounds.minimum_m)


def lowest_collider_point(points, position, quaternion):
    """Return a real minimum-z mesh vertex, not (link.x, link.y, AABB.min.z)."""
    norm = math.sqrt(sum(float(x) ** 2 for x in quaternion))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("invalid live quaternion")
    w, x, y, z = (float(v) / norm for v in quaternion)
    vx, vy, vz = points[:, 0], points[:, 1], points[:, 2]
    tx, ty, tz = 2 * (y * vz - z * vy), 2 * (z * vx - x * vz), 2 * (x * vy - y * vx)
    world_z = position[2] + vz + w * tz + x * ty - y * tx
    index = int(world_z.argmin())
    local = [float(v) for v in points[index]]
    return {"body_local_point_m": local, "world_point_m": transform_point(local, position, quaternion),
            "vertex_index": index, "selection": "first minimum-z vertex of enabled collider mesh point union"}


def resolve_rr_mount(provider):
    """Read the authored USD joint frame on its parent link; never a COM frame."""
    from pxr import Usd, UsdPhysics
    root = provider.stage.GetPrimAtPath(provider.robot_prim_path)
    matches = [prim for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
               if prim.GetName() == "rear_right_hip" and prim.IsA(UsdPhysics.Joint)]
    if len(matches) != 1:
        raise ValueError(f"expected one RR hip USD joint, found {len(matches)}")
    joint = UsdPhysics.Joint(matches[0])
    targets = joint.GetBody0Rel().GetTargets()
    local = joint.GetLocalPos0Attr()
    if len(targets) != 1 or not local.HasAuthoredValueOpinion():
        raise ValueError("RR hip parent relationship/authored localPos0 unavailable")
    parent = str(targets[0])
    if not parent.startswith(provider.robot_prim_path + "/"):
        raise ValueError("RR hip parent outside this robot")
    return {"joint_prim_path": str(matches[0].GetPath()), "parent_prim_path": parent,
            "parent_body_name": targets[0].name, "local_pos0_m": [float(x) for x in local.Get()],
            "source": "USD joint body0/localPos0 plus live parent body_link pose"}


GETTERS = {
    "stiffness": "get_dof_stiffnesses", "damping": "get_dof_dampings",
    "armature": "get_dof_armatures", "friction_properties": "get_dof_friction_properties",
    "position_limits_rad": "get_dof_limits", "velocity_limits_rad_s": "get_dof_max_velocities",
    "effort_limits_Nm": "get_dof_max_forces",
}


class HeightDiagnostics:
    """Own only new evidence files; best-effort I/O must not change task result."""
    def __init__(self, root, backend):
        self.root, self.backend = Path(root), backend
        self.stream = None
        self.errors, self.last_tick, self.rows = [], None, 0
        self.provider, self.robot, self.mount = None, None, None
        self.hip_mounts = None
        self.assets, self.asset_errors = {}, {}
        try:
            self.stream = (self.root / "height_diagnostics.jsonl").open("x", encoding="utf-8")
        except Exception as exc:
            self.errors.append(f"open: {type(exc).__name__}: {exc}")

    def _clock(self):
        return {"episode_tick": getattr(self.backend, "_episode_tick", None),
                "controller_next_tick": getattr(getattr(self.backend, "_controller", None), "physics_tick", None),
                "reset_count": getattr(self.backend, "_reset_count", None)}

    def start(self, frame):
        before = self._clock()
        startup = {"schema": "wlr50_clean.height_diagnostic_startup.v1", "clock_before": before,
                   "frame_tick": frame.physics_tick, "read_only": True,
                   "physical_setters_resets_steps_updates": 0}
        try:
            self.robot = self.backend._adapter.robot
            geometry = self.backend._reader.geometry_backend
            original = getattr(geometry, "provider", None) or getattr(geometry, "bounds_provider", None)
            self.provider = UsdCollisionBoundsProvider(original.stage, robot_prim_path=original.robot_prim_path)
        except Exception as exc:
            self.errors.append(f"independent provider: {type(exc).__name__}: {exc}")
        startup["joint_names_native_order"] = read_value(lambda: list(self.robot.joint_names), "robot.joint_names")
        startup["body_names_native_order"] = read_value(lambda: list(self.robot.body_names), "robot.body_names")
        startup["drive_getters"] = {name: read_value(
            lambda method=method: getattr(self.robot.root_physx_view, method)(), f"root_physx_view.{method}")
            for name, method in GETTERS.items()}
        startup["drive_getter_semantics"] = "native DOF order; configured effective limits/gains, not measured motor torque"
        mount = read_value(lambda: resolve_rr_mount(self.provider), "USD RR hip joint localPos0")
        self.mount = mount["value"]
        startup["rr_hip_mount_definition"] = mount
        # Resolve the four installation frames from this actual stage only.
        # A failure is diagnostic N/A, never a physical execution condition.
        self.hip_mounts = resolve_hip_mounts(self.provider)
        startup["same_rigid_body_hip_mount_definitions"] = self.hip_mounts
        startup["modules"] = {name: getattr(sys.modules.get(type(obj).__module__), "__file__", None)
                              for name, obj in (("articulation", self.robot), ("diagnostics", self))}
        startup["clock_after"] = self._clock()
        startup["clock_unchanged"] = before == startup["clock_after"]
        startup["errors"] = list(self.errors)
        try:
            with (self.root / "height_diagnostics_startup.json").open("x", encoding="utf-8") as stream:
                json.dump(startup, stream, allow_nan=False, indent=2)
        except Exception as exc:
            self.errors.append(f"startup write: {type(exc).__name__}: {exc}")
        self.sample(frame, force=True)

    def _pose(self, body):
        index = list(self.robot.body_names).index(body)
        return (plain(self.robot.data.body_link_pos_w[0, index]),
                plain(self.robot.data.body_link_quat_w[0, index]))

    def _bounds(self, body):
        position, quaternion = self._pose(body)
        if body not in self.assets:
            if body in self.asset_errors:
                raise ValueError(self.asset_errors[body])
            try:
                self.provider.collision_bounds(body, body_position_w_m=position, body_orientation_wxyz=quaternion)
                points = _body_local_point_array(self.provider._body_local_points.get(body))
                if points is None:
                    raise ValueError("enabled collider body-local mesh points unavailable")
                self.assets[body] = points
            except Exception as exc:
                self.asset_errors[body] = f"{type(exc).__name__}: {exc}"
                raise
        bounds = _world_bounds_from_body_local_array(self.assets[body],
            position_w_m=position, orientation_wxyz=quaternion)
        if bounds is None:
            raise ValueError("current-pose bounds unavailable")
        return {"minimum_m": list(bounds.minimum_m), "maximum_m": list(bounds.maximum_m),
                "link_origin_w_m": position, "link_quat_wxyz": quaternion,
                "lowest_collider_point": lowest_collider_point(self.assets[body], position, quaternion),
                "point_count": len(self.assets[body]), "collider_paths": self.provider._collider_paths[body],
                "source": "independently_loaded_body_local_mesh_points_transformed_at_current_live_pose",
                "world_extent_cache_used": False}

    def _mount_world(self):
        if self.mount is None:
            raise ValueError("RR mount unresolved; see startup reason")
        position, quaternion = self._pose(self.mount["parent_body_name"])
        return transform_point(self.mount["local_pos0_m"], position, quaternion)

    def _efforts(self):
        result = {}
        for name, actuator in self.robot.actuators.items():
            implicit = bool(getattr(actuator, "is_implicit_model", False))
            result[name] = {"joint_names": list(actuator.joint_names), "is_implicit_model": implicit,
                "computed_effort_Nm": read_value(lambda a=actuator: a.computed_effort,
                    "implicit_PD_estimate_before_clip" if implicit else "explicit_actuator_model_computed_effort"),
                "applied_effort_Nm": read_value(lambda a=actuator: a.applied_effort,
                    "implicit_PD_estimate_after_clip" if implicit else "explicit_actuator_model_applied_effort"),
                "sampling_semantics": "most recent actuator compute, preceding dispatch; not recomputed at this post-step pose",
                "measured_physx_drive_torque_Nm": None,
                "measured_torque_reason": "no verified isolated drive-torque measurement API used"}
        return result

    def sample(self, frame, *, force=False, terminal=False):
        tick = frame.physics_tick
        if tick == self.last_tick or (not force and not terminal and tick % 8):
            return
        try:
            raw = frame.info.get("raw_observation")
            bounds = {body: read_value(lambda body=body: self._bounds(body), "fresh independent live collider bounds")
                      for body in (BASE_BODY, *WHEEL_BODIES)}
            base = bounds[BASE_BODY]["value"]
            row = {"schema": "wlr50_clean.height_diagnostic_sample.v1", "physics_tick": tick,
                "simulation_time_s": tick / 120., "phase": frame.state_id, "terminal_sample": terminal,
                "clock_before": self._clock(), "frame_raw_physics_tick": member(raw, "physics_tick"),
                "body_collision_minimum_z_w_m": None if base is None else base["minimum_m"][2],
                "body_collision_minimum_reason": bounds[BASE_BODY]["reason"], "fresh_collider_bounds": bounds,
                "recorded_evaluator_body_bounds": read_value(lambda: member(raw, "body_bounds_w_m"), "unchanged raw evaluator geometry"),
                "recorded_evaluator_geometry_pose_aware": member(raw, "geometry_pose_aware"),
                "rr_hip_mount_w_m": read_value(self._mount_world, "USD localPos0 transformed with live parent link pose"),
                "same_rigid_body_hip_mount_geometry": read_value(
                    lambda: measure_hip_mounts(self.hip_mounts, raw),
                    "all_four_actual_USD_body0_localPos0_with_same_tick_observed_base_link_pose"),
                "joint_position_native_rad": read_value(lambda: self.robot.data.joint_pos[0], "robot.data.joint_pos"),
                "joint_velocity_native_rad_s": read_value(lambda: self.robot.data.joint_vel[0], "robot.data.joint_vel"),
                "joints_canonical": read_value(lambda: member(raw, "joints"), "unchanged sensor canonical joint record"),
                "base": read_value(lambda: member(raw, "base"), "unchanged sensor base origin, NOT collider minimum"),
                "center_of_mass": read_value(lambda: member(raw, "center_of_mass"), "unchanged sensor CoM"),
                "actuator_effort_estimates": read_value(self._efforts, "last computed actuator buffers; never measured PhysX drive torque")}
            row["clock_after"] = self._clock()
            row["clock_unchanged"] = row["clock_before"] == row["clock_after"]
            if self.stream is not None:
                self.stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
                self.stream.flush()
                self.last_tick, self.rows = tick, self.rows + 1
        except Exception as exc:
            self.errors.append(f"sample {tick}: {type(exc).__name__}: {exc}")

    def close(self, frame=None):
        if frame is not None:
            self.sample(frame, force=True, terminal=True)
        try:
            if self.stream is not None:
                self.stream.close()
        except Exception as exc:
            self.errors.append(f"close: {type(exc).__name__}: {exc}")
        return {"schema": "wlr50_clean.height_diagnostic_receipt.v1", "rows": self.rows,
                "last_sample_tick": self.last_tick, "errors": list(self.errors),
                "read_only": True, "observation_or_evaluator_input": False,
                "stream_closed": self.stream is None or self.stream.closed}
