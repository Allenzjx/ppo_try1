"""Versioned physical sensing for all-stage PPO; frozen A modules stay unchanged."""
from __future__ import annotations

from dataclasses import dataclass, field, fields
import math
from typing import Any, Mapping

from wlr50_clean.sensing.contact_classifier import (
    BASE_BODY, GROUND_PAIR, OBSTACLE_PAIR, WHEEL_BODIES, ContactContractError,
    _norm, _vec3,
)
from wlr50_clean.sensing.geometry import (
    Aabb, ColliderGeometryCache, GeometrySnapshot, WheelGeometry,
    WHEEL_JOINT_TO_BODY, aabb_intersection_depth, obstacle_aabb,
    _optional_quat, _optional_vec3, _world_bounds_from_body_local_points,
)
from wlr50_clean.sensing.observation import Observation
from wlr50_clean.sensing.sensor_reader import (
    SensorReader, SensingContractError, create_live_sensing_backends,
)


@dataclass(frozen=True, slots=True)
class SemanticPhysicalObservation(Observation):
    body_bounds_w_m: Mapping[str, Any] = field(default_factory=dict)
    geometry_pose_aware: bool = True


_OBSERVATION_FIELDS = tuple(item.name for item in fields(Observation))


class SemanticColliderGeometry:
    """Cache body-local assets, transform their current physical pose per tick."""
    def __init__(self, legacy_geometry: ColliderGeometryCache):
        if not isinstance(legacy_geometry, ColliderGeometryCache):
            raise SensingContractError("all-stage sensing requires measured collider geometry")
        self.provider = legacy_geometry.bounds_provider
        self.obstacle = legacy_geometry.obstacle
        self.last_snapshot = None

    def sample(self, body_positions_w_m, body_orientations_wxyz=None):
        orientations = body_orientations_wxyz or {}
        bounds_by_body, paths_by_body, quality = {}, {}, []
        for body in (BASE_BODY, *WHEEL_BODIES):
            position = _optional_vec3(body_positions_w_m.get(body))
            orientation = _optional_quat(orientations.get(body))
            if position is None or orientation is None:
                quality.append(f"missing current pose for {body}")
                continue
            points = getattr(self.provider, "_body_local_points", {}).get(body)
            if points is not None:
                # The frozen provider already stores immutable collider-local
                # points. Avoid its USD traversal on the live hot path.
                bounds = _world_bounds_from_body_local_points(points,
                    position_w_m=position, orientation_wxyz=orientation)
                paths = self.provider._collider_paths[body]
            else:
                bounds, paths = self.provider.collision_bounds(body,
                    body_position_w_m=position, body_orientation_wxyz=orientation)
            if bounds is None or not all(math.isfinite(v) for v in (*bounds.minimum_m, *bounds.maximum_m)):
                quality.append(f"unverified current-pose collider bounds for {body}")
                continue
            bounds_by_body[body], paths_by_body[body] = bounds, paths
        wheels = {}
        for joint, body in WHEEL_JOINT_TO_BODY.items():
            center = _optional_vec3(body_positions_w_m.get(body))
            bounds = bounds_by_body.get(body)
            valid = center is not None and bounds is not None
            wheels[joint] = WheelGeometry(
                joint_name=joint, body_name=body, center_w_m=center,
                bottom_w_m=(center[0], center[1], bounds.minimum_m[2]) if valid else None,
                collider_bounds_w_m=bounds,
                geometry_source=("body_local_collider_live_pose:" + ",".join(paths_by_body[body])
                                 if valid else "unverified_current_pose"),
                verified=valid,
            )
        result = GeometrySnapshot(wheels=wheels, body_bounds_w_m=bounds_by_body,
            base_obstacle_penetration_m=aabb_intersection_depth(bounds_by_body.get(BASE_BODY), obstacle_aabb(self.obstacle)),
            quality=tuple(quality))
        self.last_snapshot = result
        return result


class SemanticSensorReader(SensorReader):
    """Original atomic read, versioned geometry and extra measured DTO fields."""
    def __init__(self, adapter, *, geometry_backend, physical_acceptance_version="all_stage_v1", **kwargs):
        if physical_acceptance_version != "all_stage_v1":
            raise SensingContractError("SemanticSensorReader requires all_stage_v1")
        geometry = (geometry_backend if isinstance(geometry_backend, SemanticColliderGeometry)
                    else SemanticColliderGeometry(geometry_backend))
        super().__init__(adapter, geometry_backend=geometry, **kwargs)

    @classmethod
    def from_live_scene(cls, scene_handle, adapter, *, backends=None, physical_acceptance_version="all_stage_v1"):
        if backends is None:
            backends = create_live_sensing_backends(sim=scene_handle.sim, robot=scene_handle.robot)
        return cls(adapter, contact_backend=backends.contact_backend,
            geometry_backend=backends.geometry_backend,
            physics_dt_s=float(scene_handle.sim.get_physics_dt()),
            physical_acceptance_version=physical_acceptance_version)

    def read(self, *args, **kwargs):
        raw = super().read(*args, **kwargs)
        geometry = self.geometry_backend.last_snapshot
        return SemanticPhysicalObservation(**{name: getattr(raw, name) for name in _OBSERVATION_FIELDS},
            body_bounds_w_m=geometry.body_bounds_w_m, geometry_pose_aware=True)


def physical_contact_surface(pair, *, kind: str, obstacle, tolerance_m: float,
                             force_noise_floor_n: float) -> dict:
    """All-stage opt-in interpretation; never rewrite the raw exact pair.

    ContactSensor supplies a resultant force and an optional mean point, NOT
    individual contact normals. Corner/mixed/missing point evidence stays
    ambiguous. Reaction magnitude and gravity-bearing force are distinct.
    """
    def member(value, key, default=None):
        return value.get(key, default) if isinstance(value, Mapping) else getattr(value, key, default)

    result = dict(surface="UNVERIFIED", verified=False, pair_valid=False,
                  reaction=False, reaction_force_n=None, bearing_force_n=0.,
                  bearing_verified=False, source="exact_pair_force_and_mean_point_v1")
    if member(pair, "pair_verified") is not True or type(member(pair, "active")) is not bool:
        return result
    try:
        force = _vec3(member(pair, "force_w_n"))
    except (TypeError, ValueError, ContactContractError):
        return result
    result.update(pair_valid=True, reaction=member(pair, "active"), reaction_force_n=_norm(force))
    if not result["reaction"]:
        result.update(surface="NONE", verified=True, bearing_verified=True)
        return result
    upward = max(0., force[2])
    horizontal = math.hypot(force[0], force[1])
    if kind == GROUND_PAIR:
        # The exact ground filter determines its plane. Tangential reaction
        # alone is not vertical support, regardless of its norm.
        result.update(surface="GROUND", verified=True, bearing_force_n=upward,
                      bearing_verified=True)
        return result
    if kind != OBSTACLE_PAIR:
        raise ContactContractError("unknown physical contact surface pair")
    try:
        point = _vec3(member(pair, "contact_point_w_m"))
    except (TypeError, ValueError, ContactContractError):
        result["surface"] = "OBSTACLE_AMBIGUOUS"
        return result
    front, back, left, right, top = (float(member(obstacle, name)) for name in
        ("front_x_m", "back_x_m", "left_y_m", "right_y_m", "top_z_m"))
    lateral = right-tolerance_m <= point[1] <= left+tolerance_m
    on_top = lateral and front-tolerance_m <= point[0] <= back+tolerance_m and abs(point[2]-top) <= tolerance_m
    if on_top and upward >= force_noise_floor_n and upward > horizontal:
        result.update(surface="TOP", verified=True, bearing_force_n=upward,
                      bearing_verified=True)
    elif (lateral and abs(point[0]-front) <= tolerance_m and point[2] < top-tolerance_m
          and abs(force[0]) > abs(force[2])):
        result.update(surface="FRONT_WALL", verified=True)
    else:
        result["surface"] = "OBSTACLE_AMBIGUOUS"
    return result
