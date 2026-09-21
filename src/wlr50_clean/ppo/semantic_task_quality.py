"""Task-space quality measurements, not a new sensor or physical acceptance rule."""
from __future__ import annotations

import math
from typing import Any, Mapping

OBJECTIVE = "task_conditioned_hip_wheel_quality_v1"
SPACE_CONFIG = {"phases": ["P01", "P02", "P05", "P06", "P07", "P08", "P09"],
                "clearance_margin_m": .020, "front_fraction": .5,
                "geometry_fraction": .5, "sample_audit": True}


def _vector(value, size, name):
    if not isinstance(value, (tuple, list)) or len(value) != size:
        raise ValueError(f"missing {size}-component {name}")
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value):
        raise ValueError(f"nonfinite/non-numeric {name}")
    return tuple(float(x) for x in value)


def task_space_quality_sample(task: Mapping[str, Any], metrics: Mapping[str, Any], config) -> dict:
    """Missing data stays null. Callers decide terminal versus invalid-sample handling.

    Distance between current collider AABB and the frozen obstacle AABB is a
    conservative Euclidean separation lower bound, not mesh distance, body z,
    contact force or a collision classifier. No benefit above the finite margin.
    """
    phase = task["stage_id"]
    result = dict(schema="wlr50_clean.task_space_quality_sample.v1", phase=phase,
        sim_time_s=metrics["sim_time_s"], eligible=phase in config["phases"], valid=False,
        clearance_margin_m=config["clearance_margin_m"], separation_lower_bound_m=None,
        raw_geometry_cost=None, body_collider_minimum_w_m=None, body_collider_maximum_w_m=None,
        obstacle_planes_world_m=None, reason="outside_declared_task_space_window",
        measurement="current_collider_obstacle_AABB_Euclidean_separation_lower_bound_not_exact_mesh_clearance")
    if not result["eligible"]:
        return result
    try:
        ev = task["physical_evaluator"]
        if ev.get("valid") is not True:
            raise ValueError("current physical evaluator measurement is invalid")
        geometry = ev["body_traversal_geometry"]
        if geometry.get("valid") is not True:
            raise ValueError("current collider bounds are invalid")
        low = _vector(geometry["minimum_w_m"], 3, "collider minimum")
        high = _vector(geometry["maximum_w_m"], 3, "collider maximum")
        planes = _vector(metrics["obstacle_planes_world_m"], 6, "live obstacle planes")
        front, back, left, right, bottom, top = planes
        if any(a > b for a, b in zip(low, high)) or not (front < back and right < left and bottom < top):
            raise ValueError("unordered collider/obstacle bounds")
        if not math.isclose(float(ev["simulation_time_s"]), float(metrics["sim_time_s"]), abs_tol=1e-8, rel_tol=0.):
            raise ValueError("collider measurement and reward clock differ")
        separation = tuple(max(a-y, x-b, 0.) for a, b, x, y in
            zip(low, high, (front, right, bottom), (back, left, top)))
        distance = math.sqrt(sum(x*x for x in separation))
        if not math.isfinite(distance):
            raise ValueError("nonfinite collider/obstacle separation")
        deficit = min(1., max(0., (config["clearance_margin_m"]-distance)/config["clearance_margin_m"]))
        result.update(valid=True, separation_lower_bound_m=distance, raw_geometry_cost=deficit**2,
            body_collider_minimum_w_m=low, body_collider_maximum_w_m=high,
            obstacle_planes_world_m=planes, reason=None)
    except (KeyError, TypeError, ValueError, AttributeError, OverflowError) as exc:
        result["reason"] = f"{type(exc).__name__}: {exc}"
    return result
