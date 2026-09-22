"""Optional read-only probe evidence; instantiate after HeightDiagnostics.start.

Returns data only: no file writes, setters, stepping, CoM reconstruction or reward.
"""
import math
from wlr50_clean.ppo.semantic_height_diagnostics import member, read_value, transform_point

HIP_JOINTS = dict(FL="front_left_hip", FR="front_right_hip", RL="rear_left_hip", RR="rear_right_hip")


def resolve_hip_mount(provider, joint_name):
    from pxr import Usd, UsdPhysics
    root = provider.stage.GetPrimAtPath(provider.robot_prim_path)
    matches = [p for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
               if p.GetName() == joint_name and p.IsA(UsdPhysics.Joint)]
    if len(matches) != 1:
        raise ValueError(f"expected one {joint_name} USD joint, found {len(matches)}")
    joint = UsdPhysics.Joint(matches[0])
    targets, local = joint.GetBody0Rel().GetTargets(), joint.GetLocalPos0Attr()
    if len(targets) != 1 or not local.HasAuthoredValueOpinion():
        raise ValueError("hip body0 relationship/authored localPos0 unavailable")
    if not str(targets[0]).startswith(provider.robot_prim_path + "/"):
        raise ValueError("hip parent outside this robot")
    point = tuple(float(x) for x in local.Get())
    if len(point) != 3 or not all(math.isfinite(x) for x in point):
        raise ValueError("invalid authored hip localPos0")
    return dict(joint_prim_path=str(matches[0].GetPath()), parent_prim_path=str(targets[0]),
                parent_body_name=targets[0].name, local_pos0_m=point)


def body_clearance_evidence(bounds, obstacle):
    """AABB distance is a conservative lower bound, never exact mesh separation."""
    low, high = bounds["minimum_m"], bounds["maximum_m"]
    point = bounds["lowest_collider_point"]["world_point_m"]
    front, back, left, right, bottom, top = (float(member(obstacle, k)) for k in
        ("front_x_m", "back_x_m", "left_y_m", "right_y_m", "bottom_z_m", "top_z_m"))
    if (any(len(v) != 3 for v in (low, high, point)) or
            not all(math.isfinite(x) for x in (*low, *high, *point, front, back, left, right, bottom, top))
            or not (front < back and right < left and bottom < top)
            or any(a > b for a, b in zip(low, high)) or any(not a-1e-10 <= p <= b+1e-10 for a, p, b in zip(low, point, high)) or
            not math.isclose(point[2], low[2], abs_tol=1e-10, rel_tol=0.)):
        raise ValueError("invalid current collider/obstacle geometry")
    separation = [max(a-y, x-b, 0.) for a, b, x, y in
                  zip(low, high, (front, right, bottom), (back, left, top))]
    return dict(collider_minimum_world_z_m=low[2], lowest_mesh_vertex_w_m=point,
        collider_aabb_w_m=dict(minimum_m=low, maximum_m=high),
        collider_aabb_xy_overlaps_obstacle=low[0] <= back and high[0] >= front and low[1] <= left and high[1] >= right,
        lowest_vertex_in_top_xy=front <= point[0] <= back and right <= point[1] <= left,
        global_minimum_minus_top_vertical_lower_bound_m=low[2]-top,
        aabb_separation_lower_bound_m=math.sqrt(sum(x*x for x in separation)),
        exact_mesh_obstacle_clearance_m=None,
        interpretation="global minimum minus top is not footprint-clipped clearance; zero AABB distance is not collision")


class ProbeGeometry:
    def __init__(self, height_diagnostics):
        self.height = height_diagnostics
        self.mounts = {leg: read_value(lambda name=name: resolve_hip_mount(self.height.provider, name),
                       "USD joint body0/localPos0") for leg, name in HIP_JOINTS.items()}

    def _mount_world(self, definition):
        if definition["value"] is None:
            raise ValueError(definition["reason"])
        mount = definition["value"]
        return transform_point(mount["local_pos0_m"], *self.height._pose(mount["parent_body_name"]))

    def sample(self, frame):
        before = self.height._clock()
        raw = member(frame.info, "raw_observation")
        result = dict(schema="wlr50_clean.task_probe_geometry.v1", physics_tick=frame.physics_tick,
            simulation_time_s=frame.sim_time_s, raw_physics_tick=member(raw, "physics_tick"),
            read_only=True, mount_definitions=self.mounts,
            hip_mount_world_m={leg: read_value(lambda d=d: self._mount_world(d),
                "authored joint frame transformed by live parent body_link pose") for leg, d in self.mounts.items()},
            body_geometry=read_value(lambda: body_clearance_evidence(self.height._bounds("base_link"),
                member(raw, "obstacle")), "independent enabled collider mesh at live pose"))
        result.update(clock_before=before, clock_after=self.height._clock())
        result["clock_unchanged"] = result["clock_before"] == result["clock_after"]
        result["frame_clock_aligned"] = before.get("episode_tick") == frame.physics_tick == result["raw_physics_tick"]
        return result
