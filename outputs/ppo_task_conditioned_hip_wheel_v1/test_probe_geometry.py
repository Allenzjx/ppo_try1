"""Pure CPU fake-USD tests; no Isaac app, stage, setter, rollout or optimizer."""
from copy import deepcopy
import math
from pathlib import Path, PurePosixPath
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from probe_geometry import HIP_JOINTS, ProbeGeometry, body_clearance_evidence, resolve_hip_mount, transform_point
from wlr50_clean.ppo.semantic_height_diagnostics import lowest_collider_point
from wlr50_clean.ppo.semantic_physical_sensing import _body_local_point_array


class Prim:
    def __init__(self, name, *, targets=None, point=(.1, .2, .3), authored=True):
        self.name, self.point, self.authored = name, point, authored
        self.targets = [PurePosixPath("/Robot/base_link")] if targets is None else targets
    def GetName(self): return self.name
    def GetPath(self): return "/Robot/joints/" + self.name
    def IsA(self, kind): return kind is Joint


class Joint:
    def __init__(self, prim): self.prim = prim
    def GetBody0Rel(self): return SimpleNamespace(GetTargets=lambda: self.prim.targets)
    def GetLocalPos0Attr(self):
        return SimpleNamespace(HasAuthoredValueOpinion=lambda: self.prim.authored, Get=lambda: self.prim.point)


def provider(prims):
    return SimpleNamespace(robot_prim_path="/Robot", stage=SimpleNamespace(GetPrimAtPath=lambda path: prims))


PXR = SimpleNamespace(Usd=SimpleNamespace(PrimRange=lambda root, flags: root,
    TraverseInstanceProxies=lambda: object()), UsdPhysics=SimpleNamespace(Joint=Joint))
OBSTACLE = dict(front_x_m=0., back_x_m=1., left_y_m=1., right_y_m=-1., bottom_z_m=0., top_z_m=.05)
BOUNDS = dict(minimum_m=[.1, -.2, .08], maximum_m=[.8, .2, .2],
    lowest_collider_point=dict(world_point_m=[.3, -.1, .08]))


class GeometryTests(unittest.TestCase):
    def test_lowest_point_is_transformed_mesh_vertex_not_link_xy(self):
        points = _body_local_point_array(((.1, .2, .3), (.4, .5, .2), (-.1, .3, .7)))
        pose, quat = (1., 2., 3.), (math.sqrt(.5), 0., math.sqrt(.5), 0.)
        result = lowest_collider_point(points, pose, quat)
        transformed = [transform_point(p, pose, quat) for p in points]
        expected = min(transformed, key=lambda p: p[2])
        self.assertEqual(result["world_point_m"], expected)
        self.assertNotEqual(result["world_point_m"][:2], list(pose[:2]))

    def test_transform_uses_rotated_parent_not_com_or_origin(self):
        result = transform_point((.1, 0., .2), (1., 2., 3.), (math.sqrt(.5), 0., math.sqrt(.5), 0.))
        for actual, expected in zip(result, (1.2, 2., 2.9)):
            self.assertAlmostEqual(actual, expected)

    def test_transform_invalid_quaternion_rejected(self):
        with self.assertRaises(ValueError): transform_point((0., 0., 0.), (0., 0., 0.), (0.,)*4)
        with self.assertRaises(ValueError): transform_point((math.nan, 0., 0.), (0., 0., 0.), (1., 0., 0., 0.))

    def test_all_four_authored_frames_resolve(self):
        p = provider([Prim(name, point=(i*.01, .02, .03)) for i, name in enumerate(HIP_JOINTS.values())])
        with patch.dict(sys.modules, pxr=PXR):
            mounts = {leg: resolve_hip_mount(p, name) for leg, name in HIP_JOINTS.items()}
        self.assertEqual(tuple(mounts), ("FL", "FR", "RL", "RR"))
        self.assertEqual(mounts["RR"]["local_pos0_m"], (.03, .02, .03))
        self.assertTrue(all(x["parent_body_name"] == "base_link" for x in mounts.values()))

    def test_bad_usd_definitions_rejected(self):
        cases = [[], [Prim("rear_right_hip"), Prim("rear_right_hip")],
                 [Prim("rear_right_hip", targets=[])], [Prim("rear_right_hip", authored=False)],
                 [Prim("rear_right_hip", targets=[PurePosixPath("/RobotOther/base_link")])],
                 [Prim("rear_right_hip", point=(0., math.nan, 0.))]]
        with patch.dict(sys.modules, pxr=PXR):
            for prims in cases:
                with self.subTest(prims=prims), self.assertRaises(ValueError):
                    resolve_hip_mount(provider(prims), "rear_right_hip")

    def test_above_platform_bounds_not_exact_distance(self):
        row = body_clearance_evidence(BOUNDS, OBSTACLE)
        self.assertEqual(row["collider_aabb_w_m"]["minimum_m"], BOUNDS["minimum_m"])
        self.assertAlmostEqual(row["aabb_separation_lower_bound_m"], .03)
        self.assertAlmostEqual(row["global_minimum_minus_top_vertical_lower_bound_m"], .03)
        self.assertTrue(row["lowest_vertex_in_top_xy"])
        self.assertIsNone(row["exact_mesh_obstacle_clearance_m"])

    def test_low_vertex_outside_platform_is_not_collision(self):
        b = deepcopy(BOUNDS)
        b.update(minimum_m=[-1., -.2, .01], maximum_m=[-.2, .2, .2],
                 lowest_collider_point=dict(world_point_m=[-.8, 0., .01]))
        row = body_clearance_evidence(b, OBSTACLE)
        self.assertAlmostEqual(row["global_minimum_minus_top_vertical_lower_bound_m"], -.04)
        self.assertAlmostEqual(row["aabb_separation_lower_bound_m"], .2)
        self.assertFalse(row["lowest_vertex_in_top_xy"])
        self.assertFalse(row["collider_aabb_xy_overlaps_obstacle"])

    def test_aabb_overlap_not_reported_as_collision_or_exact_zero(self):
        b = deepcopy(BOUNDS)
        b["minimum_m"][2] = b["lowest_collider_point"]["world_point_m"][2] = .04
        row = body_clearance_evidence(b, OBSTACLE)
        self.assertEqual(row["aabb_separation_lower_bound_m"], 0.)
        self.assertIsNone(row["exact_mesh_obstacle_clearance_m"])
        self.assertNotIn("collision", row)

    def test_missing_nan_and_inconsistent_geometry_rejected(self):
        for change in (lambda b: b.pop("lowest_collider_point"),
                       lambda b: b["minimum_m"].__setitem__(2, math.nan),
                       lambda b: b["lowest_collider_point"]["world_point_m"].__setitem__(0, 99.),
                       lambda b: b["lowest_collider_point"]["world_point_m"].__setitem__(2, .09)):
            b = deepcopy(BOUNDS)
            change(b)
            with self.assertRaises((ValueError, KeyError)): body_clearance_evidence(b, OBSTACLE)

    def test_missing_hip_is_null_without_discarding_other_hips(self):
        p = provider([Prim(n) for n in HIP_JOINTS.values() if n != "front_left_hip"])
        height = SimpleNamespace(provider=p, _pose=lambda name: ((1., 2., 3.), (1., 0., 0., 0.)),
            _bounds=lambda name: deepcopy(BOUNDS), _clock=lambda: dict(episode_tick=8))
        frame = SimpleNamespace(physics_tick=8, sim_time_s=8/120,
            info=dict(raw_observation=dict(physics_tick=8, obstacle=OBSTACLE)))
        with patch.dict(sys.modules, pxr=PXR): geometry = ProbeGeometry(height)
        row = geometry.sample(frame)
        self.assertIsNone(row["hip_mount_world_m"]["FL"]["value"])
        self.assertIn("found 0", row["hip_mount_world_m"]["FL"]["reason"])
        self.assertEqual(row["hip_mount_world_m"]["RR"]["value"], [1.1, 2.2, 3.3])
        self.assertTrue(row["clock_unchanged"] and row["frame_clock_aligned"])
        self.assertNotIn("center_of_mass", row)
        self.assertNotIn("CoM", row)

    def test_pose_missing_and_bad_clock_remain_explicit(self):
        def missing(name): raise KeyError("missing live parent pose")
        ticks = iter((dict(episode_tick=9), dict(episode_tick=10)))
        height = SimpleNamespace(provider=provider([Prim(n) for n in HIP_JOINTS.values()]),
            _pose=missing, _bounds=lambda name: None, _clock=lambda: next(ticks))
        frame = SimpleNamespace(physics_tick=8, sim_time_s=8/120,
            info=dict(raw_observation=dict(physics_tick=8, obstacle=OBSTACLE)))
        with patch.dict(sys.modules, pxr=PXR): geometry = ProbeGeometry(height)
        row = geometry.sample(frame)
        self.assertFalse(row["clock_unchanged"] or row["frame_clock_aligned"])
        self.assertTrue(all(v["value"] is None for v in row["hip_mount_world_m"].values()))
        self.assertIsNone(row["body_geometry"]["value"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
