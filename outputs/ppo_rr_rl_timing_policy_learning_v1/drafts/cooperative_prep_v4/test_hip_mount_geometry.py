"""Stdlib-only synthetic tests: zero physical/training credit."""
import copy
import math
import types
import unittest

from hip_mount_geometry import JOINT_NAMES, LEGS, resolve_hip_mounts, measure_hip_mounts


class Attr:
    def __init__(self, value):
        self.value, self.authored = value, True
    def HasAuthoredValueOpinion(self): return self.authored
    def Get(self): return self.value
    def GetPropertyStack(self):
        return [types.SimpleNamespace(layer=types.SimpleNamespace(identifier="locked_robot.usd"))]


class Prim:
    def __init__(self, name, local=None, parent="/World/WLRRobot/base_link", rigid=False):
        self.name, self.local, self.parent, self.rigid = name, Attr(local), parent, rigid
    def GetName(self): return self.name
    def GetPath(self): return "/World/WLRRobot/joints/"+self.name
    def IsA(self, cls): return self.name in JOINT_NAMES.values()
    def HasAPI(self, cls): return self.rigid
    def IsValid(self): return True
    def GetBody0Rel(self): return types.SimpleNamespace(GetTargets=lambda: [self.parent])
    def GetLocalPos0Attr(self): return self.local


def fixture():
    points = ((1., 1., 0.), (1., -1., 0.), (-1., 1., 0.), (-1., -1., 0.))
    prims = [Prim(JOINT_NAMES[leg], point) for leg, point in zip(LEGS, points)]
    root, base = Prim("WLRRobot"), Prim("base_link", rigid=True)
    root.children = prims
    stage = types.SimpleNamespace(
        GetPrimAtPath=lambda p: base if p.endswith("/base_link") else root,
        GetRootLayer=lambda: types.SimpleNamespace(identifier="actual_test_stage.usd"),
        GetSessionLayer=lambda: types.SimpleNamespace(identifier="anon:test-session"))
    provider = types.SimpleNamespace(stage=stage, robot_prim_path="/World/WLRRobot")
    usd = types.SimpleNamespace(PrimRange=lambda r, flag: r.children,
                               TraverseInstanceProxies=lambda: object())
    physics = types.SimpleNamespace(Joint=lambda prim: prim, RigidBodyAPI=object())
    return provider, usd, physics, prims, base


def raw(q=(1., 0., 0., 0.), p=(0., 0., 2.)):
    return {"physics_tick": 9656, "simulation_time_s": 9656/120.,
            "bodies": {"base_link": {"name": "base_link", "position_w_m": p,
                                      "orientation_wxyz": q}}}


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.provider, self.usd, self.physics, self.prims, self.base = fixture()
    def resolve(self):
        return resolve_hip_mounts(self.provider, usd=self.usd, usd_physics=self.physics)
    def test_four_exact_named_mounts_actual_stage_provenance(self):
        r = self.resolve()
        self.assertTrue(r["source_verified"])
        self.assertEqual(set(r["mounts"]), set(LEGS))
        self.assertEqual(r["provenance"]["stage_root_layer_identifier"], "actual_test_stage.usd")
        self.assertEqual(r["mounts"]["RR"]["local_pos0_m"], [-1., -1., 0.])
    def test_identity_frame_clock_and_FR_height(self):
        m = measure_hip_mounts(self.resolve(), raw())
        self.assertTrue(m["valid"])
        self.assertEqual(m["physics_tick"], 9656)
        self.assertEqual(m["mount_world_m"]["FR"], [1., -1., 2.])
        self.assertEqual(m["FR_world_z_m"], 2.)
        self.assertEqual(m["left_minus_right_mean_z_m"], 0.)
    def test_translation_changes_heights_not_differences(self):
        q = (math.cos(.2), math.sin(.2), 0., 0.)
        a = measure_hip_mounts(self.resolve(), raw(q))
        b = measure_hip_mounts(self.resolve(), raw(q, (10., -8., 5.)))
        for key in ("left_minus_right_mean_z_m", "rear_minus_front_mean_z_m"):
            self.assertAlmostEqual(a[key], b[key])
        self.assertAlmostEqual(b["FR_world_z_m"]-a["FR_world_z_m"], 3.)
    def test_positive_roll_left_higher(self):
        a = .3; m = measure_hip_mounts(self.resolve(), raw((math.cos(a/2), math.sin(a/2), 0., 0.)))
        self.assertAlmostEqual(m["left_minus_right_mean_z_m"], 2*math.sin(a))
        self.assertAlmostEqual(m["rear_minus_front_mean_z_m"], 0.)
    def test_positive_pitch_rear_higher(self):
        a = .3; m = measure_hip_mounts(self.resolve(), raw((math.cos(a/2), 0., math.sin(a/2), 0.)))
        self.assertAlmostEqual(m["rear_minus_front_mean_z_m"], 2*math.sin(a))
    def test_yaw_after_roll_does_not_change_height_differences(self):
        a, b = .3, .7
        q = (math.cos(b/2)*math.cos(a/2), math.cos(b/2)*math.sin(a/2),
             math.sin(b/2)*math.sin(a/2), math.sin(b/2)*math.cos(a/2))
        m = measure_hip_mounts(self.resolve(), raw(q))
        self.assertAlmostEqual(m["left_minus_right_mean_z_m"], 2*math.sin(a))
        self.assertAlmostEqual(m["rear_minus_front_mean_z_m"], 0.)
    def test_quaternion_sign_scale_invariance(self):
        q = (.9, .1, .2, .3)
        a = measure_hip_mounts(self.resolve(), raw(q))
        b = measure_hip_mounts(self.resolve(), raw(tuple(-2*x for x in q)))
        self.assertEqual(a["mount_world_m"], b["mount_world_m"])
    def test_wrong_joint_parent_rejected_no_partial_mounts(self):
        self.prims[2].parent = "/World/WLRRobot/rear_left_upper"
        r = self.resolve(); self.assertFalse(r["source_verified"]); self.assertIsNone(r["mounts"])
    def test_parent_must_be_rigid_body(self):
        self.base.rigid = False
        self.assertFalse(self.resolve()["source_verified"])
    def test_duplicate_joint_rejected(self):
        self.prims.append(copy.deepcopy(self.prims[0]))
        self.assertFalse(self.resolve()["source_verified"])
    def test_missing_joint_rejected(self):
        self.prims.pop()
        self.assertFalse(self.resolve()["source_verified"])
    def test_nonauthored_point_rejected(self):
        self.prims[0].local.authored = False
        self.assertFalse(self.resolve()["source_verified"])
    def test_nonfinite_local_rejected(self):
        self.prims[1].local.value = [float("nan"), 0., 0.]
        self.assertFalse(self.resolve()["source_verified"])
    def test_invalid_resolution_has_no_fallback(self):
        self.prims.pop()
        m = measure_hip_mounts(self.resolve(), raw())
        self.assertFalse(m["valid"]); self.assertIsNone(m["mount_world_m"])
    def test_invalid_parent_pose_is_NA(self):
        for q, p in [((0., 0., 0., 0.), (0., 0., 0.)), ((1., 0., 0., 0.), (float("inf"), 0., 0.))]:
            m = measure_hip_mounts(self.resolve(), raw(q, p))
            self.assertFalse(m["valid"]); self.assertIsNone(m["FR_world_z_m"])
    def test_never_uses_moving_upper_link_origin(self):
        r = self.resolve(); a = raw(); b = copy.deepcopy(a)
        for name in ("front_left_upper", "front_right_upper", "rear_left_upper", "rear_right_upper"):
            b["bodies"][name] = {"position_w_m": [100., -100., 999.]}
        self.assertEqual(measure_hip_mounts(r, a), measure_hip_mounts(r, b))
    def test_no_mutation_or_physics_operations(self):
        r, o = self.resolve(), raw(); before = copy.deepcopy((r, o))
        m = measure_hip_mounts(r, o)
        self.assertEqual((r, o), before)
        self.assertEqual(m["physics_writes"], 0)
        self.assertFalse(m["reward_or_target_or_permission"])


if __name__ == "__main__":
    unittest.main()
