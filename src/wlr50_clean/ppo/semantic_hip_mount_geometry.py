"""Read-only USD hip-installation geometry diagnosis.

No targets, reward, physics operations, model imports, or URDF fallback.
Resolve once from the actual live USD stage; transform from each observed
base_link pose. PXR is imported only by the live resolver, not pure analysis.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

LEGS = ("FL", "FR", "RL", "RR")
JOINT_NAMES = dict(zip(LEGS, ("front_left_hip", "front_right_hip",
                            "rear_left_hip", "rear_right_hip")))
RESOLUTION_SCHEMA = "readonly.same_rigid_body_hip_mount_resolution.v1"
FRAME_SCHEMA = "readonly.same_rigid_body_hip_mount_world_geometry.v1"


def _field(value, name):
    return value[name] if isinstance(value, Mapping) else getattr(value, name)


def _vector(value, count):
    if isinstance(value, (str, bytes)):
        raise ValueError("vector cannot be text")
    values = tuple(value)
    if len(values) != count or any(isinstance(x, bool) for x in values):
        raise ValueError("wrong vector shape/type")
    result = tuple(float(x) for x in values)
    if not all(math.isfinite(x) for x in result):
        raise ValueError("nonfinite geometry")
    return result


def _layer_id(layer):
    value = str(layer.identifier)
    if not value:
        raise ValueError("USD layer identifier unavailable")
    return value


def resolve_hip_mounts(provider, *, usd=None, usd_physics=None):
    """Use the same USD traversal/body0/localPos0 rules as resolve_rr_mount.

    Optional USD interfaces are for isolated stdlib test doubles only. The
    default imports actual PXR and reads the provider's actual live stage.
    An invalid result deliberately contains no mount coordinates.
    """
    out = {"schema": RESOLUTION_SCHEMA, "source_verified": False,
           "reason": None, "mounts": None, "parent_body_name": "base_link",
           "provenance": None, "physics_writes": 0,
           "source": "actual_USD_joint_body0_localPos0_not_URDF_or_moving_link_origin"}
    try:
        if usd is None or usd_physics is None:
            if usd is not None or usd_physics is not None:
                raise ValueError("supply both test USD interfaces or neither")
            from pxr import Usd, UsdPhysics
            usd, usd_physics = Usd, UsdPhysics
        stage = provider.stage
        root_path = str(provider.robot_prim_path).rstrip("/")
        if not root_path.startswith("/") or root_path == "":
            raise ValueError("absolute robot prim path required")
        root = stage.GetPrimAtPath(root_path)
        if not root or not root.IsValid():
            raise ValueError("robot prim unavailable on actual stage")
        parent_path = root_path + "/base_link"
        parent = stage.GetPrimAtPath(parent_path)
        if not parent or not parent.IsValid() or not parent.HasAPI(usd_physics.RigidBodyAPI):
            raise ValueError("base_link must be an actual rigid body")
        matches = {name: [] for name in JOINT_NAMES.values()}
        for prim in usd.PrimRange(root, usd.TraverseInstanceProxies()):
            name = prim.GetName()
            if name in matches and prim.IsA(usd_physics.Joint):
                matches[name].append(prim)
        mounts = {}
        for leg in LEGS:
            candidates = matches[JOINT_NAMES[leg]]
            if len(candidates) != 1:
                raise ValueError(f"{leg}: expected one named USD hip Joint, found {len(candidates)}")
            prim = candidates[0]
            joint = usd_physics.Joint(prim)
            targets = joint.GetBody0Rel().GetTargets()
            local = joint.GetLocalPos0Attr()
            if len(targets) != 1 or str(targets[0]) != parent_path:
                raise ValueError(f"{leg}: hip body0 is not the same rigid base_link")
            if not local.HasAuthoredValueOpinion():
                raise ValueError(f"{leg}: localPos0 has no authored value")
            point = _vector(local.Get(), 3)
            stack = local.GetPropertyStack()
            layers = sorted({_layer_id(spec.layer) for spec in stack})
            if not layers:
                raise ValueError(f"{leg}: localPos0 authored-layer provenance unavailable")
            mounts[leg] = {"joint_name": JOINT_NAMES[leg],
                "joint_prim_path": str(prim.GetPath()), "parent_prim_path": parent_path,
                "local_pos0_m": list(point), "authored_property_layers": layers}
        out.update(source_verified=True, mounts=mounts,
            provenance={"robot_prim_path": root_path, "parent_prim_path": parent_path,
                "stage_root_layer_identifier": _layer_id(stage.GetRootLayer()),
                "stage_session_layer_identifier": _layer_id(stage.GetSessionLayer()),
                "joint_order": [JOINT_NAMES[leg] for leg in LEGS],
                "read_method": "Usd.PrimRange(TraverseInstanceProxies)/UsdPhysics.Joint/body0/localPos0",
                "asset_hash_binding": "caller_must_retain_the_existing_runtime_asset_hash_receipt"})
    except Exception as exc:
        out.update(source_verified=False, mounts=None,
                   reason=f"{type(exc).__name__}: {exc}")
    return out
def _world_point(local, position, quaternion):
    point = _vector(local, 3)
    q = _vector(quaternion, 4)
    norm = math.sqrt(sum(x*x for x in q))
    if norm <= 1.e-12:
        raise ValueError("zero parent quaternion")
    w, x, y, z = (v/norm for v in q)
    vx, vy, vz = point
    tx, ty, tz = 2*(y*vz-z*vy), 2*(z*vx-x*vz), 2*(x*vy-y*vx)
    return [position[0]+vx+w*tx+y*tz-z*ty,
            position[1]+vy+w*ty+z*tx-x*tz,
            position[2]+vz+w*tz+x*ty-y*tx]


def measure_hip_mounts(resolution, raw_observation):
    """Pure same-tick base-link transform; no moving upper-link access.

    Diagnostics only. source_verified is not a contact, support, safety,
    readiness or task-success certificate. No unobserved temporal state.
    """
    out = {"schema": FRAME_SCHEMA, "valid": False, "source_verified": False,
           "reason": None, "physics_tick": None, "simulation_time_s": None,
           "frame": "world_z_gravity_axis_metres", "mount_world_m": None,
           "mount_world_z_m": None, "left_mean_z_m": None, "right_mean_z_m": None,
           "rear_mean_z_m": None, "front_mean_z_m": None,
           "left_minus_right_mean_z_m": None, "rear_minus_front_mean_z_m": None,
           "FR_world_z_m": None, "parent_pose": None, "source_resolution": resolution,
           "physics_writes": 0, "reward_or_target_or_permission": False}
    try:
        tick = _field(raw_observation, "physics_tick")
        time_s = _field(raw_observation, "simulation_time_s")
        if type(tick) is not int or tick < 0 or isinstance(time_s, bool):
            raise ValueError("invalid observed frame clock")
        time_s = float(time_s)
        if not math.isfinite(time_s) or time_s < 0.:
            raise ValueError("invalid observed time")
        out.update(physics_tick=tick, simulation_time_s=time_s)
        if resolution.get("schema") != RESOLUTION_SCHEMA or resolution.get("source_verified") is not True:
            raise ValueError("all four actual USD mounts must be verified first")
        mounts = resolution["mounts"]
        if set(mounts) != set(LEGS) or resolution.get("parent_body_name") != "base_link":
            raise ValueError("wrong resolved mount/body contract")
        parent_path = resolution["provenance"]["parent_prim_path"]
        if any(mounts[leg]["parent_prim_path"] != parent_path or
               mounts[leg]["joint_name"] != JOINT_NAMES[leg] for leg in LEGS):
            raise ValueError("inconsistent mount parents or canonical names")
        body = _field(raw_observation, "bodies")["base_link"]
        if _field(body, "name") != "base_link":
            raise ValueError("observed parent is not base_link")
        position = _vector(_field(body, "position_w_m"), 3)
        q = _vector(_field(body, "orientation_wxyz"), 4)
        positions = {leg: _world_point(mounts[leg]["local_pos0_m"], position, q) for leg in LEGS}
        heights = {leg: positions[leg][2] for leg in LEGS}
        left = (heights["FL"]+heights["RL"])/2
        right = (heights["FR"]+heights["RR"])/2
        rear = (heights["RL"]+heights["RR"])/2
        front = (heights["FL"]+heights["FR"])/2
        out.update(valid=True, source_verified=True, mount_world_m=positions,
            mount_world_z_m=heights, left_mean_z_m=left, right_mean_z_m=right,
            rear_mean_z_m=rear, front_mean_z_m=front,
            left_minus_right_mean_z_m=left-right, rear_minus_front_mean_z_m=rear-front,
            FR_world_z_m=heights["FR"],
            parent_pose={"body": "base_link", "position_w_m": list(position),
                         "orientation_wxyz": list(q), "observation_tick": tick},
            sign_semantics={"left_minus_right": "positive_left_higher",
                            "rear_minus_front": "positive_rear_higher"})
    except Exception as exc:
        out["reason"] = f"{type(exc).__name__}: {exc}"
    return out
