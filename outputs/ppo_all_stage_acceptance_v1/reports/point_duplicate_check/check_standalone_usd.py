"""One bounded OFFLINE USD count; no simulator, tensor runtime, or geometry change."""
from __future__ import annotations

import ast
import gc
import importlib.abc
import json
import math
import os
from pathlib import Path
import struct
import sys
import time

print(f"OWNED_OFFLINE_PID={os.getpid()}", flush=True)
STARTED = time.monotonic()
ASSET = Path("C:/robotics_sim/wlr_robot/usd/wlr_robot_drive_test.usd")
PROJECT = Path("C:/robotics_sim/wlr_robot/fsm_base_on_recording_ppo_phase_v1")
RECEIPT = Path(__file__).with_name("standalone_usd_duplicate_receipt.json")
BODIES = ("base_link", "front_left_wheel", "front_right_wheel", "rear_left_wheel", "rear_right_wheel")
FORBIDDEN = {"torch", "isaacsim", "isaaclab", "omni", "carb", "kit"}


class NoSimulationImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in FORBIDDEN:
            raise ImportError(f"Offline check forbids runtime import: {fullname}")
        return None


sys.meta_path.insert(0, NoSimulationImports())
receipt = {
    "schema": "all_stage.offline_exact_point_duplicate_ratio.v1",
    "owned_pid": os.getpid(), "asset_path": str(ASSET),
    "status": "STARTED", "bodies": [],
    "method": "Exact existing provider functions extracted unchanged by AST; body-local float64 triple keys via struct.pack('<ddd'), first-occurrence membership, signed-zero patterns distinct.",
    "limits": ["Offline counts only; no quaternion/minmax equivalence or throughput proof.",
               "No physics, live process memory, active audit, Torch, Kit/AppLauncher, hash or asset modification.",
               "External controller enforces120s and768MiB working-set limit on this PID only."],
}


def finish(status, **details):
    receipt.update(status=status, elapsed_s=time.monotonic() - STARTED, **details)
    receipt["forbidden_modules_loaded"] = sorted(name for name in sys.modules if name.split(".")[0] in FORBIDDEN)
    with RECEIPT.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(receipt, ensure_ascii=False, allow_nan=False), flush=True)


def main():
    if RECEIPT.exists():
        raise RuntimeError("receipt already exists; this is a one-shot check")
    try:
        from pxr import Gf, Usd, UsdGeom, UsdPhysics
    except ImportError as exc:
        finish("BLOCKED_STANDALONE_USD_UNAVAILABLE", error=str(exc))
        return 2
    if any(name.split(".")[0] in FORBIDDEN for name in sys.modules):
        raise RuntimeError("standalone USD unexpectedly loaded a prohibited runtime")
    source = PROJECT / "src/wlr50_clean/sensing/geometry.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    names = {"_collision_enabled", "_body_local_collider_points"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in functions} != names:
        raise RuntimeError("exact provider functions unavailable")
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *functions], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"math": math}
    exec(compile(module, str(source), "exec"), namespace)
    stage = Usd.Stage.Open(str(ASSET))
    if stage is None:
        raise RuntimeError("standalone stage open failed")
    found = {name: [] for name in BODIES}
    for prim in stage.Traverse():
        if prim.GetName() in found:
            found[prim.GetName()].append(prim)
    for name in BODIES:
        if len(found[name]) != 1:
            raise RuntimeError(f"body {name}: expected unique asset prim, found{len(found[name])}")
        body = found[name][0]
        colliders = [prim for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies())
                     if prim.IsValid() and prim.HasAPI(UsdPhysics.CollisionAPI)
                     and namespace["_collision_enabled"](prim, UsdPhysics)]
        if not colliders:
            raise RuntimeError(f"no enabled exact collider for{name}")
        points = namespace["_body_local_collider_points"](body, colliders, Usd=Usd, UsdGeom=UsdGeom, Gf=Gf)
        if not points:
            raise RuntimeError(f"provider could not validate points for{name}")
        seen = set()
        for point in points:
            seen.add(struct.pack("<ddd", *point))
        original, unique = len(points), len(seen)
        receipt["bodies"].append({"body": name, "body_path": str(body.GetPath()),
            "enabled_collider_count": len(colliders), "original_points": original,
            "bitwise_distinct_points": unique, "duplicate_points": original - unique,
            "duplicate_fraction": (original - unique) / original})
        print(f"COUNTED_BODY={name} ORIGINAL={original} DISTINCT={unique}", flush=True)
        del points, seen
        gc.collect()
    original = sum(row["original_points"] for row in receipt["bodies"])
    unique = sum(row["bitwise_distinct_points"] for row in receipt["bodies"])
    finish("SUCCEEDED_OFFLINE_COUNT_ONLY", original_points=original,
           bitwise_distinct_points=unique, duplicate_fraction=(original - unique) / original)
    return 0


try:
    raise SystemExit(main())
except Exception as exc:
    if not RECEIPT.exists():
        finish("BLOCKED_OR_FAILED_OFFLINE_COUNT", error=f"{type(exc).__name__}: {exc}")
    raise
