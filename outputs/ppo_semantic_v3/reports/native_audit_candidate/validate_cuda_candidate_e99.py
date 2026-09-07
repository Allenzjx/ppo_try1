"""UNRUN e99 revision: finite CUDA equivalence + whole-function host timing.

Do not run alongside Isaac/PPO. This file does not launch Isaac or simulate a
robot. The fixture uses the existing TargetBufferRobot; steady-boundary cases
warm the real mapper on CPU before moving all tensors to CUDA for the final
real semantic dispatch. write_data_to_sim is a fake tensor copy, NOT PhysX.
Ordinary/geometry fixture dispatch is on CUDA. Geometry uses synthetic J.
Thus results cannot prove the live PhysX producer-stream dependency or speedup.

Exact PowerShell command, ONLY after the root clears the live barrier:
  $out = 'C:\\robotics_sim\\wlr_robot\\fsm_base_on_recording_ppo_phase_v1\\outputs\\ppo_semantic_v3\\reports\\native_audit_candidate\\cuda_' + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssfffffffZ') + '.json'
  & 'C:\\Users\\kskzz\\miniconda3\\envs\\env_isaaclab\\python.exe' -P 'C:\\robotics_sim\\wlr_robot\\fsm_base_on_recording_ppo_phase_v1\\outputs\\ppo_semantic_v3\\reports\\native_audit_candidate\\validate_cuda_candidate_e99.py' --run-cuda --ack-exclusive-device --device cuda:0 --warmup 5 --samples 40 --max-seconds 120 --output $out

Default invocation / --help does not import torch or initialize CUDA. Explicit
flags, an exclusive fresh output, source pins and a read-only process check
precede torch import. The process check is a snapshot, NOT a scheduler lock:
the human/root must keep the device exclusive for this short measurement.
No checkpoint, optimizer, migration, production source or old report is edited.
Exit 0: fixed tensor checks passed + timing produced; 2: mismatch/failure.
Neither exit means that production integration or GPU/PhysX parity is approved.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
REFERENCE_SHA = "8b2c26469c2dfc723d4ef8c89505e6272b5cc67af569cff1d15774bcf1b28b1d"
CANDIDATE_SHA = "3b0ad2fd40cc5e9951b326bc23784448c3601f86c1378d5d6487ffe052606601"
DEPENDENCY_PINS = {
    "src/wlr50_clean/ppo/semantic_headroom.py": "e1ea16538d516425c73acd8f49bada84609fc2708f3dfef6e9ca434cd6d13147",
    "src/wlr50_clean/ppo/semantic_residual_adapter.py": "9bf73c501ad00e2d3361ab76abb1f5dfeb67cde27ba42c2fb6b57c2b1b8306ac",
    "src/wlr50_clean/ppo/semantic_nominal_geometry.py": "218db2636e3723df7c3217111f55795e0390ad9bb64a7f9fdde926d177e99e8b",
    "src/wlr50_clean/ppo/semantic_nominal_projection.py": "3c75ff06be900cc3eee7cb74bbcf826add2ed8faff557b4c2e6fd2533f5f1c9d",
    "src/wlr50_clean/ppo/isaac_fsm_backend.py": "cbdc6a980ef5bcf4405655e48be0d467d7f36f72918f64960dbce3526d98ba87",
    "src/wlr50_clean/infrastructure/robot_adapter.py": "f00a8cc83c05aa1dea590d5e9765865c79f9f7f7cfc024f01b9d08a5f4f4a8a2",
    "src/wlr50_clean/infrastructure/servo_target_mapper.py": "ca18b55f60994a6d346a559fccd02f7fdef3dbb924e0cc65efd68035cf165ffa",
    "src/wlr50_clean/infrastructure/command_batch.py": "dbcee551cd1eba697b06b5f29439325cc9e36938ecdfaaa30ca718632937f0db",
    "tests/unit/test_actuator_target_effect.py": "88fd95ff1cd54039a8cec9bd825cf0223535ddebc4504100c77b716e9c63fa8e",
}
ZERO = (0.,) * 12
TARGETS = (("data", "joint_pos_target"), ("data", "joint_vel_target"),
           ("robot", "_joint_pos_target_sim"), ("robot", "_joint_vel_target_sim"))


def canonical(value):
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def no_other_runtime_processes():
    if os.name != "nt":
        raise RuntimeError("This bounded launcher is reviewed only for the installed Windows host")
    # No stdout secrets/process command lines are retained in the result.
    command = ("$ErrorActionPreference='Stop'; @(Get-CimInstance Win32_Process | "
        "Where-Object { $_.ProcessId -ne " + str(os.getpid()) + " -and ("
        "$_.Name -match '^(kit|isaac-sim)(\\.exe)?$' -or "
        "($_.Name -match '^python(w)?\\.exe$' -and $_.CommandLine -match "
        "'(env_isaaclab|isaacsim|isaaclab|omni\\.kit|wlr50_clean\\.ppo)')) } | "
        "Select-Object ProcessId,Name) | ConvertTo-Json -Compress")
    result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
        capture_output=True, text=True, timeout=20, creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode or result.stderr.strip():
        raise RuntimeError("Read-only exclusive-process check failed")
    found = json.loads(result.stdout) if result.stdout.strip() else []
    if found:
        raise RuntimeError(f"Another Isaac/locked-Python process is active: {found}")


def full(value, channel):
    result = list(ZERO)
    result[channel] = value
    return tuple(result)


def first_differences(left, right, path="$", limit=16):
    """JSON-aware leaf paths, including -0 vs +0; no allclose replacement."""
    found = []
    def walk(a, b, at):
        if len(found) >= limit or canonical(a) == canonical(b):
            return
        if isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys():
            for key in sorted(a):
                walk(a[key], b[key], at + "." + key)
        elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
            for index, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f"{at}[{index}]")
        else:
            found.append({"path": at, "reference": a, "candidate": b})
    walk(left, right, path)
    return found


def percentile(values, fraction):
    ordered = sorted(values)
    at = fraction * (len(ordered) - 1)
    low = math.floor(at)
    high = math.ceil(at)
    return ordered[low] + (ordered[high] - ordered[low]) * (at - low)


def execute(args, report):
    # No import below here is executed without explicit authorization flags.
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "tests/unit"))
    import torch
    # Production training wraps env.step/audit in inference_mode. Keep that
    # context outside every measured call, not a per-call timing surcharge.
    with torch.inference_mode():
        return execute_on_cuda(args, report, torch)


def execute_on_cuda(args, report, torch):
    from test_actuator_target_effect import _adapter
    from wlr50_clean.infrastructure.command_batch import SERVO_ORDER, servo_limits_deg
    from wlr50_clean.ppo import actuator_target_effect as reference
    from wlr50_clean.ppo.isaac_fsm_backend import (
        IsaacFSMBackend, build_residual_actuation_plan, _live_source_mapper_state)
    from wlr50_clean.ppo.semantic_residual_adapter import SemanticActuationDispatch
    from wlr50_clean.ppo.semantic_nominal_geometry import CONTEXT_SCHEMA, MODE
    from wlr50_clean.ppo.semantic_headroom import HEADROOM_MODE

    spec = importlib.util.spec_from_file_location("unwired_cuda_native_candidate",
                                                HERE / "actuator_target_effect_candidate_e99.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    device = torch.device(args.device)
    if device.type != "cuda" or device.index is None or not torch.cuda.is_available():
        raise RuntimeError("An explicit available indexed CUDA device is required")
    torch.cuda.set_device(device)
    gpu_started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)

    def check_budget():
        if time.monotonic()-gpu_started > args.max_seconds:
            raise RuntimeError("Finite GPU microbenchmark time budget exceeded; partial evidence retained")
        if torch.cuda.max_memory_reserved(device) > 256*1024*1024:
            raise RuntimeError("Synthetic tensor allocator exceeded 256 MiB budget")

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    report["runtime"] = {"python": sys.version, "torch": torch.__version__,
        "cuda_build": torch.version.cuda, "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device), "host_threads": 1,
        "inference_mode": torch.is_inference_mode_enabled(),
        "floating_point_modes": ["explicit_preserve_denormals", "explicit_flush_denormals"]}
    functions = {"reference": reference.build_actuator_target_effect_audit,
                 "candidate": module.build_actuator_target_effect_audit}

    def owner(inputs, target):
        robot = inputs["adapter"].robot
        return robot.data if target[0] == "data" else robot

    def make_plan(residual, controller, nominal):
        return build_residual_actuation_plan(tuple(a+b for a, b in zip(nominal, residual)),
            frozen_nominal_full12=nominal, drive_feedback_bias_full12=controller,
            normal_drive_bias_full12=ZERO)

    def dispatch(adapter, plan, tick, context=None, *, headroom_mode=None, tracking=()):
        return IsaacFSMBackend._atomic_apply(None,
            SemanticActuationDispatch(adapter, plan, nominal_geometry_context=context,
                                      policy_headroom_mode=headroom_mode),
            plan.frozen_nominal_full12, physics_tick=tick, tracking_servo_names=tracking,
            drive_feedback_bias_full12=plan.combined_post_mapper_bias_full12)

    def fixture(case):
        # Build with known host denormal mode. Later audit modes inspect these
        # fixed already-dispatched targets; they do not alter a prepared scene.
        if not torch.set_flush_denormal(False):
            raise RuntimeError("Cannot explicitly establish host denormal mode")
        adapter = _adapter()
        robot = adapter.robot
        channel = case.get("channel", 6)
        phase = "P12" if channel in (4, 5) else "P09"
        nominal = case.get("nominal", 0.)
        residual = case.get("residual", .5)
        controller = case.get("controller", 0.)
        headroom_mode = HEADROOM_MODE if case.get("headroom") else None
        tracking = (SERVO_ORDER[channel],) if case.get("tracking") else ()
        tick, context = 1, None
        # Explicit synthetic CPU warm-up only for steady boundary fixtures. It
        # executes the real frozen mapper, not a PhysX state reconstruction.
        # All four tensors move to CUDA before the measured final dispatch.
        if case.get("warm"):
            if case.get("measured_offset_deg") is not None:
                robot.data.joint_pos[0, adapter.joint_map.servo_ids[channel]] = (
                    .75 + math.radians(case["measured_offset_deg"]))
            warm_count = max(80 if tracking else 24, int(math.ceil(abs(nominal)/1.25)) + 12)
            for tick in range(1, warm_count+1):
                dispatch(adapter, make_plan(ZERO, full(controller,channel), full(nominal,channel)),
                         tick, tracking=tracking)
            tick = warm_count+1
        for obj in (robot, robot.data):
            for key, value in tuple(vars(obj).items()):
                if isinstance(value, torch.Tensor):
                    setattr(obj, key, value.to(device=device))
        adapter._standing_servo_tensor = adapter._standing_servo_tensor.to(device=device)
        if case.get("geometry"):
            for tick in range(1, 21):
                dispatch(adapter, make_plan(full(-20., channel), ZERO, full(20., channel)), tick)
            tick, nominal = 21, 20.
            pair = (4, 5) if channel == 4 else (6, 7)
            context = {"schema": CONTEXT_SCHEMA, "mode": MODE, "source_phase_id": phase,
                "source_control_tick": 20, "source_sim_time_s": 20./120., "dispatch_physics_tick": tick,
                "active_leg": "RL" if channel == 4 else "RR", "canonical_servo_indices": pair,
                "physical_q_rad": tuple(float(robot.data.joint_pos[0, adapter.joint_map.servo_ids[i]]) for i in pair),
                "jacobian_x_m_per_rad": (0., -1.), "jacobian_z_m_per_rad": (1., 0.),
                "clearance_m": .015, "clearance_margin_m": .015, "place_xy": False,
                "ground_contact": False, "physical_motion_guaranteed": False}
        previous = tuple(adapter._final_drive_servo_deg[name] for name in SERVO_ORDER)
        actuation = make_plan(full(residual, channel), full(controller, channel), full(nominal, channel))
        ack = dispatch(adapter, actuation, tick, context, headroom_mode=headroom_mode, tracking=tracking)
        inputs = dict(adapter=adapter, actuation=actuation, raw_ack=ack,
            previous_final_drive_servo_deg=previous, source_phase_id=phase,
            policy_request=reference.actuator_target_audit_request(phase, full(.01, channel), (1,)*12),
            policy_headroom_mode=headroom_mode)
        if "signed_actual_zero" in case:
            joint = adapter.joint_map.wheel_ids[channel-8]
            value = case["signed_actual_zero"]
            robot.data.joint_vel_target[0, joint] = value
            robot._joint_vel_target_sim[0, joint] = value
        fault = case.get("fault")
        if fault:
            target = TARGETS[case.get("target", 0)]
            obj, name = owner(inputs, target), target[1]
            tensor = getattr(obj, name)
            ids = adapter.joint_map.servo_ids if "pos" in name else adapter.joint_map.wheel_ids
            if fault == "nan": tensor[0, ids[0]] = float("nan")
            elif fault == "float64": setattr(obj, name, tensor.double())
            elif fault == "mixed_device": setattr(obj, name, tensor.cpu())
            elif fault == "staged_mismatch": robot.data.joint_pos_target[0, adapter.joint_map.servo_ids[0]] += .1
            elif fault == "expected_mismatch":
                robot.data.joint_pos_target[0, adapter.joint_map.servo_ids[0]] += .1
                robot._joint_pos_target_sim[0, adapter.joint_map.servo_ids[0]] += .1
            elif fault == "geometry_missing": del ack["nominal_geometry_adjustment_full12"]
            elif fault.startswith("headroom_"):
                proof = ack.get("policy_headroom_evidence", {})
                if fault == "headroom_missing_mode": del ack["policy_headroom_mode"]
                elif fault == "headroom_unknown_mode": inputs["policy_headroom_mode"] = "unknown"
                elif fault == "headroom_missing_evidence": del ack["policy_headroom_evidence"]
                elif fault == "headroom_unexpected": inputs["policy_headroom_mode"] = None
                elif fault == "headroom_effective": proof["effective_policy_residual_full12"][channel] += .1
                elif fault == "headroom_combined": proof["effective_combined_post_mapper_bias_full12"][channel] += .1
                elif fault == "headroom_native": proof["geometry_corrected_native_full12"][channel] += .1
                elif fault == "headroom_reserve": proof["servo_reserve_deg"] = 1.
                elif fault == "headroom_interval": proof["policy_residual_intervals_servo_deg"][channel][0] -= .1
                elif fault == "headroom_bool": proof["servo_reserve_deg"] = True
                elif fault == "headroom_nan": proof["servo_reserve_deg"] = float("nan")
                elif fault == "headroom_request": ack["independent_policy_residual_requested_full12"][channel] += .1
                elif fault == "headroom_controller": ack["bounded_controller_bias_requested_full12"][channel] += .1
                elif fault == "headroom_geometry_missing": del ack["nominal_geometry_adjustment_full12"]
                else: raise ValueError("unknown headroom fixture fault")
            else: raise ValueError("unknown fixture fault")
        for target in TARGETS:
            tensor = getattr(owner(inputs, target), target[1])
            if not fault and (tensor.device != device or tensor.dtype != torch.float32):
                raise RuntimeError("Positive fixture is not all-four CUDA float32 buffers")
        return inputs

    def finite_snapshot(value):
        # Snapshot-only encoding lets NaN/Inf proof corruption reach each real
        # fail-closed audit. It does not sanitize inputs given to either audit.
        if isinstance(value,float) and not math.isfinite(value):
            return {"nonfinite_float_snapshot": value.hex()}
        if isinstance(value,dict): return {k:finite_snapshot(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)): return [finite_snapshot(v) for v in value]
        return value

    def snapshot(inputs):
        adapter = inputs["adapter"]
        tensors = {}
        for target in TARGETS:
            value = getattr(owner(inputs, target), target[1])
            # Byte view avoids signed-zero/subnormal conversion and NaN==NaN.
            tensors[".".join(target)] = {"device": str(value.device), "dtype": str(value.dtype),
                "shape": list(value.shape), "bytes": value.detach().contiguous().view(torch.uint8).cpu().tolist()}
        return canonical({"targets": tensors, "events": list(adapter.robot.events),
            "write_count": adapter.write_count, "ack": finite_snapshot(inputs["raw_ack"]),
            "request": inputs["policy_request"], "previous": inputs["previous_final_drive_servo_deg"],
            "expected_headroom_mode": inputs["policy_headroom_mode"],
            "mapper": _live_source_mapper_state(adapter, source_control_physics_tick=1)})

    @contextlib.contextmanager
    def forbid_mutations(inputs):
        saved = []
        def forbidden(*unused, **ignored):
            raise RuntimeError("AUDIT_ATTEMPTED_MUTATION_OR_EXTRA_READBACK")
        adapter = inputs["adapter"]
        for obj, names in ((adapter.servo_target_mapper, ("advance",)),
                           (adapter, ("apply_full12", "get_actual_full12")),
                           (adapter.robot, ("set_joint_position_target", "set_joint_velocity_target", "write_data_to_sim", "update"))):
            for name in names:
                saved.append((obj, name, getattr(obj, name)))
                setattr(obj, name, forbidden)
        try:
            yield
        finally:
            for obj, name, value in saved:
                setattr(obj, name, value)

    tiny, ulp = math.ldexp(1., -126), math.ldexp(1., -149)
    cases = [{"name": f"ordinary_c{channel}_{sign:+g}", "channel": channel, "residual": sign*.15}
             for channel in range(12) for sign in (-1., 1.)]
    cases += [{"name": f"quantized_servo_{value}", "channel": 0, "residual": value} for value in (1.e-9, -1.e-12)]
    cases += [{"name": f"signed_zero_{sign}", "channel": 8, "residual": 0., "signed_actual_zero": sign}
              for sign in (0., -0.)]
    cases += [{"name": f"subnormal_wheel_{value}", "channel": 9, "residual": value}
              for value in (ulp, -ulp, 1.e-40, -1.e-40)]
    # Both actual and counterfactual operands are NORMAL float32; their exact
    # difference is the smallest subnormal. Test both difference directions.
    cases += [{"name": f"normal_neighbors_{sign}_{direction}", "channel": 9,
               "nominal": sign*(tiny + (ulp if direction < 0 else 0.)),
               "residual": sign*direction*ulp, "normal_neighbor_case": True}
              for sign in (-1., 1.) for direction in (-1, 1)]
    cases += [{"name": f"geometry_{channel}_{residual}", "channel": channel,
               "residual": residual, "controller": 2., "geometry": True}
              for channel in (4, 6) for residual in (0., .5, 2., 1.e-12)]
    cases += [{"name": f"bad_{target}_{fault}", "target": target, "fault": fault}
              for target in range(4) for fault in ("nan", "float64")]
    cases += [{"name": fault, "fault": fault, "geometry": fault == "geometry_missing"}
              for fault in ("mixed_device", "staged_mismatch", "expected_mismatch", "geometry_missing")]
    cases += [{"name":f"headroom_ordinary_c{channel}_{sign:+g}", "channel":channel,
               "residual":sign*.15, "headroom":True}
              for channel in range(12) for sign in (-1.,1.)]
    for channel in range(8):
        lo, hi = servo_limits_deg(SERVO_ORDER[channel])
        for side, native, outward in (("lower",lo+1.,-.5),("upper",hi-1.,.5)):
            for tag, residual in (("zero",0.),("outward",outward),("inward",-outward)):
                cases.append({"name":f"headroom_c{channel}_{side}_{tag}", "channel":channel,
                              "nominal":native, "residual":residual, "warm":True, "headroom":True,
                              "expected_effective":0. if tag in ("zero","outward") else residual})
    cases += [{"name":f"headroom_controller_outside_{residual}", "channel":7, "nominal":-56.,
               "controller":-3., "residual":residual, "warm":True, "headroom":True,
               "expected_effective":0. if residual <= 0. else residual}
              for residual in (-.5,0.,.5,1.e-12)]
    cases += [{"name":"headroom_recovered_tracking", "channel":7, "nominal":-37.8,
               "tracking":True, "measured_offset_deg":60., "residual":-24.98,
               "warm":True, "headroom":True, "expected_effective":-24.98}]
    cases += [{"name":f"headroom_geometry_{channel}_{residual}", "channel":channel,
               "controller":2., "residual":residual, "geometry":True, "headroom":True}
              for channel in (4,6) for residual in (0.,.5,134.,1.e-12)]
    cases += [{"name":f"headroom_signed_zero_{sign}", "channel":8, "residual":0.,
               "headroom":True, "signed_actual_zero":sign} for sign in (0.,-0.)]
    cases += [{"name":f"headroom_subnormal_{value}", "channel":9, "residual":value,
               "headroom":True} for value in (ulp,-ulp,tiny,-tiny)]
    cases += [{"name":"headroom_normal_neighbor", "channel":9,"nominal":tiny,
               "residual":ulp,"headroom":True,"normal_neighbor_case":True}]
    cases += [{"name":f"headroom_buffer_{target}_{fault}", "target":target,
               "fault":fault, "headroom":True} for target in range(4) for fault in ("nan","float64")]
    cases += [{"name":fault, "fault":fault, "channel":6, "headroom":True,
               "geometry":fault == "headroom_geometry_missing"}
              for fault in ("headroom_missing_mode","headroom_unknown_mode","headroom_missing_evidence",
                            "headroom_unexpected","headroom_effective","headroom_combined","headroom_native",
                            "headroom_reserve","headroom_interval","headroom_bool","headroom_nan",
                            "headroom_request","headroom_controller","headroom_geometry_missing")]
    report["case_count_per_mode_stream"] = len(cases)
    report["equivalence"] = []
    failures = 0
    for separate_stream in (False, True):
        producer = torch.cuda.Stream(device=device)
        consumer = torch.cuda.Stream(device=device) if separate_stream else producer
        for flush in (False, True):
            for case_index, case in enumerate(cases):
                check_budget()
                with torch.cuda.stream(producer):
                    inputs = fixture(case)
                    ready = torch.cuda.Event()
                    ready.record(producer)
                # Explicit producer dependency; never try an unsafe un-ordered
                # race and mistake nondeterministic equality for proof.
                with torch.cuda.stream(consumer):
                    consumer.wait_event(ready)
                    if not torch.set_flush_denormal(flush):
                        raise RuntimeError("Host denormal-mode control unsupported")
                    before = snapshot(inputs)
                    outcomes = {}
                    order = ("reference", "candidate") if case_index % 2 == 0 else ("candidate", "reference")
                    with forbid_mutations(inputs):
                        for name in order:
                            try:
                                outcomes[name] = {"ok": True, "value": functions[name](**inputs)}
                            except Exception as error:
                                outcomes[name] = {"ok": False, "error_type": type(error).__name__, "error": str(error)}
                    unchanged = before == snapshot(inputs)
                left, right = outcomes["reference"], outcomes["candidate"]
                json_equal = canonical(left) == canonical(right)
                attempted_mutation = any(not result["ok"] and
                    "AUDIT_ATTEMPTED_MUTATION_OR_EXTRA_READBACK" in result.get("error", "")
                    for result in outcomes.values())
                if case.get("fault"):
                    # Mixed-device rejection is deliberately stricter in the
                    # candidate. All other selected single-fault messages must
                    # match; mutating then raising is never accepted.
                    passed = (not left["ok"] and not right["ok"] and unchanged
                              and (json_equal or case["fault"] == "mixed_device"))
                else:
                    passed = left["ok"] and right["ok"] and json_equal and unchanged
                passed = passed and not attempted_mutation
                entry = {"case": case, "host_flush_denormals": flush,
                    "stream_route": "explicit_event_wait" if separate_stream else "same_producer_stream",
                    "both_outputs": outcomes, "full_json_equal": json_equal,
                    "input_state_unchanged": unchanged, "passed": passed}
                if case.get("normal_neighbor_case") and left["ok"]:
                    actual = left["value"]["actual_native_targets"]["wheel_velocity_rad_s"][1]
                    nominal = left["value"]["counterfactual_native_targets"]["wheel_velocity_rad_s"][1]
                    entry["normal_neighbor_coverage"] = {"actual": actual, "nominal": nominal,
                        "both_normal_distinct": abs(actual) >= tiny and abs(nominal) >= tiny and actual != nominal,
                        "exact_double_difference": actual-nominal,
                        "actual_float32_bits": struct.unpack("<I", struct.pack("<f", actual))[0],
                        "nominal_float32_bits": struct.unpack("<I", struct.pack("<f", nominal))[0]}
                    if not entry["normal_neighbor_coverage"]["both_normal_distinct"]:
                        entry["passed"] = False
                if "expected_effective" in case and left["ok"]:
                    observed = left["value"]["policy_headroom_evidence"]["effective_policy_residual_full12"][case["channel"]]
                    entry["headroom_semantic_coverage"] = {"expected":case["expected_effective"],"observed":observed}
                    if abs(observed-case["expected_effective"]) > 5.e-12:
                        entry["passed"] = False
                if not entry["passed"]:
                    failures += 1
                    entry["first_differences"] = first_differences(left, right)
                report["equivalence"].append(entry)
    torch.set_flush_denormal(False)
    report["equivalence_failures"] = failures
    report["timings"] = []
    if failures:
        report["timing_status"] = "SKIPPED: exact equivalence or rejection/state checks failed"
        return False

    # Only whole-call host time. No CUDA-event/kernel-only number is presented
    # as end-to-end. Result creation/.tolist() is inside the function; test JSON
    # comparison, filesystem writes and fixture/dispatch setup are outside.
    timing_cases = ({"name": "ordinary", "channel": 6, "residual": .5, "controller": .3},
                    {"name": "geometry", "channel": 6, "residual": .5, "controller": 2., "geometry": True},
                    {"name": "headroom", "channel": 7, "nominal":-56., "controller":-3.,
                     "residual":-.5,"warm":True,"headroom":True},
                    {"name": "geometry_headroom", "channel": 6, "residual": .5, "controller": 2.,
                     "geometry": True,"headroom":True})
    for case in timing_cases:
        inputs = fixture(case)
        torch.cuda.synchronize(device)
        before = snapshot(inputs)
        values = {name: [] for name in functions}
        baseline_json = canonical(functions["reference"](**inputs))
        with forbid_mutations(inputs):
            for repetition in range(args.warmup + args.samples):
                check_budget()
                order = ("reference", "candidate") if repetition % 2 == 0 else ("candidate", "reference")
                for name in order:
                    torch.cuda.synchronize(device)  # excluded: drain previous work symmetrically
                    started = time.perf_counter_ns()
                    actual = functions[name](**inputs)
                    torch.cuda.synchronize(device)  # included: finish all this call's device work
                    elapsed_us = (time.perf_counter_ns() - started) / 1000.
                    if canonical(actual) != baseline_json:
                        raise RuntimeError("Timed call ceased to match the full reference JSON")
                    if repetition >= args.warmup:
                        values[name].append(elapsed_us)
        if before != snapshot(inputs):
            raise RuntimeError("Timed audit mutated input targets/history")
        report["timings"].append({"case": case["name"], "samples_per_function": args.samples,
            "warmup_per_function": args.warmup, "alternation": "AB/BA; equal counts",
            "host_time_us": {name: {"p50": percentile(data, .50), "p95": percentile(data, .95),
                "p99": percentile(data, .99), "min": min(data), "max": max(data), "samples": data}
                for name, data in values.items()}, "host_flush_denormals": False})
    report["timing_status"] = "MEASURED_SYNTHETIC_CUDA_TENSOR_FIXTURE_ONLY"
    report["gpu_bounded_elapsed_s"] = time.monotonic()-gpu_started
    report["torch_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-cuda", action="store_true")
    parser.add_argument("--ack-exclusive-device", action="store_true")
    parser.add_argument("--device", default="cuda:0", choices=("cuda:0",))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--max-seconds", type=int, default=120)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.run_cuda or not args.ack_exclusive_device:
        parser.error("No CUDA was initialized: both explicit flags are required after the live barrier clears")
    if not 1 <= args.warmup <= 20 or not 20 <= args.samples <= 100 or not 30 <= args.max_seconds <= 180 or args.output is None:
        parser.error("Provide fresh --output, warmup1..20, samples20..100, max-seconds30..180")
    destination = args.output.resolve()
    if destination.parent != HERE or destination.suffix != ".json":
        parser.error("Output must be a fresh .json directly in this isolated candidate directory")
    reference = ROOT / "src/wlr50_clean/ppo/actuator_target_effect.py"
    candidate = HERE / "actuator_target_effect_candidate_e99.py"
    if digest(reference) != REFERENCE_SHA or digest(candidate) != CANDIDATE_SHA:
        raise RuntimeError("Reviewed reference/candidate source changed; do not silently compare a different revision")
    dependency_hashes = {name: digest(ROOT/name) for name in DEPENDENCY_PINS}
    if dependency_hashes != DEPENDENCY_PINS:
        raise RuntimeError("Reviewed e99 fixture/helper/mapper/adapter dependencies changed; re-review, not repin blindly")
    no_other_runtime_processes()
    # Reserve before any GPU import. Never overwrite another measurement.
    with destination.open("x", encoding="utf-8") as output:
        report = {"schema": "unwired_native_audit_cuda_comparison.e99.v2", "status": "FAILED",
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "reference_sha256": REFERENCE_SHA, "candidate_sha256": CANDIDATE_SHA,
            "reviewed_revision": "e99fde1b3e8366f0ff1d484140b82877df745f0c",
            "dependency_pins_verified": dependency_hashes,
            "requested_budget": {"max_seconds_after_cuda_setup": args.max_seconds,
                "warmup_per_function_per_case": args.warmup, "samples_per_function_per_case": args.samples,
                "torch_reserved_byte_ceiling": 256*1024*1024},
            "script_sha256": digest(Path(__file__)),
            "fixture_source_sha256": dependency_hashes["tests/unit/test_actuator_target_effect.py"],
            "mapping_sources_sha256": {name: dependency_hashes[name] for name in (
                "src/wlr50_clean/infrastructure/robot_adapter.py",
                "src/wlr50_clean/infrastructure/servo_target_mapper.py",
                "src/wlr50_clean/infrastructure/command_batch.py",
                "src/wlr50_clean/ppo/semantic_residual_adapter.py",
                "src/wlr50_clean/ppo/semantic_headroom.py",
                "src/wlr50_clean/ppo/semantic_nominal_geometry.py",
                "src/wlr50_clean/ppo/semantic_nominal_projection.py")},
            "scope": "real frozen/semantic adapter and helper; synthetic GPU tensor setters/J; no PhysX",
            "unverified": ["actual PhysX/articulation producer stream visibility", "live 120Hz integration",
                "physical kinematics/contact", "whole-training speedup", "other devices/drivers/runtime versions"],
            "exclusive_process_check": "snapshot passed before torch import; caller maintains exclusivity"}
        passed = False
        try:
            passed = execute(args, report)
            report["status"] = "TENSOR_EQUIVALENCE_PASSED" if passed else "TENSOR_EQUIVALENCE_FAILED"
        except Exception as error:
            report["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        finally:
            report["ended_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            json.dump(report, output, allow_nan=False, sort_keys=True, indent=2)
            output.write("\n")
    print(json.dumps({"status": report["status"], "output": str(destination),
                      "equivalence_failures": report.get("equivalence_failures")}, allow_nan=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
