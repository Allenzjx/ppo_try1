"""Fixed CPU equivalence only; no CUDA/native launch or throughput claim."""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "tests/unit"))
from test_actuator_target_effect import _adapter, _dispatch, ZERO
from test_semantic_residual_adapter import plan
from test_semantic_nominal_geometry_dispatch import dispatch
from wlr50_clean.infrastructure.command_batch import SERVO_ORDER, servo_limits_deg
from wlr50_clean.ppo import actuator_target_effect as baseline
from wlr50_clean.ppo.isaac_fsm_backend import _live_source_mapper_state
from wlr50_clean.ppo.semantic_nominal_geometry import CONTEXT_SCHEMA, MODE

module_path = Path(__file__).with_name("actuator_target_effect_candidate.py")
spec = importlib.util.spec_from_file_location("unwired_native_audit_candidate", module_path)
candidate = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(candidate)


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    def forbidden(*args, **kwargs):
        pytest.fail("CPU correctness prototype must never initialize CUDA")
    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    yield
    torch.set_num_threads(previous)


def frozen_snapshot(inputs):
    adapter = inputs["adapter"]
    robot = adapter.robot
    buffers = {}
    for owner_name, owner in (("data", robot.data), ("robot", robot)):
        for name, value in vars(owner).items():
            if isinstance(value, torch.Tensor):
                assert value.device.type == "cpu"
                buffers[f"{owner_name}.{name}"] = (str(value.dtype), tuple(value.shape),
                    tuple(value.stride()), value.detach().numpy().tobytes())
    return {
        "buffers": buffers, "events": list(robot.events), "write_count": adapter.write_count,
        "mapper": copy.deepcopy(_live_source_mapper_state(adapter, source_control_physics_tick=1)),
        "last_ack": copy.deepcopy(adapter.last_ack), "raw_ack": copy.deepcopy(inputs["raw_ack"]),
        "previous": tuple(inputs["previous_final_drive_servo_deg"]),
        "request": copy.deepcopy(inputs["policy_request"]),
        "plan": repr(inputs["actuation"]),
    }


def json_exact(value):
    # JSON string equality also checks signed-zero serialization, which plain
    # dict/list numeric equality deliberately treats as equal.
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))


def compare(inputs, monkeypatch):
    before = frozen_snapshot(inputs)
    adapter = inputs["adapter"]
    def forbidden(*args, **kwargs):
        pytest.fail("audit must not advance/apply/set/write/update or reread live joints")
    monkeypatch.setattr(adapter.servo_target_mapper, "advance", forbidden)
    for name in ("apply_full12", "get_actual_full12"):
        monkeypatch.setattr(adapter, name, forbidden)
    for name in ("set_joint_position_target", "set_joint_velocity_target", "write_data_to_sim", "update"):
        monkeypatch.setattr(adapter.robot, name, forbidden)
    left = baseline.build_actuator_target_effect_audit(**inputs)
    right = candidate.build_actuator_target_effect_audit(**inputs)
    assert left == right
    assert json_exact(left) == json_exact(right)
    assert frozen_snapshot(inputs) == before
    return right


def full(value, channel):
    values = list(ZERO)
    values[channel] = value
    return tuple(values)


def semantic_inputs(*, channel=6, residual=.5, controller=0., nominal=0., geometry=False,
                    geometry_status="projected", hard_previous=None):
    adapter = _adapter()
    phase = "P12" if channel in (4, 5) else "P09"
    previous = tuple(adapter._final_drive_servo_deg.values())
    if hard_previous is not None:
        adapter._final_drive_servo_deg[SERVO_ORDER[channel]] = hard_previous
        previous = tuple(adapter._final_drive_servo_deg.values())
    tick, context = 1, None
    if geometry:
        # Genuine adapter dispatches warm its mapper to native20 while a
        # residual keeps actual final0. The J is an explicit synthetic local
        # CPU model, not a live contact/kinematics measurement.
        for tick in range(1, 21):
            dispatch(adapter, plan(full(-20., channel), nominal=full(20., channel)), tick)
        tick = 21
        previous = tuple(adapter._final_drive_servo_deg.values())
        pair = (4, 5) if channel == 4 else (6, 7)
        context = {
            "schema": CONTEXT_SCHEMA, "mode": MODE, "source_phase_id": phase,
            "source_control_tick": 20, "source_sim_time_s": 20./120., "dispatch_physics_tick": tick,
            "active_leg": "RL" if channel == 4 else "RR", "canonical_servo_indices": pair,
            "physical_q_rad": tuple(float(adapter.robot.data.joint_pos[0, adapter.joint_map.servo_ids[i]])
                                     for i in pair),
            "jacobian_x_m_per_rad": (0., -1.), "jacobian_z_m_per_rad": (1., 0.),
            "clearance_m": .015, "clearance_margin_m": .015, "place_xy": False,
            "ground_contact": False, "physical_motion_guaranteed": False,
        }
        nominal = 20.
    actuation = plan(full(residual, channel), full(controller, channel), full(nominal, channel))
    ack = dispatch(adapter, actuation, tick, context)
    if geometry and geometry_status != "projected":
        # Identity/degraded records use a separate real dispatch with no
        # correction, then attach internally consistent zero-adjustment proof.
        # This covers the audit's evidence branch, not geometry's status logic.
        adapter = _adapter()
        previous = tuple(adapter._final_drive_servo_deg.values())
        ack = dispatch(adapter, actuation, 1)
        ack.update(geometry_adjusted_native_full12=list(ack["native_drive_target_full12"]),
                   nominal_geometry_adjustment_full12=list(ZERO),
                   nominal_geometry_evidence={"status": geometry_status})
    return dict(adapter=adapter, actuation=actuation, raw_ack=ack,
        previous_final_drive_servo_deg=previous, source_phase_id=phase,
        policy_request=baseline.actuator_target_audit_request(phase, full(.01, channel), (1,)*12))


def test_reference_is_the_fixed_reviewed_source_and_only_one_cpu_boundary():
    raw = Path(baseline.__file__).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == "7ec65a2e425681d2e172caa409ae0735983d451bdf7346fa0cecba63cb268721"
    tree = ast.parse(module_path.read_text())
    cpu_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == "cpu"]
    assert len(cpu_calls) == 1
    assert candidate.ActuatorTargetEffectError is baseline.ActuatorTargetEffectError


@pytest.mark.parametrize("channel", range(12))
@pytest.mark.parametrize("residual", [.15, -.15])
def test_real_adapter_all12_canonical_reversed_ids(channel, residual, monkeypatch):
    result = compare(_dispatch(channel=channel, residual=residual,
        phase=f"P{channel+1:02d}"), monkeypatch)
    assert result["changed_target_channel_count"] == 1
    assert result["changed_channels_full12"] == [i == channel for i in range(12)]


@pytest.mark.parametrize("channel", range(8))
@pytest.mark.parametrize("residual", [1.e-9, -1.e-9, 1.e-12, -1.e-12])
def test_servo_float32_quantized_zero_is_not_effect(channel, residual, monkeypatch):
    result = compare(_dispatch(channel=channel, residual=residual, phase="P13"), monkeypatch)
    assert result["changed_target_channel_count"] == 0


@pytest.mark.parametrize("value", [0., -0., 1.e-45, -1.e-45, 1.e-40, -1.e-40])
def test_cpu_wheel_zero_and_subnormal_serialization_is_exact(value, monkeypatch):
    # CPU equivalence only. GPU FTZ/subtraction parity remains unmeasured and
    # none of these synthetic targets are claimed to move a physical robot.
    compare(_dispatch(channel=8, residual=value), monkeypatch)


@pytest.mark.parametrize("signed", [0., -0.])
def test_actual_signed_zero_preserved_but_not_counted(signed, monkeypatch):
    inputs = _dispatch(channel=8, residual=0.)
    robot, joint = inputs["adapter"].robot, inputs["adapter"].joint_map.wheel_ids[0]
    robot.data.joint_vel_target[0, joint] = signed
    robot._joint_vel_target_sim[0, joint] = signed
    result = compare(inputs, monkeypatch)
    assert result["changed_target_channel_count"] == 0
    assert math.copysign(1., result["actual_native_targets"]["wheel_velocity_rad_s"][0]) == math.copysign(1., signed)


@pytest.mark.parametrize("controller,previous", [(.3, 0.), (1.25, 0.), (-1.25, 0.), (0., 1.5)])
def test_controller_counterfactual_and_frozen_final_slew(controller, previous, monkeypatch):
    compare(_dispatch(controller_bias=controller, previous_final=previous), monkeypatch)


@pytest.mark.parametrize("channel,boundary", [(0, -135.), (0, 135.), (1, -60.), (1, 210.)])
def test_adversarial_large_independent_residual_respects_real_hard_limit(channel, boundary, monkeypatch):
    # Representative valid t-1 final state plus deliberately extreme policy
    # input tests the adapter's hard intersection, not configured PPO caps.
    inputs = semantic_inputs(channel=channel, residual=math.copysign(1000., boundary), hard_previous=boundary)
    assert inputs["raw_ack"]["drive_target_full12"][channel] == boundary
    compare(inputs, monkeypatch)


@pytest.mark.parametrize("channel", range(8, 12))
@pytest.mark.parametrize("sign", [-1., 1.])
def test_wheel_hard_clamp_after_independent_policy(channel, sign, monkeypatch):
    inputs = semantic_inputs(channel=channel, nominal=sign*2., residual=sign*3.)
    assert inputs["raw_ack"]["drive_target_full12"][channel] == pytest.approx(sign*2.0943951023931953)
    compare(inputs, monkeypatch)


def test_safety_projected_zero_with_nonzero_policy_request(monkeypatch):
    inputs = _dispatch(residual=0.)
    assert inputs["policy_request"]["raw_policy_action_full12"][0] != 0.
    assert compare(inputs, monkeypatch)["changed_target_channel_count"] == 0


@pytest.mark.parametrize("channel", [4, 6])
@pytest.mark.parametrize("controller", [0., 2.])
@pytest.mark.parametrize("residual", [0., .5, -.5, 2., 1.e-12])
def test_real_geometry_projection_three_branch_full_result(channel, controller, residual, monkeypatch):
    inputs = semantic_inputs(channel=channel, controller=controller, residual=residual, geometry=True)
    result = compare(inputs, monkeypatch)
    assert result["nominal_geometry_changed_target_channel_count"] == 1
    assert result["geometry_nominal_native_targets"] == result["counterfactual_native_targets"]


@pytest.mark.parametrize("status", ["unchanged", "degraded_bypass"])
def test_identity_geometry_evidence_is_retained(status, monkeypatch):
    result = compare(semantic_inputs(geometry=True, geometry_status=status), monkeypatch)
    assert result["nominal_geometry_evidence"]["status"] == status
    assert result["nominal_geometry_changed_target_channel_count"] == 0


TARGETS = (("data", "joint_pos_target"), ("data", "joint_vel_target"),
           ("robot", "_joint_pos_target_sim"), ("robot", "_joint_vel_target_sim"))


def target_owner(inputs, target):
    robot = inputs["adapter"].robot
    return (robot.data if target[0] == "data" else robot), target[1]


def compare_rejection(inputs, monkeypatch, *, exact_message=True):
    # A bad fixture may deliberately contain NaN; encode state using tensor
    # bytes and repr for error-path immutability rather than NaN==NaN.
    before = repr(frozen_snapshot(inputs))
    errors = []
    for function in (baseline.build_actuator_target_effect_audit, candidate.build_actuator_target_effect_audit):
        with pytest.raises((baseline.ActuatorTargetEffectError, IndexError, TypeError, RuntimeError, KeyError)) as error:
            function(**inputs)
        errors.append(error.value)
    if exact_message:
        assert type(errors[0]) is type(errors[1])
        assert str(errors[0]) == str(errors[1])
    assert repr(frozen_snapshot(inputs)) == before


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("bad", ["missing", "float64", "rank1", "batch2", "empty_batch", "narrow", "nan", "inf"])
def test_every_staged_dispatch_tensor_bad_shape_dtype_finite_rejected(target, bad, monkeypatch):
    inputs = _dispatch()
    owner, name = target_owner(inputs, target)
    tensor = getattr(owner, name)
    if bad == "missing": delattr(owner, name)
    elif bad == "float64": setattr(owner, name, tensor.double())
    elif bad == "rank1": setattr(owner, name, tensor[0])
    elif bad == "batch2": setattr(owner, name, tensor.repeat(2, 1))
    elif bad == "empty_batch": setattr(owner, name, tensor[:0])
    elif bad == "narrow": setattr(owner, name, tensor[:, :1])
    else:
        ids = inputs["adapter"].joint_map.servo_ids if "pos" in name else inputs["adapter"].joint_map.wheel_ids
        tensor[0, ids[0]] = float(bad)
    compare_rejection(inputs, monkeypatch)


@pytest.mark.parametrize("bad", ["duplicate", "length", "out_of_bounds", "swapped"])
def test_joint_id_binding_rejected(bad, monkeypatch):
    inputs = _dispatch()
    mapping = inputs["adapter"].joint_map
    ids = list(mapping.servo_ids)
    if bad == "duplicate": ids[0] = mapping.wheel_ids[0]
    elif bad == "length": ids.pop()
    elif bad == "out_of_bounds": ids[0] = 99
    else: ids[0], ids[1] = ids[1], ids[0]
    inputs["adapter"].joint_map = replace(mapping, servo_ids=tuple(ids))
    compare_rejection(inputs, monkeypatch)


@pytest.mark.parametrize("bad", ["staged", "dispatch", "expected", "bias", "slew", "native_nan", "previous_nan",
                                 "previous_length", "request_phase", "source_phase", "writes"])
def test_ack_request_expected_dispatch_mismatches_fail_closed(bad, monkeypatch):
    inputs = _dispatch()
    robot, joint = inputs["adapter"].robot, inputs["adapter"].joint_map.servo_ids[0]
    if bad in ("staged", "expected"): robot.data.joint_pos_target[0, joint] += .1
    if bad in ("dispatch", "expected"): robot._joint_pos_target_sim[0, joint] += .1
    if bad == "bias": inputs["raw_ack"]["drive_feedback_bias_requested_full12"][0] += .1
    elif bad == "slew": inputs["raw_ack"]["drive_feedback_final_slew_limit_deg_per_tick"] += .1
    elif bad == "native_nan": inputs["raw_ack"]["native_drive_target_full12"][0] = float("nan")
    elif bad == "previous_nan": inputs["previous_final_drive_servo_deg"] = (float("nan"),)*8
    elif bad == "previous_length": inputs["previous_final_drive_servo_deg"] = (0.,)*7
    elif bad == "request_phase": inputs["policy_request"]["policy_request_phase"] = "bad"
    elif bad == "source_phase": inputs["source_phase_id"] = "bad"
    elif bad == "writes": inputs["raw_ack"]["articulation_writes_this_call"] = 2
    compare_rejection(inputs, monkeypatch)


@pytest.mark.parametrize("bad", ["missing", "delta", "corrected", "proof", "front", "two_rears"])
def test_geometry_evidence_mismatch_rejected(bad, monkeypatch):
    inputs = semantic_inputs(geometry=True)
    ack = inputs["raw_ack"]
    if bad == "missing": del ack["nominal_geometry_adjustment_full12"]
    elif bad == "delta": ack["nominal_geometry_adjustment_full12"][6] += .1
    elif bad == "proof": ack["nominal_geometry_evidence"] = None
    else:
        indices = {"corrected": (6,), "front": (0,), "two_rears": (4, 6)}[bad]
        for index in indices:
            ack["geometry_adjusted_native_full12"][index] += .25
        ack["nominal_geometry_adjustment_full12"] = [a-b for a, b in zip(
            ack["geometry_adjusted_native_full12"], ack["native_drive_target_full12"])]
    compare_rejection(inputs, monkeypatch)


def test_two_simultaneous_bad_tensors_still_reject_with_documented_priority(monkeypatch):
    inputs = _dispatch()
    robot = inputs["adapter"].robot
    robot.data.joint_pos_target[0, inputs["adapter"].joint_map.servo_ids[0]] = float("nan")
    robot._joint_vel_target_sim = robot._joint_vel_target_sim.double()
    # Packing must structurally validate all tensors before one transfer; the
    # chosen message may thus precede an earlier finite-value error. Both fail.
    compare_rejection(inputs, monkeypatch, exact_message=False)


@pytest.mark.parametrize("geometry", [False, True])
def test_exactly_one_explicit_cpu_snapshot_call_no_other_device_copy(geometry, monkeypatch):
    inputs = semantic_inputs(geometry=True) if geometry else _dispatch()
    original_cpu = torch.Tensor.cpu
    calls = []
    def counted(tensor, *args, **kwargs):
        calls.append((tuple(tensor.shape), tensor.dtype, tensor.device.type))
        return original_cpu(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "cpu", counted)
    candidate.build_actuator_target_effect_audit(**inputs)
    assert calls == [((1, 24), torch.float32, "cpu")]


def test_successive_real_dispatches_never_reuse_old_snapshot(monkeypatch):
    inputs = _dispatch(residual=.15)
    first = candidate.build_actuator_target_effect_audit(**inputs)
    adapter = inputs["adapter"]
    previous = tuple(adapter._final_drive_servo_deg.values())
    actuation = plan(full(-.2, 0))
    ack = dispatch(adapter, actuation, 181)
    inputs.update(actuation=actuation, raw_ack=ack, previous_final_drive_servo_deg=previous)
    second = compare(inputs, monkeypatch)
    assert first["actual_native_targets"] != second["actual_native_targets"]
    assert first["physics_tick"] == 180 and second["physics_tick"] == 181
