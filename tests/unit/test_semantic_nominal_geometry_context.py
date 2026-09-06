"""CPU articulation ABI and same-state tests for the real geometry helper.

The synthetic articulation has a noncanonical DOF order, nonzero COM offset,
angular Jacobian rows, joint velocities, and floating-base velocities. No Isaac
or helper mock is used. Algebraic identities are not physical-motion guarantees.
"""
from __future__ import annotations

import math
from types import SimpleNamespace as NS

import pytest
import torch

from wlr50_clean.infrastructure.command_batch import (
    FULL12_ORDER, SERVO_COMMAND_SIGN, SERVO_ORDER, WHEEL_ORDER,
    resolve_joint_indices, servo_limits_deg,
)
from wlr50_clean.infrastructure.robot_adapter import bounded_drive_feedback_step
from wlr50_clean.ppo.semantic_nominal_geometry import (
    ACTIVE, CONTEXT_SCHEMA, EVIDENCE_SCHEMA, MODE, NominalGeometryError,
    capture_nominal_geometry_context, correct_nominal_geometry,
)
from wlr50_clean.sensing.geometry import WHEEL_JOINT_TO_BODY


def cross(a, b):
    """Independent scalar twist translation, not the helper's tensor expression."""
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def no_write(*args, **kwargs):
    raise AssertionError("capture/correction must not write, reset, step or advance")


def fixture(*, phase="P09", fixed=False, dtype=torch.float64, clearance=0.035,
            canonical_zero=False):
    names = tuple(reversed(FULL12_ORDER))
    joint_map = resolve_joint_indices(names)
    body_names = ("base_link", "aux_link", WHEEL_JOINT_TO_BODY[WHEEL_ORDER[3]],
                  WHEEL_JOINT_TO_BODY[WHEEL_ORDER[1]], WHEEL_JOINT_TO_BODY[WHEEL_ORDER[2]],
                  WHEEL_JOINT_TO_BODY[WHEEL_ORDER[0]])
    leg, indices, wheel_index = ACTIVE[phase]
    body_index = body_names.index(WHEEL_JOINT_TO_BODY[WHEEL_ORDER[wheel_index]])
    body_row = body_index - int(fixed)
    offset_columns = 0 if fixed else 6
    columns = tuple(joint_map.servo_ids[i] + offset_columns for i in indices)
    standing = {name: 20.0 + 4.0*i for i, name in enumerate(SERVO_ORDER)}
    canonical = (0.,)*8 if canonical_zero else (4., -7., 3., 2., -5., 9., -8., 11.)
    q = torch.zeros((1, 12), dtype=dtype)
    for i, name in enumerate(SERVO_ORDER):
        q[0, joint_map.servo_ids[i]] = math.radians(standing[name] + SERVO_COMMAND_SIGN[name]*canonical[i])
    for i, dof in enumerate(joint_map.wheel_ids):
        q[0, dof] = 0.1*(i+1)
    qd = torch.arange(1, 13, dtype=dtype).reshape(1, 12) * 0.01
    count = len(body_names)
    link = torch.zeros((1, count, 3), dtype=dtype)
    com = torch.zeros_like(link)
    link[0, body_index] = torch.tensor((0.35, 0.08, 0.12), dtype=dtype)
    offset = torch.tensor((0.04, -0.03, 0.02), dtype=dtype)
    com[0, body_index] = link[0, body_index] - offset
    jac = torch.zeros((1, count-int(fixed), 6, 12+offset_columns), dtype=dtype)
    selected = jac[0, body_row]
    for col in range(12+offset_columns):
        k = col+1
        selected[:, col] = torch.tensor((.01*k, -.004*k, .002*k, .003*k, -.001*k, .005*k), dtype=dtype)
    if not fixed:
        # Root COM is the world origin. Root translational columns are identity;
        # root angular columns translate this body's COM and rotate its twist.
        for axis in range(3):
            unit = tuple(float(i == axis) for i in range(3))
            selected[:, axis] = torch.tensor((*unit, 0., 0., 0.), dtype=dtype)
            selected[:, axis+3] = torch.tensor((*cross(unit, tuple(com[0, body_index].tolist())), *unit), dtype=dtype)
    selected[:, columns[0]] = torch.tensor((.3, -.15, -.2, .1, .4, -.1), dtype=dtype)
    selected[:, columns[1]] = torch.tensor((.25, .05, .15, -.3, .2, .35), dtype=dtype)
    root_vel = torch.tensor(((.12, -.07, .04, .05, -.06, .08),), dtype=dtype)
    generalized = qd[0] if fixed else torch.cat((root_vel[0], qd[0]))
    body_com_vel = torch.zeros((1, count, 6), dtype=dtype)
    body_link_vel = torch.zeros_like(body_com_vel)
    body_com_vel[0, body_index] = selected @ generalized
    measured_twist = tuple(body_com_vel[0, body_index].tolist())
    translated = cross(measured_twist[3:], tuple((link[0, body_index]-com[0, body_index]).tolist()))
    body_link_vel[0, body_index] = torch.tensor(
        (*(measured_twist[i]+translated[i] for i in range(3)), *measured_twist[3:]), dtype=dtype)
    data = NS(joint_pos=q, joint_vel=qd, body_link_pos_w=link, body_com_pos_w=com,
              body_com_vel_w=body_com_vel, body_link_vel_w=body_link_vel,
              root_com_vel_w=root_vel)
    calls = []

    def get_jacobians():
        calls.append("get_jacobians")
        return jac

    robot = NS(body_names=body_names, joint_names=names, is_fixed_base=fixed, data=data,
               root_physx_view=NS(get_jacobians=get_jacobians),
               write_data_to_sim=no_write, set_joint_position_target=no_write,
               set_joint_velocity_target=no_write, reset=no_write)
    adapter = NS(robot=robot, joint_map=joint_map, standing_pose_deg=standing,
                 _final_drive_servo_deg={name: 0. for name in SERVO_ORDER},
                 servo_target_mapper=NS(maximum_delta_deg=1.25, advance=no_write))
    center = tuple(link[0, body_index].tolist())
    bottom = (center[0], center[1], .05+clearance)
    wheel = NS(center_w_m=center, bottom_w_m=bottom, geometry_verified=True)
    observation = NS(physics_tick=240, simulation_time_s=2.0,
                     wheels={WHEEL_ORDER[wheel_index]: wheel}, obstacle=NS(top_z_m=.05),
                     joints={name: NS(position_deg=canonical[i]) for i, name in enumerate(SERVO_ORDER)})
    current = {"within_top_xy": False, "ground_contact": False,
               "within_lateral_span": True, "clearance_m": bottom[2]-.05}
    task = {"termination_reason": None,
            "physical_evaluator": {"valid": True, "termination_reason": None,
                                   "physics_tick": 240, "simulation_time_s": 2.,
                                   "current_legs": {leg: current}}}
    return NS(adapter=adapter, observation=observation,
              source=NS(state_id=phase, physics_tick=240, sim_time_s=2.), task=task,
              clearance_margin_m=.015, dispatch_tick=391, calls=calls, jac=jac,
              body_index=body_index, body_row=body_row, columns=columns, indices=indices,
              leg=leg, current=current, wheel=wheel, canonical=canonical)


def capture(f):
    return capture_nominal_geometry_context(
        adapter=f.adapter, observation=f.observation, source_frame=f.source,
        task_snapshot=f.task, clearance_margin_m=f.clearance_margin_m,
        physics_tick=f.dispatch_tick)


@pytest.mark.parametrize("phase", ["P09", "P12"])
@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_real_shapes_permuted_dofs_world_com_to_link_and_velocity_identity(phase, fixed, dtype):
    f = fixture(phase=phase, fixed=fixed, dtype=dtype)
    before = {key: value.clone() for key, value in vars(f.adapter.robot.data).items()}
    jac_before = f.jac.clone()
    result = capture(f)
    assert result["schema"] == CONTEXT_SCHEMA
    assert result["source_phase_id"] == phase and result["active_leg"] == f.leg
    assert result["canonical_servo_indices"] == f.indices
    assert result["jacobian_body_row"] == f.body_index-int(fixed)
    assert result["jacobian_joint_columns"] == f.columns
    assert result["physical_joint_ids"] == tuple(f.adapter.joint_map.servo_ids[i] for i in f.indices)
    assert result["fixed_base"] is fixed
    # The angular shift is nonzero and analytically different from COM rows.
    assert result["jacobian_x_m_per_rad"] == pytest.approx((.305, .2645), abs=1e-7)
    assert result["jacobian_z_m_per_rad"] == pytest.approx((-.219, .151), abs=1e-7)
    assert result["jacobian_x_m_per_rad"] != pytest.approx((.3, .25), abs=1e-5)
    assert result["link_minus_com_world_m"] == pytest.approx((.04, -.03, .02), abs=1e-7)
    expected_q = tuple(math.radians(f.adapter.standing_pose_deg[SERVO_ORDER[i]]
                                    + SERVO_COMMAND_SIGN[SERVO_ORDER[i]]*f.canonical[i]) for i in f.indices)
    assert result["physical_q_rad"] == pytest.approx(expected_q, abs=1e-7)
    assert result["COM_velocity_identity_max_error"] < 1e-7
    assert result["link_velocity_identity_max_error_m_s"] < 1e-7
    assert result["source_control_tick"] == 240
    assert result["dispatch_physics_tick"] == 391  # Real dispatch may have priming offsets.
    assert result["physical_motion_guaranteed"] is False
    assert result["clearance_m"] == f.wheel.bottom_w_m[2]-f.observation.obstacle.top_z_m
    assert f.calls == ["get_jacobians"]
    assert torch.equal(f.jac, jac_before)
    assert all(torch.equal(getattr(f.adapter.robot.data, key), value) for key, value in before.items())


@pytest.mark.parametrize("bypass", ["phase", "ground", "place", "lateral", "invalid", "terminal", "task_terminal"])
def test_ineligible_context_bypasses_before_articulation_read(bypass):
    f = fixture()
    if bypass == "phase":
        f.source.state_id = "P06"
    elif bypass == "ground":
        f.current["ground_contact"] = True
    elif bypass == "place":
        f.current["within_top_xy"] = True
    elif bypass == "lateral":
        f.current["within_lateral_span"] = False
    elif bypass == "invalid":
        f.task["physical_evaluator"]["valid"] = False
    elif bypass == "terminal":
        f.task["physical_evaluator"]["termination_reason"] = "FALL"
    else:
        f.task["termination_reason"] = "WHOLE_TASK_SUCCESS"
    f.adapter.robot = None
    assert capture(f) is None
    assert not f.calls


@pytest.mark.parametrize("flag", ["within_top_xy", "ground_contact", "within_lateral_span"])
def test_missing_or_numeric_physical_booleans_rejected(flag):
    for value in (None, 0, 1):
        f = fixture()
        f.current[flag] = value
        with pytest.raises(NominalGeometryError, match="physical boolean"):
            capture(f)


@pytest.mark.parametrize("change", ["source_tick", "obs_time", "source_time", "dispatch_before_source", "negative_source", "nan_time"])
def test_stale_or_invalid_clocks_rejected(change):
    f = fixture()
    if change == "source_tick":
        f.source.physics_tick -= 1
    elif change == "obs_time":
        f.observation.simulation_time_s += 1./120.
    elif change == "source_time":
        f.source.sim_time_s += 1./120.
    elif change == "dispatch_before_source":
        f.dispatch_tick = 239
    elif change == "negative_source":
        f.source.physics_tick = f.observation.physics_tick = -1
        f.source.sim_time_s = f.observation.simulation_time_s = -1./120.
    else:
        f.observation.simulation_time_s = float("nan")
    with pytest.raises(NominalGeometryError, match="clock"):
        capture(f)


@pytest.mark.parametrize("change", ["tick", "time", "missing_tick", "nan_time"])
def test_evaluator_place_ground_flags_must_belong_to_the_same_measured_tick(change):
    f = fixture()
    evaluation = f.task["physical_evaluator"]
    if change == "tick":
        evaluation["physics_tick"] -= 1
    elif change == "time":
        evaluation["simulation_time_s"] -= 1./120.
    elif change == "missing_tick":
        evaluation.pop("physics_tick")
    else:
        evaluation["simulation_time_s"] = float("nan")
    # Clearance itself is unchanged: checking it alone cannot establish the
    # provenance of current place/ground/lateral eligibility from the evaluator.
    with pytest.raises(NominalGeometryError, match="clock"):
        capture(f)


@pytest.mark.parametrize("field", ["joint_pos", "joint_vel", "body_link_pos_w", "body_com_pos_w",
                                  "body_com_vel_w", "body_link_vel_w", "root_com_vel_w", "jacobian"])
def test_nonfinite_actual_tensor_rejected(field):
    f = fixture()
    value = f.jac if field == "jacobian" else getattr(f.adapter.robot.data, field)
    value.reshape(-1)[0] = float("nan")
    with pytest.raises(NominalGeometryError, match="tensor"):
        capture(f)


@pytest.mark.parametrize("field", ["joint_pos", "joint_vel", "body_link_pos_w", "body_com_pos_w",
                                  "body_com_vel_w", "body_link_vel_w", "root_com_vel_w", "jacobian"])
def test_wrong_batch_or_jacobian_shapes_rejected(field):
    f = fixture()
    if field == "jacobian":
        f.adapter.robot.root_physx_view.get_jacobians = lambda: f.jac[:, :, :, :-1]
    else:
        value = getattr(f.adapter.robot.data, field)
        setattr(f.adapter.robot.data, field, value.repeat(2, *([1]*(value.ndim-1))))
    with pytest.raises(NominalGeometryError, match="tensor"):
        capture(f)


@pytest.mark.parametrize("field", ["center", "bottom", "obstacle_top", "clearance", "observed_joint", "standing"])
def test_nonfinite_sensor_or_coordinate_inputs_cannot_hide_in_close(field):
    f = fixture()
    if field == "center":
        f.wheel.center_w_m = (float("nan"), .08, .12)
    elif field == "bottom":
        f.wheel.bottom_w_m = (.35, .08, float("nan"))
    elif field == "obstacle_top":
        f.observation.obstacle.top_z_m = float("nan")
    elif field == "clearance":
        f.current["clearance_m"] = float("nan")
    elif field == "observed_joint":
        f.observation.joints[SERVO_ORDER[f.indices[0]]].position_deg = float("nan")
    else:
        f.adapter.standing_pose_deg[SERVO_ORDER[f.indices[0]]] = float("nan")
    with pytest.raises(NominalGeometryError):
        capture(f)


@pytest.mark.parametrize("change", ["center", "clearance", "geometry_unverified", "canonical_sign", "standing"])
def test_wheel_sensor_evaluator_and_physical_q_must_be_same_source(change):
    f = fixture()
    if change == "center":
        f.wheel.center_w_m = (f.wheel.center_w_m[0]+.01, *f.wheel.center_w_m[1:])
    elif change == "clearance":
        f.current["clearance_m"] += .01
    elif change == "geometry_unverified":
        f.wheel.geometry_verified = False
    elif change == "canonical_sign":
        name = SERVO_ORDER[f.indices[0]]
        f.observation.joints[name].position_deg *= -1.
    else:
        f.adapter.standing_pose_deg[SERVO_ORDER[f.indices[0]]] += 1.
    with pytest.raises(NominalGeometryError):
        capture(f)


@pytest.mark.parametrize("bad", ["duplicate", "permuted", "negative", "out_of_range", "bool", "body_duplicate", "fixed_nonbool"])
def test_wrong_body_or_dof_mappings_rejected(bad):
    f = fixture()
    servo = list(f.adapter.joint_map.servo_ids)
    if bad == "duplicate":
        servo[0] = servo[1]
    elif bad == "permuted":
        servo[0], servo[1] = servo[1], servo[0]
    elif bad == "negative":
        servo[0] = -1
    elif bad == "out_of_range":
        servo[0] = 12
    elif bad == "bool":
        servo[0] = True
    elif bad == "body_duplicate":
        names = list(f.adapter.robot.body_names)
        names[1] = names[f.body_index]
        f.adapter.robot.body_names = tuple(names)
    else:
        f.adapter.robot.is_fixed_base = 0
    f.adapter.joint_map = NS(servo_ids=tuple(servo), wheel_ids=f.adapter.joint_map.wheel_ids)
    with pytest.raises(NominalGeometryError):
        capture(f)


@pytest.mark.parametrize("fixed, field", [
    (fixed, field) for fixed in (False, True)
    for field in ("body_com_vel_w", "body_link_vel_w", "joint_vel", "root_com_vel_w")
    if not (fixed and field == "root_com_vel_w")
])
def test_nonzero_velocity_identity_detects_wrong_reference_or_live_state(fixed, field):
    f = fixture(fixed=fixed)
    value = getattr(f.adapter.robot.data, field)
    if field.startswith("body_"):
        value[0, f.body_index, 0] += .1
    else:
        value[0, 0] += .5
    with pytest.raises(NominalGeometryError, match="velocities"):
        capture(f)


def correction_fixture(phase="P09"):
    f = fixture(phase=phase, clearance=.015, canonical_zero=True)
    context = capture(f)
    # Pure target algebra uses an explicitly chosen local two-row model. Capture
    # ABI/velocity validity is tested independently above and in the next test.
    context["jacobian_x_m_per_rad"] = (0., -1.)
    context["jacobian_z_m_per_rad"] = (1., 0.)
    native = [3., -2., 4., -3., 0., 0., 0., 0., .1, -.2, .3, -.4]
    bias = [0.]*12
    native[f.indices[0]], native[f.indices[1]] = 20., .5
    bias[f.indices[0]], bias[f.indices[1]] = 2., -.1
    return f, context, tuple(native), tuple(bias)


@pytest.mark.parametrize("phase", ["P09", "P12"])
def test_real_capture_to_real_projection_and_rear_target_inverse(phase):
    f = fixture(phase=phase, clearance=.015, canonical_zero=True)
    context = capture(f)
    native = [0.]*12
    native[f.indices[0]] = -20.
    adjusted, evidence = correct_nominal_geometry(
        adapter=f.adapter, native_full12=native, controller_bias_full12=(0.,)*12, context=context)
    d = math.radians(1.25)
    # Exact-forward, zero-down solution of [.305,.2645]dq=.305*d and
    # [-.219,.151]dq=0, independent of the implementation's polygon algorithm.
    hip = .305*d/(.305+.2645*.219/.151)
    knee = .219/.151*hip
    expected = (-math.degrees(hip), -math.degrees(knee))
    assert tuple(adjusted[i] for i in f.indices) == pytest.approx(expected, abs=1e-10)
    assert evidence["schema"] == EVIDENCE_SCHEMA
    assert evidence["projection"]["mathematical_contract_verified"] is True
    assert evidence["current_policy_residual_used"] is False
    assert evidence["physical_motion_guaranteed"] is False
    assert f.calls == ["get_jacobians"]


@pytest.mark.parametrize("phase", ["P09", "P12"])
def test_active_inverse_subtracts_controller_bias_from_desired_not_bounded_delta(phase):
    f, context, native, bias = correction_fixture(phase)
    adjusted, evidence = correct_nominal_geometry(
        adapter=f.adapter, native_full12=native, controller_bias_full12=bias, context=context)
    hip, knee = f.indices
    assert adjusted[hip] == pytest.approx(-2., abs=1e-10)
    assert adjusted[knee] == pytest.approx(native[knee], abs=1e-10)
    assert all(adjusted[i] == native[i] for i in range(12) if i not in f.indices)
    assert adjusted[8:] == native[8:]
    for i, desired in zip(f.indices, evidence["desired_zero_policy_canonical_target_deg"]):
        lo, hi = servo_limits_deg(SERVO_ORDER[i])
        reconstructed = bounded_drive_feedback_step(
            previous_deg=0., native_deg=adjusted[i], bias_deg=bias[i],
            maximum_delta_deg=1.25, lower_deg=lo, upper_deg=hi)
        assert reconstructed == pytest.approx(desired, abs=1e-10)
    assert evidence["old_zero_policy_physical_target_rad"][0] == pytest.approx(
        math.radians(f.adapter.standing_pose_deg[SERVO_ORDER[hip]]-1.25))
    assert f.adapter._final_drive_servo_deg == {name: 0. for name in SERVO_ORDER}


def test_identity_keeps_raw_native_not_slew_bounded_reconstruction():
    f, context, native, bias = correction_fixture()
    context["jacobian_z_m_per_rad"] = (-1., 0.)  # This rear-sign nominal moves up.
    adjusted, evidence = correct_nominal_geometry(
        adapter=f.adapter, native_full12=native, controller_bias_full12=bias, context=context)
    assert adjusted == native
    assert evidence["desired_zero_policy_canonical_target_deg"] is None
    assert evidence["nominal_geometry_adjustment_full12"] == [0.]*12
    assert evidence["status"] == "identity_within_descent_allowance"


def test_infeasible_slew_box_is_explicit_degraded_bypass_without_freezing_or_residual_input():
    f, context, native, bias = correction_fixture()
    f.adapter._final_drive_servo_deg[SERVO_ORDER[f.indices[0]]] = 20.
    adjusted, evidence = correct_nominal_geometry(
        adapter=f.adapter, native_full12=native, controller_bias_full12=bias, context=context)
    assert adjusted == native
    assert evidence["status"] == "degraded_bypass_infeasible_box_downward"
    assert evidence["projection"]["mathematical_contract_verified"] is False
    assert evidence["degraded_bypass_is_clearance_guarantee"] is False
    assert evidence["current_policy_residual_used"] is False
    assert evidence["desired_zero_policy_canonical_target_deg"] is None


@pytest.mark.parametrize("change", ["schema", "mode", "indices", "place", "ground", "margin", "q"])
def test_target_correction_rejects_invalid_or_ineligible_context(change):
    f, context, native, bias = correction_fixture()
    if change in ("schema", "mode"):
        context[change] = "wrong"
    elif change == "indices":
        context["canonical_servo_indices"] = (4, 5)
    elif change in ("place", "ground"):
        context["place_xy" if change == "place" else "ground_contact"] = True
    elif change == "margin":
        context["clearance_margin_m"] = 0.
    else:
        context["physical_q_rad"] = (float("nan"), 0.)
    with pytest.raises(NominalGeometryError):
        correct_nominal_geometry(adapter=f.adapter, native_full12=native,
                                 controller_bias_full12=bias, context=context)


def test_target_helper_has_no_current_residual_argument():
    f, context, native, bias = correction_fixture()
    with pytest.raises(TypeError):
        correct_nominal_geometry(adapter=f.adapter, native_full12=native,
                                 controller_bias_full12=bias, context=context,
                                 current_policy_residual_full12=(0.,)*12)
