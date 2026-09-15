"""CPU target/geometry contracts for v3; these are not contact-motion proofs."""
from __future__ import annotations

from copy import deepcopy
import math

import pytest
import torch

from test_semantic_nominal_geometry_context import cross, fixture
from wlr50_clean.infrastructure.command_batch import SERVO_COMMAND_SIGN, SERVO_ORDER, servo_limits_deg
from wlr50_clean.infrastructure.robot_adapter import bounded_drive_feedback_step
from wlr50_clean.ppo.semantic_nominal_geometry import (
    BOUNDED_RR_MODE, FUNCTIONAL_RR_MODE, NominalGeometryError,
    capture_nominal_geometry_context, correct_nominal_geometry,
)


def bounded_fixture(*, fixed=False, phase="P09"):
    f = fixture(fixed=fixed, phase=phase, canonical_zero=True)
    # A nontrivial live wheel rotation changes which immutable collider vertex
    # is lowest. Its bottom derivative differs from both COM and link origin.
    angle = math.pi / 6
    f.points = ((0., 0., -.08), (.05, .01, -.06), (-.03, -.02, .02))
    f.rotated = tuple((math.cos(angle)*x + math.sin(angle)*z, y,
                       -math.sin(angle)*x + math.cos(angle)*z)
                      for x, y, z in f.points)
    data = f.adapter.robot.data
    data.body_link_quat_w = torch.tensor([[(1., 0., 0., 0.)] * len(f.adapter.robot.body_names)],
                                         dtype=data.joint_pos.dtype)
    data.body_link_quat_w[0, f.body_index] = torch.tensor(
        (math.cos(angle/2), 0., math.sin(angle/2), 0.), dtype=data.joint_pos.dtype)
    f.lowest = min(range(len(f.rotated)), key=lambda i: f.rotated[i][2])
    f.wheel.bottom_w_m = tuple(a+b for a, b in zip(f.wheel.center_w_m, f.rotated[f.lowest]))
    f.current.update(clearance_m=f.wheel.bottom_w_m[2]-.05,
                     ground_relative_lift_m=.01, front_distance_m=-.03,
                     contact_surface="AIR", air=True, top_surface_contact=False,
                     bearing_verified=False, support=False, obstacle_pair_active=False)
    return f


def capture(f, **overrides):
    kwargs = dict(adapter=f.adapter, observation=f.observation, source_frame=f.source,
                  task_snapshot=f.task, clearance_margin_m=.015, physics_tick=f.dispatch_tick,
                  mode=BOUNDED_RR_MODE, minimum_lift_gain_m=.008, workspace_min_m=-.22,
                  collider_local_points=f.points, top_gap_min_m=-.015)
    kwargs.update(overrides)
    return capture_nominal_geometry_context(**kwargs)


def physical(f, pair):
    return tuple(math.radians(f.adapter.standing_pose_deg[SERVO_ORDER[i]]
                             + SERVO_COMMAND_SIGN[SERVO_ORDER[i]]*value)
                 for i, value in zip((6, 7), pair))


def target_fixture(*, source=(0., -27.8), previous=None, actual=None, jx=(0., 0.), jz=(0., 1.)):
    f = bounded_fixture()
    context = capture(f)
    previous = source if previous is None else previous
    actual = previous if actual is None else actual
    for i, value in zip((6, 7), previous):
        f.adapter._final_drive_servo_deg[SERVO_ORDER[i]] = value
    context.update(physical_q_rad=physical(f, actual), jacobian_x_m_per_rad=jx,
                   jacobian_z_m_per_rad=jz, clearance_margin_m=context["clearance_m"])
    native = (3., -2., 4., -3., 5., -4., *source, .1, -.2, .3, -.4)
    return f, context, native


def correct(f, context, native, bias=(0.,)*12):
    return correct_nominal_geometry(adapter=f.adapter, native_full12=native,
                                    controller_bias_full12=bias, context=context)


def reconstruct(f, adjusted, bias=(0.,)*12):
    return tuple(bounded_drive_feedback_step(
        previous_deg=f.adapter._final_drive_servo_deg[SERVO_ORDER[i]],
        native_deg=adjusted[i], bias_deg=bias[i],
        maximum_delta_deg=f.adapter.servo_target_mapper.maximum_delta_deg,
        lower_deg=servo_limits_deg(SERVO_ORDER[i])[0], upper_deg=servo_limits_deg(SERVO_ORDER[i])[1])
        for i in (6, 7))


def rotate_world_increment(point, omega, epsilon):
    """Independent Rodrigues perturbation of an already-world-axis vector."""
    length = math.sqrt(sum(v*v for v in omega))
    if length == 0.:
        return point
    unit = tuple(v/length for v in omega)
    angle = length*epsilon
    perpendicular = cross(unit, point)
    parallel = sum(a*b for a, b in zip(unit, point))
    return tuple(p*math.cos(angle) + c*math.sin(angle) + u*parallel*(1.-math.cos(angle))
                 for p, c, u in zip(point, perpendicular, unit))


@pytest.mark.parametrize("fixed", [False, True])
def test_current_lowest_collider_jacobian_matches_independent_pose_finite_difference(fixed):
    f = bounded_fixture(fixed=fixed)
    before = {key: value.clone() for key, value in vars(f.adapter.robot.data).items()}
    context = capture(f)
    assert f.lowest == 1
    assert context["lowest_collider_vertex_index"] == 1
    assert context["geometry_model"] == "current_pose_collider_lowest_vertex"
    assert context["jacobian_joint_columns"] == f.columns
    assert context["jacobian_body_row"] == f.body_index-int(fixed)
    assert context["lowest_collider_point_world_m"] == pytest.approx(f.wheel.bottom_w_m)
    assert context["jacobian_x_m_per_rad"] == pytest.approx((.305, .2645))
    assert context["link_origin_jacobian_z_m_per_rad"] == pytest.approx((-.219, .151))
    assert context["jacobian_z_m_per_rad"] != pytest.approx(context["link_origin_jacobian_z_m_per_rad"])
    offset = (.04, -.03, .02)
    derivatives = []
    for column in f.columns:
        twist = tuple(f.jac[0, f.body_row, :, column].tolist())
        translated = cross(twist[3:], offset)
        link_velocity = tuple(a+b for a, b in zip(twist[:3], translated))
        epsilon = 1e-6
        bottom = []
        for sign in (1., -1.):
            moved = [rotate_world_increment(point, twist[3:], sign*epsilon) for point in f.rotated]
            bottom.append(f.wheel.center_w_m[2] + sign*epsilon*link_velocity[2]
                          + min(point[2] for point in moved))
        derivatives.append((bottom[0]-bottom[1])/(2*epsilon))
    assert context["jacobian_z_m_per_rad"] == pytest.approx(derivatives, abs=1e-10)
    assert not context["physical_motion_guaranteed"]
    assert all(torch.equal(value, getattr(f.adapter.robot.data, key)) for key, value in before.items())
    assert f.calls == ["get_jacobians"]


def test_repeated_actual_offsets_do_not_integrate_fixed_source_toward_hard_limit():
    f, context, native = target_fixture()
    original = tuple(native)
    bands = None
    for _ in range(120):
        previous = tuple(f.adapter._final_drive_servo_deg[SERVO_ORDER[i]] for i in (6, 7))
        # A target-only counterexample to accumulated correction: continually
        # lagged actual pose, not a claimed simulation of contact dynamics.
        context["physical_q_rad"] = physical(f, (previous[0], previous[1]-.5))
        adjusted, evidence = correct(f, context, native)
        final = reconstruct(f, adjusted)
        assert final == pytest.approx(evidence["desired_zero_policy_canonical_target_deg"])
        assert all(abs(a-b) <= 1.25+1e-10 for a, b in zip(final, previous))
        assert -37.8-1e-10 <= final[1] <= -17.8+1e-10
        assert final[1] > -58.
        assert native == original
        if bands is None:
            bands = deepcopy(evidence["source_relative_operating_bands_deg"])
        assert evidence["source_relative_operating_bands_deg"] == bands
        assert all(adjusted[i] == native[i] for i in range(12) if i not in (6, 7))
        for i, value in zip((6, 7), final):
            f.adapter._final_drive_servo_deg[SERVO_ORDER[i]] = value
    assert final[1] == pytest.approx(-37.8)
    assert evidence["operating_envelope_reached"]
    assert evidence["needs_whole_body_height_or_support_reconfiguration"]
    assert evidence["nonintegrating_source_relative_bound"]
    assert not evidence["physical_motion_guaranteed"]


def test_existing_boundary_ack_recovers_inward_with_frozen_slew_not_instant_reset():
    f, context, native = target_fixture(previous=(0., -60.))
    targets = []
    for _ in range(24):
        previous = tuple(f.adapter._final_drive_servo_deg[SERVO_ORDER[i]] for i in (6, 7))
        context["physical_q_rad"] = physical(f, previous)
        adjusted, evidence = correct(f, context, native)
        final = reconstruct(f, adjusted)
        assert final[1] > -60.
        assert 0. <= final[1]-previous[1] <= 1.25+1e-10
        assert all(adjusted[i] == native[i] for i in range(12) if i not in (6, 7))
        targets.append(final[1])
        for i, value in zip((6, 7), final):
            f.adapter._final_drive_servo_deg[SERVO_ORDER[i]] = value
    assert targets[0] == pytest.approx(-58.75)
    assert targets[-1] == pytest.approx(-37.8)


def test_source_outside_operating_reserve_does_not_legitimize_hard_limit_target():
    f, context, native = target_fixture(source=(0., -60.), previous=(0., -58.), jz=(0., 0.))
    adjusted, evidence = correct(f, context, native)
    assert reconstruct(f, adjusted)[1] == pytest.approx(-58.)
    assert evidence["source_relative_operating_bands_deg"][1] == (-58., -50.)
    assert evidence["needs_whole_body_height_or_support_reconfiguration"]


@pytest.mark.parametrize("actual", [(0., -35.), (10., -27.8)])
def test_actual_tracking_outside_two_degree_trust_holds_without_extrapolated_projection(actual):
    f, context, native = target_fixture(actual=actual)
    adjusted, evidence = correct(f, context, native)
    assert reconstruct(f, adjusted) == pytest.approx((0., -27.8))
    assert evidence["projection"] is None
    assert not evidence["tracking_inside_linear_trust"]
    assert evidence["status"] == "bounded_hold_or_inward_recovery_tracking_exceeds_linear_trust"
    assert evidence["needs_whole_body_height_or_support_reconfiguration"]
    assert not evidence["fallback_is_clearance_guarantee"]


def test_bias_is_subtracted_once_and_ten_other_channels_and_adapter_history_are_unchanged():
    f, context, native = target_fixture(actual=(0., -28.3))
    bias = (0.,)*6 + (1., -2.) + (.01, -.02, .03, -.04)
    before = dict(f.adapter._final_drive_servo_deg)
    adjusted, evidence = correct(f, context, native, bias)
    assert reconstruct(f, adjusted, bias) == pytest.approx(evidence["desired_zero_policy_canonical_target_deg"])
    assert evidence["source_relative_operating_bands_deg"] == [(-9., 11.), (-39.8, -19.8)]
    assert all(adjusted[i] == native[i] for i in range(12) if i not in (6, 7))
    assert f.adapter._final_drive_servo_deg == before
    assert not evidence["current_policy_residual_used"]
    with pytest.raises(TypeError):
        correct_nominal_geometry(adapter=f.adapter, native_full12=native, controller_bias_full12=bias,
                                 context=context, current_policy_residual_full12=(1.,)*12)


def set_top(f):
    f.current.update(contact_surface="TOP", air=False, top_surface_contact=True,
                     bearing_verified=True, support=True, obstacle_pair_active=True)


def test_current_verified_top_uses_existing_contact_band_not_air_zero_descent_floor():
    air = bounded_fixture()
    air_context = capture(air)
    top = bounded_fixture()
    set_top(top)
    top_context = capture(top)
    assert air_context["clearance_margin_m"] == pytest.approx(air_context["clearance_m"])
    assert top_context["clearance_margin_m"] == -.015
    assert top_context["clearance_m"]-top_context["clearance_margin_m"] > 0.
    assert top_context["clearance_floor_semantics"] == "current_verified_TOP_contact_existing_gap_band"
    assert not top_context["place_xy"]


@pytest.mark.parametrize("field,value", [
    ("contact_surface", "FRONT_WALL"), ("contact_surface", "EDGE"), ("contact_surface", None),
    ("top_surface_contact", False), ("top_surface_contact", 1),
    ("bearing_verified", False), ("bearing_verified", None),
    ("support", False), ("obstacle_pair_active", False),
])
def test_unknown_wall_or_unverified_support_cannot_receive_top_descent_allowance(field, value):
    f = bounded_fixture()
    set_top(f)
    f.current[field] = value
    context = capture(f)
    assert context["clearance_margin_m"] == pytest.approx(context["clearance_m"])
    assert context["clearance_floor_semantics"] != "current_verified_TOP_contact_existing_gap_band"


@pytest.mark.parametrize("change", ["ground", "placed", "lateral", "invalid", "terminal", "missing_ground_reference"])
def test_no_stale_lift_hold_after_ground_or_placement_and_no_articulation_read_when_ineligible(change):
    f = bounded_fixture()
    if change == "ground":
        f.current["ground_contact"] = True
    elif change == "placed":
        f.current["within_top_xy"] = True
    elif change == "lateral":
        f.current["within_lateral_span"] = False
    elif change == "invalid":
        f.task["physical_evaluator"]["valid"] = False
    elif change == "terminal":
        f.task["termination_reason"] = "FALL"
    else:
        f.current["ground_relative_lift_m"] = None
    f.adapter.robot = None
    assert capture(f) is None
    assert not f.calls


@pytest.mark.parametrize("bad", ["points_empty", "points_nan", "points_shape", "quaternion_nan",
                                  "quaternion_shape", "quaternion_nonunit", "bottom_mismatch"])
def test_lowest_point_capture_fails_closed_on_invalid_or_inconsistent_geometry(bad):
    f = bounded_fixture()
    if bad == "points_empty":
        f.points = ()
    elif bad == "points_nan":
        f.points = ((0., 0., float("nan")),)
    elif bad == "points_shape":
        f.points = ((0., 0.),)
    elif bad == "quaternion_nan":
        f.adapter.robot.data.body_link_quat_w[0, f.body_index, 0] = float("nan")
    elif bad == "quaternion_shape":
        f.adapter.robot.data.body_link_quat_w = f.adapter.robot.data.body_link_quat_w[:, :, :3]
    elif bad == "quaternion_nonunit":
        f.adapter.robot.data.body_link_quat_w[0, f.body_index] *= 2.
    else:
        f.wheel.bottom_w_m = (*f.wheel.bottom_w_m[:2], f.wheel.bottom_w_m[2]+.001)
        f.current["clearance_m"] += .001
    with pytest.raises(NominalGeometryError):
        capture(f)


@pytest.mark.parametrize("bad", ["q", "jacobian_x", "jacobian_z", "clearance", "margin", "native",
                                  "bias", "previous", "slew_nan", "slew_zero", "placed", "ground", "indices"])
def test_bounded_target_correction_rejects_bad_context_before_any_output(bad):
    f, context, native = target_fixture()
    bias = (0.,)*12
    if bad == "q":
        context["physical_q_rad"] = (float("nan"), 0.)
    elif bad.startswith("jacobian"):
        context[bad + "_m_per_rad"] = (0., float("inf"))
    elif bad in ("clearance", "margin"):
        context["clearance_m" if bad == "clearance" else "clearance_margin_m"] = float("nan")
    elif bad in ("native", "bias"):
        row = list(native if bad == "native" else bias)
        row[7] = float("inf")
        if bad == "native":
            native = row
        else:
            bias = row
    elif bad == "previous":
        f.adapter._final_drive_servo_deg[SERVO_ORDER[7]] = float("nan")
    elif bad.startswith("slew"):
        f.adapter.servo_target_mapper.maximum_delta_deg = float("nan") if bad == "slew_nan" else 0.
    elif bad in ("placed", "ground"):
        context["place_xy" if bad == "placed" else "ground_contact"] = True
    else:
        context["canonical_servo_indices"] = (4, 5)
    with pytest.raises(NominalGeometryError):
        correct(f, context, native, bias)


def test_p12_retains_legacy_platform_floor_and_no_new_collider_requirement():
    f = fixture(phase="P12", canonical_zero=True)
    kwargs = dict(adapter=f.adapter, observation=f.observation, source_frame=f.source,
                  task_snapshot=f.task, clearance_margin_m=.015, physics_tick=f.dispatch_tick)
    old_context = capture_nominal_geometry_context(**kwargs, mode=FUNCTIONAL_RR_MODE)
    new_context = capture_nominal_geometry_context(**kwargs, mode=BOUNDED_RR_MODE)
    native = (0.,)*4 + (-20., 0.) + (0.,)*6
    old, old_evidence = correct(f, old_context, native)
    new, new_evidence = correct(f, new_context, native)
    assert old == new
    assert old_evidence["projection"] == new_evidence["projection"]
    assert new_context["geometry_model"] == "link_origin_proxy_sensor_bottom_pose_dependence_not_modelled"
    assert new_context["clearance_margin_m"] == .015
