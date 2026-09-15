"""Measured, nominal-only rear-leg pre-placement target adjustment.

The fixed-base first-order model adjusts a nominal target, not the current PPO
residual. It cannot guarantee the next physical displacement under contacts.
The frozen mapper is advanced elsewhere exactly once; this module only reads.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from wlr50_clean.infrastructure.command_batch import (
    SERVO_ORDER, SERVO_COMMAND_SIGN, WHEEL_ORDER, servo_limits_deg,
)
from wlr50_clean.infrastructure.robot_adapter import bounded_drive_feedback_step
from wlr50_clean.sensing.geometry import WHEEL_JOINT_TO_BODY
from .semantic_nominal_projection import project_nominal_downward

MODE = "preplace_world_down_nominal_advisory_v1"
FUNCTIONAL_RR_MODE = "functional_rr_preplace_nominal_advisory_v2"
BOUNDED_RR_MODE = "contact_aware_bounded_rr_nominal_v3"
CONTEXT_SCHEMA = "wlr50_clean.semantic_nominal_geometry_context.v1"
EVIDENCE_SCHEMA = "wlr50_clean.semantic_nominal_geometry.v1"
ACTIVE = {"P09": ("RR", (6, 7), 3), "P12": ("RL", (4, 5), 2)}


class NominalGeometryError(ValueError):
    """A required current-state measurement/target contract was not valid."""


def _member(owner, name):
    return owner.get(name) if isinstance(owner, Mapping) else getattr(owner, name, None)


def _finite(values, length, label):
    try:
        row = tuple(float(v) for v in values)
    except (TypeError, ValueError) as exc:
        raise NominalGeometryError(f"invalid {label}") from exc
    if len(row) != length or not all(math.isfinite(v) for v in row):
        raise NominalGeometryError(f"nonfinite/wrong-size {label}")
    return row


def _close(a, b, tolerance, label):
    if (len(a) != len(b) or not all(math.isfinite(v) for v in (*a, *b))
            or max((abs(x-y) for x, y in zip(a, b)), default=0.) > tolerance):
        raise NominalGeometryError(f"current-state mismatch: {label}")


def _physical_deg_to_rad(adapter, index, value):
    name = SERVO_ORDER[index]
    return math.radians(float(adapter.standing_pose_deg[name]) + SERVO_COMMAND_SIGN[name] * value)


def capture_nominal_geometry_context(*, adapter, observation, source_frame,
                                     task_snapshot, clearance_margin_m, physics_tick,
                                     mode=MODE, minimum_lift_gain_m=None, workspace_min_m=None,
                                     collider_local_points=None, top_gap_min_m=None):
    """Read a same-state wheel-link Jacobian without a write, step or reset.

    PhysX dense Jacobians are at each link COM in world axes:
    https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/docs/Articulations.html#jacobian
    The semantic sensor rotates collider-local vertices with the live pose.
    Legacy modes approximate its bottom derivative with link-origin velocity;
    bounded RR v3 instead differentiates the current lowest collider vertex.
    Neither model changes the evaluator or models contact-induced body motion.
    """
    phase = str(_member(source_frame, "state_id"))
    if mode not in (MODE, FUNCTIONAL_RR_MODE, BOUNDED_RR_MODE):
        raise NominalGeometryError("unknown nominal geometry mode")
    if phase not in ACTIVE:
        return None
    ev = task_snapshot.get("physical_evaluator", {})
    if ev.get("valid") is not True or ev.get("termination_reason") is not None or task_snapshot.get("termination_reason") is not None:
        return None
    leg, indices, wheel_index = ACTIVE[phase]
    current = ev.get("current_legs", {}).get(leg)
    if not isinstance(current, Mapping):
        raise NominalGeometryError("missing current rear-leg geometry")
    for flag in ("within_top_xy", "ground_contact", "within_lateral_span"):
        if type(current.get(flag)) is not bool:
            raise NominalGeometryError(f"missing physical boolean: {flag}")
    tick = _member(source_frame, "physics_tick")
    obs_tick = _member(observation, "physics_tick")
    now = _member(observation, "simulation_time_s")
    if (type(tick) is not int or tick < 0 or tick != obs_tick
            or type(physics_tick) is not int or physics_tick < tick
            or not isinstance(now, (int, float)) or not math.isfinite(now)
            or not math.isclose(now, tick/120., rel_tol=0., abs_tol=1e-8)
            or not math.isclose(float(_member(source_frame, "sim_time_s")), now, rel_tol=0., abs_tol=1e-8)):
        raise NominalGeometryError("nominal geometry observation/controller clock mismatch")
    ev_time = ev.get("simulation_time_s")
    if (type(ev.get("physics_tick")) is not int or ev["physics_tick"] != tick
            or not isinstance(ev_time, (int, float)) or not math.isfinite(ev_time)
            or not math.isclose(ev_time, now, rel_tol=0., abs_tol=1e-8)):
        raise NominalGeometryError("nominal geometry physical evaluator clock mismatch")
    # Never hold a previous peak pose on a new grounded attempt. The residual
    # and the finite source sequence remain free to create the next live lift.
    if current["within_top_xy"] or current["ground_contact"] or not current["within_lateral_span"]:
        return None
    clearance, margin = _finite((current["clearance_m"], clearance_margin_m), 2, "clearance")
    if margin <= 0.:
        raise NominalGeometryError("positive existing clearance margin required")
    floor_semantics = "existing_above_top_margin"
    if mode in (FUNCTIONAL_RR_MODE, BOUNDED_RR_MODE) and leg == "RR":
        # This is a nominal descent advisory, never an action/lift entry gate.
        # Far from the front, preserve the established ground-relative lift;
        # below the top, do not force a timed nominal descent before carry.
        # The residual remains independently available in all twelve channels.
        lift_scale, workspace = _finite((minimum_lift_gain_m, workspace_min_m), 2, "RR physical scales")
        if lift_scale <= 0.:
            raise NominalGeometryError("positive existing lift scale required")
        gain = current.get("ground_relative_lift_m")
        distance = _finite((current.get("front_distance_m"),), 1, "RR front distance")[0]
        if gain is None:
            # Missing ground reference cannot be invented. Do not claim a
            # clearance guarantee or suppress safe probing from absent data.
            return None
        gain = _finite((gain,), 1, "RR ground-relative lift")[0]
        if distance < workspace:
            margin = clearance - gain + min(max(gain, 0.), lift_scale)
            floor_semantics = "measured_ground_reference_existing_lift_scale"
        else:
            margin = min(clearance, 0.)
            floor_semantics = "current_below_top_or_surface_floor_before_cross"
        if (mode == BOUNDED_RR_MODE and current.get("contact_surface") == "TOP"
                and current.get("top_surface_contact") is True
                and current.get("bearing_verified") is True
                and current.get("support") is True and current.get("obstacle_pair_active") is True):
            # A currently supported TOP corner is not a freely airborne wheel.
            # Use the existing allowable contact band, not a perpetual zero-
            # descent AIR constraint. Crossing/placement/safety stay unchanged.
            contact_floor = _finite((top_gap_min_m,), 1, "existing TOP gap floor")[0]
            margin = min(clearance, contact_floor)
            floor_semantics = "current_verified_TOP_contact_existing_gap_band"

    import torch
    robot = adapter.robot
    names = tuple(robot.body_names)
    joint_names = tuple(robot.joint_names)
    servo_ids = tuple(adapter.joint_map.servo_ids)
    wheel_ids = tuple(adapter.joint_map.wheel_ids)
    if len(servo_ids) != 8 or len(wheel_ids) != 4 or len(set(servo_ids + wheel_ids)) != 12:
        raise NominalGeometryError("twelve distinct actuator DOFs required")
    if any(type(j) is not int or not 0 <= j < len(joint_names) for j in servo_ids + wheel_ids):
        raise NominalGeometryError("actuator DOF indices are outside this articulation")
    if any(joint_names[j] != name for j, name in zip(servo_ids, SERVO_ORDER)):
        raise NominalGeometryError("canonical servo/DOF mapping differs")
    if any(joint_names[j] != name for j, name in zip(wheel_ids, WHEEL_ORDER)):
        raise NominalGeometryError("canonical wheel/DOF mapping differs")
    wheel_name = WHEEL_ORDER[wheel_index]
    body_name = WHEEL_JOINT_TO_BODY[wheel_name]
    if names.count(body_name) != 1:
        raise NominalGeometryError("rear wheel body mapping is not unique")
    body_index = names.index(body_name)
    fixed = robot.is_fixed_base
    if type(fixed) is not bool:
        raise NominalGeometryError("explicit articulation fixed-base flag required")
    body_row = body_index - int(fixed)
    dof_offset = 0 if fixed else 6
    columns = tuple(servo_ids[i] + dof_offset for i in indices)
    data = robot.data

    def tensor(value, shape, label):
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                or not value.is_floating_point() or not bool(torch.isfinite(value).all().item())):
            raise NominalGeometryError(f"invalid current tensor: {label}")
        return value

    dof_count = len(joint_names)
    q_all = tensor(data.joint_pos, (1, dof_count), "joint_pos")
    qd_all = tensor(data.joint_vel, (1, dof_count), "joint_vel")
    link_pos = tensor(data.body_link_pos_w, (1, len(names), 3), "body_link_pos_w")
    com_pos = tensor(data.body_com_pos_w, (1, len(names), 3), "body_com_pos_w")
    com_vel = tensor(data.body_com_vel_w, (1, len(names), 6), "body_com_vel_w")
    link_vel = tensor(data.body_link_vel_w, (1, len(names), 6), "body_link_vel_w")
    jac = tensor(robot.root_physx_view.get_jacobians(),
                 (1, len(names)-int(fixed), 6, dof_count+dof_offset), "world_COM_jacobian")
    if body_row < 0 or body_row >= jac.shape[1]:
        raise NominalGeometryError("invalid wheel Jacobian row")
    wheel = _member(observation, "wheels")[wheel_name]
    measured_center = _finite(_member(wheel, "center_w_m"), 3, "measured wheel center")
    measured_bottom = _finite(_member(wheel, "bottom_w_m"), 3, "measured wheel bottom")
    if _member(wheel, "geometry_verified") is not True:
        raise NominalGeometryError("unverified measured wheel geometry")
    live_center = tuple(link_pos[0, body_index].detach().cpu().tolist())
    _close(live_center, measured_center, 1e-6, "wheel link versus sensor center")
    obstacle_top = float(_member(_member(observation, "obstacle"), "top_z_m"))
    _close((clearance,), (measured_bottom[2]-obstacle_top,), 1e-8, "evaluator clearance")
    q = tuple(q_all[0, [servo_ids[i] for i in indices]].detach().cpu().tolist())
    observed_joints = _member(observation, "joints")
    expected_q = tuple(_physical_deg_to_rad(adapter, i,
                       float(_member(observed_joints[SERVO_ORDER[i]], "position_deg"))) for i in indices)
    _close(q, expected_q, 1e-6, "physical q versus sensor canonical angles")

    offset = link_pos[0, body_index] - com_pos[0, body_index]
    j_com = jac[0, body_row]
    # For each column: v_link = v_com + omega x (p_link-p_com).
    j_link_linear = j_com[:3] + torch.cross(
        j_com[3:].transpose(0, 1), offset.expand(j_com.shape[1], 3), dim=1).transpose(0, 1)
    if fixed:
        generalized_vel = qd_all[0]
    else:
        root_vel = tensor(data.root_com_vel_w, (1, 6), "root_com_vel_w")
        generalized_vel = torch.cat((root_vel[0], qd_all[0]))
    predicted_com = j_com @ generalized_vel
    predicted_link = j_link_linear @ generalized_vel
    com_error = float((predicted_com-com_vel[0, body_index]).abs().max().item())
    link_error = float((predicted_link-link_vel[0, body_index, :3]).abs().max().item())
    # ABI/reference-point consistency, not a task/clearance/success gate.
    if max(com_error, link_error) > 1e-3:
        raise NominalGeometryError("Jacobian does not match same-state COM/link velocities")
    jx = tuple(j_link_linear[0, list(columns)].detach().cpu().tolist())
    jz = tuple(j_link_linear[2, list(columns)].detach().cpu().tolist())
    point_evidence = {}
    if mode == BOUNDED_RR_MODE and leg == "RR":
        # The current semantic reader rotates immutable collider vertices on
        # every tick. Its bottom is NOT a translated constant world extent.
        # X remains wheel-link origin; Z follows the actual lowest collider
        # vertex, including its rotational point-velocity term.
        import numpy as np
        try:
            points = np.asarray(collider_local_points, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise NominalGeometryError("invalid collider-local points") from exc
        if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
            raise NominalGeometryError("bounded RR geometry requires measured collider-local points")
        orientation = tensor(data.body_link_quat_w, (1, len(names), 4), "body_link_quat_w")
        w, x, y, z = orientation[0, body_index].detach().cpu().tolist()
        if not math.isclose(w*w+x*x+y*y+z*z, 1., abs_tol=1e-5):
            raise NominalGeometryError("nonunit live wheel orientation")
        vx, vy, vz = points[:, 0], points[:, 1], points[:, 2]
        tx, ty, tz = 2*(y*vz-z*vy), 2*(z*vx-x*vz), 2*(x*vy-y*vx)
        rotated = np.column_stack((vx+w*tx+y*tz-z*ty,
                                   vy+w*ty+z*tx-x*tz, vz+w*tz+x*ty-y*tx))
        lowest = int(np.argmin(rotated[:, 2]))
        point = tuple(float(a+b) for a,b in zip(live_center, rotated[lowest]))
        _close((point[2],), (measured_bottom[2],), 2e-6, "lowest live collision point versus sensor bottom")
        point_offset = torch.tensor(rotated[lowest], dtype=j_com.dtype, device=j_com.device)
        j_bottom = j_link_linear + torch.cross(j_com[3:].transpose(0, 1),
            point_offset.expand(j_com.shape[1], 3), dim=1).transpose(0, 1)
        point_evidence = {"lowest_collider_vertex_index": lowest,
            "lowest_collider_point_world_m": point, "collider_point_count": len(points),
            "link_origin_jacobian_z_m_per_rad": jz,
            "bottom_derivative_scope": "current_minimum_vertex_piecewise_linear_not_contact_dynamics"}
        jz = tuple(j_bottom[2, list(columns)].detach().cpu().tolist())
    return {
        "schema": CONTEXT_SCHEMA, "mode": mode, "source_phase_id": phase,
        "source_control_tick": tick, "dispatch_physics_tick": physics_tick,
        "source_sim_time_s": now, "active_leg": leg,
        "canonical_servo_indices": indices, "physical_q_rad": q,
        "jacobian_x_m_per_rad": jx, "jacobian_z_m_per_rad": jz,
        "clearance_m": clearance, "clearance_margin_m": margin,
        "clearance_floor_semantics": floor_semantics,
        "place_xy": False, "ground_contact": False,
        "body_name": body_name, "body_index": body_index, "jacobian_body_row": body_row,
        "physical_joint_ids": tuple(servo_ids[i] for i in indices),
        "jacobian_joint_columns": columns, "fixed_base": fixed,
        "link_minus_com_world_m": tuple(offset.detach().cpu().tolist()),
        "COM_velocity_identity_max_error": com_error,
        "link_velocity_identity_max_error_m_s": link_error,
        "geometry_model": ("current_pose_collider_lowest_vertex" if point_evidence else
                           "link_origin_proxy_sensor_bottom_pose_dependence_not_modelled"),
        "jacobian_model": ("world_COM_shifted_to_link_origin_X_and_lowest_collider_Z_joint_columns"
                           if point_evidence else "world_COM_shifted_to_measured_link_origin_fixed_base_joint_columns"),
        "contact_surface": current.get("contact_surface"), **point_evidence,
        "physical_motion_guaranteed": False,
    }


def correct_nominal_geometry(*, adapter, native_full12, controller_bias_full12, context):
    """Return adjusted native nominal and evidence; never consume current r."""
    native = _finite(native_full12, 12, "raw native nominal")
    bias = _finite(controller_bias_full12, 12, "bounded controller bias")
    if not isinstance(context, Mapping) or context.get("schema") != CONTEXT_SCHEMA or context.get("mode") not in (MODE, FUNCTIONAL_RR_MODE, BOUNDED_RR_MODE):
        raise NominalGeometryError("versioned nominal geometry context required")
    phase = context.get("source_phase_id")
    if phase not in ACTIVE or tuple(context.get("canonical_servo_indices", ())) != ACTIVE[phase][1]:
        raise NominalGeometryError("active leg/context index mismatch")
    indices = ACTIVE[phase][1]
    if context.get("place_xy") is not False or context.get("ground_contact") is not False:
        raise NominalGeometryError("ineligible context must bypass before mapper dispatch")
    if context["mode"] == BOUNDED_RR_MODE and phase == "P09":
        return _bounded_rr_correction(adapter=adapter, native=native, bias=bias, context=context)
    q = _finite(context["physical_q_rad"], 2, "current physical q")
    previous = tuple(float(adapter._final_drive_servo_deg[name]) for name in SERVO_ORDER)
    maximum_delta = float(adapter.servo_target_mapper.maximum_delta_deg)
    if not math.isfinite(maximum_delta) or maximum_delta <= 0.:
        raise NominalGeometryError("invalid frozen final slew")
    old_targets, lower_targets, upper_targets = [], [], []
    for index in indices:
        name = SERVO_ORDER[index]
        lo, hi = servo_limits_deg(name)
        target = bounded_drive_feedback_step(previous_deg=previous[index], native_deg=native[index],
            bias_deg=bias[index], maximum_delta_deg=maximum_delta, lower_deg=lo, upper_deg=hi)
        lower = max(lo, previous[index]-maximum_delta)
        upper = min(hi, previous[index]+maximum_delta)
        bounds = sorted((_physical_deg_to_rad(adapter, index, lower),
                         _physical_deg_to_rad(adapter, index, upper)))
        old_targets.append(_physical_deg_to_rad(adapter, index, target))
        lower_targets.append(bounds[0]); upper_targets.append(bounds[1])
    clearance, margin = _finite((context["clearance_m"], context["clearance_margin_m"]), 2, "clearance")
    if margin <= 0. and not (context["mode"] == FUNCTIONAL_RR_MODE and phase == "P09"):
        raise NominalGeometryError("invalid existing margin")
    result = project_nominal_downward(
        nominal_delta_rad=tuple(t-p for t,p in zip(old_targets,q)),
        jacobian_x_m_per_rad=context["jacobian_x_m_per_rad"],
        jacobian_z_m_per_rad=context["jacobian_z_m_per_rad"],
        delta_lower_rad=tuple(t-p for t,p in zip(lower_targets,q)),
        delta_upper_rad=tuple(t-p for t,p in zip(upper_targets,q)),
        available_descent_m=max(0.,clearance-margin), place_xy=False)
    adjusted = list(native)
    status = result.status
    desired = None
    if not result.feasible:
        # An honest non-guaranteeing fallback, not a new failure/reset or frozen pose.
        status = "degraded_bypass_" + status
    elif result.nominal_correction_rad != (0., 0.):
        desired = []
        for index, actual, delta in zip(indices, q, result.corrected_delta_rad):
            name = SERVO_ORDER[index]
            value = (math.degrees(actual+delta)-float(adapter.standing_pose_deg[name])) / SERVO_COMMAND_SIGN[name]
            adjusted[index] = value-bias[index]
            desired.append(value)
        # Active-only inverse: desired-(m+c), not desired-old_bounded_nominal.
        for index, value in zip(indices, desired):
            lo, hi = servo_limits_deg(SERVO_ORDER[index])
            reconstructed = bounded_drive_feedback_step(previous_deg=previous[index],
                native_deg=adjusted[index], bias_deg=bias[index],
                maximum_delta_deg=maximum_delta, lower_deg=lo, upper_deg=hi)
            _close((value,), (reconstructed,), 1e-9, "adjusted zero-policy final nominal")
    evidence = {
        "schema": EVIDENCE_SCHEMA, "mode": context["mode"], "status": status,
        "context": dict(context), "projection": dict(result.proof),
        "old_zero_policy_physical_target_rad": old_targets,
        "desired_zero_policy_canonical_target_deg": desired,
        "nominal_geometry_adjustment_full12": [a-b for a,b in zip(adjusted,native)],
        "current_policy_residual_used": False, "physical_motion_guaranteed": False,
        "degraded_bypass_is_clearance_guarantee": False,
    }
    return tuple(adjusted), evidence


def _bounded_rr_correction(*, adapter, native, bias, context):
    """Non-integrating local model, existing reserve, explicit infeasibility.

    No assertion of whole-body feasibility: body/support candidates must supply
    missing reachability. The correction envelope is relative to THIS mapped
    source, never yesterday's corrected output; it cannot chase a hard limit.
    """
    from .semantic_headroom import SERVO_RESERVE_DEG
    from wlr50_clean.infrastructure.servo_target_mapper import SERVO_TRACKING_COMPENSATION_MAX_DEG
    indices = ACTIVE["P09"][1]
    q = _finite(context["physical_q_rad"], 2, "physical q")
    slew = float(adapter.servo_target_mapper.maximum_delta_deg)
    if not math.isfinite(slew) or slew <= 0.:
        raise NominalGeometryError("invalid frozen final slew")
    clearance, margin = _finite((context["clearance_m"], context["clearance_margin_m"]), 2, "clearance")
    jx = _finite(context["jacobian_x_m_per_rad"], 2, "Jacobian X")
    jz = _finite(context["jacobian_z_m_per_rad"], 2, "Jacobian Z")
    cap = SERVO_TRACKING_COMPENSATION_MAX_DEG
    trust = math.radians(SERVO_RESERVE_DEG)
    targets, lower, upper, fallback = [], [], [], []
    tracking_valid = True
    bands, recovering = [], []
    for i, actual in zip(indices, q):
        name = SERVO_ORDER[i]
        previous = float(adapter._final_drive_servo_deg[name])
        if not math.isfinite(previous):
            raise NominalGeometryError("nonfinite previous final target")
        hard_lo, hard_hi = servo_limits_deg(name)
        base = native[i]+bias[i]
        band_lo, band_hi = max(hard_lo+SERVO_RESERVE_DEG, base-cap), min(hard_hi-SERVO_RESERVE_DEG, base+cap)
        if band_lo > band_hi:
            # An already out-of-band source is reported, not legitimized by a
            # relaxed hard limit. Move toward the existing operating interval.
            band_lo = band_hi = max(hard_lo+SERVO_RESERVE_DEG, min(hard_hi-SERVO_RESERVE_DEG, base))
        lo, hi = max(band_lo, previous-slew), min(band_hi, previous+slew)
        recovering.append(lo > hi)
        if lo > hi:
            toward = max(band_lo, min(band_hi, previous))
            lo = hi = previous+max(-slew, min(slew, toward-previous))
        desired = max(lo, min(hi, base))
        p_lo, p_hi = sorted((_physical_deg_to_rad(adapter, i, lo), _physical_deg_to_rad(adapter, i, hi)))
        fallback.append(_physical_deg_to_rad(adapter, i, max(lo, min(hi, previous))))
        t_lo, t_hi = max(p_lo, actual-trust), min(p_hi, actual+trust)
        if t_lo > t_hi:
            tracking_valid = False
            t_lo, t_hi = p_lo, p_hi
        lower.append(t_lo-actual); upper.append(t_hi-actual)
        targets.append(max(t_lo, min(t_hi, _physical_deg_to_rad(adapter, i, desired)))-actual)
        bands.append((band_lo, band_hi))
    result = None
    if tracking_valid:
        result = project_nominal_downward(nominal_delta_rad=targets,
            jacobian_x_m_per_rad=jx,
            jacobian_z_m_per_rad=jz,
            delta_lower_rad=lower, delta_upper_rad=upper,
            available_descent_m=max(0., clearance-margin), place_xy=False)
    needs_body = result is None or not result.feasible
    if result is not None and result.feasible:
        selected, status = result.corrected_delta_rad, result.status
    elif tracking_valid:
        # Best vertical step inside the bounded local box, not an unconstrained
        # reissue of the old downward source. Report that the requested linear
        # clearance/forward combination has NO feasible local solution.
        selected = tuple(hi if row > 0 else lo if row < 0 else max(lo, min(hi, 0.))
                         for row,lo,hi in zip(jz, lower, upper))
        status = "bounded_best_vertical_step_needs_whole_body_height"
    else:
        selected = tuple(t-p for t,p in zip(fallback,q))
        status = "bounded_hold_or_inward_recovery_tracking_exceeds_linear_trust"
    adjusted, desired = list(native), []
    for i, actual, delta in zip(indices, q, selected):
        name = SERVO_ORDER[i]
        value = (math.degrees(actual+delta)-float(adapter.standing_pose_deg[name]))/SERVO_COMMAND_SIGN[name]
        adjusted[i] = value-bias[i]
        desired.append(value)
        lo, hi = servo_limits_deg(name)
        reconstructed = bounded_drive_feedback_step(previous_deg=float(adapter._final_drive_servo_deg[name]),
            native_deg=adjusted[i], bias_deg=bias[i], maximum_delta_deg=slew,
            lower_deg=lo, upper_deg=hi)
        _close((value,), (reconstructed,), 1e-9, "bounded adjusted zero-policy final nominal")
    at_envelope = any(min(value-lo, hi-value) <= 1e-6 for value,(lo,hi) in zip(desired,bands))
    return tuple(adjusted), {"schema": EVIDENCE_SCHEMA, "mode": context["mode"], "status": status,
        "context": dict(context), "projection": dict(result.proof) if result is not None else None,
        "desired_zero_policy_canonical_target_deg": desired,
        "nominal_geometry_adjustment_full12": [a-b for a,b in zip(adjusted,native)],
        "operating_reserve_deg": SERVO_RESERVE_DEG, "mapped_source_correction_envelope_deg": cap,
        "actual_q_linear_trust_deg": SERVO_RESERVE_DEG, "tracking_inside_linear_trust": tracking_valid,
        "source_relative_operating_bands_deg": bands, "operating_envelope_reached": at_envelope,
        "slew_recovery_toward_source_band": recovering,
        "predicted_local_delta_x_m": sum(a*b for a,b in zip(jx, selected)),
        "predicted_local_delta_z_m": sum(a*b for a,b in zip(jz, selected)),
        "needs_whole_body_height_or_support_reconfiguration": needs_body or at_envelope,
        "current_policy_residual_used": False, "physical_motion_guaranteed": False,
        "fallback_is_clearance_guarantee": False, "nonintegrating_source_relative_bound": True}
