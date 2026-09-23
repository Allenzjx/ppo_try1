"""Pure P09 support-wheel envelope; no simulator, clock advance or policy edit.

Draft for explicit versioned adoption. Positive canonical wheel targets are an
actuator direction, not a claim about measured traction. Source evidence is
committed in the ordinary ACK, including when the envelope is unarmed. The
next call verifies that ACK; it never advances a second source or keeps a latch.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

from wlr50_clean.infrastructure.command_batch import (
    PHYSICS_DT_S, WHEEL_ORDER, WHEEL_VELOCITY_LIMIT_RAD_S,
)
from wlr50_clean.ppo.semantic_rr_capture_context import verified_current_support

MODE = "rr_capture_support_forward_projection_v1"
SEMANTICS = "P09_post_source_committed_stop_current_RR_AIR_Q_cross_supported_FL_FR_RL_nonnegative_depth_gap_floor_previous_FINAL_1p8_slew_TOP_release"
CONTEXT_SCHEMA = "wlr50_clean.rr_carry_wheel_context.v1"
EVIDENCE_SCHEMA = "wlr50_clean.rr_carry_wheel_evidence.v1"
ACK_KEY = "rr_carry_wheel_evidence"
WHEEL_RATE_RAD_S2 = 1.8  # Exact existing execution-profile residual wheel rate.
SUPPORT_LEGS = ("FL", "FR", "RL")
SUPPORT_INDICES = (8, 9, 10)


def _get(obj, name, default=None):
    return obj.get(name, default) if isinstance(obj, Mapping) else getattr(obj, name, default)


def _number(value, label):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value)):
        raise ValueError(f"finite numeric {label} required")
    return float(value)


def _vector(value, n, label):
    if not isinstance(value, (list, tuple)) or len(value) != n:
        raise ValueError(f"{label} must have {n} entries")
    return tuple(_number(v, label) for v in value)


def _tick(value):
    return type(value) is int and value >= 0


def _clip(value, lo, hi):
    return max(lo, min(hi, value))


def _source_evidence(provider, *, phase, source_tick, dispatch_tick):
    """Read existing finite-source state; missing evidence cannot arm anything."""
    out = dict(source_control_tick=source_tick, dispatch_physics_tick=dispatch_tick,
        phase=phase, source_phase="P09", available=False, endpoint_issued=False,
        advanced_this_tick=False, fresh_wheel_owners=[], finite_wheel_owners=[],
        source_wheel_target_rad_s=None, sample_tick=None, endpoint_tick=None,
        last_wheel_stop_tick=None, last_wheel_stop_source_time_s=None,
        last_wheel_stop_channels=[], reason="P09_source_layer_missing")
    layers = _get(provider, "_continuous_layers", ())
    for layer in layers:
        sample = layer.get("sample")
        if sample is None:
            continue
        if layer.get("advanced_this_tick") is True:
            channels = sorted({name for group in sample.atomic_groups
                for name in group.channels if name in WHEEL_ORDER})
            if channels:
                out["fresh_wheel_owners"].append({"stage": layer["stage"], "channels": channels})
        if layer.get("stage") in ("P07", "P09") and sample.endpoint_issued is not True:
            out["finite_wheel_owners"].append(layer["stage"])
    matches = [layer for layer in layers if layer.get("stage") == "P09"]
    if len(matches) != 1 or matches[0].get("sample") is None:
        return out
    layer = matches[0]
    motion, sample = layer["motion"], layer["sample"]
    source = motion.phase
    groups = [group for group in source.atomic_groups if set(group.channels).intersection(WHEEL_ORDER)]
    if not groups:
        out["reason"] = "no_authored_wheel_stop"
        return out
    last_time = max(_number(group.time_s, "source group time") for group in groups)
    last_tick = motion._scaled_source_tick(last_time)  # Pure existing scheduler quantization.
    stop_waypoints = [waypoint for waypoint in source.waypoints
        if motion._scaled_source_tick(waypoint.time_s) == last_tick]
    stop_channels = sorted({name for group in groups if group.time_s == last_time
        for name in group.channels if name in WHEEL_ORDER})
    wheels = _vector(sample.full12, 12, "source Full12")[8:]
    end = _vector(source.end_full12, 12, "source endpoint Full12")[8:]
    endpoint_tick = round(_number(motion.effective_active_duration_s, "source endpoint duration")
                          * _number(motion.physics_hz, "source physics rate"))
    stop_is_zero = bool(stop_waypoints and all(
        _vector(w.full12, 12, "stop waypoint Full12")[8:] == (0.,)*4 for w in stop_waypoints))
    out.update(available=True, endpoint_issued=sample.endpoint_issued is True,
        advanced_this_tick=layer.get("advanced_this_tick") is True,
        source_wheel_target_rad_s=list(wheels), sample_tick=sample.tick_index,
        endpoint_tick=endpoint_tick, last_wheel_stop_tick=last_tick,
        last_wheel_stop_source_time_s=last_time, last_wheel_stop_channels=stop_channels,
        authored_final_stop_zero=stop_is_zero and end == (0.,)*4,
        reason="source_state_read_only_not_independent_actuator_ACK")
    return out


def _committed_endpoint(source, ack, previous_write_count):
    reasons = []
    if not isinstance(ack, Mapping):
        return False, ["previous_ACK_missing"]
    previous = ack.get(ACK_KEY, {}).get("source_evidence", {})
    if not isinstance(previous, Mapping):
        previous = {}
    checks = {
        "adjacent_ACK_tick": _tick(source["dispatch_physics_tick"])
            and ack.get("physics_tick") == source["dispatch_physics_tick"]-1,
        "pre_dispatch_write_count": _tick(previous_write_count) and previous_write_count > 0
            and type(ack.get("write_count")) is int and ack["write_count"] == previous_write_count,
        "one_actual_write": ack.get("articulation_writes_this_call") == 1,
        "previous_committed_source_phase": previous.get("phase") == "P09"
            and previous.get("source_phase") == "P09",
        "previous_source_tick_adjacent": _tick(source["source_control_tick"])
            and previous.get("source_control_tick") == source["source_control_tick"]-1,
        "previous_dispatch_matches_ACK": previous.get("dispatch_physics_tick") == ack.get("physics_tick"),
        "previous_endpoint_issued": previous.get("endpoint_issued") is True
            and previous.get("advanced_this_tick") is True,
        "previous_sample_adjacent": _tick(source.get("sample_tick"))
            and previous.get("sample_tick") == source["sample_tick"]-1,
        "same_source_stop": previous.get("last_wheel_stop_tick") == source.get("last_wheel_stop_tick")
            and previous.get("endpoint_tick") == source.get("endpoint_tick")
            and previous.get("last_wheel_stop_source_time_s") == source.get("last_wheel_stop_source_time_s")
            and previous.get("last_wheel_stop_channels") == source.get("last_wheel_stop_channels"),
        "previous_authored_stop_zero": previous.get("authored_final_stop_zero") is True,
        "previous_source_held_zero": previous.get("source_wheel_target_rad_s") == [0.]*4,
    }
    try:
        checks["previous_requested_wheels_zero"] = _vector(ack.get("requested_full12"), 12, "ACK request")[8:] == (0.,)*4
    except ValueError:
        checks["previous_requested_wheels_zero"] = False
    for name, result in checks.items():
        if not result:
            reasons.append(name)
    return not reasons, reasons


def build_rr_carry_wheel_context(*, task, observation, source_frame, nominal_provider,
        physics_tick, source_ack=None, previous_write_count=None, support_spec=None,
        wheel_rate_rad_s2=WHEEL_RATE_RAD_S2, mode="off"):
    """Read same-tick inputs AFTER the one source evaluation, BEFORE dispatch.

    Caller must pass the real pre-dispatch ACK/write count, not the new receipt.
    Commit result.source_evidence in ACK[ACK_KEY] even on an inactive call.
    No claim is made that all source/contact internals are individually observed.
    """
    if mode not in ("off", MODE):
        raise ValueError("unknown RR carry wheel mode")
    ev = task.get("physical_evaluator", {})
    phase, tick = task.get("stage_id"), _get(source_frame, "physics_tick")
    source = _source_evidence(nominal_provider, phase=phase, source_tick=tick, dispatch_tick=physics_tick)
    out = dict(schema=CONTEXT_SCHEMA, mode=mode, source_evidence=source,
        dispatch_physics_tick=physics_tick, source_control_tick=tick,
        phase=phase, envelope_active=False, action="bypass", reasons=[], semantics=SEMANTICS,
        gain=0., forward_floor_rad_s=0., selected_full12_indices=[], verified_support_legs=[],
        previous_final_wheel_rad_s=None, wheel_rate_rad_s2=wheel_rate_rad_s2,
        actor_flag_semantics="X409_current_envelope_armed_not_actual_projection_or_traction",
        source_proof_semantics="prior_committed_source_endpoint_ACK_then_adjacent_physics_tick_not_measured_stop")
    if mode == "off":
        out["reasons"] = ["mode_off"]
        return out
    if (not _tick(tick) or not _tick(physics_tick) or physics_tick < tick
            or _get(observation, "physics_tick") != tick or ev.get("physics_tick") != tick
            or _get(source_frame, "state_id") != phase):
        raise ValueError("RR wheel context requires a same-observation task/source clock")
    if _number(wheel_rate_rad_s2, "wheel rate") != WHEEL_RATE_RAD_S2:
        raise ValueError("use the unchanged existing 1.8 rad/s2 residual wheel rate")
    spec = nominal_provider.spec
    geometry = spec["geometry"]
    support = spec["support"] if support_spec is None else support_spec
    floor = _number(support["force_noise_floor_n"], "support noise floor")
    near = _number(geometry["xy_measurement_tolerance_m"], "depth near scale")
    deep = _number(geometry["workspace_max_m"], "depth taper scale")
    clear = _number(geometry["airborne_clearance_above_top_m"], "clearance scale")
    prior = _vector(spec["nominal"]["approach_wheel_prior_rad_s"], 4, "existing P01 wheel prior")
    if floor < 0. or not 0. <= near < deep or clear <= 0. or prior != (.3,)*4:
        raise ValueError("existing finite support/geometry/.3 wheel prior contract required")
    legs, hist = ev.get("current_legs", {}), ev.get("history", {})
    rr = legs.get("RR", {})
    supported = [leg for leg in SUPPORT_LEGS if verified_current_support(legs.get(leg, {}), floor)]
    committed, proof_reasons = _committed_endpoint(source, source_ack, previous_write_count)
    out.update(verified_support_legs=supported, source_commit_verified=committed,
        source_commit_reasons=proof_reasons, geometry_scales=dict(depth_near_m=near,
            depth_taper_m=deep, gap_taper_m=clear), existing_wheel_prior_rad_s=list(prior))
    checks = {
        "P09_only": phase == "P09",
        "valid_current_physics": ev.get("valid") is True and ev.get("termination_reason") is None
            and task.get("termination_reason") is None,
        "current_source_endpoint": source.get("available") is True
            and source.get("endpoint_issued") is True and source.get("advanced_this_tick") is True,
        "after_authored_stop": source.get("authored_final_stop_zero") is True
            and _tick(source.get("sample_tick")) and _tick(source.get("last_wheel_stop_tick"))
            and source["sample_tick"] > source["last_wheel_stop_tick"]
            and source.get("source_wheel_target_rad_s") == [0.]*4,
        "committed_endpoint": committed,
        "fresh_source_owner_wins": not source["fresh_wheel_owners"] and not source["finite_wheel_owners"],
        "FL_plus_other_measured_support": "FL" in supported and len(supported) >= 2,
        "RR_real_cross_history": hist.get("front_edge_crossed", {}).get("RR") is True,
        "not_RL_transfer": hist.get("placed", {}).get("RL") is False,
        "not_GROUND_or_wall": rr.get("ground_contact") is False
            and rr.get("contact_surface") in ("NONE", "TOP"),
    }
    out["reasons"] = [name for name, result in checks.items() if not result]
    if out["reasons"]:
        return out
    previous_final = _vector(source_ack.get("drive_target_full12"), 12, "previous FINAL Full12")[8:]
    if any(abs(v) > WHEEL_VELOCITY_LIMIT_RAD_S for v in previous_final):
        raise ValueError("previous FINAL wheel exceeds unchanged physical hard limit")
    gap = _number(rr.get("clearance_m"), "RR gap")
    distance = _number(rr.get("front_distance_m"), "RR front distance")
    booleans = ("air", "current_lift_valid", "within_top_xy", "within_lateral_span", "top_surface_contact")
    if any(type(rr.get(name)) is not bool for name in booleans):
        out["reasons"] = ["missing_current_RR_physical_boolean"]
        return out
    air = (rr["air"] and rr["current_lift_valid"] and rr["within_top_xy"]
        and rr["within_lateral_span"] and not rr["top_surface_contact"]
        and rr["contact_surface"] == "NONE" and hist.get("active_lift", {}).get("RR") is True and gap >= 0.)
    # TOP/contact or a lost current AIR geometry condition has NO forward floor.
    # Within this P09/source/support scope only, return to the original candidate
    # at the same rate. No phase, contact, or completion history is manufactured.
    gain = _clip((deep-distance)/(deep-near), 0., 1.) * _clip(gap/clear, 0., 1.) if air else 0.
    out.update(envelope_active=True, action="forward_floor" if air else "release_slew",
        gain=gain, forward_floor_rad_s=.3*gain, previous_final_wheel_rad_s=list(previous_final),
        selected_full12_indices=[8+SUPPORT_LEGS.index(leg) for leg in supported],
        rr_gap_m=gap, rr_front_distance_m=distance,
        rr_current_air_qualified=air, rr_top_surface_contact=rr["top_surface_contact"],
        reasons=["qualified_RR_AIR_support_forward_envelope" if air else "P09_return_to_original_candidate_no_floor"])
    return out


def project_rr_carry_wheels(candidate_full12, *, context, previous_final_wheel_rad_s,
        physics_dt_s=PHYSICS_DT_S):
    """Project a bounded pre-envelope Full12; use identically for N+r and N+0.

    This changes the deterministic environment mapping, NEVER raw actions/logp.
    The independent previous FINAL vector is supplied by the pre-dispatch audit.
    """
    candidate = _vector(candidate_full12, 12, "candidate Full12")
    if any(abs(v) > WHEEL_VELOCITY_LIMIT_RAD_S for v in candidate[8:]):
        raise ValueError("candidate must already obey the original wheel hard bounds")
    if context.get("schema") != CONTEXT_SCHEMA or context.get("mode") not in ("off", MODE):
        raise ValueError("versioned RR wheel context required")
    result, desired = list(candidate), list(candidate)
    armed = context.get("envelope_active") is True
    indices = tuple(context.get("selected_full12_indices", ())) if armed else ()
    if armed:
        if (context["mode"] != MODE or context.get("phase") != "P09"
                or context.get("action") not in ("forward_floor", "release_slew")
                or not indices or len(set(indices)) != len(indices)
                or not set(indices) <= set(SUPPORT_INDICES) or 8 not in indices):
            raise ValueError("invalid armed RR wheel scope")
        previous = _vector(previous_final_wheel_rad_s, 4, "independent previous FINAL wheels")
        recorded = _vector(context.get("previous_final_wheel_rad_s"), 4, "context previous FINAL wheels")
        if previous != recorded or any(abs(v) > WHEEL_VELOCITY_LIMIT_RAD_S for v in previous):
            raise ValueError("independent pre-dispatch FINAL wheels do not match context ACK")
        dt = _number(physics_dt_s, "physics dt")
        if not math.isclose(dt, PHYSICS_DT_S, rel_tol=0., abs_tol=1e-15):
            raise ValueError("unchanged 120 Hz wheel projection required")
        rate = _number(context.get("wheel_rate_rad_s2"), "wheel rate")
        if rate != WHEEL_RATE_RAD_S2:
            raise ValueError("unchanged wheel rate required")
        floor = _number(context.get("forward_floor_rad_s"), "forward floor")
        if not 0. <= floor <= .3:
            raise ValueError("bounded existing .3 forward floor required")
        for index in indices:
            if context["action"] == "forward_floor":
                desired[index] = max(candidate[index], floor)  # floor=0 still scoped nonnegative.
            result[index] = _clip(previous[index-8] + _clip(desired[index]-previous[index-8],
                -rate*dt, rate*dt), -WHEEL_VELOCITY_LIMIT_RAD_S, WHEEL_VELOCITY_LIMIT_RAD_S)
    return dict(schema=EVIDENCE_SCHEMA, mode=context["mode"],
        source_evidence=dict(context["source_evidence"]), context=dict(context),
        envelope_active=armed, actual_projection_changed=tuple(result) != candidate,
        selected_indices=list(indices), candidate_full12=list(candidate),
        desired_before_slew_full12=desired, output_full12=result,
        controller_delta_full12=[after-before for after, before in zip(result, candidate)],
        previous_final_wheel_rad_s=(list(previous_final_wheel_rad_s)
            if previous_final_wheel_rad_s is not None else None),
        raw_policy_and_log_probability_unchanged=True,
        current_TOP_has_forward_floor=False, added_forward_is_measured_traction=False)
