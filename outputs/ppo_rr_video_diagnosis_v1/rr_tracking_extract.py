"""Bounded completed-video evidence extraction; stdlib only, JSON only, no writes to runs.

Run after both videos complete:
  python rr_tracking_extract.py --zero-run RUN --ppo-run RUN --output-prefix NEW_PATH
Only first <=2400 native episode ticks in P01-P03; no old-A scan or 15-Hz forward fill.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

LEGS = {"FL": "front_left", "FR": "front_right", "RL": "rear_left", "RR": "rear_right"}
JOINTS = tuple(f"{name}_{joint}" for name in LEGS.values() for joint in ("hip", "knee"))
WHEELS = tuple(f"{name}_ankle" for name in LEGS.values())
SERVO_SIGNS = (1., 1., 1., 1., -1., -1., -1., -1.)
WHEEL_SIGNS = (-1., 1., -1., 1.)
PHASES = {"P01", "P02", "P03"}
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get(value, *keys):
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key)
        elif isinstance(key, int) and isinstance(value, (list, tuple)) and 0 <= key < len(value):
            value = value[key]
        else:
            return None
    return value


def number(value):
    return value if type(value) in (float, int) and math.isfinite(value) else None


def difference(a, b):
    a, b = number(a), number(b)
    return a-b if a is not None and b is not None else None


def sequence(value, length):
    return isinstance(value, list) and len(value) == length and all(number(v) is not None for v in value)


def euler(q):
    if not sequence(q, 4):
        return [None]*3
    norm = math.sqrt(sum(v*v for v in q))
    if norm <= 1e-12:
        return [None]*3
    w, x, y, z = (v/norm for v in q)
    return [math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y)),
            math.asin(max(-1., min(1., 2*(w*y-z*x)))),
            math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))]


def calibrated_euler(q, fixed):
    if not sequence(q, 4) or not sequence(fixed, 4):
        return None
    qnorm, fnorm = math.sqrt(sum(v*v for v in q)), math.sqrt(sum(v*v for v in fixed))
    if min(qnorm, fnorm) <= 1e-12:
        return None
    w, x, y, z = (v/qnorm for v in q)
    a, b, c, d = (v/fnorm for v in fixed)
    # Exact semantic_observation convention: q_world_body * fixed_chassis_to_body.
    return euler([w*a-x*b-y*c-z*d, w*b+x*a+y*d-z*c,
                  w*c-x*d+y*a+z*b, w*d+x*c-y*b+z*a])


def load_calibration(manifest):
    record = get(manifest, "runtime_contract", "selected_configuration", "observation_schema.json")
    if not isinstance(record, dict) or not record.get("path") or not record.get("sha256"):
        return None, {"status": "missing_recorded_calibration_unknown_not_identity"}
    path = (PROJECT_ROOT/record["path"]).resolve()
    if not path.is_relative_to(PROJECT_ROOT):
        raise ValueError("Recorded calibration is outside this project")
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != record["sha256"]:
        raise ValueError("Recorded observation calibration hash differs; refusing invented attitude")
    fixed = json.loads(payload.decode("utf-8-sig")).get("fixed_chassis_to_body_wxyz")
    if not sequence(fixed, 4) or sum(v*v for v in fixed) <= 1e-24:
        raise ValueError("Invalid recorded fixed chassis quaternion")
    return fixed, {"status": "recorded_configuration_sha256_verified", "path": str(path),
                   "sha256": digest, "fixed_chassis_to_body_wxyz": fixed,
                   "formula": "normalized(q_world_body * normalized(fixed_chassis_to_body)); semantic_observation.py convention"}


def lines(path):
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                yield line_number, json.loads(line)


def completed_source(run):
    run = Path(run).resolve()
    if run.name == "source":
        run = run.parent
    with (run/"run_manifest.json").open(encoding="utf-8-sig") as stream:
        manifest = json.load(stream)
    if not manifest.get("completed_at_utc") or manifest.get("lifecycle") in (None, "STARTED", "RUNNING"):
        raise ValueError(f"Run is not finalized: {run}")
    if manifest.get("optimizer_updates") not in (0, None):
        raise ValueError("This helper accepts evaluation runs, not training")
    return run/"source", manifest


def physical_schema(row):
    if type(row.get("physics_tick")) is not int or row["physics_tick"] < 0:
        raise ValueError("Invalid physical episode tick")
    tick, t = row["physics_tick"], number(row.get("simulation_time_s"))
    if t is None or abs(t-tick/120.) > 1e-8:
        raise ValueError("Physical episode clock is not 120 Hz")
    if not all(isinstance(row.get(k), dict) for k in ("joints", "wheels", "base")):
        raise ValueError("Physical joint/wheel/body dictionaries absent")


def native_schema(row):
    audit = row.get("native_audit")
    if type(row.get("episode_physics_tick")) is not int or not isinstance(audit, dict):
        raise ValueError("Native episode tick/audit absent")
    if audit.get("canonical_order") != list(JOINTS+WHEELS):
        raise ValueError("Unknown Full12 order; refusing guessed index mapping")
    if row.get("source_phase_id") != audit.get("source_phase_id"):
        raise ValueError("Native source phase disagreement")


def flatten(physical, native, label, source_lines, transition=None, calibration=None):
    physical_schema(physical)
    tick = physical["physics_tick"]
    if native is not None:
        native_schema(native)
        if native["episode_physics_tick"] != tick:
            raise ValueError("Refusing pre/post-step or native-clock misjoin")
    audit = get(native, "native_audit") or {}
    head = get(audit, "policy_headroom_evidence") or {}
    tracking = get(audit, "tracking_reference_evidence") or {}
    mapped = get(audit, "native_drive_target_full12")
    final = physical.get("commanded_full12")
    row = {"run_label": label, "episode_physics_tick": tick,
           "simulation_time_s": physical["simulation_time_s"],
           "source_phase_id": get(native, "source_phase_id"),
           "substage": None, "substage_source": "not_in_120Hz_stream_no_15Hz_fill",
           "native_internal_tick": get(audit, "physics_tick"),
           "physical_source_line": source_lines[0], "native_source_line": source_lines[1],
           "native_verified": get(audit, "verified"),
           "dispatch_targets_equal": get(audit, "setter_dispatch_targets_equal"),
           "mapping_matches_dispatch": get(audit, "actual_mapping_matches_dispatch"),
           "raw_residual_full12": get(audit, "raw_policy_action_full12"),
           "projected_residual_full12": get(native, "projected_residual_full12"),
           "phase_mask_full12": get(audit, "phase_mask_full12"),
           "headroom_clipped_servo_indices": get(head, "clipped_servo_indices"),
           "projection_clip_flags": None, "projection_rate_limit_flags": None,
           "projection_flag_status": "not_in_native_stream_unknown_not_false",
           "counterfactual_scope": get(audit, "counterfactual_scope"),
           "base_origin_height_world_m": get(physical, "base", "position_w_m", 2),
           "base_chassis_rpy_derived_rad": euler(get(physical, "base", "orientation_wxyz")),
           "calibrated_body_rpy_rad": calibrated_euler(get(physical, "base", "orientation_wxyz"), calibration),
           "body_attitude_status": ("derived_same_tick_using_SHA256_verified_recorded_fixed_calibration" if calibration is not None
                                    else "raw_base_chassis_Euler_only; per_run_calibration_missing; not_reward_attitude"),
           "body_angular_velocity_world_rad_s": get(physical, "base", "angular_velocity_w_rad_s"),
           "imu_angular_velocity_base_frame_rad_s": get(physical, "imu", "angular_velocity_b_rad_s"),
           "body_linear_velocity_world_m_s": get(physical, "base", "linear_velocity_w_m_s"),
           "com_position_world_m": get(physical, "center_of_mass", "position_w_m"),
           "com_velocity_world_m_s": get(physical, "center_of_mass", "velocity_w_m_s"),
           "com_mass_kg": get(physical, "center_of_mass", "total_mass_kg"),
           "com_valid": get(physical, "center_of_mass", "valid"),
           "com_source": get(physical, "center_of_mass", "source"),
           "com_included_bodies": get(physical, "center_of_mass", "included_bodies"),
           "body_collision": get(physical, "body_collision"),
           "data_quality": physical.get("data_quality"),
           "task_transition_at_exact_tick": transition,
           "joint_computed_torque_nm": None, "joint_applied_torque_nm": None,
           "joint_solver_reaction": None,
           "torque_status": "not_recorded_unknown_not_zero"}
    for i, name in enumerate(JOINTS):
        leg = tuple(LEGS)[i//2]
        short = f"{leg}_{name.rsplit('_', 1)[1]}"
        actual, velocity = get(physical, "joints", name, "position_deg"), get(physical, "joints", name, "velocity_deg_s")
        target = get(physical, "joints", name, "command_deg")
        if number(target) is not None and number(get(final, i)) is not None and abs(target-final[i]) > 1e-9:
            raise ValueError(f"Physical command disagreement at {tick}/{name}")
        candidate = get(head, "candidate_native_target_before_final_slew_full12", i)
        final_modified = difference(target, candidate)
        native_delta = difference(get(audit, "actual_native_targets", "servo_position_rad", i),
                                  get(audit, "counterfactual_native_targets", "servo_position_rad", i))
        channels = tracking.get("channels")
        channel = get(channels, name) if isinstance(channels, dict) else None
        if isinstance(channels, list):
            matches = [item for item in channels if isinstance(item, dict) and item.get("servo") == name]
            channel = matches[0] if len(matches) == 1 else None
        row.update({f"{short}_nominal_request_deg": get(native, "nominal_full12", i),
                    f"{short}_mapped_N_deg": get(mapped, i), f"{short}_final_target_deg": target,
                    f"{short}_actual_deg": actual, f"{short}_velocity_deg_s": velocity,
                    f"{short}_e_tracking_actual_minus_final_deg": difference(actual, target),
                    f"{short}_e_residual_final_minus_mapped_N_deg": difference(target, get(mapped, i)),
                    f"{short}_controller_bias_deg": get(audit, "controller_drive_bias_full12", i),
                    f"{short}_combined_bias_deg": get(audit, "combined_post_mapper_bias_full12", i),
                    f"{short}_effective_policy_bias_deg": get(head, "effective_policy_residual_full12", i),
                    f"{short}_geometry_adjustment_deg": get(audit, "nominal_geometry_adjustment_full12", i),
                    f"{short}_direct_same_tick_policy_delta_deg": None if native_delta is None else math.degrees(native_delta)/SERVO_SIGNS[i],
                    f"{short}_final_minus_preslew_candidate_deg": final_modified,
                    f"{short}_final_slew_or_clamp_modified_derived": None if final_modified is None else abs(final_modified) > 1e-8,
                    f"{short}_tracking_scheduled": get(channel, "scheduled"),
                    f"{short}_tracking_reference_used": get(channel, "reference_used")})
    for i, (leg, prefix) in enumerate(LEGS.items()):
        wheel = get(physical, "wheels", WHEELS[i]) or {}
        body = wheel.get("body_name")
        contact = get(physical, "contacts", body) or {}
        row.update({f"{leg}_wheel_nominal_rad_s": get(native, "nominal_full12", i+8),
                    f"{leg}_wheel_final_rad_s": wheel.get("command_rad_s"),
                    f"{leg}_wheel_actual_rad_s": wheel.get("velocity_rad_s"),
                    f"{leg}_contact_class": contact.get("contact_class"),
                    f"{leg}_normalized_bearing_load_fraction": None,
                    f"{leg}_normalized_bearing_load_valid": None,
                    f"{leg}_contact_body_name": body})
        for surface in ("ground", "obstacle"):
            for field in ("active", "pair_verified", "normal_force_n", "force_w_n", "source"):
                row[f"{leg}_{surface}_{field}"] = get(contact, surface, field)
        row[f"{leg}_front_distance_derived_m"] = difference(get(wheel, "center_w_m", 0), get(physical, "obstacle", "front_x_m"))
        row[f"{leg}_top_clearance_derived_m"] = difference(get(wheel, "bottom_w_m", 2), get(physical, "obstacle", "top_z_m"))
        row[f"{leg}_geometry_verified"] = wheel.get("geometry_verified")
    return row


def initial_state(physical, first_native):
    tracking = get(first_native, "native_audit", "tracking_reference_evidence") or {}
    return {"episode_tick": physical["physics_tick"], "simulation_time_s": physical["simulation_time_s"],
            "joint_position_canonical_deg": [get(physical, "joints", n, "position_deg") for n in JOINTS],
            "joint_velocity_canonical_deg_s": [get(physical, "joints", n, "velocity_deg_s") for n in JOINTS],
            "joint_position_native_pre_first_step_rad": tracking.get("actual_measured_physical_rad") if get(first_native, "episode_physics_tick") == 1 else None,
            "standing_pose_deg": tracking.get("standing_pose_deg"),
            "wheel_velocity_rad_s": [get(physical, "wheels", n, "velocity_rad_s") for n in WHEELS],
            "root_position_world_m": get(physical, "base", "position_w_m"),
            "root_quaternion_wxyz": get(physical, "base", "orientation_wxyz"),
            "root_linear_velocity_world_m_s": get(physical, "base", "linear_velocity_w_m_s"),
            "root_angular_velocity_world_rad_s": get(physical, "base", "angular_velocity_w_rad_s"),
            "com_position_world_m": get(physical, "center_of_mass", "position_w_m"),
            "com_velocity_world_m_s": get(physical, "center_of_mass", "velocity_w_m_s"),
            "com_valid": get(physical, "center_of_mass", "valid"),
            "contact_pairs": {leg: get(physical, "contacts", get(physical, "wheels", WHEELS[i], "body_name")) for i, leg in enumerate(LEGS)}}


def extract(run, label, maximum_tick):
    source, manifest = completed_source(run)
    calibration, calibration_proof = load_calibration(manifest)
    native, native_lines = {}, {}
    stop = "native_stream_end"
    for line_no, value in lines(source/"native_tick_audit.jsonl"):
        native_schema(value)
        tick = value["episode_physics_tick"]
        if tick > maximum_tick or value["source_phase_id"] not in PHASES:
            stop = "2400_tick_or_requested_bound" if tick > maximum_tick else "left_P01_P03"
            break
        if tick in native or tick <= 0 or (native and tick != max(native)+1):
            raise ValueError("Noncontiguous/duplicate native episode ticks")
        native[tick], native_lines[tick] = value, line_no
    if not native:
        raise ValueError("No completed early native tick records")
    end = max(native)
    transitions = {}
    transition_path = source/"stage_transition_evidence.jsonl"
    if transition_path.exists():
        for _, value in lines(transition_path):
            tick = value.get("physics_tick")
            if type(tick) is not int:
                raise ValueError("Transition lacks exact physical tick")
            if tick > end:
                break
            transitions.setdefault(tick, []).append(value)
    records, seen, initial = [], set(), None
    for line_no, value in lines(source/"physical_observations.jsonl"):
        physical_schema(value)
        tick = value["physics_tick"]
        if tick > end:
            break
        if tick in seen:
            raise ValueError("Duplicate physical tick")
        seen.add(tick)
        if tick == 0:
            initial = initial_state(value, native.get(1))
        records.append(flatten(value, native.get(tick), label, (line_no, native_lines.get(tick)), transitions.get(tick), calibration))
    if initial is None or set(native)-seen:
        raise ValueError("Initial physical state or exact native/physical join missing")
    if seen != set(range(end+1)):
        raise ValueError("Physical prefix is not contiguous from natural tick0")
    peaks = {}
    for short in ("RR_hip", "RR_knee", "RL_hip", "FR_knee"):
        key = f"{short}_e_tracking_actual_minus_final_deg"
        valid = [r for r in records if number(r.get(key)) is not None]
        peak = max(valid, key=lambda r: abs(r[key])) if valid else None
        peaks[short] = None if peak is None else {"tick": peak["episode_physics_tick"], "signed_error_deg": peak[key]}
    summary = {"source": str(source), "lifecycle": manifest.get("lifecycle"),
               "completed_at_utc": manifest.get("completed_at_utc"),
               "optimizer_updates": manifest.get("optimizer_updates"),
               "native_ticks": len(native), "record_count_including_initial": len(records),
               "end_tick": end, "stop_reason": stop,
               "phase_tick_counts": dict(Counter(r["source_phase_id"] for r in records if r["source_phase_id"])),
               "tracking_error_peaks_in_bounded_prefix": peaks,
               "exact_join_complete": True, "initial_state": initial,
               "attitude_calibration": calibration_proof}
    return summary, records


def compare_initial(a, b):
    comparison = {}
    for key in a:
        x, y = a[key], b.get(key)
        if isinstance(x, list) and isinstance(y, list) and len(x) == len(y) and all(number(v) is not None for v in x+y):
            delta = [u-v for u, v in zip(y, x)]
            comparison[key] = {"ppo_minus_zero": delta, "maximum_absolute_difference": max(map(abs, delta), default=0.), "exact_equal": x == y}
    comparison["com_valid"] = {"zero": a.get("com_valid"), "ppo": b.get("com_valid")}
    comparison["contact_pairs_exact_equal"] = a.get("contact_pairs") == b.get("contact_pairs")
    comparison["interpretation"] = "Measured initial components; no seed-only pairing claim, no contact-history restoration claim. Quaternion component differences are not an angular distance."
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zero-run")
    parser.add_argument("--ppo-run")
    parser.add_argument("--output-prefix")
    parser.add_argument("--maximum-tick", type=int, default=2400)
    parser.add_argument("--validate-schema-only", metavar="COMPLETED_RUN")
    args = parser.parse_args()
    if args.validate_schema_only:
        source, manifest = completed_source(args.validate_schema_only)
        calibration, calibration_proof = load_calibration(manifest)
        _, p = next(lines(source/"physical_observations.jsonl"))
        _, n = next(lines(source/"native_tick_audit.jsonl"))
        physical_schema(p)
        native_schema(n)
        r = flatten(p, None, "schema_validation_only", (1, None), calibration=calibration)
        assert r["episode_physics_tick"] == 0 and n["episode_physics_tick"] == 1
        assert r["RR_knee_e_tracking_actual_minus_final_deg"] == difference(get(p, "joints", "rear_right_knee", "position_deg"), get(p, "joints", "rear_right_knee", "command_deg"))
        try:
            flatten(p, n, "invalid_join", (1, 1))
        except ValueError:
            pass
        else:
            raise AssertionError("Misaligned pre/post-step join accepted")
        # In-memory clock-only fixture exercises field extraction, never evidence.
        fixture = dict(p, physics_tick=1, simulation_time_s=1/120.)
        expanded = flatten(fixture, n, "synthetic_schema_fixture_not_evidence", (None, None))
        assert expanded["RR_knee_mapped_N_deg"] == get(n, "native_audit", "native_drive_target_full12", 7)
        assert expanded["joint_applied_torque_nm"] is None
        initial = initial_state(p, n)
        compared = compare_initial(initial, initial)
        assert compared["joint_position_canonical_deg"]["maximum_absolute_difference"] == 0.
        assert expanded["counterfactual_scope"] == get(n, "native_audit", "counterfactual_scope")
        assert calibrated_euler([1., 0., 0., 0.], [1., 0., 0., 0.]) == [0., 0., 0.]
        assert calibrated_euler([1., 0., 0., 0.], None) is None
        print(json.dumps({"schema_validated": True, "read_scope": "one initial physical row, one native row, recorded small calibration config only", "records_written": 0, "pre_post_misjoin_rejected": True, "synthetic_in_memory_field_mapping_test": True, "calibration_status": calibration_proof["status"]}))
        return
    if not all((args.zero_run, args.ppo_run, args.output_prefix)) or not 1 <= args.maximum_tick <= 2400:
        parser.error("Provide both completed runs, fresh output prefix, and maximum tick1..2400")
    prefix = Path(args.output_prefix).resolve()
    summary_path, records_path = Path(str(prefix)+".summary.json"), Path(str(prefix)+".records.json")
    if summary_path.exists() or records_path.exists():
        raise FileExistsError("Outputs already exist; never overwriting prior evidence")
    if not prefix.parent.is_dir():
        raise FileNotFoundError("Output parent must already exist")
    zero, zero_records = extract(args.zero_run, "B0_zero", args.maximum_tick)
    ppo, ppo_records = extract(args.ppo_run, "C0_ppo", args.maximum_tick)
    summary = {"schema": "rr_tracking_bounded_json.v1", "created_utc": datetime.now(timezone.utc).isoformat(),
               "maximum_tick": args.maximum_tick, "zero": zero, "ppo": ppo,
               "initial_comparison": compare_initial(zero["initial_state"], ppo["initial_state"]),
               "record_file": str(records_path),
               "evidence_limits": ["No cross-run counterfactual causality", "No old A scan", "No 15Hz forward fill", "No missing loads or torques replaced by zero", "Euler/geometry/deltas explicitly derived", "Final-minus-mapped-N includes non-policy bias and slew; direct counterfactual is same-prestate current-policy only"]}
    with records_path.open("x", encoding="utf-8") as stream:
        json.dump(zero_records+ppo_records, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    with summary_path.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, allow_nan=False, indent=2)
    print(json.dumps({"summary": str(summary_path), "records": str(records_path),
                      "zero_native_ticks": zero["native_ticks"], "ppo_native_ticks": ppo["native_ticks"]}))


if __name__ == "__main__":
    main()
