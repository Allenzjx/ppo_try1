"""Read only A/B2 logs. Derived local differentials are not dynamic predictions."""
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
from wlr50_clean.ppo.semantic_height_diagnostics import transform_point

A = PROJECT.parent / "fsm_50mm_recording_shaped_clean_v1/runs/trial_043_20260902_clean_v010"
B = PROJECT / "runs/ppo_fsm_reference_p09_stable_v2/video_eval/prior_B/20260914T2310036230645Z_g4db24eabd129_332256dc4c39442d8d4f8b257a0df053/source"
URDF = PROJECT.parent / "sw2urdf_output/wlr_robot_isaac/urdf/wlr_robot_isaac.urdf"
LEGS = {"FL": "front_left", "FR": "front_right", "RL": "rear_left", "RR": "rear_right"}
INDICES = {"FL": 0, "FR": 2, "RL": 4, "RR": 6}


def selected(path, key, ticks):
    result = {}
    with path.open() as stream:
        for line, text in enumerate(stream, 1):
            raw = json.loads(text)
            tick = raw[key]
            if tick in ticks:
                result[tick] = (raw, line)
            if tick > max(ticks):
                break
    assert set(result) == set(ticks), (path, set(ticks) - result.keys())
    return result


def rotation(quat, vector):
    return np.array(transform_point(vector, [0., 0., 0.], quat))


def origins():
    root = ET.parse(URDF).getroot()
    result = {}
    for short, prefix in LEGS.items():
        joint = root.find(f"joint[@name='{prefix}_hip']")
        origin = joint.find("origin")
        r, p, y = map(float, origin.attrib["rpy"].split())
        rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
        ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
        rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
        result[short] = {"xyz": list(map(float, origin.attrib["xyz"].split())),
                         "parent": joint.find("parent").attrib["link"],
                         "axis_in_parent": rz @ ry @ rx @ np.array([0., 0., 1.])}
    return result


def main():
    mounts = origins()
    datasets = [
        ("A_FROZEN", A / "observation_120hz.jsonl", A / "full12_commands_120hz.jsonl",
         [6648, 6688, 6808, 6856, 6912, 6918, 7109, 7579, 7777]),
        ("B2_ZERO", B / "physical_observations.jsonl", B / "native_tick_audit.jsonl",
         [5160, 5200, 5320, 5360, 5408, 5421, 5425, 6273, 6274, 6280, 6374]),
    ]
    retained = json.loads((PROJECT / "outputs/ppo_timing_task_priority_v1/B_AFTER_FULL.records.json").read_text())
    semantic = {x["episode_physics_tick"]: x for x in retained if x["semantic_endpoint_observation_tick"] == x["episode_physics_tick"]}
    rows = []
    for label, physical_path, command_path, ticks in datasets:
        observations = selected(physical_path, "physics_tick", ticks)
        # A dispatch t-1 produced observation t; B native ledger already uses post-step t.
        commands = selected(command_path, "control_physics_tick" if label == "A_FROZEN" else "episode_physics_tick",
                            [t - 1 for t in ticks] if label == "A_FROZEN" else ticks)
        for tick in ticks:
            raw, line = observations[tick]
            command, command_line = commands[tick - 1 if label == "A_FROZEN" else tick]
            bounds = raw.get("body_bounds_w_m") or {}
            base = raw["base"]
            row = {"run_label": label, "physics_tick": tick, "simulation_time_s": tick / 120,
                "source_physical_path": str(physical_path), "source_physical_line": line,
                "command_pre_step_tick": tick - 1, "command_post_step_tick": tick,
                "source_command_path": str(command_path), "source_command_line": command_line,
                "base_origin_z_m": base["position_w_m"][2],
                "base_quaternion_wxyz": base["orientation_wxyz"],
                "body_collision_min_z_m": (bounds.get("base_link") or {}).get("minimum_m", [None]*3)[2],
                "body_collision_min_reason": "recorded actual-pose transformed mesh AABB" if bounds else "A raw stream did not retain body bounds; base origin is not a substitute",
                "body_collision_detected": raw["body_collision"]["detected"],
                "com_position_w_m": raw["center_of_mass"].get("position_w_m"),
                "legs": {}}
            for short, prefix in LEGS.items():
                i = INDICES[short]
                wheel, upper = raw["wheels"][prefix + "_ankle"], raw["bodies"][prefix + "_upper"]
                joint = raw["joints"][prefix + "_hip"]
                origin = mounts[short]
                derived_mount = transform_point(origin["xyz"], base["position_w_m"], base["orientation_wxyz"])
                mount_error = max(abs(a-b) for a,b in zip(derived_mount, upper["position_w_m"]))
                axis = rotation(base["orientation_wxyz"], origin["axis_in_parent"])
                axis_error = float(np.max(np.abs(axis - rotation(upper["orientation_wxyz"], [0., 0., 1.]))))
                sign = 1. if short in ("FL", "FR") else -1.
                derivative = np.cross(axis, np.array(wheel["center_w_m"]) - np.array(derived_mount)) * sign * math.pi / 180
                verified = mount_error < 2e-6 and axis_error < 2e-6
                contact = raw["contacts"][prefix + "_wheel"]
                record = {"hip_N_deg": command["nominal_full12"][i], "hip_final_deg": joint["command_deg"],
                    "hip_actual_deg": joint["position_deg"], "hip_velocity_deg_s": joint["velocity_deg_s"],
                    "hip_hard_margin_deg": min(joint["position_deg"] + 135, 135 - joint["position_deg"]),
                    "hip_upper_link_origin_w_m": upper["position_w_m"],
                    "hip_mount_URDF_transformed_w_m": derived_mount,
                    "mount_vs_recorded_upper_origin_max_error_m": mount_error,
                    "axis_vs_recorded_upper_axis_max_error": axis_error,
                    "URDF_mount_and_axis_match_recorded_body_pose": verified,
                    "USD_joint_mount_w_m": None, "USD_mount_reason": "not retained in old run; URDF-derived point and recorded upper-link origin shown separately, live USD diagnostic added for next run",
                    "wheel_center_w_m": wheel["center_w_m"], "wheel_bottom_z_m": wheel["bottom_w_m"][2],
                    "wheel_geometry_source": wheel["geometry_source"],
                    "contact_class": contact["contact_class"],
                    "pairs": {key: {k: contact[key].get(k) for k in ("active", "pair_verified", "force_w_n", "normal_force_n")}
                              for key in ("ground", "obstacle")},
                    "fixed_body_d_wheel_center_xyz_m_per_canonical_hip_deg": derivative.tolist() if verified else None,
                    "single_fixed_foot_translation_only_d_body_z_m_per_canonical_hip_deg": float(-derivative[2]) if verified else None}
                if short == "RR":
                    knee = raw["joints"]["rear_right_knee"]
                    record.update(knee_N_deg=command["nominal_full12"][7], knee_final_deg=knee["command_deg"],
                                  knee_actual_deg=knee["position_deg"], knee_lower_margin_deg=knee["position_deg"] + 60,
                                  front_distance_m=wheel["center_w_m"][0] - raw["obstacle"]["front_x_m"],
                                  top_gap_m=wheel["bottom_w_m"][2] - raw["obstacle"]["top_z_m"])
                sem = semantic.get(tick) if label == "B2_ZERO" else None
                record["semantic_load_at_exact_tick"] = None if sem is None else {
                    "observation_tick": tick, **{k: None if sem[k] is None else sem[k][i//2] for k in
                        ("legs_bearing_force_n", "legs_bearing_verified", "legs_load_fraction", "legs_load_fraction_valid", "legs_support", "legs_contact_surface")}}
                row["legs"][short] = record
            rows.append(row)
    output = {"schema": "wlr50_clean.existing_A_B2_height_window.v1", "rows": rows,
        "derived_geometry_source": str(URDF), "units": "canonical hip degrees relative to standing; physical native FL sign+1, RL sign-1",
        "limits": ["No new physical simulation or image measurements.", "Command t-1 produced physical sample t; listed nominal is previous dispatch, not a newly activated command at sample t.",
                   "Joint-axis differential assumes fixed body and other joints, and is wheel-center not collider-lowest derivative.",
                   "Negated vertical differential is ONLY a single fixed-foot pure-body-translation toy constraint, not a floating-base multi-contact prediction.",
                   "AIR legs cannot bear load; unknown semantic load is null, exact-pair forces are not invented support fractions.",
                   "A wheel bottoms use its retained frozen cached extent; B2 uses actual-pose collider geometry. They are not silently equated."]}
    target = Path(__file__).with_name("existing_A_B2_height_window.json")
    with target.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, allow_nan=False)
    print(json.dumps({"output": str(target), "rows": len(rows), "brief": [{"run": x["run_label"], "tick": x["physics_tick"],
        "base_z": x["base_origin_z_m"], "body_min": x["body_collision_min_z_m"],
        "RR_mount_URDF_z": x["legs"]["RR"]["hip_mount_URDF_transformed_w_m"][2],
        "FL_actual": x["legs"]["FL"]["hip_actual_deg"], "RL_actual": x["legs"]["RL"]["hip_actual_deg"],
        "FL_dz_per_deg": x["legs"]["FL"]["fixed_body_d_wheel_center_xyz_m_per_canonical_hip_deg"],
        "RL_dz_per_deg": x["legs"]["RL"]["fixed_body_d_wheel_center_xyz_m_per_canonical_hip_deg"],
        "FL_contact": x["legs"]["FL"]["contact_class"], "RL_contact": x["legs"]["RL"]["contact_class"]} for x in rows]}, indent=2))


if __name__ == "__main__":
    main()
