"""Read-only historical A/C continuity analysis; never imports Isaac or policy code.

Generated tables are diagnosis, not reclassification or a new physical gate.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "A": ROOT / "runs/ppo_semantic_v2/video_eval/baseline_A/20260905T0902190128552Z_g00b94fbb276a_e0ddaa746bc84bddb8e4f80506e09fac/source",
    "C": ROOT / "runs/ppo_semantic_v2/validation/20260905T0847270847096Z_gd19c655713bf_bb1e5013136b4b8787f63444760db2d8",
}
LEGS = ("FL", "FR", "RL", "RR")
NAMES = ("front_left", "front_right", "rear_left", "rear_right")
CHANNELS = tuple(f"{name}_{joint}" for name in NAMES for joint in ("hip", "knee")) + tuple(f"{name}_ankle" for name in NAMES)


def rows(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def norm(values):
    return math.sqrt(sum(x*x for x in values))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_cell(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def analyze(role, source):
    transition_rows = list(rows(source / "stage_transition_evidence.jsonl"))
    transitions = [{k: row[k] for k in ("physics_tick", "sim_time_s", "from_stage", "to_stage")}
                   for row in transition_rows if row["from_stage"] != row["to_stage"]]
    by_tick = {}
    verification = {"ticks": 0, "verified": 0, "nonzero_raw_ticks": 0, "actual_effect_ticks": 0,
                    "forbidden_write_ticks": 0}
    for item in rows(source / "native_tick_audit.jsonl"):
        audit = item["native_audit"]
        verification["ticks"] += 1
        verification["verified"] += bool(audit["verified"] and audit["actual_mapping_matches_dispatch"] and audit["setter_dispatch_targets_equal"])
        verification["nonzero_raw_ticks"] += any(audit["raw_policy_action_full12"])
        verification["actual_effect_ticks"] += audit["changed_target_channel_count"] > 0
        verification["forbidden_write_ticks"] += any(item[k] for k in item if k.startswith("in_episode_"))
        tick = item["episode_physics_tick"]
        if tick >= 40*120:
            by_tick[tick] = item
    events = transition_rows[-1]["physical_history"]["lift_attempt_events"]
    history = {kind: dict.fromkeys(LEGS, False) for kind in ("active_lift", "front_edge_crossed", "placed")}
    history_index = 0
    data = []
    zero = None
    counts = {"all_finite": 0, "raw_ticks": 0, "body_collision": 0, "exact_pairs_missing": 0, "com_missing": 0}
    for raw in rows(source / "physical_observations.jsonl"):
        tick = raw["physics_tick"]
        counts["raw_ticks"] += 1
        counts["all_finite"] += raw["all_finite"] is True
        counts["body_collision"] += raw["body_collision"]["detected"] is True
        while history_index < len(transition_rows) and transition_rows[history_index]["physics_tick"] <= tick:
            history = transition_rows[history_index]["physical_history"]
            history_index += 1
        if tick == 0:
            zero = {"base": raw["base"], "joints": raw["joints"], "commanded_full12": raw["commanded_full12"]}
        if tick not in by_tick:
            continue
        native = by_tick[tick]
        audit = native["native_audit"]
        base = raw["base"]
        com = raw["center_of_mass"]
        has_com = bool(com and com.get("valid") and com.get("position_w_m") is not None)
        counts["com_missing"] += not has_com
        forces = []
        for name in NAMES:
            contact = raw["contacts"][name+"_wheel"]
            forces.append(sum(pair["normal_force_n"] for pair in (contact["ground"], contact["obstacle"]) if pair["active"]))
        total_force = sum(forces)
        q = base["orientation_wxyz"]
        w,x,y,z = q
        roll = math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))
        pitch = math.asin(max(-1., min(1., 2*(w*y-z*x))))
        row = {"role": role, "tick": tick, "time_s": raw["simulation_time_s"], "phase_source": native["source_phase_id"],
               "body_roll_deg": math.degrees(roll), "body_pitch_deg": math.degrees(pitch),
               "body_speed_m_s": norm(base["linear_velocity_w_m_s"]), "body_angular_speed_rad_s": norm(base["angular_velocity_w_rad_s"]),
               "native_verified": audit["verified"], "native_changed_channels": audit["changed_target_channel_count"]}
        for i,axis in enumerate("xyz"):
            row[f"body_{axis}_m"] = base["position_w_m"][i]
            row[f"body_v{axis}_m_s"] = base["linear_velocity_w_m_s"][i]
            row[f"com_{axis}_m"] = com["position_w_m"][i] if has_com else None
            row[f"com_v{axis}_m_s"] = com["velocity_w_m_s"][i] if has_com else None
        fl = raw["wheels"]["front_left_ankle"]["center_w_m"]
        direction = [fl[i]-base["position_w_m"][i] for i in (0,1)]
        length = norm(direction)
        row["com_velocity_toward_current_FL_side_m_s"] = sum(com["velocity_w_m_s"][i]*direction[i]/length for i in (0,1)) if has_com and length else None
        row["com_relative_body_y_m"] = com["position_w_m"][1]-base["position_w_m"][1] if has_com else None
        for i,(leg,name) in enumerate(zip(LEGS,NAMES)):
            wheel = raw["wheels"][name+"_ankle"]
            body = raw["bodies"][name+"_wheel"]
            contact = raw["contacts"][name+"_wheel"]
            ground, obstacle = contact["ground"],contact["obstacle"]
            counts["exact_pairs_missing"] += not (ground["pair_verified"] and obstacle["pair_verified"])
            row.update({f"{leg}_front_m":wheel["center_w_m"][0]-raw["obstacle"]["front_x_m"],
                f"{leg}_clearance_m":wheel["bottom_w_m"][2]-raw["obstacle"]["top_z_m"],
                f"{leg}_ground":ground["active"],f"{leg}_obstacle":obstacle["active"],
                f"{leg}_AIR":not ground["active"] and not obstacle["active"],
                f"{leg}_force_N":forces[i],f"{leg}_load_fraction":forces[i]/total_force if total_force else None,
                f"{leg}_contact_class":contact["contact_class"],
                f"{leg}_contact_point_m":json_cell(obstacle.get("contact_point_w_m")),
                f"{leg}_active_lift_recorded":history["active_lift"][leg],
                f"{leg}_crossed_recorded":history["front_edge_crossed"][leg],
                f"{leg}_placed_recorded":history["placed"][leg]})
            for j,axis in enumerate("xyz"):
                row[f"{leg}_center_{axis}_m"] = wheel["center_w_m"][j]
                row[f"{leg}_relative_body_{axis}_m"] = wheel["center_w_m"][j]-base["position_w_m"][j]
                row[f"{leg}_v{axis}_m_s"] = body["linear_velocity_w_m_s"][j]
        for i,channel in enumerate(CHANNELS):
            row[f"nominal_{channel}"] = native["nominal_full12"][i]
            row[f"mapped_{channel}"] = audit["native_drive_target_full12"][i]
            row[f"raw_{channel}"] = audit["raw_policy_action_full12"][i]
            row[f"residual_{channel}"] = native["projected_residual_full12"][i]
            row[f"bias_{channel}"] = audit["combined_post_mapper_bias_full12"][i]
            row[f"drive_{channel}"] = raw["commanded_full12"][i]
            row[f"native_target_{channel}"] = (audit["actual_native_targets"]["servo_position_rad"]+audit["actual_native_targets"]["wheel_velocity_rad_s"])[i]
            if i < 8:
                row[f"q_{channel}_deg"] = raw["joints"][channel]["position_deg"]
                row[f"dq_{channel}_deg_s"] = raw["joints"][channel]["velocity_deg_s"]
            else:
                row[f"dq_{channel}_rad_s"] = raw["wheels"][channel]["velocity_rad_s"]
        data.append(row)
    manifest_name = "semantic_video_source_manifest.json" if role == "A" else "evaluation_manifest.json"
    manifest = json.loads((source / manifest_name).read_text())
    physical = manifest["physical_episode"] if role == "A" else manifest
    key_ticks = {int(event["physics_tick"]) for event in transitions if event["to_stage"] in ("P07","P08","P09","P10")}
    key_ticks |= {int(event["physics_tick"]) for event in events if event["leg"] == "RR"}
    p09 = [row for row in data if row["phase_source"] == "P09"]
    if p09:
        key_ticks.add(max(p09,key=lambda row:row["RR_clearance_m"])["tick"])
        key_ticks.add(p09[-1]["tick"])
        for i,row in enumerate(p09[1:],1):
            if row["nominal_rear_right_hip"] < p09[i-1]["nominal_rear_right_hip"]-1e-9:
                key_ticks.add(row["tick"]);break
    key_ticks.add(data[-1]["tick"])
    summary = {"role":role,"source":str(source),"manifest_sha256":sha(source/manifest_name),
        "duration_s":physical["physical_task_duration_s"],"task_success":physical["task_success"],
        "physical_reason":physical["physical_task_evaluation"]["termination_reason"],
        "physical_final":physical["physical_task_evaluation"],"native_audit":verification,
        "sensor_counts":counts,"transitions":transitions,"lift_events":events,
        "keyframes":[row for row in data if row["tick"] in key_ticks],
        "p09_stats":{},"initial":zero}
    for channel in CHANNELS:
        if not p09:continue
        summary["p09_stats"][channel]={
            "max_abs_raw":max(abs(row[f"raw_{channel}"]) for row in p09),
            "residual_min":min(row[f"residual_{channel}"] for row in p09),
            "residual_max":max(row[f"residual_{channel}"] for row in p09),
            "max_abs_drive_minus_native_plus_bias":max(abs(row[f"drive_{channel}"]-row[f"mapped_{channel}"]-row[f"bias_{channel}"]) for row in p09)}
    return data,summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output-dir",type=Path,default=ROOT/"outputs/ppo_semantic_v3/reports")
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    csv_path=args.output_dir/"continuity_diagnosis.csv"
    json_path=args.output_dir/"continuity_diagnosis_facts.json"
    if csv_path.exists() or json_path.exists():raise FileExistsError("preserve existing diagnosis; choose fresh output directory")
    all_rows=[];summaries={}
    for role,source in SOURCES.items():
        data,summary=analyze(role,source);all_rows.extend(data);summaries[role]=summary
    with csv_path.open("x",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
    json_path.write_text(json.dumps(summaries,ensure_ascii=False,indent=2),encoding="utf-8")
    for role,summary in summaries.items():
        print(json_cell({"role":role,"duration_s":summary["duration_s"],"native_audit":summary["native_audit"],
            "sensor_counts":summary["sensor_counts"],"transitions":summary["transitions"],"lift_events":summary["lift_events"]}))
        keys=("tick","time_s","phase_source","RR_front_m","RR_clearance_m","RR_AIR","RR_ground","RR_load_fraction",
            "FL_clearance_m","FL_AIR","FL_load_fraction","com_relative_body_y_m","com_velocity_toward_current_FL_side_m_s",
            "nominal_rear_right_hip","nominal_rear_right_knee","q_rear_right_hip_deg","q_rear_right_knee_deg")
        for row in summary["keyframes"]:print(json_cell({key:row[key] for key in keys}))
    print(json_cell({"csv":str(csv_path),"facts":str(json_path),"rows":len(all_rows)}))


if __name__=="__main__":main()
