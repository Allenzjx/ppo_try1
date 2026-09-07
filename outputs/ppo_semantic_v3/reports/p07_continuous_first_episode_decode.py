"""Read only the first 78 real PPO observations; no model, CUDA, or simulation."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "runs/ppo_semantic_v3/train/20260906T1805562366558Z_gc34262abffc1_6e47bada5505418e811416a50bd871f5"
LEGS = ("FL", "FR", "RL", "RR")
rows = []
with (RUN / "residual_and_projection_audit.jsonl").open(encoding="utf-8") as stream:
    for index, line in enumerate(stream):
        rows.append(json.loads(line))
        if index == 77:
            break
assert [r["global_policy_decision"] for r in rows] == list(range(73089, 73167))
assert [i for i, r in enumerate(rows) if r["terminal"]] == [77]
rollout = torch.load(RUN / "rollouts/rollout_000537.pt", map_location="cpu", weights_only=False)
schema_path = "configs/ppo_semantic_v3/observation_schema.json"
for relative in (schema_path, "src/wlr50_clean/ppo/semantic_observation.py",
                 "src/wlr50_clean/infrastructure/command_batch.py"):
    assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == rollout["runtime_contract"]["files"][relative]
schema = json.loads((ROOT / schema_path).read_text())
obs = rollout["observations"]["policy"][:78, 0]
assert obs.shape == (78, 324) and bool(torch.isfinite(obs).all())
assert torch.equal(obs, rollout["observations"]["critic"][:78, 0])
groups = {}
clipped = {}
offset = 0
for spec in schema["feature_groups"]:
    part = obs[:, offset:offset + spec["size"]]
    groups[spec["name"]] = (part.double() * torch.tensor(spec["scale"], dtype=torch.float64)).tolist()
    clipped[spec["name"]] = int((part.abs() >= schema["clip"]).sum())
    offset += spec["size"]
assert offset == 324
start_tick = rows[0]["applied_audit"]["curriculum_start"]["physics_tick"]
assert start_tick == 5952

def rotate(q, vector, inverse=False):
    w, x, y, z = q
    length = math.sqrt(sum(v*v for v in q))
    w, x, y, z = (v/length for v in (w,x,y,z))
    if inverse:
        x,y,z = -x,-y,-z
    vx,vy,vz = vector
    tx,ty,tz = 2*(y*vz-z*vy),2*(z*vx-x*vz),2*(x*vy-y*vx)
    return [vx+w*tx+y*tz-z*ty, vy+w*ty+z*tx-x*tz, vz+w*tz+x*ty-y*tx]

pre_ticks = []
for index, row in enumerate(rows):
    a = row["applied_audit"]
    pre_tick = start_tick if index == 0 else rows[index-1]["applied_audit"]["physics_tick"]
    assert a["physics_tick"] - a["physics_ticks"] == pre_tick
    assert abs(groups["task_times"][index][0] - pre_tick/120) < 2e-5
    assert groups["stage_one_hot"][index].index(1.0) + 1 == int(a["phase_id"][1:])
    assert torch.equal(rollout["actions"][index,0], torch.tensor(row["raw_policy_action_full12"]))
    assert rollout["values"][index,0,0].item() == row["old_value"]
    assert bool(rollout["dones"][index,0,0]) == bool(row["terminal"])
    pre_ticks.append(pre_tick)

def end_sample(index):
    row=rows[index]; a=row["applied_audit"]; task=a["semantic_task"]; ev=task["physical_evaluator"]
    return {"global": row["global_policy_decision"], "phase": a["phase_id"], "end_phase": a["end_phase_id"],
            "tick": a["physics_tick"], "time_s": a["sim_time_s"], "terminal":row["terminal"],
            "nominal": a["nominal_action_full12"], "residual":a["projected_residual_full12"],
            "drive": a["actual_drive_target_full12"],
            "native_wheel": a["actuator_target_effect_audit"]["actual_native_targets"]["wheel_velocity_rad_s"],
            "legs": {leg:{k:ev["current_legs"][leg][k] for k in
                       ("front_distance_m","clearance_m","load_fraction","air","ground_contact",
                        "obstacle_pair_active","top_contact","support","consecutive_air_samples",
                        "consecutive_top_samples","initial_clearance") } for leg in LEGS},
            "RR_qualified": task["active_lift_history"]["RR"],
            "RR_crossed":task["front_edge_crossed_history"]["RR"],
            "RR_placed":task["placed_history"]["RR"],
            "capture":task["completion_values"], "body_speed":task["goal_features"]["body_linear_speed_m_s"]}

def pre_sample(index):
    names=("actual_joint_position_deg","actual_joint_velocity_deg_s","actual_wheel_velocity_rad_s",
           "raw_body_orientation_wxyz","projected_gravity_chassis","chassis_rpy_rad",
           "euler_roll_pitch_rate_rad_s","body_linear_velocity","body_angular_velocity",
           "com_position_relative_base","com_velocity_world","wheel_pair_normal_force",
           "wheel_pair_active","wheel_load_fraction","support_diagnostics",
           "mapper_tracking_compensation_deg","mapper_final_drive_servo_deg","previous_residual_full12")
    result={name:groups[name][index] for name in names}
    q=groups["raw_body_orientation_wxyz"][index]
    result.update(global_decision=rows[index]["global_policy_decision"],pre_tick=pre_ticks[index],
                  pre_time_s=pre_ticks[index]/120,
                  phase=rows[index]["applied_audit"]["phase_id"],
                  body_linear_velocity_world=rotate(q,groups["body_linear_velocity"][index]),
                  com_offset_body_axes=rotate(q,groups["com_position_relative_base"][index],True),
                  com_velocity_body_axes=rotate(q,groups["com_velocity_world"][index],True))
    return result

def segments(leg):
    result=[]
    for index,row in enumerate(rows):
        current=row["applied_audit"]["semantic_task"]["physical_evaluator"]["current_legs"][leg]
        state="AIR" if current["air"] else "TOP" if current["top_contact"] else "GROUND" if current["ground_contact"] else "OBSTACLE_OTHER"
        if result and result[-1]["state"]==state:
            result[-1].update(last_tick=row["applied_audit"]["physics_tick"],count=result[-1]["count"]+1)
        else:
            result.append(dict(state=state,first_tick=row["applied_audit"]["physics_tick"],
                               last_tick=row["applied_audit"]["physics_tick"],count=1))
    return result

selected=[0,1,2,3,10,35,52,53,65,77]
peak_index=max(range(78),key=lambda i:rows[i]["applied_audit"]["semantic_task"]["goal_features"]["RR_clearance_m"])
closest_index=max(range(78),key=lambda i:rows[i]["applied_audit"]["semantic_task"]["goal_features"]["RR_front_distance_m"])
selected=sorted(set(selected+[peak_index,closest_index]))
last=rows[-1]["applied_audit"]
events=[event for event in last["semantic_task"]["history"]["lift_attempt_events"] if event["leg"]=="RR"]
transitions=[{k:event[k] for k in ("from_stage","to_stage","physics_tick","sim_time_s","completion_values")}
             for event in last["semantic_task"]["transition_evidence"] if event["from_stage"] in ("P06","P07","P08")]
boundary=[]
for row in rows[:4]:
    a=row["applied_audit"]
    boundary.append({"global":row["global_policy_decision"],"phase":a["phase_id"],"end_phase":a["end_phase_id"],
                     "terminal":row["terminal"],"native_summary":a["actuator_target_effect_audit_summary"],
                     "ticks":a["actuator_target_effect_audit_ticks"],
                     "handoff":[{k:e[k] for k in ("from_state_id","to_state_id","handoff_hold_used",
                         "max_abs_servo_action_jump_deg","max_abs_wheel_action_jump_rad_s",
                         "max_abs_servo_residual_step_deg","max_abs_wheel_residual_step_rad_s")}
                                for e in a["phase_transition_action_jump"]]})
com_stats={}
for name in ("com_position_relative_base","com_velocity_world"):
    values=groups[name]
    com_stats[name]={"first":values[0],"last_pre":values[-1],
                     "minimum":[min(v[j] for v in values) for j in range(3)],
                     "maximum":[max(v[j] for v in values) for j in range(3)]}
result={"schema":"wlr50_clean.p07_first_episode_readonly_decode.v1","global_range":[73089,73166],
        "runtime_commit":rollout["runtime_contract"]["source_git_commit"],"selected_schema_and_source_hashes_verified":True,
        "observation_boundary":"predecision; not same-row final tick", "source_tick_range":[pre_ticks[0],pre_ticks[-1]],
        "phase_counts":dict(Counter(r["applied_audit"]["phase_id"] for r in rows)),
        "clipped_group_element_counts":{k:v for k,v in clipped.items() if v},
        "transition_evidence":transitions,"RR_events":events,
        "contact_boundary_segments":{leg:segments(leg) for leg in LEGS},
        "boundary_audits":boundary,"end_samples":[end_sample(i) for i in selected],
        "pre_observation_samples":[pre_sample(i) for i in (0,1,2,10,52,53,77)],
        "com_stats":com_stats,"RR_peak_end_sample":end_sample(peak_index),"RR_closest_end_sample":end_sample(closest_index),
        "terminal_event_ticks":last["semantic_task"]["history"]["event_ticks"],
        "terminal_reason":last["termination_reason"],"terminal_valid":last["semantic_task"]["physical_evaluator"]["valid"],
        "terminal_last_tick_count":last["physics_ticks"],"terminal_native_audit":last["actuator_target_effect_audit_summary"],
        "nominal_diagnostics_first":rows[0]["applied_audit"]["semantic_task"]["nominal_provider_diagnostics"],
        "all_no_state_writes":all(r["applied_audit"]["no_in_episode_state_writes_verified"] for r in rows),
        "all_native_ticks_verified":all(r["applied_audit"]["actuator_target_effect_audit_summary"]["all_ticks_verified"] for r in rows)}
print(json.dumps(result,allow_nan=False,separators=(",",":")))
