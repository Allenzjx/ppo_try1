"""Fixed completed episodes only; CPU readback and arithmetic, no model/optimizer."""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
os.environ["OMP_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/"runs/ppo_semantic_v3/train/20260906T1805562366558Z_gc34262abffc1_6e47bada5505418e811416a50bd871f5"
LEGS=("FL","FR","RL","RR")
episodes=[]
with (RUN/"completed_episodes.jsonl").open(encoding="utf-8") as f:
    for i,line in enumerate(f):
        episodes.append(json.loads(line))
        if i==2: break
assert [(e["episode_index"],e["policy_decisions"]) for e in episodes]==[(0,78),(1,452),(2,452)]
total=sum(e["policy_decisions"] for e in episodes)
rows=[]
with (RUN/"residual_and_projection_audit.jsonl").open(encoding="utf-8") as f:
    for i,line in enumerate(f):
        row=json.loads(line); a=row["applied_audit"]; task=a["semantic_task"]; ev=task["physical_evaluator"]
        rows.append(dict(g=row["global_policy_decision"],raw=row["raw_policy_action_full12"],old_value=row["old_value"],
            terminal=row["terminal"],phase=a["phase_id"],end_phase=a["end_phase_id"],tick=a["physics_tick"],
            ticks=a["physics_ticks"],time=a["sim_time_s"],phi=task["task_progress_potential"],
            reward=a["reward_breakdown"],nom=a["nominal_action_full12"],res=a["projected_residual_full12"],
            drive=a["actual_drive_target_full12"],legs=ev["current_legs"],
            body_speed=task["goal_features"]["body_linear_speed_m_s"],omega_norm=task["goal_features"]["body_angular_speed_rad_s"],
            wheel_speed=ev["measured_wheel_velocity_rad_s"],valid=ev["valid"],physical_failure=ev["termination_reason"],
            task_reason=a["termination_reason"],completion=task["completion_values"],
            Q=task["active_lift_history"],C=task["front_edge_crossed_history"],P=task["placed_history"],
            native=a["actuator_target_effect_audit_summary"],no_writes=a["no_in_episode_state_writes_verified"],
            handoff=a["phase_transition_action_jump"],history=task["history"] if row["terminal"] else None,
            prefix=a["curriculum_start"] if not rows or rows[-1]["terminal"] else None))
        if i==total-1: break
assert [r["g"] for r in rows]==list(range(73089,74071))
assert [r["g"] for r in rows if r["terminal"]]==[73166,73618,74070]
schema=json.loads((ROOT/"configs/ppo_semantic_v3/observation_schema.json").read_text())
groups={}; off=0
for s in schema["feature_groups"]:
    groups[s["name"]]=(off,off+s["size"],s["scale"]);off+=s["size"]
assert off==324
observations={}; clipped=Counter(); contract=None
for update in range(537,545):
    ro=torch.load(RUN/f"rollouts/rollout_{update:06d}.pt",map_location="cpu",weights_only=False)
    if contract is None:
        contract=ro["runtime_contract"]
        for p in ("configs/ppo_semantic_v3/observation_schema.json","src/wlr50_clean/ppo/semantic_observation.py"):
            assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==contract["files"][p]
    assert ro["runtime_contract"]==contract
    for local in range(128):
        index=(update-537)*128+local
        if index>=total: break
        if index<78: continue
        row=rows[index]; obs=ro["observations"]["policy"][local,0]
        assert obs.shape==(324,) and bool(torch.isfinite(obs).all())
        assert torch.equal(obs,ro["observations"]["critic"][local,0])
        assert torch.equal(torch.tensor(row["raw"]),ro["actions"][local,0])
        assert row["old_value"]==ro["values"][local,0,0].item()
        assert row["terminal"]==bool(ro["dones"][local,0,0])
        decoded={}
        for name,(a,b,scale) in groups.items():
            decoded[name]=(obs[a:b].double()*torch.tensor(scale,dtype=torch.float64)).tolist()
            clipped[name]+=int((obs[a:b].abs()>=schema["clip"]).sum())
        pre=row["tick"]-row["ticks"]
        assert abs(decoded["task_times"][0]-pre/120)<2e-5
        assert decoded["stage_one_hot"].index(1.0)+1==int(row["phase"][1:])
        observations[index]=dict(tick=pre,groups=decoded)
    del ro

def mean(values):return sum(values)/len(values)
def stats(values):return dict(first=values[0],last=values[-1],minimum=min(values),maximum=max(values),mean=mean(values))
def vector_stats(values):return [stats([v[i] for v in values]) for i in range(len(values[0]))]
def point(r):
    return {k:r[k] for k in ("g","tick","time","phase","end_phase","terminal","phi","completion","nom","res","drive","wheel_speed","body_speed","omega_norm","Q","C","P","legs")}
def contact(r,leg):
    x=r["legs"][leg]
    return "AIR" if x["air"] else "TOP" if x["top_contact"] else "GROUND" if x["ground_contact"] else "OBSTACLE_OTHER"

results=[]; start=0
for e in episodes:
    seq=rows[start:start+e["policy_decisions"]]; stop=start+len(seq); n=len(seq)
    family={name:sum(r["reward"]["families"][name] for r in seq) for name in seq[0]["reward"]["families"]}
    raw_total=sum(r["reward"]["total"] for r in seq)
    assert abs(raw_total-sum(family.values()))<1e-8
    formula_error=max(abs(r["reward"]["potential_shaping"]-5*(.995*r["reward"]["potential_after"]-r["reward"]["potential_before"])) for r in seq)
    continuity_error=max(abs(seq[i]["reward"]["potential_before"]-seq[i-1]["reward"]["potential_after"]) for i in range(1,n))
    phi_binding_error=max(abs(r["reward"]["potential_after"]-(0 if r["terminal"] else r["phi"])) for r in seq)
    assert formula_error<1e-12 and continuity_error<1e-12 and phi_binding_error<1e-12
    discounted_pbrs=sum(.995**i*r["reward"]["potential_shaping"] for i,r in enumerate(seq))
    telescoping=-5*seq[0]["reward"]["potential_before"]
    reward={"families":family,"total":raw_total,"shaping_sum":sum(r["reward"]["potential_shaping"] for r in seq),
        "terminal_events":sum(r["reward"]["terminal_event"] for r in seq),
        "time_cost":-.02*sum(r["reward"]["elapsed_physics_s"] for r in seq),
        "phi_initial":seq[0]["reward"]["potential_before"],"phi_last_physical":seq[-1]["phi"],
        "phi_before_terminal":seq[-1]["reward"]["potential_before"],"terminal_shaping":seq[-1]["reward"]["potential_shaping"],
        "shaping_positive_count":sum(r["reward"]["potential_shaping"]>0 for r in seq),
        "shaping_positive_sum":sum(max(0,r["reward"]["potential_shaping"]) for r in seq),
        "shaping_negative_sum":sum(min(0,r["reward"]["potential_shaping"]) for r in seq),
        "positive_shaping_but_negative_total":sum(r["reward"]["potential_shaping"]>0 and r["reward"]["total"]<0 for r in seq),
        "cost_components":{k:sum(r["reward"]["cost_components"].get(k,0) for r in seq) for k in seq[0]["reward"]["cost_components"]},
        "formula_error":formula_error,"continuity_error":continuity_error,"phi_binding_error":phi_binding_error,
        "discounted_shaping":discounted_pbrs,"terminal_telescoping_expected":telescoping,
        "terminal_reward":seq[-1]["reward"]}
    examples=[]
    for i in range(1,n):
        a,b=seq[i-1],seq[i]; delta=b["legs"]["RR"]["front_distance_m"]-a["legs"]["RR"]["front_distance_m"]
        if not b["terminal"] and delta>.0005 and b["reward"]["potential_shaping"]>0:
            examples.append(dict(g=b["g"],tick=b["tick"],front_delta_m=delta,
                RR_clear_before=a["legs"]["RR"]["clearance_m"],RR_clear_after=b["legs"]["RR"]["clearance_m"],
                phi_before=b["reward"]["potential_before"],phi_after=b["reward"]["potential_after"],
                shaping=b["reward"]["potential_shaping"],families=b["reward"]["families"],total=b["reward"]["total"]))
    chosen=[next((x for x in examples if x["total"]<0),None),next((x for x in examples if x["total"]>0),None)]
    late=seq[-150:] if n>=150 else seq
    def wheels(part):
        return {"count":len(part),"nominal":vector_stats([r["nom"][8:] for r in part]),
          "residual":vector_stats([r["res"][8:] for r in part]),"actual_drive":vector_stats([r["drive"][8:] for r in part]),
          "measured":vector_stats([r["wheel_speed"] for r in part]),
          "residual_opposes_nominal_count":[sum(r["res"][8+j]*r["nom"][8+j]<0 for r in part) for j in range(4)],
          "drive_opposes_nominal_count":[sum(r["drive"][8+j]*r["nom"][8+j]<0 for r in part) for j in range(4)]}
    rr_events=[x for x in seq[-1]["history"]["lift_attempt_events"] if x["leg"]=="RR" and x["physics_tick"]>5952]
    output=dict(episode=e["episode_index"],global_range=[seq[0]["g"],seq[-1]["g"]],n=n,
      phase_counts=dict(Counter(r["phase"] for r in seq)),prefix=seq[0]["prefix"],
      phase_transitions=[dict(g=r["g"],tick=r["tick"],phase=r["phase"],end_phase=r["end_phase"],terminal=r["terminal"]) for r in seq if r["phase"]!=r["end_phase"]],
      terminal_reason=seq[-1]["task_reason"],physical_failure=seq[-1]["physical_failure"],
      valid_count=sum(r["valid"] for r in seq),all_no_writes=all(r["no_writes"] for r in seq),
      all_native_verified=all(r["native"]["all_ticks_verified"] for r in seq),
      own_effect_ticks=sum(r["native"]["own_phase_request_effect_tick_count"] for r in seq),
      physical_ticks=sum(r["ticks"] for r in seq),
      RR_events=rr_events,RR_qualified_boundary_count=sum(r["Q"]["RR"] for r in seq),
      RR_crossed_boundary_count=sum(r["C"]["RR"] for r in seq),RR_placed_boundary_count=sum(r["P"]["RR"] for r in seq),
      terminal=point(seq[-1]),RR_peak=point(max(seq,key=lambda r:r["legs"]["RR"]["clearance_m"])),
      RR_closest=point(max(seq,key=lambda r:r["legs"]["RR"]["front_distance_m"])),
      contacts={leg:dict(Counter(contact(r,leg) for r in seq)) for leg in LEGS},
      loads={leg:stats([r["legs"][leg]["load_fraction"] for r in seq]) for leg in LEGS},
      body_speed=stats([r["body_speed"] for r in seq]),omega_norm=stats([r["omega_norm"] for r in seq]),
      reward=reward,forward_reward_examples=[x for x in chosen if x],wheels=wheels(seq),late_wheels=wheels(late),
      late_range=[late[0]["g"],late[-1]["g"]],late_front_delta_RR=late[-1]["legs"]["RR"]["front_distance_m"]-late[0]["legs"]["RR"]["front_distance_m"])
    if e["episode_index"] in (1,2):
        items=[observations[i] for i in range(start,stop)]
        names=("actual_joint_position_deg","actual_joint_velocity_deg_s","actual_wheel_velocity_rad_s","chassis_rpy_rad", "body_angular_velocity","body_linear_velocity","com_position_relative_base","com_velocity_world","projected_gravity_chassis")
        output["pre_observation_statistics"]={name:vector_stats([i["groups"][name] for i in items]) for name in names}
        selected=sorted(set([start,start+2,start+75,stop-151,stop-1]))
        output["pre_observation_samples"]=[dict(g=rows[i]["g"],pre_tick=observations[i]["tick"],**{name:observations[i]["groups"][name] for name in names}) for i in selected]
    results.append(output);start=stop
print(json.dumps(dict(schema="wlr50_clean.p07_completed_episodes_1_2_readonly.v1",runtime_commit=contract["source_git_commit"],
    total_fixed_rows=total,decoded_observations=904,clipped_elements={k:v for k,v in clipped.items() if v},
    episodes=results,cpu_threads=1,model_updates=0),allow_nan=False,separators=(",",":")))
