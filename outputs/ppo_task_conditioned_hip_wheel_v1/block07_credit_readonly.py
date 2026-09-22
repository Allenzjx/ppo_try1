"""Saved reward/return audit only; does not load a policy or run physics."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT / "src"))
from rr_probe_readonly import lines, stats
from wlr50_clean.ppo.semantic_supervisor import TaskStageSupervisor

RUN = ROOT / "runs/ppo_task_conditioned_hip_wheel_v1/train/20260921T0849405935866Z_gee5a9651591d_12b59026033c4ad08c274c684926dadc"
supervisor = TaskStageSupervisor(ROOT / "configs/ppo_task_conditioned_hip_wheel_v1/stage_task_spec.yaml")
rows = list(lines(RUN / "residual_and_projection_audit.jsonl"))
assert len(rows) == 512
bytick = {r["applied_audit"]["physics_tick"]:r for r in rows}
selected = []
for tick in (6408,6784,6792,6808,7208,8528,9552):
    s = bytick[tick]["applied_audit"]; task = s["semantic_task"]; ev = task["physical_evaluator"]
    phi = supervisor.physical_potential(ev)
    assert abs(phi-task["task_progress_potential"]) < 1e-12
    retention = {leg:supervisor._current_capture_retention(leg,ev) for leg in ("FR","FL","RR")}
    placed_phi = {leg:.85/4*(.8+.2*x) for leg,x in retention.items()}
    selected.append(dict(tick=tick,phase=s["phase_id"],potential=phi,placed_retention=retention,
        placed_leg_phi_shares=placed_phi,RL_phi_share=phi-sum(placed_phi.values()),
        RR_gap_m=ev["current_legs"]["RR"]["clearance_m"],
        RR_outside_top_xy_distance_m=ev["current_legs"]["RR"]["top_xy_outside_distance_m"],
        RR_ground=ev["current_legs"]["RR"]["ground_contact"],
        body_front_m=ev["goal_features"]["body_forward_m"],
        RL_placed=ev["history"]["placed"]["RL"], logged_reward=s["reward_breakdown"]["total"]))

windows = []
for lo,hi in ((6408,6808),(8528,9552)):
    ss = [r["applied_audit"] for r in rows if lo < r["applied_audit"]["physics_tick"] <= hi]
    rewards = [s["reward_breakdown"] for s in ss]
    assert len(ss) == (hi-lo)//8 and all(s["physics_ticks"] == 8 for s in ss)
    before = bytick[lo]["applied_audit"]["semantic_task"]; after = bytick[hi]["applied_audit"]["semantic_task"]
    windows.append(dict(start_observation_tick=lo,end_observation_tick=hi,decisions=len(ss),
        duration_s=(hi-lo)/120.,potential_before=before["task_progress_potential"],potential_after=after["task_progress_potential"],
        reward_total_sum=sum(r["total"] for r in rewards),
        potential_shaping_sum=sum(r["potential_shaping"] for r in rewards),
        time_cost_sum=.02*sum(r["elapsed_physics_s"] for r in rewards),
        terminal_event_sum=sum(r["terminal_event"] for r in rewards),
        other_family_sums={name:sum(r["families"][name] for r in rewards) for name in ("body_stability","contact_motion_quality","control_smoothness","control_regularization")},
        reward_stats=stats(r["total"] for r in rewards),
        body_forward_displacement_m=after["goal_features"]["body_forward_m"]-before["goal_features"]["body_forward_m"]))

advantage = list(lines(RUN / "advantage_audit.jsonl"))[-1]
last = rows[-1]; s = last["applied_audit"]
assert s["physics_tick"] == 9552 and not last["terminal"] and s["terminal_bootstrap_allowed"]
result = dict(schema="block07.reward_credit_readonly.v1", source=str(RUN),
    saved_reward_recomputed_potential_matches=True, selected_states=selected,reward_windows=windows,
    last_budget_boundary=dict(global_policy_decision=last["global_policy_decision"],episode_tick=s["physics_tick"],
        terminal=last["terminal"],termination_reason=s["termination_reason"],time_outs=s["time_outs"],
        terminal_bootstrap_allowed=s["terminal_bootstrap_allowed"],local_timeout=s["semantic_task"]["local_timeout"],
        stage_elapsed_s=s["semantic_task"]["stage_elapsed_s"], completed_episode_rows=len(list(lines(RUN/"completed_episodes.jsonl"))),
        reward=s["reward_breakdown"]["total"],terminal_event=s["reward_breakdown"]["terminal_event"]),
    last_rollout_advantage_audit={key:advantage[key] for key in ("ppo_update_intended","first_global_policy_decision","last_global_policy_decision","gamma","lambda","overall","tail_bootstrap")},
    restrictions="Only saved reward/physical evaluator fields; no rerun, new reward, state injection, policy forward or production edits.")
with (OUT / "block07_reward_credit_readonly.json").open("x",encoding="utf-8") as stream:
    json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
print(json.dumps({k:v for k,v in result.items() if k != "last_rollout_advantage_audit"},ensure_ascii=False,indent=2))
