"""Bounded saved-frame semantic replay; no simulation, policy update or credit."""
import json
from pathlib import Path
from wlr50_clean.ppo.semantic_cooperative_preparation import measure_preparation, workspace_progress
from wlr50_clean.ppo.semantic_rr_capture_context import rr_capture_transfer_context
from wlr50_clean.infrastructure.command_batch import servo_limits_deg

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'runs/ppo_rr_rl_timing_policy_learning_v1/video_eval/validation/20260923T2245371502769Z_gd7e97ee7b7e4_083d82482a6541f8a129462a60dd3dd0/source'
TICK=9656
with (SOURCE/'physical_observations.jsonl').open() as stream:
    for line in stream:
        raw=json.loads(line)
        if raw['physics_tick']==TICK:break
    else:raise ValueError('recorded physical frame absent')
with (SOURCE/'video_policy_decisions.jsonl').open() as stream:
    for line in stream:
        row=json.loads(line)
        if row['end_tick']==TICK:break
    else:raise ValueError('recorded decision endpoint absent')
task=row['step_info']['semantic_task'];ev=task['physical_evaluator']
support=dict(force_noise_floor_n=.2,minimum_other_supports=2,unloaded_leg_maximum_load_fraction=.2)
context=rr_capture_transfer_context(task=task,observation=raw,support_spec=support)
result=measure_preparation(observation=raw,evaluation=ev,rr_context=context,
    support_spec=support,joint_limits=servo_limits_deg('front_left_knee'),clearance_scale_m=.020)
assert result['relevant'] and not result['current_rr_bearing'] and not result['air_is_support']
assert result['fl_range_credit']==1. # Accepted FL is not currently at its negative limit.
print(json.dumps(dict(source=str(SOURCE),replay_not_new_physics=True,
    added_policy_decisions=0,added_ppo_updates=0,diagnostic=result),indent=2))
