"""Compact completed1377 event/reward proof; no actor/optimizer/physics."""
from pathlib import Path
import json
import math
import torch
from audit_first_completed_update import read_prefix

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
RUN=ROOT/'runs/ppo_fl_capture_quality_v1/train/20260918T0627100878188Z_g3a50657a96c9_92db59ba0d774fa78e4cc53d5fad8c77'
torch.set_num_threads(1)
allrows,_=read_prefix(RUN/'residual_and_projection_audit.jsonl',256)
rs=allrows[128:]
saved=torch.load(RUN/'rollouts/rollout_001377.pt',map_location='cpu',weights_only=False)
assert rs[-1]['global_policy_decision']==180736
events=[]
for i,r in enumerate(rs):
    a=r['applied_audit']; q=r['policy_request']; t=a['semantic_task']; ev=t['physical_evaluator']
    if i in (22,23,24,25,26,99,100,101,106,107,108):
        h=a['actuator_target_effect_audit']['policy_headroom_evidence']
        events.append(dict(index=i,global_decision=r['global_policy_decision'],physics_tick=a['physics_tick'],
            phase=a['phase_id'],end_phase=a['end_phase_id'],terminal=r['terminal'],termination_reason=t.get('termination_reason'),
            saved_done=bool(saved['dones'][i].item()),saved_reward=float(saved['rewards'][i].item()),
            saved_value=float(saved['values'][i].item()),saved_return=float(saved['returns'][i].item()),
            saved_advantage=float(saved['advantages'][i].item()),
            reward_families=a['reward_breakdown']['families'],terminal_bootstrap_allowed=a.get('terminal_bootstrap_allowed'),
            cap_gate=q['cap_transition_gate_full12'],center=q['history_center_full12'],previous_raw=q['previous_raw_from_current_observation_full12'],
            FR_knee=dict(base_mean=q['base_mean_full12'][3],conditional=q['conditional_mean_full12'][3],raw=q['selected_raw_full12'][3],
                previous_request=q['previous_filtered_request_full12'][3],request=a['projected_residual_full12'][3],
                effective=h['effective_policy_residual_full12'][3],mapped_N=h['baseline_native_plus_controller_full12'][3],
                final=a['actual_drive_target_full12'][3]),
            events=ev['history']['event_ticks'],FL={k:ev['current_legs']['FL'].get(k) for k in ('air','support','top_contact','bearing_force_n','clearance_m')},
            transition=a['phase_transition_action_jump']))
gate=rs[24]['policy_request']; oldmean=.1*gate['base_mean_full12'][3]+.9*gate['previous_raw_from_current_observation_full12'][3]
oldraw=gate['selected_raw_full12'][3]+oldmean-gate['conditional_mean_full12'][3]
likelihood=json.loads((RUN/'rollouts/update_001377_likelihood.json').read_text())
gate_optimization=[]
for m in likelihood['minibatches']:
    for j,indices in enumerate(m['rollout_flat_indices']):
        if 24 in indices:
            gate_optimization.append(dict(minibatch=m['minibatch_index'],actual_advantage=m['actual_advantage'][j],
                old_logp=m['old_log_probability'][j],new_logp=m['optimization_log_probability'][j],ratio=m['ratio'][j],
                strict_clipped=m['clipped_branch_strictly_active'][j]))
report=dict(schema='wlr50_clean.actual1377_handoff_reward.v1',run=str(RUN),source_samples=128,
    actual_gate_index=24,events=events,gate_optimizer_uses=gate_optimization,
    FR_knee_old_kernel_same_saved_base_and_innovation_algebra_not_run=dict(old_center=gate['previous_raw_from_current_observation_full12'][3],
        new_center=gate['history_center_full12'][3],old_conditional=oldmean,new_conditional=gate['conditional_mean_full12'][3],
        old_hypothetical_unfiltered_request=112*math.tanh(oldraw),actual_unfiltered_request=112*math.tanh(gate['selected_raw_full12'][3])),
    done_indices=[i for i in range(128) if saved['dones'][i].item()],
    no_actor_forward_or_rng_or_optimization=True)
with (OUT/'update1377_handoff_reward.json').open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2,allow_nan=False)
print(json.dumps(report,indent=2))
