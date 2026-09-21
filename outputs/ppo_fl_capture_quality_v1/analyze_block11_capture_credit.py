"""Completed update1394 only: saved tensors + actual optimizer hooks, no model."""
from pathlib import Path
from collections import Counter
import json
import math
import torch
from audit_first_completed_update import read_prefix

torch.set_num_threads(1)
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
RUN = ROOT / 'runs/ppo_fl_capture_quality_v1/train/20260918T0833241304449Z_g3a50657a96c9_f4abda62b32041808fdc243bdac4e573'
updates, update_hash = read_prefix(RUN / 'optimizer_updates.jsonl', 4)
rows, prefix_hash = read_prefix(RUN / 'residual_and_projection_audit.jsonl', 512)
adv_audit, _ = read_prefix(RUN / 'advantage_audit.jsonl', 4)
assert [u['ppo_update'] for u in updates] == [1392, 1393, 1394, 1395]
assert updates[2]['global_policy_decisions'] == 182912
rows = rows[256:384]
saved = torch.load(RUN / 'rollouts/rollout_001394.pt', map_location='cpu', weights_only=False)
hook = json.loads((RUN / 'rollouts/update_001394_likelihood.json').read_text())
assert len(hook['minibatches']) == updates[2]['optimizer_steps'] == 20
raw_gae = (saved['returns'] - saved['values']).flatten()
normalized = (raw_gae - raw_gae.mean()) / (raw_gae.std() + 1.e-8)
saved_adv = saved['advantages'].flatten()
recurrence_errors = []
for i in range(127):
    expected = saved['rewards'][i] + .9985 * saved['values'][i+1] - saved['values'][i] + .9985 * .99 * raw_gae[i+1]
    recurrence_errors.append(abs(float(expected.item()) - float(raw_gae[i])))
checks = {
    'raw_exact_storage': torch.equal(torch.tensor([r['raw_policy_action_full12'] for r in rows]), saved['actions'][:,0]),
    'reward_exact_storage': torch.equal(torch.tensor([r['reward'] for r in rows]), saved['rewards'][:,0,0]),
    'reward_family_float32_exact_storage': torch.equal(torch.tensor([sum(r['applied_audit']['reward_breakdown']['families'].values()) for r in rows]), saved['rewards'][:,0,0]),
    'old_mean_exact_storage': torch.equal(torch.tensor([r['old_distribution_mean_full12'] for r in rows]), saved['distribution_params'][0][:,0]),
    'old_sigma_exact_storage': torch.equal(torch.tensor([r['old_distribution_std_full12'] for r in rows]), saved['distribution_params'][1][:,0]),
    'old_logp_exact_storage': torch.equal(torch.tensor([r['old_log_probability'] for r in rows]), saved['actions_log_prob'][:,0,0]),
    'no_dones_or_terminal_in_update': not bool(saved['dones'].any()) and not any(r['terminal'] for r in rows),
    'request_raw_matches_collection': all(r['policy_request']['selected_raw_full12'] == r['raw_policy_action_full12'] for r in rows),
}
uses = {i: [] for i in range(128)}
hook_logp_error = hook_adv_error = 0.
for m in hook['minibatches']:
    for j, indices in enumerate(m['rollout_flat_indices']):
        assert len(indices) == 1
        i = indices[0]
        hook_logp_error = max(hook_logp_error, abs(m['old_log_probability'][j] - saved['actions_log_prob'][i].item()))
        hook_adv_error = max(hook_adv_error, abs(m['actual_advantage'][j] - saved_adv[i].item()))
        old_mu = saved['distribution_params'][0][i,0].tolist()
        old_sd = saved['distribution_params'][1][i,0].tolist()
        action = saved['actions'][i,0].tolist()
        new_mu = m['current_conditional_mean'][j]
        new_sd = m['current_conditional_sigma'][j]
        channel_delta = [(-math.log(new_sd[k]) - .5*((action[k]-new_mu[k])/new_sd[k])**2)
                         - (-math.log(old_sd[k]) - .5*((action[k]-old_mu[k])/old_sd[k])**2) for k in range(12)]
        strict_expected = (m['actual_advantage'][j] > 0 and m['ratio'][j] > 1 + hook['clip_param']) or (m['actual_advantage'][j] < 0 and m['ratio'][j] < 1 - hook['clip_param'])
        assert strict_expected == m['clipped_branch_strictly_active'][j]
        uses[i].append(dict(minibatch=m['minibatch_index'], ratio=m['ratio'][j], normalized_advantage=m['actual_advantage'][j],
            strict_clipped=m['clipped_branch_strictly_active'][j], joint_delta_logp=m['optimization_log_probability'][j]-m['old_log_probability'][j],
            conditional_FL_hip_knee=new_mu[:2], sigma_FL_hip_knee=new_sd[:2],
            arithmetic_channel_delta_logp_FL_hip_knee=channel_delta[:2],
            arithmetic_other10_sum_delta_logp=sum(channel_delta[2:])))
checks.update(all_samples_used_five_times=all(len(x)==5 for x in uses.values()), hook_logp_exact=hook_logp_error==0., hook_advantage_exact=hook_adv_error==0.)
assert all(checks.values()), checks

indices = [79,83,87,95,96,97,98,99,100,101,103,105,111,127]
selected = []
for i in indices:
    row = rows[i]; a = row['applied_audit']; q = row['policy_request']; rb = a['reward_breakdown']
    ev = a['semantic_task']['physical_evaluator']; fl = ev['current_legs']['FL']
    effect = a['actuator_target_effect_audit']; h = effect['policy_headroom_evidence']
    old_mu = saved['distribution_params'][0][i,0].tolist(); old_sd = saved['distribution_params'][1][i,0].tolist()
    value = saved['values'][i].item()
    vn = saved['values'][i+1].item() if i<127 else None
    selected.append(dict(index=i, global_policy_decision=row['global_policy_decision'], tick=a['physics_tick'], phase=a['phase_id'], end_phase=a['end_phase_id'],
        FL={k:fl[k] for k in ('clearance_m','front_distance_m','air','top_contact','support','bearing_force_n')},
        historical_FL_placed=ev['history']['placed']['FL'],
        reward=saved['rewards'][i].item(), reward_families=rb['families'], reward_potential_before=rb['potential_before'],
        reward_potential_after=rb['potential_after'], potential_shaping=rb['potential_shaping'], terminal_event_reward=rb['terminal_event'],
        value=value, next_value=vn, td_delta=saved['rewards'][i].item()+.9985*vn-value if vn is not None else None,
        raw_gae=float(raw_gae[i]), normalized_advantage=float(saved_adv[i]), done=bool(saved['dones'][i].item()),
        FL_hip_knee=dict(base_mean=q['base_mean_full12'][:2], conditional_mean=old_mu[:2], sigma=old_sd[:2], raw=q['selected_raw_full12'][:2],
            nominal=a['nominal_action_full12'][:2], mapped_baseline=h['baseline_native_plus_controller_full12'][:2],
            filtered_request=h['requested_policy_residual_full12'][:2], effective_headroom_residual=h['effective_policy_residual_full12'][:2],
            final=a['actual_drive_target_full12'][:2], physical_units='canonical_servo_degrees',
            request_equals_effective=h['requested_policy_residual_full12'][:2]==h['effective_policy_residual_full12'][:2]),
        dispatch_verified=all(effect[k] is True for k in ('verified','setter_dispatch_targets_equal','actual_mapping_matches_dispatch')),
        actual_optimizer_uses=uses[i], last_observed_use_mean_delta_FL_hip_knee=[uses[i][-1]['conditional_FL_hip_knee'][k]-old_mu[k] for k in range(2)],
        last_observed_use_sigma_ratio_FL_hip_knee=[uses[i][-1]['sigma_FL_hip_knee'][k]/old_sd[k] for k in range(2)],
        signed_joint_likelihood_change=float(saved_adv[i])*uses[i][-1]['joint_delta_logp']))

comparison = json.loads((OUT / 'CP182528_P05_diagnosis.json').read_text())
receipt = dict(schema='wlr50_clean.block11_capture_credit.v1', run=str(RUN),
    source_prefix_hashes={'optimizer_first4_lines':update_hash,'audit_first512_lines':prefix_hash},
    completed_updates_seen=[dict(update=u['ppo_update'],global_decisions=u['global_policy_decisions'],optimizer_steps=u['optimizer_steps']) for u in updates],
    deep_audit_scope='update1394 only; 128 saved decisions; 20 actual minibatch hooks; selected14 records; no active tail',
    update=1394, collection_weight_note='Collected after completed1392/1393; not fixed CP182528 trajectory.',
    checks=checks, max_normalization_error=float((normalized-saved_adv).abs().max()), max_interior_GAE_recurrence_error=max(recurrence_errors),
    first_unupdated_minibatch_max_ratio_error=max(abs(x-1) for x in hook['minibatches'][0]['ratio']),
    first_minibatch_includes_capture=99 in [x[0] for x in hook['minibatches'][0]['rollout_flat_indices']],
    whole_rollout_advantage=adv_audit[2]['overall'], request_phase_advantage=adv_audit[2]['by_request_phase'],
    capture_index=99, capture_reward_positive_linear_GAE_contribution={str(i): (.9985*.99)**(99-i)*saved['rewards'][99].item() for i in (83,95,96,97,98,99)},
    linear_contribution_scope='Holding old values and all other rewards fixed; raw GAE only, not a simulated counterfactual or normalized-gradient isolation.',
    selected=selected, deterministic_comparison={'source':comparison['source'],'checkpoint':comparison['checkpoint'],
        'terminal_gap_m':comparison['terminal_FL']['clearance_m'],'no_FL_placed':comparison['events']['placed'].get('FL') is None,
        'FL_hip_request_terminal_deg':4.176236889,'FL_hip_base_terminal':.23521884,
        'scope':'Existing sealed fixed CP182528 run, different observations/trajectory and earlier weights; descriptive, not causal same-state comparison.'},
    limitations=['The last observed minibatch use is before that minibatch update, not final saved-checkpoint distribution.',
        'A joint 12D likelihood rise cannot establish individual-joint credit, mean improvement or whole-task success.',
        'No final-bootstrap value stored here: checked all127 interior recurrences, did not fabricate tail value.',
        'Observed descent combines changing N/mapper, learned request and whole-body dynamics; hip sign alone is not a causal proof.',
        'No future block11 outcomes or incomplete updates were read or credited.'],
    no_model_forward_optimizer_simulator_or_production_write=True)
with (OUT / 'block11_capture_credit.json').open('x',encoding='utf-8') as f:
    json.dump(receipt,f,indent=2,allow_nan=False)
print(json.dumps({k:receipt[k] for k in ('checks','max_normalization_error','max_interior_GAE_recurrence_error','first_unupdated_minibatch_max_ratio_error','capture_reward_positive_linear_GAE_contribution')},indent=2))
for s in selected:
    u=s['actual_optimizer_uses'][-1]
    print(s['tick'],s['raw_gae'],s['normalized_advantage'],u['ratio'],s['last_observed_use_mean_delta_FL_hip_knee'],u['arithmetic_channel_delta_logp_FL_hip_knee'],u['arithmetic_other10_sum_delta_logp'])
