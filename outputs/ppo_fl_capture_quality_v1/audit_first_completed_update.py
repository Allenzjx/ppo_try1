"""Bounded read-only audit of one completed128-row update; no active full-log scan."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics


def read_prefix(path, count):
    rows, digest = [], hashlib.sha256()
    with path.open('rb') as stream:
        for _ in range(count):
            line = stream.readline()
            if not line or not line.endswith(b'\n'):
                raise ValueError('required completed prefix is not yet durable')
            rows.append(json.loads(line))
            digest.update(line)
    return rows, digest.hexdigest()


def stats(values):
    values = list(values)
    return {'count': len(values), 'minimum': min(values) if values else None,
        'maximum': max(values) if values else None, 'mean': statistics.fmean(values) if values else None,
        'sum': sum(values), 'positive_count': sum(v > 0 for v in values)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    run = args.run.resolve(strict=True)
    updates, update_hash = read_prefix(run/'optimizer_updates.jsonl', 1)
    update = updates[0]
    number = update['ppo_update']
    rows, row_hash = read_prefix(run/'residual_and_projection_audit.jsonl', 128)
    advantage, _ = read_prefix(run/'advantage_audit.jsonl', 1)
    likelihood_path = run/f'rollouts/update_{number:06d}_likelihood.json'
    likelihood = json.loads(likelihood_path.read_text())
    import torch
    torch.set_num_threads(1)
    saved = torch.load(run/f'rollouts/rollout_{number:06d}.pt', map_location='cpu', weights_only=False)
    assert saved['actions'].shape == (128, 1, 12)
    assert rows[-1]['global_policy_decision'] == update['global_policy_decisions']
    assert len(likelihood['minibatches']) == update['optimizer_steps'] == 20
    assert advantage[0]['ppo_update_intended'] == number
    phases = Counter(r['applied_audit']['phase_id'] for r in rows)
    assert phases == Counter({p: x['sample_count'] for p, x in advantage[0]['by_request_phase'].items()})
    native = [r['applied_audit']['actuator_target_effect_audit'] for r in rows]
    requested = [n['policy_headroom_evidence'] for n in native]
    quality_samples = [s for r in rows for s in r['applied_audit']['reward_breakdown']['front_quality_sample_audit']]
    family_error = []
    substate = defaultdict(list)
    for row in rows:
        reward = row['applied_audit']['reward_breakdown']
        samples = reward['front_quality_sample_audit']
        family_error.append(abs(sum(s['weighted_quality_cost'] for s in samples)+reward['families']['body_stability']))
        for sample in samples: substate[sample['substate']].append(sample)
    model = [r['policy_request'] for r in rows]
    raw = torch.tensor([r['raw_policy_action_full12'] for r in rows], dtype=torch.float32)
    checks = {
        'raw_exact_saved': torch.equal(raw, saved['actions'][:,0]),
        'request_raw_equal_collection': all(r['policy_request']['selected_raw_full12'] == r['raw_policy_action_full12'] for r in rows),
        'mean_exact_saved': torch.equal(torch.tensor([r['old_distribution_mean_full12'] for r in rows]), saved['distribution_params'][0][:,0]),
        'sigma_exact_saved': torch.equal(torch.tensor([r['old_distribution_std_full12'] for r in rows]), saved['distribution_params'][1][:,0]),
        'logp_exact_saved': torch.equal(torch.tensor([r['old_log_probability'] for r in rows]), saved['actions_log_prob'][:,0,0]),
        'conditional_mean_equal_collected': all(r['policy_request']['conditional_mean_full12'] == r['old_distribution_mean_full12'] for r in rows),
        'conditional_sigma_equal_collected': all(r['policy_request']['effective_sigma_full12'] == r['old_distribution_std_full12'] for r in rows),
        'selected_logp_equal_collected': all(r['policy_request']['selected_raw_log_probability'] == r['old_log_probability'] for r in rows),
        'all_masks_full12_one': all(n['phase_mask_full12'] == [1.]*12 for n in native),
        'all_native_endpoint_verifications': all(all(n[k] is True for k in ('verified','setter_dispatch_targets_equal','actual_mapping_matches_dispatch','same_tick_counterfactual')) for n in native),
        'all_physical_tick_audits_verified': all(r['applied_audit']['actuator_target_effect_audit_summary']['all_ticks_verified'] for r in rows),
        'front_beta_positive_range': bool(quality_samples) and all(.015-1e-12 <= s['effective_beta_per_s'] <= .03+1e-12 for s in quality_samples),
        'front_cost_family_sum_error_below_1e_12': max(family_error) < 1e-12,
        'one_real_actor_sample_no_extra_rng': all(r['mode'] == 'training_style_conditional_gaussian' and r['sampling_draws'] == 1 and r['extra_random_draws'] == r['extra_model_forwards'] == 0 for r in model),
    }
    first = likelihood['minibatches'][0]
    checks['first_minibatch_ratio_near_one'] = max(abs(v-1.) for v in first['ratio']) < 2e-5
    if not all(checks.values()): raise RuntimeError(checks)
    channel_rows = []
    for i, name in enumerate(native[0]['canonical_order']):
        values = [n['native_target_delta']['servo_position_rad'][i] *180/math.pi if i < 8
            else n['native_target_delta']['wheel_velocity_rad_s'][i-8] for n in native]
        channel_rows.append({'channel': name, 'physical_unit': 'deg' if i < 8 else 'rad/s',
            'native_delta_sign_semantics': 'actual_joint_axis_not_canonical_servo_sign',
            'base_mean': stats(r['base_mean_full12'][i] for r in model),
            'conditional_mean': stats(r['conditional_mean_full12'][i] for r in model),
            'effective_sigma': stats(r['effective_sigma_full12'][i] for r in model),
            'selected_raw': stats(r['selected_raw_full12'][i] for r in model),
            'requested_physical_residual': stats(r['requested_policy_residual_full12'][i] for r in requested),
            'effective_headroom_residual': stats(r['effective_policy_residual_full12'][i] for r in requested),
            'same_tick_native_delta': stats(values), 'same_tick_native_delta_abs_mean': statistics.fmean(abs(v) for v in values)})
    body = [r['applied_audit']['reward_breakdown']['families']['body_stability'] for r in rows]
    task = [r['applied_audit']['reward_breakdown']['families']['task_progress'] for r in rows]
    final = rows[-1]['applied_audit']['semantic_task']['physical_evaluator']
    receipt = {'schema':'wlr50_clean.first_completed_fl_update_audit.v1', 'run': str(run),
        'scope': 'first_completed_update_only;128_decisions;decision_endpoint_execution_samples',
        'source_prefix_sha256': {'audit_first128_raw_lines':row_hash, 'optimizer_first_line':update_hash},
        'ppo_update':number, 'global_policy_decisions':update['global_policy_decisions'], 'new_decisions':128,
        'optimizer_steps_this_update':update['optimizer_steps'], 'counts_by_request_phase':dict(phases),
        'completed_update':update, 'checks':checks,
        'first_minibatch_ratio':stats(first['ratio']),
        'first_minibatch_max_abs_ratio_minus_one':max(abs(v-1.) for v in first['ratio']),
        'quality_physics_sample_count':len(quality_samples), 'quality_physics_seconds':sum(s['dt_s'] for s in quality_samples),
        'quality_beta':stats(s['effective_beta_per_s'] for s in quality_samples),
        'quality_cost_positive_physics_samples':sum(s['weighted_quality_cost'] > 0 for s in quality_samples),
        'body_family_signed':stats(body), 'task_family_signed':stats(task),
        'quality_family_sum_max_abs_error':max(family_error),
        'quality_by_substate':{key:{'samples':len(values),'physical_seconds':sum(s['dt_s'] for s in values),
            'beta':stats(s['effective_beta_per_s'] for s in values),
            'weighted_cost':sum(s['weighted_quality_cost'] for s in values)} for key, values in substate.items()},
        'native_counterfactual_scopes':sorted({n['counterfactual_scope'] for n in native}),
        'native_endpoint_nonzero_effect_count':sum(n['changed_target_channel_count']>0 for n in native),
        'endpoint_headroom_clip_count':sum(bool(r['clipped_servo_indices']) for r in requested),
        'endpoint_final_slew_or_limit_modified_servo_count':sum(any(abs(a-b)>1e-7 for a,b in zip(
            row['applied_audit']['actual_drive_target_full12'][:8],
            req['candidate_native_target_before_final_slew_full12'][:8])) for row,req in zip(rows,requested)),
        'channels':channel_rows,
        'last_row':{'phase':rows[-1]['applied_audit']['end_phase_id'],
            'tick':rows[-1]['applied_audit']['physics_tick'],
            'FR':final['current_legs']['FR'], 'FL':final['current_legs']['FL'],
            'placed':final['history']['placed'], 'crossed':final['history']['front_edge_crossed']},
        'phase_advantages':advantage[0]['by_request_phase'],
        'claim_limits':['No P05 credit unless counts_by_request_phase contains real P05 samples.',
            'Nonzero quality cost and nonzero actuator target effects do not establish stability improvement.',
            'Native target delta is a same-pre-tick counterfactual, not comparison to a separately run zero trajectory.',
            'Native target evidence does not by itself prove actual joint-speed tracking.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream: json.dump(receipt, stream, indent=2, allow_nan=False)
    print(json.dumps({key:receipt[key] for key in ('ppo_update','global_policy_decisions','new_decisions',
        'counts_by_request_phase','quality_physics_sample_count','quality_physics_seconds','quality_beta',
        'body_family_signed','task_family_signed','first_minibatch_max_abs_ratio_minus_one',
        'native_endpoint_nonzero_effect_count','endpoint_headroom_clip_count','endpoint_final_slew_or_limit_modified_servo_count','checks')},indent=2))


if __name__ == '__main__': main()
