"""Read one durable 128-decision update; no actor forward or optimizer replay."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import torch
from audit_first_completed_update import read_prefix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--update-index', type=int, default=1, help='One-based completed update within this run')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    run = args.run.resolve(strict=True)
    if args.update_index < 1:
        raise ValueError('update-index must be positive')
    updates, _ = read_prefix(run/'optimizer_updates.jsonl', args.update_index)
    update = updates[-1]
    number = update['ppo_update']
    all_rows, _ = read_prefix(run/'residual_and_projection_audit.jsonl', 128*args.update_index)
    rows = all_rows[-128:]
    saved = torch.load(run/f'rollouts/rollout_{number:06d}.pt', map_location='cpu', weights_only=False)
    likelihood = json.loads((run/f'rollouts/update_{number:06d}_likelihood.json').read_text())
    obs = saved['observations']['policy'][:, 0]
    assert tuple(obs.shape) == (128, 372)
    assert rows[-1]['global_policy_decision'] == update['global_policy_decisions']
    requests = [row['policy_request'] for row in rows]
    tensor = lambda key: torch.tensor([row[key] for row in requests], dtype=torch.float32)
    stages = obs[:, :13].argmax(-1)
    scale = torch.ones(128, 12)
    scale[stages >= 5, 3] = 24/112
    expected = tensor('learned_sigma_full12')*.25*scale
    effective = tensor('effective_sigma_full12')
    relative_sigma_error = float(((effective-expected).abs()/expected).max())
    normal = torch.distributions.Normal(*[value[:, 0] for value in saved['distribution_params']])
    logp_error = float((normal.log_prob(saved['actions'][:, 0]).sum(-1)-saved['actions_log_prob'][:, 0, 0]).abs().max())
    uses = Counter(i for batch in likelihood['minibatches'] for group in batch['rollout_flat_indices'] for i in group)
    first = likelihood['minibatches'][0]
    ratio_error = max(abs(v-1) for v in first['ratio'])
    checks = {
        'actual_raw_saved_exact': torch.equal(tensor('selected_raw_full12'), saved['actions'][:, 0]),
        'actual_mu_saved_exact': torch.equal(tensor('conditional_mean_full12'), saved['distribution_params'][0][:, 0]),
        'actual_sigma_saved_exact': torch.equal(effective, saved['distribution_params'][1][:, 0]),
        'actual_sigma_gate_exact': torch.equal(tensor('sigma_scaling_gate_full12').bool(), scale != 1),
        'actual_sigma_multiplier_exact': torch.equal(tensor('innovation_sigma_multiplier_full12'), scale),
        'actual_logp_saved_exact': torch.equal(torch.tensor([r['old_log_probability'] for r in rows]), saved['actions_log_prob'][:, 0, 0]),
        'effective_sigma_matches_single_channel_formula': relative_sigma_error < 2e-6,
        'raw_normal_logp_matches_storage': logp_error < 2e-5,
        'first_unupdated_minibatch_ratio_near_one': ratio_error < 2e-5,
        'all_128_used_five_times': uses == Counter({i: 5 for i in range(128)}),
        'twenty_actual_optimizer_minibatches': len(likelihood['minibatches']) == update['optimizer_steps'] == 20,
        'one_draw_no_extra_forward': all(r['sampling_draws'] == 1 and r['extra_random_draws'] == r['extra_model_forwards'] == 0 for r in requests),
        'all_physics_ticks_verified': all(r['applied_audit']['actuator_target_effect_audit_summary']['all_ticks_verified'] for r in rows),
        'residual_permission_full12': all(r['applied_audit']['actuator_target_effect_audit']['phase_mask_full12'] == [1.]*12 for r in rows),
        'ordinary_phase_changes_not_done': all(not r['terminal'] for r in rows if r['applied_audit']['phase_id'] != r['applied_audit']['end_phase_id']),
    }
    result = dict(schema='wlr50_clean.actual_physical_innovation_update_audit.v1', run=str(run),
        update=update, checks=checks, all_checks_passed=all(checks.values()),
        actual_phase_counts=dict(Counter(r['applied_audit']['phase_id'] for r in rows)),
        policy_versions=sorted({r.get('policy_version', '') for r in requests}),
        scaled_decisions=int((stages >= 5).sum()), unscaled_decisions=int((stages < 5).sum()),
        effective_sigma_max_relative_error=relative_sigma_error, gaussian_logp_cpu_max_error=logp_error,
        first_unupdated_minibatch_max_ratio_error=ratio_error,
        limits=['Only first actual32-sample minibatch is pre-update, not all128.',
            'Read-only arithmetic verifies saved distribution wiring, not improved motion.',
            'No new physical decisions, model forwards, random draws or optimizer steps.'])
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))
    if not all(checks.values()):
        raise RuntimeError('Saved first-update audit failed; see preserved output')


if __name__ == '__main__':
    main()
