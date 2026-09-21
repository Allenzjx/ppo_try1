"""Compact accounting of completed real PPO blocks, never requested phase counts."""
import argparse
import collections
import json
from pathlib import Path


def rows(path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def summarize(run):
    run = Path(run).resolve(strict=True)
    manifest = json.loads((run/'run_manifest.json').read_text(encoding='utf-8'))
    assert manifest['lifecycle'] in ('SUCCEEDED', 'STOPPED_AT_VERIFIED_UPDATE_BOUNDARY')
    coverage = collections.Counter()
    rewards = collections.defaultdict(lambda: collections.defaultdict(float))
    quality = collections.defaultdict(lambda: collections.defaultdict(float))
    substates = {}
    terminals = []
    phase_changes = []
    first, last = None, None
    for row in rows(run/'residual_and_projection_audit.jsonl'):
        info = row.get('applied_audit') or row.get('info') or {}
        phase = info['phase_id']
        coverage[phase] += 1
        decision = row['global_policy_decision']
        first = decision if first is None else first
        if last is not None:
            assert decision == last + 1
        last = decision
        reward = info.get('reward_breakdown') or info.get('reward') or {}
        for key, value in reward.get('families', {}).items():
            rewards[phase][key] += value
        for key, value in reward.get('cost_components', {}).items():
            if isinstance(value, (int, float)):
                quality[phase][key] += value
        for sample in reward.get('front_quality_sample_audit', []):
            key = sample['phase'] + '/' + sample['substate']
            part = substates.setdefault(key, {'physics_samples': 0, 'duration_s': 0.,
                'beta_min_per_s': float('inf'), 'beta_max_per_s': 0.,
                'raw_tilt_integral': 0., 'raw_rate_integral': 0., 'actual_quality_cost': 0.})
            part['physics_samples'] += 1
            part['duration_s'] += sample['dt_s']
            part['beta_min_per_s'] = min(part['beta_min_per_s'], sample['effective_beta_per_s'])
            part['beta_max_per_s'] = max(part['beta_max_per_s'], sample['effective_beta_per_s'])
            part['raw_tilt_integral'] += sample['raw_tilt_cost']*sample['dt_s']
            part['raw_rate_integral'] += sample['raw_rate_cost']*sample['dt_s']
            part['actual_quality_cost'] += sample['weighted_quality_cost']
        if info.get('termination_reason'):
            terminals.append({'decision': decision, 'phase': phase,
                              'reason': info['termination_reason'], 'task_success': info.get('task_success')})
        if phase != info.get('end_phase_id', phase):
            phase_changes.append({'decision': decision, 'from': phase, 'to': info['end_phase_id'],
                                  'terminal': bool(info.get('termination_reason'))})
    updates = list(rows(run/'optimizer_updates.jsonl'))
    advantages = list(rows(run/'advantage_audit.jsonl'))
    assert sum(coverage.values()) == len(updates)*128
    return {'run': str(run), 'lifecycle': manifest['lifecycle'],
            'first_global_decision': first, 'last_global_decision': last,
            'new_decisions': sum(coverage.values()), 'new_ppo_updates': len(updates),
            'new_optimizer_steps': sum(x['optimizer_steps'] for x in updates),
            'actual_request_phase_coverage': dict(coverage), 'signed_family_contribution_sums': dict(rewards),
            'cost_component_integrals_by_phase': dict(quality), 'terminals': terminals,
            'actual_front_quality_by_task_substate': substates,
            'ordinary_phase_changes': phase_changes,
            'updates': [{k: x.get(k) for k in ('ppo_update', 'global_policy_decisions', 'kl_mean',
                         'value_loss', 'optimizer_learning_rate', 'finite_nonzero_gradient_observed')}
                        for x in updates],
            'advantage_by_update': [{k: x.get(k) for k in ('ppo_update', 'stored_advantage_semantics',
                                    'overall', 'by_request_phase', 'terminal_samples')} for x in advantages],
            'teacher_prefixes_are_excluded_from_coverage': True,
            'training_execution_success_is_not_full_task_success': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.run)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({k: result[k] for k in ('new_decisions', 'new_ppo_updates', 'new_optimizer_steps',
                     'actual_request_phase_coverage', 'signed_family_contribution_sums', 'terminals')}, indent=2))
