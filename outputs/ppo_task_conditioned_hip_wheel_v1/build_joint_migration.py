"""Bind the reviewed task sigma/reward boundary to committed bytes, no simulation."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    from wlr50_clean.ppo import semantic_migration as m, semantic_cli as cli, semantic_training as t
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--expected-head', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    checkpoint, output = args.checkpoint.resolve(strict=True), args.output.resolve()
    if output.exists() or not output.is_relative_to(Path(__file__).resolve().parent):
        raise ValueError('write a new migration file within this experiment output directory')
    metadata = m.checkpoint_metadata(checkpoint)
    contract = cli.runtime_contract(expected_head=args.expected_head, semantic_version='v3',
                                    experiment_id='task_conditioned_hip_wheel_v1')
    old = metadata['runtime_contract']
    changed = sorted(p for p in old['files'].keys() | contract['files'].keys()
                     if old['files'].get(p) != contract['files'].get(p))
    code = {p: contract['files'][p] for p in changed if p.startswith(('src/', 'scripts/'))}
    plan = m.build_migration_plan(checkpoint, contract, allowed_changed_files=changed,
        reason='Joint version boundary: observed-task positive full12 physical innovation sigma, current front rolling retention and bounded collider-space quality; preserve learned weights, complete Adam, effective LR, Identity and RNG; collect fresh on-policy rollouts.',
        task_conditioned_hip_wheel_review={
            'reason': 'Reviewed actor/sample/likelihood/loader and soft reward scopes. Mean architecture, rho=.9, residual capacity, nominal, dispatch, hard task events, physics and actuator limits remain unchanged. Existing front quality retains its .03/s maximum; new geometry costs at most .03/s. No directional bias, masked nominal, manual action deployment, auxiliary update or physical success is implied.',
            'reviewed_code_sha256': code})
    t.write_json(output, plan)
    verified = m.validate_migration_plan(checkpoint, contract, output)
    factor = verified['task_conditioned_hip_wheel_factor']
    print(json.dumps({'migration': str(output), 'plan_sha256': verified['plan_sha256'],
        'source_checkpoint': str(checkpoint), 'counter_origin': factor['counter_origin'],
        'effective_learning_rate': factor['source_effective_learning_rate'],
        'target_policy_version': factor['target_policy_version'],
        'target_head': contract['source_git_commit'], 'changed_files': changed,
        'new_policy_decisions': 0, 'new_ppo_updates': 0, 'new_optimizer_steps': 0}, indent=2))


if __name__ == '__main__':
    main()
