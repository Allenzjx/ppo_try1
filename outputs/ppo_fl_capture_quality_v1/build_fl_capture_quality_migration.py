"""Build/revalidate the reviewed real checkpoint boundary; no simulator/optimizer."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))


def main():
    from wlr50_clean.ppo import semantic_migration as m, semantic_cli as cli, semantic_training as t
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--expected-head', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    cp, output = args.checkpoint.resolve(strict=True), args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    metadata = m.checkpoint_metadata(cp)
    contract = cli.runtime_contract(expected_head=args.expected_head, semantic_version='v3', experiment_id='fl_capture_quality_v1')
    old = metadata['runtime_contract']
    delta = sorted(path for path in old['files'].keys() | contract['files'].keys()
        if old['files'].get(path) != contract['files'].get(path))
    plan = m.build_migration_plan(cp, contract, allowed_changed_files=delta,
        reason='Preserve learned RR branch and enable FL fine-gap capture plus bounded P01/P02 body quality on fresh rollouts',
        fl_capture_quality_review={'reason': 'Reviewed reward/potential-feature-only task boundary; same N, hard acceptance, quarter HISTORY and actuator chain',
            'reviewed_code_sha256': {path: contract['files'][path] for path in delta if path in m.FL_CAPTURE_QUALITY_FILES}})
    t.write_json(output, plan)
    verified = m.validate_migration_plan(cp, contract, output)
    print(json.dumps({'migration': str(output), 'plan_sha256': verified['plan_sha256'],
        'source_checkpoint': str(cp), 'source_counts': plan['fl_capture_quality_same372_factor']['counter_origin'],
        'target_head': contract['source_git_commit'], 'target_runtime': contract['runtime_content_sha256'],
        'changed_file_count': len(delta), 'new_optimizer_steps': 0, 'new_policy_decisions': 0}, indent=2))


if __name__ == '__main__':
    main()
