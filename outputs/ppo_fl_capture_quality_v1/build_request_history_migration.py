"""Build a real committed-runtime migration only; no simulator or optimizer."""
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
    cp, output = args.checkpoint.resolve(strict=True), args.output.resolve()
    if output.exists() or not output.is_relative_to(Path(__file__).resolve().parent):
        raise ValueError('migration output must be new and inside this output directory')
    metadata = m.checkpoint_metadata(cp)
    contract = cli.runtime_contract(expected_head=args.expected_head,
        semantic_version='v3', experiment_id='fl_capture_quality_v1')
    old = metadata['runtime_contract']
    delta = sorted(path for path in old['files'].keys() | contract['files'].keys()
                   if old['files'].get(path) != contract['files'].get(path))
    plan = m.build_migration_plan(cp, contract, allowed_changed_files=delta,
        reason='Represent carried filtered physical REQUEST under enlarged phase caps at the observed entry decision; preserve learned state and collect fresh on-policy data',
        request_history_kernel_review={
            'reason': 'Reviewed six-file conditional-mean boundary; all six configs, nominal, actuator chain, hard acceptance, reward, sigma and temperature unchanged. First-entry correction does not remove a harmful later learned base mean.',
            'reviewed_code_sha256': {path: contract['files'][path]
                                    for path in sorted(m.REQUEST_HISTORY_KERNEL_FILES)}})
    t.write_json(output, plan)
    verified = m.validate_migration_plan(cp, contract, output)
    print(json.dumps({'migration': str(output), 'plan_sha256': verified['plan_sha256'],
        'source_checkpoint': str(cp),
        'counter_origin': plan['request_history_kernel_factor']['counter_origin'],
        'target_head': contract['source_git_commit'],
        'target_runtime': contract['runtime_content_sha256'], 'changed_files': delta,
        'new_policy_decisions': 0, 'new_optimizer_steps': 0}, indent=2))


if __name__ == '__main__':
    main()
