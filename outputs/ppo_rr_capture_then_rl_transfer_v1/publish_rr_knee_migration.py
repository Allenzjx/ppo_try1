"""Publish a reviewed same410 hip-then-knee control boundary without learning."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess

from wlr50_clean.ppo.semantic_cli import runtime_contract
from wlr50_clean.ppo.semantic_migration import checkpoint_metadata, file_sha
from wlr50_clean.ppo.semantic_rr_capture_knee_migration import (
    build_rr_capture_knee_migration, publish_rr_capture_knee_checkpoint,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--publish', action='store_true')
    parser.add_argument('--expected-head', required=True)
    args = parser.parse_args()
    out = Path(__file__).resolve().parent
    root = out.parents[1]
    source = out / 'checkpoints/history/checkpoint_rr_capture_feedback_v2_step_000220544_g26db2a1946e8.pt'
    if file_sha(source) != 'a32a4f01afbfbcda06ba59f7e439bb5ab9aa1dbefb3d8e3093e0dd86c827e85e':
        raise ValueError('bound complete 26db source checkpoint changed')
    head = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if args.expected_head != head:
        raise ValueError('publication requires the explicit actual frozen HEAD')
    contract = runtime_contract(expected_head=head, semantic_version='v3',
                                experiment_id='rr_capture_then_rl_transfer_v1')
    metadata = checkpoint_metadata(source)
    old = metadata['runtime_contract']
    review = {p: h for p, h in contract['files'].items() if old['files'].get(p) != h}
    plan = build_rr_capture_knee_migration(source, contract,
        reason=('Sealed 26db natural P01 deterministic video: RR crosses but remains45.7mm airborne '
                'after the actual20deg hip-only search; held knee tracks-45.36deg. Keep that achieved '
                'hip and permit a separate finite positive-knee feedback search without recharging '
                'hip travel/exposure. Both axes are public through explicit combined counter semantics; '
                'contact remains measured. Same410 exact learned state and fresh rollout. '
                'No wheel intervention, source/P10/reward/cap/sigma/physics change.'),
        reviewed_code_sha256=review)
    stem = f'CP220544_RR410_knee_v3_g{head[:12]}'
    target = out / f'checkpoints/history/checkpoint_rr_capture_knee_v3_step_000220544_g{head[:12]}.pt'
    plan_path, receipt_path = out / (stem + '_migration.json'), out / (stem + '_publication.json')
    if not args.publish:
        print(json.dumps({'validated_only': True, 'published': False,
                          'target_checkpoint': str(target), 'target_head': head}))
        return
    for path in (target, target.with_name(target.stem + '_manifest.json'), plan_path, receipt_path):
        if path.exists():
            raise FileExistsError(path)
    with plan_path.open('x', encoding='utf-8') as stream:
        json.dump(plan, stream, indent=2, allow_nan=False)
    receipt = publish_rr_capture_knee_checkpoint(source, contract, plan_path, target)
    final = checkpoint_metadata(target)
    for key in ('actor_parameter_sha256', 'critic_parameter_sha256', 'optimizer_state_sha256',
                'normalizer_state_sha256', 'training_rng_state', 'runner_config',
                'rr_capture_feedback_peak_v2_migration'):
        if final[key] != metadata[key]:
            raise RuntimeError('same410 publication changed preserved state: ' + key)
    receipt.update(checkpoint_sha256=file_sha(target), source_checkpoint_sha256=file_sha(source),
                   target_git_commit=head, control_version='window_peak_hip_then_knee_v3',
                   physical_evaluation='NOT_YET_EVALUATED', no_pointer_promotion=True,
                   exact_same410_learned_state_preserved=True)
    with receipt_path.open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
    print(json.dumps({k: receipt[k] for k in ('checkpoint', 'checkpoint_sha256',
        'target_git_commit', 'control_version', 'exact_same410_learned_state_preserved')}, allow_nan=False))


if __name__ == '__main__':
    main()
