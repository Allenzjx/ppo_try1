"""Explicit zero-credit CP225280 front-preserving sibling publication.

Run only at the verified normal boundary with no Isaac process active.
No old checkpoint, pointer or runtime file is changed by this entry point.
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--expected-head', required=True)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    from wlr50_clean.ppo.semantic_cli import runtime_contract
    from wlr50_clean.ppo.semantic_front_preservation import (
        SOURCE_SHA, SOURCE_MANIFEST_SHA, SOURCE_STEM, SOURCE_BRANCH, BRANCH_NAME,
        EXPERIMENT, build_front_preservation_plan, publish_front_preservation_checkpoint)
    from wlr50_clean.ppo.semantic_training import write_json
    root = Path(__file__).resolve().parents[3]
    output = Path(__file__).resolve().parent
    source = (root / 'outputs' / ('ppo_' + EXPERIMENT) / 'branches' / SOURCE_BRANCH
              / 'checkpoints/history' / (SOURCE_STEM + '.pt'))
    destination = (root / 'outputs' / ('ppo_' + EXPERIMENT) / 'branches' / BRANCH_NAME
                   / 'checkpoints/history' / ('checkpoint_front_preserved_CP225280_g' + args.expected_head[:12] + '.pt'))
    plan_path = output / ('front_preserved_CP225280_g' + args.expected_head[:12] + '_plan.json')
    receipt_path = output / ('front_preserved_CP225280_g' + args.expected_head[:12] + '_publication.json')
    if any(p.exists() for p in (plan_path, receipt_path, destination,
                               destination.with_name(destination.stem + '_manifest.json'))):
        raise FileExistsError('Preserve existing publications; do not overwrite or republish blindly')
    contract = runtime_contract(expected_head=args.expected_head, semantic_version='v3', experiment_id=EXPERIMENT)
    spec = json.loads((output / 'front_replay_regularizer_config.json').read_text(encoding='utf-8'))
    plan = build_front_preservation_plan(source, contract,
        expected_source_sha256=SOURCE_SHA, expected_manifest_sha256=SOURCE_MANIFEST_SHA,
        reason=('User urgent steer: restore the compatible CP225280 front baseline on an independent branch; '
                'retain declared rear-owner/edge handling, same N/HISTORY/FL assist, existing72e rear reward; '
                'front-only real-state reference Gaussian KL in actual PPO minibatches; no rear realtime teacher'),
        front_replay_spec=spec)
    if not args.publish:
        print(json.dumps(dict(validated=True, source=str(source), target=str(destination),
            changed_paths=sorted(plan['changed_file_hashes']), zero_new_learning_credit=True)))
        return
    write_json(plan_path, plan)
    receipt = publish_front_preservation_checkpoint(source, contract, plan_path, destination)
    write_json(receipt_path, receipt)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
