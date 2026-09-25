"""Publish/reload the explicit full-state225280->439 boundary; zero training credit."""
import argparse
import json
from pathlib import Path
from wlr50_clean.ppo.semantic_cli import runtime_contract
from wlr50_clean.ppo.semantic_rear_owner_migration import (
    EXPERIMENT, BRANCH_NAME, build_rear_owner_migration, publish_rear_owner_checkpoint)
from wlr50_clean.ppo.semantic_training import write_json

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--expected-head', required=True)
    p.add_argument('--publish', action='store_true')
    args = p.parse_args()
    contract = runtime_contract(expected_head=args.expected_head, semantic_version='v3', experiment_id=EXPERIMENT)
    branch = ROOT/'outputs'/('ppo_'+EXPERIMENT)/'branches'/BRANCH_NAME
    source = branch/'checkpoints/history/checkpoint_step_000225280.pt'
    label = 'rear_owner_CP225280_g'+args.expected_head[:12]
    output = Path(__file__).resolve().parent
    plan_path = output/(label+'_migration.json')
    target = branch/'checkpoints/history'/('checkpoint_'+label+'.pt')
    receipt_path = output/(label+'_publication.json')
    if any(path.exists() for path in (plan_path,target,target.with_name(target.stem+'_manifest.json'),receipt_path)):
        raise FileExistsError('preserve previous publication; choose a new frozen version')
    plan = build_rear_owner_migration(source,contract,reason=(
        'Latest learned CP225280; explicit439 owner memory; same-attempt RL EDGE task; '
        'cancel inapplicable issued dependent servo goals while preserving policy recovery/stops; '
        'bounded task proxies and rear-heavy continuous curriculum; full compatible state, zero new credit'))
    if not args.publish:
        print(json.dumps(dict(validated=True,source=str(source),target=str(target),
                             changed_paths=sorted(plan['changed_file_hashes']))))
        return
    write_json(plan_path,plan)
    receipt = publish_rear_owner_checkpoint(source,contract,plan_path,target)
    write_json(receipt_path,receipt)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
