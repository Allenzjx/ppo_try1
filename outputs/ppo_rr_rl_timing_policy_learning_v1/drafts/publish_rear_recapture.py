"""DRAFT publication entrypoint; use only after root freezes reviewed production."""
import argparse
import json
import subprocess
from pathlib import Path
from wlr50_clean.ppo.semantic_cli import runtime_contract
from wlr50_clean.ppo.semantic_training import write_json
from wlr50_clean.ppo.semantic_migration import checkpoint_metadata
from wlr50_clean.ppo.semantic_rear_recapture_migration import (
    build_rear_recapture_migration, publish_rear_recapture_checkpoint, EXPERIMENT)

ROOT = Path(__file__).resolve().parents[3]  # This draft stays in outputs/<namespace>/drafts.

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--source-sha',required=True)
    parser.add_argument('--source-manifest-sha',required=True)
    parser.add_argument('--expected-head',required=True)
    parser.add_argument('--publish',action='store_true')
    args=parser.parse_args()
    head=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    if head != args.expected_head:
        raise ValueError('actual target HEAD differs from explicit publication request')
    contract=runtime_contract(expected_head=head,semantic_version='v3',experiment_id=EXPERIMENT)
    plan=build_rear_recapture_migration(args.source,contract,
        reason='Current RR support/recapture and existing retention-share semantics; same419 full-state identity; fresh rollout zero added learning',
        expected_source_sha256=args.source_sha,expected_manifest_sha256=args.source_manifest_sha)
    metadata=checkpoint_metadata(args.source)
    directory=ROOT/'outputs'/('ppo_'+EXPERIMENT)
    label='rear_recapture_CP'+str(metadata['global_policy_decisions'])+'_g'+head[:12]
    plan_path=directory/(label+'_migration.json')
    target=directory/'checkpoints/history'/('checkpoint_'+label+'.pt')
    receipt_path=directory/(label+'_publication.json')
    if any(path.exists() for path in (plan_path,target,receipt_path)):
        raise FileExistsError('unique publication output already exists; do not overwrite')
    if not args.publish:
        print(json.dumps(dict(validated=True,publish=False,source=str(args.source),target=str(target))))
        return
    write_json(plan_path,plan)
    receipt=publish_rear_recapture_checkpoint(args.source,contract,plan_path,target)
    write_json(receipt_path,receipt)
    print(json.dumps(receipt))

if __name__=='__main__':
    main()

