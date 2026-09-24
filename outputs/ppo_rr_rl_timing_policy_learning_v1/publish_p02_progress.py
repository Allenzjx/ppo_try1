"""Explicit unique same-branch publication; no pointer promotion or learning."""
import argparse
import json
import subprocess
from pathlib import Path
from wlr50_clean.ppo.semantic_cli import runtime_contract
from wlr50_clean.ppo.semantic_training import write_json
from wlr50_clean.ppo.semantic_p02_progress_migration import (
    build_p02_progress_migration,publish_p02_progress_checkpoint,
    EXPERIMENT,BRANCH_NAME,REVISION_ORIGIN,SOURCE_SHA,SOURCE_MANIFEST_SHA)

ROOT=Path(__file__).resolve().parents[2]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-head',required=True)
    parser.add_argument('--publish',action='store_true')
    args=parser.parse_args()
    head=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    if head!=args.expected_head:
        raise ValueError('explicit frozen target HEAD differs from actual HEAD')
    contract=runtime_contract(expected_head=head,semantic_version='v3',experiment_id=EXPERIMENT)
    out=ROOT/'outputs'/('ppo_'+EXPERIMENT)
    branch=out/'branches'/BRANCH_NAME
    source=branch/'checkpoints/history/checkpoint_step_000221696.pt'
    label='p02_progress_CP221696_g'+head[:12]
    plan_path=out/(label+'_migration.json')
    target=branch/'checkpoints/history'/('checkpoint_'+label+'.pt')
    receipt_path=out/(label+'_publication.json')
    if any(path.exists() for path in (plan_path,target,receipt_path,target.with_name(target.stem+'_manifest.json'))):
        raise FileExistsError('unique publication already exists; no overwrite')
    plan=build_p02_progress_migration(source,contract,
        reason='Measured P02 progress continuation; preserve learned221696 and full optimizer; append3 observable state columns with zero weights/moments; unchanged sigma419/history/rear control; longer natural learner course and fresh rollout')
    if not args.publish:
        print(json.dumps(dict(validated=True,publish=False,source=str(source),source_sha256=SOURCE_SHA,
            source_manifest_sha256=SOURCE_MANIFEST_SHA,source_counters=REVISION_ORIGIN,target=str(target))))
        return
    # Source-device CUDA visibility must match the actual source RNG topology.
    write_json(plan_path,plan)
    receipt=publish_p02_progress_checkpoint(source,contract,plan_path,target)
    write_json(receipt_path,receipt)
    print(json.dumps(receipt))


if __name__=='__main__':
    main()
