"""Bound CP220544 replacement; defaults to read-only validation, never physics."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
from wlr50_clean.ppo.semantic_cli import runtime_contract
from wlr50_clean.ppo.semantic_migration import checkpoint_metadata,file_sha
from wlr50_clean.ppo.semantic_rr_capture_migration import (
    build_rr_capture_migration,publish_rr_capture_checkpoint,TARGET_EXPERIMENT,FACTOR_KEY,RECROSS_MODE)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--publish',action='store_true',help='Official original-device save and independent reload; no optimizer')
    parser.add_argument('--expected-head',help='Optional explicit frozen HEAD; current committed HEAD must match')
    args=parser.parse_args()
    out=Path(__file__).resolve().parent;root=out.parents[1]
    source=root/'outputs/ppo_p05_hip_only_continuation_v1/checkpoints/history/checkpoint_step_000220544.pt'
    first=out/'checkpoints/history/checkpoint_rr_capture_step_000220544.pt'
    first_plan=out/'CP220544_to_RR410_migration.json'
    if file_sha(source)!='56239937cd0aaccdc8b7ea36c6266a41b3bed9df67b00f84a06806d2a6fa09b7':
        raise ValueError('original learned CP220544 changed')
    if file_sha(first)!='46ff3a18024ad0fa94a30d393dbfe8bd75913a91da825c63bc524fe0e5547fff':
        raise ValueError('first zero-update RR410 changed')
    head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    if args.expected_head is not None and args.expected_head!=head:raise ValueError('unexpected publication HEAD')
    contract=runtime_contract(expected_head=head,semantic_version='v3',experiment_id=TARGET_EXPERIMENT)
    old=checkpoint_metadata(source)['runtime_contract']
    review={p:h for p,h in contract['files'].items() if old['files'].get(p)!=h and not p.startswith('configs/')}
    plan=build_rr_capture_migration(source,contract,
        reason='Zero-update control replacement only: preserve CP220544 all learned state; same-continuous-AIR P05 recross and finite current-capture/handoff recovery, RR410 assist unchanged.',
        reviewed_code_sha256=review,initial_checkpoint=first,initial_plan_path=first_plan)
    stem=f'CP220544_RR410_recross_v2_g{head[:12]}'
    target=out/f'checkpoints/history/checkpoint_rr_capture_p05_recross_v2_step_000220544_g{head[:12]}.pt'
    plan_path=out/(stem+'_migration.json');receipt_path=out/(stem+'_publication.json')
    if not args.publish:
        print(json.dumps({'validated_only':True,'published':False,'target_checkpoint':str(target),
            'source_checkpoint_sha256':file_sha(source),'first_checkpoint_sha256':file_sha(first),
            'target_head':head,'control_revision':RECROSS_MODE,
            'exact_zero_update_boundary':plan[FACTOR_KEY]['superseded_zero_update_boundary']}))
        return
    for path in (target,target.with_name(target.stem+'_manifest.json'),plan_path,receipt_path):
        if path.exists():raise FileExistsError(path)
    with plan_path.open('x',encoding='utf-8') as stream:json.dump(plan,stream,indent=2,allow_nan=False)
    receipt=publish_rr_capture_checkpoint(source,contract,plan_path,target)
    receipt.update(checkpoint_sha256=file_sha(target),source_checkpoint_sha256=file_sha(source),
        initial_rr_checkpoint_sha256=file_sha(first),target_git_commit=head,control_version=RECROSS_MODE,
        physical_evaluation='NOT_YET_EVALUATED',no_pointer_promotion=True)
    final=checkpoint_metadata(target);initial=checkpoint_metadata(first)
    for key in ('actor_parameter_sha256','critic_parameter_sha256','optimizer_state_sha256',
                'normalizer_state_sha256','training_rng_state','runner_config'):
        if final[key]!=initial[key]:raise RuntimeError('replacement changed identical410 initial state: '+key)
    receipt['complete_initial410_state_preserved']=True
    with receipt_path.open('x',encoding='utf-8') as stream:json.dump(receipt,stream,indent=2,allow_nan=False)
    print(json.dumps(receipt))


if __name__=='__main__':main()
