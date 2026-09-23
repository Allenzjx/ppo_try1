"""Zero-update official same410 v4 publication; explicit immutable source/HEAD."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import subprocess

from wlr50_clean.ppo.semantic_cli import runtime_contract
from wlr50_clean.ppo.semantic_migration import checkpoint_metadata, file_sha
from wlr50_clean.ppo.semantic_rr_carry_handoff_migration import (
    SOURCE_HEAD, SOURCE_CHECKPOINT_SHA256, MIGRATION, COUNTERS, preserved_keys,
    build_rr_carry_handoff_migration, validate_rr_carry_handoff_migration,
)
from wlr50_clean.ppo.semantic_rr_capture_migration import _shape_env
from wlr50_clean.ppo.semantic_rr_capture_profile import RR_CAPTURE_POLICY, RR_CAPTURE_OBSERVATION_LAYOUT
from wlr50_clean.ppo.semantic_training import construct_semantic_runner, load_semantic_checkpoint, save_semantic_checkpoint


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--expected-head', required=True)
    parser.add_argument('--publish', action='store_true')
    args=parser.parse_args()
    out=Path(__file__).resolve().parent; root=out.parents[1]
    head=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    if head != args.expected_head:
        raise ValueError('explicit target HEAD differs')
    source=out/'checkpoints/history/checkpoint_step_000221184.pt'
    metadata=checkpoint_metadata(source)
    contract=runtime_contract(expected_head=head,semantic_version='v3',experiment_id='rr_capture_then_rl_transfer_v1')
    old=metadata['runtime_contract']
    review={p:h for p,h in contract['files'].items() if old['files'].get(p)!=h}
    plan=build_rr_carry_handoff_migration(source,contract,
        expected_source_sha256=SOURCE_CHECKPOINT_SHA256,expected_source_head=SOURCE_HEAD,
        expected_target_head=head,reviewed_code_sha256=review,
        reason=('Sealed e24a real block640/5: hip20 plus knee20 search remains AIR34.788mm '
                'and landing depth retreats; source stop is valid but policy FL reverse persists. '
                'Add declared current-support, geometry-tapered nonnegative forward floor after '
                'committed P09 source stop with observed previous-FINAL slew; preserve raw Gaussian. '
                'Add finite real TOP/HOLD confirmation and next normal decision handoff, never '
                'historical placed AIR support. Same410 exact learned state; new rollout.'))
    stem=f'CP221184_RR410_carry_handoff_v4_g{head[:12]}'
    target=out/f'checkpoints/history/checkpoint_rr_carry_handoff_v4_step_000221184_g{head[:12]}.pt'
    plan_path=out/(stem+'_migration.json'); receipt_path=out/(stem+'_publication.json')
    if not args.publish:
        print(json.dumps(dict(validated_only=True,target_checkpoint=str(target),target_head=head)))
        return
    for path in (target,target.with_name(target.stem+'_manifest.json'),plan_path,receipt_path):
        if path.exists(): raise FileExistsError(path)
    with plan_path.open('x',encoding='utf-8') as f: json.dump(plan,f,indent=2,allow_nan=False)
    verified=validate_rr_carry_handoff_migration(source,contract,plan_path)
    device=metadata['runner_config']['device']
    def make():
        return construct_semantic_runner(_shape_env(410,device),seed=metadata['seed'],device=device,
            policy_version=RR_CAPTURE_POLICY,observation_layout=RR_CAPTURE_OBSERVATION_LAYOUT,
            initialize_actor=False)[0]
    runner=make()
    infos=load_semantic_checkpoint(runner,source,contract=contract,seed=metadata['seed'],migration=verified)
    if infos.get(MIGRATION)!=verified: raise RuntimeError('new receipt missing')
    infos.update(runtime_contract=copy.deepcopy(contract),resume_migration=verified,old_rollout_inherited=False,
        migration_publication_context='zero_update_same410_support_forward_and_contact_handoff_no_environment')
    saved,manifest=save_semantic_checkpoint(runner,target,infos)
    loaded=load_semantic_checkpoint(make(),saved,contract=contract,seed=metadata['seed'])
    if any(loaded[k]!=metadata[k] for k in preserved_keys(metadata)) or loaded[MIGRATION]!=verified:
        raise RuntimeError('official fresh reload changed learned state or lineage')
    receipt=dict(checkpoint=str(saved),manifest=str(manifest),checkpoint_sha256=file_sha(saved),
        source_checkpoint=str(source),source_checkpoint_sha256=file_sha(source),target_git_commit=head,
        save_load_round_trip=True,exact_same410_learned_state_preserved=True,
        **{k:loaded[k] for k in COUNTERS},rr_capture_transfer_branch_counts=loaded['rr_capture_transfer_branch_counts'],
        migration_added_policy_decisions=0,migration_added_ppo_updates=0,migration_added_optimizer_steps=0,
        migration_added_auxiliary_updates=0,old_rollout_inherited=False,no_pointer_promotion=True,
        actual_effective_learning_rate=loaded['optimizer_learning_rate'],
        physical_evaluation='NOT_YET_EVALUATED',control_version='v4_support_forward_contact_handoff')
    with receipt_path.open('x',encoding='utf-8') as f: json.dump(receipt,f,indent=2,allow_nan=False)
    print(json.dumps(receipt,allow_nan=False))


if __name__=='__main__': main()
