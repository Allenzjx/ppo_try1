"""OPTIONAL cold publisher draft. Not run; never starts Isaac or performs an update.

Requires the exact sealed CP228864/local3584/PPO7/Adam140/AUX64 identity source.
Root must choose this only after its DET evidence; no live pointer selection.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys
import traceback

ROOT=Path(__file__).resolve().parents[3]
NAME='ppo_rr_capture_first_cp225280_v1'
EXPECTED_COUNTS={'local_policy_decisions':3584,'local_ppo_updates':7,
                 'local_optimizer_steps':140,'auxiliary_updates':64}


def validate_source_metadata(metadata,source_head):
    if (metadata.get('schema')!='wlr50_clean.frozen_prior_rr_capture_checkpoint.v2'
            or metadata.get('runtime_contract',{}).get('source_git_commit')!=source_head
            or metadata.get('rollout_empty') is not True or metadata.get('save_load_round_trip') is not True
            or metadata.get('front_FL_assist') is not True or metadata.get('rear_task_assists') is not False):
        raise ValueError('exact sealed old0ff complete448 source required')
    counts=metadata.get('counts',{})
    if any(type(counts.get(k)) is not int or counts[k]!=v for k,v in EXPECTED_COUNTS.items()):
        raise ValueError('only explicit CP228864 full-block3584/7/140/AUX64 source')
    if metadata.get('local_mean_coordinate_migrations'):
        raise ValueError('source already has a coordinate migration')


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--isaac-stopped',action='store_true')
    parser.add_argument('--expected-head',required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--checkpoint-sha256',required=True)
    parser.add_argument('--manifest-sha256',required=True)
    parser.add_argument('--source-run',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    args=parser.parse_args()
    if not args.isaac_stopped:raise ValueError('No Torch until sole Isaac process exit is confirmed')
    sys.path.insert(0,str(ROOT/'src'))
    from wlr50_clean.ppo import semantic_rr_capture_local as route
    from wlr50_clean.ppo import semantic_rr_mean_coordinates as coordinates
    from wlr50_clean.ppo.semantic_rr_capture_local_aux import validate_local_auxiliary_events
    checkpoint=args.checkpoint.resolve(strict=True)
    sidecar=checkpoint.with_name(checkpoint.stem+'_manifest.json')
    if route.sha(checkpoint)!=args.checkpoint_sha256 or route.sha(sidecar)!=args.manifest_sha256:
        raise ValueError('explicit complete checkpoint/sidecar SHA mismatch')
    metadata=json.loads(sidecar.read_text())
    validate_source_metadata(metadata,coordinates.SOURCE_HEAD)
    if (metadata['checkpoint_sha256']!=args.checkpoint_sha256
            or Path(metadata['checkpoint']).resolve(strict=True)!=checkpoint):
        raise ValueError('sidecar source checkpoint identity mismatch')
    old_runtime=metadata['runtime_contract']
    runtime=route.contract(args.expected_head)
    changes=coordinates.validate_coordinate_runtime_change(old_runtime,runtime)
    ledger=validate_local_auxiliary_events(metadata.get('local_auxiliary_events'),metadata['counts'])
    source_run=args.source_run.resolve(strict=True)
    sealed_path=source_run/'run_manifest.json'
    sealed=json.loads(sealed_path.read_text())
    if (sealed.get('lifecycle')!='COMPLETE' or sealed.get('mode')!='train'
            or Path(metadata['source_run']).resolve(strict=True)!=source_run
            or Path(sealed['result']['checkpoint']).resolve(strict=True)!=checkpoint
            or sealed['result']['checkpoint_sha256']!=args.checkpoint_sha256
            or sealed['result']['manifest_sha256']!=args.manifest_sha256):
        raise ValueError('source block is not sealed against this exact full checkpoint')
    decisions=source_run/'decisions.jsonl';decisions_sha=route.sha(decisions)
    observations=[]
    with decisions.open() as stream:
        for line in stream:
            row=json.loads(line)
            if row.get('kind')=='activated_on_policy':
                if row.get('PPO_credit')!=1 or len(row['observation'])!=448:
                    raise ValueError('source reference observations incompatible')
                observations.append(row['observation'])
    if len(observations)!=512:raise ValueError('this source must be exactly one completed fresh512 block')
    # Algebraic inactive-gate counterfactual only, not a physical prefix/teacher.
    inactive=copy.deepcopy(observations[0]);inactive[439]=0.
    observations.append(inactive)
    run=args.run_dir.resolve()
    if run==ROOT/'runs'/NAME or not run.is_relative_to(ROOT/'runs'/NAME):
        raise ValueError('new isolated run child required')
    run.mkdir(parents=True,exist_ok=False)
    try:
        import torch
        from wlr50_clean.ppo.semantic_training import state_hash
        from wlr50_clean.ppo.rl_library_wrapper import capture_training_rng_state,seed_training_rngs
        route.write(run/'run_manifest.started.json',dict(mode='rr_mean_coordinates_only',runtime_contract=runtime,
            checkpoint=str(checkpoint),checkpoint_sha256=args.checkpoint_sha256,manifest_sha256=args.manifest_sha256,
            source_run=str(source_run),source_run_manifest_sha256=route.sha(sealed_path),source_decisions_sha256=decisions_sha,
            driver={'path':str(Path(__file__).resolve()),'sha256':route.sha(__file__)},no_Isaac=True))
        data=torch.load(checkpoint,map_location='cpu',weights_only=False)
        keys={'actor_state_dict','critic_state_dict','optimizer_state_dict'}
        if (set(data)!=keys|{'infos','iter'} or set(metadata['state_hashes'])!=keys
                or set(metadata)-{'checkpoint','checkpoint_sha256','save_load_round_trip'}!=set(data['infos'])
                or any(metadata[k]!=v for k,v in data['infos'].items())
                or data['iter']!=EXPECTED_COUNTS['local_ppo_updates']
                or any(state_hash(data[k])!=metadata['state_hashes'][k] for k in keys)
                or not data['optimizer_state_dict']['param_groups']
                or any(g['lr']!=metadata['learning_rate'] for g in data['optimizer_state_dict']['param_groups'])):
            raise ValueError('source embedded/sidecar/actual state/step/LR mismatch')
        del data
        seed_training_rngs(1001)
        device=metadata['runner_config']['device']
        runner=route.make_runner(device,1001,coordinate_source_config=metadata['runner_config'],
                                 coordinate_source_head=coordinates.SOURCE_HEAD)
        # Ordinary strict load under exact OLD runtime; not metadata rebind bypass.
        prior,counts=route.load(runner,checkpoint,old_runtime)
        if capture_training_rng_state(seed=1001)!=metadata['training_rng']:
            raise RuntimeError('strict source load did not restore complete RNG')
        receipt=coordinates.migrate_rr_mean_coordinates(runner,observations,
            source_runtime_gain=coordinates.IDENTITY,target_runtime_gain=coordinates.RR10,isaac_stopped=True)
        route.write(run/'coordinate_receipt.json',receipt)
        if receipt['status']!='PASS' or not all(receipt['flags'].values()):
            raise RuntimeError('coordinate checks failed; retain evidence, do not publish')
        if runner.local_auxiliary_events!=ledger or counts!=metadata['counts']:
            raise RuntimeError('migration changed AUX lineage or counters')
        event=dict(schema='wlr50_clean.rr_mean_coordinate_publication.v1',
            source_head=coordinates.SOURCE_HEAD,destination_head=runtime['source_git_commit'],
            source_gain=list(coordinates.IDENTITY),target_gain=list(coordinates.RR10),
            source_checkpoint=str(checkpoint),source_checkpoint_sha256=args.checkpoint_sha256,
            source_manifest_sha256=args.manifest_sha256,source_runtime_sha256=old_runtime['runtime_content_sha256'],
            destination_runtime_sha256=runtime['runtime_content_sha256'],changed_files=changes,
            receipt={'path':str(run/'coordinate_receipt.json'),'sha256':route.sha(run/'coordinate_receipt.json')},
            source_counts=copy.deepcopy(counts),destination_counts=copy.deepcopy(counts),
            new_PPO_updates=0,new_PPO_Adam_steps=0,new_policy_decisions=0,new_AUX_steps=0,
            identity_normalizers_preserved=True,old_AUX_recipe_not_reused=True,optimizer_dynamics_equivalent=False)
        runner.local_mean_coordinate_migrations=[event]
        coordinates.assert_coordinate_binding(runner,runtime)
        if route.contract(args.expected_head)!=runtime or route.sha(decisions)!=decisions_sha:
            raise RuntimeError('runtime/sealed source changed during cold migration')
        pointer=route.save(runner,runtime,prior,counts,source_run=run,publish_pointer=False)
        expected={k:state_hash(v) for k,v in runner.alg.save().items()}
        fresh=route.make_runner(device,1001)
        restored_prior,restored_counts=route.load(fresh,pointer['checkpoint'],runtime)
        if (restored_prior!=prior or restored_counts!=counts or fresh.local_auxiliary_events!=ledger
                or fresh.local_mean_coordinate_migrations!=[event]
                or any(state_hash(fresh.alg.save()[k])!=v for k,v in expected.items())
                or capture_training_rng_state(seed=1001)!=metadata['training_rng']):
            raise RuntimeError('fresh strict reload changed gain10 state/RNG/lineage')
        if route.sha(checkpoint)!=args.checkpoint_sha256 or route.sha(sidecar)!=args.manifest_sha256:
            raise RuntimeError('immutable identity parent changed')
        # Publish pointer only after actual fresh construction+strict reload.
        pointer_path=route.OUTPUT/'checkpoints/checkpoint_last_pointer.json'
        route.write(pointer_path,pointer,replace=pointer_path.exists())
        route.write(run/'run_manifest.json',dict(lifecycle='COMPLETE',mode='rr_mean_coordinates_only',
            result=pointer,new_PPO_updates=0,new_PPO_Adam_steps=0,new_AUX_steps=0,new_policy_decisions=0,
            fresh_runner_strict_reload=True,physical_capability='NOT_EVALUATED',fresh_rollout_required=True))
        print(json.dumps({'lifecycle':'COMPLETE','result':pointer}),flush=True)
    except BaseException:
        route.write(run/'failure.json',dict(lifecycle='FAILED',traceback=traceback.format_exc(),
            no_new_learning_credit=True,physical_success_not_claimed=True))
        raise


if __name__=='__main__':main()
