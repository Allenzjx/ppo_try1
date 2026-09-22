"""Opt-in offline auxiliary candidate. Default is CPU inspection, not training.

No Isaac import, no physics calls, no fake on-policy rollout, no latest pointer.
Execution must be an explicitly reviewed separate process on the saved device.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from tensordict import TensorDict
import finite_auxiliary_mean as a
from reviewed_data import load_reviewed_selection,sha
from wlr50_clean.ppo import semantic_cli as cli,semantic_migration as migration
from wlr50_clean.ppo.semantic_policy_distribution import TASK_CONDITIONED_HIP_WHEEL_POLICY

OUT=a.ROOT/'outputs/ppo_task_conditioned_hip_wheel_v1'


def helpers():
    return {name:sha(a.HERE/name) for name in ('finite_auxiliary_mean.py','reviewed_data.py','auxiliary_cli.py')}


def output_path(path):
    path=Path(path).resolve()
    a.require(path.is_relative_to(OUT.resolve()) and not path.exists(), 'use a new path inside this experiment output root')
    return path


def verify_data_compatibility(metadata,receipt,contract):
    a.require(metadata['runtime_contract']==contract and contract['experiment_id']==a.EXPERIMENT
        and metadata['policy_contract']==receipt['diagnostic_policy_contract']
        and metadata['policy_contract']['version']==TASK_CONDITIONED_HIP_WHEEL_POLICY,
        'latest checkpoint, current runtime and diagnostic policy do not match')
    old=receipt['diagnostic_runtime_contract']
    if old==contract:return {'mode':'exact_same_runtime_and_policy'}
    extension=metadata.get('training_quantity_budget_extension',{})
    path=Path(extension['plan_path']).resolve(strict=True)
    supplied=json.loads(path.read_text(encoding='utf-8'))
    verified=migration.validate_migration_plan(Path(supplied['source_checkpoint']),contract,path)
    a.require(extension['plan_sha256']==verified['plan_sha256']
        and extension['factor']==verified['training_quantity_budget_factor']
        and extension['source_contract_sha256']==verified['source_contract_sha256']==migration.digest(old)
        and extension['target_contract_sha256']==verified['target_contract_sha256']==migration.digest(contract)
        and extension['source_checkpoint_sha256']==verified['source_checkpoint_sha256'],
        'diagnostic data crosses an unverified non-quantity runtime boundary')
    f=extension['factor']
    a.require(f['training_quantity_only'] is True and f['kernel_changed'] is False
        and f['new_mdp'] is False and f['observation_semantics_changed']==[],
        'data compatibility cannot waive MDP/observation/kernel changes')
    return {'mode':'strict_revalidated_quantity_only_same_MDP',
        'plan_path':str(path),'plan_sha256':verified['plan_sha256']}


def observation_runner(observation,*,seed,device):
    class ObservationOnly:
        num_envs,num_actions=1,12
        cfg={'evaluation':True,'semantic_version':'v3'}
        def get_observations(self):
            x=torch.tensor([observation],dtype=torch.float32,device=device)
            return TensorDict({'policy':x,'critic':x.clone()},batch_size=[1],device=device)
    runner,_=a.training.construct_semantic_runner(ObservationOnly(),seed=seed,device=device,
        initialize_actor=False,policy_version=TASK_CONDITIONED_HIP_WHEEL_POLICY,
        observation_layout='diagonal_transfer_state_v1')
    return runner


def cpu_inspection(checkpoint,metadata,train,target,holdout):
    """Honest read-only actor forward; NOT a device-relocated official resume."""
    a.require(not torch.cuda.is_available(), 'inspection requires CPU-only visibility; do not claim GPU resource')
    rng=a.training.capture_training_rng_state(seed=metadata['seed'])
    try:
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        expected={k:v for k,v in metadata.items() if k not in ('checkpoint_path','checkpoint_sha256','save_load_round_trip')}
        a.require(payload['infos']==expected and a.training.state_hash(payload['optimizer_state_dict'])
            ==metadata['optimizer_state_sha256'], 'saved checkpoint payload/optimizer integrity mismatch')
        runner=observation_runner(train[0],seed=metadata['seed'],device='cpu')
        runner.alg.actor.load_state_dict(payload['actor_state_dict'],strict=True)
        a.require(a.training.parameter_hash(runner.alg.actor)==metadata['actor_parameter_sha256']
            and a.training._runner_policy_contract(runner)==metadata['policy_contract'], 'CPU read-only actor mismatch')
        runner.alg.actor.eval()
        result=a.inspect(runner.alg.actor,a.tensors(train,device='cpu'),target,a.tensors(holdout,device='cpu'))
        result['load_semantics']='read-only CPU actor-state copy; not official device-relocated resume; no optimizer loaded or stepped'
        return result
    finally:a.training.restore_training_rng_state(rng,expected_seed=metadata['seed'])


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--expected-head',required=True)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--execute-aux',action='store_true')
    p.add_argument('--inspection-receipt',type=Path)
    p.add_argument('--aux-checkpoint',type=Path)
    args=p.parse_args(argv)
    report_path=output_path(args.report)
    checkpoint=args.checkpoint.resolve(strict=True);metadata=migration.checkpoint_metadata(checkpoint)
    contract=cli.runtime_contract(expected_head=args.expected_head,semantic_version='v3',experiment_id=a.EXPERIMENT)
    train,target,holdout,receipt=load_reviewed_selection(args.selection)
    compatibility=verify_data_compatibility(metadata,receipt,contract)
    source={'path':str(checkpoint),'sha256':metadata['checkpoint_sha256'],
        'manifest_sha256':sha(checkpoint.with_name(checkpoint.stem+'_manifest.json'))}
    binding={'source_checkpoint':source,'selection_sha256':receipt['selection_sha256'],
        'helper_sha256':helpers(),'runtime_contract_sha256':migration.digest(contract)}
    result={'schema':'wlr50_clean.finite_auxiliary_candidate_receipt.v1','binding':binding,
        'data_receipt':receipt,'data_runtime_compatibility':compatibility,
        'PPO_decisions_added':0,'PPO_updates_added':0,'PPO_optimizer_steps_added':0,
        'original_PPO_counters':{k:metadata[k] for k in ('global_policy_decisions','ppo_updates','optimizer_steps')},
        'teacher_deployed':False,'capture_claimed':False,'physical_success_claimed':False}
    if not args.execute_aux:
        a.require(args.aux_checkpoint is None and args.inspection_receipt is None,'inspection does not save checkpoints or accept an execution receipt')
        result.update(mode='read_only_current_checkpoint_on_real_saved_states',auxiliary_updates=0,
            inspection=cpu_inspection(checkpoint,metadata,train,target,holdout),automatic_aux_enabled=False)
    else:
        a.require(args.inspection_receipt is not None and args.aux_checkpoint is not None,
            'explicit execution requires an already reviewed same-source inspection and unique auxiliary checkpoint')
        prior=json.loads(args.inspection_receipt.read_text(encoding='utf-8'))
        a.require(prior.get('mode')=='read_only_current_checkpoint_on_real_saved_states'
            and prior.get('binding')==binding and prior.get('auxiliary_updates')==0,
            'inspection is stale, not this checkpoint/data/helper, or itself optimized')
        destination=output_path(args.aux_checkpoint)
        a.require(destination.name.startswith('checkpoint_aux_flmean_') and destination.suffix=='.pt', 'explicit aux checkpoint filename required')
        # No device override: do not forge source CUDA metadata for a CPU resume.
        device=metadata['runner_config']['device'];seed=metadata['seed']
        a.training.seed_training_rngs(seed)
        runner=observation_runner(train[0],seed=seed,device=device)
        infos=a.training.load_semantic_checkpoint(runner,checkpoint,contract=contract,seed=seed)
        report=a.fit_mean_row(runner,a.tensors(train,device=device),target,a.tensors(holdout,device=device),authorized=True)
        result.update(mode='explicit_finite_auxiliary_not_PPO',auxiliary_updates=report['accepted_auxiliary_updates'],fit=report,
            reviewed_inspection={'path':str(args.inspection_receipt.resolve()),'sha256':sha(args.inspection_receipt)})
        if report['accepted_auxiliary_updates']:
            receipt['helper_bundle_sha256']=helpers()
            pair=a.save_auxiliary_checkpoint(runner,destination,infos,report=report,data_receipt=receipt,source_checkpoint=source)
            result['auxiliary_checkpoint']={'path':str(pair[0]),'manifest':str(pair[1]),'sha256':sha(pair[0])}
        else:result['auxiliary_checkpoint']=None
    a.training.write_json(report_path,result)
    print(json.dumps({'report':str(report_path),'mode':result['mode'],'auxiliary_updates':result['auxiliary_updates']}))
    return result


if __name__=='__main__':main()
