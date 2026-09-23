"""Default CPU zero-step inspection; finite AUX needs separate explicit approval.

Source checkpoint/hash/manifest are mandatory. Later sealed checkpoints in the
same frozen v9 branch are allowed; never silently fall back to CP221952.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import torch
from tensordict import TensorDict
import front_retention as kernel
from reviewed_data_410 import ROOT,read,sha,digest,require,load_reviewed_data
from wlr50_clean.ppo import semantic_cli,semantic_migration,semantic_training as training
from wlr50_clean.ppo.semantic_rr_capture_profile import RR_CAPTURE_POLICY,RR_CAPTURE_OBSERVATION_LAYOUT
from wlr50_clean.ppo.semantic_rr_postcapture_wheel_migration import validate_v9_branch_receipt,SOURCE_BRANCH,SOURCE_COUNTS

HERE=Path(__file__).resolve().parent
OUT=ROOT/'outputs/ppo_rr_capture_then_rl_transfer_v1'
BRANCH=OUT/'branches'/SOURCE_BRANCH
FROZEN_HEAD='3edda51732f4ff85717fcb3491bf5c8c5766474d'
COUNTERS=('global_policy_decisions','ppo_updates','optimizer_steps')
LEDGER_KEY='front_retention_auxiliary'
LEDGER_SCHEMA='wlr50_clean.RR410_front_retention_auxiliary.v1'
SCHEMA='wlr50_clean.RR410_front_retention_receipt.v1'
PARAMETERS=['actor.mlp.0.weight[:,0:2]']


def helpers():
    return {name:sha(HERE/name) for name in ('front_retention.py','reviewed_data_410.py','retention_cli.py')}


def output_path(path,*,checkpoint=False):
    path=Path(path).resolve()
    if checkpoint:
        require(path.parent==(BRANCH/'checkpoints/history').resolve()
            and path.name.startswith('checkpoint_aux_frontretention410_') and path.suffix=='.pt'
            and not path.with_name(path.stem+'_manifest.json').exists(),'AUX needs unique immutable same-branch history destination')
    else:
        require(path.is_relative_to(HERE.resolve()),'new helper reports stay in front_retention_410_v1')
    require(not path.exists(),'output already exists; no overwrite permitted')
    return path


def validate_source(checkpoint,metadata,contract,*,expected_sha,expected_manifest_sha):
    checkpoint=Path(checkpoint).resolve(strict=True)
    require(checkpoint.parent==(BRANCH/'checkpoints/history').resolve(),'source must be explicit immutable same-branch history')
    require(sha(checkpoint)==expected_sha==metadata['checkpoint_sha256']
        and sha(checkpoint.with_name(checkpoint.stem+'_manifest.json'))==expected_manifest_sha,
        'explicit source/checkpoint sidecar binding mismatch')
    require(metadata['runtime_contract']==contract and contract['source_git_commit']==FROZEN_HEAD
        and contract['experiment_id']=='rr_capture_then_rl_transfer_v1'
        and metadata['policy_contract']['version']==RR_CAPTURE_POLICY
        and metadata.get('save_load_round_trip') is True
        and all(type(metadata[k]) is int and metadata[k]>=SOURCE_COUNTS[k] for k in COUNTERS),
        'requires frozen v9 actual branch full-update checkpoint, not old count rollback')
    origin=metadata['rr_capture_transfer_branch']['counter_origin']
    require(metadata['rr_capture_transfer_branch_counts']=={k:metadata[k]-origin[k] for k in COUNTERS}
        and metadata['global_policy_decisions']%128==0 and metadata['optimizer_steps']==20*metadata['ppo_updates'],
        'source must retain exact completed-update branch/PPO/Adam counts')
    validate_v9_branch_receipt(metadata,contract,metadata['checkpoint_output_routing'])
    return {'path':str(checkpoint),'sha256':expected_sha,'manifest_sha256':expected_manifest_sha,
        'counters':{k:metadata[k] for k in COUNTERS},'runtime_contract_sha256':digest(contract),
        'checkpoint_output_routing_sha256':digest(metadata['checkpoint_output_routing'])}


def observation_runner(observation,metadata,*,device):
    class ObservationOnly:
        num_envs,num_actions=1,12
        cfg={'evaluation':True,'semantic_version':'v3'}
        def get_observations(self):
            x=torch.as_tensor(observation,dtype=torch.float32,device=device).reshape(1,410)
            return TensorDict({'policy':x,'critic':x.clone()},batch_size=[1],device=device)
    runner,_=training.construct_semantic_runner(ObservationOnly(),seed=metadata['seed'],device=device,
        initialize_actor=False,policy_version=RR_CAPTURE_POLICY,observation_layout=RR_CAPTURE_OBSERVATION_LAYOUT)
    expected=deepcopy(metadata['runner_config']);expected['device']=device
    require(runner._semantic_runner_config==expected and training._runner_policy_contract(runner)==metadata['policy_contract'],
        'current410 runner differs beyond the explicitly stated inspection device')
    return runner


def fit_arguments(data,device):
    return (kernel.tensors(data['train_observations'],device=device),data['train_raw_targets'].to(device),
        kernel.tensors(data['validation_observations'],device=device),data['validation_raw_targets'].to(device),
        kernel.tensors(data['invariance_observations'],device=device))


def cpu_inspection(checkpoint,metadata,data):
    require(not torch.cuda.is_available(),'inspection requires CUDA_VISIBLE_DEVICES=-1')
    rng=training.capture_training_rng_state(seed=metadata['seed'])
    try:
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        expected={k:v for k,v in metadata.items() if k not in ('checkpoint_path','checkpoint_sha256','save_load_round_trip')}
        require(payload['infos']==expected and training.state_hash(payload['optimizer_state_dict'])==metadata['optimizer_state_sha256'],
            'actual source embedded metadata or full Adam integrity mismatch')
        runner=observation_runner(data['train_observations'][0],metadata,device='cpu')
        runner.alg.actor.load_state_dict(payload['actor_state_dict'],strict=True)
        require(training.parameter_hash(runner.alg.actor)==metadata['actor_parameter_sha256'],'current source actor-copy mismatch')
        runner.alg.actor.eval()
        report=kernel.inspect(runner.alg.actor,*fit_arguments(data,'cpu'))
        require(training.parameter_hash(runner.alg.actor)==metadata['actor_parameter_sha256'],'inspection mutated actor')
        directions={}
        for name in ('train','validation'):
            tangent=torch.tensor(report['frozen_first_gradient_response'][name]['requested_residual_per_unit_lr_full12'])[:,8:]
            directions[name]={'wheel_order':['FL','FR','RL','RR'],
                'mean_REQUEST_delta_per_unit_learning_rate_rad_s':tangent.mean(0).tolist(),
                'minimum_REQUEST_delta_per_unit_learning_rate_rad_s':tangent.amin(0).tolist(),
                'maximum_REQUEST_delta_per_unit_learning_rate_rad_s':tangent.amax(0).tolist(),
                'negative_direction_sample_counts':(tangent<0).sum(0).tolist(),
                'positive_direction_sample_counts':(tangent>0).sum(0).tolist(),
                'meaning':'initial negative-loss-gradient tangent only; not finite fit or actuator/traction prediction'}
        return {**report,'load_semantics':'CPU exact actor-state copy, not device-relocated official resume',
            'optimizer_loaded_or_stepped':False,'checkpoint_written':False,
            'four_wheel_initial_REQUEST_direction':directions,
            'JVP_is_initial_direction_not_finite_step_or_closed_loop_prediction':True}
    finally:
        training.restore_training_rng_state(rng,expected_seed=metadata['seed'])


def append_ledger(infos,*,report,report_binding,data_receipt,source_binding,helper_sha256,budget):
    accepted,attempted=report['accepted_auxiliary_updates'],report['attempted_auxiliary_optimizer_steps']
    require(type(accepted) is int and type(attempted) is int and 0<accepted<=attempted<=32,'only accepted finite AUX may be saved')
    result=deepcopy(infos);branch=result['rr_capture_transfer_branch']
    require(branch['counter_origin']==dict(global_policy_decisions=220544,ppo_updates=1688,optimizer_steps=33760),
        'original RR branch origin changed')
    ledger=deepcopy(branch.get(LEDGER_KEY,{'schema':LEDGER_SCHEMA,'events':[]}))
    require(ledger['schema']==LEDGER_SCHEMA and isinstance(ledger['events'],list)
        and [e['event_index'] for e in ledger['events']]==list(range(1,len(ledger['events'])+1)),'retention ledger sequence differs')
    event={'event_index':len(ledger['events'])+1,'kind':'RR410_finite_P01_P02_raw_action_phase_columns_not_PPO',
        'source_checkpoint':source_binding,'helpers_sha256':helper_sha256,'data_receipt_sha256':data_receipt['receipt_content_sha256'],
        'fit_report':report_binding,'budget':budget,'accepted_auxiliary_updates':accepted,
        'attempted_auxiliary_optimizer_steps':attempted,'optimized_parameters':PARAMETERS,'optimized_scalar_count':512,
        'phase_scope':['P01','P02'],'mean_and_log_sigma_may_change':True,
        'same_input_P03_P13_Gaussian_unchanged':True,'same_future_trajectory_claimed':False,
        'PPO_counters_unchanged':{k:infos[k] for k in COUNTERS},
        'PPO_decisions_added':0,'PPO_updates_added':0,'PPO_optimizer_steps_added':0,
        'teacher_deployed':False,'physical_success_claimed':False,
        'old_front_AUX96_96_RR7_8_and_legacy7_8_ledger_unchanged':True}
    ledger['events'].append(json.loads(json.dumps(event,allow_nan=False)))
    ledger['accepted_auxiliary_updates_total']=sum(e['accepted_auxiliary_updates'] for e in ledger['events'])
    ledger['attempted_auxiliary_optimizer_steps_total']=sum(e['attempted_auxiliary_optimizer_steps'] for e in ledger['events'])
    branch[LEDGER_KEY]=ledger
    restored=deepcopy(result);restored['rr_capture_transfer_branch']=deepcopy(infos['rr_capture_transfer_branch'])
    require(restored==infos,'new ledger altered protected historical metadata')
    return result


def save_auxiliary_checkpoint(runner,path,infos,*,report,report_binding,data,source_binding,budget):
    path=output_path(path,checkpoint=True)
    require(runner.alg.storage.step==0 and runner.alg.transition.actions is None,'save requires fresh empty rollout')
    require(infos['actor_parameter_sha256']==report['actor_parameter_sha256_before']
        and training.parameter_hash(runner.alg.actor)==report['actor_parameter_sha256_after'],'accepted actor hash mismatch')
    for key,actual in (('critic_parameter_sha256',training.parameter_hash(runner.alg.critic)),
        ('optimizer_state_sha256',training.state_hash(runner.alg.optimizer.state_dict())),
        ('normalizer_state_sha256',training.state_hash(training._normalizers(runner)))):
        require(infos[key]==actual,'AUX altered protected state: '+key)
    require(infos['optimizer_learning_rate']==training.optimizer_learning_rate(runner)
        and infos['runner_config']==runner._semantic_runner_config
        and infos['training_rng_state']==training.capture_training_rng_state(seed=infos['seed']),
        'AUX changed source device/config/effective LR/full RNG')
    payload=append_ledger(infos,report=report,report_binding=report_binding,data_receipt=data['receipt'],
        source_binding=source_binding,helper_sha256=helpers(),budget=budget)
    pointer_paths=[base/'checkpoints'/name for base in (OUT,BRANCH) for name in ('checkpoint_last.pt','resume_state.json')]
    before={str(p):sha(p) if p.exists() else None for p in pointer_paths}
    saved,manifest=training.save_semantic_checkpoint(runner,path,payload)
    metadata=semantic_migration.checkpoint_metadata(saved)
    fresh=observation_runner(data['train_observations'][0],metadata,device=metadata['runner_config']['device'])
    loaded=training.load_semantic_checkpoint(fresh,saved,contract=metadata['runtime_contract'],seed=metadata['seed'])
    require(all(loaded[k]==infos[k] for k in COUNTERS),'AUX invented PPO credit')
    require(loaded['rr_capture_transfer_branch']==payload['rr_capture_transfer_branch']
        and fresh.alg.storage.step==0 and fresh.alg.transition.actions is None,'independent ledger/empty-rollout reload failed')
    for key in infos:
        if key.endswith(('_branch','_migration','_branch_counts')) and key!='rr_capture_transfer_branch':
            require(loaded[key]==infos[key],'old ancestry/AUX changed: '+key)
    require(loaded['checkpoint_output_routing']==infos['checkpoint_output_routing']
        and training.parameter_hash(fresh.alg.actor)==report['actor_parameter_sha256_after']
        and training.state_hash(fresh.alg.optimizer.state_dict())==infos['optimizer_state_sha256']
        and training.capture_training_rng_state(seed=infos['seed'])==infos['training_rng_state'],'official independent state reload failed')
    validate_v9_branch_receipt(loaded,loaded['runtime_contract'],loaded['checkpoint_output_routing'])
    require(before=={str(p):sha(p) if p.exists() else None for p in pointer_paths},'AUX must not publish any latest pointer')
    return {'path':str(saved),'sha256':sha(saved),'manifest':str(manifest),'independent_official_reload_verified':True,
        'latest_pointer_published':False,'subsequent_real_PPO_ledger_carry_not_yet_verified':True}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--expected-source-sha256',required=True)
    p.add_argument('--expected-manifest-sha256',required=True)
    p.add_argument('--expected-head',required=True)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--execute-aux',action='store_true')
    p.add_argument('--inspection-receipt',type=Path)
    p.add_argument('--aux-checkpoint',type=Path)
    p.add_argument('--budget',type=Path)
    args=p.parse_args(argv);report_path=output_path(args.report)
    if not args.execute_aux:
        require(args.inspection_receipt is None and args.aux_checkpoint is None and args.budget is None,
            'read-only inspection cannot accept execution arguments')
        require(not torch.cuda.is_available(),'CPU inspection must hide CUDA')
    checkpoint=args.checkpoint.resolve(strict=True);metadata=semantic_migration.checkpoint_metadata(checkpoint)
    contract=semantic_cli.runtime_contract(expected_head=args.expected_head,semantic_version='v3',experiment_id='rr_capture_then_rl_transfer_v1')
    source=validate_source(checkpoint,metadata,contract,expected_sha=args.expected_source_sha256,
        expected_manifest_sha=args.expected_manifest_sha256)
    data=load_reviewed_data(metadata,contract)
    binding={'source_checkpoint':source,'helpers_sha256':helpers(),
        'data_receipt_sha256':data['receipt']['receipt_content_sha256'],'runtime_contract_sha256':digest(contract)}
    result={'schema':SCHEMA,'binding':binding,'data_receipt':data['receipt'],
        'PPO_decisions_added':0,'PPO_updates_added':0,'PPO_optimizer_steps_added':0,
        'teacher_deployed':False,'physical_success_claimed':False,'original_PPO_counters':{k:metadata[k] for k in COUNTERS}}
    if not args.execute_aux:
        result.update(mode='read_only_current410_on_admitted_historical_front_states',auxiliary_updates=0,
            inspection=cpu_inspection(checkpoint,metadata,data),automatic_aux_enabled=False,
            per_row_semantics_proof=data['per_row_semantics_proof'])
    else:
        require(args.inspection_receipt is not None and args.aux_checkpoint is not None and args.budget is not None,
            'execution requires explicit same-source inspection, unique output and finite reviewed budget')
        prior=read(args.inspection_receipt)
        require(prior['mode']=='read_only_current410_on_admitted_historical_front_states' and prior['auxiliary_updates']==0
            and prior['binding']==binding,'stale/different source/helper/data inspection')
        budget=kernel.Budget(**read(args.budget));budget.validate()
        destination=output_path(args.aux_checkpoint,checkpoint=True)
        fit_path=output_path(report_path.with_name(report_path.stem+'_fit.json'))
        device=metadata['runner_config']['device']
        if str(device).startswith('cuda'): require(torch.cuda.is_available(),'official source-device execution requires full CUDA RNG visibility')
        runner=observation_runner(data['train_observations'][0],metadata,device=device)
        infos=training.load_semantic_checkpoint(runner,checkpoint,contract=contract,seed=metadata['seed'])
        report=kernel.fit(runner,*fit_arguments(data,device),budget=budget,authorized=True)
        training.write_json(fit_path,report)
        report_binding={'path':str(fit_path),'sha256':sha(fit_path),'content_sha256':digest(report)}
        result.update(mode='explicit_finite410_auxiliary_not_PPO',auxiliary_updates=report['accepted_auxiliary_updates'],
            fit_report=report_binding,budget=asdict(budget),auxiliary_checkpoint=None,
            reviewed_inspection={'path':str(args.inspection_receipt.resolve()),'sha256':sha(args.inspection_receipt)})
        if report['accepted_auxiliary_updates']:
            result['auxiliary_checkpoint']=save_auxiliary_checkpoint(runner,destination,infos,report=report,
                report_binding=report_binding,data=data,source_binding=source,budget=asdict(budget))
    training.write_json(report_path,result)
    print(json.dumps({'report':str(report_path),'mode':result['mode'],'auxiliary_updates':result['auxiliary_updates']}))
    return result


if __name__=='__main__': main()
