"""Default readonly CPU inspection; explicitly bound finite mean-head AUX only.

No automatic checkpoint selection, objective, budget, retry or publication.
Inspect maps only the source actor to CPU. Execute officially reloads the full
source on its original CUDA device, then saves a unique checkpoint and proves
an independent official reload. Existing AUX/PPO lineage is never rewritten.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib
import json
from pathlib import Path

import rr_mean_rehearsal as kernel
from wlr50_clean.ppo import semantic_cli, semantic_migration
from wlr50_clean.ppo.semantic_p05_capture_profile import P05_CAPTURE_POLICY,P05_CAPTURE_OBSERVATION_LAYOUT

HERE=Path(__file__).resolve().parent; OUT=HERE.parent
torch,training=kernel.torch,kernel.training
require=kernel.require
SCHEMA=kernel.SCHEMA
HEAD='5fd88852bf20c94cd74405c791a13a9fd9e0a3d8'
SOURCE_SHA='8d0305626decff2214f8c3c0eee4031bdb791013d82a641a8ac4fc077ce235e9'
SOURCE_DECISIONS=214400
LEDGER_KEY='front_rehearsal_auxiliary'
LEDGER_SCHEMA='wlr50_clean.front_rehearsal_auxiliary.v1'
EVENT_KIND='finite_supervised_existing_mean_head_actual_RR_capture_continuation_raw_actions_not_PPO'
COUNTERS=('global_policy_decisions','ppo_updates','optimizer_steps')


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def json_value(value):return json.loads(json.dumps(value,allow_nan=False))


def auxiliary_credit_breakdown(report):
    accepted=report['accepted_auxiliary_updates'];attempted=report['attempted_auxiliary_optimizer_steps']
    return {'inherited_historical_auxiliary':{'accepted':7,'attempted':8},
        'inherited_front_auxiliary':{'accepted':96,'attempted':96},
        'this_RR_auxiliary':{'accepted':accepted,'attempted':attempted},
        'cumulative_mixed_ledger_auxiliary':{'accepted':96+accepted,'attempted':96+attempted},
        'legacy_storage_key_does_not_mean_all_events_are_front_AUX':True,
        'all_auxiliary_counts_are_separate_from_PPO':True}


def helper_hashes():
    require(sha(kernel.FROZEN_MEAN)==kernel.FROZEN_MEAN_SHA256,'immutable reviewed mean-head mechanics changed')
    require(sha(kernel.FROZEN)==kernel.FROZEN_SHA256,'immutable original389 kernel changed')
    return {name:sha(HERE/name) for name in
            ('rr_mean_rehearsal.py','fit_rr_mean_core.py','execute_rr_mean_rehearsal.py','data_v4.py')}


def load_objective(path):
    envelope=read(path)
    require(set(envelope)=={'schema','objective'} and envelope['schema']==SCHEMA+'.objective',
            'explicit objective envelope required; no implicit phase weights')
    objective=kernel.Objective(**envelope['objective']);objective.validate()
    return objective,{'path':str(Path(path).resolve()),'sha256':sha(path),'envelope':envelope}


def source_identity(path,metadata):
    path=Path(path).resolve()
    return {'path':str(path),'sha256':metadata['checkpoint_sha256'],
        'manifest_sha256':sha(path.with_name(path.stem+'_manifest.json')),
        'actor_sha256':metadata['actor_parameter_sha256'],
        'PPO_counters':{key:metadata[key] for key in COUNTERS},
        'runtime_contract_sha256':digest(metadata['runtime_contract'])}


def prepare_context(checkpoint,metadata,contract,*,objective_path):
    """Read-only data/objective/helper binding; never creates a budget."""
    require(contract['source_git_commit']==HEAD and contract==metadata['runtime_contract'],
            'actual source must use the reviewed current5fd runtime')
    require(metadata['checkpoint_sha256']==SOURCE_SHA and metadata['global_policy_decisions']==SOURCE_DECISIONS,
            'this bounded RR operation binds exactly actual CP214400, never an automatically selected source')
    objective,objective_receipt=load_objective(objective_path)
    dataset=importlib.import_module('data_v4')
    data=dataset.load_reviewed_data(metadata,contract)
    binding={'source_checkpoint':source_identity(checkpoint,metadata),'helpers':helper_hashes(),
        'data_manifest':{'path':str(dataset.MANIFEST.resolve()),'sha256':sha(dataset.MANIFEST)},
        'data_receipt_sha256':digest(data['receipt']),'objective':objective_receipt,'expected_head':HEAD}
    return binding,data,objective


def verify_inspection(path,base_binding):
    value=read(path)
    require(value['schema']==SCHEMA+'.readonly' and value['binding']==base_binding,
            'inspection is stale for source/data/helpers/objective')
    require(value['inspection_only'] is True and value['optimizer_steps_performed']==0
            and value['PPO_updates_added']==0 and value['checkpoint_written'] is False,
            'genuine readonly inspection required')
    report=value['inspection']
    require(report['schema']==SCHEMA+'.inspection' and report['optimized_parameters']==kernel.PARAMETERS
            and report['optimized_scalar_count']==3084 and report['optimizer_steps_performed']==0
            and report['same_input_sigma_exact_by_frozen_trunk_and_sigma_rows'] is True
            and report['P03plus_mean_bitwise_invariance_claimed'] is False,
            'inspection parameterization/protection semantics differ')
    require(report['objective']==base_binding['objective']['envelope']['objective']
            and report['actor_sha256_unchanged']==base_binding['source_checkpoint']['actor_sha256'],
            'inspection objective/actor differs')
    require(set(report['phase_sample_counts'])==set(kernel.POSITIVE_PHASES)
            and report['phase_sample_counts']['P10']==1
            and set(report['validation_phase_sample_counts'])==set(kernel.VALIDATION_PHASES)
            and report['actual_protection_phases']==list(kernel.PROTECTION_PHASES)
            and report['P10_independent_validation_rows']==0
            and report['P13_actual_protection_coverage'] is False
            and report['single_historical_trajectory_train_validation_correlated'] is True,
            'inspection must retain reviewed RR/protection coverage and its explicit limitations')
    return {'path':str(Path(path).resolve()),'sha256':sha(path),'schema':value['schema']}


def prepare_binding(checkpoint,metadata,contract,*,objective_path,inspection_path):
    base,data,objective=prepare_context(checkpoint,metadata,contract,objective_path=objective_path)
    binding={**base,'inspection':verify_inspection(inspection_path,base)}
    return binding,data,objective


def load_budget(path,binding):
    value=read(path)
    require(set(value)=={'schema','binding','budget'} and value['schema']==SCHEMA+'.explicit_budget'
            and value['binding']==binding,'explicit budget must bind exact reviewed source/data/objective/inspection/helpers')
    budget=kernel.Budget(**value['budget']);budget.validate()
    return budget,{'path':str(Path(path).resolve()),'sha256':sha(path),'envelope':value}


def observation_runner(observation,metadata,*,device,inspect_copy=False):
    class ObservationOnly:
        num_envs,num_actions=1,12
        cfg={'evaluation':True,'semantic_version':'v3'}
        def get_observations(self):
            x=torch.as_tensor(observation,dtype=torch.float32,device=device).reshape(1,389)
            return kernel.kernel.TensorDict({'policy':x,'critic':x.clone()},batch_size=[1],device=device)
    runner,_=training.construct_semantic_runner(ObservationOnly(),seed=metadata['seed'],device=device,
        initialize_actor=False,policy_version=P05_CAPTURE_POLICY,observation_layout=P05_CAPTURE_OBSERVATION_LAYOUT)
    expected=deepcopy(metadata['runner_config'])
    if inspect_copy: expected['device']='cpu'
    require(runner._semantic_runner_config==expected and training._runner_policy_contract(runner)==metadata['policy_contract'],
            'runner device/config/policy contract differs')
    return runner


def cpu_inspection(checkpoint,metadata,data,objective):
    require(not torch.cuda.is_available(),'readonly inspector requires CUDA_VISIBLE_DEVICES=-1')
    rng=training.capture_training_rng_state(seed=metadata['seed'])
    try:
        # CPU mapping is an explicitly labelled readonly actor copy, not a new
        # source checkpoint, full Adam restore, or device metadata migration.
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        require(payload['infos']['actor_parameter_sha256']==metadata['actor_parameter_sha256'],'source actor metadata differs')
        run=observation_runner(data['train_observations'][0],metadata,device='cpu',inspect_copy=True)
        run.alg.actor.load_state_dict(payload['actor_state_dict'],strict=True)
        require(training.parameter_hash(run.alg.actor)==metadata['actor_parameter_sha256'],'CPU actor copy differs')
        return kernel.inspect(run.alg.actor,data,objective=objective)
    finally:
        training.restore_training_rng_state(rng,expected_seed=metadata['seed'])


def _new_path(path,*,checkpoint=False):
    path=Path(path).resolve()
    require(path.is_relative_to(OUT.resolve()) and not path.exists(),'require a new destination inside experiment outputs')
    if checkpoint:
        require(path.suffix=='.pt' and path.name=='checkpoint_aux_meanrr_step_000214400_v4.pt'
                and not path.with_name(path.stem+'_manifest.json').exists(),'unique explicit checkpoint_aux_meanrr_step_000214400_v4.pt required')
    return path


def _validate_fit(report):
    require(report['schema']==SCHEMA+'.fit' and report['optimized_parameters']==kernel.PARAMETERS
            and report['optimized_scalar_count']==3084,'wrong fit schema/mean-only parameter scope')
    accepted,attempts=report['accepted_auxiliary_updates'],report['attempted_auxiliary_optimizer_steps']
    require(type(accepted) is int and type(attempts) is int and 0 < accepted <= attempts <= 32,
            'only real accepted finite AUX updates may be published')
    require(report['actor_parameter_sha256_before']!=report['actor_parameter_sha256_after']
            and report['same_input_sigma_exact'] is True,'accepted update requires actual mean change and fixed same-input sigma')
    require(report['training_rng_preserved'] is True and report['fresh_PPO_rollout_required'] is True
            and report['teacher_deployed'] is False and report['physical_success_claimed'] is False
            and all(report[key]==0 for key in ('PPO_decisions_added','PPO_updates_added','PPO_optimizer_steps_added')),
            'false preservation/provenance/PPO credit in fit report')
    require(report['positive_label_phases']==list(kernel.POSITIVE_PHASES)
            and report['actual_protection_phases']==list(kernel.PROTECTION_PHASES)
            and report['P10_independent_validation_rows']==0
            and report['single_historical_trajectory_train_validation_correlated'] is True,
            'RR positive/protection scope or correlated single-trajectory evidence differs')


def append_event(infos,*,report,binding,data_receipt,budget_receipt):
    _validate_fit(report)
    require(json_value(report['objective'])==binding['objective']['envelope']['objective']
            and json_value(report['budget'])==budget_receipt['envelope']['budget']
            and digest(data_receipt)==binding['data_receipt_sha256']
            and binding['source_checkpoint']['PPO_counters']=={key:infos[key] for key in COUNTERS},
            'actual fit objective/budget/data/source counters differ from authorization binding')
    result=deepcopy(infos);branch=result['rr_postcross_workspace_branch'];ledger=branch[LEDGER_KEY]
    require(branch['schema']=='wlr50_clean.rr_postcross_workspace_same389.v1'
            and branch['semantics']=='current_qualified_RR_over_top_receiver_retirement_v1'
            and branch['migration_added_updates']==0 and ledger['schema']==LEDGER_SCHEMA,
            'original current RR/AUX lineage differs')
    require(len(ledger['events'])==3 and [e['event_index'] for e in ledger['events']]==[1,2,3]
            and ledger['accepted_auxiliary_updates_total']==96
            and ledger['attempted_auxiliary_optimizer_steps_total']==96,
            'this bounded operation appends event4 to immutable existing events1-3 (96/96), never overwrites/retries')
    require(sum(e['fit_report']['accepted_auxiliary_updates'] for e in ledger['events'])==96
            and sum(e['fit_report']['attempted_auxiliary_optimizer_steps'] for e in ledger['events'])==96,
            'inherited ledger totals differ from actual old event counts')
    old=deepcopy(ledger['events'])
    inherited=result['task_conditioned_hip_wheel_branch']['auxiliary_mean_learning']
    require(inherited['accepted_auxiliary_updates_total']==7
            and inherited['attempted_auxiliary_optimizer_steps_total']==8,
            'inherited historical AUX7/8 counters differ')
    event={'event_index':4,'kind':EVENT_KIND,'source_checkpoint':deepcopy(binding['source_checkpoint']),
        'helper_sha256':deepcopy(binding['helpers']),'binding':deepcopy(binding),'data_receipt':deepcopy(data_receipt),
        'fit_report':deepcopy(report),'fit_report_sha256':digest(report),'objective':deepcopy(binding['objective']),
        'budget':deepcopy(budget_receipt['envelope']['budget']),'budget_receipt':deepcopy(budget_receipt),
        'optimized_parameters':kernel.PARAMETERS.copy(),'optimized_scalar_count':3084,
        'positive_phase_scope':['P09','P10','P11'],'affected_mean_phase_scope':['P%02d'%p for p in range(1,14)],
        'mean_and_log_sigma_may_change':False,'mean_may_change_all_phases':True,'same_input_sigma_fixed':True,
        'protection_is_finite_observed_state_loss_and_trust_not_global_mean_invariance':True,
        'source_positive_observations_directly_saved389':True,
        'current_positive_observations_are_original_unmodified389':False,
        'current_potential_observation_mapping_index':17,
        'target_semantics':'actually_executed_stochastic_raw12_not_stored_conditional_mean',
        'validation_phase_scope':['P09','P11'],'P10_positive_rows':1,'P10_independent_validation_rows':0,
        'single_historical_trajectory_train_validation_correlated':True,
        'auxiliary_credit_breakdown':auxiliary_credit_breakdown(report),
        'actual_protection_phases':deepcopy(report['actual_protection_phases']),
        'missing_actual_protection_phases':deepcopy(report['missing_actual_protection_phases']),
        'P13_actual_protection_coverage':False,'fresh_PPO_rollout_required':True,
        'teacher_deployed':False,'physical_success_claimed':False,
        'PPO_counters_unchanged':{k:infos[k] for k in COUNTERS},
        'stage_requested_decisions_unchanged':deepcopy(infos['stage_requested_decisions']),
        'PPO_decisions_added':0,'PPO_updates_added':0,'PPO_optimizer_steps_added':0}
    ledger['events'].append(json_value(event))
    for output_key,input_key in [('accepted_auxiliary_updates_total','accepted_auxiliary_updates'),
                                  ('attempted_auxiliary_optimizer_steps_total','attempted_auxiliary_optimizer_steps')]:
        ledger[output_key]=sum(e['fit_report'][input_key] for e in ledger['events'])
    require(ledger['events'][:3]==old,'old event1/2/3 changed')
    original=deepcopy(result);original['rr_postcross_workspace_branch'][LEDGER_KEY]=deepcopy(infos['rr_postcross_workspace_branch'][LEDGER_KEY])
    require(original==infos,'old branches, migration records, origins or AUX7/8 changed')
    return result


def protected_saved_state(run,infos):
    for key,actual in [('critic_parameter_sha256',training.parameter_hash(run.alg.critic)),
        ('optimizer_state_sha256',training.state_hash(run.alg.optimizer.state_dict())),
        ('normalizer_state_sha256',training.state_hash(training._normalizers(run)))]:
        require(actual==infos[key],'protected source training state changed: '+key)
    require(run._semantic_runner_config==infos['runner_config']
        and training.optimizer_learning_rate(run)==infos['optimizer_learning_rate']
        and training.capture_training_rng_state(seed=infos['seed'])==infos['training_rng_state'],
        'source device/LR/config/full CPU-CUDA RNG changed')


def save_candidate(run,path,infos,*,data,report,binding,budget_receipt):
    path=_new_path(path,checkpoint=True);_validate_fit(report)
    require(run.alg.storage.step==0 and run.alg.transition.actions is None,'official save requires empty fresh rollout')
    require(infos['actor_parameter_sha256']==report['actor_parameter_sha256_before']
        and training.parameter_hash(run.alg.actor)==report['actor_parameter_sha256_after'],'actor source/result hash differs')
    protected_saved_state(run,infos)
    payload=append_event(infos,report=report,binding=binding,data_receipt=data['receipt'],budget_receipt=budget_receipt)
    payload['stage']='auxiliary_existing_mean_head_not_PPO'
    cp,manifest=training.save_semantic_checkpoint(run,path,payload)
    meta=semantic_migration.checkpoint_metadata(cp)
    fresh=observation_runner(data['train_observations'][0],meta,device=meta['runner_config']['device'])
    loaded=training.load_semantic_checkpoint(fresh,cp,contract=meta['runtime_contract'],seed=meta['seed'])
    require(all(loaded[k]==infos[k] for k in COUNTERS) and fresh.alg.storage.step==0
            and fresh.alg.transition.actions is None,'independent reload changed counters/fresh rollout')
    require(loaded['rr_postcross_workspace_branch']==payload['rr_postcross_workspace_branch'],'independent complete ledger reload differs')
    for key in infos:
        if key.endswith(('_branch','_migration')) and key!='rr_postcross_workspace_branch':
            require(loaded[key]==infos[key],'historical branch/migration differs after reload: '+key)
    require(training.parameter_hash(fresh.alg.actor)==report['actor_parameter_sha256_after'],'independent actor reload differs')
    protected_saved_state(fresh,infos)
    return {'path':str(cp),'manifest':str(manifest),'sha256':sha(cp),'independent_official_reload_verified':True,
        'source_device_preserved':meta['runner_config']['device'],'latest_pointer_published':False,
        'normal_PPO_save_carry_after_this_real_AUX_not_yet_verified':True}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','objective','report'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--expected-checkpoint-sha256',required=True)
    parser.add_argument('--expected-policy-decisions',type=int,required=True)
    parser.add_argument('--expected-head',required=True)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--inspect',action='store_true');mode.add_argument('--execute-aux',action='store_true')
    for name in ('budget','inspection-receipt','aux-checkpoint'):parser.add_argument('--'+name,type=Path)
    args=parser.parse_args(argv)
    require(args.expected_head==HEAD,'current reviewed runtime head required')
    report_path=_new_path(args.report);cp=args.checkpoint.resolve(strict=True)
    meta=semantic_migration.checkpoint_metadata(cp)
    require(meta['checkpoint_sha256']==args.expected_checkpoint_sha256
            and meta['global_policy_decisions']==args.expected_policy_decisions
            and args.expected_checkpoint_sha256==SOURCE_SHA and args.expected_policy_decisions==SOURCE_DECISIONS,
            'explicit actual CP214400 source SHA/counter mismatch')
    contract=semantic_cli.runtime_contract(expected_head=HEAD,semantic_version='v3',experiment_id='p05_hip_only_continuation_v1')
    require(meta['runtime_contract']==contract,'source/current runtime contract differs')
    if not args.execute_aux:
        require(args.budget is None and args.inspection_receipt is None and args.aux_checkpoint is None,
                'readonly inspection accepts no execution budget/destination')
        binding,data,objective=prepare_context(cp,meta,contract,objective_path=args.objective)
        report=cpu_inspection(cp,meta,data,objective)
        result={'schema':SCHEMA+'.readonly','binding':binding,'data_receipt':data['receipt'],'inspection':report,
            'inspection_only':True,'optimizer_steps_performed':0,'PPO_updates_added':0,'checkpoint_written':False,
            'source_device_metadata_unchanged':True,'actor_copy_inspection_device':'cpu'}
    else:
        require(args.budget is not None and args.inspection_receipt is not None and args.aux_checkpoint is not None,
                'explicit execute requires reviewed inspection, bound finite budget, unique checkpoint')
        target=_new_path(args.aux_checkpoint,checkpoint=True)
        binding,data,objective=prepare_binding(cp,meta,contract,objective_path=args.objective,inspection_path=args.inspection_receipt)
        budget,budget_receipt=load_budget(args.budget,binding)
        device=meta['runner_config']['device']
        require(str(device).startswith('cuda') and torch.cuda.is_available(),
                'real execution requires original source CUDA device; CPU relocation forbidden')
        aliases=[OUT/'checkpoints'/name for name in ('checkpoint_last_pointer.json','checkpoint_last.pt',
            'checkpoint_last_manifest.json','resume_state.json')]
        before={str(p):sha(p) if p.exists() else None for p in aliases}
        run=observation_runner(data['train_observations'][0],meta,device=device)
        infos=training.load_semantic_checkpoint(run,cp,contract=contract,seed=meta['seed'])
        core=importlib.import_module('fit_rr_mean_core')
        report=core.fit_mean_head(run,data,objective=objective,budget=budget,authorized=True)
        result={'schema':SCHEMA+'.execution','binding':binding,'budget_receipt':budget_receipt,
            'fit_report':report,'auxiliary_checkpoint':None,'teacher_deployed':False,'physical_success_claimed':False,
            'auxiliary_credit_breakdown':auxiliary_credit_breakdown(report),
            'PPO_decisions_added':0,'PPO_updates_added':0,'PPO_optimizer_steps_added':0}
        if report['accepted_auxiliary_updates']>0:
            result['auxiliary_checkpoint']=save_candidate(run,target,infos,data=data,report=report,binding=binding,budget_receipt=budget_receipt)
        require({str(p):sha(p) if p.exists() else None for p in aliases}==before,'latest-pointer aliases changed')
        result['latest_pointer_artifacts_unchanged']=True
    training.write_json(report_path,result)
    print(json.dumps({'report':str(report_path),'mode':'execute' if args.execute_aux else 'readonly',
        'PPO_added':0,'checkpoint_written':bool(result.get('auxiliary_checkpoint'))}))
    return result


if __name__=='__main__':main()
