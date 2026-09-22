"""Read-only composition of frozen historical front candidates and probes.

No actor construction, gradients, fit, budget or checkpoint output. Current
checkpoint selection belongs to the caller and must be separately SHA-bound.
"""
from copy import deepcopy
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
ROOT = OUT.parents[1]
MANIFEST = HERE / 'data_manifest.json'
SCHEMA = 'wlr50_clean.front_mean_rehearsal_data_composition.v3'
COUNTERS = ('global_policy_decisions', 'ppo_updates', 'optimizer_steps')


def require(value, message):
    if not bool(value):
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def bound_path(binding):
    path = (OUT / binding['relative_path']).resolve(strict=True)
    require(path.is_relative_to(OUT.resolve()) and sha(path) == binding['sha256'], 'frozen data/helper binding changed')
    return path


def validate_current_metadata(metadata, contract, manifest, reference):
    """Same current runtime and inherited AUX lineage; weights may be newer.

    This is not a claim that an earlier admission inspected these new weights.
    It rejects another runtime, an older branch, or altered old AUX records.
    """
    from wlr50_clean.ppo.semantic_migration import _contract
    require(_contract(contract) == metadata['runtime_contract'] == reference['runtime_contract'], 'current source must retain exact reviewed5fd runtime contract')
    require(contract['source_git_commit'] == manifest['runtime_head']
            and contract['runtime_content_sha256'] == manifest['runtime_content_sha256'], 'composition runtime differs')
    for key in ('policy_contract','normalization','normalizer_state_sha256'):
        require(metadata[key] == reference[key], 'current source observation/kernel normalization differs')
    for name in ('p05_capture_assist','capture_feedback_semantics','rr_postcross_workspace'):
        require(metadata[name+'_migration'] == reference[name+'_migration'], 'immutable semantic migration lineage changed')
        require(metadata[name+'_branch']['counter_origin'] == reference[name+'_branch']['counter_origin'], 'semantic counter origin changed')
    require(metadata['task_conditioned_hip_wheel_branch']['auxiliary_mean_learning']
            == reference['task_conditioned_hip_wheel_branch']['auxiliary_mean_learning'], 'historical AUX7/8 lineage changed')
    require(metadata['rr_postcross_workspace_branch'] == reference['rr_postcross_workspace_branch'], 'current source must preserve entire existing two-event AUX ledger and RR origin')
    require(all(type(metadata[key]) is int and metadata[key] >= reference[key] for key in COUNTERS), 'source predates the event2 runtime/lineage reference')
    require(len(metadata['checkpoint_sha256']) == 64 and metadata['checkpoint_path'], 'caller must identify the actual checkpoint, not only reference semantics')
    return {'actual_caller_checkpoint': metadata['checkpoint_path'], 'actual_caller_checkpoint_sha256':metadata['checkpoint_sha256'],
        'actual_caller_PPO_counters':{key:metadata[key] for key in COUNTERS},
        'runtime_contract_sha256':digest(contract), 'actor_sha256':metadata['actor_parameter_sha256'],
        'historical_fixed_state_semantics_compatible':True,
        'old_admission_was_rebound_to_current_weights':False,
        'current_weights_or_trajectory_equivalent_to_demonstration_claimed':False}


def load_P01_candidate(path):
    m = read(path)
    require(m['schema']=='wlr50_clean.det_P01_second_decision_candidate.v1'
            and m['status']=='ONE_ROW_CANDIDATE_FEASIBILITY_NOT_ADMITTED_TO_FIT'
            and m['rows']==1, 'wrong P01 one-row candidate')
    require(m['original_389_directly_saved'] is False and m['field_supported_reconstruction_no_guessed_state'] is True,
            'P01 reconstruction provenance overstated')
    require(m['AUX_updates']==m['PPO_decisions_added']==m['PPO_updates_added']==0 and not m['checkpoint_written'], 'P01 candidate has false learning credit')
    require(sha(m['dataset']['path'])==m['dataset']['sha256'] and sha(m['helper']['path'])==m['helper']['sha256'], 'P01 frozen array/builder differs')
    require(m['fixed_CPU_CUDA_numeric_atol']==1e-6 and all(0 <= value <= 1e-6 for value in m['source_mean_history_sigma_errors'].values())
            and m['source_mean_history_sigma_errors']['history_center']==0, 'P01 original actor verification failed')
    compatibility=m['current_semantic_compatibility']
    require(compatibility['P01_WAIT_does_not_trigger_hold_air_revision'] and compatibility['RR_qualified_crossed_placed_all_false']
            and not compatibility['RR_retirement_active'] and compatibility['X17_float32_equal'], 'P01 fixed-state semantic proof missing')
    with np.load(m['dataset']['path'],allow_pickle=False) as loaded:
        a={key:loaded[key].copy() for key in loaded.files}
    x,y=a['X389_reconstructed'],a['raw12_actual_deterministic']
    require(x.shape==(1,389) and y.shape==(1,12) and x.dtype==y.dtype==np.float32
            and np.isfinite(x).all() and np.isfinite(y).all(), 'invalid P01 candidate tensors')
    require(np.array_equal(x[:,:13],np.asarray([[1.]+[0.]*12],dtype=np.float32))
            and np.array_equal(y,a['source_conditional_mean12'])
            and a['source_decision'].tolist()==[2] and a['input_tick'].tolist()==[8], 'P01 deterministic row identity changed')
    require(np.all(x[:,372:389]==0) and np.float32(compatibility['current_phi'])==x[0,17], 'P01 observable compatibility differs')
    # Read only the bounded source prefixes already declared by the constructor.
    for binding in m['source_prefixes']:
        source_path=Path(binding['path']); h=hashlib.sha256()
        with source_path.open('rb') as stream:
            for index in range(binding['rows_read']):
                raw=next(stream);h.update(raw)
                if source_path.name=='video_policy_decisions.jsonl' and index==1:
                    row=json.loads(raw)
                    require(hashlib.sha256(raw).hexdigest()==m['source_decision2_line_sha256']
                            and row['request_phase']=='P01' and row['start_tick']==8
                            and row['policy_request']['mode']=='deterministic_conditional_mean'
                            and row['raw_policy_action_full12']==y[0].tolist(), 'P01 actual executed raw binding changed')
        require(h.hexdigest()==binding['prefix_bytes_sha256'], 'P01 bounded source prefix changed')
    return torch.from_numpy(x.copy()),torch.from_numpy(y.copy()),m


def rr_rear_probe_reason(evaluation):
    """Eligibility is explicit; placed history bypasses workspace consumption.

    The predicate itself can still be true for a placed leg. Do not claim it
    is false when the actual potential instead skips its entire workspace term.
    """
    history=evaluation['history']
    if history['placed']['RR']:
        return 'already_placed_RR_bypasses_workspace_term'
    if not history['active_lift']['RR'] or not history['front_edge_crossed']['RR']:
        return 'RR_qualification_or_crossing_not_earned'
    return None


def load_rear_protection(metadata,contract,source_receipt):
    """At most one first eligible direct389 input per P07..P12, block03 only."""
    import yaml
    from wlr50_clean.ppo.semantic_supervisor import TaskStageSupervisor,_current_rr_receiver_preparation_retired
    target_phases=[f'P{i:02}' for i in range(7,13)]
    run=Path(source_receipt['run'])
    require(run.name=='20260922T0705260953371Z_ga802b24d78df_76c646571d794b4e96a82f69956928ff'
            and source_receipt['episode_index']==0,'extra probes restricted to reviewed block03 episode0')
    spec_path=ROOT/'configs/ppo_p05_hip_only_continuation_v1/stage_task_spec.yaml'
    require(sha(spec_path)==contract['files'][spec_path.relative_to(ROOT).as_posix()],'current potential source changed')
    current=object.__new__(TaskStageSupervisor);current.spec=yaml.safe_load(spec_path.read_text(encoding='utf-8'))
    old=object.__new__(TaskStageSupervisor);old.spec=deepcopy(current.spec);old.spec.pop('rr_postcross_workspace_semantics')
    selected={};cache={};bindings={};prior=None;last_read_index=-1
    with (run/'residual_and_projection_audit.jsonl').open('rb') as stream:
        for index,raw_line in enumerate(stream):
            # This first episode was already audited and ended at1721 decisions;
            # never scan later episodes, other runs or an unbounded history.
            require(index<1721,'reviewed first episode exceeded its known bounded extent')
            row=json.loads(raw_line);a=row['applied_audit'];phase=a['phase_id'];last_read_index=index
            require(a['decision_count']==index+1 and row['global_policy_decision']==203777+index,'block03 first-episode identity differs')
            if phase in target_phases and phase not in selected:
                require(prior is not None and not prior['terminal'],'rear input has no adjacent nonterminal endpoint')
                previous=prior['applied_audit'];ev=previous['semantic_task']['physical_evaluator']
                require(previous['physics_tick']==a['physics_tick']-a['physics_ticks']==ev['physics_tick'], 'rear input tick not adjacent')
                require(previous['end_phase_id']==phase and ev['valid'] is True and ev['termination_reason'] is None,'rear input phase/evaluator invalid')
                reason=rr_rear_probe_reason(ev)
                if reason is not None:
                    update=1558+index//128;offset=index%128
                    if update not in cache:
                        path=run/f'rollouts/rollout_{update:06}.pt'
                        cache[update]=torch.load(path,map_location='cpu',weights_only=False)
                        bindings[str(update)]={'path':str(path),'sha256':sha(path)}
                    batch=cache[update]
                    require(batch['runtime_contract']==source_receipt['source_runtime_contract']
                            and batch['policy_contract']==metadata['policy_contract']
                            and batch['curriculum_epoch']['prefix_request'] is None, 'rear direct rollout contract/prefix differs')
                    x=batch['observations']['policy'][offset,0].clone();action=batch['actions'][offset,0]
                    require(x.shape==(389,) and x.dtype==torch.float32 and torch.isfinite(x).all()
                            and torch.equal(x,batch['observations']['critic'][offset,0])
                            and int(x[:13].argmax())==int(phase[1:])-1, 'rear direct389 tensor/phase differs')
                    mean,std=(value[offset,0] for value in batch['distribution_params'])
                    for tensor,field in ((action,'raw_policy_action_full12'),(mean,'old_distribution_mean_full12'),(std,'old_distribution_std_full12')):
                        require(torch.equal(tensor,torch.tensor(row[field],dtype=tensor.dtype)),'rear sealed raw/distribution source mismatch')
                    logp=batch['actions_log_prob'][offset,0]
                    require(torch.equal(logp,torch.tensor(row['old_log_probability'],dtype=logp.dtype).reshape(logp.shape)), 'rear logp source mismatch')
                    logp_error=float((torch.distributions.Normal(mean,std).log_prob(action).sum()-logp.reshape(())).abs())
                    require(logp_error<=1e-5,'rear actual raw likelihood differs')
                    old_phi,new_phi=old.physical_potential(ev),current.physical_potential(ev)
                    require(old_phi==new_phi and abs(old_phi-a['reward_breakdown']['potential_before'])<1e-12
                            and torch.tensor(new_phi,dtype=torch.float32).item()==x[17].item(),'rear current potential is not unchanged')
                    gate=bool(_current_rr_receiver_preparation_retired(current.spec,'RR',ev))
                    consumed=gate and not ev['history']['placed']['RR']
                    require(not consumed,'RR retirement actually consumes workspace on rear probe')
                    native=a['actuator_target_effect_audit']
                    require(native['verified'] and native['phase_mask_full12']==[1]*12
                            and not row['terminal'],'rear protection input is invalid/terminal intervention')
                    selected[phase]={'observation':x,'receipt':{'phase':phase,'source_episode_index':0,'source_index':index,
                        'global_policy_decision':row['global_policy_decision'],'input_tick':ev['physics_tick'],
                        'rollout_update':update,'rollout_offset':offset,'source_row_sha256':hashlib.sha256(raw_line).hexdigest(),
                        'observation_float32_sha256':tensor_sha(x),'directly_saved389':True,'used_as_action_label':False,
                        'eligibility_reason':reason,'RR_qualified_history':ev['history']['active_lift']['RR'],
                        'RR_crossed_history':ev['history']['front_edge_crossed']['RR'],'RR_placed_history':ev['history']['placed']['RR'],
                        'RR_retirement_predicate_active':gate,'RR_retirement_workspace_term_consumed':consumed,
                        'old_phi':old_phi,'current_phi':new_phi,'X17':x[17].item(),'X17_float32_equal':True,
                        'raw_Gaussian_logp_max_error':logp_error}}
            if len(selected)==len(target_phases) or row['terminal']:
                break
            prior=row
    ordered=[phase for phase in target_phases if phase in selected]
    tensors=torch.stack([selected[phase]['observation'] for phase in ordered]) if ordered else torch.empty((0,389),dtype=torch.float32)
    return tensors,{'scope':'only_reviewed_block03_episode0_direct_rollouts_first_eligible_per_phase',
        'source_run':str(run),'source_rows_read':last_read_index+1,'maximum_source_rows_allowed':1721,
        'requested_phases':target_phases,'actual_phases':ordered,'missing_phases':[p for p in target_phases if p not in selected],
        'P13_real_coverage_claimed':False,'rollout_bindings':bindings,'rows':[selected[p]['receipt'] for p in ordered],
        'source_actions_checked_only_as_provenance_not_returned_as_targets':True,
        'labels_or_new_PPO_credit':0,'mean_head_mathematical_invariance_claimed':False}


def load_reviewed_data(metadata, contract):
    manifest=read(MANIFEST)
    require(manifest['schema']==SCHEMA and manifest['status']=='READONLY_COMPOSITION_NOT_OPTIMIZATION_AUTHORIZATION', 'unsupported composition')
    paths={name:bound_path(binding) for name,binding in manifest['bindings'].items()}
    reference_path=bound_path(manifest['reference_checkpoint'])
    reference_manifest=reference_path.with_name(reference_path.stem+'_manifest.json')
    require(sha(reference_manifest)==manifest['reference_checkpoint']['manifest_sha256'], 'semantic reference manifest changed')
    reference=read(reference_manifest)
    current=validate_current_metadata(metadata,contract,manifest,reference)
    old_admission=read(paths['P02_current_semantics'])
    require(old_admission['result']=='PASS_BOUNDED_P02_HISTORICAL_SUPERVISED_DATA_SEMANTICS_ONLY'
            and old_admission['rows_checked']==254 and old_admission['all_float32_X17_exact']
            and old_admission['current_runtime']==contract['source_git_commit'], 'fixed P02 semantic compatibility missing')
    for name,record in (('feedback',metadata['capture_feedback_semantics_migration']),('RR_potential',metadata['rr_postcross_workspace_migration'])):
        require(record['plan_sha256']==old_admission['migration_plan_sha256'][name] and sha(record['plan_path'])==record['plan_sha256'], 'immutable migration proof differs')
    name='frozen_det_v2_data_for_mean_v3'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,paths['frozen_det_loader']);module=importlib.util.module_from_spec(spec)
        sys.modules[name]=module;spec.loader.exec_module(module)
    # Only invoke its read-only data validation path, never inspect/fit/save.
    frozen=sys.modules[name].load_validated_data(paths['P02_manifest'],metadata,contract)
    p01, p01raw, p01receipt=load_P01_candidate(paths['P01_manifest'])
    require(p01receipt['current_semantic_reference']['runtime']==contract['source_git_commit']
            and p01receipt['source_checkpoint']['sha256']==manifest['positive_labels']['source_checkpoint_sha256'], 'P01 source/runtime binding differs')
    as_tensor=lambda value: torch.as_tensor(value,dtype=torch.float32,device='cpu').clone()
    p02,raw02=as_tensor(frozen['train_observations']),as_tensor(frozen['train_raw_targets'])
    vx,vy=as_tensor(frozen['validation_observations']),as_tensor(frozen['validation_raw_targets'])
    front_protection=as_tensor(frozen['invariance_observations']);probe=as_tensor(frozen['p01_observations'])
    require(p02.shape==(85,389) and raw02.shape==(85,12) and vx.shape==(85,389) and vy.shape==(85,12), 'fixed P02 split changed')
    require(front_protection.shape==(13,389) and probe.shape==(2,389) and bool((front_protection[:,:2]==0).all())
            and bool((probe[:,0]==1).all()), 'direct protection/probe selection changed')
    rear,rear_receipt=load_rear_protection(metadata,contract,
        frozen['receipt']['protection_holdouts']['full_source_validation_receipt'])
    protection=torch.cat((front_protection,rear),0)
    tx,ty=torch.cat((p01,p02),0),torch.cat((p01raw,raw02),0)
    groups={'P01':[0],'P02':list(range(1,86))}
    receipt={'schema':SCHEMA+'.loaded','composition_manifest':{'path':str(MANIFEST),'sha256':sha(MANIFEST)},
        'loader':{'path':str(Path(__file__).resolve()),'sha256':sha(Path(__file__))},
        'actual_current_source_binding':current,
        'reference_checkpoint_role':manifest['reference_checkpoint']['role'],
        'historical_P02_admission_reference_preserved':{'path':str(paths['P02_current_semantics']),
            'sha256':manifest['bindings']['P02_current_semantics']['sha256'],
            'original_current_checkpoint':old_admission['current_checkpoint'],
            'original_current_checkpoint_sha256':old_admission['current_checkpoint_sha256'],
            'fixed_input_semantic_proof_only_not_current_weight_inspection':True},
        'deterministic_P02_source_receipt':deepcopy(frozen['receipt']), 'deterministic_P01_source_receipt':p01receipt,
        'additional_rear_protection_receipt':rear_receipt,
        'selection':{'train_phase_groups':groups,'train_rows':86,'P02_validation_rows':85,
            'P01_independent_validation_rows':0,'P01_deterministic_training_and_probe_same_row':True,
            'P01_stochastic_probes_used_as_labels':False,'P03_P06_protection_rows':13,
            'additional_P07_P12_protection_rows':len(rear),'total_protection_rows':len(protection),
            'actual_protection_phases':[f'P{i+1:02}' for i in sorted(set(protection[:,:13].argmax(-1).tolist()))],
            'missing_rear_protection_phases':rear_receipt['missing_phases'],'P13_real_coverage_claimed':False,'P01_stochastic_probe_rows':2,
            'P02_source_only_original_rows_retained':84,'P01_reset_row_not_reconstructed':True},
        'tensor_sha256':{key:tensor_sha(value) for key,value in {'train_observations':tx,'train_raw_targets':ty,
            'validation_observations':vx,'validation_raw_targets':vy,'protection_observations':protection,
            'p01_observations':probe,'p01_deterministic_probe_observations':p01,'p01_deterministic_probe_raw_targets':p01raw}.items()},
        'protection_semantics':'finite_real_state_probes_require_explicit_mean_loss_or_trust_checks;not_P03plus_mathematical_invariance',
        'loss_weights_selected_by_loader':False,'current_actor_or_future_trajectory_equivalence_claimed':False,
        'observations_directly_saved389_for_positive_labels':False,
        'offpolicy_historical_supervised_only_not_PPO':True,'optimization_authorized_by_loader':False,
        'AUX_updates':0,'PPO_decisions_added':0,'PPO_updates_added':0,'limitations':manifest['limitations']}
    receipt['receipt_content_sha256']=digest(receipt)
    return {'train_observations':tx,'train_raw_targets':ty,'train_phase_groups':groups,
        'validation_observations':vx,'validation_raw_targets':vy,
        'p01_deterministic_probe_observations':p01,'p01_deterministic_probe_raw_targets':p01raw,
        'protection_observations':protection,'p01_observations':probe,'receipt':receipt}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--expected-sha256',required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    require(not torch.cuda.is_available(),'standalone read-only test must be CPU-only')
    torch.set_num_threads(1)
    from wlr50_clean.ppo.semantic_migration import checkpoint_metadata
    from wlr50_clean.ppo.semantic_cli import runtime_contract
    metadata=checkpoint_metadata(args.checkpoint.resolve(strict=True))
    require(metadata['checkpoint_sha256']==args.expected_sha256,'explicit caller checkpoint binding differs')
    contract=runtime_contract(expected_head=read(MANIFEST)['runtime_head'],semantic_version='v3',experiment_id='p05_hip_only_continuation_v1')
    data=load_reviewed_data(metadata,contract)
    target=args.report.resolve()
    require(target.is_relative_to(HERE.resolve()) and not target.exists(),'read test report requires fresh path in this output directory')
    target.write_text(json.dumps(data['receipt'],indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'read_only':'PASS','source_SHA':metadata['checkpoint_sha256'],'source_decisions':metadata['global_policy_decisions'],
        'shapes':{key:list(value.shape) for key,value in data.items() if torch.is_tensor(value)},
        'train_phase_counts':{key:len(value) for key,value in data['train_phase_groups'].items()},'report':str(target),
        'fit_performed':False,'checkpoint_written':False},indent=2))


if __name__=='__main__':
    main()
