"""Small sealed-run receipt wrapper; logs only, no torch/model/simulator imports."""
import argparse
import collections
import hashlib
import importlib.util
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
EXPERIMENT='task_conditioned_hip_wheel_v1'
POLICY='task_conditioned_hip_wheel_sigma_v1'
OBJECTIVE='task_conditioned_hip_wheel_quality_v1'
PHASES=tuple(f'P{i:02d}' for i in range(1,14))
spec=importlib.util.spec_from_file_location('_old_real_receipt',ROOT/'outputs/ppo_fl_capture_quality_v1/training_receipt.py')
base=importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):value.update(chunk)
    return value.hexdigest()


def require(condition,message):
    if not condition:raise ValueError(message)


def outcome(row):
    info=row['applied_audit'];task=info['semantic_task']
    completed=task.get('completed_stage_ids')
    require(isinstance(completed,list) and all(p in PHASES for p in completed),'missing valid completed-stage evidence')
    first=next((p for p in PHASES if p not in completed),None)
    return dict(global_policy_decision=row['global_policy_decision'],terminal=row['terminal'],
        phase=info['phase_id'],end_phase=info.get('end_phase_id'),termination_reason=info.get('termination_reason'),
        task_success=info.get('task_success'),full_task_success=info.get('full_task_success'),
        task_outcome_label=info.get('task_outcome_label'),task_result_scope=info.get('task_result_scope'),
        curriculum_start=info.get('curriculum_start'),first_unfinished_stage_by_logged_completion=first,
        current_task_purpose=task.get('purpose'),completed_stage_ids=completed,
        completion_values=task.get('completion_values'),sim_time_s=info.get('sim_time_s'),
        classification=('recorded_terminal' if row['terminal'] else 'budget_boundary_not_a_task_failure_or_success'))


def summarize(run,*,project_root=ROOT):
    run=Path(run).resolve(strict=True);project_root=Path(project_root).resolve()
    require(run.parent==project_root/'runs'/('ppo_'+EXPERIMENT)/'train','not a formal train run in the new experiment')
    manifest=read(run/'run_manifest.json');training=read(run/'training_manifest.json')
    require(manifest['lifecycle'] in ('SUCCEEDED','STOPPED_AT_VERIFIED_UPDATE_BOUNDARY'),'run is not sealed at a complete update')
    require(manifest['result']==training,'final run/training manifests differ')
    require(manifest['runtime_contract']==training['runtime_contract'],'run/training runtime differs')
    require(training['runtime_contract']['experiment_id']==EXPERIMENT,'wrong experiment')
    require(training['policy_contract']['version']==POLICY and training['num_envs']==1,'wrong policy or unsupported topology')
    result=base.summarize(run)
    records=list(base.rows(run/'residual_and_projection_audit.jsonl'))
    updates=list(base.rows(run/'optimizer_updates.jsonl'))
    require(records and updates,'no completed learning data')
    require(result['new_decisions']==training['actual_policy_decisions'],'actual decision count differs')
    require(result['new_ppo_updates']==training['ppo_updates_this_run'],'actual update count differs')
    require(result['new_optimizer_steps']==training['optimizer_steps_this_run'],'actual optimizer-step count differs')
    require(result['last_global_decision']==training['global_policy_decisions'],'last counter differs')
    require(training['runner_config']['num_steps_per_env']==128,'old wrapper requires 128-step official rollout')
    for index,update in enumerate(updates):
        require(update['global_policy_decisions']==records[(index+1)*128-1]['global_policy_decision'],'update/sample boundary differs')
        if index:require(update['ppo_update']==updates[index-1]['ppo_update']+1,'nonconsecutive update IDs')
        require((run/f"rollouts/rollout_{update['ppo_update']:06d}.pt").is_file(),'missing saved real rollout')
        require((run/f"rollouts/update_{update['ppo_update']:06d}_likelihood.json").is_file(),'missing actual likelihood hook')
    require(all(r['applied_audit']['phase_id'] in PHASES for r in records),'invalid actual requested phase')
    require(all(r['policy_request']['policy_version']==POLICY for r in records),'mixed collection policy')
    require(all(r['applied_audit'].get('prefix_teacher_data_in_ppo_storage') is not True for r in records),'prefix data credited as learner data')
    checkpoints=training['checkpoints'];require(bool(checkpoints),'no saved checkpoint')
    final=checkpoints[-1];cp=Path(final['checkpoint']).resolve(strict=True);side=Path(final['manifest']).resolve(strict=True)
    require(cp.parent==project_root/'outputs'/('ppo_'+EXPERIMENT)/'checkpoints/history','checkpoint is outside new history')
    require(side==cp.with_name(cp.stem+'_manifest.json'),'checkpoint sidecar mismatch')
    metadata=read(side)
    require(metadata['save_load_round_trip'] is True and metadata['checkpoint_sha256']==sha(cp),'checkpoint round-trip/hash invalid')
    require(metadata['runtime_contract']==training['runtime_contract'] and metadata['policy_contract']==training['policy_contract'],'checkpoint/runtime/policy differs')
    require(Path(metadata['source_run']).resolve()==run,'checkpoint belongs to another run')
    require(metadata['global_policy_decisions']==result['last_global_decision']==final['global_policy_decisions'],'checkpoint counter mismatch')
    require(metadata['ppo_updates']==updates[-1]['ppo_update'],'checkpoint update mismatch')
    ancestry=metadata['resume_ancestry']
    require(ancestry['source_global_policy_decisions']==result['first_global_decision']-1,'resume decision ancestry mismatch')
    require(metadata['ppo_updates']-ancestry['source_ppo_updates']==len(updates),'resume update ancestry mismatch')
    require(metadata['optimizer_steps']-ancestry['source_optimizer_steps']==result['new_optimizer_steps'],'resume optimizer ancestry mismatch')
    origin=metadata['task_conditioned_hip_wheel_branch']['counter_origin'];counts=metadata['task_conditioned_hip_wheel_branch_counts']
    require(all(counts[k]==metadata[k]-origin[k] and counts[k]>0 for k in ('global_policy_decisions','ppo_updates','optimizer_steps')),'invalid new-branch counts')
    telemetry=training['telemetry'];core=telemetry['core']
    require(telemetry['policy_decisions']==core['decisions']==len(records),'telemetry credits differ')
    suffix=training['phase_suffix_curriculum_implemented']
    if suffix:
        attempts=core['prefix_attempts'];prefix_decisions=core['prefix_behavior_decisions'];prefix_ticks=core['prefix_physics_ticks']
        require(sum(x['prefix_decisions'] for x in attempts)==prefix_decisions,'prefix attempt decision totals differ')
        require(sum(x['prefix_physics_ticks'] for x in attempts)==prefix_ticks,'prefix attempt tick totals differ')
    else:
        require(not core.get('prefix_behavior_decisions',0),'unexpected prefix in natural P01 run')
        attempts=[];prefix_decisions=prefix_ticks=0
    prefix_log=run/'prefix_evidence.jsonl';prefix_log_count=None
    if prefix_log.exists():
        prefix_records=list(base.rows(prefix_log))
        require(all(r.get('policy_credit') is False for r in prefix_records),'prefix log contains learner credit')
        prefix_log_count=sum(r.get('kind') in ('reset_only_prefix_decision','checkpoint_prefix_decision') for r in prefix_records)
        require(prefix_log_count==prefix_decisions,'prefix persistent log differs from telemetry')
    geometry={};profiles=collections.Counter();epsilon=collections.Counter();terminal=[]
    for row in records:
        info=row['applied_audit'];reward=info['reward_breakdown']
        profiles[reward['objective_profile']]+=1;epsilon[str(reward['quality_epsilon'])]+=1
        require(reward['objective_profile']==OBJECTIVE,'old/mixed reward objective')
        if row['terminal']:terminal.append(outcome(row))
        for sample in reward['task_space_quality_sample_audit']:
            phase=sample['phase'];require(phase in PHASES,'unknown geometry sample phase')
            part=geometry.setdefault(phase,dict(physics_samples=0,eligible_samples=0,valid_eligible_samples=0,
                terminal_measurement_omitted=0,known_weighted_geometry_cost=0.,valid_duration_s=0.,
                beta_min_per_s=None,beta_max_per_s=None,raw_geometry_cost_integral=0.))
            part['physics_samples']+=1
            if not sample['eligible']:continue
            part['eligible_samples']+=1
            beta=sample['effective_beta_per_s'];require(math.isfinite(beta) and beta>=0,'invalid actual geometry coefficient')
            part['beta_min_per_s']=beta if part['beta_min_per_s'] is None else min(beta,part['beta_min_per_s'])
            part['beta_max_per_s']=beta if part['beta_max_per_s'] is None else max(beta,part['beta_max_per_s'])
            if sample['valid']:
                cost=sample['weighted_geometry_cost'];expected=beta*sample['raw_geometry_cost']*sample['dt_s']
                require(math.isclose(cost,expected,abs_tol=1e-12,rel_tol=1e-10),'logged geometry arithmetic differs')
                part['valid_eligible_samples']+=1;part['valid_duration_s']+=sample['dt_s']
                part['known_weighted_geometry_cost']+=cost
                part['raw_geometry_cost_integral']+=sample['raw_geometry_cost']*sample['dt_s']
            else:
                require(sample['weighted_geometry_cost'] is None and sample['terminal_measurement_omitted'] is True,'invalid geometry disguised as zero')
                part['terminal_measurement_omitted']+=1
    episodes=list(base.rows(run/'completed_episodes.jsonl'))
    require(len(terminal)==len(episodes)==telemetry['completed_episode_count'],'terminal ledger counts differ')
    for event,episode in zip(terminal,episodes):
        require(event['termination_reason']==episode['termination_reason'] and event['task_success']==episode['task_success'],'terminal ledgers differ')
    full_success=sum(x.get('full_task_success') is True for x in episodes)
    suffix_success=sum(x.get('task_success') is True and x.get('full_task_success') is not True for x in episodes)
    require(full_success==telemetry['success_count'],'natural full-success ledger differs')
    front_cost=sum(v['actual_quality_cost'] for v in result['actual_front_quality_by_task_substate'].values())
    geometry_cost=sum(v['known_weighted_geometry_cost'] for v in geometry.values())
    body_contribution=sum(v.get('body_stability',0.) for v in result['signed_family_contribution_sums'].values())
    require(math.isclose(body_contribution,-front_cost-geometry_cost,abs_tol=1e-8,rel_tol=1e-8),'actual body contribution differs from front+geometry audit')
    advantages=list(base.rows(run/'advantage_audit.jsonl'))
    require([x['ppo_update_intended'] for x in advantages]==[x['ppo_update'] for x in updates],'advantage/update IDs differ')
    result.update(schema='wlr50_clean.task_conditioned_sealed_training_receipt.v1',
        actual_request_phase_coverage={p:result['actual_request_phase_coverage'].get(p,0) for p in PHASES},
        requested_sampling=training['implemented_sampling'],coverage_source='actual learner request phases only; not requested curriculum, prefix, diagnostic or video counts',
        prefix=dict(behavior_decisions=prefix_decisions,physics_ticks=prefix_ticks,learner_credit=0,attempts=attempts,
            persistent_decision_log_count=prefix_log_count,source='training_manifest.telemetry.core; persistent prefix log crosscheck when present',
            provenance=core.get('prefix_policy_provenance')),
        final_checkpoint=dict(path=str(cp),manifest=str(side),sha256=metadata['checkpoint_sha256'],save_load_round_trip=True,
            current_run_checkpoint_list=checkpoints,branch=metadata['task_conditioned_hip_wheel_branch'],branch_counts=counts,
            lifetime_counters={k:metadata[k] for k in ('global_policy_decisions','ppo_updates','optimizer_steps')}),
        objective_profiles=dict(profiles),recorded_quality_epsilon_counts=dict(epsilon),actual_geometry_quality_by_phase=geometry,
        actual_quality_contributions=dict(front_cost=front_cost,geometry_cost_known=geometry_cost,signed_body_stability=body_contribution,
            arithmetic_verified=True,unknown_terminal_geometry_is_null_not_measured_zero=True),
        terminal_outcomes=terminal,first_terminal_outcome=terminal[0] if terminal else None,last_sample_outcome=outcome(records[-1]),
        natural_P01_full_task_success_count=full_success,suffix_success_count=suffix_success,
        advantage_by_update=[{k:x.get(k) for k in ('ppo_update_intended','stored_advantage_semantics','overall','by_request_phase','terminal_samples')} for x in advantages],
        input_sha256={str(path):sha(path) for path in (run/'run_manifest.json',run/'training_manifest.json',run/'optimizer_updates.jsonl',side)},
        no_new_policy_training_or_physics=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();receipt=summarize(args.run)
    with args.output.open('x',encoding='utf-8') as stream:json.dump(receipt,stream,indent=2,allow_nan=False)
    print(json.dumps({k:receipt[k] for k in ('new_decisions','new_ppo_updates','new_optimizer_steps',
        'actual_request_phase_coverage','actual_quality_contributions','natural_P01_full_task_success_count',
        'suffix_success_count','last_sample_outcome')},indent=2))
