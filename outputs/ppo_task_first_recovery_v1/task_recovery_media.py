"""Output-only task-recovery media adapter; immutable source and original zero.

Reuses the proven complete decode/PTS/remux/compare implementation. Different
runtime versions are disclosed and accepted only after explicit invariant review.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
spec=importlib.util.spec_from_file_location('task_media_existing_after',ROOT/'outputs/ppo_timing_task_priority_v1/after_media.py')
prior=importlib.util.module_from_spec(spec); spec.loader.exec_module(prior)
base=prior.base
OUT=Path(__file__).resolve().parent
OLD_ZERO=ROOT/'outputs/ppo_height_and_p02_recovery_v1/videos/zero_RLminus3_final_stop_latest.media.json'
ALLOWED_CHANGED_CODE={f'src/wlr50_clean/ppo/{name}.py' for name in (
    'semantic_reward','semantic_training','semantic_migration','semantic_cli','semantic_video_cli',
    'semantic_video','semantic_checkpoint_prefix')}
ALLOWED_CHANGED_CODE|={'scripts/run_semantic_ppo.ps1','scripts/run_semantic_video.ps1','scripts/initialize_task_recovery_branch.py'}
CONFIG_FIELDS={'action_schema_path':'action_schema.json','execution_profile':'execution_profile.yaml',
    'observation_schema_path':'observation_schema.json','quality_score_path':'quality_score.yaml',
    'reward_config_path':'reward_config.yaml','task_spec_path':'stage_task_spec.yaml'}
COUNTERS=('global_policy_decisions','ppo_updates','optimizer_steps')
TASK_ORIGIN={'global_policy_decisions':174592,'ppo_updates':1329,'optimizer_steps':26580}
EXECUTION_FIX_CODE={f'src/wlr50_clean/ppo/{name}.py' for name in
    ('semantic_residual_adapter','semantic_nominal_geometry')}
EXECUTION_FIX_SCHEMA='wlr50_clean.independent_post_mapper_residual_same372_fix.v1'
ZERO_REPLAY=OUT/'retained_zero_geometry_replay_receipt.json'
FINAL_STOP_CODE={'src/wlr50_clean/ppo/semantic_supervisor.py'}
FINAL_STOP_MIGRATION_CODE=FINAL_STOP_CODE|{f'src/wlr50_clean/ppo/{name}.py' for name in ('semantic_migration','semantic_training')}
FINAL_STOP_SCHEMA='wlr50_clean.final_stop_handoff_same372_fix.v1'
FINAL_STOP_ZERO_REPLAY=OUT/'retained_zero_p13_stop_replay_receipt.json'
QUARTER_CODE={f'src/wlr50_clean/ppo/{name}.py' for name in ('semantic_history_actor','semantic_policy_distribution')}
QUARTER_REVIEW_CODE=QUARTER_CODE|{f'src/wlr50_clean/ppo/{name}.py' for name in ('semantic_cli','semantic_migration','semantic_training')}
HALF_POLICY='history_conditioned_heteroscedastic_log_temperature_v1'
QUARTER_POLICY='history_conditioned_heteroscedastic_log_temperature_quarter_v1'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def contract_digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def bound_checkpoint_metadata(binding):
    """Use official-load source binding; do not reload/re-hash model tensors."""
    base.require(isinstance(binding,dict),'Missing immutable source checkpoint binding')
    path=Path(binding['manifest'])
    base.require(path.parent.name=='history' and sha(path)==binding['manifest_sha256'],'Source checkpoint manifest is not immutable/bound')
    metadata=base.read_json(path)
    base.require(Path(metadata['checkpoint_path']).resolve()==Path(binding['checkpoint']).resolve()
        and metadata['checkpoint_sha256']==binding['checkpoint_sha256'],'Checkpoint/manifest binding differs')
    return metadata

def learned_checkpoint_evidence(proof):
    """Initialization may retain large lifetime counters, but is not learning."""
    base.require(proof.get('checkpoint_loaded_and_verified') is True,'No verified official checkpoint load')
    metadata=bound_checkpoint_metadata(proof.get('source'))
    base.require(metadata['global_policy_decisions']==proof.get('saved_global_policy_decisions'),'Saved count differs from actual manifest')
    base.require(metadata.get('actor_parameter_sha256')==proof.get('parameter_hashes',{}).get('actor_parameter_sha256'),'Loaded actor differs from manifest')
    base.require(metadata.get('stage')!='initial_task_recovery_mean_head'
        and not Path(metadata['checkpoint_path']).name.startswith('checkpoint_initial_'),'Mean-head initialization is not trained policy evidence')
    branch=metadata.get('task_recovery_branch')
    if branch is not None:
        counts=metadata.get('task_recovery_branch_counts',{})
        origin=branch.get('counter_origin',{})
        base.require(all(type(counts.get(k)) is int and counts[k]>0 for k in COUNTERS),
            'Recovery branch must have actual post-initialization decisions, PPO updates and optimizer steps')
        base.require(all(type(origin.get(k)) is int and metadata[k]-origin[k]==counts[k] for k in COUNTERS),
            'Recovery branch counters do not equal actual lifetime-minus-origin counts')
        return {'mode':'learned_mean_head_recovery_branch','branch_id':branch.get('branch_id'),
            'origin':origin,'new_learning_counts':counts,'source_manifest':proof['source']['manifest'],
            'initialization_alone_is_learning_success':False}
    # Reward-only continuation: follow existing immutable ancestry to the
    # known CP174592 start, never infer learning from a filename alone.
    delta={k:metadata[k]-TASK_ORIGIN[k] for k in COUNTERS}
    base.require(all(type(v) is int and v>0 for v in delta.values()),'No new task-recovery optimizer training since CP174592')
    cursor=metadata; ancestors=[]; seen=set()
    while cursor['global_policy_decisions']>174592:
        binding=cursor.get('resume_ancestry',{}).get('source_checkpoint')
        base.require(isinstance(binding,dict) and binding.get('manifest') not in seen,'Missing/cyclic recovery checkpoint ancestry')
        seen.add(binding['manifest']); ancestors.append(binding['manifest'])
        previous=bound_checkpoint_metadata(binding)
        base.require(previous['global_policy_decisions']<cursor['global_policy_decisions'],'Reward-only ancestry must contain real training progress')
        cursor=previous
    base.require(all(cursor[k]==TASK_ORIGIN[k] for k in COUNTERS),'Recovery ancestry does not reach preserved CP174592')
    return {'mode':'learned_reward_only_continuation','origin':TASK_ORIGIN,'new_learning_counts':delta,
        'source_manifest':proof['source']['manifest'],'verified_ancestry_manifests':ancestors,
        'initialization_alone_is_learning_success':False}

def execution_factor_ancestry(metadata,target_contract):
    """Read immutable lineage; a file allowlist alone never proves migration."""
    cursor=metadata; ancestors=[]; seen=set()
    while True:
        ancestry=cursor.get('resume_ancestry') or {}
        migration=ancestry.get('resume_migration') or {}
        factor=migration.get('execution_composition_factor')
        binding=ancestry.get('source_checkpoint')
        base.require(isinstance(binding,dict) and binding.get('manifest') not in seen,
            'Unreviewed execution fix: missing/cyclic explicit migration ancestry')
        seen.add(binding['manifest']); ancestors.append(binding['manifest'])
        previous=bound_checkpoint_metadata(binding)
        if factor is None:
            cursor=previous; continue
        base.require(factor.get('schema')==EXECUTION_FIX_SCHEMA
            and factor.get('nominal_history_semantics')=='same_actual_state_nominal_command_history_excludes_policy_v1',
            'Unreviewed execution composition schema/semantics')
        base.require(all(factor.get(k) is False for k in ('physical_scene_changed','actuator_capability_changed',
            'nominal_source_schedule_changed','task_acceptance_changed','reward_changed','action_ranges_changed',
            'old_rollout_inherited','trajectory_equivalence_claimed')),'Execution fix alters protected task/physics/config')
        before,after=previous['runtime_contract'],cursor['runtime_contract']
        base.require(ancestry.get('source_runtime_contract')==before,'Execution migration source runtime differs')
        base.require(migration['source_contract_sha256']==contract_digest(before)
            and migration['target_contract_sha256']==contract_digest(after),'Execution contract digest mismatch')
        base.require(Path(migration['source_checkpoint']).resolve()==Path(binding['checkpoint']).resolve()
            and migration['source_checkpoint_sha256']==binding['checkpoint_sha256']
            and migration['source_manifest_sha256']==binding['manifest_sha256'],'Execution migration checkpoint binding mismatch')
        plan_path=Path(migration['plan_path'])
        base.require(sha(plan_path)==migration['plan_sha256'],'Execution migration plan changed')
        base.require(base.read_json(plan_path)=={k:v for k,v in migration.items() if k not in ('plan_path','plan_sha256')},
            'Saved execution migration differs from its plan')
        delta={k:{'before':before['files'].get(k),'after':after['files'].get(k)}
            for k in before['files'].keys()|after['files'].keys() if before['files'].get(k)!=after['files'].get(k)}
        base.require(delta==migration['changed_file_hashes'] and set(delta)==set(migration['allowed_changed_files'])
            and set(delta)<=ALLOWED_CHANGED_CODE|EXECUTION_FIX_CODE
            and set(delta)&EXECUTION_FIX_CODE,'Execution migration changed-file inventory differs')
        base.require(factor['reviewed_code_sha256']=={k:v['after'] for k,v in delta.items()},'Execution reviewed hashes differ')
        base.require(all(after['files'][k]==target_contract['files'][k] for k in EXECUTION_FIX_CODE),
            'Formal runtime core execution fix differs from verified migration')
        base.require(before['selected_configuration']==after['selected_configuration']==target_contract['selected_configuration'],
            'Execution migration changes selected configuration')
        return {'verified_ancestry_manifests':ancestors,'migration_plan':str(plan_path),
            'migration_plan_sha256':migration['plan_sha256'],'execution_composition_factor':factor,
            'source_contract_sha256':migration['source_contract_sha256'],
            'target_contract_sha256':migration['target_contract_sha256']}

def execution_fix_comparison_evidence(left,right):
    source=Path(right['source_manifest'])
    base.require(sha(source)==right['source_manifest_sha256'],'Formal video source changed')
    manifest=base.read_json(source); proof=manifest['checkpoint_load_provenance']
    base.require(manifest['runtime_contract']==right['runtime_contract'],'Unreviewed execution fix: formal video runtime differs')
    metadata=bound_checkpoint_metadata(proof['source'])
    base.require(metadata['runtime_contract']==right['runtime_contract'],'Formal checkpoint/runtime differs')
    migration=execution_factor_ancestry(metadata,right['runtime_contract'])
    replay=base.read_json(ZERO_REPLAY)
    base.require(replay['schema']=='retained_zero_geometry_explicit_nominal_history_replay.v1'
        and replay['source_manifest_sha256']==left['source_manifest_sha256'],'Zero replay refers to another baseline')
    base.require(replay['replayed_geometry_active_counts']=={'P09':717,'P12':431}
        and all(replay.get(k) is True for k in ('geometry_adjusted_target_exact_numeric_equality',
            'geometry_status_exact_equality','actuator_target_float32_exact_equality','no_simulator','no_optimizer'))
        and replay['geometry_adjusted_target_max_error_deg_or_rad_s']==0,'Zero recorded replay failed')
    fixture_path=Path(replay['fixture'])
    base.require(sha(fixture_path)==replay['fixture_sha256'],'Zero replay fixture changed')
    fixture=base.read_json(fixture_path)
    base.require(fixture['runtime_contract']==left['runtime_contract']
        and fixture['source_manifest_sha256']==left['source_manifest_sha256'],'Replay fixture is not retained zero')
    base.require(replay['new_geometry_source_sha256']==right['runtime_contract']['files']['src/wlr50_clean/ppo/semantic_nominal_geometry.py'],
        'Zero replay geometry hash differs from formal execution fix')
    return {**migration,'zero_computational_replay_receipt':str(ZERO_REPLAY),
        'zero_computational_replay_receipt_sha256':sha(ZERO_REPLAY),'zero_replay_fixture_sha256':sha(fixture_path),
        'zero_replay_scope':replay['scope'],'new_paired_physical_B_rerun':False,
        'comparison_scope':'retained old-build successful ZERO versus learned execution-history-fix PPO; same N source schedule/physics, not same execution build',
        'zero_trajectory_equivalence_not_proven_by_offline_replay':True}

def final_stop_factor_ancestry(metadata,target_contract):
    """Independent P13 factor; never enlarge execution_composition_factor."""
    cursor=metadata; ancestors=[]; seen=set()
    while True:
        ancestry=cursor.get('resume_ancestry') or {}
        migration=ancestry.get('resume_migration') or {}
        factor=migration.get('final_stop_handoff_factor')
        binding=ancestry.get('source_checkpoint')
        base.require(isinstance(binding,dict) and binding.get('manifest') not in seen,
            'Unreviewed P13 handoff: missing/cyclic independent migration ancestry')
        seen.add(binding['manifest']); ancestors.append(binding['manifest'])
        previous=bound_checkpoint_metadata(binding)
        if factor is None:
            cursor=previous; continue
        base.require(factor.get('schema')==FINAL_STOP_SCHEMA and factor.get('stop_owner_acquisition_semantics')==
            'post_window_triggered_nominal_stop_takeover_v2','Unreviewed P13 handoff schema/semantics')
        base.require(all(factor.get(k) is False for k in ('task_evaluator_changed','reward_changed',
            'fixed_post_completion_window_changed','physical_scene_changed','actuator_capability_changed',
            'residual_composition_changed','controller_history_reset_on_phase_transition',
            'recorded_FSM_source_changed','task_acceptance_changed','action_ranges_changed','kernel_changed',
            'old_rollout_inherited','trajectory_equivalence_claimed')),
            'P13 handoff changes protected evaluator/window/reward/physics')
        scope=factor.get('supervisor_scope') or {}
        base.require(scope.get('method')=='NominalMotionProvider._observe_final_stop_owner'
            and scope.get('method_signature_unchanged') is True
            and scope.get('evaluator_and_other_methods_unchanged') is True
            and all(isinstance(scope.get(k),str) and len(scope[k])==64 for k in
                ('unchanged_outside_method_sha256','source_method_sha256','target_method_sha256')),
            'P13 handoff lacks explicit method-only scope proof')
        before,after=previous['runtime_contract'],cursor['runtime_contract']
        base.require(ancestry.get('source_runtime_contract')==before,'P13 handoff source runtime differs')
        base.require(migration['source_contract_sha256']==contract_digest(before)
            and migration['target_contract_sha256']==contract_digest(after)
            and migration['source_runtime_content_sha256']==before['runtime_content_sha256']
            and migration['target_runtime_content_sha256']==after['runtime_content_sha256'],
            'P13 handoff runtime/contract digest mismatch')
        base.require(Path(migration['source_checkpoint']).resolve()==Path(binding['checkpoint']).resolve()
            and migration['source_checkpoint_sha256']==binding['checkpoint_sha256']
            and migration['source_manifest_sha256']==binding['manifest_sha256'],'P13 handoff checkpoint binding mismatch')
        path=Path(migration['plan_path'])
        base.require(sha(path)==migration['plan_sha256'] and base.read_json(path)==
            {k:v for k,v in migration.items() if k not in ('plan_path','plan_sha256')},'P13 handoff plan changed')
        delta={k:{'before':before['files'].get(k),'after':after['files'].get(k)}
            for k in before['files'].keys()|after['files'].keys() if before['files'].get(k)!=after['files'].get(k)}
        base.require(delta==migration['changed_file_hashes'] and set(delta)==set(migration['allowed_changed_files'])
            and set(delta)<=FINAL_STOP_MIGRATION_CODE and FINAL_STOP_CODE<=set(delta),
            'P13 handoff exceeds independent three-file inventory')
        base.require(factor['reviewed_code_sha256']=={k:v['after'] for k,v in delta.items()},'P13 handoff reviewed hashes differ')
        base.require(all(after['files'][k]==target_contract['files'][k] for k in FINAL_STOP_CODE),
            'Formal runtime P13 supervisor differs from verified handoff')
        base.require(before['selected_configuration']==after['selected_configuration']==target_contract['selected_configuration'],
            'P13 handoff changes selected configuration')
        return {'verified_ancestry_manifests':ancestors,'migration_plan':str(path),
            'migration_plan_sha256':migration['plan_sha256'],'final_stop_handoff_factor':factor,
            'source_contract_sha256':migration['source_contract_sha256'],'target_contract_sha256':migration['target_contract_sha256'],
            'source_runtime_content_sha256':migration['source_runtime_content_sha256'],
            'target_runtime_content_sha256':migration['target_runtime_content_sha256']}

def final_stop_comparison_evidence(left,right):
    source=Path(right['source_manifest'])
    base.require(sha(source)==right['source_manifest_sha256'],'P13 formal video source changed')
    manifest=base.read_json(source)
    base.require(manifest['runtime_contract']==right['runtime_contract'],'Unreviewed P13 fix: formal runtime differs')
    metadata=bound_checkpoint_metadata(manifest['checkpoint_load_provenance']['source'])
    base.require(metadata['runtime_contract']==right['runtime_contract'],'P13 formal checkpoint/runtime differs')
    migration=final_stop_factor_ancestry(metadata,right['runtime_contract'])
    replay=base.read_json(FINAL_STOP_ZERO_REPLAY)
    base.require(replay['schema']=='wlr50_clean.retained_zero_p13_stop_replay.v1'
        and replay['baseline_source_manifest_sha256']==left['source_manifest_sha256']
        and Path(replay['baseline_source_manifest']).resolve()==Path(left['source_manifest']).resolve(),
        'P13 replay is not bound to retained zero')
    base.require(replay['all_checks_passed'] is True and replay['checks']
        and all(v is True for v in replay['checks'].values())
        and replay['complete_P13_input_rows']==2130 and replay['recorded_P13_dispatch_rows']==2129
        and replay['first_P13_observation_tick']==6728 and replay['last_P13_observation_tick']==8857
        and replay['first_new_owner_tick']==replay['first_old_owner_tick']==replay['first_post_window_tick']==8737
        and replay['max_old_new_nominal_error']==replay['max_old_recorded_nominal_error']==0
        and replay['first_difference'] is None,'P13 recorded-zero command replay failed')
    base.require(replay['not_a_new_physical_run'] is True and replay['physical_inputs_unmodified'] is True
        and replay['optimizer_updates']==replay['physics_steps']==0,'P13 replay cannot be relabelled physical B')
    supervisor='src/wlr50_clean/ppo/semantic_supervisor.py'
    base.require(replay['old_supervisor_sha256']==left['runtime_contract']['files'][supervisor]
        and replay['new_supervisor_sha256']==right['runtime_contract']['files'][supervisor],
        'P13 zero replay supervisor hashes differ from actual compared runtimes')
    return {**migration,'zero_P13_replay_receipt':str(FINAL_STOP_ZERO_REPLAY),
        'zero_P13_replay_receipt_sha256':sha(FINAL_STOP_ZERO_REPLAY),
        'zero_P13_replay_scope':'2130 recorded P13 inputs and 2129 dispatches preserve nominal/controller/tracking and owner tick; no new physical result',
        'new_paired_physical_B_rerun':False,'same_execution_build_claim':False,
        'comparison_scope':'retained old ZERO versus learned residual-history plus P13 stop-handoff build; N source schedule/physics/evaluator retained'}

def quarter_factor_ancestry(metadata,target_contract):
    """Sampling-kernel migration is distinct from execution and P13 fixes."""
    base.require(metadata.get('policy_contract',{}).get('version')==QUARTER_POLICY
        and metadata['runtime_contract']['experiment_id']=='task_first_recovery_v1','Unreviewed quarter policy/namespace')
    cursor=metadata; ancestors=[]; seen=set()
    while True:
        ancestry=cursor.get('resume_ancestry') or {}
        migration=ancestry.get('resume_migration') or {}
        factor=migration.get('exploration_temperature_factor')
        binding=ancestry.get('source_checkpoint')
        base.require(isinstance(binding,dict) and binding.get('manifest') not in seen,'Missing/cyclic quarter migration ancestry')
        seen.add(binding['manifest']); ancestors.append(binding['manifest'])
        previous=bound_checkpoint_metadata(binding)
        if factor is None:
            cursor=previous; continue
        base.require(factor.get('schema')=='wlr50_clean.history372_innovation_temperature_continuation.v1'
            and factor.get('source_policy_version')==HALF_POLICY and factor.get('target_policy_version')==QUARTER_POLICY
            and factor.get('source_exploration_std_temperature')==.5
            and factor.get('target_exploration_std_temperature')==.25,'Not the reviewed half-to-quarter transition')
        base.require(factor.get('kernel_changed') is True and all(factor.get(k) is False for k in
            ('physical_mdp_changed','nominal_control_changed','reward_changed','task_acceptance_changed','action_ranges_changed'))
            and factor.get('observation_semantics_changed')==[], 'Quarter temperature alters protected task/control semantics')
        base.require(factor.get('deterministic_same_weights_same_observation')=='exact_original_conditional_mean_path',
            'Quarter migration does not preserve deterministic mean path')
        scopes=factor.get('code_scope',{})
        actor='src/wlr50_clean/ppo/semantic_history_actor.py'; cli='src/wlr50_clean/ppo/semantic_cli.py'
        base.require(scopes.get(actor,{}).get('reviewed_regions')==['SemanticQuarterTemperedHistoryMLPModel']
            and scopes.get(cli,{}).get('reviewed_regions')==[]
            and all(row.get('protected_ast_identical') is True for row in scopes.values()),
            'Quarter scope changes old half actor/HISTORY/CLI')
        before,after=previous['runtime_contract'],cursor['runtime_contract']
        base.require(previous.get('policy_contract')==factor['source_policy_contract']
            and cursor.get('policy_contract')==factor['target_policy_contract'],'Quarter checkpoint policy contract mismatch')
        base.require(ancestry.get('source_runtime_contract')==before
            and migration['source_contract_sha256']==contract_digest(before)
            and migration['target_contract_sha256']==contract_digest(after),'Quarter runtime contract mismatch')
        base.require(Path(migration['source_checkpoint']).resolve()==Path(binding['checkpoint']).resolve()
            and migration['source_checkpoint_sha256']==binding['checkpoint_sha256']
            and migration['source_manifest_sha256']==binding['manifest_sha256'],'Quarter immutable checkpoint mismatch')
        path=Path(migration['plan_path'])
        base.require(sha(path)==migration['plan_sha256'] and base.read_json(path)==
            {k:v for k,v in migration.items() if k not in ('plan_path','plan_sha256')},'Quarter migration plan changed')
        delta={k:{'before':before['files'].get(k),'after':after['files'].get(k)}
            for k in before['files'].keys()|after['files'].keys() if before['files'].get(k)!=after['files'].get(k)}
        base.require(delta==migration['changed_file_hashes'] and set(delta)==set(migration['allowed_changed_files'])
            and QUARTER_CODE<=set(delta) and set(delta)<=QUARTER_REVIEW_CODE-{cli},'Quarter migration exceeds exact unchanged-CLI boundary')
        base.require(factor['reviewed_code_sha256']=={k:after['files'][k] for k in QUARTER_REVIEW_CODE},
            'Quarter reviewed source hashes differ')
        base.require(all(after['files'][k]==target_contract['files'][k] for k in QUARTER_CODE),
            'Formal quarter kernel differs from reviewed migration')
        base.require(before['selected_configuration']==after['selected_configuration']==target_contract['selected_configuration'],
            'Quarter continuation changes six configuration bindings')
        return {'verified_ancestry_manifests':ancestors,'migration_plan':str(path),
            'migration_plan_sha256':migration['plan_sha256'],'exploration_temperature_factor':factor,
            'source_checkpoint':binding,'source_contract_sha256':migration['source_contract_sha256'],
            'target_contract_sha256':migration['target_contract_sha256'],
            'stochastic_kernel_changed':True,'deterministic_evaluation_uses_saved_conditional_mean':True,
            'evaluation_time_action_scaling':False,'post_training_mean_or_trajectory_identity_claim':False,
            'new_paired_physical_B_rerun':False}

def quarter_comparison_evidence(right):
    source=Path(right['source_manifest'])
    base.require(sha(source)==right['source_manifest_sha256'],'Quarter formal source changed')
    manifest=base.read_json(source)
    base.require(manifest['runtime_contract']==right['runtime_contract'],'Unreviewed quarter formal runtime differs')
    metadata=bound_checkpoint_metadata(manifest['checkpoint_load_provenance']['source'])
    base.require(metadata['runtime_contract']==right['runtime_contract'],'Quarter checkpoint/runtime differs')
    return quarter_factor_ancestry(metadata,right['runtime_contract'])

def same_task_invariants(left,right):
    """No blanket runtime waiver: compare frozen controls and parsed configs."""
    a,b=left['runtime_contract'],right['runtime_contract']
    base.require(a['experiment_id']=='fsm_reference_p09_stable_v2' and b['experiment_id']=='task_first_recovery_v1','Unexpected experiment pairing')
    variable={'files','source_git_commit','runtime_content_sha256','selected_configuration','experiment_id'}
    invariant_a={k:v for k,v in a.items() if k not in variable}
    invariant_b={k:v for k,v in b.items() if k not in variable}
    base.require(invariant_a==invariant_b,'Runtime physics/package/frozen A invariants differ')
    configuration=[]; selected_paths=set()
    for field,name in CONFIG_FIELDS.items():
        data=[]
        for item,contract in ((left,a),(right,b)):
            binding=item['evaluation_configuration'][field]
            path=Path(binding['path']); relative=path.relative_to(ROOT).as_posix()
            selected_paths.add(relative)
            base.require(sha(path)==binding['sha256']==contract['files'][relative],'Selected config no longer matches immutable run')
            base.require(contract['selected_configuration'][name]=={'path':relative,'sha256':binding['sha256']},'Config mapping does not match source manifest')
            data.append(yaml.safe_load(path.read_text(encoding='utf8')))
        same=data[0]==data[1]
        if name!='reward_config.yaml':base.require(same,f'Physical/N/mapper/task config changed: {name}')
        configuration.append({'name':name,'parsed_content_equal':same,'left':left['evaluation_configuration'][field],'right':right['evaluation_configuration'][field]})
    inventory_a={k:v for k,v in a['files'].items() if k not in selected_paths}
    inventory_b={k:v for k,v in b['files'].items() if k not in selected_paths}
    changes={k:{'left_sha256':inventory_a.get(k),'right_sha256':inventory_b.get(k)}
        for k in sorted(inventory_a.keys()|inventory_b.keys()) if inventory_a.get(k)!=inventory_b.get(k)}
    extra=set(changes)-ALLOWED_CHANGED_CODE
    base.require(extra<=EXECUTION_FIX_CODE|FINAL_STOP_CODE|QUARTER_CODE,f'Unreviewed control/physics/source change: {extra}')
    execution=None; handoff=None; quarter=None
    if extra&EXECUTION_FIX_CODE:
        base.require('source_manifest' in right,'Unreviewed execution fix: no formal source binding')
        execution=execution_fix_comparison_evidence(left,right)
    if extra&FINAL_STOP_CODE:
        base.require('source_manifest' in right,'Unreviewed P13 fix: no formal source binding')
        handoff=final_stop_comparison_evidence(left,right)
    if extra&QUARTER_CODE:
        base.require('source_manifest' in right,'Unreviewed quarter policy: no formal source binding')
        quarter=quarter_comparison_evidence(right)
    base.require(left['camera']==right['camera'] and left['seed']==right['seed'],'Camera/seed differ')
    audit={'schema':'wlr50_clean.task_recovery_same_N_physics_comparison.v1',
        'same_build_claim':False,'same_reward_claim':False,'same_N_mapper_physical_evaluator_config':True,
        'unchanged_other_runtime_files':True,'code_changes':changes,'configuration':configuration,
        'code_change_review_scope':'task reward/profile routing, actual minibatch diagnostics, reset-only curriculum/explicit recovery branch support; no shared nominal/mapper/physics/evaluator code changed',
        'left_binding':{k:a[k] for k in ('source_git_commit','runtime_content_sha256','experiment_id')},
        'right_binding':{k:b[k] for k in ('source_git_commit','runtime_content_sha256','experiment_id')}}
    if execution is not None:
        audit.update(execution_composition_comparison=execution,same_execution_composition_claim=False,
            code_change_review_scope='explicit checkpoint execution_composition_factor plus exact recorded-zero geometry replay; shared residual-history execution changed, N source/physics/evaluator unchanged',
            new_paired_physical_B_rerun=False)
    if handoff is not None:
        audit.update(final_stop_handoff_comparison=handoff,same_execution_composition_claim=False,
            code_change_review_scope='independent execution_composition_factor and final_stop_handoff_factor with their separate recorded-zero replays; execution/P13 handoff build changed, N source/physics/evaluator retained',
            new_paired_physical_B_rerun=False)
    if quarter is not None:
        audit.update(quarter_sampling_comparison=quarter,same_stochastic_kernel_claim=False,
            code_change_review_scope='separate execution/P13 fixes and explicit half-to-quarter sampling migration; deterministic evaluation uses saved mean without temporary scaling, N source/physics/evaluator retained',
            new_paired_physical_B_rerun=False)
    projection={'schema':'reviewed_invariant_encoder_projection_not_runtime_identity','invariants':invariant_a,
        'unchanged_files':{k:v for k,v in inventory_a.items() if k not in changes},
        'same_N_mapper_physics_task_config':True,'same_build':False}
    return audit,projection

def export(source,output):
    source=source.resolve(); output=output.resolve()
    manifest=base.read_json(source/'semantic_video_source_manifest.json')
    run=base.read_json(source.parent/'run_manifest.json')
    base.require(run.get('completed_at_utc') and run.get('lifecycle') not in (None,'STARTED','RUNNING'),'Run not sealed')
    base.require(run['runtime_contract']==manifest['runtime_contract'],'Run/source runtime mismatch')
    base.require(manifest['runtime_contract']['experiment_id']=='task_first_recovery_v1','Not task recovery namespace')
    proof=manifest.get('checkpoint_load_provenance') or {}
    learning=learned_checkpoint_evidence(proof)
    metadata=bound_checkpoint_metadata(proof['source'])
    quarter=(quarter_factor_ancestry(metadata,manifest['runtime_contract'])
        if metadata.get('policy_contract',{}).get('version')==QUARTER_POLICY else None)
    base.export(source,output,'C0')
    prior.annotate_new_receipt(output.with_suffix('.media.json'),{
        'revision_scope':'task_first_recovery_v1','base_media_role':'C0','label':'PPO TASK RECOVERY',
        'same_build_as_preserved_zero_claim':False,'official_full12_evaluation':True,
        'export_adapter':str(Path(__file__).resolve()),'export_adapter_sha256':sha(__file__),
        'checkpoint_decisions':proof['saved_global_policy_decisions'],'initialization_alone_is_learning_success':False,
        'actual_new_learning_evidence':learning,
        **({'quarter_sampling_evidence':quarter} if quarter is not None else {})})

def compare(left_path,right_path,output):
    left,right=[base.read_json(p) for p in (left_path,right_path)]
    base.require(left.get('base_media_role')=='B0' and left.get('physical_task_success') is True,'Not retained successful zero')
    base.require(right.get('official_full12_evaluation') is True and right.get('revision_scope')=='task_first_recovery_v1','Not official recovery C')
    audit,projection=same_task_invariants(left,right)
    initial=prior.initial_pair_audit(left,right)
    original_read,original_run=base.read_json,base.run
    def compared_read(path):
        item=original_read(path)
        if Path(path).resolve() in (left_path.resolve(),right_path.resolve()):
            # Transparent in-memory encoder-only invariant projection; source and
            # individual receipts retain their complete different true runtimes.
            item={**item,'label':'B0' if Path(path).resolve()==left_path.resolve() else 'C0',
                'runtime_contract':projection,'evaluation_configuration':{'same_N_physics_task':True,'same_reward':False}}
        return item
    def labelled_run(command):
        if '-filter_complex' in command:
            i=command.index('-filter_complex')+1
            handoff='final_stop_handoff_comparison' in audit
            quarter='quarter_sampling_comparison' in audit
            fixed='execution_composition_comparison' in audit or handoff
            right_label='PPO quarter-trained mean' if quarter else 'PPO execution + P13 handoff' if handoff else 'PPO residual-history fix' if fixed else 'PPO task recovery'
            command[i]=command[i].replace("text='B0 |","text='ZERO old build |" if fixed else "text='ZERO retained |").replace("text='C0 |",f"text='{right_label} |")
            caption=('1x - old ZERO vs new build - saved mean, no eval scaling' if quarter else
                '1x - same N/physics - old ZERO vs exec + P13 fix' if handoff else
                '1x - same N/physics - old ZERO vs execution fix' if fixed else '1x - same N and physics - different reward/build')
            command[i]=command[i].replace('1x real elapsed time - 15 Hz interval endpoints',caption)
        return original_run(command)
    try:
        base.read_json,base.run=compared_read,labelled_run
        base.compare(left_path,right_path,output)
    finally:base.read_json,base.run=original_read,original_run
    prior.annotate_new_receipt(output.with_suffix('.media.json'),{'same_N_physics_runtime_delta_audit':audit,
        'initial_physical_state_equality':initial,'same_build_claim':False,
        'runtime_match_policy':('retained old ZERO vs execution/P13 fixes plus independently migrated stochastic quarter kernel; saved-mean deterministic evaluation without temporary scaling; N source/physics/evaluator unchanged, not same build; no new physical B'
            if 'quarter_sampling_comparison' in audit else 'retained old ZERO vs independently migrated execution and P13 handoff fixes; same N source/physics/evaluator, not same build; no new paired physical B'
            if 'final_stop_handoff_comparison' in audit else 'retained old ZERO vs explicitly migrated residual-history fix; same N source/physics/task, not same execution; no new paired physical B'
            if 'execution_composition_comparison' in audit else 'reviewed same N/mapper/physics/task; different reward/routing/training audit build'),
        'preserved_zero_final_camera_limitation':'far-side final stop partly outside fixed frame; original frame and physics evidence preserved'})

def m1_excerpt(full_receipt,output):
    receipt=base.read_json(full_receipt); source=Path(receipt['source_manifest']).parent
    base.require(receipt.get('official_full12_evaluation') is True,'Not official C')
    manifest=base.read_json(source/'semantic_video_source_manifest.json')
    base.require(sha(source/'semantic_video_source_manifest.json')==receipt['source_manifest_sha256'],'Source changed')
    observed=None
    for line in (source/'video_policy_decisions.jsonl').open(encoding='utf8'):
        row=json.loads(line); task=(row.get('step_info') or {}).get('semantic_task',{})
        evaluator=task.get('physical_evaluator',{}); hist=evaluator.get('history',{})
        if all(hist.get(k,{}).get('FR') is True for k in ('active_lift','front_edge_crossed','placed')):
            current=evaluator.get('current_legs',{}).get('FR',{})
            if current.get('top_contact') and current.get('top_surface_contact') and current.get('bearing_verified') and current.get('bearing_force_n',0)>0:
                observed={'decision':row.get('decision'),'tick':evaluator.get('physics_tick'),
                    'history':hist,'current_FR':current}; break
    base.require(observed is not None,'No actual qualified FR crossing and captured bearing contact; no M1 success filename')
    ledger=base.load_viewport_frame_ledger(source/'viewport_frame_ledger.jsonl')
    # Preserve one extra real second if available, never simulate/freeze it.
    target_tick=min(receipt['physical_ticks'],observed['tick']+120)
    ending=next((i for i,f in enumerate(ledger) if f.sim_step>=target_tick),len(ledger)-1)
    count=ending+1
    output=output.resolve(); base.require(not output.exists() and not output.with_suffix('.media.json').exists(),'Never overwrite')
    ffmpeg=base.find_ffmpeg()
    command=[str(ffmpeg),'-hide_banner','-nostdin','-v','error','-n','-i',receipt['output'],
        '-vf',f'trim=start_frame=0:end_frame={count},setpts=PTS-STARTPTS','-frames:v',str(count),
        '-an','-c:v','libx264','-preset','veryfast','-crf','18','-pix_fmt','yuv420p','-threads','2','-r','15','-movflags','+faststart',str(output)]
    base.run(command)
    validation=base.validate_mp4(output,ffmpeg=ffmpeg,expected_frame_count=count,require_sane_container_duration=True)
    base.require(validation['valid'] and count/15<=200,'M1 excerpt decode failed')
    base.write_new_json(output.with_suffix('.media.json'),{'schema':'task_recovery_M1_contiguous_excerpt.v1',
        'full_attempt_receipt':str(full_receipt.resolve()),'source_manifest':receipt['source_manifest'],
        'source_manifest_sha256':receipt['source_manifest_sha256'],'output':str(output),'M1':True,
        'full_task_success':receipt['physical_task_success'],'later_failure_not_removed_from_separate_full_attempt':True,
        'physical_evidence':observed,'last_actual_frame_tick':ledger[ending].sim_step,'frame_count':count,
        'media_duration_s':count/15,'starts_at_natural_P01':True,'single_contiguous_source':True,
        'speed_modified':False,'teacher':False,'diagnostic_mask':None,'validation':base.compact_validation(validation),
        'previews':base.previews(output,count,ffmpeg),'command':command})

def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    e=sub.add_parser('export');e.add_argument('--source',type=Path,required=True);e.add_argument('--output',type=Path,required=True)
    c=sub.add_parser('compare');c.add_argument('--left',type=Path,default=OLD_ZERO);c.add_argument('--right',type=Path,required=True);c.add_argument('--output',type=Path,required=True)
    m=sub.add_parser('m1');m.add_argument('--full-receipt',type=Path,required=True);m.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    base.require(args.output.resolve().is_relative_to(OUT),'Only task recovery output directory allowed')
    if args.command=='export':export(args.source,args.output)
    elif args.command=='compare':compare(args.left,args.right,args.output)
    else:m1_excerpt(args.full_receipt,args.output)

if __name__=='__main__':main()
