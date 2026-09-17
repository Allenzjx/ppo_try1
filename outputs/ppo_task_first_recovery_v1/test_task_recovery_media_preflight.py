"""Metadata-only preflight. Synthetic right media metadata is never published."""
import copy
import importlib.util
import json
from pathlib import Path
import pytest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('task_media_test_subject',HERE/'task_recovery_media.py')
media=importlib.util.module_from_spec(spec);spec.loader.exec_module(media)
TRAIN=media.ROOT/'runs/ppo_task_first_recovery_v1/train/20260916T0342024652780Z_gb0438f66ec63_a11da4badf6f4422b0af0decfc19053c/run_manifest.started.json'

def fixture():
    left=media.base.read_json(media.OLD_ZERO)
    contract=media.base.read_json(TRAIN)['runtime_contract']
    right=copy.deepcopy(left);right['runtime_contract']=contract
    right['evaluation_configuration']={}
    for field,name in media.CONFIG_FIELDS.items():
        binding=contract['selected_configuration'][name]
        path=media.ROOT/binding['path']
        right['evaluation_configuration'][field]={'path':str(path),'sha256':binding['sha256'],'bytes':path.stat().st_size}
    return left,right

def test_reviewed_runtime_changes_pass_without_claiming_same_build():
    a,b=fixture();audit,projection=media.same_task_invariants(a,b)
    assert not audit['same_build_claim'] and not audit['same_reward_claim']
    assert not projection['same_build']
    assert sum(row['parsed_content_equal'] for row in audit['configuration'])==5
    assert audit['same_N_mapper_physical_evaluator_config']
    assert 'runtime_content_sha256' not in projection

@pytest.mark.parametrize('path',['src/wlr50_clean/infrastructure/servo_target_mapper.py',
    'src/wlr50_clean/ppo/semantic_supervisor.py','src/wlr50_clean/ppo/semantic_nominal_geometry.py',
    'configs/environment_lock.json'])
def test_unreviewed_control_physics_and_task_changes_rejected(path):
    a,b=fixture();b['runtime_contract']['files'][path]='0'*64
    with pytest.raises(RuntimeError,match='Unreviewed'):media.same_task_invariants(a,b)

def test_actual_config_binding_change_rejected_without_silent_semantic_waiver():
    a,b=fixture();b['evaluation_configuration']['execution_profile']['sha256']='0'*64
    with pytest.raises(RuntimeError,match='Selected config'):media.same_task_invariants(a,b)

def test_physics_frequency_change_rejected():
    a,b=fixture();b['runtime_contract']['physics_hz']=240
    with pytest.raises(RuntimeError,match='invariants differ'):media.same_task_invariants(a,b)


def branch_fixture():
    origin={'global_policy_decisions':176640,'ppo_updates':1345,'optimizer_steps':26900}
    counts={'global_policy_decisions':128,'ppo_updates':1,'optimizer_steps':20}
    metadata={'checkpoint_path':'checkpoint_step_000176768.pt','stage':'full_episode',
        'actor_parameter_sha256':'actor','task_recovery_branch':{'branch_id':'recovery_fixture','counter_origin':origin},
        'task_recovery_branch_counts':counts,**{k:origin[k]+counts[k] for k in media.COUNTERS}}
    proof={'checkpoint_loaded_and_verified':True,'saved_global_policy_decisions':176768,
        'parameter_hashes':{'actor_parameter_sha256':'actor'},'source':{'manifest':'fixture_only'}}
    return metadata,proof


def test_trained_branch_counts_are_distinct_from_lifetime(monkeypatch):
    metadata,proof=branch_fixture()
    monkeypatch.setattr(media,'bound_checkpoint_metadata',lambda binding:metadata)
    evidence=media.learned_checkpoint_evidence(proof)
    assert evidence['new_learning_counts']=={'global_policy_decisions':128,'ppo_updates':1,'optimizer_steps':20}
    assert evidence['mode']=='learned_mean_head_recovery_branch'


@pytest.mark.parametrize('missing',media.COUNTERS)
def test_large_lifetime_counts_do_not_pass_when_any_branch_learning_count_zero(monkeypatch,missing):
    metadata,proof=branch_fixture();metadata['task_recovery_branch_counts'][missing]=0
    monkeypatch.setattr(media,'bound_checkpoint_metadata',lambda binding:metadata)
    with pytest.raises(RuntimeError,match='post-initialization'):media.learned_checkpoint_evidence(proof)


@pytest.mark.parametrize('field,value',[('stage','initial_task_recovery_mean_head'),
    ('checkpoint_path','checkpoint_initial_recovery_fixture_from_000176640.pt')])
def test_initialization_stage_or_filename_rejected_even_with_positive_fake_counts(monkeypatch,field,value):
    metadata,proof=branch_fixture();metadata[field]=value
    monkeypatch.setattr(media,'bound_checkpoint_metadata',lambda binding:metadata)
    with pytest.raises(RuntimeError,match='initialization'):media.learned_checkpoint_evidence(proof)


def test_real_immutable_reward_only_manifest_has_actual_new_learning():
    path=media.ROOT/'outputs/ppo_task_first_recovery_v1/checkpoints/history/checkpoint_step_000174848_manifest.json'
    metadata=media.base.read_json(path)
    binding={'manifest':str(path),'manifest_sha256':media.sha(path),
        'checkpoint':metadata['checkpoint_path'],'checkpoint_sha256':metadata['checkpoint_sha256']}
    proof={'checkpoint_loaded_and_verified':True,'source':binding,'saved_global_policy_decisions':174848,
        'parameter_hashes':{'actor_parameter_sha256':metadata['actor_parameter_sha256']}}
    result=media.learned_checkpoint_evidence(proof)
    assert result['new_learning_counts']=={'global_policy_decisions':256,'ppo_updates':2,'optimizer_steps':40}
    assert len(result['verified_ancestry_manifests'])==1


def test_learned_branch_counts_are_independent_of_execution_migration_factor(monkeypatch):
    metadata,proof=branch_fixture()
    metadata['resume_ancestry']={'resume_migration':{'execution_composition_factor':{'schema':media.EXECUTION_FIX_SCHEMA}}}
    monkeypatch.setattr(media,'bound_checkpoint_metadata',lambda binding:metadata)
    assert media.learned_checkpoint_evidence(proof)['new_learning_counts']['ppo_updates']==1


def execution_fixture(tmp_path,monkeypatch):
    before={'files':{k:'old'+k for k in media.EXECUTION_FIX_CODE},'selected_configuration':{'fixture':'unchanged'}}
    after={'files':{k:'new'+k for k in media.EXECUTION_FIX_CODE},'selected_configuration':{'fixture':'unchanged'}}
    binding={'checkpoint':str(tmp_path/'history/source.pt'),'checkpoint_sha256':'cp',
        'manifest':str(tmp_path/'history/source_manifest.json'),'manifest_sha256':'manifest'}
    factor={'schema':media.EXECUTION_FIX_SCHEMA,'nominal_history_semantics':'same_actual_state_nominal_command_history_excludes_policy_v1',
        'reviewed_code_sha256':after['files'],**{k:False for k in ('physical_scene_changed','actuator_capability_changed',
        'nominal_source_schedule_changed','task_acceptance_changed','reward_changed','action_ranges_changed',
        'old_rollout_inherited','trajectory_equivalence_claimed')}}
    plan={'source_checkpoint':binding['checkpoint'],'source_checkpoint_sha256':'cp','source_manifest_sha256':'manifest',
        'source_contract_sha256':media.contract_digest(before),'target_contract_sha256':media.contract_digest(after),
        'changed_file_hashes':{k:{'before':before['files'][k],'after':after['files'][k]} for k in media.EXECUTION_FIX_CODE},
        'allowed_changed_files':sorted(media.EXECUTION_FIX_CODE),'execution_composition_factor':factor}
    path=tmp_path/'synthetic_plan_not_published.json';path.write_text(json.dumps(plan),encoding='utf8')
    migration={**plan,'plan_path':str(path),'plan_sha256':media.sha(path)}
    metadata={'runtime_contract':after,'resume_ancestry':{'resume_migration':migration,
        'source_checkpoint':binding,'source_runtime_contract':before}}
    monkeypatch.setattr(media,'bound_checkpoint_metadata',lambda binding:{'runtime_contract':before})
    return metadata,after


def test_explicit_execution_plan_hash_and_contract_ancestry_pass(tmp_path,monkeypatch):
    metadata,after=execution_fixture(tmp_path,monkeypatch)
    result=media.execution_factor_ancestry(metadata,after)
    assert result['execution_composition_factor']['schema']==media.EXECUTION_FIX_SCHEMA
    assert result['verified_ancestry_manifests']


def test_execution_migration_plan_file_change_rejected(tmp_path,monkeypatch):
    metadata,after=execution_fixture(tmp_path,monkeypatch)
    Path(metadata['resume_ancestry']['resume_migration']['plan_path']).write_text('{}',encoding='utf8')
    with pytest.raises(RuntimeError,match='plan changed'):media.execution_factor_ancestry(metadata,after)


def test_execution_factor_cannot_authorize_physics_change(tmp_path,monkeypatch):
    metadata,after=execution_fixture(tmp_path,monkeypatch)
    metadata['resume_ancestry']['resume_migration']['execution_composition_factor']['physical_scene_changed']=True
    with pytest.raises(RuntimeError,match='protected'):media.execution_factor_ancestry(metadata,after)


def test_verified_migration_cannot_authorize_later_core_hash_change(tmp_path,monkeypatch):
    metadata,after=execution_fixture(tmp_path,monkeypatch)
    target=copy.deepcopy(after);target['files']['src/wlr50_clean/ppo/semantic_nominal_geometry.py']='unreviewed'
    with pytest.raises(RuntimeError,match='Formal runtime core'):media.execution_factor_ancestry(metadata,target)


@pytest.mark.parametrize('wrong_geometry',[False,True])
def test_zero_replay_bound_to_formal_geometry_hash(tmp_path,monkeypatch,wrong_geometry):
    left=media.base.read_json(media.OLD_ZERO)
    replay=media.base.read_json(media.ZERO_REPLAY)
    right=copy.deepcopy(left)
    right['runtime_contract']['files']['src/wlr50_clean/ppo/semantic_nominal_geometry.py']=(
        'wrong' if wrong_geometry else replay['new_geometry_source_sha256'])
    source=tmp_path/'synthetic_source_not_published.json'
    source.write_text(json.dumps({'runtime_contract':right['runtime_contract'],'checkpoint_load_provenance':{'source':{}}}),encoding='utf8')
    right.update(source_manifest=str(source),source_manifest_sha256=media.sha(source))
    monkeypatch.setattr(media,'bound_checkpoint_metadata',lambda binding:{'runtime_contract':right['runtime_contract']})
    monkeypatch.setattr(media,'execution_factor_ancestry',lambda metadata,target:{'unit_test_only':True})
    if wrong_geometry:
        with pytest.raises(RuntimeError,match='geometry hash'):media.execution_fix_comparison_evidence(left,right)
    else:
        result=media.execution_fix_comparison_evidence(left,right)
        assert result['new_paired_physical_B_rerun'] is False
        assert result['zero_trajectory_equivalence_not_proven_by_offline_replay'] is True


def final_stop_fixture(tmp_path,monkeypatch):
    before={'files':{k:'old'+k for k in media.FINAL_STOP_MIGRATION_CODE},
        'selected_configuration':{'fixture':'unchanged'},'runtime_content_sha256':'old_runtime'}
    after={'files':{k:'new'+k for k in media.FINAL_STOP_MIGRATION_CODE},
        'selected_configuration':{'fixture':'unchanged'},'runtime_content_sha256':'new_runtime'}
    binding={'checkpoint':str(tmp_path/'history/source.pt'),'checkpoint_sha256':'cp',
        'manifest':str(tmp_path/'history/source_manifest.json'),'manifest_sha256':'manifest'}
    factor={'schema':media.FINAL_STOP_SCHEMA,'stop_owner_acquisition_semantics':'post_window_triggered_nominal_stop_takeover_v2',
        'reviewed_code_sha256':after['files'],**{k:False for k in ('task_evaluator_changed','reward_changed',
        'fixed_post_completion_window_changed','physical_scene_changed','actuator_capability_changed',
        'residual_composition_changed','controller_history_reset_on_phase_transition',
        'recorded_FSM_source_changed','task_acceptance_changed','action_ranges_changed','kernel_changed',
        'old_rollout_inherited','trajectory_equivalence_claimed')},
        'supervisor_scope':{'method':'NominalMotionProvider._observe_final_stop_owner','method_signature_unchanged':True,
            'evaluator_and_other_methods_unchanged':True,'unchanged_outside_method_sha256':'a'*64,
            'source_method_sha256':'b'*64,'target_method_sha256':'c'*64}}
    plan={'source_checkpoint':binding['checkpoint'],'source_checkpoint_sha256':'cp','source_manifest_sha256':'manifest',
        'source_contract_sha256':media.contract_digest(before),'target_contract_sha256':media.contract_digest(after),
        'source_runtime_content_sha256':'old_runtime','target_runtime_content_sha256':'new_runtime',
        'changed_file_hashes':{k:{'before':before['files'][k],'after':after['files'][k]} for k in media.FINAL_STOP_MIGRATION_CODE},
        'allowed_changed_files':sorted(media.FINAL_STOP_MIGRATION_CODE),'final_stop_handoff_factor':factor}
    path=tmp_path/'synthetic_P13_plan_not_published.json';path.write_text(json.dumps(plan),encoding='utf8')
    migration={**plan,'plan_path':str(path),'plan_sha256':media.sha(path)}
    metadata={'runtime_contract':after,'resume_ancestry':{'resume_migration':migration,
        'source_checkpoint':binding,'source_runtime_contract':before}}
    monkeypatch.setattr(media,'bound_checkpoint_metadata',lambda binding:{'runtime_contract':before})
    return metadata,after


def test_independent_p13_handoff_bound_plan_passes(tmp_path,monkeypatch):
    metadata,target=final_stop_fixture(tmp_path,monkeypatch)
    result=media.final_stop_factor_ancestry(metadata,target)
    assert result['final_stop_handoff_factor']['schema']==media.FINAL_STOP_SCHEMA
    assert result['target_runtime_content_sha256']=='new_runtime'


def test_final_stop_factor_does_not_pass_as_execution_factor(tmp_path,monkeypatch):
    metadata,target=final_stop_fixture(tmp_path,monkeypatch)
    with pytest.raises(RuntimeError,match='explicit migration ancestry'):
        media.execution_factor_ancestry(metadata,target)


@pytest.mark.parametrize('flag',['task_evaluator_changed','fixed_post_completion_window_changed'])
def test_p13_handoff_cannot_waive_evaluator_or_window_change(tmp_path,monkeypatch,flag):
    metadata,target=final_stop_fixture(tmp_path,monkeypatch)
    metadata['resume_ancestry']['resume_migration']['final_stop_handoff_factor'][flag]=True
    with pytest.raises(RuntimeError,match='protected'):media.final_stop_factor_ancestry(metadata,target)


def test_p13_handoff_requires_method_scope_proof(tmp_path,monkeypatch):
    metadata,target=final_stop_fixture(tmp_path,monkeypatch)
    metadata['resume_ancestry']['resume_migration']['final_stop_handoff_factor']['supervisor_scope']={}
    with pytest.raises(RuntimeError,match='method-only'):media.final_stop_factor_ancestry(metadata,target)


@pytest.mark.parametrize('wrong_supervisor',[False,True])
def test_p13_real_recorded_zero_receipt_bound_to_new_supervisor(tmp_path,monkeypatch,wrong_supervisor):
    left=media.base.read_json(media.OLD_ZERO)
    replay=media.base.read_json(media.FINAL_STOP_ZERO_REPLAY)
    right=copy.deepcopy(left)
    right['runtime_contract']['files']['src/wlr50_clean/ppo/semantic_supervisor.py']=(
        'wrong' if wrong_supervisor else replay['new_supervisor_sha256'])
    source=tmp_path/'synthetic_P13_source_not_published.json'
    source.write_text(json.dumps({'runtime_contract':right['runtime_contract'],'checkpoint_load_provenance':{'source':{}}}),encoding='utf8')
    right.update(source_manifest=str(source),source_manifest_sha256=media.sha(source))
    monkeypatch.setattr(media,'bound_checkpoint_metadata',lambda binding:{'runtime_contract':right['runtime_contract']})
    monkeypatch.setattr(media,'final_stop_factor_ancestry',lambda metadata,target:{'unit_test_only':True})
    if wrong_supervisor:
        with pytest.raises(RuntimeError,match='supervisor hashes'):media.final_stop_comparison_evidence(left,right)
    else:
        result=media.final_stop_comparison_evidence(left,right)
        assert result['new_paired_physical_B_rerun'] is False
        assert result['same_execution_build_claim'] is False


def test_execution_and_handoff_factors_remain_separate_allowlists():
    assert not media.FINAL_STOP_CODE & media.EXECUTION_FIX_CODE
    assert not media.FINAL_STOP_CODE & media.ALLOWED_CHANGED_CODE
    assert not media.EXECUTION_FIX_CODE & media.FINAL_STOP_MIGRATION_CODE


def quarter_checkpoint_metadata():
    return media.base.read_json(media.ROOT/'outputs/ppo_task_first_recovery_v1/checkpoints/history/checkpoint_step_000177280_manifest.json')


def test_real_quarter_checkpoint_keeps_three_independent_migration_ancestors():
    metadata=quarter_checkpoint_metadata();target=metadata['runtime_contract']
    quarter=media.quarter_factor_ancestry(metadata,target)
    execution=media.execution_factor_ancestry(metadata,target)
    stop=media.final_stop_factor_ancestry(metadata,target)
    assert quarter['exploration_temperature_factor']['target_exploration_std_temperature']==.25
    assert quarter['deterministic_evaluation_uses_saved_conditional_mean'] is True
    assert quarter['evaluation_time_action_scaling'] is False
    assert quarter['new_paired_physical_B_rerun'] is False
    assert execution['execution_composition_factor']['schema']==media.EXECUTION_FIX_SCHEMA
    assert stop['final_stop_handoff_factor']['schema']==media.FINAL_STOP_SCHEMA


@pytest.mark.parametrize('fault',['tau','physics','old_half_actor'])
def test_quarter_media_rejects_unreviewed_kernel_or_physics_claim(fault):
    metadata=quarter_checkpoint_metadata()
    factor=metadata['resume_ancestry']['resume_migration']['exploration_temperature_factor']
    if fault=='tau':factor['target_exploration_std_temperature']=.2
    elif fault=='physics':factor['physical_mdp_changed']=True
    else:factor['code_scope']['src/wlr50_clean/ppo/semantic_history_actor.py']['reviewed_regions']=['SemanticTemperedHistoryMLPModel']
    with pytest.raises(RuntimeError):media.quarter_factor_ancestry(metadata,metadata['runtime_contract'])


def test_quarter_media_rejects_later_kernel_change_without_migration():
    metadata=quarter_checkpoint_metadata();target=copy.deepcopy(metadata['runtime_contract'])
    target['files']['src/wlr50_clean/ppo/semantic_history_actor.py']='unreviewed'
    with pytest.raises(RuntimeError,match='Formal quarter kernel'):
        media.quarter_factor_ancestry(metadata,target)
