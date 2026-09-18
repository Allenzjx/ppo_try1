"""Output-only wiring fixtures, not physical success or media evidence."""
import copy
import importlib.util
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('sealed_rr_review_test',Path(__file__).with_name('sealed_rr_review.py'))
review=importlib.util.module_from_spec(spec); spec.loader.exec_module(review)


def test_active_refused_before_any_source_or_large_log_read(tmp_path):
    source=tmp_path/'source'; source.mkdir()
    (tmp_path/'run_manifest.json').write_text(json.dumps({'lifecycle':'RUNNING'}))
    with pytest.raises(RuntimeError,match='not finalized'):review.sealed_source(source,'C')


def test_sealed_incomplete_zero_is_accepted_not_a_success_gate(tmp_path):
    source=tmp_path/'source'; source.mkdir()
    runtime={'source_git_commit':'fixture'}
    (tmp_path/'run_manifest.json').write_text(json.dumps({'lifecycle':'DIAGNOSTIC_FAILURE',
        'completed_at_utc':'fixture sealed timestamp','runtime_contract':runtime}))
    manifest={'role':'B','experiment_id':'non_residual_refine_v1','diagnostic_intervention':None,
        'from_phase':'P01','optimizer_updates':0,'fresh_process_single_episode':True,'episode_count':1,
        'runtime_contract':runtime,'physical_task_success':False,
        'camera':{**review.camera_for_experiment('non_residual_refine_v1'),'resolution':[1280,720],'fps':15}}
    (source/'semantic_video_source_manifest.json').write_text(json.dumps(manifest))
    assert review.sealed_source(source,'B')['physical_task_success'] is False


def learned_fixture():
    origin=dict(zip(review.COUNTERS,(177792,1354,27080)))
    added=dict(zip(review.COUNTERS,(128,1,20)))
    metadata={k:origin[k]+added[k] for k in origin}
    metadata.update(checkpoint_path='checkpoint_177920.pt',stage='full',actor_parameter_sha256='actor',
        rr_task_branch={'branch_id':'residual_rr_fix_v1','counter_origin':origin},rr_task_branch_counts=added)
    proof={'checkpoint_loaded_and_verified':True,'saved_global_policy_decisions':177920,
        'parameter_hashes':{'actor_parameter_sha256':'actor'}}
    return metadata,proof


def test_real_positive_branch_learning_required_not_lifetime_alone():
    metadata,proof=learned_fixture()
    assert review.learning_counts(metadata,proof)['actual_added_counts']['ppo_updates']==1
    metadata['rr_task_branch_counts']['ppo_updates']=0
    with pytest.raises(RuntimeError,match='positive verified'):review.learning_counts(metadata,proof)


def test_initializer_cannot_masquerade_as_learned():
    metadata,proof=learned_fixture(); metadata['stage']='initial_task_recovery_mean_head'
    with pytest.raises(RuntimeError,match='Initialization'):review.learning_counts(metadata,proof)


def test_wrong_branch_cannot_be_formal_rr():
    metadata,proof=learned_fixture(); metadata['rr_task_branch']['branch_id']='task_first_recovery_v1'
    with pytest.raises(RuntimeError,match='Not RR'):review.learning_counts(metadata,proof)


def task(tick=16,source_tick=15):
    return {'physical_evaluator':{'physics_tick':tick,'current_legs':{'RR':{
        'clearance_m':-.012,'unsupported_free_lift_m':.006,'current_lift_valid':False,'air':True,'contact_mode':'AIR'}}},
        'nominal_provider_diagnostics':{'source_partial_order':{'layers':[{
            'stage':'P09','observation_tick':source_tick,'status':'holding','wait_reason':'current_free_lift_before_pending_knee'}]}}}


def test_exact_physics_and_previous_source_check_are_separately_bound():
    row=review.semantic_sample(task(),16)
    assert row['RR_gap_m']==-.012 and row['RR_current_lift'] is False
    assert row['source_readiness']['observation_tick']==15


def test_no_forward_fill_or_future_check():
    row=review.semantic_sample(task(),24)
    assert row['RR_gap_m'] is None and row['RR_current_lift'] is None
    assert review.value(row['RR_free_gain_m'])=='N/A'
    with pytest.raises(RuntimeError,match='future'):review.semantic_sample(task(source_tick=17),16)


def test_missing_physical_sample_refused(tmp_path):
    path=tmp_path/'physical.jsonl'; path.write_text(json.dumps({'physics_tick':8})+'\n')
    manifest={'artifacts':{'physical.jsonl':{'sha256':review.sha(path)}}}
    with pytest.raises(RuntimeError,match='no forward fill'):
        review.selected_rows(tmp_path,manifest,'physical.jsonl','physics_tick',{8,16})


def stage_pair():
    zero={'revision':'fsm_reference_p09_stable_v2','p09_lift_semantics':'functional_lift_edge_v2',
        'nominal':{'final_stop_owner':'source_home_after_physical_stop_v2'},'safety':{'wheel_only_climb':True}}
    ppo=copy.deepcopy(zero); ppo.update(revision='residual_rr_fix_v1',p09_lift_semantics='functional_free_air_lift_v3')
    ppo['nominal']['rr_carry_source_semantics']='current_free_lift_before_pending_knee_and_roll_v1'
    return zero,ppo


def test_only_declared_stage_differences_and_disclose_not_same_controller():
    zero,ppo=stage_pair(); disclosure=review.stage_difference(zero,ppo)
    assert disclosure['same_task_acceptance_claimed'] is False
    assert disclosure['same_nominal_source_schedule_claimed'] is False
    ppo['safety']['wheel_only_climb']=False
    with pytest.raises(RuntimeError,match='Unreviewed'):review.stage_difference(zero,ppo)


def test_failure_clip_keeps_final_endpoint_and_context():
    rows=[{'actual_physics_tick':8*i,'phase':'P08' if i<600 else 'P09'} for i in range(1,901)]
    start,end=review.failure_window(rows)
    assert end==900 and rows[end-1]['actual_physics_tick']==7200
    assert rows[start]['actual_physics_tick']==4800
    assert end-start+1<=451


def test_ready_panel_fits_external_margin():
    row={'actual_sim_time_s':123.456,'actual_physics_tick':14816,'phase':'P09',
        'measured_native_qd_rad_s':[-10.,10.,-10.,10.],**review.semantic_sample(task(),16)}
    lines=review.panel_lines(row,'PPO FULL12 CP177920 | recorded TASK_INCOMPLETE',True)
    assert 'N check@15' in lines[4]
    assert 'FL -10.000' in lines[2] and 'RR +10.000' in lines[2]
    font=review.ImageFont.truetype('C:/Windows/Fonts/consola.ttf',20)
    draw=review.ImageDraw.Draw(review.Image.new('RGB',(1280,160)))
    assert max(draw.textlength(line,font=font) for line in lines[:4])<=1248


def test_qd_panel_uses_native_not_canonical_values():
    row={'actual_sim_time_s':1.,'actual_physics_tick':120,'phase':'P02',
        'measured_native_qd_rad_s':[-.3,.4,-.5,.6],
        'measured_canonical_qd_rad_s':[.3,.4,.5,.6],**review.semantic_sample({},120)}
    text=review.panel_lines(row,'fixture')[2]
    assert 'native rad/s' in text and 'FL -0.300' in text and 'RL -0.500' in text
    assert 'FL +0.300' not in text
