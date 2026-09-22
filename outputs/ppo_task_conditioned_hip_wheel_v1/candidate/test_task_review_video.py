"""Synthetic metadata only, no video generation, Isaac or model evaluation."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

path=Path(__file__).resolve().parents[1]/'review_video.py'
spec=importlib.util.spec_from_file_location('task_review',path)
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)


@pytest.fixture
def capture(tmp_path,monkeypatch):
    out=tmp_path/'SYNTHETIC_ONLY_outputs';monkeypatch.setattr(r,'OUT',out)
    history=out/'checkpoints/history';history.mkdir(parents=True)
    cp=history/'checkpoint_step_000185984.pt';cp.write_bytes(b'SYNTHETIC_METADATA_TEST_NOT_A_MODEL')
    metadata_path=cp.with_name(cp.stem+'_manifest.json')
    contract={'experiment_id':r.EXPERIMENT,'source_git_commit':'c'*40,'runtime_content_sha256':'d'*64}
    policy={'version':r.POLICY_VERSION,'rho':.9}
    counts={'global_policy_decisions':128,'ppo_updates':1,'optimizer_steps':20}
    origin={'global_policy_decisions':185856,'ppo_updates':1417,'optimizer_steps':28340}
    metadata={'checkpoint_path':str(cp),'checkpoint_sha256':r.sha256(cp),'save_load_round_trip':True,
        'actor_parameter_sha256':'a'*64,'runtime_contract':contract,'policy_contract':policy,
        **{k:origin[k]+counts[k] for k in counts},
        'task_conditioned_hip_wheel_branch':{'branch_id':r.EXPERIMENT,'counter_origin':origin},
        'task_conditioned_hip_wheel_branch_counts':counts}
    proof={'checkpoint_loaded_and_verified':True,'official_load_semantic_checkpoint':True,
        'policy_version':r.POLICY_VERSION,'policy_contract':policy,'optimizer_updates':0,
        'saved_global_policy_decisions':185984,'parameter_hashes':{'actor_parameter_sha256':'a'*64},
        'policy_sampling_mode':'deterministic_conditional_mean','stochastic_policy':False,'policy_seed':None,
        'source':{'checkpoint':str(cp),'checkpoint_sha256':r.sha256(cp),'manifest':str(metadata_path)}}
    source=tmp_path/'SYNTHETIC_ONLY_run/source';source.mkdir(parents=True)
    manifest={'experiment_id':r.EXPERIMENT,'runtime_contract':contract,'role':'C','from_phase':'P01',
        'episode_count':1,'fresh_process_single_episode':True,'optimizer_updates':0,'diagnostic_intervention':None,
        'seed':4001,'checkpoint_load_provenance':proof,'policy_sampling_mode':'deterministic_conditional_mean',
        'physical_task_success':False,'camera':{'name':'same_test_camera'},
        'evaluation_configuration':{'test':'same'},'natural_reset_proof':{'entry':'natural_test'}}
    run={'completed_at_utc':'SYNTHETIC_ONLY','lifecycle':'DIAGNOSTIC_FAILURE','runtime_contract':contract}
    def save():
        metadata_path.write_text(json.dumps(metadata),encoding='utf-8')
        proof['source']['manifest_sha256']=r.sha256(metadata_path)
        (source/'semantic_video_source_manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
        (source.parent/'run_manifest.json').write_text(json.dumps(run),encoding='utf-8')
    save()
    return {'source':source,'metadata':metadata,'proof':proof,'manifest':manifest,'run':run,'save':save}


def test_new_branch_positive_learning_and_neutral_complete_attempt_names(capture):
    context=r.sealed_source(capture['source'])
    assert context['role']=='C' and context['mode']=='deterministic_conditional_mean'
    assert context['checkpoint']['task_conditioned_hip_wheel_branch_counts']['ppo_updates']==1
    assert r.output_name('C',context['mode'],capture['proof'])=='CP185984_deterministic_P01_full.mp4'
    proof=copy.deepcopy(capture['proof']);proof.update(stochastic_policy=True,policy_seed=4101,
        policy_sampling_mode='training_style_conditional_gaussian')
    assert r.output_name('C','training_style_conditional_gaussian',proof)=='CP185984_stochastic_P01_full_seed4101.mp4'


@pytest.mark.parametrize('fault',['old_experiment','old_policy','old_branch','zero_updates','counter_mismatch',
    'unsealed','intervention','wrong_mode','det_seed','runtime_difference','unverified_load'])
def test_rejects_old_or_mislabelled_formal_sources(capture,fault):
    m,p,c,run=capture['metadata'],capture['proof'],capture['manifest'],capture['run']
    if fault=='old_experiment':c['experiment_id']='fl_capture_quality_v1'
    elif fault=='old_policy':p['policy_version']='request_history_FR_knee_P06plus_physical_innovation_sigma_v1'
    elif fault=='old_branch':m['task_conditioned_hip_wheel_branch']['branch_id']='fl_capture_quality_v1'
    elif fault=='zero_updates':m['task_conditioned_hip_wheel_branch_counts']['ppo_updates']=0
    elif fault=='counter_mismatch':m['task_conditioned_hip_wheel_branch_counts']['optimizer_steps']=19
    elif fault=='unsealed':run['lifecycle']='RUNNING'
    elif fault=='intervention':c['diagnostic_intervention']={'mask':[0]}
    elif fault=='wrong_mode':c['policy_sampling_mode']='training_style_conditional_gaussian'
    elif fault=='det_seed':p['policy_seed']=4101
    elif fault=='runtime_difference':c['runtime_contract']={**c['runtime_contract'],'source_git_commit':'f'*40}
    elif fault=='unverified_load':p['official_load_semantic_checkpoint']=False
    capture['save']()
    with pytest.raises((RuntimeError,ValueError)):r.sealed_source(capture['source'])


def modes(capture):
    context=r.sealed_source(capture['source'])
    det={'manifest':copy.deepcopy(context['manifest']),'receipt':{'role':'C','mode':'deterministic_conditional_mean',
        'checkpoint_identity':context['checkpoint']}}
    stochastic=copy.deepcopy(det);stochastic['receipt']['mode']='training_style_conditional_gaussian'
    stochastic['manifest']['checkpoint_load_provenance']['policy_seed']=4101
    return det,stochastic


def test_same_checkpoint_modes_and_changed_checkpoint_rejected(capture):
    det,stoch=modes(capture)
    assert r.strict_policy_modes(det,stoch)['stochastic_policy_seed']==4101
    stoch['receipt']['checkpoint_identity']['checkpoint_sha256']='e'*64
    with pytest.raises((RuntimeError,ValueError)):r.strict_policy_modes(det,stoch)


@pytest.mark.parametrize('field',['camera','runtime_contract','evaluation_configuration','seed'])
def test_common_pair_does_not_silently_accept_cross_build_or_condition(capture,field):
    left,right=modes(capture)
    right['manifest'][field]='different'
    with pytest.raises((RuntimeError,ValueError)):r.strict_common_capture(left,right)


def test_old_review_file_not_modified():
    original=r.ROOT/'outputs/ppo_fl_capture_quality_v1/review_video.py'
    assert "decisions > 178432" in original.read_text(encoding='utf-8')
    assert "'fl_capture_quality_branch_counts'" in original.read_text(encoding='utf-8')
    assert 'task_conditioned_hip_wheel' not in original.read_text(encoding='utf-8')
