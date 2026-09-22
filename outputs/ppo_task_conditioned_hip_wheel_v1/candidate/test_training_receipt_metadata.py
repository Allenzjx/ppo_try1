"""Synthetic metadata only; no checkpoint model or real training credit."""
from pathlib import Path
import hashlib
import importlib.util
import json

import pytest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('new_training_receipt',HERE.parent/'training_receipt.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


def jsonl(path,values):
    path.write_text(''.join(json.dumps(v)+'\n' for v in values),encoding='utf-8')


def fixture(root,*,suffix=False):
    run=root/'runs/ppo_task_conditioned_hip_wheel_v1/train/synthetic_not_real'
    run.mkdir(parents=True)
    cp=root/'outputs/ppo_task_conditioned_hip_wheel_v1/checkpoints/history/checkpoint_step_000185984.pt'
    cp.parent.mkdir(parents=True);cp.write_bytes(b'SYNTHETIC METADATA TEST; NOT A MODEL')
    side=cp.with_name(cp.stem+'_manifest.json')
    runtime={'experiment_id':r.EXPERIMENT};policy={'version':r.POLICY}
    counters={'global_policy_decisions':185984,'ppo_updates':1418,'optimizer_steps':28360}
    branch={'counter_origin':{'global_policy_decisions':185856,'ppo_updates':1417,'optimizer_steps':28340},'branch_id':'synthetic'}
    metadata=dict(**counters,runtime_contract=runtime,policy_contract=policy,save_load_round_trip=True,
        checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),source_run=str(run),
        resume_ancestry=dict(source_global_policy_decisions=185856,source_ppo_updates=1417,source_optimizer_steps=28340),
        task_conditioned_hip_wheel_branch=branch,
        task_conditioned_hip_wheel_branch_counts=dict(global_policy_decisions=128,ppo_updates=1,optimizer_steps=20))
    write(side,metadata)
    core=dict(decisions=128)
    if suffix:
        core.update(prefix_behavior_decisions=2,prefix_physics_ticks=16,prefix_attempts=[dict(prefix_decisions=2,prefix_physics_ticks=16,accepted=True)])
        jsonl(run/'prefix_evidence.jsonl',[dict(kind='checkpoint_prefix_decision',policy_credit=False) for _ in range(2)])
    training=dict(lifecycle='SUCCEEDED',runtime_contract=runtime,policy_contract=policy,num_envs=1,
        actual_policy_decisions=128,ppo_updates_this_run=1,optimizer_steps_this_run=20,global_policy_decisions=185984,
        runner_config={'num_steps_per_env':128},phase_suffix_curriculum_implemented=suffix,
        implemented_sampling='synthetic_nominal_prefix' if suffix else 'P01_only',
        checkpoints=[dict(checkpoint=str(cp),manifest=str(side),global_policy_decisions=185984)],
        telemetry=dict(core=core,policy_decisions=128,completed_episode_count=0,success_count=0))
    write(run/'training_manifest.json',training)
    write(run/'run_manifest.json',dict(lifecycle='SUCCEEDED',runtime_contract=runtime,result=training))
    records=[]
    for i in range(128):
        geometry=dict(phase='P06',eligible=True,valid=True,effective_beta_per_s=.03,
            weighted_geometry_cost=.0001,raw_geometry_cost=.05,dt_s=1/15,terminal_measurement_omitted=False)
        reward=dict(objective_profile=r.OBJECTIVE,quality_epsilon=.06,
            families={'task_progress':.1,'body_stability':-.0001},cost_components={},front_quality_sample_audit=[],
            task_space_quality_sample_audit=[geometry])
        info=dict(phase_id='P06',end_phase_id='P06',reward_breakdown=reward,termination_reason=None,
            task_success=False,full_task_success=False,
            semantic_task=dict(completed_stage_ids=list(r.PHASES[:5]),purpose='synthetic P06',completion_values={'rear_approach':.2}))
        if suffix:info['prefix_teacher_data_in_ppo_storage']=False
        records.append(dict(global_policy_decision=185857+i,terminal=False,policy_request={'policy_version':r.POLICY},applied_audit=info))
    jsonl(run/'residual_and_projection_audit.jsonl',records)
    jsonl(run/'optimizer_updates.jsonl',[dict(ppo_update=1418,global_policy_decisions=185984,optimizer_steps=20)])
    jsonl(run/'advantage_audit.jsonl',[dict(ppo_update_intended=1418)])
    jsonl(run/'completed_episodes.jsonl',[])
    (run/'rollouts').mkdir();(run/'rollouts/rollout_001418.pt').write_bytes(b'SYNTHETIC PLACEHOLDER NOT A ROLLOUT')
    write(run/'rollouts/update_001418_likelihood.json',{'synthetic_placeholder':True})
    return run,side


def edit(path,fn):
    data=r.read(path);fn(data);write(path,data)


def test_sealed_counts_actual_phase_and_geometry(tmp_path):
    run,_=fixture(tmp_path);out=r.summarize(run,project_root=tmp_path)
    assert (out['new_decisions'],out['new_ppo_updates'],out['new_optimizer_steps'])==(128,1,20)
    assert out['actual_request_phase_coverage']['P06']==128
    assert out['actual_request_phase_coverage']['P08']==0
    assert out['prefix']['behavior_decisions']==0
    assert out['actual_geometry_quality_by_phase']['P06']['beta_min_per_s']==.03
    assert out['actual_quality_contributions']['geometry_cost_known']==pytest.approx(.0128)
    assert out['last_sample_outcome']['first_unfinished_stage_by_logged_completion']=='P06'
    assert out['last_sample_outcome']['classification']=='budget_boundary_not_a_task_failure_or_success'
    assert out['natural_P01_full_task_success_count']==0
    assert out['advantage_by_update'][0]['ppo_update_intended']==1418


def test_prefix_separate_not_phase_credit(tmp_path):
    run,_=fixture(tmp_path,suffix=True);out=r.summarize(run,project_root=tmp_path)
    assert out['prefix']['behavior_decisions']==2 and out['prefix']['physics_ticks']==16
    assert out['prefix']['learner_credit']==0
    assert out['prefix']['persistent_decision_log_count']==2
    assert sum(out['actual_request_phase_coverage'].values())==128


@pytest.mark.parametrize('mutation,match',[
    (lambda d:d.update(lifecycle='RUNNING'),'not sealed'),
    (lambda d:d['runtime_contract'].update(experiment_id='old'),'wrong experiment'),
    (lambda d:d['result'].update(actual_policy_decisions=256),'manifests differ'),
])
def test_wrong_or_live_manifest_rejected(tmp_path,mutation,match):
    run,_=fixture(tmp_path);edit(run/'run_manifest.json',mutation)
    # Runtime dicts are separate serialized objects. Exercise coherent wrong ID.
    if match=='wrong experiment':
        edit(run/'training_manifest.json',lambda d:d['runtime_contract'].update(experiment_id='old'))
        edit(run/'run_manifest.json',lambda d:d['result']['runtime_contract'].update(experiment_id='old'))
    with pytest.raises(ValueError,match=match):r.summarize(run,project_root=tmp_path)


def test_wrong_checkpoint_branch_count_rejected(tmp_path):
    run,side=fixture(tmp_path);edit(side,lambda d:d['task_conditioned_hip_wheel_branch_counts'].update(global_policy_decisions=0))
    with pytest.raises(ValueError,match='branch counts'):r.summarize(run,project_root=tmp_path)


def test_wrong_geometry_cost_rejected(tmp_path):
    run,_=fixture(tmp_path);records=list(r.base.rows(run/'residual_and_projection_audit.jsonl'))
    records[0]['applied_audit']['reward_breakdown']['task_space_quality_sample_audit'][0]['weighted_geometry_cost']=1.
    jsonl(run/'residual_and_projection_audit.jsonl',records)
    with pytest.raises(ValueError,match='geometry arithmetic'):r.summarize(run,project_root=tmp_path)


def test_prefix_leak_rejected(tmp_path):
    run,_=fixture(tmp_path,suffix=True)
    jsonl(run/'prefix_evidence.jsonl',[dict(kind='checkpoint_prefix_decision',policy_credit=True)])
    with pytest.raises(ValueError,match='prefix log contains learner credit'):r.summarize(run,project_root=tmp_path)


def test_missing_actual_likelihood_rejected(tmp_path):
    run,_=fixture(tmp_path)
    # Missing-file fixture via rename, not recursive deletion.
    (run/'rollouts/update_001418_likelihood.json').rename(run/'rollouts/not_an_actual_hook.json')
    with pytest.raises(ValueError,match='likelihood hook'):r.summarize(run,project_root=tmp_path)


def test_terminal_failure_not_success_and_null_geometry_preserved(tmp_path):
    run,_=fixture(tmp_path);records=list(r.base.rows(run/'residual_and_projection_audit.jsonl'))
    last=records[-1];last['terminal']=True;info=last['applied_audit'];info['termination_reason']='BODY_COLLISION'
    sample=info['reward_breakdown']['task_space_quality_sample_audit'][0]
    sample.update(valid=False,weighted_geometry_cost=None,raw_geometry_cost=None,terminal_measurement_omitted=True)
    info['reward_breakdown']['families']['body_stability']=0.
    jsonl(run/'residual_and_projection_audit.jsonl',records)
    jsonl(run/'completed_episodes.jsonl',[dict(termination_reason='BODY_COLLISION',task_success=False,full_task_success=False)])
    edit(run/'training_manifest.json',lambda d:d['telemetry'].update(completed_episode_count=1))
    edit(run/'run_manifest.json',lambda d:d.update(result=r.read(run/'training_manifest.json')))
    out=r.summarize(run,project_root=tmp_path)
    assert out['first_terminal_outcome']['termination_reason']=='BODY_COLLISION'
    assert out['natural_P01_full_task_success_count']==0
    assert out['actual_geometry_quality_by_phase']['P06']['terminal_measurement_omitted']==1

