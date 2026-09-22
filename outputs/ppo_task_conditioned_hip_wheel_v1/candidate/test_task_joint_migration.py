"""Candidate-only CPU contracts; synthetic target revision, zero physical/PPO credit."""
from __future__ import annotations
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT/'src'))
from candidate_bootstrap import CANDIDATE
from wlr50_clean.ppo import semantic_migration as m, semantic_cli as cli
from wlr50_clean.ppo.semantic_policy_distribution import CONFIG_NAMES, TASK_CONDITIONED_HIP_WHEEL_POLICY
from wlr50_clean.ppo import semantic_checkpoint_prefix_policy as prefix, semantic_training as training
assert m is CANDIDATE['semantic_migration'] and cli is CANDIDATE['semantic_cli']

CP = ROOT/'outputs/ppo_fl_capture_quality_v1/checkpoints/history/checkpoint_step_000185856.pt'
REVISION = 'd'*40


@pytest.fixture
def candidate(monkeypatch):
    metadata = m.checkpoint_metadata(CP)
    old, new = metadata['runtime_contract'], copy.deepcopy(metadata['runtime_contract'])
    data = {str(p.relative_to(HERE)).replace('\\','/'): p.read_bytes()
            for p in (HERE/'src/wlr50_clean/ppo').glob('*.py')}
    data.update({f'configs/ppo_task_conditioned_hip_wheel_v1/{name}':
        (HERE/'configs/ppo_task_conditioned_hip_wheel_v1'/name).read_bytes() for name in CONFIG_NAMES})
    new.update(source_git_commit=REVISION, experiment_id='task_conditioned_hip_wheel_v1')
    new['files'].update({p:hashlib.sha256(b).hexdigest() for p,b in data.items()})
    new['selected_configuration']={n:{'path':f'configs/ppo_task_conditioned_hip_wheel_v1/{n}',
        'sha256':new['files'][f'configs/ppo_task_conditioned_hip_wheel_v1/{n}']} for n in CONFIG_NAMES}
    new['runtime_content_sha256']=m.digest(new['files'])
    original_bytes, original_sha, original_run = m._version_bytes, m.file_sha, subprocess.run
    def version_bytes(root, contract, relative, *, prefer_worktree=False):
        if contract['source_git_commit'] == REVISION:
            value = data[relative] if relative in data else (ROOT/relative).read_bytes()
            if hashlib.sha256(value).hexdigest() != contract['files'][relative]:
                raise ValueError('synthetic target bytes differ')
            return value
        return original_bytes(ROOT, contract, relative, prefer_worktree=True)
    def file_sha(path):
        path=Path(path)
        relative=str(path.relative_to(ROOT)).replace('\\','/') if path.is_relative_to(ROOT) else None
        return hashlib.sha256(data[relative]).hexdigest() if relative in data else original_sha(path)
    def run(command,*a,**kw):
        if command[-2:]==['rev-parse','HEAD']:
            return SimpleNamespace(stdout=REVISION)
        return original_run(command,*a,**kw)
    monkeypatch.setattr(m,'_version_bytes',version_bytes)
    monkeypatch.setattr(m,'file_sha',file_sha)
    monkeypatch.setattr(m.subprocess,'run',run)
    return SimpleNamespace(old=old,new=new,data=data,metadata=metadata)


def plan(f, **options):
    delta=sorted(p for p in f.old['files'].keys()|f.new['files'].keys() if f.old['files'].get(p)!=f.new['files'].get(p))
    code={p:f.new['files'][p] for p in delta if p.startswith(('src/','scripts/'))}
    return m.build_migration_plan(CP,f.new,allowed_changed_files=delta,
        reason='CPU-only synthetic target review, not an adoptable plan', project_root=ROOT,
        task_conditioned_hip_wheel_review={'reason':'joint reward/potential and task sigma; physical control unchanged',
            'reviewed_code_sha256':code}, **options)


def mutate(f,path,transform):
    f.data[path]=transform(f.data[path]);f.new['files'][path]=hashlib.sha256(f.data[path]).hexdigest()
    for binding in f.new['selected_configuration'].values():
        if binding['path']==path:binding['sha256']=f.new['files'][path]
    f.new['runtime_content_sha256']=m.digest(f.new['files'])


def test_actual_source_joint_contract(candidate):
    result=plan(candidate);factor=result['task_conditioned_hip_wheel_factor']
    assert factor['counter_origin']=={'global_policy_decisions':185856,'ppo_updates':1417,'optimizer_steps':28340}
    assert factor['target_policy_version']==TASK_CONDITIONED_HIP_WHEEL_POLICY
    assert factor['source_effective_learning_rate']==factor['target_effective_learning_rate']==1e-5
    assert factor['kernel_changed'] and factor['reward_changed'] and not factor['same_mdp_claimed']
    assert not factor['task_acceptance_changed'] and not factor['nominal_control_changed']
    assert all(v['protected_ast_identical'] for v in factor['quality_code_scope'].values())
    assert sum(x['bytes_identical'] for x in factor['configuration_bindings'].values())==4
    assert result['discard_old_rollout_storage'] and factor['migration_added_updates']==0


@pytest.mark.parametrize('fault',['namespace','missing_config','wrong_policy','zero_lr','bool_lr','mixed','hard_spec','action_capacity','nominal_class','observation_other','reward_other'])
def test_rejects_unrelated_changes(candidate,monkeypatch,fault):
    f=candidate
    if fault=='namespace': f.new['experiment_id']='fl_capture_quality_v1'
    elif fault=='missing_config':f.new['selected_configuration'].pop('quality_score.yaml')
    elif fault in ('wrong_policy','zero_lr','bool_lr'):
        metadata=copy.deepcopy(f.metadata)
        if fault=='wrong_policy': metadata['policy_contract']['version']=TASK_CONDITIONED_HIP_WHEEL_POLICY
        else:metadata['optimizer_learning_rate']=False if fault=='bool_lr' else 0
        monkeypatch.setattr(m,'checkpoint_metadata',lambda _:metadata)
    elif fault=='hard_spec':mutate(f,'configs/ppo_task_conditioned_hip_wheel_v1/stage_task_spec.yaml',lambda b:b.replace(b'minimum_air_samples: 2',b'minimum_air_samples: 1'))
    elif fault=='action_capacity':mutate(f,'configs/ppo_task_conditioned_hip_wheel_v1/execution_profile.yaml',lambda b:b.replace(b'servo_rate_deg_s: 60.0',b'servo_rate_deg_s: 61.0'))
    elif fault=='nominal_class':mutate(f,'src/wlr50_clean/ppo/semantic_supervisor.py',lambda b:b.replace(b'class NominalMotionProvider:',b'class NominalMotionProvider:\n    unsafe_candidate = True'))
    elif fault=='observation_other':mutate(f,'src/wlr50_clean/ppo/semantic_observation.py',lambda b:b+b'\nUNKNOWN_GEOMETRY_OVERRIDE = True\n')
    elif fault=='reward_other':mutate(f,'src/wlr50_clean/ppo/semantic_reward.py',lambda b:b+b'\nUNKNOWN_REWARD_OVERRIDE = True\n')
    with pytest.raises((ValueError,KeyError)):
        plan(f,**({'archive_only_exact_bytes_review':{'reason':'mixed'}} if fault=='mixed' else {}))


def test_plan_revalidation_rejects_tampered_semantics(candidate,tmp_path):
    result=plan(candidate);result['task_conditioned_hip_wheel_factor']['reward_changed']=False
    path=tmp_path/'tampered.json';path.write_text(json.dumps(result),encoding='utf-8')
    with pytest.raises(ValueError,match='exactly bound'):
        m.validate_migration_plan(CP,candidate.new,path,project_root=ROOT)


def test_old_migration_functions_unchanged():
    original=ast.parse((ROOT/'src/wlr50_clean/ppo/semantic_migration.py').read_text())
    changed=ast.parse((HERE/'src/wlr50_clean/ppo/semantic_migration.py').read_text())
    funcs={n.name:ast.dump(n,include_attributes=False) for n in changed.body if isinstance(n,ast.FunctionDef)}
    for node in original.body:
        if isinstance(node,ast.FunctionDef) and node.name not in ('experiment_namespace','build_migration_plan','validate_migration_plan'):
            assert funcs[node.name]==ast.dump(node,include_attributes=False),node.name


def archive_plan():
    metadata=m.checkpoint_metadata(CP);target=copy.deepcopy(metadata['runtime_contract'])
    target['source_git_commit']=subprocess.run(['git','-C',str(ROOT),'rev-parse','HEAD'],
        check=True,capture_output=True,text=True).stdout.strip()
    result=m.build_migration_plan(CP,target,allowed_changed_files=[],reason='CPU archive-only verification',
        archive_only_exact_bytes_review={'reason':'only the actual archive HEAD changed'},project_root=ROOT)
    return metadata,target,result


def test_real_archive_only_exact_bytes_and_old_policy():
    metadata,target,result=archive_plan()
    factor=result['archive_only_exact_bytes_factor']
    assert result['allowed_changed_files']==[] and result['changed_file_hashes']=={}
    assert factor['source_policy_contract']==factor['target_policy_contract']==metadata['policy_contract']
    assert factor['runtime_bytes_identical'] and not factor['kernel_changed']
    assert result['source_runtime_content_sha256']==result['target_runtime_content_sha256']


@pytest.mark.parametrize('fault',['config','unknown_policy','changed_files','mixed'])
def test_archive_rejects_non_archive_boundary(monkeypatch,fault):
    metadata,target,_=archive_plan()
    if fault=='config':target['task_timeout_s']=201.
    elif fault=='unknown_policy':
        metadata['policy_contract']['version']='unsupported_same_dim'
        monkeypatch.setattr(m,'checkpoint_metadata',lambda _:metadata)
    with pytest.raises((ValueError,KeyError)):
        m.build_migration_plan(CP,target,allowed_changed_files=['src/change.py'] if fault=='changed_files' else [],
            reason='CPU invalid archive',archive_only_exact_bytes_review={'reason':'invalid'},project_root=ROOT,
            **({'task_conditioned_hip_wheel_review':{}} if fault=='mixed' else {}))


def test_joint_preflight_and_prefix_provenance(candidate,tmp_path,monkeypatch):
    result=plan(candidate);path=tmp_path/'cpu_plan.json';path.write_text(json.dumps(result),encoding='utf-8')
    original=m.validate_migration_plan
    monkeypatch.setattr(m,'validate_migration_plan',lambda cp,c,p:original(cp,c,p,project_root=ROOT))
    monkeypatch.setattr(cli,'_request_paths',lambda _: (tmp_path,tmp_path,HERE/'configs/ppo_task_conditioned_hip_wheel_v1'))
    args=SimpleNamespace(checkpoint=CP,semantic_version='v3',experiment_id='task_conditioned_hip_wheel_v1',
        command='train',num_envs=1,seed=1001,device='cuda:0',new_mdp_warm_start=False,
        resume_migration=path,policy_distribution_migration=False,target_policy_version=None)
    cli._preflight_checkpoint(args,candidate.new)
    assert cli._resolved_policy_version(args)==TASK_CONDITIONED_HIP_WHEEL_POLICY
    evidence=cli._request_history_prefix_provenance(args,candidate.new,candidate.metadata)
    record={'checkpoint_path':str(CP),'checkpoint_sha256':candidate.metadata['checkpoint_sha256'],
        'actor_parameter_sha256':candidate.metadata['actor_parameter_sha256'],
        'source_global_policy_decisions':185856,'source_ppo_updates':1417,
        'policy_contract':result['task_conditioned_hip_wheel_factor']['target_policy_contract'],
        'source_runtime_content_sha256':candidate.old['runtime_content_sha256'],**evidence}
    assert prefix._source_record(record)==record
    for fault in ('no_plan','bad_plan','source_relabel','mixed'):
        bad=copy.deepcopy(record)
        if fault=='no_plan':bad.pop('task_conditioned_hip_wheel_migration')
        elif fault=='bad_plan':bad['task_conditioned_hip_wheel_migration']['plan_sha256']='0'*64
        elif fault=='source_relabel':bad['source_policy_contract']=bad['policy_contract']
        else:bad['physical_innovation_sigma_migration']={}
        with pytest.raises(ValueError):prefix._source_record(bad)
    exact=copy.deepcopy(record)
    exact.update(source_policy_contract=exact['policy_contract'],
        source_runtime_content_sha256=exact['effective_runtime_content_sha256'],task_conditioned_hip_wheel_migration=None)
    assert prefix._source_record(exact)==exact


class CPUCore:
    """Synthetic algorithm fixture; native fields are test doubles, not physical evidence."""
    def __init__(self):self.calls=0;self.frame=SimpleNamespace(sim_time_s=0.)
    def observation(self,raw=None):
        x=[0.]*372;x[5]=1.;x[20]=.01+self.tick/3000;x[158:163]=[1.]*5
        if raw is not None:x[195:207]=[max(-19.,min(19.,v)) for v in raw]
        return tuple(x)
    def reset(self,*,seed=1001,options=None):
        self.tick=0;self.frame.sim_time_s=0.;return self.observation()
    def step(self,raw):
        self.tick+=1;self.calls+=1;self.frame.sim_time_s=self.tick/15
        done=self.tick==17
        return SimpleNamespace(observation=self.observation(raw),reward=1+raw[0]-.1*raw[1]**2,
            terminated=done,truncated=False,info={'phase_id':'P06','raw_policy_action_full12':raw,
            'applied_action_full12':tuple(.1*v for v in raw),'task_success':False,
            'termination_reason':'CPU_TEST_DOUBLE_NOT_PHYSICS' if done else None,
            'actuator_target_effect_audit':{'schema':'wlr50_clean.actuator_target_effect_audit.v1',
                'verified':True,'actual_mapping_matches_dispatch':True,'setter_dispatch_targets_equal':True,
                'same_tick_counterfactual':True,'raw_policy_action_full12':raw,'target_dtype':'torch.float32',
                'changed_target_channel_count':12}})
    def telemetry_summary(self):return {'synthetic_CPU_test_only':True,'calls':self.calls}


def test_joint_true_validator_official_load_prefix_update_and_exact_resume(candidate,tmp_path,monkeypatch):
    import torch
    from wlr50_clean.ppo.semantic_policy_distribution import FR_KNEE_PHYSICAL_INNOVATION_POLICY as OLD
    rng,threads=torch.get_rng_state(),torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    monkeypatch.setattr(training,'__file__',str(ROOT/'src/wlr50_clean/ppo/semantic_training.py'))
    def runner(version):
        env=training.SemanticRslAdapter(CPUCore(),seed=1001,device='cpu');env.cfg['semantic_version']='v3'
        model,_=training.construct_semantic_runner(env,seed=1001,device='cpu',initialize_actor=False,
            policy_version=version,observation_layout='diagonal_transfer_state_v1')
        return model,env
    try:
        training.seed_training_rngs(1001)
        source,env=runner(OLD)
        result=training.train_semantic(source,env,run_dir=tmp_path/'old_run',output_root=tmp_path/'old_output',
            stage='full_episode',decisions=128,contract=candidate.old,seed=1001)
        metadata=json.loads(Path(result['checkpoints'][-1]['manifest']).read_text())
        for key in ('checkpoint_path','checkpoint_sha256','save_load_round_trip'):metadata.pop(key)
        source.alg.learning_rate=2.3e-5
        for group in source.alg.optimizer.param_groups:group.update(lr=2.3e-5,betas=(.87,.996),eps=2e-8)
        cp=tmp_path/'CPU_synthetic_FR_source.pt';training.save_semantic_checkpoint(source,cp,metadata)
        saved=m.checkpoint_metadata(cp)
        monkeypatch.setattr(sys.modules[__name__],'CP',cp)
        supplied=plan(candidate);path=tmp_path/'CPU_joint_plan.json';path.write_text(json.dumps(supplied),encoding='utf-8')
        original=m.validate_migration_plan
        monkeypatch.setattr(m,'validate_migration_plan',lambda checkpoint,c,p:original(checkpoint,c,p,project_root=ROOT))
        verified=m.validate_migration_plan(cp,candidate.new,path)
        monkeypatch.setattr(cli,'_request_paths',lambda _: (tmp_path,tmp_path,HERE/'configs/ppo_task_conditioned_hip_wheel_v1'))
        args=SimpleNamespace(checkpoint=cp,semantic_version='v3',experiment_id='task_conditioned_hip_wheel_v1',
            command='train',num_envs=1,seed=1001,device='cpu',new_mdp_warm_start=False,resume_migration=path,
            policy_distribution_migration=False,target_policy_version=None)
        cli._preflight_checkpoint(args,candidate.new)
        for fault in ('pending','partial','wrong_actor'):
            invalid,_=runner(OLD if fault=='wrong_actor' else TASK_CONDITIONED_HIP_WHEEL_POLICY)
            if fault=='pending':invalid.alg.transition.actions=torch.zeros(1,12)
            if fault=='partial':invalid.alg.storage.step=1
            with pytest.raises((RuntimeError,ValueError)):
                training.load_semantic_checkpoint(invalid,cp,contract=candidate.new,seed=1001,migration=verified)
        target,newenv=runner(TASK_CONDITIONED_HIP_WHEEL_POLICY)
        loaded=training.load_semantic_checkpoint(target,cp,contract=candidate.new,seed=1001,migration=verified)
        assert newenv.core.calls==0 and target.alg.storage.step==0 and target.alg.transition.actions is None
        assert target.alg.learning_rate==training.optimizer_learning_rate(target)==2.3e-5
        assert all(g['lr']==2.3e-5 for g in target.alg.optimizer.param_groups)
        assert training.state_hash(target.alg.optimizer.state_dict())==saved['optimizer_state_sha256']
        assert training.state_hash(training._normalizers(target))==saved['normalizer_state_sha256']
        for key in ('global_policy_decisions','ppo_updates','optimizer_steps'):assert loaded[key]==saved[key]
        assert training.capture_training_rng_state(seed=1001)==saved['training_rng_state']
        assert training.parameter_hash(target.alg.actor)==saved['actor_parameter_sha256']
        assert training.parameter_hash(target.alg.critic)==saved['critic_parameter_sha256']
        record={'checkpoint_path':str(cp),'checkpoint_sha256':saved['checkpoint_sha256'],
            'actor_parameter_sha256':saved['actor_parameter_sha256'],
            'source_global_policy_decisions':saved['global_policy_decisions'],'source_ppo_updates':saved['ppo_updates'],
            'policy_contract':supplied['task_conditioned_hip_wheel_factor']['target_policy_contract'],
            'source_runtime_content_sha256':candidate.old['runtime_content_sha256'],
            **cli._request_history_prefix_provenance(args,candidate.new,loaded)}
        frozen=prefix.build_frozen_checkpoint_prefix_policy(target.alg.actor,record)
        assert len(frozen(newenv.core.observation()))==12
        assert training.capture_training_rng_state(seed=1001)==saved['training_rng_state']
        continued=training.train_semantic(target,newenv,run_dir=tmp_path/'new_run',output_root=tmp_path/'new_output',
            stage='full_episode',decisions=128,contract=candidate.new,seed=1001,resume_infos=loaded)
        finalcp=Path(continued['checkpoints'][-1]['checkpoint']);final,finalenv=runner(TASK_CONDITIONED_HIP_WHEEL_POLICY)
        finalinfo=training.load_semantic_checkpoint(final,finalcp,contract=candidate.new,seed=1001)
        assert (finalinfo['global_policy_decisions'],finalinfo['ppo_updates'],finalinfo['optimizer_steps'])==(256,2,40)
        assert finalinfo['task_conditioned_hip_wheel_branch_counts']=={'global_policy_decisions':128,'ppo_updates':1,'optimizer_steps':20}
        assert training.parameter_hash(final.alg.actor)==training.parameter_hash(target.alg.actor)
        assert training.state_hash(final.alg.optimizer.state_dict())==training.state_hash(target.alg.optimizer.state_dict())
        args.checkpoint=finalcp;args.resume_migration=None;cli._preflight_checkpoint(args,candidate.new)
        record.update(checkpoint_path=str(finalcp),checkpoint_sha256=m.file_sha(finalcp),
            actor_parameter_sha256=finalinfo['actor_parameter_sha256'],source_global_policy_decisions=256,
            source_ppo_updates=2,source_runtime_content_sha256=candidate.new['runtime_content_sha256'],
            **cli._request_history_prefix_provenance(args,candidate.new,finalinfo))
        assert prefix.build_frozen_checkpoint_prefix_policy(final.alg.actor,record).provenance['source_policy_contract']==record['policy_contract']
        assert finalenv.core.calls==0
    finally:
        torch.set_rng_state(rng);torch.set_num_threads(threads)
