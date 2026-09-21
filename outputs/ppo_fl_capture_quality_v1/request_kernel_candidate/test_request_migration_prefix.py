"""Output-only CPU candidate: virtual reviewed bytes, never a deployable plan.

No Git commits, source checkpoint writes, Isaac, or physical success claim.
The version reader alone maps an uncommitted candidate to a TEST revision;
all byte hashes, actual AST scopes, plan reconstruction and RSL loading run.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from wlr50_clean.ppo import semantic_migration as m, semantic_training as t, semantic_cli as cli
from wlr50_clean.ppo.semantic_checkpoint_prefix_policy import build_frozen_checkpoint_prefix_policy
from wlr50_clean.ppo.semantic_policy_distribution import (
    HISTORY_QUARTER_TEMPERED_POLICY as OLD, HISTORY_REQUEST_CAP_TRANSITION_POLICY as NEW,
    ROLE_OBSERVATION_LAYOUT as LAYOUT, policy_contract)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REAL_CP = ROOT/'outputs/ppo_fl_capture_quality_v1/checkpoints/history/checkpoint_step_000180480.pt'
TEST_REVISION = 'c' * 40


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    threads, rng = torch.get_num_threads(), torch.get_rng_state()
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    yield
    torch.set_rng_state(rng)
    torch.set_num_threads(threads)


@pytest.fixture
def reviewed(monkeypatch):
    metadata = m.checkpoint_metadata(REAL_CP)
    old, new = metadata['runtime_contract'], copy.deepcopy(metadata['runtime_contract'])
    new['source_git_commit'] = TEST_REVISION
    data = {p:(HERE/p).read_bytes() for p in m.REQUEST_HISTORY_KERNEL_FILES}
    new['files'].update({p:hashlib.sha256(v).hexdigest() for p,v in data.items()})
    new['runtime_content_sha256'] = m.digest(new['files'])
    original = m._version_bytes
    def version_bytes(project_root, contract, relative, *, prefer_worktree=False):
        if contract['source_git_commit'] == TEST_REVISION:
            raw = data.get(relative, (ROOT/relative).read_bytes())
            if hashlib.sha256(raw).hexdigest() != contract['files'][relative]:
                raise ValueError('virtual candidate hash mismatch')
            return raw
        return original(ROOT, contract, relative, prefer_worktree=True)
    monkeypatch.setattr(m, '_version_bytes', version_bytes)
    return SimpleNamespace(old=old, new=new, data=data, metadata=metadata)


def plan(f, checkpoint=REAL_CP, **options):
    return m.build_migration_plan(checkpoint, f.new,
        allowed_changed_files=sorted(m.REQUEST_HISTORY_KERNEL_FILES),
        reason='CPU candidate verification only; TEST revision is not adoptable',
        request_history_kernel_review={'reason':'Exact six candidate AST scopes; all six configs unchanged',
            'reviewed_code_sha256':{p:f.new['files'][p] for p in sorted(m.REQUEST_HISTORY_KERNEL_FILES)}},
        project_root=ROOT, **options)


def verified_plan(f, checkpoint, tmp_path, monkeypatch):
    path = tmp_path/'cpu_test_only_plan.json'
    path.write_text(json.dumps(plan(f,checkpoint),indent=2),encoding='utf-8')
    original = m.validate_migration_plan
    monkeypatch.setattr(m,'validate_migration_plan',
        lambda cp,c,p:original(cp,c,p,project_root=ROOT))
    return m.validate_migration_plan(checkpoint,f.new,path)


def test_real_checkpoint_metadata_actual_candidate_scopes_json_roundtrip(reviewed,tmp_path,monkeypatch):
    before = m.file_sha(REAL_CP),m.file_sha(REAL_CP.with_name(REAL_CP.stem+'_manifest.json'))
    verified = verified_plan(reviewed,REAL_CP,tmp_path,monkeypatch)
    factor = verified['request_history_kernel_factor']
    assert factor['source_effective_learning_rate'] == factor['target_effective_learning_rate'] == 1e-5
    assert factor['migration_added_updates'] == factor['migration_added_policy_decisions'] == 0
    assert factor['counter_origin']['global_policy_decisions'] == 180480
    assert all(x['bytes_identical'] for x in factor['configuration_bindings'].values())
    assert len(factor['code_scope']) == 6
    assert all(x['protected_ast_identical'] for x in factor['code_scope'].values())
    assert 'not_behavior_equivalent' in factor['deterministic_same_weights_same_observation']
    assert before == (m.file_sha(REAL_CP),m.file_sha(REAL_CP.with_name(REAL_CP.stem+'_manifest.json')))


@pytest.mark.parametrize('module', ['semantic_history_actor','semantic_cli','semantic_checkpoint_prefix_policy'])
def test_unreviewed_runtime_region_rejected(reviewed,module):
    p=f'src/wlr50_clean/ppo/{module}.py'
    reviewed.data[p] += b'\nUNREVIEWED_PHYSICAL_CHANGE = True\n'
    reviewed.new['files'][p] = hashlib.sha256(reviewed.data[p]).hexdigest()
    reviewed.new['runtime_content_sha256'] = m.digest(reviewed.new['files'])
    with pytest.raises(ValueError,match='outside its named review scope'):
        plan(reviewed)


@pytest.mark.parametrize('fault',['config','namespace','same_revision','mixed','caps','observation_scale','graph'])
def test_single_factor_and_runtime_bindings_rejected(reviewed,fault):
    if fault in ('caps','observation_scale','graph'):
        import yaml
        cfg=ROOT/'configs/ppo_fl_capture_quality_v1'
        spec=yaml.safe_load((cfg/'stage_task_spec.yaml').read_text())
        execution=yaml.safe_load((cfg/'execution_profile.yaml').read_text())
        observation=json.loads((cfg/'observation_schema.json').read_text())
        if fault=='caps': execution['residual']['phase_caps_full12']['P06'][3] += 1
        elif fault=='graph': spec['stages']['P05']['next_phase']='P09'
        else: next(g for g in observation['feature_groups'] if g['name']=='previous_residual_full12')['scale'][3]=4
        with pytest.raises(ValueError):
            m._request_history_runtime_binding(spec,execution,observation,policy_contract(NEW,observation_layout=LAYOUT))
        return
    if fault=='config': reviewed.new['selected_configuration']['reward_config.yaml']['sha256']='f'*64
    elif fault=='namespace': reviewed.new['experiment_id']='rr_fix_v1'
    elif fault=='same_revision': reviewed.new['source_git_commit']=reviewed.old['source_git_commit']
    with pytest.raises(ValueError):
        plan(reviewed,**({'exploration_temperature_review':{}} if fault=='mixed' else {}))


@pytest.mark.parametrize('field',['history_center_semantics','target_policy_contract','target_runner_config','optimizer'])
def test_tampered_plan_rejected(reviewed,tmp_path,field):
    value=plan(reviewed)
    value['request_history_kernel_factor'][field]='tampered'
    path=tmp_path/'tampered.json';path.write_text(json.dumps(value),encoding='utf-8')
    with pytest.raises(ValueError,match='exactly bound'):
        m.validate_migration_plan(REAL_CP,reviewed.new,path,project_root=ROOT)


class ValidCore372:
    """Synthetic RSL ABI only; no contact/task/physics claim."""
    def __init__(self):
        self.calls,self.resets=0,0
        self.frame=SimpleNamespace(sim_time_s=0.)
    def observation(self,raw=None):
        values=[0.]*372;values[0]=1.;values[20]=self.tick/15/200
        if raw is not None: values[195:207]=[max(-19.,min(19.,v)) for v in raw]
        return tuple(values)
    def reset(self,*,seed=1001,options=None):
        self.resets+=1;self.tick=0;self.done=False;self.frame.sim_time_s=0.
        return self.observation()
    def step(self,raw):
        self.calls+=1;self.tick+=1;self.frame.sim_time_s=self.tick/15;self.done=self.tick==17
        return SimpleNamespace(observation=self.observation(raw),reward=1.+raw[0]-.1*raw[1]**2,
            terminated=self.done,truncated=False,info={'phase_id':'P01','raw_policy_action_full12':raw,
                'applied_action_full12':tuple(.1*v for v in raw),'task_success':False,
                'termination_reason':'synthetic_timeout' if self.done else None,
                'actuator_target_effect_audit':{'schema':'wlr50_clean.actuator_target_effect_audit.v1',
                    'verified':True,'actual_mapping_matches_dispatch':True,'setter_dispatch_targets_equal':True,
                    'same_tick_counterfactual':True,'raw_policy_action_full12':raw,'target_dtype':'torch.float32',
                    'changed_target_channel_count':12}})
    def telemetry_summary(self):return {'calls':self.calls,'resets':self.resets}


def runner(version):
    env=t.SemanticRslAdapter(ValidCore372(),seed=1001,device='cpu');env.cfg['semantic_version']='v3'
    model,_=t.construct_semantic_runner(env,seed=1001,device='cpu',initialize_actor=False,
        policy_version=version,observation_layout=LAYOUT)
    return model,env


def args(checkpoint,verified):
    return SimpleNamespace(checkpoint=checkpoint,semantic_version='v3',experiment_id='fl_capture_quality_v1',
        command='train',num_envs=1,seed=1001,device='cpu',new_mdp_warm_start=False,
        resume_migration=Path(verified['plan_path']),policy_distribution_migration=False,target_policy_version=None)


def prefix_record(checkpoint,loaded,arguments,contract):
    return {'checkpoint_path':str(checkpoint),'checkpoint_sha256':m.file_sha(checkpoint),
        'actor_parameter_sha256':loaded['actor_parameter_sha256'],
        'source_global_policy_decisions':loaded['global_policy_decisions'],'source_ppo_updates':loaded['ppo_updates'],
        'policy_contract':policy_contract(NEW,observation_layout=LAYOUT),
        'source_runtime_content_sha256':loaded['runtime_contract']['runtime_content_sha256'],
        **cli._request_history_prefix_provenance(arguments,contract,loaded)}


def test_official_cpu_load_prefix_update_reload_preserves_all_state(reviewed,tmp_path,monkeypatch):
    monkeypatch.setattr(cli,'PROJECT_ROOT',ROOT)
    t.seed_training_rngs(1001)
    source,env=runner(OLD)
    initial=t.train_semantic(source,env,run_dir=tmp_path/'source_run',output_root=tmp_path/'source_output',
        stage='full_episode',decisions=128,contract=reviewed.old,seed=1001)
    metadata=json.loads(Path(initial['checkpoints'][-1]['manifest']).read_text())
    for key in ('checkpoint_path','checkpoint_sha256','save_load_round_trip'):metadata.pop(key)
    source.alg.learning_rate=1e-5
    for group in source.alg.optimizer.param_groups:group.update(lr=1e-5,betas=(.87,.996),eps=2e-8)
    cp=tmp_path/'synthetic_old_quarter.pt';t.save_semantic_checkpoint(source,cp,metadata)
    saved=m.checkpoint_metadata(cp);assert source.alg.optimizer.state_dict()['state']
    verified=verified_plan(reviewed,cp,tmp_path,monkeypatch)
    arguments=args(cp,verified)
    cli._preflight_checkpoint(arguments,reviewed.new)
    assert cli._resolved_policy_version(arguments)==NEW
    for fault in ('partial_storage','pending_action','wrong_actor'):
        invalid,_=runner(NEW)
        if fault=='partial_storage':invalid.alg.storage.step=1
        elif fault=='pending_action':invalid.alg.transition.actions=torch.zeros(1,12)
        else:invalid._semantic_policy_version=OLD
        def forbidden(*args,**kwargs):
            raise AssertionError('invalid target reached tensor load')
        with monkeypatch.context() as invalid_scope:
            invalid_scope.setattr(t,'load_checkpoint_round_trip',forbidden)
            with pytest.raises((RuntimeError,ValueError)):
                t.load_semantic_checkpoint(invalid,cp,contract=reviewed.new,seed=1001,migration=verified)
    target,newenv=runner(NEW)
    loaded=t.load_semantic_checkpoint(target,cp,contract=reviewed.new,seed=1001,migration=verified)
    for role in ('actor','critic'):assert t.parameter_hash(getattr(source.alg,role))==t.parameter_hash(getattr(target.alg,role))
    assert t.state_hash(target.alg.optimizer.state_dict())==saved['optimizer_state_sha256']
    assert t.state_hash(t._normalizers(target))==saved['normalizer_state_sha256']
    assert t.capture_training_rng_state(seed=1001)==saved['training_rng_state']
    assert target.alg.learning_rate==t.optimizer_learning_rate(target)==1e-5
    assert target.alg.storage.step==0 and target.alg.transition.actions is None and newenv.core.calls==0
    for key in ('global_policy_decisions','ppo_updates','optimizer_steps'):assert loaded[key]==saved[key]
    record=prefix_record(cp,loaded,arguments,reviewed.new)
    rng=t.capture_training_rng_state(seed=1001)
    prefix=build_frozen_checkpoint_prefix_policy(target.alg.actor,record)
    assert len(prefix(newenv.core.observation()))==12
    assert t.capture_training_rng_state(seed=1001)==rng
    assert prefix.provenance['source_policy_contract']['version']==OLD
    assert prefix.provenance['effective_policy_contract']['version']==NEW
    for fault in ('missing_plan','wrong_hash','source_class','missing_source_contract'):
        bad=copy.deepcopy(record);actor=target.alg.actor
        if fault=='missing_plan':bad.pop('request_history_kernel_migration')
        elif fault=='wrong_hash':bad['request_history_kernel_migration']['plan_sha256']='0'*64
        elif fault=='source_class':actor=source.alg.actor
        else:bad.pop('source_policy_contract')
        with pytest.raises((RuntimeError,ValueError)):build_frozen_checkpoint_prefix_policy(actor,bad)
    continued=t.train_semantic(target,newenv,run_dir=tmp_path/'new_run',output_root=tmp_path/'new_output',
        stage='full_episode',decisions=128,contract=reviewed.new,seed=1001,resume_infos=loaded)
    final,finalenv=runner(NEW);finalcp=Path(continued['checkpoints'][-1]['checkpoint'])
    info=t.load_semantic_checkpoint(final,finalcp,contract=reviewed.new,seed=1001)
    assert (info['global_policy_decisions'],info['ppo_updates'],info['optimizer_steps'])==(256,2,40)
    assert info['policy_contract']==policy_contract(NEW,observation_layout=LAYOUT)
    assert t.parameter_hash(final.alg.actor)==t.parameter_hash(target.alg.actor)
    assert t.state_hash(final.alg.optimizer.state_dict())==t.state_hash(target.alg.optimizer.state_dict())
    assert info['resume_ancestry']['resume_migration']['request_history_kernel_factor']==verified['request_history_kernel_factor']
    exactargs=args(finalcp,verified);exactargs.resume_migration=None
    cli._preflight_checkpoint(exactargs,reviewed.new)
    exactrecord=prefix_record(finalcp,info,exactargs,reviewed.new)
    assert exactrecord['request_history_kernel_migration'] is None
    assert build_frozen_checkpoint_prefix_policy(final.alg.actor,exactrecord).provenance['source_policy_contract']['version']==NEW
    assert finalenv.core.calls==0
    # Existing unmodified video loader, both real code paths, no AppLauncher.
    from wlr50_clean.ppo.semantic_video_cli import checkpoint_loader
    from tensordict import TensorDict
    values=list(finalenv.core.observation())
    values[:13]=[0.]*13;values[5]=1.;values[20]=0.;values[158:163]=[1.]*5
    values[195:207]=[.4]*12;values[207:219]=[.01]*12
    inputs=torch.tensor([values],dtype=torch.float32)
    observation=TensorDict({'policy':inputs,'critic':inputs.clone()},batch_size=[1])
    for stochastic in (False,True):
        evaluation=args(finalcp,verified);evaluation.resume_migration=None
        evaluation.command='eval';evaluation.seed=4001
        evaluation.stochastic_policy=stochastic;evaluation.policy_seed=51001 if stochastic else None
        cli._preflight_checkpoint(evaluation,reviewed.new)
        action,proof,unchanged=checkpoint_loader(evaluation,reviewed.new)(tuple(values))
        rng=torch.get_rng_state()
        with torch.inference_mode():expected=final.alg.actor(observation,stochastic_output=stochastic)[0].tolist()
        torch.set_rng_state(rng)
        assert action(tuple(values),0)==tuple(expected)
        assert proof['policy_version']==NEW and proof['optimizer_updates']==0
        assert action.last_request['policy_version']==NEW
        assert sum(action.last_request['cap_transition_gate_full12'])==10
        unchanged()
