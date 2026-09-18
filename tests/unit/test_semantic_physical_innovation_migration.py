"""CPU-only explicit sigma migration/prefix integration; zero physics credit.

Real sealed checkpoint metadata is optional and read-only. Synthetic CPU
checkpoints live only in pytest temporary directories, never output history.
"""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('rsl_rl')
from wlr50_clean.ppo import semantic_migration as m, semantic_training as t, semantic_cli as cli
from wlr50_clean.ppo.semantic_checkpoint_prefix_policy import build_frozen_checkpoint_prefix_policy
from wlr50_clean.ppo.semantic_policy_distribution import (
    HISTORY_REQUEST_CAP_TRANSITION_POLICY as OLD, FR_KNEE_PHYSICAL_INNOVATION_POLICY as NEW,
    ROLE_OBSERVATION_LAYOUT as LAYOUT, policy_contract)

ROOT = Path(__file__).resolve().parents[2]
REAL_CP = ROOT/'outputs/ppo_fl_capture_quality_v1/checkpoints/history/checkpoint_step_000184320.pt'
TEST_REVISION = 'c' * 40


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    threads, rng = torch.get_num_threads(), torch.get_rng_state()
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    yield
    torch.set_rng_state(rng); torch.set_num_threads(threads)


@pytest.fixture
def reviewed(monkeypatch):
    if not REAL_CP.is_file():
        pytest.skip('optional sealed source checkpoint unavailable; no model fabricated')
    metadata = m.checkpoint_metadata(REAL_CP)
    old, new = metadata['runtime_contract'], copy.deepcopy(metadata['runtime_contract'])
    new['source_git_commit'] = TEST_REVISION
    data = {p:(ROOT/p).read_bytes() for p in m.PHYSICAL_INNOVATION_SIGMA_FILES}
    new['files'].update({p:hashlib.sha256(v).hexdigest() for p,v in data.items()})
    new['runtime_content_sha256'] = m.digest(new['files'])
    original = m._version_bytes
    def version_bytes(project_root, contract, relative, *, prefer_worktree=False):
        if contract['source_git_commit'] == TEST_REVISION:
            raw = data.get(relative, (ROOT/relative).read_bytes())
            if hashlib.sha256(raw).hexdigest() != contract['files'][relative]:
                raise ValueError('virtual reviewed bytes mismatch')
            return raw
        return original(ROOT, contract, relative, prefer_worktree=True)
    monkeypatch.setattr(m, '_version_bytes', version_bytes)
    return SimpleNamespace(old=old, new=new, data=data, metadata=metadata)


def plan(f, checkpoint=REAL_CP, **options):
    return m.build_migration_plan(checkpoint, f.new,
        allowed_changed_files=sorted(m.PHYSICAL_INNOVATION_SIGMA_FILES),
        reason='CPU TEST revision only, not an adoptable committed runtime',
        physical_innovation_sigma_review={'reason':'Only FR knee P06+ effective log-sigma factor; mean and six configs unchanged',
            'reviewed_code_sha256':{p:f.new['files'][p] for p in sorted(m.PHYSICAL_INNOVATION_SIGMA_FILES)}},
        project_root=ROOT, **options)


def verified_plan(f, checkpoint, tmp_path, monkeypatch):
    path = tmp_path/'cpu_only_plan.json'
    path.write_text(json.dumps(plan(f,checkpoint),indent=2),encoding='utf-8')
    original = m.validate_migration_plan
    monkeypatch.setattr(m,'validate_migration_plan',lambda cp,c,p:original(cp,c,p,project_root=ROOT))
    return m.validate_migration_plan(checkpoint,f.new,path)


def test_real_latest_metadata_and_exact_scoped_plan(reviewed,tmp_path,monkeypatch):
    verified = verified_plan(reviewed,REAL_CP,tmp_path,monkeypatch)
    factor = verified['physical_innovation_sigma_factor']
    assert factor['counter_origin']['global_policy_decisions'] == 184320
    assert factor['source_effective_learning_rate'] == factor['target_effective_learning_rate'] == reviewed.metadata['optimizer_learning_rate']
    assert factor['migration_added_updates'] == factor['migration_added_policy_decisions'] == 0
    assert factor['action_ranges_changed'] is factor['nominal_control_changed'] is False
    assert all(x['bytes_identical'] for x in factor['configuration_bindings'].values())
    assert len(factor['code_scope']) == 6
    assert all(x['protected_ast_identical'] for x in factor['code_scope'].values())
    assert verified['discard_old_rollout_storage']


@pytest.mark.parametrize('fault',['config','namespace','same_revision','mixed','source_version','zero_LR','bool_LR'])
def test_boundary_rejects_unrelated_or_invalid_state(reviewed,fault,monkeypatch):
    if fault=='config': reviewed.new['selected_configuration']['reward_config.yaml']['sha256']='f'*64
    elif fault=='namespace': reviewed.new['experiment_id']='residual_rr_fix_v1'
    elif fault=='same_revision': reviewed.new['source_git_commit']=reviewed.old['source_git_commit']
    elif fault in ('source_version','zero_LR','bool_LR'):
        metadata=copy.deepcopy(reviewed.metadata)
        if fault=='source_version':metadata['policy_contract']=policy_contract(NEW,observation_layout=LAYOUT)
        else:metadata['optimizer_learning_rate']=0 if fault=='zero_LR' else True
        monkeypatch.setattr(m,'checkpoint_metadata',lambda cp:metadata)
    with pytest.raises(ValueError):
        plan(reviewed,**({'request_history_kernel_review':{}} if fault=='mixed' else {}))


@pytest.mark.parametrize('module',['semantic_history_actor','semantic_cli','semantic_checkpoint_prefix_policy'])
def test_unreviewed_AST_region_rejected(reviewed,module):
    path=f'src/wlr50_clean/ppo/{module}.py'
    reviewed.data[path]+=b'\nUNREVIEWED_CONTROL_CHANGE = True\n'
    reviewed.new['files'][path]=hashlib.sha256(reviewed.data[path]).hexdigest()
    reviewed.new['runtime_content_sha256']=m.digest(reviewed.new['files'])
    with pytest.raises(ValueError,match='outside its named review scope'):plan(reviewed)


@pytest.mark.parametrize('field',['sigma_scaling_semantics','target_policy_contract','target_effective_learning_rate','optimizer'])
def test_tampered_factor_rejected(reviewed,tmp_path,field):
    supplied=plan(reviewed);supplied['physical_innovation_sigma_factor'][field]='tampered'
    path=tmp_path/'tampered.json';path.write_text(json.dumps(supplied),encoding='utf-8')
    with pytest.raises(ValueError,match='exactly bound'):
        m.validate_migration_plan(REAL_CP,reviewed.new,path,project_root=ROOT)


class CPUCore:
    def __init__(self):self.calls=0;self.frame=SimpleNamespace(sim_time_s=0.)
    def observation(self,raw=None):
        values=[0.]*372;values[5]=1.;values[20]=self.tick/3000;values[158:163]=[1.]*5
        if raw is not None:values[195:207]=[max(-19.,min(19.,v)) for v in raw]
        return tuple(values)
    def reset(self,*,seed=1001,options=None):
        self.tick=0;self.done=False;self.frame.sim_time_s=0.;return self.observation()
    def step(self,raw):
        self.calls+=1;self.tick+=1;self.frame.sim_time_s=self.tick/15;self.done=self.tick==17
        return SimpleNamespace(observation=self.observation(raw),reward=1.+raw[0]-.1*raw[1]**2,
            terminated=self.done,truncated=False,info={'phase_id':'P06','raw_policy_action_full12':raw,
                'applied_action_full12':tuple(.1*v for v in raw),'task_success':False,
                'termination_reason':'synthetic_timeout' if self.done else None,
                'actuator_target_effect_audit':{'schema':'wlr50_clean.actuator_target_effect_audit.v1',
                    'verified':True,'actual_mapping_matches_dispatch':True,'setter_dispatch_targets_equal':True,
                    'same_tick_counterfactual':True,'raw_policy_action_full12':raw,'target_dtype':'torch.float32',
                    'changed_target_channel_count':12}})
    def telemetry_summary(self):return {'calls':self.calls}


def runner(version):
    env=t.SemanticRslAdapter(CPUCore(),seed=1001,device='cpu');env.cfg['semantic_version']='v3'
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


def test_official_load_fresh_rollout_frozen_prefix_and_exact_new_resume(reviewed,tmp_path,monkeypatch):
    monkeypatch.setattr(cli,'PROJECT_ROOT',ROOT)
    t.seed_training_rngs(1001)
    source,env=runner(OLD)
    initial=t.train_semantic(source,env,run_dir=tmp_path/'source_run',output_root=tmp_path/'source_output',
        stage='full_episode',decisions=128,contract=reviewed.old,seed=1001)
    metadata=json.loads(Path(initial['checkpoints'][-1]['manifest']).read_text())
    for key in ('checkpoint_path','checkpoint_sha256','save_load_round_trip'):metadata.pop(key)
    source.alg.learning_rate=2.3e-5
    for group in source.alg.optimizer.param_groups:group.update(lr=2.3e-5,betas=(.87,.996),eps=2e-8)
    cp=tmp_path/'synthetic_old_request.pt';t.save_semantic_checkpoint(source,cp,metadata)
    saved=m.checkpoint_metadata(cp);assert source.alg.optimizer.state_dict()['state']
    verified=verified_plan(reviewed,cp,tmp_path,monkeypatch)
    arguments=args(cp,verified);cli._preflight_checkpoint(arguments,reviewed.new)
    assert cli._resolved_policy_version(arguments)==NEW
    for fault in ('partial_storage','pending_action','wrong_actor'):
        invalid,_=runner(NEW)
        if fault=='partial_storage':invalid.alg.storage.step=1
        elif fault=='pending_action':invalid.alg.transition.actions=torch.zeros(1,12)
        else:invalid._semantic_policy_version=OLD
        with pytest.raises((RuntimeError,ValueError)):
            t.load_semantic_checkpoint(invalid,cp,contract=reviewed.new,seed=1001,migration=verified)
    target,newenv=runner(NEW)
    loaded=t.load_semantic_checkpoint(target,cp,contract=reviewed.new,seed=1001,migration=verified)
    for role in ('actor','critic'):assert t.parameter_hash(getattr(source.alg,role))==t.parameter_hash(getattr(target.alg,role))
    assert t.state_hash(target.alg.optimizer.state_dict())==saved['optimizer_state_sha256']
    assert t.state_hash(t._normalizers(target))==saved['normalizer_state_sha256']
    assert t.capture_training_rng_state(seed=1001)==saved['training_rng_state']
    assert target.alg.learning_rate==t.optimizer_learning_rate(target)==2.3e-5
    assert target.alg.storage.step==0 and target.alg.transition.actions is None and newenv.core.calls==0
    for key in ('global_policy_decisions','ppo_updates','optimizer_steps'):assert loaded[key]==saved[key]
    record=prefix_record(cp,loaded,arguments,reviewed.new)
    rng=t.capture_training_rng_state(seed=1001)
    prefix=build_frozen_checkpoint_prefix_policy(target.alg.actor,record)
    assert len(prefix(newenv.core.observation()))==12
    assert t.capture_training_rng_state(seed=1001)==rng
    assert prefix.provenance['source_policy_contract']['version']==OLD
    assert prefix.provenance['effective_policy_contract']['version']==NEW
    for fault in ('missing_plan','wrong_hash','source_class','missing_source_contract','mixed_provenance'):
        bad=copy.deepcopy(record);actor=target.alg.actor
        if fault=='missing_plan':bad.pop('physical_innovation_sigma_migration')
        elif fault=='wrong_hash':bad['physical_innovation_sigma_migration']['plan_sha256']='0'*64
        elif fault=='source_class':actor=source.alg.actor
        elif fault=='mixed_provenance':bad['request_history_kernel_migration']={}
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
    assert info['resume_ancestry']['resume_migration']['physical_innovation_sigma_factor']==verified['physical_innovation_sigma_factor']
    exactargs=args(finalcp,verified);exactargs.resume_migration=None
    cli._preflight_checkpoint(exactargs,reviewed.new)
    record=prefix_record(finalcp,info,exactargs,reviewed.new)
    assert record['physical_innovation_sigma_migration'] is None
    assert build_frozen_checkpoint_prefix_policy(final.alg.actor,record).provenance['source_policy_contract']['version']==NEW
    assert finalenv.core.calls==0
    # Same committed-loader APIs used by deterministic/stochastic video, but
    # synthetic observations only: no AppLauncher, simulation or task claim.
    from wlr50_clean.ppo.semantic_video_cli import checkpoint_loader
    from tensordict import TensorDict
    values=finalenv.core.observation()
    tensor=torch.tensor([values],dtype=torch.float32)
    observation=TensorDict({'policy':tensor,'critic':tensor.clone()},batch_size=[1])
    for stochastic in (False,True):
        evaluation=args(finalcp,verified);evaluation.resume_migration=None
        evaluation.command='eval';evaluation.seed=4001
        evaluation.stochastic_policy=stochastic;evaluation.policy_seed=51001 if stochastic else None
        cli._preflight_checkpoint(evaluation,reviewed.new)
        action,proof,unchanged=checkpoint_loader(evaluation,reviewed.new)(values)
        rng=torch.get_rng_state()
        with torch.inference_mode():expected=final.alg.actor(observation,stochastic_output=stochastic)[0].tolist()
        torch.set_rng_state(rng)
        assert action(values,0)==tuple(expected)
        assert proof['policy_version']==NEW and proof['optimizer_updates']==0
        assert action.last_request['policy_version']==NEW
        assert action.last_request['innovation_sigma_multiplier_full12'][3]==pytest.approx(24/112)
        unchanged()
