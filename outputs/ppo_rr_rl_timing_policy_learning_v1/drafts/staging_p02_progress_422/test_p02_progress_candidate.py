"""Synthetic candidate tests loaded in memory; production files never changed."""
import copy
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import types
import pytest

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests/unit'),str(HERE)]
import build_integration_patch as draft

# Populate the old package graph, then install this review candidate only in this process.
from wlr50_clean.ppo import semantic_training as t
from wlr50_clean.ppo import semantic_policy_distribution as distribution
from wlr50_clean.ppo import semantic_migration as generic
from wlr50_clean.ppo import semantic_rear_policy_timing_migration as parent
if not (ROOT/draft.CODE/'semantic_p02_progress_profile.py').exists():
    for name in ('semantic_p02_progress_profile','semantic_p02_progress_actor'):
        full='wlr50_clean.ppo.'+name
        module=types.ModuleType(full);module.__file__=str(ROOT/draft.CODE/(name+'.py'));module.__package__='wlr50_clean.ppo'
        sys.modules[full]=module
        exec(compile((HERE/(name+'.py')).read_text(),module.__file__,'exec'),module.__dict__)
    for name in ('semantic_policy_distribution','semantic_migration','semantic_training',
                 'semantic_observation','semantic_rear_policy_timing_migration','semantic_cli',
                 'semantic_checkpoint_prefix_policy','semantic_checkpoint_prefix'):
        module=importlib.import_module('wlr50_clean.ppo.'+name)
        text=draft.change(name+'.py',(ROOT/draft.CODE/(name+'.py')).read_text())
        exec(compile(text,module.__file__,'exec'),module.__dict__)
    name='semantic_p02_progress_migration';full='wlr50_clean.ppo.'+name
    module=types.ModuleType(full);module.__file__=str(ROOT/draft.CODE/(name+'.py'));module.__package__='wlr50_clean.ppo'
    sys.modules[full]=module
    exec(compile((HERE/(name+'.py')).read_text(),module.__file__,'exec'),module.__dict__)
m=importlib.import_module('wlr50_clean.ppo.semantic_p02_progress_migration')
import torch
from tensordict import TensorDict
from wlr50_clean.ppo.semantic_p02_progress_profile import *
from wlr50_clean.ppo.semantic_p02_progress_actor import SemanticP02ProgressHistoryMLPModel,p02_progress_effective_log_std
from wlr50_clean.ppo.semantic_rear_policy_timing_actor import rear_policy_timing_effective_log_std
from wlr50_clean.ppo.semantic_rear_policy_timing_profile import REAR_POLICY_TIMING_POLICY,REAR_POLICY_TIMING_OBSERVATION_LAYOUT
from wlr50_clean.ppo.semantic_rr_capture_migration import _shape_env


@pytest.fixture(autouse=True)
def cpu():
    assert not torch.cuda.is_available()
    torch.set_num_threads(1)


def make(width):
    return t.construct_semantic_runner(_shape_env(width,'cpu'),seed=1001,device='cpu',initialize_actor=False,
        policy_version=REAR_POLICY_TIMING_POLICY if width==419 else P02_PROGRESS_POLICY,
        observation_layout=REAR_POLICY_TIMING_OBSERVATION_LAYOUT if width==419 else P02_PROGRESS_OBSERVATION_LAYOUT)[0]


def obs(x):return TensorDict({'policy':x,'critic':x.clone()},batch_size=[x.shape[0]])


@pytest.mark.parametrize('phase',[0,1,8,11,12])
def test_zero_append_actor_critic_and_shared_sigma(phase):
    source,target=make(419),make(422)
    for role in ('actor','critic'):
        state=copy.deepcopy(getattr(source.alg,role).state_dict())
        state['mlp.0.weight']=torch.cat((state['mlp.0.weight'],torch.zeros(256,3)),dim=1)
        getattr(target.alg,role).load_state_dict(state)
    x=torch.zeros(2,422);x[:,phase]=1;x[:,20]=.02;x[:,158:158+phase]=1
    x[:,195:207]=torch.linspace(-.3,.3,12)
    if phase==1:x[:,419:]=torch.tensor([.03,.8,1.])
    if phase==8:x[:,410]=x[:,404]=1
    if phase==11:x[:,412]=1
    with torch.no_grad():
        for role in ('actor','critic'):
            torch.testing.assert_close(getattr(source.alg,role)(obs(x[:,:419])),
                getattr(target.alg,role)(obs(x)),atol=2e-7,rtol=2e-6)
        head=target.alg.actor.mlp(x)[...,1,:]
        assert torch.equal(p02_progress_effective_log_std(head,x)[0],rear_policy_timing_effective_log_std(head,x[:,:419])[0])
        target.alg.actor(obs(x),stochastic_output=True)
        raw=target.alg.actor.distribution.sample()
        normal=torch.distributions.Normal(target.alg.actor.output_mean,target.alg.actor.output_std)
        assert torch.equal(target.alg.actor.get_output_log_prob(raw),normal.log_prob(raw).sum(-1))


@pytest.mark.parametrize('stochastic',[False,True])
def test_actual_request_audit_and_prefix(stochastic):
    from wlr50_clean.ppo.semantic_checkpoint_prefix_policy import FrozenCheckpointPrefixPolicy
    runner=make(422);actor=runner.alg.actor;x=torch.zeros(1,422);x[:,1]=1;x[:,158]=1;x[:,20]=.1;x[:,419:]=torch.tensor([.03,.4,1.])
    with torch.no_grad():
        raw,receipt=t.audited_history_policy_request(actor,obs(x),lambda:actor(obs(x),stochastic_output=stochastic),stochastic=stochastic)
    assert receipt['policy_version']==P02_PROGRESS_POLICY
    assert receipt['p02_progress_observed_features']==x[0,419:].tolist()
    assert receipt['sampling_draws']==int(stochastic) and receipt['extra_model_forwards']==0
    policy=t._runner_policy_contract(runner)
    record=dict(checkpoint_path='synthetic_only.pt',checkpoint_sha256='a'*64,actor_parameter_sha256=t.parameter_hash(actor),
        source_global_policy_decisions=0,source_ppo_updates=0,policy_contract=policy,source_policy_contract=policy,
        effective_policy_contract=policy,source_runtime_content_sha256='b'*64,effective_runtime_content_sha256='b'*64)
    prefix=FrozenCheckpointPrefixPolicy(actor,record)
    assert len(prefix(tuple(x[0].tolist())))==12
    assert distribution.supported_heteroscedastic_contract_version(policy)==P02_PROGRESS_POLICY


@pytest.mark.parametrize('row,stage',[
    ({},'P02'),({'best_remaining_m':-.01,'credit_fraction':0.,'current_eligible':False},'P02'),
    ({'best_remaining_m':.01,'credit_fraction':1.1,'current_eligible':False},'P02'),
    ({'best_remaining_m':.01,'credit_fraction':.1,'current_eligible':1},'P02'),
    ({'best_remaining_m':.01,'credit_fraction':.1,'current_eligible':True},'P03')])
def test_bad_public_state_fails(row,stage):
    with pytest.raises(ValueError):p02_progress_features(row,stage)


def test_schema_preserves419_prefix(tmp_path):
    from wlr50_clean.ppo.semantic_observation import load_semantic_observation_schema
    data=json.loads((ROOT/'configs/ppo_rr_rl_timing_policy_learning_v1/observation_schema.json').read_text())
    original=copy.deepcopy(data);data['p02_progress_features_version']=P02_PROGRESS_OBSERVATION_LAYOUT
    data['feature_groups'].append(dict(name=P02_PROGRESS_GROUP,size=3,scale=1))
    path=tmp_path/'schema.json';path.write_text(json.dumps(data))
    schema=load_semantic_observation_schema(path)
    assert schema.dimension==422 and schema.observation_layout==P02_PROGRESS_OBSERVATION_LAYOUT
    assert list(schema.groups[:-1])==original['feature_groups']
    assert p02_progress_features(dict(best_remaining_m=0.,credit_fraction=0.,current_eligible=False),'P03')==(0.,0.,0.)


def test_populated_adam_official_append_save_reload_and_normal_carry(tmp_path,monkeypatch):
    from wlr50_clean.ppo.semantic_migration import digest
    source=make(419)
    for p in list(source.alg.actor.parameters())+list(source.alg.critic.parameters()):p.grad=torch.ones_like(p)
    source.alg.optimizer.step();source.alg.optimizer.zero_grad()
    for s in source.alg.optimizer.state.values():s['step'].fill_(33940.)
    source.alg.learning_rate=1e-5
    for g in source.alg.optimizer.param_groups:g['lr']=1e-5
    source.current_learning_iteration=1697
    real=ROOT/'outputs/ppo_rr_rl_timing_policy_learning_v1/branches/ancestor220544_recapture_v2/checkpoints/history/checkpoint_step_000221696_manifest.json'
    base=json.loads(real.read_text())
    for key in ('checkpoint_path','checkpoint_sha256','save_load_round_trip'):base.pop(key,None)
    source_path,side=t.save_semantic_checkpoint(source,tmp_path/'synthetic_source.pt',base)
    old=json.loads(side.read_text());current=copy.deepcopy(old['runtime_contract'])
    current.update(source_git_commit='f'*40,runtime_content_sha256='e'*64)
    record=dict(schema=m.SCHEMA,plan_path=str(tmp_path/'synthetic_plan.json'),
        source_checkpoint_sha256=m.SOURCE_SHA,source_manifest_sha256=m.SOURCE_MANIFEST_SHA,
        source_git_commit=m.SOURCE_HEAD,target_git_commit=current['source_git_commit'],
        source_contract_sha256=digest(old['runtime_contract']),target_contract_sha256=digest(current),
        source_runtime_content_sha256=old['runtime_contract']['runtime_content_sha256'],target_runtime_content_sha256=current['runtime_content_sha256'],
        changed_file_hashes={},source_changed_configuration={},
        **{m.FACTOR_KEY:dict(schema=m.SCHEMA,source_policy_contract=old['policy_contract'],
            target_policy_contract=distribution.policy_contract(P02_PROGRESS_POLICY,observation_layout=P02_PROGRESS_OBSERVATION_LAYOUT),
            revision_counter_origin=m.REVISION_ORIGIN,original_branch_origin=m.SOURCE_COUNTS,
            preserved_metadata_sha256={k:digest(old[k]) for k in (*m.preserved_keys(old),'checkpoint_output_routing')},
            added_policy_decisions=0,added_ppo_updates=0,added_optimizer_steps=0,added_auxiliary_updates=0)})
    monkeypatch.setattr(m,'validate_p02_progress_migration',lambda *a,**kw:record)
    mapped=m.zero_append_p02_training_state(torch.load(source_path,weights_only=False))
    runner=make(422)
    infos=t.load_semantic_checkpoint(runner,source_path,contract=current,seed=1001,migration=record)
    for role in ('actor','critic'):
        assert t.state_hash(getattr(runner.alg,role).state_dict())==t.state_hash(mapped[role+'_state_dict'])
    assert t.state_hash(runner.alg.optimizer.state_dict())==t.state_hash(mapped['optimizer_state_dict'])
    for ident in (0,6):
        for key in ('exp_avg','exp_avg_sq'):assert torch.count_nonzero(mapped['optimizer_state_dict']['state'][ident][key][:,419:])==0
    assert runner.alg.learning_rate==1e-5 and t.capture_training_rng_state(seed=1001)==old['training_rng_state']
    saved,manifest=t.save_semantic_checkpoint(runner,tmp_path/'synthetic_migrated.pt',infos)
    fresh=make(422);loaded=t.load_semantic_checkpoint(fresh,saved,contract=current,seed=1001)
    for key in m.preserved_keys(old):assert loaded[key]==old[key]
    assert t.state_hash(fresh.alg.optimizer.state_dict())==t.state_hash(mapped['optimizer_state_dict'])
    route=loaded['checkpoint_output_routing']
    m.validate_p02_progress_lineage(loaded,current,Path(route['output_root']),checkpoint_output_routing=route)
    with pytest.raises(ValueError):m.validate_p02_progress_lineage(loaded,current,tmp_path,checkpoint_output_routing=route)
    # Ordinary serializer carry, not a real update or simulation.
    again,again_manifest=t.save_semantic_checkpoint(fresh,tmp_path/'synthetic_normal_save.pt',loaded)
    carried=json.loads(again_manifest.read_text())
    assert carried[m.MIGRATION]==loaded[m.MIGRATION] and carried['rear_policy_timing_branch']==old['rear_policy_timing_branch']
    assert {k:carried[k] for k in m.COUNTERS}==m.REVISION_ORIGIN


def test_bad_mapping_and_wrong_registered_source(tmp_path):
    with pytest.raises((KeyError,ValueError)):m.zero_append_p02_training_state({})
    with pytest.raises((ValueError,FileNotFoundError)):m.build_p02_progress_migration(tmp_path/'absent.pt',{},reason='test')


def test_one_synthetic422_update_keeps_raw_likelihood_and_five_exposures(tmp_path):
    from collections import Counter
    from test_semantic_rear_policy_training_audit import Synthetic419Core
    class Synthetic422Core(Synthetic419Core):
        def observation(self,raw=(0.,)*12):return super().observation(raw)+(0.,0.,0.)
    env=t.SemanticRslAdapter(Synthetic422Core(),seed=1001,device='cpu');env.cfg['semantic_version']='v3'
    runner,_=t.construct_semantic_runner(env,seed=1001,device='cpu',initialize_actor=False,
        policy_version=P02_PROGRESS_POLICY,observation_layout=P02_PROGRESS_OBSERVATION_LAYOUT)
    contract=dict(experiment_id=m.EXPERIMENT,semantic_version='v3',training_budgets=t.training_quantity_budgets(m.EXPERIMENT),
        evidence='synthetic CPU ABI only; no real robot or learning credit')
    result=t.train_semantic(runner,env,run_dir=tmp_path/'run',output_root=tmp_path/'out',
        stage='full_episode',decisions=128,contract=contract,seed=1001,checkpoint_interval_updates=1)
    assert result['actual_policy_decisions']==128 and result['ppo_updates_this_run']==1 and result['optimizer_steps_this_run']==20
    rollout=torch.load(tmp_path/'run/rollouts/rollout_000001.pt',map_location='cpu',weights_only=False)
    assert rollout['observations']['policy'].shape==(128,1,422)
    rows=[json.loads(line) for line in (tmp_path/'run/residual_and_projection_audit.jsonl').read_text().splitlines()]
    for i,row in enumerate(rows):
        request=row['policy_request']
        assert request['policy_version']==P02_PROGRESS_POLICY
        assert request['p02_progress_observed_features']==rollout['observations']['policy'][i,0,419:].tolist()
        assert request['selected_raw_full12']==rollout['actions'][i,0].tolist()==row['raw_policy_action_full12']
        assert request['selected_raw_log_probability']==rollout['actions_log_prob'][i,0].item()==row['old_log_probability']
        assert request['effective_sigma_full12']==rollout['distribution_params'][1][i,0].tolist()
    files=list((tmp_path/'run').rglob('*likelihood*.json'))
    audits=[json.loads(p.read_text()) for p in files]
    audit=next(a for a in audits if a.get('schema')=='wlr50_clean.task_recovery_minibatch_likelihood.v1')
    assert len(audit['minibatches'])==20
    for batch in audit['minibatches']:
        assert all(len(indices)==1 for indices in batch['rollout_flat_indices'])
        for key in ('current_network_mean_full12','current_network_log_sigma_full12',
                    'loss_gradient_wrt_network_mean_full12','loss_gradient_wrt_network_log_sigma_full12'):
            assert torch.tensor(batch[key]).shape==(32,12)
    exposures=Counter(indices[0] for batch in audit['minibatches'] for indices in batch['rollout_flat_indices'])
    assert exposures==Counter({i:5 for i in range(128)})
