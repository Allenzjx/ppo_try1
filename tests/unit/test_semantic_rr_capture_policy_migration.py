"""Synthetic CPU 389->410 tests; no real task, policy update or checkpoint credit."""
from copy import deepcopy
import ast
import inspect
import json
from pathlib import Path
import subprocess
import pytest
torch=pytest.importorskip('torch')
TensorDict=pytest.importorskip('tensordict').TensorDict
from wlr50_clean.ppo import semantic_rr_capture_migration as m
from wlr50_clean.ppo import semantic_training as t
from wlr50_clean.ppo import semantic_migration as shared
from wlr50_clean.ppo.semantic_rr_capture_profile import *
from wlr50_clean.ppo.semantic_p05_capture_profile import P05_CAPTURE_POLICY,P05_CAPTURE_OBSERVATION_LAYOUT
from wlr50_clean.ppo.semantic_observation import load_semantic_observation_schema,SemanticObservationBuilder,HISTORY_GROUPS,SemanticObservationError
from wlr50_clean.ppo.semantic_policy_distribution import policy_contract,supported_heteroscedastic_contract_version

ROOT=Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def cpu_state():
    assert not torch.cuda.is_available()
    state=torch.get_rng_state();threads=torch.get_num_threads();torch.set_num_threads(1);torch.manual_seed(318)
    yield
    torch.set_rng_state(state);torch.set_num_threads(threads)


def make(dimension):
    return t.construct_semantic_runner(m._shape_env(dimension,'cpu'),seed=1001,device='cpu',initialize_actor=False,
        policy_version=P05_CAPTURE_POLICY if dimension==389 else RR_CAPTURE_POLICY,
        observation_layout=P05_CAPTURE_OBSERVATION_LAYOUT if dimension==389 else RR_CAPTURE_OBSERVATION_LAYOUT)[0]


def source():
    runner=make(389);runner.alg.learning_rate=2.25e-5
    for group in runner.alg.optimizer.param_groups:
        group['lr']=2.25e-5
        for p in group['params']:
            runner.alg.optimizer.state[p]={'step':torch.tensor(9.),'exp_avg':torch.full_like(p,.001),'exp_avg_sq':torch.full_like(p,.002)}
    return runner,{'actor_state_dict':runner.alg.actor.state_dict(),'critic_state_dict':runner.alg.critic.state_dict(),
        'optimizer_state_dict':runner.alg.optimizer.state_dict(),'iter':9,'infos':{'synthetic':True}}


def test_exact_zero_append_not_reset_and_original_bundle_immutable():
    runner,bundle=source();before=t.state_hash(bundle);new=m.zero_append_rr_training_state(bundle)
    assert t.state_hash(bundle)==before
    for role in ('actor','critic'):
        for key,old in bundle[role+'_state_dict'].items():
            value=new[role+'_state_dict'][key]
            if key=='mlp.0.weight':
                assert value.shape==(256,410) and torch.equal(value[:,:389],old) and torch.count_nonzero(value[:,389:])==0
            else: assert torch.equal(value,old)
    assert new['optimizer_state_dict']['param_groups']==bundle['optimizer_state_dict']['param_groups']
    for ident,row in bundle['optimizer_state_dict']['state'].items():
        for key,old in row.items():
            value=new['optimizer_state_dict']['state'][ident][key]
            if ident in (0,6) and key!='step':assert torch.equal(value[:,:389],old) and torch.count_nonzero(value[:,389:])==0
            else:assert torch.equal(value,old)


@pytest.mark.parametrize('phase',range(13))
def test_old_kernel_gaussian_mean_sigma_raw_likelihood_preserved_for_any_new_state(phase):
    old,bundle=source();new=make(410);mapped=m.zero_append_rr_training_state(bundle)
    for role in ('actor','critic'):getattr(new.alg,role).load_state_dict(mapped[role+'_state_dict'])
    x=torch.zeros(1,389);x[:,phase]=1;x[:,20]=.02;x[:,158:158+phase]=1
    x[:,195:207]=torch.linspace(-.9,.9,12);x[:,157]=float(phase>=10)
    oldobs=TensorDict({'policy':x,'critic':x.clone()},batch_size=[1])
    y=torch.cat((x,torch.linspace(-.6,.8,21).reshape(1,21)),dim=-1)
    newobs=TensorDict({'policy':y,'critic':y.clone()},batch_size=[1])
    with torch.no_grad():
        expected=old.alg.actor(oldobs)
        actual,receipt=t.audited_history_policy_request(new.alg.actor,newobs,lambda:new.alg.actor(newobs),stochastic=False)
        torch.testing.assert_close(actual,expected,atol=2e-7,rtol=2e-6)
        torch.testing.assert_close(new.alg.critic(newobs),old.alg.critic(oldobs),atol=2e-7,rtol=2e-6)
        old.alg.actor(oldobs,stochastic_output=True)
        raw,receipt=t.audited_history_policy_request(new.alg.actor,newobs,lambda:new.alg.actor(newobs,stochastic_output=True),stochastic=True)
        torch.testing.assert_close(new.alg.actor.output_std,old.alg.actor.output_std,atol=2e-7,rtol=2e-6)
        lp=torch.distributions.Normal(new.alg.actor.output_mean,new.alg.actor.output_std).log_prob(raw).sum(-1)
        assert torch.equal(lp,new.alg.actor.get_output_log_prob(raw))
        assert receipt['policy_version']==RR_CAPTURE_POLICY and receipt['sampling_draws']==1
        assert len(receipt['rr_capture_assist_observed_features'])==14 and len(receipt['rr_capture_transfer_observed_features'])==7
        assert receipt['transformed_actuator_targets_are_not_policy_samples']


def test_explicit_layout_old389_prefix_and_strict_live_fields():
    from test_semantic_observation_reward_env import _frame,ZERO12
    from wlr50_clean.ppo.semantic_capture_assist import HipOnlyCaptureAssist
    from wlr50_clean.ppo.semantic_rr_capture_assist import RRHipOnlyCaptureAssist
    old=load_semantic_observation_schema(ROOT/'configs/ppo_p05_hip_only_continuation_v1/observation_schema.json')
    new=load_semantic_observation_schema(ROOT/'configs/ppo_rr_capture_then_rl_transfer_v1/observation_schema.json')
    assert new.dimension==410 and new.groups[:-2]==old.groups and new.observation_layout==RR_CAPTURE_OBSERVATION_LAYOUT
    frame=_frame(stage='P09');task=frame.info['semantic_task']
    task.update(capture_continuation={'mode':'p05_hip_only_continuation_v1','scheduler_advanced_pending':False},
        fl_capture_pending=False,allow_capture_continuation=False,p05_local_deadline_warning=False,capture_pending_elapsed_s=0.)
    frame.info['capture_assist']=HipOnlyCaptureAssist().snapshot()
    frame.info['rr_capture_assist']=RRHipOnlyCaptureAssist().snapshot()
    frame.info['rr_capture_transfer_context']=dict.fromkeys(RR_TASK_FIELDS,False)
    history=dict.fromkeys(HISTORY_GROUPS,ZERO12)
    old_values=old.encode(SemanticObservationBuilder(old).build(frame,history).groups)
    values=new.encode(SemanticObservationBuilder(new).build(frame,history).groups)
    assert values[:389]==old_values and values[389:]==(0.,)*21
    frame.info['rr_capture_transfer_context']['rr_top_contact']=True
    values=new.encode(SemanticObservationBuilder(new).build(frame,history).groups)
    assert values[RR_TASK_START+2]==1 and values[RR_TASK_START+3]==0
    del frame.info['rr_capture_transfer_context']['rr_current_bearing']
    with pytest.raises(SemanticObservationError):SemanticObservationBuilder(new).build(frame,history)
    frame.info['rr_capture_transfer_context']['rr_current_bearing']=0
    with pytest.raises(SemanticObservationError):SemanticObservationBuilder(new).build(frame,history)


def test_profile_requires_versioned_dimension_and_has_no_sigma_change():
    contract=policy_contract(RR_CAPTURE_POLICY,observation_layout=RR_CAPTURE_OBSERVATION_LAYOUT)
    assert supported_heteroscedastic_contract_version(contract)==RR_CAPTURE_POLICY
    assert contract['sigma_kernel_observation_slice']==[0,372] and contract['rho']==.9
    assert contract['preserved_observation_prefix_dimension']==389
    with pytest.raises(ValueError):policy_contract(RR_CAPTURE_POLICY,observation_layout=P05_CAPTURE_OBSERVATION_LAYOUT)


def test_new_namespace_and_saved410_frozen_prefix_are_explicit():
    from wlr50_clean.ppo import semantic_cli as cli
    from wlr50_clean.ppo.semantic_checkpoint_prefix_policy import build_frozen_checkpoint_prefix_policy
    assert cli.version_paths('v3',experiment_id=m.TARGET_EXPERIMENT)==tuple(ROOT/k/('ppo_'+m.TARGET_EXPERIMENT) for k in ('runs','outputs','configs'))
    assert t.training_quantity_budgets(m.TARGET_EXPERIMENT)==t.training_quantity_budgets(m.SOURCE_EXPERIMENT)
    runner=make(410);actor=runner.alg.actor;contract=policy_contract(RR_CAPTURE_POLICY,observation_layout=RR_CAPTURE_OBSERVATION_LAYOUT)
    record={'checkpoint_path':'synthetic_saved410.pt','checkpoint_sha256':'a'*64,'actor_parameter_sha256':t.parameter_hash(actor),
        'source_global_policy_decisions':300000,'source_ppo_updates':2000,'policy_contract':contract,
        'source_policy_contract':contract,'effective_policy_contract':contract,
        'source_runtime_content_sha256':'b'*64,'effective_runtime_content_sha256':'b'*64}
    obs=runner.env.get_observations()
    expected=actor(obs).detach()[0].tolist();before=torch.get_rng_state()
    prefix=build_frozen_checkpoint_prefix_policy(actor,record)
    assert prefix(obs['policy'][0].tolist())==tuple(expected) and torch.equal(before,torch.get_rng_state())
    bad=deepcopy(record);bad['source_policy_contract']=policy_contract(P05_CAPTURE_POLICY,observation_layout=P05_CAPTURE_OBSERVATION_LAYOUT)
    with pytest.raises(ValueError):build_frozen_checkpoint_prefix_policy(actor,bad)
    bad=deepcopy(record);bad['effective_runtime_content_sha256']='c'*64
    with pytest.raises(ValueError):build_frozen_checkpoint_prefix_policy(actor,bad)


def fixture(root):
    from test_semantic_p05_preedge_migration import boundary
    meta=boundary()[0]
    origin=dict(global_policy_decisions=216448,ppo_updates=1656,optimizer_steps=33120)
    meta['p05_preedge_approach_recovery_branch']={'schema':'synthetic_existing_preedge','counter_origin':origin}
    meta['p05_preedge_approach_recovery_migration']={'schema':'synthetic_existing_preedge','unchanged':True}
    meta['p05_preedge_approach_recovery_branch_counts']={k:meta[k]-origin[k] for k in m.COUNTERS}
    def write(path,raw):
        path=root/path;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    def git(*args):return subprocess.run(['git','-C',str(root),*args],capture_output=True,text=True,check=True).stdout.strip()
    def contract(namespace):
        from wlr50_clean.ppo.semantic_policy_distribution import CONFIG_NAMES
        files={p.relative_to(root).as_posix():shared.file_sha(p) for p in root.rglob('*') if p.is_file() and '.git' not in p.parts}
        return {'schema':'wlr50_clean.semantic_runtime_contract.v1','semantic_version':'v3','source_git_commit':git('rev-parse','HEAD'),
            'runtime_content_sha256':shared.digest(files),'files':files,'experiment_id':namespace,
            'selected_configuration':{name:{'path':f'configs/ppo_{namespace}/{name}','sha256':files[f'configs/ppo_{namespace}/{name}']} for name in CONFIG_NAMES}}
    root.mkdir();git('init');git('config','user.name','synthetic');git('config','user.email','fixture@example.invalid')
    for p in (ROOT/'configs/ppo_p05_hip_only_continuation_v1').iterdir():write(p.relative_to(ROOT),p.read_bytes())
    write(Path('src/wlr50_clean/ppo/semantic_supervisor.py'),b'# synthetic source supervisor\n')
    git('add','.');git('commit','-m','synthetic_source');old=contract(m.SOURCE_EXPERIMENT)
    for p in (ROOT/'configs/ppo_rr_capture_then_rl_transfer_v1').iterdir():
        raw=p.read_bytes()
        if p.name=='stage_task_spec.yaml':
            import yaml
            initial=yaml.safe_load(raw)
            initial['nominal']['p05_preedge_approach_recovery']='p05_preedge_approach_recovery_v1'
            initial.pop('p05_finite_recovery_timeout_semantics',None)
            raw=yaml.safe_dump(initial).encode()
        write(p.relative_to(ROOT),raw)
    write(Path('src/wlr50_clean/ppo/semantic_supervisor.py'),b'# synthetic initial RR supervisor\n')
    for name in ('profile','actor','migration','assist','context'):
        write(Path(f'src/wlr50_clean/ppo/semantic_rr_capture_{name}.py'),b'# synthetic reviewed source file\n')
    git('add','.');git('commit','-m','synthetic_target');new=contract(m.TARGET_EXPERIMENT)
    meta['runtime_contract']=old
    return meta,old,new


def test_actual_generic_route_build_validate_publish_reload_and_ordinary_save(tmp_path,monkeypatch):
    runner,_=source();meta,old,new=fixture(tmp_path/'repo')
    source_cp,_=t.save_semantic_checkpoint(runner,tmp_path/'synthetic_source.pt',meta)
    original=shared.checkpoint_metadata(source_cp)
    review={p:h for p,h in new['files'].items() if p.startswith('src/')}
    plan=m.build_rr_capture_migration(source_cp,new,reason='CPU synthetic append test',reviewed_code_sha256=review,project_root=tmp_path/'repo')
    path=tmp_path/'plan.json';path.write_text(json.dumps(plan),encoding='utf-8')
    monkeypatch.setattr(shared,'PROJECT_ROOT',tmp_path/'repo')
    published=m.publish_rr_capture_checkpoint(source_cp,new,path,tmp_path/'synthetic_target.pt')
    final=shared.checkpoint_metadata(Path(published['checkpoint']))
    assert published['save_load_round_trip'] and all(final[k]==original[k] for k in m.preserved_keys(original))
    assert final['rr_capture_transfer_branch_counts']==dict.fromkeys(m.COUNTERS,0)
    fresh=make(410);loaded=t.load_semantic_checkpoint(fresh,Path(published['checkpoint']),contract=new,seed=1001)
    assert fresh.alg.storage.step==0 and fresh.alg.transition.actions is None
    assert final['runner_config']['device']=='cpu' and final['optimizer_learning_rate']==2.25e-5
    code=ast.parse(inspect.getsource(t.train_semantic))
    carry=[ast.literal_eval(n.iter) for n in ast.walk(code) if isinstance(n,ast.For) and isinstance(n.iter,ast.Tuple)
           and any(isinstance(v,ast.Constant) and v.value=='new_mdp_warm_start' for v in n.iter.elts)][0]
    assert {'rr_capture_transfer_branch','rr_capture_transfer_migration'}<=set(carry)
    loaded.update({k:loaded[k]+d for k,d in zip(m.COUNTERS,(128,1,20))})
    later,_=t.save_semantic_checkpoint(fresh,tmp_path/'synthetic_metadata_carry.pt',loaded)
    after=shared.checkpoint_metadata(later)
    assert after['rr_capture_transfer_branch_counts']==dict(zip(m.COUNTERS,(128,1,20)))
    assert after['rr_postcross_workspace_branch']==final['rr_postcross_workspace_branch']
    bad=deepcopy(plan);bad[m.FACTOR_KEY]['sigma_kernel_changed']=True
    path.write_text(json.dumps(bad),encoding='utf-8')
    with pytest.raises(ValueError):m.validate_rr_capture_migration(source_cp,new,path)
    # Even a freshly committed/rehashed target cannot hide new caps/reward,
    # changed old389 scaling, wheel enablement, or additional task rules.
    import yaml
    original_files={name:(tmp_path/'repo'/binding['path']).read_bytes() for name,binding in new['selected_configuration'].items()}
    for name,kind in [('action_schema.json','bytes'),('reward_config.yaml','bytes'),('quality_score.yaml','bytes'),
                      ('execution_profile.yaml','wheel'),('stage_task_spec.yaml','rule'),('observation_schema.json','scale')]:
        for restore,raw in original_files.items():(tmp_path/'repo'/new['selected_configuration'][restore]['path']).write_bytes(raw)
        target_path=tmp_path/'repo'/new['selected_configuration'][name]['path']
        if kind=='bytes':target_path.write_bytes(original_files[name]+b'\n')
        else:
            data=json.loads(original_files[name]) if name.endswith('.json') else yaml.safe_load(original_files[name])
            if kind=='wheel':data['rr_capture_wheel_mode']='enabled'
            elif kind=='rule':data['episode_maximum_duration_s']=201.
            else:data['feature_groups'][0]['scale']=2.
            target_path.write_text(json.dumps(data) if name.endswith('.json') else yaml.safe_dump(data),encoding='utf-8')
        def git(*args):return subprocess.run(['git','-C',str(tmp_path/'repo'),*args],capture_output=True,text=True,check=True).stdout.strip()
        git('add','.');git('commit','-m','synthetic_rejected_'+kind+'_'+name)
        modified=deepcopy(new);modified['source_git_commit']=git('rev-parse','HEAD')
        for key in modified['files']:modified['files'][key]=shared.file_sha(tmp_path/'repo'/key)
        modified['runtime_content_sha256']=shared.digest(modified['files'])
        for binding in modified['selected_configuration'].values():binding['sha256']=modified['files'][binding['path']]
        with pytest.raises(ValueError):m.build_rr_capture_migration(source_cp,modified,reason='reject unrelated changes',reviewed_code_sha256=review,project_root=tmp_path/'repo')


def recross_boundary(tmp_path,monkeypatch):
    """Two real synthetic publications, never a fixture pretending to be training."""
    import yaml
    root=tmp_path/'repo';runner,_=source();meta,old,first=fixture(root)
    monkeypatch.setattr(shared,'PROJECT_ROOT',root)
    source_cp,_=t.save_semantic_checkpoint(runner,tmp_path/'source389.pt',meta)
    review={p:h for p,h in first['files'].items() if old['files'].get(p)!=h and p.startswith('src/')}
    plan=m.build_rr_capture_migration(source_cp,first,reason='synthetic first control',reviewed_code_sha256=review)
    plan_path=tmp_path/'initial_plan.json';plan_path.write_text(json.dumps(plan),encoding='utf-8')
    initial_cp=tmp_path/'initial410.pt'
    m.publish_rr_capture_checkpoint(source_cp,first,plan_path,initial_cp)
    target=deepcopy(first)
    task_path=root/first['selected_configuration']['stage_task_spec.yaml']['path']
    task=json.loads(json.dumps(yaml.safe_load(task_path.read_text())))
    task['nominal']['p05_preedge_approach_recovery']=m.RECROSS_MODE
    task['p05_finite_recovery_timeout_semantics']=m.RECROSS_TIMEOUT
    task_path.write_text(yaml.safe_dump(task),encoding='utf-8')
    for p in m.RECROSS_RUNTIME_DELTA:
        if p.startswith('src/'):(root/p).write_bytes((root/p).read_bytes()+b'# synthetic recross revision\n')
    def commit():
        for args in (('add','.'),('commit','-m','synthetic_recross_candidate')):
            subprocess.run(['git','-C',str(root),*args],capture_output=True,text=True,check=True)
        target['source_git_commit']=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
        for p in target['files']:target['files'][p]=shared.file_sha(root/p)
        target['runtime_content_sha256']=shared.digest(target['files'])
        for binding in target['selected_configuration'].values():binding['sha256']=target['files'][binding['path']]
    commit()
    def build():
        return m.build_rr_capture_migration(source_cp,target,reason='synthetic zero-update recross replacement',
            reviewed_code_sha256={p:h for p,h in target['files'].items() if old['files'].get(p)!=h and p.startswith('src/')},
            initial_checkpoint=initial_cp,initial_plan_path=plan_path)
    return root,source_cp,initial_cp,plan_path,first,target,build,commit


def test_recross_strict_variant_official_publish_reload_carries_all_origins(tmp_path,monkeypatch):
    root,source_cp,initial_cp,initial_plan,first,target,build,commit=recross_boundary(tmp_path,monkeypatch)
    plan=build();factor=plan[m.FACTOR_KEY]
    assert factor['target_control_revision']==m.RECROSS_MODE and factor['termination_semantics_changed']
    assert factor['changed_files_since_initial_rr']==sorted(m.RECROSS_RUNTIME_DELTA)
    assert not factor['superseded_zero_update_boundary']['replacement_loses_learning']
    assert factor['superseded_zero_update_boundary']['checkpoint_sha256']==shared.file_sha(initial_cp)
    path=tmp_path/'recross_plan.json';path.write_text(json.dumps(plan),encoding='utf-8')
    record=shared.validate_migration_plan(source_cp,target,path,project_root=root)
    assert record[m.FACTOR_KEY]==factor
    result=m.publish_rr_capture_checkpoint(source_cp,target,path,tmp_path/'recross410.pt')
    initial=shared.checkpoint_metadata(initial_cp);new=shared.checkpoint_metadata(Path(result['checkpoint']))
    assert result['save_load_round_trip'] and new['rr_capture_transfer_branch_counts']==dict.fromkeys(m.COUNTERS,0)
    for key in ('actor_parameter_sha256','critic_parameter_sha256','optimizer_state_sha256','runner_config','training_rng_state'):
        assert new[key]==initial[key]
    source_meta=shared.checkpoint_metadata(source_cp)
    assert all(new[key]==source_meta[key] for key in m.preserved_keys(source_meta))
    fresh=make(410);loaded=t.load_semantic_checkpoint(fresh,Path(result['checkpoint']),contract=target,seed=1001)
    assert fresh.alg.storage.step==0 and fresh.alg.transition.actions is None
    loaded.update({k:loaded[k]+d for k,d in zip(m.COUNTERS,(128,1,20))})
    later,_=t.save_semantic_checkpoint(fresh,tmp_path/'metadata_only_carry_not_actual_PPO.pt',loaded)
    final=shared.checkpoint_metadata(later)
    assert final['rr_capture_transfer_migration']==new['rr_capture_transfer_migration']
    assert final['rr_capture_transfer_branch_counts']==dict(zip(m.COUNTERS,(128,1,20)))


def test_recross_rejects_missing_binding_extra_files_configs_and_partial_modes(tmp_path,monkeypatch):
    import yaml
    root,source_cp,initial_cp,initial_plan,first,target,build,commit=recross_boundary(tmp_path,monkeypatch)
    original={p:(root/p).read_bytes() for p in target['files']}
    review={p:h for p,h in target['files'].items() if p.startswith('src/')}
    with pytest.raises(ValueError,match='requires its first'):
        m.build_rr_capture_migration(source_cp,target,reason='missing original',reviewed_code_sha256=review)
    task_path=target['selected_configuration']['stage_task_spec.yaml']['path']
    mutations=[('actor','src/wlr50_clean/ppo/semantic_rr_capture_actor.py'),
        ('reward',target['selected_configuration']['reward_config.yaml']['path']),
        ('timeout',task_path),('mode',task_path),('extra_rule',task_path)]
    for kind,path in mutations:
        for p,raw in original.items():(root/p).write_bytes(raw)
        if kind in ('actor','reward'):(root/path).write_bytes(original[path]+b'\n# unrelated\n')
        else:
            task=yaml.safe_load(original[path])
            if kind=='timeout':task.pop('p05_finite_recovery_timeout_semantics')
            elif kind=='mode':task['nominal']['p05_preedge_approach_recovery']='p05_preedge_approach_recovery_v1'
            else:task['episode_maximum_duration_s']=201.
            (root/path).write_text(yaml.safe_dump(task),encoding='utf-8')
        commit()
        with pytest.raises(ValueError):build()


@pytest.mark.parametrize('kind',['actor_old','actor_append','critic','adam_moment','adam_step','iteration','rng','aux','counter'])
def test_recross_cannot_discard_any_initial410_learning_even_rehashed(tmp_path,monkeypatch,kind):
    root,source_cp,initial_cp,initial_plan,first,target,build,commit=recross_boundary(tmp_path,monkeypatch)
    payload=torch.load(initial_cp,map_location='cpu',weights_only=False)
    if kind=='actor_old':payload['actor_state_dict']['mlp.0.weight'][0,0]+=.01
    elif kind=='actor_append':payload['actor_state_dict']['mlp.0.weight'][0,389]=.01
    elif kind=='critic':payload['critic_state_dict']['mlp.4.bias'][0]+=.01
    elif kind=='adam_moment':payload['optimizer_state_dict']['state'][0]['exp_avg'][0,389]=.01
    elif kind=='adam_step':payload['optimizer_state_dict']['state'][0]['step']+=1
    elif kind=='iteration':payload['iter']+=1
    elif kind=='rng':payload['infos']['training_rng_state']['synthetic_tampering']=True
    elif kind=='aux':payload['infos']['rr_postcross_workspace_branch']['synthetic_aux_tampering']=True
    else:payload['infos']['global_policy_decisions']+=128
    torch.save(payload,initial_cp)
    sidecar={**payload['infos'],'checkpoint_path':str(initial_cp.resolve()),'checkpoint_sha256':shared.file_sha(initial_cp),'save_load_round_trip':True}
    initial_cp.with_name(initial_cp.stem+'_manifest.json').write_text(json.dumps(sidecar),encoding='utf-8')
    with pytest.raises(ValueError):build()
