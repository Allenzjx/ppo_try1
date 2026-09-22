"""Synthetic official CPU updates only; never physical task or real aux credit."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import finite_auxiliary_mean as a
import torch
from wlr50_clean.ppo.semantic_policy_distribution import TASK_CONDITIONED_HIP_WHEEL_POLICY


class CPUCore:
    def __init__(self):
        self.calls=0;self.frame=SimpleNamespace(sim_time_s=0.)
    def observation(self,raw=None):
        obs=[0.]*372;obs[0]=1.;obs[20]=.01+self.tick/3000;obs[158:163]=[1.]*5
        if raw is not None:obs[195:207]=[max(-19.,min(19.,v)) for v in raw]
        return tuple(obs)
    def reset(self,*,seed=1001,options=None):
        self.tick=0;self.frame.sim_time_s=0.;return self.observation()
    def step(self,raw):
        self.tick+=1;self.calls+=1;self.frame.sim_time_s=self.tick/15
        return SimpleNamespace(observation=self.observation(raw),reward=1+raw[0]-.1*raw[1]**2,
            terminated=self.tick==17,truncated=False,info={'phase_id':'P01','raw_policy_action_full12':raw,
            'applied_action_full12':tuple(.1*v for v in raw),'task_success':False,
            'termination_reason':'CPU_TEST_DOUBLE_NOT_PHYSICS' if self.tick==17 else None,
            'actuator_target_effect_audit':{'schema':'wlr50_clean.actuator_target_effect_audit.v1',
                'verified':True,'actual_mapping_matches_dispatch':True,'setter_dispatch_targets_equal':True,
                'same_tick_counterfactual':True,'raw_policy_action_full12':raw,'target_dtype':'torch.float32',
                'changed_target_channel_count':12}})
    def telemetry_summary(self):return {'synthetic_CPU_only':True,'calls':self.calls}


def runner():
    env=a.training.SemanticRslAdapter(CPUCore(),seed=1001,device='cpu');env.cfg['semantic_version']='v3'
    run,_=a.training.construct_semantic_runner(env,seed=1001,device='cpu',initialize_actor=False,
        policy_version=TASK_CONDITIONED_HIP_WHEEL_POLICY,observation_layout='diagonal_transfer_state_v1')
    return run,env


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    before=torch.get_rng_state();threads=torch.get_num_threads();torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    yield
    torch.set_rng_state(before);torch.set_num_threads(threads)


def states(run):
    # These are explicitly synthetic API fixtures, never training evidence.
    x=torch.zeros(4,372);x[:,4]=1.;x[:,150]=1.;x[:,21]=.4;x[:,195]=torch.tensor([-.4,-.3,.2,.4])
    x[:,158:163]=1.
    h=x.clone();h[:,:13]=0.;h[:,1]=1.
    tx=a.tensors(x,device='cpu');hx=a.tensors(h,device='cpu')
    target=a.distribution(run.alg.actor,tx)['mean'][:,0]-.02
    return tx,target,hx


def synthetic_source(tmp_path):
    a.training.seed_training_rngs(1001);run,env=runner()
    r=a.training.train_semantic(run,env,run_dir=tmp_path/'first_run',output_root=tmp_path/'first_out',
        stage='full_episode',decisions=128,contract={'CPU_synthetic_no_physical_credit':True},seed=1001)
    infos=json.loads(Path(r['checkpoints'][-1]['manifest']).read_text())
    for key in ('checkpoint_path','checkpoint_sha256','save_load_round_trip'):infos.pop(key)
    origin={'global_policy_decisions':0,'ppo_updates':0,'optimizer_steps':0}
    infos['task_conditioned_hip_wheel_branch']={'branch_id':a.EXPERIMENT,'counter_origin':origin,'synthetic_test':True}
    infos['task_conditioned_hip_wheel_branch_counts']={key:infos[key] for key in origin}
    infos['new_mdp_origin_global_policy_decisions']=0
    cp=tmp_path/'synthetic_source.pt';a.training.save_semantic_checkpoint(run,cp,infos)
    return run,env,infos,cp


@pytest.mark.parametrize('change',[{'max_steps':17},{'max_steps':True},{'learning_rate':.0501},
    {'maximum_train_request_shift_deg':1.01},{'maximum_holdout_request_shift_deg':.251},
    {'maximum_per_state_conditional_kl':.101}])
def test_budget_cannot_expand(change):
    with pytest.raises(ValueError):a.Budget(**change).validate()


def test_only_explicit_authorization_and_fresh_boundary():
    run,_=runner();tx,target,hx=states(run)
    with pytest.raises(ValueError,match='authorization'):a.fit_mean_row(run,tx,target,hx)
    for pending in (False,True):
        run.alg.storage.step=0 if pending else 1
        run.alg.transition.actions=torch.zeros(1,12) if pending else None
        with pytest.raises(ValueError,match='boundary'):a.fit_mean_row(run,tx,target,hx,authorized=True)


def test_real_history_derivative_is_point_one_not_amplified():
    run,_=runner();tx,target,hx=states(run);layer=a.mean_layer(run.alg.actor)
    w=torch.nn.Parameter(layer.weight[:1].detach().clone());b=torch.nn.Parameter(layer.bias[:1].detach().clone())
    for obs in (tx,hx):
        mean=a.functional_mean(run.alg.actor,obs,w,b)
        gradient=torch.autograd.grad(mean[:,0].sum(),b)[0]
        assert torch.allclose(gradient,torch.tensor([.4]),rtol=0,atol=1e-7)
        d=a.distribution(run.alg.actor,obs)
        assert torch.equal(d['history'][:,0],obs['policy'][:,195])
        assert torch.equal(mean,d['mean'])


def test_readonly_linear_prediction_uses_true_autograd_first_gradient():
    run,_=runner();tx,target,hx=states(run);layer=a.mean_layer(run.alg.actor)
    w=torch.nn.Parameter(layer.weight[:1].detach().clone());b=torch.nn.Parameter(layer.bias[:1].detach().clone())
    loss=.5*(a.functional_mean(run.alg.actor,tx,w,b)[:,0]-target).square().mean()
    grad=torch.cat([g.flatten() for g in torch.autograd.grad(loss,(w,b))])
    before=a.training.parameter_hash(run.alg.actor)
    result=a.linear_budget_prediction(run.alg.actor,tx,target,hx)
    assert result['first_gradient_norm']==pytest.approx(float(grad.norm()),rel=1e-6)
    assert result['first_gradient_bias']==pytest.approx(float(grad[-1]),rel=1e-6)
    assert result['sum_decaying_learning_rates']==pytest.approx(.425)
    assert a.training.parameter_hash(run.alg.actor)==before


def test_trust_rejection_rolls_back_parameter_candidate():
    run,_=runner();tx,target,hx=states(run);before=a.training.parameter_hash(run.alg.actor)
    report=a.fit_mean_row(run,tx,target-1.,hx,authorized=True,
        budget=a.Budget(maximum_holdout_request_shift_deg=1e-12))
    assert report['accepted_auxiliary_updates']==0 and report['attempted_auxiliary_optimizer_steps']==1
    assert a.training.parameter_hash(run.alg.actor)==before


def test_only_257_scalars_Adam_and_full_state_unchanged_then_official_chain(tmp_path,monkeypatch):
    run,env,infos,source=synthetic_source(tmp_path);tx,target,hx=states(run)
    original=copy.deepcopy(run.alg.actor.state_dict());adam=copy.deepcopy(run.alg.optimizer.state_dict())
    assert adam['state'] and any(bool(s['exp_avg'].abs().sum()>0) for s in adam['state'].values())
    original_step=run.alg.optimizer.step
    def forbid(*args,**kwargs):raise AssertionError('original PPO Adam was stepped')
    monkeypatch.setattr(run.alg.optimizer,'step',forbid)
    report=a.fit_mean_row(run,tx,target,hx,authorized=True)
    assert 1<=report['accepted_auxiliary_updates']<=16
    assert report['after']['FL_request_absolute_error_mean_deg']<report['before']['FL_request_absolute_error_mean_deg']
    assert a.training.state_hash(run.alg.optimizer.state_dict())==a.training.state_hash(adam)
    assert all(torch.equal(v[1:],original[k][1:]) if k in ('mlp.4.weight','mlp.4.bias')
        else torch.equal(v,original[k]) for k,v in run.alg.actor.state_dict().items())
    monkeypatch.setattr(run.alg.optimizer,'step',original_step)
    cp=tmp_path/'auxiliary'/'checkpoint_aux_flmean_CPU_fixture.pt'
    a.save_auxiliary_checkpoint(run,cp,infos,report=report,data_receipt={'CPU_fixture_not_real_data':True},
        source_checkpoint={'path':str(source),'sha256':a.training.sha256_file(source)})
    assert not (cp.parent/'latest.json').exists()
    fresh,env2=runner();loaded=a.training.load_semantic_checkpoint(fresh,cp,contract=infos['runtime_contract'],seed=1001)
    assert fresh.alg.storage.step==0 and fresh.alg.transition.actions is None
    assert loaded['task_conditioned_hip_wheel_branch_counts']==infos['task_conditioned_hip_wheel_branch_counts']
    ledger=copy.deepcopy(loaded['task_conditioned_hip_wheel_branch'][a.LEDGER_KEY])
    assert loaded['global_policy_decisions']==128 and loaded['ppo_updates']==1 and loaded['optimizer_steps']==20
    assert a.training.state_hash(fresh.alg.optimizer.state_dict())==a.training.state_hash(adam)
    again=a.training.train_semantic(fresh,env2,run_dir=tmp_path/'next_run',output_root=tmp_path/'next_out',
        stage='full_episode',decisions=128,contract=infos['runtime_contract'],seed=1001,resume_infos=loaded)
    final,lastenv=runner();last=a.training.load_semantic_checkpoint(final,Path(again['checkpoints'][-1]['checkpoint']),
        contract=infos['runtime_contract'],seed=1001)
    assert last['task_conditioned_hip_wheel_branch'][a.LEDGER_KEY]==ledger
    assert (last['global_policy_decisions'],last['ppo_updates'],last['optimizer_steps'])==(256,2,40)
    assert last['task_conditioned_hip_wheel_branch_counts']=={'global_policy_decisions':256,'ppo_updates':2,'optimizer_steps':40}
    assert a.training.parameter_hash(final.alg.actor)==a.training.parameter_hash(fresh.alg.actor)
    assert a.training.state_hash(final.alg.optimizer.state_dict())==a.training.state_hash(fresh.alg.optimizer.state_dict())
    with pytest.raises(ValueError,match='new explicit'):a.save_auxiliary_checkpoint(run,cp,infos,
        report=report,data_receipt={},source_checkpoint={})


def test_original_real_75_and_24_readonly_no_model_no_optimization():
    from reviewed_data import load_reviewed_selection
    data,target,hold,receipt=load_reviewed_selection(a.ROOT/'outputs/ppo_task_conditioned_hip_wheel_v1/FL_minus6_gap_approach_selection.json')
    assert len(data)==len(target)==75 and len(hold)==24
    assert receipt['capture_labels'] is False and receipt['PPO_credit']==0
    assert receipt['window_summary']['locally_gap_increasing_decisions']==36


def real_pair():
    selection=json.loads((a.ROOT/'outputs/ppo_task_conditioned_hip_wheel_v1/FL_minus6_gap_approach_selection.json').read_text())
    row=selection['rows'][0];pair=[]
    with Path(selection['source_files']['probe_decisions.jsonl']['path']).open('rb') as stream:
        for kind in ('pre','post'):
            stream.seek(row[kind]['byte_offset']);pair.append(json.loads(stream.read(row[kind]['byte_length'])))
    return *pair,row


@pytest.mark.parametrize('fault',['history','ACK','other_channel','permission_mask','capture_label','different_intervention','receipt_join'])
def test_real_data_tampering_is_rejected_without_forward(fault):
    from reviewed_data import validate_pair
    import hashlib
    pre,post,row=real_pair()
    if fault=='history':pre['actual_live_history']['previous_raw_full12'][0]=0.
    if fault=='ACK':pre['previous_actual_ACK']['independent_policy_residual_requested_full12'][0]+=.1
    if fault=='other_channel':pre['manually_selected_raw_full12'][1]+=.1
    if fault=='permission_mask':pre['combined_residual_permission_mask_full12'][4]=0.
    if fault=='capture_label':row['capture_label']=True
    if fault=='different_intervention':pre['intervention']['delta_deg']['0']=-7.
    clean={k:v for k,v in pre.items() if k!='pre_action_receipt_sha256'}
    digest=hashlib.sha256(json.dumps(clean,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    pre['pre_action_receipt_sha256']=post['pre_action_receipt_sha256']=row['pre_action_receipt_sha256']=digest
    if fault=='receipt_join':post['decision']+=1
    with pytest.raises(ValueError):validate_pair(pre,post,row,training=True)


def test_truncated_still_contiguous_window_rejected(tmp_path):
    from reviewed_data import load_reviewed_selection
    selection=json.loads((a.ROOT/'outputs/ppo_task_conditioned_hip_wheel_v1/FL_minus6_gap_approach_selection.json').read_text())
    selection['rows']=selection['rows'][1:];selection['decision_indices']=selection['decision_indices'][1:]
    path=tmp_path/'wrong_window.json';path.write_text(json.dumps(selection))
    with pytest.raises(ValueError,match='whole contiguous'):load_reviewed_selection(path)


def test_no_update_cannot_publish_aux_and_original_branch_is_preserved():
    infos={'task_conditioned_hip_wheel_branch':{'branch_id':a.EXPERIMENT,'counter_origin':{'a':1}},
        'global_policy_decisions':100,'ppo_updates':7,'optimizer_steps':140,'stage_requested_decisions':{'full_episode':100}}
    with pytest.raises(ValueError,match='no-update'):
        a.append_ledger(infos,report={'accepted_auxiliary_updates':0},data_receipt={},source_checkpoint={},helper_sha256='x')
    before=copy.deepcopy(infos)
    out=a.append_ledger(infos,report={'accepted_auxiliary_updates':2,'attempted_auxiliary_optimizer_steps':3},
        data_receipt={},source_checkpoint={},helper_sha256='x')
    assert infos==before and out['task_conditioned_hip_wheel_branch']['counter_origin']=={'a':1}
    assert out['task_conditioned_hip_wheel_branch'][a.LEDGER_KEY]['accepted_auxiliary_updates_total']==2


def test_cpu_inspection_restores_rng_and_does_not_optimizer_step(tmp_path,monkeypatch):
    import auxiliary_cli as c
    run,env,infos,cp=synthetic_source(tmp_path);tx,target,hx=states(run)
    metadata=json.loads(cp.with_name(cp.stem+'_manifest.json').read_text())
    before=a.training.capture_training_rng_state(seed=1001)
    def forbid(*args,**kwargs):raise AssertionError('inspection tried to update')
    monkeypatch.setattr(torch.optim.Adam,'step',forbid)
    monkeypatch.setattr(torch.optim.SGD,'step',forbid)
    result=c.cpu_inspection(cp,metadata,tx['policy'].tolist(),target.tolist(),hx['policy'].tolist())
    assert result['automatic_aux_enabled'] is False
    assert a.training.capture_training_rng_state(seed=1001)==before


def test_compatibility_rejects_changed_MDP_and_policy():
    import auxiliary_cli as c
    contract={'experiment_id':a.EXPERIMENT}
    metadata={'runtime_contract':contract,'policy_contract':{'version':TASK_CONDITIONED_HIP_WHEEL_POLICY}}
    receipt={'diagnostic_policy_contract':metadata['policy_contract'],'diagnostic_runtime_contract':contract}
    assert c.verify_data_compatibility(metadata,receipt,contract)['mode']=='exact_same_runtime_and_policy'
    wrong=copy.deepcopy(receipt);wrong['diagnostic_policy_contract']['version']='wrong'
    with pytest.raises(ValueError):c.verify_data_compatibility(metadata,wrong,contract)
    with pytest.raises(ValueError):c.verify_data_compatibility(metadata,receipt,{'experiment_id':'wrong'})
