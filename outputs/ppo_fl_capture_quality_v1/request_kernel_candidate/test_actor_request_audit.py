"""Isolated copied-source tests; no Isaac, no GPU, no real training credit."""
from __future__ import annotations
import ast
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable
from wlr50_clean.ppo import semantic_history_actor as a
from wlr50_clean.ppo import semantic_policy_distribution as p
from wlr50_clean.ppo import semantic_training as t

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CP = ROOT/'outputs/ppo_fl_capture_quality_v1/checkpoints/history/checkpoint_step_000178688.pt'
RUN = ROOT/'runs/ppo_fl_capture_quality_v1/train/20260918T0412242524179Z_gf2e552406ea7_541c9a6d6c8e436d811fa7bcd64a885d'
NEW = p.HISTORY_REQUEST_CAP_TRANSITION_POLICY
OLD = p.HISTORY_QUARTER_TEMPERED_POLICY
LAYOUT = p.ROLE_OBSERVATION_LAYOUT


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    threads,rng=torch.get_num_threads(),torch.get_rng_state()
    torch.set_num_threads(1)
    torch.manual_seed(85101)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    yield
    torch.set_rng_state(rng)
    torch.set_num_threads(threads)


@pytest.fixture(scope='module')
def saved():
    return torch.load(CP,map_location='cpu',weights_only=False),torch.load(
        RUN/'rollouts/rollout_001362.pt',map_location='cpu',weights_only=False)


def obs(value):
    return TensorDict({'policy':value,'critic':value.clone()},batch_size=value.shape[:-1])


def config(version):
    return t.semantic_runner_config(seed=1001,device='cpu',semantic_version='v3',
        policy_version=version,observation_layout=LAYOUT,return_profile=t.RETURN_PROFILE)


def make_actor(saved,version=NEW):
    c=copy.deepcopy(config(version)['actor'])
    cls=resolve_callable(c.pop('class_name'))
    model=cls(obs(saved[1]['observations']['policy'][:1,0]),{'actor':['policy']},'actor',12,**c)
    model.load_state_dict(saved[0]['actor_state_dict'],strict=True)
    return model


def exact(a,b):
    if isinstance(a,torch.Tensor): assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a: exact(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert type(a) is type(b) and len(a)==len(b)
        for x,y in zip(a,b): exact(x,y)
    else: assert a==b


def test_overlay_old_classes_and_contracts_unchanged(saved):
    for module in (a,p,t): assert HERE in Path(module.__file__).parents
    original=(ROOT/'src/wlr50_clean/ppo/semantic_history_actor.py').read_text()
    copied=Path(a.__file__).read_text()
    trees=[ast.parse(v) for v in (original,copied)]
    for node in trees[0].body:
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)):
            candidate=next(x for x in trees[1].body if isinstance(x,type(node)) and x.name==node.name)
            assert ast.get_source_segment(original,node)==ast.get_source_segment(copied,candidate)
    old_metadata=saved[0]['infos']
    assert p.policy_contract(OLD,observation_layout=LAYOUT)==old_metadata['policy_contract']
    assert p.policy_version_from_metadata(old_metadata)==OLD
    old,new=config(OLD),config(NEW)
    old['actor']['class_name']=p.HISTORY_REQUEST_CAP_TRANSITION_ACTOR_CLASS
    assert old==new
    metadata={**old_metadata,'runner_config':new,'policy_contract':p.policy_contract(NEW,observation_layout=LAYOUT)}
    assert p.policy_version_from_metadata(metadata)==NEW
    assert p.supported_heteroscedastic_contract_version(metadata['policy_contract'])==NEW


@pytest.mark.parametrize('stochastic',[False,True])
@pytest.mark.parametrize('version',[OLD,NEW])
@pytest.mark.parametrize('index',[14,15])
def test_request_hook_one_forward_same_rng_and_truthful_history(saved,stochastic,version,index):
    model=make_actor(saved,version)
    observation=obs(saved[1]['observations']['policy'][index:index+1,0].clone())
    before=copy.deepcopy(model.state_dict())
    rng=torch.get_rng_state()
    expected=model(observation,stochastic_output=stochastic)
    after=torch.get_rng_state()
    torch.set_rng_state(rng)
    actual,record=t.audited_history_policy_request(model,observation,
        lambda:model(observation,stochastic_output=stochastic),stochastic=stochastic)
    assert torch.equal(actual,expected) and torch.equal(after,torch.get_rng_state())
    exact(before,model.state_dict())
    assert record['sampling_draws']==int(stochastic)
    assert record['extra_model_forwards']==record['extra_random_draws']==0
    assert record['previous_raw_from_current_observation_full12']==observation['policy'][0,195:207].tolist()
    if version==NEW:
        center,evidence=a.cap_transition_request_history(observation['policy'])
        assert record['history_center_full12']==center[0].tolist()
        assert record['cap_transition_gate_full12']==evidence['gate_full12'][0].tolist()
        assert sum(record['cap_transition_gate_full12'])==(10 if index==14 else 0)
        assert record['previous_filtered_request_full12']==evidence['previous_filtered_request_full12'][0].tolist()
        assert record['policy_version']==NEW
        if index==14: assert record['history_center_full12'][3] != record['previous_raw_from_current_observation_full12'][3]
    else:
        assert record['schema']=='wlr50_clean.actual_history_policy_request.v1'
        assert 'history_center_full12' not in record


def test_real_saved_batch_only_ten_means_change_sigma_and_nonentry_samples_bitwise(saved):
    old,new=make_actor(saved,OLD),make_actor(saved)
    observation=obs(saved[1]['observations']['policy'][:,0].clone())
    _,evidence=a.cap_transition_request_history(observation['policy'])
    gate=evidence['gate_full12']
    first=old(observation);second=new(observation)
    assert int(gate.sum())==10
    assert torch.equal(first[~gate],second[~gate])
    torch.testing.assert_close(second[14,3],torch.tensor(.150666125),rtol=0,atol=2e-7)
    old(observation,stochastic_output=True);new(observation,stochastic_output=True)
    assert torch.equal(old.output_std,new.output_std)
    actions=saved[1]['actions'][:,0]
    expected=torch.distributions.Normal(new.output_mean,new.output_std)
    assert torch.equal(new.get_output_log_prob(actions),expected.log_prob(actions).sum(-1))
    exact(old.state_dict(),new.state_dict())
    assert set(vars(old))==set(vars(new))


@pytest.mark.parametrize('phase,age,predecessor,indices',[
    (0,0.,False,[]),(2,0.,True,[8,10]),(5,0.,True,list(range(10))),
    (5,1/15,True,[]),(5,0.,False,[]),(4,0.,True,[]),
])
def test_observable_gate_zero_equivalence_and_exact_inverse(saved,phase,age,predecessor,indices):
    values=saved[1]['observations']['policy'][14:15,0].clone()
    values[:,:13]=0;values[:,phase]=1;values[:,20]=age/200
    values[:,158:171]=0
    if predecessor: values[:,158:158+phase]=1
    values[:,207:219]=torch.tensor(p.REQUEST_HISTORY_CAPS[max(phase-1,0)])*.6/torch.tensor(p.REQUEST_HISTORY_SCALES)
    center,evidence=a.cap_transition_request_history(values)
    assert evidence['gate_full12'][0].nonzero().flatten().tolist()==indices
    assert torch.equal(center[~evidence['gate_full12']],values[:,195:207][~evidence['gate_full12']])
    if indices:
        recovered=center[0,indices].tanh()*torch.tensor(p.REQUEST_HISTORY_CAPS[phase])[indices]
        torch.testing.assert_close(recovered,evidence['previous_filtered_request_full12'][0,indices],rtol=1e-6,atol=1e-6)
    values[:,195:231]=0
    old,new=make_actor(saved,OLD),make_actor(saved)
    assert torch.equal(old(obs(values)),new(obs(values)))


@pytest.mark.parametrize('fault',['nan','clip','no_stage','double_stage','fractional_completed','negative_age','outside_predecessor_cap','dimension'])
def test_invalid_request_history_encoding_rejected(saved,fault):
    values=saved[1]['observations']['policy'][14:15,0].clone()
    if fault=='nan': values[0,210]=float('nan')
    elif fault=='clip': values[0,195]=20.01
    elif fault=='no_stage': values[0,:13]=0
    elif fault=='double_stage': values[0,0]=1
    elif fault=='fractional_completed': values[0,162]=.5
    elif fault=='negative_age': values[0,20]=-.001
    elif fault=='outside_predecessor_cap': values[0,210]=25/6
    elif fault=='dimension': values=values[:,:371]
    with pytest.raises(ValueError): a.cap_transition_request_history(values)


def test_shuffled_padded_minibatch_reconstructs_same_conditional_distribution(saved):
    model=make_actor(saved)
    values=saved[1]['observations']['policy'][10:22,0].clone()
    direct=model(obs(values))
    order=torch.tensor([4,10,1,5,9,2,6,11,0,7,3,8])
    torch.testing.assert_close(model(obs(values[order])),direct[order],rtol=1e-6,atol=2e-7)
    padded=values.reshape(4,3,372).clone()
    masks=torch.tensor([[True,True,True],[True,True,True],[True,True,False],[False,False,False]])
    padded[~masks]=float('nan') # Padding must be removed before semantic validation.
    from rsl_rl.utils import unpad_trajectories
    observation=obs(padded)
    expected=model(unpad_trajectories(observation,masks))
    assert torch.equal(model(observation,masks=masks),expected)


@pytest.mark.parametrize('field',['policy_contract','class','rho','cap','request_scale','layout'])
def test_metadata_tamper_and_mixed_kernel_rejected(saved,field):
    metadata={**saved[0]['infos'],'runner_config':config(NEW),'policy_contract':p.policy_contract(NEW,observation_layout=LAYOUT)}
    if field=='policy_contract': metadata['policy_contract']=p.policy_contract(OLD,observation_layout=LAYOUT)
    elif field=='class': metadata['runner_config']['actor']['class_name']=p.HISTORY_QUARTER_TEMPERED_ACTOR_CLASS
    elif field=='rho': metadata['policy_contract']['rho']=.8
    elif field=='cap': metadata['policy_contract']['phase_caps_full12'][5][3]=111.
    elif field=='request_scale': metadata['policy_contract']['history_request_scales'][3]=4.
    elif field=='layout': metadata['runner_config']['actor']['observation_layout']=None
    with pytest.raises(ValueError): p.policy_version_from_metadata(metadata)


def make_algorithm(saved):
    initial=obs(saved[1]['observations']['policy'][:1,0])
    c=copy.deepcopy(config(NEW)['critic']);c.pop('class_name')
    critic=MLPModel(initial,{'critic':['critic']},'critic',1,**c)
    algorithm=PPO(make_actor(saved),critic,RolloutStorage('rl',1,32,initial,[12],device='cpu'),
        num_learning_epochs=5,num_mini_batches=4,learning_rate=1e-5,schedule='adaptive',desired_kl=.01,
        gamma=.9985,lam=.99,device='cpu')
    # CPU Adam.load_state_dict may retain tensor storage from its input mapping;
    # independent runs must not share the in-memory fixture's moment buffers.
    algorithm.load(copy.deepcopy(saved[0]),load_cfg=None,strict=True)
    return algorithm


def collect(algorithm,saved):
    with torch.no_grad():
        for k in range(32):
            observation=obs(saved[1]['observations']['policy'][k:k+1,0].clone())
            t.audited_history_policy_request(algorithm.actor,observation,lambda:algorithm.act(observation),stochastic=True)
            algorithm.process_env_step(observation,torch.tensor([math.sin(k)*.02]),torch.tensor([k==31]),{})
        algorithm.compute_returns(observation)


def test_actual_PPO_likelihood_audit_no_model_optimizer_rng_change(saved,tmp_path):
    left,right=make_algorithm(saved),make_algorithm(saved)
    rng=torch.get_rng_state()
    collect(left,saved)
    after_collection=torch.get_rng_state()
    torch.set_rng_state(rng)
    collect(right,saved)
    assert torch.equal(after_collection,torch.get_rng_state())
    exact(left.storage.actions,right.storage.actions)
    update_rng=torch.get_rng_state()
    left.update()
    final_rng=torch.get_rng_state()
    torch.set_rng_state(update_rng)
    path=tmp_path/'actual_likelihood.json'
    result=t.audited_ppo_update(SimpleNamespace(alg=right,_semantic_policy_version=NEW),likelihood_audit_path=path)
    assert result['optimizer_steps']==20 and result['actor_parameters_changed']
    exact(left.save(),right.save())
    assert torch.equal(final_rng,torch.get_rng_state())
    audit=json.loads(path.read_text())
    assert len(audit['minibatches'])==20
    assert max(abs(v-1) for v in audit['minibatches'][0]['ratio'])<2e-5
    assert all('request207_219' in row['history_source'] for row in audit['minibatches'])
    assert audit['extra_model_forwards']==audit['extra_random_draws']==0
    assert left.storage.step==right.storage.step==0
