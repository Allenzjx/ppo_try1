"""Synthetic CPU-only phase adapter tests; fixture budgets are NOT real authority."""
from dataclasses import replace
from copy import deepcopy
import pytest
import torch
import rr_mean_rehearsal as m


@pytest.fixture(autouse=True)
def cpu_only():
    assert not torch.cuda.is_available(),'CUDA_VISIBLE_DEVICES=-1 required'
    before=torch.get_rng_state();threads=torch.get_num_threads()
    torch.set_num_threads(1);torch.manual_seed(719)
    yield
    torch.set_rng_state(before);torch.set_num_threads(threads)


def actor():
    x=torch.zeros(1,389);x[:,0]=1
    from wlr50_clean.ppo.semantic_p05_capture_profile import P05_CAPTURE_OBSERVATION_LAYOUT
    return m.kernel.SemanticP05CaptureHistoryMLPModel(m.kernel.tensors(x),
        {'actor':['policy'],'critic':['critic']},'actor',12,hidden_dims=(256,256),activation='elu',
        obs_normalization=False,distribution_cfg={'class_name':'HeteroscedasticGaussianDistribution',
        'std_type':'log','init_std':.15},observation_layout=P05_CAPTURE_OBSERVATION_LAYOUT,
        exploration_std_temperature=.25)


def data(a):
    x=torch.zeros(7,389);x[torch.arange(7),torch.tensor([8,8,8,9,10,10,10])]=1
    x[:,20]=.01;x[:,30]=torch.linspace(-.1,.1,7)
    x[:,195:207]=torch.linspace(-.2,.2,84).reshape(7,12)
    x[3:,157]=1  # Exercise existing receiving-wheel sigma on actual phase bits.
    v=x[[0,2,4,6]].clone();v[:,30]+=.031
    p=x[:1].repeat(9,1);p[:,:13]=0;p[torch.arange(9),torch.tensor([0,1,2,3,4,5,6,7,11])]=1
    p[:,30]+=torch.arange(9)*.01
    p01=x[:2].clone();p01[:,:13]=0;p01[:,0]=1;p01[:,30]+=.012
    with torch.no_grad():
        y=m.distribution(a,m.kernel.tensors(x))['mean']+torch.linspace(.01,.03,12)
        vy=m.distribution(a,m.kernel.tensors(v))['mean']+torch.linspace(.01,.03,12)
    return {'train_observations':x,'train_raw_targets':y,'validation_observations':v,
        'validation_raw_targets':vy,'train_phase_groups':{'P09':[0,1,2],'P10':[3],'P11':[4,5,6]},
        'validation_phase_groups':{'P09':[0,1],'P11':[2,3]},
        'protection_observations':p,'p01_observations':p01,'receipt':{'synthetic_only':True}}


def objective():return m.Objective({'P09':.45,'P10':.1,'P11':.45},1.,(1.,)*12)


def budget(**kwargs):
    return replace(m.Budget(2,.01,(3.,)*8+(.15,)*4,(3.,)*8+(.15,)*4,
        (1.,)*8+(.05,)*4,(3.,)*8+(.15,)*4,.1,.03),**kwargs)


def test_frozen_mechanics_layout_and_full_gaussian_match():
    assert m.distribution is m.frozen.distribution and m.guard_candidate is m.frozen.guard_candidate
    a=actor();obs,_,_=m.dataset(a,data(a));leaf=m.selected_leaf(a)
    assert leaf.shape==(12,257)
    for x in obs.values():
        before=m.kernel.distribution(a,x);after=m.distribution(a,x,leaf)
        assert all(torch.equal(before[key],after[key]) for key in ('mean','sigma','log_sigma','request'))
        assert torch.equal(after['mean'],a(x,stochastic_output=False))


def test_all_twelve_mean_rows_can_change_jointly_sigma_and_point1_chain_remain_exact():
    a=actor();obs,_,_=m.dataset(a,data(a));leaf=m.selected_leaf(a);candidate=leaf.clone()
    candidate[:,256]+=.02
    for x in obs.values():
        before=m.distribution(a,x,leaf);after=m.distribution(a,x,candidate)
        assert torch.equal(before['sigma'],after['sigma']) and torch.equal(before['log_sigma'],after['log_sigma'])
        assert torch.allclose(after['mean']-before['mean'],torch.full_like(before['mean'],.002),atol=4e-8,rtol=0)


def test_inspect_phase_balancing_current_protection_and_rng_no_fit(monkeypatch):
    a=actor();d=data(a);saved=deepcopy(a.state_dict());rng=m.training.capture_training_rng_state(seed=0)
    def forbid(*args,**kwargs):raise AssertionError('readonly optimizer called')
    monkeypatch.setattr(torch.optim.SGD,'step',forbid);monkeypatch.setattr(torch.optim.Adam,'step',forbid)
    r=m.inspect(a,d,objective=objective());terms=r['phase_separated_losses_and_gradients']
    assert r['phase_sample_counts']=={'P09':3,'P10':1,'P11':3}
    assert r['validation_phase_sample_counts']=={'P09':2,'P11':2} and r['P10_independent_validation_rows']==0
    assert r['actual_protection_phases']==list(m.PROTECTION_PHASES)
    assert r['missing_actual_protection_phases']==[9,10,11,13] and not r['P13_actual_protection_coverage']
    assert terms['protection']['gradient_l2']==0 and r['protection_directional_second_derivative_at_source']>0
    assert all(terms[k]['gradient_l2']>0 for k in m.POSITIVE_PHASES)
    grad=lambda k:torch.tensor(terms[k]['gradient_full12x257'])
    assert torch.allclose(grad('total'),.45*grad('P09')+.1*grad('P10')+.45*grad('P11'),atol=1e-9,rtol=1e-6)
    assert all(v['log_sigma_per_unit_lr_exact_zero'] for v in r['negative_total_gradient_unit_lr_response'].values())
    assert r['single_historical_trajectory_train_validation_correlated'] and not r['physical_success_claimed']
    assert not r['P03plus_mean_bitwise_invariance_claimed'] and not r['future_closed_loop_invariance_claimed']
    assert all(torch.equal(v,saved[k]) for k,v in a.state_dict().items()) and all(p.grad is None for p in a.parameters())
    assert m.training.capture_training_rng_state(seed=0)==rng


def test_single_P10_weight_not_implicitly_one_seventh():
    a=actor();d=data(a);obs,targets,groups=m.dataset(a,d);reference=m.baselines(a,obs)
    leaf=m.selected_leaf(a).requires_grad_(True)
    first=m.loss_terms(a,obs,targets,groups,leaf,reference['protection']['mean'],objective())
    targets['train']=targets['train'].clone();targets['train'][3]+=.2
    second=m.loss_terms(a,obs,targets,groups,leaf,reference['protection']['mean'],objective())
    assert torch.equal(first['P09'],second['P09']) and torch.equal(first['P11'],second['P11'])
    assert torch.allclose(second['total']-first['total'],.1*(second['P10']-first['P10']))


def test_protection_is_current_mean_and_guarded_all_12_channels():
    a=actor();obs,target,groups=m.dataset(a,data(a));reference=m.baselines(a,obs)
    leaf=m.selected_leaf(a);leaf[:,256]+=.01;leaf.requires_grad_(True)
    terms=m.loss_terms(a,obs,target,groups,leaf,reference['protection']['mean'],objective())
    g,=torch.autograd.grad(terms['protection'],leaf)
    assert terms['protection']>0 and bool((g.norm(dim=1)>0).all())
    rejected=m.guard_candidate(a,obs,reference,leaf,budget=budget(maximum_protection_request_shift_full12=(1e-9,)*12))
    assert 'protection_REQUEST_bound' in rejected['rejection_reasons']
    rejected=m.guard_candidate(a,obs,reference,leaf,budget=budget(maximum_protection_full_gaussian_kl=1e-12))
    assert 'protection_full_Gaussian_KL_bound' in rejected['rejection_reasons']


@pytest.mark.parametrize('bad',['front_positive','fake_P10_validation','groups','overlap','P13_protection','no_P01_protection','wrong_probe'])
def test_scope_and_holdout_fail_closed(bad):
    a=actor();d=data(a)
    if bad=='front_positive':d['train_observations'][0,:13]=0;d['train_observations'][0,1]=1
    if bad=='fake_P10_validation':d['validation_observations'][0,:13]=0;d['validation_observations'][0,9]=1
    if bad=='groups':d['train_phase_groups']['P10']=[]
    if bad=='overlap':d['validation_observations'][0]=d['train_observations'][0]
    if bad=='P13_protection':d['protection_observations'][-1,:13]=0;d['protection_observations'][-1,12]=1
    if bad=='no_P01_protection':d['protection_observations']=d['protection_observations'][1:]
    if bad=='wrong_probe':d['p01_observations'][0,:13]=0;d['p01_observations'][0,1]=1
    with pytest.raises(ValueError):m.dataset(a,d)


@pytest.mark.parametrize('weights',[{}, {'P01':1,'P02':1},{'P09':1,'P10':0,'P11':1}])
def test_objective_requires_explicit_all_three_positive_phase_weights(weights):
    with pytest.raises(ValueError):m.Objective(weights,1.,(1.,)*12).validate()


@pytest.mark.parametrize('bad',[{'max_attempts':33},{'learning_rate':0},{'maximum_per_state_full_gaussian_kl':float('inf')}])
def test_no_automatic_budget_or_more_than_32_attempts(bad):
    with pytest.raises(ValueError):budget(**bad).validate()
