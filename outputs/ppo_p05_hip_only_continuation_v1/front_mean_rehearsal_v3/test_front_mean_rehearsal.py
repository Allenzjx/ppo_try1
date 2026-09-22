"""Synthetic CPU parameterization/inspection/trust tests; no real fit."""
from dataclasses import replace
import copy
import inspect as python_inspect
import pytest
import torch
import front_mean_rehearsal as m


@pytest.fixture(autouse=True)
def cpu_only():
    assert not torch.cuda.is_available(),'CUDA_VISIBLE_DEVICES=-1 required'
    before=torch.get_rng_state(); threads=torch.get_num_threads()
    torch.set_num_threads(1); torch.manual_seed(719)
    yield
    torch.set_rng_state(before); torch.set_num_threads(threads)


def actor():
    x=torch.zeros(1,389); x[:,0]=1
    from wlr50_clean.ppo.semantic_p05_capture_profile import P05_CAPTURE_OBSERVATION_LAYOUT
    return m.kernel.SemanticP05CaptureHistoryMLPModel(m.kernel.tensors(x),
        {'actor':['policy'],'critic':['critic']},'actor',12,hidden_dims=(256,256),activation='elu',
        obs_normalization=False,distribution_cfg={'class_name':'HeteroscedasticGaussianDistribution',
        'std_type':'log','init_std':.15},observation_layout=P05_CAPTURE_OBSERVATION_LAYOUT,
        exploration_std_temperature=.25)


def data(a):
    x=torch.zeros(7,389); x[:,1]=1; x[0,:2]=torch.tensor([1.,0.]); x[:,20]=.01
    x[:,30]=torch.linspace(-.1,.1,7); x[:,195:207]=torch.linspace(-.2,.2,84).reshape(7,12)
    v=x[1:].clone(); v[:,30]+=.031
    p=x[:4].clone(); p[:,:13]=0; p[torch.arange(4),torch.tensor([2,3,4,5])]=1
    p01=x[:2].clone(); p01[:,:13]=0; p01[:,0]=1; p01[:,30]+=.012
    with torch.no_grad():
        y=m.distribution(a,m.kernel.tensors(x))['mean']+torch.linspace(.01,.03,12)
        vy=m.distribution(a,m.kernel.tensors(v))['mean']+torch.linspace(.01,.03,12)
    return {'train_observations':x,'train_raw_targets':y,'validation_observations':v,
        'validation_raw_targets':vy,'train_phase_groups':{'P01':[0],'P02':list(range(1,7))},
        'protection_observations':p,'p01_observations':p01,'receipt':{'synthetic_only':True}}


def objective():
    # Explicit fixture values only; NOT an executable real-data authorization.
    return m.Objective({'P01':.25,'P02':.75},1.,(1.,)*12)


def budget(**kwargs):
    return replace(m.Budget(2,.01,(3.,)*8+(.15,)*4,(3.,)*8+(.15,)*4,
        (1.,)*8+(.05,)*4,(3.,)*8+(.15,)*4,.1,.03),**kwargs)


def test_layout_and_initial_full_gaussian_equals_official_kernel():
    a=actor(); d=data(a); obs,_,_=m.dataset(a,d); leaf=m.selected_leaf(a)
    assert leaf.shape==(12,257)
    for x in obs.values():
        before=m.kernel.distribution(a,x); after=m.distribution(a,x,leaf)
        for key in ('mean','sigma','log_sigma','request'): assert torch.equal(before[key],after[key])
        assert torch.equal(after['mean'],a(x,stochastic_output=False))


def test_mean_bias_affects_one_channel_all_phases_with_true_point1_chain():
    a=actor(); obs,_,_=m.dataset(a,data(a)); leaf=m.selected_leaf(a); candidate=leaf.clone()
    candidate[3,256]+=.02
    for name,x in obs.items():
        before=m.distribution(a,x,leaf); after=m.distribution(a,x,candidate)
        assert torch.equal(before['sigma'],after['sigma']) and torch.equal(before['log_sigma'],after['log_sigma'])
        assert torch.allclose(after['mean'][:,3]-before['mean'][:,3],torch.full((len(x),),.002),atol=3e-8,rtol=0)
        other=[i for i in range(12) if i!=3]
        assert torch.equal(before['mean'][:,other],after['mean'][:,other])
        if name=='protection': assert not torch.equal(before['mean'],after['mean'])


def test_readonly_inspect_separate_phase_gradients_protection_and_rng(monkeypatch):
    a=actor(); d=data(a); saved=copy.deepcopy(a.state_dict()); rng=m.training.capture_training_rng_state(seed=0)
    def forbid(*args,**kwargs): raise AssertionError('inspection invoked optimization')
    monkeypatch.setattr(torch.optim.SGD,'step',forbid); monkeypatch.setattr(torch.optim.Adam,'step',forbid)
    result=m.inspect(a,d,objective=objective())
    assert result['optimized_scalar_count']==3084 and result['phase_sample_counts']=={'P01':1,'P02':6}
    report=result['phase_separated_losses_and_gradients']
    assert report['P01']['weight']==.25 and report['P02']['weight']==.75
    assert report['P01']['gradient_l2']>0 and report['P02']['gradient_l2']>0
    assert report['protection']['gradient_l2']==0
    assert result['protection_directional_second_derivative_at_source']>0
    g=lambda name:torch.tensor(report[name]['gradient_full12x257'])
    assert torch.allclose(g('total'),.25*g('P01')+.75*g('P02')+g('protection'),atol=1e-9,rtol=1e-6)
    assert all(r['log_sigma_per_unit_lr_exact_zero'] for r in result['negative_total_gradient_unit_lr_response'].values())
    assert result['P03plus_mean_bitwise_invariance_claimed'] is False
    assert result['optimizer_steps_performed']==0 and result['physical_success_claimed'] is False
    assert m.training.capture_training_rng_state(seed=0)==rng
    assert all(torch.equal(v,saved[k]) for k,v in a.state_dict().items())
    assert all(p.grad is None for p in a.parameters())


def test_phase_loss_not_silently_diluted_by_number_of_P02_rows():
    a=actor(); d=data(a); obs,target,groups=m.dataset(a,d); base=m.baselines(a,obs)
    leaf=m.selected_leaf(a).requires_grad_(True)
    terms=m.loss_terms(a,obs,target,groups,leaf,base['protection']['mean'],objective())
    assert torch.allclose(terms['total'],.25*terms['P01']+.75*terms['P02'])
    altered=copy.deepcopy(d); altered['train_raw_targets'][0]+=.2
    o2,t2,g2=m.dataset(a,altered)
    terms2=m.loss_terms(a,o2,t2,g2,leaf,base['protection']['mean'],objective())
    assert torch.equal(terms['P02'],terms2['P02'])
    assert torch.allclose(terms2['total']-terms['total'],.25*(terms2['P01']-terms['P01']))


def test_protection_loss_has_nonzero_gradient_after_mean_proposal():
    a=actor(); obs,target,groups=m.dataset(a,data(a)); base=m.baselines(a,obs)
    leaf=m.selected_leaf(a); leaf[:,256]+=.01; leaf.requires_grad_(True)
    terms=m.loss_terms(a,obs,target,groups,leaf,base['protection']['mean'],objective())
    grad,=torch.autograd.grad(terms['protection'],leaf)
    assert terms['protection']>0 and grad.norm()>0


def test_added_real_rear_phase_probes_and_explicit_missing_P13():
    a=actor(); d=data(a)
    extra=d['protection_observations'][:1].repeat(6,1);extra[:,:13]=0
    extra[torch.arange(6),torch.arange(6,12)]=1
    d['protection_observations']=torch.cat((d['protection_observations'],extra))
    report=m.inspect(a,d,objective=objective())
    assert report['actual_protection_phases']==list(range(3,13))
    assert report['missing_actual_protection_phases']==[13]
    assert report['P13_actual_protection_coverage'] is False
    d['protection_observations'][-1,:13]=0;d['protection_observations'][-1,12]=1
    with pytest.raises(ValueError):m.dataset(a,d)


def test_guard_accepts_small_mean_change_and_rejects_protection_request_bound():
    a=actor(); obs,_,_=m.dataset(a,data(a)); base=m.baselines(a,obs)
    saved=m.training.parameter_hash(a); leaf=m.selected_leaf(a); leaf[:,256]+=.001
    report=m.guard_candidate(a,obs,base,leaf,budget=budget())
    assert report['accepted_by_trust_only'] and not report['P03plus_mean_bitwise_invariance_claimed']
    tight=budget(maximum_protection_request_shift_full12=(1e-9,)*12)
    reject=m.guard_candidate(a,obs,base,leaf,budget=tight)
    assert not reject['accepted_by_trust_only'] and 'protection_REQUEST_bound' in reject['rejection_reasons']
    assert m.training.parameter_hash(a)==saved


def test_full_gaussian_kl_guard_and_sigma_exact_check():
    a=actor(); obs,_,_=m.dataset(a,data(a)); base=m.baselines(a,obs)
    leaf=m.selected_leaf(a); leaf[:,256]+=.01
    reject=m.guard_candidate(a,obs,base,leaf,budget=budget(maximum_protection_full_gaussian_kl=1e-12))
    assert 'protection_full_Gaussian_KL_bound' in reject['rejection_reasons']
    base['validation']['sigma']*=1.001
    reject=m.guard_candidate(a,obs,base,leaf,budget=budget())
    assert 'validation_sigma_changed' in reject['rejection_reasons']


@pytest.mark.parametrize('bad',[{}, {'P01':0,'P02':1},{'P01':True,'P02':1},{'P01':1,'P02':float('nan')}])
def test_explicit_phase_weights_required(bad):
    with pytest.raises(ValueError):m.Objective(bad,1.,(1.,)*12).validate()


@pytest.mark.parametrize('bad',[{'max_attempts':33},{'max_attempts':True},{'learning_rate':0},
    {'maximum_protection_full_gaussian_kl':float('inf')},{'maximum_p01_probe_request_shift_full12':(0.,)*12}])
def test_finite_budget_no_automatic_defaults(bad):
    with pytest.raises(ValueError):budget(**bad).validate()


@pytest.mark.parametrize('bad',['groups','phase','validation_overlap','protection_phase','probe_phase'])
def test_data_scope_checks(bad):
    a=actor(); d=data(a)
    if bad=='groups':d['train_phase_groups']['P01']=[]
    if bad=='phase':d['train_observations'][0,:3]=torch.tensor([0.,0.,1.])
    if bad=='validation_overlap':d['validation_observations'][0]=d['train_observations'][1]
    if bad=='protection_phase':d['protection_observations'][0,:13]=0;d['protection_observations'][0,1]=1
    if bad=='probe_phase':d['p01_observations'][0,:2]=torch.tensor([0.,1.])
    with pytest.raises(ValueError):m.dataset(a,d)


def test_no_fit_save_or_production_mutating_entry_point():
    assert not hasattr(m,'fit') and not hasattr(m,'main')
    source=python_inspect.getsource(m)
    assert 'torch.optim.' not in source and 'save_semantic_checkpoint(' not in source
