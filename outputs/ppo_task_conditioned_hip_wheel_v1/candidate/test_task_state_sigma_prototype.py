"""Isolated CPU algebra checks only: zero real policy decisions/PPO updates."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from wlr50_clean.ppo.semantic_history_actor import (
    HISTORY_RHO, cap_transition_request_history, history_conditioned_head,
    physical_innovation_effective_log_std,
)
from wlr50_clean.ppo.semantic_policy_distribution import REQUEST_HISTORY_CAPS, REQUEST_HISTORY_SCALES

path = Path(__file__).with_name("task_state_sigma_prototype.py")
spec = importlib.util.spec_from_file_location("output_only_sigma_candidate", path)
candidate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(candidate)


@pytest.fixture(autouse=True)
def cpu():
    state, threads = torch.get_rng_state(), torch.get_num_threads()
    torch.manual_seed(20260920)
    torch.set_num_threads(1)
    yield
    torch.set_rng_state(state)
    torch.set_num_threads(threads)


def obs(stage):
    x = torch.zeros(1,372)
    x[0,stage] = 1
    x[0,158:158+stage] = 1
    x[0,20] = .01
    x[0,195:207] = torch.linspace(-.2,.3,12)
    x[0,207:219] = .1
    return x


def conditional(x):
    head = torch.stack((torch.linspace(-.3,.25,12), torch.linspace(-1.3,-.5,12))).unsqueeze(0)
    head = head.expand(*x.shape[:-1],2,12).clone()
    center, _ = cap_transition_request_history(x)
    return history_conditioned_head(head,center,HISTORY_RHO)


@pytest.mark.parametrize("stage",range(13))
def test_all_phases_positive_full12_mean_cap_no_RNG_or_history_mutation(stage):
    x=obs(stage); original=x.clone(); head=conditional(x)
    state=torch.get_rng_state()
    result,e=candidate.candidate_effective_head(head,x)
    assert torch.equal(state,torch.get_rng_state()) and torch.equal(x,original)
    assert torch.equal(result[...,0,:],head[...,0,:])
    assert torch.equal(e['unchanged_cap_full12'],torch.tensor(REQUEST_HISTORY_CAPS[stage]).unsqueeze(0))
    assert bool((result[...,1,:].exp()>0).all())
    if stage not in (6,7,8):
        old,_=physical_innovation_effective_log_std(head[...,1,:],x,.25)
        torch.testing.assert_close(result[...,1,:],old,rtol=0,atol=2.4e-7)


def test_FR_usable_air_gate_does_not_remove_initial_preparation():
    x=obs(1); x[0,147]=1; x[0,33]=3; x[0,24]=.015
    B,cap,e=candidate.candidate_physical_scales(x)
    assert B[0,4]==cap[0,4] and B[0,6]==cap[0,6]/2
    assert torch.equal(B[0],torch.tensor(candidate.B_TABLE['FR_air_approach_proxy']))
    for idx in (133,134,155):
        y=x.clone(); y[0,idx]=1
        assert candidate.candidate_physical_scales(y)[2]['weights']['FR_air_approach_proxy'].item()==0
    x[0,147]=0
    assert torch.equal(candidate.candidate_physical_scales(x)[0],cap)


def test_FL_pending_bidirectional_opportunity_and_contact_exit_not_a_negative_bias():
    x=obs(4); x[0,150]=1; x[0,21]=.026
    head=conditional(x); result,e=candidate.candidate_effective_head(head,x)
    assert e['B_full12'][0,0]==24 and e['sigma_multiplier_full12'][0,0]==pytest.approx(4/3)
    assert torch.equal(result[...,0,:],head[...,0,:])
    x[0,21]=.0015
    B,_,_=candidate.candidate_physical_scales(x)
    assert B[0,0]==21
    for idx in (131,132,154):
        y=x.clone(); y[0,idx]=1
        assert candidate.candidate_physical_scales(y)[2]['weights']['FL_crossed_air_pending'].item()==0


def rolling_obs():
    x=obs(5); x[0,154:156]=1; x[0,132]=x[0,134]=1
    x[0,124]=x[0,126]=.1; x[0,22]=x[0,25]=.1
    x[0,100]=x[0,103]=-.1; x[0,28]=x[0,31]=-.3
    return x


def test_P06_contact_proxy_and_physical_distance_blend_not_time_or_rear_mask():
    x=rolling_obs()
    B,cap,e=candidate.candidate_physical_scales(x)
    assert torch.equal(B[0],torch.tensor(candidate.B_TABLE['P06_front_support_rolling_proxy']))
    assert torch.equal(B[0,8:],cap[0,8:])
    x[0,28]=x[0,31]=-.245
    B,_,e=candidate.candidate_physical_scales(x)
    assert e['weights']['RR_preparation'].item()==pytest.approx(.5,abs=1e-6)
    assert B[0,0].item()==pytest.approx(18,abs=1e-5)
    y=x.clone();y[0,20]=2
    assert torch.equal(B,candidate.candidate_physical_scales(y)[0])
    x[0,132]=0
    assert candidate.candidate_physical_scales(x)[2]['weights']['P06_front_support_rolling_proxy'].item()==0
    assert candidate.candidate_physical_scales(x)[2]['weights']['P06_front_support_recovery_proxy'].item()==1


def test_P06_dropout_recovery_and_either_rear_near_release_linkage_without_mean_or_cap_change():
    x=rolling_obs();x[0,132]=0
    B,cap,e=candidate.candidate_physical_scales(x)
    assert torch.equal(B[0],torch.tensor(candidate.B_TABLE['P06_front_support_recovery_proxy']))
    assert B[0,0]==24 and B[0,4]==12 and bool((B>0).all())
    head=conditional(x);new,_=candidate.candidate_effective_head(head,x)
    assert torch.equal(new[...,0,:],head[...,0,:])
    for near in (28,31):
        y=x.clone();y[0,near]=-.22
        By,Cy,ey=candidate.candidate_physical_scales(y)
        assert torch.equal(Cy,cap)
        assert ey['weights']['RR_preparation'].item()==1
        assert torch.equal(By[0],torch.tensor(candidate.B_TABLE['RR_preparation']))
    y=x.clone();y[0,28]=y[0,31]=-.27
    assert candidate.candidate_physical_scales(y)[2]['weights']['RR_preparation'].item()==0
    y=x.clone();y[0,149]=1
    assert torch.equal(candidate.candidate_physical_scales(y)[0][0],torch.tensor(candidate.B_TABLE['RR_current_valid_lift']))


def test_current_RR_qualification_handoff_revocation_and_exit_after_RR_placed():
    x=obs(8);x[0,149]=1
    B,_,e=candidate.candidate_physical_scales(x)
    assert torch.equal(B[0],torch.tensor(candidate.B_TABLE['RR_current_valid_lift']))
    x[0,149]=0
    B,_,e=candidate.candidate_physical_scales(x)
    assert torch.equal(B[0],torch.tensor(candidate.B_TABLE['RR_preparation']))
    assert B[0,6]>0 and B[0,7]>0
    x[0,149]=1;x[0,157]=1
    B,cap,e=candidate.candidate_physical_scales(x)
    assert e['weights']['RR_current_valid_lift'].item()==0
    assert e['weights']['RR_preparation'].item()==0
    assert B[0,3]==24 and torch.equal(B[0,4:],cap[0,4:])


def test_batch_independence_and_one_gaussian_formula_entropy_likelihood():
    x=torch.cat((obs(1),obs(4),rolling_obs(),obs(8)),dim=0)
    x[1,150]=1;x[1,21]=.026;x[3,149]=1
    head=conditional(x);effective,e=candidate.candidate_effective_head(head,x)
    for i in range(4):
        one,_=candidate.candidate_effective_head(head[i:i+1],x[i:i+1])
        assert torch.equal(effective[i:i+1],one)
    mu=effective[...,0,:];sigma=effective[...,1,:].exp()
    distribution=torch.distributions.Normal(mu,sigma)
    before=torch.get_rng_state();sample=distribution.sample();after=torch.get_rng_state()
    torch.set_rng_state(before)
    same=torch.distributions.Normal(mu,sigma).sample()
    assert torch.equal(sample,same) and torch.equal(after,torch.get_rng_state())
    analytic=(-.5*((sample-mu)/sigma).square()-sigma.log()-.5*math.log(2*math.pi)).sum(-1)
    torch.testing.assert_close(distribution.log_prob(sample).sum(-1),analytic,atol=2e-6,rtol=1e-6)
    entropy=(sigma.log()+.5*(1+math.log(2*math.pi))).sum(-1)
    torch.testing.assert_close(distribution.entropy().sum(-1),entropy)
    assert torch.equal((distribution.log_prob(sample).sum(-1)-analytic).exp(),torch.ones(4)) or torch.allclose((distribution.log_prob(sample).sum(-1)-analytic).exp(),torch.ones(4),atol=2e-6)


def test_P05_to_P06_REQUEST_center_and_mean_not_old_raw_capacity_jump():
    x=rolling_obs();x[0,20]=0
    previous=torch.tensor(REQUEST_HISTORY_CAPS[4])*.15
    x[0,207:219]=previous/torch.tensor(REQUEST_HISTORY_SCALES)
    center,e=cap_transition_request_history(x)
    assert bool(e['gate_full12'][0,3])
    assert center[0,3].item()==pytest.approx(math.atanh(previous[3]/112),abs=1e-8)
    head=conditional(x);new,_=candidate.candidate_effective_head(head,x)
    assert torch.equal(head[...,0,:],new[...,0,:])


@pytest.mark.parametrize("kind",["nan","wrong_shape","bad_phase","bad_contact","zero_phase","negative_age"])
def test_invalid_observation_rejected(kind):
    x=obs(1)
    if kind=='nan':x[0,21]=float('nan')
    elif kind=='wrong_shape':x=x[:,:324]
    elif kind=='bad_phase':x[0,0]=.5
    elif kind=='bad_contact':x[0,132]=.5
    elif kind=='zero_phase':x[0,:13]=0
    elif kind=='negative_age':x[0,20]=-.1
    with pytest.raises(ValueError):candidate.candidate_physical_scales(x)
