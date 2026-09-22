"""No-draw, no-gradient candidate sigma scale audit on 513 recorded inputs."""
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import sys

import torch
import torch.nn.functional as F

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT/'src'))
from wlr50_clean.ppo import semantic_history_actor as old_actor, semantic_policy_distribution as old_policy


def overlay(name):
    path=OUT/'candidate/src/wlr50_clean/ppo'/f'{name}.py'
    full='wlr50_clean.ppo.'+name
    spec=importlib.util.spec_from_file_location(full,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[full]=module
    spec.loader.exec_module(module)
    return module


p=overlay('semantic_policy_distribution')
a=overlay('semantic_history_actor')
RUN=ROOT/'runs/ppo_fl_capture_quality_v1/train/20260918T1037290906944Z_ge73542cb57ad_befe799ac0b0461ca2c4f9189be96966'
ENTRY=ROOT/'runs/ppo_task_conditioned_hip_wheel_v1/diagnostics/RR_prepare_RL_minus3_N_20260921_01/probe_entry.json'
CP=ROOT/'outputs/ppo_fl_capture_quality_v1/checkpoints/history/checkpoint_step_000185856.pt'
CHANNELS=('FL_hip','FL_knee','FR_hip','FR_knee','RL_hip','RL_knee','RR_hip','RR_knee','FL_wheel','FR_wheel','RL_wheel','RR_wheel')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(t):
    t=t.double()
    return dict(mean=t.mean().item(),min=t.min().item(),max=t.max().item())


def main():
    torch.set_num_threads(1)
    rng=torch.get_rng_state().clone()
    checkpoint=torch.load(CP,map_location='cpu',weights_only=False)
    manifest_path=CP.with_name(CP.stem+'_manifest.json')
    manifest=json.loads(manifest_path.read_text())
    assert sha(CP)==manifest['checkpoint_sha256']
    assert checkpoint['infos']['global_policy_decisions']==185856
    cfg=checkpoint['infos']['runner_config']['actor']
    assert cfg['activation']=='elu' and cfg['hidden_dims']==[256,256] and cfg['obs_normalization'] is False
    saved_paths=[RUN/f'rollouts/rollout_{i:06d}.pt' for i in range(1414,1418)]
    observations=torch.cat([torch.load(path,map_location='cpu',weights_only=False)['observations']['policy'] for path in saved_paths])[:,0]
    rows=[json.loads(line) for line in (RUN/'residual_and_projection_audit.jsonl').read_text().splitlines()]
    assert len(rows)==512 and observations.shape==(512,372)
    assert [r['global_policy_decision'] for r in rows]==list(range(185345,185857))
    entry=json.loads(ENTRY.read_text())
    assert len(entry['observation'])==372
    obs=torch.cat((observations,torch.tensor(entry['observation'],dtype=torch.float32).reshape(1,372)))
    assert obs[-1,:13].argmax().item()==7
    state=checkpoint['actor_state_dict']
    with torch.inference_mode():
        x=F.elu(F.linear(obs,state['mlp.0.weight'],state['mlp.0.bias']))
        x=F.elu(F.linear(x,state['mlp.2.weight'],state['mlp.2.bias']))
        head=F.linear(x,state['mlp.4.weight'],state['mlp.4.bias']).reshape(-1,2,12)
        old_history,old_evidence=old_actor.cap_transition_request_history(obs)
        history,evidence=a.cap_transition_request_history(obs)
        old_mean=old_actor.history_conditioned_head(head,old_history,.9)[:,0]
        mean=a.history_conditioned_head(head,history,.9)[:,0]
        old_log,_=old_actor.physical_innovation_effective_log_std(head[:,1],obs,.25)
        new_log,schedule=a.task_conditioned_effective_log_std(head[:,1],obs,.25)
        old_sigma,new_sigma=old_log.exp(),new_log.exp()
        B=schedule['physical_equivalent_B_full12'];cap=schedule['current_cap_full12']
        old_B=cap.clone();old_B[:,3]=torch.where(evidence['stage_index']>=5,24.,old_B[:,3])
        old_low=cap*torch.tanh(mean-old_sigma);old_high=cap*torch.tanh(mean+old_sigma)
        new_low=cap*torch.tanh(mean-new_sigma);new_high=cap*torch.tanh(mean+new_sigma)
        old_width=(old_high-old_low)/2;new_width=(new_high-new_low)/2
        old_neg=.5*(1+torch.erf(-mean/old_sigma/math.sqrt(2)))
        new_neg=.5*(1+torch.erf(-mean/new_sigma/math.sqrt(2)))
    assert torch.equal(old_history,history) and torch.equal(old_mean,mean)
    assert torch.equal(old_evidence['current_cap_full12'],cap)
    assert old_policy.REQUEST_HISTORY_CAPS==p.REQUEST_HISTORY_CAPS
    assert bool((old_sigma>0).all()) and bool((new_sigma>0).all()) and bool((B>0).all())
    assert bool(torch.isfinite(new_sigma).all()) and bool((new_width>0).all())
    assert torch.equal(rng,torch.get_rng_state())
    contract=p.policy_contract(p.TASK_CONDITIONED_HIP_WHEEL_POLICY,observation_layout='diagonal_transfer_state_v1')
    assert contract['physical_equivalent_B_full12']=={k:list(v) for k,v in p.TASK_CONDITIONED_PHYSICAL_B_TABLE.items()}
    reconstructed=old_B.clone()
    for name,weight in schedule['task_state_weights'].items():
        reconstructed=torch.lerp(reconstructed,obs.new_tensor(contract['physical_equivalent_B_full12'][name]),weight[:,None])
    assert torch.equal(B,reconstructed)
    # Group by the PRE-action real evaluator aligned to the saved observations,
    # not the current row's after-action contact. No new evaluator is run.
    physical=[None]+[r['applied_audit']['semantic_task']['physical_evaluator'] for r in rows[:-1]]+[entry['evaluator']]
    mismatch=0.
    for i in range(1,513):
        for offset,leg in ((21,'FL'),(24,'FR')):
            mismatch=max(mismatch,abs(float(obs[i,offset])-physical[i]['current_legs'][leg]['clearance_m']))
    assert mismatch<1e-6
    groups={key:[] for key in ('FR_current_AIR','P05_FL_current_AIR','P06_both_front_TOP_support','P06_FL_support_lost','P08_RR_A_entry')}
    for i in range(1,512):
        legs=physical[i]['current_legs'];phase=int(obs[i,:13].argmax())
        if phase<=1 and legs['FR']['air']:groups['FR_current_AIR'].append(i)
        if phase==4 and legs['FL']['air']:groups['P05_FL_current_AIR'].append(i)
        if phase==5:
            if all(legs[leg]['top_contact'] and legs[leg]['support'] for leg in ('FL','FR')):
                groups['P06_both_front_TOP_support'].append(i)
            if not (legs['FL']['top_contact'] and legs['FL']['support']):groups['P06_FL_support_lost'].append(i)
    groups['P08_RR_A_entry']=[512]
    summaries={}
    for name,indices in groups.items():
        assert indices
        idx=torch.tensor(indices)
        summaries[name]=dict(count=len(indices),block14_indices=[i for i in indices if i<512],
            state_weights={k:{**stats(v[idx]),'positive_count':int((v[idx]>0).sum()),'full_count':int((v[idx]==1).sum())} for k,v in schedule['task_state_weights'].items()},
            unique_B_full12=torch.unique(B[idx],dim=0).tolist(),
            rear_distance_m=stats(schedule['current_rear_front_distance_m'][idx]),
            RR_current_valid_count=int((obs[idx,149]==1).sum()),
            full12={channel:dict(physical_units='degree' if j<8 else 'rad/s',
                cap=stats(cap[idx,j]),B=stats(B[idx,j]), old_effective_raw_sigma=stats(old_sigma[idx,j]),new_effective_raw_sigma=stats(new_sigma[idx,j]),
                raw_sigma_ratio=stats(new_sigma[idx,j]/old_sigma[idx,j]),
                old_request_quantile_half_width=stats(old_width[idx,j]),new_request_quantile_half_width=stats(new_width[idx,j]),
                request_quantile_half_width_ratio=stats(new_width[idx,j]/old_width[idx,j]),
                probability_request_below_zero_old=stats(old_neg[idx,j]),probability_request_below_zero_new=stats(new_neg[idx,j]))
                for j,channel in enumerate(CHANNELS)})
    raw_records=[]
    for i in range(513):
        raw_records.append(dict(index=i, source='block14_saved_predecision_observation' if i<512 else 'RR_A_probe_entry_observation',
            global_decision=185345+i if i<512 else None,probe_tick=entry['tick'] if i==512 else None,
            phase=f'P{int(obs[i,:13].argmax())+1:02d}',groups=[k for k,v in groups.items() if i in v],
            conditional_mean_full12=mean[i].tolist(),old_effective_raw_sigma_full12=old_sigma[i].tolist(),new_effective_raw_sigma_full12=new_sigma[i].tolist(),
            B_full12=B[i].tolist(),cap_full12=cap[i].tolist(),
            old_request_q158655_full12=old_low[i].tolist(),old_request_q841345_full12=old_high[i].tolist(),
            new_request_q158655_full12=new_low[i].tolist(),new_request_q841345_full12=new_high[i].tolist(),
            old_request_quantile_half_width_full12=old_width[i].tolist(),new_request_quantile_half_width_full12=new_width[i].tolist(),
            task_state_weights={name:float(weight[i]) for name,weight in schedule['task_state_weights'].items()}))
    source_paths=[CP,manifest_path,ENTRY,RUN/'residual_and_projection_audit.jsonl',*saved_paths,
        OUT/'candidate/src/wlr50_clean/ppo/semantic_history_actor.py',OUT/'candidate/src/wlr50_clean/ppo/semantic_policy_distribution.py']
    result=dict(schema='wlr50_clean.candidate_real_observation_sigma_scale.v1',source_sha256={str(path):sha(path) for path in source_paths},
        checkpoint=str(CP),policy_contract=contract,grouping='Real pre-action evaluator from prior sealed block14 row aligned to current saved 372 input; RR-A uses its entry evaluator. Not post-action contact grouping.',
        old_policy='request_history_FR_knee_P06plus_physical_innovation_sigma_v1',candidate_policy=p.TASK_CONDITIONED_HIP_WHEEL_POLICY,
        checks=dict(all_513_full12_sigmas_positive_finite=True,caps_exact_unchanged=True,conditional_mean_exact_same_numeric_input=True,
            history_exact_same_numeric_input=True,B_float32_lerp_exact_contract=True,RNG_exact_unchanged=True,
            preaction_evaluator_observation_gap_max_roundoff_m=mismatch),
        interpretation='sigma and quantile ranges are CP185856 offline conditional calculations on exactly unchanged numeric 372 inputs. New rolling-retention MDP changes existing globalPhi observation semantics: this does not assert identical means for the same physical state or trajectories under the new MDP.',
        request_quantiles='cap*tanh(mu-sigma),cap*tanh(mu+sigma): 15.8655%–84.1345% marginal REQUEST quantiles; half-width=(upper-lower)/2. Not actuator measured std; excludes mapper/filter/rate/headroom/physical tracking. Intervals may be asymmetric around tanh(mu).',
        full12_order=CHANNELS,summaries=summaries,rows=raw_records,
        RR_B_included=False,RR_B_not_waited_for=True,real_training_added=dict(decisions=0,updates=0,optimizer_steps=0),
        no_sampling_optimizer_gradients_actor_construction_or_simulator=True)
    dest=OUT/'candidate_real_observation_sigma_scale.json'
    with dest.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(dict(report=str(dest),checks=result['checks']),indent=2))
    for name,g in summaries.items():
        print('\n',name,g['count'],'weights', {k:(v['min'],v['max'],v['positive_count'],v['full_count']) for k,v in g['state_weights'].items() if v['max']>0},'B_unique',g['unique_B_full12'])
        for ch,v in g['full12'].items():
            print(ch,'sigma',round(v['old_effective_raw_sigma']['mean'],6),round(v['new_effective_raw_sigma']['mean'],6),
                'sigma_ratio_range',round(v['raw_sigma_ratio']['min'],6),round(v['raw_sigma_ratio']['max'],6),
                'width',round(v['old_request_quantile_half_width']['mean'],6),round(v['new_request_quantile_half_width']['mean'],6))


if __name__=='__main__':main()
