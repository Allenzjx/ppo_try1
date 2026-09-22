"""Bounded, offline audit of sealed block14; no actor sampling or optimizer.

Only the 512 recorded decisions, four rollouts/hooks and two endpoint checkpoints
are read. Functional ELU/linear CPU forwards use saved weights and saved inputs;
no model construction, RNG draw, gradient, simulated state or policy data write.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import hashlib
import json
import math
import sys

import torch
import torch.nn.functional as F

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from wlr50_clean.ppo.semantic_history_actor import (
    cap_transition_request_history, physical_innovation_effective_log_std,
)

RUN = ROOT / 'runs/ppo_fl_capture_quality_v1/train/20260918T1037290906944Z_ge73542cb57ad_befe799ac0b0461ca2c4f9189be96966'
CPROOT = ROOT / 'outputs/ppo_fl_capture_quality_v1/checkpoints/history'
CHANNELS = ('FL_hip','FL_knee','FR_hip','FR_knee','RL_hip','RL_knee','RR_hip','RR_knee','FL_wheel','FR_wheel','RL_wheel','RR_wheel')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]


def stats(x):
    x = torch.as_tensor(x, dtype=torch.float64).flatten()
    return dict(n=x.numel(), mean=x.mean().item(), min=x.min().item(), max=x.max().item(),
                rms=x.square().mean().sqrt().item(), mean_abs=x.abs().mean().item(),
                positive=int((x>0).sum()), negative=int((x<0).sum()))


def terms(action, old_mu, old_sd, new_mu, new_sd):
    """Exact algebra, not gradient attribution; mean-first decomposition."""
    lp = lambda mu, sd: -sd.log()-.5*((action-mu)/sd).square()
    mean = lp(new_mu,old_sd)-lp(old_mu,old_sd)
    scale = lp(new_mu,new_sd)-lp(new_mu,old_sd)
    kl_mean = .5*((new_mu-old_mu)/new_sd).square()
    kl_scale = (new_sd/old_sd).log()+.5*(old_sd/new_sd).square()-.5
    return mean, scale, kl_mean, kl_scale


def endpoint_forward(state, obs, recorded_history):
    # Saved manifest verifies identity normalizer, ELU [256,256], 24 outputs.
    x = F.elu(F.linear(obs, state['mlp.0.weight'], state['mlp.0.bias']))
    x = F.elu(F.linear(x, state['mlp.2.weight'], state['mlp.2.bias']))
    head = F.linear(x, state['mlp.4.weight'], state['mlp.4.bias']).reshape(-1,2,12)
    _, evidence = cap_transition_request_history(obs)
    # Keep the actual logged GPU HISTORY center identical for both endpoints.
    # CPU atanh at enlarged-cap entries differs by <= 1.5e-8 in five scalars.
    history = recorded_history
    mean = .1*head[:,0]+.9*history
    log_sd, _ = physical_innovation_effective_log_std(head[:,1], obs, .25)
    return dict(network=head[:,0], history=history, mean=mean, sigma=log_sd.exp(),
                learned_sigma=head[:,1].exp(), cap=evidence['current_cap_full12'])


def main():
    torch.set_num_threads(1)
    rng = torch.get_rng_state().clone()
    source_paths = [RUN/'residual_and_projection_audit.jsonl', RUN/'optimizer_updates.jsonl', RUN/'advantage_audit.jsonl']
    rows, updates, advantage_audit = map(lines, source_paths)
    assert len(rows)==512 and len(updates)==4
    assert [r['global_policy_decision'] for r in rows]==list(range(185345,185857))
    assert [u['ppo_update'] for u in updates]==list(range(1414,1418))
    checkpoints = []
    for step in (185344,185856):
        path = CPROOT/f'checkpoint_step_{step:09d}.pt'
        manifest_path = path.with_name(path.stem+'_manifest.json')
        cp = torch.load(path,map_location='cpu',weights_only=False)
        manifest = json.loads(manifest_path.read_text())
        assert sha(path)==manifest['checkpoint_sha256']
        assert cp['infos']['global_policy_decisions']==step
        assert cp['infos']['runner_config']['actor']['activation']=='elu'
        assert cp['infos']['runner_config']['actor']['hidden_dims']==[256,256]
        assert cp['infos']['runner_config']['actor']['obs_normalization'] is False
        assert set(cp['actor_state_dict'])=={f'mlp.{j}.{name}' for j in (0,2,4) for name in ('weight','bias')}
        checkpoints.append(cp)
        source_paths += [path,manifest_path]
    assert checkpoints[0]['infos']['actor_parameter_sha256']==updates[0]['actor_parameter_sha256_before']
    assert checkpoints[1]['infos']['actor_parameter_sha256']==updates[-1]['actor_parameter_sha256_after']
    assert all(a['actor_parameter_sha256_after']==b['actor_parameter_sha256_before'] for a,b in zip(updates,updates[1:]))
    assert checkpoints[1]['infos']['ppo_updates']-checkpoints[0]['infos']['ppo_updates']==4
    assert checkpoints[1]['infos']['optimizer_steps']-checkpoints[0]['infos']['optimizer_steps']==80
    saved_parts=[]; actual_uses={i:[] for i in range(512)}; checks=[]; recurrence=[]
    for block, update in enumerate(range(1414,1418)):
        p=RUN/f'rollouts/rollout_{update:06d}.pt'; hp=RUN/f'rollouts/update_{update:06d}_likelihood.json'
        source_paths += [p,hp]
        s=torch.load(p,map_location='cpu',weights_only=False); saved_parts.append(s)
        h=json.loads(hp.read_text()); rs=rows[block*128:(block+1)*128]
        raw=s['returns']-s['values']; norm=(raw-raw.mean())/(raw.std()+1e-8)
        ck=dict(update=update,
            raw_exact=torch.equal(torch.tensor([r['raw_policy_action_full12'] for r in rs]),s['actions'][:,0]),
            reward_exact=torch.equal(torch.tensor([r['reward'] for r in rs]),s['rewards'][:,0,0]),
            mean_exact=torch.equal(torch.tensor([r['old_distribution_mean_full12'] for r in rs]),s['distribution_params'][0][:,0]),
            sigma_exact=torch.equal(torch.tensor([r['old_distribution_std_full12'] for r in rs]),s['distribution_params'][1][:,0]),
            logp_exact=torch.equal(torch.tensor([r['old_log_probability'] for r in rs]),s['actions_log_prob'][:,0,0]),
            no_done=not bool(s['dones'].any()) and not any(r['terminal'] for r in rs),
            normalized_A_max_error=float((norm-s['advantages']).abs().max()))
        assert all(ck[k] for k in ('raw_exact','reward_exact','mean_exact','sigma_exact','logp_exact','no_done'))
        assert ck['normalized_A_max_error'] < 1e-6
        for i in range(127):
            expected=s['rewards'][i]+.9985*s['values'][i+1]-s['values'][i]+.9985*.99*raw[i+1]
            recurrence.append(float((raw[i]-expected).abs().max()))
        assert len(h['minibatches'])==20 and h['extra_model_forwards']==0 and h['extra_random_draws']==0
        lp_errors=[]
        for mb in h['minibatches']:
            for j,indices in enumerate(mb['rollout_flat_indices']):
                assert len(indices)==1
                i=indices[0]; gi=block*128+i
                assert mb['old_log_probability'][j]==s['actions_log_prob'][i].item()
                assert mb['actual_advantage'][j]==s['advantages'][i].item()
                om=s['distribution_params'][0][i,0].double(); os=s['distribution_params'][1][i,0].double()
                nm=torch.tensor(mb['current_conditional_mean'][j],dtype=torch.float64)
                ns=torch.tensor(mb['current_conditional_sigma'][j],dtype=torch.float64)
                action=s['actions'][i,0].double(); mt,st,km,ks=terms(action,om,os,nm,ns)
                delta_lp=mb['optimization_log_probability'][j]-mb['old_log_probability'][j]
                lp_errors.append(abs((mt+st).sum().item()-delta_lp))
                strict=(mb['actual_advantage'][j]>0 and mb['ratio'][j]>1+h['clip_param']) or (mb['actual_advantage'][j]<0 and mb['ratio'][j]<1-h['clip_param'])
                assert strict==mb['clipped_branch_strictly_active'][j]
                actual_uses[gi].append(dict(minibatch=mb['minibatch_index'], conditional_mean_full12=nm.tolist(),
                    effective_sigma_full12=ns.tolist(), normalized_A=mb['actual_advantage'][j],
                    ratio=mb['ratio'][j], strict_clip=strict, joint_delta_logp=delta_lp,
                    mean_first_channel_delta_logp_full12=mt.tolist(), sigma_second_channel_delta_logp_full12=st.tolist(),
                    analytic_KL_mean_full12=km.tolist(),analytic_KL_sigma_full12=ks.tolist()))
        ck['hook_joint_logp_max_float_roundoff']=max(lp_errors)
        assert max(lp_errors)<3e-5
        ck['all_samples_used_five_times']=all(len(actual_uses[block*128+i])==5 for i in range(128))
        assert ck['all_samples_used_five_times']
        checks.append(ck)
    cat=lambda key: torch.cat([s[key] for s in saved_parts])[:,0]
    obs=torch.cat([s['observations']['policy'] for s in saved_parts])[:,0]
    assert obs.shape==(512,372)
    raw_gae=(cat('returns')-cat('values')).flatten(); advantage=cat('advantages').flatten()
    old_mean=torch.cat([s['distribution_params'][0] for s in saved_parts])[:,0]
    old_sigma=torch.cat([s['distribution_params'][1] for s in saved_parts])[:,0]
    old_base=torch.tensor([r['policy_request']['base_mean_full12'] for r in rows])
    history=torch.tensor([r['policy_request']['history_center_full12'] for r in rows])
    assert torch.equal(.1*old_base+.9*history,old_mean)
    cpu_history, history_evidence = cap_transition_request_history(obs)
    history_error=float((cpu_history-history).abs().max())
    assert history_error<3e-8
    assert not bool(((cpu_history!=history)&~history_evidence['gate_full12']).any())
    with torch.inference_mode():
        first,last=[endpoint_forward(cp['actor_state_dict'],obs,history) for cp in checkpoints]
    first_error=float((first['network'][:128]-old_base[:128]).abs().max())
    assert first_error<1e-5
    assert torch.equal(rng,torch.get_rng_state())
    actions=cat('actions').double()
    mt,st,km,ks=terms(actions,first['mean'].double(),first['sigma'].double(),last['mean'].double(),last['sigma'].double())
    endpoint_channels=[]
    masks={phase:torch.tensor([r['applied_audit']['phase_id']==phase for r in rows]) for phase in ('P05','P06')}
    masks['P05_P06']=masks['P05']|masks['P06']
    endpoint_summary={}
    hook_summary={}
    last_mu=torch.tensor([actual_uses[i][-1]['conditional_mean_full12'] for i in range(512)])
    last_sd=torch.tensor([actual_uses[i][-1]['effective_sigma_full12'] for i in range(512)])
    for phase,mask in masks.items():
        endpoint_summary[phase]={}
        hook_summary[phase]={}
        for j,ch in enumerate(CHANNELS):
            endpoint_summary[phase][ch]=dict(
                source_network_mean=stats(first['network'][mask,j]), target_network_mean=stats(last['network'][mask,j]),
                network_mean_delta=stats(last['network'][mask,j]-first['network'][mask,j]),
                conditional_mean_delta=stats(last['mean'][mask,j]-first['mean'][mask,j]),
                conditional_delta_in_source_sigma=stats((last['mean'][mask,j]-first['mean'][mask,j])/first['sigma'][mask,j]),
                effective_sigma_ratio=stats(last['sigma'][mask,j]/first['sigma'][mask,j]),
                physical_tanh_mean_request_delta=stats(last['cap'][mask,j]*(last['mean'][mask,j].tanh()-first['mean'][mask,j].tanh())),
                mean_first_logp_delta=stats(mt[mask,j]),sigma_second_logp_delta=stats(st[mask,j]),
                analytic_KL_mean=stats(km[mask,j]),analytic_KL_sigma=stats(ks[mask,j]))
            hook_summary[phase][ch]=dict(conditional_delta=stats(last_mu[mask,j]-old_mean[mask,j]),
                effective_sigma_ratio=stats(last_sd[mask,j]/old_sigma[mask,j]),
                analytic_KL_mean=stats([actual_uses[i][-1]['analytic_KL_mean_full12'][j] for i in range(512) if mask[i]]),
                analytic_KL_sigma=stats([actual_uses[i][-1]['analytic_KL_sigma_full12'][j] for i in range(512) if mask[i]]))
    capture=[i for i,r in enumerate(rows) if r['applied_audit']['phase_id']=='P05' and r['applied_audit']['end_phase_id']=='P06']
    assert capture==[358]
    selected_indices=[211,255,256,340,350,355,356,357,358,359,360,361,362,365,375,383,384,450,511]
    selected=[]
    for i in selected_indices:
        row=rows[i]; a=row['applied_audit']; q=row['policy_request']; fl=a['semantic_task']['physical_evaluator']['current_legs']['FL']
        effect=a['actuator_target_effect_audit']; hr=effect['policy_headroom_evidence']
        selected.append(dict(index=i, global_decision=row['global_policy_decision'], update=1414+i//128,
            rollout_index=i%128, phase=a['phase_id'],end_phase=a['end_phase_id'],episode_tick=a['physics_tick'],
            dispatch_tick=effect['physics_tick'],sim_time_s=a['sim_time_s'],
            FL={k:fl[k] for k in ('clearance_m','front_distance_m','air','top_contact','support','contact_surface','bearing_force_n','load_fraction')},
            historical_FL_placed=a['semantic_task']['physical_evaluator']['history']['placed']['FL'],
            reward=row['reward'],reward_families=a['reward_breakdown']['families'],
            reward_potential_before=a['reward_breakdown']['potential_before'],reward_potential_after=a['reward_breakdown']['potential_after'],
            raw_GAE=float(raw_gae[i]),normalized_A=float(advantage[i]),done=row['terminal'],
            actual_collection={k:q[k] for k in ('base_mean_full12','history_center_full12','conditional_mean_full12',
                'learned_sigma_full12','effective_sigma_full12','selected_raw_full12','current_cap_full12','cap_transition_gate_full12')},
            actual_physical=dict(requested_full12=hr['requested_policy_residual_full12'], effective_headroom_full12=hr['effective_policy_residual_full12'],
                nominal_full12=a['nominal_action_full12'], baseline_native_plus_controller_full12=hr['baseline_native_plus_controller_full12'],
                final_target_full12=a['actual_drive_target_full12'],phase_mask_full12=effect['phase_mask_full12'],
                canonical_units='first8 degrees; last4 rad/s; not all equal actual joint motion'),
            actual_optimizer_uses=actual_uses[i],
            endpoint_same_observation=dict(source_network_mean_full12=first['network'][i].tolist(),target_network_mean_full12=last['network'][i].tolist(),
                source_conditional_mean_full12=first['mean'][i].tolist(),target_conditional_mean_full12=last['mean'][i].tolist(),
                source_effective_sigma_full12=first['sigma'][i].tolist(),target_effective_sigma_full12=last['sigma'][i].tolist(),
                source_tanh_mean_request_full12=(first['cap'][i]*first['mean'][i].tanh()).tolist(),
                target_tanh_mean_request_full12=(last['cap'][i]*last['mean'][i].tanh()).tolist(),
                mean_first_channel_delta_logp_full12=mt[i].tolist(),sigma_second_channel_delta_logp_full12=st[i].tolist())))
    support_runs=[]; start=358; prev=None
    for i in range(358,512):
        fl=rows[i]['applied_audit']['semantic_task']['physical_evaluator']['current_legs']['FL']
        state=(fl['top_contact'],fl['support'])
        if prev is not None and state!=prev:
            support_runs.append(dict(first=start,last=i-1,top_contact=prev[0],support=prev[1]))
            start=i
        prev=state
    support_runs.append(dict(first=start,last=511,top_contact=prev[0],support=prev[1]))
    report=dict(schema='wlr50_clean.block14_mean_sigma_readonly.v1',run=str(RUN), source_sha256={str(p):sha(p) for p in source_paths},
        training_credit_added=dict(policy_decisions=0,PPO_updates=0,optimizer_steps=0),
        old_completed_training=dict(source_decisions=185344,target_decisions=185856,updates=[1414,1415,1416,1417],optimizer_steps=80),
        phase_samples=dict(Counter(r['applied_audit']['phase_id'] for r in rows)),checks=checks,
        max_interior_GAE_recurrence_error=max(recurrence),first_checkpoint_CPU_vs_actual_GPU_base_mean_max_error=first_error,
        CPU_vs_actual_GPU_history_center_max_error=history_error,
        endpoint_HISTORY_uses_actual_logged_GPU_center=True,
        RNG_unchanged=True,never_instantiated_actor_or_optimizer=True,all_512_no_done=True,
        endpoint_forward_scope='Same 512 saved real observations including their identical saved HISTORY; hypothetical endpoint conditional distributions, not natural deterministic rollouts. No draw, no gradient, no state/cache mutation.',
        likelihood_scope='Actual recorded hooks before each minibatch update; last use is not final post-update checkpoint. Mean-first then sigma-second logp split is algebraic and order-dependent, not causal gradient credit. Analytic KL(old||new) separates nonnegative mean and sigma terms but does not say which action improved physics.',
        endpoint_channel_summary=endpoint_summary,last_actual_hook_channel_summary=hook_summary,
        actual_updates=updates,actual_advantage_audits=advantage_audit,FL_support_segments_from_capture=support_runs,
        selected=selected,channel_order=CHANNELS,rho=.9,
        rho_interpretation=dict(base_mean_chain_rule_factor=.1,
            formula='d log pi(a|o)/d base_mu = 0.1*(a-mu_cond)/sigma_eff^2 with current observation/history held fixed',
            limitation='The 0.1 factor is not a measured 10x wall-clock slowdown or Adam learning-rate equivalence. Endogenous history/observations, sigma/clip/shared trunk and task credit all matter. This audit cannot identify changing rho as a remedy.'),
        no_source_or_production_modifications=True)
    out=OUT/'block14_mean_sigma_readonly.json'
    with out.open('x',encoding='utf-8') as f:
        json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps(dict(report=str(out),checks=checks,GAE_max_error=max(recurrence),CPU_GPU_base_error=first_error,
        phase_samples=report['phase_samples'],support_segments=support_runs),indent=2))
    for phase in ('P05','P06'):
        print(phase,'endpoint: channel delta_network delta_cond sigma_ratio KLmean KLsigma')
        for ch,c in endpoint_summary[phase].items():
            print(ch,*(round(c[k]['mean'],6) for k in ('network_mean_delta','conditional_mean_delta','effective_sigma_ratio','analytic_KL_mean','analytic_KL_sigma')))
    for s in selected:
        print('sample',s['index'],s['global_decision'],s['phase'],s['FL']['support'],s['reward'],s['raw_GAE'],s['normalized_A'],s['actual_optimizer_uses'][-1]['ratio'])


if __name__=='__main__':
    main()
