"""CPU-only sealed-run diagnosis. Never optimize, run physics, or mutate old data."""
from __future__ import annotations
import json, math, sys
from pathlib import Path
import torch
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from wlr50_clean.ppo.semantic_history_actor import SemanticTemperedHistoryMLPModel

RUN = ROOT / 'runs/ppo_fsm_reference_p09_stable_v2/train/20260915T0633266376940Z_g4a58c0190ef7_d19a81607e6044a281a010de29469635'
HISTORY = ROOT / 'outputs/ppo_fsm_reference_p09_stable_v2/checkpoints/history'
OUT = Path(__file__).resolve().parent
GAMMA, LAM = .9985, .99

def actor(step):
    checkpoint = torch.load(HISTORY / f'checkpoint_step_{step:09d}.pt', map_location='cpu', weights_only=False)
    obj = SemanticTemperedHistoryMLPModel(TensorDict({'policy': torch.zeros(1, 372)}, batch_size=[1]),
        {'actor': ['policy']}, 'actor', 12, [256,256], 'elu', False,
        {'class_name':'HeteroscedasticGaussianDistribution','init_std':.15,'std_type':'log'},
        'diagonal_transfer_state_v1', .5)
    obj.load_state_dict(checkpoint['actor_state_dict'], strict=True)
    obj.eval()
    assert isinstance(obj.obs_normalizer, torch.nn.Identity)
    return obj

def distribution(obj, obs, actions):
    # Match production stochastic forward/logprob without drawing or consuming RNG.
    latent = obj.get_latent(TensorDict({'policy':obs}, batch_size=[len(obs)]))
    head = obj.mlp(latent)
    mean = .1 * head[:,0,:] + .9 * obs[:,195:207]
    std = (head[:,1,:] + math.log(.5)).exp()
    logp = torch.distributions.Normal(mean,std).log_prob(actions).sum(-1)
    return head[:,0,:], mean, std, logp

def stat(x):
    x = torch.as_tensor(x, dtype=torch.float64).reshape(-1)
    return {'n':len(x),'mean':float(x.mean()) if len(x) else None,
        'min':float(x.min()) if len(x) else None,'max':float(x.max()) if len(x) else None,
        'positive':int((x>0).sum()),'negative':int((x<0).sum()),'finite':bool(torch.isfinite(x).all())}

def main():
    torch.set_num_threads(2)
    snapshots = [torch.load(p,map_location='cpu',weights_only=False) for p in sorted((RUN/'rollouts').glob('*.pt'))]
    assert len(snapshots)==16
    tensors = {k:torch.cat([s[k] for s in snapshots]).squeeze(1) for k in
        ('actions','actions_log_prob','values','rewards','dones','returns','advantages')}
    obs = torch.cat([s['observations']['policy'] for s in snapshots]).squeeze(1)
    old_mean, old_std = [torch.cat([s['distribution_params'][i] for s in snapshots]).squeeze(1) for i in range(2)]
    acts, adv = tensors['actions'], tensors['advantages'].flatten()
    raw_gae = (tensors['returns']-tensors['values']).flatten()
    stored_lp = tensors['actions_log_prob'].flatten()
    from_stored_lp = torch.distributions.Normal(old_mean,old_std).log_prob(acts).sum(-1)
    checks = {'n':len(obs),'finite_observations':bool(torch.isfinite(obs).all()),
        'max_abs_scaled_observation':float(obs.abs().max()),'identity_normalizer':True,
        'stored_distribution_logp_max_abs_error':float((from_stored_lp-stored_lp).abs().max()),
        'all_sample_raw_matches_audit':True, 'history_max_abs_error':0.,'history_reset_zero_count':0,
        'normalization_max_abs_error':0.,'GAE_internal_recurrence_max_abs_error':0.,
        'terminal_return_reward_max_abs_error':0.,'tail_bootstrap_inferred_not_independently_measured':[]}
    for s in snapshots:
        gae = s['returns']-s['values']
        standardized = (gae-gae.mean())/(gae.std()+1e-8)
        checks['normalization_max_abs_error'] = max(checks['normalization_max_abs_error'],float((standardized-s['advantages']).abs().max()))
        lhs = gae[:-1]
        rhs = s['rewards'][:-1]-s['values'][:-1]+(1-s['dones'][:-1].float())*GAMMA*(s['values'][1:]+LAM*gae[1:])
        checks['GAE_internal_recurrence_max_abs_error']=max(checks['GAE_internal_recurrence_max_abs_error'],float((lhs-rhs).abs().max()))
        checks['tail_bootstrap_inferred_not_independently_measured'].append(float((s['returns'][-1]-s['rewards'][-1])/GAMMA))
    for i in range(len(obs)):
        reset = i==0 or bool(tensors['dones'][i-1])
        expected = torch.zeros(12) if reset else acts[i-1].clamp(-20,20)
        checks['history_max_abs_error']=max(checks['history_max_abs_error'],float((obs[i,195:207]-expected).abs().max()))
        if reset: checks['history_reset_zero_count']+=1
        if tensors['dones'][i]: checks['terminal_return_reward_max_abs_error']=max(checks['terminal_return_reward_max_abs_error'],abs(float(tensors['returns'][i]-tensors['rewards'][i])))
    models={step:actor(step) for step in (172544,172672,173824,174592)}
    frozen_checks=[]
    with torch.no_grad():
        for step,offset in ((172544,0),(172672,128),(173824,1280)):
            a=models[step]; o=obs[offset:offset+128]; ac=acts[offset:offset+128]
            base,mu,sd,lp=distribution(a,o,ac)
            permutation=torch.arange(127,-1,-1)
            shuffled=distribution(a,o[permutation],ac[permutation])
            a.train(); train=distribution(a,o,ac); a.eval()
            # Official forward performs a random sample but its output is discarded only here.
            with torch.random.fork_rng():
                a(TensorDict({'policy':o},batch_size=[128]), stochastic_output=True)
                official_lp=a.get_output_log_prob(ac)
            frozen_checks.append({'checkpoint':step,'rollout_update':1314+offset//128,
                'mu_max_abs_error':float((mu-old_mean[offset:offset+128]).abs().max()),
                'sigma_max_abs_error':float((sd-old_std[offset:offset+128]).abs().max()),
                'logp_max_abs_error':float((lp-stored_lp[offset:offset+128]).abs().max()),
                'ratio':stat((lp-stored_lp[offset:offset+128]).exp()),
                'shuffled_mu_max_error':float((shuffled[1]-mu[permutation]).abs().max()),
                'train_eval_logp_max_error':float((train[3]-lp).abs().max()),
                'official_forward_logp_max_error':float((official_lp-lp).abs().max())})
        before=distribution(models[172544],obs,acts)
        after=distribution(models[174592],obs,acts)
        first_after=distribution(models[172672],obs[:128],acts[:128])
    records=[]; episode=0; previous=None
    families={}; signed_score=adv[:,None]*(acts-old_mean)/(old_std**2)
    # These scores are local old-distribution derivatives, not executed optimizer changes.
    for i,line in enumerate((RUN/'residual_and_projection_audit.jsonl').open(encoding='utf-8')):
        row=json.loads(line); a=row['applied_audit']; p=a['semantic_task']['physical_evaluator']; fr=p['current_legs']['FR']; r=a['reward']; hist=p['history']
        checks['all_sample_raw_matches_audit'] &= acts[i].tolist()==row['raw_policy_action_full12']
        distance=fr['front_distance_m']; clearance=fr['clearance_m']
        dd=distance-previous['FR_front_distance_m'] if previous is not None else None
        dq=clearance-previous['FR_clearance_m'] if previous is not None else None
        cross_tick=hist['event_ticks']['front_edge_crossed'].get('FR'); place_tick=hist['event_ticks']['placed'].get('FR')
        start=a['physics_tick']-a['physics_ticks']
        phase=a['phase_id']; group=[]
        if phase=='P02':
            group.append('P02_all')
            if fr['air'] and clearance>=.015 and dd is not None and dd>.0001: group.append('P02_air_usable_forward_endpoint')
            if fr['air'] and clearance>=.015 and dd is not None and dd<=.0001: group.append('P02_air_usable_nonforward_endpoint')
            if clearance<.015: group.append('P02_below_15mm_endpoint')
        if cross_tick is not None and start<cross_tick<=a['physics_tick']: group.append('FR_cross_event')
        if place_tick is not None and start<place_tick<=a['physics_tick']: group.append('FR_place_event')
        if phase=='P05': group.append('P05_after_FR_capture')
        if row['terminal']: group.append('terminal')
        rec={'index':i,'global_decision':row['global_policy_decision'],'episode':episode,
            'rollout_update':1314+i//128,'row_in_rollout':i%128,'phase':phase,'end_phase':a['end_phase_id'],
            'physics_tick':a['physics_tick'],'groups':group,'FR_front_distance_m':distance,'FR_clearance_m':clearance,
            'FR_distance_delta_m':dd,'FR_clearance_delta_m':dq,'FR_air':fr['air'],'FR_ground_contact':fr['ground_contact'],
            'FR_bearing_force_n':fr['bearing_force_n'],'FR_cross_tick':cross_tick,'FR_place_tick':place_tick,
            'terminal':row['terminal'],'reason':a['termination_reason'],
            'reward':row['reward'],'families':r['families'],'potential_shaping':r['potential_shaping'],'event':r['terminal_event'],
            'old_value':float(tensors['values'][i]),'return':float(tensors['returns'][i]),'raw_GAE':float(raw_gae[i]),'update_advantage':float(adv[i]),
            'old_logp':float(stored_lp[i]),'post_block_same_obs_logp':float(after[3][i]),
            'post_block_same_obs_ratio':float((after[3][i]-stored_lp[i]).exp()),
            'post_block_surrogate_clipped_branch':bool((adv[i]>0 and after[3][i]-stored_lp[i]>math.log(1.2)) or (adv[i]<0 and after[3][i]-stored_lp[i]<math.log(.8))),
            'nominal':a['nominal_action_full12'],'effective_residual':a['projected_residual_full12'],'final_target':a['actual_drive_target_full12'],
            'measured_native_wheel_velocity_rad_s':p['measured_wheel_velocity_rad_s']}
        records.append(rec); previous=rec
        if row['terminal']: episode+=1; previous=None
    groups={}
    for group in sorted({g for r in records for g in r['groups']}):
        ids=[r['index'] for r in records if group in r['groups']]; ix=torch.tensor(ids)
        groups[group]={'n':len(ids),'raw_GAE':stat(raw_gae[ix]),'update_advantage':stat(adv[ix]),
            'reward':stat([records[i]['reward'] for i in ids]),
            'signed_family_sum':{k:sum(records[i]['families'][k] for i in ids) for k in records[0]['families']},
            'old_wheel_mean':old_mean[ix,8:].mean(0).tolist(),'sampled_wheel_raw_mean':acts[ix,8:].mean(0).tolist(),
            'effective_wheel_sigma_mean':old_std[ix,8:].mean(0).tolist(),
            'local_likelihood_score_wheel_mean':signed_score[ix,8:].mean(0).tolist(),
            'CP174592_minus_CP172544_base_wheel_mean_same_observations':(after[0][ix,8:]-before[0][ix,8:]).mean(0).tolist(),
            'CP174592_minus_CP172544_conditional_wheel_mean_same_observations':(after[1][ix,8:]-before[1][ix,8:]).mean(0).tolist(),
            'post_block_ratio_vs_collection':stat((after[3][ix]-stored_lp[ix]).exp())}
    # First update has an exact before-and-after checkpoint; no missing-weight attribution.
    first_update=[]
    for group in ['P02_all','P02_air_usable_forward_endpoint','P02_air_usable_nonforward_endpoint','P02_below_15mm_endpoint']:
        ids=[r['index'] for r in records[:128] if group in r['groups']]
        if not ids:continue
        ix=torch.tensor(ids); dl=first_after[3][ix]-stored_lp[ix]
        first_update.append({'group':group,'n':len(ids),'mean_advantage':float(adv[ix].mean()),
            'post_update_log_probability_change':stat(dl),'advantage_times_delta_logp_sum':float((adv[ix]*dl).sum()),
            'probability_change_same_direction_as_advantage':int((adv[ix]*dl>0).sum()),
            'wheel_mean_change':(first_after[1][ix,8:]-old_mean[ix,8:]).mean(0).tolist()})
    selected_ids={0,1,49,99,127,128,150,156,246,854,1473,1674,2047}
    selected_ids.update(r['index'] for r in records if any(g in r['groups'] for g in ('FR_cross_event','FR_place_event')))
    selected=[]
    for i in sorted(selected_ids):
        rec=dict(records[i]); rec.update({'observation372':obs[i].tolist(),'previous_raw_in_observation':obs[i,195:207].tolist(),
            'old_conditional_mean':old_mean[i].tolist(),'old_effective_sigma':old_std[i].tolist(),'sampled_latent':acts[i].tolist(),
            'base_mean_recovered_from_recorded_conditional_and_history':((old_mean[i]-.9*obs[i,195:207])/.1).tolist(),
            'CP174592_base_mean_same_obs':after[0][i].tolist(),'CP174592_conditional_mean_same_obs':after[1][i].tolist(),
            'CP174592_effective_sigma_same_obs':after[2][i].tolist(),
            'historical_optimization_minibatch_logp_ratio_branch':None,
            'missing_reason':'Only aggregate historical update logp/KL/clip was saved; post-block replay is explicitly not the historical minibatch value.'})
        selected.append(rec)
    result={'schema':'task_first_saved_learning_signal.v1','source_run':str(RUN),'checks':checks,
        'matched_frozen_actor_checks':frozen_checks,'groups':groups,'first_update_exact_checkpoint_comparison':first_update,
        'selected_key_samples':selected,'all_decisions':records,
        'limitations':['No old rollout optimized or mixed with new data.','No missing intermediate actor invented.',
            'Same-observation distribution replay is not a closed-loop counterfactual.',
            'Endpoint functional groups are descriptive, not universal reward or success predicates.',
            'GAE tail last values inferred algebraically; no new critic substituted for original.',
            'No per-minibatch historical logprob was stored; actual per-sample clip branches are unavailable.']}
    target=OUT/'saved_learning_signal_evidence.json'
    target.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print(json.dumps({'checks':checks,'frozen_checks':frozen_checks,'groups':groups,'first_update':first_update,'output':str(target)},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
