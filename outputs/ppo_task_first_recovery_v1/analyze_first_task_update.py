"""Bounded CPU-only read of one sealed epsilon=0 update; no rollout optimization."""
from pathlib import Path
import importlib.util,json,math
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('existing_learning_diagnostic',HERE/'analyze_saved_learning_signal.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
RUN=ROOT/'runs/ppo_task_first_recovery_v1/train/20260916T0342024652780Z_gb0438f66ec63_a11da4badf6f4422b0af0decfc19053c'

def main():
    torch.set_num_threads(2)
    s=torch.load(RUN/'rollouts/rollout_001330.pt',map_location='cpu',weights_only=False)
    likelihood=json.loads((RUN/'rollouts/update_001330_likelihood.json').read_text())
    observation=s['observations']['policy'].squeeze(1); action=s['actions'].squeeze(1)
    old_mu,old_sigma=(x.squeeze(1) for x in s['distribution_params'])
    adv=s['advantages'].flatten();raw=(s['returns']-s['values']).flatten();old_lp=s['actions_log_prob'].flatten()
    before=helper.actor(174592)
    helper.HISTORY=HERE/'checkpoints/history';after=helper.actor(174720)
    with torch.no_grad():
        a=helper.distribution(before,observation,action); b=helper.distribution(after,observation,action)
    layout=json.loads((ROOT/'configs/ppo_task_first_recovery_v1/observation_schema.json').read_text())
    slices={};offset=0
    for group in layout['feature_groups']:
        slices[group['name']]=(offset,offset+group['size']);offset+=group['size']
    geometry=observation[:,slice(*slices['wheel_geometry_relative_obstacle'])]
    contact=observation[:,slice(*slices['wheel_pair_active'])]
    phases=[f'P{i+1:02}' for i in observation[:,:13].argmax(-1).tolist()]
    groups={'all':list(range(128)),'P01':[i for i,p in enumerate(phases) if p=='P01'],
        'P02':[i for i,p in enumerate(phases) if p=='P02'],'P02_result_air_clear_forward':[],
        'P02_result_air_clear_nonforward':[]}
    # Snapshot i+1 is the resulting state for action i, except the unavailable tail.
    for i in range(127):
        if phases[i]=='P02' and geometry[i+1,5]>=.015 and not bool(contact[i+1,2:4].any()):
            key='P02_result_air_clear_forward' if geometry[i+1,3]-geometry[i,3]>.0001 else 'P02_result_air_clear_nonforward'
            groups[key].append(i)
    visits=[[] for _ in range(128)];errors={'advantage':0.,'old_logp':0.,'likelihood_recompute':0.,'ratio_recompute':0.}
    all_rows=[]
    for batch in likelihood['minibatches']:
        mu=torch.tensor(batch['current_conditional_mean']);sigma=torch.tensor(batch['current_conditional_sigma'])
        for j,candidates in enumerate(batch['rollout_flat_indices']):
            assert len(candidates)==1
            i=candidates[0];ad=batch['actual_advantage'][j];lp=batch['optimization_log_probability'][j];old=batch['old_log_probability'][j];ratio=batch['ratio'][j]
            recomputed=float(torch.distributions.Normal(mu[j],sigma[j]).log_prob(action[i]).sum())
            errors['advantage']=max(errors['advantage'],abs(ad-float(adv[i])))
            errors['old_logp']=max(errors['old_logp'],abs(old-float(old_lp[i])))
            errors['likelihood_recompute']=max(errors['likelihood_recompute'],abs(recomputed-lp))
            errors['ratio_recompute']=max(errors['ratio_recompute'],abs(math.exp(lp-old)-ratio))
            clipped=(-ad*min(1.2,max(.8,ratio)) > -ad*ratio)
            assert clipped==batch['clipped_branch_strictly_active'][j]
            row={'row_index':i,'minibatch_index':batch['minibatch_index'],'old_logp':old,'actual_optimization_logp':lp,
                'ratio':ratio,'advantage':ad,'advantage_times_logp_change':ad*(lp-old),
                'strict_clipped_branch':clipped,'conditional_wheel_mean':mu[j,8:].tolist()}
            visits[i].append(row);all_rows.append(row)
    assert all(len(v)==5 for v in visits)
    stats={};final_delta=b[3]-old_lp
    for name,ids in groups.items():
        ix=torch.tensor(ids,dtype=torch.long)
        rows=[row for row in all_rows if row['row_index'] in ids]
        stats[name]={'samples':len(ids),'raw_GAE':helper.stat(raw[ix]),'update_advantage':helper.stat(adv[ix]),
            'rewards':helper.stat(s['rewards'].flatten()[ix]),'old_values':helper.stat(s['values'].flatten()[ix]),
            'stored_old_conditional_wheel_mean':old_mu[ix,8:].mean(0).tolist(),
            'new_minus_old_base_wheel_mean_same_obs':(b[0][ix,8:]-a[0][ix,8:]).mean(0).tolist(),
            'new_minus_old_conditional_wheel_mean_same_obs':(b[1][ix,8:]-a[1][ix,8:]).mean(0).tolist(),
            'final_CP_advantage_times_delta_logp_sum':float((adv[ix]*final_delta[ix]).sum()),
            'final_CP_change_aligned_with_advantage_count':int((adv[ix]*final_delta[ix]>0).sum()),
            'final_CP_ratio':helper.stat(final_delta[ix].exp()),
            'actual_optimizer_visits':len(rows),'actual_strict_clip_visits':sum(row['strict_clipped_branch'] for row in rows),
            'actual_outside_ratio_band_visits':sum(abs(row['ratio']-1)>.2 for row in rows),
            'actual_advantage_times_delta_logp_sum_over_visits':sum(row['advantage_times_logp_change'] for row in rows)}
    result={'schema':'task_first_recovery_first_sealed_update_learning.v1','source_run':str(RUN),
        'scope':'Only saved rollout001330, actual update001330 likelihood and immutable CP174592/174720; no active decision audit scan',
        'reward_profile':'task_first_recovery_epsilon_zero_v1','before_CP':174592,'after_CP':174720,'new_decisions':128,'new_updates':1,
        'preupdate_frozen_actor_logp_error':float((a[3]-old_lp).abs().max()),'actual_likelihood_errors':errors,
        'first_minibatch_ratio':helper.stat(likelihood['minibatches'][0]['ratio']),
        'sample_visit_counts':[len(v) for v in visits],'stats':stats,'actual_optimizer_visits':all_rows,
        'terminal_count':int(s['dones'].sum()),'tail_value_inferred_not_new_critic':float((s['returns'][-1]-s['rewards'][-1])/.9985),
        'ordinary_phase_transition_is_nonterminal':bool(not s['dones'][1]),
        'limitations':['No optimizer run or new environment data generated.','This first block has no FR capture or terminal outcome.',
            'Endpoint groups use stored next observation; final transition excluded because next observation was not stored.',
            'Same-observation distribution change does not establish deterministic closed-loop recovery.',
            'Different checkpoint/state trajectories from old-objective first update: descriptive comparison, not reward-only causal replay.',
            'Actual optimizer visits are before each optimizer step; final checkpoint replay is labelled separately.']}
    (HERE/'first_epsilon0_update_learning_signal.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['actual_optimizer_visits','sample_visit_counts']},indent=2,ensure_ascii=False))

if __name__=='__main__':main()
