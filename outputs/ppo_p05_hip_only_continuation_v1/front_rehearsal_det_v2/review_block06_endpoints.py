"""Bounded read-only CPU audit: two rollout/update files, two checkpoint actors.

Writes nothing, never constructs an optimizer, and never fits a model. The
production decision stream is scanned as bytes but only the selected 257 rows
are decoded; no other blocks or intermediate rollout actors are evaluated.
"""
from pathlib import Path
import json
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import torch
from tensordict import TensorDict
from torch.func import functional_call
from wlr50_clean.ppo.semantic_p05_capture_actor import SemanticP05CaptureHistoryMLPModel
from wlr50_clean.ppo.semantic_p05_capture_profile import P05_CAPTURE_OBSERVATION_LAYOUT

OUT = ROOT / 'outputs/ppo_p05_hip_only_continuation_v1'
AUDIT = json.loads((OUT/'block06_training_audit.json').read_text())
RUN = Path(AUDIT['run'])
CHANNELS = {'FL_knee': 1, 'FR_knee': 3, 'FL_wheel': 8, 'RL_hip': 4}


def arr(x):
    return x.detach().cpu().numpy().astype(float) if isinstance(x, torch.Tensor) else np.asarray(x, float)


def compact(values):
    values = arr(values)
    return {'n': int(values.size), 'min': float(values.min()), 'max': float(values.max()),
            'mean': float(values.mean()), 'positive': int((values > 0).sum())} if values.size else {'n': 0}


def tensor(x):
    return TensorDict({'policy': x, 'critic': x.clone()}, batch_size=[len(x)])


def model(payload):
    # Exact current model, no runner/Adam construction and no sampling.
    obs = tensor(torch.zeros(1, 389))
    actor = SemanticP05CaptureHistoryMLPModel(obs, {'actor': ['policy'], 'critic': ['critic']}, 'actor', 12,
        hidden_dims=(256,256), activation='elu', obs_normalization=False,
        distribution_cfg={'class_name': 'HeteroscedasticGaussianDistribution', 'std_type': 'log',
                          'init_std': .15}, observation_layout=P05_CAPTURE_OBSERVATION_LAYOUT,
        exploration_std_temperature=.25)
    actor.load_state_dict(payload['actor_state_dict'], strict=True)
    actor.eval()
    return actor


def mean(actor, x, state=None):
    with torch.no_grad():
        return actor(tensor(x), stochastic_output=False) if state is None else functional_call(
            actor, state, (tensor(x),), {'stochastic_output': False})


def paired(actor0, actor1, x, target=None):
    a, b = arr(mean(actor0,x)), arr(mean(actor1,x)); result = {}
    for name, index in CHANNELS.items():
        result[name] = {'source': compact(a[:,index]), 'end': compact(b[:,index]),
                        'end_minus_source': compact(b[:,index]-a[:,index])}
        if target is not None:
            result[name]['source_MAE_to_actual_historical_raw'] = float(np.abs(a[:,index]-target[:,index]).mean())
            result[name]['end_MAE_to_actual_historical_raw'] = float(np.abs(b[:,index]-target[:,index]).mean())
    return result


def scan_selected():
    selected = {}
    # Fixed rows1..128 and1920..2048: row1920 supplies predecessor of last batch.
    with (RUN/'residual_and_projection_audit.jsonl').open() as stream:
        for offset, line in enumerate(stream):
            if offset < 128 or offset >= 1919:
                row = json.loads(line)
                assert row['global_policy_decision'] == 209921 + offset
                selected[row['global_policy_decision']] = row
    assert len(selected) == 257
    return selected


def discounted_components(r, rows):
    n=128; gamma=.9985; lam=.99
    reward=arr(r['rewards']).reshape(n); value=arr(r['values']).reshape(n)
    raw=arr(r['returns']).reshape(n)-value; done=arr(r['dones']).reshape(n).astype(bool)
    next_value=np.r_[value[1:], 0. if done[-1] else (raw[-1]-reward[-1]+value[-1])/gamma]
    fields={key:np.zeros(n) for key in ['potential_shaping','terminal_event','time_cost','body_quality','geometry_quality','other_quality','critic_TD']}
    for i,row in enumerate(rows):
        q=row['applied_audit']['reward']; fs=q['families']; c=q['cost_components']
        fields['potential_shaping'][i]=q['potential_shaping']; fields['terminal_event'][i]=q['terminal_event']
        fields['time_cost'][i]=fs['task_progress']-q['potential_shaping']-q['terminal_event']
        fields['body_quality'][i]=fs['body_stability']; fields['geometry_quality'][i]=-c['task_space_weighted_geometry_cost']
        fields['other_quality'][i]=q['total']-fs['task_progress']-fs['body_stability']+c['task_space_weighted_geometry_cost']
        fields['critic_TD'][i]=gamma*(not done[i])*next_value[i]-value[i]
    contributions={}
    for key,vs in fields.items():
        out=np.zeros(n); tail=0.
        for i in reversed(range(n)):
            tail=vs[i]+gamma*lam*(not done[i])*tail; out[i]=tail
        contributions[key]=out
    reconstructed=sum(contributions.values())
    # FP64 component decomposition versus originally accumulated FP32 GAE.
    assert np.max(np.abs(reconstructed-raw)) < 2e-5, (
        float(np.max(np.abs(reconstructed-raw))),int(np.argmax(np.abs(reconstructed-raw))),
        fields['other_quality'].min(),np.flatnonzero(done).tolist(),reconstructed[:3].tolist(),raw[:3].tolist())
    return fields,contributions,raw,done


def review_rollout(update, r, selected):
    start=209921+(update-1606)*128
    rows=[selected[start+i] for i in range(128)]
    x=r['observations']['policy'][:,0]; phases=x[:,:13].argmax(-1).numpy()+1
    adv=arr(r['advantages']).reshape(128); fields,contributions,raw,done=discounted_components(r,rows)
    assert np.max(np.abs(adv-(raw-raw.mean())/(raw.std(ddof=1)+1e-8))) < 1e-5
    likelihood=json.loads((RUN/f'rollouts/update_{update:06d}_likelihood.json').read_text())
    counts=np.zeros(128,int); old_to_last_logp=np.full(128,np.nan); mean_gradient=np.zeros((128,12)); sigma_gradient=np.zeros((128,12))
    for batch in likelihood['minibatches']:
        ids=np.array(batch['rollout_flat_indices']).reshape(-1); counts[ids]+=1
        assert np.max(np.abs(arr(batch['actual_advantage']).reshape(-1)-adv[ids])) < 1e-6, (
            update,batch['minibatch_index'],arr(batch['actual_advantage']).shape,ids.shape,
            float(np.max(np.abs(arr(batch['actual_advantage']).reshape(-1)-adv[ids]))),
            arr(batch['actual_advantage']).reshape(-1)[:3].tolist(),adv[ids][:3].tolist())
        old_to_last_logp[ids]=arr(batch['optimization_log_probability']).reshape(-1)-arr(batch['old_log_probability']).reshape(-1)
        mean_gradient[ids]+=arr(batch['loss_gradient_wrt_network_mean_full12'])
        sigma_gradient[ids]+=arr(batch['loss_gradient_wrt_network_log_sigma_full12'])
    assert (counts==5).all()
    sample=[]
    for i,row in enumerate(rows):
        task=row['applied_audit']['semantic_task']; fr=task['physical_evaluator']['current_legs']['FR']
        previous=selected.get(start+i-1); delta_gap=delta_front=None
        if previous and previous['applied_audit']['sim_time_s'] < row['applied_audit']['sim_time_s']:
            previous_fr=previous['applied_audit']['semantic_task']['physical_evaluator']['current_legs']['FR']
            delta_gap=fr['clearance_m']-previous_fr['clearance_m']; delta_front=fr['front_distance_m']-previous_fr['front_distance_m']
        sample.append({'decision':start+i,'phase':int(phases[i]),'episode_time':row['applied_audit']['sim_time_s'],
            'gap_m':fr['clearance_m'],'front_m':fr['front_distance_m'],'d_gap_m':delta_gap,'d_front_m':delta_front,
            'ground':fr['ground_contact'],'crossed':task['front_edge_crossed_history']['FR'],
            'raw_GAE':float(raw[i]),'stored_advantage':float(adv[i]),'reward':row['reward'],
            'old_value':float(arr(r['values']).reshape(128)[i]),'GAE_return':float(arr(r['returns']).reshape(128)[i]),
            'last_minibatch_logp_delta':float(old_to_last_logp[i]),'terminal':bool(done[i]),
            'costs':{key:float(v[i]) for key,v in fields.items()},
            'GAE_components':{key:float(v[i]) for key,v in contributions.items()},
            'actual_mean_head_loss_gradient_sum':{name:float(mean_gradient[i,j]) for name,j in CHANNELS.items()},
            'actual_raw_minus_oldmu':{name:float(arr(r['actions'])[i,0,j]-arr(r['distribution_params'][0])[i,0,j]) for name,j in CHANNELS.items()}})
    groups={
        'P01':[i for i,s in enumerate(sample) if s['phase']==1],
        'P02_clearance_below_top':[i for i,s in enumerate(sample) if s['phase']==2 and s['gap_m']<0],
        'P02_gap_falling_front_retreating':[i for i,s in enumerate(sample) if s['phase']==2 and s['d_gap_m'] is not None and s['d_gap_m']<0 and s['d_front_m']<0],
        'P02_gap_positive_front_advancing':[i for i,s in enumerate(sample) if s['phase']==2 and s['gap_m']>0 and s['d_front_m'] is not None and s['d_front_m']>0],
        'terminal':[i for i,s in enumerate(sample) if s['terminal']]}
    group_summary={}
    for name,ids in groups.items():
        group_summary[name]={'n':len(ids),'advantage':compact(adv[ids]),'raw_GAE':compact(raw[ids]),
            'last_minibatch_logp_delta':compact(old_to_last_logp[ids]),
            'per_step_component_sums':{key:float(v[ids].sum()) for key,v in fields.items()},
            'GAE_component_means':{key:float(v[ids].mean()) if ids else None for key,v in contributions.items()}}
    picks=sorted(set([0,1,2,3,127,*groups['P01'],*groups['terminal'],
        *sorted(groups['P02_gap_falling_front_retreating'],key=lambda i:adv[i],reverse=True)[:2],
        *sorted(groups['P02_gap_positive_front_advancing'],key=lambda i:adv[i])[:2]]))
    return {'update':update,'first_decision':start,'phase_counts':{str(k):int((phases==k).sum()) for k in sorted(set(phases))},
        'raw_GAE_mean':float(raw.mean()),'raw_GAE_sample_std':float(raw.std(ddof=1)),
        'FP64_component_vs_saved_FP32_GAE_max_error':float(np.max(np.abs(sum(contributions.values())-raw))),
        'nonterminal_tail_bootstrap_inferred_algebraically_from_saved_return_not_independently_measured':True,
        'reward_sums':{key:float(v.sum()) for key,v in fields.items()},'groups':group_summary,
        'stored_advantage_recomputed_exact_tolerance':1e-5,'actual_minibatch_reuse_counts':[int(counts.min()),int(counts.max())],
        'mean_head_gradient_abs_sum':np.abs(mean_gradient).sum(0).tolist(),
        'sigma_head_gradient_abs_sum':np.abs(sigma_gradient).sum(0).tolist(),
        'picked_samples':[sample[i] for i in picks]}


def main():
    assert not torch.cuda.is_available(), 'CPU-only audit required'
    torch.set_num_threads(1)
    cp0=torch.load(AUDIT['source_checkpoint'],map_location='cpu',weights_only=False)
    cp1=torch.load(AUDIT['checkpoint'],map_location='cpu',weights_only=False)
    actor0,actor1=model(cp0),model(cp1)
    rollouts={u:torch.load(RUN/f'rollouts/rollout_{u:06d}.pt',map_location='cpu',weights_only=False) for u in [1606,1621]}
    selected=scan_selected(); output={'schema':'wlr50_clean.block06_two_endpoint_readonly.v1',
        'source':AUDIT['source_checkpoint'],'end':AUDIT['checkpoint'],'optimization_steps':0,'physics_steps':0,
        'rollouts':{},'same_input':{}}
    for u,r in rollouts.items():
        output['rollouts'][str(u)]=review_rollout(u,r,selected)
        x=r['observations']['policy'][:,0]
        output['same_input'][str(u)]=paired(actor0,actor1,x)
        p01=x[x[:,0]==1]
        output['same_input'][str(u)+'_P01']=paired(actor0,actor1,p01)
        state=dict(cp1['actor_state_dict']); state['mlp.0.weight']=state['mlp.0.weight'].clone()
        state['mlp.0.weight'][:,:2]=cp0['actor_state_dict']['mlp.0.weight'][:,:2]
        mu0=arr(mean(actor0,p01)); mu1=arr(mean(actor1,p01)); muhybrid=arr(mean(actor1,p01,state))
        output['same_input'][str(u)+'_P01_readonly_functional_phase_columns_restored']={name:{
            'total_new_minus_old':(mu1[:,j]-mu0[:,j]).tolist(),
            'new_shared_parameters_with_old_phase_columns_minus_old':(muhybrid[:,j]-mu0[:,j]).tolist()}
            for name,j in CHANNELS.items()}
    teacher=np.load(OUT/'det_front_rehearsal_data_v1/candidate.npz',allow_pickle=False)
    output['same_input']['historical_actual_deterministic_P02_254']=paired(actor0,actor1,
        torch.tensor(teacher['X389_reconstructed']),teacher['raw12_actual_deterministic'])
    print(json.dumps(output,ensure_ascii=False,allow_nan=False))


if __name__=='__main__':main()
