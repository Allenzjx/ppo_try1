"""Bounded CPU-only, no optimizer or simulator: sealed rollout 1637 only."""
from pathlib import Path
from copy import deepcopy
from collections import Counter
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch
import yaml
from tensordict import TensorDict
from wlr50_clean.ppo.semantic_p05_capture_actor import SemanticP05CaptureHistoryMLPModel, p05_capture_request_history
from wlr50_clean.ppo.semantic_p05_capture_profile import P05_CAPTURE_OBSERVATION_LAYOUT
from wlr50_clean.ppo.semantic_history_actor import history_conditioned_head, HISTORY_RHO
from wlr50_clean.ppo.semantic_receiving_wheel_sigma import receiving_wheel_effective_log_std
from wlr50_clean.ppo.semantic_supervisor import TaskStageSupervisor, _current_rr_receiver_preparation_retired

OUT = ROOT / 'outputs/ppo_p05_hip_only_continuation_v1'
RUN = ROOT / 'runs/ppo_p05_hip_only_continuation_v1/train/20260922T1301289433343Z_g5fd88852bf20_337c83c1b0214e02b62aa98b579e2812'
CHANNELS = {'RR_hip': 6, 'RR_knee': 7, 'RR_wheel': 11, 'FL_hip': 0, 'FL_knee': 1,
            'FL_wheel': 8, 'FR_wheel': 9, 'RL_wheel': 10, 'RL_hip': 4, 'FR_knee': 3}
GAMMA = .9985


def array(v):
    return v.detach().cpu().numpy().astype(float) if torch.is_tensor(v) else np.asarray(v, dtype=float)


def stats(v):
    v = array(v).reshape(-1)
    return {'n': len(v), 'first': float(v[0]), 'last': float(v[-1]), 'min': float(v.min()),
            'max': float(v.max()), 'mean': float(v.mean()), 'sum': float(v.sum()),
            'positive': int((v > 0).sum())} if len(v) else {'n': 0}


def td(x):
    return TensorDict({'policy': x, 'critic': x.clone()}, batch_size=[len(x)])


def build(cp):
    actor = SemanticP05CaptureHistoryMLPModel(td(torch.zeros(1,389)), {'actor':['policy'],'critic':['critic']},
        'actor', 12, hidden_dims=(256,256), activation='elu', obs_normalization=False,
        distribution_cfg={'class_name':'HeteroscedasticGaussianDistribution','std_type':'log','init_std':.15},
        observation_layout=P05_CAPTURE_OBSERVATION_LAYOUT, exploration_std_temperature=.25)
    actor.load_state_dict(cp['actor_state_dict'], strict=True)
    return actor.eval()


def distribution(actor, x):
    with torch.no_grad():
        center, _ = p05_capture_request_history(x)
        head = history_conditioned_head(actor.mlp(x), center, HISTORY_RHO)
        logstd, _ = receiving_wheel_effective_log_std(head[...,1,:], x[...,:372], .25)
        return array(head[...,0,:]), array(logstd.exp())


def corr(a,b):
    a,b=array(a).reshape(-1),array(b).reshape(-1)
    return float(np.corrcoef(a,b)[0,1]) if len(a)>1 and a.std()>1e-15 and b.std()>1e-15 else None


def main():
    torch.set_num_threads(1)
    assert not torch.cuda.is_available(), 'Explicit CPU-only environment required'
    rpath=RUN/'rollouts/rollout_001637.pt'
    r=torch.load(rpath,map_location='cpu',weights_only=False)
    selected=[]
    # Exactly one preceding endpoint plus the requested sealed 128 decisions.
    with (RUN/'residual_and_projection_audit.jsonl').open(encoding='utf-8') as stream:
        for index,line in enumerate(stream):
            if index>639: break
            if index>=511: selected.append(json.loads(line))
    assert [v['global_policy_decision'] for v in selected]==list(range(213888,214017))
    rows=selected[1:]
    evs=[v['applied_audit']['semantic_task']['physical_evaluator'] for v in selected]
    assert not any(v['terminal'] for v in selected)
    x=r['observations']['policy'][:,0]
    assert tuple(x.shape)==(128,389)
    actions=array(r['actions'])[:,0]
    oldmu=array(r['distribution_params'][0])[:,0]
    oldstd=array(r['distribution_params'][1])[:,0]
    assert np.max(np.abs(actions-array([v['raw_policy_action_full12'] for v in rows])))<1e-7
    assert np.max(np.abs(oldmu-array([v['old_distribution_mean_full12'] for v in rows])))<1e-7
    assert np.max(np.abs(oldstd-array([v['old_distribution_std_full12'] for v in rows])))<1e-7
    spec_path=ROOT/'configs/ppo_p05_hip_only_continuation_v1/stage_task_spec.yaml'
    spec=yaml.safe_load(spec_path.read_text())
    manifest=json.loads((RUN/'run_manifest.started.json').read_text())
    for rel in ['configs/ppo_p05_hip_only_continuation_v1/stage_task_spec.yaml','src/wlr50_clean/ppo/semantic_supervisor.py']:
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==manifest['runtime_contract']['files'][rel]
    sup=object.__new__(TaskStageSupervisor);sup.spec=spec
    old=object.__new__(TaskStageSupervisor);old.spec=deepcopy(spec);old.spec.pop('rr_postcross_workspace_semantics')
    phi=[]; cap=[]; retired=[];delta=[];gap=[];front=[];workspace=[];capture=[]
    for ev,row in zip(evs,selected):
        ph=sup.physical_potential(ev)
        assert abs(ph-row['applied_audit']['semantic_task']['task_progress_potential'])<1e-12
        rr=ev['current_legs']['RR'];hist=ev['history']
        cp=sup._current_capture_progress('RR',ev,min(1.,rr['consecutive_top_samples']/spec['history']['minimum_top_samples']))
        consumed=_current_rr_receiver_preparation_retired(spec,'RR',ev) and not hist['placed']['RR']
        phi.append(ph);cap.append(.85/4*.2*cp);capture.append(cp)
        retired.append(consumed);delta.append(ph-old.physical_potential(ev))
        gap.append(rr['clearance_m']);front.append(rr['front_distance_m'])
        workspace.append(float(ev['transfer_roles']['RR']['workspace_progress']))
    phi,cap,delta,gap,front,workspace,capture=map(np.asarray,(phi,cap,delta,gap,front,workspace,capture))
    retired=np.asarray(retired,bool)
    raw=array(r['returns']).reshape(-1)-array(r['values']).reshape(-1)
    adv=array(r['advantages']).reshape(-1)
    assert np.max(np.abs(adv-(raw-raw.mean())/(raw.std(ddof=1)+1e-8)))<1e-5
    reward=array(r['rewards']).reshape(-1)
    costs={k:np.asarray([v['applied_audit']['reward']['families'][k] for v in rows])
           for k in rows[0]['applied_audit']['reward']['families']}
    shape=5*(GAMMA*phi[1:]-phi[:-1])
    assert np.max(np.abs(shape-array([v['applied_audit']['reward']['potential_shaping'] for v in rows])))<1e-12
    capshape=5*(GAMMA*cap[1:]-cap[:-1])
    retire_shape=5*(GAMMA*delta[1:]-delta[:-1])
    time=np.asarray([-.02*v['applied_audit']['reward']['elapsed_physics_s'] for v in rows])
    value=array(r['values']).reshape(-1)
    next_value=np.r_[value[1:],(raw[-1]-reward[-1]+value[-1])/GAMMA]
    fields={'capture_shaping':capshape,'noncapture_shaping':shape-capshape,'time':time,
            'other_cost':reward-shape-time,'critic_TD':GAMMA*next_value-value}
    gae={}
    for key,values in fields.items():
        out=np.zeros(128);tail=0.
        for i in reversed(range(128)):
            tail=values[i]+GAMMA*.99*tail;out[i]=tail
        gae[key]=out
    gae_error=float(np.max(np.abs(sum(gae.values())-raw)))
    # Stored FP32 values near -18 accumulate cancellation across 128 steps.
    assert gae_error<1e-4, gae_error
    logs=json.loads((RUN/'rollouts/update_001637_likelihood.json').read_text())
    uses=np.zeros(128,int); dlogp=np.zeros(128);gmu=np.zeros((128,12));gstd=gmu.copy()
    for b in logs['minibatches']:
        ids=np.asarray(b['rollout_flat_indices']).reshape(-1);uses[ids]+=1
        assert np.max(np.abs(array(b['actual_advantage']).reshape(-1)-adv[ids]))<1e-6
        dlogp[ids]=array(b['optimization_log_probability']).reshape(-1)-array(b['old_log_probability']).reshape(-1)
        gmu[ids]+=array(b['loss_gradient_wrt_network_mean_full12'])
        gstd[ids]+=array(b['loss_gradient_wrt_network_log_sigma_full12'])
    assert (uses==5).all()
    cp0path=OUT/'checkpoints/history/checkpoint_step_000213888.pt'
    cp1path=OUT/'checkpoints/history/checkpoint_step_000214016.pt'
    cp0=torch.load(cp0path,map_location='cpu',weights_only=False)
    cp1=torch.load(cp1path,map_location='cpu',weights_only=False)
    mu0,sigma0=distribution(build(cp0),x);mu1,sigma1=distribution(build(cp1),x)
    _,history_evidence=p05_capture_request_history(x)
    caps=array(history_evidence['current_cap_full12'])
    request0,request1=caps*np.tanh(mu0),caps*np.tanh(mu1)
    assert np.max(np.abs(mu0-oldmu))<1e-6
    assert np.allclose(sigma0,oldstd,atol=1e-6,rtol=2e-5), (float(np.max(np.abs(sigma0-oldstd))),float(oldstd.max()))
    # Current geometry must stay qualified/over top for this local descent group.
    post=retired[:-1]&retired[1:]
    descent=post&(np.diff(gap)<0)
    ascent=post&(np.diff(gap)>0)
    groups={'all':np.arange(128),'stable_postcross':np.flatnonzero(post),
            'qualified_descent':np.flatnonzero(descent),'qualified_ascent':np.flatnonzero(ascent),
            'retirement_newly_active':np.flatnonzero(retired[1:]&~retired[:-1])}
    result={'schema':'wlr50_clean.block08_rollout5_RR_readonly.v1','run':str(RUN),'update':1637,
        'global_decisions':[213889,214016],'optimization_steps':0,'physics_steps':0,
        'actor_comparison':[str(cp0path),str(cp1path)],
        'sealed_rollout_sha256':hashlib.sha256(rpath.read_bytes()).hexdigest(),
        'input_phase_counts':dict(Counter(f'P{int(v)+1:02}' for v in x[:,:13].argmax(-1))),
        'endpoint_time_s':[rows[0]['applied_audit']['sim_time_s'],rows[-1]['applied_audit']['sim_time_s']],
        'raw_GAE':stats(raw),'raw_GAE_sample_std':float(raw.std(ddof=1)),
        'advantages':stats(adv),'value':stats(array(r['values'])),'returns':stats(array(r['returns'])),
        'each_actual_sample_use_count':[int(uses.min()),int(uses.max())],
        'retirement_input_count':int(retired[:-1].sum()),'retirement_endpoint_count':int(retired[1:].sum()),
        'retirement_effective_input_count':int((abs(delta[:-1])>1e-12).sum()),
        'retirement_effective_endpoint_count':int((abs(delta[1:])>1e-12).sum()),
        'retirement_potential_delta':stats(delta[1:]),'old_receiver_workspace_progress':stats(workspace[1:]),
        'raw_mean_cpu_replay_max_error':float(np.max(np.abs(mu0-oldmu))),
        'sigma_cpu_replay_max_error':float(np.max(np.abs(sigma0-oldstd))),
        'sigma_cpu_replay_max_relative_error':float(np.max(np.abs(sigma0-oldstd)/oldstd)),
        'FP64_GAE_decomposition_vs_saved_FP32_max_error':gae_error,
        'tail_bootstrap_inferred_algebraically_not_independently_measured':True,
        'newly_retired_decisions':[213889+int(i) for i in groups['retirement_newly_active']],
        'RR_lift_endpoints':sum(ev['history']['active_lift']['RR'] for ev in evs[1:]),
        'RR_crossed_endpoints':sum(ev['history']['front_edge_crossed']['RR'] for ev in evs[1:]),
        'RR_placed_endpoints':sum(ev['history']['placed']['RR'] for ev in evs[1:]),
        'RR_TOP_endpoints':sum(ev['current_legs']['RR']['top_contact'] for ev in evs[1:]),
        'groups':{},'same_input_channel_changes':{},'samples':[]}
    for name,ids in groups.items():
        result['groups'][name]={'n':len(ids),'gap_m':stats(gap[1:][ids]),'front_m':stats(front[1:][ids]),
            'gap_change_m':stats(np.diff(gap)[ids]),'capture_progress':stats(capture[1:][ids]),
            'reward':stats(reward[ids]),'raw_GAE':stats(raw[ids]),'advantage':stats(adv[ids]),
            'raw_negative_normalized_positive':int(((raw[ids]<0)&(adv[ids]>0)).sum()),
            'logp_last_pre_step_delta':stats(dlogp[ids]),
            'reward_family_sums':{k:float(v[ids].sum()) for k,v in costs.items()},
            'potential_shaping_sum':float(shape[ids].sum()),'time_cost_sum':float(time[ids].sum()),
            'capture_shaping_sum':float(capshape[ids].sum()),
            'noncapture_shaping_sum':float((shape-capshape)[ids].sum()),
            'retirement_counterfactual_shaping_difference_sum':float(retire_shape[ids].sum()),
            'undiscounted_capture_progress_reward_sum':float((5*np.diff(cap))[ids].sum()),
            'undiscounted_noncapture_progress_reward_sum':float((5*np.diff(phi-cap))[ids].sum()),
            'potential_discount_cost_sum':float((-5*(1-GAMMA)*phi[1:])[ids].sum()),
            'GAE_component_means':{key:float(v[ids].mean()) if len(ids) else None for key,v in gae.items()},
            'correlation_gap_delta_rawGAE':corr(np.diff(gap)[ids],raw[ids]),
            'correlation_gap_delta_advantage':corr(np.diff(gap)[ids],adv[ids])}
    for name,j in CHANNELS.items():
        result['same_input_channel_changes'][name]={'old_mean':stats(mu0[:,j]),'new_mean':stats(mu1[:,j]),
            'new_minus_old_mean':stats(mu1[:,j]-mu0[:,j]),'old_sigma':stats(sigma0[:,j]),
            'new_minus_old_sigma':stats(sigma1[:,j]-sigma0[:,j]),
            'raw_minus_old_mean':stats(actions[:,j]-oldmu[:,j]),
            'sum_actual_loss_gradient_wrt_network_mean':float(gmu[:,j].sum()),
            'sum_actual_loss_gradient_wrt_network_logsigma':float(gstd[:,j].sum()),
            'postcross_delta_mean':stats((mu1[:,j]-mu0[:,j])[post]),
            'conditional_mean_request_units':'degree' if j<8 else 'rad/s',
            'conditional_mean_request_semantics':'current_cap*tanh(conditional_mu), before mapper/filter/headroom/slew; not final_target or expected_tanh_of_Gaussian',
            'conditional_mean_request_before':stats(request0[:,j]),
            'conditional_mean_request_after':stats(request1[:,j]),
            'conditional_mean_request_change':stats(request1[:,j]-request0[:,j]),
            'qualified_descent_request_change':stats((request1[:,j]-request0[:,j])[descent]),
            'qualified_ascent_request_change':stats((request1[:,j]-request0[:,j])[ascent]),
            'descent_raw_innovation_advantage_correlation':corr((actions[:,j]-oldmu[:,j])[descent],adv[descent]),
            'descent_sum_actual_mean_loss_gradient':float(gmu[descent,j].sum()),
            'ascent_sum_actual_mean_loss_gradient':float(gmu[ascent,j].sum())}
    picked=sorted(set([0,127,*groups['retirement_newly_active'].tolist(),
        int(np.argmin(gap[1:])),int(np.argmax(gap[1:])),
        *sorted(groups['qualified_descent'],key=lambda i:adv[i])[:2],
        *sorted(groups['qualified_ascent'],key=lambda i:adv[i],reverse=True)[:2]]))
    for i in picked:
        result['samples'].append({'decision':213889+int(i),'t':rows[i]['applied_audit']['sim_time_s'],
            'gap_m':float(gap[i+1]),'d_gap_m':float(gap[i+1]-gap[i]),'front_m':float(front[i+1]),
            'retired_input':bool(retired[i]),'retired_endpoint':bool(retired[i+1]),
            'capture_progress':float(capture[i+1]),'capture_shaping':float(capshape[i]),
            'other_shaping':float(shape[i]-capshape[i]),'reward':float(reward[i]),
            'rawGAE':float(raw[i]),'advantage':float(adv[i]),'last_logp_delta':float(dlogp[i])})
    scale=spec['geometry']['top_gap_max_m']
    result['RR_capture_formula']={'scale_m':scale,'Phi_air_capture_for_xy1':'.02125 * .025/(.025+abs(gap_m))',
        'one_mm_descent_undiscounted_reward_at_gap':{str(g):5*.02125*scale*(1/(scale+g-.001)-1/(scale+g)) for g in [.025,.05,.066,.1]},
        'not_a_physical_action_gradient':True}
    result['limitations']=['One nonterminal 128-decision rollout, no outcome or independent true-return value target.',
        'Advantages are ordinary relative normalization, not proof of wrong credit or a bug.',
        'Same-input mu changes are this actual Adam update, not observed physical effect or per-sample causal attribution.',
        'Retirement difference is a stateless counterfactual on recorded states; no alternative trajectory was executed.',
        'No optimizer, sampling, GPU, extra teacher, source/config/checkpoint writes.']
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))


if __name__=='__main__':
    main()
