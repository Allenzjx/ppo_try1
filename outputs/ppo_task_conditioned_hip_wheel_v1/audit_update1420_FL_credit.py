"""Only sealed rollout1420 plus its prefix-state context; no optimizer replay."""
from collections import Counter
from itertools import islice
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import math
import sys
import torch
import torch.nn.functional as F
import yaml

OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1];sys.path.insert(0,str(ROOT/'src'))
from wlr50_clean.ppo.semantic_supervisor import TaskStageSupervisor
from wlr50_clean.ppo.semantic_history_actor import cap_transition_request_history,task_conditioned_effective_log_std
RUN=ROOT/'runs/ppo_task_conditioned_hip_wheel_v1/train/20260921T0459512815083Z_gee5a9651591d_8e120f68dc7c4598b08dc79fcd3d34a4'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(x):
    x=torch.as_tensor(x).double()
    return dict(n=x.numel(),mean=float(x.mean()),minimum=float(x.min()),maximum=float(x.max()),positive=int((x>0).sum()),negative=int((x<0).sum()))


def main():
    torch.set_num_threads(1);rng=torch.get_rng_state().clone()
    with (RUN/'residual_and_projection_audit.jsonl').open('rb') as f:content=b''.join(islice(f,384))
    prefix=[json.loads(line) for line in content.splitlines()];assert len(prefix)==384
    rows=prefix[256:];assert [r['global_policy_decision'] for r in rows]==list(range(186113,186241))
    sp=RUN/'rollouts/rollout_001420.pt';hp=RUN/'rollouts/update_001420_likelihood.json'
    s=torch.load(sp,map_location='cpu',weights_only=False);h=json.loads(hp.read_text());obs=s['observations']['policy'][:,0]
    for key,field in [('raw_policy_action_full12','actions'),('reward','rewards'),('old_value','values'),('old_log_probability','actions_log_prob')]:
        assert torch.equal(torch.tensor([r[key] for r in rows]).reshape_as(s[field][:,0]),s[field][:,0])
    assert not s['dones'].any() and not any(r['terminal'] for r in rows)
    gae=(s['returns']-s['values']).flatten();adv=s['advantages'].flatten()
    norm=(gae-gae.mean())/(gae.std()+1e-8);assert float((norm-adv).abs().max())<1e-6
    spec_path=ROOT/'configs/ppo_task_conditioned_hip_wheel_v1/stage_task_spec.yaml';spec=yaml.safe_load(spec_path.read_text())
    new=SimpleNamespace(spec=spec);old=SimpleNamespace(spec={k:v for k,v in spec.items() if k!='rolling_capture_retention'})
    def retention(ev):
        parts={}
        for leg in ('FL','FR'):
            active=bool(ev['history']['placed'][leg])
            new_ret=TaskStageSupervisor._current_capture_retention(new,leg,ev)
            old_ret=TaskStageSupervisor._current_capture_retention(old,leg,ev)
            parts[leg]=dict(placed=active,new_retention=new_ret,without_rolling_retention=old_ret,
                delta_Phi_existing_capture_share=.85/4*.2*(new_ret-old_ret) if active else 0.)
        parts['total_delta_Phi']=sum(parts[leg]['delta_Phi_existing_capture_share'] for leg in ('FL','FR'))
        return parts
    before_after=[]
    for i,row in enumerate(rows):
        before=prefix[255+i]['applied_audit']['semantic_task'];after=row['applied_audit']['semantic_task'];rb=row['applied_audit']['reward_breakdown']
        assert abs(rb['potential_before']-before['task_progress_potential'])<1e-12
        assert abs(rb['potential_after']-after['task_progress_potential'])<1e-12
        b=retention(before['physical_evaluator']);a=retention(after['physical_evaluator'])
        assert max(rb['potential_before']-b['total_delta_Phi'],rb['potential_after']-a['total_delta_Phi'])<1.
        delta=5*(.9985*a['total_delta_Phi']-b['total_delta_Phi'])
        before_after.append(dict(before=b,after=a,isolated_rolling_retention_PBRS_delta=delta,
            logged_total_reward_minus_only_this_delta=row['reward']-delta))
    capture=next(i for i,row in enumerate(rows) if row['applied_audit']['semantic_task']['physical_evaluator']['history']['placed']['FL'])
    assert not prefix[255+capture]['applied_audit']['semantic_task']['physical_evaluator']['history']['placed']['FL']
    lost=next(i for i in range(capture+1,128) if not rows[i]['applied_audit']['semantic_task']['physical_evaluator']['current_legs']['FL']['support'])
    assert rows[lost]['applied_audit']['phase_id']=='P06'
    history,_=cap_transition_request_history(obs);endpoint=[];cp_paths=[]
    for step in (186112,186240):
        cp=OUT/f'checkpoints/history/checkpoint_step_{step:09d}.pt';side=cp.with_name(cp.stem+'_manifest.json')
        state=torch.load(cp,map_location='cpu',weights_only=False)['actor_state_dict'];manifest=json.loads(side.read_text())
        assert sha(cp)==manifest['checkpoint_sha256'];cp_paths.extend((cp,side))
        with torch.inference_mode():
            x=F.elu(F.linear(obs,state['mlp.0.weight'],state['mlp.0.bias']))
            x=F.elu(F.linear(x,state['mlp.2.weight'],state['mlp.2.bias']))
            head=F.linear(x,state['mlp.4.weight'],state['mlp.4.bias']).reshape(128,2,12)
            mu=.1*head[:,0]+.9*history;log,_=task_conditioned_effective_log_std(head[:,1],obs,.25)
        endpoint.append(dict(network=head[:,0],mu=mu,sigma=log.exp()))
    initial_mean_error=float((endpoint[0]['mu']-s['distribution_params'][0][:,0]).abs().max());assert initial_mean_error<1e-6
    uses={i:[] for i in range(128)};gradient_error=0.
    for mb in h['minibatches']:
        for j,indices in enumerate(mb['rollout_flat_indices']):
            assert len(indices)==1;i=indices[0]
            om=s['distribution_params'][0][i,0].double();os=s['distribution_params'][1][i,0].double();action=s['actions'][i,0].double()
            nm=torch.tensor(mb['current_conditional_mean'][j],dtype=torch.float64);ns=torch.tensor(mb['current_conditional_sigma'][j],dtype=torch.float64)
            logp=lambda m,sd:-sd.log()-.5*((action-m)/sd).square()
            change=logp(nm,ns)-logp(om,os)
            ratio=mb['ratio'][j];advantage=mb['actual_advantage'][j];clipped=mb['clipped_branch_strictly_active'][j]
            assert advantage==float(adv[i])
            derivative=torch.tensor(mb['loss_gradient_wrt_network_mean_full12'][j],dtype=torch.float64)
            expected=torch.zeros(12,dtype=torch.float64) if clipped else -.1*advantage*ratio/32*(action-nm)/ns.square()
            gradient_error=max(gradient_error,float((derivative-expected).abs().max()))
            uses[i].append(dict(minibatch=mb['minibatch_index'],ratio=ratio,normalized_A=advantage,strict_clip=clipped,
                network_mean_FL_hip_knee=mb['current_network_mean_full12'][j][:2],conditional_mean_FL_hip_knee=nm[:2].tolist(),
                sigma_FL_hip_knee=ns[:2].tolist(),
                actual_total_loss_derivative_wrt_network_mean_FL_hip_knee=derivative[:2].tolist(),
                actual_total_loss_derivative_wrt_log_sigma_FL_hip_knee=mb['loss_gradient_wrt_network_log_sigma_full12'][j][:2],
                FL_hip_knee_delta_logp=change[:2].tolist(),other10_delta_logp=float(change[2:].sum()),
                actual_joint_delta_logp=mb['optimization_log_probability'][j]-mb['old_log_probability'][j]))
    assert all(len(v)==5 for v in uses.values()) and gradient_error<1e-5
    selected_indices=sorted(set([0,capture-3,capture-2,capture-1,capture,capture+1,capture+2,lost-1,lost,lost+1,lost+2,lost+5,127]))
    selected=[]
    for i in selected_indices:
        row=rows[i];a=row['applied_audit'];q=row['policy_request'];ev=a['semantic_task']['physical_evaluator'];rb=a['reward_breakdown']
        selected.append(dict(rollout_index=i,block_index=i+256,global_decision=row['global_policy_decision'],phase=a['phase_id'],end_phase=a['end_phase_id'],tick=a['physics_tick'],
            sim_time_s=a['sim_time_s'],FL={k:ev['current_legs']['FL'][k] for k in ('air','top_contact','support','bearing_force_n','clearance_m','front_distance_m')},
            FL_placed_tick=ev['history']['event_ticks']['placed'].get('FL'),reward=row['reward'],Phi_before=rb['potential_before'],Phi_after=rb['potential_after'],
            potential_shaping=rb['potential_shaping'],families=rb['families'],raw_GAE=float(gae[i]),normalized_A=float(adv[i]),
            rolling_retention=before_after[i],
            FL_actual_collection={k:q[k][:2] for k in ('base_mean_full12','history_center_full12','conditional_mean_full12','effective_sigma_full12','selected_raw_full12')},
            FL_physical_requested=a['actuator_target_effect_audit']['policy_headroom_evidence']['requested_policy_residual_full12'][:2],
            state_weights=q['task_state_weights'],actual_optimizer_uses=uses[i],
            endpoint_same_numeric_observation=dict(source_network_FL_hip_knee=endpoint[0]['network'][i,:2].tolist(),target_network_FL_hip_knee=endpoint[1]['network'][i,:2].tolist(),
                source_conditional_FL_hip_knee=endpoint[0]['mu'][i,:2].tolist(),target_conditional_FL_hip_knee=endpoint[1]['mu'][i,:2].tolist(),
                source_sigma_FL_hip_knee=endpoint[0]['sigma'][i,:2].tolist(),target_sigma_FL_hip_knee=endpoint[1]['sigma'][i,:2].tolist())))
    group={}
    for name,indices in dict(P05=[i for i,r in enumerate(rows) if r['applied_audit']['phase_id']=='P05'],
        P06_before_FL_loss=list(range(capture+1,lost)),P06_FL_loss_through_tail=list(range(lost,128))).items():
        group[name]=dict(n=len(indices),GAE=summarize(gae[indices]),normalized_A=summarize(adv[indices]),
            reward=summarize([rows[i]['reward'] for i in indices]),
            rolling_retention_reward_delta=summarize([before_after[i]['isolated_rolling_retention_PBRS_delta'] for i in indices]),
            network_mean_delta_FL_hip_knee=(endpoint[1]['network'][indices,:2]-endpoint[0]['network'][indices,:2]).mean(0).tolist(),
            conditional_mean_delta_FL_hip_knee=(endpoint[1]['mu'][indices,:2]-endpoint[0]['mu'][indices,:2]).mean(0).tolist(),
            sigma_ratio_FL_hip_knee=(endpoint[1]['sigma'][indices,:2]/endpoint[0]['sigma'][indices,:2]).mean(0).tolist())
    assert torch.equal(rng,torch.get_rng_state())
    result=dict(schema='wlr50_clean.update1420_FL_capture_credit.v1',run=str(RUN),update=1420,
        scope='Completed rollout1420 only,128 real training decisions; previous sealed rows only supply pre-action physical context. No later active tail read.',
        training_added_by_audit=0,first384_row_prefix_sha256=hashlib.sha256(content).hexdigest(),
        source_sha256={str(p):sha(p) for p in (sp,hp,spec_path,*cp_paths)},capture_rollout_index=capture,first_FL_loss_rollout_index=lost,
        actual_phase_counts=dict(Counter(r['applied_audit']['phase_id'] for r in rows)),all128_no_done=True,
        normalized_GAE_max_error=float((norm-adv).abs().max()),actual_mean_gradient_formula_max_error=gradient_error,
        source_CP_CPU_vs_collection_mean_max_error=initial_mean_error,RNG_unchanged=True,
        group_summary=group,selected=selected,
        boundaries=['The retention reward delta is exact arithmetic isolation on the logged physical states, holding every other term fixed. Removing retention would change observation globalPhi and future policy states, so this is not an alternative physical trajectory or recomputed GAE.',
            'End-point CP means use the same numeric saved observations/HISTORY, not natural deterministic entrances.',
            'Actual head derivatives are total official-loss derivatives before parameter clipping, not an isolated sample Adam update. FL mean derivative matches the clipped PPO formula; log-sigma derivative can also include entropy.',
            'A last-hook likelihood is before its minibatch optimizer step and is not final checkpoint likelihood. Joint logp does not establish FL hip absorption.',
            'Capture was real; subsequent FL loss means stable rolling/whole-task success has not been established by this block.'])
    with (OUT/'update1420_FL_capture_credit.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ('actual_phase_counts','capture_rollout_index','first_FL_loss_rollout_index','group_summary','actual_mean_gradient_formula_max_error')},indent=2))
    for q in selected:
        if q['rollout_index'] in (capture-1,capture,capture+1,lost,lost+1,127):
            print('EVENT',q['global_decision'],q['tick'],q['phase'],q['reward'],q['Phi_before'],q['Phi_after'],q['raw_GAE'],q['normalized_A'],q['rolling_retention']['isolated_rolling_retention_PBRS_delta'])
            print('FL',q['FL_actual_collection'],'ENDPOINT',q['endpoint_same_numeric_observation'])
            print('USES',q['actual_optimizer_uses'])


if __name__=='__main__':main()
