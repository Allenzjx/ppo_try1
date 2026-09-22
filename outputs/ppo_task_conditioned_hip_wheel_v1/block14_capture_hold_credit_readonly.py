"""Read two sealed real rollouts and actual optimizer hooks; no forward/update."""
from __future__ import annotations
from collections import Counter,defaultdict
import itertools
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import torch
torch.set_num_threads(1)
from wlr50_clean.ppo.semantic_supervisor import TaskStageSupervisor,load_task_spec

RUN=ROOT/'runs/ppo_task_conditioned_hip_wheel_v1/train/20260921T1355291518150Z_g649ccd906421_aac78369d59846409cb2d4039083e87a'
CHANNELS={0:'FL_hip',1:'FL_knee',4:'RL_hip',6:'RR_hip',7:'RR_knee',8:'FL_wheel',9:'FR_wheel',10:'RL_wheel',11:'RR_wheel'}


def stats(values):
    x=torch.tensor(values,dtype=torch.float64)
    return {'n':len(values),'min':float(x.min()) if len(x) else None,
        'mean':float(x.mean()) if len(x) else None,'max':float(x.max()) if len(x) else None,
        'mean_abs':float(x.abs().mean()) if len(x) else None,
        'positive':int((x>0).sum()),'negative':int((x<0).sum()),'zero':int((x==0).sum())}


def main():
    sup=TaskStageSupervisor.__new__(TaskStageSupervisor)
    sup.spec=load_task_spec(ROOT/'configs/ppo_task_conditioned_hip_wheel_v1/stage_task_spec.yaml')
    with (RUN/'residual_and_projection_audit.jsonl').open() as stream:
        rows=[json.loads(line) for line in itertools.islice(stream,256,512)]
    assert len(rows)==256 and rows[0]['global_policy_decision']==197377 and rows[-1]['global_policy_decision']==197632
    snapshots={u:torch.load(RUN/f'rollouts/rollout_{u:06d}.pt',map_location='cpu',weights_only=False) for u in (1508,1509)}
    with (RUN/'advantage_audit.jsonl').open() as stream:
        advantage_audits={v['ppo_update_intended']:v for v in (json.loads(line) for line in itertools.islice(stream,2,4))}
    assert set(advantage_audits)=={1508,1509}
    assert all(v['stored_advantage_semantics']=='official_whole_rollout_standardized_GAE' for v in advantage_audits.values())
    likelihood={u:json.loads((RUN/f'rollouts/update_{u:06d}_likelihood.json').read_text()) for u in (1508,1509)}
    hooks=defaultdict(list)
    for update,record in likelihood.items():
        assert len(record['minibatches'])==20
        assert record['extra_model_forwards']==0 and record['extra_random_draws']==0
        for batch in record['minibatches']:
            for j,matches in enumerate(batch['rollout_flat_indices']):
                assert len(matches)==1
                local=matches[0];request=rows[(update-1508)*128+local]['policy_request']
                assert batch['old_log_probability'][j]==float(snapshots[update]['actions_log_prob'][local,0,0])
                hooks[(update,local)].append({
                    'A':batch['actual_advantage'][j],'ratio':batch['ratio'][j],
                    'strict_clipped':batch['clipped_branch_strictly_active'][j],
                    'current_mu':batch['current_conditional_mean'][j],
                    'current_base':batch['current_network_mean_full12'][j],
                    'gradient':batch['loss_gradient_wrt_network_mean_full12'][j],
                    'sample_minus_current_mu':[a-b for a,b in zip(request['selected_raw_full12'],batch['current_conditional_mean'][j])]})
    data=[]
    previous=None
    max_phi_error=0.
    for index,row in enumerate(rows):
        update=1508+index//128;local=index%128;saved=snapshots[update]
        assert torch.equal(torch.tensor(row['raw_policy_action_full12']),saved['actions'][local,0])
        a=row['applied_audit'];q=a['reward'];task=a['semantic_task'];ev=task['physical_evaluator'];fl=ev['current_legs']['FL']
        phi=sup.physical_potential(ev);max_phi_error=max(max_phi_error,abs(phi-q['potential_after']))
        assert abs(phi-q['potential_after'])<1e-10
        placed=ev['history']['placed']['FL'];retention=sup._current_capture_retention('FL',ev) if placed else None
        rear=max(ev['current_legs'][leg]['front_distance_m'] for leg in ('RL','RR'))
        cfg=sup.spec['rolling_capture_retention'];weight=max(0.,min(1.,(cfg['rear_preparation_near_m']-rear)/cfg['blend_distance_m']))
        eligible=(all(ev['history']['placed'][p] for p in ('FR','FL')) and not ev['history']['placed']['RR']
            and ev['current_legs']['RR'].get('current_lift_valid') is not True)
        fl_component=.85/4*(.8+.2*retention) if placed else None
        component_shaping=(5*(.9985*fl_component-previous['FL_phi_component'])
            if previous is not None and previous['FL_phi_component'] is not None and fl_component is not None else None)
        h=hooks[(update,local)];assert len(h)==5
        r=row['policy_request'];geometry=q.get('task_space_quality_sample_audit',[])
        assert r['receiving_sigma_multiplier_full12']==[1.]*12
        assert r['sampling_draws']==1 and r['extra_model_forwards']==0 and r['extra_random_draws']==0
        assert r['rho']==.9
        event_tick=task.get('history',{}).get('event_ticks',{}).get('placed',{}).get('FL')
        item={'global_decision':row['global_policy_decision'],'learner_decision':row['global_policy_decision']-197120,
            'update':update,'local':local,'tick':a['physics_tick'],'phase':a['phase_id'],'end_phase':a['end_phase_id'],
            'FL_placed_history':placed,'FL_capture_event_tick':event_tick,'FL_surface':fl['contact_surface'],
            'FL_top_contact':fl['top_contact'],'FL_support':fl['support'],'FL_air':fl['air'],
            'FL_gap_mm':fl['clearance_m']*1000,'rear_distance_m':rear,'rolling_retention_eligible':eligible,
            'rolling_weight':weight if eligible else 0.,'FL_retention':retention,'FL_phi_component':fl_component,
            'FL_retention_component_PBRS':component_shaping,'potential_before':q['potential_before'],
            'potential_after':q['potential_after'],'potential_shaping':q['potential_shaping'],'reward':q['total'],
            'terminal_event':q['terminal_event'],'time_cost':.02*q['elapsed_physics_s'],
            'weighted_body_quality':q['families']['body_stability'],
            'weighted_contact_quality':q['families']['contact_motion_quality'],
            'weighted_smoothness':q['families']['control_smoothness'],
            'geometry_cost':q['cost_components'].get('task_space_weighted_geometry_cost',0.),
            'minimum_body_obstacle_separation_m':min((v['separation_lower_bound_m'] for v in geometry if v['valid']),default=None),
            'old_V':float(saved['values'][local,0,0]),'return':float(saved['returns'][local,0,0]),
            'raw_GAE':float(saved['returns'][local,0,0]-saved['values'][local,0,0]),
            'stored_A':float(saved['advantages'][local,0,0]),'standardized_A':stats([v['A'] for v in h]),
            'TD_delta_with_same_rollout_next_old_V':float(saved['rewards'][local,0,0]+.9985*saved['values'][local+1,0,0]-saved['values'][local,0,0]) if local<127 else None,
            'strict_clipped_count':sum(v['strict_clipped'] for v in h),'mean_ratio':sum(v['ratio'] for v in h)/5,
            'cap_entry_gate':r['cap_transition_gate_full12'],
            'channel':{name:{'base_mu':r['base_mean_full12'][ch],
                'previous_raw':r['previous_raw_from_current_observation_full12'][ch],
                'H':r['history_center_full12'][ch],'mu':r['conditional_mean_full12'][ch],
                'sigma':r['effective_sigma_full12'][ch],'raw_sample':r['selected_raw_full12'][ch],
                'cap':r['current_cap_full12'][ch],'projected_residual':a['projected_residual_full12'][ch],
                'actual_head_loss_gradient':stats([v['gradient'][ch] for v in h]),
                'sample_minus_current_mu':stats([v['sample_minus_current_mu'][ch] for v in h])}
                for ch,name in CHANNELS.items()}}
        assert all(abs(v['A']-item['stored_A'])<1e-7 for v in h)
        data.append(item);previous=item
    capture=next(i for i,row in enumerate(data) if row['FL_placed_history'])
    loss=next(i for i in range(capture+1,len(data)) if not data[i]['FL_top_contact'])
    air=next(i for i in range(capture+1,len(data)) if data[i]['FL_air'])
    p06=next(i for i,row in enumerate(data) if row['phase']=='P06')
    selected=sorted({capture-1,capture,capture+1,loss-1,loss,air,p06-1,p06,
                     next(i for i,r in enumerate(data) if r['tick']==3496),len(data)-1})
    windows={'P05_before_capture':[i for i,r in enumerate(data) if r['phase']=='P05' and i<capture],
             'placed_current_TOP':[i for i,r in enumerate(data) if r['FL_placed_history'] and r['FL_top_contact']],
             'P06_current_AIR':[i for i,r in enumerate(data) if r['phase']=='P06' and r['FL_air']]}
    summaries={}
    for label,indices in windows.items():
        summaries[label]={'n':len(indices),'first_tick':data[indices[0]]['tick'] if indices else None,
            'last_tick':data[indices[-1]]['tick'] if indices else None,
            **{key:stats([data[i][key] for i in indices]) for key in ('reward','potential_shaping','raw_GAE','stored_A','weighted_body_quality','geometry_cost','minimum_body_obstacle_separation_m')},
            'standardized_A':stats([v['A'] for i in indices for v in hooks[(data[i]['update'],data[i]['local'])]]),
            'channels':{name:{'actual_head_loss_gradient':stats([v['gradient'][ch] for i in indices for v in hooks[(data[i]['update'],data[i]['local'])]]),
                'mean_mu':stats([data[i]['channel'][name]['mu'] for i in indices]),
                'projected_residual':stats([data[i]['channel'][name]['projected_residual'] for i in indices])}
                for ch,name in CHANNELS.items()}}
    rollouts={}
    for u,s in snapshots.items():
        rollouts[str(u)]={'phase_counts':dict(Counter(d['phase'] for d in data if d['update']==u)),
            'terminal_count':int(s['dones'].sum()),'last_V':float(s['values'][-1,0,0]),
            'tail_bootstrap_last_value_from_return_identity':float((s['returns'][-1,0,0]-s['rewards'][-1,0,0])/.9985)
                if not bool(s['dones'][-1,0,0]) else None,
            'gamma':.9985,'lambda':.99,'GAE_trace_decay_per_decision':.9985*.99}
        raw=s['returns']-s['values']
        expected=(raw-raw.mean())/(raw.std()+1e-8)
        rollouts[str(u)]['stored_advantage_semantics']=advantage_audits[u]['stored_advantage_semantics']
        rollouts[str(u)]['raw_GAE_summary']=stats(raw.flatten().tolist())
        rollouts[str(u)]['stored_A_summary']=stats(s['advantages'].flatten().tolist())
        rollouts[str(u)]['standardization_max_abs_error']=float((expected-s['advantages']).abs().max())
    result={'scope':'only saved rollouts1508/1509 and actual logged optimizer derivatives; no forward/optimizer/physics',
        'run':str(RUN.relative_to(ROOT)),
        'sealed_sources_sha256':{f'rollouts/{name}':hashlib.sha256((RUN/'rollouts'/name).read_bytes()).hexdigest()
            for u in (1508,1509) for name in (f'rollout_{u:06d}.pt',f'update_{u:06d}_likelihood.json')},
        'method':'PPO + LIMITED AUX + receiving-sigma exploration; all receiving multipliers are expected 1 before P10',
        'observed_receiving_multiplier_all12':1.,'rho':.9,
        'potential_reconstruction_max_abs_error':max_phi_error,'capture_row':data[capture]['learner_decision'],
        'first_logged_top_contact_loss_row':data[loss]['learner_decision'],'first_logged_AIR_row':data[air]['learner_decision'],
        'first_P06_request_row':data[p06]['learner_decision'],'rollouts':rollouts,
        'selected_rows':[data[i] for i in selected],'window_summary':summaries,
        'gradient_limit':'Derivative is actual total-loss derivative wrt network output before parameter clipping, not isolated Adam update or physical derivative. Positive loss derivative gives a negative direct local gradient-descent direction, but shared parameters/Adam/HISTORY prevent claiming the same net future mean change.',
        'boundary':'Tick4120 is decision515/update1510, outside the two selected rollouts; not analyzed. GAE1508 does not contain actual later1509 rewards but bootstraps its tail. No synthetic advantage or extra terminal credit is inserted.'}
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
