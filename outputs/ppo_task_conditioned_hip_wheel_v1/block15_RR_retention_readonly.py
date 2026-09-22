"""Bounded, no-forward analysis of one sealed real rollout and its 128 rows."""
import hashlib
import itertools
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import torch
torch.set_num_threads(1)
from wlr50_clean.ppo.semantic_supervisor import TaskStageSupervisor,load_task_spec,_current_rr_placement_usable

RUN=ROOT/'runs/ppo_task_conditioned_hip_wheel_v1/train/20260921T1528590444842Z_g649ccd906421_b7f45c711b47473b9be9eaeced962417'


def main():
    path=RUN/'rollouts/rollout_001522.pt'
    saved=torch.load(path,map_location='cpu',weights_only=False)
    with (RUN/'residual_and_projection_audit.jsonl').open() as stream:
        rows=[json.loads(line) for line in itertools.islice(stream,128)]
    assert len(rows)==128 and rows[0]['global_policy_decision']==199169 and rows[-1]['global_policy_decision']==199296
    sup=TaskStageSupervisor.__new__(TaskStageSupervisor)
    sup.spec=load_task_spec(ROOT/'configs/ppo_task_conditioned_hip_wheel_v1/stage_task_spec.yaml')
    data=[]
    for i,row in enumerate(rows):
        assert torch.equal(saved['actions'][i,0],torch.tensor(row['raw_policy_action_full12']))
        a=row['applied_audit'];task=a['semantic_task'];ev=task['physical_evaluator'];q=a['reward']
        rr=ev['current_legs']['RR'];rl=ev['current_legs']['RL'];h=ev['history']
        usable=_current_rr_placement_usable(ev)
        assert usable is task['rr_placed_currently_usable']
        assert abs(sup.physical_potential(ev)-q['potential_after'])<1e-10
        assert h['placed']['RR']
        ret=sup._current_capture_retention('RR',ev);component=.85/4*(.8+.2*ret)
        item={'learner_decision':i+1,'global_decision':row['global_policy_decision'],'tick':a['physics_tick'],
            'phase':a['phase_id'],'RR_history_placed':h['placed']['RR'],
            'RR_placed_event_tick':h.get('event_ticks',{}).get('placed',{}).get('RR'),
            'RR_current_usable':usable,'RR_placed_predicate':sup.predicate('placed_RR',ev),
            'RR_surface':rr['contact_surface'],'RR_ground':rr['ground_contact'],'RR_top':rr['top_contact'],
            'RR_air':rr['air'],'RR_within_top_xy':rr['within_top_xy'],
            'RR_current_lift_valid':rr.get('current_lift_valid'),'RR_lift_established':rr.get('lift_established'),
            'RR_front_distance_m':rr['front_distance_m'],'RR_clearance_m':rr['clearance_m'],
            'RR_outside_m':rr['top_xy_outside_distance_m'],'RR_retention':ret,'RR_Phi_component':component,
            'RR_Phi_component_PBRS':None if not data else 5*(.9985*component-data[-1]['RR_Phi_component']),
            'RL_history_lift':h['active_lift']['RL'],'RL_history_crossed':h['front_edge_crossed']['RL'],
            'RL_history_placed':h['placed']['RL'],'RL_front_distance_m':rl['front_distance_m'],
            'entry_valid':task['entry_valid'],'entry_reasons':task['entry_reasons'],
            'completion_values':task['completion_values'],'task_termination':task['termination_reason'],
            'potential_before':q['potential_before'],'potential_after':q['potential_after'],
            'potential_shaping':q['potential_shaping'],'reward':q['total'],'families':q['families'],
            'raw_GAE':float((saved['returns']-saved['values'])[i,0,0]),
            'stored_standardized_A':float(saved['advantages'][i,0,0]),
            'receiving_active_before_action':row['policy_request']['receiving_continuation_active'],
            'receiving_multiplier_before_action':row['policy_request']['receiving_sigma_multiplier_full12']}
        item['non_RR_Phi_PBRS']=None if not data else q['potential_shaping']-item['RR_Phi_component_PBRS']
        data.append(item)
    first_unusable=next((i for i,v in enumerate(data) if not v['RR_current_usable']),None)
    first_ground=next((i for i,v in enumerate(data) if v['RR_ground']),None)
    first_lift=next((i for i,v in enumerate(data) if v['RL_history_lift']),None)
    indices={0,127}
    for idx in (first_unusable,first_ground,first_lift):
        if idx is not None: indices.update((max(0,idx-1),idx))
    for tick in (6528,6776):
        indices.update(i for i,v in enumerate(data) if v['tick']==tick)
    result={'scope':'one saved real rollout1522 and bounded first128 actual request rows; no forward/update/physics',
        'run':str(RUN.relative_to(ROOT)),'rollout_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'phase_counts':{p:sum(v['phase']==p for v in data) for p in sorted({v['phase'] for v in data})},
        'terminal_count':int(saved['dones'].sum()),'selected_rows':[data[i] for i in sorted(indices)],
        'counts':{'RR_unusable':sum(not v['RR_current_usable'] for v in data),
            'RR_ground':sum(v['RR_ground'] for v in data),'entry_invalid':sum(not v['entry_valid'] for v in data),
            'receiving_sigma_active':sum(v['receiving_active_before_action'] for v in data)},
        'RR_retention_range':[min(v['RR_retention'] for v in data),max(v['RR_retention'] for v in data)],
        'reward_sum':sum(v['reward'] for v in data),
        'current_RR_not_usable_but_positive_reward':sum(not v['RR_current_usable'] and v['reward']>0 for v in data),
        'RR_retention_decrease_rows':[{'tick':v['tick'],'RR_retention_delta':v['RR_retention']-data[i-1]['RR_retention'],
            'RR_Phi_component_PBRS':v['RR_Phi_component_PBRS'],'non_RR_Phi_PBRS':v['non_RR_Phi_PBRS'],
            'global_PBRS':v['potential_shaping'],'reward':v['reward']} for i,v in enumerate(data)
            if i and v['RR_retention']<data[i-1]['RR_retention']-.015]}
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()
