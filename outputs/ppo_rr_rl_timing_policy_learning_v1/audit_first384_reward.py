"""Bounded read-only reward diagnostic; no Torch, model forward or simulation."""
import json
from pathlib import Path
from collections import Counter
from statistics import mean
from wlr50_clean.ppo.semantic_supervisor import (
    TaskStageSupervisor,_current_rr_receiver_preparation_retired,placement_predecessors_satisfied)

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/ppo_rr_rl_timing_policy_learning_v1/train/20260923T1918580156414Z_gfa4b98ed506e_73c2c47d6e63471ebbcd876e45516dd5'
OUT=Path(__file__).parent/'P07_first384_reward_audit.json'
UPDATES=(1693,1694,1695)

def lines(path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():yield json.loads(line)

def stats(values):
    values=list(values)
    return dict(count=len(values),sum=sum(values),mean=mean(values) if values else None,
        minimum=min(values) if values else None,maximum=max(values) if values else None,
        positive=sum(x>0 for x in values),negative=sum(x<0 for x in values))

def main():
    updates=[]
    for row in lines(RUN/'optimizer_updates.jsonl'):
        if row['ppo_update'] in UPDATES:updates.append(row)
        if row['ppo_update']>=UPDATES[-1]:break
    assert [r['ppo_update'] for r in updates]==list(UPDATES)
    summaries=[]
    for row in lines(RUN/'advantage_audit.jsonl'):
        if row['ppo_update_intended'] in UPDATES:
            assert row['sample_count']==128 and not row['teacher_prefix_samples_included']
            summaries.append(dict(update=row['ppo_update_intended'],overall=row['overall'],
                by_request_phase=row['by_request_phase'],tail_bootstrap=row['tail_bootstrap']))
        if row['ppo_update_intended']>=UPDATES[-1]:break
    advantages={}
    for update in UPDATES:
        audit=json.loads((RUN/f'rollouts/update_{update:06d}_likelihood.json').read_text())
        found={};exposure=Counter()
        for batch in audit['minibatches']:
            for indices,adv in zip(batch['rollout_flat_indices'],batch['actual_advantage']):
                assert len(indices)==1
                index=indices[0];exposure[index]+=1
                if index in found:assert found[index]==adv
                found[index]=adv
        assert exposure==Counter({i:5 for i in range(128)})
        for index,value in found.items():advantages[221057+(update-UPDATES[0])*128+index]=value
    supervisor=TaskStageSupervisor(ROOT/'configs/ppo_rr_rl_timing_policy_learning_v1/stage_task_spec.yaml')
    spec=supervisor.spec;rows=[];previous=None;max_phi_error=0.
    for raw in lines(RUN/'residual_and_projection_audit.jsonl'):
        global_step=raw['global_policy_decision']
        if global_step>221440:break
        a=raw['applied_audit'];ev=a['semantic_task']['physical_evaluator'];rr=ev['current_legs']['RR'];h=ev['history'];reward=a['reward_breakdown']
        assert global_step in advantages
        phi=supervisor.physical_potential(ev)
        max_phi_error=max(max_phi_error,abs(phi-reward['potential_after']))
        qualified=rr.get('current_lift_valid') is True
        cross=h['front_edge_crossed']['RR'] is True;placed=h['placed']['RR'] is True
        part={}
        if not placed and placement_predecessors_satisfied(spec,h,'RR'):
            workspace=supervisor._workspace_potential_progress('RR',ev)
            hard=h['active_lift']['RR'];unload=ev.get('transfer_roles',{}).get('RR',{}).get('transfer_progress',0.)
            if hard and cross:unload=1.
            carry=(1. if cross else max(0.,min(1.,1+rr['front_distance_m']/.25))) if hard else 0.
            capture=supervisor._current_capture_progress('RR',ev,min(1.,rr['consecutive_top_samples']/spec['history']['minimum_top_samples']) if cross else 0.)
            initial=float(rr['initial_clearance']);lift=supervisor._current_lift_credit('RR',ev)
            part={k:.85/4*v for k,v in dict(workspace=.1*workspace,unload=.1*unload,
                initial=.25*.25*initial,lift=.25*.75*lift,carry=.35*carry,capture=.2*capture).items()}
        row=dict(global_decision=global_step,time_s=a['sim_time_s'],tick=a['physics_tick'],
            request_phase='P%02d'%(raw['policy_request']['stage_index']+1),end_phase=a['end_phase_id'],
            RR_Q=qualified,RR_history_Q=h['active_lift']['RR'],RR_cross=cross,RR_placed=placed,
            RR_TOP=rr['top_surface_contact'] and rr['top_contact'],RR_ground=rr['ground_contact'],
            RR_air=rr['air'],RR_xy=rr['within_top_xy'],RR_front_m=rr['front_distance_m'],RR_gap_m=rr['clearance_m'],
            RR_receiver_retired=_current_rr_receiver_preparation_retired(spec,'RR',ev),
            reward=raw['reward'],weighted_families=reward['families'],unweighted_families=reward['unweighted_families'],
            phi_before=reward['potential_before'],phi_after=reward['potential_after'],potential=reward['potential_shaping'],
            event=reward['terminal_event'],time_cost=-.02*reward['elapsed_physics_s'],
            GAE_standardized=advantages[global_step],RR_phi_parts=part,terminal=raw['terminal'])
        row['capture_window']=bool(cross and not placed and rr['within_top_xy'] and not rr['ground_contact'])
        if previous and a['physics_tick']==previous['tick']+a['physics_ticks']:
            row['gap_delta_m']=row['RR_gap_m']-previous['RR_gap_m']
            row['RR_Q_lost']=previous['RR_Q'] and not qualified
            row['RR_receiver_retirement_lost']=previous['RR_receiver_retired'] and not row['RR_receiver_retired']
            row['RR_phi_part_deltas']={k:v-previous['RR_phi_parts'].get(k,0) for k,v in part.items()}
        else:row['gap_delta_m']=None;row['RR_Q_lost']=False;row['RR_receiver_retirement_lost']=False
        rows.append(row);previous=row
    assert len(rows)==384 and rows[0]['global_decision']==221057 and rows[-1]['global_decision']==221440
    def summary(data):
        return dict(samples=len(data),reward=stats(r['reward'] for r in data),
            potential_shaping=stats(r['potential'] for r in data),standardized_GAE=stats(r['GAE_standardized'] for r in data),
            weighted_family_sums={k:sum(r['weighted_families'][k] for r in data) for k in rows[0]['weighted_families']},
            unweighted_smoothness=stats(r['unweighted_families']['control_smoothness'] for r in data),
            gap_m=stats(r['RR_gap_m'] for r in data),top=sum(r['RR_TOP'] for r in data),placed=sum(r['RR_placed'] for r in data),
            currentQ=sum(r['RR_Q'] for r in data),crossed=sum(r['RR_cross'] for r in data),
            receiver_retired=sum(r['RR_receiver_retired'] for r in data))
    capture=[r for r in rows if r['capture_window']]
    descending=[r for r in capture if r['gap_delta_m'] is not None and r['gap_delta_m']<0]
    losing=[r for r in rows if r['RR_Q_lost'] or r['RR_receiver_retirement_lost']]
    result=dict(scope='first_three_completed_actual_updates_only; no pending fourth samples; no model forward',
        run=str(RUN),updates=list(UPDATES),first_decision=221057,last_decision=221440,
        actual_phase_samples=dict(Counter(r['request_phase'] for r in rows)),all_samples=summary(rows),
        capture_window=summary(capture),descending_capture=summary(descending),
        exact_potential_recompute_max_abs_error=max_phi_error,
        lost_qualification_or_retirement=losing,most_negative_potential=sorted(rows,key=lambda r:r['potential'])[:6],
        descending_examples=descending[:3]+descending[-3:],update_advantage_summaries=summaries,
        source_code='frozen fa4b98ed semantic_supervisor physical_potential/current_lift_credit/current_capture_progress + semantic_reward',
        limitations='No contact not yet observed is assigned hypothetical advantage; standardized GAE is rollout/value-dependent, not isolated causal reward.')
    OUT.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('actual_phase_samples','all_samples','capture_window','descending_capture','exact_potential_recompute_max_abs_error')},indent=2))
    print('lost_q_or_retirement',[(r['global_decision'],r['RR_Q_lost'],r['RR_receiver_retirement_lost']) for r in losing])

if __name__=='__main__':main()
