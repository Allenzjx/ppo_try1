"""One sealed rollout's recorded-state potential audit; no model forward/update."""
from copy import deepcopy
import hashlib
import itertools
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from wlr50_clean.ppo.semantic_supervisor import TaskStageSupervisor,load_task_spec,RR_WORKSPACE_RETIREMENT_MODE,RR_WORKSPACE_RETIREMENT_MODE_V2,_current_rr_receiver_preparation_retired
RUN=ROOT/'runs/ppo_p05_hip_only_continuation_v1/train/20260922T1428131775302Z_g336b7c56d2f0_23b26a4c40a24a178d190def03381413'
path=RUN/'rollouts/rollout_001648.pt'
rollout=torch.load(path,map_location='cpu',weights_only=False)
with (RUN/'residual_and_projection_audit.jsonl').open('rb') as stream:
    lines=list(itertools.islice(stream,895,1024))
rows=[json.loads(x) for x in lines]
assert len(rows)==129 and rows[0]['global_policy_decision']==215296 and rows[-1]['global_policy_decision']==215424
assert all(not r['terminal'] for r in rows)
spec=load_task_spec(ROOT/'configs/ppo_p05_hip_only_continuation_v1/stage_task_spec.yaml')
assert spec['rr_postcross_workspace_semantics']==RR_WORKSPACE_RETIREMENT_MODE_V2
spec_relative='configs/ppo_p05_hip_only_continuation_v1/stage_task_spec.yaml'
spec_sha256=hashlib.sha256((ROOT/spec_relative).read_bytes()).hexdigest()
assert rollout['runtime_contract']['files'][spec_relative]==spec_sha256
old=deepcopy(spec);old['rr_postcross_workspace_semantics']=RR_WORKSPACE_RETIREMENT_MODE
supervisors=[]
for cfg in (old,spec):
    s=object.__new__(TaskStageSupervisor);s.spec=cfg;supervisors.append(s)
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def ev(r):return r['applied_audit']['semantic_task']['physical_evaluator']
def state(e):
    before=digest(e);rr=e['current_legs']['RR'];history=e['history']
    phis=[s.physical_potential(e) for s in supervisors]
    retired=[_current_rr_receiver_preparation_retired(c,'RR',e) for c in (old,spec)]
    frac=min(1.,rr['consecutive_top_samples']/spec['history']['minimum_top_samples']) if history['front_edge_crossed']['RR'] else 0.
    capture=supervisors[1]._current_capture_progress('RR',e,frac)
    assert digest(e)==before
    return dict(tick=e['physics_tick'],currentQ=rr['current_lift_valid'],historyQ=history['active_lift']['RR'],cross=history['front_edge_crossed']['RR'],placed=history['placed']['RR'],
        body_control=rr['body_control_evidence'],lift_established=rr['lift_established'],continuation_allowed=rr['motion_continuation_allowed'],
        air=rr['air'],ground=rr['ground_contact'],top_contact=rr['top_contact'],top_surface_contact=rr['top_surface_contact'],
        top_count=rr['consecutive_top_samples'],within_top_xy=rr['within_top_xy'],within_lateral_span=rr['within_lateral_span'],
        gap_m=rr['clearance_m'],front_distance_m=rr['front_distance_m'],
        ground_relative_lift_m=rr['ground_relative_lift_m'],retired_v1=retired[0],retired_v2=retired[1],phi_v1=phis[0],phi_v2=phis[1],
        phi_delta=phis[1]-phis[0],RR_capture_progress=capture,RR_capture_phi_share=.85/4*.2*capture,
        receiver_progress=e['transfer_roles']['RR']['workspace_progress'])
states=[state(ev(r)) for r in rows]
records=[]
for i,r in enumerate(rows[1:]):
    a=r['applied_audit']; reward=a['reward'];b,c=states[i:i+2]
    assert a['decision_count']==rows[i]['applied_audit']['decision_count']+1
    assert a['physics_tick']-rows[i]['applied_audit']['physics_tick']==8
    for field,value in [('actions',r['raw_policy_action_full12']),('rewards',[r['reward']]),('actions_log_prob',[r['old_log_probability']])]:
        assert torch.equal(rollout[field][i,0],torch.tensor(value,dtype=rollout[field].dtype))
    assert not rollout['dones'][i,0].item()
    assert abs(b['phi_v2']-reward['potential_before'])<1e-12 and abs(c['phi_v2']-reward['potential_after'])<1e-12
    assert abs(5*(.9985*c['phi_v2']-b['phi_v2'])-reward['potential_shaping'])<1e-12
    assert abs(reward['total']-r['reward'])<3e-7
    assert r['policy_request']['current_RR_qualification']==b['currentQ']
    gap_delta=c['gap_m']-b['gap_m']
    direction=('cross_event' if not b['cross'] and c['cross'] else
        ('descend' if gap_delta < -1e-9 else 'ascend' if gap_delta>1e-9 else 'flat') if b['cross'] and c['cross'] and b['air'] and c['air'] else 'other')
    records.append(dict(global_decision=r['global_policy_decision'],phase=a['phase_id'],input=b,next=c,direction=direction,gap_delta_m=gap_delta,
        reward_total=reward['total'],recorded_shaping=reward['potential_shaping'],RR_capture_shaping=5*(.9985*c['RR_capture_phi_share']-b['RR_capture_phi_share']),
        v2_minus_v1_shaping=5*(.9985*c['phi_delta']-b['phi_delta']),
        time_cost=-.02*reward['elapsed_physics_s'],terminal_event=reward['terminal_event'],weighted_families=reward['families'],
        geometry_cost=reward['cost_components']['task_space_weighted_geometry_cost'],
        saved_advantage=rollout['advantages'][i,0,0].item()))
def stats(v):return dict(n=len(v),minimum=min(v) if v else None,maximum=max(v) if v else None,sum=sum(v),mean=sum(v)/len(v) if v else None)
def state_counts(ss):return dict(n=len(ss),currentQ_true=sum(s['currentQ'] for s in ss),currentQ_false=sum(not s['currentQ'] for s in ss),cross=sum(s['cross'] for s in ss),
    retired_v1=sum(s['retired_v1'] for s in ss),retired_v2=sum(s['retired_v2'] for s in ss),gate_false_to_true=sum(not s['retired_v1'] and s['retired_v2'] for s in ss),
    phi_nonzero=sum(s['phi_delta']!=0 for s in ss),phi_delta=stats([s['phi_delta'] for s in ss]),top_contact=sum(s['top_contact'] for s in ss),top_surface=sum(s['top_surface_contact'] for s in ss),placed=sum(s['placed'] for s in ss))
groups={}
for name in ['cross_event','descend','ascend','flat','other']:
    selected=[r for r in records if r['direction']==name]
    groups[name]=dict(count=len(selected),**{k:stats([r[k] for r in selected]) for k in ['gap_delta_m','RR_capture_shaping','recorded_shaping','v2_minus_v1_shaping','reward_total','geometry_cost','time_cost']})
result=dict(schema='wlr50_clean.block09_rollout1648_receiver_v2_readonly.v1',run=str(RUN),rollout=str(path),rollout_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    selected_audit_129_lines_sha256=hashlib.sha256(b''.join(lines)).hexdigest(),rollout_update=1648,global_decisions=[215297,215424],input_ticks=[states[0]['tick'],states[-2]['tick']],next_ticks=[states[1]['tick'],states[-1]['tick']],
    source_head=rollout['runtime_contract']['source_git_commit'],spec_sha256=spec_sha256,rollout_runtime_binds_actual_v2_spec_bytes=True,
    sample_count=128,actual_input_counts=state_counts(states[:-1]),next_state_counts=state_counts(states[1:]),
    postcross_input_counts=state_counts([s for s in states[:-1] if s['cross']]),postcross_next_counts=state_counts([s for s in states[1:] if s['cross']]),
    stored_actions_rewards_logp_dones_matched=True,recorded_before_after_Phi_and_PBRS_match_v2=True,currentQ_request_matches_preceding_actual_state=True,
    physical_state_objects_unchanged=True,real_RR_history_events=ev(rows[-1])['history']['event_ticks'],groups=groups,
    RR_gap_range_m=stats([s['gap_m'] for s in states]),postcross_gap_range_m=stats([s['gap_m'] for s in states if s['cross']]),
    weighted_quality_family_totals={k:sum(r['weighted_families'][k] for r in records) for k in records[0]['weighted_families'] if k!='task_progress'},
    aggregate_v2_minus_v1_shaping=stats([r['v2_minus_v1_shaping'] for r in records]),
    saved_RR_obstacle_contact_endpoints=[dict(global_decision=r['global_policy_decision'],tick=ev(r)['physics_tick'],
        **{k:ev(r)['current_legs']['RR'][k] for k in ['air','ground_contact','top_contact','top_surface_contact','contact_surface',
        'obstacle_pair_active','contact_reaction_force_n','bearing_force_n','support','within_top_xy','clearance_m','front_distance_m','consecutive_top_samples','placed_on_top']})
        for r in rows if ev(r)['current_legs']['RR']['obstacle_pair_active']],
    counterfactual_only='v1 is evaluated on exactly the stored v2 physical states; not a v1 rollout, new PPO data or reconstructed GAE.',
    all128_selected_metrics_digest=digest(records),
    representative_records=[r for i,r in enumerate(records) if i in {0,127} or r['direction']=='cross_event'
        or i==min(range(128),key=lambda j:records[j]['RR_capture_shaping'])
        or i==max(range(128),key=lambda j:records[j]['RR_capture_shaping'])],
    changed_gate_input_ticks=[s['tick'] for s in states[:-1] if s['retired_v1']!=s['retired_v2']],
    nonzero_phi_delta_input_ticks=[s['tick'] for s in states[:-1] if s['phi_delta']!=0])
print(json.dumps(result,indent=2,allow_nan=False))
