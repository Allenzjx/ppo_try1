"""Bounded JSON-only audit of two completed updates. No imports/runs/writes."""
import collections
import itertools
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'runs/ppo_rr_rl_timing_policy_learning_v1/train/20260923T1918580156414Z_gfa4b98ed506e_73c2c47d6e63471ebbcd876e45516dd5'
with (RUN/'residual_and_projection_audit.jsonl').open() as stream:
    rows = [json.loads(line) for line in itertools.islice(stream, 256)]
assert len(rows) == 256
assert [r['global_policy_decision'] for r in rows] == list(range(221057,221313))
with (RUN/'optimizer_updates.jsonl').open() as stream:
    updates = [json.loads(line) for line in itertools.islice(stream, 2)]
assert [r['global_policy_decisions'] for r in updates] == [221184,221312]

def span(values):
    values=list(values)
    return [min(values), max(values)]

def extract(row):
    a=row['applied_audit']; e=a['actuator_target_effect_audit']; q=row['policy_request']
    s=a['semantic_task']; legs=s['physical_evaluator']['current_legs']; rr=legs['RR']
    layers=s['nominal_provider_diagnostics']['source_partial_order']['layers']
    p09=next((v for v in layers if v['stage']=='P09'),{})
    tr=e['tracking_reference_evidence']
    actual=[v['nominal_deg']-v['current_actual_canonical_error_deg'] for v in tr['channels']]
    def channel(i):
        return dict(raw=row['raw_policy_action_full12'][i], mu=row['old_distribution_mean_full12'][i],
            learned_sigma=q['learned_sigma_full12'][i], effective_sigma=row['old_distribution_std_full12'][i],
            previous_raw=q['previous_raw_from_current_observation_full12'][i], source_N=a['nominal_action_full12'][i],
            mapped_N=e['native_drive_target_full12'][i], requested_residual=e['policy_headroom_evidence']['requested_policy_residual_full12'][i],
            projected_residual=e['policy_headroom_evidence']['effective_policy_residual_full12'][i],
            final=a['actual_drive_target_full12'][i], actual_pre_final_tick=actual[i] if i<8 else None)
    fl=channel(8); fl['actual_end_tick']=s['physical_evaluator']['measured_wheel_velocity_rad_s'][0]
    fl['native_target_rad_s']=e['actual_native_targets']['wheel_velocity_rad_s'][0]
    fl['contact']={k:legs['FL'].get(k) for k in ['contact_mode','contact_surface','top_contact','ground_contact','air','support']}; fl['bearing_n']=legs['FL']['bearing_force_n']
    return dict(decision=row['global_policy_decision'],t=a['sim_time_s'],tick=a['physics_tick'],phase=a['phase_id'],
        RR_hip=channel(6),RR_knee=channel(7),FL_wheel=fl,
        actual_mapper_previous_ack_tick=tr['previous_ack_physics_tick'],
        rr={k:rr[k] for k in ['front_distance_m','clearance_m','within_top_xy','contact_mode','top_contact','support','bearing_force_n','bearing_verified','current_lift_valid','front_edge_crossed','placed_on_top','observed_other_support_contacts']},
        source_p09={k:p09.get(k) for k in ['status','source_ticks','wait_reason','late_group_source_tick']},
        rear_features=q['rear_policy_timing_observed_features'],
        local_sigma_multipliers=q['rear_local_sigma_multiplier_full12'],
        other_final=dict(FL_hip=a['actual_drive_target_full12'][0],FL_knee=a['actual_drive_target_full12'][1],FR_knee=a['actual_drive_target_full12'][3],RL_hip=a['actual_drive_target_full12'][4],RL_knee=a['actual_drive_target_full12'][5]),
        other_source=dict(FL_hip=a['nominal_action_full12'][0],FL_knee=a['nominal_action_full12'][1],FR_knee=a['nominal_action_full12'][3],RL_hip=a['nominal_action_full12'][4],RL_knee=a['nominal_action_full12'][5]))

samples=[extract(r) for r in rows]
waiting=[r for r in samples if r['source_p09']['wait_reason']=='current_RR_bearing_before_FL_RL_transfer']
qual=[r for r in samples if r['rr']['current_lift_valid']]
cross=[r for r in samples if r['rr']['front_edge_crossed']]
sigma=[r for r in samples if r['local_sigma_multipliers'][6]==4.]
stopped_reverse=[r for r in samples if r['FL_wheel']['source_N']==0. and r['FL_wheel']['final']<0.]
pick={0,1,2,127,255}
for group in [waiting,qual,cross,sigma,stopped_reverse]:
    if group: pick.add(group[0]['decision']-221057)
def ranges(group):
    return dict(count=len(group),
        RR_hip={k:span(r['RR_hip'][k] for r in group) for k in ['raw','mu','learned_sigma','effective_sigma','source_N','mapped_N','requested_residual','projected_residual','final','actual_pre_final_tick']},
        RR_knee={k:span(r['RR_knee'][k] for r in group) for k in ['raw','mu','effective_sigma','source_N','mapped_N','requested_residual','projected_residual','final','actual_pre_final_tick']},
        RR_gap_m=span(r['rr']['clearance_m'] for r in group),
        FL_wheel={k:span(r['FL_wheel'][k] for r in group) for k in ['source_N','requested_residual','final','actual_end_tick']},
        other_final={k:span(r['other_final'][k] for r in group) for k in group[0]['other_final']})
old_error=0.
for r in rows:
    q=r['policy_request']; e=r['applied_audit']['actuator_target_effect_audit']
    assert q['rear_task_assists_enabled'] is False and q['rr_capture_assist_observed_features']==[0.]*14
    assert not any(e['capture_assist_owned_channels_full12'][6:8])
    assert 'rr_capture_assist_evidence' not in e and 'rr_carry_wheel_evidence' not in e
    assert e['controller_drive_bias_full12']==[0.]*12 and e['phase_mask_full12']==[1]*12
    assert e['setter_dispatch_targets_equal'] and e['actual_mapping_matches_dispatch']
    assert not r['terminal'] and not r['applied_audit']['prefix_teacher_data_in_ppo_storage']
    assert q['sampling_draws']==1 and q['extra_model_forwards']==q['extra_random_draws']==0
    raw=r['raw_policy_action_full12']; mu=r['old_distribution_mean_full12']; std=r['old_distribution_std_full12']
    lp=sum(-.5*((x-m)/s)**2-math.log(s)-.5*math.log(2*math.pi) for x,m,s in zip(raw,mu,std))
    old_error=max(old_error,abs(lp-r['old_log_probability']))
transitions=[]
for r in rows[:3]:
    for t in r['applied_audit']['stage_transition_evidence']:
        transitions.append({k:t[k] for k in ['from_stage','to_stage','sim_time_s','physics_tick','reason','entry','completion_values']}|{'RR_history':{k:t['history'][k]['RR'] for k in ['active_lift','front_edge_crossed','placed']},'terminal':r['terminal']})
result=dict(schema='wlr50_clean.P07_suffix_first256_readonly.v1',run=str(RUN),
    decision_range=[221057,221312],scope='two completed updates only, successful nominal prefix, not natural P01 current policy',
    counts=dict(valid_decisions=256,completed_PPO_updates=2,optimizer_steps=sum(u['optimizer_steps'] for u in updates),phase_samples=dict(collections.Counter(r['applied_audit']['phase_id'] for r in rows))),
    prefix=rows[0]['applied_audit']['curriculum_start'],updates=updates,
    checks=dict(rear_assists_false=256,RR_WAIT14_zero=256,no_rear_task_owner=256,controller_drive_bias_zero=256,all12mask=256,verified_dispatch=256,normal_transitions_not_done=True,teacher_prefix_in_storage=False,old_logp_max_abs_error=old_error),
    transitions=transitions,ranges_all=ranges(samples),ranges_late_wait=ranges(waiting),
    first_events={k:(v[0]['decision'] if v else None) for k,v in [('qualified',qual),('crossed',cross),('sigma_x4',sigma),('late_wait',waiting),('FL_source_stop_policy_reverse',stopped_reverse)]},
    RR_TOP_samples=sum(r['rr']['top_contact'] for r in samples),RR_placed_samples=sum(r['rr']['placed_on_top'] for r in samples),
    selected_samples=[samples[i] for i in sorted(pick)],
    timing='RR actual comes from independently recorded pre-dispatch sensor reference, labelled by mapper previous_ack tick, not episode tick. RR gap/contact and FL measured speed are end-of-decision episode physics_tick. Units servo degrees, wheel canonical forward-positive rad/s unless labelled native.')
print(json.dumps(result,ensure_ascii=False,indent=2))
