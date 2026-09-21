"""Bounded sealed block06/07 JSONL reduction, CPU stdlib only; no policy forwards."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
LEGS=('FL','FR','RL','RR')

def slim(a):
    x=a['applied_audit'];s=x['semantic_task'];p=s['physical_evaluator'];n=x['actuator_target_effect_audit']
    pol=a['policy_request'];tr=n.get('tracking_reference_evidence',{});hr=n.get('policy_headroom_evidence',{})
    rr=s['transfer_roles']['RR']['transfer_direction_context']
    legs=p['current_legs'];diag=s['nominal_provider_diagnostics']
    return {'decision':a['global_policy_decision'],'tick':x['physics_tick'],'time_s':x['sim_time_s'],
      'phase':x['phase_id'],'end_phase':x['end_phase_id'],'terminal':x['termination_reason'],
      'events':s['history']['event_ticks'],'stage_age_s':s['stage_age_s'],
      'legs':{leg:{k:legs[leg].get(k) for k in ('support','contact_surface','bearing_force_n','air','clearance_m','front_distance_m','current_lift_valid','free_air_lift_gain_m')} for leg in LEGS},
      'support_count':s['goal_features']['support_count'],'body_speed':s['goal_features']['body_linear_speed_m_s'],
      'body_omega':s['goal_features']['body_angular_speed_rad_s'],'body_forward_m':s['goal_features']['body_forward_m'],
      'com_world':rr.get('mass_weighted_com_position_w_m'),'com_velocity':rr.get('mass_weighted_com_velocity_w_m_s'),
      'rr_transfer_com_toward_FL_m':rr.get('com_toward_receiver_m'),'rr_receiver_support':s['transfer_roles']['RR']['receiver_workspace_state']['receiver_current_support'],
      'nominal':x['nominal_action_full12'],'native_nominal':n['native_drive_target_full12'],
      'controller':n['controller_drive_bias_full12'],'raw':a['raw_policy_action_full12'],
      'base':pol['base_mean_full12'],'mean':pol['conditional_mean_full12'],'sigma':pol['effective_sigma_full12'],
      'center':pol['history_center_full12'],'gate':pol['cap_transition_gate_full12'],
      'innovation':[v-m for v,m in zip(a['raw_policy_action_full12'],pol['conditional_mean_full12'])],
      'request':x['projected_residual_full12'],'headroom_effective':hr.get('effective_policy_residual_full12'),
      'final':x['actual_drive_target_full12'],
      'actual_servo_at_last_dispatch':[c['nominal_deg']-c['current_actual_canonical_error_deg'] for c in tr.get('channels',[])],
      'last_dispatch_tick':n['physics_tick'],'wheel_actual':p['measured_wheel_velocity_rad_s'],
      'wheel_actual_order':p['stop_progress_wheel_order'],
      'verified_dispatch':all(n.get(k) is True for k in ('verified','setter_dispatch_targets_equal','actual_mapping_matches_dispatch')),
      'nominal_owners':{'p06_rolling_status':diag.get('p06_rolling_retirement',{}).get('status'),
        'p06_wheel_gain':diag.get('p06_rolling_retirement',{}).get('wheel_gain'),
        'p06_wheel_tail_status':diag.get('p06_wheel_tail',{}).get('status'),
        'capture_hold':{leg:{k:v.get(k) for k in ('observed_capture_tick','held_nominal_servo_deg','retired_preexisting_layers')}
          for leg,v in diag.get('capture_owner_hold',{}).get('captures',{}).items()}},
      'reward':a['reward'],'old_value':a['old_value']}

def main():
    result={'scope':{'sealed_blocks':[6,7],'policy_forwards':0,'gae_recomputed':False,'physics_started':False,
      'physical_granularity':'decision-end observations at15Hz; actual servo is last dispatch pretick sample',
      'not_causal_intervention':True},'interpretation':{
      'first_FL_support_loss_is_not_itself_task_failure':True,
      'legal_FL_AIR_and_RR_unloading_may_be_useful_preparation':True,
      'confirmed_failures':'four later P09 physical terminals; no RR cross/place in these terminated episodes',
      'common_earliest_harmful_action_or_single_channel_cause_confirmed':False,
      'fixed_FL_support_reward_or_new_gate_recommended':False,
      'course_suggestion':'after natural P01 evaluation, retain capture predecessor through P06 preparation and downstream P09 outcome credit; do not require FL support continuously'},'blocks':[]}
    for block in (6,7):
      receipt=json.loads((OUT/f'block{block:02}_768_receipt.json').read_text())
      run=Path(receipt['run']);manifest=json.loads((run/'run_manifest.json').read_text())
      assert manifest['lifecycle']=='SUCCEEDED'
      episodes=[[]]
      with (run/'residual_and_projection_audit.jsonl').open() as stream:
       for line in stream:
        a=json.loads(line);row=slim(a);episodes[-1].append(row)
        if row['terminal']:episodes.append([])
      summary={'block':block,'run':str(run),'coverage':receipt['actual_request_phase_coverage'],'episodes':[]}
      for index,rows in enumerate(episodes):
       if not rows:continue
       capture=next((i for i,r in enumerate(rows) if r['events']['placed'].get('FL')),None)
       p06=next((i for i,r in enumerate(rows) if r['phase']=='P06'),None)
       p09=next((i for i,r in enumerate(rows) if r['phase']=='P09'),None)
       after=range(capture or 0,len(rows)) if capture is not None else ()
       firstloss=next((i for i in after if not rows[i]['legs']['FL']['support']),None)
       sustained=next((i for i in after if i+2<len(rows) and all(not r['legs']['FL']['support'] for r in rows[i:i+3])),None)
       few=next((i for i in after if rows[i]['support_count']<3),None)
       fast=next((i for i in after if rows[i]['body_omega']>.5),None)
       selected=set()
       for i in (capture,p06,p09,firstloss,sustained,few,fast,len(rows)-1):
        if i is not None:selected.add(i)
       if firstloss is not None:selected.update(range(max(0,firstloss-1),min(len(rows),firstloss+2)))
       if capture is not None and capture>0:selected.add(capture-1)
       points={name:None if i is None else {'decision':rows[i]['decision'],'tick':rows[i]['tick'],'time_s':rows[i]['time_s']}
         for name,i in [('capture',capture),('P06_first',p06),('first_FL_support_loss',firstloss),('first_3decision_FL_loss',sustained),('first_support_below3',few),('first_omega_above_0.5',fast),('P09_first',p09)]}
       points['terminal']={k:rows[-1][k] for k in ('decision','tick','time_s','terminal','phase')}
       phase_summary={}
       for phase in ('P05','P06','P09'):
        rs=[r for r in rows if r['phase']==phase]
        if rs:phase_summary[phase]={'n':len(rs),'FL_support_samples':sum(r['legs']['FL']['support'] for r in rs),
          'support_below3':sum(r['support_count']<3 for r in rs),'max_omega':max(r['body_omega'] for r in rs),
          'sigma_max_full12':[max(r['sigma'][j] for r in rs) for j in range(12)],
          'raw_abs_max_full12':[max(abs(r['raw'][j]) for r in rs) for j in range(12)]}
       progress=None if p06 is None or p09 is None else {
         'RR_front_distance_gain_P06_to_P09_m':rows[p09]['legs']['RR']['front_distance_m']-rows[p06]['legs']['RR']['front_distance_m'],
         'body_forward_gain_P06_to_P09_m':rows[p09]['body_forward_m']-rows[p06]['body_forward_m'],
         'RR_bearing_first_P06_N':rows[p06]['legs']['RR']['bearing_force_n'],
         'RR_bearing_at_first_FL_support_loss_N':None if firstloss is None else rows[firstloss]['legs']['RR']['bearing_force_n']}
       summary['episodes'].append({'episode_index':index,'sample_count':len(rows),'events':rows[-1]['events'],
         'downstream_progress_despite_FL_AIR':progress,
         'points':points,'phase_summary':phase_summary,'all_dispatch_verified':all(r['verified_dispatch'] for r in rows),
         'raw_abs_max_all_samples':max(abs(v) for r in rows for v in r['raw']),
         'tanh_abs_at_least_0_95_channel_samples':sum(abs(v)>=1.831780823064823 for r in rows for v in r['raw']),
         'rows':[rows[i] for i in sorted(selected)]})
      result['blocks'].append(summary)
    path=OUT/'next_failure_course_diagnosis.json'
    path.write_text(json.dumps(result,separators=(',',':'))+'\n')
    print(json.dumps({'output':str(path),'bytes':path.stat().st_size,'episodes':[
      {'block':b['block'],'episode':e['episode_index'],'max_raw':e['raw_abs_max_all_samples'],
       'saturated_channel_samples':e['tanh_abs_at_least_0_95_channel_samples'],'dispatch_verified':e['all_dispatch_verified']}
       for b in result['blocks'] for e in b['episodes']]}))

if __name__=='__main__':main()
