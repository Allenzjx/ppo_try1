"""Same-tick sealed tail diagnostics only; never smooth or change acceptance."""
import json
from pathlib import Path

FOLDER=Path(__file__).resolve().parent/'zero_refine_3d23189_run034324'
data=json.loads((FOLDER/'zero_review_aggregate.json').read_text(encoding='utf8'))
home_tick=data['source_home_entry']['entry_observation_tick']
end=data['native_tick_count']
physical={r['tick']:r for r in data['last_two_seconds_physical']}
target=data['source_home_target_servo_deg']
legs=['FL','FR','RL','RR']


def intervals(ticks):
    output=[]
    for tick in ticks:
        if output and output[-1]['last_tick']==tick-1:output[-1]['last_tick']=tick
        else:output.append({'first_tick':tick,'last_tick':tick})
    for row in output:
        row.update(first_time_s=row['first_tick']/120,last_time_s=row['last_tick']/120,
            sample_count=row['last_tick']-row['first_tick']+1,
            sample_exposure_s=(row['last_tick']-row['first_tick']+1)/120,
            endpoint_elapsed_s=(row['last_tick']-row['first_tick'])/120)
    return output


def controlled(row):
    return row['body_speed_m_s']<=.05 and row['body_angular_speed_rad_s']<=.30 and max(map(abs,row['actual_full12'][8:]))<=.25


windows={}
for name,start in [('final_1s',end-119),('final_0_5s',end-59)]:
    selected=[physical[t] for t in range(start,end+1)]
    wheels={}
    for i,leg in enumerate(legs):
        breach=[r['tick'] for r in selected if abs(r['actual_full12'][8+i])>.25]
        wheels[leg]={'exceedance_count':len(breach),'sample_count':len(selected),'threshold_rad_s':.25,
            'maximum_abs_rad_s':max(abs(r['actual_full12'][8+i]) for r in selected),'intervals':intervals(breach)}
    windows[name]={'first_tick':start,'last_tick':end,'sample_count':len(selected),'wheels':wheels,
        'linear_speed_above_0_05_count':sum(r['body_speed_m_s']>.05 for r in selected),
        'angular_speed_above_0_30_count':sum(r['body_angular_speed_rad_s']>.30 for r in selected),
        'all_three_measured_control_thresholds_pass_count':sum(controlled(r) for r in selected)}
after=[physical[t] for t in range(home_tick+1,end+1)]
control_segments=intervals([r['tick'] for r in after if controlled(r)])
lost_segments=intervals([r['tick'] for r in after if not controlled(r)])
trace=[]
for tick in range(home_tick,end+1,12):
    r=physical[tick]; error=[q-n for q,n in zip(r['actual_full12'][:8],target)]
    trace.append({'tick':tick,'time_s':r['time_s'],'time_since_home_entry_s':(tick-home_tick)/120,
        'actual_servo_deg':r['actual_full12'][:8],'actual_minus_source_home_deg':error,
        'max_abs_source_home_error_deg':max(map(abs,error)),
        'wheel_qd_rad_s':r['actual_full12'][8:],
        'body_speed_m_s':r['body_speed_m_s'],'body_angular_speed_rad_s':r['body_angular_speed_rad_s'],
        'final_controlled_thresholds':controlled(r)})
first_saved={}; previous=None
for line in (Path(data['source'])/'video_policy_decisions.jsonl').open(encoding='utf8'):
    row=json.loads(line); ev=(row.get('step_info') or {}).get('semantic_task',{}).get('physical_evaluator',{})
    for key in ('final_region_valid','final_support_available','post_completion_observation_started'):
        if ev.get(key) is True and key not in first_saved:
            first_saved[key]={'first_saved_evaluator_tick':ev.get('physics_tick'),
                'previous_saved_evaluator_tick':None if previous is None else previous.get('physics_tick'),
                'previous_saved_value':None if previous is None else previous.get(key),
                'sampling':'actual saved policy decision endpoints; intervening event tick not invented'}
    if ev:previous=ev
receipt={'schema':'wlr50_clean.zero_source_home_tail_control.v1',
    'source':data['source'],'recorded_outcome':data['source_physical_episode']['physical_task_evaluation']['termination_reason'],
    'traversal_event_time_s':data['source_physical_episode']['physical_task_evaluation']['traversal_event_time_s'],
    'first_saved_conditions':first_saved,'home_entry_observation_tick':home_tick,
    'first_native_home_dispatch_tick':next(r['tick'] for r in data['last_two_seconds_native'] if r['nominal_full12'][:8]==target),
    'thresholds':{'maximum_wheel_speed_rad_s':.25,'maximum_body_linear_speed_m_s':.05,'maximum_body_angular_speed_rad_s':.30},
    'windows':windows,'first_measured_controlled_tick_after_home_dispatch':None if not control_segments else control_segments[0]['first_tick'],
    'all_measured_controlled_segments_after_home_dispatch':control_segments,
    'all_measured_uncontrolled_segments_after_home_dispatch':lost_segments,
    'longest_measured_controlled_segment_after_home_dispatch':None if not control_segments else max(control_segments,key=lambda r:r['sample_count']),
    'source_home_target_servo_deg':target,'servo_order':['FL hip','FL knee','FR hip','FR knee','RL hip','RL knee','RR hip','RR knee'],
    'trace_every_0_1s':trace,'terminal_remains_failure':True,'no_smoothing_or_relabeling':True,
    'scope':'final_controlled is the unchanged three measured speed thresholds, not alone full task acceptance or exact-home acceptance',
    'home_error_distinction':'source-home target is [.5,-.7,3.7,.4,-2.6,-3.9,-.5,-6]; evaluator legacy home error uses eight zeros',
    'causality_limitation':'Time alignment shows post-home dynamic coupling; it does not independently isolate motor/load/contact causes.'}
with (FOLDER/'zero_terminal_motion_review.json').open('x',encoding='utf8') as stream:json.dump(receipt,stream,indent=2)
print(json.dumps({'window_counts':{name:{leg:r['exceedance_count'] for leg,r in row['wheels'].items()} for name,row in windows.items()},
    'first_saved_conditions':first_saved,'controlled_segments':control_segments,
    'trace':[{k:r[k] for k in ('tick','time_since_home_entry_s','max_abs_source_home_error_deg','body_speed_m_s','body_angular_speed_rad_s','wheel_qd_rad_s','final_controlled_thresholds')} for r in trace]},indent=2))
