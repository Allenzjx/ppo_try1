"""Read only sealed aggregates; preserve exact endpoints and acceptance labels."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parent
V1=ROOT/'zero_refine_3d23189_run034324/zero_review_aggregate.json'
V2=ROOT/'zero_refine_6c2121b_run042420/zero_review_aggregate.json'
LEGS=('FL','FR','RL','RR')


def intervals(ticks):
    result=[]
    for tick in ticks:
        if result and result[-1]['last_tick']==tick-1:result[-1]['last_tick']=tick
        else:result.append({'first_tick':tick,'last_tick':tick})
    for row in result:
        row['sample_count']=row['last_tick']-row['first_tick']+1
        row['sample_exposure_s']=row['sample_count']/120
        row['endpoint_elapsed_s']=(row['last_tick']-row['first_tick'])/120
    return result


def controlled(row):
    return row['body_speed_m_s']<=.05 and row['body_angular_speed_rad_s']<=.30 and max(map(abs,row['actual_full12'][8:]))<=.25


def summarize(path):
    data=json.loads(path.read_text(encoding='utf8')); end=data['native_tick_count']
    home=data['source_home_entry']['entry_observation_tick']; target=data['source_home_target_servo_deg']
    physical={r['tick']:r for r in data['last_two_seconds_physical']}
    native={r['tick']:r for r in data['last_two_seconds_native']}
    windows={}
    for name,count in [('final_1s',120),('final_0_5s',60)]:
        rows=[physical[t] for t in range(end-count+1,end+1)]
        wheels={}
        for i,leg in enumerate(LEGS):
            breach=[r['tick'] for r in rows if abs(r['actual_full12'][8+i])>.25]
            wheels[leg]={'exceedance_count':len(breach),'maximum_abs_qd_rad_s':max(abs(r['actual_full12'][8+i]) for r in rows),
                'threshold_rad_s':.25,'intervals':intervals(breach)}
        windows[name]={'first_tick':rows[0]['tick'],'last_tick':end,'samples':count,'wheels':wheels,
            'body_linear_above_threshold_count':sum(r['body_speed_m_s']>.05 for r in rows),
            'body_angular_above_threshold_count':sum(r['body_angular_speed_rad_s']>.30 for r in rows),
            'all_control_thresholds_pass_count':sum(controlled(r) for r in rows)}
    after=[physical[t] for t in range(home+1,end+1)]
    segments=intervals([r['tick'] for r in after if controlled(r)])
    trace=[]
    for tick in range(home,end+1,12):
        r=physical[tick]; error=[q-n for q,n in zip(r['actual_full12'][:8],target)]
        trace.append({'tick':tick,'time_since_home_entry_s':(tick-home)/120,
            'max_abs_source_home_error_deg':max(map(abs,error)),'actual_minus_source_home_deg':error,
            'wheel_qd_canonical_rad_s':r['actual_full12'][8:],'body_speed_m_s':r['body_speed_m_s'],
            'body_angular_speed_rad_s':r['body_angular_speed_rad_s'],'measured_controlled':controlled(r)})
    terminal=physical[end]; ev=data['source_physical_episode']['physical_task_evaluation']
    deltas=[max(abs(a-b) for a,b in zip(native[t]['nominal_full12'][:8],native[t-1]['nominal_full12'][:8]))
        for t in range(home+1,end+1)]
    return data,{'source_aggregate':str(path),'source_aggregate_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'source_run':data['source'],'recorded_lifecycle':data['lifecycle'],'success':ev['success'],
        'termination_reason':ev.get('termination_reason'),'termination_source':ev.get('termination_source'),
        'ticks':end,'duration_s':end/120,'traversal_event_time_s':ev.get('traversal_event_time_s'),
        'home_entry_tick':home,'home_entry':data['source_home_entry'],
        'first_exact_source_home_target_dispatch_tick':next((t for t in sorted(native) if native[t]['nominal_full12'][:8]==target),None),
        'maximum_home_nominal_servo_step_deg':max(deltas),
        'terminal_wheel_qd_canonical_rad_s':dict(zip(LEGS,terminal['actual_full12'][8:])),
        'terminal_nominal_wheels':native[end]['nominal_full12'][8:],
        'terminal_body_speed_m_s':terminal['body_speed_m_s'],
        'terminal_body_angular_speed_rad_s':terminal['body_angular_speed_rad_s'],
        'terminal_source_home_max_error_deg':data['maximum_abs_terminal_home_error_deg'],
        'terminal_source_home_error_deg':data['terminal_actual_minus_source_home_deg'],
        'terminal_evaluator_zero_home_error_deg':max(map(abs,terminal['actual_full12'][:8])),
        'terminal_measured_controlled':controlled(terminal),'windows':windows,
        'all_measured_controlled_intervals_after_home':segments,
        'longest_measured_controlled_interval':max(segments,key=lambda r:r['sample_count']) if segments else None,
        'first_controlled_tick_after_home':segments[0]['first_tick'] if segments else None,
        'trace_every_0_1s':trace,
        'post_home_maximum_body_linear_speed_m_s':max(r['body_speed_m_s'] for r in after),
        'post_home_maximum_body_angular_speed_rad_s':max(r['body_angular_speed_rad_s'] for r in after)}


def main():
    a,v1=summarize(V1); b,v2=summarize(V2)
    home=b['source_home_entry']['entry_observation_tick']
    pa={r['tick']:r for r in a['last_two_seconds_physical']}; pb={r['tick']:r for r in b['last_two_seconds_physical']}
    before=sorted(t for t in pa.keys()&pb.keys() if t<=home)
    prefix={'first_tick':before[0],'last_tick':before[-1],'sample_count':len(before),
        'actual_full12_max_abs_difference':max(abs(x-y) for t in before for x,y in zip(pa[t]['actual_full12'],pb[t]['actual_full12'])),
        'body_position_max_abs_difference_m':max(abs(x-y) for t in before for x,y in zip(pa[t]['body_position_w_m'],pb[t]['body_position_w_m'])),
        'scope':'Only overlapping saved tail through home entry, not a full-run state-equivalence claim'}
    result={'schema':'wlr50_clean.sealed_zero_home_v1_v2_comparison.v1','v1':v1,'v2':v2,'pre_home_tail_comparison':prefix,
        'thresholds_unchanged':{'wheel_abs_rad_s':.25,'body_speed_m_s':.05,'body_angular_speed_rad_s':.30},
        'new_physics_steps':0,'no_smoothing_or_relabeling':True,
        'interpretation':'v2 changes home nominal ramp, not acceptance; same recorded endpoint remains the outcome. Exact source-home angle convergence is distinct from task/controlled-stop success.',
        'causal_limit':'Two measured closed-loop runs compare outcomes; this report does not isolate wheel/contact/motor causality or prove all four contacts visually unoccluded.'}
    output=V2.parent/'zero_home_v1_v2_comparison.json'
    with output.open('x',encoding='utf8') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    fields=('success','termination_reason','terminal_wheel_qd_canonical_rad_s','terminal_body_speed_m_s',
        'terminal_body_angular_speed_rad_s','terminal_source_home_max_error_deg','maximum_home_nominal_servo_step_deg',
        'first_exact_source_home_target_dispatch_tick','longest_measured_controlled_interval')
    print(json.dumps({'output':str(output),'v1':{k:v1[k] for k in fields},'v2':{k:v2[k] for k in fields},
        'windows':{version:{name:{'wheels_above':{leg:q['exceedance_count'] for leg,q in w['wheels'].items()},
            'control_pass':w['all_control_thresholds_pass_count'],'body_linear_above':w['body_linear_above_threshold_count'],
            'body_angular_above':w['body_angular_above_threshold_count']} for name,w in row['windows'].items()} for version,row in [('v1',v1),('v2',v2)]},
        'prefix':prefix},indent=2))


if __name__=='__main__':main()
