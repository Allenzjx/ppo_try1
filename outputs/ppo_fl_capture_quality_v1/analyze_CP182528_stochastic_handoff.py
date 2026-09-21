"""Representative rows from one sealed stochastic evaluation; no model calls."""
import json
import math
from pathlib import Path
from analyze_C_CP180480_P05 import rows

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
SOURCE=ROOT/'runs/ppo_fl_capture_quality_v1/video_eval/validation/20260918T0810262817782Z_g3a50657a96c9_63b77f8309b24cd4a750cb8bbf4e3013/source'
LEGS=('FL','FR','RL','RR')
NAMES=('front_left','front_right','rear_left','rear_right')


def norm(v): return math.sqrt(sum(x*x for x in v))


def rr_state(d):
    t=d['step_info']['semantic_task']; p=t['physical_evaluator']; r=p['current_legs']['RR']
    return dict(evaluation_tick=d['end_tick'],stage=t['stage_id'],
        current_RR={k:r.get(k) for k in ('current_lift_valid','lift_established_now','lift_established','active_attempt','contact_mode','ground_contact','top_contact','air','front_distance_m','clearance_m','bearing_force_n','bearing_verified','unsupported_free_lift_m','consecutive_free_air_samples')},
        current_history={k:p['history'][k]['RR'] for k in ('active_lift','front_edge_crossed','placed')},
        historical_event_ticks={k:p['history']['event_ticks'][k].get('RR') for k in ('active_lift','front_edge_crossed','placed')})


def main():
    manifest=json.loads((SOURCE/'semantic_video_source_manifest.json').read_text())
    assert json.loads((SOURCE.parent/'run_manifest.json').read_text())['lifecycle']=='DIAGNOSTIC_FAILURE'
    assert manifest['policy_sampling_mode']=='training_style_conditional_gaussian'
    assert manifest['episode_physics_ticks']==6699
    ds=list(rows(SOURCE/'video_policy_decisions.jsonl'))
    good=[d for d in ds if d['environment_step_returned']]
    lastgood=good[-1]; last=ds[-1]; terminal_task=lastgood['step_info']['semantic_task']
    starts={d['request_phase']:d['start_tick'] for d in reversed(ds)}
    first=next(d for d in ds if d['request_phase']=='P06'); previous=ds[ds.index(first)-1]
    handoff=first['start_tick']; mapping={tick:d for d in ds for tick in range(d['start_tick']+1,d['end_tick']+1)}
    events=[e for e in terminal_task['history']['lift_attempt_events'] if e['leg']=='RR' and e['event'] in ('qualified_measured_upward_lift','qualification_revoked_ground_before_cross','current_lift_revoked_ground')]
    selected_ticks={2761,handoff,handoff+1,handoff+8,handoff+80,5167,5394,6335,6381,6600,6696,6699}
    selected_ticks.update(starts[p] for p in ('P07','P08','P09'))
    selected=[]; ps=rows(SOURCE/'physical_observations.jsonl'); assert next(ps)['physics_tick']==0
    for n,p in zip(rows(SOURCE/'native_tick_audit.jsonl'),ps,strict=True):
        tick=p['physics_tick']; assert tick==n['episode_physics_tick']
        if tick not in selected_ticks: continue
        a=n['native_audit']; d=mapping[tick]; r=d['policy_request']; h=a['policy_headroom_evidence']
        assert a['raw_policy_action_full12']==r['selected_raw_full12']
        selected.append(dict(tick=tick,time_s=tick/120,request_phase=d['request_phase'],decision_start=d['start_tick'],
            body_z_m=p['base']['position_w_m'][2],body_linear_speed_m_s=norm(p['base']['linear_velocity_w_m_s']),
            body_angular_speed_rad_s=norm(p['base']['angular_velocity_w_rad_s']),projected_gravity=p['imu']['projected_gravity_b'],body_collision=p['body_collision'],
            channels={name:dict(base_mean=r['base_mean_full12'][i],conditional_mean=r['conditional_mean_full12'][i],sigma=r['effective_sigma_full12'][i],raw=r['selected_raw_full12'][i],
                nominal=n['nominal_full12'][i],mapped_N=a['native_drive_target_full12'][i],REQUEST=n['projected_residual_full12'][i],effective=h['effective_policy_residual_full12'][i],final=p['commanded_full12'][i],actual=p['actual_full12'][i])
                for name,i in (('FL_hip',0),('FL_knee',1),('FR_knee',3),('RR_hip',6),('RR_knee',7))},
            contacts={leg:dict(contact=p['contacts'][name+'_wheel']['contact_class'],pair_normal_force_n=sum(p['contacts'][name+'_wheel'][s]['normal_force_n'] for s in ('ground','obstacle')),
                gap_m=p['wheels'][name+'_ankle']['bottom_w_m'][2]-p['obstacle']['top_z_m']) for leg,name in zip(LEGS,NAMES)},
            request_full12=n['projected_residual_full12'],final_full12=p['commanded_full12'],actual_full12=p['actual_full12'],
            dispatch_verified=a['verified'] and a['setter_dispatch_targets_equal'] and a['actual_mapping_matches_dispatch'],headroom_clipped=h['clipped_servo_indices']))
    before=next(s for s in selected if s['tick']==handoff); after=next(s for s in selected if s['tick']==handoff+1)
    qualifications=[dict(event=e,first_returned_evaluation_after_event=rr_state(next(d for d in good if d['end_tick']>=e['physics_tick']))) for e in events]
    terminal=selected[-1]
    result=dict(schema='wlr50_clean.CP182528_stochastic_handoff_representative.v1',source=str(SOURCE),checkpoint=manifest['checkpoint_load_provenance']['source'],
        sampling=manifest['policy_sampling_mode'],policy_seed=manifest['policy_seed'],phase_starts=starts,
        final_event_history=terminal_task['history']['event_ticks'],RR_qualification_and_revocation=qualifications,
        last_returned_task_evaluation=rr_state(lastgood),last_returned_evaluation_tick=lastgood['end_tick'],terminal_tick=last['end_tick'],
        stop_reason=last['stop_reason'],terminal_environment_step_returned=last['environment_step_returned'],
        fall_evidence=dict(base_z_m=terminal['body_z_m'],height_threshold_m=.015,projected_gravity_z=terminal['projected_gravity'][2],gravity_threshold=-.30,
            linear_speed_m_s=terminal['body_linear_speed_m_s'],angular_speed_rad_s=terminal['body_angular_speed_rad_s'],body_collision=terminal['body_collision'],
            source_code='src/wlr50_clean/ppo/isaac_fsm_backend.py:_fall_and_explosion'),
        P05_P06_handoff=dict(tick=handoff,previous_phase=previous['request_phase'],raw_history_exact=first['policy_request']['previous_raw_from_current_observation_full12']==previous['raw_policy_action_full12'],
            new_kernel_request_audit=first['policy_request'],request_first_tick_max_delta=max(abs(x-y) for x,y in zip(before['request_full12'],after['request_full12'])),
            final_first_tick_max_delta=max(abs(x-y) for x,y in zip(before['final_full12'],after['final_full12'])),
            final_first_tick_max_delta_unit_caveat='Numeric maximum mixes degrees and rad/s; use separate fields.',
            final_servo_first_tick_max_delta_deg=max(abs(x-y) for x,y in zip(before['final_full12'][:8],after['final_full12'][:8])),
            final_wheel_first_tick_delta_rad_s=[y-x for x,y in zip(before['final_full12'][8:],after['final_full12'][8:])]),
        representative_physical_rows=selected,
        limitations=['Exact RR event ticks and current validity at the subsequent returned evaluator tick are explicitly separate.',
            'Final safety-aborted tick6699 has native/physical evidence but no returned semantic step_info; last task evaluation is6696, not fabricated at6699.',
            'Contact pair normal-force sum is not automatically verified top bearing or motive-force attribution.',
            'No isolated causal attribution to FR/RR or one joint; state, support, mapper, nominal and policy evolve together.',
            'No full success or stability improvement. Offset200 is only a separate curriculum candidate, not executed here.'],
        extra_model_forwards=0,new_optimizer_steps=0,new_simulations=0,production_changed=False)
    path=OUT/'CP182528_stochastic_handoff.json'
    with path.open('x',encoding='utf-8') as s: json.dump(result,s,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ('phase_starts','final_event_history','last_returned_task_evaluation','fall_evidence','terminal_tick')},indent=2))
    print(json.dumps(dict(handoff_tick=handoff,request_bridge_delta=result['P05_P06_handoff']['request_first_tick_max_delta'],final_bridge_delta=result['P05_P06_handoff']['final_first_tick_max_delta'],path=str(path)),indent=2))


if __name__=='__main__': main()
