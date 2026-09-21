"""One sealed stochastic P05->P06/FALL window, saved-data arithmetic only."""
from collections import Counter
import json
import math
from pathlib import Path
import yaml
from analyze_C_CP180480_P05 import rows, stats

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
SOURCE=ROOT/'runs/ppo_fl_capture_quality_v1/video_eval/validation/20260918T0544310970400Z_gf2e552406ea7_57ecc6333ba542e4b70b7742db002975/source'
LEGS=('FL','FR','RL','RR')
NAMES=('front_left','front_right','rear_left','rear_right')


def main():
    manifest=json.loads((SOURCE/'semantic_video_source_manifest.json').read_text())
    sealed=json.loads((SOURCE.parent/'run_manifest.json').read_text())
    assert sealed['lifecycle']=='DIAGNOSTIC_FAILURE'
    assert manifest['policy_sampling_mode']=='training_style_conditional_gaussian'
    ds=list(rows(SOURCE/'video_policy_decisions.jsonl'))
    first=next(d for d in ds if d['request_phase']=='P06')
    previous=ds[ds.index(first)-1]
    assert first['start_tick']==3000 and previous['request_phase']=='P05'
    assert first['policy_request']['previous_raw_from_current_observation_full12']==previous['raw_policy_action_full12']
    caps=yaml.safe_load((ROOT/'configs/ppo_fl_capture_quality_v1/execution_profile.yaml').read_text())['residual']['phase_caps_full12']
    mapping={tick:d for d in ds for tick in range(d['start_tick']+1,d['end_tick']+1)}
    selected_ticks={2992,2995,3000,3001,3004,3008,3024,3040,3064,3120,3200,3280,3320,3360,3368}
    samples=[]
    ps=rows(SOURCE/'physical_observations.jsonl'); assert next(ps)['physics_tick']==0
    for n,p in zip(rows(SOURCE/'native_tick_audit.jsonl'),ps,strict=True):
        tick=p['physics_tick']; assert tick==n['episode_physics_tick']
        if tick<2960: continue
        a=n['native_audit']; h=a['policy_headroom_evidence']; d=mapping[tick]
        r=d['policy_request']
        s=dict(tick=tick,time_s=tick/120,phase=d['request_phase'],decision_start=d['start_tick'],
            base_mean=r['base_mean_full12'],conditional_mean=r['conditional_mean_full12'],sigma=r['effective_sigma_full12'],
            raw=r['selected_raw_full12'],nominal=n['nominal_full12'],mapped_N=a['native_drive_target_full12'],
            requested=n['projected_residual_full12'],effective=h['effective_policy_residual_full12'],
            final=p['commanded_full12'],actual=p['actual_full12'],
            body=p['base'],gravity=p['imu']['projected_gravity_b'],collision=p['body_collision'],
            contacts={leg:dict(contact=p['contacts'][name+'_wheel']['contact_class'],
                normal_force_n=sum(p['contacts'][name+'_wheel'][surface]['normal_force_n'] for surface in ('ground','obstacle')),
                gap_m=p['wheels'][name+'_ankle']['bottom_w_m'][2]-p['obstacle']['top_z_m']) for leg,name in zip(LEGS,NAMES)},
            mask=a['phase_mask_full12'],dispatch_verified=a['verified'] and a['actual_mapping_matches_dispatch'] and a['setter_dispatch_targets_equal'],
            headroom_clipped=h['clipped_servo_indices'],
            final_slew_difference=[p['commanded_full12'][i]-h['candidate_native_target_before_final_slew_full12'][i] for i in range(8)],
            previous_final=a['previous_final_drive_servo_deg'])
        assert a['raw_policy_action_full12']==r['selected_raw_full12']
        samples.append(s)
    s3000=next(s for s in samples if s['tick']==3000)
    s3001=next(s for s in samples if s['tick']==3001)
    order=previous['step_info']['actuator_target_effect_audit']['canonical_order']
    algebra=[]
    for i,name in enumerate(order):
        oldcap,newcap=caps['P05'][i],caps['P06'][i]
        oldraw=previous['raw_policy_action_full12'][i]; r=first['policy_request']
        raw=r['selected_raw_full12'][i]; mean=r['conditional_mean_full12'][i]; base=r['base_mean_full12'][i]
        previous_request=s3000['requested'][i]
        center=math.atanh(max(-1+1e-6,min(1-1e-6,previous_request/newcap))) if newcap>oldcap else oldraw
        hypothetical_mean=.1*base+.9*center
        hypothetical_raw=raw+(hypothetical_mean-mean)
        algebra.append(dict(channel=name,oldcap=oldcap,newcap=newcap,previous_raw=oldraw,first_raw=raw,
            same_raw_oldcap=oldcap*math.tanh(oldraw),same_raw_newcap=newcap*math.tanh(oldraw),
            scale_only_increment=(newcap-oldcap)*math.tanh(oldraw),
            raw_change_increment=newcap*(math.tanh(raw)-math.tanh(oldraw)),
            first_unfiltered_request=newcap*math.tanh(raw),previous_filtered_request=previous_request,
            first_filtered_request=next(s for s in samples if s['tick']==3008)['requested'][i],
            first_base_mean=base,first_conditional_mean=mean,first_effective_sigma=r['effective_sigma_full12'][i],
            actual_innovation=raw-mean,actual_innovation_sigma=(raw-mean)/r['effective_sigma_full12'][i],
            first_entry_request_center_algebra=center,hypothetical_conditional_same_base_sigma=hypothetical_mean,
            hypothetical_same_innovation_raw=hypothetical_raw,hypothetical_unfiltered_request=newcap*math.tanh(hypothetical_raw)))
    phase_windows={}
    for phase in ('P05','P06','P07','P08','P09'):
        ss=[s for s in samples if s['phase']==phase]
        if not ss:continue
        phase_windows[phase]=dict(ticks=[ss[0]['tick'],ss[-1]['tick']],samples=len(ss),
            body_z_m=stats(s['body']['position_w_m'][2] for s in ss),
            FR_knee_base_mean=stats(s['base_mean'][3] for s in ss),FR_knee_raw=stats(s['raw'][3] for s in ss),
            FR_knee_requested=stats(s['requested'][3] for s in ss),FR_knee_effective=stats(s['effective'][3] for s in ss),
            FR_knee_mapped_N=stats(s['mapped_N'][3] for s in ss),FR_knee_final=stats(s['final'][3] for s in ss),
            FR_knee_actual=stats(s['actual'][3] for s in ss),
            contacts={leg:dict(Counter(s['contacts'][leg]['contact'] for s in ss)) for leg in LEGS},
            contact_force={leg:stats(s['contacts'][leg]['normal_force_n'] for s in ss) for leg in LEGS},
            all_dispatch_verified=all(s['dispatch_verified'] for s in ss),
            max_request_effective_delta=max(abs(s['requested'][i]-s['effective'][i]) for s in ss for i in range(12)),
            max_FR_knee_final_slew_difference=max(abs(s['final_slew_difference'][3]) for s in ss),
            max_FR_knee_final_delta_per_tick=max(abs(s['final'][3]-s['previous_final'][3]) for s in ss),
            headroom_clipped_samples=sum(bool(s['headroom_clipped']) for s in ss))
    last=samples[-1]
    report=dict(schema='wlr50_clean.CP180480_stochastic_handoff_saved_arithmetic.v1',source=str(SOURCE),
        checkpoint=manifest['checkpoint_load_provenance']['source'],sampling=manifest['policy_sampling_mode'],policy_seed=manifest['policy_seed'],
        terminal=dict(tick=last['tick'],phase=last['phase'],stop_reason=ds[-1]['stop_reason'],step_returned=ds[-1]['environment_step_returned'],
            base_z_m=last['body']['position_w_m'][2],projected_gravity=last['gravity'],
            linear_speed_m_s=math.sqrt(sum(v*v for v in last['body']['linear_velocity_w_m_s'])),
            angular_speed_rad_s=math.sqrt(sum(v*v for v in last['body']['angular_velocity_w_rad_s'])),collision=last['collision']),
        events=first['step_info']['semantic_task']['history']['event_ticks'],
        phase_starts={d['request_phase']:d['start_tick'] for d in reversed(ds)},
        raw_history_exact_previous=True,wholebody_scale_algebra=algebra,phase_windows=phase_windows,
        transition_first_tick_request_equal_previous_within_1e12=all(abs(a-b)<1e-12 for a,b in zip(s3001['requested'],s3000['requested'])),
        transition_first_tick_request_max_error=max(abs(a-b) for a,b in zip(s3001['requested'],s3000['requested'])),
        transition_first_tick_final_max_jump=max(abs(a-b) for a,b in zip(s3001['final'],s3000['final'])),
        selected=[s for s in samples if s['tick'] in selected_ticks],
        limitations=['No complete372 observation is present in video decision log; proposed center uses saved previous filtered REQUEST and current cap, with actual saved base/sigma, arithmetic only.',
            'Same-innovation hypothetical is not a new actor forward or executed rollout; subsequent observations/HISTORY/means would change.',
            'The candidate changes entry center once only. It does not freeze physical residual or remove later negative base means, learned sigma, full caps, nominal or body-support coupling.',
            'Contact pair force is not automatically evaluator verified bearing. Safety-aborted final decision has no returned step_info; native and physical samples remain available.',
            'One observational run cannot assign the FALL causally to FR knee alone or prove candidate would prevent it.'],
        no_new_simulation_forward_optimizer_or_production_change=True)
    dest=OUT/'CP180480_stochastic_handoff.json'
    with dest.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(path=str(dest),phase_starts=report['phase_starts'],terminal=report['terminal'],FR_knee=algebra[3],
        bridge_preserved=report['transition_first_tick_request_equal_previous_within_1e12'],first_tick_jump=report['transition_first_tick_final_max_jump'],phase_windows=phase_windows),indent=2))


if __name__=='__main__':main()
