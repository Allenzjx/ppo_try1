"""One sealed eval, sparse entry/offset/terminal evidence. No model or simulator."""
from pathlib import Path
import json
import math

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
SOURCE=ROOT/'runs/ppo_fl_capture_quality_v1/video_eval/validation/20260918T0851231634993Z_g3a50657a96c9_ceb7a3f60fb049a28fbd55c9143d710d/source'

def rows(path):
    with path.open(encoding='utf-8') as f:
        for line in f: yield json.loads(line)

manifest=json.loads((SOURCE/'semantic_video_source_manifest.json').read_text())
run=json.loads((SOURCE.parent/'run_manifest.json').read_text())
assert run['lifecycle']=='DIAGNOSTIC_FAILURE' and manifest['episode_physics_ticks']==5796
assert manifest['policy_sampling_mode']=='deterministic_conditional_mean'
terminal=None; entry=None; chosen_decisions=[]
for r in rows(SOURCE/'video_policy_decisions.jsonl'):
    if r['request_phase']=='P05' and entry is None: entry=r['start_tick']
    if entry is not None:
        targets={entry+1,entry+1600,2777,5776,5784,5792,5796}
        if any(r['start_tick']<t<=r['end_tick'] for t in targets): chosen_decisions.append(r)
    terminal=r['step_info']['semantic_task']
assert entry==1608
ticks={entry+1,entry+1600,2777,5776,5784,5792,5796}
selected_native={}
for n in rows(SOURCE/'native_tick_audit.jsonl'):
    if n['episode_physics_tick'] in ticks: selected_native[n['episode_physics_tick']]=n
selected=[]
for p in rows(SOURCE/'physical_observations.jsonl'):
    tick=p['physics_tick']
    if tick not in ticks: continue
    n=selected_native[tick]; a=n['native_audit'];h=a['policy_headroom_evidence']
    d=next(x for x in chosen_decisions if x['start_tick']<tick<=x['end_tick'])
    t=d['step_info']['semantic_task']; req=d['policy_request']
    assert a['raw_policy_action_full12']==req['selected_raw_full12']
    assert req['selected_raw_full12']==req['conditional_mean_full12']
    channels={}
    for i,name in enumerate(('hip','knee')):
        channels[name]=dict(base_mean=req['base_mean_full12'][i],conditional_mean=req['conditional_mean_full12'][i],
            raw=req['selected_raw_full12'][i],sigma=req['effective_sigma_full12'][i],
            nominal_deg=n['nominal_full12'][i],mapped_nominal_deg=a['native_drive_target_full12'][i],
            controller_bias_deg=a['controller_drive_bias_full12'][i],filtered_request_deg=n['projected_residual_full12'][i],
            effective_residual_deg=h['effective_policy_residual_full12'][i],
            candidate_before_slew_deg=h['candidate_native_target_before_final_slew_full12'][i],
            final_deg=p['commanded_full12'][i],actual_post_step_deg=p['actual_full12'][i],
            actual_velocity_deg_s=p['joints']['front_left_'+name]['velocity_deg_s'],
            target_actual_error_deg=p['commanded_full12'][i]-p['actual_full12'][i])
    selected.append(dict(tick=tick,time_s=tick/120,decision_start_tick=d['start_tick'],decision_end_tick=d['end_tick'],
        requested_phase=d['request_phase'],channels=channels,
        FL_gap_mm=1000*(p['wheels']['front_left_ankle']['bottom_w_m'][2]-p['obstacle']['top_z_m']),
        FL_front_mm=1000*(p['wheels']['front_left_ankle']['center_w_m'][0]-p['obstacle']['front_x_m']),
        FL_actual_contact=p['contacts']['front_left_wheel']['contact_class'],
        FL_pair_force_n={x:p['contacts']['front_left_wheel'][x]['normal_force_n'] for x in ('ground','obstacle')},
        evaluator_at_decision_end=dict(tick=d['end_tick'],FL=t['physical_evaluator']['current_legs']['FL'],
            active_lift=t['history']['active_lift']['FL'],crossed=t['history']['front_edge_crossed']['FL'],placed=t['history']['placed']['FL'],
            phase_progress=t['phase_progress']),
        full12_mask=a['phase_mask_full12'],verified=a['verified'],setter_equal=a['setter_dispatch_targets_equal'],mapping_matches=a['actual_mapping_matches_dispatch'],
        clipped_servo_indices=h['clipped_servo_indices'],body=p['base'],
        contacts={leg:dict(contact=p['contacts'][name+'_wheel']['contact_class'],
            pair_force_n={x:p['contacts'][name+'_wheel'][x]['normal_force_n'] for x in ('ground','obstacle')})
            for leg,name in zip(('FL','FR','RL','RR'),('front_left','front_right','rear_left','rear_right'))}))
prior=json.loads((OUT/'CP182528_P05_diagnosis.json').read_text())
report=dict(schema='wlr50_clean.CP183552_P05_increment.v1',source=str(SOURCE),checkpoint=manifest['checkpoint_load_provenance']['source'],
    sampling_mode=manifest['policy_sampling_mode'],lifecycle=run['lifecycle'],
    P05_entry_tick=entry,P05_first_dispatch_tick=entry+1,
    terminal_tick=5796,terminal_time_s=48.3,
    terminal={k:terminal[k] for k in ('termination_reason','termination_source','stage_id','stage_age_s','phase_progress','task_progress_potential','completion_values','local_timeout','stall_diagnostic')},
    terminal_FL=terminal['physical_evaluator']['current_legs']['FL'],events=terminal['history']['event_ticks'],
    physical_evaluator_terminal={k:terminal['physical_evaluator'].get(k) for k in ('valid','termination_reason','failure_details')},
    timeout_explanation=dict(current_top_gap_max_m=.025,current_gap_m=terminal['physical_evaluator']['current_legs']['FL']['clearance_m'],
        current_top_geometry=False,current_progress=.7,current_allowance_s=10*.7**2,current_limit_s=30+10*.7**2,
        current_entry_time_s=entry/120,prior_entry_time_s=1568/120,prior_top_geometry=prior['terminal_FL']['top_geometry'],
        prior_gap_m=prior['terminal_FL']['clearance_m'],prior_progress=prior['terminal']['phase_progress'],
        prior_allowance_s=prior['terminal']['local_timeout']['current_progress_allowance_s'],
        front_above_point15_is_not_the_direct_predicate_reason=True),
    selected_same_tick_chain=selected,
    prefix_candidate=dict(from_phase='P05',prefix_source='checkpoint_policy',teacher_offset_decisions=200,
        observed_tick=entry+1600,observed_time_s=(entry+1600)/120,
        scope='Observed in sealed deterministic CP183552 seed4001 full eval only; not seed1001 training prefix receipt or guarantee.'),
    boundaries=dict(new_forward=0,optimizer_updates=0,new_simulations=0,production_changed=False),
    limitations=['Only seven sparse physical rows: mask/dispatch conclusions are scoped to them, not every physics tick.',
        'Native final targets and actual post-step canonical q are compared at the same tick; they are not independent zero-run differences.',
        'Nonzero positive means and poor task outcome do not isolate a single joint causal effect.',
        'Evaluator state is decision-end state; its tick is explicit when differing from the selected physical tick.'])
dest=OUT/'CP183552_P05_diagnosis.json'
with dest.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,allow_nan=False)
print(json.dumps({k:report[k] for k in ('P05_entry_tick','terminal','timeout_explanation','prefix_candidate')},indent=2))
for s in selected:
    print(json.dumps({k:s[k] for k in ('tick','channels','FL_gap_mm','FL_front_mm','FL_actual_contact','FL_pair_force_n','contacts')},indent=2))
