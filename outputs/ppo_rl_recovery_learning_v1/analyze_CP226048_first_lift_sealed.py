"""Stdlib, sealed decision-source comparison; no production or model imports."""
from __future__ import annotations
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FORMAL = ROOT/'runs/ppo_rr_rl_timing_policy_learning_v1/video_eval/validation/20260924T0651033031701Z_gf6d1d2df8d87_b494c761b75a4755bd13d45572d8a2c1/source'
PROBE = ROOT/'runs/ppo_rr_rl_timing_policy_learning_v1/direction_probe/20260924_owner439_student_entry_v2'
CHANNELS = ('FL_hip','FL_knee','FR_hip','FR_knee','RL_hip','RL_knee','RR_hip','RR_knee','FL_wheel','FR_wheel','RL_wheel','RR_wheel')
LEGS = ('FL','FR','RL','RR')


def small(path):
    return json.loads(path.read_text(encoding='utf-8'))


def pick(d, keys):
    return {k:d.get(k) for k in keys}


def contact_label(leg):
    return (leg.get('contact_mode') or ('GROUND' if leg.get('ground_contact') else
        'TOP' if leg.get('top_contact') else 'AIR' if leg.get('air') else 'OTHER'))


def compact(r, probe):
    i=r['step_info']; t=i['semantic_task']; ev=t['physical_evaluator']; n=i['actuator_target_effect_audit']
    p=r['original_student_request_audit' if probe else 'policy_request']
    tracking=n['tracking_reference_evidence']
    layers=t['nominal_provider_diagnostics']['source_partial_order']['layers']
    own=p['rear_owner_observed_features']
    q=[c['nominal_deg']-c['current_actual_canonical_error_deg'] for c in tracking['channels']]
    return dict(tick=i['physics_tick'],time_s=i['sim_time_s'],request_phase=i['phase_id'],phase=i['end_phase_id'],
        decision_start_tick=r['decision_start_tick' if probe else 'start_tick'],
        termination=i['termination_reason'],outcome=i['task_outcome_label'],success=i['full_task_success'],
        event_ticks=ev['history']['event_ticks'],
        layers=layers,local_timeout=t.get('local_timeout'),stall=t.get('stall_diagnostic'),
        legs={leg:pick(ev['current_legs'][leg],('contact_mode','support','air','ground_contact','top_contact','bearing_verified',
            'bearing_force_n','load_fraction','current_lift_valid','active_attempt','clearance_m','front_distance_m',
            'unsupported_free_lift_m','consecutive_free_air_samples','free_air_reference_tick','consecutive_air_samples')) for leg in LEGS},
        rear_owner_input17=own,all12_mask=n['phase_mask_full12'],rear_assists_enabled=p['rear_task_assists_enabled'],
        raw_conditional_mean=p['conditional_mean_full12'],raw_selected=p['selected_raw_full12'],
        source_nominal_full12=i['nominal_action_full12'],mapper_nominal_full12=n['native_drive_target_full12'],
        projected_request_full12=i['projected_residual_full12'],final_full12=i['actual_drive_target_full12'],
        actual_q_pre_last_dispatch_deg=q,actual_q_episode_tick=i['physics_tick']-1,
        measured_wheels_canonical_rad_s=ev['measured_wheel_velocity_rad_s'],
        prior_ACK_requested_full12=tracking['previous_requested_full12'],
        same_tick_post_mapper_bias_full12=n['combined_post_mapper_bias_full12'],
        controller_bias_full12=n['controller_drive_bias_full12'],
        RR_transfer_direction=pick(t['transfer_roles']['RR']['transfer_direction_context'],('reference_tick','window_s',
            'fixed_direction_world','com_world_displacement_m','body_world_displacement_m','com_toward_receiver_m',
            'mass_weighted_com_position_w_m','load_fraction_change')),
        all12_unmodified=n['all12_policy_channels_unmodified_at_actuator'],
        ACK_verified=n['verified'],native_mapping_matches=n['actual_mapping_matches_dispatch'])


def run(path, filename, probe):
    rows=[]; digest=hashlib.sha256(); n=0; cutoff=None
    with (path/filename).open('rb') as f:
        for line in f:
            r=json.loads(line)
            if probe and (r['probe_active'] or r['overridden_indices']):
                cutoff=dict(first_intervened_decision_start_tick=r['decision_start_tick'],probe_active=r['probe_active'],
                    overridden_indices=r['overridden_indices']);break
            if probe:
                assert r['original_student_raw_full12']==r['applied_raw_full12']
            digest.update(line);n+=1;rows.append(compact(r,probe))
    phase_first={}; source_started={}; gate_changes=[]; prior=None
    for r in rows:
        phase_first.setdefault(r['phase'],dict(tick=r['tick'],time_s=r['time_s']))
        for layer in r['layers']:
            if 'actual_start_tick' in layer:
                source_started.setdefault(layer['stage'],layer['actual_start_tick'])
        sig=tuple((l['stage'],l.get('wait_reason')) for l in r['layers'])
        if sig!=prior and r['phase'] in ('P07','P08','P09'):
            gate_changes.append(dict(tick=r['tick'],time_s=r['time_s'],phase=r['phase'],
                layers=[pick(l,('stage','source_ticks','status','wait_reason','actual_start_tick','fr_group_start_tick',
                    'current_free_lift_source_readiness')) for l in r['layers']]))
        prior=sig
    window=[r for r in rows if r['event_ticks'].get('placed',{}).get('FL')]
    rr_window=[r for r in window if r['phase'] in ('P07','P08','P09')]
    p09wait=[r for r in rows if any(l.get('wait_reason')=='current_free_lift_before_pending_knee' for l in r['layers'])]
    selected={}
    def select(name,r):selected[name]=r
    if window:select('first_decision_after_FL_placed',window[0])
    for phase in ('P06','P07','P08','P09'):
        choices=[r for r in rows if r['phase']==phase]
        if choices:select('first_'+phase+'_decision_endpoint',choices[0])
    for stage,tick in source_started.items():
        select(stage+'_actual_source_start',min(rows,key=lambda r:abs(r['tick']-tick)))
    if 'P07' in source_started:
        for offset in (264,328):
            select('P07_source_start_plus_'+str(offset)+'ticks',min(rows,key=lambda r:abs(r['tick']-source_started['P07']-offset)))
    if p09wait:select('first_pending_knee_wait',p09wait[0])
    qlift=[r for r in rr_window if r['legs']['RR']['current_lift_valid']]
    if qlift:select('first_RR_current_qualified_decision',qlift[0])
    for sec in (47.8,55.0,57.4,58.3,60.0,65.93333333333334):
        if sec<=rows[-1]['time_s']:
            select('actual_endpoint_nearest_'+str(sec)+'s',min(rows,key=lambda r:abs(r['time_s']-sec)))
    if rr_window:
        select('maximum_RR_gap_after_P07',max(rr_window,key=lambda r:r['legs']['RR']['clearance_m']))
    select('last_in_scope',rows[-1])
    physical_transitions=[]
    with (path/'stage_transition_evidence.jsonl').open(encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            if r['physics_tick']>rows[-1]['tick']:break
            if r['from_stage']!=r['to_stage']:
                physical_transitions.append(pick(r,('physics_tick','sim_time_s','from_stage','to_stage')))
    return dict(source=str(path/filename),source_prefix_sha256=digest.hexdigest(),rows_read=n,cutoff=cutoff,
        phase_first_decision_endpoints=phase_first,actual_source_starts=source_started,
        phase_transition_exact_ticks=physical_transitions,
        final_history_event_ticks=rows[-1]['event_ticks'],
        post_FL_placement_window=dict(decisions=len(window),
            all12_mask_ones=sum(r['all12_mask']==[1]*12 for r in window),
            any_rear_owner_active=sum(any(r['rear_owner_input17'][8:12]) for r in window),
            rear_assist_enabled=sum(bool(r['rear_assists_enabled']) for r in window),
            qualified_RR=sum(r['legs']['RR']['current_lift_valid'] for r in window),
            qualified_RL=sum(r['legs']['RL']['current_lift_valid'] for r in window),
            actual_FL_contact_modes=dict(Counter(contact_label(r['legs']['FL']) for r in window)),
            first_FL_AIR_decision_after_placement=next((r['tick'] for r in window if r['legs']['FL']['air']),None),
            terminal_FL_AIR_streak_start_tick=(rows[-1]['tick']-rows[-1]['legs']['FL']['consecutive_air_samples']+1)
                if rows[-1]['legs']['FL']['air'] else None),
        P07_onward_decision_summary=dict(decisions=len(rr_window),
            RR_contact_modes=dict(Counter(r['legs']['RR']['contact_mode'] for r in rr_window)),
            RR_gap_range_m=[min(r['legs']['RR']['clearance_m'] for r in rr_window),max(r['legs']['RR']['clearance_m'] for r in rr_window)] if rr_window else None),
        pending_knee_wait_decisions=len(p09wait),
        first_pending_knee_wait_tick=p09wait[0]['tick'] if p09wait else None,
        last_pending_knee_wait_tick=p09wait[-1]['tick'] if p09wait else None,
        gate_changes=gate_changes,selected=selected),rows


def main():
    m=small(FORMAL/'semantic_video_source_manifest.json'); p=small(PROBE/'run_manifest.json')
    formal,fr=run(FORMAL,'video_policy_decisions.jsonl',False)
    probe,pr=run(PROBE,'diagnostic_decisions.jsonl',True)
    mf=m['runtime_contract']['files'];pf=p['runtime_contract']['files']
    contract_changes={k:dict(formal=mf.get(k),probe=pf.get(k)) for k in sorted(mf.keys()|pf.keys()) if mf.get(k)!=pf.get(k)}
    cm=small(Path(m['checkpoint_load_provenance']['source']['manifest']))
    cp=small(Path(p['checkpoint_manifest']))
    counter_keys=('global_policy_decisions','ppo_updates','optimizer_steps','optimizer_learning_rate','normalization','actor_parameter_sha256')
    result=dict(schema='readonly.CP226048.first_RR_lift_bounded.v1',
        semantics=['Read-only sealed decision records; no simulation, model load, optimizer update, or production edit.',
            'Both runs use seed4001, deterministic conditional mean; checkpoint weights and runtime identities differ. Not a formal same-checkpoint paired test.',
            'Probe prefix excludes first intervened decision and all later rows; no diagnostic rows are PPO samples.',
            'Canonical full12 order: '+','.join(CHANNELS),
            'Servo target/actual degrees; wheel target/actual rad/s; geometry meters and forces N.',
            'raw_selected is decision-input mean/request. Final is actual_drive_target_full12 at decision endpoint. q is independently measured before last dispatch (endpoint minus one tick).',
            'Source nominal, mapped nominal, filtered request and FINAL are separate logged fields. Their independently remapped differences are not relabelled single-step policy effects.',
            'Positive CoM projection alone does not prove lateral transfer or receiver bearing. AIR is not support.',
            'The planned retention-only reward requires historical RR placement; this formal run never qualifies first lift/placement, so that revision is not a fix for this failure.'],
        formal_manifest=dict(path=str(FORMAL/'semantic_video_source_manifest.json'),
            **pick(m,('seed','control_method','front_fl_capture_assist','rear_task_assist','physical_task_success','source_acceptance_error','episode_physics_ticks')),
            checkpoint_provenance={k:v for k,v in m['checkpoint_load_provenance'].items() if k!='policy_contract'},
            runtime_head=m['runtime_contract']['source_git_commit'],runtime_sha256=m['runtime_contract']['runtime_content_sha256']),
        probe_manifest=dict(path=str(PROBE/'run_manifest.json'),**pick(p,('seed','deterministic','run_role','checkpoint','checkpoint_sha256','formal_deterministic_policy_result')),
            runtime_head=p['runtime_contract']['source_git_commit'],runtime_sha256=p['runtime_contract']['runtime_content_sha256'],
            intervention=pick(p['probe'],('start_tick','release_tick','release_reason','status'))),
        runtime_contract_changed_files=contract_changes,formal=formal,probe_preintervention=probe)
    result['training_lineage']=dict(formal=pick(cm,counter_keys),probe=pick(cp,counter_keys),
        delta_policy_decisions=cm['global_policy_decisions']-cp['global_policy_decisions'],
        delta_PPO_updates=cm['ppo_updates']-cp['ppo_updates'],
        delta_Adam_steps=cm['optimizer_steps']-cp['optimizer_steps'])
    target=OUT/'CP226048_first_lift_sealed_readonly.json'
    target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=str(target),formal_summary={k:v for k,v in formal.items() if k not in ('selected','gate_changes')},
        probe_summary={k:v for k,v in probe.items() if k not in ('selected','gate_changes')},contract_change_names=list(contract_changes)),indent=2))


if __name__=='__main__':main()
