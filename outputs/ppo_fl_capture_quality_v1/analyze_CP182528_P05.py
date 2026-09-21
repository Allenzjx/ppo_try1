"""Bounded sealed C diagnosis. No simulation, model forward, or production writes."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / 'runs/ppo_fl_capture_quality_v1/video_eval/validation/20260918T0751563353604Z_g3a50657a96c9_9132b92ae22044568b20624cf72afc7f/source'
ORDER = ('front_left', 'front_right', 'rear_left', 'rear_right')
LEGS = ('FL', 'FR', 'RL', 'RR')


def rows(path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def stats(values):
    values = list(values)
    return dict(min=min(values), max=max(values), mean=sum(values)/len(values),
                rms=math.sqrt(sum(x*x for x in values)/len(values)))


def main():
    source_manifest = json.loads((SOURCE/'semantic_video_source_manifest.json').read_text())
    run_manifest = json.loads((SOURCE.parent/'run_manifest.json').read_text())
    assert run_manifest['lifecycle'] == 'DIAGNOSTIC_FAILURE'
    assert source_manifest['episode_physics_ticks'] == 6036
    assert source_manifest['policy_sampling_mode'] == 'deterministic_conditional_mean'
    decisions = {}
    decision_rows = []
    terminal = None
    for d in rows(SOURCE/'video_policy_decisions.jsonl'):
        a = d['step_info']; t = a['semantic_task']; r = d['policy_request']
        if d['request_phase'] == 'P05':
            c = dict(start_tick=d['start_tick'], end_tick=d['end_tick'], phase=d['request_phase'],
                     age_s=t['stage_age_s'], progress=t['phase_progress'],
                     current_FL=t['physical_evaluator']['current_legs']['FL'],
                     event_ticks=t['history']['event_ticks'],
                     task_potential=t['task_progress_potential'], stall_diagnostic=t['stall_diagnostic'],
                     local_timeout=t['local_timeout'], base_mean=r['base_mean_full12'],
                     conditional_mean=r['conditional_mean_full12'], previous_raw=r['previous_raw_from_current_observation_full12'],
                     sigma=r['effective_sigma_full12'], raw=r['selected_raw_full12'],
                     selected_equals_conditional=r['selected_raw_full12'] == r['conditional_mean_full12'],
                     support={leg: {key: t['physical_evaluator']['current_legs'][leg].get(key)
                                      for key in ('support', 'air', 'bearing_force_n', 'load_fraction', 'contact_surface')}
                              for leg in LEGS})
            decision_rows.append(c)
            for tick in range(d['start_tick']+1, d['end_tick']+1):
                decisions[tick] = c
        terminal = t
    assert terminal['termination_source'] == 'LOCAL_BOUNDED_RECOVERY_EXHAUSTED'
    contract = json.loads((ROOT/'configs/recording_motion_contract.json').read_text())
    p05 = next(p for p in contract['phases'] if p['state_id'] == 'P05')
    start_tick = decision_rows[0]['start_tick']+1
    # MotionExecutor's first sample uses source tick zero; P05 has no sequence wait.
    endpoint_tick = start_tick+round(p05['active_duration_s']*120)
    samples = []
    native_rows = rows(SOURCE/'native_tick_audit.jsonl')
    physical_rows = rows(SOURCE/'physical_observations.jsonl')
    first = next(physical_rows)
    assert first['physics_tick'] == 0
    for n, p in zip(native_rows, physical_rows, strict=True):
        tick = p['physics_tick']
        assert tick == n['episode_physics_tick']
        if tick < start_tick:
            continue
        a = n['native_audit']; h = a['policy_headroom_evidence']; tr = a['tracking_reference_evidence']
        d = decisions[tick]
        assert a['raw_policy_action_full12'] == d['raw']
        joints = [p['joints']['front_left_'+part] for part in ('hip', 'knee')]
        fl_contact = p['contacts']['front_left_wheel']
        s = dict(tick=tick, time_s=tick/120, decision_start_tick=d['start_tick'],
            FL_base_mean=d['base_mean'][:2], FL_conditional_mean=d['conditional_mean'][:2],
            FL_previous_raw=d['previous_raw'][:2], FL_sigma=d['sigma'][:2], FL_raw=d['raw'][:2],
            FL_unfiltered_cap_tanh_deg=[18*math.tanh(d['raw'][0]), 24*math.tanh(d['raw'][1])],
            N_full12=n['nominal_full12'], mapped_N_full12=a['native_drive_target_full12'],
            controller_bias_full12=a['controller_drive_bias_full12'],
            requested_residual_full12=n['projected_residual_full12'], effective_residual_full12=h['effective_policy_residual_full12'],
            final_full12=p['commanded_full12'], actual_full12=p['actual_full12'],
            FL_error_deg=[j['command_deg']-j['position_deg'] for j in joints],
            FL_velocity_deg_s=[j['velocity_deg_s'] for j in joints],
            FL_mapper_pre_compensation_deg=tr['mapper_pre_state']['tracking_compensation_deg'][:2],
            FL_reference=[{key: c[key] for key in ('scheduled', 'feedback_eligible', 'reference_used',
                'previous_requested_deg', 'active_reference_deg', 'desired_original_c0_deg', 'bounded_desired_c1_deg')}
                for c in tr['channels'][:2]],
            FL_native_target_rad=a['actual_native_targets']['servo_position_rad'][:2],
            FL_native_measured_before_dispatch_rad=tr['actual_measured_physical_rad'][:2],
            FL_gap_mm=1000*(p['wheels']['front_left_ankle']['bottom_w_m'][2]-p['obstacle']['top_z_m']),
            FL_front_mm=1000*(p['wheels']['front_left_ankle']['center_w_m'][0]-p['obstacle']['front_x_m']),
            FL_contact=fl_contact['contact_class'],
            FL_pair_forces_n={name: fl_contact[name]['normal_force_n'] for name in ('ground', 'obstacle')},
            contacts={leg: dict(contact=p['contacts'][name+'_wheel']['contact_class'],
                normal_force_n=sum(p['contacts'][name+'_wheel'][surface]['normal_force_n'] for surface in ('ground', 'obstacle')))
                for leg, name in zip(LEGS, ORDER)},
            body=p['base'], FL_center_w_m=p['wheels']['front_left_ankle']['center_w_m'],
            mask=a['phase_mask_full12'], verified=a['verified'], mapping_matches=a['actual_mapping_matches_dispatch'],
            setter_equal=a['setter_dispatch_targets_equal'],
            final_slew_difference_deg=[p['commanded_full12'][i]-h['candidate_native_target_before_final_slew_full12'][i] for i in range(2)],
            headroom_clipped_servo_indices=h['clipped_servo_indices'])
        samples.append(s)
    assert len(samples) == 6036-start_tick+1
    assert max(abs(x-y) for s in samples for x,y in zip(s['N_full12'][:8],p05['end_full12'][:8]) if s['tick']>=endpoint_tick) < 1e-10
    selected_ticks = {start_tick, 2586, endpoint_tick-120, endpoint_tick-8, endpoint_tick,
                      endpoint_tick+7, endpoint_tick+120, 3000, 4608, 5200, 6032, 6036}
    offset_ticks = {offset: decision_rows[0]['start_tick']+8*offset for offset in (150,200,240)}
    selected_ticks.update(offset_ticks.values())
    selected = [s for s in samples if s['tick'] in selected_ticks]
    summaries = {}
    for label, low, high in [('source_endpoint_pm1s', endpoint_tick-120, endpoint_tick+120),
                             ('source_endpoint_to_end',endpoint_tick,6036), ('last_6s',5317,6036)]:
        ss=[s for s in samples if low <= s['tick'] <= high]
        summaries[label]=dict(ticks=[ss[0]['tick'],ss[-1]['tick']],samples=len(ss),
            FL_gap_mm=stats(s['FL_gap_mm'] for s in ss), FL_front_mm=stats(s['FL_front_mm'] for s in ss),
            FL_error_deg=[stats(s['FL_error_deg'][i] for s in ss) for i in range(2)],
            FL_abs_error_deg=[stats(abs(s['FL_error_deg'][i]) for s in ss) for i in range(2)],
            FL_request_deg=[stats(s['requested_residual_full12'][i] for s in ss) for i in range(2)],
            FL_mapped_N_deg=[stats(s['mapped_N_full12'][i] for s in ss) for i in range(2)],
            FL_target_deg=[stats(s['final_full12'][i] for s in ss) for i in range(2)],
            FL_actual_deg=[stats(s['actual_full12'][i] for s in ss) for i in range(2)],
            FL_contacts=dict(Counter(s['FL_contact'] for s in ss)),
            body_z_m=stats(s['body']['position_w_m'][2] for s in ss),
            body_linear_speed_m_s=stats(math.sqrt(sum(v*v for v in s['body']['linear_velocity_w_m_s'])) for s in ss),
            body_angular_speed_rad_s=stats(math.sqrt(sum(v*v for v in s['body']['angular_velocity_w_rad_s'])) for s in ss),
            FL_max_obstacle_force_n=max(s['FL_pair_forces_n']['obstacle'] for s in ss),
            mapped_N_FL_hip_target_changes=sum(ss[i]['mapped_N_full12'][0]!=ss[i-1]['mapped_N_full12'][0] for i in range(1,len(ss))),
            final_slew_modified_samples=sum(any(abs(v)>1e-9 for v in s['final_slew_difference_deg']) for s in ss),
            headroom_clipped_FL_samples=sum(any(i in s['headroom_clipped_servo_indices'] for i in (0,1)) for s in ss),
            support_contact_counts={leg:dict(Counter(s['contacts'][leg]['contact'] for s in ss)) for leg in LEGS},
            support_normal_force_n={leg:stats(s['contacts'][leg]['normal_force_n'] for s in ss) for leg in LEGS},
            all12_mask_one=all(s['mask']==[1.]*12 for s in ss),
            all_dispatch_verified=all(s['verified'] and s['mapping_matches'] and s['setter_equal'] for s in ss),
            max_request_minus_effective_FL_deg=max(abs(s['requested_residual_full12'][i]-s['effective_residual_full12'][i]) for s in ss for i in (0,1)))
    record=dict(schema='wlr50_clean.sealed_C_CP182528_P05_diagnosis.v1',source=str(SOURCE),
        checkpoint=source_manifest['checkpoint_load_provenance']['source'],
        source_manifest_sha256=hashlib.sha256((SOURCE/'semantic_video_source_manifest.json').read_bytes()).hexdigest(),
        mode=source_manifest['policy_sampling_mode'], lifecycle=run_manifest['lifecycle'],
        terminal={key:terminal[key] for key in ('termination_reason','termination_source','stage_id','stage_age_s',
            'phase_progress','task_progress_potential','local_timeout','stall_diagnostic','completion_values')},
        physical_evaluator_terminal={key:terminal['physical_evaluator'].get(key) for key in ('valid','termination_reason', 'failure_details')},
        events=terminal['history']['event_ticks'], terminal_FL=terminal['physical_evaluator']['current_legs']['FL'],
        source_endpoint=dict(tick=endpoint_tick, time_s=endpoint_tick/120, first_P05_dispatch_tick=start_tick,
            authored_source_duration_s=p05['active_duration_s'], full12_endpoint=p05['end_full12'],
            status='code-and-logged-target-derived; source endpoint flag not independently logged for P05',
            note='MotionExecutor starts at local tick0; P05 sequence permission always true; all post-endpoint recorded nominal servo8 exactly equal source endpoint.'),
        windows=summaries,selected_same_tick_chain=selected,
        prefix_offset_candidates=[dict(offset_decisions=offset, tick=tick,time_s=tick/120,
            evaluated_decision=next(d for d in decision_rows if d['end_tick']==tick),
            same_tick=next(s for s in samples if s['tick']==tick),
            claim='real deterministic full-eval state only; not seed1001 prefix receipt or reachability guarantee')
            for offset,tick in offset_ticks.items()],
        deterministic_mean_max_error=max(abs(d['raw'][i]-d['conditional_mean'][i]) for d in decision_rows for i in range(12)),
        decision_samples=len(decision_rows),physics_samples=len(samples),
        boundaries=dict(new_simulations=0,new_model_forwards=0,optimizer_updates=0,production_changed=False),
        limitations=['N/mapped N here is the SAME real closed-loop state; not an independent zero trajectory.',
            'Actual canonical q/qd is read after tick; native measured_rad is pre-dispatch, explicitly separate.',
            'Pair-force sum is reported separately from evaluator verified bearing/support; it is not inferred support.',
            'No sign-only attribution or frozen-body FK is a closed-loop intervention; the exact causal contribution of each servo/support/wheel remains unisolated.',
            'P06 was never entered; the separately diagnosed P05-to-P06 cap expansion cannot be the direct cause of this run termination.'])
    dest=OUT/'CP182528_P05_diagnosis.json'
    with dest.open('x',encoding='utf-8') as stream:
        json.dump(record,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(path=str(dest),endpoint=record['source_endpoint'],windows=summaries,terminal_chain=selected[-1]),indent=2))


if __name__ == '__main__':
    main()


