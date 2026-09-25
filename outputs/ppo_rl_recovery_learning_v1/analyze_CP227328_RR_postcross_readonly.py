"""One sealed source, stdlib only; no simulator/model or production imports."""
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'runs/ppo_rr_rl_timing_policy_learning_v1/video_eval/validation/20260924T1842131087811Z_g892385cba8a7_c074d1ebec6947cc830fb3e0ec86dcec/source'
OUTPUT = Path(__file__).resolve().parent
CROSS, LAST = 8230, 11092
SIGN = [-1, 1, -1, 1]
WHEELS = ['front_left_ankle', 'front_right_ankle', 'rear_left_ankle', 'rear_right_ankle']
boundaries = {}


def rows(name):
    with (SOURCE / name).open('rb') as stream:
        stream.seek(0, 2)
        boundary = stream.tell()
        boundaries[name] = boundary
        stream.seek(0)
        while stream.tell() < boundary:
            raw = stream.readline(boundary - stream.tell())
            if not raw.endswith(b'\n'):
                raise ValueError('Sealed file has incomplete final row: ' + name)
            yield json.loads(raw)


def compact_decision(row):
    step, request = row['step_info'], row['policy_request']
    task = step['semantic_task']
    ev = task['physical_evaluator']
    audit = step['actuator_target_effect_audit']
    room = audit['policy_headroom_evidence']
    provider = task['nominal_provider_diagnostics']
    p09 = next(x for x in provider['source_partial_order']['layers'] if x['stage'] == 'P09')
    physical_keys = ('contact_mode', 'contact_surface', 'air', 'ground_contact', 'obstacle_pair_active',
                     'top_contact', 'top_surface_contact', 'bearing_force_n', 'bearing_verified', 'support',
                     'current_lift_valid', 'within_top_xy', 'front_distance_m', 'clearance_m',
                     'load_fraction', 'load_fraction_valid', 'consecutive_top_samples')
    return dict(tick=row['end_tick'], start_tick=row['start_tick'], decision=row['decision'],
        time_s=step['sim_time_s'], phase=task['stage_id'],
        raw_policy=request['selected_raw_full12'],
        raw_tanh=request['selected_tanh_full12'],
        N=step['nominal_action_full12'], mapped_N=audit['native_drive_target_full12'],
        baseline=room['baseline_native_plus_controller_full12'],
        controller_bias=audit['controller_drive_bias_full12'],
        REQUEST=room['requested_policy_residual_full12'],
        effective=room['effective_policy_residual_full12'],
        residual_intervals=room['policy_residual_intervals_servo_deg'],
        headroom_clipped=room['clipped_servo_indices'],
        candidate=room['candidate_native_target_before_final_slew_full12'],
        FINAL=step['actual_drive_target_full12'],
        phase_mask=audit['phase_mask_full12'],
        owner17_input=request['rear_owner_observed_features'],
        owner_endpoint_receipt=audit.get('rear_owner_recovery'),
        rear_timing_input9=request['rear_policy_timing_observed_features'],
        cooperative_prep_allowed_input=request['cooperative_prep_allowed'],
        rear_task_assists_enabled=request['rear_task_assists_enabled'],
        current_cap=request['current_cap_full12'],
        p09_source=p09, rr_carry_continuation=provider['rr_carry_continuation'],
        current_capture_context=task['rr_capture_continuation'],
        cooperative_preparation=task['cooperative_preparation'],
        legs={leg: {key: ev['current_legs'][leg].get(key) for key in physical_keys}
              for leg in ('FL', 'FR', 'RL', 'RR')},
        event_ticks=task['history']['event_ticks'],
        physical_evidence_status=ev.get('physical_evidence_status'),
        termination_reason=step['termination_reason'],
        task_outcome_label=step['task_outcome_label'],
        task_termination_source=task.get('termination_source'),
        local_timeout=task.get('local_timeout'))


decisions = [compact_decision(row) for row in rows('video_policy_decisions.jsonl')
             if CROSS <= row['end_tick'] <= LAST]
assert decisions[0]['tick'] == 8232 and decisions[-1]['tick'] == LAST


def first_tick(predicate):
    return next((row['tick'] for row in decisions if predicate(row)), None)


events = dict(first_postcross_decision=decisions[0]['tick'],
    first_RR_knee_not_headroom_clipped=first_tick(lambda x: 7 not in x['headroom_clipped']),
    first_RL_ground_and_obstacle=first_tick(lambda x: x['legs']['RL']['ground_contact'] and x['legs']['RL']['obstacle_pair_active']),
    first_contact_evidence_not_verified=first_tick(lambda x: x['physical_evidence_status'] != 'VERIFIED'),
    first_late_wait_reason_changed=first_tick(lambda x: x['p09_source']['wait_reason'] != 'current_RR_bearing_before_FL_RL_transfer'))
selected_ticks = sorted({decisions[0]['tick'], 8640, 9600, LAST,
                         *[v for v in events.values() if v is not None]})
# Keep a compact table; all decision summaries remain available for aggregate checks.
if len(selected_ticks) > 6:
    selected_ticks.remove(8640)
assert len(selected_ticks) <= 6
selected = {row['tick']: row for row in decisions if row['tick'] in selected_ticks}
native_counts = collections.Counter()
native_owner_fields = set()
for row in rows('native_tick_audit.jsonl'):
    tick = row['episode_physics_tick']
    if not CROSS <= tick <= LAST:
        continue
    audit = row['native_audit']
    native_counts['rows'] += 1
    native_counts['all12_mask1'] += audit['phase_mask_full12'] == [1] * 12
    native_counts['verified_dispatch'] += audit['verified'] is True
    native_counts['matching_native_dispatch'] += audit['actual_mapping_matches_dispatch'] is True
    native_counts['RR_knee_headroom_clipped'] += 7 in audit['policy_headroom_evidence']['clipped_servo_indices']
    native_owner_fields.update(key for key in audit if 'owner' in key)
    if tick in selected:
        same = selected[tick]
        assert same['REQUEST'] == audit['policy_headroom_evidence']['requested_policy_residual_full12']
        same['native_dispatch_tick'] = audit['physics_tick']
        same['native_target_wheels_direct'] = audit['actual_native_targets']['wheel_velocity_rad_s']
        same['native_final_matches_dispatch'] = audit['actual_mapping_matches_dispatch']

physical_counts = collections.Counter()
physical_ranges = collections.defaultdict(list)
first_unverified_pair = None
for row in rows('physical_observations.jsonl'):
    tick = row['physics_tick']
    if not CROSS <= tick <= LAST:
        continue
    physical_counts['rows'] += 1
    rr = row['contacts']['rear_right_wheel']
    physical_counts['RR_AIR_contact_class'] += rr['contact_class'] == 'AIR'
    physical_counts['RR_ground_pair_active'] += rr['ground']['active'] is True
    physical_counts['RR_obstacle_pair_active'] += rr['obstacle']['active'] is True
    physical_counts['RR_FINAL_knee_minus58'] += abs(row['commanded_full12'][7] + 58.) < 1e-9
    physical_counts['FL_target_canonical_negative'] += row['commanded_full12'][8] < 0.
    physical_counts['FL_measured_canonical_negative'] += row['actual_full12'][8] < 0.
    for leg, name in zip(('FL', 'FR', 'RL', 'RR'), WHEELS):
        physical_ranges[leg + '_actual_canonical_wheel_rad_s'].append(row['wheels'][name]['velocity_rad_s'])
    if tick in selected:
        same = selected[tick]
        assert same['FINAL'] == row['commanded_full12']
        assert same['time_s'] == row['simulation_time_s']
        same['actual_full12'] = row['actual_full12']
        same['native_actual_wheels_reconstructed'] = [sign * value for sign, value in zip(SIGN, row['actual_full12'][8:])]
        same['native_actual_source'] = 'inverse canonical sign mapping of stored measured joint angular velocities; native raw sensor tensor not separately serialized'
        same['sensor_wheels_canonical'] = [row['wheels'][name]['velocity_rad_s'] for name in WHEELS]
        assert same['sensor_wheels_canonical'] == same['actual_full12'][8:]
        same['CoM_world_m'] = row['center_of_mass']['position_w_m']
        same['base_world_m'] = row['base']['position_w_m']
        same['contact_sensor_wheel_pairs'] = {leg: row['contacts'][body]
            for leg, body in zip(('FL', 'FR', 'RL', 'RR'),
                ('front_left_wheel', 'front_right_wheel', 'rear_left_wheel', 'rear_right_wheel'))}


def stat(values):
    return dict(min=min(values), max=max(values), distinct_exact=len(set(values)),
                distinct_rounded_6dp=len({round(value, 6) for value in values}))


counts = dict(decisions=len(decisions), all12_mask1=sum(x['phase_mask'] == [1] * 12 for x in decisions),
    owner17_input_all0=sum(x['owner17_input'] == [0.] * 17 for x in decisions),
    owner_endpoint_receipt_missing=sum(x['owner_endpoint_receipt'] is None for x in decisions),
    cooperative_prep_allowed_input=sum(x['cooperative_prep_allowed_input'] for x in decisions),
    current_RR_top_reachable=sum(x['current_capture_context']['rr_top_reachable'] for x in decisions),
    current_RR_bearing=sum(x['current_capture_context']['rr_current_bearing'] for x in decisions),
    current_RR_TOP=sum(x['legs']['RR']['top_contact'] for x in decisions),
    current_RR_XY=sum(x['legs']['RR']['within_top_xy'] for x in decisions),
    current_RR_qualified=sum(x['legs']['RR']['current_lift_valid'] for x in decisions),
    RL_qualified=sum(x['legs']['RL']['current_lift_valid'] for x in decisions),
    FINAL_equal_headroom_candidate=sum(x['FINAL'] == x['candidate'] for x in decisions),
    RR_knee_FINAL_minus58=sum(abs(x['FINAL'][7] + 58.) < 1e-9 for x in decisions),
    RR_knee_headroom_clipped=sum(7 in x['headroom_clipped'] for x in decisions))
stats = {name: {key: stat([row[key][index] for row in decisions])
                    for key in ('raw_policy', 'REQUEST', 'effective', 'FINAL')}
         for name, index in [('RR_hip', 6), ('RR_knee', 7)]}
stats['RR_knee_request_minus_effective_deg'] = stat([x['REQUEST'][7] - x['effective'][7] for x in decisions])
stats['RR_gap_mm'] = stat([1000*x['legs']['RR']['clearance_m'] for x in decisions])
stats['RR_front_mm'] = stat([1000*x['legs']['RR']['front_distance_m'] for x in decisions])
stats['all_FINAL_ranges'] = [[min(x['FINAL'][i] for x in decisions), max(x['FINAL'][i] for x in decisions)] for i in range(12)]
manifest = json.loads((SOURCE/'semantic_video_source_manifest.json').read_text(encoding='utf-8-sig'))
result = dict(schema='CP227328_RR_postcross_same_tick_readonly_v1', checkpoint=227328,
    head='892385cba8a7089b52567bb7f558c018fac82a77', source=str(SOURCE),
    read_scope=dict(native_and_sensor_ticks=[CROSS,LAST], decision_endpoints=[decisions[0]['tick'],LAST],
        complete_sealed_file_boundaries=boundaries, no_simulation_or_model_import=True),
    source_clock_semantics='native dispatch tick is episode tick +179; command consumed prior observation then endpoint sensor readback; policy owner17/timing9 are decision INPUT, not endpoint receipts',
    counts=counts, native_120Hz_counts=dict(native_counts), physical_120Hz_counts=dict(physical_counts),
    native_owner_fields_present=sorted(native_owner_fields), events=events, stats=stats,
    late_clock_distinct=sorted({x['p09_source']['source_ticks'] for x in decisions}),
    wait_reason_counts=dict(collections.Counter(x['p09_source']['wait_reason'] for x in decisions)),
    physical_evidence_status_counts=dict(collections.Counter(x['physical_evidence_status'] for x in decisions)),
    native_measured_wheel_semantics='reconstructed from actual native joint_vel through verified inverse sign conversion [-1,+1,-1,+1]; not separately stored raw native values',
    wheel_actual_canonical_ranges={key: [min(v),max(v)] for key,v in physical_ranges.items()},
    first_unfinished_task='P09 RR real TOP capture; RL transfer/lift not reached',
    final_event_ticks=decisions[-1]['event_ticks'], selected=list(selected.values()),
    source_manifest_result={key:manifest.get(key) for key in ('issued_policy_decisions','physical_task_success','success_candidate','rear_task_assist','front_fl_capture_assist')})
dest = OUTPUT / 'CP227328_DET_RR_postcross_8230_11092.json'
dest.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n',encoding='utf-8')
print(json.dumps({key:result[key] for key in ('counts','native_120Hz_counts','physical_120Hz_counts','events','stats','wait_reason_counts','physical_evidence_status_counts')},indent=2))
print('selected ticks',selected_ticks,'saved',dest)
