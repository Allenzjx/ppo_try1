"""One CPU, one forward pass through sealed episode zero inside a live run."""
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path

if os.name == 'nt':
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN = ROOT/'runs/ppo_rr_capture_then_rl_transfer_v1/train/20260923T1045146351727Z_gd1871df37d6e_0a956b4862894edb85aa7a239aa27f0f'
HEAD = 'd1871df37d6ea909657511d0e43e7435198f6ccd'
manifest = json.loads((RUN/'run_manifest.started.json').read_text(encoding='utf-8'))
assert manifest['arguments']['expected_head'] == HEAD
assert manifest['arguments']['from_phase'] == 'P07'
assert manifest['arguments']['prefix_source'] == 'successful_nominal'

FIELDS = ('air','ground_contact','top_contact','top_surface_contact','contact_surface','obstacle_pair_active',
    'support','bearing_verified','bearing_force_n','consecutive_air_samples','consecutive_top_samples',
    'current_lift_valid','lift_established','within_top_xy','within_lateral_span','clearance_m','front_distance_m',
    'active_attempt','initial_clearance','ground_relative_lift_m','motion_continuation_allowed')


def compact(row):
    a = row['applied_audit']; t = a['semantic_task']; ev = t['physical_evaluator']
    u = a.get('actuator_target_effect_audit') or {}
    ref = u.get('tracking_reference_evidence') or {}
    rr = u.get('rr_capture_assist_evidence') or {}
    wh = u.get('rr_carry_wheel_evidence') or {}
    channels = ref.get('channels') or []
    measured = [c['nominal_deg']-c['current_actual_canonical_error_deg'] for c in channels] if len(channels)==8 else None
    return dict(global_decision=row['global_policy_decision'],post_tick=a['physics_tick'],time_s=a['sim_time_s'],
        request_phase=a['phase_id'],end_phase=a['end_phase_id'],termination=a['termination_reason'],
        history=t['history'],legs={leg:{key:ev['current_legs'][leg].get(key) for key in FIELDS} for leg in ('FL','FR','RL','RR')},
        N=a['nominal_action_full12'],REQUEST=a['projected_residual_full12'],FINAL=a['actual_drive_target_full12'],
        postmapper_before_other_bias=u.get('native_drive_target_full12'),raw_policy=row['raw_policy_action_full12'],
        mask=u.get('phase_mask_full12'),previous_FINAL_servos=u.get('previous_final_drive_servo_deg'),
        pre_actual_servo_canonical_deg_from_actual_error=measured,
        pre_actual_observation_tick=(rr.get('context') or {}).get('source_observation_tick'),
        measured_wheel_rad_s=ev.get('measured_wheel_velocity_rad_s'),
        tracking_servos=ref.get('tracking_servo_names'),tracking_channels=channels,
        rr_assist=rr,fl_assist=u.get('capture_assist_evidence'),wheel=wh,
        native_targets=u.get('actual_native_targets'),last_setter=u.get('actual_target_source'),verified=u.get('verified'),
        rr_placed_currently_usable=t.get('rr_placed_currently_usable'),
        rr_capture_continuation=t.get('rr_capture_continuation'),source_diagnostics=t.get('nominal_provider_diagnostics'),
        transitions=a.get('stage_transition_evidence'),body_geometry=ev.get('body_traversal_geometry'),
        physical_evidence_status=ev.get('physical_evidence_status'))


rows, timeline, transitions, first_events = [], [], [], {}
digest = hashlib.sha256(); count = bytes_read = 0; previous = None; top_seen = placed_seen = lost_seen = False
with (RUN/'residual_and_projection_audit.jsonl').open('rb') as stream:
    for line in stream:
        assert line.endswith(b'\n'), 'first episode has partial persisted row'
        digest.update(line); bytes_read += len(line); count += 1
        full = json.loads(line); row = compact(full)
        assert row['global_decision'] == 221056+count
        rr = row['legs']['RR']; h = row['history']
        top = rr['top_surface_contact'] is True
        placed = h['placed']['RR'] is True
        labels = []
        for phase in ('P10','P11','P12'):
            if row['end_phase']==phase and 'first_observed_'+phase not in first_events:
                labels.append('first_observed_'+phase)
        if top and not top_seen: labels.append('first_observed_RR_TOP')
        if placed and not placed_seen: labels.append('first_observed_RR_placed')
        if top_seen and previous and previous['legs']['RR']['top_surface_contact'] and not top:
            labels.append('RR_TOP_loss_observed')
            if not lost_seen: labels.append('first_RR_TOP_loss_observed')
            lost_seen = True
        if lost_seen and top and previous and not previous['legs']['RR']['top_surface_contact']:
            labels.append('RR_TOP_reacquired_observed')
        if placed_seen and rr['ground_contact'] and 'first_observed_RR_reground' not in first_events:
            labels.append('first_observed_RR_reground')
        for label in labels:
            if label not in first_events: first_events[label] = row
        if labels: timeline.append(dict(labels=labels,row=row))
        if row['transitions']: transitions.extend(row['transitions'])
        if 67 <= row['time_s'] <= 78: rows.append(row)
        if row['termination']:
            assert row['global_decision']==221908 and count==852, (count,row['global_decision'])
            terminal = row
            break
        previous = row; top_seen |= top; placed_seen |= placed
    else:
        raise RuntimeError('sealed first terminal missing; do not proceed into next prefix')
result = dict(schema='readonly.v8_first_suffix_RR_capture_loss.v1',source_run=str(RUN),source_HEAD=HEAD,
    new_physics=0,new_PPO_updates=0,new_actor_forwards=0,reader_pid=os.getpid(),
    source_prefix=dict(rows=count,bytes_read=bytes_read,sha256=digest.hexdigest(),end_global=221908,
        first_episode_sealed_only=True,live_file_not_globally_hashed=True),
    first_events=first_events,transitions=transitions,timeline=timeline,selected_67_78_s=rows,terminal=terminal,
    precision='Decision endpoints only. Event history ticks and uninterrupted AIR/TOP counts may sharpen bounds; never invent unsaved native ticks.',
    scope='Real P07 successful_nominal-prefix course; prefix has zero PPO credit, not a natural-P01 policy success.')
output = HERE/'v8_first_suffix_RR_capture_loss_readonly.json'
with output.open('x',encoding='utf-8') as target:
    json.dump(result,target,indent=2,allow_nan=False)
print(json.dumps(dict(output=str(output),reader_pid=os.getpid(),read=result['source_prefix'],selected_rows=len(rows),
    timeline=[dict(labels=x['labels'],time=x['row']['time_s'],tick=x['row']['post_tick'],global_decision=x['row']['global_decision'],
        phase=x['row']['end_phase'],RR=x['row']['legs']['RR'],assist_after=(x['row']['rr_assist'] or {}).get('state_after')) for x in timeline],
    transitions=transitions,terminal={k:terminal[k] for k in ('time_s','post_tick','termination','global_decision')},
    sha256=hashlib.sha256(output.read_bytes()).hexdigest()),allow_nan=False))
