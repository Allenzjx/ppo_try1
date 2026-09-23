"""Sealed v7 last-two-seconds reader. Standard library, no runtime imports."""
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT/'runs/ppo_rr_capture_then_rl_transfer_v1/video_eval/validation/20260923T0934264919029Z_g60abc00957c0_ef405598c8954e6280e844c4c29ed041/source'
inventory = {}


def tail(name, count):
    path = SOURCE/name
    before = path.stat()
    with path.open('rb') as stream:
        stream.seek(0, 2)
        start, data = stream.tell(), b''
        while data.count(b'\n') < count+1 and start:
            size = min(start, 65536)
            start -= size
            stream.seek(start)
            data = stream.read(size)+data
            if len(data) > 32_000_000:
                raise RuntimeError('bounded tail exceeded 32 MB')
    assert data.endswith(b'\n'), name
    lines = data.splitlines(keepends=True)[-count:]
    after = path.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), name
    inventory[name] = dict(full_bytes=after.st_size, selected_rows=len(lines), bytes_read=len(data),
        selected_tail_sha256=hashlib.sha256(b''.join(lines)).hexdigest())
    return [json.loads(line) for line in lines]


def pick(row, keys):
    return {key: row.get(key) for key in keys}


def norm(v):
    return math.sqrt(sum(x*x for x in v))


manifest = json.loads((SOURCE/'semantic_video_source_manifest.json').read_text(encoding='utf-8'))
run = json.loads((SOURCE.parent/'run_manifest.json').read_text(encoding='utf-8'))
assert run['completed_at_utc'] and run['lifecycle'] == 'DIAGNOSTIC_FAILURE'
assert manifest['episode_physics_ticks'] == 14715 and manifest['physical_task_success'] is False
assert manifest['runtime_contract']['source_git_commit'].startswith('60abc00957c0')
physical = tail('physical_observations.jsonl', 241)
native = {r['episode_physics_tick']: r for r in tail('native_tick_audit.jsonl', 241)}
capture = {r['episode_physics_tick']: r for r in tail('capture_assist_ticks.jsonl', 241)}
decisions = tail('video_policy_decisions.jsonl', 32)
assert [r['physics_tick'] for r in physical] == list(range(14475, 14716))
samples = []
for raw in physical:
    tick = raw['physics_tick']
    row, cap = native[tick], capture[tick]
    audit = row['native_audit']
    wheel = audit['rr_carry_wheel_evidence']
    context = wheel['context']
    assist = audit['rr_capture_assist_evidence']
    rr = cap['current_legs']['RR']
    gap = raw['wheels']['rear_right_ankle']['bottom_w_m'][2]-raw['obstacle']['top_z_m']
    final, previous = wheel['output_full12'][8:], wheel['previous_final_wheel_rad_s']
    assert context['source_control_tick'] == tick-1, (tick, context['source_control_tick'])
    if samples:
        assert math.isclose(context['rr_gap_m'], samples[-1]['post_gap_m'], abs_tol=1e-12)
    samples.append(dict(post_tick=tick, post_time_s=raw['simulation_time_s'], pre_tick=context['source_control_tick'],
        pre_gap_m=context['rr_gap_m'], post_gap_m=gap, post_front_m=rr['front_distance_m'],
        action=context['action'], envelope_active=context['envelope_active'], gain=context['gain'],
        reasons=context['reasons'], current_air_qualified=context['rr_current_air_qualified'],
        source_commit_verified=context['source_commit_verified'], selected_indices=wheel['selected_indices'],
        source_evidence=wheel['source_evidence'], source_nominal_wheel_rad_s=row['nominal_full12'][8:],
        policy_requested_wheel_rad_s=row['projected_residual_full12'][8:],
        raw_policy_wheel=audit.get('raw_policy_action_full12', [None]*12)[8:],
        mask_wheel=audit.get('phase_mask_full12', [None]*12)[8:],
        previous_FINAL_wheel_rad_s=previous, candidate_wheel_rad_s=wheel['candidate_full12'][8:],
        desired_wheel_rad_s=wheel['desired_before_slew_full12'][8:], final_wheel_rad_s=final,
        final_rate_rad_s2=[(a-b)*120 for a, b in zip(final, previous)],
        physical_commanded_wheel_rad_s=raw['commanded_full12'][8:], actual_wheel_rad_s=raw['actual_full12'][8:],
        native_actuator_targets=audit.get('actual_native_targets'), native_drive_full12=audit.get('native_drive_target_full12'),
        native_joint_ids={'servo': audit.get('servo_joint_ids'), 'wheel': audit.get('wheel_joint_ids')},
        last_setter=audit.get('actual_target_source'), native_verified=audit.get('verified'),
        actual_RR_deg=raw['actual_full12'][6:8], final_RR_deg=raw['commanded_full12'][6:8],
        RR_target_minus_actual_deg=[raw['commanded_full12'][i]-raw['actual_full12'][i] for i in (6,7)],
        base=raw['base'], mass_COM=raw['center_of_mass'], RR_raw_contact=raw['contacts']['rear_right_wheel'],
        RR_current=pick(rr, ('air','ground_contact','obstacle_pair_active','contact_surface','top_contact',
            'top_surface_contact','support','bearing_verified','bearing_force_n','current_lift_valid','lift_established',
            'within_top_xy','within_lateral_span','ground_relative_lift_m')),
        other_current_support={leg: pick(cap['current_legs'][leg], ('air','ground_contact','top_surface_contact',
            'bearing_force_n','bearing_verified','support')) for leg in ('FL','FR','RL')},
        assist_after=assist['state_after'], continuation=cap.get('rr_capture_continuation'),
        placed_history=cap.get('placed_history')))

negative = [s for s in samples if s['post_gap_m'] < 0]
releases = [s for s in samples if s['action'] == 'release_slew']
transitions = [s for i,s in enumerate(samples) if i and (s['action'] != samples[i-1]['action']
    or (s['post_gap_m'] < 0) != (samples[i-1]['post_gap_m'] < 0))]
gaps = [s['post_gap_m'] for s in samples]
window = dict(first_tick=14475,last_tick=14715,rows=len(samples),
    first_negative_post_tick=negative[0]['post_tick'] if negative else None,
    first_release_episode_output_tick=releases[0]['post_tick'] if releases else None,
    negative_gap_rows=len(negative), release_rows=len(releases),
    gap_min_m=min(gaps), gap_max_m=max(gaps),
    gap_sign_crossings=sum((a<0)!=(b<0) for a,b in zip(gaps,gaps[1:])),
    wheel_action_changes=sum(a['action']!=b['action'] for a,b in zip(samples,samples[1:])),
    any_raw_pair_active=any(s['RR_raw_contact'][p]['active'] for s in samples for p in ('ground','obstacle')),
    max_raw_pair_force_norm_n=max(norm(s['RR_raw_contact'][p]['force_w_n']) for s in samples for p in ('ground','obstacle')),
    RR_obstacle_friction_nonzero_rows=sum(s['RR_raw_contact']['obstacle']['tangential_force_n']>0 for s in samples),
    RR_obstacle_friction_max_n=max(s['RR_raw_contact']['obstacle']['tangential_force_n'] for s in samples),
    any_current_TOP=any(s['RR_current']['top_surface_contact'] for s in samples),
    any_current_support=any(s['RR_current']['support'] for s in samples),
    all_current_Q=all(s['RR_current']['current_lift_valid'] for s in samples),
    FL_target_min_rad_s=min(s['final_wheel_rad_s'][0] for s in samples),
    FL_target_max_rad_s=max(s['final_wheel_rad_s'][0] for s in samples),
    FL_actual_min_rad_s=min(s['actual_wheel_rad_s'][0] for s in samples),
    FL_actual_max_rad_s=max(s['actual_wheel_rad_s'][0] for s in samples),
    body_delta_w_m=[b-a for a,b in zip(samples[0]['base']['position_w_m'],samples[-1]['base']['position_w_m'])],
    mass_COM_delta_w_m=[b-a for a,b in zip(samples[0]['mass_COM']['position_w_m'],samples[-1]['mass_COM']['position_w_m'])],
    RR_front_delta_m=samples[-1]['post_front_m']-samples[0]['post_front_m'])
info = decisions[-1]['step_info']
result = dict(schema='readonly.v7_signed_gap_wheel_tail.v1',source=str(SOURCE),
    source_HEAD=manifest['runtime_contract']['source_git_commit'],new_physics_ticks=0,new_PPO_updates=0,new_actor_forwards=0,
    scope='last 2 seconds only; no claim of first event outside selected tail',inventory=inventory,window=window,
    terminal_task=pick(info['semantic_task'],('stage_id','termination_reason','termination_source','stage_age_s','local_timeout','completion_values')),
    first_negative=negative[0] if negative else None,first_release=releases[0] if releases else None,
    transitions=transitions,first=samples[0],terminal=samples[-1],samples=samples,
    video_decision_last_task=pick(info,('termination_reason','task_success','task_outcome_label','terminal_bootstrap_allowed')))
output = HERE/'v7_signed_gap_wheel_terminal_readonly.json'
with output.open('x',encoding='utf-8') as stream:
    json.dump(result,stream,indent=2,allow_nan=False)
print(json.dumps({'output':str(output),'window':window,'terminal_task':result['terminal_task']}))
chosen = {samples[0]['post_tick'],samples[-1]['post_tick']}
if negative: chosen.update(range(negative[0]['post_tick']-2,negative[0]['post_tick']+4))
chosen.update(s['post_tick'] for s in transitions)
for s in samples:
    if s['post_tick'] in chosen:
        print(json.dumps({k:s[k] for k in ('post_tick','pre_tick','pre_gap_m','post_gap_m','post_front_m',
            'action','gain','candidate_wheel_rad_s','final_wheel_rad_s','actual_wheel_rad_s','actual_RR_deg','final_RR_deg')}))
print(json.dumps({'terminal_assist':samples[-1]['assist_after'], 'terminal_raw_contact':samples[-1]['RR_raw_contact'],
    'terminal_base':samples[-1]['base'],'terminal_support':samples[-1]['other_current_support'],
    'selected_json_sha256':hashlib.sha256(output.read_bytes()).hexdigest()}))
