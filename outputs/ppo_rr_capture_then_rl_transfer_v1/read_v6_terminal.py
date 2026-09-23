"""Output-only sealed tail reader; no production imports or physical stepping."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT/'runs/ppo_rr_capture_then_rl_transfer_v1/video_eval/validation/20260923T0829436381354Z_g97c4367ee293_5795b2a4e4384ec6be3589fa43477a5c/source'
inventory={}


def tail(name, count):
    path=SOURCE/name; before=path.stat()
    with path.open('rb') as stream:
        stream.seek(0,2); start=stream.tell(); data=b''
        while data.count(b'\n')<count+1 and start:
            size=min(start,65536); start-=size; stream.seek(start)
            data=stream.read(size)+data
            if len(data)>30_000_000:raise RuntimeError('tail exceeded bound')
    assert data.endswith(b'\n')
    lines=data.splitlines(keepends=True)[-count:]
    after=path.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    inventory[name]=dict(full_bytes=after.st_size,selected_tail_rows=len(lines),
        selected_tail_sha256=hashlib.sha256(b''.join(lines)).hexdigest(),bytes_read=len(data))
    return [json.loads(line) for line in lines]


def pick(row, keys):return {key:row.get(key) for key in keys}


manifest=json.loads((SOURCE/'semantic_video_source_manifest.json').read_text(encoding='utf-8'))
run=json.loads((SOURCE.parent/'run_manifest.json').read_text(encoding='utf-8'))
assert run['completed_at_utc'] and run['lifecycle']=='DIAGNOSTIC_FAILURE'
assert manifest['episode_physics_ticks']==14600 and manifest['physical_task_success'] is False
assert manifest['runtime_contract']['source_git_commit'].startswith('97c4367ee293')
physical=tail('physical_observations.jsonl',241)
capture={r['episode_physics_tick']:r for r in tail('capture_assist_ticks.jsonl',241)}
native={r['episode_physics_tick']:r for r in tail('native_tick_audit.jsonl',241)}
decision=tail('video_policy_decisions.jsonl',2)
assert [p['physics_tick'] for p in physical]==list(range(14360,14601))
samples=[]
for raw in physical:
    t=raw['physics_tick'];c=capture[t];n=native[t]['native_audit'];rr=c['current_legs']['RR']
    assist=n['rr_capture_assist_evidence'];current=c['current_legs']
    samples.append(dict(tick=t,time_s=raw['simulation_time_s'],
        gap_m=raw['wheels']['rear_right_ankle']['bottom_w_m'][2]-raw['obstacle']['top_z_m'],
        front_m=rr['front_distance_m'],RR_raw_contact=raw['contacts']['rear_right_wheel'],
        RR_current=pick(rr,('air','ground_contact','obstacle_pair_active','contact_surface','top_surface_contact',
            'bearing_force_n','bearing_verified','support','current_lift_valid','lift_established','front_edge_crossed',
            'within_top_xy','within_lateral_span','ground_relative_lift_m','body_control_evidence','wheel_bottom_vz_m_s')),
        current_other_supports={leg:pick(current[leg],('air','ground_contact','top_surface_contact','contact_surface',
            'support','bearing_verified','bearing_force_n')) for leg in ('FL','FR','RL')},
        actual_RR_deg=raw['actual_full12'][6:8],target_RR_deg=raw['commanded_full12'][6:8],
        tracking_error_target_minus_actual_deg=[raw['commanded_full12'][i]-raw['actual_full12'][i] for i in (6,7)],
        state_before=assist['state_before'],state_after=assist['state_after'],pre_context=assist['context'],
        post_continuation=c.get('rr_capture_continuation'),placed_history=c.get('placed_history'),
        actual_last_setter=n.get('actual_target_source')))
negative=[s for s in samples if s['gap_m']<0.]
assert negative and negative[0]['tick']==14600
for s in samples:
    assert s['RR_current']['current_lift_valid'] is True
    assert s['RR_current']['within_top_xy'] is True
    assert s['RR_current']['air'] is True
    assert s['RR_current']['bearing_force_n']==0.
for name in inventory:
    inventory[name]['sealed_full_sha256_not_rehashed']=(manifest.get('artifacts',{}).get(name) or {}).get('sha256')
final_info=decision[-1]['step_info']
result=dict(schema='readonly.v6_signed_gap_terminal_tail.v1',source=str(SOURCE),
    source_HEAD=manifest['runtime_contract']['source_git_commit'],checkpoint_load_provenance=manifest['checkpoint_load_provenance'],
    new_physics_ticks=0,new_PPO_updates=0,new_actor_forwards=0,inventory=inventory,
    window=dict(first_tick=14360,last_tick=14600,rows=241,first_negative_tick=14600,
        AIR_rows=sum(s['RR_current']['air'] for s in samples),
        any_active_pair=any(s['RR_raw_contact'][pair]['active'] for s in samples for pair in ('ground','obstacle')),
        max_pair_force_norm=max(sum(x*x for x in s['RR_raw_contact'][pair]['force_w_n'])**.5 for s in samples for pair in ('ground','obstacle'))),
    selected=[s for s in samples if s['tick'] in (14360,14480) or s['tick']>=14588],
    terminal=pick(final_info,('termination_reason','task_success','time_outs','task_outcome_label','full_task_success','terminal_bootstrap_allowed')),
    terminal_task=pick(final_info['semantic_task'],('stage_id','termination_reason','termination_source','stage_age_s','local_timeout',
        'completion_values','entry_valid','rr_capture_continuation','nominal_provider_diagnostics')))
out=HERE/'v6_negative_gap_terminal_readonly.json'
with out.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
print(json.dumps(dict(output=str(out),window=result['window'],terminal=result['terminal'],terminal_task={k:v for k,v in result['terminal_task'].items() if k!='nominal_provider_diagnostics'})))
for s in samples[-8:]:
    print(json.dumps(dict(tick=s['tick'],gap_m=s['gap_m'],front_m=s['front_m'],actual=s['actual_RR_deg'],target=s['target_RR_deg'],
        mode=s['state_after']['mode_name'],travel=s['state_after']['travel_used_deg'],exposure=s['state_after']['descent_elapsed_s'],
        peak=s['state_after']['window_start_gap_m'],window=s['state_after']['window_elapsed_s'],hold=s['state_after']['hold_elapsed_s'],
        continuation=s['post_continuation'])))
