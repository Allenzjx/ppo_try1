"""Bounded sealed deterministic front comparison; no policy forwards or physics."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'runs/ppo_rr_capture_then_rl_transfer_v1/video_eval/validation'
SOURCES = {
    'accepted_front_CP220544_v7': BASE / '20260923T0934264919029Z_g60abc00957c0_ef405598c8954e6280e844c4c29ed041/source',
    'learned_CP221952_v8': BASE / '20260923T1125359120678Z_gd1871df37d6e_ab5e4253c9fe431c9e6a77d3b4d19d3c/source',
}
TIMES = (4, 8, 12, 16, 19.2, 20, 22.4)
result = {'scope': 'sealed first22.5s only; no actor forwards/new learning', 'sources': {}}
for label, source in SOURCES.items():
    decisions = []
    with (source / 'video_policy_decisions.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            info = row['step_info']
            time = info['sim_time_s']
            if time > 22.5:
                break
            task = info['semantic_task']
            ev = task['physical_evaluator']
            fr = ev['current_legs']['FR']
            decisions.append({'time': time, 'tick': row['end_tick'], 'phase': task['stage_id'],
                'raw12': row['raw_policy_action_full12'], 'nominal12': info['nominal_action_full12'],
                'projected_residual12': info['projected_residual_full12'],
                'final12': info['actual_drive_target_full12'],
                'measured_wheel_velocity_rad_s': ev['measured_wheel_velocity_rad_s'],
                'FR': {k: fr.get(k) for k in ('front_distance_m','clearance_m','contact_surface','bearing_force_n')},
                'placed': ev['history']['placed'], 'terminal': task['termination_reason']})
    samples = [dict(min(decisions, key=lambda r: abs(r['time'] - t)), requested_time=t) for t in TIMES]
    needed = {r['tick'] for r in samples}
    actual = {}
    with (source / 'physical_observations.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            r = json.loads(line)
            if r['simulation_time_s'] > 22.5:
                break
            if r['physics_tick'] in needed:
                actual[r['physics_tick']] = {'actual12': r['actual_full12'], 'base': r['base'], 'CoM': r['center_of_mass']}
    for r in samples:
        r['physical_same_tick'] = actual.get(r['tick'])
    result['sources'][label] = {'path': str(source), 'samples': samples, 'last_endpoint': decisions[-1]}
(OUT / 'front_v7_v8_bounded_comparison.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
for label, record in result['sources'].items():
    print(label)
    for r in record['samples']:
        print(json.dumps({k:r[k] for k in ('time','phase','FR','final12','measured_wheel_velocity_rad_s')}, ensure_ascii=False))
