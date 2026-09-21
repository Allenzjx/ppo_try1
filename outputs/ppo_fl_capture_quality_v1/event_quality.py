"""Read sealed physical episodes; event-aligned predeclared quality only."""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from wlr50_clean.ppo.semantic_observation import _multiply, _quaternion, _rpy, load_semantic_observation_schema


def rows(path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def analyze(source):
    source = Path(source).resolve(strict=True)
    run = json.loads((source.parent / 'run_manifest.json').read_text(encoding='utf-8'))
    assert run.get('completed_at_utc') and run['lifecycle'] != 'RUNNING'
    manifest = json.loads((source / 'semantic_video_source_manifest.json').read_text(encoding='utf-8'))
    assert manifest['from_phase'] == 'P01' and manifest['episode_count'] == 1
    events = {'active_lift': {}, 'front_edge_crossed': {}, 'placed': {}}
    phases = {}
    for row in rows(source / 'video_policy_decisions.jsonl'):
        phases.setdefault(row['request_phase'], row['start_tick'])
        ev = (row.get('step_info') or {}).get('semantic_task', {}).get('physical_evaluator', {})
        for kind in events:
            for leg, tick in ev.get('history', {}).get('event_ticks', {}).get(kind, {}).items():
                events[kind].setdefault(leg, tick)
    final = manifest['physical_episode']['physical_task_evaluation']
    for kind in events:
        for leg, tick in final['history']['event_ticks'][kind].items():
            events[kind].setdefault(leg, tick)
    schema_path = ROOT / 'configs' / ('ppo_' + manifest['experiment_id']) / 'observation_schema.json'
    schema = load_semantic_observation_schema(schema_path)
    measured = []
    previous = None
    for row in rows(source / 'physical_observations.jsonl'):
        tick = row['physics_tick']
        q = _quaternion(row['base']['orientation_wxyz'])
        rpy = _rpy(_quaternion(_multiply(q, schema.fixed_chassis_to_body_wxyz)))
        rates = None
        if previous is not None:
            assert tick == previous[0] + 1, 'Missing adjacent physical sample'
            dt = row['simulation_time_s'] - previous[1]
            assert 0 < dt < .009
            rates = [math.atan2(math.sin(a-b), math.cos(a-b))/dt for a, b in zip(rpy[:2], previous[2][:2])]
        measured.append((tick, rpy, rates))
        previous = tick, row['simulation_time_s'], rpy
    last = measured[-1][0]
    definitions = {
        'FR_preparation_to_capture': (0, events['placed'].get('FR')),
        'FL_active_lift_to_capture': (events['active_lift'].get('FL'), events['placed'].get('FL')),
        'FL_cross_to_capture': (events['front_edge_crossed'].get('FL'), events['placed'].get('FL')),
    }
    windows = {}
    for label, (start, end) in definitions.items():
        if start is None or end is None:
            windows[label] = {'complete': False, 'start_tick': start, 'end_tick': end,
                              'rate_rms_rad_s': None, 'tilt_peak_rad': None}
            continue
        subset = [r for r in measured if start <= r[0] <= end]
        rated = [r for r in subset if r[2] is not None and r[0] > start]
        assert subset and rated
        windows[label] = {
            'complete': True, 'start_tick': start, 'end_tick': end, 'duration_s': (end-start)/120,
            'physical_samples': len(subset), 'adjacent_rate_samples': len(rated),
            'rate_rms_rad_s': math.sqrt(sum(sum(x*x for x in r[2])/2 for r in rated)/len(rated)),
            'tilt_peak_rad': max(math.hypot(*r[1][:2]) for r in subset),
        }
    return {'source': str(source), 'experiment_id': manifest['experiment_id'], 'role': manifest['role'],
            'sampling_mode': manifest.get('policy_sampling_mode', 'deterministic_or_zero_legacy_manifest'),
            'full_task_success': manifest['physical_task_success'], 'physical_duration_s': last/120,
            'events_first_observed': events, 'phase_start_ticks': phases, 'windows': windows,
            'task_result_not_inferred_from_quality': True, 'missing_stages_are_not_zero': True,
            'predeclared_protocol': str(Path(__file__).with_name('QUALITY_PROTOCOL.md'))}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.source)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))
