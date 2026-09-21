"""Bounded live read via open file handle (Windows directory size can be stale)."""
import json
from pathlib import Path
import sys


def last(path):
    if not path.exists():
        return {}
    with path.open('rb') as stream:
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(max(0, size-4*1024*1024))
        lines = stream.read().splitlines()
    for line in reversed(lines):
        try:
            value = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            continue
        return value
    return {}


run = Path(sys.argv[1])
if (run/'probe_decisions.jsonl').exists():
    row = last(run/'probe_decisions.jsonl')
    info = row.get('step_info', {})
elif (run/'source/video_policy_decisions.jsonl').exists():
    row = last(run/'source/video_policy_decisions.jsonl')
    info = row.get('step_info') or {}
else:
    row = last(run/'residual_and_projection_audit.jsonl')
    info = row.get('applied_audit') or {}
task = info.get('semantic_task') or {}
ev = task.get('physical_evaluator') or {}
update = last(run/'optimizer_updates.jsonl')
prefix = last(run/'prefix_evidence.jsonl')
result = {
    'run': str(run), 'global_decision': row.get('global_policy_decision'),
    'tick': info.get('physics_tick', row.get('end_tick')), 'phase': info.get('phase_id', row.get('request_phase')),
    'termination': task.get('termination_reason'), 'placed': ev.get('history', {}).get('placed'),
    'FL': {k: ev.get('current_legs', {}).get('FL', {}).get(k) for k in ('air', 'support', 'clearance_m', 'front_distance_m', 'bearing_force_n')},
    'RR': {k: ev.get('current_legs', {}).get('RR', {}).get(k) for k in ('air', 'support', 'clearance_m', 'front_distance_m', 'bearing_force_n')},
    'event_ticks': ev.get('history', {}).get('event_ticks'),
    'intervention': row.get('intervention'),
    'update': {k: update.get(k) for k in ('ppo_update', 'global_policy_decisions', 'kl_mean', 'optimizer_learning_rate')},
    'sealed': (run/'run_manifest.json').exists(),
    'last_prefix': {k: prefix.get(k) for k in ('phase_id', 'end_phase_id', 'physics_tick', 'sim_time_s', 'termination_reason')}
        if prefix else None,
}
print(json.dumps(result, ensure_ascii=False))
