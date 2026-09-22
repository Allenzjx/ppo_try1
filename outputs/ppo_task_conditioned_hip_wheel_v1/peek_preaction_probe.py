"""Read only a bounded complete JSON tail; pre and returned states stay separate."""
import json
from pathlib import Path
import sys

run = Path(sys.argv[1])
path = run / 'probe_decisions.jsonl'
records = {}
if path.exists():
    with path.open('rb') as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - 4 * 1024 * 1024))
        lines = stream.read().splitlines()
    for line in reversed(lines):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            continue
        kind = row.get('record_kind')
        if kind not in records:
            records[kind] = row
        if 'pre_action' in records and 'step_result' in records:
            break
pre = records.get('pre_action', {})
post = records.get('step_result', {})
info = post.get('step_info', {})
task = info.get('semantic_task', {})
ev = task.get('physical_evaluator', {})
fl = ev.get('current_legs', {}).get('FL', {})
summary = {
    'run': str(run), 'sealed': (run / 'run_manifest.json').exists(),
    'latest_issued_pre': {k: pre.get(k) for k in ('decision', 'start_tick', 'phase', 'intervention')},
    'latest_returned': {'decision': post.get('decision'), 'end_tick': post.get('end_tick'),
        'phase': info.get('end_phase_id'), 'termination': task.get('termination_reason'),
        'FL': {k: fl.get(k) for k in ('air', 'support', 'clearance_m', 'bearing_force_n')},
        'placed': ev.get('history', {}).get('placed'),
        'event_ticks': ev.get('history', {}).get('event_ticks')},
    'step_exception': {k: records.get('step_exception', {}).get(k) for k in
        ('decision', 'exception', 'endpoint_tick', 'final_phase')}
        if 'step_exception' in records else None,
    'pre_and_post_can_be_different_decisions': True,
}
print(json.dumps(summary, ensure_ascii=False))
