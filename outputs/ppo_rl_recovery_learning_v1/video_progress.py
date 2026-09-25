"""Read one complete live video decision; stdlib only, no simulator imports."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from live_progress import last_complete_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    snapshot = last_complete_json(args.source / 'video_policy_decisions.jsonl', max_row_bytes=4_194_304)
    row = snapshot.get('row', {})
    info = row.get('step_info', {})
    physical = info.get('semantic_task', {}).get('physical_evaluator', {})
    leg_fields = ('air', 'contact_mode', 'top_contact', 'ground_contact',
                  'bearing_force_n', 'bearing_verified', 'current_lift_valid',
                  'clearance_m', 'front_distance_m')
    print(json.dumps(dict(
        checked_at_utc=datetime.now(timezone.utc).isoformat(),
        state=snapshot['state'], trailing_uncommitted_bytes=snapshot.get('trailing_uncommitted_bytes'),
        decision=row.get('decision'), phase=info.get('phase_id'), sim_time_s=info.get('sim_time_s'),
        termination_reason=info.get('termination_reason'), task_success=info.get('full_task_success'),
        events=physical.get('history', {}).get('event_ticks'),
        legs={leg: {key: values.get(key) for key in leg_fields}
              for leg, values in physical.get('current_legs', {}).items()},
    ), separators=(',', ':'), allow_nan=False))


if __name__ == '__main__':
    main()
