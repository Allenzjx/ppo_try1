"""Stdlib only: fixed live byte snapshot, limited to requested P06 decision652."""
import json
from pathlib import Path
from collections import Counter
from analyze_actual_metrics import DEFAULT_REFERENCE, delta, vector

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
RUN = ROOT / 'runs/ppo_rr_capture_first_cp225280_v1/eval_CP229376_gain10_aux64_DET_0d0f894'
SOURCE = RUN / 'source/video_policy_decisions.jsonl'
LAST_TICK = 5216

def read_prefix(path, consume):
    with path.open('rb') as stream:
        stream.seek(0, 2)
        boundary = stream.tell()
        stream.seek(0)
        count = 0
        while stream.tell() < boundary:
            line = stream.readline(boundary - stream.tell())
            if not line.endswith(b'\n'):
                raise ValueError('Requested prefix is not fully written at fixed snapshot boundary')
            row = json.loads(line)
            assert row['end_tick'] <= LAST_TICK
            consume(row)
            count += 1
            if row['end_tick'] == LAST_TICK:
                return {'path': str(path), 'snapshot_bytes': boundary, 'snapshot_size_source': 'open_handle_seek_END_tell',
                        'actual_consumed_bytes': stream.tell(), 'complete_rows_checked': count,
                        'through_tick': LAST_TICK, 'through_time_s': LAST_TICK / 120,
                        'later_snapshot_bytes_not_read': boundary - stream.tell()}
    raise ValueError('Requested P06 endpoint unavailable; do not wait or extend scope')

reference = {}
reference_events = {}
def baseline(row):
    info = row['step_info']
    reference[(row['start_tick'], row['end_tick'])] = (info['raw_policy_action_full12'], info['actual_drive_target_full12'])
    reference_events.update(info['semantic_task']['physical_evaluator']['history']['event_ticks'].get('placed', {}))
reference_status = read_prefix(DEFAULT_REFERENCE, baseline)
checks = Counter()
phases = Counter()
max_raw = max_final = 0.0
events = {}
assist_active_ticks = []
assist_owner_sets = set()
bad = []
last = None

def current(row):
    global max_raw, max_final, last
    info, policy = row['step_info'], row['policy_request']
    assist = info['actuator_target_effect_audit']['capture_assist_evidence']
    values = {
        'capture_active_false': policy.get('capture_active') is False,
        'local_raw_head_all_zero': vector(policy.get('local_raw_mean_delta_full12')) and all(v == 0 for v in policy['local_raw_mean_delta_full12']),
        'applied_local_delta_all_zero': vector(policy.get('applied_local_raw_mean_delta_full12')) and all(v == 0 for v in policy['applied_local_raw_mean_delta_full12']),
        'combined_mean_equals_prior': delta(policy.get('combined_raw_mean_full12'), policy.get('prior_raw_mean_full12')) == 0,
        'issued_raw_equals_prior_HISTORY_mean': delta(info.get('raw_policy_action_full12'), policy.get('prior_same_observation_conditional_mean_full12')) == 0,
        'selected_raw_equals_issued': delta(policy.get('selected_raw_full12'), info.get('raw_policy_action_full12')) == 0,
        'local_forward_count_zero': policy.get('local_network_forwards') == 0,
        'HISTORY_once': policy.get('history_kernel_applications') == 1,
        'sampling_draws_zero': policy.get('sampling_draws') == 0 and policy.get('extra_random_draws') == 0,
        'no_independent_diagnostic_override': 'independent_diagnostic' not in policy,
        'prefix_excluded_from_PPO': policy.get('prefix_excluded_from_new_PPO_credit') is True,
        'FL_assist_runtime_version': assist['state_after']['version'] == 'p05_hip_only_continuation_v1',
        'FL_assist_corrections_only_FL_hip_knee': all(v == 0 for v in assist['assist_correction_full12'][2:]),
        'FL_assist_owners_only_0_1': set(assist['owner_indices']) <= {0, 1},
        'FL_assist_policy_request_unchanged': assist['policy_request_unchanged'] is True,
        'gain10_only_RR_coordinates': policy['local_mean_coordinate_gain_full12'] == [1.,1.,1.,1.,1.,1.,10.,10.,1.,1.,1.,1.],
    }
    for name, passed in values.items():
        checks[name] += int(passed)
    if not all(values.values()):
        bad.append({'decision': row['decision'], 'checks': values})
    other = reference[(row['start_tick'], row['end_tick'])]
    dr, df = delta(info['raw_policy_action_full12'], other[0]), delta(info['actual_drive_target_full12'], other[1])
    assert dr is not None and df is not None
    max_raw, max_final = max(max_raw, dr), max(max_final, df)
    phases[info['end_phase_id']] += 1
    events.update(info['semantic_task']['physical_evaluator']['history']['event_ticks'].get('placed', {}))
    if assist['state_after']['active']:
        assist_active_ticks.append(row['end_tick'])
    assist_owner_sets.add(tuple(assist['owner_indices']))
    last = {'decision': row['decision'], 'tick': row['end_tick'], 'phase': info['end_phase_id'],
            'local_active': info['rr_capture_local']['active'], 'FL_assist_state': assist['state_after'],
            'FL_assist_context': assist['context'], 'FL_assist_correction_full12': assist['assist_correction_full12']}

current_status = read_prefix(SOURCE, current)
manifest = json.loads((RUN / 'run_manifest.started.json').read_text(encoding='utf-8'))
contract = manifest['runtime_contract']['local_contract']
count = current_status['complete_rows_checked']
assert count == 652 and not bad and max_raw == 0 and max_final == 0
assert last['phase'] == 'P06' and not last['local_active']
report = {
    'schema': 'wlr50_clean.CP229376_DET_P06_prefix_live.v1', 'live_readonly': True,
    'scope': 'Exactly requested decisions1..652 through P06/tick5216; subsequent bytes deliberately not read, no RR forecast.',
    'checkpoint': manifest['checkpoint'], 'runtime_head': manifest['runtime_contract']['source_git_commit'],
    'current_source': current_status, 'accepted_CP225280_reference_source': reference_status,
    'reference_alignment': {'key': '[start_tick,end_tick]', 'compared_rows': count, 'missing_rows': 0,
                            'raw_full12_max_abs_delta': max_raw, 'FINAL_full12_max_abs_delta': max_final,
                            'numeric_array_equality_all_rows': True, 'whole_JSON_equality_claimed': False},
    'per_check_passed_rows': dict(checks), 'all_checks_passed_all652': True, 'bad_rows': bad,
    'end_phase_counts': dict(phases),
    'placed_events': {leg: {'actual_tick': events.get(leg), 'reference_tick': reference_events.get(leg),
                           'equal': events.get(leg) == reference_events.get(leg),
                           'source': 'physical_evaluator.history.event_ticks.placed'} for leg in ('FR', 'FL')},
    'declared_runtime_helpers': {k: contract[k] for k in ('front_FL_assist', 'rear_task_assist', 'rear_owner_projection', 'source_tracking_owner_revision')},
    'actual_FL_assist': {'active_endpoint_count': len(assist_active_ticks), 'first_active_endpoint': min(assist_active_ticks),
                         'last_active_endpoint': max(assist_active_ticks), 'owner_sets': sorted(assist_owner_sets),
                         'actual_corrections_outside_FL_hip_knee': 0},
    'rear_off_evidence_boundary': 'Runtime contract rear_task_assist=false and rear_owner_projection=false; actual capture-assist receipts contain only declared FL owner0/1 and zero corrections on other10. No independent rear diagnostic override. This does not claim every generic controller/tracking bias is zero.',
    'latest_in_scope': last,
    'interpretation': 'Gain10 metadata is present but local head is not called and local/applied deltas are exactlyzero; inactive raw and FINAL remain numerically identical to accepted CP225280. No new PPO/AUXcredit in evaluation.',
}
target = OUT / 'CP229376_DET_P06_prefix_live.json'
target.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
print(json.dumps({'output': str(target), 'compared': count, 'raw_max_delta': max_raw, 'FINAL_max_delta': max_final,
                  'placed': report['placed_events'], 'actual_FL_assist': report['actual_FL_assist'], 'actual_read_bytes': current_status['actual_consumed_bytes']}))
