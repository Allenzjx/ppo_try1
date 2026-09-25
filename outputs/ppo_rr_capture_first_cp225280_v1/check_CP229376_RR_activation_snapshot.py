"""Read only first1006 decisions from a fixed byte snapshot; no native/model scan."""
import json
from pathlib import Path
from analyze_actual_metrics import DEFAULT_REFERENCE, delta, vector

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
RUN = ROOT / 'runs/ppo_rr_capture_first_cp225280_v1/eval_CP229376_gain10_aux64_DET_0d0f894'

def read_rows(path, count):
    rows = []
    with path.open('rb') as stream:
        stream.seek(0, 2)
        boundary = stream.tell()
        stream.seek(0)
        for _ in range(count):
            line = stream.readline(boundary - stream.tell())
            if not line.endswith(b'\n'):
                raise ValueError('Requested complete decision unavailable; no wait')
            rows.append(json.loads(line))
        status = {'path': str(path), 'snapshot_bytes': boundary, 'size_source': 'open_handle_seek_END_tell',
                  'actual_read_end': stream.tell(), 'complete_rows': len(rows), 'later_bytes_not_read': boundary-stream.tell()}
    return rows, status

current, source_status = read_rows(RUN / 'source/video_policy_decisions.jsonl', 1006)
reference, reference_status = read_rows(DEFAULT_REFERENCE, 998)
prefix = [r for r in current if r['policy_request']['capture_active'] is False]
active = [r for r in current if r['policy_request']['capture_active'] is True]
assert len(prefix) == len(reference) == 998 and len(active) == 8
checks = {
    'raw_full12_exact_reference': [], 'FINAL_full12_exact_reference': [], 'local_raw_zero': [],
    'applied_local_zero': [], 'combined_equals_prior': [], 'local_forward_zero': [], 'history_once': [],
    'issued_equals_prior_conditional': [], 'no_diagnostic_override': [], 'PPO_credit0': [],
}
max_raw = max_final = 0.0
for row, ref in zip(prefix, reference):
    assert (row['start_tick'],row['end_tick']) == (ref['start_tick'],ref['end_tick'])
    info, p = row['step_info'], row['policy_request']
    dr = delta(info['raw_policy_action_full12'], ref['step_info']['raw_policy_action_full12'])
    df = delta(info['actual_drive_target_full12'], ref['step_info']['actual_drive_target_full12'])
    max_raw, max_final = max(max_raw,dr), max(max_final,df)
    values = [dr==0, df==0,
              vector(p['local_raw_mean_delta_full12']) and all(x==0 for x in p['local_raw_mean_delta_full12']),
              vector(p['applied_local_raw_mean_delta_full12']) and all(x==0 for x in p['applied_local_raw_mean_delta_full12']),
              delta(p['combined_raw_mean_full12'],p['prior_raw_mean_full12'])==0,
              p['local_network_forwards']==0,p['history_kernel_applications']==1,
              delta(info['raw_policy_action_full12'],p['prior_same_observation_conditional_mean_full12'])==0,
              'independent_diagnostic' not in p,p['prefix_excluded_from_new_PPO_credit'] is True]
    for name, passed in zip(checks,values): checks[name].append(passed)
assert all(all(v) for v in checks.values())

def events(row):
    return row['step_info']['semantic_task']['physical_evaluator']['history']['event_ticks']
actual_events, original_events = events(current[-1]), events(reference[-1])
event_compare = {}
for label,kind,leg in [('FRplaced','placed','FR'),('FLplaced','placed','FL'),('RRqualified','active_lift','RR'),('RRcross','front_edge_crossed','RR')]:
    a,b = actual_events[kind][leg], original_events[kind][leg]
    event_compare[label] = {'current_tick':a,'CP225280_tick':b,'equal':a==b}
assert all(e['equal'] for e in event_compare.values())
entry = prefix[-1]['step_info']['rr_capture_local']
assert entry['activation_tick']==actual_events['front_edge_crossed']['RR']==7979
details=[]
for row in active:
    info,p=row['step_info'],row['policy_request']
    local=info['rr_capture_local']; audit=info['actuator_target_effect_audit']; h=audit['policy_headroom_evidence']
    assert p['selected_raw_full12']==info['raw_policy_action_full12'] and p['history_kernel_applications']==1
    assert p['sampling_draws']==0 and p['extra_random_draws']==0 and 'independent_diagnostic' not in p
    details.append({'decision':row['decision'],'start_tick':row['start_tick'],'end_tick':row['end_tick'],
        'metrics':local['metrics'],'hold_s':local['hold_elapsed_s'],
        'source_RR_deg':info['nominal_action_full12'][6:8],'mappedN_RR_deg':audit['native_drive_target_full12'][6:8],
        'generic_bias_RR_deg':audit['controller_drive_bias_full12'][6:8],
        'prior_raw_mean_RR':p['prior_raw_mean_full12'][6:8],
        'raw_local_head_RR':p['local_raw_mean_delta_full12'][6:8],
        'gain_full12':p['local_mean_coordinate_gain_full12'],
        'applied_local_raw_mean_RR':p['applied_local_raw_mean_delta_full12'][6:8],
        'combined_raw_mean_RR':p['combined_raw_mean_full12'][6:8],
        'history_center_RR':p['history_center_full12'][6:8], 'rho':p['rho'],
        'conditional_mean_RR':p['conditional_mean_full12'][6:8], 'issued_raw_RR':info['raw_policy_action_full12'][6:8],
        'requested_residual_RR_deg':h['requested_policy_residual_full12'][6:8],
        'effective_residual_RR_deg':h['effective_policy_residual_full12'][6:8],
        'FINAL_RR_deg':info['actual_drive_target_full12'][6:8],
        'local_forwards':p['local_network_forwards'],'history_applications':p['history_kernel_applications'],
        'all12_policy_channels_unmodified_at_actuator':audit['all12_policy_channels_unmodified_at_actuator']})
manifest=json.loads((RUN/'run_manifest.started.json').read_text(encoding='utf-8'))
report={'schema':'wlr50_clean.CP229376_DET_RR_activation_live.v1','live_readonly':True,
    'scope':'Fixed byte snapshot, decisions1..1006 only;998inactive+first8active. Reference read only through998. No native120Hz scan, no prediction of later capture.',
    'checkpoint':manifest['checkpoint'],'runtime_head':manifest['runtime_contract']['source_git_commit'],
    'current_source':source_status,'accepted_CP225280_reference_source':reference_status,
    'prefix':{'checked_rows':998,'raw_full12_max_abs_delta':max_raw,'FINAL_full12_max_abs_delta':max_final,
              'per_check_pass_counts':{k:sum(v) for k,v in checks.items()},'missing_reference_rows':0},
    'events':event_compare,'activation':{'gate_tick':entry['activation_tick'],'time_s':entry['activation_time_s'],
        'gate_rule':'Native qualified RR cross/current geometry condition; original reference suppliescross but not this localgate metadata',
        'first_visible_end_tick':prefix[-1]['end_tick'],'last_inactive_request_start_tick':prefix[-1]['start_tick'],
        'first_active_request_start_tick':active[0]['start_tick'],
        'entry_FINAL_RR_deg':[entry['entry_rr_hip_deg'],entry['entry_rr_knee_deg']],
        'entry_gap_m':entry['entry_gap_m']},
    'active_first8':details,
    'actual_scope_totals':{'active_requests':8,'TOP_endpoint_count':sum(x['metrics']['current_top_contact'] for x in details),
        'local_success_endpoints':sum(r['step_info']['local_task_success'] for r in active),'new_PPO_credit':0,'new_AUX_credit':0},
    'limits':['Gain10 applies local coordinates only after gate, followed by one HISTORY kernel; not a factor10 FINAL-angle multiplier.',
              'These are deterministic evaluation requests, not learning samples; no claim for subsequent RR segment or full traversal.']}
target=OUT/'CP229376_DET_RR_activation_live.json'
target.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps({'output':str(target),'prefix':report['prefix'],'events':event_compare,'activation':report['activation'],
                  'last':{'tick':details[-1]['end_tick'],'actual_RR':details[-1]['metrics']['actual_rr_hip_knee_deg'],
                          'FINAL_RR':details[-1]['FINAL_RR_deg'],'gap_mm':details[-1]['metrics']['gap_m']*1000},
                  'totals':report['actual_scope_totals']}))
