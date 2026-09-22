"""Same41 historical input states only; no actor, tensors, fit or simulator."""
from collections import Counter
import itertools
import json
from pathlib import Path

import yaml
from wlr50_clean.ppo.semantic_supervisor import TaskStageSupervisor, _current_rr_receiver_preparation_retired

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
ROOT = OUT.parents[1]
RUN = ROOT / 'runs/ppo_p05_hip_only_continuation_v1/train/20260922T0705260953371Z_ga802b24d78df_76c646571d794b4e96a82f69956928ff'


def main():
    spec = yaml.safe_load((ROOT / 'configs/ppo_p05_hip_only_continuation_v1/stage_task_spec.yaml').read_text())
    current = object.__new__(TaskStageSupervisor); current.spec = spec
    records = []
    with (RUN / 'residual_and_projection_audit.jsonl').open('rb') as stream:
        for index, raw_line in enumerate(itertools.islice(stream, 1101)):
            if index < 1060:
                continue
            row = json.loads(raw_line); a = row['applied_audit']; ev = a['semantic_task']['physical_evaluator']
            tick = ev['physics_tick']
            assert tick == 8 * (index + 1) and 8488 <= tick <= 8808
            assert ev['valid'] and ev['termination_reason'] is None and not row['terminal']
            rr = ev['current_legs']['RR']; role = ev['transfer_roles']['RR']
            gate = bool(_current_rr_receiver_preparation_retired(spec, 'RR', ev))
            receiver = role['workspace_progress']
            row_out = {'input_tick': tick, 'historical_qualified': ev['history']['active_lift']['RR'],
                'historical_crossed': ev['history']['front_edge_crossed']['RR'], 'placed': ev['history']['placed']['RR'],
                'RR_gap_mm': 1000 * rr['clearance_m'], 'ground': rr['ground_contact'],
                'obstacle_contact': rr.get('obstacle_contact'), 'TOP': rr['top_contact'], 'AIR': rr['air'],
                'within_top_xy': rr['within_top_xy'], 'within_lateral_span': rr['within_lateral_span'],
                'front_distance_m': rr['front_distance_m'], 'retirement_gate': gate,
                'receiver_workspace_progress': receiver,
                'workspace_progress_current': current._workspace_potential_progress('RR', ev),
                'withheld_receiver_retirement_Phi_share': 0. if ev['history']['placed']['RR'] or gate else .85 / 4 * .1 * .5 * (1. - receiver),
                'retirement_share_note': 'pure algebraic soft-Phi contribution versus receiver=1 on this same physics; not a changed evaluator or simulated action'}
            fields = ('current_lift_valid', 'current_lift_valid_reason', 'lift_established', 'initial_now',
                'lift_established_now', 'motion_continuation_allowed', 'motion_continuation_reason',
                'ground_relative_lift_m', 'recent_wheel_displacement_m', 'edge_adjustment_response_observed',
                'body_control_evidence', 'observed_other_support_contacts', 'unsupported_free_lift_m',
                'free_air_reference_tick', 'free_air_reference_bottom_z_m', 'consecutive_free_air_samples',
                'wheel_bottom_vz_m_s', 'contact_mode', 'attempt_id', 'lift_attempt_id')
            row_out['recorded_fields'] = {key: rr[key] for key in fields if key in rr}
            row_out['fields_not_stored'] = [key for key in ('current_lift_valid_reason', 'attempt_id', 'lift_attempt_id') if key not in rr]
            record_checks = {'established': bool(rr['lift_established']), 'not_ground': not rr['ground_contact'],
                'body_control_evidence': bool(rr['body_control_evidence']),
                'ground_relative_geometry_valid': rr['ground_relative_lift_m'] is not None and rr['ground_relative_lift_m'] >= spec['history']['minimum_initial_clearance_gain_m'],
                'edge_response_not_required_or_present': rr['contact_mode'] in ('AIR', 'TOP') or bool(rr['edge_adjustment_response_observed'])}
            assert all(record_checks.values()) == rr['current_lift_valid']
            row_out['failing_current_lift_conjuncts'] = [key for key, value in record_checks.items() if not value]
            records.append(row_out)
    assert len(records) == 41
    before = [r for r in records if r['input_tick'] <= 8720]
    true_indices = [i for i, r in enumerate(before) if r['recorded_fields']['current_lift_valid']]
    assert len(before) == 30 and len(true_indices) == 4
    selected = sorted({j for i in true_indices for j in (i - 1, i, i + 1) if 0 <= j < 30})
    false_rows = [r for r in before if not r['recorded_fields']['current_lift_valid']]
    result = {'schema': 'wlr50_clean.same41_RR_current_lift_gate_readonly.v1',
        'input_scope': [8488, 8808], 'pre_capture_rows': 30, 'true_ticks': [before[i]['input_tick'] for i in true_indices],
        'false_count': len(false_rows), 'false_reason_conjunct_counts': dict(Counter(reason for r in false_rows for reason in r['failing_current_lift_conjuncts'])),
        'pre_capture_ground_rows': sum(r['ground'] for r in before),
        'pre_capture_established_false_rows': sum(not r['recorded_fields']['lift_established'] for r in before),
        'pre_capture_historical_Q_or_cross_false_rows': sum(not(r['historical_qualified'] and r['historical_crossed']) for r in before),
        'pre_capture_retirement_true_rows': sum(r['retirement_gate'] for r in before),
        'receiver_workspace_reenabled_rows': sum(not r['retirement_gate'] and not r['placed'] for r in before),
        'receiver_progress_in_false_rows_min_max': [min(r['receiver_workspace_progress'] for r in false_rows), max(r['receiver_workspace_progress'] for r in false_rows)],
        'withheld_receiver_Phi_share_in_false_rows_min_max_mean': [min(r['withheld_receiver_retirement_Phi_share'] for r in false_rows), max(r['withheld_receiver_retirement_Phi_share'] for r in false_rows), sum(r['withheld_receiver_retirement_Phi_share'] for r in false_rows) / len(false_rows)],
        'actual_minimum_other_supports': spec['support']['minimum_other_supports'],
        'actual_minimum_air_samples': spec['history']['minimum_air_samples'],
        'actual_minimum_initial_clearance_gain_m': spec['history']['minimum_initial_clearance_gain_m'],
        'four_true_and_adjacent_false_rows': [before[i] for i in selected],
        'all41_records': records, 'actor_evaluations': 0, 'fit_steps': 0,
        'evaluator_bug_or_fix_claimed': False, 'physical_contact_rule_changed': False}
    (HERE / 'same41_lift_gate_readonly.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    compact = [{key: row[key] for key in ('input_tick', 'RR_gap_mm', 'historical_qualified', 'historical_crossed', 'ground', 'TOP', 'AIR', 'retirement_gate', 'receiver_workspace_progress', 'withheld_receiver_retirement_Phi_share', 'recorded_fields', 'failing_current_lift_conjuncts')} for row in result['four_true_and_adjacent_false_rows']]
    print(json.dumps({**{key: value for key, value in result.items() if key not in ('all41_records', 'four_true_and_adjacent_false_rows')}, 'rows': compact}, indent=2))


if __name__ == '__main__':
    main()
