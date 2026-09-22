"""Existing514-source-window/current single live snapshot comparison; no actor."""
from collections import Counter
import itertools
import json
from pathlib import Path

from wlr50_clean.infrastructure.command_batch import servo_limits_deg

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
ROOT = OUT.parents[1]
RUN = ROOT / 'runs/ppo_p05_hip_only_continuation_v1/train/20260922T0705260953371Z_ga802b24d78df_76c646571d794b4e96a82f69956928ff'
LIVE = ROOT / 'runs/ppo_p05_hip_only_continuation_v1/video_eval/validation/20260922T1331110282117Z_g5fd88852bf20_737e9fd721cb45909f0de51d0f854df9'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def stats(values):
    values = list(values)
    return {'min': min(values), 'max': max(values), 'mean': sum(values) / len(values)} if values else None


def state(ev):
    joints = {}
    for role in ev['transfer_roles'].values():
        for name, margin in role['receiver_workspace_state']['joint_range_margin_deg'].items():
            low, high = servo_limits_deg(name)
            value = low + margin['negative_deg']
            assert abs(value - (high - margin['positive_deg'])) < 1e-9
            joints[name] = value
    rr = ev['current_legs']['RR']
    return {'tick': ev['physics_tick'], 'gap_mm': 1000 * rr['clearance_m'],
        'RR_front_mm': 1000 * rr['front_distance_m'], 'current_Q': rr['current_lift_valid'],
        'history_cross': ev['history']['front_edge_crossed']['RR'], 'history_placed': ev['history']['placed']['RR'],
        'RR_legal_XY': rr['within_top_xy'] and rr['within_lateral_span'],
        'body_collider_min_z_mm': 1000 * ev['body_traversal_geometry']['minimum_w_m'][2],
        'actual_joints_deg_from_verified_recorded_margins': joints,
        'supports': {leg: {'surface': row['contact_surface'], 'TOP': row['top_contact'],
            'ground': row['ground_contact'], 'AIR': row['air'],
            'verified_support': bool(row['support'] and row['bearing_verified']),
            'bearing_N': row['bearing_force_n']} for leg, row in ev['current_legs'].items()}}


def summary(rows):
    return {'rows': len(rows), 'gap_mm': stats(r['gap_mm'] for r in rows),
        'RR_front_mm': stats(r['RR_front_mm'] for r in rows),
        'body_collider_min_z_mm': stats(r['body_collider_min_z_mm'] for r in rows),
        'RR_current_Q_count': sum(r['current_Q'] for r in rows),
        'actual_joints_deg': {name: stats(r['actual_joints_deg_from_verified_recorded_margins'][name] for r in rows)
            for name in rows[0]['actual_joints_deg_from_verified_recorded_margins']} if rows else {},
        'support_rows': {leg: sum(r['supports'][leg]['verified_support'] for r in rows) for leg in ('FL', 'FR', 'RL', 'RR')},
        'contact_modes': {leg: dict(Counter(r['supports'][leg]['surface'] for r in rows)) for leg in ('FL', 'FR', 'RL', 'RR')}}


def main():
    manifest = read(LIVE / 'run_manifest.started.json')
    assert Path(manifest['arguments']['checkpoint']).name == 'checkpoint_step_000214400.pt'
    path = LIVE / 'source/video_policy_decisions.jsonl'
    with path.open('rb') as stream:
        stream.seek(0, 2); size = stream.tell(); stream.seek(max(0, size - 4 * 1024 * 1024))
        tail = stream.read().splitlines(keepends=True)
    complete = [line for line in tail[1:] if line.endswith(b'\n')]
    live_row = json.loads(complete[-1])
    infos = live_row['step_info']
    a = infos[-1] if isinstance(infos, list) else infos
    ev = a['semantic_task']['physical_evaluator']
    current = state(ev)
    if current['history_placed']:
        print(json.dumps({'stop_candidate_work': True, 'reason': 'current evaluated checkpoint has actual RR placed history', 'snapshot': current}))
        return
    assert ev['valid'] and ev['termination_reason'] is None
    old_states = []
    with (RUN / 'residual_and_projection_audit.jsonl').open('rb') as stream:
        for index, raw in enumerate(itertools.islice(stream, 1260)):
            if index < 746:
                continue
            row = json.loads(raw); old_ev = row['applied_audit']['semantic_task']['physical_evaluator']
            value = state(old_ev)
            # Endpoints index746..1259 are exact subsequent inputs747..1260.
            assert value['tick'] == 8 * (index + 1)
            value['input_source_index'] = index + 1
            old_states.append(value)
    assert len(old_states) == 514
    matched_height = [r for r in old_states if 45 <= r['gap_mm'] <= 55 and not r['history_placed']]
    crossed_pre = [r for r in old_states if r['tick'] <= 8720]
    last2 = [r for r in old_states if 8488 <= r['tick'] <= 8720]
    nearest = min(crossed_pre, key=lambda r: abs(r['gap_mm'] - current['gap_mm']))
    result = {'schema': 'wlr50_clean.full12_approach_coverage_readonly.v1', 'live_snapshot_not_final': current,
        'live_decision': live_row['decision'], 'live_phase': a['end_phase_id'],
        'live_source': str(path), 'source_full514': summary(old_states),
        'source_crossed_precapture344': summary(crossed_pre), 'source_last2s30': summary(last2),
        'source_gap45to55_unplaced': summary(matched_height), 'nearest_source_gap_not_nearest_full_state': nearest,
        'actual_angle_semantics': 'exact inversion of logged actual-angle joint-limit margins, not desired/final targets',
        'full12_original_raw_labels_available': True,
        'source_actor_and_rollout_binding_reused_from': str(HERE / 'availability_report.json'),
        'source_X17_migration_required_as_previously_quantified': True,
        'possible_window_without_P12': {'source_indices': [747, 1259], 'global_decisions': [204524, 205036],
            'input_ticks': [5976, 10072], 'rows': 513, 'P09': 345, 'P10': 1, 'P11': 167},
        'no_new_dataset_fit_or_budget': True,
        'causality_or_old_action_current_success_claimed': False}
    (HERE / 'full12_approach_coverage_readonly.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
