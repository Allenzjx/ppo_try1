"""Bounded read-only completed-run wheel audit; no policy forward or optimization."""
from pathlib import Path
import json
import math
import statistics
import argparse

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / 'runs/ppo_task_first_recovery_v1/video_eval/validation/20260916T0428416958414Z_gb0438f66ec63_292464ef72094ece8a8e494452b8aa23/source'
LEGS = ['FL', 'FR', 'RL', 'RR']
NAMES = ['front_left', 'front_right', 'rear_left', 'rear_right']
SIGNS = [-1, 1, -1, 1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source', type=Path, default=SOURCE)
parser.add_argument('--checkpoint', type=int, default=176768)
parser.add_argument('--full-media-stem', default='ppo_task_recovery_full_attempt')
parser.add_argument('--m1-media-stem', default='ppo_task_recovery_p01_to_fr_capture')
parser.add_argument('--output-stem', default='four_wheel_P02_checkpoint176768')
args = parser.parse_args()
SOURCE = args.source.resolve(strict=True)


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def rows(path):
    with path.open(encoding='utf8') as stream:
        for line in stream:
            yield json.loads(line)


def stats(values):
    return {'n': len(values), 'min': min(values), 'max': max(values), 'mean': statistics.fmean(values)}


run = read(SOURCE.parent / 'run_manifest.json')
assert run.get('completed_at_utc'), 'Only sealed run may be read'
native = {}
for row in rows(SOURCE / 'native_tick_audit.jsonl'):
    if row['source_phase_id'] not in ('P01', 'P02'):
        break
    native[row['episode_physics_tick']] = row
p02 = [t for t, row in native.items() if row['source_phase_id'] == 'P02']
end = max(p02)
physical = {}
for row in rows(SOURCE / 'physical_observations.jsonl'):
    if row['physics_tick'] > end:
        break
    physical[row['physics_tick']] = row
heights = {}
for row in rows(SOURCE / 'height_diagnostics.jsonl'):
    if row['physics_tick'] > end:
        break
    heights[row['physics_tick']] = row
wheel_data = read(OUT / f'videos/{args.full_media_stem}_wheel_qd.wheel_data.json')
binding = wheel_data['wheel_bindings']


def sample(t):
    row, p = native[t], physical[t]
    a = row['native_audit']
    final_native = a['actual_native_targets']['wheel_velocity_rad_s']
    final = [v*s for v, s in zip(final_native, SIGNS)]
    actual = [p['wheels'][name + '_ankle']['velocity_rad_s'] for name in NAMES]
    h = heights.get(t)
    # Same-tick raw robot.data.joint_vel readback is available only at height samples.
    native_qd = None if h is None else [h['joint_velocity_native_rad_s']['value'][binding[leg]['native_index']] for leg in LEGS]
    contacts = [p['contacts'][name + '_wheel'] for name in NAMES]
    return {
        'tick': t, 'sim_time_s': p['simulation_time_s'], 'phase': row['source_phase_id'],
        'source_N': row['nominal_full12'][8:], 'mapped_N': a['native_drive_target_full12'][8:],
        'actual_residual_mask': a['phase_mask_full12'][8:],
        'mask_object': 'PPO additive residual, not nominal or actuator write selection',
        'raw_policy': a['raw_policy_action_full12'][8:],
        'projected_policy': row['projected_residual_full12'][8:],
        'controller_bias': a['controller_drive_bias_full12'][8:],
        'final_canonical': final, 'final_native': final_native,
        'actual_canonical': actual, 'actual_native_measured': native_qd,
        'native_readback_source': 'same-tick height robot.data.joint_vel' if h else 'not sampled',
        'RR_target_deg': p['joints']['rear_right_knee']['command_deg'],
        'RR_actual_deg': p['joints']['rear_right_knee']['position_deg'],
        'FR_gap_m': p['wheels']['front_right_ankle']['bottom_w_m'][2] - p['obstacle']['top_z_m'],
        'base_origin_z_m': p['base']['position_w_m'][2],
        'CoM_z_m': p['center_of_mass']['position_w_m'][2],
        'contact_class': [c['contact_class'] for c in contacts],
        'ground_normal_force_N': [c['ground']['normal_force_n'] for c in contacts],
        'ground_active': [c['ground']['active'] for c in contacts],
        'obstacle_active': [c['obstacle']['active'] for c in contacts],
        'verified_dispatch': a['verified'],
        'setter_dispatch_targets_equal': a['setter_dispatch_targets_equal'],
        'actual_mapping_matches_dispatch': a['actual_mapping_matches_dispatch'],
    }


samples = {t: sample(t) for t in native}
assert all(r['verified_dispatch'] and r['setter_dispatch_targets_equal'] and r['actual_mapping_matches_dispatch'] for r in samples.values())
assert all(r['actual_residual_mask'] == [1, 1, 1, 1] for r in samples.values())
chain = {
    'ticks': len(samples), 'all12_mask_all_ticks': all(row['native_audit']['phase_mask_full12'] == [1]*12 for row in native.values()),
    'dispatch_and_actual_mapping_verified_all_ticks': True,
    'source_N_minus_mapped_N_max': max(abs(r['source_N'][j]-r['mapped_N'][j]) for r in samples.values() for j in range(4)),
    'controller_bias_abs_max': max(abs(v) for r in samples.values() for v in r['controller_bias']),
    'same_tick_composition_max_error': max(abs(r['final_canonical'][j]-(r['mapped_N'][j]+r['controller_bias'][j]+r['projected_policy'][j])) for r in samples.values() for j in range(4)),
    'native_qd_sample_count': sum(r['actual_native_measured'] is not None for r in samples.values()),
    'native_qd_vs_canonical_sign_max_error': max(abs(r['actual_native_measured'][j]*SIGNS[j]-r['actual_canonical'][j]) for r in samples.values() if r['actual_native_measured'] is not None for j in range(4)),
    'owner_limitation': 'No unique per-tick source owner field. Full source N is retained; this audit does not reassign ownership or infer it from changed_channels.',
    'last_writer_evidence': 'Existing full-vector wheel velocity setter; actual_native_targets read after write_data_to_sim, same-tick verified dispatch. No synthetic last-writer claim.',
}


def window(start, finish):
    ts = [t for t in samples if start <= t <= finish]
    data = []
    for j, leg in enumerate(LEGS):
        sel = [samples[t] for t in ts if samples[t]['source_N'][j] > 0]
        data.append({
            'leg': leg, 'positive_N_ticks': len(sel),
            'raw_negative_ticks': sum(r['raw_policy'][j] < 0 for r in sel),
            'raw_positive_ticks': sum(r['raw_policy'][j] > 0 for r in sel),
            'target_negative_ticks': sum(r['final_canonical'][j] < 0 for r in sel),
            'target_abs_below_005_ticks': sum(abs(r['final_canonical'][j]) < .05 for r in sel),
            'target_below_half_N_ticks': sum(r['final_canonical'][j] < .5*r['source_N'][j] for r in sel),
            'target_above_N_ticks': sum(r['final_canonical'][j] > r['source_N'][j] + 1e-6 for r in sel),
            'actual_abs_below_005_ticks': sum(abs(r['actual_canonical'][j]) < .05 for r in sel),
            **{key: stats([r[key][j] for r in sel]) for key in ('source_N', 'raw_policy', 'projected_policy', 'final_canonical', 'actual_canonical')},
        })
    return {'start_tick': start, 'end_tick': finish, 'included_ticks': len(ts), 'by_wheel': data}


def intervals(predicate):
    """Exact contiguous physical-tick runs, not smoothed/decision interpolated."""
    matches = [t for t in p02 if predicate(samples[t])]
    grouped = []
    for t in matches:
        if not grouped or t != grouped[-1][-1]+1:
            grouped.append([t])
        else:
            grouped[-1].append(t)
    return {'ticks': len(matches), 'duration_s': len(matches)/120,
            'intervals': [{'start_tick': g[0], 'end_tick': g[-1],
                           'start_sample_s': samples[g[0]]['sim_time_s'],
                           'end_sample_s': samples[g[-1]]['sim_time_s'],
                           'sampled_duration_s': len(g)/120} for g in grouped]}


old = read(ROOT / 'outputs/ppo_height_and_p02_recovery_v1/four_wheel_P02_checkpoint174592.json')
m1 = read(OUT / f'videos/{args.m1_media_stem}.media.json')
full = read(OUT / f'videos/{args.full_media_stem}.media.json')
result = {
    'schema': 'task_first_recovery_bounded_P02_wheels.v2',
    'source': str(SOURCE), 'source_manifest_sha256': full['source_manifest_sha256'],
    'checkpoint': args.checkpoint, 'source_sealed': True, 'wheel_order': LEGS,
    'canonical_to_native_signs': SIGNS, 'wheel_bindings': binding,
    'bounds': {'read_native_and_physical_through_tick': end, 'P02_start': min(p02), 'P02_end': end,
               'window_41_includes_late_P01': 41 < min(p02), 'no_new_forward_or_optimizer': True},
    'tick400': samples[400], 'chain': chain,
    'window41_through_P02_end': window(41, end),
    'actual_P02_window': window(min(p02), end),
    'matched_time_window41_812': window(41, 812),
    'P02_positive_N_intervals': intervals(lambda r: all(v > 0 for v in r['source_N'])),
    'P02_three_cancelled_RR_boosted_intervals': intervals(lambda r:
        all(v > 0 for v in r['source_N']) and
        all(r['final_canonical'][j] < .5*r['source_N'][j] for j in range(3)) and
        r['final_canonical'][3] > r['source_N'][3]+1e-6),
    'P02_target_below_half_positive_N_intervals_by_leg': {
        leg: intervals(lambda r, j=j: r['source_N'][j] > 0 and
                       r['final_canonical'][j] < .5*r['source_N'][j]) for j,leg in enumerate(LEGS)},
    'P02_actual_near_zero_positive_N_intervals_by_leg': {
        leg: intervals(lambda r, j=j: r['source_N'][j] > 0 and
                       abs(r['actual_canonical'][j]) < .05) for j,leg in enumerate(LEGS)},
    'old_C174592_tick400': old['samples']['400']['C174592'],
    'zero_tick400': old['samples']['400']['B_ZERO_FINAL_STOP'] if 'B_ZERO_FINAL_STOP' in old['samples']['400'] else {k:v for k,v in old['samples']['400'].items() if k not in ('C174592','C172544')},
    'old_C174592_positive_N_window41_812': old['runs']['C174592']['while_Npositive'],
    'M1_physical_evidence': m1['physical_evidence'],
    'full_result': {'physical_result': full['physical_result'], 'terminal_phase': full['terminal_phase'], 'physical_duration_s': full['physical_duration_s']},
    'attribution': 'Versioned mean-head recovery ancestry followed by real PPO updates and separately versioned execution fixes; see checkpoint/run records. This read-only audit is not a new optimizer update or single-factor attribution.',
    'limitations': ['Different policy trajectories are not same-state counterfactuals.', 'No independent wheel traction causality decomposition.', 'No reward/physics/HISTORY/mask modification made by this audit.', 'P02 bounds alone cannot establish full success; complete physical result is quoted separately from sealed media receipt.'],
}
assert chain['same_tick_composition_max_error'] < 1e-6
assert chain['native_qd_vs_canonical_sign_max_error'] == 0
with (OUT / f'{args.output_stem}.json').open('x', encoding='utf8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print(json.dumps({'bounds':result['bounds'],'tick400':samples[400], 'window':result['window41_through_P02_end'], 'chain':chain}, ensure_ascii=False, indent=2))
