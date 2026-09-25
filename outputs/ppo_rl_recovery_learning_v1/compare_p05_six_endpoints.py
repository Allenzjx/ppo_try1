"""Bounded stdlib-only extraction: six sealed P05 endpoint comparisons."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCES = {
    'CP229120 formal DET': ROOT / 'runs/ppo_rr_rl_timing_policy_learning_v1/video_eval/validation/20260924T1008401445075Z_g72e63592bdf4_3f1b5877a524484aab1eecca97694515/source',
    'CP225280 original student probe': ROOT / 'runs/ppo_rr_rl_timing_policy_learning_v1/direction_probe/20260924_owner439_student_entry_v2',
}

def rows(path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)

result = []
for label, source in SOURCES.items():
    probe = 'probe' in label
    decision_path = source / ('diagnostic_decisions.jsonl' if probe else 'video_policy_decisions.jsonl')
    selected = {}
    for row in rows(decision_path):
        if probe and row['decision_index'] > 645:
            break
        step = row['step_info']
        task = step['semantic_task']
        if task['stage_id'] != 'P05':
            continue
        age = task['stage_age_s']
        for wanted in (2, 4, 6):
            distance = abs(age - wanted)
            if wanted not in selected or distance < selected[wanted][0]:
                selected[wanted] = distance, row
        if age > 6.1:
            break
    assert len(selected) == 3
    ticks = {item[1]['step_info']['physics_tick'] for item in selected.values()}
    physical = {}
    for row in rows(source / 'physical_observations.jsonl'):
        tick = row['physics_tick']
        if tick in ticks:
            physical[tick] = row
        if tick >= max(ticks):
            break
    assert set(physical) == ticks
    for wanted, (distance, row) in selected.items():
        assert distance < 1e-6
        if probe:
            assert row['original_student_raw_full12'] == row['applied_raw_full12']
            assert not row['probe_active'] and row['overridden_indices'] == []
        step = row['step_info']
        task = step['semantic_task']
        audit = step['actuator_target_effect_audit']
        p = physical[step['physics_tick']]
        assert len(p['actual_full12']) == 12 and len(step['actual_drive_target_full12']) == 12
        assert audit['actual_mapping_matches_dispatch'] and audit['setter_dispatch_targets_equal']
        result.append({
            'run': label, 'requested_P05_age_s': wanted, 'P05_age_s': task['stage_age_s'],
            'episode_time_s': step['sim_time_s'], 'episode_physics_tick': step['physics_tick'],
            'decision_id': row['decision_index'] if probe else row['decision'],
            'actuator_dispatch_tick_with_startup_offset': audit['physics_tick'],
            'source_N_full12': step['nominal_action_full12'],
            'mapped_N_full12': audit['native_drive_target_full12'],
            'final_ACK_full12': step['actual_drive_target_full12'],
            'actual_endpoint_full12': p['actual_full12'],
            'COM_world_m': p['center_of_mass']['position_w_m'],
            'base_world_m': p['base'].get('position_w_m'),
            'FL_gap_m': task['physical_evaluator']['current_legs']['FL']['clearance_m'],
            'FL_front_distance_m': task['physical_evaluator']['current_legs']['FL']['front_distance_m'],
            'legs': {leg: {key: value.get(key) for key in ('air', 'ground_contact', 'top_contact', 'within_top_xy', 'bearing_force_n')}
                     for leg, value in task['physical_evaluator']['current_legs'].items()},
            'source_decisions': str(decision_path.relative_to(ROOT)),
            'source_physical': str((source / 'physical_observations.jsonl').relative_to(ROOT)),
            'probe_no_intervention_verified': True if probe else None,
        })

result.sort(key=lambda row: (row['P05_age_s'], row['run']))
payload = {'schema': 'wlr50_clean.P05_six_endpoint_fullbody_readonly.v1',
    'joint_order': ['FL hip', 'FL knee', 'FR hip', 'FR knee', 'RL hip', 'RL knee', 'RR hip', 'RR knee'],
    'wheel_order': ['FL', 'FR', 'RL', 'RR'],
    'units': 'joint deg; wheel canonical forward-positive rad/s; position m',
    'scope': 'Six decision endpoints only. Last dispatched FINAL ACK versus endpoint measured response; no assertion that target was constant over all eight physics substeps. Two checkpoints/runtime versions and naturally different histories; not a single-factor causal experiment.',
    'rows': result}
json_path = OUT / 'CP229120_vs_CP225280_P05_six_endpoints.json'
md_path = OUT / 'CP229120_vs_CP225280_P05_six_endpoints.md'
for path in (json_path, md_path):
    if path.exists():
        raise FileExistsError(path)
json_path.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')

def vector(values, places=3, factor=1):
    return '[' + ', '.join(f'{v * factor:+.{places}f}' for v in values) + ']'

lines = ['# P05 age 2/4/6 s: sealed full-body comparison', '',
    'Joint order: **FL h/k, FR h/k, RL h/k, RR h/k** (degrees). Wheel order: **FL, FR, RL, RR** (canonical forward-positive rad/s). Positions are world xyz in mm.', '',
    'N is the source nominal request; mapped N is its logged execution baseline. FINAL is the authoritative actual-drive ACK. Actual values are the same decision endpoint physical observation, not a reconstructed target. Targets may have changed within the eight physics substeps. These are two different checkpoints/runtime versions and histories, not a single-factor causal comparison.', '',
    '| Run; P05 age; episode time / tick | Source N joints | Mapped N joints | FINAL joints | Actual joints | Wheel FINAL / actual | CoM xyz mm; FL gap/front mm |',
    '| --- | --- | --- | --- | --- | --- | --- |']
for row in result:
    lines.append('| ' + ' | '.join([
        f"{row['run']}; {row['P05_age_s']:.3f}s; {row['episode_time_s']:.4f}s / {row['episode_physics_tick']}",
        vector(row['source_N_full12'][:8], 2), vector(row['mapped_N_full12'][:8], 2),
        vector(row['final_ACK_full12'][:8], 2), vector(row['actual_endpoint_full12'][:8], 2),
        vector(row['final_ACK_full12'][8:]) + ' / ' + vector(row['actual_endpoint_full12'][8:]),
        vector(row['COM_world_m'], 2, 1000) + f"; {1000*row['FL_gap_m']:+.2f}/{1000*row['FL_front_distance_m']:+.2f}",
    ]) + ' |')
lines.extend(['', 'Both runs have source wheel requests [+0.300, +0.300, +0.300, +0.300] rad/s at these six endpoints. Full precision, contacts, exact source paths and endpoint identifiers are retained in the adjacent JSON. Only original probe rows at/before 645 were admitted, with raw == applied and no intervention verified for selected rows.', ''])
md_path.write_text('\n'.join(lines), encoding='utf-8')
print(json.dumps({'markdown': str(md_path), 'json': str(json_path), 'rows': result}, indent=2))
