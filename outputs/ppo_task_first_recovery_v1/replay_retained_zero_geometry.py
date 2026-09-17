"""Focused recorded-zero P09/P12 computational regression, never an Isaac run.

Extract the real contexts once. Replay only after the explicit nominal-history
API is ready; equality proves the pure adjustment for recorded inputs, not a
new physical success or all possible runtime history lifecycle behavior.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import collections
import hashlib
import inspect
import json
import struct
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / 'runs/ppo_fsm_reference_p09_stable_v2/video_eval/prior_B/20260915T0525295619176Z_g4a58c0190ef7_615f8fe3cfc64cff81f73ac6a0d302e6/source'
FIXTURE = OUT / 'retained_zero_P09_P12_geometry_fixture.json'


def load(path):
    return json.loads(path.read_text(encoding='utf8'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_new(path, value):
    with path.open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)


def extract():
    manifest = load(SOURCE / 'semantic_video_source_manifest.json')
    assert manifest['physical_task_success'] is True
    inventory = manifest['artifacts']['native_tick_audit.jsonl']
    digest = hashlib.sha256()
    selected = []
    all_ticks = 0
    zero_ticks = 0
    previous_tick = 0
    phases = collections.Counter()
    statuses = collections.Counter()
    native_path = SOURCE / 'native_tick_audit.jsonl'
    with native_path.open('rb') as stream:
        for line_no, raw in enumerate(stream, 1):
            digest.update(raw)
            row = json.loads(raw)
            tick, audit = row['episode_physics_tick'], row['native_audit']
            assert tick == previous_tick + 1
            previous_tick = tick
            all_ticks += 1
            assert all(value == 0 for value in row['projected_residual_full12'])
            assert all(value == 0 for value in audit['raw_policy_action_full12'])
            zero_ticks += 1
            geometry = audit.get('nominal_geometry_evidence')
            if geometry is None:
                continue
            assert row['source_phase_id'] in ('P09', 'P12')
            assert audit['verified'] and audit['actual_mapping_matches_dispatch'] and audit['setter_dispatch_targets_equal']
            reference = audit['tracking_reference_evidence']
            previous = audit['previous_final_drive_servo_deg']
            assert len(previous) == 8
            selected.append({
                'source_line': line_no, 'episode_tick': tick, 'phase': row['source_phase_id'],
                'dispatch_tick': audit['physics_tick'],
                'servo_order': audit['canonical_order'][:8],
                'recorded_previous_actual_final_servo_deg': previous,
                'nominal_previous_servo_deg_for_zero_replay': previous,
                'standing_pose_deg': reference['standing_pose_deg'],
                'maximum_delta_deg': reference['maximum_delta_deg'],
                'native_mapper_full12': audit['native_drive_target_full12'],
                'bounded_controller_full12': audit['controller_drive_bias_full12'],
                'context': geometry['context'],
                'expected_geometry_adjusted_native_full12': audit['geometry_adjusted_native_full12'],
                'expected_geometry_status': geometry['status'],
                'expected_desired_zero_policy_canonical_target_deg': geometry['desired_zero_policy_canonical_target_deg'],
                'expected_actual_native_targets': audit['actual_native_targets'],
            })
            phases[row['source_phase_id']] += 1
            statuses[row['source_phase_id'] + ':' + geometry['status']] += 1
    assert digest.hexdigest() == inventory['sha256'], 'Retained zero native log differs from its sealed manifest'
    assert selected and set(phases) == {'P09', 'P12'}
    fixture = {
        'schema': 'retained_zero_geometry_recorded_fixture.v1',
        'source': str(SOURCE), 'source_manifest_sha256': sha(SOURCE / 'semantic_video_source_manifest.json'),
        'native_audit_sha256': digest.hexdigest(), 'runtime_contract': manifest['runtime_contract'],
        'recorded_full_success': True, 'all_source_ticks': all_ticks, 'all_residual_zero_ticks': zero_ticks,
        'phase_geometry_active_counts': dict(phases), 'phase_status_counts': dict(statuses),
        'fixture_scope': 'Every recorded geometry-active P09/P12 call; exact old previous-final command, real Jacobian context, mapper output, bounded controller, standing pose and target.',
        'no_new_physics': True, 'rows': selected,
    }
    write_new(FIXTURE, fixture)
    print(json.dumps({key: fixture[key] for key in ('all_source_ticks', 'all_residual_zero_ticks', 'phase_geometry_active_counts', 'phase_status_counts')}, indent=2))


def replay(parameter, receipt):
    sys.path.insert(0, str(ROOT / 'src'))
    from wlr50_clean.ppo.semantic_nominal_geometry import correct_nominal_geometry
    from wlr50_clean.infrastructure.command_batch import Full12Command, build_physical_batch, servo_limits_deg
    from wlr50_clean.infrastructure.robot_adapter import bounded_drive_feedback_step

    assert parameter in inspect.signature(correct_nominal_geometry).parameters, 'Requested explicit nominal-history API is not ready'
    fixture = load(FIXTURE)
    phase_count = collections.Counter()
    numerical_differences = []
    target_float32_differences = []
    status_differences = []
    max_error = 0.0

    def f32(value):
        return struct.unpack('<f', struct.pack('<f', value))[0]

    for row in fixture['rows']:
        order = row['servo_order']
        standing = dict(zip(order, row['standing_pose_deg'], strict=True))
        adapter = SimpleNamespace(
            _final_drive_servo_deg=dict(zip(order, row['recorded_previous_actual_final_servo_deg'], strict=True)),
            servo_target_mapper=SimpleNamespace(maximum_delta_deg=row['maximum_delta_deg']),
            standing_pose_deg=standing,
        )
        kwargs = {parameter: tuple(row['nominal_previous_servo_deg_for_zero_replay'])}
        actual, evidence = correct_nominal_geometry(
            adapter=adapter, native_full12=row['native_mapper_full12'],
            controller_bias_full12=row['bounded_controller_full12'], context=row['context'], **kwargs)
        expected = row['expected_geometry_adjusted_native_full12']
        error = max(abs(a-b) for a, b in zip(actual, expected, strict=True))
        max_error = max(max_error, error)
        if tuple(actual) != tuple(expected):
            numerical_differences.append({'tick': row['episode_tick'], 'max_error': error})
        if evidence['status'] != row['expected_geometry_status']:
            status_differences.append(row['episode_tick'])
        final = []
        for i, name in enumerate(order):
            lower, upper = servo_limits_deg(name)
            final.append(bounded_drive_feedback_step(
                previous_deg=row['recorded_previous_actual_final_servo_deg'][i], native_deg=actual[i],
                bias_deg=row['bounded_controller_full12'][i], maximum_delta_deg=row['maximum_delta_deg'],
                lower_deg=lower, upper_deg=upper))
        wheels = tuple(actual[i]+row['bounded_controller_full12'][i] for i in range(8, 12))
        batch = build_physical_batch(Full12Command(tuple(final), wheels), standing)
        predicted = {'servo_position_rad': [f32(v) for v in batch.servo_target_rad],
                     'wheel_velocity_rad_s': [f32(v) for v in batch.wheel_target_rad_s]}
        if predicted != row['expected_actual_native_targets']:
            target_float32_differences.append({'tick': row['episode_tick'], 'predicted': predicted, 'recorded': row['expected_actual_native_targets']})
        phase_count[row['phase']] += 1
    result = {
        'schema': 'retained_zero_geometry_explicit_nominal_history_replay.v1',
        'fixture': str(FIXTURE), 'fixture_sha256': sha(FIXTURE),
        'source_manifest_sha256': fixture['source_manifest_sha256'],
        'explicit_nominal_history_parameter': parameter,
        'replayed_geometry_active_counts': dict(phase_count),
        'geometry_adjusted_target_exact_numeric_equality': not numerical_differences,
        'geometry_adjusted_target_max_error_deg_or_rad_s': max_error,
        'geometry_status_exact_equality': not status_differences,
        'actuator_target_float32_exact_equality': not target_float32_differences,
        'numerical_differences': numerical_differences,
        'status_differences': status_differences,
        'actuator_target_differences': target_float32_differences,
        'new_geometry_source_sha256': sha(ROOT / 'src/wlr50_clean/ppo/semantic_nominal_geometry.py'),
        'no_simulator': True, 'no_optimizer': True,
        'scope': 'Pure geometry and final-target reconstruction from sealed zero recorded inputs; not new physics success and not standalone proof of adapter history lifecycle.',
    }
    write_new(receipt, result)
    print(json.dumps({k:v for k,v in result.items() if not k.endswith('differences')}, indent=2))
    assert not numerical_differences and not status_differences and not target_float32_differences


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('extract')
    check = sub.add_parser('replay')
    check.add_argument('--nominal-history-parameter', required=True)
    check.add_argument('--receipt', type=Path, default=OUT / 'retained_zero_geometry_replay_receipt.json')
    args = parser.parse_args()
    if args.command == 'extract':
        extract()
    else:
        assert args.receipt.resolve().is_relative_to(OUT)
        replay(args.nominal_history_parameter, args.receipt)
