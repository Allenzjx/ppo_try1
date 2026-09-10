"""Bounded stdlib summary of an actually finalized all-stage natural-P01 C.

No simulation, checkpoint tensor/hash load, production import or full trajectory
scan. Source manifests + checkpoint sidecar + first/last audit and physical rows.
Writes an exclusive new report directory only; never updates a master manifest.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

OUTPUT_ROOT = Path(__file__).resolve().parent
PROJECT = OUTPUT_ROOT.parents[1]
INPUT_ROOT = PROJECT / 'runs/ppo_all_stage_acceptance_v1/validation'
PHASES = tuple(f'P{i:02}' for i in range(1, 14))
LEGS = ('FL', 'FR', 'RL', 'RR')
WRITES = ('in_episode_root_pose_writes', 'in_episode_root_velocity_writes',
          'in_episode_force_or_impulse_writes', 'in_episode_gravity_writes')
COMMON = ('valid', 'evaluator_version', 'run_validity', 'physical_evidence_status',
          'termination_reason', 'termination_source', 'reason', 'success',
          'traversal_event_observed', 'traversal_event_time_s', 'traversal_task_complete',
          'task_completed_controlled', 'strict_recovery_quality', 'body_traversal_geometry',
          'post_completion_elapsed_s', 'post_completion_observation_complete',
          'post_completion_loss_observed', 'final_region_valid', 'final_controlled',
          'final_support_available', 'final_stable_for_s')
LEG_FIELDS = ('front_distance_m', 'clearance_m', 'within_top_xy', 'top_geometry',
              'air', 'ground_contact', 'obstacle_pair_active', 'top_contact',
              'top_surface_contact', 'contact_surface', 'contact_reaction',
              'contact_reaction_force_n', 'bearing_force_n', 'bearing_verified',
              'support', 'load_fraction', 'load_fraction_valid', 'active_attempt',
              'crossing_evidence_status', 'consecutive_air_samples', 'consecutive_top_samples')
METRIC_KEYS = ('roll_rms_rad', 'pitch_rms_rad', 'roll_peak_rad', 'pitch_peak_rad',
               'roll_rate_rms_rad_s', 'pitch_rate_rms_rad_s',
               'angular_acceleration_rms_rad_s2', 'angular_acceleration_peak_rad_s2',
               'applied_servo_rate_rms_deg_s', 'applied_wheel_rate_rms_rad_s2',
               'applied_servo_acceleration_rms_deg_s2', 'applied_wheel_acceleration_rms_rad_s3',
               'residual_servo_rms_deg', 'residual_wheel_rms_rad_s')
MOTION_KEYS = ('body_linear_speed_m_s', 'com_speed_m_s',
               'com_linear_momentum_magnitude_kg_m_s', 'wheel_contact_count',
               *(f'{leg}_{key}' for leg in LEGS for key in (
                   'touchdown_event_count', 'touchdown_descent_speed_m_s',
                   'post_touchdown_rebound_upward_speed_m_s',
                   'contact_rolling_slip_estimate_abs_m_s')))


def need(condition, message):
    if not condition:
        raise ValueError(message)


def whole(value, name, minimum=0):
    need(type(value) is int and value >= minimum, f'invalid integer {name}')
    return value


def close(a, b):
    return (type(a) in (int, float) and type(b) in (int, float)
            and math.isfinite(a) and math.isfinite(b)
            and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-8))


def subset(value, keys):
    return {key: value.get(key) for key in keys} if isinstance(value, dict) else None


def unavailable(reason):
    return {'availability': 'UNAVAILABLE', 'value': None, 'reason': reason}


def datum(value, reason='field not recorded'):
    return unavailable(reason) if value is None else {'availability': 'AVAILABLE', 'value': value}


def decode(data):
    def bad(value):
        raise ValueError(f'nonstandard JSON constant {value}')
    value = json.loads(data.decode('utf-8-sig'), parse_constant=bad)
    need(isinstance(value, dict), 'expected a JSON object')
    return value


def digest_json(value):
    # Only a small manifest object, never checkpoint bytes or a trajectory.
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class Reads:
    def __init__(self):
        self.records = {}

    def note(self, path, scope, actual_bytes):
        state = path.stat()
        signature = (state.st_size, state.st_mtime_ns)
        entry = self.records.setdefault(path, {'signature': signature, 'reads': []})
        need(entry['signature'] == signature, f'file changed during summary: {path}')
        entry['reads'].append({'scope': scope, 'bytes_read': actual_bytes})

    def small(self, path):
        with path.open('rb') as stream:
            data = stream.read(32 * 1024 * 1024 + 1)
        need(len(data) <= 32 * 1024 * 1024, f'JSON exceeds bounded size: {path.name}')
        self.note(path, 'complete finalized JSON; no tensor load', len(data))
        return decode(data)

    def edge(self, path, *, last=False):
        maximum = 32 * 1024 * 1024
        read_bytes = 0
        with path.open('rb') as stream:
            if not last:
                data = stream.readline(maximum + 1)
                read_bytes = len(data)
            else:
                size = stream.seek(0, 2)  # Actual file seek, not stale directory Length.
                amount = min(65536, size)
                need(amount > 0, f'empty edge evidence: {path.name}')
                while True:
                    stream.seek(size - amount)
                    chunk = stream.read(amount)
                    read_bytes += len(chunk)
                    trimmed = chunk.rstrip(b'\r\n')
                    if b'\n' in trimmed or amount == size:
                        data = trimmed.rsplit(b'\n', 1)[-1]
                        break
                    need(amount < maximum, f'last row exceeds bounded size: {path.name}')
                    amount = min(amount * 2, maximum, size)
        need(data.strip() and len(data) <= maximum, f'missing/oversized edge: {path.name}')
        self.note(path, 'last JSONL row only' if last else 'first JSONL row only', read_bytes)
        return decode(data)

    def receipt(self):
        result = []
        for path, value in self.records.items():
            state = path.stat()
            need((state.st_size, state.st_mtime_ns) == value['signature'], f'file changed: {path}')
            result.append({'path': str(path), 'file_size_bytes': value['signature'][0],
                           'reads': value['reads']})
        return result


def quality_row(row, *, details=False):
    if not isinstance(row, dict) or row.get('sampled') is not True or not row.get('physics_ticks'):
        return unavailable('no sampled quality window; zero samples are not zero cost')
    result = {'availability': 'AVAILABLE', 'physics_ticks': row['physics_ticks'],
              'duration_s': row.get('duration_s'),
              'metrics': {key: datum(row.get(key)) for key in METRIC_KEYS}}
    if details:
        motion = row.get('motion_contact_diagnostics', {})
        result['motion_contact_diagnostics'] = {}
        for key in MOTION_KEYS:
            item = motion.get(key)
            if isinstance(item, dict) and item.get('available') is True:
                result['motion_contact_diagnostics'][key] = {
                    'availability': 'AVAILABLE', **subset(item, ('first', 'last', 'mean', 'rms',
                        'peak_abs', 'p95_abs', 'sum', 'integral', 'valid_ticks', 'valid_duration_s',
                        'missing_or_inapplicable_ticks'))}
            else:
                result['motion_contact_diagnostics'][key] = unavailable('not measured or no applicable events')
    return result


def classification(reason, physical):
    if physical.get('run_validity') == 'UNVERIFIED' or physical.get('termination_source') == 'UNVERIFIED_SENSOR':
        return 'UNVERIFIED_SENSOR'
    if reason == 'SUCCESS' or physical.get('success') is True:
        return 'TASK_SUCCESS'
    if reason in ('BODY_COLLISION', 'WHEEL_ONLY_CLIMB', 'TASK_FAILURE_BODY_COLLISION', 'TASK_FAILURE_WHEEL_ONLY_CLIMB'):
        return 'PHYSICAL_TASK_FAILURE'
    if reason in ('FALL', 'NAN_INF', 'HARD_JOINT_LIMIT', 'PHYSICS_EXPLOSION', 'SAFETY_ABORT'):
        return 'SAFETY_ABORT'
    if reason in ('INCOMPLETE_CONTROLLER_BLOCKED', 'INCOMPLETE_PHYSICAL_TASK', 'TASK_TIMEOUT', 'TIMEOUT'):
        return 'TASK_INCOMPLETE'
    if reason == 'INFRASTRUCTURE_ERROR':
        return 'INFRASTRUCTURE_ERROR'
    return 'UNCLASSIFIED_RECORDED_TERMINAL'


def summarize(run):
    run = Path(run).resolve(strict=True)
    need(run.parent == INPUT_ROOT.resolve(), 'only new-namespace validation run directories are accepted')
    reads = Reads()
    final = reads.small(run / 'run_manifest.json')
    actual = reads.small(run / 'evaluation_manifest.json')
    args = final.get('arguments', {})
    need(final.get('command') == 'eval' and final.get('lifecycle') == 'SUCCEEDED'
         and final.get('completed_at_utc') and final.get('result') == actual,
         'requires finalized successful execution with exactly bound evaluation result')
    need(actual.get('schema') == 'wlr50_clean.semantic_evaluation.v1', 'unknown evaluation schema')
    need(args.get('mode') == actual.get('mode') == 'semantic_residual_eval'
         and args.get('from_phase') == actual.get('from_phase') == 'P01'
         and type(args.get('num_envs')) is int and args['num_envs'] == 1,
         'requires natural P01 N1 residual-policy evaluation')
    need(args.get('teacher_offset_decisions') == 0 and args.get('new_mdp_warm_start') is False
         and args.get('prefix_source', 'frozen_fsm') != 'checkpoint_policy'
         and not (run / 'prefix_evidence.jsonl').exists(), 'prefix/warm-start is not a natural full evaluation')
    need(actual.get('deterministic_policy') is True
         and type(actual.get('optimizer_updates_during_evaluation')) is int
         and actual['optimizer_updates_during_evaluation'] == 0, 'must be deterministic and zero-optimizer')
    need(actual.get('window_ended_before_task_terminal') is False,
         'external short window cannot be promoted to a completed full evaluation')
    contract = actual.get('runtime_contract', {})
    need(contract == final.get('runtime_contract') and contract.get('experiment_id') == 'all_stage_acceptance_v1'
         and contract.get('semantic_version') == 'v3' and args.get('experiment_id') == 'all_stage_acceptance_v1',
         'runtime/experiment binding differs')
    need(actual.get('seed') == args.get('seed'), 'evaluation seed differs')
    checkpoint_value = actual.get('checkpoint')
    need(isinstance(checkpoint_value, str) and checkpoint_value and checkpoint_value == args.get('checkpoint'),
         'real checkpoint binding is absent')
    checkpoint = Path(checkpoint_value).resolve(strict=True)
    need(checkpoint.is_file() and checkpoint.suffix == '.pt'
         and OUTPUT_ROOT.resolve() / 'checkpoints' in checkpoint.parents, 'checkpoint outside new experiment output')
    cp = reads.small(checkpoint.with_name(checkpoint.stem + '_manifest.json'))
    need(cp.get('schema') == 'wlr50_clean.semantic_checkpoint.v1'
         and Path(cp.get('checkpoint_path', '')).resolve() == checkpoint
         and re.fullmatch('[0-9a-f]{64}', str(cp.get('checkpoint_sha256', '')))
         and cp['checkpoint_sha256'] == actual.get('checkpoint_sha256')
         and cp.get('save_load_round_trip') is True, 'checkpoint sidecar/recorded-SHA/roundtrip binding differs')
    migration = actual.get('checkpoint_resume_migration')
    if migration is None:
        need(cp.get('runtime_contract') == contract, 'checkpoint runtime is not the evaluation runtime')
    else:
        need(migration.get('source_checkpoint_sha256') == cp['checkpoint_sha256']
             and migration.get('source_contract_sha256') == digest_json(cp.get('runtime_contract'))
             and migration.get('target_contract_sha256') == digest_json(contract), 'recorded resume migration differs')
    need(cp.get('policy_contract') == actual.get('policy_contract') and isinstance(actual.get('policy_contract'), dict),
         'checkpoint/evaluation policy contract differs')
    decisions = whole(actual.get('policy_decisions'), 'decisions', 1)
    ticks = whole(actual.get('observed_physics_ticks'), 'physics ticks', 1)
    telemetry = actual.get('telemetry', {})
    need(telemetry.get('episodes') == 1 and telemetry.get('decisions') == decisions
         and telemetry.get('physics_ticks') == ticks, 'single-episode telemetry differs')
    need(not any(v not in (None, False, 0, [], {}) for k, v in telemetry.items() if k.startswith('prefix')),
         'nonempty prefix telemetry')
    counts = telemetry.get('phase_decisions')
    need(isinstance(counts, dict) and set(counts) <= set(PHASES)
         and sum(whole(v, 'phase count') for v in counts.values()) == decisions, 'phase ledger differs')
    first = reads.edge(run / 'residual_and_projection_audit.jsonl')
    last = reads.edge(run / 'residual_and_projection_audit.jsonl', last=True)
    initial = reads.edge(run / 'physical_observations.jsonl')
    raw = reads.edge(run / 'physical_observations.jsonl', last=True)
    need(first.get('decision_count') == 1 and first.get('phase_id') == 'P01'
         and first.get('physics_tick') == first.get('physics_ticks') and initial.get('physics_tick') == 0
         and close(initial.get('simulation_time_s'), 0.),
         'not a natural tick-zero P01 start')
    need(last.get('decision_count') == decisions and last.get('physics_tick') == raw.get('physics_tick') == ticks
         and close(last.get('sim_time_s'), ticks / 120.)
         and close(raw.get('simulation_time_s'), ticks / 120.) and close(actual.get('duration_s'), ticks / 120.),
         'final manifest/audit/raw tick-time identity differs')
    need(last.get('terminal_bootstrap_allowed') is False and last.get('time_outs') is False,
         'external truncation or bootstrap is not a full internal terminal')
    task = last.get('semantic_task', {})
    physical = actual.get('physical_task_evaluation', {})
    need(physical.get('evaluator_version') == 'all_stage_v1' and physical.get('physics_tick') == ticks,
         'final common all-stage evaluator snapshot absent')
    need(type(actual.get('task_success')) is bool and physical.get('success') is actual['task_success'],
         'physical task success binding differs')
    goals = task.get('completion_values')
    completed = task.get('completed_stage_ids')
    need(isinstance(goals, dict) and isinstance(completed, list)
         and completed == list(PHASES[:len(completed)]), 'completion evidence missing or nonsequential')
    need(all(type(v) in (int, float) and math.isfinite(v) for v in goals.values()),
         'completion values are not finite measured predicate values')
    unmet = {k: v for k, v in goals.items() if type(v) in (int, float) and v < 1.}
    quality = actual.get('quality_metrics', {})
    phase_quality = quality.get('phases', {})
    split = quality.get('transfer_capture_split', {})
    native = last.get('actuator_target_effect_audit_summary')
    if native is not None:
        need(native.get('physics_ticks') == last.get('physics_ticks'), 'terminal native interval differs')
    first_unfinished = next((p for p in PHASES if p not in completed), None)
    result = {
        'run': str(run), 'completed_at_utc': final['completed_at_utc'], 'execution_lifecycle': final['lifecycle'],
        'mode': actual['mode'], 'from_phase': 'P01', 'num_envs': 1, 'prefix_used': False,
        'deterministic_policy': True, 'seed': actual['seed'], 'optimizer_updates': 0,
        'policy_decisions': decisions, 'physics_ticks': ticks, 'duration_s': actual['duration_s'],
        'window_ended_before_task_terminal': False, 'task_success': actual['task_success'],
        'controller_task_success': actual.get('controller_task_success'),
        'termination_reason': actual.get('termination_reason'),
        'termination_source': datum(task.get('termination_source', physical.get('termination_source'))),
        'classification': classification(actual.get('termination_reason') or physical.get('termination_reason'), physical),
        'common_physical': {key: datum(physical.get(key)) for key in COMMON},
        'first_unfinished': {'ordered_stage_label': first_unfinished, 'current_stage': task.get('stage_id'),
            'completion_values': goals, 'first_unmet_completion': next(iter(unmet), None),
            'unmet_completion_values': unmet, 'entry_reasons': task.get('entry_reasons'),
            'stage_age_s': task.get('stage_age_s'), 'local_timeout': datum(task.get('local_timeout')),
            'pending_capture': datum(task.get('pending_capture'))},
        'phase_counts': {p: counts.get(p, 0) for p in PHASES},
        'phase_count_scope': 'final telemetry; no independent full decision stream scan',
        'completed_stages': completed, 'history_qcp': subset(physical.get('history'),
            ('active_lift', 'front_edge_crossed', 'placed', 'event_ticks', 'active_lift_semantics')),
        'current_legs': {leg: subset(physical.get('current_legs', {}).get(leg), LEG_FIELDS) for leg in LEGS},
        'terminal_physical': {'all_finite': raw.get('all_finite'), 'body_collision': raw.get('body_collision'),
            'body': subset(raw.get('base'), ('position_w_m', 'orientation_wxyz', 'linear_velocity_w_m_s', 'angular_velocity_w_rad_s')),
            'support': raw.get('support'), 'actual_full12': raw.get('actual_full12'),
            'geometry_pose_aware': datum(raw.get('geometry_pose_aware'))},
        'native_integrity': {'whole_episode': unavailable('no aggregate full-native verification in eval manifest; no scan performed'),
            'terminal_interval_ticks': last.get('physics_ticks'), 'terminal_interval_summary': datum(native),
            'terminal_four_write_fields': {key: datum(last.get(key)) for key in WRITES},
            'terminal_no_state_writes_verified': datum(last.get('no_in_episode_state_writes_verified')),
            'whole_episode_four_write_counts': unavailable('terminal fields are not asserted to be whole-episode totals')},
        'quality': {'scope': quality.get('comparison_window'), 'schema': quality.get('schema'),
            'score_config_recorded': datum(quality.get('score_config')), 'fixed_quality_score': datum(quality.get('fixed_quality_score')),
            'global': quality_row(quality.get('global'), details=True),
            'phases': {p: quality_row(phase_quality.get(p)) for p in PHASES},
            'transfer_capture_windows': {p: {part: quality_row(split.get(p, {}).get(part))
                for part in ('TRANSFER', 'EXECUTION', 'CAPTURE')} for p in PHASES},
            'units': 'original metric-key suffixes; event sum=count, force integral=N*s; no unit rescaling',
            'contact_limits': 'slip/rebound are saved diagnostic proxies, not ground-truth slip or a universal penalty'},
        'source_checkpoint': {'path': str(checkpoint), 'recorded_sha256': cp['checkpoint_sha256'],
            'save_load_round_trip_recorded': True, **subset(cp, ('global_policy_decisions', 'ppo_updates', 'optimizer_steps')),
            'policy_contract': actual['policy_contract'], 'runtime_commit': cp['runtime_contract'].get('source_git_commit'),
            'integrity_scope': 'file existence and immutable-sidecar/evaluation recorded binding only; no PT load or independent file rehash'},
        'runtime': {'commit': contract.get('source_git_commit'), 'runtime_content_sha256': contract.get('runtime_content_sha256'),
            'experiment_id': contract['experiment_id'], 'selected_configuration': contract.get('selected_configuration'),
            'physics_hz': contract.get('physics_hz'), 'decision_hz': contract.get('decision_hz'),
            'checkpoint_resume_migration': migration},
        'initialization_parity': {'status': 'NOT_VERIFIED_PAIRED', 'natural_P01_request_verified': True,
            'seed': actual['seed'], 'initial_physics_tick': initial['physics_tick'],
            'initial_body': subset(initial.get('base'), ('position_w_m', 'orientation_wxyz', 'linear_velocity_w_m_s', 'angular_velocity_w_rad_s')),
            'initial_projected_gravity_b': (initial.get('imu') or {}).get('projected_gravity_b'),
            'initial_center_of_mass': initial.get('center_of_mass'),
            'settle_ticks': unavailable('not recorded in this evaluation schema'),
            'extra_pre_action_ticks': unavailable('not recorded in this evaluation schema'),
            'level_reference_source': unavailable('not recorded in this evaluation schema'),
            'matched_A_or_B': False, 'superiority_claim': False},
    }
    return {'schema': 'ppo_all_stage_acceptance_v1.completed_full_evaluation_summary.v1',
            'generated_at_utc': datetime.now(timezone.utc).isoformat(), 'latest_full_evaluation': result,
            'sources': reads.receipt(), 'limits': [
                'Completed execution does not mean task success. No outcome is reclassified by this report.',
                'Only manifest and edge consistency verified; no independent replay, full native audit or physical-cause proof.',
                'Historical Q/C/P does not establish current support. Missing metrics are UNAVAILABLE, not zeros.',
                'No baseline pairing or stability superiority established; raw metadata stat check is not cryptographic evidence.',
                'Caller selects this completed run; this script does not assert it is the chronologically newest run or edit master totals.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    need(OUTPUT_ROOT.resolve() in output.parents and not output.exists(), 'output must be a new exclusive directory inside this experiment')
    report = summarize(args.run_dir)
    value = report['latest_full_evaluation']
    text = '\n'.join([
        '# Completed natural-P01 evaluation', '', f"Run: `{value['run']}`", '',
        f"Execution: {value['execution_lifecycle']}; task success: {value['task_success']}.",
        f"{value['policy_decisions']} decisions / {value['physics_ticks']} physics ticks / {value['duration_s']:.9f} s; optimizer updates 0.",
        f"Result: {value['classification']} / {value['termination_reason']}.",
        f"First unfinished stage: {value['first_unfinished']['ordered_stage_label']}; first unmet completion: {value['first_unfinished']['first_unmet_completion']}.",
        'Completion values: `' + json.dumps(value['first_unfinished']['completion_values'], ensure_ascii=False) + '`.',
        'Phase counts P01–P13: `' + ', '.join(str(value['phase_counts'][p]) for p in PHASES) + '`.', '',
        'Global/per-stage/transfer/capture quality and current contact vs Q/C/P are in the JSON; absent windows are UNAVAILABLE.',
        'Only terminal native interval is inspected; whole-run native/no-write proof is UNAVAILABLE in this lightweight report.',
        'Initialization is natural P01 but not independently paired with A/B. No success, improvement or unique cause is inferred.', '',
        *report['limits'], ''])
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    output.mkdir(parents=True, exist_ok=False)
    paths = (output / 'full_evaluation_summary.json', output / 'full_evaluation_summary.md')
    for path, contents in zip(paths, (payload, text)):
        with path.open('x', encoding='utf-8') as stream:
            stream.write(contents)
    print(json.dumps({'outputs': [str(p) for p in paths], 'task_success': value['task_success'],
                      'policy_decisions': value['policy_decisions'], 'optimizer_updates': 0}))


if __name__ == '__main__':
    main()
