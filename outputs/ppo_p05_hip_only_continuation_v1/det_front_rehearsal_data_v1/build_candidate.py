"""CPU-only, outputs-only reconstruction; no fitting or checkpoint publication.

The source video did NOT persist the 389-vector. Inputs below are reconstructed
from explicit synchronized source fields, then independently checked against
the frozen source actor's recorded history/conditional mean/sigma. This is a
candidate data artifact, not authorization to use it for AUX or PPO.
"""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import sys

import numpy as np
import torch
from tensordict import TensorDict

from wlr50_clean.ppo.semantic_observation import (
    SemanticObservationBuilder, load_semantic_observation_schema, _quaternion,
    _multiply, _rpy, _quat_rotate_inverse,
)
from wlr50_clean.ppo.semantic_p05_capture_migration import _ObservationOnlyEnv
from wlr50_clean.ppo.semantic_p05_capture_profile import P05_CAPTURE_POLICY, P05_CAPTURE_OBSERVATION_LAYOUT
from wlr50_clean.ppo.semantic_p05_capture_actor import p05_capture_request_history
from wlr50_clean.ppo.semantic_receiving_wheel_sigma import receiving_wheel_effective_log_std
from wlr50_clean.ppo.semantic_history_actor import history_conditioned_head, HISTORY_RHO
from wlr50_clean.ppo.semantic_training import construct_semantic_runner, parameter_hash

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
SOURCE = ROOT / 'runs/ppo_p05_hip_only_continuation_v1/video_eval/validation/20260922T0450447593129Z_g0001c3138b0b_f412423658e844e386aef04f0f53bcdc/source'
CP = ROOT / 'outputs/ppo_p05_hip_only_continuation_v1/checkpoints/history/checkpoint_step_000201728.pt'
EXPECTED_CP_SHA = 'a4eb243ce07ad4a9ed1cf2f7f9a22e951181bdb6e0716361b8eaeb173acfd5b6'
# Fixed before replay. No widening after a failure or selective row omission.
NUMERIC_ATOL = 1e-6
OUTPUT_NAMES = ('candidate.npz', 'row_checks.jsonl', 'candidate_manifest.json', 'README.md')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def prefix_rows(name, count):
    """Only hash/read the used prefix, never imply entire source was rehashed."""
    rows, row_shas, digest = [], [], hashlib.sha256()
    with (SOURCE / name).open('rb') as stream:
        for index in range(count):
            raw = next(stream)
            digest.update(raw)
            rows.append(json.loads(raw))
            row_shas.append(hashlib.sha256(raw).hexdigest())
    return rows, row_shas, {'path': str(SOURCE / name), 'rows_read': count,
        'prefix_bytes_sha256': digest.hexdigest(), 'whole_file_rehashed': False}


def require(condition, description):
    if not bool(condition):
        raise ValueError(description)


def check_vector(actual, expected, description, *, exact=False):
    actual = torch.as_tensor(actual).detach().cpu()
    expected = torch.tensor(expected, dtype=actual.dtype)
    require(actual.shape == expected.shape, description + ': shape mismatch')
    error = float((actual.to(torch.float64) - expected.to(torch.float64)).abs().max())
    require(torch.equal(actual, expected) if exact else error <= NUMERIC_ATOL,
            description + f': error {error} exceeds ' + ('exact equality' if exact else str(NUMERIC_ATOL)))
    return error


def placement_binding(row, leg):
    task = row['step_info']['semantic_task']
    evaluator = task['physical_evaluator']
    current = evaluator['current_legs'][leg]
    require(evaluator['valid'] is True and evaluator['termination_reason'] is None,
            leg + ' placement physical evaluator invalid')
    require(task['placed_history'][leg] is True, leg + ' placement history absent')
    require(current['contact_surface'] == 'TOP' and current['top_surface_contact'] is True
            and current['top_contact'] is True and current['within_top_xy'] is True
            and current['within_lateral_span'] is True and current['bearing_verified'] is True
            and current['support'] is True and current['ground_contact'] is False,
            leg + ' placement lacks current legal TOP/support evidence')
    return {'decision': row['decision'], 'endpoint_tick': row['end_tick'],
        'placed_event_tick': task['history']['event_ticks']['placed'][leg],
        'current_leg': current, 'next_state': task['stage_id'],
        'only_local_front_continuation_not_full_task_success': True}


def reconstruct(decision, previous, physical, capture, native, schema):
    tick = decision['start_tick']
    require(previous['end_tick'] == tick, 'previous decision endpoint discontinuity')
    current, prior, following = capture[tick], capture[tick - 1], capture[tick + 1]
    task = previous['step_info']['semantic_task']
    require(decision['request_phase'] == task['stage_id'] == current['phase'] == 'P02', 'input phase not P02')
    require(task['physical_evaluator']['physics_tick'] == tick, 'task/physics timestamp mismatch')
    require(task['physical_evaluator']['valid'] is True
            and task['physical_evaluator']['termination_reason'] is None
            and task['termination_reason'] is None, 'input physical task invalid/terminal')
    require(task['placed_history']['FR'] is False, 'input is not before FR placed')
    require(set(task['transfer_roles']) == {'FL', 'FR', 'RL', 'RR'}, 'missing transfer role; no inferred filler allowed')
    # Existing old and new RR workspace predicate is inactive here. This does
    # not itself migrate or authorize this candidate for a newer checkpoint.
    require(task['active_lift_history']['RR'] is False
            and task['front_edge_crossed_history']['RR'] is False
            and task['placed_history']['RR'] is False, 'unexpected RR history in front-only candidate')
    following_native = native[tick + 1]['native_audit']
    evidence = following['dispatch']['tracking_reference_evidence']
    require(evidence == following_native['tracking_reference_evidence'], 'mapper evidence sources disagree')
    require(evidence['previous_ack_physics_tick'] == current['dispatch']['physics_tick'], 'mapper ACK not current tick dispatch')
    require(evidence['mapper_advances'] == evidence['articulation_writes'] == 0, 'mapper capture has side effects')
    mapper = {**evidence['mapper_pre_state'], 'feedback_tick': evidence['mapper_feedback_tick'],
        'final_drive_servo_deg': following_native['previous_final_drive_servo_deg']}
    require(mapper['final_drive_servo_deg'] == current['dispatch']['drive_target_full12'][:8], 'mapper previous final target mismatch')
    history = {
        'previous_raw_full12': previous['raw_policy_action_full12'],
        'previous_residual_full12': native[tick]['projected_residual_full12'],
        'previous_previous_residual_full12': native[tick - 1]['projected_residual_full12'],
        'previous_applied_full12': current['dispatch']['drive_target_full12'],
        'previous_previous_applied_full12': prior['dispatch']['drive_target_full12'],
        'previous_nominal_full12': prior['nominal_full12'],
    }
    require(history['previous_residual_full12'] == current['dispatch']['independent_policy_residual_requested_full12'], 'REQUEST residual sources disagree')
    frame = SimpleNamespace(state_id='P02', sim_time_s=current['sim_time_s'], nominal_action_full12=current['nominal_full12'],
        info={'semantic_task': task, 'raw_observation': physical[tick], 'mapper_state_summary': mapper,
            'mapped_nominal_full12': current['dispatch']['native_drive_target_full12'], 'capture_assist': current['capture_assist']})
    # Production builder runs at every physical tick; its only mutable state is
    # the preceding physical sample's time, body-frame omega and chassis RPY.
    builder = SemanticObservationBuilder(schema)
    previous_raw = physical[tick - 1]
    q = _quaternion(previous_raw['base']['orientation_wxyz'])
    builder.previous_time = prior['sim_time_s']
    builder.previous_rpy = _rpy(_quaternion(_multiply(q, schema.fixed_chassis_to_body_wxyz)))
    builder.previous_omega = _quat_rotate_inverse(q, previous_raw['base']['angular_velocity_w_rad_s'])
    x = torch.tensor([schema.encode(builder.build(frame, history).groups)], dtype=torch.float32)
    require(x.shape == (1, 389) and torch.isfinite(x).all(), 'invalid rebuilt389')
    require(x[0, :13].tolist() == [0., 1.] + [0.] * 11, 'rebuilt phase onehot differs')
    for actual_tick in range(tick + 1, decision['end_tick'] + 1):
        observed = native[actual_tick]['native_audit']
        assist = capture[actual_tick]['dispatch']['capture_assist_evidence']
        require(observed['verified'] is True, 'unverified actuator audit')
        require(observed['policy_request_phase'] == 'P02', 'actuator request phase differs')
        require(observed['raw_policy_action_full12'] == decision['raw_policy_action_full12'], 'raw sample is not execution raw')
        require(observed['phase_mask_full12'] == [1] * 12
                and capture[actual_tick]['policy_permission_mask_full12'] == [1] * 12, 'not all12 residual permission')
        require(observed['capture_assist_owned_channels_full12'] == [False] * 12
                and assist['owner_indices'] == [] and assist['assist_correction_full12'] == [0.] * 12,
                'unexpected assist ownership in front candidate')
    return x, evidence


def main():
    require(not torch.cuda.is_available(), 'CPU-only process required')
    torch.set_num_threads(1)
    for name in OUTPUT_NAMES:
        require(not (OUT / name).exists(), 'refusing to overwrite existing candidate artifact: ' + name)
    metadata = load_json(CP.with_name(CP.stem + '_manifest.json'))
    require(sha(CP) == metadata['checkpoint_sha256'] == EXPECTED_CP_SHA, 'source checkpoint binding mismatch')
    source_manifest = load_json(SOURCE / 'semantic_video_source_manifest.json')
    require(source_manifest['checkpoint_load_provenance']['parameter_hashes']['actor_parameter_sha256']
            == metadata['actor_parameter_sha256'], 'video source actor is not CP201728')
    identical_paths = (
        'src/wlr50_clean/ppo/semantic_observation.py', 'src/wlr50_clean/ppo/semantic_env.py',
        'src/wlr50_clean/ppo/semantic_p05_capture_actor.py', 'src/wlr50_clean/ppo/semantic_history_actor.py',
        'src/wlr50_clean/ppo/semantic_receiving_wheel_sigma.py',
        'configs/ppo_p05_hip_only_continuation_v1/observation_schema.json')
    for path in identical_paths:
        require(sha(ROOT / path) == metadata['runtime_contract']['files'][path], 'source-sensitive file changed: ' + path)
    schema = load_semantic_observation_schema(ROOT / identical_paths[-1])
    decisions, decision_shas, decision_receipt = prefix_rows('video_policy_decisions.jsonl', 409)
    require([r['decision'] for r in decisions] == list(range(1, 410)), 'source decisions not contiguous')
    rows = [r for r in decisions if r['request_phase'] == 'P02']
    require([r['decision'] for r in rows] == list(range(3, 257)), 'source P02 window differs from reviewed254 rows')
    fr, fl = placement_binding(decisions[258], 'FR'), placement_binding(decisions[407], 'FL')
    require(fr['placed_event_tick'] == 2072 and fl['placed_event_tick'] == 3257, 'reviewed front event binding changed')
    require(decisions[408]['request_phase'] == 'P06', 'FL did not hand off to real P06')
    receipts = [decision_receipt]
    maps = []
    for name, key, count in (('physical_observations.jsonl', 'physics_tick', 2049),
                             ('capture_assist_ticks.jsonl', 'episode_physics_tick', 2048),
                             ('native_tick_audit.jsonl', 'episode_physics_tick', 2048)):
        values, _, receipt = prefix_rows(name, count)
        mapping = {r[key]: r for r in values}
        require(len(mapping) == count, 'duplicate physical tick in ' + name)
        maps.append(mapping)
        receipts.append(receipt)
    physical, capture, native = maps
    runner, _ = construct_semantic_runner(_ObservationOnlyEnv(389), seed=metadata['seed'], device='cpu',
        initialize_actor=False, policy_version=P05_CAPTURE_POLICY, observation_layout=P05_CAPTURE_OBSERVATION_LAYOUT)
    payload = torch.load(CP, map_location='cpu', weights_only=False)
    actor = runner.alg.actor
    actor.load_state_dict(payload['actor_state_dict'], strict=True)
    actor.eval()
    require(parameter_hash(actor) == metadata['actor_parameter_sha256'], 'CPU source actor hash mismatch')
    del payload
    rebuilt, raw_targets, means, sigmas, centers, checks = [], [], [], [], [], []
    for index, decision in enumerate(rows):
        try:
            request = decision['policy_request']
            require(request['mode'] == 'deterministic_conditional_mean'
                    and request['sampling_draws'] == request['extra_random_draws'] == 0, 'not actual deterministic raw')
            require(decision['raw_policy_action_full12'] == request['selected_raw_full12'] == request['conditional_mean_full12'], 'raw != actual deterministic mu')
            x, mapper = reconstruct(decision, decisions[decision['decision'] - 2], physical, capture, native, schema)
            obs = TensorDict({'policy': x, 'critic': x.clone()}, batch_size=[1])
            for part, key in ((x[0, 195:207], 'previous_raw_from_current_observation_full12'),
                              (x[0, 372:384], 'capture_assist_observed_features'),
                              (x[0, 384:389], 'capture_continuation_observed_features')):
                check_vector(part, request[key], key, exact=True)
            with torch.no_grad():
                raw_head = actor.mlp(x)
                center, hist = p05_capture_request_history(x)
                head = history_conditioned_head(raw_head, center, HISTORY_RHO)
                log_std, sigma_evidence = receiving_wheel_effective_log_std(head[:, 1, :], x[:, :372], .25)
                mean = actor(obs, stochastic_output=False)[0]
            history_error = check_vector(center[0], request['history_center_full12'], 'history center', exact=True)
            require(hist['gate_full12'][0].tolist() == request['cap_transition_gate_full12'], 'history cap gate differs')
            require(int(hist['stage_index'][0]) == request['stage_index'] == 1, 'history phase differs')
            errors = {
                'conditional_mean': check_vector(mean, request['conditional_mean_full12'], 'conditional mean'),
                'base_mean': check_vector(raw_head[0, 0], request['base_mean_full12'], 'base mean'),
                'history_center': history_error,
                'learned_sigma': check_vector(raw_head[0, 1].exp(), request['learned_sigma_full12'], 'learned sigma'),
                'effective_sigma': check_vector(log_std[0].exp(), request['effective_sigma_full12'], 'effective sigma'),
                'effective_log_std': check_vector(log_std[0], request['effective_log_std_full12'], 'effective log std'),
                'effective_innovation_sigma_multiplier': check_vector(sigma_evidence['effective_innovation_sigma_multiplier_full12'][0],
                    request['effective_innovation_sigma_multiplier_full12'], 'state sigma multiplier'),
            }
            require(sigma_evidence['receiving_sigma_gate_full12'][0].tolist() == request['receiving_sigma_gate_full12'], 'receiving sigma gate differs')
            raw = np.asarray(decision['raw_policy_action_full12'], dtype=np.float32)
            require(raw.tolist() == decision['raw_policy_action_full12'], 'float32 raw conversion changed actual labels')
            rebuilt.append(x[0].numpy().copy()); raw_targets.append(raw)
            means.append(np.asarray(request['conditional_mean_full12'], dtype=np.float32))
            sigmas.append(np.asarray(request['effective_sigma_full12'], dtype=np.float32))
            centers.append(np.asarray(request['history_center_full12'], dtype=np.float32))
            checks.append({'candidate_index': index, 'decision': decision['decision'], 'start_tick': decision['start_tick'],
                'end_tick': decision['end_tick'], 'source_decision_line_sha256': decision_shas[decision['decision'] - 1],
                'reconstructed_float32_observation_sha256': hashlib.sha256(rebuilt[-1].tobytes()).hexdigest(),
                'errors_max_abs': errors, 'history_and_assist_exact': True, 'all12_permission_no_assist_each_physics_tick': True,
                'mapper_previous_ack_physics_tick': mapper['previous_ack_physics_tick'],
                'mapper_feedback_tick': mapper['mapper_feedback_tick']})
        except Exception as exc:
            failure = {'status': 'STOPPED_AT_FIRST_UNRESOLVED_SAMPLE', 'candidate_index': index,
                'decision': decision['decision'], 'start_tick': decision['start_tick'], 'error': str(exc),
                'accepted_prefix_rows': len(checks), 'no_dataset_published': True, 'no_fit_or_checkpoint_update': True}
            (OUT / 'first_gap.json').write_text(json.dumps(failure, indent=2) + '\n', encoding='utf-8')
            print(json.dumps(failure, indent=2))
            raise
    require(parameter_hash(actor) == metadata['actor_parameter_sha256'], 'actor changed during read-only replay')
    n = len(checks)
    arrays = {'X389_reconstructed': np.stack(rebuilt), 'raw12_actual_deterministic': np.stack(raw_targets),
        'source_conditional_mean12': np.stack(means), 'source_effective_sigma12': np.stack(sigmas),
        'source_history_center12': np.stack(centers),
        'source_decision': np.asarray([r['decision'] for r in rows], dtype=np.int64),
        'input_tick': np.asarray([r['start_tick'] for r in rows], dtype=np.int64),
        'suggested_train_indices': np.arange(0, n, 3, dtype=np.int64),
        'suggested_validation_indices': np.arange(1, n, 3, dtype=np.int64),
        'source_only_indices': np.arange(2, n, 3, dtype=np.int64)}
    np.savez(OUT / 'candidate.npz', **arrays)
    with (OUT / 'row_checks.jsonl').open('w', encoding='utf-8') as stream:
        for row in checks:
            stream.write(json.dumps(row, sort_keys=True) + '\n')
    maxima = {key: max(row['errors_max_abs'][key] for row in checks) for key in checks[0]['errors_max_abs']}
    manifest = {'schema': 'wlr50_clean.det_front_reconstruction_candidate.v1',
        'status': 'CANDIDATE_ONLY_NOT_ADMITTED_TO_LEARNING', 'source_checkpoint': {'path': str(CP), 'sha256': EXPECTED_CP_SHA,
            'actor_sha256': metadata['actor_parameter_sha256'], 'manifest_sha256': sha(CP.with_name(CP.stem + '_manifest.json'))},
        'source_video_manifest': {'path': str(SOURCE / 'semantic_video_source_manifest.json'), 'sha256': sha(SOURCE / 'semantic_video_source_manifest.json')},
        'source_prefixes': receipts, 'source_sensitive_runtime_sha_identical': list(identical_paths),
        'helper': {'path': str(Path(__file__).resolve()), 'sha256': sha(Path(__file__))},
        'dataset': {'path': str(OUT / 'candidate.npz'), 'sha256': sha(OUT / 'candidate.npz'),
            'rows': n, 'phase': 'P02', 'source_decisions_inclusive': [3, 256], 'input_ticks_inclusive': [16, 2040],
            'last_executed_tick': 2048, 'dtype': 'float32', 'shapes': {key: list(value.shape) for key, value in arrays.items()}},
        'checks': {'path': str(OUT / 'row_checks.jsonl'), 'sha256': sha(OUT / 'row_checks.jsonl'),
            'numeric_atol_fixed_before_replay': NUMERIC_ATOL, 'maximum_absolute_errors': maxima,
            'all_history_center_exact': True, 'all_previous_raw_and_assist_features_exact': True,
            'all_raw_labels_exactly_recorded_mu_and_executed_raw': True,
            'all12_permission_and_no_assist_in_2032_actual_physics_ticks': True},
        'observation_provenance': {'direct389_tensor_was_saved_by_source': False,
            'reconstructed_from_explicit_synchronized_fields': True, 'missing_fields_guessed_or_filled': False,
            'original_CUDA_vs_CPU_is_not_bitwise_proof': True,
            'matching12_outputs_alone_cannot_prove389_input_identity': True,
            'mapper_current_state_source': 'next_physics_dispatch_pre_state_bound_to_current_ACK_and_previous_final_target',
            'finite_difference_source': 'immediately_previous_physics_sample_not_previous_policy_decision',
            'task_potential_source': 'original_logged_task_potential_not_independent_current_recomputation'},
        'local_front_evidence': {'FR_placement': fr, 'FL_placement': fl,
            'P06_continuation_decision': 409, 'FL_continuation_may_include_declared_capture_assist': True,
            'later_rear_outcome_not_analyzed_or_used_as_positive': True},
        'suggested_split_only': {'train': 'range(0,254,3)', 'train_count': 85,
            'validation': 'range(1,254,3)', 'validation_count': 85,
            'source_only': 'range(2,254,3)', 'source_only_count': 84,
            'temporal_correlation_warning': 'same trajectory interleaving, not independent-episode or closed-loop validation',
            'no_fit_performed': True},
        'learning': {'AUX_updates': 0, 'PPO_decisions_added': 0, 'PPO_updates_added': 0,
            'checkpoint_written': False, 'current_policy_compatibility_or_training_admission_claimed': False,
            'full_task_success_claimed': False}}
    (OUT / 'candidate_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    readback = np.load(OUT / 'candidate.npz', allow_pickle=False)
    require(all(np.array_equal(readback[key], value) for key, value in arrays.items()), 'candidate serialization mismatch')
    readback.close()
    text = f'''# CP201728 deterministic P02 candidate data (read-only preparation)

254 rows: decisions 3–256, pre-action ticks 16–2040 (0.133333–17.0 s), actions end at tick 2048. Only P02; P01/reset is excluded.

`candidate.npz` contains reconstructed float32 X389, unchanged recorded raw12 deterministic actions, recorded conditional μ/σ/history, source IDs and suggested split indices. `row_checks.jsonl` records every source-row binding and replay error. `candidate_manifest.json` binds source checkpoint, source-prefix bytes, helper and artifacts.

The source did **not** save the 389 input tensor. This is field-supported reconstruction, not a directly recorded tensor and not a bitwise proof: original execution was CUDA, replay is CPU. Matching 12 outputs alone cannot establish all 389 inputs; explicit field/time provenance is also recorded. No guessed histories, zero-filled missing fields, or reconstructed video pixels were used.

All 254 rows passed the fixed predeclared absolute tolerance {NUMERIC_ATOL:g}. Maximum μ error {maxima['conditional_mean']:.12g}; base μ {maxima['base_mean']:.12g}; history {maxima['history_center']:.12g}; learned σ {maxima['learned_sigma']:.12g}; effective σ {maxima['effective_sigma']:.12g}; effective logσ {maxima['effective_log_std']:.12g}. Previous raw, appended assist/continuation and history center match exactly. All 2032 action-interval physical ticks have all12 residual permission and zero assist ownership/correction.

Local evidence only: FR actual legal TOP/support and placed tick 2072 (decision 259 endpoint); FL placed tick 3257, legal TOP/support at decision 408 endpoint3264, followed by actual P06 decision409. FL continuation may include the declared assist; this is not a pure-policy FL or whole-task success claim. Later rear outcomes are not analyzed or used as positive labels.

Suggested fixed split, not fitted: train indices `range(0,254,3)` (85); validation `range(1,254,3)` (85); retain remaining `range(2,254,3)` (84) as source-only. Interleaving is temporally correlated and is not independent-episode validation or proof of closed-loop improvement.

The labels are the actual source deterministic conditional mean/raw actions, not independently recomputed means, tanh requests or final servo targets. Source checkpoint is CP201728, not a proposal to roll back the latest model. Source task-potential values are retained; current-policy compatibility/admission would require separate review.

Preparation added **0 AUX updates / 0 PPO decisions / 0 PPO updates**. No fitting, simulator, GPU, production edits, checkpoint writes, or edits to existing hash-bound v1 rehearsal helpers. CPU process exited after writing and independently re-reading these candidate arrays.
'''
    (OUT / 'README.md').write_text(text, encoding='utf-8')
    print(json.dumps({'status': manifest['status'], 'rows': n, 'errors': maxima,
        'candidate_sha256': manifest['dataset']['sha256'], 'manifest': str(OUT / 'candidate_manifest.json'),
        'no_fit_or_checkpoint_update': True}, indent=2))


if __name__ == '__main__':
    main()
