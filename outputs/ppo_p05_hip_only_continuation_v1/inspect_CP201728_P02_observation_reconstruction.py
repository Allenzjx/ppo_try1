"""Outputs-only: three deterministic source samples, no dataset or fit.

Stdout JSON. No simulator, optimizer step, checkpoint write or old-helper edit.
"""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import itertools
import json

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

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'runs/ppo_p05_hip_only_continuation_v1/video_eval/validation/20260922T0450447593129Z_g0001c3138b0b_f412423658e844e386aef04f0f53bcdc/source'
CP = ROOT / 'outputs/ppo_p05_hip_only_continuation_v1/checkpoints/history/checkpoint_step_000201728.pt'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def prefix(name, count):
    rows = []
    with (SOURCE / name).open('rb') as stream:
        for raw in itertools.islice(stream, count):
            rows.append(json.loads(raw))
    assert len(rows) == count
    return rows


def main():
    assert not torch.cuda.is_available(), 'CPU-only process required'
    torch.set_num_threads(1)
    metadata = read(CP.with_name(CP.stem + '_manifest.json'))
    assert sha(CP) == metadata['checkpoint_sha256'] == 'a4eb243ce07ad4a9ed1cf2f7f9a22e951181bdb6e0716361b8eaeb173acfd5b6'
    files = ('src/wlr50_clean/ppo/semantic_observation.py', 'src/wlr50_clean/ppo/semantic_env.py',
        'src/wlr50_clean/ppo/semantic_p05_capture_actor.py', 'src/wlr50_clean/ppo/semantic_history_actor.py',
        'src/wlr50_clean/ppo/semantic_receiving_wheel_sigma.py',
        'configs/ppo_p05_hip_only_continuation_v1/observation_schema.json')
    for path in files:
        assert sha(ROOT / path) == metadata['runtime_contract']['files'][path], 'source-sensitive runtime bytes changed: ' + path
    decisions = prefix('video_policy_decisions.jsonl', 5)
    physical = {r['physics_tick']: r for r in prefix('physical_observations.jsonl', 34)}
    capture = {r['episode_physics_tick']: r for r in prefix('capture_assist_ticks.jsonl', 33)}
    native = {r['episode_physics_tick']: r for r in prefix('native_tick_audit.jsonl', 33)}
    schema = load_semantic_observation_schema(ROOT / files[-1])
    runner, _ = construct_semantic_runner(_ObservationOnlyEnv(389), seed=metadata['seed'], device='cpu',
        initialize_actor=False, policy_version=P05_CAPTURE_POLICY, observation_layout=P05_CAPTURE_OBSERVATION_LAYOUT)
    payload = torch.load(CP, map_location='cpu', weights_only=False)
    runner.alg.actor.load_state_dict(payload['actor_state_dict'], strict=True)
    assert parameter_hash(runner.alg.actor) == metadata['actor_parameter_sha256']
    runner.alg.actor.eval()
    results = []
    for decision in decisions[2:5]:
        tick = decision['start_tick']; previous_decision = decisions[decision['decision'] - 2]
        assert tick in (16, 24, 32) and decision['request_phase'] == 'P02'
        assert previous_decision['end_tick'] == tick
        current, prior, following = capture[tick], capture[tick - 1], capture[tick + 1]
        request = decision['policy_request']
        task = previous_decision['step_info']['semantic_task']
        assert task['stage_id'] == current['phase'] == 'P02'
        following_native = native[tick + 1]['native_audit']
        evidence = following['dispatch']['tracking_reference_evidence']
        assert evidence == following_native['tracking_reference_evidence']
        assert evidence['previous_ack_physics_tick'] == current['dispatch']['physics_tick']
        assert evidence['mapper_advances'] == evidence['articulation_writes'] == 0
        mapper = {**evidence['mapper_pre_state'],
            'feedback_tick': evidence['mapper_feedback_tick'],
            'final_drive_servo_deg': following_native['previous_final_drive_servo_deg']}
        assert mapper['final_drive_servo_deg'] == current['dispatch']['drive_target_full12'][:8]
        history = {
            'previous_raw_full12': previous_decision['raw_policy_action_full12'],
            'previous_residual_full12': native[tick]['projected_residual_full12'],
            'previous_previous_residual_full12': native[tick - 1]['projected_residual_full12'],
            'previous_applied_full12': current['dispatch']['drive_target_full12'],
            'previous_previous_applied_full12': prior['dispatch']['drive_target_full12'],
            'previous_nominal_full12': prior['nominal_full12'],
        }
        assert history['previous_residual_full12'] == current['dispatch']['independent_policy_residual_requested_full12']
        frame = SimpleNamespace(state_id='P02', sim_time_s=current['sim_time_s'], nominal_action_full12=current['nominal_full12'],
            info={'semantic_task': task, 'raw_observation': physical[tick], 'mapper_state_summary': mapper,
                  'mapped_nominal_full12': current['dispatch']['native_drive_target_full12'], 'capture_assist': current['capture_assist']})
        # The production builder advances EVERY physics tick. Recover only its
        # three finite-difference variables from the immediately preceding raw
        # physical sample, not from the previous 15Hz policy decision.
        builder = SemanticObservationBuilder(schema)
        previous_raw = physical[tick - 1]
        q = _quaternion(previous_raw['base']['orientation_wxyz'])
        builder.previous_time = prior['sim_time_s']
        builder.previous_rpy = _rpy(_quaternion(_multiply(q, schema.fixed_chassis_to_body_wxyz)))
        builder.previous_omega = _quat_rotate_inverse(q, previous_raw['base']['angular_velocity_w_rad_s'])
        built = builder.build(frame, history)
        x = torch.tensor([schema.encode(built.groups)], dtype=torch.float32)
        obs = TensorDict({'policy': x, 'critic': x.clone()}, batch_size=[1])
        assert x.shape == (1, 389) and torch.isfinite(x).all()
        assert x[0, 195:207].tolist() == request['previous_raw_from_current_observation_full12']
        assert x[0, 372:384].tolist() == request['capture_assist_observed_features']
        assert x[0, 384:389].tolist() == request['capture_continuation_observed_features']
        with torch.no_grad():
            raw_head = runner.alg.actor.mlp(x)
            center, _ = p05_capture_request_history(x)
            head = history_conditioned_head(raw_head, center, HISTORY_RHO)
            log_std, _ = receiving_wheel_effective_log_std(head[:, 1, :], x[:, :372], .25)
            mean = runner.alg.actor(obs, stochastic_output=False)[0]
        expected = torch.tensor(request['conditional_mean_full12'], dtype=torch.float32)
        base = torch.tensor(request['base_mean_full12'], dtype=torch.float32)
        sigma = torch.tensor(request['effective_sigma_full12'], dtype=torch.float32)
        assert decision['raw_policy_action_full12'] == request['conditional_mean_full12'] == request['selected_raw_full12']
        results.append({'decision': decision['decision'], 'start_tick': tick, 'time_s': tick / 120,
            'constructed_observation_dim': 389,
            'observation_float32_le_sha256': hashlib.sha256(x.numpy().tobytes()).hexdigest(),
            'raw_is_actually_executed_deterministic_mean': True,
            'mean_bitwise_equal_CPU_vs_source_CUDA': bool(torch.equal(mean, expected)),
            'mean_abs_error_full12': (mean - expected).abs().tolist(),
            'mean_abs_error_max': float((mean - expected).abs().max()),
            'base_mean_abs_error_max': float((raw_head[0, 0] - base).abs().max()),
            'effective_sigma_abs_error_max': float((log_std.exp()[0] - sigma).abs().max()),
            'encoded_history_and_assist_fields_exact': True,
            'mapper_previous_ack_tick': evidence['previous_ack_physics_tick'],
            'mapper_feedback_tick': mapper['feedback_tick'],
            'source_mapper_pre_state_is_next_dispatch_pre_state': True})
    print(json.dumps({'scope': '3 P02 source states only; no optimization or eligible dataset publication',
        'source_checkpoint_sha256': metadata['checkpoint_sha256'], 'source_directory': str(SOURCE),
        'source_sensitive_encoder_actor_history_schema_files_byte_identical': list(files),
        'no_guessed_or_zero_filled_fields': True, 'samples': results,
        'CPU_vs_CUDA_roundoff_is_not_bitwise_389_input_proof': True,
        'full_front_dataset_training_authorized_or_published': False}, indent=2))


if __name__ == '__main__':
    main()
