"""Independent read-only CPU audit of the single actually published event2.

Never imports the execution wrapper, optimizes, reloads CUDA state or writes a
checkpoint. Output is only this audit JSON/Markdown alongside the real receipt.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from wlr50_clean.ppo.semantic_training import state_hash

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
RECEIPT = HERE / 'CP211968_actual_detP02_AUX_execution.json'
EXPECTED_SOURCE = '5d0658331204c2bc79d71fd223582637e68dbb986ce4bf57596648f02c77d881'
EXPECTED_TARGET = '27bc7cbdd98775c4602057d7776dd3becc9eef9b42ec62f4691d2dae7bb06d55'
COUNTERS = ('global_policy_decisions', 'ppo_updates', 'optimizer_steps')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def parameter_sha(state):
    value = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        tensor = tensor.detach().cpu().contiguous()
        value.update(key.encode()); value.update(str(tensor.dtype).encode())
        value.update(str(tuple(tensor.shape)).encode()); value.update(tensor.numpy().tobytes())
    return value.hexdigest()


def main():
    assert not torch.cuda.is_available(), 'CPU-only process required'
    torch.set_num_threads(1)
    receipt = read(RECEIPT)
    binding = receipt['binding']; fit = receipt['fit_report']; kernel = fit['frozen_kernel_fit_report']
    source_path = Path(binding['source_checkpoint']['path'])
    target_path = Path(receipt['auxiliary_checkpoint']['path'])
    sm = read(source_path.with_name(source_path.stem + '_manifest.json'))
    tm = read(target_path.with_name(target_path.stem + '_manifest.json'))
    assert sha(source_path) == sm['checkpoint_sha256'] == binding['source_checkpoint']['sha256'] == EXPECTED_SOURCE
    assert sha(target_path) == tm['checkpoint_sha256'] == receipt['auxiliary_checkpoint']['sha256'] == EXPECTED_TARGET
    assert sha(source_path.with_name(source_path.stem + '_manifest.json')) == binding['source_checkpoint']['manifest_sha256']
    assert fit['schema'] == 'wlr50_clean.finite_reconstructed_deterministic_P02_aux.v2'
    assert receipt['schema'] == fit['schema'] + '.execution'
    assert fit['accepted_auxiliary_updates'] == fit['attempted_auxiliary_optimizer_steps'] == 32
    assert fit['optimized_parameters'] == ['actor.mlp.0.weight[:,1]'] and fit['optimized_scalar_count'] == 256
    assert fit['P01_initial_gradient_exact_zero'] is True and all(fit['protected_entire_Gaussian_bitwise_equal'].values())
    assert all(receipt[key] == fit[key] == 0 for key in ('PPO_decisions_added', 'PPO_updates_added', 'PPO_optimizer_steps_added'))
    assert {key: sm[key] for key in COUNTERS} == {key: tm[key] for key in COUNTERS} == dict(global_policy_decisions=211968, ppo_updates=1621, optimizer_steps=32420)
    source, target = (torch.load(path, map_location='cpu', weights_only=False) for path in (source_path, target_path))
    actual_changed = 0
    for key, prior in source['actor_state_dict'].items():
        after = target['actor_state_dict'][key]
        if key == 'mlp.0.weight':
            assert prior.shape == after.shape == (256, 389)
            assert torch.equal(prior[:, 0], after[:, 0]) and torch.equal(prior[:, 2:], after[:, 2:])
            actual_changed = int(torch.count_nonzero(prior[:, 1] != after[:, 1]))
            maximum_parameter_change = float((prior[:, 1] - after[:, 1]).abs().max())
        else:
            assert torch.equal(prior, after), 'non-P02 actor tensor changed: ' + key
    assert actual_changed == fit['actually_changed_scalar_count'] and 0 < actual_changed <= 256
    assert set(source['actor_state_dict']) == set(target['actor_state_dict'])
    assert set(source['critic_state_dict']) == set(target['critic_state_dict'])
    assert all(torch.equal(value, target['critic_state_dict'][key]) for key, value in source['critic_state_dict'].items())
    for payload, metadata in ((source, sm), (target, tm)):
        assert parameter_sha(payload['actor_state_dict']) == metadata['actor_parameter_sha256']
        assert parameter_sha(payload['critic_state_dict']) == metadata['critic_parameter_sha256']
        assert state_hash(payload['optimizer_state_dict']) == metadata['optimizer_state_sha256']
        assert all(payload['infos'][key] == value for key, value in metadata.items() if key in payload['infos'])
    assert state_hash(source['optimizer_state_dict']) == state_hash(target['optimizer_state_dict'])
    protected_metadata = ('training_rng_state', 'optimizer_learning_rate', 'runner_config', 'policy_contract',
        'normalization', 'normalizer_state_sha256', 'stage_requested_decisions', 'runtime_contract')
    assert all(sm[key] == tm[key] for key in protected_metadata)
    assert sm['runner_config']['device'] == tm['runner_config']['device'] == 'cuda:0'
    assert sm['optimizer_learning_rate'] == tm['optimizer_learning_rate'] == 1e-5
    lineage = [key for key in sm if key != 'resume_migration' and key.endswith(('_branch', '_branch_counts', '_migration'))]
    for key in lineage:
        if key == 'rr_postcross_workspace_branch':
            new = deepcopy(tm[key]); new['front_rehearsal_auxiliary'] = deepcopy(sm[key]['front_rehearsal_auxiliary'])
            assert new == sm[key], 'original RR branch lineage changed'
        else:
            assert tm[key] == sm[key], 'historical lineage changed: ' + key
    old_ledger = sm['rr_postcross_workspace_branch']['front_rehearsal_auxiliary']
    new_ledger = tm['rr_postcross_workspace_branch']['front_rehearsal_auxiliary']
    assert len(old_ledger['events']) == 1 and len(new_ledger['events']) == 2
    assert new_ledger['events'][0] == old_ledger['events'][0]
    assert new_ledger['schema'] == old_ledger['schema'] and new_ledger['training_lineage_label'] == old_ledger['training_lineage_label']
    event = new_ledger['events'][1]
    assert event['event_index'] == 2 and event['kind'] == 'finite_supervised_reconstructed_deterministic_P02_raw_actions_not_PPO'
    assert event['fit_report'] == fit and event['fit_report_sha256'] == digest(fit)
    assert event['binding'] == binding and event['source_checkpoint'] == binding['source_checkpoint']
    assert event['phase_scope'] == ['P02'] and event['optimized_parameters'] == ['actor.mlp.0.weight[:,1]']
    assert event['source_observations_directly_saved389'] is False and event['physical_success_claimed'] is False
    assert new_ledger['accepted_auxiliary_updates_total'] == new_ledger['attempted_auxiliary_optimizer_steps_total'] == 64
    old_aux = tm['task_conditioned_hip_wheel_branch']['auxiliary_mean_learning']
    assert old_aux == sm['task_conditioned_hip_wheel_branch']['auxiliary_mean_learning']
    assert old_aux['accepted_auxiliary_updates_total'] == 7 and old_aux['attempted_auxiliary_optimizer_steps_total'] == 8
    for name, expected in binding['helpers'].items():
        path = HERE / name if '/' not in name and (HERE / name).exists() else OUT / name
        assert sha(path) == expected, 'bound helper changed: ' + name
    for name in ('inspection', 'admission'):
        assert sha(binding[name]['path']) == binding[name]['sha256'], name + ' receipt changed'
    pointer = read(OUT / 'checkpoints/checkpoint_last_pointer.json')
    assert pointer['checkpoint_sha256'] == EXPECTED_SOURCE and Path(pointer['checkpoint']) == source_path
    assert receipt['latest_pointer_artifacts_unchanged'] is True and receipt['auxiliary_checkpoint']['latest_pointer_published'] is False
    assert receipt['auxiliary_checkpoint']['independent_official_reload_verified'] is True
    assert receipt['auxiliary_checkpoint']['source_device_preserved'] == 'cuda:0'
    # Bounds are cumulative to the original model, not32 separately reset limits.
    request_max = np.zeros(12); log_sigma_max = forward_max = reverse_max = 0.
    for step in kernel['steps']:
        assert step['accepted'] is True and step['learning_rate'] == 500.
        for split in ('train', 'validation'):
            delta = step[split + '_change_from_original']
            request_max = np.maximum(request_max, np.abs(delta['requested_residual_delta']).max(axis=0))
            log_sigma_max = max(log_sigma_max, float(np.abs(delta['log_sigma_delta']).max()))
            forward_max = max(forward_max, float(np.max(delta['kl_original_to_candidate'])))
            reverse_max = max(reverse_max, float(np.max(delta['kl_candidate_to_original'])))
    assert len(kernel['steps']) == 32 and max(forward_max, reverse_max) <= .5 and log_sigma_max <= .25
    assert np.all(request_max <= np.array([3.] * 8 + [.15] * 4))
    mse = {split: {'before': kernel['before'][split]['conditional_raw_mse'],
                  'after': kernel['after'][split]['conditional_raw_mse']} for split in ('train', 'validation')}
    report = {'schema': 'wlr50_clean.actual_det_P02_aux_event2_cpu_audit.v1', 'result': 'PASS',
        'receipt_path': str(RECEIPT), 'receipt_sha256': sha(RECEIPT), 'source_checkpoint': str(source_path),
        'source_sha256': EXPECTED_SOURCE, 'target_checkpoint': str(target_path), 'target_sha256': EXPECTED_TARGET,
        'actual_event': {'event_index': 2, 'accepted': 32, 'attempted': 32, 'changed_scalars': actual_changed,
            'only_parameter_scope': 'actor.mlp.0.weight[:,1]', 'maximum_parameter_change': maximum_parameter_change},
        'PPO_counters_unchanged': {key: tm[key] for key in COUNTERS}, 'PPO_added': 0,
        'actor_hash_before': sm['actor_parameter_sha256'], 'actor_hash_after': tm['actor_parameter_sha256'],
        'full_Adam_exact': True, 'critic_exact': True, 'full_CPU_CUDA_RNG_exact': True,
        'Identity_LR_runner_runtime_policy_exact': True, 'source_device': 'cuda:0',
        'event1_full_record_exact': True, 'old_AUX_7_8_full_record_exact': True,
        'new_front_AUX_total_accepted_attempted': [64, 64], 'historical_lineage_fields_exact': lineage,
        'actual_source_device_official_save_and_fresh_independent_reload_recorded': True,
        'latest_pointer_still_source': True, 'MSE': mse,
        'cumulative_max_requested_residual_delta_full12': request_max.tolist(),
        'cumulative_max_absolute_log_sigma_change': log_sigma_max,
        'cumulative_max_KL_original_to_candidate': forward_max, 'cumulative_max_KL_candidate_to_original': reverse_max,
        'reported_same_input_entire_Gaussian_invariance': fit['protected_entire_Gaussian_bitwise_equal'],
        'independent_algebraic_scope_proof': 'Every actor weight/buffer except P02 first-layer column is exactly equal. Any same input with P02 onehot=0 retains the entire Gaussian; future state trajectories are not thereby fixed.',
        'new_normal_PPO_carry_after_event2_not_yet_verified': True,
        'physical_success_claimed': False, 'historical_inputs_reconstructed_not_direct389': True,
        'this_audit_optimization_or_checkpoint_writes': 0}
    (HERE / 'actual_aux_v2_audit.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    md = f'''# Actual deterministic-P02 AUX event2 audit

**PASS. Actual saved artifact, not a synthetic fit.** CPU-only independent source/target tensor and ledger checks; no fitting or simulator use by this audit.

- Source CP211968 `{EXPECTED_SOURCE}` → unique AUX candidate `{EXPECTED_TARGET}`. Official source-device CUDA save and fresh independent official reload are recorded true; actual tensor hashes and embedded/sidecar values match. Latest pointer still names the pre-AUX source.
- Actual event2 **32 accepted /32 attempted**. Exactly **{actual_changed} scalars**, all in `actor.mlp.0.weight[:,1]`, changed. All388 non-P02 columns (including P01), other actor tensors and every critic tensor are identical.
- Full Adam state, LR1e-5, Identity, source CUDA:0 runner config, full saved CPU/CUDA RNG, runtime/policy and PPO counters are identical. Counters remain **211968 decisions /1621 PPO /32420 Adam**; AUX adds0 PPO. All historical lineage fields and all3 origins persist.
- Event1 is byte-equivalent as parsed canonical data, including its original source/data/helper/report bindings; previous AUX7/8 ledger is intact. Front ledger now has event1=32/32 + event2=32/32 = **64/64**. Historical credit is not relabelled.
- Actual train raw MSE **{mse['train']['before']:.12g} → {mse['train']['after']:.12g}**; validation **{mse['validation']['before']:.12g} → {mse['validation']['after']:.12g}**. These are fixed historical-state losses, not physical success.
- Maximum cumulative REQUEST shifts across all32 steps and both sets: FL knee {request_max[1]:.9g}°, FR hip {request_max[2]:.9g}°, FR knee {request_max[3]:.9g}°; largest wheel {max(request_max[8:]):.9g} rad/s. Maximum |Δlogσ| {log_sigma_max:.9g}; full Gaussian KL forward/reverse {forward_max:.9g}/{reverse_max:.9g}. Thus this is not mean-only learning.
- Reported real P01, real P03–P06 and synthetic P01/P03–P13 same-input entire-Gaussian checks pass. Independently, exact parameter support establishes no change for any same input with P02 one-hot0. It does **not** prove future-state or closed-loop trajectory invariance.

Inputs remain explicitly field-reconstructed historical deterministic P02 data, not directly saved389 or new on-policy samples. Their bounded admission and final source-specific inspection hashes match the real event. Current physical evaluation is separate. **Ordinary real PPO carry after event2 is not yet verified** (the earlier event1 carry was verified separately).

No frozen helper or production file was modified. Audit JSON gives full12 maxima and lineage keys; this CPU process exits after the report.
'''
    (HERE / 'actual_aux_v2_audit.md').write_text(md, encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('result', 'actual_event', 'PPO_counters_unchanged', 'MSE',
        'cumulative_max_requested_residual_delta_full12', 'cumulative_max_absolute_log_sigma_change',
        'cumulative_max_KL_original_to_candidate', 'cumulative_max_KL_candidate_to_original')}, indent=2))


if __name__ == '__main__':
    main()
