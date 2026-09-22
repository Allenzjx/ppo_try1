"""Actual event3 source/target audit; CPU only, no optimizer/checkpoint writes."""
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch
from wlr50_clean.ppo.semantic_training import state_hash

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
RECEIPT = HERE / 'CP213376_actual_mean_head_AUX_execution.json'
SOURCE_SHA = '8e5961e3e78260f12124447bff72a7459b7323d2da8f83cc467a59ddc9290e54'
TARGET_SHA = '945b05e2763396c0f83c582eb85d34d778e6fb8c74b041177c3bdea9f7077cd3'
COUNTERS = ('global_policy_decisions', 'ppo_updates', 'optimizer_steps')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def parameter_sha(state):
    result = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        result.update(key.encode()); result.update(str(value.dtype).encode())
        result.update(str(tuple(value.shape)).encode()); result.update(value.numpy().tobytes())
    return result.hexdigest()


def main():
    assert not torch.cuda.is_available()
    torch.set_num_threads(1)
    receipt = read(RECEIPT)
    fit, binding = receipt['fit_report'], receipt['binding']
    source_path = Path(binding['source_checkpoint']['path'])
    target_path = Path(receipt['auxiliary_checkpoint']['path'])
    sm = read(source_path.with_name(source_path.stem + '_manifest.json'))
    tm = read(target_path.with_name(target_path.stem + '_manifest.json'))
    assert sha(source_path) == sm['checkpoint_sha256'] == binding['source_checkpoint']['sha256'] == SOURCE_SHA
    assert sha(target_path) == tm['checkpoint_sha256'] == receipt['auxiliary_checkpoint']['sha256'] == TARGET_SHA
    assert sha(source_path.with_name(source_path.stem + '_manifest.json')) == binding['source_checkpoint']['manifest_sha256']
    assert fit['accepted_auxiliary_updates'] == fit['attempted_auxiliary_optimizer_steps'] == 32
    assert fit['optimized_parameters'] == ['actor.mlp.4.weight[:12,:]', 'actor.mlp.4.bias[:12]']
    assert fit['optimized_scalar_count'] == fit['actually_changed_scalar_count'] == 3084
    assert fit['same_input_sigma_exact'] and not fit['P03plus_mean_bitwise_invariance_claimed']
    assert all(receipt[key] == fit[key] == 0 for key in ('PPO_decisions_added', 'PPO_updates_added', 'PPO_optimizer_steps_added'))
    expected_counts = dict(global_policy_decisions=213376, ppo_updates=1632, optimizer_steps=32640)
    assert {key: sm[key] for key in COUNTERS} == {key: tm[key] for key in COUNTERS} == expected_counts
    source, target = (torch.load(path, map_location='cpu', weights_only=False) for path in (source_path, target_path))
    changed = {}
    maximum_parameter_delta = {}
    assert set(source['actor_state_dict']) == set(target['actor_state_dict'])
    for key, before in source['actor_state_dict'].items():
        after = target['actor_state_dict'][key]
        if key in ('mlp.4.weight', 'mlp.4.bias'):
            assert torch.equal(before[12:], after[12:]), 'sigma output rows changed'
            changed[key] = int(torch.count_nonzero(before[:12] != after[:12]))
            maximum_parameter_delta[key] = float((before[:12] - after[:12]).abs().max())
        else:
            assert torch.equal(before, after), 'non-mean-head actor tensor changed: ' + key
    assert sum(changed.values()) == 3084
    assert source['actor_state_dict']['mlp.4.weight'].shape == (24, 256)
    assert source['actor_state_dict']['mlp.4.bias'].shape == (24,)
    assert set(source['critic_state_dict']) == set(target['critic_state_dict'])
    assert all(torch.equal(value, target['critic_state_dict'][key]) for key, value in source['critic_state_dict'].items())
    for payload, metadata in ((source, sm), (target, tm)):
        assert parameter_sha(payload['actor_state_dict']) == metadata['actor_parameter_sha256']
        assert parameter_sha(payload['critic_state_dict']) == metadata['critic_parameter_sha256']
        assert state_hash(payload['optimizer_state_dict']) == metadata['optimizer_state_sha256']
        assert all(payload['infos'][key] == value for key, value in metadata.items() if key in payload['infos'])
    assert state_hash(source['optimizer_state_dict']) == state_hash(target['optimizer_state_dict']) == fit['PPO_Adam_preserved_sha256']
    protected = ('training_rng_state', 'optimizer_learning_rate', 'runner_config', 'policy_contract',
        'normalization', 'normalizer_state_sha256', 'stage_requested_decisions', 'runtime_contract')
    assert all(sm[key] == tm[key] for key in protected)
    assert sm['runner_config']['device'] == tm['runner_config']['device'] == 'cuda:0'
    assert sm['optimizer_learning_rate'] == tm['optimizer_learning_rate'] == 1e-5
    lineage = [key for key in sm if key != 'resume_migration' and key.endswith(('_branch', '_branch_counts', '_migration'))]
    for key in lineage:
        if key == 'rr_postcross_workspace_branch':
            patched = deepcopy(tm[key])
            patched['front_rehearsal_auxiliary'] = deepcopy(sm[key]['front_rehearsal_auxiliary'])
            assert patched == sm[key]
        else:
            assert tm[key] == sm[key]
    old_ledger = sm['rr_postcross_workspace_branch']['front_rehearsal_auxiliary']
    ledger = tm['rr_postcross_workspace_branch']['front_rehearsal_auxiliary']
    assert len(old_ledger['events']) == 2 and len(ledger['events']) == 3
    assert ledger['events'][:2] == old_ledger['events']
    event = ledger['events'][2]
    assert event['event_index'] == 3 and event['kind'] == 'finite_supervised_existing_mean_head_actual_front_raw_actions_not_PPO'
    assert event['fit_report'] == fit and event['fit_report_sha256'] == digest(fit)
    assert event['binding'] == binding and event['source_checkpoint'] == binding['source_checkpoint']
    assert event['same_input_sigma_fixed'] and event['mean_may_change_all_phases']
    assert ledger['accepted_auxiliary_updates_total'] == ledger['attempted_auxiliary_optimizer_steps_total'] == 96
    old_aux = tm['task_conditioned_hip_wheel_branch']['auxiliary_mean_learning']
    assert old_aux == sm['task_conditioned_hip_wheel_branch']['auxiliary_mean_learning']
    assert old_aux['accepted_auxiliary_updates_total'] == 7 and old_aux['attempted_auxiliary_optimizer_steps_total'] == 8
    for filename, expected in binding['helpers'].items():
        assert sha(HERE / filename) == expected
    for key in ('inspection', 'data_manifest', 'objective'):
        assert sha(binding[key]['path']) == binding[key]['sha256']
    assert sha(receipt['budget_receipt']['path']) == receipt['budget_receipt']['sha256']
    assert receipt['budget_receipt']['envelope']['binding'] == binding
    assert receipt['budget_receipt']['envelope']['budget'] == fit['budget']
    assert receipt['auxiliary_checkpoint']['independent_official_reload_verified']
    assert receipt['auxiliary_checkpoint']['source_device_preserved'] == 'cuda:0'
    assert not receipt['auxiliary_checkpoint']['latest_pointer_published']
    assert receipt['latest_pointer_artifacts_unchanged'] and tm['save_load_round_trip']
    pointer = read(OUT / 'checkpoints/checkpoint_last_pointer.json')
    assert pointer['checkpoint_sha256'] == SOURCE_SHA
    bounds = fit['budget']
    maxima = {}
    for name in ('train', 'validation', 'protection', 'p01_probe'):
        values = [step['trust']['change_from_original'][name] for step in fit['steps']]
        maximum_request = np.max([v['requested_residual_delta_max_abs_per_channel'] for v in values], axis=0)
        maximum_kl = max(max(v['kl_original_to_candidate_max_abs'], v['kl_candidate_to_original_max_abs']) for v in values)
        maximum_raw = max(v['raw_mean_delta_max_abs'] for v in values)
        assert all(v['log_sigma_delta_max_abs'] == 0 for v in values)
        assert np.all(maximum_request <= np.array(bounds['maximum_' + name + '_request_shift_full12']))
        assert maximum_kl <= bounds['maximum_protection_full_gaussian_kl' if name == 'protection' else 'maximum_per_state_full_gaussian_kl']
        maxima[name] = {'raw_mean_max_abs': maximum_raw, 'REQUEST_max_abs_per_channel': maximum_request.tolist(),
            'bidirectional_KL_max': maximum_kl, 'log_sigma_max_abs': 0.0}
    assert len(fit['steps']) == 32 and all(step['accepted'] and step['learning_rate'] == 4 for step in fit['steps'])
    # Independent no-gradient CPU evaluations on the same source-bound finite
    # data. This imports the frozen wrapper but invokes no fit/save entry point.
    import execute_mean_rehearsal as method
    data = method.dataset.load_reviewed_data(sm, sm['runtime_contract'])
    assert digest(data['receipt']) == binding['data_receipt_sha256']
    assert data['protection_observations'].shape == (19, 389)
    with redirect_stdout(io.StringIO()):
        runner = method.observation_runner(data['train_observations'][0], sm, device='cpu', inspect_copy=True)
    actor = runner.alg.actor
    actor.load_state_dict(source['actor_state_dict'], strict=True)
    obs, targets, groups = method.kernel.dataset(actor, data)
    with torch.no_grad():
        before_dist = {key: method.kernel.distribution(actor, value) for key, value in obs.items()}
        actor.load_state_dict(target['actor_state_dict'], strict=True)
        after_dist = {key: method.kernel.distribution(actor, value) for key, value in obs.items()}
    independent = {}
    for name in obs:
        prior, after = before_dist[name], after_dist[name]
        assert torch.equal(prior['sigma'], after['sigma']) and torch.equal(prior['log_sigma'], after['log_sigma'])
        delta = method.kernel.kernel.gaussian_change(prior, after)
        summary = {'rows': len(obs[name]['policy']), 'same_input_sigma_bitwise_equal_CPU': True,
            'raw_mean_max_abs': float(delta['raw_mean_delta'].abs().max()),
            'REQUEST_max_abs_per_channel': delta['requested_residual_delta'].abs().amax(0).tolist(),
            'bidirectional_KL_max': max(float(delta['kl_original_to_candidate'].max()), float(delta['kl_candidate_to_original'].max()))}
        reported = fit['final_change_from_original'][name]
        assert abs(summary['raw_mean_max_abs'] - reported['raw_mean_delta_max_abs']) < 1e-6
        assert np.allclose(summary['REQUEST_max_abs_per_channel'], reported['requested_residual_delta_max_abs_per_channel'], atol=2e-5, rtol=1e-5)
        assert abs(summary['bidirectional_KL_max'] - reported['kl_original_to_candidate_max_abs']) < 2e-5
        independent[name] = summary
    losses = {'P01_train': {side: fit[side]['train_by_phase']['P01']['raw_MSE'] for side in ('before', 'after')},
        'P02_train': {side: fit[side]['train_by_phase']['P02']['raw_MSE'] for side in ('before', 'after')},
        'P02_validation': {side: fit[side]['validation']['raw_MSE'] for side in ('before', 'after')},
        'total_explicit_objective': {side: fit[side]['phase_weighted_half_MSE']['total'] for side in ('before', 'after')},
        'protection_half_MSE': {side: fit[side]['phase_weighted_half_MSE']['protection'] for side in ('before', 'after')}}
    report = {'schema': 'wlr50_clean.actual_mean_head_aux_event3_cpu_audit.v1', 'result': 'PASS',
        'receipt': {'path': str(RECEIPT), 'sha256': sha(RECEIPT)}, 'source_sha256': SOURCE_SHA,
        'target_sha256': TARGET_SHA, 'source_checkpoint': str(source_path), 'target_checkpoint': str(target_path),
        'actual_AUX_accepted_attempted': [32, 32], 'actually_changed_scalars': changed,
        'maximum_parameter_delta': maximum_parameter_delta, 'only_existing_mean_head_changed': True,
        'same_input_sigma_exact_parameter_support_proof': True, 'CPU_same_input_distribution_checks': independent,
        'critic_full_Adam_Identity_LR_full_CPU_CUDA_RNG_exact': True, 'PPO_counters_unchanged': expected_counts,
        'PPO_credit_added': 0, 'old_event1_event2_full_records_exact': True, 'front_AUX_totals': [96, 96],
        'old_AUX_totals': [7, 8], 'all_original_lineage_fields_preserved': lineage,
        'actual_source_device_official_save_fresh_reload_recorded': True, 'latest_pointer_still_plain_source': True,
        'losses': losses, 'all32_cumulative_to_original_bounds': maxima, 'explicit_budget': bounds,
        'actual_protection_rows': 19, 'actual_protection_phases': fit['actual_protection_phases'],
        'missing_actual_protection_phases': fit['missing_actual_protection_phases'],
        'P01_independent_validation_rows': 0, 'P02_validation_temporally_correlated_single_episode': True,
        'P03plus_global_mean_invariance_claimed': False, 'physical_success_claimed': False,
        'ordinary_real_PPO_carry_after_event3_not_yet_verified': True, 'audit_fit_or_checkpoint_writes': 0}
    assert report['actual_protection_phases'] == list(range(3, 13)) and report['missing_actual_protection_phases'] == [13]
    (HERE / 'actual_mean_head_aux_audit.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    loss_rows = '\n'.join(f"| {name} | {value['before']:.12g} | {value['after']:.12g} |" for name, value in losses.items())
    p = maxima['protection']; r = p['REQUEST_max_abs_per_channel']
    md = f'''# Actual mean-head AUX event3 — independent CPU audit

**PASS.** Plain CP213376 `{SOURCE_SHA}` → unique AUX candidate `{TARGET_SHA}`. Actual source-device official save and independent fresh reload are recorded true; embedded/sidecar, tensor and complete optimizer hashes match. Latest pointer remains the plain source; this audit changed no pointer or checkpoint.

- Actual **32 accepted / 32 attempted**, LR4 finite independent SGD. Exactly **3084** existing mean-head scalars changed: 3072 weights and 12 biases. All trunk, log-sigma rows, other actor state and critic tensors are exactly unchanged.
- Same-input sigma/log-sigma are **bitwise equal** in independent CPU checks on all train/validation/19 protection/2 unlabeled P01 probes, and follow algebraically from the frozen trunk, sigma head and state-dependent sigma kernel. This is not a future-trajectory statement.
- Full PPO Adam/moments, LR1e-5, Identity, runner CUDA:0, full Python/NumPy/CPU/CUDA RNG and all three origins/migrations are identical. PPO remains **213376 / 1632 / 32640**, adds **0**. Old events1/2 remain exactly intact; front AUX now **96/96**, older ledger **7/8** unchanged.

| Fixed-state metric | Before | After |
| --- | ---: | ---: |
{loss_rows}

Across all 32 proposals relative to the original model, the 19 real P03–P12 protection states changed: max raw μ {p['raw_mean_max_abs']:.9g}, full Gaussian KL {p['bidirectional_KL_max']:.9g} (limit .5), largest joint REQUEST shift {max(r[:8]):.9g}° (limit1°), largest wheel REQUEST shift {max(r[8:]):.9g} rad/s (limit .03). FL knee {r[1]:.9g}°, FR knee {r[3]:.9g}°. σ shift is zero. Independent final CPU distributions agree with the actual CUDA receipt within declared numerical comparison tolerance.

This **does not preserve P03+ means globally**: the mean head affects all phases, and measured protection drift is nonzero. P13 has no real protection observation. Only one P01 positive exists and it has no independent validation; the 85 P02 validation rows are correlated with the 85 training rows from one historical trajectory. Positive observations are explicitly field-reconstructed historical389 inputs, not new on-policy data. These lower losses establish neither current physical success nor stability improvement.

Actual ordinary PPO carry after event3 is still unverified. CPU-only read audit; no optimizer step, GPU/Isaac, production edit, frozen-helper edit or checkpoint write. Helper exits after this report.
'''
    (HERE / 'actual_mean_head_aux_audit.md').write_text(md, encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('result', 'actually_changed_scalars', 'PPO_counters_unchanged',
        'front_AUX_totals', 'losses', 'all32_cumulative_to_original_bounds', 'CPU_same_input_distribution_checks')}, indent=2))


if __name__ == '__main__':
    main()
