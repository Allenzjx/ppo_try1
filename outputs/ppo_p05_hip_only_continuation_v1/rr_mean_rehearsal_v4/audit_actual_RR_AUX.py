"""Independent CPU audit of actual event4; no fit/save/current-runtime CLI."""
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
RECEIPT = HERE / 'CP214400_actual_RR_mean_AUX_execution.json'
SOURCE_SHA = '8d0305626decff2214f8c3c0eee4031bdb791013d82a641a8ac4fc077ce235e9'
TARGET_SHA = '9987a4df3b4a2afaeddac794aa97b650e6133fe7ec89e8484cf59084cda69b6a'
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
    assert not torch.cuda.is_available(); torch.set_num_threads(1)
    r = read(RECEIPT); fit = r['fit_report']; binding = r['binding']
    sp, tp = Path(binding['source_checkpoint']['path']), Path(r['auxiliary_checkpoint']['path'])
    sm, tm = (read(p.with_name(p.stem + '_manifest.json')) for p in (sp, tp))
    assert sha(sp) == sm['checkpoint_sha256'] == binding['source_checkpoint']['sha256'] == SOURCE_SHA
    assert sha(tp) == tm['checkpoint_sha256'] == r['auxiliary_checkpoint']['sha256'] == TARGET_SHA
    assert sha(sp.with_name(sp.stem + '_manifest.json')) == binding['source_checkpoint']['manifest_sha256']
    source, target = (torch.load(p, map_location='cpu', weights_only=False) for p in (sp, tp))
    assert fit['accepted_auxiliary_updates'] == 7 and fit['attempted_auxiliary_optimizer_steps'] == 8
    assert len(fit['steps']) == 8 and [s['attempt'] for s in fit['steps']] == list(range(1, 9))
    assert [s['accepted'] for s in fit['steps']] == [True] * 7 + [False]
    assert all(s['learning_rate'] == .25 for s in fit['steps'])
    rejected = fit['steps'][-1]
    assert rejected['rejection_reasons'] == ['protection_full_Gaussian_KL_bound']
    assert rejected['rejected_full3084_leaf_discarded']
    assert fit['stop_reason'] == 'first_candidate_rejected_then_stop_no_LR_search'
    # Actual saved final output is the seventh accepted proposal, NOT the eighth.
    assert fit['after'] == fit['steps'][6]['candidate'] and fit['after'] != rejected['candidate']
    assert fit['final_change_from_original'] == fit['steps'][6]['trust']['change_from_original']
    assert fit['optimized_parameters'] == ['actor.mlp.4.weight[:12,:]', 'actor.mlp.4.bias[:12]']
    assert fit['optimized_scalar_count'] == fit['actually_changed_scalar_count'] == 3084
    changed = {}
    assert set(source['actor_state_dict']) == set(target['actor_state_dict'])
    for key, before in source['actor_state_dict'].items():
        after = target['actor_state_dict'][key]
        if key in ('mlp.4.weight', 'mlp.4.bias'):
            assert torch.equal(before[12:], after[12:]), 'sigma output rows changed'
            changed[key] = int(torch.count_nonzero(before[:12] != after[:12]))
        else:
            assert torch.equal(before, after), 'non-mean-head actor changed: ' + key
    assert sum(changed.values()) == 3084
    assert source['actor_state_dict']['mlp.4.weight'].shape == (24, 256)
    assert source['actor_state_dict']['mlp.4.bias'].shape == (24,)
    assert all(torch.equal(value, target['critic_state_dict'][key]) for key, value in source['critic_state_dict'].items())
    for payload, metadata in ((source, sm), (target, tm)):
        assert parameter_sha(payload['actor_state_dict']) == metadata['actor_parameter_sha256']
        assert parameter_sha(payload['critic_state_dict']) == metadata['critic_parameter_sha256']
        assert state_hash(payload['optimizer_state_dict']) == metadata['optimizer_state_sha256']
        assert all(payload['infos'][k] == v for k, v in metadata.items() if k in payload['infos'])
    assert fit['actor_parameter_sha256_before'] == sm['actor_parameter_sha256']
    assert fit['actor_parameter_sha256_after'] == tm['actor_parameter_sha256']
    assert state_hash(source['optimizer_state_dict']) == state_hash(target['optimizer_state_dict']) == fit['PPO_Adam_preserved_sha256']
    protected = ('training_rng_state', 'optimizer_learning_rate', 'runner_config', 'policy_contract',
        'normalization', 'normalizer_state_sha256', 'stage_requested_decisions', 'runtime_contract')
    assert all(sm[k] == tm[k] for k in protected)
    assert sm['runtime_contract']['source_git_commit'] == '5fd88852bf20c94cd74405c791a13a9fd9e0a3d8'
    assert sm['runner_config']['device'] == tm['runner_config']['device'] == 'cuda:0'
    assert sm['optimizer_learning_rate'] == tm['optimizer_learning_rate'] == 1e-5
    counts = dict(zip(COUNTERS, (214400, 1640, 32800)))
    assert {k: sm[k] for k in COUNTERS} == {k: tm[k] for k in COUNTERS} == counts
    assert all(r[k] == fit[k] == 0 for k in ('PPO_decisions_added', 'PPO_updates_added', 'PPO_optimizer_steps_added'))
    lineage = [k for k in sm if k != 'resume_migration' and k.endswith(('_branch', '_branch_counts', '_migration'))]
    for key in lineage:
        if key == 'rr_postcross_workspace_branch':
            candidate = deepcopy(tm[key]); candidate['front_rehearsal_auxiliary'] = deepcopy(sm[key]['front_rehearsal_auxiliary'])
            assert candidate == sm[key]
        else:
            assert tm[key] == sm[key]
    old_ledger = sm['rr_postcross_workspace_branch']['front_rehearsal_auxiliary']
    ledger = tm['rr_postcross_workspace_branch']['front_rehearsal_auxiliary']
    assert len(old_ledger['events']) == 3 and len(ledger['events']) == 4 and ledger['events'][:3] == old_ledger['events']
    assert [old_ledger[k] for k in ('accepted_auxiliary_updates_total', 'attempted_auxiliary_optimizer_steps_total')] == [96, 96]
    assert [ledger[k] for k in ('accepted_auxiliary_updates_total', 'attempted_auxiliary_optimizer_steps_total')] == [103, 104]
    event = ledger['events'][3]
    assert event['event_index'] == 4 and event['kind'] == 'finite_supervised_existing_mean_head_actual_RR_capture_continuation_raw_actions_not_PPO'
    assert event['fit_report'] == fit and event['fit_report_sha256'] == digest(fit) and event['binding'] == binding
    old_aux = tm['task_conditioned_hip_wheel_branch']['auxiliary_mean_learning']
    assert old_aux == sm['task_conditioned_hip_wheel_branch']['auxiliary_mean_learning']
    assert [old_aux[k] for k in ('accepted_auxiliary_updates_total', 'attempted_auxiliary_optimizer_steps_total')] == [7, 8]
    for filename, expected in binding['helpers'].items():
        assert sha(HERE / filename) == expected
    for key in ('inspection', 'data_manifest', 'objective'):
        assert sha(binding[key]['path']) == binding[key]['sha256']
    assert sha(r['budget_receipt']['path']) == r['budget_receipt']['sha256']
    assert r['budget_receipt']['envelope']['binding'] == binding and r['budget_receipt']['envelope']['budget'] == fit['budget']
    assert r['auxiliary_checkpoint']['independent_official_reload_verified'] and tm['save_load_round_trip']
    assert r['auxiliary_checkpoint']['source_device_preserved'] == 'cuda:0'
    assert not r['auxiliary_checkpoint']['latest_pointer_published'] and r['latest_pointer_artifacts_unchanged']
    # Do not inspect/rebind the current live runtime: a subsequent migration may
    # already be in progress. Both immutable payloads above really are old5fd.
    import execute_rr_mean_rehearsal as method
    import data_v4 as dataset
    sm['checkpoint_path'] = str(sp)
    data = dataset.load_reviewed_data(sm, sm['runtime_contract'])
    assert digest(data['receipt']) == binding['data_receipt_sha256']
    assert data['train_observations'].shape == (342, 389) and data['validation_observations'].shape == (171, 389)
    with redirect_stdout(io.StringIO()):
        runner = method.observation_runner(data['train_observations'][0], sm, device='cpu', inspect_copy=True)
    assert type(runner.alg.actor.obs_normalizer) is type(runner.alg.critic.obs_normalizer) is torch.nn.Identity
    actor = runner.alg.actor; actor.load_state_dict(source['actor_state_dict'], strict=True)
    obs, targets, groups = method.kernel.dataset(actor, data)
    with torch.no_grad():
        before = {k: method.kernel.distribution(actor, x) for k, x in obs.items()}
        actor.load_state_dict(target['actor_state_dict'], strict=True)
        after = {k: method.kernel.distribution(actor, x) for k, x in obs.items()}
    independent = {}
    for name in obs:
        assert torch.equal(before[name]['sigma'], after[name]['sigma']) and torch.equal(before[name]['log_sigma'], after[name]['log_sigma'])
        change = method.kernel.kernel.gaussian_change(before[name], after[name])
        current = {'rows': len(obs[name]['policy']), 'sigma_logsigma_exact': True,
            'raw_mean_max_abs': float(change['raw_mean_delta'].abs().max()),
            'REQUEST_max_abs_per_channel': change['requested_residual_delta'].abs().amax(0).tolist(),
            'bidirectional_KL_max': max(float(change['kl_original_to_candidate'].max()), float(change['kl_candidate_to_original'].max()))}
        recorded = fit['final_change_from_original'][name]
        assert abs(current['raw_mean_max_abs'] - recorded['raw_mean_delta_max_abs']) < 1e-6
        assert np.allclose(current['REQUEST_max_abs_per_channel'], recorded['requested_residual_delta_max_abs_per_channel'], atol=2e-5, rtol=1e-5)
        assert abs(current['bidirectional_KL_max'] - recorded['kl_original_to_candidate_max_abs']) < 2e-5
        independent[name] = current
    losses = {}
    for section, key in (('train', 'train_phase_groups'), ('validation', 'validation_phase_groups')):
        for phase, ids in data[key].items():
            values = {label: float((dist[section]['mean'][ids] - targets[section][ids]).square().mean())
                for label, dist in (('before', before), ('after', after))}
            if section == 'train':
                for side in values:
                    assert abs(values[side] - fit[side]['train_by_phase'][phase]['raw_MSE']) <= 2e-5
            losses[section + '_' + phase] = values
    for name in ('train', 'validation', 'protection', 'p01_probe'):
        for step in fit['steps'][:7]:
            change = step['trust']['change_from_original'][name]
            assert change['log_sigma_delta_max_abs'] == 0
            assert np.all(np.asarray(change['requested_residual_delta_max_abs_per_channel']) <= fit['budget']['maximum_' + name + '_request_shift_full12'])
            bound = fit['budget']['maximum_protection_full_gaussian_kl' if name == 'protection' else 'maximum_per_state_full_gaussian_kl']
            assert max(change['kl_original_to_candidate_max_abs'], change['kl_candidate_to_original_max_abs']) <= bound
    rejected_kl = rejected['trust']['change_from_original']['protection']['kl_original_to_candidate_max_abs']
    assert rejected_kl > fit['budget']['maximum_protection_full_gaussian_kl']
    report = {'schema': 'wlr50_clean.actual_RR_mean_AUX_event4_cpu_audit.v1', 'result': 'PASS',
        'execution_receipt': {'path': str(RECEIPT), 'sha256': sha(RECEIPT)},
        'source_checkpoint': {'path': str(sp), 'sha256': SOURCE_SHA}, 'target_checkpoint': {'path': str(tp), 'sha256': TARGET_SHA},
        'actual_RR_AUX_accepted_attempted': [7, 8], 'fixed_auxiliary_LR': .25, 'no_retry': True,
        'rejection': {'attempt': 8, 'reason': rejected['rejection_reasons'], 'protection_KL': rejected_kl,
            'bound': fit['budget']['maximum_protection_full_gaussian_kl'], 'discarded_full3084_leaf': True,
            'saved_final_outputs_exactly_seventh_candidate_not_eighth': True},
        'actually_changed_scalars': changed, 'only_mean_head_changed': True, 'same_input_sigma_exact': True,
        'critic_Adam_Identity_LR_full_Python_NumPy_CPU_CUDA_RNG_exact': True, 'PPO_counters_unchanged': counts,
        'PPO_credit_added': 0, 'inherited_front_AUX': [96, 96], 'event4_RR_AUX': [7, 8],
        'mixed_ledger_total': [103, 104], 'older_limited_AUX_separate': [7, 8],
        'old_three_full_event_records_preserved': True, 'old_three_event_records_sha256': digest(old_ledger['events']),
        'all_original_lineage_fields_preserved': lineage, 'source_device_official_save_independent_fresh_reload_recorded': True,
        'latest_pointer_unchanged_by_AUX_recorded_not_rechecked_after_subsequent_migration': True,
        'raw_label_phase_MSE_CPU': losses, 'actual_CUDA_total_objective': {side: fit[side]['phase_weighted_half_MSE']['total'] for side in ('before', 'after')},
        'actual_CUDA_protection_half_MSE': {side: fit[side]['phase_weighted_half_MSE']['protection'] for side in ('before', 'after')},
        'CPU_final_distribution_change': independent,
        'protection_rows': 305, 'protection_phases': fit['actual_protection_phases'], 'P13_real_coverage': False,
        'P10_training_rows': 1, 'P10_validation_rows': 0, 'single_trajectory_correlated_validation': True,
        'current_X17_explicitly_derived_rows': 237, 'old_source_raw_labels_unchanged': True,
        'source_mean_not_substituted_for_raw': True, 'P03plus_global_mean_invariance_claimed': False,
        'physical_success_claimed': False, 'ordinary_PPO_carry_after_event4_not_yet_verified': True,
        'audit_optimizer_steps_or_checkpoint_writes': 0}
    (HERE / 'actual_RR_AUX_audit.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    lossrows = '\n'.join(f"| {name} | {v['before']:.10g} | {v['after']:.10g} |" for name, v in losses.items())
    protection = independent['protection']; request = protection['REQUEST_max_abs_per_channel']
    md = f'''# Actual RR AUX event4 — independent CPU audit

**PASS.** Plain CP214400 `{SOURCE_SHA}` → unique RR AUX `{TARGET_SHA}`. Both saved payloads use old5fd; this audit does not consult or rebind a subsequent live runtime migration.

- Actual **7 accepted / 8 attempted**, fixed SGD LR0.25, no retry. Attempt8 exceeded protection KL: **{rejected_kl:.9g} > 0.5**, so the full3084 proposal was discarded. Saved final outputs and change metrics exactly equal accepted attempt7, not rejected attempt8; independent CPU evaluation reproduces the saved receipt.
- Exactly **3072 mean weights +12 mean biases** changed. Every trunk/sigma-head/other actor tensor, critic, full PPO Adam/moments, Identity, PPO LR1e-5, CUDA:0 runner and Python/NumPy/CPU/CUDA RNG is unchanged. Same-input σ/logσ are exactly equal on all342 train/171 validation/305 protection/2 P01 probes.
- PPO stays **214400 / 1640 / 32800**, +0. Old full events1–3 stay exact (**front96/96**); new **RR7/8** gives mixed ledger **103/104**, while older separate AUX **7/8** remains unchanged. All original origins/migrations persist. Official CUDA save and independent fresh reload are recorded successful. No pointer or checkpoint was written by this audit.

| Actual-raw target MSE (independent CPU) | Before | After |
| --- | ---: | ---: |
{lossrows}

Saved protection max KL is **{protection['bidirectional_KL_max']:.9g}**; largest joint REQUEST drift **{max(request[:8]):.9g}°**, wheel drift **{max(request[8:]):.9g} rad/s**. σ is fixed, but means may change across all phases. These are finite fixed-state guards, not global P03+ mean invariance or future-trajectory safety proof.

Targets remain all12 actually executed stochastic raw values, not source μ, final transformed targets or nominal labels. Only current X17 was explicitly derived on237 historical rows. P10 has one train/no validation row; validation is correlated within one episode. P13 has no real protection coverage. Lower supervised loss establishes neither current RR placement nor full-task success. Ordinary real PPO carry after event4 is not yet verified.

CPU-only audit; zero fit, GPU/Isaac, production/frozen-helper edits or checkpoint writes. Helper exits after this report.
'''
    (HERE / 'actual_RR_AUX_audit.md').write_text(md, encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('result', 'actual_RR_AUX_accepted_attempted', 'actually_changed_scalars',
        'PPO_counters_unchanged', 'rejection', 'raw_label_phase_MSE_CPU', 'CPU_final_distribution_change')}, indent=2))


if __name__ == '__main__':
    main()
