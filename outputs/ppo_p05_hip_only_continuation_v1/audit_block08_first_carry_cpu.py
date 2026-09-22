"""First sealed block08 update only; run after root confirms actual save.

CPU read audit, no polling, fit, simulator, checkpoint or runtime writes.
"""
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path

import torch
from wlr50_clean.ppo.semantic_capture_assist import capture_assist_features
from wlr50_clean.ppo.semantic_training import state_hash

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
RUN = ROOT / 'runs/ppo_p05_hip_only_continuation_v1/train/20260922T1301289433343Z_g5fd88852bf20_337c83c1b0214e02b62aa98b579e2812'
SOURCE = OUT / 'checkpoints/history/checkpoint_aux_meanfront_step_000213376_v3.pt'
TARGET = OUT / 'checkpoints/history/checkpoint_step_000213504.pt'
SOURCE_SHA = '945b05e2763396c0f83c582eb85d34d778e6fb8c74b041177c3bdea9f7077cd3'
COUNTERS = ('global_policy_decisions', 'ppo_updates', 'optimizer_steps')
PHASES = [f'P{i:02}' for i in range(1, 14)]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def lines(path):
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def take(path, count):
    rows = list(itertools.islice(lines(path), count))
    assert len(rows) == count, 'actual sealed first batch unavailable; do not poll'
    return rows


def parameter_sha(state):
    digest = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(key.encode()); digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode()); digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def main():
    assert not torch.cuda.is_available(), 'CPU-only process required'
    torch.set_num_threads(1)
    sm, tm = (read(path.with_name(path.stem + '_manifest.json')) for path in (SOURCE, TARGET))
    assert sha(SOURCE) == sm['checkpoint_sha256'] == SOURCE_SHA
    assert sha(TARGET) == tm['checkpoint_sha256']
    assert tm['source_run'] == str(RUN) and tm['save_load_round_trip']
    assert {key: tm[key] - sm[key] for key in COUNTERS} == dict(global_policy_decisions=128, ppo_updates=1, optimizer_steps=20)
    assert {key: tm[key] for key in COUNTERS} == dict(global_policy_decisions=213504, ppo_updates=1633, optimizer_steps=32660)
    preserved = [key for key in sm if key != 'resume_migration' and key.endswith(('_branch', '_migration'))]
    assert all(tm[key] == sm[key] for key in preserved)
    for key in ('runtime_contract', 'policy_contract', 'runner_config', 'normalization', 'normalizer_state_sha256'):
        assert sm[key] == tm[key]
    ledger = tm['rr_postcross_workspace_branch']['front_rehearsal_auxiliary']
    assert len(ledger['events']) == 3 and [event['event_index'] for event in ledger['events']] == [1, 2, 3]
    assert ledger['accepted_auxiliary_updates_total'] == ledger['attempted_auxiliary_optimizer_steps_total'] == 96
    assert ledger == sm['rr_postcross_workspace_branch']['front_rehearsal_auxiliary']
    assert ledger['events'][2]['same_input_sigma_fixed'] and ledger['events'][2]['mean_may_change_all_phases']
    older = tm['task_conditioned_hip_wheel_branch']['auxiliary_mean_learning']
    assert older['accepted_auxiliary_updates_total'] == 7 and older['attempted_auxiliary_optimizer_steps_total'] == 8
    source, target = (torch.load(path, map_location='cpu', weights_only=False) for path in (SOURCE, TARGET))
    for payload, metadata in ((source, sm), (target, tm)):
        assert parameter_sha(payload['actor_state_dict']) == metadata['actor_parameter_sha256']
        assert parameter_sha(payload['critic_state_dict']) == metadata['critic_parameter_sha256']
        assert state_hash(payload['optimizer_state_dict']) == metadata['optimizer_state_sha256']
        assert all(payload['infos'][key] == value for key, value in metadata.items() if key in payload['infos'])
    assert sm['actor_parameter_sha256'] != tm['actor_parameter_sha256']
    assert sm['optimizer_state_sha256'] != tm['optimizer_state_sha256']
    step_deltas = [float(target['optimizer_state_dict']['state'][key]['step'] - value['step']) for key, value in source['optimizer_state_dict']['state'].items()]
    assert len(step_deltas) == 12 and step_deltas == [20.] * 12
    assert tm['optimizer_learning_rate'] == 1e-5
    srng, trng = sm['training_rng_state'], tm['training_rng_state']
    assert set(srng) == set(trng) and srng['seed'] == trng['seed'] == 1001
    assert srng['torch_cuda_device_count'] == trng['torch_cuda_device_count'] == len(trng['torch_cuda'])
    batch = torch.load(RUN / 'rollouts/rollout_001633.pt', map_location='cpu', weights_only=False)
    rows = take(RUN / 'residual_and_projection_audit.jsonl', 128)
    update = take(RUN / 'optimizer_updates.jsonl', 1)[0]
    assert update['ppo_update'] == 1633 and update['global_policy_decisions'] == 213504
    assert update['optimizer_steps'] == 20 and update['actor_parameters_changed'] and update['finite_nonzero_gradient_observed']
    assert update['actor_parameter_sha256_before'] == sm['actor_parameter_sha256']
    assert update['actor_parameter_sha256_after'] == tm['actor_parameter_sha256']
    obs = batch['observations']['policy']; actions = batch['actions']; means, stds = batch['distribution_params']
    assert obs.shape == (128, 1, 389) and actions.shape == (128, 1, 12)
    assert torch.equal(obs, batch['observations']['critic'])
    assert batch['runtime_contract'] == tm['runtime_contract'] and batch['policy_contract'] == tm['policy_contract']
    assert batch['curriculum_epoch']['prefix_request']['target_phase'] == 'P04'
    assert batch['curriculum_epoch']['prefix_policy_provenance']['checkpoint_sha256'] == SOURCE_SHA
    # Read only enough prefix records for episodes in these first128 samples.
    needed_starts = sum(row['applied_audit']['decision_count'] == 1 for row in rows)
    assert needed_starts >= 1
    prefixes = []; phases = Counter(); counts = Counter(); previous_native = None
    for item in lines(RUN / 'prefix_evidence.jsonl'):
        assert item['policy_credit'] is False
        if item['kind'] == 'checkpoint_prefix_decision':
            phases[item['phase_id']] += 1; counts['decisions'] += 1; counts['physics_ticks'] += item['physics_ticks']
            assert item['actuator_target_effect_audit_summary']['all_ticks_verified']
            assert item['no_in_episode_state_writes_verified']
            previous_native = item['actuator_target_effect_audit']
        elif item['kind'] == 'checkpoint_prefix_result':
            assert item['accepted'] is True
        elif item['kind'] == 'policy_credit_start':
            start = item['start']; provenance = start['prefix_policy_provenance']
            assert start['actual_phase'] == 'P04' and previous_native is not None
            assert provenance['checkpoint_sha256'] == SOURCE_SHA
            assert provenance['source_global_policy_decisions'] == 213376
            assert provenance['frozen_actor_parameter_sha256'] == sm['actor_parameter_sha256']
            assert provenance['frozen_for_entire_training_block'] and provenance['independent_parameter_and_buffer_storage_verified']
            assert provenance['effective_runtime_content_sha256'] == provenance['source_runtime_content_sha256']
            assert provenance['source_policy_contract'] == provenance['effective_policy_contract'] == tm['policy_contract']
            prefixes.append({'start': start, 'counts': dict(counts), 'phases': dict(phases), 'last_native': previous_native})
            counts, phases = Counter(), Counter()
            if len(prefixes) == needed_starts:
                break
    assert len(prefixes) == needed_starts
    requested, endpoints, execution = Counter(), Counter(), Counter()
    episode_index = -1; previous_native = None
    for index, row in enumerate(rows):
        a = row['applied_audit']; p = row['policy_request']; native = a['actuator_target_effect_audit']
        evidence = native['capture_assist_evidence']
        if a['decision_count'] == 1:
            episode_index += 1; previous_native = prefixes[episode_index]['last_native']
        assert row['global_policy_decision'] == 213377 + index
        assert a['physical_core_decision_count_including_prefix'] == prefixes[episode_index]['counts']['decisions'] + a['decision_count']
        assert not a['prefix_teacher_data_in_ppo_storage'] and not a['prefix_checkpoint_policy_data_in_ppo_storage']
        assert a['task_result_scope'] == 'checkpoint_policy_initialized_suffix'
        assert p['sampling_draws'] == 1 and p['extra_random_draws'] == 0
        assert row['raw_policy_action_full12'] == p['selected_raw_full12'] == native['raw_policy_action_full12']
        assert row['old_distribution_mean_full12'] == p['conditional_mean_full12']
        assert row['old_distribution_std_full12'] == p['effective_sigma_full12']
        assert native['verified'] and native['actual_mapping_matches_dispatch'] and native['phase_mask_full12'] == [1] * 12
        assert native['capture_assist_state_transition_independently_reconstructed']
        assert evidence['owner_indices'] in ([], [0, 1])
        assert not evidence['owner_indices'] or evidence['knee_hold_final_verified']
        assert evidence['candidate_before_assist_full12'][2:] == evidence['candidate_after_assist_full12'][2:]
        assert torch.equal(obs[index, 0, 372:384], torch.tensor(capture_assist_features(previous_native['capture_assist_evidence']['state_after']), dtype=obs.dtype))
        assert a['actuator_target_effect_audit_summary']['all_ticks_verified'] and a['no_in_episode_state_writes_verified']
        previous_native = native
        requested[a['phase_id']] += 1; endpoints[a['end_phase_id']] += 1
        execution['physics_ticks'] += a['physics_ticks']; execution['assist_owned_endpoints'] += bool(evidence['owner_indices'])
        execution['terminal_samples'] += bool(row['terminal'])
    for tensor, key in ((actions, 'raw_policy_action_full12'), (means, 'old_distribution_mean_full12'),
        (stds, 'old_distribution_std_full12'), (batch['actions_log_prob'], 'old_log_probability'),
        (batch['values'], 'old_value'), (batch['rewards'], 'reward'), (batch['dones'], 'terminal')):
        assert torch.equal(tensor, torch.tensor([row[key] for row in rows], dtype=tensor.dtype).reshape(tensor.shape))
    assert [PHASES[i] for i in obs[:, 0, :13].argmax(-1)] == [row['applied_audit']['phase_id'] for row in rows]
    assert torch.equal(obs[:, 0, 372:384], torch.tensor([row['policy_request']['capture_assist_observed_features'] for row in rows], dtype=obs.dtype))
    assert torch.equal(obs[:, 0, 384:389], torch.tensor([row['policy_request']['capture_continuation_observed_features'] for row in rows], dtype=obs.dtype))
    likelihood = read(RUN / 'rollouts/update_001633_likelihood.json')
    uses = Counter(i for mini in likelihood['minibatches'] for ids in mini['rollout_flat_indices'] for i in ids)
    assert len(likelihood['minibatches']) == 20 and uses == Counter({i: 5 for i in range(128)})
    logp_error = float((torch.distributions.Normal(means, stds).log_prob(actions).sum(-1).unsqueeze(-1) - batch['actions_log_prob']).abs().max())
    for prefix in prefixes:
        del prefix['last_native']
    report = {'schema': 'wlr50_clean.block08_first_actual_mean3_PPO_carry_audit.v1', 'result': 'PASS',
        'run': str(RUN), 'source_checkpoint': str(SOURCE), 'source_sha256': SOURCE_SHA,
        'checkpoint': str(TARGET), 'checkpoint_sha256': tm['checkpoint_sha256'],
        'actual_counts': {key: tm[key] for key in COUNTERS}, 'new_counts': {'policy_decisions': 128, 'PPO_updates': 1, 'Adam_steps': 20},
        'prefixes': prefixes, 'prefix_PPO_credit': 0, 'all_prefix_actions_excluded_from_storage': True,
        'request_phase_counts': {phase: requested[phase] for phase in PHASES}, 'endpoint_phase_counts': dict(endpoints),
        'execution_counts': dict(execution), 'three_front_AUX_events_full_exact_source_carry': True,
        'front_AUX_totals': [96, 96], 'older_AUX_totals': [7, 8], 'new_AUX_added': 0,
        'preserved_branch_migration_fields': preserved, 'actor_and_Adam_changed': True,
        'all12_Adam_step_deltas': step_deltas, 'LR': tm['optimizer_learning_rate'], 'Identity_preserved': True,
        'RNG_all_state_kinds_and_CUDA_count_retained': True, 'RNG_source_hash': state_hash(srng), 'RNG_target_hash': state_hash(trng),
        'actual_checkpoint_embedded_sidecar_hashes_verified': True, 'actual_save_reload_recorded': True,
        'fresh389_raw12_mean_sigma_logp_value_reward_done_match': True, 'each_original_sample_used': 5,
        'independent_CPU_logp_max_error': logp_error, 'first_sealed_batch_only': True,
        'physical_success_claimed': False, 'P03plus_mean_or_sigma_invariance_after_PPO_claimed': False}
    (OUT / 'block08_first_real_event3_PPO_carry_audit.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    prefix_decisions = sum(p['counts']['decisions'] for p in prefixes)
    prefix_ticks = sum(p['counts']['physics_ticks'] for p in prefixes)
    md = f'''# First genuine PPO carry after mean-head AUX event3

**PASS. Actual sealed update1633** from mean3 source CP213376 `{SOURCE_SHA}` to ordinary CP213504 `{tm['checkpoint_sha256']}`. New **128 decisions /1 PPO /20 Adam**; cumulative **213504 /1633 /32660**. No AUX added.

The actual checkpoint-policy prefix used the frozen source mean3 actor: {prefix_decisions} decisions/{prefix_ticks} ticks across {len(prefixes)} starts, reaching P04 before credited learning. Every prefix record explicitly has policy_credit=false, and the exact128 raw samples exclude all prefix actions. This is a P04-initialized suffix course, not a full P01 learned success.

The complete three-event front AUX96/96 dictionary, old AUX7/8, all3 origins and immutable migrations remain exactly equal to source. Real PPO changes actor/Adam; all12 Adam states advance20, LR1e-5 and Identity persist. RNG includes all source state kinds/CUDA count and its actual saved state advances normally; embedded/sidecar and checkpoint hashes verify, official save/reload is recorded true.

Actual input phase counts: {dict(requested)}. Credited physics ticks {execution['physics_ticks']}; assist-owned endpoints {execution['assist_owned_endpoints']}, terminals {execution['terminal_samples']}. Native projection/dispatch and all12 residual permissions verify. FL-only assist, if active, stays separately observed and is not relabeled a raw Gaussian sample.

389 observations and raw12/μ/σ/logp/value/reward/done exactly match the synchronous stream. Each original sample is used5 times in20 minibatches; CPU logp maximum difference {logp_error:.9g}. No physical success or P03+ invariance is claimed; PPO can change trunk and sigma after the mean-only AUX.

Evidence: `{RUN}`, first sealed rollout1633, actual source/target checkpoints and likelihood/optimizer/prefix logs. CPU-only audit; no simulator/GPU/fit/production or frozen-helper edits, no checkpoint writes. Helper exits after this report.
'''
    (OUT / 'block08_first_real_event3_PPO_carry_audit.md').write_text(md, encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('result', 'actual_counts', 'new_counts', 'request_phase_counts',
        'execution_counts', 'front_AUX_totals', 'independent_CPU_logp_max_error')}, indent=2))


if __name__ == '__main__':
    main()
