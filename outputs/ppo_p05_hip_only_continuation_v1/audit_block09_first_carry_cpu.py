"""One sealed real update after RR AUX4 + receiver-v2; CPU/read-only."""
from collections import Counter
import json
from pathlib import Path

import torch
import audit_block08_first_carry_cpu as h
from wlr50_clean.ppo.semantic_training import state_hash

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
RUN = ROOT / 'runs/ppo_p05_hip_only_continuation_v1/train/20260922T1428131775302Z_g336b7c56d2f0_23b26a4c40a24a178d190def03381413'
SOURCE = OUT / 'checkpoints/history/checkpoint_rr_receiver_v2_step_000214400.pt'
TARGET = OUT / 'checkpoints/history/checkpoint_step_000214528.pt'
SOURCE_SHA = 'a039f071def736bcb8d2f9b6fb0c7e4691d9aeb74715918521b2268e9a521d9a'
COUNTERS = ('global_policy_decisions', 'ppo_updates', 'optimizer_steps')


def main():
    assert not torch.cuda.is_available(); torch.set_num_threads(1)
    sm, tm = (h.read(path.with_name(path.stem + '_manifest.json')) for path in (SOURCE, TARGET))
    assert h.sha(SOURCE) == sm['checkpoint_sha256'] == SOURCE_SHA
    assert h.sha(TARGET) == tm['checkpoint_sha256'] and tm['source_run'] == str(RUN) and tm['save_load_round_trip']
    assert tuple(tm[k] - sm[k] for k in COUNTERS) == (128, 1, 20)
    assert tuple(tm[k] for k in COUNTERS) == (214528, 1641, 32820)
    preserved = [k for k in sm if k != 'resume_migration' and k.endswith(('_branch', '_migration'))]
    assert all(sm[k] == tm[k] for k in preserved)
    assert 'rr_receiver_retirement_v2_branch' in preserved and 'rr_receiver_retirement_v2_migration' in preserved
    origins = {}
    for name in ('p05_capture_assist', 'capture_feedback_semantics', 'rr_postcross_workspace', 'rr_receiver_retirement_v2'):
        origin = tm[name + '_branch']['counter_origin']; actual = tm[name + '_branch_counts']
        assert actual == {k: tm[k] - origin[k] for k in COUNTERS}
        assert {k: actual[k] - sm[name + '_branch_counts'][k] for k in COUNTERS} == dict(zip(COUNTERS, (128, 1, 20)))
        origins[name] = {'counter_origin': origin, 'actual_counts': actual}
    assert origins['rr_receiver_retirement_v2']['counter_origin'] == dict(zip(COUNTERS, (214400, 1640, 32800)))
    migration = tm['rr_receiver_retirement_v2_migration']
    assert h.sha(migration['plan_path']) == migration['plan_sha256']
    assert migration['source_checkpoint_sha256'] == '9987a4df3b4a2afaeddac794aa97b650e6133fe7ec89e8484cf59084cda69b6a'
    assert migration['target_git_commit'] == tm['runtime_contract']['source_git_commit']
    assert tm['runtime_contract']['source_git_commit'].startswith('336b7c56d2f0')
    for key in ('runtime_contract', 'policy_contract', 'runner_config', 'normalization', 'normalizer_state_sha256'):
        assert sm[key] == tm[key]
    ledger = tm['rr_postcross_workspace_branch']['front_rehearsal_auxiliary']
    assert ledger == sm['rr_postcross_workspace_branch']['front_rehearsal_auxiliary']
    assert len(ledger['events']) == 4 and [e['event_index'] for e in ledger['events']] == [1, 2, 3, 4]
    assert [ledger[k] for k in ('accepted_auxiliary_updates_total', 'attempted_auxiliary_optimizer_steps_total')] == [103, 104]
    assert [(e['fit_report']['accepted_auxiliary_updates'], e['fit_report']['attempted_auxiliary_optimizer_steps']) for e in ledger['events']] == [(32,32),(32,32),(32,32),(7,8)]
    assert ledger['events'][3]['kind'] == 'finite_supervised_existing_mean_head_actual_RR_capture_continuation_raw_actions_not_PPO'
    older = tm['task_conditioned_hip_wheel_branch']['auxiliary_mean_learning']
    assert [older[k] for k in ('accepted_auxiliary_updates_total', 'attempted_auxiliary_optimizer_steps_total')] == [7, 8]
    source, target = (torch.load(p, map_location='cpu', weights_only=False) for p in (SOURCE, TARGET))
    for payload, metadata in ((source, sm), (target, tm)):
        assert h.parameter_sha(payload['actor_state_dict']) == metadata['actor_parameter_sha256']
        assert h.parameter_sha(payload['critic_state_dict']) == metadata['critic_parameter_sha256']
        assert state_hash(payload['optimizer_state_dict']) == metadata['optimizer_state_sha256']
        assert all(payload['infos'][k] == v for k, v in metadata.items() if k in payload['infos'])
    assert sm['actor_parameter_sha256'] != tm['actor_parameter_sha256'] and sm['optimizer_state_sha256'] != tm['optimizer_state_sha256']
    deltas = [float(target['optimizer_state_dict']['state'][k]['step'] - v['step']) for k, v in source['optimizer_state_dict']['state'].items()]
    assert len(deltas) == 12 and deltas == [20.] * 12
    assert tm['optimizer_learning_rate'] == 1e-5 and 'identity_RSL_normalizer' in tm['normalization']
    srng, trng = sm['training_rng_state'], tm['training_rng_state']
    assert set(srng) == set(trng) and srng['seed'] == trng['seed'] == 1001
    assert srng['torch_cuda_device_count'] == trng['torch_cuda_device_count'] == len(trng['torch_cuda']) == 1
    assert state_hash(srng) != state_hash(trng), 'real sampling/optimizer RNG did not advance'
    batch = torch.load(RUN / 'rollouts/rollout_001641.pt', map_location='cpu', weights_only=False)
    rows = h.take(RUN / 'residual_and_projection_audit.jsonl', 128)
    update = h.take(RUN / 'optimizer_updates.jsonl', 1)[0]
    assert update['ppo_update'] == 1641 and update['global_policy_decisions'] == 214528
    assert update['optimizer_steps'] == 20 and update['actor_parameters_changed'] and update['finite_nonzero_gradient_observed']
    assert update['actor_parameter_sha256_before'] == sm['actor_parameter_sha256'] and update['actor_parameter_sha256_after'] == tm['actor_parameter_sha256']
    obs = batch['observations']['policy']; actions = batch['actions']; means, stds = batch['distribution_params']
    assert obs.shape == (128,1,389) and actions.shape == (128,1,12) and torch.equal(obs,batch['observations']['critic'])
    assert batch['runtime_contract'] == tm['runtime_contract'] and batch['policy_contract'] == tm['policy_contract']
    assert batch['curriculum_epoch']['prefix_request']['target_phase'] == 'P04'
    assert batch['curriculum_epoch']['prefix_policy_provenance']['checkpoint_sha256'] == SOURCE_SHA
    # This bounded batch has one real prefix; no future stream is needed.
    assert sum(row['applied_audit']['decision_count'] == 1 for row in rows) == 1
    prefix_counts = Counter(); prefix_phases = Counter(); previous_native = None; start = None
    for row in h.lines(RUN / 'prefix_evidence.jsonl'):
        assert row['policy_credit'] is False
        if row['kind'] == 'checkpoint_prefix_decision':
            prefix_counts['decisions'] += 1; prefix_counts['physics_ticks'] += row['physics_ticks']
            prefix_phases[row['phase_id']] += 1
            assert row['actuator_target_effect_audit_summary']['all_ticks_verified'] and row['no_in_episode_state_writes_verified']
            previous_native = row['actuator_target_effect_audit']
        elif row['kind'] == 'checkpoint_prefix_result':
            assert row['accepted'] is True
        elif row['kind'] == 'policy_credit_start':
            start = row['start']; provenance = start['prefix_policy_provenance']
            assert start['actual_phase'] == 'P04' and previous_native is not None
            assert provenance['checkpoint_sha256'] == SOURCE_SHA and provenance['source_global_policy_decisions'] == 214400
            assert provenance['frozen_actor_parameter_sha256'] == sm['actor_parameter_sha256']
            assert provenance['frozen_for_entire_training_block'] and provenance['independent_parameter_and_buffer_storage_verified']
            assert provenance['effective_runtime_content_sha256'] == provenance['source_runtime_content_sha256'] == sm['runtime_contract']['runtime_content_sha256']
            assert provenance['source_policy_contract'] == provenance['effective_policy_contract'] == tm['policy_contract']
            break
    assert start is not None
    phases, endpoints, execution = Counter(), Counter(), Counter(); previous = None
    for index, row in enumerate(rows):
        a = row['applied_audit']; p = row['policy_request']; native = a['actuator_target_effect_audit']; assist = native['capture_assist_evidence']
        assert row['global_policy_decision'] == 214401 + index and a['decision_count'] == index + 1
        assert a['physical_core_decision_count_including_prefix'] == prefix_counts['decisions'] + index + 1
        assert not a['prefix_teacher_data_in_ppo_storage'] and not a['prefix_checkpoint_policy_data_in_ppo_storage']
        assert a['task_result_scope'] == 'checkpoint_policy_initialized_suffix'
        assert p['sampling_draws'] == 1 and p['extra_random_draws'] == 0
        assert row['raw_policy_action_full12'] == p['selected_raw_full12'] == native['raw_policy_action_full12']
        assert row['old_distribution_mean_full12'] == p['conditional_mean_full12'] and row['old_distribution_std_full12'] == p['effective_sigma_full12']
        assert native['verified'] and native['actual_mapping_matches_dispatch'] and native['phase_mask_full12'] == [1] * 12
        assert native['capture_assist_state_transition_independently_reconstructed']
        assert assist['owner_indices'] in ([], [0,1]) and (not assist['owner_indices'] or assist['knee_hold_final_verified'])
        assert assist['candidate_before_assist_full12'][2:] == assist['candidate_after_assist_full12'][2:]
        assert torch.equal(obs[index,0,372:384], torch.tensor(h.capture_assist_features(previous_native['capture_assist_evidence']['state_after']),dtype=obs.dtype))
        assert a['actuator_target_effect_audit_summary']['all_ticks_verified'] and a['no_in_episode_state_writes_verified']
        assert len(a['actuator_target_effect_audit_ticks']) == a['physics_ticks'] and all(t['verified'] for t in a['actuator_target_effect_audit_ticks'])
        if previous is not None:
            assert not previous['terminal'] and previous['applied_audit']['end_phase_id'] == a['phase_id']
            assert previous['applied_audit']['physics_tick'] == a['physics_tick'] - a['physics_ticks']
        if a['phase_id'] != a['end_phase_id']:
            execution['ordinary_phase_transitions'] += 1
            assert not row['terminal'], 'ordinary stage transition cut PPO episode'
        previous = row; previous_native = native
        phases[a['phase_id']] += 1; endpoints[a['end_phase_id']] += 1
        execution['physics_ticks'] += a['physics_ticks']; execution['assist_owned_endpoints'] += bool(assist['owner_indices'])
        execution['terminal_samples'] += bool(row['terminal'])
    for tensor,key in ((actions,'raw_policy_action_full12'),(means,'old_distribution_mean_full12'),(stds,'old_distribution_std_full12'),
        (batch['actions_log_prob'],'old_log_probability'),(batch['values'],'old_value'),(batch['rewards'],'reward'),(batch['dones'],'terminal')):
        assert torch.isfinite(tensor).all() and torch.equal(tensor,torch.tensor([r[key] for r in rows],dtype=tensor.dtype).reshape(tensor.shape))
    assert [h.PHASES[i] for i in obs[:,0,:13].argmax(-1)] == [r['applied_audit']['phase_id'] for r in rows]
    for begin,end,key in ((372,384,'capture_assist_observed_features'),(384,389,'capture_continuation_observed_features')):
        assert torch.equal(obs[:,0,begin:end],torch.tensor([r['policy_request'][key] for r in rows],dtype=obs.dtype))
    assert torch.equal(obs[:,0,17],torch.tensor([r['applied_audit']['reward_breakdown']['potential_before'] for r in rows],dtype=obs.dtype))
    # Front-state batch: no actual RR retirement consumption claimed.
    rr = {name: int(obs[:,0,index].sum()) for name,index in (('qualified_current',149),('crossed_history',153),('placed_history',157))}
    assert rr == dict(qualified_current=0,crossed_history=0,placed_history=0)
    likelihood = h.read(RUN / 'rollouts/update_001641_likelihood.json')
    uses = Counter(i for mini in likelihood['minibatches'] for ids in mini['rollout_flat_indices'] for i in ids)
    assert len(likelihood['minibatches']) == 20 and uses == Counter({i:5 for i in range(128)})
    logp_error = float((torch.distributions.Normal(means,stds).log_prob(actions).sum(-1).unsqueeze(-1)-batch['actions_log_prob']).abs().max())
    assert logp_error <= 1e-5
    report = {'schema':'wlr50_clean.block09_first_real_event4_receiver_v2_PPO_carry.v1','result':'PASS',
        'run':str(RUN),'source_checkpoint':str(SOURCE),'source_sha256':SOURCE_SHA,'checkpoint':str(TARGET),'checkpoint_sha256':tm['checkpoint_sha256'],
        'actual_counts':{k:tm[k] for k in COUNTERS},'new_counts':dict(policy_decisions=128,PPO_updates=1,Adam_steps=20),
        'prefix_counts':dict(prefix_counts),'prefix_phase_counts':dict(prefix_phases),'prefix_PPO_credit':0,
        'prefix_actual_phase':start['actual_phase'],'prefix_all_actions_excluded':True,'first_learning_input_tick':rows[0]['applied_audit']['physics_tick']-rows[0]['applied_audit']['physics_ticks'],
        'request_phase_counts':{phase:phases[phase] for phase in h.PHASES},'endpoint_phase_counts':dict(endpoints),'execution_counts':dict(execution),
        'raw389_full12_mean_sigma_logp_value_reward_done_sealed_exact':True,'each_original_sample_used':5,'CPU_logp_max_error':logp_error,
        'actor_and_Adam_changed':True,'all12_Adam_step_deltas':deltas,'effective_LR':tm['optimizer_learning_rate'],'Identity_preserved':True,
        'RNG_all_state_kinds_and_CUDA_count_preserved_and_advanced':True,'RNG_source_sha256':state_hash(srng),'RNG_target_sha256':state_hash(trng),
        'embedded_sidecar_payload_hashes_verified':True,'actual_save_reload_recorded':True,
        'all_four_events_full_exact_source_carry':True,'inherited_front_AUX':[96,96],'inherited_RR_AUX':[7,8],
        'mixed_ledger_AUX':[103,104],'older_separate_AUX':[7,8],'new_AUX_added':0,
        'preserved_branch_migration_fields':preserved,'four_origins_and_actual_branch_counts':origins,
        'receiver_v2_migration_plan_sha256':migration['plan_sha256'],'actual_RR_input_coverage':rr,'RR_retirement_active_learning_claimed':False,
        'physical_success_claimed':False,'suffix_not_full_P01_learning':True,'first_sealed_batch_only':True,'audit_fit_or_checkpoint_writes':0}
    (OUT/'block09_first_real_event4_v2_PPO_carry_audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    md=f'''# Block09 first actual PPO carry — RR AUX4 + receiver-v2

**PASS.** Migrated CP214400 `{SOURCE_SHA}` → actual CP214528 `{tm['checkpoint_sha256']}`. New **128 decisions /1 PPO /20 Adam**, cumulative **214528 /1641 /32820**. No new AUX.

All four full AUX events carry exactly: prior front96/96 + RR7/8 = mixed103/104; older separate7/8 unchanged. Original three origins/migrations persist, and the new receiver-v2 origin214400/1640/32800 records exactly128/1/20. The migration record and plan binding remain intact.

Frozen actual migrated-source prefix: **{prefix_counts['decisions']} actions /{prefix_counts['physics_ticks']} ticks**, reaching P04 before learning; every prefix action has zero PPO credit. The128 learner inputs are **{dict(phases)}**, with {execution['physics_ticks']} verified physics ticks, {execution['ordinary_phase_transitions']} ordinary phase handoffs, {execution['terminal_samples']} terminal samples and {execution['assist_owned_endpoints']} assist-owned endpoints. Handoffs remain continuous and do not set done. This is a P04-initialized suffix, not full P01 learned success.

Direct389/raw12/μ/σ/logp/value/reward/done match the sealed rollout and synchronous native audit. Each original sample is used5 times in20 minibatches; CPU Gaussian logp max error **{logp_error:.9g}**. All12 residual permissions and separately observed FL-only assist execution validate. RR qualified/crossed/placed input counts are0: this first front-state batch does not yet prove active RR reward learning.

Actor and Adam truly change; all12 Adam states advance20, effective LR1e-5 and Identity persist. Full saved RNG schema/CUDA count remain and state advances. Embedded/sidecar/tensor/optimizer hashes verify and actual official save/reload is recorded true. PPO may now change sigma/trunk normally; mean-head AUX invariance is not falsely carried over to learning.

Read-only CPU audit of first sealed rollout1641 and actual source/target; no physics/GPU/fit/production edits/checkpoint writes. CPU helper exits after writing this compact report.
'''
    (OUT/'block09_first_real_event4_v2_PPO_carry_audit.md').write_text(md,encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('result','actual_counts','new_counts','prefix_counts','request_phase_counts','execution_counts','CPU_logp_max_error','four_origins_and_actual_branch_counts')},indent=2))


if __name__=='__main__':
    main()
