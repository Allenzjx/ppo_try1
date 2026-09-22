"""Read sealed block04 only; stdout JSON, no writes or runtime replay/imports."""
import hashlib
import json
import io
import math
from collections import Counter
from pathlib import Path

import torch

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'runs/ppo_p05_hip_only_continuation_v1/train/20260922T0814225788964Z_ga802b24d78df_790387038d25495ea5f001f934d3ade2'
CPDIR = ROOT / 'outputs/ppo_p05_hip_only_continuation_v1/checkpoints/history'
PHASES = [f'P{i:02}' for i in range(1, 14)]
MODES = ['WAIT', 'DESCEND', 'HOLD', 'BLOCKED', 'RELEASE', 'RELEASED']


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def lines(name):
    with (RUN / name).open(encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def finite_tree(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def all_phases(counter):
    return {phase: counter[phase] for phase in PHASES}


manifest = read_json(RUN / 'training_manifest.json')
cm = read_json(CPDIR / 'checkpoint_step_000207872_manifest.json')
assert manifest['lifecycle'] == 'STOPPED_AT_VERIFIED_UPDATE_BOUNDARY'
assert manifest['planned_requested_policy_decisions'] == 4096 and manifest['actual_policy_decisions'] == 2048
assert manifest['unconsumed_requested_policy_decisions'] == 2048
requests, endpoints, ep_requests = Counter(), Counter(), {}
rr_end, modes_end, owner_phases, counts = Counter(), Counter(), Counter(), Counter()
records, episode_stats = [], {}
previous_ev = None
for index, row in enumerate(lines('residual_and_projection_audit.jsonl')):
    a = row['applied_audit']; task = a['semantic_task']; ev = task['physical_evaluator']
    rr = ev['current_legs']['RR']; history = ev['history']; audit = a['actuator_target_effect_audit']
    state = audit['capture_assist_evidence']['state_after']
    episode = a['curriculum_start']['prefix_attempt_index']
    if a['decision_count'] == 1:
        previous_ev = None
    requests[a['phase_id']] += 1; endpoints[a['end_phase_id']] += 1
    ep_requests.setdefault(episode, Counter())[a['phase_id']] += 1
    rr_end['qualified'] += bool(rr.get('current_lift_valid'))
    rr_end['established'] += bool(rr.get('lift_established'))
    rr_end['crossed'] += bool(history['front_edge_crossed']['RR'])
    rr_end['TOP'] += bool(rr['top_contact'])
    rr_end['top_surface_contact'] += bool(rr['top_surface_contact'])
    rr_end['placed'] += bool(history['placed']['RR'])
    rr_end['crossed_unplaced'] += bool(history['front_edge_crossed']['RR'] and not history['placed']['RR'])
    counts['input_TOP_known'] += previous_ev is not None
    counts['input_TOP_positive_known'] += bool(previous_ev and previous_ev['current_legs']['RR']['top_contact'])
    previous_ev = ev
    owned = audit['capture_assist_evidence']['owner_indices']
    assert owned in ([], [0, 1])
    modes_end[state['mode_name']] += 1
    if owned:
        counts['assist_owned_endpoints'] += 1; owner_phases[a['end_phase_id']] += 1
    receipt = audit['capture_assist_evidence']
    assert receipt['candidate_before_assist_full12'][2:] == receipt['candidate_after_assist_full12'][2:]
    assert audit['verified'] and audit['actual_mapping_matches_dispatch']
    assert audit['capture_assist_state_transition_independently_reconstructed']
    assert audit['phase_mask_full12'] == [1] * 12
    assert a['no_in_episode_state_writes_verified']
    assert not a['prefix_teacher_data_in_ppo_storage'] and not a['prefix_checkpoint_policy_data_in_ppo_storage']
    assert a['task_result_scope'] == 'checkpoint_policy_initialized_suffix'
    assert a['actuator_target_effect_audit_summary']['all_ticks_verified']
    assert len(a['actuator_target_effect_audit_ticks']) == a['physics_ticks']
    assert all(t['verified'] for t in a['actuator_target_effect_audit_ticks'])
    assert row['raw_policy_action_full12'] == a['raw_policy_action_full12'] == audit['raw_policy_action_full12'] == row['policy_request']['selected_raw_full12']
    assert row['old_distribution_mean_full12'] == row['policy_request']['conditional_mean_full12']
    assert row['old_distribution_std_full12'] == row['policy_request']['effective_sigma_full12']
    assert row['policy_request']['sampling_draws'] == 1 and row['policy_request']['extra_random_draws'] == 0
    counts['credited_physics_ticks'] += a['physics_ticks']
    counts['verified_native_ticks'] += a['actuator_target_effect_audit_summary']['verified_tick_count']
    counts['assist_inactive_all12_unmodified_endpoints'] += bool(audit['all12_policy_channels_unmodified_at_actuator'])
    counts['pending_endpoints'] += bool(task['fl_capture_pending'])
    counts['allow_continuation_endpoints'] += bool(task['allow_capture_continuation'])
    counts['pending_advanced_endpoints'] += bool(task['capture_continuation']['scheduler_advanced_pending'])
    counts['terminal_decisions'] += bool(row['terminal'])
    assert row['global_policy_decision'] == 205825 + index
    assert finite_tree([row['raw_policy_action_full12'], row['old_log_probability'], row['old_value'], row['reward']])
    records.append(dict(global_decision=row['global_policy_decision'], phase=a['phase_id'], first_input=a['decision_count']==1,
        raw=row['raw_policy_action_full12'], mean=row['old_distribution_mean_full12'], std=row['old_distribution_std_full12'],
        logp=row['old_log_probability'], terminal=row['terminal'], reward=row['reward'], policy=row['policy_request']))
    stat = episode_stats.setdefault(episode, dict(decisions=0, physics_ticks=0, RR_qualified=0, RR_crossed=0, RR_TOP=0, RR_placed=0))
    stat['decisions'] += 1; stat['physics_ticks'] += a['physics_ticks']
    for key, flag in [('RR_qualified', rr.get('current_lift_valid')), ('RR_crossed', history['front_edge_crossed']['RR']),
                      ('RR_TOP', rr['top_contact']), ('RR_placed', history['placed']['RR'])]:
        stat[key] += bool(flag)
    stat.update(last_global_decision=row['global_policy_decision'], last_phase=a['end_phase_id'],
        last_physics_tick=a['physics_tick'], physical_duration_s=a['sim_time_s'],
        prefix_ticks=a['curriculum_start']['physics_tick'], terminal=row['terminal'],
        termination_reason=a['termination_reason'], placed_history=history['placed'], event_ticks=history['event_ticks'],
        last_RR={k:rr[k] for k in ('current_lift_valid','top_contact','top_surface_contact','ground_contact','air','clearance_m','front_distance_m','bearing_force_n','contact_mode')})

assert len(records) == 2048
rr_input, modes_input, per_rollout, input_counts = Counter(), Counter(), [], Counter()
handoff_ground_inputs = []
maximum_logp_error = 0.
for update in range(1574, 1590):
    data = torch.load(RUN / f'rollouts/rollout_{update:06}.pt', map_location='cpu', weights_only=False)
    assert finite_tree(data)
    obs = data['observations']['policy']; act = data['actions']; mean, std = data['distribution_params']
    assert list(obs.shape) == [128, 1, 389] and list(act.shape) == [128, 1, 12]
    assert torch.equal(obs, data['observations']['critic'])
    assert data['runtime_contract']['source_git_commit'] == 'a802b24d78df5f1b8f0caf914ce7a8a68a98db8e'
    start = (update - 1574) * 128; segment = records[start:start + 128]
    def equal_tensor(actual, values):
        assert torch.equal(actual, torch.tensor(values, dtype=actual.dtype).reshape(actual.shape))
    equal_tensor(act, [r['raw'] for r in segment]); equal_tensor(mean, [r['mean'] for r in segment])
    equal_tensor(std, [r['std'] for r in segment]); equal_tensor(data['actions_log_prob'], [r['logp'] for r in segment])
    equal_tensor(data['rewards'], [r['reward'] for r in segment]); equal_tensor(data['dones'], [r['terminal'] for r in segment])
    err = (torch.distributions.Normal(mean, std).log_prob(act).sum(-1).unsqueeze(-1) - data['actions_log_prob']).abs().max().item()
    maximum_logp_error = max(maximum_logp_error, err)
    phase_ids = obs[:, 0, :13].argmax(-1).tolist(); phase_counts = Counter(PHASES[i] for i in phase_ids)
    for i, r in enumerate(segment):
        assert PHASES[phase_ids[i]] == r['phase']
        assert bool(obs[i, 0, 149]) == r['policy']['current_RR_qualification']
        assert bool(obs[i, 0, 157]) == r['policy']['RR_placed_history']
        if r['first_input']:
            # Existing389 index137 is RR exact ground-pair active (FL FR RL RR).
            # In all_stage_v1 loaded/TOP excludes ground, so this proves NOT TOP.
            assert float(obs[i, 0, 137]) == 1.
            handoff_ground_inputs.append(r['global_decision'])
    equal_tensor(obs[:, 0, 372:384], [r['policy']['capture_assist_observed_features'] for r in segment])
    equal_tensor(obs[:, 0, 384:389], [r['policy']['capture_continuation_observed_features'] for r in segment])
    q = int((obs[:, 0, 149] == 1).sum()); c = int((obs[:, 0, 153] == 1).sum()); p = int((obs[:, 0, 157] == 1).sum())
    rr_input.update(qualified=q, crossed=c, placed=p, crossed_unplaced=int(((obs[:, 0, 153] == 1) & (obs[:, 0, 157] == 0)).sum()))
    modes_input.update(MODES[round(float(v)*5)] for v in obs[:, 0, 372])
    input_counts['pending'] += int((obs[:, 0, 384] == 1).sum())
    input_counts['allow_continuation'] += int((obs[:, 0, 385] == 1).sum())
    input_counts['pending_advanced'] += int((obs[:, 0, 386] == 1).sum())
    likelihood = read_json(RUN / f'rollouts/update_{update:06}_likelihood.json')
    use = Counter(j for batch in likelihood['minibatches'] for ids in batch['rollout_flat_indices'] for j in ids)
    assert len(likelihood['minibatches']) == 20 and use == Counter({i: 5 for i in range(128)})
    assert finite_tree(likelihood)
    per_rollout.append(dict(ppo_update=update, request_phases=dict(phase_counts), RR_qualified_input=q, RR_crossed_input=c, RR_placed_input=p))

updates = list(lines('optimizer_updates.jsonl'))
assert len(updates) == 16 and [u['ppo_update'] for u in updates] == list(range(1574,1590))
assert all(finite_tree(u) and u['actor_parameters_changed'] and u['finite_nonzero_gradient_observed'] and u['optimizer_steps']==20 for u in updates)
assert all(a['actor_parameter_sha256_after']==b['actor_parameter_sha256_before'] for a,b in zip(updates,updates[1:]))
prefix_counts, prefix_phases, starts, attempts = Counter(), Counter(), [], []
for row in lines('prefix_evidence.jsonl'):
    assert row['policy_credit'] is False
    if row['kind'] == 'checkpoint_prefix_decision':
        prefix_counts['decisions'] += 1; prefix_counts['physics_ticks'] += row['physics_ticks']; prefix_phases[row['phase_id']] += 1
        assert row['actuator_target_effect_audit_summary']['all_ticks_verified'] and row['no_in_episode_state_writes_verified']
    elif row['kind'] == 'checkpoint_prefix_result':
        attempts.append({k:row[k] for k in ('prefix_decisions','prefix_physics_ticks','accepted','miss')})
    elif row['kind'] == 'policy_credit_start':
        starts.append({k:row['start'][k] for k in ('mode','physics_tick','sim_time_s','from_P01_current_policy')})
assert len(attempts)==3 and all(a['accepted'] for a in attempts)
assert all(a['prefix_decisions']==562 and a['prefix_physics_ticks']==4496 for a in attempts)
assert len(handoff_ground_inputs)==3
assert prefix_counts['decisions']==1686 and prefix_counts['physics_ticks']==13488
completed = []
for row in lines('completed_episodes.jsonl'):
    completed.append({k:v for k,v in row.items() if k!='terminal_info'})
assert [r['policy_decisions'] for r in completed] == [721,700]
assert episode_stats[2]['decisions']==627 and not episode_stats[2]['terminal']
for e, stat in episode_stats.items():
    stat['request_phases'] = all_phases(ep_requests[e]); stat['credited_duration_s'] = stat['physics_ticks']/120

source = torch.load(CPDIR/'checkpoint_step_000205824.pt',map_location='cpu',weights_only=False)
target_path = CPDIR/'checkpoint_step_000207872.pt'
target = torch.load(target_path,map_location='cpu',weights_only=False)
assert finite_tree(target)
step_delta = {str(k):float(target['optimizer_state_dict']['state'][k]['step']-v['step']) for k,v in source['optimizer_state_dict']['state'].items()}
assert len(step_delta)==12 and set(step_delta.values())=={320.}
with target_path.open('rb') as stream:
    checkpoint_hash = hashlib.file_digest(stream,'sha256').hexdigest()
assert checkpoint_hash==cm['checkpoint_sha256']=='8f15de6495adc49c6d7e19000bc56ca612ce9861efaa9811bfab5bd057ea9cf5'

def exact_tree(a,b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and torch.equal(a,b)
    if isinstance(a,dict):
        return isinstance(b,dict) and a.keys()==b.keys() and all(exact_tree(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):
        return type(a) is type(b) and len(a)==len(b) and all(exact_tree(x,y) for x,y in zip(a,b))
    if hasattr(a,'shape') and hasattr(a,'dtype'):
        return type(a) is type(b) and a.shape==b.shape and a.dtype==b.dtype and bool((a==b).all())
    return a==b

# Fresh serialization/reload is CPU in memory only, NOT a new simulator resume.
buffer=io.BytesIO()
torch.save(target,buffer)
buffer.seek(0)
fresh=torch.load(buffer,map_location='cpu',weights_only=False)
assert exact_tree(target,fresh) and finite_tree(fresh)
with (CPDIR/'checkpoint_step_000207872_manifest.json').open('rb') as stream:
    manifest_hash=hashlib.file_digest(stream,'sha256').hexdigest()
with (RUN/'training_manifest.json').open('rb') as stream:
    training_hash=hashlib.file_digest(stream,'sha256').hexdigest()

result = dict(schema='wlr50_clean.p05_capture_block04_training_audit.v1',
    scope='read_only_CPU_audit_of_sealed_old_runtime_logs; no_native_replay_with_modified_capture_class',
    run_directory=RUN.as_posix(), runtime_commit='a802b24d78df5f1b8f0caf914ce7a8a68a98db8e',
    checkpoint=target_path.as_posix(), checkpoint_sha256=checkpoint_hash,
    recorded_save_load_round_trip=cm['save_load_round_trip'], lifecycle=manifest['lifecycle'],
    planned_decisions=4096, consumed_decisions=2048, unconsumed_decisions=2048,
    p05_capture_branch_cumulative=cm['p05_capture_assist_branch_counts'],
    feedback_v2_branch_cumulative=cm['capture_feedback_semantics_branch_counts'],
    source_checkpoint=(CPDIR/'checkpoint_step_000205824.pt').as_posix(),
    source_checkpoint_sha256=manifest['telemetry']['core']['prefix_policy_provenance']['checkpoint_sha256'],
    fresh_CPU_in_memory_checkpoint_roundtrip=True,
    checkpoint_manifest_sha256=manifest_hash, training_manifest_sha256=training_hash,
    optimizer_update_range=[1574,1589], wall_time_s=manifest['wall_time_s'],
    frozen_prefix_source=manifest['telemetry']['core']['prefix_policy_provenance']['source_global_policy_decisions'],
    new_counts=dict(policy_decisions=2048,ppo_updates=16,optimizer_steps=320),
    lifetime_counts={k:cm[k] for k in ('global_policy_decisions','ppo_updates','optimizer_steps')},
    request_phase_counts=all_phases(requests), endpoint_phase_counts=all_phases(endpoints),
    RR_policy_input_counts=dict(rr_input), RR_endpoint_counts=dict(rr_end),
    RR_input_TOP_measurement=dict(known_previous_credited_endpoint_count=counts['input_TOP_known'],
        positive_known_count=counts['input_TOP_positive_known'], exact_positive_total=counts['input_TOP_positive_known'],
        prefix_handoff_proven_not_TOP_by_saved_ground_pair=handoff_ground_inputs, unknown_input_count=0,
        note='2045 input TOP states equal prior physical endpoints; 3 prefix handoff inputs have saved raw389 RR ground-pair flag1, excluding TOP by all_stage_v1 semantics.'),
    capture_modes_policy_input=dict(modes_input), capture_modes_endpoint=dict(modes_end),
    capture_owned_endpoint_phases=all_phases(owner_phases), continuation_policy_input_counts=dict(input_counts),
    execution_counts=dict(counts), prefix_counts=dict(prefix_counts), prefix_request_phase_counts=all_phases(prefix_phases),
    prefix_attempts=attempts, credit_starts=starts, prefix_samples_in_PPO_storage=0,
    completed_episodes=completed, episode_details=episode_stats, total_physics_ticks_including_prefix=counts['credited_physics_ticks']+prefix_counts['physics_ticks'],
    physics_scope='Actual task simulation intervals; excludes reset/settling. Partial last episode is not a task terminal.',
    all16_updates_finite_nonzero_gradient_actor_change=True, all16_rollouts_finite_and_original_gaussian_samples_match=True,
    observation_shape_each_rollout=[128,1,389], action_shape_each_rollout=[128,1,12],
    all16_actual_likelihood_logs_20_minibatches_each_sample_used5=True, raw_logp_max_CPU_error=maximum_logp_error,
    Adam_parameter_state_count=12, Adam_each_parameter_step_delta=320,
    actor_parameter_sha256_before=manifest['actor_parameter_sha256_before'], actor_parameter_sha256_after=manifest['actor_parameter_sha256_after'],
    actor_changed_parameter_tensors=sum(not torch.equal(v,target['actor_state_dict'][k]) for k,v in source['actor_state_dict'].items()),
    critic_changed_parameter_tensors=sum(not torch.equal(v,target['critic_state_dict'][k]) for k,v in source['critic_state_dict'].items()),
    normalization=cm['normalization'], optimizer_learning_rate=cm['optimizer_learning_rate'], per_rollout_counts=per_rollout,
    limitations=['Checkpoint-prefix suffix training is not full-P01 PPO success.',
        'Original runtime recorded native audit flags are trusted; updated production class was not imported or used to reconstruct them.',
        'Assist ownership is FL hip/knee only, not pure-policy FL capture.',
        'Input and endpoint truth are distinct sampling instants.'])
assert result['total_physics_ticks_including_prefix']==manifest['telemetry']['core']['physical_core_including_prefix']['physics_ticks']
print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
