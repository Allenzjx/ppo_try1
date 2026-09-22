"""Bounded sealed block03 counts: CPU/stdout only, no runtime replay or writes."""
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import torch

torch.set_num_threads(1)
ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'runs/ppo_p05_hip_only_continuation_v1/train/20260922T0705260953371Z_ga802b24d78df_76c646571d794b4e96a82f69956928ff'
CP = ROOT / 'outputs/ppo_p05_hip_only_continuation_v1/checkpoints/history/checkpoint_step_000205824.pt'
PHASES = [f'P{i:02}' for i in range(1, 14)]
LEGS = ['FL', 'FR', 'RL', 'RR']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def lines(name):
    with (RUN / name).open(encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def finite(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


manifest = read(RUN / 'training_manifest.json')
cm = read(CP.with_name(CP.stem + '_manifest.json'))
assert manifest['lifecycle'] == 'SUCCEEDED' and manifest['stage'] == 'full_episode'
assert manifest['actual_policy_decisions'] == 2048
assert not manifest['phase_suffix_curriculum_implemented']
assert not (RUN / 'prefix_evidence.jsonl').exists()
request, endpoint, totals = Counter(), Counter(), Counter()
events_end = {leg: Counter() for leg in LEGS}
top_input_known = Counter()
episodes, compact = [], []
previous_ev = None
for index, row in enumerate(lines('residual_and_projection_audit.jsonl')):
    a = row['applied_audit']; task = a['semantic_task']; ev = task['physical_evaluator']
    native = a['actuator_target_effect_audit']; hist = ev['history']
    assert row['global_policy_decision'] == 203777 + index
    if a['decision_count'] == 1:
        previous_ev = None
        episodes.append(dict(episode_index=len(episodes), policy_decisions=0, physics_ticks=0,
                             request_phases=Counter()))
        assert a['phase_id'] == 'P01'
    stat = episodes[-1]
    stat['policy_decisions'] += 1; stat['physics_ticks'] += a['physics_ticks']
    stat['request_phases'][a['phase_id']] += 1
    stat.update(last_phase=a['end_phase_id'], duration_s=a['sim_time_s'],
                terminal=bool(row['terminal']), termination_reason=a['termination_reason'],
                full_task_success=a['full_task_success'], event_ticks=hist['event_ticks'],
                placed_history=hist['placed'])
    request[a['phase_id']] += 1; endpoint[a['end_phase_id']] += 1
    if previous_ev is None:
        totals['input_TOP_unknown_natural_reset'] += 1
    else:
        totals['input_TOP_known_previous_endpoint'] += 1
        for leg in LEGS:
            top_input_known[leg] += bool(previous_ev['current_legs'][leg]['top_contact'])
    previous_ev = ev
    for leg in LEGS:
        current = ev['current_legs'][leg]
        events_end[leg]['lift_feature_equivalent'] += bool(current.get('current_lift_valid') if leg == 'RR' else hist['active_lift'][leg])
        events_end[leg]['crossed_history'] += bool(hist['front_edge_crossed'][leg])
        events_end[leg]['placed_history'] += bool(hist['placed'][leg])
        events_end[leg]['TOP'] += bool(current['top_contact'])
    assert native['verified'] and native['actual_mapping_matches_dispatch']
    assert native['phase_mask_full12'] == [1] * 12
    assert a['no_in_episode_state_writes_verified']
    assert a['actuator_target_effect_audit_summary']['all_ticks_verified']
    assert not a.get('prefix_teacher_data_in_ppo_storage', False)
    assert not a.get('prefix_checkpoint_policy_data_in_ppo_storage', False)
    assert row['raw_policy_action_full12'] == row['policy_request']['selected_raw_full12'] == native['raw_policy_action_full12']
    assert row['policy_request']['sampling_draws'] == 1 and row['policy_request']['extra_random_draws'] == 0
    totals['physics_ticks'] += a['physics_ticks']
    totals['verified_native_ticks'] += a['actuator_target_effect_audit_summary']['verified_tick_count']
    totals['assist_owned_endpoints'] += bool(native['capture_assist_evidence']['owner_indices'])
    totals['pending_endpoints'] += bool(task['fl_capture_pending'])
    totals['scheduler_advanced_pending_endpoints'] += bool(task['capture_continuation']['scheduler_advanced_pending'])
    compact.append(dict(phase=a['phase_id'], raw=row['raw_policy_action_full12'],
        mean=row['old_distribution_mean_full12'], std=row['old_distribution_std_full12'],
        logp=row['old_log_probability'], reward=row['reward'], terminal=row['terminal'],
        qRR=row['policy_request']['current_RR_qualification'], pRR=row['policy_request']['RR_placed_history']))
assert len(compact) == 2048
input_events = {leg: Counter() for leg in LEGS}
max_logp_error = 0.
for update in range(1558, 1574):
    data = torch.load(RUN / f'rollouts/rollout_{update:06}.pt', map_location='cpu', weights_only=False)
    obs = data['observations']['policy']; acts = data['actions']; means, stds = data['distribution_params']
    assert obs.shape == (128, 1, 389) and acts.shape == (128, 1, 12)
    assert finite(data) and torch.equal(obs, data['observations']['critic'])
    segment = compact[(update-1558)*128:(update-1557)*128]
    for tensor, key in [(acts, 'raw'), (means, 'mean'), (stds, 'std'), (data['actions_log_prob'], 'logp'),
                        (data['rewards'], 'reward'), (data['dones'], 'terminal')]:
        assert torch.equal(tensor, torch.tensor([r[key] for r in segment], dtype=tensor.dtype).reshape(tensor.shape))
    assert [PHASES[i] for i in obs[:, 0, :13].argmax(-1).tolist()] == [r['phase'] for r in segment]
    assert [bool(v) for v in obs[:, 0, 149]] == [r['qRR'] for r in segment]
    assert [bool(v) for v in obs[:, 0, 157]] == [r['pRR'] for r in segment]
    for j, leg in enumerate(LEGS):
        for name, base in [('lift_observation_bit', 146), ('crossed_history', 150), ('placed_history', 154)]:
            input_events[leg][name] += int((obs[:, 0, base+j] == 1).sum())
    max_logp_error = max(max_logp_error, (torch.distributions.Normal(means, stds).log_prob(acts).sum(-1).unsqueeze(-1) - data['actions_log_prob']).abs().max().item())
    likelihood = read(RUN / f'rollouts/update_{update:06}_likelihood.json')
    use = Counter(i for batch in likelihood['minibatches'] for ids in batch['rollout_flat_indices'] for i in ids)
    assert len(likelihood['minibatches']) == 20 and use == Counter({i: 5 for i in range(128)})
updates = list(lines('optimizer_updates.jsonl'))
assert [u['ppo_update'] for u in updates] == list(range(1558, 1574))
assert all(finite(u) and u['actor_parameters_changed'] and u['finite_nonzero_gradient_observed'] and u['optimizer_steps'] == 20 for u in updates)
assert all(a['actor_parameter_sha256_after'] == b['actor_parameter_sha256_before'] for a,b in zip(updates, updates[1:]))
with CP.open('rb') as stream:
    cp_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
assert cp_hash == cm['checkpoint_sha256']
assert [e['policy_decisions'] for e in episodes] == [1721, 235, 92]
assert totals['physics_ticks'] == manifest['telemetry']['core']['physics_ticks']
for stat in episodes:
    stat['request_phases'] = {p: stat['request_phases'][p] for p in PHASES}
result = dict(schema='wlr50_clean.p05_capture_block03_training_audit.v1',
    run=RUN.as_posix(), runtime='a802b24d78df5f1b8f0caf914ce7a8a68a98db8e', lifecycle=manifest['lifecycle'],
    new_counts=dict(policy_decisions=2048, ppo_updates=16, optimizer_steps=320),
    lifetime_counts={k:cm[k] for k in ['global_policy_decisions','ppo_updates','optimizer_steps']},
    P05_branch_cumulative=dict(policy_decisions=6144,ppo_updates=48,optimizer_steps=960),
    feedback_v2_branch_learned=dict(policy_decisions=2048,ppo_updates=16,optimizer_steps=320),
    checkpoint=CP.as_posix(),checkpoint_sha256=cp_hash, recorded_save_load_round_trip=cm['save_load_round_trip'],
    request_phase_counts={p:request[p] for p in PHASES},endpoint_phase_counts={p:endpoint[p] for p in PHASES},
    physical_input_counts=input_events, physical_endpoint_counts=events_end,
    exact_TOP_input_known_counts=dict(top_input_known),execution_counts=dict(totals),episodes=episodes,
    prefix_policy_decisions=0, optimizer_update_range=[1558,1573], all16_updates_finite_nonzero_gradient_actor_changed=True,
    all16_rollouts_finite_original_raw_samples_match=True,each_sample_used_five_times_in20_minibatches=True,
    independent_CPU_raw_logp_max_error=max_logp_error,
    normalization=cm['normalization'],learning_rate=cm['optimizer_learning_rate'],
    limitations=['Pure CPU counts/arithmetic only; no physics replay or policy inference.',
        'Lift input bits are the saved schema: RR means current qualification; other legs use active-lift history.',
        'Exact TOP not an explicit raw389 classifier: three initial P01 reset inputs unlogged here, not filled as zero.',
        'PPO + FL capture assist + inherited limited AUX, not pure policy; no new AUX performed.',
        'No complete-task success in this block; last partial episode is not a terminal.'])
print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
