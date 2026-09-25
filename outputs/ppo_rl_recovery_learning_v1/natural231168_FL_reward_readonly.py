"""Read the one sealed natural CP230144->231168 run; stdlib only; stdout report."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'runs/ppo_rr_rl_timing_policy_learning_v1/train/20260924T1257554297942Z_g59e868f3e223_8a6cf8fd89554916abdc8a6b9f6fe1ac'
GAMMA, WEIGHT, TIME_COST = .9985, 5., .02


def snapshot_rows(path):
    # Use actual open-file extent, not Windows directory/stat metadata.
    with path.open('rb') as stream:
        stream.seek(0, 2)
        boundary = stream.tell()
        stream.seek(0)
        data = stream.read(boundary)
    assert data.endswith(b'\n'), path
    return [json.loads(line) for line in data.splitlines()], {
        'path': str(path), 'open_seek_tell_snapshot_bytes': boundary,
        'sha256': hashlib.sha256(data).hexdigest(), 'all_snapshot_rows_complete': True}


def audit(row):
    return row['applied_audit']


def reward(row):
    return audit(row)['reward_breakdown']


def aggregate(rows):
    parts = [reward(row) for row in rows]
    p0, pn = parts[0]['potential_before'], parts[-1]['potential_after']
    continuity = [abs(left['potential_after'] - right['potential_before'])
                  for left, right in zip(parts, parts[1:])]
    shaping = sum(part['potential_shaping'] for part in parts)
    decay = -WEIGHT * (1. - GAMMA) * sum(part['potential_after'] for part in parts)
    discounted = sum(GAMMA**i * part['potential_shaping'] for i, part in enumerate(parts))
    return {
        'decisions': len(rows), 'first_global': rows[0]['global_policy_decision'],
        'last_global': rows[-1]['global_policy_decision'],
        'first_start_tick': audit(rows[0])['physics_tick'] - audit(rows[0])['physics_ticks'],
        'last_end_tick': audit(rows[-1])['physics_tick'],
        'elapsed_s': sum(part['elapsed_physics_s'] for part in parts),
        'potential_before': p0, 'potential_after': pn, 'delta_potential': pn - p0,
        'potential_shaping': shaping, 'endpoint_delta_contribution': WEIGHT * (pn - p0),
        'discount_decay_contribution': decay,
        'undiscounted_telescope_error': shaping - WEIGHT * (pn - p0) - decay,
        'discounted_potential_shaping': discounted,
        'discounted_telescope_error': discounted - WEIGHT * (GAMMA**len(parts) * pn - p0),
        'max_potential_continuity_error': max(continuity, default=0.),
        'max_formula_error': max(abs(part['potential_shaping'] - WEIGHT *
                                     (GAMMA * part['potential_after'] - part['potential_before']))
                                 for part in parts),
        'time_cost': -TIME_COST * sum(part['elapsed_physics_s'] for part in parts),
        'terminal_event': sum(part['terminal_event'] for part in parts),
        'counterroll_cost': -sum(part.get('cooperative_counterroll_cost', 0.) for part in parts),
        'families': {key: sum(part['families'][key] for part in parts)
                     for key in parts[0]['families']},
        'total': sum(part['total'] for part in parts),
        'stored_float32_reward_total': sum(row['reward'] for row in rows),
        'discounted_total': sum(GAMMA**i * part['total'] for i, part in enumerate(parts)),
        'positive_reward_decisions': sum(part['total'] > 0 for part in parts),
        'negative_reward_decisions': sum(part['total'] < 0 for part in parts),
        'terminal_count': sum(row['terminal'] for row in rows),
        'rr_replacement_scope_unchanged': all(
            part['rr_retention_reward_only'][side]['scope'] == 'unchanged'
            for part in parts for side in ('before', 'after')),
    }


rows, source = snapshot_rows(RUN / 'residual_and_projection_audit.jsonl')
advantages, adv_source = snapshot_rows(RUN / 'advantage_audit.jsonl')
episodes = [[]]
for row in rows:
    if episodes[-1] and audit(row)['physics_tick'] <= audit(episodes[-1][-1])['physics_tick']:
        episodes.append([])
    episodes[-1].append(row)
assert [len(episode) for episode in episodes] == [243, 781]
episode = episodes[1]
p05 = [row for row in episode if audit(row)['phase_id'] == 'P05']
assert len(p05) == 530
final = audit(episode[-1])
history = final['semantic_task']['physical_evaluator']['history']
events = [event for event in history['lift_attempt_events'] if event['leg'] == 'FL'
          and event['event'] in ('qualified_measured_upward_lift', 'qualification_revoked_ground_before_cross')]
assert len(events) == 32


def enclosing_index(tick):
    indices = [i for i, row in enumerate(episode)
               if audit(row)['physics_tick'] - audit(row)['physics_ticks'] < tick <= audit(row)['physics_tick']]
    assert len(indices) == 1, (tick, indices)
    return indices[0]


def endpoint(i):
    row = episode[i]
    a, r = audit(row), reward(row)
    leg = a['semantic_task']['physical_evaluator']['current_legs']['FL']
    return {key: value for key, value in {
        'global_policy_decision': row['global_policy_decision'], 'end_tick': a['physics_tick'],
        'total': r['total'], 'potential_before': r['potential_before'],
        'potential_after': r['potential_after'], 'delta_potential': r['potential_after'] - r['potential_before'],
        'potential_shaping': r['potential_shaping'], 'terminal_event': r['terminal_event'],
        'old_value': row['old_value'], 'FL_active_lift': a['semantic_task']['physical_evaluator']['history']['active_lift']['FL'],
        'FL_ground_contact': leg['ground_contact'], 'FL_air': leg['air'],
        'FL_top_contact': leg['top_contact'], 'FL_front_m': leg['front_distance_m'],
        'FL_clearance_m': leg['clearance_m']}.items()}


attempts = []
previous_revoke_index = None
for number, (qualified, revoked) in enumerate(zip(events[::2], events[1::2]), 1):
    assert qualified['event'] == 'qualified_measured_upward_lift'
    assert revoked['event'] == 'qualification_revoked_ground_before_cross'
    qi, ri = enclosing_index(qualified['physics_tick']), enclosing_index(revoked['physics_tick'])
    ground_cycle = aggregate(episode[previous_revoke_index + 1:ri + 1]) if previous_revoke_index is not None else None
    attempts.append({'attempt': number, 'qualified_event': qualified, 'revoked_event': revoked,
                     'qualification_enclosing_decision': endpoint(qi),
                     'revocation_enclosing_decision': endpoint(ri),
                     'qualification_through_revoke_inclusive': aggregate(episode[qi:ri + 1]),
                     'previous_revoke_endpoint_to_this_revoke_endpoint': ground_cycle})
    previous_revoke_index = ri
qrows = [episode[enclosing_index(event['physics_tick'])] for event in events[::2]]
rrows = [episode[enclosing_index(event['physics_tick'])] for event in events[1::2]]
event_stats = {
    'qualification_decision_total_sum': sum(reward(row)['total'] for row in qrows),
    'qualification_decision_delta_potential_sum': sum(reward(row)['potential_after'] - reward(row)['potential_before'] for row in qrows),
    'revocation_decision_total_sum': sum(reward(row)['total'] for row in rrows),
    'revocation_decision_delta_potential_sum': sum(reward(row)['potential_after'] - reward(row)['potential_before'] for row in rrows),
    'revocation_decisions_negative': sum(reward(row)['total'] < 0 for row in rrows),
    'revocation_decisions_potential_drop': sum(reward(row)['potential_after'] < reward(row)['potential_before'] for row in rrows),
    'qualified_to_revoked_windows_positive': sum(a['qualification_through_revoke_inclusive']['total'] > 0 for a in attempts),
    'ground_endpoint_cycles_positive': sum(a['previous_revoke_endpoint_to_this_revoke_endpoint']['total'] > 0 for a in attempts[1:]),
}
critic = [{key: record[key] for key in ('ppo_update_intended', 'first_global_policy_decision',
                                     'last_global_policy_decision', 'gamma', 'lambda', 'tail_bootstrap')}
          | {'P05': record['by_request_phase']['P05']} for record in advantages]
result = {
    'schema': 'bounded_readonly_natural231168_FL_repeat_reward_v1', 'run_dir': str(RUN),
    'evidence': [source, adv_source],
    'evidence_scope': '1024 complete decision JSON endpoints; embedded event ticks and existing reward per-physics sample audits; no separate full 120Hz physical trajectory, model, rollout tensors, or new simulation read',
    'production_edits': False, 'reward_or_advantage_edits': False,
    'constants': {'gamma': GAMMA, 'potential_weight': WEIGHT, 'time_cost_per_s': TIME_COST},
    'episode_lengths': [len(value) for value in episodes],
    'second_episode': aggregate(episode), 'P05_request_decisions': aggregate(p05),
    'first_qualification_through_last_revocation': aggregate(episode[enclosing_index(events[0]['physics_tick']):enclosing_index(events[-1]['physics_tick']) + 1]),
    'repeated_ground_endpoint_cycles_2_to_16': aggregate(episode[enclosing_index(events[1]['physics_tick']) + 1:enclosing_index(events[-1]['physics_tick']) + 1]),
    'final_state': {'tick': final['physics_tick'], 'time_s': final['sim_time_s'], 'stage': final['end_phase_id'],
                    'terminal': episode[-1]['terminal'], 'termination_reason': final['termination_reason'],
                    'active_lift': history['active_lift'], 'crossed': history['front_edge_crossed'],
                    'placed': history['placed'], 'FL': final['semantic_task']['physical_evaluator']['current_legs']['FL']},
    'event_stats': event_stats, 'attempts': attempts, 'existing_preupdate_advantage_audit': critic,
}
print(json.dumps(result, indent=2))
