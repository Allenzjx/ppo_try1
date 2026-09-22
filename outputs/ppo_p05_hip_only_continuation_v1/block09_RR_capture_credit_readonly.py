"""CPU-only arithmetic over two sealed rollouts; no model, update, or physics."""
import hashlib
import itertools
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'runs/ppo_p05_hip_only_continuation_v1/train/20260922T1428131775302Z_g336b7c56d2f0_23b26a4c40a24a178d190def03381413'
GAMMA, LAMBDA = .9985, .99
SELECTIONS = {1648: (215369, 215404), 1655: (216225, 216288)}


def summary(values):
    values = [float(v) for v in values]
    return dict(count=len(values), minimum=min(values), maximum=max(values),
                mean=sum(values)/len(values), total=sum(values),
                positive=sum(v > 0 for v in values), negative=sum(v < 0 for v in values))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def group(rows):
    return {**{k: summary(r[k] for r in rows) for k in (
        'raw_gae', 'actual_standardized_advantage', 'reward', 'potential_shaping',
        'phi_delta', 'RR_gap_delta_mm')},
        'samples': len(rows), 'actual_PPO_exposures': sum(r['PPO']['exposures'] for r in rows),
        'clipped_exposures': sum(r['PPO']['clipped_exposures'] for r in rows),
        'last_observed_ratio_above_one': sum(r['PPO']['last_observed_ratio'] > 1 for r in rows),
        'all_done_false': all(not r['done'] for r in rows)}


def main():
    torch.set_num_threads(1)
    selected = set(itertools.chain.from_iterable(range(a-1, b+1) for a,b in SELECTIONS.values()))
    audit = {}
    with (RUN / 'residual_and_projection_audit.jsonl').open(encoding='utf-8') as stream:
        for index, line in enumerate(itertools.islice(stream, max(selected)-214400)):
            expected = 214401 + index
            if expected in selected:
                row = json.loads(line)
                assert row['global_policy_decision'] == expected
                audit[expected] = row
    updates = {r['ppo_update']: r for r in map(json.loads, (RUN/'optimizer_updates.jsonl').read_text().splitlines())}
    gae_audits = {r['ppo_update_intended']: r for r in map(json.loads, (RUN/'advantage_audit.jsonl').read_text().splitlines())}
    result = {'scope': 'readonly_100_actual_decisions_two_sealed_rollouts_no_model_or_optimizer',
              'run': str(RUN), 'gamma': GAMMA, 'lambda': LAMBDA, 'updates': {}}
    for update, (first, last) in SELECTIONS.items():
        path = RUN / f'rollouts/rollout_{update:06d}.pt'
        lp_path = RUN / f'rollouts/update_{update:06d}_likelihood.json'
        data = torch.load(path, map_location='cpu', weights_only=False)
        likelihood = json.loads(lp_path.read_text())
        ga = gae_audits[update]
        start = ga['first_global_policy_decision']
        values = data['values'][:,0,0]
        raw = (data['returns']-data['values'])[:,0,0]
        normalized = data['advantages'][:,0,0]
        assert ga['stored_advantage_semantics'] == 'official_whole_rollout_standardized_GAE'
        normalize_error = float((normalized-(raw-raw.mean())/(raw.std()+1e-8)).abs().max())
        recurrence = raw[:-1]-(data['rewards'][:-1,0,0] + GAMMA*values[1:]-values[:-1] + GAMMA*LAMBDA*raw[1:])
        assert not bool(data['dones'].any())
        uses = {i: [] for i in range(128)}
        for mb in likelihood['minibatches']:
            for j, indices in enumerate(mb['rollout_flat_indices']):
                assert len(indices) == 1
                i = indices[0]
                assert mb['actual_advantage'][j] == float(normalized[i])
                assert mb['old_log_probability'][j] == float(data['actions_log_prob'][i,0,0])
                uses[i].append({'minibatch': mb['minibatch_index'], 'ratio': mb['ratio'][j],
                                'clipped': mb['clipped_branch_strictly_active'][j]})
        assert set(map(len, uses.values())) == {5}
        rows = []
        for decision in range(first, last+1):
            i = decision-start
            row = audit[decision]
            prev = audit[decision-1]['applied_audit']['semantic_task']['physical_evaluator']['current_legs']['RR']
            task = row['applied_audit']['semantic_task']
            ev = task['physical_evaluator']; rr = ev['current_legs']['RR']
            reward = row['applied_audit']['reward']
            for key, target in (('actions', row['raw_policy_action_full12']), ('rewards', [row['reward']]),
                                ('actions_log_prob', [row['old_log_probability']]), ('dones', [row['terminal']])):
                assert torch.equal(data[key][i,0], torch.tensor(target, dtype=data[key].dtype))
            assert abs(reward['potential_shaping']-5*(GAMMA*reward['potential_after']-reward['potential_before'])) < 1e-12
            assert abs(reward['total']-(reward['potential_shaping']-.02*reward['elapsed_physics_s'])) < 1e-12
            assert all(v == 0 for k,v in reward['families'].items() if k != 'task_progress')
            exposure = uses[i]
            rows.append(dict(global_decision=decision, rollout_index=i, tick=row['applied_audit']['physics_tick'],
                request_phase=row['applied_audit']['phase_id'], end_phase=task['stage_id'],
                reward=reward['total'], reward_families=reward['families'], terminal_event=reward['terminal_event'],
                time_cost=-.02*reward['elapsed_physics_s'], phi_before=reward['potential_before'],
                phi_after=reward['potential_after'], phi_delta=reward['potential_after']-reward['potential_before'],
                potential_shaping=reward['potential_shaping'], done=bool(data['dones'][i,0,0]),
                terminal_bootstrap_allowed=reward['terminal_bootstrap_allowed'],
                old_value=float(values[i]), stored_return=float(data['returns'][i,0,0]),
                raw_gae=float(raw[i]), actual_standardized_advantage=float(normalized[i]),
                RR_contact_surface=rr['contact_surface'], RR_qualified_TOP=rr['top_surface_contact'],
                RR_placed_history=ev['history']['placed']['RR'], RR_currently_usable=task['rr_placed_currently_usable'],
                RR_gap_mm=1000*rr['clearance_m'], RR_front_mm=1000*rr['front_distance_m'],
                RR_gap_delta_mm=1000*(rr['clearance_m']-prev['clearance_m']),
                event_ticks=ev['history']['event_ticks'],
                PPO=dict(exposures=len(exposure), clipped_exposures=sum(u['clipped'] for u in exposure),
                         first_observed_ratio=exposure[0]['ratio'], last_observed_ratio=exposure[-1]['ratio'],
                         last_ratio_measurement='before_sample_final_minibatch_optimizer_step_not_final_policy')))
        groups = {'all_selected': group(rows)}
        if update == 1655:
            bounds = {'pre_cross': (216225,216237), 'cross_through_capture': (216238,216244),
                      'P10_P11_early_P12_current_TOP': (216245,216252), 'after_initial_TOP_loss': (216253,216288)}
            groups.update({name: group([r for r in rows if a <= r['global_decision'] <= b]) for name,(a,b) in bounds.items()})
        else:
            for direction in ('descent','ascent'):
                chosen = [r for r in rows if (r['RR_gap_delta_mm'] < 0 if direction == 'descent' else r['RR_gap_delta_mm'] > 0)]
                groups[direction] = group(chosen)
        result['updates'][str(update)] = dict(
            hashes={'rollout': digest(path), 'actual_likelihood': digest(lp_path)},
            source_runtime_commit=data['runtime_contract']['source_git_commit'],
            selected_range=[first,last], selected_count=len(rows), full_rollout_GAE=ga['overall'],
            normalization_max_abs_error=normalize_error, GAE_interior_recurrence_max_abs_error=float(recurrence.abs().max()),
            tail_bootstrap=ga['tail_bootstrap'], transitions=ga['ordinary_phase_change_samples'],
            actual_update=updates[update], groups=groups, rows=rows)
    for item in result['updates'].values():
        item['event_ticks_at_selection_end'] = item['rows'][-1]['event_ticks']
        item['verified_all_selected_quality_families_zero'] = True
        item['verified_all_selected_terminal_event_zero'] = True
        item['verified_all_selected_bootstrap_allowed'] = True
        item['ratio_measurement'] = 'before_sample_final_minibatch_optimizer_step_not_final_policy'
        for row in item['rows']:
            for key in ('event_ticks', 'reward_families', 'terminal_event', 'terminal_bootstrap_allowed'):
                del row[key]
            del row['PPO']['last_ratio_measurement']
        item['row_columns'] = list(item['rows'][0])
        item['rows'] = [[row[key] for key in item['row_columns']] for row in item['rows']]
    print(json.dumps(result, separators=(',', ':'), allow_nan=False))


if __name__ == '__main__':
    main()
