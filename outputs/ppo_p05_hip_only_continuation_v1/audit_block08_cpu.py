"""Sealed block08 only; run after root confirms completion. No polling/physics/fit."""
from collections import Counter
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import torch
import yaml
from wlr50_clean.ppo.semantic_training import state_hash
from wlr50_clean.ppo.semantic_supervisor import TaskStageSupervisor, _current_rr_receiver_preparation_retired

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
RUN = ROOT / 'runs/ppo_p05_hip_only_continuation_v1/train/20260922T1301289433343Z_g5fd88852bf20_337c83c1b0214e02b62aa98b579e2812'
SOURCE = OUT / 'checkpoints/history/checkpoint_aux_meanfront_step_000213376_v3.pt'
SOURCE_SHA = '945b05e2763396c0f83c582eb85d34d778e6fb8c74b041177c3bdea9f7077cd3'
PHASES = [f'P{i:02}' for i in range(1, 14)]
COUNTERS = ('global_policy_decisions', 'ppo_updates', 'optimizer_steps')


def load_helpers():
    spec = importlib.util.spec_from_file_location('bounded_first_carry_helpers', OUT / 'audit_block08_first_carry_cpu.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def main():
    h = load_helpers()
    manifest = h.read(RUN / 'training_manifest.json')
    assert manifest['lifecycle'] == 'SUCCEEDED', 'not sealed; stop without polling'
    assert manifest['stage'] == 'phase_suffix' and manifest['phase_suffix_curriculum_implemented']
    actual = manifest['actual_policy_decisions']
    assert actual == 1024 and manifest['unconsumed_requested_policy_decisions'] == 0
    assert not torch.cuda.is_available(); torch.set_num_threads(1)
    target_path = Path(manifest['checkpoints'][-1]['checkpoint'])
    sm, tm = (h.read(p.with_name(p.stem + '_manifest.json')) for p in (SOURCE, target_path))
    assert h.sha(SOURCE) == sm['checkpoint_sha256'] == SOURCE_SHA
    assert h.sha(target_path) == tm['checkpoint_sha256'] and tm['source_run'] == str(RUN) and tm['save_load_round_trip']
    assert {key: tm[key] for key in COUNTERS} == dict(global_policy_decisions=214400, ppo_updates=1640, optimizer_steps=32800)
    assert [tm[key] - sm[key] for key in COUNTERS] == [1024, 8, 160]
    task_path = ROOT / 'configs/ppo_p05_hip_only_continuation_v1/stage_task_spec.yaml'
    for path in (task_path, ROOT / 'src/wlr50_clean/ppo/semantic_supervisor.py'):
        assert h.sha(path) == tm['runtime_contract']['files'][path.relative_to(ROOT).as_posix()]
    spec = yaml.safe_load(task_path.read_text())
    current = object.__new__(TaskStageSupervisor); current.spec = spec
    old = object.__new__(TaskStageSupervisor); old.spec = deepcopy(spec); old.spec.pop('rr_postcross_workspace_semantics')

    def retirement(ev):
        predicate = bool(_current_rr_receiver_preparation_retired(spec, 'RR', ev))
        consumed = predicate and not ev['history']['placed']['RR']
        delta = current.physical_potential(ev) - old.physical_potential(ev)
        assert delta >= -1e-12 and (consumed or abs(delta) < 1e-12)
        return dict(predicate_active=predicate, workspace_share_consumed=consumed, effective_difference=abs(delta) > 1e-12, delta=delta)

    prefix_counts, prefix_phases, prefixes, attempts = Counter(), Counter(), [], []
    local = Counter()
    for row in h.lines(RUN / 'prefix_evidence.jsonl'):
        assert row['policy_credit'] is False
        if row['kind'] == 'checkpoint_prefix_decision':
            prefix_counts['decisions'] += 1; local['decisions'] += 1
            prefix_counts['ticks'] += row['physics_ticks']; local['ticks'] += row['physics_ticks']
            prefix_phases[row['phase_id']] += 1
            assert row['actuator_target_effect_audit_summary']['all_ticks_verified'] and row['no_in_episode_state_writes_verified']
            native = row['actuator_target_effect_audit']
            prefix_counts['assist_owned_endpoints'] += bool(native['capture_assist_evidence']['owner_indices'])
        elif row['kind'] == 'checkpoint_prefix_result':
            assert row['accepted'] is True
            attempts.append({key: row[key] for key in ('prefix_decisions', 'prefix_physics_ticks', 'accepted', 'miss')})
        elif row['kind'] == 'policy_credit_start':
            start = row['start']; prov = start['prefix_policy_provenance']
            assert start['actual_phase'] == 'P04' and prov['checkpoint_sha256'] == SOURCE_SHA
            assert prov['frozen_actor_parameter_sha256'] == sm['actor_parameter_sha256']
            assert prov['frozen_for_entire_training_block'] and prov['independent_parameter_and_buffer_storage_verified']
            assert prov['source_policy_contract'] == prov['effective_policy_contract'] == tm['policy_contract']
            assert prov['source_runtime_content_sha256'] == prov['effective_runtime_content_sha256']
            prefixes.append({'start': start, 'counts': dict(local)}); local = Counter()
    assert len(prefixes) == len(attempts) == 2
    records, episodes = [], []
    requested, endpoints, counts, rr_inputs, rr_ends, ri, re = (Counter() for _ in range(7))
    previous_ev = None
    for index, row in enumerate(h.lines(RUN / 'residual_and_projection_audit.jsonl')):
        assert index < actual and row['global_policy_decision'] == 213377 + index
        a, p = row['applied_audit'], row['policy_request']
        task = a['semantic_task']; ev = task['physical_evaluator']; native = a['actuator_target_effect_audit']
        evidence = native['capture_assist_evidence']; state = evidence['state_after']
        if a['decision_count'] == 1:
            assert not episodes or episodes[-1]['terminal']
            episodes.append({'episode_index': len(episodes), 'decisions': 0, 'physics_ticks': 0,
                'input_phases': Counter(), 'assist_modes': Counter(), 'assist_initialized_endpoints': 0,
                'assist_owned_endpoints': 0, 'first_FL_placed_endpoint': None})
            previous_ev = None
        ep = episodes[-1]; prefix = prefixes[ep['episode_index']]
        assert a['decision_count'] == ep['decisions'] + 1
        assert a['physical_core_decision_count_including_prefix'] == prefix['counts']['decisions'] + a['decision_count']
        assert not a['prefix_teacher_data_in_ppo_storage'] and not a['prefix_checkpoint_policy_data_in_ppo_storage']
        assert a['task_result_scope'] == 'checkpoint_policy_initialized_suffix'
        assert native['verified'] and native['actual_mapping_matches_dispatch'] and native['phase_mask_full12'] == [1] * 12
        assert a['no_in_episode_state_writes_verified'] and a['actuator_target_effect_audit_summary']['all_ticks_verified']
        assert len(a['actuator_target_effect_audit_ticks']) == a['physics_ticks'] and all(t['verified'] for t in a['actuator_target_effect_audit_ticks'])
        assert evidence['owner_indices'] in ([], [0, 1])
        assert evidence['candidate_before_assist_full12'][2:] == evidence['candidate_after_assist_full12'][2:]
        assert row['raw_policy_action_full12'] == p['selected_raw_full12'] == native['raw_policy_action_full12']
        assert row['old_distribution_mean_full12'] == p['conditional_mean_full12'] and row['old_distribution_std_full12'] == p['effective_sigma_full12']
        assert p['sampling_draws'] == 1 and p['extra_random_draws'] == 0
        ep['decisions'] += 1; ep['physics_ticks'] += a['physics_ticks']; ep['input_phases'][a['phase_id']] += 1
        ep['assist_modes'][state['mode_name']] += 1
        ep['assist_initialized_endpoints'] += bool(state['initialized']); ep['assist_owned_endpoints'] += bool(evidence['owner_indices'])
        if ev['history']['placed']['FL'] and ep['first_FL_placed_endpoint'] is None:
            ep['first_FL_placed_endpoint'] = {'global_decision': row['global_policy_decision'], 'endpoint_tick': a['physics_tick'],
                'physical_event_tick': ev['history']['event_ticks']['placed']['FL'], 'assist_state': state,
                'owners': evidence['owner_indices'], 'current_FL': ev['current_legs']['FL']}
        ep.update(last_global_decision=row['global_policy_decision'], physical_duration_s=a['sim_time_s'],
            last_phase=a['end_phase_id'], terminal=bool(row['terminal']), termination_reason=a['termination_reason'],
            full_task_success=bool(a['full_task_success']), placed_history=ev['history']['placed'],
            event_ticks=ev['history']['event_ticks'], final_RR=ev['current_legs']['RR'],
            prefix_decisions=prefix['counts']['decisions'], prefix_ticks=prefix['counts']['ticks'])
        requested[a['phase_id']] += 1; endpoints[a['end_phase_id']] += 1
        counts['credited_physics_ticks'] += a['physics_ticks']; counts['native_verified_ticks'] += len(a['actuator_target_effect_audit_ticks'])
        counts['assist_owned_endpoints'] += bool(evidence['owner_indices'])
        counts['assist_initialized_endpoints'] += bool(state['initialized'])
        for key, value in (('qualified_current', ev['current_legs']['RR']['current_lift_valid']),
            ('crossed_history', ev['history']['front_edge_crossed']['RR']), ('placed_history', ev['history']['placed']['RR']),
            ('TOP', ev['current_legs']['RR']['top_contact'])):
            rr_ends[key] += bool(value)
        rs = retirement(ev)
        for key in ('predicate_active', 'workspace_share_consumed', 'effective_difference'):
            re[key] += int(rs[key])
        previous_rs = retirement(previous_ev) if previous_ev is not None else None
        records.append({'row': row, 'retirement_input': previous_rs, 'episode': ep['episode_index']})
        assert abs(current.physical_potential(ev) - task['task_progress_potential']) < 1e-12
        previous_ev = ev
    assert len(records) == actual and len(episodes) == 2
    updates = list(h.lines(RUN / 'optimizer_updates.jsonl'))
    assert [u['ppo_update'] for u in updates] == list(range(1633, 1641))
    assert all(u['optimizer_steps'] == 20 and u['actor_parameters_changed'] and u['finite_nonzero_gradient_observed'] for u in updates)
    assert updates[0]['actor_parameter_sha256_before'] == sm['actor_parameter_sha256']
    assert updates[-1]['actor_parameter_sha256_after'] == tm['actor_parameter_sha256']
    assert all(a['actor_parameter_sha256_after'] == b['actor_parameter_sha256_before'] for a, b in zip(updates, updates[1:]))
    max_logp_error = 0.; per_rollout = []
    for block, update in enumerate(range(1633, 1641)):
        batch = torch.load(RUN / f'rollouts/rollout_{update:06}.pt', map_location='cpu', weights_only=False)
        obs = batch['observations']['policy']; actions = batch['actions']; means, stds = batch['distribution_params']
        assert obs.shape == (128, 1, 389) and actions.shape == (128, 1, 12) and torch.equal(obs, batch['observations']['critic'])
        assert batch['runtime_contract'] == tm['runtime_contract'] and batch['policy_contract'] == tm['policy_contract']
        assert batch['curriculum_epoch']['prefix_policy_provenance']['checkpoint_sha256'] == SOURCE_SHA
        segment = records[block * 128:(block + 1) * 128]; rows = [r['row'] for r in segment]
        for tensor, key in ((actions, 'raw_policy_action_full12'), (means, 'old_distribution_mean_full12'),
            (stds, 'old_distribution_std_full12'), (batch['actions_log_prob'], 'old_log_probability'),
            (batch['values'], 'old_value'), (batch['rewards'], 'reward'), (batch['dones'], 'terminal')):
            assert torch.isfinite(tensor).all()
            assert torch.equal(tensor, torch.tensor([r[key] for r in rows], dtype=tensor.dtype).reshape(tensor.shape))
        assert torch.isfinite(obs).all()
        assert [PHASES[i] for i in obs[:, 0, :13].argmax(-1)] == [r['applied_audit']['phase_id'] for r in rows]
        for start, end, key in ((372, 384, 'capture_assist_observed_features'), (384, 389, 'capture_continuation_observed_features')):
            assert torch.equal(obs[:, 0, start:end], torch.tensor([r['policy_request'][key] for r in rows], dtype=obs.dtype))
        assert torch.equal(obs[:, 0, 17], torch.tensor([r['applied_audit']['reward_breakdown']['potential_before'] for r in rows], dtype=obs.dtype))
        local = Counter()
        for i, record in enumerate(segment):
            for key, column in (('qualified_current', 149), ('crossed_history', 153), ('placed_history', 157)):
                rr_inputs[key] += int(obs[i, 0, column] == 1)
            if record['retirement_input'] is None:
                assert not bool(obs[i, 0, 149]) and not bool(obs[i, 0, 153])
                counts['first_input_retirement_proven_false_by_required_bits'] += 1
            else:
                for key in ('predicate_active', 'workspace_share_consumed', 'effective_difference'):
                    ri[key] += int(record['retirement_input'][key]); local[key] += int(record['retirement_input'][key])
        likelihood = h.read(RUN / f'rollouts/update_{update:06}_likelihood.json')
        uses = Counter(i for mini in likelihood['minibatches'] for indices in mini['rollout_flat_indices'] for i in indices)
        assert len(likelihood['minibatches']) == 20 and uses == Counter({i: 5 for i in range(128)})
        max_logp_error = max(max_logp_error, float((torch.distributions.Normal(means, stds).log_prob(actions).sum(-1).unsqueeze(-1) - batch['actions_log_prob']).abs().max()))
        per_rollout.append({'update': update, 'phases': dict(Counter(r['applied_audit']['phase_id'] for r in rows)), 'RR_retirement_inputs': dict(local)})
    completed = list(h.lines(RUN / 'completed_episodes.jsonl'))
    ended = [e for e in episodes if e['terminal']]
    assert len(completed) == len(ended)
    for stat, recorded in zip(ended, completed):
        assert stat['decisions'] == recorded['policy_decisions']
        assert stat['termination_reason'] == recorded['termination_reason']
        assert stat['full_task_success'] == recorded['full_task_success']
        assert abs(stat['physical_duration_s'] - recorded['duration_s']) < 1e-9
    assert sum(e['decisions'] for e in episodes) == 1024
    assert counts['credited_physics_ticks'] == counts['native_verified_ticks'] == manifest['telemetry']['core']['physics_ticks']
    assert counts['credited_physics_ticks'] + prefix_counts['ticks'] == manifest['telemetry']['core']['physical_core_including_prefix']['physics_ticks']
    source, target = (torch.load(p, map_location='cpu', weights_only=False) for p in (SOURCE, target_path))
    for payload, meta in ((source, sm), (target, tm)):
        assert h.parameter_sha(payload['actor_state_dict']) == meta['actor_parameter_sha256']
        assert h.parameter_sha(payload['critic_state_dict']) == meta['critic_parameter_sha256']
        assert state_hash(payload['optimizer_state_dict']) == meta['optimizer_state_sha256']
        assert all(payload['infos'][key] == value for key, value in meta.items() if key in payload['infos'])
    deltas = [float(target['optimizer_state_dict']['state'][key]['step'] - state['step']) for key, state in source['optimizer_state_dict']['state'].items()]
    assert len(deltas) == 12 and deltas == [160.] * 12
    preserved = [key for key in sm if key != 'resume_migration' and key.endswith(('_branch', '_migration'))]
    preserved += ['normalization', 'normalizer_state_sha256', 'runner_config', 'policy_contract', 'runtime_contract']
    assert all(sm[key] == tm[key] for key in preserved)
    ledger = tm['rr_postcross_workspace_branch']['front_rehearsal_auxiliary']
    assert len(ledger['events']) == 3 and ledger['accepted_auxiliary_updates_total'] == ledger['attempted_auxiliary_optimizer_steps_total'] == 96
    older = tm['task_conditioned_hip_wheel_branch']['auxiliary_mean_learning']
    assert older['accepted_auxiliary_updates_total'] == 7 and older['attempted_auxiliary_optimizer_steps_total'] == 8
    assert tm['optimizer_learning_rate'] == 1e-5
    branches = ('p05_capture_assist', 'capture_feedback_semantics', 'rr_postcross_workspace')
    for branch in branches:
        assert tm[branch + '_branch_counts'] == {key: tm[key] - value for key, value in tm[branch + '_branch']['counter_origin'].items()}
    for ep in episodes:
        ep['input_phases'] = {phase: ep['input_phases'][phase] for phase in PHASES}
        ep['credited_duration_s'] = ep['physics_ticks'] / 120
    result = {'schema': 'wlr50_clean.block08_sealed_training_audit.v1', 'result': 'PASS', 'run': str(RUN),
        'source_checkpoint': str(SOURCE), 'source_sha256': SOURCE_SHA, 'checkpoint': str(target_path),
        'checkpoint_sha256': tm['checkpoint_sha256'], 'lifecycle': manifest['lifecycle'],
        'new_counts': dict(policy_decisions=1024, PPO_updates=8, Adam_steps=160),
        'lifetime_counts': {key: tm[key] for key in COUNTERS}, 'planned': 1024, 'unconsumed': 0,
        'request_phase_counts': {phase: requested[phase] for phase in PHASES}, 'endpoint_phase_counts': dict(endpoints),
        'prefix_counts': dict(prefix_counts), 'prefix_phases': dict(prefix_phases), 'prefix_starts': prefixes,
        'prefix_attempts': attempts, 'prefix_PPO_credit': 0, 'execution_counts': dict(counts), 'episodes': episodes,
        'ended_episodes': len(ended), 'sampling_boundary_partial_episodes': len(episodes) - len(ended),
        'RR_input_counts': dict(rr_inputs), 'RR_endpoint_counts': dict(rr_ends),
        'RR_retirement_actual_input_counts': dict(ri), 'RR_retirement_actual_endpoint_counts': dict(re),
        'per_rollout': per_rollout, 'raw_storage_and_five_PPO_uses_verified': True,
        'CPU_Normal_logp_max_error': max_logp_error, 'Adam_step_deltas': deltas,
        'all8_updates_actor_changed_finite_gradient': True, 'Identity_and_LR_preserved': True,
        'full_front_AUX96_96_and_old7_8_exact_source_carry': True, 'AUX_added': 0,
        'counter_origins': {b: tm[b + '_branch']['counter_origin'] for b in branches},
        'branch_added_counts': {b: tm[b + '_branch_counts'] for b in branches},
        'RNG_source_hash': state_hash(sm['training_rng_state']), 'RNG_final_hash': state_hash(tm['training_rng_state']),
        'checkpoint_sidecar_embedded_equal': True, 'actual_official_save_reload_recorded': True,
        'physical_full_P01_PPO_success_claimed': False,
        'limitations': ['Prefix-initialized suffix data is not full P01 PPO success.',
            'Compact tick records verify native execution but do not separately persist per-tick mask/assist owner.',
            'All decision endpoints retain initialized=false only if the actual counts establish that; the latched initialized flag precludes hidden assist activation between them.',
            'RR retirement predicate, consumed workspace share and effective potential difference are counted separately.']}
    (OUT / 'block08_training_audit.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    rows = '\n'.join(f"| {e['episode_index']} | {e['decisions']} | {e['physical_duration_s']:.6f} | {e['credited_duration_s']:.6f} | {'ended' if e['terminal'] else 'partial; done=false'} | {e['last_phase']} | {e['termination_reason'] or 'sampling boundary'} |" for e in episodes)
    md = f'''# Block08 sealed genuine PPO audit

**Verified real 1024 decisions /8 PPO /160 Adam**, cumulative214400/1640/32800. Training lifecycle SUCCEEDED is not task success. Final checkpoint SHA `{tm['checkpoint_sha256']}`; actual source is mean-head AUX3 `{SOURCE_SHA}`. Planned1024, unconsumed0.

- Credited input counts: {dict(requested)}; all unlisted P01–P13 phases have0. Full counts are in JSON.
- Two frozen source-checkpoint prefixes: {prefix_counts['decisions']} actions/{prefix_counts['ticks']} ticks, all policy_credit=false and excluded from PPO storage. Learner physics {counts['credited_physics_ticks']} verified ticks.
- Learner assist-owned endpoints {counts['assist_owned_endpoints']}; initialized endpoints {counts['assist_initialized_endpoints']}. Per-episode modes and first actual FL placement/event tick are retained in JSON. No assist or pure-policy label is inferred merely from a stage name.
- RR actual learner input counts {dict(rr_inputs)}; endpoint counts {dict(rr_ends)}. Retirement inputs {dict(ri)}, endpoints {dict(re)}. This is recorded physical activation, not configuration presence.
- All8 actual actor updates have finite nonzero gradients; all12 Adam states +160. LR1e-5/Identity, original3 origins/migrations and complete front AUX96/96/old7/8 ledgers preserved; AUX added0. Saved checkpoint hashes/embedded metadata and official roundtrip verify.
- Raw389/raw12/μ/σ/logp/value/reward/done match sealed storage; each sample used5 times. CPU logp error max {max_logp_error:.9g}. Projected/assisted actions remain separate from original Gaussian samples.

| Episode | Learner decisions | Total physical seconds | Credited seconds | Status | First unfinished stage | Result |
| --- | ---: | ---: | ---: | --- | --- | --- |
{rows}

The incomplete first episode and nonterminal collection-boundary partial retain their actual meanings. Neither the uncredited front prefix nor a later suffix can be reported as full P01 PPO success. Evidence is confined to `{RUN}`,8 sealed rollouts1633–1640 and actual source/final checkpoints. No other historical runs were audited. CPU-only read audit, no checkpoint/runtime/frozen-helper changes or simulator; helper exits after writing reports.
'''
    (OUT / 'block08_training_audit.md').write_text(md, encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('result', 'new_counts', 'lifetime_counts', 'request_phase_counts',
        'prefix_counts', 'execution_counts', 'RR_input_counts', 'RR_retirement_actual_input_counts',
        'RR_retirement_actual_endpoint_counts', 'CPU_Normal_logp_max_error')}, indent=2))


if __name__ == '__main__':
    main()
