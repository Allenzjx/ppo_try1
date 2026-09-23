"""One-pass CPU audit of a SEALED RR410 training block; never polls or trains.

Run only after the owning process exits. No simulator, actor forward, checkpoint
write, current-runtime waiver, or edits to source manifests. Reports are new files
beside this script. Endpoint coverage is explicitly not every-physics-tick coverage.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re

OUT = Path(__file__).resolve().parent
COUNTERS = ('global_policy_decisions', 'ppo_updates', 'optimizer_steps')
PHASES = tuple(f'P{i:02}' for i in range(1, 14))
SEALED = ('SUCCEEDED', 'STOPPED_AT_VERIFIED_UPDATE_BOUNDARY')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def lines(path):
    with Path(path).open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            require(bool(line.strip()), f'blank JSONL row: {path}:{number}')
            yield json.loads(line)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def parameter_sha(state):
    result = hashlib.sha256()
    for key, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        result.update(key.encode()); result.update(str(value.dtype).encode())
        result.update(str(tuple(value.shape)).encode()); result.update(value.numpy().tobytes())
    return result.hexdigest()


def prefix_summary(run, source, source_sha, target):
    """Keep only compact start records, including honest fresh-P01 fallbacks."""
    path = run / 'prefix_evidence.jsonl'
    if not path.exists():
        return [], {'decisions': 0, 'physics_ticks': 0, 'PPO_credit': 0}
    starts, totals, phases = [], Counter(), Counter()
    count, ticks, attempt, request = 0, 0, None, None
    for row in lines(path):
        require(row['policy_credit'] is False, 'prefix was credited as learning')
        kind = row['kind']
        if kind == 'checkpoint_prefix_start':
            require(count == 0 and attempt is None, 'unclosed prefix attempt')
            request = row['request']
        elif kind == 'checkpoint_prefix_decision':
            require(row['actuator_target_effect_audit_summary']['all_ticks_verified']
                    and row['no_in_episode_state_writes_verified'], 'unverified prefix physics')
            count += 1; ticks += row['physics_ticks']; phases[row['phase_id']] += 1
        elif kind == 'checkpoint_prefix_result':
            require(row['prefix_decisions'] == count and row['prefix_physics_ticks'] == ticks,
                    'prefix attempt counters disagree')
            attempt = row
        elif kind == 'policy_credit_start':
            start = row['start']; provenance = start['prefix_policy_provenance']
            require(attempt is not None, 'missing completed prefix attempt')
            require(provenance['checkpoint_sha256'] == source_sha
                    and provenance['source_global_policy_decisions'] == source['global_policy_decisions']
                    and provenance['frozen_actor_parameter_sha256'] == source['actor_parameter_sha256']
                    and provenance['frozen_for_entire_training_block']
                    and provenance['independent_parameter_and_buffer_storage_verified'], 'prefix source mismatch')
            require(provenance['source_policy_contract'] == provenance['effective_policy_contract'] == target['policy_contract']
                    and provenance['source_runtime_content_sha256'] == provenance['effective_runtime_content_sha256']
                    == target['runtime_contract']['runtime_content_sha256'], 'prefix effective contract mismatch')
            accepted = attempt['accepted'] is True
            require((accepted and start['actual_phase'] == start['requested_phase']) or
                    (not accepted and start['mode'] == 'fresh_P01_fallback'
                     and start['actual_phase'] == 'P01' and start['physics_tick'] == 0), 'prefix/fallback mismatch')
            starts.append({'attempt_decisions': count, 'attempt_physics_ticks': ticks,
                'accepted': accepted, 'miss': attempt['miss'], 'request': request,
                'phase_counts': dict(phases), 'start_tick': start['physics_tick'],
                'start_time_s': start['sim_time_s'], 'actual_phase': start['actual_phase'],
                'requested_phase': start['requested_phase'], 'mode': start['mode'],
                'physical_predecessor_decisions_in_current_episode': count if accepted else 0})
            totals['decisions'] += count; totals['physics_ticks'] += ticks
            totals['accepted_attempts' if accepted else 'failed_prefix_fresh_reset_fallbacks'] += 1
            count, ticks, attempt, phases = 0, 0, None, Counter()
    require(count == 0 and attempt is None, 'prefix stream not sealed at a complete credit-start record')
    return starts, {**dict(totals), 'PPO_credit': 0}


def bound_support_floor(target):
    """Read the target's immutable task-spec bytes, never a current worktree default."""
    import yaml
    from wlr50_clean.ppo.semantic_migration import _version_bytes
    contract = target['runtime_contract']
    binding = contract['selected_configuration']['stage_task_spec.yaml']
    require(binding['path'] == f"configs/ppo_{contract['experiment_id']}/stage_task_spec.yaml"
            and binding['sha256'] == contract['files'][binding['path']], 'task-spec binding is inconsistent')
    raw = _version_bytes(OUT.parents[1], contract, binding['path'])
    require(hashlib.sha256(raw).hexdigest() == binding['sha256'], 'versioned task-spec SHA mismatch')
    floor = yaml.safe_load(raw)['support']['force_noise_floor_n']
    require(type(floor) in (int, float) and math.isfinite(floor) and floor >= 0., 'invalid bound support force noise floor')
    return float(floor), {**binding, 'source_git_commit': contract['source_git_commit'],
                          'force_noise_floor_n': float(floor), 'source': 'target_runtime_versioned_git_bytes'}


def rr_endpoint(ev, force_noise_floor_n):
    rr, history = ev['current_legs']['RR'], ev['history']
    top = all(rr.get(k) is True for k in ('top_contact', 'top_surface_contact', 'obstacle_pair_active', 'within_top_xy'))
    top = top and rr.get('contact_surface') == 'TOP' and rr.get('ground_contact') is False
    force = rr.get('bearing_force_n')
    bearing = bool(top and rr.get('air') is False and rr.get('support') is True and rr.get('bearing_verified') is True
                   and type(force) in (int, float) and math.isfinite(force) and force >= force_noise_floor_n)
    return {'raw_current_lift_valid': rr.get('current_lift_valid') is True,
        'qualified_current': bool(rr.get('current_lift_valid') is True and history['active_lift']['RR'] is True
                                  and rr.get('ground_contact') is False),
        'qualified_history_active': history['active_lift']['RR'] is True,
        'crossed_history': history['front_edge_crossed']['RR'] is True,
        'current_TOP': top, 'current_TOP_verified_bearing': bearing,
        'placed_history': history['placed']['RR'] is True, 'grounded_current': rr.get('ground_contact') is True,
        'gap_m': rr['clearance_m'], 'front_distance_m': rr['front_distance_m'], 'bearing_force_n': force}


def audit(args):
    require(re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]*', args.report_name), 'report-name must be a plain new basename')
    reports = [OUT / (args.report_name + suffix) for suffix in ('.json', '.md')]
    require(not any(p.exists() for p in reports), 'report exists; use a new name, do not overwrite evidence')
    run, source_path, target_path = (Path(x).resolve(strict=True) for x in (args.run, args.source, args.target))
    # Reject active/incomplete runs BEFORE importing torch or reading streams.
    manifest, outer = read(run / 'training_manifest.json'), read(run / 'run_manifest.json')
    require(manifest['lifecycle'] in SEALED, 'training not sealed at a complete update; do not poll')
    require(outer['lifecycle'] in (*SEALED, 'FAILED'), 'outer process not sealed; run this only after process exit')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '-1', 'set CUDA_VISIBLE_DEVICES=-1 before invoking the CPU audit')
    import torch
    from wlr50_clean.ppo.semantic_training import state_hash
    require(not torch.cuda.is_available(), 'CPU-only audit required')
    torch.set_num_threads(1)
    sm, tm = (read(p.with_name(p.stem + '_manifest.json')) for p in (source_path, target_path))
    source_sha, target_sha = sha(source_path), sha(target_path)
    require(source_sha == sm['checkpoint_sha256'] and target_sha == tm['checkpoint_sha256'], 'checkpoint bytes do not match manifest')
    require(Path(tm['source_run']).resolve() == run and tm['save_load_round_trip'] is True, 'target is not an official saved checkpoint of this run')
    require(outer['runtime_contract'] == tm['runtime_contract'], 'sealed run and checkpoint runtime contracts differ')
    require(sm['policy_contract']['observation_dimension'] == tm['policy_contract']['observation_dimension'] == 410,
            'requires the actual RR410 ABI')
    force_noise_floor_n, support_binding = bound_support_floor(tm)
    delta = {k: tm[k] - sm[k] for k in COUNTERS}
    decisions, updates_n, adam_steps = (delta[k] for k in COUNTERS)
    require(decisions > 0 and decisions % 128 == 0 and updates_n == decisions // 128 and adam_steps == updates_n * 20,
            'source/target count delta is not complete128 / PPO1 / Adam20 updates')
    require(manifest['actual_policy_decisions'] == decisions and manifest['ppo_updates_this_run'] == updates_n
            and manifest['optimizer_steps_this_run'] == adam_steps and manifest['rounding_overrun'] == 0,
            'sealed training counts disagree with saved source/target')
    require(manifest['planned_requested_policy_decisions'] - manifest['unconsumed_requested_policy_decisions'] == decisions,
            'planned minus unconsumed differs from actual credit')
    preserved = sorted({k for k in sm if k != 'resume_migration' and k.endswith(('_branch', '_migration'))}
        | {'runtime_contract', 'policy_contract', 'runner_config', 'normalization', 'normalizer_state_sha256'})
    require(all(sm[k] == tm.get(k) for k in preserved), 'ordinary PPO altered a branch, migration, AUX or immutable contract')
    for key in ('rr_capture_feedback_peak_v2_migration', 'rr_capture_knee_v3_migration'):
        require(key in sm and key in tm, 'missing required v2/v3 controller receipt: ' + key)
    origins = {}
    for key in preserved:
        if key.endswith('_branch') and isinstance(sm[key], dict) and set(sm[key].get('counter_origin', {})) == set(COUNTERS):
            origin = sm[key]['counter_origin']; count_key = key + '_counts'
            require(tm[count_key] == {k: tm[k] - origin[k] for k in COUNTERS}, 'branch counts disagree: ' + key)
            require(sm[count_key] == {k: sm[k] - origin[k] for k in COUNTERS}, 'source branch counts disagree: ' + key)
            origins[key] = {'counter_origin': origin, 'target_counts': tm[count_key]}
    ledger = tm['rr_postcross_workspace_branch']['front_rehearsal_auxiliary']
    old_aux = tm['task_conditioned_hip_wheel_branch']['auxiliary_mean_learning']
    aux = {'accepted': ledger['accepted_auxiliary_updates_total'], 'attempted': ledger['attempted_auxiliary_optimizer_steps_total'],
        'events': [{'event_index': e['event_index'], 'kind': e.get('kind'),
                    'accepted': e['fit_report']['accepted_auxiliary_updates'],
                    'attempted': e['fit_report']['attempted_auxiliary_optimizer_steps']} for e in ledger['events']],
        'old_separate_accepted': old_aux['accepted_auxiliary_updates_total'],
        'old_separate_attempted': old_aux['attempted_auxiliary_optimizer_steps_total'], 'new_AUX': 0}
    starts, prefix_totals = prefix_summary(run, sm, source_sha, tm)
    phases, endpoint_phases, rr_counts, assist_counts, execution = (Counter() for _ in range(5))
    episodes, update_summary = [], []
    maxima = dict(combined_target_travel_deg=0., knee_target_travel_deg=0., total_exposure_s=0., derived_hip_exposure_s=0.)
    rows, updates = iter(lines(run / 'residual_and_projection_audit.jsonl')), iter(lines(run / 'optimizer_updates.jsonl'))
    previous, previous_actor_hash, consumed, max_logp = None, sm['actor_parameter_sha256'], 0, 0.
    for block in range(updates_n):
        update = next(updates, None); require(update is not None, 'missing sealed update row')
        number = sm['ppo_updates'] + block + 1
        require(update['ppo_update'] == number and update['global_policy_decisions'] == sm['global_policy_decisions'] + (block + 1) * 128,
                'update counter sequence mismatch')
        require(update['optimizer_steps'] == 20 and update['actor_parameter_sha256_before'] == previous_actor_hash,
                'optimizer/actor update chain mismatch')
        previous_actor_hash = update['actor_parameter_sha256_after']
        selected = list(itertools.islice(rows, 128)); require(len(selected) == 128, 'missing sealed rollout rows')
        batch = torch.load(run / f'rollouts/rollout_{number:06}.pt', map_location='cpu', weights_only=False)
        obs, actions = batch['observations']['policy'], batch['actions']
        means, stds = batch['distribution_params']
        require(tuple(obs.shape) == (128, 1, 410) and tuple(actions.shape) == (128, 1, 12)
                and torch.equal(obs, batch['observations']['critic']), 'wrong fresh rollout ABI')
        require(batch['runtime_contract'] == tm['runtime_contract'] and batch['policy_contract'] == tm['policy_contract'], 'rollout contract mismatch')
        for offset, row in enumerate(selected):
            a, request = row['applied_audit'], row['policy_request']
            native = a['actuator_target_effect_audit']; ev = a['semantic_task']['physical_evaluator']
            if a['decision_count'] == 1:
                require(not episodes or episodes[-1]['terminal'], 'reset without a terminal episode')
                prefix = starts[len(episodes)] if starts else None
                start_tick = a['physics_tick'] - a['physics_ticks']
                require(start_tick == (prefix['start_tick'] if prefix else 0), 'learner begins from an unbound physical state')
                require(a['phase_id'] == (prefix['actual_phase'] if prefix else 'P01'), 'learner entry phase differs from prefix/reset')
                episodes.append({'episode_index': len(episodes), 'learner_decisions': 0, 'learner_physics_ticks': 0,
                    'input_phases': Counter(), 'start_tick': start_tick, 'initialization': prefix['mode'] if prefix else 'natural_P01',
                    'prefix': prefix, 'RR_endpoint_counts': Counter(), 'first_observed_RR_placed_tick': None,
                    'first_RR_regrounded_after_placement_endpoint': None})
                previous = None
            require(bool(episodes), 'first learner row is not episode decision1')
            ep = episodes[-1]; consumed += 1
            require(row['global_policy_decision'] == sm['global_policy_decisions'] + consumed
                    and a['decision_count'] == ep['learner_decisions'] + 1, 'credited decision sequence mismatch')
            require(not a.get('prefix_teacher_data_in_ppo_storage', False)
                    and not a.get('prefix_checkpoint_policy_data_in_ppo_storage', False), 'prefix entered on-policy storage')
            if previous is not None:
                require(not previous['terminal'] and previous['applied_audit']['end_phase_id'] == a['phase_id']
                        and previous['applied_audit']['physics_tick'] == a['physics_tick'] - a['physics_ticks'], 'phase/physics continuity broken')
            if a['phase_id'] != a['end_phase_id']:
                # A physical safety terminal may coincide with a phase change;
                # do not relabel it an ordinary handoff or invent a continuity failure.
                if row['terminal']:
                    require(a['termination_reason'] is not None, 'terminal phase change lacks a real terminal reason')
                    execution['terminal_endpoints_with_phase_change'] += 1
                else:
                    execution['nonterminal_phase_handoffs'] += 1
            require(request['sampling_draws'] == 1 and request['extra_random_draws'] == request['extra_model_forwards'] == 0,
                    'request audit changed original Gaussian sampling')
            require(row['raw_policy_action_full12'] == request['selected_raw_full12'] == native['raw_policy_action_full12']
                    and row['old_distribution_mean_full12'] == request['conditional_mean_full12']
                    and row['old_distribution_std_full12'] == request['effective_sigma_full12']
                    and row['old_log_probability'] == request['selected_raw_log_probability'], 'original Gaussian evidence mismatch')
            require(native['verified'] and native['actual_mapping_matches_dispatch'] and native['phase_mask_full12'] == [1] * 12,
                    'native dispatch/mask verification failed')
            require(a['actuator_target_effect_audit_summary']['all_ticks_verified'] and a['no_in_episode_state_writes_verified'], 'unverified physical execution')
            ticks = a['actuator_target_effect_audit_ticks']
            require(len(ticks) == a['physics_ticks'] and all(t['verified'] for t in ticks), 'missing native tick verification')
            rr_assist = native['rr_capture_assist_evidence']; state = rr_assist['state_after']; before = rr_assist['state_before']
            require(native['rr_capture_assist_state_transition_independently_reconstructed']
                    and rr_assist['owner_indices'] in ([], [6, 7]) and rr_assist['policy_request_unchanged']
                    and rr_assist['knee_hold_final_verified'], 'RR transform not independently verified')
            require(all(rr_assist['candidate_before_assist_full12'][i] == rr_assist['candidate_after_assist_full12'][i]
                        for i in range(12) if i not in (6, 7)), 'RR assist changed an unowned channel')
            travel, elapsed = state['travel_used_deg'], state['descent_elapsed_s']
            knee_used, hip_elapsed = max(travel - 20., 0.), elapsed - max(travel - 20., 0.)
            require(all(math.isfinite(x) for x in (travel, elapsed)) and 0 <= travel <= 40. + 1e-9
                    and knee_used <= 20. + 1e-9 and 0 <= elapsed <= 32. + 1e-9
                    and -1e-9 <= hip_elapsed <= 12. + 1e-9, 'RR public cumulative budget exceeded')
            for key, value in zip(maxima, (travel, knee_used, elapsed, hip_elapsed)):
                maxima[key] = max(maxima[key], value)
            assist_counts['owned_endpoints'] += bool(rr_assist['owner_indices'])
            assist_counts['mode_' + state['mode_name']] += 1
            assist_counts['knee_axis_available_endpoints'] += state['mode_name'] == 'DESCEND' and travel >= 20.
            assist_counts['knee_command_increment_last_tick'] += knee_used > max(before['travel_used_deg'] - 20., 0.)
            assist_counts['FL_owned_endpoints'] += bool(native.get('capture_assist_evidence', {}).get('owner_indices', []))
            endpoint = rr_endpoint(ev, force_noise_floor_n)
            for key, value in endpoint.items():
                if type(value) is bool: rr_counts[key] += value; ep['RR_endpoint_counts'][key] += value
            if endpoint['placed_history'] and ep['first_observed_RR_placed_tick'] is None:
                ep['first_observed_RR_placed_tick'] = ev['history']['event_ticks']['placed']['RR']
            if endpoint['placed_history'] and endpoint['grounded_current'] and ep['first_RR_regrounded_after_placement_endpoint'] is None:
                ep['first_RR_regrounded_after_placement_endpoint'] = a['physics_tick']
            ep['learner_decisions'] += 1; ep['learner_physics_ticks'] += a['physics_ticks']; ep['input_phases'][a['phase_id']] += 1
            ep.update(physical_duration_s=a['sim_time_s'], last_phase=a['end_phase_id'], terminal=bool(row['terminal']),
                termination_reason=a['termination_reason'], task_result_scope=a.get('task_result_scope'),
                recorded_full_task_success=bool(a['full_task_success']), placed_history=ev['history']['placed'],
                event_ticks=ev['history']['event_ticks'], final_RR=endpoint,
                final_RL={k: ev['current_legs']['RL'].get(k) for k in ('clearance_m', 'front_distance_m', 'top_contact', 'ground_contact', 'air', 'support')})
            phases[a['phase_id']] += 1; endpoint_phases[a['end_phase_id']] += 1
            execution['learner_physics_ticks'] += a['physics_ticks']; execution['native_verified_ticks'] += len(ticks)
            previous = row
        for tensor, key in ((actions, 'raw_policy_action_full12'), (means, 'old_distribution_mean_full12'),
                (stds, 'old_distribution_std_full12'), (batch['actions_log_prob'], 'old_log_probability'),
                (batch['values'], 'old_value'), (batch['rewards'], 'reward'), (batch['dones'], 'terminal')):
            require(bool(torch.isfinite(tensor).all()) and torch.equal(tensor, torch.tensor([r[key] for r in selected], dtype=tensor.dtype).reshape(tensor.shape)),
                    'saved raw/likelihood/reward/done differs from synchronized stream: ' + key)
        require([PHASES[int(i)] for i in obs[:, 0, :13].argmax(-1)] == [r['applied_audit']['phase_id'] for r in selected], 'input phase one-hot mismatch')
        for begin, end, key in ((389, 403, 'rr_capture_assist_observed_features'), (403, 410, 'rr_capture_transfer_observed_features')):
            require(torch.equal(obs[:, 0, begin:end], torch.tensor([r['policy_request'][key] for r in selected], dtype=obs.dtype)), 'RR410 observed feature audit mismatch')
        require(torch.equal(obs[:, 0, 17], torch.tensor([r['applied_audit']['reward_breakdown']['potential_before'] for r in selected], dtype=obs.dtype)), 'task potential input mismatch')
        likelihood = read(run / f'rollouts/update_{number:06}_likelihood.json')
        uses = Counter(i for mini in likelihood['minibatches'] for ids in mini['rollout_flat_indices'] for i in ids)
        require(len(likelihood['minibatches']) == 20 and uses == Counter({i: 5 for i in range(128)}), 'not20 minibatches/five uses of each original sample')
        require(likelihood['extra_random_draws'] == likelihood['extra_model_forwards'] == 0, 'likelihood audit perturbed training')
        error = float((torch.distributions.Normal(means, stds).log_prob(actions).sum(-1).unsqueeze(-1) - batch['actions_log_prob']).abs().max())
        max_logp = max(max_logp, error); require(error <= 1e-5, 'raw Gaussian old likelihood mismatch')
        update_summary.append({'update': number, 'effective_LR': update['optimizer_learning_rate'],
            'actor_changed': update['actor_parameters_changed'], 'finite_nonzero_gradient': update['finite_nonzero_gradient_observed']})
    require(next(rows, None) is None and next(updates, None) is None and consumed == decisions, 'target does not cover all sealed rows/updates')
    require(previous_actor_hash == tm['actor_parameter_sha256'], 'final actor hash chain mismatch')
    require(len(starts) in (0, len(episodes), len(episodes) + 1), 'prefix/episode alignment mismatch')
    completed = list(lines(run / 'completed_episodes.jsonl')); ended = [ep for ep in episodes if ep['terminal']]
    require(len(completed) == len(ended), 'completed episodes do not match actual terminal rows')
    for ep, row in zip(ended, completed):
        require((ep['learner_decisions'], ep['termination_reason'], ep['physical_duration_s'], ep['recorded_full_task_success'])
                == (row['policy_decisions'], row['termination_reason'], row['duration_s'], row['full_task_success']), 'episode summary mismatch')
    require(manifest['telemetry']['core']['physics_ticks'] == execution['learner_physics_ticks'], 'actual physics tick count mismatch')
    payloads = [torch.load(p, map_location='cpu', weights_only=False) for p in (source_path, target_path)]
    for payload, meta in zip(payloads, (sm, tm)):
        require(parameter_sha(payload['actor_state_dict']) == meta['actor_parameter_sha256']
                and parameter_sha(payload['critic_state_dict']) == meta['critic_parameter_sha256']
                and state_hash(payload['optimizer_state_dict']) == meta['optimizer_state_sha256'], 'actual checkpoint payload hashes disagree')
        require(all(meta.get(k) == v for k, v in payload['infos'].items()), 'embedded infos differ from sidecar')
    before_adam, after_adam = (p['optimizer_state_dict']['state'] for p in payloads)
    require(set(before_adam) == set(after_adam), 'Adam state identifiers changed')
    step_deltas = [float(after_adam[k]['step'] - v['step']) for k, v in before_adam.items()]
    require(step_deltas and all(x == adam_steps for x in step_deltas), 'actual Adam steps disagree with new PPO credit')
    require(tm['optimizer_learning_rate'] == update_summary[-1]['effective_LR'] and 'identity_RSL_normalizer' in tm['normalization'], 'effective LR/Identity metadata mismatch')
    srng, trng = sm['training_rng_state'], tm['training_rng_state']
    require(set(srng) == set(trng) and srng['seed'] == trng['seed']
            and srng['torch_cuda_device_count'] == trng['torch_cuda_device_count'] == len(trng['torch_cuda']), 'full saved RNG schema/device count changed')
    for ep in episodes:
        ep['input_phases'] = {p: ep['input_phases'][p] for p in PHASES}
        ep['RR_endpoint_counts'] = dict(ep['RR_endpoint_counts'])
        ep['outcome_scope'] = ('recorded_terminal_not_necessarily_success' if ep['terminal'] else 'sampling_boundary_partial_not_failure_or_success')
        if ep['initialization'] not in ('natural_P01', 'fresh_P01_fallback'):
            require(not ep['recorded_full_task_success'], 'suffix was labeled full-P01 policy success')
    report = {'schema': 'wlr50_clean.sealed_RR410_training_audit.v1', 'integrity_result': 'PASS',
        'run': str(run), 'source_checkpoint': str(source_path), 'source_sha256': source_sha,
        'target_checkpoint': str(target_path), 'target_sha256': target_sha, 'actual_added_counts': delta,
        'actual_lifetime_counts': {k: tm[k] for k in COUNTERS}, 'training_lifecycle': manifest['lifecycle'],
        'outer_lifecycle_preserved': outer['lifecycle'], 'outer_error_preserved': outer.get('error'),
        'outer_failure_not_reclassified': outer['lifecycle'] == 'FAILED',
        'planned_decisions': manifest['planned_requested_policy_decisions'],
        'unconsumed_decisions_no_credit': manifest['unconsumed_requested_policy_decisions'],
        'phase_input_counts': {p: phases[p] for p in PHASES}, 'phase_endpoint_counts': dict(endpoint_phases),
        'execution': dict(execution), 'RR_endpoint_counts': dict(rr_counts), 'assist_endpoint_counts': dict(assist_counts),
        'current_support_task_spec_binding': support_binding,
        'RR_assist_maximum_public_budgets': maxima, 'coverage_scope': 'decision endpoints; tick receipts independently verified, not rescored as endpoint samples',
        'prefix_totals': prefix_totals, 'prefix_starts': starts, 'episodes': episodes,
        'terminal_episodes': len(ended), 'partial_episodes': len(episodes) - len(ended),
        'all_origins': origins, 'all_source_branch_and_migration_fields_preserved': preserved, 'AUX': aux,
        'raw410_full12_gaussian_value_reward_done_exact': True, 'sample_uses_per_update': 5,
        'CPU_logp_max_error': max_logp, 'per_update': update_summary, 'actual_Adam_step_deltas': step_deltas,
        'actor_changed': sm['actor_parameter_sha256'] != tm['actor_parameter_sha256'],
        'Adam_changed': sm['optimizer_state_sha256'] != tm['optimizer_state_sha256'], 'Identity_preserved': True,
        'full_RNG_schema_preserved': True, 'saved_RNG_advanced': state_hash(srng) != state_hash(trng),
        'official_save_reload_recorded': True, 'physical_success_inferred_from_lifecycle': False,
        'audit_simulation_fit_or_checkpoint_writes': 0}
    with reports[0].open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    episode_table = '\n'.join(f"| {e['episode_index']} | {e['initialization']} | {e['learner_decisions']} | {e['physical_duration_s']:.6f} | {e['last_phase']} | {e['termination_reason'] if e['terminal'] else 'sampling-boundary partial'} |" for e in episodes)
    md = f'''# {args.report_name}

Integrity PASS. Actual new decisions/PPO/Adam: {decisions}/{updates_n}/{adam_steps}. Lifetime: {tm['global_policy_decisions']}/{tm['ppo_updates']}/{tm['optimizer_steps']}.

Training lifecycle `{manifest['lifecycle']}`; original outer lifecycle `{outer['lifecycle']}` (error: {outer.get('error')!r}). Neither lifecycle implies physical task success. Unconsumed {report['unconsumed_decisions_no_credit']} decisions receive no credit.

Phase inputs: {report['phase_input_counts']}. Frozen prefix: {prefix_totals}; all zero PPO credit. Native-verified learner ticks: {execution['native_verified_ticks']}.

| Episode | Initialization | Learner decisions | Physical seconds | Final/current phase | Recorded result |
| --- | --- | ---: | ---: | --- | --- |
{episode_table}

RR decision-endpoint coverage: {dict(rr_counts)}. Assist endpoint counts: {dict(assist_counts)}. Maximum public budgets: {maxima}. A knee-axis-available endpoint is not itself proof of a knee command or task success. Historical placement is not current TOP/bearing. Qualified-current requires current_lift_valid plus active_lift and explicit non-ground; raw_current_lift_valid is reported separately. Current TOP bearing additionally requires air=False, support/bearing_verified, and finite measured force >= {force_noise_floor_n:.9g} N from the target's SHA-bound task specification ({support_binding['source_git_commit']}, {support_binding['sha256']}). No worktree/default threshold is substituted.

All source branches/migrations, including v2 and v3 controller receipts, and whole AUX ledgers remain exact. AUX summary: {aux}. New AUX0. Raw410/12-action Gaussian μ/σ/logp/value/reward/done match sealed storage; each sample is used5 times. Maximum CPU logp error {max_logp:.9g}. Adam states advance{adam_steps}; Identity and saved full RNG schema persist. Effective LR is reported per update, final{tm['optimizer_learning_rate']:.9g}; it is not assumed constant.

Source `{source_path}` SHA {source_sha}; target `{target_path}` SHA {target_sha}. Official save/reload receipt and actual payload hashes verified. Prefix-initialized suffix completion is not full-P01 PPO success; nonterminal partials are neither failures nor successes. This audit does no simulation, optimization, production editing or checkpoint writing.
'''
    with reports[1].open('x', encoding='utf-8') as stream:
        stream.write(md)
    print(json.dumps({k: report[k] for k in ('integrity_result', 'actual_added_counts', 'phase_input_counts', 'terminal_episodes', 'partial_episodes')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'source', 'target', 'report-name'):
        parser.add_argument('--' + name, required=True)
    audit(parser.parse_args())
