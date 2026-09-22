"""Bounded block03 RR capture data feasibility only; no dataset/fit/publication."""
from collections import Counter
from copy import deepcopy
import hashlib
import itertools
import json
from pathlib import Path

import torch
import yaml
from wlr50_clean.ppo.semantic_supervisor import TaskStageSupervisor, _current_rr_placement_usable

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
ROOT = OUT.parents[1]
RUN = ROOT / 'runs/ppo_p05_hip_only_continuation_v1/train/20260922T0705260953371Z_ga802b24d78df_76c646571d794b4e96a82f69956928ff'
CURRENT = OUT / 'checkpoints/history/checkpoint_step_000214400.pt'
CURRENT_SHA = '8d0305626decff2214f8c3c0eee4031bdb791013d82a641a8ac4fc077ce235e9'
CPDIR = OUT / 'checkpoints/history'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def compact_runs(rows, key):
    intervals = []
    for row in rows:
        if row[key]:
            if intervals and intervals[-1]['last_input_tick'] + 8 == row['input_tick']:
                intervals[-1]['last_input_tick'] = row['input_tick']; intervals[-1]['samples'] += 1
            else:
                intervals.append({'first_input_tick': row['input_tick'], 'last_input_tick': row['input_tick'], 'samples': 1})
    return intervals


def summarize(rows):
    if not rows:
        return {'rows': 0}
    deltas = [r['current_phi'] - r['old_phi'] for r in rows]
    return {'rows': len(rows), 'source_index_range': [rows[0]['index'], rows[-1]['index']],
        'global_decision_range': [rows[0]['global'], rows[-1]['global']],
        'input_tick_range': [rows[0]['input_tick'], rows[-1]['input_tick']],
        'phases': dict(Counter(r['phase'] for r in rows)),
        'RR_placed_history_inputs': sum(r['placed'] for r in rows),
        'RR_exact_current_legal_TOP_inputs': sum(r['legal_TOP'] for r in rows),
        'RR_current_placement_usable_inputs': sum(r['usable'] for r in rows),
        'RR_AIR_inputs': sum(r['air'] for r in rows), 'RR_ground_inputs': sum(r['ground'] for r in rows),
        'old_to_current_Phi_changed_float32_rows': sum(r['changed_float32'] for r in rows),
        'old_to_current_Phi_delta_min': min(deltas), 'old_to_current_Phi_delta_max': max(deltas),
        'old_to_current_Phi_delta_mean': sum(deltas) / len(deltas),
        'assist_owned_actions': sum(bool(r['owners']) for r in rows),
        'RR_assist_owned_actions': sum(6 in r['owners'] or 7 in r['owners'] for r in rows),
        'raw_equals_stored_mean_rows': sum(r['raw_equals_mean'] for r in rows),
        'terminal_actions': sum(r['terminal'] for r in rows)}


def main():
    assert not torch.cuda.is_available(); torch.set_num_threads(1)
    cm = read(CURRENT.with_name(CURRENT.stem + '_manifest.json'))
    assert sha(CURRENT) == cm['checkpoint_sha256'] == CURRENT_SHA
    migration = cm['rr_postcross_workspace_migration']
    assert sha(migration['plan_path']) == migration['plan_sha256']
    factor = migration['rr_postcross_workspace_factor']
    assert factor['observation_contract']['observation_dimension'] == 389
    assert factor['reward_changed'] and not factor['policy_kernel_changed'] and not factor['sigma_changed']
    assert not factor['nominal_changed'] and not factor['physical_dynamics_changed'] and not factor['capture_assist_changed']
    spec_path = ROOT / 'configs/ppo_p05_hip_only_continuation_v1/stage_task_spec.yaml'
    assert sha(spec_path) == cm['runtime_contract']['files'][spec_path.relative_to(ROOT).as_posix()]
    current = object.__new__(TaskStageSupervisor); current.spec = yaml.safe_load(spec_path.read_text())
    old = object.__new__(TaskStageSupervisor); old.spec = deepcopy(current.spec); old.spec.pop('rr_postcross_workspace_semantics')
    updates = {}
    with (RUN / 'optimizer_updates.jsonl').open(encoding='utf-8') as stream:
        for raw_line in itertools.islice(stream, 10):
            row = json.loads(raw_line); updates[row['ppo_update']] = row
    batches, bindings = {}, {}
    records = []
    previous = None
    max_logp_error = 0.
    placement_endpoint = None
    # Reads known first episode only, stops at first P12 input index1260.
    # Skipped prior lines are not parsed and do not become candidate labels.
    with (RUN / 'residual_and_projection_audit.jsonl').open('rb') as stream:
        for index, raw_line in enumerate(itertools.islice(stream, 1261)):
            if index < 746:
                continue
            row = json.loads(raw_line); a = row['applied_audit']; ev_end = a['semantic_task']['physical_evaluator']
            assert a['decision_count'] == index + 1 and row['global_policy_decision'] == 203777 + index
            if index == 746:
                assert a['physics_tick'] == 5976 and ev_end['history']['event_ticks']['front_edge_crossed']['RR'] == 5976
                previous = row; continue
            assert previous is not None and not previous['terminal']
            ev = previous['applied_audit']['semantic_task']['physical_evaluator']
            tick = a['physics_tick'] - a['physics_ticks']
            assert tick == ev['physics_tick'] == previous['applied_audit']['physics_tick']
            assert ev['valid'] and ev['termination_reason'] is None and not row['terminal']
            update, offset = 1558 + index // 128, index % 128
            if update not in batches:
                path = RUN / f'rollouts/rollout_{update:06}.pt'
                batch = torch.load(path, map_location='cpu', weights_only=False)
                assert batch['policy_contract'] == cm['policy_contract'] and batch['curriculum_epoch']['prefix_request'] is None
                assert batch['runtime_contract']['source_git_commit'] == migration['source_git_commit']
                assert batch['runtime_contract']['runtime_content_sha256'] == migration['source_runtime_content_sha256']
                source_files = batch['runtime_contract']['files']; current_files = cm['runtime_contract']['files']
                actual_changed_files = {name: {'before': source_files.get(name), 'after': current_files.get(name)}
                    for name in set(source_files) | set(current_files) if source_files.get(name) != current_files.get(name)}
                assert actual_changed_files == migration['changed_file_hashes']
                before_step = 203776 + (update - 1558) * 128
                checkpoint = CPDIR / f'checkpoint_step_{before_step:09}.pt'
                bm = read(checkpoint.with_name(checkpoint.stem + '_manifest.json'))
                assert sha(checkpoint) == bm['checkpoint_sha256']
                assert bm['actor_parameter_sha256'] == updates[update]['actor_parameter_sha256_before']
                assert bm['runtime_contract'] == batch['runtime_contract'] and bm['policy_contract'] == batch['policy_contract']
                assert bm['ppo_updates'] == update - 1 and bm['global_policy_decisions'] == before_step
                rng = bm['training_rng_state']
                assert rng['seed'] == 1001 and len(rng['torch_cuda']) == rng['torch_cuda_device_count'] == 1
                batches[update] = batch
                bindings[str(update)] = {'rollout_path': str(path), 'rollout_sha256': sha(path),
                    'collection_source_checkpoint': str(checkpoint), 'source_checkpoint_sha256': bm['checkpoint_sha256'],
                    'source_actor_sha256': bm['actor_parameter_sha256'], 'source_PPO_update': bm['ppo_updates'],
                    'full_boundary_RNG_present': True, 'RNG_keys': sorted(rng),
                    'per_decision_RNG_snapshot_present_in_rollout': any('rng' in key.lower() for key in batch)}
            batch = batches[update]; x = batch['observations']['policy'][offset, 0]
            raw = batch['actions'][offset, 0]; mean, std = (d[offset, 0] for d in batch['distribution_params'])
            assert x.shape == (389,) and torch.equal(x, batch['observations']['critic'][offset, 0])
            assert torch.isfinite(x).all() and int(x[:13].argmax()) == int(a['phase_id'][1:]) - 1
            for tensor, field in ((raw, 'raw_policy_action_full12'), (mean, 'old_distribution_mean_full12'), (std, 'old_distribution_std_full12')):
                assert torch.equal(tensor, torch.tensor(row[field], dtype=tensor.dtype))
            native = a['actuator_target_effect_audit']; p = row['policy_request']; assist = native['capture_assist_evidence']
            assert p['sampling_draws'] == 1 and p['extra_random_draws'] == 0
            assert row['raw_policy_action_full12'] == p['selected_raw_full12'] == native['raw_policy_action_full12']
            assert native['verified'] and native['actual_mapping_matches_dispatch'] and native['phase_mask_full12'] == [1] * 12
            assert a['actuator_target_effect_audit_summary']['all_ticks_verified'] and a['no_in_episode_state_writes_verified']
            assert assist['owner_indices'] in ([], [0, 1])
            assert assist['candidate_before_assist_full12'][6:8] == assist['candidate_after_assist_full12'][6:8]
            logp = batch['actions_log_prob'][offset, 0].reshape(())
            assert torch.equal(logp, torch.tensor(row['old_log_probability'], dtype=logp.dtype))
            max_logp_error = max(max_logp_error, float((torch.distributions.Normal(mean, std).log_prob(raw).sum() - logp).abs()))
            old_phi, new_phi = old.physical_potential(ev), current.physical_potential(ev)
            assert abs(old_phi - a['reward_breakdown']['potential_before']) < 1e-12
            assert torch.tensor(old_phi, dtype=x.dtype).item() == x[17].item()
            rr = ev['current_legs']['RR']
            legal_top = bool(rr['top_contact'] and rr['top_surface_contact'] and rr['contact_surface'] == 'TOP'
                and rr['within_top_xy'] and rr['within_lateral_span'] and not rr['ground_contact'])
            records.append({'index': index, 'global': row['global_policy_decision'], 'input_tick': tick, 'phase': a['phase_id'],
                'old_phi': old_phi, 'current_phi': new_phi,
                'changed_float32': torch.tensor(new_phi, dtype=x.dtype).item() != x[17].item(),
                'placed': ev['history']['placed']['RR'], 'legal_TOP': legal_top,
                'usable': _current_rr_placement_usable(ev), 'air': rr['air'], 'ground': rr['ground_contact'],
                'owners': assist['owner_indices'], 'raw_equals_mean': torch.equal(raw, mean), 'terminal': row['terminal']})
            if ev_end['history']['placed']['RR'] and placement_endpoint is None:
                placement_endpoint = {'global_decision': row['global_policy_decision'], 'input_tick': tick,
                    'endpoint_tick': a['physics_tick'], 'event_tick': ev_end['history']['event_ticks']['placed']['RR'],
                    'current_RR_at_first_sampled_endpoint': ev_end['current_legs']['RR'],
                    'capture_assist_owner_indices': assist['owner_indices']}
            previous = row
    assert len(records) == 514 and max_logp_error <= 1e-5
    assert placement_endpoint['event_tick'] == 8726
    windows = {'crossed_to_P12_entry_inclusive': records,
        'crossed_P09_preplacement_inputs': [r for r in records if r['input_tick'] < 8728],
        'short_capture_and_immediate_handoff': [r for r in records if 8696 <= r['input_tick'] <= 8752],
        'capture_approach_through_P12_entry': [r for r in records if r['input_tick'] >= 8696],
        'postplacement_inputs_through_P12_entry': [r for r in records if r['input_tick'] >= 8728]}
    report = {'schema': 'wlr50_clean.RR_capture_historical_data_availability_readonly.v1',
        'status': 'CONDITIONAL_OFF_POLICY_SOURCE_AVAILABLE_NOT_ADMITTED_FOR_FITTING', 'source_run': str(RUN),
        'current_reference_checkpoint': str(CURRENT), 'current_reference_sha256': CURRENT_SHA,
        'migration_plan_sha256': migration['plan_sha256'],
        'actual_source_current_file_delta_exactly_matches_existing_migration': True,
        'changed_runtime_files': sorted(migration['changed_file_hashes']),
        'windows': {k: summarize(v) for k, v in windows.items()},
        'placement_event': placement_endpoint, 'bindings': bindings, 'max_CPU_raw_Gaussian_logp_error': max_logp_error,
        'RR_legal_TOP_input_runs_after_capture': compact_runs(windows['postplacement_inputs_through_P12_entry'], 'legal_TOP'),
        'RR_current_placement_usable_input_runs_after_capture': compact_runs(windows['postplacement_inputs_through_P12_entry'], 'usable'),
        'same389_old_index17_not_current_semantics': True, 'per_decision_RNG_replay_proven': False,
        'RNG_scope': 'Full Python/NumPy/TorchCPU/CUDA state is saved at each collection-boundary model checkpoint, not a per-action RNG snapshot. Actual raw/mean/std/logp are saved; no stochastic replay is needed or claimed.',
        'data_or_targets_written': False, 'fit_executed': False, 'budget_selected': False,
        'PPO_or_AUX_credit_added': 0, 'full_task_success_claimed': False,
        'conceptual_only_migration': 'A future explicitly admitted off-policy dataset could retain immutable old X389/raw12/provenance and form a separate derived X389 with only index17 replaced by current pure physical_potential of the exactly aligned input evaluator. Bind both hashes and delta, retain actual raw target unchanged. Old means/std/logp remain source provenance; old value/GAE/reward are NOT current on-policy truth. This script writes no such dataset.'}
    (HERE / 'availability_report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    rows = '\n'.join(f"| {name} | {v['rows']} | {v['input_tick_range']} | {v['old_to_current_Phi_changed_float32_rows']} | {v['old_to_current_Phi_delta_min']:.9g}…{v['old_to_current_Phi_delta_max']:.9g} |" for name, v in report['windows'].items())
    post = report['windows']['postplacement_inputs_through_P12_entry']
    md = f'''# RR capture source availability — read-only, conditional only

Direct **389 inputs + actual sampled raw12 + stored μ/σ/logp** exist in sealed rollouts1563–1567, with the exact collecting model/optimizer-update chain and full boundary RNG. Every inspected raw label matches source storage, selected sample and native dispatch evidence; CPU logp max error {max_logp_error:.9g}. None is substituted by stored μ or nominal/final targets. No RR assist ownership occurs. This does not relabel earlier FL-assisted preparation as pure-policy.

Only block03 episode0 indices747–1260 were analyzed (514 rows), using index746 solely as the preceding input-state endpoint. All are nonterminal, native-verified original learner samples, not prefix/evaluation actions. Per-decision RNG snapshots are not stored/proven; boundary RNG and actual samples are sufficient provenance, not a claim of bitwise simulator replay.

| Possible bounded scope (not a selected dataset) | Rows | Input ticks | Changed X17 rows | Current minus old Φ |
| --- | ---: | --- | ---: | --- |
{rows}

Actual RR placement event is tick8726 in action global204867 (input8720, endpoint8728). That first sampled endpoint is already AIR, so placement history must not be called sustained contact. Real TOP follows at8736/8744 and further handoff states. Among {post['rows']} post-placement inputs through P12 entry, {post['RR_exact_current_legal_TOP_inputs']} have current legal TOP and {post['RR_current_placement_usable_inputs']} meet the existing current placement-usability rule; contact interruptions remain visible in JSON. P10/P11 continue, and P12 is entered at tick10080 with genuine RR TOP. Later RL failure and RR retreat are excluded from these proposed local scopes, not relabeled success.

**Current5fd cannot consume the whole old389 window unchanged.** The reviewed a802→5fd migration changes reward potential and X17 numerical semantics, not physics/nominal/caps/sigma/other codec positions. The recomputed old Φ exactly reproduces stored X17; the table quantifies changed current Φ. For any future separately approved OFF-POLICY supervised use, retain the immutable source and explicitly derive only X17 from the aligned physical evaluator, recording old/new values and hashes while retaining actual raw labels. Old source μ/σ/logp remain provenance, not current-model likelihood; old values/GAE/returns must never enter current on-policy PPO. Changed X17 can itself change the current actor's output, so source action equivalence is not claimed.

This is source feasibility only: no fitted dataset, teacher/new demonstration, new optimizer, model change, credit or budget. The 8-row immediate capture/handoff or 174-row broader approach/handoff scopes are possible temporal bounds, not blanket physical-positive eligibility for every AIR/contact-interruption row. Eligibility/split/parameter subspace and current-source binding would require a separate decision. If the current natural-P01 evaluation already captures RR, this candidate need not proceed. CPU helper exits after this report.
'''
    (HERE / 'availability_report.md').write_text(md, encoding='utf-8')
    print(json.dumps({'status': report['status'], 'windows': report['windows'],
        'placement_event': {k: v for k, v in placement_endpoint.items() if k != 'current_RR_at_first_sampled_endpoint'},
        'max_CPU_raw_Gaussian_logp_error': max_logp_error, 'files_written': 'reports only; no dataset or checkpoint'}, indent=2))


if __name__ == '__main__':
    main()
