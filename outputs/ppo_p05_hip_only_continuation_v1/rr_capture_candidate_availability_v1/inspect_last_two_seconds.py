"""41-state CPU-only inspection, no gradient, dataset export or fitting."""
from collections import Counter
from contextlib import redirect_stdout
import io
import itertools
import json
import sys

import torch
import yaml
from audit_availability import HERE, OUT, ROOT, RUN, CURRENT, CURRENT_SHA, read, sha
from copy import deepcopy
from wlr50_clean.ppo.semantic_supervisor import TaskStageSupervisor, _current_rr_placement_usable

NAMES = ['FL_hip', 'FL_knee', 'FR_hip', 'FR_knee', 'RL_hip', 'RL_knee', 'RR_hip', 'RR_knee', 'FL_wheel', 'FR_wheel', 'RL_wheel', 'RR_wheel']


def main():
    assert not torch.cuda.is_available(); torch.set_num_threads(1)
    cm = read(CURRENT.with_name(CURRENT.stem + '_manifest.json'))
    assert sha(CURRENT) == cm['checkpoint_sha256'] == CURRENT_SHA
    previous_report = read(HERE / 'availability_report.json')
    assert previous_report['current_reference_sha256'] == CURRENT_SHA
    spec_path = ROOT / 'configs/ppo_p05_hip_only_continuation_v1/stage_task_spec.yaml'
    assert sha(spec_path) == cm['runtime_contract']['files'][spec_path.relative_to(ROOT).as_posix()]
    current = object.__new__(TaskStageSupervisor); current.spec = yaml.safe_load(spec_path.read_text())
    old = object.__new__(TaskStageSupervisor); old.spec = deepcopy(current.spec); old.spec.pop('rr_postcross_workspace_semantics')
    batch_path = RUN / 'rollouts/rollout_001566.pt'
    assert sha(batch_path) == previous_report['bindings']['1566']['rollout_sha256']
    batch = torch.load(batch_path, map_location='cpu', weights_only=False)
    assert batch['policy_contract'] == cm['policy_contract']
    xs, targets, source_means, physical = [], [], [], []
    previous = None
    with (RUN / 'residual_and_projection_audit.jsonl').open('rb') as stream:
        for index, raw_line in enumerate(itertools.islice(stream, 1102)):
            if index < 1060:
                continue
            row = json.loads(raw_line)
            if index == 1060:
                previous = row; continue
            a = row['applied_audit']; prior = previous['applied_audit']; ev = prior['semantic_task']['physical_evaluator']
            tick = a['physics_tick'] - a['physics_ticks']
            assert tick == ev['physics_tick'] == prior['physics_tick'] == 8 * index
            assert 8488 <= tick <= 8808 and ev['valid'] and ev['termination_reason'] is None
            assert not previous['terminal'] and not row['terminal']
            offset = index % 128
            original = batch['observations']['policy'][offset, 0]
            x = original.clone(); raw = batch['actions'][offset, 0]; mu, sigma = (d[offset, 0] for d in batch['distribution_params'])
            assert torch.equal(raw, torch.tensor(row['raw_policy_action_full12'], dtype=raw.dtype))
            assert torch.equal(mu, torch.tensor(row['old_distribution_mean_full12'], dtype=mu.dtype))
            assert torch.equal(sigma, torch.tensor(row['old_distribution_std_full12'], dtype=sigma.dtype))
            native = a['actuator_target_effect_audit']
            assert native['verified'] and native['phase_mask_full12'] == [1] * 12
            assert native['capture_assist_evidence']['owner_indices'] == []
            assert row['raw_policy_action_full12'] == native['raw_policy_action_full12'] == row['policy_request']['selected_raw_full12']
            old_phi, new_phi = old.physical_potential(ev), current.physical_potential(ev)
            assert float(torch.tensor(old_phi, dtype=x.dtype)) == float(x[17])
            x[17] = new_phi
            assert torch.equal(x[:17], original[:17]) and torch.equal(x[18:], original[18:])
            xs.append(x); targets.append(raw); source_means.append(mu)
            rr = ev['current_legs']['RR']
            supports = {leg: {key: ev['current_legs'][leg][key] for key in
                ('contact_surface', 'top_contact', 'top_surface_contact', 'ground_contact', 'air', 'support', 'bearing_verified', 'bearing_force_n')}
                for leg in ('FL', 'FR', 'RL', 'RR')}
            physical.append({'input_tick': tick, 'source_index': index, 'global_decision': row['global_policy_decision'],
                'phase': a['phase_id'], 'RR_gap_mm': 1000 * rr['clearance_m'],
                'RR_current_qualified': rr['current_lift_valid'], 'RR_lift_established': rr['lift_established'],
                'RR_legal_XY': bool(rr['within_top_xy'] and rr['within_lateral_span'] and rr['front_distance_m'] >= 0),
                'RR_front_mm': 1000 * rr['front_distance_m'], 'RR_air': rr['air'],
                'RR_current_TOP': bool(rr['top_contact'] and rr['top_surface_contact'] and rr['contact_surface'] == 'TOP'),
                'RR_placed_history': ev['history']['placed']['RR'], 'RR_current_placement_usable': _current_rr_placement_usable(ev),
                'old_Phi': old_phi, 'current_Phi': new_phi, 'X17_float32_changed': float(x[17]) != float(original[17]),
                'support': supports})
            previous = row
    assert len(xs) == 41 and physical[0]['input_tick'] == 8488 and physical[-1]['input_tick'] == 8808
    x = torch.stack(xs); target = torch.stack(targets); source_mu = torch.stack(source_means)
    sys.path.insert(0, str(OUT / 'front_mean_rehearsal_v3'))
    import execute_mean_rehearsal as method
    with redirect_stdout(io.StringIO()):
        runner = method.observation_runner(x[0], cm, device='cpu', inspect_copy=True)
    payload = torch.load(CURRENT, map_location='cpu', weights_only=False)
    runner.alg.actor.load_state_dict(payload['actor_state_dict'], strict=True)
    assert method.training.parameter_hash(runner.alg.actor) == cm['actor_parameter_sha256']
    with torch.no_grad():
        dist = method.kernel.distribution(runner.alg.actor, method.kernel.kernel.tensors(x, device='cpu'))
    caps = dist['caps']; request_target = caps * target.tanh()
    request_source_mean = caps * source_mu.tanh()
    errors = dist['request'] - request_target
    source_noise = request_source_mean - request_target
    channels = []
    for i, name in enumerate(NAMES):
        channels.append({'channel': name, 'REQUEST_unit': 'degree' if i < 8 else 'rad/s',
            'actual_source_raw_mean': float(target[:, i].mean()), 'actual_source_raw_min': float(target[:, i].min()),
            'actual_source_raw_max': float(target[:, i].max()), 'stored_source_mu_mean_provenance_only': float(source_mu[:, i].mean()),
            'current_mu_mean_on_reencoded_historical_inputs': float(dist['mean'][:, i].mean()),
            'current_sigma_mean': float(dist['sigma'][:, i].mean()),
            'current_vs_actual_raw_REQUEST_MAE': float(errors[:, i].abs().mean()),
            'current_vs_actual_raw_REQUEST_signed_mean': float(errors[:, i].mean()),
            'current_vs_actual_raw_REQUEST_max_abs': float(errors[:, i].abs().max()),
            'source_mu_vs_actual_raw_REQUEST_MAE_provenance_only': float(source_noise[:, i].abs().mean())})
    partitions = {'pre_capture_2s': list(range(30)), 'first_postplacement_AIR': [30], 'continuous_TOP10': list(range(31, 41))}
    groups = {}
    for name, indices in partitions.items():
        rows = [physical[i] for i in indices]; gaps = [r['RR_gap_mm'] for r in rows]
        groups[name] = {'rows': len(rows), 'ticks': [rows[0]['input_tick'], rows[-1]['input_tick']],
            'RR_gap_first_last_min_max_mm': [gaps[0], gaps[-1], min(gaps), max(gaps)],
            'RR_current_qualified_rows': sum(r['RR_current_qualified'] for r in rows),
            'RR_legal_XY_rows': sum(r['RR_legal_XY'] for r in rows), 'RR_current_TOP_rows': sum(r['RR_current_TOP'] for r in rows),
            'RR_current_placement_usable_rows': sum(r['RR_current_placement_usable'] for r in rows),
            'X17_changed_rows': sum(r['X17_float32_changed'] for r in rows),
            'current_REQUEST_MAE_full12': errors[indices].abs().mean(0).tolist(),
            'source_mu_to_actual_sample_REQUEST_MAE_full12_provenance_only': source_noise[indices].abs().mean(0).tolist(),
            'other_legs_bearing_verified_support_rows': {leg: sum(r['support'][leg]['support'] and r['support'][leg]['bearing_verified'] for r in rows) for leg in ('FL', 'FR', 'RL')},
            'other_legs_current_TOP_rows': {leg: sum(r['support'][leg]['top_contact'] and r['support'][leg]['top_surface_contact'] for r in rows) for leg in ('FL', 'FR', 'RL')}}
    result = {'schema': 'wlr50_clean.RR_last2s_fixed41_current_actor_readonly.v1', 'status': 'READONLY_INFORMATION_NOT_DATASET_ADMISSION',
        'current_checkpoint': str(CURRENT), 'current_sha256': CURRENT_SHA, 'rollout_path': str(batch_path),
        'source_rollout_sha256': sha(batch_path), 'source_indices': [1061, 1101], 'source_input_ticks': [8488, 8808],
        'global_decisions': [204838, 204878], 'groups': groups, 'channels': channels,
        'physical_rows': physical, 'derived_X17_copies_in_memory_only': True,
        'actual_raw_labels_kept_original': True, 'source_mu_is_provenance_not_label': True,
        'dataset_exported': False, 'gradients_or_optimization_run': False, 'budget_selected': False,
        'old_action_current_closed_loop_success_claimed': False,
        'limitations': ['Historical stochastic raw actions, not the stored source mean, are the actual executed labels.',
            'Thirty pre-contact AIR inputs are not automatically all positives merely because later contact occurred.',
            'Current actor was evaluated on separately reencoded historical inputs, not on the live/current trajectory.',
            'One short source trajectory and current policy-request errors do not prove an auxiliary update will generalize or fix rear capture.',
            'REQUEST is capped raw action, not final actuator target or a claim of current physical replay.']}
    (HERE / 'last_two_seconds_current_actor_readonly.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    rows = '\n'.join(f"| {c['channel']} | {c['actual_source_raw_mean']:.6g} | {c['stored_source_mu_mean_provenance_only']:.6g} | {c['current_mu_mean_on_reencoded_historical_inputs']:.6g} | {c['current_sigma_mean']:.6g} | {c['current_vs_actual_raw_REQUEST_MAE']:.6g} | {c['current_vs_actual_raw_REQUEST_signed_mean']:.6g} |" for c in channels)
    md = f'''# Last 2 seconds before RR capture + first TOP interval: 41-state read-only check

Exact input scope **8488–8808**, indices1061–1101/global204838–204878, entirely sealed rollout1566. The 2-second cutoff is8726−240=8486;8488 is the first existing8-tick input. No reset input was invented. Current actor is explicit CP214400 `{CURRENT_SHA}` on in-memory derived X389 with only Φ/index17 reencoded; raw targets remain the actual stored stochastic samples. No dataset, gradient, budget or fit is produced.

- Pre-contact30 inputs8488–8720: RR gap first→last **{groups['pre_capture_2s']['RR_gap_first_last_min_max_mm'][0]:.6g}→{groups['pre_capture_2s']['RR_gap_first_last_min_max_mm'][1]:.6g}mm**; min/max {groups['pre_capture_2s']['RR_gap_first_last_min_max_mm'][2:]}mm. Qualified rows {groups['pre_capture_2s']['RR_current_qualified_rows']}/30, legalXY {groups['pre_capture_2s']['RR_legal_XY_rows']}/30; currentTOP0. {groups['pre_capture_2s']['X17_changed_rows']} need changed X17. Per-row gap/support is retained in JSON, not assumed monotonic.
- Actual placement8726; first input after it8728 is AIR, not continued contact. Exact currentTOP input8736–8808 has10 rows and hands off throughP10/P11. After8808, previously audited8816 is AIR again; no longer contact run is claimed.
- Pre-contact verified support counts FL/FR/RL: {groups['pre_capture_2s']['other_legs_bearing_verified_support_rows']}; TOP10 counts {groups['continuous_TOP10']['other_legs_bearing_verified_support_rows']}. The JSON preserves classifications/forces, so body support is not inferred from limb position.

The following is a per-channel **fixed historical-state** overview. Raw/μ/σ are latent units. REQUEST error is current conditional mean's capped request minus the actual sampled action's capped request; first8 are degrees, wheels rad/s. Source μ is provenance only, never a replacement label.

| Channel | Actual raw mean | Stored source μ mean | Current μ mean | Current σ mean | REQUEST MAE | REQUEST signed mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{rows}

This short sequence contains real contact-and-continuation information and measurable current-versus-actual action differences. It is **not** blanket eligibility for every preceding AIR row, proof old actions would succeed from current states, or proof finite auxiliary learning will help. The immediate contact interruption and later interval boundary remain negative qualifications. Current natural evaluation takes priority; a real current RR capture could make this candidate unnecessary. CPU-only process exits after report creation.
'''
    (HERE / 'last_two_seconds_current_actor_readonly.md').write_text(md, encoding='utf-8')
    print(json.dumps({'groups': groups, 'RR_channels': channels[6:8], 'whole_body_channels': [channels[i] for i in (1, 3, 8, 11)], 'optimization_steps': 0}, indent=2))


if __name__ == '__main__':
    main()
