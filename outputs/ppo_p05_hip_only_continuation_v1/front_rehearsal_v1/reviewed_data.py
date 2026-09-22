"""Exact sealed block03 first-episode observations; no video reconstruction."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / 'runs/ppo_p05_hip_only_continuation_v1/train/20260922T0705260953371Z_ga802b24d78df_76c646571d794b4e96a82f69956928ff'
TRAIN = [0, 1, *range(2, 280, 3)]
VALIDATION = list(range(3, 280, 3))
INVARIANCE = [*range(280, 284), 284, *range(285, 289), *range(461, 465)]
SCHEMA = 'wlr50_clean.front_rehearsal_exact_block03_selection.v1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def tensor_sha(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def selection_indices():
    require(len(TRAIN) == 95 and len(VALIDATION) == 93 and len(INVARIANCE) == 13,
            'fixed reviewed selection changed')
    require(not set(TRAIN) & set(VALIDATION) and not set(INVARIANCE) & set(TRAIN + VALIDATION),
            'training and validation/holdout overlap')
    return TRAIN.copy(), VALIDATION.copy(), INVARIANCE.copy()


def validate_tensor_row(observation, action, mean, std, logp, row, *, index):
    """Exact original float32 join, not an independently reconstructed policy."""
    a = row['applied_audit']; p = row['policy_request']; n = a['actuator_target_effect_audit']
    require(row['global_policy_decision'] == 203777 + index and a['decision_count'] == index + 1,
            'data left first natural episode or global decision sequence')
    require(observation.shape == (389,) and bool(torch.isfinite(observation).all())
            and bool((observation.abs() <= 20).all()), 'not an exact finite 389 observation')
    require(float(observation[:13].sum()) == 1. and bool(((observation[:13] == 0) | (observation[:13] == 1)).all())
            and int(observation[:13].argmax()) + 1 == int(a['phase_id'][1:]), 'phase one-hot mismatch')
    for tensor, key in ((action, 'raw_policy_action_full12'), (mean, 'old_distribution_mean_full12'),
                        (std, 'old_distribution_std_full12'), (logp, 'old_log_probability')):
        expected = torch.tensor(row[key], dtype=tensor.dtype).reshape(tensor.shape)
        require(torch.equal(tensor, expected), 'sealed tensor/audit mismatch: ' + key)
    require(action.shape == mean.shape == std.shape == (12,) and bool((std > 0).all()), 'invalid Gaussian')
    require(all(bool(torch.isfinite(x).all()) for x in (action, mean, std, logp)), 'nonfinite source')
    require(row['raw_policy_action_full12'] == p['selected_raw_full12'] == n['raw_policy_action_full12']
            and row['old_distribution_mean_full12'] == p['conditional_mean_full12']
            and row['old_distribution_std_full12'] == p['effective_sigma_full12'], 'raw/distribution provenance mismatch')
    require(p['sampling_draws'] == 1 and p['extra_random_draws'] == 0, 'source was not one true raw draw')
    require(torch.equal(observation[195:207], torch.tensor(p['previous_raw_from_current_observation_full12'], dtype=torch.float32)),
            'source HISTORY differs from original input')
    require(torch.equal(observation[372:384], torch.tensor(p['capture_assist_observed_features'], dtype=torch.float32))
            and torch.equal(observation[384:389], torch.tensor(p['capture_continuation_observed_features'], dtype=torch.float32)),
            'source assist/pending observations differ')
    require(n['verified'] and n['actual_mapping_matches_dispatch'] and n['phase_mask_full12'] == [1] * 12
            and a['actuator_target_effect_audit_summary']['all_ticks_verified']
            and a['no_in_episode_state_writes_verified'] and not row['terminal'], 'invalid physical/native source')
    ev = a['semantic_task']['physical_evaluator']
    require(ev['valid'] is True and ev.get('termination_reason') is None,
            'source evaluator reports invalidity or physical termination')
    if index < 280:
        require(a['phase_id'] in ('P01', 'P02') and n['capture_assist_evidence']['owner_indices'] == [],
                'front rehearsal must not contain capture-assist actions')
    else:
        require(a['phase_id'] in ('P03', 'P04', 'P05', 'P06'), 'holdout left the declared continuation window')
    error = float((torch.distributions.Normal(mean, std).log_prob(action).sum() - logp.reshape(())).abs())
    require(error <= 1e-5, 'stored raw Gaussian likelihood mismatch')
    return error


def verify_rr_boundary(metadata, contract, source_contract):
    """Only the existing immutable RR-potential boundary, never a generic waiver."""
    from wlr50_clean.ppo.semantic_migration import validate_migration_plan
    record = metadata['rr_postcross_workspace_migration']
    verified = validate_migration_plan(Path(record['source_checkpoint']), contract, Path(record['plan_path']))
    require(verified == record and record['source_contract_sha256'] == digest(source_contract)
            and record['target_contract_sha256'] == digest(contract), 'wrong source/target RR migration binding')
    factor = record['rr_postcross_workspace_factor']
    require(factor['policy_kernel_changed'] is False and factor['capture_assist_changed'] is False
            and factor['observation_shape_changed'] is False
            and factor['observation_semantics_changed'][0]['index'] == 17, 'unreviewed data semantics boundary')
    return {'mode': 'verified_existing_RR_potential_boundary_with_per_state_zero_delta',
            'plan_path': record['plan_path'], 'plan_sha256': record['plan_sha256'],
            'source_contract_sha256': record['source_contract_sha256'],
            'target_contract_sha256': record['target_contract_sha256']}


def verify_potential(observations, rows, spec):
    from wlr50_clean.ppo.semantic_supervisor import TaskStageSupervisor, _current_rr_receiver_preparation_retired
    new = object.__new__(TaskStageSupervisor); new.spec = deepcopy(spec)
    old = object.__new__(TaskStageSupervisor); old.spec = deepcopy(spec)
    old.spec.pop('rr_postcross_workspace_semantics')
    max_difference = 0.
    for i, (obs, row) in enumerate(zip(observations, rows, strict=True)):
        before = row['applied_audit']['reward_breakdown']['potential_before']
        require(obs[17].item() == torch.tensor(before, dtype=torch.float32).item(), 'stored index17 differs from actual reward pre-state')
        if i == 0:
            # No full reset evaluator was saved. Do not invent measured geometry.
            # Exact stored history bits are sufficient to disprove the only new
            # potential branch for every possible reset geometry/contact state.
            require(obs[149].item() == obs[153].item() == obs[157].item() == 0., 'reset has RR progress')
            necessary_only = {'valid': True, 'termination_reason': None,
                'history': {'active_lift': {'RR': False}, 'front_edge_crossed': {'RR': False}},
                'current_legs': {'RR': {}}}
            require(not _current_rr_receiver_preparation_retired(spec, 'RR', necessary_only),
                    'reset RR zero-history implication failed')
            continue
        previous = rows[i - 1]['applied_audit']
        require(previous['decision_count'] + 1 == row['applied_audit']['decision_count'], 'pre-state endpoint is not adjacent')
        ev = previous['semantic_task']['physical_evaluator']
        old_phi, new_phi = old.physical_potential(ev), new.physical_potential(ev)
        difference = abs(new_phi - old_phi); max_difference = max(max_difference, difference)
        require(difference == 0. and abs(old_phi - before) < 1e-12
                and torch.tensor(new_phi, dtype=torch.float32).item() == obs[17].item(),
                'old/new physical potential is not identical on this source input')
        require(not _current_rr_receiver_preparation_retired(spec, 'RR', ev), 'RR retirement active in front/holdout data')
    return {'full_adjacent_evaluator_recomputed_inputs': len(rows) - 1,
            'reset_input_count': 1, 'reset_proof': 'exact sealed Phi plus zero qualified/cross/placed; pure new-branch impossibility, not a reconstructed full reset evaluator',
            'maximum_new_minus_old_potential_abs': max_difference,
            'all_stored_float32_index17_unchanged': True}


def load_reviewed_data(metadata, contract):
    train, validation, holdout = selection_indices()
    manifest = read(RUN / 'training_manifest.json')
    require(manifest['lifecycle'] == 'SUCCEEDED' and manifest['stage'] == 'full_episode'
            and manifest['actual_policy_decisions'] == 2048 and not manifest['phase_suffix_curriculum_implemented']
            and not (RUN / 'prefix_evidence.jsonl').exists(), 'source is not sealed natural-P01 block03')
    rows = []
    with (RUN / 'residual_and_projection_audit.jsonl').open(encoding='utf-8') as stream:
        for _ in range(465):
            rows.append(json.loads(next(stream)))
    observations, actions, means, stds, logps = [], [], [], [], []
    source_files = {}
    source_contract = None
    for update in range(1558, 1562):
        path = RUN / f'rollouts/rollout_{update:06}.pt'
        data = torch.load(path, map_location='cpu', weights_only=False)
        obs = data['observations']['policy']; raw = data['actions']; mu, sigma = data['distribution_params']
        require(data['schema'] == 'wlr50_clean.semantic_on_policy_rollout.v1'
                and obs.shape == (128, 1, 389) and raw.shape == (128, 1, 12)
                and torch.equal(obs, data['observations']['critic']), 'unrecognized sealed storage')
        if source_contract is None: source_contract = data['runtime_contract']
        require(data['runtime_contract'] == source_contract and data['policy_contract'] == metadata['policy_contract']
                and data['curriculum_epoch']['prefix_request'] is None, 'rollout source policy/runtime/prefix mismatch')
        observations.extend(obs[:, 0]); actions.extend(raw[:, 0]); means.extend(mu[:, 0]); stds.extend(sigma[:, 0])
        logps.extend(data['actions_log_prob'][:, 0])
        source_files[path.relative_to(RUN).as_posix()] = {'path': str(path), 'sha256': sha(path)}
    x, y, mu, std, lp = [torch.stack(v[:465]) for v in (observations, actions, means, stds, logps)]
    likelihood_error = max(validate_tensor_row(x[i], y[i], mu[i], std[i], lp[i], row, index=i) for i, row in enumerate(rows))
    require([rows[i]['applied_audit']['phase_id'] for i in range(280)] == ['P01'] * 2 + ['P02'] * 278,
            'first-episode front window changed')
    require(all(x[i, 0].item() == x[i, 1].item() == 0. for i in holdout), 'holdout uses supervised phase')
    for index, leg, expected_tick in ((283, 'FR', 2268), (460, 'FL', 3687)):
        ev = rows[index]['applied_audit']['semantic_task']['physical_evaluator']
        contact = ev['current_legs'][leg]
        require(ev['valid'] and not ev['termination_reason'] and ev['history']['placed'][leg]
                and ev['history']['event_ticks']['placed'][leg] == expected_tick
                and contact['top_contact'] and contact['top_surface_contact'] and contact['contact_surface'] == 'TOP'
                and contact['within_top_xy'] and contact['within_lateral_span'] and not contact['ground_contact']
                and contact['bearing_verified'] and contact['bearing_force_n'] > 0.,
                'actual later legal TOP bearing contact qualification is missing')
    compatibility = verify_rr_boundary(metadata, contract, source_contract)
    spec_path = ROOT / 'configs/ppo_p05_hip_only_continuation_v1/stage_task_spec.yaml'
    require(sha(spec_path) == contract['files'][spec_path.relative_to(ROOT).as_posix()], 'current task bytes differ')
    potential = verify_potential(x, rows, yaml.safe_load(spec_path.read_text(encoding='utf-8')))
    for name in ('training_manifest.json', 'run_manifest.json', 'residual_and_projection_audit.jsonl'):
        source_files[name] = {'path': str(RUN / name), 'sha256': sha(RUN / name)}
    groups = {}
    for name, ids in (('train', train), ('validation', validation), ('invariance', holdout)):
        groups[name] = {'indices': ids, 'global_decisions': [203777 + i for i in ids],
            'observations_float32_le_sha256': tensor_sha(x[ids]), 'raw_targets_float32_le_sha256': tensor_sha(y[ids]),
            'recorded_mean_float32_le_sha256': tensor_sha(mu[ids]), 'count': len(ids)}
    receipt = {'schema': SCHEMA, 'run': str(RUN), 'episode_index': 0, 'source_files': source_files,
        'source_runtime_contract': source_contract, 'source_policy_contract': metadata['policy_contract'],
        'data_runtime_compatibility': compatibility, 'potential_semantics_verification': potential,
        'groups': groups, 'front_source_count': 280, 'front_all_original_rows_retained': True,
        'unselected_front_indices': [i for i in range(280) if i not in train and i not in validation],
        'raw_targets_unmodified': True, 'stored_mean_is_diagnostic_not_executed_target': True,
        'label_scope': 'first_episode_real_FR_front_segment_with_later_contact_and_continuation_not_whole_episode_success',
        'local_qualification': {'FR_placed': {'global_decision': 204060, 'physics_tick': 2268, 'assist_owned': False},
            'FL_placed_and_P06': {'global_decision': 204237, 'physics_tick': 3687, 'assist_owner_indices': [0, 1]}},
        'source_episode_later_result': 'P12_INCOMPLETE_CONTROLLER_BLOCKED',
        'excluded_from_positive_labels': 'later rear failure region; episode1 failure; episode2 partial',
        'invariance_holdout_phase_counts': {p: sum(rows[i]['applied_audit']['phase_id'] == p for i in holdout) for p in ('P03', 'P04', 'P05', 'P06')},
        'real_P07_to_P13_holdout_coverage_claimed': False,
        'normal_log_probability_max_abs_error': likelihood_error,
        'front_raw_minus_recorded_mean_max_abs': float((y[:280] - mu[:280]).abs().max()),
        'teacher_deployed': False, 'new_PPO_credit': 0, 'new_auxiliary_credit': 0}
    receipt['receipt_content_sha256'] = digest(receipt)
    return {'train_observations': x[train], 'train_raw_targets': y[train],
            'validation_observations': x[validation], 'validation_raw_targets': y[validation],
            'invariance_observations': x[holdout], 'receipt': receipt}
