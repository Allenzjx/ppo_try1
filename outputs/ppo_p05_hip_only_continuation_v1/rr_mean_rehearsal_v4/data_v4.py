"""Fixed historical RR actual-raw data, with explicit current5fd X17 derivation.

Build/inspect only: no actor construction, optimizer, budget or checkpoint write.
Protection inputs have NO action labels; the caller must use current actor means.
"""
from collections import Counter
from copy import deepcopy
import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
ROOT = OUT.parents[1]
MANIFEST = HERE / 'data_manifest.json'
DATA = HERE / 'candidate_data.npz'
SCHEMA = 'wlr50_clean.RR_actual_raw_capture_continuation_data.v4'
HEAD = '5fd88852bf20c94cd74405c791a13a9fd9e0a3d8'
CONTENT = '992515c30a3ecb868198f38fb986b0a04d690557861e78d9595f8071a8fbc76e'
SOURCE = OUT / 'checkpoints/history/checkpoint_step_000214400.pt'
SOURCE_SHA = '8d0305626decff2214f8c3c0eee4031bdb791013d82a641a8ac4fc077ce235e9'
RUN03 = ROOT / 'runs/ppo_p05_hip_only_continuation_v1/train/20260922T0705260953371Z_ga802b24d78df_76c646571d794b4e96a82f69956928ff'
RUN08 = ROOT / 'runs/ppo_p05_hip_only_continuation_v1/train/20260922T1301289433343Z_g5fd88852bf20_337c83c1b0214e02b62aa98b579e2812'
COUNTERS = ('global_policy_decisions', 'ppo_updates', 'optimizer_steps')
FROZEN_V3_SHA = '4e016816a06a3ff743883faab3309d1058a7c53bcc8495a9a36761908caddf59'
FROZEN_P02_SHA = 'b336419401b5a34354cef070cdc095133343233c6af1df15cddebeea045c490d'
PROTECTION_PHASES = {1, 2, 3, 4, 5, 6, 7, 8, 12}


def require(value, message):
    if not bool(value):
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def binding(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def import_frozen(name, path, expected_sha):
    require(sha(path) == expected_sha, 'immutable historical reader changed')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def phase_groups(x):
    phase = np.asarray(x)[:, :13].argmax(-1) + 1
    return {f'P{p:02}': np.flatnonzero(phase == p).tolist() for p in sorted(set(phase.tolist()))}


def fixed_split(phases):
    """All513 rows, fixed interleaving within phase, never outcome selection."""
    phases = np.asarray(phases)
    require(Counter(phases.tolist()) == {9: 345, 10: 1, 11: 167}, 'fixed RR phase/source window changed')
    train, validation = [], []
    for phase in (9, 10, 11):
        ids = np.flatnonzero(phases == phase).tolist()
        for offset, index in enumerate(ids):
            (train if phase == 10 or offset % 3 in (0, 2) else validation).append(index)
    return np.asarray(sorted(train), dtype=np.int64), np.asarray(sorted(validation), dtype=np.int64)


def uniform_indices(indices, maximum=16):
    indices = np.asarray(indices, dtype=np.int64)
    require(len(indices) > 0 and maximum > 0, 'empty recent protection phase')
    return indices[np.linspace(0, len(indices) - 1, min(len(indices), maximum), dtype=np.int64)]


def validate_current_metadata(metadata, contract, reference):
    from wlr50_clean.ppo.semantic_migration import _contract
    require(_contract(contract) == metadata['runtime_contract'] == reference['runtime_contract'], 'exact current5fd runtime required')
    require(contract['source_git_commit'] == HEAD and contract['runtime_content_sha256'] == CONTENT, 'other potential/runtime semantics rejected')
    require(metadata['checkpoint_sha256'] == reference['checkpoint_sha256'] == SOURCE_SHA, 'only explicitly selected actual CP214400 is admitted')
    require(tuple(metadata[k] for k in COUNTERS) == (214400, 1640, 32800), 'source counters differ')
    require(Path(metadata['checkpoint_path']).resolve() == SOURCE.resolve(), 'source checkpoint path differs')
    for key in ('actor_parameter_sha256', 'policy_contract', 'normalization', 'normalizer_state_sha256',
                'task_conditioned_hip_wheel_branch', 'p05_capture_assist_migration', 'p05_capture_assist_branch',
                'capture_feedback_semantics_migration', 'capture_feedback_semantics_branch',
                'rr_postcross_workspace_migration', 'rr_postcross_workspace_branch'):
        require(metadata[key] == reference[key], 'current source or immutable AUX/semantic lineage differs: ' + key)
    ledger = metadata['rr_postcross_workspace_branch']['front_rehearsal_auxiliary']
    require(len(ledger['events']) == 3 and ledger['accepted_auxiliary_updates_total'] == 96
            and ledger['attempted_auxiliary_optimizer_steps_total'] == 96, 'all three real AUX events96/96 required')
    old_aux = metadata['task_conditioned_hip_wheel_branch']['auxiliary_mean_learning']
    return {'checkpoint': binding(SOURCE), 'actor_sha256': metadata['actor_parameter_sha256'],
        'PPO_counters': {k: metadata[k] for k in COUNTERS}, 'runtime_contract_sha256': digest(metadata['runtime_contract']),
        'entire_existing_front_AUX_ledger': {'full_object_sha256': digest(ledger),
            'accepted_auxiliary_updates_total': ledger['accepted_auxiliary_updates_total'],
            'attempted_auxiliary_optimizer_steps_total': ledger['attempted_auxiliary_optimizer_steps_total'],
            'events': [{'event_index': e['event_index'], 'kind': e['kind'], 'full_event_sha256': digest(e),
                'accepted': e['fit_report']['accepted_auxiliary_updates'],
                'attempted': e['fit_report']['attempted_auxiliary_optimizer_steps'],
                'source_checkpoint_sha256': e['source_checkpoint']['sha256']} for e in ledger['events']],
            'complete_object_verified_against_actual_source_sidecar': True},
        'old_limited_AUX_record': {'full_object_sha256': digest(old_aux),
            'accepted_auxiliary_updates_total': old_aux['accepted_auxiliary_updates_total'],
            'attempted_auxiliary_optimizer_steps_total': old_aux['attempted_auxiliary_optimizer_steps_total']},
        'counter_origins': {name: deepcopy(metadata[name + '_branch']['counter_origin'])
            for name in ('p05_capture_assist', 'capture_feedback_semantics', 'rr_postcross_workspace')},
        'immutable_migrations_sha256': {name: digest(metadata[name + '_migration'])
            for name in ('p05_capture_assist', 'capture_feedback_semantics', 'rr_postcross_workspace')}}


def check_row(batch, offset, row):
    a, native = row['applied_audit'], row['applied_audit']['actuator_target_effect_audit']
    x = batch['observations']['policy'][offset, 0]
    require(x.shape == (389,) and x.dtype == torch.float32 and torch.isfinite(x).all()
            and torch.equal(x, batch['observations']['critic'][offset, 0]), 'direct389 tensor invalid')
    require(int(x[:13].argmax()) == int(a['phase_id'][1:]) - 1, 'request phase differs')
    values = (batch['actions'][offset, 0], *(t[offset, 0] for t in batch['distribution_params']),
              batch['actions_log_prob'][offset, 0].reshape(()))
    for tensor, key in zip(values, ('raw_policy_action_full12', 'old_distribution_mean_full12',
                                  'old_distribution_std_full12', 'old_log_probability')):
        require(torch.equal(tensor, torch.tensor(row[key], dtype=tensor.dtype)), 'sealed raw/mean/std/logp mismatch')
    raw, mean, std, logp = values
    error = float((torch.distributions.Normal(mean, std).log_prob(raw).sum() - logp).abs())
    require(error <= 1e-5, 'source actual-raw Gaussian likelihood mismatch')
    p = row['policy_request']
    require(p['sampling_draws'] == 1 and p['extra_random_draws'] == 0
            and row['raw_policy_action_full12'] == p['selected_raw_full12'] == native['raw_policy_action_full12'], 'not original single sampled raw')
    require(native['verified'] and native['actual_mapping_matches_dispatch'] and native['phase_mask_full12'] == [1] * 12
            and a['actuator_target_effect_audit_summary']['all_ticks_verified'] and a['no_in_episode_state_writes_verified'], 'invalid/intervened execution')
    require(a['semantic_task']['physical_evaluator']['valid']
            and a['semantic_task']['physical_evaluator']['termination_reason'] is None and not row['terminal'], 'invalid/terminal positive or protection row')
    require(bool(batch['dones'][offset, 0]) == row['terminal']
            and torch.equal(batch['rewards'][offset, 0].reshape(()), torch.tensor(row['reward'], dtype=torch.float32)), 'sealed terminal/reward mismatch')
    for start, end, field in ((372, 384, 'capture_assist_observed_features'), (384, 389, 'capture_continuation_observed_features')):
        require(torch.equal(x[start:end], torch.tensor(p[field], dtype=torch.float32)), 'observable capture state differs')
    return x, raw, mean, std, logp, error


def build_positive(reference):
    from wlr50_clean.ppo.semantic_supervisor import TaskStageSupervisor
    migration = reference['rr_postcross_workspace_migration']
    require(sha(migration['plan_path']) == migration['plan_sha256'], 'RR semantics migration binding differs')
    factor = migration['rr_postcross_workspace_factor']
    require(factor['reward_changed'] and not any(factor[k] for k in
        ('policy_kernel_changed', 'sigma_changed', 'nominal_changed', 'physical_dynamics_changed', 'capture_assist_changed')), 'not reviewed single potential change')
    spec_path = ROOT / 'configs/ppo_p05_hip_only_continuation_v1/stage_task_spec.yaml'
    require(sha(spec_path) == reference['runtime_contract']['files'][spec_path.relative_to(ROOT).as_posix()], 'current spec changed')
    current = object.__new__(TaskStageSupervisor); current.spec = yaml.safe_load(spec_path.read_text())
    old = object.__new__(TaskStageSupervisor); old.spec = deepcopy(current.spec); old.spec.pop('rr_postcross_workspace_semantics')
    availability = read(OUT / 'rr_capture_candidate_availability_v1/availability_report.json')
    batches, records, arrays, previous = {}, [], [], None
    for update, b in availability['bindings'].items():
        path = Path(b['rollout_path'])
        require(sha(path) == b['rollout_sha256'], 'previously sealed source rollout differs')
        batch = torch.load(path, map_location='cpu', weights_only=False)
        require(batch['policy_contract'] == reference['policy_contract'] and batch['curriculum_epoch']['prefix_request'] is None, 'not original natural learner')
        require(batch['runtime_contract']['source_git_commit'] == migration['source_git_commit']
                and batch['runtime_contract']['runtime_content_sha256'] == migration['source_runtime_content_sha256'], 'source runtime differs')
        before, after = batch['runtime_contract']['files'], reference['runtime_contract']['files']
        delta = {name: {'before': before.get(name), 'after': after.get(name)}
                 for name in set(before) | set(after) if before.get(name) != after.get(name)}
        require(delta == migration['changed_file_hashes'], 'reviewed old/current file delta changed')
        require(sha(b['collection_source_checkpoint']) == b['source_checkpoint_sha256'], 'collecting checkpoint differs')
        batches[int(update)] = batch
    with (RUN03 / 'residual_and_projection_audit.jsonl').open('rb') as stream:
        for index, line in enumerate(itertools.islice(stream, 1260)):
            if index < 746:
                continue
            row = json.loads(line)
            if index == 746:
                previous = row
                continue
            a = row['applied_audit']; ev = previous['applied_audit']['semantic_task']['physical_evaluator']
            require(index in range(747, 1260) and row['global_policy_decision'] == 203777 + index
                    and a['decision_count'] == index + 1, 'fixed first-episode indexing differs')
            update, offset = 1558 + index // 128, index % 128
            x, raw, mean, std, logp, err = check_row(batches[update], offset, row)
            tick = a['physics_tick'] - a['physics_ticks']
            require(tick == 8 * index == ev['physics_tick'] and ev['valid'] and ev['termination_reason'] is None
                    and not previous['terminal'], 'input evaluator is not exactly aligned')
            owners = a['actuator_target_effect_audit']['capture_assist_evidence']['owner_indices']
            require(owners == [], 'positive RR window has assist-owned actions')
            old_phi, new_phi = old.physical_potential(ev), current.physical_potential(ev)
            require(abs(old_phi - a['reward_breakdown']['potential_before']) < 1e-12
                    and np.float32(old_phi) == x[17].item(), 'source X17 cannot be reproduced')
            derived = x.clone(); derived[17] = new_phi
            rr = ev['current_legs']['RR']
            records.append({'source_index': index, 'global_policy_decision': row['global_policy_decision'],
                'input_tick': tick, 'phase': a['phase_id'], 'episode_index': 0, 'rollout_update': update, 'rollout_offset': offset,
                'source_audit_row_sha256': hashlib.sha256(line).hexdigest(), 'old_phi': old_phi, 'current_phi': new_phi,
                'X17_float32_changed': bool(x[17] != derived[17]), 'RR_qualified_current': rr['current_lift_valid'],
                'RR_crossed_history': ev['history']['front_edge_crossed']['RR'], 'RR_placed_history': ev['history']['placed']['RR'],
                'RR_legal_TOP': bool(rr['top_contact'] and rr['top_surface_contact'] and rr['contact_surface'] == 'TOP'
                    and rr['within_top_xy'] and rr['within_lateral_span'] and not rr['ground_contact']),
                'RR_AIR': rr['air'], 'RR_ground': rr['ground_contact'], 'RR_gap_m': rr['clearance_m'],
                'capture_assist_owner_indices': owners, 'source_logp_recompute_error': err})
            arrays.append((x, derived, raw, mean, std, logp)); previous = row
    require(len(records) == 513 and records[0]['input_tick'] == 5976 and records[-1]['input_tick'] == 10072, 'fixed513 window changed')
    keys = ('source_observations389', 'observations389_current_X17', 'raw12_actual', 'source_mean12', 'source_sigma12', 'source_logp')
    result = {key: torch.stack([row[j] for row in arrays]).numpy() for j, key in enumerate(keys)}
    result.update(source_indices=np.arange(747, 1260, dtype=np.int64), input_ticks=np.arange(5976, 10073, 8, dtype=np.int64),
                  global_decisions=np.arange(204524, 205037, dtype=np.int64))
    return result, records, deepcopy(availability['bindings'])


def build_recent_protection(reference):
    manifest = read(RUN08 / 'training_manifest.json')
    require(manifest['lifecycle'] == 'SUCCEEDED' and manifest['actual_policy_decisions'] == 1024, 'block08 not sealed')
    batches = [torch.load(RUN08 / f'rollouts/rollout_{u:06}.pt', map_location='cpu', weights_only=False) for u in range(1633, 1641)]
    require(all(b['runtime_contract'] == reference['runtime_contract'] and b['policy_contract'] == reference['policy_contract'] for b in batches), 'recent protection not current semantics')
    xall = torch.cat([b['observations']['policy'][:, 0] for b in batches])
    phases = xall[:, :13].argmax(-1).numpy() + 1
    selected = sorted(i for p in (4, 5, 6) for i in uniform_indices(np.flatnonzero(phases == p)).tolist())
    records, prior, episode = [], None, -1
    with (RUN08 / 'residual_and_projection_audit.jsonl').open('rb') as stream:
        for index, line in enumerate(itertools.islice(stream, 1024)):
            row = json.loads(line); a = row['applied_audit']
            if a['decision_count'] == 1:
                episode += 1; prior = None
            if index in selected:
                x, _, _, _, _, err = check_row(batches[index // 128], index % 128, row)
                require(np.float32(a['reward_breakdown']['potential_before']) == x[17].item(), 'recent direct current X17 differs')
                require(row['global_policy_decision'] == 213377 + index and not a['prefix_checkpoint_policy_data_in_ppo_storage']
                        and not a['prefix_teacher_data_in_ppo_storage'], 'frozen prefix credited as learner protection')
                records.append({'source_index': index, 'global_policy_decision': row['global_policy_decision'],
                    'episode_index': episode, 'input_tick': a['physics_tick'] - a['physics_ticks'], 'phase': a['phase_id'],
                    'rollout_update': 1633 + index // 128, 'rollout_offset': index % 128,
                    'source_audit_row_sha256': hashlib.sha256(line).hexdigest(), 'source_logp_recompute_error': err,
                    'original_direct_current389': True, 'action_used_as_label': False,
                    'input_endpoint_available_in_learner_stream': prior is not None,
                    'prefix_actions_used_as_protection': False})
            prior = row
    require(len(records) == len(selected) == 34, 'recent protection rows differ')
    return xall[selected].numpy(), {'source_run': str(RUN08), 'selection': 'per phase source-order uniform linspace, at most16, no outcome selection',
        'source_phase_counts': {f'P{p:02}': int((phases == p).sum()) for p in (4, 5, 6)},
        'selected_phase_counts': dict(Counter(r['phase'] for r in records)), 'rows': records,
        'rollouts': [binding(RUN08 / f'rollouts/rollout_{u:06}.pt') for u in sorted({r['rollout_update'] for r in records})]}


def build_protection(reference):
    v3 = import_frozen('_RR_v4_frozen_v3_data', OUT / 'front_mean_rehearsal_v3/data_v3.py', FROZEN_V3_SHA)
    # Historical2-event metadata is used ONLY to read the already frozen probes.
    # It is never substituted for this operation's actual3-event CP214400.
    historical_path = OUT / 'checkpoints/history/checkpoint_step_000213376.pt'
    historical = read(historical_path.with_name(historical_path.stem + '_manifest.json'))
    historical['checkpoint_path'] = str(historical_path)
    require(historical['checkpoint_sha256'] == '8e5961e3e78260f12124447bff72a7459b7323d2da8f83cc467a59ddc9290e54'
            and sha(historical_path) == historical['checkpoint_sha256'], 'historical protection-reader reference differs')
    frozen = v3.load_reviewed_data(historical, historical['runtime_contract'])
    prior = frozen['protection_observations']
    ids = [i for i, p in enumerate(prior[:, :13].argmax(-1).tolist()) if p + 1 in {3, 4, 5, 6, 7, 8, 12}]
    require(len(ids) == 16, 'frozen protection filter changed')
    p02reader = import_frozen('_RR_v4_frozen_P02_reader', OUT / 'det_front_rehearsal_data_v1/read_candidate.py', FROZEN_P02_SHA)
    p02, p02receipt = p02reader.load_candidate(OUT / 'det_front_rehearsal_data_v1/candidate_manifest.json')
    recent, recent_receipt = build_recent_protection(reference)
    p01 = frozen['p01_deterministic_probe_observations'].numpy()
    combined = np.concatenate((p01, p02['X389_reconstructed'], prior[ids].numpy(), recent), axis=0)
    require(combined.shape == (305, 389), 'protection total changed')
    receipt = {'target_rule': 'detach the ACTUAL CP214400 conditional mean on each protection input; NO old actions returned as labels',
        'historical_reader_reference_only': binding(historical_path), 'actual_source_is_not_this_historical_reader_reference': True,
        'frozen_v3_receipt_sha256': digest(frozen['receipt']),
        'P01_field_reconstructed_one_row': {'rows': 1, 'independent_validation_rows': 0,
            'source_manifest': binding(OUT / 'det_P01_second_decision_v1/candidate_manifest.json')},
        'P02_field_reconstructed254': {'rows': 254, 'new_reconstruction_performed': False,
            'source_manifest': binding(OUT / 'det_front_rehearsal_data_v1/candidate_manifest.json'), 'frozen_receipt_sha256': digest(p02receipt),
            'includes84_previously_source_only_rows_as_unlabeled_protection': True},
        'existing_direct19_filtered': {'kept_indices': ids, 'kept_rows': 16, 'excluded_phases': ['P09', 'P10', 'P11'],
            'frozen_v3_manifest': binding(OUT / 'front_mean_rehearsal_v3/data_manifest.json')},
        'recent_block08': recent_receipt, 'combined_phase_counts': {k: len(v) for k, v in phase_groups(combined).items()},
        'separate_stochastic_P01_probes': 2, 'P01_probe_actions_not_labels': True, 'P13_actual_coverage': False,
        'global_phase_mean_invariance_claimed': False, 'current_trajectory_equivalence_claimed': False}
    return combined, frozen['p01_observations'].numpy(), receipt


def validate_arrays(a):
    for key in ('source_observations389', 'observations389_current_X17', 'raw12_actual', 'source_mean12', 'source_sigma12',
                'source_logp', 'protection_observations', 'p01_observations'):
        require(a[key].dtype == np.float32 and np.isfinite(a[key]).all(), 'nonfloat32/nonfinite data: ' + key)
    old, current = a['source_observations389'], a['observations389_current_X17']
    require(old.shape == current.shape == (513, 389), 'fixed positive shape differs')
    keep = np.arange(389) != 17
    require(np.array_equal(old[:, keep], current[:, keep]), 'unapproved observation reencoding outside X17')
    require(np.count_nonzero(old[:, 17] != current[:, 17]) == 237, 'expected current X17 derivation coverage differs')
    require(a['raw12_actual'].shape == a['source_mean12'].shape == a['source_sigma12'].shape == (513, 12)
            and a['source_logp'].shape == (513,), 'actual raw/provenance shape differs')
    require(np.all(a['source_sigma12'] > 0) and not np.any(np.all(a['raw12_actual'] == a['source_mean12'], axis=1)), 'actual stochastic labels replaced by stored mean')
    require(np.array_equal(a['source_indices'], np.arange(747, 1260)) and np.array_equal(a['input_ticks'], np.arange(5976, 10073, 8))
            and np.array_equal(a['global_decisions'], np.arange(204524, 205037)), 'fixed source window differs')
    train, validation = fixed_split(current[:, :13].argmax(-1) + 1)
    require(np.array_equal(train, a['train_indices']) and np.array_equal(validation, a['validation_indices']), 'fixed per-phase split altered')
    require(len(train) == 342 and len(validation) == 171 and not set(train) & set(validation), 'split overlap/count')
    require(a['protection_observations'].shape == (305, 389) and set(a['protection_observations'][:, :13].argmax(-1) + 1) == PROTECTION_PHASES,
            'protection phases/shape conflict with positive phase or omit real coverage')
    require(a['p01_observations'].shape == (2, 389) and np.all(a['p01_observations'][:, 0] == 1), 'P01 unlabeled probes changed')
    for x in (old, current, a['protection_observations'], a['p01_observations']):
        require(np.all((x[:, :13] == 0) | (x[:, :13] == 1)) and np.all(x[:, :13].sum(-1) == 1), 'phase is not exact onehot')


def build():
    require(not torch.cuda.is_available(), 'builder is CPU-only')
    torch.set_num_threads(1)
    require(not MANIFEST.exists() and not DATA.exists(), 'do not overwrite frozen v4 data')
    reference_path = SOURCE.with_name(SOURCE.stem + '_manifest.json')
    reference = read(reference_path); reference['checkpoint_path'] = str(SOURCE)
    require(sha(SOURCE) == SOURCE_SHA, 'actual source checkpoint SHA differs')
    source = validate_current_metadata(reference, reference['runtime_contract'], reference)
    arrays, rows, source_bindings = build_positive(reference)
    arrays['train_indices'], arrays['validation_indices'] = fixed_split(arrays['source_observations389'][:, :13].argmax(-1) + 1)
    arrays['protection_observations'], arrays['p01_observations'], protection = build_protection(reference)
    validate_arrays(arrays)
    np.savez_compressed(DATA, **arrays)
    old, new = arrays['source_observations389'][:, 17], arrays['observations389_current_X17'][:, 17]
    manifest = {'schema': SCHEMA, 'status': 'FIXED_OFFPOLICY_DATA_NOT_OPTIMIZATION_AUTHORIZATION',
        'source_checkpoint': source, 'source_manifest': binding(reference_path), 'dataset': binding(DATA), 'loader': binding(Path(__file__)),
        'runtime_head': HEAD, 'runtime_content_sha256': CONTENT, 'source_run': str(RUN03),
        'source_original_direct389': True, 'current_unmodified389': False,
        'labels': 'actually_executed_stochastic_raw12_not_stored_conditional_mean',
        'X17_migration': {'only_derived_column': 17, 'source_old_phi_matches_direct_float32': True,
            'current_phi_recomputed_from_exact_previous_endpoint': True, 'changed_rows': int(np.count_nonzero(old != new)),
            'float32_delta_min': float((new - old).min()), 'float32_delta_max': float((new - old).max()),
            'existing_migration_plan_sha256': reference['rr_postcross_workspace_migration']['plan_sha256'],
            'new_physical_trajectory_or_current_actor_equivalence_claimed': False},
        'selection': {'source_indices_inclusive': [747, 1259], 'input_ticks_inclusive': [5976, 10072], 'all12_raw_channels': True,
            'source_rows': 513, 'train_rows': 342, 'validation_rows': 171,
            'train_phase_counts': {k: len(v) for k, v in phase_groups(arrays['observations389_current_X17'][arrays['train_indices']]).items()},
            'validation_phase_counts': {k: len(v) for k, v in phase_groups(arrays['observations389_current_X17'][arrays['validation_indices']]).items()},
            'rule': 'Within P09/P11 source-order offsets0,2 mod3 train; offset1 validation. P10 singleton train only. No outcome/action selection.',
            'P10_independent_validation_rows': 0, 'same_episode_correlated_split': True,
            'P12_and_later_RL_failure_excluded': True, 'all_AIR_rows_individually_successful_claimed': False},
        'physical_coverage': {'RR_actual_placement_event_tick': 8726, 'first_P10_input_tick': 8736, 'first_P11_input_tick': 8744,
            'current_qualified_inputs': sum(r['RR_qualified_current'] for r in rows), 'placed_history_inputs': sum(r['RR_placed_history'] for r in rows),
            'legal_TOP_inputs': sum(r['RR_legal_TOP'] for r in rows), 'AIR_inputs': sum(r['RR_AIR'] for r in rows),
            'ground_inputs': sum(r['RR_ground'] for r in rows), 'current_contact_not_continuous': True,
            'assist_owned_labels': 0, 'earlier_FL_preparation_pure_policy_claimed': False},
        'source_collection_bindings': source_bindings, 'positive_rows': rows, 'protection': protection,
        'source_boundary_full_RNG_present': True, 'per_decision_RNG_replay_proven': False,
        'source_mean_sigma_logp_are_provenance_not_current_onpolicy': True,
        'old_reward_GAE_returns_used_as_targets': False, 'protection_targets_in_dataset': False,
        'fit_executed': False, 'budget_selected': False, 'AUX_updates_added': 0, 'PPO_decisions_added': 0, 'PPO_updates_added': 0,
        'teacher_deployed': False, 'full_task_success_claimed': False,
        'limitations': ['One historical stochastic trajectory; interleaved validation is correlated, not independent closed-loop evidence.',
            'P10 has one training row and zero validation; no P13 actual protection coverage.',
            'Fixed-state current X17 reencoding does not imply the old raw action would succeed from a current learner state.',
            'Raw labels include actual stochastic innovation and whole-body coupling; stored source mean is not an executed label.',
            'Protection means are current-source targets, not historical action labels or a global phase invariance guarantee.',
            'Capture history is not uninterrupted TOP contact; retain AIR/current qualification variation.']}
    MANIFEST.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


def load_reviewed_data(metadata, contract):
    m = read(MANIFEST)
    require(m['schema'] == SCHEMA and m['status'] == 'FIXED_OFFPOLICY_DATA_NOT_OPTIMIZATION_AUTHORIZATION', 'unsupported RR data schema')
    for name in ('source_manifest', 'dataset', 'loader'):
        require(sha(m[name]['path']) == m[name]['sha256'], 'frozen v4 binding changed: ' + name)
    require(Path(m['dataset']['path']).resolve() == DATA.resolve() and Path(m['loader']['path']).resolve() == Path(__file__).resolve(), 'v4 bound paths differ')
    reference = read(m['source_manifest']['path'])
    source = validate_current_metadata(metadata, contract, reference)
    require(source == m['source_checkpoint'], 'source three-event ledger/origins drift')
    require(m['X17_migration']['existing_migration_plan_sha256'] == metadata['rr_postcross_workspace_migration']['plan_sha256']
            and sha(metadata['rr_postcross_workspace_migration']['plan_path']) == m['X17_migration']['existing_migration_plan_sha256'], 'potential version binding differs')
    with np.load(DATA, allow_pickle=False) as loaded:
        a = {key: loaded[key].copy() for key in loaded.files}
    validate_arrays(a)
    t = lambda x: torch.from_numpy(x.copy())
    train, validation = a['train_indices'], a['validation_indices']
    data = {'train_observations': t(a['observations389_current_X17'][train]), 'train_raw_targets': t(a['raw12_actual'][train]),
        'validation_observations': t(a['observations389_current_X17'][validation]), 'validation_raw_targets': t(a['raw12_actual'][validation]),
        'train_phase_groups': phase_groups(a['observations389_current_X17'][train]),
        'validation_phase_groups': phase_groups(a['observations389_current_X17'][validation]),
        'protection_observations': t(a['protection_observations']), 'p01_observations': t(a['p01_observations'])}
    receipt = {key: deepcopy(m[key]) for key in ('schema', 'source_checkpoint', 'source_original_direct389', 'current_unmodified389',
        'labels', 'X17_migration', 'selection', 'physical_coverage', 'source_collection_bindings', 'protection', 'limitations')}
    receipt.update(schema=SCHEMA + '.loaded', data_manifest=binding(MANIFEST), dataset=m['dataset'], loader=m['loader'],
        tensor_sha256={key: tensor_sha(value) for key, value in data.items() if torch.is_tensor(value)},
        offpolicy_historical_supervised_only_not_PPO=True, current_source_checkpoint_exactly_bound=True,
        protection_targets_must_be_current_actor_means=True, current_runtime_only=HEAD,
        optimization_authorized_by_loader=False, optimizer_steps_performed=0, AUX_updates_added=0, PPO_decisions_added=0, PPO_updates_added=0)
    receipt['receipt_content_sha256'] = digest(receipt)
    data['receipt'] = receipt
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--read-report', type=Path)
    args = parser.parse_args()
    require(not torch.cuda.is_available(), 'standalone data preparation is CPU-only')
    torch.set_num_threads(1)
    if args.build:
        build()
    from wlr50_clean.ppo.semantic_migration import checkpoint_metadata
    from wlr50_clean.ppo.semantic_cli import runtime_contract
    metadata = checkpoint_metadata(SOURCE)
    contract = runtime_contract(expected_head=HEAD, semantic_version='v3', experiment_id='p05_hip_only_continuation_v1')
    data = load_reviewed_data(metadata, contract)
    if args.read_report:
        path = args.read_report.resolve()
        require(path.is_relative_to(HERE) and not path.exists(), 'read report must be fresh inside v4 output directory')
        path.write_text(json.dumps({'schema': SCHEMA + '.readonly_test', 'result': 'PASS',
            'manifest': binding(MANIFEST), 'data_receipt_sha256': digest(data['receipt']),
            'source_checkpoint_sha256': SOURCE_SHA,
            'shapes': {k: list(v.shape) for k, v in data.items() if torch.is_tensor(v)},
            'fit_executed': False, 'checkpoint_written': False}, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'read_only_data': 'PASS', 'shapes': {k: list(v.shape) for k, v in data.items() if torch.is_tensor(v)},
        'train_counts': {k: len(v) for k, v in data['train_phase_groups'].items()},
        'validation_counts': {k: len(v) for k, v in data['validation_phase_groups'].items()},
        'source_SHA': SOURCE_SHA, 'dataset_SHA': sha(DATA), 'manifest_SHA': sha(MANIFEST),
        'fit_executed': False, 'checkpoint_written': False}, indent=2))


if __name__ == '__main__':
    main()
