"""Default CPU inspection; explicit separately authorized finite AUX only.

No simulator import, no source checkpoint overwrite, no latest pointer. This
does not make old raw labels on-policy and never adds PPO learning credit.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

import torch
from tensordict import TensorDict

import front_rehearsal as kernel
from reviewed_data import ROOT, digest, load_reviewed_data, read, require, sha
from wlr50_clean.ppo import semantic_cli, semantic_migration, semantic_training as training
from wlr50_clean.ppo.semantic_p05_capture_profile import P05_CAPTURE_POLICY, P05_CAPTURE_OBSERVATION_LAYOUT

HERE = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/ppo_p05_hip_only_continuation_v1'
LEDGER_KEY = 'front_rehearsal_auxiliary'
LEDGER_SCHEMA = 'wlr50_clean.front_rehearsal_auxiliary.v1'
COUNTERS = ('global_policy_decisions', 'ppo_updates', 'optimizer_steps')
PARAMETERS = ['actor.mlp.0.weight[:,0:2]']


def helpers():
    return {name: sha(HERE / name) for name in ('front_rehearsal.py', 'reviewed_data.py', 'rehearsal_cli.py')}


def output_path(path):
    path = Path(path).resolve()
    require(path.is_relative_to(OUT.resolve()) and not path.exists(), 'use a new path in the current experiment outputs')
    return path


def observation_runner(observation, metadata, *, device):
    """No physics: construct the exact current policy/return profile on a stated device."""
    class ObservationOnly:
        num_envs, num_actions = 1, 12
        cfg = {'evaluation': True, 'semantic_version': 'v3'}
        def get_observations(self):
            x = torch.as_tensor(observation, dtype=torch.float32, device=device).reshape(1, 389)
            return TensorDict({'policy': x, 'critic': x.clone()}, batch_size=[1], device=device)
    runner, _ = training.construct_semantic_runner(ObservationOnly(), seed=metadata['seed'], device=device,
        initialize_actor=False, policy_version=P05_CAPTURE_POLICY,
        observation_layout=P05_CAPTURE_OBSERVATION_LAYOUT)
    expected_config = deepcopy(metadata['runner_config']); expected_config['device'] = device
    require(runner._semantic_runner_config == expected_config, 'constructed runner differs beyond explicitly stated inspection device')
    require(training._runner_policy_contract(runner) == metadata['policy_contract'], 'constructed current389 contract differs')
    return runner


def fit_arguments(data, device):
    return (kernel.tensors(data['train_observations'], device=device), data['train_raw_targets'].to(device),
            kernel.tensors(data['validation_observations'], device=device), data['validation_raw_targets'].to(device),
            kernel.tensors(data['invariance_observations'], device=device))


def cpu_inspection(checkpoint, metadata, data):
    require(not torch.cuda.is_available(), 'default inspection requires CUDA_VISIBLE_DEVICES=-1')
    before_rng = training.capture_training_rng_state(seed=metadata['seed'])
    try:
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        expected = {k: v for k, v in metadata.items() if k not in ('checkpoint_path', 'checkpoint_sha256', 'save_load_round_trip')}
        require(payload['infos'] == expected and training.state_hash(payload['optimizer_state_dict']) == metadata['optimizer_state_sha256'],
                'source embedded metadata/Adam integrity mismatch')
        runner = observation_runner(data['train_observations'][0], metadata, device='cpu')
        runner.alg.actor.load_state_dict(payload['actor_state_dict'], strict=True)
        require(training.parameter_hash(runner.alg.actor) == metadata['actor_parameter_sha256'], 'read-only current actor copy mismatch')
        runner.alg.actor.eval()
        before = training.parameter_hash(runner.alg.actor)
        result = kernel.inspect(runner.alg.actor, *fit_arguments(data, 'cpu'))
        require(training.parameter_hash(runner.alg.actor) == before, 'inspection mutated actor')
        return {**result, 'load_semantics': 'CPU actor-state copy only, not official device-relocated resume',
                'optimizer_loaded_or_stepped': False, 'checkpoint_written': False}
    finally:
        training.restore_training_rng_state(before_rng, expected_seed=metadata['seed'])


def append_ledger(infos, *, report, data_receipt, source_checkpoint, helper_sha256, budget):
    """New current-branch event; do not rewrite any historical AUX/migration fields."""
    require(type(report.get('accepted_auxiliary_updates')) is int and report['accepted_auxiliary_updates'] > 0,
            'no accepted AUX update; no learned checkpoint may be saved')
    require(type(report.get('attempted_auxiliary_optimizer_steps')) is int
            and report['accepted_auxiliary_updates'] <= report['attempted_auxiliary_optimizer_steps'] <= 32,
            'invalid independent AUX counts')
    result = deepcopy(infos)
    branch = result['rr_postcross_workspace_branch']
    require(branch['schema'] == 'wlr50_clean.rr_postcross_workspace_same389.v1'
            and branch['semantics'] == 'current_qualified_RR_over_top_receiver_retirement_v1'
            and branch['migration_added_updates'] == 0, 'current RR branch lineage missing')
    ledger = deepcopy(branch.get(LEDGER_KEY, {'schema': LEDGER_SCHEMA, 'events': []}))
    require(ledger['schema'] == LEDGER_SCHEMA and isinstance(ledger['events'], list), 'wrong rehearsal ledger schema')
    require([e['event_index'] for e in ledger['events']] == list(range(1, len(ledger['events']) + 1)),
            'rehearsal event indices are not sequential')
    event = {'event_index': len(ledger['events']) + 1,
        'kind': 'finite_supervised_raw_action_phase_columns_not_PPO',
        'source_checkpoint': deepcopy(source_checkpoint), 'helper_sha256': deepcopy(helper_sha256),
        'data_receipt': deepcopy(data_receipt), 'fit_report': deepcopy(report), 'fit_report_sha256': digest(report),
        'budget': deepcopy(budget), 'optimized_parameters': PARAMETERS.copy(),
        'target_semantics': 'actual_logged_raw_samples_not_stored_mean_or_final_target',
        'phase_scope': ['P01', 'P02'], 'mean_and_log_sigma_may_change': True,
        'physical_success_claimed': False, 'teacher_deployed': False,
        'PPO_counters_unchanged': {k: infos[k] for k in COUNTERS},
        'stage_requested_decisions_unchanged': deepcopy(infos['stage_requested_decisions']),
        'PPO_decisions_added': 0, 'PPO_updates_added': 0, 'PPO_optimizer_steps_added': 0}
    # The PT payload and JSON sidecar must contain the same new JSON value;
    # dataclass tuple bounds otherwise become lists only in the sidecar.
    event = json.loads(json.dumps(event, allow_nan=False))
    ledger['events'].append(event)
    ledger['accepted_auxiliary_updates_total'] = sum(e['fit_report']['accepted_auxiliary_updates'] for e in ledger['events'])
    ledger['attempted_auxiliary_optimizer_steps_total'] = sum(e['fit_report']['attempted_auxiliary_optimizer_steps'] for e in ledger['events'])
    ledger['training_lineage_label'] = 'PPO_plus_explicit_phase_column_raw_action_rehearsal'
    branch[LEDGER_KEY] = ledger
    # The original branch keys and every other top-level metadata value remain
    # exact. Ordinary training's whole-branch carry retains this nested ledger.
    restored = deepcopy(result); restored['rr_postcross_workspace_branch'] = deepcopy(infos['rr_postcross_workspace_branch'])
    require(restored == infos, 'rehearsal changed protected historical metadata')
    return result


def save_auxiliary_checkpoint(runner, path, infos, *, report, data, source_checkpoint, helper_sha256, budget):
    path = output_path(path)
    require(path.name.startswith('checkpoint_aux_frontrehearsal_') and path.suffix == '.pt'
            and not path.with_name(path.stem + '_manifest.json').exists(), 'use a unique explicit front-rehearsal AUX checkpoint')
    require(runner.alg.storage.step == 0 and runner.alg.transition.actions is None, 'AUX save requires an empty fresh rollout')
    require(infos['actor_parameter_sha256'] == report['actor_parameter_sha256_before']
            and training.parameter_hash(runner.alg.actor) == report['actor_parameter_sha256_after'], 'AUX actor report mismatch')
    for key, actual in (('critic_parameter_sha256', training.parameter_hash(runner.alg.critic)),
                        ('optimizer_state_sha256', training.state_hash(runner.alg.optimizer.state_dict())),
                        ('normalizer_state_sha256', training.state_hash(training._normalizers(runner)))):
        require(infos[key] == actual, 'AUX altered protected state: ' + key)
    require(infos['optimizer_learning_rate'] == training.optimizer_learning_rate(runner)
            and infos['runner_config'] == runner._semantic_runner_config
            and infos['training_rng_state'] == training.capture_training_rng_state(seed=infos['seed']),
            'AUX changed effective LR/config/RNG')
    payload = append_ledger(infos, report=report, data_receipt=data['receipt'],
        source_checkpoint=source_checkpoint, helper_sha256=helper_sha256, budget=budget)
    payload['stage'] = 'auxiliary_phase_column_rehearsal_not_PPO'
    pair = training.save_semantic_checkpoint(runner, path, payload)
    # A distinct constructed runner and official loader, not only an in-place
    # roundtrip. Use the original device and real CUDA visibility.
    metadata = semantic_migration.checkpoint_metadata(pair[0])
    fresh = observation_runner(data['train_observations'][0], metadata, device=metadata['runner_config']['device'])
    loaded = training.load_semantic_checkpoint(fresh, pair[0], contract=metadata['runtime_contract'], seed=metadata['seed'])
    for key in COUNTERS:
        require(loaded[key] == infos[key], 'AUX invented PPO credit')
    require(loaded['rr_postcross_workspace_branch'] == payload['rr_postcross_workspace_branch']
            and fresh.alg.storage.step == 0 and fresh.alg.transition.actions is None, 'independent AUX ledger/empty-storage reload failed')
    for key in infos:
        if key.endswith(('_branch', '_migration')) and key != 'rr_postcross_workspace_branch':
            require(loaded[key] == infos[key], 'independent reload changed historical branch/migration: ' + key)
    require(training.parameter_hash(fresh.alg.actor) == report['actor_parameter_sha256_after']
            and training.state_hash(fresh.alg.optimizer.state_dict()) == infos['optimizer_state_sha256']
            and training.capture_training_rng_state(seed=infos['seed']) == infos['training_rng_state'],
            'independent actor/Adam/RNG reload failed')
    return {'path': str(pair[0]), 'manifest': str(pair[1]), 'sha256': sha(pair[0]),
            'independent_official_reload_verified': True, 'latest_pointer_published': False,
            'normal_PPO_save_carry_after_this_real_AUX_not_yet_verified': True}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--expected-head', required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--execute-aux', action='store_true')
    p.add_argument('--inspection-receipt', type=Path)
    p.add_argument('--aux-checkpoint', type=Path)
    p.add_argument('--budget', type=Path)
    args = p.parse_args(argv)
    report_path = output_path(args.report)
    if not args.execute_aux:
        require(args.inspection_receipt is None and args.aux_checkpoint is None and args.budget is None,
                'read-only inspection does not accept execution arguments')
        require(not torch.cuda.is_available(), 'CPU inspection must use CUDA_VISIBLE_DEVICES=-1')
    checkpoint = args.checkpoint.resolve(strict=True)
    metadata = semantic_migration.checkpoint_metadata(checkpoint)
    contract = semantic_cli.runtime_contract(expected_head=args.expected_head, semantic_version='v3',
                                            experiment_id='p05_hip_only_continuation_v1')
    require(metadata['runtime_contract'] == contract and metadata['policy_contract']['version'] == P05_CAPTURE_POLICY,
            'current source checkpoint/runtime/profile mismatch')
    data = load_reviewed_data(metadata, contract)
    source = {'path': str(checkpoint), 'sha256': metadata['checkpoint_sha256'],
              'manifest_sha256': sha(checkpoint.with_name(checkpoint.stem + '_manifest.json'))}
    binding = {'source_checkpoint': source, 'helper_sha256': helpers(),
               'data_receipt_content_sha256': data['receipt']['receipt_content_sha256'],
               'runtime_contract_sha256': digest(contract)}
    result = {'schema': 'wlr50_clean.front_rehearsal_receipt.v1', 'binding': binding,
        'data_receipt': data['receipt'], 'teacher_deployed': False, 'physical_success_claimed': False,
        'original_PPO_counters': {k: metadata[k] for k in COUNTERS},
        'PPO_decisions_added': 0, 'PPO_updates_added': 0, 'PPO_optimizer_steps_added': 0}
    if not args.execute_aux:
        result.update(mode='read_only_current_checkpoint_on_exact_real_states', auxiliary_updates=0,
                      inspection=cpu_inspection(checkpoint, metadata, data), automatic_aux_enabled=False)
    else:
        require(args.inspection_receipt is not None and args.aux_checkpoint is not None and args.budget is not None,
                'explicit AUX requires reviewed same-source inspection, unique destination and explicit finite budget')
        prior = read(args.inspection_receipt)
        require(prior['mode'] == 'read_only_current_checkpoint_on_exact_real_states'
                and prior['auxiliary_updates'] == 0 and prior['binding'] == binding, 'stale or wrong inspection binding')
        spec = read(args.budget); budget = kernel.Budget(**spec); budget.validate()
        destination = output_path(args.aux_checkpoint)
        device = metadata['runner_config']['device']
        if str(device).startswith('cuda'):
            require(torch.cuda.is_available(), 'real source-device resume requires original CUDA visibility')
        runner = observation_runner(data['train_observations'][0], metadata, device=device)
        infos = training.load_semantic_checkpoint(runner, checkpoint, contract=contract, seed=metadata['seed'])
        report = kernel.fit(runner, *fit_arguments(data, device), budget=budget, authorized=True)
        result.update(mode='explicit_finite_auxiliary_not_PPO', auxiliary_updates=report['accepted_auxiliary_updates'],
            fit_report=report, budget=asdict(budget),
            reviewed_inspection={'path': str(args.inspection_receipt.resolve()), 'sha256': sha(args.inspection_receipt)})
        result['auxiliary_checkpoint'] = None
        if report['accepted_auxiliary_updates']:
            result['auxiliary_checkpoint'] = save_auxiliary_checkpoint(runner, destination, infos, report=report,
                data=data, source_checkpoint=source, helper_sha256=helpers(), budget=asdict(budget))
    training.write_json(report_path, result)
    print(json.dumps({'report': str(report_path), 'mode': result['mode'], 'auxiliary_updates': result['auxiliary_updates']}))
    return result


if __name__ == '__main__':
    main()
