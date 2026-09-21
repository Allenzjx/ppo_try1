"""CPU-tensor exact-source load + real-plan/CLI validation, without fake device metadata.

The source is a CUDA-trained checkpoint. Official RNG restore validates/restores
its declared CUDA RNG as well, but model, observations, optimizer tensors and
all numerical verification here stay on CPU. No optimizer or simulator is run.
Actual CUDA migration is left to the production launcher; its same-device CPU
analogue is covered by the focused unit test, not credited as physical training.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--migration', type=Path, required=True)
    p.add_argument('--expected-head', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    import torch
    from tensordict import TensorDict
    from wlr50_clean.ppo import semantic_migration as m, semantic_training as t, semantic_cli as cli
    torch.set_num_threads(1)
    cp, plan_path = args.checkpoint.resolve(strict=True), args.migration.resolve(strict=True)
    metadata = m.checkpoint_metadata(cp)
    source_hash = m.file_sha(cp)
    class ObservationEnv:
        num_envs, num_actions = 1, 12
        cfg = {'evaluation': True, 'semantic_version': 'v3'}
        def get_observations(self):
            value = torch.zeros((1, 372), device='cpu')
            return TensorDict({'policy': value, 'critic': value.clone()}, batch_size=[1], device='cpu')
    runner, _ = t.construct_semantic_runner(ObservationEnv(), seed=metadata['seed'], device='cpu',
        policy_version=metadata['policy_contract']['version'],
        observation_layout=metadata['policy_contract']['observation_layout'], initialize_actor=False)
    info = t.load_semantic_checkpoint(runner, cp, contract=metadata['runtime_contract'], seed=metadata['seed'])
    checks = {name: actual == metadata[name] for name, actual in (
        ('actor_parameter_sha256', t.parameter_hash(runner.alg.actor)),
        ('critic_parameter_sha256', t.parameter_hash(runner.alg.critic)),
        ('optimizer_state_sha256', t.state_hash(runner.alg.optimizer.state_dict())),
        ('normalizer_state_sha256', t.state_hash(t._normalizers(runner))))}
    checks.update(
        training_rng_equal=t.capture_training_rng_state(seed=metadata['seed']) == metadata['training_rng_state'],
        effective_lr_equal=t.optimizer_learning_rate(runner) == metadata['optimizer_learning_rate'] == 1e-5,
        fresh_rollout=runner.alg.storage.step == 0 and runner.alg.transition.actions is None,
        identity_normalizers=type(runner.alg.actor.obs_normalizer) is torch.nn.Identity and type(runner.alg.critic.obs_normalizer) is torch.nn.Identity,
        models_cpu=all(p.device.type == 'cpu' for model in (runner.alg.actor, runner.alg.critic) for p in model.parameters()),
        optimizer_cpu=all(v.device.type == 'cpu' for row in runner.alg.optimizer.state.values() for v in row.values() if torch.is_tensor(v)),
        lifetime_counters_equal=all(info[k] == metadata[k] for k in ('global_policy_decisions', 'ppo_updates', 'optimizer_steps')),
        source_checkpoint_unchanged=m.file_sha(cp) == source_hash)
    if not all(checks.values()):
        raise RuntimeError(checks)
    contract = cli.runtime_contract(expected_head=args.expected_head, semantic_version='v3', experiment_id='fl_capture_quality_v1')
    verified = m.validate_migration_plan(cp, contract, plan_path)
    argv = ['train', '--semantic-version', 'v3', '--experiment-id', 'fl_capture_quality_v1',
        '--run-dir', str(ROOT/'runs/ppo_fl_capture_quality_v1/train/preflight_only_no_run_created'),
        '--expected-head', args.expected_head, '--checkpoint', str(cp), '--resume-migration', str(plan_path),
        '--stage', 'full_episode', '--decisions', '512', '--seed', '1001', '--device', 'cuda:0',
        '--checkpoint-interval-updates', '1', '--from-phase', 'P01']
    request = cli.parser().parse_args(argv)
    cli.validate_request(request)
    cli._preflight_checkpoint(request, contract)
    receipt = {'schema': 'wlr50_clean.fl_quality_real_checkpoint_cpu_receipt.v1',
        'source_checkpoint': str(cp), 'source_checkpoint_sha256': source_hash,
        'migration_path': str(plan_path), 'migration_sha256': verified['plan_sha256'],
        'target_head': contract['source_git_commit'], 'target_runtime': contract['runtime_content_sha256'],
        'checks': checks, 'all_checks_passed': True, 'official_exact_source_load': True,
        'cpu_models_optimizer_only': True, 'source_runner_device_metadata': metadata['runner_config']['device'],
        'cuda_rng_states_restored_by_official_loader': metadata['training_rng_state']['torch_cuda_device_count'],
        'actual_cuda_migration_load_claimed': False,
        'cuda_migration_plan_and_first512_cli_preflight_passed': True,
        'counts': {k: info[k] for k in ('global_policy_decisions', 'ppo_updates', 'optimizer_steps')},
        'effective_learning_rate': t.optimizer_learning_rate(runner),
        'rollout_shape': list(runner.alg.storage.actions.shape), 'new_policy_decisions': 0,
        'new_ppo_updates': 0, 'new_optimizer_steps': 0, 'simulation_started': False,
        'first512_validated_python_arguments': argv}
    t.write_json(args.output.resolve(), receipt)
    print(json.dumps({k: receipt[k] for k in ('all_checks_passed', 'counts', 'effective_learning_rate',
        'cuda_migration_plan_and_first512_cli_preflight_passed', 'actual_cuda_migration_load_claimed')}, indent=2))


if __name__ == '__main__':
    main()
