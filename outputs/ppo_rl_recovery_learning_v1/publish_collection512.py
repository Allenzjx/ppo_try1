"""One explicit, zero-credit collection boundary after a sealed physical run."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
BRANCH = ROOT / 'outputs/ppo_rr_rl_timing_policy_learning_v1/branches/ancestor220544_recapture_v2'
SEALED = ROOT / 'runs/ppo_rr_rl_timing_policy_learning_v1/train/20260924T1046432416199Z_g65a9255be6d9_bff8c2f6d5574a42925e6b9d8f382a94'


def main():
    import torch
    from tensordict import TensorDict
    sys.path.insert(0, str(HERE / 'staged_front_retention439'))
    import front_retention439 as kernel
    from wlr50_clean.ppo import semantic_training as training
    from wlr50_clean.ppo.semantic_cli import runtime_contract
    from wlr50_clean.ppo.semantic_front_retention439 import publish_collection512_checkpoint, COLLECTION_KEY
    from wlr50_clean.ppo.semantic_migration import file_sha
    from wlr50_clean.ppo.semantic_rr_capture_migration import _shape_env
    from wlr50_clean.ppo.semantic_return_profile import COLLECTION_512
    from wlr50_clean.ppo.semantic_rear_owner_profile import REAR_OWNER_POLICY, REAR_OWNER_OBSERVATION_LAYOUT
    from wlr50_clean.ppo.rl_library_wrapper import restore_training_rng_state, capture_training_rng_state
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--expected-head', required=True)
    args = parser.parse_args()
    pointer_path = BRANCH / 'checkpoints/checkpoint_last_pointer.json'
    pointer = json.loads(pointer_path.read_text())
    metadata = json.loads(Path(pointer['manifest']).read_text())
    assert (metadata['global_policy_decisions'], metadata['ppo_updates'], metadata['optimizer_steps']) == (229632, 1759, 35180)
    assert file_sha(Path(pointer['checkpoint'])) == pointer['checkpoint_sha256']
    assert file_sha(Path(pointer['manifest'])) == pointer['manifest_sha256']
    contract = runtime_contract(expected_head=args.expected_head, semantic_version='v3', experiment_id='rr_rl_timing_policy_learning_v1')
    label = 'collection512_CP229632_g' + args.expected_head[:12]
    destination = BRANCH / 'checkpoints/history' / ('checkpoint_' + label + '.pt')
    output = HERE / (label + '_publication.json')
    assert not destination.exists() and not output.exists()
    receipt = publish_collection512_checkpoint(Path(pointer['checkpoint']), contract, destination,
        reason='512 consecutive learner decisions to include preparation, support loss and bounded recovery outcomes; same reward, control, Gaussian/HISTORY and complete learned state; 5 epochs/4 minibatches now 128 samples per minibatch',
        expected_source_sha256=pointer['checkpoint_sha256'], expected_manifest_sha256=pointer['manifest_sha256'])
    # Independent official fixed-input verification on actual saved learner
    # observations; no resampling, PPO, AUX, teacher, or physical execution.
    observations, provenance = [], []
    for path in sorted((SEALED / 'rollouts').glob('rollout_*.pt')):
        saved = torch.load(path, map_location='cpu', weights_only=False)
        rows = saved['observations']['policy'].reshape(-1, 439)
        ids = [0, len(rows)//2, len(rows)-1]
        observations.extend(rows[ids])
        provenance.append(dict(path=str(path), sha256=file_sha(path), flat_indices=ids))
    device = metadata['runner_config']['device']
    batch = torch.stack(observations).to(device)
    obs = TensorDict({'policy':batch, 'critic':batch.clone()}, batch_size=[len(batch)], device=device)
    def load(path, marker, runtime):
        runner, _ = training.construct_semantic_runner(_shape_env(439, device), seed=metadata['seed'], device=device,
            policy_version=REAR_OWNER_POLICY, observation_layout=REAR_OWNER_OBSERVATION_LAYOUT,
            collection_profile=marker, initialize_actor=False)
        infos = training.load_semantic_checkpoint(runner, path, contract=runtime, seed=metadata['seed'])
        runner.alg.eval_mode()
        with torch.inference_mode():
            mean = runner.alg.actor(obs, stochastic_output=False).clone()
            dist = kernel.distribution(runner.alg.actor, obs)
            assert torch.equal(mean, dist['mean'])
            params = (dist['mean'].clone(), dist['sigma'].clone())
            value = runner.alg.critic(obs).clone()
        return runner, infos, (mean, *params, value)
    old, old_infos, old_response = load(Path(pointer['checkpoint']), None, metadata['runtime_contract'])
    new, new_infos, new_response = load(destination, COLLECTION_512, contract)
    assert all(torch.equal(a,b) for a,b in zip(old_response,new_response,strict=True))
    assert new.alg.storage.step == 0 and new.alg.transition.actions is None
    assert tuple(new.alg.storage.actions.shape) == (512,1,12)
    assert training.state_hash(old.alg.optimizer.state_dict()) == training.state_hash(new.alg.optimizer.state_dict())
    assert new_infos['front_retention439_auxiliary'] == old_infos['front_retention439_auxiliary']
    assert new_infos[COLLECTION_KEY]['counter_origin'] == {k:metadata[k] for k in ('global_policy_decisions','ppo_updates','optimizer_steps')}
    restore_training_rng_state(metadata['training_rng_state'], expected_seed=metadata['seed'])
    assert capture_training_rng_state(seed=metadata['seed']) == metadata['training_rng_state']
    assert json.loads(pointer_path.read_text()) == pointer
    receipt.update(source_checkpoint=pointer, official_same_input_mean_sigma_value_bitwise_equal=True,
        fixed_input_provenance=provenance, physical_steps_added=0, new_auxiliary_updates=0,
        actual_collection_shape=list(new.alg.storage.actions.shape), auxiliary_ledger_preserved=True,
        training_rng_preserved=True, source_latest_pointer_unchanged=True)
    training.write_json(output, receipt)
    print(json.dumps({'publication':str(output), **receipt}))


if __name__ == '__main__':
    main()
