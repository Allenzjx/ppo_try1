"""Cold, explicitly bounded offline AUX publication. Never starts Isaac.

This driver is NOT executed merely by preparing it. It requires the new
committed metadata-only runtime, sealed update6 and an explicit recipe.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
NAME = 'ppo_rr_capture_first_cp225280_v1'


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--isaac-stopped', action='store_true')
    parser.add_argument('--expected-head', required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--learning-rate', type=float, required=True)
    parser.add_argument('--max-shift-sigma', type=float, required=True)
    args = parser.parse_args()
    if not args.isaac_stopped:
        raise ValueError('confirm the sole Isaac process exited before this offline driver')
    sys.path.insert(0, str(ROOT/'src'))
    from wlr50_clean.ppo import semantic_rr_capture_local as route
    from wlr50_clean.ppo import semantic_rr_capture_local_aux as aux
    checkpoint = args.checkpoint.resolve(strict=True)
    sidecar = checkpoint.with_name(checkpoint.stem+'_manifest.json')
    parent = json.loads(sidecar.read_text())
    package = aux.prepare_success_package(args.source_run.resolve(strict=True),
        expected_runtime=parent['runtime_contract'], parent_checkpoint=checkpoint,
        checkpoint_sha256=args.checkpoint_sha256, manifest_sha256=args.manifest_sha256)
    recipe = aux.recipe(steps=args.steps, learning_rate=args.learning_rate,
                        max_shift_sigma=args.max_shift_sigma)
    run = args.run_dir.resolve()
    if not run.is_relative_to(ROOT/'runs'/NAME) or run == ROOT/'runs'/NAME:
        raise ValueError('AUX run must be a new child of the isolated namespace')
    run.mkdir(parents=True, exist_ok=False)
    result = None
    try:
        import torch
        torch.set_num_threads(1)
        from wlr50_clean.ppo.rl_library_wrapper import seed_training_rngs
        runtime = route.contract(args.expected_head)
        route.write(run/'run_manifest.started.json', dict(mode='finite_rr_mean_aux',
            runtime_contract=runtime, checkpoint=str(checkpoint), recipe=recipe,
            driver=dict(path=str(Path(__file__).resolve()), sha256=route.sha(__file__)),
            no_Isaac=True, no_PPO_storage_data=True, rear_task_assists=False))
        route.write(run/'source_package.json', package)
        route.write(run/'recipe.json', recipe)
        seed_training_rngs(1001)
        runner = route.make_runner('cuda:0', 1001)
        prior, counts = route.rebind_checkpoint(runner, checkpoint, runtime,
            checkpoint_sha256=args.checkpoint_sha256, manifest_sha256=args.manifest_sha256)
        if counts != package['parent_counts']:
            raise RuntimeError('cold rebind changed source counters')
        result = aux.fit_rr_mean_rows(runner, package, recipe, isaac_stopped=True)
        route.write(run/'fit_receipt.json', result['receipt'])
        torch.save(result['aux_optimizer_state'], run/'independent_aux_optimizer.pt')
        receipt = result['receipt']
        if (receipt['status'] not in ('COMPLETE', 'STOPPED_CONSTRAINT')
                or not receipt['actor_rows_committed'] or not all(receipt['invariants'].values())
                or result['event'] is None):
            raise RuntimeError('AUX did not produce a valid committed candidate; preserve receipt, do not publish')
        event = copy.deepcopy(result['event'])
        event['published_evidence'] = {
            name: dict(path=str(run/name), sha256=route.sha(run/name))
            for name in ('source_package.json', 'recipe.json', 'fit_receipt.json',
                         'independent_aux_optimizer.pt')}
        event['event_id'] = aux.canonical_sha({k:v for k,v in event.items() if k != 'event_id'})
        runner.local_auxiliary_events = copy.deepcopy(runner.local_auxiliary_events) + [event]
        counts = result['counts_after']
        aux.validate_local_auxiliary_events(runner.local_auxiliary_events, counts)
        if route.contract(args.expected_head) != runtime:
            raise RuntimeError('source changed during the isolated AUX operation')
        pointer = route.save(runner, runtime, prior, counts, source_run=run)
        saved_events = copy.deepcopy(runner.local_auxiliary_events)
        loaded_prior, loaded_counts = route.load(runner, pointer['checkpoint'], runtime)
        if (loaded_prior != prior or loaded_counts != counts
                or runner.local_auxiliary_events != saved_events):
            raise RuntimeError('AUX package did not retain its lineage on strict reload')
        if route.sha(checkpoint) != args.checkpoint_sha256 or route.sha(sidecar) != args.manifest_sha256:
            raise RuntimeError('immutable parent changed')
        route.write(run/'run_manifest.json', dict(lifecycle='COMPLETE', mode='finite_rr_mean_aux',
            result=pointer, new_PPO_decisions=0, new_PPO_updates=0, new_PPO_Adam_steps=0,
            actual_AUX_steps=receipt['AUX_optimizer_steps'], accepted_AUX_steps=receipt['AUX_accepted_steps'],
            fresh_on_policy_collection_required=True, physical_capability='NOT_EVALUATED',
            rear_task_assists=False))
        print(json.dumps(dict(lifecycle='COMPLETE', result=pointer,
            actual_AUX_steps=receipt['AUX_optimizer_steps'], accepted_AUX_steps=receipt['AUX_accepted_steps'])), flush=True)
    except BaseException:
        failure = dict(lifecycle='FAILED', traceback=traceback.format_exc(),
            actual_AUX_steps=(result or {}).get('receipt', {}).get('AUX_optimizer_steps', 0),
            publication_not_claimed=True)
        route.write(run/'failure.json', failure)
        print(json.dumps(failure), flush=True)
        raise


if __name__ == '__main__':
    main()
