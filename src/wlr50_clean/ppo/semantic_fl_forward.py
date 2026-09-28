"""Continue the complete validated A policy with one P06 FL coordinate.

The full A actor and its normalizers remain frozen. Continuous physical
context is retained for value/GAE, with actor versus critic-only rows counted.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

from . import semantic_finish_advance as base

ROOT = base.ROOT
NAME = 'ppo_finish_advance_fl_forward_v1'
CONFIG = ROOT/'configs'/NAME
OUTPUT = ROOT/'outputs'/NAME
SCHEMA = 'wlr50_clean.frozen_complete_A_p06_fl_checkpoint.v1'
sha, write, line = base.sha, base.write, base.line


def settings():
    return json.loads((CONFIG/'local_training.json').read_text(encoding='utf-8'))


def source_metadata():
    cfg = settings()
    path = ROOT/cfg['accepted_A_checkpoint']
    side = path.with_name(path.stem+'_manifest.json')
    if sha(path) != cfg['accepted_A_sha256'] or sha(side) != cfg['accepted_A_manifest_sha256']:
        raise ValueError('immutable demonstrated complete A package changed')
    meta = json.loads(side.read_text())
    if (meta['schema'] != base.SCHEMA or not meta['save_load_round_trip']
            or not meta['rollout_empty'] or meta['observation_dimension'] != 531
            or meta['runtime_contract']['local_contract'] != base.settings()):
        raise ValueError('compatible saved complete A531 package required')
    evaluation_path = ROOT/cfg['accepted_A_evaluation_manifest']
    if sha(evaluation_path) != cfg['accepted_A_evaluation_sha256']:
        raise ValueError('demonstrated A physical result binding changed')
    evaluation = json.loads(evaluation_path.read_text())
    proof = evaluation.get('checkpoint_load_provenance', {})
    if (evaluation.get('full_task_success') is not True
            or evaluation.get('continuous_natural_P01') is not True
            or evaluation.get('checkpoint_model_unchanged') is not True
            or evaluation.get('policy_switches') != 0
            or proof.get('checkpoint_sha256') != sha(path)
            or proof.get('manifest_sha256') != sha(side)):
        raise ValueError('complete A source must match its actual saved/reloaded full physical success')
    return path, meta


def contract(expected_head):
    result = base.contract(expected_head)
    _, meta = source_metadata()
    result.update(experiment_id=NAME, local_contract=settings(),
        accepted_A_configuration=copy.deepcopy(meta['runtime_contract']['selected_configuration']),
        selected_configuration={'local_training.json': dict(
            path=(CONFIG/'local_training.json').relative_to(ROOT).as_posix(),
            sha256=sha(CONFIG/'local_training.json'))})
    return result


def tensor_observation(values, device):
    import torch
    from tensordict import TensorDict
    tensor = torch.tensor([values], dtype=torch.float32, device=device)
    if tensor.shape != (1, settings()['observation_dimension']) or not bool(torch.isfinite(tensor).all()):
        raise ValueError('finite complete A531 plus explicit FL window required')
    return TensorDict({'policy': tensor, 'critic': tensor.clone()}, batch_size=[1], device=device)


def make_runner(device, *, saved_configuration=None):
    import torch
    from types import SimpleNamespace
    from .rl_library_wrapper import construct_runner
    cfg = settings()
    _, meta = source_metadata()
    configuration = copy.deepcopy(meta['runner_config'])
    configuration['device'] = device
    configuration['actor'] = dict(
        class_name='wlr50_clean.ppo.semantic_fl_forward_actor:WindowFLActor',
        obs_normalization=False,
        frozen_A_configuration=copy.deepcopy(meta['runner_config']['actor']),
        initial_std=cfg['initial_FL_conditional_std'],
        observation_layout=cfg['observation_layout'])
    if saved_configuration is not None:
        actual = copy.deepcopy(saved_configuration)
        actual['device'] = device
        checked = copy.deepcopy(actual)
        checked['actor'].pop('expected_frozen_A_sha256', None)
        if checked != configuration:
            raise ValueError('saved window actor runner recipe mismatch')
        configuration = actual
    env = SimpleNamespace(num_envs=1, num_actions=12, device=device, cfg={'window_task': NAME},
        get_observations=lambda: tensor_observation([0.]*cfg['observation_dimension'], device))
    runner = construct_runner(env, configuration, log_dir=None)
    runner.alg.optimizer = torch.optim.Adam(list(runner.alg.actor.trainable_parameters())+
        list(runner.alg.critic.parameters()), lr=meta['learning_rate'])
    runner.alg.learning_rate = meta['learning_rate']
    runner._semantic_policy_version = runner.alg.actor.policy_version
    runner._semantic_front_replay = None
    runner.logger.writer = None
    runner.local_configuration = copy.deepcopy(configuration)
    return runner


def initialize_from_A(runner):
    import torch
    from .semantic_training import state_hash
    from .semantic_rr_capture_local_actor import tensor_state_sha256
    from .rl_library_wrapper import restore_training_rng_state
    path, meta = source_metadata()
    source = base.make_runner(runner.device, saved_configuration=meta['runner_config'])
    base.load(source, path, meta['runtime_contract'])
    old_state = source.alg.save()
    digest = runner.alg.actor.load_A_state(old_state['actor_state_dict'],
        expected_sha256=tensor_state_sha256(old_state['actor_state_dict']))
    expanded, input_changes = {}, []
    target = runner.alg.critic.state_dict()
    for name, value in old_state['critic_state_dict'].items():
        if value.shape == target[name].shape:
            expanded[name] = value.clone()
        elif value.ndim == 2 and value.shape[1] == 531 and target[name].shape == (
                value.shape[0], settings()['observation_dimension']):
            expanded[name] = torch.zeros_like(target[name])
            expanded[name][:, :531].copy_(value)
            input_changes.append(name)
        else:
            raise ValueError('unexpected B critic migration shape')
    if len(input_changes) != 1:
        raise ValueError('one declared critic input extension required')
    runner.alg.critic.load_state_dict(expanded, strict=True)
    old_actor = list(source.alg.actor.trainable_parameters())
    new_actor = list(runner.alg.actor.trainable_parameters())
    if sum(p.numel() for p in new_actor) != 2:
        raise ValueError('B first candidate permits only FL mean and log std')
    old_opt = old_state['optimizer_state_dict']
    new_opt = runner.alg.optimizer.state_dict()
    if len(old_opt['param_groups']) != 1 or len(new_opt['param_groups']) != 1:
        raise ValueError('explicit single inherited Adam group required')
    old_ids, new_ids = old_opt['param_groups'][0]['params'], new_opt['param_groups'][0]['params']
    group = copy.deepcopy(old_opt['param_groups'][0])
    group['params'] = list(new_ids)
    migrated = dict(state={}, param_groups=[group])
    moment_changes = []
    for (old_name, op), (new_name, np), oi, ni in zip(
            source.alg.critic.named_parameters(), runner.alg.critic.named_parameters(),
            old_ids[len(old_actor):], new_ids[len(new_actor):], strict=True):
        if old_name != new_name or oi not in old_opt['state']:
            raise ValueError('trained critic Adam binding missing')
        state = copy.deepcopy(old_opt['state'][oi])
        for key, value in tuple(state.items()):
            if torch.is_tensor(value) and value.shape == op.shape and op.shape != np.shape:
                padded = value.new_zeros(np.shape)
                padded[:, :531].copy_(value)
                state[key] = padded
                moment_changes.append(new_name+'/'+key)
            elif torch.is_tensor(value) and value.ndim > 0 and value.shape != np.shape:
                raise ValueError('unexpected critic Adam moment shape')
        collapsed = copy.deepcopy(state)
        if op.shape != np.shape:
            for key, value in tuple(collapsed.items()):
                if torch.is_tensor(value) and value.shape == np.shape:
                    if torch.count_nonzero(value[:, 531:]):
                        raise RuntimeError('new critic moment columns not zero')
                    collapsed[key] = value[:, :531].clone()
        if state_hash(collapsed) != state_hash(old_opt['state'][oi]):
            raise RuntimeError('critic source Adam moments or steps changed')
        migrated['state'][ni] = state
    runner.alg.optimizer.load_state_dict(migrated)
    runner.alg.learning_rate = meta['learning_rate']
    runner.local_configuration['actor']['expected_frozen_A_sha256'] = digest
    runner.alg.actor.assert_frozen_state(runner.alg.optimizer)
    restore_training_rng_state(meta['training_rng'], expected_seed=1001)
    lineage = dict(schema='wlr50_clean.frozen_A_scalar_FL_migration.v1',
        accepted_A_checkpoint=str(path), accepted_A_sha256=sha(path),
        accepted_A_manifest_sha256=sha(path.with_name(path.stem+'_manifest.json')),
        accepted_A_tensor_sha256=digest, accepted_A_counts=copy.deepcopy(meta['counts']),
        accepted_A_state_hashes=copy.deepcopy(meta['state_hashes']),
        accepted_A_lineage=copy.deepcopy(meta['lineage']),
        entire_A_actor_frozen=True, old_actor_Adam_preserved_in_immutable_source=True,
        new_actor_parameter_count=2, new_actor_Adam_empty=True,
        critic_Adam_inherited=True, critic_zero_columns=input_changes,
        critic_Adam_zero_columns=moment_changes, actual_learning_rate=meta['learning_rate'],
        old_rollout_reused=False, RNG_preserved=True, Identity_normalizers_unchanged=True,
        historical_AUX_updates=meta['historical_AUX_updates'], new_AUX_updates=0)
    counts = dict(fl_policy_decisions=0, critic_only_decisions=0, continuous_storage_decisions=0,
        prefix_decisions=0, ppo_updates=0, optimizer_steps=0, actor_optimizer_steps=0,
        episodes=0, full_successes=0, opportunities=0, auxiliary_updates=0)
    return lineage, counts


def validate_runner(runner, runtime):
    cfg = settings()
    if (runtime['experiment_id'] != NAME or runtime['local_contract'] != cfg
            or runner.alg.actor.policy_version != cfg['version']):
        raise ValueError('B window runtime/actor/config mismatch')
    runner.alg.actor.assert_frozen_state(runner.alg.optimizer)


def checkpoint_path(counts, runtime):
    return OUTPUT/'checkpoints/history'/(
        'checkpoint_CP%d_FL%06d_data%06d_g%s.pt' % (
            settings()['historical_decision_origin']+counts['fl_policy_decisions'],
            counts['fl_policy_decisions'], counts['continuous_storage_decisions'],
            runtime['source_git_commit'][:12]))


def save(runner, runtime, lineage, counts, *, source_run, publish_pointer=True):
    import torch
    from .semantic_training import state_hash
    from .rl_library_wrapper import capture_training_rng_state
    validate_runner(runner, runtime)
    if (counts['auxiliary_updates'] or runner.alg.storage.step != 0
            or runner.alg.transition.actions is not None):
        raise RuntimeError('complete update and empty rollout required')
    target = checkpoint_path(counts, runtime)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    payload = runner.alg.save()
    hashes = {k: state_hash(v) for k, v in payload.items()}
    infos = dict(schema=SCHEMA, runtime_contract=runtime, lineage=copy.deepcopy(lineage),
        counts=dict(counts), runner_config=copy.deepcopy(runner.local_configuration),
        state_hashes=hashes, learning_rate=runner.alg.learning_rate,
        training_rng=capture_training_rng_state(seed=1001), source_run=str(source_run),
        rollout_empty=True, front_FL_assist=True, rear_task_assists=False,
        public_preparation_module=True, public_finish_module=True,
        observation_dimension=settings()['observation_dimension'],
        historical_AUX_updates=lineage['historical_AUX_updates'], local_auxiliary_events=[])
    payload.update(infos=infos, iter=counts['ppo_updates'])
    torch.save(payload, target)
    restored = torch.load(target, map_location=runner.device, weights_only=False)
    if restored['infos'] != infos or any(state_hash(restored[k]) != v for k, v in hashes.items()):
        raise RuntimeError('serialized B package differs')
    runner.alg.load(restored, None, True)
    runner.alg.learning_rate = infos['learning_rate']
    validate_runner(runner, runtime)
    if any(state_hash(runner.alg.save()[k]) != v for k, v in hashes.items()):
        raise RuntimeError('strict B state/Adam reload differs')
    side = target.with_name(target.stem+'_manifest.json')
    write(side, dict(infos, checkpoint=str(target), checkpoint_sha256=sha(target), save_load_round_trip=True))
    pointer = dict(checkpoint=str(target), checkpoint_sha256=sha(target), manifest=str(side),
        manifest_sha256=sha(side), counts=dict(counts), evaluated=False)
    if publish_pointer:
        path = OUTPUT/'checkpoints/checkpoint_last_pointer.json'
        write(path, pointer, replace=path.exists())
    return pointer


def load(runner, path, runtime):
    import torch
    from .semantic_training import state_hash
    from .rl_library_wrapper import restore_training_rng_state
    path = Path(path).resolve(strict=True)
    side = path.with_name(path.stem+'_manifest.json')
    meta = json.loads(side.read_text())
    if (meta['schema'] != SCHEMA or meta['runtime_contract'] != runtime
            or meta['checkpoint_sha256'] != sha(path) or not meta['save_load_round_trip']
            or not meta['rollout_empty'] or meta['counts']['auxiliary_updates']):
        raise ValueError('B package contract/hash/ledger mismatch')
    cfg = copy.deepcopy(meta['runner_config'])
    cfg['device'] = runner.device
    current = copy.deepcopy(runner.local_configuration)
    current['actor'].setdefault('expected_frozen_A_sha256', cfg['actor']['expected_frozen_A_sha256'])
    if cfg != current or meta['lineage']['accepted_A_sha256'] != settings()['accepted_A_sha256']:
        raise ValueError('B construction or complete A binding changed')
    runner.alg.actor.expected_frozen_A_sha256 = cfg['actor']['expected_frozen_A_sha256']
    payload = torch.load(path, map_location=runner.device, weights_only=False)
    if any(meta.get(k) != v for k, v in payload['infos'].items()):
        raise ValueError('embedded B metadata differs')
    runner.alg.load(payload, None, True)
    runner.alg.learning_rate = meta['learning_rate']
    runner.local_configuration = cfg
    validate_runner(runner, runtime)
    if any(state_hash(runner.alg.save()[k]) != v for k, v in meta['state_hashes'].items()):
        raise ValueError('reloaded B actor/critic/Adam differs')
    restore_training_rng_state(meta['training_rng'], expected_seed=1001)
    runner.current_learning_iteration = meta['counts']['ppo_updates']
    runner.checkpoint_load_provenance = dict(checkpoint=str(path), checkpoint_sha256=sha(path),
        manifest=str(side), manifest_sha256=sha(side), strict_actual_composite_load_verified=True,
        actor_critic_optimizer_hashes_verified=True, whole_A531_frozen=True)
    return copy.deepcopy(meta['lineage']), dict(meta['counts'])


def request(runner, observation, *, stochastic):
    from .semantic_fl_forward_actor import audited_fl_request
    return audited_fl_request(runner.alg.actor, observation,
        (lambda: runner.alg.act(observation)) if stochastic else
        (lambda: runner.alg.actor(observation, stochastic_output=False)), stochastic=stochastic)


def prefix(core, runner, stream, counts):
    import torch
    core.reset(seed=1001)
    while not core.fl_window['active'] and not core.done:
        obs = tensor_observation(core.observation, runner.device)
        with torch.inference_mode():
            raw, audit = request(runner, obs, stochastic=False)
        step = core.step(raw[0].cpu().tolist())
        counts['prefix_decisions'] += 1
        line(stream, dict(kind='frozen_A_pre_window_prefix', actor_credit=0, critic_credit=0,
            physical_tick=core.frame.physics_tick, phase=core.frame.state_id, policy_request=audit,
            fl_forward_window=core.fl_window))
    if core.done:
        raise RuntimeError('frozen A ended before P06 FL opportunity: '+str(step.info.get('termination_reason')))
    counts['opportunities'] += 1


def train(core, runner, runtime, lineage, counts, run, active_decisions):
    import torch
    from .semantic_training import verified_native_effect, _stop_request
    from .semantic_fl_forward_actor import audited_fl_ppo_update
    if not 512 <= active_decisions <= 1024 or settings()['rollout_length'] != 512:
        raise ValueError('first B check block requires512–1024 active FL decisions and512 contiguous storage')
    # A lower check threshold, inspected only after a complete contiguous
    # rollout. Default512 may finish with512..1023 actual active rows; report
    # the ledger rather than pretending it is an exact sample cap.
    goal = counts['fl_policy_decisions']+active_decisions
    runner.alg.train_mode()
    (run/'rollouts').mkdir()
    with (run/'decisions.jsonl').open('x') as stream, (run/'updates.jsonl').open('x') as updates:
        while counts['fl_policy_decisions'] < goal:
            phases, modes = Counter(), Counter()
            for index in range(512):
                if core.done:
                    prefix(core, runner, stream, counts)
                obs = tensor_observation(core.observation, runner.device)
                before = copy.deepcopy(core.fl_window)
                active = bool(before['active'])
                with torch.inference_mode():
                    raw, audit = request(runner, obs, stochastic=True)
                    selected = raw.detach().clone()
                    logp = runner.alg.transition.actions_log_prob.detach().clone()
                    step = core.step(raw[0].cpu().tolist())
                    issued = step.info.get('applied_raw_full12', raw[0].cpu().tolist())
                    if list(issued) != raw[0].cpu().tolist():
                        raise RuntimeError('sampled complete B raw action cannot be replaced')
                    verified_native_effect(step.info, tuple(issued))
                    nxt = tensor_observation(step.observation, runner.device)
                    done = torch.tensor([step.terminated], dtype=torch.bool, device=runner.device)
                    runner.alg.process_env_step(nxt, torch.tensor([step.reward], device=runner.device),
                        done, {'time_outs': torch.zeros_like(done)})
                    if (not torch.equal(runner.alg.storage.actions[index], selected)
                            or not torch.equal(runner.alg.storage.actions_log_prob[index].view(-1), logp.view(-1))):
                        raise RuntimeError('raw sample/old scalar logp changed before storage')
                counts['continuous_storage_decisions'] += 1
                counts['fl_policy_decisions' if active else 'critic_only_decisions'] += 1
                phases[step.info['phase_id']] += 1
                modes['P06_FL_active' if active else 'frozen_A_critic_only'] += 1
                line(stream, dict(kind='P06_FL_on_policy' if active else 'frozen_A_continuous_critic_only',
                    actor_credit=int(active), critic_credit=1, storage_decision=counts['continuous_storage_decisions'],
                    fl_policy_decision=counts['fl_policy_decisions'], observation=obs['policy'][0].cpu().tolist(),
                    policy_request=audit, old_logp=logp.cpu().tolist(), before_window=before, step_info=step.info))
                if step.terminated:
                    counts['episodes'] += 1
                    counts['full_successes'] += int(step.info.get('full_task_success', False))
                if (index+1)%128 == 0:
                    print(json.dumps(dict(counts=counts, time_s=core.frame.sim_time_s,
                        phase=core.frame.state_id, window=core.fl_window)), flush=True)
            with torch.inference_mode():
                runner.alg.compute_returns(nxt)
            storage = runner.alg.storage
            snapshot = {k: getattr(storage, k).detach().cpu().clone() for k in
                ('actions', 'actions_log_prob', 'rewards', 'dones', 'values', 'returns', 'advantages')}
            snapshot['observations'] = {k: v.detach().cpu().clone() for k, v in storage.observations.items()}
            snapshot['distribution_params'] = tuple(v.detach().cpu().clone() for v in storage.distribution_params)
            number = counts['ppo_updates']+1
            torch.save(snapshot, run/'rollouts'/f'rollout_{number:04d}.pt')
            report = audited_fl_ppo_update(runner,
                likelihood_audit_path=run/'rollouts'/f'likelihood_{number:04d}.json')
            validate_runner(runner, runtime)
            counts['ppo_updates'] += 1
            counts['optimizer_steps'] += report['optimizer_steps']
            counts['actor_optimizer_steps'] += report['actor_optimizer_steps']
            runner.current_learning_iteration = counts['ppo_updates']
            report.update(counts=dict(counts), actual_phase_counts=dict(phases), actual_row_modes=dict(modes),
                ordinary_phase_done=False, skipped_off_window_time=False, AUX_updates=0)
            line(updates, report)
            pointer = save(runner, runtime, lineage, counts, source_run=run)
            print(json.dumps(dict(completed_update=number, checkpoint=pointer)), flush=True)
            stop = _stop_request(run, runtime)
            if stop is not None:
                write(run/'stop_after_update.accepted.json', dict(stop, checkpoint=pointer, rollout_empty=True))
                break
    return pointer


def evaluate(core, runner, runtime, lineage, counts, run):
    cfg = settings()
    route = dict(validate_runner=validate_runner, auxiliary_ledger=[], settings=cfg,
        config_dir=ROOT/cfg['accepted_control_config'], tensor_observation=tensor_observation, request=request,
        control_contributions=dict(entire_accepted_A_frozen=lineage, public_preparation_module=True,
            public_finish_module=True, requested_finish=cfg['requested_finish'], new_AUX_updates=0,
            FL_window='P06 ordinary forward only; postRR prep/transfer/recoil/stops unchanged',
            FL_window_reward=cfg['window_reward'],
            FL_trainable_parameter_count=2, FL_control_projection=False,
            FL_branch_counters=dict(counts), historical_AUX_updates=lineage['historical_AUX_updates'],
            displayed_label='PPO + original FL assist/front preparation + measured advance/home; P06 scalar FL learning; rear helpers OFF'))
    return base.old.evaluate(core, runner, runtime, lineage, counts, run, False, route=route)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('migrate', 'train', 'eval'))
    parser.add_argument('--expected-head', required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--active-decisions', type=int, default=512)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    runtime = contract(args.expected_head)
    run = args.run_dir.resolve()
    if not run.is_relative_to(ROOT/'runs'/NAME):
        raise ValueError('isolated B namespace required')
    if (args.mode == 'migrate') != (args.checkpoint is None):
        raise ValueError('migration pins source; train/eval must use saved complete B')
    run.mkdir(parents=True, exist_ok=False)
    write(run/'run_manifest.started.json', dict(runtime_contract=runtime, mode=args.mode,
        checkpoint=str(args.checkpoint), started_utc=datetime.now(timezone.utc).isoformat()))
    app = None
    try:
        import torch
        import tensordict
        from .rl_library_wrapper import seed_training_rngs
        if args.mode != 'migrate':
            from isaaclab.app import AppLauncher
            app = AppLauncher(headless=args.mode == 'train', enable_cameras=False).app
            app.update()
        seed_training_rngs(1001)
        meta = json.loads(args.checkpoint.with_name(args.checkpoint.stem+'_manifest.json').read_text()) if args.checkpoint else None
        runner = make_runner(args.device, saved_configuration=meta['runner_config'] if meta else None)
        lineage, counts = load(runner, args.checkpoint, runtime) if args.checkpoint else initialize_from_A(runner)
        if args.mode == 'migrate':
            result = save(runner, runtime, lineage, counts, source_run=run)
        else:
            from dataclasses import asdict, replace
            from .semantic_fl_forward_task import build_core, config_from_sources
            window_config = replace(config_from_sources(ROOT/'configs/recording_motion_contract.json',
                ROOT/'configs/fsm_states.yaml'), **settings()['window_reward'])
            write(run/'FL_window_source_receipt.json', asdict(window_config))
            core = build_core(app, config=window_config)
            if args.mode == 'train':
                result = train(core, runner, runtime, lineage, counts, run, args.active_decisions)
            else:
                runner.alg.eval_mode()
                result = evaluate(core, runner, runtime, lineage, counts, run)
        write(run/'run_manifest.json', dict(lifecycle='COMPLETE', mode=args.mode, result=result))
        print(json.dumps(dict(run=str(run), lifecycle='COMPLETE')), flush=True)
    except BaseException:
        write(run/'failure.json', {'traceback': traceback.format_exc()})
        raise
    finally:
        if app is not None:
            app.close(wait_for_replicator=False, skip_cleanup=True)


if __name__ == '__main__':
    main()
