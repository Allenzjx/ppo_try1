"""Cold continuation of B535 with observed P06 FL prior retirement.

The existing B scalars, full A actor, Adam and RNG are preserved. Only the
old frozen FL innovation is retired in its explicitly observed P06 window.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import traceback
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from wlr50_clean.ppo import semantic_fl_forward as old

ROOT = old.ROOT
MODULE_DIR = Path(__file__).resolve().parent
NAME = 'ppo_fl_forward_prior_retirement_v1'
CONFIG = ROOT/'configs'/NAME
OUTPUT = ROOT/'outputs'/NAME
SCHEMA = 'wlr50_clean.frozen_B535_prior_retirement_checkpoint.v1'
sha, write, line = old.sha, old.write, old.line


def reference_config():
    return json.loads((CONFIG/'local_training.json').read_text(encoding='utf-8'))


def source_metadata():
    cfg = reference_config(); path = ROOT/cfg['source_checkpoint']
    side = path.with_name(path.stem+'_manifest.json')
    if sha(path) != cfg['source_checkpoint_sha256'] or sha(side) != cfg['source_manifest_sha256']:
        raise ValueError('sealed B source/sidecar binding changed')
    meta = json.loads(side.read_text())
    if (meta['schema'] != old.SCHEMA or meta['observation_dimension'] != 535
            or meta['runtime_contract']['source_git_commit'] != cfg['source_head']
            or meta['runtime_contract']['local_contract'] != old.settings()
            or not meta['rollout_empty'] or not meta['save_load_round_trip']
            or meta['counts']['auxiliary_updates']):
        raise ValueError('strict unmodified B535 source required')
    proof_path = ROOT/cfg['source_B_evaluation_manifest']
    if sha(proof_path) != cfg['source_B_evaluation_sha256']:
        raise ValueError('sealed full B physical result binding changed')
    proof = json.loads(proof_path.read_text())
    seal = json.loads((proof_path.parent.parent/'run_manifest.json').read_text())
    provenance = proof.get('checkpoint_load_provenance', {})
    if (seal.get('lifecycle') != 'COMPLETE' or seal.get('mode') != 'eval'
            or seal.get('result') != proof or proof.get('full_task_success') is not True
            or proof.get('continuous_natural_P01') is not True or proof.get('checkpoint_model_unchanged') is not True
            or proof.get('policy_switches') != 0 or proof.get('PPO_updates') != 0 or proof.get('error') is not None
            or provenance.get('checkpoint_sha256') != cfg['source_checkpoint_sha256']
            or provenance.get('manifest_sha256') != cfg['source_manifest_sha256']):
        raise ValueError('unchanged single-package naturalP01 full B result required')
    return path, meta


def settings():
    result = copy.deepcopy(old.settings()); cfg = reference_config()
    result.update({k: cfg[k] for k in ('version', 'observation_dimension', 'observation_layout')})
    result['schema'] = cfg['schema']
    result['prior_retirement'] = cfg
    result['FL_old_frozen_innovation_retirement'] = True
    result['source_tracking_owner_revision'] = 'unchanged_native_owner_P06_observed_old_FL_innovation_retirement_v1'
    return result


def _verify_source_bytes(meta):
    changed = [p for p, digest in meta['runtime_contract']['files'].items() if sha(ROOT/p) != digest]
    if changed: raise ValueError('old B runtime bytes changed: '+repr(changed))


def contract(expected_head):
    # Old files must remain exact; new additive files do not change their bytes.
    _, meta = source_metadata()
    _verify_source_bytes(meta)
    result = old.contract(expected_head)
    result.update(experiment_id=NAME, local_contract=settings(),
        source_B_checkpoint_sha256=meta['checkpoint_sha256'],
        source_B_configuration=copy.deepcopy(meta['runtime_contract']['selected_configuration']),
        selected_configuration={'local_training.json': dict(
            path=(CONFIG/'local_training.json').relative_to(ROOT).as_posix(),
            sha256=sha(CONFIG/'local_training.json'))},
        prior_retirement_production_files={p.relative_to(ROOT).as_posix(): sha(p) for p in (Path(__file__),
            MODULE_DIR/'semantic_fl_prior_retirement_actor.py', MODULE_DIR/'semantic_fl_prior_retirement_task.py',
            CONFIG/'local_training.json')})
    return result


def tensor_observation(values, device):
    import torch
    from tensordict import TensorDict
    value = torch.tensor([values], dtype=torch.float32, device=device)
    if value.shape != (1, 536) or not bool(torch.isfinite(value).all()):
        raise ValueError('finite explicit536 observation required')
    return TensorDict({'policy': value, 'critic': value.clone()}, batch_size=[1], device=device)


def make_runner(device, *, saved_configuration=None):
    import torch
    from wlr50_clean.ppo.rl_library_wrapper import construct_runner
    _, meta = source_metadata(); cfg = copy.deepcopy(meta['runner_config'])
    cfg['device'] = device
    cfg['actor'].update(class_name=reference_config()['actor_class'], observation_layout=settings()['observation_layout'])
    if saved_configuration is not None:
        checked = copy.deepcopy(saved_configuration); checked['device'] = device
        if checked != cfg:
            raise ValueError('new runner construction recipe changed')
        cfg = checked
    env = SimpleNamespace(num_envs=1, num_actions=12, device=device, cfg={'window_task': NAME},
        get_observations=lambda: tensor_observation([0.]*536, device))
    runner = construct_runner(env, cfg, log_dir=None)
    runner.alg.optimizer = torch.optim.Adam(list(runner.alg.actor.trainable_parameters())+
        list(runner.alg.critic.parameters()), lr=meta['learning_rate'])
    runner.alg.learning_rate = meta['learning_rate']
    runner.local_configuration = copy.deepcopy(cfg); runner.logger.writer = None
    runner._semantic_policy_version = runner.alg.actor.policy_version; runner._semantic_front_replay = None
    return runner


def _parameters(runner):
    return [('actor.'+k, p) for k, p in runner.alg.actor.named_parameters() if p.requires_grad]+[
        ('critic.'+k, p) for k, p in runner.alg.critic.named_parameters()]


def migrate_from_B535(runner):
    import torch
    from wlr50_clean.ppo.semantic_training import state_hash
    from wlr50_clean.ppo.rl_library_wrapper import restore_training_rng_state
    path, meta = source_metadata()
    _verify_source_bytes(meta)
    source = old.make_runner(runner.device, saved_configuration=meta['runner_config'])
    old.load(source, path, meta['runtime_contract'])  # Original loader, OLD contract.
    before = source.alg.save(); runner.alg.actor.load_state_dict(before['actor_state_dict'], strict=True)
    if state_hash(runner.alg.actor.state_dict()) != state_hash(before['actor_state_dict']):
        raise RuntimeError('entire B actor changed during migration')
    target = runner.alg.critic.state_dict(); expanded = {}; changed = []
    if target.keys() != before['critic_state_dict'].keys():
        raise ValueError('critic keys changed')
    for name, value in before['critic_state_dict'].items():
        if value.shape == target[name].shape:
            expanded[name] = value.clone()
        elif value.ndim == 2 and value.shape[1] == 535 and target[name].shape == (value.shape[0], 536):
            expanded[name] = torch.zeros_like(target[name]); expanded[name][:, :535].copy_(value); changed.append(name)
        else:
            raise ValueError('undeclared critic shape change')
    if len(changed) != 1:
        raise ValueError('exactly one critic input extension required')
    runner.alg.critic.load_state_dict(expanded, strict=True)
    oldopt = before['optimizer_state_dict']; fresh = runner.alg.optimizer.state_dict()
    if len(oldopt['param_groups']) != 1 or len(fresh['param_groups']) != 1:
        raise ValueError('explicit one-group Adam expected')
    oldids = oldopt['param_groups'][0]['params']; newids = fresh['param_groups'][0]['params']
    op, np = _parameters(source), _parameters(runner)
    if len(op) != len(np) or len(oldids) != len(op) or len(newids) != len(np):
        raise ValueError('optimizer parameter cardinality changed')
    group = copy.deepcopy(oldopt['param_groups'][0]); group['params'] = newids
    migrated = {'state': {}, 'param_groups': [group]}; moments = []
    for (oname, oval), (nname, nval), oi, ni in zip(op, np, oldids, newids, strict=True):
        if oname != nname or oi not in oldopt['state']:
            raise ValueError('saved named Adam state missing or reordered')
        state = copy.deepcopy(oldopt['state'][oi]); collapsed = copy.deepcopy(state)
        for key, value in tuple(state.items()):
            if torch.is_tensor(value) and value.shape == oval.shape and oval.shape != nval.shape:
                if oname != 'critic.'+changed[0]:
                    raise ValueError('only critic input moment expansion permitted')
                padded = value.new_zeros(nval.shape); padded[:, :535].copy_(value)
                state[key] = padded; collapsed[key] = padded[:, :535].clone(); moments.append(oname+'/'+key)
            elif torch.is_tensor(value) and value.ndim > 0 and value.shape != nval.shape:
                raise ValueError('unexpected Adam tensor shape')
        if state_hash(collapsed) != state_hash(oldopt['state'][oi]):
            raise RuntimeError('Adam source moments/step changed')
        migrated['state'][ni] = state
    runner.alg.optimizer.load_state_dict(migrated); runner.alg.learning_rate = meta['learning_rate']
    if state_hash(runner.alg.optimizer.state_dict()) != state_hash(migrated):
        raise RuntimeError('actual inherited Adam differs after load')
    actual = runner.alg.save(); collapsed = copy.deepcopy(actual['critic_state_dict'])
    for name in changed:
        if torch.count_nonzero(collapsed[name][:, 535:]): raise RuntimeError('new critic column not zero')
        collapsed[name] = collapsed[name][:, :535].clone()
    if (state_hash(collapsed) != state_hash(before['critic_state_dict'])
            or state_hash(actual['actor_state_dict']) != state_hash(before['actor_state_dict'])):
        raise RuntimeError('inverse-projected actor/critic differs')
    runner.current_learning_iteration = meta['counts']['ppo_updates']
    runner.alg.actor.assert_frozen_state(runner.alg.optimizer)
    if any(group['lr'] != runner.alg.learning_rate for group in runner.alg.optimizer.param_groups):
        raise ValueError('actual Adam LR differs from saved algorithm LR')
    if runner.alg.storage.step or runner.alg.transition.actions is not None:
        raise RuntimeError('migration must not retain rollout or transition')
    restore_training_rng_state(meta['training_rng'], expected_seed=1001)
    lineage = dict(meta['lineage'], prior_retirement_migration=dict(
        source_checkpoint=str(path), source_checkpoint_sha256=sha(path),
        source_manifest_sha256=reference_config()['source_manifest_sha256'], source_state_hashes=meta['state_hashes'],
        inherited_counts=copy.deepcopy(meta['counts']), critic_zero_columns=changed, Adam_zero_columns=moments,
        entire_B_actor_preserved=True, entire_B_Adam_inherited=True, actual_learning_rate=meta['learning_rate'],
        RNG_preserved=True, normalizers_unchanged=True, old_rollout_reused=False, added_PPO_or_AUX=0,
        reference_routing_is_not_PPO_gain=True, source_B_physical_evaluation_status='FULL_TRAVERSAL_AND_FINISH_SUCCESS_FL_IMPROVEMENT_FAILED',
        source_B_evaluation_manifest=reference_config()['source_B_evaluation_manifest'],
        source_B_evaluation_sha256=reference_config()['source_B_evaluation_sha256']))
    return lineage, copy.deepcopy(meta['counts'])


def retirement_counts(lineage, counts):
    origin = lineage['prior_retirement_migration']['inherited_counts']
    if origin != source_metadata()[1]['counts']:
        raise ValueError('inherited counter origin differs from immutable B source')
    if counts.keys() != origin.keys() or any(type(v) is not int or v < origin[k] for k, v in counts.items()):
        raise ValueError('invalid inherited/new counter split')
    return {k: counts[k]-origin[k] for k in counts}


def validate_runner(runner, runtime):
    if (runtime['experiment_id'] != NAME or runtime['local_contract'] != settings()
            or runner.alg.actor.policy_version != settings()['version'] or runner.alg.actor.obs_dim != 536):
        raise ValueError('strict536 runtime/layout binding differs')
    runner.alg.actor.assert_frozen_state(runner.alg.optimizer)
    if any(g['lr'] != runner.alg.learning_rate for g in runner.alg.optimizer.param_groups):
        raise ValueError('actual optimizer and algorithm learning rates differ')


def save(runner, runtime, lineage, counts, *, source_run, publish_pointer=True):
    import torch
    from wlr50_clean.ppo.semantic_training import state_hash
    from wlr50_clean.ppo.rl_library_wrapper import capture_training_rng_state
    validate_runner(runner, runtime); added = retirement_counts(lineage, counts)
    if counts['auxiliary_updates'] or runner.alg.storage.step or runner.alg.transition.actions is not None:
        raise RuntimeError('AUX0 and empty complete-update boundary required')
    target = OUTPUT/'checkpoints/history'/('checkpoint_CP%d_FL%06d_data%06d_ret%06d_g%s.pt' % (
        settings()['historical_decision_origin']+counts['fl_policy_decisions'], counts['fl_policy_decisions'],
        counts['continuous_storage_decisions'], added['continuous_storage_decisions'], runtime['source_git_commit'][:12]))
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists(): raise FileExistsError(target)
    payload = runner.alg.save(); hashes = {k: state_hash(v) for k, v in payload.items()}
    infos = dict(schema=SCHEMA, runtime_contract=runtime, lineage=copy.deepcopy(lineage), counts=dict(counts),
        retirement_counts=added, runner_config=copy.deepcopy(runner.local_configuration), state_hashes=hashes,
        learning_rate=runner.alg.learning_rate, training_rng=capture_training_rng_state(seed=1001),
        source_run=str(source_run), rollout_empty=True, observation_dimension=536,
        front_FL_assist=True, rear_task_assists=False, public_preparation_module=True, public_finish_module=True,
        historical_AUX_updates=lineage['historical_AUX_updates'], local_auxiliary_events=[],
        physical_evaluation_status='NOT_NEWLY_EVALUATED', reference_routing_is_not_PPO_gain=True)
    payload.update(infos=infos, iter=counts['ppo_updates']); torch.save(payload, target)
    restored = torch.load(target, map_location=runner.device, weights_only=False)
    if restored['infos'] != infos or any(state_hash(restored[k]) != v for k, v in hashes.items()):
        raise RuntimeError('serialized payload differs')
    runner.alg.load(restored, None, True); runner.alg.learning_rate = infos['learning_rate']; validate_runner(runner, runtime)
    if any(state_hash(runner.alg.save()[k]) != v for k, v in hashes.items()):
        raise RuntimeError('strict saved actor/critic/Adam round-trip differs')
    if capture_training_rng_state(seed=1001) != infos['training_rng']:
        raise RuntimeError('save/reload consumed training RNG')
    side = target.with_name(target.stem+'_manifest.json')
    write(side, dict(infos, checkpoint=str(target), checkpoint_sha256=sha(target), save_load_round_trip=True))
    pointer = dict(checkpoint=str(target), checkpoint_sha256=sha(target), manifest=str(side),
        manifest_sha256=sha(side), counts=dict(counts), retirement_counts=added, evaluated=False)
    if publish_pointer:
        p = OUTPUT/'checkpoints/checkpoint_last_pointer.json'; write(p, pointer, replace=p.exists())
    return pointer


def load(runner, path, runtime):
    import torch
    from wlr50_clean.ppo.semantic_training import state_hash
    from wlr50_clean.ppo.rl_library_wrapper import restore_training_rng_state, capture_training_rng_state
    path = Path(path).resolve(strict=True); side = path.with_name(path.stem+'_manifest.json')
    meta = json.loads(side.read_text()); cfg = copy.deepcopy(meta['runner_config']); cfg['device'] = runner.device
    if (meta['schema'] != SCHEMA or meta['runtime_contract'] != runtime or meta['checkpoint_sha256'] != sha(path)
            or not meta['save_load_round_trip'] or not meta['rollout_empty'] or meta['counts']['auxiliary_updates']
            or cfg != runner.local_configuration or meta['retirement_counts'] != retirement_counts(meta['lineage'], meta['counts'])):
        raise ValueError('strict new536 package binding differs')
    receipt = meta['lineage']['prior_retirement_migration']
    if receipt['source_checkpoint_sha256'] != reference_config()['source_checkpoint_sha256']:
        raise ValueError('cold B parent differs')
    payload = torch.load(path, map_location=runner.device, weights_only=False)
    if any(meta.get(k) != v for k, v in payload['infos'].items()): raise ValueError('embedded metadata differs')
    runner.alg.load(payload, None, True); runner.alg.learning_rate = meta['learning_rate']; validate_runner(runner, runtime)
    if any(state_hash(runner.alg.save()[k]) != v for k, v in meta['state_hashes'].items()):
        raise ValueError('restored actor/critic/Adam differs')
    restore_training_rng_state(meta['training_rng'], expected_seed=1001)
    if capture_training_rng_state(seed=1001) != meta['training_rng']:
        raise RuntimeError('actual restored training RNG differs')
    runner.current_learning_iteration = meta['counts']['ppo_updates']
    runner.checkpoint_load_provenance = dict(checkpoint=str(path), checkpoint_sha256=sha(path),
        manifest=str(side), manifest_sha256=sha(side), strict_actual_composite_load_verified=True)
    return copy.deepcopy(meta['lineage']), dict(meta['counts'])


def request(runner, observation, *, stochastic):
    from .semantic_fl_prior_retirement_actor import audited_retirement_request
    return audited_retirement_request(runner.alg.actor, observation,
        (lambda: runner.alg.act(observation)) if stochastic else
        (lambda: runner.alg.actor(observation, stochastic_output=False)), stochastic=stochastic)


def update_complete_rollout(runner):
    from wlr50_clean.ppo.semantic_fl_forward_actor import audited_fl_ppo_update
    if runner.alg.storage.step != 512: raise ValueError('512 actual contiguous rows required')
    return audited_fl_ppo_update(runner)  # Original PPO/Adam/statistics; no conditional-statistics patch.


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
        raise ValueError('prior-retirement check block requires512–1024 active FL decisions and512 contiguous storage')
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
                        raise RuntimeError('sampled prior-retirement raw action cannot be replaced')
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
            report.update(counts=dict(counts), retirement_counts=retirement_counts(lineage, counts), actual_phase_counts=dict(phases), actual_row_modes=dict(modes),
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
        control_contributions=dict(entire_accepted_A_frozen=True, prior_retirement_migration=lineage,
            public_preparation_module=True, public_finish_module=True, requested_finish=cfg['requested_finish'],
            reference_routing_is_not_PPO_gain=True, observed_lambda=True,
            FL_window='P06 ordinary forward only; source stops/postRR preparation/recoil unchanged',
            FL_window_reward=cfg['window_reward'], FL_trainable_parameter_count=2, FL_control_projection=False,
            FL_prior_retirement_ramp_s=reference_config()['retirement_ramp_duration_s'], FL_branch_counters=dict(counts),
            prior_retirement_counters=retirement_counts(lineage, counts), new_AUX_updates=0,
            historical_AUX_updates=lineage['historical_AUX_updates'],
            displayed_label='PPO + explicit P06 FL prior retirement; original FL assist; rear helpers OFF'))
    return old.base.old.evaluate(core, runner, runtime, lineage, counts, run, False, route=route)

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
        lineage, counts = load(runner, args.checkpoint, runtime) if args.checkpoint else migrate_from_B535(runner)
        if args.mode == 'migrate':
            result = save(runner, runtime, lineage, counts, source_run=run)
        else:
            from dataclasses import asdict, replace
            from .semantic_fl_prior_retirement_task import build_core
            from .semantic_fl_forward_task import config_from_sources
            window_config = replace(config_from_sources(ROOT/'configs/recording_motion_contract.json',
                ROOT/'configs/fsm_states.yaml'), **settings()['window_reward'])
            write(run/'FL_window_source_receipt.json', dict(asdict(window_config),
                prior_retirement_ramp_s=reference_config()['retirement_ramp_duration_s'],
                observed_reference_field=reference_config()['retirement_field'],
                source_stops_unchanged=True, physical_state_writes=0))
            core = build_core(app, config=window_config,
                ramp_duration_s=reference_config()['retirement_ramp_duration_s'])
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

