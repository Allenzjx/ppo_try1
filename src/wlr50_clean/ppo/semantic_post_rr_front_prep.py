"""Frozen accepted CP231936 composite + post-touch-only RSL-RL PPO route.

The whole predecessor, not merely its CP225280 child, is immutable. Historical
CP232448 supplies the critic and actual LR/RNG, never the predecessor actor.
New optimizer moments are explicitly new. No old rollout or AUX data is used.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

from . import semantic_rr_capture_local as old

ROOT = old.ROOT
NAME = 'ppo_post_rr_front_pair_rl_v1'
CONFIG = ROOT / 'configs' / NAME
OUTPUT = ROOT / 'outputs' / NAME
SCHEMA = 'wlr50_clean.frozen_accepted_post_rr_checkpoint.v1'
sha, write, line = old.sha, old.write, old.line
CONTROL_FILES = ('execution_profile.yaml', 'stage_task_spec.yaml', 'observation_schema.json',
                 'reward_config.yaml', 'quality_score.yaml', 'action_schema.json', 'local_training.json')


def settings():
    return json.loads((CONFIG / 'local_training.json').read_text(encoding='utf-8'))


def source_metadata(kind):
    cfg = settings()
    key = 'accepted' if kind == 'anchor' else 'critic_source' if kind == 'critic' else None
    if key is None:
        raise ValueError('source kind must be anchor or critic')
    path = ROOT / cfg[key + '_checkpoint']
    sidecar = path.with_name(path.stem + '_manifest.json')
    if sha(path) != cfg[key + '_checkpoint_sha256'] or sha(sidecar) != cfg[key + '_manifest_sha256']:
        raise ValueError('immutable ' + kind + ' checkpoint/manifest changed')
    meta = json.loads(sidecar.read_text())
    if (meta['checkpoint_sha256'] != cfg[key + '_checkpoint_sha256']
            or not meta.get('save_load_round_trip') or not meta.get('rollout_empty')
            or meta['runner_config']['actor']['observation_layout'] != 'role439_rr_rl_continuous_v3'):
        raise ValueError('sealed accepted465 source required')
    return path, meta


def contract(expected_head):
    from .semantic_cli import runtime_contract
    cfg = settings()
    result = runtime_contract(expected_head=expected_head, semantic_version='v3',
                              experiment_id='rr_rl_timing_policy_learning_v1')
    _, anchor = source_metadata('anchor')
    source_metadata('critic')
    control = ROOT / cfg['accepted_control_config']
    selected = {}
    for name in CONTROL_FILES:
        path = control / name
        expected = anchor['runtime_contract']['selected_configuration'][name]['sha256']
        if sha(path) != expected:
            raise ValueError('accepted predecessor configuration changed: ' + name)
        selected[name] = {'path':path.relative_to(ROOT).as_posix(), 'sha256':expected}
    result.update(experiment_id=NAME, local_contract=cfg, accepted_configuration=selected,
        selected_configuration={'local_training.json':{'path':str((CONFIG/'local_training.json').relative_to(ROOT)),
                                                       'sha256':sha(CONFIG/'local_training.json')}})
    return result


def tensor_observation(values, device):
    import torch
    from tensordict import TensorDict
    data = torch.tensor([values], dtype=torch.float32, device=device)
    if data.shape != (1, settings()['observation_dimension']) or not bool(torch.isfinite(data).all()):
        raise ValueError('finite explicit post-RR observation required')
    return TensorDict({'policy':data, 'critic':data.clone()}, batch_size=[1], device=device)


class PostRRCore(old.CaptureCore):
    def _encode(self):
        accepted = super()._encode()
        self.observation = tuple(accepted) + tuple(self.task.post_rr_observation())
        if len(accepted) != 465 or len(self.observation) != settings()['observation_dimension']:
            raise ValueError('accepted465 plus explicit post-touch tail required')
        return self.observation


def build_core(app):
    import yaml
    from .semantic_backend import SemanticIsaacBackend
    from .semantic_env import SemanticEpisodeEnv
    from .semantic_rr_capture_continuation_task import RRCaptureContinuationTaskConfig
    from .semantic_post_rr_front_prep_task import PostRRFrontPrepTask, PostRRFrontPrepTaskConfig
    from .semantic_post_rr_front_prep_source import MODE, controller_factory
    cfg = settings()
    control = ROOT / cfg['accepted_control_config']
    accepted = json.loads((control/'local_training.json').read_text())
    profile = yaml.safe_load((control/'execution_profile.yaml').read_text())
    task = PostRRFrontPrepTask(PostRRFrontPrepTaskConfig(**cfg['preparation']),
        continuation_config=RRCaptureContinuationTaskConfig(**accepted['continuation_task']),
        phase_caps_full12=profile['residual']['phase_caps_full12'])
    if cfg['capture_source_dispatch'] != MODE or cfg['rr_authorized_assist'] is not False:
        raise ValueError('declared front preparation with rear completion OFF required')
    factory = controller_factory(task_spec_path=control/'stage_task_spec.yaml',
        read_local_state=lambda:task.accepted.snapshot(), read_post_rr_state=task.snapshot)
    backend = SemanticIsaacBackend(app, audit_actuator_target_effect=True,
        execution_profile=control/'execution_profile.yaml', task_spec_path=control/'stage_task_spec.yaml',
        controller_factory=factory, continuation_source_pause=True)
    return PostRRCore(SemanticEpisodeEnv(backend, collect_trace=False,
        action_config=control/'execution_profile.yaml', reward_config_path=control/'reward_config.yaml',
        observation_schema_path=control/'observation_schema.json'), task=task)


def make_runner(device, seed=1001, *, saved_configuration=None):
    import torch
    from types import SimpleNamespace
    from .rl_library_wrapper import build_rsl_runner_config, construct_runner
    cfg = settings()
    _, anchor = source_metadata('anchor')
    _, critic = source_metadata('critic')
    if cfg['learning_rate'] != critic['learning_rate']:
        raise ValueError('initial LR must retain the actual latest saved LR')
    profile = SimpleNamespace(activation='elu', entropy_start=cfg['entropy_coef'],
        actor_hidden_dims=cfg['actor_hidden_dims'], critic_hidden_dims=cfg['critic_hidden_dims'],
        initial_action_std=.15, rollout_length=cfg['rollout_length'], update_epochs=cfg['update_epochs'],
        num_minibatches=cfg['num_minibatches'], clip_ratio=.2, gamma=cfg['gamma'], lam=cfg['lambda'],
        value_loss_coefficient=1., learning_rate=critic['learning_rate'], max_grad_norm=1.,
        schedule=cfg['schedule'], target_kl=cfg['target_kl'])
    configuration = build_rsl_runner_config(profile, seed=seed, max_iterations=2000, experiment_name=NAME)
    configuration['device'] = device
    anchor_cfg = copy.deepcopy(anchor['runner_config']['actor'])
    anchor_cfg.pop('class_name')
    configuration['actor'].update(
        class_name='wlr50_clean.ppo.semantic_post_rr_front_prep_actor:SemanticPostRRHistoryMLPModel',
        observation_layout=cfg['observation_layout'], frozen_anchor_configuration=anchor_cfg,
        initial_post_std=cfg['active_initial_sigma_full12'],
        distribution_cfg={'class_name':'HeteroscedasticGaussianDistribution','init_std':.15,'std_type':'log'})
    if saved_configuration is not None:
        source = copy.deepcopy(saved_configuration)
        source['device'] = device
        expected = copy.deepcopy(configuration)
        source['actor'].pop('expected_anchor_state_sha256', None)
        if source != expected:
            raise ValueError('saved post-RR runner configuration mismatch')
        configuration = copy.deepcopy(saved_configuration)
        configuration['device'] = device
    # Instantiation shapes only, never a physical transition or PPO sample.
    initial = [0.] * cfg['observation_dimension']
    env = SimpleNamespace(num_envs=1, num_actions=12, cfg={'post_rr_task':NAME}, device=device,
                          get_observations=lambda:tensor_observation(initial, device))
    runner = construct_runner(env, configuration, log_dir=None)
    params = list(runner.alg.actor.trainable_parameters()) + list(runner.alg.critic.parameters())
    runner.alg.optimizer = torch.optim.Adam(params, lr=critic['learning_rate'])
    runner.alg.learning_rate = critic['learning_rate']
    runner._semantic_policy_version = runner.alg.actor.policy_version
    runner._semantic_front_replay = None
    runner.logger.writer = None
    runner.local_configuration = copy.deepcopy(configuration)
    return runner


def expanded_critic_state(source, target):
    """Preserve all465 weights/buffers; exactly one first-layer input grows."""
    import torch
    if set(source) != set(target):
        raise ValueError('critic keys changed')
    result, changes = {}, []
    for key, value in source.items():
        shape = target[key].shape
        if value.shape == shape:
            result[key] = value.detach().clone()
        elif (value.ndim == 2 and value.shape[1] == 465
              and tuple(shape) == (value.shape[0], settings()['observation_dimension'])):
            result[key] = torch.zeros_like(target[key])
            result[key][:, :465].copy_(value)
            changes.append(key)
        else:
            raise ValueError('unexpected critic shape change: ' + key)
    if len(changes) != 1:
        raise ValueError('exactly one critic input expansion required')
    return result, changes[0]


def load_historical_source_on_device(source, path, metadata):
    """Explicit map-location exception, not a change to the old model recipe.

    The historical strict validator compares the serialized construction recipe
    including device. Keep that recipe verbatim, after proving the live recipe
    differs only in the requested map location. The ordinary old loader still
    verifies embedded metadata, AUX, every actor/critic/Adam tensor and RNG.
    No old validator or process-global setting is weakened.
    """
    serialized = copy.deepcopy(metadata['runner_config'])
    expected_live = copy.deepcopy(serialized)
    expected_live['device'] = source.device
    actual_live = copy.deepcopy(source.local_configuration)
    # A prior source load retained the immutable serialized recipe. Its already
    # declared runtime map-location remains the only permissible difference.
    if getattr(source, 'source_runtime_device_override', None) is not None:
        actual_live['device'] = source.device
    if actual_live != expected_live:
        raise ValueError('historical source configuration differs beyond device')
    receipt = dict(serialized_device=serialized['device'],runtime_device=str(source.device),
                   override_only='torch_map_location_no_tensor_or_model_recipe_change')
    source.local_configuration = serialized
    source.source_runtime_device_override = copy.deepcopy(receipt)
    old.load(source,path,metadata['runtime_contract'])
    return receipt


def initialize_from_accepted(runner):
    from .semantic_training import state_hash
    from .rl_library_wrapper import restore_training_rng_state
    from .semantic_rr_capture_local_actor import tensor_state_sha256
    anchor_path, anchor_meta = source_metadata('anchor')
    critic_path, critic_meta = source_metadata('critic')
    source = old.make_runner(runner.device, 1001, saved_configuration=anchor_meta['runner_config'])
    anchor_device = load_historical_source_on_device(source, anchor_path, anchor_meta)
    anchor_state = source.alg.actor.state_dict()
    if state_hash(anchor_state) != anchor_meta['state_hashes']['actor_state_dict']:
        raise ValueError('actual accepted full composite differs from manifest')
    anchor_hash = runner.alg.actor.load_frozen_anchor_state(anchor_state,
        expected_sha256=tensor_state_sha256(anchor_state))
    critic_device = load_historical_source_on_device(source, critic_path, critic_meta)
    critic_state, expanded_key = expanded_critic_state(source.alg.critic.state_dict(), runner.alg.critic.state_dict())
    runner.alg.critic.load_state_dict(critic_state, strict=True)
    if runner.alg.optimizer.state:
        raise RuntimeError('new post-RR Adam must not contain historical moments')
    runner.alg.actor.assert_frozen_state(runner.alg.optimizer)
    runner.local_configuration['actor']['expected_anchor_state_sha256'] = anchor_hash
    lineage = dict(schema='wlr50_clean.accepted_composite_post_rr_initialization.v1',
        accepted_anchor={'checkpoint':str(anchor_path),'checkpoint_sha256':sha(anchor_path),
            'manifest_sha256':sha(anchor_path.with_name(anchor_path.stem+'_manifest.json')),
            'state_hash':anchor_meta['state_hashes']['actor_state_dict'],'tensor_state_sha256':anchor_hash,
            'counts':copy.deepcopy(anchor_meta['counts']),
            'local_auxiliary_events':copy.deepcopy(anchor_meta['local_auxiliary_events'])},
        latest_preserved_critic_source={'checkpoint':str(critic_path),'checkpoint_sha256':sha(critic_path),
            'manifest_sha256':sha(critic_path.with_name(critic_path.stem+'_manifest.json')),
            'counts':copy.deepcopy(critic_meta['counts']),
            'state_hashes':copy.deepcopy(critic_meta['state_hashes'])},
        critic_expanded_input=expanded_key, critic_new_columns_zero=True,
        optimizer='new_Adam_for_new_post_head_and_independent_critic',
        old_Adam_loaded_into_new_optimizer=False, normalizers='Identity_unchanged',
        historical_gain10_reapplied=False, historical_rollout_reused=False,
        old_RR2_AUX_package_used=False, learning_rate=critic_meta['learning_rate'],
        source_runtime_device_overrides={'accepted_anchor':anchor_device,'latest_critic':critic_device},
        rng_origin='restored_actual_CP232448_after_new_parameter_construction',
        old512_is_new_RL_T1=False)
    runner.post_rr_lineage = copy.deepcopy(lineage)
    restore_training_rng_state(critic_meta['training_rng'], expected_seed=1001)
    counts = dict(post_rr_policy_decisions=0,post_rr_ppo_updates=0,post_rr_optimizer_steps=0,
        prefix_decisions=0,post_rr_opportunities=0,completed_episodes=0,full_successes=0,auxiliary_updates=0)
    return lineage, counts


def validate_runner(runner, runtime):
    cfg = settings()
    if (runtime.get('experiment_id') != NAME or runtime.get('local_contract') != cfg
            or runner.alg.actor.policy_version != cfg['version']
            or runner.alg.actor.observation_dimension != cfg['observation_dimension']):
        raise ValueError('post-RR runtime/actor identity mismatch')
    runner.alg.actor.assert_frozen_state(runner.alg.optimizer)


def checkpoint_path(counts, runtime):
    n = counts['post_rr_policy_decisions']
    return OUTPUT/'checkpoints/history'/('checkpoint_CP%d_postRR%06d_g%s.pt' % (
        settings()['historical_decision_origin']+n,n,runtime['source_git_commit'][:12]))


def save(runner, runtime, lineage, counts, *, source_run, publish_pointer=True):
    import torch
    from .semantic_training import state_hash
    from .rl_library_wrapper import capture_training_rng_state
    validate_runner(runner,runtime)
    if counts['auxiliary_updates'] != 0:
        raise ValueError('this new branch has no authorized AUX implementation')
    if runner.alg.storage.step != 0 or runner.alg.transition.actions is not None:
        raise RuntimeError('complete update and empty rollout required')
    target = checkpoint_path(counts,runtime)
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    payload = runner.alg.save()
    hashes = {k:state_hash(v) for k,v in payload.items()}
    infos = dict(schema=SCHEMA,runtime_contract=runtime,lineage=copy.deepcopy(lineage),counts=dict(counts),
        runner_config=copy.deepcopy(runner.local_configuration),state_hashes=hashes,
        learning_rate=runner.alg.learning_rate,training_rng=capture_training_rng_state(seed=1001),
        source_run=str(source_run),rollout_empty=True,front_FL_assist=True,rear_task_assists=False,
        public_preparation_module=True,observation_dimension=settings()['observation_dimension'],
        local_auxiliary_events=[],historical_AUX_updates=lineage['accepted_anchor']['counts']['auxiliary_updates'])
    payload.update(infos=infos,iter=counts['post_rr_ppo_updates'])
    torch.save(payload,target)
    reloaded = torch.load(target,map_location=runner.device,weights_only=False)
    if reloaded['infos'] != infos or any(state_hash(reloaded[k]) != v for k,v in hashes.items()):
        raise RuntimeError('serialized checkpoint state/metadata changed')
    runner.alg.load(reloaded,None,True)
    runner.alg.learning_rate = infos['learning_rate']
    validate_runner(runner,runtime)
    if any(state_hash(runner.alg.save()[k]) != v for k,v in hashes.items()):
        raise RuntimeError('actual complete state reload changed')
    manifest = target.with_name(target.stem+'_manifest.json')
    write(manifest,dict(infos,checkpoint=str(target),checkpoint_sha256=sha(target),save_load_round_trip=True))
    pointer = dict(checkpoint=str(target),checkpoint_sha256=sha(target),manifest=str(manifest),
                   manifest_sha256=sha(manifest),counts=dict(counts),evaluated=False)
    if publish_pointer:
        p = OUTPUT/'checkpoints/checkpoint_last_pointer.json'
        write(p,pointer,replace=p.exists())
    return pointer


def load(runner,path,runtime):
    import torch
    from .semantic_training import state_hash
    from .rl_library_wrapper import restore_training_rng_state
    path = Path(path).resolve(strict=True)
    manifest = path.with_name(path.stem+'_manifest.json')
    meta = json.loads(manifest.read_text())
    if (meta['schema'] != SCHEMA or meta['runtime_contract'] != runtime
            or meta['checkpoint_sha256'] != sha(path) or not meta['save_load_round_trip']
            or meta['counts']['auxiliary_updates'] != 0 or meta.get('local_auxiliary_events') != []):
        raise ValueError('new post-RR checkpoint contract/hash/ledger mismatch')
    expected_cfg = copy.deepcopy(meta['runner_config']); expected_cfg['device'] = runner.device
    current_cfg = copy.deepcopy(runner.local_configuration)
    expected_hash = expected_cfg['actor']['expected_anchor_state_sha256']
    current_cfg['actor'].setdefault('expected_anchor_state_sha256',expected_hash)
    if current_cfg != expected_cfg:
        raise ValueError('actual constructed runner differs from saved recipe')
    cfg = settings()
    anchor = meta['lineage']['accepted_anchor']
    if (anchor['checkpoint_sha256'] != cfg['accepted_checkpoint_sha256']
            or anchor['manifest_sha256'] != cfg['accepted_manifest_sha256']
            or anchor['tensor_state_sha256'] != expected_hash):
        raise ValueError('accepted whole-composite binding changed')
    runner.alg.actor.expected_anchor_state_sha256 = expected_hash
    data = torch.load(path,map_location=runner.device,weights_only=False)
    if any(meta.get(k) != v for k,v in data['infos'].items()):
        raise ValueError('embedded checkpoint metadata differs')
    runner.alg.load(data,None,True)
    runner.alg.learning_rate = meta['learning_rate']
    runner.local_configuration = expected_cfg
    validate_runner(runner,runtime)
    if any(state_hash(runner.alg.save()[k]) != v for k,v in meta['state_hashes'].items()):
        raise ValueError('actual reloaded actor/critic/Adam state differs')
    runner.current_learning_iteration = meta['counts']['post_rr_ppo_updates']
    runner.post_rr_lineage = copy.deepcopy(meta['lineage'])
    restore_training_rng_state(meta['training_rng'],expected_seed=1001)
    runner.checkpoint_load_provenance = dict(checkpoint=str(path),checkpoint_sha256=sha(path),
        manifest=str(manifest),manifest_sha256=sha(manifest),strict_actual_composite_load_verified=True,
        actor_critic_optimizer_hashes_verified=True,accepted_whole_composite_frozen=True)
    return copy.deepcopy(meta['lineage']),dict(meta['counts'])


def request(runner,observation,*,stochastic):
    from .semantic_post_rr_front_prep_actor import audited_post_rr_policy_request
    return audited_post_rr_policy_request(runner.alg.actor,observation,
        (lambda:runner.alg.act(observation)) if stochastic else
        (lambda:runner.alg.actor(observation,stochastic_output=False)),stochastic=stochastic)


def prefix(core,runner,stream,counts):
    import torch
    core.reset(seed=1001)
    start = counts['prefix_decisions']
    while not core.task.snapshot()['post_rr_active'] and not core.done:
        obs = tensor_observation(core.observation,runner.device)
        with torch.inference_mode():
            raw,audit = request(runner,obs,stochastic=False)
        step = core.step(raw[0].cpu().tolist())
        counts['prefix_decisions'] += 1
        line(stream,dict(kind='frozen_accepted_composite_prefix',PPO_credit=0,
            physical_tick=core.frame.physics_tick,phase=core.frame.state_id,policy_request=audit,
            local=core.task.snapshot()))
    if core.done:
        raise RuntimeError('accepted predecessor ended before post-touch learning: '+str(step.info.get('termination_reason')))
    counts['post_rr_opportunities'] += 1
    print(json.dumps({'post_touch_entry':core.task.snapshot(),
        'uncredited_prefix_decisions':counts['prefix_decisions']-start}),flush=True)


def train(core,runner,runtime,lineage,counts,run,decisions):
    import torch
    from .semantic_training import audited_ppo_update,verified_native_effect,_stop_request
    if decisions <= 0 or decisions % 512 or settings()['rollout_length'] != 512:
        raise ValueError('post-touch training requires positive complete512 budget')
    runner.alg.train_mode()
    (run/'rollouts').mkdir()
    with (run/'decisions.jsonl').open('x') as stream,(run/'updates.jsonl').open('x') as updates:
        for _ in range(decisions//512):
            phases,coverage = Counter(),Counter()
            for index in range(512):
                if core.done:
                    prefix(core,runner,stream,counts)
                obs = tensor_observation(core.observation,runner.device)
                before = core.task.snapshot()
                if not before['post_rr_active'] or float(obs['policy'][0,465]) != 1.:
                    raise RuntimeError('pre-touch samples cannot enter new PPO storage')
                with torch.inference_mode():
                    raw,audit = request(runner,obs,stochastic=True)
                    selected = raw.detach().clone()
                    old_logp = runner.alg.transition.actions_log_prob.detach().clone()
                    step = core.step(raw[0].cpu().tolist())
                    issued = step.info.get('applied_raw_full12',raw[0].cpu().tolist())
                    if list(issued) != raw[0].cpu().tolist():
                        raise RuntimeError('this route cannot disguise a runtime teacher as a policy sample')
                    verified_native_effect(step.info,tuple(issued))
                    nxt = tensor_observation(step.observation,runner.device)
                    done = torch.tensor([step.terminated],dtype=torch.bool,device=runner.device)
                    runner.alg.process_env_step(nxt,torch.tensor([step.reward],device=runner.device),done,
                                                {'time_outs':torch.zeros_like(done)})
                    if (not torch.equal(runner.alg.storage.actions[index],selected)
                            or not torch.equal(runner.alg.storage.actions_log_prob[index].view(-1),old_logp.view(-1))):
                        raise RuntimeError('stored raw sample/old composed likelihood changed')
                counts['post_rr_policy_decisions'] += 1
                phases[step.info['phase_id']] += 1
                coverage[before['post_rr_stage']] += 1
                for name in ('rr_contact_now','rr_bearing_now','rr_support_continuation_valid','post_rr_prepared'):
                    coverage[name] += int(before[name])
                for name in ('current_qualified','current_air_lift_valid','current_top_contact'):
                    coverage['RL_'+name] += int(before['rl_metrics'].get(name,False))
                line(stream,dict(kind='post_touch_on_policy',PPO_credit=1,
                    global_decision=settings()['historical_decision_origin']+counts['post_rr_policy_decisions'],
                    post_rr_decision=counts['post_rr_policy_decisions'],observation=obs['policy'][0].cpu().tolist(),
                    policy_request=audit,old_logp=old_logp.cpu().tolist(),before_local=before,step_info=step.info))
                if step.terminated:
                    counts['completed_episodes'] += 1
                    counts['full_successes'] += int(step.info.get('full_task_success',False))
                if (index+1)%128 == 0:
                    print(json.dumps({'post_rr_collected':counts['post_rr_policy_decisions'],
                        'phase':core.frame.state_id,'time_s':core.frame.sim_time_s,'local':core.task.snapshot()}),flush=True)
            with torch.inference_mode():
                runner.alg.compute_returns(nxt)
            storage = runner.alg.storage
            snapshot = {k:getattr(storage,k).detach().cpu().clone() for k in
                        ('actions','actions_log_prob','rewards','dones','values','returns','advantages')}
            snapshot['observations'] = {k:v.detach().cpu().clone() for k,v in storage.observations.items()}
            snapshot['distribution_params'] = tuple(v.detach().cpu().clone() for v in storage.distribution_params)
            number = counts['post_rr_ppo_updates']+1
            torch.save(snapshot,run/'rollouts'/f'rollout_{number:04d}.pt')
            report = audited_ppo_update(runner,likelihood_audit_path=run/'rollouts'/f'likelihood_{number:04d}.json')
            validate_runner(runner,runtime)
            counts['post_rr_ppo_updates'] += 1
            counts['post_rr_optimizer_steps'] += report['optimizer_steps']
            runner.current_learning_iteration = counts['post_rr_ppo_updates']
            report.update(counts=dict(counts),actual_phase_counts=dict(phases),
                actual_optimizer_sample_coverage=dict(coverage),coverage_overlaps_not_additive=True,
                accepted_whole_composite_unchanged=True,AUX_updates=0)
            line(updates,report)
            pointer = save(runner,runtime,lineage,counts,source_run=run)
            print(json.dumps({'completed_update':number,'checkpoint':pointer}),flush=True)
            stop = _stop_request(run,runtime)
            if stop is not None:
                write(run/'stop_after_update.accepted.json',dict(stop,checkpoint=pointer,rollout_empty=True))
                break
    return pointer


def evaluate(core,runner,runtime,lineage,counts,run):
    cfg = settings()
    route = dict(validate_runner=validate_runner,auxiliary_ledger=[],settings=cfg,
        config_dir=ROOT/cfg['accepted_control_config'],tensor_observation=tensor_observation,request=request,
        control_contributions=dict(accepted_full_composite_anchor=lineage['accepted_anchor'],
            public_preparation_module=True,preparation_channels=[1,3,8,9],
            preparation_candidate=cfg['preparation'],historical_AUX_updates=lineage['accepted_anchor']['counts']['auxiliary_updates'],
            new_AUX_updates=0,post_touch_branch_counters=dict(counts),
            displayed_label='PPO + local preparation module; full CP231936 anchor frozen; rear helpers OFF'))
    return old.evaluate(core,runner,runtime,lineage,counts,run,False,route=route)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('initialize','train','eval'))
    parser.add_argument('--expected-head',required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--decisions',type=int,default=512)
    parser.add_argument('--device',default='cuda:0')
    args = parser.parse_args()
    runtime = contract(args.expected_head)
    run = args.run_dir.resolve()
    if not run.is_relative_to(ROOT/'runs'/NAME):
        raise ValueError('new route requires its isolated runs namespace')
    if args.mode != 'initialize' and args.checkpoint is None:
        raise ValueError('train/eval must reload the same packaged checkpoint')
    run.mkdir(parents=True,exist_ok=False)
    write(run/'run_manifest.started.json',dict(runtime_contract=runtime,mode=args.mode,
        checkpoint=str(args.checkpoint),started_utc=datetime.now(timezone.utc).isoformat()))
    app = None
    try:
        import torch
        import tensordict
        from .rl_library_wrapper import seed_training_rngs
        if args.mode != 'initialize':
            from isaaclab.app import AppLauncher
            app = AppLauncher(headless=args.mode=='train',enable_cameras=False).app
            app.update()
        seed_training_rngs(1001)
        saved = (json.loads(args.checkpoint.with_name(args.checkpoint.stem+'_manifest.json').read_text())
                 if args.checkpoint else None)
        runner = make_runner(args.device,1001,saved_configuration=saved['runner_config'] if saved else None)
        lineage,counts = (load(runner,args.checkpoint,runtime) if args.checkpoint else initialize_from_accepted(runner))
        if args.mode == 'initialize':
            result = save(runner,runtime,lineage,counts,source_run=run)
        else:
            core = build_core(app)
            if args.mode == 'train':
                result = train(core,runner,runtime,lineage,counts,run,args.decisions)
            else:
                runner.alg.eval_mode()
                result = evaluate(core,runner,runtime,lineage,counts,run)
        write(run/'run_manifest.json',dict(lifecycle='COMPLETE',mode=args.mode,result=result))
        print(json.dumps({'run':str(run),'lifecycle':'COMPLETE'}),flush=True)
    except BaseException:
        write(run/'failure.json',{'traceback':traceback.format_exc()})
        raise
    finally:
        if app is not None:
            app.close(wait_for_replicator=False,skip_cleanup=True)


if __name__ == '__main__':
    main()
