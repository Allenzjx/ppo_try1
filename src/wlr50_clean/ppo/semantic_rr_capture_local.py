"""Isolated frozen-CP225280/local-RR PPO route, using official RSL-RL PPO.

The existing semantic routes and validators are unchanged. The six control
files are the accepted49eb configuration; only the new observed local task,
actor and critic differ. No task helper writes rear actuator targets.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import traceback

ROOT = Path(__file__).resolve().parents[3]
NAME = 'ppo_rr_capture_first_cp225280_v1'
CONFIG = ROOT / 'configs' / NAME
OUTPUT = ROOT / 'outputs' / NAME
SOURCE = ROOT / 'outputs/ppo_rr_rl_timing_policy_learning_v1/branches/ancestor220544_recapture_v2/checkpoints/history/checkpoint_rear_owner_CP225280_gf6d1d2df8d87.pt'
SOURCE_MANIFEST = SOURCE.with_name(SOURCE.stem + '_manifest.json')
SCHEMA = 'wlr50_clean.frozen_prior_rr_capture_checkpoint.v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def settings():
    return json.loads((CONFIG / 'local_training.json').read_text(encoding='utf-8'))


def write(path, data, *, replace=False):
    from .semantic_training import write_json
    write_json(Path(path), data, replace=replace)


def line(stream, data):
    from .semantic_legacy_evaluation import physical_json
    stream.write(json.dumps(physical_json(data), allow_nan=False) + '\n')
    stream.flush()


def interval_receipt(ticks):
    if type(ticks) is not int or not 0 < ticks <= 24000:
        raise ValueError('local capture video requires a real endpoint inside200s')
    count=(ticks+7)//8
    final_ticks=ticks-8*(count-1)
    return {'schema':'wlr50_clean.task_interval_video_window.v1','experiment_id':NAME,
        'first_episode_tick':0,'endpoint_episode_tick':ticks,'frame_count':count,
        'frame_sample':'actual_executed_interval_right_endpoint',
        'first_frame_episode_tick':min(8,ticks),'last_frame_episode_tick':ticks,
        'final_interval_physics_ticks':final_ticks,
        'terminal_frame_display_quantization_s':(8-final_ticks)/120.,
        'encoded_duration_s':count/15.,'physical_duration_s':ticks/120.,
        'tick0_observation_retained':True,'tick0_encoded_as_extra_frame':False,
        'extra_physics_ticks':0,'extra_pre_frames':0,'extra_post_frames':0}


def contract(expected_head):
    from .semantic_cli import runtime_contract
    result = runtime_contract(expected_head=expected_head, semantic_version='v3',
                              experiment_id='rr_rl_timing_policy_learning_v1')
    result.update(experiment_id=NAME, local_contract=settings(),
                  selected_configuration={p.name: {'path': p.relative_to(ROOT).as_posix(),
                                                    'sha256': sha(p)}
                                          for p in sorted(CONFIG.iterdir()) if p.is_file()})
    cfg = settings()
    if sha(SOURCE) != cfg['prior_checkpoint_sha256'] or sha(SOURCE_MANIFEST) != cfg['prior_manifest_sha256']:
        raise ValueError('immutable prior checkpoint/manifest binding changed')
    # Select the exact accepted control/observation recipe, not just old weights
    # under a more recent owner/proxy interface. No global checkout/revert.
    for name in ('execution_profile.yaml', 'stage_task_spec.yaml', 'observation_schema.json',
                 'reward_config.yaml', 'quality_score.yaml', 'action_schema.json'):
        original = subprocess.run(['git', 'show', cfg['control_configuration_source'] +
            ':configs/ppo_rr_rl_timing_policy_learning_v1/' + name], cwd=ROOT,
            check=True, capture_output=True).stdout.decode('utf-8').replace('\r\n', '\n')
        if (CONFIG / name).read_text(encoding='utf-8').replace('\r\n', '\n') != original:
            raise ValueError('accepted CP225280 configuration changed: ' + name)
    return result


def tensor_observation(values, device):
    import torch
    from tensordict import TensorDict
    data = torch.tensor([values], dtype=torch.float32, device=device)
    if tuple(data.shape) != (1, 447) or not bool(torch.isfinite(data).all()):
        raise ValueError('finite447 local observation required')
    return TensorDict({'policy': data, 'critic': data.clone()}, batch_size=[1], device=device)


class CaptureCore:
    """Task-only wrapper; unchanged12-channel physics/mapper and native history."""
    def __init__(self, inner):
        from .semantic_rr_capture_local_task import RRCaptureLocalTask
        self.inner = inner
        self.task = RRCaptureLocalTask()
        self.tick_observer = None
        self.inner.tick_observer = self._observe
        self.observation = None
        self.done = True

    def __getattr__(self, key):
        return getattr(self.inner, key)

    def _observe(self, before, after, projection):
        self.task.observe(after)
        if self.tick_observer is not None:
            self.tick_observer(before, after, projection)

    def _encode(self):
        # The owner controller is absent, not a hidden unobserved transform.
        if len(self.inner.observation) != 422:
            raise ValueError('accepted CP225280 control must expose original422')
        self.observation = tuple(self.inner.observation) + (0.,) * 17 + tuple(self.task.obs8())
        return self.observation

    def reset(self, seed=1001):
        self.inner.reset(seed=seed)
        self.task.reset()
        self.task.observe(self.inner.frame)
        self.done = False
        return self._encode()

    def step(self, raw):
        from .residual_direct_env import ResidualStep
        if self.done:
            raise RuntimeError('cannot advance a completed local task')
        before = self.task.snapshot()
        step = self.inner.step(raw)
        local = self.task.reward(before, termination_reason=step.info.get('termination_reason'))
        snapshot = self.task.snapshot()
        self.done = bool(step.terminated or snapshot['local_success'])
        info = dict(step.info)
        info.update(rr_capture_local=snapshot, local_reward=local,
                    local_task_success=bool(snapshot['local_success']),
                    full_task_success=bool(step.info.get('full_task_success')),
                    original_global_reward_not_optimized=step.reward)
        if snapshot['local_success'] and not step.terminated:
            info.update(termination_reason='RR_CAPTURE_HOLD_LOCAL_SUCCESS',
                        task_outcome_label='RR_LOCAL_SUCCESS_NOT_FULL_TASK',
                        terminal_bootstrap_allowed=False)
        return ResidualStep(self._encode(), local['reward'], self.done, False, info)


def build_core(app):
    from .semantic_backend import SemanticIsaacBackend
    from .semantic_env import SemanticEpisodeEnv
    backend = SemanticIsaacBackend(app, audit_actuator_target_effect=True,
        execution_profile=CONFIG/'execution_profile.yaml', task_spec_path=CONFIG/'stage_task_spec.yaml')
    return CaptureCore(SemanticEpisodeEnv(backend, collect_trace=False,
        action_config=CONFIG/'execution_profile.yaml', reward_config_path=CONFIG/'reward_config.yaml',
        observation_schema_path=CONFIG/'observation_schema.json'))


def make_runner(device, seed):
    import torch
    from types import SimpleNamespace
    from .rl_library_wrapper import build_rsl_runner_config, construct_runner
    cfg = settings()
    # Real saved regression observation used only to instantiate tensor shapes;
    # never treated as an environment transition or copied into PPO storage.
    data = json.loads((ROOT/'outputs/ppo_rl_recovery_learning_v1/staged_cp225280_front_branch/CP225280_front_replay_dataset.json').read_text())
    initial = data['training_rows'][0]['observation'][:422] + [0.] * 25
    env = SimpleNamespace(num_envs=1, num_actions=12, cfg={'local_capture_task': NAME},
        device=device, get_observations=lambda: tensor_observation(initial, device))
    profile = SimpleNamespace(activation='elu', entropy_start=cfg['entropy_coef'],
        actor_hidden_dims=cfg['actor_hidden_dims'], critic_hidden_dims=cfg['critic_hidden_dims'],
        initial_action_std=.15, rollout_length=cfg['rollout_length'], update_epochs=cfg['update_epochs'],
        num_minibatches=cfg['num_minibatches'], clip_ratio=.2, gamma=cfg['gamma'], lam=cfg['lambda'],
        value_loss_coefficient=1., learning_rate=cfg['learning_rate'], max_grad_norm=1.,
        schedule=cfg['schedule'], target_kl=cfg['target_kl'])
    configuration = build_rsl_runner_config(profile, seed=seed, max_iterations=2000, experiment_name=NAME)
    configuration['device'] = device
    configuration['actor'].update(
        class_name='wlr50_clean.ppo.semantic_rr_capture_local_actor:SemanticRRCaptureLocalHistoryMLPModel',
        observation_layout='role439_rr_capture_local_v1',
        distribution_cfg={'class_name': 'HeteroscedasticGaussianDistribution', 'init_std': .15, 'std_type': 'log'},
        initial_capture_std=cfg['active_initial_sigma_full12'])
    runner = construct_runner(env, configuration, log_dir=None)
    # Explicit exclusion of prior parameters, including weight decay/momentum.
    params = list(runner.alg.actor.trainable_parameters()) + list(runner.alg.critic.parameters())
    runner.alg.optimizer = torch.optim.Adam(params, lr=cfg['learning_rate'])
    runner._semantic_policy_version = 'frozen_prior_rr_capture_local447_v1'
    runner._semantic_front_replay = None
    runner.logger.writer = None
    runner.local_configuration = copy.deepcopy(configuration)
    return runner


def initialize_prior(runner):
    import torch
    from .semantic_training import parameter_hash
    metadata = json.loads(SOURCE_MANIFEST.read_text())
    payload = torch.load(SOURCE, map_location=runner.device, weights_only=False)
    for key, value in payload['infos'].items():
        if key not in metadata or metadata[key] != value:
            raise ValueError('prior embedded metadata differs from sidecar: ' + key)
    runner.alg.actor.load_frozen_prior_state(payload['actor_state_dict'])
    if parameter_hash(runner.alg.actor.frozen_prior) != metadata['actor_parameter_sha256']:
        raise ValueError('actual loaded prior tensor hash differs')
    # The zero-learning439 publication must preserve every original422 weight.
    original = SOURCE.with_name('checkpoint_step_000225280.pt')
    original_payload = torch.load(original, map_location=runner.device, weights_only=False)
    old = original_payload['actor_state_dict']
    new = runner.alg.actor.frozen_prior.state_dict()
    for key, value in old.items():
        target = new[key]
        if value.shape != target.shape:
            if value.ndim != 2 or value.shape[1] != 422 or target.shape[1] != 439:
                raise ValueError('unexpected CP225280 prior topology migration')
            if not torch.equal(value, target[:, :422]) or torch.count_nonzero(target[:, 422:]):
                raise ValueError('prior original422 columns were not preserved')
        elif not torch.equal(value, target):
            raise ValueError('prior original tensor changed: ' + key)
    runner.alg.actor.assert_frozen_state(runner.alg.optimizer)
    return {'path': str(SOURCE), 'sha256': sha(SOURCE), 'manifest_sha256': sha(SOURCE_MANIFEST),
            'historical_counters': {key:metadata[key] for key in
                ('global_policy_decisions','ppo_updates','optimizer_steps')},
            'inherited_auxiliary_lineage': 'retained_in_immutable_source_manifest_not_new_local_PPO',
            'original422_checkpoint': str(original), 'original422_sha256': sha(original),
            'original422_tensor_identity_verified': True, 'prior_optimizer_loaded': False,
            'new_local_mean_initialized_zero': True, 'prior_weights_fully_frozen': True}


def checkpoint_path(decisions):
    return OUTPUT/'checkpoints/history'/f'checkpoint_CP{225280+decisions}_local{decisions:06d}.pt'


def save(runner, runtime, prior, counts, *, source_run):
    import torch
    from .semantic_training import state_hash
    from .rl_library_wrapper import capture_training_rng_state
    target = checkpoint_path(counts['local_policy_decisions'])
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    if runner.alg.storage.step != 0 or runner.alg.transition.actions is not None:
        raise RuntimeError('checkpoint publication requires a complete update and empty rollout')
    runner.alg.actor.assert_frozen_state(runner.alg.optimizer)
    payload = runner.alg.save()
    identity = {key: state_hash(value) for key, value in payload.items()}
    infos = dict(schema=SCHEMA, runtime_contract=runtime, prior=prior, counts=dict(counts),
        source_run=str(source_run), runner_config=runner.local_configuration,
        state_hashes=identity, learning_rate=runner.alg.learning_rate,
        training_rng=capture_training_rng_state(seed=1001),
        rollout_empty=True, front_FL_assist=True, rear_task_assists=False,
        full_task_success=False, local_task_semantics=settings()['local_terminal'])
    payload.update(infos=infos, iter=counts['local_ppo_updates'])
    torch.save(payload, target)
    loaded = torch.load(target, map_location=runner.device, weights_only=False)
    if any(state_hash(loaded[key]) != value for key, value in identity.items()):
        raise RuntimeError('checkpoint serialized round trip changed training state')
    runner.alg.load(loaded, None, True)
    runner.alg.learning_rate = infos['learning_rate']
    runner.alg.actor.assert_frozen_state()
    if any(state_hash(runner.alg.save()[key]) != value for key, value in identity.items()):
        raise RuntimeError('actual save/load changed actor/critic/optimizer')
    manifest = target.with_name(target.stem + '_manifest.json')
    write(manifest, dict(infos, checkpoint=str(target), checkpoint_sha256=sha(target), save_load_round_trip=True))
    pointer = {'checkpoint': str(target), 'checkpoint_sha256': sha(target), 'manifest': str(manifest),
               'manifest_sha256': sha(manifest), 'counts': dict(counts), 'evaluated': False}
    write(OUTPUT/'checkpoints/checkpoint_last_pointer.json', pointer, replace=(OUTPUT/'checkpoints/checkpoint_last_pointer.json').exists())
    return pointer


def load(runner, path, runtime):
    import torch
    from .semantic_training import state_hash, parameter_hash
    from .rl_library_wrapper import restore_training_rng_state
    path = Path(path).resolve(strict=True)
    metadata = json.loads(path.with_name(path.stem + '_manifest.json').read_text())
    if (metadata['schema'] != SCHEMA or metadata['runtime_contract'] != runtime
            or metadata['checkpoint_sha256'] != sha(path) or not metadata['save_load_round_trip']):
        raise ValueError('new-route checkpoint contract/hash mismatch')
    data = torch.load(path, map_location=runner.device, weights_only=False)
    if any(metadata.get(k) != v for k, v in data['infos'].items()):
        raise ValueError('new checkpoint embedded infos differ from manifest')
    runner.alg.load(data, None, True)
    runner.alg.actor.assert_frozen_state(runner.alg.optimizer)
    source_metadata = json.loads(SOURCE_MANIFEST.read_text())
    if (metadata['prior']['sha256'] != settings()['prior_checkpoint_sha256']
            or parameter_hash(runner.alg.actor.frozen_prior) != source_metadata['actor_parameter_sha256']):
        raise ValueError('composite checkpoint no longer contains the immutable CP225280 prior')
    for key, value in metadata['state_hashes'].items():
        if state_hash(runner.alg.save()[key]) != value:
            raise ValueError('actual reloaded state differs: ' + key)
    runner.alg.learning_rate = metadata['learning_rate']
    runner.current_learning_iteration = metadata['counts']['local_ppo_updates']
    restore_training_rng_state(metadata['training_rng'], expected_seed=1001)
    runner.checkpoint_load_provenance = {'checkpoint': str(path), 'checkpoint_sha256':sha(path),
        'manifest':str(path.with_name(path.stem+'_manifest.json')),
        'manifest_sha256':sha(path.with_name(path.stem+'_manifest.json')),
        'strict_actual_composite_load_verified':True,'actor_critic_optimizer_hashes_verified':True}
    return metadata['prior'], metadata['counts']


def request(runner, observation, *, stochastic):
    from .semantic_rr_capture_local_actor import audited_capture_local_policy_request
    return audited_capture_local_policy_request(runner.alg.actor, observation,
        (lambda: runner.alg.act(observation)) if stochastic else
        (lambda: runner.alg.actor(observation, stochastic_output=False)), stochastic=stochastic)


def prefix(core, runner, stream, counts):
    import torch
    core.reset(seed=1001)
    beginning = counts['prefix_decisions']
    while not core.task.snapshot()['active'] and not core.done:
        obs = tensor_observation(core.observation, runner.device)
        with torch.inference_mode():
            raw, audit = request(runner, obs, stochastic=False)
        step = core.step(raw[0].cpu().tolist())
        counts['prefix_decisions'] += 1
        line(stream, {'kind': 'frozen_prior_prefix', 'PPO_credit': 0,
            'physical_tick': core.frame.physics_tick, 'phase': core.frame.state_id,
            'local': core.task.snapshot()})
    if core.done:
        raise RuntimeError('frozen accepted prior failed before capture opportunity: ' + str(step.info.get('termination_reason')))
    counts['capture_opportunities'] += 1
    print(json.dumps({'capture_entry': core.task.snapshot(),
                      'uncredited_prefix_decisions': counts['prefix_decisions']-beginning}), flush=True)


def train(core, runner, runtime, prior, counts, run, decisions):
    import torch
    from .semantic_training import audited_ppo_update, verified_native_effect
    if decisions <= 0 or decisions % 512:
        raise ValueError('training budget must be a positive complete512 multiple')
    runner.alg.train_mode()
    (run/'rollouts').mkdir()
    with (run/'decisions.jsonl').open('x') as stream, (run/'updates.jsonl').open('x') as updates:
        for iteration in range(decisions//512):
            if core.done:
                prefix(core, runner, stream, counts)
            phase_counts = Counter()
            for index in range(512):
                if core.done:
                    prefix(core, runner, stream, counts)
                obs = tensor_observation(core.observation, runner.device)
                if float(obs['policy'][0,439]) != 1.:
                    raise RuntimeError('inactive prefix cannot enter PPO storage')
                with torch.inference_mode():
                    raw, audit = request(runner, obs, stochastic=True)
                    old_logp = runner.alg.transition.actions_log_prob.detach().clone()
                    selected = raw.detach().clone()
                    before = core.task.snapshot()
                    step = core.step(raw[0].cpu().tolist())
                    verified_native_effect(step.info, tuple(raw[0].cpu().tolist()))
                    nxt = tensor_observation(step.observation, runner.device)
                    reward = torch.tensor([step.reward], device=runner.device)
                    done = torch.tensor([step.terminated], dtype=torch.bool, device=runner.device)
                    runner.alg.process_env_step(nxt, reward, done, {'time_outs': torch.zeros_like(done)})
                    if not torch.equal(runner.alg.storage.actions[index], selected):
                        raise RuntimeError('PPO storage action differs from issued raw sample')
                    if not torch.equal(runner.alg.storage.actions_log_prob[index].view(-1), old_logp.view(-1)):
                        raise RuntimeError('PPO old likelihood differs from composed sampled Gaussian')
                counts['local_policy_decisions'] += 1
                phase_counts[step.info['phase_id']] += 1
                line(stream, {'kind':'activated_on_policy', 'global_decision':225280+counts['local_policy_decisions'],
                    'PPO_credit':1, 'observation':obs['policy'][0].cpu().tolist(),
                    'policy_request':audit, 'old_logp':old_logp.cpu().tolist(),
                    'before_local':before, 'step_info':step.info})
                if step.terminated:
                    counts['local_successes'] += int(step.info['local_task_success'])
                    counts['completed_local_episodes'] += 1
                if (index+1)%128 == 0:
                    print(json.dumps({'collected':counts['local_policy_decisions'],
                        'phase':core.frame.state_id,'time_s':core.frame.sim_time_s,
                        'local':core.task.snapshot()}), flush=True)
            with torch.inference_mode():
                runner.alg.compute_returns(nxt)
            storage = runner.alg.storage
            snapshot = {key:getattr(storage,key).detach().cpu().clone() for key in
                ('actions','actions_log_prob','rewards','dones','values','returns','advantages')}
            snapshot['observations'] = {k:v.detach().cpu().clone() for k,v in storage.observations.items()}
            snapshot['distribution_params'] = tuple(x.detach().cpu().clone() for x in storage.distribution_params)
            update_number = counts['local_ppo_updates']+1
            torch.save(snapshot, run/'rollouts'/f'rollout_{update_number:04d}.pt')
            report = audited_ppo_update(runner, likelihood_audit_path=run/'rollouts'/f'likelihood_{update_number:04d}.json')
            runner.alg.actor.assert_frozen_state()
            counts['local_ppo_updates'] += 1
            counts['local_optimizer_steps'] += report['optimizer_steps']
            runner.current_learning_iteration = counts['local_ppo_updates']
            report.update(counts=dict(counts), actual_phase_counts=dict(phase_counts))
            line(updates, report)
            pointer = save(runner,runtime,prior,counts,source_run=run)
            print(json.dumps({'completed_update':counts['local_ppo_updates'],'checkpoint':pointer}),flush=True)
    return pointer


def diagnostic_request(core, prior_raw, entry, elapsed, captured_targets=None):
    """Labelled exogenous direction probe; never used by formal PPO/eval."""
    ack = core.frame.info['atomic_ack']
    baseline = ack['policy_headroom_evidence']['baseline_native_plus_controller_full12']
    caps = core.backend.execution_profile['residual']['phase_caps_full12'][core.frame.state_id]
    result = list(prior_raw)
    fraction = min(1., max(0., elapsed)/1.5)
    targets = captured_targets or {6:entry[6]-25.*fraction, 7:entry[7]+20.*fraction}
    for index, target in targets.items():
        ratio = max(-.985, min(.985, (target-baseline[index])/caps[index]))
        result[index] = math.atanh(ratio)
    return result, {'intervention':'RR_entry_FINAL_hip_minus25_knee_plus20_ramped1.5s',
                    'candidate_absolute_targets':targets, 'baseline_from_previous_committed_ack':baseline,
                    'not_same_tick_counterfactual':True, 'PPO_credit':0,
                    'contact_holds_actual_FINAL_no_further_angle_ramp':captured_targets is not None}


def evaluate(core, runner, runtime, prior, counts, run, diagnostic=False):
    import torch
    from .semantic_video import REVIEW_CAMERA, capture_task_interval_frame, capture_assist_tick_evidence
    from .semantic_legacy_evaluation import PhysicalEvaluationRecorder
    from .semantic_height_diagnostics import HeightDiagnostics
    from wlr50_clean.infrastructure.video_capture import ActiveViewportVideoRecorder
    from .semantic_training import state_hash
    source=run/'source'; source.mkdir()
    backend=core.backend
    backend.configure_video_camera(**REVIEW_CAMERA)
    core.reset(seed=4001)
    recorder=ActiveViewportVideoRecorder(source)
    physical=PhysicalEvaluationRecorder(source,task_spec_path=CONFIG/'stage_task_spec.yaml',
                                        quality_score_path=CONFIG/'quality_score.yaml')
    physical.start(core.frame)
    heights=HeightDiagnostics(source,backend); heights.start(core.frame)
    for _ in range(3): backend.render_video_frame()
    if not recorder.start(): raise RuntimeError('viewport recorder failed')
    before_hash=state_hash(runner.alg.save())
    last_captured=0; completed=0; error=None; last_info=None; entry=None; entry_time=None; captured_targets=None
    with (source/'video_policy_decisions.jsonl').open('x') as decisions, \
         (source/'capture_assist_ticks.jsonl').open('x') as assist:
        def observer(before, after, projection):
            nonlocal last_captured
            physical.observe(before,after,projection)
            heights.sample(after,terminal=False)
            line(assist,capture_assist_tick_evidence(after))
            if after.physics_tick%8==0:
                capture_task_interval_frame(recorder,backend,after.physics_tick)
                last_captured=after.physics_tick
        core.tick_observer=observer
        try:
            while not core.done:
                obs=tensor_observation(core.observation,runner.device)
                with torch.inference_mode(): raw,audit=request(runner,obs,stochastic=False)
                issued=raw[0].cpu().tolist()
                if diagnostic and core.task.snapshot()['active']:
                    if entry is None:
                        entry=list(core.frame.info['drive_target_full12']); entry_time=core.frame.sim_time_s
                    if captured_targets is None and core.task.snapshot()['metrics']['current_top_contact']:
                        final=core.frame.info['drive_target_full12']
                        captured_targets={6:final[6],7:final[7]}
                    issued,intervention=diagnostic_request(core,issued,entry,core.frame.sim_time_s-entry_time,captured_targets)
                    audit['independent_diagnostic']=intervention
                phase=core.frame.state_id; start=core.frame.physics_tick
                step=core.step(issued); completed+=1; last_info=step.info
                line(decisions,{'decision':completed,'request_phase':phase,'start_tick':start,
                    'end_tick':core.frame.physics_tick,'physics_ticks':core.frame.physics_tick-start,
                    'environment_step_returned':True,'policy_request':audit,'step_info':step.info})
                if completed%150==0:
                    print(json.dumps({'video_decisions':completed,'time_s':core.frame.sim_time_s,
                        'phase':core.frame.state_id,'local':core.task.snapshot()}),flush=True)
        except BaseException as exc:
            error=traceback.format_exc()
            raise
        finally:
            if core.frame.physics_tick and core.frame.physics_tick!=last_captured:
                capture_task_interval_frame(recorder,backend,core.frame.physics_tick)
            heights_receipt=heights.close(core.frame)
            summary=physical.summary(); physical.close()
            media=recorder.finalize()
            unchanged=before_hash==state_hash(runner.alg.save())
            result={'schema':'wlr50_clean.frozen_prior_rr_capture_video.v1','runtime_contract':runtime,
                'camera':REVIEW_CAMERA,
                'mode':'INDEPENDENT_DIRECTION_DIAGNOSTIC' if diagnostic else 'DETERMINISTIC_COMPOSITE_POLICY',
                'prior':prior,'counts':counts,'source_run':str(run),'continuous_natural_P01':True,
                'checkpoint_load_provenance':runner.checkpoint_load_provenance,
                'rear_owner_projection':False,'ignored_legacy_sampling_profile':True,
                'control_contributions':{'policy_version':settings()['version'], 'observation_dimension':447,
                    'local_branch_counters':dict(counts), 'frozen_prior':prior,
                    'FL_capture_assist_mode':'p05_hip_only_continuation_v1',
                    'rear_owner_projection':False,'rr_capture_assist_mode':None,
                    'nominal_geometry_advisory':None,'rr_capture_wheel_mode':'off',
                    'diagnostic_only_RR_joint_override':diagnostic},
                'checkpoint_model_unchanged':unchanged,'policy_switches':0,'front_FL_assist':True,
                'rear_task_assists':False,'diagnostic_intervention':diagnostic,'PPO_updates':0,
                'physical_summary':summary,'local_task':core.task.snapshot(),'terminal_info':last_info,
                'actual_ticks':core.frame.physics_tick,'time_s':core.frame.sim_time_s,
                'time_receipt':interval_receipt(core.frame.physics_tick),
                'recorder':media,'height_diagnostics':heights_receipt,'error':error,
                'full_task_success':bool(summary.get('task_success')),
                'local_RR_success':bool(core.task.snapshot()['local_success'])}
            result['sealed_files']={p.name:{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
                for p in source.iterdir() if p.is_file() and p.name!='source_manifest.json'}
            write(source/'source_manifest.json',result)
            if not unchanged: raise RuntimeError('evaluation modified packaged learned state')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('initialize','diagnostic','train','eval'))
    parser.add_argument('--expected-head',required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--decisions',type=int,default=2048)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    runtime=contract(args.expected_head)
    run=args.run_dir.resolve()
    if not run.is_relative_to(ROOT/'runs'/NAME): raise ValueError('new run must use isolated namespace')
    run.mkdir(parents=True,exist_ok=False)
    write(run/'run_manifest.started.json',{'runtime_contract':runtime,'mode':args.mode,
        'checkpoint':str(args.checkpoint),'started_utc':datetime.now(timezone.utc).isoformat()})
    app=None
    try:
        # Known Windows native DLL order. One process owns Torch and Isaac.
        import torch
        import tensordict
        from .rl_library_wrapper import seed_training_rngs
        if args.mode!='initialize':
            from isaaclab.app import AppLauncher
            app=AppLauncher(headless=args.mode=='train',enable_cameras=False).app
            app.update()
        seed_training_rngs(1001)
        runner=make_runner(args.device,1001)
        if args.checkpoint:
            prior,counts=load(runner,args.checkpoint,runtime)
        else:
            prior=initialize_prior(runner)
            counts=dict(local_policy_decisions=0,local_ppo_updates=0,local_optimizer_steps=0,
                prefix_decisions=0,capture_opportunities=0,local_successes=0,completed_local_episodes=0,
                auxiliary_updates=0)
        if args.mode=='initialize': result=save(runner,runtime,prior,counts,source_run=run)
        else:
            core=build_core(app)
            if args.mode=='train': result=train(core,runner,runtime,prior,counts,run,args.decisions)
            else:
                if not args.checkpoint: raise ValueError('evaluation must reload a saved packaged checkpoint')
                runner.alg.eval_mode()
                result=evaluate(core,runner,runtime,prior,counts,run,args.mode=='diagnostic')
        write(run/'run_manifest.json',{'lifecycle':'COMPLETE','mode':args.mode,'result':result})
        print(json.dumps({'run':str(run),'lifecycle':'COMPLETE'}),flush=True)
    except BaseException:
        write(run/'failure.json',{'traceback':traceback.format_exc()})
        raise
    finally:
        if app is not None: app.close(wait_for_replicator=False,skip_cleanup=True)


if __name__=='__main__': main()
