"""Explicit CP225280/f6d439 sibling branch; no old-lineage relabelling.

The module is stdlib-only on import. Tensor/publication work is lazy and must be
called only after the sole physical collector has sealed a complete update.
The original source manifest/receipts are embedded unchanged in the identity.
"""
from __future__ import annotations

import copy
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

SCHEMA = 'wlr50_clean.cp225280_front_preserved439.v1'
IDENTITY = 'front_preservation439_branch_identity'
COUNTS_KEY = 'front_preservation439_branch_counts'
BRANCH_NAME = 'cp225280_front_preserved_v1'
EXPERIMENT = 'rr_rl_timing_policy_learning_v1'
SOURCE_BRANCH = 'ancestor220544_recapture_v2'
SOURCE_STEM = 'checkpoint_rear_owner_CP225280_gf6d1d2df8d87'
SOURCE_HEAD = 'f6d1d2df8d87d5f3eaaefc2254adb5f81fc52e2b'
SOURCE_SHA = 'fbe28e3718a5796afc2271e7b10aa04017369593b5c62e0bde1973a20f68032f'
SOURCE_MANIFEST_SHA = 'f501b7a735c0aaa42be00114e09f80d46f7f009d7717aa416fc30ad3cc837d6d'
COUNTERS = ('global_policy_decisions', 'ppo_updates', 'optimizer_steps')
ORIGIN = dict(zip(COUNTERS, (225280, 1725, 34500)))
POLICY = 'rear_owner_recovery_history_v1'
LAYOUT = 'role422_rear_owner_recovery_v1'
COLLECTION_KEY = 'semantic_collection_profile'
COLLECTION_512 = 'n1_rear_owner439_collection512_v1'
REPLAY_SCHEMA = 'wlr50_clean.front_replay_regularizer.v1'
REPLAY_VERSION = 'cp225280_front_gaussian_kl_v1'
REPLAY_COUNTS_KEY = 'front_replay_counts'
REWARD_SHA = '7d06694bc5ec1abf24d8a55795fd647e59822d1cdbcdb271b5c4985013ef43c3'
CODE = 'src/wlr50_clean/ppo/'
# Existing f6d->59e files plus this explicit identity and the agreed replay loss.
# All configuration, N, physical execution, mapper, HISTORY and FL assist bytes
# are outside this set and must stay exactly f6d. Never accept caller widening.
ALLOWED = frozenset(CODE + name for name in (
    'semantic_reward.py', 'semantic_rr_retention_migration.py',
    'semantic_video_cli.py', 'semantic_policy_distribution.py',
    'semantic_training.py', 'semantic_rear_policy_timing_migration.py',
    'semantic_return_profile.py', 'semantic_front_retention439.py',
    'semantic_cli.py', 'semantic_migration.py',
    'semantic_front_preservation.py', 'semantic_front_replay.py'))
FORBIDDEN_LATER_LINEAGE = ('rr_retention_reward_migration',
    'front_retention439_runtime_identity', 'front_retention439_auxiliary',
    'collection_horizon439')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def _root(project_root=None):
    return Path(project_root or Path(__file__).resolve().parents[3]).resolve()


def _branch(root, name=BRANCH_NAME):
    return root / 'outputs' / ('ppo_' + EXPERIMENT) / 'branches' / name


def _source_binding(root):
    checkpoint = _branch(root, SOURCE_BRANCH) / 'checkpoints/history' / (SOURCE_STEM + '.pt')
    return dict(checkpoint=str(checkpoint), checkpoint_sha256=SOURCE_SHA,
                manifest=str(checkpoint.with_name(SOURCE_STEM + '_manifest.json')),
                manifest_sha256=SOURCE_MANIFEST_SHA)


def _source_checkpoint(root):
    binding = _source_binding(root)
    require(file_sha(binding['checkpoint']) == SOURCE_SHA and
            file_sha(binding['manifest']) == SOURCE_MANIFEST_SHA,
            'exact immutable f6d439 CP225280 checkpoint/sidecar required')
    source = json.loads(Path(binding['manifest']).read_text(encoding='utf-8'))
    require(source.get('checkpoint_sha256') == SOURCE_SHA and
            Path(source['checkpoint_path']).resolve() == Path(binding['checkpoint']) and
            source['runtime_contract']['source_git_commit'] == SOURCE_HEAD and
            {key: source[key] for key in COUNTERS} == ORIGIN and
            source['policy_contract']['version'] == POLICY and
            source['policy_contract']['observation_dimension'] == 439 and
            source['policy_contract']['observation_layout'] == LAYOUT and
            source.get('save_load_round_trip') is True and
            not any(key in source for key in FORBIDDEN_LATER_LINEAGE + (IDENTITY,)),
            'source is not the original zero-update f6d439 CP225280 publication')
    return source, binding


def _validate_source_lineage(source):
    # Validate the source under ITS original contract and route, never the new
    # target contract/sibling path. This is not a bypass of the old validator.
    from .semantic_rear_owner_migration import validate_rear_owner_lineage
    route = source['checkpoint_output_routing']
    validate_rear_owner_lineage(source, source['runtime_contract'],
        Path(route['output_root']), checkpoint_output_routing=route)


def _validate_replay_spec(spec):
    require(isinstance(spec, dict) and spec.get('schema') == REPLAY_SCHEMA,
            'explicit versioned front replay spec required before publication')
    require('version' not in spec or spec['version'] == REPLAY_VERSION,
            'front replay version differs')
    path = Path(spec.get('dataset_path', ''))
    sha = spec.get('dataset_sha256')
    require(path.is_absolute() and path.is_file() and isinstance(sha, str) and
            re.fullmatch('[0-9a-f]{64}', sha) and file_sha(path) == sha,
            'immutable real front dataset path/hash differs')
    require(type(spec.get('coefficient')) in (int, float) and spec['coefficient'] == 1.0
            and type(spec.get('minibatch_size')) is int and spec['minibatch_size'] == 32,
            'front replay coefficient1/batch32 differs')
    return copy.deepcopy(spec)


def _runtime_delta(source_contract, target_contract, root, *, verify_bytes):
    old, new = source_contract, target_contract
    variable = {'files', 'source_git_commit', 'runtime_content_sha256'}
    require({k: v for k, v in old.items() if k not in variable} ==
            {k: v for k, v in new.items() if k not in variable},
            'front-preservation cannot change configuration, physics, task, or observation contract')
    require(old['source_git_commit'] == SOURCE_HEAD and
            new['source_git_commit'] != SOURCE_HEAD and
            isinstance(new['source_git_commit'], str) and
            re.fullmatch('[0-9a-f]{40}', new['source_git_commit']),
            'new frozen target commit required')
    changed = {p for p in old['files'].keys() | new['files'].keys()
               if old['files'].get(p) != new['files'].get(p)}
    require(changed <= ALLOWED and CODE + 'semantic_front_preservation.py' in changed
            and CODE + 'semantic_front_replay.py' in changed and
            set(old['files']) <= set(new['files']),
            'unreviewed runtime path or missing front-preservation/replay implementation')
    require(new['files'].get(CODE + 'semantic_reward.py') == REWARD_SHA,
            'only the declared existing 72e RR-retention reward is allowed')
    require(all(digest(c['files']) == c['runtime_content_sha256'] for c in (old, new)),
            'runtime file inventory digest differs')
    if verify_bytes:
        require(all(file_sha(root / p) == sha for p, sha in new['files'].items()),
                'new runtime bytes differ from its frozen contract')
    return {p: dict(before=old['files'].get(p), after=new['files'][p]) for p in sorted(changed)}


def _target_runner(source):
    config = copy.deepcopy(source['runner_config'])
    require(config.get('num_steps_per_env') == 128 and COLLECTION_KEY not in config
            and config['algorithm']['gamma'] == .9985 and config['algorithm']['lam'] == .99
            and config['algorithm']['num_learning_epochs'] == 5
            and config['algorithm']['num_mini_batches'] == 4,
            'unchanged original128/gamma/lambda/5epochs/4minibatches required')
    config['num_steps_per_env'] = 512
    config[COLLECTION_KEY] = COLLECTION_512
    return config


def _route(root, binding):
    return dict(schema='wlr50_clean.checkpoint_output_routing.v1', branch=BRANCH_NAME,
        output_root=str(_branch(root)), main_latest_pointer_promotion=False,
        source_selection=dict(source_role='explicit_CP225280_front_preservation_continuation',
            checkpoint_sha256=binding['checkpoint_sha256'],
            manifest_sha256=binding['manifest_sha256'], counters=dict(ORIGIN), output_branch=BRANCH_NAME))


def _make_plan(source, binding, contract, reason, replay_spec, root, *, verify_bytes):
    require(isinstance(reason, str) and reason.strip(), 'explicit branch selection reason required')
    delta = _runtime_delta(source['runtime_contract'], contract, root, verify_bytes=verify_bytes)
    return dict(schema=SCHEMA, reason=reason.strip(), source_checkpoint=copy.deepcopy(binding),
        source_manifest=copy.deepcopy(source), source_manifest_content_sha256=digest(source),
        source_contract_sha256=digest(source['runtime_contract']), target_contract_sha256=digest(contract),
        target_git_commit=contract['source_git_commit'],
        target_runtime_content_sha256=contract['runtime_content_sha256'],
        changed_file_hashes=delta, counter_origin=dict(ORIGIN),
        target_runner_config=_target_runner(source), target_policy_contract=copy.deepcopy(source['policy_contract']),
        checkpoint_output_routing=_route(root, binding), front_replay_spec=_validate_replay_spec(replay_spec),
        parameter_mapping='identity_actor_critic_full_Adam_options_steps_LR_Identity_normalizer_RNG',
        control_N_mapper_HISTORY_FLassist_unchanged=True, rear_task_assists_enabled=False,
        reward_change='original_f6d_to_existing_72e_RR_retention_reward_only; no_new_front_reward',
        return_estimator_changed=False, collection_steps_before=128, collection_steps_after=512,
        optimizer_steps_per_update=20, physical_state_resumed=False,
        old_rollout_inherited=False, fresh_rollout_required=True,
        added_policy_decisions=0, added_ppo_updates=0, added_optimizer_steps=0, added_auxiliary_updates=0,
        current_branch_latest_pointer_read_or_overwritten=False)


def build_front_preservation_plan(checkpoint, contract, *, expected_source_sha256,
        expected_manifest_sha256, reason, front_replay_spec, project_root=None):
    root = _root(project_root)
    source, binding = _source_checkpoint(root)
    require(Path(checkpoint).resolve(strict=True) == Path(binding['checkpoint']) and
            expected_source_sha256 == SOURCE_SHA and expected_manifest_sha256 == SOURCE_MANIFEST_SHA,
            'caller must explicitly select immutable f6d439 CP225280 source')
    _validate_source_lineage(source)
    return _make_plan(source, binding, contract, reason, front_replay_spec, root, verify_bytes=True)


def _counter_delta(metadata):
    require(all(type(metadata.get(k)) is int and metadata[k] >= ORIGIN[k] for k in COUNTERS),
            'front-preservation counters rolled back or missing')
    updates = metadata['ppo_updates'] - ORIGIN['ppo_updates']
    require(metadata['global_policy_decisions'] - ORIGIN['global_policy_decisions'] == 512 * updates and
            metadata['optimizer_steps'] - ORIGIN['optimizer_steps'] == 20 * updates,
            'new branch requires exactly512 on-policy decisions and20 official Adam steps per update')
    return {k: metadata[k] - ORIGIN[k] for k in COUNTERS}


def front_preservation_branch_counts(metadata):
    return {**metadata, COUNTS_KEY: _counter_delta(metadata)}


def expected_front_replay_counts(ppo_updates):
    """Expected accounting, not evidence that a replay minibatch was executed."""
    require(type(ppo_updates) is int and ppo_updates >= 0, 'finite completed replay update count required')
    return dict(ppo_updates_with_replay=ppo_updates, optimizer_minibatches=20 * ppo_updates,
        replay_row_exposures=640 * ppo_updates, on_policy_samples_added=0,
        separate_auxiliary_optimizer_steps=0)


def front_preservation_collection_options(metadata, *, experiment_id):
    """Constructor choice only; the official loader then validates full identity."""
    receipt = metadata.get(IDENTITY, {})
    config = metadata.get('runner_config', {})
    policy = metadata.get('policy_contract', {})
    require(experiment_id == EXPERIMENT and receipt.get('schema') == SCHEMA
            and config == receipt.get('target_runner_config')
            and config.get('num_steps_per_env') == 512
            and config.get(COLLECTION_KEY) == COLLECTION_512
            and policy == receipt.get('target_policy_contract')
            and policy.get('version') == POLICY and policy.get('observation_dimension') == 439
            and policy.get('observation_layout') == LAYOUT,
            'front-preservation constructor requires its explicit439/512 identity')
    return {'collection_profile': COLLECTION_512}


def _validate_actual_replay_report(metadata, replay_spec):
    report = metadata.get('last_update', {}).get('front_replay_regularization', {})
    batches = report.get('minibatches')
    require(report.get('schema') == REPLAY_SCHEMA and report.get('spec') == replay_spec
            and isinstance(batches, list) and len(batches) == 20
            and report.get('actual_replay_row_exposures') == 640
            and report.get('on_policy_samples_added') == 0
            and report.get('separate_auxiliary_optimizer_steps') == 0
            and report.get('realtime_teacher_deployed') is False,
            'complete update must contain its actual20 replay minibatches/640 exposures and zero extra credit')
    # Hash already checked above; read only the immutable bounded data table.
    data = json.loads(Path(replay_spec['dataset_path']).read_text(encoding='utf-8'))
    rows = data.get('training_rows')
    require(isinstance(rows, list) and 0 < len(rows) <= 256,
            'front replay requires its bounded immutable real training partition')
    phases = frozenset('P%02d' % i for i in range(1, 7))
    total = Counter()
    first_index = (metadata['ppo_updates'] - 1) * 20
    for ordinal, batch in enumerate(batches):
        index = first_index + ordinal
        expected_indices = [(index * 32 + offset) % len(rows) for offset in range(32)]
        expected_rows = [rows[i] for i in expected_indices]
        expected_phases = dict(Counter(row.get('phase') for row in expected_rows))
        require(all(phase in phases for phase in expected_phases)
                and type(batch.get('global_minibatch_index')) is int
                and batch['global_minibatch_index'] == index
                and batch.get('dataset_indices') == expected_indices
                and batch.get('source_decision_indices') == [row.get('decision_index') for row in expected_rows]
                and batch.get('phase_counts') == expected_phases
                and all(type(value) is int and value > 0 for value in batch['phase_counts'].values())
                and sum(batch['phase_counts'].values()) == 32
                and batch.get('gradient_consumed_once') is True
                and batch.get('coefficient') == replay_spec['coefficient']
                and batch.get('on_policy_samples_added') == 0 and batch.get('separate_optimizer_steps') == 0
                and batch.get('extra_random_draws') == 0 and batch.get('extra_actor_forwards') == 1,
                'actual replay minibatch source/phase/count/one-use gradient evidence differs')
        for key in ('kl_reference_to_current', 'unclipped_replay_actor_gradient_norm'):
            require(type(batch.get(key)) in (int, float) and math.isfinite(batch[key]),
                    'nonfinite replay metric is not a valid completed update')
        total.update(batch['phase_counts'])
    require(report.get('actual_replay_phase_exposures') == dict(total) and sum(total.values()) == 640,
            'actual per-phase replay exposure aggregate differs from its minibatch rows')


def build_front_preservation_output_routing(metadata, contract, destination):
    destination = Path(destination).resolve()
    root = destination.parents[3]
    require(destination == _branch(root), 'front-preservation allows only its explicit sibling branch')
    source, binding = _source_checkpoint(root)
    receipt = metadata.get(IDENTITY)
    if receipt is None:
        require(metadata == source, 'only exact original CP225280 may initialize this sibling route')
    else:
        require(receipt.get('schema') == SCHEMA and receipt.get('source_checkpoint') == binding
                and receipt.get('target_contract_sha256') == digest(contract),
                'descendant route requires its exact front-preservation identity')
    route = _route(root, binding)
    if receipt is not None:
        require(receipt['checkpoint_output_routing'] == route and
                metadata.get('checkpoint_output_routing') == route,
                'front-preservation cannot move, promote, or borrow another branch route')
    return route


def validate_front_preservation_lineage(metadata, contract, output_root, *, checkpoint_output_routing=None):
    destination = Path(output_root).resolve()
    root = destination.parents[3]
    require(destination == _branch(root), 'front-preservation output namespace differs')
    source, binding = _source_checkpoint(root)
    _validate_source_lineage(source)
    receipt = metadata.get(IDENTITY, {})
    expected = _make_plan(source, binding, contract, receipt.get('reason', ''),
                          receipt.get('front_replay_spec'), root, verify_bytes=False)
    require(receipt == expected and metadata.get('runtime_contract') == contract and
            metadata.get('runner_config') == expected['target_runner_config'] and
            metadata.get('policy_contract') == expected['target_policy_contract'] and
            metadata.get('checkpoint_output_routing') == expected['checkpoint_output_routing'] and
            checkpoint_output_routing == expected['checkpoint_output_routing'],
            'front-preservation identity, source snapshot, target config or route differs')
    require(not any(k in metadata for k in FORBIDDEN_LATER_LINEAGE),
            'CP225280 branch must not borrow later72e/65a/AUX32/512 receipts')
    for key, value in source.items():
        if key != 'resume_migration' and key.endswith(('_migration', '_branch')):
            require(metadata.get(key) == value, 'historical source receipt changed: ' + key)
    require(metadata.get('seed') == source['seed'] and metadata.get('normalization') == source['normalization']
            and metadata.get('normalizer_state_sha256') == source['normalizer_state_sha256'],
            'source seed or Identity normalizer changed')
    counts = _counter_delta(metadata)
    require(metadata.get(COUNTS_KEY) == counts, 'new branch reported counter delta differs')
    require(metadata.get(REPLAY_COUNTS_KEY) == expected_front_replay_counts(counts['ppo_updates']),
            'front replay must report actual20 minibatches/640 replay exposures per PPO update; zero PPO/AUX credit')
    if counts['ppo_updates'] == 0:
        for key in ('actor_parameter_sha256', 'critic_parameter_sha256', 'optimizer_state_sha256',
                    'optimizer_learning_rate', 'training_rng_state', 'stage_requested_decisions'):
            require(metadata.get(key) == source[key], 'zero-update publication changed learned state: ' + key)
    else:
        last = metadata.get('last_update', {})
        require(last.get('global_policy_decisions') == metadata['global_policy_decisions'] and
                last.get('ppo_update') == metadata['ppo_updates'] and last.get('optimizer_steps') == 20 and
                last.get('actor_parameter_sha256_after') == metadata['actor_parameter_sha256'] and
                last.get('optimizer_learning_rate') == metadata['optimizer_learning_rate'],
                'descendant must end at its actual complete official PPO update')
        require(all(type(metadata.get('stage_requested_decisions', {}).get(key)) is int and
                    metadata['stage_requested_decisions'][key] >= value
                    for key, value in source['stage_requested_decisions'].items()),
                'historical sampling counters rolled back')
        _validate_actual_replay_report(metadata, expected['front_replay_spec'])
    return receipt


def _front_gaussian_identity_evidence(runner, replay_spec, *, seed, tolerance=2e-6):
    """Publication-only real-input kernel check; never a physical success score."""
    import torch
    from .semantic_front_replay import validate_replay_spec, current_front_gaussian
    from .semantic_training import state_hash
    from .rl_library_wrapper import capture_training_rng_state
    data = validate_replay_spec(replay_spec)
    actor = runner.alg.actor
    cache = actor.distribution._distribution
    def cache_hash():
        current = actor.distribution._distribution
        return state_hash(None if current is None else {'loc': current.loc, 'scale': current.scale})
    before = dict(actor_state=state_hash(actor.state_dict()),
                  critic_state=state_hash(runner.alg.critic.state_dict()),
                  optimizer_state=state_hash(runner.alg.optimizer.state_dict()),
                  RNG=capture_training_rng_state(seed=seed), distribution_cache=cache_hash())
    partitions = {}
    with torch.no_grad():
        for name in ('training_rows', 'heldout_rows'):
            rows = data[name]
            observation, reference_mean, reference_sigma = (
                torch.tensor([row[field] for row in rows], dtype=torch.float32, device=runner.device)
                for field in ('observation', 'reference_mean', 'reference_sigma'))
            mean, log_sigma = current_front_gaussian(actor, observation)
            sigma = log_sigma.exp()
            require(bool(torch.isfinite(mean).all()) and bool(torch.isfinite(sigma).all()),
                    'nonfinite initial front Gaussian')
            mean_error = float((mean - reference_mean).abs().max())
            sigma_error = float((sigma - reference_sigma).abs().max())
            require(mean_error <= tolerance and sigma_error <= tolerance,
                    'initial real-front Gaussian differs from CP225280 beyond batch-kernel tolerance')
            partitions[name] = dict(rows=len(rows), phase_counts=dict(Counter(row['phase'] for row in rows)),
                max_abs_conditional_mean_error=mean_error, max_abs_conditional_sigma_error=sigma_error)
    after = dict(actor_state=state_hash(actor.state_dict()),
                 critic_state=state_hash(runner.alg.critic.state_dict()),
                 optimizer_state=state_hash(runner.alg.optimizer.state_dict()),
                 RNG=capture_training_rng_state(seed=seed), distribution_cache=cache_hash())
    require(before == after and actor.distribution._distribution is cache,
            'front identity evaluation mutated learned state, RNG or distribution cache')
    return dict(schema='wlr50_clean.front_preservation_initial_real_input_Gaussian.v1',
        dataset_sha256=replay_spec['dataset_sha256'], absolute_tolerance=tolerance,
        partitions=partitions, learned_state_RNG_cache_unchanged=True,
        optimizer_steps=0, PPO_samples=0, AUX_updates=0, physical_retention_proven=False)


def publish_front_preservation_checkpoint(checkpoint, contract, plan_path, output_checkpoint):
    """Zero-update full-state copy + independent official reload; no pointer write."""
    from .semantic_training import (construct_semantic_runner, load_semantic_checkpoint,
        save_semantic_checkpoint, parameter_hash, state_hash, _normalizers)
    from .semantic_rr_capture_migration import _shape_env
    from .rl_library_wrapper import (restore_training_rng_state, capture_training_rng_state,
                                    optimizer_learning_rate)
    root = _root()
    source, binding = _source_checkpoint(root)
    plan = json.loads(Path(plan_path).read_text(encoding='utf-8'))
    expected = build_front_preservation_plan(checkpoint, contract,
        expected_source_sha256=SOURCE_SHA, expected_manifest_sha256=SOURCE_MANIFEST_SHA,
        reason=plan.get('reason', ''), front_replay_spec=plan.get('front_replay_spec'))
    require(plan == expected, 'publication plan changed after review')
    destination = Path(output_checkpoint).resolve()
    branch = _branch(root)
    require(destination.parent == branch / 'checkpoints/history' and not destination.exists() and
            not destination.with_name(destination.stem + '_manifest.json').exists(),
            'unique new sibling checkpoint required; never overwrite source/latest')
    def make(marker):
        device = source['runner_config']['device']
        return construct_semantic_runner(_shape_env(439, device), seed=source['seed'], device=device,
            policy_version=POLICY, observation_layout=LAYOUT, initialize_actor=False,
            collection_profile=marker)[0]
    historical = make(None)
    require(historical._semantic_runner_config == source['runner_config'], 'historical128 constructor differs')
    infos = load_semantic_checkpoint(historical, checkpoint,
        contract=source['runtime_contract'], seed=source['seed'])
    runner = make(COLLECTION_512)
    require(runner._semantic_runner_config == plan['target_runner_config'], 'target512 constructor differs')
    runner.alg.actor.load_state_dict(historical.alg.actor.state_dict(), strict=True)
    runner.alg.critic.load_state_dict(historical.alg.critic.state_dict(), strict=True)
    runner.alg.optimizer.load_state_dict(copy.deepcopy(historical.alg.optimizer.state_dict()))
    runner.alg.learning_rate = optimizer_learning_rate(historical)
    runner.current_learning_iteration = historical.current_learning_iteration
    require(runner.alg.storage.step == 0 and runner.alg.transition.actions is None and
            tuple(runner.alg.storage.actions.shape) == (512, 1, 12) and
            all(tuple(runner.alg.storage.observations[k].shape) == (512, 1, 439) for k in ('policy', 'critic')),
            'new branch requires fresh empty512x1 storage; no old partial rollout')
    restore_training_rng_state(source['training_rng_state'], expected_seed=source['seed'])
    def verify_identity(candidate):
        require(parameter_hash(candidate.alg.actor) == source['actor_parameter_sha256'] and
                parameter_hash(candidate.alg.critic) == source['critic_parameter_sha256'] and
                state_hash(candidate.alg.optimizer.state_dict()) == source['optimizer_state_sha256'] and
                state_hash(_normalizers(candidate)) == source['normalizer_state_sha256'] and
                optimizer_learning_rate(candidate) == source['optimizer_learning_rate'] and
                capture_training_rng_state(seed=source['seed']) == source['training_rng_state'],
                'publication changed actor/critic/fullAdam/LR/Identity/RNG')
    verify_identity(runner)
    kernel_before_save = _front_gaussian_identity_evidence(runner, plan['front_replay_spec'], seed=source['seed'])
    verify_identity(runner)
    infos.update(runtime_contract=copy.deepcopy(contract), runner_config=copy.deepcopy(plan['target_runner_config']),
        checkpoint_output_routing=copy.deepcopy(plan['checkpoint_output_routing']),
        old_rollout_inherited=False, physical_env_state_saved=False,
        resume_physics='fresh_natural_P01_or_legal_continuous_prefix;not_bitwise_physics_resume')
    infos[IDENTITY] = copy.deepcopy(plan)
    infos[REPLAY_COUNTS_KEY] = expected_front_replay_counts(0)
    infos = front_preservation_branch_counts(infos)
    saved, sidecar = save_semantic_checkpoint(runner, destination, infos)
    fresh = make(COLLECTION_512)
    loaded = load_semantic_checkpoint(fresh, saved, contract=contract, seed=source['seed'])
    verify_identity(fresh)
    kernel_after_reload = _front_gaussian_identity_evidence(fresh, plan['front_replay_spec'], seed=source['seed'])
    verify_identity(fresh)
    validate_front_preservation_lineage(loaded, contract, branch,
        checkpoint_output_routing=plan['checkpoint_output_routing'])
    return dict(checkpoint=str(saved), checkpoint_sha256=file_sha(saved), manifest=str(sidecar),
        manifest_sha256=file_sha(sidecar), save_load_round_trip=True, latest_pointer_published=False,
        branch=BRANCH_NAME, **{key: loaded[key] for key in COUNTERS},
        added_policy_decisions=0, added_ppo_updates=0, added_optimizer_steps=0, added_auxiliary_updates=0,
        replay_spec=copy.deepcopy(plan['front_replay_spec']),
        real_front_Gaussian_before_save=kernel_before_save, real_front_Gaussian_after_reload=kernel_after_reload,
        rear_assists=False, FL_assist='retained_declared')
