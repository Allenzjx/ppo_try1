"""Latest learned422 -> public-owner439; exact old state, zero new columns."""
from __future__ import annotations
import copy
import json
from pathlib import Path
from .semantic_rear_policy_timing_migration import EXPERIMENT, COUNTERS, SOURCE_COUNTS, preserved_keys
from .semantic_rear_recapture_migration import BRANCH_NAME, CODE, CONFIG
from .semantic_cooperative_prep_migration import _previous_contract as _previous_cooperative_contract, validate_cooperative_prep_lineage
from .semantic_rear_cooperative_prep_profile import COOPERATIVE_PREP_POLICY, COOPERATIVE_PREP_OBSERVATION_LAYOUT
from .semantic_rear_owner_profile import (REAR_OWNER_POLICY, REAR_OWNER_OBSERVATION_LAYOUT,
    REAR_OWNER_GROUP, REAR_OWNER_MODE)
from .semantic_cooperative_preparation import MODE, TASK_PROXY_MODE, REWARD_CONFIG, TASK_PROXY_REWARD_CONFIG

SCHEMA = 'wlr50_clean.rear_owner_recovery_append439.v1'
FACTOR_KEY = 'rear_owner_append439_factor'
MIGRATION = 'rear_owner_recovery_migration'
SOURCE_HEAD = '49eb23163a6e20bc56301dbafb59b137ecebce66'
SOURCE_SHA = '21897a4b2d2a85f01092c187c1b1ac824d8e61b01edaf732a6385ca951ab1dcf'
SOURCE_MANIFEST_SHA = '6e05b4e279c1d8ac99447f91d69b7db3f6dd6afaab6a326e610283f783b1f185'
REVISION_ORIGIN = dict(zip(COUNTERS, (225280,1725,34500)))
ALLOWED = frozenset(CODE+n for n in (
    'semantic_rear_owner_profile.py','semantic_rear_owner_actor.py','semantic_rear_owner_migration.py',
    'semantic_rear_owner_recovery.py','semantic_rear_policy_timing.py','semantic_supervisor.py',
    'semantic_backend.py','semantic_residual_adapter.py','actuator_target_effect.py',
    'semantic_cooperative_preparation.py','semantic_reward.py','semantic_observation.py',
    'semantic_policy_distribution.py','semantic_checkpoint_prefix_policy.py','semantic_checkpoint_prefix.py','semantic_training.py',
    'semantic_cli.py','semantic_migration.py','semantic_rear_policy_timing_migration.py')) | frozenset(
    CONFIG+n for n in ('execution_profile.yaml','stage_task_spec.yaml','reward_config.yaml','observation_schema.json','curriculum_plan.json'))
CURRICULUM_WEIGHTS = {'P01': .25, 'P07': .25, 'P10': .5}


def _previous_contract(current, receipt):
    previous = copy.deepcopy(current)
    previous['training_budgets'] = copy.deepcopy(receipt['source_training_budgets'])
    return _previous_cooperative_contract(previous, receipt)


def _policy(target=False):
    from .semantic_policy_distribution import policy_contract
    return policy_contract(REAR_OWNER_POLICY if target else COOPERATIVE_PREP_POLICY,
        observation_layout=REAR_OWNER_OBSERVATION_LAYOUT if target else COOPERATIVE_PREP_OBSERVATION_LAYOUT)


def _preserved(metadata):
    return tuple(sorted(set(preserved_keys(metadata)) | {'checkpoint_output_routing'}))


def _configs(root, old, new):
    import yaml
    from .semantic_migration import file_sha, _version_bytes
    if set(old['selected_configuration']) != set(new['selected_configuration']):
        raise ValueError('owner migration cannot change configuration inventory')
    changed = {}
    for name, entry in new['selected_configuration'].items():
        prior = old['selected_configuration'][name]
        if (entry['path'] != prior['path'] or file_sha(root/entry['path']) != entry['sha256']
                or old['files'].get(prior['path']) != prior['sha256']
                or new['files'].get(entry['path']) != entry['sha256']):
            raise ValueError('owner configuration binding differs')
        before, after = _version_bytes(root, old, prior['path']), (root/entry['path']).read_bytes()
        if prior != entry: changed[name] = copy.deepcopy(prior)
        if name not in ('execution_profile.yaml','stage_task_spec.yaml','reward_config.yaml','observation_schema.json','curriculum_plan.json'):
            if before != after: raise ValueError('unrelated configuration changed: '+name)
            continue
        a, b = yaml.safe_load(before), yaml.safe_load(after)
        if name in ('execution_profile.yaml','stage_task_spec.yaml'):
            if (a.pop('cooperative_preparation_mode', None) != MODE
                    or b.pop('cooperative_preparation_mode', None) != TASK_PROXY_MODE
                    or a.get('rear_owner_recovery_mode') is not None
                    or b.pop('rear_owner_recovery_mode', None) != REAR_OWNER_MODE):
                raise ValueError('explicit owner and preparation revision required')
            if name == 'execution_profile.yaml':
                if (a.pop('rear_policy_timing_mode') != 'rr_live_swing_evidence_v3'
                        or b.pop('rear_policy_timing_mode') != 'rr_rl_edge_recovery_v4'
                        or b.pop('revision') != 'rear_owner_edge_recovery_v5'):
                    raise ValueError('reviewed edge recovery execution revision differs')
                a.pop('revision')
                if (a['training_budgets'].pop('phase_suffix') != 100000
                        or b['training_budgets'].pop('phase_suffix') != 131072):
                    raise ValueError('only reviewed suffix quantity ceiling may change')
                for key, phase in (('full_P01','P01'),('predecessor_P07_suffix','P07'),('second_rear_P10_suffix','P10')):
                    if b['sampling_target'].pop(key) != CURRICULUM_WEIGHTS[phase]:
                        raise ValueError('rear-heavy curriculum sampling target differs')
                    a['sampling_target'].pop(key)
            else:
                if (a['nominal'].pop('rear_policy_timing') != 'rr_live_swing_evidence_v3'
                        or b['nominal'].pop('rear_policy_timing') != 'rr_rl_edge_recovery_v4'
                        or b.pop('cooperative_fl_knee_capacity_deg', None) != 36.):
                    raise ValueError('task edge/proxy revision differs')
        elif name == 'reward_config.yaml':
            if (a.pop('cooperative_preparation') != REWARD_CONFIG
                    or b.pop('cooperative_preparation') != TASK_PROXY_REWARD_CONFIG):
                raise ValueError('preparation must reuse identical reward coefficients/shares')
        elif name == 'observation_schema.json':
            if (a.get('rear_owner_recovery_features_version') is not None
                    or b.pop('rear_owner_recovery_features_version', None) != REAR_OWNER_OBSERVATION_LAYOUT
                    or b['feature_groups'].pop() != dict(name=REAR_OWNER_GROUP,size=17,scale=1)):
                raise ValueError('exact public17 append required')
        else:
            if [row['from_phase'] for row in b['entries']] != ['P01','P07','P10']:
                raise ValueError('declared curriculum entry inventory changed')
            for prior_row, row in zip(a['entries'],b['entries'],strict=True):
                if row.pop('weight') != CURRICULUM_WEIGHTS[row['from_phase']]:
                    raise ValueError('rear-heavy curriculum weights differ')
                prior_row.pop('weight')
        if a != b: raise ValueError('unrelated physical/learning/source configuration changed: '+name)
    return changed


def build_rear_owner_migration(checkpoint, current_contract, *, reason, project_root=None):
    from .semantic_migration import checkpoint_metadata, _contract, file_sha, digest, source_num_envs
    from .semantic_training import semantic_runner_config
    root = Path(project_root or Path(__file__).resolve().parents[3])
    checkpoint = Path(checkpoint).resolve(strict=True)
    branch = root/'outputs'/('ppo_'+EXPERIMENT)/'branches'/BRANCH_NAME
    if (checkpoint.parent != (branch/'checkpoints/history').resolve()
            or file_sha(checkpoint) != SOURCE_SHA
            or file_sha(checkpoint.with_name(checkpoint.stem+'_manifest.json')) != SOURCE_MANIFEST_SHA):
        raise ValueError('requires exact learned225280 source; no ancestor rollback')
    m = checkpoint_metadata(checkpoint)
    old, new = _contract(m['runtime_contract']), _contract(current_contract)
    validate_cooperative_prep_lineage(m, old, branch, checkpoint_output_routing=m.get('checkpoint_output_routing'))
    old_runner = semantic_runner_config(seed=m['seed'], device=m['runner_config']['device'], semantic_version='v3',
        policy_version=COOPERATIVE_PREP_POLICY, observation_layout=COOPERATIVE_PREP_OBSERVATION_LAYOUT)
    new_runner = semantic_runner_config(seed=m['seed'], device=m['runner_config']['device'], semantic_version='v3',
        policy_version=REAR_OWNER_POLICY, observation_layout=REAR_OWNER_OBSERVATION_LAYOUT)
    if (old['source_git_commit'] != SOURCE_HEAD or new['source_git_commit'] == SOURCE_HEAD
            or old['experiment_id'] != EXPERIMENT or new['experiment_id'] != EXPERIMENT
            or not isinstance(reason, str) or not reason.strip() or MIGRATION in m
            or {k:m[k] for k in COUNTERS} != REVISION_ORIGIN or source_num_envs(m) != 1
            or m['policy_contract'] != _policy() or m['runner_config'] != old_runner
            or m.get('save_load_round_trip') is not True):
        raise ValueError('source learned422 policy/counts/full state not intact')
    changed = sorted(p for p in set(old['files'])|set(new['files']) if old['files'].get(p) != new['files'].get(p))
    if not changed or not set(changed) <= ALLOWED or any(p not in new['files'] for p in changed):
        raise ValueError('unreviewed owner runtime change')
    for path, sha in new['files'].items():
        if file_sha(root/path) != sha: raise ValueError('target runtime bytes changed: '+path)
    source_config = _configs(root, old, new)
    budget = copy.deepcopy(old['training_budgets'])
    if budget.get('phase_suffix') != 100000:
        raise ValueError('unexpected historical suffix quantity ceiling')
    budget['phase_suffix'] = 131072
    if new['training_budgets'] != budget:
        raise ValueError('only explicit suffix quantity ceiling may change')
    factor = dict(schema=SCHEMA, source_policy_contract=_policy(), target_policy_contract=_policy(True),
        revision_counter_origin=REVISION_ORIGIN, original_branch_origin=SOURCE_COUNTS,
        source_runner_config=old_runner, target_runner_config=new_runner,
        source_effective_learning_rate=m['optimizer_learning_rate'],
        preserved_metadata_sha256={k:digest(m[k]) for k in _preserved(m)},
        actor_critic_mapping='old422_columns_exact_new17_zero',
        Adam_mapping='all_old_moments_steps_options_exact_new17_first_weight_moment_columns_zero',
        rear_owner_recovery_mode=REAR_OWNER_MODE, cooperative_preparation_revision=TASK_PROXY_MODE,
        same_mdp_claimed=False, physical_dynamics_changed=False,
        current_control_semantics='cancel_inapplicable_committed_dependent_owner_via_public_anchors; preserve_policy_recovery_and_wheel_stops; RL_edge_is_not_bearing',
        old422_codec_preserved=True, task_state_and_potential_semantics_changed=True,
        raw_sigma_kernel='unchanged_cooperative422_slice_0_422;new_features_only_enter_MLP',
        planned_curriculum_weights=dict(CURRICULUM_WEIGHTS), rollout_length_preserved=128,
        quantity_ceiling_change='phase_suffix100000_to131072_only;entropy_LR_horizon_unchanged;used_counts_preserved',
        old_rollout_inherited=False, added_policy_decisions=0, added_ppo_updates=0,
        added_optimizer_steps=0, added_auxiliary_updates=0)
    record = dict(schema=SCHEMA, reason=reason.strip(), source_checkpoint=str(checkpoint),
        source_checkpoint_sha256=SOURCE_SHA, source_manifest_sha256=SOURCE_MANIFEST_SHA,
        source_git_commit=old['source_git_commit'], target_git_commit=new['source_git_commit'],
        source_runtime_content_sha256=old['runtime_content_sha256'], target_runtime_content_sha256=new['runtime_content_sha256'],
        source_contract_sha256=digest(old), target_contract_sha256=digest(new),
        changed_file_hashes={p:dict(before=old['files'].get(p),after=new['files'][p]) for p in changed},
        source_changed_configuration=source_config, discard_old_rollout_storage=True, **{FACTOR_KEY:factor})
    record['source_training_budgets'] = copy.deepcopy(old['training_budgets'])
    if _previous_contract(new,record) != old: raise ValueError('runtime changed outside reviewed delta')
    return record


def validate_rear_owner_migration(checkpoint, contract, plan_path, *, project_root=None):
    from .semantic_migration import file_sha
    path = Path(plan_path).resolve(strict=True); record = json.loads(path.read_text(encoding='utf-8'))
    expected = build_rear_owner_migration(checkpoint,contract,reason=record.get('reason',''),project_root=project_root)
    if record != expected: raise ValueError('explicit owner append plan changed')
    return {**record,'plan_path':str(path),'plan_sha256':file_sha(path)}


def zero_append_rear_owner_training_state(bundle):
    import torch
    result = copy.deepcopy(bundle); shapes = []
    names = ['mlp.0.weight','mlp.0.bias','mlp.2.weight','mlp.2.bias','mlp.4.weight','mlp.4.bias']
    for role in ('actor','critic'):
        state = result[role+'_state_dict']
        if list(state) != names or tuple(state[names[0]].shape) != (256,422): raise ValueError('exact422 MLP required')
        shapes.extend(tuple(v.shape) for v in state.values())
        weight = state[names[0]]; state[names[0]] = torch.cat((weight,weight.new_zeros(256,17)),dim=1)
    optimizer = result['optimizer_state_dict']
    if (len(optimizer['param_groups']) != 1 or optimizer['param_groups'][0]['params'] != list(range(12))
            or set(optimizer['state']) != set(range(12))): raise ValueError('complete official actor-critic Adam required')
    for ident, shape in enumerate(shapes):
        state = optimizer['state'][ident]
        if set(state) != {'step','exp_avg','exp_avg_sq'} or state['step'].numel() != 1: raise ValueError('full Adam state required')
        for key in ('exp_avg','exp_avg_sq'):
            value = state[key]
            if tuple(value.shape) != shape: raise ValueError('Adam moment mapping differs')
            if ident in (0,6): state[key] = torch.cat((value,value.new_zeros(256,17)),dim=1)
    return result


def validate_rear_owner_lineage(metadata, contract, output_root, *, checkpoint_output_routing=None):
    from .semantic_migration import digest
    receipt = metadata.get(MIGRATION,{}); factor = receipt.get(FACTOR_KEY,{})
    if (receipt.get('schema') != SCHEMA or factor.get('schema') != SCHEMA
            or receipt.get('source_checkpoint_sha256') != SOURCE_SHA or receipt.get('source_manifest_sha256') != SOURCE_MANIFEST_SHA
            or receipt.get('source_git_commit') != SOURCE_HEAD or receipt.get('target_git_commit') != contract.get('source_git_commit')
            or receipt.get('target_contract_sha256') != digest(contract)
            or receipt.get('target_runtime_content_sha256') != contract.get('runtime_content_sha256')
            or factor.get('revision_counter_origin') != REVISION_ORIGIN or factor.get('original_branch_origin') != SOURCE_COUNTS
            or factor.get('source_policy_contract') != _policy() or factor.get('target_policy_contract') != _policy(True)
            or metadata.get('policy_contract') != _policy(True) or metadata.get('runner_config') != factor.get('target_runner_config')
            or factor.get('actor_critic_mapping') != 'old422_columns_exact_new17_zero'
            or factor.get('rear_owner_recovery_mode') != REAR_OWNER_MODE
            or factor.get('cooperative_preparation_revision') != TASK_PROXY_MODE
            or any(type(metadata.get(k)) is not int or metadata[k] < REVISION_ORIGIN[k] for k in COUNTERS)
            or any(factor.get(k) != 0 for k in ('added_policy_decisions','added_ppo_updates','added_optimizer_steps','added_auxiliary_updates'))):
        raise ValueError('owner append lineage/policy/counts changed')
    for key, sha in factor['preserved_metadata_sha256'].items():
        if key.endswith(('_branch','_migration')) or key == 'checkpoint_output_routing':
            if digest(metadata.get(key)) != sha: raise ValueError('protected source lineage changed: '+key)
    old = _previous_contract(contract,receipt)
    historical = {k:v for k,v in metadata.items() if k != MIGRATION}
    historical.update(runtime_contract=old,policy_contract=factor['source_policy_contract'],runner_config=factor['source_runner_config'])
    validate_cooperative_prep_lineage(historical,old,output_root,checkpoint_output_routing=checkpoint_output_routing)
    return receipt


def load_rear_owner_migration(runner, checkpoint, *, contract, seed, record):
    import torch
    from .semantic_migration import checkpoint_metadata, continuation_topology, digest
    from .semantic_training import construct_semantic_runner, load_semantic_checkpoint, _runner_policy_contract, state_hash, _normalizers, parameter_hash
    from .semantic_rr_capture_migration import _shape_env
    from .rl_library_wrapper import optimizer_learning_rate, restore_training_rng_state
    verified = validate_rear_owner_migration(checkpoint,contract,record['plan_path'])
    m = checkpoint_metadata(Path(checkpoint)); factor = verified[FACTOR_KEY]; device = str(runner.device)
    if (verified != dict(record) or seed != m['seed'] or _runner_policy_contract(runner) != factor['target_policy_contract']
            or runner._semantic_runner_config != factor['target_runner_config']
            or runner.alg.storage.step != 0 or runner.alg.transition.actions is not None
            or tuple(runner.alg.storage.actions.shape) != (128,1,12)
            or any(tuple(runner.alg.storage.observations[k].shape) != (128,1,439) for k in ('policy','critic'))):
        raise ValueError('owner append requires original seed/device/PPO config and empty439 rollout')
    source,_ = construct_semantic_runner(_shape_env(422,device),seed=seed,device=device,
        policy_version=COOPERATIVE_PREP_POLICY,observation_layout=COOPERATIVE_PREP_OBSERVATION_LAYOUT,initialize_actor=False)
    if source._semantic_runner_config != factor['source_runner_config']: raise ValueError('source PPO config differs')
    infos = load_semantic_checkpoint(source,Path(checkpoint),contract=m['runtime_contract'],seed=seed)
    mapped = zero_append_rear_owner_training_state(torch.load(checkpoint,map_location=device,weights_only=False))
    for role in ('actor','critic'):
        model = getattr(runner.alg,role)
        if type(model.obs_normalizer) is not torch.nn.Identity: raise ValueError('Identity normalizer required')
        model.load_state_dict(mapped[role+'_state_dict'],strict=True)
    runner.alg.optimizer.load_state_dict(mapped['optimizer_state_dict'])
    runner.alg.learning_rate = m['optimizer_learning_rate']; runner.current_learning_iteration = mapped['iter']
    if optimizer_learning_rate(runner) != m['optimizer_learning_rate'] or state_hash(_normalizers(runner)) != m['normalizer_state_sha256']:
        raise ValueError('LR/Identity changed')
    for key, sha in factor['preserved_metadata_sha256'].items():
        if digest(infos[key]) != sha: raise ValueError('source fullstate metadata changed: '+key)
    restore_training_rng_state(m['training_rng_state'],expected_seed=seed)
    sampling = 'P01_full_task_only_initial_version'
    return {**infos,'runtime_contract':dict(contract),'runner_config':copy.deepcopy(runner._semantic_runner_config),
        'policy_contract':_runner_policy_contract(runner),'resume_migration':verified,MIGRATION:verified,
        'actor_parameter_sha256':parameter_hash(runner.alg.actor),'critic_parameter_sha256':parameter_hash(runner.alg.critic),
        'optimizer_state_sha256':state_hash(runner.alg.optimizer.state_dict()),
        'sampling':sampling,'stage':'full_episode','implemented_reset_sampling':sampling,
        'phase_suffix_curriculum_implemented':False,'curriculum_epoch':{'reset_sampling':sampling,'prefix_request':None},
        'execution_topology':continuation_topology(sampling,None,observation_layout=REAR_OWNER_OBSERVATION_LAYOUT),
        'old_rollout_inherited':False,'physical_env_state_saved':False,'resume_physics':'fresh_legal_reset'}


def publish_rear_owner_checkpoint(checkpoint, contract, plan_path, output_checkpoint):
    from .semantic_migration import checkpoint_metadata, file_sha
    from .semantic_training import construct_semantic_runner, load_semantic_checkpoint, save_semantic_checkpoint, state_hash, parameter_hash
    from .semantic_rr_capture_migration import _shape_env
    m = checkpoint_metadata(Path(checkpoint)); record = validate_rear_owner_migration(checkpoint,contract,plan_path)
    destination = Path(output_checkpoint).resolve(); route = m['checkpoint_output_routing']
    if (destination.parent != Path(route['output_root'])/'checkpoints/history' or destination.exists()
            or destination.with_name(destination.stem+'_manifest.json').exists()):
        raise ValueError('unique same-branch publication required; never promote latest pointer here')
    def make():
        device = m['runner_config']['device']
        return construct_semantic_runner(_shape_env(439,device),seed=m['seed'],device=device,
            policy_version=REAR_OWNER_POLICY,observation_layout=REAR_OWNER_OBSERVATION_LAYOUT,initialize_actor=False)[0]
    runner = make(); infos = load_rear_owner_migration(runner,checkpoint,contract=contract,seed=m['seed'],record=record)
    saved, manifest = save_semantic_checkpoint(runner,destination,infos)
    fresh = make(); loaded = load_semantic_checkpoint(fresh,saved,contract=contract,seed=m['seed'])
    for key in _preserved(m):
        if loaded[key] != m[key]: raise RuntimeError('publication changed original state/lineage: '+key)
    for role in ('actor','critic'):
        if parameter_hash(getattr(fresh.alg,role)) != parameter_hash(getattr(runner.alg,role)):
            raise RuntimeError('independent weight reload differs')
    if state_hash(fresh.alg.optimizer.state_dict()) != state_hash(runner.alg.optimizer.state_dict()):
        raise RuntimeError('independent full Adam reload differs')
    validate_rear_owner_lineage(loaded,contract,Path(route['output_root']),checkpoint_output_routing=route)
    return dict(checkpoint=str(saved),checkpoint_sha256=file_sha(saved),manifest=str(manifest),manifest_sha256=file_sha(manifest),
        save_load_round_trip=True,**{k:loaded[k] for k in COUNTERS},rear_policy_timing_branch_counts=loaded['rear_policy_timing_branch_counts'],
        added_policy_decisions=0,added_ppo_updates=0,added_optimizer_steps=0,added_auxiliary_updates=0)
