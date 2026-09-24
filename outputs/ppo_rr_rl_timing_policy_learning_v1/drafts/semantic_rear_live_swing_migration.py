"""UNAPPLIED: thin learned-branch same419 continuation, pending control review."""
from __future__ import annotations
import copy
import json
from pathlib import Path

from .semantic_rear_recapture_migration import (
    _preserved, _verify_identity, CODE, CONFIG, BRANCH_NAME,
    TARGET_MODE as SOURCE_MODE, MIGRATION as PARENT_RECEIPT)
from .semantic_rear_policy_timing_migration import EXPERIMENT, COUNTERS, SOURCE_COUNTS
from .semantic_rear_policy_timing_profile import (
    REAR_POLICY_TIMING_POLICY, REAR_POLICY_TIMING_OBSERVATION_LAYOUT)

SCHEMA = 'wlr50_clean.rear_live_swing_same419.v3'
FACTOR_KEY = 'rear_live_swing_same419_factor'
MIGRATION = 'rear_live_swing_migration'
TARGET_MODE = 'rr_live_swing_evidence_v3'  # Proposed, pending final control review.
CONTROL_REVIEW_COMPLETE = False  # Fail closed until root reviews the actual patch.
SOURCE_HEAD = '44219b4fdc4d36d33be489b833c03b897766045b'
SOURCE_SHA = '1827b5d59935b31e1e27cc7ae0c364dd0203a9d7f77a7d9e9502189bbd31c7d5'
SOURCE_MANIFEST_SHA = 'b27d373edb11ed5d9455bc3d6b32dda4657c9b08b08f91d9d42de0d425686b75'
REVISION_ORIGIN = dict(zip(COUNTERS, (221568, 1696, 33920)))
ALLOWED = frozenset(CODE + name for name in (
    'semantic_rear_live_swing_migration.py', 'semantic_rear_policy_timing_migration.py',
    'semantic_migration.py', 'semantic_training.py', 'semantic_cli.py',
    'semantic_rear_policy_timing.py', 'semantic_supervisor.py')) | {
    CONFIG+'execution_profile.yaml', CONFIG+'stage_task_spec.yaml'}


def _previous_contract(current, receipt):
    """Reverse only bound file/config deltas; never duplicate a whole runtime."""
    from .semantic_migration import digest
    previous = copy.deepcopy(current)
    previous['source_git_commit'] = receipt['source_git_commit']
    previous['runtime_content_sha256'] = receipt['source_runtime_content_sha256']
    for path, hashes in receipt['changed_file_hashes'].items():
        if previous['files'].get(path) != hashes['after']:
            raise ValueError('current runtime does not match declared live-swing delta')
        if hashes['before'] is None:
            previous['files'].pop(path)
        else:
            previous['files'][path] = hashes['before']
    previous['selected_configuration'].update(receipt['source_changed_configuration'])
    if digest(previous) != receipt['source_contract_sha256']:
        raise ValueError('historical v2 runtime reconstruction is not exact')
    return previous


def build_rear_live_swing_migration(checkpoint, current_contract, *, reason, project_root=None):
    import yaml
    from .semantic_migration import checkpoint_metadata, digest, file_sha, _contract, _version_bytes
    from .semantic_rear_policy_timing_migration import validate_rear_policy_namespace
    from .semantic_policy_distribution import policy_contract
    if CONTROL_REVIEW_COMPLETE is not True:
        raise ValueError('draft only: final RL evidence/control semantics not reviewed')
    root = Path(project_root or Path(__file__).resolve().parents[3])
    checkpoint = Path(checkpoint).resolve(strict=True)
    branch = root/'outputs'/('ppo_'+EXPERIMENT)/'branches'/BRANCH_NAME
    sidecar = checkpoint.with_name(checkpoint.stem+'_manifest.json')
    if (checkpoint.parent != (branch/'checkpoints/history').resolve()
            or file_sha(checkpoint) != SOURCE_SHA or file_sha(sidecar) != SOURCE_MANIFEST_SHA):
        raise ValueError('requires the registered latest learned v2 branch source, not ancestor rollback')
    metadata = checkpoint_metadata(checkpoint)
    old, new = _contract(metadata['runtime_contract']), _contract(current_contract)
    route = metadata.get('checkpoint_output_routing')
    validate_rear_policy_namespace(metadata, old, branch, checkpoint_output_routing=route)
    policy = policy_contract(REAR_POLICY_TIMING_POLICY, observation_layout=REAR_POLICY_TIMING_OBSERVATION_LAYOUT)
    if (old['source_git_commit'] != SOURCE_HEAD or MIGRATION in metadata or not reason.strip()
            or {k:metadata.get(k) for k in COUNTERS} != REVISION_ORIGIN
            or metadata.get('rear_policy_timing_branch_counts') != {k:REVISION_ORIGIN[k]-SOURCE_COUNTS[k] for k in COUNTERS}
            or metadata.get('policy_contract') != policy or metadata.get('save_load_round_trip') is not True
            or metadata['runtime_contract']['experiment_id'] != EXPERIMENT):
        raise ValueError('sealed same419 learned branch/policy/counters are not intact')
    changed = sorted(p for p in set(old['files'])|set(new['files']) if old['files'].get(p) != new['files'].get(p))
    if not changed or not set(changed) <= ALLOWED or any(p not in new['files'] for p in changed):
        raise ValueError('unreviewed runtime delta or removed file')
    for path, value in new['files'].items():
        if file_sha(root/path) != value:
            raise ValueError('target bytes changed: '+path)
    if set(old['selected_configuration']) != set(new['selected_configuration']):
        raise ValueError('configuration inventory changed')
    old_bindings = {}
    for name, entry in new['selected_configuration'].items():
        before_entry = old['selected_configuration'][name]
        if entry['path'] != before_entry['path'] or file_sha(root/entry['path']) != entry['sha256']:
            raise ValueError('configuration path/hash mismatch')
        before = _version_bytes(root, old, before_entry['path'])
        after = (root/entry['path']).read_bytes()
        if entry != before_entry:
            old_bindings[name] = before_entry
        if name not in ('execution_profile.yaml','stage_task_spec.yaml'):
            if before != after:
                raise ValueError('unrelated config changed: '+name)
            continue
        a, b = yaml.safe_load(before), yaml.safe_load(after)
        if name == 'execution_profile.yaml':
            if a.get('rear_policy_timing_mode') != SOURCE_MODE or b.get('rear_policy_timing_mode') != TARGET_MODE:
                raise ValueError('source/target execution mode missing')
            for cfg in (a,b):
                cfg.pop('revision',None); cfg.pop('rear_policy_timing_mode',None)
        else:
            if a['nominal'].get('rear_policy_timing') != SOURCE_MODE or b['nominal'].get('rear_policy_timing') != TARGET_MODE:
                raise ValueError('source/target task mode missing')
            a['nominal'].pop('rear_policy_timing'); b['nominal'].pop('rear_policy_timing')
        if a != b:
            raise ValueError('unrelated physical/nominal/assist/reward configuration changed')
    record = dict(schema=SCHEMA,reason=reason.strip(),source_checkpoint=str(checkpoint),
        source_checkpoint_sha256=SOURCE_SHA,source_manifest_sha256=SOURCE_MANIFEST_SHA,
        source_git_commit=old['source_git_commit'],target_git_commit=new['source_git_commit'],
        source_runtime_content_sha256=old['runtime_content_sha256'],target_runtime_content_sha256=new['runtime_content_sha256'],
        source_contract_sha256=digest(old),target_contract_sha256=digest(new),
        changed_file_hashes={p:dict(before=old['files'].get(p),after=new['files'][p]) for p in changed},
        source_changed_configuration=old_bindings,discard_old_rollout_storage=True,
        source_role='latest_sealed_learned_recapture_branch_not_ancestor',
        **{FACTOR_KEY:dict(schema=SCHEMA,source_mode=SOURCE_MODE,target_mode=TARGET_MODE,
            source_policy_contract=policy,target_policy_contract=policy,
            revision_counter_origin=REVISION_ORIGIN,original_branch_origin=SOURCE_COUNTS,
            preserved_metadata_sha256={k:digest(metadata[k]) for k in (*_preserved(metadata),'checkpoint_output_routing')},
            source_effective_learning_rate=metadata['optimizer_learning_rate'],
            parameter_mapping='identity_all_actor_critic_buffers_full_Adam_and_rng',
            observation_dimension=419,action_dimension=12,num_envs=1,
            same_numeric_input_Gaussian_preserved=True,same_mdp_claimed=False,
            physical_dynamics_changed=False,old_rollout_inherited=False,
            changed_semantics='v3_RL_current_attempt_qualification_published_in_existing_RL_active_lift_history_slot_and_current_fields_ground_resets_it_public_swing_dependency_and_retention_follow_current_evidence',
            added_policy_decisions=0,added_ppo_updates=0,added_optimizer_steps=0,added_auxiliary_updates=0)})
    # Also rejects unrelated contract-field drift, including policy/runtime settings.
    if _previous_contract(new,record) != old:
        raise ValueError('same419 runtime contract changed outside reviewed source/config delta')
    return record


def validate_rear_live_swing_migration(checkpoint, contract, plan_path, *, project_root=None):
    from .semantic_migration import file_sha
    path=Path(plan_path).resolve(strict=True); record=json.loads(path.read_text(encoding='utf-8'))
    expected=build_rear_live_swing_migration(checkpoint,contract,reason=record.get('reason',''),project_root=project_root)
    if record != expected:
        raise ValueError('live-swing migration/source/current runtime differs from reviewed plan')
    return {**record,'plan_path':str(path),'plan_sha256':file_sha(path)}


def validate_rear_live_swing_lineage(metadata, contract):
    from .semantic_migration import digest
    from .semantic_rear_policy_timing_migration import validate_rear_policy_namespace
    receipt=metadata.get(MIGRATION,{})
    factor=receipt.get(FACTOR_KEY,{})
    route=metadata.get('checkpoint_output_routing')
    if (receipt.get('schema') != SCHEMA or factor.get('schema') != SCHEMA
            or receipt.get('source_checkpoint_sha256') != SOURCE_SHA or receipt.get('source_manifest_sha256') != SOURCE_MANIFEST_SHA
            or receipt.get('source_git_commit') != SOURCE_HEAD or receipt.get('target_git_commit') != contract.get('source_git_commit')
            or receipt.get('target_contract_sha256') != digest(contract)
            or receipt.get('target_runtime_content_sha256') != contract.get('runtime_content_sha256')
            or factor.get('source_mode') != SOURCE_MODE or factor.get('target_mode') != TARGET_MODE
            or factor.get('revision_counter_origin') != REVISION_ORIGIN or factor.get('original_branch_origin') != SOURCE_COUNTS
            or factor.get('source_policy_contract') != metadata.get('policy_contract')
            or factor.get('target_policy_contract') != metadata.get('policy_contract')
            or any(type(metadata.get(k)) is not int or metadata[k] < REVISION_ORIGIN[k] for k in COUNTERS)
            or any(factor.get(k) != 0 for k in ('added_policy_decisions','added_ppo_updates','added_optimizer_steps','added_auxiliary_updates'))
            or factor.get('preserved_metadata_sha256',{}).get(PARENT_RECEIPT) != digest(metadata.get(PARENT_RECEIPT))
            or factor.get('preserved_metadata_sha256',{}).get('checkpoint_output_routing') != digest(route)):
        raise ValueError('live-swing continuation lost source updates, parent receipt or branch route')
    old=_previous_contract(contract,receipt)
    historical={k:v for k,v in metadata.items() if k!=MIGRATION}
    historical['runtime_contract']=old
    validate_rear_policy_namespace(historical,old,Path(route['output_root']),checkpoint_output_routing=route)
    return receipt


def load_rear_live_swing_migration(runner, checkpoint, *, contract, seed, record):
    from .semantic_migration import checkpoint_metadata
    from .semantic_training import load_semantic_checkpoint, _runner_policy_contract
    verified=validate_rear_live_swing_migration(checkpoint,contract,record['plan_path'])
    source=checkpoint_metadata(Path(checkpoint))
    if verified!=dict(record) or seed!=source['seed'] or _runner_policy_contract(runner)!=verified[FACTOR_KEY]['target_policy_contract']:
        raise ValueError('live-swing plan/seed/policy changed before load')
    infos=load_semantic_checkpoint(runner,Path(checkpoint),contract=source['runtime_contract'],seed=seed)
    _verify_identity(runner,infos,verified[FACTOR_KEY])
    return {**infos,'runtime_contract':dict(contract),'resume_migration':verified,MIGRATION:verified,
        'old_rollout_inherited':False,'physical_env_state_saved':False,
        'resume_physics':'fresh_legal_reset_not_bitwise_simulator_resume'}


def publish_rear_live_swing_checkpoint(checkpoint, contract, plan_path, output_checkpoint):
    from .semantic_migration import checkpoint_metadata,file_sha
    from .semantic_training import construct_semantic_runner,load_semantic_checkpoint,save_semantic_checkpoint
    from .semantic_rr_capture_migration import _shape_env
    source=checkpoint_metadata(Path(checkpoint)); record=validate_rear_live_swing_migration(checkpoint,contract,plan_path)
    route=source['checkpoint_output_routing']; destination=Path(output_checkpoint).resolve()
    if destination.parent != Path(route['output_root'])/'checkpoints/history' or destination.exists():
        raise ValueError('unique immutable same-branch output required; never main pointer')
    def make():
        device=source['runner_config']['device']
        return construct_semantic_runner(_shape_env(419,device),seed=source['seed'],device=device,
            policy_version=REAR_POLICY_TIMING_POLICY,observation_layout=REAR_POLICY_TIMING_OBSERVATION_LAYOUT,
            initialize_actor=False)[0]
    runner=make(); infos=load_rear_live_swing_migration(runner,checkpoint,contract=contract,seed=source['seed'],record=record)
    path,sidecar=save_semantic_checkpoint(runner,destination,infos)
    fresh=make(); loaded=load_semantic_checkpoint(fresh,path,contract=contract,seed=source['seed'])
    _verify_identity(fresh,loaded,record[FACTOR_KEY]); validate_rear_live_swing_lineage(loaded,contract)
    return dict(checkpoint=str(path),checkpoint_sha256=file_sha(path),manifest=str(sidecar),manifest_sha256=file_sha(sidecar),
        save_load_round_trip=True,**{k:loaded[k] for k in COUNTERS},
        rear_policy_timing_branch_counts=loaded['rear_policy_timing_branch_counts'],
        added_policy_decisions=0,added_ppo_updates=0,added_optimizer_steps=0,added_auxiliary_updates=0)
