"""Reviewed latest-learned 419->422 append; no optimizer reset or borrowed credit."""
from __future__ import annotations
import copy
import json
from pathlib import Path
from .semantic_rear_policy_timing_migration import (
    EXPERIMENT, COUNTERS, SOURCE_COUNTS, preserved_keys, validate_rear_policy_namespace)
from .semantic_rear_live_swing_migration import _previous_contract
from .semantic_rear_recapture_migration import CODE, CONFIG, BRANCH_NAME
from .semantic_rear_policy_timing_profile import REAR_POLICY_TIMING_POLICY, REAR_POLICY_TIMING_OBSERVATION_LAYOUT
from .semantic_p02_progress_profile import (
    P02_PROGRESS_POLICY, P02_PROGRESS_OBSERVATION_LAYOUT, P02_PROGRESS_MODE, P02_PROGRESS_GROUP)

SCHEMA='wlr50_clean.p02_progress_append.v1'
FACTOR_KEY='p02_progress_append_factor'
MIGRATION='p02_progress_migration'
SOURCE_HEAD='45862675a18a5bb2b75f0d4883e0cee2a3be24e3'
SOURCE_SHA='da63ac820ab7b260e161a116b5c925370090c087b4cce222859626b8faec24e1'
SOURCE_MANIFEST_SHA='956737aab9e91c3fc11d77a237f4ec03613877c53081f16c4b6b81e82c5501b6'
REVISION_ORIGIN=dict(zip(COUNTERS,(221696,1697,33940)))
ALLOWED=frozenset(CODE+n for n in (
    'semantic_p02_progress_profile.py','semantic_p02_progress_actor.py','semantic_p02_progress_migration.py',
    'semantic_supervisor.py','semantic_backend.py','semantic_observation.py','semantic_policy_distribution.py',
    'semantic_training.py','semantic_migration.py','semantic_cli.py','semantic_checkpoint_prefix.py',
    'semantic_checkpoint_prefix_policy.py','semantic_rear_policy_timing_migration.py')) | {
    CONFIG+n for n in ('execution_profile.yaml','stage_task_spec.yaml','observation_schema.json','curriculum_plan.json')}
WEIGHTS=(2/3,1/6,1/6)


def _policy(version,layout):
    from .semantic_policy_distribution import policy_contract
    return policy_contract(version,observation_layout=layout)


def _configs(root,old,new):
    import yaml
    from .semantic_migration import file_sha,_version_bytes
    if set(old['selected_configuration'])!=set(new['selected_configuration']):
        raise ValueError('configuration inventory changed')
    changed={}
    for name,entry in new['selected_configuration'].items():
        previous=old['selected_configuration'][name]
        if entry['path']!=previous['path'] or file_sha(root/entry['path'])!=entry['sha256']:
            raise ValueError('configuration binding differs')
        before=_version_bytes(root,old,previous['path']);after=(root/entry['path']).read_bytes()
        if previous!=entry:changed[name]=previous
        if name not in ('execution_profile.yaml','stage_task_spec.yaml','observation_schema.json','curriculum_plan.json'):
            if before!=after:raise ValueError('unrelated configuration changed: '+name)
            continue
        a,b=yaml.safe_load(before),yaml.safe_load(after)
        if name in ('execution_profile.yaml','stage_task_spec.yaml'):
            if a.get('p02_progress_credit_mode') is not None or b.pop('p02_progress_credit_mode',None)!=P02_PROGRESS_MODE:
                raise ValueError('explicit new P02 flag required')
            if name=='execution_profile.yaml':
                if a['rear_policy_timing_mode']!='rr_live_swing_evidence_v3' or b['rear_policy_timing_mode']!=a['rear_policy_timing_mode']:
                    raise ValueError('rear control mode changed')
                a.pop('revision',None);b.pop('revision',None)
                for key,value in zip(('full_P01','predecessor_P07_suffix','second_rear_P10_suffix'),WEIGHTS):
                    if b['sampling_target'].get(key)!=value:raise ValueError('curriculum allocation differs')
                    a['sampling_target'].pop(key);b['sampling_target'].pop(key)
        elif name=='observation_schema.json':
            if (a.get('p02_progress_features_version') is not None
                    or b.pop('p02_progress_features_version',None)!=P02_PROGRESS_OBSERVATION_LAYOUT
                    or b['feature_groups'].pop()!=dict(name=P02_PROGRESS_GROUP,size=3,scale=1)):
                raise ValueError('exact public3 append schema required')
            a.pop('revision',None);b.pop('revision',None)
        else:
            if [e['from_phase'] for e in b['entries']]!=['P01','P07','P10']:
                raise ValueError('P01-first curriculum order required')
            for prior,target,weight in zip(a['entries'],b['entries'],WEIGHTS,strict=True):
                if target.get('weight')!=weight:raise ValueError('whole-block curriculum weights differ')
                prior.pop('weight');target.pop('weight')
        if a!=b:raise ValueError('unrelated control/physical/codec/curriculum config changed: '+name)
    return changed


def build_p02_progress_migration(checkpoint,current_contract,*,reason,project_root=None):
    from .semantic_migration import checkpoint_metadata,_contract,digest,file_sha,source_num_envs
    root=Path(project_root or Path(__file__).resolve().parents[3])
    checkpoint=Path(checkpoint).resolve(strict=True);metadata=checkpoint_metadata(checkpoint)
    branch=root/'outputs'/('ppo_'+EXPERIMENT)/'branches'/BRANCH_NAME
    if (checkpoint.parent!=(branch/'checkpoints/history').resolve() or file_sha(checkpoint)!=SOURCE_SHA
            or file_sha(checkpoint.with_name(checkpoint.stem+'_manifest.json'))!=SOURCE_MANIFEST_SHA):
        raise ValueError('requires exact sealed learned221696 source, no ancestor rollback')
    old,new=_contract(metadata['runtime_contract']),_contract(current_contract)
    validate_rear_policy_namespace(metadata,old,branch,checkpoint_output_routing=metadata.get('checkpoint_output_routing'))
    source_policy=_policy(REAR_POLICY_TIMING_POLICY,REAR_POLICY_TIMING_OBSERVATION_LAYOUT)
    target_policy=_policy(P02_PROGRESS_POLICY,P02_PROGRESS_OBSERVATION_LAYOUT)
    if (old['source_git_commit']!=SOURCE_HEAD or not reason.strip() or MIGRATION in metadata
            or {k:metadata[k] for k in COUNTERS}!=REVISION_ORIGIN or source_num_envs(metadata)!=1
            or metadata['policy_contract']!=source_policy or metadata.get('save_load_round_trip') is not True):
        raise ValueError('source419 learned state/runtime/counts not intact')
    changed=sorted(p for p in set(old['files'])|set(new['files']) if old['files'].get(p)!=new['files'].get(p))
    if not changed or not set(changed)<=ALLOWED or any(p not in new['files'] for p in changed):
        raise ValueError('unreviewed runtime change')
    for path,value in new['files'].items():
        if file_sha(root/path)!=value:raise ValueError('target bytes changed: '+path)
    source_config=_configs(root,old,new)
    factor=dict(schema=SCHEMA,source_policy_contract=source_policy,target_policy_contract=target_policy,
        revision_counter_origin=REVISION_ORIGIN,original_branch_origin=SOURCE_COUNTS,
        source_effective_learning_rate=metadata['optimizer_learning_rate'],
        preserved_metadata_sha256={k:digest(metadata[k]) for k in (*preserved_keys(metadata),'checkpoint_output_routing')},
        actor_critic_mapping='old419_columns_exact_new3_zero',
        Adam_mapping='all_old_moments_steps_options_exact_new3_first_weight_moment_columns_zero',
        same_mdp_claimed=False,physical_dynamics_changed=False,rear_control_changed=False,
        raw_sigma_kernel='unchanged_rear419_slice_0_419; new_features_only_enter_MLP',
        old_rollout_inherited=False,added_policy_decisions=0,added_ppo_updates=0,
        added_optimizer_steps=0,added_auxiliary_updates=0,
        sampling_plan=dict(natural_P01=1024,P07=256,P10=256,total=1536,
            plan_only_not_earned=True,reason='longer_natural_front_coverage; old_short_blocks_never_sampled_P03_P06'))
    record=dict(schema=SCHEMA,reason=reason.strip(),source_checkpoint=str(checkpoint),
        source_checkpoint_sha256=SOURCE_SHA,source_manifest_sha256=SOURCE_MANIFEST_SHA,
        source_git_commit=old['source_git_commit'],target_git_commit=new['source_git_commit'],
        source_runtime_content_sha256=old['runtime_content_sha256'],target_runtime_content_sha256=new['runtime_content_sha256'],
        source_contract_sha256=digest(old),target_contract_sha256=digest(new),
        changed_file_hashes={p:dict(before=old['files'].get(p),after=new['files'][p]) for p in changed},
        source_changed_configuration=source_config,discard_old_rollout_storage=True,**{FACTOR_KEY:factor})
    if _previous_contract(new,record)!=old:raise ValueError('contract changed outside reviewed delta')
    return record


def validate_p02_progress_migration(checkpoint,contract,plan_path,*,project_root=None):
    from .semantic_migration import file_sha
    path=Path(plan_path).resolve(strict=True);record=json.loads(path.read_text(encoding='utf-8'))
    expected=build_p02_progress_migration(checkpoint,contract,reason=record.get('reason',''),project_root=project_root)
    if record!=expected:raise ValueError('P02 append/source/runtime differs from explicit plan')
    return {**record,'plan_path':str(path),'plan_sha256':file_sha(path)}


def zero_append_p02_training_state(bundle):
    import torch
    result=copy.deepcopy(bundle);shapes=[]
    names=['mlp.0.weight','mlp.0.bias','mlp.2.weight','mlp.2.bias','mlp.4.weight','mlp.4.bias']
    for role in ('actor','critic'):
        state=result[role+'_state_dict']
        if list(state)!=names or tuple(state[names[0]].shape)!=(256,419):raise ValueError('exact419 MLP required')
        shapes.extend(tuple(v.shape) for v in state.values())
        weight=state[names[0]];state[names[0]]=torch.cat((weight,weight.new_zeros(256,3)),dim=1)
    optimizer=result['optimizer_state_dict']
    if (len(optimizer['param_groups'])!=1 or optimizer['param_groups'][0]['params']!=list(range(12))
            or set(optimizer['state'])!=set(range(12))):raise ValueError('complete official actor-critic Adam required')
    for ident,shape in enumerate(shapes):
        state=optimizer['state'][ident]
        if set(state)!={'step','exp_avg','exp_avg_sq'} or state['step'].numel()!=1:raise ValueError('full Adam state required')
        for key in ('exp_avg','exp_avg_sq'):
            value=state[key]
            if tuple(value.shape)!=shape:raise ValueError('Adam moment mapping differs')
            if ident in (0,6):state[key]=torch.cat((value,value.new_zeros(256,3)),dim=1)
    return result


def validate_p02_progress_lineage(metadata,contract,output_root,*,checkpoint_output_routing=None):
    from .semantic_migration import digest
    receipt=metadata.get(MIGRATION,{});factor=receipt.get(FACTOR_KEY,{})
    if (receipt.get('schema')!=SCHEMA or factor.get('schema')!=SCHEMA
            or receipt.get('source_checkpoint_sha256')!=SOURCE_SHA or receipt.get('source_manifest_sha256')!=SOURCE_MANIFEST_SHA
            or receipt.get('source_git_commit')!=SOURCE_HEAD or receipt.get('target_git_commit')!=contract.get('source_git_commit')
            or receipt.get('target_contract_sha256')!=digest(contract)
            or factor.get('revision_counter_origin')!=REVISION_ORIGIN or factor.get('original_branch_origin')!=SOURCE_COUNTS
            or factor.get('source_policy_contract')!=_policy(REAR_POLICY_TIMING_POLICY,REAR_POLICY_TIMING_OBSERVATION_LAYOUT)
            or factor.get('target_policy_contract')!=_policy(P02_PROGRESS_POLICY,P02_PROGRESS_OBSERVATION_LAYOUT)
            or metadata.get('policy_contract')!=factor['target_policy_contract']
            or any(type(metadata.get(k)) is not int or metadata[k]<REVISION_ORIGIN[k] for k in COUNTERS)
            or any(factor.get(k)!=0 for k in ('added_policy_decisions','added_ppo_updates','added_optimizer_steps','added_auxiliary_updates'))):
        raise ValueError('P02 append lineage/policy/counts changed')
    for key in ('rear_live_swing_migration','rear_recapture_migration','rear_policy_timing_migration',
                'rear_policy_timing_branch','checkpoint_output_routing'):
        if factor['preserved_metadata_sha256'].get(key)!=digest(metadata.get(key)):
            raise ValueError('protected source lineage changed: '+key)
    old=_previous_contract(contract,receipt)
    historical={k:v for k,v in metadata.items() if k!=MIGRATION}
    historical.update(runtime_contract=old,policy_contract=factor['source_policy_contract'])
    validate_rear_policy_namespace(historical,old,output_root,checkpoint_output_routing=checkpoint_output_routing)
    return receipt


def load_p02_progress_migration(runner,checkpoint,*,contract,seed,record):
    import torch
    from .semantic_migration import checkpoint_metadata,continuation_topology,digest
    from .semantic_training import construct_semantic_runner,load_semantic_checkpoint,_runner_policy_contract,state_hash,_normalizers,parameter_hash
    from .semantic_rr_capture_migration import _shape_env
    from .rl_library_wrapper import optimizer_learning_rate,restore_training_rng_state
    verified=validate_p02_progress_migration(checkpoint,contract,record['plan_path'])
    m=checkpoint_metadata(Path(checkpoint));device=str(runner.device);factor=verified[FACTOR_KEY]
    if (verified!=dict(record) or seed!=m['seed'] or _runner_policy_contract(runner)!=factor['target_policy_contract']
            or runner.alg.storage.step!=0 or runner.alg.transition.actions is not None
            or tuple(runner.alg.storage.actions.shape)!=(128,1,12)
            or any(tuple(runner.alg.storage.observations[k].shape)!=(128,1,422) for k in ('policy','critic'))):
        raise ValueError('P02 append requires original seed and empty422 rollout')
    source,_=construct_semantic_runner(_shape_env(419,device),seed=seed,device=device,
        policy_version=REAR_POLICY_TIMING_POLICY,observation_layout=REAR_POLICY_TIMING_OBSERVATION_LAYOUT,initialize_actor=False)
    expected=copy.deepcopy(source._semantic_runner_config)
    expected['actor'].update(class_name=factor['target_policy_contract']['actor_class'],observation_layout=P02_PROGRESS_OBSERVATION_LAYOUT)
    if runner._semantic_runner_config!=expected:raise ValueError('unrelated PPO configuration changed')
    infos=load_semantic_checkpoint(source,Path(checkpoint),contract=m['runtime_contract'],seed=seed)
    mapped=zero_append_p02_training_state(torch.load(checkpoint,map_location=device,weights_only=False))
    for role in ('actor','critic'):
        model=getattr(runner.alg,role)
        if type(model.obs_normalizer) is not torch.nn.Identity:raise ValueError('Identity normalizer required')
        model.load_state_dict(mapped[role+'_state_dict'],strict=True)
    runner.alg.optimizer.load_state_dict(mapped['optimizer_state_dict'])
    runner.alg.learning_rate=m['optimizer_learning_rate'];runner.current_learning_iteration=mapped['iter']
    if optimizer_learning_rate(runner)!=m['optimizer_learning_rate'] or state_hash(_normalizers(runner))!=m['normalizer_state_sha256']:
        raise ValueError('LR/Identity changed')
    for key,value in factor['preserved_metadata_sha256'].items():
        if digest(infos[key])!=value:raise ValueError('preserved source metadata changed: '+key)
    restore_training_rng_state(m['training_rng_state'],expected_seed=seed)
    sampling='P01_full_task_only_initial_version'
    return {**infos,'runtime_contract':dict(contract),'runner_config':copy.deepcopy(runner._semantic_runner_config),
        'policy_contract':_runner_policy_contract(runner),'resume_migration':verified,MIGRATION:verified,
        'actor_parameter_sha256':parameter_hash(runner.alg.actor),'critic_parameter_sha256':parameter_hash(runner.alg.critic),
        'optimizer_state_sha256':state_hash(runner.alg.optimizer.state_dict()),
        'sampling':sampling,'stage':'full_episode','implemented_reset_sampling':sampling,
        'phase_suffix_curriculum_implemented':False,'curriculum_epoch':{'reset_sampling':sampling,'prefix_request':None},
        'execution_topology':continuation_topology(sampling,None,observation_layout=P02_PROGRESS_OBSERVATION_LAYOUT),
        'old_rollout_inherited':False,'physical_env_state_saved':False,'resume_physics':'fresh_legal_reset'}


def publish_p02_progress_checkpoint(checkpoint,contract,plan_path,output_checkpoint):
    from .semantic_migration import checkpoint_metadata,file_sha
    from .semantic_training import construct_semantic_runner,load_semantic_checkpoint,save_semantic_checkpoint,state_hash,parameter_hash
    from .semantic_rr_capture_migration import _shape_env
    m=checkpoint_metadata(Path(checkpoint));record=validate_p02_progress_migration(checkpoint,contract,plan_path)
    destination=Path(output_checkpoint).resolve();route=m['checkpoint_output_routing']
    if (destination.parent!=Path(route['output_root'])/'checkpoints/history' or destination.exists()
            or destination.with_name(destination.stem+'_manifest.json').exists()):
        raise ValueError('unique same-branch publication required')
    def make():
        device=m['runner_config']['device']
        return construct_semantic_runner(_shape_env(422,device),seed=m['seed'],device=device,
            policy_version=P02_PROGRESS_POLICY,observation_layout=P02_PROGRESS_OBSERVATION_LAYOUT,initialize_actor=False)[0]
    runner=make();infos=load_p02_progress_migration(runner,checkpoint,contract=contract,seed=m['seed'],record=record)
    saved,manifest=save_semantic_checkpoint(runner,destination,infos)
    fresh=make();loaded=load_semantic_checkpoint(fresh,saved,contract=contract,seed=m['seed'])
    for key in preserved_keys(m):
        if loaded[key]!=m[key]:raise RuntimeError('publication changed original lineage/counts/RNG: '+key)
    for role in ('actor','critic'):
        if parameter_hash(getattr(fresh.alg,role))!=parameter_hash(getattr(runner.alg,role)):
            raise RuntimeError('independent fresh weight reload differs')
    if state_hash(fresh.alg.optimizer.state_dict())!=state_hash(runner.alg.optimizer.state_dict()):
        raise RuntimeError('independent full Adam reload differs')
    validate_p02_progress_lineage(loaded,contract,Path(route['output_root']),checkpoint_output_routing=route)
    return dict(checkpoint=str(saved),checkpoint_sha256=file_sha(saved),manifest=str(manifest),manifest_sha256=file_sha(manifest),
        save_load_round_trip=True,**{k:loaded[k] for k in COUNTERS},rear_policy_timing_branch_counts=loaded['rear_policy_timing_branch_counts'],
        added_policy_decisions=0,added_ppo_updates=0,added_optimizer_steps=0,added_auxiliary_updates=0)
