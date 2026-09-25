"""Explicit selected-ancestor 410->419 continuation; preserve full Adam and RNG.

This is a new control/distribution MDP, not a claim of equivalent trajectories.
The exact source is retained in its old namespace; publication never promotes it.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
import subprocess
from .semantic_rear_policy_timing_profile import (REAR_POLICY_TIMING_POLICY,
    REAR_POLICY_TIMING_OBSERVATION_LAYOUT, REAR_POLICY_TIMING_OBSERVATION_DIM)
from .semantic_rr_capture_profile import RR_CAPTURE_POLICY, RR_CAPTURE_OBSERVATION_LAYOUT

SCHEMA='wlr50_clean.rear_policy_timing_append.v1'
FACTOR_KEY='rear_policy_timing_factor'
MIGRATION='rear_policy_timing_migration'
EXPERIMENT='rr_rl_timing_policy_learning_v1'
COUNTERS=('global_policy_decisions','ppo_updates','optimizer_steps')
SOURCE_SHA='47fdec0614ed2683a0eae6fdef736598400f3c6833473b551a8c93f6f0d8f430'
SOURCE_MANIFEST_SHA='a0906988337c97f5d67108c289b4ec46d6497716cac56df39bedf8d9597362a1'
SOURCE_COUNTS=dict(zip(COUNTERS,(220544,1688,33760)))

def preserved_keys(metadata):
    return tuple(sorted(set(COUNTERS+('seed','stage_requested_decisions','optimizer_learning_rate',
        'normalization','normalizer_state_sha256','training_rng_state')) |
        {k for k in metadata if k!='resume_migration' and k.endswith(('_branch','_branch_counts','_migration'))}))

def build_rear_policy_timing_migration(checkpoint,current_contract,*,reason,project_root=None):
    import yaml
    from .semantic_migration import checkpoint_metadata,digest,file_sha,source_num_envs,_contract,_version_bytes
    from .semantic_policy_distribution import policy_contract
    root=Path(project_root or Path(__file__).resolve().parents[3]);checkpoint=Path(checkpoint).resolve(strict=True)
    metadata=checkpoint_metadata(checkpoint);old=_contract(metadata['runtime_contract']);new=_contract(current_contract)
    sidecar=checkpoint.with_name(checkpoint.stem+'_manifest.json')
    if (file_sha(checkpoint)!=SOURCE_SHA or file_sha(sidecar)!=SOURCE_MANIFEST_SHA
            or {k:metadata.get(k) for k in COUNTERS}!=SOURCE_COUNTS
            or not metadata.get('rr_signed_contact_v7_migration') or source_num_envs(metadata)!=1
            or metadata.get('policy_contract')!=policy_contract(RR_CAPTURE_POLICY,observation_layout=RR_CAPTURE_OBSERVATION_LAYOUT)
            or new.get('experiment_id')!=EXPERIMENT or not reason.strip()
            or len(new.get('source_git_commit',''))!=40 or new.get('frozen_A_files')!=old.get('frozen_A_files')):
        raise ValueError('rear timing requires the registered intact front-validated v7 ancestor and explicit target runtime')
    for relative,expected in new['files'].items():
        if file_sha(root/relative)!=expected:raise ValueError('target runtime bytes changed: '+relative)
    source_configs={};target_configs={}
    for name,entry in new['selected_configuration'].items():
        path=root/entry['path']
        if file_sha(path)!=entry['sha256']:raise ValueError('target configuration binding changed')
        if name not in old['selected_configuration']:continue
        oldpath=old['selected_configuration'][name]['path']
        blob=_version_bytes(root,old,oldpath)
        source_configs[name]=yaml.safe_load(blob);target_configs[name]=yaml.safe_load(path.read_text(encoding='utf-8'))
    a,b=copy.deepcopy(source_configs['action_schema.json']),copy.deepcopy(target_configs['action_schema.json'])
    for value in (a,b):
        value.pop('config_source',None);value.pop('rear_task_assist',None)
    if a!=b:raise ValueError('rear timing cannot change Full12 channel order, masks or action semantics')
    for name in ('reward_config.yaml','quality_score.yaml'):
        if source_configs[name]!=target_configs[name]:raise ValueError('rear timing preserves reward/quality configuration')
    oldprofile,profile=source_configs['execution_profile.yaml'],target_configs['execution_profile.yaml']
    for key in ('physics_hz','decision_hz','episode_timeout_s','mapper_version','residual'):
        if oldprofile[key]!=profile[key]:raise ValueError('rear timing changed physics/caps/mapper: '+key)
    if (profile.get('rear_policy_timing_mode')!='rr_capture_before_rl_transfer_v1'
            or profile.get('rr_capture_assist_mode') is not None or profile.get('rr_capture_feedback_revision') is not None
            or profile.get('rr_capture_wheel_mode')!='off' or profile.get('nominal_geometry_advisory') is not None
            or profile.get('capture_assist_mode')!=oldprofile.get('capture_assist_mode')):
        raise ValueError('formal rear policy requires rear task assists OFF and preserved explicit FL assist')
    oldspec,spec=source_configs['stage_task_spec.yaml'],target_configs['stage_task_spec.yaml']
    for key in set(oldspec)|set(spec):
        if key=='nominal':continue
        if oldspec.get(key)!=spec.get(key):raise ValueError('unexpected physical task/reward acceptance change: '+key)
    if spec['nominal'].get('rear_policy_timing')!='rr_capture_before_rl_transfer_v1':
        raise ValueError('rear source timing opt-in missing')
    target=policy_contract(REAR_POLICY_TIMING_POLICY,observation_layout=REAR_POLICY_TIMING_OBSERVATION_LAYOUT)
    factor=dict(schema=SCHEMA,source_policy_contract=metadata['policy_contract'],target_policy_contract=target,
        counter_origin=SOURCE_COUNTS,source_role='explicit_front_and_RR_validated_ancestor_continuation',
        preserved_metadata_sha256={k:digest(metadata[k]) for k in preserved_keys(metadata)},
        actor_critic_mapping='old410_columns_exact_new9_zero',Adam_mapping='old_moments_steps_options_exact_new9_moment_columns_zero',
        same_mdp_claimed=False,physical_dynamics_changed=False,old_rollout_inherited=False,
        exploration_mapping_changed=True,learned_mu_log_sigma_heads_preserved=True,
        reward_potential_semantics_changed='new_RL_unload_credit_requires_current_RR_plus_front_support_until_RL_current_qualified_AIR;historical_RR_placed_is_not_current_support',
        Gaussian_equivalence_claimed=False,deterministic_same_numeric_prefix_preserved_up_to_float_roundoff=True,
        rear_task_assists_enabled=False,FL_capture_assist_enabled=True,
        added_policy_decisions=0,added_ppo_updates=0,added_optimizer_steps=0,added_auxiliary_updates=0)
    return dict(schema=SCHEMA,reason=reason.strip(),source_checkpoint=str(checkpoint),source_checkpoint_sha256=SOURCE_SHA,
        source_manifest_sha256=SOURCE_MANIFEST_SHA,source_contract_sha256=digest(old),
        source_runtime_content_sha256=old['runtime_content_sha256'],target_contract_sha256=digest(new),
        target_runtime_content_sha256=new['runtime_content_sha256'],target_git_commit=new['source_git_commit'],
        **{FACTOR_KEY:factor})

def validate_rear_policy_timing_migration(checkpoint,current_contract,plan_path,*,project_root=None):
    from .semantic_migration import file_sha
    path=Path(plan_path).resolve(strict=True);plan=json.loads(path.read_text(encoding='utf-8'))
    expected=build_rear_policy_timing_migration(checkpoint,current_contract,reason=plan.get('reason',''),project_root=project_root)
    if plan!=expected:raise ValueError('rear timing plan/source/runtime changed')
    return {**plan,'plan_path':str(path),'plan_sha256':file_sha(path)}

def zero_append_rear_policy_training_state(bundle):
    import torch
    result=copy.deepcopy(bundle);shapes=[]
    names=['mlp.0.weight','mlp.0.bias','mlp.2.weight','mlp.2.bias','mlp.4.weight','mlp.4.bias']
    for role in ('actor','critic'):
        state=result[role+'_state_dict']
        if list(state)!=names or tuple(state[names[0]].shape)!=(256,410):raise ValueError('exact Identity410 MLP required')
        shapes.extend(tuple(v.shape) for v in state.values())
        weight=state[names[0]];state[names[0]]=torch.cat((weight,weight.new_zeros(256,9)),dim=1)
    optimizer=result['optimizer_state_dict']
    if len(optimizer['param_groups'])!=1 or optimizer['param_groups'][0]['params']!=list(range(12)) or set(optimizer['state'])!=set(range(12)):
        raise ValueError('complete official actor-then-critic Adam required')
    for ident,shape in enumerate(shapes):
        state=optimizer['state'][ident]
        if set(state)!={'step','exp_avg','exp_avg_sq'} or state['step'].numel()!=1:raise ValueError('full Adam state required')
        for key in ('exp_avg','exp_avg_sq'):
            value=state[key]
            if tuple(value.shape)!=shape:raise ValueError('Adam moment mapping mismatch')
            if ident in (0,6):state[key]=torch.cat((value,value.new_zeros(256,9)),dim=1)
    return result

def rear_policy_timing_branch_counts(infos):
    result=dict(infos);origin=result['rear_policy_timing_branch']['counter_origin']
    if origin!=SOURCE_COUNTS or any(type(result.get(k)) is not int or result[k]<origin[k] for k in COUNTERS):
        raise ValueError('rear timing branch cannot borrow or reset counters')
    result['rear_policy_timing_branch_counts']={k:result[k]-origin[k] for k in COUNTERS}
    return result

def build_rear_policy_output_routing(metadata, contract, destination):
    """Bind this reviewed ancestor branch, never a historical RR410 output route."""
    if "front_preservation439_branch_identity" in metadata:
        from .semantic_front_preservation import build_front_preservation_output_routing
        return build_front_preservation_output_routing(metadata, contract, destination)
    from .semantic_rear_recapture_migration import BRANCH_NAME, SOURCE_SELECTION, MIGRATION as RECAPTURE
    destination = Path(destination).resolve()
    route = dict(schema='wlr50_clean.checkpoint_output_routing.v1',branch=BRANCH_NAME,
        output_root=str(destination),main_latest_pointer_promotion=False,
        source_selection=copy.deepcopy(SOURCE_SELECTION))
    if (destination.name != BRANCH_NAME or destination.parent.name != 'branches'
            or destination.parent.parent.name != 'ppo_'+EXPERIMENT or RECAPTURE not in metadata):
        raise ValueError('rear419 output route requires the reviewed recapture ancestor branch')
    inherited = metadata.get('checkpoint_output_routing')
    if inherited is not None:
        if inherited != route:
            raise ValueError('rear419 continuation must retain its exact output branch')
    elif ({k:metadata.get(k) for k in COUNTERS} != SOURCE_COUNTS
            or metadata.get('rear_policy_timing_branch_counts') != dict.fromkeys(COUNTERS,0)):
        raise ValueError('initial rear419 branch cannot borrow completed updates')
    return route


def validate_rear_policy_namespace(metadata,contract,output_root, *, checkpoint_output_routing=None):
    from .semantic_migration import digest
    from .semantic_policy_distribution import policy_contract
    if "front_preservation439_branch_identity" in metadata:
        from .semantic_front_preservation import validate_front_preservation_lineage
        return validate_front_preservation_lineage(metadata, contract, output_root,
            checkpoint_output_routing=checkpoint_output_routing)
    if 'front_retention439_runtime_identity' in metadata or 'front_retention439_auxiliary' in metadata:
        from .semantic_front_retention439 import validate_front_retention439_lineage
        validate_front_retention439_lineage(metadata,contract,output_root,
            checkpoint_output_routing=checkpoint_output_routing)
        return
    if metadata.get('rr_retention_reward_migration') is not None:
        from .semantic_rr_retention_migration import validate_rr_retention_lineage
        validate_rr_retention_lineage(metadata,contract,output_root,
            checkpoint_output_routing=checkpoint_output_routing)
        return
    if metadata.get('rear_owner_recovery_migration') is not None:
        from .semantic_rear_owner_migration import validate_rear_owner_lineage
        validate_rear_owner_lineage(metadata,contract,output_root,
            checkpoint_output_routing=checkpoint_output_routing)
        return
    if metadata.get('cooperative_prep_migration') is not None:
        from .semantic_cooperative_prep_migration import validate_cooperative_prep_lineage
        validate_cooperative_prep_lineage(metadata,contract,output_root,
            checkpoint_output_routing=checkpoint_output_routing)
        return
    if metadata.get('p02_progress_migration') is not None:
        from .semantic_p02_progress_migration import validate_p02_progress_lineage
        validate_p02_progress_lineage(metadata,contract,output_root,checkpoint_output_routing=checkpoint_output_routing)
        return
    receipt=metadata.get(MIGRATION,{})
    runtime_bound = receipt.get('target_contract_sha256') == digest(contract)
    if metadata.get('rear_live_swing_migration') is not None:
        from .semantic_rear_live_swing_migration import validate_rear_live_swing_lineage
        validate_rear_live_swing_lineage(metadata, contract)
        runtime_bound = True
    elif metadata.get('rear_recapture_migration') is not None:
        from .semantic_rear_recapture_migration import validate_rear_recapture_lineage
        validate_rear_recapture_lineage(metadata, contract, receipt)
        runtime_bound = True
    destination=Path(output_root).resolve()
    if checkpoint_output_routing is not None:
        expected_route=build_rear_policy_output_routing(metadata,contract,destination)
        if checkpoint_output_routing != expected_route:
            raise ValueError('rear419 explicit route differs from its reviewed branch')
    elif destination.name != 'ppo_'+EXPERIMENT or metadata.get('checkpoint_output_routing') is not None:
        raise ValueError('rear419 branch requires its explicit route and namespace')
    expected_policy=policy_contract(REAR_POLICY_TIMING_POLICY,observation_layout=REAR_POLICY_TIMING_OBSERVATION_LAYOUT)
    if (contract.get('experiment_id')!=EXPERIMENT
            or receipt.get('schema')!=SCHEMA or not runtime_bound
            or receipt.get('source_checkpoint_sha256')!=SOURCE_SHA
            or metadata.get('policy_contract') != expected_policy
            or metadata.get('rear_policy_timing_branch',{}).get('counter_origin')!=SOURCE_COUNTS):
        raise ValueError('rear timing continuation requires its explicit new namespace and immutable ancestor receipt')
    computed=rear_policy_timing_branch_counts(metadata)['rear_policy_timing_branch_counts']
    if metadata.get('rear_policy_timing_branch_counts') != computed:
        raise ValueError('rear419 recorded branch counts differ from its immutable counter origin')

def load_rear_policy_timing_migration(runner,checkpoint,*,contract,seed,record):
    import torch
    from .semantic_migration import checkpoint_metadata,continuation_topology,digest
    from .semantic_training import construct_semantic_runner,load_semantic_checkpoint,state_hash,_normalizers,_runner_policy_contract,parameter_hash
    from .semantic_rr_capture_migration import _shape_env
    from .rl_library_wrapper import optimizer_learning_rate,restore_training_rng_state
    verified=validate_rear_policy_timing_migration(checkpoint,contract,record['plan_path'])
    if verified!=dict(record):raise RuntimeError('rear timing plan changed after preflight')
    m=checkpoint_metadata(Path(checkpoint));device=str(runner.device)
    if (seed!=m['seed'] or _runner_policy_contract(runner)!=verified[FACTOR_KEY]['target_policy_contract']
            or tuple(runner.alg.storage.actions.shape)!=(128,1,12)
            or any(tuple(runner.alg.storage.observations[k].shape)!=(128,1,419) for k in ('policy','critic'))
            or runner.alg.storage.step!=0 or runner.alg.transition.actions is not None):
        raise RuntimeError('rear timing requires original seed and fresh exact419 storage')
    source,_=construct_semantic_runner(_shape_env(410,device),seed=seed,device=device,policy_version=RR_CAPTURE_POLICY,
        observation_layout=RR_CAPTURE_OBSERVATION_LAYOUT,initialize_actor=False)
    expected_runner=copy.deepcopy(source._semantic_runner_config)
    expected_runner['actor']['class_name']=verified[FACTOR_KEY]['target_policy_contract']['actor_class']
    expected_runner['actor']['observation_layout']=REAR_POLICY_TIMING_OBSERVATION_LAYOUT
    if runner._semantic_runner_config!=expected_runner:
        raise RuntimeError('rear timing cannot change PPO hyperparameters beyond the declared actor/layout')
    infos=load_semantic_checkpoint(source,Path(checkpoint),contract=m['runtime_contract'],seed=seed)
    mapped=zero_append_rear_policy_training_state(torch.load(checkpoint,map_location=device,weights_only=False))
    for role in ('actor','critic'):
        model=getattr(runner.alg,role)
        if type(model.obs_normalizer) is not torch.nn.Identity:raise RuntimeError('Identity required')
        model.load_state_dict(mapped[role+'_state_dict'],strict=True)
    runner.alg.optimizer.load_state_dict(mapped['optimizer_state_dict']);runner.alg.learning_rate=m['optimizer_learning_rate']
    runner.current_learning_iteration=mapped['iter']
    if optimizer_learning_rate(runner)!=m['optimizer_learning_rate'] or state_hash(_normalizers(runner))!=m['normalizer_state_sha256']:
        raise RuntimeError('rear timing altered LR or Identity')
    for key,wanted in verified[FACTOR_KEY]['preserved_metadata_sha256'].items():
        if digest(infos[key])!=wanted:raise RuntimeError('source lineage changed: '+key)
    restore_training_rng_state(m['training_rng_state'],expected_seed=seed)
    sampling='P01_full_task_only_initial_version'
    return {**infos,'runtime_contract':dict(contract),'runner_config':copy.deepcopy(runner._semantic_runner_config),
        'policy_contract':_runner_policy_contract(runner),'sampling':sampling,'stage':'full_episode',
        'implemented_reset_sampling':sampling,'phase_suffix_curriculum_implemented':False,
        'execution_topology':continuation_topology(sampling,None,observation_layout=REAR_POLICY_TIMING_OBSERVATION_LAYOUT),
        'curriculum_epoch':{'reset_sampling':sampling,'prefix_request':None},
        'actor_parameter_sha256':parameter_hash(runner.alg.actor),'critic_parameter_sha256':parameter_hash(runner.alg.critic),
        'optimizer_state_sha256':state_hash(runner.alg.optimizer.state_dict()),'resume_migration':verified,MIGRATION:verified,
        'rear_policy_timing_branch':{'schema':SCHEMA,'counter_origin':SOURCE_COUNTS,'source_checkpoint_sha256':SOURCE_SHA,
            'source_role':verified[FACTOR_KEY]['source_role'],'migration_added_updates':0},
        'old_rollout_inherited':False,'physical_env_state_saved':False,'resume_physics':'fresh_legal_P01_reset'}

def publish_rear_policy_timing_checkpoint(checkpoint,current_contract,plan_path,output_checkpoint):
    from .semantic_migration import checkpoint_metadata
    from .semantic_training import construct_semantic_runner,load_semantic_checkpoint,save_semantic_checkpoint,state_hash,parameter_hash
    from .semantic_rr_capture_migration import _shape_env
    m=checkpoint_metadata(Path(checkpoint));record=validate_rear_policy_timing_migration(checkpoint,current_contract,plan_path)
    device=m['runner_config']['device']
    def make():return construct_semantic_runner(_shape_env(419,device),seed=m['seed'],device=device,
        policy_version=REAR_POLICY_TIMING_POLICY,observation_layout=REAR_POLICY_TIMING_OBSERVATION_LAYOUT,initialize_actor=False)[0]
    runner=make();infos=load_rear_policy_timing_migration(runner,checkpoint,contract=current_contract,seed=m['seed'],record=record)
    path,manifest=save_semantic_checkpoint(runner,Path(output_checkpoint),infos)
    fresh=make();loaded=load_semantic_checkpoint(fresh,path,contract=current_contract,seed=m['seed'])
    for key in preserved_keys(m):
        if loaded[key]!=m[key]:raise RuntimeError('publication changed inherited metadata: '+key)
    for role in ('actor','critic'):
        if parameter_hash(getattr(fresh.alg,role))!=parameter_hash(getattr(runner.alg,role)):raise RuntimeError('fresh reload weights differ')
    if state_hash(fresh.alg.optimizer.state_dict())!=state_hash(runner.alg.optimizer.state_dict()):raise RuntimeError('fresh reload Adam differs')
    return dict(checkpoint=str(path),manifest=str(manifest),save_load_round_trip=True,
        **{k:loaded[k] for k in COUNTERS},rear_policy_timing_branch_counts=loaded['rear_policy_timing_branch_counts'],
        added_policy_decisions=0,added_ppo_updates=0,added_optimizer_steps=0,added_auxiliary_updates=0)
