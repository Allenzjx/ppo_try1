"""Explicit 389->RR-capture append: preserve learned state, never reset a head."""
from __future__ import annotations
import copy
import json
import math
from pathlib import Path
import subprocess
from .semantic_p05_capture_profile import P05_CAPTURE_POLICY, P05_CAPTURE_OBSERVATION_LAYOUT
from .semantic_rr_capture_profile import (RR_CAPTURE_POLICY, RR_CAPTURE_OBSERVATION_LAYOUT,
    RR_CAPTURE_OBSERVATION_DIM, RR_ASSIST_START, RR_TASK_START, RR_ASSIST_GROUP, RR_TASK_GROUP, RR_TASK_FIELDS)

SCHEMA='wlr50_clean.rr_capture_transfer_append.v1'
FACTOR_KEY='rr_capture_transfer_factor'
SOURCE_EXPERIMENT='p05_hip_only_continuation_v1'
TARGET_EXPERIMENT='rr_capture_then_rl_transfer_v1'
COUNTERS=('global_policy_decisions','ppo_updates','optimizer_steps')
PRIOR_BRANCHES=('p05_capture_assist','capture_feedback_semantics','rr_postcross_workspace',
               'rr_receiver_retirement_v2','p05_preedge_approach_recovery')
REVIEWABLE_CODE=frozenset('src/wlr50_clean/ppo/'+name+'.py' for name in (
    'semantic_rr_capture_profile','semantic_rr_capture_actor','semantic_rr_capture_migration',
    'semantic_rr_capture_assist','semantic_rr_capture_context','semantic_observation','semantic_policy_distribution',
    'semantic_training','semantic_migration','semantic_checkpoint_prefix_policy','semantic_checkpoint_prefix',
    'semantic_cli','semantic_video_cli','semantic_video','semantic_supervisor','semantic_residual_adapter',
    'semantic_backend','semantic_env','actuator_target_effect')) | {'scripts/run_semantic_ppo.ps1','scripts/run_semantic_video.ps1'}


def preserved_keys(metadata):
    return tuple(sorted(set(COUNTERS+('seed','optimizer_learning_rate','normalization','normalizer_state_sha256','training_rng_state')) |
        {k for k in metadata if k!='resume_migration' and k.endswith(('_branch','_branch_counts','_migration'))}))


def build_rr_capture_migration(checkpoint,current_contract,*,reason,reviewed_code_sha256,project_root=None):
    import yaml
    from .semantic_migration import PROJECT_ROOT,_contract,checkpoint_metadata,digest,file_sha,source_num_envs,_version_bytes
    from .semantic_policy_distribution import policy_contract,policy_version_from_metadata,CONFIG_NAMES
    from .semantic_observation import load_semantic_observation_schema
    root=Path(project_root or PROJECT_ROOT).resolve();checkpoint=Path(checkpoint).resolve(strict=True)
    m=checkpoint_metadata(checkpoint);old,new=_contract(m['runtime_contract']),_contract(current_contract)
    if (old.get('experiment_id')!=SOURCE_EXPERIMENT or new.get('experiment_id')!=TARGET_EXPERIMENT
            or old.get('semantic_version')!='v3' or new.get('semantic_version')!='v3'
            or policy_version_from_metadata(m)!=P05_CAPTURE_POLICY or source_num_envs(m)!=1
            or m['policy_contract']!=policy_contract(P05_CAPTURE_POLICY,observation_layout=P05_CAPTURE_OBSERVATION_LAYOUT)):
        raise ValueError('RR append requires the complete current P05 389/v3/N1 source contract')
    if not isinstance(reason,str) or not reason.strip():raise ValueError('explicit RR migration reason required')
    variable={'source_git_commit','runtime_content_sha256','files','selected_configuration','experiment_id'}
    if {k:v for k,v in old.items() if k not in variable}!={k:v for k,v in new.items() if k not in variable}:
        raise ValueError('RR append cannot change physics, rates, budgets, stack or horizon')
    if 'rr_capture_transfer_branch' in m or 'rr_capture_transfer_migration' in m:
        raise ValueError('RR append cannot be repeated or overwrite existing lineage')
    for key in COUNTERS:
        if type(m.get(key)) is not int or m[key]<0:raise ValueError('actual source counters required')
    for name in PRIOR_BRANCHES:
        origin=m.get(name+'_branch',{}).get('counter_origin',{})
        if (set(origin)!=set(COUNTERS) or not isinstance(m.get(name+'_migration'),dict)
                or any(type(origin[k]) is not int or not 0<=origin[k]<=m[k] for k in COUNTERS)
                or m.get(name+'_branch_counts')!={k:m[k]-origin[k] for k in COUNTERS}):
            raise ValueError('all five prior origins, counts and receipts must remain intact')
    aux=m['rr_postcross_workspace_branch'].get('front_rehearsal_auxiliary')
    if not isinstance(aux,dict) or len(aux.get('events',()))<4 or not isinstance(m['task_conditioned_hip_wheel_branch'].get('auxiliary_mean_learning'),dict):
        raise ValueError('all existing real AUX events and historical AUX ledger are required')
    rate=m['optimizer_learning_rate']
    if type(rate) not in (int,float) or not math.isfinite(rate) or rate<=0:raise ValueError('actual positive Adam LR required')
    head=subprocess.run(['git','-C',str(root),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
    if new['source_git_commit']!=head or head==old['source_git_commit']:
        raise ValueError('target must bind the actual new committed runtime HEAD')
    dirty=subprocess.run(['git','-C',str(root),'status','--porcelain=v1','--untracked-files=all','--',
        'src/wlr50_clean','scripts','configs','artifacts/ppo_phase_v1_start','pyproject.toml'],
        capture_output=True,text=True,check=True).stdout.strip()
    if dirty:raise ValueError('RR target runtime must be clean and committed before migration')
    if set(old['files'])-set(new['files']):raise ValueError('RR migration cannot remove old runtime files')
    delta=sorted(p for p in new['files'] if old['files'].get(p)!=new['files'][p])
    configs={f'configs/ppo_{TARGET_EXPERIMENT}/{name}' for name in CONFIG_NAMES}
    code=set(delta)-configs
    required={f'src/wlr50_clean/ppo/semantic_rr_capture_{name}.py' for name in ('actor','profile','migration','assist','context')}
    if not required<=code or not code<=REVIEWABLE_CODE or dict(reviewed_code_sha256)!={p:new['files'][p] for p in code}:
        raise ValueError('RR append requires its exact bounded reviewed implementation inventory')
    if any(file_sha(root/p)!=v for p,v in new['files'].items()):raise ValueError('target runtime file bytes differ')
    parsed={}
    for side,contract,namespace in (('source',old,SOURCE_EXPERIMENT),('target',new,TARGET_EXPERIMENT)):
        selected=contract.get('selected_configuration',{})
        if set(selected)!=CONFIG_NAMES:raise ValueError('six exact selected configurations required')
        for name,binding in selected.items():
            path=f'configs/ppo_{namespace}/{name}'
            if binding!={'path':path,'sha256':contract['files'].get(path)}:raise ValueError('configuration binding mismatch')
            raw=_version_bytes(root,old,path) if side=='source' else (root/path).read_bytes()
            parsed[side,name]=(raw,json.loads(raw) if name.endswith('.json') else yaml.safe_load(raw))
    for name in ('action_schema.json','quality_score.yaml','reward_config.yaml'):
        if parsed['source',name][0]!=parsed['target',name][0]:raise ValueError('RR first boundary preserves exact bytes: '+name)
    before=parsed['source','execution_profile.yaml'][1];after=parsed['target','execution_profile.yaml'][1]
    if after!={**before,'revision':'rr_capture_then_rl_transfer_v1_hip_only_direction_check',
               'rr_capture_assist_mode':'rr_hip_only_capture_v1','rr_capture_wheel_mode':'off'}:
        raise ValueError('RR execution may only add explicit hip-only assist and wheel-off mode')
    before=parsed['source','stage_task_spec.yaml'][1];after=parsed['target','stage_task_spec.yaml'][1]
    if after!={**before,'rr_capture_continuation_semantics':TARGET_EXPERIMENT}:
        raise ValueError('RR task spec may only add its explicit first-version continuation')
    before=parsed['source','observation_schema.json'][1];after=parsed['target','observation_schema.json'][1]
    schema=load_semantic_observation_schema(root/new['selected_configuration']['observation_schema.json']['path'])
    if (schema.observation_layout!=RR_CAPTURE_OBSERVATION_LAYOUT or schema.dimension!=RR_CAPTURE_OBSERVATION_DIM
            or after['feature_groups'][:-2]!=before['feature_groups']
            or {k:v for k,v in after.items() if k not in ('revision','rr_capture_features_version','feature_groups')}
            !={k:v for k,v in before.items() if k not in ('revision','feature_groups')}):
        raise ValueError('RR observation must preserve every old389 codec field and append explicit14+7')
    factor={'schema':SCHEMA,'source_policy_contract':m['policy_contract'],
        'target_policy_contract':policy_contract(RR_CAPTURE_POLICY,observation_layout=RR_CAPTURE_OBSERVATION_LAYOUT),
        'source_observation_dimension':389,'target_observation_dimension':RR_CAPTURE_OBSERVATION_DIM,
        'source_observation_layout':P05_CAPTURE_OBSERVATION_LAYOUT,'target_observation_layout':RR_CAPTURE_OBSERVATION_LAYOUT,
        'appended_columns':[389,RR_CAPTURE_OBSERVATION_DIM],'first_layer_parameter':'mlp.0.weight',
        'parameter_mapping':'copy_all_old389_actor_critic_columns_and_all_other_parameters_zero_append_only',
        'optimizer_mapping':'preserve_all_Adam_steps_groups_LR_old_moments_zero_append_only_first_layer_moments',
        'reviewed_code_sha256':dict(reviewed_code_sha256),'source_effective_learning_rate':rate,
        'counter_origin':{k:m[k] for k in COUNTERS},'preserved_metadata_sha256':{k:digest(m[k]) for k in preserved_keys(m)},
        'source_resume_migration':copy.deepcopy(m.get('resume_migration')),
        'normalizers':'preserve_Identity','training_rng':'restore_exact_source_state','same_mdp_claimed':False,
        'control_semantics_changed':True,'physical_dynamics_changed':False,'caps_changed':False,'sigma_kernel_changed':False,
        'reward_config_changed':False,'same_input_old_prefix_function_preserved':True,
        'same_physical_trajectory_equivalence_claimed':False,'old_rollout_inherited':False,'physical_state_inherited':False,
        'added_policy_decisions':0,'added_ppo_updates':0,'added_optimizer_steps':0,'added_auxiliary_updates':0}
    return {'schema':SCHEMA,'reason':reason.strip(),'source_checkpoint':str(checkpoint),
        'source_checkpoint_sha256':file_sha(checkpoint),'source_manifest_sha256':file_sha(checkpoint.with_name(checkpoint.stem+'_manifest.json')),
        'source_contract_sha256':digest(old),'target_contract_sha256':digest(new),
        'source_git_commit':old['source_git_commit'],'target_git_commit':new['source_git_commit'],
        'source_runtime_content_sha256':old['runtime_content_sha256'],'target_runtime_content_sha256':new['runtime_content_sha256'],
        'allowed_changed_files':delta,'changed_file_hashes':{p:{'before':old['files'].get(p),'after':new['files'][p]} for p in delta},
        'observation_dimension':RR_CAPTURE_OBSERVATION_DIM,'action_dimension':12,'discard_old_rollout_storage':True,FACTOR_KEY:factor}


def validate_rr_capture_migration(checkpoint,current_contract,plan_path,*,project_root=None):
    from .semantic_migration import file_sha
    path=Path(plan_path).resolve(strict=True);supplied=json.loads(path.read_text(encoding='utf-8'))
    expected=build_rr_capture_migration(checkpoint,current_contract,reason=supplied.get('reason'),
        reviewed_code_sha256=supplied.get(FACTOR_KEY,{}).get('reviewed_code_sha256',{}),project_root=project_root)
    if supplied!=expected:raise ValueError('RR append plan differs from immutable source and target')
    return {**expected,'plan_path':str(path),'plan_sha256':file_sha(path)}


def zero_append_rr_training_state(bundle):
    """Pure mapping, strict official actor/critic parameter IDs; never mutate source."""
    import torch
    result=copy.deepcopy(bundle);shapes=[]
    names=['mlp.0.weight','mlp.0.bias','mlp.2.weight','mlp.2.bias','mlp.4.weight','mlp.4.bias']
    for role in ('actor','critic'):
        state=result[role+'_state_dict']
        if list(state)!=names or tuple(state[names[0]].shape)!=(256,389):raise ValueError('exact Identity389 MLP required')
        shapes.extend(tuple(v.shape) for v in state.values());w=state[names[0]]
        state[names[0]]=torch.cat((w,w.new_zeros(256,RR_CAPTURE_OBSERVATION_DIM-389)),dim=1)
    opt=result['optimizer_state_dict']
    if len(opt['param_groups'])!=1 or opt['param_groups'][0]['params']!=list(range(12)) or set(opt['state'])!=set(range(12)):
        raise ValueError('official full Adam actor-then-critic IDs required')
    for ident,shape in enumerate(shapes):
        state=opt['state'][ident]
        if set(state)!={'step','exp_avg','exp_avg_sq'} or state['step'].numel()!=1:raise ValueError('complete Adam state required')
        for key in ('exp_avg','exp_avg_sq'):
            value=state[key]
            if tuple(value.shape)!=shape:raise ValueError('Adam moment mapping mismatch')
            if ident in (0,6):state[key]=torch.cat((value,value.new_zeros(256,RR_CAPTURE_OBSERVATION_DIM-389)),dim=1)
    return result


def rr_capture_branch_counts(infos):
    result=dict(infos)
    if 'rr_capture_transfer_branch' in result:
        origin=result['rr_capture_transfer_branch']['counter_origin']
        if set(origin)!=set(COUNTERS) or any(type(origin[k]) is not int or origin[k]<0 or type(result[k]) is not int or result[k]<origin[k] for k in COUNTERS):
            raise RuntimeError('invalid RR branch origin')
        result['rr_capture_transfer_branch_counts']={k:result[k]-origin[k] for k in COUNTERS}
    return result


def _shape_env(dimension,device):
    from .semantic_p05_capture_migration import _ObservationOnlyEnv
    class ShapeEnv(_ObservationOnlyEnv):
        def __init__(self):
            super().__init__(dimension);self.device=device;self.episode_length_buf=self.episode_length_buf.to(device)
        def get_observations(self):return super().get_observations().to(self.device)
    return ShapeEnv()


def load_rr_capture_migration(runner,checkpoint,*,contract,seed,record):
    import torch
    from .semantic_migration import checkpoint_metadata,continuation_topology,digest
    from .semantic_training import construct_semantic_runner,load_semantic_checkpoint,state_hash,_normalizers,_runner_policy_contract,parameter_hash
    from .rl_library_wrapper import optimizer_learning_rate,restore_training_rng_state
    verified=validate_rr_capture_migration(checkpoint,contract,record['plan_path']);factor=verified[FACTOR_KEY]
    if verified!=dict(record):raise RuntimeError('RR migration changed since preflight')
    m=checkpoint_metadata(Path(checkpoint));device=str(runner.device)
    if (seed!=m['seed'] or _runner_policy_contract(runner)!=factor['target_policy_contract']
            or tuple(runner.alg.storage.actions.shape)!=(128,1,12)
            or any(tuple(runner.alg.storage.observations[k].shape)!=(128,1,RR_CAPTURE_OBSERVATION_DIM) for k in ('policy','critic'))
            or runner.alg.storage.step!=0 or runner.alg.transition.actions is not None):
        raise RuntimeError('RR append requires original seed and empty exact target storage')
    source,_=construct_semantic_runner(_shape_env(389,device),seed=seed,device=device,policy_version=P05_CAPTURE_POLICY,
        observation_layout=P05_CAPTURE_OBSERVATION_LAYOUT,initialize_actor=False)
    infos=load_semantic_checkpoint(source,Path(checkpoint),contract=m['runtime_contract'],seed=seed)
    if any(digest(infos[k])!=v for k,v in factor['preserved_metadata_sha256'].items()):raise RuntimeError('RR source metadata changed')
    mapped=zero_append_rr_training_state(torch.load(checkpoint,map_location=device,weights_only=False))
    for role in ('actor','critic'):
        model=getattr(runner.alg,role)
        if type(model.obs_normalizer) is not torch.nn.Identity:raise RuntimeError('RR append must preserve Identity')
        model.load_state_dict(mapped[role+'_state_dict'],strict=True)
    runner.alg.optimizer.load_state_dict(mapped['optimizer_state_dict']);runner.alg.learning_rate=m['optimizer_learning_rate']
    runner.current_learning_iteration=mapped['iter']
    if optimizer_learning_rate(runner)!=m['optimizer_learning_rate'] or state_hash(_normalizers(runner))!=m['normalizer_state_sha256']:
        raise RuntimeError('RR append changed LR or normalizer')
    restore_training_rng_state(m['training_rng_state'],expected_seed=seed)
    sampling='P01_full_task_only_initial_version'
    return {**infos,'runtime_contract':dict(contract),'runner_config':copy.deepcopy(runner._semantic_runner_config),
        'policy_contract':_runner_policy_contract(runner),'sampling':sampling,'stage':'full_episode',
        'implemented_reset_sampling':sampling,'phase_suffix_curriculum_implemented':False,
        'execution_topology':continuation_topology(sampling,None,observation_layout=RR_CAPTURE_OBSERVATION_LAYOUT),
        'curriculum_epoch':{'reset_sampling':sampling,'prefix_request':None},
        'source_training_context':{k:copy.deepcopy(m[k]) for k in ('stage','sampling','implemented_reset_sampling','phase_suffix_curriculum_implemented','execution_topology','curriculum_epoch') if k in m},
        'actor_parameter_sha256':parameter_hash(runner.alg.actor),'critic_parameter_sha256':parameter_hash(runner.alg.critic),
        'optimizer_state_sha256':state_hash(runner.alg.optimizer.state_dict()),'resume_migration':verified,
        'rr_capture_transfer_migration':verified,'rr_capture_transfer_branch':{'schema':SCHEMA,'counter_origin':factor['counter_origin'],
            'source_checkpoint_sha256':verified['source_checkpoint_sha256'],'migration_added_updates':0},
        'old_rollout_inherited':False,'physical_env_state_saved':False,'resume_physics':'fresh_legal_P01_reset'}


def publish_rr_capture_checkpoint(checkpoint,current_contract,plan_path,output_checkpoint):
    """Official source-device save/fresh reload, no optimizer/physics or pointer."""
    from .semantic_migration import checkpoint_metadata
    from .semantic_training import construct_semantic_runner,load_semantic_checkpoint,save_semantic_checkpoint,state_hash,parameter_hash
    m=checkpoint_metadata(Path(checkpoint));verified=validate_rr_capture_migration(checkpoint,current_contract,plan_path)
    device=m['runner_config']['device']
    def make():return construct_semantic_runner(_shape_env(RR_CAPTURE_OBSERVATION_DIM,device),seed=m['seed'],device=device,
        policy_version=RR_CAPTURE_POLICY,observation_layout=RR_CAPTURE_OBSERVATION_LAYOUT,initialize_actor=False)[0]
    runner=make();infos=load_rr_capture_migration(runner,checkpoint,contract=current_contract,seed=m['seed'],record=verified)
    path,manifest=save_semantic_checkpoint(runner,Path(output_checkpoint),infos)
    fresh=make();loaded=load_semantic_checkpoint(fresh,path,contract=current_contract,seed=m['seed'])
    for key in preserved_keys(m):
        if loaded[key]!=m[key]:raise RuntimeError('RR publication changed preserved metadata: '+key)
    if (parameter_hash(fresh.alg.actor)!=parameter_hash(runner.alg.actor) or parameter_hash(fresh.alg.critic)!=parameter_hash(runner.alg.critic)
            or state_hash(fresh.alg.optimizer.state_dict())!=state_hash(runner.alg.optimizer.state_dict())
            or loaded['runner_config']!=runner._semantic_runner_config):raise RuntimeError('RR fresh reload differs')
    return {'checkpoint':str(path),'manifest':str(manifest),'save_load_round_trip':True,**{k:loaded[k] for k in COUNTERS},
        'observation_dimension':RR_CAPTURE_OBSERVATION_DIM,'rr_capture_transfer_branch_counts':loaded['rr_capture_transfer_branch_counts'],
        'migration_added_policy_decisions':0,'migration_added_ppo_updates':0,'migration_added_optimizer_steps':0,'migration_added_auxiliary_updates':0}
