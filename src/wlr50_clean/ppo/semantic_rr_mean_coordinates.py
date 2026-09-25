"""Optional cold in-memory coordinates only. No imports of Torch at module load.

The caller owns strict checkpoint/provenance loading, versioned runtime/config,
empty rollout publication and full RNG restore around its model construction.
This helper neither loads files nor constructs models/optimizers nor publishes.
"""
from __future__ import annotations
import copy
import math
import hashlib
import json

GAIN_KEY='local_mean_coordinate_gain_full12'
IDENTITY=(1.,)*12
RR10=(1.,)*6+(10.,10.)+(1.,)*4
SOURCE_HEAD='0ff03eafeeb75ba8505a99478cea2a6243cf93f5'
CHANGED_RUNTIME_FILES=frozenset({
    'src/wlr50_clean/ppo/semantic_rr_capture_local.py',
    'src/wlr50_clean/ppo/semantic_rr_capture_local_actor.py',
    'src/wlr50_clean/ppo/semantic_rr_capture_local_aux.py',
    'configs/ppo_rr_capture_first_cp225280_v1/local_training.json',
})
ADDED_RUNTIME_FILE='src/wlr50_clean/ppo/semantic_rr_mean_coordinates.py'


def checked_gain(value):
    if (not isinstance(value,(tuple,list)) or len(value)!=12
            or any(type(x) not in (int,float) or not math.isfinite(x) for x in value)
            or tuple(value) not in (IDENTITY,RR10)):
        raise ValueError('only identity or explicit RR6/7 gain10 coordinates')
    return tuple(float(x) for x in value)


def configuration_with_gain(configuration,gain):
    """Return only the actor-option change; caller serializes runtime same value."""
    result=copy.deepcopy(configuration)
    if not isinstance(result.get('actor'),dict):
        raise ValueError('explicit actor runner configuration required')
    result['actor'][GAIN_KEY]=list(checked_gain(gain))
    return result


def validate_coordinate_runtime_change(old,new):
    """Only this declared0ff->gain10 change; no normal resume bypass."""
    def valid_hash(value,length=64):
        return isinstance(value,str) and len(value)==length and all(c in '0123456789abcdef' for c in value)
    if (old.get('source_git_commit')!=SOURCE_HEAD or not valid_hash(new.get('source_git_commit'),40)
            or new['source_git_commit']==SOURCE_HEAD):raise ValueError('exact0ff source/new committed HEAD required')
    allowed={'source_git_commit','runtime_content_sha256','files','local_contract','selected_configuration'}
    if set(old)!=set(new) or any(old[k]!=new[k] for k in old if k not in allowed):
        raise ValueError('protected runtime/library/physics/history fields changed')
    old_cfg,new_cfg=copy.deepcopy(old['local_contract']),copy.deepcopy(new['local_contract'])
    if GAIN_KEY in old_cfg or checked_gain(new_cfg.pop(GAIN_KEY,None))!=RR10 or old_cfg!=new_cfg:
        raise ValueError('local config may only add the explicit RR10 coordinates')
    if (old_cfg.get('observation_dimension')!=448 or old_cfg.get('rear_task_assist') is not False
            or old_cfg.get('source_tracking_owner_revision')!='pending_source_tracking_inheritance_v1'):
        raise ValueError('unchanged448 no-rear-assist task required')
    for value in (old,new):
        files=value['files']
        if not files or any(not valid_hash(v) for v in files.values()):raise ValueError('invalid file inventory')
        digest=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if value['runtime_content_sha256']!=digest:raise ValueError('runtime inventory SHA mismatch')
        for name,item in value['selected_configuration'].items():
            if item['path']!='configs/ppo_rr_capture_first_cp225280_v1/'+name or files.get(item['path'])!=item['sha256']:
                raise ValueError('selected config missing exact runtime binding')
    old_files,new_files=old['files'],new['files']
    changed={p for p in old_files.keys()&new_files.keys() if old_files[p]!=new_files[p]}
    if set(old_files)-set(new_files) or set(new_files)-set(old_files)!={ADDED_RUNTIME_FILE} or changed!=CHANGED_RUNTIME_FILES:
        raise ValueError('unapproved runtime source difference')
    if set(old['selected_configuration'])!=set(new['selected_configuration']):raise ValueError('configuration set changed')
    if any(old['selected_configuration'][k]!=new['selected_configuration'][k]
           for k in old['selected_configuration'] if k!='local_training.json'):
        raise ValueError('protected selected control configuration changed')
    return {p:{'before':old_files.get(p),'after':new_files[p]} for p in sorted(changed|{ADDED_RUNTIME_FILE})}


def assert_coordinate_binding(runner,runtime,metadata=None):
    """Strict read/save binding; missing gain means identity only in old0ff."""
    gain=runtime['local_contract'].get(GAIN_KEY)
    old_source=runtime.get('source_git_commit')==SOURCE_HEAD
    if gain is None:
        if not old_source:raise ValueError('new runtime must explicitly serialize mean coordinates')
        gain=IDENTITY
    gain=checked_gain(gain)
    if old_source and gain!=IDENTITY:raise ValueError('old0ff cannot advertise gain10')
    if (checked_gain(getattr(runner.alg.actor,GAIN_KEY,None))!=gain
            or checked_gain(runner.local_configuration['actor'].get(GAIN_KEY))!=gain):
        raise ValueError('runtime/constructed actor/runner config coordinate mismatch')
    if metadata is not None:
        configuration=copy.deepcopy(metadata['runner_config'])
        if GAIN_KEY not in configuration['actor']:
            if not old_source:raise ValueError('new saved config omitted coordinates')
            configuration=configuration_with_gain(configuration,IDENTITY)
        if configuration!=runner.local_configuration:raise ValueError('saved/constructed runner configuration differs')
        history=metadata.get('local_mean_coordinate_migrations',[])
    else:history=getattr(runner,'local_mean_coordinate_migrations',[])
    if not isinstance(history,list):raise ValueError('coordinate lineage must be a list')
    if gain==RR10:
        if len(history)!=1:raise ValueError('gain10 requires its once-only coordinate migration receipt')
        event=history[0]
        if (event.get('schema')!='wlr50_clean.rr_mean_coordinate_publication.v1'
                or event.get('source_head')!=SOURCE_HEAD
                or event.get('destination_head')!=runtime['source_git_commit']
                or event.get('source_gain')!=list(IDENTITY) or event.get('target_gain')!=list(RR10)
                or event.get('new_PPO_updates')!=0 or event.get('new_AUX_steps')!=0):
            raise ValueError('coordinate lineage does not match this gain/runtime')
    return gain


def _optimizer_bindings(actor,critic,optimizer):
    named=[('actor.'+n,p) for n,p in actor.named_parameters()]
    named += [('critic.'+n,p) for n,p in critic.named_parameters()]
    by_id={id(p):(n,p) for n,p in named}
    if len(by_id)!=len(named):raise ValueError('shared actor/critic parameters unsupported')
    serialized=optimizer.state_dict()
    if len(serialized['param_groups'])!=len(optimizer.param_groups):raise ValueError('optimizer groups mismatch')
    result={};seen=set()
    for group,saved in zip(optimizer.param_groups,serialized['param_groups']):
        if len(group['params'])!=len(saved['params']):raise ValueError('optimizer members mismatch')
        for param,pid in zip(group['params'],saved['params']):
            if id(param) not in by_id:raise ValueError('unmapped optimizer Parameter')
            name,_=by_id[id(param)]
            if name in result or pid in seen or not param.requires_grad or name.startswith('actor.frozen_prior.'):
                raise ValueError('frozen or duplicate optimizer parameter')
            result[name]={'parameter':param,'serialized_id':pid}
            seen.add(pid)
    if set(result)!={n for n,p in named if p.requires_grad} or not set(serialized['state']).issubset(seen):
        raise ValueError('optimizer must map actual complete trainable actor/critic identities')
    return result,serialized


def migrate_rr_mean_coordinates(runner,reference_observations,*,source_runtime_gain,
                                target_runtime_gain,isaac_stopped=False):
    """One-way identity->RR10 on an unpublished, strictly loaded complete runner.

Both runtime gains are explicit: for an old runtime without the option the cold
publisher supplies IDENTITY and records this interpretation in its receipt.
At least one active and one inactive448 input are required for function checks.
Mutates existing Parameter/moment objects, actor gain and runner.local_configuration.
Return status must be PASS before caller publishes. A failure is NOT permission
to use the candidate: discard this unpublished runner, retain the saved parent.
"""
    if not isaac_stopped:raise ValueError('explicit Isaac exit required before Torch')
    source,target=checked_gain(source_runtime_gain),checked_gain(target_runtime_gain)
    if source!=IDENTITY or target!=RR10:raise ValueError('only once-only identity->RR10 migration')
    import torch
    from wlr50_clean.ppo.semantic_training import state_hash
    from wlr50_clean.ppo.rl_library_wrapper import capture_training_rng_state
    actor,critic,optimizer=runner.alg.actor,runner.alg.critic,runner.alg.optimizer
    if (type(optimizer) is not torch.optim.Adam or runner.alg.storage.step!=0
            or runner.alg.transition.actions is not None):
        raise ValueError('actual Adam and complete update/empty rollout required')
    if actor.observation_layout!='role439_rr_capture_local_v2' or getattr(actor,'legacy447_migration_only',False):
        raise ValueError('only formal same448 local actor supported')
    if checked_gain(getattr(actor,GAIN_KEY,None))!=source:
        raise ValueError('actor lacks explicit patched source coordinate contract')
    configuration=runner.local_configuration
    if checked_gain(configuration['actor'].get(GAIN_KEY,IDENTITY))!=source:
        raise ValueError('saved runner configuration differs from loaded actor gain')
    if any(g.get('weight_decay',0)!=0 for g in optimizer.param_groups):
        raise ValueError('nonzero weight decay needs a different coordinate derivation')
    if any(p.grad is not None for p in actor.parameters()) or any(p.grad is not None for p in critic.parameters()):
        raise ValueError('cold coordinate migration requires no outstanding gradients')
    actor.assert_frozen_state(optimizer)
    layers=[(n,m) for n,m in actor.mlp.named_modules() if isinstance(m,torch.nn.Linear)]
    name,last=layers[-1]
    if last.out_features!=24 or last.bias is None:raise ValueError('expected mean12/std12 last Linear')
    keys=['mlp.'+name+'.weight','mlp.'+name+'.bias']
    bindings,before_optimizer=_optimizer_bindings(actor,critic,optimizer)
    plans=[]
    for key,param in zip(keys,(last.weight,last.bias)):
        binding=bindings['actor.'+key]
        if binding['parameter'] is not param:raise ValueError('last Linear actual Parameter identity mismatch')
        state=optimizer.state.get(param)
        if (not isinstance(state,dict) or not {'step','exp_avg','exp_avg_sq'}.issubset(state)
                or not set(state).issubset({'step','exp_avg','exp_avg_sq','max_exp_avg_sq'})):
            raise ValueError('actual complete old Adam moments required, never fabricate them')
        moment_plan={}
        for field in ('exp_avg','exp_avg_sq','max_exp_avg_sq'):
            if field not in state:continue
            value=state[field]
            if (not isinstance(value,torch.Tensor) or value.shape!=param.shape
                    or not bool(torch.isfinite(value).all())):raise ValueError('Adam moment shape/nonfinite')
            moment_plan[field]=value[6:8].detach().clone()*(10 if field=='exp_avg' else 100)
            if not bool(torch.isfinite(moment_plan[field]).all()):
                raise ValueError('coordinate transform overflows Adam moment')
        if not bool(torch.isfinite(param).all()):raise ValueError('nonfinite old mean parameter')
        plans.append((key,param,state,param[6:8].detach().clone()/10,moment_plan))
    obs=torch.as_tensor(reference_observations,dtype=last.weight.dtype,device=last.weight.device)
    if obs.ndim!=2 or obs.shape[1]!=448 or not bool(torch.isfinite(obs).all()):
        raise ValueError('finite448 function-check observations required')
    active=obs[:,439]==1;inactive=obs[:,439]==0
    if not bool(active.any()) or not bool(inactive.any()) or not bool((active|inactive).all()):
        raise ValueError('both explicit active and inactive reference inputs required')
    old_cache=actor._last_forward_evidence
    def evaluate():
        with torch.no_grad():
            mean=actor({'policy':obs},stochastic_output=False).clone()
            evidence=actor._last_forward_evidence
            return mean,evidence['head_log_std'].exp().clone(),evidence['applied_local_raw_mean_delta'].clone()
    rng=capture_training_rng_state(seed=1001)
    original_actor={k:v.detach().clone() for k,v in actor.state_dict().items()}
    critic_hash=state_hash(critic.state_dict())
    optimizer_snapshot=copy.deepcopy(before_optimizer)
    parameter_ids={n:id(p) for n,p in actor.named_parameters()}
    optimizer_ids=tuple(tuple(id(p) for p in g['params']) for g in optimizer.param_groups)
    old_learning_rate=runner.alg.learning_rate
    old_aux_ledger=copy.deepcopy(getattr(runner,'local_auxiliary_events',None))
    before_mean,before_std,before_delta=evaluate()
    with torch.no_grad():
        for key,param,state,new_rows,moments in plans:
            param[6:8].copy_(new_rows)
            for field,rows in moments.items():state[field][6:8].copy_(rows)
    setattr(actor,GAIN_KEY,target)
    runner.local_configuration=configuration_with_gain(configuration,target)
    after_mean,after_std,after_delta=evaluate()
    actor._last_forward_evidence=old_cache
    expected_optimizer=copy.deepcopy(optimizer_snapshot)
    for key,param,state,rows,moments in plans:
        pid=bindings['actor.'+key]['serialized_id']
        for field,value in moments.items():expected_optimizer['state'][pid][field][6:8].copy_(value)
    protected=[]
    for key,value in actor.state_dict().items():
        old=original_actor[key]
        if key in keys:
            keep=[i for i in range(24) if i not in (6,7)]
            protected.append(torch.equal(old[keep],value[keep]))
        else:protected.append(torch.equal(old,value))
    other=[0,1,2,3,4,5,8,9,10,11]
    # Same action variable: no parameter-coordinate Jacobian in raw log density.
    fixed_action=before_mean+.123*before_std
    old_logp=torch.distributions.Normal(before_mean,before_std).log_prob(fixed_action).sum(-1)
    new_logp=torch.distributions.Normal(after_mean,after_std).log_prob(fixed_action).sum(-1)
    flags=dict(protected_actor_state_exact=all(protected),critic_exact=state_hash(critic.state_dict())==critic_hash,
        mapped_Adam_complete_state_exact=state_hash(optimizer.state_dict())==state_hash(expected_optimizer),
        original_Adam_object_kept=runner.alg.optimizer is optimizer,
        actual_Adam_parameter_order_exact=optimizer_ids==tuple(tuple(id(p) for p in g['params']) for g in optimizer.param_groups),
        PPO_learning_rate_exact=runner.alg.learning_rate==old_learning_rate,
        AUX_ledger_exact=getattr(runner,'local_auxiliary_events',None)==old_aux_ledger,
        actor_Parameter_objects_kept=parameter_ids=={n:id(p) for n,p in actor.named_parameters()},
        effective_local_delta_close=torch.allclose(before_delta,after_delta,rtol=2e-6,atol=1e-6),
        conditional_mean_close=torch.allclose(before_mean,after_mean,rtol=2e-6,atol=1e-6),
        conditional_std_exact=torch.equal(before_std,after_std),
        other10_conditional_mean_exact=torch.equal(before_mean[:,other],after_mean[:,other]),
        inactive_output_exact=torch.equal(before_mean[inactive],after_mean[inactive]),
        raw_logp_close=torch.allclose(old_logp,new_logp,rtol=0,atol=1e-4),
        full_RNG_exact=capture_training_rng_state(seed=1001)==rng)
    actor.assert_frozen_state(optimizer)
    return dict(schema='candidate.rr_mean_coordinate_migration.v1',status='PASS' if all(flags.values()) else 'FAIL',
        source_gain=list(source),target_gain=list(target),flags=flags,
        maximum_mean_error=float((before_mean-after_mean).abs().max()),
        maximum_delta_error=float((before_delta-after_delta).abs().max()),
        maximum_raw_logp_error=float((old_logp-new_logp).abs().max()),
        affected_named_parameters=keys,actual_serialized_Adam_ids={key:bindings['actor.'+key]['serialized_id'] for key in keys},
        mapping='W/b RR6:8 /10; exp_avg *10; exp_avg_sq/max_exp_avg_sq *100; step/LR/options unchanged',
        runtime_local_contract_required={GAIN_KEY:list(target)},
        new_policy_decisions=0,new_PPO_updates=0,new_PPO_Adam_steps=0,new_AUX_steps=0,
        AUX_ledger_unchanged=True,optimizer_dynamics_equivalent=False,
        publication='caller must version/configure runtime, strict reload, and collect fresh rollout')
