"""Cold continuation migration; no model construction, file I/O or import side effects.

Call only after the parent confirms Isaac has exited. The caller must construct
the old448 and versioned new465 actor/critic correctly and load the OLD optimizer
against the actual old model objects. New optimizer must have no accumulated
state. This helper cannot publish manifests, reset a rollout, or prove the new
physical event semantics. No checkpoints are changed by importing this file.
"""
from __future__ import annotations

import copy


def validate_continuation_runtime_change(old, new):
    """Declare the small post-activation code/schema change, not physics drift."""
    changed_allowed = {
        'src/wlr50_clean/ppo/semantic_rr_capture_local.py',
        'src/wlr50_clean/ppo/semantic_rr_capture_local_actor.py',
        'src/wlr50_clean/ppo/semantic_rr_mean_coordinates.py',
        'src/wlr50_clean/ppo/semantic_residual_adapter.py',
        'src/wlr50_clean/ppo/semantic_backend.py',
        'src/wlr50_clean/ppo/actuator_target_effect.py',
        'configs/ppo_rr_capture_first_cp225280_v1/local_training.json',
    }
    added_allowed = {
        'src/wlr50_clean/ppo/semantic_rr_capture_continuation_task.py',
        'src/wlr50_clean/ppo/semantic_rr_capture_continuation_source.py',
        'src/wlr50_clean/ppo/semantic_rr_authorized_assist.py',
        'src/wlr50_clean/ppo/semantic_rr_continuation_migration.py',
        'src/wlr50_clean/ppo/semantic_rr_gain_aware_aux.py',
        'src/wlr50_clean/ppo/semantic_rr_continuation_pause.py',
    }
    runtime_fields = {'source_git_commit', 'runtime_content_sha256', 'files',
                      'local_contract', 'selected_configuration'}
    if set(old) != set(new) or any(old[k] != new[k] for k in old if k not in runtime_fields):
        raise ValueError('protected runtime/library/physics contract changed')
    old_files, new_files = old['files'], new['files']
    changed = {p for p in old_files.keys() & new_files.keys() if old_files[p] != new_files[p]}
    added = set(new_files)-set(old_files)
    if set(old_files)-set(new_files) or not changed <= changed_allowed or not added <= added_allowed:
        raise ValueError('unreviewed continuation runtime file change')
    if not changed or old['source_git_commit'] == new['source_git_commit']:
        raise ValueError('continuation publication requires a distinct committed version')
    for name, item in old['selected_configuration'].items():
        if name != 'local_training.json' and new['selected_configuration'].get(name) != item:
            raise ValueError('accepted prior control/profile changed: '+name)
    a, b = old['local_contract'], new['local_contract']
    protected = ('prior_checkpoint_sha256', 'prior_manifest_sha256',
        'control_configuration_source', 'local_mean_coordinate_gain_full12',
        'active_initial_sigma_full12', 'actor_hidden_dims', 'critic_hidden_dims',
        'learning_rate', 'schedule', 'target_kl', 'gamma', 'lambda',
        'normalizer', 'history_composition', 'front_FL_assist')
    if any(a.get(k) != b.get(k) for k in protected):
        raise ValueError('continuation must retain weights coordinates/prior/HISTORY/Adam recipe')
    if a.get('observation_dimension') != 448 or b.get('observation_dimension') != 465:
        raise ValueError('this publication is exactly448->465')
    if not b.get('continuous_rr_rl') or b.get('rr_authorized_assist') is not False:
        raise ValueError('first continuation publication is explicit, unassisted continuous mode')
    return {p: {'before': old_files.get(p), 'after': new_files[p]} for p in sorted(changed|added)}


def named_optimizer_layout(actor, critic, optimizer):
    """Join real Parameter identity to qualified names and serialized IDs.

    Never infer Adam IDs from an assumed 0..N range or state-dict key order.
    The IDs returned by this exact optimizer's state_dict are paired only with
    that optimizer's current live parameter groups, whose order is verified.
    """
    named = [("actor."+name,param) for name,param in actor.named_parameters()]
    named += [("critic."+name,param) for name,param in critic.named_parameters()]
    ids = [id(param) for _,param in named]
    if len(ids) != len(set(ids)):
        raise ValueError("shared or duplicate actor/critic Parameter identity")
    by_identity = {id(param):(name,param) for name,param in named}
    serialized = optimizer.state_dict()
    if len(serialized["param_groups"]) != len(optimizer.param_groups):
        raise ValueError("serialized/live optimizer group count differs")
    groups, bindings, visited = [], {}, set()
    for live,saved in zip(optimizer.param_groups,serialized["param_groups"]):
        if len(live["params"]) != len(saved["params"]):
            raise ValueError("serialized/live optimizer parameter count differs")
        names=[]
        for parameter,serialized_id in zip(live["params"],saved["params"]):
            if id(parameter) not in by_identity:
                raise ValueError("optimizer parameter is outside actor/critic")
            name,_ = by_identity[id(parameter)]
            if (name.startswith("actor.frozen_prior.") or not parameter.requires_grad
                    or name in bindings or serialized_id in visited):
                raise ValueError("frozen/duplicate optimizer member")
            names.append(name)
            bindings[name]=dict(parameter=parameter,serialized_id=serialized_id)
            visited.add(serialized_id)
        groups.append(names)
    expected = {name for name,param in named if param.requires_grad}
    if set(bindings) != expected:
        raise ValueError("optimizer does not cover exactly all trainable actor/critic parameters")
    if not set(serialized["state"]).issubset(visited):
        raise ValueError("Adam state contains unmapped serialized IDs")
    return groups,bindings,serialized


def migrate_local448_to465(old_actor,new_actor,old_critic,new_critic,
                           old_optimizer,new_optimizer,*,isaac_stopped=False):
    """Copy model/Adam history, adding only zero input columns.

    Only new model objects and new optimizer are mutated; old objects are
    read-only. All shapes/names/groups/moments are validated before loading.
    This is not an atomic filesystem publisher; on a load failure discard the
    unpublished new objects. Caller retains its saved old checkpoint intact.
    """
    if not isaac_stopped:
        raise ValueError("explicit Isaac exit confirmation required before Torch")
    import torch
    if type(old_optimizer) is not torch.optim.Adam or type(new_optimizer) is not torch.optim.Adam:
        raise ValueError("this narrow draft only supports actual torch.optim.Adam")
    if new_optimizer.state:
        raise ValueError("new optimizer already contains state; refuse overwrite")
    old_groups,old_bindings,old_serialized=named_optimizer_layout(old_actor,old_critic,old_optimizer)
    new_groups,new_bindings,new_serialized=named_optimizer_layout(new_actor,new_critic,new_optimizer)
    if old_groups != new_groups:
        raise ValueError("actual qualified optimizer parameter order changed")
    model_states, expanded = {},set()
    for role,old,new in (("actor",old_actor,new_actor),("critic",old_critic,new_critic)):
        first_old=next(((n,m) for n,m in old.named_modules() if n.startswith("mlp.") and isinstance(m,torch.nn.Linear)),None)
        first_new=next(((n,m) for n,m in new.named_modules() if n.startswith("mlp.") and isinstance(m,torch.nn.Linear)),None)
        if first_old is None or first_new is None or first_old[0] != first_new[0]:
            raise ValueError(role+" first input Linear not explicit or name changed")
        if first_old[1].in_features != 448 or first_new[1].in_features != 465:
            raise ValueError(role+" must expand exactly448->465")
        key=first_old[0]+".weight"
        expanded.add(role+"."+key)
        source,target=old.state_dict(),new.state_dict()
        if source.keys()!=target.keys():
            raise ValueError(role+" state key set changed")
        result={}
        for name,value in source.items():
            destination=target[name]
            if not isinstance(value,torch.Tensor) or value.dtype != destination.dtype:
                raise ValueError("unsupported state type/dtype migration: "+role+"."+name)
            if name==key:
                if (value.ndim!=2 or destination.shape != (value.shape[0],465)
                        or value.shape[1]!=448):
                    raise ValueError("first input weight is not seventeen-column extension")
                result[name]=torch.cat((value.detach().clone(),value.new_zeros((value.shape[0],17))),dim=1)
            else:
                if value.shape != destination.shape:
                    raise ValueError("unapproved state shape change: "+role+"."+name)
                result[name]=value.detach().clone()
        model_states[role]=result
    old_prior={k:v for k,v in model_states["actor"].items() if k.startswith("frozen_prior.")}
    if not old_prior:
        raise ValueError("composite actor is missing frozen prior")
    for actor in (old_actor,new_actor):
        prior_input=next((module for module in actor.frozen_prior.modules() if isinstance(module,torch.nn.Linear)),None)
        if prior_input is None or prior_input.in_features!=439:
            raise ValueError("frozen prior input must remain439")
    if any(p.requires_grad for p in old_actor.frozen_prior.parameters()) or any(p.requires_grad for p in new_actor.frozen_prior.parameters()):
        raise ValueError("prior must already be fully frozen")
    new_state={}
    moment_names={"exp_avg","exp_avg_sq","max_exp_avg_sq"}
    copied_state_names=[]
    for name,old_binding in old_bindings.items():
        old_id,new_id=old_binding["serialized_id"],new_bindings[name]["serialized_id"]
        old_shape=tuple(old_binding["parameter"].shape)
        new_shape=tuple(new_bindings[name]["parameter"].shape)
        state=old_serialized["state"].get(old_id)
        if state is None:
            # A truly absent old Adam state stays absent, not fabricated.
            continue
        if not set(state).issubset(moment_names|{"step"}) or not {"step","exp_avg","exp_avg_sq"}.issubset(state):
            raise ValueError("unexpected/incomplete Adam state for "+name)
        new_entry={}
        for state_name,value in state.items():
            if state_name=="step":
                if isinstance(value,torch.Tensor):
                    if value.numel()!=1:
                        raise ValueError("Adam step must be scalar")
                    new_entry[state_name]=value.detach().clone()
                elif type(value) in (int,float):
                    new_entry[state_name]=value
                else:
                    raise ValueError("unsupported Adam step type")
            else:
                if not isinstance(value,torch.Tensor) or tuple(value.shape)!=old_shape:
                    raise ValueError("Adam moment/Parameter shape mismatch: "+name)
                if name in expanded:
                    if new_shape != (old_shape[0],465) or old_shape[1]!=448:
                        raise ValueError("bad Adam input-column expansion")
                    new_entry[state_name]=torch.cat((value.detach().clone(),value.new_zeros((value.shape[0],17))),dim=1)
                else:
                    if old_shape!=new_shape:
                        raise ValueError("unapproved Adam shape change")
                    new_entry[state_name]=value.detach().clone()
        new_state[new_id]=new_entry
        copied_state_names.append(name)
    new_param_groups=[]
    for old_group,new_group in zip(old_serialized["param_groups"],new_serialized["param_groups"]):
        group=copy.deepcopy(old_group)
        group["params"]=list(new_group["params"])
        new_param_groups.append(group)
    # No calls above sample randomness; zeros/copies only. Caller-created new
    # modules may have consumed RNG before entry; caller must restore the saved
    # checkpoint RNG afterwards, not pretend constructor RNG was continuation.
    rng_before=torch.get_rng_state().clone()
    new_actor.load_state_dict(model_states["actor"],strict=True)
    new_critic.load_state_dict(model_states["critic"],strict=True)
    new_optimizer.load_state_dict(dict(state=new_state,param_groups=new_param_groups))
    if not torch.equal(rng_before,torch.get_rng_state()):
        raise RuntimeError("migration unexpectedly consumed CPU RNG")
    for name,value in old_prior.items():
        if not torch.equal(new_actor.state_dict()[name].cpu(),value.cpu()):
            raise RuntimeError("frozen prior changed")
    actual_groups,actual_bindings,actual_state=named_optimizer_layout(new_actor,new_critic,new_optimizer)
    if actual_groups!=old_groups:
        raise RuntimeError("post-load parameter group identity/order changed")
    for name in copied_state_names:
        expected=new_state[new_bindings[name]["serialized_id"]]
        actual=actual_state["state"][actual_bindings[name]["serialized_id"]]
        for state_name,value in expected.items():
            target=actual[state_name]
            if isinstance(value,torch.Tensor):
                if not torch.equal(target.cpu(),value.cpu()):
                    raise RuntimeError("Adam migration changed "+name+":"+state_name)
            elif target!=value:
                raise RuntimeError("Adam scalar changed")
    return dict(schema="wlr50_clean.local448_to465.zero_input_columns.v3",
        expanded_parameter_names=sorted(expanded),optimizer_group_names=old_groups,
        copied_Adam_parameter_states=len(copied_state_names),
        preserved_group_options=[{k:copy.deepcopy(v) for k,v in g.items() if k!="params"} for g in actual_state["param_groups"]],
        prior439_unchanged=True,learned_mean_not_reinitialized=True,
        new_input_weight_and_Adam_columns_zero=True,CPU_RNG_unchanged_during_helper=True,
        RNG_obligation="Caller restores immutable checkpoint full Python/NumPy/CPU/CUDA RNG after model construction and migration; helper does not read or replace external RNG records.",
        rollout_obligation="Fresh465 observations only; discard no committed update; never reuse unfinished448 rollout.")
