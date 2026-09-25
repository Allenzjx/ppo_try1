"""Finite cold AUX candidate; importing/preparing uses only the standard library.

No CLI, simulator, publisher, task mutation, gain migration or PPO-buffer writer.
Prepare on actual448 first, or actual462 later: never fabricate continuation memory.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
from pathlib import Path
import traceback

RR = (6, 7)
OTHER = (0, 1, 2, 3, 4, 5, 8, 9, 10, 11)
PACKAGE = 'wlr50_clean.gainaware_rr_aux_evidence.v2'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def finite_vector(value, length):
    return (isinstance(value, list) and len(value) == length and
            all(type(x) in (int, float) and math.isfinite(x) for x in value))


def checked_gain(value):
    if not finite_vector(list(value), 12) or tuple(value) not in ((1.,)*12, (1.,)*6+(10.,10.)+(1.,)*4):
        raise ValueError('explicit identity or existing RR gain10 required; no migration here')
    return tuple(float(x) for x in value)


def recipe(*, steps=8, learning_rate=.001, max_shift_sigma=.5, backtracks=8):
    if type(steps) is not int or not 1 <= steps <= 64:
        raise ValueError('1..64 actual independent SGD proposals required')
    if type(learning_rate) not in (int, float) or not 0 < learning_rate <= .02:
        raise ValueError('finite effective-coordinate learning rate in (0,.02] required')
    if type(max_shift_sigma) not in (int, float) or not 0 < max_shift_sigma <= 8.:
        raise ValueError('explicit cumulative conditional sigma budget in (0,8] required')
    if type(backtracks) is not int or not 0 <= backtracks <= 12:
        raise ValueError('finite 0..12 backtracks per proposal required')
    return dict(schema='wlr50_clean.gainaware_rr_aux_recipe.v2', steps=steps,
        learning_rate=float(learning_rate), max_shift_sigma=float(max_shift_sigma), backtracks=backtracks,
        recipe_gainaware=True, coordinate='zero_delta_effective_U_v_then_W0_plus_delta_div_gain',
        loss='parent_sigma_scaled_actual_conditional_raw_smooth_l1_RR_only',
        optimizer='independent_SGD_momentum0_weight_decay0', mutable_rows=[6,7],
        trust='cumulative_from_loaded_parent_actual_conditional_mean_over_parent_sigma',
        PPO_credit=0, default_is_proposal_not_execution=True)


def verified_row(record, *, source_binding, line_number, collector_id, segment_id, group):
    """Normalize an ORIGINAL saved on-policy row, retaining its exact identity.

    Caller selects bounded contiguous AIR->real TOP windows, not isolated lucky
    samples or an entire failed episode. Package construction rechecks source
    line hashes and each update's actual collector checkpoint/range.
    """
    if group not in ('entry_AIR', 'AIR', 'TOP', 'drop', 'inactive'):
        raise ValueError('explicit physical reference group required')
    observation = record.get('observation')
    dimension = len(observation) if isinstance(observation, list) else 0
    if dimension < 448 or not finite_vector(observation, dimension):
        raise ValueError('actual saved full observation required; no padding/memory invention')
    policy, info = record['policy_request'], record['step_info']
    before, after = record['before_local']['metrics'], info['rr_capture_local']['metrics']
    raw = info['raw_policy_action_full12']
    if (record.get('kind') != 'activated_on_policy' or record.get('PPO_credit') != 1
            or policy.get('independent_diagnostic') or
            (info.get('authorized_RR_assist') and info['authorized_RR_assist'].get('active') is not False)
            or policy.get('history_kernel_applications') != 1 or policy.get('sampling_draws') != 1
            or policy.get('extra_random_draws') != 0 or not finite_vector(raw,12)
            or raw != policy.get('selected_raw_full12')
            or policy.get('selected_raw_log_probability') != record.get('old_logp',[None])[0]):
        raise ValueError('teacher must be genuine actual issued on-policy raw, not FINAL/override/fictitious logp')
    if group == 'inactive' or observation[439] != 1.:
        raise ValueError('this collector adapter has only active observations; inactive needs separate real evidence')
    if (group in ('entry_AIR','AIR') and before.get('free_air') is not True
            or group == 'TOP' and before.get('current_top_contact') is not True):
        raise ValueError('physical group differs from recorded PRE-action state')
    audit = info['actuator_target_effect_audit']
    if audit.get('verified') is not True or audit.get('phase_mask_full12') != [1]*12:
        raise ValueError('projection/dispatch evidence missing')
    if type(line_number) is not int or line_number < 1 or not source_binding or not collector_id or not segment_id:
        raise ValueError('explicit source line, collector and continuous-segment identity required')
    return dict(source_binding=source_binding, source_line=line_number, source_record_sha256=digest(record),
        collector_id=collector_id, segment_id=segment_id, group=group,
        global_decision=record['global_decision'], start_tick=before['tick'], end_tick=after['tick'],
        observation=copy.deepcopy(observation), actual_issued_raw_full12=copy.deepcopy(raw),
        before_metrics=copy.deepcopy(before), after_metrics=copy.deepcopy(after),
        original_local_success=bool(info.get('local_task_success')), full_task_success_claim=False,
        rr_headroom_clipped=bool(set(audit['policy_headroom_evidence']['clipped_servo_indices']).intersection(RR)),
        projection_and_other10_context=copy.deepcopy(audit),
        actual_FINAL_not_teacher=copy.deepcopy(info['actual_drive_target_full12']))


def _check_source_lines(rows, bindings):
    selected = {}
    for row in rows:
        wanted = selected.setdefault(row['source_binding'], {})
        number, expected = row['source_line'], row['source_record_sha256']
        if number in wanted and wanted[number][0]['source_record_sha256'] != expected:
            raise ValueError('conflicting source-row identity')
        wanted.setdefault(number, []).append(row)
    for name, wanted in selected.items():
        with Path(bindings[name]['path']).open('rb') as stream:
            for number, line in enumerate(stream,1):
                if number in wanted:
                    record = json.loads(line)
                    if not line.endswith(b'\n') or digest(record) != wanted[number][0]['source_record_sha256']:
                        raise ValueError('sealed source row hash mismatch')
                    for row in wanted[number]:
                        normalized = verified_row(record, source_binding=name, line_number=number,
                            collector_id=row['collector_id'], segment_id=row['segment_id'], group=row['group'])
                        if normalized != row:
                            raise ValueError('normalized observation/action differs from actual source row')
                    del wanted[number]
                if not wanted:
                    break
        if wanted:
            raise ValueError('source rows are not present in sealed data')


def prepare_package(*, parent_metadata, parent_checkpoint_binding, parent_manifest_binding,
                    bindings, collectors, teacher_rows, reference_rows, scope):
    """Bind explicit files; collector ranges derive from each CP's actual count.

    bindings: name -> {path,sha256}. collectors: id -> {checkpoint_binding,
    manifest_binding,complete_run_manifest_binding}. Each actual collector CP
    owns at most its following512 active samples; parent may be a later CP.
    Parent owns source/version approval and continuous-window selection.
    """
    dimension = parent_metadata['runtime_contract']['local_contract']['observation_dimension']
    result = dict(schema=PACKAGE, observation_dimension=dimension, scope=scope,
        parent_metadata=copy.deepcopy(parent_metadata), parent_checkpoint_binding=parent_checkpoint_binding,
        parent_manifest_binding=parent_manifest_binding, bindings=copy.deepcopy(bindings),
        collectors=copy.deepcopy(collectors), teacher_rows=copy.deepcopy(teacher_rows),
        reference_rows=copy.deepcopy(reference_rows), teacher_target='actual_issued_raw_not_FINAL',
        diagnostic_success_claim=False, on_policy_reuse_forbidden=True)
    result['package_sha256'] = digest(result)
    validate_package(result, verify_files=True)
    return result


def validate_package(package, *, verify_files=True):
    if (package.get('schema') != PACKAGE or package.get('package_sha256') !=
            digest({k:v for k,v in package.items() if k != 'package_sha256'})):
        raise ValueError('gain-aware package schema/digest mismatch')
    dim = package['observation_dimension']
    if type(dim) is not int or dim < 448 or not package['teacher_rows'] or not package['reference_rows']:
        raise ValueError('nonempty genuine full-layout teacher and reference sets required')
    parent = package['parent_metadata']
    if parent.get('rollout_empty') is not True or parent.get('save_load_round_trip') is not True:
        raise ValueError('strict reloaded complete parent required')
    checked_gain(parent['runtime_contract']['local_contract']['local_mean_coordinate_gain_full12'])
    if type(parent['counts'].get('auxiliary_updates')) is not int:
        raise ValueError('existing AUX count must be retained')
    rows = package['teacher_rows'] + package['reference_rows']
    for row in rows:
        if not finite_vector(row['observation'],dim) or not finite_vector(row['actual_issued_raw_full12'],12):
            raise ValueError('row dimension/raw mismatch;448 and462 never mixed')
        if not row['start_tick'] < row['end_tick'] or row['collector_id'] not in package['collectors']:
            raise ValueError('row clock/collector mismatch')
    segments = {}
    for row in package['teacher_rows']:
        if row.get('rr_headroom_clipped') is not False:
            raise ValueError('clipped RR rows are reference-only, not direct raw teacher labels')
        segments.setdefault(row['segment_id'], []).append(row)
    for selected in segments.values():
        for before, after in zip(selected, selected[1:]):
            if before['end_tick'] != after['start_tick'] or before['global_decision']+1 != after['global_decision']:
                raise ValueError('teacher window must preserve contiguous recorded consequences')
        if not any(r['after_metrics'].get('current_top_contact') is True and
                   r['after_metrics'].get('current_top_bearing') is True and
                   r['after_metrics'].get('bearing_force_n',0.) >= .2 for r in selected):
            raise ValueError('this actual-success-window adapter requires real TOP bearing evidence')
    if not verify_files:
        return
    bindings = package['bindings']
    for item in bindings.values():
        if file_sha(item['path']) != item['sha256']:
            raise ValueError('sealed file changed')
    manifest = json.loads(Path(bindings[package['parent_manifest_binding']]['path']).read_text())
    if manifest != parent or parent['checkpoint_sha256'] != bindings[package['parent_checkpoint_binding']]['sha256']:
        raise ValueError('bound parent checkpoint/metadata differ')
    for identity, collector in package['collectors'].items():
        meta = json.loads(Path(bindings[collector['manifest_binding']]['path']).read_text())
        complete = json.loads(Path(bindings[collector['complete_run_manifest_binding']]['path']).read_text())
        if complete.get('lifecycle') != 'COMPLETE' or complete.get('mode') != 'train':
            raise ValueError('unsealed/non-training collector source rejected')
        if (meta['checkpoint_sha256'] != bindings[collector['checkpoint_binding']]['sha256']
                or meta.get('rollout_empty') is not True or meta.get('save_load_round_trip') is not True
                or meta['runtime_contract'] != parent['runtime_contract']):
            raise ValueError('collector checkpoint/observation binding mismatch')
        source_folder = Path(bindings[collector['complete_run_manifest_binding']]['path']).resolve().parent
        if any(Path(bindings[row['source_binding']]['path']).resolve().parent != source_folder
               for row in rows if row['collector_id'] == identity):
            raise ValueError('source decisions and COMPLETE manifest belong to different runs')
        first = 225280 + meta['counts']['local_policy_decisions'] + 1
        if any(not first <= row['global_decision'] <= first+511 for row in rows if row['collector_id'] == identity):
            raise ValueError('wrong collector checkpoint for actual update512 window')
    _check_source_lines(rows,bindings)


def fit_effective_rr_rows(runner, package, sealed_recipe, *, isaac_stopped=False):
    """Cold-only candidate. Caller must seal receipt, ledger and strict new CP.

    One SGD proposal counts one AUX step even if backtracked or rejected. Every
    trust comparison stays relative to the original loaded parent, not the last
    accepted step. No automatic retry/restart or implicit optimization budget.
    """
    if not isaac_stopped:
        raise ValueError('explicit Isaac exit acknowledgement required before Torch')
    validate_package(package)
    if sealed_recipe != recipe(**{k:sealed_recipe[k] for k in ('steps','learning_rate','max_shift_sigma','backtracks')}):
        raise ValueError('unrecognized gain-aware recipe')
    import torch
    from wlr50_clean.ppo.semantic_training import state_hash
    from wlr50_clean.ppo.rl_library_wrapper import capture_training_rng_state, restore_training_rng_state
    actor, critic, adam = runner.alg.actor, runner.alg.critic, runner.alg.optimizer
    parent, dim = package['parent_metadata'], package['observation_dimension']
    gain = checked_gain(list(actor.local_mean_coordinate_gain_full12))
    if (getattr(actor,'observation_dimension',None) != dim or runner.alg.storage.step != 0
            or runner.alg.transition.actions is not None or type(adam) is not torch.optim.Adam
            or tuple(parent['runtime_contract']['local_contract']['local_mean_coordinate_gain_full12']) != gain):
        raise ValueError('actual loaded layout/gain/Adam/empty-rollout contract mismatch')
    binding = package['bindings'][package['parent_checkpoint_binding']]
    if runner.checkpoint_load_provenance['checkpoint_sha256'] != binding['sha256']:
        raise ValueError('runner was not loaded from declared newest AUX parent')
    saved = runner.alg.save()
    if any(state_hash(saved[k]) != h for k,h in parent['state_hashes'].items()):
        raise ValueError('actual actor/critic/Adam no longer equals declared parent')
    if (runner.alg.learning_rate != parent['learning_rate'] or
            any(g['lr'] != parent['learning_rate'] for g in adam.param_groups)):
        raise ValueError('PPO learning rate differs from parent')
    ledger = copy.deepcopy(getattr(runner,'local_auxiliary_events',[]))
    if ledger != parent.get('local_auxiliary_events',[]):
        raise ValueError('historical AUX ledger differs; never reset old64')
    actor.assert_frozen_state(adam)
    if any(p.grad is not None for model in (actor,critic) for p in model.parameters()):
        raise ValueError('cold AUX requires no outstanding gradients')
    name,last = [(n,m) for n,m in actor.mlp.named_modules() if isinstance(m,torch.nn.Linear)][-1]
    if last.out_features != 24 or last.bias is None:
        raise ValueError('expected mean12/std12 final Linear')
    wk,bk = 'mlp.'+name+'.weight','mlp.'+name+'.bias'
    original = {k:v.detach().clone() for k,v in actor.state_dict().items()}
    flags = [(p,p.requires_grad) for p in actor.parameters()]
    ids = {n:id(p) for n,p in actor.named_parameters()}
    optimizer_ids = tuple(tuple(id(p) for p in g['params']) for g in adam.param_groups)
    before_hashes = {k:state_hash(v) for k,v in saved.items()}
    old_cache = actor._last_forward_evidence
    rng = capture_training_rng_state(seed=1001)
    # Optimize EFFECTIVE coordinates. No W0*g/g round-trip at delta0.
    du = torch.nn.Parameter(torch.zeros_like(last.weight[6:8]))
    dv = torch.nn.Parameter(torch.zeros_like(last.bias[6:8]))
    independent = torch.optim.SGD([du,dv],lr=sealed_recipe['learning_rate'],momentum=0.,weight_decay=0.)
    teachers = package['teacher_rows']
    references = package['reference_rows'] + teachers  # trust includes every supervised point
    obs = torch.tensor([r['observation'] for r in teachers],dtype=last.weight.dtype,device=last.weight.device)
    ref = torch.tensor([r['observation'] for r in references],dtype=last.weight.dtype,device=last.weight.device)
    target = torch.tensor([r['actual_issued_raw_full12'][6:8] for r in teachers],dtype=last.weight.dtype,device=last.weight.device)
    def rows():
        return original[wk][6:8]+du/gain[6], original[bk][6:8]+dv/gain[7]
    def forward(values):
        w,b = rows()
        override = {wk:torch.cat((original[wk][:6],w,original[wk][8:])),
                    bk:torch.cat((original[bk][:6],b,original[bk][8:]))}
        mu = torch.func.functional_call(actor,override,({'policy':values},),
                                        dict(stochastic_output=False),strict=False)
        return mu,actor._last_forward_evidence['head_log_std'].exp().clone()
    def effective_state():
        return dict(delta_U=du.detach().cpu().tolist(),delta_v=dv.detach().cpu().tolist())
    attempts,actual_steps,accepted = [],0,0
    accepted_state = (du.detach().clone(),dv.detach().clone())
    status,reason,committed = 'COMPLETE','finite_budget_exhausted',False
    groups,teacher_fit = {},{}
    initial_loss = final_loss = None
    committed_function_exact = None
    try:
        for p,_ in flags:
            p.requires_grad_(False)
        with torch.no_grad():
            parent_ref = actor({'policy':ref},stochastic_output=False).clone()
            parent_std = actor._last_forward_evidence['head_log_std'].exp().clone()
            parent_mu = actor({'policy':obs},stochastic_output=False).clone()
            teacher_std = actor._last_forward_evidence['head_log_std'].exp()[:,6:8].clone()
            zero_mu,zero_std = forward(ref)
            if (not torch.equal(zero_mu,parent_ref) or not torch.equal(zero_std,parent_std)
                    or not bool(torch.isfinite(parent_std).all()) or not bool((parent_std>0).all())):
                raise RuntimeError('zero effective increment changed function/sigma or invalid parent sigma')
        def loss_fn(mu):
            return torch.nn.functional.smooth_l1_loss(mu[:,6:8]/teacher_std,target/teacher_std)
        previous_loss = initial_loss = float(loss_fn(parent_mu))
        accepted_state = (du.detach().clone(),dv.detach().clone())
        for number in range(1,sealed_recipe['steps']+1):
            independent.zero_grad(set_to_none=True)
            mu,_ = forward(obs)
            loss = loss_fn(mu)
            du.grad,dv.grad = torch.autograd.grad(loss,(du,dv))
            independent.step()
            actual_steps += 1
            proposal = (du.detach().clone(),dv.detach().clone())
            trial_records = []
            passed = False
            for backtrack in range(sealed_recipe['backtracks']+1):
                alpha = 2.**(-backtrack)
                with torch.no_grad():
                    du.copy_(accepted_state[0]+alpha*(proposal[0]-accepted_state[0]))
                    dv.copy_(accepted_state[1]+alpha*(proposal[1]-accepted_state[1]))
                    candidate_mu,_ = forward(obs)
                    value = float(loss_fn(candidate_mu))
                    reference_mu,reference_std = forward(ref)
                    shift = float(((reference_mu[:,6:8]-parent_ref[:,6:8])/parent_std[:,6:8]).abs().max())
                    frozen_outputs = torch.equal(reference_mu[:,OTHER],parent_ref[:,OTHER]) and torch.equal(reference_std,parent_std)
                    passed = (math.isfinite(value) and math.isfinite(shift) and value <= previous_loss
                              and shift <= sealed_recipe['max_shift_sigma'] and frozen_outputs)
                trial_records.append(dict(alpha=alpha,loss=value,cumulative_shift_sigma=shift,
                                          frozen_outputs_exact=frozen_outputs,accepted=passed))
                if passed:
                    break
            attempts.append(dict(proposal=number,backtracks=trial_records,
                proposed_effective_delta={'delta_U':proposal[0].cpu().tolist(),'delta_v':proposal[1].cpu().tolist()},
                accepted_effective_delta=effective_state() if passed else None))
            if not passed:
                with torch.no_grad():
                    du.copy_(accepted_state[0]); dv.copy_(accepted_state[1])
                status,reason = 'STOPPED_CONSTRAINT','bounded backtracking exhausted; no restart'
                break
            accepted += 1
            previous_loss = value
            accepted_state = (du.detach().clone(),dv.detach().clone())
        with torch.no_grad():
            final_ref,final_std = forward(ref)
            final_teacher,_ = forward(obs)
            final_loss = float(loss_fn(final_teacher))
            teacher_fit = dict(channels=['RR_hip','RR_knee'],
                actual_raw_target_mean=target.mean(0).cpu().tolist(),
                conditional_mean_before=parent_mu[:,6:8].mean(0).cpu().tolist(),
                conditional_mean_after=final_teacher[:,6:8].mean(0).cpu().tolist(),
                parent_sigma_mean=teacher_std.mean(0).cpu().tolist(),
                conditional_RMSE_before=(parent_mu[:,6:8]-target).square().mean(0).sqrt().cpu().tolist(),
                conditional_RMSE_after=(final_teacher[:,6:8]-target).square().mean(0).sqrt().cpu().tolist())
            for group in sorted({r['group'] for r in references}):
                ix = [i for i,r in enumerate(references) if r['group']==group]
                groups[group] = dict(rows=len(ix),
                    conditional_mean_before=parent_ref[ix,6:8].mean(0).cpu().tolist(),
                    conditional_mean_after=final_ref[ix,6:8].mean(0).cpu().tolist(),
                    cumulative_max_shift_sigma=float(((final_ref[ix,6:8]-parent_ref[ix,6:8])/parent_std[ix,6:8]).abs().max()))
            w,b = rows()
            last.weight[6:8].copy_(w); last.bias[6:8].copy_(b)
            committed = accepted > 0
            # Check the actual mutated actor, not only functional-call candidates.
            actual_committed = actor({'policy':ref},stochastic_output=False).clone()
            actual_std = actor._last_forward_evidence['head_log_std'].exp().clone()
            committed_function_exact = torch.equal(actual_committed,final_ref) and torch.equal(actual_std,final_std)
            if not committed_function_exact:
                raise RuntimeError('committed actor differs from accepted functional candidate')
    except Exception:
        status,reason = 'FAILED',traceback.format_exc()
        attempts.append(dict(failed=True,actual_optimizer_steps_at_failure=actual_steps,
                             candidate_effective_delta=effective_state(),error=reason))
        with torch.no_grad():
            du.copy_(accepted_state[0]); dv.copy_(accepted_state[1])
            if committed:
                last.weight[6:8].copy_(original[wk][6:8]); last.bias[6:8].copy_(original[bk][6:8])
                committed=False
    finally:
        for p,flag in flags:
            p.requires_grad_(flag)
        actor._last_forward_evidence = old_cache
        restore_training_rng_state(rng,expected_seed=1001)
    after_hashes = {k:state_hash(v) for k,v in runner.alg.save().items()}
    keep = [i for i in range(24) if i not in RR]
    invariants = dict(
        protected_actor_exact=all(torch.equal(v[keep],actor.state_dict()[k][keep]) if k in (wk,bk)
                                  else torch.equal(v,actor.state_dict()[k]) for k,v in original.items()),
        critic_exact=before_hashes['critic_state_dict']==after_hashes['critic_state_dict'],
        PPO_Adam_exact=before_hashes['optimizer_state_dict']==after_hashes['optimizer_state_dict'],
        Parameter_objects_exact=ids=={n:id(p) for n,p in actor.named_parameters()},
        Adam_object_order_exact=runner.alg.optimizer is adam and optimizer_ids==tuple(tuple(id(p) for p in g['params']) for g in adam.param_groups),
        gain_unchanged=tuple(actor.local_mean_coordinate_gain_full12)==gain,
        old_AUX_ledger_exact=ledger==getattr(runner,'local_auxiliary_events',[]),
        PPO_LR_exact=runner.alg.learning_rate==parent['learning_rate'] and all(g['lr']==parent['learning_rate'] for g in adam.param_groups),
        full_RNG_exact=capture_training_rng_state(seed=1001)==rng)
    if not all(invariants.values()):
        status,reason = 'FAILED_INVARIANT','do not publish; preserve failed receipt'
    try:
        actor.assert_frozen_state(adam)
    except Exception:
        status,reason = 'FAILED_INVARIANT',traceback.format_exc()
    receipt = dict(schema='wlr50_clean.gainaware_rr_aux_receipt.v2',status=status,reason=reason,
        recipe=sealed_recipe,package_sha256=package['package_sha256'],parent_checkpoint=binding,
        AUX_optimizer_steps=actual_steps,AUX_accepted_steps=accepted,actor_rows_committed=committed,
        attempts=attempts,reference_groups=groups,inactive_real_reference_present='inactive' in groups,
        accepted_effective_delta=effective_state(),gain=list(gain),invariants=invariants,
        teacher_fit=teacher_fit,initial_loss=initial_loss,final_loss=final_loss,
        actual_committed_function_exact=committed_function_exact,
        state_hashes_before=before_hashes,state_hashes_after=after_hashes,
        new_PPO_updates=0,new_PPO_Adam_steps=0,new_policy_decisions=0,
        automatic_success_claim=False,publication='caller-owned new checkpoint and strict reload; fresh PPO mandatory')
    # Keep failures serializable without silently dropping the attempted step.
    def safe(value):
        if isinstance(value,dict): return {k:safe(v) for k,v in value.items()}
        if isinstance(value,list): return [safe(v) for v in value]
        if isinstance(value,float) and not math.isfinite(value): return {'nonfinite':repr(value)}
        return value
    receipt = safe(receipt)
    counts = copy.deepcopy(parent['counts'])
    event = None
    if actual_steps:
        event = dict(schema='wlr50_clean.finite_rr_mean_row_aux_event.v1',recipe_gainaware=True,
            actual_optimizer_steps=actual_steps,accepted_steps=accepted,PPO_credit=0,
            source_checkpoint=binding,package_sha256=package['package_sha256'],
            recipe_sha256=digest(sealed_recipe),receipt_sha256=digest(receipt),
            status=status,actor_rows_committed=committed,
            purpose='bounded_gainaware_actual_raw_RR_AUX_not_PPO_or_physical_success')
        event['event_id'] = digest(event)
        counts['auxiliary_updates'] += actual_steps
    return dict(receipt=receipt,event=event,counts_after=counts,
                aux_optimizer_state=independent.state_dict(),
                historical_AUX_events_unchanged=ledger,
                append_event_and_publish_is_callers_responsibility=True)
