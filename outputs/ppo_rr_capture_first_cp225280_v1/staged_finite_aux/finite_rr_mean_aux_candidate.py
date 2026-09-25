"""Optional finite RR mean-row AUX. Prepared only; not a route or publisher.

All imports are stdlib until fit_rr_mean_rows(..., isaac_stopped=True).
No file/model publication, PPO step, physics, reward edit or teacher success claim.
Parent owns cold runtime migration, unique checkpoint publication and fresh PPO.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
from pathlib import Path
import traceback

ROOT = Path(__file__).resolve().parents[3]
SOURCE_RUN_NAME = 'train_tracking_fixed512_1e10d39'
SOURCE_HEAD = '1e10d39c9a80c017cb7e4a0035d00c40a5b4dd81'
RR = (6,7)


def validate_local_auxiliary_events(events, counts):
    """Shared stdlib append-only ledger validation; never discard extra fields."""
    if events is None:
        events = []
    if not isinstance(events,list):
        raise ValueError('local_auxiliary_events must be a list')
    expected = counts.get('auxiliary_updates')
    if type(expected) is not int or expected < 0:
        raise ValueError('auxiliary_updates must be a nonnegative integer')
    seen,actual = set(),0
    for event in events:
        if not isinstance(event,dict) or event.get('schema') != 'wlr50_clean.finite_rr_mean_row_aux_event.v1':
            raise ValueError('unsupported local AUX event schema')
        identity = event.get('event_id')
        if (not isinstance(identity,str) or len(identity) != 64
                or any(c not in '0123456789abcdef' for c in identity) or identity in seen):
            raise ValueError('invalid/duplicate local AUX event id')
        steps,accepted = event.get('actual_optimizer_steps'),event.get('accepted_steps')
        if (type(steps) is not int or not 1 <= steps <= 64 or type(accepted) is not int
                or not 0 <= accepted <= steps or type(event.get('PPO_credit')) is not int
                or event['PPO_credit'] != 0):
            raise ValueError('invalid local AUX step/acceptance/PPO accounting')
        seen.add(identity)
        actual += steps
    if actual != expected:
        raise ValueError('AUX event actual optimizer steps disagree with auxiliary_updates')
    return copy.deepcopy(events)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def canonical_sha(data):
    return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':'),
        allow_nan=False).encode()).hexdigest()


def _failure_json_safe(value):
    """Preserve nonfinite failed candidates explicitly, without invalid JSON."""
    if isinstance(value,dict):
        return {key:_failure_json_safe(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):
        return [_failure_json_safe(item) for item in value]
    if isinstance(value,float) and not math.isfinite(value):
        return {'nonfinite_numeric_evidence':repr(value)}
    return value


def recipe(*, steps=8, learning_rate=.001, max_shift_sigma=.25):
    if type(steps) is not int or not 1 <= steps <= 64:
        raise ValueError('finite AUX budget must be 1..64 actual independent steps')
    if type(learning_rate) not in (float,int) or not math.isfinite(learning_rate) or not 0 < learning_rate <= .01:
        raise ValueError('fixed small learning rate required')
    if type(max_shift_sigma) not in (float,int) or not math.isfinite(max_shift_sigma) or not 0 < max_shift_sigma <= 1.:
        raise ValueError('positive bounded conditional shift budget required')
    result = dict(schema='wlr50_clean.finite_rr_mean_row_aux_recipe.v1',
        steps=steps,learning_rate=float(learning_rate),max_shift_sigma=float(max_shift_sigma),
        optimizer='independent_SGD_momentum0_weight_decay0',loss='sigma_scaled_conditional_raw_smooth_l1',
        teacher_rows='first_opportunity_31_to41_inclusive',supervised_action_channels=[6,7],
        mutable='local_mean_final_linear_weight_rows6_7_and_bias6_7_only',
        sigma='unchanged',PPO_credit=0,default8_is_proposal_not_a_completed_experiment=True)
    return result


def _resolve(path):
    p = Path(path)
    return (p if p.is_absolute() else ROOT/p).resolve(strict=True)


def _checked_cp(path, checkpoint_sha256, manifest_sha256):
    path = _resolve(path)
    sidecar = path.with_name(path.stem+'_manifest.json')
    if sha(path) != checkpoint_sha256 or sha(sidecar) != manifest_sha256:
        raise ValueError('explicit sealed CP/sidecar SHA mismatch')
    meta = json.loads(sidecar.read_text())
    if (meta['checkpoint_sha256'] != checkpoint_sha256 or meta['rollout_empty'] is not True
            or meta['save_load_round_trip'] is not True
            or meta['schema'] != 'wlr50_clean.frozen_prior_rr_capture_checkpoint.v2'):
        raise ValueError('complete same448/v2 checkpoint required')
    return path,sidecar,meta


def prepare_success_package(run, *, expected_runtime, parent_checkpoint,
                            checkpoint_sha256, manifest_sha256):
    """Read a COMPLETE block only. Returns evidence, not approved training credit."""
    run = _resolve(run)
    manifest_path = run/'run_manifest.json'
    if not manifest_path.exists():
        raise ValueError('unsealed run: COMPLETE manifest required before AUX preparation')
    finished = json.loads(manifest_path.read_text())
    if finished.get('lifecycle') != 'COMPLETE' or finished.get('mode') != 'train':
        raise ValueError('unsealed/incomplete/non-training source rejected')
    started = json.loads((run/'run_manifest.started.json').read_text())
    if (run.name != SOURCE_RUN_NAME or started['runtime_contract'] != expected_runtime
            or expected_runtime['source_git_commit'] != SOURCE_HEAD
            or expected_runtime['local_contract']['observation_dimension'] != 448
            or expected_runtime['local_contract']['source_tracking_owner_revision'] !=
                'pending_source_tracking_inheritance_v1'):
        raise ValueError('this candidate binds only the actual current448 fixed-tracking success')
    path,sidecar,parent = _checked_cp(parent_checkpoint,checkpoint_sha256,manifest_sha256)
    pointer = finished['result']
    if (path != _resolve(pointer['checkpoint']) or checkpoint_sha256 != pointer['checkpoint_sha256']
            or manifest_sha256 != pointer['manifest_sha256'] or parent['runtime_contract'] != expected_runtime):
        raise ValueError('AUX parent must be this sealed completed block checkpoint/runtime')
    collected_path = _resolve(started['checkpoint'])
    collected_sidecar = collected_path.with_name(collected_path.stem+'_manifest.json')
    collected = json.loads(collected_sidecar.read_text())
    _checked_cp(collected_path,collected['checkpoint_sha256'],sha(collected_sidecar))
    if collected['runtime_contract'] != expected_runtime:
        raise ValueError('successful source collection control differs')
    decisions_path = run/'decisions.jsonl'
    first = []
    with decisions_path.open('rb') as stream:
        for line in stream:
            if not line.endswith(b'\n'):
                raise ValueError('incomplete row in purported sealed source')
            row = json.loads(line)
            if row['kind'] == 'frozen_prior_prefix':
                if row['PPO_credit'] != 0:
                    raise ValueError('prefix credit contamination')
                if first:
                    raise ValueError('episode switched before the verified first successful terminal')
                continue
            if row['kind'] != 'activated_on_policy' or row['PPO_credit'] != 1:
                raise ValueError('unexpected source intervention/credit')
            p,info = row['policy_request'],row['step_info']
            before,after = row['before_local']['metrics'],info['rr_capture_local']['metrics']
            obs = row['observation']
            if (len(obs) != 448 or not all(math.isfinite(x) for x in obs)
                    or p['schema'] != 'wlr50_clean.actual_rr_capture_local_request.v2'
                    or p['policy_version'] != 'frozen_cp225280_rr_capture_local_history_v2'
                    or p.get('independent_diagnostic') or p['history_kernel_applications'] != 1
                    or p['sampling_draws'] != 1 or p['extra_random_draws'] != 0
                    or p['selected_raw_full12'] != info['raw_policy_action_full12']
                    or row['old_logp'][0] != p['selected_raw_log_probability']
                    or info['actuator_target_effect_audit']['phase_mask_full12'] != [1]*12):
                raise ValueError('raw/observation/current-version evidence mismatch')
            if first and first[-1]['end_tick'] != before['tick']:
                raise ValueError('noncontinuous successful source')
            first.append(dict(global_decision=row['global_decision'],start_tick=before['tick'],
                end_tick=after['tick'],observation_full448=obs,
                actual_issued_raw_full12=p['selected_raw_full12'],
                collected_conditional_mean_full12=p['conditional_mean_full12'],
                collected_sigma_full12=p['active_conditional_std_full12'],
                history_center_full12=p['history_center_full12'],rho=p['rho'],
                before_metrics=before,after_metrics=after,
                after_hold_s=info['rr_capture_local']['hold_elapsed_s'],
                local_success=info['local_task_success'],
                final_target_evidence_not_teacher=info['actual_drive_target_full12'],
                source_nominal_evidence_not_teacher=info['nominal_action_full12'],
                native_dispatch_evidence=info['actuator_target_effect_audit']))
            if info['local_reward']['terminated']:
                break
    if (len(first) != 41 or not first[-1]['local_success'] or first[-1]['end_tick'] != 8312
            or first[-1]['after_hold_s'] < .5):
        raise ValueError('source is not the identified actual first41 capture/hold success')
    selected = first[30:41]
    if (selected[0]['start_tick'] != 8224 or selected[0]['global_decision'] != 227871
            or any(not x['before_metrics']['current_attempt_capture_eligible'] for x in selected)
            or any(x['after_metrics']['ground_contact'] for x in selected)
            or any(x['native_dispatch_evidence']['policy_headroom_evidence']['clipped_servo_indices']
                   for x in selected)):
        raise ValueError('candidate window/qualification/projection changed')
    result = dict(schema='wlr50_clean.current448_finite_aux_evidence.v1',
        runtime_contract=expected_runtime,source_run=str(run),
        bindings={'run_manifest':dict(path=str(manifest_path),sha256=sha(manifest_path)),
            'started_manifest':dict(path=str(run/'run_manifest.started.json'),sha256=sha(run/'run_manifest.started.json')),
            'full_decisions':dict(path=str(decisions_path),sha256=sha(decisions_path),bytes=decisions_path.stat().st_size),
            'collection_checkpoint':dict(path=str(collected_path),sha256=sha(collected_path)),
            'collection_manifest':dict(path=str(collected_sidecar),sha256=sha(collected_sidecar)),
            'parent_checkpoint':dict(path=str(path),sha256=checkpoint_sha256),
            'parent_manifest':dict(path=str(sidecar),sha256=manifest_sha256)},
        parent_state_hashes=parent['state_hashes'],parent_counts=parent['counts'],
        parent_learning_rate=parent['learning_rate'],
        selected_rows=selected,reference_observations_full448=[x['observation_full448'] for x in first],
        scope='11 real terminal-approach/contact/hold rows, not proof of59mm entry or naturalP01 determinism',
        on_policy_reuse_forbidden=True,automatic_success_label=False,teacher_target='actual_issued_conditional_raw_not_FINAL')
    result['package_sha256'] = canonical_sha(result)
    return result


def fit_rr_mean_rows(runner, package, sealed_recipe, *, isaac_stopped=False):
    """Bounded independent AUX only. Returns ALL attempts; never publishes."""
    if not isaac_stopped:
        raise ValueError('explicit Isaac exit acknowledgement required before any Torch import')
    if package.get('package_sha256') != canonical_sha({k:v for k,v in package.items() if k!='package_sha256'}):
        raise ValueError('AUX evidence package digest mismatch')
    for item in package['bindings'].values():
        if sha(item['path']) != item['sha256']:
            raise ValueError('sealed AUX source binding changed')
    if sealed_recipe != recipe(steps=sealed_recipe['steps'],learning_rate=sealed_recipe['learning_rate'],
                               max_shift_sigma=sealed_recipe['max_shift_sigma']):
        raise ValueError('unknown/unsealed AUX recipe')
    # Deliberately no model/optimizer imports before guards above.
    import torch
    from wlr50_clean.ppo.semantic_training import state_hash
    from wlr50_clean.ppo.rl_library_wrapper import capture_training_rng_state, restore_training_rng_state
    actor,critic,adam = runner.alg.actor,runner.alg.critic,runner.alg.optimizer
    if (actor.observation_layout != 'role439_rr_capture_local_v2'
            or runner.alg.storage.step != 0 or runner.alg.transition.actions is not None):
        raise ValueError('same448 complete update/empty rollout required')
    if runner.checkpoint_load_provenance['checkpoint_sha256'] != package['bindings']['parent_checkpoint']['sha256']:
        raise ValueError('runner was not loaded from the explicitly bound AUX parent')
    current = runner.alg.save()
    if any(state_hash(current[k]) != h for k,h in package['parent_state_hashes'].items()):
        raise ValueError('actual loaded actor/critic/Adam differs from AUX parent')
    if (runner.alg.learning_rate != package['parent_learning_rate']
            or any(g['lr'] != runner.alg.learning_rate for g in adam.param_groups)):
        raise ValueError('actual PPO LR mismatch')
    actor.assert_frozen_state(adam)
    modules = [(name,mod) for name,mod in actor.mlp.named_modules() if isinstance(mod,torch.nn.Linear)]
    name,last = modules[-1]
    if last.out_features != 24 or last.bias is None:
        raise ValueError('expected actual24-row local mean12/std12 final Linear')
    weight_key,bias_key = 'mlp.'+name+'.weight','mlp.'+name+'.bias'
    actor_before = {k:v.detach().clone() for k,v in actor.state_dict().items()}
    flags = [(p,p.requires_grad) for p in actor.parameters()]
    if any(p.grad is not None for p,_ in flags):
        raise ValueError('cold AUX requires no outstanding actor gradients')
    ids_before = {n:id(p) for n,p in actor.named_parameters()}
    adam_ids = tuple(tuple(id(p) for p in g['params']) for g in adam.param_groups)
    rng_before = capture_training_rng_state(seed=1001)
    before_hashes = {k:state_hash(v) for k,v in current.items()}
    before_cache = actor._last_forward_evidence
    attempts,accepted,optimizer_steps = [],0,0
    status,reason = 'COMPLETE','budget_exhausted'
    committed = False
    rr_weight = torch.nn.Parameter(last.weight.detach()[6:8].clone())
    rr_bias = torch.nn.Parameter(last.bias.detach()[6:8].clone())
    independent = torch.optim.SGD([rr_weight,rr_bias],lr=sealed_recipe['learning_rate'],
        momentum=0.,weight_decay=0.)
    reference = torch.tensor(package['reference_observations_full448'],dtype=last.weight.dtype,device=last.weight.device)
    obs = torch.tensor([r['observation_full448'] for r in package['selected_rows']],dtype=last.weight.dtype,device=last.weight.device)
    target = torch.tensor([r['actual_issued_raw_full12'][6:8] for r in package['selected_rows']],dtype=last.weight.dtype,device=last.weight.device)
    def override():
        return {weight_key:torch.cat((actor_before[weight_key][:6],rr_weight,actor_before[weight_key][8:])),
                bias_key:torch.cat((actor_before[bias_key][:6],rr_bias,actor_before[bias_key][8:]))}
    def forward(values):
        return torch.func.functional_call(actor,override(),({'policy':values},),
            dict(stochastic_output=False),strict=False)
    def row_state():
        return {'weight':rr_weight.detach().cpu().tolist(),'bias':rr_bias.detach().cpu().tolist()}
    original_mean = None
    grouped_fit = {}
    try:
        for parameter,_ in flags:
            parameter.requires_grad_(False)
        with torch.no_grad():
            original_reference = actor({'policy':reference},stochastic_output=False).clone()
            original_reference_std = actor._last_forward_evidence['head_log_std'].exp().clone()
            original_mean = actor({'policy':obs},stochastic_output=False).clone()
            fixed_sigma = actor._last_forward_evidence['head_log_std'].exp()[:,6:8].clone()
            initial_function = forward(obs)
            if not torch.equal(initial_function,original_mean):
                raise RuntimeError('independent row substitution changed the initial function')
        def loss_fn(mu):
            return torch.nn.functional.smooth_l1_loss(mu[:,6:8]/fixed_sigma,target/fixed_sigma)
        initial_loss = float(loss_fn(original_mean))
        accepted_state = (rr_weight.detach().clone(),rr_bias.detach().clone())
        previous_loss = initial_loss
        for number in range(1,sealed_recipe['steps']+1):
            independent.zero_grad(set_to_none=True)
            loss = loss_fn(forward(obs))
            gradients = torch.autograd.grad(loss,(rr_weight,rr_bias))
            rr_weight.grad,rr_bias.grad = gradients
            grad_norm = float(torch.sqrt(sum(g.detach().square().sum() for g in gradients)))
            independent.step()
            optimizer_steps += 1
            with torch.no_grad():
                candidate_mu = forward(obs).clone()
                candidate_loss = float(loss_fn(candidate_mu))
                reference_mu = forward(reference).clone()
                reference_std = actor._last_forward_evidence['head_log_std'].exp().clone()
                other = [0,1,2,3,4,5,8,9,10,11]
                exact_others = torch.equal(reference_mu[:,other],original_reference[:,other])
                exact_std = torch.equal(reference_std,original_reference_std)
                shift = float(((reference_mu[:,6:8]-original_reference[:,6:8])/
                    original_reference_std[:,6:8]).abs().max())
            accepted_step = (math.isfinite(candidate_loss) and math.isfinite(shift)
                and candidate_loss <= previous_loss and shift <= sealed_recipe['max_shift_sigma']
                and exact_others and exact_std)
            attempts.append(dict(attempt=number,loss_before=float(loss),loss_after=candidate_loss,
                gradient_norm=grad_norm,reference_max_mean_shift_sigma=shift,
                other10_conditional_mean_exact=exact_others,std_exact=exact_std,
                accepted=accepted_step,candidate_RR_rows=row_state()))
            if not accepted_step:
                status,reason = 'STOPPED_CONSTRAINT','loss/nonfinite/trust/frozen-output constraint'
                with torch.no_grad():
                    rr_weight.copy_(accepted_state[0]); rr_bias.copy_(accepted_state[1])
                break  # Attempt remains counted and its candidate rows retained.
            accepted += 1
            previous_loss = candidate_loss
            accepted_state = (rr_weight.detach().clone(),rr_bias.detach().clone())
        # Original optimizer/parameter objects are never replaced.
        with torch.no_grad():
            last.weight[6:8].copy_(rr_weight); last.bias[6:8].copy_(rr_bias)
        committed = bool(accepted)
    except BaseException:
        status,reason = 'FAILED',traceback.format_exc()
        # No earlier candidate was written to the real actor before final commit.
        # Preserve failed attempt metadata/rows instead of claiming zero attempts.
        attempts.append(dict(attempt=len(attempts)+1,failed=True,error=reason,
            candidate_RR_rows=row_state(),actual_optimizer_steps_at_failure=optimizer_steps))
    finally:
        for parameter,flag in flags:
            parameter.requires_grad_(flag)
        actor._last_forward_evidence = before_cache
        restore_training_rng_state(rng_before,expected_seed=1001)
    # Audit the real committed actor, not the last rejected temporary candidate.
    try:
        with torch.no_grad():
            final_mean = actor({'policy':obs},stochastic_output=False).clone()
            if original_mean is not None:
                for label,is_top in (('AIR_before',False),('TOP_before',True)):
                    indices = [i for i,row in enumerate(package['selected_rows'])
                               if bool(row['before_metrics']['current_top_contact']) is is_top]
                    if indices:
                        before_mu,after_mu = original_mean[indices,6:8],final_mean[indices,6:8]
                        truth = target[indices]
                        grouped_fit[label] = dict(rows=len(indices),channels=['RR_hip','RR_knee'],
                            conditional_mean_before=before_mu.mean(0).cpu().tolist(),
                            actual_raw_target_mean=truth.mean(0).cpu().tolist(),
                            conditional_mean_after=after_mu.mean(0).cpu().tolist(),
                            RMSE_before=(before_mu-truth).square().mean(0).sqrt().cpu().tolist(),
                            RMSE_after=(after_mu-truth).square().mean(0).sqrt().cpu().tolist())
    except BaseException:
        status,reason = 'FAILED_AUDIT',traceback.format_exc()
    finally:
        actor._last_forward_evidence = before_cache
        restore_training_rng_state(rng_before,expected_seed=1001)
    after = actor.state_dict()
    protected = {}
    for key,old in actor_before.items():
        if key in (weight_key,bias_key):
            mask = [i for i in range(24) if i not in RR]
            protected[key] = torch.equal(old[mask],after[key][mask])
        else:
            protected[key] = torch.equal(old,after[key])
    after_hashes = {k:state_hash(v) for k,v in runner.alg.save().items()}
    invariants = dict(protected_actor_exact=all(protected.values()),
        critic_exact=after_hashes['critic_state_dict']==before_hashes['critic_state_dict'],
        PPO_Adam_exact=after_hashes['optimizer_state_dict']==before_hashes['optimizer_state_dict'],
        actor_parameter_objects_exact=ids_before=={n:id(p) for n,p in actor.named_parameters()},
        PPO_Adam_object_and_order_exact=runner.alg.optimizer is adam and
            adam_ids==tuple(tuple(id(p) for p in g['params']) for g in adam.param_groups),
        PPO_LR_exact=runner.alg.learning_rate==package['parent_learning_rate'] and
            all(g['lr']==package['parent_learning_rate'] for g in adam.param_groups),
        full_RNG_exact=capture_training_rng_state(seed=1001)==rng_before)
    if not all(invariants.values()):
        status,reason = 'FAILED_INVARIANT','do not publish; preserve receipt and investigate'
    try:
        actor.assert_frozen_state(adam)
    except BaseException:
        status,reason = 'FAILED_INVARIANT',traceback.format_exc()
    receipt = dict(schema='wlr50_clean.finite_rr_mean_row_aux_receipt.v1',status=status,reason=reason,
        source_package_sha256=package['package_sha256'],recipe=sealed_recipe,recipe_sha256=canonical_sha(sealed_recipe),
        parent_checkpoint=package['bindings']['parent_checkpoint'],parent_counts=package['parent_counts'],
        new_policy_decisions=0,new_PPO_updates=0,new_PPO_Adam_steps=0,
        AUX_optimizer_steps=optimizer_steps,AUX_attempt_records=len(attempts),
        AUX_accepted_steps=accepted,actor_rows_committed=committed,attempts=attempts,
        AIR_TOP_before_fit=grouped_fit,
        accepted_RR_rows=row_state(),original_RR_rows={
            'weight':actor_before[weight_key][6:8].cpu().tolist(),
            'bias':actor_before[bias_key][6:8].cpu().tolist()},
        mutable_parameter_names=[weight_key,bias_key],mutable_rows=[6,7],
        independent_optimizer_state=independent.state_dict(),state_hashes_before=before_hashes,
        state_hashes_after=after_hashes,invariants=invariants,
        RNG_sha256=canonical_sha(rng_before),
        observation_action_distribution_versions_unchanged=True,automatic_success_claim=False,
        fresh_on_policy_collection_required=True,natural_P01_physical_evaluation_required=True,
        source_scope=package['scope'],publication='parent must publish unique AUX-lineage checkpoint; helper publishes nothing')
    receipt = _failure_json_safe(receipt)
    event = None
    counts_after = dict(package['parent_counts'])
    if optimizer_steps:
        event = dict(schema='wlr50_clean.finite_rr_mean_row_aux_event.v1',
            actual_optimizer_steps=optimizer_steps,accepted_steps=accepted,
            source_checkpoint=package['bindings']['parent_checkpoint'],
            source_manifest=package['bindings']['parent_manifest'],
            source_data_sha256=package['bindings']['full_decisions']['sha256'],
            package_sha256=package['package_sha256'],
            source_runtime_content_sha256=package['runtime_contract']['runtime_content_sha256'],
            source_runtime_head=package['runtime_contract']['source_git_commit'],
            recipe_sha256=canonical_sha(sealed_recipe),receipt_sha256=canonical_sha(receipt),
            purpose='finite_RR_conditional_mean_terminal_capture_only_not_full_task',
            status=status,actor_rows_committed=committed,PPO_credit=0,
            auxiliary_optimizer='independent_SGD_not_original_PPO_Adam')
        event['event_id'] = canonical_sha(event)
        counts_after['auxiliary_updates'] += optimizer_steps
    return dict(event=event,receipt=receipt,aux_optimizer_state=independent.state_dict(),
                counts_after=counts_after)
