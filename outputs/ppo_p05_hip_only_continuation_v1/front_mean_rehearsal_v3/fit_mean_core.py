"""Explicit finite SGD on existing mean rows; no CLI or import-time work.

All proposals are temporary. A rejected proposal is discarded in full and
stops the finite run. Only the last accepted mean-row leaf is copied back.
The caller must validate source/data/inspection/budget before authorization.
"""
from dataclasses import asdict
from copy import deepcopy
import math
import front_mean_rehearsal as method

torch, training, require = method.torch, method.training, method.require
SCHEMA = method.SCHEMA + '.fit'


def _compact_change(change):
    """Retain per-channel magnitudes/KL without duplicating every source row."""
    result = {}
    for name, values in change.items():
        x = torch.as_tensor(values, dtype=torch.float64)
        result[name + '_max_abs'] = float(x.abs().max())
        if x.ndim == 2 and x.shape[1] == 12:
            result[name + '_max_abs_per_channel'] = x.abs().amax(0).tolist()
            result[name + '_mean_per_channel'] = x.mean(0).tolist()
    return result


def _guard_summary(guard):
    return {'accepted_by_trust_only': guard['accepted_by_trust_only'],
            'rejection_reasons': list(guard['rejection_reasons']),
            'change_from_original': {name: _compact_change(values)
                for name, values in guard['change_from_original'].items()}}


def _loss_snapshot(actor, obs, targets, groups, leaf, reference, objective):
    with torch.no_grad():
        terms = method.loss_terms(actor, obs, targets, groups, leaf,
                                 reference['protection']['mean'], objective)
        result = {'phase_weighted_half_MSE': {k: float(v) for k, v in terms.items()}}
        for name in ('train', 'validation'):
            dist = method.distribution(actor, obs[name], leaf)
            delta = dist['mean'] - targets[name]
            request_delta = dist['request'] - dist['caps'] * targets[name].tanh()
            result[name] = {'raw_MSE': float(delta.square().mean()),
                'raw_MAE_per_channel': delta.abs().mean(0).tolist(),
                'REQUEST_MAE_per_channel': request_delta.abs().mean(0).tolist()}
        train = method.distribution(actor, obs['train'], leaf)
        result['train_by_phase'] = {}
        for phase, ids in groups.items():
            delta = train['mean'][ids] - targets['train'][ids]
            result['train_by_phase'][phase] = {
                'rows': len(ids), 'raw_MSE': float(delta.square().mean()),
                'raw_MAE_per_channel': delta.abs().mean(0).tolist()}
    return result


def fit_mean_head(runner, data, *, objective, budget, authorized=False):
    require(authorized is True, 'explicit admitted finite mean-head AUX authorization required')
    objective.validate(); budget.validate()
    actor, critic = runner.alg.actor, runner.alg.critic
    layer = method.final_layer(actor)
    require(type(critic.obs_normalizer) is torch.nn.Identity, 'critic must retain Identity normalization')
    require(runner.alg.storage.step == 0 and runner.alg.transition.actions is None,
            'AUX requires empty fresh PPO rollout and no pending action')
    obs, targets, groups = method.dataset(actor, data)
    saved = {key: value.detach().clone() for key, value in actor.state_dict().items()}
    saved_grads = {key: None if value.grad is None else value.grad.detach().clone()
                   for key, value in actor.named_parameters()}
    saved_critic = deepcopy(critic.state_dict())
    saved_critic_grads = {key: None if value.grad is None else value.grad.detach().clone()
                          for key, value in critic.named_parameters()}
    saved_optimizer = deepcopy(runner.alg.optimizer.state_dict())
    saved_normalizers = (deepcopy(actor.obs_normalizer), deepcopy(critic.obs_normalizer))
    saved_configs = (deepcopy(runner.cfg), deepcopy(runner._semantic_runner_config))
    hashes = {'actor': training.parameter_hash(actor), 'critic': training.parameter_hash(critic),
              'optimizer': training.state_hash(runner.alg.optimizer.state_dict()),
              'normalizers': training.state_hash(training._normalizers(runner))}
    lr = (runner.alg.learning_rate, training.optimizer_learning_rate(runner))
    rng = training.capture_training_rng_state(seed=runner.cfg['seed'])
    try:
        reference = method.baselines(actor, obs)
        require(torch.equal(reference['train']['mean'], actor(obs['train'], stochastic_output=False)),
                'source mean differs from official deterministic policy')
        leaf = method.selected_leaf(actor).requires_grad_(True)
        initial_leaf = leaf.detach().clone()
        before = _loss_snapshot(actor, obs, targets, groups, leaf, reference, objective)
        records = []; stop = 'finite_budget_exhausted'
        for attempt in range(1, budget.max_attempts + 1):
            terms = method.loss_terms(actor, obs, targets, groups, leaf,
                                     reference['protection']['mean'], objective)
            loss = terms['total']
            gradient, = torch.autograd.grad(loss, (leaf,))
            require(bool(torch.isfinite(loss)) and bool(torch.isfinite(gradient).all()),
                    'nonfinite objective/gradient; no actor update is permitted')
            row = {'attempt': attempt, 'learning_rate': budget.learning_rate,
                   'loss_before': float(loss.detach()),
                   'loss_terms_before': {key: float(value.detach()) for key, value in terms.items()},
                   'selected_gradient_l2': float(gradient.norm())}
            previous = leaf.detach().clone()
            candidate = previous - budget.learning_rate * gradient.detach()
            reasons = []
            try:
                require(bool(torch.isfinite(candidate).all()), 'nonfinite mean-head candidate')
                guard = method.guard_candidate(actor, obs, reference, candidate, budget=budget)
                row['trust'] = _guard_summary(guard)
                reasons.extend(guard['rejection_reasons'])
                after_candidate = _loss_snapshot(actor, obs, targets, groups, candidate, reference, objective)
                row['candidate'] = after_candidate
                next_loss = after_candidate['phase_weighted_half_MSE']['total']
                if not math.isfinite(next_loss) or next_loss > row['loss_before']:
                    reasons.append('explicit_total_objective_increased_or_nonfinite')
                if torch.equal(candidate, previous):
                    reasons.append('no_representable_parameter_change')
            except (ValueError, RuntimeError, OverflowError) as exc:
                reasons.append('unrepresentable_candidate')
                row['candidate_error'] = str(exc)
            row.update(accepted=not reasons, rejection_reasons=reasons)
            if reasons:
                # Neither live actor nor accepted leaf was changed by proposal.
                require(torch.equal(leaf.detach(), previous), 'rejected proposal altered accepted leaf')
                row['rejected_full3084_leaf_discarded'] = True
                stop = 'first_candidate_rejected_then_stop_no_LR_search'
            else:
                leaf = candidate.detach().requires_grad_(True)
            records.append(row)
            if reasons:
                break
        require(training.parameter_hash(actor) == hashes['actor'], 'proposal changed live actor')
        with torch.no_grad():
            layer.weight[:12].copy_(leaf[:, :256])
            layer.bias[:12].copy_(leaf[:, 256])
        for key, value in actor.state_dict().items():
            if key in ('mlp.4.weight', 'mlp.4.bias'):
                require(torch.equal(value[12:], saved[key][12:]), 'frozen log-sigma rows changed')
            else:
                require(torch.equal(value, saved[key]), 'non-selected actor state changed: ' + key)
        for name, value in obs.items():
            actual = method.distribution(actor, value)
            expected = method.distribution(actor, value, leaf)
            require(torch.equal(actual['mean'], expected['mean']), 'copied candidate mean mismatch')
            require(torch.equal(reference[name]['sigma'], actual['sigma'])
                    and torch.equal(reference[name]['log_sigma'], actual['log_sigma']),
                    'same-input sigma changed: ' + name)
        require(training.parameter_hash(critic) == hashes['critic']
                and training.state_hash(runner.alg.optimizer.state_dict()) == hashes['optimizer']
                and training.state_hash(training._normalizers(runner)) == hashes['normalizers'],
                'critic/PPO Adam/normalizer state changed')
        require(type(actor.obs_normalizer) is type(critic.obs_normalizer) is torch.nn.Identity
                and runner.cfg == saved_configs[0] and runner._semantic_runner_config == saved_configs[1],
                'normalizer type or source runner configuration changed')
        require((runner.alg.learning_rate, training.optimizer_learning_rate(runner)) == lr
                and training.capture_training_rng_state(seed=runner.cfg['seed']) == rng,
                'PPO LR or full CPU/CUDA RNG changed')
        for key, value in actor.named_parameters():
            require(value.grad is None if saved_grads[key] is None else value.grad is not None
                    and torch.equal(value.grad, saved_grads[key]), 'live actor gradients changed')
        for key, value in critic.named_parameters():
            require(value.grad is None if saved_critic_grads[key] is None else value.grad is not None
                    and torch.equal(value.grad, saved_critic_grads[key]), 'live critic gradients changed')
        require(runner.alg.storage.step == 0 and runner.alg.transition.actions is None,
                'AUX populated PPO storage')
        changed = int(torch.count_nonzero(leaf.detach() - initial_leaf))
        accepted = sum(row['accepted'] for row in records)
        require((accepted == 0 and changed == 0) or (accepted > 0 and 0 < changed <= 3084),
                'AUX count must correspond to real selected parameter change')
        final_guard = method.guard_candidate(actor, obs, reference, leaf.detach(), budget=budget)
        require(final_guard['accepted_by_trust_only'], 'final candidate violates original-source trust')
        actual_phases = sorted(set(int(v) + 1 for v in obs['protection']['policy'][:, :13].argmax(-1)))
        return {'schema': SCHEMA, 'objective': asdict(objective), 'budget': asdict(budget),
            'steps': records, 'accepted_auxiliary_updates': accepted,
            'attempted_auxiliary_optimizer_steps': len(records), 'stop_reason': stop,
            'before': before, 'after': _loss_snapshot(actor, obs, targets, groups, None, reference, objective),
            'final_change_from_original': _guard_summary(final_guard)['change_from_original'],
            'optimized_parameters': method.PARAMETERS.copy(), 'optimized_scalar_count': 3084,
            'actually_changed_scalar_count': changed,
            'auxiliary_optimizer': 'independent finite constant-LR SGD; no momentum or weight decay',
            'same_input_sigma_exact': True, 'P03plus_mean_bitwise_invariance_claimed': False,
            'affected_mean_phases': list(range(1, 14)), 'positive_label_phases': ['P01', 'P02'],
            'actual_protection_phases': actual_phases,
            'missing_actual_protection_phases': [i for i in range(3, 14) if i not in actual_phases],
            'P01_independent_validation_rows': 0,
            'validation_target_loss_is_reported_not_an_accept_rate_gate': True,
            'actor_parameter_sha256_before': hashes['actor'],
            'actor_parameter_sha256_after': training.parameter_hash(actor),
            'PPO_Adam_preserved_sha256': hashes['optimizer'], 'PPO_LR_preserved': lr[0],
            'training_rng_preserved': True, 'fresh_PPO_rollout_required': True,
            'kernel_or_sigma_rule_changed': False, 'MDP_or_control_changed': False,
            'PPO_decisions_added': 0, 'PPO_updates_added': 0, 'PPO_optimizer_steps_added': 0,
            'teacher_deployed': False, 'physical_success_claimed': False,
            'same_input_only_not_same_future_state_trajectory': True}
    except BaseException:
        # A failed auxiliary process must not leave RNG, gradients or training
        # state partially advanced, even when the failure detector itself is
        # exercising an injected unexpected mutation. No checkpoint is saved.
        actor.obs_normalizer, critic.obs_normalizer = saved_normalizers
        actor.load_state_dict(saved, strict=True)
        critic.load_state_dict(saved_critic, strict=True)
        runner.alg.optimizer.load_state_dict(saved_optimizer)
        runner.alg.learning_rate = lr[0]
        runner.cfg, runner._semantic_runner_config = saved_configs
        for module, gradients in ((actor, saved_grads), (critic, saved_critic_grads)):
            for key, value in module.named_parameters():
                value.grad = None if gradients[key] is None else gradients[key].detach().clone()
        runner.alg.storage.step = 0
        runner.alg.transition.actions = None
        training.restore_training_rng_state(rng, expected_seed=runner.cfg['seed'])
        raise
