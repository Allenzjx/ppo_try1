"""CPU tests for same422 cooperative sigma; synthetic updates are not robot credit."""
import itertools
import json
from copy import deepcopy
from collections import Counter
from pathlib import Path
import pytest
import torch
from tensordict import TensorDict
from wlr50_clean.ppo.semantic_rear_cooperative_prep_sigma import (
    cooperative_prep_effective_log_std, cooperative_prep_multipliers)
from wlr50_clean.ppo.semantic_rear_cooperative_prep_actor import SemanticRearCooperativePrepHistoryMLPModel
from wlr50_clean.ppo.semantic_rear_cooperative_prep_profile import (
    cooperative_prep_policy_contract, COOPERATIVE_PREP_POLICY)
from wlr50_clean.ppo.semantic_p02_progress_actor import (
    SemanticP02ProgressHistoryMLPModel, p02_progress_effective_log_std)
from wlr50_clean.ppo.semantic_p02_progress_profile import P02_PROGRESS_OBSERVATION_LAYOUT
from wlr50_clean.ppo.semantic_receiving_wheel_sigma import receiving_wheel_effective_log_std


@pytest.fixture(autouse=True)
def cpu_scope():
    assert not torch.cuda.is_available(), 'run with CUDA_VISIBLE_DEVICES=-1'
    state, threads = torch.get_rng_state(), torch.get_num_threads()
    torch.manual_seed(20260924)
    torch.set_num_threads(1)
    yield
    torch.set_rng_state(state)
    torch.set_num_threads(threads)


def latent(phase=8, count=2):
    x = torch.zeros(count, 422)
    x[:, phase] = 1.
    x[:, 20] = .025
    x[:, 158:158+phase] = 1.
    x[:, 195:207] = torch.linspace(-.7, .9, 12)
    return x


def obs(x):
    return TensorDict({'policy': x, 'critic': x.clone()}, batch_size=list(x.shape[:-1]))


def model(x, cls):
    return cls(obs(x), {'actor': ['policy']}, 'actor', 12,
        observation_layout=P02_PROGRESS_OBSERVATION_LAYOUT,
        exploration_std_temperature=.25, obs_normalization=False,
        distribution_cfg={'class_name': 'HeteroscedasticGaussianDistribution',
                          'std_type': 'log', 'init_std': .15})


@pytest.mark.parametrize('carry,reachable,prep,receiving', [
    (False, False, False, False), (True, False, False, False),
    (True, True, False, False), (False, False, True, True), (True, True, True, True)])
def test_tensor_matches_scalar_parent_totals_and_preserves_old_unselected_channels(carry, reachable, prep, receiving):
    x = latent(9 if receiving else 8)
    x[:, 410], x[:, 404], x[:, 412], x[:, 157] = float(carry), float(reachable), float(prep), float(receiving)
    head = torch.linspace(-3., -.2, x.shape[0]*12).reshape(-1, 12)
    saved, rng = x.clone(), torch.get_rng_state()
    parent, parent_ev = receiving_wheel_effective_log_std(head, x[:, :372], .25)
    old, _ = p02_progress_effective_log_std(head, x, .25)
    new, ev = cooperative_prep_effective_log_std(head, x, .25)
    expected, _ = cooperative_prep_multipliers(rr_carry_capture=carry,
        rr_top_reachable=reachable, rl_prep_transfer=prep,
        receiving_continuation_active=receiving)
    assert bool((parent_ev['receiving_continuation_active'] == receiving).all())
    torch.testing.assert_close(new, parent+torch.tensor(expected).log(), rtol=0, atol=0)
    assert torch.equal(new[:, [0, 2, 6, 7, 9, 10, 11]], old[:, [0, 2, 6, 7, 9, 10, 11]])
    assert torch.equal(x, saved) and torch.equal(torch.get_rng_state(), rng)
    assert ev['cooperative_support_transfer_permission_modified'] is False


@pytest.mark.parametrize('phase', range(6))
def test_early_front_offgate_sigma_bitwise_equal_old422(phase):
    x = latent(phase)
    head = torch.linspace(-3., -.2, x.shape[0]*12).reshape(-1, 12)
    old, _ = p02_progress_effective_log_std(head, x, .25)
    new, _ = cooperative_prep_effective_log_std(head, x, .25)
    assert torch.equal(new, old)


def test_same_state_dict_and_full_adam_survive_without_mean_change():
    x = latent(); x[:, 410] = x[:, 404] = 1.
    old = model(x, SemanticP02ProgressHistoryMLPModel)
    old_optimizer = torch.optim.Adam(old.parameters(), lr=.00003)
    old(obs(x)).square().mean().backward()
    old_optimizer.step()  # CPU synthetic fixture only; not physical PPO credit.
    state, adam = deepcopy(old.state_dict()), deepcopy(old_optimizer.state_dict())
    new = model(x, SemanticRearCooperativePrepHistoryMLPModel)
    new.load_state_dict(state, strict=True)
    new_optimizer = torch.optim.Adam(new.parameters(), lr=.00003)
    new_optimizer.load_state_dict(adam)
    assert [(n, tuple(p.shape)) for n, p in old.named_parameters()] == [
        (n, tuple(p.shape)) for n, p in new.named_parameters()]
    assert old.state_dict().keys() == new.state_dict().keys()
    assert all(torch.equal(v, new.state_dict()[k]) for k, v in state.items())
    assert torch.equal(old(obs(x)), new(obs(x)))
    other = new_optimizer.state_dict()
    assert other['param_groups'] == adam['param_groups'] and other['state'].keys() == adam['state'].keys()
    for parameter, row in adam['state'].items():
        for key, value in row.items():
            assert torch.equal(value, other['state'][parameter][key]) if isinstance(value, torch.Tensor) else value == other['state'][parameter][key]


def test_actual_actor_sample_log_probability_uses_shared_kernel_and_ratio_one_without_update():
    x = latent(); x[:, 410] = x[:, 404] = 1.
    actor = model(x, SemanticRearCooperativePrepHistoryMLPModel)
    heads = []
    handle = actor.mlp.register_forward_hook(lambda _m, _a, output: heads.append(output.detach().clone()))
    selected = actor(obs(x), stochastic_output=True)
    handle.remove()
    assert len(heads) == 1
    expected_log_std, _ = cooperative_prep_effective_log_std(heads[0][..., 1, :], x, .25)
    torch.testing.assert_close(actor.output_std, expected_log_std.exp(), rtol=0, atol=0)
    old_logp = actor.get_output_log_prob(selected).detach()
    manual = torch.distributions.Normal(actor.output_mean, actor.output_std).log_prob(selected).sum(-1)
    torch.testing.assert_close(old_logp, manual)
    actor(obs(x), stochastic_output=True)
    current_logp = actor.get_output_log_prob(selected)
    torch.testing.assert_close(torch.exp(current_logp-old_logp), torch.ones_like(old_logp))


def test_contract_explicitly_versions_only_kernel_not_observations_or_control():
    contract = cooperative_prep_policy_contract(observation_layout=P02_PROGRESS_OBSERVATION_LAYOUT)
    assert contract['version'] == COOPERATIVE_PREP_POLICY
    assert contract['observation_dimension'] == 422
    assert contract['observation_layout'] == 'role419_p02_progress_v1'
    assert contract['deterministic_mean_change'] is False
    assert contract['control_or_reward_change'] is False
    assert contract['stochastic_kernel_change'] is True
    assert contract['support_transfer_permission_modified'] is False


def test_scalar_full_truth_table_preserves_other_channels_and_total_not_compound():
    for carry, reachable, prep, receiving in itertools.product((False, True), repeat=4):
        multiplier, evidence = cooperative_prep_multipliers(rr_carry_capture=carry,
            rr_top_reachable=reachable, rl_prep_transfer=prep,
            receiving_continuation_active=receiving)
        assert multiplier[6] == (4. if carry and reachable else 1.)
        allowed = (carry and reachable) or prep
        for i, total in ((3,16.), (1,2.), (4,2.), (5,8.)):
            assert multiplier[i] == (total if allowed else 1.)
        assert multiplier[8] == (2. if carry and reachable and not receiving else 1.)
        assert all(multiplier[i] == 1. for i in (0,2,7,9,10,11))
        assert evidence['support_transfer_permission_modified'] is False


def test_kernel_gradient_reaches_original_sigma_head_without_clamp_or_detach():
    x = latent(); x[:,410] = x[:,404] = 1.
    head = torch.full((2,12), -2., requires_grad=True)
    output, _ = cooperative_prep_effective_log_std(head, x)
    output.sum().backward()
    assert torch.equal(head.grad, torch.ones_like(head))


def runner_and_environment():
    from test_semantic_rear_policy_training_audit import Synthetic419Core
    from wlr50_clean.ppo import semantic_training as training
    class Synthetic422Core(Synthetic419Core):
        def observation(self, raw=(0.,)*12):
            return super().observation(raw)+(0.,0.,0.)
    env = training.SemanticRslAdapter(Synthetic422Core(), seed=1001, device='cpu')
    env.cfg['semantic_version'] = 'v3'
    runner, _ = training.construct_semantic_runner(env, seed=1001, device='cpu', initialize_actor=False,
        policy_version=COOPERATIVE_PREP_POLICY, observation_layout=P02_PROGRESS_OBSERVATION_LAYOUT)
    return runner, env


@pytest.mark.parametrize('stochastic', [False, True])
def test_production_request_audit_and_saved_exact_checkpoint_prefix(stochastic):
    from wlr50_clean.ppo import semantic_training as training
    from wlr50_clean.ppo import semantic_policy_distribution as distribution
    from wlr50_clean.ppo.semantic_checkpoint_prefix_policy import FrozenCheckpointPrefixPolicy
    runner, _ = runner_and_environment()
    actor = runner.alg.actor
    x = latent(count=1); x[:,410] = x[:,404] = 1.
    with torch.no_grad():
        raw, receipt = training.audited_history_policy_request(actor, obs(x),
            lambda: actor(obs(x), stochastic_output=stochastic), stochastic=stochastic)
    assert receipt['policy_version'] == COOPERATIVE_PREP_POLICY
    assert receipt['schema'] == 'wlr50_clean.actual_rear_cooperative_prep_request.v1'
    assert receipt['p02_progress_observed_features'] == [0.,0.,0.]
    assert receipt['cooperative_observed_rr_carry_capture'] is True
    assert receipt['cooperative_observed_rr_top_reachable'] is True
    assert receipt['cooperative_observed_rl_prep_transfer'] is False
    assert receipt['cooperative_parent_receiving_continuation_active'] is False
    assert receipt['cooperative_prep_allowed'] is True
    assert receipt['cooperative_fl_wheel_extra_active'] is True
    assert receipt['rear_local_sigma_multiplier_full12'] == [1.,2.,1.,16.,2.,8.,4.,1.,2.,1.,1.,1.]
    assert receipt['sampling_draws'] == int(stochastic)
    assert receipt['extra_random_draws'] == receipt['extra_model_forwards'] == 0
    policy = training._runner_policy_contract(runner)
    record = dict(checkpoint_path='synthetic_only.pt', checkpoint_sha256='a'*64,
        actor_parameter_sha256=training.parameter_hash(actor),
        source_global_policy_decisions=0, source_ppo_updates=0,
        policy_contract=policy, source_policy_contract=policy, effective_policy_contract=policy,
        source_runtime_content_sha256='b'*64, effective_runtime_content_sha256='b'*64)
    prefix = FrozenCheckpointPrefixPolicy(actor, record)
    assert len(prefix(tuple(x[0].tolist()))) == 12
    assert distribution.supported_heteroscedastic_contract_version(policy) == COOPERATIVE_PREP_POLICY
    stale = deepcopy(record); stale['source_policy_contract']['version'] = 'rear_policy_p02_progress_history_v1'
    with pytest.raises(ValueError):
        FrozenCheckpointPrefixPolicy(actor, stale)


def test_synthetic_official_update_keeps_raw_likelihood_and_observed_sigma_gates(tmp_path):
    from wlr50_clean.ppo import semantic_training as training
    runner, env = runner_and_environment()
    contract = dict(experiment_id='rr_rl_timing_policy_learning_v1', semantic_version='v3',
        training_budgets=training.training_quantity_budgets('rr_rl_timing_policy_learning_v1'),
        evidence='synthetic CPU ABI only; zero real robot policy decisions or updates')
    result = training.train_semantic(runner, env, run_dir=tmp_path/'run', output_root=tmp_path/'out',
        stage='full_episode', decisions=128, contract=contract, seed=1001, checkpoint_interval_updates=1)
    assert result['actual_policy_decisions'] == 128 and result['ppo_updates_this_run'] == 1
    assert result['optimizer_steps_this_run'] == 20 and result['finite_nonzero_gradient_observed']
    rollout = torch.load(tmp_path/'run/rollouts/rollout_000001.pt', map_location='cpu', weights_only=False)
    rows = [json.loads(line) for line in (tmp_path/'run/residual_and_projection_audit.jsonl').read_text().splitlines()]
    for index, row in enumerate(rows):
        request = row['policy_request']
        assert request['policy_version'] == COOPERATIVE_PREP_POLICY
        assert request['selected_raw_full12'] == rollout['actions'][index,0].tolist() == row['raw_policy_action_full12']
        assert request['selected_raw_full12'] != row['applied_audit']['applied_action_full12']
        assert request['selected_raw_log_probability'] == rollout['actions_log_prob'][index,0].item()
        assert request['effective_sigma_full12'] == rollout['distribution_params'][1][index,0].tolist()
        assert request['cooperative_prep_allowed'] is True
        assert request['rear_local_sigma_multiplier_full12'][3] == 16.
    audit = json.loads((tmp_path/'run/rollouts/update_000001_likelihood.json').read_text())
    assert len(audit['minibatches']) == 20
    exposure = Counter()
    for batch in audit['minibatches']:
        assert 'cooperative' in batch['sigma_source']
        assert batch['cooperative_prep_allowed'] == [True]*32
        assert all(row[3] == 16. for row in batch['rear_local_sigma_multiplier_full12'])
        for indices in batch['rollout_flat_indices']:
            assert len(indices) == 1
            exposure[indices[0]] += 1
        for key in ('loss_gradient_wrt_network_mean_full12','loss_gradient_wrt_network_log_sigma_full12'):
            assert torch.isfinite(torch.tensor(batch[key])).all()
    assert exposure == Counter({i:5 for i in range(128)})
    fresh, _ = runner_and_environment()
    training.load_semantic_checkpoint(fresh, Path(result['checkpoints'][0]['checkpoint']), contract=contract, seed=1001)
    assert training.parameter_hash(fresh.alg.actor) == training.parameter_hash(runner.alg.actor)
    assert training.state_hash(fresh.alg.optimizer.state_dict()) == training.state_hash(runner.alg.optimizer.state_dict())
