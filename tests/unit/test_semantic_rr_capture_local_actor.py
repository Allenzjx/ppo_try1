"""Directed CPU tests; synthetic observations, no robot or on-policy credit."""
import copy
from itertools import chain
from unittest.mock import patch

import pytest
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from rsl_rl.algorithms import PPO
from rsl_rl.storage import RolloutStorage

from wlr50_clean.ppo import semantic_rr_capture_local_actor as module
from wlr50_clean.ppo.semantic_rear_owner_actor import SemanticRearOwnerRecoveryHistoryMLPModel
from wlr50_clean.ppo.semantic_rear_owner_profile import REAR_OWNER_OBSERVATION_LAYOUT


@pytest.fixture(autouse=True)
def cpu_rng():
    state, threads = torch.get_rng_state(), torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(225280)
    yield
    torch.set_rng_state(state)
    torch.set_num_threads(threads)


def observations(active=False, batch=1, phase=8):
    x = torch.zeros(batch, 447)
    x[:, phase] = 1.
    x[:, 20] = .02
    x[:, 158:158 + phase] = 1.
    x[:, 195:207] = torch.linspace(-.4, .4, 12)
    if phase == 8:
        x[:, 410] = x[:, 404] = 1.
    x[:, 439] = float(active)
    if active:
        x[:, 440:444] = torch.tensor([.01, .04, -.32, .55])
    return TensorDict({"policy": x, "critic": x.clone()}, batch_size=[batch])


def cfg():
    return dict(class_name="HeteroscedasticGaussianDistribution", std_type="log", init_std=.15)


def prior(obs):
    return SemanticRearOwnerRecoveryHistoryMLPModel(
        {"policy": obs["policy"][:, :439]}, {"actor": ["policy"]}, "actor", 12,
        distribution_cfg=cfg(), observation_layout=REAR_OWNER_OBSERVATION_LAYOUT,
        exploration_std_temperature=.25)


def composite(obs=None, source=None, bind=True, **kwargs):
    obs = observations() if obs is None else obs
    actor = module.SemanticRRCaptureLocalHistoryMLPModel(
        obs, {"actor": ["policy"]}, "actor", 12, distribution_cfg=cfg(),
        observation_layout=module.OBSERVATION_LAYOUT, **kwargs)
    if bind:
        source = prior(obs) if source is None else source
        actor.load_frozen_prior_state(source.state_dict())
    return actor


def head(actor):
    return [m for m in actor.mlp.modules() if isinstance(m, torch.nn.Linear)][-1]


def source_obs(obs):
    x = obs["policy"][:, :439]
    return TensorDict({"policy": x, "critic": x.clone()}, batch_size=[x.shape[0]])


@pytest.mark.parametrize("phase", [1, 4, 5, 7, 8])
def test_inactive_exact_source_no_local_forward_rng_or_cache_change(phase):
    obs = observations(phase=phase)
    source = prior(obs)
    actor = composite(obs, source=source)
    with torch.no_grad():
        head(actor).weight.fill_(3.)
        head(actor).bias[:12].fill_(-4.)
    rng = torch.get_rng_state().clone()
    before = actor.distribution._distribution
    source_cache = actor.frozen_prior.distribution._distribution
    with patch.object(actor.mlp, "forward", side_effect=AssertionError("inactive local forward")):
        raw, audit = module.audited_capture_local_policy_request(
            actor, obs, lambda: actor(obs), stochastic=False)
    assert torch.equal(raw, source(source_obs(obs)))
    assert torch.equal(rng, torch.get_rng_state())
    assert actor.distribution._distribution is before
    assert actor.frozen_prior.distribution._distribution is source_cache
    assert audit["local_network_forwards"] == 0
    assert audit["selected_raw_log_probability"] is None
    assert audit["active_conditional_std_full12"] is None
    assert audit["prefix_excluded_from_new_PPO_credit"] is True
    assert audit["applied_local_raw_mean_delta_full12"] == [0.] * 12


def test_unloaded_and_rebinding_rejected():
    obs = observations()
    actor = composite(obs, bind=False)
    with pytest.raises(RuntimeError, match="load the verified"):
        actor(obs)
    actor.load_frozen_prior_state(prior(obs).state_dict())
    with pytest.raises(RuntimeError, match="already bound"):
        actor.load_frozen_prior_state(prior(obs).state_dict())


def test_active_zero_local_mean_matches_prior_and_one_history_only():
    obs = observations(True)
    source = prior(obs)
    actor = composite(obs, source=source)
    with patch.object(module, "history_conditioned_head", wraps=module.history_conditioned_head) as history:
        value = actor(obs)
    assert history.call_count == 1
    assert torch.equal(value, source(source_obs(obs)))
    assert actor._last_forward_evidence["local_raw_mean_delta"].count_nonzero() == 0


def test_zero_owner439_prior_preserves_422_function_before_and_at_activation():
    from wlr50_clean.ppo.semantic_rear_cooperative_prep_actor import SemanticRearCooperativePrepHistoryMLPModel
    from wlr50_clean.ppo.semantic_rear_cooperative_prep_profile import COOPERATIVE_PREP_OBSERVATION_LAYOUT
    obs = observations(True, batch=2)
    old_x = obs["policy"][:, :422]
    old = SemanticRearCooperativePrepHistoryMLPModel(
        {"policy": old_x}, {"actor": ["policy"]}, "actor", 12,
        distribution_cfg=cfg(), observation_layout=COOPERATIVE_PREP_OBSERVATION_LAYOUT,
        exploration_std_temperature=.25)
    old_state = copy.deepcopy(old.state_dict())
    old_state["mlp.0.weight"] = torch.cat((old_state["mlp.0.weight"], torch.zeros(256, 17)), dim=1)
    source = prior(obs)
    source.load_state_dict(old_state, strict=True)
    assert source.mlp[0].weight[:, 422:].count_nonzero() == 0
    actor = composite(obs, source=source)
    assert obs["policy"][:, 422:439].count_nonzero() == 0
    expected = old(TensorDict({"policy": old_x}, batch_size=[2]))
    for active in (0., 1.):
        obs["policy"][:, 439] = active
        torch.testing.assert_close(actor(obs), expected, atol=3e-7, rtol=3e-6)
    actor.assert_frozen_state()


def test_all12_local_mean_composed_before_single_history():
    obs = observations(True)
    actor = composite(obs)
    initial = actor(obs).detach()
    delta = torch.linspace(-.6, .6, 12)
    with torch.no_grad():
        head(actor).bias[:12].copy_(delta)
    with patch.object(module, "history_conditioned_head", wraps=module.history_conditioned_head) as history:
        value = actor(obs)
    assert history.call_count == 1
    torch.testing.assert_close(value - initial, delta.unsqueeze(0) * .1, atol=3e-8, rtol=2e-6)
    assert torch.equal(actor._last_forward_evidence["applied_local_raw_mean_delta"][0], delta)


def test_inactive_or_mixed_stochastic_batch_rejected_without_rng_draw():
    for batch in (1, 2):
        obs = observations(False, batch=batch)
        if batch == 2:
            obs["policy"][1, 439] = 1.
        actor = composite(obs)
        rng = torch.get_rng_state().clone()
        with pytest.raises(ValueError, match="inactive frozen prefix"):
            actor(obs, stochastic_output=True)
        assert torch.equal(rng, torch.get_rng_state())
        assert actor.distribution._distribution is None


def test_real_gaussian_logprob_mean_std_and_single_draw_audit():
    obs = observations(True)
    actor = composite(obs, initial_capture_std=[.12] * 12)
    with torch.no_grad():
        head(actor).bias[:12].copy_(torch.linspace(-1., 1., 12))
    with patch.object(actor.distribution, "sample", wraps=actor.distribution.sample) as sample:
        raw, audit = module.audited_capture_local_policy_request(
            actor, obs, lambda: actor(obs, stochastic_output=True), stochastic=True)
    assert sample.call_count == 1
    mean, std = actor.output_distribution_params
    expected = torch.distributions.Normal(mean, std).log_prob(raw).sum(-1)
    assert torch.equal(actor.get_output_log_prob(raw), expected)
    assert audit["selected_raw_log_probability"] == float(expected[0])
    assert audit["conditional_mean_full12"] == mean[0].tolist()
    assert audit["active_conditional_std_full12"] == std[0].tolist()
    assert audit["sampling_draws"] == 1 and audit["extra_model_forwards"] == 0


def test_freeze_train_mode_normalizer_and_actual_adam_isolation():
    obs = observations(True)
    actor = composite(obs)
    original = copy.deepcopy(actor.frozen_prior.state_dict())
    local_before = copy.deepcopy(actor.mlp.state_dict())
    optimizer = torch.optim.Adam(actor.trainable_parameters(), lr=1e-3, weight_decay=.01)
    actor.train()
    assert actor.training and not actor.frozen_prior.training
    for _ in range(2):
        raw = actor(obs, stochastic_output=True)
        loss = -actor.get_output_log_prob(raw.detach()).mean() - .001 * actor.output_entropy.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        actor.update_normalization(obs)
        actor.assert_frozen_state(optimizer)
    assert all(torch.equal(value, actor.frozen_prior.state_dict()[name]) for name, value in original.items())
    assert any(not torch.equal(value, actor.mlp.state_dict()[name]) for name, value in local_before.items())
    bad_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-5)
    with pytest.raises(RuntimeError, match="must not be present"):
        actor.assert_frozen_state(bad_optimizer)


def test_official_PPO_storage_ratio_and_update_only_active_rows():
    obs = observations(True)
    actor = composite(obs)
    critic = MLPModel(obs, {"critic": ["critic"]}, "critic", 1,
                      hidden_dims=(32, 32), obs_normalization=False)
    storage = RolloutStorage("rl", 1, 4, obs, [12], "cpu")
    alg = PPO(actor, critic, storage, num_learning_epochs=1, num_mini_batches=1,
              learning_rate=1e-4, schedule="fixed", device="cpu")
    # Integration owns this NEW optimizer, never the old source Adam state.
    alg.optimizer = torch.optim.Adam(chain(actor.trainable_parameters(), critic.parameters()), lr=1e-4)
    with torch.inference_mode():
        for i in range(4):
            raw = alg.act(obs)
            before = alg.transition.actions_log_prob.clone()
            expected = actor.get_output_log_prob(raw)
            assert torch.equal(before, expected)
            next_obs = obs.clone()
            next_obs["policy"][:, 195:207] = raw.clamp(-20, 20)
            next_obs["critic"] = next_obs["policy"].clone()
            alg.process_env_step(next_obs, torch.tensor([float(i % 2)]), torch.tensor([i == 3]), {})
            assert torch.equal(storage.actions[i], raw)
            assert torch.equal(storage.actions_log_prob[i, :, 0], before)
            obs = next_obs
        alg.compute_returns(obs)
    assert bool((storage.observations["policy"][..., 439] == 1).all())
    old_prior = actor.assert_frozen_state(alg.optimizer)
    update = alg.update()
    assert all(math_isfinite(value) for value in update.values())
    assert actor.assert_frozen_state(alg.optimizer) == old_prior


def math_isfinite(value):
    return bool(torch.isfinite(torch.as_tensor(value)).all())


def test_full_composite_save_reload_does_not_reset_local(tmp_path):
    obs = observations(True)
    actor = composite(obs)
    with torch.no_grad():
        head(actor).bias[:12].fill_(.7)
    expected = actor(obs).detach().clone()
    path = tmp_path / "synthetic_composite.pt"
    torch.save(actor.state_dict(), path)
    restored = composite(obs, bind=False, expected_prior_state_sha256=actor.assert_frozen_state())
    restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    restored.train()
    assert torch.equal(restored(obs), expected)
    assert restored.assert_frozen_state() == actor.assert_frozen_state()
    assert not restored.frozen_prior.training


def test_hash_mismatch_and_prior_mutation_detected():
    obs = observations()
    source = prior(obs).state_dict()
    actor = composite(obs, bind=False, expected_prior_state_sha256="0" * 64)
    with pytest.raises(ValueError, match="declared frozen prior"):
        actor.load_frozen_prior_state(source)
    actor = composite(obs)
    state = copy.deepcopy(actor.state_dict())
    state["frozen_prior.mlp.0.weight"][0, 0] += 1.
    with pytest.raises(ValueError, match="replace the frozen prior"):
        actor.load_state_dict(state)
    with torch.no_grad():
        actor.frozen_prior.mlp[0].weight[0, 0] += .01
    with pytest.raises(RuntimeError, match="parameter/buffer changed"):
        actor.assert_frozen_state()


def test_contact_does_not_turn_off_latched_gate_and_reset_is_observable():
    obs = observations(True)
    actor = composite(obs)
    with torch.no_grad():
        head(actor).bias[:12].fill_(.5)
    for contact, bearing, hold in ((0., 0., 0.), (1., 1., .5), (0., 0., 0.)):
        obs["policy"][0, 444:447] = torch.tensor([contact, bearing, hold])
        actor(obs)
        assert actor._last_forward_evidence["applied_local_raw_mean_delta"].eq(.5).all()
    reset_obs = observations(False)
    actor.reset(torch.tensor([True]))  # no hidden gate/HISTORY cache to reset
    assert torch.equal(actor(reset_obs), actor.frozen_prior(source_obs(reset_obs)))


@pytest.mark.parametrize("index,value", [(439, .5), (440, -1.), (444, .1), (445, 1.), (446, 1.1), (441, float('nan'))])
def test_invalid_observable_gate_state_rejected(index, value):
    obs = observations()
    obs["policy"][0, index] = value
    actor = composite(obs)
    with pytest.raises(ValueError):
        actor(obs)


def test_std_validation_and_audit_double_call_rejected():
    with pytest.raises(ValueError, match="positive raw"):
        composite(initial_capture_std=[0.] * 12)
    obs = observations(True)
    actor = composite(obs)

    def twice():
        actor(obs)
        return actor(obs)

    with pytest.raises(RuntimeError, match="exactly one"):
        module.audited_capture_local_policy_request(actor, obs, twice, stochastic=False)
