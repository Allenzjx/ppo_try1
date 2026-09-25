"""CPU integration only: real immutable prior, synthetic transitions, no PPO credit.

All checkpoint/likelihood artifacts are pytest temporary files. No simulator,
robot rollout, formal checkpoint or production pointer is created by this file.
"""
import json
from pathlib import Path

import pytest
import torch

from wlr50_clean.ppo import semantic_rr_capture_local as route
from wlr50_clean.ppo.rl_library_wrapper import capture_training_rng_state, seed_training_rngs
from wlr50_clean.ppo.semantic_training import audited_ppo_update, parameter_hash, state_hash


@pytest.fixture(autouse=True)
def cpu_rng():
    from wlr50_clean.ppo.rl_library_wrapper import restore_training_rng_state
    before = capture_training_rng_state(seed=1001)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    seed_training_rngs(1001)
    yield
    restore_training_rng_state(before, expected_seed=1001)
    torch.set_num_threads(threads)


def observation(*, active, index=0):
    values = [0.] * 447
    values[8] = 1.
    values[20] = .02
    values[158:166] = [1.] * 8
    values[195:207] = [(.02 * j - .1) + index * .00001 for j in range(12)]
    values[404] = values[410] = 1.
    values[439] = float(active)
    if active:
        values[440:444] = [.01 + index / 30000., .04, -.32, .55]
    return route.tensor_observation(values, "cpu")


def initialized():
    runner = route.make_runner("cpu", 1001)
    provenance = route.initialize_prior(runner)
    return runner, provenance


def test_actual_source_initialization_and_prefix_exclusion():
    runner, provenance = initialized()
    actor = runner.alg.actor
    assert provenance["sha256"] == route.settings()["prior_checkpoint_sha256"]
    assert provenance["manifest_sha256"] == route.settings()["prior_manifest_sha256"]
    assert provenance["original422_tensor_identity_verified"]
    assert provenance["prior_optimizer_loaded"] is False
    assert not runner.alg.optimizer.state
    prior_ids = {id(p) for p in actor.frozen_prior.parameters()}
    optimizer_ids = {id(p) for g in runner.alg.optimizer.param_groups for p in g["params"]}
    expected_ids = {id(p) for p in actor.trainable_parameters()} | {
        id(p) for p in runner.alg.critic.parameters()}
    assert optimizer_ids == expected_ids
    assert not prior_ids & optimizer_ids
    assert isinstance(actor.obs_normalizer, torch.nn.Identity)
    assert isinstance(actor.frozen_prior.obs_normalizer, torch.nn.Identity)
    assert isinstance(runner.alg.critic.obs_normalizer, torch.nn.Identity)
    runner.alg.train_mode()
    assert actor.training and not actor.frozen_prior.training
    before_rng = capture_training_rng_state(seed=1001)
    obs = observation(active=False)
    with torch.inference_mode():
        raw, audit = route.request(runner, obs, stochastic=False)
        source_obs = {"policy": obs["policy"][:, :439]}
        expected = actor.frozen_prior(source_obs, stochastic_output=False)
    assert torch.equal(raw, expected)
    assert capture_training_rng_state(seed=1001) == before_rng
    assert runner.alg.storage.step == 0
    assert runner.alg.transition.actions is None
    assert audit["prefix_excluded_from_new_PPO_credit"] is True
    assert audit["selected_raw_log_probability"] is None
    assert audit["local_network_forwards"] == 0
    with torch.inference_mode():
        active_raw, _ = route.request(runner, observation(active=True), stochastic=False)
    assert torch.equal(active_raw, raw)
    actor.assert_frozen_state(runner.alg.optimizer)


@pytest.mark.parametrize("terminal_tail", [False, True])
def test_official_512_storage_audit_and_temporary_reload(tmp_path, monkeypatch, terminal_tail):
    runner, prior = initialized()
    runner.alg.train_mode()
    actor = runner.alg.actor
    prior_hash = parameter_hash(actor.frozen_prior)
    local_hash = parameter_hash(actor.mlp)
    assert len(runner.alg.storage.rewards) == 512
    for index in range(512):
        obs = observation(active=True, index=index)
        nxt = observation(active=True, index=index + 1)
        with torch.inference_mode():
            raw, audit = route.request(runner, obs, stochastic=True)
            old_logp = runner.alg.transition.actions_log_prob.clone()
            mean, sigma = [x.clone() for x in actor.output_distribution_params]
            reference = torch.distributions.Normal(mean, sigma).log_prob(raw).sum(-1)
            assert torch.equal(old_logp, reference)
            assert audit["prefix_excluded_from_new_PPO_credit"] is False
            assert audit["history_kernel_applications"] == 1
            done = torch.tensor([terminal_tail and index == 511])
            reward = torch.tensor([-.02 + .005 * (index % 7)])
            runner.alg.process_env_step(nxt, reward, done, {"time_outs": torch.zeros_like(done)})
        assert torch.equal(runner.alg.storage.actions[index], raw)
        assert torch.equal(runner.alg.storage.actions_log_prob[index].view(-1), old_logp)
        assert torch.equal(runner.alg.storage.distribution_params[0][index], mean)
        assert torch.equal(runner.alg.storage.distribution_params[1][index], sigma)
    with torch.inference_mode():
        tail_value = runner.alg.critic(nxt).flatten()[0]
        runner.alg.compute_returns(nxt)
    storage = runner.alg.storage
    expected_tail = reward[0] + (0. if terminal_tail else runner.alg.gamma * tail_value)
    assert torch.allclose(storage.returns[-1, 0, 0], expected_tail, atol=2e-6, rtol=0)
    assert bool(storage.dones[:-1].any()) is False
    assert bool((storage.observations["policy"][:, :, 439] == 1.).all())
    assert torch.isfinite(storage.returns).all()
    likelihood_path = tmp_path / "synthetic_likelihood_not_physics.json"
    report = audited_ppo_update(runner, likelihood_audit_path=likelihood_path)
    assert report["optimizer_steps"] == 20
    assert report["finite_nonzero_gradient_observed"]
    assert report["actor_parameters_changed"]
    assert storage.step == 0
    assert parameter_hash(actor.mlp) != local_hash
    assert parameter_hash(actor.frozen_prior) == prior_hash
    actor.assert_frozen_state(runner.alg.optimizer)
    likelihood = json.loads(likelihood_path.read_text())
    assert len(likelihood["minibatches"]) == 20
    assert likelihood["extra_random_draws"] == 0
    assert likelihood["extra_model_forwards"] == 0
    for batch in likelihood["minibatches"]:
        assert len(batch["old_log_probability"]) == 128
        assert len(batch["current_conditional_mean"]) == 128
        assert len(batch["current_conditional_sigma"]) == 128
    counts = dict(local_policy_decisions=512, local_ppo_updates=1, local_optimizer_steps=20,
                  prefix_decisions=0, capture_opportunities=0, local_successes=0,
                  completed_local_episodes=0, auxiliary_updates=0)
    runtime = {"test_only": "synthetic_CPU_not_physical_or_formal_PPO"}
    monkeypatch.setattr(route, "OUTPUT", tmp_path / "isolated_test_output")
    before_save = state_hash(runner.alg.save())
    before_rng = capture_training_rng_state(seed=1001)
    pointer = route.save(runner, runtime, prior, counts, source_run=tmp_path / "synthetic_test")
    assert Path(pointer["checkpoint"]).is_relative_to(tmp_path)
    assert capture_training_rng_state(seed=1001) == before_rng
    assert state_hash(runner.alg.save()) == before_save
    expected_random = torch.rand(8)
    restored = route.make_runner("cpu", 1001)
    restored_prior, restored_counts = route.load(restored, pointer["checkpoint"], runtime)
    assert torch.equal(torch.rand(8), expected_random)
    assert restored_prior == prior and restored_counts == counts
    assert state_hash(restored.alg.save()) == before_save
    assert restored.alg.learning_rate == runner.alg.learning_rate
    assert parameter_hash(restored.alg.actor.frozen_prior) == prior_hash
    restored.alg.actor.assert_frozen_state(restored.alg.optimizer)
    with torch.inference_mode():
        assert torch.equal(restored.alg.actor(nxt), actor(nxt))
    with pytest.raises(ValueError, match="contract/hash mismatch"):
        route.load(restored, pointer["checkpoint"], {"wrong_runtime": True})
