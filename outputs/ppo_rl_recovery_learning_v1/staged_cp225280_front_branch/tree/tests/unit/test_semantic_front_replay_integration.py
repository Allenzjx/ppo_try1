"""One official512/439 PPO update with declared replay; synthetic, no robot credit."""
from collections import Counter
import hashlib
import json

import pytest


def test_official512_439_replay_20steps_640_exposures_raw_likelihood(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    torch = pytest.importorskip("torch")
    pytest.importorskip("rsl_rl")
    from test_semantic_collection512 import Synthetic439LongCore
    from wlr50_clean.ppo import semantic_training as training
    from wlr50_clean.ppo.semantic_rear_owner_profile import (
        REAR_OWNER_POLICY, REAR_OWNER_OBSERVATION_LAYOUT)
    from wlr50_clean.ppo.semantic_return_profile import COLLECTION_512
    from wlr50_clean.ppo.semantic_front_replay import (
        SCHEMA, DATA_SCHEMA, VERSION, FrontReplayRegularizer, current_front_gaussian)

    rng, threads = torch.get_rng_state(), torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        core = Synthetic439LongCore()
        env = training.SemanticRslAdapter(core, seed=1001, device="cpu")
        env.cfg["semantic_version"] = "v3"
        runner, cfg = training.construct_semantic_runner(
            env, seed=1001, device="cpu", initialize_actor=False,
            policy_version=REAR_OWNER_POLICY,
            observation_layout=REAR_OWNER_OBSERVATION_LAYOUT,
            collection_profile=COLLECTION_512)
        assert cfg["num_steps_per_env"] == 512
        alg, storage = runner.alg, runner.alg.storage
        obs = env.get_observations()
        # Official storage receives only fresh sampled raw actions, never replay.
        with torch.inference_mode():
            for _ in range(512):
                raw = alg.act(obs)
                obs, reward, done, extras = env.step(raw)
                alg.process_env_step(obs, reward, done, extras)
            alg.compute_returns(obs)
        assert storage.step == 512 and storage.actions.shape == (512, 1, 12)
        assert storage.observations["policy"].shape == (512, 1, 439)
        saved = {key: getattr(storage, key).clone() for key in
                 ("actions", "actions_log_prob", "advantages", "returns", "dones")}
        saved_obs = storage.observations["policy"].clone()
        assert torch.where(saved["dones"][:, 0, 0].bool())[0].tolist() == [451]

        # Independent valid FRONT rows. No rows are added to on-policy storage.
        front = torch.zeros(3, 439)
        for i, phase in enumerate((0, 4, 5)):
            front[i, phase] = 1.0
            front[i, 20] = .02
            front[i, 158:158+phase] = 1.0
        with torch.no_grad():
            mean, log_sigma = current_front_gaussian(alg.actor, front)
        rows = [dict(decision_index=9000+i, phase=("P01", "P05", "P06")[i],
                     observation=front[i].tolist(),
                     reference_mean=(mean[i] + .02).tolist(),
                     reference_sigma=log_sigma[i].exp().tolist())
                for i in range(3)]
        payload = json.dumps(dict(schema=DATA_SCHEMA, training_rows=rows[:2],
                                  heldout_rows=rows[2:])).encode()
        dataset = tmp_path / "synthetic_front.json"
        dataset.write_bytes(payload)
        runner._semantic_front_replay = FrontReplayRegularizer(dict(
            schema=SCHEMA, version=VERSION, coefficient=1.0, minibatch_size=32,
            dataset_path=str(dataset.resolve()),
            dataset_sha256=hashlib.sha256(payload).hexdigest()), device="cpu")

        old_generator = storage.mini_batch_generator
        old_logp, old_kl = alg.actor.get_output_log_prob, alg.actor.get_kl_divergence
        hook_counts = [len(p._backward_hooks or {}) for p in alg.actor.parameters()]
        audit_path = tmp_path / "likelihood.json"
        update = training.audited_ppo_update(runner, likelihood_audit_path=audit_path)
        audit = json.loads(audit_path.read_text())
        report = update["front_replay_regularization"]
        assert update["optimizer_steps"] == 20
        assert update["finite_nonzero_gradient_observed"] and update["actor_parameters_changed"]
        assert len(report["minibatches"]) == 20
        assert report["actual_replay_row_exposures"] == 640
        assert report["on_policy_samples_added"] == 0
        assert report["separate_auxiliary_optimizer_steps"] == 0
        assert all(r["gradient_consumed_once"] for r in report["minibatches"])
        assert all(9002 not in r["source_decision_indices"] for r in report["minibatches"])
        assert all(float(state["step"]) == 20.0 for state in alg.optimizer.state.values())
        assert audit["extra_model_forwards"] == 24 and audit["extra_random_draws"] == 0
        assert audit["extra_forward_breakdown"] == dict(
            reference_KL_training=20, reference_KL_fit_heldout_before_after=4,
            on_policy_likelihood_audit=0)

        # Official generator/hooks are restored and storage contents not rewritten.
        assert storage.mini_batch_generator == old_generator
        assert alg.actor.get_output_log_prob == old_logp
        assert alg.actor.get_kl_divergence == old_kl
        assert [len(p._backward_hooks or {}) for p in alg.actor.parameters()] == hook_counts
        assert storage.step == 0  # normal RSL clear after the single update
        for key, value in saved.items():
            assert torch.equal(getattr(storage, key), value)
        assert torch.equal(storage.observations["policy"], saved_obs)

        exposures = Counter()
        assert len(audit["minibatches"]) == 20
        for batch in audit["minibatches"]:
            assert len(batch["rollout_flat_indices"]) == 128
            selected = []
            for j, indexes in enumerate(batch["rollout_flat_indices"]):
                assert len(indexes) == 1
                index = indexes[0]
                exposures[index] += 1
                selected.append(index)
                assert batch["old_log_probability"][j] == saved["actions_log_prob"][index, 0].item()
            # Independent density from the ACTUAL current official cache, using
            # the saved raw, not applied robot target or replay reference rows.
            actual_mean = torch.tensor(batch["current_conditional_mean"])
            actual_sigma = torch.tensor(batch["current_conditional_sigma"])
            raw = saved["actions"][selected, 0]
            logp = torch.distributions.Normal(actual_mean, actual_sigma).log_prob(raw).sum(-1)
            torch.testing.assert_close(logp, torch.tensor(batch["optimization_log_probability"]),
                                       rtol=2e-5, atol=2e-5)
            assert "official_PPO_head_derivative_only" in batch["gradient_semantics"]
        assert exposures == Counter({i: 5 for i in range(512)})
    finally:
        torch.set_rng_state(rng)
        torch.set_num_threads(threads)

