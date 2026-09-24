"""One real official CPU collector/update on synthetic439 data, zero robot credit.

This exercises collection/GAE/likelihood wiring, not the immutable real-source
publication lineage (which has its own actual-checkpoint verification).
"""
from collections import Counter
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("rsl_rl")

from test_semantic_rear_policy_training_audit import Synthetic419Core
from wlr50_clean.ppo import semantic_training as training
from wlr50_clean.ppo.semantic_rear_owner_profile import (
    REAR_OWNER_POLICY, REAR_OWNER_OBSERVATION_LAYOUT)
from wlr50_clean.ppo.semantic_return_profile import COLLECTION_512, COLLECTION_PROFILE_KEY


def test_common_policy_resolver_requires_explicit512_receipt_and_keeps128():
    import copy
    from wlr50_clean.ppo.semantic_front_retention439 import COLLECTION_KEY, COLLECTION_SCHEMA
    from wlr50_clean.ppo.semantic_policy_distribution import policy_version_from_metadata, policy_contract
    config = training.semantic_runner_config(seed=1001, device="cpu", semantic_version="v3",
        policy_version=REAR_OWNER_POLICY, observation_layout=REAR_OWNER_OBSERVATION_LAYOUT,
        collection_profile=COLLECTION_512)
    metadata = dict(seed=1001, semantic_version="v3", runner_config=config,
        policy_contract=policy_contract(REAR_OWNER_POLICY, observation_layout=REAR_OWNER_OBSERVATION_LAYOUT),
        runtime_contract={"experiment_id":"rr_rl_timing_policy_learning_v1"})
    with pytest.raises(ValueError, match="unreceipted"):
        policy_version_from_metadata(metadata)
    metadata[COLLECTION_KEY] = dict(schema=COLLECTION_SCHEMA, target_runner_config=copy.deepcopy(config))
    assert policy_version_from_metadata(metadata) == REAR_OWNER_POLICY
    wrong = copy.deepcopy(metadata)
    wrong["runner_config"]["algorithm"]["gamma"] = .995
    with pytest.raises(ValueError):
        policy_version_from_metadata(wrong)
    wrong = copy.deepcopy(metadata)
    wrong["runtime_contract"]["experiment_id"] = "another_experiment"
    with pytest.raises(ValueError):
        policy_version_from_metadata(wrong)
    old = copy.deepcopy(metadata)
    del old[COLLECTION_KEY]
    del old["runner_config"][COLLECTION_PROFILE_KEY]
    old["runner_config"]["num_steps_per_env"] = 128
    assert policy_version_from_metadata(old) == REAR_OWNER_POLICY


class Synthetic439LongCore(Synthetic419Core):
    """Ordinary P09->P10 at step4, real synthetic episode terminal at452."""
    def __init__(self):
        super().__init__(terminal_at=452)

    def observation(self, raw=(0.,) * 12):
        return super().observation(raw) + (0.,) * 20


def test_official512_collector_terminal452_tail_bootstrap_and_raw_likelihood(tmp_path, monkeypatch):
    assert not torch.cuda.is_available(), "run with CUDA_VISIBLE_DEVICES=-1"
    rng = torch.get_rng_state()
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        core = Synthetic439LongCore()
        env = training.SemanticRslAdapter(core, seed=1001, device="cpu")
        env.cfg["semantic_version"] = "v3"
        runner, config = training.construct_semantic_runner(env, seed=1001, device="cpu",
            policy_version=REAR_OWNER_POLICY, observation_layout=REAR_OWNER_OBSERVATION_LAYOUT,
            initialize_actor=False, collection_profile=COLLECTION_512)
        assert config[COLLECTION_PROFILE_KEY] == COLLECTION_512
        assert config["num_steps_per_env"] == 512
        assert (runner.alg.gamma, runner.alg.lam) == (.9985, .99)
        assert runner.alg.storage.actions.shape == (512, 1, 12)
        assert runner.alg.storage.observations["policy"].shape == (512, 1, 439)
        assert runner.alg.storage.step == 0 and runner.alg.transition.actions is None

        original_returns = runner.alg.compute_returns
        successor_values = []
        def capture_actual_successor(obs):
            # Test-only critic evaluation; no extra actor sampling or policy draw.
            with torch.no_grad():
                successor_values.append(runner.alg.critic(obs).detach().cpu().clone())
            return original_returns(obs)
        monkeypatch.setattr(runner.alg, "compute_returns", capture_actual_successor)
        contract = dict(experiment_id="rr_rl_timing_policy_learning_v1", semantic_version="v3",
            training_budgets=training.training_quantity_budgets("rr_rl_timing_policy_learning_v1"),
            evidence="synthetic CPU439 collector only; zero physical PPO/AUX credit")
        result = training.train_semantic(runner, env, run_dir=tmp_path/"run",
            output_root=tmp_path/"output", stage="full_episode", decisions=512,
            contract=contract, seed=1001, checkpoint_interval_updates=1)
        assert (result["actual_policy_decisions"], result["ppo_updates_this_run"],
                result["optimizer_steps_this_run"]) == (512, 1, 20)
        assert core.calls == 512 and core.resets == 2
        assert result["finite_nonzero_gradient_observed"]
        assert result["actor_parameter_sha256_before"] != result["actor_parameter_sha256_after"]
        assert len(successor_values) == 1
        rollout = torch.load(tmp_path/"run/rollouts/rollout_000001.pt",
                             map_location="cpu", weights_only=False)
        assert rollout["observations"]["policy"].shape == (512, 1, 439)
        dones = rollout["dones"][:, 0, 0].bool()
        assert torch.where(dones)[0].tolist() == [451]
        rewards, values, returns = (rollout[key][:, 0, 0] for key in ("rewards", "values", "returns"))
        assert returns[451].item() == rewards[451].item()
        assert returns[511].item() == pytest.approx(
            (rewards[511] + runner.alg.gamma * successor_values[0].reshape(-1)[0]).item(), abs=2e-6)

        # Expanded recurrence over the saved real collector tensors. Done452
        # cancels both reset-value bootstrap and the following episode's trace.
        expected = torch.empty_like(returns)
        advantage = torch.zeros(())
        next_value = successor_values[0].reshape(-1)[0]
        for index in range(511, -1, -1):
            alive = 1.0 - dones[index].float()
            delta = rewards[index] + runner.alg.gamma * alive * next_value - values[index]
            advantage = delta + runner.alg.gamma * runner.alg.lam * alive * advantage
            expected[index] = advantage + values[index]
            next_value = values[index]
        torch.testing.assert_close(returns, expected, rtol=3e-6, atol=3e-6)

        rows = [json.loads(line) for line in
                (tmp_path/"run/residual_and_projection_audit.jsonl").read_text().splitlines()]
        assert len(rows) == 512 and [r["global_policy_decision"] for r in rows] == list(range(1, 513))
        changes = [i for i,r in enumerate(rows)
                   if r["applied_audit"]["phase_id"] != r["applied_audit"]["end_phase_id"]]
        assert changes == [3, 455]
        assert all(not rows[i]["terminal"] for i in changes)
        assert not rows[-1]["terminal"] and rows[-1]["applied_audit"]["terminal_bootstrap_allowed"]
        for i, row in enumerate(rows):
            request = row["policy_request"]
            assert request["sampling_draws"] == 1
            assert request["extra_random_draws"] == request["extra_model_forwards"] == 0
            assert request["selected_raw_full12"] == row["raw_policy_action_full12"] == rollout["actions"][i, 0].tolist()
            assert request["selected_raw_log_probability"] == row["old_log_probability"] == rollout["actions_log_prob"][i, 0].item()
            assert request["effective_sigma_full12"] == rollout["distribution_params"][1][i, 0].tolist()
            assert request["rear_owner_observed_features"] == rollout["observations"]["policy"][i, 0, 422:].tolist()
            assert row["raw_policy_action_full12"] != row["applied_audit"]["applied_action_full12"]

        audit = json.loads((tmp_path/"run/rollouts/update_000001_likelihood.json").read_text())
        assert len(audit["minibatches"]) == 20
        exposure = Counter()
        for batch in audit["minibatches"]:
            assert len(batch["rollout_flat_indices"]) == 128
            for j, indices in enumerate(batch["rollout_flat_indices"]):
                assert len(indices) == 1
                exposure[indices[0]] += 1
                assert batch["old_log_probability"][j] == rollout["actions_log_prob"][indices[0], 0].item()
        assert exposure == Counter({i: 5 for i in range(512)})
        manifest = json.loads((tmp_path/"output/checkpoints/history/checkpoint_step_000000512_manifest.json").read_text())
        assert (manifest["global_policy_decisions"], manifest["ppo_updates"], manifest["optimizer_steps"]) == (512, 1, 20)
        assert manifest["runner_config"][COLLECTION_PROFILE_KEY] == COLLECTION_512
        assert manifest["save_load_round_trip"] is True
    finally:
        torch.set_rng_state(rng)
        torch.set_num_threads(threads)
