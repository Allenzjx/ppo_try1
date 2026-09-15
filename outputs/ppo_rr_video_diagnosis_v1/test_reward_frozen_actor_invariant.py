"""One output-only CPU fixture; no learned checkpoint or Isaac/optimizer updates."""
from __future__ import annotations

import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"tests"/"unit"))
sys.path.insert(0, str(ROOT/"src"))


def test_reward_only_does_not_change_saved_reloaded_frozen_history372_actor(tmp_path):
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    import torch
    from tensordict import TensorDict
    from test_semantic_v3_continuation import Core324
    from test_semantic_observation_reward_env import _built, _frame, _raw, _sample
    from wlr50_clean.ppo import semantic_training as training
    from wlr50_clean.ppo.semantic_policy_distribution import HISTORY_POLICY
    from wlr50_clean.ppo.semantic_transfer_roles import ROLE_OBSERVATION_LAYOUT
    from wlr50_clean.ppo.semantic_reward import SemanticRewardCalculator, load_semantic_reward_config

    class Core372(Core324):
        def reset(self, **kwargs):
            return super().reset(**kwargs)+(0.,)*48

    def make_actor():
        env = training.SemanticRslAdapter(Core372(), seed=1001, device="cpu")
        env.cfg["semantic_version"] = "v3"
        runner, _ = training.construct_semantic_runner(env, seed=1001, device="cpu",
            policy_version=HISTORY_POLICY, observation_layout=ROLE_OBSERVATION_LAYOUT,
            initialize_actor=False)
        runner.alg.actor.eval()
        for parameter in runner.alg.actor.parameters():
            parameter.requires_grad_(False)
        return runner

    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(1001)
        source = make_actor()
        fixture = tmp_path/"synthetic_untrained_HISTORY372_actor_fixture.pt"
        torch.save({"actor": source.alg.actor.state_dict(),
                    "normalizer": source.alg.actor.obs_normalizer.state_dict()}, fixture)
        loaded = torch.load(fixture, map_location="cpu", weights_only=True)
        target = make_actor()
        target.alg.actor.load_state_dict(loaded["actor"], strict=True)
        target.alg.actor.obs_normalizer.load_state_dict(loaded["normalizer"], strict=True)
        assert training.state_hash(source.alg.actor.state_dict()) == training.state_hash(target.alg.actor.state_dict())
        assert training.state_hash(source.alg.actor.obs_normalizer.state_dict()) == training.state_hash(target.alg.actor.obs_normalizer.state_dict())
        assert isinstance(target.alg.actor.obs_normalizer, torch.nn.Identity)

        # Deliberately nonzero fixed synthetic observation, including raw action history.
        values = torch.linspace(-.2, .2, 372, dtype=torch.float32).reshape(1, 372)
        values[:, 195:207] = torch.linspace(-.6, .6, 12).reshape(1, 12)
        inputs = TensorDict({"policy": values.clone(), "critic": values.clone()}, batch_size=[1])
        inputs_before = {key: value.clone() for key, value in inputs.items()}
        actor_before = copy.deepcopy(target.alg.actor.state_dict())
        norm_before = copy.deepcopy(target.alg.actor.obs_normalizer.state_dict())
        optimizer_before = training.state_hash(target.alg.optimizer.state_dict())
        rng_before = torch.get_rng_state().clone()
        with torch.inference_mode():
            original_action = source.alg.actor(inputs, stochastic_output=False).clone()
            action_before = target.alg.actor(inputs, stochastic_output=False).clone()
        assert torch.equal(original_action, action_before)
        assert bool(torch.any(action_before != 0))

        before = _built(_frame(raw=_raw(pitch=.3)))
        after = _built(_frame(1, raw=_raw(1, pitch=.3)))
        role = {"transfer_roles_version": "diagonal_transfer_roles_v1", "physical_transfer_fraction": .75}
        before = replace(before, task={**before.task, **role})
        after = replace(after, task={**after.task, **role})
        sample = _sample(before, after)
        config = load_semantic_reward_config(ROOT/"configs"/"ppo_fsm_reference_p09_stable_v2"/"reward_config.yaml")
        original_values = copy.deepcopy(config.values)
        altered = replace(config, values={**copy.deepcopy(config.values), "transfer_attitude_weight": 1.})
        reward_default = SemanticRewardCalculator(config).evaluate(before, after, [sample], termination_reason=None, task_success=False)
        reward_altered = SemanticRewardCalculator(altered).evaluate(before, after, [sample], termination_reason=None, task_success=False)
        assert config.values == original_values
        assert altered.values["transfer_attitude_weight"] != config.values["transfer_attitude_weight"]
        assert reward_default["cost_components"]["gravity_attitude"] > 0
        assert reward_default["families"]["body_stability"] != reward_altered["families"]["body_stability"]
        assert reward_default["total"] != reward_altered["total"]
        with torch.inference_mode():
            action_after = target.alg.actor(inputs, stochastic_output=False).clone()
        assert torch.equal(action_before, action_after)
        assert all(torch.equal(inputs[key], value) for key, value in inputs_before.items())
        assert all(torch.equal(target.alg.actor.state_dict()[key], value) for key, value in actor_before.items())
        assert training.state_hash(target.alg.actor.obs_normalizer.state_dict()) == training.state_hash(norm_before)
        assert training.state_hash(target.alg.optimizer.state_dict()) == optimizer_before
        assert torch.equal(torch.get_rng_state(), rng_before)
        assert source.alg.storage.step == target.alg.storage.step == 0
        assert source.alg.transition.actions is target.alg.transition.actions is None
        receipt = {"scope": "synthetic CPU regression, not learned policy evaluation or policy improvement",
            "python": sys.executable, "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "observation_dimension": 372, "policy": HISTORY_POLICY,
            "history_slice": [195, 207], "fixture_checkpoint": str(fixture),
            "real_project_checkpoint_loaded": False, "physics_ticks": 0, "ppo_updates": 0,
            "optimizer_steps": 0, "default_transfer_attitude_weight": config.values["transfer_attitude_weight"],
            "altered_transfer_attitude_weight": 1., "fixed_physical_transfer_fraction": .75,
            "default_reward_total": reward_default["total"], "altered_reward_total": reward_altered["total"],
            "default_body_stability": reward_default["families"]["body_stability"],
            "altered_body_stability": reward_altered["families"]["body_stability"],
            "action_bitwise_equal": True, "action": action_after.tolist()[0],
            "observation_bitwise_unchanged": True, "actor_state_unchanged": True,
            "identity_normalizer_unchanged": True, "optimizer_unchanged": True, "torch_rng_unchanged": True,
            "actor_state_sha256": training.state_hash(actor_before),
            "normalizer_state_sha256": training.state_hash(norm_before),
            "observation_sha256": training.state_hash(inputs_before)}
        path = Path(__file__).with_name("reward_frozen_actor_regression_receipt.json")
        with path.open("x", encoding="utf-8") as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
    finally:
        torch.set_num_threads(old_threads)
