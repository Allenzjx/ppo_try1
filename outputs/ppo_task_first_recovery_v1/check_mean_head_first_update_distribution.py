"""One bounded CPU diagnostic on saved first-branch rollout; never recompute GAE."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main():
    import torch
    from tensordict import TensorDict
    from wlr50_clean.ppo.semantic_history_actor import (
        SemanticTemperedHistoryMLPModel, history_conditioned_head, HISTORY_START, HISTORY_STOP,
    )
    from wlr50_clean.ppo.semantic_training import state_hash, write_json
    from wlr50_clean.ppo.semantic_transfer_roles import ROLE_OBSERVATION_LAYOUT
    torch.set_num_threads(1)
    base = ROOT / "outputs/ppo_task_first_recovery_v1"
    history = base / "checkpoints/history"
    run = ROOT / "runs/ppo_task_first_recovery_v1/train/20260916T0426012963516Z_gb0438f66ec63_0e70fe7e93a64b768d439966436da7b7"
    paths = {
        "before_mean_reset": history / "checkpoint_step_000176640.pt",
        "corrected_initial": history / "checkpoint_initial_mean_head_recovery_v1_from_000176640_rng_corrected.pt",
        "after_real_update": history / "checkpoint_step_000176768.pt",
    }
    payloads = {key: torch.load(path, map_location="cpu", weights_only=False) for key, path in paths.items()}
    rollout_path = run / "rollouts/rollout_001346.pt"
    rollout = torch.load(rollout_path, map_location="cpu", weights_only=False)
    observation = rollout["observations"]["policy"].reshape(-1,372)
    samples = rollout["actions"].reshape(-1,12)
    assert observation.shape == (128,372) and samples.shape == (128,12)
    td = TensorDict({"policy": observation}, batch_size=[len(observation)])
    result = {}
    for key in ("corrected_initial", "after_real_update"):
        payload = payloads[key]
        cfg = copy.deepcopy(payload["infos"]["runner_config"]["actor"])
        cfg.pop("class_name")
        actor = SemanticTemperedHistoryMLPModel(td, {"actor": ["policy"]}, "actor", 12, **cfg)
        actor.load_state_dict(payload["actor_state_dict"], strict=True)
        actor.eval()
        with torch.no_grad():
            head = actor.mlp(actor.get_latent(td))
            conditioned = history_conditioned_head(head, observation[:,HISTORY_START:HISTORY_STOP])
            result[key] = {"base_mean": head[:,0].clone(), "conditional_mean": conditioned[:,0].clone(),
                "learned_sigma": head[:,1].exp(), "effective_sigma": (head[:,1]+math.log(.5)).exp()}
            assert torch.equal(actor(td), conditioned[:,0])
    initial, learned = result["corrected_initial"], result["after_real_update"]
    saved_mean, saved_sigma = [v.reshape(-1,12) for v in rollout["distribution_params"]]
    old_logprob = rollout["actions_log_prob"].reshape(-1)
    recomputed_logprob = torch.distributions.Normal(initial["conditional_mean"], initial["effective_sigma"]).log_prob(samples).sum(-1)
    logprob_delta = recomputed_logprob-old_logprob
    before_state = payloads["before_mean_reset"]["actor_state_dict"]
    init_state = payloads["corrected_initial"]["actor_state_dict"]
    after_state = payloads["after_real_update"]["actor_state_dict"]
    mean_keys = {"mlp.4.weight", "mlp.4.bias"}
    features = sorted(set(init_state)-mean_keys)
    def summary(tensor):
        value = tensor.double()
        return {"mean": float(value.mean()), "min": float(value.min()), "max": float(value.max()),
                "rms": float(value.square().mean().sqrt()), "abs_max": float(value.abs().max())}
    def channel(i):
        row = {name: {key: summary(values[key][:,i]) for key in values} for name,values in result.items()}
        row["change_after_update"] = {key: summary(learned[key][:,i]-initial[key][:,i]) for key in initial}
        row["sampled_raw"] = summary(samples[:,i])
        row["sampled_raw_abs_gt3_count"] = int((samples[:,i].abs()>3).sum())
        row["learned_base_mean_abs_gt3_count"] = int((learned["base_mean"][:,i].abs()>3).sum())
        row["learned_conditional_mean_abs_gt3_count"] = int((learned["conditional_mean"][:,i].abs()>3).sum())
        return row
    labels = ["FL_hip","FL_knee","FR_hip","FR_knee","RL_hip","RL_knee","RR_hip","RR_knee","FL_wheel","FR_wheel","RL_wheel","RR_wheel"]
    output = {
        "schema": "wlr50_clean.mean_head_first_update_fixed_observation_distribution.v1",
        "source_checkpoints": {key: str(path) for key,path in paths.items()},
        "saved_rollout": str(rollout_path), "rollout_update": 1346, "observations": 128,
        "same_saved_preprocessed_observations_used_for_both_models": True,
        "normalizer": "Identity; saved observation scales/clips are not recomputed",
        "history": {"rho": .9, "source": "each saved observation columns195:207; no global cache", "summary": summary(observation[:,195:207])},
        "temperature": .5, "observation_layout": ROLE_OBSERVATION_LAYOUT,
        "no_GAE_returns_values_recomputed": True,
        "fixed_observation_comparison_is_not_new_closed_loop_success": True,
        "initialization_checks": {
            "all_actor_features_exactly_preserved": all(torch.equal(before_state[k],init_state[k]) for k in features),
            "all_sigma_rows_exactly_preserved": all(torch.equal(before_state[k][12:],init_state[k][12:]) for k in mean_keys),
            "only_mean_rows_zero": all(not bool(init_state[k][:12].any()) for k in mean_keys),
            "critic_exactly_preserved": state_hash(payloads["before_mean_reset"]["critic_state_dict"]) == state_hash(payloads["corrected_initial"]["critic_state_dict"]),
        },
        "real_update_checks": {
            "global_policy_decisions": payloads["after_real_update"]["infos"]["global_policy_decisions"],
            "branch_counts": payloads["after_real_update"]["infos"]["task_recovery_branch_counts"],
            "changed_feature_parameters": [k for k in features if not torch.equal(init_state[k],after_state[k])],
            "changed_sigma_rows": [k for k in mean_keys if not torch.equal(init_state[k][12:],after_state[k][12:])],
            "changed_mean_rows": [k for k in mean_keys if not torch.equal(init_state[k][:12],after_state[k][:12])],
            "critic_changed": state_hash(payloads["corrected_initial"]["critic_state_dict"]) != state_hash(payloads["after_real_update"]["critic_state_dict"]),
        },
        "stored_distribution_reproduction": {
            "mean_max_abs_error": float((initial["conditional_mean"]-saved_mean).abs().max()),
            "sigma_max_abs_error": float((initial["effective_sigma"]-saved_sigma).abs().max()),
            "old_logprob_max_abs_error": float(logprob_delta.abs().max()),
            "ratio_max_abs_error_from_one": float((logprob_delta.exp()-1).abs().max()),
            "tolerance_note": "CPU reevaluation of CUDA-recorded tensors can differ by float32 rounding; not rewritten stored data",
        },
        "channels": {label: channel(i) for i,label in enumerate(labels)},
        "wheel_components_after_update": {
            "base_common_mean": summary(learned["base_mean"][:,8:].mean(-1)),
            "conditional_common_mean": summary(learned["conditional_mean"][:,8:].mean(-1)),
            "base_differential": summary(learned["base_mean"][:,8:]-learned["base_mean"][:,8:].mean(-1,keepdim=True)),
            "conditional_differential": summary(learned["conditional_mean"][:,8:]-learned["conditional_mean"][:,8:].mean(-1,keepdim=True)),
            "descriptive_only_not_net_traction_or_projected_wheel_target": True,
        },
        "saturation_definition": "abs(raw_latent)>3 implies abs(tanh(raw))>.995; descriptive not a safety/acceptance gate; mapper history/slew not independently reconstructed",
    }
    path = base / "mean_head_first_update_distribution.json"
    write_json(path, output)
    print(json.dumps({"path": str(path), "checks": output["real_update_checks"],
        "reproduction": output["stored_distribution_reproduction"],
        "wheels": {label: output["channels"][label] for label in labels[8:]}}, indent=2))


if __name__ == "__main__":
    main()
