"""One bounded CPU-only shadow fit; reads real data and prints JSON, never saves weights.

This is not PPO continuation. Exactly two 20-step critic-only fits are performed
only after checkpoint/rollout/audit binding and old-value reproduction succeed.
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import copy
import hashlib
import json
from pathlib import Path

import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "runs/ppo_semantic_v3/train/20260906T1650134938238Z_g73e937039708_7681a31f01714528b75509300dfac00f"
CHECKPOINT = ROOT / "outputs/ppo_semantic_v3/checkpoints/history/checkpoint_step_000069760.pt"
ROLLOUT = RUN / "rollouts/rollout_000511.pt"
LR = 1e-5
CLIP = 0.2
PERMUTATION_SEED = 20260906


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def tensor_digest(state: dict) -> str:
    result = hashlib.sha256()
    for name, value in state.items():
        require(value.device.type == "cpu", "Non-CPU state")
        result.update(name.encode())
        result.update(value.detach().contiguous().numpy().tobytes())
    return result.hexdigest()


def main() -> dict:
    require(CHECKPOINT.is_file() and ROLLOUT.is_file(), "Missing exact source artifacts")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    rollout = torch.load(ROLLOUT, map_location="cpu", weights_only=False)
    metadata = checkpoint["infos"]
    manifest = json.loads(CHECKPOINT.with_name(CHECKPOINT.stem + "_manifest.json").read_text())
    require(metadata["global_policy_decisions"] == 69760, "Wrong checkpoint global")
    require(metadata["ppo_updates"] == 510 and checkpoint["iter"] == 510, "Wrong checkpoint iteration")
    require(Path(metadata["source_run"]).resolve() == RUN.resolve(), "Wrong source run")
    require(rollout["schema"] == "wlr50_clean.semantic_on_policy_rollout.v1", "Wrong rollout schema")
    require(rollout["runtime_contract"] == metadata["runtime_contract"], "Runtime mismatch")
    require(rollout["policy_contract"] == metadata["policy_contract"], "Policy mismatch")
    for key in ("runtime_contract", "global_policy_decisions", "ppo_updates", "optimizer_steps",
                "actor_parameter_sha256", "critic_parameter_sha256", "optimizer_state_sha256"):
        require(manifest[key] == metadata[key], "Checkpoint sidecar mismatch: " + key)
    config = metadata["runner_config"]
    algorithm = config["algorithm"]
    require(config["num_steps_per_env"] == 128, "Wrong rollout length")
    require(algorithm["num_learning_epochs"] == 5 and algorithm["num_mini_batches"] == 4,
            "Wrong official minibatch schedule")
    require(algorithm["clip_param"] == CLIP and algorithm["use_clipped_value_loss"] is True,
            "Wrong source value loss")
    require(algorithm["value_loss_coef"] == 1 and algorithm["max_grad_norm"] == 1,
            "Unexpected value weight or gradient clipping")
    require(algorithm["optimizer"] == "adam", "Wrong optimizer")
    require(config["critic"]["obs_normalization"] is False, "Nonidentity normalizer")

    observations = TensorDict({key: value.flatten(0, 1)
                               for key, value in rollout["observations"].items()}, batch_size=[128])
    old_values = rollout["values"].flatten(0, 1)
    returns = rollout["returns"].flatten(0, 1)
    rewards = rollout["rewards"].flatten(0, 1)
    dones = rollout["dones"].flatten(0, 1).bool()
    for value in (*observations.values(), old_values, returns, rewards):
        require(value.device.type == "cpu" and bool(torch.isfinite(value).all()), "Invalid saved tensor")
    require(observations["critic"].shape == (128, 324), "Wrong real observation shape")
    require(old_values.shape == returns.shape == rewards.shape == dones.shape == (128, 1),
            "Wrong saved training tensor shapes")
    terminal_indices = torch.nonzero(dones.flatten()).flatten().tolist()
    require(terminal_indices == [45], "Unexpected terminal batch")
    require(torch.equal(returns[dones], rewards[dones]), "True terminal return differs from reward")

    audit_rows = []
    with (RUN / "residual_and_projection_audit.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            global_decision = row["global_policy_decision"]
            if 69761 <= global_decision <= 69888:
                audit_rows.append(row)
            if global_decision >= 69888:
                break
    require([row["global_policy_decision"] for row in audit_rows] == list(range(69761, 69889)),
            "Audit range is not exact and contiguous")
    for index, row in enumerate(audit_rows):
        require(row["old_value"] == old_values[index].item(), "Saved old value not audit-bound")
        require(row["reward"] == rewards[index].item(), "Saved reward not audit-bound")
        require(bool(row["terminal"]) == dones[index].item(), "Saved done not audit-bound")
        require(torch.equal(torch.tensor(row["raw_policy_action_full12"]), rollout["actions"][index, 0]),
                "Saved raw action not audit-bound")
    terminal_audit = audit_rows[45]["applied_audit"]
    require(terminal_audit["terminal_bootstrap_allowed"] is False
            and terminal_audit["time_outs"] is False, "Terminal bootstrap unexpectedly enabled")

    def model(kind: str, output_dim: int) -> MLPModel:
        model_config = copy.deepcopy(config[kind])
        require(model_config.pop("class_name") == "MLPModel", "Unexpected model class")
        result = MLPModel(observations, config["obs_groups"], kind, output_dim, **model_config)
        result.load_state_dict(checkpoint[kind + "_state_dict"], strict=True)
        return result

    critic = model("critic", 1)
    actor = model("actor", 12)
    actor.requires_grad_(False)
    actor_digest = tensor_digest(actor.state_dict())
    critic.eval()
    with torch.no_grad():
        reproduced = critic(observations)
    max_value_difference = (reproduced - old_values).abs().max().item()
    # Source values were computed on CUDA; retain them unchanged and disclose CPU roundoff.
    require(torch.allclose(reproduced, old_values, rtol=2e-5, atol=2e-5),
            f"Old-value reproduction failed before fitting: maximum absolute error {max_value_difference}")

    actor_names = [name for name, _ in actor.named_parameters()]
    critic_names = [name for name, _ in critic.named_parameters()]
    require(actor_names == list(checkpoint["actor_state_dict"]), "Actor state includes unexpected nonparameters")
    require(critic_names == list(checkpoint["critic_state_dict"]), "Critic state includes unexpected nonparameters")
    source_optimizer = checkpoint["optimizer_state_dict"]
    require(len(source_optimizer["param_groups"]) == 1, "Unexpected optimizer parameter groups")
    group = source_optimizer["param_groups"][0]
    source_ids = group["params"]
    require(len(source_ids) == len(actor_names) + len(critic_names), "Official actor-then-critic count mismatch")
    require(group["lr"] == metadata["optimizer_learning_rate"] == LR, "Source effective LR differs")
    critic_source_ids = source_ids[len(actor_names):]
    copied_group = copy.deepcopy(group)
    copied_group["params"] = list(range(len(critic_names)))
    copied_group["lr"] = LR
    copied_states = {}
    for index, (source_id, parameter) in enumerate(zip(critic_source_ids, critic.parameters(), strict=True)):
        state = copy.deepcopy(source_optimizer["state"][source_id])
        require(set(state) == {"step", "exp_avg", "exp_avg_sq"}, "Unexpected Adam state keys")
        require(state["exp_avg"].shape == parameter.shape == state["exp_avg_sq"].shape,
                "Source Adam parameter shape mismatch")
        require(all(value.device.type == "cpu" and bool(torch.isfinite(value).all())
                    for value in state.values()), "Invalid source Adam state")
        copied_states[index] = state
    initial_optimizer = {"state": copied_states, "param_groups": [copied_group]}
    initial_steps = [state["step"].item() for state in copied_states.values()]
    private_generator = torch.Generator(device="cpu").manual_seed(PERMUTATION_SEED)
    permutation = torch.randperm(128, generator=private_generator)
    minibatches = list(permutation.split(32))

    def measure(values: torch.Tensor) -> dict:
        error = (values - returns).square()
        clipped_values = old_values + (values - old_values).clamp(-CLIP, CLIP)
        clipped_error = (clipped_values - returns).square()
        flat = ((values - old_values).abs() > CLIP) & (clipped_error > error)
        return {"all_mse": error.mean().item(), "terminal_mse": error[dones].mean().item(),
                "terminal_prediction": values[dones].item(), "flat_count": int(flat.sum()),
                "terminal_flat": bool(flat[dones].item())}

    def fit(clipping: bool) -> dict:
        shadow = copy.deepcopy(critic)
        shadow.train()
        optimizer = torch.optim.Adam(shadow.parameters(), lr=LR)
        optimizer.load_state_dict(copy.deepcopy(initial_optimizer))
        for index, state in optimizer.state_dict()["state"].items():
            for key, value in state.items():
                require(torch.equal(value, copied_states[index][key]), "Adam state not preserved exactly")
        require(tensor_digest(shadow.state_dict()) == tensor_digest(critic.state_dict()),
                "Initial critic is not identical")
        with torch.no_grad():
            initial = measure(shadow(observations))
        steps = []
        for epoch in range(5):
            for minibatch_number, indices in enumerate(minibatches):
                optimizer.zero_grad(set_to_none=True)
                predictions = shadow(observations[indices])
                target = returns[indices]
                source_value = old_values[indices]
                ordinary_error = (predictions - target).pow(2)
                value_clipped = source_value + (predictions - source_value).clamp(-CLIP, CLIP)
                clipped_error = (value_clipped - target).pow(2)
                flat = ((predictions - source_value).abs() > CLIP) & (clipped_error > ordinary_error)
                # Exact official RSL-RL PPO value-loss formula; no surrogate or entropy terms.
                loss = torch.max(ordinary_error, clipped_error).mean() if clipping else ordinary_error.mean()
                output_gradient = torch.autograd.grad(loss, predictions, retain_graph=True)[0]
                if clipping and bool(flat.any()):
                    require(bool((output_gradient[flat] == 0).all()), "Clipped-flat derivative is not zero")
                loss.backward()
                gradient_before = torch.nn.utils.clip_grad_norm_(shadow.parameters(), 1.0)
                gradient_after = sum(parameter.grad.square().sum() for parameter in shadow.parameters()).sqrt()
                require(bool(torch.isfinite(gradient_before)) and bool(torch.isfinite(gradient_after)),
                        "Nonfinite shadow gradient")
                require(optimizer.param_groups[0]["lr"] == LR, "Shadow LR changed")
                optimizer.step()
                with torch.no_grad():
                    metrics = measure(shadow(observations))
                steps.append({"step": len(steps) + 1, "epoch": epoch + 1, "minibatch": minibatch_number + 1,
                              "terminal_in_minibatch": bool((indices == 45).any()), "lr": LR,
                              "loss_before": loss.item(), "minibatch_flat_count_before": int(flat.sum()),
                              "minibatch_size": len(indices), "gradient_norm_before": gradient_before.item(),
                              "gradient_norm_after": gradient_after.item(), **metrics})
        require(len(steps) == 20, "Wrong shadow update count")
        require(tensor_digest(actor.state_dict()) == actor_digest, "Actor was changed")
        return {"clipping": clipping, "initial": initial, "final": metrics, "steps": steps,
                "final_adam_steps": [state["step"].item() for state in optimizer.state_dict()["state"].values()]}

    result = {
        "schema": "wlr50_clean.critic_fixed_batch_cpu_experiment.v1",
        "checkpoint": str(CHECKPOINT), "rollout": str(ROLLOUT), "global_range": [69761, 69888],
        "source_global": 69760, "source_ppo_updates": 510, "source_optimizer_steps": 10200,
        "terminal_global": 69806, "terminal_index": 45,
        "terminal_phase": terminal_audit["end_phase_id"],
        "terminal_reason": terminal_audit["termination_reason"],
        "terminal_outcome": terminal_audit["task_outcome_label"],
        "terminal_old_value": old_values[dones].item(), "terminal_return": returns[dones].item(),
        "runtime_and_policy_equal": True, "all_128_audit_rows_bound": True,
        "max_cpu_old_value_difference": max_value_difference,
        "old_value_reproduction_rtol": 2e-5, "old_value_reproduction_atol": 2e-5,
        "normalizer": type(critic.obs_normalizer).__name__, "actor_not_updated": True,
        "critic_optimizer_source_ids": critic_source_ids, "critic_parameter_names": critic_names,
        "initial_adam_steps": initial_steps, "permutation_seed": PERMUTATION_SEED,
        "permutation": permutation.tolist(), "minibatches_reused_each_epoch": True,
        "cpu_threads": torch.get_num_threads(), "cpu_interop_threads": torch.get_num_interop_threads(),
        "value_loss_coef": 1, "gradient_clip_norm": 1, "learning_rate": LR,
        "clipping_on": fit(True), "clipping_off": fit(False),
        "real_policy_decisions_added": 0, "real_optimizer_steps_added": 0,
        "weights_saved": False,
    }
    return result


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, allow_nan=False))
