"""Two fixed checkpoints, 37 genuine saved observations, CPU-only inference.

No model construction, sampling, RNG restoration, optimizer, simulator or pointer
write. JSON is offline analysis, never new on-policy data or physical evidence.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wlr50_clean.ppo.semantic_history_actor import (
    cap_transition_request_history, history_conditioned_head,
    task_conditioned_effective_log_std,
)

RUN = ROOT / "runs/ppo_task_conditioned_hip_wheel_v1/train/20260921T0727365666248Z_gee5a9651591d_6c75964ec21a46e296fcc856b2ee2064"
CHANNELS = ("FL_hip", "FL_knee", "FR_hip", "FR_knee", "RL_hip", "RL_knee", "RR_hip", "RR_knee", "FL_wheel", "FR_wheel", "RL_wheel", "RR_wheel")
GROUPS = {
    "episode0_FR_approach": [2, 16, 32, 64, 96, 128, 160, 192, 200],
    "episode0_FL_capture_and_initial_AIR": [376, 379, 380, 381, 382, 383, 384, 385, 388, 389],
    "episode0_P06_later": [400, 448, 512, 560, 579, 580],
    "episode0_P09_failure_approach": [583, 584, 592, 608, 620, 624],
    "episode1_P09_failure_approach": [1289, 1290, 1298, 1315, 1327, 1331],
}
TARGET_STEP = int(sys.argv[1]) if len(sys.argv) == 2 else 189440
assert TARGET_STEP in (189440, 189952), "only the two explicitly requested fixed targets are supported"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(tensor):
    x = tensor.double()
    return {"mean": x.mean().item(), "min": x.min().item(), "max": x.max().item(),
            "mean_abs": x.abs().mean().item()}


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    rng_before = torch.get_rng_state().clone()
    selections = sorted(d for ids in GROUPS.values() for d in ids)
    assert len(selections) == len(set(selections)) == 37
    chosen = set(selections)
    rows = {}
    # Only selected rows are decoded; stop at the last requested fixed state.
    with (RUN / "residual_and_projection_audit.jsonl").open(encoding="utf-8") as stream:
        for decision, line in enumerate(stream, 1):
            if decision in chosen:
                rows[decision] = json.loads(line)
                assert rows[decision]["global_policy_decision"] == 187904 + decision
            if decision == selections[-1]:
                break
    assert set(rows) == chosen
    rollouts = {}
    inputs = []
    state_evidence = []
    for decision in selections:
        update = 1434 + (decision - 1) // 128
        index = (decision - 1) % 128
        if update not in rollouts:
            path = RUN / f"rollouts/rollout_{update:06d}.pt"
            rollouts[update] = (torch.load(path, map_location="cpu", weights_only=False), path)
        storage, path = rollouts[update]
        assert storage["observations"]["policy"].shape == (128, 1, 372)
        row, obs = rows[decision], storage["observations"]["policy"][index, 0]
        assert torch.equal(storage["actions"][index, 0], torch.tensor(row["raw_policy_action_full12"]))
        assert torch.equal(storage["distribution_params"][0][index, 0], torch.tensor(row["old_distribution_mean_full12"]))
        assert torch.equal(storage["distribution_params"][1][index, 0], torch.tensor(row["old_distribution_std_full12"]))
        assert storage["dones"][index, 0].item() == row["terminal"]
        a = row["applied_audit"]
        assert int(obs[:13].argmax()) + 1 == int(a["phase_id"][1:])
        inputs.append(obs)
        legs = a["semantic_task"]["physical_evaluator"]["current_legs"]
        state_evidence.append({
            "run_decision": decision, "global_decision": 187904 + decision,
            "group": next(name for name, values in GROUPS.items() if decision in values),
            "rollout_update": update, "rollout_index_zero_based": index,
            "input_phase": a["phase_id"], "after_action_phase": a["end_phase_id"],
            "after_action_physics_tick": a["physics_tick"], "after_action_time_s": a["sim_time_s"],
            "terminal": row["terminal"], "after_action_reward": row["reward"],
            "after_action_contacts_not_input_labels": {
                leg: {k: legs[leg][k] for k in ("air", "support", "contact_surface", "clearance_m", "front_distance_m")}
                for leg in ("FR", "FL", "RL", "RR")},
            "observation_f32_sha256": hashlib.sha256(obs.numpy().tobytes()).hexdigest(),
        })
    obs = torch.stack(inputs)
    history = torch.tensor([rows[d]["policy_request"]["history_center_full12"] for d in selections])
    cpu_history, history_evidence = cap_transition_request_history(obs)
    history_error = (cpu_history - history).abs().max().item()
    assert history_error < 3e-8
    assert not bool(((cpu_history != history) & ~history_evidence["gate_full12"]).any())
    cap = history_evidence["current_cap_full12"]
    assert torch.equal(cap, torch.tensor([rows[d]["policy_request"]["current_cap_full12"] for d in selections]))
    results, checkpoints = [], []
    for step in (187904, TARGET_STEP):
        path = OUT / f"checkpoints/history/checkpoint_step_{step:09d}.pt"
        manifest_path = path.with_name(path.stem + "_manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cp = torch.load(path, map_location="cpu", weights_only=False)
        infos, state = cp["infos"], cp["actor_state_dict"]
        assert infos["global_policy_decisions"] == step
        assert sha(path) == manifest["checkpoint_sha256"]
        cfg = infos["runner_config"]["actor"]
        assert cfg["class_name"].endswith(":SemanticTaskConditionedHipWheelHistoryMLPModel")
        assert cfg["activation"] == "elu" and cfg["hidden_dims"] == [256, 256]
        assert cfg["obs_normalization"] is False and cfg["exploration_std_temperature"] == .25
        assert set(state) == {f"mlp.{layer}.{key}" for layer in (0, 2, 4) for key in ("weight", "bias")}
        assert state["mlp.0.weight"].shape == (256, 372)
        with torch.inference_mode():
            x = F.elu(F.linear(obs, state["mlp.0.weight"], state["mlp.0.bias"]))
            x = F.elu(F.linear(x, state["mlp.2.weight"], state["mlp.2.bias"]))
            head = F.linear(x, state["mlp.4.weight"], state["mlp.4.bias"]).reshape(-1, 2, 12)
            conditioned = history_conditioned_head(head, history)
            log_std, sigma_evidence = task_conditioned_effective_log_std(head[:, 1], obs, .25)
            results.append({"network_mean": head[:, 0], "network_mean_weighted_term": .1 * head[:, 0],
                            "history_center": history, "history_weighted_term": .9 * history,
                            "conditional_mean": conditioned[:, 0], "learned_log_std": head[:, 1],
                            "learned_sigma": head[:, 1].exp(), "effective_sigma": log_std.exp(),
                            "deterministic_REQUEST": cap * conditioned[:, 0].tanh()})
        checkpoints.append({"path": str(path), "sha256": sha(path),
                            "manifest_path": str(manifest_path), "manifest_sha256": sha(manifest_path),
                            "global_policy_decisions": step, "ppo_updates": infos["ppo_updates"],
                            "optimizer_steps": infos["optimizer_steps"],
                            "actor_parameter_sha256": infos["actor_parameter_sha256"]})
    first, last = results
    first_indices = [i for i, d in enumerate(selections) if d <= 128]
    base_recorded = torch.tensor([rows[selections[i]]["policy_request"]["base_mean_full12"] for i in first_indices])
    mean_recorded = torch.tensor([rows[selections[i]]["old_distribution_mean_full12"] for i in first_indices])
    sigma_recorded = torch.tensor([rows[selections[i]]["old_distribution_std_full12"] for i in first_indices])
    check = {"history_from_saved_observation_vs_original_GPU_log_max_error": history_error,
             "source_CP_first_rollout_network_max_error": (first["network_mean"][first_indices] - base_recorded).abs().max().item(),
             "source_CP_first_rollout_conditional_mean_max_error": (first["conditional_mean"][first_indices] - mean_recorded).abs().max().item(),
             "source_CP_first_rollout_sigma_max_error": (first["effective_sigma"][first_indices] - sigma_recorded).abs().max().item()}
    assert max(check.values()) < 1e-5
    mu_delta = last["conditional_mean"].double() - first["conditional_mean"].double()
    sd0, sd1 = first["effective_sigma"].double(), last["effective_sigma"].double()
    kl_mean = .5 * (mu_delta / sd1).square()
    kl_sigma = (sd1/sd0).log() + .5*(sd0/sd1).square() - .5
    comparisons = {}
    for name, decisions in GROUPS.items():
        ix = [selections.index(d) for d in decisions]
        comparisons[name] = {"sample_count": len(ix), "channels": {},
                             "Gaussian_KL_old_to_new_mean_component": kl_mean[ix].sum(-1).mean().item(),
                             "Gaussian_KL_old_to_new_sigma_component": kl_sigma[ix].sum(-1).mean().item()}
        for j, channel in enumerate(CHANNELS):
            row = {}
            for key in ("network_mean", "history_center", "history_weighted_term", "conditional_mean", "deterministic_REQUEST", "effective_sigma"):
                row[key] = {"source": summary(first[key][ix, j]), "target": summary(last[key][ix, j]),
                            "delta": summary(last[key][ix, j] - first[key][ix, j])}
            row["effective_sigma_ratio"] = summary(sd1[ix, j]/sd0[ix, j])
            row["conditional_mean_delta_in_source_sigma"] = summary(mu_delta[ix, j]/sd0[ix, j])
            comparisons[name]["channels"][channel] = row
    for i, evidence in enumerate(state_evidence):
        evidence["real_observation_f32_372"] = obs[i].tolist()
        evidence["current_cap_full12"] = cap[i].tolist()
        evidence["task_sigma_state_weights"] = {key: value[i].item() for key, value in sigma_evidence["task_state_weights"].items()}
        evidence["source_CP_response"] = {key: value[i].tolist() for key, value in first.items()}
        evidence["target_CP_response"] = {key: value[i].tolist() for key, value in last.items()}
    assert torch.equal(rng_before, torch.get_rng_state())
    output = {"schema": f"offline_same_observation_CP187904_CP{TARGET_STEP}.v1", "run": str(RUN),
              "read_only_no_simulation_no_optimizer_no_sampling_no_cuda": True,
              "exact_CP_count": 2, "real_state_count": len(selections), "checkpoints": checkpoints,
              "checks": check, "CPU_RNG_unchanged": True, "channels": CHANNELS,
              "units": "network/history/conditional mean and sigma: raw latent; REQUEST hips/knees deg, wheels rad/s",
              "scope": "same-state response only; REQUEST is cap*tanh(mean) before downstream filter/rate/headroom, not actuator target or expected stochastic action",
              "history_source": "unaltered original collection history_center, independently matched to exact saved 372 input and cap-transition kernel",
              "KL_semantics": "closed-form Gaussian old||new decomposition; not measured PPO KL or causal gradient attribution",
              "rollout_sources": [{"path": str(path), "sha256": sha(path)} for _, path in rollouts.values()],
              "window_summary": comparisons, "selected_states": state_evidence}
    target = OUT / f"CP187904_CP{TARGET_STEP}_same_state_response.json"
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    focus = ("FL_hip", "FL_knee", "RL_hip", "RR_hip", "RR_knee", "FL_wheel", "FR_wheel", "RL_wheel", "RR_wheel")
    compact = {"output": str(target), "sha256": sha(target), "checks": check, "windows": {}}
    for name, record in comparisons.items():
        compact["windows"][name] = {"n": record["sample_count"], "KL_mean": record["Gaussian_KL_old_to_new_mean_component"],
                                   "KL_sigma": record["Gaussian_KL_old_to_new_sigma_component"], "channels": {}}
        for channel in focus:
            s = record["channels"][channel]
            compact["windows"][name]["channels"][channel] = {
                "base": [s["network_mean"][side]["mean"] for side in ("source", "target")],
                "history_term": s["history_weighted_term"]["source"]["mean"],
                "REQUEST": [s["deterministic_REQUEST"][side]["mean"] for side in ("source", "target")],
                "sigma_ratio": s["effective_sigma_ratio"]["mean"]}
    print(json.dumps(compact, allow_nan=False))


if __name__ == "__main__":
    main()
