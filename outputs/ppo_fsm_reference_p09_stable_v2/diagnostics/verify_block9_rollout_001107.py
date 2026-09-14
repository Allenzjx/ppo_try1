"""Bounded CPU-only inspection of an already-saved trusted local rollout."""
from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

torch.set_num_threads(1)
PROJECT = Path(__file__).resolve().parents[3]
RUN_ID = "20260910T1155110470637Z_g7db0d17f398d_94db61a60fb54319aef3eb1d71d97e00"
RUN = PROJECT / "runs/ppo_fsm_reference_p09_stable_v2/train" / RUN_ID
ROLLOUT = RUN / "rollouts/rollout_001107.pt"


def max_diff(left, right):
    return float((left.float() - right.float()).abs().max())


def main():
    saved = torch.load(ROLLOUT, map_location="cpu", weights_only=False)
    with (RUN / "residual_and_projection_audit.jsonl").open(encoding="utf-8") as stream:
        first256 = [json.loads(line) for line in itertools.islice(stream, 256)]
    assert len(first256) == 256
    rows = first256[128:256]
    assert [row["global_policy_decision"] for row in rows] == list(range(146049, 146177))
    assert tuple(saved["actions"].shape) == (128, 1, 12)
    vectors = {"actions": "raw_policy_action_full12"}
    scalars = {"actions_log_prob": "old_log_probability", "values": "old_value", "rewards": "reward", "dones": "terminal"}
    checks = {}
    for key, field in {**vectors, **scalars}.items():
        actual = saved[key]
        audit = torch.tensor([row[field] for row in rows], dtype=actual.dtype).reshape(actual.shape)
        checks[key] = {"exactly_equal": torch.equal(actual, audit), "max_abs_difference": max_diff(actual, audit)}
        assert checks[key]["exactly_equal"], key
    for actual, field in zip(saved["distribution_params"], ("old_distribution_mean_full12", "old_distribution_std_full12")):
        audit = torch.tensor([row[field] for row in rows], dtype=actual.dtype).reshape(actual.shape)
        checks[field] = {"exactly_equal": torch.equal(actual, audit), "max_abs_difference": max_diff(actual, audit)}
        assert checks[field]["exactly_equal"], field
    mean, std = saved["distribution_params"]
    assert bool((std > 0).all())
    independently_computed_logprob = torch.distributions.Normal(mean, std).log_prob(saved["actions"]).sum(dim=-1, keepdim=True)
    logprob_difference = max_diff(independently_computed_logprob, saved["actions_log_prob"])
    assert logprob_difference < 2e-5
    done_indices = (saved["dones"].reshape(-1) != 0).nonzero().reshape(-1).tolist()
    assert done_indices == [19, 127]
    assert all(row["applied_audit"]["time_outs"] is False for row in rows)
    # Continuing transitions retain ordinary value continuation; only true
    # terminals must disallow bootstrap. This is not a phase-boundary gate.
    assert all(rows[index]["applied_audit"]["terminal_bootstrap_allowed"] is False for index in done_indices)
    assert all(row["applied_audit"]["prefix_teacher_data_in_ppo_storage"] is False for row in rows)
    ordinary_boundaries = []
    for index, row in enumerate(rows):
        info = row["applied_audit"]
        if info["phase_id"] != info["end_phase_id"] and not row["terminal"]:
            ordinary_boundaries.append({"rollout_index_1based": index + 1, "global_policy_decision": row["global_policy_decision"],
                                        "from_phase": info["phase_id"], "to_phase": info["end_phase_id"], "done": bool(saved["dones"][index].item())})
            assert not bool(saved["dones"][index].item())
    reward_config_path = PROJECT / "configs/ppo_fsm_reference_p09_stable_v2/reward_config.yaml"
    reward_config = yaml.safe_load(reward_config_path.read_text(encoding="utf-8"))
    assert reward_config["return_profile"] == "v3_gamma_09985_lambda_099_v1"
    gamma, lam = reward_config["gamma"], 0.99
    assert gamma == 0.9985
    def recompute(last_value):
        returns = torch.zeros_like(saved["returns"])
        advantage = 0
        for step in reversed(range(128)):
            next_values = torch.full_like(saved["values"][step], last_value) if step == 127 else saved["values"][step + 1]
            mask = 1.0 - saved["dones"][step].float()
            delta = saved["rewards"][step] + mask * gamma * next_values - saved["values"][step]
            advantage = delta + mask * gamma * lam * advantage
            returns[step] = advantage + saved["values"][step]
        return returns
    recomputed_returns = recompute(0.0)
    assert torch.equal(recomputed_returns, recompute(1000000.0))
    return_difference = max_diff(recomputed_returns, saved["returns"])
    assert return_difference < 2e-5
    raw_advantages = saved["returns"] - saved["values"]
    normalized_advantages = (raw_advantages - raw_advantages.mean()) / (raw_advantages.std() + 1e-8)
    advantage_difference = max_diff(normalized_advantages, saved["advantages"])
    assert advantage_difference < 2e-5
    terminal_rows = []
    for index in done_indices:
        row = rows[index]
        info = row["applied_audit"]
        final_obs = row["terminal_observation"]
        observation_shapes = {key: list(torch.tensor(value).shape) for key, value in final_obs.items()}
        assert observation_shapes == {"policy": [1, 372], "critic": [1, 372]}
        assert all(bool(torch.isfinite(torch.tensor(value)).all()) for value in final_obs.values())
        terminal_rows.append({
            "rollout_index_1based": index + 1, "global_policy_decision": row["global_policy_decision"],
            "phase": info["phase_id"], "termination_reason": info["termination_reason"],
            "physics_tick": info["physics_tick"], "sim_time_s": info["sim_time_s"],
            "decision_physics_ticks": info["physics_ticks"], "done": True,
            "gae_continuation_mask": 0.0, "time_outs": info["time_outs"],
            "terminal_bootstrap_allowed": info["terminal_bootstrap_allowed"],
            "reward": float(saved["rewards"][index]), "value": float(saved["values"][index]),
            "return": float(saved["returns"][index]),
            "return_minus_reward": float(saved["returns"][index] - saved["rewards"][index]),
            "raw_advantage": float(raw_advantages[index]), "saved_normalized_advantage": float(saved["advantages"][index]),
            "final_observation_shapes": observation_shapes, "final_observation_finite": True,
            "final_observation_policy_critic_equal": final_obs["policy"] == final_obs["critic"],
        })
    tensors = [saved[key] for key in (*vectors, *scalars, "returns", "advantages")] + list(saved["distribution_params"]) + list(saved["observations"].values())
    assert all(value.device.type == "cpu" and bool(torch.isfinite(value).all()) for value in tensors)
    with (RUN / "optimizer_updates.jsonl").open(encoding="utf-8") as stream:
        updates = [json.loads(line) for line in itertools.islice(stream, 2)]
    assert updates[1]["ppo_update"] == 1107 and updates[1]["global_policy_decisions"] == 146176
    report = {
        "schema": "wlr50_clean.saved_rollout_diagnostic.v1", "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS", "read_only_cpu_inspection": True, "run_id": RUN_ID,
        "rollout": str(ROLLOUT), "rollout_bytes": ROLLOUT.stat().st_size,
        "rollout_sha256": hashlib.sha256(ROLLOUT.read_bytes()).hexdigest(),
        "source_git_commit": saved["runtime_contract"]["source_git_commit"],
        "runtime_content_sha256": saved["runtime_contract"]["runtime_content_sha256"],
        "global_decisions": [146049, 146176], "decisions_checked": 128, "audit_read_cap": 256,
        "phase_counts_by_action_owner": dict(sorted(Counter(row["applied_audit"]["phase_id"] for row in rows).items())),
        "audit_equality": checks, "independent_gaussian_logprob_max_abs_difference": logprob_difference,
        "terminal_indices_1based": [index + 1 for index in done_indices],
        "terminal_rows": terminal_rows, "ordinary_phase_boundaries": ordinary_boundaries,
        "bootstrap_allowed_audit_counts": dict(Counter(str(row["applied_audit"]["terminal_bootstrap_allowed"]) for row in rows)),
        "teacher_prefix_rows_in_ppo_storage": 0, "all_saved_tensors_finite": True,
        "gamma_per_policy_decision": gamma, "lambda": lam,
        "recomputed_return_max_abs_difference": return_difference,
        "tail_bootstrap_candidate_0_vs_1million_returns_exactly_equal": True,
        "saved_advantage_normalization": "whole-rollout (returns-values) centered and divided by sample std + 1e-8 before save",
        "normalized_advantage_max_abs_difference": advantage_difference,
        "normalized_advantage_mean": float(saved["advantages"].mean()), "normalized_advantage_sample_std": float(saved["advantages"].std()),
        "actual_optimizer_update": updates[1],
        "units": {"raw_action_mean_std": "dimensionless Gaussian latent, not projected joint degrees or wheel rad/s", "old_log_probability": "sum of 12 conditional Gaussian log-density terms (natural log)", "value_return_raw_advantage": "configured discounted reward units", "saved_advantage": "dimensionless standardized rollout advantage", "sim_time": "seconds", "physics_tick": "episode-relative 120Hz ticks"},
        "limitations": ["The saved rollout has no independently captured next/reset observation or reset-start timestamp. Terminal audit contains finite true final 372-vectors, not an extra timestamp proof.", "The zero tail mask and stored return numerically prove no bootstrap contribution; this inspection does not load critic weights to evaluate the terminal state.", "No post-256 active audit records, checkpoint, process, production code, or main report were touched. This is diagnosis, not a new training gate."],
        "inspector_correction": "Initial diagnostic assertion incorrectly required bootstrap_allowed=false for nonterminal rows too. Corrected to true-terminal-only; ordinary continuing transitions must retain value continuation. No production change.",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
