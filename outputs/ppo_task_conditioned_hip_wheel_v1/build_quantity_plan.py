"""Bind the reviewed quantity-only continuation after its real runtime commit.

No model loading, simulation, counter rewriting, or migration execution.
The immutable source checkpoint and old manifest remain untouched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve(strict=True)
    output = args.output.resolve()
    if (not checkpoint.is_relative_to((OUT / "checkpoints/history").resolve())
            or not output.is_relative_to(OUT) or output == OUT):
        raise ValueError("use the actual task checkpoint and a new isolated output file")
    from wlr50_clean.ppo.semantic_cli import runtime_contract
    from wlr50_clean.ppo.semantic_migration import build_migration_plan, validate_migration_plan
    from wlr50_clean.ppo.semantic_training import write_json
    contract = runtime_contract(expected_head=args.expected_head, semantic_version="v3",
                                experiment_id="task_conditioned_hip_wheel_v1")
    code = [f"src/wlr50_clean/ppo/{name}.py" for name in
            ("semantic_cli", "semantic_training", "semantic_migration")]
    changed = sorted(code + ["configs/ppo_task_conditioned_hip_wheel_v1/execution_profile.yaml"])
    reason = ("Continue user-authorized real PPO beyond the historical full_episode quantity ceiling; "
              "preserve the 210000 entropy horizon, all control/MDP semantics, weights, full Adam, "
              "Identity normalization, training RNG and original lifetime/branch counters.")
    plan = build_migration_plan(checkpoint, contract, allowed_changed_files=changed,
        reason=reason, training_quantity_budget_review={"reason": reason,
            "reviewed_code_sha256": {path: contract["files"][path] for path in sorted(code)}})
    write_json(output, plan)
    verified = validate_migration_plan(checkpoint, contract, output)
    factor = verified["training_quantity_budget_factor"]
    print(json.dumps({"plan": str(output), "sha256": verified["plan_sha256"],
        "source_checkpoint": str(checkpoint), "stage_counters": factor["target_stage_requested_decisions"],
        "new_full_episode_ceiling": factor["target_training_budgets"]["full_episode"],
        "added_decisions": 0, "added_updates": 0}, indent=2))


if __name__ == "__main__":
    main()
