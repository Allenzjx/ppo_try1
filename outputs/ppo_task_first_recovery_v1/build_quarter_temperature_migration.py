"""Explicit single-factor P06 sampling continuation; no simulation or optimizer credit."""
import argparse
import json
from pathlib import Path

from wlr50_clean.ppo.semantic_cli import runtime_contract
from wlr50_clean.ppo.semantic_migration import (
    EXPLORATION_TEMPERATURE_FILES, PROJECT_ROOT, build_migration_plan,
    checkpoint_metadata, validate_migration_plan,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve(strict=True)
    source = checkpoint_metadata(checkpoint)
    target = runtime_contract(expected_head=args.expected_head, semantic_version="v3",
        experiment_id="task_first_recovery_v1")
    before, after = source["runtime_contract"]["files"], target["files"]
    delta = sorted(p for p in before.keys() | after.keys() if before.get(p) != after.get(p))
    plan = build_migration_plan(checkpoint, target, allowed_changed_files=delta,
        reason="Independent P06 quarter-temperature candidate after preserving full-task CP177152",
        exploration_temperature_review={
            "reason": "Saved P06 on-policy samples show large state-dependent innovation and HISTORY carry with tanh saturation before collision. Test only innovation temperature 0.5 to 0.25; preserve learned mean/sigma weights, rho, all12 capacity, N, epsilon0, Adam and physical acceptance. This is a sampling hypothesis, not a causal proof or deterministic task improvement.",
            "reviewed_code_sha256": {p: after[p] for p in EXPLORATION_TEMPERATURE_FILES}},
        project_root=PROJECT_ROOT)
    factor = plan["exploration_temperature_factor"]
    assert (factor["source_exploration_std_temperature"], factor["target_exploration_std_temperature"]) == (.5, .25)
    output = args.output.resolve()
    if not output.is_relative_to(PROJECT_ROOT / "outputs/ppo_task_first_recovery_v1"):
        raise ValueError("quarter-temperature migration receipt must stay in task-first outputs")
    with output.open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    verified = validate_migration_plan(checkpoint, target, output)
    print(json.dumps({"plan": str(output), "source_checkpoint": str(checkpoint),
        "source_counts": {k: source[k] for k in ("global_policy_decisions", "ppo_updates", "optimizer_steps")},
        "source_checkpoint_sha256": source["checkpoint_sha256"],
        "target_runtime": target["runtime_content_sha256"], "changed_files": delta,
        "source_policy": factor["source_policy_version"], "target_policy": factor["target_policy_version"],
        "verified": verified["exploration_temperature_factor"] == factor}))


if __name__ == "__main__":
    main()
