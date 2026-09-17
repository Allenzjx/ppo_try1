"""Bind the latest actual legal-boundary checkpoint to the committed execution repair."""
import argparse
import json
from pathlib import Path

from wlr50_clean.ppo.semantic_cli import runtime_contract
from wlr50_clean.ppo.semantic_migration import (
    PROJECT_ROOT, build_migration_plan, checkpoint_metadata, validate_migration_plan,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve(strict=True)
    metadata = checkpoint_metadata(checkpoint)
    target = runtime_contract(expected_head=args.expected_head, semantic_version="v3",
        experiment_id="task_first_recovery_v1")
    before, after = metadata["runtime_contract"]["files"], target["files"]
    delta = sorted(p for p in before.keys() | after.keys() if before.get(p) != after.get(p))
    plan = build_migration_plan(checkpoint, target, allowed_changed_files=delta,
        reason="Legal-boundary continuation after fixing repeated PPO residual application in nominal geometry history",
        execution_composition_review={
            "reason": "Separate same-actual-state nominal servo history from combined final command; unchanged successful N source, six configs, epsilon0, caps and HISTORY372. Effective action execution changes, no trajectory-equivalence claim.",
            "reviewed_code_sha256": {p: after[p] for p in delta}}, project_root=PROJECT_ROOT)
    output = args.output.resolve()
    if not output.is_relative_to(PROJECT_ROOT / "outputs/ppo_task_first_recovery_v1"):
        raise ValueError("composition migration receipt must stay in task-first outputs")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    verified = validate_migration_plan(checkpoint, target, output)
    print(json.dumps({"plan": str(output), "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": metadata["checkpoint_sha256"],
        "source_decisions": metadata["global_policy_decisions"], "source_updates": metadata["ppo_updates"],
        "source_optimizer_steps": metadata["optimizer_steps"], "target_runtime": target["runtime_content_sha256"],
        "factor": verified["execution_composition_factor"], "changed_files": delta}, ensure_ascii=False))


if __name__ == "__main__":
    main()
