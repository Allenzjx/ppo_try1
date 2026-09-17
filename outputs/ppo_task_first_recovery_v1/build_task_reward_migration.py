"""Create the explicit same-N task-first reward-only continuation receipt."""
import argparse
import json
from pathlib import Path

from wlr50_clean.ppo.semantic_cli import runtime_contract
from wlr50_clean.ppo.semantic_migration import PROJECT_ROOT, build_migration_plan, validate_migration_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve(strict=True)
    metadata = json.loads(checkpoint.with_name(checkpoint.stem + "_manifest.json").read_text())
    target = runtime_contract(expected_head=args.expected_head, semantic_version="v3",
        experiment_id="task_first_recovery_v1")
    before, after = metadata["runtime_contract"]["files"], target["files"]
    delta = sorted(path for path in before.keys() | after.keys() if before.get(path) != after.get(path))
    plan = build_migration_plan(checkpoint, target, allowed_changed_files=delta,
        reason="Task-first epsilon zero: preserve successful N, all12 capacity, mean/sigma/Adam/RNG; new on-policy reward data only",
        task_first_reward_review={
            "reason": "Disable four secondary quality families; keep task/potential/time/safety and exact physical controller. Namespace/video/audit plumbing reviewed separately.",
            "reviewed_code_sha256": {path: after[path] for path in delta
                if not path.startswith("configs/ppo_task_first_recovery_v1/")}},
        project_root=PROJECT_ROOT)
    output = args.output.resolve()
    if not output.is_relative_to(PROJECT_ROOT / "outputs/ppo_task_first_recovery_v1"):
        raise ValueError("migration receipt must be isolated in task-first output")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    verified = validate_migration_plan(checkpoint, target, output)
    print(json.dumps({"plan": str(output), "source_decisions": metadata["global_policy_decisions"],
        "source_updates": metadata["ppo_updates"], "target_runtime": target["runtime_content_sha256"],
        "factor": verified["task_first_reward_factor"], "changed_files": delta}, ensure_ascii=False))


if __name__ == "__main__":
    main()
