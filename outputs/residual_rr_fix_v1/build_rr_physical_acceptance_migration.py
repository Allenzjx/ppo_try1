"""Create one explicit RR task/nominal continuation receipt after target commit.

No tensor mutation, physical simulation, optimizer update or checkpoint copy.
"""
import argparse
import json
from pathlib import Path

from wlr50_clean.ppo.semantic_cli import runtime_contract
from wlr50_clean.ppo.semantic_migration import (
    PROJECT_ROOT, RR_PHYSICAL_ACCEPTANCE_FILES, build_migration_plan,
    checkpoint_metadata, validate_migration_plan,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve(strict=True)
    source = checkpoint_metadata(checkpoint)
    target = runtime_contract(expected_head=args.expected_head, semantic_version="v3",
        experiment_id="residual_rr_fix_v1")
    before, after = source["runtime_contract"]["files"], target["files"]
    delta = sorted(p for p in before.keys() | after.keys() if before.get(p) != after.get(p))
    plan = build_migration_plan(checkpoint, target, allowed_changed_files=delta,
        reason="Legal-boundary continuation from latest compatible quarter checkpoint after RR free-air qualification, current carry and source-home repair",
        rr_physical_acceptance_review={
            "reason": "Reviewed task/observation semantics and nominal source/carry/home changes, plus isolated routes and video-only camera. Same372 full12 quarter.25 rho.9; five non-stage configs, reward epsilon0, geometry v3 and physical capability unchanged. All learned tensors/Adam/effective LR/Identity/RNG/counters preserved; old rollout discarded. This is not MDP or trajectory equivalence.",
            "reviewed_code_sha256": {p: after[p] for p in delta if p in RR_PHYSICAL_ACCEPTANCE_FILES}},
        project_root=PROJECT_ROOT)
    output = args.output.resolve()
    if not output.is_relative_to(PROJECT_ROOT / "outputs/residual_rr_fix_v1"):
        raise ValueError("RR migration receipt must stay in outputs/residual_rr_fix_v1")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    verified = validate_migration_plan(checkpoint, target, output)
    print(json.dumps({"plan": str(output), "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": source["checkpoint_sha256"],
        "source_counters": verified["rr_physical_acceptance_same372_factor"]["counter_origin"],
        "target_git_commit": target["source_git_commit"],
        "target_runtime_content_sha256": target["runtime_content_sha256"],
        "factor": verified["rr_physical_acceptance_same372_factor"],
        "changed_files": delta}, ensure_ascii=False))


if __name__ == "__main__":
    main()
