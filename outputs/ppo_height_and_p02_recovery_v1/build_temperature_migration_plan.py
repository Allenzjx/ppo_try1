"""Build one immutable, reviewed half-temperature continuation; no Isaac/model load."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from wlr50_clean.ppo.semantic_cli import runtime_contract
from wlr50_clean.ppo.semantic_migration import (
    EXPLORATION_TEMPERATURE_FILES, PROJECT_ROOT, build_migration_plan, checkpoint_metadata,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--review-reason", required=True)
    args = parser.parse_args()
    checkpoint = (PROJECT_ROOT / args.checkpoint).resolve(strict=True)
    output = (PROJECT_ROOT / args.output).resolve()
    allowed = (PROJECT_ROOT / "outputs/ppo_height_and_p02_recovery_v1").resolve()
    if not output.is_relative_to(allowed) or output.suffix != ".json" or output.exists():
        raise ValueError("Use a new immutable JSON in this round's output directory")
    target = runtime_contract(expected_head=args.expected_head, semantic_version="v3",
        experiment_id="fsm_reference_p09_stable_v2")
    metadata = checkpoint_metadata(checkpoint)
    old = metadata["runtime_contract"]["files"]
    delta = sorted(p for p in set(old) | set(target["files"]) if old.get(p) != target["files"].get(p))
    plan = build_migration_plan(checkpoint, target, allowed_changed_files=delta,
        reason="Closer-to-conditional-mean sampling candidate; preserve all learned state and collect fresh on-policy data",
        exploration_temperature_review={"reason": args.review_reason,
            "reviewed_code_sha256": {p: target["files"][p] for p in EXPLORATION_TEMPERATURE_FILES}},
        project_root=PROJECT_ROOT)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    factor = plan["exploration_temperature_factor"]
    print(json.dumps({"plan": str(output), "source_policy_decisions": metadata["global_policy_decisions"],
        "source_ppo_updates": metadata["ppo_updates"], "target_head": target["source_git_commit"],
        "changed_files": delta, "source_policy_version": factor["source_policy_version"],
        "target_policy_version": factor["target_policy_version"],
        "temperature": factor["target_exploration_std_temperature"],
        "preserve_all_learned_state": plan["preserve_actor_critic_optimizer_normalizer_rng_and_budget"]}))


if __name__ == "__main__":
    main()
