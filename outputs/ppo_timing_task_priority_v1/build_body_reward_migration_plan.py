"""Build the reviewed same-N body-reward continuation plan; no model or simulator load."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from wlr50_clean.ppo.semantic_cli import runtime_contract
from wlr50_clean.ppo.semantic_migration import PROJECT_ROOT, build_migration_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--nominal-reference-manifest", type=Path, required=True)
    parser.add_argument("--timing-plan", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = (PROJECT_ROOT / args.checkpoint).resolve(strict=True)
    reference = (PROJECT_ROOT / args.nominal_reference_manifest).resolve(strict=True)
    timing_plan = None if args.timing_plan is None else (PROJECT_ROOT / args.timing_plan).resolve(strict=True)
    output = (PROJECT_ROOT / args.output).resolve()
    allowed_output = (PROJECT_ROOT / "outputs/ppo_timing_task_priority_v1").resolve()
    if not output.is_relative_to(allowed_output) or output.suffix != ".json" or output.exists():
        raise ValueError("Use a new JSON output inside outputs/ppo_timing_task_priority_v1")
    metadata = json.loads(checkpoint.with_name(checkpoint.stem + "_manifest.json").read_text(encoding="utf-8"))
    target = runtime_contract(expected_head=args.expected_head, semantic_version="v3",
                              experiment_id="fsm_reference_p09_stable_v2")
    old_files, new_files = metadata["runtime_contract"]["files"], target["files"]
    delta = sorted(path for path in set(old_files) | set(new_files)
                   if old_files.get(path) != new_files.get(path))
    plan = build_migration_plan(checkpoint, target, allowed_changed_files=delta,
        reason="Functional carry/capture-settle body allowance after a frozen same-N evaluation; preserve learned state and collect fresh rollouts",
        body_reward_review={
            "reason": "Exact two reward keys and isolated body-weight consumer; retain evaluated nominal provider, task acceptance, physical configuration, contact/smoothness/potential terms and HISTORY372",
            "nominal_reference_manifest": str(reference),
            "timing_plan": None if timing_plan is None else str(timing_plan)},
        project_root=PROJECT_ROOT)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"plan": str(output), "checkpoint": str(checkpoint),
        "source_policy_decisions": metadata["global_policy_decisions"], "allowed_changed_files": delta,
        "schema": plan["body_reward_factor"]["schema"],
        "nominal_changed_since_reference": plan["body_reward_factor"]["nominal_changed_since_reference"],
        "preserve_Adam": True, "discard_old_rollout_storage": plan["discard_old_rollout_storage"]}))


if __name__ == "__main__":
    main()
