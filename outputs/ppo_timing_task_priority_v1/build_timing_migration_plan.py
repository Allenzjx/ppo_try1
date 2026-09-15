"""Build an immutable reviewed timing plan only; no model or simulator load."""
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = (PROJECT_ROOT / args.checkpoint).resolve(strict=True)
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
    # The production builder validates the immutable checkpoint/sidecar, exact
    # one-key spec delta, class-only scope and all protected/runtime bindings.
    plan = build_migration_plan(checkpoint, target, allowed_changed_files=delta,
        reason="Nominal source partial order with physical readiness; preserve compatible learned state, collect fresh rollouts",
        timing_review={"reason": "Timing-only first candidate: one nominal sequence opt-in and NominalMotionProvider class; reward, task acceptance, physical parameters and HISTORY372 are unchanged"},
        project_root=PROJECT_ROOT)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"plan": str(output), "checkpoint": str(checkpoint),
        "source_policy_decisions": metadata["global_policy_decisions"], "allowed_changed_files": delta,
        "schema": plan["nominal_timing_factor"]["schema"],
        "preserve_Adam": True, "discard_old_rollout_storage": plan["discard_old_rollout_storage"]}))


if __name__ == "__main__":
    main()
